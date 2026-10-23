; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/retained_ds_pair_schedule/ds_pair_prefetch.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %sched.n5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %sched.n6 = sext i32 %sched.n5 to i64, !dbg !7
  %sched.n7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %sched.n8 = sext i32 %sched.n7 to i64, !dbg !7
  %sched.n9 = shl nsw i64 %sched.n8, 5, !dbg !7
  %sched.n10 = add nsw i64 %sched.n9, %sched.n6, !dbg !7
  %sched.n11 = lshr i64 %sched.n10, 5, !dbg !7
  %sched.n12 = and i64 %sched.n11, 576460752303423480, !dbg !7
  %sched.n13 = and i64 %sched.n6, 7, !dbg !7
  %sched.n14 = or disjoint i64 %sched.n12, %sched.n13, !dbg !7
  %sched.n15 = sub nsw i64 31, %sched.n14, !dbg !7
  %sched.n16 = lshr i64 %sched.n10, 3, !dbg !7
  %sched.n17 = icmp ugt i64 %sched.n10, 1023, !dbg !8
  %sched.n18 = add nsw i64 %sched.n14, -32, !dbg !8
  %sched.n19 = select i1 %sched.n17, i64 %sched.n18, i64 %sched.n15, !dbg !8
  %sched.n20 = sdiv i64 %sched.n19, 32, !dbg !8
  %sched.n21 = sext i1 %sched.n17 to i64, !dbg !8
  %sched.n22 = xor i64 %sched.n20, %sched.n21, !dbg !8
  %sched.n23 = add nsw i64 %sched.n22, %sched.n16, !dbg !8
  %sched.n25 = shl i64 %sched.n15, 7, !dbg !8
  %5 = shl i64 %sched.n22, 12, !dbg !8
  %sched.n27 = sub i64 %sched.n25, %5, !dbg !8
  %sched.n24 = shl i64 %sched.n23, 19, !dbg !8
  %sched.n28 = and i64 %sched.n24, 16252928, !dbg !8
  %sched.n31 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %sched.n33.idx = shl i64 %sched.n27, 8, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local336 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local337 = and i64 %address.local336, 1536, !dbg !8
  %address.local338 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local337, !dbg !8
  %address.local339 = ptrtoint ptr addrspace(3) %address.local338 to i64, !dbg !8
  %address.local340 = trunc i64 %address.local339 to i32, !dbg !8
  %sched.n34 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !8
  %address.local341 = shl nuw nsw i64 %address.local, 8, !dbg !8
  %address.local342 = and i64 %address.local341, 32512, !dbg !8
  %sched.n37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx = shl nuw nsw i64 %sched.n28, 1, !dbg !8
  %6 = add i64 %.idx, %sched.n33.idx, !dbg !8
  %7 = lshr i64 %address.local, 3, !dbg !8
  %address.local346.idx = and i64 %7, 16, !dbg !8
  %8 = or disjoint i64 %6, %address.local346.idx, !dbg !8
  %sched.n39 = or disjoint i64 %8, %address.local342, !dbg !8
  %dma.buffer.offset1597 = trunc i64 %sched.n39 to i32, !dbg !8
  %9 = insertelement <2 x i64> poison, i64 %sched.n37, i64 0, !dbg !8
  %sched.n44 = bitcast <2 x i64> %9 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1598 = shufflevector <4 x i32> %sched.n44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1599 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n34), !dbg !8
  %sched.n47 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1599, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %address.local347 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local348 = shl nuw nsw i64 %address.local347, 3, !dbg !8
  %address.local349 = and i64 %address.local348, 3584, !dbg !8
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local349, !dbg !8
  %address.local351 = ptrtoint ptr addrspace(3) %address.local350 to i64, !dbg !8
  %address.local352 = trunc i64 %address.local351 to i32, !dbg !8
  %sched.n48 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !8
  %10 = lshr i64 %address.local347, 3, !dbg !8
  %address.local358.idx = and i64 %10, 48, !dbg !8
  %11 = or disjoint i64 %6, %address.local358.idx, !dbg !8
  %sched.n53 = or disjoint i64 %11, %address.local342, !dbg !8
  %dma.buffer.offset1594 = trunc i64 %sched.n53 to i32, !dbg !8
  %dma.buffer.lds1596 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n48), !dbg !8
  %sched.n61 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1596, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %sched.n62 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %sched.n63 = sext i32 %sched.n62 to i64, !dbg !7
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local359 = lshr i32 %address.thread, 2, !dbg !10
  %address.local360 = and i32 %address.local359, 4, !dbg !10
  %address.local361 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local362 = and i32 %address.local361, 384, !dbg !10
  %address.local363 = or disjoint i32 %address.local360, %address.local362, !dbg !10
  %address.local364 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local365 = and i32 %address.local364, 120, !dbg !10
  %address.local366 = or disjoint i32 %address.local363, %address.local365, !dbg !10
  %address.local367 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local368 = and i32 %address.local367, 1024, !dbg !10
  %address.local369 = or disjoint i32 %address.local366, %address.local368, !dbg !10
  %address.local370 = zext nneg i32 %address.local369 to i64, !dbg !10
  %address.local371 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local370, !dbg !10
  %sched.n66 = load <4 x half>, ptr addrspace(3) %address.local371, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local371, i64 4096, !dbg !10
  %sched.n67 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %sched.n68 = fmul <4 x half> %sched.n66, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1393 = getelementptr i8, ptr addrspace(3) %address.local371, i64 1024, !dbg !10
  %sched.n69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1393, align 8, !dbg !10
  %sched.n70 = fmul <4 x half> %sched.n67, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1394 = getelementptr i8, ptr addrspace(3) %address.local371, i64 5120, !dbg !10
  %sched.n71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1394, align 8, !dbg !10
  %sched.n72 = fmul <4 x half> %sched.n69, splat (half 1.275630e-01), !dbg !10
  %sched.n73 = fmul <4 x half> %sched.n71, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %sched.n74 = or disjoint i64 %sched.n25, 127, !dbg !12
  %sched.n75 = add i64 %sched.n25, 128, !dbg !12
  %sched.n76 = icmp uge i64 %sched.n75, %sched.n74, !dbg !12
  %sched.n77 = sub nsw i64 32, %sched.n14, !dbg !13
  %sched.n79 = shl nuw nsw i32 %sched.n31, 2, !dbg !14
  %sched.n81 = and i32 %sched.n79, 12, !dbg !14
  %sched.n82 = and i32 %sched.n31, 1, !dbg !14
  %sched.n83 = or disjoint i32 %sched.n81, %sched.n82, !dbg !14
  %12 = trunc nuw nsw i32 %sched.n83 to i16, !dbg !14
  %sched.n85.lhs.trunc = add nuw nsw i16 %12, 1, !dbg !14
  %sched.n855354 = lshr i16 %sched.n85.lhs.trunc, 1, !dbg !14
  %sched.n85.sext = zext nneg i16 %sched.n855354 to i64, !dbg !14
  %sched.n88 = shl i64 %sched.n16, 7, !dbg !15
  %sched.n89 = and i64 %sched.n88, 3968, !dbg !15
  %sched.n90 = shl nsw i64 %sched.n63, 12, !dbg !15
  %sched.n91 = or disjoint i64 %sched.n89, %sched.n90, !dbg !15
  %sched.n10355 = icmp ult i64 %sched.n10, 1024, !dbg !11
  br i1 %sched.n10355, label %sched.n104.lr.ph, label %.sched.n1581_crit_edge, !dbg !11

.sched.n1581_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre69 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre70 = add nuw nsw i32 %.pre69, 64, !dbg !11
  %.pre71 = xor i32 %.pre69, 16, !dbg !11
  %.pre72 = and i32 %.pre70, 192, !dbg !11
  %.pre73 = xor i32 %.pre69, 32, !dbg !11
  br label %sched.n1581, !dbg !11

sched.n104.lr.ph:                                 ; preds = %4
  %sched.n32 = zext nneg i32 %sched.n31 to i64, !dbg !8
  %sched.n121 = ptrtoint ptr addrspace(1) %1 to i64
  %13 = insertelement <2 x i64> poison, i64 %sched.n121, i64 0
  %sched.n128 = bitcast <2 x i64> %13 to <4 x i32>
  %dma.buffer.rsrc1592 = shufflevector <4 x i32> %sched.n128, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %sched.n608 = ptrtoint ptr addrspace(1) %2 to i64
  %14 = insertelement <2 x i64> poison, i64 %sched.n608, i64 0
  %sched.n615 = bitcast <2 x i64> %14 to <4 x i32>
  %dma.buffer.rsrc1550 = shufflevector <4 x i32> %sched.n615, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %15 = lshr i64 %sched.n32, 2
  %sched.n703 = and i64 %15, 48
  %sched.n705 = and i32 %sched.n31, 240
  %index.compact = zext nneg i32 %sched.n705 to i64
  %sched.n702 = sub nsw i64 %sched.n32, %index.compact
  %sched.n704 = add nsw i64 %sched.n702, %sched.n703
  %sched.n706 = add nsw i64 %sched.n704, %sched.n25
  %sched.n707 = and i64 %sched.n32, 48
  %sched.n709 = add i64 %sched.n706, 1
  %sched.n710 = lshr i64 %sched.n709, 32
  %compare.high = trunc nuw i64 %sched.n710 to i32
  %compare.low.base = trunc i64 %sched.n709 to i32
  %sched.n744 = add i64 %sched.n706, 65
  %sched.n745 = lshr i64 %sched.n744, 32
  %compare.high1261 = trunc nuw i64 %sched.n745 to i32
  %compare.low.base1265 = trunc i64 %sched.n744 to i32
  %16 = insertelement <32 x i32> poison, i32 %compare.low.base1265, i64 0, !dbg !16
  %17 = shufflevector <32 x i32> %16, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  %18 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %19 = shufflevector <32 x i32> %18, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  br label %sched.n104, !dbg !11

sched.n104:                                       ; preds = %sched.n104.lr.ph, %sched.n847
  %sched.n10261 = phi [2 x <8 x float>] [ zeroinitializer, %sched.n104.lr.ph ], [ %sched.n1579, %sched.n847 ]
  %sched.n10160 = phi [2 x <8 x float>] [ zeroinitializer, %sched.n104.lr.ph ], [ %sched.n1575, %sched.n847 ]
  %sched.n10059 = phi [2 x <8 x float>] [ zeroinitializer, %sched.n104.lr.ph ], [ %sched.n1571, %sched.n847 ]
  %sched.n9958 = phi [2 x <1 x float>] [ zeroinitializer, %sched.n104.lr.ph ], [ %sched.n1189, %sched.n847 ]
  %sched.n9857 = phi [2 x <8 x float>] [ zeroinitializer, %sched.n104.lr.ph ], [ %sched.n1567, %sched.n847 ]
  %sched.n9756 = phi i64 [ 0, %sched.n104.lr.ph ], [ %sched.n1580, %sched.n847 ]
  %sched.n105 = shl nuw nsw i64 %sched.n9756, 7, !dbg !11
  %sched.n106 = lshr i64 %sched.n9756, 12, !dbg !15
  %sched.n107 = add nuw nsw i64 %sched.n106, %sched.n16, !dbg !15
  %sched.n108 = lshr i64 %sched.n9756, 5, !dbg !15
  %sched.n109 = and i64 %sched.n108, 127, !dbg !15
  %sched.n110 = shl i64 %sched.n107, 19, !dbg !15
  %sched.n111 = and i64 %sched.n110, 16252928, !dbg !15
  %sched.n112 = shl i64 %sched.n108, 19, !dbg !15
  %sched.n113 = shl nuw nsw i64 %sched.n9756, 14, !dbg !15
  %sched.n114 = sub i64 %sched.n113, %sched.n112, !dbg !15
  %address.thread378 = tail call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #8, !dbg !15, !range !9, !frisk.register_lifetimes !5
  %address.local379 = zext nneg i32 %address.thread378 to i64, !dbg !8
  %address.local380 = shl nuw nsw i64 %address.local379, 3, !dbg !8
  %address.local381 = and i64 %address.local380, 1536, !dbg !8
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !8
  %address.local385 = lshr i64 %address.local379, 4, !dbg !15
  %address.local386 = and i64 %address.local385, 8, !dbg !15
  %address.local387 = and i64 %address.local379, 123, !dbg !15
  %address.local388 = lshr i64 %address.local379, 2, !dbg !15
  %address.local389 = and i64 %address.local388, 4, !dbg !15
  %address.local390 = add nuw nsw i64 %address.local389, %address.local379, !dbg !15
  %address.local391 = and i64 %address.local390, 4, !dbg !15
  %address.local392 = or disjoint i64 %address.local391, %address.local387, !dbg !15
  %address.local393 = shl nuw nsw i64 %address.local392, 7, !dbg !15
  %address.local394 = or disjoint i64 %address.local393, %address.local386, !dbg !15
  %sched.n115.idx4 = or disjoint i64 %sched.n111, %sched.n109, !dbg !15
  %20 = add i64 %sched.n115.idx4, %sched.n114, !dbg !15
  %21 = add i64 %address.local394, %20, !dbg !15
  %.tr = trunc i64 %21 to i32, !dbg !15
  %dma.buffer.offset1591 = shl i32 %.tr, 1, !dbg !15
  %address.local396 = or disjoint i64 %address.local379, 256, !dbg !8
  %address.local397 = shl nuw nsw i64 %address.local396, 3, !dbg !8
  %address.local398 = and i64 %address.local397, 3584, !dbg !8
  %address.local399 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local398, !dbg !8
  %address.local402 = lshr i64 %address.local396, 4, !dbg !15
  %address.local403 = and i64 %address.local402, 24, !dbg !15
  %address.local411 = or disjoint i64 %address.local393, %address.local403, !dbg !15
  %22 = add i64 %address.local411, %20, !dbg !15
  %.tr6 = trunc i64 %22 to i32, !dbg !15
  %dma.buffer.offset1588 = shl i32 %.tr6, 1, !dbg !15
  %address.local417 = shl nuw nsw i64 %address.local379, 1, !dbg !14
  %address.local418 = and i64 %address.local417, 24, !dbg !14
  %address.local420 = shl i64 %address.local379, 62, !dbg !14
  %address.local421 = ashr exact i64 %address.local420, 57, !dbg !14
  %address.local422 = and i64 %address.local421, -64, !dbg !14
  %address.local425 = and i64 %address.local388, 12, !dbg !10
  %23 = and i64 %address.local379, 32, !dbg !10
  %.not7 = icmp eq i64 %23, 0, !dbg !10
  %address.local429 = select i1 %.not7, i64 0, i64 1016, !dbg !10
  %.idx8 = mul nuw nsw i64 %address.local379, 320, !dbg !14
  %24 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx8, !dbg !14
  %.idx9 = mul nsw i64 %address.local388, -1280, !dbg !14
  %25 = getelementptr i8, ptr addrspace(3) %24, i64 %.idx9, !dbg !14
  %26 = getelementptr [2 x i8], ptr addrspace(3) %25, i64 %address.local418, !dbg !14
  %27 = getelementptr [2 x i8], ptr addrspace(3) %26, i64 %address.local422, !dbg !14
  %28 = getelementptr [2 x i8], ptr addrspace(3) %27, i64 %address.local425, !dbg !14
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) %28, i64 %address.local429, !dbg !14
  %lds.const.gep1395 = getelementptr i8, ptr addrspace(3) %address.local431, i64 4096, !dbg !14
  %address.local442 = shl nuw nsw i32 %address.thread378, 1, !dbg !14
  %address.local443 = and i32 %address.local442, 24, !dbg !14
  %address.local444 = shl nuw nsw i32 %address.thread378, 8, !dbg !14
  %address.local445 = and i32 %address.local444, 768, !dbg !14
  %address.local446 = or disjoint i32 %address.local443, %address.local445, !dbg !14
  %address.local447 = shl nuw nsw i32 %address.thread378, 5, !dbg !14
  %address.local448 = and i32 %address.local447, 96, !dbg !14
  %address.local449 = or disjoint i32 %address.local446, %address.local448, !dbg !14
  %address.local450 = zext nneg i32 %address.local449 to i64, !dbg !14
  %address.local451 = lshr i32 %address.thread378, 1, !dbg !14
  %address.local452 = and i32 %address.local451, 1, !dbg !14
  %address.local453 = zext nneg i32 %address.local452 to i64, !dbg !14
  %lds.const.gep1397 = getelementptr i8, ptr addrspace(3) %address.local431, i64 128, !dbg !14
  %lds.const.gep1398 = getelementptr i8, ptr addrspace(3) %address.local431, i64 4224, !dbg !14
  %lds.const.gep1401 = getelementptr i8, ptr addrspace(3) %address.local431, i64 1024, !dbg !14
  %lds.const.gep1402 = getelementptr i8, ptr addrspace(3) %address.local431, i64 5120, !dbg !14
  %lds.const.gep1405 = getelementptr i8, ptr addrspace(3) %address.local431, i64 1152, !dbg !14
  %lds.const.gep1406 = getelementptr i8, ptr addrspace(3) %address.local431, i64 5248, !dbg !14
  %address.local769 = shl nuw nsw i64 %address.local379, 7, !dbg !8
  %address.local770 = and i64 %address.local769, 16256, !dbg !8
  %address.local773 = or disjoint i64 %address.local770, %address.local386, !dbg !8
  %address.local774.idx = shl nuw nsw i64 %address.local773, 1, !dbg !8
  %29 = or disjoint i64 %address.local774.idx, %6, !dbg !8
  %30 = trunc i64 %29 to i32, !dbg !8
  %dma.buffer.offset1585 = or disjoint i32 %30, 64, !dbg !8
  %address.local779 = or disjoint i64 %address.local403, %address.local770, !dbg !8
  %address.local780.idx = shl nuw nsw i64 %address.local779, 1, !dbg !8
  %31 = or disjoint i64 %6, %address.local780.idx, !dbg !8
  %32 = trunc i64 %31 to i32, !dbg !8
  %dma.buffer.offset1582 = or disjoint i32 %32, 64, !dbg !8
  %sched.n238 = add nuw nsw i64 %sched.n9756, 1024, !dbg !15
  %sched.n24212 = lshr i64 %sched.n238, 12, !dbg !15
  %sched.n244 = add nuw nsw i64 %sched.n24212, %sched.n16, !dbg !15
  %sched.n245 = add nsw i64 %sched.n108, %sched.n91, !dbg !15
  %sched.n255 = shl i64 %sched.n244, 19, !dbg !15
  %sched.n256 = and i64 %sched.n255, 16252928, !dbg !15
  %address.local781 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local381, !dbg !15
  %sched.n254 = add i64 %sched.n114, %sched.n108, !dbg !15
  %sched.n257.idx13 = add i64 %sched.n254, %sched.n91, !dbg !15
  %33 = add i64 %sched.n257.idx13, %sched.n256, !dbg !15
  %address.local785 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local398, !dbg !15
  %34 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local382, i64 0, !dbg !8
  %35 = insertelement <4 x ptr addrspace(3)> %34, ptr addrspace(3) %address.local399, i64 1, !dbg !8
  %36 = insertelement <4 x ptr addrspace(3)> %35, ptr addrspace(3) %address.local781, i64 2, !dbg !8
  %37 = insertelement <4 x ptr addrspace(3)> %36, ptr addrspace(3) %address.local785, i64 3, !dbg !8
  %38 = ptrtoint <4 x ptr addrspace(3)> %37 to <4 x i64>, !dbg !8
  %39 = trunc <4 x i64> %38 to <4 x i32>, !dbg !8
  %40 = extractelement <4 x i32> %39, i64 0, !dbg !15
  %sched.n118 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %40), !dbg !15
  %dma.buffer.lds1593 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n118), !dbg !15
  %sched.n131 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %41 = extractelement <4 x i32> %39, i64 1, !dbg !15
  %sched.n132 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %41), !dbg !15
  %dma.buffer.lds1590 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n132), !dbg !15
  %sched.n145 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %sched.n146 = load <4 x half>, ptr addrspace(3) %address.local431, align 8, !dbg !14
  %sched.n147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n146, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1395, align 8, !dbg !14
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %sched.n85.sext) #8, !dbg !14, !frisk.register_lifetimes !5
  %address.local454 = add nsw i64 %address.base, %address.local453, !dbg !14
  %address.local455 = shl i64 %address.local454, 6, !dbg !14
  %address.local456 = sub i64 %address.local450, %address.local455, !dbg !14
  %42 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local456, !dbg !14
  %43 = getelementptr [2 x i8], ptr addrspace(3) %42, i64 %address.local425, !dbg !14
  %44 = getelementptr [2 x i8], ptr addrspace(3) %43, i64 %address.local429, !dbg !14
  %address.local460 = getelementptr i8, ptr addrspace(3) %44, i64 64, !dbg !14
  %sched.n149 = load <4 x half>, ptr addrspace(3) %address.local460, align 8, !dbg !14
  %lds.const.gep1396 = getelementptr i8, ptr addrspace(3) %44, i64 4160, !dbg !14
  %sched.n152 = load <4 x half>, ptr addrspace(3) %lds.const.gep1396, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n148, <4 x float> %sched.n147) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n149, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n153 = load <4 x half>, ptr addrspace(3) %lds.const.gep1397, align 8, !dbg !14
  %sched.n156 = load <4 x half>, ptr addrspace(3) %lds.const.gep1398, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n152, <4 x float> %sched.n151) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n153, <4 x float> zeroinitializer) #8, !dbg !14
  %lds.const.gep1399 = getelementptr i8, ptr addrspace(3) %44, i64 192, !dbg !14
  %sched.n157 = load <4 x half>, ptr addrspace(3) %lds.const.gep1399, align 8, !dbg !14
  %lds.const.gep1400 = getelementptr i8, ptr addrspace(3) %44, i64 4288, !dbg !14
  %sched.n160 = load <4 x half>, ptr addrspace(3) %lds.const.gep1400, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n156, <4 x float> %sched.n155) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n157, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1401, align 8, !dbg !14
  %sched.n164 = load <4 x half>, ptr addrspace(3) %lds.const.gep1402, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n160, <4 x float> %sched.n159) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n161, <4 x float> zeroinitializer) #8, !dbg !14
  %lds.const.gep1403 = getelementptr i8, ptr addrspace(3) %44, i64 1088, !dbg !14
  %sched.n165 = load <4 x half>, ptr addrspace(3) %lds.const.gep1403, align 8, !dbg !14
  %lds.const.gep1404 = getelementptr i8, ptr addrspace(3) %44, i64 5184, !dbg !14
  %sched.n168 = load <4 x half>, ptr addrspace(3) %lds.const.gep1404, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n164, <4 x float> %sched.n163) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n165, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n169 = load <4 x half>, ptr addrspace(3) %lds.const.gep1405, align 8, !dbg !14
  %sched.n172 = load <4 x half>, ptr addrspace(3) %lds.const.gep1406, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n168, <4 x float> %sched.n167) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n169, <4 x float> zeroinitializer) #8, !dbg !14
  %lds.const.gep1407 = getelementptr i8, ptr addrspace(3) %44, i64 1216, !dbg !14
  %sched.n173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1407, align 8, !dbg !14
  %lds.const.gep1408 = getelementptr i8, ptr addrspace(3) %44, i64 5312, !dbg !14
  %sched.n176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1408, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n172, <4 x float> %sched.n171) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n173, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n176, <4 x float> %sched.n175) #8, !dbg !14
  %sched.n179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n146, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n148, <4 x float> %sched.n179) #8, !dbg !14
  %sched.n183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n149, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n152, <4 x float> %sched.n183) #8, !dbg !14
  %sched.n187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n153, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n156, <4 x float> %sched.n187) #8, !dbg !14
  %sched.n191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n157, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n160, <4 x float> %sched.n191) #8, !dbg !14
  %sched.n195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n161, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n164, <4 x float> %sched.n195) #8, !dbg !14
  %sched.n199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n165, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n168, <4 x float> %sched.n199) #8, !dbg !14
  %sched.n203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n169, <4 x float> zeroinitializer) #8, !dbg !14
  %sched.n206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n172, <4 x float> %sched.n203) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %sched.n208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n173, <4 x float> zeroinitializer) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %sched.n209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n176, <4 x float> %sched.n208) #8, !dbg !14
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %sched.n223 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %sched.n237 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %45 = extractelement <4 x i32> %39, i64 2, !dbg !15
  %sched.n262 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %45), !dbg !15
  %dma.buffer.lds1581 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n262), !dbg !15
  %46 = extractelement <4 x i32> %39, i64 3, !dbg !15
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8192, !dbg !14
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12288, !dbg !14
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %44, i64 8256, !dbg !14
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %44, i64 12352, !dbg !14
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8320, !dbg !14
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12416, !dbg !14
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %44, i64 8384, !dbg !14
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %44, i64 12480, !dbg !14
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9216, !dbg !14
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13312, !dbg !14
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %44, i64 9280, !dbg !14
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %44, i64 13376, !dbg !14
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9344, !dbg !14
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13440, !dbg !14
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %44, i64 9408, !dbg !14
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %44, i64 13504, !dbg !14
  %dma.buffer.offset1573 = or disjoint i32 %30, 128, !dbg !8
  %dma.buffer.offset1570 = or disjoint i32 %32, 128, !dbg !8
  %sched.n390 = add nuw nsw i64 %sched.n9756, 2048, !dbg !15
  %sched.n39419 = lshr i64 %sched.n390, 12, !dbg !15
  %sched.n396 = add nuw nsw i64 %sched.n39419, %sched.n16, !dbg !15
  %47 = insertelement <2 x i64> poison, i64 %sched.n245, i64 0, !dbg !15
  %48 = shufflevector <2 x i64> %47, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !15
  %49 = icmp slt <2 x i64> %48, <i64 -32, i64 -64>, !dbg !15
  %50 = extractelement <2 x i1> %49, i64 0, !dbg !15
  %sched.n251 = sext i1 %50 to i64, !dbg !15
  %51 = add nsw <2 x i64> %48, <i64 32, i64 64>, !dbg !15
  %52 = sub nsw <2 x i64> <i64 -33, i64 -65>, %48, !dbg !15
  %53 = select <2 x i1> %49, <2 x i64> %52, <2 x i64> %51, !dbg !15
  %54 = sdiv <2 x i64> %53, splat (i64 128), !dbg !15
  %55 = extractelement <2 x i64> %54, i64 0, !dbg !15
  %sched.n252 = xor i64 %55, %sched.n251, !dbg !15
  %56 = shl nsw i64 %sched.n252, 7, !dbg !15
  %57 = sub i64 %33, %56, !dbg !15
  %58 = add i64 %address.local394, %57, !dbg !15
  %.tr14 = trunc i64 %58 to i32, !dbg !15
  %59 = shl i32 %.tr14, 1, !dbg !15
  %dma.buffer.offset1579 = add i32 %59, 64, !dbg !15
  %sched.n275 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %sched.n276 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !15
  %60 = add i64 %address.local411, %57, !dbg !15
  %.tr16 = trunc i64 %60 to i32, !dbg !15
  %61 = shl i32 %.tr16, 1, !dbg !15
  %dma.buffer.offset1576 = add i32 %61, 64, !dbg !15
  %dma.buffer.lds1578 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n276), !dbg !15
  %sched.n289 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %sched.n290 = load <4 x half>, ptr addrspace(3) %address.local371, align 8, !dbg !10
  %sched.n291 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %sched.n292 = load <4 x half>, ptr addrspace(3) %lds.const.gep1393, align 8, !dbg !10
  %sched.n293 = fmul <4 x half> %sched.n290, splat (half 1.275630e-01), !dbg !10
  %sched.n294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1394, align 8, !dbg !10
  %sched.n295 = fmul <4 x half> %sched.n291, splat (half 1.275630e-01), !dbg !10
  %sched.n296 = fmul <4 x half> %sched.n292, splat (half 1.275630e-01), !dbg !10
  %sched.n297 = fmul <4 x half> %sched.n294, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %sched.n298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !14
  %sched.n299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n298, <4 x float> %sched.n150) #8, !dbg !14
  %sched.n300 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !14
  %sched.n301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !14
  %sched.n304 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n300, <4 x float> %sched.n299) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n301, <4 x float> %sched.n154) #8, !dbg !14
  %sched.n305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !14
  %sched.n308 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n304, <4 x float> %sched.n303) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n305, <4 x float> %sched.n158) #8, !dbg !14
  %sched.n309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !14
  %sched.n312 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n308, <4 x float> %sched.n307) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n309, <4 x float> %sched.n162) #8, !dbg !14
  %sched.n313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !14
  %sched.n316 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n312, <4 x float> %sched.n311) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n313, <4 x float> %sched.n166) #8, !dbg !14
  %sched.n317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !14
  %sched.n320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n316, <4 x float> %sched.n315) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n317, <4 x float> %sched.n170) #8, !dbg !14
  %sched.n321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !14
  %sched.n324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n320, <4 x float> %sched.n319) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n321, <4 x float> %sched.n174) #8, !dbg !14
  %sched.n325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !14
  %sched.n328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n324, <4 x float> %sched.n323) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n325, <4 x float> %sched.n178) #8, !dbg !14
  %sched.n330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n328, <4 x float> %sched.n327) #8, !dbg !14
  %sched.n331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n298, <4 x float> %sched.n182) #8, !dbg !14
  %sched.n334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n300, <4 x float> %sched.n331) #8, !dbg !14
  %sched.n335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n301, <4 x float> %sched.n186) #8, !dbg !14
  %sched.n338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n304, <4 x float> %sched.n335) #8, !dbg !14
  %sched.n339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n305, <4 x float> %sched.n190) #8, !dbg !14
  %sched.n342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n308, <4 x float> %sched.n339) #8, !dbg !14
  %sched.n343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n309, <4 x float> %sched.n194) #8, !dbg !14
  %sched.n346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n312, <4 x float> %sched.n343) #8, !dbg !14
  %sched.n347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n313, <4 x float> %sched.n198) #8, !dbg !14
  %sched.n350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n316, <4 x float> %sched.n347) #8, !dbg !14
  %sched.n351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n317, <4 x float> %sched.n202) #8, !dbg !14
  %sched.n354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n320, <4 x float> %sched.n351) #8, !dbg !14
  %sched.n355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n321, <4 x float> %sched.n206) #8, !dbg !14
  %sched.n358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n324, <4 x float> %sched.n355) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %sched.n360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n325, <4 x float> %sched.n209) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %sched.n361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n328, <4 x float> %sched.n360) #8, !dbg !14
  %sched.n375 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %sched.n389 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %62 = extractelement <2 x i1> %49, i64 1, !dbg !15
  %sched.n402 = sext i1 %62 to i64, !dbg !15
  %63 = extractelement <2 x i64> %54, i64 1, !dbg !15
  %sched.n403 = xor i64 %63, %sched.n402, !dbg !15
  %sched.n406 = shl i64 %sched.n396, 19, !dbg !15
  %sched.n407 = and i64 %sched.n406, 16252928, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %64 = add i64 %sched.n257.idx13, %sched.n407, !dbg !15
  %65 = shl nsw i64 %sched.n403, 7, !dbg !15
  %66 = sub i64 %64, %65, !dbg !15
  %67 = add i64 %address.local394, %66, !dbg !15
  %.tr21 = trunc i64 %67 to i32, !dbg !15
  %68 = shl i32 %.tr21, 1, !dbg !15
  %dma.buffer.offset1567 = add i32 %68, 128, !dbg !15
  %sched.n426 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %69 = add i64 %address.local411, %66, !dbg !15
  %.tr23 = trunc i64 %69 to i32, !dbg !15
  %70 = shl i32 %.tr23, 1, !dbg !15
  %dma.buffer.offset1564 = add i32 %70, 128, !dbg !15
  %sched.n440 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %sched.n441 = load <4 x half>, ptr addrspace(3) %address.local371, align 8, !dbg !10
  %sched.n442 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %sched.n443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1393, align 8, !dbg !10
  %sched.n444 = fmul <4 x half> %sched.n441, splat (half 1.275630e-01), !dbg !10
  %sched.n445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1394, align 8, !dbg !10
  %sched.n446 = fmul <4 x half> %sched.n442, splat (half 1.275630e-01), !dbg !10
  %sched.n447 = fmul <4 x half> %sched.n443, splat (half 1.275630e-01), !dbg !10
  %sched.n448 = fmul <4 x half> %sched.n445, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %sched.n449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !14
  %sched.n450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n449, <4 x float> %sched.n302) #8, !dbg !14
  %sched.n451 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !14
  %sched.n452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !14
  %sched.n455 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n451, <4 x float> %sched.n450) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n452, <4 x float> %sched.n306) #8, !dbg !14
  %sched.n456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !14
  %sched.n459 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n455, <4 x float> %sched.n454) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n458 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n456, <4 x float> %sched.n310) #8, !dbg !14
  %sched.n460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !14
  %sched.n463 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n461 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n459, <4 x float> %sched.n458) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n460, <4 x float> %sched.n314) #8, !dbg !14
  %sched.n464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !14
  %sched.n467 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n465 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n463, <4 x float> %sched.n462) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n466 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n464, <4 x float> %sched.n318) #8, !dbg !14
  %sched.n468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !14
  %sched.n471 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n469 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n467, <4 x float> %sched.n466) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n468, <4 x float> %sched.n322) #8, !dbg !14
  %sched.n472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !14
  %sched.n475 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n473 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n471, <4 x float> %sched.n470) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n474 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n472, <4 x float> %sched.n326) #8, !dbg !14
  %sched.n476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !14
  %sched.n479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n477 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n475, <4 x float> %sched.n474) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n476, <4 x float> %sched.n330) #8, !dbg !14
  %sched.n481 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n479, <4 x float> %sched.n478) #8, !dbg !14
  %sched.n482 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n449, <4 x float> %sched.n334) #8, !dbg !14
  %sched.n485 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n451, <4 x float> %sched.n482) #8, !dbg !14
  %sched.n486 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n452, <4 x float> %sched.n338) #8, !dbg !14
  %sched.n489 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n455, <4 x float> %sched.n486) #8, !dbg !14
  %sched.n490 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n456, <4 x float> %sched.n342) #8, !dbg !14
  %sched.n493 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n459, <4 x float> %sched.n490) #8, !dbg !14
  %sched.n494 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n460, <4 x float> %sched.n346) #8, !dbg !14
  %sched.n497 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n463, <4 x float> %sched.n494) #8, !dbg !14
  %sched.n498 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n464, <4 x float> %sched.n350) #8, !dbg !14
  %sched.n501 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n467, <4 x float> %sched.n498) #8, !dbg !14
  %sched.n502 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n468, <4 x float> %sched.n354) #8, !dbg !14
  %sched.n505 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n471, <4 x float> %sched.n502) #8, !dbg !14
  %sched.n506 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n472, <4 x float> %sched.n358) #8, !dbg !14
  %sched.n509 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n475, <4 x float> %sched.n506) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %sched.n511 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n476, <4 x float> %sched.n361) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %sched.n512 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n479, <4 x float> %sched.n511) #8, !dbg !14
  %dma.buffer.offset1561 = or disjoint i32 %30, 192, !dbg !8
  %sched.n526 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %dma.buffer.offset1558 = or disjoint i32 %32, 192, !dbg !8
  %sched.n540 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %sched.n541 = add nuw nsw i64 %sched.n9756, 3072, !dbg !15
  %sched.n54526 = lshr i64 %sched.n541, 12, !dbg !15
  %sched.n547 = add nuw nsw i64 %sched.n54526, %sched.n16, !dbg !15
  %sched.n548 = add nsw i64 %sched.n245, 96, !dbg !15
  %sched.n549 = icmp slt i64 %sched.n245, -96, !dbg !15
  %sched.n550 = sub nsw i64 -97, %sched.n245, !dbg !15
  %sched.n551 = select i1 %sched.n549, i64 %sched.n550, i64 %sched.n548, !dbg !15
  %sched.n552 = sdiv i64 %sched.n551, 128, !dbg !15
  %sched.n553 = sext i1 %sched.n549 to i64, !dbg !15
  %sched.n554 = xor i64 %sched.n552, %sched.n553, !dbg !15
  %sched.n557 = shl i64 %sched.n547, 19, !dbg !15
  %sched.n558 = and i64 %sched.n557, 16252928, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %71 = add i64 %sched.n257.idx13, %sched.n558, !dbg !15
  %72 = shl nsw i64 %sched.n554, 7, !dbg !15
  %73 = sub i64 %71, %72, !dbg !15
  %74 = add i64 %address.local394, %73, !dbg !15
  %.tr28 = trunc i64 %74 to i32, !dbg !15
  %75 = shl i32 %.tr28, 1, !dbg !15
  %dma.buffer.offset1555 = add i32 %75, 192, !dbg !15
  %sched.n577 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %76 = add i64 %address.local411, %73, !dbg !15
  %.tr30 = trunc i64 %76 to i32, !dbg !15
  %77 = shl i32 %.tr30, 1, !dbg !15
  %dma.buffer.offset1552 = add i32 %77, 192, !dbg !15
  %sched.n591 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %sched.n592 = load <4 x half>, ptr addrspace(3) %address.local371, align 8, !dbg !10
  %sched.n593 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %sched.n594 = load <4 x half>, ptr addrspace(3) %lds.const.gep1393, align 8, !dbg !10
  %sched.n595 = fmul <4 x half> %sched.n592, splat (half 1.275630e-01), !dbg !10
  %sched.n596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1394, align 8, !dbg !10
  %sched.n597 = fmul <4 x half> %sched.n593, splat (half 1.275630e-01), !dbg !10
  %sched.n598 = fmul <4 x half> %sched.n594, splat (half 1.275630e-01), !dbg !10
  %sched.n599 = fmul <4 x half> %sched.n596, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %sched.n600 = add nuw nsw i64 %sched.n108, %sched.n16, !dbg !17
  %sched.n601 = shl i64 %sched.n600, 19, !dbg !17
  %sched.n602 = and i64 %sched.n601, 16252928, !dbg !17
  %address.local832 = shl nuw nsw i64 %address.local379, 6, !dbg !17
  %address.local833 = and i64 %address.local832, 16128, !dbg !17
  %address.local835 = and i64 %address.local380, 24, !dbg !17
  %sched.n603.idx32 = add i64 %sched.n602, %sched.n114, !dbg !17
  %78 = or disjoint i64 %address.local835, %sched.n603.idx32, !dbg !17
  %79 = shl i64 %78, 1, !dbg !17
  %sched.n610 = or disjoint i64 %79, %address.local833, !dbg !17
  %dma.buffer.offset1549 = trunc i64 %sched.n610 to i32, !dbg !17
  %sched.n618 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %address.local838 = shl nuw nsw i64 %address.local396, 6, !dbg !17
  %address.local839 = and i64 %address.local838, 32512, !dbg !17
  %sched.n624 = or disjoint i64 %79, %address.local839, !dbg !17
  %dma.buffer.offset1546 = trunc i64 %sched.n624 to i32, !dbg !17
  %sched.n632 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %sched.n633 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !14
  %sched.n634 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n633, <4 x float> %sched.n453) #8, !dbg !14
  %sched.n635 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !14
  %sched.n636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !14
  %sched.n639 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n635, <4 x float> %sched.n634) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n636, <4 x float> %sched.n457) #8, !dbg !14
  %sched.n640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !14
  %sched.n643 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n641 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n639, <4 x float> %sched.n638) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n642 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n640, <4 x float> %sched.n461) #8, !dbg !14
  %sched.n644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !14
  %sched.n647 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n645 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n643, <4 x float> %sched.n642) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n646 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n644, <4 x float> %sched.n465) #8, !dbg !14
  %sched.n648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !14
  %sched.n651 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n649 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n647, <4 x float> %sched.n646) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n650 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n648, <4 x float> %sched.n469) #8, !dbg !14
  %sched.n652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !14
  %sched.n655 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n653 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n651, <4 x float> %sched.n650) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n654 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n652, <4 x float> %sched.n473) #8, !dbg !14
  %sched.n656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !14
  %sched.n659 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n657 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n655, <4 x float> %sched.n654) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n658 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n656, <4 x float> %sched.n477) #8, !dbg !14
  %sched.n660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !14
  %sched.n663 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n661 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n659, <4 x float> %sched.n658) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %sched.n662 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n660, <4 x float> %sched.n481) #8, !dbg !14
  %sched.n665 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n663, <4 x float> %sched.n662) #8, !dbg !14
  %sched.n666 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n633, <4 x float> %sched.n485) #8, !dbg !14
  %sched.n669 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n635, <4 x float> %sched.n666) #8, !dbg !14
  %sched.n670 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n636, <4 x float> %sched.n489) #8, !dbg !14
  %sched.n673 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n639, <4 x float> %sched.n670) #8, !dbg !14
  %sched.n674 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n640, <4 x float> %sched.n493) #8, !dbg !14
  %sched.n677 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n643, <4 x float> %sched.n674) #8, !dbg !14
  %sched.n678 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n644, <4 x float> %sched.n497) #8, !dbg !14
  %sched.n681 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n647, <4 x float> %sched.n678) #8, !dbg !14
  %sched.n682 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n648, <4 x float> %sched.n501) #8, !dbg !14
  %sched.n685 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n651, <4 x float> %sched.n682) #8, !dbg !14
  %sched.n686 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n652, <4 x float> %sched.n505) #8, !dbg !14
  %sched.n689 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n655, <4 x float> %sched.n686) #8, !dbg !14
  %sched.n690 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n656, <4 x float> %sched.n509) #8, !dbg !14
  %sched.n693 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n659, <4 x float> %sched.n690) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %sched.n695 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n660, <4 x float> %sched.n512) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %sched.n696 = or disjoint i64 %sched.n105, 127, !dbg !12
  %sched.n697 = icmp uge i64 %sched.n25, %sched.n696, !dbg !12
  %sched.n698 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n663, <4 x float> %sched.n695) #8, !dbg !14
  %sched.n699 = and i1 %sched.n76, %sched.n697, !dbg !12
  br i1 %sched.n699, label %sched.n847, label %sched.n701, !dbg !12

sched.n701:                                       ; preds = %sched.n104
  %sched.n708 = or disjoint i64 %sched.n105, %sched.n707, !dbg !12
  %sched.n711 = lshr i64 %sched.n9756, 25, !dbg !16
  %compare.high1137 = trunc nuw i64 %sched.n711 to i32, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1137, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1137, !dbg !16
  %compare.low.base1138 = trunc i64 %sched.n708 to i32, !dbg !16
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 1) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1141 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 2) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1145 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 3) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1149 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 4) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1153 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 5) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1157 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 6) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1161 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 7) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1165 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 8) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1169 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 9) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1173 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 10) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1177 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 11) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1181 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 12) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1185 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 13) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1189 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 14) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1193 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 15) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1197 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 16) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1201 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 17) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1205 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 18) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1209 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 19) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1213 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 20) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1217 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 21) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1221 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 22) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1225 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 23) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1229 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 24) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1233 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 25) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1237 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 26) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1241 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 27) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1245 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 28) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1249 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 29) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1253 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 30) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1257 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 31) #8, !dbg !16, !frisk.register_lifetimes !5
  %80 = insertelement <32 x i32> poison, i32 %compare.low.base1138, i64 0, !dbg !16
  %81 = insertelement <32 x i32> %80, i32 %compare.local.base, i64 1, !dbg !16
  %82 = insertelement <32 x i32> %81, i32 %compare.local.base1141, i64 2, !dbg !16
  %83 = insertelement <32 x i32> %82, i32 %compare.local.base1145, i64 3, !dbg !16
  %84 = insertelement <32 x i32> %83, i32 %compare.local.base1149, i64 4, !dbg !16
  %85 = insertelement <32 x i32> %84, i32 %compare.local.base1153, i64 5, !dbg !16
  %86 = insertelement <32 x i32> %85, i32 %compare.local.base1157, i64 6, !dbg !16
  %87 = insertelement <32 x i32> %86, i32 %compare.local.base1161, i64 7, !dbg !16
  %88 = insertelement <32 x i32> %87, i32 %compare.local.base1165, i64 8, !dbg !16
  %89 = insertelement <32 x i32> %88, i32 %compare.local.base1169, i64 9, !dbg !16
  %90 = insertelement <32 x i32> %89, i32 %compare.local.base1173, i64 10, !dbg !16
  %91 = insertelement <32 x i32> %90, i32 %compare.local.base1177, i64 11, !dbg !16
  %92 = insertelement <32 x i32> %91, i32 %compare.local.base1181, i64 12, !dbg !16
  %93 = insertelement <32 x i32> %92, i32 %compare.local.base1185, i64 13, !dbg !16
  %94 = insertelement <32 x i32> %93, i32 %compare.local.base1189, i64 14, !dbg !16
  %95 = insertelement <32 x i32> %94, i32 %compare.local.base1193, i64 15, !dbg !16
  %96 = insertelement <32 x i32> %95, i32 %compare.local.base1197, i64 16, !dbg !16
  %97 = insertelement <32 x i32> %96, i32 %compare.local.base1201, i64 17, !dbg !16
  %98 = insertelement <32 x i32> %97, i32 %compare.local.base1205, i64 18, !dbg !16
  %99 = insertelement <32 x i32> %98, i32 %compare.local.base1209, i64 19, !dbg !16
  %100 = insertelement <32 x i32> %99, i32 %compare.local.base1213, i64 20, !dbg !16
  %101 = insertelement <32 x i32> %100, i32 %compare.local.base1217, i64 21, !dbg !16
  %102 = insertelement <32 x i32> %101, i32 %compare.local.base1221, i64 22, !dbg !16
  %103 = insertelement <32 x i32> %102, i32 %compare.local.base1225, i64 23, !dbg !16
  %104 = insertelement <32 x i32> %103, i32 %compare.local.base1229, i64 24, !dbg !16
  %105 = insertelement <32 x i32> %104, i32 %compare.local.base1233, i64 25, !dbg !16
  %106 = insertelement <32 x i32> %105, i32 %compare.local.base1237, i64 26, !dbg !16
  %107 = insertelement <32 x i32> %106, i32 %compare.local.base1241, i64 27, !dbg !16
  %108 = insertelement <32 x i32> %107, i32 %compare.local.base1245, i64 28, !dbg !16
  %109 = insertelement <32 x i32> %108, i32 %compare.local.base1249, i64 29, !dbg !16
  %110 = insertelement <32 x i32> %109, i32 %compare.local.base1253, i64 30, !dbg !16
  %111 = insertelement <32 x i32> %110, i32 %compare.local.base1257, i64 31, !dbg !16
  %112 = or <32 x i32> %111, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %113 = icmp ugt <32 x i32> %19, %112, !dbg !16
  %114 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %115 = shufflevector <32 x i1> %114, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %116 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %117 = shufflevector <32 x i1> %116, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %118 = select <32 x i1> %115, <32 x i1> %113, <32 x i1> %117, !dbg !16
  %119 = select <32 x i1> %118, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %compare.high.equal1263 = icmp eq i32 %compare.high1261, %compare.high1137, !dbg !16
  %compare.high.order1264 = icmp ugt i32 %compare.high1261, %compare.high1137, !dbg !16
  %compare.local.base1269 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 32) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1273 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 33) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1277 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 34) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1281 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 35) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1285 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 36) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1289 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 37) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1293 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 38) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1297 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 39) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1301 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 40) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1305 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 41) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1309 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 42) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1313 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 43) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1317 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 44) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1321 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 45) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1325 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 46) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1329 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 47) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1333 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 48) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1337 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 49) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1341 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 50) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1345 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 51) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1349 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 52) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1353 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 53) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1357 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 54) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1361 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 55) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1365 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 56) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1369 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 57) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1373 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 58) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1377 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 59) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1381 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 60) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1385 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 61) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1389 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 62) #8, !dbg !16, !frisk.register_lifetimes !5
  %120 = insertelement <32 x i32> %80, i32 %compare.local.base1269, i64 1, !dbg !16
  %121 = insertelement <32 x i32> %120, i32 %compare.local.base1273, i64 2, !dbg !16
  %122 = insertelement <32 x i32> %121, i32 %compare.local.base1277, i64 3, !dbg !16
  %123 = insertelement <32 x i32> %122, i32 %compare.local.base1281, i64 4, !dbg !16
  %124 = insertelement <32 x i32> %123, i32 %compare.local.base1285, i64 5, !dbg !16
  %125 = insertelement <32 x i32> %124, i32 %compare.local.base1289, i64 6, !dbg !16
  %126 = insertelement <32 x i32> %125, i32 %compare.local.base1293, i64 7, !dbg !16
  %127 = insertelement <32 x i32> %126, i32 %compare.local.base1297, i64 8, !dbg !16
  %128 = insertelement <32 x i32> %127, i32 %compare.local.base1301, i64 9, !dbg !16
  %129 = insertelement <32 x i32> %128, i32 %compare.local.base1305, i64 10, !dbg !16
  %130 = insertelement <32 x i32> %129, i32 %compare.local.base1309, i64 11, !dbg !16
  %131 = insertelement <32 x i32> %130, i32 %compare.local.base1313, i64 12, !dbg !16
  %132 = insertelement <32 x i32> %131, i32 %compare.local.base1317, i64 13, !dbg !16
  %133 = insertelement <32 x i32> %132, i32 %compare.local.base1321, i64 14, !dbg !16
  %134 = insertelement <32 x i32> %133, i32 %compare.local.base1325, i64 15, !dbg !16
  %135 = insertelement <32 x i32> %134, i32 %compare.local.base1329, i64 16, !dbg !16
  %136 = insertelement <32 x i32> %135, i32 %compare.local.base1333, i64 17, !dbg !16
  %137 = insertelement <32 x i32> %136, i32 %compare.local.base1337, i64 18, !dbg !16
  %138 = insertelement <32 x i32> %137, i32 %compare.local.base1341, i64 19, !dbg !16
  %139 = insertelement <32 x i32> %138, i32 %compare.local.base1345, i64 20, !dbg !16
  %140 = insertelement <32 x i32> %139, i32 %compare.local.base1349, i64 21, !dbg !16
  %141 = insertelement <32 x i32> %140, i32 %compare.local.base1353, i64 22, !dbg !16
  %142 = insertelement <32 x i32> %141, i32 %compare.local.base1357, i64 23, !dbg !16
  %143 = insertelement <32 x i32> %142, i32 %compare.local.base1361, i64 24, !dbg !16
  %144 = insertelement <32 x i32> %143, i32 %compare.local.base1365, i64 25, !dbg !16
  %145 = insertelement <32 x i32> %144, i32 %compare.local.base1369, i64 26, !dbg !16
  %146 = insertelement <32 x i32> %145, i32 %compare.local.base1373, i64 27, !dbg !16
  %147 = insertelement <32 x i32> %146, i32 %compare.local.base1377, i64 28, !dbg !16
  %148 = insertelement <32 x i32> %147, i32 %compare.local.base1381, i64 29, !dbg !16
  %149 = insertelement <32 x i32> %148, i32 %compare.local.base1385, i64 30, !dbg !16
  %150 = insertelement <32 x i32> %149, i32 %compare.local.base1389, i64 31, !dbg !16
  %151 = or <32 x i32> %150, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %152 = icmp ugt <32 x i32> %17, %151, !dbg !16
  %153 = insertelement <32 x i1> poison, i1 %compare.high.equal1263, i64 0, !dbg !16
  %154 = shufflevector <32 x i1> %153, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %155 = insertelement <32 x i1> poison, i1 %compare.high.order1264, i64 0, !dbg !16
  %156 = shufflevector <32 x i1> %155, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %157 = select <32 x i1> %154, <32 x i1> %152, <32 x i1> %156, !dbg !16
  %158 = select <32 x i1> %157, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %sched.n811 = insertvalue [2 x <32 x float>] poison, <32 x float> %119, 0, !dbg !12
  %sched.n844 = insertvalue [2 x <32 x float>] %sched.n811, <32 x float> %158, 1, !dbg !12
  br label %sched.n847, !dbg !12

sched.n847:                                       ; preds = %sched.n701, %sched.n104
  %sched.n846 = phi [2 x <32 x float>] [ %sched.n844, %sched.n701 ], [ zeroinitializer, %sched.n104 ], !dbg !12
  %sched.n848 = extractvalue [2 x <32 x float>] %sched.n846, 0, !dbg !19
  %sched.n849 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %sched.n850 = fadd <4 x float> %sched.n637, %sched.n849, !dbg !19
  %sched.n851 = extractelement <4 x float> %sched.n850, i64 0, !dbg !20
  %sched.n852 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n851), !dbg !20
  %sched.n853 = insertelement <4 x float> poison, float %sched.n852, i64 0, !dbg !21
  %sched.n854 = extractelement <4 x float> %sched.n850, i64 1, !dbg !20
  %sched.n855 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n854), !dbg !20
  %sched.n856 = insertelement <4 x float> %sched.n853, float %sched.n855, i64 1, !dbg !21
  %sched.n857 = fadd float %sched.n852, %sched.n855, !dbg !22
  %sched.n858 = extractelement <4 x float> %sched.n850, i64 2, !dbg !20
  %sched.n859 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n858), !dbg !20
  %sched.n860 = insertelement <4 x float> %sched.n856, float %sched.n859, i64 2, !dbg !21
  %sched.n861 = fadd float %sched.n859, %sched.n857, !dbg !22
  %sched.n862 = extractelement <4 x float> %sched.n850, i64 3, !dbg !20
  %sched.n863 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n862), !dbg !20
  %sched.n864 = insertelement <4 x float> %sched.n860, float %sched.n863, i64 3, !dbg !21
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n864) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n865 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !21
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n865) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n866 = fadd float %sched.n863, %sched.n861, !dbg !22
  %sched.n867 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %sched.n868 = fadd <4 x float> %sched.n641, %sched.n867, !dbg !19
  %sched.n869 = extractelement <4 x float> %sched.n868, i64 0, !dbg !20
  %sched.n870 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n869), !dbg !20
  %sched.n871 = insertelement <4 x float> poison, float %sched.n870, i64 0, !dbg !21
  %sched.n872 = fadd float %sched.n870, %sched.n866, !dbg !22
  %sched.n873 = extractelement <4 x float> %sched.n868, i64 1, !dbg !20
  %sched.n874 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n873), !dbg !20
  %sched.n875 = insertelement <4 x float> %sched.n871, float %sched.n874, i64 1, !dbg !21
  %sched.n876 = fadd float %sched.n874, %sched.n872, !dbg !22
  %sched.n877 = extractelement <4 x float> %sched.n868, i64 2, !dbg !20
  %sched.n878 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n877), !dbg !20
  %sched.n879 = insertelement <4 x float> %sched.n875, float %sched.n878, i64 2, !dbg !21
  %sched.n880 = fadd float %sched.n878, %sched.n876, !dbg !22
  %sched.n881 = extractelement <4 x float> %sched.n868, i64 3, !dbg !20
  %sched.n882 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n881), !dbg !20
  %sched.n883 = insertelement <4 x float> %sched.n879, float %sched.n882, i64 3, !dbg !21
  %math.result306 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n883) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n884 = fptrunc <4 x float> %math.result306 to <4 x half>, !dbg !21
  %math.packed307 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n884) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n885 = fadd float %sched.n882, %sched.n880, !dbg !22
  %sched.n886 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %sched.n887 = fadd <4 x float> %sched.n645, %sched.n886, !dbg !19
  %sched.n888 = extractelement <4 x float> %sched.n887, i64 0, !dbg !20
  %sched.n889 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n888), !dbg !20
  %sched.n890 = insertelement <4 x float> poison, float %sched.n889, i64 0, !dbg !21
  %sched.n891 = fadd float %sched.n889, %sched.n885, !dbg !22
  %sched.n892 = extractelement <4 x float> %sched.n887, i64 1, !dbg !20
  %sched.n893 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n892), !dbg !20
  %sched.n894 = insertelement <4 x float> %sched.n890, float %sched.n893, i64 1, !dbg !21
  %sched.n895 = fadd float %sched.n893, %sched.n891, !dbg !22
  %sched.n896 = extractelement <4 x float> %sched.n887, i64 2, !dbg !20
  %sched.n897 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n896), !dbg !20
  %sched.n898 = insertelement <4 x float> %sched.n894, float %sched.n897, i64 2, !dbg !21
  %sched.n899 = fadd float %sched.n897, %sched.n895, !dbg !22
  %sched.n900 = extractelement <4 x float> %sched.n887, i64 3, !dbg !20
  %sched.n901 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n900), !dbg !20
  %sched.n902 = insertelement <4 x float> %sched.n898, float %sched.n901, i64 3, !dbg !21
  %math.result308 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n902) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n903 = fptrunc <4 x float> %math.result308 to <4 x half>, !dbg !21
  %math.packed309 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n903) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n904 = fadd float %sched.n901, %sched.n899, !dbg !22
  %sched.n905 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %sched.n906 = fadd <4 x float> %sched.n649, %sched.n905, !dbg !19
  %sched.n907 = extractelement <4 x float> %sched.n906, i64 0, !dbg !20
  %sched.n908 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n907), !dbg !20
  %sched.n909 = insertelement <4 x float> poison, float %sched.n908, i64 0, !dbg !21
  %sched.n910 = fadd float %sched.n908, %sched.n904, !dbg !22
  %sched.n911 = extractelement <4 x float> %sched.n906, i64 1, !dbg !20
  %sched.n912 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n911), !dbg !20
  %sched.n913 = insertelement <4 x float> %sched.n909, float %sched.n912, i64 1, !dbg !21
  %sched.n914 = fadd float %sched.n912, %sched.n910, !dbg !22
  %sched.n915 = extractelement <4 x float> %sched.n906, i64 2, !dbg !20
  %sched.n916 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n915), !dbg !20
  %sched.n917 = insertelement <4 x float> %sched.n913, float %sched.n916, i64 2, !dbg !21
  %sched.n918 = fadd float %sched.n916, %sched.n914, !dbg !22
  %sched.n919 = extractelement <4 x float> %sched.n906, i64 3, !dbg !20
  %sched.n920 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n919), !dbg !20
  %sched.n921 = insertelement <4 x float> %sched.n917, float %sched.n920, i64 3, !dbg !21
  %math.result310 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n921) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n922 = fptrunc <4 x float> %math.result310 to <4 x half>, !dbg !21
  %math.packed311 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n922) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n923 = fadd float %sched.n920, %sched.n918, !dbg !22
  %sched.n924 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %sched.n925 = fadd <4 x float> %sched.n653, %sched.n924, !dbg !19
  %sched.n926 = extractelement <4 x float> %sched.n925, i64 0, !dbg !20
  %sched.n927 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n926), !dbg !20
  %sched.n928 = insertelement <4 x float> poison, float %sched.n927, i64 0, !dbg !21
  %sched.n929 = fadd float %sched.n927, %sched.n923, !dbg !22
  %sched.n930 = extractelement <4 x float> %sched.n925, i64 1, !dbg !20
  %sched.n931 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n930), !dbg !20
  %sched.n932 = insertelement <4 x float> %sched.n928, float %sched.n931, i64 1, !dbg !21
  %sched.n933 = fadd float %sched.n931, %sched.n929, !dbg !22
  %sched.n934 = extractelement <4 x float> %sched.n925, i64 2, !dbg !20
  %sched.n935 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n934), !dbg !20
  %sched.n936 = insertelement <4 x float> %sched.n932, float %sched.n935, i64 2, !dbg !21
  %sched.n937 = fadd float %sched.n935, %sched.n933, !dbg !22
  %sched.n938 = extractelement <4 x float> %sched.n925, i64 3, !dbg !20
  %sched.n939 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n938), !dbg !20
  %sched.n940 = insertelement <4 x float> %sched.n936, float %sched.n939, i64 3, !dbg !21
  %math.result312 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n940) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n941 = fptrunc <4 x float> %math.result312 to <4 x half>, !dbg !21
  %math.packed313 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n941) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n942 = fadd float %sched.n939, %sched.n937, !dbg !22
  %sched.n943 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %sched.n944 = fadd <4 x float> %sched.n657, %sched.n943, !dbg !19
  %sched.n945 = extractelement <4 x float> %sched.n944, i64 0, !dbg !20
  %sched.n946 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n945), !dbg !20
  %sched.n947 = insertelement <4 x float> poison, float %sched.n946, i64 0, !dbg !21
  %sched.n948 = fadd float %sched.n946, %sched.n942, !dbg !22
  %sched.n949 = extractelement <4 x float> %sched.n944, i64 1, !dbg !20
  %sched.n950 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n949), !dbg !20
  %sched.n951 = insertelement <4 x float> %sched.n947, float %sched.n950, i64 1, !dbg !21
  %sched.n952 = fadd float %sched.n950, %sched.n948, !dbg !22
  %sched.n953 = extractelement <4 x float> %sched.n944, i64 2, !dbg !20
  %sched.n954 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n953), !dbg !20
  %sched.n955 = insertelement <4 x float> %sched.n951, float %sched.n954, i64 2, !dbg !21
  %sched.n956 = fadd float %sched.n954, %sched.n952, !dbg !22
  %sched.n957 = extractelement <4 x float> %sched.n944, i64 3, !dbg !20
  %sched.n958 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n957), !dbg !20
  %sched.n959 = insertelement <4 x float> %sched.n955, float %sched.n958, i64 3, !dbg !21
  %math.result314 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n959) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n960 = fptrunc <4 x float> %math.result314 to <4 x half>, !dbg !21
  %math.packed315 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n960) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n961 = fadd float %sched.n958, %sched.n956, !dbg !22
  %sched.n962 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %sched.n963 = fadd <4 x float> %sched.n661, %sched.n962, !dbg !19
  %sched.n964 = extractelement <4 x float> %sched.n963, i64 0, !dbg !20
  %sched.n965 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n964), !dbg !20
  %sched.n966 = insertelement <4 x float> poison, float %sched.n965, i64 0, !dbg !21
  %sched.n967 = fadd float %sched.n965, %sched.n961, !dbg !22
  %sched.n968 = extractelement <4 x float> %sched.n963, i64 1, !dbg !20
  %sched.n969 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n968), !dbg !20
  %sched.n970 = insertelement <4 x float> %sched.n966, float %sched.n969, i64 1, !dbg !21
  %sched.n971 = fadd float %sched.n969, %sched.n967, !dbg !22
  %sched.n972 = extractelement <4 x float> %sched.n963, i64 2, !dbg !20
  %sched.n973 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n972), !dbg !20
  %sched.n974 = insertelement <4 x float> %sched.n970, float %sched.n973, i64 2, !dbg !21
  %sched.n975 = fadd float %sched.n973, %sched.n971, !dbg !22
  %sched.n976 = extractelement <4 x float> %sched.n963, i64 3, !dbg !20
  %sched.n977 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n976), !dbg !20
  %sched.n978 = insertelement <4 x float> %sched.n974, float %sched.n977, i64 3, !dbg !21
  %math.result316 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n978) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n979 = fptrunc <4 x float> %math.result316 to <4 x half>, !dbg !21
  %math.packed317 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n979) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n980 = fadd float %sched.n977, %sched.n975, !dbg !22
  %sched.n981 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %sched.n982 = fadd <4 x float> %sched.n665, %sched.n981, !dbg !19
  %sched.n983 = extractelement <4 x float> %sched.n982, i64 0, !dbg !20
  %sched.n984 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n983), !dbg !20
  %sched.n985 = insertelement <4 x float> poison, float %sched.n984, i64 0, !dbg !21
  %sched.n986 = fadd float %sched.n984, %sched.n980, !dbg !22
  %sched.n987 = extractelement <4 x float> %sched.n982, i64 1, !dbg !20
  %sched.n988 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n987), !dbg !20
  %sched.n989 = insertelement <4 x float> %sched.n985, float %sched.n988, i64 1, !dbg !21
  %sched.n990 = fadd float %sched.n988, %sched.n986, !dbg !22
  %sched.n991 = extractelement <4 x float> %sched.n982, i64 2, !dbg !20
  %sched.n992 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n991), !dbg !20
  %sched.n993 = insertelement <4 x float> %sched.n989, float %sched.n992, i64 2, !dbg !21
  %sched.n994 = fadd float %sched.n992, %sched.n990, !dbg !22
  %sched.n995 = extractelement <4 x float> %sched.n982, i64 3, !dbg !20
  %sched.n996 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n995), !dbg !20
  %sched.n997 = insertelement <4 x float> %sched.n993, float %sched.n996, i64 3, !dbg !21
  %math.result318 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n997) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n998 = fptrunc <4 x float> %math.result318 to <4 x half>, !dbg !21
  %math.packed319 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n998) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n999 = fadd float %sched.n996, %sched.n994, !dbg !22
  %sched.n1000 = extractvalue [2 x <32 x float>] %sched.n846, 1, !dbg !19
  %sched.n1001 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %sched.n1002 = fadd <4 x float> %sched.n669, %sched.n1001, !dbg !19
  %sched.n1003 = extractelement <4 x float> %sched.n1002, i64 0, !dbg !20
  %sched.n1004 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1003), !dbg !20
  %sched.n1005 = insertelement <4 x float> poison, float %sched.n1004, i64 0, !dbg !21
  %sched.n1006 = extractelement <4 x float> %sched.n1002, i64 1, !dbg !20
  %sched.n1007 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1006), !dbg !20
  %sched.n1008 = insertelement <4 x float> %sched.n1005, float %sched.n1007, i64 1, !dbg !21
  %sched.n1009 = fadd float %sched.n1004, %sched.n1007, !dbg !22
  %sched.n1010 = extractelement <4 x float> %sched.n1002, i64 2, !dbg !20
  %sched.n1011 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1010), !dbg !20
  %sched.n1012 = insertelement <4 x float> %sched.n1008, float %sched.n1011, i64 2, !dbg !21
  %sched.n1013 = fadd float %sched.n1011, %sched.n1009, !dbg !22
  %sched.n1014 = extractelement <4 x float> %sched.n1002, i64 3, !dbg !20
  %sched.n1015 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1014), !dbg !20
  %sched.n1016 = insertelement <4 x float> %sched.n1012, float %sched.n1015, i64 3, !dbg !21
  %math.result320 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1016) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n1017 = fptrunc <4 x float> %math.result320 to <4 x half>, !dbg !21
  %math.packed321 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1017) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n1018 = fadd float %sched.n1015, %sched.n1013, !dbg !22
  %sched.n1019 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %sched.n1020 = fadd <4 x float> %sched.n673, %sched.n1019, !dbg !19
  %sched.n1021 = extractelement <4 x float> %sched.n1020, i64 0, !dbg !20
  %sched.n1022 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1021), !dbg !20
  %sched.n1023 = insertelement <4 x float> poison, float %sched.n1022, i64 0, !dbg !21
  %sched.n1024 = fadd float %sched.n1022, %sched.n1018, !dbg !22
  %sched.n1025 = extractelement <4 x float> %sched.n1020, i64 1, !dbg !20
  %sched.n1026 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1025), !dbg !20
  %sched.n1027 = insertelement <4 x float> %sched.n1023, float %sched.n1026, i64 1, !dbg !21
  %sched.n1028 = fadd float %sched.n1026, %sched.n1024, !dbg !22
  %sched.n1029 = extractelement <4 x float> %sched.n1020, i64 2, !dbg !20
  %sched.n1030 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1029), !dbg !20
  %sched.n1031 = insertelement <4 x float> %sched.n1027, float %sched.n1030, i64 2, !dbg !21
  %sched.n1032 = fadd float %sched.n1030, %sched.n1028, !dbg !22
  %sched.n1033 = extractelement <4 x float> %sched.n1020, i64 3, !dbg !20
  %sched.n1034 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1033), !dbg !20
  %sched.n1035 = insertelement <4 x float> %sched.n1031, float %sched.n1034, i64 3, !dbg !21
  %math.result322 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1035) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n1036 = fptrunc <4 x float> %math.result322 to <4 x half>, !dbg !21
  %math.packed323 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1036) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n1037 = fadd float %sched.n1034, %sched.n1032, !dbg !22
  %sched.n1038 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %sched.n1039 = fadd <4 x float> %sched.n677, %sched.n1038, !dbg !19
  %sched.n1040 = extractelement <4 x float> %sched.n1039, i64 0, !dbg !20
  %sched.n1041 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1040), !dbg !20
  %sched.n1042 = insertelement <4 x float> poison, float %sched.n1041, i64 0, !dbg !21
  %sched.n1043 = fadd float %sched.n1041, %sched.n1037, !dbg !22
  %sched.n1044 = extractelement <4 x float> %sched.n1039, i64 1, !dbg !20
  %sched.n1045 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1044), !dbg !20
  %sched.n1046 = insertelement <4 x float> %sched.n1042, float %sched.n1045, i64 1, !dbg !21
  %sched.n1047 = fadd float %sched.n1045, %sched.n1043, !dbg !22
  %sched.n1048 = extractelement <4 x float> %sched.n1039, i64 2, !dbg !20
  %sched.n1049 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1048), !dbg !20
  %sched.n1050 = insertelement <4 x float> %sched.n1046, float %sched.n1049, i64 2, !dbg !21
  %sched.n1051 = fadd float %sched.n1049, %sched.n1047, !dbg !22
  %sched.n1052 = extractelement <4 x float> %sched.n1039, i64 3, !dbg !20
  %sched.n1053 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1052), !dbg !20
  %sched.n1054 = insertelement <4 x float> %sched.n1050, float %sched.n1053, i64 3, !dbg !21
  %math.result324 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1054) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n1055 = fptrunc <4 x float> %math.result324 to <4 x half>, !dbg !21
  %math.packed325 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1055) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n1056 = fadd float %sched.n1053, %sched.n1051, !dbg !22
  %sched.n1057 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %sched.n1058 = fadd <4 x float> %sched.n681, %sched.n1057, !dbg !19
  %sched.n1059 = extractelement <4 x float> %sched.n1058, i64 0, !dbg !20
  %sched.n1060 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1059), !dbg !20
  %sched.n1061 = insertelement <4 x float> poison, float %sched.n1060, i64 0, !dbg !21
  %sched.n1062 = fadd float %sched.n1060, %sched.n1056, !dbg !22
  %sched.n1063 = extractelement <4 x float> %sched.n1058, i64 1, !dbg !20
  %sched.n1064 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1063), !dbg !20
  %sched.n1065 = insertelement <4 x float> %sched.n1061, float %sched.n1064, i64 1, !dbg !21
  %sched.n1066 = fadd float %sched.n1064, %sched.n1062, !dbg !22
  %sched.n1067 = extractelement <4 x float> %sched.n1058, i64 2, !dbg !20
  %sched.n1068 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1067), !dbg !20
  %sched.n1069 = insertelement <4 x float> %sched.n1065, float %sched.n1068, i64 2, !dbg !21
  %sched.n1070 = fadd float %sched.n1068, %sched.n1066, !dbg !22
  %sched.n1071 = extractelement <4 x float> %sched.n1058, i64 3, !dbg !20
  %sched.n1072 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1071), !dbg !20
  %sched.n1073 = insertelement <4 x float> %sched.n1069, float %sched.n1072, i64 3, !dbg !21
  %math.result326 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1073) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n1074 = fptrunc <4 x float> %math.result326 to <4 x half>, !dbg !21
  %math.packed327 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1074) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n1075 = fadd float %sched.n1072, %sched.n1070, !dbg !22
  %sched.n1076 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %sched.n1077 = fadd <4 x float> %sched.n685, %sched.n1076, !dbg !19
  %sched.n1078 = extractelement <4 x float> %sched.n1077, i64 0, !dbg !20
  %sched.n1079 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1078), !dbg !20
  %sched.n1080 = insertelement <4 x float> poison, float %sched.n1079, i64 0, !dbg !21
  %sched.n1081 = fadd float %sched.n1079, %sched.n1075, !dbg !22
  %sched.n1082 = extractelement <4 x float> %sched.n1077, i64 1, !dbg !20
  %sched.n1083 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1082), !dbg !20
  %sched.n1084 = insertelement <4 x float> %sched.n1080, float %sched.n1083, i64 1, !dbg !21
  %sched.n1085 = fadd float %sched.n1083, %sched.n1081, !dbg !22
  %sched.n1086 = extractelement <4 x float> %sched.n1077, i64 2, !dbg !20
  %sched.n1087 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1086), !dbg !20
  %sched.n1088 = insertelement <4 x float> %sched.n1084, float %sched.n1087, i64 2, !dbg !21
  %sched.n1089 = fadd float %sched.n1087, %sched.n1085, !dbg !22
  %sched.n1090 = extractelement <4 x float> %sched.n1077, i64 3, !dbg !20
  %sched.n1091 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1090), !dbg !20
  %sched.n1092 = insertelement <4 x float> %sched.n1088, float %sched.n1091, i64 3, !dbg !21
  %math.result328 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1092) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n1093 = fptrunc <4 x float> %math.result328 to <4 x half>, !dbg !21
  %math.packed329 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1093) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n1094 = fadd float %sched.n1091, %sched.n1089, !dbg !22
  %sched.n1095 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %sched.n1096 = fadd <4 x float> %sched.n689, %sched.n1095, !dbg !19
  %sched.n1097 = extractelement <4 x float> %sched.n1096, i64 0, !dbg !20
  %sched.n1098 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1097), !dbg !20
  %sched.n1099 = insertelement <4 x float> poison, float %sched.n1098, i64 0, !dbg !21
  %sched.n1100 = fadd float %sched.n1098, %sched.n1094, !dbg !22
  %sched.n1101 = extractelement <4 x float> %sched.n1096, i64 1, !dbg !20
  %sched.n1102 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1101), !dbg !20
  %sched.n1103 = insertelement <4 x float> %sched.n1099, float %sched.n1102, i64 1, !dbg !21
  %sched.n1104 = fadd float %sched.n1102, %sched.n1100, !dbg !22
  %sched.n1105 = extractelement <4 x float> %sched.n1096, i64 2, !dbg !20
  %sched.n1106 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1105), !dbg !20
  %sched.n1107 = insertelement <4 x float> %sched.n1103, float %sched.n1106, i64 2, !dbg !21
  %sched.n1108 = fadd float %sched.n1106, %sched.n1104, !dbg !22
  %sched.n1109 = extractelement <4 x float> %sched.n1096, i64 3, !dbg !20
  %sched.n1110 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1109), !dbg !20
  %sched.n1111 = insertelement <4 x float> %sched.n1107, float %sched.n1110, i64 3, !dbg !21
  %math.result330 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1111) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n1112 = fptrunc <4 x float> %math.result330 to <4 x half>, !dbg !21
  %math.packed331 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1112) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n1113 = fadd float %sched.n1110, %sched.n1108, !dbg !22
  %sched.n1114 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %sched.n1115 = fadd <4 x float> %sched.n693, %sched.n1114, !dbg !19
  %sched.n1116 = extractelement <4 x float> %sched.n1115, i64 0, !dbg !20
  %sched.n1117 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1116), !dbg !20
  %sched.n1118 = insertelement <4 x float> poison, float %sched.n1117, i64 0, !dbg !21
  %sched.n1119 = fadd float %sched.n1117, %sched.n1113, !dbg !22
  %sched.n1120 = extractelement <4 x float> %sched.n1115, i64 1, !dbg !20
  %sched.n1121 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1120), !dbg !20
  %sched.n1122 = insertelement <4 x float> %sched.n1118, float %sched.n1121, i64 1, !dbg !21
  %sched.n1123 = fadd float %sched.n1121, %sched.n1119, !dbg !22
  %sched.n1124 = extractelement <4 x float> %sched.n1115, i64 2, !dbg !20
  %sched.n1125 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1124), !dbg !20
  %sched.n1126 = insertelement <4 x float> %sched.n1122, float %sched.n1125, i64 2, !dbg !21
  %sched.n1127 = fadd float %sched.n1125, %sched.n1123, !dbg !22
  %sched.n1128 = extractelement <4 x float> %sched.n1115, i64 3, !dbg !20
  %sched.n1129 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1128), !dbg !20
  %sched.n1130 = insertelement <4 x float> %sched.n1126, float %sched.n1129, i64 3, !dbg !21
  %math.result332 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1130) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n1131 = fptrunc <4 x float> %math.result332 to <4 x half>, !dbg !21
  %math.packed333 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1131) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n1132 = fadd float %sched.n1129, %sched.n1127, !dbg !22
  %sched.n1133 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %sched.n1134 = fadd <4 x float> %sched.n698, %sched.n1133, !dbg !19
  %sched.n1135 = extractelement <4 x float> %sched.n1134, i64 0, !dbg !20
  %sched.n1136 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1135), !dbg !20
  %sched.n1137 = insertelement <4 x float> poison, float %sched.n1136, i64 0, !dbg !21
  %sched.n1138 = fadd float %sched.n1136, %sched.n1132, !dbg !22
  %sched.n1139 = extractelement <4 x float> %sched.n1134, i64 1, !dbg !20
  %sched.n1140 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1139), !dbg !20
  %sched.n1141 = insertelement <4 x float> %sched.n1137, float %sched.n1140, i64 1, !dbg !21
  %sched.n1142 = fadd float %sched.n1140, %sched.n1138, !dbg !22
  %sched.n1143 = extractelement <4 x float> %sched.n1134, i64 2, !dbg !20
  %sched.n1144 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1143), !dbg !20
  %sched.n1145 = insertelement <4 x float> %sched.n1141, float %sched.n1144, i64 2, !dbg !21
  %sched.n1146 = fadd float %sched.n1144, %sched.n1142, !dbg !22
  %sched.n1147 = extractelement <4 x float> %sched.n1134, i64 3, !dbg !20
  %sched.n1148 = tail call float @llvm.amdgcn.exp2.f32(float %sched.n1147), !dbg !20
  %sched.n1149 = insertelement <4 x float> %sched.n1145, float %sched.n1148, i64 3, !dbg !21
  %math.result334 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1149) #8, !dbg !21, !frisk.register_lifetimes !5
  %sched.n1150 = fptrunc <4 x float> %math.result334 to <4 x half>, !dbg !21
  %math.packed335 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1150) #8, !dbg !22, !frisk.register_lifetimes !5
  %sched.n1151 = fadd float %sched.n1148, %sched.n1146, !dbg !22
  %sched.n1152 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !22
  %sched.n1153 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %sched.n1152), !dbg !22
  %sched.n1154 = add nuw nsw i32 %sched.n1153, 64, !dbg !22
  %sched.n1155 = xor i32 %sched.n1153, 16, !dbg !22
  %sched.n1156 = and i32 %sched.n1154, 192, !dbg !22
  %sched.n1157 = icmp samesign ult i32 %sched.n1155, %sched.n1156, !dbg !22
  %sched.n1158 = select i1 %sched.n1157, i32 %sched.n1155, i32 %sched.n1153, !dbg !22
  %sched.n1159 = shl nuw nsw i32 %sched.n1158, 2, !dbg !22
  %sched.n1160 = bitcast float %sched.n999 to i32, !dbg !22
  %sched.n1161 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1159, i32 %sched.n1160), !dbg !22
  %sched.n1162 = bitcast i32 %sched.n1161 to float, !dbg !22
  %sched.n1163 = fadd float %sched.n999, %sched.n1162, !dbg !22
  %sched.n1164 = xor i32 %sched.n1153, 32, !dbg !22
  %sched.n1165 = icmp samesign ult i32 %sched.n1164, %sched.n1156, !dbg !22
  %sched.n1166 = select i1 %sched.n1165, i32 %sched.n1164, i32 %sched.n1153, !dbg !22
  %sched.n1167 = shl nuw nsw i32 %sched.n1166, 2, !dbg !22
  %sched.n1168 = bitcast float %sched.n1163 to i32, !dbg !22
  %sched.n1169 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1167, i32 %sched.n1168), !dbg !22
  %sched.n1170 = bitcast i32 %sched.n1169 to float, !dbg !22
  %sched.n1171 = fadd float %sched.n1163, %sched.n1170, !dbg !22
  %sched.n1172 = extractvalue [2 x <1 x float>] %sched.n9958, 0, !dbg !22
  %sched.n1173 = extractelement <1 x float> %sched.n1172, i64 0, !dbg !22
  %sched.n1174 = fadd float %sched.n1173, %sched.n1171, !dbg !22
  %sched.n1175 = insertelement <1 x float> poison, float %sched.n1174, i64 0, !dbg !11
  %sched.n1176 = insertvalue [2 x <1 x float>] poison, <1 x float> %sched.n1175, 0, !dbg !11
  %sched.n1177 = bitcast float %sched.n1151 to i32, !dbg !22
  %sched.n1178 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1159, i32 %sched.n1177), !dbg !22
  %sched.n1179 = bitcast i32 %sched.n1178 to float, !dbg !22
  %sched.n1180 = fadd float %sched.n1151, %sched.n1179, !dbg !22
  %sched.n1181 = bitcast float %sched.n1180 to i32, !dbg !22
  %sched.n1182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1167, i32 %sched.n1181), !dbg !22
  %sched.n1183 = bitcast i32 %sched.n1182 to float, !dbg !22
  %sched.n1184 = fadd float %sched.n1180, %sched.n1183, !dbg !22
  %sched.n1185 = extractvalue [2 x <1 x float>] %sched.n9958, 1, !dbg !22
  %sched.n1186 = extractelement <1 x float> %sched.n1185, i64 0, !dbg !22
  %sched.n1187 = fadd float %sched.n1186, %sched.n1184, !dbg !22
  %sched.n1188 = insertelement <1 x float> poison, float %sched.n1187, i64 0, !dbg !11
  %sched.n1189 = insertvalue [2 x <1 x float>] %sched.n1176, <1 x float> %sched.n1188, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread844 = tail call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #8, !dbg !23, !range !9, !frisk.register_lifetimes !5
  %address.local845 = shl nuw nsw i32 %address.thread844, 3, !dbg !23
  %address.local846 = and i32 %address.local845, 120, !dbg !23
  %address.local847 = shl nuw nsw i32 %address.thread844, 5, !dbg !23
  %address.local848 = and i32 %address.local847, 1536, !dbg !23
  %address.local849 = or disjoint i32 %address.local846, %address.local848, !dbg !23
  %address.local850 = zext nneg i32 %address.local849 to i64, !dbg !23
  %address.local851 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local850, !dbg !23
  %sched.n1190 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #11, !dbg !23
  %sched.n1191 = bitcast <4 x i32> %sched.n1190 to <8 x half>, !dbg !23
  %address.local852 = zext nneg i32 %address.thread844 to i64, !dbg !8
  %address.local853 = and i64 %address.local852, 60, !dbg !23
  %address.local854 = add nuw nsw i64 %address.local853, %address.local852, !dbg !23
  %address.local856.idx = shl nuw nsw i64 %address.local854, 4, !dbg !23
  %address.local856 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local856.idx, !dbg !23
  %address.local857 = lshr i64 %address.local852, 4, !dbg !23
  %address.local858 = and i64 %address.local857, 3, !dbg !23
  %address.local860.idx = mul nuw nsw i64 %address.local858, 768, !dbg !23
  %address.local860 = getelementptr i8, ptr addrspace(3) %address.local856, i64 %address.local860.idx, !dbg !23
  %address.local861 = lshr i64 %address.local852, 2, !dbg !14
  %address.local863.idx = mul nsw i64 %address.local861, -64, !dbg !23
  %address.local863 = getelementptr i8, ptr addrspace(3) %address.local860, i64 %address.local863.idx, !dbg !23
  %address.local864 = getelementptr i8, ptr addrspace(3) %address.local863, i64 256, !dbg !23
  %sched.n1192 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #11, !dbg !23
  %sched.n1193 = bitcast <4 x i32> %sched.n1192 to <8 x half>, !dbg !23
  %address.local868 = getelementptr i8, ptr addrspace(3) %address.local863, i64 512, !dbg !23
  %sched.n1194 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #11, !dbg !23
  %sched.n1195 = bitcast <4 x i32> %sched.n1194 to <8 x half>, !dbg !23
  %address.local872 = getelementptr i8, ptr addrspace(3) %address.local863, i64 768, !dbg !23
  %sched.n1196 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #11, !dbg !23
  %sched.n1197 = bitcast <4 x i32> %sched.n1196 to <8 x half>, !dbg !23
  %address.local874 = getelementptr i8, ptr addrspace(3) %address.local851, i64 4096, !dbg !23
  %sched.n1198 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #11, !dbg !23
  %sched.n1199 = bitcast <4 x i32> %sched.n1198 to <8 x half>, !dbg !23
  %address.local878 = getelementptr i8, ptr addrspace(3) %address.local863, i64 4352, !dbg !23
  %sched.n1200 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #11, !dbg !23
  %sched.n1201 = bitcast <4 x i32> %sched.n1200 to <8 x half>, !dbg !23
  %address.local882 = getelementptr i8, ptr addrspace(3) %address.local863, i64 4608, !dbg !23
  %sched.n1202 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #11, !dbg !23
  %sched.n1203 = bitcast <4 x i32> %sched.n1202 to <8 x half>, !dbg !23
  %address.local886 = getelementptr i8, ptr addrspace(3) %address.local863, i64 4864, !dbg !23
  %sched.n1204 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #11, !dbg !23
  %sched.n1205 = bitcast <4 x i32> %sched.n1204 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local887 = shl nuw nsw i64 %address.local852, 3, !dbg !8
  %address.local888 = and i64 %address.local887, 1536, !dbg !8
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local888, !dbg !8
  %address.local890 = ptrtoint ptr addrspace(3) %address.local889 to i64, !dbg !8
  %address.local891 = trunc i64 %address.local890 to i32, !dbg !8
  %sched.n1212 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local891), !dbg !17
  %address.local892 = shl nuw nsw i64 %address.local852, 5, !dbg !17
  %address.local893 = and i64 %address.local892, 8064, !dbg !17
  %address.local895 = and i64 %address.local887, 24, !dbg !17
  %address.local896 = or disjoint i64 %address.local893, %address.local895, !dbg !17
  %159 = or disjoint i64 %sched.n603.idx32, %address.local896, !dbg !17
  %160 = trunc i64 %159 to i32, !dbg !17
  %.tr37 = shl i32 %160, 1, !dbg !17
  %dma.buffer.offset1543 = or i32 %.tr37, 64, !dbg !17
  %dma.buffer.lds1545 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n1212), !dbg !17
  %sched.n1225 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1545, i32 %dma.buffer.offset1543, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %address.local898 = or disjoint i64 %address.local852, 256, !dbg !8
  %address.local899 = shl nuw nsw i64 %address.local898, 3, !dbg !8
  %address.local900 = and i64 %address.local899, 3584, !dbg !8
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local900, !dbg !8
  %address.local902 = ptrtoint ptr addrspace(3) %address.local901 to i64, !dbg !8
  %address.local903 = trunc i64 %address.local902 to i32, !dbg !8
  %sched.n1226 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local903), !dbg !17
  %address.local904 = shl nuw nsw i64 %address.local898, 5, !dbg !17
  %address.local905 = and i64 %address.local904, 16256, !dbg !17
  %address.local908 = or disjoint i64 %address.local905, %address.local895, !dbg !17
  %161 = or disjoint i64 %sched.n603.idx32, %address.local908, !dbg !17
  %162 = trunc i64 %161 to i32, !dbg !17
  %.tr40 = shl i32 %162, 1, !dbg !17
  %dma.buffer.offset1540 = or i32 %.tr40, 64, !dbg !17
  %dma.buffer.lds1542 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n1226), !dbg !17
  %sched.n1239 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1542, i32 %dma.buffer.offset1540, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %sched.n1240 = shufflevector <8 x half> %sched.n1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1241 = extractvalue [2 x <8 x float>] %sched.n9857, 0, !dbg !24
  %sched.n1242 = shufflevector <8 x float> %sched.n1241, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %sched.n1243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1240, <4 x float> %sched.n1242) #8, !dbg !23
  %sched.n1244 = shufflevector <8 x half> %sched.n1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1244, <4 x float> %sched.n1243) #8, !dbg !23
  %sched.n1246 = shufflevector <8 x half> %sched.n1195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1246, <4 x float> %sched.n1245) #8, !dbg !23
  %sched.n1248 = shufflevector <8 x half> %sched.n1197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1248, <4 x float> %sched.n1247) #8, !dbg !23
  %sched.n1250 = shufflevector <8 x half> %sched.n1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1250, <4 x float> %sched.n1249) #8, !dbg !23
  %sched.n1252 = shufflevector <8 x half> %sched.n1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1252, <4 x float> %sched.n1251) #8, !dbg !23
  %sched.n1254 = shufflevector <8 x half> %sched.n1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1254, <4 x float> %sched.n1253) #8, !dbg !23
  %sched.n1256 = shufflevector <8 x half> %sched.n1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1256, <4 x float> %sched.n1255) #8, !dbg !23
  %sched.n1258 = shufflevector <8 x half> %sched.n1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1259 = shufflevector <8 x float> %sched.n1241, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %sched.n1260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1258, <4 x float> %sched.n1259) #8, !dbg !23
  %sched.n1261 = shufflevector <8 x half> %sched.n1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1261, <4 x float> %sched.n1260) #8, !dbg !23
  %sched.n1263 = shufflevector <8 x half> %sched.n1195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1263, <4 x float> %sched.n1262) #8, !dbg !23
  %sched.n1265 = shufflevector <8 x half> %sched.n1197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1265, <4 x float> %sched.n1264) #8, !dbg !23
  %sched.n1267 = shufflevector <8 x half> %sched.n1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1267, <4 x float> %sched.n1266) #8, !dbg !23
  %sched.n1269 = shufflevector <8 x half> %sched.n1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1269, <4 x float> %sched.n1268) #8, !dbg !23
  %sched.n1271 = shufflevector <8 x half> %sched.n1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1271, <4 x float> %sched.n1270) #8, !dbg !23
  %sched.n1273 = shufflevector <8 x half> %sched.n1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1273, <4 x float> %sched.n1272) #8, !dbg !23
  %sched.n1275 = extractvalue [2 x <8 x float>] %sched.n9857, 1, !dbg !24
  %sched.n1276 = shufflevector <8 x float> %sched.n1275, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %sched.n1277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1240, <4 x float> %sched.n1276) #8, !dbg !23
  %sched.n1278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1244, <4 x float> %sched.n1277) #8, !dbg !23
  %sched.n1279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1246, <4 x float> %sched.n1278) #8, !dbg !23
  %sched.n1280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1248, <4 x float> %sched.n1279) #8, !dbg !23
  %sched.n1281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1250, <4 x float> %sched.n1280) #8, !dbg !23
  %sched.n1282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1252, <4 x float> %sched.n1281) #8, !dbg !23
  %sched.n1283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1254, <4 x float> %sched.n1282) #8, !dbg !23
  %sched.n1284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1256, <4 x float> %sched.n1283) #8, !dbg !23
  %sched.n1285 = shufflevector <8 x float> %sched.n1275, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %sched.n1286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1258, <4 x float> %sched.n1285) #8, !dbg !23
  %sched.n1287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1261, <4 x float> %sched.n1286) #8, !dbg !23
  %sched.n1288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1263, <4 x float> %sched.n1287) #8, !dbg !23
  %sched.n1289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1265, <4 x float> %sched.n1288) #8, !dbg !23
  %sched.n1290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1267, <4 x float> %sched.n1289) #8, !dbg !23
  %sched.n1291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1269, <4 x float> %sched.n1290) #8, !dbg !23
  %sched.n1292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1271, <4 x float> %sched.n1291) #8, !dbg !23
  %sched.n1293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1273, <4 x float> %sched.n1292) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %sched.n1294 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #11, !dbg !23
  %sched.n1295 = bitcast <4 x i32> %sched.n1294 to <8 x half>, !dbg !23
  %sched.n1296 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #11, !dbg !23
  %sched.n1297 = bitcast <4 x i32> %sched.n1296 to <8 x half>, !dbg !23
  %sched.n1298 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #11, !dbg !23
  %sched.n1299 = bitcast <4 x i32> %sched.n1298 to <8 x half>, !dbg !23
  %sched.n1300 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #11, !dbg !23
  %sched.n1301 = bitcast <4 x i32> %sched.n1300 to <8 x half>, !dbg !23
  %sched.n1302 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #11, !dbg !23
  %sched.n1303 = bitcast <4 x i32> %sched.n1302 to <8 x half>, !dbg !23
  %sched.n1304 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #11, !dbg !23
  %sched.n1305 = bitcast <4 x i32> %sched.n1304 to <8 x half>, !dbg !23
  %sched.n1306 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #11, !dbg !23
  %sched.n1307 = bitcast <4 x i32> %sched.n1306 to <8 x half>, !dbg !23
  %sched.n1308 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #11, !dbg !23
  %sched.n1309 = bitcast <4 x i32> %sched.n1308 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %dma.buffer.offset1537 = or i32 %.tr37, 128, !dbg !17
  %sched.n1325 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1545, i32 %dma.buffer.offset1537, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %dma.buffer.offset1534 = or i32 %.tr40, 128, !dbg !17
  %sched.n1339 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1542, i32 %dma.buffer.offset1534, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %sched.n1340 = shufflevector <8 x half> %sched.n1295, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1341 = extractvalue [2 x <8 x float>] %sched.n10059, 0, !dbg !24
  %sched.n1342 = shufflevector <8 x float> %sched.n1341, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %sched.n1343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1340, <4 x float> %sched.n1342) #8, !dbg !23
  %sched.n1344 = shufflevector <8 x half> %sched.n1297, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1344, <4 x float> %sched.n1343) #8, !dbg !23
  %sched.n1346 = shufflevector <8 x half> %sched.n1299, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1346, <4 x float> %sched.n1345) #8, !dbg !23
  %sched.n1348 = shufflevector <8 x half> %sched.n1301, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1348, <4 x float> %sched.n1347) #8, !dbg !23
  %sched.n1350 = shufflevector <8 x half> %sched.n1303, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1350, <4 x float> %sched.n1349) #8, !dbg !23
  %sched.n1352 = shufflevector <8 x half> %sched.n1305, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1352, <4 x float> %sched.n1351) #8, !dbg !23
  %sched.n1354 = shufflevector <8 x half> %sched.n1307, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1354, <4 x float> %sched.n1353) #8, !dbg !23
  %sched.n1356 = shufflevector <8 x half> %sched.n1309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1356, <4 x float> %sched.n1355) #8, !dbg !23
  %sched.n1358 = shufflevector <8 x half> %sched.n1295, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1359 = shufflevector <8 x float> %sched.n1341, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %sched.n1360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1358, <4 x float> %sched.n1359) #8, !dbg !23
  %sched.n1361 = shufflevector <8 x half> %sched.n1297, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1361, <4 x float> %sched.n1360) #8, !dbg !23
  %sched.n1363 = shufflevector <8 x half> %sched.n1299, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1363, <4 x float> %sched.n1362) #8, !dbg !23
  %sched.n1365 = shufflevector <8 x half> %sched.n1301, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1365, <4 x float> %sched.n1364) #8, !dbg !23
  %sched.n1367 = shufflevector <8 x half> %sched.n1303, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1367, <4 x float> %sched.n1366) #8, !dbg !23
  %sched.n1369 = shufflevector <8 x half> %sched.n1305, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1369, <4 x float> %sched.n1368) #8, !dbg !23
  %sched.n1371 = shufflevector <8 x half> %sched.n1307, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1371, <4 x float> %sched.n1370) #8, !dbg !23
  %sched.n1373 = shufflevector <8 x half> %sched.n1309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1373, <4 x float> %sched.n1372) #8, !dbg !23
  %sched.n1375 = extractvalue [2 x <8 x float>] %sched.n10059, 1, !dbg !24
  %sched.n1376 = shufflevector <8 x float> %sched.n1375, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %sched.n1377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1340, <4 x float> %sched.n1376) #8, !dbg !23
  %sched.n1378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1344, <4 x float> %sched.n1377) #8, !dbg !23
  %sched.n1379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1346, <4 x float> %sched.n1378) #8, !dbg !23
  %sched.n1380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1348, <4 x float> %sched.n1379) #8, !dbg !23
  %sched.n1381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1350, <4 x float> %sched.n1380) #8, !dbg !23
  %sched.n1382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1352, <4 x float> %sched.n1381) #8, !dbg !23
  %sched.n1383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1354, <4 x float> %sched.n1382) #8, !dbg !23
  %sched.n1384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1356, <4 x float> %sched.n1383) #8, !dbg !23
  %sched.n1385 = shufflevector <8 x float> %sched.n1375, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %sched.n1386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1358, <4 x float> %sched.n1385) #8, !dbg !23
  %sched.n1387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1361, <4 x float> %sched.n1386) #8, !dbg !23
  %sched.n1388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1363, <4 x float> %sched.n1387) #8, !dbg !23
  %sched.n1389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1365, <4 x float> %sched.n1388) #8, !dbg !23
  %sched.n1390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1367, <4 x float> %sched.n1389) #8, !dbg !23
  %sched.n1391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1369, <4 x float> %sched.n1390) #8, !dbg !23
  %sched.n1392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1371, <4 x float> %sched.n1391) #8, !dbg !23
  %sched.n1393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1373, <4 x float> %sched.n1392) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %sched.n1394 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #11, !dbg !23
  %sched.n1395 = bitcast <4 x i32> %sched.n1394 to <8 x half>, !dbg !23
  %sched.n1396 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #11, !dbg !23
  %sched.n1397 = bitcast <4 x i32> %sched.n1396 to <8 x half>, !dbg !23
  %sched.n1398 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #11, !dbg !23
  %sched.n1399 = bitcast <4 x i32> %sched.n1398 to <8 x half>, !dbg !23
  %sched.n1400 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #11, !dbg !23
  %sched.n1401 = bitcast <4 x i32> %sched.n1400 to <8 x half>, !dbg !23
  %sched.n1402 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #11, !dbg !23
  %sched.n1403 = bitcast <4 x i32> %sched.n1402 to <8 x half>, !dbg !23
  %sched.n1404 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #11, !dbg !23
  %sched.n1405 = bitcast <4 x i32> %sched.n1404 to <8 x half>, !dbg !23
  %sched.n1406 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #11, !dbg !23
  %sched.n1407 = bitcast <4 x i32> %sched.n1406 to <8 x half>, !dbg !23
  %sched.n1408 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #11, !dbg !23
  %sched.n1409 = bitcast <4 x i32> %sched.n1408 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %163 = or disjoint i64 %sched.n603.idx32, 96, !dbg !17
  %164 = or disjoint i64 %address.local896, %163, !dbg !17
  %.tr49 = trunc i64 %164 to i32, !dbg !17
  %dma.buffer.offset1531 = shl i32 %.tr49, 1, !dbg !17
  %sched.n1425 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1545, i32 %dma.buffer.offset1531, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %165 = or disjoint i64 %address.local908, %163, !dbg !17
  %.tr52 = trunc i64 %165 to i32, !dbg !17
  %dma.buffer.offset = shl i32 %.tr52, 1, !dbg !17
  %sched.n1439 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1542, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %sched.n1440 = shufflevector <8 x half> %sched.n1395, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1441 = extractvalue [2 x <8 x float>] %sched.n10160, 0, !dbg !24
  %sched.n1442 = shufflevector <8 x float> %sched.n1441, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %sched.n1443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1440, <4 x float> %sched.n1442) #8, !dbg !23
  %sched.n1444 = shufflevector <8 x half> %sched.n1397, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1444, <4 x float> %sched.n1443) #8, !dbg !23
  %sched.n1446 = shufflevector <8 x half> %sched.n1399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1446, <4 x float> %sched.n1445) #8, !dbg !23
  %sched.n1448 = shufflevector <8 x half> %sched.n1401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1448, <4 x float> %sched.n1447) #8, !dbg !23
  %sched.n1450 = shufflevector <8 x half> %sched.n1403, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1450, <4 x float> %sched.n1449) #8, !dbg !23
  %sched.n1452 = shufflevector <8 x half> %sched.n1405, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1452, <4 x float> %sched.n1451) #8, !dbg !23
  %sched.n1454 = shufflevector <8 x half> %sched.n1407, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1455 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1454, <4 x float> %sched.n1453) #8, !dbg !23
  %sched.n1456 = shufflevector <8 x half> %sched.n1409, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1456, <4 x float> %sched.n1455) #8, !dbg !23
  %sched.n1458 = shufflevector <8 x half> %sched.n1395, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1459 = shufflevector <8 x float> %sched.n1441, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %sched.n1460 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1458, <4 x float> %sched.n1459) #8, !dbg !23
  %sched.n1461 = shufflevector <8 x half> %sched.n1397, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1461, <4 x float> %sched.n1460) #8, !dbg !23
  %sched.n1463 = shufflevector <8 x half> %sched.n1399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1463, <4 x float> %sched.n1462) #8, !dbg !23
  %sched.n1465 = shufflevector <8 x half> %sched.n1401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1466 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1465, <4 x float> %sched.n1464) #8, !dbg !23
  %sched.n1467 = shufflevector <8 x half> %sched.n1403, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1468 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1467, <4 x float> %sched.n1466) #8, !dbg !23
  %sched.n1469 = shufflevector <8 x half> %sched.n1405, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1469, <4 x float> %sched.n1468) #8, !dbg !23
  %sched.n1471 = shufflevector <8 x half> %sched.n1407, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1472 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1471, <4 x float> %sched.n1470) #8, !dbg !23
  %sched.n1473 = shufflevector <8 x half> %sched.n1409, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1474 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1473, <4 x float> %sched.n1472) #8, !dbg !23
  %sched.n1475 = extractvalue [2 x <8 x float>] %sched.n10160, 1, !dbg !24
  %sched.n1476 = shufflevector <8 x float> %sched.n1475, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %sched.n1477 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1440, <4 x float> %sched.n1476) #8, !dbg !23
  %sched.n1478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1444, <4 x float> %sched.n1477) #8, !dbg !23
  %sched.n1479 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1446, <4 x float> %sched.n1478) #8, !dbg !23
  %sched.n1480 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1448, <4 x float> %sched.n1479) #8, !dbg !23
  %sched.n1481 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1450, <4 x float> %sched.n1480) #8, !dbg !23
  %sched.n1482 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1452, <4 x float> %sched.n1481) #8, !dbg !23
  %sched.n1483 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1454, <4 x float> %sched.n1482) #8, !dbg !23
  %sched.n1484 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1456, <4 x float> %sched.n1483) #8, !dbg !23
  %sched.n1485 = shufflevector <8 x float> %sched.n1475, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %sched.n1486 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1458, <4 x float> %sched.n1485) #8, !dbg !23
  %sched.n1487 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1461, <4 x float> %sched.n1486) #8, !dbg !23
  %sched.n1488 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1463, <4 x float> %sched.n1487) #8, !dbg !23
  %sched.n1489 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1465, <4 x float> %sched.n1488) #8, !dbg !23
  %sched.n1490 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1467, <4 x float> %sched.n1489) #8, !dbg !23
  %sched.n1491 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1469, <4 x float> %sched.n1490) #8, !dbg !23
  %sched.n1492 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1471, <4 x float> %sched.n1491) #8, !dbg !23
  %sched.n1493 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1473, <4 x float> %sched.n1492) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %sched.n1494 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #11, !dbg !23
  %sched.n1495 = bitcast <4 x i32> %sched.n1494 to <8 x half>, !dbg !23
  %sched.n1496 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #11, !dbg !23
  %sched.n1497 = bitcast <4 x i32> %sched.n1496 to <8 x half>, !dbg !23
  %sched.n1498 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #11, !dbg !23
  %sched.n1499 = bitcast <4 x i32> %sched.n1498 to <8 x half>, !dbg !23
  %sched.n1500 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #11, !dbg !23
  %sched.n1501 = bitcast <4 x i32> %sched.n1500 to <8 x half>, !dbg !23
  %sched.n1502 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #11, !dbg !23
  %sched.n1503 = bitcast <4 x i32> %sched.n1502 to <8 x half>, !dbg !23
  %sched.n1504 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #11, !dbg !23
  %sched.n1505 = bitcast <4 x i32> %sched.n1504 to <8 x half>, !dbg !23
  %sched.n1506 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #11, !dbg !23
  %sched.n1507 = bitcast <4 x i32> %sched.n1506 to <8 x half>, !dbg !23
  %sched.n1508 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #11, !dbg !23
  %sched.n1509 = bitcast <4 x i32> %sched.n1508 to <8 x half>, !dbg !23
  %sched.n1510 = shufflevector <8 x half> %sched.n1495, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1511 = extractvalue [2 x <8 x float>] %sched.n10261, 0, !dbg !24
  %sched.n1512 = shufflevector <8 x float> %sched.n1511, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %sched.n1513 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1510, <4 x float> %sched.n1512) #8, !dbg !23
  %sched.n1514 = shufflevector <8 x half> %sched.n1497, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1515 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1514, <4 x float> %sched.n1513) #8, !dbg !23
  %sched.n1516 = shufflevector <8 x half> %sched.n1499, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1517 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1516, <4 x float> %sched.n1515) #8, !dbg !23
  %sched.n1518 = shufflevector <8 x half> %sched.n1501, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1519 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1518, <4 x float> %sched.n1517) #8, !dbg !23
  %sched.n1520 = shufflevector <8 x half> %sched.n1503, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1521 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1520, <4 x float> %sched.n1519) #8, !dbg !23
  %sched.n1522 = shufflevector <8 x half> %sched.n1505, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1523 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1522, <4 x float> %sched.n1521) #8, !dbg !23
  %sched.n1524 = shufflevector <8 x half> %sched.n1507, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1525 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1524, <4 x float> %sched.n1523) #8, !dbg !23
  %sched.n1526 = shufflevector <8 x half> %sched.n1509, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %sched.n1527 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1526, <4 x float> %sched.n1525) #8, !dbg !23
  %sched.n1528 = shufflevector <8 x half> %sched.n1495, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1529 = shufflevector <8 x float> %sched.n1511, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %sched.n1530 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1528, <4 x float> %sched.n1529) #8, !dbg !23
  %sched.n1531 = shufflevector <8 x half> %sched.n1497, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1532 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1531, <4 x float> %sched.n1530) #8, !dbg !23
  %sched.n1533 = shufflevector <8 x half> %sched.n1499, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1534 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1533, <4 x float> %sched.n1532) #8, !dbg !23
  %sched.n1535 = shufflevector <8 x half> %sched.n1501, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1536 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1535, <4 x float> %sched.n1534) #8, !dbg !23
  %sched.n1537 = shufflevector <8 x half> %sched.n1503, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1538 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1537, <4 x float> %sched.n1536) #8, !dbg !23
  %sched.n1539 = shufflevector <8 x half> %sched.n1505, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1540 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1539, <4 x float> %sched.n1538) #8, !dbg !23
  %sched.n1541 = shufflevector <8 x half> %sched.n1507, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1541, <4 x float> %sched.n1540) #8, !dbg !23
  %sched.n1543 = shufflevector <8 x half> %sched.n1509, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %sched.n1544 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1543, <4 x float> %sched.n1542) #8, !dbg !23
  %sched.n1545 = extractvalue [2 x <8 x float>] %sched.n10261, 1, !dbg !24
  %sched.n1546 = shufflevector <8 x float> %sched.n1545, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %sched.n1547 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1510, <4 x float> %sched.n1546) #8, !dbg !23
  %sched.n1548 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1514, <4 x float> %sched.n1547) #8, !dbg !23
  %sched.n1549 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1516, <4 x float> %sched.n1548) #8, !dbg !23
  %sched.n1550 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1518, <4 x float> %sched.n1549) #8, !dbg !23
  %sched.n1551 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1520, <4 x float> %sched.n1550) #8, !dbg !23
  %sched.n1552 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1522, <4 x float> %sched.n1551) #8, !dbg !23
  %sched.n1553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1524, <4 x float> %sched.n1552) #8, !dbg !23
  %sched.n1554 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1526, <4 x float> %sched.n1553) #8, !dbg !23
  %sched.n1555 = shufflevector <8 x float> %sched.n1545, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %sched.n1556 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1528, <4 x float> %sched.n1555) #8, !dbg !23
  %sched.n1557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1531, <4 x float> %sched.n1556) #8, !dbg !23
  %sched.n1558 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1533, <4 x float> %sched.n1557) #8, !dbg !23
  %sched.n1559 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1535, <4 x float> %sched.n1558) #8, !dbg !23
  %sched.n1560 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1537, <4 x float> %sched.n1559) #8, !dbg !23
  %sched.n1561 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1539, <4 x float> %sched.n1560) #8, !dbg !23
  %sched.n1562 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1541, <4 x float> %sched.n1561) #8, !dbg !23
  %sched.n1563 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1543, <4 x float> %sched.n1562) #8, !dbg !23
  %sched.n1564 = shufflevector <4 x float> %sched.n1257, <4 x float> %sched.n1274, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %sched.n1565 = insertvalue [2 x <8 x float>] poison, <8 x float> %sched.n1564, 0, !dbg !11
  %sched.n1566 = shufflevector <4 x float> %sched.n1284, <4 x float> %sched.n1293, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %sched.n1567 = insertvalue [2 x <8 x float>] %sched.n1565, <8 x float> %sched.n1566, 1, !dbg !11
  %sched.n1568 = shufflevector <4 x float> %sched.n1357, <4 x float> %sched.n1374, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %sched.n1569 = insertvalue [2 x <8 x float>] poison, <8 x float> %sched.n1568, 0, !dbg !11
  %sched.n1570 = shufflevector <4 x float> %sched.n1384, <4 x float> %sched.n1393, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %sched.n1571 = insertvalue [2 x <8 x float>] %sched.n1569, <8 x float> %sched.n1570, 1, !dbg !11
  %sched.n1572 = shufflevector <4 x float> %sched.n1457, <4 x float> %sched.n1474, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %sched.n1573 = insertvalue [2 x <8 x float>] poison, <8 x float> %sched.n1572, 0, !dbg !11
  %sched.n1574 = shufflevector <4 x float> %sched.n1484, <4 x float> %sched.n1493, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %sched.n1575 = insertvalue [2 x <8 x float>] %sched.n1573, <8 x float> %sched.n1574, 1, !dbg !11
  %sched.n1576 = shufflevector <4 x float> %sched.n1527, <4 x float> %sched.n1544, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %sched.n1577 = insertvalue [2 x <8 x float>] poison, <8 x float> %sched.n1576, 0, !dbg !11
  %sched.n1578 = shufflevector <4 x float> %sched.n1554, <4 x float> %sched.n1563, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %sched.n1579 = insertvalue [2 x <8 x float>] %sched.n1577, <8 x float> %sched.n1578, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %sched.n1580 = add nuw nsw i64 %sched.n9756, 1, !dbg !11
  %sched.n103 = icmp slt i64 %sched.n1580, %sched.n77, !dbg !11
  br i1 %sched.n103, label %sched.n104, label %sched.n1581, !dbg !11

sched.n1581:                                      ; preds = %sched.n847, %.sched.n1581_crit_edge
  %sched.n1610.pre-phi = phi i32 [ %.pre73, %.sched.n1581_crit_edge ], [ %sched.n1164, %sched.n847 ], !dbg !11
  %sched.n1594.pre-phi = phi i32 [ %.pre72, %.sched.n1581_crit_edge ], [ %sched.n1156, %sched.n847 ], !dbg !11
  %sched.n1593.pre-phi = phi i32 [ %.pre71, %.sched.n1581_crit_edge ], [ %sched.n1155, %sched.n847 ], !dbg !11
  %sched.n1591.pre-phi = phi i32 [ %.pre69, %.sched.n1581_crit_edge ], [ %sched.n1153, %sched.n847 ], !dbg !11
  %sched.n98.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.sched.n1581_crit_edge ], [ %sched.n1567, %sched.n847 ], !dbg !11
  %sched.n99.lcssa = phi [2 x <1 x float>] [ zeroinitializer, %.sched.n1581_crit_edge ], [ %sched.n1189, %sched.n847 ], !dbg !11
  %sched.n100.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.sched.n1581_crit_edge ], [ %sched.n1571, %sched.n847 ], !dbg !11
  %sched.n101.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.sched.n1581_crit_edge ], [ %sched.n1575, %sched.n847 ], !dbg !11
  %sched.n102.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.sched.n1581_crit_edge ], [ %sched.n1579, %sched.n847 ], !dbg !11
  %sched.n1582 = and i32 %sched.n31, 16, !dbg !11
  %.not = icmp eq i32 %sched.n1582, 0, !dbg !11
  %sched.n1583 = and i32 %sched.n31, 32, !dbg !11
  %.not36 = icmp eq i32 %sched.n1583, 0, !dbg !11
  %sched.n1584 = extractvalue [2 x <8 x float>] %sched.n102.lcssa, 0, !dbg !11
  %sched.n1585 = extractelement <8 x float> %sched.n1584, i64 0, !dbg !11
  %sched.n1586 = extractelement <8 x float> %sched.n1584, i64 1, !dbg !11
  %sched.n1587 = extractelement <8 x float> %sched.n1584, i64 2, !dbg !11
  %sched.n1588 = extractelement <8 x float> %sched.n1584, i64 3, !dbg !11
  %sched.n1589 = select i1 %.not, float %sched.n1586, float %sched.n1585, !dbg !11
  %sched.n1595 = icmp samesign ult i32 %sched.n1593.pre-phi, %sched.n1594.pre-phi, !dbg !11
  %sched.n1596 = select i1 %sched.n1595, i32 %sched.n1593.pre-phi, i32 %sched.n1591.pre-phi, !dbg !11
  %sched.n1597 = shl nuw nsw i32 %sched.n1596, 2, !dbg !11
  %sched.n1598 = bitcast float %sched.n1589 to i32, !dbg !11
  %sched.n1599 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1598), !dbg !11
  %sched.n1600 = bitcast i32 %sched.n1599 to float, !dbg !11
  %sched.n1601 = select i1 %.not, float %sched.n1585, float %sched.n1600, !dbg !11
  %sched.n1602 = select i1 %.not, float %sched.n1600, float %sched.n1586, !dbg !11
  %sched.n1603 = select i1 %.not, float %sched.n1588, float %sched.n1587, !dbg !11
  %sched.n1604 = bitcast float %sched.n1603 to i32, !dbg !11
  %sched.n1605 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1604), !dbg !11
  %sched.n1606 = bitcast i32 %sched.n1605 to float, !dbg !11
  %sched.n1607 = select i1 %.not, float %sched.n1587, float %sched.n1606, !dbg !11
  %sched.n1608 = select i1 %.not, float %sched.n1606, float %sched.n1588, !dbg !11
  %sched.n1609 = select i1 %.not36, float %sched.n1607, float %sched.n1601, !dbg !11
  %sched.n1611 = icmp samesign ult i32 %sched.n1610.pre-phi, %sched.n1594.pre-phi, !dbg !11
  %sched.n1612 = select i1 %sched.n1611, i32 %sched.n1610.pre-phi, i32 %sched.n1591.pre-phi, !dbg !11
  %sched.n1613 = shl nuw nsw i32 %sched.n1612, 2, !dbg !11
  %sched.n1614 = bitcast float %sched.n1609 to i32, !dbg !11
  %sched.n1615 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1614), !dbg !11
  %sched.n1616 = bitcast i32 %sched.n1615 to float, !dbg !11
  %sched.n1619 = select i1 %.not36, float %sched.n1608, float %sched.n1602, !dbg !11
  %sched.n1620 = bitcast float %sched.n1619 to i32, !dbg !11
  %sched.n1621 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1620), !dbg !11
  %sched.n1622 = bitcast i32 %sched.n1621 to float, !dbg !11
  %166 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %167 = shufflevector <4 x i1> %166, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %168 = insertelement <4 x float> poison, float %sched.n1601, i64 0, !dbg !11
  %169 = insertelement <4 x float> %168, float %sched.n1602, i64 1, !dbg !11
  %170 = insertelement <4 x float> %169, float %sched.n1616, i64 2, !dbg !11
  %171 = insertelement <4 x float> %170, float %sched.n1622, i64 3, !dbg !11
  %172 = insertelement <4 x float> poison, float %sched.n1616, i64 0, !dbg !11
  %173 = insertelement <4 x float> %172, float %sched.n1622, i64 1, !dbg !11
  %174 = insertelement <4 x float> %173, float %sched.n1607, i64 2, !dbg !11
  %175 = insertelement <4 x float> %174, float %sched.n1608, i64 3, !dbg !11
  %176 = select <4 x i1> %167, <4 x float> %171, <4 x float> %175, !dbg !11
  %sched.n1625 = extractvalue [2 x <8 x float>] %sched.n101.lcssa, 0, !dbg !11
  %sched.n1626 = extractelement <8 x float> %sched.n1625, i64 0, !dbg !11
  %sched.n1627 = extractelement <8 x float> %sched.n1625, i64 1, !dbg !11
  %sched.n1628 = extractelement <8 x float> %sched.n1625, i64 2, !dbg !11
  %sched.n1629 = extractelement <8 x float> %sched.n1625, i64 3, !dbg !11
  %sched.n1630 = select i1 %.not, float %sched.n1627, float %sched.n1626, !dbg !11
  %sched.n1631 = bitcast float %sched.n1630 to i32, !dbg !11
  %sched.n1632 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1631), !dbg !11
  %sched.n1633 = bitcast i32 %sched.n1632 to float, !dbg !11
  %sched.n1634 = select i1 %.not, float %sched.n1626, float %sched.n1633, !dbg !11
  %sched.n1635 = select i1 %.not, float %sched.n1633, float %sched.n1627, !dbg !11
  %sched.n1636 = select i1 %.not, float %sched.n1629, float %sched.n1628, !dbg !11
  %sched.n1637 = bitcast float %sched.n1636 to i32, !dbg !11
  %sched.n1638 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1637), !dbg !11
  %sched.n1639 = bitcast i32 %sched.n1638 to float, !dbg !11
  %sched.n1640 = select i1 %.not, float %sched.n1628, float %sched.n1639, !dbg !11
  %sched.n1641 = select i1 %.not, float %sched.n1639, float %sched.n1629, !dbg !11
  %sched.n1642 = select i1 %.not36, float %sched.n1640, float %sched.n1634, !dbg !11
  %sched.n1643 = bitcast float %sched.n1642 to i32, !dbg !11
  %sched.n1644 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1643), !dbg !11
  %sched.n1645 = bitcast i32 %sched.n1644 to float, !dbg !11
  %sched.n1648 = select i1 %.not36, float %sched.n1641, float %sched.n1635, !dbg !11
  %sched.n1649 = bitcast float %sched.n1648 to i32, !dbg !11
  %sched.n1650 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1649), !dbg !11
  %sched.n1651 = bitcast i32 %sched.n1650 to float, !dbg !11
  %177 = insertelement <4 x float> poison, float %sched.n1634, i64 0, !dbg !11
  %178 = insertelement <4 x float> %177, float %sched.n1635, i64 1, !dbg !11
  %179 = insertelement <4 x float> %178, float %sched.n1645, i64 2, !dbg !11
  %180 = insertelement <4 x float> %179, float %sched.n1651, i64 3, !dbg !11
  %181 = insertelement <4 x float> poison, float %sched.n1645, i64 0, !dbg !11
  %182 = insertelement <4 x float> %181, float %sched.n1651, i64 1, !dbg !11
  %183 = insertelement <4 x float> %182, float %sched.n1640, i64 2, !dbg !11
  %184 = insertelement <4 x float> %183, float %sched.n1641, i64 3, !dbg !11
  %185 = select <4 x i1> %167, <4 x float> %180, <4 x float> %184, !dbg !11
  %sched.n1654 = extractvalue [2 x <8 x float>] %sched.n100.lcssa, 0, !dbg !11
  %sched.n1655 = extractelement <8 x float> %sched.n1654, i64 0, !dbg !11
  %sched.n1656 = extractelement <8 x float> %sched.n1654, i64 1, !dbg !11
  %sched.n1657 = extractelement <8 x float> %sched.n1654, i64 2, !dbg !11
  %sched.n1658 = extractelement <8 x float> %sched.n1654, i64 3, !dbg !11
  %sched.n1659 = select i1 %.not, float %sched.n1656, float %sched.n1655, !dbg !11
  %sched.n1660 = bitcast float %sched.n1659 to i32, !dbg !11
  %sched.n1661 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1660), !dbg !11
  %sched.n1662 = bitcast i32 %sched.n1661 to float, !dbg !11
  %sched.n1663 = select i1 %.not, float %sched.n1655, float %sched.n1662, !dbg !11
  %sched.n1664 = select i1 %.not, float %sched.n1662, float %sched.n1656, !dbg !11
  %sched.n1665 = select i1 %.not, float %sched.n1658, float %sched.n1657, !dbg !11
  %sched.n1666 = bitcast float %sched.n1665 to i32, !dbg !11
  %sched.n1667 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1666), !dbg !11
  %sched.n1668 = bitcast i32 %sched.n1667 to float, !dbg !11
  %sched.n1669 = select i1 %.not, float %sched.n1657, float %sched.n1668, !dbg !11
  %sched.n1670 = select i1 %.not, float %sched.n1668, float %sched.n1658, !dbg !11
  %sched.n1671 = select i1 %.not36, float %sched.n1669, float %sched.n1663, !dbg !11
  %sched.n1672 = bitcast float %sched.n1671 to i32, !dbg !11
  %sched.n1673 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1672), !dbg !11
  %sched.n1674 = bitcast i32 %sched.n1673 to float, !dbg !11
  %sched.n1677 = select i1 %.not36, float %sched.n1670, float %sched.n1664, !dbg !11
  %sched.n1678 = bitcast float %sched.n1677 to i32, !dbg !11
  %sched.n1679 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1678), !dbg !11
  %sched.n1680 = bitcast i32 %sched.n1679 to float, !dbg !11
  %186 = insertelement <4 x float> poison, float %sched.n1663, i64 0, !dbg !11
  %187 = insertelement <4 x float> %186, float %sched.n1664, i64 1, !dbg !11
  %188 = insertelement <4 x float> %187, float %sched.n1674, i64 2, !dbg !11
  %189 = insertelement <4 x float> %188, float %sched.n1680, i64 3, !dbg !11
  %190 = insertelement <4 x float> poison, float %sched.n1674, i64 0, !dbg !11
  %191 = insertelement <4 x float> %190, float %sched.n1680, i64 1, !dbg !11
  %192 = insertelement <4 x float> %191, float %sched.n1669, i64 2, !dbg !11
  %193 = insertelement <4 x float> %192, float %sched.n1670, i64 3, !dbg !11
  %194 = select <4 x i1> %167, <4 x float> %189, <4 x float> %193, !dbg !11
  %sched.n1683 = extractvalue [2 x <8 x float>] %sched.n98.lcssa, 0, !dbg !11
  %sched.n1684 = extractelement <8 x float> %sched.n1683, i64 0, !dbg !11
  %sched.n1685 = extractelement <8 x float> %sched.n1683, i64 1, !dbg !11
  %sched.n1686 = extractelement <8 x float> %sched.n1683, i64 2, !dbg !11
  %sched.n1687 = extractelement <8 x float> %sched.n1683, i64 3, !dbg !11
  %sched.n1688 = select i1 %.not, float %sched.n1685, float %sched.n1684, !dbg !11
  %sched.n1689 = bitcast float %sched.n1688 to i32, !dbg !11
  %sched.n1690 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1689), !dbg !11
  %sched.n1691 = bitcast i32 %sched.n1690 to float, !dbg !11
  %sched.n1692 = select i1 %.not, float %sched.n1684, float %sched.n1691, !dbg !11
  %sched.n1693 = select i1 %.not, float %sched.n1691, float %sched.n1685, !dbg !11
  %sched.n1694 = select i1 %.not, float %sched.n1687, float %sched.n1686, !dbg !11
  %sched.n1695 = bitcast float %sched.n1694 to i32, !dbg !11
  %sched.n1696 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1695), !dbg !11
  %sched.n1697 = bitcast i32 %sched.n1696 to float, !dbg !11
  %sched.n1698 = select i1 %.not, float %sched.n1686, float %sched.n1697, !dbg !11
  %sched.n1699 = select i1 %.not, float %sched.n1697, float %sched.n1687, !dbg !11
  %sched.n1700 = select i1 %.not36, float %sched.n1698, float %sched.n1692, !dbg !11
  %sched.n1701 = bitcast float %sched.n1700 to i32, !dbg !11
  %sched.n1702 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1701), !dbg !11
  %sched.n1703 = select i1 %.not36, float %sched.n1699, float %sched.n1693, !dbg !11
  %sched.n1704 = bitcast float %sched.n1703 to i32, !dbg !11
  %sched.n1705 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1704), !dbg !11
  %sched.n1706 = extractvalue [2 x <1 x float>] %sched.n99.lcssa, 0, !dbg !25
  %sched.n1707 = shufflevector <1 x float> %sched.n1706, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %sched.n1708 = extractelement <8 x float> %sched.n1584, i64 4, !dbg !11
  %sched.n1709 = extractelement <8 x float> %sched.n1584, i64 5, !dbg !11
  %sched.n1710 = extractelement <8 x float> %sched.n1584, i64 6, !dbg !11
  %sched.n1711 = extractelement <8 x float> %sched.n1584, i64 7, !dbg !11
  %sched.n1712 = select i1 %.not, float %sched.n1709, float %sched.n1708, !dbg !11
  %sched.n1713 = bitcast float %sched.n1712 to i32, !dbg !11
  %sched.n1714 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1713), !dbg !11
  %sched.n1715 = bitcast i32 %sched.n1714 to float, !dbg !11
  %sched.n1716 = select i1 %.not, float %sched.n1708, float %sched.n1715, !dbg !11
  %sched.n1717 = select i1 %.not, float %sched.n1715, float %sched.n1709, !dbg !11
  %sched.n1718 = select i1 %.not, float %sched.n1711, float %sched.n1710, !dbg !11
  %sched.n1719 = bitcast float %sched.n1718 to i32, !dbg !11
  %sched.n1720 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1719), !dbg !11
  %sched.n1721 = bitcast i32 %sched.n1720 to float, !dbg !11
  %sched.n1722 = select i1 %.not, float %sched.n1710, float %sched.n1721, !dbg !11
  %sched.n1723 = select i1 %.not, float %sched.n1721, float %sched.n1711, !dbg !11
  %sched.n1724 = select i1 %.not36, float %sched.n1722, float %sched.n1716, !dbg !11
  %sched.n1725 = bitcast float %sched.n1724 to i32, !dbg !11
  %sched.n1726 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1725), !dbg !11
  %sched.n1727 = bitcast i32 %sched.n1726 to float, !dbg !11
  %sched.n1730 = select i1 %.not36, float %sched.n1723, float %sched.n1717, !dbg !11
  %sched.n1731 = bitcast float %sched.n1730 to i32, !dbg !11
  %sched.n1732 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1731), !dbg !11
  %sched.n1733 = bitcast i32 %sched.n1732 to float, !dbg !11
  %195 = insertelement <4 x float> poison, float %sched.n1716, i64 0, !dbg !11
  %196 = insertelement <4 x float> %195, float %sched.n1717, i64 1, !dbg !11
  %197 = insertelement <4 x float> %196, float %sched.n1727, i64 2, !dbg !11
  %198 = insertelement <4 x float> %197, float %sched.n1733, i64 3, !dbg !11
  %199 = insertelement <4 x float> poison, float %sched.n1727, i64 0, !dbg !11
  %200 = insertelement <4 x float> %199, float %sched.n1733, i64 1, !dbg !11
  %201 = insertelement <4 x float> %200, float %sched.n1722, i64 2, !dbg !11
  %202 = insertelement <4 x float> %201, float %sched.n1723, i64 3, !dbg !11
  %203 = select <4 x i1> %167, <4 x float> %198, <4 x float> %202, !dbg !11
  %sched.n1736 = extractelement <8 x float> %sched.n1625, i64 4, !dbg !11
  %sched.n1737 = extractelement <8 x float> %sched.n1625, i64 5, !dbg !11
  %sched.n1738 = extractelement <8 x float> %sched.n1625, i64 6, !dbg !11
  %sched.n1739 = extractelement <8 x float> %sched.n1625, i64 7, !dbg !11
  %sched.n1740 = select i1 %.not, float %sched.n1737, float %sched.n1736, !dbg !11
  %sched.n1741 = bitcast float %sched.n1740 to i32, !dbg !11
  %sched.n1742 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1741), !dbg !11
  %sched.n1743 = bitcast i32 %sched.n1742 to float, !dbg !11
  %sched.n1744 = select i1 %.not, float %sched.n1736, float %sched.n1743, !dbg !11
  %sched.n1745 = select i1 %.not, float %sched.n1743, float %sched.n1737, !dbg !11
  %sched.n1746 = select i1 %.not, float %sched.n1739, float %sched.n1738, !dbg !11
  %sched.n1747 = bitcast float %sched.n1746 to i32, !dbg !11
  %sched.n1748 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1747), !dbg !11
  %sched.n1749 = bitcast i32 %sched.n1748 to float, !dbg !11
  %sched.n1750 = select i1 %.not, float %sched.n1738, float %sched.n1749, !dbg !11
  %sched.n1751 = select i1 %.not, float %sched.n1749, float %sched.n1739, !dbg !11
  %sched.n1752 = select i1 %.not36, float %sched.n1750, float %sched.n1744, !dbg !11
  %sched.n1753 = bitcast float %sched.n1752 to i32, !dbg !11
  %sched.n1754 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1753), !dbg !11
  %sched.n1755 = bitcast i32 %sched.n1754 to float, !dbg !11
  %sched.n1758 = select i1 %.not36, float %sched.n1751, float %sched.n1745, !dbg !11
  %sched.n1759 = bitcast float %sched.n1758 to i32, !dbg !11
  %sched.n1760 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1759), !dbg !11
  %sched.n1761 = bitcast i32 %sched.n1760 to float, !dbg !11
  %204 = insertelement <4 x float> poison, float %sched.n1744, i64 0, !dbg !11
  %205 = insertelement <4 x float> %204, float %sched.n1745, i64 1, !dbg !11
  %206 = insertelement <4 x float> %205, float %sched.n1755, i64 2, !dbg !11
  %207 = insertelement <4 x float> %206, float %sched.n1761, i64 3, !dbg !11
  %208 = insertelement <4 x float> poison, float %sched.n1755, i64 0, !dbg !11
  %209 = insertelement <4 x float> %208, float %sched.n1761, i64 1, !dbg !11
  %210 = insertelement <4 x float> %209, float %sched.n1750, i64 2, !dbg !11
  %211 = insertelement <4 x float> %210, float %sched.n1751, i64 3, !dbg !11
  %212 = select <4 x i1> %167, <4 x float> %207, <4 x float> %211, !dbg !11
  %sched.n1764 = extractelement <8 x float> %sched.n1654, i64 4, !dbg !11
  %sched.n1765 = extractelement <8 x float> %sched.n1654, i64 5, !dbg !11
  %sched.n1766 = extractelement <8 x float> %sched.n1654, i64 6, !dbg !11
  %sched.n1767 = extractelement <8 x float> %sched.n1654, i64 7, !dbg !11
  %sched.n1768 = select i1 %.not, float %sched.n1765, float %sched.n1764, !dbg !11
  %sched.n1769 = bitcast float %sched.n1768 to i32, !dbg !11
  %sched.n1770 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1769), !dbg !11
  %sched.n1771 = bitcast i32 %sched.n1770 to float, !dbg !11
  %sched.n1772 = select i1 %.not, float %sched.n1764, float %sched.n1771, !dbg !11
  %sched.n1773 = select i1 %.not, float %sched.n1771, float %sched.n1765, !dbg !11
  %sched.n1774 = select i1 %.not, float %sched.n1767, float %sched.n1766, !dbg !11
  %sched.n1775 = bitcast float %sched.n1774 to i32, !dbg !11
  %sched.n1776 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1775), !dbg !11
  %sched.n1777 = bitcast i32 %sched.n1776 to float, !dbg !11
  %sched.n1778 = select i1 %.not, float %sched.n1766, float %sched.n1777, !dbg !11
  %sched.n1779 = select i1 %.not, float %sched.n1777, float %sched.n1767, !dbg !11
  %sched.n1780 = select i1 %.not36, float %sched.n1778, float %sched.n1772, !dbg !11
  %sched.n1781 = bitcast float %sched.n1780 to i32, !dbg !11
  %sched.n1782 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1781), !dbg !11
  %sched.n1783 = bitcast i32 %sched.n1782 to float, !dbg !11
  %sched.n1786 = select i1 %.not36, float %sched.n1779, float %sched.n1773, !dbg !11
  %sched.n1787 = bitcast float %sched.n1786 to i32, !dbg !11
  %sched.n1788 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1787), !dbg !11
  %sched.n1789 = bitcast i32 %sched.n1788 to float, !dbg !11
  %213 = insertelement <4 x float> poison, float %sched.n1772, i64 0, !dbg !11
  %214 = insertelement <4 x float> %213, float %sched.n1773, i64 1, !dbg !11
  %215 = insertelement <4 x float> %214, float %sched.n1783, i64 2, !dbg !11
  %216 = insertelement <4 x float> %215, float %sched.n1789, i64 3, !dbg !11
  %217 = insertelement <4 x float> poison, float %sched.n1783, i64 0, !dbg !11
  %218 = insertelement <4 x float> %217, float %sched.n1789, i64 1, !dbg !11
  %219 = insertelement <4 x float> %218, float %sched.n1778, i64 2, !dbg !11
  %220 = insertelement <4 x float> %219, float %sched.n1779, i64 3, !dbg !11
  %221 = select <4 x i1> %167, <4 x float> %216, <4 x float> %220, !dbg !11
  %sched.n1792 = extractelement <8 x float> %sched.n1683, i64 4, !dbg !11
  %sched.n1793 = extractelement <8 x float> %sched.n1683, i64 5, !dbg !11
  %sched.n1794 = extractelement <8 x float> %sched.n1683, i64 6, !dbg !11
  %sched.n1795 = extractelement <8 x float> %sched.n1683, i64 7, !dbg !11
  %sched.n1796 = select i1 %.not, float %sched.n1793, float %sched.n1792, !dbg !11
  %sched.n1797 = bitcast float %sched.n1796 to i32, !dbg !11
  %sched.n1798 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1797), !dbg !11
  %sched.n1799 = bitcast i32 %sched.n1798 to float, !dbg !11
  %sched.n1800 = select i1 %.not, float %sched.n1792, float %sched.n1799, !dbg !11
  %sched.n1801 = select i1 %.not, float %sched.n1799, float %sched.n1793, !dbg !11
  %sched.n1802 = select i1 %.not, float %sched.n1795, float %sched.n1794, !dbg !11
  %sched.n1803 = bitcast float %sched.n1802 to i32, !dbg !11
  %sched.n1804 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1803), !dbg !11
  %sched.n1805 = bitcast i32 %sched.n1804 to float, !dbg !11
  %sched.n1806 = select i1 %.not, float %sched.n1794, float %sched.n1805, !dbg !11
  %sched.n1807 = select i1 %.not, float %sched.n1805, float %sched.n1795, !dbg !11
  %sched.n1808 = select i1 %.not36, float %sched.n1806, float %sched.n1800, !dbg !11
  %sched.n1809 = bitcast float %sched.n1808 to i32, !dbg !11
  %sched.n1810 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1809), !dbg !11
  %sched.n1811 = select i1 %.not36, float %sched.n1807, float %sched.n1801, !dbg !11
  %sched.n1812 = bitcast float %sched.n1811 to i32, !dbg !11
  %sched.n1813 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1812), !dbg !11
  %sched.n1814 = extractvalue [2 x <8 x float>] %sched.n102.lcssa, 1, !dbg !11
  %sched.n1815 = extractelement <8 x float> %sched.n1814, i64 0, !dbg !11
  %sched.n1816 = extractelement <8 x float> %sched.n1814, i64 1, !dbg !11
  %sched.n1817 = extractelement <8 x float> %sched.n1814, i64 2, !dbg !11
  %sched.n1818 = extractelement <8 x float> %sched.n1814, i64 3, !dbg !11
  %sched.n1819 = select i1 %.not, float %sched.n1816, float %sched.n1815, !dbg !11
  %sched.n1820 = bitcast float %sched.n1819 to i32, !dbg !11
  %sched.n1821 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1820), !dbg !11
  %sched.n1822 = bitcast i32 %sched.n1821 to float, !dbg !11
  %sched.n1823 = select i1 %.not, float %sched.n1815, float %sched.n1822, !dbg !11
  %sched.n1824 = select i1 %.not, float %sched.n1822, float %sched.n1816, !dbg !11
  %sched.n1825 = select i1 %.not, float %sched.n1818, float %sched.n1817, !dbg !11
  %sched.n1826 = bitcast float %sched.n1825 to i32, !dbg !11
  %sched.n1827 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1826), !dbg !11
  %sched.n1828 = bitcast i32 %sched.n1827 to float, !dbg !11
  %sched.n1829 = select i1 %.not, float %sched.n1817, float %sched.n1828, !dbg !11
  %sched.n1830 = select i1 %.not, float %sched.n1828, float %sched.n1818, !dbg !11
  %sched.n1831 = select i1 %.not36, float %sched.n1829, float %sched.n1823, !dbg !11
  %sched.n1832 = bitcast float %sched.n1831 to i32, !dbg !11
  %sched.n1833 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1832), !dbg !11
  %sched.n1834 = bitcast i32 %sched.n1833 to float, !dbg !11
  %sched.n1837 = select i1 %.not36, float %sched.n1830, float %sched.n1824, !dbg !11
  %sched.n1838 = bitcast float %sched.n1837 to i32, !dbg !11
  %sched.n1839 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1838), !dbg !11
  %sched.n1840 = bitcast i32 %sched.n1839 to float, !dbg !11
  %222 = insertelement <4 x float> poison, float %sched.n1823, i64 0, !dbg !11
  %223 = insertelement <4 x float> %222, float %sched.n1824, i64 1, !dbg !11
  %224 = insertelement <4 x float> %223, float %sched.n1834, i64 2, !dbg !11
  %225 = insertelement <4 x float> %224, float %sched.n1840, i64 3, !dbg !11
  %226 = insertelement <4 x float> poison, float %sched.n1834, i64 0, !dbg !11
  %227 = insertelement <4 x float> %226, float %sched.n1840, i64 1, !dbg !11
  %228 = insertelement <4 x float> %227, float %sched.n1829, i64 2, !dbg !11
  %229 = insertelement <4 x float> %228, float %sched.n1830, i64 3, !dbg !11
  %230 = select <4 x i1> %167, <4 x float> %225, <4 x float> %229, !dbg !11
  %sched.n1843 = extractvalue [2 x <8 x float>] %sched.n101.lcssa, 1, !dbg !11
  %sched.n1844 = extractelement <8 x float> %sched.n1843, i64 0, !dbg !11
  %sched.n1845 = extractelement <8 x float> %sched.n1843, i64 1, !dbg !11
  %sched.n1846 = extractelement <8 x float> %sched.n1843, i64 2, !dbg !11
  %sched.n1847 = extractelement <8 x float> %sched.n1843, i64 3, !dbg !11
  %sched.n1848 = select i1 %.not, float %sched.n1845, float %sched.n1844, !dbg !11
  %sched.n1849 = bitcast float %sched.n1848 to i32, !dbg !11
  %sched.n1850 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1849), !dbg !11
  %sched.n1851 = bitcast i32 %sched.n1850 to float, !dbg !11
  %sched.n1852 = select i1 %.not, float %sched.n1844, float %sched.n1851, !dbg !11
  %sched.n1853 = select i1 %.not, float %sched.n1851, float %sched.n1845, !dbg !11
  %sched.n1854 = select i1 %.not, float %sched.n1847, float %sched.n1846, !dbg !11
  %sched.n1855 = bitcast float %sched.n1854 to i32, !dbg !11
  %sched.n1856 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1855), !dbg !11
  %sched.n1857 = bitcast i32 %sched.n1856 to float, !dbg !11
  %sched.n1858 = select i1 %.not, float %sched.n1846, float %sched.n1857, !dbg !11
  %sched.n1859 = select i1 %.not, float %sched.n1857, float %sched.n1847, !dbg !11
  %sched.n1860 = select i1 %.not36, float %sched.n1858, float %sched.n1852, !dbg !11
  %sched.n1861 = bitcast float %sched.n1860 to i32, !dbg !11
  %sched.n1862 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1861), !dbg !11
  %sched.n1863 = bitcast i32 %sched.n1862 to float, !dbg !11
  %sched.n1866 = select i1 %.not36, float %sched.n1859, float %sched.n1853, !dbg !11
  %sched.n1867 = bitcast float %sched.n1866 to i32, !dbg !11
  %sched.n1868 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1867), !dbg !11
  %sched.n1869 = bitcast i32 %sched.n1868 to float, !dbg !11
  %231 = insertelement <4 x float> poison, float %sched.n1852, i64 0, !dbg !11
  %232 = insertelement <4 x float> %231, float %sched.n1853, i64 1, !dbg !11
  %233 = insertelement <4 x float> %232, float %sched.n1863, i64 2, !dbg !11
  %234 = insertelement <4 x float> %233, float %sched.n1869, i64 3, !dbg !11
  %235 = insertelement <4 x float> poison, float %sched.n1863, i64 0, !dbg !11
  %236 = insertelement <4 x float> %235, float %sched.n1869, i64 1, !dbg !11
  %237 = insertelement <4 x float> %236, float %sched.n1858, i64 2, !dbg !11
  %238 = insertelement <4 x float> %237, float %sched.n1859, i64 3, !dbg !11
  %239 = select <4 x i1> %167, <4 x float> %234, <4 x float> %238, !dbg !11
  %sched.n1872 = extractvalue [2 x <8 x float>] %sched.n100.lcssa, 1, !dbg !11
  %sched.n1873 = extractelement <8 x float> %sched.n1872, i64 0, !dbg !11
  %sched.n1874 = extractelement <8 x float> %sched.n1872, i64 1, !dbg !11
  %sched.n1875 = extractelement <8 x float> %sched.n1872, i64 2, !dbg !11
  %sched.n1876 = extractelement <8 x float> %sched.n1872, i64 3, !dbg !11
  %sched.n1877 = select i1 %.not, float %sched.n1874, float %sched.n1873, !dbg !11
  %sched.n1878 = bitcast float %sched.n1877 to i32, !dbg !11
  %sched.n1879 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1878), !dbg !11
  %sched.n1880 = bitcast i32 %sched.n1879 to float, !dbg !11
  %sched.n1881 = select i1 %.not, float %sched.n1873, float %sched.n1880, !dbg !11
  %sched.n1882 = select i1 %.not, float %sched.n1880, float %sched.n1874, !dbg !11
  %sched.n1883 = select i1 %.not, float %sched.n1876, float %sched.n1875, !dbg !11
  %sched.n1884 = bitcast float %sched.n1883 to i32, !dbg !11
  %sched.n1885 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1884), !dbg !11
  %sched.n1886 = bitcast i32 %sched.n1885 to float, !dbg !11
  %sched.n1887 = select i1 %.not, float %sched.n1875, float %sched.n1886, !dbg !11
  %sched.n1888 = select i1 %.not, float %sched.n1886, float %sched.n1876, !dbg !11
  %sched.n1889 = select i1 %.not36, float %sched.n1887, float %sched.n1881, !dbg !11
  %sched.n1890 = bitcast float %sched.n1889 to i32, !dbg !11
  %sched.n1891 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1890), !dbg !11
  %sched.n1892 = bitcast i32 %sched.n1891 to float, !dbg !11
  %sched.n1895 = select i1 %.not36, float %sched.n1888, float %sched.n1882, !dbg !11
  %sched.n1896 = bitcast float %sched.n1895 to i32, !dbg !11
  %sched.n1897 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1896), !dbg !11
  %sched.n1898 = bitcast i32 %sched.n1897 to float, !dbg !11
  %240 = insertelement <4 x float> poison, float %sched.n1881, i64 0, !dbg !11
  %241 = insertelement <4 x float> %240, float %sched.n1882, i64 1, !dbg !11
  %242 = insertelement <4 x float> %241, float %sched.n1892, i64 2, !dbg !11
  %243 = insertelement <4 x float> %242, float %sched.n1898, i64 3, !dbg !11
  %244 = insertelement <4 x float> poison, float %sched.n1892, i64 0, !dbg !11
  %245 = insertelement <4 x float> %244, float %sched.n1898, i64 1, !dbg !11
  %246 = insertelement <4 x float> %245, float %sched.n1887, i64 2, !dbg !11
  %247 = insertelement <4 x float> %246, float %sched.n1888, i64 3, !dbg !11
  %248 = select <4 x i1> %167, <4 x float> %243, <4 x float> %247, !dbg !11
  %sched.n1901 = extractvalue [2 x <8 x float>] %sched.n98.lcssa, 1, !dbg !11
  %sched.n1902 = extractelement <8 x float> %sched.n1901, i64 0, !dbg !11
  %sched.n1903 = extractelement <8 x float> %sched.n1901, i64 1, !dbg !11
  %sched.n1904 = extractelement <8 x float> %sched.n1901, i64 2, !dbg !11
  %sched.n1905 = extractelement <8 x float> %sched.n1901, i64 3, !dbg !11
  %sched.n1906 = select i1 %.not, float %sched.n1903, float %sched.n1902, !dbg !11
  %sched.n1907 = bitcast float %sched.n1906 to i32, !dbg !11
  %sched.n1908 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1907), !dbg !11
  %sched.n1909 = bitcast i32 %sched.n1908 to float, !dbg !11
  %sched.n1910 = select i1 %.not, float %sched.n1902, float %sched.n1909, !dbg !11
  %sched.n1911 = select i1 %.not, float %sched.n1909, float %sched.n1903, !dbg !11
  %sched.n1912 = select i1 %.not, float %sched.n1905, float %sched.n1904, !dbg !11
  %sched.n1913 = bitcast float %sched.n1912 to i32, !dbg !11
  %sched.n1914 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1913), !dbg !11
  %sched.n1915 = bitcast i32 %sched.n1914 to float, !dbg !11
  %sched.n1916 = select i1 %.not, float %sched.n1904, float %sched.n1915, !dbg !11
  %sched.n1917 = select i1 %.not, float %sched.n1915, float %sched.n1905, !dbg !11
  %sched.n1918 = select i1 %.not36, float %sched.n1916, float %sched.n1910, !dbg !11
  %sched.n1919 = bitcast float %sched.n1918 to i32, !dbg !11
  %sched.n1920 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1919), !dbg !11
  %sched.n1921 = select i1 %.not36, float %sched.n1917, float %sched.n1911, !dbg !11
  %sched.n1922 = bitcast float %sched.n1921 to i32, !dbg !11
  %sched.n1923 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1922), !dbg !11
  %sched.n1924 = extractvalue [2 x <1 x float>] %sched.n99.lcssa, 1, !dbg !25
  %sched.n1925 = shufflevector <1 x float> %sched.n1924, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %sched.n1926 = extractelement <8 x float> %sched.n1814, i64 4, !dbg !11
  %sched.n1927 = extractelement <8 x float> %sched.n1814, i64 5, !dbg !11
  %sched.n1928 = extractelement <8 x float> %sched.n1814, i64 6, !dbg !11
  %sched.n1929 = extractelement <8 x float> %sched.n1814, i64 7, !dbg !11
  %sched.n1930 = select i1 %.not, float %sched.n1927, float %sched.n1926, !dbg !11
  %sched.n1931 = bitcast float %sched.n1930 to i32, !dbg !11
  %sched.n1932 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1931), !dbg !11
  %sched.n1933 = bitcast i32 %sched.n1932 to float, !dbg !11
  %sched.n1934 = select i1 %.not, float %sched.n1926, float %sched.n1933, !dbg !11
  %sched.n1935 = select i1 %.not, float %sched.n1933, float %sched.n1927, !dbg !11
  %sched.n1936 = select i1 %.not, float %sched.n1929, float %sched.n1928, !dbg !11
  %sched.n1937 = bitcast float %sched.n1936 to i32, !dbg !11
  %sched.n1938 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1937), !dbg !11
  %sched.n1939 = bitcast i32 %sched.n1938 to float, !dbg !11
  %sched.n1940 = select i1 %.not, float %sched.n1928, float %sched.n1939, !dbg !11
  %sched.n1941 = select i1 %.not, float %sched.n1939, float %sched.n1929, !dbg !11
  %sched.n1942 = select i1 %.not36, float %sched.n1940, float %sched.n1934, !dbg !11
  %sched.n1943 = bitcast float %sched.n1942 to i32, !dbg !11
  %sched.n1944 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1943), !dbg !11
  %sched.n1945 = bitcast i32 %sched.n1944 to float, !dbg !11
  %sched.n1948 = select i1 %.not36, float %sched.n1941, float %sched.n1935, !dbg !11
  %sched.n1949 = bitcast float %sched.n1948 to i32, !dbg !11
  %sched.n1950 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1949), !dbg !11
  %sched.n1951 = bitcast i32 %sched.n1950 to float, !dbg !11
  %249 = insertelement <4 x float> poison, float %sched.n1934, i64 0, !dbg !11
  %250 = insertelement <4 x float> %249, float %sched.n1935, i64 1, !dbg !11
  %251 = insertelement <4 x float> %250, float %sched.n1945, i64 2, !dbg !11
  %252 = insertelement <4 x float> %251, float %sched.n1951, i64 3, !dbg !11
  %253 = insertelement <4 x float> poison, float %sched.n1945, i64 0, !dbg !11
  %254 = insertelement <4 x float> %253, float %sched.n1951, i64 1, !dbg !11
  %255 = insertelement <4 x float> %254, float %sched.n1940, i64 2, !dbg !11
  %256 = insertelement <4 x float> %255, float %sched.n1941, i64 3, !dbg !11
  %257 = select <4 x i1> %167, <4 x float> %252, <4 x float> %256, !dbg !11
  %sched.n1954 = extractelement <8 x float> %sched.n1843, i64 4, !dbg !11
  %sched.n1955 = extractelement <8 x float> %sched.n1843, i64 5, !dbg !11
  %sched.n1956 = extractelement <8 x float> %sched.n1843, i64 6, !dbg !11
  %sched.n1957 = extractelement <8 x float> %sched.n1843, i64 7, !dbg !11
  %sched.n1958 = select i1 %.not, float %sched.n1955, float %sched.n1954, !dbg !11
  %sched.n1959 = bitcast float %sched.n1958 to i32, !dbg !11
  %sched.n1960 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1959), !dbg !11
  %sched.n1961 = bitcast i32 %sched.n1960 to float, !dbg !11
  %sched.n1962 = select i1 %.not, float %sched.n1954, float %sched.n1961, !dbg !11
  %sched.n1963 = select i1 %.not, float %sched.n1961, float %sched.n1955, !dbg !11
  %sched.n1964 = select i1 %.not, float %sched.n1957, float %sched.n1956, !dbg !11
  %sched.n1965 = bitcast float %sched.n1964 to i32, !dbg !11
  %sched.n1966 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1965), !dbg !11
  %sched.n1967 = bitcast i32 %sched.n1966 to float, !dbg !11
  %sched.n1968 = select i1 %.not, float %sched.n1956, float %sched.n1967, !dbg !11
  %sched.n1969 = select i1 %.not, float %sched.n1967, float %sched.n1957, !dbg !11
  %sched.n1970 = select i1 %.not36, float %sched.n1968, float %sched.n1962, !dbg !11
  %sched.n1971 = bitcast float %sched.n1970 to i32, !dbg !11
  %sched.n1972 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1971), !dbg !11
  %sched.n1973 = bitcast i32 %sched.n1972 to float, !dbg !11
  %sched.n1976 = select i1 %.not36, float %sched.n1969, float %sched.n1963, !dbg !11
  %sched.n1977 = bitcast float %sched.n1976 to i32, !dbg !11
  %sched.n1978 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1977), !dbg !11
  %sched.n1979 = bitcast i32 %sched.n1978 to float, !dbg !11
  %258 = insertelement <4 x float> poison, float %sched.n1962, i64 0, !dbg !11
  %259 = insertelement <4 x float> %258, float %sched.n1963, i64 1, !dbg !11
  %260 = insertelement <4 x float> %259, float %sched.n1973, i64 2, !dbg !11
  %261 = insertelement <4 x float> %260, float %sched.n1979, i64 3, !dbg !11
  %262 = insertelement <4 x float> poison, float %sched.n1973, i64 0, !dbg !11
  %263 = insertelement <4 x float> %262, float %sched.n1979, i64 1, !dbg !11
  %264 = insertelement <4 x float> %263, float %sched.n1968, i64 2, !dbg !11
  %265 = insertelement <4 x float> %264, float %sched.n1969, i64 3, !dbg !11
  %266 = select <4 x i1> %167, <4 x float> %261, <4 x float> %265, !dbg !11
  %sched.n1982 = extractelement <8 x float> %sched.n1872, i64 4, !dbg !11
  %sched.n1983 = extractelement <8 x float> %sched.n1872, i64 5, !dbg !11
  %sched.n1984 = extractelement <8 x float> %sched.n1872, i64 6, !dbg !11
  %sched.n1985 = extractelement <8 x float> %sched.n1872, i64 7, !dbg !11
  %sched.n1986 = select i1 %.not, float %sched.n1983, float %sched.n1982, !dbg !11
  %sched.n1987 = bitcast float %sched.n1986 to i32, !dbg !11
  %sched.n1988 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1987), !dbg !11
  %sched.n1989 = bitcast i32 %sched.n1988 to float, !dbg !11
  %sched.n1990 = select i1 %.not, float %sched.n1982, float %sched.n1989, !dbg !11
  %sched.n1991 = select i1 %.not, float %sched.n1989, float %sched.n1983, !dbg !11
  %sched.n1992 = select i1 %.not, float %sched.n1985, float %sched.n1984, !dbg !11
  %sched.n1993 = bitcast float %sched.n1992 to i32, !dbg !11
  %sched.n1994 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1993), !dbg !11
  %sched.n1995 = bitcast i32 %sched.n1994 to float, !dbg !11
  %sched.n1996 = select i1 %.not, float %sched.n1984, float %sched.n1995, !dbg !11
  %sched.n1997 = select i1 %.not, float %sched.n1995, float %sched.n1985, !dbg !11
  %sched.n1998 = select i1 %.not36, float %sched.n1996, float %sched.n1990, !dbg !11
  %sched.n1999 = bitcast float %sched.n1998 to i32, !dbg !11
  %sched.n2000 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1999), !dbg !11
  %sched.n2001 = bitcast i32 %sched.n2000 to float, !dbg !11
  %sched.n2004 = select i1 %.not36, float %sched.n1997, float %sched.n1991, !dbg !11
  %sched.n2005 = bitcast float %sched.n2004 to i32, !dbg !11
  %sched.n2006 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n2005), !dbg !11
  %sched.n2007 = bitcast i32 %sched.n2006 to float, !dbg !11
  %267 = insertelement <4 x float> poison, float %sched.n1990, i64 0, !dbg !11
  %268 = insertelement <4 x float> %267, float %sched.n1991, i64 1, !dbg !11
  %269 = insertelement <4 x float> %268, float %sched.n2001, i64 2, !dbg !11
  %270 = insertelement <4 x float> %269, float %sched.n2007, i64 3, !dbg !11
  %271 = insertelement <4 x float> poison, float %sched.n2001, i64 0, !dbg !11
  %272 = insertelement <4 x float> %271, float %sched.n2007, i64 1, !dbg !11
  %273 = insertelement <4 x float> %272, float %sched.n1996, i64 2, !dbg !11
  %274 = insertelement <4 x float> %273, float %sched.n1997, i64 3, !dbg !11
  %275 = select <4 x i1> %167, <4 x float> %270, <4 x float> %274, !dbg !11
  %sched.n2010 = extractelement <8 x float> %sched.n1901, i64 4, !dbg !11
  %sched.n2011 = extractelement <8 x float> %sched.n1901, i64 5, !dbg !11
  %sched.n2012 = extractelement <8 x float> %sched.n1901, i64 6, !dbg !11
  %sched.n2013 = extractelement <8 x float> %sched.n1901, i64 7, !dbg !11
  %sched.n2014 = select i1 %.not, float %sched.n2011, float %sched.n2010, !dbg !11
  %sched.n2015 = bitcast float %sched.n2014 to i32, !dbg !11
  %sched.n2016 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n2015), !dbg !11
  %sched.n2017 = bitcast i32 %sched.n2016 to float, !dbg !11
  %sched.n2018 = select i1 %.not, float %sched.n2010, float %sched.n2017, !dbg !11
  %sched.n2019 = select i1 %.not, float %sched.n2017, float %sched.n2011, !dbg !11
  %sched.n2020 = select i1 %.not, float %sched.n2013, float %sched.n2012, !dbg !11
  %sched.n2021 = bitcast float %sched.n2020 to i32, !dbg !11
  %sched.n2022 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n2021), !dbg !11
  %sched.n2023 = bitcast i32 %sched.n2022 to float, !dbg !11
  %sched.n2024 = select i1 %.not, float %sched.n2012, float %sched.n2023, !dbg !11
  %sched.n2025 = select i1 %.not, float %sched.n2023, float %sched.n2013, !dbg !11
  %sched.n2026 = select i1 %.not36, float %sched.n2024, float %sched.n2018, !dbg !11
  %sched.n2027 = bitcast float %sched.n2026 to i32, !dbg !11
  %sched.n2028 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n2027), !dbg !11
  %sched.n2029 = select i1 %.not36, float %sched.n2025, float %sched.n2019, !dbg !11
  %sched.n2030 = bitcast float %sched.n2029 to i32, !dbg !11
  %sched.n2031 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n2030), !dbg !11
  %sched.n2033 = bitcast i32 %sched.n2028 to float, !dbg !11
  %sched.n2036 = bitcast i32 %sched.n2031 to float, !dbg !11
  %276 = insertelement <4 x float> poison, float %sched.n2018, i64 0, !dbg !11
  %277 = insertelement <4 x float> %276, float %sched.n2019, i64 1, !dbg !11
  %278 = insertelement <4 x float> %277, float %sched.n2033, i64 2, !dbg !11
  %279 = insertelement <4 x float> %278, float %sched.n2036, i64 3, !dbg !11
  %280 = insertelement <4 x float> poison, float %sched.n2033, i64 0, !dbg !11
  %281 = insertelement <4 x float> %280, float %sched.n2036, i64 1, !dbg !11
  %282 = insertelement <4 x float> %281, float %sched.n2024, i64 2, !dbg !11
  %283 = insertelement <4 x float> %282, float %sched.n2025, i64 3, !dbg !11
  %284 = select <4 x i1> %167, <4 x float> %279, <4 x float> %283, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %sched.n1925, !dbg !25
  %normalized = fmul <4 x float> %row.reciprocal, %284, !dbg !25
  %sched.n2043 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !26
  %sched.n2044 = extractelement <4 x half> %sched.n2043, i64 3, !dbg !26
  %sched.n2045 = extractelement <4 x half> %sched.n2043, i64 2, !dbg !26
  %sched.n2046 = extractelement <4 x half> %sched.n2043, i64 1, !dbg !26
  %sched.n2047 = extractelement <4 x half> %sched.n2043, i64 0, !dbg !26
  %sched.n2048 = bitcast i32 %sched.n1920 to float, !dbg !11
  %sched.n2051 = bitcast i32 %sched.n1923 to float, !dbg !11
  %285 = insertelement <4 x float> poison, float %sched.n1910, i64 0, !dbg !11
  %286 = insertelement <4 x float> %285, float %sched.n1911, i64 1, !dbg !11
  %287 = insertelement <4 x float> %286, float %sched.n2048, i64 2, !dbg !11
  %288 = insertelement <4 x float> %287, float %sched.n2051, i64 3, !dbg !11
  %289 = insertelement <4 x float> poison, float %sched.n2048, i64 0, !dbg !11
  %290 = insertelement <4 x float> %289, float %sched.n2051, i64 1, !dbg !11
  %291 = insertelement <4 x float> %290, float %sched.n1916, i64 2, !dbg !11
  %292 = insertelement <4 x float> %291, float %sched.n1917, i64 3, !dbg !11
  %293 = select <4 x i1> %167, <4 x float> %288, <4 x float> %292, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %293, !dbg !25
  %sched.n2058 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !26
  %sched.n2059 = extractelement <4 x half> %sched.n2058, i64 3, !dbg !26
  %sched.n2060 = extractelement <4 x half> %sched.n2058, i64 2, !dbg !26
  %sched.n2061 = extractelement <4 x half> %sched.n2058, i64 1, !dbg !26
  %sched.n2062 = extractelement <4 x half> %sched.n2058, i64 0, !dbg !26
  %sched.n2063 = bitcast i32 %sched.n1810 to float, !dbg !11
  %sched.n2066 = bitcast i32 %sched.n1813 to float, !dbg !11
  %294 = insertelement <4 x float> poison, float %sched.n1800, i64 0, !dbg !11
  %295 = insertelement <4 x float> %294, float %sched.n1801, i64 1, !dbg !11
  %296 = insertelement <4 x float> %295, float %sched.n2063, i64 2, !dbg !11
  %297 = insertelement <4 x float> %296, float %sched.n2066, i64 3, !dbg !11
  %298 = insertelement <4 x float> poison, float %sched.n2063, i64 0, !dbg !11
  %299 = insertelement <4 x float> %298, float %sched.n2066, i64 1, !dbg !11
  %300 = insertelement <4 x float> %299, float %sched.n1806, i64 2, !dbg !11
  %301 = insertelement <4 x float> %300, float %sched.n1807, i64 3, !dbg !11
  %302 = select <4 x i1> %167, <4 x float> %297, <4 x float> %301, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %sched.n1707, !dbg !25
  %normalized106 = fmul <4 x float> %row.reciprocal105, %302, !dbg !25
  %sched.n2073 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !26
  %sched.n2074 = extractelement <4 x half> %sched.n2073, i64 3, !dbg !26
  %sched.n2075 = extractelement <4 x half> %sched.n2073, i64 2, !dbg !26
  %sched.n2076 = extractelement <4 x half> %sched.n2073, i64 1, !dbg !26
  %sched.n2077 = extractelement <4 x half> %sched.n2073, i64 0, !dbg !26
  %sched.n2078 = bitcast i32 %sched.n1702 to float, !dbg !11
  %sched.n2081 = bitcast i32 %sched.n1705 to float, !dbg !11
  %303 = insertelement <4 x float> poison, float %sched.n1692, i64 0, !dbg !11
  %304 = insertelement <4 x float> %303, float %sched.n1693, i64 1, !dbg !11
  %305 = insertelement <4 x float> %304, float %sched.n2078, i64 2, !dbg !11
  %306 = insertelement <4 x float> %305, float %sched.n2081, i64 3, !dbg !11
  %307 = insertelement <4 x float> poison, float %sched.n2078, i64 0, !dbg !11
  %308 = insertelement <4 x float> %307, float %sched.n2081, i64 1, !dbg !11
  %309 = insertelement <4 x float> %308, float %sched.n1698, i64 2, !dbg !11
  %310 = insertelement <4 x float> %309, float %sched.n1699, i64 3, !dbg !11
  %311 = select <4 x i1> %167, <4 x float> %306, <4 x float> %310, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %311, !dbg !25
  %sched.n2088 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !26
  %sched.n2089 = extractelement <4 x half> %sched.n2088, i64 3, !dbg !26
  %sched.n2090 = extractelement <4 x half> %sched.n2088, i64 2, !dbg !26
  %sched.n2091 = extractelement <4 x half> %sched.n2088, i64 1, !dbg !26
  %sched.n2092 = extractelement <4 x half> %sched.n2088, i64 0, !dbg !26
  %address.thread914 = tail call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #8, !dbg !27, !range !9, !frisk.register_lifetimes !5
  %address.local915 = zext nneg i32 %address.thread914 to i64, !dbg !8
  %312 = lshr i64 %address.local915, 2, !dbg !27
  %address.local918 = and i64 %312, 48, !dbg !27
  %address.local920 = and i64 %address.local915, 240, !dbg !27
  %address.local919 = add i64 %sched.n27, %address.local915, !dbg !27
  %address.local921 = sub i64 %address.local919, %address.local920, !dbg !27
  %address.local922 = add i64 %address.local921, %address.local918, !dbg !27
  %313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %sched.n28, !dbg !27
  %address.local925.idx = shl nuw nsw i64 %address.local922, 8, !dbg !27
  %address.local925 = getelementptr inbounds nuw i8, ptr addrspace(1) %313, i64 %address.local925.idx, !dbg !27
  %address.local927 = and i64 %312, 12, !dbg !10
  %address.local928 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local927, !dbg !27
  store half %sched.n2092, ptr addrspace(1) %address.local928, align 2, !dbg !27
  %address.local930 = or disjoint i64 %address.local927, 1, !dbg !27
  %address.local931 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local930, !dbg !27
  store half %sched.n2091, ptr addrspace(1) %address.local931, align 2, !dbg !27
  %address.local933 = or disjoint i64 %address.local927, 2, !dbg !27
  %address.local934 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local933, !dbg !27
  store half %sched.n2090, ptr addrspace(1) %address.local934, align 2, !dbg !27
  %address.local936 = or disjoint i64 %address.local927, 3, !dbg !27
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local936, !dbg !27
  store half %sched.n2089, ptr addrspace(1) %address.local937, align 2, !dbg !27
  %address.local939 = or disjoint i64 %address.local927, 16, !dbg !27
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local939, !dbg !27
  store half %sched.n2077, ptr addrspace(1) %address.local940, align 2, !dbg !27
  %address.local942 = or disjoint i64 %address.local927, 17, !dbg !27
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local942, !dbg !27
  store half %sched.n2076, ptr addrspace(1) %address.local943, align 2, !dbg !27
  %address.local945 = or disjoint i64 %address.local927, 18, !dbg !27
  %address.local946 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local945, !dbg !27
  store half %sched.n2075, ptr addrspace(1) %address.local946, align 2, !dbg !27
  %address.local948 = or disjoint i64 %address.local927, 19, !dbg !27
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local948, !dbg !27
  store half %sched.n2074, ptr addrspace(1) %address.local949, align 2, !dbg !27
  %address.local953 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local925, i64 16384, !dbg !27
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local927, !dbg !27
  store half %sched.n2062, ptr addrspace(1) %address.local954, align 2, !dbg !27
  %address.local956 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local930, !dbg !27
  store half %sched.n2061, ptr addrspace(1) %address.local956, align 2, !dbg !27
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local933, !dbg !27
  store half %sched.n2060, ptr addrspace(1) %address.local958, align 2, !dbg !27
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local936, !dbg !27
  store half %sched.n2059, ptr addrspace(1) %address.local960, align 2, !dbg !27
  %address.local962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local939, !dbg !27
  store half %sched.n2047, ptr addrspace(1) %address.local962, align 2, !dbg !27
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local942, !dbg !27
  store half %sched.n2046, ptr addrspace(1) %address.local964, align 2, !dbg !27
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local945, !dbg !27
  store half %sched.n2045, ptr addrspace(1) %address.local966, align 2, !dbg !27
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local948, !dbg !27
  store half %sched.n2044, ptr addrspace(1) %address.local968, align 2, !dbg !27
  %normalized109 = fmul <4 x float> %row.reciprocal, %275, !dbg !25
  %sched.n2102 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !26
  %sched.n2103 = extractelement <4 x half> %sched.n2102, i64 3, !dbg !26
  %sched.n2104 = extractelement <4 x half> %sched.n2102, i64 2, !dbg !26
  %sched.n2105 = extractelement <4 x half> %sched.n2102, i64 1, !dbg !26
  %sched.n2106 = extractelement <4 x half> %sched.n2102, i64 0, !dbg !26
  %normalized110 = fmul <4 x float> %row.reciprocal, %248, !dbg !25
  %sched.n2111 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !26
  %sched.n2112 = extractelement <4 x half> %sched.n2111, i64 3, !dbg !26
  %sched.n2113 = extractelement <4 x half> %sched.n2111, i64 2, !dbg !26
  %sched.n2114 = extractelement <4 x half> %sched.n2111, i64 1, !dbg !26
  %sched.n2115 = extractelement <4 x half> %sched.n2111, i64 0, !dbg !26
  %normalized112 = fmul <4 x float> %row.reciprocal105, %221, !dbg !25
  %sched.n2120 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !26
  %sched.n2121 = extractelement <4 x half> %sched.n2120, i64 3, !dbg !26
  %sched.n2122 = extractelement <4 x half> %sched.n2120, i64 2, !dbg !26
  %sched.n2123 = extractelement <4 x half> %sched.n2120, i64 1, !dbg !26
  %sched.n2124 = extractelement <4 x half> %sched.n2120, i64 0, !dbg !26
  %normalized113 = fmul <4 x float> %row.reciprocal105, %194, !dbg !25
  %sched.n2129 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !26
  %sched.n2130 = extractelement <4 x half> %sched.n2129, i64 3, !dbg !26
  %sched.n2131 = extractelement <4 x half> %sched.n2129, i64 2, !dbg !26
  %sched.n2132 = extractelement <4 x half> %sched.n2129, i64 1, !dbg !26
  %sched.n2133 = extractelement <4 x half> %sched.n2129, i64 0, !dbg !26
  %address.local983 = or disjoint i64 %address.local927, 32, !dbg !27
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local983, !dbg !27
  store half %sched.n2133, ptr addrspace(1) %address.local984, align 2, !dbg !27
  %address.local986 = or disjoint i64 %address.local927, 33, !dbg !27
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local986, !dbg !27
  store half %sched.n2132, ptr addrspace(1) %address.local987, align 2, !dbg !27
  %address.local989 = or disjoint i64 %address.local927, 34, !dbg !27
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local989, !dbg !27
  store half %sched.n2131, ptr addrspace(1) %address.local990, align 2, !dbg !27
  %address.local992 = or disjoint i64 %address.local927, 35, !dbg !27
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local992, !dbg !27
  store half %sched.n2130, ptr addrspace(1) %address.local993, align 2, !dbg !27
  %address.local995 = or disjoint i64 %address.local927, 48, !dbg !27
  %address.local996 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local995, !dbg !27
  store half %sched.n2124, ptr addrspace(1) %address.local996, align 2, !dbg !27
  %address.local998 = or disjoint i64 %address.local927, 49, !dbg !27
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local998, !dbg !27
  store half %sched.n2123, ptr addrspace(1) %address.local999, align 2, !dbg !27
  %address.local1001 = or disjoint i64 %address.local927, 50, !dbg !27
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1001, !dbg !27
  store half %sched.n2122, ptr addrspace(1) %address.local1002, align 2, !dbg !27
  %address.local1004 = or i64 %312, 51, !dbg !27
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1004, !dbg !27
  store half %sched.n2121, ptr addrspace(1) %address.local1005, align 2, !dbg !27
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local983, !dbg !27
  store half %sched.n2115, ptr addrspace(1) %address.local1010, align 2, !dbg !27
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local986, !dbg !27
  store half %sched.n2114, ptr addrspace(1) %address.local1012, align 2, !dbg !27
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local989, !dbg !27
  store half %sched.n2113, ptr addrspace(1) %address.local1014, align 2, !dbg !27
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local992, !dbg !27
  store half %sched.n2112, ptr addrspace(1) %address.local1016, align 2, !dbg !27
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local995, !dbg !27
  store half %sched.n2106, ptr addrspace(1) %address.local1018, align 2, !dbg !27
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local998, !dbg !27
  store half %sched.n2105, ptr addrspace(1) %address.local1020, align 2, !dbg !27
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1001, !dbg !27
  store half %sched.n2104, ptr addrspace(1) %address.local1022, align 2, !dbg !27
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1004, !dbg !27
  store half %sched.n2103, ptr addrspace(1) %address.local1024, align 2, !dbg !27
  %normalized115 = fmul <4 x float> %row.reciprocal, %266, !dbg !25
  %sched.n2141 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !26
  %sched.n2142 = extractelement <4 x half> %sched.n2141, i64 3, !dbg !26
  %sched.n2143 = extractelement <4 x half> %sched.n2141, i64 2, !dbg !26
  %sched.n2144 = extractelement <4 x half> %sched.n2141, i64 1, !dbg !26
  %sched.n2145 = extractelement <4 x half> %sched.n2141, i64 0, !dbg !26
  %normalized116 = fmul <4 x float> %row.reciprocal, %239, !dbg !25
  %sched.n2150 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !26
  %sched.n2151 = extractelement <4 x half> %sched.n2150, i64 3, !dbg !26
  %sched.n2152 = extractelement <4 x half> %sched.n2150, i64 2, !dbg !26
  %sched.n2153 = extractelement <4 x half> %sched.n2150, i64 1, !dbg !26
  %sched.n2154 = extractelement <4 x half> %sched.n2150, i64 0, !dbg !26
  %normalized118 = fmul <4 x float> %row.reciprocal105, %212, !dbg !25
  %sched.n2159 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !26
  %sched.n2160 = extractelement <4 x half> %sched.n2159, i64 3, !dbg !26
  %sched.n2161 = extractelement <4 x half> %sched.n2159, i64 2, !dbg !26
  %sched.n2162 = extractelement <4 x half> %sched.n2159, i64 1, !dbg !26
  %sched.n2163 = extractelement <4 x half> %sched.n2159, i64 0, !dbg !26
  %normalized119 = fmul <4 x float> %row.reciprocal105, %185, !dbg !25
  %sched.n2168 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !26
  %sched.n2169 = extractelement <4 x half> %sched.n2168, i64 3, !dbg !26
  %sched.n2170 = extractelement <4 x half> %sched.n2168, i64 2, !dbg !26
  %sched.n2171 = extractelement <4 x half> %sched.n2168, i64 1, !dbg !26
  %sched.n2172 = extractelement <4 x half> %sched.n2168, i64 0, !dbg !26
  %address.local1039 = or disjoint i64 %address.local927, 64, !dbg !27
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1039, !dbg !27
  store half %sched.n2172, ptr addrspace(1) %address.local1040, align 2, !dbg !27
  %address.local1042 = or disjoint i64 %address.local927, 65, !dbg !27
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1042, !dbg !27
  store half %sched.n2171, ptr addrspace(1) %address.local1043, align 2, !dbg !27
  %address.local1045 = or disjoint i64 %address.local927, 66, !dbg !27
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1045, !dbg !27
  store half %sched.n2170, ptr addrspace(1) %address.local1046, align 2, !dbg !27
  %address.local1048 = or disjoint i64 %address.local927, 67, !dbg !27
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1048, !dbg !27
  store half %sched.n2169, ptr addrspace(1) %address.local1049, align 2, !dbg !27
  %address.local1051 = or disjoint i64 %address.local927, 80, !dbg !27
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1051, !dbg !27
  store half %sched.n2163, ptr addrspace(1) %address.local1052, align 2, !dbg !27
  %address.local1054 = or disjoint i64 %address.local927, 81, !dbg !27
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1054, !dbg !27
  store half %sched.n2162, ptr addrspace(1) %address.local1055, align 2, !dbg !27
  %address.local1057 = or disjoint i64 %address.local927, 82, !dbg !27
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1057, !dbg !27
  store half %sched.n2161, ptr addrspace(1) %address.local1058, align 2, !dbg !27
  %address.local1060 = or disjoint i64 %address.local927, 83, !dbg !27
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1060, !dbg !27
  store half %sched.n2160, ptr addrspace(1) %address.local1061, align 2, !dbg !27
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1039, !dbg !27
  store half %sched.n2154, ptr addrspace(1) %address.local1066, align 2, !dbg !27
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1042, !dbg !27
  store half %sched.n2153, ptr addrspace(1) %address.local1068, align 2, !dbg !27
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1045, !dbg !27
  store half %sched.n2152, ptr addrspace(1) %address.local1070, align 2, !dbg !27
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1048, !dbg !27
  store half %sched.n2151, ptr addrspace(1) %address.local1072, align 2, !dbg !27
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1051, !dbg !27
  store half %sched.n2145, ptr addrspace(1) %address.local1074, align 2, !dbg !27
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1054, !dbg !27
  store half %sched.n2144, ptr addrspace(1) %address.local1076, align 2, !dbg !27
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1057, !dbg !27
  store half %sched.n2143, ptr addrspace(1) %address.local1078, align 2, !dbg !27
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1060, !dbg !27
  store half %sched.n2142, ptr addrspace(1) %address.local1080, align 2, !dbg !27
  %normalized121 = fmul <4 x float> %row.reciprocal, %257, !dbg !25
  %sched.n2180 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !26
  %sched.n2181 = extractelement <4 x half> %sched.n2180, i64 3, !dbg !26
  %sched.n2182 = extractelement <4 x half> %sched.n2180, i64 2, !dbg !26
  %sched.n2183 = extractelement <4 x half> %sched.n2180, i64 1, !dbg !26
  %sched.n2184 = extractelement <4 x half> %sched.n2180, i64 0, !dbg !26
  %normalized122 = fmul <4 x float> %row.reciprocal, %230, !dbg !25
  %sched.n2189 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !26
  %sched.n2190 = extractelement <4 x half> %sched.n2189, i64 3, !dbg !26
  %sched.n2191 = extractelement <4 x half> %sched.n2189, i64 2, !dbg !26
  %sched.n2192 = extractelement <4 x half> %sched.n2189, i64 1, !dbg !26
  %sched.n2193 = extractelement <4 x half> %sched.n2189, i64 0, !dbg !26
  %normalized124 = fmul <4 x float> %row.reciprocal105, %203, !dbg !25
  %sched.n2198 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !26
  %sched.n2199 = extractelement <4 x half> %sched.n2198, i64 3, !dbg !26
  %sched.n2200 = extractelement <4 x half> %sched.n2198, i64 2, !dbg !26
  %sched.n2201 = extractelement <4 x half> %sched.n2198, i64 1, !dbg !26
  %sched.n2202 = extractelement <4 x half> %sched.n2198, i64 0, !dbg !26
  %normalized125 = fmul <4 x float> %row.reciprocal105, %176, !dbg !25
  %sched.n2207 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !26
  %sched.n2208 = extractelement <4 x half> %sched.n2207, i64 3, !dbg !26
  %sched.n2209 = extractelement <4 x half> %sched.n2207, i64 2, !dbg !26
  %sched.n2210 = extractelement <4 x half> %sched.n2207, i64 1, !dbg !26
  %sched.n2211 = extractelement <4 x half> %sched.n2207, i64 0, !dbg !26
  %address.local1095 = or disjoint i64 %address.local927, 96, !dbg !27
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1095, !dbg !27
  store half %sched.n2211, ptr addrspace(1) %address.local1096, align 2, !dbg !27
  %address.local1098 = or disjoint i64 %address.local927, 97, !dbg !27
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1098, !dbg !27
  store half %sched.n2210, ptr addrspace(1) %address.local1099, align 2, !dbg !27
  %address.local1101 = or disjoint i64 %address.local927, 98, !dbg !27
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1101, !dbg !27
  store half %sched.n2209, ptr addrspace(1) %address.local1102, align 2, !dbg !27
  %address.local1104 = or disjoint i64 %address.local927, 99, !dbg !27
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1104, !dbg !27
  store half %sched.n2208, ptr addrspace(1) %address.local1105, align 2, !dbg !27
  %address.local1107 = or disjoint i64 %address.local927, 112, !dbg !27
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1107, !dbg !27
  store half %sched.n2202, ptr addrspace(1) %address.local1108, align 2, !dbg !27
  %address.local1110 = or disjoint i64 %address.local927, 113, !dbg !27
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1110, !dbg !27
  store half %sched.n2201, ptr addrspace(1) %address.local1111, align 2, !dbg !27
  %address.local1113 = or disjoint i64 %address.local927, 114, !dbg !27
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1113, !dbg !27
  store half %sched.n2200, ptr addrspace(1) %address.local1114, align 2, !dbg !27
  %address.local1116 = or i64 %312, 115, !dbg !27
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1116, !dbg !27
  store half %sched.n2199, ptr addrspace(1) %address.local1117, align 2, !dbg !27
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1095, !dbg !27
  store half %sched.n2193, ptr addrspace(1) %address.local1122, align 2, !dbg !27
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1098, !dbg !27
  store half %sched.n2192, ptr addrspace(1) %address.local1124, align 2, !dbg !27
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1101, !dbg !27
  store half %sched.n2191, ptr addrspace(1) %address.local1126, align 2, !dbg !27
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1104, !dbg !27
  store half %sched.n2190, ptr addrspace(1) %address.local1128, align 2, !dbg !27
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1107, !dbg !27
  store half %sched.n2184, ptr addrspace(1) %address.local1130, align 2, !dbg !27
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1110, !dbg !27
  store half %sched.n2183, ptr addrspace(1) %address.local1132, align 2, !dbg !27
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1113, !dbg !27
  store half %sched.n2182, ptr addrspace(1) %address.local1134, align 2, !dbg !27
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1116, !dbg !27
  store half %sched.n2181, ptr addrspace(1) %address.local1136, align 2, !dbg !27
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

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #4

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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
!1 = !DIFile(filename: "nohint_unrolled.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 22, type: !4, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 22, column: 3, scope: !3)
!8 = !DILocation(line: 66, column: 22, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 68, column: 24, scope: !3)
!11 = !DILocation(line: 56, column: 24, scope: !3)
!12 = !DILocation(line: 73, column: 19, scope: !3)
!13 = !DILocation(line: 0, scope: !3)
!14 = !DILocation(line: 69, column: 26, scope: !3)
!15 = !DILocation(line: 67, column: 22, scope: !3)
!16 = !DILocation(line: 76, column: 23, scope: !3)
!17 = !DILocation(line: 92, column: 20, scope: !3)
!18 = !DILocation(line: 77, column: 26, scope: !3)
!19 = !DILocation(line: 84, column: 21, scope: !3)
!20 = !DILocation(line: 85, column: 22, scope: !3)
!21 = !DILocation(line: 87, column: 16, scope: !3)
!22 = !DILocation(line: 86, column: 20, scope: !3)
!23 = !DILocation(line: 93, column: 23, scope: !3)
!24 = !DILocation(line: 94, column: 20, scope: !3)
!25 = !DILocation(line: 97, column: 23, scope: !3)
!26 = !DILocation(line: 98, column: 19, scope: !3)
!27 = !DILocation(line: 99, column: 9, scope: !3)
