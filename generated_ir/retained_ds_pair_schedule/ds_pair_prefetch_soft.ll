; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !32 {
  %sched.n5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %sched.n6 = sext i32 %sched.n5 to i64, !dbg !5
  %sched.n7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %sched.n8 = sext i32 %sched.n7 to i64, !dbg !5
  %sched.n9 = shl nsw i64 %sched.n8, 5, !dbg !5
  %sched.n10 = add nsw i64 %sched.n9, %sched.n6, !dbg !5
  %sched.n11 = lshr i64 %sched.n10, 5, !dbg !5
  %sched.n12 = and i64 %sched.n11, 576460752303423480, !dbg !5
  %sched.n13 = and i64 %sched.n6, 7, !dbg !5
  %sched.n14 = or disjoint i64 %sched.n12, %sched.n13, !dbg !5
  %sched.n15 = sub nsw i64 31, %sched.n14, !dbg !5
  %sched.n16 = lshr i64 %sched.n10, 3, !dbg !5
  %sched.n17 = icmp ugt i64 %sched.n10, 1023, !dbg !8
  %sched.n18 = add nsw i64 %sched.n14, -32, !dbg !8
  %sched.n19 = select i1 %sched.n17, i64 %sched.n18, i64 %sched.n15, !dbg !8
  %sched.n20 = sdiv i64 %sched.n19, 32, !dbg !8
  %sched.n21 = sext i1 %sched.n17 to i64, !dbg !8
  %sched.n22 = xor i64 %sched.n20, %sched.n21, !dbg !8
  %sched.n23 = add nsw i64 %sched.n16, %sched.n22, !dbg !8
  %sched.n24 = and i64 %sched.n23, 31, !dbg !8
  %sched.n25 = shl nsw i64 %sched.n15, 7, !dbg !8
  %sched.n26 = mul nsw i64 %sched.n22, -4096, !dbg !8
  %sched.n27 = add i64 %sched.n25, %sched.n26, !dbg !8
  %sched.n28 = shl nuw nsw i64 %sched.n24, 19, !dbg !8
  %sched.n29 = shl nsw i64 %sched.n27, 7, !dbg !8
  %sched.n30 = add i64 %sched.n28, %sched.n29, !dbg !8
  %sched.n31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !34
  %sched.n32 = zext nneg i32 %sched.n31 to i64, !dbg !8
  %sched.n33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %sched.n30, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #9, !dbg !8, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local336 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local337 = and i64 %address.local336, 1536, !dbg !8
  %address.local338 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local337, !dbg !8
  %address.local339 = ptrtoint ptr addrspace(3) %address.local338 to i64, !dbg !8
  %address.local340 = trunc i64 %address.local339 to i32, !dbg !8
  %sched.n34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !8
  %sched.n35 = zext i32 %sched.n34 to i64, !dbg !8
  %sched.n36 = inttoptr i64 %sched.n35 to ptr addrspace(3), !dbg !8
  %address.local341 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local342 = and i64 %address.local341, 16256, !dbg !8
  %address.local343 = lshr i64 %address.local, 4, !dbg !8
  %address.local344 = and i64 %address.local343, 8, !dbg !8
  %address.local345 = or disjoint i64 %address.local342, %address.local344, !dbg !8
  %address.local346 = getelementptr [2 x i8], ptr addrspace(1) %sched.n33, i64 %address.local345, !dbg !8
  %sched.n37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %sched.n38 = ptrtoint ptr addrspace(1) %address.local346 to i64, !dbg !8
  %sched.n39 = sub i64 %sched.n38, %sched.n37, !dbg !8
  %dma.buffer.offset1597 = trunc i64 %sched.n39 to i32, !dbg !8
  %sched.n40 = trunc i64 %sched.n37 to i32, !dbg !8
  %sched.n41 = insertelement <4 x i32> poison, i32 %sched.n40, i64 0, !dbg !8
  %sched.n42 = lshr i64 %sched.n37, 32, !dbg !8
  %sched.n43 = trunc i64 %sched.n42 to i32, !dbg !8
  %sched.n44 = insertelement <4 x i32> %sched.n41, i32 %sched.n43, i64 1, !dbg !8
  %sched.n45 = insertelement <4 x i32> %sched.n44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1598 = insertelement <4 x i32> %sched.n45, i32 131072, i64 3, !dbg !8
  %sched.n46 = ptrtoint ptr addrspace(3) %sched.n36 to i32, !dbg !8
  %dma.buffer.lds1599 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n46), !dbg !8
  %sched.n47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1599, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1598) #10, !dbg !8
  %address.local347 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local348 = shl nuw nsw i64 %address.local347, 3, !dbg !8
  %address.local349 = and i64 %address.local348, 3584, !dbg !8
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local349, !dbg !8
  %address.local351 = ptrtoint ptr addrspace(3) %address.local350 to i64, !dbg !8
  %address.local352 = trunc i64 %address.local351 to i32, !dbg !8
  %sched.n48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !8
  %sched.n49 = zext i32 %sched.n48 to i64, !dbg !8
  %sched.n50 = inttoptr i64 %sched.n49 to ptr addrspace(3), !dbg !8
  %address.local353 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local354 = and i64 %address.local353, 16256, !dbg !8
  %address.local355 = lshr i64 %address.local347, 4, !dbg !8
  %address.local356 = and i64 %address.local355, 24, !dbg !8
  %address.local357 = or disjoint i64 %address.local354, %address.local356, !dbg !8
  %address.local358 = getelementptr [2 x i8], ptr addrspace(1) %sched.n33, i64 %address.local357, !dbg !8
  %sched.n51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %sched.n52 = ptrtoint ptr addrspace(1) %address.local358 to i64, !dbg !8
  %sched.n53 = sub i64 %sched.n52, %sched.n51, !dbg !8
  %dma.buffer.offset1594 = trunc i64 %sched.n53 to i32, !dbg !8
  %sched.n54 = trunc i64 %sched.n51 to i32, !dbg !8
  %sched.n55 = insertelement <4 x i32> poison, i32 %sched.n54, i64 0, !dbg !8
  %sched.n56 = lshr i64 %sched.n51, 32, !dbg !8
  %sched.n57 = trunc i64 %sched.n56 to i32, !dbg !8
  %sched.n58 = insertelement <4 x i32> %sched.n55, i32 %sched.n57, i64 1, !dbg !8
  %sched.n59 = insertelement <4 x i32> %sched.n58, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1595 = insertelement <4 x i32> %sched.n59, i32 131072, i64 3, !dbg !8
  %sched.n60 = ptrtoint ptr addrspace(3) %sched.n50 to i32, !dbg !8
  %dma.buffer.lds1596 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n60), !dbg !8
  %sched.n61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1596, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1595) #10, !dbg !8
  %sched.n62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %sched.n63 = sext i32 %sched.n62 to i64, !dbg !5
  %sched.n64 = shl i64 %sched.n15, 7, !dbg !10
  %sched.n65 = lshr i64 %sched.n32, 6, !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local359 = lshr i32 %address.thread, 2, !dbg !11
  %address.local360 = and i32 %address.local359, 4, !dbg !11
  %address.local361 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local362 = and i32 %address.local361, 384, !dbg !11
  %address.local363 = or disjoint i32 %address.local360, %address.local362, !dbg !11
  %address.local364 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local365 = and i32 %address.local364, 120, !dbg !11
  %address.local366 = or disjoint i32 %address.local363, %address.local365, !dbg !11
  %address.local367 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local368 = and i32 %address.local367, 1024, !dbg !11
  %address.local369 = or disjoint i32 %address.local366, %address.local368, !dbg !11
  %address.local370 = zext nneg i32 %address.local369 to i64, !dbg !11
  %address.local371 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local370, !dbg !11
  %sched.n66 = load <4 x half>, ptr addrspace(3) %address.local371, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local371, i64 4096, !dbg !11
  %sched.n67 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %sched.n68 = fmul <4 x half> %sched.n66, splat (half 0xH3015), !dbg !11
  %lds.const.gep1393 = getelementptr i8, ptr addrspace(3) %address.local371, i64 1024, !dbg !11
  %sched.n69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1393, align 8, !dbg !11
  %sched.n70 = fmul <4 x half> %sched.n67, splat (half 0xH3015), !dbg !11
  %lds.const.gep1394 = getelementptr i8, ptr addrspace(3) %address.local371, i64 5120, !dbg !11
  %sched.n71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1394, align 8, !dbg !11
  %sched.n72 = fmul <4 x half> %sched.n69, splat (half 0xH3015), !dbg !11
  %sched.n73 = fmul <4 x half> %sched.n71, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %sched.n74 = or disjoint i64 %sched.n64, 127, !dbg !14
  %sched.n75 = add i64 %sched.n64, 128, !dbg !14
  %sched.n76 = icmp uge i64 %sched.n75, %sched.n74, !dbg !14
  %sched.n77 = sub nsw i64 32, %sched.n14, !dbg !16
  %sched.n78 = lshr i64 %sched.n32, 2, !dbg !18
  %sched.n79 = shl nuw nsw i64 %sched.n32, 2, !dbg !18
  %sched.n80 = mul nsw i64 %sched.n78, -16, !dbg !18
  %sched.n81 = add nsw i64 %sched.n79, %sched.n80, !dbg !18
  %sched.n82 = and i32 %sched.n31, 1, !dbg !18
  %index.compact279 = zext nneg i32 %sched.n82 to i64, !dbg !18
  %sched.n83 = or disjoint i64 %sched.n81, %index.compact279, !dbg !18
  %sched.n84 = add nuw nsw i64 %sched.n83, 1, !dbg !18
  %sched.n85 = sdiv i64 %sched.n84, 2, !dbg !18
  %sched.n86 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %sched.n30, !dbg !8
  %sched.n87 = getelementptr i8, ptr addrspace(1) %sched.n86, i64 64, !dbg !8
  %sched.n88 = shl i64 %sched.n16, 7, !dbg !17
  %sched.n89 = and i64 %sched.n88, 3968, !dbg !17
  %sched.n90 = shl nsw i64 %sched.n63, 12, !dbg !17
  %sched.n91 = or disjoint i64 %sched.n89, %sched.n90, !dbg !17
  %sched.n92 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %sched.n30, !dbg !8
  %sched.n93 = getelementptr i8, ptr addrspace(1) %sched.n92, i64 128, !dbg !8
  %sched.n94 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %sched.n30, !dbg !8
  %sched.n95 = getelementptr i8, ptr addrspace(1) %sched.n94, i64 192, !dbg !8
  br label %sched.n96, !dbg !12

sched.n96:                                               ; preds = %sched.n847, %4
  %sched.n97 = phi i64 [ %sched.n1580, %sched.n847 ], [ 0, %4 ], !dbg !12
  %sched.n98 = phi [2 x <8 x float>] [ %sched.n1567, %sched.n847 ], [ zeroinitializer, %4 ], !dbg !12
  %sched.n99 = phi [2 x <1 x float>] [ %sched.n1189, %sched.n847 ], [ zeroinitializer, %4 ], !dbg !12
  %sched.n100 = phi [2 x <8 x float>] [ %sched.n1571, %sched.n847 ], [ zeroinitializer, %4 ], !dbg !12
  %sched.n101 = phi [2 x <8 x float>] [ %sched.n1575, %sched.n847 ], [ zeroinitializer, %4 ], !dbg !12
  %sched.n102 = phi [2 x <8 x float>] [ %sched.n1579, %sched.n847 ], [ zeroinitializer, %4 ], !dbg !12
  %sched.n103 = icmp slt i64 %sched.n97, %sched.n77, !dbg !12
  br i1 %sched.n103, label %sched.n104, label %sched.n1581, !dbg !12

sched.n104:                                              ; preds = %sched.n96
  %sched.n105 = shl nsw i64 %sched.n97, 7, !dbg !12
  %sched.n106 = lshr i64 %sched.n97, 12, !dbg !17
  %sched.n107 = add nuw nsw i64 %sched.n106, %sched.n16, !dbg !17
  %sched.n108 = lshr i64 %sched.n97, 5, !dbg !17
  %sched.n109 = and i64 %sched.n108, 127, !dbg !17
  %sched.n110 = shl i64 %sched.n107, 19, !dbg !17
  %sched.n111 = and i64 %sched.n110, 16252928, !dbg !17
  %sched.n112 = shl i64 %sched.n108, 19, !dbg !17
  %sched.n113 = shl i64 %sched.n97, 14, !dbg !17
  %sched.n114 = sub i64 %sched.n113, %sched.n112, !dbg !17
  %sched.n115 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %sched.n111, !dbg !17
  %sched.n116 = getelementptr [2 x i8], ptr addrspace(1) %sched.n115, i64 %sched.n109, !dbg !17
  %sched.n117 = getelementptr [2 x i8], ptr addrspace(1) %sched.n116, i64 %sched.n114, !dbg !17
  %address.thread378 = call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #9, !dbg !17, !range !34, !frisk.register_lifetimes !1
  %address.local379 = zext nneg i32 %address.thread378 to i64, !dbg !8
  %address.local380 = shl nuw nsw i64 %address.local379, 3, !dbg !8
  %address.local381 = and i64 %address.local380, 1536, !dbg !8
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !8
  %address.local383 = ptrtoint ptr addrspace(3) %address.local382 to i64, !dbg !8
  %address.local384 = trunc i64 %address.local383 to i32, !dbg !8
  %sched.n118 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local384), !dbg !17
  %sched.n119 = zext i32 %sched.n118 to i64, !dbg !17
  %sched.n120 = inttoptr i64 %sched.n119 to ptr addrspace(3), !dbg !17
  %address.local385 = lshr i64 %address.local379, 4, !dbg !17
  %address.local386 = and i64 %address.local385, 8, !dbg !17
  %address.local387 = and i64 %address.local379, 123, !dbg !17
  %address.local388 = lshr i64 %address.local379, 2, !dbg !17
  %address.local389 = and i64 %address.local388, 4, !dbg !17
  %address.local390 = add nuw nsw i64 %address.local389, %address.local379, !dbg !17
  %address.local391 = and i64 %address.local390, 4, !dbg !17
  %address.local392 = or disjoint i64 %address.local387, %address.local391, !dbg !17
  %address.local393 = shl nuw nsw i64 %address.local392, 7, !dbg !17
  %address.local394 = or disjoint i64 %address.local386, %address.local393, !dbg !17
  %address.local395 = getelementptr [2 x i8], ptr addrspace(1) %sched.n117, i64 %address.local394, !dbg !17
  %sched.n121 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %sched.n122 = ptrtoint ptr addrspace(1) %address.local395 to i64, !dbg !17
  %sched.n123 = sub i64 %sched.n122, %sched.n121, !dbg !17
  %dma.buffer.offset1591 = trunc i64 %sched.n123 to i32, !dbg !17
  %sched.n124 = trunc i64 %sched.n121 to i32, !dbg !17
  %sched.n125 = insertelement <4 x i32> poison, i32 %sched.n124, i64 0, !dbg !17
  %sched.n126 = lshr i64 %sched.n121, 32, !dbg !17
  %sched.n127 = trunc i64 %sched.n126 to i32, !dbg !17
  %sched.n128 = insertelement <4 x i32> %sched.n125, i32 %sched.n127, i64 1, !dbg !17
  %sched.n129 = insertelement <4 x i32> %sched.n128, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1592 = insertelement <4 x i32> %sched.n129, i32 131072, i64 3, !dbg !17
  %sched.n130 = ptrtoint ptr addrspace(3) %sched.n120 to i32, !dbg !17
  %dma.buffer.lds1593 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n130), !dbg !17
  %sched.n131 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1592) #10, !dbg !17
  %address.local396 = or disjoint i64 %address.local379, 256, !dbg !8
  %address.local397 = shl nuw nsw i64 %address.local396, 3, !dbg !8
  %address.local398 = and i64 %address.local397, 3584, !dbg !8
  %address.local399 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local398, !dbg !8
  %address.local400 = ptrtoint ptr addrspace(3) %address.local399 to i64, !dbg !8
  %address.local401 = trunc i64 %address.local400 to i32, !dbg !8
  %sched.n132 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local401), !dbg !17
  %sched.n133 = zext i32 %sched.n132 to i64, !dbg !17
  %sched.n134 = inttoptr i64 %sched.n133 to ptr addrspace(3), !dbg !17
  %address.local402 = lshr i64 %address.local396, 4, !dbg !17
  %address.local403 = and i64 %address.local402, 24, !dbg !17
  %address.local404 = and i64 %address.local379, 123, !dbg !17
  %address.local405 = lshr i64 %address.local379, 2, !dbg !17
  %address.local406 = and i64 %address.local405, 4, !dbg !17
  %address.local407 = add nuw nsw i64 %address.local406, %address.local379, !dbg !17
  %address.local408 = and i64 %address.local407, 4, !dbg !17
  %address.local409 = or disjoint i64 %address.local404, %address.local408, !dbg !17
  %address.local410 = shl nuw nsw i64 %address.local409, 7, !dbg !17
  %address.local411 = or disjoint i64 %address.local403, %address.local410, !dbg !17
  %address.local412 = getelementptr [2 x i8], ptr addrspace(1) %sched.n117, i64 %address.local411, !dbg !17
  %sched.n135 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %sched.n136 = ptrtoint ptr addrspace(1) %address.local412 to i64, !dbg !17
  %sched.n137 = sub i64 %sched.n136, %sched.n135, !dbg !17
  %dma.buffer.offset1588 = trunc i64 %sched.n137 to i32, !dbg !17
  %sched.n138 = trunc i64 %sched.n135 to i32, !dbg !17
  %sched.n139 = insertelement <4 x i32> poison, i32 %sched.n138, i64 0, !dbg !17
  %sched.n140 = lshr i64 %sched.n135, 32, !dbg !17
  %sched.n141 = trunc i64 %sched.n140 to i32, !dbg !17
  %sched.n142 = insertelement <4 x i32> %sched.n139, i32 %sched.n141, i64 1, !dbg !17
  %sched.n143 = insertelement <4 x i32> %sched.n142, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1589 = insertelement <4 x i32> %sched.n143, i32 131072, i64 3, !dbg !17
  %sched.n144 = ptrtoint ptr addrspace(3) %sched.n134 to i32, !dbg !17
  %dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n144), !dbg !17
  %sched.n145 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #10, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local413 = mul nuw nsw i64 %address.local379, 160, !dbg !18
  %address.local414 = lshr i64 %address.local379, 2, !dbg !18
  %address.local415 = mul nsw i64 %address.local414, -640, !dbg !18
  %address.local416 = add nsw i64 %address.local413, %address.local415, !dbg !18
  %address.local417 = shl nuw nsw i64 %address.local379, 1, !dbg !18
  %address.local418 = and i64 %address.local417, 24, !dbg !18
  %address.local419 = or disjoint i64 %address.local416, %address.local418, !dbg !18
  %address.local420 = shl i64 %address.local379, 62, !dbg !18
  %address.local421 = ashr exact i64 %address.local420, 57, !dbg !18
  %address.local422 = and i64 %address.local421, -64, !dbg !18
  %address.local423 = add nsw i64 %address.local419, %address.local422, !dbg !18
  %address.local424 = lshr i64 %address.local379, 2, !dbg !11
  %address.local425 = and i64 %address.local424, 12, !dbg !11
  %address.local426 = add nsw i64 %address.local423, %address.local425, !dbg !18
  %address.local427 = and i64 %address.local379, 63, !dbg !11
  %address.local428 = lshr i64 %address.local427, 5, !dbg !11
  %address.local429 = mul nuw nsw i64 %address.local428, 1016, !dbg !11
  %address.local430 = add i64 %address.local426, %address.local429, !dbg !18
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local430, !dbg !18
  %sched.n146 = load <4 x half>, ptr addrspace(3) %address.local431, align 8, !dbg !18
  %sched.n147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n146, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1395 = getelementptr i8, ptr addrspace(3) %address.local431, i64 4096, !dbg !18
  %sched.n148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1395, align 8, !dbg !18
  %address.local442 = shl nuw nsw i32 %address.thread378, 1, !dbg !18
  %address.local443 = and i32 %address.local442, 24, !dbg !18
  %address.local444 = shl nuw nsw i32 %address.thread378, 8, !dbg !18
  %address.local445 = and i32 %address.local444, 768, !dbg !18
  %address.local446 = or disjoint i32 %address.local443, %address.local445, !dbg !18
  %address.local447 = shl nuw nsw i32 %address.thread378, 5, !dbg !18
  %address.local448 = and i32 %address.local447, 96, !dbg !18
  %address.local449 = or disjoint i32 %address.local446, %address.local448, !dbg !18
  %address.local450 = zext nneg i32 %address.local449 to i64, !dbg !18
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %sched.n85) #9, !dbg !18, !frisk.register_lifetimes !1
  %address.local451 = lshr i32 %address.thread378, 1, !dbg !18
  %address.local452 = and i32 %address.local451, 1, !dbg !18
  %address.local453 = zext nneg i32 %address.local452 to i64, !dbg !18
  %address.local454 = add nsw i64 %address.base, %address.local453, !dbg !18
  %address.local455 = shl i64 %address.local454, 6, !dbg !18
  %address.local456 = sub i64 %address.local450, %address.local455, !dbg !18
  %address.local457 = add i64 %address.local456, %address.local425, !dbg !18
  %address.local458 = add i64 %address.local457, %address.local429, !dbg !18
  %address.local459 = add i64 %address.local458, 32, !dbg !18
  %address.local460 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local459, !dbg !18
  %sched.n149 = load <4 x half>, ptr addrspace(3) %address.local460, align 8, !dbg !18
  %lds.const.gep1396 = getelementptr i8, ptr addrspace(3) %address.local460, i64 4096, !dbg !18
  %sched.n152 = load <4 x half>, ptr addrspace(3) %lds.const.gep1396, align 8, !dbg !18
  %sched.n150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n148, <4 x float> %sched.n147) #9, !dbg !18
  %sched.n151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n149, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1397 = getelementptr i8, ptr addrspace(3) %address.local431, i64 128, !dbg !18
  %sched.n153 = load <4 x half>, ptr addrspace(3) %lds.const.gep1397, align 8, !dbg !18
  %lds.const.gep1398 = getelementptr i8, ptr addrspace(3) %address.local431, i64 4224, !dbg !18
  %sched.n156 = load <4 x half>, ptr addrspace(3) %lds.const.gep1398, align 8, !dbg !18
  %sched.n154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n152, <4 x float> %sched.n151) #9, !dbg !18
  %sched.n155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n153, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1399 = getelementptr i8, ptr addrspace(3) %address.local460, i64 128, !dbg !18
  %sched.n157 = load <4 x half>, ptr addrspace(3) %lds.const.gep1399, align 8, !dbg !18
  %lds.const.gep1400 = getelementptr i8, ptr addrspace(3) %address.local460, i64 4224, !dbg !18
  %sched.n160 = load <4 x half>, ptr addrspace(3) %lds.const.gep1400, align 8, !dbg !18
  %sched.n158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n156, <4 x float> %sched.n155) #9, !dbg !18
  %sched.n159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n157, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1401 = getelementptr i8, ptr addrspace(3) %address.local431, i64 1024, !dbg !18
  %sched.n161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1401, align 8, !dbg !18
  %lds.const.gep1402 = getelementptr i8, ptr addrspace(3) %address.local431, i64 5120, !dbg !18
  %sched.n164 = load <4 x half>, ptr addrspace(3) %lds.const.gep1402, align 8, !dbg !18
  %sched.n162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n160, <4 x float> %sched.n159) #9, !dbg !18
  %sched.n163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n161, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1403 = getelementptr i8, ptr addrspace(3) %address.local460, i64 1024, !dbg !18
  %sched.n165 = load <4 x half>, ptr addrspace(3) %lds.const.gep1403, align 8, !dbg !18
  %lds.const.gep1404 = getelementptr i8, ptr addrspace(3) %address.local460, i64 5120, !dbg !18
  %sched.n168 = load <4 x half>, ptr addrspace(3) %lds.const.gep1404, align 8, !dbg !18
  %sched.n166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n164, <4 x float> %sched.n163) #9, !dbg !18
  %sched.n167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n165, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1405 = getelementptr i8, ptr addrspace(3) %address.local431, i64 1152, !dbg !18
  %sched.n169 = load <4 x half>, ptr addrspace(3) %lds.const.gep1405, align 8, !dbg !18
  %lds.const.gep1406 = getelementptr i8, ptr addrspace(3) %address.local431, i64 5248, !dbg !18
  %sched.n172 = load <4 x half>, ptr addrspace(3) %lds.const.gep1406, align 8, !dbg !18
  %sched.n170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n168, <4 x float> %sched.n167) #9, !dbg !18
  %sched.n171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n169, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1407 = getelementptr i8, ptr addrspace(3) %address.local460, i64 1152, !dbg !18
  %sched.n173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1407, align 8, !dbg !18
  %lds.const.gep1408 = getelementptr i8, ptr addrspace(3) %address.local460, i64 5248, !dbg !18
  %sched.n176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1408, align 8, !dbg !18
  %sched.n174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n172, <4 x float> %sched.n171) #9, !dbg !18
  %sched.n175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n68, <4 x half> %sched.n173, <4 x float> zeroinitializer) #9, !dbg !18
  %sched.n177 = load <4 x half>, ptr addrspace(3) %address.local431, align 8, !dbg !18
  %sched.n178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n70, <4 x half> %sched.n176, <4 x float> %sched.n175) #9, !dbg !18
  %sched.n179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n177, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1409 = getelementptr i8, ptr addrspace(3) %address.local431, i64 4096, !dbg !18
  %sched.n180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1409, align 8, !dbg !18
  %sched.n181 = load <4 x half>, ptr addrspace(3) %address.local460, align 8, !dbg !18
  %sched.n182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n180, <4 x float> %sched.n179) #9, !dbg !18
  %sched.n183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n181, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1410 = getelementptr i8, ptr addrspace(3) %address.local460, i64 4096, !dbg !18
  %sched.n184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1410, align 8, !dbg !18
  %lds.const.gep1411 = getelementptr i8, ptr addrspace(3) %address.local431, i64 128, !dbg !18
  %sched.n185 = load <4 x half>, ptr addrspace(3) %lds.const.gep1411, align 8, !dbg !18
  %sched.n186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n184, <4 x float> %sched.n183) #9, !dbg !18
  %sched.n187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n185, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1412 = getelementptr i8, ptr addrspace(3) %address.local431, i64 4224, !dbg !18
  %sched.n188 = load <4 x half>, ptr addrspace(3) %lds.const.gep1412, align 8, !dbg !18
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local460, i64 128, !dbg !18
  %sched.n189 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !18
  %sched.n190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n188, <4 x float> %sched.n187) #9, !dbg !18
  %sched.n191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n189, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local460, i64 4224, !dbg !18
  %sched.n192 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !18
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local431, i64 1024, !dbg !18
  %sched.n193 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !18
  %sched.n194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n192, <4 x float> %sched.n191) #9, !dbg !18
  %sched.n195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n193, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local431, i64 5120, !dbg !18
  %sched.n196 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !18
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local460, i64 1024, !dbg !18
  %sched.n197 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !18
  %sched.n198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n196, <4 x float> %sched.n195) #9, !dbg !18
  %sched.n199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n197, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local460, i64 5120, !dbg !18
  %sched.n200 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !18
  %lds.const.gep1419 = getelementptr i8, ptr addrspace(3) %address.local431, i64 1152, !dbg !18
  %sched.n201 = load <4 x half>, ptr addrspace(3) %lds.const.gep1419, align 8, !dbg !18
  %sched.n202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n200, <4 x float> %sched.n199) #9, !dbg !18
  %sched.n203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n201, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1420 = getelementptr i8, ptr addrspace(3) %address.local431, i64 5248, !dbg !18
  %sched.n204 = load <4 x half>, ptr addrspace(3) %lds.const.gep1420, align 8, !dbg !18
  %lds.const.gep1421 = getelementptr i8, ptr addrspace(3) %address.local460, i64 1152, !dbg !18
  %sched.n205 = load <4 x half>, ptr addrspace(3) %lds.const.gep1421, align 8, !dbg !18
  %sched.n206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n204, <4 x float> %sched.n203) #9, !dbg !18
  %lds.const.gep1422 = getelementptr i8, ptr addrspace(3) %address.local460, i64 5248, !dbg !18
  %sched.n207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1422, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %sched.n208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n72, <4 x half> %sched.n205, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %sched.n209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n73, <4 x half> %sched.n207, <4 x float> %sched.n208) #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %sched.n210 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local384), !dbg !8
  %sched.n211 = zext i32 %sched.n210 to i64, !dbg !8
  %sched.n212 = inttoptr i64 %sched.n211 to ptr addrspace(3), !dbg !8
  %address.local769 = shl nuw nsw i64 %address.local379, 7, !dbg !8
  %address.local770 = and i64 %address.local769, 16256, !dbg !8
  %address.local771 = lshr i64 %address.local379, 4, !dbg !8
  %address.local772 = and i64 %address.local771, 8, !dbg !8
  %address.local773 = or disjoint i64 %address.local770, %address.local772, !dbg !8
  %address.local774 = getelementptr [2 x i8], ptr addrspace(1) %sched.n87, i64 %address.local773, !dbg !8
  %sched.n213 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %sched.n214 = ptrtoint ptr addrspace(1) %address.local774 to i64, !dbg !8
  %sched.n215 = sub i64 %sched.n214, %sched.n213, !dbg !8
  %dma.buffer.offset1585 = trunc i64 %sched.n215 to i32, !dbg !8
  %sched.n216 = trunc i64 %sched.n213 to i32, !dbg !8
  %sched.n217 = insertelement <4 x i32> poison, i32 %sched.n216, i64 0, !dbg !8
  %sched.n218 = lshr i64 %sched.n213, 32, !dbg !8
  %sched.n219 = trunc i64 %sched.n218 to i32, !dbg !8
  %sched.n220 = insertelement <4 x i32> %sched.n217, i32 %sched.n219, i64 1, !dbg !8
  %sched.n221 = insertelement <4 x i32> %sched.n220, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1586 = insertelement <4 x i32> %sched.n221, i32 131072, i64 3, !dbg !8
  %sched.n222 = ptrtoint ptr addrspace(3) %sched.n212 to i32, !dbg !8
  %dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n222), !dbg !8
  %sched.n223 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1586) #10, !dbg !8
  %sched.n224 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local401), !dbg !8
  %sched.n225 = zext i32 %sched.n224 to i64, !dbg !8
  %sched.n226 = inttoptr i64 %sched.n225 to ptr addrspace(3), !dbg !8
  %address.local775 = shl nuw nsw i64 %address.local379, 7, !dbg !8
  %address.local776 = and i64 %address.local775, 16256, !dbg !8
  %address.local777 = lshr i64 %address.local396, 4, !dbg !8
  %address.local778 = and i64 %address.local777, 24, !dbg !8
  %address.local779 = or disjoint i64 %address.local776, %address.local778, !dbg !8
  %address.local780 = getelementptr [2 x i8], ptr addrspace(1) %sched.n87, i64 %address.local779, !dbg !8
  %sched.n227 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %sched.n228 = ptrtoint ptr addrspace(1) %address.local780 to i64, !dbg !8
  %sched.n229 = sub i64 %sched.n228, %sched.n227, !dbg !8
  %dma.buffer.offset1582 = trunc i64 %sched.n229 to i32, !dbg !8
  %sched.n230 = trunc i64 %sched.n227 to i32, !dbg !8
  %sched.n231 = insertelement <4 x i32> poison, i32 %sched.n230, i64 0, !dbg !8
  %sched.n232 = lshr i64 %sched.n227, 32, !dbg !8
  %sched.n233 = trunc i64 %sched.n232 to i32, !dbg !8
  %sched.n234 = insertelement <4 x i32> %sched.n231, i32 %sched.n233, i64 1, !dbg !8
  %sched.n235 = insertelement <4 x i32> %sched.n234, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1583 = insertelement <4 x i32> %sched.n235, i32 131072, i64 3, !dbg !8
  %sched.n236 = ptrtoint ptr addrspace(3) %sched.n226 to i32, !dbg !8
  %dma.buffer.lds1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n236), !dbg !8
  %sched.n237 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1583) #10, !dbg !8
  %sched.n238 = add nuw i64 %sched.n97, 1024, !dbg !17
  %sched.n239 = icmp slt i64 %sched.n238, 0, !dbg !17
  %sched.n240 = sub nuw i64 -1025, %sched.n97, !dbg !17
  %sched.n241 = select i1 %sched.n239, i64 %sched.n240, i64 %sched.n238, !dbg !17
  %sched.n242 = sdiv i64 %sched.n241, 4096, !dbg !17
  %.lobit37 = ashr i64 %sched.n238, 63, !dbg !17
  %sched.n243 = xor i64 %sched.n242, %.lobit37, !dbg !17
  %sched.n244 = add nsw i64 %sched.n243, %sched.n16, !dbg !17
  %sched.n245 = add nsw i64 %sched.n108, %sched.n91, !dbg !17
  %sched.n246 = add nsw i64 %sched.n245, 32, !dbg !17
  %sched.n247 = icmp slt i64 %sched.n245, -32, !dbg !17
  %sched.n248 = sub nsw i64 -33, %sched.n245, !dbg !17
  %sched.n249 = select i1 %sched.n247, i64 %sched.n248, i64 %sched.n246, !dbg !17
  %sched.n250 = sdiv i64 %sched.n249, 128, !dbg !17
  %sched.n251 = sext i1 %sched.n247 to i64, !dbg !17
  %sched.n252 = xor i64 %sched.n250, %sched.n251, !dbg !17
  %sched.n253 = shl nsw i64 %sched.n252, 7, !dbg !17
  %sched.n254 = sub i64 %sched.n91, %sched.n253, !dbg !17
  %sched.n255 = shl i64 %sched.n244, 19, !dbg !17
  %sched.n256 = and i64 %sched.n255, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %sched.n257 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %sched.n256, !dbg !17
  %sched.n258 = getelementptr [2 x i8], ptr addrspace(1) %sched.n257, i64 %sched.n108, !dbg !17
  %sched.n259 = getelementptr [2 x i8], ptr addrspace(1) %sched.n258, i64 %sched.n254, !dbg !17
  %sched.n260 = getelementptr i8, ptr addrspace(1) %sched.n259, i64 64, !dbg !17
  %sched.n261 = getelementptr [2 x i8], ptr addrspace(1) %sched.n260, i64 %sched.n114, !dbg !17
  %address.local781 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local381, !dbg !17
  %address.local782 = ptrtoint ptr addrspace(3) %address.local781 to i64, !dbg !17
  %address.local783 = trunc i64 %address.local782 to i32, !dbg !17
  %sched.n262 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local783), !dbg !17
  %sched.n263 = zext i32 %sched.n262 to i64, !dbg !17
  %sched.n264 = inttoptr i64 %sched.n263 to ptr addrspace(3), !dbg !17
  %address.local784 = getelementptr [2 x i8], ptr addrspace(1) %sched.n261, i64 %address.local394, !dbg !17
  %sched.n265 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %sched.n266 = ptrtoint ptr addrspace(1) %address.local784 to i64, !dbg !17
  %sched.n267 = sub i64 %sched.n266, %sched.n265, !dbg !17
  %dma.buffer.offset1579 = trunc i64 %sched.n267 to i32, !dbg !17
  %sched.n268 = trunc i64 %sched.n265 to i32, !dbg !17
  %sched.n269 = insertelement <4 x i32> poison, i32 %sched.n268, i64 0, !dbg !17
  %sched.n270 = lshr i64 %sched.n265, 32, !dbg !17
  %sched.n271 = trunc i64 %sched.n270 to i32, !dbg !17
  %sched.n272 = insertelement <4 x i32> %sched.n269, i32 %sched.n271, i64 1, !dbg !17
  %sched.n273 = insertelement <4 x i32> %sched.n272, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1580 = insertelement <4 x i32> %sched.n273, i32 131072, i64 3, !dbg !17
  %sched.n274 = ptrtoint ptr addrspace(3) %sched.n264 to i32, !dbg !17
  %dma.buffer.lds1581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n274), !dbg !17
  %sched.n275 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1580) #10, !dbg !17
  %address.local785 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local398, !dbg !17
  %address.local786 = ptrtoint ptr addrspace(3) %address.local785 to i64, !dbg !17
  %address.local787 = trunc i64 %address.local786 to i32, !dbg !17
  %sched.n276 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local787), !dbg !17
  %sched.n277 = zext i32 %sched.n276 to i64, !dbg !17
  %sched.n278 = inttoptr i64 %sched.n277 to ptr addrspace(3), !dbg !17
  %address.local788 = getelementptr [2 x i8], ptr addrspace(1) %sched.n261, i64 %address.local411, !dbg !17
  %sched.n279 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %sched.n280 = ptrtoint ptr addrspace(1) %address.local788 to i64, !dbg !17
  %sched.n281 = sub i64 %sched.n280, %sched.n279, !dbg !17
  %dma.buffer.offset1576 = trunc i64 %sched.n281 to i32, !dbg !17
  %sched.n282 = trunc i64 %sched.n279 to i32, !dbg !17
  %sched.n283 = insertelement <4 x i32> poison, i32 %sched.n282, i64 0, !dbg !17
  %sched.n284 = lshr i64 %sched.n279, 32, !dbg !17
  %sched.n285 = trunc i64 %sched.n284 to i32, !dbg !17
  %sched.n286 = insertelement <4 x i32> %sched.n283, i32 %sched.n285, i64 1, !dbg !17
  %sched.n287 = insertelement <4 x i32> %sched.n286, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1577 = insertelement <4 x i32> %sched.n287, i32 131072, i64 3, !dbg !17
  %sched.n288 = ptrtoint ptr addrspace(3) %sched.n278 to i32, !dbg !17
  %dma.buffer.lds1578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n288), !dbg !17
  %sched.n289 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1577) #10, !dbg !17
  %lds.const.gep1423 = getelementptr i8, ptr addrspace(3) %address.local371, i64 0, !dbg !11
  %sched.n290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1423, align 8, !dbg !11
  %lds.const.gep1424 = getelementptr i8, ptr addrspace(3) %address.local371, i64 4096, !dbg !11
  %sched.n291 = load <4 x half>, ptr addrspace(3) %lds.const.gep1424, align 8, !dbg !11
  %lds.const.gep1425 = getelementptr i8, ptr addrspace(3) %address.local371, i64 1024, !dbg !11
  %sched.n292 = load <4 x half>, ptr addrspace(3) %lds.const.gep1425, align 8, !dbg !11
  %sched.n293 = fmul <4 x half> %sched.n290, splat (half 0xH3015), !dbg !11
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local371, i64 5120, !dbg !11
  %sched.n294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !11
  %sched.n295 = fmul <4 x half> %sched.n291, splat (half 0xH3015), !dbg !11
  %sched.n296 = fmul <4 x half> %sched.n292, splat (half 0xH3015), !dbg !11
  %sched.n297 = fmul <4 x half> %sched.n294, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8192, !dbg !18
  %sched.n298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !18
  %sched.n299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n298, <4 x float> %sched.n150) #9, !dbg !18
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12288, !dbg !18
  %sched.n300 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !18
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8192, !dbg !18
  %sched.n301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !18
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12288, !dbg !18
  %sched.n304 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !18
  %sched.n302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n300, <4 x float> %sched.n299) #9, !dbg !18
  %sched.n303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n301, <4 x float> %sched.n154) #9, !dbg !18
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8320, !dbg !18
  %sched.n305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !18
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12416, !dbg !18
  %sched.n308 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !18
  %sched.n306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n304, <4 x float> %sched.n303) #9, !dbg !18
  %sched.n307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n305, <4 x float> %sched.n158) #9, !dbg !18
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8320, !dbg !18
  %sched.n309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !18
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12416, !dbg !18
  %sched.n312 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !18
  %sched.n310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n308, <4 x float> %sched.n307) #9, !dbg !18
  %sched.n311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n309, <4 x float> %sched.n162) #9, !dbg !18
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9216, !dbg !18
  %sched.n313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !18
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13312, !dbg !18
  %sched.n316 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !18
  %sched.n314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n312, <4 x float> %sched.n311) #9, !dbg !18
  %sched.n315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n313, <4 x float> %sched.n166) #9, !dbg !18
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9216, !dbg !18
  %sched.n317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !18
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13312, !dbg !18
  %sched.n320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !18
  %sched.n318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n316, <4 x float> %sched.n315) #9, !dbg !18
  %sched.n319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n317, <4 x float> %sched.n170) #9, !dbg !18
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9344, !dbg !18
  %sched.n321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !18
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13440, !dbg !18
  %sched.n324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !18
  %sched.n322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n320, <4 x float> %sched.n319) #9, !dbg !18
  %sched.n323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n321, <4 x float> %sched.n174) #9, !dbg !18
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9344, !dbg !18
  %sched.n325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !18
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13440, !dbg !18
  %sched.n328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !18
  %sched.n326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n324, <4 x float> %sched.n323) #9, !dbg !18
  %sched.n327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n293, <4 x half> %sched.n325, <4 x float> %sched.n178) #9, !dbg !18
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8192, !dbg !18
  %sched.n329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !18
  %sched.n330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n295, <4 x half> %sched.n328, <4 x float> %sched.n327) #9, !dbg !18
  %sched.n331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n329, <4 x float> %sched.n182) #9, !dbg !18
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12288, !dbg !18
  %sched.n332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !18
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8192, !dbg !18
  %sched.n333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !18
  %sched.n334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n332, <4 x float> %sched.n331) #9, !dbg !18
  %sched.n335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n333, <4 x float> %sched.n186) #9, !dbg !18
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12288, !dbg !18
  %sched.n336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !18
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8320, !dbg !18
  %sched.n337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !18
  %sched.n338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n336, <4 x float> %sched.n335) #9, !dbg !18
  %sched.n339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n337, <4 x float> %sched.n190) #9, !dbg !18
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12416, !dbg !18
  %sched.n340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !18
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8320, !dbg !18
  %sched.n341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !18
  %sched.n342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n340, <4 x float> %sched.n339) #9, !dbg !18
  %sched.n343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n341, <4 x float> %sched.n194) #9, !dbg !18
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12416, !dbg !18
  %sched.n344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !18
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9216, !dbg !18
  %sched.n345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !18
  %sched.n346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n344, <4 x float> %sched.n343) #9, !dbg !18
  %sched.n347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n345, <4 x float> %sched.n198) #9, !dbg !18
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13312, !dbg !18
  %sched.n348 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !18
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9216, !dbg !18
  %sched.n349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !18
  %sched.n350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n348, <4 x float> %sched.n347) #9, !dbg !18
  %sched.n351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n349, <4 x float> %sched.n202) #9, !dbg !18
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13312, !dbg !18
  %sched.n352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !18
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9344, !dbg !18
  %sched.n353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !18
  %sched.n354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n352, <4 x float> %sched.n351) #9, !dbg !18
  %sched.n355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n353, <4 x float> %sched.n206) #9, !dbg !18
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13440, !dbg !18
  %sched.n356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !18
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9344, !dbg !18
  %sched.n357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !18
  %sched.n358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n356, <4 x float> %sched.n355) #9, !dbg !18
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13440, !dbg !18
  %sched.n359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %sched.n360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n296, <4 x half> %sched.n357, <4 x float> %sched.n209) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %sched.n361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n297, <4 x half> %sched.n359, <4 x float> %sched.n360) #9, !dbg !18
  %sched.n362 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local384), !dbg !8
  %sched.n363 = zext i32 %sched.n362 to i64, !dbg !8
  %sched.n364 = inttoptr i64 %sched.n363 to ptr addrspace(3), !dbg !8
  %address.local824 = getelementptr [2 x i8], ptr addrspace(1) %sched.n93, i64 %address.local773, !dbg !8
  %sched.n365 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %sched.n366 = ptrtoint ptr addrspace(1) %address.local824 to i64, !dbg !8
  %sched.n367 = sub i64 %sched.n366, %sched.n365, !dbg !8
  %dma.buffer.offset1573 = trunc i64 %sched.n367 to i32, !dbg !8
  %sched.n368 = trunc i64 %sched.n365 to i32, !dbg !8
  %sched.n369 = insertelement <4 x i32> poison, i32 %sched.n368, i64 0, !dbg !8
  %sched.n370 = lshr i64 %sched.n365, 32, !dbg !8
  %sched.n371 = trunc i64 %sched.n370 to i32, !dbg !8
  %sched.n372 = insertelement <4 x i32> %sched.n369, i32 %sched.n371, i64 1, !dbg !8
  %sched.n373 = insertelement <4 x i32> %sched.n372, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1574 = insertelement <4 x i32> %sched.n373, i32 131072, i64 3, !dbg !8
  %sched.n374 = ptrtoint ptr addrspace(3) %sched.n364 to i32, !dbg !8
  %dma.buffer.lds1575 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n374), !dbg !8
  %sched.n375 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1574) #10, !dbg !8
  %sched.n376 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local401), !dbg !8
  %sched.n377 = zext i32 %sched.n376 to i64, !dbg !8
  %sched.n378 = inttoptr i64 %sched.n377 to ptr addrspace(3), !dbg !8
  %address.local825 = getelementptr [2 x i8], ptr addrspace(1) %sched.n93, i64 %address.local779, !dbg !8
  %sched.n379 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %sched.n380 = ptrtoint ptr addrspace(1) %address.local825 to i64, !dbg !8
  %sched.n381 = sub i64 %sched.n380, %sched.n379, !dbg !8
  %dma.buffer.offset1570 = trunc i64 %sched.n381 to i32, !dbg !8
  %sched.n382 = trunc i64 %sched.n379 to i32, !dbg !8
  %sched.n383 = insertelement <4 x i32> poison, i32 %sched.n382, i64 0, !dbg !8
  %sched.n384 = lshr i64 %sched.n379, 32, !dbg !8
  %sched.n385 = trunc i64 %sched.n384 to i32, !dbg !8
  %sched.n386 = insertelement <4 x i32> %sched.n383, i32 %sched.n385, i64 1, !dbg !8
  %sched.n387 = insertelement <4 x i32> %sched.n386, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1571 = insertelement <4 x i32> %sched.n387, i32 131072, i64 3, !dbg !8
  %sched.n388 = ptrtoint ptr addrspace(3) %sched.n378 to i32, !dbg !8
  %dma.buffer.lds1572 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n388), !dbg !8
  %sched.n389 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1572, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1571) #10, !dbg !8
  %sched.n390 = add nuw i64 %sched.n97, 2048, !dbg !17
  %sched.n391 = icmp slt i64 %sched.n390, 0, !dbg !17
  %sched.n392 = sub nuw i64 -2049, %sched.n97, !dbg !17
  %sched.n393 = select i1 %sched.n391, i64 %sched.n392, i64 %sched.n390, !dbg !17
  %sched.n394 = sdiv i64 %sched.n393, 4096, !dbg !17
  %.lobit38 = ashr i64 %sched.n390, 63, !dbg !17
  %sched.n395 = xor i64 %sched.n394, %.lobit38, !dbg !17
  %sched.n396 = add nsw i64 %sched.n395, %sched.n16, !dbg !17
  %sched.n397 = add nsw i64 %sched.n245, 64, !dbg !17
  %sched.n398 = icmp slt i64 %sched.n245, -64, !dbg !17
  %sched.n399 = sub nsw i64 -65, %sched.n245, !dbg !17
  %sched.n400 = select i1 %sched.n398, i64 %sched.n399, i64 %sched.n397, !dbg !17
  %sched.n401 = sdiv i64 %sched.n400, 128, !dbg !17
  %sched.n402 = sext i1 %sched.n398 to i64, !dbg !17
  %sched.n403 = xor i64 %sched.n401, %sched.n402, !dbg !17
  %sched.n404 = shl nsw i64 %sched.n403, 7, !dbg !17
  %sched.n405 = sub i64 %sched.n91, %sched.n404, !dbg !17
  %sched.n406 = shl i64 %sched.n396, 19, !dbg !17
  %sched.n407 = and i64 %sched.n406, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %sched.n408 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %sched.n407, !dbg !17
  %sched.n409 = getelementptr [2 x i8], ptr addrspace(1) %sched.n408, i64 %sched.n108, !dbg !17
  %sched.n410 = getelementptr [2 x i8], ptr addrspace(1) %sched.n409, i64 %sched.n405, !dbg !17
  %sched.n411 = getelementptr i8, ptr addrspace(1) %sched.n410, i64 128, !dbg !17
  %sched.n412 = getelementptr [2 x i8], ptr addrspace(1) %sched.n411, i64 %sched.n114, !dbg !17
  %sched.n413 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local783), !dbg !17
  %sched.n414 = zext i32 %sched.n413 to i64, !dbg !17
  %sched.n415 = inttoptr i64 %sched.n414 to ptr addrspace(3), !dbg !17
  %address.local826 = getelementptr [2 x i8], ptr addrspace(1) %sched.n412, i64 %address.local394, !dbg !17
  %sched.n416 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %sched.n417 = ptrtoint ptr addrspace(1) %address.local826 to i64, !dbg !17
  %sched.n418 = sub i64 %sched.n417, %sched.n416, !dbg !17
  %dma.buffer.offset1567 = trunc i64 %sched.n418 to i32, !dbg !17
  %sched.n419 = trunc i64 %sched.n416 to i32, !dbg !17
  %sched.n420 = insertelement <4 x i32> poison, i32 %sched.n419, i64 0, !dbg !17
  %sched.n421 = lshr i64 %sched.n416, 32, !dbg !17
  %sched.n422 = trunc i64 %sched.n421 to i32, !dbg !17
  %sched.n423 = insertelement <4 x i32> %sched.n420, i32 %sched.n422, i64 1, !dbg !17
  %sched.n424 = insertelement <4 x i32> %sched.n423, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1568 = insertelement <4 x i32> %sched.n424, i32 131072, i64 3, !dbg !17
  %sched.n425 = ptrtoint ptr addrspace(3) %sched.n415 to i32, !dbg !17
  %dma.buffer.lds1569 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n425), !dbg !17
  %sched.n426 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1569, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1568) #10, !dbg !17
  %sched.n427 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local787), !dbg !17
  %sched.n428 = zext i32 %sched.n427 to i64, !dbg !17
  %sched.n429 = inttoptr i64 %sched.n428 to ptr addrspace(3), !dbg !17
  %address.local827 = getelementptr [2 x i8], ptr addrspace(1) %sched.n412, i64 %address.local411, !dbg !17
  %sched.n430 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %sched.n431 = ptrtoint ptr addrspace(1) %address.local827 to i64, !dbg !17
  %sched.n432 = sub i64 %sched.n431, %sched.n430, !dbg !17
  %dma.buffer.offset1564 = trunc i64 %sched.n432 to i32, !dbg !17
  %sched.n433 = trunc i64 %sched.n430 to i32, !dbg !17
  %sched.n434 = insertelement <4 x i32> poison, i32 %sched.n433, i64 0, !dbg !17
  %sched.n435 = lshr i64 %sched.n430, 32, !dbg !17
  %sched.n436 = trunc i64 %sched.n435 to i32, !dbg !17
  %sched.n437 = insertelement <4 x i32> %sched.n434, i32 %sched.n436, i64 1, !dbg !17
  %sched.n438 = insertelement <4 x i32> %sched.n437, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1565 = insertelement <4 x i32> %sched.n438, i32 131072, i64 3, !dbg !17
  %sched.n439 = ptrtoint ptr addrspace(3) %sched.n429 to i32, !dbg !17
  %dma.buffer.lds1566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n439), !dbg !17
  %sched.n440 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1566, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #10, !dbg !17
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local371, i64 0, !dbg !11
  %sched.n441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !11
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local371, i64 4096, !dbg !11
  %sched.n442 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !11
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local371, i64 1024, !dbg !11
  %sched.n443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !11
  %sched.n444 = fmul <4 x half> %sched.n441, splat (half 0xH3015), !dbg !11
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local371, i64 5120, !dbg !11
  %sched.n445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !11
  %sched.n446 = fmul <4 x half> %sched.n442, splat (half 0xH3015), !dbg !11
  %sched.n447 = fmul <4 x half> %sched.n443, splat (half 0xH3015), !dbg !11
  %sched.n448 = fmul <4 x half> %sched.n445, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8192, !dbg !18
  %sched.n449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !18
  %sched.n450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n449, <4 x float> %sched.n302) #9, !dbg !18
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12288, !dbg !18
  %sched.n451 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !18
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8192, !dbg !18
  %sched.n452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !18
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12288, !dbg !18
  %sched.n455 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !18
  %sched.n453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n451, <4 x float> %sched.n450) #9, !dbg !18
  %sched.n454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n452, <4 x float> %sched.n306) #9, !dbg !18
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8320, !dbg !18
  %sched.n456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !18
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12416, !dbg !18
  %sched.n459 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !18
  %sched.n457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n455, <4 x float> %sched.n454) #9, !dbg !18
  %sched.n458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n456, <4 x float> %sched.n310) #9, !dbg !18
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8320, !dbg !18
  %sched.n460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !18
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12416, !dbg !18
  %sched.n463 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !18
  %sched.n461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n459, <4 x float> %sched.n458) #9, !dbg !18
  %sched.n462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n460, <4 x float> %sched.n314) #9, !dbg !18
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9216, !dbg !18
  %sched.n464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !18
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13312, !dbg !18
  %sched.n467 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !18
  %sched.n465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n463, <4 x float> %sched.n462) #9, !dbg !18
  %sched.n466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n464, <4 x float> %sched.n318) #9, !dbg !18
  %lds.const.gep1473 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9216, !dbg !18
  %sched.n468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1473, align 8, !dbg !18
  %lds.const.gep1474 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13312, !dbg !18
  %sched.n471 = load <4 x half>, ptr addrspace(3) %lds.const.gep1474, align 8, !dbg !18
  %sched.n469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n467, <4 x float> %sched.n466) #9, !dbg !18
  %sched.n470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n468, <4 x float> %sched.n322) #9, !dbg !18
  %lds.const.gep1475 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9344, !dbg !18
  %sched.n472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1475, align 8, !dbg !18
  %lds.const.gep1476 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13440, !dbg !18
  %sched.n475 = load <4 x half>, ptr addrspace(3) %lds.const.gep1476, align 8, !dbg !18
  %sched.n473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n471, <4 x float> %sched.n470) #9, !dbg !18
  %sched.n474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n472, <4 x float> %sched.n326) #9, !dbg !18
  %lds.const.gep1477 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9344, !dbg !18
  %sched.n476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1477, align 8, !dbg !18
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13440, !dbg !18
  %sched.n479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !18
  %sched.n477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n475, <4 x float> %sched.n474) #9, !dbg !18
  %sched.n478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n444, <4 x half> %sched.n476, <4 x float> %sched.n330) #9, !dbg !18
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8192, !dbg !18
  %sched.n480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !18
  %sched.n481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n446, <4 x half> %sched.n479, <4 x float> %sched.n478) #9, !dbg !18
  %sched.n482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n480, <4 x float> %sched.n334) #9, !dbg !18
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12288, !dbg !18
  %sched.n483 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !18
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8192, !dbg !18
  %sched.n484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !18
  %sched.n485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n483, <4 x float> %sched.n482) #9, !dbg !18
  %sched.n486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n484, <4 x float> %sched.n338) #9, !dbg !18
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12288, !dbg !18
  %sched.n487 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !18
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8320, !dbg !18
  %sched.n488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !18
  %sched.n489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n487, <4 x float> %sched.n486) #9, !dbg !18
  %sched.n490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n488, <4 x float> %sched.n342) #9, !dbg !18
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12416, !dbg !18
  %sched.n491 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !18
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8320, !dbg !18
  %sched.n492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !18
  %sched.n493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n491, <4 x float> %sched.n490) #9, !dbg !18
  %sched.n494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n492, <4 x float> %sched.n346) #9, !dbg !18
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12416, !dbg !18
  %sched.n495 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !18
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9216, !dbg !18
  %sched.n496 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !18
  %sched.n497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n495, <4 x float> %sched.n494) #9, !dbg !18
  %sched.n498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n496, <4 x float> %sched.n350) #9, !dbg !18
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13312, !dbg !18
  %sched.n499 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !18
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9216, !dbg !18
  %sched.n500 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !18
  %sched.n501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n499, <4 x float> %sched.n498) #9, !dbg !18
  %sched.n502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n500, <4 x float> %sched.n354) #9, !dbg !18
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13312, !dbg !18
  %sched.n503 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !18
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9344, !dbg !18
  %sched.n504 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !18
  %sched.n505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n503, <4 x float> %sched.n502) #9, !dbg !18
  %sched.n506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n504, <4 x float> %sched.n358) #9, !dbg !18
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13440, !dbg !18
  %sched.n507 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !18
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9344, !dbg !18
  %sched.n508 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !18
  %sched.n509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n507, <4 x float> %sched.n506) #9, !dbg !18
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13440, !dbg !18
  %sched.n510 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %sched.n511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n447, <4 x half> %sched.n508, <4 x float> %sched.n361) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %sched.n512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n448, <4 x half> %sched.n510, <4 x float> %sched.n511) #9, !dbg !18
  %sched.n513 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local384), !dbg !8
  %sched.n514 = zext i32 %sched.n513 to i64, !dbg !8
  %sched.n515 = inttoptr i64 %sched.n514 to ptr addrspace(3), !dbg !8
  %address.local828 = getelementptr [2 x i8], ptr addrspace(1) %sched.n95, i64 %address.local773, !dbg !8
  %sched.n516 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %sched.n517 = ptrtoint ptr addrspace(1) %address.local828 to i64, !dbg !8
  %sched.n518 = sub i64 %sched.n517, %sched.n516, !dbg !8
  %dma.buffer.offset1561 = trunc i64 %sched.n518 to i32, !dbg !8
  %sched.n519 = trunc i64 %sched.n516 to i32, !dbg !8
  %sched.n520 = insertelement <4 x i32> poison, i32 %sched.n519, i64 0, !dbg !8
  %sched.n521 = lshr i64 %sched.n516, 32, !dbg !8
  %sched.n522 = trunc i64 %sched.n521 to i32, !dbg !8
  %sched.n523 = insertelement <4 x i32> %sched.n520, i32 %sched.n522, i64 1, !dbg !8
  %sched.n524 = insertelement <4 x i32> %sched.n523, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1562 = insertelement <4 x i32> %sched.n524, i32 131072, i64 3, !dbg !8
  %sched.n525 = ptrtoint ptr addrspace(3) %sched.n515 to i32, !dbg !8
  %dma.buffer.lds1563 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n525), !dbg !8
  %sched.n526 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1563, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1562) #10, !dbg !8
  %sched.n527 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local401), !dbg !8
  %sched.n528 = zext i32 %sched.n527 to i64, !dbg !8
  %sched.n529 = inttoptr i64 %sched.n528 to ptr addrspace(3), !dbg !8
  %address.local829 = getelementptr [2 x i8], ptr addrspace(1) %sched.n95, i64 %address.local779, !dbg !8
  %sched.n530 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %sched.n531 = ptrtoint ptr addrspace(1) %address.local829 to i64, !dbg !8
  %sched.n532 = sub i64 %sched.n531, %sched.n530, !dbg !8
  %dma.buffer.offset1558 = trunc i64 %sched.n532 to i32, !dbg !8
  %sched.n533 = trunc i64 %sched.n530 to i32, !dbg !8
  %sched.n534 = insertelement <4 x i32> poison, i32 %sched.n533, i64 0, !dbg !8
  %sched.n535 = lshr i64 %sched.n530, 32, !dbg !8
  %sched.n536 = trunc i64 %sched.n535 to i32, !dbg !8
  %sched.n537 = insertelement <4 x i32> %sched.n534, i32 %sched.n536, i64 1, !dbg !8
  %sched.n538 = insertelement <4 x i32> %sched.n537, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1559 = insertelement <4 x i32> %sched.n538, i32 131072, i64 3, !dbg !8
  %sched.n539 = ptrtoint ptr addrspace(3) %sched.n529 to i32, !dbg !8
  %dma.buffer.lds1560 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n539), !dbg !8
  %sched.n540 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1559) #10, !dbg !8
  %sched.n541 = add nuw i64 %sched.n97, 3072, !dbg !17
  %sched.n542 = icmp slt i64 %sched.n541, 0, !dbg !17
  %sched.n543 = sub nuw i64 -3073, %sched.n97, !dbg !17
  %sched.n544 = select i1 %sched.n542, i64 %sched.n543, i64 %sched.n541, !dbg !17
  %sched.n545 = sdiv i64 %sched.n544, 4096, !dbg !17
  %.lobit39 = ashr i64 %sched.n541, 63, !dbg !17
  %sched.n546 = xor i64 %sched.n545, %.lobit39, !dbg !17
  %sched.n547 = add nsw i64 %sched.n546, %sched.n16, !dbg !17
  %sched.n548 = add nsw i64 %sched.n245, 96, !dbg !17
  %sched.n549 = icmp slt i64 %sched.n245, -96, !dbg !17
  %sched.n550 = sub nsw i64 -97, %sched.n245, !dbg !17
  %sched.n551 = select i1 %sched.n549, i64 %sched.n550, i64 %sched.n548, !dbg !17
  %sched.n552 = sdiv i64 %sched.n551, 128, !dbg !17
  %sched.n553 = sext i1 %sched.n549 to i64, !dbg !17
  %sched.n554 = xor i64 %sched.n552, %sched.n553, !dbg !17
  %sched.n555 = shl nsw i64 %sched.n554, 7, !dbg !17
  %sched.n556 = sub i64 %sched.n91, %sched.n555, !dbg !17
  %sched.n557 = shl i64 %sched.n547, 19, !dbg !17
  %sched.n558 = and i64 %sched.n557, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %sched.n559 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %sched.n558, !dbg !17
  %sched.n560 = getelementptr [2 x i8], ptr addrspace(1) %sched.n559, i64 %sched.n108, !dbg !17
  %sched.n561 = getelementptr [2 x i8], ptr addrspace(1) %sched.n560, i64 %sched.n556, !dbg !17
  %sched.n562 = getelementptr i8, ptr addrspace(1) %sched.n561, i64 192, !dbg !17
  %sched.n563 = getelementptr [2 x i8], ptr addrspace(1) %sched.n562, i64 %sched.n114, !dbg !17
  %sched.n564 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local783), !dbg !17
  %sched.n565 = zext i32 %sched.n564 to i64, !dbg !17
  %sched.n566 = inttoptr i64 %sched.n565 to ptr addrspace(3), !dbg !17
  %address.local830 = getelementptr [2 x i8], ptr addrspace(1) %sched.n563, i64 %address.local394, !dbg !17
  %sched.n567 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %sched.n568 = ptrtoint ptr addrspace(1) %address.local830 to i64, !dbg !17
  %sched.n569 = sub i64 %sched.n568, %sched.n567, !dbg !17
  %dma.buffer.offset1555 = trunc i64 %sched.n569 to i32, !dbg !17
  %sched.n570 = trunc i64 %sched.n567 to i32, !dbg !17
  %sched.n571 = insertelement <4 x i32> poison, i32 %sched.n570, i64 0, !dbg !17
  %sched.n572 = lshr i64 %sched.n567, 32, !dbg !17
  %sched.n573 = trunc i64 %sched.n572 to i32, !dbg !17
  %sched.n574 = insertelement <4 x i32> %sched.n571, i32 %sched.n573, i64 1, !dbg !17
  %sched.n575 = insertelement <4 x i32> %sched.n574, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1556 = insertelement <4 x i32> %sched.n575, i32 131072, i64 3, !dbg !17
  %sched.n576 = ptrtoint ptr addrspace(3) %sched.n566 to i32, !dbg !17
  %dma.buffer.lds1557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n576), !dbg !17
  %sched.n577 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1556) #10, !dbg !17
  %sched.n578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local787), !dbg !17
  %sched.n579 = zext i32 %sched.n578 to i64, !dbg !17
  %sched.n580 = inttoptr i64 %sched.n579 to ptr addrspace(3), !dbg !17
  %address.local831 = getelementptr [2 x i8], ptr addrspace(1) %sched.n563, i64 %address.local411, !dbg !17
  %sched.n581 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %sched.n582 = ptrtoint ptr addrspace(1) %address.local831 to i64, !dbg !17
  %sched.n583 = sub i64 %sched.n582, %sched.n581, !dbg !17
  %dma.buffer.offset1552 = trunc i64 %sched.n583 to i32, !dbg !17
  %sched.n584 = trunc i64 %sched.n581 to i32, !dbg !17
  %sched.n585 = insertelement <4 x i32> poison, i32 %sched.n584, i64 0, !dbg !17
  %sched.n586 = lshr i64 %sched.n581, 32, !dbg !17
  %sched.n587 = trunc i64 %sched.n586 to i32, !dbg !17
  %sched.n588 = insertelement <4 x i32> %sched.n585, i32 %sched.n587, i64 1, !dbg !17
  %sched.n589 = insertelement <4 x i32> %sched.n588, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1553 = insertelement <4 x i32> %sched.n589, i32 131072, i64 3, !dbg !17
  %sched.n590 = ptrtoint ptr addrspace(3) %sched.n580 to i32, !dbg !17
  %dma.buffer.lds1554 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n590), !dbg !17
  %sched.n591 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1554, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1553) #10, !dbg !17
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local371, i64 0, !dbg !11
  %sched.n592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !11
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %address.local371, i64 4096, !dbg !11
  %sched.n593 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !11
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local371, i64 1024, !dbg !11
  %sched.n594 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !11
  %sched.n595 = fmul <4 x half> %sched.n592, splat (half 0xH3015), !dbg !11
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local371, i64 5120, !dbg !11
  %sched.n596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !11
  %sched.n597 = fmul <4 x half> %sched.n593, splat (half 0xH3015), !dbg !11
  %sched.n598 = fmul <4 x half> %sched.n594, splat (half 0xH3015), !dbg !11
  %sched.n599 = fmul <4 x half> %sched.n596, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %sched.n600 = add nuw nsw i64 %sched.n108, %sched.n16, !dbg !19
  %sched.n601 = shl i64 %sched.n600, 19, !dbg !19
  %sched.n602 = and i64 %sched.n601, 16252928, !dbg !19
  %sched.n603 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %sched.n602, !dbg !19
  %sched.n604 = getelementptr [2 x i8], ptr addrspace(1) %sched.n603, i64 %sched.n114, !dbg !19
  %sched.n605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local384), !dbg !19
  %sched.n606 = zext i32 %sched.n605 to i64, !dbg !19
  %sched.n607 = inttoptr i64 %sched.n606 to ptr addrspace(3), !dbg !19
  %address.local832 = shl nuw nsw i64 %address.local379, 5, !dbg !19
  %address.local833 = and i64 %address.local832, 8064, !dbg !19
  %address.local834 = shl nuw nsw i64 %address.local379, 3, !dbg !19
  %address.local835 = and i64 %address.local834, 24, !dbg !19
  %address.local836 = or disjoint i64 %address.local833, %address.local835, !dbg !19
  %address.local837 = getelementptr [2 x i8], ptr addrspace(1) %sched.n604, i64 %address.local836, !dbg !19
  %sched.n608 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %sched.n609 = ptrtoint ptr addrspace(1) %address.local837 to i64, !dbg !19
  %sched.n610 = sub i64 %sched.n609, %sched.n608, !dbg !19
  %dma.buffer.offset1549 = trunc i64 %sched.n610 to i32, !dbg !19
  %sched.n611 = trunc i64 %sched.n608 to i32, !dbg !19
  %sched.n612 = insertelement <4 x i32> poison, i32 %sched.n611, i64 0, !dbg !19
  %sched.n613 = lshr i64 %sched.n608, 32, !dbg !19
  %sched.n614 = trunc i64 %sched.n613 to i32, !dbg !19
  %sched.n615 = insertelement <4 x i32> %sched.n612, i32 %sched.n614, i64 1, !dbg !19
  %sched.n616 = insertelement <4 x i32> %sched.n615, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1550 = insertelement <4 x i32> %sched.n616, i32 131072, i64 3, !dbg !19
  %sched.n617 = ptrtoint ptr addrspace(3) %sched.n607 to i32, !dbg !19
  %dma.buffer.lds1551 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n617), !dbg !19
  %sched.n618 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1551, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #10, !dbg !19
  %sched.n619 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local401), !dbg !19
  %sched.n620 = zext i32 %sched.n619 to i64, !dbg !19
  %sched.n621 = inttoptr i64 %sched.n620 to ptr addrspace(3), !dbg !19
  %address.local838 = shl nuw nsw i64 %address.local396, 5, !dbg !19
  %address.local839 = and i64 %address.local838, 16256, !dbg !19
  %address.local840 = shl nuw nsw i64 %address.local379, 3, !dbg !19
  %address.local841 = and i64 %address.local840, 24, !dbg !19
  %address.local842 = or disjoint i64 %address.local839, %address.local841, !dbg !19
  %address.local843 = getelementptr [2 x i8], ptr addrspace(1) %sched.n604, i64 %address.local842, !dbg !19
  %sched.n622 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %sched.n623 = ptrtoint ptr addrspace(1) %address.local843 to i64, !dbg !19
  %sched.n624 = sub i64 %sched.n623, %sched.n622, !dbg !19
  %dma.buffer.offset1546 = trunc i64 %sched.n624 to i32, !dbg !19
  %sched.n625 = trunc i64 %sched.n622 to i32, !dbg !19
  %sched.n626 = insertelement <4 x i32> poison, i32 %sched.n625, i64 0, !dbg !19
  %sched.n627 = lshr i64 %sched.n622, 32, !dbg !19
  %sched.n628 = trunc i64 %sched.n627 to i32, !dbg !19
  %sched.n629 = insertelement <4 x i32> %sched.n626, i32 %sched.n628, i64 1, !dbg !19
  %sched.n630 = insertelement <4 x i32> %sched.n629, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1547 = insertelement <4 x i32> %sched.n630, i32 131072, i64 3, !dbg !19
  %sched.n631 = ptrtoint ptr addrspace(3) %sched.n621 to i32, !dbg !19
  %dma.buffer.lds1548 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n631), !dbg !19
  %sched.n632 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1548, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1547) #10, !dbg !19
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8192, !dbg !18
  %sched.n633 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !18
  %sched.n634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n633, <4 x float> %sched.n453) #9, !dbg !18
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12288, !dbg !18
  %sched.n635 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !18
  %lds.const.gep1501 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8192, !dbg !18
  %sched.n636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1501, align 8, !dbg !18
  %lds.const.gep1502 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12288, !dbg !18
  %sched.n639 = load <4 x half>, ptr addrspace(3) %lds.const.gep1502, align 8, !dbg !18
  %sched.n637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n635, <4 x float> %sched.n634) #9, !dbg !18
  %sched.n638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n636, <4 x float> %sched.n457) #9, !dbg !18
  %lds.const.gep1503 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8320, !dbg !18
  %sched.n640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1503, align 8, !dbg !18
  %lds.const.gep1504 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12416, !dbg !18
  %sched.n643 = load <4 x half>, ptr addrspace(3) %lds.const.gep1504, align 8, !dbg !18
  %sched.n641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n639, <4 x float> %sched.n638) #9, !dbg !18
  %sched.n642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n640, <4 x float> %sched.n461) #9, !dbg !18
  %lds.const.gep1505 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8320, !dbg !18
  %sched.n644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1505, align 8, !dbg !18
  %lds.const.gep1506 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12416, !dbg !18
  %sched.n647 = load <4 x half>, ptr addrspace(3) %lds.const.gep1506, align 8, !dbg !18
  %sched.n645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n643, <4 x float> %sched.n642) #9, !dbg !18
  %sched.n646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n644, <4 x float> %sched.n465) #9, !dbg !18
  %lds.const.gep1507 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9216, !dbg !18
  %sched.n648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1507, align 8, !dbg !18
  %lds.const.gep1508 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13312, !dbg !18
  %sched.n651 = load <4 x half>, ptr addrspace(3) %lds.const.gep1508, align 8, !dbg !18
  %sched.n649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n647, <4 x float> %sched.n646) #9, !dbg !18
  %sched.n650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n648, <4 x float> %sched.n469) #9, !dbg !18
  %lds.const.gep1509 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9216, !dbg !18
  %sched.n652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1509, align 8, !dbg !18
  %lds.const.gep1510 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13312, !dbg !18
  %sched.n655 = load <4 x half>, ptr addrspace(3) %lds.const.gep1510, align 8, !dbg !18
  %sched.n653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n651, <4 x float> %sched.n650) #9, !dbg !18
  %sched.n654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n652, <4 x float> %sched.n473) #9, !dbg !18
  %lds.const.gep1511 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9344, !dbg !18
  %sched.n656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1511, align 8, !dbg !18
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13440, !dbg !18
  %sched.n659 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !18
  %sched.n657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n655, <4 x float> %sched.n654) #9, !dbg !18
  %sched.n658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n656, <4 x float> %sched.n477) #9, !dbg !18
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9344, !dbg !18
  %sched.n660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !18
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13440, !dbg !18
  %sched.n663 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !18
  %sched.n661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n659, <4 x float> %sched.n658) #9, !dbg !18
  %sched.n662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n595, <4 x half> %sched.n660, <4 x float> %sched.n481) #9, !dbg !18
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8192, !dbg !18
  %sched.n664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !18
  %sched.n665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n597, <4 x half> %sched.n663, <4 x float> %sched.n662) #9, !dbg !18
  %sched.n666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n664, <4 x float> %sched.n485) #9, !dbg !18
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12288, !dbg !18
  %sched.n667 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !18
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8192, !dbg !18
  %sched.n668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !18
  %sched.n669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n667, <4 x float> %sched.n666) #9, !dbg !18
  %sched.n670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n668, <4 x float> %sched.n489) #9, !dbg !18
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12288, !dbg !18
  %sched.n671 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !18
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8320, !dbg !18
  %sched.n672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !18
  %sched.n673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n671, <4 x float> %sched.n670) #9, !dbg !18
  %sched.n674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n672, <4 x float> %sched.n493) #9, !dbg !18
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12416, !dbg !18
  %sched.n675 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !18
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local460, i64 8320, !dbg !18
  %sched.n676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !18
  %sched.n677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n675, <4 x float> %sched.n674) #9, !dbg !18
  %sched.n678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n676, <4 x float> %sched.n497) #9, !dbg !18
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %address.local460, i64 12416, !dbg !18
  %sched.n679 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !18
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9216, !dbg !18
  %sched.n680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !18
  %sched.n681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n679, <4 x float> %sched.n678) #9, !dbg !18
  %sched.n682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n680, <4 x float> %sched.n501) #9, !dbg !18
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13312, !dbg !18
  %sched.n683 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !18
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9216, !dbg !18
  %sched.n684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !18
  %sched.n685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n683, <4 x float> %sched.n682) #9, !dbg !18
  %sched.n686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n684, <4 x float> %sched.n505) #9, !dbg !18
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13312, !dbg !18
  %sched.n687 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !18
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9344, !dbg !18
  %sched.n688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !18
  %sched.n689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n687, <4 x float> %sched.n686) #9, !dbg !18
  %sched.n690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n688, <4 x float> %sched.n509) #9, !dbg !18
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13440, !dbg !18
  %sched.n691 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !18
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local460, i64 9344, !dbg !18
  %sched.n692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !18
  %sched.n693 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n691, <4 x float> %sched.n690) #9, !dbg !18
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local460, i64 13440, !dbg !18
  %sched.n694 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %sched.n695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n598, <4 x half> %sched.n692, <4 x float> %sched.n512) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %sched.n696 = or disjoint i64 %sched.n105, 127, !dbg !14
  %sched.n697 = icmp uge i64 %sched.n64, %sched.n696, !dbg !14
  %sched.n698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %sched.n599, <4 x half> %sched.n694, <4 x float> %sched.n695) #9, !dbg !18
  %sched.n699 = and i1 %sched.n76, %sched.n697, !dbg !14
  br i1 %sched.n699, label %sched.n700, label %sched.n701, !dbg !14

sched.n700:                                              ; preds = %sched.n104
  br label %sched.n845, !dbg !14

sched.n701:                                              ; preds = %sched.n104
  %sched.n702 = add i64 %sched.n64, %sched.n32, !dbg !14
  %sched.n703 = shl nuw nsw i64 %sched.n65, 4, !dbg !14
  %sched.n704 = add i64 %sched.n702, %sched.n703, !dbg !14
  %sched.n705 = and i32 %sched.n31, 240, !dbg !14
  %index.compact = zext nneg i32 %sched.n705 to i64, !dbg !14
  %sched.n706 = sub i64 %sched.n704, %index.compact, !dbg !14
  %sched.n707 = and i64 %sched.n32, 48, !dbg !14
  %sched.n708 = or disjoint i64 %sched.n105, %sched.n707, !dbg !14
  %sched.n709 = add i64 %sched.n706, 1, !dbg !24
  %sched.n710 = lshr i64 %sched.n709, 32, !dbg !25
  %compare.high = trunc i64 %sched.n710 to i32, !dbg !25
  %sched.n711 = lshr i64 %sched.n708, 32, !dbg !25
  %compare.high1137 = trunc i64 %sched.n711 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1137, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1137, !dbg !25
  %compare.low.base = trunc i64 %sched.n709 to i32, !dbg !25
  %compare.low.base1138 = trunc i64 %sched.n708 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1138, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %sched.n712 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low1139 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result1140 = select i1 %compare.high.equal, i1 %compare.low1139, i1 %compare.high.order, !dbg !25
  %sched.n713 = select i1 %compare.result1140, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1141 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1142 = or i32 %compare.local.base1141, 2, !dbg !25
  %compare.low1143 = icmp ugt i32 %compare.low.base, %compare.local1142, !dbg !25
  %compare.result1144 = select i1 %compare.high.equal, i1 %compare.low1143, i1 %compare.high.order, !dbg !25
  %sched.n714 = select i1 %compare.result1144, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1145 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1146 = or i32 %compare.local.base1145, 3, !dbg !25
  %compare.low1147 = icmp ugt i32 %compare.low.base, %compare.local1146, !dbg !25
  %compare.result1148 = select i1 %compare.high.equal, i1 %compare.low1147, i1 %compare.high.order, !dbg !25
  %sched.n715 = select i1 %compare.result1148, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1149 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1150 = or i32 %compare.local.base1149, 4, !dbg !25
  %compare.low1151 = icmp ugt i32 %compare.low.base, %compare.local1150, !dbg !25
  %compare.result1152 = select i1 %compare.high.equal, i1 %compare.low1151, i1 %compare.high.order, !dbg !25
  %sched.n716 = select i1 %compare.result1152, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1153 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1154 = or i32 %compare.local.base1153, 5, !dbg !25
  %compare.low1155 = icmp ugt i32 %compare.low.base, %compare.local1154, !dbg !25
  %compare.result1156 = select i1 %compare.high.equal, i1 %compare.low1155, i1 %compare.high.order, !dbg !25
  %sched.n717 = select i1 %compare.result1156, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1157 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1158 = or i32 %compare.local.base1157, 6, !dbg !25
  %compare.low1159 = icmp ugt i32 %compare.low.base, %compare.local1158, !dbg !25
  %compare.result1160 = select i1 %compare.high.equal, i1 %compare.low1159, i1 %compare.high.order, !dbg !25
  %sched.n718 = select i1 %compare.result1160, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1161 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1162 = or i32 %compare.local.base1161, 7, !dbg !25
  %compare.low1163 = icmp ugt i32 %compare.low.base, %compare.local1162, !dbg !25
  %compare.result1164 = select i1 %compare.high.equal, i1 %compare.low1163, i1 %compare.high.order, !dbg !25
  %sched.n719 = select i1 %compare.result1164, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1165 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1166 = or i32 %compare.local.base1165, 8, !dbg !25
  %compare.low1167 = icmp ugt i32 %compare.low.base, %compare.local1166, !dbg !25
  %compare.result1168 = select i1 %compare.high.equal, i1 %compare.low1167, i1 %compare.high.order, !dbg !25
  %sched.n720 = select i1 %compare.result1168, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1169 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1170 = or i32 %compare.local.base1169, 9, !dbg !25
  %compare.low1171 = icmp ugt i32 %compare.low.base, %compare.local1170, !dbg !25
  %compare.result1172 = select i1 %compare.high.equal, i1 %compare.low1171, i1 %compare.high.order, !dbg !25
  %sched.n721 = select i1 %compare.result1172, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1173 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1174 = or i32 %compare.local.base1173, 10, !dbg !25
  %compare.low1175 = icmp ugt i32 %compare.low.base, %compare.local1174, !dbg !25
  %compare.result1176 = select i1 %compare.high.equal, i1 %compare.low1175, i1 %compare.high.order, !dbg !25
  %sched.n722 = select i1 %compare.result1176, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1177 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1178 = or i32 %compare.local.base1177, 11, !dbg !25
  %compare.low1179 = icmp ugt i32 %compare.low.base, %compare.local1178, !dbg !25
  %compare.result1180 = select i1 %compare.high.equal, i1 %compare.low1179, i1 %compare.high.order, !dbg !25
  %sched.n723 = select i1 %compare.result1180, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1181 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1182 = or i32 %compare.local.base1181, 12, !dbg !25
  %compare.low1183 = icmp ugt i32 %compare.low.base, %compare.local1182, !dbg !25
  %compare.result1184 = select i1 %compare.high.equal, i1 %compare.low1183, i1 %compare.high.order, !dbg !25
  %sched.n724 = select i1 %compare.result1184, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1185 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1186 = or i32 %compare.local.base1185, 13, !dbg !25
  %compare.low1187 = icmp ugt i32 %compare.low.base, %compare.local1186, !dbg !25
  %compare.result1188 = select i1 %compare.high.equal, i1 %compare.low1187, i1 %compare.high.order, !dbg !25
  %sched.n725 = select i1 %compare.result1188, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1189 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1190 = or i32 %compare.local.base1189, 14, !dbg !25
  %compare.low1191 = icmp ugt i32 %compare.low.base, %compare.local1190, !dbg !25
  %compare.result1192 = select i1 %compare.high.equal, i1 %compare.low1191, i1 %compare.high.order, !dbg !25
  %sched.n726 = select i1 %compare.result1192, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1193 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 15) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1194 = or i32 %compare.local.base1193, 15, !dbg !25
  %compare.low1195 = icmp ugt i32 %compare.low.base, %compare.local1194, !dbg !25
  %compare.result1196 = select i1 %compare.high.equal, i1 %compare.low1195, i1 %compare.high.order, !dbg !25
  %sched.n727 = select i1 %compare.result1196, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1197 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 16) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1198 = or i32 %compare.local.base1197, 64, !dbg !25
  %compare.low1199 = icmp ugt i32 %compare.low.base, %compare.local1198, !dbg !25
  %compare.result1200 = select i1 %compare.high.equal, i1 %compare.low1199, i1 %compare.high.order, !dbg !25
  %sched.n728 = select i1 %compare.result1200, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1201 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 17) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1202 = or i32 %compare.local.base1201, 65, !dbg !25
  %compare.low1203 = icmp ugt i32 %compare.low.base, %compare.local1202, !dbg !25
  %compare.result1204 = select i1 %compare.high.equal, i1 %compare.low1203, i1 %compare.high.order, !dbg !25
  %sched.n729 = select i1 %compare.result1204, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1205 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 18) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1206 = or i32 %compare.local.base1205, 66, !dbg !25
  %compare.low1207 = icmp ugt i32 %compare.low.base, %compare.local1206, !dbg !25
  %compare.result1208 = select i1 %compare.high.equal, i1 %compare.low1207, i1 %compare.high.order, !dbg !25
  %sched.n730 = select i1 %compare.result1208, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1209 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 19) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1210 = or i32 %compare.local.base1209, 67, !dbg !25
  %compare.low1211 = icmp ugt i32 %compare.low.base, %compare.local1210, !dbg !25
  %compare.result1212 = select i1 %compare.high.equal, i1 %compare.low1211, i1 %compare.high.order, !dbg !25
  %sched.n731 = select i1 %compare.result1212, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1213 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 20) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1214 = or i32 %compare.local.base1213, 68, !dbg !25
  %compare.low1215 = icmp ugt i32 %compare.low.base, %compare.local1214, !dbg !25
  %compare.result1216 = select i1 %compare.high.equal, i1 %compare.low1215, i1 %compare.high.order, !dbg !25
  %sched.n732 = select i1 %compare.result1216, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1217 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 21) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1218 = or i32 %compare.local.base1217, 69, !dbg !25
  %compare.low1219 = icmp ugt i32 %compare.low.base, %compare.local1218, !dbg !25
  %compare.result1220 = select i1 %compare.high.equal, i1 %compare.low1219, i1 %compare.high.order, !dbg !25
  %sched.n733 = select i1 %compare.result1220, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1221 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 22) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1222 = or i32 %compare.local.base1221, 70, !dbg !25
  %compare.low1223 = icmp ugt i32 %compare.low.base, %compare.local1222, !dbg !25
  %compare.result1224 = select i1 %compare.high.equal, i1 %compare.low1223, i1 %compare.high.order, !dbg !25
  %sched.n734 = select i1 %compare.result1224, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1225 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 23) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1226 = or i32 %compare.local.base1225, 71, !dbg !25
  %compare.low1227 = icmp ugt i32 %compare.low.base, %compare.local1226, !dbg !25
  %compare.result1228 = select i1 %compare.high.equal, i1 %compare.low1227, i1 %compare.high.order, !dbg !25
  %sched.n735 = select i1 %compare.result1228, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1229 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 24) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1230 = or i32 %compare.local.base1229, 72, !dbg !25
  %compare.low1231 = icmp ugt i32 %compare.low.base, %compare.local1230, !dbg !25
  %compare.result1232 = select i1 %compare.high.equal, i1 %compare.low1231, i1 %compare.high.order, !dbg !25
  %sched.n736 = select i1 %compare.result1232, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1233 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 25) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1234 = or i32 %compare.local.base1233, 73, !dbg !25
  %compare.low1235 = icmp ugt i32 %compare.low.base, %compare.local1234, !dbg !25
  %compare.result1236 = select i1 %compare.high.equal, i1 %compare.low1235, i1 %compare.high.order, !dbg !25
  %sched.n737 = select i1 %compare.result1236, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1237 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 26) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1238 = or i32 %compare.local.base1237, 74, !dbg !25
  %compare.low1239 = icmp ugt i32 %compare.low.base, %compare.local1238, !dbg !25
  %compare.result1240 = select i1 %compare.high.equal, i1 %compare.low1239, i1 %compare.high.order, !dbg !25
  %sched.n738 = select i1 %compare.result1240, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1241 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 27) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1242 = or i32 %compare.local.base1241, 75, !dbg !25
  %compare.low1243 = icmp ugt i32 %compare.low.base, %compare.local1242, !dbg !25
  %compare.result1244 = select i1 %compare.high.equal, i1 %compare.low1243, i1 %compare.high.order, !dbg !25
  %sched.n739 = select i1 %compare.result1244, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1245 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 28) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1246 = or i32 %compare.local.base1245, 76, !dbg !25
  %compare.low1247 = icmp ugt i32 %compare.low.base, %compare.local1246, !dbg !25
  %compare.result1248 = select i1 %compare.high.equal, i1 %compare.low1247, i1 %compare.high.order, !dbg !25
  %sched.n740 = select i1 %compare.result1248, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1249 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 29) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1250 = or i32 %compare.local.base1249, 77, !dbg !25
  %compare.low1251 = icmp ugt i32 %compare.low.base, %compare.local1250, !dbg !25
  %compare.result1252 = select i1 %compare.high.equal, i1 %compare.low1251, i1 %compare.high.order, !dbg !25
  %sched.n741 = select i1 %compare.result1252, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1253 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 30) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1254 = or i32 %compare.local.base1253, 78, !dbg !25
  %compare.low1255 = icmp ugt i32 %compare.low.base, %compare.local1254, !dbg !25
  %compare.result1256 = select i1 %compare.high.equal, i1 %compare.low1255, i1 %compare.high.order, !dbg !25
  %sched.n742 = select i1 %compare.result1256, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1257 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 31) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1258 = or i32 %compare.local.base1257, 79, !dbg !25
  %compare.low1259 = icmp ugt i32 %compare.low.base, %compare.local1258, !dbg !25
  %compare.result1260 = select i1 %compare.high.equal, i1 %compare.low1259, i1 %compare.high.order, !dbg !25
  %sched.n743 = select i1 %compare.result1260, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %sched.n744 = add i64 %sched.n706, 65, !dbg !24
  %sched.n745 = lshr i64 %sched.n744, 32, !dbg !25
  %compare.high1261 = trunc i64 %sched.n745 to i32, !dbg !25
  %sched.n746 = lshr i64 %sched.n708, 32, !dbg !25
  %compare.high1262 = trunc i64 %sched.n746 to i32, !dbg !25
  %compare.high.equal1263 = icmp eq i32 %compare.high1261, %compare.high1262, !dbg !25
  %compare.high.order1264 = icmp ugt i32 %compare.high1261, %compare.high1262, !dbg !25
  %compare.low.base1265 = trunc i64 %sched.n744 to i32, !dbg !25
  %compare.low.base1266 = trunc i64 %sched.n708 to i32, !dbg !25
  %compare.low1267 = icmp ugt i32 %compare.low.base1265, %compare.low.base1266, !dbg !25
  %compare.result1268 = select i1 %compare.high.equal1263, i1 %compare.low1267, i1 %compare.high.order1264, !dbg !25
  %sched.n747 = select i1 %compare.result1268, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1269 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 32) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1270 = or i32 %compare.local.base1269, 1, !dbg !25
  %compare.low1271 = icmp ugt i32 %compare.low.base1265, %compare.local1270, !dbg !25
  %compare.result1272 = select i1 %compare.high.equal1263, i1 %compare.low1271, i1 %compare.high.order1264, !dbg !25
  %sched.n748 = select i1 %compare.result1272, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 33) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1274 = or i32 %compare.local.base1273, 2, !dbg !25
  %compare.low1275 = icmp ugt i32 %compare.low.base1265, %compare.local1274, !dbg !25
  %compare.result1276 = select i1 %compare.high.equal1263, i1 %compare.low1275, i1 %compare.high.order1264, !dbg !25
  %sched.n749 = select i1 %compare.result1276, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 34) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1278 = or i32 %compare.local.base1277, 3, !dbg !25
  %compare.low1279 = icmp ugt i32 %compare.low.base1265, %compare.local1278, !dbg !25
  %compare.result1280 = select i1 %compare.high.equal1263, i1 %compare.low1279, i1 %compare.high.order1264, !dbg !25
  %sched.n750 = select i1 %compare.result1280, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1281 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 35) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1282 = or i32 %compare.local.base1281, 4, !dbg !25
  %compare.low1283 = icmp ugt i32 %compare.low.base1265, %compare.local1282, !dbg !25
  %compare.result1284 = select i1 %compare.high.equal1263, i1 %compare.low1283, i1 %compare.high.order1264, !dbg !25
  %sched.n751 = select i1 %compare.result1284, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1285 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 36) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1286 = or i32 %compare.local.base1285, 5, !dbg !25
  %compare.low1287 = icmp ugt i32 %compare.low.base1265, %compare.local1286, !dbg !25
  %compare.result1288 = select i1 %compare.high.equal1263, i1 %compare.low1287, i1 %compare.high.order1264, !dbg !25
  %sched.n752 = select i1 %compare.result1288, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 37) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1290 = or i32 %compare.local.base1289, 6, !dbg !25
  %compare.low1291 = icmp ugt i32 %compare.low.base1265, %compare.local1290, !dbg !25
  %compare.result1292 = select i1 %compare.high.equal1263, i1 %compare.low1291, i1 %compare.high.order1264, !dbg !25
  %sched.n753 = select i1 %compare.result1292, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 38) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1294 = or i32 %compare.local.base1293, 7, !dbg !25
  %compare.low1295 = icmp ugt i32 %compare.low.base1265, %compare.local1294, !dbg !25
  %compare.result1296 = select i1 %compare.high.equal1263, i1 %compare.low1295, i1 %compare.high.order1264, !dbg !25
  %sched.n754 = select i1 %compare.result1296, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 39) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1298 = or i32 %compare.local.base1297, 8, !dbg !25
  %compare.low1299 = icmp ugt i32 %compare.low.base1265, %compare.local1298, !dbg !25
  %compare.result1300 = select i1 %compare.high.equal1263, i1 %compare.low1299, i1 %compare.high.order1264, !dbg !25
  %sched.n755 = select i1 %compare.result1300, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 40) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1302 = or i32 %compare.local.base1301, 9, !dbg !25
  %compare.low1303 = icmp ugt i32 %compare.low.base1265, %compare.local1302, !dbg !25
  %compare.result1304 = select i1 %compare.high.equal1263, i1 %compare.low1303, i1 %compare.high.order1264, !dbg !25
  %sched.n756 = select i1 %compare.result1304, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 41) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1306 = or i32 %compare.local.base1305, 10, !dbg !25
  %compare.low1307 = icmp ugt i32 %compare.low.base1265, %compare.local1306, !dbg !25
  %compare.result1308 = select i1 %compare.high.equal1263, i1 %compare.low1307, i1 %compare.high.order1264, !dbg !25
  %sched.n757 = select i1 %compare.result1308, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 42) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1310 = or i32 %compare.local.base1309, 11, !dbg !25
  %compare.low1311 = icmp ugt i32 %compare.low.base1265, %compare.local1310, !dbg !25
  %compare.result1312 = select i1 %compare.high.equal1263, i1 %compare.low1311, i1 %compare.high.order1264, !dbg !25
  %sched.n758 = select i1 %compare.result1312, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 43) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1314 = or i32 %compare.local.base1313, 12, !dbg !25
  %compare.low1315 = icmp ugt i32 %compare.low.base1265, %compare.local1314, !dbg !25
  %compare.result1316 = select i1 %compare.high.equal1263, i1 %compare.low1315, i1 %compare.high.order1264, !dbg !25
  %sched.n759 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 44) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 13, !dbg !25
  %compare.low1319 = icmp ugt i32 %compare.low.base1265, %compare.local1318, !dbg !25
  %compare.result1320 = select i1 %compare.high.equal1263, i1 %compare.low1319, i1 %compare.high.order1264, !dbg !25
  %sched.n760 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 45) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 14, !dbg !25
  %compare.low1323 = icmp ugt i32 %compare.low.base1265, %compare.local1322, !dbg !25
  %compare.result1324 = select i1 %compare.high.equal1263, i1 %compare.low1323, i1 %compare.high.order1264, !dbg !25
  %sched.n761 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 46) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 15, !dbg !25
  %compare.low1327 = icmp ugt i32 %compare.low.base1265, %compare.local1326, !dbg !25
  %compare.result1328 = select i1 %compare.high.equal1263, i1 %compare.low1327, i1 %compare.high.order1264, !dbg !25
  %sched.n762 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 47) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 64, !dbg !25
  %compare.low1331 = icmp ugt i32 %compare.low.base1265, %compare.local1330, !dbg !25
  %compare.result1332 = select i1 %compare.high.equal1263, i1 %compare.low1331, i1 %compare.high.order1264, !dbg !25
  %sched.n763 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 48) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 65, !dbg !25
  %compare.low1335 = icmp ugt i32 %compare.low.base1265, %compare.local1334, !dbg !25
  %compare.result1336 = select i1 %compare.high.equal1263, i1 %compare.low1335, i1 %compare.high.order1264, !dbg !25
  %sched.n764 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 49) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 66, !dbg !25
  %compare.low1339 = icmp ugt i32 %compare.low.base1265, %compare.local1338, !dbg !25
  %compare.result1340 = select i1 %compare.high.equal1263, i1 %compare.low1339, i1 %compare.high.order1264, !dbg !25
  %sched.n765 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 50) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 67, !dbg !25
  %compare.low1343 = icmp ugt i32 %compare.low.base1265, %compare.local1342, !dbg !25
  %compare.result1344 = select i1 %compare.high.equal1263, i1 %compare.low1343, i1 %compare.high.order1264, !dbg !25
  %sched.n766 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 51) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 68, !dbg !25
  %compare.low1347 = icmp ugt i32 %compare.low.base1265, %compare.local1346, !dbg !25
  %compare.result1348 = select i1 %compare.high.equal1263, i1 %compare.low1347, i1 %compare.high.order1264, !dbg !25
  %sched.n767 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 52) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 69, !dbg !25
  %compare.low1351 = icmp ugt i32 %compare.low.base1265, %compare.local1350, !dbg !25
  %compare.result1352 = select i1 %compare.high.equal1263, i1 %compare.low1351, i1 %compare.high.order1264, !dbg !25
  %sched.n768 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 53) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 70, !dbg !25
  %compare.low1355 = icmp ugt i32 %compare.low.base1265, %compare.local1354, !dbg !25
  %compare.result1356 = select i1 %compare.high.equal1263, i1 %compare.low1355, i1 %compare.high.order1264, !dbg !25
  %sched.n769 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 54) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 71, !dbg !25
  %compare.low1359 = icmp ugt i32 %compare.low.base1265, %compare.local1358, !dbg !25
  %compare.result1360 = select i1 %compare.high.equal1263, i1 %compare.low1359, i1 %compare.high.order1264, !dbg !25
  %sched.n770 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 55) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 72, !dbg !25
  %compare.low1363 = icmp ugt i32 %compare.low.base1265, %compare.local1362, !dbg !25
  %compare.result1364 = select i1 %compare.high.equal1263, i1 %compare.low1363, i1 %compare.high.order1264, !dbg !25
  %sched.n771 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 56) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 73, !dbg !25
  %compare.low1367 = icmp ugt i32 %compare.low.base1265, %compare.local1366, !dbg !25
  %compare.result1368 = select i1 %compare.high.equal1263, i1 %compare.low1367, i1 %compare.high.order1264, !dbg !25
  %sched.n772 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 57) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 74, !dbg !25
  %compare.low1371 = icmp ugt i32 %compare.low.base1265, %compare.local1370, !dbg !25
  %compare.result1372 = select i1 %compare.high.equal1263, i1 %compare.low1371, i1 %compare.high.order1264, !dbg !25
  %sched.n773 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 58) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 75, !dbg !25
  %compare.low1375 = icmp ugt i32 %compare.low.base1265, %compare.local1374, !dbg !25
  %compare.result1376 = select i1 %compare.high.equal1263, i1 %compare.low1375, i1 %compare.high.order1264, !dbg !25
  %sched.n774 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 59) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 76, !dbg !25
  %compare.low1379 = icmp ugt i32 %compare.low.base1265, %compare.local1378, !dbg !25
  %compare.result1380 = select i1 %compare.high.equal1263, i1 %compare.low1379, i1 %compare.high.order1264, !dbg !25
  %sched.n775 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 60) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 77, !dbg !25
  %compare.low1383 = icmp ugt i32 %compare.low.base1265, %compare.local1382, !dbg !25
  %compare.result1384 = select i1 %compare.high.equal1263, i1 %compare.low1383, i1 %compare.high.order1264, !dbg !25
  %sched.n776 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 61) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 78, !dbg !25
  %compare.low1387 = icmp ugt i32 %compare.low.base1265, %compare.local1386, !dbg !25
  %compare.result1388 = select i1 %compare.high.equal1263, i1 %compare.low1387, i1 %compare.high.order1264, !dbg !25
  %sched.n777 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1266, i32 62) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 79, !dbg !25
  %compare.low1391 = icmp ugt i32 %compare.low.base1265, %compare.local1390, !dbg !25
  %compare.result1392 = select i1 %compare.high.equal1263, i1 %compare.low1391, i1 %compare.high.order1264, !dbg !25
  %sched.n778 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %sched.n779 = insertelement <32 x float> poison, float %sched.n712, i64 0, !dbg !14
  %sched.n780 = insertelement <32 x float> %sched.n779, float %sched.n713, i64 1, !dbg !14
  %sched.n781 = insertelement <32 x float> %sched.n780, float %sched.n714, i64 2, !dbg !14
  %sched.n782 = insertelement <32 x float> %sched.n781, float %sched.n715, i64 3, !dbg !14
  %sched.n783 = insertelement <32 x float> %sched.n782, float %sched.n716, i64 4, !dbg !14
  %sched.n784 = insertelement <32 x float> %sched.n783, float %sched.n717, i64 5, !dbg !14
  %sched.n785 = insertelement <32 x float> %sched.n784, float %sched.n718, i64 6, !dbg !14
  %sched.n786 = insertelement <32 x float> %sched.n785, float %sched.n719, i64 7, !dbg !14
  %sched.n787 = insertelement <32 x float> %sched.n786, float %sched.n720, i64 8, !dbg !14
  %sched.n788 = insertelement <32 x float> %sched.n787, float %sched.n721, i64 9, !dbg !14
  %sched.n789 = insertelement <32 x float> %sched.n788, float %sched.n722, i64 10, !dbg !14
  %sched.n790 = insertelement <32 x float> %sched.n789, float %sched.n723, i64 11, !dbg !14
  %sched.n791 = insertelement <32 x float> %sched.n790, float %sched.n724, i64 12, !dbg !14
  %sched.n792 = insertelement <32 x float> %sched.n791, float %sched.n725, i64 13, !dbg !14
  %sched.n793 = insertelement <32 x float> %sched.n792, float %sched.n726, i64 14, !dbg !14
  %sched.n794 = insertelement <32 x float> %sched.n793, float %sched.n727, i64 15, !dbg !14
  %sched.n795 = insertelement <32 x float> %sched.n794, float %sched.n728, i64 16, !dbg !14
  %sched.n796 = insertelement <32 x float> %sched.n795, float %sched.n729, i64 17, !dbg !14
  %sched.n797 = insertelement <32 x float> %sched.n796, float %sched.n730, i64 18, !dbg !14
  %sched.n798 = insertelement <32 x float> %sched.n797, float %sched.n731, i64 19, !dbg !14
  %sched.n799 = insertelement <32 x float> %sched.n798, float %sched.n732, i64 20, !dbg !14
  %sched.n800 = insertelement <32 x float> %sched.n799, float %sched.n733, i64 21, !dbg !14
  %sched.n801 = insertelement <32 x float> %sched.n800, float %sched.n734, i64 22, !dbg !14
  %sched.n802 = insertelement <32 x float> %sched.n801, float %sched.n735, i64 23, !dbg !14
  %sched.n803 = insertelement <32 x float> %sched.n802, float %sched.n736, i64 24, !dbg !14
  %sched.n804 = insertelement <32 x float> %sched.n803, float %sched.n737, i64 25, !dbg !14
  %sched.n805 = insertelement <32 x float> %sched.n804, float %sched.n738, i64 26, !dbg !14
  %sched.n806 = insertelement <32 x float> %sched.n805, float %sched.n739, i64 27, !dbg !14
  %sched.n807 = insertelement <32 x float> %sched.n806, float %sched.n740, i64 28, !dbg !14
  %sched.n808 = insertelement <32 x float> %sched.n807, float %sched.n741, i64 29, !dbg !14
  %sched.n809 = insertelement <32 x float> %sched.n808, float %sched.n742, i64 30, !dbg !14
  %sched.n810 = insertelement <32 x float> %sched.n809, float %sched.n743, i64 31, !dbg !14
  %sched.n811 = insertvalue [2 x <32 x float>] poison, <32 x float> %sched.n810, 0, !dbg !14
  %sched.n812 = insertelement <32 x float> poison, float %sched.n747, i64 0, !dbg !14
  %sched.n813 = insertelement <32 x float> %sched.n812, float %sched.n748, i64 1, !dbg !14
  %sched.n814 = insertelement <32 x float> %sched.n813, float %sched.n749, i64 2, !dbg !14
  %sched.n815 = insertelement <32 x float> %sched.n814, float %sched.n750, i64 3, !dbg !14
  %sched.n816 = insertelement <32 x float> %sched.n815, float %sched.n751, i64 4, !dbg !14
  %sched.n817 = insertelement <32 x float> %sched.n816, float %sched.n752, i64 5, !dbg !14
  %sched.n818 = insertelement <32 x float> %sched.n817, float %sched.n753, i64 6, !dbg !14
  %sched.n819 = insertelement <32 x float> %sched.n818, float %sched.n754, i64 7, !dbg !14
  %sched.n820 = insertelement <32 x float> %sched.n819, float %sched.n755, i64 8, !dbg !14
  %sched.n821 = insertelement <32 x float> %sched.n820, float %sched.n756, i64 9, !dbg !14
  %sched.n822 = insertelement <32 x float> %sched.n821, float %sched.n757, i64 10, !dbg !14
  %sched.n823 = insertelement <32 x float> %sched.n822, float %sched.n758, i64 11, !dbg !14
  %sched.n824 = insertelement <32 x float> %sched.n823, float %sched.n759, i64 12, !dbg !14
  %sched.n825 = insertelement <32 x float> %sched.n824, float %sched.n760, i64 13, !dbg !14
  %sched.n826 = insertelement <32 x float> %sched.n825, float %sched.n761, i64 14, !dbg !14
  %sched.n827 = insertelement <32 x float> %sched.n826, float %sched.n762, i64 15, !dbg !14
  %sched.n828 = insertelement <32 x float> %sched.n827, float %sched.n763, i64 16, !dbg !14
  %sched.n829 = insertelement <32 x float> %sched.n828, float %sched.n764, i64 17, !dbg !14
  %sched.n830 = insertelement <32 x float> %sched.n829, float %sched.n765, i64 18, !dbg !14
  %sched.n831 = insertelement <32 x float> %sched.n830, float %sched.n766, i64 19, !dbg !14
  %sched.n832 = insertelement <32 x float> %sched.n831, float %sched.n767, i64 20, !dbg !14
  %sched.n833 = insertelement <32 x float> %sched.n832, float %sched.n768, i64 21, !dbg !14
  %sched.n834 = insertelement <32 x float> %sched.n833, float %sched.n769, i64 22, !dbg !14
  %sched.n835 = insertelement <32 x float> %sched.n834, float %sched.n770, i64 23, !dbg !14
  %sched.n836 = insertelement <32 x float> %sched.n835, float %sched.n771, i64 24, !dbg !14
  %sched.n837 = insertelement <32 x float> %sched.n836, float %sched.n772, i64 25, !dbg !14
  %sched.n838 = insertelement <32 x float> %sched.n837, float %sched.n773, i64 26, !dbg !14
  %sched.n839 = insertelement <32 x float> %sched.n838, float %sched.n774, i64 27, !dbg !14
  %sched.n840 = insertelement <32 x float> %sched.n839, float %sched.n775, i64 28, !dbg !14
  %sched.n841 = insertelement <32 x float> %sched.n840, float %sched.n776, i64 29, !dbg !14
  %sched.n842 = insertelement <32 x float> %sched.n841, float %sched.n777, i64 30, !dbg !14
  %sched.n843 = insertelement <32 x float> %sched.n842, float %sched.n778, i64 31, !dbg !14
  %sched.n844 = insertvalue [2 x <32 x float>] %sched.n811, <32 x float> %sched.n843, 1, !dbg !14
  br label %sched.n845, !dbg !14

sched.n845:                                              ; preds = %sched.n700, %sched.n701
  %sched.n846 = phi [2 x <32 x float>] [ %sched.n844, %sched.n701 ], [ zeroinitializer, %sched.n700 ], !dbg !14
  br label %sched.n847, !dbg !14

sched.n847:                                              ; preds = %sched.n845
  %sched.n848 = extractvalue [2 x <32 x float>] %sched.n846, 0, !dbg !27
  %sched.n849 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %sched.n850 = fadd <4 x float> %sched.n637, %sched.n849, !dbg !27
  %sched.n851 = extractelement <4 x float> %sched.n850, i64 0, !dbg !28
  %sched.n852 = call float @llvm.amdgcn.exp2.f32(float %sched.n851), !dbg !28
  %sched.n853 = insertelement <4 x float> poison, float %sched.n852, i64 0, !dbg !30
  %sched.n854 = extractelement <4 x float> %sched.n850, i64 1, !dbg !28
  %sched.n855 = call float @llvm.amdgcn.exp2.f32(float %sched.n854), !dbg !28
  %sched.n856 = insertelement <4 x float> %sched.n853, float %sched.n855, i64 1, !dbg !30
  %sched.n857 = fadd float %sched.n852, %sched.n855, !dbg !29
  %sched.n858 = extractelement <4 x float> %sched.n850, i64 2, !dbg !28
  %sched.n859 = call float @llvm.amdgcn.exp2.f32(float %sched.n858), !dbg !28
  %sched.n860 = insertelement <4 x float> %sched.n856, float %sched.n859, i64 2, !dbg !30
  %sched.n861 = fadd float %sched.n857, %sched.n859, !dbg !29
  %sched.n862 = extractelement <4 x float> %sched.n850, i64 3, !dbg !28
  %sched.n863 = call float @llvm.amdgcn.exp2.f32(float %sched.n862), !dbg !28
  %sched.n864 = insertelement <4 x float> %sched.n860, float %sched.n863, i64 3, !dbg !30
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n864) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n865 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !30
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n865) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n866 = fadd float %sched.n861, %sched.n863, !dbg !29
  %sched.n867 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %sched.n868 = fadd <4 x float> %sched.n641, %sched.n867, !dbg !27
  %sched.n869 = extractelement <4 x float> %sched.n868, i64 0, !dbg !28
  %sched.n870 = call float @llvm.amdgcn.exp2.f32(float %sched.n869), !dbg !28
  %sched.n871 = insertelement <4 x float> poison, float %sched.n870, i64 0, !dbg !30
  %sched.n872 = fadd float %sched.n866, %sched.n870, !dbg !29
  %sched.n873 = extractelement <4 x float> %sched.n868, i64 1, !dbg !28
  %sched.n874 = call float @llvm.amdgcn.exp2.f32(float %sched.n873), !dbg !28
  %sched.n875 = insertelement <4 x float> %sched.n871, float %sched.n874, i64 1, !dbg !30
  %sched.n876 = fadd float %sched.n872, %sched.n874, !dbg !29
  %sched.n877 = extractelement <4 x float> %sched.n868, i64 2, !dbg !28
  %sched.n878 = call float @llvm.amdgcn.exp2.f32(float %sched.n877), !dbg !28
  %sched.n879 = insertelement <4 x float> %sched.n875, float %sched.n878, i64 2, !dbg !30
  %sched.n880 = fadd float %sched.n876, %sched.n878, !dbg !29
  %sched.n881 = extractelement <4 x float> %sched.n868, i64 3, !dbg !28
  %sched.n882 = call float @llvm.amdgcn.exp2.f32(float %sched.n881), !dbg !28
  %sched.n883 = insertelement <4 x float> %sched.n879, float %sched.n882, i64 3, !dbg !30
  %math.result306 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n883) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n884 = fptrunc <4 x float> %math.result306 to <4 x half>, !dbg !30
  %math.packed307 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n884) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n885 = fadd float %sched.n880, %sched.n882, !dbg !29
  %sched.n886 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %sched.n887 = fadd <4 x float> %sched.n645, %sched.n886, !dbg !27
  %sched.n888 = extractelement <4 x float> %sched.n887, i64 0, !dbg !28
  %sched.n889 = call float @llvm.amdgcn.exp2.f32(float %sched.n888), !dbg !28
  %sched.n890 = insertelement <4 x float> poison, float %sched.n889, i64 0, !dbg !30
  %sched.n891 = fadd float %sched.n885, %sched.n889, !dbg !29
  %sched.n892 = extractelement <4 x float> %sched.n887, i64 1, !dbg !28
  %sched.n893 = call float @llvm.amdgcn.exp2.f32(float %sched.n892), !dbg !28
  %sched.n894 = insertelement <4 x float> %sched.n890, float %sched.n893, i64 1, !dbg !30
  %sched.n895 = fadd float %sched.n891, %sched.n893, !dbg !29
  %sched.n896 = extractelement <4 x float> %sched.n887, i64 2, !dbg !28
  %sched.n897 = call float @llvm.amdgcn.exp2.f32(float %sched.n896), !dbg !28
  %sched.n898 = insertelement <4 x float> %sched.n894, float %sched.n897, i64 2, !dbg !30
  %sched.n899 = fadd float %sched.n895, %sched.n897, !dbg !29
  %sched.n900 = extractelement <4 x float> %sched.n887, i64 3, !dbg !28
  %sched.n901 = call float @llvm.amdgcn.exp2.f32(float %sched.n900), !dbg !28
  %sched.n902 = insertelement <4 x float> %sched.n898, float %sched.n901, i64 3, !dbg !30
  %math.result308 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n902) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n903 = fptrunc <4 x float> %math.result308 to <4 x half>, !dbg !30
  %math.packed309 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n903) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n904 = fadd float %sched.n899, %sched.n901, !dbg !29
  %sched.n905 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %sched.n906 = fadd <4 x float> %sched.n649, %sched.n905, !dbg !27
  %sched.n907 = extractelement <4 x float> %sched.n906, i64 0, !dbg !28
  %sched.n908 = call float @llvm.amdgcn.exp2.f32(float %sched.n907), !dbg !28
  %sched.n909 = insertelement <4 x float> poison, float %sched.n908, i64 0, !dbg !30
  %sched.n910 = fadd float %sched.n904, %sched.n908, !dbg !29
  %sched.n911 = extractelement <4 x float> %sched.n906, i64 1, !dbg !28
  %sched.n912 = call float @llvm.amdgcn.exp2.f32(float %sched.n911), !dbg !28
  %sched.n913 = insertelement <4 x float> %sched.n909, float %sched.n912, i64 1, !dbg !30
  %sched.n914 = fadd float %sched.n910, %sched.n912, !dbg !29
  %sched.n915 = extractelement <4 x float> %sched.n906, i64 2, !dbg !28
  %sched.n916 = call float @llvm.amdgcn.exp2.f32(float %sched.n915), !dbg !28
  %sched.n917 = insertelement <4 x float> %sched.n913, float %sched.n916, i64 2, !dbg !30
  %sched.n918 = fadd float %sched.n914, %sched.n916, !dbg !29
  %sched.n919 = extractelement <4 x float> %sched.n906, i64 3, !dbg !28
  %sched.n920 = call float @llvm.amdgcn.exp2.f32(float %sched.n919), !dbg !28
  %sched.n921 = insertelement <4 x float> %sched.n917, float %sched.n920, i64 3, !dbg !30
  %math.result310 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n921) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n922 = fptrunc <4 x float> %math.result310 to <4 x half>, !dbg !30
  %math.packed311 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n922) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n923 = fadd float %sched.n918, %sched.n920, !dbg !29
  %sched.n924 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %sched.n925 = fadd <4 x float> %sched.n653, %sched.n924, !dbg !27
  %sched.n926 = extractelement <4 x float> %sched.n925, i64 0, !dbg !28
  %sched.n927 = call float @llvm.amdgcn.exp2.f32(float %sched.n926), !dbg !28
  %sched.n928 = insertelement <4 x float> poison, float %sched.n927, i64 0, !dbg !30
  %sched.n929 = fadd float %sched.n923, %sched.n927, !dbg !29
  %sched.n930 = extractelement <4 x float> %sched.n925, i64 1, !dbg !28
  %sched.n931 = call float @llvm.amdgcn.exp2.f32(float %sched.n930), !dbg !28
  %sched.n932 = insertelement <4 x float> %sched.n928, float %sched.n931, i64 1, !dbg !30
  %sched.n933 = fadd float %sched.n929, %sched.n931, !dbg !29
  %sched.n934 = extractelement <4 x float> %sched.n925, i64 2, !dbg !28
  %sched.n935 = call float @llvm.amdgcn.exp2.f32(float %sched.n934), !dbg !28
  %sched.n936 = insertelement <4 x float> %sched.n932, float %sched.n935, i64 2, !dbg !30
  %sched.n937 = fadd float %sched.n933, %sched.n935, !dbg !29
  %sched.n938 = extractelement <4 x float> %sched.n925, i64 3, !dbg !28
  %sched.n939 = call float @llvm.amdgcn.exp2.f32(float %sched.n938), !dbg !28
  %sched.n940 = insertelement <4 x float> %sched.n936, float %sched.n939, i64 3, !dbg !30
  %math.result312 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n940) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n941 = fptrunc <4 x float> %math.result312 to <4 x half>, !dbg !30
  %math.packed313 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n941) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n942 = fadd float %sched.n937, %sched.n939, !dbg !29
  %sched.n943 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %sched.n944 = fadd <4 x float> %sched.n657, %sched.n943, !dbg !27
  %sched.n945 = extractelement <4 x float> %sched.n944, i64 0, !dbg !28
  %sched.n946 = call float @llvm.amdgcn.exp2.f32(float %sched.n945), !dbg !28
  %sched.n947 = insertelement <4 x float> poison, float %sched.n946, i64 0, !dbg !30
  %sched.n948 = fadd float %sched.n942, %sched.n946, !dbg !29
  %sched.n949 = extractelement <4 x float> %sched.n944, i64 1, !dbg !28
  %sched.n950 = call float @llvm.amdgcn.exp2.f32(float %sched.n949), !dbg !28
  %sched.n951 = insertelement <4 x float> %sched.n947, float %sched.n950, i64 1, !dbg !30
  %sched.n952 = fadd float %sched.n948, %sched.n950, !dbg !29
  %sched.n953 = extractelement <4 x float> %sched.n944, i64 2, !dbg !28
  %sched.n954 = call float @llvm.amdgcn.exp2.f32(float %sched.n953), !dbg !28
  %sched.n955 = insertelement <4 x float> %sched.n951, float %sched.n954, i64 2, !dbg !30
  %sched.n956 = fadd float %sched.n952, %sched.n954, !dbg !29
  %sched.n957 = extractelement <4 x float> %sched.n944, i64 3, !dbg !28
  %sched.n958 = call float @llvm.amdgcn.exp2.f32(float %sched.n957), !dbg !28
  %sched.n959 = insertelement <4 x float> %sched.n955, float %sched.n958, i64 3, !dbg !30
  %math.result314 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n959) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n960 = fptrunc <4 x float> %math.result314 to <4 x half>, !dbg !30
  %math.packed315 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n960) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n961 = fadd float %sched.n956, %sched.n958, !dbg !29
  %sched.n962 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %sched.n963 = fadd <4 x float> %sched.n661, %sched.n962, !dbg !27
  %sched.n964 = extractelement <4 x float> %sched.n963, i64 0, !dbg !28
  %sched.n965 = call float @llvm.amdgcn.exp2.f32(float %sched.n964), !dbg !28
  %sched.n966 = insertelement <4 x float> poison, float %sched.n965, i64 0, !dbg !30
  %sched.n967 = fadd float %sched.n961, %sched.n965, !dbg !29
  %sched.n968 = extractelement <4 x float> %sched.n963, i64 1, !dbg !28
  %sched.n969 = call float @llvm.amdgcn.exp2.f32(float %sched.n968), !dbg !28
  %sched.n970 = insertelement <4 x float> %sched.n966, float %sched.n969, i64 1, !dbg !30
  %sched.n971 = fadd float %sched.n967, %sched.n969, !dbg !29
  %sched.n972 = extractelement <4 x float> %sched.n963, i64 2, !dbg !28
  %sched.n973 = call float @llvm.amdgcn.exp2.f32(float %sched.n972), !dbg !28
  %sched.n974 = insertelement <4 x float> %sched.n970, float %sched.n973, i64 2, !dbg !30
  %sched.n975 = fadd float %sched.n971, %sched.n973, !dbg !29
  %sched.n976 = extractelement <4 x float> %sched.n963, i64 3, !dbg !28
  %sched.n977 = call float @llvm.amdgcn.exp2.f32(float %sched.n976), !dbg !28
  %sched.n978 = insertelement <4 x float> %sched.n974, float %sched.n977, i64 3, !dbg !30
  %math.result316 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n978) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n979 = fptrunc <4 x float> %math.result316 to <4 x half>, !dbg !30
  %math.packed317 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n979) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n980 = fadd float %sched.n975, %sched.n977, !dbg !29
  %sched.n981 = shufflevector <32 x float> %sched.n848, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %sched.n982 = fadd <4 x float> %sched.n665, %sched.n981, !dbg !27
  %sched.n983 = extractelement <4 x float> %sched.n982, i64 0, !dbg !28
  %sched.n984 = call float @llvm.amdgcn.exp2.f32(float %sched.n983), !dbg !28
  %sched.n985 = insertelement <4 x float> poison, float %sched.n984, i64 0, !dbg !30
  %sched.n986 = fadd float %sched.n980, %sched.n984, !dbg !29
  %sched.n987 = extractelement <4 x float> %sched.n982, i64 1, !dbg !28
  %sched.n988 = call float @llvm.amdgcn.exp2.f32(float %sched.n987), !dbg !28
  %sched.n989 = insertelement <4 x float> %sched.n985, float %sched.n988, i64 1, !dbg !30
  %sched.n990 = fadd float %sched.n986, %sched.n988, !dbg !29
  %sched.n991 = extractelement <4 x float> %sched.n982, i64 2, !dbg !28
  %sched.n992 = call float @llvm.amdgcn.exp2.f32(float %sched.n991), !dbg !28
  %sched.n993 = insertelement <4 x float> %sched.n989, float %sched.n992, i64 2, !dbg !30
  %sched.n994 = fadd float %sched.n990, %sched.n992, !dbg !29
  %sched.n995 = extractelement <4 x float> %sched.n982, i64 3, !dbg !28
  %sched.n996 = call float @llvm.amdgcn.exp2.f32(float %sched.n995), !dbg !28
  %sched.n997 = insertelement <4 x float> %sched.n993, float %sched.n996, i64 3, !dbg !30
  %math.result318 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n997) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n998 = fptrunc <4 x float> %math.result318 to <4 x half>, !dbg !30
  %math.packed319 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n998) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n999 = fadd float %sched.n994, %sched.n996, !dbg !29
  %sched.n1000 = extractvalue [2 x <32 x float>] %sched.n846, 1, !dbg !27
  %sched.n1001 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %sched.n1002 = fadd <4 x float> %sched.n669, %sched.n1001, !dbg !27
  %sched.n1003 = extractelement <4 x float> %sched.n1002, i64 0, !dbg !28
  %sched.n1004 = call float @llvm.amdgcn.exp2.f32(float %sched.n1003), !dbg !28
  %sched.n1005 = insertelement <4 x float> poison, float %sched.n1004, i64 0, !dbg !30
  %sched.n1006 = extractelement <4 x float> %sched.n1002, i64 1, !dbg !28
  %sched.n1007 = call float @llvm.amdgcn.exp2.f32(float %sched.n1006), !dbg !28
  %sched.n1008 = insertelement <4 x float> %sched.n1005, float %sched.n1007, i64 1, !dbg !30
  %sched.n1009 = fadd float %sched.n1004, %sched.n1007, !dbg !29
  %sched.n1010 = extractelement <4 x float> %sched.n1002, i64 2, !dbg !28
  %sched.n1011 = call float @llvm.amdgcn.exp2.f32(float %sched.n1010), !dbg !28
  %sched.n1012 = insertelement <4 x float> %sched.n1008, float %sched.n1011, i64 2, !dbg !30
  %sched.n1013 = fadd float %sched.n1009, %sched.n1011, !dbg !29
  %sched.n1014 = extractelement <4 x float> %sched.n1002, i64 3, !dbg !28
  %sched.n1015 = call float @llvm.amdgcn.exp2.f32(float %sched.n1014), !dbg !28
  %sched.n1016 = insertelement <4 x float> %sched.n1012, float %sched.n1015, i64 3, !dbg !30
  %math.result320 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1016) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n1017 = fptrunc <4 x float> %math.result320 to <4 x half>, !dbg !30
  %math.packed321 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1017) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n1018 = fadd float %sched.n1013, %sched.n1015, !dbg !29
  %sched.n1019 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %sched.n1020 = fadd <4 x float> %sched.n673, %sched.n1019, !dbg !27
  %sched.n1021 = extractelement <4 x float> %sched.n1020, i64 0, !dbg !28
  %sched.n1022 = call float @llvm.amdgcn.exp2.f32(float %sched.n1021), !dbg !28
  %sched.n1023 = insertelement <4 x float> poison, float %sched.n1022, i64 0, !dbg !30
  %sched.n1024 = fadd float %sched.n1018, %sched.n1022, !dbg !29
  %sched.n1025 = extractelement <4 x float> %sched.n1020, i64 1, !dbg !28
  %sched.n1026 = call float @llvm.amdgcn.exp2.f32(float %sched.n1025), !dbg !28
  %sched.n1027 = insertelement <4 x float> %sched.n1023, float %sched.n1026, i64 1, !dbg !30
  %sched.n1028 = fadd float %sched.n1024, %sched.n1026, !dbg !29
  %sched.n1029 = extractelement <4 x float> %sched.n1020, i64 2, !dbg !28
  %sched.n1030 = call float @llvm.amdgcn.exp2.f32(float %sched.n1029), !dbg !28
  %sched.n1031 = insertelement <4 x float> %sched.n1027, float %sched.n1030, i64 2, !dbg !30
  %sched.n1032 = fadd float %sched.n1028, %sched.n1030, !dbg !29
  %sched.n1033 = extractelement <4 x float> %sched.n1020, i64 3, !dbg !28
  %sched.n1034 = call float @llvm.amdgcn.exp2.f32(float %sched.n1033), !dbg !28
  %sched.n1035 = insertelement <4 x float> %sched.n1031, float %sched.n1034, i64 3, !dbg !30
  %math.result322 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1035) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n1036 = fptrunc <4 x float> %math.result322 to <4 x half>, !dbg !30
  %math.packed323 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1036) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n1037 = fadd float %sched.n1032, %sched.n1034, !dbg !29
  %sched.n1038 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %sched.n1039 = fadd <4 x float> %sched.n677, %sched.n1038, !dbg !27
  %sched.n1040 = extractelement <4 x float> %sched.n1039, i64 0, !dbg !28
  %sched.n1041 = call float @llvm.amdgcn.exp2.f32(float %sched.n1040), !dbg !28
  %sched.n1042 = insertelement <4 x float> poison, float %sched.n1041, i64 0, !dbg !30
  %sched.n1043 = fadd float %sched.n1037, %sched.n1041, !dbg !29
  %sched.n1044 = extractelement <4 x float> %sched.n1039, i64 1, !dbg !28
  %sched.n1045 = call float @llvm.amdgcn.exp2.f32(float %sched.n1044), !dbg !28
  %sched.n1046 = insertelement <4 x float> %sched.n1042, float %sched.n1045, i64 1, !dbg !30
  %sched.n1047 = fadd float %sched.n1043, %sched.n1045, !dbg !29
  %sched.n1048 = extractelement <4 x float> %sched.n1039, i64 2, !dbg !28
  %sched.n1049 = call float @llvm.amdgcn.exp2.f32(float %sched.n1048), !dbg !28
  %sched.n1050 = insertelement <4 x float> %sched.n1046, float %sched.n1049, i64 2, !dbg !30
  %sched.n1051 = fadd float %sched.n1047, %sched.n1049, !dbg !29
  %sched.n1052 = extractelement <4 x float> %sched.n1039, i64 3, !dbg !28
  %sched.n1053 = call float @llvm.amdgcn.exp2.f32(float %sched.n1052), !dbg !28
  %sched.n1054 = insertelement <4 x float> %sched.n1050, float %sched.n1053, i64 3, !dbg !30
  %math.result324 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1054) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n1055 = fptrunc <4 x float> %math.result324 to <4 x half>, !dbg !30
  %math.packed325 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1055) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n1056 = fadd float %sched.n1051, %sched.n1053, !dbg !29
  %sched.n1057 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %sched.n1058 = fadd <4 x float> %sched.n681, %sched.n1057, !dbg !27
  %sched.n1059 = extractelement <4 x float> %sched.n1058, i64 0, !dbg !28
  %sched.n1060 = call float @llvm.amdgcn.exp2.f32(float %sched.n1059), !dbg !28
  %sched.n1061 = insertelement <4 x float> poison, float %sched.n1060, i64 0, !dbg !30
  %sched.n1062 = fadd float %sched.n1056, %sched.n1060, !dbg !29
  %sched.n1063 = extractelement <4 x float> %sched.n1058, i64 1, !dbg !28
  %sched.n1064 = call float @llvm.amdgcn.exp2.f32(float %sched.n1063), !dbg !28
  %sched.n1065 = insertelement <4 x float> %sched.n1061, float %sched.n1064, i64 1, !dbg !30
  %sched.n1066 = fadd float %sched.n1062, %sched.n1064, !dbg !29
  %sched.n1067 = extractelement <4 x float> %sched.n1058, i64 2, !dbg !28
  %sched.n1068 = call float @llvm.amdgcn.exp2.f32(float %sched.n1067), !dbg !28
  %sched.n1069 = insertelement <4 x float> %sched.n1065, float %sched.n1068, i64 2, !dbg !30
  %sched.n1070 = fadd float %sched.n1066, %sched.n1068, !dbg !29
  %sched.n1071 = extractelement <4 x float> %sched.n1058, i64 3, !dbg !28
  %sched.n1072 = call float @llvm.amdgcn.exp2.f32(float %sched.n1071), !dbg !28
  %sched.n1073 = insertelement <4 x float> %sched.n1069, float %sched.n1072, i64 3, !dbg !30
  %math.result326 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1073) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n1074 = fptrunc <4 x float> %math.result326 to <4 x half>, !dbg !30
  %math.packed327 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1074) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n1075 = fadd float %sched.n1070, %sched.n1072, !dbg !29
  %sched.n1076 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %sched.n1077 = fadd <4 x float> %sched.n685, %sched.n1076, !dbg !27
  %sched.n1078 = extractelement <4 x float> %sched.n1077, i64 0, !dbg !28
  %sched.n1079 = call float @llvm.amdgcn.exp2.f32(float %sched.n1078), !dbg !28
  %sched.n1080 = insertelement <4 x float> poison, float %sched.n1079, i64 0, !dbg !30
  %sched.n1081 = fadd float %sched.n1075, %sched.n1079, !dbg !29
  %sched.n1082 = extractelement <4 x float> %sched.n1077, i64 1, !dbg !28
  %sched.n1083 = call float @llvm.amdgcn.exp2.f32(float %sched.n1082), !dbg !28
  %sched.n1084 = insertelement <4 x float> %sched.n1080, float %sched.n1083, i64 1, !dbg !30
  %sched.n1085 = fadd float %sched.n1081, %sched.n1083, !dbg !29
  %sched.n1086 = extractelement <4 x float> %sched.n1077, i64 2, !dbg !28
  %sched.n1087 = call float @llvm.amdgcn.exp2.f32(float %sched.n1086), !dbg !28
  %sched.n1088 = insertelement <4 x float> %sched.n1084, float %sched.n1087, i64 2, !dbg !30
  %sched.n1089 = fadd float %sched.n1085, %sched.n1087, !dbg !29
  %sched.n1090 = extractelement <4 x float> %sched.n1077, i64 3, !dbg !28
  %sched.n1091 = call float @llvm.amdgcn.exp2.f32(float %sched.n1090), !dbg !28
  %sched.n1092 = insertelement <4 x float> %sched.n1088, float %sched.n1091, i64 3, !dbg !30
  %math.result328 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1092) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n1093 = fptrunc <4 x float> %math.result328 to <4 x half>, !dbg !30
  %math.packed329 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1093) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n1094 = fadd float %sched.n1089, %sched.n1091, !dbg !29
  %sched.n1095 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %sched.n1096 = fadd <4 x float> %sched.n689, %sched.n1095, !dbg !27
  %sched.n1097 = extractelement <4 x float> %sched.n1096, i64 0, !dbg !28
  %sched.n1098 = call float @llvm.amdgcn.exp2.f32(float %sched.n1097), !dbg !28
  %sched.n1099 = insertelement <4 x float> poison, float %sched.n1098, i64 0, !dbg !30
  %sched.n1100 = fadd float %sched.n1094, %sched.n1098, !dbg !29
  %sched.n1101 = extractelement <4 x float> %sched.n1096, i64 1, !dbg !28
  %sched.n1102 = call float @llvm.amdgcn.exp2.f32(float %sched.n1101), !dbg !28
  %sched.n1103 = insertelement <4 x float> %sched.n1099, float %sched.n1102, i64 1, !dbg !30
  %sched.n1104 = fadd float %sched.n1100, %sched.n1102, !dbg !29
  %sched.n1105 = extractelement <4 x float> %sched.n1096, i64 2, !dbg !28
  %sched.n1106 = call float @llvm.amdgcn.exp2.f32(float %sched.n1105), !dbg !28
  %sched.n1107 = insertelement <4 x float> %sched.n1103, float %sched.n1106, i64 2, !dbg !30
  %sched.n1108 = fadd float %sched.n1104, %sched.n1106, !dbg !29
  %sched.n1109 = extractelement <4 x float> %sched.n1096, i64 3, !dbg !28
  %sched.n1110 = call float @llvm.amdgcn.exp2.f32(float %sched.n1109), !dbg !28
  %sched.n1111 = insertelement <4 x float> %sched.n1107, float %sched.n1110, i64 3, !dbg !30
  %math.result330 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1111) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n1112 = fptrunc <4 x float> %math.result330 to <4 x half>, !dbg !30
  %math.packed331 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1112) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n1113 = fadd float %sched.n1108, %sched.n1110, !dbg !29
  %sched.n1114 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %sched.n1115 = fadd <4 x float> %sched.n693, %sched.n1114, !dbg !27
  %sched.n1116 = extractelement <4 x float> %sched.n1115, i64 0, !dbg !28
  %sched.n1117 = call float @llvm.amdgcn.exp2.f32(float %sched.n1116), !dbg !28
  %sched.n1118 = insertelement <4 x float> poison, float %sched.n1117, i64 0, !dbg !30
  %sched.n1119 = fadd float %sched.n1113, %sched.n1117, !dbg !29
  %sched.n1120 = extractelement <4 x float> %sched.n1115, i64 1, !dbg !28
  %sched.n1121 = call float @llvm.amdgcn.exp2.f32(float %sched.n1120), !dbg !28
  %sched.n1122 = insertelement <4 x float> %sched.n1118, float %sched.n1121, i64 1, !dbg !30
  %sched.n1123 = fadd float %sched.n1119, %sched.n1121, !dbg !29
  %sched.n1124 = extractelement <4 x float> %sched.n1115, i64 2, !dbg !28
  %sched.n1125 = call float @llvm.amdgcn.exp2.f32(float %sched.n1124), !dbg !28
  %sched.n1126 = insertelement <4 x float> %sched.n1122, float %sched.n1125, i64 2, !dbg !30
  %sched.n1127 = fadd float %sched.n1123, %sched.n1125, !dbg !29
  %sched.n1128 = extractelement <4 x float> %sched.n1115, i64 3, !dbg !28
  %sched.n1129 = call float @llvm.amdgcn.exp2.f32(float %sched.n1128), !dbg !28
  %sched.n1130 = insertelement <4 x float> %sched.n1126, float %sched.n1129, i64 3, !dbg !30
  %math.result332 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1130) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n1131 = fptrunc <4 x float> %math.result332 to <4 x half>, !dbg !30
  %math.packed333 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1131) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n1132 = fadd float %sched.n1127, %sched.n1129, !dbg !29
  %sched.n1133 = shufflevector <32 x float> %sched.n1000, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %sched.n1134 = fadd <4 x float> %sched.n698, %sched.n1133, !dbg !27
  %sched.n1135 = extractelement <4 x float> %sched.n1134, i64 0, !dbg !28
  %sched.n1136 = call float @llvm.amdgcn.exp2.f32(float %sched.n1135), !dbg !28
  %sched.n1137 = insertelement <4 x float> poison, float %sched.n1136, i64 0, !dbg !30
  %sched.n1138 = fadd float %sched.n1132, %sched.n1136, !dbg !29
  %sched.n1139 = extractelement <4 x float> %sched.n1134, i64 1, !dbg !28
  %sched.n1140 = call float @llvm.amdgcn.exp2.f32(float %sched.n1139), !dbg !28
  %sched.n1141 = insertelement <4 x float> %sched.n1137, float %sched.n1140, i64 1, !dbg !30
  %sched.n1142 = fadd float %sched.n1138, %sched.n1140, !dbg !29
  %sched.n1143 = extractelement <4 x float> %sched.n1134, i64 2, !dbg !28
  %sched.n1144 = call float @llvm.amdgcn.exp2.f32(float %sched.n1143), !dbg !28
  %sched.n1145 = insertelement <4 x float> %sched.n1141, float %sched.n1144, i64 2, !dbg !30
  %sched.n1146 = fadd float %sched.n1142, %sched.n1144, !dbg !29
  %sched.n1147 = extractelement <4 x float> %sched.n1134, i64 3, !dbg !28
  %sched.n1148 = call float @llvm.amdgcn.exp2.f32(float %sched.n1147), !dbg !28
  %sched.n1149 = insertelement <4 x float> %sched.n1145, float %sched.n1148, i64 3, !dbg !30
  %math.result334 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %sched.n1149) #9, !dbg !30, !frisk.register_lifetimes !1
  %sched.n1150 = fptrunc <4 x float> %math.result334 to <4 x half>, !dbg !30
  %math.packed335 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %sched.n1150) #9, !dbg !29, !frisk.register_lifetimes !1
  %sched.n1151 = fadd float %sched.n1146, %sched.n1148, !dbg !29
  %sched.n1152 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !29
  %sched.n1153 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %sched.n1152), !dbg !29
  %sched.n1154 = add nuw nsw i32 %sched.n1153, 64, !dbg !29
  %sched.n1155 = xor i32 %sched.n1153, 16, !dbg !29
  %sched.n1156 = and i32 %sched.n1154, 192, !dbg !29
  %sched.n1157 = icmp samesign ult i32 %sched.n1155, %sched.n1156, !dbg !29
  %sched.n1158 = select i1 %sched.n1157, i32 %sched.n1155, i32 %sched.n1153, !dbg !29
  %sched.n1159 = shl nuw nsw i32 %sched.n1158, 2, !dbg !29
  %sched.n1160 = bitcast float %sched.n999 to i32, !dbg !29
  %sched.n1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1159, i32 %sched.n1160), !dbg !29
  %sched.n1162 = bitcast i32 %sched.n1161 to float, !dbg !29
  %sched.n1163 = fadd float %sched.n999, %sched.n1162, !dbg !29
  %sched.n1164 = xor i32 %sched.n1153, 32, !dbg !29
  %sched.n1165 = icmp samesign ult i32 %sched.n1164, %sched.n1156, !dbg !29
  %sched.n1166 = select i1 %sched.n1165, i32 %sched.n1164, i32 %sched.n1153, !dbg !29
  %sched.n1167 = shl nuw nsw i32 %sched.n1166, 2, !dbg !29
  %sched.n1168 = bitcast float %sched.n1163 to i32, !dbg !29
  %sched.n1169 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1167, i32 %sched.n1168), !dbg !29
  %sched.n1170 = bitcast i32 %sched.n1169 to float, !dbg !29
  %sched.n1171 = fadd float %sched.n1163, %sched.n1170, !dbg !29
  %sched.n1172 = extractvalue [2 x <1 x float>] %sched.n99, 0, !dbg !29
  %sched.n1173 = extractelement <1 x float> %sched.n1172, i64 0, !dbg !29
  %sched.n1174 = fadd float %sched.n1171, %sched.n1173, !dbg !29
  %sched.n1175 = insertelement <1 x float> poison, float %sched.n1174, i64 0, !dbg !12
  %sched.n1176 = insertvalue [2 x <1 x float>] poison, <1 x float> %sched.n1175, 0, !dbg !12
  %sched.n1177 = bitcast float %sched.n1151 to i32, !dbg !29
  %sched.n1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1159, i32 %sched.n1177), !dbg !29
  %sched.n1179 = bitcast i32 %sched.n1178 to float, !dbg !29
  %sched.n1180 = fadd float %sched.n1151, %sched.n1179, !dbg !29
  %sched.n1181 = bitcast float %sched.n1180 to i32, !dbg !29
  %sched.n1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1167, i32 %sched.n1181), !dbg !29
  %sched.n1183 = bitcast i32 %sched.n1182 to float, !dbg !29
  %sched.n1184 = fadd float %sched.n1180, %sched.n1183, !dbg !29
  %sched.n1185 = extractvalue [2 x <1 x float>] %sched.n99, 1, !dbg !29
  %sched.n1186 = extractelement <1 x float> %sched.n1185, i64 0, !dbg !29
  %sched.n1187 = fadd float %sched.n1184, %sched.n1186, !dbg !29
  %sched.n1188 = insertelement <1 x float> poison, float %sched.n1187, i64 0, !dbg !12
  %sched.n1189 = insertvalue [2 x <1 x float>] %sched.n1176, <1 x float> %sched.n1188, 1, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread844 = call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #9, !dbg !15, !range !34, !frisk.register_lifetimes !1
  %address.local845 = shl nuw nsw i32 %address.thread844, 3, !dbg !15
  %address.local846 = and i32 %address.local845, 120, !dbg !15
  %address.local847 = shl nuw nsw i32 %address.thread844, 5, !dbg !15
  %address.local848 = and i32 %address.local847, 1536, !dbg !15
  %address.local849 = or disjoint i32 %address.local846, %address.local848, !dbg !15
  %address.local850 = zext nneg i32 %address.local849 to i64, !dbg !15
  %address.local851 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local850, !dbg !15
  %sched.n1190 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #0, !dbg !15
  %sched.n1191 = bitcast <4 x i32> %sched.n1190 to <8 x half>, !dbg !15
  %address.local852 = zext nneg i32 %address.thread844 to i64, !dbg !8
  %address.local853 = and i64 %address.local852, 60, !dbg !15
  %address.local854 = add nuw nsw i64 %address.local853, %address.local852, !dbg !15
  %address.local855 = shl nuw nsw i64 %address.local854, 3, !dbg !15
  %address.local856 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local855, !dbg !15
  %address.local857 = and i64 %address.local852, 63, !dbg !11
  %address.local858 = lshr i64 %address.local857, 4, !dbg !15
  %address.local859 = mul nuw nsw i64 %address.local858, 384, !dbg !15
  %address.local860 = getelementptr [2 x i8], ptr addrspace(3) %address.local856, i64 %address.local859, !dbg !15
  %address.local861 = lshr i64 %address.local852, 2, !dbg !18
  %address.local862 = mul nsw i64 %address.local861, -32, !dbg !15
  %address.local863 = getelementptr [2 x i8], ptr addrspace(3) %address.local860, i64 %address.local862, !dbg !15
  %address.local864 = getelementptr i8, ptr addrspace(3) %address.local863, i64 256, !dbg !15
  %sched.n1192 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #0, !dbg !15
  %sched.n1193 = bitcast <4 x i32> %sched.n1192 to <8 x half>, !dbg !15
  %address.local865 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local855, !dbg !15
  %address.local866 = getelementptr [2 x i8], ptr addrspace(3) %address.local865, i64 %address.local859, !dbg !15
  %address.local867 = getelementptr [2 x i8], ptr addrspace(3) %address.local866, i64 %address.local862, !dbg !15
  %address.local868 = getelementptr i8, ptr addrspace(3) %address.local867, i64 512, !dbg !15
  %sched.n1194 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #0, !dbg !15
  %sched.n1195 = bitcast <4 x i32> %sched.n1194 to <8 x half>, !dbg !15
  %address.local869 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local855, !dbg !15
  %address.local870 = getelementptr [2 x i8], ptr addrspace(3) %address.local869, i64 %address.local859, !dbg !15
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) %address.local870, i64 %address.local862, !dbg !15
  %address.local872 = getelementptr i8, ptr addrspace(3) %address.local871, i64 768, !dbg !15
  %sched.n1196 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #0, !dbg !15
  %sched.n1197 = bitcast <4 x i32> %sched.n1196 to <8 x half>, !dbg !15
  %address.local873 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local850, !dbg !15
  %address.local874 = getelementptr i8, ptr addrspace(3) %address.local873, i64 4096, !dbg !15
  %sched.n1198 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #0, !dbg !15
  %sched.n1199 = bitcast <4 x i32> %sched.n1198 to <8 x half>, !dbg !15
  %address.local875 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local855, !dbg !15
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) %address.local875, i64 %address.local859, !dbg !15
  %address.local877 = getelementptr [2 x i8], ptr addrspace(3) %address.local876, i64 %address.local862, !dbg !15
  %address.local878 = getelementptr i8, ptr addrspace(3) %address.local877, i64 4352, !dbg !15
  %sched.n1200 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #0, !dbg !15
  %sched.n1201 = bitcast <4 x i32> %sched.n1200 to <8 x half>, !dbg !15
  %address.local879 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local855, !dbg !15
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) %address.local879, i64 %address.local859, !dbg !15
  %address.local881 = getelementptr [2 x i8], ptr addrspace(3) %address.local880, i64 %address.local862, !dbg !15
  %address.local882 = getelementptr i8, ptr addrspace(3) %address.local881, i64 4608, !dbg !15
  %sched.n1202 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #0, !dbg !15
  %sched.n1203 = bitcast <4 x i32> %sched.n1202 to <8 x half>, !dbg !15
  %address.local883 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local855, !dbg !15
  %address.local884 = getelementptr [2 x i8], ptr addrspace(3) %address.local883, i64 %address.local859, !dbg !15
  %address.local885 = getelementptr [2 x i8], ptr addrspace(3) %address.local884, i64 %address.local862, !dbg !15
  %address.local886 = getelementptr i8, ptr addrspace(3) %address.local885, i64 4864, !dbg !15
  %sched.n1204 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #0, !dbg !15
  %sched.n1205 = bitcast <4 x i32> %sched.n1204 to <8 x half>, !dbg !15
  %sched.n1206 = add nuw nsw i64 %sched.n108, %sched.n16, !dbg !19
  %sched.n1207 = shl i64 %sched.n1206, 19, !dbg !19
  %sched.n1208 = and i64 %sched.n1207, 16252928, !dbg !19
  %sched.n1209 = add i64 %sched.n1208, %sched.n114, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %sched.n1210 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %sched.n1209, !dbg !19
  %sched.n1211 = getelementptr i8, ptr addrspace(1) %sched.n1210, i64 64, !dbg !19
  %address.local887 = shl nuw nsw i64 %address.local852, 3, !dbg !8
  %address.local888 = and i64 %address.local887, 1536, !dbg !8
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local888, !dbg !8
  %address.local890 = ptrtoint ptr addrspace(3) %address.local889 to i64, !dbg !8
  %address.local891 = trunc i64 %address.local890 to i32, !dbg !8
  %sched.n1212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local891), !dbg !19
  %sched.n1213 = zext i32 %sched.n1212 to i64, !dbg !19
  %sched.n1214 = inttoptr i64 %sched.n1213 to ptr addrspace(3), !dbg !19
  %address.local892 = shl nuw nsw i64 %address.local852, 5, !dbg !19
  %address.local893 = and i64 %address.local892, 8064, !dbg !19
  %address.local894 = shl nuw nsw i64 %address.local852, 3, !dbg !19
  %address.local895 = and i64 %address.local894, 24, !dbg !19
  %address.local896 = or disjoint i64 %address.local893, %address.local895, !dbg !19
  %address.local897 = getelementptr [2 x i8], ptr addrspace(1) %sched.n1211, i64 %address.local896, !dbg !19
  %sched.n1215 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %sched.n1216 = ptrtoint ptr addrspace(1) %address.local897 to i64, !dbg !19
  %sched.n1217 = sub i64 %sched.n1216, %sched.n1215, !dbg !19
  %dma.buffer.offset1543 = trunc i64 %sched.n1217 to i32, !dbg !19
  %sched.n1218 = trunc i64 %sched.n1215 to i32, !dbg !19
  %sched.n1219 = insertelement <4 x i32> poison, i32 %sched.n1218, i64 0, !dbg !19
  %sched.n1220 = lshr i64 %sched.n1215, 32, !dbg !19
  %sched.n1221 = trunc i64 %sched.n1220 to i32, !dbg !19
  %sched.n1222 = insertelement <4 x i32> %sched.n1219, i32 %sched.n1221, i64 1, !dbg !19
  %sched.n1223 = insertelement <4 x i32> %sched.n1222, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1544 = insertelement <4 x i32> %sched.n1223, i32 131072, i64 3, !dbg !19
  %sched.n1224 = ptrtoint ptr addrspace(3) %sched.n1214 to i32, !dbg !19
  %dma.buffer.lds1545 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n1224), !dbg !19
  %sched.n1225 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1545, i32 %dma.buffer.offset1543, <4 x i32> %dma.buffer.rsrc1544) #10, !dbg !19
  %address.local898 = or disjoint i64 %address.local852, 256, !dbg !8
  %address.local899 = shl nuw nsw i64 %address.local898, 3, !dbg !8
  %address.local900 = and i64 %address.local899, 3584, !dbg !8
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local900, !dbg !8
  %address.local902 = ptrtoint ptr addrspace(3) %address.local901 to i64, !dbg !8
  %address.local903 = trunc i64 %address.local902 to i32, !dbg !8
  %sched.n1226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local903), !dbg !19
  %sched.n1227 = zext i32 %sched.n1226 to i64, !dbg !19
  %sched.n1228 = inttoptr i64 %sched.n1227 to ptr addrspace(3), !dbg !19
  %address.local904 = shl nuw nsw i64 %address.local898, 5, !dbg !19
  %address.local905 = and i64 %address.local904, 16256, !dbg !19
  %address.local906 = shl nuw nsw i64 %address.local852, 3, !dbg !19
  %address.local907 = and i64 %address.local906, 24, !dbg !19
  %address.local908 = or disjoint i64 %address.local905, %address.local907, !dbg !19
  %address.local909 = getelementptr [2 x i8], ptr addrspace(1) %sched.n1211, i64 %address.local908, !dbg !19
  %sched.n1229 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %sched.n1230 = ptrtoint ptr addrspace(1) %address.local909 to i64, !dbg !19
  %sched.n1231 = sub i64 %sched.n1230, %sched.n1229, !dbg !19
  %dma.buffer.offset1540 = trunc i64 %sched.n1231 to i32, !dbg !19
  %sched.n1232 = trunc i64 %sched.n1229 to i32, !dbg !19
  %sched.n1233 = insertelement <4 x i32> poison, i32 %sched.n1232, i64 0, !dbg !19
  %sched.n1234 = lshr i64 %sched.n1229, 32, !dbg !19
  %sched.n1235 = trunc i64 %sched.n1234 to i32, !dbg !19
  %sched.n1236 = insertelement <4 x i32> %sched.n1233, i32 %sched.n1235, i64 1, !dbg !19
  %sched.n1237 = insertelement <4 x i32> %sched.n1236, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1541 = insertelement <4 x i32> %sched.n1237, i32 131072, i64 3, !dbg !19
  %sched.n1238 = ptrtoint ptr addrspace(3) %sched.n1228 to i32, !dbg !19
  %dma.buffer.lds1542 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n1238), !dbg !19
  %sched.n1239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1542, i32 %dma.buffer.offset1540, <4 x i32> %dma.buffer.rsrc1541) #10, !dbg !19
  %sched.n1240 = shufflevector <8 x half> %sched.n1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1241 = extractvalue [2 x <8 x float>] %sched.n98, 0, !dbg !31
  %sched.n1242 = shufflevector <8 x float> %sched.n1241, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %sched.n1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1240, <4 x float> %sched.n1242) #9, !dbg !15
  %sched.n1244 = shufflevector <8 x half> %sched.n1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1244, <4 x float> %sched.n1243) #9, !dbg !15
  %sched.n1246 = shufflevector <8 x half> %sched.n1195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1246, <4 x float> %sched.n1245) #9, !dbg !15
  %sched.n1248 = shufflevector <8 x half> %sched.n1197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1248, <4 x float> %sched.n1247) #9, !dbg !15
  %sched.n1250 = shufflevector <8 x half> %sched.n1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1250, <4 x float> %sched.n1249) #9, !dbg !15
  %sched.n1252 = shufflevector <8 x half> %sched.n1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1252, <4 x float> %sched.n1251) #9, !dbg !15
  %sched.n1254 = shufflevector <8 x half> %sched.n1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1254, <4 x float> %sched.n1253) #9, !dbg !15
  %sched.n1256 = shufflevector <8 x half> %sched.n1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1256, <4 x float> %sched.n1255) #9, !dbg !15
  %sched.n1258 = shufflevector <8 x half> %sched.n1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1259 = shufflevector <8 x float> %sched.n1241, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %sched.n1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1258, <4 x float> %sched.n1259) #9, !dbg !15
  %sched.n1261 = shufflevector <8 x half> %sched.n1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1261, <4 x float> %sched.n1260) #9, !dbg !15
  %sched.n1263 = shufflevector <8 x half> %sched.n1195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1263, <4 x float> %sched.n1262) #9, !dbg !15
  %sched.n1265 = shufflevector <8 x half> %sched.n1197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1265, <4 x float> %sched.n1264) #9, !dbg !15
  %sched.n1267 = shufflevector <8 x half> %sched.n1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1267, <4 x float> %sched.n1266) #9, !dbg !15
  %sched.n1269 = shufflevector <8 x half> %sched.n1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1269, <4 x float> %sched.n1268) #9, !dbg !15
  %sched.n1271 = shufflevector <8 x half> %sched.n1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1271, <4 x float> %sched.n1270) #9, !dbg !15
  %sched.n1273 = shufflevector <8 x half> %sched.n1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1273, <4 x float> %sched.n1272) #9, !dbg !15
  %sched.n1275 = extractvalue [2 x <8 x float>] %sched.n98, 1, !dbg !31
  %sched.n1276 = shufflevector <8 x float> %sched.n1275, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %sched.n1277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1240, <4 x float> %sched.n1276) #9, !dbg !15
  %sched.n1278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1244, <4 x float> %sched.n1277) #9, !dbg !15
  %sched.n1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1246, <4 x float> %sched.n1278) #9, !dbg !15
  %sched.n1280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1248, <4 x float> %sched.n1279) #9, !dbg !15
  %sched.n1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1250, <4 x float> %sched.n1280) #9, !dbg !15
  %sched.n1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1252, <4 x float> %sched.n1281) #9, !dbg !15
  %sched.n1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1254, <4 x float> %sched.n1282) #9, !dbg !15
  %sched.n1284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1256, <4 x float> %sched.n1283) #9, !dbg !15
  %sched.n1285 = shufflevector <8 x float> %sched.n1275, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %sched.n1286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1258, <4 x float> %sched.n1285) #9, !dbg !15
  %sched.n1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1261, <4 x float> %sched.n1286) #9, !dbg !15
  %sched.n1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1263, <4 x float> %sched.n1287) #9, !dbg !15
  %sched.n1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1265, <4 x float> %sched.n1288) #9, !dbg !15
  %sched.n1290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1267, <4 x float> %sched.n1289) #9, !dbg !15
  %sched.n1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1269, <4 x float> %sched.n1290) #9, !dbg !15
  %sched.n1292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1271, <4 x float> %sched.n1291) #9, !dbg !15
  %sched.n1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1273, <4 x float> %sched.n1292) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %sched.n1294 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #0, !dbg !15
  %sched.n1295 = bitcast <4 x i32> %sched.n1294 to <8 x half>, !dbg !15
  %sched.n1296 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #0, !dbg !15
  %sched.n1297 = bitcast <4 x i32> %sched.n1296 to <8 x half>, !dbg !15
  %sched.n1298 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #0, !dbg !15
  %sched.n1299 = bitcast <4 x i32> %sched.n1298 to <8 x half>, !dbg !15
  %sched.n1300 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #0, !dbg !15
  %sched.n1301 = bitcast <4 x i32> %sched.n1300 to <8 x half>, !dbg !15
  %sched.n1302 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #0, !dbg !15
  %sched.n1303 = bitcast <4 x i32> %sched.n1302 to <8 x half>, !dbg !15
  %sched.n1304 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #0, !dbg !15
  %sched.n1305 = bitcast <4 x i32> %sched.n1304 to <8 x half>, !dbg !15
  %sched.n1306 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #0, !dbg !15
  %sched.n1307 = bitcast <4 x i32> %sched.n1306 to <8 x half>, !dbg !15
  %sched.n1308 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #0, !dbg !15
  %sched.n1309 = bitcast <4 x i32> %sched.n1308 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %sched.n1310 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %sched.n1209, !dbg !19
  %sched.n1311 = getelementptr i8, ptr addrspace(1) %sched.n1310, i64 128, !dbg !19
  %sched.n1312 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local891), !dbg !19
  %sched.n1313 = zext i32 %sched.n1312 to i64, !dbg !19
  %sched.n1314 = inttoptr i64 %sched.n1313 to ptr addrspace(3), !dbg !19
  %address.local910 = getelementptr [2 x i8], ptr addrspace(1) %sched.n1311, i64 %address.local896, !dbg !19
  %sched.n1315 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %sched.n1316 = ptrtoint ptr addrspace(1) %address.local910 to i64, !dbg !19
  %sched.n1317 = sub i64 %sched.n1316, %sched.n1315, !dbg !19
  %dma.buffer.offset1537 = trunc i64 %sched.n1317 to i32, !dbg !19
  %sched.n1318 = trunc i64 %sched.n1315 to i32, !dbg !19
  %sched.n1319 = insertelement <4 x i32> poison, i32 %sched.n1318, i64 0, !dbg !19
  %sched.n1320 = lshr i64 %sched.n1315, 32, !dbg !19
  %sched.n1321 = trunc i64 %sched.n1320 to i32, !dbg !19
  %sched.n1322 = insertelement <4 x i32> %sched.n1319, i32 %sched.n1321, i64 1, !dbg !19
  %sched.n1323 = insertelement <4 x i32> %sched.n1322, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1538 = insertelement <4 x i32> %sched.n1323, i32 131072, i64 3, !dbg !19
  %sched.n1324 = ptrtoint ptr addrspace(3) %sched.n1314 to i32, !dbg !19
  %dma.buffer.lds1539 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n1324), !dbg !19
  %sched.n1325 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1539, i32 %dma.buffer.offset1537, <4 x i32> %dma.buffer.rsrc1538) #10, !dbg !19
  %sched.n1326 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local903), !dbg !19
  %sched.n1327 = zext i32 %sched.n1326 to i64, !dbg !19
  %sched.n1328 = inttoptr i64 %sched.n1327 to ptr addrspace(3), !dbg !19
  %address.local911 = getelementptr [2 x i8], ptr addrspace(1) %sched.n1311, i64 %address.local908, !dbg !19
  %sched.n1329 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %sched.n1330 = ptrtoint ptr addrspace(1) %address.local911 to i64, !dbg !19
  %sched.n1331 = sub i64 %sched.n1330, %sched.n1329, !dbg !19
  %dma.buffer.offset1534 = trunc i64 %sched.n1331 to i32, !dbg !19
  %sched.n1332 = trunc i64 %sched.n1329 to i32, !dbg !19
  %sched.n1333 = insertelement <4 x i32> poison, i32 %sched.n1332, i64 0, !dbg !19
  %sched.n1334 = lshr i64 %sched.n1329, 32, !dbg !19
  %sched.n1335 = trunc i64 %sched.n1334 to i32, !dbg !19
  %sched.n1336 = insertelement <4 x i32> %sched.n1333, i32 %sched.n1335, i64 1, !dbg !19
  %sched.n1337 = insertelement <4 x i32> %sched.n1336, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1535 = insertelement <4 x i32> %sched.n1337, i32 131072, i64 3, !dbg !19
  %sched.n1338 = ptrtoint ptr addrspace(3) %sched.n1328 to i32, !dbg !19
  %dma.buffer.lds1536 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n1338), !dbg !19
  %sched.n1339 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1536, i32 %dma.buffer.offset1534, <4 x i32> %dma.buffer.rsrc1535) #10, !dbg !19
  %sched.n1340 = shufflevector <8 x half> %sched.n1295, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1341 = extractvalue [2 x <8 x float>] %sched.n100, 0, !dbg !31
  %sched.n1342 = shufflevector <8 x float> %sched.n1341, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %sched.n1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1340, <4 x float> %sched.n1342) #9, !dbg !15
  %sched.n1344 = shufflevector <8 x half> %sched.n1297, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1344, <4 x float> %sched.n1343) #9, !dbg !15
  %sched.n1346 = shufflevector <8 x half> %sched.n1299, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1346, <4 x float> %sched.n1345) #9, !dbg !15
  %sched.n1348 = shufflevector <8 x half> %sched.n1301, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1348, <4 x float> %sched.n1347) #9, !dbg !15
  %sched.n1350 = shufflevector <8 x half> %sched.n1303, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1350, <4 x float> %sched.n1349) #9, !dbg !15
  %sched.n1352 = shufflevector <8 x half> %sched.n1305, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1352, <4 x float> %sched.n1351) #9, !dbg !15
  %sched.n1354 = shufflevector <8 x half> %sched.n1307, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1354, <4 x float> %sched.n1353) #9, !dbg !15
  %sched.n1356 = shufflevector <8 x half> %sched.n1309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1356, <4 x float> %sched.n1355) #9, !dbg !15
  %sched.n1358 = shufflevector <8 x half> %sched.n1295, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1359 = shufflevector <8 x float> %sched.n1341, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %sched.n1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1358, <4 x float> %sched.n1359) #9, !dbg !15
  %sched.n1361 = shufflevector <8 x half> %sched.n1297, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1361, <4 x float> %sched.n1360) #9, !dbg !15
  %sched.n1363 = shufflevector <8 x half> %sched.n1299, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1363, <4 x float> %sched.n1362) #9, !dbg !15
  %sched.n1365 = shufflevector <8 x half> %sched.n1301, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1365, <4 x float> %sched.n1364) #9, !dbg !15
  %sched.n1367 = shufflevector <8 x half> %sched.n1303, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1367, <4 x float> %sched.n1366) #9, !dbg !15
  %sched.n1369 = shufflevector <8 x half> %sched.n1305, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1369, <4 x float> %sched.n1368) #9, !dbg !15
  %sched.n1371 = shufflevector <8 x half> %sched.n1307, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1371, <4 x float> %sched.n1370) #9, !dbg !15
  %sched.n1373 = shufflevector <8 x half> %sched.n1309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1373, <4 x float> %sched.n1372) #9, !dbg !15
  %sched.n1375 = extractvalue [2 x <8 x float>] %sched.n100, 1, !dbg !31
  %sched.n1376 = shufflevector <8 x float> %sched.n1375, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %sched.n1377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1340, <4 x float> %sched.n1376) #9, !dbg !15
  %sched.n1378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1344, <4 x float> %sched.n1377) #9, !dbg !15
  %sched.n1379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1346, <4 x float> %sched.n1378) #9, !dbg !15
  %sched.n1380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1348, <4 x float> %sched.n1379) #9, !dbg !15
  %sched.n1381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1350, <4 x float> %sched.n1380) #9, !dbg !15
  %sched.n1382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1352, <4 x float> %sched.n1381) #9, !dbg !15
  %sched.n1383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1354, <4 x float> %sched.n1382) #9, !dbg !15
  %sched.n1384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1356, <4 x float> %sched.n1383) #9, !dbg !15
  %sched.n1385 = shufflevector <8 x float> %sched.n1375, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %sched.n1386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1358, <4 x float> %sched.n1385) #9, !dbg !15
  %sched.n1387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1361, <4 x float> %sched.n1386) #9, !dbg !15
  %sched.n1388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1363, <4 x float> %sched.n1387) #9, !dbg !15
  %sched.n1389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1365, <4 x float> %sched.n1388) #9, !dbg !15
  %sched.n1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1367, <4 x float> %sched.n1389) #9, !dbg !15
  %sched.n1391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1369, <4 x float> %sched.n1390) #9, !dbg !15
  %sched.n1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1371, <4 x float> %sched.n1391) #9, !dbg !15
  %sched.n1393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1373, <4 x float> %sched.n1392) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %sched.n1394 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #0, !dbg !15
  %sched.n1395 = bitcast <4 x i32> %sched.n1394 to <8 x half>, !dbg !15
  %sched.n1396 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #0, !dbg !15
  %sched.n1397 = bitcast <4 x i32> %sched.n1396 to <8 x half>, !dbg !15
  %sched.n1398 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #0, !dbg !15
  %sched.n1399 = bitcast <4 x i32> %sched.n1398 to <8 x half>, !dbg !15
  %sched.n1400 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #0, !dbg !15
  %sched.n1401 = bitcast <4 x i32> %sched.n1400 to <8 x half>, !dbg !15
  %sched.n1402 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #0, !dbg !15
  %sched.n1403 = bitcast <4 x i32> %sched.n1402 to <8 x half>, !dbg !15
  %sched.n1404 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #0, !dbg !15
  %sched.n1405 = bitcast <4 x i32> %sched.n1404 to <8 x half>, !dbg !15
  %sched.n1406 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #0, !dbg !15
  %sched.n1407 = bitcast <4 x i32> %sched.n1406 to <8 x half>, !dbg !15
  %sched.n1408 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #0, !dbg !15
  %sched.n1409 = bitcast <4 x i32> %sched.n1408 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %sched.n1410 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %sched.n1209, !dbg !19
  %sched.n1411 = getelementptr i8, ptr addrspace(1) %sched.n1410, i64 192, !dbg !19
  %sched.n1412 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local891), !dbg !19
  %sched.n1413 = zext i32 %sched.n1412 to i64, !dbg !19
  %sched.n1414 = inttoptr i64 %sched.n1413 to ptr addrspace(3), !dbg !19
  %address.local912 = getelementptr [2 x i8], ptr addrspace(1) %sched.n1411, i64 %address.local896, !dbg !19
  %sched.n1415 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %sched.n1416 = ptrtoint ptr addrspace(1) %address.local912 to i64, !dbg !19
  %sched.n1417 = sub i64 %sched.n1416, %sched.n1415, !dbg !19
  %dma.buffer.offset1531 = trunc i64 %sched.n1417 to i32, !dbg !19
  %sched.n1418 = trunc i64 %sched.n1415 to i32, !dbg !19
  %sched.n1419 = insertelement <4 x i32> poison, i32 %sched.n1418, i64 0, !dbg !19
  %sched.n1420 = lshr i64 %sched.n1415, 32, !dbg !19
  %sched.n1421 = trunc i64 %sched.n1420 to i32, !dbg !19
  %sched.n1422 = insertelement <4 x i32> %sched.n1419, i32 %sched.n1421, i64 1, !dbg !19
  %sched.n1423 = insertelement <4 x i32> %sched.n1422, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1532 = insertelement <4 x i32> %sched.n1423, i32 131072, i64 3, !dbg !19
  %sched.n1424 = ptrtoint ptr addrspace(3) %sched.n1414 to i32, !dbg !19
  %dma.buffer.lds1533 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n1424), !dbg !19
  %sched.n1425 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1533, i32 %dma.buffer.offset1531, <4 x i32> %dma.buffer.rsrc1532) #10, !dbg !19
  %sched.n1426 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local903), !dbg !19
  %sched.n1427 = zext i32 %sched.n1426 to i64, !dbg !19
  %sched.n1428 = inttoptr i64 %sched.n1427 to ptr addrspace(3), !dbg !19
  %address.local913 = getelementptr [2 x i8], ptr addrspace(1) %sched.n1411, i64 %address.local908, !dbg !19
  %sched.n1429 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %sched.n1430 = ptrtoint ptr addrspace(1) %address.local913 to i64, !dbg !19
  %sched.n1431 = sub i64 %sched.n1430, %sched.n1429, !dbg !19
  %dma.buffer.offset = trunc i64 %sched.n1431 to i32, !dbg !19
  %sched.n1432 = trunc i64 %sched.n1429 to i32, !dbg !19
  %sched.n1433 = insertelement <4 x i32> poison, i32 %sched.n1432, i64 0, !dbg !19
  %sched.n1434 = lshr i64 %sched.n1429, 32, !dbg !19
  %sched.n1435 = trunc i64 %sched.n1434 to i32, !dbg !19
  %sched.n1436 = insertelement <4 x i32> %sched.n1433, i32 %sched.n1435, i64 1, !dbg !19
  %sched.n1437 = insertelement <4 x i32> %sched.n1436, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc = insertelement <4 x i32> %sched.n1437, i32 131072, i64 3, !dbg !19
  %sched.n1438 = ptrtoint ptr addrspace(3) %sched.n1428 to i32, !dbg !19
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sched.n1438), !dbg !19
  %sched.n1439 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !19
  %sched.n1440 = shufflevector <8 x half> %sched.n1395, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1441 = extractvalue [2 x <8 x float>] %sched.n101, 0, !dbg !31
  %sched.n1442 = shufflevector <8 x float> %sched.n1441, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %sched.n1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1440, <4 x float> %sched.n1442) #9, !dbg !15
  %sched.n1444 = shufflevector <8 x half> %sched.n1397, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1444, <4 x float> %sched.n1443) #9, !dbg !15
  %sched.n1446 = shufflevector <8 x half> %sched.n1399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1446, <4 x float> %sched.n1445) #9, !dbg !15
  %sched.n1448 = shufflevector <8 x half> %sched.n1401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1448, <4 x float> %sched.n1447) #9, !dbg !15
  %sched.n1450 = shufflevector <8 x half> %sched.n1403, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1450, <4 x float> %sched.n1449) #9, !dbg !15
  %sched.n1452 = shufflevector <8 x half> %sched.n1405, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1452, <4 x float> %sched.n1451) #9, !dbg !15
  %sched.n1454 = shufflevector <8 x half> %sched.n1407, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1454, <4 x float> %sched.n1453) #9, !dbg !15
  %sched.n1456 = shufflevector <8 x half> %sched.n1409, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1456, <4 x float> %sched.n1455) #9, !dbg !15
  %sched.n1458 = shufflevector <8 x half> %sched.n1395, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1459 = shufflevector <8 x float> %sched.n1441, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %sched.n1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1458, <4 x float> %sched.n1459) #9, !dbg !15
  %sched.n1461 = shufflevector <8 x half> %sched.n1397, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1461, <4 x float> %sched.n1460) #9, !dbg !15
  %sched.n1463 = shufflevector <8 x half> %sched.n1399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1463, <4 x float> %sched.n1462) #9, !dbg !15
  %sched.n1465 = shufflevector <8 x half> %sched.n1401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1465, <4 x float> %sched.n1464) #9, !dbg !15
  %sched.n1467 = shufflevector <8 x half> %sched.n1403, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1467, <4 x float> %sched.n1466) #9, !dbg !15
  %sched.n1469 = shufflevector <8 x half> %sched.n1405, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1469, <4 x float> %sched.n1468) #9, !dbg !15
  %sched.n1471 = shufflevector <8 x half> %sched.n1407, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1471, <4 x float> %sched.n1470) #9, !dbg !15
  %sched.n1473 = shufflevector <8 x half> %sched.n1409, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1473, <4 x float> %sched.n1472) #9, !dbg !15
  %sched.n1475 = extractvalue [2 x <8 x float>] %sched.n101, 1, !dbg !31
  %sched.n1476 = shufflevector <8 x float> %sched.n1475, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %sched.n1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1440, <4 x float> %sched.n1476) #9, !dbg !15
  %sched.n1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1444, <4 x float> %sched.n1477) #9, !dbg !15
  %sched.n1479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1446, <4 x float> %sched.n1478) #9, !dbg !15
  %sched.n1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1448, <4 x float> %sched.n1479) #9, !dbg !15
  %sched.n1481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1450, <4 x float> %sched.n1480) #9, !dbg !15
  %sched.n1482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1452, <4 x float> %sched.n1481) #9, !dbg !15
  %sched.n1483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1454, <4 x float> %sched.n1482) #9, !dbg !15
  %sched.n1484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1456, <4 x float> %sched.n1483) #9, !dbg !15
  %sched.n1485 = shufflevector <8 x float> %sched.n1475, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %sched.n1486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1458, <4 x float> %sched.n1485) #9, !dbg !15
  %sched.n1487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1461, <4 x float> %sched.n1486) #9, !dbg !15
  %sched.n1488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1463, <4 x float> %sched.n1487) #9, !dbg !15
  %sched.n1489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1465, <4 x float> %sched.n1488) #9, !dbg !15
  %sched.n1490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1467, <4 x float> %sched.n1489) #9, !dbg !15
  %sched.n1491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1469, <4 x float> %sched.n1490) #9, !dbg !15
  %sched.n1492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1471, <4 x float> %sched.n1491) #9, !dbg !15
  %sched.n1493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1473, <4 x float> %sched.n1492) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %sched.n1494 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #0, !dbg !15
  %sched.n1495 = bitcast <4 x i32> %sched.n1494 to <8 x half>, !dbg !15
  %sched.n1496 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #0, !dbg !15
  %sched.n1497 = bitcast <4 x i32> %sched.n1496 to <8 x half>, !dbg !15
  %sched.n1498 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #0, !dbg !15
  %sched.n1499 = bitcast <4 x i32> %sched.n1498 to <8 x half>, !dbg !15
  %sched.n1500 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #0, !dbg !15
  %sched.n1501 = bitcast <4 x i32> %sched.n1500 to <8 x half>, !dbg !15
  %sched.n1502 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #0, !dbg !15
  %sched.n1503 = bitcast <4 x i32> %sched.n1502 to <8 x half>, !dbg !15
  %sched.n1504 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #0, !dbg !15
  %sched.n1505 = bitcast <4 x i32> %sched.n1504 to <8 x half>, !dbg !15
  %sched.n1506 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #0, !dbg !15
  %sched.n1507 = bitcast <4 x i32> %sched.n1506 to <8 x half>, !dbg !15
  %sched.n1508 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #0, !dbg !15
  %sched.n1509 = bitcast <4 x i32> %sched.n1508 to <8 x half>, !dbg !15
  %sched.n1510 = shufflevector <8 x half> %sched.n1495, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1511 = extractvalue [2 x <8 x float>] %sched.n102, 0, !dbg !31
  %sched.n1512 = shufflevector <8 x float> %sched.n1511, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %sched.n1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1510, <4 x float> %sched.n1512) #9, !dbg !15
  %sched.n1514 = shufflevector <8 x half> %sched.n1497, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1514, <4 x float> %sched.n1513) #9, !dbg !15
  %sched.n1516 = shufflevector <8 x half> %sched.n1499, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1516, <4 x float> %sched.n1515) #9, !dbg !15
  %sched.n1518 = shufflevector <8 x half> %sched.n1501, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1518, <4 x float> %sched.n1517) #9, !dbg !15
  %sched.n1520 = shufflevector <8 x half> %sched.n1503, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1520, <4 x float> %sched.n1519) #9, !dbg !15
  %sched.n1522 = shufflevector <8 x half> %sched.n1505, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1522, <4 x float> %sched.n1521) #9, !dbg !15
  %sched.n1524 = shufflevector <8 x half> %sched.n1507, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1524, <4 x float> %sched.n1523) #9, !dbg !15
  %sched.n1526 = shufflevector <8 x half> %sched.n1509, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %sched.n1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1526, <4 x float> %sched.n1525) #9, !dbg !15
  %sched.n1528 = shufflevector <8 x half> %sched.n1495, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1529 = shufflevector <8 x float> %sched.n1511, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %sched.n1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %sched.n1528, <4 x float> %sched.n1529) #9, !dbg !15
  %sched.n1531 = shufflevector <8 x half> %sched.n1497, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %sched.n1531, <4 x float> %sched.n1530) #9, !dbg !15
  %sched.n1533 = shufflevector <8 x half> %sched.n1499, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %sched.n1533, <4 x float> %sched.n1532) #9, !dbg !15
  %sched.n1535 = shufflevector <8 x half> %sched.n1501, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %sched.n1535, <4 x float> %sched.n1534) #9, !dbg !15
  %sched.n1537 = shufflevector <8 x half> %sched.n1503, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %sched.n1537, <4 x float> %sched.n1536) #9, !dbg !15
  %sched.n1539 = shufflevector <8 x half> %sched.n1505, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %sched.n1539, <4 x float> %sched.n1538) #9, !dbg !15
  %sched.n1541 = shufflevector <8 x half> %sched.n1507, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %sched.n1541, <4 x float> %sched.n1540) #9, !dbg !15
  %sched.n1543 = shufflevector <8 x half> %sched.n1509, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %sched.n1544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %sched.n1543, <4 x float> %sched.n1542) #9, !dbg !15
  %sched.n1545 = extractvalue [2 x <8 x float>] %sched.n102, 1, !dbg !31
  %sched.n1546 = shufflevector <8 x float> %sched.n1545, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %sched.n1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1510, <4 x float> %sched.n1546) #9, !dbg !15
  %sched.n1548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1514, <4 x float> %sched.n1547) #9, !dbg !15
  %sched.n1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1516, <4 x float> %sched.n1548) #9, !dbg !15
  %sched.n1550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1518, <4 x float> %sched.n1549) #9, !dbg !15
  %sched.n1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1520, <4 x float> %sched.n1550) #9, !dbg !15
  %sched.n1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1522, <4 x float> %sched.n1551) #9, !dbg !15
  %sched.n1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1524, <4 x float> %sched.n1552) #9, !dbg !15
  %sched.n1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1526, <4 x float> %sched.n1553) #9, !dbg !15
  %sched.n1555 = shufflevector <8 x float> %sched.n1545, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %sched.n1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %sched.n1528, <4 x float> %sched.n1555) #9, !dbg !15
  %sched.n1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %sched.n1531, <4 x float> %sched.n1556) #9, !dbg !15
  %sched.n1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %sched.n1533, <4 x float> %sched.n1557) #9, !dbg !15
  %sched.n1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %sched.n1535, <4 x float> %sched.n1558) #9, !dbg !15
  %sched.n1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %sched.n1537, <4 x float> %sched.n1559) #9, !dbg !15
  %sched.n1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %sched.n1539, <4 x float> %sched.n1560) #9, !dbg !15
  %sched.n1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %sched.n1541, <4 x float> %sched.n1561) #9, !dbg !15
  %sched.n1563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %sched.n1543, <4 x float> %sched.n1562) #9, !dbg !15
  %sched.n1564 = shufflevector <4 x float> %sched.n1257, <4 x float> %sched.n1274, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %sched.n1565 = insertvalue [2 x <8 x float>] poison, <8 x float> %sched.n1564, 0, !dbg !12
  %sched.n1566 = shufflevector <4 x float> %sched.n1284, <4 x float> %sched.n1293, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %sched.n1567 = insertvalue [2 x <8 x float>] %sched.n1565, <8 x float> %sched.n1566, 1, !dbg !12
  %sched.n1568 = shufflevector <4 x float> %sched.n1357, <4 x float> %sched.n1374, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %sched.n1569 = insertvalue [2 x <8 x float>] poison, <8 x float> %sched.n1568, 0, !dbg !12
  %sched.n1570 = shufflevector <4 x float> %sched.n1384, <4 x float> %sched.n1393, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %sched.n1571 = insertvalue [2 x <8 x float>] %sched.n1569, <8 x float> %sched.n1570, 1, !dbg !12
  %sched.n1572 = shufflevector <4 x float> %sched.n1457, <4 x float> %sched.n1474, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %sched.n1573 = insertvalue [2 x <8 x float>] poison, <8 x float> %sched.n1572, 0, !dbg !12
  %sched.n1574 = shufflevector <4 x float> %sched.n1484, <4 x float> %sched.n1493, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %sched.n1575 = insertvalue [2 x <8 x float>] %sched.n1573, <8 x float> %sched.n1574, 1, !dbg !12
  %sched.n1576 = shufflevector <4 x float> %sched.n1527, <4 x float> %sched.n1544, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %sched.n1577 = insertvalue [2 x <8 x float>] poison, <8 x float> %sched.n1576, 0, !dbg !12
  %sched.n1578 = shufflevector <4 x float> %sched.n1554, <4 x float> %sched.n1563, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %sched.n1579 = insertvalue [2 x <8 x float>] %sched.n1577, <8 x float> %sched.n1578, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %sched.n1580 = add nuw nsw i64 %sched.n97, 1, !dbg !12
  br label %sched.n96, !dbg !12

sched.n1581:                                             ; preds = %sched.n96
  %sched.n1582 = and i32 %sched.n31, 16, !dbg !12
  %.not = icmp eq i32 %sched.n1582, 0, !dbg !12
  %sched.n1583 = and i32 %sched.n31, 32, !dbg !12
  %.not36 = icmp eq i32 %sched.n1583, 0, !dbg !12
  %sched.n1584 = extractvalue [2 x <8 x float>] %sched.n102, 0, !dbg !12
  %sched.n1585 = extractelement <8 x float> %sched.n1584, i64 0, !dbg !12
  %sched.n1586 = extractelement <8 x float> %sched.n1584, i64 1, !dbg !12
  %sched.n1587 = extractelement <8 x float> %sched.n1584, i64 2, !dbg !12
  %sched.n1588 = extractelement <8 x float> %sched.n1584, i64 3, !dbg !12
  %sched.n1589 = select i1 %.not, float %sched.n1586, float %sched.n1585, !dbg !12
  %sched.n1590 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !12
  %sched.n1591 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %sched.n1590), !dbg !12
  %sched.n1592 = add nuw nsw i32 %sched.n1591, 64, !dbg !12
  %sched.n1593 = xor i32 %sched.n1591, 16, !dbg !12
  %sched.n1594 = and i32 %sched.n1592, 192, !dbg !12
  %sched.n1595 = icmp samesign ult i32 %sched.n1593, %sched.n1594, !dbg !12
  %sched.n1596 = select i1 %sched.n1595, i32 %sched.n1593, i32 %sched.n1591, !dbg !12
  %sched.n1597 = shl nuw nsw i32 %sched.n1596, 2, !dbg !12
  %sched.n1598 = bitcast float %sched.n1589 to i32, !dbg !12
  %sched.n1599 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1598), !dbg !12
  %sched.n1600 = bitcast i32 %sched.n1599 to float, !dbg !12
  %sched.n1601 = select i1 %.not, float %sched.n1585, float %sched.n1600, !dbg !12
  %sched.n1602 = select i1 %.not, float %sched.n1600, float %sched.n1586, !dbg !12
  %sched.n1603 = select i1 %.not, float %sched.n1588, float %sched.n1587, !dbg !12
  %sched.n1604 = bitcast float %sched.n1603 to i32, !dbg !12
  %sched.n1605 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1604), !dbg !12
  %sched.n1606 = bitcast i32 %sched.n1605 to float, !dbg !12
  %sched.n1607 = select i1 %.not, float %sched.n1587, float %sched.n1606, !dbg !12
  %sched.n1608 = select i1 %.not, float %sched.n1606, float %sched.n1588, !dbg !12
  %sched.n1609 = select i1 %.not36, float %sched.n1607, float %sched.n1601, !dbg !12
  %sched.n1610 = xor i32 %sched.n1591, 32, !dbg !12
  %sched.n1611 = icmp samesign ult i32 %sched.n1610, %sched.n1594, !dbg !12
  %sched.n1612 = select i1 %sched.n1611, i32 %sched.n1610, i32 %sched.n1591, !dbg !12
  %sched.n1613 = shl nuw nsw i32 %sched.n1612, 2, !dbg !12
  %sched.n1614 = bitcast float %sched.n1609 to i32, !dbg !12
  %sched.n1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1614), !dbg !12
  %sched.n1616 = bitcast i32 %sched.n1615 to float, !dbg !12
  %sched.n1617 = select i1 %.not36, float %sched.n1601, float %sched.n1616, !dbg !12
  %sched.n1618 = select i1 %.not36, float %sched.n1616, float %sched.n1607, !dbg !12
  %sched.n1619 = select i1 %.not36, float %sched.n1608, float %sched.n1602, !dbg !12
  %sched.n1620 = bitcast float %sched.n1619 to i32, !dbg !12
  %sched.n1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1620), !dbg !12
  %sched.n1622 = bitcast i32 %sched.n1621 to float, !dbg !12
  %sched.n1623 = select i1 %.not36, float %sched.n1602, float %sched.n1622, !dbg !12
  %sched.n1624 = select i1 %.not36, float %sched.n1622, float %sched.n1608, !dbg !12
  %sched.n1625 = extractvalue [2 x <8 x float>] %sched.n101, 0, !dbg !12
  %sched.n1626 = extractelement <8 x float> %sched.n1625, i64 0, !dbg !12
  %sched.n1627 = extractelement <8 x float> %sched.n1625, i64 1, !dbg !12
  %sched.n1628 = extractelement <8 x float> %sched.n1625, i64 2, !dbg !12
  %sched.n1629 = extractelement <8 x float> %sched.n1625, i64 3, !dbg !12
  %sched.n1630 = select i1 %.not, float %sched.n1627, float %sched.n1626, !dbg !12
  %sched.n1631 = bitcast float %sched.n1630 to i32, !dbg !12
  %sched.n1632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1631), !dbg !12
  %sched.n1633 = bitcast i32 %sched.n1632 to float, !dbg !12
  %sched.n1634 = select i1 %.not, float %sched.n1626, float %sched.n1633, !dbg !12
  %sched.n1635 = select i1 %.not, float %sched.n1633, float %sched.n1627, !dbg !12
  %sched.n1636 = select i1 %.not, float %sched.n1629, float %sched.n1628, !dbg !12
  %sched.n1637 = bitcast float %sched.n1636 to i32, !dbg !12
  %sched.n1638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1637), !dbg !12
  %sched.n1639 = bitcast i32 %sched.n1638 to float, !dbg !12
  %sched.n1640 = select i1 %.not, float %sched.n1628, float %sched.n1639, !dbg !12
  %sched.n1641 = select i1 %.not, float %sched.n1639, float %sched.n1629, !dbg !12
  %sched.n1642 = select i1 %.not36, float %sched.n1640, float %sched.n1634, !dbg !12
  %sched.n1643 = bitcast float %sched.n1642 to i32, !dbg !12
  %sched.n1644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1643), !dbg !12
  %sched.n1645 = bitcast i32 %sched.n1644 to float, !dbg !12
  %sched.n1646 = select i1 %.not36, float %sched.n1634, float %sched.n1645, !dbg !12
  %sched.n1647 = select i1 %.not36, float %sched.n1645, float %sched.n1640, !dbg !12
  %sched.n1648 = select i1 %.not36, float %sched.n1641, float %sched.n1635, !dbg !12
  %sched.n1649 = bitcast float %sched.n1648 to i32, !dbg !12
  %sched.n1650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1649), !dbg !12
  %sched.n1651 = bitcast i32 %sched.n1650 to float, !dbg !12
  %sched.n1652 = select i1 %.not36, float %sched.n1635, float %sched.n1651, !dbg !12
  %sched.n1653 = select i1 %.not36, float %sched.n1651, float %sched.n1641, !dbg !12
  %sched.n1654 = extractvalue [2 x <8 x float>] %sched.n100, 0, !dbg !12
  %sched.n1655 = extractelement <8 x float> %sched.n1654, i64 0, !dbg !12
  %sched.n1656 = extractelement <8 x float> %sched.n1654, i64 1, !dbg !12
  %sched.n1657 = extractelement <8 x float> %sched.n1654, i64 2, !dbg !12
  %sched.n1658 = extractelement <8 x float> %sched.n1654, i64 3, !dbg !12
  %sched.n1659 = select i1 %.not, float %sched.n1656, float %sched.n1655, !dbg !12
  %sched.n1660 = bitcast float %sched.n1659 to i32, !dbg !12
  %sched.n1661 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1660), !dbg !12
  %sched.n1662 = bitcast i32 %sched.n1661 to float, !dbg !12
  %sched.n1663 = select i1 %.not, float %sched.n1655, float %sched.n1662, !dbg !12
  %sched.n1664 = select i1 %.not, float %sched.n1662, float %sched.n1656, !dbg !12
  %sched.n1665 = select i1 %.not, float %sched.n1658, float %sched.n1657, !dbg !12
  %sched.n1666 = bitcast float %sched.n1665 to i32, !dbg !12
  %sched.n1667 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1666), !dbg !12
  %sched.n1668 = bitcast i32 %sched.n1667 to float, !dbg !12
  %sched.n1669 = select i1 %.not, float %sched.n1657, float %sched.n1668, !dbg !12
  %sched.n1670 = select i1 %.not, float %sched.n1668, float %sched.n1658, !dbg !12
  %sched.n1671 = select i1 %.not36, float %sched.n1669, float %sched.n1663, !dbg !12
  %sched.n1672 = bitcast float %sched.n1671 to i32, !dbg !12
  %sched.n1673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1672), !dbg !12
  %sched.n1674 = bitcast i32 %sched.n1673 to float, !dbg !12
  %sched.n1675 = select i1 %.not36, float %sched.n1663, float %sched.n1674, !dbg !12
  %sched.n1676 = select i1 %.not36, float %sched.n1674, float %sched.n1669, !dbg !12
  %sched.n1677 = select i1 %.not36, float %sched.n1670, float %sched.n1664, !dbg !12
  %sched.n1678 = bitcast float %sched.n1677 to i32, !dbg !12
  %sched.n1679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1678), !dbg !12
  %sched.n1680 = bitcast i32 %sched.n1679 to float, !dbg !12
  %sched.n1681 = select i1 %.not36, float %sched.n1664, float %sched.n1680, !dbg !12
  %sched.n1682 = select i1 %.not36, float %sched.n1680, float %sched.n1670, !dbg !12
  %sched.n1683 = extractvalue [2 x <8 x float>] %sched.n98, 0, !dbg !12
  %sched.n1684 = extractelement <8 x float> %sched.n1683, i64 0, !dbg !12
  %sched.n1685 = extractelement <8 x float> %sched.n1683, i64 1, !dbg !12
  %sched.n1686 = extractelement <8 x float> %sched.n1683, i64 2, !dbg !12
  %sched.n1687 = extractelement <8 x float> %sched.n1683, i64 3, !dbg !12
  %sched.n1688 = select i1 %.not, float %sched.n1685, float %sched.n1684, !dbg !12
  %sched.n1689 = bitcast float %sched.n1688 to i32, !dbg !12
  %sched.n1690 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1689), !dbg !12
  %sched.n1691 = bitcast i32 %sched.n1690 to float, !dbg !12
  %sched.n1692 = select i1 %.not, float %sched.n1684, float %sched.n1691, !dbg !12
  %sched.n1693 = select i1 %.not, float %sched.n1691, float %sched.n1685, !dbg !12
  %sched.n1694 = select i1 %.not, float %sched.n1687, float %sched.n1686, !dbg !12
  %sched.n1695 = bitcast float %sched.n1694 to i32, !dbg !12
  %sched.n1696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1695), !dbg !12
  %sched.n1697 = bitcast i32 %sched.n1696 to float, !dbg !12
  %sched.n1698 = select i1 %.not, float %sched.n1686, float %sched.n1697, !dbg !12
  %sched.n1699 = select i1 %.not, float %sched.n1697, float %sched.n1687, !dbg !12
  %sched.n1700 = select i1 %.not36, float %sched.n1698, float %sched.n1692, !dbg !12
  %sched.n1701 = bitcast float %sched.n1700 to i32, !dbg !12
  %sched.n1702 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1701), !dbg !12
  %sched.n1703 = select i1 %.not36, float %sched.n1699, float %sched.n1693, !dbg !12
  %sched.n1704 = bitcast float %sched.n1703 to i32, !dbg !12
  %sched.n1705 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1704), !dbg !12
  %sched.n1706 = extractvalue [2 x <1 x float>] %sched.n99, 0, !dbg !20
  %sched.n1707 = shufflevector <1 x float> %sched.n1706, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %sched.n1708 = extractelement <8 x float> %sched.n1584, i64 4, !dbg !12
  %sched.n1709 = extractelement <8 x float> %sched.n1584, i64 5, !dbg !12
  %sched.n1710 = extractelement <8 x float> %sched.n1584, i64 6, !dbg !12
  %sched.n1711 = extractelement <8 x float> %sched.n1584, i64 7, !dbg !12
  %sched.n1712 = select i1 %.not, float %sched.n1709, float %sched.n1708, !dbg !12
  %sched.n1713 = bitcast float %sched.n1712 to i32, !dbg !12
  %sched.n1714 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1713), !dbg !12
  %sched.n1715 = bitcast i32 %sched.n1714 to float, !dbg !12
  %sched.n1716 = select i1 %.not, float %sched.n1708, float %sched.n1715, !dbg !12
  %sched.n1717 = select i1 %.not, float %sched.n1715, float %sched.n1709, !dbg !12
  %sched.n1718 = select i1 %.not, float %sched.n1711, float %sched.n1710, !dbg !12
  %sched.n1719 = bitcast float %sched.n1718 to i32, !dbg !12
  %sched.n1720 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1719), !dbg !12
  %sched.n1721 = bitcast i32 %sched.n1720 to float, !dbg !12
  %sched.n1722 = select i1 %.not, float %sched.n1710, float %sched.n1721, !dbg !12
  %sched.n1723 = select i1 %.not, float %sched.n1721, float %sched.n1711, !dbg !12
  %sched.n1724 = select i1 %.not36, float %sched.n1722, float %sched.n1716, !dbg !12
  %sched.n1725 = bitcast float %sched.n1724 to i32, !dbg !12
  %sched.n1726 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1725), !dbg !12
  %sched.n1727 = bitcast i32 %sched.n1726 to float, !dbg !12
  %sched.n1728 = select i1 %.not36, float %sched.n1716, float %sched.n1727, !dbg !12
  %sched.n1729 = select i1 %.not36, float %sched.n1727, float %sched.n1722, !dbg !12
  %sched.n1730 = select i1 %.not36, float %sched.n1723, float %sched.n1717, !dbg !12
  %sched.n1731 = bitcast float %sched.n1730 to i32, !dbg !12
  %sched.n1732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1731), !dbg !12
  %sched.n1733 = bitcast i32 %sched.n1732 to float, !dbg !12
  %sched.n1734 = select i1 %.not36, float %sched.n1717, float %sched.n1733, !dbg !12
  %sched.n1735 = select i1 %.not36, float %sched.n1733, float %sched.n1723, !dbg !12
  %sched.n1736 = extractelement <8 x float> %sched.n1625, i64 4, !dbg !12
  %sched.n1737 = extractelement <8 x float> %sched.n1625, i64 5, !dbg !12
  %sched.n1738 = extractelement <8 x float> %sched.n1625, i64 6, !dbg !12
  %sched.n1739 = extractelement <8 x float> %sched.n1625, i64 7, !dbg !12
  %sched.n1740 = select i1 %.not, float %sched.n1737, float %sched.n1736, !dbg !12
  %sched.n1741 = bitcast float %sched.n1740 to i32, !dbg !12
  %sched.n1742 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1741), !dbg !12
  %sched.n1743 = bitcast i32 %sched.n1742 to float, !dbg !12
  %sched.n1744 = select i1 %.not, float %sched.n1736, float %sched.n1743, !dbg !12
  %sched.n1745 = select i1 %.not, float %sched.n1743, float %sched.n1737, !dbg !12
  %sched.n1746 = select i1 %.not, float %sched.n1739, float %sched.n1738, !dbg !12
  %sched.n1747 = bitcast float %sched.n1746 to i32, !dbg !12
  %sched.n1748 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1747), !dbg !12
  %sched.n1749 = bitcast i32 %sched.n1748 to float, !dbg !12
  %sched.n1750 = select i1 %.not, float %sched.n1738, float %sched.n1749, !dbg !12
  %sched.n1751 = select i1 %.not, float %sched.n1749, float %sched.n1739, !dbg !12
  %sched.n1752 = select i1 %.not36, float %sched.n1750, float %sched.n1744, !dbg !12
  %sched.n1753 = bitcast float %sched.n1752 to i32, !dbg !12
  %sched.n1754 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1753), !dbg !12
  %sched.n1755 = bitcast i32 %sched.n1754 to float, !dbg !12
  %sched.n1756 = select i1 %.not36, float %sched.n1744, float %sched.n1755, !dbg !12
  %sched.n1757 = select i1 %.not36, float %sched.n1755, float %sched.n1750, !dbg !12
  %sched.n1758 = select i1 %.not36, float %sched.n1751, float %sched.n1745, !dbg !12
  %sched.n1759 = bitcast float %sched.n1758 to i32, !dbg !12
  %sched.n1760 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1759), !dbg !12
  %sched.n1761 = bitcast i32 %sched.n1760 to float, !dbg !12
  %sched.n1762 = select i1 %.not36, float %sched.n1745, float %sched.n1761, !dbg !12
  %sched.n1763 = select i1 %.not36, float %sched.n1761, float %sched.n1751, !dbg !12
  %sched.n1764 = extractelement <8 x float> %sched.n1654, i64 4, !dbg !12
  %sched.n1765 = extractelement <8 x float> %sched.n1654, i64 5, !dbg !12
  %sched.n1766 = extractelement <8 x float> %sched.n1654, i64 6, !dbg !12
  %sched.n1767 = extractelement <8 x float> %sched.n1654, i64 7, !dbg !12
  %sched.n1768 = select i1 %.not, float %sched.n1765, float %sched.n1764, !dbg !12
  %sched.n1769 = bitcast float %sched.n1768 to i32, !dbg !12
  %sched.n1770 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1769), !dbg !12
  %sched.n1771 = bitcast i32 %sched.n1770 to float, !dbg !12
  %sched.n1772 = select i1 %.not, float %sched.n1764, float %sched.n1771, !dbg !12
  %sched.n1773 = select i1 %.not, float %sched.n1771, float %sched.n1765, !dbg !12
  %sched.n1774 = select i1 %.not, float %sched.n1767, float %sched.n1766, !dbg !12
  %sched.n1775 = bitcast float %sched.n1774 to i32, !dbg !12
  %sched.n1776 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1775), !dbg !12
  %sched.n1777 = bitcast i32 %sched.n1776 to float, !dbg !12
  %sched.n1778 = select i1 %.not, float %sched.n1766, float %sched.n1777, !dbg !12
  %sched.n1779 = select i1 %.not, float %sched.n1777, float %sched.n1767, !dbg !12
  %sched.n1780 = select i1 %.not36, float %sched.n1778, float %sched.n1772, !dbg !12
  %sched.n1781 = bitcast float %sched.n1780 to i32, !dbg !12
  %sched.n1782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1781), !dbg !12
  %sched.n1783 = bitcast i32 %sched.n1782 to float, !dbg !12
  %sched.n1784 = select i1 %.not36, float %sched.n1772, float %sched.n1783, !dbg !12
  %sched.n1785 = select i1 %.not36, float %sched.n1783, float %sched.n1778, !dbg !12
  %sched.n1786 = select i1 %.not36, float %sched.n1779, float %sched.n1773, !dbg !12
  %sched.n1787 = bitcast float %sched.n1786 to i32, !dbg !12
  %sched.n1788 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1787), !dbg !12
  %sched.n1789 = bitcast i32 %sched.n1788 to float, !dbg !12
  %sched.n1790 = select i1 %.not36, float %sched.n1773, float %sched.n1789, !dbg !12
  %sched.n1791 = select i1 %.not36, float %sched.n1789, float %sched.n1779, !dbg !12
  %sched.n1792 = extractelement <8 x float> %sched.n1683, i64 4, !dbg !12
  %sched.n1793 = extractelement <8 x float> %sched.n1683, i64 5, !dbg !12
  %sched.n1794 = extractelement <8 x float> %sched.n1683, i64 6, !dbg !12
  %sched.n1795 = extractelement <8 x float> %sched.n1683, i64 7, !dbg !12
  %sched.n1796 = select i1 %.not, float %sched.n1793, float %sched.n1792, !dbg !12
  %sched.n1797 = bitcast float %sched.n1796 to i32, !dbg !12
  %sched.n1798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1797), !dbg !12
  %sched.n1799 = bitcast i32 %sched.n1798 to float, !dbg !12
  %sched.n1800 = select i1 %.not, float %sched.n1792, float %sched.n1799, !dbg !12
  %sched.n1801 = select i1 %.not, float %sched.n1799, float %sched.n1793, !dbg !12
  %sched.n1802 = select i1 %.not, float %sched.n1795, float %sched.n1794, !dbg !12
  %sched.n1803 = bitcast float %sched.n1802 to i32, !dbg !12
  %sched.n1804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1803), !dbg !12
  %sched.n1805 = bitcast i32 %sched.n1804 to float, !dbg !12
  %sched.n1806 = select i1 %.not, float %sched.n1794, float %sched.n1805, !dbg !12
  %sched.n1807 = select i1 %.not, float %sched.n1805, float %sched.n1795, !dbg !12
  %sched.n1808 = select i1 %.not36, float %sched.n1806, float %sched.n1800, !dbg !12
  %sched.n1809 = bitcast float %sched.n1808 to i32, !dbg !12
  %sched.n1810 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1809), !dbg !12
  %sched.n1811 = select i1 %.not36, float %sched.n1807, float %sched.n1801, !dbg !12
  %sched.n1812 = bitcast float %sched.n1811 to i32, !dbg !12
  %sched.n1813 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1812), !dbg !12
  %sched.n1814 = extractvalue [2 x <8 x float>] %sched.n102, 1, !dbg !12
  %sched.n1815 = extractelement <8 x float> %sched.n1814, i64 0, !dbg !12
  %sched.n1816 = extractelement <8 x float> %sched.n1814, i64 1, !dbg !12
  %sched.n1817 = extractelement <8 x float> %sched.n1814, i64 2, !dbg !12
  %sched.n1818 = extractelement <8 x float> %sched.n1814, i64 3, !dbg !12
  %sched.n1819 = select i1 %.not, float %sched.n1816, float %sched.n1815, !dbg !12
  %sched.n1820 = bitcast float %sched.n1819 to i32, !dbg !12
  %sched.n1821 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1820), !dbg !12
  %sched.n1822 = bitcast i32 %sched.n1821 to float, !dbg !12
  %sched.n1823 = select i1 %.not, float %sched.n1815, float %sched.n1822, !dbg !12
  %sched.n1824 = select i1 %.not, float %sched.n1822, float %sched.n1816, !dbg !12
  %sched.n1825 = select i1 %.not, float %sched.n1818, float %sched.n1817, !dbg !12
  %sched.n1826 = bitcast float %sched.n1825 to i32, !dbg !12
  %sched.n1827 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1826), !dbg !12
  %sched.n1828 = bitcast i32 %sched.n1827 to float, !dbg !12
  %sched.n1829 = select i1 %.not, float %sched.n1817, float %sched.n1828, !dbg !12
  %sched.n1830 = select i1 %.not, float %sched.n1828, float %sched.n1818, !dbg !12
  %sched.n1831 = select i1 %.not36, float %sched.n1829, float %sched.n1823, !dbg !12
  %sched.n1832 = bitcast float %sched.n1831 to i32, !dbg !12
  %sched.n1833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1832), !dbg !12
  %sched.n1834 = bitcast i32 %sched.n1833 to float, !dbg !12
  %sched.n1835 = select i1 %.not36, float %sched.n1823, float %sched.n1834, !dbg !12
  %sched.n1836 = select i1 %.not36, float %sched.n1834, float %sched.n1829, !dbg !12
  %sched.n1837 = select i1 %.not36, float %sched.n1830, float %sched.n1824, !dbg !12
  %sched.n1838 = bitcast float %sched.n1837 to i32, !dbg !12
  %sched.n1839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1838), !dbg !12
  %sched.n1840 = bitcast i32 %sched.n1839 to float, !dbg !12
  %sched.n1841 = select i1 %.not36, float %sched.n1824, float %sched.n1840, !dbg !12
  %sched.n1842 = select i1 %.not36, float %sched.n1840, float %sched.n1830, !dbg !12
  %sched.n1843 = extractvalue [2 x <8 x float>] %sched.n101, 1, !dbg !12
  %sched.n1844 = extractelement <8 x float> %sched.n1843, i64 0, !dbg !12
  %sched.n1845 = extractelement <8 x float> %sched.n1843, i64 1, !dbg !12
  %sched.n1846 = extractelement <8 x float> %sched.n1843, i64 2, !dbg !12
  %sched.n1847 = extractelement <8 x float> %sched.n1843, i64 3, !dbg !12
  %sched.n1848 = select i1 %.not, float %sched.n1845, float %sched.n1844, !dbg !12
  %sched.n1849 = bitcast float %sched.n1848 to i32, !dbg !12
  %sched.n1850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1849), !dbg !12
  %sched.n1851 = bitcast i32 %sched.n1850 to float, !dbg !12
  %sched.n1852 = select i1 %.not, float %sched.n1844, float %sched.n1851, !dbg !12
  %sched.n1853 = select i1 %.not, float %sched.n1851, float %sched.n1845, !dbg !12
  %sched.n1854 = select i1 %.not, float %sched.n1847, float %sched.n1846, !dbg !12
  %sched.n1855 = bitcast float %sched.n1854 to i32, !dbg !12
  %sched.n1856 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1855), !dbg !12
  %sched.n1857 = bitcast i32 %sched.n1856 to float, !dbg !12
  %sched.n1858 = select i1 %.not, float %sched.n1846, float %sched.n1857, !dbg !12
  %sched.n1859 = select i1 %.not, float %sched.n1857, float %sched.n1847, !dbg !12
  %sched.n1860 = select i1 %.not36, float %sched.n1858, float %sched.n1852, !dbg !12
  %sched.n1861 = bitcast float %sched.n1860 to i32, !dbg !12
  %sched.n1862 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1861), !dbg !12
  %sched.n1863 = bitcast i32 %sched.n1862 to float, !dbg !12
  %sched.n1864 = select i1 %.not36, float %sched.n1852, float %sched.n1863, !dbg !12
  %sched.n1865 = select i1 %.not36, float %sched.n1863, float %sched.n1858, !dbg !12
  %sched.n1866 = select i1 %.not36, float %sched.n1859, float %sched.n1853, !dbg !12
  %sched.n1867 = bitcast float %sched.n1866 to i32, !dbg !12
  %sched.n1868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1867), !dbg !12
  %sched.n1869 = bitcast i32 %sched.n1868 to float, !dbg !12
  %sched.n1870 = select i1 %.not36, float %sched.n1853, float %sched.n1869, !dbg !12
  %sched.n1871 = select i1 %.not36, float %sched.n1869, float %sched.n1859, !dbg !12
  %sched.n1872 = extractvalue [2 x <8 x float>] %sched.n100, 1, !dbg !12
  %sched.n1873 = extractelement <8 x float> %sched.n1872, i64 0, !dbg !12
  %sched.n1874 = extractelement <8 x float> %sched.n1872, i64 1, !dbg !12
  %sched.n1875 = extractelement <8 x float> %sched.n1872, i64 2, !dbg !12
  %sched.n1876 = extractelement <8 x float> %sched.n1872, i64 3, !dbg !12
  %sched.n1877 = select i1 %.not, float %sched.n1874, float %sched.n1873, !dbg !12
  %sched.n1878 = bitcast float %sched.n1877 to i32, !dbg !12
  %sched.n1879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1878), !dbg !12
  %sched.n1880 = bitcast i32 %sched.n1879 to float, !dbg !12
  %sched.n1881 = select i1 %.not, float %sched.n1873, float %sched.n1880, !dbg !12
  %sched.n1882 = select i1 %.not, float %sched.n1880, float %sched.n1874, !dbg !12
  %sched.n1883 = select i1 %.not, float %sched.n1876, float %sched.n1875, !dbg !12
  %sched.n1884 = bitcast float %sched.n1883 to i32, !dbg !12
  %sched.n1885 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1884), !dbg !12
  %sched.n1886 = bitcast i32 %sched.n1885 to float, !dbg !12
  %sched.n1887 = select i1 %.not, float %sched.n1875, float %sched.n1886, !dbg !12
  %sched.n1888 = select i1 %.not, float %sched.n1886, float %sched.n1876, !dbg !12
  %sched.n1889 = select i1 %.not36, float %sched.n1887, float %sched.n1881, !dbg !12
  %sched.n1890 = bitcast float %sched.n1889 to i32, !dbg !12
  %sched.n1891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1890), !dbg !12
  %sched.n1892 = bitcast i32 %sched.n1891 to float, !dbg !12
  %sched.n1893 = select i1 %.not36, float %sched.n1881, float %sched.n1892, !dbg !12
  %sched.n1894 = select i1 %.not36, float %sched.n1892, float %sched.n1887, !dbg !12
  %sched.n1895 = select i1 %.not36, float %sched.n1888, float %sched.n1882, !dbg !12
  %sched.n1896 = bitcast float %sched.n1895 to i32, !dbg !12
  %sched.n1897 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1896), !dbg !12
  %sched.n1898 = bitcast i32 %sched.n1897 to float, !dbg !12
  %sched.n1899 = select i1 %.not36, float %sched.n1882, float %sched.n1898, !dbg !12
  %sched.n1900 = select i1 %.not36, float %sched.n1898, float %sched.n1888, !dbg !12
  %sched.n1901 = extractvalue [2 x <8 x float>] %sched.n98, 1, !dbg !12
  %sched.n1902 = extractelement <8 x float> %sched.n1901, i64 0, !dbg !12
  %sched.n1903 = extractelement <8 x float> %sched.n1901, i64 1, !dbg !12
  %sched.n1904 = extractelement <8 x float> %sched.n1901, i64 2, !dbg !12
  %sched.n1905 = extractelement <8 x float> %sched.n1901, i64 3, !dbg !12
  %sched.n1906 = select i1 %.not, float %sched.n1903, float %sched.n1902, !dbg !12
  %sched.n1907 = bitcast float %sched.n1906 to i32, !dbg !12
  %sched.n1908 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1907), !dbg !12
  %sched.n1909 = bitcast i32 %sched.n1908 to float, !dbg !12
  %sched.n1910 = select i1 %.not, float %sched.n1902, float %sched.n1909, !dbg !12
  %sched.n1911 = select i1 %.not, float %sched.n1909, float %sched.n1903, !dbg !12
  %sched.n1912 = select i1 %.not, float %sched.n1905, float %sched.n1904, !dbg !12
  %sched.n1913 = bitcast float %sched.n1912 to i32, !dbg !12
  %sched.n1914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1913), !dbg !12
  %sched.n1915 = bitcast i32 %sched.n1914 to float, !dbg !12
  %sched.n1916 = select i1 %.not, float %sched.n1904, float %sched.n1915, !dbg !12
  %sched.n1917 = select i1 %.not, float %sched.n1915, float %sched.n1905, !dbg !12
  %sched.n1918 = select i1 %.not36, float %sched.n1916, float %sched.n1910, !dbg !12
  %sched.n1919 = bitcast float %sched.n1918 to i32, !dbg !12
  %sched.n1920 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1919), !dbg !12
  %sched.n1921 = select i1 %.not36, float %sched.n1917, float %sched.n1911, !dbg !12
  %sched.n1922 = bitcast float %sched.n1921 to i32, !dbg !12
  %sched.n1923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1922), !dbg !12
  %sched.n1924 = extractvalue [2 x <1 x float>] %sched.n99, 1, !dbg !20
  %sched.n1925 = shufflevector <1 x float> %sched.n1924, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %sched.n1926 = extractelement <8 x float> %sched.n1814, i64 4, !dbg !12
  %sched.n1927 = extractelement <8 x float> %sched.n1814, i64 5, !dbg !12
  %sched.n1928 = extractelement <8 x float> %sched.n1814, i64 6, !dbg !12
  %sched.n1929 = extractelement <8 x float> %sched.n1814, i64 7, !dbg !12
  %sched.n1930 = select i1 %.not, float %sched.n1927, float %sched.n1926, !dbg !12
  %sched.n1931 = bitcast float %sched.n1930 to i32, !dbg !12
  %sched.n1932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1931), !dbg !12
  %sched.n1933 = bitcast i32 %sched.n1932 to float, !dbg !12
  %sched.n1934 = select i1 %.not, float %sched.n1926, float %sched.n1933, !dbg !12
  %sched.n1935 = select i1 %.not, float %sched.n1933, float %sched.n1927, !dbg !12
  %sched.n1936 = select i1 %.not, float %sched.n1929, float %sched.n1928, !dbg !12
  %sched.n1937 = bitcast float %sched.n1936 to i32, !dbg !12
  %sched.n1938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1937), !dbg !12
  %sched.n1939 = bitcast i32 %sched.n1938 to float, !dbg !12
  %sched.n1940 = select i1 %.not, float %sched.n1928, float %sched.n1939, !dbg !12
  %sched.n1941 = select i1 %.not, float %sched.n1939, float %sched.n1929, !dbg !12
  %sched.n1942 = select i1 %.not36, float %sched.n1940, float %sched.n1934, !dbg !12
  %sched.n1943 = bitcast float %sched.n1942 to i32, !dbg !12
  %sched.n1944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1943), !dbg !12
  %sched.n1945 = bitcast i32 %sched.n1944 to float, !dbg !12
  %sched.n1946 = select i1 %.not36, float %sched.n1934, float %sched.n1945, !dbg !12
  %sched.n1947 = select i1 %.not36, float %sched.n1945, float %sched.n1940, !dbg !12
  %sched.n1948 = select i1 %.not36, float %sched.n1941, float %sched.n1935, !dbg !12
  %sched.n1949 = bitcast float %sched.n1948 to i32, !dbg !12
  %sched.n1950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1949), !dbg !12
  %sched.n1951 = bitcast i32 %sched.n1950 to float, !dbg !12
  %sched.n1952 = select i1 %.not36, float %sched.n1935, float %sched.n1951, !dbg !12
  %sched.n1953 = select i1 %.not36, float %sched.n1951, float %sched.n1941, !dbg !12
  %sched.n1954 = extractelement <8 x float> %sched.n1843, i64 4, !dbg !12
  %sched.n1955 = extractelement <8 x float> %sched.n1843, i64 5, !dbg !12
  %sched.n1956 = extractelement <8 x float> %sched.n1843, i64 6, !dbg !12
  %sched.n1957 = extractelement <8 x float> %sched.n1843, i64 7, !dbg !12
  %sched.n1958 = select i1 %.not, float %sched.n1955, float %sched.n1954, !dbg !12
  %sched.n1959 = bitcast float %sched.n1958 to i32, !dbg !12
  %sched.n1960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1959), !dbg !12
  %sched.n1961 = bitcast i32 %sched.n1960 to float, !dbg !12
  %sched.n1962 = select i1 %.not, float %sched.n1954, float %sched.n1961, !dbg !12
  %sched.n1963 = select i1 %.not, float %sched.n1961, float %sched.n1955, !dbg !12
  %sched.n1964 = select i1 %.not, float %sched.n1957, float %sched.n1956, !dbg !12
  %sched.n1965 = bitcast float %sched.n1964 to i32, !dbg !12
  %sched.n1966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1965), !dbg !12
  %sched.n1967 = bitcast i32 %sched.n1966 to float, !dbg !12
  %sched.n1968 = select i1 %.not, float %sched.n1956, float %sched.n1967, !dbg !12
  %sched.n1969 = select i1 %.not, float %sched.n1967, float %sched.n1957, !dbg !12
  %sched.n1970 = select i1 %.not36, float %sched.n1968, float %sched.n1962, !dbg !12
  %sched.n1971 = bitcast float %sched.n1970 to i32, !dbg !12
  %sched.n1972 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1971), !dbg !12
  %sched.n1973 = bitcast i32 %sched.n1972 to float, !dbg !12
  %sched.n1974 = select i1 %.not36, float %sched.n1962, float %sched.n1973, !dbg !12
  %sched.n1975 = select i1 %.not36, float %sched.n1973, float %sched.n1968, !dbg !12
  %sched.n1976 = select i1 %.not36, float %sched.n1969, float %sched.n1963, !dbg !12
  %sched.n1977 = bitcast float %sched.n1976 to i32, !dbg !12
  %sched.n1978 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1977), !dbg !12
  %sched.n1979 = bitcast i32 %sched.n1978 to float, !dbg !12
  %sched.n1980 = select i1 %.not36, float %sched.n1963, float %sched.n1979, !dbg !12
  %sched.n1981 = select i1 %.not36, float %sched.n1979, float %sched.n1969, !dbg !12
  %sched.n1982 = extractelement <8 x float> %sched.n1872, i64 4, !dbg !12
  %sched.n1983 = extractelement <8 x float> %sched.n1872, i64 5, !dbg !12
  %sched.n1984 = extractelement <8 x float> %sched.n1872, i64 6, !dbg !12
  %sched.n1985 = extractelement <8 x float> %sched.n1872, i64 7, !dbg !12
  %sched.n1986 = select i1 %.not, float %sched.n1983, float %sched.n1982, !dbg !12
  %sched.n1987 = bitcast float %sched.n1986 to i32, !dbg !12
  %sched.n1988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1987), !dbg !12
  %sched.n1989 = bitcast i32 %sched.n1988 to float, !dbg !12
  %sched.n1990 = select i1 %.not, float %sched.n1982, float %sched.n1989, !dbg !12
  %sched.n1991 = select i1 %.not, float %sched.n1989, float %sched.n1983, !dbg !12
  %sched.n1992 = select i1 %.not, float %sched.n1985, float %sched.n1984, !dbg !12
  %sched.n1993 = bitcast float %sched.n1992 to i32, !dbg !12
  %sched.n1994 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n1993), !dbg !12
  %sched.n1995 = bitcast i32 %sched.n1994 to float, !dbg !12
  %sched.n1996 = select i1 %.not, float %sched.n1984, float %sched.n1995, !dbg !12
  %sched.n1997 = select i1 %.not, float %sched.n1995, float %sched.n1985, !dbg !12
  %sched.n1998 = select i1 %.not36, float %sched.n1996, float %sched.n1990, !dbg !12
  %sched.n1999 = bitcast float %sched.n1998 to i32, !dbg !12
  %sched.n2000 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n1999), !dbg !12
  %sched.n2001 = bitcast i32 %sched.n2000 to float, !dbg !12
  %sched.n2002 = select i1 %.not36, float %sched.n1990, float %sched.n2001, !dbg !12
  %sched.n2003 = select i1 %.not36, float %sched.n2001, float %sched.n1996, !dbg !12
  %sched.n2004 = select i1 %.not36, float %sched.n1997, float %sched.n1991, !dbg !12
  %sched.n2005 = bitcast float %sched.n2004 to i32, !dbg !12
  %sched.n2006 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n2005), !dbg !12
  %sched.n2007 = bitcast i32 %sched.n2006 to float, !dbg !12
  %sched.n2008 = select i1 %.not36, float %sched.n1991, float %sched.n2007, !dbg !12
  %sched.n2009 = select i1 %.not36, float %sched.n2007, float %sched.n1997, !dbg !12
  %sched.n2010 = extractelement <8 x float> %sched.n1901, i64 4, !dbg !12
  %sched.n2011 = extractelement <8 x float> %sched.n1901, i64 5, !dbg !12
  %sched.n2012 = extractelement <8 x float> %sched.n1901, i64 6, !dbg !12
  %sched.n2013 = extractelement <8 x float> %sched.n1901, i64 7, !dbg !12
  %sched.n2014 = select i1 %.not, float %sched.n2011, float %sched.n2010, !dbg !12
  %sched.n2015 = bitcast float %sched.n2014 to i32, !dbg !12
  %sched.n2016 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n2015), !dbg !12
  %sched.n2017 = bitcast i32 %sched.n2016 to float, !dbg !12
  %sched.n2018 = select i1 %.not, float %sched.n2010, float %sched.n2017, !dbg !12
  %sched.n2019 = select i1 %.not, float %sched.n2017, float %sched.n2011, !dbg !12
  %sched.n2020 = select i1 %.not, float %sched.n2013, float %sched.n2012, !dbg !12
  %sched.n2021 = bitcast float %sched.n2020 to i32, !dbg !12
  %sched.n2022 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1597, i32 %sched.n2021), !dbg !12
  %sched.n2023 = bitcast i32 %sched.n2022 to float, !dbg !12
  %sched.n2024 = select i1 %.not, float %sched.n2012, float %sched.n2023, !dbg !12
  %sched.n2025 = select i1 %.not, float %sched.n2023, float %sched.n2013, !dbg !12
  %sched.n2026 = select i1 %.not36, float %sched.n2024, float %sched.n2018, !dbg !12
  %sched.n2027 = bitcast float %sched.n2026 to i32, !dbg !12
  %sched.n2028 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n2027), !dbg !12
  %sched.n2029 = select i1 %.not36, float %sched.n2025, float %sched.n2019, !dbg !12
  %sched.n2030 = bitcast float %sched.n2029 to i32, !dbg !12
  %sched.n2031 = call i32 @llvm.amdgcn.ds.bpermute(i32 %sched.n1613, i32 %sched.n2030), !dbg !12
  br i1 true, label %sched.n2032, label %sched.n2096, !dbg !22

sched.n2032:                                             ; preds = %sched.n1581
  %sched.n2033 = bitcast i32 %sched.n2028 to float, !dbg !12
  %sched.n2034 = select i1 %.not36, float %sched.n2018, float %sched.n2033, !dbg !12
  %sched.n2035 = insertelement <4 x float> poison, float %sched.n2034, i64 0, !dbg !20
  %sched.n2036 = bitcast i32 %sched.n2031 to float, !dbg !12
  %sched.n2037 = select i1 %.not36, float %sched.n2019, float %sched.n2036, !dbg !12
  %sched.n2038 = insertelement <4 x float> %sched.n2035, float %sched.n2037, i64 1, !dbg !20
  %sched.n2039 = select i1 %.not36, float %sched.n2033, float %sched.n2024, !dbg !12
  %sched.n2040 = insertelement <4 x float> %sched.n2038, float %sched.n2039, i64 2, !dbg !20
  %sched.n2041 = select i1 %.not36, float %sched.n2036, float %sched.n2025, !dbg !12
  %sched.n2042 = insertelement <4 x float> %sched.n2040, float %sched.n2041, i64 3, !dbg !20
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %sched.n1925, !dbg !20
  %normalized = fmul <4 x float> %sched.n2042, %row.reciprocal, !dbg !20
  %sched.n2043 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !21
  %sched.n2044 = extractelement <4 x half> %sched.n2043, i64 3, !dbg !21
  %sched.n2045 = extractelement <4 x half> %sched.n2043, i64 2, !dbg !21
  %sched.n2046 = extractelement <4 x half> %sched.n2043, i64 1, !dbg !21
  %sched.n2047 = extractelement <4 x half> %sched.n2043, i64 0, !dbg !21
  %sched.n2048 = bitcast i32 %sched.n1920 to float, !dbg !12
  %sched.n2049 = select i1 %.not36, float %sched.n1910, float %sched.n2048, !dbg !12
  %sched.n2050 = insertelement <4 x float> poison, float %sched.n2049, i64 0, !dbg !20
  %sched.n2051 = bitcast i32 %sched.n1923 to float, !dbg !12
  %sched.n2052 = select i1 %.not36, float %sched.n1911, float %sched.n2051, !dbg !12
  %sched.n2053 = insertelement <4 x float> %sched.n2050, float %sched.n2052, i64 1, !dbg !20
  %sched.n2054 = select i1 %.not36, float %sched.n2048, float %sched.n1916, !dbg !12
  %sched.n2055 = insertelement <4 x float> %sched.n2053, float %sched.n2054, i64 2, !dbg !20
  %sched.n2056 = select i1 %.not36, float %sched.n2051, float %sched.n1917, !dbg !12
  %sched.n2057 = insertelement <4 x float> %sched.n2055, float %sched.n2056, i64 3, !dbg !20
  %normalized104 = fmul <4 x float> %sched.n2057, %row.reciprocal, !dbg !20
  %sched.n2058 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !21
  %sched.n2059 = extractelement <4 x half> %sched.n2058, i64 3, !dbg !21
  %sched.n2060 = extractelement <4 x half> %sched.n2058, i64 2, !dbg !21
  %sched.n2061 = extractelement <4 x half> %sched.n2058, i64 1, !dbg !21
  %sched.n2062 = extractelement <4 x half> %sched.n2058, i64 0, !dbg !21
  %sched.n2063 = bitcast i32 %sched.n1810 to float, !dbg !12
  %sched.n2064 = select i1 %.not36, float %sched.n1800, float %sched.n2063, !dbg !12
  %sched.n2065 = insertelement <4 x float> poison, float %sched.n2064, i64 0, !dbg !20
  %sched.n2066 = bitcast i32 %sched.n1813 to float, !dbg !12
  %sched.n2067 = select i1 %.not36, float %sched.n1801, float %sched.n2066, !dbg !12
  %sched.n2068 = insertelement <4 x float> %sched.n2065, float %sched.n2067, i64 1, !dbg !20
  %sched.n2069 = select i1 %.not36, float %sched.n2063, float %sched.n1806, !dbg !12
  %sched.n2070 = insertelement <4 x float> %sched.n2068, float %sched.n2069, i64 2, !dbg !20
  %sched.n2071 = select i1 %.not36, float %sched.n2066, float %sched.n1807, !dbg !12
  %sched.n2072 = insertelement <4 x float> %sched.n2070, float %sched.n2071, i64 3, !dbg !20
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %sched.n1707, !dbg !20
  %normalized106 = fmul <4 x float> %sched.n2072, %row.reciprocal105, !dbg !20
  %sched.n2073 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !21
  %sched.n2074 = extractelement <4 x half> %sched.n2073, i64 3, !dbg !21
  %sched.n2075 = extractelement <4 x half> %sched.n2073, i64 2, !dbg !21
  %sched.n2076 = extractelement <4 x half> %sched.n2073, i64 1, !dbg !21
  %sched.n2077 = extractelement <4 x half> %sched.n2073, i64 0, !dbg !21
  %sched.n2078 = bitcast i32 %sched.n1702 to float, !dbg !12
  %sched.n2079 = select i1 %.not36, float %sched.n1692, float %sched.n2078, !dbg !12
  %sched.n2080 = insertelement <4 x float> poison, float %sched.n2079, i64 0, !dbg !20
  %sched.n2081 = bitcast i32 %sched.n1705 to float, !dbg !12
  %sched.n2082 = select i1 %.not36, float %sched.n1693, float %sched.n2081, !dbg !12
  %sched.n2083 = insertelement <4 x float> %sched.n2080, float %sched.n2082, i64 1, !dbg !20
  %sched.n2084 = select i1 %.not36, float %sched.n2078, float %sched.n1698, !dbg !12
  %sched.n2085 = insertelement <4 x float> %sched.n2083, float %sched.n2084, i64 2, !dbg !20
  %sched.n2086 = select i1 %.not36, float %sched.n2081, float %sched.n1699, !dbg !12
  %sched.n2087 = insertelement <4 x float> %sched.n2085, float %sched.n2086, i64 3, !dbg !20
  %normalized107 = fmul <4 x float> %sched.n2087, %row.reciprocal105, !dbg !20
  %sched.n2088 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !21
  %sched.n2089 = extractelement <4 x half> %sched.n2088, i64 3, !dbg !21
  %sched.n2090 = extractelement <4 x half> %sched.n2088, i64 2, !dbg !21
  %sched.n2091 = extractelement <4 x half> %sched.n2088, i64 1, !dbg !21
  %sched.n2092 = extractelement <4 x half> %sched.n2088, i64 0, !dbg !21
  %sched.n2093 = add nsw i64 %sched.n16, %sched.n22, !dbg !22
  %sched.n2094 = shl i64 %sched.n2093, 19, !dbg !22
  %sched.n2095 = and i64 %sched.n2094, 16252928, !dbg !22
  %address.thread914 = call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local915 = zext nneg i32 %address.thread914 to i64, !dbg !8
  %address.local916 = add i64 %sched.n25, %address.local915, !dbg !22
  %address.local917 = lshr i64 %address.local915, 6, !dbg !11
  %address.local918 = shl nuw nsw i64 %address.local917, 4, !dbg !22
  %address.local919 = add i64 %address.local916, %address.local918, !dbg !22
  %address.local920 = and i64 %address.local915, 240, !dbg !22
  %address.local921 = sub i64 %address.local919, %address.local920, !dbg !22
  %address.local922 = add i64 %address.local921, %sched.n26, !dbg !22
  %address.local923 = shl nuw nsw i64 %address.local922, 7, !dbg !22
  %address.local924 = add nuw nsw i64 %sched.n2095, %address.local923, !dbg !22
  %address.local925 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local924, !dbg !22
  %address.local926 = lshr i64 %address.local915, 2, !dbg !11
  %address.local927 = and i64 %address.local926, 12, !dbg !11
  %address.local928 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local927, !dbg !22
  store half %sched.n2092, ptr addrspace(1) %address.local928, align 2, !dbg !22
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local924, !dbg !22
  %address.local930 = or disjoint i64 %address.local927, 1, !dbg !22
  %address.local931 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local929, i64 %address.local930, !dbg !22
  store half %sched.n2091, ptr addrspace(1) %address.local931, align 2, !dbg !22
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local924, !dbg !22
  %address.local933 = or disjoint i64 %address.local927, 2, !dbg !22
  %address.local934 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local932, i64 %address.local933, !dbg !22
  store half %sched.n2090, ptr addrspace(1) %address.local934, align 2, !dbg !22
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local924, !dbg !22
  %address.local936 = or disjoint i64 %address.local927, 3, !dbg !22
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local935, i64 %address.local936, !dbg !22
  store half %sched.n2089, ptr addrspace(1) %address.local937, align 2, !dbg !22
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local924, !dbg !22
  %address.local939 = or disjoint i64 %address.local927, 16, !dbg !22
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local938, i64 %address.local939, !dbg !22
  store half %sched.n2077, ptr addrspace(1) %address.local940, align 2, !dbg !22
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local924, !dbg !22
  %address.local942 = or disjoint i64 %address.local927, 17, !dbg !22
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local941, i64 %address.local942, !dbg !22
  store half %sched.n2076, ptr addrspace(1) %address.local943, align 2, !dbg !22
  %address.local944 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local924, !dbg !22
  %address.local945 = or disjoint i64 %address.local927, 18, !dbg !22
  %address.local946 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local945, !dbg !22
  store half %sched.n2075, ptr addrspace(1) %address.local946, align 2, !dbg !22
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local924, !dbg !22
  %address.local948 = or disjoint i64 %address.local927, 19, !dbg !22
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local947, i64 %address.local948, !dbg !22
  store half %sched.n2074, ptr addrspace(1) %address.local949, align 2, !dbg !22
  %address.local950 = shl i64 %address.local922, 7, !dbg !22
  %address.local951 = add i64 %address.local950, 8192, !dbg !22
  %address.local952 = add nuw nsw i64 %sched.n2095, %address.local951, !dbg !22
  %address.local953 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local952, !dbg !22
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local927, !dbg !22
  store half %sched.n2062, ptr addrspace(1) %address.local954, align 2, !dbg !22
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local952, !dbg !22
  %address.local956 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local955, i64 %address.local930, !dbg !22
  store half %sched.n2061, ptr addrspace(1) %address.local956, align 2, !dbg !22
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local952, !dbg !22
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local957, i64 %address.local933, !dbg !22
  store half %sched.n2060, ptr addrspace(1) %address.local958, align 2, !dbg !22
  %address.local959 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local952, !dbg !22
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local959, i64 %address.local936, !dbg !22
  store half %sched.n2059, ptr addrspace(1) %address.local960, align 2, !dbg !22
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local952, !dbg !22
  %address.local962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local961, i64 %address.local939, !dbg !22
  store half %sched.n2047, ptr addrspace(1) %address.local962, align 2, !dbg !22
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local952, !dbg !22
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local963, i64 %address.local942, !dbg !22
  store half %sched.n2046, ptr addrspace(1) %address.local964, align 2, !dbg !22
  %address.local965 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local952, !dbg !22
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local965, i64 %address.local945, !dbg !22
  store half %sched.n2045, ptr addrspace(1) %address.local966, align 2, !dbg !22
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local952, !dbg !22
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local967, i64 %address.local948, !dbg !22
  store half %sched.n2044, ptr addrspace(1) %address.local968, align 2, !dbg !22
  br label %sched.n2096, !dbg !22

sched.n2096:                                             ; preds = %sched.n2032, %sched.n1581
  br i1 true, label %sched.n2097, label %sched.n2135, !dbg !22

sched.n2097:                                             ; preds = %sched.n2096
  %sched.n2098 = insertelement <4 x float> poison, float %sched.n2002, i64 0, !dbg !20
  %sched.n2099 = insertelement <4 x float> %sched.n2098, float %sched.n2008, i64 1, !dbg !20
  %sched.n2100 = insertelement <4 x float> %sched.n2099, float %sched.n2003, i64 2, !dbg !20
  %sched.n2101 = insertelement <4 x float> %sched.n2100, float %sched.n2009, i64 3, !dbg !20
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %sched.n1925, !dbg !20
  %normalized109 = fmul <4 x float> %sched.n2101, %row.reciprocal108, !dbg !20
  %sched.n2102 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !21
  %sched.n2103 = extractelement <4 x half> %sched.n2102, i64 3, !dbg !21
  %sched.n2104 = extractelement <4 x half> %sched.n2102, i64 2, !dbg !21
  %sched.n2105 = extractelement <4 x half> %sched.n2102, i64 1, !dbg !21
  %sched.n2106 = extractelement <4 x half> %sched.n2102, i64 0, !dbg !21
  %sched.n2107 = insertelement <4 x float> poison, float %sched.n1893, i64 0, !dbg !20
  %sched.n2108 = insertelement <4 x float> %sched.n2107, float %sched.n1899, i64 1, !dbg !20
  %sched.n2109 = insertelement <4 x float> %sched.n2108, float %sched.n1894, i64 2, !dbg !20
  %sched.n2110 = insertelement <4 x float> %sched.n2109, float %sched.n1900, i64 3, !dbg !20
  %normalized110 = fmul <4 x float> %sched.n2110, %row.reciprocal108, !dbg !20
  %sched.n2111 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !21
  %sched.n2112 = extractelement <4 x half> %sched.n2111, i64 3, !dbg !21
  %sched.n2113 = extractelement <4 x half> %sched.n2111, i64 2, !dbg !21
  %sched.n2114 = extractelement <4 x half> %sched.n2111, i64 1, !dbg !21
  %sched.n2115 = extractelement <4 x half> %sched.n2111, i64 0, !dbg !21
  %sched.n2116 = insertelement <4 x float> poison, float %sched.n1784, i64 0, !dbg !20
  %sched.n2117 = insertelement <4 x float> %sched.n2116, float %sched.n1790, i64 1, !dbg !20
  %sched.n2118 = insertelement <4 x float> %sched.n2117, float %sched.n1785, i64 2, !dbg !20
  %sched.n2119 = insertelement <4 x float> %sched.n2118, float %sched.n1791, i64 3, !dbg !20
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %sched.n1707, !dbg !20
  %normalized112 = fmul <4 x float> %sched.n2119, %row.reciprocal111, !dbg !20
  %sched.n2120 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !21
  %sched.n2121 = extractelement <4 x half> %sched.n2120, i64 3, !dbg !21
  %sched.n2122 = extractelement <4 x half> %sched.n2120, i64 2, !dbg !21
  %sched.n2123 = extractelement <4 x half> %sched.n2120, i64 1, !dbg !21
  %sched.n2124 = extractelement <4 x half> %sched.n2120, i64 0, !dbg !21
  %sched.n2125 = insertelement <4 x float> poison, float %sched.n1675, i64 0, !dbg !20
  %sched.n2126 = insertelement <4 x float> %sched.n2125, float %sched.n1681, i64 1, !dbg !20
  %sched.n2127 = insertelement <4 x float> %sched.n2126, float %sched.n1676, i64 2, !dbg !20
  %sched.n2128 = insertelement <4 x float> %sched.n2127, float %sched.n1682, i64 3, !dbg !20
  %normalized113 = fmul <4 x float> %sched.n2128, %row.reciprocal111, !dbg !20
  %sched.n2129 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !21
  %sched.n2130 = extractelement <4 x half> %sched.n2129, i64 3, !dbg !21
  %sched.n2131 = extractelement <4 x half> %sched.n2129, i64 2, !dbg !21
  %sched.n2132 = extractelement <4 x half> %sched.n2129, i64 1, !dbg !21
  %sched.n2133 = extractelement <4 x half> %sched.n2129, i64 0, !dbg !21
  %sched.n2134 = shl nuw nsw i64 %sched.n24, 19, !dbg !22
  %address.thread969 = call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local970 = zext nneg i32 %address.thread969 to i64, !dbg !8
  %address.local971 = add i64 %sched.n25, %address.local970, !dbg !22
  %address.local972 = lshr i64 %address.local970, 6, !dbg !11
  %address.local973 = shl nuw nsw i64 %address.local972, 4, !dbg !22
  %address.local974 = add i64 %address.local971, %address.local973, !dbg !22
  %address.local975 = and i64 %address.local970, 240, !dbg !22
  %address.local976 = sub i64 %address.local974, %address.local975, !dbg !22
  %address.local977 = add i64 %address.local976, %sched.n26, !dbg !22
  %address.local978 = shl nuw nsw i64 %address.local977, 7, !dbg !22
  %address.local979 = add nuw nsw i64 %sched.n2134, %address.local978, !dbg !22
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local979, !dbg !22
  %address.local981 = lshr i64 %address.local970, 2, !dbg !11
  %address.local982 = and i64 %address.local981, 12, !dbg !11
  %address.local983 = or disjoint i64 %address.local982, 32, !dbg !22
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local980, i64 %address.local983, !dbg !22
  store half %sched.n2133, ptr addrspace(1) %address.local984, align 2, !dbg !22
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local979, !dbg !22
  %address.local986 = or disjoint i64 %address.local982, 33, !dbg !22
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local986, !dbg !22
  store half %sched.n2132, ptr addrspace(1) %address.local987, align 2, !dbg !22
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local979, !dbg !22
  %address.local989 = or disjoint i64 %address.local982, 34, !dbg !22
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local989, !dbg !22
  store half %sched.n2131, ptr addrspace(1) %address.local990, align 2, !dbg !22
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local979, !dbg !22
  %address.local992 = or disjoint i64 %address.local982, 35, !dbg !22
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local991, i64 %address.local992, !dbg !22
  store half %sched.n2130, ptr addrspace(1) %address.local993, align 2, !dbg !22
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local979, !dbg !22
  %address.local995 = or disjoint i64 %address.local982, 48, !dbg !22
  %address.local996 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local994, i64 %address.local995, !dbg !22
  store half %sched.n2124, ptr addrspace(1) %address.local996, align 2, !dbg !22
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local979, !dbg !22
  %address.local998 = or disjoint i64 %address.local982, 49, !dbg !22
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local998, !dbg !22
  store half %sched.n2123, ptr addrspace(1) %address.local999, align 2, !dbg !22
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local979, !dbg !22
  %address.local1001 = or disjoint i64 %address.local982, 50, !dbg !22
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local1001, !dbg !22
  store half %sched.n2122, ptr addrspace(1) %address.local1002, align 2, !dbg !22
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local979, !dbg !22
  %address.local1004 = or i64 %address.local981, 51, !dbg !22
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1004, !dbg !22
  store half %sched.n2121, ptr addrspace(1) %address.local1005, align 2, !dbg !22
  %address.local1006 = shl i64 %address.local977, 7, !dbg !22
  %address.local1007 = add i64 %address.local1006, 8192, !dbg !22
  %address.local1008 = add nuw nsw i64 %sched.n2134, %address.local1007, !dbg !22
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1008, !dbg !22
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1009, i64 %address.local983, !dbg !22
  store half %sched.n2115, ptr addrspace(1) %address.local1010, align 2, !dbg !22
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1008, !dbg !22
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local986, !dbg !22
  store half %sched.n2114, ptr addrspace(1) %address.local1012, align 2, !dbg !22
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1008, !dbg !22
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local989, !dbg !22
  store half %sched.n2113, ptr addrspace(1) %address.local1014, align 2, !dbg !22
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1008, !dbg !22
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local992, !dbg !22
  store half %sched.n2112, ptr addrspace(1) %address.local1016, align 2, !dbg !22
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1008, !dbg !22
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local995, !dbg !22
  store half %sched.n2106, ptr addrspace(1) %address.local1018, align 2, !dbg !22
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1008, !dbg !22
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local998, !dbg !22
  store half %sched.n2105, ptr addrspace(1) %address.local1020, align 2, !dbg !22
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1008, !dbg !22
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local1001, !dbg !22
  store half %sched.n2104, ptr addrspace(1) %address.local1022, align 2, !dbg !22
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1008, !dbg !22
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1004, !dbg !22
  store half %sched.n2103, ptr addrspace(1) %address.local1024, align 2, !dbg !22
  br label %sched.n2135, !dbg !22

sched.n2135:                                             ; preds = %sched.n2097, %sched.n2096
  br i1 true, label %sched.n2136, label %sched.n2174, !dbg !22

sched.n2136:                                             ; preds = %sched.n2135
  %sched.n2137 = insertelement <4 x float> poison, float %sched.n1974, i64 0, !dbg !20
  %sched.n2138 = insertelement <4 x float> %sched.n2137, float %sched.n1980, i64 1, !dbg !20
  %sched.n2139 = insertelement <4 x float> %sched.n2138, float %sched.n1975, i64 2, !dbg !20
  %sched.n2140 = insertelement <4 x float> %sched.n2139, float %sched.n1981, i64 3, !dbg !20
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %sched.n1925, !dbg !20
  %normalized115 = fmul <4 x float> %sched.n2140, %row.reciprocal114, !dbg !20
  %sched.n2141 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !21
  %sched.n2142 = extractelement <4 x half> %sched.n2141, i64 3, !dbg !21
  %sched.n2143 = extractelement <4 x half> %sched.n2141, i64 2, !dbg !21
  %sched.n2144 = extractelement <4 x half> %sched.n2141, i64 1, !dbg !21
  %sched.n2145 = extractelement <4 x half> %sched.n2141, i64 0, !dbg !21
  %sched.n2146 = insertelement <4 x float> poison, float %sched.n1864, i64 0, !dbg !20
  %sched.n2147 = insertelement <4 x float> %sched.n2146, float %sched.n1870, i64 1, !dbg !20
  %sched.n2148 = insertelement <4 x float> %sched.n2147, float %sched.n1865, i64 2, !dbg !20
  %sched.n2149 = insertelement <4 x float> %sched.n2148, float %sched.n1871, i64 3, !dbg !20
  %normalized116 = fmul <4 x float> %sched.n2149, %row.reciprocal114, !dbg !20
  %sched.n2150 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !21
  %sched.n2151 = extractelement <4 x half> %sched.n2150, i64 3, !dbg !21
  %sched.n2152 = extractelement <4 x half> %sched.n2150, i64 2, !dbg !21
  %sched.n2153 = extractelement <4 x half> %sched.n2150, i64 1, !dbg !21
  %sched.n2154 = extractelement <4 x half> %sched.n2150, i64 0, !dbg !21
  %sched.n2155 = insertelement <4 x float> poison, float %sched.n1756, i64 0, !dbg !20
  %sched.n2156 = insertelement <4 x float> %sched.n2155, float %sched.n1762, i64 1, !dbg !20
  %sched.n2157 = insertelement <4 x float> %sched.n2156, float %sched.n1757, i64 2, !dbg !20
  %sched.n2158 = insertelement <4 x float> %sched.n2157, float %sched.n1763, i64 3, !dbg !20
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %sched.n1707, !dbg !20
  %normalized118 = fmul <4 x float> %sched.n2158, %row.reciprocal117, !dbg !20
  %sched.n2159 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !21
  %sched.n2160 = extractelement <4 x half> %sched.n2159, i64 3, !dbg !21
  %sched.n2161 = extractelement <4 x half> %sched.n2159, i64 2, !dbg !21
  %sched.n2162 = extractelement <4 x half> %sched.n2159, i64 1, !dbg !21
  %sched.n2163 = extractelement <4 x half> %sched.n2159, i64 0, !dbg !21
  %sched.n2164 = insertelement <4 x float> poison, float %sched.n1646, i64 0, !dbg !20
  %sched.n2165 = insertelement <4 x float> %sched.n2164, float %sched.n1652, i64 1, !dbg !20
  %sched.n2166 = insertelement <4 x float> %sched.n2165, float %sched.n1647, i64 2, !dbg !20
  %sched.n2167 = insertelement <4 x float> %sched.n2166, float %sched.n1653, i64 3, !dbg !20
  %normalized119 = fmul <4 x float> %sched.n2167, %row.reciprocal117, !dbg !20
  %sched.n2168 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !21
  %sched.n2169 = extractelement <4 x half> %sched.n2168, i64 3, !dbg !21
  %sched.n2170 = extractelement <4 x half> %sched.n2168, i64 2, !dbg !21
  %sched.n2171 = extractelement <4 x half> %sched.n2168, i64 1, !dbg !21
  %sched.n2172 = extractelement <4 x half> %sched.n2168, i64 0, !dbg !21
  %sched.n2173 = shl nuw nsw i64 %sched.n24, 19, !dbg !22
  %address.thread1025 = call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local1026 = zext nneg i32 %address.thread1025 to i64, !dbg !8
  %address.local1027 = add i64 %sched.n25, %address.local1026, !dbg !22
  %address.local1028 = lshr i64 %address.local1026, 6, !dbg !11
  %address.local1029 = shl nuw nsw i64 %address.local1028, 4, !dbg !22
  %address.local1030 = add i64 %address.local1027, %address.local1029, !dbg !22
  %address.local1031 = and i64 %address.local1026, 240, !dbg !22
  %address.local1032 = sub i64 %address.local1030, %address.local1031, !dbg !22
  %address.local1033 = add i64 %address.local1032, %sched.n26, !dbg !22
  %address.local1034 = shl nuw nsw i64 %address.local1033, 7, !dbg !22
  %address.local1035 = add nuw nsw i64 %sched.n2173, %address.local1034, !dbg !22
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1035, !dbg !22
  %address.local1037 = lshr i64 %address.local1026, 2, !dbg !11
  %address.local1038 = and i64 %address.local1037, 12, !dbg !11
  %address.local1039 = or disjoint i64 %address.local1038, 64, !dbg !22
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1036, i64 %address.local1039, !dbg !22
  store half %sched.n2172, ptr addrspace(1) %address.local1040, align 2, !dbg !22
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1035, !dbg !22
  %address.local1042 = or disjoint i64 %address.local1038, 65, !dbg !22
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1042, !dbg !22
  store half %sched.n2171, ptr addrspace(1) %address.local1043, align 2, !dbg !22
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1035, !dbg !22
  %address.local1045 = or disjoint i64 %address.local1038, 66, !dbg !22
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1044, i64 %address.local1045, !dbg !22
  store half %sched.n2170, ptr addrspace(1) %address.local1046, align 2, !dbg !22
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1035, !dbg !22
  %address.local1048 = or disjoint i64 %address.local1038, 67, !dbg !22
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1047, i64 %address.local1048, !dbg !22
  store half %sched.n2169, ptr addrspace(1) %address.local1049, align 2, !dbg !22
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1035, !dbg !22
  %address.local1051 = or disjoint i64 %address.local1038, 80, !dbg !22
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1051, !dbg !22
  store half %sched.n2163, ptr addrspace(1) %address.local1052, align 2, !dbg !22
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1035, !dbg !22
  %address.local1054 = or disjoint i64 %address.local1038, 81, !dbg !22
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1054, !dbg !22
  store half %sched.n2162, ptr addrspace(1) %address.local1055, align 2, !dbg !22
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1035, !dbg !22
  %address.local1057 = or disjoint i64 %address.local1038, 82, !dbg !22
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1057, !dbg !22
  store half %sched.n2161, ptr addrspace(1) %address.local1058, align 2, !dbg !22
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1035, !dbg !22
  %address.local1060 = or disjoint i64 %address.local1038, 83, !dbg !22
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1059, i64 %address.local1060, !dbg !22
  store half %sched.n2160, ptr addrspace(1) %address.local1061, align 2, !dbg !22
  %address.local1062 = shl i64 %address.local1033, 7, !dbg !22
  %address.local1063 = add i64 %address.local1062, 8192, !dbg !22
  %address.local1064 = add nuw nsw i64 %sched.n2173, %address.local1063, !dbg !22
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !22
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1065, i64 %address.local1039, !dbg !22
  store half %sched.n2154, ptr addrspace(1) %address.local1066, align 2, !dbg !22
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !22
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1042, !dbg !22
  store half %sched.n2153, ptr addrspace(1) %address.local1068, align 2, !dbg !22
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !22
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1045, !dbg !22
  store half %sched.n2152, ptr addrspace(1) %address.local1070, align 2, !dbg !22
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !22
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1048, !dbg !22
  store half %sched.n2151, ptr addrspace(1) %address.local1072, align 2, !dbg !22
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !22
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1051, !dbg !22
  store half %sched.n2145, ptr addrspace(1) %address.local1074, align 2, !dbg !22
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !22
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1054, !dbg !22
  store half %sched.n2144, ptr addrspace(1) %address.local1076, align 2, !dbg !22
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !22
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1057, !dbg !22
  store half %sched.n2143, ptr addrspace(1) %address.local1078, align 2, !dbg !22
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !22
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1060, !dbg !22
  store half %sched.n2142, ptr addrspace(1) %address.local1080, align 2, !dbg !22
  br label %sched.n2174, !dbg !22

sched.n2174:                                             ; preds = %sched.n2136, %sched.n2135
  br i1 true, label %sched.n2175, label %sched.n2213, !dbg !22

sched.n2175:                                             ; preds = %sched.n2174
  %sched.n2176 = insertelement <4 x float> poison, float %sched.n1946, i64 0, !dbg !20
  %sched.n2177 = insertelement <4 x float> %sched.n2176, float %sched.n1952, i64 1, !dbg !20
  %sched.n2178 = insertelement <4 x float> %sched.n2177, float %sched.n1947, i64 2, !dbg !20
  %sched.n2179 = insertelement <4 x float> %sched.n2178, float %sched.n1953, i64 3, !dbg !20
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %sched.n1925, !dbg !20
  %normalized121 = fmul <4 x float> %sched.n2179, %row.reciprocal120, !dbg !20
  %sched.n2180 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !21
  %sched.n2181 = extractelement <4 x half> %sched.n2180, i64 3, !dbg !21
  %sched.n2182 = extractelement <4 x half> %sched.n2180, i64 2, !dbg !21
  %sched.n2183 = extractelement <4 x half> %sched.n2180, i64 1, !dbg !21
  %sched.n2184 = extractelement <4 x half> %sched.n2180, i64 0, !dbg !21
  %sched.n2185 = insertelement <4 x float> poison, float %sched.n1835, i64 0, !dbg !20
  %sched.n2186 = insertelement <4 x float> %sched.n2185, float %sched.n1841, i64 1, !dbg !20
  %sched.n2187 = insertelement <4 x float> %sched.n2186, float %sched.n1836, i64 2, !dbg !20
  %sched.n2188 = insertelement <4 x float> %sched.n2187, float %sched.n1842, i64 3, !dbg !20
  %normalized122 = fmul <4 x float> %sched.n2188, %row.reciprocal120, !dbg !20
  %sched.n2189 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !21
  %sched.n2190 = extractelement <4 x half> %sched.n2189, i64 3, !dbg !21
  %sched.n2191 = extractelement <4 x half> %sched.n2189, i64 2, !dbg !21
  %sched.n2192 = extractelement <4 x half> %sched.n2189, i64 1, !dbg !21
  %sched.n2193 = extractelement <4 x half> %sched.n2189, i64 0, !dbg !21
  %sched.n2194 = insertelement <4 x float> poison, float %sched.n1728, i64 0, !dbg !20
  %sched.n2195 = insertelement <4 x float> %sched.n2194, float %sched.n1734, i64 1, !dbg !20
  %sched.n2196 = insertelement <4 x float> %sched.n2195, float %sched.n1729, i64 2, !dbg !20
  %sched.n2197 = insertelement <4 x float> %sched.n2196, float %sched.n1735, i64 3, !dbg !20
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %sched.n1707, !dbg !20
  %normalized124 = fmul <4 x float> %sched.n2197, %row.reciprocal123, !dbg !20
  %sched.n2198 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !21
  %sched.n2199 = extractelement <4 x half> %sched.n2198, i64 3, !dbg !21
  %sched.n2200 = extractelement <4 x half> %sched.n2198, i64 2, !dbg !21
  %sched.n2201 = extractelement <4 x half> %sched.n2198, i64 1, !dbg !21
  %sched.n2202 = extractelement <4 x half> %sched.n2198, i64 0, !dbg !21
  %sched.n2203 = insertelement <4 x float> poison, float %sched.n1617, i64 0, !dbg !20
  %sched.n2204 = insertelement <4 x float> %sched.n2203, float %sched.n1623, i64 1, !dbg !20
  %sched.n2205 = insertelement <4 x float> %sched.n2204, float %sched.n1618, i64 2, !dbg !20
  %sched.n2206 = insertelement <4 x float> %sched.n2205, float %sched.n1624, i64 3, !dbg !20
  %normalized125 = fmul <4 x float> %sched.n2206, %row.reciprocal123, !dbg !20
  %sched.n2207 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !21
  %sched.n2208 = extractelement <4 x half> %sched.n2207, i64 3, !dbg !21
  %sched.n2209 = extractelement <4 x half> %sched.n2207, i64 2, !dbg !21
  %sched.n2210 = extractelement <4 x half> %sched.n2207, i64 1, !dbg !21
  %sched.n2211 = extractelement <4 x half> %sched.n2207, i64 0, !dbg !21
  %sched.n2212 = shl nuw nsw i64 %sched.n24, 19, !dbg !22
  %address.thread1081 = call i32 asm sideeffect "", "=v,0"(i32 %sched.n31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local1082 = zext nneg i32 %address.thread1081 to i64, !dbg !8
  %address.local1083 = add i64 %sched.n25, %address.local1082, !dbg !22
  %address.local1084 = lshr i64 %address.local1082, 6, !dbg !11
  %address.local1085 = shl nuw nsw i64 %address.local1084, 4, !dbg !22
  %address.local1086 = add i64 %address.local1083, %address.local1085, !dbg !22
  %address.local1087 = and i64 %address.local1082, 240, !dbg !22
  %address.local1088 = sub i64 %address.local1086, %address.local1087, !dbg !22
  %address.local1089 = add i64 %address.local1088, %sched.n26, !dbg !22
  %address.local1090 = shl nuw nsw i64 %address.local1089, 7, !dbg !22
  %address.local1091 = add nuw nsw i64 %sched.n2212, %address.local1090, !dbg !22
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !22
  %address.local1093 = lshr i64 %address.local1082, 2, !dbg !11
  %address.local1094 = and i64 %address.local1093, 12, !dbg !11
  %address.local1095 = or disjoint i64 %address.local1094, 96, !dbg !22
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1095, !dbg !22
  store half %sched.n2211, ptr addrspace(1) %address.local1096, align 2, !dbg !22
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !22
  %address.local1098 = or disjoint i64 %address.local1094, 97, !dbg !22
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1098, !dbg !22
  store half %sched.n2210, ptr addrspace(1) %address.local1099, align 2, !dbg !22
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !22
  %address.local1101 = or disjoint i64 %address.local1094, 98, !dbg !22
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1100, i64 %address.local1101, !dbg !22
  store half %sched.n2209, ptr addrspace(1) %address.local1102, align 2, !dbg !22
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !22
  %address.local1104 = or disjoint i64 %address.local1094, 99, !dbg !22
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1103, i64 %address.local1104, !dbg !22
  store half %sched.n2208, ptr addrspace(1) %address.local1105, align 2, !dbg !22
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !22
  %address.local1107 = or disjoint i64 %address.local1094, 112, !dbg !22
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1106, i64 %address.local1107, !dbg !22
  store half %sched.n2202, ptr addrspace(1) %address.local1108, align 2, !dbg !22
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !22
  %address.local1110 = or disjoint i64 %address.local1094, 113, !dbg !22
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1109, i64 %address.local1110, !dbg !22
  store half %sched.n2201, ptr addrspace(1) %address.local1111, align 2, !dbg !22
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !22
  %address.local1113 = or disjoint i64 %address.local1094, 114, !dbg !22
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1113, !dbg !22
  store half %sched.n2200, ptr addrspace(1) %address.local1114, align 2, !dbg !22
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !22
  %address.local1116 = or i64 %address.local1093, 115, !dbg !22
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1116, !dbg !22
  store half %sched.n2199, ptr addrspace(1) %address.local1117, align 2, !dbg !22
  %address.local1118 = shl i64 %address.local1089, 7, !dbg !22
  %address.local1119 = add i64 %address.local1118, 8192, !dbg !22
  %address.local1120 = add nuw nsw i64 %sched.n2212, %address.local1119, !dbg !22
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1120, !dbg !22
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1095, !dbg !22
  store half %sched.n2193, ptr addrspace(1) %address.local1122, align 2, !dbg !22
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1120, !dbg !22
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1098, !dbg !22
  store half %sched.n2192, ptr addrspace(1) %address.local1124, align 2, !dbg !22
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1120, !dbg !22
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1101, !dbg !22
  store half %sched.n2191, ptr addrspace(1) %address.local1126, align 2, !dbg !22
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1120, !dbg !22
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1104, !dbg !22
  store half %sched.n2190, ptr addrspace(1) %address.local1128, align 2, !dbg !22
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1120, !dbg !22
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1107, !dbg !22
  store half %sched.n2184, ptr addrspace(1) %address.local1130, align 2, !dbg !22
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1120, !dbg !22
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1110, !dbg !22
  store half %sched.n2183, ptr addrspace(1) %address.local1132, align 2, !dbg !22
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1120, !dbg !22
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1113, !dbg !22
  store half %sched.n2182, ptr addrspace(1) %address.local1134, align 2, !dbg !22
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1120, !dbg !22
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1116, !dbg !22
  store half %sched.n2181, ptr addrspace(1) %address.local1136, align 2, !dbg !22
  br label %sched.n2213, !dbg !22

sched.n2213:                                             ; preds = %sched.n2175, %sched.n2174
  ret void, !dbg !5
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
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!33}

!0 = !DIFile(filename: "nohint_unrolled.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!8 = !DILocation(line: 66, column: 22, scope: !4)
!10 = !DILocation(line: 46, column: 17, scope: !4)
!11 = !DILocation(line: 68, column: 24, scope: !4)
!12 = !DILocation(line: 56, column: 24, scope: !4)
!14 = !DILocation(line: 73, column: 19, scope: !4)
!15 = !DILocation(line: 93, column: 23, scope: !4)
!16 = !DILocation(line: 0, scope: !4)
!17 = !DILocation(line: 67, column: 22, scope: !4)
!18 = !DILocation(line: 69, column: 26, scope: !4)
!19 = !DILocation(line: 92, column: 20, scope: !4)
!20 = !DILocation(line: 97, column: 23, scope: !4)
!21 = !DILocation(line: 98, column: 19, scope: !4)
!22 = !DILocation(line: 99, column: 9, scope: !4)
!24 = !DILocation(line: 75, column: 24, scope: !4)
!25 = !DILocation(line: 76, column: 23, scope: !4)
!26 = !DILocation(line: 77, column: 26, scope: !4)
!27 = !DILocation(line: 84, column: 21, scope: !4)
!28 = !DILocation(line: 85, column: 22, scope: !4)
!29 = !DILocation(line: 86, column: 20, scope: !4)
!30 = !DILocation(line: 87, column: 16, scope: !4)
!31 = !DILocation(line: 94, column: 20, scope: !4)
!32 = !{i32 256, i32 1, i32 1}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !{i32 0, i32 256}
