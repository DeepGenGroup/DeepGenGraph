; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !32 {
  %bnk.n5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %bnk.n6 = sext i32 %bnk.n5 to i64, !dbg !5
  %bnk.n7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %bnk.n8 = sext i32 %bnk.n7 to i64, !dbg !5
  %bnk.n9 = shl nsw i64 %bnk.n8, 5, !dbg !5
  %bnk.n10 = add nsw i64 %bnk.n9, %bnk.n6, !dbg !5
  %bnk.n11 = lshr i64 %bnk.n10, 5, !dbg !5
  %bnk.n12 = and i64 %bnk.n11, 576460752303423480, !dbg !5
  %bnk.n13 = and i64 %bnk.n6, 7, !dbg !5
  %bnk.n14 = or disjoint i64 %bnk.n12, %bnk.n13, !dbg !5
  %bnk.n15 = sub nsw i64 31, %bnk.n14, !dbg !5
  %bnk.n16 = lshr i64 %bnk.n10, 3, !dbg !5
  %bnk.n17 = icmp ugt i64 %bnk.n10, 1023, !dbg !8
  %bnk.n18 = add nsw i64 %bnk.n14, -32, !dbg !8
  %bnk.n19 = select i1 %bnk.n17, i64 %bnk.n18, i64 %bnk.n15, !dbg !8
  %bnk.n20 = sdiv i64 %bnk.n19, 32, !dbg !8
  %bnk.n21 = sext i1 %bnk.n17 to i64, !dbg !8
  %bnk.n22 = xor i64 %bnk.n20, %bnk.n21, !dbg !8
  %bnk.n23 = add nsw i64 %bnk.n16, %bnk.n22, !dbg !8
  %bnk.n24 = and i64 %bnk.n23, 31, !dbg !8
  %bnk.n25 = shl nsw i64 %bnk.n15, 7, !dbg !8
  %bnk.n26 = mul nsw i64 %bnk.n22, -4096, !dbg !8
  %bnk.n27 = add i64 %bnk.n25, %bnk.n26, !dbg !8
  %bnk.n28 = shl nuw nsw i64 %bnk.n24, 19, !dbg !8
  %bnk.n29 = shl nsw i64 %bnk.n27, 7, !dbg !8
  %bnk.n30 = add i64 %bnk.n28, %bnk.n29, !dbg !8
  %bnk.n31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !34
  %bnk.n32 = zext nneg i32 %bnk.n31 to i64, !dbg !8
  %bnk.n33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %bnk.n30, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #9, !dbg !8, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local339 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local340 = and i64 %address.local339, 1536, !dbg !8
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !8
  %address.local342 = ptrtoint ptr addrspace(3) %address.local341 to i64, !dbg !8
  %address.local343 = trunc i64 %address.local342 to i32, !dbg !8
  %bnk.n34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !8
  %bnk.n35 = zext i32 %bnk.n34 to i64, !dbg !8
  %bnk.n36 = inttoptr i64 %bnk.n35 to ptr addrspace(3), !dbg !8
  %address.local344 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local345 = and i64 %address.local344, 16256, !dbg !8
  %address.local346 = lshr i64 %address.local, 4, !dbg !8
  %address.local347 = and i64 %address.local346, 8, !dbg !8
  %address.local348 = or disjoint i64 %address.local345, %address.local347, !dbg !8
  %address.local349 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n33, i64 %address.local348, !dbg !8
  %bnk.n37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %bnk.n38 = ptrtoint ptr addrspace(1) %address.local349 to i64, !dbg !8
  %bnk.n39 = sub i64 %bnk.n38, %bnk.n37, !dbg !8
  %dma.buffer.offset1612 = trunc i64 %bnk.n39 to i32, !dbg !8
  %bnk.n40 = trunc i64 %bnk.n37 to i32, !dbg !8
  %bnk.n41 = insertelement <4 x i32> poison, i32 %bnk.n40, i64 0, !dbg !8
  %bnk.n42 = lshr i64 %bnk.n37, 32, !dbg !8
  %bnk.n43 = trunc i64 %bnk.n42 to i32, !dbg !8
  %bnk.n44 = insertelement <4 x i32> %bnk.n41, i32 %bnk.n43, i64 1, !dbg !8
  %bnk.n45 = insertelement <4 x i32> %bnk.n44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1613 = insertelement <4 x i32> %bnk.n45, i32 131072, i64 3, !dbg !8
  %bnk.n46 = ptrtoint ptr addrspace(3) %bnk.n36 to i32, !dbg !8
  %dma.buffer.lds1614 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n46), !dbg !8
  %bnk.n47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1612, <4 x i32> %dma.buffer.rsrc1613) #10, !dbg !8
  %address.local350 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local351 = shl nuw nsw i64 %address.local350, 3, !dbg !8
  %address.local352 = and i64 %address.local351, 3584, !dbg !8
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !8
  %address.local354 = ptrtoint ptr addrspace(3) %address.local353 to i64, !dbg !8
  %address.local355 = trunc i64 %address.local354 to i32, !dbg !8
  %bnk.n48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !8
  %bnk.n49 = zext i32 %bnk.n48 to i64, !dbg !8
  %bnk.n50 = inttoptr i64 %bnk.n49 to ptr addrspace(3), !dbg !8
  %address.local356 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local357 = and i64 %address.local356, 16256, !dbg !8
  %address.local358 = lshr i64 %address.local350, 4, !dbg !8
  %address.local359 = and i64 %address.local358, 24, !dbg !8
  %address.local360 = or disjoint i64 %address.local357, %address.local359, !dbg !8
  %address.local361 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n33, i64 %address.local360, !dbg !8
  %bnk.n51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %bnk.n52 = ptrtoint ptr addrspace(1) %address.local361 to i64, !dbg !8
  %bnk.n53 = sub i64 %bnk.n52, %bnk.n51, !dbg !8
  %dma.buffer.offset1609 = trunc i64 %bnk.n53 to i32, !dbg !8
  %bnk.n54 = trunc i64 %bnk.n51 to i32, !dbg !8
  %bnk.n55 = insertelement <4 x i32> poison, i32 %bnk.n54, i64 0, !dbg !8
  %bnk.n56 = lshr i64 %bnk.n51, 32, !dbg !8
  %bnk.n57 = trunc i64 %bnk.n56 to i32, !dbg !8
  %bnk.n58 = insertelement <4 x i32> %bnk.n55, i32 %bnk.n57, i64 1, !dbg !8
  %bnk.n59 = insertelement <4 x i32> %bnk.n58, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1610 = insertelement <4 x i32> %bnk.n59, i32 131072, i64 3, !dbg !8
  %bnk.n60 = ptrtoint ptr addrspace(3) %bnk.n50 to i32, !dbg !8
  %dma.buffer.lds1611 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n60), !dbg !8
  %bnk.n61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1609, <4 x i32> %dma.buffer.rsrc1610) #10, !dbg !8
  %bnk.n62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %bnk.n63 = sext i32 %bnk.n62 to i64, !dbg !5
  %bnk.n64 = shl i64 %bnk.n15, 7, !dbg !10
  %bnk.n65 = lshr i64 %bnk.n32, 6, !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %bnk.n66 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %bnk.n30, !dbg !8
  %bnk.n67 = getelementptr i8, ptr addrspace(1) %bnk.n66, i64 64, !dbg !8
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !8
  %address.local363 = ptrtoint ptr addrspace(3) %address.local362 to i64, !dbg !8
  %address.local364 = trunc i64 %address.local363 to i32, !dbg !8
  %bnk.n68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !8
  %bnk.n69 = zext i32 %bnk.n68 to i64, !dbg !8
  %bnk.n70 = inttoptr i64 %bnk.n69 to ptr addrspace(3), !dbg !8
  %address.local365 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n67, i64 %address.local348, !dbg !8
  %bnk.n71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %bnk.n72 = ptrtoint ptr addrspace(1) %address.local365 to i64, !dbg !8
  %bnk.n73 = sub i64 %bnk.n72, %bnk.n71, !dbg !8
  %dma.buffer.offset1606 = trunc i64 %bnk.n73 to i32, !dbg !8
  %bnk.n74 = trunc i64 %bnk.n71 to i32, !dbg !8
  %bnk.n75 = insertelement <4 x i32> poison, i32 %bnk.n74, i64 0, !dbg !8
  %bnk.n76 = lshr i64 %bnk.n71, 32, !dbg !8
  %bnk.n77 = trunc i64 %bnk.n76 to i32, !dbg !8
  %bnk.n78 = insertelement <4 x i32> %bnk.n75, i32 %bnk.n77, i64 1, !dbg !8
  %bnk.n79 = insertelement <4 x i32> %bnk.n78, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1607 = insertelement <4 x i32> %bnk.n79, i32 131072, i64 3, !dbg !8
  %bnk.n80 = ptrtoint ptr addrspace(3) %bnk.n70 to i32, !dbg !8
  %dma.buffer.lds1608 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n80), !dbg !8
  %bnk.n81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1606, <4 x i32> %dma.buffer.rsrc1607) #10, !dbg !8
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !8
  %address.local367 = ptrtoint ptr addrspace(3) %address.local366 to i64, !dbg !8
  %address.local368 = trunc i64 %address.local367 to i32, !dbg !8
  %bnk.n82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !8
  %bnk.n83 = zext i32 %bnk.n82 to i64, !dbg !8
  %bnk.n84 = inttoptr i64 %bnk.n83 to ptr addrspace(3), !dbg !8
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n67, i64 %address.local360, !dbg !8
  %bnk.n85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %bnk.n86 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !8
  %bnk.n87 = sub i64 %bnk.n86, %bnk.n85, !dbg !8
  %dma.buffer.offset1603 = trunc i64 %bnk.n87 to i32, !dbg !8
  %bnk.n88 = trunc i64 %bnk.n85 to i32, !dbg !8
  %bnk.n89 = insertelement <4 x i32> poison, i32 %bnk.n88, i64 0, !dbg !8
  %bnk.n90 = lshr i64 %bnk.n85, 32, !dbg !8
  %bnk.n91 = trunc i64 %bnk.n90 to i32, !dbg !8
  %bnk.n92 = insertelement <4 x i32> %bnk.n89, i32 %bnk.n91, i64 1, !dbg !8
  %bnk.n93 = insertelement <4 x i32> %bnk.n92, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1604 = insertelement <4 x i32> %bnk.n93, i32 131072, i64 3, !dbg !8
  %bnk.n94 = ptrtoint ptr addrspace(3) %bnk.n84 to i32, !dbg !8
  %dma.buffer.lds1605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n94), !dbg !8
  %bnk.n95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1603, <4 x i32> %dma.buffer.rsrc1604) #10, !dbg !8
  %address.local370 = lshr i32 %address.thread, 2, !dbg !11
  %address.local371 = and i32 %address.local370, 4, !dbg !11
  %address.local372 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local373 = and i32 %address.local372, 384, !dbg !11
  %address.local374 = or disjoint i32 %address.local371, %address.local373, !dbg !11
  %address.local375 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local376 = and i32 %address.local375, 120, !dbg !11
  %address.local377 = or disjoint i32 %address.local374, %address.local376, !dbg !11
  %address.local378 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local379 = and i32 %address.local378, 1024, !dbg !11
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !11
  %address.local381 = zext nneg i32 %address.local380 to i64, !dbg !11
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !11
  %bnk.n96 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !11
  %bnk.n97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %bnk.n98 = fmul <4 x half> %bnk.n96, splat (half 0xH3015), !dbg !11
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !11
  %bnk.n99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !11
  %bnk.n100 = fmul <4 x half> %bnk.n97, splat (half 0xH3015), !dbg !11
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !11
  %bnk.n101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !11
  %bnk.n102 = fmul <4 x half> %bnk.n99, splat (half 0xH3015), !dbg !11
  %bnk.n103 = fmul <4 x half> %bnk.n101, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %bnk.n104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %bnk.n30, !dbg !8
  %bnk.n105 = getelementptr i8, ptr addrspace(1) %bnk.n104, i64 128, !dbg !8
  %bnk.n106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !8
  %bnk.n107 = zext i32 %bnk.n106 to i64, !dbg !8
  %bnk.n108 = inttoptr i64 %bnk.n107 to ptr addrspace(3), !dbg !8
  %address.local425 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n105, i64 %address.local348, !dbg !8
  %bnk.n109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %bnk.n110 = ptrtoint ptr addrspace(1) %address.local425 to i64, !dbg !8
  %bnk.n111 = sub i64 %bnk.n110, %bnk.n109, !dbg !8
  %dma.buffer.offset1600 = trunc i64 %bnk.n111 to i32, !dbg !8
  %bnk.n112 = trunc i64 %bnk.n109 to i32, !dbg !8
  %bnk.n113 = insertelement <4 x i32> poison, i32 %bnk.n112, i64 0, !dbg !8
  %bnk.n114 = lshr i64 %bnk.n109, 32, !dbg !8
  %bnk.n115 = trunc i64 %bnk.n114 to i32, !dbg !8
  %bnk.n116 = insertelement <4 x i32> %bnk.n113, i32 %bnk.n115, i64 1, !dbg !8
  %bnk.n117 = insertelement <4 x i32> %bnk.n116, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1601 = insertelement <4 x i32> %bnk.n117, i32 131072, i64 3, !dbg !8
  %bnk.n118 = ptrtoint ptr addrspace(3) %bnk.n108 to i32, !dbg !8
  %dma.buffer.lds1602 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n118), !dbg !8
  %bnk.n119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1602, i32 %dma.buffer.offset1600, <4 x i32> %dma.buffer.rsrc1601) #10, !dbg !8
  %bnk.n120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !8
  %bnk.n121 = zext i32 %bnk.n120 to i64, !dbg !8
  %bnk.n122 = inttoptr i64 %bnk.n121 to ptr addrspace(3), !dbg !8
  %address.local426 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n105, i64 %address.local360, !dbg !8
  %bnk.n123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %bnk.n124 = ptrtoint ptr addrspace(1) %address.local426 to i64, !dbg !8
  %bnk.n125 = sub i64 %bnk.n124, %bnk.n123, !dbg !8
  %dma.buffer.offset1597 = trunc i64 %bnk.n125 to i32, !dbg !8
  %bnk.n126 = trunc i64 %bnk.n123 to i32, !dbg !8
  %bnk.n127 = insertelement <4 x i32> poison, i32 %bnk.n126, i64 0, !dbg !8
  %bnk.n128 = lshr i64 %bnk.n123, 32, !dbg !8
  %bnk.n129 = trunc i64 %bnk.n128 to i32, !dbg !8
  %bnk.n130 = insertelement <4 x i32> %bnk.n127, i32 %bnk.n129, i64 1, !dbg !8
  %bnk.n131 = insertelement <4 x i32> %bnk.n130, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1598 = insertelement <4 x i32> %bnk.n131, i32 131072, i64 3, !dbg !8
  %bnk.n132 = ptrtoint ptr addrspace(3) %bnk.n122 to i32, !dbg !8
  %dma.buffer.lds1599 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n132), !dbg !8
  %bnk.n133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1599, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1598) #10, !dbg !8
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !11
  %bnk.n134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !11
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !11
  %bnk.n135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !11
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !11
  %bnk.n136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !11
  %bnk.n137 = fmul <4 x half> %bnk.n134, splat (half 0xH3015), !dbg !11
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !11
  %bnk.n138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !11
  %bnk.n139 = fmul <4 x half> %bnk.n135, splat (half 0xH3015), !dbg !11
  %bnk.n140 = fmul <4 x half> %bnk.n136, splat (half 0xH3015), !dbg !11
  %bnk.n141 = fmul <4 x half> %bnk.n138, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %bnk.n142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %bnk.n30, !dbg !8
  %bnk.n143 = getelementptr i8, ptr addrspace(1) %bnk.n142, i64 192, !dbg !8
  %bnk.n144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !8
  %bnk.n145 = zext i32 %bnk.n144 to i64, !dbg !8
  %bnk.n146 = inttoptr i64 %bnk.n145 to ptr addrspace(3), !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n143, i64 %address.local348, !dbg !8
  %bnk.n147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %bnk.n148 = ptrtoint ptr addrspace(1) %address.local431 to i64, !dbg !8
  %bnk.n149 = sub i64 %bnk.n148, %bnk.n147, !dbg !8
  %dma.buffer.offset1594 = trunc i64 %bnk.n149 to i32, !dbg !8
  %bnk.n150 = trunc i64 %bnk.n147 to i32, !dbg !8
  %bnk.n151 = insertelement <4 x i32> poison, i32 %bnk.n150, i64 0, !dbg !8
  %bnk.n152 = lshr i64 %bnk.n147, 32, !dbg !8
  %bnk.n153 = trunc i64 %bnk.n152 to i32, !dbg !8
  %bnk.n154 = insertelement <4 x i32> %bnk.n151, i32 %bnk.n153, i64 1, !dbg !8
  %bnk.n155 = insertelement <4 x i32> %bnk.n154, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1595 = insertelement <4 x i32> %bnk.n155, i32 131072, i64 3, !dbg !8
  %bnk.n156 = ptrtoint ptr addrspace(3) %bnk.n146 to i32, !dbg !8
  %dma.buffer.lds1596 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n156), !dbg !8
  %bnk.n157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1596, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1595) #10, !dbg !8
  %bnk.n158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !8
  %bnk.n159 = zext i32 %bnk.n158 to i64, !dbg !8
  %bnk.n160 = inttoptr i64 %bnk.n159 to ptr addrspace(3), !dbg !8
  %address.local432 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n143, i64 %address.local360, !dbg !8
  %bnk.n161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %bnk.n162 = ptrtoint ptr addrspace(1) %address.local432 to i64, !dbg !8
  %bnk.n163 = sub i64 %bnk.n162, %bnk.n161, !dbg !8
  %dma.buffer.offset1591 = trunc i64 %bnk.n163 to i32, !dbg !8
  %bnk.n164 = trunc i64 %bnk.n161 to i32, !dbg !8
  %bnk.n165 = insertelement <4 x i32> poison, i32 %bnk.n164, i64 0, !dbg !8
  %bnk.n166 = lshr i64 %bnk.n161, 32, !dbg !8
  %bnk.n167 = trunc i64 %bnk.n166 to i32, !dbg !8
  %bnk.n168 = insertelement <4 x i32> %bnk.n165, i32 %bnk.n167, i64 1, !dbg !8
  %bnk.n169 = insertelement <4 x i32> %bnk.n168, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1592 = insertelement <4 x i32> %bnk.n169, i32 131072, i64 3, !dbg !8
  %bnk.n170 = ptrtoint ptr addrspace(3) %bnk.n160 to i32, !dbg !8
  %dma.buffer.lds1593 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n170), !dbg !8
  %bnk.n171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1592) #10, !dbg !8
  %bnk.n172 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !11
  %lds.const.gep1419 = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !11
  %bnk.n173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1419, align 8, !dbg !11
  %lds.const.gep1420 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !11
  %bnk.n174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1420, align 8, !dbg !11
  %bnk.n175 = fmul <4 x half> %bnk.n172, splat (half 0xH3015), !dbg !11
  %lds.const.gep1421 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !11
  %bnk.n176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1421, align 8, !dbg !11
  %bnk.n177 = fmul <4 x half> %bnk.n173, splat (half 0xH3015), !dbg !11
  %bnk.n178 = fmul <4 x half> %bnk.n174, splat (half 0xH3015), !dbg !11
  %bnk.n179 = fmul <4 x half> %bnk.n176, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1422 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !11
  %bnk.n180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1422, align 8, !dbg !11
  %lds.const.gep1423 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !11
  %bnk.n181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1423, align 8, !dbg !11
  %lds.const.gep1424 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !11
  %bnk.n182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1424, align 8, !dbg !11
  %bnk.n183 = fmul <4 x half> %bnk.n180, splat (half 0xH3015), !dbg !11
  %lds.const.gep1425 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !11
  %bnk.n184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1425, align 8, !dbg !11
  %bnk.n185 = fmul <4 x half> %bnk.n181, splat (half 0xH3015), !dbg !11
  %bnk.n186 = fmul <4 x half> %bnk.n182, splat (half 0xH3015), !dbg !11
  %bnk.n187 = fmul <4 x half> %bnk.n184, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %bnk.n188 = or disjoint i64 %bnk.n64, 127, !dbg !14
  %bnk.n189 = add i64 %bnk.n64, 128, !dbg !14
  %bnk.n190 = icmp uge i64 %bnk.n189, %bnk.n188, !dbg !14
  %bnk.n191 = sub nsw i64 32, %bnk.n14, !dbg !16
  %bnk.n192 = shl i64 %bnk.n16, 7, !dbg !17
  %bnk.n193 = and i64 %bnk.n192, 3968, !dbg !17
  %bnk.n194 = shl nsw i64 %bnk.n63, 12, !dbg !17
  %bnk.n195 = or disjoint i64 %bnk.n193, %bnk.n194, !dbg !17
  %bnk.n196 = lshr i64 %bnk.n32, 2, !dbg !18
  %bnk.n197 = shl nuw nsw i64 %bnk.n32, 2, !dbg !18
  %bnk.n198 = mul nsw i64 %bnk.n196, -16, !dbg !18
  %bnk.n199 = add nsw i64 %bnk.n197, %bnk.n198, !dbg !18
  %bnk.n200 = and i32 %bnk.n31, 1, !dbg !18
  %index.compact279 = zext nneg i32 %bnk.n200 to i64, !dbg !18
  %bnk.n201 = or disjoint i64 %bnk.n199, %index.compact279, !dbg !18
  %bnk.n202 = add nuw nsw i64 %bnk.n201, 1, !dbg !18
  %bnk.n203 = sdiv i64 %bnk.n202, 2, !dbg !18
; Prime K0 for BN=0 after the Q handoff.
  %k0.prime.address.thread433 = call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #9, !dbg !17, !range !34, !frisk.register_lifetimes !1
  %k0.prime.address.local434 = zext nneg i32 %k0.prime.address.thread433 to i64, !dbg !8
  %k0.prime.address.local435 = shl nuw nsw i64 %k0.prime.address.local434, 3, !dbg !8
  %k0.prime.address.local436 = and i64 %k0.prime.address.local435, 1536, !dbg !8
  %k0.prime.address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local436, !dbg !8
  %k0.prime.address.local438 = ptrtoint ptr addrspace(3) %k0.prime.address.local437 to i64, !dbg !8
  %k0.prime.address.local439 = trunc i64 %k0.prime.address.local438 to i32, !dbg !8
  %k0.prime.bnk.n226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local439), !dbg !17
  %k0.prime.bnk.n227 = zext i32 %k0.prime.bnk.n226 to i64, !dbg !17
  %k0.prime.bnk.n228 = inttoptr i64 %k0.prime.bnk.n227 to ptr addrspace(3), !dbg !17
  %k0.prime.bnk.n238 = ptrtoint ptr addrspace(3) %k0.prime.bnk.n228 to i32, !dbg !17
  %k0.prime.dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.bnk.n238), !dbg !17
  %k0.prime.bnk.n214 = lshr i64 0, 12, !dbg !17
  %k0.prime.bnk.n215 = add nuw nsw i64 %k0.prime.bnk.n214, %bnk.n16, !dbg !17
  %k0.prime.bnk.n218 = shl i64 %k0.prime.bnk.n215, 19, !dbg !17
  %k0.prime.bnk.n219 = and i64 %k0.prime.bnk.n218, 16252928, !dbg !17
  %k0.prime.bnk.n223 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %k0.prime.bnk.n219, !dbg !17
  %k0.prime.bnk.n216 = lshr i64 0, 5, !dbg !17
  %k0.prime.bnk.n217 = and i64 %k0.prime.bnk.n216, 127, !dbg !17
  %k0.prime.bnk.n224 = getelementptr [2 x i8], ptr addrspace(1) %k0.prime.bnk.n223, i64 %k0.prime.bnk.n217, !dbg !17
  %k0.prime.bnk.n221 = shl i64 0, 14, !dbg !17
  %k0.prime.bnk.n220 = shl i64 %k0.prime.bnk.n216, 19, !dbg !17
  %k0.prime.bnk.n222 = sub i64 %k0.prime.bnk.n221, %k0.prime.bnk.n220, !dbg !17
  %k0.prime.bnk.n225 = getelementptr [2 x i8], ptr addrspace(1) %k0.prime.bnk.n224, i64 %k0.prime.bnk.n222, !dbg !17
  %k0.prime.address.local440 = lshr i64 %k0.prime.address.local434, 4, !dbg !17
  %k0.prime.address.local441 = and i64 %k0.prime.address.local440, 8, !dbg !17
  %k0.prime.address.local442 = and i64 %k0.prime.address.local434, 123, !dbg !17
  %k0.prime.address.local443 = lshr i64 %k0.prime.address.local434, 2, !dbg !17
  %k0.prime.address.local444 = and i64 %k0.prime.address.local443, 4, !dbg !17
  %k0.prime.address.local445 = add nuw nsw i64 %k0.prime.address.local444, %k0.prime.address.local434, !dbg !17
  %k0.prime.address.local446 = and i64 %k0.prime.address.local445, 4, !dbg !17
  %k0.prime.address.local447 = or disjoint i64 %k0.prime.address.local442, %k0.prime.address.local446, !dbg !17
  %k0.prime.address.local448 = shl nuw nsw i64 %k0.prime.address.local447, 7, !dbg !17
  %k0.prime.address.local449 = or disjoint i64 %k0.prime.address.local441, %k0.prime.address.local448, !dbg !17
  %k0.prime.address.local450 = getelementptr [2 x i8], ptr addrspace(1) %k0.prime.bnk.n225, i64 %k0.prime.address.local449, !dbg !17
  %k0.prime.bnk.n230 = ptrtoint ptr addrspace(1) %k0.prime.address.local450 to i64, !dbg !17
  %k0.prime.bnk.n229 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %k0.prime.bnk.n231 = sub i64 %k0.prime.bnk.n230, %k0.prime.bnk.n229, !dbg !17
  %k0.prime.dma.buffer.offset1588 = trunc i64 %k0.prime.bnk.n231 to i32, !dbg !17
  %k0.prime.bnk.n232 = trunc i64 %k0.prime.bnk.n229 to i32, !dbg !17
  %k0.prime.bnk.n233 = insertelement <4 x i32> poison, i32 %k0.prime.bnk.n232, i64 0, !dbg !17
  %k0.prime.bnk.n234 = lshr i64 %k0.prime.bnk.n229, 32, !dbg !17
  %k0.prime.bnk.n235 = trunc i64 %k0.prime.bnk.n234 to i32, !dbg !17
  %k0.prime.bnk.n236 = insertelement <4 x i32> %k0.prime.bnk.n233, i32 %k0.prime.bnk.n235, i64 1, !dbg !17
  %k0.prime.bnk.n237 = insertelement <4 x i32> %k0.prime.bnk.n236, i32 33554432, i64 2, !dbg !17
  %k0.prime.dma.buffer.rsrc1589 = insertelement <4 x i32> %k0.prime.bnk.n237, i32 131072, i64 3, !dbg !17
  %k0.prime.bnk.n239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1590, i32 %k0.prime.dma.buffer.offset1588, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #10, !dbg !17
  %k0.prime.address.local451 = or disjoint i64 %k0.prime.address.local434, 256, !dbg !8
  %k0.prime.address.local452 = shl nuw nsw i64 %k0.prime.address.local451, 3, !dbg !8
  %k0.prime.address.local453 = and i64 %k0.prime.address.local452, 3584, !dbg !8
  %k0.prime.address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local453, !dbg !8
  %k0.prime.address.local455 = ptrtoint ptr addrspace(3) %k0.prime.address.local454 to i64, !dbg !8
  %k0.prime.address.local456 = trunc i64 %k0.prime.address.local455 to i32, !dbg !8
  %k0.prime.bnk.n240 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local456), !dbg !17
  %k0.prime.bnk.n241 = zext i32 %k0.prime.bnk.n240 to i64, !dbg !17
  %k0.prime.bnk.n242 = inttoptr i64 %k0.prime.bnk.n241 to ptr addrspace(3), !dbg !17
  %k0.prime.bnk.n252 = ptrtoint ptr addrspace(3) %k0.prime.bnk.n242 to i32, !dbg !17
  %k0.prime.dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.bnk.n252), !dbg !17
  %k0.prime.address.local457 = lshr i64 %k0.prime.address.local451, 4, !dbg !17
  %k0.prime.address.local458 = and i64 %k0.prime.address.local457, 24, !dbg !17
  %k0.prime.address.local459 = and i64 %k0.prime.address.local434, 123, !dbg !17
  %k0.prime.address.local460 = lshr i64 %k0.prime.address.local434, 2, !dbg !17
  %k0.prime.address.local461 = and i64 %k0.prime.address.local460, 4, !dbg !17
  %k0.prime.address.local462 = add nuw nsw i64 %k0.prime.address.local461, %k0.prime.address.local434, !dbg !17
  %k0.prime.address.local463 = and i64 %k0.prime.address.local462, 4, !dbg !17
  %k0.prime.address.local464 = or disjoint i64 %k0.prime.address.local459, %k0.prime.address.local463, !dbg !17
  %k0.prime.address.local465 = shl nuw nsw i64 %k0.prime.address.local464, 7, !dbg !17
  %k0.prime.address.local466 = or disjoint i64 %k0.prime.address.local458, %k0.prime.address.local465, !dbg !17
  %k0.prime.address.local467 = getelementptr [2 x i8], ptr addrspace(1) %k0.prime.bnk.n225, i64 %k0.prime.address.local466, !dbg !17
  %k0.prime.bnk.n244 = ptrtoint ptr addrspace(1) %k0.prime.address.local467 to i64, !dbg !17
  %k0.prime.bnk.n243 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %k0.prime.bnk.n245 = sub i64 %k0.prime.bnk.n244, %k0.prime.bnk.n243, !dbg !17
  %k0.prime.dma.buffer.offset1585 = trunc i64 %k0.prime.bnk.n245 to i32, !dbg !17
  %k0.prime.bnk.n246 = trunc i64 %k0.prime.bnk.n243 to i32, !dbg !17
  %k0.prime.bnk.n247 = insertelement <4 x i32> poison, i32 %k0.prime.bnk.n246, i64 0, !dbg !17
  %k0.prime.bnk.n248 = lshr i64 %k0.prime.bnk.n243, 32, !dbg !17
  %k0.prime.bnk.n249 = trunc i64 %k0.prime.bnk.n248 to i32, !dbg !17
  %k0.prime.bnk.n250 = insertelement <4 x i32> %k0.prime.bnk.n247, i32 %k0.prime.bnk.n249, i64 1, !dbg !17
  %k0.prime.bnk.n251 = insertelement <4 x i32> %k0.prime.bnk.n250, i32 33554432, i64 2, !dbg !17
  %k0.prime.dma.buffer.rsrc1586 = insertelement <4 x i32> %k0.prime.bnk.n251, i32 131072, i64 3, !dbg !17
  %k0.prime.bnk.n253 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1587, i32 %k0.prime.dma.buffer.offset1585, <4 x i32> %k0.prime.dma.buffer.rsrc1586) #10, !dbg !17
  br label %bnk.n204, !dbg !13

bnk.n204:                                              ; preds = %bnk.n847, %4
  %bnk.n205 = phi i64 [ %bnk.n1580, %k0.join ], [ 0, %4 ], !dbg !13
  %bnk.n206 = phi [2 x <8 x float>] [ %bnk.n1567, %k0.join ], [ zeroinitializer, %4 ], !dbg !13
  %bnk.n207 = phi [2 x <1 x float>] [ %bnk.n1189, %k0.join ], [ zeroinitializer, %4 ], !dbg !13
  %bnk.n208 = phi [2 x <8 x float>] [ %bnk.n1571, %k0.join ], [ zeroinitializer, %4 ], !dbg !13
  %bnk.n209 = phi [2 x <8 x float>] [ %bnk.n1575, %k0.join ], [ zeroinitializer, %4 ], !dbg !13
  %bnk.n210 = phi [2 x <8 x float>] [ %bnk.n1579, %k0.join ], [ zeroinitializer, %4 ], !dbg !13
  %bnk.n211 = icmp slt i64 %bnk.n205, %bnk.n191, !dbg !13
  br i1 %bnk.n211, label %bnk.n212, label %bnk.n1581, !dbg !13

bnk.n212:                                              ; preds = %bnk.n204
  %bnk.n213 = shl nsw i64 %bnk.n205, 7, !dbg !13
  %bnk.n214 = lshr i64 %bnk.n205, 12, !dbg !17
  %bnk.n215 = add nuw nsw i64 %bnk.n214, %bnk.n16, !dbg !17
  %bnk.n216 = lshr i64 %bnk.n205, 5, !dbg !17
  %bnk.n217 = and i64 %bnk.n216, 127, !dbg !17
  %bnk.n218 = shl i64 %bnk.n215, 19, !dbg !17
  %bnk.n219 = and i64 %bnk.n218, 16252928, !dbg !17
  %bnk.n220 = shl i64 %bnk.n216, 19, !dbg !17
  %bnk.n221 = shl i64 %bnk.n205, 14, !dbg !17
  %bnk.n222 = sub i64 %bnk.n221, %bnk.n220, !dbg !17
  %bnk.n223 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %bnk.n219, !dbg !17
  %bnk.n224 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n223, i64 %bnk.n217, !dbg !17
  %bnk.n225 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n224, i64 %bnk.n222, !dbg !17
  %address.thread433 = call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #9, !dbg !17, !range !34, !frisk.register_lifetimes !1
  %address.local434 = zext nneg i32 %address.thread433 to i64, !dbg !8
  %address.local435 = shl nuw nsw i64 %address.local434, 3, !dbg !8
  %address.local436 = and i64 %address.local435, 1536, !dbg !8
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !8
  %address.local438 = ptrtoint ptr addrspace(3) %address.local437 to i64, !dbg !8
  %address.local439 = trunc i64 %address.local438 to i32, !dbg !8
  %bnk.n226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !17
  %bnk.n227 = zext i32 %bnk.n226 to i64, !dbg !17
  %bnk.n228 = inttoptr i64 %bnk.n227 to ptr addrspace(3), !dbg !17
  %address.local440 = lshr i64 %address.local434, 4, !dbg !17
  %address.local441 = and i64 %address.local440, 8, !dbg !17
  %address.local442 = and i64 %address.local434, 123, !dbg !17
  %address.local443 = lshr i64 %address.local434, 2, !dbg !17
  %address.local444 = and i64 %address.local443, 4, !dbg !17
  %address.local445 = add nuw nsw i64 %address.local444, %address.local434, !dbg !17
  %address.local446 = and i64 %address.local445, 4, !dbg !17
  %address.local447 = or disjoint i64 %address.local442, %address.local446, !dbg !17
  %address.local448 = shl nuw nsw i64 %address.local447, 7, !dbg !17
  %address.local449 = or disjoint i64 %address.local441, %address.local448, !dbg !17
  %address.local450 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n225, i64 %address.local449, !dbg !17
  %bnk.n229 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %bnk.n230 = ptrtoint ptr addrspace(1) %address.local450 to i64, !dbg !17
  %bnk.n231 = sub i64 %bnk.n230, %bnk.n229, !dbg !17
  %dma.buffer.offset1588 = trunc i64 %bnk.n231 to i32, !dbg !17
  %bnk.n232 = trunc i64 %bnk.n229 to i32, !dbg !17
  %bnk.n233 = insertelement <4 x i32> poison, i32 %bnk.n232, i64 0, !dbg !17
  %bnk.n234 = lshr i64 %bnk.n229, 32, !dbg !17
  %bnk.n235 = trunc i64 %bnk.n234 to i32, !dbg !17
  %bnk.n236 = insertelement <4 x i32> %bnk.n233, i32 %bnk.n235, i64 1, !dbg !17
  %bnk.n237 = insertelement <4 x i32> %bnk.n236, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1589 = insertelement <4 x i32> %bnk.n237, i32 131072, i64 3, !dbg !17
  %bnk.n238 = ptrtoint ptr addrspace(3) %bnk.n228 to i32, !dbg !17
  %dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n238), !dbg !17
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !8
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !8
  %address.local453 = and i64 %address.local452, 3584, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !8
  %address.local455 = ptrtoint ptr addrspace(3) %address.local454 to i64, !dbg !8
  %address.local456 = trunc i64 %address.local455 to i32, !dbg !8
  %bnk.n240 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !17
  %bnk.n241 = zext i32 %bnk.n240 to i64, !dbg !17
  %bnk.n242 = inttoptr i64 %bnk.n241 to ptr addrspace(3), !dbg !17
  %address.local457 = lshr i64 %address.local451, 4, !dbg !17
  %address.local458 = and i64 %address.local457, 24, !dbg !17
  %address.local459 = and i64 %address.local434, 123, !dbg !17
  %address.local460 = lshr i64 %address.local434, 2, !dbg !17
  %address.local461 = and i64 %address.local460, 4, !dbg !17
  %address.local462 = add nuw nsw i64 %address.local461, %address.local434, !dbg !17
  %address.local463 = and i64 %address.local462, 4, !dbg !17
  %address.local464 = or disjoint i64 %address.local459, %address.local463, !dbg !17
  %address.local465 = shl nuw nsw i64 %address.local464, 7, !dbg !17
  %address.local466 = or disjoint i64 %address.local458, %address.local465, !dbg !17
  %address.local467 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n225, i64 %address.local466, !dbg !17
  %bnk.n243 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %bnk.n244 = ptrtoint ptr addrspace(1) %address.local467 to i64, !dbg !17
  %bnk.n245 = sub i64 %bnk.n244, %bnk.n243, !dbg !17
  %dma.buffer.offset1585 = trunc i64 %bnk.n245 to i32, !dbg !17
  %bnk.n246 = trunc i64 %bnk.n243 to i32, !dbg !17
  %bnk.n247 = insertelement <4 x i32> poison, i32 %bnk.n246, i64 0, !dbg !17
  %bnk.n248 = lshr i64 %bnk.n243, 32, !dbg !17
  %bnk.n249 = trunc i64 %bnk.n248 to i32, !dbg !17
  %bnk.n250 = insertelement <4 x i32> %bnk.n247, i32 %bnk.n249, i64 1, !dbg !17
  %bnk.n251 = insertelement <4 x i32> %bnk.n250, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1586 = insertelement <4 x i32> %bnk.n251, i32 131072, i64 3, !dbg !17
  %bnk.n252 = ptrtoint ptr addrspace(3) %bnk.n242 to i32, !dbg !17
  %dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n252), !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %bnk.n254 = add nuw i64 %bnk.n205, 1024, !dbg !17
  %bnk.n255 = icmp slt i64 %bnk.n254, 0, !dbg !17
  %bnk.n256 = sub nuw i64 -1025, %bnk.n205, !dbg !17
  %bnk.n257 = select i1 %bnk.n255, i64 %bnk.n256, i64 %bnk.n254, !dbg !17
  %bnk.n258 = sdiv i64 %bnk.n257, 4096, !dbg !17
  %.lobit37 = ashr i64 %bnk.n254, 63, !dbg !17
  %bnk.n259 = xor i64 %bnk.n258, %.lobit37, !dbg !17
  %bnk.n260 = add nsw i64 %bnk.n259, %bnk.n16, !dbg !17
  %bnk.n261 = add nsw i64 %bnk.n216, %bnk.n195, !dbg !17
  %bnk.n262 = add nsw i64 %bnk.n261, 32, !dbg !17
  %bnk.n263 = icmp slt i64 %bnk.n261, -32, !dbg !17
  %bnk.n264 = sub nsw i64 -33, %bnk.n261, !dbg !17
  %bnk.n265 = select i1 %bnk.n263, i64 %bnk.n264, i64 %bnk.n262, !dbg !17
  %bnk.n266 = sdiv i64 %bnk.n265, 128, !dbg !17
  %bnk.n267 = sext i1 %bnk.n263 to i64, !dbg !17
  %bnk.n268 = xor i64 %bnk.n266, %bnk.n267, !dbg !17
  %bnk.n269 = shl nsw i64 %bnk.n268, 7, !dbg !17
  %bnk.n270 = sub i64 %bnk.n195, %bnk.n269, !dbg !17
  %bnk.n271 = shl i64 %bnk.n260, 19, !dbg !17
  %bnk.n272 = and i64 %bnk.n271, 16252928, !dbg !17
  %bnk.n273 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %bnk.n272, !dbg !17
  %bnk.n274 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n273, i64 %bnk.n216, !dbg !17
  %bnk.n275 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n274, i64 %bnk.n270, !dbg !17
  %bnk.n276 = getelementptr i8, ptr addrspace(1) %bnk.n275, i64 64, !dbg !17
  %bnk.n277 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n276, i64 %bnk.n222, !dbg !17
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !8
  %address.local469 = ptrtoint ptr addrspace(3) %address.local468 to i64, !dbg !8
  %address.local470 = trunc i64 %address.local469 to i32, !dbg !8
  %bnk.n278 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !17
  %bnk.n279 = zext i32 %bnk.n278 to i64, !dbg !17
  %bnk.n280 = inttoptr i64 %bnk.n279 to ptr addrspace(3), !dbg !17
  %address.local471 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n277, i64 %address.local449, !dbg !17
  %bnk.n281 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %bnk.n282 = ptrtoint ptr addrspace(1) %address.local471 to i64, !dbg !17
  %bnk.n283 = sub i64 %bnk.n282, %bnk.n281, !dbg !17
  %dma.buffer.offset1582 = trunc i64 %bnk.n283 to i32, !dbg !17
  %bnk.n284 = trunc i64 %bnk.n281 to i32, !dbg !17
  %bnk.n285 = insertelement <4 x i32> poison, i32 %bnk.n284, i64 0, !dbg !17
  %bnk.n286 = lshr i64 %bnk.n281, 32, !dbg !17
  %bnk.n287 = trunc i64 %bnk.n286 to i32, !dbg !17
  %bnk.n288 = insertelement <4 x i32> %bnk.n285, i32 %bnk.n287, i64 1, !dbg !17
  %bnk.n289 = insertelement <4 x i32> %bnk.n288, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1583 = insertelement <4 x i32> %bnk.n289, i32 131072, i64 3, !dbg !17
  %bnk.n290 = ptrtoint ptr addrspace(3) %bnk.n280 to i32, !dbg !17
  %dma.buffer.lds1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n290), !dbg !17
  %bnk.n291 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1583) #10, !dbg !17
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !8
  %address.local473 = ptrtoint ptr addrspace(3) %address.local472 to i64, !dbg !8
  %address.local474 = trunc i64 %address.local473 to i32, !dbg !8
  %bnk.n292 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !17
  %bnk.n293 = zext i32 %bnk.n292 to i64, !dbg !17
  %bnk.n294 = inttoptr i64 %bnk.n293 to ptr addrspace(3), !dbg !17
  %address.local475 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n277, i64 %address.local466, !dbg !17
  %bnk.n295 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %bnk.n296 = ptrtoint ptr addrspace(1) %address.local475 to i64, !dbg !17
  %bnk.n297 = sub i64 %bnk.n296, %bnk.n295, !dbg !17
  %dma.buffer.offset1579 = trunc i64 %bnk.n297 to i32, !dbg !17
  %bnk.n298 = trunc i64 %bnk.n295 to i32, !dbg !17
  %bnk.n299 = insertelement <4 x i32> poison, i32 %bnk.n298, i64 0, !dbg !17
  %bnk.n300 = lshr i64 %bnk.n295, 32, !dbg !17
  %bnk.n301 = trunc i64 %bnk.n300 to i32, !dbg !17
  %bnk.n302 = insertelement <4 x i32> %bnk.n299, i32 %bnk.n301, i64 1, !dbg !17
  %bnk.n303 = insertelement <4 x i32> %bnk.n302, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1580 = insertelement <4 x i32> %bnk.n303, i32 131072, i64 3, !dbg !17
  %bnk.n304 = ptrtoint ptr addrspace(3) %bnk.n294 to i32, !dbg !17
  %dma.buffer.lds1581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n304), !dbg !17
  %bnk.n305 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1580) #10, !dbg !17
  %address.local476 = mul nuw nsw i64 %address.local434, 160, !dbg !18
  %address.local477 = lshr i64 %address.local434, 2, !dbg !18
  %address.local478 = mul nsw i64 %address.local477, -640, !dbg !18
  %address.local479 = add nsw i64 %address.local476, %address.local478, !dbg !18
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !18
  %address.local481 = and i64 %address.local480, 24, !dbg !18
  %address.local482 = or disjoint i64 %address.local479, %address.local481, !dbg !18
  %address.local483 = shl i64 %address.local434, 62, !dbg !18
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !18
  %address.local485 = and i64 %address.local484, -64, !dbg !18
  %address.local486 = add nsw i64 %address.local482, %address.local485, !dbg !18
  %address.local487 = lshr i64 %address.local434, 2, !dbg !11
  %address.local488 = and i64 %address.local487, 12, !dbg !11
  %address.local489 = add nsw i64 %address.local486, %address.local488, !dbg !18
  %address.local490 = and i64 %address.local434, 63, !dbg !11
  %address.local491 = lshr i64 %address.local490, 5, !dbg !11
  %address.local492 = mul nuw nsw i64 %address.local491, 1016, !dbg !11
  %address.local493 = add i64 %address.local489, %address.local492, !dbg !18
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local493, !dbg !18
  %bnk.n306 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !18
  %bnk.n307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n306, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !18
  %bnk.n308 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !18
  %address.local505 = shl nuw nsw i32 %address.thread433, 1, !dbg !18
  %address.local506 = and i32 %address.local505, 24, !dbg !18
  %address.local507 = shl nuw nsw i32 %address.thread433, 8, !dbg !18
  %address.local508 = and i32 %address.local507, 768, !dbg !18
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !18
  %address.local510 = shl nuw nsw i32 %address.thread433, 5, !dbg !18
  %address.local511 = and i32 %address.local510, 96, !dbg !18
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !18
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !18
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %bnk.n203) #9, !dbg !18, !frisk.register_lifetimes !1
  %address.local514 = lshr i32 %address.thread433, 1, !dbg !18
  %address.local515 = and i32 %address.local514, 1, !dbg !18
  %address.local516 = zext nneg i32 %address.local515 to i64, !dbg !18
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !18
  %address.local518 = shl i64 %address.local517, 6, !dbg !18
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !18
  %address.local520 = add i64 %address.local519, %address.local488, !dbg !18
  %address.local521 = add i64 %address.local520, %address.local492, !dbg !18
  %address.local522 = add i64 %address.local521, 32, !dbg !18
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !18
  %bnk.n309 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !18
  %bnk.n310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n308, <4 x float> %bnk.n307) #9, !dbg !18
  %bnk.n311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n309, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !18
  %bnk.n312 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !18
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !18
  %bnk.n313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !18
  %bnk.n314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n312, <4 x float> %bnk.n311) #9, !dbg !18
  %bnk.n315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n313, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !18
  %bnk.n316 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !18
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !18
  %bnk.n317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !18
  %bnk.n318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n316, <4 x float> %bnk.n315) #9, !dbg !18
  %bnk.n319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n317, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !18
  %bnk.n320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !18
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !18
  %bnk.n321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !18
  %bnk.n322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n320, <4 x float> %bnk.n319) #9, !dbg !18
  %bnk.n323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n321, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !18
  %bnk.n324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !18
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !18
  %bnk.n325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !18
  %bnk.n326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n324, <4 x float> %bnk.n323) #9, !dbg !18
  %bnk.n327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n325, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !18
  %bnk.n328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !18
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !18
  %bnk.n329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !18
  %bnk.n330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n328, <4 x float> %bnk.n327) #9, !dbg !18
  %bnk.n331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n329, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !18
  %bnk.n332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !18
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !18
  %bnk.n333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !18
  %bnk.n334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n332, <4 x float> %bnk.n331) #9, !dbg !18
  %bnk.n335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n333, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !18
  %bnk.n336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !18
  %bnk.n337 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !18
  %bnk.n338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n336, <4 x float> %bnk.n335) #9, !dbg !18
  %bnk.n339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n337, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !18
  %bnk.n340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !18
  %bnk.n341 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !18
  %bnk.n342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n340, <4 x float> %bnk.n339) #9, !dbg !18
  %bnk.n343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n341, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !18
  %bnk.n344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !18
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !18
  %bnk.n345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !18
  %bnk.n346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n344, <4 x float> %bnk.n343) #9, !dbg !18
  %bnk.n347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n345, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !18
  %bnk.n348 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !18
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !18
  %bnk.n349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !18
  %bnk.n350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n348, <4 x float> %bnk.n347) #9, !dbg !18
  %bnk.n351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n349, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !18
  %bnk.n352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !18
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !18
  %bnk.n353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !18
  %bnk.n354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n352, <4 x float> %bnk.n351) #9, !dbg !18
  %bnk.n355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n353, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !18
  %bnk.n356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !18
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !18
  %bnk.n357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !18
  %bnk.n358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n356, <4 x float> %bnk.n355) #9, !dbg !18
  %bnk.n359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n357, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !18
  %bnk.n360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !18
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !18
  %bnk.n361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !18
  %bnk.n362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n360, <4 x float> %bnk.n359) #9, !dbg !18
  %bnk.n363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n361, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !18
  %bnk.n364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !18
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !18
  %bnk.n365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !18
  %bnk.n366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n364, <4 x float> %bnk.n363) #9, !dbg !18
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !18
  %bnk.n367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %bnk.n368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n365, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %bnk.n369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n367, <4 x float> %bnk.n368) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %bnk.n370 = add nuw i64 %bnk.n205, 2048, !dbg !17
  %bnk.n371 = icmp slt i64 %bnk.n370, 0, !dbg !17
  %bnk.n372 = sub nuw i64 -2049, %bnk.n205, !dbg !17
  %bnk.n373 = select i1 %bnk.n371, i64 %bnk.n372, i64 %bnk.n370, !dbg !17
  %bnk.n374 = sdiv i64 %bnk.n373, 4096, !dbg !17
  %.lobit38 = ashr i64 %bnk.n370, 63, !dbg !17
  %bnk.n375 = xor i64 %bnk.n374, %.lobit38, !dbg !17
  %bnk.n376 = add nsw i64 %bnk.n375, %bnk.n16, !dbg !17
  %bnk.n377 = add nsw i64 %bnk.n261, 64, !dbg !17
  %bnk.n378 = icmp slt i64 %bnk.n261, -64, !dbg !17
  %bnk.n379 = sub nsw i64 -65, %bnk.n261, !dbg !17
  %bnk.n380 = select i1 %bnk.n378, i64 %bnk.n379, i64 %bnk.n377, !dbg !17
  %bnk.n381 = sdiv i64 %bnk.n380, 128, !dbg !17
  %bnk.n382 = sext i1 %bnk.n378 to i64, !dbg !17
  %bnk.n383 = xor i64 %bnk.n381, %bnk.n382, !dbg !17
  %bnk.n384 = shl nsw i64 %bnk.n383, 7, !dbg !17
  %bnk.n385 = sub i64 %bnk.n195, %bnk.n384, !dbg !17
  %bnk.n386 = shl i64 %bnk.n376, 19, !dbg !17
  %bnk.n387 = and i64 %bnk.n386, 16252928, !dbg !17
  %bnk.n388 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %bnk.n387, !dbg !17
  %bnk.n389 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n388, i64 %bnk.n216, !dbg !17
  %bnk.n390 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n389, i64 %bnk.n385, !dbg !17
  %bnk.n391 = getelementptr i8, ptr addrspace(1) %bnk.n390, i64 128, !dbg !17
  %bnk.n392 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n391, i64 %bnk.n222, !dbg !17
  %bnk.n393 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !17
  %bnk.n394 = zext i32 %bnk.n393 to i64, !dbg !17
  %bnk.n395 = inttoptr i64 %bnk.n394 to ptr addrspace(3), !dbg !17
  %address.local832 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n392, i64 %address.local449, !dbg !17
  %bnk.n396 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %bnk.n397 = ptrtoint ptr addrspace(1) %address.local832 to i64, !dbg !17
  %bnk.n398 = sub i64 %bnk.n397, %bnk.n396, !dbg !17
  %dma.buffer.offset1576 = trunc i64 %bnk.n398 to i32, !dbg !17
  %bnk.n399 = trunc i64 %bnk.n396 to i32, !dbg !17
  %bnk.n400 = insertelement <4 x i32> poison, i32 %bnk.n399, i64 0, !dbg !17
  %bnk.n401 = lshr i64 %bnk.n396, 32, !dbg !17
  %bnk.n402 = trunc i64 %bnk.n401 to i32, !dbg !17
  %bnk.n403 = insertelement <4 x i32> %bnk.n400, i32 %bnk.n402, i64 1, !dbg !17
  %bnk.n404 = insertelement <4 x i32> %bnk.n403, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1577 = insertelement <4 x i32> %bnk.n404, i32 131072, i64 3, !dbg !17
  %bnk.n405 = ptrtoint ptr addrspace(3) %bnk.n395 to i32, !dbg !17
  %dma.buffer.lds1578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n405), !dbg !17
  %bnk.n406 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1577) #10, !dbg !17
  %bnk.n407 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !17
  %bnk.n408 = zext i32 %bnk.n407 to i64, !dbg !17
  %bnk.n409 = inttoptr i64 %bnk.n408 to ptr addrspace(3), !dbg !17
  %address.local833 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n392, i64 %address.local466, !dbg !17
  %bnk.n410 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %bnk.n411 = ptrtoint ptr addrspace(1) %address.local833 to i64, !dbg !17
  %bnk.n412 = sub i64 %bnk.n411, %bnk.n410, !dbg !17
  %dma.buffer.offset1573 = trunc i64 %bnk.n412 to i32, !dbg !17
  %bnk.n413 = trunc i64 %bnk.n410 to i32, !dbg !17
  %bnk.n414 = insertelement <4 x i32> poison, i32 %bnk.n413, i64 0, !dbg !17
  %bnk.n415 = lshr i64 %bnk.n410, 32, !dbg !17
  %bnk.n416 = trunc i64 %bnk.n415 to i32, !dbg !17
  %bnk.n417 = insertelement <4 x i32> %bnk.n414, i32 %bnk.n416, i64 1, !dbg !17
  %bnk.n418 = insertelement <4 x i32> %bnk.n417, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1574 = insertelement <4 x i32> %bnk.n418, i32 131072, i64 3, !dbg !17
  %bnk.n419 = ptrtoint ptr addrspace(3) %bnk.n409 to i32, !dbg !17
  %dma.buffer.lds1575 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n419), !dbg !17
  %bnk.n420 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1574) #10, !dbg !17
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !18
  %bnk.n421 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !18
  %bnk.n422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n421, <4 x float> %bnk.n310) #9, !dbg !18
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !18
  %bnk.n423 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !18
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !18
  %bnk.n424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !18
  %bnk.n425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n423, <4 x float> %bnk.n422) #9, !dbg !18
  %bnk.n426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n424, <4 x float> %bnk.n314) #9, !dbg !18
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !18
  %bnk.n427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !18
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !18
  %bnk.n428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !18
  %bnk.n429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n427, <4 x float> %bnk.n426) #9, !dbg !18
  %bnk.n430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n428, <4 x float> %bnk.n318) #9, !dbg !18
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !18
  %bnk.n431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !18
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !18
  %bnk.n432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !18
  %bnk.n433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n431, <4 x float> %bnk.n430) #9, !dbg !18
  %bnk.n434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n432, <4 x float> %bnk.n322) #9, !dbg !18
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !18
  %bnk.n435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !18
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !18
  %bnk.n436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !18
  %bnk.n437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n435, <4 x float> %bnk.n434) #9, !dbg !18
  %bnk.n438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n436, <4 x float> %bnk.n326) #9, !dbg !18
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !18
  %bnk.n439 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !18
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !18
  %bnk.n440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !18
  %bnk.n441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n439, <4 x float> %bnk.n438) #9, !dbg !18
  %bnk.n442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n440, <4 x float> %bnk.n330) #9, !dbg !18
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !18
  %bnk.n443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !18
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !18
  %bnk.n444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !18
  %bnk.n445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n443, <4 x float> %bnk.n442) #9, !dbg !18
  %bnk.n446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n444, <4 x float> %bnk.n334) #9, !dbg !18
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !18
  %bnk.n447 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !18
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !18
  %bnk.n448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !18
  %bnk.n449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n447, <4 x float> %bnk.n446) #9, !dbg !18
  %bnk.n450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n448, <4 x float> %bnk.n338) #9, !dbg !18
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !18
  %bnk.n451 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !18
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !18
  %bnk.n452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !18
  %bnk.n453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n451, <4 x float> %bnk.n450) #9, !dbg !18
  %bnk.n454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n452, <4 x float> %bnk.n342) #9, !dbg !18
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !18
  %bnk.n455 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !18
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !18
  %bnk.n456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !18
  %bnk.n457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n455, <4 x float> %bnk.n454) #9, !dbg !18
  %bnk.n458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n456, <4 x float> %bnk.n346) #9, !dbg !18
  %lds.const.gep1473 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !18
  %bnk.n459 = load <4 x half>, ptr addrspace(3) %lds.const.gep1473, align 8, !dbg !18
  %lds.const.gep1474 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !18
  %bnk.n460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1474, align 8, !dbg !18
  %bnk.n461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n459, <4 x float> %bnk.n458) #9, !dbg !18
  %bnk.n462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n460, <4 x float> %bnk.n350) #9, !dbg !18
  %lds.const.gep1475 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !18
  %bnk.n463 = load <4 x half>, ptr addrspace(3) %lds.const.gep1475, align 8, !dbg !18
  %lds.const.gep1476 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !18
  %bnk.n464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1476, align 8, !dbg !18
  %bnk.n465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n463, <4 x float> %bnk.n462) #9, !dbg !18
  %bnk.n466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n464, <4 x float> %bnk.n354) #9, !dbg !18
  %lds.const.gep1477 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !18
  %bnk.n467 = load <4 x half>, ptr addrspace(3) %lds.const.gep1477, align 8, !dbg !18
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !18
  %bnk.n468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !18
  %bnk.n469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n467, <4 x float> %bnk.n466) #9, !dbg !18
  %bnk.n470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n468, <4 x float> %bnk.n358) #9, !dbg !18
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !18
  %bnk.n471 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !18
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !18
  %bnk.n472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !18
  %bnk.n473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n471, <4 x float> %bnk.n470) #9, !dbg !18
  %bnk.n474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n472, <4 x float> %bnk.n362) #9, !dbg !18
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !18
  %bnk.n475 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !18
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !18
  %bnk.n476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !18
  %bnk.n477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n475, <4 x float> %bnk.n474) #9, !dbg !18
  %bnk.n478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n476, <4 x float> %bnk.n366) #9, !dbg !18
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !18
  %bnk.n479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !18
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !18
  %bnk.n480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !18
  %bnk.n481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n479, <4 x float> %bnk.n478) #9, !dbg !18
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !18
  %bnk.n482 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %bnk.n483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n480, <4 x float> %bnk.n369) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %bnk.n484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n482, <4 x float> %bnk.n483) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %bnk.n485 = add nuw i64 %bnk.n205, 3072, !dbg !17
  %bnk.n486 = icmp slt i64 %bnk.n485, 0, !dbg !17
  %bnk.n487 = sub nuw i64 -3073, %bnk.n205, !dbg !17
  %bnk.n488 = select i1 %bnk.n486, i64 %bnk.n487, i64 %bnk.n485, !dbg !17
  %bnk.n489 = sdiv i64 %bnk.n488, 4096, !dbg !17
  %.lobit39 = ashr i64 %bnk.n485, 63, !dbg !17
  %bnk.n490 = xor i64 %bnk.n489, %.lobit39, !dbg !17
  %bnk.n491 = add nsw i64 %bnk.n490, %bnk.n16, !dbg !17
  %bnk.n492 = add nsw i64 %bnk.n261, 96, !dbg !17
  %bnk.n493 = icmp slt i64 %bnk.n261, -96, !dbg !17
  %bnk.n494 = sub nsw i64 -97, %bnk.n261, !dbg !17
  %bnk.n495 = select i1 %bnk.n493, i64 %bnk.n494, i64 %bnk.n492, !dbg !17
  %bnk.n496 = sdiv i64 %bnk.n495, 128, !dbg !17
  %bnk.n497 = sext i1 %bnk.n493 to i64, !dbg !17
  %bnk.n498 = xor i64 %bnk.n496, %bnk.n497, !dbg !17
  %bnk.n499 = shl nsw i64 %bnk.n498, 7, !dbg !17
  %bnk.n500 = sub i64 %bnk.n195, %bnk.n499, !dbg !17
  %bnk.n501 = shl i64 %bnk.n491, 19, !dbg !17
  %bnk.n502 = and i64 %bnk.n501, 16252928, !dbg !17
  %bnk.n503 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %bnk.n502, !dbg !17
  %bnk.n504 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n503, i64 %bnk.n216, !dbg !17
  %bnk.n505 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n504, i64 %bnk.n500, !dbg !17
  %bnk.n506 = getelementptr i8, ptr addrspace(1) %bnk.n505, i64 192, !dbg !17
  %bnk.n507 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n506, i64 %bnk.n222, !dbg !17
  %bnk.n508 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !17
  %bnk.n509 = zext i32 %bnk.n508 to i64, !dbg !17
  %bnk.n510 = inttoptr i64 %bnk.n509 to ptr addrspace(3), !dbg !17
  %address.local850 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n507, i64 %address.local449, !dbg !17
  %bnk.n511 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %bnk.n512 = ptrtoint ptr addrspace(1) %address.local850 to i64, !dbg !17
  %bnk.n513 = sub i64 %bnk.n512, %bnk.n511, !dbg !17
  %dma.buffer.offset1570 = trunc i64 %bnk.n513 to i32, !dbg !17
  %bnk.n514 = trunc i64 %bnk.n511 to i32, !dbg !17
  %bnk.n515 = insertelement <4 x i32> poison, i32 %bnk.n514, i64 0, !dbg !17
  %bnk.n516 = lshr i64 %bnk.n511, 32, !dbg !17
  %bnk.n517 = trunc i64 %bnk.n516 to i32, !dbg !17
  %bnk.n518 = insertelement <4 x i32> %bnk.n515, i32 %bnk.n517, i64 1, !dbg !17
  %bnk.n519 = insertelement <4 x i32> %bnk.n518, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1571 = insertelement <4 x i32> %bnk.n519, i32 131072, i64 3, !dbg !17
  %bnk.n520 = ptrtoint ptr addrspace(3) %bnk.n510 to i32, !dbg !17
  %dma.buffer.lds1572 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n520), !dbg !17
  %bnk.n521 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1572, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1571) #10, !dbg !17
  %bnk.n522 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !17
  %bnk.n523 = zext i32 %bnk.n522 to i64, !dbg !17
  %bnk.n524 = inttoptr i64 %bnk.n523 to ptr addrspace(3), !dbg !17
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n507, i64 %address.local466, !dbg !17
  %bnk.n525 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %bnk.n526 = ptrtoint ptr addrspace(1) %address.local851 to i64, !dbg !17
  %bnk.n527 = sub i64 %bnk.n526, %bnk.n525, !dbg !17
  %dma.buffer.offset1567 = trunc i64 %bnk.n527 to i32, !dbg !17
  %bnk.n528 = trunc i64 %bnk.n525 to i32, !dbg !17
  %bnk.n529 = insertelement <4 x i32> poison, i32 %bnk.n528, i64 0, !dbg !17
  %bnk.n530 = lshr i64 %bnk.n525, 32, !dbg !17
  %bnk.n531 = trunc i64 %bnk.n530 to i32, !dbg !17
  %bnk.n532 = insertelement <4 x i32> %bnk.n529, i32 %bnk.n531, i64 1, !dbg !17
  %bnk.n533 = insertelement <4 x i32> %bnk.n532, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1568 = insertelement <4 x i32> %bnk.n533, i32 131072, i64 3, !dbg !17
  %bnk.n534 = ptrtoint ptr addrspace(3) %bnk.n524 to i32, !dbg !17
  %dma.buffer.lds1569 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n534), !dbg !17
  %bnk.n535 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1569, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1568) #10, !dbg !17
  %bnk.n536 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !18
  %bnk.n537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n536, <4 x float> %bnk.n425) #9, !dbg !18
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !18
  %bnk.n538 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !18
  %bnk.n539 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !18
  %bnk.n540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n538, <4 x float> %bnk.n537) #9, !dbg !18
  %bnk.n541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n539, <4 x float> %bnk.n429) #9, !dbg !18
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !18
  %bnk.n542 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !18
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !18
  %bnk.n543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !18
  %bnk.n544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n542, <4 x float> %bnk.n541) #9, !dbg !18
  %bnk.n545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n543, <4 x float> %bnk.n433) #9, !dbg !18
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !18
  %bnk.n546 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !18
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !18
  %bnk.n547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !18
  %bnk.n548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n546, <4 x float> %bnk.n545) #9, !dbg !18
  %bnk.n549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n547, <4 x float> %bnk.n437) #9, !dbg !18
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !18
  %bnk.n550 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !18
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !18
  %bnk.n551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !18
  %bnk.n552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n550, <4 x float> %bnk.n549) #9, !dbg !18
  %bnk.n553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n551, <4 x float> %bnk.n441) #9, !dbg !18
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !18
  %bnk.n554 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !18
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !18
  %bnk.n555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !18
  %bnk.n556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n554, <4 x float> %bnk.n553) #9, !dbg !18
  %bnk.n557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n555, <4 x float> %bnk.n445) #9, !dbg !18
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !18
  %bnk.n558 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !18
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !18
  %bnk.n559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !18
  %bnk.n560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n558, <4 x float> %bnk.n557) #9, !dbg !18
  %bnk.n561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n559, <4 x float> %bnk.n449) #9, !dbg !18
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !18
  %bnk.n562 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !18
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !18
  %bnk.n563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !18
  %bnk.n564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n562, <4 x float> %bnk.n561) #9, !dbg !18
  %bnk.n565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n563, <4 x float> %bnk.n453) #9, !dbg !18
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !18
  %bnk.n566 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !18
  %bnk.n567 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !18
  %bnk.n568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n566, <4 x float> %bnk.n565) #9, !dbg !18
  %bnk.n569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n567, <4 x float> %bnk.n457) #9, !dbg !18
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !18
  %bnk.n570 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !18
  %bnk.n571 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !18
  %bnk.n572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n570, <4 x float> %bnk.n569) #9, !dbg !18
  %bnk.n573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n571, <4 x float> %bnk.n461) #9, !dbg !18
  %lds.const.gep1501 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !18
  %bnk.n574 = load <4 x half>, ptr addrspace(3) %lds.const.gep1501, align 8, !dbg !18
  %lds.const.gep1502 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !18
  %bnk.n575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1502, align 8, !dbg !18
  %bnk.n576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n574, <4 x float> %bnk.n573) #9, !dbg !18
  %bnk.n577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n575, <4 x float> %bnk.n465) #9, !dbg !18
  %lds.const.gep1503 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !18
  %bnk.n578 = load <4 x half>, ptr addrspace(3) %lds.const.gep1503, align 8, !dbg !18
  %lds.const.gep1504 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !18
  %bnk.n579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1504, align 8, !dbg !18
  %bnk.n580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n578, <4 x float> %bnk.n577) #9, !dbg !18
  %bnk.n581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n579, <4 x float> %bnk.n469) #9, !dbg !18
  %lds.const.gep1505 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !18
  %bnk.n582 = load <4 x half>, ptr addrspace(3) %lds.const.gep1505, align 8, !dbg !18
  %lds.const.gep1506 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !18
  %bnk.n583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1506, align 8, !dbg !18
  %bnk.n584 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n582, <4 x float> %bnk.n581) #9, !dbg !18
  %bnk.n585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n583, <4 x float> %bnk.n473) #9, !dbg !18
  %lds.const.gep1507 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !18
  %bnk.n586 = load <4 x half>, ptr addrspace(3) %lds.const.gep1507, align 8, !dbg !18
  %lds.const.gep1508 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !18
  %bnk.n587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1508, align 8, !dbg !18
  %bnk.n588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n586, <4 x float> %bnk.n585) #9, !dbg !18
  %bnk.n589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n587, <4 x float> %bnk.n477) #9, !dbg !18
  %lds.const.gep1509 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !18
  %bnk.n590 = load <4 x half>, ptr addrspace(3) %lds.const.gep1509, align 8, !dbg !18
  %lds.const.gep1510 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !18
  %bnk.n591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1510, align 8, !dbg !18
  %bnk.n592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n590, <4 x float> %bnk.n589) #9, !dbg !18
  %bnk.n593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n591, <4 x float> %bnk.n481) #9, !dbg !18
  %lds.const.gep1511 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !18
  %bnk.n594 = load <4 x half>, ptr addrspace(3) %lds.const.gep1511, align 8, !dbg !18
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !18
  %bnk.n595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !18
  %bnk.n596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n594, <4 x float> %bnk.n593) #9, !dbg !18
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !18
  %bnk.n597 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %bnk.n598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n595, <4 x float> %bnk.n484) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %bnk.n599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n597, <4 x float> %bnk.n598) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %bnk.n600 = add nuw nsw i64 %bnk.n216, %bnk.n16, !dbg !19
  %bnk.n601 = shl i64 %bnk.n600, 19, !dbg !19
  %bnk.n602 = and i64 %bnk.n601, 16252928, !dbg !19
  %bnk.n603 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %bnk.n602, !dbg !19
  %bnk.n604 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n603, i64 %bnk.n222, !dbg !19
  %bnk.n605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !19
  %bnk.n606 = zext i32 %bnk.n605 to i64, !dbg !19
  %bnk.n607 = inttoptr i64 %bnk.n606 to ptr addrspace(3), !dbg !19
  %address.local852 = shl nuw nsw i64 %address.local434, 5, !dbg !19
  %address.local853 = and i64 %address.local852, 8064, !dbg !19
  %address.local854 = shl nuw nsw i64 %address.local434, 3, !dbg !19
  %address.local855 = and i64 %address.local854, 24, !dbg !19
  %address.local856 = or disjoint i64 %address.local853, %address.local855, !dbg !19
  %address.local857 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n604, i64 %address.local856, !dbg !19
  %bnk.n608 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %bnk.n609 = ptrtoint ptr addrspace(1) %address.local857 to i64, !dbg !19
  %bnk.n610 = sub i64 %bnk.n609, %bnk.n608, !dbg !19
  %dma.buffer.offset1564 = trunc i64 %bnk.n610 to i32, !dbg !19
  %bnk.n611 = trunc i64 %bnk.n608 to i32, !dbg !19
  %bnk.n612 = insertelement <4 x i32> poison, i32 %bnk.n611, i64 0, !dbg !19
  %bnk.n613 = lshr i64 %bnk.n608, 32, !dbg !19
  %bnk.n614 = trunc i64 %bnk.n613 to i32, !dbg !19
  %bnk.n615 = insertelement <4 x i32> %bnk.n612, i32 %bnk.n614, i64 1, !dbg !19
  %bnk.n616 = insertelement <4 x i32> %bnk.n615, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1565 = insertelement <4 x i32> %bnk.n616, i32 131072, i64 3, !dbg !19
  %bnk.n617 = ptrtoint ptr addrspace(3) %bnk.n607 to i32, !dbg !19
  %dma.buffer.lds1566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n617), !dbg !19
  %bnk.n618 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1566, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #10, !dbg !19
  %bnk.n619 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !19
  %bnk.n620 = zext i32 %bnk.n619 to i64, !dbg !19
  %bnk.n621 = inttoptr i64 %bnk.n620 to ptr addrspace(3), !dbg !19
  %address.local858 = shl nuw nsw i64 %address.local451, 5, !dbg !19
  %address.local859 = and i64 %address.local858, 16256, !dbg !19
  %address.local860 = shl nuw nsw i64 %address.local434, 3, !dbg !19
  %address.local861 = and i64 %address.local860, 24, !dbg !19
  %address.local862 = or disjoint i64 %address.local859, %address.local861, !dbg !19
  %address.local863 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n604, i64 %address.local862, !dbg !19
  %bnk.n622 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %bnk.n623 = ptrtoint ptr addrspace(1) %address.local863 to i64, !dbg !19
  %bnk.n624 = sub i64 %bnk.n623, %bnk.n622, !dbg !19
  %dma.buffer.offset1561 = trunc i64 %bnk.n624 to i32, !dbg !19
  %bnk.n625 = trunc i64 %bnk.n622 to i32, !dbg !19
  %bnk.n626 = insertelement <4 x i32> poison, i32 %bnk.n625, i64 0, !dbg !19
  %bnk.n627 = lshr i64 %bnk.n622, 32, !dbg !19
  %bnk.n628 = trunc i64 %bnk.n627 to i32, !dbg !19
  %bnk.n629 = insertelement <4 x i32> %bnk.n626, i32 %bnk.n628, i64 1, !dbg !19
  %bnk.n630 = insertelement <4 x i32> %bnk.n629, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1562 = insertelement <4 x i32> %bnk.n630, i32 131072, i64 3, !dbg !19
  %bnk.n631 = ptrtoint ptr addrspace(3) %bnk.n621 to i32, !dbg !19
  %dma.buffer.lds1563 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n631), !dbg !19
  %bnk.n632 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1563, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1562) #10, !dbg !19
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !18
  %bnk.n633 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !18
  %bnk.n634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n633, <4 x float> %bnk.n540) #9, !dbg !18
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !18
  %bnk.n635 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !18
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !18
  %bnk.n636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !18
  %bnk.n637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n635, <4 x float> %bnk.n634) #9, !dbg !18
  %bnk.n638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n636, <4 x float> %bnk.n544) #9, !dbg !18
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !18
  %bnk.n639 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !18
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !18
  %bnk.n640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !18
  %bnk.n641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n639, <4 x float> %bnk.n638) #9, !dbg !18
  %bnk.n642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n640, <4 x float> %bnk.n548) #9, !dbg !18
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !18
  %bnk.n643 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !18
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !18
  %bnk.n644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !18
  %bnk.n645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n643, <4 x float> %bnk.n642) #9, !dbg !18
  %bnk.n646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n644, <4 x float> %bnk.n552) #9, !dbg !18
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !18
  %bnk.n647 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !18
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !18
  %bnk.n648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !18
  %bnk.n649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n647, <4 x float> %bnk.n646) #9, !dbg !18
  %bnk.n650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n648, <4 x float> %bnk.n556) #9, !dbg !18
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !18
  %bnk.n651 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !18
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !18
  %bnk.n652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !18
  %bnk.n653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n651, <4 x float> %bnk.n650) #9, !dbg !18
  %bnk.n654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n652, <4 x float> %bnk.n560) #9, !dbg !18
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !18
  %bnk.n655 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !18
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !18
  %bnk.n656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !18
  %bnk.n657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n655, <4 x float> %bnk.n654) #9, !dbg !18
  %bnk.n658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n656, <4 x float> %bnk.n564) #9, !dbg !18
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !18
  %bnk.n659 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !18
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !18
  %bnk.n660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !18
  %bnk.n661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n659, <4 x float> %bnk.n658) #9, !dbg !18
  %bnk.n662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n660, <4 x float> %bnk.n568) #9, !dbg !18
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !18
  %bnk.n663 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !18
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !18
  %bnk.n664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !18
  %bnk.n665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n663, <4 x float> %bnk.n662) #9, !dbg !18
  %bnk.n666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n664, <4 x float> %bnk.n572) #9, !dbg !18
  %lds.const.gep1531 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !18
  %bnk.n667 = load <4 x half>, ptr addrspace(3) %lds.const.gep1531, align 8, !dbg !18
  %lds.const.gep1532 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !18
  %bnk.n668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1532, align 8, !dbg !18
  %bnk.n669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n667, <4 x float> %bnk.n666) #9, !dbg !18
  %bnk.n670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n668, <4 x float> %bnk.n576) #9, !dbg !18
  %lds.const.gep1533 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !18
  %bnk.n671 = load <4 x half>, ptr addrspace(3) %lds.const.gep1533, align 8, !dbg !18
  %lds.const.gep1534 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !18
  %bnk.n672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1534, align 8, !dbg !18
  %bnk.n673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n671, <4 x float> %bnk.n670) #9, !dbg !18
  %bnk.n674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n672, <4 x float> %bnk.n580) #9, !dbg !18
  %lds.const.gep1535 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !18
  %bnk.n675 = load <4 x half>, ptr addrspace(3) %lds.const.gep1535, align 8, !dbg !18
  %lds.const.gep1536 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !18
  %bnk.n676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1536, align 8, !dbg !18
  %bnk.n677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n675, <4 x float> %bnk.n674) #9, !dbg !18
  %bnk.n678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n676, <4 x float> %bnk.n584) #9, !dbg !18
  %lds.const.gep1537 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !18
  %bnk.n679 = load <4 x half>, ptr addrspace(3) %lds.const.gep1537, align 8, !dbg !18
  %lds.const.gep1538 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !18
  %bnk.n680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1538, align 8, !dbg !18
  %bnk.n681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n679, <4 x float> %bnk.n678) #9, !dbg !18
  %bnk.n682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n680, <4 x float> %bnk.n588) #9, !dbg !18
  %lds.const.gep1539 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !18
  %bnk.n683 = load <4 x half>, ptr addrspace(3) %lds.const.gep1539, align 8, !dbg !18
  %lds.const.gep1540 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !18
  %bnk.n684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1540, align 8, !dbg !18
  %bnk.n685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n683, <4 x float> %bnk.n682) #9, !dbg !18
  %bnk.n686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n684, <4 x float> %bnk.n592) #9, !dbg !18
  %lds.const.gep1541 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !18
  %bnk.n687 = load <4 x half>, ptr addrspace(3) %lds.const.gep1541, align 8, !dbg !18
  %lds.const.gep1542 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !18
  %bnk.n688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1542, align 8, !dbg !18
  %bnk.n689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n687, <4 x float> %bnk.n686) #9, !dbg !18
  %bnk.n690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n688, <4 x float> %bnk.n596) #9, !dbg !18
  %lds.const.gep1543 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !18
  %bnk.n691 = load <4 x half>, ptr addrspace(3) %lds.const.gep1543, align 8, !dbg !18
  %lds.const.gep1544 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !18
  %bnk.n692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1544, align 8, !dbg !18
  %bnk.n693 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n691, <4 x float> %bnk.n690) #9, !dbg !18
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !18
  %bnk.n694 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %bnk.n695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n692, <4 x float> %bnk.n599) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %bnk.n696 = or disjoint i64 %bnk.n213, 127, !dbg !14
  %bnk.n697 = icmp uge i64 %bnk.n64, %bnk.n696, !dbg !14
  %bnk.n698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n694, <4 x float> %bnk.n695) #9, !dbg !18
  %bnk.n699 = and i1 %bnk.n190, %bnk.n697, !dbg !14
  br i1 %bnk.n699, label %bnk.n700, label %bnk.n701, !dbg !14

bnk.n700:                                              ; preds = %bnk.n212
  br label %bnk.n845, !dbg !14

bnk.n701:                                              ; preds = %bnk.n212
  %bnk.n702 = add i64 %bnk.n64, %bnk.n32, !dbg !14
  %bnk.n703 = shl nuw nsw i64 %bnk.n65, 4, !dbg !14
  %bnk.n704 = add i64 %bnk.n702, %bnk.n703, !dbg !14
  %bnk.n705 = and i32 %bnk.n31, 240, !dbg !14
  %index.compact = zext nneg i32 %bnk.n705 to i64, !dbg !14
  %bnk.n706 = sub i64 %bnk.n704, %index.compact, !dbg !14
  %bnk.n707 = and i64 %bnk.n32, 48, !dbg !14
  %bnk.n708 = or disjoint i64 %bnk.n213, %bnk.n707, !dbg !14
  %bnk.n709 = add i64 %bnk.n706, 1, !dbg !24
  %bnk.n710 = lshr i64 %bnk.n709, 32, !dbg !25
  %compare.high = trunc i64 %bnk.n710 to i32, !dbg !25
  %bnk.n711 = lshr i64 %bnk.n708, 32, !dbg !25
  %compare.high1157 = trunc i64 %bnk.n711 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !25
  %compare.low.base = trunc i64 %bnk.n709 to i32, !dbg !25
  %compare.low.base1158 = trunc i64 %bnk.n708 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1158, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %bnk.n712 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low1159 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result1160 = select i1 %compare.high.equal, i1 %compare.low1159, i1 %compare.high.order, !dbg !25
  %bnk.n713 = select i1 %compare.result1160, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1161 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1162 = or i32 %compare.local.base1161, 2, !dbg !25
  %compare.low1163 = icmp ugt i32 %compare.low.base, %compare.local1162, !dbg !25
  %compare.result1164 = select i1 %compare.high.equal, i1 %compare.low1163, i1 %compare.high.order, !dbg !25
  %bnk.n714 = select i1 %compare.result1164, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1165 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1166 = or i32 %compare.local.base1165, 3, !dbg !25
  %compare.low1167 = icmp ugt i32 %compare.low.base, %compare.local1166, !dbg !25
  %compare.result1168 = select i1 %compare.high.equal, i1 %compare.low1167, i1 %compare.high.order, !dbg !25
  %bnk.n715 = select i1 %compare.result1168, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1169 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1170 = or i32 %compare.local.base1169, 4, !dbg !25
  %compare.low1171 = icmp ugt i32 %compare.low.base, %compare.local1170, !dbg !25
  %compare.result1172 = select i1 %compare.high.equal, i1 %compare.low1171, i1 %compare.high.order, !dbg !25
  %bnk.n716 = select i1 %compare.result1172, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1173 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1174 = or i32 %compare.local.base1173, 5, !dbg !25
  %compare.low1175 = icmp ugt i32 %compare.low.base, %compare.local1174, !dbg !25
  %compare.result1176 = select i1 %compare.high.equal, i1 %compare.low1175, i1 %compare.high.order, !dbg !25
  %bnk.n717 = select i1 %compare.result1176, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1177 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1178 = or i32 %compare.local.base1177, 6, !dbg !25
  %compare.low1179 = icmp ugt i32 %compare.low.base, %compare.local1178, !dbg !25
  %compare.result1180 = select i1 %compare.high.equal, i1 %compare.low1179, i1 %compare.high.order, !dbg !25
  %bnk.n718 = select i1 %compare.result1180, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1181 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1182 = or i32 %compare.local.base1181, 7, !dbg !25
  %compare.low1183 = icmp ugt i32 %compare.low.base, %compare.local1182, !dbg !25
  %compare.result1184 = select i1 %compare.high.equal, i1 %compare.low1183, i1 %compare.high.order, !dbg !25
  %bnk.n719 = select i1 %compare.result1184, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1185 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1186 = or i32 %compare.local.base1185, 8, !dbg !25
  %compare.low1187 = icmp ugt i32 %compare.low.base, %compare.local1186, !dbg !25
  %compare.result1188 = select i1 %compare.high.equal, i1 %compare.low1187, i1 %compare.high.order, !dbg !25
  %bnk.n720 = select i1 %compare.result1188, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1189 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1190 = or i32 %compare.local.base1189, 9, !dbg !25
  %compare.low1191 = icmp ugt i32 %compare.low.base, %compare.local1190, !dbg !25
  %compare.result1192 = select i1 %compare.high.equal, i1 %compare.low1191, i1 %compare.high.order, !dbg !25
  %bnk.n721 = select i1 %compare.result1192, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1193 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1194 = or i32 %compare.local.base1193, 10, !dbg !25
  %compare.low1195 = icmp ugt i32 %compare.low.base, %compare.local1194, !dbg !25
  %compare.result1196 = select i1 %compare.high.equal, i1 %compare.low1195, i1 %compare.high.order, !dbg !25
  %bnk.n722 = select i1 %compare.result1196, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1197 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1198 = or i32 %compare.local.base1197, 11, !dbg !25
  %compare.low1199 = icmp ugt i32 %compare.low.base, %compare.local1198, !dbg !25
  %compare.result1200 = select i1 %compare.high.equal, i1 %compare.low1199, i1 %compare.high.order, !dbg !25
  %bnk.n723 = select i1 %compare.result1200, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1201 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1202 = or i32 %compare.local.base1201, 12, !dbg !25
  %compare.low1203 = icmp ugt i32 %compare.low.base, %compare.local1202, !dbg !25
  %compare.result1204 = select i1 %compare.high.equal, i1 %compare.low1203, i1 %compare.high.order, !dbg !25
  %bnk.n724 = select i1 %compare.result1204, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1205 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1206 = or i32 %compare.local.base1205, 13, !dbg !25
  %compare.low1207 = icmp ugt i32 %compare.low.base, %compare.local1206, !dbg !25
  %compare.result1208 = select i1 %compare.high.equal, i1 %compare.low1207, i1 %compare.high.order, !dbg !25
  %bnk.n725 = select i1 %compare.result1208, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1209 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1210 = or i32 %compare.local.base1209, 14, !dbg !25
  %compare.low1211 = icmp ugt i32 %compare.low.base, %compare.local1210, !dbg !25
  %compare.result1212 = select i1 %compare.high.equal, i1 %compare.low1211, i1 %compare.high.order, !dbg !25
  %bnk.n726 = select i1 %compare.result1212, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1213 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 15) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1214 = or i32 %compare.local.base1213, 15, !dbg !25
  %compare.low1215 = icmp ugt i32 %compare.low.base, %compare.local1214, !dbg !25
  %compare.result1216 = select i1 %compare.high.equal, i1 %compare.low1215, i1 %compare.high.order, !dbg !25
  %bnk.n727 = select i1 %compare.result1216, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1217 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 16) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1218 = or i32 %compare.local.base1217, 64, !dbg !25
  %compare.low1219 = icmp ugt i32 %compare.low.base, %compare.local1218, !dbg !25
  %compare.result1220 = select i1 %compare.high.equal, i1 %compare.low1219, i1 %compare.high.order, !dbg !25
  %bnk.n728 = select i1 %compare.result1220, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1221 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 17) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1222 = or i32 %compare.local.base1221, 65, !dbg !25
  %compare.low1223 = icmp ugt i32 %compare.low.base, %compare.local1222, !dbg !25
  %compare.result1224 = select i1 %compare.high.equal, i1 %compare.low1223, i1 %compare.high.order, !dbg !25
  %bnk.n729 = select i1 %compare.result1224, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1225 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 18) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1226 = or i32 %compare.local.base1225, 66, !dbg !25
  %compare.low1227 = icmp ugt i32 %compare.low.base, %compare.local1226, !dbg !25
  %compare.result1228 = select i1 %compare.high.equal, i1 %compare.low1227, i1 %compare.high.order, !dbg !25
  %bnk.n730 = select i1 %compare.result1228, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1229 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 19) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1230 = or i32 %compare.local.base1229, 67, !dbg !25
  %compare.low1231 = icmp ugt i32 %compare.low.base, %compare.local1230, !dbg !25
  %compare.result1232 = select i1 %compare.high.equal, i1 %compare.low1231, i1 %compare.high.order, !dbg !25
  %bnk.n731 = select i1 %compare.result1232, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1233 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 20) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1234 = or i32 %compare.local.base1233, 68, !dbg !25
  %compare.low1235 = icmp ugt i32 %compare.low.base, %compare.local1234, !dbg !25
  %compare.result1236 = select i1 %compare.high.equal, i1 %compare.low1235, i1 %compare.high.order, !dbg !25
  %bnk.n732 = select i1 %compare.result1236, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1237 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 21) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1238 = or i32 %compare.local.base1237, 69, !dbg !25
  %compare.low1239 = icmp ugt i32 %compare.low.base, %compare.local1238, !dbg !25
  %compare.result1240 = select i1 %compare.high.equal, i1 %compare.low1239, i1 %compare.high.order, !dbg !25
  %bnk.n733 = select i1 %compare.result1240, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1241 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 22) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1242 = or i32 %compare.local.base1241, 70, !dbg !25
  %compare.low1243 = icmp ugt i32 %compare.low.base, %compare.local1242, !dbg !25
  %compare.result1244 = select i1 %compare.high.equal, i1 %compare.low1243, i1 %compare.high.order, !dbg !25
  %bnk.n734 = select i1 %compare.result1244, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1245 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 23) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1246 = or i32 %compare.local.base1245, 71, !dbg !25
  %compare.low1247 = icmp ugt i32 %compare.low.base, %compare.local1246, !dbg !25
  %compare.result1248 = select i1 %compare.high.equal, i1 %compare.low1247, i1 %compare.high.order, !dbg !25
  %bnk.n735 = select i1 %compare.result1248, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1249 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 24) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1250 = or i32 %compare.local.base1249, 72, !dbg !25
  %compare.low1251 = icmp ugt i32 %compare.low.base, %compare.local1250, !dbg !25
  %compare.result1252 = select i1 %compare.high.equal, i1 %compare.low1251, i1 %compare.high.order, !dbg !25
  %bnk.n736 = select i1 %compare.result1252, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1253 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 25) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1254 = or i32 %compare.local.base1253, 73, !dbg !25
  %compare.low1255 = icmp ugt i32 %compare.low.base, %compare.local1254, !dbg !25
  %compare.result1256 = select i1 %compare.high.equal, i1 %compare.low1255, i1 %compare.high.order, !dbg !25
  %bnk.n737 = select i1 %compare.result1256, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1257 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 26) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1258 = or i32 %compare.local.base1257, 74, !dbg !25
  %compare.low1259 = icmp ugt i32 %compare.low.base, %compare.local1258, !dbg !25
  %compare.result1260 = select i1 %compare.high.equal, i1 %compare.low1259, i1 %compare.high.order, !dbg !25
  %bnk.n738 = select i1 %compare.result1260, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1261 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 27) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1262 = or i32 %compare.local.base1261, 75, !dbg !25
  %compare.low1263 = icmp ugt i32 %compare.low.base, %compare.local1262, !dbg !25
  %compare.result1264 = select i1 %compare.high.equal, i1 %compare.low1263, i1 %compare.high.order, !dbg !25
  %bnk.n739 = select i1 %compare.result1264, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1265 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 28) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1266 = or i32 %compare.local.base1265, 76, !dbg !25
  %compare.low1267 = icmp ugt i32 %compare.low.base, %compare.local1266, !dbg !25
  %compare.result1268 = select i1 %compare.high.equal, i1 %compare.low1267, i1 %compare.high.order, !dbg !25
  %bnk.n740 = select i1 %compare.result1268, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1269 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 29) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1270 = or i32 %compare.local.base1269, 77, !dbg !25
  %compare.low1271 = icmp ugt i32 %compare.low.base, %compare.local1270, !dbg !25
  %compare.result1272 = select i1 %compare.high.equal, i1 %compare.low1271, i1 %compare.high.order, !dbg !25
  %bnk.n741 = select i1 %compare.result1272, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 30) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1274 = or i32 %compare.local.base1273, 78, !dbg !25
  %compare.low1275 = icmp ugt i32 %compare.low.base, %compare.local1274, !dbg !25
  %compare.result1276 = select i1 %compare.high.equal, i1 %compare.low1275, i1 %compare.high.order, !dbg !25
  %bnk.n742 = select i1 %compare.result1276, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 31) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1278 = or i32 %compare.local.base1277, 79, !dbg !25
  %compare.low1279 = icmp ugt i32 %compare.low.base, %compare.local1278, !dbg !25
  %compare.result1280 = select i1 %compare.high.equal, i1 %compare.low1279, i1 %compare.high.order, !dbg !25
  %bnk.n743 = select i1 %compare.result1280, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %bnk.n744 = add i64 %bnk.n706, 65, !dbg !24
  %bnk.n745 = lshr i64 %bnk.n744, 32, !dbg !25
  %compare.high1281 = trunc i64 %bnk.n745 to i32, !dbg !25
  %bnk.n746 = lshr i64 %bnk.n708, 32, !dbg !25
  %compare.high1282 = trunc i64 %bnk.n746 to i32, !dbg !25
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1282, !dbg !25
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1282, !dbg !25
  %compare.low.base1285 = trunc i64 %bnk.n744 to i32, !dbg !25
  %compare.low.base1286 = trunc i64 %bnk.n708 to i32, !dbg !25
  %compare.low1287 = icmp ugt i32 %compare.low.base1285, %compare.low.base1286, !dbg !25
  %compare.result1288 = select i1 %compare.high.equal1283, i1 %compare.low1287, i1 %compare.high.order1284, !dbg !25
  %bnk.n747 = select i1 %compare.result1288, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 32) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1290 = or i32 %compare.local.base1289, 1, !dbg !25
  %compare.low1291 = icmp ugt i32 %compare.low.base1285, %compare.local1290, !dbg !25
  %compare.result1292 = select i1 %compare.high.equal1283, i1 %compare.low1291, i1 %compare.high.order1284, !dbg !25
  %bnk.n748 = select i1 %compare.result1292, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 33) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1294 = or i32 %compare.local.base1293, 2, !dbg !25
  %compare.low1295 = icmp ugt i32 %compare.low.base1285, %compare.local1294, !dbg !25
  %compare.result1296 = select i1 %compare.high.equal1283, i1 %compare.low1295, i1 %compare.high.order1284, !dbg !25
  %bnk.n749 = select i1 %compare.result1296, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 34) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1298 = or i32 %compare.local.base1297, 3, !dbg !25
  %compare.low1299 = icmp ugt i32 %compare.low.base1285, %compare.local1298, !dbg !25
  %compare.result1300 = select i1 %compare.high.equal1283, i1 %compare.low1299, i1 %compare.high.order1284, !dbg !25
  %bnk.n750 = select i1 %compare.result1300, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 35) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1302 = or i32 %compare.local.base1301, 4, !dbg !25
  %compare.low1303 = icmp ugt i32 %compare.low.base1285, %compare.local1302, !dbg !25
  %compare.result1304 = select i1 %compare.high.equal1283, i1 %compare.low1303, i1 %compare.high.order1284, !dbg !25
  %bnk.n751 = select i1 %compare.result1304, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 36) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1306 = or i32 %compare.local.base1305, 5, !dbg !25
  %compare.low1307 = icmp ugt i32 %compare.low.base1285, %compare.local1306, !dbg !25
  %compare.result1308 = select i1 %compare.high.equal1283, i1 %compare.low1307, i1 %compare.high.order1284, !dbg !25
  %bnk.n752 = select i1 %compare.result1308, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 37) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1310 = or i32 %compare.local.base1309, 6, !dbg !25
  %compare.low1311 = icmp ugt i32 %compare.low.base1285, %compare.local1310, !dbg !25
  %compare.result1312 = select i1 %compare.high.equal1283, i1 %compare.low1311, i1 %compare.high.order1284, !dbg !25
  %bnk.n753 = select i1 %compare.result1312, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 38) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1314 = or i32 %compare.local.base1313, 7, !dbg !25
  %compare.low1315 = icmp ugt i32 %compare.low.base1285, %compare.local1314, !dbg !25
  %compare.result1316 = select i1 %compare.high.equal1283, i1 %compare.low1315, i1 %compare.high.order1284, !dbg !25
  %bnk.n754 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 39) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 8, !dbg !25
  %compare.low1319 = icmp ugt i32 %compare.low.base1285, %compare.local1318, !dbg !25
  %compare.result1320 = select i1 %compare.high.equal1283, i1 %compare.low1319, i1 %compare.high.order1284, !dbg !25
  %bnk.n755 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 40) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 9, !dbg !25
  %compare.low1323 = icmp ugt i32 %compare.low.base1285, %compare.local1322, !dbg !25
  %compare.result1324 = select i1 %compare.high.equal1283, i1 %compare.low1323, i1 %compare.high.order1284, !dbg !25
  %bnk.n756 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 41) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 10, !dbg !25
  %compare.low1327 = icmp ugt i32 %compare.low.base1285, %compare.local1326, !dbg !25
  %compare.result1328 = select i1 %compare.high.equal1283, i1 %compare.low1327, i1 %compare.high.order1284, !dbg !25
  %bnk.n757 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 42) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 11, !dbg !25
  %compare.low1331 = icmp ugt i32 %compare.low.base1285, %compare.local1330, !dbg !25
  %compare.result1332 = select i1 %compare.high.equal1283, i1 %compare.low1331, i1 %compare.high.order1284, !dbg !25
  %bnk.n758 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 43) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 12, !dbg !25
  %compare.low1335 = icmp ugt i32 %compare.low.base1285, %compare.local1334, !dbg !25
  %compare.result1336 = select i1 %compare.high.equal1283, i1 %compare.low1335, i1 %compare.high.order1284, !dbg !25
  %bnk.n759 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 44) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 13, !dbg !25
  %compare.low1339 = icmp ugt i32 %compare.low.base1285, %compare.local1338, !dbg !25
  %compare.result1340 = select i1 %compare.high.equal1283, i1 %compare.low1339, i1 %compare.high.order1284, !dbg !25
  %bnk.n760 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 45) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 14, !dbg !25
  %compare.low1343 = icmp ugt i32 %compare.low.base1285, %compare.local1342, !dbg !25
  %compare.result1344 = select i1 %compare.high.equal1283, i1 %compare.low1343, i1 %compare.high.order1284, !dbg !25
  %bnk.n761 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 46) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 15, !dbg !25
  %compare.low1347 = icmp ugt i32 %compare.low.base1285, %compare.local1346, !dbg !25
  %compare.result1348 = select i1 %compare.high.equal1283, i1 %compare.low1347, i1 %compare.high.order1284, !dbg !25
  %bnk.n762 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 47) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 64, !dbg !25
  %compare.low1351 = icmp ugt i32 %compare.low.base1285, %compare.local1350, !dbg !25
  %compare.result1352 = select i1 %compare.high.equal1283, i1 %compare.low1351, i1 %compare.high.order1284, !dbg !25
  %bnk.n763 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 48) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 65, !dbg !25
  %compare.low1355 = icmp ugt i32 %compare.low.base1285, %compare.local1354, !dbg !25
  %compare.result1356 = select i1 %compare.high.equal1283, i1 %compare.low1355, i1 %compare.high.order1284, !dbg !25
  %bnk.n764 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 49) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 66, !dbg !25
  %compare.low1359 = icmp ugt i32 %compare.low.base1285, %compare.local1358, !dbg !25
  %compare.result1360 = select i1 %compare.high.equal1283, i1 %compare.low1359, i1 %compare.high.order1284, !dbg !25
  %bnk.n765 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 50) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 67, !dbg !25
  %compare.low1363 = icmp ugt i32 %compare.low.base1285, %compare.local1362, !dbg !25
  %compare.result1364 = select i1 %compare.high.equal1283, i1 %compare.low1363, i1 %compare.high.order1284, !dbg !25
  %bnk.n766 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 51) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 68, !dbg !25
  %compare.low1367 = icmp ugt i32 %compare.low.base1285, %compare.local1366, !dbg !25
  %compare.result1368 = select i1 %compare.high.equal1283, i1 %compare.low1367, i1 %compare.high.order1284, !dbg !25
  %bnk.n767 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 52) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 69, !dbg !25
  %compare.low1371 = icmp ugt i32 %compare.low.base1285, %compare.local1370, !dbg !25
  %compare.result1372 = select i1 %compare.high.equal1283, i1 %compare.low1371, i1 %compare.high.order1284, !dbg !25
  %bnk.n768 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 53) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 70, !dbg !25
  %compare.low1375 = icmp ugt i32 %compare.low.base1285, %compare.local1374, !dbg !25
  %compare.result1376 = select i1 %compare.high.equal1283, i1 %compare.low1375, i1 %compare.high.order1284, !dbg !25
  %bnk.n769 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 54) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 71, !dbg !25
  %compare.low1379 = icmp ugt i32 %compare.low.base1285, %compare.local1378, !dbg !25
  %compare.result1380 = select i1 %compare.high.equal1283, i1 %compare.low1379, i1 %compare.high.order1284, !dbg !25
  %bnk.n770 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 55) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 72, !dbg !25
  %compare.low1383 = icmp ugt i32 %compare.low.base1285, %compare.local1382, !dbg !25
  %compare.result1384 = select i1 %compare.high.equal1283, i1 %compare.low1383, i1 %compare.high.order1284, !dbg !25
  %bnk.n771 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 56) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 73, !dbg !25
  %compare.low1387 = icmp ugt i32 %compare.low.base1285, %compare.local1386, !dbg !25
  %compare.result1388 = select i1 %compare.high.equal1283, i1 %compare.low1387, i1 %compare.high.order1284, !dbg !25
  %bnk.n772 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 57) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 74, !dbg !25
  %compare.low1391 = icmp ugt i32 %compare.low.base1285, %compare.local1390, !dbg !25
  %compare.result1392 = select i1 %compare.high.equal1283, i1 %compare.low1391, i1 %compare.high.order1284, !dbg !25
  %bnk.n773 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 58) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1394 = or i32 %compare.local.base1393, 75, !dbg !25
  %compare.low1395 = icmp ugt i32 %compare.low.base1285, %compare.local1394, !dbg !25
  %compare.result1396 = select i1 %compare.high.equal1283, i1 %compare.low1395, i1 %compare.high.order1284, !dbg !25
  %bnk.n774 = select i1 %compare.result1396, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 59) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1398 = or i32 %compare.local.base1397, 76, !dbg !25
  %compare.low1399 = icmp ugt i32 %compare.low.base1285, %compare.local1398, !dbg !25
  %compare.result1400 = select i1 %compare.high.equal1283, i1 %compare.low1399, i1 %compare.high.order1284, !dbg !25
  %bnk.n775 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 60) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 77, !dbg !25
  %compare.low1403 = icmp ugt i32 %compare.low.base1285, %compare.local1402, !dbg !25
  %compare.result1404 = select i1 %compare.high.equal1283, i1 %compare.low1403, i1 %compare.high.order1284, !dbg !25
  %bnk.n776 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 61) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 78, !dbg !25
  %compare.low1407 = icmp ugt i32 %compare.low.base1285, %compare.local1406, !dbg !25
  %compare.result1408 = select i1 %compare.high.equal1283, i1 %compare.low1407, i1 %compare.high.order1284, !dbg !25
  %bnk.n777 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 62) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 79, !dbg !25
  %compare.low1411 = icmp ugt i32 %compare.low.base1285, %compare.local1410, !dbg !25
  %compare.result1412 = select i1 %compare.high.equal1283, i1 %compare.low1411, i1 %compare.high.order1284, !dbg !25
  %bnk.n778 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %bnk.n779 = insertelement <32 x float> poison, float %bnk.n712, i64 0, !dbg !14
  %bnk.n780 = insertelement <32 x float> %bnk.n779, float %bnk.n713, i64 1, !dbg !14
  %bnk.n781 = insertelement <32 x float> %bnk.n780, float %bnk.n714, i64 2, !dbg !14
  %bnk.n782 = insertelement <32 x float> %bnk.n781, float %bnk.n715, i64 3, !dbg !14
  %bnk.n783 = insertelement <32 x float> %bnk.n782, float %bnk.n716, i64 4, !dbg !14
  %bnk.n784 = insertelement <32 x float> %bnk.n783, float %bnk.n717, i64 5, !dbg !14
  %bnk.n785 = insertelement <32 x float> %bnk.n784, float %bnk.n718, i64 6, !dbg !14
  %bnk.n786 = insertelement <32 x float> %bnk.n785, float %bnk.n719, i64 7, !dbg !14
  %bnk.n787 = insertelement <32 x float> %bnk.n786, float %bnk.n720, i64 8, !dbg !14
  %bnk.n788 = insertelement <32 x float> %bnk.n787, float %bnk.n721, i64 9, !dbg !14
  %bnk.n789 = insertelement <32 x float> %bnk.n788, float %bnk.n722, i64 10, !dbg !14
  %bnk.n790 = insertelement <32 x float> %bnk.n789, float %bnk.n723, i64 11, !dbg !14
  %bnk.n791 = insertelement <32 x float> %bnk.n790, float %bnk.n724, i64 12, !dbg !14
  %bnk.n792 = insertelement <32 x float> %bnk.n791, float %bnk.n725, i64 13, !dbg !14
  %bnk.n793 = insertelement <32 x float> %bnk.n792, float %bnk.n726, i64 14, !dbg !14
  %bnk.n794 = insertelement <32 x float> %bnk.n793, float %bnk.n727, i64 15, !dbg !14
  %bnk.n795 = insertelement <32 x float> %bnk.n794, float %bnk.n728, i64 16, !dbg !14
  %bnk.n796 = insertelement <32 x float> %bnk.n795, float %bnk.n729, i64 17, !dbg !14
  %bnk.n797 = insertelement <32 x float> %bnk.n796, float %bnk.n730, i64 18, !dbg !14
  %bnk.n798 = insertelement <32 x float> %bnk.n797, float %bnk.n731, i64 19, !dbg !14
  %bnk.n799 = insertelement <32 x float> %bnk.n798, float %bnk.n732, i64 20, !dbg !14
  %bnk.n800 = insertelement <32 x float> %bnk.n799, float %bnk.n733, i64 21, !dbg !14
  %bnk.n801 = insertelement <32 x float> %bnk.n800, float %bnk.n734, i64 22, !dbg !14
  %bnk.n802 = insertelement <32 x float> %bnk.n801, float %bnk.n735, i64 23, !dbg !14
  %bnk.n803 = insertelement <32 x float> %bnk.n802, float %bnk.n736, i64 24, !dbg !14
  %bnk.n804 = insertelement <32 x float> %bnk.n803, float %bnk.n737, i64 25, !dbg !14
  %bnk.n805 = insertelement <32 x float> %bnk.n804, float %bnk.n738, i64 26, !dbg !14
  %bnk.n806 = insertelement <32 x float> %bnk.n805, float %bnk.n739, i64 27, !dbg !14
  %bnk.n807 = insertelement <32 x float> %bnk.n806, float %bnk.n740, i64 28, !dbg !14
  %bnk.n808 = insertelement <32 x float> %bnk.n807, float %bnk.n741, i64 29, !dbg !14
  %bnk.n809 = insertelement <32 x float> %bnk.n808, float %bnk.n742, i64 30, !dbg !14
  %bnk.n810 = insertelement <32 x float> %bnk.n809, float %bnk.n743, i64 31, !dbg !14
  %bnk.n811 = insertvalue [2 x <32 x float>] poison, <32 x float> %bnk.n810, 0, !dbg !14
  %bnk.n812 = insertelement <32 x float> poison, float %bnk.n747, i64 0, !dbg !14
  %bnk.n813 = insertelement <32 x float> %bnk.n812, float %bnk.n748, i64 1, !dbg !14
  %bnk.n814 = insertelement <32 x float> %bnk.n813, float %bnk.n749, i64 2, !dbg !14
  %bnk.n815 = insertelement <32 x float> %bnk.n814, float %bnk.n750, i64 3, !dbg !14
  %bnk.n816 = insertelement <32 x float> %bnk.n815, float %bnk.n751, i64 4, !dbg !14
  %bnk.n817 = insertelement <32 x float> %bnk.n816, float %bnk.n752, i64 5, !dbg !14
  %bnk.n818 = insertelement <32 x float> %bnk.n817, float %bnk.n753, i64 6, !dbg !14
  %bnk.n819 = insertelement <32 x float> %bnk.n818, float %bnk.n754, i64 7, !dbg !14
  %bnk.n820 = insertelement <32 x float> %bnk.n819, float %bnk.n755, i64 8, !dbg !14
  %bnk.n821 = insertelement <32 x float> %bnk.n820, float %bnk.n756, i64 9, !dbg !14
  %bnk.n822 = insertelement <32 x float> %bnk.n821, float %bnk.n757, i64 10, !dbg !14
  %bnk.n823 = insertelement <32 x float> %bnk.n822, float %bnk.n758, i64 11, !dbg !14
  %bnk.n824 = insertelement <32 x float> %bnk.n823, float %bnk.n759, i64 12, !dbg !14
  %bnk.n825 = insertelement <32 x float> %bnk.n824, float %bnk.n760, i64 13, !dbg !14
  %bnk.n826 = insertelement <32 x float> %bnk.n825, float %bnk.n761, i64 14, !dbg !14
  %bnk.n827 = insertelement <32 x float> %bnk.n826, float %bnk.n762, i64 15, !dbg !14
  %bnk.n828 = insertelement <32 x float> %bnk.n827, float %bnk.n763, i64 16, !dbg !14
  %bnk.n829 = insertelement <32 x float> %bnk.n828, float %bnk.n764, i64 17, !dbg !14
  %bnk.n830 = insertelement <32 x float> %bnk.n829, float %bnk.n765, i64 18, !dbg !14
  %bnk.n831 = insertelement <32 x float> %bnk.n830, float %bnk.n766, i64 19, !dbg !14
  %bnk.n832 = insertelement <32 x float> %bnk.n831, float %bnk.n767, i64 20, !dbg !14
  %bnk.n833 = insertelement <32 x float> %bnk.n832, float %bnk.n768, i64 21, !dbg !14
  %bnk.n834 = insertelement <32 x float> %bnk.n833, float %bnk.n769, i64 22, !dbg !14
  %bnk.n835 = insertelement <32 x float> %bnk.n834, float %bnk.n770, i64 23, !dbg !14
  %bnk.n836 = insertelement <32 x float> %bnk.n835, float %bnk.n771, i64 24, !dbg !14
  %bnk.n837 = insertelement <32 x float> %bnk.n836, float %bnk.n772, i64 25, !dbg !14
  %bnk.n838 = insertelement <32 x float> %bnk.n837, float %bnk.n773, i64 26, !dbg !14
  %bnk.n839 = insertelement <32 x float> %bnk.n838, float %bnk.n774, i64 27, !dbg !14
  %bnk.n840 = insertelement <32 x float> %bnk.n839, float %bnk.n775, i64 28, !dbg !14
  %bnk.n841 = insertelement <32 x float> %bnk.n840, float %bnk.n776, i64 29, !dbg !14
  %bnk.n842 = insertelement <32 x float> %bnk.n841, float %bnk.n777, i64 30, !dbg !14
  %bnk.n843 = insertelement <32 x float> %bnk.n842, float %bnk.n778, i64 31, !dbg !14
  %bnk.n844 = insertvalue [2 x <32 x float>] %bnk.n811, <32 x float> %bnk.n843, 1, !dbg !14
  br label %bnk.n845, !dbg !14

bnk.n845:                                              ; preds = %bnk.n700, %bnk.n701
  %bnk.n846 = phi [2 x <32 x float>] [ %bnk.n844, %bnk.n701 ], [ zeroinitializer, %bnk.n700 ], !dbg !14
  br label %bnk.n847, !dbg !14

bnk.n847:                                              ; preds = %bnk.n845
  %bnk.n848 = extractvalue [2 x <32 x float>] %bnk.n846, 0, !dbg !27
  %bnk.n849 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %bnk.n850 = fadd <4 x float> %bnk.n637, %bnk.n849, !dbg !27
  %bnk.n851 = extractelement <4 x float> %bnk.n850, i64 0, !dbg !28
  %bnk.n852 = call float @llvm.amdgcn.exp2.f32(float %bnk.n851), !dbg !28
  %bnk.n853 = insertelement <4 x float> poison, float %bnk.n852, i64 0, !dbg !30
  %bnk.n854 = extractelement <4 x float> %bnk.n850, i64 1, !dbg !28
  %bnk.n855 = call float @llvm.amdgcn.exp2.f32(float %bnk.n854), !dbg !28
  %bnk.n856 = insertelement <4 x float> %bnk.n853, float %bnk.n855, i64 1, !dbg !30
  %bnk.n857 = fadd float %bnk.n852, %bnk.n855, !dbg !29
  %bnk.n858 = extractelement <4 x float> %bnk.n850, i64 2, !dbg !28
  %bnk.n859 = call float @llvm.amdgcn.exp2.f32(float %bnk.n858), !dbg !28
  %bnk.n860 = insertelement <4 x float> %bnk.n856, float %bnk.n859, i64 2, !dbg !30
  %bnk.n861 = fadd float %bnk.n857, %bnk.n859, !dbg !29
  %bnk.n862 = extractelement <4 x float> %bnk.n850, i64 3, !dbg !28
  %bnk.n863 = call float @llvm.amdgcn.exp2.f32(float %bnk.n862), !dbg !28
  %bnk.n864 = insertelement <4 x float> %bnk.n860, float %bnk.n863, i64 3, !dbg !30
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n864) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n865 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !30
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n865) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n866 = fadd float %bnk.n861, %bnk.n863, !dbg !29
  %bnk.n867 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %bnk.n868 = fadd <4 x float> %bnk.n641, %bnk.n867, !dbg !27
  %bnk.n869 = extractelement <4 x float> %bnk.n868, i64 0, !dbg !28
  %bnk.n870 = call float @llvm.amdgcn.exp2.f32(float %bnk.n869), !dbg !28
  %bnk.n871 = insertelement <4 x float> poison, float %bnk.n870, i64 0, !dbg !30
  %bnk.n872 = fadd float %bnk.n866, %bnk.n870, !dbg !29
  %bnk.n873 = extractelement <4 x float> %bnk.n868, i64 1, !dbg !28
  %bnk.n874 = call float @llvm.amdgcn.exp2.f32(float %bnk.n873), !dbg !28
  %bnk.n875 = insertelement <4 x float> %bnk.n871, float %bnk.n874, i64 1, !dbg !30
  %bnk.n876 = fadd float %bnk.n872, %bnk.n874, !dbg !29
  %bnk.n877 = extractelement <4 x float> %bnk.n868, i64 2, !dbg !28
  %bnk.n878 = call float @llvm.amdgcn.exp2.f32(float %bnk.n877), !dbg !28
  %bnk.n879 = insertelement <4 x float> %bnk.n875, float %bnk.n878, i64 2, !dbg !30
  %bnk.n880 = fadd float %bnk.n876, %bnk.n878, !dbg !29
  %bnk.n881 = extractelement <4 x float> %bnk.n868, i64 3, !dbg !28
  %bnk.n882 = call float @llvm.amdgcn.exp2.f32(float %bnk.n881), !dbg !28
  %bnk.n883 = insertelement <4 x float> %bnk.n879, float %bnk.n882, i64 3, !dbg !30
  %math.result309 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n883) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n884 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !30
  %math.packed310 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n884) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n885 = fadd float %bnk.n880, %bnk.n882, !dbg !29
  %bnk.n886 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %bnk.n887 = fadd <4 x float> %bnk.n645, %bnk.n886, !dbg !27
  %bnk.n888 = extractelement <4 x float> %bnk.n887, i64 0, !dbg !28
  %bnk.n889 = call float @llvm.amdgcn.exp2.f32(float %bnk.n888), !dbg !28
  %bnk.n890 = insertelement <4 x float> poison, float %bnk.n889, i64 0, !dbg !30
  %bnk.n891 = fadd float %bnk.n885, %bnk.n889, !dbg !29
  %bnk.n892 = extractelement <4 x float> %bnk.n887, i64 1, !dbg !28
  %bnk.n893 = call float @llvm.amdgcn.exp2.f32(float %bnk.n892), !dbg !28
  %bnk.n894 = insertelement <4 x float> %bnk.n890, float %bnk.n893, i64 1, !dbg !30
  %bnk.n895 = fadd float %bnk.n891, %bnk.n893, !dbg !29
  %bnk.n896 = extractelement <4 x float> %bnk.n887, i64 2, !dbg !28
  %bnk.n897 = call float @llvm.amdgcn.exp2.f32(float %bnk.n896), !dbg !28
  %bnk.n898 = insertelement <4 x float> %bnk.n894, float %bnk.n897, i64 2, !dbg !30
  %bnk.n899 = fadd float %bnk.n895, %bnk.n897, !dbg !29
  %bnk.n900 = extractelement <4 x float> %bnk.n887, i64 3, !dbg !28
  %bnk.n901 = call float @llvm.amdgcn.exp2.f32(float %bnk.n900), !dbg !28
  %bnk.n902 = insertelement <4 x float> %bnk.n898, float %bnk.n901, i64 3, !dbg !30
  %math.result311 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n902) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n903 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !30
  %math.packed312 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n903) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n904 = fadd float %bnk.n899, %bnk.n901, !dbg !29
  %bnk.n905 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %bnk.n906 = fadd <4 x float> %bnk.n649, %bnk.n905, !dbg !27
  %bnk.n907 = extractelement <4 x float> %bnk.n906, i64 0, !dbg !28
  %bnk.n908 = call float @llvm.amdgcn.exp2.f32(float %bnk.n907), !dbg !28
  %bnk.n909 = insertelement <4 x float> poison, float %bnk.n908, i64 0, !dbg !30
  %bnk.n910 = fadd float %bnk.n904, %bnk.n908, !dbg !29
  %bnk.n911 = extractelement <4 x float> %bnk.n906, i64 1, !dbg !28
  %bnk.n912 = call float @llvm.amdgcn.exp2.f32(float %bnk.n911), !dbg !28
  %bnk.n913 = insertelement <4 x float> %bnk.n909, float %bnk.n912, i64 1, !dbg !30
  %bnk.n914 = fadd float %bnk.n910, %bnk.n912, !dbg !29
  %bnk.n915 = extractelement <4 x float> %bnk.n906, i64 2, !dbg !28
  %bnk.n916 = call float @llvm.amdgcn.exp2.f32(float %bnk.n915), !dbg !28
  %bnk.n917 = insertelement <4 x float> %bnk.n913, float %bnk.n916, i64 2, !dbg !30
  %bnk.n918 = fadd float %bnk.n914, %bnk.n916, !dbg !29
  %bnk.n919 = extractelement <4 x float> %bnk.n906, i64 3, !dbg !28
  %bnk.n920 = call float @llvm.amdgcn.exp2.f32(float %bnk.n919), !dbg !28
  %bnk.n921 = insertelement <4 x float> %bnk.n917, float %bnk.n920, i64 3, !dbg !30
  %math.result313 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n921) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n922 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !30
  %math.packed314 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n922) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n923 = fadd float %bnk.n918, %bnk.n920, !dbg !29
  %bnk.n924 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %bnk.n925 = fadd <4 x float> %bnk.n653, %bnk.n924, !dbg !27
  %bnk.n926 = extractelement <4 x float> %bnk.n925, i64 0, !dbg !28
  %bnk.n927 = call float @llvm.amdgcn.exp2.f32(float %bnk.n926), !dbg !28
  %bnk.n928 = insertelement <4 x float> poison, float %bnk.n927, i64 0, !dbg !30
  %bnk.n929 = fadd float %bnk.n923, %bnk.n927, !dbg !29
  %bnk.n930 = extractelement <4 x float> %bnk.n925, i64 1, !dbg !28
  %bnk.n931 = call float @llvm.amdgcn.exp2.f32(float %bnk.n930), !dbg !28
  %bnk.n932 = insertelement <4 x float> %bnk.n928, float %bnk.n931, i64 1, !dbg !30
  %bnk.n933 = fadd float %bnk.n929, %bnk.n931, !dbg !29
  %bnk.n934 = extractelement <4 x float> %bnk.n925, i64 2, !dbg !28
  %bnk.n935 = call float @llvm.amdgcn.exp2.f32(float %bnk.n934), !dbg !28
  %bnk.n936 = insertelement <4 x float> %bnk.n932, float %bnk.n935, i64 2, !dbg !30
  %bnk.n937 = fadd float %bnk.n933, %bnk.n935, !dbg !29
  %bnk.n938 = extractelement <4 x float> %bnk.n925, i64 3, !dbg !28
  %bnk.n939 = call float @llvm.amdgcn.exp2.f32(float %bnk.n938), !dbg !28
  %bnk.n940 = insertelement <4 x float> %bnk.n936, float %bnk.n939, i64 3, !dbg !30
  %math.result315 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n940) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n941 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !30
  %math.packed316 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n941) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n942 = fadd float %bnk.n937, %bnk.n939, !dbg !29
  %bnk.n943 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %bnk.n944 = fadd <4 x float> %bnk.n657, %bnk.n943, !dbg !27
  %bnk.n945 = extractelement <4 x float> %bnk.n944, i64 0, !dbg !28
  %bnk.n946 = call float @llvm.amdgcn.exp2.f32(float %bnk.n945), !dbg !28
  %bnk.n947 = insertelement <4 x float> poison, float %bnk.n946, i64 0, !dbg !30
  %bnk.n948 = fadd float %bnk.n942, %bnk.n946, !dbg !29
  %bnk.n949 = extractelement <4 x float> %bnk.n944, i64 1, !dbg !28
  %bnk.n950 = call float @llvm.amdgcn.exp2.f32(float %bnk.n949), !dbg !28
  %bnk.n951 = insertelement <4 x float> %bnk.n947, float %bnk.n950, i64 1, !dbg !30
  %bnk.n952 = fadd float %bnk.n948, %bnk.n950, !dbg !29
  %bnk.n953 = extractelement <4 x float> %bnk.n944, i64 2, !dbg !28
  %bnk.n954 = call float @llvm.amdgcn.exp2.f32(float %bnk.n953), !dbg !28
  %bnk.n955 = insertelement <4 x float> %bnk.n951, float %bnk.n954, i64 2, !dbg !30
  %bnk.n956 = fadd float %bnk.n952, %bnk.n954, !dbg !29
  %bnk.n957 = extractelement <4 x float> %bnk.n944, i64 3, !dbg !28
  %bnk.n958 = call float @llvm.amdgcn.exp2.f32(float %bnk.n957), !dbg !28
  %bnk.n959 = insertelement <4 x float> %bnk.n955, float %bnk.n958, i64 3, !dbg !30
  %math.result317 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n959) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n960 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !30
  %math.packed318 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n960) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n961 = fadd float %bnk.n956, %bnk.n958, !dbg !29
  %bnk.n962 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %bnk.n963 = fadd <4 x float> %bnk.n661, %bnk.n962, !dbg !27
  %bnk.n964 = extractelement <4 x float> %bnk.n963, i64 0, !dbg !28
  %bnk.n965 = call float @llvm.amdgcn.exp2.f32(float %bnk.n964), !dbg !28
  %bnk.n966 = insertelement <4 x float> poison, float %bnk.n965, i64 0, !dbg !30
  %bnk.n967 = fadd float %bnk.n961, %bnk.n965, !dbg !29
  %bnk.n968 = extractelement <4 x float> %bnk.n963, i64 1, !dbg !28
  %bnk.n969 = call float @llvm.amdgcn.exp2.f32(float %bnk.n968), !dbg !28
  %bnk.n970 = insertelement <4 x float> %bnk.n966, float %bnk.n969, i64 1, !dbg !30
  %bnk.n971 = fadd float %bnk.n967, %bnk.n969, !dbg !29
  %bnk.n972 = extractelement <4 x float> %bnk.n963, i64 2, !dbg !28
  %bnk.n973 = call float @llvm.amdgcn.exp2.f32(float %bnk.n972), !dbg !28
  %bnk.n974 = insertelement <4 x float> %bnk.n970, float %bnk.n973, i64 2, !dbg !30
  %bnk.n975 = fadd float %bnk.n971, %bnk.n973, !dbg !29
  %bnk.n976 = extractelement <4 x float> %bnk.n963, i64 3, !dbg !28
  %bnk.n977 = call float @llvm.amdgcn.exp2.f32(float %bnk.n976), !dbg !28
  %bnk.n978 = insertelement <4 x float> %bnk.n974, float %bnk.n977, i64 3, !dbg !30
  %math.result319 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n978) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n979 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !30
  %math.packed320 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n979) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n980 = fadd float %bnk.n975, %bnk.n977, !dbg !29
  %bnk.n981 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %bnk.n982 = fadd <4 x float> %bnk.n665, %bnk.n981, !dbg !27
  %bnk.n983 = extractelement <4 x float> %bnk.n982, i64 0, !dbg !28
  %bnk.n984 = call float @llvm.amdgcn.exp2.f32(float %bnk.n983), !dbg !28
  %bnk.n985 = insertelement <4 x float> poison, float %bnk.n984, i64 0, !dbg !30
  %bnk.n986 = fadd float %bnk.n980, %bnk.n984, !dbg !29
  %bnk.n987 = extractelement <4 x float> %bnk.n982, i64 1, !dbg !28
  %bnk.n988 = call float @llvm.amdgcn.exp2.f32(float %bnk.n987), !dbg !28
  %bnk.n989 = insertelement <4 x float> %bnk.n985, float %bnk.n988, i64 1, !dbg !30
  %bnk.n990 = fadd float %bnk.n986, %bnk.n988, !dbg !29
  %bnk.n991 = extractelement <4 x float> %bnk.n982, i64 2, !dbg !28
  %bnk.n992 = call float @llvm.amdgcn.exp2.f32(float %bnk.n991), !dbg !28
  %bnk.n993 = insertelement <4 x float> %bnk.n989, float %bnk.n992, i64 2, !dbg !30
  %bnk.n994 = fadd float %bnk.n990, %bnk.n992, !dbg !29
  %bnk.n995 = extractelement <4 x float> %bnk.n982, i64 3, !dbg !28
  %bnk.n996 = call float @llvm.amdgcn.exp2.f32(float %bnk.n995), !dbg !28
  %bnk.n997 = insertelement <4 x float> %bnk.n993, float %bnk.n996, i64 3, !dbg !30
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n997) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n998 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !30
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n998) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n999 = fadd float %bnk.n994, %bnk.n996, !dbg !29
  %bnk.n1000 = extractvalue [2 x <32 x float>] %bnk.n846, 1, !dbg !27
  %bnk.n1001 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %bnk.n1002 = fadd <4 x float> %bnk.n669, %bnk.n1001, !dbg !27
  %bnk.n1003 = extractelement <4 x float> %bnk.n1002, i64 0, !dbg !28
  %bnk.n1004 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1003), !dbg !28
  %bnk.n1005 = insertelement <4 x float> poison, float %bnk.n1004, i64 0, !dbg !30
  %bnk.n1006 = extractelement <4 x float> %bnk.n1002, i64 1, !dbg !28
  %bnk.n1007 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1006), !dbg !28
  %bnk.n1008 = insertelement <4 x float> %bnk.n1005, float %bnk.n1007, i64 1, !dbg !30
  %bnk.n1009 = fadd float %bnk.n1004, %bnk.n1007, !dbg !29
  %bnk.n1010 = extractelement <4 x float> %bnk.n1002, i64 2, !dbg !28
  %bnk.n1011 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1010), !dbg !28
  %bnk.n1012 = insertelement <4 x float> %bnk.n1008, float %bnk.n1011, i64 2, !dbg !30
  %bnk.n1013 = fadd float %bnk.n1009, %bnk.n1011, !dbg !29
  %bnk.n1014 = extractelement <4 x float> %bnk.n1002, i64 3, !dbg !28
  %bnk.n1015 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1014), !dbg !28
  %bnk.n1016 = insertelement <4 x float> %bnk.n1012, float %bnk.n1015, i64 3, !dbg !30
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1016) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n1017 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !30
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1017) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n1018 = fadd float %bnk.n1013, %bnk.n1015, !dbg !29
  %bnk.n1019 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %bnk.n1020 = fadd <4 x float> %bnk.n673, %bnk.n1019, !dbg !27
  %bnk.n1021 = extractelement <4 x float> %bnk.n1020, i64 0, !dbg !28
  %bnk.n1022 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1021), !dbg !28
  %bnk.n1023 = insertelement <4 x float> poison, float %bnk.n1022, i64 0, !dbg !30
  %bnk.n1024 = fadd float %bnk.n1018, %bnk.n1022, !dbg !29
  %bnk.n1025 = extractelement <4 x float> %bnk.n1020, i64 1, !dbg !28
  %bnk.n1026 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1025), !dbg !28
  %bnk.n1027 = insertelement <4 x float> %bnk.n1023, float %bnk.n1026, i64 1, !dbg !30
  %bnk.n1028 = fadd float %bnk.n1024, %bnk.n1026, !dbg !29
  %bnk.n1029 = extractelement <4 x float> %bnk.n1020, i64 2, !dbg !28
  %bnk.n1030 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1029), !dbg !28
  %bnk.n1031 = insertelement <4 x float> %bnk.n1027, float %bnk.n1030, i64 2, !dbg !30
  %bnk.n1032 = fadd float %bnk.n1028, %bnk.n1030, !dbg !29
  %bnk.n1033 = extractelement <4 x float> %bnk.n1020, i64 3, !dbg !28
  %bnk.n1034 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1033), !dbg !28
  %bnk.n1035 = insertelement <4 x float> %bnk.n1031, float %bnk.n1034, i64 3, !dbg !30
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1035) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n1036 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !30
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1036) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n1037 = fadd float %bnk.n1032, %bnk.n1034, !dbg !29
  %bnk.n1038 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %bnk.n1039 = fadd <4 x float> %bnk.n677, %bnk.n1038, !dbg !27
  %bnk.n1040 = extractelement <4 x float> %bnk.n1039, i64 0, !dbg !28
  %bnk.n1041 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1040), !dbg !28
  %bnk.n1042 = insertelement <4 x float> poison, float %bnk.n1041, i64 0, !dbg !30
  %bnk.n1043 = fadd float %bnk.n1037, %bnk.n1041, !dbg !29
  %bnk.n1044 = extractelement <4 x float> %bnk.n1039, i64 1, !dbg !28
  %bnk.n1045 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1044), !dbg !28
  %bnk.n1046 = insertelement <4 x float> %bnk.n1042, float %bnk.n1045, i64 1, !dbg !30
  %bnk.n1047 = fadd float %bnk.n1043, %bnk.n1045, !dbg !29
  %bnk.n1048 = extractelement <4 x float> %bnk.n1039, i64 2, !dbg !28
  %bnk.n1049 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1048), !dbg !28
  %bnk.n1050 = insertelement <4 x float> %bnk.n1046, float %bnk.n1049, i64 2, !dbg !30
  %bnk.n1051 = fadd float %bnk.n1047, %bnk.n1049, !dbg !29
  %bnk.n1052 = extractelement <4 x float> %bnk.n1039, i64 3, !dbg !28
  %bnk.n1053 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1052), !dbg !28
  %bnk.n1054 = insertelement <4 x float> %bnk.n1050, float %bnk.n1053, i64 3, !dbg !30
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1054) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n1055 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !30
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1055) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n1056 = fadd float %bnk.n1051, %bnk.n1053, !dbg !29
  %bnk.n1057 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %bnk.n1058 = fadd <4 x float> %bnk.n681, %bnk.n1057, !dbg !27
  %bnk.n1059 = extractelement <4 x float> %bnk.n1058, i64 0, !dbg !28
  %bnk.n1060 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1059), !dbg !28
  %bnk.n1061 = insertelement <4 x float> poison, float %bnk.n1060, i64 0, !dbg !30
  %bnk.n1062 = fadd float %bnk.n1056, %bnk.n1060, !dbg !29
  %bnk.n1063 = extractelement <4 x float> %bnk.n1058, i64 1, !dbg !28
  %bnk.n1064 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1063), !dbg !28
  %bnk.n1065 = insertelement <4 x float> %bnk.n1061, float %bnk.n1064, i64 1, !dbg !30
  %bnk.n1066 = fadd float %bnk.n1062, %bnk.n1064, !dbg !29
  %bnk.n1067 = extractelement <4 x float> %bnk.n1058, i64 2, !dbg !28
  %bnk.n1068 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1067), !dbg !28
  %bnk.n1069 = insertelement <4 x float> %bnk.n1065, float %bnk.n1068, i64 2, !dbg !30
  %bnk.n1070 = fadd float %bnk.n1066, %bnk.n1068, !dbg !29
  %bnk.n1071 = extractelement <4 x float> %bnk.n1058, i64 3, !dbg !28
  %bnk.n1072 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1071), !dbg !28
  %bnk.n1073 = insertelement <4 x float> %bnk.n1069, float %bnk.n1072, i64 3, !dbg !30
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1073) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n1074 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !30
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1074) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n1075 = fadd float %bnk.n1070, %bnk.n1072, !dbg !29
  %bnk.n1076 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %bnk.n1077 = fadd <4 x float> %bnk.n685, %bnk.n1076, !dbg !27
  %bnk.n1078 = extractelement <4 x float> %bnk.n1077, i64 0, !dbg !28
  %bnk.n1079 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1078), !dbg !28
  %bnk.n1080 = insertelement <4 x float> poison, float %bnk.n1079, i64 0, !dbg !30
  %bnk.n1081 = fadd float %bnk.n1075, %bnk.n1079, !dbg !29
  %bnk.n1082 = extractelement <4 x float> %bnk.n1077, i64 1, !dbg !28
  %bnk.n1083 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1082), !dbg !28
  %bnk.n1084 = insertelement <4 x float> %bnk.n1080, float %bnk.n1083, i64 1, !dbg !30
  %bnk.n1085 = fadd float %bnk.n1081, %bnk.n1083, !dbg !29
  %bnk.n1086 = extractelement <4 x float> %bnk.n1077, i64 2, !dbg !28
  %bnk.n1087 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1086), !dbg !28
  %bnk.n1088 = insertelement <4 x float> %bnk.n1084, float %bnk.n1087, i64 2, !dbg !30
  %bnk.n1089 = fadd float %bnk.n1085, %bnk.n1087, !dbg !29
  %bnk.n1090 = extractelement <4 x float> %bnk.n1077, i64 3, !dbg !28
  %bnk.n1091 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1090), !dbg !28
  %bnk.n1092 = insertelement <4 x float> %bnk.n1088, float %bnk.n1091, i64 3, !dbg !30
  %math.result331 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1092) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n1093 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !30
  %math.packed332 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1093) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n1094 = fadd float %bnk.n1089, %bnk.n1091, !dbg !29
  %bnk.n1095 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %bnk.n1096 = fadd <4 x float> %bnk.n689, %bnk.n1095, !dbg !27
  %bnk.n1097 = extractelement <4 x float> %bnk.n1096, i64 0, !dbg !28
  %bnk.n1098 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1097), !dbg !28
  %bnk.n1099 = insertelement <4 x float> poison, float %bnk.n1098, i64 0, !dbg !30
  %bnk.n1100 = fadd float %bnk.n1094, %bnk.n1098, !dbg !29
  %bnk.n1101 = extractelement <4 x float> %bnk.n1096, i64 1, !dbg !28
  %bnk.n1102 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1101), !dbg !28
  %bnk.n1103 = insertelement <4 x float> %bnk.n1099, float %bnk.n1102, i64 1, !dbg !30
  %bnk.n1104 = fadd float %bnk.n1100, %bnk.n1102, !dbg !29
  %bnk.n1105 = extractelement <4 x float> %bnk.n1096, i64 2, !dbg !28
  %bnk.n1106 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1105), !dbg !28
  %bnk.n1107 = insertelement <4 x float> %bnk.n1103, float %bnk.n1106, i64 2, !dbg !30
  %bnk.n1108 = fadd float %bnk.n1104, %bnk.n1106, !dbg !29
  %bnk.n1109 = extractelement <4 x float> %bnk.n1096, i64 3, !dbg !28
  %bnk.n1110 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1109), !dbg !28
  %bnk.n1111 = insertelement <4 x float> %bnk.n1107, float %bnk.n1110, i64 3, !dbg !30
  %math.result333 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1111) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n1112 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !30
  %math.packed334 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1112) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n1113 = fadd float %bnk.n1108, %bnk.n1110, !dbg !29
  %bnk.n1114 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %bnk.n1115 = fadd <4 x float> %bnk.n693, %bnk.n1114, !dbg !27
  %bnk.n1116 = extractelement <4 x float> %bnk.n1115, i64 0, !dbg !28
  %bnk.n1117 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1116), !dbg !28
  %bnk.n1118 = insertelement <4 x float> poison, float %bnk.n1117, i64 0, !dbg !30
  %bnk.n1119 = fadd float %bnk.n1113, %bnk.n1117, !dbg !29
  %bnk.n1120 = extractelement <4 x float> %bnk.n1115, i64 1, !dbg !28
  %bnk.n1121 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1120), !dbg !28
  %bnk.n1122 = insertelement <4 x float> %bnk.n1118, float %bnk.n1121, i64 1, !dbg !30
  %bnk.n1123 = fadd float %bnk.n1119, %bnk.n1121, !dbg !29
  %bnk.n1124 = extractelement <4 x float> %bnk.n1115, i64 2, !dbg !28
  %bnk.n1125 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1124), !dbg !28
  %bnk.n1126 = insertelement <4 x float> %bnk.n1122, float %bnk.n1125, i64 2, !dbg !30
  %bnk.n1127 = fadd float %bnk.n1123, %bnk.n1125, !dbg !29
  %bnk.n1128 = extractelement <4 x float> %bnk.n1115, i64 3, !dbg !28
  %bnk.n1129 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1128), !dbg !28
  %bnk.n1130 = insertelement <4 x float> %bnk.n1126, float %bnk.n1129, i64 3, !dbg !30
  %math.result335 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1130) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n1131 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !30
  %math.packed336 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1131) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n1132 = fadd float %bnk.n1127, %bnk.n1129, !dbg !29
  %bnk.n1133 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %bnk.n1134 = fadd <4 x float> %bnk.n698, %bnk.n1133, !dbg !27
  %bnk.n1135 = extractelement <4 x float> %bnk.n1134, i64 0, !dbg !28
  %bnk.n1136 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1135), !dbg !28
  %bnk.n1137 = insertelement <4 x float> poison, float %bnk.n1136, i64 0, !dbg !30
  %bnk.n1138 = fadd float %bnk.n1132, %bnk.n1136, !dbg !29
  %bnk.n1139 = extractelement <4 x float> %bnk.n1134, i64 1, !dbg !28
  %bnk.n1140 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1139), !dbg !28
  %bnk.n1141 = insertelement <4 x float> %bnk.n1137, float %bnk.n1140, i64 1, !dbg !30
  %bnk.n1142 = fadd float %bnk.n1138, %bnk.n1140, !dbg !29
  %bnk.n1143 = extractelement <4 x float> %bnk.n1134, i64 2, !dbg !28
  %bnk.n1144 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1143), !dbg !28
  %bnk.n1145 = insertelement <4 x float> %bnk.n1141, float %bnk.n1144, i64 2, !dbg !30
  %bnk.n1146 = fadd float %bnk.n1142, %bnk.n1144, !dbg !29
  %bnk.n1147 = extractelement <4 x float> %bnk.n1134, i64 3, !dbg !28
  %bnk.n1148 = call float @llvm.amdgcn.exp2.f32(float %bnk.n1147), !dbg !28
  %bnk.n1149 = insertelement <4 x float> %bnk.n1145, float %bnk.n1148, i64 3, !dbg !30
  %math.result337 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1149) #9, !dbg !30, !frisk.register_lifetimes !1
  %bnk.n1150 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !30
  %math.packed338 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1150) #9, !dbg !29, !frisk.register_lifetimes !1
  %bnk.n1151 = fadd float %bnk.n1146, %bnk.n1148, !dbg !29
  %bnk.n1152 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !29
  %bnk.n1153 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %bnk.n1152), !dbg !29
  %bnk.n1154 = add nuw nsw i32 %bnk.n1153, 64, !dbg !29
  %bnk.n1155 = xor i32 %bnk.n1153, 16, !dbg !29
  %bnk.n1156 = and i32 %bnk.n1154, 192, !dbg !29
  %bnk.n1157 = icmp samesign ult i32 %bnk.n1155, %bnk.n1156, !dbg !29
  %bnk.n1158 = select i1 %bnk.n1157, i32 %bnk.n1155, i32 %bnk.n1153, !dbg !29
  %bnk.n1159 = shl nuw nsw i32 %bnk.n1158, 2, !dbg !29
  %bnk.n1160 = bitcast float %bnk.n999 to i32, !dbg !29
  %bnk.n1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1159, i32 %bnk.n1160), !dbg !29
  %bnk.n1162 = bitcast i32 %bnk.n1161 to float, !dbg !29
  %bnk.n1163 = fadd float %bnk.n999, %bnk.n1162, !dbg !29
  %bnk.n1164 = xor i32 %bnk.n1153, 32, !dbg !29
  %bnk.n1165 = icmp samesign ult i32 %bnk.n1164, %bnk.n1156, !dbg !29
  %bnk.n1166 = select i1 %bnk.n1165, i32 %bnk.n1164, i32 %bnk.n1153, !dbg !29
  %bnk.n1167 = shl nuw nsw i32 %bnk.n1166, 2, !dbg !29
  %bnk.n1168 = bitcast float %bnk.n1163 to i32, !dbg !29
  %bnk.n1169 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1167, i32 %bnk.n1168), !dbg !29
  %bnk.n1170 = bitcast i32 %bnk.n1169 to float, !dbg !29
  %bnk.n1171 = fadd float %bnk.n1163, %bnk.n1170, !dbg !29
  %bnk.n1172 = extractvalue [2 x <1 x float>] %bnk.n207, 0, !dbg !29
  %bnk.n1173 = extractelement <1 x float> %bnk.n1172, i64 0, !dbg !29
  %bnk.n1174 = fadd float %bnk.n1171, %bnk.n1173, !dbg !29
  %bnk.n1175 = insertelement <1 x float> poison, float %bnk.n1174, i64 0, !dbg !13
  %bnk.n1176 = insertvalue [2 x <1 x float>] poison, <1 x float> %bnk.n1175, 0, !dbg !13
  %bnk.n1177 = bitcast float %bnk.n1151 to i32, !dbg !29
  %bnk.n1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1159, i32 %bnk.n1177), !dbg !29
  %bnk.n1179 = bitcast i32 %bnk.n1178 to float, !dbg !29
  %bnk.n1180 = fadd float %bnk.n1151, %bnk.n1179, !dbg !29
  %bnk.n1181 = bitcast float %bnk.n1180 to i32, !dbg !29
  %bnk.n1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1167, i32 %bnk.n1181), !dbg !29
  %bnk.n1183 = bitcast i32 %bnk.n1182 to float, !dbg !29
  %bnk.n1184 = fadd float %bnk.n1180, %bnk.n1183, !dbg !29
  %bnk.n1185 = extractvalue [2 x <1 x float>] %bnk.n207, 1, !dbg !29
  %bnk.n1186 = extractelement <1 x float> %bnk.n1185, i64 0, !dbg !29
  %bnk.n1187 = fadd float %bnk.n1184, %bnk.n1186, !dbg !29
  %bnk.n1188 = insertelement <1 x float> poison, float %bnk.n1187, i64 0, !dbg !13
  %bnk.n1189 = insertvalue [2 x <1 x float>] %bnk.n1176, <1 x float> %bnk.n1188, 1, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread864 = call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #9, !dbg !15, !range !34, !frisk.register_lifetimes !1
  %address.local865 = shl nuw nsw i32 %address.thread864, 3, !dbg !15
  %address.local866 = and i32 %address.local865, 120, !dbg !15
  %address.local867 = shl nuw nsw i32 %address.thread864, 5, !dbg !15
  %address.local868 = and i32 %address.local867, 1536, !dbg !15
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !15
  %address.local870 = zext nneg i32 %address.local869 to i64, !dbg !15
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !15
  %bnk.n1190 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !15
  %bnk.n1191 = bitcast <4 x i32> %bnk.n1190 to <8 x half>, !dbg !15
  %address.local872 = zext nneg i32 %address.thread864 to i64, !dbg !8
  %address.local873 = and i64 %address.local872, 60, !dbg !15
  %address.local874 = add nuw nsw i64 %address.local873, %address.local872, !dbg !15
  %address.local875 = shl nuw nsw i64 %address.local874, 3, !dbg !15
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local877 = and i64 %address.local872, 63, !dbg !11
  %address.local878 = lshr i64 %address.local877, 4, !dbg !15
  %address.local879 = mul nuw nsw i64 %address.local878, 384, !dbg !15
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) %address.local876, i64 %address.local879, !dbg !15
  %address.local881 = lshr i64 %address.local872, 2, !dbg !18
  %address.local882 = mul nsw i64 %address.local881, -32, !dbg !15
  %address.local883 = getelementptr [2 x i8], ptr addrspace(3) %address.local880, i64 %address.local882, !dbg !15
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 256, !dbg !15
  %bnk.n1192 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !15
  %bnk.n1193 = bitcast <4 x i32> %bnk.n1192 to <8 x half>, !dbg !15
  %address.local885 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local886 = getelementptr [2 x i8], ptr addrspace(3) %address.local885, i64 %address.local879, !dbg !15
  %address.local887 = getelementptr [2 x i8], ptr addrspace(3) %address.local886, i64 %address.local882, !dbg !15
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local887, i64 512, !dbg !15
  %bnk.n1194 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !15
  %bnk.n1195 = bitcast <4 x i32> %bnk.n1194 to <8 x half>, !dbg !15
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local890 = getelementptr [2 x i8], ptr addrspace(3) %address.local889, i64 %address.local879, !dbg !15
  %address.local891 = getelementptr [2 x i8], ptr addrspace(3) %address.local890, i64 %address.local882, !dbg !15
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local891, i64 768, !dbg !15
  %bnk.n1196 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !15
  %bnk.n1197 = bitcast <4 x i32> %bnk.n1196 to <8 x half>, !dbg !15
  %address.local893 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !15
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local893, i64 4096, !dbg !15
  %bnk.n1198 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !15
  %bnk.n1199 = bitcast <4 x i32> %bnk.n1198 to <8 x half>, !dbg !15
  %address.local895 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local896 = getelementptr [2 x i8], ptr addrspace(3) %address.local895, i64 %address.local879, !dbg !15
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) %address.local896, i64 %address.local882, !dbg !15
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local897, i64 4352, !dbg !15
  %bnk.n1200 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !15
  %bnk.n1201 = bitcast <4 x i32> %bnk.n1200 to <8 x half>, !dbg !15
  %address.local899 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local900 = getelementptr [2 x i8], ptr addrspace(3) %address.local899, i64 %address.local879, !dbg !15
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) %address.local900, i64 %address.local882, !dbg !15
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local901, i64 4608, !dbg !15
  %bnk.n1202 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !15
  %bnk.n1203 = bitcast <4 x i32> %bnk.n1202 to <8 x half>, !dbg !15
  %address.local903 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) %address.local903, i64 %address.local879, !dbg !15
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) %address.local904, i64 %address.local882, !dbg !15
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 4864, !dbg !15
  %bnk.n1204 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !15
  %bnk.n1205 = bitcast <4 x i32> %bnk.n1204 to <8 x half>, !dbg !15
  %bnk.n1206 = add nuw nsw i64 %bnk.n216, %bnk.n16, !dbg !19
  %bnk.n1207 = shl i64 %bnk.n1206, 19, !dbg !19
  %bnk.n1208 = and i64 %bnk.n1207, 16252928, !dbg !19
  %bnk.n1209 = add i64 %bnk.n1208, %bnk.n222, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %bnk.n1210 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %bnk.n1209, !dbg !19
  %bnk.n1211 = getelementptr i8, ptr addrspace(1) %bnk.n1210, i64 64, !dbg !19
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !8
  %address.local908 = and i64 %address.local907, 1536, !dbg !8
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !8
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !8
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !8
  %bnk.n1212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !19
  %bnk.n1213 = zext i32 %bnk.n1212 to i64, !dbg !19
  %bnk.n1214 = inttoptr i64 %bnk.n1213 to ptr addrspace(3), !dbg !19
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !19
  %address.local913 = and i64 %address.local912, 8064, !dbg !19
  %address.local914 = shl nuw nsw i64 %address.local872, 3, !dbg !19
  %address.local915 = and i64 %address.local914, 24, !dbg !19
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !19
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n1211, i64 %address.local916, !dbg !19
  %bnk.n1215 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %bnk.n1216 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !19
  %bnk.n1217 = sub i64 %bnk.n1216, %bnk.n1215, !dbg !19
  %dma.buffer.offset1558 = trunc i64 %bnk.n1217 to i32, !dbg !19
  %bnk.n1218 = trunc i64 %bnk.n1215 to i32, !dbg !19
  %bnk.n1219 = insertelement <4 x i32> poison, i32 %bnk.n1218, i64 0, !dbg !19
  %bnk.n1220 = lshr i64 %bnk.n1215, 32, !dbg !19
  %bnk.n1221 = trunc i64 %bnk.n1220 to i32, !dbg !19
  %bnk.n1222 = insertelement <4 x i32> %bnk.n1219, i32 %bnk.n1221, i64 1, !dbg !19
  %bnk.n1223 = insertelement <4 x i32> %bnk.n1222, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1559 = insertelement <4 x i32> %bnk.n1223, i32 131072, i64 3, !dbg !19
  %bnk.n1224 = ptrtoint ptr addrspace(3) %bnk.n1214 to i32, !dbg !19
  %dma.buffer.lds1560 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n1224), !dbg !19
  %bnk.n1225 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1559) #10, !dbg !19
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !8
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !8
  %address.local920 = and i64 %address.local919, 3584, !dbg !8
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !8
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !8
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !8
  %bnk.n1226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !19
  %bnk.n1227 = zext i32 %bnk.n1226 to i64, !dbg !19
  %bnk.n1228 = inttoptr i64 %bnk.n1227 to ptr addrspace(3), !dbg !19
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !19
  %address.local925 = and i64 %address.local924, 16256, !dbg !19
  %address.local926 = shl nuw nsw i64 %address.local872, 3, !dbg !19
  %address.local927 = and i64 %address.local926, 24, !dbg !19
  %address.local928 = or disjoint i64 %address.local925, %address.local927, !dbg !19
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n1211, i64 %address.local928, !dbg !19
  %bnk.n1229 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %bnk.n1230 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !19
  %bnk.n1231 = sub i64 %bnk.n1230, %bnk.n1229, !dbg !19
  %dma.buffer.offset1555 = trunc i64 %bnk.n1231 to i32, !dbg !19
  %bnk.n1232 = trunc i64 %bnk.n1229 to i32, !dbg !19
  %bnk.n1233 = insertelement <4 x i32> poison, i32 %bnk.n1232, i64 0, !dbg !19
  %bnk.n1234 = lshr i64 %bnk.n1229, 32, !dbg !19
  %bnk.n1235 = trunc i64 %bnk.n1234 to i32, !dbg !19
  %bnk.n1236 = insertelement <4 x i32> %bnk.n1233, i32 %bnk.n1235, i64 1, !dbg !19
  %bnk.n1237 = insertelement <4 x i32> %bnk.n1236, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1556 = insertelement <4 x i32> %bnk.n1237, i32 131072, i64 3, !dbg !19
  %bnk.n1238 = ptrtoint ptr addrspace(3) %bnk.n1228 to i32, !dbg !19
  %dma.buffer.lds1557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n1238), !dbg !19
  %bnk.n1239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1556) #10, !dbg !19
  %bnk.n1240 = shufflevector <8 x half> %bnk.n1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1241 = extractvalue [2 x <8 x float>] %bnk.n206, 0, !dbg !31
  %bnk.n1242 = shufflevector <8 x float> %bnk.n1241, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %bnk.n1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1240, <4 x float> %bnk.n1242) #9, !dbg !15
  %bnk.n1244 = shufflevector <8 x half> %bnk.n1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1244, <4 x float> %bnk.n1243) #9, !dbg !15
  %bnk.n1246 = shufflevector <8 x half> %bnk.n1195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1246, <4 x float> %bnk.n1245) #9, !dbg !15
  %bnk.n1248 = shufflevector <8 x half> %bnk.n1197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1248, <4 x float> %bnk.n1247) #9, !dbg !15
  %bnk.n1250 = shufflevector <8 x half> %bnk.n1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1250, <4 x float> %bnk.n1249) #9, !dbg !15
  %bnk.n1252 = shufflevector <8 x half> %bnk.n1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1252, <4 x float> %bnk.n1251) #9, !dbg !15
  %bnk.n1254 = shufflevector <8 x half> %bnk.n1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1254, <4 x float> %bnk.n1253) #9, !dbg !15
  %bnk.n1256 = shufflevector <8 x half> %bnk.n1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1256, <4 x float> %bnk.n1255) #9, !dbg !15
  %bnk.n1258 = shufflevector <8 x half> %bnk.n1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1259 = shufflevector <8 x float> %bnk.n1241, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %bnk.n1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1258, <4 x float> %bnk.n1259) #9, !dbg !15
  %bnk.n1261 = shufflevector <8 x half> %bnk.n1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1261, <4 x float> %bnk.n1260) #9, !dbg !15
  %bnk.n1263 = shufflevector <8 x half> %bnk.n1195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1263, <4 x float> %bnk.n1262) #9, !dbg !15
  %bnk.n1265 = shufflevector <8 x half> %bnk.n1197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1265, <4 x float> %bnk.n1264) #9, !dbg !15
  %bnk.n1267 = shufflevector <8 x half> %bnk.n1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1267, <4 x float> %bnk.n1266) #9, !dbg !15
  %bnk.n1269 = shufflevector <8 x half> %bnk.n1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1269, <4 x float> %bnk.n1268) #9, !dbg !15
  %bnk.n1271 = shufflevector <8 x half> %bnk.n1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1271, <4 x float> %bnk.n1270) #9, !dbg !15
  %bnk.n1273 = shufflevector <8 x half> %bnk.n1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1273, <4 x float> %bnk.n1272) #9, !dbg !15
  %bnk.n1275 = extractvalue [2 x <8 x float>] %bnk.n206, 1, !dbg !31
  %bnk.n1276 = shufflevector <8 x float> %bnk.n1275, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %bnk.n1277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1240, <4 x float> %bnk.n1276) #9, !dbg !15
  %bnk.n1278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1244, <4 x float> %bnk.n1277) #9, !dbg !15
  %bnk.n1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1246, <4 x float> %bnk.n1278) #9, !dbg !15
  %bnk.n1280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1248, <4 x float> %bnk.n1279) #9, !dbg !15
  %bnk.n1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1250, <4 x float> %bnk.n1280) #9, !dbg !15
  %bnk.n1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1252, <4 x float> %bnk.n1281) #9, !dbg !15
  %bnk.n1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1254, <4 x float> %bnk.n1282) #9, !dbg !15
  %bnk.n1284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1256, <4 x float> %bnk.n1283) #9, !dbg !15
  %bnk.n1285 = shufflevector <8 x float> %bnk.n1275, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %bnk.n1286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1258, <4 x float> %bnk.n1285) #9, !dbg !15
  %bnk.n1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1261, <4 x float> %bnk.n1286) #9, !dbg !15
  %bnk.n1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1263, <4 x float> %bnk.n1287) #9, !dbg !15
  %bnk.n1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1265, <4 x float> %bnk.n1288) #9, !dbg !15
  %bnk.n1290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1267, <4 x float> %bnk.n1289) #9, !dbg !15
  %bnk.n1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1269, <4 x float> %bnk.n1290) #9, !dbg !15
  %bnk.n1292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1271, <4 x float> %bnk.n1291) #9, !dbg !15
  %bnk.n1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1273, <4 x float> %bnk.n1292) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %bnk.n1294 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !15
  %bnk.n1295 = bitcast <4 x i32> %bnk.n1294 to <8 x half>, !dbg !15
  %bnk.n1296 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !15
  %bnk.n1297 = bitcast <4 x i32> %bnk.n1296 to <8 x half>, !dbg !15
  %bnk.n1298 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !15
  %bnk.n1299 = bitcast <4 x i32> %bnk.n1298 to <8 x half>, !dbg !15
  %bnk.n1300 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !15
  %bnk.n1301 = bitcast <4 x i32> %bnk.n1300 to <8 x half>, !dbg !15
  %bnk.n1302 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !15
  %bnk.n1303 = bitcast <4 x i32> %bnk.n1302 to <8 x half>, !dbg !15
  %bnk.n1304 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !15
  %bnk.n1305 = bitcast <4 x i32> %bnk.n1304 to <8 x half>, !dbg !15
  %bnk.n1306 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !15
  %bnk.n1307 = bitcast <4 x i32> %bnk.n1306 to <8 x half>, !dbg !15
  %bnk.n1308 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !15
  %bnk.n1309 = bitcast <4 x i32> %bnk.n1308 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %bnk.n1310 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %bnk.n1209, !dbg !19
  %bnk.n1311 = getelementptr i8, ptr addrspace(1) %bnk.n1310, i64 128, !dbg !19
  %bnk.n1312 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !19
  %bnk.n1313 = zext i32 %bnk.n1312 to i64, !dbg !19
  %bnk.n1314 = inttoptr i64 %bnk.n1313 to ptr addrspace(3), !dbg !19
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n1311, i64 %address.local916, !dbg !19
  %bnk.n1315 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %bnk.n1316 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !19
  %bnk.n1317 = sub i64 %bnk.n1316, %bnk.n1315, !dbg !19
  %dma.buffer.offset1552 = trunc i64 %bnk.n1317 to i32, !dbg !19
  %bnk.n1318 = trunc i64 %bnk.n1315 to i32, !dbg !19
  %bnk.n1319 = insertelement <4 x i32> poison, i32 %bnk.n1318, i64 0, !dbg !19
  %bnk.n1320 = lshr i64 %bnk.n1315, 32, !dbg !19
  %bnk.n1321 = trunc i64 %bnk.n1320 to i32, !dbg !19
  %bnk.n1322 = insertelement <4 x i32> %bnk.n1319, i32 %bnk.n1321, i64 1, !dbg !19
  %bnk.n1323 = insertelement <4 x i32> %bnk.n1322, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1553 = insertelement <4 x i32> %bnk.n1323, i32 131072, i64 3, !dbg !19
  %bnk.n1324 = ptrtoint ptr addrspace(3) %bnk.n1314 to i32, !dbg !19
  %dma.buffer.lds1554 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n1324), !dbg !19
  %bnk.n1325 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1554, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1553) #10, !dbg !19
  %bnk.n1326 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !19
  %bnk.n1327 = zext i32 %bnk.n1326 to i64, !dbg !19
  %bnk.n1328 = inttoptr i64 %bnk.n1327 to ptr addrspace(3), !dbg !19
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n1311, i64 %address.local928, !dbg !19
  %bnk.n1329 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %bnk.n1330 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !19
  %bnk.n1331 = sub i64 %bnk.n1330, %bnk.n1329, !dbg !19
  %dma.buffer.offset1549 = trunc i64 %bnk.n1331 to i32, !dbg !19
  %bnk.n1332 = trunc i64 %bnk.n1329 to i32, !dbg !19
  %bnk.n1333 = insertelement <4 x i32> poison, i32 %bnk.n1332, i64 0, !dbg !19
  %bnk.n1334 = lshr i64 %bnk.n1329, 32, !dbg !19
  %bnk.n1335 = trunc i64 %bnk.n1334 to i32, !dbg !19
  %bnk.n1336 = insertelement <4 x i32> %bnk.n1333, i32 %bnk.n1335, i64 1, !dbg !19
  %bnk.n1337 = insertelement <4 x i32> %bnk.n1336, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1550 = insertelement <4 x i32> %bnk.n1337, i32 131072, i64 3, !dbg !19
  %bnk.n1338 = ptrtoint ptr addrspace(3) %bnk.n1328 to i32, !dbg !19
  %dma.buffer.lds1551 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n1338), !dbg !19
  %bnk.n1339 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1551, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #10, !dbg !19
  %bnk.n1340 = shufflevector <8 x half> %bnk.n1295, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1341 = extractvalue [2 x <8 x float>] %bnk.n208, 0, !dbg !31
  %bnk.n1342 = shufflevector <8 x float> %bnk.n1341, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %bnk.n1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1340, <4 x float> %bnk.n1342) #9, !dbg !15
  %bnk.n1344 = shufflevector <8 x half> %bnk.n1297, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1344, <4 x float> %bnk.n1343) #9, !dbg !15
  %bnk.n1346 = shufflevector <8 x half> %bnk.n1299, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1346, <4 x float> %bnk.n1345) #9, !dbg !15
  %bnk.n1348 = shufflevector <8 x half> %bnk.n1301, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1348, <4 x float> %bnk.n1347) #9, !dbg !15
  %bnk.n1350 = shufflevector <8 x half> %bnk.n1303, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1350, <4 x float> %bnk.n1349) #9, !dbg !15
  %bnk.n1352 = shufflevector <8 x half> %bnk.n1305, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1352, <4 x float> %bnk.n1351) #9, !dbg !15
  %bnk.n1354 = shufflevector <8 x half> %bnk.n1307, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1354, <4 x float> %bnk.n1353) #9, !dbg !15
  %bnk.n1356 = shufflevector <8 x half> %bnk.n1309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1356, <4 x float> %bnk.n1355) #9, !dbg !15
  %bnk.n1358 = shufflevector <8 x half> %bnk.n1295, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1359 = shufflevector <8 x float> %bnk.n1341, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %bnk.n1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1358, <4 x float> %bnk.n1359) #9, !dbg !15
  %bnk.n1361 = shufflevector <8 x half> %bnk.n1297, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1361, <4 x float> %bnk.n1360) #9, !dbg !15
  %bnk.n1363 = shufflevector <8 x half> %bnk.n1299, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1363, <4 x float> %bnk.n1362) #9, !dbg !15
  %bnk.n1365 = shufflevector <8 x half> %bnk.n1301, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1365, <4 x float> %bnk.n1364) #9, !dbg !15
  %bnk.n1367 = shufflevector <8 x half> %bnk.n1303, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1367, <4 x float> %bnk.n1366) #9, !dbg !15
  %bnk.n1369 = shufflevector <8 x half> %bnk.n1305, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1369, <4 x float> %bnk.n1368) #9, !dbg !15
  %bnk.n1371 = shufflevector <8 x half> %bnk.n1307, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1371, <4 x float> %bnk.n1370) #9, !dbg !15
  %bnk.n1373 = shufflevector <8 x half> %bnk.n1309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1373, <4 x float> %bnk.n1372) #9, !dbg !15
  %bnk.n1375 = extractvalue [2 x <8 x float>] %bnk.n208, 1, !dbg !31
  %bnk.n1376 = shufflevector <8 x float> %bnk.n1375, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %bnk.n1377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1340, <4 x float> %bnk.n1376) #9, !dbg !15
  %bnk.n1378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1344, <4 x float> %bnk.n1377) #9, !dbg !15
  %bnk.n1379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1346, <4 x float> %bnk.n1378) #9, !dbg !15
  %bnk.n1380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1348, <4 x float> %bnk.n1379) #9, !dbg !15
  %bnk.n1381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1350, <4 x float> %bnk.n1380) #9, !dbg !15
  %bnk.n1382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1352, <4 x float> %bnk.n1381) #9, !dbg !15
  %bnk.n1383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1354, <4 x float> %bnk.n1382) #9, !dbg !15
  %bnk.n1384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1356, <4 x float> %bnk.n1383) #9, !dbg !15
  %bnk.n1385 = shufflevector <8 x float> %bnk.n1375, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %bnk.n1386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1358, <4 x float> %bnk.n1385) #9, !dbg !15
  %bnk.n1387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1361, <4 x float> %bnk.n1386) #9, !dbg !15
  %bnk.n1388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1363, <4 x float> %bnk.n1387) #9, !dbg !15
  %bnk.n1389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1365, <4 x float> %bnk.n1388) #9, !dbg !15
  %bnk.n1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1367, <4 x float> %bnk.n1389) #9, !dbg !15
  %bnk.n1391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1369, <4 x float> %bnk.n1390) #9, !dbg !15
  %bnk.n1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1371, <4 x float> %bnk.n1391) #9, !dbg !15
  %bnk.n1393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1373, <4 x float> %bnk.n1392) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %bnk.n1394 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !15
  %bnk.n1395 = bitcast <4 x i32> %bnk.n1394 to <8 x half>, !dbg !15
  %bnk.n1396 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !15
  %bnk.n1397 = bitcast <4 x i32> %bnk.n1396 to <8 x half>, !dbg !15
  %bnk.n1398 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !15
  %bnk.n1399 = bitcast <4 x i32> %bnk.n1398 to <8 x half>, !dbg !15
  %bnk.n1400 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !15
  %bnk.n1401 = bitcast <4 x i32> %bnk.n1400 to <8 x half>, !dbg !15
  %bnk.n1402 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !15
  %bnk.n1403 = bitcast <4 x i32> %bnk.n1402 to <8 x half>, !dbg !15
  %bnk.n1404 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !15
  %bnk.n1405 = bitcast <4 x i32> %bnk.n1404 to <8 x half>, !dbg !15
  %bnk.n1406 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !15
  %bnk.n1407 = bitcast <4 x i32> %bnk.n1406 to <8 x half>, !dbg !15
  %bnk.n1408 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !15
  %bnk.n1409 = bitcast <4 x i32> %bnk.n1408 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %bnk.n1410 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %bnk.n1209, !dbg !19
  %bnk.n1411 = getelementptr i8, ptr addrspace(1) %bnk.n1410, i64 192, !dbg !19
  %bnk.n1412 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !19
  %bnk.n1413 = zext i32 %bnk.n1412 to i64, !dbg !19
  %bnk.n1414 = inttoptr i64 %bnk.n1413 to ptr addrspace(3), !dbg !19
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n1411, i64 %address.local916, !dbg !19
  %bnk.n1415 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %bnk.n1416 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !19
  %bnk.n1417 = sub i64 %bnk.n1416, %bnk.n1415, !dbg !19
  %dma.buffer.offset1546 = trunc i64 %bnk.n1417 to i32, !dbg !19
  %bnk.n1418 = trunc i64 %bnk.n1415 to i32, !dbg !19
  %bnk.n1419 = insertelement <4 x i32> poison, i32 %bnk.n1418, i64 0, !dbg !19
  %bnk.n1420 = lshr i64 %bnk.n1415, 32, !dbg !19
  %bnk.n1421 = trunc i64 %bnk.n1420 to i32, !dbg !19
  %bnk.n1422 = insertelement <4 x i32> %bnk.n1419, i32 %bnk.n1421, i64 1, !dbg !19
  %bnk.n1423 = insertelement <4 x i32> %bnk.n1422, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1547 = insertelement <4 x i32> %bnk.n1423, i32 131072, i64 3, !dbg !19
  %bnk.n1424 = ptrtoint ptr addrspace(3) %bnk.n1414 to i32, !dbg !19
  %dma.buffer.lds1548 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n1424), !dbg !19
  %bnk.n1425 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1548, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1547) #10, !dbg !19
  %bnk.n1426 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !19
  %bnk.n1427 = zext i32 %bnk.n1426 to i64, !dbg !19
  %bnk.n1428 = inttoptr i64 %bnk.n1427 to ptr addrspace(3), !dbg !19
  %address.local933 = getelementptr [2 x i8], ptr addrspace(1) %bnk.n1411, i64 %address.local928, !dbg !19
  %bnk.n1429 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %bnk.n1430 = ptrtoint ptr addrspace(1) %address.local933 to i64, !dbg !19
  %bnk.n1431 = sub i64 %bnk.n1430, %bnk.n1429, !dbg !19
  %dma.buffer.offset = trunc i64 %bnk.n1431 to i32, !dbg !19
  %bnk.n1432 = trunc i64 %bnk.n1429 to i32, !dbg !19
  %bnk.n1433 = insertelement <4 x i32> poison, i32 %bnk.n1432, i64 0, !dbg !19
  %bnk.n1434 = lshr i64 %bnk.n1429, 32, !dbg !19
  %bnk.n1435 = trunc i64 %bnk.n1434 to i32, !dbg !19
  %bnk.n1436 = insertelement <4 x i32> %bnk.n1433, i32 %bnk.n1435, i64 1, !dbg !19
  %bnk.n1437 = insertelement <4 x i32> %bnk.n1436, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc = insertelement <4 x i32> %bnk.n1437, i32 131072, i64 3, !dbg !19
  %bnk.n1438 = ptrtoint ptr addrspace(3) %bnk.n1428 to i32, !dbg !19
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n1438), !dbg !19
  %bnk.n1439 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !19
  %bnk.n1440 = shufflevector <8 x half> %bnk.n1395, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1441 = extractvalue [2 x <8 x float>] %bnk.n209, 0, !dbg !31
  %bnk.n1442 = shufflevector <8 x float> %bnk.n1441, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %bnk.n1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1440, <4 x float> %bnk.n1442) #9, !dbg !15
  %bnk.n1444 = shufflevector <8 x half> %bnk.n1397, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1444, <4 x float> %bnk.n1443) #9, !dbg !15
  %bnk.n1446 = shufflevector <8 x half> %bnk.n1399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1446, <4 x float> %bnk.n1445) #9, !dbg !15
  %bnk.n1448 = shufflevector <8 x half> %bnk.n1401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1448, <4 x float> %bnk.n1447) #9, !dbg !15
  %bnk.n1450 = shufflevector <8 x half> %bnk.n1403, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1450, <4 x float> %bnk.n1449) #9, !dbg !15
  %bnk.n1452 = shufflevector <8 x half> %bnk.n1405, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1452, <4 x float> %bnk.n1451) #9, !dbg !15
  %bnk.n1454 = shufflevector <8 x half> %bnk.n1407, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1454, <4 x float> %bnk.n1453) #9, !dbg !15
  %bnk.n1456 = shufflevector <8 x half> %bnk.n1409, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1456, <4 x float> %bnk.n1455) #9, !dbg !15
  %bnk.n1458 = shufflevector <8 x half> %bnk.n1395, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1459 = shufflevector <8 x float> %bnk.n1441, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %bnk.n1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1458, <4 x float> %bnk.n1459) #9, !dbg !15
  %bnk.n1461 = shufflevector <8 x half> %bnk.n1397, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1461, <4 x float> %bnk.n1460) #9, !dbg !15
  %bnk.n1463 = shufflevector <8 x half> %bnk.n1399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1463, <4 x float> %bnk.n1462) #9, !dbg !15
  %bnk.n1465 = shufflevector <8 x half> %bnk.n1401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1465, <4 x float> %bnk.n1464) #9, !dbg !15
  %bnk.n1467 = shufflevector <8 x half> %bnk.n1403, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1467, <4 x float> %bnk.n1466) #9, !dbg !15
  %bnk.n1469 = shufflevector <8 x half> %bnk.n1405, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1469, <4 x float> %bnk.n1468) #9, !dbg !15
  %bnk.n1471 = shufflevector <8 x half> %bnk.n1407, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1471, <4 x float> %bnk.n1470) #9, !dbg !15
  %bnk.n1473 = shufflevector <8 x half> %bnk.n1409, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1473, <4 x float> %bnk.n1472) #9, !dbg !15
  %bnk.n1475 = extractvalue [2 x <8 x float>] %bnk.n209, 1, !dbg !31
  %bnk.n1476 = shufflevector <8 x float> %bnk.n1475, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %bnk.n1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1440, <4 x float> %bnk.n1476) #9, !dbg !15
  %bnk.n1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1444, <4 x float> %bnk.n1477) #9, !dbg !15
  %bnk.n1479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1446, <4 x float> %bnk.n1478) #9, !dbg !15
  %bnk.n1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1448, <4 x float> %bnk.n1479) #9, !dbg !15
  %bnk.n1481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1450, <4 x float> %bnk.n1480) #9, !dbg !15
  %bnk.n1482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1452, <4 x float> %bnk.n1481) #9, !dbg !15
  %bnk.n1483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1454, <4 x float> %bnk.n1482) #9, !dbg !15
  %bnk.n1484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1456, <4 x float> %bnk.n1483) #9, !dbg !15
  %bnk.n1485 = shufflevector <8 x float> %bnk.n1475, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %bnk.n1486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1458, <4 x float> %bnk.n1485) #9, !dbg !15
  %bnk.n1487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1461, <4 x float> %bnk.n1486) #9, !dbg !15
  %bnk.n1488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1463, <4 x float> %bnk.n1487) #9, !dbg !15
  %bnk.n1489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1465, <4 x float> %bnk.n1488) #9, !dbg !15
  %bnk.n1490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1467, <4 x float> %bnk.n1489) #9, !dbg !15
  %bnk.n1491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1469, <4 x float> %bnk.n1490) #9, !dbg !15
  %bnk.n1492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1471, <4 x float> %bnk.n1491) #9, !dbg !15
  %bnk.n1493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1473, <4 x float> %bnk.n1492) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %bnk.n1494 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !15
  %bnk.n1495 = bitcast <4 x i32> %bnk.n1494 to <8 x half>, !dbg !15
  %bnk.n1496 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !15
  %bnk.n1497 = bitcast <4 x i32> %bnk.n1496 to <8 x half>, !dbg !15
  %bnk.n1498 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !15
  %bnk.n1499 = bitcast <4 x i32> %bnk.n1498 to <8 x half>, !dbg !15
  %bnk.n1500 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !15
  %bnk.n1501 = bitcast <4 x i32> %bnk.n1500 to <8 x half>, !dbg !15
  %bnk.n1502 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !15
  %bnk.n1503 = bitcast <4 x i32> %bnk.n1502 to <8 x half>, !dbg !15
  %bnk.n1504 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !15
  %bnk.n1505 = bitcast <4 x i32> %bnk.n1504 to <8 x half>, !dbg !15
  %bnk.n1506 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !15
  %bnk.n1507 = bitcast <4 x i32> %bnk.n1506 to <8 x half>, !dbg !15
  %bnk.n1508 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !15
  %bnk.n1509 = bitcast <4 x i32> %bnk.n1508 to <8 x half>, !dbg !15
; V3 readers complete before low LDS is reused.
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %bnk.n1580 = add nuw nsw i64 %bnk.n205, 1, !dbg !13
  %k0.has.next = icmp slt i64 %bnk.n1580, %bnk.n191
  br i1 %k0.has.next, label %k0.issue, label %k0.join

k0.issue:
  %k0.next.address.thread433 = call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #9, !dbg !17, !range !34, !frisk.register_lifetimes !1
  %k0.next.address.local434 = zext nneg i32 %k0.next.address.thread433 to i64, !dbg !8
  %k0.next.address.local435 = shl nuw nsw i64 %k0.next.address.local434, 3, !dbg !8
  %k0.next.address.local436 = and i64 %k0.next.address.local435, 1536, !dbg !8
  %k0.next.address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.next.address.local436, !dbg !8
  %k0.next.address.local438 = ptrtoint ptr addrspace(3) %k0.next.address.local437 to i64, !dbg !8
  %k0.next.address.local439 = trunc i64 %k0.next.address.local438 to i32, !dbg !8
  %k0.next.bnk.n226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.address.local439), !dbg !17
  %k0.next.bnk.n227 = zext i32 %k0.next.bnk.n226 to i64, !dbg !17
  %k0.next.bnk.n228 = inttoptr i64 %k0.next.bnk.n227 to ptr addrspace(3), !dbg !17
  %k0.next.bnk.n238 = ptrtoint ptr addrspace(3) %k0.next.bnk.n228 to i32, !dbg !17
  %k0.next.dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.bnk.n238), !dbg !17
  %k0.next.bnk.n214 = lshr i64 %bnk.n1580, 12, !dbg !17
  %k0.next.bnk.n215 = add nuw nsw i64 %k0.next.bnk.n214, %bnk.n16, !dbg !17
  %k0.next.bnk.n218 = shl i64 %k0.next.bnk.n215, 19, !dbg !17
  %k0.next.bnk.n219 = and i64 %k0.next.bnk.n218, 16252928, !dbg !17
  %k0.next.bnk.n223 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %k0.next.bnk.n219, !dbg !17
  %k0.next.bnk.n216 = lshr i64 %bnk.n1580, 5, !dbg !17
  %k0.next.bnk.n217 = and i64 %k0.next.bnk.n216, 127, !dbg !17
  %k0.next.bnk.n224 = getelementptr [2 x i8], ptr addrspace(1) %k0.next.bnk.n223, i64 %k0.next.bnk.n217, !dbg !17
  %k0.next.bnk.n221 = shl i64 %bnk.n1580, 14, !dbg !17
  %k0.next.bnk.n220 = shl i64 %k0.next.bnk.n216, 19, !dbg !17
  %k0.next.bnk.n222 = sub i64 %k0.next.bnk.n221, %k0.next.bnk.n220, !dbg !17
  %k0.next.bnk.n225 = getelementptr [2 x i8], ptr addrspace(1) %k0.next.bnk.n224, i64 %k0.next.bnk.n222, !dbg !17
  %k0.next.address.local440 = lshr i64 %k0.next.address.local434, 4, !dbg !17
  %k0.next.address.local441 = and i64 %k0.next.address.local440, 8, !dbg !17
  %k0.next.address.local442 = and i64 %k0.next.address.local434, 123, !dbg !17
  %k0.next.address.local443 = lshr i64 %k0.next.address.local434, 2, !dbg !17
  %k0.next.address.local444 = and i64 %k0.next.address.local443, 4, !dbg !17
  %k0.next.address.local445 = add nuw nsw i64 %k0.next.address.local444, %k0.next.address.local434, !dbg !17
  %k0.next.address.local446 = and i64 %k0.next.address.local445, 4, !dbg !17
  %k0.next.address.local447 = or disjoint i64 %k0.next.address.local442, %k0.next.address.local446, !dbg !17
  %k0.next.address.local448 = shl nuw nsw i64 %k0.next.address.local447, 7, !dbg !17
  %k0.next.address.local449 = or disjoint i64 %k0.next.address.local441, %k0.next.address.local448, !dbg !17
  %k0.next.address.local450 = getelementptr [2 x i8], ptr addrspace(1) %k0.next.bnk.n225, i64 %k0.next.address.local449, !dbg !17
  %k0.next.bnk.n230 = ptrtoint ptr addrspace(1) %k0.next.address.local450 to i64, !dbg !17
  %k0.next.bnk.n229 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %k0.next.bnk.n231 = sub i64 %k0.next.bnk.n230, %k0.next.bnk.n229, !dbg !17
  %k0.next.dma.buffer.offset1588 = trunc i64 %k0.next.bnk.n231 to i32, !dbg !17
  %k0.next.bnk.n232 = trunc i64 %k0.next.bnk.n229 to i32, !dbg !17
  %k0.next.bnk.n233 = insertelement <4 x i32> poison, i32 %k0.next.bnk.n232, i64 0, !dbg !17
  %k0.next.bnk.n234 = lshr i64 %k0.next.bnk.n229, 32, !dbg !17
  %k0.next.bnk.n235 = trunc i64 %k0.next.bnk.n234 to i32, !dbg !17
  %k0.next.bnk.n236 = insertelement <4 x i32> %k0.next.bnk.n233, i32 %k0.next.bnk.n235, i64 1, !dbg !17
  %k0.next.bnk.n237 = insertelement <4 x i32> %k0.next.bnk.n236, i32 33554432, i64 2, !dbg !17
  %k0.next.dma.buffer.rsrc1589 = insertelement <4 x i32> %k0.next.bnk.n237, i32 131072, i64 3, !dbg !17
  %k0.next.bnk.n239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.next.dma.buffer.lds1590, i32 %k0.next.dma.buffer.offset1588, <4 x i32> %k0.next.dma.buffer.rsrc1589) #10, !dbg !17
  %k0.next.address.local451 = or disjoint i64 %k0.next.address.local434, 256, !dbg !8
  %k0.next.address.local452 = shl nuw nsw i64 %k0.next.address.local451, 3, !dbg !8
  %k0.next.address.local453 = and i64 %k0.next.address.local452, 3584, !dbg !8
  %k0.next.address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.next.address.local453, !dbg !8
  %k0.next.address.local455 = ptrtoint ptr addrspace(3) %k0.next.address.local454 to i64, !dbg !8
  %k0.next.address.local456 = trunc i64 %k0.next.address.local455 to i32, !dbg !8
  %k0.next.bnk.n240 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.address.local456), !dbg !17
  %k0.next.bnk.n241 = zext i32 %k0.next.bnk.n240 to i64, !dbg !17
  %k0.next.bnk.n242 = inttoptr i64 %k0.next.bnk.n241 to ptr addrspace(3), !dbg !17
  %k0.next.bnk.n252 = ptrtoint ptr addrspace(3) %k0.next.bnk.n242 to i32, !dbg !17
  %k0.next.dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.bnk.n252), !dbg !17
  %k0.next.address.local457 = lshr i64 %k0.next.address.local451, 4, !dbg !17
  %k0.next.address.local458 = and i64 %k0.next.address.local457, 24, !dbg !17
  %k0.next.address.local459 = and i64 %k0.next.address.local434, 123, !dbg !17
  %k0.next.address.local460 = lshr i64 %k0.next.address.local434, 2, !dbg !17
  %k0.next.address.local461 = and i64 %k0.next.address.local460, 4, !dbg !17
  %k0.next.address.local462 = add nuw nsw i64 %k0.next.address.local461, %k0.next.address.local434, !dbg !17
  %k0.next.address.local463 = and i64 %k0.next.address.local462, 4, !dbg !17
  %k0.next.address.local464 = or disjoint i64 %k0.next.address.local459, %k0.next.address.local463, !dbg !17
  %k0.next.address.local465 = shl nuw nsw i64 %k0.next.address.local464, 7, !dbg !17
  %k0.next.address.local466 = or disjoint i64 %k0.next.address.local458, %k0.next.address.local465, !dbg !17
  %k0.next.address.local467 = getelementptr [2 x i8], ptr addrspace(1) %k0.next.bnk.n225, i64 %k0.next.address.local466, !dbg !17
  %k0.next.bnk.n244 = ptrtoint ptr addrspace(1) %k0.next.address.local467 to i64, !dbg !17
  %k0.next.bnk.n243 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %k0.next.bnk.n245 = sub i64 %k0.next.bnk.n244, %k0.next.bnk.n243, !dbg !17
  %k0.next.dma.buffer.offset1585 = trunc i64 %k0.next.bnk.n245 to i32, !dbg !17
  %k0.next.bnk.n246 = trunc i64 %k0.next.bnk.n243 to i32, !dbg !17
  %k0.next.bnk.n247 = insertelement <4 x i32> poison, i32 %k0.next.bnk.n246, i64 0, !dbg !17
  %k0.next.bnk.n248 = lshr i64 %k0.next.bnk.n243, 32, !dbg !17
  %k0.next.bnk.n249 = trunc i64 %k0.next.bnk.n248 to i32, !dbg !17
  %k0.next.bnk.n250 = insertelement <4 x i32> %k0.next.bnk.n247, i32 %k0.next.bnk.n249, i64 1, !dbg !17
  %k0.next.bnk.n251 = insertelement <4 x i32> %k0.next.bnk.n250, i32 33554432, i64 2, !dbg !17
  %k0.next.dma.buffer.rsrc1586 = insertelement <4 x i32> %k0.next.bnk.n251, i32 131072, i64 3, !dbg !17
  %k0.next.bnk.n253 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.next.dma.buffer.lds1587, i32 %k0.next.dma.buffer.offset1585, <4 x i32> %k0.next.dma.buffer.rsrc1586) #10, !dbg !17
  br label %k0.join

k0.join:
  %bnk.n1510 = shufflevector <8 x half> %bnk.n1495, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1511 = extractvalue [2 x <8 x float>] %bnk.n210, 0, !dbg !31
  %bnk.n1512 = shufflevector <8 x float> %bnk.n1511, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %bnk.n1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1510, <4 x float> %bnk.n1512) #9, !dbg !15
  %bnk.n1514 = shufflevector <8 x half> %bnk.n1497, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1514, <4 x float> %bnk.n1513) #9, !dbg !15
  %bnk.n1516 = shufflevector <8 x half> %bnk.n1499, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1516, <4 x float> %bnk.n1515) #9, !dbg !15
  %bnk.n1518 = shufflevector <8 x half> %bnk.n1501, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1518, <4 x float> %bnk.n1517) #9, !dbg !15
  %bnk.n1520 = shufflevector <8 x half> %bnk.n1503, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1520, <4 x float> %bnk.n1519) #9, !dbg !15
  %bnk.n1522 = shufflevector <8 x half> %bnk.n1505, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1522, <4 x float> %bnk.n1521) #9, !dbg !15
  %bnk.n1524 = shufflevector <8 x half> %bnk.n1507, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1524, <4 x float> %bnk.n1523) #9, !dbg !15
  %bnk.n1526 = shufflevector <8 x half> %bnk.n1509, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %bnk.n1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1526, <4 x float> %bnk.n1525) #9, !dbg !15
  %bnk.n1528 = shufflevector <8 x half> %bnk.n1495, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1529 = shufflevector <8 x float> %bnk.n1511, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %bnk.n1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1528, <4 x float> %bnk.n1529) #9, !dbg !15
  %bnk.n1531 = shufflevector <8 x half> %bnk.n1497, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1531, <4 x float> %bnk.n1530) #9, !dbg !15
  %bnk.n1533 = shufflevector <8 x half> %bnk.n1499, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1533, <4 x float> %bnk.n1532) #9, !dbg !15
  %bnk.n1535 = shufflevector <8 x half> %bnk.n1501, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1535, <4 x float> %bnk.n1534) #9, !dbg !15
  %bnk.n1537 = shufflevector <8 x half> %bnk.n1503, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1537, <4 x float> %bnk.n1536) #9, !dbg !15
  %bnk.n1539 = shufflevector <8 x half> %bnk.n1505, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1539, <4 x float> %bnk.n1538) #9, !dbg !15
  %bnk.n1541 = shufflevector <8 x half> %bnk.n1507, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1541, <4 x float> %bnk.n1540) #9, !dbg !15
  %bnk.n1543 = shufflevector <8 x half> %bnk.n1509, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %bnk.n1544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1543, <4 x float> %bnk.n1542) #9, !dbg !15
  %bnk.n1545 = extractvalue [2 x <8 x float>] %bnk.n210, 1, !dbg !31
  %bnk.n1546 = shufflevector <8 x float> %bnk.n1545, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %bnk.n1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1510, <4 x float> %bnk.n1546) #9, !dbg !15
  %bnk.n1548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1514, <4 x float> %bnk.n1547) #9, !dbg !15
  %bnk.n1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1516, <4 x float> %bnk.n1548) #9, !dbg !15
  %bnk.n1550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1518, <4 x float> %bnk.n1549) #9, !dbg !15
  %bnk.n1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1520, <4 x float> %bnk.n1550) #9, !dbg !15
  %bnk.n1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1522, <4 x float> %bnk.n1551) #9, !dbg !15
  %bnk.n1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1524, <4 x float> %bnk.n1552) #9, !dbg !15
  %bnk.n1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1526, <4 x float> %bnk.n1553) #9, !dbg !15
  %bnk.n1555 = shufflevector <8 x float> %bnk.n1545, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %bnk.n1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1528, <4 x float> %bnk.n1555) #9, !dbg !15
  %bnk.n1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1531, <4 x float> %bnk.n1556) #9, !dbg !15
  %bnk.n1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1533, <4 x float> %bnk.n1557) #9, !dbg !15
  %bnk.n1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1535, <4 x float> %bnk.n1558) #9, !dbg !15
  %bnk.n1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1537, <4 x float> %bnk.n1559) #9, !dbg !15
  %bnk.n1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1539, <4 x float> %bnk.n1560) #9, !dbg !15
  %bnk.n1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1541, <4 x float> %bnk.n1561) #9, !dbg !15
  %bnk.n1563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1543, <4 x float> %bnk.n1562) #9, !dbg !15
  %bnk.n1564 = shufflevector <4 x float> %bnk.n1257, <4 x float> %bnk.n1274, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %bnk.n1565 = insertvalue [2 x <8 x float>] poison, <8 x float> %bnk.n1564, 0, !dbg !13
  %bnk.n1566 = shufflevector <4 x float> %bnk.n1284, <4 x float> %bnk.n1293, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %bnk.n1567 = insertvalue [2 x <8 x float>] %bnk.n1565, <8 x float> %bnk.n1566, 1, !dbg !13
  %bnk.n1568 = shufflevector <4 x float> %bnk.n1357, <4 x float> %bnk.n1374, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %bnk.n1569 = insertvalue [2 x <8 x float>] poison, <8 x float> %bnk.n1568, 0, !dbg !13
  %bnk.n1570 = shufflevector <4 x float> %bnk.n1384, <4 x float> %bnk.n1393, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %bnk.n1571 = insertvalue [2 x <8 x float>] %bnk.n1569, <8 x float> %bnk.n1570, 1, !dbg !13
  %bnk.n1572 = shufflevector <4 x float> %bnk.n1457, <4 x float> %bnk.n1474, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %bnk.n1573 = insertvalue [2 x <8 x float>] poison, <8 x float> %bnk.n1572, 0, !dbg !13
  %bnk.n1574 = shufflevector <4 x float> %bnk.n1484, <4 x float> %bnk.n1493, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %bnk.n1575 = insertvalue [2 x <8 x float>] %bnk.n1573, <8 x float> %bnk.n1574, 1, !dbg !13
  %bnk.n1576 = shufflevector <4 x float> %bnk.n1527, <4 x float> %bnk.n1544, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %bnk.n1577 = insertvalue [2 x <8 x float>] poison, <8 x float> %bnk.n1576, 0, !dbg !13
  %bnk.n1578 = shufflevector <4 x float> %bnk.n1554, <4 x float> %bnk.n1563, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %bnk.n1579 = insertvalue [2 x <8 x float>] %bnk.n1577, <8 x float> %bnk.n1578, 1, !dbg !13
  br label %bnk.n204, !dbg !13

bnk.n1581:                                             ; preds = %bnk.n204
  %bnk.n1582 = and i32 %bnk.n31, 16, !dbg !13
  %.not = icmp eq i32 %bnk.n1582, 0, !dbg !13
  %bnk.n1583 = and i32 %bnk.n31, 32, !dbg !13
  %.not36 = icmp eq i32 %bnk.n1583, 0, !dbg !13
  %bnk.n1584 = extractvalue [2 x <8 x float>] %bnk.n210, 0, !dbg !13
  %bnk.n1585 = extractelement <8 x float> %bnk.n1584, i64 0, !dbg !13
  %bnk.n1586 = extractelement <8 x float> %bnk.n1584, i64 1, !dbg !13
  %bnk.n1587 = extractelement <8 x float> %bnk.n1584, i64 2, !dbg !13
  %bnk.n1588 = extractelement <8 x float> %bnk.n1584, i64 3, !dbg !13
  %bnk.n1589 = select i1 %.not, float %bnk.n1586, float %bnk.n1585, !dbg !13
  %bnk.n1590 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %bnk.n1591 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %bnk.n1590), !dbg !13
  %bnk.n1592 = add nuw nsw i32 %bnk.n1591, 64, !dbg !13
  %bnk.n1593 = xor i32 %bnk.n1591, 16, !dbg !13
  %bnk.n1594 = and i32 %bnk.n1592, 192, !dbg !13
  %bnk.n1595 = icmp samesign ult i32 %bnk.n1593, %bnk.n1594, !dbg !13
  %bnk.n1596 = select i1 %bnk.n1595, i32 %bnk.n1593, i32 %bnk.n1591, !dbg !13
  %bnk.n1597 = shl nuw nsw i32 %bnk.n1596, 2, !dbg !13
  %bnk.n1598 = bitcast float %bnk.n1589 to i32, !dbg !13
  %bnk.n1599 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1598), !dbg !13
  %bnk.n1600 = bitcast i32 %bnk.n1599 to float, !dbg !13
  %bnk.n1601 = select i1 %.not, float %bnk.n1585, float %bnk.n1600, !dbg !13
  %bnk.n1602 = select i1 %.not, float %bnk.n1600, float %bnk.n1586, !dbg !13
  %bnk.n1603 = select i1 %.not, float %bnk.n1588, float %bnk.n1587, !dbg !13
  %bnk.n1604 = bitcast float %bnk.n1603 to i32, !dbg !13
  %bnk.n1605 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1604), !dbg !13
  %bnk.n1606 = bitcast i32 %bnk.n1605 to float, !dbg !13
  %bnk.n1607 = select i1 %.not, float %bnk.n1587, float %bnk.n1606, !dbg !13
  %bnk.n1608 = select i1 %.not, float %bnk.n1606, float %bnk.n1588, !dbg !13
  %bnk.n1609 = select i1 %.not36, float %bnk.n1607, float %bnk.n1601, !dbg !13
  %bnk.n1610 = xor i32 %bnk.n1591, 32, !dbg !13
  %bnk.n1611 = icmp samesign ult i32 %bnk.n1610, %bnk.n1594, !dbg !13
  %bnk.n1612 = select i1 %bnk.n1611, i32 %bnk.n1610, i32 %bnk.n1591, !dbg !13
  %bnk.n1613 = shl nuw nsw i32 %bnk.n1612, 2, !dbg !13
  %bnk.n1614 = bitcast float %bnk.n1609 to i32, !dbg !13
  %bnk.n1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1614), !dbg !13
  %bnk.n1616 = bitcast i32 %bnk.n1615 to float, !dbg !13
  %bnk.n1617 = select i1 %.not36, float %bnk.n1601, float %bnk.n1616, !dbg !13
  %bnk.n1618 = select i1 %.not36, float %bnk.n1616, float %bnk.n1607, !dbg !13
  %bnk.n1619 = select i1 %.not36, float %bnk.n1608, float %bnk.n1602, !dbg !13
  %bnk.n1620 = bitcast float %bnk.n1619 to i32, !dbg !13
  %bnk.n1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1620), !dbg !13
  %bnk.n1622 = bitcast i32 %bnk.n1621 to float, !dbg !13
  %bnk.n1623 = select i1 %.not36, float %bnk.n1602, float %bnk.n1622, !dbg !13
  %bnk.n1624 = select i1 %.not36, float %bnk.n1622, float %bnk.n1608, !dbg !13
  %bnk.n1625 = extractvalue [2 x <8 x float>] %bnk.n209, 0, !dbg !13
  %bnk.n1626 = extractelement <8 x float> %bnk.n1625, i64 0, !dbg !13
  %bnk.n1627 = extractelement <8 x float> %bnk.n1625, i64 1, !dbg !13
  %bnk.n1628 = extractelement <8 x float> %bnk.n1625, i64 2, !dbg !13
  %bnk.n1629 = extractelement <8 x float> %bnk.n1625, i64 3, !dbg !13
  %bnk.n1630 = select i1 %.not, float %bnk.n1627, float %bnk.n1626, !dbg !13
  %bnk.n1631 = bitcast float %bnk.n1630 to i32, !dbg !13
  %bnk.n1632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1631), !dbg !13
  %bnk.n1633 = bitcast i32 %bnk.n1632 to float, !dbg !13
  %bnk.n1634 = select i1 %.not, float %bnk.n1626, float %bnk.n1633, !dbg !13
  %bnk.n1635 = select i1 %.not, float %bnk.n1633, float %bnk.n1627, !dbg !13
  %bnk.n1636 = select i1 %.not, float %bnk.n1629, float %bnk.n1628, !dbg !13
  %bnk.n1637 = bitcast float %bnk.n1636 to i32, !dbg !13
  %bnk.n1638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1637), !dbg !13
  %bnk.n1639 = bitcast i32 %bnk.n1638 to float, !dbg !13
  %bnk.n1640 = select i1 %.not, float %bnk.n1628, float %bnk.n1639, !dbg !13
  %bnk.n1641 = select i1 %.not, float %bnk.n1639, float %bnk.n1629, !dbg !13
  %bnk.n1642 = select i1 %.not36, float %bnk.n1640, float %bnk.n1634, !dbg !13
  %bnk.n1643 = bitcast float %bnk.n1642 to i32, !dbg !13
  %bnk.n1644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1643), !dbg !13
  %bnk.n1645 = bitcast i32 %bnk.n1644 to float, !dbg !13
  %bnk.n1646 = select i1 %.not36, float %bnk.n1634, float %bnk.n1645, !dbg !13
  %bnk.n1647 = select i1 %.not36, float %bnk.n1645, float %bnk.n1640, !dbg !13
  %bnk.n1648 = select i1 %.not36, float %bnk.n1641, float %bnk.n1635, !dbg !13
  %bnk.n1649 = bitcast float %bnk.n1648 to i32, !dbg !13
  %bnk.n1650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1649), !dbg !13
  %bnk.n1651 = bitcast i32 %bnk.n1650 to float, !dbg !13
  %bnk.n1652 = select i1 %.not36, float %bnk.n1635, float %bnk.n1651, !dbg !13
  %bnk.n1653 = select i1 %.not36, float %bnk.n1651, float %bnk.n1641, !dbg !13
  %bnk.n1654 = extractvalue [2 x <8 x float>] %bnk.n208, 0, !dbg !13
  %bnk.n1655 = extractelement <8 x float> %bnk.n1654, i64 0, !dbg !13
  %bnk.n1656 = extractelement <8 x float> %bnk.n1654, i64 1, !dbg !13
  %bnk.n1657 = extractelement <8 x float> %bnk.n1654, i64 2, !dbg !13
  %bnk.n1658 = extractelement <8 x float> %bnk.n1654, i64 3, !dbg !13
  %bnk.n1659 = select i1 %.not, float %bnk.n1656, float %bnk.n1655, !dbg !13
  %bnk.n1660 = bitcast float %bnk.n1659 to i32, !dbg !13
  %bnk.n1661 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1660), !dbg !13
  %bnk.n1662 = bitcast i32 %bnk.n1661 to float, !dbg !13
  %bnk.n1663 = select i1 %.not, float %bnk.n1655, float %bnk.n1662, !dbg !13
  %bnk.n1664 = select i1 %.not, float %bnk.n1662, float %bnk.n1656, !dbg !13
  %bnk.n1665 = select i1 %.not, float %bnk.n1658, float %bnk.n1657, !dbg !13
  %bnk.n1666 = bitcast float %bnk.n1665 to i32, !dbg !13
  %bnk.n1667 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1666), !dbg !13
  %bnk.n1668 = bitcast i32 %bnk.n1667 to float, !dbg !13
  %bnk.n1669 = select i1 %.not, float %bnk.n1657, float %bnk.n1668, !dbg !13
  %bnk.n1670 = select i1 %.not, float %bnk.n1668, float %bnk.n1658, !dbg !13
  %bnk.n1671 = select i1 %.not36, float %bnk.n1669, float %bnk.n1663, !dbg !13
  %bnk.n1672 = bitcast float %bnk.n1671 to i32, !dbg !13
  %bnk.n1673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1672), !dbg !13
  %bnk.n1674 = bitcast i32 %bnk.n1673 to float, !dbg !13
  %bnk.n1675 = select i1 %.not36, float %bnk.n1663, float %bnk.n1674, !dbg !13
  %bnk.n1676 = select i1 %.not36, float %bnk.n1674, float %bnk.n1669, !dbg !13
  %bnk.n1677 = select i1 %.not36, float %bnk.n1670, float %bnk.n1664, !dbg !13
  %bnk.n1678 = bitcast float %bnk.n1677 to i32, !dbg !13
  %bnk.n1679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1678), !dbg !13
  %bnk.n1680 = bitcast i32 %bnk.n1679 to float, !dbg !13
  %bnk.n1681 = select i1 %.not36, float %bnk.n1664, float %bnk.n1680, !dbg !13
  %bnk.n1682 = select i1 %.not36, float %bnk.n1680, float %bnk.n1670, !dbg !13
  %bnk.n1683 = extractvalue [2 x <8 x float>] %bnk.n206, 0, !dbg !13
  %bnk.n1684 = extractelement <8 x float> %bnk.n1683, i64 0, !dbg !13
  %bnk.n1685 = extractelement <8 x float> %bnk.n1683, i64 1, !dbg !13
  %bnk.n1686 = extractelement <8 x float> %bnk.n1683, i64 2, !dbg !13
  %bnk.n1687 = extractelement <8 x float> %bnk.n1683, i64 3, !dbg !13
  %bnk.n1688 = select i1 %.not, float %bnk.n1685, float %bnk.n1684, !dbg !13
  %bnk.n1689 = bitcast float %bnk.n1688 to i32, !dbg !13
  %bnk.n1690 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1689), !dbg !13
  %bnk.n1691 = bitcast i32 %bnk.n1690 to float, !dbg !13
  %bnk.n1692 = select i1 %.not, float %bnk.n1684, float %bnk.n1691, !dbg !13
  %bnk.n1693 = select i1 %.not, float %bnk.n1691, float %bnk.n1685, !dbg !13
  %bnk.n1694 = select i1 %.not, float %bnk.n1687, float %bnk.n1686, !dbg !13
  %bnk.n1695 = bitcast float %bnk.n1694 to i32, !dbg !13
  %bnk.n1696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1695), !dbg !13
  %bnk.n1697 = bitcast i32 %bnk.n1696 to float, !dbg !13
  %bnk.n1698 = select i1 %.not, float %bnk.n1686, float %bnk.n1697, !dbg !13
  %bnk.n1699 = select i1 %.not, float %bnk.n1697, float %bnk.n1687, !dbg !13
  %bnk.n1700 = select i1 %.not36, float %bnk.n1698, float %bnk.n1692, !dbg !13
  %bnk.n1701 = bitcast float %bnk.n1700 to i32, !dbg !13
  %bnk.n1702 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1701), !dbg !13
  %bnk.n1703 = select i1 %.not36, float %bnk.n1699, float %bnk.n1693, !dbg !13
  %bnk.n1704 = bitcast float %bnk.n1703 to i32, !dbg !13
  %bnk.n1705 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1704), !dbg !13
  %bnk.n1706 = extractvalue [2 x <1 x float>] %bnk.n207, 0, !dbg !20
  %bnk.n1707 = shufflevector <1 x float> %bnk.n1706, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %bnk.n1708 = extractelement <8 x float> %bnk.n1584, i64 4, !dbg !13
  %bnk.n1709 = extractelement <8 x float> %bnk.n1584, i64 5, !dbg !13
  %bnk.n1710 = extractelement <8 x float> %bnk.n1584, i64 6, !dbg !13
  %bnk.n1711 = extractelement <8 x float> %bnk.n1584, i64 7, !dbg !13
  %bnk.n1712 = select i1 %.not, float %bnk.n1709, float %bnk.n1708, !dbg !13
  %bnk.n1713 = bitcast float %bnk.n1712 to i32, !dbg !13
  %bnk.n1714 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1713), !dbg !13
  %bnk.n1715 = bitcast i32 %bnk.n1714 to float, !dbg !13
  %bnk.n1716 = select i1 %.not, float %bnk.n1708, float %bnk.n1715, !dbg !13
  %bnk.n1717 = select i1 %.not, float %bnk.n1715, float %bnk.n1709, !dbg !13
  %bnk.n1718 = select i1 %.not, float %bnk.n1711, float %bnk.n1710, !dbg !13
  %bnk.n1719 = bitcast float %bnk.n1718 to i32, !dbg !13
  %bnk.n1720 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1719), !dbg !13
  %bnk.n1721 = bitcast i32 %bnk.n1720 to float, !dbg !13
  %bnk.n1722 = select i1 %.not, float %bnk.n1710, float %bnk.n1721, !dbg !13
  %bnk.n1723 = select i1 %.not, float %bnk.n1721, float %bnk.n1711, !dbg !13
  %bnk.n1724 = select i1 %.not36, float %bnk.n1722, float %bnk.n1716, !dbg !13
  %bnk.n1725 = bitcast float %bnk.n1724 to i32, !dbg !13
  %bnk.n1726 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1725), !dbg !13
  %bnk.n1727 = bitcast i32 %bnk.n1726 to float, !dbg !13
  %bnk.n1728 = select i1 %.not36, float %bnk.n1716, float %bnk.n1727, !dbg !13
  %bnk.n1729 = select i1 %.not36, float %bnk.n1727, float %bnk.n1722, !dbg !13
  %bnk.n1730 = select i1 %.not36, float %bnk.n1723, float %bnk.n1717, !dbg !13
  %bnk.n1731 = bitcast float %bnk.n1730 to i32, !dbg !13
  %bnk.n1732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1731), !dbg !13
  %bnk.n1733 = bitcast i32 %bnk.n1732 to float, !dbg !13
  %bnk.n1734 = select i1 %.not36, float %bnk.n1717, float %bnk.n1733, !dbg !13
  %bnk.n1735 = select i1 %.not36, float %bnk.n1733, float %bnk.n1723, !dbg !13
  %bnk.n1736 = extractelement <8 x float> %bnk.n1625, i64 4, !dbg !13
  %bnk.n1737 = extractelement <8 x float> %bnk.n1625, i64 5, !dbg !13
  %bnk.n1738 = extractelement <8 x float> %bnk.n1625, i64 6, !dbg !13
  %bnk.n1739 = extractelement <8 x float> %bnk.n1625, i64 7, !dbg !13
  %bnk.n1740 = select i1 %.not, float %bnk.n1737, float %bnk.n1736, !dbg !13
  %bnk.n1741 = bitcast float %bnk.n1740 to i32, !dbg !13
  %bnk.n1742 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1741), !dbg !13
  %bnk.n1743 = bitcast i32 %bnk.n1742 to float, !dbg !13
  %bnk.n1744 = select i1 %.not, float %bnk.n1736, float %bnk.n1743, !dbg !13
  %bnk.n1745 = select i1 %.not, float %bnk.n1743, float %bnk.n1737, !dbg !13
  %bnk.n1746 = select i1 %.not, float %bnk.n1739, float %bnk.n1738, !dbg !13
  %bnk.n1747 = bitcast float %bnk.n1746 to i32, !dbg !13
  %bnk.n1748 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1747), !dbg !13
  %bnk.n1749 = bitcast i32 %bnk.n1748 to float, !dbg !13
  %bnk.n1750 = select i1 %.not, float %bnk.n1738, float %bnk.n1749, !dbg !13
  %bnk.n1751 = select i1 %.not, float %bnk.n1749, float %bnk.n1739, !dbg !13
  %bnk.n1752 = select i1 %.not36, float %bnk.n1750, float %bnk.n1744, !dbg !13
  %bnk.n1753 = bitcast float %bnk.n1752 to i32, !dbg !13
  %bnk.n1754 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1753), !dbg !13
  %bnk.n1755 = bitcast i32 %bnk.n1754 to float, !dbg !13
  %bnk.n1756 = select i1 %.not36, float %bnk.n1744, float %bnk.n1755, !dbg !13
  %bnk.n1757 = select i1 %.not36, float %bnk.n1755, float %bnk.n1750, !dbg !13
  %bnk.n1758 = select i1 %.not36, float %bnk.n1751, float %bnk.n1745, !dbg !13
  %bnk.n1759 = bitcast float %bnk.n1758 to i32, !dbg !13
  %bnk.n1760 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1759), !dbg !13
  %bnk.n1761 = bitcast i32 %bnk.n1760 to float, !dbg !13
  %bnk.n1762 = select i1 %.not36, float %bnk.n1745, float %bnk.n1761, !dbg !13
  %bnk.n1763 = select i1 %.not36, float %bnk.n1761, float %bnk.n1751, !dbg !13
  %bnk.n1764 = extractelement <8 x float> %bnk.n1654, i64 4, !dbg !13
  %bnk.n1765 = extractelement <8 x float> %bnk.n1654, i64 5, !dbg !13
  %bnk.n1766 = extractelement <8 x float> %bnk.n1654, i64 6, !dbg !13
  %bnk.n1767 = extractelement <8 x float> %bnk.n1654, i64 7, !dbg !13
  %bnk.n1768 = select i1 %.not, float %bnk.n1765, float %bnk.n1764, !dbg !13
  %bnk.n1769 = bitcast float %bnk.n1768 to i32, !dbg !13
  %bnk.n1770 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1769), !dbg !13
  %bnk.n1771 = bitcast i32 %bnk.n1770 to float, !dbg !13
  %bnk.n1772 = select i1 %.not, float %bnk.n1764, float %bnk.n1771, !dbg !13
  %bnk.n1773 = select i1 %.not, float %bnk.n1771, float %bnk.n1765, !dbg !13
  %bnk.n1774 = select i1 %.not, float %bnk.n1767, float %bnk.n1766, !dbg !13
  %bnk.n1775 = bitcast float %bnk.n1774 to i32, !dbg !13
  %bnk.n1776 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1775), !dbg !13
  %bnk.n1777 = bitcast i32 %bnk.n1776 to float, !dbg !13
  %bnk.n1778 = select i1 %.not, float %bnk.n1766, float %bnk.n1777, !dbg !13
  %bnk.n1779 = select i1 %.not, float %bnk.n1777, float %bnk.n1767, !dbg !13
  %bnk.n1780 = select i1 %.not36, float %bnk.n1778, float %bnk.n1772, !dbg !13
  %bnk.n1781 = bitcast float %bnk.n1780 to i32, !dbg !13
  %bnk.n1782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1781), !dbg !13
  %bnk.n1783 = bitcast i32 %bnk.n1782 to float, !dbg !13
  %bnk.n1784 = select i1 %.not36, float %bnk.n1772, float %bnk.n1783, !dbg !13
  %bnk.n1785 = select i1 %.not36, float %bnk.n1783, float %bnk.n1778, !dbg !13
  %bnk.n1786 = select i1 %.not36, float %bnk.n1779, float %bnk.n1773, !dbg !13
  %bnk.n1787 = bitcast float %bnk.n1786 to i32, !dbg !13
  %bnk.n1788 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1787), !dbg !13
  %bnk.n1789 = bitcast i32 %bnk.n1788 to float, !dbg !13
  %bnk.n1790 = select i1 %.not36, float %bnk.n1773, float %bnk.n1789, !dbg !13
  %bnk.n1791 = select i1 %.not36, float %bnk.n1789, float %bnk.n1779, !dbg !13
  %bnk.n1792 = extractelement <8 x float> %bnk.n1683, i64 4, !dbg !13
  %bnk.n1793 = extractelement <8 x float> %bnk.n1683, i64 5, !dbg !13
  %bnk.n1794 = extractelement <8 x float> %bnk.n1683, i64 6, !dbg !13
  %bnk.n1795 = extractelement <8 x float> %bnk.n1683, i64 7, !dbg !13
  %bnk.n1796 = select i1 %.not, float %bnk.n1793, float %bnk.n1792, !dbg !13
  %bnk.n1797 = bitcast float %bnk.n1796 to i32, !dbg !13
  %bnk.n1798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1797), !dbg !13
  %bnk.n1799 = bitcast i32 %bnk.n1798 to float, !dbg !13
  %bnk.n1800 = select i1 %.not, float %bnk.n1792, float %bnk.n1799, !dbg !13
  %bnk.n1801 = select i1 %.not, float %bnk.n1799, float %bnk.n1793, !dbg !13
  %bnk.n1802 = select i1 %.not, float %bnk.n1795, float %bnk.n1794, !dbg !13
  %bnk.n1803 = bitcast float %bnk.n1802 to i32, !dbg !13
  %bnk.n1804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1803), !dbg !13
  %bnk.n1805 = bitcast i32 %bnk.n1804 to float, !dbg !13
  %bnk.n1806 = select i1 %.not, float %bnk.n1794, float %bnk.n1805, !dbg !13
  %bnk.n1807 = select i1 %.not, float %bnk.n1805, float %bnk.n1795, !dbg !13
  %bnk.n1808 = select i1 %.not36, float %bnk.n1806, float %bnk.n1800, !dbg !13
  %bnk.n1809 = bitcast float %bnk.n1808 to i32, !dbg !13
  %bnk.n1810 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1809), !dbg !13
  %bnk.n1811 = select i1 %.not36, float %bnk.n1807, float %bnk.n1801, !dbg !13
  %bnk.n1812 = bitcast float %bnk.n1811 to i32, !dbg !13
  %bnk.n1813 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1812), !dbg !13
  %bnk.n1814 = extractvalue [2 x <8 x float>] %bnk.n210, 1, !dbg !13
  %bnk.n1815 = extractelement <8 x float> %bnk.n1814, i64 0, !dbg !13
  %bnk.n1816 = extractelement <8 x float> %bnk.n1814, i64 1, !dbg !13
  %bnk.n1817 = extractelement <8 x float> %bnk.n1814, i64 2, !dbg !13
  %bnk.n1818 = extractelement <8 x float> %bnk.n1814, i64 3, !dbg !13
  %bnk.n1819 = select i1 %.not, float %bnk.n1816, float %bnk.n1815, !dbg !13
  %bnk.n1820 = bitcast float %bnk.n1819 to i32, !dbg !13
  %bnk.n1821 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1820), !dbg !13
  %bnk.n1822 = bitcast i32 %bnk.n1821 to float, !dbg !13
  %bnk.n1823 = select i1 %.not, float %bnk.n1815, float %bnk.n1822, !dbg !13
  %bnk.n1824 = select i1 %.not, float %bnk.n1822, float %bnk.n1816, !dbg !13
  %bnk.n1825 = select i1 %.not, float %bnk.n1818, float %bnk.n1817, !dbg !13
  %bnk.n1826 = bitcast float %bnk.n1825 to i32, !dbg !13
  %bnk.n1827 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1826), !dbg !13
  %bnk.n1828 = bitcast i32 %bnk.n1827 to float, !dbg !13
  %bnk.n1829 = select i1 %.not, float %bnk.n1817, float %bnk.n1828, !dbg !13
  %bnk.n1830 = select i1 %.not, float %bnk.n1828, float %bnk.n1818, !dbg !13
  %bnk.n1831 = select i1 %.not36, float %bnk.n1829, float %bnk.n1823, !dbg !13
  %bnk.n1832 = bitcast float %bnk.n1831 to i32, !dbg !13
  %bnk.n1833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1832), !dbg !13
  %bnk.n1834 = bitcast i32 %bnk.n1833 to float, !dbg !13
  %bnk.n1835 = select i1 %.not36, float %bnk.n1823, float %bnk.n1834, !dbg !13
  %bnk.n1836 = select i1 %.not36, float %bnk.n1834, float %bnk.n1829, !dbg !13
  %bnk.n1837 = select i1 %.not36, float %bnk.n1830, float %bnk.n1824, !dbg !13
  %bnk.n1838 = bitcast float %bnk.n1837 to i32, !dbg !13
  %bnk.n1839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1838), !dbg !13
  %bnk.n1840 = bitcast i32 %bnk.n1839 to float, !dbg !13
  %bnk.n1841 = select i1 %.not36, float %bnk.n1824, float %bnk.n1840, !dbg !13
  %bnk.n1842 = select i1 %.not36, float %bnk.n1840, float %bnk.n1830, !dbg !13
  %bnk.n1843 = extractvalue [2 x <8 x float>] %bnk.n209, 1, !dbg !13
  %bnk.n1844 = extractelement <8 x float> %bnk.n1843, i64 0, !dbg !13
  %bnk.n1845 = extractelement <8 x float> %bnk.n1843, i64 1, !dbg !13
  %bnk.n1846 = extractelement <8 x float> %bnk.n1843, i64 2, !dbg !13
  %bnk.n1847 = extractelement <8 x float> %bnk.n1843, i64 3, !dbg !13
  %bnk.n1848 = select i1 %.not, float %bnk.n1845, float %bnk.n1844, !dbg !13
  %bnk.n1849 = bitcast float %bnk.n1848 to i32, !dbg !13
  %bnk.n1850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1849), !dbg !13
  %bnk.n1851 = bitcast i32 %bnk.n1850 to float, !dbg !13
  %bnk.n1852 = select i1 %.not, float %bnk.n1844, float %bnk.n1851, !dbg !13
  %bnk.n1853 = select i1 %.not, float %bnk.n1851, float %bnk.n1845, !dbg !13
  %bnk.n1854 = select i1 %.not, float %bnk.n1847, float %bnk.n1846, !dbg !13
  %bnk.n1855 = bitcast float %bnk.n1854 to i32, !dbg !13
  %bnk.n1856 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1855), !dbg !13
  %bnk.n1857 = bitcast i32 %bnk.n1856 to float, !dbg !13
  %bnk.n1858 = select i1 %.not, float %bnk.n1846, float %bnk.n1857, !dbg !13
  %bnk.n1859 = select i1 %.not, float %bnk.n1857, float %bnk.n1847, !dbg !13
  %bnk.n1860 = select i1 %.not36, float %bnk.n1858, float %bnk.n1852, !dbg !13
  %bnk.n1861 = bitcast float %bnk.n1860 to i32, !dbg !13
  %bnk.n1862 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1861), !dbg !13
  %bnk.n1863 = bitcast i32 %bnk.n1862 to float, !dbg !13
  %bnk.n1864 = select i1 %.not36, float %bnk.n1852, float %bnk.n1863, !dbg !13
  %bnk.n1865 = select i1 %.not36, float %bnk.n1863, float %bnk.n1858, !dbg !13
  %bnk.n1866 = select i1 %.not36, float %bnk.n1859, float %bnk.n1853, !dbg !13
  %bnk.n1867 = bitcast float %bnk.n1866 to i32, !dbg !13
  %bnk.n1868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1867), !dbg !13
  %bnk.n1869 = bitcast i32 %bnk.n1868 to float, !dbg !13
  %bnk.n1870 = select i1 %.not36, float %bnk.n1853, float %bnk.n1869, !dbg !13
  %bnk.n1871 = select i1 %.not36, float %bnk.n1869, float %bnk.n1859, !dbg !13
  %bnk.n1872 = extractvalue [2 x <8 x float>] %bnk.n208, 1, !dbg !13
  %bnk.n1873 = extractelement <8 x float> %bnk.n1872, i64 0, !dbg !13
  %bnk.n1874 = extractelement <8 x float> %bnk.n1872, i64 1, !dbg !13
  %bnk.n1875 = extractelement <8 x float> %bnk.n1872, i64 2, !dbg !13
  %bnk.n1876 = extractelement <8 x float> %bnk.n1872, i64 3, !dbg !13
  %bnk.n1877 = select i1 %.not, float %bnk.n1874, float %bnk.n1873, !dbg !13
  %bnk.n1878 = bitcast float %bnk.n1877 to i32, !dbg !13
  %bnk.n1879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1878), !dbg !13
  %bnk.n1880 = bitcast i32 %bnk.n1879 to float, !dbg !13
  %bnk.n1881 = select i1 %.not, float %bnk.n1873, float %bnk.n1880, !dbg !13
  %bnk.n1882 = select i1 %.not, float %bnk.n1880, float %bnk.n1874, !dbg !13
  %bnk.n1883 = select i1 %.not, float %bnk.n1876, float %bnk.n1875, !dbg !13
  %bnk.n1884 = bitcast float %bnk.n1883 to i32, !dbg !13
  %bnk.n1885 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1884), !dbg !13
  %bnk.n1886 = bitcast i32 %bnk.n1885 to float, !dbg !13
  %bnk.n1887 = select i1 %.not, float %bnk.n1875, float %bnk.n1886, !dbg !13
  %bnk.n1888 = select i1 %.not, float %bnk.n1886, float %bnk.n1876, !dbg !13
  %bnk.n1889 = select i1 %.not36, float %bnk.n1887, float %bnk.n1881, !dbg !13
  %bnk.n1890 = bitcast float %bnk.n1889 to i32, !dbg !13
  %bnk.n1891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1890), !dbg !13
  %bnk.n1892 = bitcast i32 %bnk.n1891 to float, !dbg !13
  %bnk.n1893 = select i1 %.not36, float %bnk.n1881, float %bnk.n1892, !dbg !13
  %bnk.n1894 = select i1 %.not36, float %bnk.n1892, float %bnk.n1887, !dbg !13
  %bnk.n1895 = select i1 %.not36, float %bnk.n1888, float %bnk.n1882, !dbg !13
  %bnk.n1896 = bitcast float %bnk.n1895 to i32, !dbg !13
  %bnk.n1897 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1896), !dbg !13
  %bnk.n1898 = bitcast i32 %bnk.n1897 to float, !dbg !13
  %bnk.n1899 = select i1 %.not36, float %bnk.n1882, float %bnk.n1898, !dbg !13
  %bnk.n1900 = select i1 %.not36, float %bnk.n1898, float %bnk.n1888, !dbg !13
  %bnk.n1901 = extractvalue [2 x <8 x float>] %bnk.n206, 1, !dbg !13
  %bnk.n1902 = extractelement <8 x float> %bnk.n1901, i64 0, !dbg !13
  %bnk.n1903 = extractelement <8 x float> %bnk.n1901, i64 1, !dbg !13
  %bnk.n1904 = extractelement <8 x float> %bnk.n1901, i64 2, !dbg !13
  %bnk.n1905 = extractelement <8 x float> %bnk.n1901, i64 3, !dbg !13
  %bnk.n1906 = select i1 %.not, float %bnk.n1903, float %bnk.n1902, !dbg !13
  %bnk.n1907 = bitcast float %bnk.n1906 to i32, !dbg !13
  %bnk.n1908 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1907), !dbg !13
  %bnk.n1909 = bitcast i32 %bnk.n1908 to float, !dbg !13
  %bnk.n1910 = select i1 %.not, float %bnk.n1902, float %bnk.n1909, !dbg !13
  %bnk.n1911 = select i1 %.not, float %bnk.n1909, float %bnk.n1903, !dbg !13
  %bnk.n1912 = select i1 %.not, float %bnk.n1905, float %bnk.n1904, !dbg !13
  %bnk.n1913 = bitcast float %bnk.n1912 to i32, !dbg !13
  %bnk.n1914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1913), !dbg !13
  %bnk.n1915 = bitcast i32 %bnk.n1914 to float, !dbg !13
  %bnk.n1916 = select i1 %.not, float %bnk.n1904, float %bnk.n1915, !dbg !13
  %bnk.n1917 = select i1 %.not, float %bnk.n1915, float %bnk.n1905, !dbg !13
  %bnk.n1918 = select i1 %.not36, float %bnk.n1916, float %bnk.n1910, !dbg !13
  %bnk.n1919 = bitcast float %bnk.n1918 to i32, !dbg !13
  %bnk.n1920 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1919), !dbg !13
  %bnk.n1921 = select i1 %.not36, float %bnk.n1917, float %bnk.n1911, !dbg !13
  %bnk.n1922 = bitcast float %bnk.n1921 to i32, !dbg !13
  %bnk.n1923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1922), !dbg !13
  %bnk.n1924 = extractvalue [2 x <1 x float>] %bnk.n207, 1, !dbg !20
  %bnk.n1925 = shufflevector <1 x float> %bnk.n1924, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %bnk.n1926 = extractelement <8 x float> %bnk.n1814, i64 4, !dbg !13
  %bnk.n1927 = extractelement <8 x float> %bnk.n1814, i64 5, !dbg !13
  %bnk.n1928 = extractelement <8 x float> %bnk.n1814, i64 6, !dbg !13
  %bnk.n1929 = extractelement <8 x float> %bnk.n1814, i64 7, !dbg !13
  %bnk.n1930 = select i1 %.not, float %bnk.n1927, float %bnk.n1926, !dbg !13
  %bnk.n1931 = bitcast float %bnk.n1930 to i32, !dbg !13
  %bnk.n1932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1931), !dbg !13
  %bnk.n1933 = bitcast i32 %bnk.n1932 to float, !dbg !13
  %bnk.n1934 = select i1 %.not, float %bnk.n1926, float %bnk.n1933, !dbg !13
  %bnk.n1935 = select i1 %.not, float %bnk.n1933, float %bnk.n1927, !dbg !13
  %bnk.n1936 = select i1 %.not, float %bnk.n1929, float %bnk.n1928, !dbg !13
  %bnk.n1937 = bitcast float %bnk.n1936 to i32, !dbg !13
  %bnk.n1938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1937), !dbg !13
  %bnk.n1939 = bitcast i32 %bnk.n1938 to float, !dbg !13
  %bnk.n1940 = select i1 %.not, float %bnk.n1928, float %bnk.n1939, !dbg !13
  %bnk.n1941 = select i1 %.not, float %bnk.n1939, float %bnk.n1929, !dbg !13
  %bnk.n1942 = select i1 %.not36, float %bnk.n1940, float %bnk.n1934, !dbg !13
  %bnk.n1943 = bitcast float %bnk.n1942 to i32, !dbg !13
  %bnk.n1944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1943), !dbg !13
  %bnk.n1945 = bitcast i32 %bnk.n1944 to float, !dbg !13
  %bnk.n1946 = select i1 %.not36, float %bnk.n1934, float %bnk.n1945, !dbg !13
  %bnk.n1947 = select i1 %.not36, float %bnk.n1945, float %bnk.n1940, !dbg !13
  %bnk.n1948 = select i1 %.not36, float %bnk.n1941, float %bnk.n1935, !dbg !13
  %bnk.n1949 = bitcast float %bnk.n1948 to i32, !dbg !13
  %bnk.n1950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1949), !dbg !13
  %bnk.n1951 = bitcast i32 %bnk.n1950 to float, !dbg !13
  %bnk.n1952 = select i1 %.not36, float %bnk.n1935, float %bnk.n1951, !dbg !13
  %bnk.n1953 = select i1 %.not36, float %bnk.n1951, float %bnk.n1941, !dbg !13
  %bnk.n1954 = extractelement <8 x float> %bnk.n1843, i64 4, !dbg !13
  %bnk.n1955 = extractelement <8 x float> %bnk.n1843, i64 5, !dbg !13
  %bnk.n1956 = extractelement <8 x float> %bnk.n1843, i64 6, !dbg !13
  %bnk.n1957 = extractelement <8 x float> %bnk.n1843, i64 7, !dbg !13
  %bnk.n1958 = select i1 %.not, float %bnk.n1955, float %bnk.n1954, !dbg !13
  %bnk.n1959 = bitcast float %bnk.n1958 to i32, !dbg !13
  %bnk.n1960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1959), !dbg !13
  %bnk.n1961 = bitcast i32 %bnk.n1960 to float, !dbg !13
  %bnk.n1962 = select i1 %.not, float %bnk.n1954, float %bnk.n1961, !dbg !13
  %bnk.n1963 = select i1 %.not, float %bnk.n1961, float %bnk.n1955, !dbg !13
  %bnk.n1964 = select i1 %.not, float %bnk.n1957, float %bnk.n1956, !dbg !13
  %bnk.n1965 = bitcast float %bnk.n1964 to i32, !dbg !13
  %bnk.n1966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1965), !dbg !13
  %bnk.n1967 = bitcast i32 %bnk.n1966 to float, !dbg !13
  %bnk.n1968 = select i1 %.not, float %bnk.n1956, float %bnk.n1967, !dbg !13
  %bnk.n1969 = select i1 %.not, float %bnk.n1967, float %bnk.n1957, !dbg !13
  %bnk.n1970 = select i1 %.not36, float %bnk.n1968, float %bnk.n1962, !dbg !13
  %bnk.n1971 = bitcast float %bnk.n1970 to i32, !dbg !13
  %bnk.n1972 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1971), !dbg !13
  %bnk.n1973 = bitcast i32 %bnk.n1972 to float, !dbg !13
  %bnk.n1974 = select i1 %.not36, float %bnk.n1962, float %bnk.n1973, !dbg !13
  %bnk.n1975 = select i1 %.not36, float %bnk.n1973, float %bnk.n1968, !dbg !13
  %bnk.n1976 = select i1 %.not36, float %bnk.n1969, float %bnk.n1963, !dbg !13
  %bnk.n1977 = bitcast float %bnk.n1976 to i32, !dbg !13
  %bnk.n1978 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1977), !dbg !13
  %bnk.n1979 = bitcast i32 %bnk.n1978 to float, !dbg !13
  %bnk.n1980 = select i1 %.not36, float %bnk.n1963, float %bnk.n1979, !dbg !13
  %bnk.n1981 = select i1 %.not36, float %bnk.n1979, float %bnk.n1969, !dbg !13
  %bnk.n1982 = extractelement <8 x float> %bnk.n1872, i64 4, !dbg !13
  %bnk.n1983 = extractelement <8 x float> %bnk.n1872, i64 5, !dbg !13
  %bnk.n1984 = extractelement <8 x float> %bnk.n1872, i64 6, !dbg !13
  %bnk.n1985 = extractelement <8 x float> %bnk.n1872, i64 7, !dbg !13
  %bnk.n1986 = select i1 %.not, float %bnk.n1983, float %bnk.n1982, !dbg !13
  %bnk.n1987 = bitcast float %bnk.n1986 to i32, !dbg !13
  %bnk.n1988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1987), !dbg !13
  %bnk.n1989 = bitcast i32 %bnk.n1988 to float, !dbg !13
  %bnk.n1990 = select i1 %.not, float %bnk.n1982, float %bnk.n1989, !dbg !13
  %bnk.n1991 = select i1 %.not, float %bnk.n1989, float %bnk.n1983, !dbg !13
  %bnk.n1992 = select i1 %.not, float %bnk.n1985, float %bnk.n1984, !dbg !13
  %bnk.n1993 = bitcast float %bnk.n1992 to i32, !dbg !13
  %bnk.n1994 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1993), !dbg !13
  %bnk.n1995 = bitcast i32 %bnk.n1994 to float, !dbg !13
  %bnk.n1996 = select i1 %.not, float %bnk.n1984, float %bnk.n1995, !dbg !13
  %bnk.n1997 = select i1 %.not, float %bnk.n1995, float %bnk.n1985, !dbg !13
  %bnk.n1998 = select i1 %.not36, float %bnk.n1996, float %bnk.n1990, !dbg !13
  %bnk.n1999 = bitcast float %bnk.n1998 to i32, !dbg !13
  %bnk.n2000 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1999), !dbg !13
  %bnk.n2001 = bitcast i32 %bnk.n2000 to float, !dbg !13
  %bnk.n2002 = select i1 %.not36, float %bnk.n1990, float %bnk.n2001, !dbg !13
  %bnk.n2003 = select i1 %.not36, float %bnk.n2001, float %bnk.n1996, !dbg !13
  %bnk.n2004 = select i1 %.not36, float %bnk.n1997, float %bnk.n1991, !dbg !13
  %bnk.n2005 = bitcast float %bnk.n2004 to i32, !dbg !13
  %bnk.n2006 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n2005), !dbg !13
  %bnk.n2007 = bitcast i32 %bnk.n2006 to float, !dbg !13
  %bnk.n2008 = select i1 %.not36, float %bnk.n1991, float %bnk.n2007, !dbg !13
  %bnk.n2009 = select i1 %.not36, float %bnk.n2007, float %bnk.n1997, !dbg !13
  %bnk.n2010 = extractelement <8 x float> %bnk.n1901, i64 4, !dbg !13
  %bnk.n2011 = extractelement <8 x float> %bnk.n1901, i64 5, !dbg !13
  %bnk.n2012 = extractelement <8 x float> %bnk.n1901, i64 6, !dbg !13
  %bnk.n2013 = extractelement <8 x float> %bnk.n1901, i64 7, !dbg !13
  %bnk.n2014 = select i1 %.not, float %bnk.n2011, float %bnk.n2010, !dbg !13
  %bnk.n2015 = bitcast float %bnk.n2014 to i32, !dbg !13
  %bnk.n2016 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n2015), !dbg !13
  %bnk.n2017 = bitcast i32 %bnk.n2016 to float, !dbg !13
  %bnk.n2018 = select i1 %.not, float %bnk.n2010, float %bnk.n2017, !dbg !13
  %bnk.n2019 = select i1 %.not, float %bnk.n2017, float %bnk.n2011, !dbg !13
  %bnk.n2020 = select i1 %.not, float %bnk.n2013, float %bnk.n2012, !dbg !13
  %bnk.n2021 = bitcast float %bnk.n2020 to i32, !dbg !13
  %bnk.n2022 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n2021), !dbg !13
  %bnk.n2023 = bitcast i32 %bnk.n2022 to float, !dbg !13
  %bnk.n2024 = select i1 %.not, float %bnk.n2012, float %bnk.n2023, !dbg !13
  %bnk.n2025 = select i1 %.not, float %bnk.n2023, float %bnk.n2013, !dbg !13
  %bnk.n2026 = select i1 %.not36, float %bnk.n2024, float %bnk.n2018, !dbg !13
  %bnk.n2027 = bitcast float %bnk.n2026 to i32, !dbg !13
  %bnk.n2028 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n2027), !dbg !13
  %bnk.n2029 = select i1 %.not36, float %bnk.n2025, float %bnk.n2019, !dbg !13
  %bnk.n2030 = bitcast float %bnk.n2029 to i32, !dbg !13
  %bnk.n2031 = call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n2030), !dbg !13
  br i1 true, label %bnk.n2032, label %bnk.n2096, !dbg !22

bnk.n2032:                                             ; preds = %bnk.n1581
  %bnk.n2033 = bitcast i32 %bnk.n2028 to float, !dbg !13
  %bnk.n2034 = select i1 %.not36, float %bnk.n2018, float %bnk.n2033, !dbg !13
  %bnk.n2035 = insertelement <4 x float> poison, float %bnk.n2034, i64 0, !dbg !20
  %bnk.n2036 = bitcast i32 %bnk.n2031 to float, !dbg !13
  %bnk.n2037 = select i1 %.not36, float %bnk.n2019, float %bnk.n2036, !dbg !13
  %bnk.n2038 = insertelement <4 x float> %bnk.n2035, float %bnk.n2037, i64 1, !dbg !20
  %bnk.n2039 = select i1 %.not36, float %bnk.n2033, float %bnk.n2024, !dbg !13
  %bnk.n2040 = insertelement <4 x float> %bnk.n2038, float %bnk.n2039, i64 2, !dbg !20
  %bnk.n2041 = select i1 %.not36, float %bnk.n2036, float %bnk.n2025, !dbg !13
  %bnk.n2042 = insertelement <4 x float> %bnk.n2040, float %bnk.n2041, i64 3, !dbg !20
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %bnk.n1925, !dbg !20
  %normalized = fmul <4 x float> %bnk.n2042, %row.reciprocal, !dbg !20
  %bnk.n2043 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !21
  %bnk.n2044 = extractelement <4 x half> %bnk.n2043, i64 3, !dbg !21
  %bnk.n2045 = extractelement <4 x half> %bnk.n2043, i64 2, !dbg !21
  %bnk.n2046 = extractelement <4 x half> %bnk.n2043, i64 1, !dbg !21
  %bnk.n2047 = extractelement <4 x half> %bnk.n2043, i64 0, !dbg !21
  %bnk.n2048 = bitcast i32 %bnk.n1920 to float, !dbg !13
  %bnk.n2049 = select i1 %.not36, float %bnk.n1910, float %bnk.n2048, !dbg !13
  %bnk.n2050 = insertelement <4 x float> poison, float %bnk.n2049, i64 0, !dbg !20
  %bnk.n2051 = bitcast i32 %bnk.n1923 to float, !dbg !13
  %bnk.n2052 = select i1 %.not36, float %bnk.n1911, float %bnk.n2051, !dbg !13
  %bnk.n2053 = insertelement <4 x float> %bnk.n2050, float %bnk.n2052, i64 1, !dbg !20
  %bnk.n2054 = select i1 %.not36, float %bnk.n2048, float %bnk.n1916, !dbg !13
  %bnk.n2055 = insertelement <4 x float> %bnk.n2053, float %bnk.n2054, i64 2, !dbg !20
  %bnk.n2056 = select i1 %.not36, float %bnk.n2051, float %bnk.n1917, !dbg !13
  %bnk.n2057 = insertelement <4 x float> %bnk.n2055, float %bnk.n2056, i64 3, !dbg !20
  %normalized104 = fmul <4 x float> %bnk.n2057, %row.reciprocal, !dbg !20
  %bnk.n2058 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !21
  %bnk.n2059 = extractelement <4 x half> %bnk.n2058, i64 3, !dbg !21
  %bnk.n2060 = extractelement <4 x half> %bnk.n2058, i64 2, !dbg !21
  %bnk.n2061 = extractelement <4 x half> %bnk.n2058, i64 1, !dbg !21
  %bnk.n2062 = extractelement <4 x half> %bnk.n2058, i64 0, !dbg !21
  %bnk.n2063 = bitcast i32 %bnk.n1810 to float, !dbg !13
  %bnk.n2064 = select i1 %.not36, float %bnk.n1800, float %bnk.n2063, !dbg !13
  %bnk.n2065 = insertelement <4 x float> poison, float %bnk.n2064, i64 0, !dbg !20
  %bnk.n2066 = bitcast i32 %bnk.n1813 to float, !dbg !13
  %bnk.n2067 = select i1 %.not36, float %bnk.n1801, float %bnk.n2066, !dbg !13
  %bnk.n2068 = insertelement <4 x float> %bnk.n2065, float %bnk.n2067, i64 1, !dbg !20
  %bnk.n2069 = select i1 %.not36, float %bnk.n2063, float %bnk.n1806, !dbg !13
  %bnk.n2070 = insertelement <4 x float> %bnk.n2068, float %bnk.n2069, i64 2, !dbg !20
  %bnk.n2071 = select i1 %.not36, float %bnk.n2066, float %bnk.n1807, !dbg !13
  %bnk.n2072 = insertelement <4 x float> %bnk.n2070, float %bnk.n2071, i64 3, !dbg !20
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %bnk.n1707, !dbg !20
  %normalized106 = fmul <4 x float> %bnk.n2072, %row.reciprocal105, !dbg !20
  %bnk.n2073 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !21
  %bnk.n2074 = extractelement <4 x half> %bnk.n2073, i64 3, !dbg !21
  %bnk.n2075 = extractelement <4 x half> %bnk.n2073, i64 2, !dbg !21
  %bnk.n2076 = extractelement <4 x half> %bnk.n2073, i64 1, !dbg !21
  %bnk.n2077 = extractelement <4 x half> %bnk.n2073, i64 0, !dbg !21
  %bnk.n2078 = bitcast i32 %bnk.n1702 to float, !dbg !13
  %bnk.n2079 = select i1 %.not36, float %bnk.n1692, float %bnk.n2078, !dbg !13
  %bnk.n2080 = insertelement <4 x float> poison, float %bnk.n2079, i64 0, !dbg !20
  %bnk.n2081 = bitcast i32 %bnk.n1705 to float, !dbg !13
  %bnk.n2082 = select i1 %.not36, float %bnk.n1693, float %bnk.n2081, !dbg !13
  %bnk.n2083 = insertelement <4 x float> %bnk.n2080, float %bnk.n2082, i64 1, !dbg !20
  %bnk.n2084 = select i1 %.not36, float %bnk.n2078, float %bnk.n1698, !dbg !13
  %bnk.n2085 = insertelement <4 x float> %bnk.n2083, float %bnk.n2084, i64 2, !dbg !20
  %bnk.n2086 = select i1 %.not36, float %bnk.n2081, float %bnk.n1699, !dbg !13
  %bnk.n2087 = insertelement <4 x float> %bnk.n2085, float %bnk.n2086, i64 3, !dbg !20
  %normalized107 = fmul <4 x float> %bnk.n2087, %row.reciprocal105, !dbg !20
  %bnk.n2088 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !21
  %bnk.n2089 = extractelement <4 x half> %bnk.n2088, i64 3, !dbg !21
  %bnk.n2090 = extractelement <4 x half> %bnk.n2088, i64 2, !dbg !21
  %bnk.n2091 = extractelement <4 x half> %bnk.n2088, i64 1, !dbg !21
  %bnk.n2092 = extractelement <4 x half> %bnk.n2088, i64 0, !dbg !21
  %bnk.n2093 = add nsw i64 %bnk.n16, %bnk.n22, !dbg !22
  %bnk.n2094 = shl i64 %bnk.n2093, 19, !dbg !22
  %bnk.n2095 = and i64 %bnk.n2094, 16252928, !dbg !22
  %address.thread934 = call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !8
  %address.local936 = add i64 %bnk.n25, %address.local935, !dbg !22
  %address.local937 = lshr i64 %address.local935, 6, !dbg !11
  %address.local938 = shl nuw nsw i64 %address.local937, 4, !dbg !22
  %address.local939 = add i64 %address.local936, %address.local938, !dbg !22
  %address.local940 = and i64 %address.local935, 240, !dbg !22
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !22
  %address.local942 = add i64 %address.local941, %bnk.n26, !dbg !22
  %address.local943 = shl nuw nsw i64 %address.local942, 7, !dbg !22
  %address.local944 = add nuw nsw i64 %bnk.n2095, %address.local943, !dbg !22
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local946 = lshr i64 %address.local935, 2, !dbg !11
  %address.local947 = and i64 %address.local946, 12, !dbg !11
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !22
  store half %bnk.n2092, ptr addrspace(1) %address.local948, align 2, !dbg !22
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !22
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local949, i64 %address.local950, !dbg !22
  store half %bnk.n2091, ptr addrspace(1) %address.local951, align 2, !dbg !22
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !22
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local952, i64 %address.local953, !dbg !22
  store half %bnk.n2090, ptr addrspace(1) %address.local954, align 2, !dbg !22
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !22
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local955, i64 %address.local956, !dbg !22
  store half %bnk.n2089, ptr addrspace(1) %address.local957, align 2, !dbg !22
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !22
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local958, i64 %address.local959, !dbg !22
  store half %bnk.n2077, ptr addrspace(1) %address.local960, align 2, !dbg !22
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !22
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local961, i64 %address.local962, !dbg !22
  store half %bnk.n2076, ptr addrspace(1) %address.local963, align 2, !dbg !22
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !22
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local964, i64 %address.local965, !dbg !22
  store half %bnk.n2075, ptr addrspace(1) %address.local966, align 2, !dbg !22
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !22
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local967, i64 %address.local968, !dbg !22
  store half %bnk.n2074, ptr addrspace(1) %address.local969, align 2, !dbg !22
  %address.local970 = shl i64 %address.local942, 7, !dbg !22
  %address.local971 = add i64 %address.local970, 8192, !dbg !22
  %address.local972 = add nuw nsw i64 %bnk.n2095, %address.local971, !dbg !22
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !22
  store half %bnk.n2062, ptr addrspace(1) %address.local974, align 2, !dbg !22
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local975, i64 %address.local950, !dbg !22
  store half %bnk.n2061, ptr addrspace(1) %address.local976, align 2, !dbg !22
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local953, !dbg !22
  store half %bnk.n2060, ptr addrspace(1) %address.local978, align 2, !dbg !22
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local979, i64 %address.local956, !dbg !22
  store half %bnk.n2059, ptr addrspace(1) %address.local980, align 2, !dbg !22
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local981, i64 %address.local959, !dbg !22
  store half %bnk.n2047, ptr addrspace(1) %address.local982, align 2, !dbg !22
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local962, !dbg !22
  store half %bnk.n2046, ptr addrspace(1) %address.local984, align 2, !dbg !22
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local965, !dbg !22
  store half %bnk.n2045, ptr addrspace(1) %address.local986, align 2, !dbg !22
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local987, i64 %address.local968, !dbg !22
  store half %bnk.n2044, ptr addrspace(1) %address.local988, align 2, !dbg !22
  br label %bnk.n2096, !dbg !22

bnk.n2096:                                             ; preds = %bnk.n2032, %bnk.n1581
  br i1 true, label %bnk.n2097, label %bnk.n2135, !dbg !22

bnk.n2097:                                             ; preds = %bnk.n2096
  %bnk.n2098 = insertelement <4 x float> poison, float %bnk.n2002, i64 0, !dbg !20
  %bnk.n2099 = insertelement <4 x float> %bnk.n2098, float %bnk.n2008, i64 1, !dbg !20
  %bnk.n2100 = insertelement <4 x float> %bnk.n2099, float %bnk.n2003, i64 2, !dbg !20
  %bnk.n2101 = insertelement <4 x float> %bnk.n2100, float %bnk.n2009, i64 3, !dbg !20
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %bnk.n1925, !dbg !20
  %normalized109 = fmul <4 x float> %bnk.n2101, %row.reciprocal108, !dbg !20
  %bnk.n2102 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !21
  %bnk.n2103 = extractelement <4 x half> %bnk.n2102, i64 3, !dbg !21
  %bnk.n2104 = extractelement <4 x half> %bnk.n2102, i64 2, !dbg !21
  %bnk.n2105 = extractelement <4 x half> %bnk.n2102, i64 1, !dbg !21
  %bnk.n2106 = extractelement <4 x half> %bnk.n2102, i64 0, !dbg !21
  %bnk.n2107 = insertelement <4 x float> poison, float %bnk.n1893, i64 0, !dbg !20
  %bnk.n2108 = insertelement <4 x float> %bnk.n2107, float %bnk.n1899, i64 1, !dbg !20
  %bnk.n2109 = insertelement <4 x float> %bnk.n2108, float %bnk.n1894, i64 2, !dbg !20
  %bnk.n2110 = insertelement <4 x float> %bnk.n2109, float %bnk.n1900, i64 3, !dbg !20
  %normalized110 = fmul <4 x float> %bnk.n2110, %row.reciprocal108, !dbg !20
  %bnk.n2111 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !21
  %bnk.n2112 = extractelement <4 x half> %bnk.n2111, i64 3, !dbg !21
  %bnk.n2113 = extractelement <4 x half> %bnk.n2111, i64 2, !dbg !21
  %bnk.n2114 = extractelement <4 x half> %bnk.n2111, i64 1, !dbg !21
  %bnk.n2115 = extractelement <4 x half> %bnk.n2111, i64 0, !dbg !21
  %bnk.n2116 = insertelement <4 x float> poison, float %bnk.n1784, i64 0, !dbg !20
  %bnk.n2117 = insertelement <4 x float> %bnk.n2116, float %bnk.n1790, i64 1, !dbg !20
  %bnk.n2118 = insertelement <4 x float> %bnk.n2117, float %bnk.n1785, i64 2, !dbg !20
  %bnk.n2119 = insertelement <4 x float> %bnk.n2118, float %bnk.n1791, i64 3, !dbg !20
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %bnk.n1707, !dbg !20
  %normalized112 = fmul <4 x float> %bnk.n2119, %row.reciprocal111, !dbg !20
  %bnk.n2120 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !21
  %bnk.n2121 = extractelement <4 x half> %bnk.n2120, i64 3, !dbg !21
  %bnk.n2122 = extractelement <4 x half> %bnk.n2120, i64 2, !dbg !21
  %bnk.n2123 = extractelement <4 x half> %bnk.n2120, i64 1, !dbg !21
  %bnk.n2124 = extractelement <4 x half> %bnk.n2120, i64 0, !dbg !21
  %bnk.n2125 = insertelement <4 x float> poison, float %bnk.n1675, i64 0, !dbg !20
  %bnk.n2126 = insertelement <4 x float> %bnk.n2125, float %bnk.n1681, i64 1, !dbg !20
  %bnk.n2127 = insertelement <4 x float> %bnk.n2126, float %bnk.n1676, i64 2, !dbg !20
  %bnk.n2128 = insertelement <4 x float> %bnk.n2127, float %bnk.n1682, i64 3, !dbg !20
  %normalized113 = fmul <4 x float> %bnk.n2128, %row.reciprocal111, !dbg !20
  %bnk.n2129 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !21
  %bnk.n2130 = extractelement <4 x half> %bnk.n2129, i64 3, !dbg !21
  %bnk.n2131 = extractelement <4 x half> %bnk.n2129, i64 2, !dbg !21
  %bnk.n2132 = extractelement <4 x half> %bnk.n2129, i64 1, !dbg !21
  %bnk.n2133 = extractelement <4 x half> %bnk.n2129, i64 0, !dbg !21
  %bnk.n2134 = shl nuw nsw i64 %bnk.n24, 19, !dbg !22
  %address.thread989 = call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local990 = zext nneg i32 %address.thread989 to i64, !dbg !8
  %address.local991 = add i64 %bnk.n25, %address.local990, !dbg !22
  %address.local992 = lshr i64 %address.local990, 6, !dbg !11
  %address.local993 = shl nuw nsw i64 %address.local992, 4, !dbg !22
  %address.local994 = add i64 %address.local991, %address.local993, !dbg !22
  %address.local995 = and i64 %address.local990, 240, !dbg !22
  %address.local996 = sub i64 %address.local994, %address.local995, !dbg !22
  %address.local997 = add i64 %address.local996, %bnk.n26, !dbg !22
  %address.local998 = shl nuw nsw i64 %address.local997, 7, !dbg !22
  %address.local999 = add nuw nsw i64 %bnk.n2134, %address.local998, !dbg !22
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1001 = lshr i64 %address.local990, 2, !dbg !11
  %address.local1002 = and i64 %address.local1001, 12, !dbg !11
  %address.local1003 = or disjoint i64 %address.local1002, 32, !dbg !22
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local1003, !dbg !22
  store half %bnk.n2133, ptr addrspace(1) %address.local1004, align 2, !dbg !22
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1006 = or disjoint i64 %address.local1002, 33, !dbg !22
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1005, i64 %address.local1006, !dbg !22
  store half %bnk.n2132, ptr addrspace(1) %address.local1007, align 2, !dbg !22
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1009 = or disjoint i64 %address.local1002, 34, !dbg !22
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local1009, !dbg !22
  store half %bnk.n2131, ptr addrspace(1) %address.local1010, align 2, !dbg !22
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1012 = or disjoint i64 %address.local1002, 35, !dbg !22
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local1012, !dbg !22
  store half %bnk.n2130, ptr addrspace(1) %address.local1013, align 2, !dbg !22
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1015 = or disjoint i64 %address.local1002, 48, !dbg !22
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local1015, !dbg !22
  store half %bnk.n2124, ptr addrspace(1) %address.local1016, align 2, !dbg !22
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1018 = or disjoint i64 %address.local1002, 49, !dbg !22
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local1018, !dbg !22
  store half %bnk.n2123, ptr addrspace(1) %address.local1019, align 2, !dbg !22
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1021 = or disjoint i64 %address.local1002, 50, !dbg !22
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1021, !dbg !22
  store half %bnk.n2122, ptr addrspace(1) %address.local1022, align 2, !dbg !22
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1024 = or i64 %address.local1001, 51, !dbg !22
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1024, !dbg !22
  store half %bnk.n2121, ptr addrspace(1) %address.local1025, align 2, !dbg !22
  %address.local1026 = shl i64 %address.local997, 7, !dbg !22
  %address.local1027 = add i64 %address.local1026, 8192, !dbg !22
  %address.local1028 = add nuw nsw i64 %bnk.n2134, %address.local1027, !dbg !22
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1029, i64 %address.local1003, !dbg !22
  store half %bnk.n2115, ptr addrspace(1) %address.local1030, align 2, !dbg !22
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1006, !dbg !22
  store half %bnk.n2114, ptr addrspace(1) %address.local1032, align 2, !dbg !22
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1009, !dbg !22
  store half %bnk.n2113, ptr addrspace(1) %address.local1034, align 2, !dbg !22
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1012, !dbg !22
  store half %bnk.n2112, ptr addrspace(1) %address.local1036, align 2, !dbg !22
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1015, !dbg !22
  store half %bnk.n2106, ptr addrspace(1) %address.local1038, align 2, !dbg !22
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1018, !dbg !22
  store half %bnk.n2105, ptr addrspace(1) %address.local1040, align 2, !dbg !22
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1021, !dbg !22
  store half %bnk.n2104, ptr addrspace(1) %address.local1042, align 2, !dbg !22
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1024, !dbg !22
  store half %bnk.n2103, ptr addrspace(1) %address.local1044, align 2, !dbg !22
  br label %bnk.n2135, !dbg !22

bnk.n2135:                                             ; preds = %bnk.n2097, %bnk.n2096
  br i1 true, label %bnk.n2136, label %bnk.n2174, !dbg !22

bnk.n2136:                                             ; preds = %bnk.n2135
  %bnk.n2137 = insertelement <4 x float> poison, float %bnk.n1974, i64 0, !dbg !20
  %bnk.n2138 = insertelement <4 x float> %bnk.n2137, float %bnk.n1980, i64 1, !dbg !20
  %bnk.n2139 = insertelement <4 x float> %bnk.n2138, float %bnk.n1975, i64 2, !dbg !20
  %bnk.n2140 = insertelement <4 x float> %bnk.n2139, float %bnk.n1981, i64 3, !dbg !20
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %bnk.n1925, !dbg !20
  %normalized115 = fmul <4 x float> %bnk.n2140, %row.reciprocal114, !dbg !20
  %bnk.n2141 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !21
  %bnk.n2142 = extractelement <4 x half> %bnk.n2141, i64 3, !dbg !21
  %bnk.n2143 = extractelement <4 x half> %bnk.n2141, i64 2, !dbg !21
  %bnk.n2144 = extractelement <4 x half> %bnk.n2141, i64 1, !dbg !21
  %bnk.n2145 = extractelement <4 x half> %bnk.n2141, i64 0, !dbg !21
  %bnk.n2146 = insertelement <4 x float> poison, float %bnk.n1864, i64 0, !dbg !20
  %bnk.n2147 = insertelement <4 x float> %bnk.n2146, float %bnk.n1870, i64 1, !dbg !20
  %bnk.n2148 = insertelement <4 x float> %bnk.n2147, float %bnk.n1865, i64 2, !dbg !20
  %bnk.n2149 = insertelement <4 x float> %bnk.n2148, float %bnk.n1871, i64 3, !dbg !20
  %normalized116 = fmul <4 x float> %bnk.n2149, %row.reciprocal114, !dbg !20
  %bnk.n2150 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !21
  %bnk.n2151 = extractelement <4 x half> %bnk.n2150, i64 3, !dbg !21
  %bnk.n2152 = extractelement <4 x half> %bnk.n2150, i64 2, !dbg !21
  %bnk.n2153 = extractelement <4 x half> %bnk.n2150, i64 1, !dbg !21
  %bnk.n2154 = extractelement <4 x half> %bnk.n2150, i64 0, !dbg !21
  %bnk.n2155 = insertelement <4 x float> poison, float %bnk.n1756, i64 0, !dbg !20
  %bnk.n2156 = insertelement <4 x float> %bnk.n2155, float %bnk.n1762, i64 1, !dbg !20
  %bnk.n2157 = insertelement <4 x float> %bnk.n2156, float %bnk.n1757, i64 2, !dbg !20
  %bnk.n2158 = insertelement <4 x float> %bnk.n2157, float %bnk.n1763, i64 3, !dbg !20
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %bnk.n1707, !dbg !20
  %normalized118 = fmul <4 x float> %bnk.n2158, %row.reciprocal117, !dbg !20
  %bnk.n2159 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !21
  %bnk.n2160 = extractelement <4 x half> %bnk.n2159, i64 3, !dbg !21
  %bnk.n2161 = extractelement <4 x half> %bnk.n2159, i64 2, !dbg !21
  %bnk.n2162 = extractelement <4 x half> %bnk.n2159, i64 1, !dbg !21
  %bnk.n2163 = extractelement <4 x half> %bnk.n2159, i64 0, !dbg !21
  %bnk.n2164 = insertelement <4 x float> poison, float %bnk.n1646, i64 0, !dbg !20
  %bnk.n2165 = insertelement <4 x float> %bnk.n2164, float %bnk.n1652, i64 1, !dbg !20
  %bnk.n2166 = insertelement <4 x float> %bnk.n2165, float %bnk.n1647, i64 2, !dbg !20
  %bnk.n2167 = insertelement <4 x float> %bnk.n2166, float %bnk.n1653, i64 3, !dbg !20
  %normalized119 = fmul <4 x float> %bnk.n2167, %row.reciprocal117, !dbg !20
  %bnk.n2168 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !21
  %bnk.n2169 = extractelement <4 x half> %bnk.n2168, i64 3, !dbg !21
  %bnk.n2170 = extractelement <4 x half> %bnk.n2168, i64 2, !dbg !21
  %bnk.n2171 = extractelement <4 x half> %bnk.n2168, i64 1, !dbg !21
  %bnk.n2172 = extractelement <4 x half> %bnk.n2168, i64 0, !dbg !21
  %bnk.n2173 = shl nuw nsw i64 %bnk.n24, 19, !dbg !22
  %address.thread1045 = call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local1046 = zext nneg i32 %address.thread1045 to i64, !dbg !8
  %address.local1047 = add i64 %bnk.n25, %address.local1046, !dbg !22
  %address.local1048 = lshr i64 %address.local1046, 6, !dbg !11
  %address.local1049 = shl nuw nsw i64 %address.local1048, 4, !dbg !22
  %address.local1050 = add i64 %address.local1047, %address.local1049, !dbg !22
  %address.local1051 = and i64 %address.local1046, 240, !dbg !22
  %address.local1052 = sub i64 %address.local1050, %address.local1051, !dbg !22
  %address.local1053 = add i64 %address.local1052, %bnk.n26, !dbg !22
  %address.local1054 = shl nuw nsw i64 %address.local1053, 7, !dbg !22
  %address.local1055 = add nuw nsw i64 %bnk.n2173, %address.local1054, !dbg !22
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1057 = lshr i64 %address.local1046, 2, !dbg !11
  %address.local1058 = and i64 %address.local1057, 12, !dbg !11
  %address.local1059 = or disjoint i64 %address.local1058, 64, !dbg !22
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1059, !dbg !22
  store half %bnk.n2172, ptr addrspace(1) %address.local1060, align 2, !dbg !22
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1062 = or disjoint i64 %address.local1058, 65, !dbg !22
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1061, i64 %address.local1062, !dbg !22
  store half %bnk.n2171, ptr addrspace(1) %address.local1063, align 2, !dbg !22
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1065 = or disjoint i64 %address.local1058, 66, !dbg !22
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1064, i64 %address.local1065, !dbg !22
  store half %bnk.n2170, ptr addrspace(1) %address.local1066, align 2, !dbg !22
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1068 = or disjoint i64 %address.local1058, 67, !dbg !22
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1068, !dbg !22
  store half %bnk.n2169, ptr addrspace(1) %address.local1069, align 2, !dbg !22
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1071 = or disjoint i64 %address.local1058, 80, !dbg !22
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1071, !dbg !22
  store half %bnk.n2163, ptr addrspace(1) %address.local1072, align 2, !dbg !22
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1074 = or disjoint i64 %address.local1058, 81, !dbg !22
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1074, !dbg !22
  store half %bnk.n2162, ptr addrspace(1) %address.local1075, align 2, !dbg !22
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1077 = or disjoint i64 %address.local1058, 82, !dbg !22
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1076, i64 %address.local1077, !dbg !22
  store half %bnk.n2161, ptr addrspace(1) %address.local1078, align 2, !dbg !22
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1080 = or disjoint i64 %address.local1058, 83, !dbg !22
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1080, !dbg !22
  store half %bnk.n2160, ptr addrspace(1) %address.local1081, align 2, !dbg !22
  %address.local1082 = shl i64 %address.local1053, 7, !dbg !22
  %address.local1083 = add i64 %address.local1082, 8192, !dbg !22
  %address.local1084 = add nuw nsw i64 %bnk.n2173, %address.local1083, !dbg !22
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1085, i64 %address.local1059, !dbg !22
  store half %bnk.n2154, ptr addrspace(1) %address.local1086, align 2, !dbg !22
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1087, i64 %address.local1062, !dbg !22
  store half %bnk.n2153, ptr addrspace(1) %address.local1088, align 2, !dbg !22
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1065, !dbg !22
  store half %bnk.n2152, ptr addrspace(1) %address.local1090, align 2, !dbg !22
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1091, i64 %address.local1068, !dbg !22
  store half %bnk.n2151, ptr addrspace(1) %address.local1092, align 2, !dbg !22
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1071, !dbg !22
  store half %bnk.n2145, ptr addrspace(1) %address.local1094, align 2, !dbg !22
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1074, !dbg !22
  store half %bnk.n2144, ptr addrspace(1) %address.local1096, align 2, !dbg !22
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1077, !dbg !22
  store half %bnk.n2143, ptr addrspace(1) %address.local1098, align 2, !dbg !22
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1080, !dbg !22
  store half %bnk.n2142, ptr addrspace(1) %address.local1100, align 2, !dbg !22
  br label %bnk.n2174, !dbg !22

bnk.n2174:                                             ; preds = %bnk.n2136, %bnk.n2135
  br i1 true, label %bnk.n2175, label %bnk.n2213, !dbg !22

bnk.n2175:                                             ; preds = %bnk.n2174
  %bnk.n2176 = insertelement <4 x float> poison, float %bnk.n1946, i64 0, !dbg !20
  %bnk.n2177 = insertelement <4 x float> %bnk.n2176, float %bnk.n1952, i64 1, !dbg !20
  %bnk.n2178 = insertelement <4 x float> %bnk.n2177, float %bnk.n1947, i64 2, !dbg !20
  %bnk.n2179 = insertelement <4 x float> %bnk.n2178, float %bnk.n1953, i64 3, !dbg !20
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %bnk.n1925, !dbg !20
  %normalized121 = fmul <4 x float> %bnk.n2179, %row.reciprocal120, !dbg !20
  %bnk.n2180 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !21
  %bnk.n2181 = extractelement <4 x half> %bnk.n2180, i64 3, !dbg !21
  %bnk.n2182 = extractelement <4 x half> %bnk.n2180, i64 2, !dbg !21
  %bnk.n2183 = extractelement <4 x half> %bnk.n2180, i64 1, !dbg !21
  %bnk.n2184 = extractelement <4 x half> %bnk.n2180, i64 0, !dbg !21
  %bnk.n2185 = insertelement <4 x float> poison, float %bnk.n1835, i64 0, !dbg !20
  %bnk.n2186 = insertelement <4 x float> %bnk.n2185, float %bnk.n1841, i64 1, !dbg !20
  %bnk.n2187 = insertelement <4 x float> %bnk.n2186, float %bnk.n1836, i64 2, !dbg !20
  %bnk.n2188 = insertelement <4 x float> %bnk.n2187, float %bnk.n1842, i64 3, !dbg !20
  %normalized122 = fmul <4 x float> %bnk.n2188, %row.reciprocal120, !dbg !20
  %bnk.n2189 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !21
  %bnk.n2190 = extractelement <4 x half> %bnk.n2189, i64 3, !dbg !21
  %bnk.n2191 = extractelement <4 x half> %bnk.n2189, i64 2, !dbg !21
  %bnk.n2192 = extractelement <4 x half> %bnk.n2189, i64 1, !dbg !21
  %bnk.n2193 = extractelement <4 x half> %bnk.n2189, i64 0, !dbg !21
  %bnk.n2194 = insertelement <4 x float> poison, float %bnk.n1728, i64 0, !dbg !20
  %bnk.n2195 = insertelement <4 x float> %bnk.n2194, float %bnk.n1734, i64 1, !dbg !20
  %bnk.n2196 = insertelement <4 x float> %bnk.n2195, float %bnk.n1729, i64 2, !dbg !20
  %bnk.n2197 = insertelement <4 x float> %bnk.n2196, float %bnk.n1735, i64 3, !dbg !20
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %bnk.n1707, !dbg !20
  %normalized124 = fmul <4 x float> %bnk.n2197, %row.reciprocal123, !dbg !20
  %bnk.n2198 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !21
  %bnk.n2199 = extractelement <4 x half> %bnk.n2198, i64 3, !dbg !21
  %bnk.n2200 = extractelement <4 x half> %bnk.n2198, i64 2, !dbg !21
  %bnk.n2201 = extractelement <4 x half> %bnk.n2198, i64 1, !dbg !21
  %bnk.n2202 = extractelement <4 x half> %bnk.n2198, i64 0, !dbg !21
  %bnk.n2203 = insertelement <4 x float> poison, float %bnk.n1617, i64 0, !dbg !20
  %bnk.n2204 = insertelement <4 x float> %bnk.n2203, float %bnk.n1623, i64 1, !dbg !20
  %bnk.n2205 = insertelement <4 x float> %bnk.n2204, float %bnk.n1618, i64 2, !dbg !20
  %bnk.n2206 = insertelement <4 x float> %bnk.n2205, float %bnk.n1624, i64 3, !dbg !20
  %normalized125 = fmul <4 x float> %bnk.n2206, %row.reciprocal123, !dbg !20
  %bnk.n2207 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !21
  %bnk.n2208 = extractelement <4 x half> %bnk.n2207, i64 3, !dbg !21
  %bnk.n2209 = extractelement <4 x half> %bnk.n2207, i64 2, !dbg !21
  %bnk.n2210 = extractelement <4 x half> %bnk.n2207, i64 1, !dbg !21
  %bnk.n2211 = extractelement <4 x half> %bnk.n2207, i64 0, !dbg !21
  %bnk.n2212 = shl nuw nsw i64 %bnk.n24, 19, !dbg !22
  %address.thread1101 = call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local1102 = zext nneg i32 %address.thread1101 to i64, !dbg !8
  %address.local1103 = add i64 %bnk.n25, %address.local1102, !dbg !22
  %address.local1104 = lshr i64 %address.local1102, 6, !dbg !11
  %address.local1105 = shl nuw nsw i64 %address.local1104, 4, !dbg !22
  %address.local1106 = add i64 %address.local1103, %address.local1105, !dbg !22
  %address.local1107 = and i64 %address.local1102, 240, !dbg !22
  %address.local1108 = sub i64 %address.local1106, %address.local1107, !dbg !22
  %address.local1109 = add i64 %address.local1108, %bnk.n26, !dbg !22
  %address.local1110 = shl nuw nsw i64 %address.local1109, 7, !dbg !22
  %address.local1111 = add nuw nsw i64 %bnk.n2212, %address.local1110, !dbg !22
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1113 = lshr i64 %address.local1102, 2, !dbg !11
  %address.local1114 = and i64 %address.local1113, 12, !dbg !11
  %address.local1115 = or disjoint i64 %address.local1114, 96, !dbg !22
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1115, !dbg !22
  store half %bnk.n2211, ptr addrspace(1) %address.local1116, align 2, !dbg !22
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1118 = or disjoint i64 %address.local1114, 97, !dbg !22
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1118, !dbg !22
  store half %bnk.n2210, ptr addrspace(1) %address.local1119, align 2, !dbg !22
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1121 = or disjoint i64 %address.local1114, 98, !dbg !22
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1121, !dbg !22
  store half %bnk.n2209, ptr addrspace(1) %address.local1122, align 2, !dbg !22
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1124 = or disjoint i64 %address.local1114, 99, !dbg !22
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1124, !dbg !22
  store half %bnk.n2208, ptr addrspace(1) %address.local1125, align 2, !dbg !22
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1127 = or disjoint i64 %address.local1114, 112, !dbg !22
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1127, !dbg !22
  store half %bnk.n2202, ptr addrspace(1) %address.local1128, align 2, !dbg !22
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1130 = or disjoint i64 %address.local1114, 113, !dbg !22
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1130, !dbg !22
  store half %bnk.n2201, ptr addrspace(1) %address.local1131, align 2, !dbg !22
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1133 = or disjoint i64 %address.local1114, 114, !dbg !22
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1132, i64 %address.local1133, !dbg !22
  store half %bnk.n2200, ptr addrspace(1) %address.local1134, align 2, !dbg !22
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1136 = or i64 %address.local1113, 115, !dbg !22
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1136, !dbg !22
  store half %bnk.n2199, ptr addrspace(1) %address.local1137, align 2, !dbg !22
  %address.local1138 = shl i64 %address.local1109, 7, !dbg !22
  %address.local1139 = add i64 %address.local1138, 8192, !dbg !22
  %address.local1140 = add nuw nsw i64 %bnk.n2212, %address.local1139, !dbg !22
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1115, !dbg !22
  store half %bnk.n2193, ptr addrspace(1) %address.local1142, align 2, !dbg !22
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1118, !dbg !22
  store half %bnk.n2192, ptr addrspace(1) %address.local1144, align 2, !dbg !22
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1121, !dbg !22
  store half %bnk.n2191, ptr addrspace(1) %address.local1146, align 2, !dbg !22
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1124, !dbg !22
  store half %bnk.n2190, ptr addrspace(1) %address.local1148, align 2, !dbg !22
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1127, !dbg !22
  store half %bnk.n2184, ptr addrspace(1) %address.local1150, align 2, !dbg !22
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1130, !dbg !22
  store half %bnk.n2183, ptr addrspace(1) %address.local1152, align 2, !dbg !22
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1133, !dbg !22
  store half %bnk.n2182, ptr addrspace(1) %address.local1154, align 2, !dbg !22
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1136, !dbg !22
  store half %bnk.n2181, ptr addrspace(1) %address.local1156, align 2, !dbg !22
  br label %bnk.n2213, !dbg !22

bnk.n2213:                                             ; preds = %bnk.n2175, %bnk.n2174
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
!13 = !DILocation(line: 56, column: 24, scope: !4)
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
