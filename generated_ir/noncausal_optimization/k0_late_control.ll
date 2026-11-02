; ModuleID = '/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD_noncasual.mlir'
source_filename = "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD_noncasual.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
  %nc.n5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %nc.n6 = sext i32 %nc.n5 to i64, !dbg !5
  %nc.n7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %nc.n8 = sext i32 %nc.n7 to i64, !dbg !5
  %nc.n9 = shl nsw i64 %nc.n8, 5, !dbg !5
  %nc.n10 = add nsw i64 %nc.n9, %nc.n6, !dbg !5
  %nc.n11 = lshr i64 %nc.n10, 5, !dbg !5
  %nc.n12 = and i64 %nc.n11, 576460752303423484, !dbg !5
  %nc.n13 = and i64 %nc.n6, 3, !dbg !5
  %nc.n14 = or disjoint i64 %nc.n12, %nc.n13, !dbg !5
  %nc.n15 = sub nsw i64 31, %nc.n14, !dbg !5
  %nc.n16 = lshr i64 %nc.n10, 2, !dbg !5
  %nc.n17 = icmp ugt i64 %nc.n10, 1023, !dbg !8
  %nc.n18 = add nsw i64 %nc.n14, -32, !dbg !8
  %nc.n19 = select i1 %nc.n17, i64 %nc.n18, i64 %nc.n15, !dbg !8
  %nc.n20 = sdiv i64 %nc.n19, 32, !dbg !8
  %nc.n21 = sext i1 %nc.n17 to i64, !dbg !8
  %nc.n22 = xor i64 %nc.n20, %nc.n21, !dbg !8
  %nc.n23 = add nsw i64 %nc.n16, %nc.n22, !dbg !8
  %nc.n24 = and i64 %nc.n23, 31, !dbg !8
  %nc.n25 = shl nsw i64 %nc.n15, 7, !dbg !8
  %nc.n26 = mul nsw i64 %nc.n22, -4096, !dbg !8
  %nc.n27 = add i64 %nc.n25, %nc.n26, !dbg !8
  %nc.n28 = shl nuw nsw i64 %nc.n24, 19, !dbg !8
  %nc.n29 = shl nsw i64 %nc.n27, 7, !dbg !8
  %nc.n30 = add i64 %nc.n28, %nc.n29, !dbg !8
  %nc.n31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !32
  %nc.n32 = zext nneg i32 %nc.n31 to i64, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #9, !dbg !8, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !8
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !8
  %nc.n33 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %nc.n34 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %nc.n30, %address.local413, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %nc.n35 = insertelement <2 x i64> poison, i64 %nc.n34, i64 0, !dbg !8
  %nc.n36 = bitcast <2 x i64> %nc.n35 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %nc.n36, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n33), !dbg !8
  %nc.n37 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !8
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !8
  %nc.n38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !8
  %address.local421 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local422 = and i64 %address.local421, 16256, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local422, %address.local424, !dbg !8
  %nc.n39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1400 = or disjoint i64 %nc.n30, %address.local425, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %nc.n40 = insertelement <2 x i64> poison, i64 %nc.n39, i64 0, !dbg !8
  %nc.n41 = bitcast <2 x i64> %nc.n40 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1394 = shufflevector <4 x i32> %nc.n41, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n38), !dbg !8
  %nc.n42 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !8
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !8
  %nc.n43 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !8
  %nc.n44 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %nc.n45 = or disjoint i64 %nc.n30, 32, !dbg !8
  %nc.n46 = or disjoint i64 %address.local413, %nc.n45, !dbg !8
  %.tr = trunc i64 %nc.n46 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %nc.n47 = insertelement <2 x i64> poison, i64 %nc.n44, i64 0, !dbg !8
  %nc.n48 = bitcast <2 x i64> %nc.n47 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1391 = shufflevector <4 x i32> %nc.n48, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n43), !dbg !8
  %nc.n49 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1391) #10, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !8
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !8
  %nc.n50 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !8
  %nc.n51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %nc.n52 = or disjoint i64 %nc.n30, 32, !dbg !8
  %nc.n53 = or disjoint i64 %address.local425, %nc.n52, !dbg !8
  %.tr1401 = trunc i64 %nc.n53 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %nc.n54 = insertelement <2 x i64> poison, i64 %nc.n51, i64 0, !dbg !8
  %nc.n55 = bitcast <2 x i64> %nc.n54 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1388 = shufflevector <4 x i32> %nc.n55, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n50), !dbg !8
  %nc.n56 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1388) #10, !dbg !8
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
  %nc.n57 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %nc.n58 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %nc.n59 = fmul <4 x half> %nc.n57, splat (half 0xH3015), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %nc.n60 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %nc.n61 = fmul <4 x half> %nc.n58, splat (half 0xH3015), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %nc.n62 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %nc.n63 = fmul <4 x half> %nc.n60, splat (half 0xH3015), !dbg !10
  %nc.n64 = fmul <4 x half> %nc.n62, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %nc.n65 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !8
  %nc.n66 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %nc.n67 = or disjoint i64 %nc.n30, 64, !dbg !8
  %nc.n68 = or disjoint i64 %address.local413, %nc.n67, !dbg !8
  %.tr1402 = trunc i64 %nc.n68 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %nc.n69 = insertelement <2 x i64> poison, i64 %nc.n66, i64 0, !dbg !8
  %nc.n70 = bitcast <2 x i64> %nc.n69 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1385 = shufflevector <4 x i32> %nc.n70, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1386 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n65), !dbg !8
  %nc.n71 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1386, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1385) #10, !dbg !8
  %nc.n72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !8
  %nc.n73 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %nc.n74 = or disjoint i64 %nc.n30, 64, !dbg !8
  %nc.n75 = or disjoint i64 %address.local425, %nc.n74, !dbg !8
  %.tr1403 = trunc i64 %nc.n75 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %nc.n76 = insertelement <2 x i64> poison, i64 %nc.n73, i64 0, !dbg !8
  %nc.n77 = bitcast <2 x i64> %nc.n76 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1382 = shufflevector <4 x i32> %nc.n77, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1383 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n72), !dbg !8
  %nc.n78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1383, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1382) #10, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %nc.n79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %nc.n80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %nc.n81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %nc.n82 = fmul <4 x half> %nc.n79, splat (half 0xH3015), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %nc.n83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %nc.n84 = fmul <4 x half> %nc.n80, splat (half 0xH3015), !dbg !10
  %nc.n85 = fmul <4 x half> %nc.n81, splat (half 0xH3015), !dbg !10
  %nc.n86 = fmul <4 x half> %nc.n83, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %nc.n87 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !8
  %nc.n88 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %nc.n89 = or disjoint i64 %nc.n30, 96, !dbg !8
  %nc.n90 = or disjoint i64 %address.local413, %nc.n89, !dbg !8
  %.tr1404 = trunc i64 %nc.n90 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %nc.n91 = insertelement <2 x i64> poison, i64 %nc.n88, i64 0, !dbg !8
  %nc.n92 = bitcast <2 x i64> %nc.n91 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1379 = shufflevector <4 x i32> %nc.n92, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1380 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n87), !dbg !8
  %nc.n93 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1380, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1379) #10, !dbg !8
  %nc.n94 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !8
  %nc.n95 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %nc.n96 = or disjoint i64 %nc.n30, 96, !dbg !8
  %nc.n97 = or disjoint i64 %address.local425, %nc.n96, !dbg !8
  %.tr1405 = trunc i64 %nc.n97 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %nc.n98 = insertelement <2 x i64> poison, i64 %nc.n95, i64 0, !dbg !8
  %nc.n99 = bitcast <2 x i64> %nc.n98 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1376 = shufflevector <4 x i32> %nc.n99, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n94), !dbg !8
  %nc.n100 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !8
  %nc.n101 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep1203 = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %nc.n102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1203, align 8, !dbg !10
  %lds.const.gep1204 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %nc.n103 = load <4 x half>, ptr addrspace(3) %lds.const.gep1204, align 8, !dbg !10
  %nc.n104 = fmul <4 x half> %nc.n101, splat (half 0xH3015), !dbg !10
  %lds.const.gep1205 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %nc.n105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1205, align 8, !dbg !10
  %nc.n106 = fmul <4 x half> %nc.n102, splat (half 0xH3015), !dbg !10
  %nc.n107 = fmul <4 x half> %nc.n103, splat (half 0xH3015), !dbg !10
  %nc.n108 = fmul <4 x half> %nc.n105, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1206 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %nc.n109 = load <4 x half>, ptr addrspace(3) %lds.const.gep1206, align 8, !dbg !10
  %lds.const.gep1207 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %nc.n110 = load <4 x half>, ptr addrspace(3) %lds.const.gep1207, align 8, !dbg !10
  %lds.const.gep1208 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %nc.n111 = load <4 x half>, ptr addrspace(3) %lds.const.gep1208, align 8, !dbg !10
  %nc.n112 = fmul <4 x half> %nc.n109, splat (half 0xH3015), !dbg !10
  %lds.const.gep1209 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %nc.n113 = load <4 x half>, ptr addrspace(3) %lds.const.gep1209, align 8, !dbg !10
  %nc.n114 = fmul <4 x half> %nc.n110, splat (half 0xH3015), !dbg !10
  %nc.n115 = fmul <4 x half> %nc.n111, splat (half 0xH3015), !dbg !10
  %nc.n116 = fmul <4 x half> %nc.n113, splat (half 0xH3015), !dbg !10
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %nc.n117 = shl nuw nsw i64 %nc.n32, 2, !dbg !14
  %nc.n118 = shl nuw nsw i64 %nc.n32, 2, !dbg !14
  %nc.n119 = and i64 %nc.n118, 1008, !dbg !14
  %nc.n120 = sub nsw i64 %nc.n117, %nc.n119, !dbg !14
  %nc.n121 = and i32 %nc.n31, 1, !dbg !14
  %index.compact344 = zext nneg i32 %nc.n121 to i64, !dbg !14
  %nc.n122 = or disjoint i64 %nc.n120, %index.compact344, !dbg !14
  %nc.n123 = add nuw nsw i64 %nc.n122, 1, !dbg !14
  %nc.n124 = sdiv i64 %nc.n123, 2, !dbg !14
; Prime K0 after the Q reader handoff.
  %k0.prime.address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %k0.prime.address.local499 = zext nneg i32 %k0.prime.address.thread498 to i64, !dbg !8
  %k0.prime.address.local500 = shl nuw nsw i64 %k0.prime.address.local499, 3, !dbg !8
  %k0.prime.address.local501 = and i64 %k0.prime.address.local500, 1536, !dbg !8
  %k0.prime.address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local501, !dbg !8
  %k0.prime.address.local503 = ptrtoint ptr addrspace(3) %k0.prime.address.local502 to i64, !dbg !8
  %k0.prime.address.local504 = trunc i64 %k0.prime.address.local503 to i32, !dbg !8
  %k0.prime.nc.n139 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local504), !dbg !12
  %k0.prime.dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.nc.n139), !dbg !12
  %k0.prime.nc.n135 = shl i64 %nc.n16, 19, !dbg !12
  %k0.prime.nc.n136 = and i64 %k0.prime.nc.n135, 16252928, !dbg !12
  %k0.prime.nc.n137 = shl nuw nsw i64 0, 14, !dbg !12
  %k0.prime.nc.n138 = or disjoint i64 %k0.prime.nc.n136, %k0.prime.nc.n137, !dbg !12
  %k0.prime.address.local505 = lshr i64 %k0.prime.address.local499, 4, !dbg !12
  %k0.prime.address.local506 = and i64 %k0.prime.address.local505, 8, !dbg !12
  %k0.prime.address.local507 = and i64 %k0.prime.address.local499, 123, !dbg !12
  %k0.prime.address.local508 = lshr i64 %k0.prime.address.local499, 2, !dbg !12
  %k0.prime.address.local509 = and i64 %k0.prime.address.local508, 4, !dbg !12
  %k0.prime.address.local510 = add nuw nsw i64 %k0.prime.address.local509, %k0.prime.address.local499, !dbg !12
  %k0.prime.address.local511 = and i64 %k0.prime.address.local510, 4, !dbg !12
  %k0.prime.address.local512 = or disjoint i64 %k0.prime.address.local507, %k0.prime.address.local511, !dbg !12
  %k0.prime.address.local513 = shl nuw nsw i64 %k0.prime.address.local512, 7, !dbg !12
  %k0.prime.address.local514 = or disjoint i64 %k0.prime.address.local506, %k0.prime.address.local513, !dbg !12
  %k0.prime..idx1406 = or disjoint i64 %k0.prime.nc.n138, %k0.prime.address.local514, !dbg !12
  %k0.prime..idx1406.tr = trunc nuw i64 %k0.prime..idx1406 to i32, !dbg !12
  %k0.prime.dma.buffer.offset1372 = shl nuw nsw i32 %k0.prime..idx1406.tr, 1, !dbg !12
  %k0.prime.nc.n140 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %k0.prime.nc.n141 = insertelement <2 x i64> poison, i64 %k0.prime.nc.n140, i64 0, !dbg !12
  %k0.prime.nc.n142 = bitcast <2 x i64> %k0.prime.nc.n141 to <4 x i32>, !dbg !12
  %k0.prime.dma.buffer.rsrc1373 = shufflevector <4 x i32> %k0.prime.nc.n142, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %k0.prime.nc.n143 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1374, i32 %k0.prime.dma.buffer.offset1372, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #10, !dbg !12
  %k0.prime.address.local516 = or disjoint i64 %k0.prime.address.local499, 256, !dbg !8
  %k0.prime.address.local517 = shl nuw nsw i64 %k0.prime.address.local516, 3, !dbg !8
  %k0.prime.address.local518 = and i64 %k0.prime.address.local517, 3584, !dbg !8
  %k0.prime.address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local518, !dbg !8
  %k0.prime.address.local520 = ptrtoint ptr addrspace(3) %k0.prime.address.local519 to i64, !dbg !8
  %k0.prime.address.local521 = trunc i64 %k0.prime.address.local520 to i32, !dbg !8
  %k0.prime.nc.n144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local521), !dbg !12
  %k0.prime.dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.nc.n144), !dbg !12
  %k0.prime.address.local522 = lshr i64 %k0.prime.address.local516, 4, !dbg !12
  %k0.prime.address.local523 = and i64 %k0.prime.address.local522, 24, !dbg !12
  %k0.prime.address.local524 = and i64 %k0.prime.address.local499, 123, !dbg !12
  %k0.prime.address.local525 = lshr i64 %k0.prime.address.local499, 2, !dbg !12
  %k0.prime.address.local526 = and i64 %k0.prime.address.local525, 4, !dbg !12
  %k0.prime.address.local527 = add nuw nsw i64 %k0.prime.address.local526, %k0.prime.address.local499, !dbg !12
  %k0.prime.address.local528 = and i64 %k0.prime.address.local527, 4, !dbg !12
  %k0.prime.address.local529 = or disjoint i64 %k0.prime.address.local524, %k0.prime.address.local528, !dbg !12
  %k0.prime.address.local530 = shl nuw nsw i64 %k0.prime.address.local529, 7, !dbg !12
  %k0.prime.address.local531 = or disjoint i64 %k0.prime.address.local523, %k0.prime.address.local530, !dbg !12
  %k0.prime..idx1407 = or disjoint i64 %k0.prime.nc.n138, %k0.prime.address.local531, !dbg !12
  %k0.prime..idx1407.tr = trunc nuw i64 %k0.prime..idx1407 to i32, !dbg !12
  %k0.prime.dma.buffer.offset1369 = shl nuw nsw i32 %k0.prime..idx1407.tr, 1, !dbg !12
  %k0.prime.nc.n145 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %k0.prime.nc.n146 = insertelement <2 x i64> poison, i64 %k0.prime.nc.n145, i64 0, !dbg !12
  %k0.prime.nc.n147 = bitcast <2 x i64> %k0.prime.nc.n146 to <4 x i32>, !dbg !12
  %k0.prime.dma.buffer.rsrc1370 = shufflevector <4 x i32> %k0.prime.nc.n147, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %k0.prime.nc.n148 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1371, i32 %k0.prime.dma.buffer.offset1369, <4 x i32> %k0.prime.dma.buffer.rsrc1370) #10, !dbg !12
  br label %nc.n125, !dbg !7

nc.n125:                                              ; preds = %nc.n134, %4
  %nc.n126 = phi i64 [ %nc.n1152, %k0.join ], [ 0, %4 ], !dbg !7
  %nc.n127 = phi [2 x <8 x float>] [ %nc.n1137, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %nc.n128 = phi [2 x <1 x float>] [ %nc.n807, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %nc.n129 = phi [2 x <1 x float>] [ %nc.n1139, %k0.join ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %nc.n130 = phi [2 x <8 x float>] [ %nc.n1143, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %nc.n131 = phi [2 x <8 x float>] [ %nc.n1147, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %nc.n132 = phi [2 x <8 x float>] [ %nc.n1151, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %nc.n133 = icmp samesign ult i64 %nc.n126, 32, !dbg !7
  br i1 %nc.n133, label %nc.n134, label %nc.n1153, !dbg !7

nc.n134:                                              ; preds = %nc.n125
  %nc.n135 = shl i64 %nc.n16, 19, !dbg !12
  %nc.n136 = and i64 %nc.n135, 16252928, !dbg !12
  %nc.n137 = shl nuw nsw i64 %nc.n126, 14, !dbg !12
  %nc.n138 = or disjoint i64 %nc.n136, %nc.n137, !dbg !12
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !8
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !8
  %nc.n139 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
  %address.local505 = lshr i64 %address.local499, 4, !dbg !12
  %address.local506 = and i64 %address.local505, 8, !dbg !12
  %address.local507 = and i64 %address.local499, 123, !dbg !12
  %address.local508 = lshr i64 %address.local499, 2, !dbg !12
  %address.local509 = and i64 %address.local508, 4, !dbg !12
  %address.local510 = add nuw nsw i64 %address.local509, %address.local499, !dbg !12
  %address.local511 = and i64 %address.local510, 4, !dbg !12
  %address.local512 = or disjoint i64 %address.local507, %address.local511, !dbg !12
  %address.local513 = shl nuw nsw i64 %address.local512, 7, !dbg !12
  %address.local514 = or disjoint i64 %address.local506, %address.local513, !dbg !12
  %nc.n140 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1406 = or disjoint i64 %nc.n138, %address.local514, !dbg !12
  %.idx1406.tr = trunc nuw i64 %.idx1406 to i32, !dbg !12
  %dma.buffer.offset1372 = shl nuw nsw i32 %.idx1406.tr, 1, !dbg !12
  %nc.n141 = insertelement <2 x i64> poison, i64 %nc.n140, i64 0, !dbg !12
  %nc.n142 = bitcast <2 x i64> %nc.n141 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %nc.n142, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n139), !dbg !12
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !8
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !8
  %nc.n144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
  %address.local522 = lshr i64 %address.local516, 4, !dbg !12
  %address.local523 = and i64 %address.local522, 24, !dbg !12
  %address.local524 = and i64 %address.local499, 123, !dbg !12
  %address.local525 = lshr i64 %address.local499, 2, !dbg !12
  %address.local526 = and i64 %address.local525, 4, !dbg !12
  %address.local527 = add nuw nsw i64 %address.local526, %address.local499, !dbg !12
  %address.local528 = and i64 %address.local527, 4, !dbg !12
  %address.local529 = or disjoint i64 %address.local524, %address.local528, !dbg !12
  %address.local530 = shl nuw nsw i64 %address.local529, 7, !dbg !12
  %address.local531 = or disjoint i64 %address.local523, %address.local530, !dbg !12
  %nc.n145 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1407 = or disjoint i64 %nc.n138, %address.local531, !dbg !12
  %.idx1407.tr = trunc nuw i64 %.idx1407 to i32, !dbg !12
  %dma.buffer.offset1369 = shl nuw nsw i32 %.idx1407.tr, 1, !dbg !12
  %nc.n146 = insertelement <2 x i64> poison, i64 %nc.n145, i64 0, !dbg !12
  %nc.n147 = bitcast <2 x i64> %nc.n146 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1370 = shufflevector <4 x i32> %nc.n147, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n144), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %nc.n149 = or disjoint i64 %nc.n138, 32, !dbg !12
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !8
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !8
  %nc.n150 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %nc.n151 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1408 = or disjoint i64 %nc.n149, %address.local514, !dbg !12
  %.idx1408.tr = trunc nuw i64 %.idx1408 to i32, !dbg !12
  %dma.buffer.offset1366 = shl nuw nsw i32 %.idx1408.tr, 1, !dbg !12
  %nc.n152 = insertelement <2 x i64> poison, i64 %nc.n151, i64 0, !dbg !12
  %nc.n153 = bitcast <2 x i64> %nc.n152 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1367 = shufflevector <4 x i32> %nc.n153, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n150), !dbg !12
  %nc.n154 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1366, <4 x i32> %dma.buffer.rsrc1367) #10, !dbg !12
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !8
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !8
  %nc.n155 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %nc.n156 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1409 = or disjoint i64 %nc.n149, %address.local531, !dbg !12
  %.idx1409.tr = trunc nuw i64 %.idx1409 to i32, !dbg !12
  %dma.buffer.offset1363 = shl nuw nsw i32 %.idx1409.tr, 1, !dbg !12
  %nc.n157 = insertelement <2 x i64> poison, i64 %nc.n156, i64 0, !dbg !12
  %nc.n158 = bitcast <2 x i64> %nc.n157 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1364 = shufflevector <4 x i32> %nc.n158, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n155), !dbg !12
  %nc.n159 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1363, <4 x i32> %dma.buffer.rsrc1364) #10, !dbg !12
  %address.local542 = lshr i64 %address.local499, 2, !dbg !14
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !14
  %address.local546 = and i64 %address.local545, 24, !dbg !14
  %address.local548 = shl i64 %address.local499, 62, !dbg !14
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !14
  %address.local550 = and i64 %address.local549, -64, !dbg !14
  %address.local552 = lshr i64 %address.local499, 2, !dbg !10
  %address.local553 = and i64 %address.local552, 12, !dbg !10
  %address.local555 = and i64 %address.local499, 63, !dbg !10
  %address.local556 = lshr i64 %address.local555, 5, !dbg !10
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !14
  %nc.n160 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !14
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !14
  %nc.n161 = getelementptr i8, ptr addrspace(3) %nc.n160, i64 %.idx1410, !dbg !14
  %nc.n162 = getelementptr [2 x i8], ptr addrspace(3) %nc.n161, i64 %address.local546, !dbg !14
  %nc.n163 = getelementptr [2 x i8], ptr addrspace(3) %nc.n162, i64 %address.local550, !dbg !14
  %nc.n164 = getelementptr [2 x i8], ptr addrspace(3) %nc.n163, i64 %address.local553, !dbg !14
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %nc.n164, i64 %address.local557, !dbg !14
  %nc.n165 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %nc.n166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n165, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1210 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %nc.n167 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !14
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !14
  %address.local571 = and i32 %address.local570, 24, !dbg !14
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !14
  %address.local573 = and i32 %address.local572, 768, !dbg !14
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !14
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !14
  %address.local576 = and i32 %address.local575, 96, !dbg !14
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !14
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !14
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %nc.n124) #9, !dbg !14, !frisk.register_lifetimes !1
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !14
  %address.local580 = and i32 %address.local579, 1, !dbg !14
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !14
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !14
  %address.local583 = shl i64 %address.local582, 6, !dbg !14
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !14
  %nc.n168 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !14
  %nc.n169 = getelementptr [2 x i8], ptr addrspace(3) %nc.n168, i64 %address.local553, !dbg !14
  %nc.n170 = getelementptr [2 x i8], ptr addrspace(3) %nc.n169, i64 %address.local557, !dbg !14
  %address.local588 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 64, !dbg !14
  %nc.n171 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %nc.n172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n167, <4 x float> %nc.n166) #9, !dbg !14
  %nc.n173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n171, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1211 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 4160, !dbg !14
  %nc.n174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !14
  %lds.const.gep1212 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %nc.n175 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !14
  %nc.n176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n174, <4 x float> %nc.n173) #9, !dbg !14
  %nc.n177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n175, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1213 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %nc.n178 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !14
  %lds.const.gep1214 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 192, !dbg !14
  %nc.n179 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !14
  %nc.n180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n178, <4 x float> %nc.n177) #9, !dbg !14
  %nc.n181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n179, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 4288, !dbg !14
  %nc.n182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !14
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %nc.n183 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !14
  %nc.n184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n182, <4 x float> %nc.n181) #9, !dbg !14
  %nc.n185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n183, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %nc.n186 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !14
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 1088, !dbg !14
  %nc.n187 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !14
  %nc.n188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n186, <4 x float> %nc.n185) #9, !dbg !14
  %nc.n189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n187, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 5184, !dbg !14
  %nc.n190 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !14
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %nc.n191 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !14
  %nc.n192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n190, <4 x float> %nc.n189) #9, !dbg !14
  %nc.n193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n191, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %nc.n194 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !14
  %lds.const.gep1222 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 1216, !dbg !14
  %nc.n195 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !14
  %nc.n196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n194, <4 x float> %nc.n193) #9, !dbg !14
  %nc.n197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n195, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1223 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 5312, !dbg !14
  %nc.n198 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !14
  %nc.n199 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %nc.n200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n198, <4 x float> %nc.n197) #9, !dbg !14
  %nc.n201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n199, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1224 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %nc.n202 = load <4 x half>, ptr addrspace(3) %lds.const.gep1224, align 8, !dbg !14
  %nc.n203 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %nc.n204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n202, <4 x float> %nc.n201) #9, !dbg !14
  %nc.n205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n203, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1225 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 4160, !dbg !14
  %nc.n206 = load <4 x half>, ptr addrspace(3) %lds.const.gep1225, align 8, !dbg !14
  %lds.const.gep1226 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %nc.n207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1226, align 8, !dbg !14
  %nc.n208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n206, <4 x float> %nc.n205) #9, !dbg !14
  %nc.n209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n207, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1227 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %nc.n210 = load <4 x half>, ptr addrspace(3) %lds.const.gep1227, align 8, !dbg !14
  %lds.const.gep1228 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 192, !dbg !14
  %nc.n211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1228, align 8, !dbg !14
  %nc.n212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n210, <4 x float> %nc.n209) #9, !dbg !14
  %nc.n213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n211, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1229 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 4288, !dbg !14
  %nc.n214 = load <4 x half>, ptr addrspace(3) %lds.const.gep1229, align 8, !dbg !14
  %lds.const.gep1230 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %nc.n215 = load <4 x half>, ptr addrspace(3) %lds.const.gep1230, align 8, !dbg !14
  %nc.n216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n214, <4 x float> %nc.n213) #9, !dbg !14
  %nc.n217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n215, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1231 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %nc.n218 = load <4 x half>, ptr addrspace(3) %lds.const.gep1231, align 8, !dbg !14
  %lds.const.gep1232 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 1088, !dbg !14
  %nc.n219 = load <4 x half>, ptr addrspace(3) %lds.const.gep1232, align 8, !dbg !14
  %nc.n220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n218, <4 x float> %nc.n217) #9, !dbg !14
  %nc.n221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n219, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1233 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 5184, !dbg !14
  %nc.n222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1233, align 8, !dbg !14
  %lds.const.gep1234 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %nc.n223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1234, align 8, !dbg !14
  %nc.n224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n222, <4 x float> %nc.n221) #9, !dbg !14
  %nc.n225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n223, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1235 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %nc.n226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1235, align 8, !dbg !14
  %lds.const.gep1236 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 1216, !dbg !14
  %nc.n227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1236, align 8, !dbg !14
  %nc.n228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n226, <4 x float> %nc.n225) #9, !dbg !14
  %nc.n229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n227, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1237 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 5312, !dbg !14
  %nc.n230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1237, align 8, !dbg !14
  %nc.n231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n230, <4 x float> %nc.n229) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %nc.n232 = or disjoint i64 %nc.n138, 64, !dbg !12
  %nc.n233 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
  %nc.n234 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14111412 = or disjoint i64 %nc.n232, %address.local514, !dbg !12
  %.idx14111412.tr = trunc nuw i64 %.idx14111412 to i32, !dbg !12
  %dma.buffer.offset1360 = shl nuw nsw i32 %.idx14111412.tr, 1, !dbg !12
  %nc.n235 = insertelement <2 x i64> poison, i64 %nc.n234, i64 0, !dbg !12
  %nc.n236 = bitcast <2 x i64> %nc.n235 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1361 = shufflevector <4 x i32> %nc.n236, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1362 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n233), !dbg !12
  %nc.n237 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1362, i32 %dma.buffer.offset1360, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !12
  %nc.n238 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
  %nc.n239 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14131414 = or disjoint i64 %nc.n232, %address.local531, !dbg !12
  %.idx14131414.tr = trunc nuw i64 %.idx14131414 to i32, !dbg !12
  %dma.buffer.offset1357 = shl nuw nsw i32 %.idx14131414.tr, 1, !dbg !12
  %nc.n240 = insertelement <2 x i64> poison, i64 %nc.n239, i64 0, !dbg !12
  %nc.n241 = bitcast <2 x i64> %nc.n240 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1358 = shufflevector <4 x i32> %nc.n241, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1359 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n238), !dbg !12
  %nc.n242 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1359, i32 %dma.buffer.offset1357, <4 x i32> %dma.buffer.rsrc1358) #10, !dbg !12
  %lds.const.gep1238 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %nc.n243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !14
  %nc.n244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n243, <4 x float> %nc.n172) #9, !dbg !14
  %lds.const.gep1239 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %nc.n245 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !14
  %lds.const.gep1240 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 8256, !dbg !14
  %nc.n246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !14
  %nc.n247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n245, <4 x float> %nc.n244) #9, !dbg !14
  %nc.n248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n246, <4 x float> %nc.n176) #9, !dbg !14
  %lds.const.gep1241 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 12352, !dbg !14
  %nc.n249 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !14
  %lds.const.gep1242 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %nc.n250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !14
  %nc.n251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n249, <4 x float> %nc.n248) #9, !dbg !14
  %nc.n252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n250, <4 x float> %nc.n180) #9, !dbg !14
  %lds.const.gep1243 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %nc.n253 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !14
  %lds.const.gep1244 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 8384, !dbg !14
  %nc.n254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !14
  %nc.n255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n253, <4 x float> %nc.n252) #9, !dbg !14
  %nc.n256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n254, <4 x float> %nc.n184) #9, !dbg !14
  %lds.const.gep1245 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 12480, !dbg !14
  %nc.n257 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !14
  %lds.const.gep1246 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %nc.n258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !14
  %nc.n259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n257, <4 x float> %nc.n256) #9, !dbg !14
  %nc.n260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n258, <4 x float> %nc.n188) #9, !dbg !14
  %lds.const.gep1247 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %nc.n261 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !14
  %lds.const.gep1248 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 9280, !dbg !14
  %nc.n262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !14
  %nc.n263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n261, <4 x float> %nc.n260) #9, !dbg !14
  %nc.n264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n262, <4 x float> %nc.n192) #9, !dbg !14
  %lds.const.gep1249 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 13376, !dbg !14
  %nc.n265 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !14
  %lds.const.gep1250 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %nc.n266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !14
  %nc.n267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n265, <4 x float> %nc.n264) #9, !dbg !14
  %nc.n268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n266, <4 x float> %nc.n196) #9, !dbg !14
  %lds.const.gep1251 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %nc.n269 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !14
  %lds.const.gep1252 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 9408, !dbg !14
  %nc.n270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !14
  %nc.n271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n269, <4 x float> %nc.n268) #9, !dbg !14
  %nc.n272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n270, <4 x float> %nc.n200) #9, !dbg !14
  %lds.const.gep1253 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 13504, !dbg !14
  %nc.n273 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !14
  %lds.const.gep1254 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %nc.n274 = load <4 x half>, ptr addrspace(3) %lds.const.gep1254, align 8, !dbg !14
  %nc.n275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n273, <4 x float> %nc.n272) #9, !dbg !14
  %nc.n276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n274, <4 x float> %nc.n204) #9, !dbg !14
  %lds.const.gep1255 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %nc.n277 = load <4 x half>, ptr addrspace(3) %lds.const.gep1255, align 8, !dbg !14
  %lds.const.gep1256 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 8256, !dbg !14
  %nc.n278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1256, align 8, !dbg !14
  %nc.n279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n277, <4 x float> %nc.n276) #9, !dbg !14
  %nc.n280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n278, <4 x float> %nc.n208) #9, !dbg !14
  %lds.const.gep1257 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 12352, !dbg !14
  %nc.n281 = load <4 x half>, ptr addrspace(3) %lds.const.gep1257, align 8, !dbg !14
  %lds.const.gep1258 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %nc.n282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1258, align 8, !dbg !14
  %nc.n283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n281, <4 x float> %nc.n280) #9, !dbg !14
  %nc.n284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n282, <4 x float> %nc.n212) #9, !dbg !14
  %lds.const.gep1259 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %nc.n285 = load <4 x half>, ptr addrspace(3) %lds.const.gep1259, align 8, !dbg !14
  %lds.const.gep1260 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 8384, !dbg !14
  %nc.n286 = load <4 x half>, ptr addrspace(3) %lds.const.gep1260, align 8, !dbg !14
  %nc.n287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n285, <4 x float> %nc.n284) #9, !dbg !14
  %nc.n288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n286, <4 x float> %nc.n216) #9, !dbg !14
  %lds.const.gep1261 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 12480, !dbg !14
  %nc.n289 = load <4 x half>, ptr addrspace(3) %lds.const.gep1261, align 8, !dbg !14
  %lds.const.gep1262 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %nc.n290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1262, align 8, !dbg !14
  %nc.n291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n289, <4 x float> %nc.n288) #9, !dbg !14
  %nc.n292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n290, <4 x float> %nc.n220) #9, !dbg !14
  %lds.const.gep1263 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %nc.n293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1263, align 8, !dbg !14
  %lds.const.gep1264 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 9280, !dbg !14
  %nc.n294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1264, align 8, !dbg !14
  %nc.n295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n293, <4 x float> %nc.n292) #9, !dbg !14
  %nc.n296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n294, <4 x float> %nc.n224) #9, !dbg !14
  %lds.const.gep1265 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 13376, !dbg !14
  %nc.n297 = load <4 x half>, ptr addrspace(3) %lds.const.gep1265, align 8, !dbg !14
  %lds.const.gep1266 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %nc.n298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1266, align 8, !dbg !14
  %nc.n299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n297, <4 x float> %nc.n296) #9, !dbg !14
  %nc.n300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n298, <4 x float> %nc.n228) #9, !dbg !14
  %lds.const.gep1267 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %nc.n301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1267, align 8, !dbg !14
  %lds.const.gep1268 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 9408, !dbg !14
  %nc.n302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !14
  %nc.n303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n301, <4 x float> %nc.n300) #9, !dbg !14
  %nc.n304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n302, <4 x float> %nc.n231) #9, !dbg !14
  %lds.const.gep1269 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 13504, !dbg !14
  %nc.n305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !14
  %nc.n306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n305, <4 x float> %nc.n304) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %nc.n307 = or disjoint i64 %nc.n138, 96, !dbg !12
  %nc.n308 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %nc.n309 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14151416 = or disjoint i64 %nc.n307, %address.local514, !dbg !12
  %.idx14151416.tr = trunc nuw i64 %.idx14151416 to i32, !dbg !12
  %dma.buffer.offset1354 = shl nuw nsw i32 %.idx14151416.tr, 1, !dbg !12
  %nc.n310 = insertelement <2 x i64> poison, i64 %nc.n309, i64 0, !dbg !12
  %nc.n311 = bitcast <2 x i64> %nc.n310 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1355 = shufflevector <4 x i32> %nc.n311, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1356 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n308), !dbg !12
  %nc.n312 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1356, i32 %dma.buffer.offset1354, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !12
  %nc.n313 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %nc.n314 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14171418 = or disjoint i64 %nc.n307, %address.local531, !dbg !12
  %.idx14171418.tr = trunc nuw i64 %.idx14171418 to i32, !dbg !12
  %dma.buffer.offset1351 = shl nuw nsw i32 %.idx14171418.tr, 1, !dbg !12
  %nc.n315 = insertelement <2 x i64> poison, i64 %nc.n314, i64 0, !dbg !12
  %nc.n316 = bitcast <2 x i64> %nc.n315 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1352 = shufflevector <4 x i32> %nc.n316, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1353 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n313), !dbg !12
  %nc.n317 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1353, i32 %dma.buffer.offset1351, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !12
  %nc.n318 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %nc.n319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n318, <4 x float> %nc.n247) #9, !dbg !14
  %lds.const.gep1270 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %nc.n320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !14
  %nc.n321 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %nc.n322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n320, <4 x float> %nc.n319) #9, !dbg !14
  %nc.n323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n321, <4 x float> %nc.n251) #9, !dbg !14
  %lds.const.gep1271 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 4160, !dbg !14
  %nc.n324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1271, align 8, !dbg !14
  %lds.const.gep1272 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %nc.n325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1272, align 8, !dbg !14
  %nc.n326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n324, <4 x float> %nc.n323) #9, !dbg !14
  %nc.n327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n325, <4 x float> %nc.n255) #9, !dbg !14
  %lds.const.gep1273 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %nc.n328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1273, align 8, !dbg !14
  %lds.const.gep1274 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 192, !dbg !14
  %nc.n329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1274, align 8, !dbg !14
  %nc.n330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n328, <4 x float> %nc.n327) #9, !dbg !14
  %nc.n331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n329, <4 x float> %nc.n259) #9, !dbg !14
  %lds.const.gep1275 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 4288, !dbg !14
  %nc.n332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1275, align 8, !dbg !14
  %lds.const.gep1276 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %nc.n333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1276, align 8, !dbg !14
  %nc.n334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n332, <4 x float> %nc.n331) #9, !dbg !14
  %nc.n335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n333, <4 x float> %nc.n263) #9, !dbg !14
  %lds.const.gep1277 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %nc.n336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1277, align 8, !dbg !14
  %lds.const.gep1278 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 1088, !dbg !14
  %nc.n337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1278, align 8, !dbg !14
  %nc.n338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n336, <4 x float> %nc.n335) #9, !dbg !14
  %nc.n339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n337, <4 x float> %nc.n267) #9, !dbg !14
  %lds.const.gep1279 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 5184, !dbg !14
  %nc.n340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1279, align 8, !dbg !14
  %lds.const.gep1280 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %nc.n341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1280, align 8, !dbg !14
  %nc.n342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n340, <4 x float> %nc.n339) #9, !dbg !14
  %nc.n343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n341, <4 x float> %nc.n271) #9, !dbg !14
  %lds.const.gep1281 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %nc.n344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1281, align 8, !dbg !14
  %lds.const.gep1282 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 1216, !dbg !14
  %nc.n345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1282, align 8, !dbg !14
  %nc.n346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n344, <4 x float> %nc.n343) #9, !dbg !14
  %nc.n347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n345, <4 x float> %nc.n275) #9, !dbg !14
  %lds.const.gep1283 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 5312, !dbg !14
  %nc.n348 = load <4 x half>, ptr addrspace(3) %lds.const.gep1283, align 8, !dbg !14
  %nc.n349 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %nc.n350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n348, <4 x float> %nc.n347) #9, !dbg !14
  %nc.n351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n349, <4 x float> %nc.n279) #9, !dbg !14
  %lds.const.gep1284 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %nc.n352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1284, align 8, !dbg !14
  %nc.n353 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %nc.n354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n352, <4 x float> %nc.n351) #9, !dbg !14
  %nc.n355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n353, <4 x float> %nc.n283) #9, !dbg !14
  %lds.const.gep1285 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 4160, !dbg !14
  %nc.n356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1285, align 8, !dbg !14
  %lds.const.gep1286 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %nc.n357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1286, align 8, !dbg !14
  %nc.n358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n356, <4 x float> %nc.n355) #9, !dbg !14
  %nc.n359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n357, <4 x float> %nc.n287) #9, !dbg !14
  %lds.const.gep1287 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %nc.n360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1287, align 8, !dbg !14
  %lds.const.gep1288 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 192, !dbg !14
  %nc.n361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1288, align 8, !dbg !14
  %nc.n362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n360, <4 x float> %nc.n359) #9, !dbg !14
  %nc.n363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n361, <4 x float> %nc.n291) #9, !dbg !14
  %lds.const.gep1289 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 4288, !dbg !14
  %nc.n364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1289, align 8, !dbg !14
  %lds.const.gep1290 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %nc.n365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1290, align 8, !dbg !14
  %nc.n366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n364, <4 x float> %nc.n363) #9, !dbg !14
  %nc.n367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n365, <4 x float> %nc.n295) #9, !dbg !14
  %lds.const.gep1291 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %nc.n368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1291, align 8, !dbg !14
  %lds.const.gep1292 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 1088, !dbg !14
  %nc.n369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1292, align 8, !dbg !14
  %nc.n370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n368, <4 x float> %nc.n367) #9, !dbg !14
  %nc.n371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n369, <4 x float> %nc.n299) #9, !dbg !14
  %lds.const.gep1293 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 5184, !dbg !14
  %nc.n372 = load <4 x half>, ptr addrspace(3) %lds.const.gep1293, align 8, !dbg !14
  %lds.const.gep1294 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %nc.n373 = load <4 x half>, ptr addrspace(3) %lds.const.gep1294, align 8, !dbg !14
  %nc.n374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n372, <4 x float> %nc.n371) #9, !dbg !14
  %nc.n375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n373, <4 x float> %nc.n303) #9, !dbg !14
  %lds.const.gep1295 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %nc.n376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1295, align 8, !dbg !14
  %lds.const.gep1296 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 1216, !dbg !14
  %nc.n377 = load <4 x half>, ptr addrspace(3) %lds.const.gep1296, align 8, !dbg !14
  %nc.n378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n376, <4 x float> %nc.n375) #9, !dbg !14
  %nc.n379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n377, <4 x float> %nc.n306) #9, !dbg !14
  %lds.const.gep1297 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 5312, !dbg !14
  %nc.n380 = load <4 x half>, ptr addrspace(3) %lds.const.gep1297, align 8, !dbg !14
  %nc.n381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n380, <4 x float> %nc.n379) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %nc.n382 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !15
  %address.local918 = and i64 %address.local917, 8064, !dbg !15
  %address.local919 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local920 = and i64 %address.local919, 24, !dbg !15
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !15
  %nc.n383 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14191420 = or disjoint i64 %nc.n138, %address.local921, !dbg !15
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !15
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !15
  %nc.n384 = insertelement <2 x i64> poison, i64 %nc.n383, i64 0, !dbg !15
  %nc.n385 = bitcast <2 x i64> %nc.n384 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %nc.n385, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1350 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n382), !dbg !15
  %nc.n386 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1350, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %nc.n387 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !15
  %address.local924 = and i64 %address.local923, 16256, !dbg !15
  %address.local925 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local926 = and i64 %address.local925, 24, !dbg !15
  %address.local927 = or disjoint i64 %address.local924, %address.local926, !dbg !15
  %nc.n388 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14211422 = or disjoint i64 %nc.n138, %address.local927, !dbg !15
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !15
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !15
  %nc.n389 = insertelement <2 x i64> poison, i64 %nc.n388, i64 0, !dbg !15
  %nc.n390 = bitcast <2 x i64> %nc.n389 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1346 = shufflevector <4 x i32> %nc.n390, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1347 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n387), !dbg !15
  %nc.n391 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1347, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1346) #10, !dbg !15
  %lds.const.gep1298 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %nc.n392 = load <4 x half>, ptr addrspace(3) %lds.const.gep1298, align 8, !dbg !14
  %nc.n393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n392, <4 x float> %nc.n322) #9, !dbg !14
  %lds.const.gep1299 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %nc.n394 = load <4 x half>, ptr addrspace(3) %lds.const.gep1299, align 8, !dbg !14
  %lds.const.gep1300 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 8256, !dbg !14
  %nc.n395 = load <4 x half>, ptr addrspace(3) %lds.const.gep1300, align 8, !dbg !14
  %nc.n396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n394, <4 x float> %nc.n393) #9, !dbg !14
  %nc.n397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n395, <4 x float> %nc.n326) #9, !dbg !14
  %nc.n398 = extractelement <4 x float> %nc.n396, i64 0, !dbg !19
  %nc.n399 = call float @llvm.maxnum.f32(float %nc.n398, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %nc.n400 = extractelement <4 x float> %nc.n396, i64 1, !dbg !19
  %nc.n401 = extractelement <4 x float> %nc.n396, i64 2, !dbg !19
  %nc.n402 = extractelement <4 x float> %nc.n396, i64 3, !dbg !19
  %lds.const.gep1301 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 12352, !dbg !14
  %nc.n403 = load <4 x half>, ptr addrspace(3) %lds.const.gep1301, align 8, !dbg !14
  %lds.const.gep1302 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %nc.n404 = load <4 x half>, ptr addrspace(3) %lds.const.gep1302, align 8, !dbg !14
  %nc.n405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n403, <4 x float> %nc.n397) #9, !dbg !14
  %nc.n406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n404, <4 x float> %nc.n330) #9, !dbg !14
  %nc.n407 = extractelement <4 x float> %nc.n405, i64 0, !dbg !19
  %local.reduce151 = call float @llvm.maxnum.f32(float %nc.n399, float %nc.n407), !dbg !20, !frisk.local_reduction !1
  %nc.n408 = extractelement <4 x float> %nc.n405, i64 1, !dbg !19
  %local.reduce152 = call float @llvm.maxnum.f32(float %nc.n400, float %nc.n408), !dbg !20, !frisk.local_reduction !1
  %nc.n409 = extractelement <4 x float> %nc.n405, i64 2, !dbg !19
  %local.reduce153 = call float @llvm.maxnum.f32(float %nc.n401, float %nc.n409), !dbg !20, !frisk.local_reduction !1
  %nc.n410 = extractelement <4 x float> %nc.n405, i64 3, !dbg !19
  %local.reduce154 = call float @llvm.maxnum.f32(float %nc.n402, float %nc.n410), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1303 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %nc.n411 = load <4 x half>, ptr addrspace(3) %lds.const.gep1303, align 8, !dbg !14
  %lds.const.gep1304 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 8384, !dbg !14
  %nc.n412 = load <4 x half>, ptr addrspace(3) %lds.const.gep1304, align 8, !dbg !14
  %nc.n413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n411, <4 x float> %nc.n406) #9, !dbg !14
  %nc.n414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n412, <4 x float> %nc.n334) #9, !dbg !14
  %nc.n415 = extractelement <4 x float> %nc.n413, i64 0, !dbg !19
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %nc.n415), !dbg !20, !frisk.local_reduction !1
  %nc.n416 = extractelement <4 x float> %nc.n413, i64 1, !dbg !19
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %nc.n416), !dbg !20, !frisk.local_reduction !1
  %nc.n417 = extractelement <4 x float> %nc.n413, i64 2, !dbg !19
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %nc.n417), !dbg !20, !frisk.local_reduction !1
  %nc.n418 = extractelement <4 x float> %nc.n413, i64 3, !dbg !19
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %nc.n418), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1305 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 12480, !dbg !14
  %nc.n419 = load <4 x half>, ptr addrspace(3) %lds.const.gep1305, align 8, !dbg !14
  %lds.const.gep1306 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %nc.n420 = load <4 x half>, ptr addrspace(3) %lds.const.gep1306, align 8, !dbg !14
  %nc.n421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n419, <4 x float> %nc.n414) #9, !dbg !14
  %nc.n422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n420, <4 x float> %nc.n338) #9, !dbg !14
  %nc.n423 = extractelement <4 x float> %nc.n421, i64 0, !dbg !19
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %nc.n423), !dbg !20, !frisk.local_reduction !1
  %nc.n424 = extractelement <4 x float> %nc.n421, i64 1, !dbg !19
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %nc.n424), !dbg !20, !frisk.local_reduction !1
  %nc.n425 = extractelement <4 x float> %nc.n421, i64 2, !dbg !19
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %nc.n425), !dbg !20, !frisk.local_reduction !1
  %nc.n426 = extractelement <4 x float> %nc.n421, i64 3, !dbg !19
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %nc.n426), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1307 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %nc.n427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1307, align 8, !dbg !14
  %lds.const.gep1308 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 9280, !dbg !14
  %nc.n428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1308, align 8, !dbg !14
  %nc.n429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n427, <4 x float> %nc.n422) #9, !dbg !14
  %nc.n430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n428, <4 x float> %nc.n342) #9, !dbg !14
  %nc.n431 = extractelement <4 x float> %nc.n429, i64 0, !dbg !19
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %nc.n431), !dbg !20, !frisk.local_reduction !1
  %nc.n432 = extractelement <4 x float> %nc.n429, i64 1, !dbg !19
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %nc.n432), !dbg !20, !frisk.local_reduction !1
  %nc.n433 = extractelement <4 x float> %nc.n429, i64 2, !dbg !19
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %nc.n433), !dbg !20, !frisk.local_reduction !1
  %nc.n434 = extractelement <4 x float> %nc.n429, i64 3, !dbg !19
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %nc.n434), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1309 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 13376, !dbg !14
  %nc.n435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1309, align 8, !dbg !14
  %lds.const.gep1310 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %nc.n436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1310, align 8, !dbg !14
  %nc.n437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n435, <4 x float> %nc.n430) #9, !dbg !14
  %nc.n438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n436, <4 x float> %nc.n346) #9, !dbg !14
  %nc.n439 = extractelement <4 x float> %nc.n437, i64 0, !dbg !19
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %nc.n439), !dbg !20, !frisk.local_reduction !1
  %nc.n440 = extractelement <4 x float> %nc.n437, i64 1, !dbg !19
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %nc.n440), !dbg !20, !frisk.local_reduction !1
  %nc.n441 = extractelement <4 x float> %nc.n437, i64 2, !dbg !19
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %nc.n441), !dbg !20, !frisk.local_reduction !1
  %nc.n442 = extractelement <4 x float> %nc.n437, i64 3, !dbg !19
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %nc.n442), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1311 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %nc.n443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1311, align 8, !dbg !14
  %lds.const.gep1312 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 9408, !dbg !14
  %nc.n444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1312, align 8, !dbg !14
  %nc.n445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n443, <4 x float> %nc.n438) #9, !dbg !14
  %nc.n446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n444, <4 x float> %nc.n350) #9, !dbg !14
  %nc.n447 = extractelement <4 x float> %nc.n445, i64 0, !dbg !19
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %nc.n447), !dbg !20, !frisk.local_reduction !1
  %nc.n448 = extractelement <4 x float> %nc.n445, i64 1, !dbg !19
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %nc.n448), !dbg !20, !frisk.local_reduction !1
  %nc.n449 = extractelement <4 x float> %nc.n445, i64 2, !dbg !19
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %nc.n449), !dbg !20, !frisk.local_reduction !1
  %nc.n450 = extractelement <4 x float> %nc.n445, i64 3, !dbg !19
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %nc.n450), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1313 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 13504, !dbg !14
  %nc.n451 = load <4 x half>, ptr addrspace(3) %lds.const.gep1313, align 8, !dbg !14
  %lds.const.gep1314 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %nc.n452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1314, align 8, !dbg !14
  %nc.n453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n451, <4 x float> %nc.n446) #9, !dbg !14
  %nc.n454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n452, <4 x float> %nc.n354) #9, !dbg !14
  %nc.n455 = extractelement <4 x float> %nc.n453, i64 0, !dbg !19
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %nc.n455), !dbg !20, !frisk.local_reduction !1
  %nc.n456 = extractelement <4 x float> %nc.n453, i64 1, !dbg !19
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %nc.n456), !dbg !20, !frisk.local_reduction !1
  %nc.n457 = extractelement <4 x float> %nc.n453, i64 2, !dbg !19
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %nc.n457), !dbg !20, !frisk.local_reduction !1
  %nc.n458 = extractelement <4 x float> %nc.n453, i64 3, !dbg !19
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %nc.n458), !dbg !20, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !20, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !20, !frisk.local_reduction !1
  %nc.n459 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1315 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %nc.n460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1315, align 8, !dbg !14
  %lds.const.gep1316 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 8256, !dbg !14
  %nc.n461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1316, align 8, !dbg !14
  %nc.n462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n460, <4 x float> %nc.n454) #9, !dbg !14
  %nc.n463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n461, <4 x float> %nc.n358) #9, !dbg !14
  %nc.n464 = extractelement <4 x float> %nc.n462, i64 0, !dbg !19
  %nc.n465 = extractelement <4 x float> %nc.n462, i64 1, !dbg !19
  %nc.n466 = extractelement <4 x float> %nc.n462, i64 2, !dbg !19
  %nc.n467 = extractelement <4 x float> %nc.n462, i64 3, !dbg !19
  %lds.const.gep1317 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 12352, !dbg !14
  %nc.n468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1317, align 8, !dbg !14
  %lds.const.gep1318 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %nc.n469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1318, align 8, !dbg !14
  %nc.n470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n468, <4 x float> %nc.n463) #9, !dbg !14
  %nc.n471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n469, <4 x float> %nc.n362) #9, !dbg !14
  %nc.n472 = extractelement <4 x float> %nc.n470, i64 0, !dbg !19
  %nc.n473 = extractelement <4 x float> %nc.n470, i64 1, !dbg !19
  %nc.n474 = extractelement <4 x float> %nc.n470, i64 2, !dbg !19
  %nc.n475 = extractelement <4 x float> %nc.n470, i64 3, !dbg !19
  %lds.const.gep1319 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %nc.n476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1319, align 8, !dbg !14
  %lds.const.gep1320 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 8384, !dbg !14
  %nc.n477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1320, align 8, !dbg !14
  %nc.n478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n476, <4 x float> %nc.n471) #9, !dbg !14
  %nc.n479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n477, <4 x float> %nc.n366) #9, !dbg !14
  %nc.n480 = extractelement <4 x float> %nc.n478, i64 0, !dbg !19
  %nc.n481 = extractelement <4 x float> %nc.n478, i64 1, !dbg !19
  %nc.n482 = extractelement <4 x float> %nc.n478, i64 2, !dbg !19
  %nc.n483 = extractelement <4 x float> %nc.n478, i64 3, !dbg !19
  %lds.const.gep1321 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 12480, !dbg !14
  %nc.n484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1321, align 8, !dbg !14
  %lds.const.gep1322 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %nc.n485 = load <4 x half>, ptr addrspace(3) %lds.const.gep1322, align 8, !dbg !14
  %nc.n486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n484, <4 x float> %nc.n479) #9, !dbg !14
  %nc.n487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n485, <4 x float> %nc.n370) #9, !dbg !14
  %nc.n488 = extractelement <4 x float> %nc.n486, i64 0, !dbg !19
  %nc.n489 = extractelement <4 x float> %nc.n486, i64 1, !dbg !19
  %nc.n490 = extractelement <4 x float> %nc.n486, i64 2, !dbg !19
  %nc.n491 = extractelement <4 x float> %nc.n486, i64 3, !dbg !19
  %lds.const.gep1323 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %nc.n492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1323, align 8, !dbg !14
  %lds.const.gep1324 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 9280, !dbg !14
  %nc.n493 = load <4 x half>, ptr addrspace(3) %lds.const.gep1324, align 8, !dbg !14
  %nc.n494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n492, <4 x float> %nc.n487) #9, !dbg !14
  %nc.n495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n493, <4 x float> %nc.n374) #9, !dbg !14
  %nc.n496 = extractelement <4 x float> %nc.n494, i64 0, !dbg !19
  %nc.n497 = extractelement <4 x float> %nc.n494, i64 1, !dbg !19
  %nc.n498 = extractelement <4 x float> %nc.n494, i64 2, !dbg !19
  %nc.n499 = extractelement <4 x float> %nc.n494, i64 3, !dbg !19
  %lds.const.gep1325 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 13376, !dbg !14
  %nc.n500 = load <4 x half>, ptr addrspace(3) %lds.const.gep1325, align 8, !dbg !14
  %lds.const.gep1326 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %nc.n501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1326, align 8, !dbg !14
  %nc.n502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n500, <4 x float> %nc.n495) #9, !dbg !14
  %nc.n503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n501, <4 x float> %nc.n378) #9, !dbg !14
  %nc.n504 = extractelement <4 x float> %nc.n502, i64 0, !dbg !19
  %nc.n505 = extractelement <4 x float> %nc.n502, i64 1, !dbg !19
  %nc.n506 = extractelement <4 x float> %nc.n502, i64 2, !dbg !19
  %nc.n507 = extractelement <4 x float> %nc.n502, i64 3, !dbg !19
  %lds.const.gep1327 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %nc.n508 = load <4 x half>, ptr addrspace(3) %lds.const.gep1327, align 8, !dbg !14
  %lds.const.gep1328 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 9408, !dbg !14
  %nc.n509 = load <4 x half>, ptr addrspace(3) %lds.const.gep1328, align 8, !dbg !14
  %nc.n510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n508, <4 x float> %nc.n503) #9, !dbg !14
  %nc.n511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n509, <4 x float> %nc.n381) #9, !dbg !14
  %nc.n512 = extractelement <4 x float> %nc.n510, i64 0, !dbg !19
  %nc.n513 = extractelement <4 x float> %nc.n510, i64 1, !dbg !19
  %nc.n514 = extractelement <4 x float> %nc.n510, i64 2, !dbg !19
  %nc.n515 = extractelement <4 x float> %nc.n510, i64 3, !dbg !19
  %lds.const.gep1329 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 13504, !dbg !14
  %nc.n516 = load <4 x half>, ptr addrspace(3) %lds.const.gep1329, align 8, !dbg !14
  %nc.n517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n516, <4 x float> %nc.n511) #9, !dbg !14
  %nc.n518 = extractelement <4 x float> %nc.n517, i64 0, !dbg !19
  %nc.n519 = extractelement <4 x float> %nc.n517, i64 1, !dbg !19
  %nc.n520 = extractelement <4 x float> %nc.n517, i64 2, !dbg !19
  %nc.n521 = extractelement <4 x float> %nc.n517, i64 3, !dbg !19
  %nc.n522 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %nc.n523 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %nc.n522), !dbg !20
  %nc.n524 = add nuw nsw i32 %nc.n523, 64, !dbg !20
  %nc.n525 = xor i32 %nc.n523, 16, !dbg !20
  %nc.n526 = and i32 %nc.n524, 192, !dbg !20
  %nc.n527 = icmp samesign ult i32 %nc.n525, %nc.n526, !dbg !20
  %nc.n528 = select i1 %nc.n527, i32 %nc.n525, i32 %nc.n523, !dbg !20
  %nc.n529 = shl nuw nsw i32 %nc.n528, 2, !dbg !20
  %nc.n530 = bitcast float %nc.n459 to i32, !dbg !20
  %nc.n531 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n530), !dbg !20
  %nc.n532 = bitcast i32 %nc.n531 to float, !dbg !20
  %nc.n533 = call float @llvm.maxnum.f32(float %nc.n459, float %nc.n532), !dbg !20
  %nc.n534 = xor i32 %nc.n523, 32, !dbg !20
  %nc.n535 = icmp samesign ult i32 %nc.n534, %nc.n526, !dbg !20
  %nc.n536 = select i1 %nc.n535, i32 %nc.n534, i32 %nc.n523, !dbg !20
  %nc.n537 = shl nuw nsw i32 %nc.n536, 2, !dbg !20
  %nc.n538 = bitcast float %nc.n533 to i32, !dbg !20
  %nc.n539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n538), !dbg !20
  %nc.n540 = bitcast i32 %nc.n539 to float, !dbg !20
  %nc.n541 = call float @llvm.maxnum.f32(float %nc.n533, float %nc.n540), !dbg !20
  %nc.n542 = extractvalue [2 x <1 x float>] %nc.n129, 0, !dbg !20
  %nc.n543 = insertelement <1 x float> poison, float %nc.n541, i64 0, !dbg !20
  %nc.n544 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %nc.n543, <1 x float> %nc.n542), !dbg !20
  %nc.n545 = call float @llvm.maxnum.f32(float %nc.n464, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %nc.n545, float %nc.n472), !dbg !20, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %nc.n465, float %nc.n473), !dbg !20, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %nc.n466, float %nc.n474), !dbg !20, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %nc.n467, float %nc.n475), !dbg !20, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %nc.n480), !dbg !20, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %nc.n481), !dbg !20, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %nc.n482), !dbg !20, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %nc.n483), !dbg !20, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %nc.n488), !dbg !20, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %nc.n489), !dbg !20, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %nc.n490), !dbg !20, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %nc.n491), !dbg !20, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %nc.n496), !dbg !20, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %nc.n497), !dbg !20, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %nc.n498), !dbg !20, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %nc.n499), !dbg !20, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %nc.n504), !dbg !20, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %nc.n505), !dbg !20, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %nc.n506), !dbg !20, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %nc.n507), !dbg !20, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %nc.n512), !dbg !20, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %nc.n513), !dbg !20, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %nc.n514), !dbg !20, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %nc.n515), !dbg !20, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %nc.n518), !dbg !20, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %nc.n519), !dbg !20, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %nc.n520), !dbg !20, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %nc.n521), !dbg !20, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !20, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !20, !frisk.local_reduction !1
  %nc.n546 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !20, !frisk.local_reduction !1
  %nc.n547 = bitcast float %nc.n546 to i32, !dbg !20
  %nc.n548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n547), !dbg !20
  %nc.n549 = bitcast i32 %nc.n548 to float, !dbg !20
  %nc.n550 = call float @llvm.maxnum.f32(float %nc.n546, float %nc.n549), !dbg !20
  %nc.n551 = bitcast float %nc.n550 to i32, !dbg !20
  %nc.n552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n551), !dbg !20
  %nc.n553 = bitcast i32 %nc.n552 to float, !dbg !20
  %nc.n554 = call float @llvm.maxnum.f32(float %nc.n550, float %nc.n553), !dbg !20
  %nc.n555 = extractvalue [2 x <1 x float>] %nc.n129, 1, !dbg !20
  %nc.n556 = insertelement <1 x float> poison, float %nc.n554, i64 0, !dbg !20
  %nc.n557 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %nc.n556, <1 x float> %nc.n555), !dbg !20
  %nc.n558 = shufflevector <1 x float> %nc.n544, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %nc.n559 = fsub <4 x float> %nc.n396, %nc.n558, !dbg !21
  %nc.n560 = fsub <4 x float> %nc.n405, %nc.n558, !dbg !21
  %nc.n561 = fsub <4 x float> %nc.n413, %nc.n558, !dbg !21
  %nc.n562 = fsub <4 x float> %nc.n421, %nc.n558, !dbg !21
  %nc.n563 = fsub <4 x float> %nc.n429, %nc.n558, !dbg !21
  %nc.n564 = fsub <4 x float> %nc.n437, %nc.n558, !dbg !21
  %nc.n565 = fsub <4 x float> %nc.n445, %nc.n558, !dbg !21
  %nc.n566 = fsub <4 x float> %nc.n453, %nc.n558, !dbg !21
  %nc.n567 = shufflevector <1 x float> %nc.n557, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %nc.n568 = fsub <4 x float> %nc.n462, %nc.n567, !dbg !21
  %nc.n569 = fsub <4 x float> %nc.n470, %nc.n567, !dbg !21
  %nc.n570 = fsub <4 x float> %nc.n478, %nc.n567, !dbg !21
  %nc.n571 = fsub <4 x float> %nc.n486, %nc.n567, !dbg !21
  %nc.n572 = fsub <4 x float> %nc.n494, %nc.n567, !dbg !21
  %nc.n573 = fsub <4 x float> %nc.n502, %nc.n567, !dbg !21
  %nc.n574 = fsub <4 x float> %nc.n510, %nc.n567, !dbg !21
  %nc.n575 = fsub <4 x float> %nc.n517, %nc.n567, !dbg !21
  %nc.n576 = fsub <1 x float> %nc.n542, %nc.n544, !dbg !22
  %nc.n577 = extractelement <1 x float> %nc.n576, i64 0, !dbg !23
  %nc.n578 = call float @llvm.amdgcn.exp2.f32(float %nc.n577), !dbg !23
  %nc.n579 = insertelement <4 x float> poison, float %nc.n578, i64 0, !dbg !28
  %nc.n580 = shufflevector <4 x float> %nc.n579, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %nc.n581 = extractvalue [2 x <1 x float>] %nc.n128, 0, !dbg !24
  %nc.n582 = extractelement <1 x float> %nc.n581, i64 0, !dbg !24
  %nc.n583 = fmul float %nc.n582, %nc.n578, !dbg !24
  %nc.n584 = fsub <1 x float> %nc.n555, %nc.n557, !dbg !22
  %nc.n585 = extractelement <1 x float> %nc.n584, i64 0, !dbg !23
  %nc.n586 = call float @llvm.amdgcn.exp2.f32(float %nc.n585), !dbg !23
  %nc.n587 = insertelement <4 x float> poison, float %nc.n586, i64 0, !dbg !28
  %nc.n588 = shufflevector <4 x float> %nc.n587, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %nc.n589 = extractvalue [2 x <1 x float>] %nc.n128, 1, !dbg !24
  %nc.n590 = extractelement <1 x float> %nc.n589, i64 0, !dbg !24
  %nc.n591 = fmul float %nc.n590, %nc.n586, !dbg !24
  %nc.n592 = extractelement <4 x float> %nc.n559, i64 0, !dbg !25
  %nc.n593 = call float @llvm.amdgcn.exp2.f32(float %nc.n592), !dbg !25
  %nc.n594 = insertelement <4 x float> poison, float %nc.n593, i64 0, !dbg !27
  %nc.n595 = extractelement <4 x float> %nc.n559, i64 1, !dbg !25
  %nc.n596 = call float @llvm.amdgcn.exp2.f32(float %nc.n595), !dbg !25
  %nc.n597 = insertelement <4 x float> %nc.n594, float %nc.n596, i64 1, !dbg !27
  %nc.n598 = extractelement <4 x float> %nc.n559, i64 2, !dbg !25
  %nc.n599 = call float @llvm.amdgcn.exp2.f32(float %nc.n598), !dbg !25
  %nc.n600 = insertelement <4 x float> %nc.n597, float %nc.n599, i64 2, !dbg !27
  %nc.n601 = extractelement <4 x float> %nc.n559, i64 3, !dbg !25
  %nc.n602 = call float @llvm.amdgcn.exp2.f32(float %nc.n601), !dbg !25
  %nc.n603 = insertelement <4 x float> %nc.n600, float %nc.n602, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n603) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n604 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n604) #9, !dbg !25, !frisk.register_lifetimes !1
  %nc.n605 = extractelement <4 x float> %nc.n560, i64 0, !dbg !25
  %nc.n606 = call float @llvm.amdgcn.exp2.f32(float %nc.n605), !dbg !25
  %nc.n607 = insertelement <4 x float> poison, float %nc.n606, i64 0, !dbg !27
  %local.reduce89 = fadd float %nc.n593, %nc.n606, !dbg !26, !frisk.local_reduction !1
  %nc.n608 = extractelement <4 x float> %nc.n560, i64 1, !dbg !25
  %nc.n609 = call float @llvm.amdgcn.exp2.f32(float %nc.n608), !dbg !25
  %nc.n610 = insertelement <4 x float> %nc.n607, float %nc.n609, i64 1, !dbg !27
  %local.reduce90 = fadd float %nc.n596, %nc.n609, !dbg !26, !frisk.local_reduction !1
  %nc.n611 = extractelement <4 x float> %nc.n560, i64 2, !dbg !25
  %nc.n612 = call float @llvm.amdgcn.exp2.f32(float %nc.n611), !dbg !25
  %nc.n613 = insertelement <4 x float> %nc.n610, float %nc.n612, i64 2, !dbg !27
  %local.reduce91 = fadd float %nc.n599, %nc.n612, !dbg !26, !frisk.local_reduction !1
  %nc.n614 = extractelement <4 x float> %nc.n560, i64 3, !dbg !25
  %nc.n615 = call float @llvm.amdgcn.exp2.f32(float %nc.n614), !dbg !25
  %nc.n616 = insertelement <4 x float> %nc.n613, float %nc.n615, i64 3, !dbg !27
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n616) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n617 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !27
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n617) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce92 = fadd float %nc.n602, %nc.n615, !dbg !26, !frisk.local_reduction !1
  %nc.n618 = extractelement <4 x float> %nc.n561, i64 0, !dbg !25
  %nc.n619 = call float @llvm.amdgcn.exp2.f32(float %nc.n618), !dbg !25
  %nc.n620 = insertelement <4 x float> poison, float %nc.n619, i64 0, !dbg !27
  %local.reduce93 = fadd float %local.reduce89, %nc.n619, !dbg !26, !frisk.local_reduction !1
  %nc.n621 = extractelement <4 x float> %nc.n561, i64 1, !dbg !25
  %nc.n622 = call float @llvm.amdgcn.exp2.f32(float %nc.n621), !dbg !25
  %nc.n623 = insertelement <4 x float> %nc.n620, float %nc.n622, i64 1, !dbg !27
  %local.reduce94 = fadd float %local.reduce90, %nc.n622, !dbg !26, !frisk.local_reduction !1
  %nc.n624 = extractelement <4 x float> %nc.n561, i64 2, !dbg !25
  %nc.n625 = call float @llvm.amdgcn.exp2.f32(float %nc.n624), !dbg !25
  %nc.n626 = insertelement <4 x float> %nc.n623, float %nc.n625, i64 2, !dbg !27
  %local.reduce95 = fadd float %local.reduce91, %nc.n625, !dbg !26, !frisk.local_reduction !1
  %nc.n627 = extractelement <4 x float> %nc.n561, i64 3, !dbg !25
  %nc.n628 = call float @llvm.amdgcn.exp2.f32(float %nc.n627), !dbg !25
  %nc.n629 = insertelement <4 x float> %nc.n626, float %nc.n628, i64 3, !dbg !27
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n629) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n630 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !27
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n630) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce96 = fadd float %local.reduce92, %nc.n628, !dbg !26, !frisk.local_reduction !1
  %nc.n631 = extractelement <4 x float> %nc.n562, i64 0, !dbg !25
  %nc.n632 = call float @llvm.amdgcn.exp2.f32(float %nc.n631), !dbg !25
  %nc.n633 = insertelement <4 x float> poison, float %nc.n632, i64 0, !dbg !27
  %local.reduce97 = fadd float %local.reduce93, %nc.n632, !dbg !26, !frisk.local_reduction !1
  %nc.n634 = extractelement <4 x float> %nc.n562, i64 1, !dbg !25
  %nc.n635 = call float @llvm.amdgcn.exp2.f32(float %nc.n634), !dbg !25
  %nc.n636 = insertelement <4 x float> %nc.n633, float %nc.n635, i64 1, !dbg !27
  %local.reduce98 = fadd float %local.reduce94, %nc.n635, !dbg !26, !frisk.local_reduction !1
  %nc.n637 = extractelement <4 x float> %nc.n562, i64 2, !dbg !25
  %nc.n638 = call float @llvm.amdgcn.exp2.f32(float %nc.n637), !dbg !25
  %nc.n639 = insertelement <4 x float> %nc.n636, float %nc.n638, i64 2, !dbg !27
  %local.reduce99 = fadd float %local.reduce95, %nc.n638, !dbg !26, !frisk.local_reduction !1
  %nc.n640 = extractelement <4 x float> %nc.n562, i64 3, !dbg !25
  %nc.n641 = call float @llvm.amdgcn.exp2.f32(float %nc.n640), !dbg !25
  %nc.n642 = insertelement <4 x float> %nc.n639, float %nc.n641, i64 3, !dbg !27
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n642) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n643 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !27
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n643) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce100 = fadd float %local.reduce96, %nc.n641, !dbg !26, !frisk.local_reduction !1
  %nc.n644 = extractelement <4 x float> %nc.n563, i64 0, !dbg !25
  %nc.n645 = call float @llvm.amdgcn.exp2.f32(float %nc.n644), !dbg !25
  %nc.n646 = insertelement <4 x float> poison, float %nc.n645, i64 0, !dbg !27
  %local.reduce101 = fadd float %local.reduce97, %nc.n645, !dbg !26, !frisk.local_reduction !1
  %nc.n647 = extractelement <4 x float> %nc.n563, i64 1, !dbg !25
  %nc.n648 = call float @llvm.amdgcn.exp2.f32(float %nc.n647), !dbg !25
  %nc.n649 = insertelement <4 x float> %nc.n646, float %nc.n648, i64 1, !dbg !27
  %local.reduce102 = fadd float %local.reduce98, %nc.n648, !dbg !26, !frisk.local_reduction !1
  %nc.n650 = extractelement <4 x float> %nc.n563, i64 2, !dbg !25
  %nc.n651 = call float @llvm.amdgcn.exp2.f32(float %nc.n650), !dbg !25
  %nc.n652 = insertelement <4 x float> %nc.n649, float %nc.n651, i64 2, !dbg !27
  %local.reduce103 = fadd float %local.reduce99, %nc.n651, !dbg !26, !frisk.local_reduction !1
  %nc.n653 = extractelement <4 x float> %nc.n563, i64 3, !dbg !25
  %nc.n654 = call float @llvm.amdgcn.exp2.f32(float %nc.n653), !dbg !25
  %nc.n655 = insertelement <4 x float> %nc.n652, float %nc.n654, i64 3, !dbg !27
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n655) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n656 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !27
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n656) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce104 = fadd float %local.reduce100, %nc.n654, !dbg !26, !frisk.local_reduction !1
  %nc.n657 = extractelement <4 x float> %nc.n564, i64 0, !dbg !25
  %nc.n658 = call float @llvm.amdgcn.exp2.f32(float %nc.n657), !dbg !25
  %nc.n659 = insertelement <4 x float> poison, float %nc.n658, i64 0, !dbg !27
  %local.reduce105 = fadd float %local.reduce101, %nc.n658, !dbg !26, !frisk.local_reduction !1
  %nc.n660 = extractelement <4 x float> %nc.n564, i64 1, !dbg !25
  %nc.n661 = call float @llvm.amdgcn.exp2.f32(float %nc.n660), !dbg !25
  %nc.n662 = insertelement <4 x float> %nc.n659, float %nc.n661, i64 1, !dbg !27
  %local.reduce106 = fadd float %local.reduce102, %nc.n661, !dbg !26, !frisk.local_reduction !1
  %nc.n663 = extractelement <4 x float> %nc.n564, i64 2, !dbg !25
  %nc.n664 = call float @llvm.amdgcn.exp2.f32(float %nc.n663), !dbg !25
  %nc.n665 = insertelement <4 x float> %nc.n662, float %nc.n664, i64 2, !dbg !27
  %local.reduce107 = fadd float %local.reduce103, %nc.n664, !dbg !26, !frisk.local_reduction !1
  %nc.n666 = extractelement <4 x float> %nc.n564, i64 3, !dbg !25
  %nc.n667 = call float @llvm.amdgcn.exp2.f32(float %nc.n666), !dbg !25
  %nc.n668 = insertelement <4 x float> %nc.n665, float %nc.n667, i64 3, !dbg !27
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n668) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n669 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !27
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n669) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce108 = fadd float %local.reduce104, %nc.n667, !dbg !26, !frisk.local_reduction !1
  %nc.n670 = extractelement <4 x float> %nc.n565, i64 0, !dbg !25
  %nc.n671 = call float @llvm.amdgcn.exp2.f32(float %nc.n670), !dbg !25
  %nc.n672 = insertelement <4 x float> poison, float %nc.n671, i64 0, !dbg !27
  %local.reduce109 = fadd float %local.reduce105, %nc.n671, !dbg !26, !frisk.local_reduction !1
  %nc.n673 = extractelement <4 x float> %nc.n565, i64 1, !dbg !25
  %nc.n674 = call float @llvm.amdgcn.exp2.f32(float %nc.n673), !dbg !25
  %nc.n675 = insertelement <4 x float> %nc.n672, float %nc.n674, i64 1, !dbg !27
  %local.reduce110 = fadd float %local.reduce106, %nc.n674, !dbg !26, !frisk.local_reduction !1
  %nc.n676 = extractelement <4 x float> %nc.n565, i64 2, !dbg !25
  %nc.n677 = call float @llvm.amdgcn.exp2.f32(float %nc.n676), !dbg !25
  %nc.n678 = insertelement <4 x float> %nc.n675, float %nc.n677, i64 2, !dbg !27
  %local.reduce111 = fadd float %local.reduce107, %nc.n677, !dbg !26, !frisk.local_reduction !1
  %nc.n679 = extractelement <4 x float> %nc.n565, i64 3, !dbg !25
  %nc.n680 = call float @llvm.amdgcn.exp2.f32(float %nc.n679), !dbg !25
  %nc.n681 = insertelement <4 x float> %nc.n678, float %nc.n680, i64 3, !dbg !27
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n681) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n682 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !27
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n682) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce112 = fadd float %local.reduce108, %nc.n680, !dbg !26, !frisk.local_reduction !1
  %nc.n683 = extractelement <4 x float> %nc.n566, i64 0, !dbg !25
  %nc.n684 = call float @llvm.amdgcn.exp2.f32(float %nc.n683), !dbg !25
  %nc.n685 = insertelement <4 x float> poison, float %nc.n684, i64 0, !dbg !27
  %local.reduce113 = fadd float %local.reduce109, %nc.n684, !dbg !26, !frisk.local_reduction !1
  %nc.n686 = extractelement <4 x float> %nc.n566, i64 1, !dbg !25
  %nc.n687 = call float @llvm.amdgcn.exp2.f32(float %nc.n686), !dbg !25
  %nc.n688 = insertelement <4 x float> %nc.n685, float %nc.n687, i64 1, !dbg !27
  %local.reduce114 = fadd float %local.reduce110, %nc.n687, !dbg !26, !frisk.local_reduction !1
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !26, !frisk.local_reduction !1
  %nc.n689 = extractelement <4 x float> %nc.n566, i64 2, !dbg !25
  %nc.n690 = call float @llvm.amdgcn.exp2.f32(float %nc.n689), !dbg !25
  %nc.n691 = insertelement <4 x float> %nc.n688, float %nc.n690, i64 2, !dbg !27
  %local.reduce115 = fadd float %local.reduce111, %nc.n690, !dbg !26, !frisk.local_reduction !1
  %nc.n692 = extractelement <4 x float> %nc.n566, i64 3, !dbg !25
  %nc.n693 = call float @llvm.amdgcn.exp2.f32(float %nc.n692), !dbg !25
  %nc.n694 = insertelement <4 x float> %nc.n691, float %nc.n693, i64 3, !dbg !27
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n694) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n695 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !27
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n695) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce116 = fadd float %local.reduce112, %nc.n693, !dbg !26, !frisk.local_reduction !1
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !26, !frisk.local_reduction !1
  %nc.n696 = fadd float %local.reduce117, %local.reduce118, !dbg !26, !frisk.local_reduction !1
  %nc.n697 = fadd float %nc.n696, %nc.n583, !dbg !26
  %nc.n698 = insertelement <1 x float> poison, float %nc.n697, i64 0, !dbg !7
  %nc.n699 = insertvalue [2 x <1 x float>] poison, <1 x float> %nc.n698, 0, !dbg !7
  %nc.n700 = extractelement <4 x float> %nc.n568, i64 0, !dbg !25
  %nc.n701 = call float @llvm.amdgcn.exp2.f32(float %nc.n700), !dbg !25
  %nc.n702 = insertelement <4 x float> poison, float %nc.n701, i64 0, !dbg !27
  %nc.n703 = extractelement <4 x float> %nc.n568, i64 1, !dbg !25
  %nc.n704 = call float @llvm.amdgcn.exp2.f32(float %nc.n703), !dbg !25
  %nc.n705 = insertelement <4 x float> %nc.n702, float %nc.n704, i64 1, !dbg !27
  %nc.n706 = extractelement <4 x float> %nc.n568, i64 2, !dbg !25
  %nc.n707 = call float @llvm.amdgcn.exp2.f32(float %nc.n706), !dbg !25
  %nc.n708 = insertelement <4 x float> %nc.n705, float %nc.n707, i64 2, !dbg !27
  %nc.n709 = extractelement <4 x float> %nc.n568, i64 3, !dbg !25
  %nc.n710 = call float @llvm.amdgcn.exp2.f32(float %nc.n709), !dbg !25
  %nc.n711 = insertelement <4 x float> %nc.n708, float %nc.n710, i64 3, !dbg !27
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n711) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n712 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !27
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n712) #9, !dbg !25, !frisk.register_lifetimes !1
  %nc.n713 = extractelement <4 x float> %nc.n569, i64 0, !dbg !25
  %nc.n714 = call float @llvm.amdgcn.exp2.f32(float %nc.n713), !dbg !25
  %nc.n715 = insertelement <4 x float> poison, float %nc.n714, i64 0, !dbg !27
  %local.reduce = fadd float %nc.n701, %nc.n714, !dbg !26, !frisk.local_reduction !1
  %nc.n716 = extractelement <4 x float> %nc.n569, i64 1, !dbg !25
  %nc.n717 = call float @llvm.amdgcn.exp2.f32(float %nc.n716), !dbg !25
  %nc.n718 = insertelement <4 x float> %nc.n715, float %nc.n717, i64 1, !dbg !27
  %local.reduce59 = fadd float %nc.n704, %nc.n717, !dbg !26, !frisk.local_reduction !1
  %nc.n719 = extractelement <4 x float> %nc.n569, i64 2, !dbg !25
  %nc.n720 = call float @llvm.amdgcn.exp2.f32(float %nc.n719), !dbg !25
  %nc.n721 = insertelement <4 x float> %nc.n718, float %nc.n720, i64 2, !dbg !27
  %local.reduce60 = fadd float %nc.n707, %nc.n720, !dbg !26, !frisk.local_reduction !1
  %nc.n722 = extractelement <4 x float> %nc.n569, i64 3, !dbg !25
  %nc.n723 = call float @llvm.amdgcn.exp2.f32(float %nc.n722), !dbg !25
  %nc.n724 = insertelement <4 x float> %nc.n721, float %nc.n723, i64 3, !dbg !27
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n724) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n725 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !27
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n725) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce61 = fadd float %nc.n710, %nc.n723, !dbg !26, !frisk.local_reduction !1
  %nc.n726 = extractelement <4 x float> %nc.n570, i64 0, !dbg !25
  %nc.n727 = call float @llvm.amdgcn.exp2.f32(float %nc.n726), !dbg !25
  %nc.n728 = insertelement <4 x float> poison, float %nc.n727, i64 0, !dbg !27
  %local.reduce62 = fadd float %local.reduce, %nc.n727, !dbg !26, !frisk.local_reduction !1
  %nc.n729 = extractelement <4 x float> %nc.n570, i64 1, !dbg !25
  %nc.n730 = call float @llvm.amdgcn.exp2.f32(float %nc.n729), !dbg !25
  %nc.n731 = insertelement <4 x float> %nc.n728, float %nc.n730, i64 1, !dbg !27
  %local.reduce63 = fadd float %local.reduce59, %nc.n730, !dbg !26, !frisk.local_reduction !1
  %nc.n732 = extractelement <4 x float> %nc.n570, i64 2, !dbg !25
  %nc.n733 = call float @llvm.amdgcn.exp2.f32(float %nc.n732), !dbg !25
  %nc.n734 = insertelement <4 x float> %nc.n731, float %nc.n733, i64 2, !dbg !27
  %local.reduce64 = fadd float %local.reduce60, %nc.n733, !dbg !26, !frisk.local_reduction !1
  %nc.n735 = extractelement <4 x float> %nc.n570, i64 3, !dbg !25
  %nc.n736 = call float @llvm.amdgcn.exp2.f32(float %nc.n735), !dbg !25
  %nc.n737 = insertelement <4 x float> %nc.n734, float %nc.n736, i64 3, !dbg !27
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n737) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n738 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !27
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n738) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce65 = fadd float %local.reduce61, %nc.n736, !dbg !26, !frisk.local_reduction !1
  %nc.n739 = extractelement <4 x float> %nc.n571, i64 0, !dbg !25
  %nc.n740 = call float @llvm.amdgcn.exp2.f32(float %nc.n739), !dbg !25
  %nc.n741 = insertelement <4 x float> poison, float %nc.n740, i64 0, !dbg !27
  %local.reduce66 = fadd float %local.reduce62, %nc.n740, !dbg !26, !frisk.local_reduction !1
  %nc.n742 = extractelement <4 x float> %nc.n571, i64 1, !dbg !25
  %nc.n743 = call float @llvm.amdgcn.exp2.f32(float %nc.n742), !dbg !25
  %nc.n744 = insertelement <4 x float> %nc.n741, float %nc.n743, i64 1, !dbg !27
  %local.reduce67 = fadd float %local.reduce63, %nc.n743, !dbg !26, !frisk.local_reduction !1
  %nc.n745 = extractelement <4 x float> %nc.n571, i64 2, !dbg !25
  %nc.n746 = call float @llvm.amdgcn.exp2.f32(float %nc.n745), !dbg !25
  %nc.n747 = insertelement <4 x float> %nc.n744, float %nc.n746, i64 2, !dbg !27
  %local.reduce68 = fadd float %local.reduce64, %nc.n746, !dbg !26, !frisk.local_reduction !1
  %nc.n748 = extractelement <4 x float> %nc.n571, i64 3, !dbg !25
  %nc.n749 = call float @llvm.amdgcn.exp2.f32(float %nc.n748), !dbg !25
  %nc.n750 = insertelement <4 x float> %nc.n747, float %nc.n749, i64 3, !dbg !27
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n750) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n751 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !27
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n751) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce69 = fadd float %local.reduce65, %nc.n749, !dbg !26, !frisk.local_reduction !1
  %nc.n752 = extractelement <4 x float> %nc.n572, i64 0, !dbg !25
  %nc.n753 = call float @llvm.amdgcn.exp2.f32(float %nc.n752), !dbg !25
  %nc.n754 = insertelement <4 x float> poison, float %nc.n753, i64 0, !dbg !27
  %local.reduce70 = fadd float %local.reduce66, %nc.n753, !dbg !26, !frisk.local_reduction !1
  %nc.n755 = extractelement <4 x float> %nc.n572, i64 1, !dbg !25
  %nc.n756 = call float @llvm.amdgcn.exp2.f32(float %nc.n755), !dbg !25
  %nc.n757 = insertelement <4 x float> %nc.n754, float %nc.n756, i64 1, !dbg !27
  %local.reduce71 = fadd float %local.reduce67, %nc.n756, !dbg !26, !frisk.local_reduction !1
  %nc.n758 = extractelement <4 x float> %nc.n572, i64 2, !dbg !25
  %nc.n759 = call float @llvm.amdgcn.exp2.f32(float %nc.n758), !dbg !25
  %nc.n760 = insertelement <4 x float> %nc.n757, float %nc.n759, i64 2, !dbg !27
  %local.reduce72 = fadd float %local.reduce68, %nc.n759, !dbg !26, !frisk.local_reduction !1
  %nc.n761 = extractelement <4 x float> %nc.n572, i64 3, !dbg !25
  %nc.n762 = call float @llvm.amdgcn.exp2.f32(float %nc.n761), !dbg !25
  %nc.n763 = insertelement <4 x float> %nc.n760, float %nc.n762, i64 3, !dbg !27
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n763) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n764 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !27
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n764) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce73 = fadd float %local.reduce69, %nc.n762, !dbg !26, !frisk.local_reduction !1
  %nc.n765 = extractelement <4 x float> %nc.n573, i64 0, !dbg !25
  %nc.n766 = call float @llvm.amdgcn.exp2.f32(float %nc.n765), !dbg !25
  %nc.n767 = insertelement <4 x float> poison, float %nc.n766, i64 0, !dbg !27
  %local.reduce74 = fadd float %local.reduce70, %nc.n766, !dbg !26, !frisk.local_reduction !1
  %nc.n768 = extractelement <4 x float> %nc.n573, i64 1, !dbg !25
  %nc.n769 = call float @llvm.amdgcn.exp2.f32(float %nc.n768), !dbg !25
  %nc.n770 = insertelement <4 x float> %nc.n767, float %nc.n769, i64 1, !dbg !27
  %local.reduce75 = fadd float %local.reduce71, %nc.n769, !dbg !26, !frisk.local_reduction !1
  %nc.n771 = extractelement <4 x float> %nc.n573, i64 2, !dbg !25
  %nc.n772 = call float @llvm.amdgcn.exp2.f32(float %nc.n771), !dbg !25
  %nc.n773 = insertelement <4 x float> %nc.n770, float %nc.n772, i64 2, !dbg !27
  %local.reduce76 = fadd float %local.reduce72, %nc.n772, !dbg !26, !frisk.local_reduction !1
  %nc.n774 = extractelement <4 x float> %nc.n573, i64 3, !dbg !25
  %nc.n775 = call float @llvm.amdgcn.exp2.f32(float %nc.n774), !dbg !25
  %nc.n776 = insertelement <4 x float> %nc.n773, float %nc.n775, i64 3, !dbg !27
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n776) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n777 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !27
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n777) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce77 = fadd float %local.reduce73, %nc.n775, !dbg !26, !frisk.local_reduction !1
  %nc.n778 = extractelement <4 x float> %nc.n574, i64 0, !dbg !25
  %nc.n779 = call float @llvm.amdgcn.exp2.f32(float %nc.n778), !dbg !25
  %nc.n780 = insertelement <4 x float> poison, float %nc.n779, i64 0, !dbg !27
  %local.reduce78 = fadd float %local.reduce74, %nc.n779, !dbg !26, !frisk.local_reduction !1
  %nc.n781 = extractelement <4 x float> %nc.n574, i64 1, !dbg !25
  %nc.n782 = call float @llvm.amdgcn.exp2.f32(float %nc.n781), !dbg !25
  %nc.n783 = insertelement <4 x float> %nc.n780, float %nc.n782, i64 1, !dbg !27
  %local.reduce79 = fadd float %local.reduce75, %nc.n782, !dbg !26, !frisk.local_reduction !1
  %nc.n784 = extractelement <4 x float> %nc.n574, i64 2, !dbg !25
  %nc.n785 = call float @llvm.amdgcn.exp2.f32(float %nc.n784), !dbg !25
  %nc.n786 = insertelement <4 x float> %nc.n783, float %nc.n785, i64 2, !dbg !27
  %local.reduce80 = fadd float %local.reduce76, %nc.n785, !dbg !26, !frisk.local_reduction !1
  %nc.n787 = extractelement <4 x float> %nc.n574, i64 3, !dbg !25
  %nc.n788 = call float @llvm.amdgcn.exp2.f32(float %nc.n787), !dbg !25
  %nc.n789 = insertelement <4 x float> %nc.n786, float %nc.n788, i64 3, !dbg !27
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n789) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n790 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !27
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n790) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce81 = fadd float %local.reduce77, %nc.n788, !dbg !26, !frisk.local_reduction !1
  %nc.n791 = extractelement <4 x float> %nc.n575, i64 0, !dbg !25
  %nc.n792 = call float @llvm.amdgcn.exp2.f32(float %nc.n791), !dbg !25
  %nc.n793 = insertelement <4 x float> poison, float %nc.n792, i64 0, !dbg !27
  %local.reduce82 = fadd float %local.reduce78, %nc.n792, !dbg !26, !frisk.local_reduction !1
  %nc.n794 = extractelement <4 x float> %nc.n575, i64 1, !dbg !25
  %nc.n795 = call float @llvm.amdgcn.exp2.f32(float %nc.n794), !dbg !25
  %nc.n796 = insertelement <4 x float> %nc.n793, float %nc.n795, i64 1, !dbg !27
  %local.reduce83 = fadd float %local.reduce79, %nc.n795, !dbg !26, !frisk.local_reduction !1
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !26, !frisk.local_reduction !1
  %nc.n797 = extractelement <4 x float> %nc.n575, i64 2, !dbg !25
  %nc.n798 = call float @llvm.amdgcn.exp2.f32(float %nc.n797), !dbg !25
  %nc.n799 = insertelement <4 x float> %nc.n796, float %nc.n798, i64 2, !dbg !27
  %local.reduce84 = fadd float %local.reduce80, %nc.n798, !dbg !26, !frisk.local_reduction !1
  %nc.n800 = extractelement <4 x float> %nc.n575, i64 3, !dbg !25
  %nc.n801 = call float @llvm.amdgcn.exp2.f32(float %nc.n800), !dbg !25
  %nc.n802 = insertelement <4 x float> %nc.n799, float %nc.n801, i64 3, !dbg !27
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n802) #9, !dbg !27, !frisk.register_lifetimes !1
  %nc.n803 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !27
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n803) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce85 = fadd float %local.reduce81, %nc.n801, !dbg !26, !frisk.local_reduction !1
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !26, !frisk.local_reduction !1
  %nc.n804 = fadd float %local.reduce86, %local.reduce87, !dbg !26, !frisk.local_reduction !1
  %nc.n805 = fadd float %nc.n804, %nc.n591, !dbg !26
  %nc.n806 = insertelement <1 x float> poison, float %nc.n805, i64 0, !dbg !7
  %nc.n807 = insertvalue [2 x <1 x float>] %nc.n699, <1 x float> %nc.n806, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !13
  %address.local930 = and i32 %address.local929, 120, !dbg !13
  %address.local931 = shl nuw nsw i32 %address.thread498, 5, !dbg !13
  %address.local932 = and i32 %address.local931, 1536, !dbg !13
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !13
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !13
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %nc.n808 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13
  %nc.n809 = bitcast <4 x i32> %nc.n808 to <8 x half>, !dbg !13
  %address.local936 = and i64 %address.local499, 60, !dbg !13
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !13
  %address.local938 = shl nuw nsw i64 %address.local937, 3, !dbg !13
  %address.local939 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local940 = lshr i64 %address.local555, 4, !dbg !13
  %address.local941 = mul nuw nsw i64 %address.local940, 384, !dbg !13
  %address.local942 = getelementptr [2 x i8], ptr addrspace(3) %address.local939, i64 %address.local941, !dbg !13
  %address.local943 = mul nsw i64 %address.local542, -32, !dbg !13
  %address.local944 = getelementptr [2 x i8], ptr addrspace(3) %address.local942, i64 %address.local943, !dbg !13
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !13
  %nc.n810 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13
  %nc.n811 = bitcast <4 x i32> %nc.n810 to <8 x half>, !dbg !13
  %address.local946 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local947 = getelementptr [2 x i8], ptr addrspace(3) %address.local946, i64 %address.local941, !dbg !13
  %address.local948 = getelementptr [2 x i8], ptr addrspace(3) %address.local947, i64 %address.local943, !dbg !13
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local948, i64 512, !dbg !13
  %nc.n812 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13
  %nc.n813 = bitcast <4 x i32> %nc.n812 to <8 x half>, !dbg !13
  %address.local950 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local951 = getelementptr [2 x i8], ptr addrspace(3) %address.local950, i64 %address.local941, !dbg !13
  %address.local952 = getelementptr [2 x i8], ptr addrspace(3) %address.local951, i64 %address.local943, !dbg !13
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local952, i64 768, !dbg !13
  %nc.n814 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13
  %nc.n815 = bitcast <4 x i32> %nc.n814 to <8 x half>, !dbg !13
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local954, i64 4096, !dbg !13
  %nc.n816 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13
  %nc.n817 = bitcast <4 x i32> %nc.n816 to <8 x half>, !dbg !13
  %address.local956 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) %address.local956, i64 %address.local941, !dbg !13
  %address.local958 = getelementptr [2 x i8], ptr addrspace(3) %address.local957, i64 %address.local943, !dbg !13
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local958, i64 4352, !dbg !13
  %nc.n818 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13
  %nc.n819 = bitcast <4 x i32> %nc.n818 to <8 x half>, !dbg !13
  %address.local960 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local961 = getelementptr [2 x i8], ptr addrspace(3) %address.local960, i64 %address.local941, !dbg !13
  %address.local962 = getelementptr [2 x i8], ptr addrspace(3) %address.local961, i64 %address.local943, !dbg !13
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4608, !dbg !13
  %nc.n820 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13
  %nc.n821 = bitcast <4 x i32> %nc.n820 to <8 x half>, !dbg !13
  %address.local964 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local965 = getelementptr [2 x i8], ptr addrspace(3) %address.local964, i64 %address.local941, !dbg !13
  %address.local966 = getelementptr [2 x i8], ptr addrspace(3) %address.local965, i64 %address.local943, !dbg !13
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local966, i64 4864, !dbg !13
  %nc.n822 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13
  %nc.n823 = bitcast <4 x i32> %nc.n822 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %nc.n824 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %nc.n825 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14231424 = or disjoint i64 %nc.n149, %address.local921, !dbg !15
  %.idx14231424.tr = trunc nuw i64 %.idx14231424 to i32, !dbg !15
  %dma.buffer.offset1342 = shl nuw nsw i32 %.idx14231424.tr, 1, !dbg !15
  %nc.n826 = insertelement <2 x i64> poison, i64 %nc.n825, i64 0, !dbg !15
  %nc.n827 = bitcast <2 x i64> %nc.n826 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1343 = shufflevector <4 x i32> %nc.n827, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1344 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n824), !dbg !15
  %nc.n828 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1344, i32 %dma.buffer.offset1342, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !15
  %nc.n829 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %nc.n830 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14251426 = or disjoint i64 %nc.n149, %address.local927, !dbg !15
  %.idx14251426.tr = trunc nuw i64 %.idx14251426 to i32, !dbg !15
  %dma.buffer.offset1339 = shl nuw nsw i32 %.idx14251426.tr, 1, !dbg !15
  %nc.n831 = insertelement <2 x i64> poison, i64 %nc.n830, i64 0, !dbg !15
  %nc.n832 = bitcast <2 x i64> %nc.n831 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1340 = shufflevector <4 x i32> %nc.n832, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1341 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n829), !dbg !15
  %nc.n833 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1341, i32 %dma.buffer.offset1339, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !15
  %nc.n834 = shufflevector <8 x half> %nc.n809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n835 = extractvalue [2 x <8 x float>] %nc.n127, 0, !dbg !28
  %nc.n836 = shufflevector <8 x float> %nc.n835, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %nc.n837 = fmul <4 x float> %nc.n836, %nc.n580, !dbg !28
  %nc.n838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n834, <4 x float> %nc.n837) #9, !dbg !13
  %nc.n839 = shufflevector <8 x half> %nc.n811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n839, <4 x float> %nc.n838) #9, !dbg !13
  %nc.n841 = shufflevector <8 x half> %nc.n813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n841, <4 x float> %nc.n840) #9, !dbg !13
  %nc.n843 = shufflevector <8 x half> %nc.n815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n843, <4 x float> %nc.n842) #9, !dbg !13
  %nc.n845 = shufflevector <8 x half> %nc.n817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n845, <4 x float> %nc.n844) #9, !dbg !13
  %nc.n847 = shufflevector <8 x half> %nc.n819, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n847, <4 x float> %nc.n846) #9, !dbg !13
  %nc.n849 = shufflevector <8 x half> %nc.n821, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n849, <4 x float> %nc.n848) #9, !dbg !13
  %nc.n851 = shufflevector <8 x half> %nc.n823, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n851, <4 x float> %nc.n850) #9, !dbg !13
  %nc.n853 = shufflevector <8 x half> %nc.n809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n854 = shufflevector <8 x float> %nc.n835, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %nc.n855 = fmul <4 x float> %nc.n854, %nc.n580, !dbg !28
  %nc.n856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n853, <4 x float> %nc.n855) #9, !dbg !13
  %nc.n857 = shufflevector <8 x half> %nc.n811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n857, <4 x float> %nc.n856) #9, !dbg !13
  %nc.n859 = shufflevector <8 x half> %nc.n813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n859, <4 x float> %nc.n858) #9, !dbg !13
  %nc.n861 = shufflevector <8 x half> %nc.n815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n861, <4 x float> %nc.n860) #9, !dbg !13
  %nc.n863 = shufflevector <8 x half> %nc.n817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n863, <4 x float> %nc.n862) #9, !dbg !13
  %nc.n865 = shufflevector <8 x half> %nc.n819, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n865, <4 x float> %nc.n864) #9, !dbg !13
  %nc.n867 = shufflevector <8 x half> %nc.n821, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n867, <4 x float> %nc.n866) #9, !dbg !13
  %nc.n869 = shufflevector <8 x half> %nc.n823, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n869, <4 x float> %nc.n868) #9, !dbg !13
  %nc.n871 = extractvalue [2 x <8 x float>] %nc.n127, 1, !dbg !28
  %nc.n872 = shufflevector <8 x float> %nc.n871, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %nc.n873 = fmul <4 x float> %nc.n872, %nc.n588, !dbg !28
  %nc.n874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n834, <4 x float> %nc.n873) #9, !dbg !13
  %nc.n875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n839, <4 x float> %nc.n874) #9, !dbg !13
  %nc.n876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n841, <4 x float> %nc.n875) #9, !dbg !13
  %nc.n877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n843, <4 x float> %nc.n876) #9, !dbg !13
  %nc.n878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n845, <4 x float> %nc.n877) #9, !dbg !13
  %nc.n879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n847, <4 x float> %nc.n878) #9, !dbg !13
  %nc.n880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n849, <4 x float> %nc.n879) #9, !dbg !13
  %nc.n881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n851, <4 x float> %nc.n880) #9, !dbg !13
  %nc.n882 = shufflevector <8 x float> %nc.n871, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %nc.n883 = fmul <4 x float> %nc.n882, %nc.n588, !dbg !28
  %nc.n884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n853, <4 x float> %nc.n883) #9, !dbg !13
  %nc.n885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n857, <4 x float> %nc.n884) #9, !dbg !13
  %nc.n886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n859, <4 x float> %nc.n885) #9, !dbg !13
  %nc.n887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n861, <4 x float> %nc.n886) #9, !dbg !13
  %nc.n888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n863, <4 x float> %nc.n887) #9, !dbg !13
  %nc.n889 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n865, <4 x float> %nc.n888) #9, !dbg !13
  %nc.n890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n867, <4 x float> %nc.n889) #9, !dbg !13
  %nc.n891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n869, <4 x float> %nc.n890) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %nc.n892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13
  %nc.n893 = bitcast <4 x i32> %nc.n892 to <8 x half>, !dbg !13
  %nc.n894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13
  %nc.n895 = bitcast <4 x i32> %nc.n894 to <8 x half>, !dbg !13
  %nc.n896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13
  %nc.n897 = bitcast <4 x i32> %nc.n896 to <8 x half>, !dbg !13
  %nc.n898 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13
  %nc.n899 = bitcast <4 x i32> %nc.n898 to <8 x half>, !dbg !13
  %nc.n900 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13
  %nc.n901 = bitcast <4 x i32> %nc.n900 to <8 x half>, !dbg !13
  %nc.n902 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13
  %nc.n903 = bitcast <4 x i32> %nc.n902 to <8 x half>, !dbg !13
  %nc.n904 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13
  %nc.n905 = bitcast <4 x i32> %nc.n904 to <8 x half>, !dbg !13
  %nc.n906 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13
  %nc.n907 = bitcast <4 x i32> %nc.n906 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %nc.n908 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %nc.n909 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14271428 = or disjoint i64 %nc.n232, %address.local921, !dbg !15
  %.idx14271428.tr = trunc nuw i64 %.idx14271428 to i32, !dbg !15
  %dma.buffer.offset1336 = shl nuw nsw i32 %.idx14271428.tr, 1, !dbg !15
  %nc.n910 = insertelement <2 x i64> poison, i64 %nc.n909, i64 0, !dbg !15
  %nc.n911 = bitcast <2 x i64> %nc.n910 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1337 = shufflevector <4 x i32> %nc.n911, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1338 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n908), !dbg !15
  %nc.n912 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1338, i32 %dma.buffer.offset1336, <4 x i32> %dma.buffer.rsrc1337) #10, !dbg !15
  %nc.n913 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %nc.n914 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14291430 = or disjoint i64 %nc.n232, %address.local927, !dbg !15
  %.idx14291430.tr = trunc nuw i64 %.idx14291430 to i32, !dbg !15
  %dma.buffer.offset1333 = shl nuw nsw i32 %.idx14291430.tr, 1, !dbg !15
  %nc.n915 = insertelement <2 x i64> poison, i64 %nc.n914, i64 0, !dbg !15
  %nc.n916 = bitcast <2 x i64> %nc.n915 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1334 = shufflevector <4 x i32> %nc.n916, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1335 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n913), !dbg !15
  %nc.n917 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1335, i32 %dma.buffer.offset1333, <4 x i32> %dma.buffer.rsrc1334) #10, !dbg !15
  %nc.n918 = shufflevector <8 x half> %nc.n893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n919 = extractvalue [2 x <8 x float>] %nc.n130, 0, !dbg !28
  %nc.n920 = shufflevector <8 x float> %nc.n919, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %nc.n921 = fmul <4 x float> %nc.n920, %nc.n580, !dbg !28
  %nc.n922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n918, <4 x float> %nc.n921) #9, !dbg !13
  %nc.n923 = shufflevector <8 x half> %nc.n895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n923, <4 x float> %nc.n922) #9, !dbg !13
  %nc.n925 = shufflevector <8 x half> %nc.n897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n925, <4 x float> %nc.n924) #9, !dbg !13
  %nc.n927 = shufflevector <8 x half> %nc.n899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n927, <4 x float> %nc.n926) #9, !dbg !13
  %nc.n929 = shufflevector <8 x half> %nc.n901, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n929, <4 x float> %nc.n928) #9, !dbg !13
  %nc.n931 = shufflevector <8 x half> %nc.n903, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n931, <4 x float> %nc.n930) #9, !dbg !13
  %nc.n933 = shufflevector <8 x half> %nc.n905, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n933, <4 x float> %nc.n932) #9, !dbg !13
  %nc.n935 = shufflevector <8 x half> %nc.n907, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n935, <4 x float> %nc.n934) #9, !dbg !13
  %nc.n937 = shufflevector <8 x half> %nc.n893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n938 = shufflevector <8 x float> %nc.n919, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %nc.n939 = fmul <4 x float> %nc.n938, %nc.n580, !dbg !28
  %nc.n940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n937, <4 x float> %nc.n939) #9, !dbg !13
  %nc.n941 = shufflevector <8 x half> %nc.n895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n941, <4 x float> %nc.n940) #9, !dbg !13
  %nc.n943 = shufflevector <8 x half> %nc.n897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n943, <4 x float> %nc.n942) #9, !dbg !13
  %nc.n945 = shufflevector <8 x half> %nc.n899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n945, <4 x float> %nc.n944) #9, !dbg !13
  %nc.n947 = shufflevector <8 x half> %nc.n901, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n947, <4 x float> %nc.n946) #9, !dbg !13
  %nc.n949 = shufflevector <8 x half> %nc.n903, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n949, <4 x float> %nc.n948) #9, !dbg !13
  %nc.n951 = shufflevector <8 x half> %nc.n905, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n951, <4 x float> %nc.n950) #9, !dbg !13
  %nc.n953 = shufflevector <8 x half> %nc.n907, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n953, <4 x float> %nc.n952) #9, !dbg !13
  %nc.n955 = extractvalue [2 x <8 x float>] %nc.n130, 1, !dbg !28
  %nc.n956 = shufflevector <8 x float> %nc.n955, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %nc.n957 = fmul <4 x float> %nc.n956, %nc.n588, !dbg !28
  %nc.n958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n918, <4 x float> %nc.n957) #9, !dbg !13
  %nc.n959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n923, <4 x float> %nc.n958) #9, !dbg !13
  %nc.n960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n925, <4 x float> %nc.n959) #9, !dbg !13
  %nc.n961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n927, <4 x float> %nc.n960) #9, !dbg !13
  %nc.n962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n929, <4 x float> %nc.n961) #9, !dbg !13
  %nc.n963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n931, <4 x float> %nc.n962) #9, !dbg !13
  %nc.n964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n933, <4 x float> %nc.n963) #9, !dbg !13
  %nc.n965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n935, <4 x float> %nc.n964) #9, !dbg !13
  %nc.n966 = shufflevector <8 x float> %nc.n955, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %nc.n967 = fmul <4 x float> %nc.n966, %nc.n588, !dbg !28
  %nc.n968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n937, <4 x float> %nc.n967) #9, !dbg !13
  %nc.n969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n941, <4 x float> %nc.n968) #9, !dbg !13
  %nc.n970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n943, <4 x float> %nc.n969) #9, !dbg !13
  %nc.n971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n945, <4 x float> %nc.n970) #9, !dbg !13
  %nc.n972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n947, <4 x float> %nc.n971) #9, !dbg !13
  %nc.n973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n949, <4 x float> %nc.n972) #9, !dbg !13
  %nc.n974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n951, <4 x float> %nc.n973) #9, !dbg !13
  %nc.n975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n953, <4 x float> %nc.n974) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %nc.n976 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13
  %nc.n977 = bitcast <4 x i32> %nc.n976 to <8 x half>, !dbg !13
  %nc.n978 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13
  %nc.n979 = bitcast <4 x i32> %nc.n978 to <8 x half>, !dbg !13
  %nc.n980 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13
  %nc.n981 = bitcast <4 x i32> %nc.n980 to <8 x half>, !dbg !13
  %nc.n982 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13
  %nc.n983 = bitcast <4 x i32> %nc.n982 to <8 x half>, !dbg !13
  %nc.n984 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13
  %nc.n985 = bitcast <4 x i32> %nc.n984 to <8 x half>, !dbg !13
  %nc.n986 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13
  %nc.n987 = bitcast <4 x i32> %nc.n986 to <8 x half>, !dbg !13
  %nc.n988 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13
  %nc.n989 = bitcast <4 x i32> %nc.n988 to <8 x half>, !dbg !13
  %nc.n990 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13
  %nc.n991 = bitcast <4 x i32> %nc.n990 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %nc.n992 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %nc.n993 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14311432 = or disjoint i64 %nc.n307, %address.local921, !dbg !15
  %.idx14311432.tr = trunc nuw i64 %.idx14311432 to i32, !dbg !15
  %dma.buffer.offset1330 = shl nuw nsw i32 %.idx14311432.tr, 1, !dbg !15
  %nc.n994 = insertelement <2 x i64> poison, i64 %nc.n993, i64 0, !dbg !15
  %nc.n995 = bitcast <2 x i64> %nc.n994 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1331 = shufflevector <4 x i32> %nc.n995, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1332 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n992), !dbg !15
  %nc.n996 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1332, i32 %dma.buffer.offset1330, <4 x i32> %dma.buffer.rsrc1331) #10, !dbg !15
  %nc.n997 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %nc.n998 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14331434 = or disjoint i64 %nc.n307, %address.local927, !dbg !15
  %.idx14331434.tr = trunc nuw i64 %.idx14331434 to i32, !dbg !15
  %dma.buffer.offset = shl nuw nsw i32 %.idx14331434.tr, 1, !dbg !15
  %nc.n999 = insertelement <2 x i64> poison, i64 %nc.n998, i64 0, !dbg !15
  %nc.n1000 = bitcast <2 x i64> %nc.n999 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc = shufflevector <4 x i32> %nc.n1000, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n997), !dbg !15
  %nc.n1001 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !15
  %nc.n1002 = shufflevector <8 x half> %nc.n977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1003 = extractvalue [2 x <8 x float>] %nc.n131, 0, !dbg !28
  %nc.n1004 = shufflevector <8 x float> %nc.n1003, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %nc.n1005 = fmul <4 x float> %nc.n1004, %nc.n580, !dbg !28
  %nc.n1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n1002, <4 x float> %nc.n1005) #9, !dbg !13
  %nc.n1007 = shufflevector <8 x half> %nc.n979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n1007, <4 x float> %nc.n1006) #9, !dbg !13
  %nc.n1009 = shufflevector <8 x half> %nc.n981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n1009, <4 x float> %nc.n1008) #9, !dbg !13
  %nc.n1011 = shufflevector <8 x half> %nc.n983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n1011, <4 x float> %nc.n1010) #9, !dbg !13
  %nc.n1013 = shufflevector <8 x half> %nc.n985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n1013, <4 x float> %nc.n1012) #9, !dbg !13
  %nc.n1015 = shufflevector <8 x half> %nc.n987, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n1015, <4 x float> %nc.n1014) #9, !dbg !13
  %nc.n1017 = shufflevector <8 x half> %nc.n989, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n1017, <4 x float> %nc.n1016) #9, !dbg !13
  %nc.n1019 = shufflevector <8 x half> %nc.n991, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n1019, <4 x float> %nc.n1018) #9, !dbg !13
  %nc.n1021 = shufflevector <8 x half> %nc.n977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1022 = shufflevector <8 x float> %nc.n1003, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %nc.n1023 = fmul <4 x float> %nc.n1022, %nc.n580, !dbg !28
  %nc.n1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n1021, <4 x float> %nc.n1023) #9, !dbg !13
  %nc.n1025 = shufflevector <8 x half> %nc.n979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n1025, <4 x float> %nc.n1024) #9, !dbg !13
  %nc.n1027 = shufflevector <8 x half> %nc.n981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n1027, <4 x float> %nc.n1026) #9, !dbg !13
  %nc.n1029 = shufflevector <8 x half> %nc.n983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n1029, <4 x float> %nc.n1028) #9, !dbg !13
  %nc.n1031 = shufflevector <8 x half> %nc.n985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n1031, <4 x float> %nc.n1030) #9, !dbg !13
  %nc.n1033 = shufflevector <8 x half> %nc.n987, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n1033, <4 x float> %nc.n1032) #9, !dbg !13
  %nc.n1035 = shufflevector <8 x half> %nc.n989, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n1035, <4 x float> %nc.n1034) #9, !dbg !13
  %nc.n1037 = shufflevector <8 x half> %nc.n991, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n1037, <4 x float> %nc.n1036) #9, !dbg !13
  %nc.n1039 = extractvalue [2 x <8 x float>] %nc.n131, 1, !dbg !28
  %nc.n1040 = shufflevector <8 x float> %nc.n1039, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %nc.n1041 = fmul <4 x float> %nc.n1040, %nc.n588, !dbg !28
  %nc.n1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n1002, <4 x float> %nc.n1041) #9, !dbg !13
  %nc.n1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n1007, <4 x float> %nc.n1042) #9, !dbg !13
  %nc.n1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n1009, <4 x float> %nc.n1043) #9, !dbg !13
  %nc.n1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n1011, <4 x float> %nc.n1044) #9, !dbg !13
  %nc.n1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n1013, <4 x float> %nc.n1045) #9, !dbg !13
  %nc.n1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n1015, <4 x float> %nc.n1046) #9, !dbg !13
  %nc.n1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n1017, <4 x float> %nc.n1047) #9, !dbg !13
  %nc.n1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n1019, <4 x float> %nc.n1048) #9, !dbg !13
  %nc.n1050 = shufflevector <8 x float> %nc.n1039, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %nc.n1051 = fmul <4 x float> %nc.n1050, %nc.n588, !dbg !28
  %nc.n1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n1021, <4 x float> %nc.n1051) #9, !dbg !13
  %nc.n1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n1025, <4 x float> %nc.n1052) #9, !dbg !13
  %nc.n1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n1027, <4 x float> %nc.n1053) #9, !dbg !13
  %nc.n1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n1029, <4 x float> %nc.n1054) #9, !dbg !13
  %nc.n1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n1031, <4 x float> %nc.n1055) #9, !dbg !13
  %nc.n1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n1033, <4 x float> %nc.n1056) #9, !dbg !13
  %nc.n1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n1035, <4 x float> %nc.n1057) #9, !dbg !13
  %nc.n1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n1037, <4 x float> %nc.n1058) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %nc.n1060 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13
  %nc.n1061 = bitcast <4 x i32> %nc.n1060 to <8 x half>, !dbg !13
  %nc.n1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13
  %nc.n1063 = bitcast <4 x i32> %nc.n1062 to <8 x half>, !dbg !13
  %nc.n1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13
  %nc.n1065 = bitcast <4 x i32> %nc.n1064 to <8 x half>, !dbg !13
  %nc.n1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13
  %nc.n1067 = bitcast <4 x i32> %nc.n1066 to <8 x half>, !dbg !13
  %nc.n1068 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13
  %nc.n1069 = bitcast <4 x i32> %nc.n1068 to <8 x half>, !dbg !13
  %nc.n1070 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13
  %nc.n1071 = bitcast <4 x i32> %nc.n1070 to <8 x half>, !dbg !13
  %nc.n1072 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13
  %nc.n1073 = bitcast <4 x i32> %nc.n1072 to <8 x half>, !dbg !13
  %nc.n1074 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13
  %nc.n1075 = bitcast <4 x i32> %nc.n1074 to <8 x half>, !dbg !13
; V3 readers complete before reusing low LDS.
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %nc.n1152 = add nuw nsw i64 %nc.n126, 1, !dbg !7
  %nc.n1076 = shufflevector <8 x half> %nc.n1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1077 = extractvalue [2 x <8 x float>] %nc.n132, 0, !dbg !28
  %nc.n1078 = shufflevector <8 x float> %nc.n1077, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %nc.n1079 = fmul <4 x float> %nc.n1078, %nc.n580, !dbg !28
  %nc.n1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n1076, <4 x float> %nc.n1079) #9, !dbg !13
  %nc.n1081 = shufflevector <8 x half> %nc.n1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n1081, <4 x float> %nc.n1080) #9, !dbg !13
  %nc.n1083 = shufflevector <8 x half> %nc.n1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n1083, <4 x float> %nc.n1082) #9, !dbg !13
  %nc.n1085 = shufflevector <8 x half> %nc.n1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n1085, <4 x float> %nc.n1084) #9, !dbg !13
  %nc.n1087 = shufflevector <8 x half> %nc.n1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n1087, <4 x float> %nc.n1086) #9, !dbg !13
  %nc.n1089 = shufflevector <8 x half> %nc.n1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n1089, <4 x float> %nc.n1088) #9, !dbg !13
  %nc.n1091 = shufflevector <8 x half> %nc.n1073, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n1091, <4 x float> %nc.n1090) #9, !dbg !13
  %nc.n1093 = shufflevector <8 x half> %nc.n1075, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %nc.n1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n1093, <4 x float> %nc.n1092) #9, !dbg !13
  %nc.n1095 = shufflevector <8 x half> %nc.n1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1096 = shufflevector <8 x float> %nc.n1077, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %nc.n1097 = fmul <4 x float> %nc.n1096, %nc.n580, !dbg !28
  %nc.n1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n1095, <4 x float> %nc.n1097) #9, !dbg !13
  %nc.n1099 = shufflevector <8 x half> %nc.n1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n1099, <4 x float> %nc.n1098) #9, !dbg !13
  %nc.n1101 = shufflevector <8 x half> %nc.n1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n1101, <4 x float> %nc.n1100) #9, !dbg !13
  %nc.n1103 = shufflevector <8 x half> %nc.n1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n1103, <4 x float> %nc.n1102) #9, !dbg !13
  %nc.n1105 = shufflevector <8 x half> %nc.n1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n1105, <4 x float> %nc.n1104) #9, !dbg !13
  %nc.n1107 = shufflevector <8 x half> %nc.n1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n1107, <4 x float> %nc.n1106) #9, !dbg !13
  %nc.n1109 = shufflevector <8 x half> %nc.n1073, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n1109, <4 x float> %nc.n1108) #9, !dbg !13
  %nc.n1111 = shufflevector <8 x half> %nc.n1075, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %nc.n1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n1111, <4 x float> %nc.n1110) #9, !dbg !13
  %nc.n1113 = extractvalue [2 x <8 x float>] %nc.n132, 1, !dbg !28
  %nc.n1114 = shufflevector <8 x float> %nc.n1113, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %nc.n1115 = fmul <4 x float> %nc.n1114, %nc.n588, !dbg !28
  %nc.n1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n1076, <4 x float> %nc.n1115) #9, !dbg !13
  %nc.n1117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n1081, <4 x float> %nc.n1116) #9, !dbg !13
  %nc.n1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n1083, <4 x float> %nc.n1117) #9, !dbg !13
  %nc.n1119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n1085, <4 x float> %nc.n1118) #9, !dbg !13
  %nc.n1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n1087, <4 x float> %nc.n1119) #9, !dbg !13
  %nc.n1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n1089, <4 x float> %nc.n1120) #9, !dbg !13
  %nc.n1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n1091, <4 x float> %nc.n1121) #9, !dbg !13
  %nc.n1123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n1093, <4 x float> %nc.n1122) #9, !dbg !13
  %nc.n1124 = shufflevector <8 x float> %nc.n1113, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %nc.n1125 = fmul <4 x float> %nc.n1124, %nc.n588, !dbg !28
  %nc.n1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n1095, <4 x float> %nc.n1125) #9, !dbg !13
  %nc.n1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n1099, <4 x float> %nc.n1126) #9, !dbg !13
  %nc.n1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n1101, <4 x float> %nc.n1127) #9, !dbg !13
  %nc.n1129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n1103, <4 x float> %nc.n1128) #9, !dbg !13
  %nc.n1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n1105, <4 x float> %nc.n1129) #9, !dbg !13
  %nc.n1131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n1107, <4 x float> %nc.n1130) #9, !dbg !13
  %nc.n1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n1109, <4 x float> %nc.n1131) #9, !dbg !13
  %nc.n1133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n1111, <4 x float> %nc.n1132) #9, !dbg !13
  %nc.n1134 = shufflevector <4 x float> %nc.n852, <4 x float> %nc.n870, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %nc.n1135 = insertvalue [2 x <8 x float>] poison, <8 x float> %nc.n1134, 0, !dbg !7
  %nc.n1136 = shufflevector <4 x float> %nc.n881, <4 x float> %nc.n891, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %nc.n1137 = insertvalue [2 x <8 x float>] %nc.n1135, <8 x float> %nc.n1136, 1, !dbg !7
  %nc.n1138 = insertvalue [2 x <1 x float>] poison, <1 x float> %nc.n544, 0, !dbg !7
  %nc.n1139 = insertvalue [2 x <1 x float>] %nc.n1138, <1 x float> %nc.n557, 1, !dbg !7
  %nc.n1140 = shufflevector <4 x float> %nc.n936, <4 x float> %nc.n954, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %nc.n1141 = insertvalue [2 x <8 x float>] poison, <8 x float> %nc.n1140, 0, !dbg !7
  %nc.n1142 = shufflevector <4 x float> %nc.n965, <4 x float> %nc.n975, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %nc.n1143 = insertvalue [2 x <8 x float>] %nc.n1141, <8 x float> %nc.n1142, 1, !dbg !7
  %nc.n1144 = shufflevector <4 x float> %nc.n1020, <4 x float> %nc.n1038, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %nc.n1145 = insertvalue [2 x <8 x float>] poison, <8 x float> %nc.n1144, 0, !dbg !7
  %nc.n1146 = shufflevector <4 x float> %nc.n1049, <4 x float> %nc.n1059, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %nc.n1147 = insertvalue [2 x <8 x float>] %nc.n1145, <8 x float> %nc.n1146, 1, !dbg !7
  %nc.n1148 = shufflevector <4 x float> %nc.n1094, <4 x float> %nc.n1112, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %nc.n1149 = insertvalue [2 x <8 x float>] poison, <8 x float> %nc.n1148, 0, !dbg !7
  %nc.n1150 = shufflevector <4 x float> %nc.n1123, <4 x float> %nc.n1133, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %nc.n1151 = insertvalue [2 x <8 x float>] %nc.n1149, <8 x float> %nc.n1150, 1, !dbg !7
  %k0.has.next = icmp ult i64 %nc.n1152, 32
  br i1 %k0.has.next, label %k0.issue, label %k0.join

k0.issue:
  %k0.next.address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %k0.next.address.local499 = zext nneg i32 %k0.next.address.thread498 to i64, !dbg !8
  %k0.next.address.local500 = shl nuw nsw i64 %k0.next.address.local499, 3, !dbg !8
  %k0.next.address.local501 = and i64 %k0.next.address.local500, 1536, !dbg !8
  %k0.next.address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.next.address.local501, !dbg !8
  %k0.next.address.local503 = ptrtoint ptr addrspace(3) %k0.next.address.local502 to i64, !dbg !8
  %k0.next.address.local504 = trunc i64 %k0.next.address.local503 to i32, !dbg !8
  %k0.next.nc.n139 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.address.local504), !dbg !12
  %k0.next.dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.nc.n139), !dbg !12
  %k0.next.nc.n135 = shl i64 %nc.n16, 19, !dbg !12
  %k0.next.nc.n136 = and i64 %k0.next.nc.n135, 16252928, !dbg !12
  %k0.next.nc.n137 = shl nuw nsw i64 %nc.n1152, 14, !dbg !12
  %k0.next.nc.n138 = or disjoint i64 %k0.next.nc.n136, %k0.next.nc.n137, !dbg !12
  %k0.next.address.local505 = lshr i64 %k0.next.address.local499, 4, !dbg !12
  %k0.next.address.local506 = and i64 %k0.next.address.local505, 8, !dbg !12
  %k0.next.address.local507 = and i64 %k0.next.address.local499, 123, !dbg !12
  %k0.next.address.local508 = lshr i64 %k0.next.address.local499, 2, !dbg !12
  %k0.next.address.local509 = and i64 %k0.next.address.local508, 4, !dbg !12
  %k0.next.address.local510 = add nuw nsw i64 %k0.next.address.local509, %k0.next.address.local499, !dbg !12
  %k0.next.address.local511 = and i64 %k0.next.address.local510, 4, !dbg !12
  %k0.next.address.local512 = or disjoint i64 %k0.next.address.local507, %k0.next.address.local511, !dbg !12
  %k0.next.address.local513 = shl nuw nsw i64 %k0.next.address.local512, 7, !dbg !12
  %k0.next.address.local514 = or disjoint i64 %k0.next.address.local506, %k0.next.address.local513, !dbg !12
  %k0.next..idx1406 = or disjoint i64 %k0.next.nc.n138, %k0.next.address.local514, !dbg !12
  %k0.next..idx1406.tr = trunc nuw i64 %k0.next..idx1406 to i32, !dbg !12
  %k0.next.dma.buffer.offset1372 = shl nuw nsw i32 %k0.next..idx1406.tr, 1, !dbg !12
  %k0.next.nc.n140 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %k0.next.nc.n141 = insertelement <2 x i64> poison, i64 %k0.next.nc.n140, i64 0, !dbg !12
  %k0.next.nc.n142 = bitcast <2 x i64> %k0.next.nc.n141 to <4 x i32>, !dbg !12
  %k0.next.dma.buffer.rsrc1373 = shufflevector <4 x i32> %k0.next.nc.n142, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %k0.next.nc.n143 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.next.dma.buffer.lds1374, i32 %k0.next.dma.buffer.offset1372, <4 x i32> %k0.next.dma.buffer.rsrc1373) #10, !dbg !12
  %k0.next.address.local516 = or disjoint i64 %k0.next.address.local499, 256, !dbg !8
  %k0.next.address.local517 = shl nuw nsw i64 %k0.next.address.local516, 3, !dbg !8
  %k0.next.address.local518 = and i64 %k0.next.address.local517, 3584, !dbg !8
  %k0.next.address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.next.address.local518, !dbg !8
  %k0.next.address.local520 = ptrtoint ptr addrspace(3) %k0.next.address.local519 to i64, !dbg !8
  %k0.next.address.local521 = trunc i64 %k0.next.address.local520 to i32, !dbg !8
  %k0.next.nc.n144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.address.local521), !dbg !12
  %k0.next.dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.nc.n144), !dbg !12
  %k0.next.address.local522 = lshr i64 %k0.next.address.local516, 4, !dbg !12
  %k0.next.address.local523 = and i64 %k0.next.address.local522, 24, !dbg !12
  %k0.next.address.local524 = and i64 %k0.next.address.local499, 123, !dbg !12
  %k0.next.address.local525 = lshr i64 %k0.next.address.local499, 2, !dbg !12
  %k0.next.address.local526 = and i64 %k0.next.address.local525, 4, !dbg !12
  %k0.next.address.local527 = add nuw nsw i64 %k0.next.address.local526, %k0.next.address.local499, !dbg !12
  %k0.next.address.local528 = and i64 %k0.next.address.local527, 4, !dbg !12
  %k0.next.address.local529 = or disjoint i64 %k0.next.address.local524, %k0.next.address.local528, !dbg !12
  %k0.next.address.local530 = shl nuw nsw i64 %k0.next.address.local529, 7, !dbg !12
  %k0.next.address.local531 = or disjoint i64 %k0.next.address.local523, %k0.next.address.local530, !dbg !12
  %k0.next..idx1407 = or disjoint i64 %k0.next.nc.n138, %k0.next.address.local531, !dbg !12
  %k0.next..idx1407.tr = trunc nuw i64 %k0.next..idx1407 to i32, !dbg !12
  %k0.next.dma.buffer.offset1369 = shl nuw nsw i32 %k0.next..idx1407.tr, 1, !dbg !12
  %k0.next.nc.n145 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %k0.next.nc.n146 = insertelement <2 x i64> poison, i64 %k0.next.nc.n145, i64 0, !dbg !12
  %k0.next.nc.n147 = bitcast <2 x i64> %k0.next.nc.n146 to <4 x i32>, !dbg !12
  %k0.next.dma.buffer.rsrc1370 = shufflevector <4 x i32> %k0.next.nc.n147, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %k0.next.nc.n148 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.next.dma.buffer.lds1371, i32 %k0.next.dma.buffer.offset1369, <4 x i32> %k0.next.dma.buffer.rsrc1370) #10, !dbg !12
  br label %k0.join

k0.join:
  br label %nc.n125, !dbg !7

nc.n1153:                                             ; preds = %nc.n125
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %nc.n128, 1, !dbg !7
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !7
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !7
  %nc.n1154 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %nc.n1155 = xor i32 %nc.n1154, 64, !dbg !7
  %nc.n1156 = bitcast float %lane.sum.local185 to i32, !dbg !7
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1155, i32 %nc.n1156), !dbg !7
  %nc.n1157 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !7
  %lane.sum.final189 = fadd float %lane.sum.local185, %nc.n1157, !dbg !7
  %nc.n1158 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %nc.n1159 = xor i32 %nc.n1158, 128, !dbg !7
  %nc.n1160 = bitcast float %lane.sum.final189 to i32, !dbg !7
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1159, i32 %nc.n1160), !dbg !7
  %nc.n1161 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !7
  %lane.sum.final191 = fadd float %lane.sum.final189, %nc.n1161, !dbg !7
  %nc.n1162 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %nc.n128, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !7
  %nc.n1163 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %nc.n1164 = xor i32 %nc.n1163, 64, !dbg !7
  %nc.n1165 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1164, i32 %nc.n1165), !dbg !7
  %nc.n1166 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %nc.n1166, !dbg !7
  %nc.n1167 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %nc.n1168 = xor i32 %nc.n1167, 128, !dbg !7
  %nc.n1169 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1168, i32 %nc.n1169), !dbg !7
  %nc.n1170 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !7
  %lane.sum.final183 = fadd float %lane.sum.final, %nc.n1170, !dbg !7
  %nc.n1171 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !7
  %nc.n1172 = and i32 %nc.n31, 16, !dbg !7
  %.not = icmp eq i32 %nc.n1172, 0, !dbg !7
  %nc.n1173 = and i32 %nc.n31, 32, !dbg !7
  %.not36 = icmp eq i32 %nc.n1173, 0, !dbg !7
  %nc.n1174 = extractvalue [2 x <8 x float>] %nc.n132, 0, !dbg !7
  %nc.n1175 = extractelement <8 x float> %nc.n1174, i64 0, !dbg !7
  %nc.n1176 = extractelement <8 x float> %nc.n1174, i64 1, !dbg !7
  %nc.n1177 = extractelement <8 x float> %nc.n1174, i64 2, !dbg !7
  %nc.n1178 = extractelement <8 x float> %nc.n1174, i64 3, !dbg !7
  %nc.n1179 = select i1 %.not, float %nc.n1176, float %nc.n1175, !dbg !7
  %nc.n1180 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %nc.n1181 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %nc.n1180), !dbg !7
  %nc.n1182 = add nuw nsw i32 %nc.n1181, 64, !dbg !7
  %nc.n1183 = xor i32 %nc.n1181, 16, !dbg !7
  %nc.n1184 = and i32 %nc.n1182, 192, !dbg !7
  %nc.n1185 = icmp samesign ult i32 %nc.n1183, %nc.n1184, !dbg !7
  %nc.n1186 = select i1 %nc.n1185, i32 %nc.n1183, i32 %nc.n1181, !dbg !7
  %nc.n1187 = shl nuw nsw i32 %nc.n1186, 2, !dbg !7
  %nc.n1188 = bitcast float %nc.n1179 to i32, !dbg !7
  %nc.n1189 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1188), !dbg !7
  %nc.n1190 = bitcast i32 %nc.n1189 to float, !dbg !7
  %nc.n1191 = select i1 %.not, float %nc.n1175, float %nc.n1190, !dbg !7
  %nc.n1192 = select i1 %.not, float %nc.n1190, float %nc.n1176, !dbg !7
  %nc.n1193 = select i1 %.not, float %nc.n1178, float %nc.n1177, !dbg !7
  %nc.n1194 = bitcast float %nc.n1193 to i32, !dbg !7
  %nc.n1195 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1194), !dbg !7
  %nc.n1196 = bitcast i32 %nc.n1195 to float, !dbg !7
  %nc.n1197 = select i1 %.not, float %nc.n1177, float %nc.n1196, !dbg !7
  %nc.n1198 = select i1 %.not, float %nc.n1196, float %nc.n1178, !dbg !7
  %nc.n1199 = select i1 %.not36, float %nc.n1197, float %nc.n1191, !dbg !7
  %nc.n1200 = xor i32 %nc.n1181, 32, !dbg !7
  %nc.n1201 = icmp samesign ult i32 %nc.n1200, %nc.n1184, !dbg !7
  %nc.n1202 = select i1 %nc.n1201, i32 %nc.n1200, i32 %nc.n1181, !dbg !7
  %nc.n1203 = shl nuw nsw i32 %nc.n1202, 2, !dbg !7
  %nc.n1204 = bitcast float %nc.n1199 to i32, !dbg !7
  %nc.n1205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1204), !dbg !7
  %nc.n1206 = bitcast i32 %nc.n1205 to float, !dbg !7
  %nc.n1207 = select i1 %.not36, float %nc.n1191, float %nc.n1206, !dbg !7
  %nc.n1208 = select i1 %.not36, float %nc.n1206, float %nc.n1197, !dbg !7
  %nc.n1209 = select i1 %.not36, float %nc.n1198, float %nc.n1192, !dbg !7
  %nc.n1210 = bitcast float %nc.n1209 to i32, !dbg !7
  %nc.n1211 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1210), !dbg !7
  %nc.n1212 = bitcast i32 %nc.n1211 to float, !dbg !7
  %nc.n1213 = select i1 %.not36, float %nc.n1192, float %nc.n1212, !dbg !7
  %nc.n1214 = select i1 %.not36, float %nc.n1212, float %nc.n1198, !dbg !7
  %nc.n1215 = extractvalue [2 x <8 x float>] %nc.n131, 0, !dbg !7
  %nc.n1216 = extractelement <8 x float> %nc.n1215, i64 0, !dbg !7
  %nc.n1217 = extractelement <8 x float> %nc.n1215, i64 1, !dbg !7
  %nc.n1218 = extractelement <8 x float> %nc.n1215, i64 2, !dbg !7
  %nc.n1219 = extractelement <8 x float> %nc.n1215, i64 3, !dbg !7
  %nc.n1220 = select i1 %.not, float %nc.n1217, float %nc.n1216, !dbg !7
  %nc.n1221 = bitcast float %nc.n1220 to i32, !dbg !7
  %nc.n1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1221), !dbg !7
  %nc.n1223 = bitcast i32 %nc.n1222 to float, !dbg !7
  %nc.n1224 = select i1 %.not, float %nc.n1216, float %nc.n1223, !dbg !7
  %nc.n1225 = select i1 %.not, float %nc.n1223, float %nc.n1217, !dbg !7
  %nc.n1226 = select i1 %.not, float %nc.n1219, float %nc.n1218, !dbg !7
  %nc.n1227 = bitcast float %nc.n1226 to i32, !dbg !7
  %nc.n1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1227), !dbg !7
  %nc.n1229 = bitcast i32 %nc.n1228 to float, !dbg !7
  %nc.n1230 = select i1 %.not, float %nc.n1218, float %nc.n1229, !dbg !7
  %nc.n1231 = select i1 %.not, float %nc.n1229, float %nc.n1219, !dbg !7
  %nc.n1232 = select i1 %.not36, float %nc.n1230, float %nc.n1224, !dbg !7
  %nc.n1233 = bitcast float %nc.n1232 to i32, !dbg !7
  %nc.n1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1233), !dbg !7
  %nc.n1235 = bitcast i32 %nc.n1234 to float, !dbg !7
  %nc.n1236 = select i1 %.not36, float %nc.n1224, float %nc.n1235, !dbg !7
  %nc.n1237 = select i1 %.not36, float %nc.n1235, float %nc.n1230, !dbg !7
  %nc.n1238 = select i1 %.not36, float %nc.n1231, float %nc.n1225, !dbg !7
  %nc.n1239 = bitcast float %nc.n1238 to i32, !dbg !7
  %nc.n1240 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1239), !dbg !7
  %nc.n1241 = bitcast i32 %nc.n1240 to float, !dbg !7
  %nc.n1242 = select i1 %.not36, float %nc.n1225, float %nc.n1241, !dbg !7
  %nc.n1243 = select i1 %.not36, float %nc.n1241, float %nc.n1231, !dbg !7
  %nc.n1244 = extractvalue [2 x <8 x float>] %nc.n130, 0, !dbg !7
  %nc.n1245 = extractelement <8 x float> %nc.n1244, i64 0, !dbg !7
  %nc.n1246 = extractelement <8 x float> %nc.n1244, i64 1, !dbg !7
  %nc.n1247 = extractelement <8 x float> %nc.n1244, i64 2, !dbg !7
  %nc.n1248 = extractelement <8 x float> %nc.n1244, i64 3, !dbg !7
  %nc.n1249 = select i1 %.not, float %nc.n1246, float %nc.n1245, !dbg !7
  %nc.n1250 = bitcast float %nc.n1249 to i32, !dbg !7
  %nc.n1251 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1250), !dbg !7
  %nc.n1252 = bitcast i32 %nc.n1251 to float, !dbg !7
  %nc.n1253 = select i1 %.not, float %nc.n1245, float %nc.n1252, !dbg !7
  %nc.n1254 = select i1 %.not, float %nc.n1252, float %nc.n1246, !dbg !7
  %nc.n1255 = select i1 %.not, float %nc.n1248, float %nc.n1247, !dbg !7
  %nc.n1256 = bitcast float %nc.n1255 to i32, !dbg !7
  %nc.n1257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1256), !dbg !7
  %nc.n1258 = bitcast i32 %nc.n1257 to float, !dbg !7
  %nc.n1259 = select i1 %.not, float %nc.n1247, float %nc.n1258, !dbg !7
  %nc.n1260 = select i1 %.not, float %nc.n1258, float %nc.n1248, !dbg !7
  %nc.n1261 = select i1 %.not36, float %nc.n1259, float %nc.n1253, !dbg !7
  %nc.n1262 = bitcast float %nc.n1261 to i32, !dbg !7
  %nc.n1263 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1262), !dbg !7
  %nc.n1264 = bitcast i32 %nc.n1263 to float, !dbg !7
  %nc.n1265 = select i1 %.not36, float %nc.n1253, float %nc.n1264, !dbg !7
  %nc.n1266 = select i1 %.not36, float %nc.n1264, float %nc.n1259, !dbg !7
  %nc.n1267 = select i1 %.not36, float %nc.n1260, float %nc.n1254, !dbg !7
  %nc.n1268 = bitcast float %nc.n1267 to i32, !dbg !7
  %nc.n1269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1268), !dbg !7
  %nc.n1270 = bitcast i32 %nc.n1269 to float, !dbg !7
  %nc.n1271 = select i1 %.not36, float %nc.n1254, float %nc.n1270, !dbg !7
  %nc.n1272 = select i1 %.not36, float %nc.n1270, float %nc.n1260, !dbg !7
  %nc.n1273 = extractvalue [2 x <8 x float>] %nc.n127, 0, !dbg !7
  %nc.n1274 = extractelement <8 x float> %nc.n1273, i64 0, !dbg !7
  %nc.n1275 = extractelement <8 x float> %nc.n1273, i64 1, !dbg !7
  %nc.n1276 = extractelement <8 x float> %nc.n1273, i64 2, !dbg !7
  %nc.n1277 = extractelement <8 x float> %nc.n1273, i64 3, !dbg !7
  %nc.n1278 = select i1 %.not, float %nc.n1275, float %nc.n1274, !dbg !7
  %nc.n1279 = bitcast float %nc.n1278 to i32, !dbg !7
  %nc.n1280 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1279), !dbg !7
  %nc.n1281 = bitcast i32 %nc.n1280 to float, !dbg !7
  %nc.n1282 = select i1 %.not, float %nc.n1274, float %nc.n1281, !dbg !7
  %nc.n1283 = select i1 %.not, float %nc.n1281, float %nc.n1275, !dbg !7
  %nc.n1284 = select i1 %.not, float %nc.n1277, float %nc.n1276, !dbg !7
  %nc.n1285 = bitcast float %nc.n1284 to i32, !dbg !7
  %nc.n1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1285), !dbg !7
  %nc.n1287 = bitcast i32 %nc.n1286 to float, !dbg !7
  %nc.n1288 = select i1 %.not, float %nc.n1276, float %nc.n1287, !dbg !7
  %nc.n1289 = select i1 %.not, float %nc.n1287, float %nc.n1277, !dbg !7
  %nc.n1290 = select i1 %.not36, float %nc.n1288, float %nc.n1282, !dbg !7
  %nc.n1291 = bitcast float %nc.n1290 to i32, !dbg !7
  %nc.n1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1291), !dbg !7
  %nc.n1293 = select i1 %.not36, float %nc.n1289, float %nc.n1283, !dbg !7
  %nc.n1294 = bitcast float %nc.n1293 to i32, !dbg !7
  %nc.n1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1294), !dbg !7
  %nc.n1296 = shufflevector <1 x float> %nc.n1171, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %nc.n1297 = extractelement <8 x float> %nc.n1174, i64 4, !dbg !7
  %nc.n1298 = extractelement <8 x float> %nc.n1174, i64 5, !dbg !7
  %nc.n1299 = extractelement <8 x float> %nc.n1174, i64 6, !dbg !7
  %nc.n1300 = extractelement <8 x float> %nc.n1174, i64 7, !dbg !7
  %nc.n1301 = select i1 %.not, float %nc.n1298, float %nc.n1297, !dbg !7
  %nc.n1302 = bitcast float %nc.n1301 to i32, !dbg !7
  %nc.n1303 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1302), !dbg !7
  %nc.n1304 = bitcast i32 %nc.n1303 to float, !dbg !7
  %nc.n1305 = select i1 %.not, float %nc.n1297, float %nc.n1304, !dbg !7
  %nc.n1306 = select i1 %.not, float %nc.n1304, float %nc.n1298, !dbg !7
  %nc.n1307 = select i1 %.not, float %nc.n1300, float %nc.n1299, !dbg !7
  %nc.n1308 = bitcast float %nc.n1307 to i32, !dbg !7
  %nc.n1309 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1308), !dbg !7
  %nc.n1310 = bitcast i32 %nc.n1309 to float, !dbg !7
  %nc.n1311 = select i1 %.not, float %nc.n1299, float %nc.n1310, !dbg !7
  %nc.n1312 = select i1 %.not, float %nc.n1310, float %nc.n1300, !dbg !7
  %nc.n1313 = select i1 %.not36, float %nc.n1311, float %nc.n1305, !dbg !7
  %nc.n1314 = bitcast float %nc.n1313 to i32, !dbg !7
  %nc.n1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1314), !dbg !7
  %nc.n1316 = bitcast i32 %nc.n1315 to float, !dbg !7
  %nc.n1317 = select i1 %.not36, float %nc.n1305, float %nc.n1316, !dbg !7
  %nc.n1318 = select i1 %.not36, float %nc.n1316, float %nc.n1311, !dbg !7
  %nc.n1319 = select i1 %.not36, float %nc.n1312, float %nc.n1306, !dbg !7
  %nc.n1320 = bitcast float %nc.n1319 to i32, !dbg !7
  %nc.n1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1320), !dbg !7
  %nc.n1322 = bitcast i32 %nc.n1321 to float, !dbg !7
  %nc.n1323 = select i1 %.not36, float %nc.n1306, float %nc.n1322, !dbg !7
  %nc.n1324 = select i1 %.not36, float %nc.n1322, float %nc.n1312, !dbg !7
  %nc.n1325 = extractelement <8 x float> %nc.n1215, i64 4, !dbg !7
  %nc.n1326 = extractelement <8 x float> %nc.n1215, i64 5, !dbg !7
  %nc.n1327 = extractelement <8 x float> %nc.n1215, i64 6, !dbg !7
  %nc.n1328 = extractelement <8 x float> %nc.n1215, i64 7, !dbg !7
  %nc.n1329 = select i1 %.not, float %nc.n1326, float %nc.n1325, !dbg !7
  %nc.n1330 = bitcast float %nc.n1329 to i32, !dbg !7
  %nc.n1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1330), !dbg !7
  %nc.n1332 = bitcast i32 %nc.n1331 to float, !dbg !7
  %nc.n1333 = select i1 %.not, float %nc.n1325, float %nc.n1332, !dbg !7
  %nc.n1334 = select i1 %.not, float %nc.n1332, float %nc.n1326, !dbg !7
  %nc.n1335 = select i1 %.not, float %nc.n1328, float %nc.n1327, !dbg !7
  %nc.n1336 = bitcast float %nc.n1335 to i32, !dbg !7
  %nc.n1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1336), !dbg !7
  %nc.n1338 = bitcast i32 %nc.n1337 to float, !dbg !7
  %nc.n1339 = select i1 %.not, float %nc.n1327, float %nc.n1338, !dbg !7
  %nc.n1340 = select i1 %.not, float %nc.n1338, float %nc.n1328, !dbg !7
  %nc.n1341 = select i1 %.not36, float %nc.n1339, float %nc.n1333, !dbg !7
  %nc.n1342 = bitcast float %nc.n1341 to i32, !dbg !7
  %nc.n1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1342), !dbg !7
  %nc.n1344 = bitcast i32 %nc.n1343 to float, !dbg !7
  %nc.n1345 = select i1 %.not36, float %nc.n1333, float %nc.n1344, !dbg !7
  %nc.n1346 = select i1 %.not36, float %nc.n1344, float %nc.n1339, !dbg !7
  %nc.n1347 = select i1 %.not36, float %nc.n1340, float %nc.n1334, !dbg !7
  %nc.n1348 = bitcast float %nc.n1347 to i32, !dbg !7
  %nc.n1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1348), !dbg !7
  %nc.n1350 = bitcast i32 %nc.n1349 to float, !dbg !7
  %nc.n1351 = select i1 %.not36, float %nc.n1334, float %nc.n1350, !dbg !7
  %nc.n1352 = select i1 %.not36, float %nc.n1350, float %nc.n1340, !dbg !7
  %nc.n1353 = extractelement <8 x float> %nc.n1244, i64 4, !dbg !7
  %nc.n1354 = extractelement <8 x float> %nc.n1244, i64 5, !dbg !7
  %nc.n1355 = extractelement <8 x float> %nc.n1244, i64 6, !dbg !7
  %nc.n1356 = extractelement <8 x float> %nc.n1244, i64 7, !dbg !7
  %nc.n1357 = select i1 %.not, float %nc.n1354, float %nc.n1353, !dbg !7
  %nc.n1358 = bitcast float %nc.n1357 to i32, !dbg !7
  %nc.n1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1358), !dbg !7
  %nc.n1360 = bitcast i32 %nc.n1359 to float, !dbg !7
  %nc.n1361 = select i1 %.not, float %nc.n1353, float %nc.n1360, !dbg !7
  %nc.n1362 = select i1 %.not, float %nc.n1360, float %nc.n1354, !dbg !7
  %nc.n1363 = select i1 %.not, float %nc.n1356, float %nc.n1355, !dbg !7
  %nc.n1364 = bitcast float %nc.n1363 to i32, !dbg !7
  %nc.n1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1364), !dbg !7
  %nc.n1366 = bitcast i32 %nc.n1365 to float, !dbg !7
  %nc.n1367 = select i1 %.not, float %nc.n1355, float %nc.n1366, !dbg !7
  %nc.n1368 = select i1 %.not, float %nc.n1366, float %nc.n1356, !dbg !7
  %nc.n1369 = select i1 %.not36, float %nc.n1367, float %nc.n1361, !dbg !7
  %nc.n1370 = bitcast float %nc.n1369 to i32, !dbg !7
  %nc.n1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1370), !dbg !7
  %nc.n1372 = bitcast i32 %nc.n1371 to float, !dbg !7
  %nc.n1373 = select i1 %.not36, float %nc.n1361, float %nc.n1372, !dbg !7
  %nc.n1374 = select i1 %.not36, float %nc.n1372, float %nc.n1367, !dbg !7
  %nc.n1375 = select i1 %.not36, float %nc.n1368, float %nc.n1362, !dbg !7
  %nc.n1376 = bitcast float %nc.n1375 to i32, !dbg !7
  %nc.n1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1376), !dbg !7
  %nc.n1378 = bitcast i32 %nc.n1377 to float, !dbg !7
  %nc.n1379 = select i1 %.not36, float %nc.n1362, float %nc.n1378, !dbg !7
  %nc.n1380 = select i1 %.not36, float %nc.n1378, float %nc.n1368, !dbg !7
  %nc.n1381 = extractelement <8 x float> %nc.n1273, i64 4, !dbg !7
  %nc.n1382 = extractelement <8 x float> %nc.n1273, i64 5, !dbg !7
  %nc.n1383 = extractelement <8 x float> %nc.n1273, i64 6, !dbg !7
  %nc.n1384 = extractelement <8 x float> %nc.n1273, i64 7, !dbg !7
  %nc.n1385 = select i1 %.not, float %nc.n1382, float %nc.n1381, !dbg !7
  %nc.n1386 = bitcast float %nc.n1385 to i32, !dbg !7
  %nc.n1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1386), !dbg !7
  %nc.n1388 = bitcast i32 %nc.n1387 to float, !dbg !7
  %nc.n1389 = select i1 %.not, float %nc.n1381, float %nc.n1388, !dbg !7
  %nc.n1390 = select i1 %.not, float %nc.n1388, float %nc.n1382, !dbg !7
  %nc.n1391 = select i1 %.not, float %nc.n1384, float %nc.n1383, !dbg !7
  %nc.n1392 = bitcast float %nc.n1391 to i32, !dbg !7
  %nc.n1393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1392), !dbg !7
  %nc.n1394 = bitcast i32 %nc.n1393 to float, !dbg !7
  %nc.n1395 = select i1 %.not, float %nc.n1383, float %nc.n1394, !dbg !7
  %nc.n1396 = select i1 %.not, float %nc.n1394, float %nc.n1384, !dbg !7
  %nc.n1397 = select i1 %.not36, float %nc.n1395, float %nc.n1389, !dbg !7
  %nc.n1398 = bitcast float %nc.n1397 to i32, !dbg !7
  %nc.n1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1398), !dbg !7
  %nc.n1400 = select i1 %.not36, float %nc.n1396, float %nc.n1390, !dbg !7
  %nc.n1401 = bitcast float %nc.n1400 to i32, !dbg !7
  %nc.n1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1401), !dbg !7
  %nc.n1403 = extractvalue [2 x <8 x float>] %nc.n132, 1, !dbg !7
  %nc.n1404 = extractelement <8 x float> %nc.n1403, i64 0, !dbg !7
  %nc.n1405 = extractelement <8 x float> %nc.n1403, i64 1, !dbg !7
  %nc.n1406 = extractelement <8 x float> %nc.n1403, i64 2, !dbg !7
  %nc.n1407 = extractelement <8 x float> %nc.n1403, i64 3, !dbg !7
  %nc.n1408 = select i1 %.not, float %nc.n1405, float %nc.n1404, !dbg !7
  %nc.n1409 = bitcast float %nc.n1408 to i32, !dbg !7
  %nc.n1410 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1409), !dbg !7
  %nc.n1411 = bitcast i32 %nc.n1410 to float, !dbg !7
  %nc.n1412 = select i1 %.not, float %nc.n1404, float %nc.n1411, !dbg !7
  %nc.n1413 = select i1 %.not, float %nc.n1411, float %nc.n1405, !dbg !7
  %nc.n1414 = select i1 %.not, float %nc.n1407, float %nc.n1406, !dbg !7
  %nc.n1415 = bitcast float %nc.n1414 to i32, !dbg !7
  %nc.n1416 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1415), !dbg !7
  %nc.n1417 = bitcast i32 %nc.n1416 to float, !dbg !7
  %nc.n1418 = select i1 %.not, float %nc.n1406, float %nc.n1417, !dbg !7
  %nc.n1419 = select i1 %.not, float %nc.n1417, float %nc.n1407, !dbg !7
  %nc.n1420 = select i1 %.not36, float %nc.n1418, float %nc.n1412, !dbg !7
  %nc.n1421 = bitcast float %nc.n1420 to i32, !dbg !7
  %nc.n1422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1421), !dbg !7
  %nc.n1423 = bitcast i32 %nc.n1422 to float, !dbg !7
  %nc.n1424 = select i1 %.not36, float %nc.n1412, float %nc.n1423, !dbg !7
  %nc.n1425 = select i1 %.not36, float %nc.n1423, float %nc.n1418, !dbg !7
  %nc.n1426 = select i1 %.not36, float %nc.n1419, float %nc.n1413, !dbg !7
  %nc.n1427 = bitcast float %nc.n1426 to i32, !dbg !7
  %nc.n1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1427), !dbg !7
  %nc.n1429 = bitcast i32 %nc.n1428 to float, !dbg !7
  %nc.n1430 = select i1 %.not36, float %nc.n1413, float %nc.n1429, !dbg !7
  %nc.n1431 = select i1 %.not36, float %nc.n1429, float %nc.n1419, !dbg !7
  %nc.n1432 = extractvalue [2 x <8 x float>] %nc.n131, 1, !dbg !7
  %nc.n1433 = extractelement <8 x float> %nc.n1432, i64 0, !dbg !7
  %nc.n1434 = extractelement <8 x float> %nc.n1432, i64 1, !dbg !7
  %nc.n1435 = extractelement <8 x float> %nc.n1432, i64 2, !dbg !7
  %nc.n1436 = extractelement <8 x float> %nc.n1432, i64 3, !dbg !7
  %nc.n1437 = select i1 %.not, float %nc.n1434, float %nc.n1433, !dbg !7
  %nc.n1438 = bitcast float %nc.n1437 to i32, !dbg !7
  %nc.n1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1438), !dbg !7
  %nc.n1440 = bitcast i32 %nc.n1439 to float, !dbg !7
  %nc.n1441 = select i1 %.not, float %nc.n1433, float %nc.n1440, !dbg !7
  %nc.n1442 = select i1 %.not, float %nc.n1440, float %nc.n1434, !dbg !7
  %nc.n1443 = select i1 %.not, float %nc.n1436, float %nc.n1435, !dbg !7
  %nc.n1444 = bitcast float %nc.n1443 to i32, !dbg !7
  %nc.n1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1444), !dbg !7
  %nc.n1446 = bitcast i32 %nc.n1445 to float, !dbg !7
  %nc.n1447 = select i1 %.not, float %nc.n1435, float %nc.n1446, !dbg !7
  %nc.n1448 = select i1 %.not, float %nc.n1446, float %nc.n1436, !dbg !7
  %nc.n1449 = select i1 %.not36, float %nc.n1447, float %nc.n1441, !dbg !7
  %nc.n1450 = bitcast float %nc.n1449 to i32, !dbg !7
  %nc.n1451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1450), !dbg !7
  %nc.n1452 = bitcast i32 %nc.n1451 to float, !dbg !7
  %nc.n1453 = select i1 %.not36, float %nc.n1441, float %nc.n1452, !dbg !7
  %nc.n1454 = select i1 %.not36, float %nc.n1452, float %nc.n1447, !dbg !7
  %nc.n1455 = select i1 %.not36, float %nc.n1448, float %nc.n1442, !dbg !7
  %nc.n1456 = bitcast float %nc.n1455 to i32, !dbg !7
  %nc.n1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1456), !dbg !7
  %nc.n1458 = bitcast i32 %nc.n1457 to float, !dbg !7
  %nc.n1459 = select i1 %.not36, float %nc.n1442, float %nc.n1458, !dbg !7
  %nc.n1460 = select i1 %.not36, float %nc.n1458, float %nc.n1448, !dbg !7
  %nc.n1461 = extractvalue [2 x <8 x float>] %nc.n130, 1, !dbg !7
  %nc.n1462 = extractelement <8 x float> %nc.n1461, i64 0, !dbg !7
  %nc.n1463 = extractelement <8 x float> %nc.n1461, i64 1, !dbg !7
  %nc.n1464 = extractelement <8 x float> %nc.n1461, i64 2, !dbg !7
  %nc.n1465 = extractelement <8 x float> %nc.n1461, i64 3, !dbg !7
  %nc.n1466 = select i1 %.not, float %nc.n1463, float %nc.n1462, !dbg !7
  %nc.n1467 = bitcast float %nc.n1466 to i32, !dbg !7
  %nc.n1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1467), !dbg !7
  %nc.n1469 = bitcast i32 %nc.n1468 to float, !dbg !7
  %nc.n1470 = select i1 %.not, float %nc.n1462, float %nc.n1469, !dbg !7
  %nc.n1471 = select i1 %.not, float %nc.n1469, float %nc.n1463, !dbg !7
  %nc.n1472 = select i1 %.not, float %nc.n1465, float %nc.n1464, !dbg !7
  %nc.n1473 = bitcast float %nc.n1472 to i32, !dbg !7
  %nc.n1474 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1473), !dbg !7
  %nc.n1475 = bitcast i32 %nc.n1474 to float, !dbg !7
  %nc.n1476 = select i1 %.not, float %nc.n1464, float %nc.n1475, !dbg !7
  %nc.n1477 = select i1 %.not, float %nc.n1475, float %nc.n1465, !dbg !7
  %nc.n1478 = select i1 %.not36, float %nc.n1476, float %nc.n1470, !dbg !7
  %nc.n1479 = bitcast float %nc.n1478 to i32, !dbg !7
  %nc.n1480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1479), !dbg !7
  %nc.n1481 = bitcast i32 %nc.n1480 to float, !dbg !7
  %nc.n1482 = select i1 %.not36, float %nc.n1470, float %nc.n1481, !dbg !7
  %nc.n1483 = select i1 %.not36, float %nc.n1481, float %nc.n1476, !dbg !7
  %nc.n1484 = select i1 %.not36, float %nc.n1477, float %nc.n1471, !dbg !7
  %nc.n1485 = bitcast float %nc.n1484 to i32, !dbg !7
  %nc.n1486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1485), !dbg !7
  %nc.n1487 = bitcast i32 %nc.n1486 to float, !dbg !7
  %nc.n1488 = select i1 %.not36, float %nc.n1471, float %nc.n1487, !dbg !7
  %nc.n1489 = select i1 %.not36, float %nc.n1487, float %nc.n1477, !dbg !7
  %nc.n1490 = extractvalue [2 x <8 x float>] %nc.n127, 1, !dbg !7
  %nc.n1491 = extractelement <8 x float> %nc.n1490, i64 0, !dbg !7
  %nc.n1492 = extractelement <8 x float> %nc.n1490, i64 1, !dbg !7
  %nc.n1493 = extractelement <8 x float> %nc.n1490, i64 2, !dbg !7
  %nc.n1494 = extractelement <8 x float> %nc.n1490, i64 3, !dbg !7
  %nc.n1495 = select i1 %.not, float %nc.n1492, float %nc.n1491, !dbg !7
  %nc.n1496 = bitcast float %nc.n1495 to i32, !dbg !7
  %nc.n1497 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1496), !dbg !7
  %nc.n1498 = bitcast i32 %nc.n1497 to float, !dbg !7
  %nc.n1499 = select i1 %.not, float %nc.n1491, float %nc.n1498, !dbg !7
  %nc.n1500 = select i1 %.not, float %nc.n1498, float %nc.n1492, !dbg !7
  %nc.n1501 = select i1 %.not, float %nc.n1494, float %nc.n1493, !dbg !7
  %nc.n1502 = bitcast float %nc.n1501 to i32, !dbg !7
  %nc.n1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1502), !dbg !7
  %nc.n1504 = bitcast i32 %nc.n1503 to float, !dbg !7
  %nc.n1505 = select i1 %.not, float %nc.n1493, float %nc.n1504, !dbg !7
  %nc.n1506 = select i1 %.not, float %nc.n1504, float %nc.n1494, !dbg !7
  %nc.n1507 = select i1 %.not36, float %nc.n1505, float %nc.n1499, !dbg !7
  %nc.n1508 = bitcast float %nc.n1507 to i32, !dbg !7
  %nc.n1509 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1508), !dbg !7
  %nc.n1510 = select i1 %.not36, float %nc.n1506, float %nc.n1500, !dbg !7
  %nc.n1511 = bitcast float %nc.n1510 to i32, !dbg !7
  %nc.n1512 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1511), !dbg !7
  %nc.n1513 = shufflevector <1 x float> %nc.n1162, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %nc.n1514 = extractelement <8 x float> %nc.n1403, i64 4, !dbg !7
  %nc.n1515 = extractelement <8 x float> %nc.n1403, i64 5, !dbg !7
  %nc.n1516 = extractelement <8 x float> %nc.n1403, i64 6, !dbg !7
  %nc.n1517 = extractelement <8 x float> %nc.n1403, i64 7, !dbg !7
  %nc.n1518 = select i1 %.not, float %nc.n1515, float %nc.n1514, !dbg !7
  %nc.n1519 = bitcast float %nc.n1518 to i32, !dbg !7
  %nc.n1520 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1519), !dbg !7
  %nc.n1521 = bitcast i32 %nc.n1520 to float, !dbg !7
  %nc.n1522 = select i1 %.not, float %nc.n1514, float %nc.n1521, !dbg !7
  %nc.n1523 = select i1 %.not, float %nc.n1521, float %nc.n1515, !dbg !7
  %nc.n1524 = select i1 %.not, float %nc.n1517, float %nc.n1516, !dbg !7
  %nc.n1525 = bitcast float %nc.n1524 to i32, !dbg !7
  %nc.n1526 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1525), !dbg !7
  %nc.n1527 = bitcast i32 %nc.n1526 to float, !dbg !7
  %nc.n1528 = select i1 %.not, float %nc.n1516, float %nc.n1527, !dbg !7
  %nc.n1529 = select i1 %.not, float %nc.n1527, float %nc.n1517, !dbg !7
  %nc.n1530 = select i1 %.not36, float %nc.n1528, float %nc.n1522, !dbg !7
  %nc.n1531 = bitcast float %nc.n1530 to i32, !dbg !7
  %nc.n1532 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1531), !dbg !7
  %nc.n1533 = bitcast i32 %nc.n1532 to float, !dbg !7
  %nc.n1534 = select i1 %.not36, float %nc.n1522, float %nc.n1533, !dbg !7
  %nc.n1535 = select i1 %.not36, float %nc.n1533, float %nc.n1528, !dbg !7
  %nc.n1536 = select i1 %.not36, float %nc.n1529, float %nc.n1523, !dbg !7
  %nc.n1537 = bitcast float %nc.n1536 to i32, !dbg !7
  %nc.n1538 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1537), !dbg !7
  %nc.n1539 = bitcast i32 %nc.n1538 to float, !dbg !7
  %nc.n1540 = select i1 %.not36, float %nc.n1523, float %nc.n1539, !dbg !7
  %nc.n1541 = select i1 %.not36, float %nc.n1539, float %nc.n1529, !dbg !7
  %nc.n1542 = extractelement <8 x float> %nc.n1432, i64 4, !dbg !7
  %nc.n1543 = extractelement <8 x float> %nc.n1432, i64 5, !dbg !7
  %nc.n1544 = extractelement <8 x float> %nc.n1432, i64 6, !dbg !7
  %nc.n1545 = extractelement <8 x float> %nc.n1432, i64 7, !dbg !7
  %nc.n1546 = select i1 %.not, float %nc.n1543, float %nc.n1542, !dbg !7
  %nc.n1547 = bitcast float %nc.n1546 to i32, !dbg !7
  %nc.n1548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1547), !dbg !7
  %nc.n1549 = bitcast i32 %nc.n1548 to float, !dbg !7
  %nc.n1550 = select i1 %.not, float %nc.n1542, float %nc.n1549, !dbg !7
  %nc.n1551 = select i1 %.not, float %nc.n1549, float %nc.n1543, !dbg !7
  %nc.n1552 = select i1 %.not, float %nc.n1545, float %nc.n1544, !dbg !7
  %nc.n1553 = bitcast float %nc.n1552 to i32, !dbg !7
  %nc.n1554 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1553), !dbg !7
  %nc.n1555 = bitcast i32 %nc.n1554 to float, !dbg !7
  %nc.n1556 = select i1 %.not, float %nc.n1544, float %nc.n1555, !dbg !7
  %nc.n1557 = select i1 %.not, float %nc.n1555, float %nc.n1545, !dbg !7
  %nc.n1558 = select i1 %.not36, float %nc.n1556, float %nc.n1550, !dbg !7
  %nc.n1559 = bitcast float %nc.n1558 to i32, !dbg !7
  %nc.n1560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1559), !dbg !7
  %nc.n1561 = bitcast i32 %nc.n1560 to float, !dbg !7
  %nc.n1562 = select i1 %.not36, float %nc.n1550, float %nc.n1561, !dbg !7
  %nc.n1563 = select i1 %.not36, float %nc.n1561, float %nc.n1556, !dbg !7
  %nc.n1564 = select i1 %.not36, float %nc.n1557, float %nc.n1551, !dbg !7
  %nc.n1565 = bitcast float %nc.n1564 to i32, !dbg !7
  %nc.n1566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1565), !dbg !7
  %nc.n1567 = bitcast i32 %nc.n1566 to float, !dbg !7
  %nc.n1568 = select i1 %.not36, float %nc.n1551, float %nc.n1567, !dbg !7
  %nc.n1569 = select i1 %.not36, float %nc.n1567, float %nc.n1557, !dbg !7
  %nc.n1570 = extractelement <8 x float> %nc.n1461, i64 4, !dbg !7
  %nc.n1571 = extractelement <8 x float> %nc.n1461, i64 5, !dbg !7
  %nc.n1572 = extractelement <8 x float> %nc.n1461, i64 6, !dbg !7
  %nc.n1573 = extractelement <8 x float> %nc.n1461, i64 7, !dbg !7
  %nc.n1574 = select i1 %.not, float %nc.n1571, float %nc.n1570, !dbg !7
  %nc.n1575 = bitcast float %nc.n1574 to i32, !dbg !7
  %nc.n1576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1575), !dbg !7
  %nc.n1577 = bitcast i32 %nc.n1576 to float, !dbg !7
  %nc.n1578 = select i1 %.not, float %nc.n1570, float %nc.n1577, !dbg !7
  %nc.n1579 = select i1 %.not, float %nc.n1577, float %nc.n1571, !dbg !7
  %nc.n1580 = select i1 %.not, float %nc.n1573, float %nc.n1572, !dbg !7
  %nc.n1581 = bitcast float %nc.n1580 to i32, !dbg !7
  %nc.n1582 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1581), !dbg !7
  %nc.n1583 = bitcast i32 %nc.n1582 to float, !dbg !7
  %nc.n1584 = select i1 %.not, float %nc.n1572, float %nc.n1583, !dbg !7
  %nc.n1585 = select i1 %.not, float %nc.n1583, float %nc.n1573, !dbg !7
  %nc.n1586 = select i1 %.not36, float %nc.n1584, float %nc.n1578, !dbg !7
  %nc.n1587 = bitcast float %nc.n1586 to i32, !dbg !7
  %nc.n1588 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1587), !dbg !7
  %nc.n1589 = bitcast i32 %nc.n1588 to float, !dbg !7
  %nc.n1590 = select i1 %.not36, float %nc.n1578, float %nc.n1589, !dbg !7
  %nc.n1591 = select i1 %.not36, float %nc.n1589, float %nc.n1584, !dbg !7
  %nc.n1592 = select i1 %.not36, float %nc.n1585, float %nc.n1579, !dbg !7
  %nc.n1593 = bitcast float %nc.n1592 to i32, !dbg !7
  %nc.n1594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1593), !dbg !7
  %nc.n1595 = bitcast i32 %nc.n1594 to float, !dbg !7
  %nc.n1596 = select i1 %.not36, float %nc.n1579, float %nc.n1595, !dbg !7
  %nc.n1597 = select i1 %.not36, float %nc.n1595, float %nc.n1585, !dbg !7
  %nc.n1598 = extractelement <8 x float> %nc.n1490, i64 4, !dbg !7
  %nc.n1599 = extractelement <8 x float> %nc.n1490, i64 5, !dbg !7
  %nc.n1600 = extractelement <8 x float> %nc.n1490, i64 6, !dbg !7
  %nc.n1601 = extractelement <8 x float> %nc.n1490, i64 7, !dbg !7
  %nc.n1602 = select i1 %.not, float %nc.n1599, float %nc.n1598, !dbg !7
  %nc.n1603 = bitcast float %nc.n1602 to i32, !dbg !7
  %nc.n1604 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1603), !dbg !7
  %nc.n1605 = bitcast i32 %nc.n1604 to float, !dbg !7
  %nc.n1606 = select i1 %.not, float %nc.n1598, float %nc.n1605, !dbg !7
  %nc.n1607 = select i1 %.not, float %nc.n1605, float %nc.n1599, !dbg !7
  %nc.n1608 = select i1 %.not, float %nc.n1601, float %nc.n1600, !dbg !7
  %nc.n1609 = bitcast float %nc.n1608 to i32, !dbg !7
  %nc.n1610 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1187, i32 %nc.n1609), !dbg !7
  %nc.n1611 = bitcast i32 %nc.n1610 to float, !dbg !7
  %nc.n1612 = select i1 %.not, float %nc.n1600, float %nc.n1611, !dbg !7
  %nc.n1613 = select i1 %.not, float %nc.n1611, float %nc.n1601, !dbg !7
  %nc.n1614 = select i1 %.not36, float %nc.n1612, float %nc.n1606, !dbg !7
  %nc.n1615 = bitcast float %nc.n1614 to i32, !dbg !7
  %nc.n1616 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1615), !dbg !7
  %nc.n1617 = select i1 %.not36, float %nc.n1613, float %nc.n1607, !dbg !7
  %nc.n1618 = bitcast float %nc.n1617 to i32, !dbg !7
  %nc.n1619 = call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1203, i32 %nc.n1618), !dbg !7
  %nc.n1620 = bitcast i32 %nc.n1616 to float, !dbg !7
  %nc.n1621 = select i1 %.not36, float %nc.n1606, float %nc.n1620, !dbg !7
  %nc.n1622 = insertelement <4 x float> poison, float %nc.n1621, i64 0, !dbg !16
  %nc.n1623 = bitcast i32 %nc.n1619 to float, !dbg !7
  %nc.n1624 = select i1 %.not36, float %nc.n1607, float %nc.n1623, !dbg !7
  %nc.n1625 = insertelement <4 x float> %nc.n1622, float %nc.n1624, i64 1, !dbg !16
  %nc.n1626 = select i1 %.not36, float %nc.n1620, float %nc.n1612, !dbg !7
  %nc.n1627 = insertelement <4 x float> %nc.n1625, float %nc.n1626, i64 2, !dbg !16
  %nc.n1628 = select i1 %.not36, float %nc.n1623, float %nc.n1613, !dbg !7
  %nc.n1629 = insertelement <4 x float> %nc.n1627, float %nc.n1628, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %nc.n1513, !dbg !16
  %normalized = fmul <4 x float> %nc.n1629, %row.reciprocal, !dbg !16
  %nc.n1630 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %nc.n1631 = extractelement <4 x half> %nc.n1630, i64 3, !dbg !17
  %nc.n1632 = extractelement <4 x half> %nc.n1630, i64 2, !dbg !17
  %nc.n1633 = extractelement <4 x half> %nc.n1630, i64 1, !dbg !17
  %nc.n1634 = extractelement <4 x half> %nc.n1630, i64 0, !dbg !17
  %nc.n1635 = bitcast i32 %nc.n1509 to float, !dbg !7
  %nc.n1636 = select i1 %.not36, float %nc.n1499, float %nc.n1635, !dbg !7
  %nc.n1637 = insertelement <4 x float> poison, float %nc.n1636, i64 0, !dbg !16
  %nc.n1638 = bitcast i32 %nc.n1512 to float, !dbg !7
  %nc.n1639 = select i1 %.not36, float %nc.n1500, float %nc.n1638, !dbg !7
  %nc.n1640 = insertelement <4 x float> %nc.n1637, float %nc.n1639, i64 1, !dbg !16
  %nc.n1641 = select i1 %.not36, float %nc.n1635, float %nc.n1505, !dbg !7
  %nc.n1642 = insertelement <4 x float> %nc.n1640, float %nc.n1641, i64 2, !dbg !16
  %nc.n1643 = select i1 %.not36, float %nc.n1638, float %nc.n1506, !dbg !7
  %nc.n1644 = insertelement <4 x float> %nc.n1642, float %nc.n1643, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %nc.n1644, %row.reciprocal, !dbg !16
  %nc.n1645 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %nc.n1646 = extractelement <4 x half> %nc.n1645, i64 3, !dbg !17
  %nc.n1647 = extractelement <4 x half> %nc.n1645, i64 2, !dbg !17
  %nc.n1648 = extractelement <4 x half> %nc.n1645, i64 1, !dbg !17
  %nc.n1649 = extractelement <4 x half> %nc.n1645, i64 0, !dbg !17
  %nc.n1650 = bitcast i32 %nc.n1399 to float, !dbg !7
  %nc.n1651 = select i1 %.not36, float %nc.n1389, float %nc.n1650, !dbg !7
  %nc.n1652 = insertelement <4 x float> poison, float %nc.n1651, i64 0, !dbg !16
  %nc.n1653 = bitcast i32 %nc.n1402 to float, !dbg !7
  %nc.n1654 = select i1 %.not36, float %nc.n1390, float %nc.n1653, !dbg !7
  %nc.n1655 = insertelement <4 x float> %nc.n1652, float %nc.n1654, i64 1, !dbg !16
  %nc.n1656 = select i1 %.not36, float %nc.n1650, float %nc.n1395, !dbg !7
  %nc.n1657 = insertelement <4 x float> %nc.n1655, float %nc.n1656, i64 2, !dbg !16
  %nc.n1658 = select i1 %.not36, float %nc.n1653, float %nc.n1396, !dbg !7
  %nc.n1659 = insertelement <4 x float> %nc.n1657, float %nc.n1658, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %nc.n1296, !dbg !16
  %normalized39 = fmul <4 x float> %nc.n1659, %row.reciprocal38, !dbg !16
  %nc.n1660 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %nc.n1661 = extractelement <4 x half> %nc.n1660, i64 3, !dbg !17
  %nc.n1662 = extractelement <4 x half> %nc.n1660, i64 2, !dbg !17
  %nc.n1663 = extractelement <4 x half> %nc.n1660, i64 1, !dbg !17
  %nc.n1664 = extractelement <4 x half> %nc.n1660, i64 0, !dbg !17
  %nc.n1665 = bitcast i32 %nc.n1292 to float, !dbg !7
  %nc.n1666 = select i1 %.not36, float %nc.n1282, float %nc.n1665, !dbg !7
  %nc.n1667 = insertelement <4 x float> poison, float %nc.n1666, i64 0, !dbg !16
  %nc.n1668 = bitcast i32 %nc.n1295 to float, !dbg !7
  %nc.n1669 = select i1 %.not36, float %nc.n1283, float %nc.n1668, !dbg !7
  %nc.n1670 = insertelement <4 x float> %nc.n1667, float %nc.n1669, i64 1, !dbg !16
  %nc.n1671 = select i1 %.not36, float %nc.n1665, float %nc.n1288, !dbg !7
  %nc.n1672 = insertelement <4 x float> %nc.n1670, float %nc.n1671, i64 2, !dbg !16
  %nc.n1673 = select i1 %.not36, float %nc.n1668, float %nc.n1289, !dbg !7
  %nc.n1674 = insertelement <4 x float> %nc.n1672, float %nc.n1673, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %nc.n1674, %row.reciprocal38, !dbg !16
  %nc.n1675 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %nc.n1676 = extractelement <4 x half> %nc.n1675, i64 3, !dbg !17
  %nc.n1677 = extractelement <4 x half> %nc.n1675, i64 2, !dbg !17
  %nc.n1678 = extractelement <4 x half> %nc.n1675, i64 1, !dbg !17
  %nc.n1679 = extractelement <4 x half> %nc.n1675, i64 0, !dbg !17
  %nc.n1680 = add nsw i64 %nc.n16, %nc.n22, !dbg !18
  %nc.n1681 = shl i64 %nc.n1680, 19, !dbg !18
  %nc.n1682 = and i64 %nc.n1681, 16252928, !dbg !18
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %address.local976 = add i64 %nc.n25, %address.local975, !dbg !18
  %nc.n1683 = lshr i64 %address.local975, 2, !dbg !18
  %address.local978 = and i64 %nc.n1683, 48, !dbg !18
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !18
  %address.local980 = and i64 %address.local975, 240, !dbg !18
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !18
  %address.local982 = add i64 %address.local981, %nc.n26, !dbg !18
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !18
  %address.local984 = add nuw nsw i64 %nc.n1682, %address.local983, !dbg !18
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local986 = lshr i64 %address.local975, 2, !dbg !10
  %address.local987 = and i64 %address.local986, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !18
  store half %nc.n1679, ptr addrspace(1) %address.local988, align 2, !dbg !18
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !18
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local989, i64 %address.local990, !dbg !18
  store half %nc.n1678, ptr addrspace(1) %address.local991, align 2, !dbg !18
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local993, !dbg !18
  store half %nc.n1677, ptr addrspace(1) %address.local994, align 2, !dbg !18
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local995, i64 %address.local996, !dbg !18
  store half %nc.n1676, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local999, !dbg !18
  store half %nc.n1664, ptr addrspace(1) %address.local1000, align 2, !dbg !18
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1001, i64 %address.local1002, !dbg !18
  store half %nc.n1663, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1005, !dbg !18
  store half %nc.n1662, ptr addrspace(1) %address.local1006, align 2, !dbg !18
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !18
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1007, i64 %address.local1008, !dbg !18
  store half %nc.n1661, ptr addrspace(1) %address.local1009, align 2, !dbg !18
  %address.local1010 = shl i64 %address.local982, 7, !dbg !18
  %address.local1011 = add i64 %address.local1010, 8192, !dbg !18
  %address.local1012 = add nuw nsw i64 %nc.n1682, %address.local1011, !dbg !18
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !18
  store half %nc.n1649, ptr addrspace(1) %address.local1014, align 2, !dbg !18
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local990, !dbg !18
  store half %nc.n1648, ptr addrspace(1) %address.local1016, align 2, !dbg !18
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local993, !dbg !18
  store half %nc.n1647, ptr addrspace(1) %address.local1018, align 2, !dbg !18
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local996, !dbg !18
  store half %nc.n1646, ptr addrspace(1) %address.local1020, align 2, !dbg !18
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local999, !dbg !18
  store half %nc.n1634, ptr addrspace(1) %address.local1022, align 2, !dbg !18
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1002, !dbg !18
  store half %nc.n1633, ptr addrspace(1) %address.local1024, align 2, !dbg !18
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1025, i64 %address.local1005, !dbg !18
  store half %nc.n1632, ptr addrspace(1) %address.local1026, align 2, !dbg !18
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1008, !dbg !18
  store half %nc.n1631, ptr addrspace(1) %address.local1028, align 2, !dbg !18
  %nc.n1684 = insertelement <4 x float> poison, float %nc.n1590, i64 0, !dbg !16
  %nc.n1685 = insertelement <4 x float> %nc.n1684, float %nc.n1596, i64 1, !dbg !16
  %nc.n1686 = insertelement <4 x float> %nc.n1685, float %nc.n1591, i64 2, !dbg !16
  %nc.n1687 = insertelement <4 x float> %nc.n1686, float %nc.n1597, i64 3, !dbg !16
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %nc.n1513, !dbg !16
  %normalized42 = fmul <4 x float> %nc.n1687, %row.reciprocal41, !dbg !16
  %nc.n1688 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %nc.n1689 = extractelement <4 x half> %nc.n1688, i64 3, !dbg !17
  %nc.n1690 = extractelement <4 x half> %nc.n1688, i64 2, !dbg !17
  %nc.n1691 = extractelement <4 x half> %nc.n1688, i64 1, !dbg !17
  %nc.n1692 = extractelement <4 x half> %nc.n1688, i64 0, !dbg !17
  %nc.n1693 = insertelement <4 x float> poison, float %nc.n1482, i64 0, !dbg !16
  %nc.n1694 = insertelement <4 x float> %nc.n1693, float %nc.n1488, i64 1, !dbg !16
  %nc.n1695 = insertelement <4 x float> %nc.n1694, float %nc.n1483, i64 2, !dbg !16
  %nc.n1696 = insertelement <4 x float> %nc.n1695, float %nc.n1489, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %nc.n1696, %row.reciprocal41, !dbg !16
  %nc.n1697 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %nc.n1698 = extractelement <4 x half> %nc.n1697, i64 3, !dbg !17
  %nc.n1699 = extractelement <4 x half> %nc.n1697, i64 2, !dbg !17
  %nc.n1700 = extractelement <4 x half> %nc.n1697, i64 1, !dbg !17
  %nc.n1701 = extractelement <4 x half> %nc.n1697, i64 0, !dbg !17
  %nc.n1702 = insertelement <4 x float> poison, float %nc.n1373, i64 0, !dbg !16
  %nc.n1703 = insertelement <4 x float> %nc.n1702, float %nc.n1379, i64 1, !dbg !16
  %nc.n1704 = insertelement <4 x float> %nc.n1703, float %nc.n1374, i64 2, !dbg !16
  %nc.n1705 = insertelement <4 x float> %nc.n1704, float %nc.n1380, i64 3, !dbg !16
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %nc.n1296, !dbg !16
  %normalized45 = fmul <4 x float> %nc.n1705, %row.reciprocal44, !dbg !16
  %nc.n1706 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %nc.n1707 = extractelement <4 x half> %nc.n1706, i64 3, !dbg !17
  %nc.n1708 = extractelement <4 x half> %nc.n1706, i64 2, !dbg !17
  %nc.n1709 = extractelement <4 x half> %nc.n1706, i64 1, !dbg !17
  %nc.n1710 = extractelement <4 x half> %nc.n1706, i64 0, !dbg !17
  %nc.n1711 = insertelement <4 x float> poison, float %nc.n1265, i64 0, !dbg !16
  %nc.n1712 = insertelement <4 x float> %nc.n1711, float %nc.n1271, i64 1, !dbg !16
  %nc.n1713 = insertelement <4 x float> %nc.n1712, float %nc.n1266, i64 2, !dbg !16
  %nc.n1714 = insertelement <4 x float> %nc.n1713, float %nc.n1272, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %nc.n1714, %row.reciprocal44, !dbg !16
  %nc.n1715 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %nc.n1716 = extractelement <4 x half> %nc.n1715, i64 3, !dbg !17
  %nc.n1717 = extractelement <4 x half> %nc.n1715, i64 2, !dbg !17
  %nc.n1718 = extractelement <4 x half> %nc.n1715, i64 1, !dbg !17
  %nc.n1719 = extractelement <4 x half> %nc.n1715, i64 0, !dbg !17
  %nc.n1720 = shl nuw nsw i64 %nc.n24, 19, !dbg !18
  %address.thread1029 = call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1030 = zext nneg i32 %address.thread1029 to i64, !dbg !8
  %address.local1031 = add i64 %nc.n25, %address.local1030, !dbg !18
  %nc.n1721 = lshr i64 %address.local1030, 2, !dbg !18
  %address.local1033 = and i64 %nc.n1721, 48, !dbg !18
  %address.local1034 = add i64 %address.local1031, %address.local1033, !dbg !18
  %address.local1035 = and i64 %address.local1030, 240, !dbg !18
  %address.local1036 = sub i64 %address.local1034, %address.local1035, !dbg !18
  %address.local1037 = add i64 %address.local1036, %nc.n26, !dbg !18
  %address.local1038 = shl nuw nsw i64 %address.local1037, 7, !dbg !18
  %address.local1039 = add nuw nsw i64 %nc.n1720, %address.local1038, !dbg !18
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1041 = lshr i64 %address.local1030, 2, !dbg !10
  %address.local1042 = and i64 %address.local1041, 12, !dbg !10
  %address.local1043 = or disjoint i64 %address.local1042, 32, !dbg !18
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1040, i64 %address.local1043, !dbg !18
  store half %nc.n1719, ptr addrspace(1) %address.local1044, align 2, !dbg !18
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1046 = or disjoint i64 %address.local1042, 33, !dbg !18
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1045, i64 %address.local1046, !dbg !18
  store half %nc.n1718, ptr addrspace(1) %address.local1047, align 2, !dbg !18
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1049 = or disjoint i64 %address.local1042, 34, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1049, !dbg !18
  store half %nc.n1717, ptr addrspace(1) %address.local1050, align 2, !dbg !18
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1052 = or disjoint i64 %address.local1042, 35, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1051, i64 %address.local1052, !dbg !18
  store half %nc.n1716, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1055 = or disjoint i64 %address.local1042, 48, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1055, !dbg !18
  store half %nc.n1710, ptr addrspace(1) %address.local1056, align 2, !dbg !18
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1058 = or disjoint i64 %address.local1042, 49, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1057, i64 %address.local1058, !dbg !18
  store half %nc.n1709, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1061 = or disjoint i64 %address.local1042, 50, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1061, !dbg !18
  store half %nc.n1708, ptr addrspace(1) %address.local1062, align 2, !dbg !18
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1064 = or i64 %address.local1041, 51, !dbg !18
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1063, i64 %address.local1064, !dbg !18
  store half %nc.n1707, ptr addrspace(1) %address.local1065, align 2, !dbg !18
  %address.local1066 = shl i64 %address.local1037, 7, !dbg !18
  %address.local1067 = add i64 %address.local1066, 8192, !dbg !18
  %address.local1068 = add nuw nsw i64 %nc.n1720, %address.local1067, !dbg !18
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1043, !dbg !18
  store half %nc.n1701, ptr addrspace(1) %address.local1070, align 2, !dbg !18
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1046, !dbg !18
  store half %nc.n1700, ptr addrspace(1) %address.local1072, align 2, !dbg !18
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1049, !dbg !18
  store half %nc.n1699, ptr addrspace(1) %address.local1074, align 2, !dbg !18
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1052, !dbg !18
  store half %nc.n1698, ptr addrspace(1) %address.local1076, align 2, !dbg !18
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1055, !dbg !18
  store half %nc.n1692, ptr addrspace(1) %address.local1078, align 2, !dbg !18
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1058, !dbg !18
  store half %nc.n1691, ptr addrspace(1) %address.local1080, align 2, !dbg !18
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !18
  store half %nc.n1690, ptr addrspace(1) %address.local1082, align 2, !dbg !18
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1083, i64 %address.local1064, !dbg !18
  store half %nc.n1689, ptr addrspace(1) %address.local1084, align 2, !dbg !18
  %nc.n1722 = insertelement <4 x float> poison, float %nc.n1562, i64 0, !dbg !16
  %nc.n1723 = insertelement <4 x float> %nc.n1722, float %nc.n1568, i64 1, !dbg !16
  %nc.n1724 = insertelement <4 x float> %nc.n1723, float %nc.n1563, i64 2, !dbg !16
  %nc.n1725 = insertelement <4 x float> %nc.n1724, float %nc.n1569, i64 3, !dbg !16
  %row.reciprocal47 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %nc.n1513, !dbg !16
  %normalized48 = fmul <4 x float> %nc.n1725, %row.reciprocal47, !dbg !16
  %nc.n1726 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %nc.n1727 = extractelement <4 x half> %nc.n1726, i64 3, !dbg !17
  %nc.n1728 = extractelement <4 x half> %nc.n1726, i64 2, !dbg !17
  %nc.n1729 = extractelement <4 x half> %nc.n1726, i64 1, !dbg !17
  %nc.n1730 = extractelement <4 x half> %nc.n1726, i64 0, !dbg !17
  %nc.n1731 = insertelement <4 x float> poison, float %nc.n1453, i64 0, !dbg !16
  %nc.n1732 = insertelement <4 x float> %nc.n1731, float %nc.n1459, i64 1, !dbg !16
  %nc.n1733 = insertelement <4 x float> %nc.n1732, float %nc.n1454, i64 2, !dbg !16
  %nc.n1734 = insertelement <4 x float> %nc.n1733, float %nc.n1460, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %nc.n1734, %row.reciprocal47, !dbg !16
  %nc.n1735 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %nc.n1736 = extractelement <4 x half> %nc.n1735, i64 3, !dbg !17
  %nc.n1737 = extractelement <4 x half> %nc.n1735, i64 2, !dbg !17
  %nc.n1738 = extractelement <4 x half> %nc.n1735, i64 1, !dbg !17
  %nc.n1739 = extractelement <4 x half> %nc.n1735, i64 0, !dbg !17
  %nc.n1740 = insertelement <4 x float> poison, float %nc.n1345, i64 0, !dbg !16
  %nc.n1741 = insertelement <4 x float> %nc.n1740, float %nc.n1351, i64 1, !dbg !16
  %nc.n1742 = insertelement <4 x float> %nc.n1741, float %nc.n1346, i64 2, !dbg !16
  %nc.n1743 = insertelement <4 x float> %nc.n1742, float %nc.n1352, i64 3, !dbg !16
  %row.reciprocal50 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %nc.n1296, !dbg !16
  %normalized51 = fmul <4 x float> %nc.n1743, %row.reciprocal50, !dbg !16
  %nc.n1744 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %nc.n1745 = extractelement <4 x half> %nc.n1744, i64 3, !dbg !17
  %nc.n1746 = extractelement <4 x half> %nc.n1744, i64 2, !dbg !17
  %nc.n1747 = extractelement <4 x half> %nc.n1744, i64 1, !dbg !17
  %nc.n1748 = extractelement <4 x half> %nc.n1744, i64 0, !dbg !17
  %nc.n1749 = insertelement <4 x float> poison, float %nc.n1236, i64 0, !dbg !16
  %nc.n1750 = insertelement <4 x float> %nc.n1749, float %nc.n1242, i64 1, !dbg !16
  %nc.n1751 = insertelement <4 x float> %nc.n1750, float %nc.n1237, i64 2, !dbg !16
  %nc.n1752 = insertelement <4 x float> %nc.n1751, float %nc.n1243, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %nc.n1752, %row.reciprocal50, !dbg !16
  %nc.n1753 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %nc.n1754 = extractelement <4 x half> %nc.n1753, i64 3, !dbg !17
  %nc.n1755 = extractelement <4 x half> %nc.n1753, i64 2, !dbg !17
  %nc.n1756 = extractelement <4 x half> %nc.n1753, i64 1, !dbg !17
  %nc.n1757 = extractelement <4 x half> %nc.n1753, i64 0, !dbg !17
  %nc.n1758 = shl nuw nsw i64 %nc.n24, 19, !dbg !18
  %address.thread1085 = call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1086 = zext nneg i32 %address.thread1085 to i64, !dbg !8
  %address.local1087 = add i64 %nc.n25, %address.local1086, !dbg !18
  %nc.n1759 = lshr i64 %address.local1086, 2, !dbg !18
  %address.local1089 = and i64 %nc.n1759, 48, !dbg !18
  %address.local1090 = add i64 %address.local1087, %address.local1089, !dbg !18
  %address.local1091 = and i64 %address.local1086, 240, !dbg !18
  %address.local1092 = sub i64 %address.local1090, %address.local1091, !dbg !18
  %address.local1093 = add i64 %address.local1092, %nc.n26, !dbg !18
  %address.local1094 = shl nuw nsw i64 %address.local1093, 7, !dbg !18
  %address.local1095 = add nuw nsw i64 %nc.n1758, %address.local1094, !dbg !18
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1097 = lshr i64 %address.local1086, 2, !dbg !10
  %address.local1098 = and i64 %address.local1097, 12, !dbg !10
  %address.local1099 = or disjoint i64 %address.local1098, 64, !dbg !18
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1096, i64 %address.local1099, !dbg !18
  store half %nc.n1757, ptr addrspace(1) %address.local1100, align 2, !dbg !18
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1102 = or disjoint i64 %address.local1098, 65, !dbg !18
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !18
  store half %nc.n1756, ptr addrspace(1) %address.local1103, align 2, !dbg !18
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1105 = or disjoint i64 %address.local1098, 66, !dbg !18
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !18
  store half %nc.n1755, ptr addrspace(1) %address.local1106, align 2, !dbg !18
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1108 = or disjoint i64 %address.local1098, 67, !dbg !18
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !18
  store half %nc.n1754, ptr addrspace(1) %address.local1109, align 2, !dbg !18
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1111 = or disjoint i64 %address.local1098, 80, !dbg !18
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1110, i64 %address.local1111, !dbg !18
  store half %nc.n1748, ptr addrspace(1) %address.local1112, align 2, !dbg !18
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1114 = or disjoint i64 %address.local1098, 81, !dbg !18
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1114, !dbg !18
  store half %nc.n1747, ptr addrspace(1) %address.local1115, align 2, !dbg !18
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1117 = or disjoint i64 %address.local1098, 82, !dbg !18
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1117, !dbg !18
  store half %nc.n1746, ptr addrspace(1) %address.local1118, align 2, !dbg !18
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1120 = or disjoint i64 %address.local1098, 83, !dbg !18
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1120, !dbg !18
  store half %nc.n1745, ptr addrspace(1) %address.local1121, align 2, !dbg !18
  %address.local1122 = shl i64 %address.local1093, 7, !dbg !18
  %address.local1123 = add i64 %address.local1122, 8192, !dbg !18
  %address.local1124 = add nuw nsw i64 %nc.n1758, %address.local1123, !dbg !18
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1099, !dbg !18
  store half %nc.n1739, ptr addrspace(1) %address.local1126, align 2, !dbg !18
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1102, !dbg !18
  store half %nc.n1738, ptr addrspace(1) %address.local1128, align 2, !dbg !18
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1105, !dbg !18
  store half %nc.n1737, ptr addrspace(1) %address.local1130, align 2, !dbg !18
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1108, !dbg !18
  store half %nc.n1736, ptr addrspace(1) %address.local1132, align 2, !dbg !18
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1111, !dbg !18
  store half %nc.n1730, ptr addrspace(1) %address.local1134, align 2, !dbg !18
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1114, !dbg !18
  store half %nc.n1729, ptr addrspace(1) %address.local1136, align 2, !dbg !18
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1117, !dbg !18
  store half %nc.n1728, ptr addrspace(1) %address.local1138, align 2, !dbg !18
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1120, !dbg !18
  store half %nc.n1727, ptr addrspace(1) %address.local1140, align 2, !dbg !18
  %nc.n1760 = insertelement <4 x float> poison, float %nc.n1534, i64 0, !dbg !16
  %nc.n1761 = insertelement <4 x float> %nc.n1760, float %nc.n1540, i64 1, !dbg !16
  %nc.n1762 = insertelement <4 x float> %nc.n1761, float %nc.n1535, i64 2, !dbg !16
  %nc.n1763 = insertelement <4 x float> %nc.n1762, float %nc.n1541, i64 3, !dbg !16
  %row.reciprocal53 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %nc.n1513, !dbg !16
  %normalized54 = fmul <4 x float> %nc.n1763, %row.reciprocal53, !dbg !16
  %nc.n1764 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %nc.n1765 = extractelement <4 x half> %nc.n1764, i64 3, !dbg !17
  %nc.n1766 = extractelement <4 x half> %nc.n1764, i64 2, !dbg !17
  %nc.n1767 = extractelement <4 x half> %nc.n1764, i64 1, !dbg !17
  %nc.n1768 = extractelement <4 x half> %nc.n1764, i64 0, !dbg !17
  %nc.n1769 = insertelement <4 x float> poison, float %nc.n1424, i64 0, !dbg !16
  %nc.n1770 = insertelement <4 x float> %nc.n1769, float %nc.n1430, i64 1, !dbg !16
  %nc.n1771 = insertelement <4 x float> %nc.n1770, float %nc.n1425, i64 2, !dbg !16
  %nc.n1772 = insertelement <4 x float> %nc.n1771, float %nc.n1431, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %nc.n1772, %row.reciprocal53, !dbg !16
  %nc.n1773 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %nc.n1774 = extractelement <4 x half> %nc.n1773, i64 3, !dbg !17
  %nc.n1775 = extractelement <4 x half> %nc.n1773, i64 2, !dbg !17
  %nc.n1776 = extractelement <4 x half> %nc.n1773, i64 1, !dbg !17
  %nc.n1777 = extractelement <4 x half> %nc.n1773, i64 0, !dbg !17
  %nc.n1778 = insertelement <4 x float> poison, float %nc.n1317, i64 0, !dbg !16
  %nc.n1779 = insertelement <4 x float> %nc.n1778, float %nc.n1323, i64 1, !dbg !16
  %nc.n1780 = insertelement <4 x float> %nc.n1779, float %nc.n1318, i64 2, !dbg !16
  %nc.n1781 = insertelement <4 x float> %nc.n1780, float %nc.n1324, i64 3, !dbg !16
  %row.reciprocal56 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %nc.n1296, !dbg !16
  %normalized57 = fmul <4 x float> %nc.n1781, %row.reciprocal56, !dbg !16
  %nc.n1782 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %nc.n1783 = extractelement <4 x half> %nc.n1782, i64 3, !dbg !17
  %nc.n1784 = extractelement <4 x half> %nc.n1782, i64 2, !dbg !17
  %nc.n1785 = extractelement <4 x half> %nc.n1782, i64 1, !dbg !17
  %nc.n1786 = extractelement <4 x half> %nc.n1782, i64 0, !dbg !17
  %nc.n1787 = insertelement <4 x float> poison, float %nc.n1207, i64 0, !dbg !16
  %nc.n1788 = insertelement <4 x float> %nc.n1787, float %nc.n1213, i64 1, !dbg !16
  %nc.n1789 = insertelement <4 x float> %nc.n1788, float %nc.n1208, i64 2, !dbg !16
  %nc.n1790 = insertelement <4 x float> %nc.n1789, float %nc.n1214, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %nc.n1790, %row.reciprocal56, !dbg !16
  %nc.n1791 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %nc.n1792 = extractelement <4 x half> %nc.n1791, i64 3, !dbg !17
  %nc.n1793 = extractelement <4 x half> %nc.n1791, i64 2, !dbg !17
  %nc.n1794 = extractelement <4 x half> %nc.n1791, i64 1, !dbg !17
  %nc.n1795 = extractelement <4 x half> %nc.n1791, i64 0, !dbg !17
  %nc.n1796 = shl nuw nsw i64 %nc.n24, 19, !dbg !18
  %address.thread1141 = call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1142 = zext nneg i32 %address.thread1141 to i64, !dbg !8
  %address.local1143 = add i64 %nc.n25, %address.local1142, !dbg !18
  %nc.n1797 = lshr i64 %address.local1142, 2, !dbg !18
  %address.local1145 = and i64 %nc.n1797, 48, !dbg !18
  %address.local1146 = add i64 %address.local1143, %address.local1145, !dbg !18
  %address.local1147 = and i64 %address.local1142, 240, !dbg !18
  %address.local1148 = sub i64 %address.local1146, %address.local1147, !dbg !18
  %address.local1149 = add i64 %address.local1148, %nc.n26, !dbg !18
  %address.local1150 = shl nuw nsw i64 %address.local1149, 7, !dbg !18
  %address.local1151 = add nuw nsw i64 %nc.n1796, %address.local1150, !dbg !18
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1153 = lshr i64 %address.local1142, 2, !dbg !10
  %address.local1154 = and i64 %address.local1153, 12, !dbg !10
  %address.local1155 = or disjoint i64 %address.local1154, 96, !dbg !18
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1152, i64 %address.local1155, !dbg !18
  store half %nc.n1795, ptr addrspace(1) %address.local1156, align 2, !dbg !18
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1158 = or disjoint i64 %address.local1154, 97, !dbg !18
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1157, i64 %address.local1158, !dbg !18
  store half %nc.n1794, ptr addrspace(1) %address.local1159, align 2, !dbg !18
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1161 = or disjoint i64 %address.local1154, 98, !dbg !18
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1160, i64 %address.local1161, !dbg !18
  store half %nc.n1793, ptr addrspace(1) %address.local1162, align 2, !dbg !18
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1164 = or disjoint i64 %address.local1154, 99, !dbg !18
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1163, i64 %address.local1164, !dbg !18
  store half %nc.n1792, ptr addrspace(1) %address.local1165, align 2, !dbg !18
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1167 = or disjoint i64 %address.local1154, 112, !dbg !18
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1166, i64 %address.local1167, !dbg !18
  store half %nc.n1786, ptr addrspace(1) %address.local1168, align 2, !dbg !18
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1170 = or disjoint i64 %address.local1154, 113, !dbg !18
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1169, i64 %address.local1170, !dbg !18
  store half %nc.n1785, ptr addrspace(1) %address.local1171, align 2, !dbg !18
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1173 = or disjoint i64 %address.local1154, 114, !dbg !18
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1172, i64 %address.local1173, !dbg !18
  store half %nc.n1784, ptr addrspace(1) %address.local1174, align 2, !dbg !18
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1176 = or i64 %address.local1153, 115, !dbg !18
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1176, !dbg !18
  store half %nc.n1783, ptr addrspace(1) %address.local1177, align 2, !dbg !18
  %address.local1178 = shl i64 %address.local1149, 7, !dbg !18
  %address.local1179 = add i64 %address.local1178, 8192, !dbg !18
  %address.local1180 = add nuw nsw i64 %nc.n1796, %address.local1179, !dbg !18
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1155, !dbg !18
  store half %nc.n1777, ptr addrspace(1) %address.local1182, align 2, !dbg !18
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1158, !dbg !18
  store half %nc.n1776, ptr addrspace(1) %address.local1184, align 2, !dbg !18
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1161, !dbg !18
  store half %nc.n1775, ptr addrspace(1) %address.local1186, align 2, !dbg !18
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1187, i64 %address.local1164, !dbg !18
  store half %nc.n1774, ptr addrspace(1) %address.local1188, align 2, !dbg !18
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1167, !dbg !18
  store half %nc.n1768, ptr addrspace(1) %address.local1190, align 2, !dbg !18
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1170, !dbg !18
  store half %nc.n1767, ptr addrspace(1) %address.local1192, align 2, !dbg !18
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1193, i64 %address.local1173, !dbg !18
  store half %nc.n1766, ptr addrspace(1) %address.local1194, align 2, !dbg !18
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1176, !dbg !18
  store half %nc.n1765, ptr addrspace(1) %address.local1196, align 2, !dbg !18
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
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!31}

!0 = !DIFile(filename: "test_input_spidD_noncasual.mlir", directory: "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 57, column: 24, scope: !4)
!8 = !DILocation(line: 67, column: 22, scope: !4)
!10 = !DILocation(line: 69, column: 24, scope: !4)
!12 = !DILocation(line: 68, column: 22, scope: !4)
!13 = !DILocation(line: 87, column: 23, scope: !4)
!14 = !DILocation(line: 70, column: 26, scope: !4)
!15 = !DILocation(line: 86, column: 20, scope: !4)
!16 = !DILocation(line: 92, column: 23, scope: !4)
!17 = !DILocation(line: 93, column: 19, scope: !4)
!18 = !DILocation(line: 94, column: 9, scope: !4)
!19 = !DILocation(line: 71, column: 23, scope: !4)
!20 = !DILocation(line: 74, column: 20, scope: !4)
!21 = !DILocation(line: 75, column: 22, scope: !4)
!22 = !DILocation(line: 76, column: 23, scope: !4)
!23 = !DILocation(line: 77, column: 20, scope: !4)
!24 = !DILocation(line: 78, column: 22, scope: !4)
!25 = !DILocation(line: 79, column: 22, scope: !4)
!26 = !DILocation(line: 80, column: 20, scope: !4)
!27 = !DILocation(line: 81, column: 16, scope: !4)
!28 = !DILocation(line: 88, column: 22, scope: !4)
!30 = !{i32 256, i32 1, i32 1}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{i32 0, i32 256}
