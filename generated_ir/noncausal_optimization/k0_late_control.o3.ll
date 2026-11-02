; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/k0_late_control.ll'
source_filename = "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD_noncasual.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %nc.n5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %nc.n6 = sext i32 %nc.n5 to i64, !dbg !7
  %nc.n7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %nc.n8 = sext i32 %nc.n7 to i64, !dbg !7
  %nc.n9 = shl nsw i64 %nc.n8, 5, !dbg !7
  %nc.n10 = add nsw i64 %nc.n9, %nc.n6, !dbg !7
  %nc.n11 = lshr i64 %nc.n10, 5, !dbg !7
  %nc.n12 = and i64 %nc.n11, 576460752303423484, !dbg !7
  %nc.n13 = and i64 %nc.n6, 3, !dbg !7
  %nc.n14 = or disjoint i64 %nc.n12, %nc.n13, !dbg !7
  %nc.n15 = sub nsw i64 31, %nc.n14, !dbg !7
  %nc.n16 = lshr i64 %nc.n10, 2, !dbg !7
  %nc.n17 = icmp ugt i64 %nc.n10, 1023, !dbg !8
  %nc.n18 = add nsw i64 %nc.n14, -32, !dbg !8
  %nc.n19 = select i1 %nc.n17, i64 %nc.n18, i64 %nc.n15, !dbg !8
  %nc.n20 = sdiv i64 %nc.n19, 32, !dbg !8
  %nc.n21 = sext i1 %nc.n17 to i64, !dbg !8
  %nc.n22 = xor i64 %nc.n20, %nc.n21, !dbg !8
  %nc.n23 = add nsw i64 %nc.n22, %nc.n16, !dbg !8
  %nc.n25 = shl nsw i64 %nc.n15, 7, !dbg !8
  %5 = shl i64 %nc.n22, 12, !dbg !8
  %nc.n27 = sub i64 %nc.n25, %5, !dbg !8
  %nc.n24 = shl i64 %nc.n23, 19, !dbg !8
  %nc.n28 = and i64 %nc.n24, 16252928, !dbg !8
  %nc.n29 = shl nsw i64 %nc.n27, 7, !dbg !8
  %nc.n30 = add i64 %nc.n28, %nc.n29, !dbg !8
  %nc.n31 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %nc.n34 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %address.local413, %nc.n30, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %nc.n35 = insertelement <2 x i64> poison, i64 %nc.n34, i64 0, !dbg !8
  %nc.n36 = bitcast <2 x i64> %nc.n35 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %nc.n36, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local424, %address.local410, !dbg !8
  %.idx1400 = or disjoint i64 %address.local425, %nc.n30, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %nc.n45 = or disjoint i64 %nc.n30, 32, !dbg !8
  %nc.n46 = or disjoint i64 %address.local413, %nc.n45, !dbg !8
  %.tr = trunc i64 %nc.n46 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %6 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local406, i64 0, !dbg !8
  %7 = insertelement <4 x ptr addrspace(3)> %6, ptr addrspace(3) %address.local418, i64 1, !dbg !8
  %8 = insertelement <4 x ptr addrspace(3)> %7, ptr addrspace(3) %address.local427, i64 2, !dbg !8
  %9 = insertelement <4 x ptr addrspace(3)> %8, ptr addrspace(3) %address.local431, i64 3, !dbg !8
  %10 = ptrtoint <4 x ptr addrspace(3)> %9 to <4 x i64>, !dbg !8
  %11 = trunc <4 x i64> %10 to <4 x i32>, !dbg !8
  %12 = extractelement <4 x i32> %11, i64 0, !dbg !8
  %nc.n33 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %12), !dbg !8
  %dma.buffer.lds1398 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n33), !dbg !8
  %nc.n37 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %13 = extractelement <4 x i32> %11, i64 1, !dbg !8
  %nc.n38 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %13), !dbg !8
  %dma.buffer.lds1395 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n38), !dbg !8
  %nc.n42 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %14 = extractelement <4 x i32> %11, i64 2, !dbg !8
  %nc.n43 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %14), !dbg !8
  %dma.buffer.lds1392 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n43), !dbg !8
  %nc.n49 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %15 = extractelement <4 x i32> %11, i64 3, !dbg !8
  %nc.n50 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %15), !dbg !8
  %nc.n53 = or disjoint i64 %address.local425, %nc.n45, !dbg !8
  %.tr1401 = trunc i64 %nc.n53 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %dma.buffer.lds1389 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n50), !dbg !8
  %nc.n56 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
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
  %nc.n59 = fmul <4 x half> %nc.n57, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %nc.n60 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %nc.n61 = fmul <4 x half> %nc.n58, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %nc.n62 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %nc.n63 = fmul <4 x half> %nc.n60, splat (half 1.275630e-01), !dbg !10
  %nc.n64 = fmul <4 x half> %nc.n62, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %nc.n67 = or disjoint i64 %nc.n30, 64, !dbg !8
  %nc.n68 = or disjoint i64 %address.local413, %nc.n67, !dbg !8
  %.tr1402 = trunc i64 %nc.n68 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %nc.n71 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %nc.n75 = or disjoint i64 %address.local425, %nc.n67, !dbg !8
  %.tr1403 = trunc i64 %nc.n75 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %nc.n78 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %nc.n79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %nc.n80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %nc.n81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %nc.n82 = fmul <4 x half> %nc.n79, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %nc.n83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %nc.n84 = fmul <4 x half> %nc.n80, splat (half 1.275630e-01), !dbg !10
  %nc.n85 = fmul <4 x half> %nc.n81, splat (half 1.275630e-01), !dbg !10
  %nc.n86 = fmul <4 x half> %nc.n83, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %nc.n89 = or disjoint i64 %nc.n30, 96, !dbg !8
  %nc.n90 = or disjoint i64 %address.local413, %nc.n89, !dbg !8
  %.tr1404 = trunc i64 %nc.n90 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %nc.n93 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %nc.n97 = or disjoint i64 %address.local425, %nc.n89, !dbg !8
  %.tr1405 = trunc i64 %nc.n97 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %nc.n100 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %nc.n101 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %nc.n102 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %nc.n103 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %nc.n104 = fmul <4 x half> %nc.n101, splat (half 1.275630e-01), !dbg !10
  %nc.n105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %nc.n106 = fmul <4 x half> %nc.n102, splat (half 1.275630e-01), !dbg !10
  %nc.n107 = fmul <4 x half> %nc.n103, splat (half 1.275630e-01), !dbg !10
  %nc.n108 = fmul <4 x half> %nc.n105, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %nc.n109 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %nc.n110 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %nc.n111 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %nc.n112 = fmul <4 x half> %nc.n109, splat (half 1.275630e-01), !dbg !10
  %nc.n113 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %nc.n114 = fmul <4 x half> %nc.n110, splat (half 1.275630e-01), !dbg !10
  %nc.n115 = fmul <4 x half> %nc.n111, splat (half 1.275630e-01), !dbg !10
  %nc.n116 = fmul <4 x half> %nc.n113, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %16 = shl nuw nsw i32 %nc.n31, 2, !dbg !12
  %17 = and i32 %16, 12, !dbg !12
  %nc.n121 = and i32 %nc.n31, 1, !dbg !12
  %nc.n12210 = add nuw nsw i32 %nc.n121, 1, !dbg !12
  %narrow = or disjoint i32 %nc.n12210, %17, !dbg !12
  %18 = lshr i32 %narrow, 1, !dbg !12
  %nc.n124 = zext nneg i32 %18 to i64, !dbg !12
  %k0.prime.nc.n135 = shl i64 %nc.n16, 19, !dbg !13
  %k0.prime.nc.n136 = and i64 %k0.prime.nc.n135, 16252928, !dbg !13
  %k0.prime.address.local507 = and i64 %address.local, 123, !dbg !13
  %k0.prime.address.local508 = lshr i64 %address.local, 2, !dbg !13
  %k0.prime.address.local509 = and i64 %k0.prime.address.local508, 4, !dbg !13
  %k0.prime.address.local510 = add nuw nsw i64 %k0.prime.address.local509, %address.local, !dbg !13
  %k0.prime.address.local511 = and i64 %k0.prime.address.local510, 4, !dbg !13
  %k0.prime.address.local512 = or disjoint i64 %k0.prime.address.local511, %k0.prime.address.local507, !dbg !13
  %k0.prime.address.local513 = shl nuw nsw i64 %k0.prime.address.local512, 7, !dbg !13
  %k0.prime.address.local514 = or disjoint i64 %k0.prime.address.local513, %k0.prime.nc.n136, !dbg !13
  %k0.prime..idx1406 = or disjoint i64 %k0.prime.address.local514, %address.local412, !dbg !13
  %k0.prime..idx1406.tr = trunc nuw nsw i64 %k0.prime..idx1406 to i32, !dbg !13
  %k0.prime.dma.buffer.offset1372 = shl nuw nsw i32 %k0.prime..idx1406.tr, 1, !dbg !13
  %k0.prime.nc.n140 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.nc.n141 = insertelement <2 x i64> poison, i64 %k0.prime.nc.n140, i64 0, !dbg !13
  %k0.prime.nc.n142 = bitcast <2 x i64> %k0.prime.nc.n141 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1373 = shufflevector <4 x i32> %k0.prime.nc.n142, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.nc.n143 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %k0.prime.dma.buffer.offset1372, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %k0.prime..idx1407 = or disjoint i64 %k0.prime.address.local514, %address.local424, !dbg !13
  %k0.prime..idx1407.tr = trunc nuw nsw i64 %k0.prime..idx1407 to i32, !dbg !13
  %k0.prime.dma.buffer.offset1369 = shl nuw nsw i32 %k0.prime..idx1407.tr, 1, !dbg !13
  %k0.prime.nc.n148 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %k0.prime.dma.buffer.offset1369, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %nc.n383 = ptrtoint ptr addrspace(1) %2 to i64
  %nc.n384 = insertelement <2 x i64> poison, i64 %nc.n383, i64 0
  %nc.n385 = bitcast <2 x i64> %nc.n384 to <4 x i32>
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %nc.n385, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %nc.n134, !dbg !11

nc.n134:                                          ; preds = %k0.join, %4
  %nc.n13217 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %nc.n1151, %k0.join ]
  %nc.n13116 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %nc.n1147, %k0.join ]
  %nc.n13015 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %nc.n1143, %k0.join ]
  %nc.n12914 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %nc.n1139, %k0.join ]
  %nc.n12813 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %nc.n807, %k0.join ]
  %nc.n12712 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %nc.n1137, %k0.join ]
  %nc.n12611 = phi i64 [ 0, %4 ], [ %nc.n1152, %k0.join ]
  %nc.n137 = shl nuw nsw i64 %nc.n12611, 14, !dbg !13
  %nc.n138 = add nuw nsw i64 %nc.n137, %k0.prime.nc.n136, !dbg !13
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
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
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local522 = lshr i64 %address.local516, 4, !dbg !13
  %address.local523 = and i64 %address.local522, 24, !dbg !13
  %address.local531 = or disjoint i64 %address.local513, %address.local523, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %nc.n149 = or disjoint i64 %nc.n138, 32, !dbg !13
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %.idx1408 = or disjoint i64 %address.local514, %nc.n149, !dbg !13
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
  %nc.n139 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %25), !dbg !13
  %26 = extractelement <4 x i32> %24, i64 1, !dbg !13
  %nc.n144 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %26), !dbg !13
  %27 = extractelement <4 x i32> %24, i64 2, !dbg !13
  %nc.n150 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %27), !dbg !13
  %dma.buffer.lds1368 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n150), !dbg !13
  %nc.n154 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1366, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %28 = extractelement <4 x i32> %24, i64 3, !dbg !13
  %nc.n155 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %28), !dbg !13
  %.idx1409 = or disjoint i64 %address.local531, %nc.n149, !dbg !13
  %.idx1409.tr = trunc nuw nsw i64 %.idx1409 to i32, !dbg !13
  %dma.buffer.offset1363 = shl nuw nsw i32 %.idx1409.tr, 1, !dbg !13
  %dma.buffer.lds1365 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n155), !dbg !13
  %nc.n159 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1363, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
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
  %nc.n160 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !12
  %.idx1410 = mul nsw i64 %address.local508, -1280, !dbg !12
  %nc.n161 = getelementptr i8, ptr addrspace(3) %nc.n160, i64 %.idx1410, !dbg !12
  %nc.n162 = getelementptr [2 x i8], ptr addrspace(3) %nc.n161, i64 %address.local546, !dbg !12
  %nc.n163 = getelementptr [2 x i8], ptr addrspace(3) %nc.n162, i64 %address.local550, !dbg !12
  %nc.n164 = getelementptr [2 x i8], ptr addrspace(3) %nc.n163, i64 %address.local553, !dbg !12
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %nc.n164, i64 %address.local557, !dbg !12
  %nc.n165 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !12
  %nc.n166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n165, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1210 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !12
  %nc.n167 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !12
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !12
  %address.local571 = and i32 %address.local570, 24, !dbg !12
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !12
  %address.local573 = and i32 %address.local572, 768, !dbg !12
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !12
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !12
  %address.local576 = and i32 %address.local575, 96, !dbg !12
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !12
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !12
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %nc.n124) #8, !dbg !12, !frisk.register_lifetimes !5
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !12
  %address.local580 = and i32 %address.local579, 1, !dbg !12
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !12
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !12
  %address.local583 = shl i64 %address.local582, 6, !dbg !12
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !12
  %nc.n168 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !12
  %nc.n169 = getelementptr [2 x i8], ptr addrspace(3) %nc.n168, i64 %address.local553, !dbg !12
  %nc.n170 = getelementptr [2 x i8], ptr addrspace(3) %nc.n169, i64 %address.local557, !dbg !12
  %address.local588 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 64, !dbg !12
  %nc.n171 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !12
  %nc.n172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n167, <4 x float> %nc.n166) #8, !dbg !12
  %nc.n173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n171, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1211 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 4160, !dbg !12
  %nc.n174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !12
  %lds.const.gep1212 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !12
  %nc.n175 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !12
  %nc.n176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n174, <4 x float> %nc.n173) #8, !dbg !12
  %nc.n177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n175, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1213 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !12
  %nc.n178 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !12
  %lds.const.gep1214 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 192, !dbg !12
  %nc.n179 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !12
  %nc.n180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n178, <4 x float> %nc.n177) #8, !dbg !12
  %nc.n181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n179, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 4288, !dbg !12
  %nc.n182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !12
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !12
  %nc.n183 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %nc.n184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n182, <4 x float> %nc.n181) #8, !dbg !12
  %nc.n185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n183, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !12
  %nc.n186 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 1088, !dbg !12
  %nc.n187 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %nc.n188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n186, <4 x float> %nc.n185) #8, !dbg !12
  %nc.n189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n187, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 5184, !dbg !12
  %nc.n190 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !12
  %nc.n191 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  %nc.n192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n190, <4 x float> %nc.n189) #8, !dbg !12
  %nc.n193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n191, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !12
  %nc.n194 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !12
  %lds.const.gep1222 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 1216, !dbg !12
  %nc.n195 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !12
  %nc.n196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n194, <4 x float> %nc.n193) #8, !dbg !12
  %nc.n197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n59, <4 x half> %nc.n195, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1223 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 5312, !dbg !12
  %nc.n198 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !12
  %nc.n200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n61, <4 x half> %nc.n198, <4 x float> %nc.n197) #8, !dbg !12
  %nc.n201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n165, <4 x float> zeroinitializer) #8, !dbg !12
  %nc.n204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n167, <4 x float> %nc.n201) #8, !dbg !12
  %nc.n205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n171, <4 x float> zeroinitializer) #8, !dbg !12
  %nc.n208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n174, <4 x float> %nc.n205) #8, !dbg !12
  %nc.n209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n175, <4 x float> zeroinitializer) #8, !dbg !12
  %nc.n212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n178, <4 x float> %nc.n209) #8, !dbg !12
  %nc.n213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n179, <4 x float> zeroinitializer) #8, !dbg !12
  %nc.n216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n182, <4 x float> %nc.n213) #8, !dbg !12
  %nc.n217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n183, <4 x float> zeroinitializer) #8, !dbg !12
  %nc.n220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n186, <4 x float> %nc.n217) #8, !dbg !12
  %nc.n221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n187, <4 x float> zeroinitializer) #8, !dbg !12
  %nc.n224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n190, <4 x float> %nc.n221) #8, !dbg !12
  %nc.n225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n191, <4 x float> zeroinitializer) #8, !dbg !12
  %nc.n228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n194, <4 x float> %nc.n225) #8, !dbg !12
  %nc.n229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n63, <4 x half> %nc.n195, <4 x float> zeroinitializer) #8, !dbg !12
  %nc.n231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n64, <4 x half> %nc.n198, <4 x float> %nc.n229) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %nc.n232 = or disjoint i64 %nc.n138, 64, !dbg !13
  %.idx14111412 = or disjoint i64 %address.local514, %nc.n232, !dbg !13
  %.idx14111412.tr = trunc nuw nsw i64 %.idx14111412 to i32, !dbg !13
  %dma.buffer.offset1360 = shl nuw nsw i32 %.idx14111412.tr, 1, !dbg !13
  %dma.buffer.lds1362 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n139), !dbg !13
  %nc.n237 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1362, i32 %dma.buffer.offset1360, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %.idx14131414 = or disjoint i64 %address.local531, %nc.n232, !dbg !13
  %.idx14131414.tr = trunc nuw nsw i64 %.idx14131414 to i32, !dbg !13
  %dma.buffer.offset1357 = shl nuw nsw i32 %.idx14131414.tr, 1, !dbg !13
  %dma.buffer.lds1359 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %nc.n144), !dbg !13
  %nc.n242 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1359, i32 %dma.buffer.offset1357, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %lds.const.gep1238 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !12
  %nc.n243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !12
  %nc.n244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n243, <4 x float> %nc.n172) #8, !dbg !12
  %lds.const.gep1239 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !12
  %nc.n245 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !12
  %lds.const.gep1240 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 8256, !dbg !12
  %nc.n246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !12
  %nc.n247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n245, <4 x float> %nc.n244) #8, !dbg !12
  %nc.n248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n246, <4 x float> %nc.n176) #8, !dbg !12
  %lds.const.gep1241 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 12352, !dbg !12
  %nc.n249 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !12
  %lds.const.gep1242 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !12
  %nc.n250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !12
  %nc.n251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n249, <4 x float> %nc.n248) #8, !dbg !12
  %nc.n252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n250, <4 x float> %nc.n180) #8, !dbg !12
  %lds.const.gep1243 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !12
  %nc.n253 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !12
  %lds.const.gep1244 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 8384, !dbg !12
  %nc.n254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !12
  %nc.n255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n253, <4 x float> %nc.n252) #8, !dbg !12
  %nc.n256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n254, <4 x float> %nc.n184) #8, !dbg !12
  %lds.const.gep1245 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 12480, !dbg !12
  %nc.n257 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !12
  %lds.const.gep1246 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !12
  %nc.n258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !12
  %nc.n259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n257, <4 x float> %nc.n256) #8, !dbg !12
  %nc.n260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n258, <4 x float> %nc.n188) #8, !dbg !12
  %lds.const.gep1247 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !12
  %nc.n261 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !12
  %lds.const.gep1248 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 9280, !dbg !12
  %nc.n262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !12
  %nc.n263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n261, <4 x float> %nc.n260) #8, !dbg !12
  %nc.n264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n262, <4 x float> %nc.n192) #8, !dbg !12
  %lds.const.gep1249 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 13376, !dbg !12
  %nc.n265 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !12
  %lds.const.gep1250 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !12
  %nc.n266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !12
  %nc.n267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n265, <4 x float> %nc.n264) #8, !dbg !12
  %nc.n268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n266, <4 x float> %nc.n196) #8, !dbg !12
  %lds.const.gep1251 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !12
  %nc.n269 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !12
  %lds.const.gep1252 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 9408, !dbg !12
  %nc.n270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !12
  %nc.n271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n269, <4 x float> %nc.n268) #8, !dbg !12
  %nc.n272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n82, <4 x half> %nc.n270, <4 x float> %nc.n200) #8, !dbg !12
  %lds.const.gep1253 = getelementptr i8, ptr addrspace(3) %nc.n170, i64 13504, !dbg !12
  %nc.n273 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !12
  %nc.n275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n84, <4 x half> %nc.n273, <4 x float> %nc.n272) #8, !dbg !12
  %nc.n276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n243, <4 x float> %nc.n204) #8, !dbg !12
  %nc.n279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n245, <4 x float> %nc.n276) #8, !dbg !12
  %nc.n280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n246, <4 x float> %nc.n208) #8, !dbg !12
  %nc.n283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n249, <4 x float> %nc.n280) #8, !dbg !12
  %nc.n284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n250, <4 x float> %nc.n212) #8, !dbg !12
  %nc.n287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n253, <4 x float> %nc.n284) #8, !dbg !12
  %nc.n288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n254, <4 x float> %nc.n216) #8, !dbg !12
  %nc.n291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n257, <4 x float> %nc.n288) #8, !dbg !12
  %nc.n292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n258, <4 x float> %nc.n220) #8, !dbg !12
  %nc.n295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n261, <4 x float> %nc.n292) #8, !dbg !12
  %nc.n296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n262, <4 x float> %nc.n224) #8, !dbg !12
  %nc.n299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n265, <4 x float> %nc.n296) #8, !dbg !12
  %nc.n300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n266, <4 x float> %nc.n228) #8, !dbg !12
  %nc.n303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n269, <4 x float> %nc.n300) #8, !dbg !12
  %nc.n304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n85, <4 x half> %nc.n270, <4 x float> %nc.n231) #8, !dbg !12
  %nc.n306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n86, <4 x half> %nc.n273, <4 x float> %nc.n304) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %nc.n307 = or disjoint i64 %nc.n138, 96, !dbg !13
  %.idx14151416 = or disjoint i64 %address.local514, %nc.n307, !dbg !13
  %.idx14151416.tr = trunc nuw nsw i64 %.idx14151416 to i32, !dbg !13
  %dma.buffer.offset1354 = shl nuw nsw i32 %.idx14151416.tr, 1, !dbg !13
  %nc.n312 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1354, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %.idx14171418 = or disjoint i64 %address.local531, %nc.n307, !dbg !13
  %.idx14171418.tr = trunc nuw nsw i64 %.idx14171418 to i32, !dbg !13
  %dma.buffer.offset1351 = shl nuw nsw i32 %.idx14171418.tr, 1, !dbg !13
  %nc.n317 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1351, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %nc.n318 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !12
  %nc.n319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n318, <4 x float> %nc.n247) #8, !dbg !12
  %nc.n320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !12
  %nc.n321 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !12
  %nc.n322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n320, <4 x float> %nc.n319) #8, !dbg !12
  %nc.n323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n321, <4 x float> %nc.n251) #8, !dbg !12
  %nc.n324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !12
  %nc.n325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !12
  %nc.n326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n324, <4 x float> %nc.n323) #8, !dbg !12
  %nc.n327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n325, <4 x float> %nc.n255) #8, !dbg !12
  %nc.n328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !12
  %nc.n329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !12
  %nc.n330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n328, <4 x float> %nc.n327) #8, !dbg !12
  %nc.n331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n329, <4 x float> %nc.n259) #8, !dbg !12
  %nc.n332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !12
  %nc.n333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %nc.n334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n332, <4 x float> %nc.n331) #8, !dbg !12
  %nc.n335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n333, <4 x float> %nc.n263) #8, !dbg !12
  %nc.n336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %nc.n337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %nc.n338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n336, <4 x float> %nc.n335) #8, !dbg !12
  %nc.n339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n337, <4 x float> %nc.n267) #8, !dbg !12
  %nc.n340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %nc.n341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  %nc.n342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n340, <4 x float> %nc.n339) #8, !dbg !12
  %nc.n343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n341, <4 x float> %nc.n271) #8, !dbg !12
  %nc.n344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !12
  %nc.n345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !12
  %nc.n346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n344, <4 x float> %nc.n343) #8, !dbg !12
  %nc.n347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n104, <4 x half> %nc.n345, <4 x float> %nc.n275) #8, !dbg !12
  %nc.n348 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !12
  %nc.n350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n106, <4 x half> %nc.n348, <4 x float> %nc.n347) #8, !dbg !12
  %nc.n351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n318, <4 x float> %nc.n279) #8, !dbg !12
  %nc.n354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n320, <4 x float> %nc.n351) #8, !dbg !12
  %nc.n355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n321, <4 x float> %nc.n283) #8, !dbg !12
  %nc.n358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n324, <4 x float> %nc.n355) #8, !dbg !12
  %nc.n359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n325, <4 x float> %nc.n287) #8, !dbg !12
  %nc.n362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n328, <4 x float> %nc.n359) #8, !dbg !12
  %nc.n363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n329, <4 x float> %nc.n291) #8, !dbg !12
  %nc.n366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n332, <4 x float> %nc.n363) #8, !dbg !12
  %nc.n367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n333, <4 x float> %nc.n295) #8, !dbg !12
  %nc.n370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n336, <4 x float> %nc.n367) #8, !dbg !12
  %nc.n371 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n337, <4 x float> %nc.n299) #8, !dbg !12
  %nc.n374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n340, <4 x float> %nc.n371) #8, !dbg !12
  %nc.n375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n341, <4 x float> %nc.n303) #8, !dbg !12
  %nc.n378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n344, <4 x float> %nc.n375) #8, !dbg !12
  %nc.n379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n107, <4 x half> %nc.n345, <4 x float> %nc.n306) #8, !dbg !12
  %nc.n381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n108, <4 x half> %nc.n348, <4 x float> %nc.n379) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !14
  %address.local918 = and i64 %address.local917, 8064, !dbg !14
  %address.local920 = and i64 %address.local500, 24, !dbg !14
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !14
  %.idx14191420 = or disjoint i64 %address.local921, %nc.n138, !dbg !14
  %.idx14191420.tr = trunc nuw nsw i64 %.idx14191420 to i32, !dbg !14
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !14
  %nc.n386 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1362, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !14
  %address.local924 = and i64 %address.local923, 16256, !dbg !14
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !14
  %.idx14211422 = or disjoint i64 %address.local927, %nc.n138, !dbg !14
  %.idx14211422.tr = trunc nuw nsw i64 %.idx14211422 to i32, !dbg !14
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !14
  %nc.n391 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1359, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %nc.n392 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !12
  %nc.n393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n392, <4 x float> %nc.n322) #8, !dbg !12
  %nc.n394 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !12
  %nc.n395 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !12
  %nc.n396 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n394, <4 x float> %nc.n393) #8, !dbg !12
  %nc.n397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n395, <4 x float> %nc.n326) #8, !dbg !12
  %nc.n398 = extractelement <4 x float> %nc.n396, i64 0, !dbg !15
  %nc.n399 = tail call float @llvm.maxnum.f32(float %nc.n398, float -inf), !dbg !16, !frisk.local_reduction !5
  %nc.n400 = extractelement <4 x float> %nc.n396, i64 1, !dbg !15
  %nc.n401 = extractelement <4 x float> %nc.n396, i64 2, !dbg !15
  %nc.n402 = extractelement <4 x float> %nc.n396, i64 3, !dbg !15
  %nc.n403 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !12
  %nc.n404 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !12
  %nc.n405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n403, <4 x float> %nc.n397) #8, !dbg !12
  %nc.n406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n404, <4 x float> %nc.n330) #8, !dbg !12
  %nc.n407 = extractelement <4 x float> %nc.n405, i64 0, !dbg !15
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %nc.n399, float %nc.n407), !dbg !16, !frisk.local_reduction !5
  %nc.n408 = extractelement <4 x float> %nc.n405, i64 1, !dbg !15
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %nc.n400, float %nc.n408), !dbg !16, !frisk.local_reduction !5
  %nc.n409 = extractelement <4 x float> %nc.n405, i64 2, !dbg !15
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %nc.n401, float %nc.n409), !dbg !16, !frisk.local_reduction !5
  %nc.n410 = extractelement <4 x float> %nc.n405, i64 3, !dbg !15
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %nc.n402, float %nc.n410), !dbg !16, !frisk.local_reduction !5
  %nc.n411 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !12
  %nc.n412 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !12
  %nc.n413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n411, <4 x float> %nc.n406) #8, !dbg !12
  %nc.n414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n412, <4 x float> %nc.n334) #8, !dbg !12
  %nc.n415 = extractelement <4 x float> %nc.n413, i64 0, !dbg !15
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %nc.n415), !dbg !16, !frisk.local_reduction !5
  %nc.n416 = extractelement <4 x float> %nc.n413, i64 1, !dbg !15
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %nc.n416), !dbg !16, !frisk.local_reduction !5
  %nc.n417 = extractelement <4 x float> %nc.n413, i64 2, !dbg !15
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %nc.n417), !dbg !16, !frisk.local_reduction !5
  %nc.n418 = extractelement <4 x float> %nc.n413, i64 3, !dbg !15
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %nc.n418), !dbg !16, !frisk.local_reduction !5
  %nc.n419 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !12
  %nc.n420 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !12
  %nc.n421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n419, <4 x float> %nc.n414) #8, !dbg !12
  %nc.n422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n420, <4 x float> %nc.n338) #8, !dbg !12
  %nc.n423 = extractelement <4 x float> %nc.n421, i64 0, !dbg !15
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %nc.n423), !dbg !16, !frisk.local_reduction !5
  %nc.n424 = extractelement <4 x float> %nc.n421, i64 1, !dbg !15
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %nc.n424), !dbg !16, !frisk.local_reduction !5
  %nc.n425 = extractelement <4 x float> %nc.n421, i64 2, !dbg !15
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %nc.n425), !dbg !16, !frisk.local_reduction !5
  %nc.n426 = extractelement <4 x float> %nc.n421, i64 3, !dbg !15
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %nc.n426), !dbg !16, !frisk.local_reduction !5
  %nc.n427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !12
  %nc.n428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !12
  %nc.n429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n427, <4 x float> %nc.n422) #8, !dbg !12
  %nc.n430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n428, <4 x float> %nc.n342) #8, !dbg !12
  %nc.n431 = extractelement <4 x float> %nc.n429, i64 0, !dbg !15
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %nc.n431), !dbg !16, !frisk.local_reduction !5
  %nc.n432 = extractelement <4 x float> %nc.n429, i64 1, !dbg !15
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %nc.n432), !dbg !16, !frisk.local_reduction !5
  %nc.n433 = extractelement <4 x float> %nc.n429, i64 2, !dbg !15
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %nc.n433), !dbg !16, !frisk.local_reduction !5
  %nc.n434 = extractelement <4 x float> %nc.n429, i64 3, !dbg !15
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %nc.n434), !dbg !16, !frisk.local_reduction !5
  %nc.n435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !12
  %nc.n436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !12
  %nc.n437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n435, <4 x float> %nc.n430) #8, !dbg !12
  %nc.n438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n436, <4 x float> %nc.n346) #8, !dbg !12
  %nc.n439 = extractelement <4 x float> %nc.n437, i64 0, !dbg !15
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %nc.n439), !dbg !16, !frisk.local_reduction !5
  %nc.n440 = extractelement <4 x float> %nc.n437, i64 1, !dbg !15
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %nc.n440), !dbg !16, !frisk.local_reduction !5
  %nc.n441 = extractelement <4 x float> %nc.n437, i64 2, !dbg !15
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %nc.n441), !dbg !16, !frisk.local_reduction !5
  %nc.n442 = extractelement <4 x float> %nc.n437, i64 3, !dbg !15
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %nc.n442), !dbg !16, !frisk.local_reduction !5
  %nc.n443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !12
  %nc.n444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !12
  %nc.n445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n443, <4 x float> %nc.n438) #8, !dbg !12
  %nc.n446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n112, <4 x half> %nc.n444, <4 x float> %nc.n350) #8, !dbg !12
  %nc.n447 = extractelement <4 x float> %nc.n445, i64 0, !dbg !15
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %nc.n447), !dbg !16, !frisk.local_reduction !5
  %nc.n448 = extractelement <4 x float> %nc.n445, i64 1, !dbg !15
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %nc.n448), !dbg !16, !frisk.local_reduction !5
  %nc.n449 = extractelement <4 x float> %nc.n445, i64 2, !dbg !15
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %nc.n449), !dbg !16, !frisk.local_reduction !5
  %nc.n450 = extractelement <4 x float> %nc.n445, i64 3, !dbg !15
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %nc.n450), !dbg !16, !frisk.local_reduction !5
  %nc.n451 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !12
  %nc.n453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n114, <4 x half> %nc.n451, <4 x float> %nc.n446) #8, !dbg !12
  %nc.n454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n392, <4 x float> %nc.n354) #8, !dbg !12
  %nc.n455 = extractelement <4 x float> %nc.n453, i64 0, !dbg !15
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %nc.n455), !dbg !16, !frisk.local_reduction !5
  %nc.n456 = extractelement <4 x float> %nc.n453, i64 1, !dbg !15
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %nc.n456), !dbg !16, !frisk.local_reduction !5
  %nc.n457 = extractelement <4 x float> %nc.n453, i64 2, !dbg !15
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %nc.n457), !dbg !16, !frisk.local_reduction !5
  %nc.n458 = extractelement <4 x float> %nc.n453, i64 3, !dbg !15
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %nc.n458), !dbg !16, !frisk.local_reduction !5
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !16, !frisk.local_reduction !5
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !16, !frisk.local_reduction !5
  %nc.n459 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !16, !frisk.local_reduction !5
  %nc.n462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n394, <4 x float> %nc.n454) #8, !dbg !12
  %nc.n463 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n395, <4 x float> %nc.n358) #8, !dbg !12
  %nc.n464 = extractelement <4 x float> %nc.n462, i64 0, !dbg !15
  %nc.n465 = extractelement <4 x float> %nc.n462, i64 1, !dbg !15
  %nc.n466 = extractelement <4 x float> %nc.n462, i64 2, !dbg !15
  %nc.n467 = extractelement <4 x float> %nc.n462, i64 3, !dbg !15
  %nc.n470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n403, <4 x float> %nc.n463) #8, !dbg !12
  %nc.n471 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n404, <4 x float> %nc.n362) #8, !dbg !12
  %nc.n472 = extractelement <4 x float> %nc.n470, i64 0, !dbg !15
  %nc.n473 = extractelement <4 x float> %nc.n470, i64 1, !dbg !15
  %nc.n474 = extractelement <4 x float> %nc.n470, i64 2, !dbg !15
  %nc.n475 = extractelement <4 x float> %nc.n470, i64 3, !dbg !15
  %nc.n478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n411, <4 x float> %nc.n471) #8, !dbg !12
  %nc.n479 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n412, <4 x float> %nc.n366) #8, !dbg !12
  %nc.n480 = extractelement <4 x float> %nc.n478, i64 0, !dbg !15
  %nc.n481 = extractelement <4 x float> %nc.n478, i64 1, !dbg !15
  %nc.n482 = extractelement <4 x float> %nc.n478, i64 2, !dbg !15
  %nc.n483 = extractelement <4 x float> %nc.n478, i64 3, !dbg !15
  %nc.n486 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n419, <4 x float> %nc.n479) #8, !dbg !12
  %nc.n487 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n420, <4 x float> %nc.n370) #8, !dbg !12
  %nc.n488 = extractelement <4 x float> %nc.n486, i64 0, !dbg !15
  %nc.n489 = extractelement <4 x float> %nc.n486, i64 1, !dbg !15
  %nc.n490 = extractelement <4 x float> %nc.n486, i64 2, !dbg !15
  %nc.n491 = extractelement <4 x float> %nc.n486, i64 3, !dbg !15
  %nc.n494 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n427, <4 x float> %nc.n487) #8, !dbg !12
  %nc.n495 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n428, <4 x float> %nc.n374) #8, !dbg !12
  %nc.n496 = extractelement <4 x float> %nc.n494, i64 0, !dbg !15
  %nc.n497 = extractelement <4 x float> %nc.n494, i64 1, !dbg !15
  %nc.n498 = extractelement <4 x float> %nc.n494, i64 2, !dbg !15
  %nc.n499 = extractelement <4 x float> %nc.n494, i64 3, !dbg !15
  %nc.n502 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n435, <4 x float> %nc.n495) #8, !dbg !12
  %nc.n503 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n436, <4 x float> %nc.n378) #8, !dbg !12
  %nc.n504 = extractelement <4 x float> %nc.n502, i64 0, !dbg !15
  %nc.n505 = extractelement <4 x float> %nc.n502, i64 1, !dbg !15
  %nc.n506 = extractelement <4 x float> %nc.n502, i64 2, !dbg !15
  %nc.n507 = extractelement <4 x float> %nc.n502, i64 3, !dbg !15
  %nc.n510 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n443, <4 x float> %nc.n503) #8, !dbg !12
  %nc.n511 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n115, <4 x half> %nc.n444, <4 x float> %nc.n381) #8, !dbg !12
  %nc.n512 = extractelement <4 x float> %nc.n510, i64 0, !dbg !15
  %nc.n513 = extractelement <4 x float> %nc.n510, i64 1, !dbg !15
  %nc.n514 = extractelement <4 x float> %nc.n510, i64 2, !dbg !15
  %nc.n515 = extractelement <4 x float> %nc.n510, i64 3, !dbg !15
  %nc.n517 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %nc.n116, <4 x half> %nc.n451, <4 x float> %nc.n511) #8, !dbg !12
  %nc.n518 = extractelement <4 x float> %nc.n517, i64 0, !dbg !15
  %nc.n519 = extractelement <4 x float> %nc.n517, i64 1, !dbg !15
  %nc.n520 = extractelement <4 x float> %nc.n517, i64 2, !dbg !15
  %nc.n521 = extractelement <4 x float> %nc.n517, i64 3, !dbg !15
  %nc.n522 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %nc.n523 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %nc.n522), !dbg !16
  %nc.n524 = add nuw nsw i32 %nc.n523, 64, !dbg !16
  %nc.n525 = xor i32 %nc.n523, 16, !dbg !16
  %nc.n526 = and i32 %nc.n524, 192, !dbg !16
  %nc.n527 = icmp samesign ult i32 %nc.n525, %nc.n526, !dbg !16
  %nc.n528 = select i1 %nc.n527, i32 %nc.n525, i32 %nc.n523, !dbg !16
  %nc.n529 = shl nuw nsw i32 %nc.n528, 2, !dbg !16
  %nc.n530 = bitcast float %nc.n459 to i32, !dbg !16
  %nc.n531 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n530), !dbg !16
  %nc.n532 = bitcast i32 %nc.n531 to float, !dbg !16
  %nc.n533 = tail call float @llvm.maxnum.f32(float %nc.n459, float %nc.n532), !dbg !16
  %nc.n534 = xor i32 %nc.n523, 32, !dbg !16
  %nc.n535 = icmp samesign ult i32 %nc.n534, %nc.n526, !dbg !16
  %nc.n536 = select i1 %nc.n535, i32 %nc.n534, i32 %nc.n523, !dbg !16
  %nc.n537 = shl nuw nsw i32 %nc.n536, 2, !dbg !16
  %nc.n538 = bitcast float %nc.n533 to i32, !dbg !16
  %nc.n539 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n538), !dbg !16
  %nc.n540 = bitcast i32 %nc.n539 to float, !dbg !16
  %nc.n541 = tail call float @llvm.maxnum.f32(float %nc.n533, float %nc.n540), !dbg !16
  %nc.n542 = extractvalue [2 x <1 x float>] %nc.n12914, 0, !dbg !16
  %nc.n543 = insertelement <1 x float> poison, float %nc.n541, i64 0, !dbg !16
  %nc.n544 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %nc.n543, <1 x float> %nc.n542), !dbg !16
  %nc.n545 = tail call float @llvm.maxnum.f32(float %nc.n464, float -inf), !dbg !16, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %nc.n545, float %nc.n472), !dbg !16, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %nc.n465, float %nc.n473), !dbg !16, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %nc.n466, float %nc.n474), !dbg !16, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %nc.n467, float %nc.n475), !dbg !16, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %nc.n480), !dbg !16, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %nc.n481), !dbg !16, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %nc.n482), !dbg !16, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %nc.n483), !dbg !16, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %nc.n488), !dbg !16, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %nc.n489), !dbg !16, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %nc.n490), !dbg !16, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %nc.n491), !dbg !16, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %nc.n496), !dbg !16, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %nc.n497), !dbg !16, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %nc.n498), !dbg !16, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %nc.n499), !dbg !16, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %nc.n504), !dbg !16, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %nc.n505), !dbg !16, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %nc.n506), !dbg !16, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %nc.n507), !dbg !16, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %nc.n512), !dbg !16, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %nc.n513), !dbg !16, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %nc.n514), !dbg !16, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %nc.n515), !dbg !16, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %nc.n518), !dbg !16, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %nc.n519), !dbg !16, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %nc.n520), !dbg !16, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %nc.n521), !dbg !16, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !16, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !16, !frisk.local_reduction !5
  %nc.n546 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !16, !frisk.local_reduction !5
  %nc.n547 = bitcast float %nc.n546 to i32, !dbg !16
  %nc.n548 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n547), !dbg !16
  %nc.n549 = bitcast i32 %nc.n548 to float, !dbg !16
  %nc.n550 = tail call float @llvm.maxnum.f32(float %nc.n546, float %nc.n549), !dbg !16
  %nc.n551 = bitcast float %nc.n550 to i32, !dbg !16
  %nc.n552 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n551), !dbg !16
  %nc.n553 = bitcast i32 %nc.n552 to float, !dbg !16
  %nc.n554 = tail call float @llvm.maxnum.f32(float %nc.n550, float %nc.n553), !dbg !16
  %nc.n555 = extractvalue [2 x <1 x float>] %nc.n12914, 1, !dbg !16
  %nc.n556 = insertelement <1 x float> poison, float %nc.n554, i64 0, !dbg !16
  %nc.n557 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %nc.n556, <1 x float> %nc.n555), !dbg !16
  %nc.n558 = shufflevector <1 x float> %nc.n544, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %nc.n559 = fsub <4 x float> %nc.n396, %nc.n558, !dbg !17
  %nc.n560 = fsub <4 x float> %nc.n405, %nc.n558, !dbg !17
  %nc.n561 = fsub <4 x float> %nc.n413, %nc.n558, !dbg !17
  %nc.n562 = fsub <4 x float> %nc.n421, %nc.n558, !dbg !17
  %nc.n563 = fsub <4 x float> %nc.n429, %nc.n558, !dbg !17
  %nc.n564 = fsub <4 x float> %nc.n437, %nc.n558, !dbg !17
  %nc.n565 = fsub <4 x float> %nc.n445, %nc.n558, !dbg !17
  %nc.n566 = fsub <4 x float> %nc.n453, %nc.n558, !dbg !17
  %nc.n567 = shufflevector <1 x float> %nc.n557, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %nc.n568 = fsub <4 x float> %nc.n462, %nc.n567, !dbg !17
  %nc.n569 = fsub <4 x float> %nc.n470, %nc.n567, !dbg !17
  %nc.n570 = fsub <4 x float> %nc.n478, %nc.n567, !dbg !17
  %nc.n571 = fsub <4 x float> %nc.n486, %nc.n567, !dbg !17
  %nc.n572 = fsub <4 x float> %nc.n494, %nc.n567, !dbg !17
  %nc.n573 = fsub <4 x float> %nc.n502, %nc.n567, !dbg !17
  %nc.n574 = fsub <4 x float> %nc.n510, %nc.n567, !dbg !17
  %nc.n575 = fsub <4 x float> %nc.n517, %nc.n567, !dbg !17
  %nc.n576 = fsub <1 x float> %nc.n542, %nc.n544, !dbg !18
  %nc.n577 = extractelement <1 x float> %nc.n576, i64 0, !dbg !19
  %nc.n578 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n577), !dbg !19
  %nc.n579 = insertelement <4 x float> poison, float %nc.n578, i64 0, !dbg !20
  %nc.n580 = shufflevector <4 x float> %nc.n579, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %nc.n581 = extractvalue [2 x <1 x float>] %nc.n12813, 0, !dbg !21
  %nc.n582 = extractelement <1 x float> %nc.n581, i64 0, !dbg !21
  %nc.n583 = fmul float %nc.n582, %nc.n578, !dbg !21
  %nc.n584 = fsub <1 x float> %nc.n555, %nc.n557, !dbg !18
  %nc.n585 = extractelement <1 x float> %nc.n584, i64 0, !dbg !19
  %nc.n586 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n585), !dbg !19
  %nc.n587 = insertelement <4 x float> poison, float %nc.n586, i64 0, !dbg !20
  %nc.n588 = shufflevector <4 x float> %nc.n587, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %nc.n589 = extractvalue [2 x <1 x float>] %nc.n12813, 1, !dbg !21
  %nc.n590 = extractelement <1 x float> %nc.n589, i64 0, !dbg !21
  %nc.n591 = fmul float %nc.n590, %nc.n586, !dbg !21
  %nc.n592 = extractelement <4 x float> %nc.n559, i64 0, !dbg !22
  %nc.n593 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n592), !dbg !22
  %nc.n594 = insertelement <4 x float> poison, float %nc.n593, i64 0, !dbg !23
  %nc.n595 = extractelement <4 x float> %nc.n559, i64 1, !dbg !22
  %nc.n596 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n595), !dbg !22
  %nc.n597 = insertelement <4 x float> %nc.n594, float %nc.n596, i64 1, !dbg !23
  %nc.n598 = extractelement <4 x float> %nc.n559, i64 2, !dbg !22
  %nc.n599 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n598), !dbg !22
  %nc.n600 = insertelement <4 x float> %nc.n597, float %nc.n599, i64 2, !dbg !23
  %nc.n601 = extractelement <4 x float> %nc.n559, i64 3, !dbg !22
  %nc.n602 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n601), !dbg !22
  %nc.n603 = insertelement <4 x float> %nc.n600, float %nc.n602, i64 3, !dbg !23
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n603) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n604 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !23
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n604) #8, !dbg !22, !frisk.register_lifetimes !5
  %nc.n605 = extractelement <4 x float> %nc.n560, i64 0, !dbg !22
  %nc.n606 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n605), !dbg !22
  %nc.n607 = insertelement <4 x float> poison, float %nc.n606, i64 0, !dbg !23
  %local.reduce89 = fadd float %nc.n593, %nc.n606, !dbg !24, !frisk.local_reduction !5
  %nc.n608 = extractelement <4 x float> %nc.n560, i64 1, !dbg !22
  %nc.n609 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n608), !dbg !22
  %nc.n610 = insertelement <4 x float> %nc.n607, float %nc.n609, i64 1, !dbg !23
  %local.reduce90 = fadd float %nc.n596, %nc.n609, !dbg !24, !frisk.local_reduction !5
  %nc.n611 = extractelement <4 x float> %nc.n560, i64 2, !dbg !22
  %nc.n612 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n611), !dbg !22
  %nc.n613 = insertelement <4 x float> %nc.n610, float %nc.n612, i64 2, !dbg !23
  %local.reduce91 = fadd float %nc.n599, %nc.n612, !dbg !24, !frisk.local_reduction !5
  %nc.n614 = extractelement <4 x float> %nc.n560, i64 3, !dbg !22
  %nc.n615 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n614), !dbg !22
  %nc.n616 = insertelement <4 x float> %nc.n613, float %nc.n615, i64 3, !dbg !23
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n616) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n617 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !23
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n617) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce92 = fadd float %nc.n602, %nc.n615, !dbg !24, !frisk.local_reduction !5
  %nc.n618 = extractelement <4 x float> %nc.n561, i64 0, !dbg !22
  %nc.n619 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n618), !dbg !22
  %nc.n620 = insertelement <4 x float> poison, float %nc.n619, i64 0, !dbg !23
  %local.reduce93 = fadd float %nc.n619, %local.reduce89, !dbg !24, !frisk.local_reduction !5
  %nc.n621 = extractelement <4 x float> %nc.n561, i64 1, !dbg !22
  %nc.n622 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n621), !dbg !22
  %nc.n623 = insertelement <4 x float> %nc.n620, float %nc.n622, i64 1, !dbg !23
  %local.reduce94 = fadd float %nc.n622, %local.reduce90, !dbg !24, !frisk.local_reduction !5
  %nc.n624 = extractelement <4 x float> %nc.n561, i64 2, !dbg !22
  %nc.n625 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n624), !dbg !22
  %nc.n626 = insertelement <4 x float> %nc.n623, float %nc.n625, i64 2, !dbg !23
  %local.reduce95 = fadd float %nc.n625, %local.reduce91, !dbg !24, !frisk.local_reduction !5
  %nc.n627 = extractelement <4 x float> %nc.n561, i64 3, !dbg !22
  %nc.n628 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n627), !dbg !22
  %nc.n629 = insertelement <4 x float> %nc.n626, float %nc.n628, i64 3, !dbg !23
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n629) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n630 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !23
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n630) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce96 = fadd float %nc.n628, %local.reduce92, !dbg !24, !frisk.local_reduction !5
  %nc.n631 = extractelement <4 x float> %nc.n562, i64 0, !dbg !22
  %nc.n632 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n631), !dbg !22
  %nc.n633 = insertelement <4 x float> poison, float %nc.n632, i64 0, !dbg !23
  %local.reduce97 = fadd float %nc.n632, %local.reduce93, !dbg !24, !frisk.local_reduction !5
  %nc.n634 = extractelement <4 x float> %nc.n562, i64 1, !dbg !22
  %nc.n635 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n634), !dbg !22
  %nc.n636 = insertelement <4 x float> %nc.n633, float %nc.n635, i64 1, !dbg !23
  %local.reduce98 = fadd float %nc.n635, %local.reduce94, !dbg !24, !frisk.local_reduction !5
  %nc.n637 = extractelement <4 x float> %nc.n562, i64 2, !dbg !22
  %nc.n638 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n637), !dbg !22
  %nc.n639 = insertelement <4 x float> %nc.n636, float %nc.n638, i64 2, !dbg !23
  %local.reduce99 = fadd float %nc.n638, %local.reduce95, !dbg !24, !frisk.local_reduction !5
  %nc.n640 = extractelement <4 x float> %nc.n562, i64 3, !dbg !22
  %nc.n641 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n640), !dbg !22
  %nc.n642 = insertelement <4 x float> %nc.n639, float %nc.n641, i64 3, !dbg !23
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n642) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n643 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !23
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n643) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce100 = fadd float %nc.n641, %local.reduce96, !dbg !24, !frisk.local_reduction !5
  %nc.n644 = extractelement <4 x float> %nc.n563, i64 0, !dbg !22
  %nc.n645 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n644), !dbg !22
  %nc.n646 = insertelement <4 x float> poison, float %nc.n645, i64 0, !dbg !23
  %local.reduce101 = fadd float %nc.n645, %local.reduce97, !dbg !24, !frisk.local_reduction !5
  %nc.n647 = extractelement <4 x float> %nc.n563, i64 1, !dbg !22
  %nc.n648 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n647), !dbg !22
  %nc.n649 = insertelement <4 x float> %nc.n646, float %nc.n648, i64 1, !dbg !23
  %local.reduce102 = fadd float %nc.n648, %local.reduce98, !dbg !24, !frisk.local_reduction !5
  %nc.n650 = extractelement <4 x float> %nc.n563, i64 2, !dbg !22
  %nc.n651 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n650), !dbg !22
  %nc.n652 = insertelement <4 x float> %nc.n649, float %nc.n651, i64 2, !dbg !23
  %local.reduce103 = fadd float %nc.n651, %local.reduce99, !dbg !24, !frisk.local_reduction !5
  %nc.n653 = extractelement <4 x float> %nc.n563, i64 3, !dbg !22
  %nc.n654 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n653), !dbg !22
  %nc.n655 = insertelement <4 x float> %nc.n652, float %nc.n654, i64 3, !dbg !23
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n655) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n656 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !23
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n656) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce104 = fadd float %nc.n654, %local.reduce100, !dbg !24, !frisk.local_reduction !5
  %nc.n657 = extractelement <4 x float> %nc.n564, i64 0, !dbg !22
  %nc.n658 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n657), !dbg !22
  %nc.n659 = insertelement <4 x float> poison, float %nc.n658, i64 0, !dbg !23
  %local.reduce105 = fadd float %nc.n658, %local.reduce101, !dbg !24, !frisk.local_reduction !5
  %nc.n660 = extractelement <4 x float> %nc.n564, i64 1, !dbg !22
  %nc.n661 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n660), !dbg !22
  %nc.n662 = insertelement <4 x float> %nc.n659, float %nc.n661, i64 1, !dbg !23
  %local.reduce106 = fadd float %nc.n661, %local.reduce102, !dbg !24, !frisk.local_reduction !5
  %nc.n663 = extractelement <4 x float> %nc.n564, i64 2, !dbg !22
  %nc.n664 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n663), !dbg !22
  %nc.n665 = insertelement <4 x float> %nc.n662, float %nc.n664, i64 2, !dbg !23
  %local.reduce107 = fadd float %nc.n664, %local.reduce103, !dbg !24, !frisk.local_reduction !5
  %nc.n666 = extractelement <4 x float> %nc.n564, i64 3, !dbg !22
  %nc.n667 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n666), !dbg !22
  %nc.n668 = insertelement <4 x float> %nc.n665, float %nc.n667, i64 3, !dbg !23
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n668) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n669 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !23
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n669) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce108 = fadd float %nc.n667, %local.reduce104, !dbg !24, !frisk.local_reduction !5
  %nc.n670 = extractelement <4 x float> %nc.n565, i64 0, !dbg !22
  %nc.n671 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n670), !dbg !22
  %nc.n672 = insertelement <4 x float> poison, float %nc.n671, i64 0, !dbg !23
  %local.reduce109 = fadd float %nc.n671, %local.reduce105, !dbg !24, !frisk.local_reduction !5
  %nc.n673 = extractelement <4 x float> %nc.n565, i64 1, !dbg !22
  %nc.n674 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n673), !dbg !22
  %nc.n675 = insertelement <4 x float> %nc.n672, float %nc.n674, i64 1, !dbg !23
  %local.reduce110 = fadd float %nc.n674, %local.reduce106, !dbg !24, !frisk.local_reduction !5
  %nc.n676 = extractelement <4 x float> %nc.n565, i64 2, !dbg !22
  %nc.n677 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n676), !dbg !22
  %nc.n678 = insertelement <4 x float> %nc.n675, float %nc.n677, i64 2, !dbg !23
  %local.reduce111 = fadd float %nc.n677, %local.reduce107, !dbg !24, !frisk.local_reduction !5
  %nc.n679 = extractelement <4 x float> %nc.n565, i64 3, !dbg !22
  %nc.n680 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n679), !dbg !22
  %nc.n681 = insertelement <4 x float> %nc.n678, float %nc.n680, i64 3, !dbg !23
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n681) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n682 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !23
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n682) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce112 = fadd float %nc.n680, %local.reduce108, !dbg !24, !frisk.local_reduction !5
  %nc.n683 = extractelement <4 x float> %nc.n566, i64 0, !dbg !22
  %nc.n684 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n683), !dbg !22
  %nc.n685 = insertelement <4 x float> poison, float %nc.n684, i64 0, !dbg !23
  %local.reduce113 = fadd float %nc.n684, %local.reduce109, !dbg !24, !frisk.local_reduction !5
  %nc.n686 = extractelement <4 x float> %nc.n566, i64 1, !dbg !22
  %nc.n687 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n686), !dbg !22
  %nc.n688 = insertelement <4 x float> %nc.n685, float %nc.n687, i64 1, !dbg !23
  %local.reduce114 = fadd float %nc.n687, %local.reduce110, !dbg !24, !frisk.local_reduction !5
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !24, !frisk.local_reduction !5
  %nc.n689 = extractelement <4 x float> %nc.n566, i64 2, !dbg !22
  %nc.n690 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n689), !dbg !22
  %nc.n691 = insertelement <4 x float> %nc.n688, float %nc.n690, i64 2, !dbg !23
  %local.reduce115 = fadd float %nc.n690, %local.reduce111, !dbg !24, !frisk.local_reduction !5
  %nc.n692 = extractelement <4 x float> %nc.n566, i64 3, !dbg !22
  %nc.n693 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n692), !dbg !22
  %nc.n694 = insertelement <4 x float> %nc.n691, float %nc.n693, i64 3, !dbg !23
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n694) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n695 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !23
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n695) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce116 = fadd float %nc.n693, %local.reduce112, !dbg !24, !frisk.local_reduction !5
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !24, !frisk.local_reduction !5
  %nc.n696 = fadd float %local.reduce117, %local.reduce118, !dbg !24, !frisk.local_reduction !5
  %nc.n697 = fadd float %nc.n583, %nc.n696, !dbg !24
  %nc.n700 = extractelement <4 x float> %nc.n568, i64 0, !dbg !22
  %nc.n701 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n700), !dbg !22
  %nc.n702 = insertelement <4 x float> poison, float %nc.n701, i64 0, !dbg !23
  %nc.n703 = extractelement <4 x float> %nc.n568, i64 1, !dbg !22
  %nc.n704 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n703), !dbg !22
  %nc.n705 = insertelement <4 x float> %nc.n702, float %nc.n704, i64 1, !dbg !23
  %nc.n706 = extractelement <4 x float> %nc.n568, i64 2, !dbg !22
  %nc.n707 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n706), !dbg !22
  %nc.n708 = insertelement <4 x float> %nc.n705, float %nc.n707, i64 2, !dbg !23
  %nc.n709 = extractelement <4 x float> %nc.n568, i64 3, !dbg !22
  %nc.n710 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n709), !dbg !22
  %nc.n711 = insertelement <4 x float> %nc.n708, float %nc.n710, i64 3, !dbg !23
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n711) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n712 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !23
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n712) #8, !dbg !22, !frisk.register_lifetimes !5
  %nc.n713 = extractelement <4 x float> %nc.n569, i64 0, !dbg !22
  %nc.n714 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n713), !dbg !22
  %nc.n715 = insertelement <4 x float> poison, float %nc.n714, i64 0, !dbg !23
  %local.reduce = fadd float %nc.n701, %nc.n714, !dbg !24, !frisk.local_reduction !5
  %nc.n716 = extractelement <4 x float> %nc.n569, i64 1, !dbg !22
  %nc.n717 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n716), !dbg !22
  %nc.n718 = insertelement <4 x float> %nc.n715, float %nc.n717, i64 1, !dbg !23
  %local.reduce59 = fadd float %nc.n704, %nc.n717, !dbg !24, !frisk.local_reduction !5
  %nc.n719 = extractelement <4 x float> %nc.n569, i64 2, !dbg !22
  %nc.n720 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n719), !dbg !22
  %nc.n721 = insertelement <4 x float> %nc.n718, float %nc.n720, i64 2, !dbg !23
  %local.reduce60 = fadd float %nc.n707, %nc.n720, !dbg !24, !frisk.local_reduction !5
  %nc.n722 = extractelement <4 x float> %nc.n569, i64 3, !dbg !22
  %nc.n723 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n722), !dbg !22
  %nc.n724 = insertelement <4 x float> %nc.n721, float %nc.n723, i64 3, !dbg !23
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n724) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n725 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !23
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n725) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce61 = fadd float %nc.n710, %nc.n723, !dbg !24, !frisk.local_reduction !5
  %nc.n726 = extractelement <4 x float> %nc.n570, i64 0, !dbg !22
  %nc.n727 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n726), !dbg !22
  %nc.n728 = insertelement <4 x float> poison, float %nc.n727, i64 0, !dbg !23
  %local.reduce62 = fadd float %nc.n727, %local.reduce, !dbg !24, !frisk.local_reduction !5
  %nc.n729 = extractelement <4 x float> %nc.n570, i64 1, !dbg !22
  %nc.n730 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n729), !dbg !22
  %nc.n731 = insertelement <4 x float> %nc.n728, float %nc.n730, i64 1, !dbg !23
  %local.reduce63 = fadd float %nc.n730, %local.reduce59, !dbg !24, !frisk.local_reduction !5
  %nc.n732 = extractelement <4 x float> %nc.n570, i64 2, !dbg !22
  %nc.n733 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n732), !dbg !22
  %nc.n734 = insertelement <4 x float> %nc.n731, float %nc.n733, i64 2, !dbg !23
  %local.reduce64 = fadd float %nc.n733, %local.reduce60, !dbg !24, !frisk.local_reduction !5
  %nc.n735 = extractelement <4 x float> %nc.n570, i64 3, !dbg !22
  %nc.n736 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n735), !dbg !22
  %nc.n737 = insertelement <4 x float> %nc.n734, float %nc.n736, i64 3, !dbg !23
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n737) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n738 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !23
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n738) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce65 = fadd float %nc.n736, %local.reduce61, !dbg !24, !frisk.local_reduction !5
  %nc.n739 = extractelement <4 x float> %nc.n571, i64 0, !dbg !22
  %nc.n740 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n739), !dbg !22
  %nc.n741 = insertelement <4 x float> poison, float %nc.n740, i64 0, !dbg !23
  %local.reduce66 = fadd float %nc.n740, %local.reduce62, !dbg !24, !frisk.local_reduction !5
  %nc.n742 = extractelement <4 x float> %nc.n571, i64 1, !dbg !22
  %nc.n743 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n742), !dbg !22
  %nc.n744 = insertelement <4 x float> %nc.n741, float %nc.n743, i64 1, !dbg !23
  %local.reduce67 = fadd float %nc.n743, %local.reduce63, !dbg !24, !frisk.local_reduction !5
  %nc.n745 = extractelement <4 x float> %nc.n571, i64 2, !dbg !22
  %nc.n746 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n745), !dbg !22
  %nc.n747 = insertelement <4 x float> %nc.n744, float %nc.n746, i64 2, !dbg !23
  %local.reduce68 = fadd float %nc.n746, %local.reduce64, !dbg !24, !frisk.local_reduction !5
  %nc.n748 = extractelement <4 x float> %nc.n571, i64 3, !dbg !22
  %nc.n749 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n748), !dbg !22
  %nc.n750 = insertelement <4 x float> %nc.n747, float %nc.n749, i64 3, !dbg !23
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n750) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n751 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !23
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n751) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce69 = fadd float %nc.n749, %local.reduce65, !dbg !24, !frisk.local_reduction !5
  %nc.n752 = extractelement <4 x float> %nc.n572, i64 0, !dbg !22
  %nc.n753 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n752), !dbg !22
  %nc.n754 = insertelement <4 x float> poison, float %nc.n753, i64 0, !dbg !23
  %local.reduce70 = fadd float %nc.n753, %local.reduce66, !dbg !24, !frisk.local_reduction !5
  %nc.n755 = extractelement <4 x float> %nc.n572, i64 1, !dbg !22
  %nc.n756 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n755), !dbg !22
  %nc.n757 = insertelement <4 x float> %nc.n754, float %nc.n756, i64 1, !dbg !23
  %local.reduce71 = fadd float %nc.n756, %local.reduce67, !dbg !24, !frisk.local_reduction !5
  %nc.n758 = extractelement <4 x float> %nc.n572, i64 2, !dbg !22
  %nc.n759 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n758), !dbg !22
  %nc.n760 = insertelement <4 x float> %nc.n757, float %nc.n759, i64 2, !dbg !23
  %local.reduce72 = fadd float %nc.n759, %local.reduce68, !dbg !24, !frisk.local_reduction !5
  %nc.n761 = extractelement <4 x float> %nc.n572, i64 3, !dbg !22
  %nc.n762 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n761), !dbg !22
  %nc.n763 = insertelement <4 x float> %nc.n760, float %nc.n762, i64 3, !dbg !23
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n763) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n764 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !23
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n764) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce73 = fadd float %nc.n762, %local.reduce69, !dbg !24, !frisk.local_reduction !5
  %nc.n765 = extractelement <4 x float> %nc.n573, i64 0, !dbg !22
  %nc.n766 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n765), !dbg !22
  %nc.n767 = insertelement <4 x float> poison, float %nc.n766, i64 0, !dbg !23
  %local.reduce74 = fadd float %nc.n766, %local.reduce70, !dbg !24, !frisk.local_reduction !5
  %nc.n768 = extractelement <4 x float> %nc.n573, i64 1, !dbg !22
  %nc.n769 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n768), !dbg !22
  %nc.n770 = insertelement <4 x float> %nc.n767, float %nc.n769, i64 1, !dbg !23
  %local.reduce75 = fadd float %nc.n769, %local.reduce71, !dbg !24, !frisk.local_reduction !5
  %nc.n771 = extractelement <4 x float> %nc.n573, i64 2, !dbg !22
  %nc.n772 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n771), !dbg !22
  %nc.n773 = insertelement <4 x float> %nc.n770, float %nc.n772, i64 2, !dbg !23
  %local.reduce76 = fadd float %nc.n772, %local.reduce72, !dbg !24, !frisk.local_reduction !5
  %nc.n774 = extractelement <4 x float> %nc.n573, i64 3, !dbg !22
  %nc.n775 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n774), !dbg !22
  %nc.n776 = insertelement <4 x float> %nc.n773, float %nc.n775, i64 3, !dbg !23
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n776) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n777 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !23
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n777) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce77 = fadd float %nc.n775, %local.reduce73, !dbg !24, !frisk.local_reduction !5
  %nc.n778 = extractelement <4 x float> %nc.n574, i64 0, !dbg !22
  %nc.n779 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n778), !dbg !22
  %nc.n780 = insertelement <4 x float> poison, float %nc.n779, i64 0, !dbg !23
  %local.reduce78 = fadd float %nc.n779, %local.reduce74, !dbg !24, !frisk.local_reduction !5
  %nc.n781 = extractelement <4 x float> %nc.n574, i64 1, !dbg !22
  %nc.n782 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n781), !dbg !22
  %nc.n783 = insertelement <4 x float> %nc.n780, float %nc.n782, i64 1, !dbg !23
  %local.reduce79 = fadd float %nc.n782, %local.reduce75, !dbg !24, !frisk.local_reduction !5
  %nc.n784 = extractelement <4 x float> %nc.n574, i64 2, !dbg !22
  %nc.n785 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n784), !dbg !22
  %nc.n786 = insertelement <4 x float> %nc.n783, float %nc.n785, i64 2, !dbg !23
  %local.reduce80 = fadd float %nc.n785, %local.reduce76, !dbg !24, !frisk.local_reduction !5
  %nc.n787 = extractelement <4 x float> %nc.n574, i64 3, !dbg !22
  %nc.n788 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n787), !dbg !22
  %nc.n789 = insertelement <4 x float> %nc.n786, float %nc.n788, i64 3, !dbg !23
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n789) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n790 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !23
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n790) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce81 = fadd float %nc.n788, %local.reduce77, !dbg !24, !frisk.local_reduction !5
  %nc.n791 = extractelement <4 x float> %nc.n575, i64 0, !dbg !22
  %nc.n792 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n791), !dbg !22
  %nc.n793 = insertelement <4 x float> poison, float %nc.n792, i64 0, !dbg !23
  %local.reduce82 = fadd float %nc.n792, %local.reduce78, !dbg !24, !frisk.local_reduction !5
  %nc.n794 = extractelement <4 x float> %nc.n575, i64 1, !dbg !22
  %nc.n795 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n794), !dbg !22
  %nc.n796 = insertelement <4 x float> %nc.n793, float %nc.n795, i64 1, !dbg !23
  %local.reduce83 = fadd float %nc.n795, %local.reduce79, !dbg !24, !frisk.local_reduction !5
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !24, !frisk.local_reduction !5
  %nc.n797 = extractelement <4 x float> %nc.n575, i64 2, !dbg !22
  %nc.n798 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n797), !dbg !22
  %nc.n799 = insertelement <4 x float> %nc.n796, float %nc.n798, i64 2, !dbg !23
  %local.reduce84 = fadd float %nc.n798, %local.reduce80, !dbg !24, !frisk.local_reduction !5
  %nc.n800 = extractelement <4 x float> %nc.n575, i64 3, !dbg !22
  %nc.n801 = tail call float @llvm.amdgcn.exp2.f32(float %nc.n800), !dbg !22
  %nc.n802 = insertelement <4 x float> %nc.n799, float %nc.n801, i64 3, !dbg !23
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %nc.n802) #8, !dbg !23, !frisk.register_lifetimes !5
  %nc.n803 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !23
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %nc.n803) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce85 = fadd float %nc.n801, %local.reduce81, !dbg !24, !frisk.local_reduction !5
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !24, !frisk.local_reduction !5
  %nc.n804 = fadd float %local.reduce86, %local.reduce87, !dbg !24, !frisk.local_reduction !5
  %nc.n805 = fadd float %nc.n591, %nc.n804, !dbg !24
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
  %nc.n808 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #10, !dbg !25
  %nc.n809 = bitcast <4 x i32> %nc.n808 to <8 x half>, !dbg !25
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
  %nc.n810 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #10, !dbg !25
  %nc.n811 = bitcast <4 x i32> %nc.n810 to <8 x half>, !dbg !25
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !25
  %nc.n812 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #10, !dbg !25
  %nc.n813 = bitcast <4 x i32> %nc.n812 to <8 x half>, !dbg !25
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !25
  %nc.n814 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !25
  %nc.n815 = bitcast <4 x i32> %nc.n814 to <8 x half>, !dbg !25
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !25
  %nc.n816 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !25
  %nc.n817 = bitcast <4 x i32> %nc.n816 to <8 x half>, !dbg !25
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !25
  %nc.n818 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !25
  %nc.n819 = bitcast <4 x i32> %nc.n818 to <8 x half>, !dbg !25
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !25
  %nc.n820 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !25
  %nc.n821 = bitcast <4 x i32> %nc.n820 to <8 x half>, !dbg !25
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !25
  %nc.n822 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #10, !dbg !25
  %nc.n823 = bitcast <4 x i32> %nc.n822 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx14231424 = or disjoint i64 %address.local921, %nc.n149, !dbg !14
  %.idx14231424.tr = trunc nuw nsw i64 %.idx14231424 to i32, !dbg !14
  %dma.buffer.offset1342 = shl nuw nsw i32 %.idx14231424.tr, 1, !dbg !14
  %nc.n828 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1362, i32 %dma.buffer.offset1342, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %.idx14251426 = or disjoint i64 %address.local927, %nc.n149, !dbg !14
  %.idx14251426.tr = trunc nuw nsw i64 %.idx14251426 to i32, !dbg !14
  %dma.buffer.offset1339 = shl nuw nsw i32 %.idx14251426.tr, 1, !dbg !14
  %nc.n833 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1359, i32 %dma.buffer.offset1339, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %nc.n834 = shufflevector <8 x half> %nc.n809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n835 = extractvalue [2 x <8 x float>] %nc.n12712, 0, !dbg !20
  %nc.n836 = shufflevector <8 x float> %nc.n835, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %nc.n837 = fmul <4 x float> %nc.n836, %nc.n580, !dbg !20
  %nc.n838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n834, <4 x float> %nc.n837) #8, !dbg !25
  %nc.n839 = shufflevector <8 x half> %nc.n811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n839, <4 x float> %nc.n838) #8, !dbg !25
  %nc.n841 = shufflevector <8 x half> %nc.n813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n841, <4 x float> %nc.n840) #8, !dbg !25
  %nc.n843 = shufflevector <8 x half> %nc.n815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n843, <4 x float> %nc.n842) #8, !dbg !25
  %nc.n845 = shufflevector <8 x half> %nc.n817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n845, <4 x float> %nc.n844) #8, !dbg !25
  %nc.n847 = shufflevector <8 x half> %nc.n819, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n847, <4 x float> %nc.n846) #8, !dbg !25
  %nc.n849 = shufflevector <8 x half> %nc.n821, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n849, <4 x float> %nc.n848) #8, !dbg !25
  %nc.n851 = shufflevector <8 x half> %nc.n823, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n851, <4 x float> %nc.n850) #8, !dbg !25
  %nc.n853 = shufflevector <8 x half> %nc.n809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n854 = shufflevector <8 x float> %nc.n835, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %nc.n855 = fmul <4 x float> %nc.n854, %nc.n580, !dbg !20
  %nc.n856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n853, <4 x float> %nc.n855) #8, !dbg !25
  %nc.n857 = shufflevector <8 x half> %nc.n811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n857, <4 x float> %nc.n856) #8, !dbg !25
  %nc.n859 = shufflevector <8 x half> %nc.n813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n860 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n859, <4 x float> %nc.n858) #8, !dbg !25
  %nc.n861 = shufflevector <8 x half> %nc.n815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n862 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n861, <4 x float> %nc.n860) #8, !dbg !25
  %nc.n863 = shufflevector <8 x half> %nc.n817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n864 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n863, <4 x float> %nc.n862) #8, !dbg !25
  %nc.n865 = shufflevector <8 x half> %nc.n819, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n866 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n865, <4 x float> %nc.n864) #8, !dbg !25
  %nc.n867 = shufflevector <8 x half> %nc.n821, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n867, <4 x float> %nc.n866) #8, !dbg !25
  %nc.n869 = shufflevector <8 x half> %nc.n823, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n870 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n869, <4 x float> %nc.n868) #8, !dbg !25
  %nc.n871 = extractvalue [2 x <8 x float>] %nc.n12712, 1, !dbg !20
  %nc.n872 = shufflevector <8 x float> %nc.n871, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %nc.n873 = fmul <4 x float> %nc.n872, %nc.n588, !dbg !20
  %nc.n874 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n834, <4 x float> %nc.n873) #8, !dbg !25
  %nc.n875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n839, <4 x float> %nc.n874) #8, !dbg !25
  %nc.n876 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n841, <4 x float> %nc.n875) #8, !dbg !25
  %nc.n877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n843, <4 x float> %nc.n876) #8, !dbg !25
  %nc.n878 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n845, <4 x float> %nc.n877) #8, !dbg !25
  %nc.n879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n847, <4 x float> %nc.n878) #8, !dbg !25
  %nc.n880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n849, <4 x float> %nc.n879) #8, !dbg !25
  %nc.n881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n851, <4 x float> %nc.n880) #8, !dbg !25
  %nc.n882 = shufflevector <8 x float> %nc.n871, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %nc.n883 = fmul <4 x float> %nc.n882, %nc.n588, !dbg !20
  %nc.n884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n853, <4 x float> %nc.n883) #8, !dbg !25
  %nc.n885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n857, <4 x float> %nc.n884) #8, !dbg !25
  %nc.n886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n859, <4 x float> %nc.n885) #8, !dbg !25
  %nc.n887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n861, <4 x float> %nc.n886) #8, !dbg !25
  %nc.n888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n863, <4 x float> %nc.n887) #8, !dbg !25
  %nc.n889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n865, <4 x float> %nc.n888) #8, !dbg !25
  %nc.n890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n867, <4 x float> %nc.n889) #8, !dbg !25
  %nc.n891 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n869, <4 x float> %nc.n890) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %nc.n892 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #10, !dbg !25
  %nc.n893 = bitcast <4 x i32> %nc.n892 to <8 x half>, !dbg !25
  %nc.n894 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #10, !dbg !25
  %nc.n895 = bitcast <4 x i32> %nc.n894 to <8 x half>, !dbg !25
  %nc.n896 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #10, !dbg !25
  %nc.n897 = bitcast <4 x i32> %nc.n896 to <8 x half>, !dbg !25
  %nc.n898 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !25
  %nc.n899 = bitcast <4 x i32> %nc.n898 to <8 x half>, !dbg !25
  %nc.n900 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !25
  %nc.n901 = bitcast <4 x i32> %nc.n900 to <8 x half>, !dbg !25
  %nc.n902 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !25
  %nc.n903 = bitcast <4 x i32> %nc.n902 to <8 x half>, !dbg !25
  %nc.n904 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !25
  %nc.n905 = bitcast <4 x i32> %nc.n904 to <8 x half>, !dbg !25
  %nc.n906 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #10, !dbg !25
  %nc.n907 = bitcast <4 x i32> %nc.n906 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx14271428 = or disjoint i64 %address.local921, %nc.n232, !dbg !14
  %.idx14271428.tr = trunc nuw nsw i64 %.idx14271428 to i32, !dbg !14
  %dma.buffer.offset1336 = shl nuw nsw i32 %.idx14271428.tr, 1, !dbg !14
  %nc.n912 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1362, i32 %dma.buffer.offset1336, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %.idx14291430 = or disjoint i64 %address.local927, %nc.n232, !dbg !14
  %.idx14291430.tr = trunc nuw nsw i64 %.idx14291430 to i32, !dbg !14
  %dma.buffer.offset1333 = shl nuw nsw i32 %.idx14291430.tr, 1, !dbg !14
  %nc.n917 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1359, i32 %dma.buffer.offset1333, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %nc.n918 = shufflevector <8 x half> %nc.n893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n919 = extractvalue [2 x <8 x float>] %nc.n13015, 0, !dbg !20
  %nc.n920 = shufflevector <8 x float> %nc.n919, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %nc.n921 = fmul <4 x float> %nc.n920, %nc.n580, !dbg !20
  %nc.n922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n918, <4 x float> %nc.n921) #8, !dbg !25
  %nc.n923 = shufflevector <8 x half> %nc.n895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n923, <4 x float> %nc.n922) #8, !dbg !25
  %nc.n925 = shufflevector <8 x half> %nc.n897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n925, <4 x float> %nc.n924) #8, !dbg !25
  %nc.n927 = shufflevector <8 x half> %nc.n899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n927, <4 x float> %nc.n926) #8, !dbg !25
  %nc.n929 = shufflevector <8 x half> %nc.n901, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n929, <4 x float> %nc.n928) #8, !dbg !25
  %nc.n931 = shufflevector <8 x half> %nc.n903, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n932 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n931, <4 x float> %nc.n930) #8, !dbg !25
  %nc.n933 = shufflevector <8 x half> %nc.n905, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n934 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n933, <4 x float> %nc.n932) #8, !dbg !25
  %nc.n935 = shufflevector <8 x half> %nc.n907, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n935, <4 x float> %nc.n934) #8, !dbg !25
  %nc.n937 = shufflevector <8 x half> %nc.n893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n938 = shufflevector <8 x float> %nc.n919, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %nc.n939 = fmul <4 x float> %nc.n938, %nc.n580, !dbg !20
  %nc.n940 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n937, <4 x float> %nc.n939) #8, !dbg !25
  %nc.n941 = shufflevector <8 x half> %nc.n895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n942 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n941, <4 x float> %nc.n940) #8, !dbg !25
  %nc.n943 = shufflevector <8 x half> %nc.n897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n944 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n943, <4 x float> %nc.n942) #8, !dbg !25
  %nc.n945 = shufflevector <8 x half> %nc.n899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n946 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n945, <4 x float> %nc.n944) #8, !dbg !25
  %nc.n947 = shufflevector <8 x half> %nc.n901, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n947, <4 x float> %nc.n946) #8, !dbg !25
  %nc.n949 = shufflevector <8 x half> %nc.n903, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n950 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n949, <4 x float> %nc.n948) #8, !dbg !25
  %nc.n951 = shufflevector <8 x half> %nc.n905, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n951, <4 x float> %nc.n950) #8, !dbg !25
  %nc.n953 = shufflevector <8 x half> %nc.n907, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n953, <4 x float> %nc.n952) #8, !dbg !25
  %nc.n955 = extractvalue [2 x <8 x float>] %nc.n13015, 1, !dbg !20
  %nc.n956 = shufflevector <8 x float> %nc.n955, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %nc.n957 = fmul <4 x float> %nc.n956, %nc.n588, !dbg !20
  %nc.n958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n918, <4 x float> %nc.n957) #8, !dbg !25
  %nc.n959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n923, <4 x float> %nc.n958) #8, !dbg !25
  %nc.n960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n925, <4 x float> %nc.n959) #8, !dbg !25
  %nc.n961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n927, <4 x float> %nc.n960) #8, !dbg !25
  %nc.n962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n929, <4 x float> %nc.n961) #8, !dbg !25
  %nc.n963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n931, <4 x float> %nc.n962) #8, !dbg !25
  %nc.n964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n933, <4 x float> %nc.n963) #8, !dbg !25
  %nc.n965 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n935, <4 x float> %nc.n964) #8, !dbg !25
  %nc.n966 = shufflevector <8 x float> %nc.n955, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %nc.n967 = fmul <4 x float> %nc.n966, %nc.n588, !dbg !20
  %nc.n968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n937, <4 x float> %nc.n967) #8, !dbg !25
  %nc.n969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n941, <4 x float> %nc.n968) #8, !dbg !25
  %nc.n970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n943, <4 x float> %nc.n969) #8, !dbg !25
  %nc.n971 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n945, <4 x float> %nc.n970) #8, !dbg !25
  %nc.n972 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n947, <4 x float> %nc.n971) #8, !dbg !25
  %nc.n973 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n949, <4 x float> %nc.n972) #8, !dbg !25
  %nc.n974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n951, <4 x float> %nc.n973) #8, !dbg !25
  %nc.n975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n953, <4 x float> %nc.n974) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %nc.n976 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #10, !dbg !25
  %nc.n977 = bitcast <4 x i32> %nc.n976 to <8 x half>, !dbg !25
  %nc.n978 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #10, !dbg !25
  %nc.n979 = bitcast <4 x i32> %nc.n978 to <8 x half>, !dbg !25
  %nc.n980 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #10, !dbg !25
  %nc.n981 = bitcast <4 x i32> %nc.n980 to <8 x half>, !dbg !25
  %nc.n982 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !25
  %nc.n983 = bitcast <4 x i32> %nc.n982 to <8 x half>, !dbg !25
  %nc.n984 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !25
  %nc.n985 = bitcast <4 x i32> %nc.n984 to <8 x half>, !dbg !25
  %nc.n986 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !25
  %nc.n987 = bitcast <4 x i32> %nc.n986 to <8 x half>, !dbg !25
  %nc.n988 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !25
  %nc.n989 = bitcast <4 x i32> %nc.n988 to <8 x half>, !dbg !25
  %nc.n990 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #10, !dbg !25
  %nc.n991 = bitcast <4 x i32> %nc.n990 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx14311432 = or disjoint i64 %address.local921, %nc.n307, !dbg !14
  %.idx14311432.tr = trunc nuw nsw i64 %.idx14311432 to i32, !dbg !14
  %dma.buffer.offset1330 = shl nuw nsw i32 %.idx14311432.tr, 1, !dbg !14
  %nc.n996 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1362, i32 %dma.buffer.offset1330, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %.idx14331434 = or disjoint i64 %address.local927, %nc.n307, !dbg !14
  %.idx14331434.tr = trunc nuw nsw i64 %.idx14331434 to i32, !dbg !14
  %dma.buffer.offset = shl nuw nsw i32 %.idx14331434.tr, 1, !dbg !14
  %nc.n1001 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1359, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %nc.n1002 = shufflevector <8 x half> %nc.n977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1003 = extractvalue [2 x <8 x float>] %nc.n13116, 0, !dbg !20
  %nc.n1004 = shufflevector <8 x float> %nc.n1003, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %nc.n1005 = fmul <4 x float> %nc.n1004, %nc.n580, !dbg !20
  %nc.n1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n1002, <4 x float> %nc.n1005) #8, !dbg !25
  %nc.n1007 = shufflevector <8 x half> %nc.n979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n1007, <4 x float> %nc.n1006) #8, !dbg !25
  %nc.n1009 = shufflevector <8 x half> %nc.n981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n1009, <4 x float> %nc.n1008) #8, !dbg !25
  %nc.n1011 = shufflevector <8 x half> %nc.n983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n1011, <4 x float> %nc.n1010) #8, !dbg !25
  %nc.n1013 = shufflevector <8 x half> %nc.n985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n1013, <4 x float> %nc.n1012) #8, !dbg !25
  %nc.n1015 = shufflevector <8 x half> %nc.n987, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1016 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n1015, <4 x float> %nc.n1014) #8, !dbg !25
  %nc.n1017 = shufflevector <8 x half> %nc.n989, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n1017, <4 x float> %nc.n1016) #8, !dbg !25
  %nc.n1019 = shufflevector <8 x half> %nc.n991, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n1019, <4 x float> %nc.n1018) #8, !dbg !25
  %nc.n1021 = shufflevector <8 x half> %nc.n977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1022 = shufflevector <8 x float> %nc.n1003, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %nc.n1023 = fmul <4 x float> %nc.n1022, %nc.n580, !dbg !20
  %nc.n1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n1021, <4 x float> %nc.n1023) #8, !dbg !25
  %nc.n1025 = shufflevector <8 x half> %nc.n979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n1025, <4 x float> %nc.n1024) #8, !dbg !25
  %nc.n1027 = shufflevector <8 x half> %nc.n981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1028 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n1027, <4 x float> %nc.n1026) #8, !dbg !25
  %nc.n1029 = shufflevector <8 x half> %nc.n983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n1029, <4 x float> %nc.n1028) #8, !dbg !25
  %nc.n1031 = shufflevector <8 x half> %nc.n985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1032 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n1031, <4 x float> %nc.n1030) #8, !dbg !25
  %nc.n1033 = shufflevector <8 x half> %nc.n987, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1034 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n1033, <4 x float> %nc.n1032) #8, !dbg !25
  %nc.n1035 = shufflevector <8 x half> %nc.n989, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1036 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n1035, <4 x float> %nc.n1034) #8, !dbg !25
  %nc.n1037 = shufflevector <8 x half> %nc.n991, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1038 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n1037, <4 x float> %nc.n1036) #8, !dbg !25
  %nc.n1039 = extractvalue [2 x <8 x float>] %nc.n13116, 1, !dbg !20
  %nc.n1040 = shufflevector <8 x float> %nc.n1039, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %nc.n1041 = fmul <4 x float> %nc.n1040, %nc.n588, !dbg !20
  %nc.n1042 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n1002, <4 x float> %nc.n1041) #8, !dbg !25
  %nc.n1043 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n1007, <4 x float> %nc.n1042) #8, !dbg !25
  %nc.n1044 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n1009, <4 x float> %nc.n1043) #8, !dbg !25
  %nc.n1045 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n1011, <4 x float> %nc.n1044) #8, !dbg !25
  %nc.n1046 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n1013, <4 x float> %nc.n1045) #8, !dbg !25
  %nc.n1047 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n1015, <4 x float> %nc.n1046) #8, !dbg !25
  %nc.n1048 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n1017, <4 x float> %nc.n1047) #8, !dbg !25
  %nc.n1049 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n1019, <4 x float> %nc.n1048) #8, !dbg !25
  %nc.n1050 = shufflevector <8 x float> %nc.n1039, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %nc.n1051 = fmul <4 x float> %nc.n1050, %nc.n588, !dbg !20
  %nc.n1052 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n1021, <4 x float> %nc.n1051) #8, !dbg !25
  %nc.n1053 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n1025, <4 x float> %nc.n1052) #8, !dbg !25
  %nc.n1054 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n1027, <4 x float> %nc.n1053) #8, !dbg !25
  %nc.n1055 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n1029, <4 x float> %nc.n1054) #8, !dbg !25
  %nc.n1056 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n1031, <4 x float> %nc.n1055) #8, !dbg !25
  %nc.n1057 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n1033, <4 x float> %nc.n1056) #8, !dbg !25
  %nc.n1058 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n1035, <4 x float> %nc.n1057) #8, !dbg !25
  %nc.n1059 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n1037, <4 x float> %nc.n1058) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %nc.n1060 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #10, !dbg !25
  %nc.n1061 = bitcast <4 x i32> %nc.n1060 to <8 x half>, !dbg !25
  %nc.n1062 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #10, !dbg !25
  %nc.n1063 = bitcast <4 x i32> %nc.n1062 to <8 x half>, !dbg !25
  %nc.n1064 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #10, !dbg !25
  %nc.n1065 = bitcast <4 x i32> %nc.n1064 to <8 x half>, !dbg !25
  %nc.n1066 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !25
  %nc.n1067 = bitcast <4 x i32> %nc.n1066 to <8 x half>, !dbg !25
  %nc.n1068 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !25
  %nc.n1069 = bitcast <4 x i32> %nc.n1068 to <8 x half>, !dbg !25
  %nc.n1070 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !25
  %nc.n1071 = bitcast <4 x i32> %nc.n1070 to <8 x half>, !dbg !25
  %nc.n1072 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !25
  %nc.n1073 = bitcast <4 x i32> %nc.n1072 to <8 x half>, !dbg !25
  %nc.n1074 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #10, !dbg !25
  %nc.n1075 = bitcast <4 x i32> %nc.n1074 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %nc.n1076 = shufflevector <8 x half> %nc.n1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1077 = extractvalue [2 x <8 x float>] %nc.n13217, 0, !dbg !20
  %nc.n1078 = shufflevector <8 x float> %nc.n1077, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %nc.n1079 = fmul <4 x float> %nc.n1078, %nc.n580, !dbg !20
  %nc.n1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n1076, <4 x float> %nc.n1079) #8, !dbg !25
  %nc.n1081 = shufflevector <8 x half> %nc.n1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n1081, <4 x float> %nc.n1080) #8, !dbg !25
  %nc.n1083 = shufflevector <8 x half> %nc.n1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n1083, <4 x float> %nc.n1082) #8, !dbg !25
  %nc.n1085 = shufflevector <8 x half> %nc.n1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n1085, <4 x float> %nc.n1084) #8, !dbg !25
  %nc.n1087 = shufflevector <8 x half> %nc.n1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n1087, <4 x float> %nc.n1086) #8, !dbg !25
  %nc.n1089 = shufflevector <8 x half> %nc.n1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n1089, <4 x float> %nc.n1088) #8, !dbg !25
  %nc.n1091 = shufflevector <8 x half> %nc.n1073, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n1091, <4 x float> %nc.n1090) #8, !dbg !25
  %nc.n1093 = shufflevector <8 x half> %nc.n1075, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %nc.n1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n1093, <4 x float> %nc.n1092) #8, !dbg !25
  %nc.n1095 = shufflevector <8 x half> %nc.n1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1096 = shufflevector <8 x float> %nc.n1077, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %nc.n1097 = fmul <4 x float> %nc.n1096, %nc.n580, !dbg !20
  %nc.n1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %nc.n1095, <4 x float> %nc.n1097) #8, !dbg !25
  %nc.n1099 = shufflevector <8 x half> %nc.n1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %nc.n1099, <4 x float> %nc.n1098) #8, !dbg !25
  %nc.n1101 = shufflevector <8 x half> %nc.n1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %nc.n1101, <4 x float> %nc.n1100) #8, !dbg !25
  %nc.n1103 = shufflevector <8 x half> %nc.n1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %nc.n1103, <4 x float> %nc.n1102) #8, !dbg !25
  %nc.n1105 = shufflevector <8 x half> %nc.n1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %nc.n1105, <4 x float> %nc.n1104) #8, !dbg !25
  %nc.n1107 = shufflevector <8 x half> %nc.n1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1108 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %nc.n1107, <4 x float> %nc.n1106) #8, !dbg !25
  %nc.n1109 = shufflevector <8 x half> %nc.n1073, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1110 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %nc.n1109, <4 x float> %nc.n1108) #8, !dbg !25
  %nc.n1111 = shufflevector <8 x half> %nc.n1075, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %nc.n1112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %nc.n1111, <4 x float> %nc.n1110) #8, !dbg !25
  %nc.n1113 = extractvalue [2 x <8 x float>] %nc.n13217, 1, !dbg !20
  %nc.n1114 = shufflevector <8 x float> %nc.n1113, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %nc.n1115 = fmul <4 x float> %nc.n1114, %nc.n588, !dbg !20
  %nc.n1116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n1076, <4 x float> %nc.n1115) #8, !dbg !25
  %nc.n1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n1081, <4 x float> %nc.n1116) #8, !dbg !25
  %nc.n1118 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n1083, <4 x float> %nc.n1117) #8, !dbg !25
  %nc.n1119 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n1085, <4 x float> %nc.n1118) #8, !dbg !25
  %nc.n1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n1087, <4 x float> %nc.n1119) #8, !dbg !25
  %nc.n1121 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n1089, <4 x float> %nc.n1120) #8, !dbg !25
  %nc.n1122 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n1091, <4 x float> %nc.n1121) #8, !dbg !25
  %nc.n1123 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n1093, <4 x float> %nc.n1122) #8, !dbg !25
  %nc.n1124 = shufflevector <8 x float> %nc.n1113, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %nc.n1125 = fmul <4 x float> %nc.n1124, %nc.n588, !dbg !20
  %nc.n1126 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %nc.n1095, <4 x float> %nc.n1125) #8, !dbg !25
  %nc.n1127 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %nc.n1099, <4 x float> %nc.n1126) #8, !dbg !25
  %nc.n1128 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %nc.n1101, <4 x float> %nc.n1127) #8, !dbg !25
  %nc.n1129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %nc.n1103, <4 x float> %nc.n1128) #8, !dbg !25
  %nc.n1130 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %nc.n1105, <4 x float> %nc.n1129) #8, !dbg !25
  %nc.n1131 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %nc.n1107, <4 x float> %nc.n1130) #8, !dbg !25
  %nc.n1132 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %nc.n1109, <4 x float> %nc.n1131) #8, !dbg !25
  %nc.n1133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %nc.n1111, <4 x float> %nc.n1132) #8, !dbg !25
  %k0.has.next.not = icmp eq i64 %nc.n12611, 31
  br i1 %k0.has.next.not, label %nc.n1153, label %k0.join

k0.join:                                          ; preds = %nc.n134
  %nc.n1148 = shufflevector <4 x float> %nc.n1094, <4 x float> %nc.n1112, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %nc.n1149 = insertvalue [2 x <8 x float>] poison, <8 x float> %nc.n1148, 0, !dbg !11
  %nc.n1150 = shufflevector <4 x float> %nc.n1123, <4 x float> %nc.n1133, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %nc.n1151 = insertvalue [2 x <8 x float>] %nc.n1149, <8 x float> %nc.n1150, 1, !dbg !11
  %nc.n1144 = shufflevector <4 x float> %nc.n1020, <4 x float> %nc.n1038, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %nc.n1145 = insertvalue [2 x <8 x float>] poison, <8 x float> %nc.n1144, 0, !dbg !11
  %nc.n1146 = shufflevector <4 x float> %nc.n1049, <4 x float> %nc.n1059, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %nc.n1147 = insertvalue [2 x <8 x float>] %nc.n1145, <8 x float> %nc.n1146, 1, !dbg !11
  %nc.n1140 = shufflevector <4 x float> %nc.n936, <4 x float> %nc.n954, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %nc.n1141 = insertvalue [2 x <8 x float>] poison, <8 x float> %nc.n1140, 0, !dbg !11
  %nc.n1142 = shufflevector <4 x float> %nc.n965, <4 x float> %nc.n975, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %nc.n1143 = insertvalue [2 x <8 x float>] %nc.n1141, <8 x float> %nc.n1142, 1, !dbg !11
  %nc.n1138 = insertvalue [2 x <1 x float>] poison, <1 x float> %nc.n544, 0, !dbg !11
  %nc.n1139 = insertvalue [2 x <1 x float>] %nc.n1138, <1 x float> %nc.n557, 1, !dbg !11
  %nc.n1134 = shufflevector <4 x float> %nc.n852, <4 x float> %nc.n870, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %nc.n1135 = insertvalue [2 x <8 x float>] poison, <8 x float> %nc.n1134, 0, !dbg !11
  %nc.n1136 = shufflevector <4 x float> %nc.n881, <4 x float> %nc.n891, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %nc.n1137 = insertvalue [2 x <8 x float>] %nc.n1135, <8 x float> %nc.n1136, 1, !dbg !11
  %nc.n1152 = add nuw nsw i64 %nc.n12611, 1, !dbg !11
  %nc.n698 = insertelement <1 x float> poison, float %nc.n697, i64 0, !dbg !11
  %nc.n699 = insertvalue [2 x <1 x float>] poison, <1 x float> %nc.n698, 0, !dbg !11
  %nc.n806 = insertelement <1 x float> poison, float %nc.n805, i64 0, !dbg !11
  %nc.n807 = insertvalue [2 x <1 x float>] %nc.n699, <1 x float> %nc.n806, 1, !dbg !11
  %k0.next.address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %k0.next.address.local499 = zext nneg i32 %k0.next.address.thread498 to i64, !dbg !8
  %k0.next.address.local500 = shl nuw nsw i64 %k0.next.address.local499, 3, !dbg !8
  %k0.next.address.local501 = and i64 %k0.next.address.local500, 1536, !dbg !8
  %k0.next.address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.next.address.local501, !dbg !8
  %k0.next.address.local503 = ptrtoint ptr addrspace(3) %k0.next.address.local502 to i64, !dbg !8
  %k0.next.address.local504 = trunc i64 %k0.next.address.local503 to i32, !dbg !8
  %k0.next.nc.n139 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.address.local504), !dbg !13
  %k0.next.dma.buffer.lds1374 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.nc.n139), !dbg !13
  %k0.next.nc.n137 = shl nuw nsw i64 %nc.n1152, 14, !dbg !13
  %k0.next.nc.n138 = add nuw nsw i64 %k0.next.nc.n137, %k0.prime.nc.n136, !dbg !13
  %k0.next.address.local505 = lshr i64 %k0.next.address.local499, 4, !dbg !13
  %k0.next.address.local506 = and i64 %k0.next.address.local505, 8, !dbg !13
  %k0.next.address.local507 = and i64 %k0.next.address.local499, 123, !dbg !13
  %k0.next.address.local508 = lshr i64 %k0.next.address.local499, 2, !dbg !13
  %k0.next.address.local509 = and i64 %k0.next.address.local508, 4, !dbg !13
  %k0.next.address.local510 = add nuw nsw i64 %k0.next.address.local509, %k0.next.address.local499, !dbg !13
  %k0.next.address.local511 = and i64 %k0.next.address.local510, 4, !dbg !13
  %k0.next.address.local512 = or disjoint i64 %k0.next.address.local511, %k0.next.address.local507, !dbg !13
  %k0.next.address.local513 = shl nuw nsw i64 %k0.next.address.local512, 7, !dbg !13
  %k0.next.address.local514 = or disjoint i64 %k0.next.address.local506, %k0.next.nc.n138, !dbg !13
  %k0.next..idx1406 = or disjoint i64 %k0.next.address.local514, %k0.next.address.local513, !dbg !13
  %k0.next..idx1406.tr = trunc nuw nsw i64 %k0.next..idx1406 to i32, !dbg !13
  %k0.next.dma.buffer.offset1372 = shl nuw nsw i32 %k0.next..idx1406.tr, 1, !dbg !13
  %k0.next.nc.n143 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.next.dma.buffer.lds1374, i32 %k0.next.dma.buffer.offset1372, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %k0.next.address.local516 = or disjoint i64 %k0.next.address.local499, 256, !dbg !8
  %k0.next.address.local517 = shl nuw nsw i64 %k0.next.address.local516, 3, !dbg !8
  %k0.next.address.local518 = and i64 %k0.next.address.local517, 3584, !dbg !8
  %k0.next.address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.next.address.local518, !dbg !8
  %k0.next.address.local520 = ptrtoint ptr addrspace(3) %k0.next.address.local519 to i64, !dbg !8
  %k0.next.address.local521 = trunc i64 %k0.next.address.local520 to i32, !dbg !8
  %k0.next.nc.n144 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.address.local521), !dbg !13
  %k0.next.dma.buffer.lds1371 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.nc.n144), !dbg !13
  %k0.next.address.local522 = lshr i64 %k0.next.address.local516, 4, !dbg !13
  %k0.next.address.local523 = and i64 %k0.next.address.local522, 24, !dbg !13
  %k0.next.address.local531 = or disjoint i64 %k0.next.address.local523, %k0.next.nc.n138, !dbg !13
  %k0.next..idx1407 = or disjoint i64 %k0.next.address.local531, %k0.next.address.local513, !dbg !13
  %k0.next..idx1407.tr = trunc nuw nsw i64 %k0.next..idx1407 to i32, !dbg !13
  %k0.next.dma.buffer.offset1369 = shl nuw nsw i32 %k0.next..idx1407.tr, 1, !dbg !13
  %k0.next.nc.n148 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.next.dma.buffer.lds1371, i32 %k0.next.dma.buffer.offset1369, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  br label %nc.n134, !dbg !11

nc.n1153:                                         ; preds = %nc.n134
  %nc.n1154 = shl nuw nsw i32 %nc.n523, 2, !dbg !11
  %nc.n1155 = xor i32 %nc.n1154, 64, !dbg !11
  %nc.n1156 = bitcast float %nc.n805 to i32, !dbg !11
  %lane.sum.shuffle188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1155, i32 %nc.n1156), !dbg !11
  %nc.n1157 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !11
  %lane.sum.final189 = fadd float %nc.n805, %nc.n1157, !dbg !11
  %nc.n1159 = xor i32 %nc.n1154, 128, !dbg !11
  %nc.n1160 = bitcast float %lane.sum.final189 to i32, !dbg !11
  %lane.sum.shuffle190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1159, i32 %nc.n1160), !dbg !11
  %nc.n1161 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !11
  %lane.sum.final191 = fadd float %lane.sum.final189, %nc.n1161, !dbg !11
  %nc.n1162 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !11
  %nc.n1165 = bitcast float %nc.n697 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1155, i32 %nc.n1165), !dbg !11
  %nc.n1166 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %nc.n697, %nc.n1166, !dbg !11
  %nc.n1169 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n1159, i32 %nc.n1169), !dbg !11
  %nc.n1170 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !11
  %lane.sum.final183 = fadd float %lane.sum.final, %nc.n1170, !dbg !11
  %nc.n1171 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !11
  %nc.n1172 = and i32 %nc.n31, 16, !dbg !11
  %.not = icmp eq i32 %nc.n1172, 0, !dbg !11
  %nc.n1173 = and i32 %nc.n31, 32, !dbg !11
  %.not36 = icmp eq i32 %nc.n1173, 0, !dbg !11
  %nc.n1175 = extractelement <4 x float> %nc.n1094, i64 0, !dbg !11
  %nc.n1176 = extractelement <4 x float> %nc.n1094, i64 1, !dbg !11
  %nc.n1177 = extractelement <4 x float> %nc.n1094, i64 2, !dbg !11
  %nc.n1178 = extractelement <4 x float> %nc.n1094, i64 3, !dbg !11
  %nc.n1179 = select i1 %.not, float %nc.n1176, float %nc.n1175, !dbg !11
  %nc.n1188 = bitcast float %nc.n1179 to i32, !dbg !11
  %nc.n1189 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1188), !dbg !11
  %nc.n1190 = bitcast i32 %nc.n1189 to float, !dbg !11
  %nc.n1191 = select i1 %.not, float %nc.n1175, float %nc.n1190, !dbg !11
  %nc.n1192 = select i1 %.not, float %nc.n1190, float %nc.n1176, !dbg !11
  %nc.n1193 = select i1 %.not, float %nc.n1178, float %nc.n1177, !dbg !11
  %nc.n1194 = bitcast float %nc.n1193 to i32, !dbg !11
  %nc.n1195 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1194), !dbg !11
  %nc.n1196 = bitcast i32 %nc.n1195 to float, !dbg !11
  %nc.n1197 = select i1 %.not, float %nc.n1177, float %nc.n1196, !dbg !11
  %nc.n1198 = select i1 %.not, float %nc.n1196, float %nc.n1178, !dbg !11
  %nc.n1199 = select i1 %.not36, float %nc.n1197, float %nc.n1191, !dbg !11
  %nc.n1204 = bitcast float %nc.n1199 to i32, !dbg !11
  %nc.n1205 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1204), !dbg !11
  %nc.n1206 = bitcast i32 %nc.n1205 to float, !dbg !11
  %nc.n1209 = select i1 %.not36, float %nc.n1198, float %nc.n1192, !dbg !11
  %nc.n1210 = bitcast float %nc.n1209 to i32, !dbg !11
  %nc.n1211 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1210), !dbg !11
  %nc.n1212 = bitcast i32 %nc.n1211 to float, !dbg !11
  %29 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %30 = shufflevector <4 x i1> %29, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %31 = insertelement <4 x float> poison, float %nc.n1191, i64 0, !dbg !11
  %32 = insertelement <4 x float> %31, float %nc.n1192, i64 1, !dbg !11
  %33 = insertelement <4 x float> %32, float %nc.n1206, i64 2, !dbg !11
  %34 = insertelement <4 x float> %33, float %nc.n1212, i64 3, !dbg !11
  %35 = insertelement <4 x float> poison, float %nc.n1206, i64 0, !dbg !11
  %36 = insertelement <4 x float> %35, float %nc.n1212, i64 1, !dbg !11
  %37 = insertelement <4 x float> %36, float %nc.n1197, i64 2, !dbg !11
  %38 = insertelement <4 x float> %37, float %nc.n1198, i64 3, !dbg !11
  %39 = select <4 x i1> %30, <4 x float> %34, <4 x float> %38, !dbg !11
  %nc.n1216 = extractelement <4 x float> %nc.n1020, i64 0, !dbg !11
  %nc.n1217 = extractelement <4 x float> %nc.n1020, i64 1, !dbg !11
  %nc.n1218 = extractelement <4 x float> %nc.n1020, i64 2, !dbg !11
  %nc.n1219 = extractelement <4 x float> %nc.n1020, i64 3, !dbg !11
  %nc.n1220 = select i1 %.not, float %nc.n1217, float %nc.n1216, !dbg !11
  %nc.n1221 = bitcast float %nc.n1220 to i32, !dbg !11
  %nc.n1222 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1221), !dbg !11
  %nc.n1223 = bitcast i32 %nc.n1222 to float, !dbg !11
  %nc.n1224 = select i1 %.not, float %nc.n1216, float %nc.n1223, !dbg !11
  %nc.n1225 = select i1 %.not, float %nc.n1223, float %nc.n1217, !dbg !11
  %nc.n1226 = select i1 %.not, float %nc.n1219, float %nc.n1218, !dbg !11
  %nc.n1227 = bitcast float %nc.n1226 to i32, !dbg !11
  %nc.n1228 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1227), !dbg !11
  %nc.n1229 = bitcast i32 %nc.n1228 to float, !dbg !11
  %nc.n1230 = select i1 %.not, float %nc.n1218, float %nc.n1229, !dbg !11
  %nc.n1231 = select i1 %.not, float %nc.n1229, float %nc.n1219, !dbg !11
  %nc.n1232 = select i1 %.not36, float %nc.n1230, float %nc.n1224, !dbg !11
  %nc.n1233 = bitcast float %nc.n1232 to i32, !dbg !11
  %nc.n1234 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1233), !dbg !11
  %nc.n1235 = bitcast i32 %nc.n1234 to float, !dbg !11
  %nc.n1238 = select i1 %.not36, float %nc.n1231, float %nc.n1225, !dbg !11
  %nc.n1239 = bitcast float %nc.n1238 to i32, !dbg !11
  %nc.n1240 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1239), !dbg !11
  %nc.n1241 = bitcast i32 %nc.n1240 to float, !dbg !11
  %40 = insertelement <4 x float> poison, float %nc.n1224, i64 0, !dbg !11
  %41 = insertelement <4 x float> %40, float %nc.n1225, i64 1, !dbg !11
  %42 = insertelement <4 x float> %41, float %nc.n1235, i64 2, !dbg !11
  %43 = insertelement <4 x float> %42, float %nc.n1241, i64 3, !dbg !11
  %44 = insertelement <4 x float> poison, float %nc.n1235, i64 0, !dbg !11
  %45 = insertelement <4 x float> %44, float %nc.n1241, i64 1, !dbg !11
  %46 = insertelement <4 x float> %45, float %nc.n1230, i64 2, !dbg !11
  %47 = insertelement <4 x float> %46, float %nc.n1231, i64 3, !dbg !11
  %48 = select <4 x i1> %30, <4 x float> %43, <4 x float> %47, !dbg !11
  %nc.n1245 = extractelement <4 x float> %nc.n936, i64 0, !dbg !11
  %nc.n1246 = extractelement <4 x float> %nc.n936, i64 1, !dbg !11
  %nc.n1247 = extractelement <4 x float> %nc.n936, i64 2, !dbg !11
  %nc.n1248 = extractelement <4 x float> %nc.n936, i64 3, !dbg !11
  %nc.n1249 = select i1 %.not, float %nc.n1246, float %nc.n1245, !dbg !11
  %nc.n1250 = bitcast float %nc.n1249 to i32, !dbg !11
  %nc.n1251 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1250), !dbg !11
  %nc.n1252 = bitcast i32 %nc.n1251 to float, !dbg !11
  %nc.n1253 = select i1 %.not, float %nc.n1245, float %nc.n1252, !dbg !11
  %nc.n1254 = select i1 %.not, float %nc.n1252, float %nc.n1246, !dbg !11
  %nc.n1255 = select i1 %.not, float %nc.n1248, float %nc.n1247, !dbg !11
  %nc.n1256 = bitcast float %nc.n1255 to i32, !dbg !11
  %nc.n1257 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1256), !dbg !11
  %nc.n1258 = bitcast i32 %nc.n1257 to float, !dbg !11
  %nc.n1259 = select i1 %.not, float %nc.n1247, float %nc.n1258, !dbg !11
  %nc.n1260 = select i1 %.not, float %nc.n1258, float %nc.n1248, !dbg !11
  %nc.n1261 = select i1 %.not36, float %nc.n1259, float %nc.n1253, !dbg !11
  %nc.n1262 = bitcast float %nc.n1261 to i32, !dbg !11
  %nc.n1263 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1262), !dbg !11
  %nc.n1264 = bitcast i32 %nc.n1263 to float, !dbg !11
  %nc.n1267 = select i1 %.not36, float %nc.n1260, float %nc.n1254, !dbg !11
  %nc.n1268 = bitcast float %nc.n1267 to i32, !dbg !11
  %nc.n1269 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1268), !dbg !11
  %nc.n1270 = bitcast i32 %nc.n1269 to float, !dbg !11
  %49 = insertelement <4 x float> poison, float %nc.n1253, i64 0, !dbg !11
  %50 = insertelement <4 x float> %49, float %nc.n1254, i64 1, !dbg !11
  %51 = insertelement <4 x float> %50, float %nc.n1264, i64 2, !dbg !11
  %52 = insertelement <4 x float> %51, float %nc.n1270, i64 3, !dbg !11
  %53 = insertelement <4 x float> poison, float %nc.n1264, i64 0, !dbg !11
  %54 = insertelement <4 x float> %53, float %nc.n1270, i64 1, !dbg !11
  %55 = insertelement <4 x float> %54, float %nc.n1259, i64 2, !dbg !11
  %56 = insertelement <4 x float> %55, float %nc.n1260, i64 3, !dbg !11
  %57 = select <4 x i1> %30, <4 x float> %52, <4 x float> %56, !dbg !11
  %nc.n1274 = extractelement <4 x float> %nc.n852, i64 0, !dbg !11
  %nc.n1275 = extractelement <4 x float> %nc.n852, i64 1, !dbg !11
  %nc.n1276 = extractelement <4 x float> %nc.n852, i64 2, !dbg !11
  %nc.n1277 = extractelement <4 x float> %nc.n852, i64 3, !dbg !11
  %nc.n1278 = select i1 %.not, float %nc.n1275, float %nc.n1274, !dbg !11
  %nc.n1279 = bitcast float %nc.n1278 to i32, !dbg !11
  %nc.n1280 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1279), !dbg !11
  %nc.n1281 = bitcast i32 %nc.n1280 to float, !dbg !11
  %nc.n1282 = select i1 %.not, float %nc.n1274, float %nc.n1281, !dbg !11
  %nc.n1283 = select i1 %.not, float %nc.n1281, float %nc.n1275, !dbg !11
  %nc.n1284 = select i1 %.not, float %nc.n1277, float %nc.n1276, !dbg !11
  %nc.n1285 = bitcast float %nc.n1284 to i32, !dbg !11
  %nc.n1286 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1285), !dbg !11
  %nc.n1287 = bitcast i32 %nc.n1286 to float, !dbg !11
  %nc.n1288 = select i1 %.not, float %nc.n1276, float %nc.n1287, !dbg !11
  %nc.n1289 = select i1 %.not, float %nc.n1287, float %nc.n1277, !dbg !11
  %nc.n1290 = select i1 %.not36, float %nc.n1288, float %nc.n1282, !dbg !11
  %nc.n1291 = bitcast float %nc.n1290 to i32, !dbg !11
  %nc.n1292 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1291), !dbg !11
  %nc.n1293 = select i1 %.not36, float %nc.n1289, float %nc.n1283, !dbg !11
  %nc.n1294 = bitcast float %nc.n1293 to i32, !dbg !11
  %nc.n1295 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1294), !dbg !11
  %nc.n1296 = shufflevector <1 x float> %nc.n1171, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %nc.n1297 = extractelement <4 x float> %nc.n1112, i64 0, !dbg !11
  %nc.n1298 = extractelement <4 x float> %nc.n1112, i64 1, !dbg !11
  %nc.n1299 = extractelement <4 x float> %nc.n1112, i64 2, !dbg !11
  %nc.n1300 = extractelement <4 x float> %nc.n1112, i64 3, !dbg !11
  %nc.n1301 = select i1 %.not, float %nc.n1298, float %nc.n1297, !dbg !11
  %nc.n1302 = bitcast float %nc.n1301 to i32, !dbg !11
  %nc.n1303 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1302), !dbg !11
  %nc.n1304 = bitcast i32 %nc.n1303 to float, !dbg !11
  %nc.n1305 = select i1 %.not, float %nc.n1297, float %nc.n1304, !dbg !11
  %nc.n1306 = select i1 %.not, float %nc.n1304, float %nc.n1298, !dbg !11
  %nc.n1307 = select i1 %.not, float %nc.n1300, float %nc.n1299, !dbg !11
  %nc.n1308 = bitcast float %nc.n1307 to i32, !dbg !11
  %nc.n1309 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1308), !dbg !11
  %nc.n1310 = bitcast i32 %nc.n1309 to float, !dbg !11
  %nc.n1311 = select i1 %.not, float %nc.n1299, float %nc.n1310, !dbg !11
  %nc.n1312 = select i1 %.not, float %nc.n1310, float %nc.n1300, !dbg !11
  %nc.n1313 = select i1 %.not36, float %nc.n1311, float %nc.n1305, !dbg !11
  %nc.n1314 = bitcast float %nc.n1313 to i32, !dbg !11
  %nc.n1315 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1314), !dbg !11
  %nc.n1316 = bitcast i32 %nc.n1315 to float, !dbg !11
  %nc.n1319 = select i1 %.not36, float %nc.n1312, float %nc.n1306, !dbg !11
  %nc.n1320 = bitcast float %nc.n1319 to i32, !dbg !11
  %nc.n1321 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1320), !dbg !11
  %nc.n1322 = bitcast i32 %nc.n1321 to float, !dbg !11
  %58 = insertelement <4 x float> poison, float %nc.n1305, i64 0, !dbg !11
  %59 = insertelement <4 x float> %58, float %nc.n1306, i64 1, !dbg !11
  %60 = insertelement <4 x float> %59, float %nc.n1316, i64 2, !dbg !11
  %61 = insertelement <4 x float> %60, float %nc.n1322, i64 3, !dbg !11
  %62 = insertelement <4 x float> poison, float %nc.n1316, i64 0, !dbg !11
  %63 = insertelement <4 x float> %62, float %nc.n1322, i64 1, !dbg !11
  %64 = insertelement <4 x float> %63, float %nc.n1311, i64 2, !dbg !11
  %65 = insertelement <4 x float> %64, float %nc.n1312, i64 3, !dbg !11
  %66 = select <4 x i1> %30, <4 x float> %61, <4 x float> %65, !dbg !11
  %nc.n1325 = extractelement <4 x float> %nc.n1038, i64 0, !dbg !11
  %nc.n1326 = extractelement <4 x float> %nc.n1038, i64 1, !dbg !11
  %nc.n1327 = extractelement <4 x float> %nc.n1038, i64 2, !dbg !11
  %nc.n1328 = extractelement <4 x float> %nc.n1038, i64 3, !dbg !11
  %nc.n1329 = select i1 %.not, float %nc.n1326, float %nc.n1325, !dbg !11
  %nc.n1330 = bitcast float %nc.n1329 to i32, !dbg !11
  %nc.n1331 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1330), !dbg !11
  %nc.n1332 = bitcast i32 %nc.n1331 to float, !dbg !11
  %nc.n1333 = select i1 %.not, float %nc.n1325, float %nc.n1332, !dbg !11
  %nc.n1334 = select i1 %.not, float %nc.n1332, float %nc.n1326, !dbg !11
  %nc.n1335 = select i1 %.not, float %nc.n1328, float %nc.n1327, !dbg !11
  %nc.n1336 = bitcast float %nc.n1335 to i32, !dbg !11
  %nc.n1337 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1336), !dbg !11
  %nc.n1338 = bitcast i32 %nc.n1337 to float, !dbg !11
  %nc.n1339 = select i1 %.not, float %nc.n1327, float %nc.n1338, !dbg !11
  %nc.n1340 = select i1 %.not, float %nc.n1338, float %nc.n1328, !dbg !11
  %nc.n1341 = select i1 %.not36, float %nc.n1339, float %nc.n1333, !dbg !11
  %nc.n1342 = bitcast float %nc.n1341 to i32, !dbg !11
  %nc.n1343 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1342), !dbg !11
  %nc.n1344 = bitcast i32 %nc.n1343 to float, !dbg !11
  %nc.n1347 = select i1 %.not36, float %nc.n1340, float %nc.n1334, !dbg !11
  %nc.n1348 = bitcast float %nc.n1347 to i32, !dbg !11
  %nc.n1349 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1348), !dbg !11
  %nc.n1350 = bitcast i32 %nc.n1349 to float, !dbg !11
  %67 = insertelement <4 x float> poison, float %nc.n1333, i64 0, !dbg !11
  %68 = insertelement <4 x float> %67, float %nc.n1334, i64 1, !dbg !11
  %69 = insertelement <4 x float> %68, float %nc.n1344, i64 2, !dbg !11
  %70 = insertelement <4 x float> %69, float %nc.n1350, i64 3, !dbg !11
  %71 = insertelement <4 x float> poison, float %nc.n1344, i64 0, !dbg !11
  %72 = insertelement <4 x float> %71, float %nc.n1350, i64 1, !dbg !11
  %73 = insertelement <4 x float> %72, float %nc.n1339, i64 2, !dbg !11
  %74 = insertelement <4 x float> %73, float %nc.n1340, i64 3, !dbg !11
  %75 = select <4 x i1> %30, <4 x float> %70, <4 x float> %74, !dbg !11
  %nc.n1353 = extractelement <4 x float> %nc.n954, i64 0, !dbg !11
  %nc.n1354 = extractelement <4 x float> %nc.n954, i64 1, !dbg !11
  %nc.n1355 = extractelement <4 x float> %nc.n954, i64 2, !dbg !11
  %nc.n1356 = extractelement <4 x float> %nc.n954, i64 3, !dbg !11
  %nc.n1357 = select i1 %.not, float %nc.n1354, float %nc.n1353, !dbg !11
  %nc.n1358 = bitcast float %nc.n1357 to i32, !dbg !11
  %nc.n1359 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1358), !dbg !11
  %nc.n1360 = bitcast i32 %nc.n1359 to float, !dbg !11
  %nc.n1361 = select i1 %.not, float %nc.n1353, float %nc.n1360, !dbg !11
  %nc.n1362 = select i1 %.not, float %nc.n1360, float %nc.n1354, !dbg !11
  %nc.n1363 = select i1 %.not, float %nc.n1356, float %nc.n1355, !dbg !11
  %nc.n1364 = bitcast float %nc.n1363 to i32, !dbg !11
  %nc.n1365 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1364), !dbg !11
  %nc.n1366 = bitcast i32 %nc.n1365 to float, !dbg !11
  %nc.n1367 = select i1 %.not, float %nc.n1355, float %nc.n1366, !dbg !11
  %nc.n1368 = select i1 %.not, float %nc.n1366, float %nc.n1356, !dbg !11
  %nc.n1369 = select i1 %.not36, float %nc.n1367, float %nc.n1361, !dbg !11
  %nc.n1370 = bitcast float %nc.n1369 to i32, !dbg !11
  %nc.n1371 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1370), !dbg !11
  %nc.n1372 = bitcast i32 %nc.n1371 to float, !dbg !11
  %nc.n1375 = select i1 %.not36, float %nc.n1368, float %nc.n1362, !dbg !11
  %nc.n1376 = bitcast float %nc.n1375 to i32, !dbg !11
  %nc.n1377 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1376), !dbg !11
  %nc.n1378 = bitcast i32 %nc.n1377 to float, !dbg !11
  %76 = insertelement <4 x float> poison, float %nc.n1361, i64 0, !dbg !11
  %77 = insertelement <4 x float> %76, float %nc.n1362, i64 1, !dbg !11
  %78 = insertelement <4 x float> %77, float %nc.n1372, i64 2, !dbg !11
  %79 = insertelement <4 x float> %78, float %nc.n1378, i64 3, !dbg !11
  %80 = insertelement <4 x float> poison, float %nc.n1372, i64 0, !dbg !11
  %81 = insertelement <4 x float> %80, float %nc.n1378, i64 1, !dbg !11
  %82 = insertelement <4 x float> %81, float %nc.n1367, i64 2, !dbg !11
  %83 = insertelement <4 x float> %82, float %nc.n1368, i64 3, !dbg !11
  %84 = select <4 x i1> %30, <4 x float> %79, <4 x float> %83, !dbg !11
  %nc.n1381 = extractelement <4 x float> %nc.n870, i64 0, !dbg !11
  %nc.n1382 = extractelement <4 x float> %nc.n870, i64 1, !dbg !11
  %nc.n1383 = extractelement <4 x float> %nc.n870, i64 2, !dbg !11
  %nc.n1384 = extractelement <4 x float> %nc.n870, i64 3, !dbg !11
  %nc.n1385 = select i1 %.not, float %nc.n1382, float %nc.n1381, !dbg !11
  %nc.n1386 = bitcast float %nc.n1385 to i32, !dbg !11
  %nc.n1387 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1386), !dbg !11
  %nc.n1388 = bitcast i32 %nc.n1387 to float, !dbg !11
  %nc.n1389 = select i1 %.not, float %nc.n1381, float %nc.n1388, !dbg !11
  %nc.n1390 = select i1 %.not, float %nc.n1388, float %nc.n1382, !dbg !11
  %nc.n1391 = select i1 %.not, float %nc.n1384, float %nc.n1383, !dbg !11
  %nc.n1392 = bitcast float %nc.n1391 to i32, !dbg !11
  %nc.n1393 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1392), !dbg !11
  %nc.n1394 = bitcast i32 %nc.n1393 to float, !dbg !11
  %nc.n1395 = select i1 %.not, float %nc.n1383, float %nc.n1394, !dbg !11
  %nc.n1396 = select i1 %.not, float %nc.n1394, float %nc.n1384, !dbg !11
  %nc.n1397 = select i1 %.not36, float %nc.n1395, float %nc.n1389, !dbg !11
  %nc.n1398 = bitcast float %nc.n1397 to i32, !dbg !11
  %nc.n1399 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1398), !dbg !11
  %nc.n1400 = select i1 %.not36, float %nc.n1396, float %nc.n1390, !dbg !11
  %nc.n1401 = bitcast float %nc.n1400 to i32, !dbg !11
  %nc.n1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1401), !dbg !11
  %nc.n1404 = extractelement <4 x float> %nc.n1123, i64 0, !dbg !11
  %nc.n1405 = extractelement <4 x float> %nc.n1123, i64 1, !dbg !11
  %nc.n1406 = extractelement <4 x float> %nc.n1123, i64 2, !dbg !11
  %nc.n1407 = extractelement <4 x float> %nc.n1123, i64 3, !dbg !11
  %nc.n1408 = select i1 %.not, float %nc.n1405, float %nc.n1404, !dbg !11
  %nc.n1409 = bitcast float %nc.n1408 to i32, !dbg !11
  %nc.n1410 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1409), !dbg !11
  %nc.n1411 = bitcast i32 %nc.n1410 to float, !dbg !11
  %nc.n1412 = select i1 %.not, float %nc.n1404, float %nc.n1411, !dbg !11
  %nc.n1413 = select i1 %.not, float %nc.n1411, float %nc.n1405, !dbg !11
  %nc.n1414 = select i1 %.not, float %nc.n1407, float %nc.n1406, !dbg !11
  %nc.n1415 = bitcast float %nc.n1414 to i32, !dbg !11
  %nc.n1416 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1415), !dbg !11
  %nc.n1417 = bitcast i32 %nc.n1416 to float, !dbg !11
  %nc.n1418 = select i1 %.not, float %nc.n1406, float %nc.n1417, !dbg !11
  %nc.n1419 = select i1 %.not, float %nc.n1417, float %nc.n1407, !dbg !11
  %nc.n1420 = select i1 %.not36, float %nc.n1418, float %nc.n1412, !dbg !11
  %nc.n1421 = bitcast float %nc.n1420 to i32, !dbg !11
  %nc.n1422 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1421), !dbg !11
  %nc.n1423 = bitcast i32 %nc.n1422 to float, !dbg !11
  %nc.n1426 = select i1 %.not36, float %nc.n1419, float %nc.n1413, !dbg !11
  %nc.n1427 = bitcast float %nc.n1426 to i32, !dbg !11
  %nc.n1428 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1427), !dbg !11
  %nc.n1429 = bitcast i32 %nc.n1428 to float, !dbg !11
  %85 = insertelement <4 x float> poison, float %nc.n1412, i64 0, !dbg !11
  %86 = insertelement <4 x float> %85, float %nc.n1413, i64 1, !dbg !11
  %87 = insertelement <4 x float> %86, float %nc.n1423, i64 2, !dbg !11
  %88 = insertelement <4 x float> %87, float %nc.n1429, i64 3, !dbg !11
  %89 = insertelement <4 x float> poison, float %nc.n1423, i64 0, !dbg !11
  %90 = insertelement <4 x float> %89, float %nc.n1429, i64 1, !dbg !11
  %91 = insertelement <4 x float> %90, float %nc.n1418, i64 2, !dbg !11
  %92 = insertelement <4 x float> %91, float %nc.n1419, i64 3, !dbg !11
  %93 = select <4 x i1> %30, <4 x float> %88, <4 x float> %92, !dbg !11
  %nc.n1433 = extractelement <4 x float> %nc.n1049, i64 0, !dbg !11
  %nc.n1434 = extractelement <4 x float> %nc.n1049, i64 1, !dbg !11
  %nc.n1435 = extractelement <4 x float> %nc.n1049, i64 2, !dbg !11
  %nc.n1436 = extractelement <4 x float> %nc.n1049, i64 3, !dbg !11
  %nc.n1437 = select i1 %.not, float %nc.n1434, float %nc.n1433, !dbg !11
  %nc.n1438 = bitcast float %nc.n1437 to i32, !dbg !11
  %nc.n1439 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1438), !dbg !11
  %nc.n1440 = bitcast i32 %nc.n1439 to float, !dbg !11
  %nc.n1441 = select i1 %.not, float %nc.n1433, float %nc.n1440, !dbg !11
  %nc.n1442 = select i1 %.not, float %nc.n1440, float %nc.n1434, !dbg !11
  %nc.n1443 = select i1 %.not, float %nc.n1436, float %nc.n1435, !dbg !11
  %nc.n1444 = bitcast float %nc.n1443 to i32, !dbg !11
  %nc.n1445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1444), !dbg !11
  %nc.n1446 = bitcast i32 %nc.n1445 to float, !dbg !11
  %nc.n1447 = select i1 %.not, float %nc.n1435, float %nc.n1446, !dbg !11
  %nc.n1448 = select i1 %.not, float %nc.n1446, float %nc.n1436, !dbg !11
  %nc.n1449 = select i1 %.not36, float %nc.n1447, float %nc.n1441, !dbg !11
  %nc.n1450 = bitcast float %nc.n1449 to i32, !dbg !11
  %nc.n1451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1450), !dbg !11
  %nc.n1452 = bitcast i32 %nc.n1451 to float, !dbg !11
  %nc.n1455 = select i1 %.not36, float %nc.n1448, float %nc.n1442, !dbg !11
  %nc.n1456 = bitcast float %nc.n1455 to i32, !dbg !11
  %nc.n1457 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1456), !dbg !11
  %nc.n1458 = bitcast i32 %nc.n1457 to float, !dbg !11
  %94 = insertelement <4 x float> poison, float %nc.n1441, i64 0, !dbg !11
  %95 = insertelement <4 x float> %94, float %nc.n1442, i64 1, !dbg !11
  %96 = insertelement <4 x float> %95, float %nc.n1452, i64 2, !dbg !11
  %97 = insertelement <4 x float> %96, float %nc.n1458, i64 3, !dbg !11
  %98 = insertelement <4 x float> poison, float %nc.n1452, i64 0, !dbg !11
  %99 = insertelement <4 x float> %98, float %nc.n1458, i64 1, !dbg !11
  %100 = insertelement <4 x float> %99, float %nc.n1447, i64 2, !dbg !11
  %101 = insertelement <4 x float> %100, float %nc.n1448, i64 3, !dbg !11
  %102 = select <4 x i1> %30, <4 x float> %97, <4 x float> %101, !dbg !11
  %nc.n1462 = extractelement <4 x float> %nc.n965, i64 0, !dbg !11
  %nc.n1463 = extractelement <4 x float> %nc.n965, i64 1, !dbg !11
  %nc.n1464 = extractelement <4 x float> %nc.n965, i64 2, !dbg !11
  %nc.n1465 = extractelement <4 x float> %nc.n965, i64 3, !dbg !11
  %nc.n1466 = select i1 %.not, float %nc.n1463, float %nc.n1462, !dbg !11
  %nc.n1467 = bitcast float %nc.n1466 to i32, !dbg !11
  %nc.n1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1467), !dbg !11
  %nc.n1469 = bitcast i32 %nc.n1468 to float, !dbg !11
  %nc.n1470 = select i1 %.not, float %nc.n1462, float %nc.n1469, !dbg !11
  %nc.n1471 = select i1 %.not, float %nc.n1469, float %nc.n1463, !dbg !11
  %nc.n1472 = select i1 %.not, float %nc.n1465, float %nc.n1464, !dbg !11
  %nc.n1473 = bitcast float %nc.n1472 to i32, !dbg !11
  %nc.n1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1473), !dbg !11
  %nc.n1475 = bitcast i32 %nc.n1474 to float, !dbg !11
  %nc.n1476 = select i1 %.not, float %nc.n1464, float %nc.n1475, !dbg !11
  %nc.n1477 = select i1 %.not, float %nc.n1475, float %nc.n1465, !dbg !11
  %nc.n1478 = select i1 %.not36, float %nc.n1476, float %nc.n1470, !dbg !11
  %nc.n1479 = bitcast float %nc.n1478 to i32, !dbg !11
  %nc.n1480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1479), !dbg !11
  %nc.n1481 = bitcast i32 %nc.n1480 to float, !dbg !11
  %nc.n1484 = select i1 %.not36, float %nc.n1477, float %nc.n1471, !dbg !11
  %nc.n1485 = bitcast float %nc.n1484 to i32, !dbg !11
  %nc.n1486 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1485), !dbg !11
  %nc.n1487 = bitcast i32 %nc.n1486 to float, !dbg !11
  %103 = insertelement <4 x float> poison, float %nc.n1470, i64 0, !dbg !11
  %104 = insertelement <4 x float> %103, float %nc.n1471, i64 1, !dbg !11
  %105 = insertelement <4 x float> %104, float %nc.n1481, i64 2, !dbg !11
  %106 = insertelement <4 x float> %105, float %nc.n1487, i64 3, !dbg !11
  %107 = insertelement <4 x float> poison, float %nc.n1481, i64 0, !dbg !11
  %108 = insertelement <4 x float> %107, float %nc.n1487, i64 1, !dbg !11
  %109 = insertelement <4 x float> %108, float %nc.n1476, i64 2, !dbg !11
  %110 = insertelement <4 x float> %109, float %nc.n1477, i64 3, !dbg !11
  %111 = select <4 x i1> %30, <4 x float> %106, <4 x float> %110, !dbg !11
  %nc.n1491 = extractelement <4 x float> %nc.n881, i64 0, !dbg !11
  %nc.n1492 = extractelement <4 x float> %nc.n881, i64 1, !dbg !11
  %nc.n1493 = extractelement <4 x float> %nc.n881, i64 2, !dbg !11
  %nc.n1494 = extractelement <4 x float> %nc.n881, i64 3, !dbg !11
  %nc.n1495 = select i1 %.not, float %nc.n1492, float %nc.n1491, !dbg !11
  %nc.n1496 = bitcast float %nc.n1495 to i32, !dbg !11
  %nc.n1497 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1496), !dbg !11
  %nc.n1498 = bitcast i32 %nc.n1497 to float, !dbg !11
  %nc.n1499 = select i1 %.not, float %nc.n1491, float %nc.n1498, !dbg !11
  %nc.n1500 = select i1 %.not, float %nc.n1498, float %nc.n1492, !dbg !11
  %nc.n1501 = select i1 %.not, float %nc.n1494, float %nc.n1493, !dbg !11
  %nc.n1502 = bitcast float %nc.n1501 to i32, !dbg !11
  %nc.n1503 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1502), !dbg !11
  %nc.n1504 = bitcast i32 %nc.n1503 to float, !dbg !11
  %nc.n1505 = select i1 %.not, float %nc.n1493, float %nc.n1504, !dbg !11
  %nc.n1506 = select i1 %.not, float %nc.n1504, float %nc.n1494, !dbg !11
  %nc.n1507 = select i1 %.not36, float %nc.n1505, float %nc.n1499, !dbg !11
  %nc.n1508 = bitcast float %nc.n1507 to i32, !dbg !11
  %nc.n1509 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1508), !dbg !11
  %nc.n1510 = select i1 %.not36, float %nc.n1506, float %nc.n1500, !dbg !11
  %nc.n1511 = bitcast float %nc.n1510 to i32, !dbg !11
  %nc.n1512 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1511), !dbg !11
  %nc.n1513 = shufflevector <1 x float> %nc.n1162, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %nc.n1514 = extractelement <4 x float> %nc.n1133, i64 0, !dbg !11
  %nc.n1515 = extractelement <4 x float> %nc.n1133, i64 1, !dbg !11
  %nc.n1516 = extractelement <4 x float> %nc.n1133, i64 2, !dbg !11
  %nc.n1517 = extractelement <4 x float> %nc.n1133, i64 3, !dbg !11
  %nc.n1518 = select i1 %.not, float %nc.n1515, float %nc.n1514, !dbg !11
  %nc.n1519 = bitcast float %nc.n1518 to i32, !dbg !11
  %nc.n1520 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1519), !dbg !11
  %nc.n1521 = bitcast i32 %nc.n1520 to float, !dbg !11
  %nc.n1522 = select i1 %.not, float %nc.n1514, float %nc.n1521, !dbg !11
  %nc.n1523 = select i1 %.not, float %nc.n1521, float %nc.n1515, !dbg !11
  %nc.n1524 = select i1 %.not, float %nc.n1517, float %nc.n1516, !dbg !11
  %nc.n1525 = bitcast float %nc.n1524 to i32, !dbg !11
  %nc.n1526 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1525), !dbg !11
  %nc.n1527 = bitcast i32 %nc.n1526 to float, !dbg !11
  %nc.n1528 = select i1 %.not, float %nc.n1516, float %nc.n1527, !dbg !11
  %nc.n1529 = select i1 %.not, float %nc.n1527, float %nc.n1517, !dbg !11
  %nc.n1530 = select i1 %.not36, float %nc.n1528, float %nc.n1522, !dbg !11
  %nc.n1531 = bitcast float %nc.n1530 to i32, !dbg !11
  %nc.n1532 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1531), !dbg !11
  %nc.n1533 = bitcast i32 %nc.n1532 to float, !dbg !11
  %nc.n1536 = select i1 %.not36, float %nc.n1529, float %nc.n1523, !dbg !11
  %nc.n1537 = bitcast float %nc.n1536 to i32, !dbg !11
  %nc.n1538 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1537), !dbg !11
  %nc.n1539 = bitcast i32 %nc.n1538 to float, !dbg !11
  %112 = insertelement <4 x float> poison, float %nc.n1522, i64 0, !dbg !11
  %113 = insertelement <4 x float> %112, float %nc.n1523, i64 1, !dbg !11
  %114 = insertelement <4 x float> %113, float %nc.n1533, i64 2, !dbg !11
  %115 = insertelement <4 x float> %114, float %nc.n1539, i64 3, !dbg !11
  %116 = insertelement <4 x float> poison, float %nc.n1533, i64 0, !dbg !11
  %117 = insertelement <4 x float> %116, float %nc.n1539, i64 1, !dbg !11
  %118 = insertelement <4 x float> %117, float %nc.n1528, i64 2, !dbg !11
  %119 = insertelement <4 x float> %118, float %nc.n1529, i64 3, !dbg !11
  %120 = select <4 x i1> %30, <4 x float> %115, <4 x float> %119, !dbg !11
  %nc.n1542 = extractelement <4 x float> %nc.n1059, i64 0, !dbg !11
  %nc.n1543 = extractelement <4 x float> %nc.n1059, i64 1, !dbg !11
  %nc.n1544 = extractelement <4 x float> %nc.n1059, i64 2, !dbg !11
  %nc.n1545 = extractelement <4 x float> %nc.n1059, i64 3, !dbg !11
  %nc.n1546 = select i1 %.not, float %nc.n1543, float %nc.n1542, !dbg !11
  %nc.n1547 = bitcast float %nc.n1546 to i32, !dbg !11
  %nc.n1548 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1547), !dbg !11
  %nc.n1549 = bitcast i32 %nc.n1548 to float, !dbg !11
  %nc.n1550 = select i1 %.not, float %nc.n1542, float %nc.n1549, !dbg !11
  %nc.n1551 = select i1 %.not, float %nc.n1549, float %nc.n1543, !dbg !11
  %nc.n1552 = select i1 %.not, float %nc.n1545, float %nc.n1544, !dbg !11
  %nc.n1553 = bitcast float %nc.n1552 to i32, !dbg !11
  %nc.n1554 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1553), !dbg !11
  %nc.n1555 = bitcast i32 %nc.n1554 to float, !dbg !11
  %nc.n1556 = select i1 %.not, float %nc.n1544, float %nc.n1555, !dbg !11
  %nc.n1557 = select i1 %.not, float %nc.n1555, float %nc.n1545, !dbg !11
  %nc.n1558 = select i1 %.not36, float %nc.n1556, float %nc.n1550, !dbg !11
  %nc.n1559 = bitcast float %nc.n1558 to i32, !dbg !11
  %nc.n1560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1559), !dbg !11
  %nc.n1561 = bitcast i32 %nc.n1560 to float, !dbg !11
  %nc.n1564 = select i1 %.not36, float %nc.n1557, float %nc.n1551, !dbg !11
  %nc.n1565 = bitcast float %nc.n1564 to i32, !dbg !11
  %nc.n1566 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1565), !dbg !11
  %nc.n1567 = bitcast i32 %nc.n1566 to float, !dbg !11
  %121 = insertelement <4 x float> poison, float %nc.n1550, i64 0, !dbg !11
  %122 = insertelement <4 x float> %121, float %nc.n1551, i64 1, !dbg !11
  %123 = insertelement <4 x float> %122, float %nc.n1561, i64 2, !dbg !11
  %124 = insertelement <4 x float> %123, float %nc.n1567, i64 3, !dbg !11
  %125 = insertelement <4 x float> poison, float %nc.n1561, i64 0, !dbg !11
  %126 = insertelement <4 x float> %125, float %nc.n1567, i64 1, !dbg !11
  %127 = insertelement <4 x float> %126, float %nc.n1556, i64 2, !dbg !11
  %128 = insertelement <4 x float> %127, float %nc.n1557, i64 3, !dbg !11
  %129 = select <4 x i1> %30, <4 x float> %124, <4 x float> %128, !dbg !11
  %nc.n1570 = extractelement <4 x float> %nc.n975, i64 0, !dbg !11
  %nc.n1571 = extractelement <4 x float> %nc.n975, i64 1, !dbg !11
  %nc.n1572 = extractelement <4 x float> %nc.n975, i64 2, !dbg !11
  %nc.n1573 = extractelement <4 x float> %nc.n975, i64 3, !dbg !11
  %nc.n1574 = select i1 %.not, float %nc.n1571, float %nc.n1570, !dbg !11
  %nc.n1575 = bitcast float %nc.n1574 to i32, !dbg !11
  %nc.n1576 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1575), !dbg !11
  %nc.n1577 = bitcast i32 %nc.n1576 to float, !dbg !11
  %nc.n1578 = select i1 %.not, float %nc.n1570, float %nc.n1577, !dbg !11
  %nc.n1579 = select i1 %.not, float %nc.n1577, float %nc.n1571, !dbg !11
  %nc.n1580 = select i1 %.not, float %nc.n1573, float %nc.n1572, !dbg !11
  %nc.n1581 = bitcast float %nc.n1580 to i32, !dbg !11
  %nc.n1582 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1581), !dbg !11
  %nc.n1583 = bitcast i32 %nc.n1582 to float, !dbg !11
  %nc.n1584 = select i1 %.not, float %nc.n1572, float %nc.n1583, !dbg !11
  %nc.n1585 = select i1 %.not, float %nc.n1583, float %nc.n1573, !dbg !11
  %nc.n1586 = select i1 %.not36, float %nc.n1584, float %nc.n1578, !dbg !11
  %nc.n1587 = bitcast float %nc.n1586 to i32, !dbg !11
  %nc.n1588 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1587), !dbg !11
  %nc.n1589 = bitcast i32 %nc.n1588 to float, !dbg !11
  %nc.n1592 = select i1 %.not36, float %nc.n1585, float %nc.n1579, !dbg !11
  %nc.n1593 = bitcast float %nc.n1592 to i32, !dbg !11
  %nc.n1594 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1593), !dbg !11
  %nc.n1595 = bitcast i32 %nc.n1594 to float, !dbg !11
  %130 = insertelement <4 x float> poison, float %nc.n1578, i64 0, !dbg !11
  %131 = insertelement <4 x float> %130, float %nc.n1579, i64 1, !dbg !11
  %132 = insertelement <4 x float> %131, float %nc.n1589, i64 2, !dbg !11
  %133 = insertelement <4 x float> %132, float %nc.n1595, i64 3, !dbg !11
  %134 = insertelement <4 x float> poison, float %nc.n1589, i64 0, !dbg !11
  %135 = insertelement <4 x float> %134, float %nc.n1595, i64 1, !dbg !11
  %136 = insertelement <4 x float> %135, float %nc.n1584, i64 2, !dbg !11
  %137 = insertelement <4 x float> %136, float %nc.n1585, i64 3, !dbg !11
  %138 = select <4 x i1> %30, <4 x float> %133, <4 x float> %137, !dbg !11
  %nc.n1598 = extractelement <4 x float> %nc.n891, i64 0, !dbg !11
  %nc.n1599 = extractelement <4 x float> %nc.n891, i64 1, !dbg !11
  %nc.n1600 = extractelement <4 x float> %nc.n891, i64 2, !dbg !11
  %nc.n1601 = extractelement <4 x float> %nc.n891, i64 3, !dbg !11
  %nc.n1602 = select i1 %.not, float %nc.n1599, float %nc.n1598, !dbg !11
  %nc.n1603 = bitcast float %nc.n1602 to i32, !dbg !11
  %nc.n1604 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1603), !dbg !11
  %nc.n1605 = bitcast i32 %nc.n1604 to float, !dbg !11
  %nc.n1606 = select i1 %.not, float %nc.n1598, float %nc.n1605, !dbg !11
  %nc.n1607 = select i1 %.not, float %nc.n1605, float %nc.n1599, !dbg !11
  %nc.n1608 = select i1 %.not, float %nc.n1601, float %nc.n1600, !dbg !11
  %nc.n1609 = bitcast float %nc.n1608 to i32, !dbg !11
  %nc.n1610 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n529, i32 %nc.n1609), !dbg !11
  %nc.n1611 = bitcast i32 %nc.n1610 to float, !dbg !11
  %nc.n1612 = select i1 %.not, float %nc.n1600, float %nc.n1611, !dbg !11
  %nc.n1613 = select i1 %.not, float %nc.n1611, float %nc.n1601, !dbg !11
  %nc.n1614 = select i1 %.not36, float %nc.n1612, float %nc.n1606, !dbg !11
  %nc.n1615 = bitcast float %nc.n1614 to i32, !dbg !11
  %nc.n1616 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1615), !dbg !11
  %nc.n1617 = select i1 %.not36, float %nc.n1613, float %nc.n1607, !dbg !11
  %nc.n1618 = bitcast float %nc.n1617 to i32, !dbg !11
  %nc.n1619 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %nc.n537, i32 %nc.n1618), !dbg !11
  %nc.n1620 = bitcast i32 %nc.n1616 to float, !dbg !11
  %nc.n1623 = bitcast i32 %nc.n1619 to float, !dbg !11
  %139 = insertelement <4 x float> poison, float %nc.n1606, i64 0, !dbg !11
  %140 = insertelement <4 x float> %139, float %nc.n1607, i64 1, !dbg !11
  %141 = insertelement <4 x float> %140, float %nc.n1620, i64 2, !dbg !11
  %142 = insertelement <4 x float> %141, float %nc.n1623, i64 3, !dbg !11
  %143 = insertelement <4 x float> poison, float %nc.n1620, i64 0, !dbg !11
  %144 = insertelement <4 x float> %143, float %nc.n1623, i64 1, !dbg !11
  %145 = insertelement <4 x float> %144, float %nc.n1612, i64 2, !dbg !11
  %146 = insertelement <4 x float> %145, float %nc.n1613, i64 3, !dbg !11
  %147 = select <4 x i1> %30, <4 x float> %142, <4 x float> %146, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %nc.n1513, !dbg !26
  %normalized = fmul <4 x float> %row.reciprocal, %147, !dbg !26
  %nc.n1630 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !27
  %nc.n1631 = extractelement <4 x half> %nc.n1630, i64 3, !dbg !27
  %nc.n1632 = extractelement <4 x half> %nc.n1630, i64 2, !dbg !27
  %nc.n1633 = extractelement <4 x half> %nc.n1630, i64 1, !dbg !27
  %nc.n1634 = extractelement <4 x half> %nc.n1630, i64 0, !dbg !27
  %nc.n1635 = bitcast i32 %nc.n1509 to float, !dbg !11
  %nc.n1638 = bitcast i32 %nc.n1512 to float, !dbg !11
  %148 = insertelement <4 x float> poison, float %nc.n1499, i64 0, !dbg !11
  %149 = insertelement <4 x float> %148, float %nc.n1500, i64 1, !dbg !11
  %150 = insertelement <4 x float> %149, float %nc.n1635, i64 2, !dbg !11
  %151 = insertelement <4 x float> %150, float %nc.n1638, i64 3, !dbg !11
  %152 = insertelement <4 x float> poison, float %nc.n1635, i64 0, !dbg !11
  %153 = insertelement <4 x float> %152, float %nc.n1638, i64 1, !dbg !11
  %154 = insertelement <4 x float> %153, float %nc.n1505, i64 2, !dbg !11
  %155 = insertelement <4 x float> %154, float %nc.n1506, i64 3, !dbg !11
  %156 = select <4 x i1> %30, <4 x float> %151, <4 x float> %155, !dbg !11
  %normalized37 = fmul <4 x float> %row.reciprocal, %156, !dbg !26
  %nc.n1645 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !27
  %nc.n1646 = extractelement <4 x half> %nc.n1645, i64 3, !dbg !27
  %nc.n1647 = extractelement <4 x half> %nc.n1645, i64 2, !dbg !27
  %nc.n1648 = extractelement <4 x half> %nc.n1645, i64 1, !dbg !27
  %nc.n1649 = extractelement <4 x half> %nc.n1645, i64 0, !dbg !27
  %nc.n1650 = bitcast i32 %nc.n1399 to float, !dbg !11
  %nc.n1653 = bitcast i32 %nc.n1402 to float, !dbg !11
  %157 = insertelement <4 x float> poison, float %nc.n1389, i64 0, !dbg !11
  %158 = insertelement <4 x float> %157, float %nc.n1390, i64 1, !dbg !11
  %159 = insertelement <4 x float> %158, float %nc.n1650, i64 2, !dbg !11
  %160 = insertelement <4 x float> %159, float %nc.n1653, i64 3, !dbg !11
  %161 = insertelement <4 x float> poison, float %nc.n1650, i64 0, !dbg !11
  %162 = insertelement <4 x float> %161, float %nc.n1653, i64 1, !dbg !11
  %163 = insertelement <4 x float> %162, float %nc.n1395, i64 2, !dbg !11
  %164 = insertelement <4 x float> %163, float %nc.n1396, i64 3, !dbg !11
  %165 = select <4 x i1> %30, <4 x float> %160, <4 x float> %164, !dbg !11
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %nc.n1296, !dbg !26
  %normalized39 = fmul <4 x float> %row.reciprocal38, %165, !dbg !26
  %nc.n1660 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !27
  %nc.n1661 = extractelement <4 x half> %nc.n1660, i64 3, !dbg !27
  %nc.n1662 = extractelement <4 x half> %nc.n1660, i64 2, !dbg !27
  %nc.n1663 = extractelement <4 x half> %nc.n1660, i64 1, !dbg !27
  %nc.n1664 = extractelement <4 x half> %nc.n1660, i64 0, !dbg !27
  %nc.n1665 = bitcast i32 %nc.n1292 to float, !dbg !11
  %nc.n1668 = bitcast i32 %nc.n1295 to float, !dbg !11
  %166 = insertelement <4 x float> poison, float %nc.n1282, i64 0, !dbg !11
  %167 = insertelement <4 x float> %166, float %nc.n1283, i64 1, !dbg !11
  %168 = insertelement <4 x float> %167, float %nc.n1665, i64 2, !dbg !11
  %169 = insertelement <4 x float> %168, float %nc.n1668, i64 3, !dbg !11
  %170 = insertelement <4 x float> poison, float %nc.n1665, i64 0, !dbg !11
  %171 = insertelement <4 x float> %170, float %nc.n1668, i64 1, !dbg !11
  %172 = insertelement <4 x float> %171, float %nc.n1288, i64 2, !dbg !11
  %173 = insertelement <4 x float> %172, float %nc.n1289, i64 3, !dbg !11
  %174 = select <4 x i1> %30, <4 x float> %169, <4 x float> %173, !dbg !11
  %normalized40 = fmul <4 x float> %row.reciprocal38, %174, !dbg !26
  %nc.n1675 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !27
  %nc.n1676 = extractelement <4 x half> %nc.n1675, i64 3, !dbg !27
  %nc.n1677 = extractelement <4 x half> %nc.n1675, i64 2, !dbg !27
  %nc.n1678 = extractelement <4 x half> %nc.n1675, i64 1, !dbg !27
  %nc.n1679 = extractelement <4 x half> %nc.n1675, i64 0, !dbg !27
  %address.thread974 = tail call i32 asm sideeffect "", "=v,0"(i32 %nc.n31) #8, !dbg !28, !range !9, !frisk.register_lifetimes !5
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %nc.n1683 = lshr i64 %address.local975, 2, !dbg !28
  %address.local978 = and i64 %nc.n1683, 48, !dbg !28
  %address.local980 = and i64 %address.local975, 240, !dbg !28
  %address.local979 = add i64 %nc.n27, %address.local975, !dbg !28
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !28
  %address.local982 = add i64 %address.local981, %address.local978, !dbg !28
  %175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %nc.n28, !dbg !28
  %address.local985.idx = shl nuw nsw i64 %address.local982, 8, !dbg !28
  %address.local985 = getelementptr inbounds nuw i8, ptr addrspace(1) %175, i64 %address.local985.idx, !dbg !28
  %address.local987 = and i64 %nc.n1683, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !28
  store half %nc.n1679, ptr addrspace(1) %address.local988, align 2, !dbg !28
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !28
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !28
  store half %nc.n1678, ptr addrspace(1) %address.local991, align 2, !dbg !28
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !28
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !28
  store half %nc.n1677, ptr addrspace(1) %address.local994, align 2, !dbg !28
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !28
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !28
  store half %nc.n1676, ptr addrspace(1) %address.local997, align 2, !dbg !28
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !28
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !28
  store half %nc.n1664, ptr addrspace(1) %address.local1000, align 2, !dbg !28
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !28
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !28
  store half %nc.n1663, ptr addrspace(1) %address.local1003, align 2, !dbg !28
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !28
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !28
  store half %nc.n1662, ptr addrspace(1) %address.local1006, align 2, !dbg !28
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !28
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !28
  store half %nc.n1661, ptr addrspace(1) %address.local1009, align 2, !dbg !28
  %address.local1013 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local985, i64 16384, !dbg !28
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !28
  store half %nc.n1649, ptr addrspace(1) %address.local1014, align 2, !dbg !28
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !28
  store half %nc.n1648, ptr addrspace(1) %address.local1016, align 2, !dbg !28
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !28
  store half %nc.n1647, ptr addrspace(1) %address.local1018, align 2, !dbg !28
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !28
  store half %nc.n1646, ptr addrspace(1) %address.local1020, align 2, !dbg !28
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !28
  store half %nc.n1634, ptr addrspace(1) %address.local1022, align 2, !dbg !28
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !28
  store half %nc.n1633, ptr addrspace(1) %address.local1024, align 2, !dbg !28
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !28
  store half %nc.n1632, ptr addrspace(1) %address.local1026, align 2, !dbg !28
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !28
  store half %nc.n1631, ptr addrspace(1) %address.local1028, align 2, !dbg !28
  %normalized42 = fmul <4 x float> %row.reciprocal, %138, !dbg !26
  %nc.n1688 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !27
  %nc.n1689 = extractelement <4 x half> %nc.n1688, i64 3, !dbg !27
  %nc.n1690 = extractelement <4 x half> %nc.n1688, i64 2, !dbg !27
  %nc.n1691 = extractelement <4 x half> %nc.n1688, i64 1, !dbg !27
  %nc.n1692 = extractelement <4 x half> %nc.n1688, i64 0, !dbg !27
  %normalized43 = fmul <4 x float> %row.reciprocal, %111, !dbg !26
  %nc.n1697 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !27
  %nc.n1698 = extractelement <4 x half> %nc.n1697, i64 3, !dbg !27
  %nc.n1699 = extractelement <4 x half> %nc.n1697, i64 2, !dbg !27
  %nc.n1700 = extractelement <4 x half> %nc.n1697, i64 1, !dbg !27
  %nc.n1701 = extractelement <4 x half> %nc.n1697, i64 0, !dbg !27
  %normalized45 = fmul <4 x float> %row.reciprocal38, %84, !dbg !26
  %nc.n1706 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !27
  %nc.n1707 = extractelement <4 x half> %nc.n1706, i64 3, !dbg !27
  %nc.n1708 = extractelement <4 x half> %nc.n1706, i64 2, !dbg !27
  %nc.n1709 = extractelement <4 x half> %nc.n1706, i64 1, !dbg !27
  %nc.n1710 = extractelement <4 x half> %nc.n1706, i64 0, !dbg !27
  %normalized46 = fmul <4 x float> %row.reciprocal38, %57, !dbg !26
  %nc.n1715 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !27
  %nc.n1716 = extractelement <4 x half> %nc.n1715, i64 3, !dbg !27
  %nc.n1717 = extractelement <4 x half> %nc.n1715, i64 2, !dbg !27
  %nc.n1718 = extractelement <4 x half> %nc.n1715, i64 1, !dbg !27
  %nc.n1719 = extractelement <4 x half> %nc.n1715, i64 0, !dbg !27
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !28
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !28
  store half %nc.n1719, ptr addrspace(1) %address.local1044, align 2, !dbg !28
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !28
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !28
  store half %nc.n1718, ptr addrspace(1) %address.local1047, align 2, !dbg !28
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !28
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !28
  store half %nc.n1717, ptr addrspace(1) %address.local1050, align 2, !dbg !28
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !28
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !28
  store half %nc.n1716, ptr addrspace(1) %address.local1053, align 2, !dbg !28
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !28
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !28
  store half %nc.n1710, ptr addrspace(1) %address.local1056, align 2, !dbg !28
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !28
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !28
  store half %nc.n1709, ptr addrspace(1) %address.local1059, align 2, !dbg !28
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !28
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !28
  store half %nc.n1708, ptr addrspace(1) %address.local1062, align 2, !dbg !28
  %address.local1064 = or i64 %nc.n1683, 51, !dbg !28
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !28
  store half %nc.n1707, ptr addrspace(1) %address.local1065, align 2, !dbg !28
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !28
  store half %nc.n1701, ptr addrspace(1) %address.local1070, align 2, !dbg !28
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !28
  store half %nc.n1700, ptr addrspace(1) %address.local1072, align 2, !dbg !28
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !28
  store half %nc.n1699, ptr addrspace(1) %address.local1074, align 2, !dbg !28
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !28
  store half %nc.n1698, ptr addrspace(1) %address.local1076, align 2, !dbg !28
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !28
  store half %nc.n1692, ptr addrspace(1) %address.local1078, align 2, !dbg !28
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !28
  store half %nc.n1691, ptr addrspace(1) %address.local1080, align 2, !dbg !28
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !28
  store half %nc.n1690, ptr addrspace(1) %address.local1082, align 2, !dbg !28
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !28
  store half %nc.n1689, ptr addrspace(1) %address.local1084, align 2, !dbg !28
  %normalized48 = fmul <4 x float> %row.reciprocal, %129, !dbg !26
  %nc.n1726 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !27
  %nc.n1727 = extractelement <4 x half> %nc.n1726, i64 3, !dbg !27
  %nc.n1728 = extractelement <4 x half> %nc.n1726, i64 2, !dbg !27
  %nc.n1729 = extractelement <4 x half> %nc.n1726, i64 1, !dbg !27
  %nc.n1730 = extractelement <4 x half> %nc.n1726, i64 0, !dbg !27
  %normalized49 = fmul <4 x float> %row.reciprocal, %102, !dbg !26
  %nc.n1735 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !27
  %nc.n1736 = extractelement <4 x half> %nc.n1735, i64 3, !dbg !27
  %nc.n1737 = extractelement <4 x half> %nc.n1735, i64 2, !dbg !27
  %nc.n1738 = extractelement <4 x half> %nc.n1735, i64 1, !dbg !27
  %nc.n1739 = extractelement <4 x half> %nc.n1735, i64 0, !dbg !27
  %normalized51 = fmul <4 x float> %row.reciprocal38, %75, !dbg !26
  %nc.n1744 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !27
  %nc.n1745 = extractelement <4 x half> %nc.n1744, i64 3, !dbg !27
  %nc.n1746 = extractelement <4 x half> %nc.n1744, i64 2, !dbg !27
  %nc.n1747 = extractelement <4 x half> %nc.n1744, i64 1, !dbg !27
  %nc.n1748 = extractelement <4 x half> %nc.n1744, i64 0, !dbg !27
  %normalized52 = fmul <4 x float> %row.reciprocal38, %48, !dbg !26
  %nc.n1753 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !27
  %nc.n1754 = extractelement <4 x half> %nc.n1753, i64 3, !dbg !27
  %nc.n1755 = extractelement <4 x half> %nc.n1753, i64 2, !dbg !27
  %nc.n1756 = extractelement <4 x half> %nc.n1753, i64 1, !dbg !27
  %nc.n1757 = extractelement <4 x half> %nc.n1753, i64 0, !dbg !27
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !28
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !28
  store half %nc.n1757, ptr addrspace(1) %address.local1100, align 2, !dbg !28
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !28
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !28
  store half %nc.n1756, ptr addrspace(1) %address.local1103, align 2, !dbg !28
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !28
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !28
  store half %nc.n1755, ptr addrspace(1) %address.local1106, align 2, !dbg !28
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !28
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !28
  store half %nc.n1754, ptr addrspace(1) %address.local1109, align 2, !dbg !28
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !28
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !28
  store half %nc.n1748, ptr addrspace(1) %address.local1112, align 2, !dbg !28
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !28
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !28
  store half %nc.n1747, ptr addrspace(1) %address.local1115, align 2, !dbg !28
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !28
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !28
  store half %nc.n1746, ptr addrspace(1) %address.local1118, align 2, !dbg !28
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !28
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !28
  store half %nc.n1745, ptr addrspace(1) %address.local1121, align 2, !dbg !28
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !28
  store half %nc.n1739, ptr addrspace(1) %address.local1126, align 2, !dbg !28
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !28
  store half %nc.n1738, ptr addrspace(1) %address.local1128, align 2, !dbg !28
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !28
  store half %nc.n1737, ptr addrspace(1) %address.local1130, align 2, !dbg !28
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !28
  store half %nc.n1736, ptr addrspace(1) %address.local1132, align 2, !dbg !28
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !28
  store half %nc.n1730, ptr addrspace(1) %address.local1134, align 2, !dbg !28
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !28
  store half %nc.n1729, ptr addrspace(1) %address.local1136, align 2, !dbg !28
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !28
  store half %nc.n1728, ptr addrspace(1) %address.local1138, align 2, !dbg !28
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !28
  store half %nc.n1727, ptr addrspace(1) %address.local1140, align 2, !dbg !28
  %normalized54 = fmul <4 x float> %row.reciprocal, %120, !dbg !26
  %nc.n1764 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !27
  %nc.n1765 = extractelement <4 x half> %nc.n1764, i64 3, !dbg !27
  %nc.n1766 = extractelement <4 x half> %nc.n1764, i64 2, !dbg !27
  %nc.n1767 = extractelement <4 x half> %nc.n1764, i64 1, !dbg !27
  %nc.n1768 = extractelement <4 x half> %nc.n1764, i64 0, !dbg !27
  %normalized55 = fmul <4 x float> %row.reciprocal, %93, !dbg !26
  %nc.n1773 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !27
  %nc.n1774 = extractelement <4 x half> %nc.n1773, i64 3, !dbg !27
  %nc.n1775 = extractelement <4 x half> %nc.n1773, i64 2, !dbg !27
  %nc.n1776 = extractelement <4 x half> %nc.n1773, i64 1, !dbg !27
  %nc.n1777 = extractelement <4 x half> %nc.n1773, i64 0, !dbg !27
  %normalized57 = fmul <4 x float> %row.reciprocal38, %66, !dbg !26
  %nc.n1782 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !27
  %nc.n1783 = extractelement <4 x half> %nc.n1782, i64 3, !dbg !27
  %nc.n1784 = extractelement <4 x half> %nc.n1782, i64 2, !dbg !27
  %nc.n1785 = extractelement <4 x half> %nc.n1782, i64 1, !dbg !27
  %nc.n1786 = extractelement <4 x half> %nc.n1782, i64 0, !dbg !27
  %normalized58 = fmul <4 x float> %row.reciprocal38, %39, !dbg !26
  %nc.n1791 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !27
  %nc.n1792 = extractelement <4 x half> %nc.n1791, i64 3, !dbg !27
  %nc.n1793 = extractelement <4 x half> %nc.n1791, i64 2, !dbg !27
  %nc.n1794 = extractelement <4 x half> %nc.n1791, i64 1, !dbg !27
  %nc.n1795 = extractelement <4 x half> %nc.n1791, i64 0, !dbg !27
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !28
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !28
  store half %nc.n1795, ptr addrspace(1) %address.local1156, align 2, !dbg !28
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !28
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !28
  store half %nc.n1794, ptr addrspace(1) %address.local1159, align 2, !dbg !28
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !28
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !28
  store half %nc.n1793, ptr addrspace(1) %address.local1162, align 2, !dbg !28
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !28
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !28
  store half %nc.n1792, ptr addrspace(1) %address.local1165, align 2, !dbg !28
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !28
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !28
  store half %nc.n1786, ptr addrspace(1) %address.local1168, align 2, !dbg !28
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !28
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !28
  store half %nc.n1785, ptr addrspace(1) %address.local1171, align 2, !dbg !28
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !28
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !28
  store half %nc.n1784, ptr addrspace(1) %address.local1174, align 2, !dbg !28
  %address.local1176 = or i64 %nc.n1683, 115, !dbg !28
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !28
  store half %nc.n1783, ptr addrspace(1) %address.local1177, align 2, !dbg !28
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !28
  store half %nc.n1777, ptr addrspace(1) %address.local1182, align 2, !dbg !28
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !28
  store half %nc.n1776, ptr addrspace(1) %address.local1184, align 2, !dbg !28
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !28
  store half %nc.n1775, ptr addrspace(1) %address.local1186, align 2, !dbg !28
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !28
  store half %nc.n1774, ptr addrspace(1) %address.local1188, align 2, !dbg !28
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !28
  store half %nc.n1768, ptr addrspace(1) %address.local1190, align 2, !dbg !28
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !28
  store half %nc.n1767, ptr addrspace(1) %address.local1192, align 2, !dbg !28
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !28
  store half %nc.n1766, ptr addrspace(1) %address.local1194, align 2, !dbg !28
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !28
  store half %nc.n1765, ptr addrspace(1) %address.local1196, align 2, !dbg !28
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
attributes #10 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_input_spidD_noncasual.mlir", directory: "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test")
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
