; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/retained_cross_bn_k0/k0_late_control.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %bnk.n5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %bnk.n6 = sext i32 %bnk.n5 to i64, !dbg !7
  %bnk.n7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %bnk.n8 = sext i32 %bnk.n7 to i64, !dbg !7
  %bnk.n9 = shl nsw i64 %bnk.n8, 5, !dbg !7
  %bnk.n10 = add nsw i64 %bnk.n9, %bnk.n6, !dbg !7
  %bnk.n11 = lshr i64 %bnk.n10, 5, !dbg !7
  %bnk.n12 = and i64 %bnk.n11, 576460752303423480, !dbg !7
  %bnk.n13 = and i64 %bnk.n6, 7, !dbg !7
  %bnk.n14 = or disjoint i64 %bnk.n12, %bnk.n13, !dbg !7
  %bnk.n15 = sub nsw i64 31, %bnk.n14, !dbg !7
  %bnk.n16 = lshr i64 %bnk.n10, 3, !dbg !7
  %bnk.n17 = icmp ugt i64 %bnk.n10, 1023, !dbg !8
  %bnk.n18 = add nsw i64 %bnk.n14, -32, !dbg !8
  %bnk.n19 = select i1 %bnk.n17, i64 %bnk.n18, i64 %bnk.n15, !dbg !8
  %bnk.n20 = sdiv i64 %bnk.n19, 32, !dbg !8
  %bnk.n21 = sext i1 %bnk.n17 to i64, !dbg !8
  %bnk.n22 = xor i64 %bnk.n20, %bnk.n21, !dbg !8
  %bnk.n23 = add nsw i64 %bnk.n22, %bnk.n16, !dbg !8
  %bnk.n25 = shl i64 %bnk.n15, 7, !dbg !8
  %5 = shl i64 %bnk.n22, 12, !dbg !8
  %bnk.n27 = sub i64 %bnk.n25, %5, !dbg !8
  %bnk.n24 = shl i64 %bnk.n23, 19, !dbg !8
  %bnk.n28 = and i64 %bnk.n24, 16252928, !dbg !8
  %bnk.n31 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %bnk.n33.idx = shl i64 %bnk.n27, 8, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local339 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local340 = and i64 %address.local339, 1536, !dbg !8
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !8
  %address.local344 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local345 = and i64 %address.local344, 16256, !dbg !8
  %address.local346 = lshr i64 %address.local, 4, !dbg !8
  %address.local347 = and i64 %address.local346, 8, !dbg !8
  %address.local348 = or disjoint i64 %address.local345, %address.local347, !dbg !8
  %bnk.n37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %6 = or disjoint i64 %address.local348, %bnk.n28, !dbg !8
  %7 = shl nuw nsw i64 %6, 1, !dbg !8
  %bnk.n39 = add i64 %7, %bnk.n33.idx, !dbg !8
  %dma.buffer.offset1612 = trunc i64 %bnk.n39 to i32, !dbg !8
  %8 = insertelement <2 x i64> poison, i64 %bnk.n37, i64 0, !dbg !8
  %bnk.n44 = bitcast <2 x i64> %8 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1613 = shufflevector <4 x i32> %bnk.n44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local350 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local351 = shl nuw nsw i64 %address.local350, 3, !dbg !8
  %address.local352 = and i64 %address.local351, 3584, !dbg !8
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !8
  %address.local358 = lshr i64 %address.local350, 4, !dbg !8
  %address.local359 = and i64 %address.local358, 24, !dbg !8
  %address.local360 = or disjoint i64 %address.local359, %address.local345, !dbg !8
  %9 = or disjoint i64 %address.local360, %bnk.n28, !dbg !8
  %10 = shl nuw nsw i64 %9, 1, !dbg !8
  %bnk.n53 = add i64 %10, %bnk.n33.idx, !dbg !8
  %dma.buffer.offset1609 = trunc i64 %bnk.n53 to i32, !dbg !8
  %bnk.n62 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %bnk.n63 = sext i32 %bnk.n62 to i64, !dbg !7
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !8
  %.idx = shl nuw nsw i64 %bnk.n28, 1, !dbg !8
  %11 = add i64 %.idx, %bnk.n33.idx, !dbg !8
  %address.local365.idx = shl nuw nsw i64 %address.local348, 1, !dbg !8
  %12 = or disjoint i64 %address.local365.idx, %11, !dbg !8
  %13 = trunc i64 %12 to i32, !dbg !8
  %dma.buffer.offset1606 = or disjoint i32 %13, 64, !dbg !8
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !8
  %14 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local341, i64 0, !dbg !8
  %15 = insertelement <4 x ptr addrspace(3)> %14, ptr addrspace(3) %address.local353, i64 1, !dbg !8
  %16 = insertelement <4 x ptr addrspace(3)> %15, ptr addrspace(3) %address.local362, i64 2, !dbg !8
  %17 = insertelement <4 x ptr addrspace(3)> %16, ptr addrspace(3) %address.local366, i64 3, !dbg !8
  %18 = ptrtoint <4 x ptr addrspace(3)> %17 to <4 x i64>, !dbg !8
  %19 = trunc <4 x i64> %18 to <4 x i32>, !dbg !8
  %20 = extractelement <4 x i32> %19, i64 0, !dbg !8
  %bnk.n34 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %20), !dbg !8
  %dma.buffer.lds1614 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n34), !dbg !8
  %bnk.n47 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1612, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %21 = extractelement <4 x i32> %19, i64 1, !dbg !8
  %bnk.n48 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %21), !dbg !8
  %dma.buffer.lds1611 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n48), !dbg !8
  %bnk.n61 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1609, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %22 = extractelement <4 x i32> %19, i64 2, !dbg !8
  %bnk.n68 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %22), !dbg !8
  %dma.buffer.lds1608 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n68), !dbg !8
  %bnk.n81 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1606, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %23 = extractelement <4 x i32> %19, i64 3, !dbg !8
  %bnk.n82 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %23), !dbg !8
  %address.local369.idx = shl nuw nsw i64 %address.local360, 1, !dbg !8
  %24 = or disjoint i64 %address.local369.idx, %11, !dbg !8
  %25 = trunc i64 %24 to i32, !dbg !8
  %dma.buffer.offset1603 = or disjoint i32 %25, 64, !dbg !8
  %dma.buffer.lds1605 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n82), !dbg !8
  %bnk.n95 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1603, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %address.local370 = lshr i32 %address.thread, 2, !dbg !10
  %address.local371 = and i32 %address.local370, 4, !dbg !10
  %address.local372 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local373 = and i32 %address.local372, 384, !dbg !10
  %address.local374 = or disjoint i32 %address.local371, %address.local373, !dbg !10
  %address.local375 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local376 = and i32 %address.local375, 120, !dbg !10
  %address.local377 = or disjoint i32 %address.local374, %address.local376, !dbg !10
  %address.local378 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local379 = and i32 %address.local378, 1024, !dbg !10
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !10
  %address.local381 = zext nneg i32 %address.local380 to i64, !dbg !10
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !10
  %bnk.n96 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !10
  %bnk.n97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %bnk.n98 = fmul <4 x half> %bnk.n96, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !10
  %bnk.n99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !10
  %bnk.n100 = fmul <4 x half> %bnk.n97, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !10
  %bnk.n101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !10
  %bnk.n102 = fmul <4 x half> %bnk.n99, splat (half 1.275630e-01), !dbg !10
  %bnk.n103 = fmul <4 x half> %bnk.n101, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1600 = or disjoint i32 %13, 128, !dbg !8
  %bnk.n119 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1600, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %dma.buffer.offset1597 = or disjoint i32 %25, 128, !dbg !8
  %bnk.n133 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !10
  %bnk.n134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !10
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !10
  %bnk.n135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !10
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !10
  %bnk.n136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !10
  %bnk.n137 = fmul <4 x half> %bnk.n134, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !10
  %bnk.n138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !10
  %bnk.n139 = fmul <4 x half> %bnk.n135, splat (half 1.275630e-01), !dbg !10
  %bnk.n140 = fmul <4 x half> %bnk.n136, splat (half 1.275630e-01), !dbg !10
  %bnk.n141 = fmul <4 x half> %bnk.n138, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1594 = or disjoint i32 %13, 192, !dbg !8
  %bnk.n157 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %dma.buffer.offset1591 = or disjoint i32 %25, 192, !dbg !8
  %bnk.n171 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %bnk.n172 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %bnk.n173 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %bnk.n174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !10
  %bnk.n175 = fmul <4 x half> %bnk.n172, splat (half 1.275630e-01), !dbg !10
  %bnk.n176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !10
  %bnk.n177 = fmul <4 x half> %bnk.n173, splat (half 1.275630e-01), !dbg !10
  %bnk.n178 = fmul <4 x half> %bnk.n174, splat (half 1.275630e-01), !dbg !10
  %bnk.n179 = fmul <4 x half> %bnk.n176, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %bnk.n180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !10
  %bnk.n181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !10
  %bnk.n182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !10
  %bnk.n183 = fmul <4 x half> %bnk.n180, splat (half 1.275630e-01), !dbg !10
  %bnk.n184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !10
  %bnk.n185 = fmul <4 x half> %bnk.n181, splat (half 1.275630e-01), !dbg !10
  %bnk.n186 = fmul <4 x half> %bnk.n182, splat (half 1.275630e-01), !dbg !10
  %bnk.n187 = fmul <4 x half> %bnk.n184, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %bnk.n188 = or disjoint i64 %bnk.n25, 127, !dbg !12
  %bnk.n189 = add i64 %bnk.n25, 128, !dbg !12
  %bnk.n190 = icmp uge i64 %bnk.n189, %bnk.n188, !dbg !12
  %bnk.n191 = sub nsw i64 32, %bnk.n14, !dbg !13
  %bnk.n192 = shl i64 %bnk.n16, 7, !dbg !14
  %bnk.n193 = and i64 %bnk.n192, 3968, !dbg !14
  %bnk.n194 = shl nsw i64 %bnk.n63, 12, !dbg !14
  %bnk.n195 = or disjoint i64 %bnk.n193, %bnk.n194, !dbg !14
  %bnk.n197 = shl nuw nsw i32 %bnk.n31, 2, !dbg !15
  %bnk.n199 = and i32 %bnk.n197, 12, !dbg !15
  %bnk.n200 = and i32 %bnk.n31, 1, !dbg !15
  %bnk.n201 = or disjoint i32 %bnk.n199, %bnk.n200, !dbg !15
  %26 = trunc nuw nsw i32 %bnk.n201 to i16, !dbg !15
  %bnk.n203.lhs.trunc = add nuw nsw i16 %26, 1, !dbg !15
  %bnk.n2035253 = lshr i16 %bnk.n203.lhs.trunc, 1, !dbg !15
  %bnk.n203.sext = zext nneg i16 %bnk.n2035253 to i64, !dbg !15
  %k0.prime.bnk.n218 = shl i64 %bnk.n16, 19, !dbg !14
  %k0.prime.bnk.n219 = and i64 %k0.prime.bnk.n218, 16252928, !dbg !14
  %k0.prime.address.local442 = and i64 %address.local, 123, !dbg !14
  %k0.prime.address.local443 = lshr i64 %address.local, 2, !dbg !14
  %k0.prime.address.local444 = and i64 %k0.prime.address.local443, 4, !dbg !14
  %k0.prime.address.local445 = add nuw nsw i64 %k0.prime.address.local444, %address.local, !dbg !14
  %k0.prime.address.local446 = and i64 %k0.prime.address.local445, 4, !dbg !14
  %k0.prime.address.local447 = or disjoint i64 %k0.prime.address.local446, %k0.prime.address.local442, !dbg !14
  %k0.prime.bnk.n229 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !14
  %k0.prime.address.local450.idx = shl nuw nsw i64 %k0.prime.address.local447, 8, !dbg !14
  %27 = or disjoint i64 %address.local347, %k0.prime.bnk.n219, !dbg !14
  %28 = shl nuw nsw i64 %27, 1, !dbg !14
  %k0.prime.bnk.n231 = or disjoint i64 %k0.prime.address.local450.idx, %28, !dbg !14
  %k0.prime.dma.buffer.offset1588 = trunc nuw nsw i64 %k0.prime.bnk.n231 to i32, !dbg !14
  %29 = insertelement <2 x i64> poison, i64 %k0.prime.bnk.n229, i64 0, !dbg !14
  %k0.prime.bnk.n236 = bitcast <2 x i64> %29 to <4 x i32>, !dbg !14
  %k0.prime.dma.buffer.rsrc1589 = shufflevector <4 x i32> %k0.prime.bnk.n236, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %k0.prime.bnk.n239 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %k0.prime.dma.buffer.offset1588, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #9, !dbg !14
  %k0.prime.bnk.n223.idx8 = or disjoint i64 %address.local359, %k0.prime.bnk.n219, !dbg !14
  %30 = shl nuw nsw i64 %k0.prime.bnk.n223.idx8, 1, !dbg !14
  %k0.prime.bnk.n245 = or disjoint i64 %k0.prime.address.local450.idx, %30, !dbg !14
  %k0.prime.dma.buffer.offset1585 = trunc nuw nsw i64 %k0.prime.bnk.n245 to i32, !dbg !14
  %k0.prime.bnk.n253 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %k0.prime.dma.buffer.offset1585, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #9, !dbg !14
  %bnk.n21154 = icmp ult i64 %bnk.n10, 1024, !dbg !11
  br i1 %bnk.n21154, label %bnk.n212.lr.ph, label %.bnk.n1581_crit_edge, !dbg !11

.bnk.n1581_crit_edge:                             ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre69 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre70 = add nuw nsw i32 %.pre69, 64, !dbg !11
  %.pre71 = xor i32 %.pre69, 16, !dbg !11
  %.pre72 = and i32 %.pre70, 192, !dbg !11
  %.pre73 = xor i32 %.pre69, 32, !dbg !11
  br label %bnk.n1581, !dbg !11

bnk.n212.lr.ph:                                   ; preds = %4
  %bnk.n32 = zext nneg i32 %bnk.n31 to i64, !dbg !8
  %bnk.n608 = ptrtoint ptr addrspace(1) %2 to i64
  %31 = insertelement <2 x i64> poison, i64 %bnk.n608, i64 0
  %bnk.n615 = bitcast <2 x i64> %31 to <4 x i32>
  %dma.buffer.rsrc1565 = shufflevector <4 x i32> %bnk.n615, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %32 = lshr i64 %bnk.n32, 2
  %bnk.n703 = and i64 %32, 48
  %bnk.n705 = and i32 %bnk.n31, 240
  %index.compact = zext nneg i32 %bnk.n705 to i64
  %bnk.n702 = sub nsw i64 %bnk.n32, %index.compact
  %bnk.n704 = add nsw i64 %bnk.n702, %bnk.n703
  %bnk.n706 = add nsw i64 %bnk.n704, %bnk.n25
  %bnk.n707 = and i64 %bnk.n32, 48
  %bnk.n709 = add i64 %bnk.n706, 1
  %bnk.n710 = lshr i64 %bnk.n709, 32
  %compare.high = trunc nuw i64 %bnk.n710 to i32
  %compare.low.base = trunc i64 %bnk.n709 to i32
  %bnk.n744 = add i64 %bnk.n706, 65
  %bnk.n745 = lshr i64 %bnk.n744, 32
  %compare.high1281 = trunc nuw i64 %bnk.n745 to i32
  %compare.low.base1285 = trunc i64 %bnk.n744 to i32
  %33 = insertelement <32 x i32> poison, i32 %compare.low.base1285, i64 0, !dbg !16
  %34 = shufflevector <32 x i32> %33, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  %35 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %36 = shufflevector <32 x i32> %35, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  br label %bnk.n212, !dbg !11

bnk.n212:                                         ; preds = %k0.join, %bnk.n212.lr.ph
  %bnk.n21060 = phi [2 x <8 x float>] [ zeroinitializer, %bnk.n212.lr.ph ], [ %bnk.n1579, %k0.join ]
  %bnk.n20959 = phi [2 x <8 x float>] [ zeroinitializer, %bnk.n212.lr.ph ], [ %bnk.n1575, %k0.join ]
  %bnk.n20858 = phi [2 x <8 x float>] [ zeroinitializer, %bnk.n212.lr.ph ], [ %bnk.n1571, %k0.join ]
  %bnk.n20757 = phi [2 x <1 x float>] [ zeroinitializer, %bnk.n212.lr.ph ], [ %bnk.n1189, %k0.join ]
  %bnk.n20656 = phi [2 x <8 x float>] [ zeroinitializer, %bnk.n212.lr.ph ], [ %bnk.n1567, %k0.join ]
  %bnk.n20555 = phi i64 [ 0, %bnk.n212.lr.ph ], [ %bnk.n1580, %k0.join ]
  %bnk.n213 = shl nuw nsw i64 %bnk.n20555, 7, !dbg !11
  %bnk.n216 = lshr i64 %bnk.n20555, 5, !dbg !14
  %bnk.n220 = shl i64 %bnk.n216, 19, !dbg !14
  %bnk.n221 = shl nuw nsw i64 %bnk.n20555, 14, !dbg !14
  %bnk.n222 = sub i64 %bnk.n221, %bnk.n220, !dbg !14
  %address.thread433 = tail call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #8, !dbg !14, !range !9, !frisk.register_lifetimes !5
  %address.local434 = zext nneg i32 %address.thread433 to i64, !dbg !8
  %address.local435 = shl nuw nsw i64 %address.local434, 3, !dbg !8
  %address.local436 = and i64 %address.local435, 1536, !dbg !8
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !8
  %address.local440 = lshr i64 %address.local434, 4, !dbg !14
  %address.local441 = and i64 %address.local440, 8, !dbg !14
  %address.local442 = and i64 %address.local434, 123, !dbg !14
  %address.local443 = lshr i64 %address.local434, 2, !dbg !14
  %address.local444 = and i64 %address.local443, 4, !dbg !14
  %address.local445 = add nuw nsw i64 %address.local444, %address.local434, !dbg !14
  %address.local446 = and i64 %address.local445, 4, !dbg !14
  %address.local447 = or disjoint i64 %address.local446, %address.local442, !dbg !14
  %address.local448 = shl nuw nsw i64 %address.local447, 7, !dbg !14
  %address.local449 = or disjoint i64 %address.local448, %address.local441, !dbg !14
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !8
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !8
  %address.local453 = and i64 %address.local452, 3584, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !8
  %address.local457 = lshr i64 %address.local451, 4, !dbg !14
  %address.local458 = and i64 %address.local457, 24, !dbg !14
  %address.local466 = or disjoint i64 %address.local448, %address.local458, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %bnk.n254 = add nuw nsw i64 %bnk.n20555, 1024, !dbg !14
  %bnk.n2589 = lshr i64 %bnk.n254, 12, !dbg !14
  %bnk.n260 = add nuw nsw i64 %bnk.n2589, %bnk.n16, !dbg !14
  %bnk.n261 = add nsw i64 %bnk.n216, %bnk.n195, !dbg !14
  %bnk.n271 = shl i64 %bnk.n260, 19, !dbg !14
  %bnk.n272 = and i64 %bnk.n271, 16252928, !dbg !14
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !8
  %bnk.n270 = add i64 %bnk.n222, %bnk.n216, !dbg !14
  %bnk.n273.idx10 = add i64 %bnk.n270, %bnk.n195, !dbg !14
  %37 = add i64 %bnk.n273.idx10, %bnk.n272, !dbg !14
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !8
  %38 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local437, i64 0, !dbg !8
  %39 = insertelement <4 x ptr addrspace(3)> %38, ptr addrspace(3) %address.local454, i64 1, !dbg !8
  %40 = insertelement <4 x ptr addrspace(3)> %39, ptr addrspace(3) %address.local468, i64 2, !dbg !8
  %41 = insertelement <4 x ptr addrspace(3)> %40, ptr addrspace(3) %address.local472, i64 3, !dbg !8
  %42 = ptrtoint <4 x ptr addrspace(3)> %41 to <4 x i64>, !dbg !8
  %43 = trunc <4 x i64> %42 to <4 x i32>, !dbg !8
  %44 = extractelement <4 x i32> %43, i64 0, !dbg !14
  %bnk.n226 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44), !dbg !14
  %45 = extractelement <4 x i32> %43, i64 1, !dbg !14
  %bnk.n240 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %45), !dbg !14
  %46 = extractelement <4 x i32> %43, i64 2, !dbg !14
  %bnk.n278 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !14
  %dma.buffer.lds1584 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n278), !dbg !14
  %47 = extractelement <4 x i32> %43, i64 3, !dbg !14
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !15
  %address.local481 = and i64 %address.local480, 24, !dbg !15
  %address.local483 = shl i64 %address.local434, 62, !dbg !15
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !15
  %address.local485 = and i64 %address.local484, -64, !dbg !15
  %address.local488 = and i64 %address.local443, 12, !dbg !10
  %48 = and i64 %address.local434, 32, !dbg !10
  %.not13 = icmp eq i64 %48, 0, !dbg !10
  %address.local492 = select i1 %.not13, i64 0, i64 1016, !dbg !10
  %.idx14 = mul nuw nsw i64 %address.local434, 320, !dbg !15
  %49 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx14, !dbg !15
  %.idx15 = mul nsw i64 %address.local443, -1280, !dbg !15
  %50 = getelementptr i8, ptr addrspace(3) %49, i64 %.idx15, !dbg !15
  %51 = getelementptr [2 x i8], ptr addrspace(3) %50, i64 %address.local481, !dbg !15
  %52 = getelementptr [2 x i8], ptr addrspace(3) %51, i64 %address.local485, !dbg !15
  %53 = getelementptr [2 x i8], ptr addrspace(3) %52, i64 %address.local488, !dbg !15
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) %53, i64 %address.local492, !dbg !15
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !15
  %address.local505 = shl nuw nsw i32 %address.thread433, 1, !dbg !15
  %address.local506 = and i32 %address.local505, 24, !dbg !15
  %address.local507 = shl nuw nsw i32 %address.thread433, 8, !dbg !15
  %address.local508 = and i32 %address.local507, 768, !dbg !15
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !15
  %address.local510 = shl nuw nsw i32 %address.thread433, 5, !dbg !15
  %address.local511 = and i32 %address.local510, 96, !dbg !15
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !15
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !15
  %address.local514 = lshr i32 %address.thread433, 1, !dbg !15
  %address.local515 = and i32 %address.local514, 1, !dbg !15
  %address.local516 = zext nneg i32 %address.local515 to i64, !dbg !15
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !15
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !15
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !15
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !15
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !15
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !15
  %bnk.n370 = add nuw nsw i64 %bnk.n20555, 2048, !dbg !14
  %bnk.n37416 = lshr i64 %bnk.n370, 12, !dbg !14
  %bnk.n376 = add nuw nsw i64 %bnk.n37416, %bnk.n16, !dbg !14
  %54 = insertelement <2 x i64> poison, i64 %bnk.n261, i64 0, !dbg !14
  %55 = shufflevector <2 x i64> %54, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !14
  %56 = icmp slt <2 x i64> %55, <i64 -32, i64 -64>, !dbg !14
  %57 = extractelement <2 x i1> %56, i64 0, !dbg !14
  %bnk.n267 = sext i1 %57 to i64, !dbg !14
  %58 = add nsw <2 x i64> %55, <i64 32, i64 64>, !dbg !14
  %59 = sub nsw <2 x i64> <i64 -33, i64 -65>, %55, !dbg !14
  %60 = select <2 x i1> %56, <2 x i64> %59, <2 x i64> %58, !dbg !14
  %61 = sdiv <2 x i64> %60, splat (i64 128), !dbg !14
  %62 = extractelement <2 x i64> %61, i64 0, !dbg !14
  %bnk.n268 = xor i64 %62, %bnk.n267, !dbg !14
  %63 = shl nsw i64 %bnk.n268, 7, !dbg !14
  %64 = sub i64 %37, %63, !dbg !14
  %65 = add i64 %address.local449, %64, !dbg !14
  %.tr = trunc i64 %65 to i32, !dbg !14
  %66 = shl i32 %.tr, 1, !dbg !14
  %dma.buffer.offset1582 = add i32 %66, 64, !dbg !14
  %bnk.n291 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #9, !dbg !14
  %bnk.n292 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47), !dbg !14
  %67 = add i64 %address.local466, %64, !dbg !14
  %.tr12 = trunc i64 %67 to i32, !dbg !14
  %68 = shl i32 %.tr12, 1, !dbg !14
  %dma.buffer.offset1579 = add i32 %68, 64, !dbg !14
  %dma.buffer.lds1581 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n292), !dbg !14
  %bnk.n305 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #9, !dbg !14
  %bnk.n306 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !15
  %bnk.n307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n306, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n308 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %bnk.n203.sext) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !15
  %address.local518 = shl i64 %address.local517, 6, !dbg !15
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !15
  %69 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local519, !dbg !15
  %70 = getelementptr [2 x i8], ptr addrspace(3) %69, i64 %address.local488, !dbg !15
  %71 = getelementptr [2 x i8], ptr addrspace(3) %70, i64 %address.local492, !dbg !15
  %address.local523 = getelementptr i8, ptr addrspace(3) %71, i64 64, !dbg !15
  %bnk.n309 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !15
  %bnk.n310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n308, <4 x float> %bnk.n307) #8, !dbg !15
  %bnk.n311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n309, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %71, i64 4160, !dbg !15
  %bnk.n312 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !15
  %bnk.n313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !15
  %bnk.n314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n312, <4 x float> %bnk.n311) #8, !dbg !15
  %bnk.n315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n313, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n316 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !15
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %71, i64 192, !dbg !15
  %bnk.n317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !15
  %bnk.n318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n316, <4 x float> %bnk.n315) #8, !dbg !15
  %bnk.n319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n317, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %71, i64 4288, !dbg !15
  %bnk.n320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !15
  %bnk.n321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !15
  %bnk.n322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n320, <4 x float> %bnk.n319) #8, !dbg !15
  %bnk.n323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n321, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !15
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %71, i64 1088, !dbg !15
  %bnk.n325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !15
  %bnk.n326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n324, <4 x float> %bnk.n323) #8, !dbg !15
  %bnk.n327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n325, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %71, i64 5184, !dbg !15
  %bnk.n328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !15
  %bnk.n329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !15
  %bnk.n330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n328, <4 x float> %bnk.n327) #8, !dbg !15
  %bnk.n331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n329, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !15
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %71, i64 1216, !dbg !15
  %bnk.n333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !15
  %bnk.n334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n332, <4 x float> %bnk.n331) #8, !dbg !15
  %bnk.n335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n98, <4 x half> %bnk.n333, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %71, i64 5312, !dbg !15
  %bnk.n336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !15
  %bnk.n338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n100, <4 x half> %bnk.n336, <4 x float> %bnk.n335) #8, !dbg !15
  %bnk.n339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n306, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n308, <4 x float> %bnk.n339) #8, !dbg !15
  %bnk.n343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n309, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n312, <4 x float> %bnk.n343) #8, !dbg !15
  %bnk.n347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n313, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n316, <4 x float> %bnk.n347) #8, !dbg !15
  %bnk.n351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n317, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n320, <4 x float> %bnk.n351) #8, !dbg !15
  %bnk.n355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n321, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n324, <4 x float> %bnk.n355) #8, !dbg !15
  %bnk.n359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n325, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n328, <4 x float> %bnk.n359) #8, !dbg !15
  %bnk.n363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n329, <4 x float> zeroinitializer) #8, !dbg !15
  %bnk.n366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n332, <4 x float> %bnk.n363) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %bnk.n368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n102, <4 x half> %bnk.n333, <4 x float> zeroinitializer) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %bnk.n369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n103, <4 x half> %bnk.n336, <4 x float> %bnk.n368) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %72 = extractelement <2 x i1> %56, i64 1, !dbg !14
  %bnk.n382 = sext i1 %72 to i64, !dbg !14
  %73 = extractelement <2 x i64> %61, i64 1, !dbg !14
  %bnk.n383 = xor i64 %73, %bnk.n382, !dbg !14
  %bnk.n386 = shl i64 %bnk.n376, 19, !dbg !14
  %bnk.n387 = and i64 %bnk.n386, 16252928, !dbg !14
  %74 = add i64 %bnk.n273.idx10, %bnk.n387, !dbg !14
  %75 = shl nsw i64 %bnk.n383, 7, !dbg !14
  %76 = sub i64 %74, %75, !dbg !14
  %77 = add i64 %address.local449, %76, !dbg !14
  %.tr18 = trunc i64 %77 to i32, !dbg !14
  %78 = shl i32 %.tr18, 1, !dbg !14
  %dma.buffer.offset1576 = add i32 %78, 128, !dbg !14
  %dma.buffer.lds1578 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n226), !dbg !14
  %bnk.n406 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #9, !dbg !14
  %79 = add i64 %address.local466, %76, !dbg !14
  %.tr20 = trunc i64 %79 to i32, !dbg !14
  %80 = shl i32 %.tr20, 1, !dbg !14
  %dma.buffer.offset1573 = add i32 %80, 128, !dbg !14
  %dma.buffer.lds1575 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n240), !dbg !14
  %bnk.n420 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1573, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #9, !dbg !14
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !15
  %bnk.n421 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !15
  %bnk.n422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n421, <4 x float> %bnk.n310) #8, !dbg !15
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !15
  %bnk.n423 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !15
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %71, i64 8256, !dbg !15
  %bnk.n424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !15
  %bnk.n425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n423, <4 x float> %bnk.n422) #8, !dbg !15
  %bnk.n426 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n424, <4 x float> %bnk.n314) #8, !dbg !15
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %71, i64 12352, !dbg !15
  %bnk.n427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !15
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !15
  %bnk.n428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !15
  %bnk.n429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n427, <4 x float> %bnk.n426) #8, !dbg !15
  %bnk.n430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n428, <4 x float> %bnk.n318) #8, !dbg !15
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !15
  %bnk.n431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !15
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %71, i64 8384, !dbg !15
  %bnk.n432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !15
  %bnk.n433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n431, <4 x float> %bnk.n430) #8, !dbg !15
  %bnk.n434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n432, <4 x float> %bnk.n322) #8, !dbg !15
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %71, i64 12480, !dbg !15
  %bnk.n435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !15
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !15
  %bnk.n436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !15
  %bnk.n437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n435, <4 x float> %bnk.n434) #8, !dbg !15
  %bnk.n438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n436, <4 x float> %bnk.n326) #8, !dbg !15
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !15
  %bnk.n439 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !15
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %71, i64 9280, !dbg !15
  %bnk.n440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !15
  %bnk.n441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n439, <4 x float> %bnk.n438) #8, !dbg !15
  %bnk.n442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n440, <4 x float> %bnk.n330) #8, !dbg !15
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %71, i64 13376, !dbg !15
  %bnk.n443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !15
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !15
  %bnk.n444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !15
  %bnk.n445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n443, <4 x float> %bnk.n442) #8, !dbg !15
  %bnk.n446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n444, <4 x float> %bnk.n334) #8, !dbg !15
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !15
  %bnk.n447 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !15
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %71, i64 9408, !dbg !15
  %bnk.n448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !15
  %bnk.n449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n447, <4 x float> %bnk.n446) #8, !dbg !15
  %bnk.n450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n137, <4 x half> %bnk.n448, <4 x float> %bnk.n338) #8, !dbg !15
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %71, i64 13504, !dbg !15
  %bnk.n451 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !15
  %bnk.n453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n139, <4 x half> %bnk.n451, <4 x float> %bnk.n450) #8, !dbg !15
  %bnk.n454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n421, <4 x float> %bnk.n342) #8, !dbg !15
  %bnk.n457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n423, <4 x float> %bnk.n454) #8, !dbg !15
  %bnk.n458 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n424, <4 x float> %bnk.n346) #8, !dbg !15
  %bnk.n461 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n427, <4 x float> %bnk.n458) #8, !dbg !15
  %bnk.n462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n428, <4 x float> %bnk.n350) #8, !dbg !15
  %bnk.n465 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n431, <4 x float> %bnk.n462) #8, !dbg !15
  %bnk.n466 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n432, <4 x float> %bnk.n354) #8, !dbg !15
  %bnk.n469 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n435, <4 x float> %bnk.n466) #8, !dbg !15
  %bnk.n470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n436, <4 x float> %bnk.n358) #8, !dbg !15
  %bnk.n473 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n439, <4 x float> %bnk.n470) #8, !dbg !15
  %bnk.n474 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n440, <4 x float> %bnk.n362) #8, !dbg !15
  %bnk.n477 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n443, <4 x float> %bnk.n474) #8, !dbg !15
  %bnk.n478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n444, <4 x float> %bnk.n366) #8, !dbg !15
  %bnk.n481 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n447, <4 x float> %bnk.n478) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %bnk.n483 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n140, <4 x half> %bnk.n448, <4 x float> %bnk.n369) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %bnk.n484 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n141, <4 x half> %bnk.n451, <4 x float> %bnk.n483) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %bnk.n485 = add nuw nsw i64 %bnk.n20555, 3072, !dbg !14
  %bnk.n48921 = lshr i64 %bnk.n485, 12, !dbg !14
  %bnk.n491 = add nuw nsw i64 %bnk.n48921, %bnk.n16, !dbg !14
  %bnk.n492 = add nsw i64 %bnk.n261, 96, !dbg !14
  %bnk.n493 = icmp slt i64 %bnk.n261, -96, !dbg !14
  %bnk.n494 = sub nsw i64 -97, %bnk.n261, !dbg !14
  %bnk.n495 = select i1 %bnk.n493, i64 %bnk.n494, i64 %bnk.n492, !dbg !14
  %bnk.n496 = sdiv i64 %bnk.n495, 128, !dbg !14
  %bnk.n497 = sext i1 %bnk.n493 to i64, !dbg !14
  %bnk.n498 = xor i64 %bnk.n496, %bnk.n497, !dbg !14
  %bnk.n501 = shl i64 %bnk.n491, 19, !dbg !14
  %bnk.n502 = and i64 %bnk.n501, 16252928, !dbg !14
  %81 = add i64 %bnk.n273.idx10, %bnk.n502, !dbg !14
  %82 = shl nsw i64 %bnk.n498, 7, !dbg !14
  %83 = sub i64 %81, %82, !dbg !14
  %84 = add i64 %address.local449, %83, !dbg !14
  %.tr23 = trunc i64 %84 to i32, !dbg !14
  %85 = shl i32 %.tr23, 1, !dbg !14
  %dma.buffer.offset1570 = add i32 %85, 192, !dbg !14
  %bnk.n521 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1570, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #9, !dbg !14
  %86 = add i64 %address.local466, %83, !dbg !14
  %.tr25 = trunc i64 %86 to i32, !dbg !14
  %87 = shl i32 %.tr25, 1, !dbg !14
  %dma.buffer.offset1567 = add i32 %87, 192, !dbg !14
  %bnk.n535 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1567, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #9, !dbg !14
  %bnk.n536 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !15
  %bnk.n537 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n536, <4 x float> %bnk.n425) #8, !dbg !15
  %bnk.n538 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !15
  %bnk.n539 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !15
  %bnk.n540 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n538, <4 x float> %bnk.n537) #8, !dbg !15
  %bnk.n541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n539, <4 x float> %bnk.n429) #8, !dbg !15
  %bnk.n542 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !15
  %bnk.n543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !15
  %bnk.n544 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n542, <4 x float> %bnk.n541) #8, !dbg !15
  %bnk.n545 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n543, <4 x float> %bnk.n433) #8, !dbg !15
  %bnk.n546 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !15
  %bnk.n547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !15
  %bnk.n548 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n546, <4 x float> %bnk.n545) #8, !dbg !15
  %bnk.n549 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n547, <4 x float> %bnk.n437) #8, !dbg !15
  %bnk.n550 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !15
  %bnk.n551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !15
  %bnk.n552 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n550, <4 x float> %bnk.n549) #8, !dbg !15
  %bnk.n553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n551, <4 x float> %bnk.n441) #8, !dbg !15
  %bnk.n554 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !15
  %bnk.n555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !15
  %bnk.n556 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n554, <4 x float> %bnk.n553) #8, !dbg !15
  %bnk.n557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n555, <4 x float> %bnk.n445) #8, !dbg !15
  %bnk.n558 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !15
  %bnk.n559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !15
  %bnk.n560 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n558, <4 x float> %bnk.n557) #8, !dbg !15
  %bnk.n561 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n559, <4 x float> %bnk.n449) #8, !dbg !15
  %bnk.n562 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !15
  %bnk.n563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !15
  %bnk.n564 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n562, <4 x float> %bnk.n561) #8, !dbg !15
  %bnk.n565 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n175, <4 x half> %bnk.n563, <4 x float> %bnk.n453) #8, !dbg !15
  %bnk.n566 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !15
  %bnk.n568 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n177, <4 x half> %bnk.n566, <4 x float> %bnk.n565) #8, !dbg !15
  %bnk.n569 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n536, <4 x float> %bnk.n457) #8, !dbg !15
  %bnk.n572 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n538, <4 x float> %bnk.n569) #8, !dbg !15
  %bnk.n573 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n539, <4 x float> %bnk.n461) #8, !dbg !15
  %bnk.n576 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n542, <4 x float> %bnk.n573) #8, !dbg !15
  %bnk.n577 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n543, <4 x float> %bnk.n465) #8, !dbg !15
  %bnk.n580 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n546, <4 x float> %bnk.n577) #8, !dbg !15
  %bnk.n581 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n547, <4 x float> %bnk.n469) #8, !dbg !15
  %bnk.n584 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n550, <4 x float> %bnk.n581) #8, !dbg !15
  %bnk.n585 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n551, <4 x float> %bnk.n473) #8, !dbg !15
  %bnk.n588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n554, <4 x float> %bnk.n585) #8, !dbg !15
  %bnk.n589 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n555, <4 x float> %bnk.n477) #8, !dbg !15
  %bnk.n592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n558, <4 x float> %bnk.n589) #8, !dbg !15
  %bnk.n593 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n559, <4 x float> %bnk.n481) #8, !dbg !15
  %bnk.n596 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n562, <4 x float> %bnk.n593) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %bnk.n598 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n178, <4 x half> %bnk.n563, <4 x float> %bnk.n484) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %bnk.n599 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n179, <4 x half> %bnk.n566, <4 x float> %bnk.n598) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %bnk.n600 = add nuw nsw i64 %bnk.n216, %bnk.n16, !dbg !17
  %bnk.n601 = shl i64 %bnk.n600, 19, !dbg !17
  %bnk.n602 = and i64 %bnk.n601, 16252928, !dbg !17
  %address.local852 = shl nuw nsw i64 %address.local434, 6, !dbg !17
  %address.local853 = and i64 %address.local852, 16128, !dbg !17
  %address.local855 = and i64 %address.local435, 24, !dbg !17
  %bnk.n603.idx27 = add i64 %bnk.n602, %bnk.n222, !dbg !17
  %88 = or disjoint i64 %address.local855, %bnk.n603.idx27, !dbg !17
  %89 = shl i64 %88, 1, !dbg !17
  %bnk.n610 = or disjoint i64 %89, %address.local853, !dbg !17
  %dma.buffer.offset1564 = trunc i64 %bnk.n610 to i32, !dbg !17
  %bnk.n618 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %address.local858 = shl nuw nsw i64 %address.local451, 6, !dbg !17
  %address.local859 = and i64 %address.local858, 32512, !dbg !17
  %bnk.n624 = or disjoint i64 %89, %address.local859, !dbg !17
  %dma.buffer.offset1561 = trunc i64 %bnk.n624 to i32, !dbg !17
  %bnk.n632 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %bnk.n633 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !15
  %bnk.n634 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n633, <4 x float> %bnk.n540) #8, !dbg !15
  %bnk.n635 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !15
  %bnk.n636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !15
  %bnk.n637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n635, <4 x float> %bnk.n634) #8, !dbg !15
  %bnk.n638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n636, <4 x float> %bnk.n544) #8, !dbg !15
  %bnk.n639 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !15
  %bnk.n640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !15
  %bnk.n641 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n639, <4 x float> %bnk.n638) #8, !dbg !15
  %bnk.n642 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n640, <4 x float> %bnk.n548) #8, !dbg !15
  %bnk.n643 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !15
  %bnk.n644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !15
  %bnk.n645 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n643, <4 x float> %bnk.n642) #8, !dbg !15
  %bnk.n646 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n644, <4 x float> %bnk.n552) #8, !dbg !15
  %bnk.n647 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !15
  %bnk.n648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !15
  %bnk.n649 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n647, <4 x float> %bnk.n646) #8, !dbg !15
  %bnk.n650 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n648, <4 x float> %bnk.n556) #8, !dbg !15
  %bnk.n651 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !15
  %bnk.n652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !15
  %bnk.n653 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n651, <4 x float> %bnk.n650) #8, !dbg !15
  %bnk.n654 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n652, <4 x float> %bnk.n560) #8, !dbg !15
  %bnk.n655 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !15
  %bnk.n656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !15
  %bnk.n657 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n655, <4 x float> %bnk.n654) #8, !dbg !15
  %bnk.n658 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n656, <4 x float> %bnk.n564) #8, !dbg !15
  %bnk.n659 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !15
  %bnk.n660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !15
  %bnk.n661 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n659, <4 x float> %bnk.n658) #8, !dbg !15
  %bnk.n662 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n183, <4 x half> %bnk.n660, <4 x float> %bnk.n568) #8, !dbg !15
  %bnk.n663 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !15
  %bnk.n665 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n185, <4 x half> %bnk.n663, <4 x float> %bnk.n662) #8, !dbg !15
  %bnk.n666 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n633, <4 x float> %bnk.n572) #8, !dbg !15
  %bnk.n669 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n635, <4 x float> %bnk.n666) #8, !dbg !15
  %bnk.n670 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n636, <4 x float> %bnk.n576) #8, !dbg !15
  %bnk.n673 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n639, <4 x float> %bnk.n670) #8, !dbg !15
  %bnk.n674 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n640, <4 x float> %bnk.n580) #8, !dbg !15
  %bnk.n677 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n643, <4 x float> %bnk.n674) #8, !dbg !15
  %bnk.n678 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n644, <4 x float> %bnk.n584) #8, !dbg !15
  %bnk.n681 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n647, <4 x float> %bnk.n678) #8, !dbg !15
  %bnk.n682 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n648, <4 x float> %bnk.n588) #8, !dbg !15
  %bnk.n685 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n651, <4 x float> %bnk.n682) #8, !dbg !15
  %bnk.n686 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n652, <4 x float> %bnk.n592) #8, !dbg !15
  %bnk.n689 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n655, <4 x float> %bnk.n686) #8, !dbg !15
  %bnk.n690 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n656, <4 x float> %bnk.n596) #8, !dbg !15
  %bnk.n693 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n659, <4 x float> %bnk.n690) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %bnk.n695 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n186, <4 x half> %bnk.n660, <4 x float> %bnk.n599) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %bnk.n696 = or disjoint i64 %bnk.n213, 127, !dbg !12
  %bnk.n697 = icmp uge i64 %bnk.n25, %bnk.n696, !dbg !12
  %bnk.n698 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %bnk.n187, <4 x half> %bnk.n663, <4 x float> %bnk.n695) #8, !dbg !15
  %bnk.n699 = and i1 %bnk.n190, %bnk.n697, !dbg !12
  br i1 %bnk.n699, label %bnk.n847, label %bnk.n701, !dbg !12

bnk.n701:                                         ; preds = %bnk.n212
  %bnk.n708 = or disjoint i64 %bnk.n213, %bnk.n707, !dbg !12
  %bnk.n711 = lshr i64 %bnk.n20555, 25, !dbg !16
  %compare.high1157 = trunc nuw i64 %bnk.n711 to i32, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !16
  %compare.low.base1158 = trunc i64 %bnk.n708 to i32, !dbg !16
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1161 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1165 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1169 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 4) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1173 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 5) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1177 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 6) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1181 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 7) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1185 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 8) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1189 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 9) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1193 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 10) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1197 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 11) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1201 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 12) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1205 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 13) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1209 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 14) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1213 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 15) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1217 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 16) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1221 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 17) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1225 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 18) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1229 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 19) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1233 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 20) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1237 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 21) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1241 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 22) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1245 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 23) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1249 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 24) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1253 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 25) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1257 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 26) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1261 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 27) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1265 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 28) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1269 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 29) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1273 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 30) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1277 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 31) #8, !dbg !16, !frisk.register_lifetimes !5
  %90 = insertelement <32 x i32> poison, i32 %compare.low.base1158, i64 0, !dbg !16
  %91 = insertelement <32 x i32> %90, i32 %compare.local.base, i64 1, !dbg !16
  %92 = insertelement <32 x i32> %91, i32 %compare.local.base1161, i64 2, !dbg !16
  %93 = insertelement <32 x i32> %92, i32 %compare.local.base1165, i64 3, !dbg !16
  %94 = insertelement <32 x i32> %93, i32 %compare.local.base1169, i64 4, !dbg !16
  %95 = insertelement <32 x i32> %94, i32 %compare.local.base1173, i64 5, !dbg !16
  %96 = insertelement <32 x i32> %95, i32 %compare.local.base1177, i64 6, !dbg !16
  %97 = insertelement <32 x i32> %96, i32 %compare.local.base1181, i64 7, !dbg !16
  %98 = insertelement <32 x i32> %97, i32 %compare.local.base1185, i64 8, !dbg !16
  %99 = insertelement <32 x i32> %98, i32 %compare.local.base1189, i64 9, !dbg !16
  %100 = insertelement <32 x i32> %99, i32 %compare.local.base1193, i64 10, !dbg !16
  %101 = insertelement <32 x i32> %100, i32 %compare.local.base1197, i64 11, !dbg !16
  %102 = insertelement <32 x i32> %101, i32 %compare.local.base1201, i64 12, !dbg !16
  %103 = insertelement <32 x i32> %102, i32 %compare.local.base1205, i64 13, !dbg !16
  %104 = insertelement <32 x i32> %103, i32 %compare.local.base1209, i64 14, !dbg !16
  %105 = insertelement <32 x i32> %104, i32 %compare.local.base1213, i64 15, !dbg !16
  %106 = insertelement <32 x i32> %105, i32 %compare.local.base1217, i64 16, !dbg !16
  %107 = insertelement <32 x i32> %106, i32 %compare.local.base1221, i64 17, !dbg !16
  %108 = insertelement <32 x i32> %107, i32 %compare.local.base1225, i64 18, !dbg !16
  %109 = insertelement <32 x i32> %108, i32 %compare.local.base1229, i64 19, !dbg !16
  %110 = insertelement <32 x i32> %109, i32 %compare.local.base1233, i64 20, !dbg !16
  %111 = insertelement <32 x i32> %110, i32 %compare.local.base1237, i64 21, !dbg !16
  %112 = insertelement <32 x i32> %111, i32 %compare.local.base1241, i64 22, !dbg !16
  %113 = insertelement <32 x i32> %112, i32 %compare.local.base1245, i64 23, !dbg !16
  %114 = insertelement <32 x i32> %113, i32 %compare.local.base1249, i64 24, !dbg !16
  %115 = insertelement <32 x i32> %114, i32 %compare.local.base1253, i64 25, !dbg !16
  %116 = insertelement <32 x i32> %115, i32 %compare.local.base1257, i64 26, !dbg !16
  %117 = insertelement <32 x i32> %116, i32 %compare.local.base1261, i64 27, !dbg !16
  %118 = insertelement <32 x i32> %117, i32 %compare.local.base1265, i64 28, !dbg !16
  %119 = insertelement <32 x i32> %118, i32 %compare.local.base1269, i64 29, !dbg !16
  %120 = insertelement <32 x i32> %119, i32 %compare.local.base1273, i64 30, !dbg !16
  %121 = insertelement <32 x i32> %120, i32 %compare.local.base1277, i64 31, !dbg !16
  %122 = or <32 x i32> %121, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %123 = icmp ugt <32 x i32> %36, %122, !dbg !16
  %124 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %125 = shufflevector <32 x i1> %124, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %126 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %127 = shufflevector <32 x i1> %126, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %128 = select <32 x i1> %125, <32 x i1> %123, <32 x i1> %127, !dbg !16
  %129 = select <32 x i1> %128, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1157, !dbg !16
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1157, !dbg !16
  %compare.local.base1289 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 32) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1293 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 33) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1297 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 34) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1301 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 35) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1305 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 36) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1309 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 37) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1313 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 38) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1317 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 39) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1321 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 40) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1325 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 41) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1329 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 42) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1333 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 43) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1337 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 44) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1341 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 45) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1345 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 46) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1349 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 47) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1353 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 48) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1357 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 49) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1361 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 50) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1365 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 51) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1369 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 52) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1373 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 53) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1377 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 54) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1381 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 55) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1385 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 56) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1389 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 57) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1393 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 58) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1397 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 59) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1401 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 60) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1405 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 61) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1409 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 62) #8, !dbg !16, !frisk.register_lifetimes !5
  %130 = insertelement <32 x i32> %90, i32 %compare.local.base1289, i64 1, !dbg !16
  %131 = insertelement <32 x i32> %130, i32 %compare.local.base1293, i64 2, !dbg !16
  %132 = insertelement <32 x i32> %131, i32 %compare.local.base1297, i64 3, !dbg !16
  %133 = insertelement <32 x i32> %132, i32 %compare.local.base1301, i64 4, !dbg !16
  %134 = insertelement <32 x i32> %133, i32 %compare.local.base1305, i64 5, !dbg !16
  %135 = insertelement <32 x i32> %134, i32 %compare.local.base1309, i64 6, !dbg !16
  %136 = insertelement <32 x i32> %135, i32 %compare.local.base1313, i64 7, !dbg !16
  %137 = insertelement <32 x i32> %136, i32 %compare.local.base1317, i64 8, !dbg !16
  %138 = insertelement <32 x i32> %137, i32 %compare.local.base1321, i64 9, !dbg !16
  %139 = insertelement <32 x i32> %138, i32 %compare.local.base1325, i64 10, !dbg !16
  %140 = insertelement <32 x i32> %139, i32 %compare.local.base1329, i64 11, !dbg !16
  %141 = insertelement <32 x i32> %140, i32 %compare.local.base1333, i64 12, !dbg !16
  %142 = insertelement <32 x i32> %141, i32 %compare.local.base1337, i64 13, !dbg !16
  %143 = insertelement <32 x i32> %142, i32 %compare.local.base1341, i64 14, !dbg !16
  %144 = insertelement <32 x i32> %143, i32 %compare.local.base1345, i64 15, !dbg !16
  %145 = insertelement <32 x i32> %144, i32 %compare.local.base1349, i64 16, !dbg !16
  %146 = insertelement <32 x i32> %145, i32 %compare.local.base1353, i64 17, !dbg !16
  %147 = insertelement <32 x i32> %146, i32 %compare.local.base1357, i64 18, !dbg !16
  %148 = insertelement <32 x i32> %147, i32 %compare.local.base1361, i64 19, !dbg !16
  %149 = insertelement <32 x i32> %148, i32 %compare.local.base1365, i64 20, !dbg !16
  %150 = insertelement <32 x i32> %149, i32 %compare.local.base1369, i64 21, !dbg !16
  %151 = insertelement <32 x i32> %150, i32 %compare.local.base1373, i64 22, !dbg !16
  %152 = insertelement <32 x i32> %151, i32 %compare.local.base1377, i64 23, !dbg !16
  %153 = insertelement <32 x i32> %152, i32 %compare.local.base1381, i64 24, !dbg !16
  %154 = insertelement <32 x i32> %153, i32 %compare.local.base1385, i64 25, !dbg !16
  %155 = insertelement <32 x i32> %154, i32 %compare.local.base1389, i64 26, !dbg !16
  %156 = insertelement <32 x i32> %155, i32 %compare.local.base1393, i64 27, !dbg !16
  %157 = insertelement <32 x i32> %156, i32 %compare.local.base1397, i64 28, !dbg !16
  %158 = insertelement <32 x i32> %157, i32 %compare.local.base1401, i64 29, !dbg !16
  %159 = insertelement <32 x i32> %158, i32 %compare.local.base1405, i64 30, !dbg !16
  %160 = insertelement <32 x i32> %159, i32 %compare.local.base1409, i64 31, !dbg !16
  %161 = or <32 x i32> %160, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %162 = icmp ugt <32 x i32> %34, %161, !dbg !16
  %163 = insertelement <32 x i1> poison, i1 %compare.high.equal1283, i64 0, !dbg !16
  %164 = shufflevector <32 x i1> %163, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %165 = insertelement <32 x i1> poison, i1 %compare.high.order1284, i64 0, !dbg !16
  %166 = shufflevector <32 x i1> %165, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %167 = select <32 x i1> %164, <32 x i1> %162, <32 x i1> %166, !dbg !16
  %168 = select <32 x i1> %167, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %bnk.n811 = insertvalue [2 x <32 x float>] poison, <32 x float> %129, 0, !dbg !12
  %bnk.n844 = insertvalue [2 x <32 x float>] %bnk.n811, <32 x float> %168, 1, !dbg !12
  br label %bnk.n847, !dbg !12

bnk.n847:                                         ; preds = %bnk.n701, %bnk.n212
  %bnk.n846 = phi [2 x <32 x float>] [ %bnk.n844, %bnk.n701 ], [ zeroinitializer, %bnk.n212 ], !dbg !12
  %bnk.n848 = extractvalue [2 x <32 x float>] %bnk.n846, 0, !dbg !19
  %bnk.n849 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %bnk.n850 = fadd <4 x float> %bnk.n637, %bnk.n849, !dbg !19
  %bnk.n851 = extractelement <4 x float> %bnk.n850, i64 0, !dbg !20
  %bnk.n852 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n851), !dbg !20
  %bnk.n853 = insertelement <4 x float> poison, float %bnk.n852, i64 0, !dbg !21
  %bnk.n854 = extractelement <4 x float> %bnk.n850, i64 1, !dbg !20
  %bnk.n855 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n854), !dbg !20
  %bnk.n856 = insertelement <4 x float> %bnk.n853, float %bnk.n855, i64 1, !dbg !21
  %bnk.n857 = fadd float %bnk.n852, %bnk.n855, !dbg !22
  %bnk.n858 = extractelement <4 x float> %bnk.n850, i64 2, !dbg !20
  %bnk.n859 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n858), !dbg !20
  %bnk.n860 = insertelement <4 x float> %bnk.n856, float %bnk.n859, i64 2, !dbg !21
  %bnk.n861 = fadd float %bnk.n859, %bnk.n857, !dbg !22
  %bnk.n862 = extractelement <4 x float> %bnk.n850, i64 3, !dbg !20
  %bnk.n863 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n862), !dbg !20
  %bnk.n864 = insertelement <4 x float> %bnk.n860, float %bnk.n863, i64 3, !dbg !21
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n864) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n865 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !21
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n865) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n866 = fadd float %bnk.n863, %bnk.n861, !dbg !22
  %bnk.n867 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %bnk.n868 = fadd <4 x float> %bnk.n641, %bnk.n867, !dbg !19
  %bnk.n869 = extractelement <4 x float> %bnk.n868, i64 0, !dbg !20
  %bnk.n870 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n869), !dbg !20
  %bnk.n871 = insertelement <4 x float> poison, float %bnk.n870, i64 0, !dbg !21
  %bnk.n872 = fadd float %bnk.n870, %bnk.n866, !dbg !22
  %bnk.n873 = extractelement <4 x float> %bnk.n868, i64 1, !dbg !20
  %bnk.n874 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n873), !dbg !20
  %bnk.n875 = insertelement <4 x float> %bnk.n871, float %bnk.n874, i64 1, !dbg !21
  %bnk.n876 = fadd float %bnk.n874, %bnk.n872, !dbg !22
  %bnk.n877 = extractelement <4 x float> %bnk.n868, i64 2, !dbg !20
  %bnk.n878 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n877), !dbg !20
  %bnk.n879 = insertelement <4 x float> %bnk.n875, float %bnk.n878, i64 2, !dbg !21
  %bnk.n880 = fadd float %bnk.n878, %bnk.n876, !dbg !22
  %bnk.n881 = extractelement <4 x float> %bnk.n868, i64 3, !dbg !20
  %bnk.n882 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n881), !dbg !20
  %bnk.n883 = insertelement <4 x float> %bnk.n879, float %bnk.n882, i64 3, !dbg !21
  %math.result309 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n883) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n884 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !21
  %math.packed310 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n884) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n885 = fadd float %bnk.n882, %bnk.n880, !dbg !22
  %bnk.n886 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %bnk.n887 = fadd <4 x float> %bnk.n645, %bnk.n886, !dbg !19
  %bnk.n888 = extractelement <4 x float> %bnk.n887, i64 0, !dbg !20
  %bnk.n889 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n888), !dbg !20
  %bnk.n890 = insertelement <4 x float> poison, float %bnk.n889, i64 0, !dbg !21
  %bnk.n891 = fadd float %bnk.n889, %bnk.n885, !dbg !22
  %bnk.n892 = extractelement <4 x float> %bnk.n887, i64 1, !dbg !20
  %bnk.n893 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n892), !dbg !20
  %bnk.n894 = insertelement <4 x float> %bnk.n890, float %bnk.n893, i64 1, !dbg !21
  %bnk.n895 = fadd float %bnk.n893, %bnk.n891, !dbg !22
  %bnk.n896 = extractelement <4 x float> %bnk.n887, i64 2, !dbg !20
  %bnk.n897 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n896), !dbg !20
  %bnk.n898 = insertelement <4 x float> %bnk.n894, float %bnk.n897, i64 2, !dbg !21
  %bnk.n899 = fadd float %bnk.n897, %bnk.n895, !dbg !22
  %bnk.n900 = extractelement <4 x float> %bnk.n887, i64 3, !dbg !20
  %bnk.n901 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n900), !dbg !20
  %bnk.n902 = insertelement <4 x float> %bnk.n898, float %bnk.n901, i64 3, !dbg !21
  %math.result311 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n902) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n903 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !21
  %math.packed312 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n903) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n904 = fadd float %bnk.n901, %bnk.n899, !dbg !22
  %bnk.n905 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %bnk.n906 = fadd <4 x float> %bnk.n649, %bnk.n905, !dbg !19
  %bnk.n907 = extractelement <4 x float> %bnk.n906, i64 0, !dbg !20
  %bnk.n908 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n907), !dbg !20
  %bnk.n909 = insertelement <4 x float> poison, float %bnk.n908, i64 0, !dbg !21
  %bnk.n910 = fadd float %bnk.n908, %bnk.n904, !dbg !22
  %bnk.n911 = extractelement <4 x float> %bnk.n906, i64 1, !dbg !20
  %bnk.n912 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n911), !dbg !20
  %bnk.n913 = insertelement <4 x float> %bnk.n909, float %bnk.n912, i64 1, !dbg !21
  %bnk.n914 = fadd float %bnk.n912, %bnk.n910, !dbg !22
  %bnk.n915 = extractelement <4 x float> %bnk.n906, i64 2, !dbg !20
  %bnk.n916 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n915), !dbg !20
  %bnk.n917 = insertelement <4 x float> %bnk.n913, float %bnk.n916, i64 2, !dbg !21
  %bnk.n918 = fadd float %bnk.n916, %bnk.n914, !dbg !22
  %bnk.n919 = extractelement <4 x float> %bnk.n906, i64 3, !dbg !20
  %bnk.n920 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n919), !dbg !20
  %bnk.n921 = insertelement <4 x float> %bnk.n917, float %bnk.n920, i64 3, !dbg !21
  %math.result313 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n921) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n922 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !21
  %math.packed314 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n922) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n923 = fadd float %bnk.n920, %bnk.n918, !dbg !22
  %bnk.n924 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %bnk.n925 = fadd <4 x float> %bnk.n653, %bnk.n924, !dbg !19
  %bnk.n926 = extractelement <4 x float> %bnk.n925, i64 0, !dbg !20
  %bnk.n927 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n926), !dbg !20
  %bnk.n928 = insertelement <4 x float> poison, float %bnk.n927, i64 0, !dbg !21
  %bnk.n929 = fadd float %bnk.n927, %bnk.n923, !dbg !22
  %bnk.n930 = extractelement <4 x float> %bnk.n925, i64 1, !dbg !20
  %bnk.n931 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n930), !dbg !20
  %bnk.n932 = insertelement <4 x float> %bnk.n928, float %bnk.n931, i64 1, !dbg !21
  %bnk.n933 = fadd float %bnk.n931, %bnk.n929, !dbg !22
  %bnk.n934 = extractelement <4 x float> %bnk.n925, i64 2, !dbg !20
  %bnk.n935 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n934), !dbg !20
  %bnk.n936 = insertelement <4 x float> %bnk.n932, float %bnk.n935, i64 2, !dbg !21
  %bnk.n937 = fadd float %bnk.n935, %bnk.n933, !dbg !22
  %bnk.n938 = extractelement <4 x float> %bnk.n925, i64 3, !dbg !20
  %bnk.n939 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n938), !dbg !20
  %bnk.n940 = insertelement <4 x float> %bnk.n936, float %bnk.n939, i64 3, !dbg !21
  %math.result315 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n940) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n941 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !21
  %math.packed316 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n941) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n942 = fadd float %bnk.n939, %bnk.n937, !dbg !22
  %bnk.n943 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %bnk.n944 = fadd <4 x float> %bnk.n657, %bnk.n943, !dbg !19
  %bnk.n945 = extractelement <4 x float> %bnk.n944, i64 0, !dbg !20
  %bnk.n946 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n945), !dbg !20
  %bnk.n947 = insertelement <4 x float> poison, float %bnk.n946, i64 0, !dbg !21
  %bnk.n948 = fadd float %bnk.n946, %bnk.n942, !dbg !22
  %bnk.n949 = extractelement <4 x float> %bnk.n944, i64 1, !dbg !20
  %bnk.n950 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n949), !dbg !20
  %bnk.n951 = insertelement <4 x float> %bnk.n947, float %bnk.n950, i64 1, !dbg !21
  %bnk.n952 = fadd float %bnk.n950, %bnk.n948, !dbg !22
  %bnk.n953 = extractelement <4 x float> %bnk.n944, i64 2, !dbg !20
  %bnk.n954 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n953), !dbg !20
  %bnk.n955 = insertelement <4 x float> %bnk.n951, float %bnk.n954, i64 2, !dbg !21
  %bnk.n956 = fadd float %bnk.n954, %bnk.n952, !dbg !22
  %bnk.n957 = extractelement <4 x float> %bnk.n944, i64 3, !dbg !20
  %bnk.n958 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n957), !dbg !20
  %bnk.n959 = insertelement <4 x float> %bnk.n955, float %bnk.n958, i64 3, !dbg !21
  %math.result317 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n959) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n960 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !21
  %math.packed318 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n960) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n961 = fadd float %bnk.n958, %bnk.n956, !dbg !22
  %bnk.n962 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %bnk.n963 = fadd <4 x float> %bnk.n661, %bnk.n962, !dbg !19
  %bnk.n964 = extractelement <4 x float> %bnk.n963, i64 0, !dbg !20
  %bnk.n965 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n964), !dbg !20
  %bnk.n966 = insertelement <4 x float> poison, float %bnk.n965, i64 0, !dbg !21
  %bnk.n967 = fadd float %bnk.n965, %bnk.n961, !dbg !22
  %bnk.n968 = extractelement <4 x float> %bnk.n963, i64 1, !dbg !20
  %bnk.n969 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n968), !dbg !20
  %bnk.n970 = insertelement <4 x float> %bnk.n966, float %bnk.n969, i64 1, !dbg !21
  %bnk.n971 = fadd float %bnk.n969, %bnk.n967, !dbg !22
  %bnk.n972 = extractelement <4 x float> %bnk.n963, i64 2, !dbg !20
  %bnk.n973 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n972), !dbg !20
  %bnk.n974 = insertelement <4 x float> %bnk.n970, float %bnk.n973, i64 2, !dbg !21
  %bnk.n975 = fadd float %bnk.n973, %bnk.n971, !dbg !22
  %bnk.n976 = extractelement <4 x float> %bnk.n963, i64 3, !dbg !20
  %bnk.n977 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n976), !dbg !20
  %bnk.n978 = insertelement <4 x float> %bnk.n974, float %bnk.n977, i64 3, !dbg !21
  %math.result319 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n978) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n979 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !21
  %math.packed320 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n979) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n980 = fadd float %bnk.n977, %bnk.n975, !dbg !22
  %bnk.n981 = shufflevector <32 x float> %bnk.n848, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %bnk.n982 = fadd <4 x float> %bnk.n665, %bnk.n981, !dbg !19
  %bnk.n983 = extractelement <4 x float> %bnk.n982, i64 0, !dbg !20
  %bnk.n984 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n983), !dbg !20
  %bnk.n985 = insertelement <4 x float> poison, float %bnk.n984, i64 0, !dbg !21
  %bnk.n986 = fadd float %bnk.n984, %bnk.n980, !dbg !22
  %bnk.n987 = extractelement <4 x float> %bnk.n982, i64 1, !dbg !20
  %bnk.n988 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n987), !dbg !20
  %bnk.n989 = insertelement <4 x float> %bnk.n985, float %bnk.n988, i64 1, !dbg !21
  %bnk.n990 = fadd float %bnk.n988, %bnk.n986, !dbg !22
  %bnk.n991 = extractelement <4 x float> %bnk.n982, i64 2, !dbg !20
  %bnk.n992 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n991), !dbg !20
  %bnk.n993 = insertelement <4 x float> %bnk.n989, float %bnk.n992, i64 2, !dbg !21
  %bnk.n994 = fadd float %bnk.n992, %bnk.n990, !dbg !22
  %bnk.n995 = extractelement <4 x float> %bnk.n982, i64 3, !dbg !20
  %bnk.n996 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n995), !dbg !20
  %bnk.n997 = insertelement <4 x float> %bnk.n993, float %bnk.n996, i64 3, !dbg !21
  %math.result321 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n997) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n998 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !21
  %math.packed322 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n998) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n999 = fadd float %bnk.n996, %bnk.n994, !dbg !22
  %bnk.n1000 = extractvalue [2 x <32 x float>] %bnk.n846, 1, !dbg !19
  %bnk.n1001 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %bnk.n1002 = fadd <4 x float> %bnk.n669, %bnk.n1001, !dbg !19
  %bnk.n1003 = extractelement <4 x float> %bnk.n1002, i64 0, !dbg !20
  %bnk.n1004 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1003), !dbg !20
  %bnk.n1005 = insertelement <4 x float> poison, float %bnk.n1004, i64 0, !dbg !21
  %bnk.n1006 = extractelement <4 x float> %bnk.n1002, i64 1, !dbg !20
  %bnk.n1007 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1006), !dbg !20
  %bnk.n1008 = insertelement <4 x float> %bnk.n1005, float %bnk.n1007, i64 1, !dbg !21
  %bnk.n1009 = fadd float %bnk.n1004, %bnk.n1007, !dbg !22
  %bnk.n1010 = extractelement <4 x float> %bnk.n1002, i64 2, !dbg !20
  %bnk.n1011 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1010), !dbg !20
  %bnk.n1012 = insertelement <4 x float> %bnk.n1008, float %bnk.n1011, i64 2, !dbg !21
  %bnk.n1013 = fadd float %bnk.n1011, %bnk.n1009, !dbg !22
  %bnk.n1014 = extractelement <4 x float> %bnk.n1002, i64 3, !dbg !20
  %bnk.n1015 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1014), !dbg !20
  %bnk.n1016 = insertelement <4 x float> %bnk.n1012, float %bnk.n1015, i64 3, !dbg !21
  %math.result323 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1016) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n1017 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !21
  %math.packed324 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1017) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n1018 = fadd float %bnk.n1015, %bnk.n1013, !dbg !22
  %bnk.n1019 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %bnk.n1020 = fadd <4 x float> %bnk.n673, %bnk.n1019, !dbg !19
  %bnk.n1021 = extractelement <4 x float> %bnk.n1020, i64 0, !dbg !20
  %bnk.n1022 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1021), !dbg !20
  %bnk.n1023 = insertelement <4 x float> poison, float %bnk.n1022, i64 0, !dbg !21
  %bnk.n1024 = fadd float %bnk.n1022, %bnk.n1018, !dbg !22
  %bnk.n1025 = extractelement <4 x float> %bnk.n1020, i64 1, !dbg !20
  %bnk.n1026 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1025), !dbg !20
  %bnk.n1027 = insertelement <4 x float> %bnk.n1023, float %bnk.n1026, i64 1, !dbg !21
  %bnk.n1028 = fadd float %bnk.n1026, %bnk.n1024, !dbg !22
  %bnk.n1029 = extractelement <4 x float> %bnk.n1020, i64 2, !dbg !20
  %bnk.n1030 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1029), !dbg !20
  %bnk.n1031 = insertelement <4 x float> %bnk.n1027, float %bnk.n1030, i64 2, !dbg !21
  %bnk.n1032 = fadd float %bnk.n1030, %bnk.n1028, !dbg !22
  %bnk.n1033 = extractelement <4 x float> %bnk.n1020, i64 3, !dbg !20
  %bnk.n1034 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1033), !dbg !20
  %bnk.n1035 = insertelement <4 x float> %bnk.n1031, float %bnk.n1034, i64 3, !dbg !21
  %math.result325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1035) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n1036 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !21
  %math.packed326 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1036) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n1037 = fadd float %bnk.n1034, %bnk.n1032, !dbg !22
  %bnk.n1038 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %bnk.n1039 = fadd <4 x float> %bnk.n677, %bnk.n1038, !dbg !19
  %bnk.n1040 = extractelement <4 x float> %bnk.n1039, i64 0, !dbg !20
  %bnk.n1041 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1040), !dbg !20
  %bnk.n1042 = insertelement <4 x float> poison, float %bnk.n1041, i64 0, !dbg !21
  %bnk.n1043 = fadd float %bnk.n1041, %bnk.n1037, !dbg !22
  %bnk.n1044 = extractelement <4 x float> %bnk.n1039, i64 1, !dbg !20
  %bnk.n1045 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1044), !dbg !20
  %bnk.n1046 = insertelement <4 x float> %bnk.n1042, float %bnk.n1045, i64 1, !dbg !21
  %bnk.n1047 = fadd float %bnk.n1045, %bnk.n1043, !dbg !22
  %bnk.n1048 = extractelement <4 x float> %bnk.n1039, i64 2, !dbg !20
  %bnk.n1049 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1048), !dbg !20
  %bnk.n1050 = insertelement <4 x float> %bnk.n1046, float %bnk.n1049, i64 2, !dbg !21
  %bnk.n1051 = fadd float %bnk.n1049, %bnk.n1047, !dbg !22
  %bnk.n1052 = extractelement <4 x float> %bnk.n1039, i64 3, !dbg !20
  %bnk.n1053 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1052), !dbg !20
  %bnk.n1054 = insertelement <4 x float> %bnk.n1050, float %bnk.n1053, i64 3, !dbg !21
  %math.result327 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1054) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n1055 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !21
  %math.packed328 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1055) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n1056 = fadd float %bnk.n1053, %bnk.n1051, !dbg !22
  %bnk.n1057 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %bnk.n1058 = fadd <4 x float> %bnk.n681, %bnk.n1057, !dbg !19
  %bnk.n1059 = extractelement <4 x float> %bnk.n1058, i64 0, !dbg !20
  %bnk.n1060 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1059), !dbg !20
  %bnk.n1061 = insertelement <4 x float> poison, float %bnk.n1060, i64 0, !dbg !21
  %bnk.n1062 = fadd float %bnk.n1060, %bnk.n1056, !dbg !22
  %bnk.n1063 = extractelement <4 x float> %bnk.n1058, i64 1, !dbg !20
  %bnk.n1064 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1063), !dbg !20
  %bnk.n1065 = insertelement <4 x float> %bnk.n1061, float %bnk.n1064, i64 1, !dbg !21
  %bnk.n1066 = fadd float %bnk.n1064, %bnk.n1062, !dbg !22
  %bnk.n1067 = extractelement <4 x float> %bnk.n1058, i64 2, !dbg !20
  %bnk.n1068 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1067), !dbg !20
  %bnk.n1069 = insertelement <4 x float> %bnk.n1065, float %bnk.n1068, i64 2, !dbg !21
  %bnk.n1070 = fadd float %bnk.n1068, %bnk.n1066, !dbg !22
  %bnk.n1071 = extractelement <4 x float> %bnk.n1058, i64 3, !dbg !20
  %bnk.n1072 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1071), !dbg !20
  %bnk.n1073 = insertelement <4 x float> %bnk.n1069, float %bnk.n1072, i64 3, !dbg !21
  %math.result329 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1073) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n1074 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !21
  %math.packed330 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1074) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n1075 = fadd float %bnk.n1072, %bnk.n1070, !dbg !22
  %bnk.n1076 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %bnk.n1077 = fadd <4 x float> %bnk.n685, %bnk.n1076, !dbg !19
  %bnk.n1078 = extractelement <4 x float> %bnk.n1077, i64 0, !dbg !20
  %bnk.n1079 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1078), !dbg !20
  %bnk.n1080 = insertelement <4 x float> poison, float %bnk.n1079, i64 0, !dbg !21
  %bnk.n1081 = fadd float %bnk.n1079, %bnk.n1075, !dbg !22
  %bnk.n1082 = extractelement <4 x float> %bnk.n1077, i64 1, !dbg !20
  %bnk.n1083 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1082), !dbg !20
  %bnk.n1084 = insertelement <4 x float> %bnk.n1080, float %bnk.n1083, i64 1, !dbg !21
  %bnk.n1085 = fadd float %bnk.n1083, %bnk.n1081, !dbg !22
  %bnk.n1086 = extractelement <4 x float> %bnk.n1077, i64 2, !dbg !20
  %bnk.n1087 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1086), !dbg !20
  %bnk.n1088 = insertelement <4 x float> %bnk.n1084, float %bnk.n1087, i64 2, !dbg !21
  %bnk.n1089 = fadd float %bnk.n1087, %bnk.n1085, !dbg !22
  %bnk.n1090 = extractelement <4 x float> %bnk.n1077, i64 3, !dbg !20
  %bnk.n1091 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1090), !dbg !20
  %bnk.n1092 = insertelement <4 x float> %bnk.n1088, float %bnk.n1091, i64 3, !dbg !21
  %math.result331 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1092) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n1093 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !21
  %math.packed332 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1093) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n1094 = fadd float %bnk.n1091, %bnk.n1089, !dbg !22
  %bnk.n1095 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %bnk.n1096 = fadd <4 x float> %bnk.n689, %bnk.n1095, !dbg !19
  %bnk.n1097 = extractelement <4 x float> %bnk.n1096, i64 0, !dbg !20
  %bnk.n1098 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1097), !dbg !20
  %bnk.n1099 = insertelement <4 x float> poison, float %bnk.n1098, i64 0, !dbg !21
  %bnk.n1100 = fadd float %bnk.n1098, %bnk.n1094, !dbg !22
  %bnk.n1101 = extractelement <4 x float> %bnk.n1096, i64 1, !dbg !20
  %bnk.n1102 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1101), !dbg !20
  %bnk.n1103 = insertelement <4 x float> %bnk.n1099, float %bnk.n1102, i64 1, !dbg !21
  %bnk.n1104 = fadd float %bnk.n1102, %bnk.n1100, !dbg !22
  %bnk.n1105 = extractelement <4 x float> %bnk.n1096, i64 2, !dbg !20
  %bnk.n1106 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1105), !dbg !20
  %bnk.n1107 = insertelement <4 x float> %bnk.n1103, float %bnk.n1106, i64 2, !dbg !21
  %bnk.n1108 = fadd float %bnk.n1106, %bnk.n1104, !dbg !22
  %bnk.n1109 = extractelement <4 x float> %bnk.n1096, i64 3, !dbg !20
  %bnk.n1110 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1109), !dbg !20
  %bnk.n1111 = insertelement <4 x float> %bnk.n1107, float %bnk.n1110, i64 3, !dbg !21
  %math.result333 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1111) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n1112 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !21
  %math.packed334 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1112) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n1113 = fadd float %bnk.n1110, %bnk.n1108, !dbg !22
  %bnk.n1114 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %bnk.n1115 = fadd <4 x float> %bnk.n693, %bnk.n1114, !dbg !19
  %bnk.n1116 = extractelement <4 x float> %bnk.n1115, i64 0, !dbg !20
  %bnk.n1117 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1116), !dbg !20
  %bnk.n1118 = insertelement <4 x float> poison, float %bnk.n1117, i64 0, !dbg !21
  %bnk.n1119 = fadd float %bnk.n1117, %bnk.n1113, !dbg !22
  %bnk.n1120 = extractelement <4 x float> %bnk.n1115, i64 1, !dbg !20
  %bnk.n1121 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1120), !dbg !20
  %bnk.n1122 = insertelement <4 x float> %bnk.n1118, float %bnk.n1121, i64 1, !dbg !21
  %bnk.n1123 = fadd float %bnk.n1121, %bnk.n1119, !dbg !22
  %bnk.n1124 = extractelement <4 x float> %bnk.n1115, i64 2, !dbg !20
  %bnk.n1125 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1124), !dbg !20
  %bnk.n1126 = insertelement <4 x float> %bnk.n1122, float %bnk.n1125, i64 2, !dbg !21
  %bnk.n1127 = fadd float %bnk.n1125, %bnk.n1123, !dbg !22
  %bnk.n1128 = extractelement <4 x float> %bnk.n1115, i64 3, !dbg !20
  %bnk.n1129 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1128), !dbg !20
  %bnk.n1130 = insertelement <4 x float> %bnk.n1126, float %bnk.n1129, i64 3, !dbg !21
  %math.result335 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1130) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n1131 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !21
  %math.packed336 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1131) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n1132 = fadd float %bnk.n1129, %bnk.n1127, !dbg !22
  %bnk.n1133 = shufflevector <32 x float> %bnk.n1000, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %bnk.n1134 = fadd <4 x float> %bnk.n698, %bnk.n1133, !dbg !19
  %bnk.n1135 = extractelement <4 x float> %bnk.n1134, i64 0, !dbg !20
  %bnk.n1136 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1135), !dbg !20
  %bnk.n1137 = insertelement <4 x float> poison, float %bnk.n1136, i64 0, !dbg !21
  %bnk.n1138 = fadd float %bnk.n1136, %bnk.n1132, !dbg !22
  %bnk.n1139 = extractelement <4 x float> %bnk.n1134, i64 1, !dbg !20
  %bnk.n1140 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1139), !dbg !20
  %bnk.n1141 = insertelement <4 x float> %bnk.n1137, float %bnk.n1140, i64 1, !dbg !21
  %bnk.n1142 = fadd float %bnk.n1140, %bnk.n1138, !dbg !22
  %bnk.n1143 = extractelement <4 x float> %bnk.n1134, i64 2, !dbg !20
  %bnk.n1144 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1143), !dbg !20
  %bnk.n1145 = insertelement <4 x float> %bnk.n1141, float %bnk.n1144, i64 2, !dbg !21
  %bnk.n1146 = fadd float %bnk.n1144, %bnk.n1142, !dbg !22
  %bnk.n1147 = extractelement <4 x float> %bnk.n1134, i64 3, !dbg !20
  %bnk.n1148 = tail call float @llvm.amdgcn.exp2.f32(float %bnk.n1147), !dbg !20
  %bnk.n1149 = insertelement <4 x float> %bnk.n1145, float %bnk.n1148, i64 3, !dbg !21
  %math.result337 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %bnk.n1149) #8, !dbg !21, !frisk.register_lifetimes !5
  %bnk.n1150 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !21
  %math.packed338 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %bnk.n1150) #8, !dbg !22, !frisk.register_lifetimes !5
  %bnk.n1151 = fadd float %bnk.n1148, %bnk.n1146, !dbg !22
  %bnk.n1152 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !22
  %bnk.n1153 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %bnk.n1152), !dbg !22
  %bnk.n1154 = add nuw nsw i32 %bnk.n1153, 64, !dbg !22
  %bnk.n1155 = xor i32 %bnk.n1153, 16, !dbg !22
  %bnk.n1156 = and i32 %bnk.n1154, 192, !dbg !22
  %bnk.n1157 = icmp samesign ult i32 %bnk.n1155, %bnk.n1156, !dbg !22
  %bnk.n1158 = select i1 %bnk.n1157, i32 %bnk.n1155, i32 %bnk.n1153, !dbg !22
  %bnk.n1159 = shl nuw nsw i32 %bnk.n1158, 2, !dbg !22
  %bnk.n1160 = bitcast float %bnk.n999 to i32, !dbg !22
  %bnk.n1161 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1159, i32 %bnk.n1160), !dbg !22
  %bnk.n1162 = bitcast i32 %bnk.n1161 to float, !dbg !22
  %bnk.n1163 = fadd float %bnk.n999, %bnk.n1162, !dbg !22
  %bnk.n1164 = xor i32 %bnk.n1153, 32, !dbg !22
  %bnk.n1165 = icmp samesign ult i32 %bnk.n1164, %bnk.n1156, !dbg !22
  %bnk.n1166 = select i1 %bnk.n1165, i32 %bnk.n1164, i32 %bnk.n1153, !dbg !22
  %bnk.n1167 = shl nuw nsw i32 %bnk.n1166, 2, !dbg !22
  %bnk.n1168 = bitcast float %bnk.n1163 to i32, !dbg !22
  %bnk.n1169 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1167, i32 %bnk.n1168), !dbg !22
  %bnk.n1170 = bitcast i32 %bnk.n1169 to float, !dbg !22
  %bnk.n1171 = fadd float %bnk.n1163, %bnk.n1170, !dbg !22
  %bnk.n1172 = extractvalue [2 x <1 x float>] %bnk.n20757, 0, !dbg !22
  %bnk.n1173 = extractelement <1 x float> %bnk.n1172, i64 0, !dbg !22
  %bnk.n1174 = fadd float %bnk.n1173, %bnk.n1171, !dbg !22
  %bnk.n1175 = insertelement <1 x float> poison, float %bnk.n1174, i64 0, !dbg !11
  %bnk.n1176 = insertvalue [2 x <1 x float>] poison, <1 x float> %bnk.n1175, 0, !dbg !11
  %bnk.n1177 = bitcast float %bnk.n1151 to i32, !dbg !22
  %bnk.n1178 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1159, i32 %bnk.n1177), !dbg !22
  %bnk.n1179 = bitcast i32 %bnk.n1178 to float, !dbg !22
  %bnk.n1180 = fadd float %bnk.n1151, %bnk.n1179, !dbg !22
  %bnk.n1181 = bitcast float %bnk.n1180 to i32, !dbg !22
  %bnk.n1182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1167, i32 %bnk.n1181), !dbg !22
  %bnk.n1183 = bitcast i32 %bnk.n1182 to float, !dbg !22
  %bnk.n1184 = fadd float %bnk.n1180, %bnk.n1183, !dbg !22
  %bnk.n1185 = extractvalue [2 x <1 x float>] %bnk.n20757, 1, !dbg !22
  %bnk.n1186 = extractelement <1 x float> %bnk.n1185, i64 0, !dbg !22
  %bnk.n1187 = fadd float %bnk.n1186, %bnk.n1184, !dbg !22
  %bnk.n1188 = insertelement <1 x float> poison, float %bnk.n1187, i64 0, !dbg !11
  %bnk.n1189 = insertvalue [2 x <1 x float>] %bnk.n1176, <1 x float> %bnk.n1188, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread864 = tail call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #8, !dbg !23, !range !9, !frisk.register_lifetimes !5
  %address.local865 = shl nuw nsw i32 %address.thread864, 3, !dbg !23
  %address.local866 = and i32 %address.local865, 120, !dbg !23
  %address.local867 = shl nuw nsw i32 %address.thread864, 5, !dbg !23
  %address.local868 = and i32 %address.local867, 1536, !dbg !23
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !23
  %address.local870 = zext nneg i32 %address.local869 to i64, !dbg !23
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !23
  %bnk.n1190 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !23
  %bnk.n1191 = bitcast <4 x i32> %bnk.n1190 to <8 x half>, !dbg !23
  %address.local872 = zext nneg i32 %address.thread864 to i64, !dbg !8
  %address.local873 = and i64 %address.local872, 60, !dbg !23
  %address.local874 = add nuw nsw i64 %address.local873, %address.local872, !dbg !23
  %address.local876.idx = shl nuw nsw i64 %address.local874, 4, !dbg !23
  %address.local876 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local876.idx, !dbg !23
  %address.local877 = lshr i64 %address.local872, 4, !dbg !23
  %address.local878 = and i64 %address.local877, 3, !dbg !23
  %address.local880.idx = mul nuw nsw i64 %address.local878, 768, !dbg !23
  %address.local880 = getelementptr i8, ptr addrspace(3) %address.local876, i64 %address.local880.idx, !dbg !23
  %address.local881 = lshr i64 %address.local872, 2, !dbg !15
  %address.local883.idx = mul nsw i64 %address.local881, -64, !dbg !23
  %address.local883 = getelementptr i8, ptr addrspace(3) %address.local880, i64 %address.local883.idx, !dbg !23
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 256, !dbg !23
  %bnk.n1192 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !23
  %bnk.n1193 = bitcast <4 x i32> %bnk.n1192 to <8 x half>, !dbg !23
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local883, i64 512, !dbg !23
  %bnk.n1194 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !23
  %bnk.n1195 = bitcast <4 x i32> %bnk.n1194 to <8 x half>, !dbg !23
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local883, i64 768, !dbg !23
  %bnk.n1196 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !23
  %bnk.n1197 = bitcast <4 x i32> %bnk.n1196 to <8 x half>, !dbg !23
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local871, i64 4096, !dbg !23
  %bnk.n1198 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !23
  %bnk.n1199 = bitcast <4 x i32> %bnk.n1198 to <8 x half>, !dbg !23
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4352, !dbg !23
  %bnk.n1200 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !23
  %bnk.n1201 = bitcast <4 x i32> %bnk.n1200 to <8 x half>, !dbg !23
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4608, !dbg !23
  %bnk.n1202 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !23
  %bnk.n1203 = bitcast <4 x i32> %bnk.n1202 to <8 x half>, !dbg !23
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4864, !dbg !23
  %bnk.n1204 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !23
  %bnk.n1205 = bitcast <4 x i32> %bnk.n1204 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !8
  %address.local908 = and i64 %address.local907, 1536, !dbg !8
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !8
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !8
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !8
  %bnk.n1212 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !17
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !17
  %address.local913 = and i64 %address.local912, 8064, !dbg !17
  %address.local915 = and i64 %address.local907, 24, !dbg !17
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !17
  %169 = or disjoint i64 %bnk.n603.idx27, %address.local916, !dbg !17
  %170 = trunc i64 %169 to i32, !dbg !17
  %.tr32 = shl i32 %170, 1, !dbg !17
  %dma.buffer.offset1558 = or i32 %.tr32, 64, !dbg !17
  %dma.buffer.lds1560 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n1212), !dbg !17
  %bnk.n1225 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !8
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !8
  %address.local920 = and i64 %address.local919, 3584, !dbg !8
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !8
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !8
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !8
  %bnk.n1226 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !17
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !17
  %address.local925 = and i64 %address.local924, 16256, !dbg !17
  %address.local928 = or disjoint i64 %address.local925, %address.local915, !dbg !17
  %171 = or disjoint i64 %bnk.n603.idx27, %address.local928, !dbg !17
  %172 = trunc i64 %171 to i32, !dbg !17
  %.tr35 = shl i32 %172, 1, !dbg !17
  %dma.buffer.offset1555 = or i32 %.tr35, 64, !dbg !17
  %dma.buffer.lds1557 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bnk.n1226), !dbg !17
  %bnk.n1239 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %bnk.n1240 = shufflevector <8 x half> %bnk.n1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1241 = extractvalue [2 x <8 x float>] %bnk.n20656, 0, !dbg !24
  %bnk.n1242 = shufflevector <8 x float> %bnk.n1241, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %bnk.n1243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1240, <4 x float> %bnk.n1242) #8, !dbg !23
  %bnk.n1244 = shufflevector <8 x half> %bnk.n1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1244, <4 x float> %bnk.n1243) #8, !dbg !23
  %bnk.n1246 = shufflevector <8 x half> %bnk.n1195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1246, <4 x float> %bnk.n1245) #8, !dbg !23
  %bnk.n1248 = shufflevector <8 x half> %bnk.n1197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1248, <4 x float> %bnk.n1247) #8, !dbg !23
  %bnk.n1250 = shufflevector <8 x half> %bnk.n1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1250, <4 x float> %bnk.n1249) #8, !dbg !23
  %bnk.n1252 = shufflevector <8 x half> %bnk.n1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1252, <4 x float> %bnk.n1251) #8, !dbg !23
  %bnk.n1254 = shufflevector <8 x half> %bnk.n1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1254, <4 x float> %bnk.n1253) #8, !dbg !23
  %bnk.n1256 = shufflevector <8 x half> %bnk.n1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1256, <4 x float> %bnk.n1255) #8, !dbg !23
  %bnk.n1258 = shufflevector <8 x half> %bnk.n1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1259 = shufflevector <8 x float> %bnk.n1241, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %bnk.n1260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1258, <4 x float> %bnk.n1259) #8, !dbg !23
  %bnk.n1261 = shufflevector <8 x half> %bnk.n1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1261, <4 x float> %bnk.n1260) #8, !dbg !23
  %bnk.n1263 = shufflevector <8 x half> %bnk.n1195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1263, <4 x float> %bnk.n1262) #8, !dbg !23
  %bnk.n1265 = shufflevector <8 x half> %bnk.n1197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1265, <4 x float> %bnk.n1264) #8, !dbg !23
  %bnk.n1267 = shufflevector <8 x half> %bnk.n1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1267, <4 x float> %bnk.n1266) #8, !dbg !23
  %bnk.n1269 = shufflevector <8 x half> %bnk.n1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1269, <4 x float> %bnk.n1268) #8, !dbg !23
  %bnk.n1271 = shufflevector <8 x half> %bnk.n1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1271, <4 x float> %bnk.n1270) #8, !dbg !23
  %bnk.n1273 = shufflevector <8 x half> %bnk.n1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1273, <4 x float> %bnk.n1272) #8, !dbg !23
  %bnk.n1275 = extractvalue [2 x <8 x float>] %bnk.n20656, 1, !dbg !24
  %bnk.n1276 = shufflevector <8 x float> %bnk.n1275, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %bnk.n1277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1240, <4 x float> %bnk.n1276) #8, !dbg !23
  %bnk.n1278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1244, <4 x float> %bnk.n1277) #8, !dbg !23
  %bnk.n1279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1246, <4 x float> %bnk.n1278) #8, !dbg !23
  %bnk.n1280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1248, <4 x float> %bnk.n1279) #8, !dbg !23
  %bnk.n1281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1250, <4 x float> %bnk.n1280) #8, !dbg !23
  %bnk.n1282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1252, <4 x float> %bnk.n1281) #8, !dbg !23
  %bnk.n1283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1254, <4 x float> %bnk.n1282) #8, !dbg !23
  %bnk.n1284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1256, <4 x float> %bnk.n1283) #8, !dbg !23
  %bnk.n1285 = shufflevector <8 x float> %bnk.n1275, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %bnk.n1286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1258, <4 x float> %bnk.n1285) #8, !dbg !23
  %bnk.n1287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1261, <4 x float> %bnk.n1286) #8, !dbg !23
  %bnk.n1288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1263, <4 x float> %bnk.n1287) #8, !dbg !23
  %bnk.n1289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1265, <4 x float> %bnk.n1288) #8, !dbg !23
  %bnk.n1290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1267, <4 x float> %bnk.n1289) #8, !dbg !23
  %bnk.n1291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1269, <4 x float> %bnk.n1290) #8, !dbg !23
  %bnk.n1292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1271, <4 x float> %bnk.n1291) #8, !dbg !23
  %bnk.n1293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1273, <4 x float> %bnk.n1292) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %bnk.n1294 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !23
  %bnk.n1295 = bitcast <4 x i32> %bnk.n1294 to <8 x half>, !dbg !23
  %bnk.n1296 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !23
  %bnk.n1297 = bitcast <4 x i32> %bnk.n1296 to <8 x half>, !dbg !23
  %bnk.n1298 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !23
  %bnk.n1299 = bitcast <4 x i32> %bnk.n1298 to <8 x half>, !dbg !23
  %bnk.n1300 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !23
  %bnk.n1301 = bitcast <4 x i32> %bnk.n1300 to <8 x half>, !dbg !23
  %bnk.n1302 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !23
  %bnk.n1303 = bitcast <4 x i32> %bnk.n1302 to <8 x half>, !dbg !23
  %bnk.n1304 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !23
  %bnk.n1305 = bitcast <4 x i32> %bnk.n1304 to <8 x half>, !dbg !23
  %bnk.n1306 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !23
  %bnk.n1307 = bitcast <4 x i32> %bnk.n1306 to <8 x half>, !dbg !23
  %bnk.n1308 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !23
  %bnk.n1309 = bitcast <4 x i32> %bnk.n1308 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %dma.buffer.offset1552 = or i32 %.tr32, 128, !dbg !17
  %bnk.n1325 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %dma.buffer.offset1549 = or i32 %.tr35, 128, !dbg !17
  %bnk.n1339 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %bnk.n1340 = shufflevector <8 x half> %bnk.n1295, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1341 = extractvalue [2 x <8 x float>] %bnk.n20858, 0, !dbg !24
  %bnk.n1342 = shufflevector <8 x float> %bnk.n1341, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %bnk.n1343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1340, <4 x float> %bnk.n1342) #8, !dbg !23
  %bnk.n1344 = shufflevector <8 x half> %bnk.n1297, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1344, <4 x float> %bnk.n1343) #8, !dbg !23
  %bnk.n1346 = shufflevector <8 x half> %bnk.n1299, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1346, <4 x float> %bnk.n1345) #8, !dbg !23
  %bnk.n1348 = shufflevector <8 x half> %bnk.n1301, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1348, <4 x float> %bnk.n1347) #8, !dbg !23
  %bnk.n1350 = shufflevector <8 x half> %bnk.n1303, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1350, <4 x float> %bnk.n1349) #8, !dbg !23
  %bnk.n1352 = shufflevector <8 x half> %bnk.n1305, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1352, <4 x float> %bnk.n1351) #8, !dbg !23
  %bnk.n1354 = shufflevector <8 x half> %bnk.n1307, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1354, <4 x float> %bnk.n1353) #8, !dbg !23
  %bnk.n1356 = shufflevector <8 x half> %bnk.n1309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1356, <4 x float> %bnk.n1355) #8, !dbg !23
  %bnk.n1358 = shufflevector <8 x half> %bnk.n1295, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1359 = shufflevector <8 x float> %bnk.n1341, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %bnk.n1360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1358, <4 x float> %bnk.n1359) #8, !dbg !23
  %bnk.n1361 = shufflevector <8 x half> %bnk.n1297, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1361, <4 x float> %bnk.n1360) #8, !dbg !23
  %bnk.n1363 = shufflevector <8 x half> %bnk.n1299, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1363, <4 x float> %bnk.n1362) #8, !dbg !23
  %bnk.n1365 = shufflevector <8 x half> %bnk.n1301, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1365, <4 x float> %bnk.n1364) #8, !dbg !23
  %bnk.n1367 = shufflevector <8 x half> %bnk.n1303, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1367, <4 x float> %bnk.n1366) #8, !dbg !23
  %bnk.n1369 = shufflevector <8 x half> %bnk.n1305, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1369, <4 x float> %bnk.n1368) #8, !dbg !23
  %bnk.n1371 = shufflevector <8 x half> %bnk.n1307, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1371, <4 x float> %bnk.n1370) #8, !dbg !23
  %bnk.n1373 = shufflevector <8 x half> %bnk.n1309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1373, <4 x float> %bnk.n1372) #8, !dbg !23
  %bnk.n1375 = extractvalue [2 x <8 x float>] %bnk.n20858, 1, !dbg !24
  %bnk.n1376 = shufflevector <8 x float> %bnk.n1375, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %bnk.n1377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1340, <4 x float> %bnk.n1376) #8, !dbg !23
  %bnk.n1378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1344, <4 x float> %bnk.n1377) #8, !dbg !23
  %bnk.n1379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1346, <4 x float> %bnk.n1378) #8, !dbg !23
  %bnk.n1380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1348, <4 x float> %bnk.n1379) #8, !dbg !23
  %bnk.n1381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1350, <4 x float> %bnk.n1380) #8, !dbg !23
  %bnk.n1382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1352, <4 x float> %bnk.n1381) #8, !dbg !23
  %bnk.n1383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1354, <4 x float> %bnk.n1382) #8, !dbg !23
  %bnk.n1384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1356, <4 x float> %bnk.n1383) #8, !dbg !23
  %bnk.n1385 = shufflevector <8 x float> %bnk.n1375, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %bnk.n1386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1358, <4 x float> %bnk.n1385) #8, !dbg !23
  %bnk.n1387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1361, <4 x float> %bnk.n1386) #8, !dbg !23
  %bnk.n1388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1363, <4 x float> %bnk.n1387) #8, !dbg !23
  %bnk.n1389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1365, <4 x float> %bnk.n1388) #8, !dbg !23
  %bnk.n1390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1367, <4 x float> %bnk.n1389) #8, !dbg !23
  %bnk.n1391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1369, <4 x float> %bnk.n1390) #8, !dbg !23
  %bnk.n1392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1371, <4 x float> %bnk.n1391) #8, !dbg !23
  %bnk.n1393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1373, <4 x float> %bnk.n1392) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %bnk.n1394 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !23
  %bnk.n1395 = bitcast <4 x i32> %bnk.n1394 to <8 x half>, !dbg !23
  %bnk.n1396 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !23
  %bnk.n1397 = bitcast <4 x i32> %bnk.n1396 to <8 x half>, !dbg !23
  %bnk.n1398 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !23
  %bnk.n1399 = bitcast <4 x i32> %bnk.n1398 to <8 x half>, !dbg !23
  %bnk.n1400 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !23
  %bnk.n1401 = bitcast <4 x i32> %bnk.n1400 to <8 x half>, !dbg !23
  %bnk.n1402 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !23
  %bnk.n1403 = bitcast <4 x i32> %bnk.n1402 to <8 x half>, !dbg !23
  %bnk.n1404 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !23
  %bnk.n1405 = bitcast <4 x i32> %bnk.n1404 to <8 x half>, !dbg !23
  %bnk.n1406 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !23
  %bnk.n1407 = bitcast <4 x i32> %bnk.n1406 to <8 x half>, !dbg !23
  %bnk.n1408 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !23
  %bnk.n1409 = bitcast <4 x i32> %bnk.n1408 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %173 = or disjoint i64 %bnk.n603.idx27, 96, !dbg !17
  %174 = or disjoint i64 %address.local916, %173, !dbg !17
  %.tr44 = trunc i64 %174 to i32, !dbg !17
  %dma.buffer.offset1546 = shl i32 %.tr44, 1, !dbg !17
  %bnk.n1425 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %175 = or disjoint i64 %address.local928, %173, !dbg !17
  %.tr47 = trunc i64 %175 to i32, !dbg !17
  %dma.buffer.offset = shl i32 %.tr47, 1, !dbg !17
  %bnk.n1439 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %bnk.n1440 = shufflevector <8 x half> %bnk.n1395, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1441 = extractvalue [2 x <8 x float>] %bnk.n20959, 0, !dbg !24
  %bnk.n1442 = shufflevector <8 x float> %bnk.n1441, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %bnk.n1443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1440, <4 x float> %bnk.n1442) #8, !dbg !23
  %bnk.n1444 = shufflevector <8 x half> %bnk.n1397, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1444, <4 x float> %bnk.n1443) #8, !dbg !23
  %bnk.n1446 = shufflevector <8 x half> %bnk.n1399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1446, <4 x float> %bnk.n1445) #8, !dbg !23
  %bnk.n1448 = shufflevector <8 x half> %bnk.n1401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1448, <4 x float> %bnk.n1447) #8, !dbg !23
  %bnk.n1450 = shufflevector <8 x half> %bnk.n1403, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1450, <4 x float> %bnk.n1449) #8, !dbg !23
  %bnk.n1452 = shufflevector <8 x half> %bnk.n1405, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1452, <4 x float> %bnk.n1451) #8, !dbg !23
  %bnk.n1454 = shufflevector <8 x half> %bnk.n1407, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1455 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1454, <4 x float> %bnk.n1453) #8, !dbg !23
  %bnk.n1456 = shufflevector <8 x half> %bnk.n1409, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1456, <4 x float> %bnk.n1455) #8, !dbg !23
  %bnk.n1458 = shufflevector <8 x half> %bnk.n1395, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1459 = shufflevector <8 x float> %bnk.n1441, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %bnk.n1460 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1458, <4 x float> %bnk.n1459) #8, !dbg !23
  %bnk.n1461 = shufflevector <8 x half> %bnk.n1397, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1461, <4 x float> %bnk.n1460) #8, !dbg !23
  %bnk.n1463 = shufflevector <8 x half> %bnk.n1399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1463, <4 x float> %bnk.n1462) #8, !dbg !23
  %bnk.n1465 = shufflevector <8 x half> %bnk.n1401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1466 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1465, <4 x float> %bnk.n1464) #8, !dbg !23
  %bnk.n1467 = shufflevector <8 x half> %bnk.n1403, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1468 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1467, <4 x float> %bnk.n1466) #8, !dbg !23
  %bnk.n1469 = shufflevector <8 x half> %bnk.n1405, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1469, <4 x float> %bnk.n1468) #8, !dbg !23
  %bnk.n1471 = shufflevector <8 x half> %bnk.n1407, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1472 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1471, <4 x float> %bnk.n1470) #8, !dbg !23
  %bnk.n1473 = shufflevector <8 x half> %bnk.n1409, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1474 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1473, <4 x float> %bnk.n1472) #8, !dbg !23
  %bnk.n1475 = extractvalue [2 x <8 x float>] %bnk.n20959, 1, !dbg !24
  %bnk.n1476 = shufflevector <8 x float> %bnk.n1475, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %bnk.n1477 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1440, <4 x float> %bnk.n1476) #8, !dbg !23
  %bnk.n1478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1444, <4 x float> %bnk.n1477) #8, !dbg !23
  %bnk.n1479 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1446, <4 x float> %bnk.n1478) #8, !dbg !23
  %bnk.n1480 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1448, <4 x float> %bnk.n1479) #8, !dbg !23
  %bnk.n1481 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1450, <4 x float> %bnk.n1480) #8, !dbg !23
  %bnk.n1482 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1452, <4 x float> %bnk.n1481) #8, !dbg !23
  %bnk.n1483 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1454, <4 x float> %bnk.n1482) #8, !dbg !23
  %bnk.n1484 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1456, <4 x float> %bnk.n1483) #8, !dbg !23
  %bnk.n1485 = shufflevector <8 x float> %bnk.n1475, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %bnk.n1486 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1458, <4 x float> %bnk.n1485) #8, !dbg !23
  %bnk.n1487 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1461, <4 x float> %bnk.n1486) #8, !dbg !23
  %bnk.n1488 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1463, <4 x float> %bnk.n1487) #8, !dbg !23
  %bnk.n1489 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1465, <4 x float> %bnk.n1488) #8, !dbg !23
  %bnk.n1490 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1467, <4 x float> %bnk.n1489) #8, !dbg !23
  %bnk.n1491 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1469, <4 x float> %bnk.n1490) #8, !dbg !23
  %bnk.n1492 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1471, <4 x float> %bnk.n1491) #8, !dbg !23
  %bnk.n1493 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1473, <4 x float> %bnk.n1492) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %bnk.n1494 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !23
  %bnk.n1495 = bitcast <4 x i32> %bnk.n1494 to <8 x half>, !dbg !23
  %bnk.n1496 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !23
  %bnk.n1497 = bitcast <4 x i32> %bnk.n1496 to <8 x half>, !dbg !23
  %bnk.n1498 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !23
  %bnk.n1499 = bitcast <4 x i32> %bnk.n1498 to <8 x half>, !dbg !23
  %bnk.n1500 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !23
  %bnk.n1501 = bitcast <4 x i32> %bnk.n1500 to <8 x half>, !dbg !23
  %bnk.n1502 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !23
  %bnk.n1503 = bitcast <4 x i32> %bnk.n1502 to <8 x half>, !dbg !23
  %bnk.n1504 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !23
  %bnk.n1505 = bitcast <4 x i32> %bnk.n1504 to <8 x half>, !dbg !23
  %bnk.n1506 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !23
  %bnk.n1507 = bitcast <4 x i32> %bnk.n1506 to <8 x half>, !dbg !23
  %bnk.n1508 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !23
  %bnk.n1509 = bitcast <4 x i32> %bnk.n1508 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %bnk.n1580 = add nuw nsw i64 %bnk.n20555, 1, !dbg !11
  %bnk.n1510 = shufflevector <8 x half> %bnk.n1495, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1511 = extractvalue [2 x <8 x float>] %bnk.n21060, 0, !dbg !24
  %bnk.n1512 = shufflevector <8 x float> %bnk.n1511, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %bnk.n1513 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1510, <4 x float> %bnk.n1512) #8, !dbg !23
  %bnk.n1514 = shufflevector <8 x half> %bnk.n1497, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1515 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1514, <4 x float> %bnk.n1513) #8, !dbg !23
  %bnk.n1516 = shufflevector <8 x half> %bnk.n1499, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1517 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1516, <4 x float> %bnk.n1515) #8, !dbg !23
  %bnk.n1518 = shufflevector <8 x half> %bnk.n1501, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1519 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1518, <4 x float> %bnk.n1517) #8, !dbg !23
  %bnk.n1520 = shufflevector <8 x half> %bnk.n1503, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1521 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1520, <4 x float> %bnk.n1519) #8, !dbg !23
  %bnk.n1522 = shufflevector <8 x half> %bnk.n1505, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1523 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1522, <4 x float> %bnk.n1521) #8, !dbg !23
  %bnk.n1524 = shufflevector <8 x half> %bnk.n1507, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1525 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1524, <4 x float> %bnk.n1523) #8, !dbg !23
  %bnk.n1526 = shufflevector <8 x half> %bnk.n1509, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %bnk.n1527 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1526, <4 x float> %bnk.n1525) #8, !dbg !23
  %bnk.n1528 = shufflevector <8 x half> %bnk.n1495, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1529 = shufflevector <8 x float> %bnk.n1511, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %bnk.n1530 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %bnk.n1528, <4 x float> %bnk.n1529) #8, !dbg !23
  %bnk.n1531 = shufflevector <8 x half> %bnk.n1497, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1532 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %bnk.n1531, <4 x float> %bnk.n1530) #8, !dbg !23
  %bnk.n1533 = shufflevector <8 x half> %bnk.n1499, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1534 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %bnk.n1533, <4 x float> %bnk.n1532) #8, !dbg !23
  %bnk.n1535 = shufflevector <8 x half> %bnk.n1501, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1536 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %bnk.n1535, <4 x float> %bnk.n1534) #8, !dbg !23
  %bnk.n1537 = shufflevector <8 x half> %bnk.n1503, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1538 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %bnk.n1537, <4 x float> %bnk.n1536) #8, !dbg !23
  %bnk.n1539 = shufflevector <8 x half> %bnk.n1505, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1540 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %bnk.n1539, <4 x float> %bnk.n1538) #8, !dbg !23
  %bnk.n1541 = shufflevector <8 x half> %bnk.n1507, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %bnk.n1541, <4 x float> %bnk.n1540) #8, !dbg !23
  %bnk.n1543 = shufflevector <8 x half> %bnk.n1509, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %bnk.n1544 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %bnk.n1543, <4 x float> %bnk.n1542) #8, !dbg !23
  %bnk.n1545 = extractvalue [2 x <8 x float>] %bnk.n21060, 1, !dbg !24
  %bnk.n1546 = shufflevector <8 x float> %bnk.n1545, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %bnk.n1547 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1510, <4 x float> %bnk.n1546) #8, !dbg !23
  %bnk.n1548 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1514, <4 x float> %bnk.n1547) #8, !dbg !23
  %bnk.n1549 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1516, <4 x float> %bnk.n1548) #8, !dbg !23
  %bnk.n1550 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1518, <4 x float> %bnk.n1549) #8, !dbg !23
  %bnk.n1551 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1520, <4 x float> %bnk.n1550) #8, !dbg !23
  %bnk.n1552 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1522, <4 x float> %bnk.n1551) #8, !dbg !23
  %bnk.n1553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1524, <4 x float> %bnk.n1552) #8, !dbg !23
  %bnk.n1554 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1526, <4 x float> %bnk.n1553) #8, !dbg !23
  %bnk.n1555 = shufflevector <8 x float> %bnk.n1545, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %bnk.n1556 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %bnk.n1528, <4 x float> %bnk.n1555) #8, !dbg !23
  %bnk.n1557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %bnk.n1531, <4 x float> %bnk.n1556) #8, !dbg !23
  %bnk.n1558 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %bnk.n1533, <4 x float> %bnk.n1557) #8, !dbg !23
  %bnk.n1559 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %bnk.n1535, <4 x float> %bnk.n1558) #8, !dbg !23
  %bnk.n1560 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %bnk.n1537, <4 x float> %bnk.n1559) #8, !dbg !23
  %bnk.n1561 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %bnk.n1539, <4 x float> %bnk.n1560) #8, !dbg !23
  %bnk.n1562 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %bnk.n1541, <4 x float> %bnk.n1561) #8, !dbg !23
  %bnk.n1563 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %bnk.n1543, <4 x float> %bnk.n1562) #8, !dbg !23
  %bnk.n1564 = shufflevector <4 x float> %bnk.n1257, <4 x float> %bnk.n1274, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %bnk.n1565 = insertvalue [2 x <8 x float>] poison, <8 x float> %bnk.n1564, 0, !dbg !11
  %bnk.n1566 = shufflevector <4 x float> %bnk.n1284, <4 x float> %bnk.n1293, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %bnk.n1567 = insertvalue [2 x <8 x float>] %bnk.n1565, <8 x float> %bnk.n1566, 1, !dbg !11
  %bnk.n1568 = shufflevector <4 x float> %bnk.n1357, <4 x float> %bnk.n1374, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %bnk.n1569 = insertvalue [2 x <8 x float>] poison, <8 x float> %bnk.n1568, 0, !dbg !11
  %bnk.n1570 = shufflevector <4 x float> %bnk.n1384, <4 x float> %bnk.n1393, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %bnk.n1571 = insertvalue [2 x <8 x float>] %bnk.n1569, <8 x float> %bnk.n1570, 1, !dbg !11
  %bnk.n1572 = shufflevector <4 x float> %bnk.n1457, <4 x float> %bnk.n1474, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %bnk.n1573 = insertvalue [2 x <8 x float>] poison, <8 x float> %bnk.n1572, 0, !dbg !11
  %bnk.n1574 = shufflevector <4 x float> %bnk.n1484, <4 x float> %bnk.n1493, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %bnk.n1575 = insertvalue [2 x <8 x float>] %bnk.n1573, <8 x float> %bnk.n1574, 1, !dbg !11
  %bnk.n1576 = shufflevector <4 x float> %bnk.n1527, <4 x float> %bnk.n1544, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %bnk.n1577 = insertvalue [2 x <8 x float>] poison, <8 x float> %bnk.n1576, 0, !dbg !11
  %bnk.n1578 = shufflevector <4 x float> %bnk.n1554, <4 x float> %bnk.n1563, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %bnk.n1579 = insertvalue [2 x <8 x float>] %bnk.n1577, <8 x float> %bnk.n1578, 1, !dbg !11
  %k0.has.next = icmp slt i64 %bnk.n1580, %bnk.n191
  br i1 %k0.has.next, label %k0.join, label %bnk.n1581

k0.join:                                          ; preds = %bnk.n847
  %k0.next.address.thread433 = tail call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #8, !dbg !14, !range !9, !frisk.register_lifetimes !5
  %k0.next.address.local434 = zext nneg i32 %k0.next.address.thread433 to i64, !dbg !8
  %k0.next.address.local435 = shl nuw nsw i64 %k0.next.address.local434, 3, !dbg !8
  %k0.next.address.local436 = and i64 %k0.next.address.local435, 1536, !dbg !8
  %k0.next.address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.next.address.local436, !dbg !8
  %k0.next.address.local438 = ptrtoint ptr addrspace(3) %k0.next.address.local437 to i64, !dbg !8
  %k0.next.address.local439 = trunc i64 %k0.next.address.local438 to i32, !dbg !8
  %k0.next.bnk.n226 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.address.local439), !dbg !14
  %k0.next.dma.buffer.lds1590 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.bnk.n226), !dbg !14
  %k0.next.bnk.n214 = lshr i64 %bnk.n1580, 12, !dbg !14
  %k0.next.bnk.n215 = add nuw nsw i64 %k0.next.bnk.n214, %bnk.n16, !dbg !14
  %k0.next.bnk.n218 = shl i64 %k0.next.bnk.n215, 19, !dbg !14
  %k0.next.bnk.n219 = and i64 %k0.next.bnk.n218, 16252928, !dbg !14
  %k0.next.bnk.n216 = lshr i64 %bnk.n1580, 5, !dbg !14
  %k0.next.bnk.n217 = and i64 %k0.next.bnk.n216, 127, !dbg !14
  %k0.next.bnk.n221 = shl nuw nsw i64 %bnk.n1580, 14, !dbg !14
  %k0.next.address.local442 = and i64 %k0.next.address.local434, 123, !dbg !14
  %k0.next.address.local443 = lshr i64 %k0.next.address.local434, 2, !dbg !14
  %k0.next.address.local444 = and i64 %k0.next.address.local443, 4, !dbg !14
  %k0.next.address.local445 = add nuw nsw i64 %k0.next.address.local444, %k0.next.address.local434, !dbg !14
  %k0.next.address.local446 = and i64 %k0.next.address.local445, 4, !dbg !14
  %k0.next.address.local447 = or disjoint i64 %k0.next.address.local446, %k0.next.address.local442, !dbg !14
  %176 = shl i64 %k0.next.bnk.n216, 19, !dbg !14
  %k0.next.bnk.n223.idx49 = sub i64 %k0.next.bnk.n221, %176, !dbg !14
  %k0.next.bnk.n222 = or disjoint i64 %k0.next.bnk.n223.idx49, %k0.next.bnk.n217, !dbg !14
  %177 = add i64 %k0.next.bnk.n222, %k0.next.bnk.n219, !dbg !14
  %178 = shl i64 %177, 1, !dbg !14
  %179 = lshr i64 %k0.next.address.local434, 3, !dbg !14
  %.idx48 = and i64 %179, 16, !dbg !14
  %k0.next.address.local450.idx = shl nuw nsw i64 %k0.next.address.local447, 8, !dbg !14
  %180 = add nuw nsw i64 %.idx48, %178, !dbg !14
  %k0.next.bnk.n231 = add i64 %180, %k0.next.address.local450.idx, !dbg !14
  %k0.next.dma.buffer.offset1588 = trunc i64 %k0.next.bnk.n231 to i32, !dbg !14
  %k0.next.bnk.n239 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.next.dma.buffer.lds1590, i32 %k0.next.dma.buffer.offset1588, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #9, !dbg !14
  %k0.next.address.local451 = or disjoint i64 %k0.next.address.local434, 256, !dbg !8
  %k0.next.address.local452 = shl nuw nsw i64 %k0.next.address.local451, 3, !dbg !8
  %k0.next.address.local453 = and i64 %k0.next.address.local452, 3584, !dbg !8
  %k0.next.address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.next.address.local453, !dbg !8
  %k0.next.address.local455 = ptrtoint ptr addrspace(3) %k0.next.address.local454 to i64, !dbg !8
  %k0.next.address.local456 = trunc i64 %k0.next.address.local455 to i32, !dbg !8
  %k0.next.bnk.n240 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.address.local456), !dbg !14
  %k0.next.dma.buffer.lds1587 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.next.bnk.n240), !dbg !14
  %181 = lshr i64 %k0.next.address.local451, 3, !dbg !14
  %.idx50 = and i64 %181, 48, !dbg !14
  %182 = add nuw nsw i64 %.idx50, %178, !dbg !14
  %k0.next.bnk.n245 = add i64 %182, %k0.next.address.local450.idx, !dbg !14
  %k0.next.dma.buffer.offset1585 = trunc i64 %k0.next.bnk.n245 to i32, !dbg !14
  %k0.next.bnk.n253 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.next.dma.buffer.lds1587, i32 %k0.next.dma.buffer.offset1585, <4 x i32> %k0.prime.dma.buffer.rsrc1589) #9, !dbg !14
  br label %bnk.n212, !dbg !11

bnk.n1581:                                        ; preds = %bnk.n847, %.bnk.n1581_crit_edge
  %bnk.n1610.pre-phi = phi i32 [ %.pre73, %.bnk.n1581_crit_edge ], [ %bnk.n1164, %bnk.n847 ], !dbg !11
  %bnk.n1594.pre-phi = phi i32 [ %.pre72, %.bnk.n1581_crit_edge ], [ %bnk.n1156, %bnk.n847 ], !dbg !11
  %bnk.n1593.pre-phi = phi i32 [ %.pre71, %.bnk.n1581_crit_edge ], [ %bnk.n1155, %bnk.n847 ], !dbg !11
  %bnk.n1591.pre-phi = phi i32 [ %.pre69, %.bnk.n1581_crit_edge ], [ %bnk.n1153, %bnk.n847 ], !dbg !11
  %bnk.n206.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.bnk.n1581_crit_edge ], [ %bnk.n1567, %bnk.n847 ], !dbg !11
  %bnk.n207.lcssa = phi [2 x <1 x float>] [ zeroinitializer, %.bnk.n1581_crit_edge ], [ %bnk.n1189, %bnk.n847 ], !dbg !11
  %bnk.n208.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.bnk.n1581_crit_edge ], [ %bnk.n1571, %bnk.n847 ], !dbg !11
  %bnk.n209.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.bnk.n1581_crit_edge ], [ %bnk.n1575, %bnk.n847 ], !dbg !11
  %bnk.n210.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.bnk.n1581_crit_edge ], [ %bnk.n1579, %bnk.n847 ], !dbg !11
  %bnk.n1582 = and i32 %bnk.n31, 16, !dbg !11
  %.not = icmp eq i32 %bnk.n1582, 0, !dbg !11
  %bnk.n1583 = and i32 %bnk.n31, 32, !dbg !11
  %.not36 = icmp eq i32 %bnk.n1583, 0, !dbg !11
  %bnk.n1584 = extractvalue [2 x <8 x float>] %bnk.n210.lcssa, 0, !dbg !11
  %bnk.n1585 = extractelement <8 x float> %bnk.n1584, i64 0, !dbg !11
  %bnk.n1586 = extractelement <8 x float> %bnk.n1584, i64 1, !dbg !11
  %bnk.n1587 = extractelement <8 x float> %bnk.n1584, i64 2, !dbg !11
  %bnk.n1588 = extractelement <8 x float> %bnk.n1584, i64 3, !dbg !11
  %bnk.n1589 = select i1 %.not, float %bnk.n1586, float %bnk.n1585, !dbg !11
  %bnk.n1595 = icmp samesign ult i32 %bnk.n1593.pre-phi, %bnk.n1594.pre-phi, !dbg !11
  %bnk.n1596 = select i1 %bnk.n1595, i32 %bnk.n1593.pre-phi, i32 %bnk.n1591.pre-phi, !dbg !11
  %bnk.n1597 = shl nuw nsw i32 %bnk.n1596, 2, !dbg !11
  %bnk.n1598 = bitcast float %bnk.n1589 to i32, !dbg !11
  %bnk.n1599 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1598), !dbg !11
  %bnk.n1600 = bitcast i32 %bnk.n1599 to float, !dbg !11
  %bnk.n1601 = select i1 %.not, float %bnk.n1585, float %bnk.n1600, !dbg !11
  %bnk.n1602 = select i1 %.not, float %bnk.n1600, float %bnk.n1586, !dbg !11
  %bnk.n1603 = select i1 %.not, float %bnk.n1588, float %bnk.n1587, !dbg !11
  %bnk.n1604 = bitcast float %bnk.n1603 to i32, !dbg !11
  %bnk.n1605 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1604), !dbg !11
  %bnk.n1606 = bitcast i32 %bnk.n1605 to float, !dbg !11
  %bnk.n1607 = select i1 %.not, float %bnk.n1587, float %bnk.n1606, !dbg !11
  %bnk.n1608 = select i1 %.not, float %bnk.n1606, float %bnk.n1588, !dbg !11
  %bnk.n1609 = select i1 %.not36, float %bnk.n1607, float %bnk.n1601, !dbg !11
  %bnk.n1611 = icmp samesign ult i32 %bnk.n1610.pre-phi, %bnk.n1594.pre-phi, !dbg !11
  %bnk.n1612 = select i1 %bnk.n1611, i32 %bnk.n1610.pre-phi, i32 %bnk.n1591.pre-phi, !dbg !11
  %bnk.n1613 = shl nuw nsw i32 %bnk.n1612, 2, !dbg !11
  %bnk.n1614 = bitcast float %bnk.n1609 to i32, !dbg !11
  %bnk.n1615 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1614), !dbg !11
  %bnk.n1616 = bitcast i32 %bnk.n1615 to float, !dbg !11
  %bnk.n1619 = select i1 %.not36, float %bnk.n1608, float %bnk.n1602, !dbg !11
  %bnk.n1620 = bitcast float %bnk.n1619 to i32, !dbg !11
  %bnk.n1621 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1620), !dbg !11
  %bnk.n1622 = bitcast i32 %bnk.n1621 to float, !dbg !11
  %183 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %184 = shufflevector <4 x i1> %183, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %185 = insertelement <4 x float> poison, float %bnk.n1601, i64 0, !dbg !11
  %186 = insertelement <4 x float> %185, float %bnk.n1602, i64 1, !dbg !11
  %187 = insertelement <4 x float> %186, float %bnk.n1616, i64 2, !dbg !11
  %188 = insertelement <4 x float> %187, float %bnk.n1622, i64 3, !dbg !11
  %189 = insertelement <4 x float> poison, float %bnk.n1616, i64 0, !dbg !11
  %190 = insertelement <4 x float> %189, float %bnk.n1622, i64 1, !dbg !11
  %191 = insertelement <4 x float> %190, float %bnk.n1607, i64 2, !dbg !11
  %192 = insertelement <4 x float> %191, float %bnk.n1608, i64 3, !dbg !11
  %193 = select <4 x i1> %184, <4 x float> %188, <4 x float> %192, !dbg !11
  %bnk.n1625 = extractvalue [2 x <8 x float>] %bnk.n209.lcssa, 0, !dbg !11
  %bnk.n1626 = extractelement <8 x float> %bnk.n1625, i64 0, !dbg !11
  %bnk.n1627 = extractelement <8 x float> %bnk.n1625, i64 1, !dbg !11
  %bnk.n1628 = extractelement <8 x float> %bnk.n1625, i64 2, !dbg !11
  %bnk.n1629 = extractelement <8 x float> %bnk.n1625, i64 3, !dbg !11
  %bnk.n1630 = select i1 %.not, float %bnk.n1627, float %bnk.n1626, !dbg !11
  %bnk.n1631 = bitcast float %bnk.n1630 to i32, !dbg !11
  %bnk.n1632 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1631), !dbg !11
  %bnk.n1633 = bitcast i32 %bnk.n1632 to float, !dbg !11
  %bnk.n1634 = select i1 %.not, float %bnk.n1626, float %bnk.n1633, !dbg !11
  %bnk.n1635 = select i1 %.not, float %bnk.n1633, float %bnk.n1627, !dbg !11
  %bnk.n1636 = select i1 %.not, float %bnk.n1629, float %bnk.n1628, !dbg !11
  %bnk.n1637 = bitcast float %bnk.n1636 to i32, !dbg !11
  %bnk.n1638 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1637), !dbg !11
  %bnk.n1639 = bitcast i32 %bnk.n1638 to float, !dbg !11
  %bnk.n1640 = select i1 %.not, float %bnk.n1628, float %bnk.n1639, !dbg !11
  %bnk.n1641 = select i1 %.not, float %bnk.n1639, float %bnk.n1629, !dbg !11
  %bnk.n1642 = select i1 %.not36, float %bnk.n1640, float %bnk.n1634, !dbg !11
  %bnk.n1643 = bitcast float %bnk.n1642 to i32, !dbg !11
  %bnk.n1644 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1643), !dbg !11
  %bnk.n1645 = bitcast i32 %bnk.n1644 to float, !dbg !11
  %bnk.n1648 = select i1 %.not36, float %bnk.n1641, float %bnk.n1635, !dbg !11
  %bnk.n1649 = bitcast float %bnk.n1648 to i32, !dbg !11
  %bnk.n1650 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1649), !dbg !11
  %bnk.n1651 = bitcast i32 %bnk.n1650 to float, !dbg !11
  %194 = insertelement <4 x float> poison, float %bnk.n1634, i64 0, !dbg !11
  %195 = insertelement <4 x float> %194, float %bnk.n1635, i64 1, !dbg !11
  %196 = insertelement <4 x float> %195, float %bnk.n1645, i64 2, !dbg !11
  %197 = insertelement <4 x float> %196, float %bnk.n1651, i64 3, !dbg !11
  %198 = insertelement <4 x float> poison, float %bnk.n1645, i64 0, !dbg !11
  %199 = insertelement <4 x float> %198, float %bnk.n1651, i64 1, !dbg !11
  %200 = insertelement <4 x float> %199, float %bnk.n1640, i64 2, !dbg !11
  %201 = insertelement <4 x float> %200, float %bnk.n1641, i64 3, !dbg !11
  %202 = select <4 x i1> %184, <4 x float> %197, <4 x float> %201, !dbg !11
  %bnk.n1654 = extractvalue [2 x <8 x float>] %bnk.n208.lcssa, 0, !dbg !11
  %bnk.n1655 = extractelement <8 x float> %bnk.n1654, i64 0, !dbg !11
  %bnk.n1656 = extractelement <8 x float> %bnk.n1654, i64 1, !dbg !11
  %bnk.n1657 = extractelement <8 x float> %bnk.n1654, i64 2, !dbg !11
  %bnk.n1658 = extractelement <8 x float> %bnk.n1654, i64 3, !dbg !11
  %bnk.n1659 = select i1 %.not, float %bnk.n1656, float %bnk.n1655, !dbg !11
  %bnk.n1660 = bitcast float %bnk.n1659 to i32, !dbg !11
  %bnk.n1661 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1660), !dbg !11
  %bnk.n1662 = bitcast i32 %bnk.n1661 to float, !dbg !11
  %bnk.n1663 = select i1 %.not, float %bnk.n1655, float %bnk.n1662, !dbg !11
  %bnk.n1664 = select i1 %.not, float %bnk.n1662, float %bnk.n1656, !dbg !11
  %bnk.n1665 = select i1 %.not, float %bnk.n1658, float %bnk.n1657, !dbg !11
  %bnk.n1666 = bitcast float %bnk.n1665 to i32, !dbg !11
  %bnk.n1667 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1666), !dbg !11
  %bnk.n1668 = bitcast i32 %bnk.n1667 to float, !dbg !11
  %bnk.n1669 = select i1 %.not, float %bnk.n1657, float %bnk.n1668, !dbg !11
  %bnk.n1670 = select i1 %.not, float %bnk.n1668, float %bnk.n1658, !dbg !11
  %bnk.n1671 = select i1 %.not36, float %bnk.n1669, float %bnk.n1663, !dbg !11
  %bnk.n1672 = bitcast float %bnk.n1671 to i32, !dbg !11
  %bnk.n1673 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1672), !dbg !11
  %bnk.n1674 = bitcast i32 %bnk.n1673 to float, !dbg !11
  %bnk.n1677 = select i1 %.not36, float %bnk.n1670, float %bnk.n1664, !dbg !11
  %bnk.n1678 = bitcast float %bnk.n1677 to i32, !dbg !11
  %bnk.n1679 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1678), !dbg !11
  %bnk.n1680 = bitcast i32 %bnk.n1679 to float, !dbg !11
  %203 = insertelement <4 x float> poison, float %bnk.n1663, i64 0, !dbg !11
  %204 = insertelement <4 x float> %203, float %bnk.n1664, i64 1, !dbg !11
  %205 = insertelement <4 x float> %204, float %bnk.n1674, i64 2, !dbg !11
  %206 = insertelement <4 x float> %205, float %bnk.n1680, i64 3, !dbg !11
  %207 = insertelement <4 x float> poison, float %bnk.n1674, i64 0, !dbg !11
  %208 = insertelement <4 x float> %207, float %bnk.n1680, i64 1, !dbg !11
  %209 = insertelement <4 x float> %208, float %bnk.n1669, i64 2, !dbg !11
  %210 = insertelement <4 x float> %209, float %bnk.n1670, i64 3, !dbg !11
  %211 = select <4 x i1> %184, <4 x float> %206, <4 x float> %210, !dbg !11
  %bnk.n1683 = extractvalue [2 x <8 x float>] %bnk.n206.lcssa, 0, !dbg !11
  %bnk.n1684 = extractelement <8 x float> %bnk.n1683, i64 0, !dbg !11
  %bnk.n1685 = extractelement <8 x float> %bnk.n1683, i64 1, !dbg !11
  %bnk.n1686 = extractelement <8 x float> %bnk.n1683, i64 2, !dbg !11
  %bnk.n1687 = extractelement <8 x float> %bnk.n1683, i64 3, !dbg !11
  %bnk.n1688 = select i1 %.not, float %bnk.n1685, float %bnk.n1684, !dbg !11
  %bnk.n1689 = bitcast float %bnk.n1688 to i32, !dbg !11
  %bnk.n1690 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1689), !dbg !11
  %bnk.n1691 = bitcast i32 %bnk.n1690 to float, !dbg !11
  %bnk.n1692 = select i1 %.not, float %bnk.n1684, float %bnk.n1691, !dbg !11
  %bnk.n1693 = select i1 %.not, float %bnk.n1691, float %bnk.n1685, !dbg !11
  %bnk.n1694 = select i1 %.not, float %bnk.n1687, float %bnk.n1686, !dbg !11
  %bnk.n1695 = bitcast float %bnk.n1694 to i32, !dbg !11
  %bnk.n1696 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1695), !dbg !11
  %bnk.n1697 = bitcast i32 %bnk.n1696 to float, !dbg !11
  %bnk.n1698 = select i1 %.not, float %bnk.n1686, float %bnk.n1697, !dbg !11
  %bnk.n1699 = select i1 %.not, float %bnk.n1697, float %bnk.n1687, !dbg !11
  %bnk.n1700 = select i1 %.not36, float %bnk.n1698, float %bnk.n1692, !dbg !11
  %bnk.n1701 = bitcast float %bnk.n1700 to i32, !dbg !11
  %bnk.n1702 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1701), !dbg !11
  %bnk.n1703 = select i1 %.not36, float %bnk.n1699, float %bnk.n1693, !dbg !11
  %bnk.n1704 = bitcast float %bnk.n1703 to i32, !dbg !11
  %bnk.n1705 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1704), !dbg !11
  %bnk.n1706 = extractvalue [2 x <1 x float>] %bnk.n207.lcssa, 0, !dbg !25
  %bnk.n1707 = shufflevector <1 x float> %bnk.n1706, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %bnk.n1708 = extractelement <8 x float> %bnk.n1584, i64 4, !dbg !11
  %bnk.n1709 = extractelement <8 x float> %bnk.n1584, i64 5, !dbg !11
  %bnk.n1710 = extractelement <8 x float> %bnk.n1584, i64 6, !dbg !11
  %bnk.n1711 = extractelement <8 x float> %bnk.n1584, i64 7, !dbg !11
  %bnk.n1712 = select i1 %.not, float %bnk.n1709, float %bnk.n1708, !dbg !11
  %bnk.n1713 = bitcast float %bnk.n1712 to i32, !dbg !11
  %bnk.n1714 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1713), !dbg !11
  %bnk.n1715 = bitcast i32 %bnk.n1714 to float, !dbg !11
  %bnk.n1716 = select i1 %.not, float %bnk.n1708, float %bnk.n1715, !dbg !11
  %bnk.n1717 = select i1 %.not, float %bnk.n1715, float %bnk.n1709, !dbg !11
  %bnk.n1718 = select i1 %.not, float %bnk.n1711, float %bnk.n1710, !dbg !11
  %bnk.n1719 = bitcast float %bnk.n1718 to i32, !dbg !11
  %bnk.n1720 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1719), !dbg !11
  %bnk.n1721 = bitcast i32 %bnk.n1720 to float, !dbg !11
  %bnk.n1722 = select i1 %.not, float %bnk.n1710, float %bnk.n1721, !dbg !11
  %bnk.n1723 = select i1 %.not, float %bnk.n1721, float %bnk.n1711, !dbg !11
  %bnk.n1724 = select i1 %.not36, float %bnk.n1722, float %bnk.n1716, !dbg !11
  %bnk.n1725 = bitcast float %bnk.n1724 to i32, !dbg !11
  %bnk.n1726 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1725), !dbg !11
  %bnk.n1727 = bitcast i32 %bnk.n1726 to float, !dbg !11
  %bnk.n1730 = select i1 %.not36, float %bnk.n1723, float %bnk.n1717, !dbg !11
  %bnk.n1731 = bitcast float %bnk.n1730 to i32, !dbg !11
  %bnk.n1732 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1731), !dbg !11
  %bnk.n1733 = bitcast i32 %bnk.n1732 to float, !dbg !11
  %212 = insertelement <4 x float> poison, float %bnk.n1716, i64 0, !dbg !11
  %213 = insertelement <4 x float> %212, float %bnk.n1717, i64 1, !dbg !11
  %214 = insertelement <4 x float> %213, float %bnk.n1727, i64 2, !dbg !11
  %215 = insertelement <4 x float> %214, float %bnk.n1733, i64 3, !dbg !11
  %216 = insertelement <4 x float> poison, float %bnk.n1727, i64 0, !dbg !11
  %217 = insertelement <4 x float> %216, float %bnk.n1733, i64 1, !dbg !11
  %218 = insertelement <4 x float> %217, float %bnk.n1722, i64 2, !dbg !11
  %219 = insertelement <4 x float> %218, float %bnk.n1723, i64 3, !dbg !11
  %220 = select <4 x i1> %184, <4 x float> %215, <4 x float> %219, !dbg !11
  %bnk.n1736 = extractelement <8 x float> %bnk.n1625, i64 4, !dbg !11
  %bnk.n1737 = extractelement <8 x float> %bnk.n1625, i64 5, !dbg !11
  %bnk.n1738 = extractelement <8 x float> %bnk.n1625, i64 6, !dbg !11
  %bnk.n1739 = extractelement <8 x float> %bnk.n1625, i64 7, !dbg !11
  %bnk.n1740 = select i1 %.not, float %bnk.n1737, float %bnk.n1736, !dbg !11
  %bnk.n1741 = bitcast float %bnk.n1740 to i32, !dbg !11
  %bnk.n1742 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1741), !dbg !11
  %bnk.n1743 = bitcast i32 %bnk.n1742 to float, !dbg !11
  %bnk.n1744 = select i1 %.not, float %bnk.n1736, float %bnk.n1743, !dbg !11
  %bnk.n1745 = select i1 %.not, float %bnk.n1743, float %bnk.n1737, !dbg !11
  %bnk.n1746 = select i1 %.not, float %bnk.n1739, float %bnk.n1738, !dbg !11
  %bnk.n1747 = bitcast float %bnk.n1746 to i32, !dbg !11
  %bnk.n1748 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1747), !dbg !11
  %bnk.n1749 = bitcast i32 %bnk.n1748 to float, !dbg !11
  %bnk.n1750 = select i1 %.not, float %bnk.n1738, float %bnk.n1749, !dbg !11
  %bnk.n1751 = select i1 %.not, float %bnk.n1749, float %bnk.n1739, !dbg !11
  %bnk.n1752 = select i1 %.not36, float %bnk.n1750, float %bnk.n1744, !dbg !11
  %bnk.n1753 = bitcast float %bnk.n1752 to i32, !dbg !11
  %bnk.n1754 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1753), !dbg !11
  %bnk.n1755 = bitcast i32 %bnk.n1754 to float, !dbg !11
  %bnk.n1758 = select i1 %.not36, float %bnk.n1751, float %bnk.n1745, !dbg !11
  %bnk.n1759 = bitcast float %bnk.n1758 to i32, !dbg !11
  %bnk.n1760 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1759), !dbg !11
  %bnk.n1761 = bitcast i32 %bnk.n1760 to float, !dbg !11
  %221 = insertelement <4 x float> poison, float %bnk.n1744, i64 0, !dbg !11
  %222 = insertelement <4 x float> %221, float %bnk.n1745, i64 1, !dbg !11
  %223 = insertelement <4 x float> %222, float %bnk.n1755, i64 2, !dbg !11
  %224 = insertelement <4 x float> %223, float %bnk.n1761, i64 3, !dbg !11
  %225 = insertelement <4 x float> poison, float %bnk.n1755, i64 0, !dbg !11
  %226 = insertelement <4 x float> %225, float %bnk.n1761, i64 1, !dbg !11
  %227 = insertelement <4 x float> %226, float %bnk.n1750, i64 2, !dbg !11
  %228 = insertelement <4 x float> %227, float %bnk.n1751, i64 3, !dbg !11
  %229 = select <4 x i1> %184, <4 x float> %224, <4 x float> %228, !dbg !11
  %bnk.n1764 = extractelement <8 x float> %bnk.n1654, i64 4, !dbg !11
  %bnk.n1765 = extractelement <8 x float> %bnk.n1654, i64 5, !dbg !11
  %bnk.n1766 = extractelement <8 x float> %bnk.n1654, i64 6, !dbg !11
  %bnk.n1767 = extractelement <8 x float> %bnk.n1654, i64 7, !dbg !11
  %bnk.n1768 = select i1 %.not, float %bnk.n1765, float %bnk.n1764, !dbg !11
  %bnk.n1769 = bitcast float %bnk.n1768 to i32, !dbg !11
  %bnk.n1770 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1769), !dbg !11
  %bnk.n1771 = bitcast i32 %bnk.n1770 to float, !dbg !11
  %bnk.n1772 = select i1 %.not, float %bnk.n1764, float %bnk.n1771, !dbg !11
  %bnk.n1773 = select i1 %.not, float %bnk.n1771, float %bnk.n1765, !dbg !11
  %bnk.n1774 = select i1 %.not, float %bnk.n1767, float %bnk.n1766, !dbg !11
  %bnk.n1775 = bitcast float %bnk.n1774 to i32, !dbg !11
  %bnk.n1776 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1775), !dbg !11
  %bnk.n1777 = bitcast i32 %bnk.n1776 to float, !dbg !11
  %bnk.n1778 = select i1 %.not, float %bnk.n1766, float %bnk.n1777, !dbg !11
  %bnk.n1779 = select i1 %.not, float %bnk.n1777, float %bnk.n1767, !dbg !11
  %bnk.n1780 = select i1 %.not36, float %bnk.n1778, float %bnk.n1772, !dbg !11
  %bnk.n1781 = bitcast float %bnk.n1780 to i32, !dbg !11
  %bnk.n1782 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1781), !dbg !11
  %bnk.n1783 = bitcast i32 %bnk.n1782 to float, !dbg !11
  %bnk.n1786 = select i1 %.not36, float %bnk.n1779, float %bnk.n1773, !dbg !11
  %bnk.n1787 = bitcast float %bnk.n1786 to i32, !dbg !11
  %bnk.n1788 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1787), !dbg !11
  %bnk.n1789 = bitcast i32 %bnk.n1788 to float, !dbg !11
  %230 = insertelement <4 x float> poison, float %bnk.n1772, i64 0, !dbg !11
  %231 = insertelement <4 x float> %230, float %bnk.n1773, i64 1, !dbg !11
  %232 = insertelement <4 x float> %231, float %bnk.n1783, i64 2, !dbg !11
  %233 = insertelement <4 x float> %232, float %bnk.n1789, i64 3, !dbg !11
  %234 = insertelement <4 x float> poison, float %bnk.n1783, i64 0, !dbg !11
  %235 = insertelement <4 x float> %234, float %bnk.n1789, i64 1, !dbg !11
  %236 = insertelement <4 x float> %235, float %bnk.n1778, i64 2, !dbg !11
  %237 = insertelement <4 x float> %236, float %bnk.n1779, i64 3, !dbg !11
  %238 = select <4 x i1> %184, <4 x float> %233, <4 x float> %237, !dbg !11
  %bnk.n1792 = extractelement <8 x float> %bnk.n1683, i64 4, !dbg !11
  %bnk.n1793 = extractelement <8 x float> %bnk.n1683, i64 5, !dbg !11
  %bnk.n1794 = extractelement <8 x float> %bnk.n1683, i64 6, !dbg !11
  %bnk.n1795 = extractelement <8 x float> %bnk.n1683, i64 7, !dbg !11
  %bnk.n1796 = select i1 %.not, float %bnk.n1793, float %bnk.n1792, !dbg !11
  %bnk.n1797 = bitcast float %bnk.n1796 to i32, !dbg !11
  %bnk.n1798 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1797), !dbg !11
  %bnk.n1799 = bitcast i32 %bnk.n1798 to float, !dbg !11
  %bnk.n1800 = select i1 %.not, float %bnk.n1792, float %bnk.n1799, !dbg !11
  %bnk.n1801 = select i1 %.not, float %bnk.n1799, float %bnk.n1793, !dbg !11
  %bnk.n1802 = select i1 %.not, float %bnk.n1795, float %bnk.n1794, !dbg !11
  %bnk.n1803 = bitcast float %bnk.n1802 to i32, !dbg !11
  %bnk.n1804 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1803), !dbg !11
  %bnk.n1805 = bitcast i32 %bnk.n1804 to float, !dbg !11
  %bnk.n1806 = select i1 %.not, float %bnk.n1794, float %bnk.n1805, !dbg !11
  %bnk.n1807 = select i1 %.not, float %bnk.n1805, float %bnk.n1795, !dbg !11
  %bnk.n1808 = select i1 %.not36, float %bnk.n1806, float %bnk.n1800, !dbg !11
  %bnk.n1809 = bitcast float %bnk.n1808 to i32, !dbg !11
  %bnk.n1810 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1809), !dbg !11
  %bnk.n1811 = select i1 %.not36, float %bnk.n1807, float %bnk.n1801, !dbg !11
  %bnk.n1812 = bitcast float %bnk.n1811 to i32, !dbg !11
  %bnk.n1813 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1812), !dbg !11
  %bnk.n1814 = extractvalue [2 x <8 x float>] %bnk.n210.lcssa, 1, !dbg !11
  %bnk.n1815 = extractelement <8 x float> %bnk.n1814, i64 0, !dbg !11
  %bnk.n1816 = extractelement <8 x float> %bnk.n1814, i64 1, !dbg !11
  %bnk.n1817 = extractelement <8 x float> %bnk.n1814, i64 2, !dbg !11
  %bnk.n1818 = extractelement <8 x float> %bnk.n1814, i64 3, !dbg !11
  %bnk.n1819 = select i1 %.not, float %bnk.n1816, float %bnk.n1815, !dbg !11
  %bnk.n1820 = bitcast float %bnk.n1819 to i32, !dbg !11
  %bnk.n1821 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1820), !dbg !11
  %bnk.n1822 = bitcast i32 %bnk.n1821 to float, !dbg !11
  %bnk.n1823 = select i1 %.not, float %bnk.n1815, float %bnk.n1822, !dbg !11
  %bnk.n1824 = select i1 %.not, float %bnk.n1822, float %bnk.n1816, !dbg !11
  %bnk.n1825 = select i1 %.not, float %bnk.n1818, float %bnk.n1817, !dbg !11
  %bnk.n1826 = bitcast float %bnk.n1825 to i32, !dbg !11
  %bnk.n1827 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1826), !dbg !11
  %bnk.n1828 = bitcast i32 %bnk.n1827 to float, !dbg !11
  %bnk.n1829 = select i1 %.not, float %bnk.n1817, float %bnk.n1828, !dbg !11
  %bnk.n1830 = select i1 %.not, float %bnk.n1828, float %bnk.n1818, !dbg !11
  %bnk.n1831 = select i1 %.not36, float %bnk.n1829, float %bnk.n1823, !dbg !11
  %bnk.n1832 = bitcast float %bnk.n1831 to i32, !dbg !11
  %bnk.n1833 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1832), !dbg !11
  %bnk.n1834 = bitcast i32 %bnk.n1833 to float, !dbg !11
  %bnk.n1837 = select i1 %.not36, float %bnk.n1830, float %bnk.n1824, !dbg !11
  %bnk.n1838 = bitcast float %bnk.n1837 to i32, !dbg !11
  %bnk.n1839 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1838), !dbg !11
  %bnk.n1840 = bitcast i32 %bnk.n1839 to float, !dbg !11
  %239 = insertelement <4 x float> poison, float %bnk.n1823, i64 0, !dbg !11
  %240 = insertelement <4 x float> %239, float %bnk.n1824, i64 1, !dbg !11
  %241 = insertelement <4 x float> %240, float %bnk.n1834, i64 2, !dbg !11
  %242 = insertelement <4 x float> %241, float %bnk.n1840, i64 3, !dbg !11
  %243 = insertelement <4 x float> poison, float %bnk.n1834, i64 0, !dbg !11
  %244 = insertelement <4 x float> %243, float %bnk.n1840, i64 1, !dbg !11
  %245 = insertelement <4 x float> %244, float %bnk.n1829, i64 2, !dbg !11
  %246 = insertelement <4 x float> %245, float %bnk.n1830, i64 3, !dbg !11
  %247 = select <4 x i1> %184, <4 x float> %242, <4 x float> %246, !dbg !11
  %bnk.n1843 = extractvalue [2 x <8 x float>] %bnk.n209.lcssa, 1, !dbg !11
  %bnk.n1844 = extractelement <8 x float> %bnk.n1843, i64 0, !dbg !11
  %bnk.n1845 = extractelement <8 x float> %bnk.n1843, i64 1, !dbg !11
  %bnk.n1846 = extractelement <8 x float> %bnk.n1843, i64 2, !dbg !11
  %bnk.n1847 = extractelement <8 x float> %bnk.n1843, i64 3, !dbg !11
  %bnk.n1848 = select i1 %.not, float %bnk.n1845, float %bnk.n1844, !dbg !11
  %bnk.n1849 = bitcast float %bnk.n1848 to i32, !dbg !11
  %bnk.n1850 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1849), !dbg !11
  %bnk.n1851 = bitcast i32 %bnk.n1850 to float, !dbg !11
  %bnk.n1852 = select i1 %.not, float %bnk.n1844, float %bnk.n1851, !dbg !11
  %bnk.n1853 = select i1 %.not, float %bnk.n1851, float %bnk.n1845, !dbg !11
  %bnk.n1854 = select i1 %.not, float %bnk.n1847, float %bnk.n1846, !dbg !11
  %bnk.n1855 = bitcast float %bnk.n1854 to i32, !dbg !11
  %bnk.n1856 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1855), !dbg !11
  %bnk.n1857 = bitcast i32 %bnk.n1856 to float, !dbg !11
  %bnk.n1858 = select i1 %.not, float %bnk.n1846, float %bnk.n1857, !dbg !11
  %bnk.n1859 = select i1 %.not, float %bnk.n1857, float %bnk.n1847, !dbg !11
  %bnk.n1860 = select i1 %.not36, float %bnk.n1858, float %bnk.n1852, !dbg !11
  %bnk.n1861 = bitcast float %bnk.n1860 to i32, !dbg !11
  %bnk.n1862 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1861), !dbg !11
  %bnk.n1863 = bitcast i32 %bnk.n1862 to float, !dbg !11
  %bnk.n1866 = select i1 %.not36, float %bnk.n1859, float %bnk.n1853, !dbg !11
  %bnk.n1867 = bitcast float %bnk.n1866 to i32, !dbg !11
  %bnk.n1868 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1867), !dbg !11
  %bnk.n1869 = bitcast i32 %bnk.n1868 to float, !dbg !11
  %248 = insertelement <4 x float> poison, float %bnk.n1852, i64 0, !dbg !11
  %249 = insertelement <4 x float> %248, float %bnk.n1853, i64 1, !dbg !11
  %250 = insertelement <4 x float> %249, float %bnk.n1863, i64 2, !dbg !11
  %251 = insertelement <4 x float> %250, float %bnk.n1869, i64 3, !dbg !11
  %252 = insertelement <4 x float> poison, float %bnk.n1863, i64 0, !dbg !11
  %253 = insertelement <4 x float> %252, float %bnk.n1869, i64 1, !dbg !11
  %254 = insertelement <4 x float> %253, float %bnk.n1858, i64 2, !dbg !11
  %255 = insertelement <4 x float> %254, float %bnk.n1859, i64 3, !dbg !11
  %256 = select <4 x i1> %184, <4 x float> %251, <4 x float> %255, !dbg !11
  %bnk.n1872 = extractvalue [2 x <8 x float>] %bnk.n208.lcssa, 1, !dbg !11
  %bnk.n1873 = extractelement <8 x float> %bnk.n1872, i64 0, !dbg !11
  %bnk.n1874 = extractelement <8 x float> %bnk.n1872, i64 1, !dbg !11
  %bnk.n1875 = extractelement <8 x float> %bnk.n1872, i64 2, !dbg !11
  %bnk.n1876 = extractelement <8 x float> %bnk.n1872, i64 3, !dbg !11
  %bnk.n1877 = select i1 %.not, float %bnk.n1874, float %bnk.n1873, !dbg !11
  %bnk.n1878 = bitcast float %bnk.n1877 to i32, !dbg !11
  %bnk.n1879 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1878), !dbg !11
  %bnk.n1880 = bitcast i32 %bnk.n1879 to float, !dbg !11
  %bnk.n1881 = select i1 %.not, float %bnk.n1873, float %bnk.n1880, !dbg !11
  %bnk.n1882 = select i1 %.not, float %bnk.n1880, float %bnk.n1874, !dbg !11
  %bnk.n1883 = select i1 %.not, float %bnk.n1876, float %bnk.n1875, !dbg !11
  %bnk.n1884 = bitcast float %bnk.n1883 to i32, !dbg !11
  %bnk.n1885 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1884), !dbg !11
  %bnk.n1886 = bitcast i32 %bnk.n1885 to float, !dbg !11
  %bnk.n1887 = select i1 %.not, float %bnk.n1875, float %bnk.n1886, !dbg !11
  %bnk.n1888 = select i1 %.not, float %bnk.n1886, float %bnk.n1876, !dbg !11
  %bnk.n1889 = select i1 %.not36, float %bnk.n1887, float %bnk.n1881, !dbg !11
  %bnk.n1890 = bitcast float %bnk.n1889 to i32, !dbg !11
  %bnk.n1891 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1890), !dbg !11
  %bnk.n1892 = bitcast i32 %bnk.n1891 to float, !dbg !11
  %bnk.n1895 = select i1 %.not36, float %bnk.n1888, float %bnk.n1882, !dbg !11
  %bnk.n1896 = bitcast float %bnk.n1895 to i32, !dbg !11
  %bnk.n1897 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1896), !dbg !11
  %bnk.n1898 = bitcast i32 %bnk.n1897 to float, !dbg !11
  %257 = insertelement <4 x float> poison, float %bnk.n1881, i64 0, !dbg !11
  %258 = insertelement <4 x float> %257, float %bnk.n1882, i64 1, !dbg !11
  %259 = insertelement <4 x float> %258, float %bnk.n1892, i64 2, !dbg !11
  %260 = insertelement <4 x float> %259, float %bnk.n1898, i64 3, !dbg !11
  %261 = insertelement <4 x float> poison, float %bnk.n1892, i64 0, !dbg !11
  %262 = insertelement <4 x float> %261, float %bnk.n1898, i64 1, !dbg !11
  %263 = insertelement <4 x float> %262, float %bnk.n1887, i64 2, !dbg !11
  %264 = insertelement <4 x float> %263, float %bnk.n1888, i64 3, !dbg !11
  %265 = select <4 x i1> %184, <4 x float> %260, <4 x float> %264, !dbg !11
  %bnk.n1901 = extractvalue [2 x <8 x float>] %bnk.n206.lcssa, 1, !dbg !11
  %bnk.n1902 = extractelement <8 x float> %bnk.n1901, i64 0, !dbg !11
  %bnk.n1903 = extractelement <8 x float> %bnk.n1901, i64 1, !dbg !11
  %bnk.n1904 = extractelement <8 x float> %bnk.n1901, i64 2, !dbg !11
  %bnk.n1905 = extractelement <8 x float> %bnk.n1901, i64 3, !dbg !11
  %bnk.n1906 = select i1 %.not, float %bnk.n1903, float %bnk.n1902, !dbg !11
  %bnk.n1907 = bitcast float %bnk.n1906 to i32, !dbg !11
  %bnk.n1908 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1907), !dbg !11
  %bnk.n1909 = bitcast i32 %bnk.n1908 to float, !dbg !11
  %bnk.n1910 = select i1 %.not, float %bnk.n1902, float %bnk.n1909, !dbg !11
  %bnk.n1911 = select i1 %.not, float %bnk.n1909, float %bnk.n1903, !dbg !11
  %bnk.n1912 = select i1 %.not, float %bnk.n1905, float %bnk.n1904, !dbg !11
  %bnk.n1913 = bitcast float %bnk.n1912 to i32, !dbg !11
  %bnk.n1914 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1913), !dbg !11
  %bnk.n1915 = bitcast i32 %bnk.n1914 to float, !dbg !11
  %bnk.n1916 = select i1 %.not, float %bnk.n1904, float %bnk.n1915, !dbg !11
  %bnk.n1917 = select i1 %.not, float %bnk.n1915, float %bnk.n1905, !dbg !11
  %bnk.n1918 = select i1 %.not36, float %bnk.n1916, float %bnk.n1910, !dbg !11
  %bnk.n1919 = bitcast float %bnk.n1918 to i32, !dbg !11
  %bnk.n1920 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1919), !dbg !11
  %bnk.n1921 = select i1 %.not36, float %bnk.n1917, float %bnk.n1911, !dbg !11
  %bnk.n1922 = bitcast float %bnk.n1921 to i32, !dbg !11
  %bnk.n1923 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1922), !dbg !11
  %bnk.n1924 = extractvalue [2 x <1 x float>] %bnk.n207.lcssa, 1, !dbg !25
  %bnk.n1925 = shufflevector <1 x float> %bnk.n1924, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %bnk.n1926 = extractelement <8 x float> %bnk.n1814, i64 4, !dbg !11
  %bnk.n1927 = extractelement <8 x float> %bnk.n1814, i64 5, !dbg !11
  %bnk.n1928 = extractelement <8 x float> %bnk.n1814, i64 6, !dbg !11
  %bnk.n1929 = extractelement <8 x float> %bnk.n1814, i64 7, !dbg !11
  %bnk.n1930 = select i1 %.not, float %bnk.n1927, float %bnk.n1926, !dbg !11
  %bnk.n1931 = bitcast float %bnk.n1930 to i32, !dbg !11
  %bnk.n1932 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1931), !dbg !11
  %bnk.n1933 = bitcast i32 %bnk.n1932 to float, !dbg !11
  %bnk.n1934 = select i1 %.not, float %bnk.n1926, float %bnk.n1933, !dbg !11
  %bnk.n1935 = select i1 %.not, float %bnk.n1933, float %bnk.n1927, !dbg !11
  %bnk.n1936 = select i1 %.not, float %bnk.n1929, float %bnk.n1928, !dbg !11
  %bnk.n1937 = bitcast float %bnk.n1936 to i32, !dbg !11
  %bnk.n1938 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1937), !dbg !11
  %bnk.n1939 = bitcast i32 %bnk.n1938 to float, !dbg !11
  %bnk.n1940 = select i1 %.not, float %bnk.n1928, float %bnk.n1939, !dbg !11
  %bnk.n1941 = select i1 %.not, float %bnk.n1939, float %bnk.n1929, !dbg !11
  %bnk.n1942 = select i1 %.not36, float %bnk.n1940, float %bnk.n1934, !dbg !11
  %bnk.n1943 = bitcast float %bnk.n1942 to i32, !dbg !11
  %bnk.n1944 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1943), !dbg !11
  %bnk.n1945 = bitcast i32 %bnk.n1944 to float, !dbg !11
  %bnk.n1948 = select i1 %.not36, float %bnk.n1941, float %bnk.n1935, !dbg !11
  %bnk.n1949 = bitcast float %bnk.n1948 to i32, !dbg !11
  %bnk.n1950 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1949), !dbg !11
  %bnk.n1951 = bitcast i32 %bnk.n1950 to float, !dbg !11
  %266 = insertelement <4 x float> poison, float %bnk.n1934, i64 0, !dbg !11
  %267 = insertelement <4 x float> %266, float %bnk.n1935, i64 1, !dbg !11
  %268 = insertelement <4 x float> %267, float %bnk.n1945, i64 2, !dbg !11
  %269 = insertelement <4 x float> %268, float %bnk.n1951, i64 3, !dbg !11
  %270 = insertelement <4 x float> poison, float %bnk.n1945, i64 0, !dbg !11
  %271 = insertelement <4 x float> %270, float %bnk.n1951, i64 1, !dbg !11
  %272 = insertelement <4 x float> %271, float %bnk.n1940, i64 2, !dbg !11
  %273 = insertelement <4 x float> %272, float %bnk.n1941, i64 3, !dbg !11
  %274 = select <4 x i1> %184, <4 x float> %269, <4 x float> %273, !dbg !11
  %bnk.n1954 = extractelement <8 x float> %bnk.n1843, i64 4, !dbg !11
  %bnk.n1955 = extractelement <8 x float> %bnk.n1843, i64 5, !dbg !11
  %bnk.n1956 = extractelement <8 x float> %bnk.n1843, i64 6, !dbg !11
  %bnk.n1957 = extractelement <8 x float> %bnk.n1843, i64 7, !dbg !11
  %bnk.n1958 = select i1 %.not, float %bnk.n1955, float %bnk.n1954, !dbg !11
  %bnk.n1959 = bitcast float %bnk.n1958 to i32, !dbg !11
  %bnk.n1960 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1959), !dbg !11
  %bnk.n1961 = bitcast i32 %bnk.n1960 to float, !dbg !11
  %bnk.n1962 = select i1 %.not, float %bnk.n1954, float %bnk.n1961, !dbg !11
  %bnk.n1963 = select i1 %.not, float %bnk.n1961, float %bnk.n1955, !dbg !11
  %bnk.n1964 = select i1 %.not, float %bnk.n1957, float %bnk.n1956, !dbg !11
  %bnk.n1965 = bitcast float %bnk.n1964 to i32, !dbg !11
  %bnk.n1966 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1965), !dbg !11
  %bnk.n1967 = bitcast i32 %bnk.n1966 to float, !dbg !11
  %bnk.n1968 = select i1 %.not, float %bnk.n1956, float %bnk.n1967, !dbg !11
  %bnk.n1969 = select i1 %.not, float %bnk.n1967, float %bnk.n1957, !dbg !11
  %bnk.n1970 = select i1 %.not36, float %bnk.n1968, float %bnk.n1962, !dbg !11
  %bnk.n1971 = bitcast float %bnk.n1970 to i32, !dbg !11
  %bnk.n1972 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1971), !dbg !11
  %bnk.n1973 = bitcast i32 %bnk.n1972 to float, !dbg !11
  %bnk.n1976 = select i1 %.not36, float %bnk.n1969, float %bnk.n1963, !dbg !11
  %bnk.n1977 = bitcast float %bnk.n1976 to i32, !dbg !11
  %bnk.n1978 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1977), !dbg !11
  %bnk.n1979 = bitcast i32 %bnk.n1978 to float, !dbg !11
  %275 = insertelement <4 x float> poison, float %bnk.n1962, i64 0, !dbg !11
  %276 = insertelement <4 x float> %275, float %bnk.n1963, i64 1, !dbg !11
  %277 = insertelement <4 x float> %276, float %bnk.n1973, i64 2, !dbg !11
  %278 = insertelement <4 x float> %277, float %bnk.n1979, i64 3, !dbg !11
  %279 = insertelement <4 x float> poison, float %bnk.n1973, i64 0, !dbg !11
  %280 = insertelement <4 x float> %279, float %bnk.n1979, i64 1, !dbg !11
  %281 = insertelement <4 x float> %280, float %bnk.n1968, i64 2, !dbg !11
  %282 = insertelement <4 x float> %281, float %bnk.n1969, i64 3, !dbg !11
  %283 = select <4 x i1> %184, <4 x float> %278, <4 x float> %282, !dbg !11
  %bnk.n1982 = extractelement <8 x float> %bnk.n1872, i64 4, !dbg !11
  %bnk.n1983 = extractelement <8 x float> %bnk.n1872, i64 5, !dbg !11
  %bnk.n1984 = extractelement <8 x float> %bnk.n1872, i64 6, !dbg !11
  %bnk.n1985 = extractelement <8 x float> %bnk.n1872, i64 7, !dbg !11
  %bnk.n1986 = select i1 %.not, float %bnk.n1983, float %bnk.n1982, !dbg !11
  %bnk.n1987 = bitcast float %bnk.n1986 to i32, !dbg !11
  %bnk.n1988 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1987), !dbg !11
  %bnk.n1989 = bitcast i32 %bnk.n1988 to float, !dbg !11
  %bnk.n1990 = select i1 %.not, float %bnk.n1982, float %bnk.n1989, !dbg !11
  %bnk.n1991 = select i1 %.not, float %bnk.n1989, float %bnk.n1983, !dbg !11
  %bnk.n1992 = select i1 %.not, float %bnk.n1985, float %bnk.n1984, !dbg !11
  %bnk.n1993 = bitcast float %bnk.n1992 to i32, !dbg !11
  %bnk.n1994 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n1993), !dbg !11
  %bnk.n1995 = bitcast i32 %bnk.n1994 to float, !dbg !11
  %bnk.n1996 = select i1 %.not, float %bnk.n1984, float %bnk.n1995, !dbg !11
  %bnk.n1997 = select i1 %.not, float %bnk.n1995, float %bnk.n1985, !dbg !11
  %bnk.n1998 = select i1 %.not36, float %bnk.n1996, float %bnk.n1990, !dbg !11
  %bnk.n1999 = bitcast float %bnk.n1998 to i32, !dbg !11
  %bnk.n2000 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n1999), !dbg !11
  %bnk.n2001 = bitcast i32 %bnk.n2000 to float, !dbg !11
  %bnk.n2004 = select i1 %.not36, float %bnk.n1997, float %bnk.n1991, !dbg !11
  %bnk.n2005 = bitcast float %bnk.n2004 to i32, !dbg !11
  %bnk.n2006 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n2005), !dbg !11
  %bnk.n2007 = bitcast i32 %bnk.n2006 to float, !dbg !11
  %284 = insertelement <4 x float> poison, float %bnk.n1990, i64 0, !dbg !11
  %285 = insertelement <4 x float> %284, float %bnk.n1991, i64 1, !dbg !11
  %286 = insertelement <4 x float> %285, float %bnk.n2001, i64 2, !dbg !11
  %287 = insertelement <4 x float> %286, float %bnk.n2007, i64 3, !dbg !11
  %288 = insertelement <4 x float> poison, float %bnk.n2001, i64 0, !dbg !11
  %289 = insertelement <4 x float> %288, float %bnk.n2007, i64 1, !dbg !11
  %290 = insertelement <4 x float> %289, float %bnk.n1996, i64 2, !dbg !11
  %291 = insertelement <4 x float> %290, float %bnk.n1997, i64 3, !dbg !11
  %292 = select <4 x i1> %184, <4 x float> %287, <4 x float> %291, !dbg !11
  %bnk.n2010 = extractelement <8 x float> %bnk.n1901, i64 4, !dbg !11
  %bnk.n2011 = extractelement <8 x float> %bnk.n1901, i64 5, !dbg !11
  %bnk.n2012 = extractelement <8 x float> %bnk.n1901, i64 6, !dbg !11
  %bnk.n2013 = extractelement <8 x float> %bnk.n1901, i64 7, !dbg !11
  %bnk.n2014 = select i1 %.not, float %bnk.n2011, float %bnk.n2010, !dbg !11
  %bnk.n2015 = bitcast float %bnk.n2014 to i32, !dbg !11
  %bnk.n2016 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n2015), !dbg !11
  %bnk.n2017 = bitcast i32 %bnk.n2016 to float, !dbg !11
  %bnk.n2018 = select i1 %.not, float %bnk.n2010, float %bnk.n2017, !dbg !11
  %bnk.n2019 = select i1 %.not, float %bnk.n2017, float %bnk.n2011, !dbg !11
  %bnk.n2020 = select i1 %.not, float %bnk.n2013, float %bnk.n2012, !dbg !11
  %bnk.n2021 = bitcast float %bnk.n2020 to i32, !dbg !11
  %bnk.n2022 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1597, i32 %bnk.n2021), !dbg !11
  %bnk.n2023 = bitcast i32 %bnk.n2022 to float, !dbg !11
  %bnk.n2024 = select i1 %.not, float %bnk.n2012, float %bnk.n2023, !dbg !11
  %bnk.n2025 = select i1 %.not, float %bnk.n2023, float %bnk.n2013, !dbg !11
  %bnk.n2026 = select i1 %.not36, float %bnk.n2024, float %bnk.n2018, !dbg !11
  %bnk.n2027 = bitcast float %bnk.n2026 to i32, !dbg !11
  %bnk.n2028 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n2027), !dbg !11
  %bnk.n2029 = select i1 %.not36, float %bnk.n2025, float %bnk.n2019, !dbg !11
  %bnk.n2030 = bitcast float %bnk.n2029 to i32, !dbg !11
  %bnk.n2031 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %bnk.n1613, i32 %bnk.n2030), !dbg !11
  %bnk.n2033 = bitcast i32 %bnk.n2028 to float, !dbg !11
  %bnk.n2036 = bitcast i32 %bnk.n2031 to float, !dbg !11
  %293 = insertelement <4 x float> poison, float %bnk.n2018, i64 0, !dbg !11
  %294 = insertelement <4 x float> %293, float %bnk.n2019, i64 1, !dbg !11
  %295 = insertelement <4 x float> %294, float %bnk.n2033, i64 2, !dbg !11
  %296 = insertelement <4 x float> %295, float %bnk.n2036, i64 3, !dbg !11
  %297 = insertelement <4 x float> poison, float %bnk.n2033, i64 0, !dbg !11
  %298 = insertelement <4 x float> %297, float %bnk.n2036, i64 1, !dbg !11
  %299 = insertelement <4 x float> %298, float %bnk.n2024, i64 2, !dbg !11
  %300 = insertelement <4 x float> %299, float %bnk.n2025, i64 3, !dbg !11
  %301 = select <4 x i1> %184, <4 x float> %296, <4 x float> %300, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %bnk.n1925, !dbg !25
  %normalized = fmul <4 x float> %row.reciprocal, %301, !dbg !25
  %bnk.n2043 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !26
  %bnk.n2044 = extractelement <4 x half> %bnk.n2043, i64 3, !dbg !26
  %bnk.n2045 = extractelement <4 x half> %bnk.n2043, i64 2, !dbg !26
  %bnk.n2046 = extractelement <4 x half> %bnk.n2043, i64 1, !dbg !26
  %bnk.n2047 = extractelement <4 x half> %bnk.n2043, i64 0, !dbg !26
  %bnk.n2048 = bitcast i32 %bnk.n1920 to float, !dbg !11
  %bnk.n2051 = bitcast i32 %bnk.n1923 to float, !dbg !11
  %302 = insertelement <4 x float> poison, float %bnk.n1910, i64 0, !dbg !11
  %303 = insertelement <4 x float> %302, float %bnk.n1911, i64 1, !dbg !11
  %304 = insertelement <4 x float> %303, float %bnk.n2048, i64 2, !dbg !11
  %305 = insertelement <4 x float> %304, float %bnk.n2051, i64 3, !dbg !11
  %306 = insertelement <4 x float> poison, float %bnk.n2048, i64 0, !dbg !11
  %307 = insertelement <4 x float> %306, float %bnk.n2051, i64 1, !dbg !11
  %308 = insertelement <4 x float> %307, float %bnk.n1916, i64 2, !dbg !11
  %309 = insertelement <4 x float> %308, float %bnk.n1917, i64 3, !dbg !11
  %310 = select <4 x i1> %184, <4 x float> %305, <4 x float> %309, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %310, !dbg !25
  %bnk.n2058 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !26
  %bnk.n2059 = extractelement <4 x half> %bnk.n2058, i64 3, !dbg !26
  %bnk.n2060 = extractelement <4 x half> %bnk.n2058, i64 2, !dbg !26
  %bnk.n2061 = extractelement <4 x half> %bnk.n2058, i64 1, !dbg !26
  %bnk.n2062 = extractelement <4 x half> %bnk.n2058, i64 0, !dbg !26
  %bnk.n2063 = bitcast i32 %bnk.n1810 to float, !dbg !11
  %bnk.n2066 = bitcast i32 %bnk.n1813 to float, !dbg !11
  %311 = insertelement <4 x float> poison, float %bnk.n1800, i64 0, !dbg !11
  %312 = insertelement <4 x float> %311, float %bnk.n1801, i64 1, !dbg !11
  %313 = insertelement <4 x float> %312, float %bnk.n2063, i64 2, !dbg !11
  %314 = insertelement <4 x float> %313, float %bnk.n2066, i64 3, !dbg !11
  %315 = insertelement <4 x float> poison, float %bnk.n2063, i64 0, !dbg !11
  %316 = insertelement <4 x float> %315, float %bnk.n2066, i64 1, !dbg !11
  %317 = insertelement <4 x float> %316, float %bnk.n1806, i64 2, !dbg !11
  %318 = insertelement <4 x float> %317, float %bnk.n1807, i64 3, !dbg !11
  %319 = select <4 x i1> %184, <4 x float> %314, <4 x float> %318, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %bnk.n1707, !dbg !25
  %normalized106 = fmul <4 x float> %row.reciprocal105, %319, !dbg !25
  %bnk.n2073 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !26
  %bnk.n2074 = extractelement <4 x half> %bnk.n2073, i64 3, !dbg !26
  %bnk.n2075 = extractelement <4 x half> %bnk.n2073, i64 2, !dbg !26
  %bnk.n2076 = extractelement <4 x half> %bnk.n2073, i64 1, !dbg !26
  %bnk.n2077 = extractelement <4 x half> %bnk.n2073, i64 0, !dbg !26
  %bnk.n2078 = bitcast i32 %bnk.n1702 to float, !dbg !11
  %bnk.n2081 = bitcast i32 %bnk.n1705 to float, !dbg !11
  %320 = insertelement <4 x float> poison, float %bnk.n1692, i64 0, !dbg !11
  %321 = insertelement <4 x float> %320, float %bnk.n1693, i64 1, !dbg !11
  %322 = insertelement <4 x float> %321, float %bnk.n2078, i64 2, !dbg !11
  %323 = insertelement <4 x float> %322, float %bnk.n2081, i64 3, !dbg !11
  %324 = insertelement <4 x float> poison, float %bnk.n2078, i64 0, !dbg !11
  %325 = insertelement <4 x float> %324, float %bnk.n2081, i64 1, !dbg !11
  %326 = insertelement <4 x float> %325, float %bnk.n1698, i64 2, !dbg !11
  %327 = insertelement <4 x float> %326, float %bnk.n1699, i64 3, !dbg !11
  %328 = select <4 x i1> %184, <4 x float> %323, <4 x float> %327, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %328, !dbg !25
  %bnk.n2088 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !26
  %bnk.n2089 = extractelement <4 x half> %bnk.n2088, i64 3, !dbg !26
  %bnk.n2090 = extractelement <4 x half> %bnk.n2088, i64 2, !dbg !26
  %bnk.n2091 = extractelement <4 x half> %bnk.n2088, i64 1, !dbg !26
  %bnk.n2092 = extractelement <4 x half> %bnk.n2088, i64 0, !dbg !26
  %address.thread934 = tail call i32 asm sideeffect "", "=v,0"(i32 %bnk.n31) #8, !dbg !27, !range !9, !frisk.register_lifetimes !5
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !8
  %329 = lshr i64 %address.local935, 2, !dbg !27
  %address.local938 = and i64 %329, 48, !dbg !27
  %address.local940 = and i64 %address.local935, 240, !dbg !27
  %address.local939 = add i64 %bnk.n27, %address.local935, !dbg !27
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !27
  %address.local942 = add i64 %address.local941, %address.local938, !dbg !27
  %330 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %bnk.n28, !dbg !27
  %address.local945.idx = shl nuw nsw i64 %address.local942, 8, !dbg !27
  %address.local945 = getelementptr inbounds nuw i8, ptr addrspace(1) %330, i64 %address.local945.idx, !dbg !27
  %address.local947 = and i64 %329, 12, !dbg !10
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !27
  store half %bnk.n2092, ptr addrspace(1) %address.local948, align 2, !dbg !27
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !27
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local950, !dbg !27
  store half %bnk.n2091, ptr addrspace(1) %address.local951, align 2, !dbg !27
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !27
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local953, !dbg !27
  store half %bnk.n2090, ptr addrspace(1) %address.local954, align 2, !dbg !27
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !27
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local956, !dbg !27
  store half %bnk.n2089, ptr addrspace(1) %address.local957, align 2, !dbg !27
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !27
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local959, !dbg !27
  store half %bnk.n2077, ptr addrspace(1) %address.local960, align 2, !dbg !27
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !27
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local962, !dbg !27
  store half %bnk.n2076, ptr addrspace(1) %address.local963, align 2, !dbg !27
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !27
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local965, !dbg !27
  store half %bnk.n2075, ptr addrspace(1) %address.local966, align 2, !dbg !27
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !27
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local968, !dbg !27
  store half %bnk.n2074, ptr addrspace(1) %address.local969, align 2, !dbg !27
  %address.local973 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local945, i64 16384, !dbg !27
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !27
  store half %bnk.n2062, ptr addrspace(1) %address.local974, align 2, !dbg !27
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local950, !dbg !27
  store half %bnk.n2061, ptr addrspace(1) %address.local976, align 2, !dbg !27
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local953, !dbg !27
  store half %bnk.n2060, ptr addrspace(1) %address.local978, align 2, !dbg !27
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local956, !dbg !27
  store half %bnk.n2059, ptr addrspace(1) %address.local980, align 2, !dbg !27
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local959, !dbg !27
  store half %bnk.n2047, ptr addrspace(1) %address.local982, align 2, !dbg !27
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local962, !dbg !27
  store half %bnk.n2046, ptr addrspace(1) %address.local984, align 2, !dbg !27
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local965, !dbg !27
  store half %bnk.n2045, ptr addrspace(1) %address.local986, align 2, !dbg !27
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local968, !dbg !27
  store half %bnk.n2044, ptr addrspace(1) %address.local988, align 2, !dbg !27
  %normalized109 = fmul <4 x float> %row.reciprocal, %292, !dbg !25
  %bnk.n2102 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !26
  %bnk.n2103 = extractelement <4 x half> %bnk.n2102, i64 3, !dbg !26
  %bnk.n2104 = extractelement <4 x half> %bnk.n2102, i64 2, !dbg !26
  %bnk.n2105 = extractelement <4 x half> %bnk.n2102, i64 1, !dbg !26
  %bnk.n2106 = extractelement <4 x half> %bnk.n2102, i64 0, !dbg !26
  %normalized110 = fmul <4 x float> %row.reciprocal, %265, !dbg !25
  %bnk.n2111 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !26
  %bnk.n2112 = extractelement <4 x half> %bnk.n2111, i64 3, !dbg !26
  %bnk.n2113 = extractelement <4 x half> %bnk.n2111, i64 2, !dbg !26
  %bnk.n2114 = extractelement <4 x half> %bnk.n2111, i64 1, !dbg !26
  %bnk.n2115 = extractelement <4 x half> %bnk.n2111, i64 0, !dbg !26
  %normalized112 = fmul <4 x float> %row.reciprocal105, %238, !dbg !25
  %bnk.n2120 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !26
  %bnk.n2121 = extractelement <4 x half> %bnk.n2120, i64 3, !dbg !26
  %bnk.n2122 = extractelement <4 x half> %bnk.n2120, i64 2, !dbg !26
  %bnk.n2123 = extractelement <4 x half> %bnk.n2120, i64 1, !dbg !26
  %bnk.n2124 = extractelement <4 x half> %bnk.n2120, i64 0, !dbg !26
  %normalized113 = fmul <4 x float> %row.reciprocal105, %211, !dbg !25
  %bnk.n2129 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !26
  %bnk.n2130 = extractelement <4 x half> %bnk.n2129, i64 3, !dbg !26
  %bnk.n2131 = extractelement <4 x half> %bnk.n2129, i64 2, !dbg !26
  %bnk.n2132 = extractelement <4 x half> %bnk.n2129, i64 1, !dbg !26
  %bnk.n2133 = extractelement <4 x half> %bnk.n2129, i64 0, !dbg !26
  %address.local1003 = or disjoint i64 %address.local947, 32, !dbg !27
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1003, !dbg !27
  store half %bnk.n2133, ptr addrspace(1) %address.local1004, align 2, !dbg !27
  %address.local1006 = or disjoint i64 %address.local947, 33, !dbg !27
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1006, !dbg !27
  store half %bnk.n2132, ptr addrspace(1) %address.local1007, align 2, !dbg !27
  %address.local1009 = or disjoint i64 %address.local947, 34, !dbg !27
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1009, !dbg !27
  store half %bnk.n2131, ptr addrspace(1) %address.local1010, align 2, !dbg !27
  %address.local1012 = or disjoint i64 %address.local947, 35, !dbg !27
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1012, !dbg !27
  store half %bnk.n2130, ptr addrspace(1) %address.local1013, align 2, !dbg !27
  %address.local1015 = or disjoint i64 %address.local947, 48, !dbg !27
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1015, !dbg !27
  store half %bnk.n2124, ptr addrspace(1) %address.local1016, align 2, !dbg !27
  %address.local1018 = or disjoint i64 %address.local947, 49, !dbg !27
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1018, !dbg !27
  store half %bnk.n2123, ptr addrspace(1) %address.local1019, align 2, !dbg !27
  %address.local1021 = or disjoint i64 %address.local947, 50, !dbg !27
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1021, !dbg !27
  store half %bnk.n2122, ptr addrspace(1) %address.local1022, align 2, !dbg !27
  %address.local1024 = or i64 %329, 51, !dbg !27
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1024, !dbg !27
  store half %bnk.n2121, ptr addrspace(1) %address.local1025, align 2, !dbg !27
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1003, !dbg !27
  store half %bnk.n2115, ptr addrspace(1) %address.local1030, align 2, !dbg !27
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1006, !dbg !27
  store half %bnk.n2114, ptr addrspace(1) %address.local1032, align 2, !dbg !27
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1009, !dbg !27
  store half %bnk.n2113, ptr addrspace(1) %address.local1034, align 2, !dbg !27
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1012, !dbg !27
  store half %bnk.n2112, ptr addrspace(1) %address.local1036, align 2, !dbg !27
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1015, !dbg !27
  store half %bnk.n2106, ptr addrspace(1) %address.local1038, align 2, !dbg !27
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1018, !dbg !27
  store half %bnk.n2105, ptr addrspace(1) %address.local1040, align 2, !dbg !27
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1021, !dbg !27
  store half %bnk.n2104, ptr addrspace(1) %address.local1042, align 2, !dbg !27
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1024, !dbg !27
  store half %bnk.n2103, ptr addrspace(1) %address.local1044, align 2, !dbg !27
  %normalized115 = fmul <4 x float> %row.reciprocal, %283, !dbg !25
  %bnk.n2141 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !26
  %bnk.n2142 = extractelement <4 x half> %bnk.n2141, i64 3, !dbg !26
  %bnk.n2143 = extractelement <4 x half> %bnk.n2141, i64 2, !dbg !26
  %bnk.n2144 = extractelement <4 x half> %bnk.n2141, i64 1, !dbg !26
  %bnk.n2145 = extractelement <4 x half> %bnk.n2141, i64 0, !dbg !26
  %normalized116 = fmul <4 x float> %row.reciprocal, %256, !dbg !25
  %bnk.n2150 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !26
  %bnk.n2151 = extractelement <4 x half> %bnk.n2150, i64 3, !dbg !26
  %bnk.n2152 = extractelement <4 x half> %bnk.n2150, i64 2, !dbg !26
  %bnk.n2153 = extractelement <4 x half> %bnk.n2150, i64 1, !dbg !26
  %bnk.n2154 = extractelement <4 x half> %bnk.n2150, i64 0, !dbg !26
  %normalized118 = fmul <4 x float> %row.reciprocal105, %229, !dbg !25
  %bnk.n2159 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !26
  %bnk.n2160 = extractelement <4 x half> %bnk.n2159, i64 3, !dbg !26
  %bnk.n2161 = extractelement <4 x half> %bnk.n2159, i64 2, !dbg !26
  %bnk.n2162 = extractelement <4 x half> %bnk.n2159, i64 1, !dbg !26
  %bnk.n2163 = extractelement <4 x half> %bnk.n2159, i64 0, !dbg !26
  %normalized119 = fmul <4 x float> %row.reciprocal105, %202, !dbg !25
  %bnk.n2168 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !26
  %bnk.n2169 = extractelement <4 x half> %bnk.n2168, i64 3, !dbg !26
  %bnk.n2170 = extractelement <4 x half> %bnk.n2168, i64 2, !dbg !26
  %bnk.n2171 = extractelement <4 x half> %bnk.n2168, i64 1, !dbg !26
  %bnk.n2172 = extractelement <4 x half> %bnk.n2168, i64 0, !dbg !26
  %address.local1059 = or disjoint i64 %address.local947, 64, !dbg !27
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1059, !dbg !27
  store half %bnk.n2172, ptr addrspace(1) %address.local1060, align 2, !dbg !27
  %address.local1062 = or disjoint i64 %address.local947, 65, !dbg !27
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1062, !dbg !27
  store half %bnk.n2171, ptr addrspace(1) %address.local1063, align 2, !dbg !27
  %address.local1065 = or disjoint i64 %address.local947, 66, !dbg !27
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1065, !dbg !27
  store half %bnk.n2170, ptr addrspace(1) %address.local1066, align 2, !dbg !27
  %address.local1068 = or disjoint i64 %address.local947, 67, !dbg !27
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1068, !dbg !27
  store half %bnk.n2169, ptr addrspace(1) %address.local1069, align 2, !dbg !27
  %address.local1071 = or disjoint i64 %address.local947, 80, !dbg !27
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1071, !dbg !27
  store half %bnk.n2163, ptr addrspace(1) %address.local1072, align 2, !dbg !27
  %address.local1074 = or disjoint i64 %address.local947, 81, !dbg !27
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1074, !dbg !27
  store half %bnk.n2162, ptr addrspace(1) %address.local1075, align 2, !dbg !27
  %address.local1077 = or disjoint i64 %address.local947, 82, !dbg !27
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1077, !dbg !27
  store half %bnk.n2161, ptr addrspace(1) %address.local1078, align 2, !dbg !27
  %address.local1080 = or disjoint i64 %address.local947, 83, !dbg !27
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1080, !dbg !27
  store half %bnk.n2160, ptr addrspace(1) %address.local1081, align 2, !dbg !27
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1059, !dbg !27
  store half %bnk.n2154, ptr addrspace(1) %address.local1086, align 2, !dbg !27
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1062, !dbg !27
  store half %bnk.n2153, ptr addrspace(1) %address.local1088, align 2, !dbg !27
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1065, !dbg !27
  store half %bnk.n2152, ptr addrspace(1) %address.local1090, align 2, !dbg !27
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1068, !dbg !27
  store half %bnk.n2151, ptr addrspace(1) %address.local1092, align 2, !dbg !27
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1071, !dbg !27
  store half %bnk.n2145, ptr addrspace(1) %address.local1094, align 2, !dbg !27
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1074, !dbg !27
  store half %bnk.n2144, ptr addrspace(1) %address.local1096, align 2, !dbg !27
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1077, !dbg !27
  store half %bnk.n2143, ptr addrspace(1) %address.local1098, align 2, !dbg !27
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1080, !dbg !27
  store half %bnk.n2142, ptr addrspace(1) %address.local1100, align 2, !dbg !27
  %normalized121 = fmul <4 x float> %row.reciprocal, %274, !dbg !25
  %bnk.n2180 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !26
  %bnk.n2181 = extractelement <4 x half> %bnk.n2180, i64 3, !dbg !26
  %bnk.n2182 = extractelement <4 x half> %bnk.n2180, i64 2, !dbg !26
  %bnk.n2183 = extractelement <4 x half> %bnk.n2180, i64 1, !dbg !26
  %bnk.n2184 = extractelement <4 x half> %bnk.n2180, i64 0, !dbg !26
  %normalized122 = fmul <4 x float> %row.reciprocal, %247, !dbg !25
  %bnk.n2189 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !26
  %bnk.n2190 = extractelement <4 x half> %bnk.n2189, i64 3, !dbg !26
  %bnk.n2191 = extractelement <4 x half> %bnk.n2189, i64 2, !dbg !26
  %bnk.n2192 = extractelement <4 x half> %bnk.n2189, i64 1, !dbg !26
  %bnk.n2193 = extractelement <4 x half> %bnk.n2189, i64 0, !dbg !26
  %normalized124 = fmul <4 x float> %row.reciprocal105, %220, !dbg !25
  %bnk.n2198 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !26
  %bnk.n2199 = extractelement <4 x half> %bnk.n2198, i64 3, !dbg !26
  %bnk.n2200 = extractelement <4 x half> %bnk.n2198, i64 2, !dbg !26
  %bnk.n2201 = extractelement <4 x half> %bnk.n2198, i64 1, !dbg !26
  %bnk.n2202 = extractelement <4 x half> %bnk.n2198, i64 0, !dbg !26
  %normalized125 = fmul <4 x float> %row.reciprocal105, %193, !dbg !25
  %bnk.n2207 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !26
  %bnk.n2208 = extractelement <4 x half> %bnk.n2207, i64 3, !dbg !26
  %bnk.n2209 = extractelement <4 x half> %bnk.n2207, i64 2, !dbg !26
  %bnk.n2210 = extractelement <4 x half> %bnk.n2207, i64 1, !dbg !26
  %bnk.n2211 = extractelement <4 x half> %bnk.n2207, i64 0, !dbg !26
  %address.local1115 = or disjoint i64 %address.local947, 96, !dbg !27
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1115, !dbg !27
  store half %bnk.n2211, ptr addrspace(1) %address.local1116, align 2, !dbg !27
  %address.local1118 = or disjoint i64 %address.local947, 97, !dbg !27
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1118, !dbg !27
  store half %bnk.n2210, ptr addrspace(1) %address.local1119, align 2, !dbg !27
  %address.local1121 = or disjoint i64 %address.local947, 98, !dbg !27
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1121, !dbg !27
  store half %bnk.n2209, ptr addrspace(1) %address.local1122, align 2, !dbg !27
  %address.local1124 = or disjoint i64 %address.local947, 99, !dbg !27
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1124, !dbg !27
  store half %bnk.n2208, ptr addrspace(1) %address.local1125, align 2, !dbg !27
  %address.local1127 = or disjoint i64 %address.local947, 112, !dbg !27
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1127, !dbg !27
  store half %bnk.n2202, ptr addrspace(1) %address.local1128, align 2, !dbg !27
  %address.local1130 = or disjoint i64 %address.local947, 113, !dbg !27
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1130, !dbg !27
  store half %bnk.n2201, ptr addrspace(1) %address.local1131, align 2, !dbg !27
  %address.local1133 = or disjoint i64 %address.local947, 114, !dbg !27
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1133, !dbg !27
  store half %bnk.n2200, ptr addrspace(1) %address.local1134, align 2, !dbg !27
  %address.local1136 = or i64 %329, 115, !dbg !27
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1136, !dbg !27
  store half %bnk.n2199, ptr addrspace(1) %address.local1137, align 2, !dbg !27
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1115, !dbg !27
  store half %bnk.n2193, ptr addrspace(1) %address.local1142, align 2, !dbg !27
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1118, !dbg !27
  store half %bnk.n2192, ptr addrspace(1) %address.local1144, align 2, !dbg !27
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1121, !dbg !27
  store half %bnk.n2191, ptr addrspace(1) %address.local1146, align 2, !dbg !27
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1124, !dbg !27
  store half %bnk.n2190, ptr addrspace(1) %address.local1148, align 2, !dbg !27
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1127, !dbg !27
  store half %bnk.n2184, ptr addrspace(1) %address.local1150, align 2, !dbg !27
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1130, !dbg !27
  store half %bnk.n2183, ptr addrspace(1) %address.local1152, align 2, !dbg !27
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1133, !dbg !27
  store half %bnk.n2182, ptr addrspace(1) %address.local1154, align 2, !dbg !27
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1136, !dbg !27
  store half %bnk.n2181, ptr addrspace(1) %address.local1156, align 2, !dbg !27
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
!14 = !DILocation(line: 67, column: 22, scope: !3)
!15 = !DILocation(line: 69, column: 26, scope: !3)
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
