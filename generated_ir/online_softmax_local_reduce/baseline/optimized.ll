; ModuleID = 'generated_ir/online_softmax_local_reduce/baseline/kernel.bc'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %v5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %v6 = sext i32 %v5 to i64, !dbg !7
  %v7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %v8 = sext i32 %v7 to i64, !dbg !7
  %v9 = shl nsw i64 %v8, 5, !dbg !7
  %v10 = add nsw i64 %v9, %v6, !dbg !7
  %v11 = lshr i64 %v10, 5, !dbg !7
  %v12 = and i64 %v11, 576460752303423484, !dbg !7
  %v13 = and i64 %v6, 3, !dbg !7
  %v14 = or disjoint i64 %v12, %v13, !dbg !7
  %v15 = sub nsw i64 31, %v14, !dbg !7
  %v16 = lshr i64 %v10, 2, !dbg !7
  %v17 = icmp ugt i64 %v10, 1023, !dbg !8
  %v18 = add nsw i64 %v14, -32, !dbg !8
  %v19 = select i1 %v17, i64 %v18, i64 %v15, !dbg !8
  %v20 = sdiv i64 %v19, 32, !dbg !8
  %v21 = sext i1 %v17 to i64, !dbg !8
  %v22 = xor i64 %v20, %v21, !dbg !8
  %v23 = add nsw i64 %v22, %v16, !dbg !8
  %v25 = shl i64 %v15, 7, !dbg !8
  %5 = shl i64 %v22, 12, !dbg !8
  %v27 = sub i64 %v25, %5, !dbg !8
  %v24 = shl i64 %v23, 19, !dbg !8
  %v28 = and i64 %v24, 16252928, !dbg !8
  %v31 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %v33.idx = shl i64 %v27, 8, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %v31) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local339 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local340 = and i64 %address.local339, 1536, !dbg !8
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !8
  %address.local344 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local345 = and i64 %address.local344, 16256, !dbg !8
  %address.local346 = lshr i64 %address.local, 4, !dbg !8
  %address.local347 = and i64 %address.local346, 8, !dbg !8
  %address.local348 = or disjoint i64 %address.local345, %address.local347, !dbg !8
  %v37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %6 = or disjoint i64 %address.local348, %v28, !dbg !8
  %7 = shl nuw nsw i64 %6, 1, !dbg !8
  %v39 = add i64 %7, %v33.idx, !dbg !8
  %dma.buffer.offset1612 = trunc i64 %v39 to i32, !dbg !8
  %8 = insertelement <2 x i64> poison, i64 %v37, i64 0, !dbg !8
  %v44 = bitcast <2 x i64> %8 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1613 = shufflevector <4 x i32> %v44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local350 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local351 = shl nuw nsw i64 %address.local350, 3, !dbg !8
  %address.local352 = and i64 %address.local351, 3584, !dbg !8
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !8
  %address.local358 = lshr i64 %address.local350, 4, !dbg !8
  %address.local359 = and i64 %address.local358, 24, !dbg !8
  %address.local360 = or disjoint i64 %address.local359, %address.local345, !dbg !8
  %9 = or disjoint i64 %address.local360, %v28, !dbg !8
  %10 = shl nuw nsw i64 %9, 1, !dbg !8
  %v53 = add i64 %10, %v33.idx, !dbg !8
  %dma.buffer.offset1609 = trunc i64 %v53 to i32, !dbg !8
  %v62 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %v63 = sext i32 %v62 to i64, !dbg !7
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !8
  %.idx = shl nuw nsw i64 %v28, 1, !dbg !8
  %11 = add i64 %.idx, %v33.idx, !dbg !8
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
  %v34 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %20), !dbg !8
  %dma.buffer.lds1614 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v34), !dbg !8
  %v47 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1612, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %21 = extractelement <4 x i32> %19, i64 1, !dbg !8
  %v48 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %21), !dbg !8
  %dma.buffer.lds1611 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v48), !dbg !8
  %v61 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1609, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %22 = extractelement <4 x i32> %19, i64 2, !dbg !8
  %v68 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %22), !dbg !8
  %dma.buffer.lds1608 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v68), !dbg !8
  %v81 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1606, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %23 = extractelement <4 x i32> %19, i64 3, !dbg !8
  %v82 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %23), !dbg !8
  %address.local369.idx = shl nuw nsw i64 %address.local360, 1, !dbg !8
  %24 = or disjoint i64 %address.local369.idx, %11, !dbg !8
  %25 = trunc i64 %24 to i32, !dbg !8
  %dma.buffer.offset1603 = or disjoint i32 %25, 64, !dbg !8
  %dma.buffer.lds1605 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v82), !dbg !8
  %v95 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1603, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
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
  %v96 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !10
  %v97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %v98 = fmul <4 x half> %v96, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !10
  %v99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !10
  %v100 = fmul <4 x half> %v97, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !10
  %v101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !10
  %v102 = fmul <4 x half> %v99, splat (half 1.275630e-01), !dbg !10
  %v103 = fmul <4 x half> %v101, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1600 = or disjoint i32 %13, 128, !dbg !8
  %v119 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1600, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %dma.buffer.offset1597 = or disjoint i32 %25, 128, !dbg !8
  %v133 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !10
  %v134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !10
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !10
  %v135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !10
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !10
  %v136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !10
  %v137 = fmul <4 x half> %v134, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !10
  %v138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !10
  %v139 = fmul <4 x half> %v135, splat (half 1.275630e-01), !dbg !10
  %v140 = fmul <4 x half> %v136, splat (half 1.275630e-01), !dbg !10
  %v141 = fmul <4 x half> %v138, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1594 = or disjoint i32 %13, 192, !dbg !8
  %v157 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %dma.buffer.offset1591 = or disjoint i32 %25, 192, !dbg !8
  %v171 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %v172 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %v173 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %v174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !10
  %v175 = fmul <4 x half> %v172, splat (half 1.275630e-01), !dbg !10
  %v176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !10
  %v177 = fmul <4 x half> %v173, splat (half 1.275630e-01), !dbg !10
  %v178 = fmul <4 x half> %v174, splat (half 1.275630e-01), !dbg !10
  %v179 = fmul <4 x half> %v176, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %v180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !10
  %v181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !10
  %v182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !10
  %v183 = fmul <4 x half> %v180, splat (half 1.275630e-01), !dbg !10
  %v184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !10
  %v185 = fmul <4 x half> %v181, splat (half 1.275630e-01), !dbg !10
  %v186 = fmul <4 x half> %v182, splat (half 1.275630e-01), !dbg !10
  %v187 = fmul <4 x half> %v184, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %v188 = or disjoint i64 %v25, 127, !dbg !12
  %v189 = add i64 %v25, 128, !dbg !12
  %v190 = icmp uge i64 %v189, %v188, !dbg !12
  %v191 = sub nsw i64 32, %v14, !dbg !13
  %v192 = shl i64 %v16, 7, !dbg !14
  %v193 = and i64 %v192, 3968, !dbg !14
  %v194 = shl nsw i64 %v63, 12, !dbg !14
  %v195 = or disjoint i64 %v193, %v194, !dbg !14
  %v197 = shl nuw nsw i32 %v31, 2, !dbg !15
  %v199 = and i32 %v197, 12, !dbg !15
  %v200 = and i32 %v31, 1, !dbg !15
  %v201 = or disjoint i32 %v199, %v200, !dbg !15
  %26 = trunc nuw nsw i32 %v201 to i16, !dbg !15
  %v203.lhs.trunc = add nuw nsw i16 %26, 1, !dbg !15
  %v2034950 = lshr i16 %v203.lhs.trunc, 1, !dbg !15
  %v203.sext = zext nneg i16 %v2034950 to i64, !dbg !15
  %v21251 = icmp ult i64 %v10, 1024, !dbg !11
  br i1 %v21251, label %v213.lr.ph, label %.v1780_crit_edge, !dbg !11

.v1780_crit_edge:                                 ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre66 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre67 = add nuw nsw i32 %.pre66, 64, !dbg !11
  %.pre68 = xor i32 %.pre66, 16, !dbg !11
  %.pre69 = and i32 %.pre67, 192, !dbg !11
  %.pre70 = xor i32 %.pre66, 32, !dbg !11
  br label %v1780, !dbg !11

v213.lr.ph:                                       ; preds = %4
  %v32 = zext nneg i32 %v31 to i64, !dbg !8
  %v230 = ptrtoint ptr addrspace(1) %1 to i64
  %27 = insertelement <2 x i64> poison, i64 %v230, i64 0
  %v237 = bitcast <2 x i64> %27 to <4 x i32>
  %dma.buffer.rsrc1589 = shufflevector <4 x i32> %v237, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %v609 = ptrtoint ptr addrspace(1) %2 to i64
  %28 = insertelement <2 x i64> poison, i64 %v609, i64 0
  %v616 = bitcast <2 x i64> %28 to <4 x i32>
  %dma.buffer.rsrc1565 = shufflevector <4 x i32> %v616, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %29 = lshr i64 %v32, 2
  %v704 = and i64 %29, 48
  %v706 = and i32 %v31, 240
  %index.compact = zext nneg i32 %v706 to i64
  %v703 = sub nsw i64 %v32, %index.compact
  %v705 = add nsw i64 %v703, %v704
  %v707 = add nsw i64 %v705, %v25
  %v708 = and i64 %v32, 48
  %v710 = add i64 %v707, 1
  %v711 = lshr i64 %v710, 32
  %compare.high = trunc nuw i64 %v711 to i32
  %compare.low.base = trunc i64 %v710 to i32
  %v745 = add i64 %v707, 65
  %v746 = lshr i64 %v745, 32
  %compare.high1281 = trunc nuw i64 %v746 to i32
  %compare.low.base1285 = trunc i64 %v745 to i32
  %30 = insertelement <4 x i32> poison, i32 %compare.low.base1285, i64 0, !dbg !16
  %31 = shufflevector <4 x i32> %30, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  %32 = insertelement <4 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %33 = shufflevector <4 x i32> %32, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  br label %v213, !dbg !11

v213:                                             ; preds = %v213.lr.ph, %v848
  %v21158 = phi [2 x <8 x float>] [ zeroinitializer, %v213.lr.ph ], [ %v1778, %v848 ]
  %v21057 = phi [2 x <8 x float>] [ zeroinitializer, %v213.lr.ph ], [ %v1774, %v848 ]
  %v20956 = phi [2 x <8 x float>] [ zeroinitializer, %v213.lr.ph ], [ %v1770, %v848 ]
  %v20855 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %v213.lr.ph ], [ %v1766, %v848 ]
  %v20754 = phi [2 x <1 x float>] [ zeroinitializer, %v213.lr.ph ], [ %v1370, %v848 ]
  %v20653 = phi [2 x <8 x float>] [ zeroinitializer, %v213.lr.ph ], [ %v1764, %v848 ]
  %v20552 = phi i64 [ 0, %v213.lr.ph ], [ %v1779, %v848 ]
  %v214 = shl nuw nsw i64 %v20552, 7, !dbg !11
  %v215 = lshr i64 %v20552, 12, !dbg !14
  %v216 = add nuw nsw i64 %v215, %v16, !dbg !14
  %v217 = lshr i64 %v20552, 5, !dbg !14
  %v218 = and i64 %v217, 127, !dbg !14
  %v219 = shl i64 %v216, 19, !dbg !14
  %v220 = and i64 %v219, 16252928, !dbg !14
  %v221 = shl i64 %v217, 19, !dbg !14
  %v222 = shl nuw nsw i64 %v20552, 14, !dbg !14
  %v223 = sub i64 %v222, %v221, !dbg !14
  %address.thread433 = tail call i32 asm sideeffect "", "=v,0"(i32 %v31) #8, !dbg !14, !range !9, !frisk.register_lifetimes !5
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
  %v224.idx6 = or disjoint i64 %v220, %v218, !dbg !14
  %34 = add i64 %v224.idx6, %v223, !dbg !14
  %35 = add i64 %address.local449, %34, !dbg !14
  %.tr = trunc i64 %35 to i32, !dbg !14
  %dma.buffer.offset1588 = shl i32 %.tr, 1, !dbg !14
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !8
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !8
  %address.local453 = and i64 %address.local452, 3584, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !8
  %address.local457 = lshr i64 %address.local451, 4, !dbg !14
  %address.local458 = and i64 %address.local457, 24, !dbg !14
  %address.local466 = or disjoint i64 %address.local448, %address.local458, !dbg !14
  %36 = add i64 %address.local466, %34, !dbg !14
  %.tr8 = trunc i64 %36 to i32, !dbg !14
  %dma.buffer.offset1585 = shl i32 %.tr8, 1, !dbg !14
  %v255 = add nuw nsw i64 %v20552, 1024, !dbg !14
  %v2599 = lshr i64 %v255, 12, !dbg !14
  %v261 = add nuw nsw i64 %v2599, %v16, !dbg !14
  %v262 = add nsw i64 %v217, %v195, !dbg !14
  %v272 = shl i64 %v261, 19, !dbg !14
  %v273 = and i64 %v272, 16252928, !dbg !14
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !8
  %v271 = add i64 %v223, %v217, !dbg !14
  %v274.idx10 = add i64 %v271, %v195, !dbg !14
  %37 = add i64 %v274.idx10, %v273, !dbg !14
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !8
  %38 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local437, i64 0, !dbg !8
  %39 = insertelement <4 x ptr addrspace(3)> %38, ptr addrspace(3) %address.local454, i64 1, !dbg !8
  %40 = insertelement <4 x ptr addrspace(3)> %39, ptr addrspace(3) %address.local468, i64 2, !dbg !8
  %41 = insertelement <4 x ptr addrspace(3)> %40, ptr addrspace(3) %address.local472, i64 3, !dbg !8
  %42 = ptrtoint <4 x ptr addrspace(3)> %41 to <4 x i64>, !dbg !8
  %43 = trunc <4 x i64> %42 to <4 x i32>, !dbg !8
  %44 = extractelement <4 x i32> %43, i64 0, !dbg !14
  %v227 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44), !dbg !14
  %dma.buffer.lds1590 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v227), !dbg !14
  %v240 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %45 = extractelement <4 x i32> %43, i64 1, !dbg !14
  %v241 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %45), !dbg !14
  %dma.buffer.lds1587 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v241), !dbg !14
  %v254 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %46 = extractelement <4 x i32> %43, i64 2, !dbg !14
  %v279 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !14
  %dma.buffer.lds1584 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v279), !dbg !14
  %47 = extractelement <4 x i32> %43, i64 3, !dbg !14
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !15
  %address.local481 = and i64 %address.local480, 24, !dbg !15
  %address.local483 = shl i64 %address.local434, 62, !dbg !15
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !15
  %address.local485 = and i64 %address.local484, -64, !dbg !15
  %address.local488 = and i64 %address.local443, 12, !dbg !10
  %48 = and i64 %address.local434, 32, !dbg !10
  %.not14 = icmp eq i64 %48, 0, !dbg !10
  %address.local492 = select i1 %.not14, i64 0, i64 1016, !dbg !10
  %.idx15 = mul nuw nsw i64 %address.local434, 320, !dbg !15
  %49 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx15, !dbg !15
  %.idx16 = mul nsw i64 %address.local443, -1280, !dbg !15
  %50 = getelementptr i8, ptr addrspace(3) %49, i64 %.idx16, !dbg !15
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
  %v371 = add nuw nsw i64 %v20552, 2048, !dbg !14
  %v37517 = lshr i64 %v371, 12, !dbg !14
  %v377 = add nuw nsw i64 %v37517, %v16, !dbg !14
  %54 = insertelement <2 x i64> poison, i64 %v262, i64 0, !dbg !14
  %55 = shufflevector <2 x i64> %54, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !14
  %56 = icmp slt <2 x i64> %55, <i64 -32, i64 -64>, !dbg !14
  %57 = extractelement <2 x i1> %56, i64 0, !dbg !14
  %v268 = sext i1 %57 to i64, !dbg !14
  %58 = add nsw <2 x i64> %55, <i64 32, i64 64>, !dbg !14
  %59 = sub nsw <2 x i64> <i64 -33, i64 -65>, %55, !dbg !14
  %60 = select <2 x i1> %56, <2 x i64> %59, <2 x i64> %58, !dbg !14
  %61 = sdiv <2 x i64> %60, splat (i64 128), !dbg !14
  %62 = extractelement <2 x i64> %61, i64 0, !dbg !14
  %v269 = xor i64 %62, %v268, !dbg !14
  %63 = shl nsw i64 %v269, 7, !dbg !14
  %64 = sub i64 %37, %63, !dbg !14
  %65 = add i64 %address.local449, %64, !dbg !14
  %.tr11 = trunc i64 %65 to i32, !dbg !14
  %66 = shl i32 %.tr11, 1, !dbg !14
  %dma.buffer.offset1582 = add i32 %66, 64, !dbg !14
  %v292 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %v293 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47), !dbg !14
  %67 = add i64 %address.local466, %64, !dbg !14
  %.tr13 = trunc i64 %67 to i32, !dbg !14
  %68 = shl i32 %.tr13, 1, !dbg !14
  %dma.buffer.offset1579 = add i32 %68, 64, !dbg !14
  %dma.buffer.lds1581 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v293), !dbg !14
  %v306 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %v307 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !15
  %v308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v307, <4 x float> zeroinitializer) #8, !dbg !15
  %v309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %v203.sext) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !15
  %address.local518 = shl i64 %address.local517, 6, !dbg !15
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !15
  %69 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local519, !dbg !15
  %70 = getelementptr [2 x i8], ptr addrspace(3) %69, i64 %address.local488, !dbg !15
  %71 = getelementptr [2 x i8], ptr addrspace(3) %70, i64 %address.local492, !dbg !15
  %address.local523 = getelementptr i8, ptr addrspace(3) %71, i64 64, !dbg !15
  %v310 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !15
  %v311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v309, <4 x float> %v308) #8, !dbg !15
  %v312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v310, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %71, i64 4160, !dbg !15
  %v313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !15
  %v314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !15
  %v315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v313, <4 x float> %v312) #8, !dbg !15
  %v316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v314, <4 x float> zeroinitializer) #8, !dbg !15
  %v317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !15
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %71, i64 192, !dbg !15
  %v318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !15
  %v319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v317, <4 x float> %v316) #8, !dbg !15
  %v320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v318, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %71, i64 4288, !dbg !15
  %v321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !15
  %v322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !15
  %v323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v321, <4 x float> %v320) #8, !dbg !15
  %v324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v322, <4 x float> zeroinitializer) #8, !dbg !15
  %v325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !15
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %71, i64 1088, !dbg !15
  %v326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !15
  %v327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v325, <4 x float> %v324) #8, !dbg !15
  %v328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v326, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %71, i64 5184, !dbg !15
  %v329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !15
  %v330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !15
  %v331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v329, <4 x float> %v328) #8, !dbg !15
  %v332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v330, <4 x float> zeroinitializer) #8, !dbg !15
  %v333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !15
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %71, i64 1216, !dbg !15
  %v334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !15
  %v335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v333, <4 x float> %v332) #8, !dbg !15
  %v336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v334, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %71, i64 5312, !dbg !15
  %v337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !15
  %v339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v337, <4 x float> %v336) #8, !dbg !15
  %v340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v307, <4 x float> zeroinitializer) #8, !dbg !15
  %v343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v309, <4 x float> %v340) #8, !dbg !15
  %v344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v310, <4 x float> zeroinitializer) #8, !dbg !15
  %v347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v313, <4 x float> %v344) #8, !dbg !15
  %v348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v314, <4 x float> zeroinitializer) #8, !dbg !15
  %v351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v317, <4 x float> %v348) #8, !dbg !15
  %v352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v318, <4 x float> zeroinitializer) #8, !dbg !15
  %v355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v321, <4 x float> %v352) #8, !dbg !15
  %v356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v322, <4 x float> zeroinitializer) #8, !dbg !15
  %v359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v325, <4 x float> %v356) #8, !dbg !15
  %v360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v326, <4 x float> zeroinitializer) #8, !dbg !15
  %v363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v329, <4 x float> %v360) #8, !dbg !15
  %v364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v330, <4 x float> zeroinitializer) #8, !dbg !15
  %v367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v333, <4 x float> %v364) #8, !dbg !15
  %v368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v334, <4 x float> zeroinitializer) #8, !dbg !15
  %v370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v337, <4 x float> %v368) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %72 = extractelement <2 x i1> %56, i64 1, !dbg !14
  %v383 = sext i1 %72 to i64, !dbg !14
  %73 = extractelement <2 x i64> %61, i64 1, !dbg !14
  %v384 = xor i64 %73, %v383, !dbg !14
  %v387 = shl i64 %v377, 19, !dbg !14
  %v388 = and i64 %v387, 16252928, !dbg !14
  %74 = add i64 %v274.idx10, %v388, !dbg !14
  %75 = shl nsw i64 %v384, 7, !dbg !14
  %76 = sub i64 %74, %75, !dbg !14
  %77 = add i64 %address.local449, %76, !dbg !14
  %.tr19 = trunc i64 %77 to i32, !dbg !14
  %78 = shl i32 %.tr19, 1, !dbg !14
  %dma.buffer.offset1576 = add i32 %78, 128, !dbg !14
  %v407 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %79 = add i64 %address.local466, %76, !dbg !14
  %.tr21 = trunc i64 %79 to i32, !dbg !14
  %80 = shl i32 %.tr21, 1, !dbg !14
  %dma.buffer.offset1573 = add i32 %80, 128, !dbg !14
  %v421 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !15
  %v422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !15
  %v423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v422, <4 x float> %v311) #8, !dbg !15
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !15
  %v424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !15
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %71, i64 8256, !dbg !15
  %v425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !15
  %v426 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v424, <4 x float> %v423) #8, !dbg !15
  %v427 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v425, <4 x float> %v315) #8, !dbg !15
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %71, i64 12352, !dbg !15
  %v428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !15
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !15
  %v429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !15
  %v430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v428, <4 x float> %v427) #8, !dbg !15
  %v431 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v429, <4 x float> %v319) #8, !dbg !15
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !15
  %v432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !15
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %71, i64 8384, !dbg !15
  %v433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !15
  %v434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v432, <4 x float> %v431) #8, !dbg !15
  %v435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v433, <4 x float> %v323) #8, !dbg !15
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %71, i64 12480, !dbg !15
  %v436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !15
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !15
  %v437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !15
  %v438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v436, <4 x float> %v435) #8, !dbg !15
  %v439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v437, <4 x float> %v327) #8, !dbg !15
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !15
  %v440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !15
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %71, i64 9280, !dbg !15
  %v441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !15
  %v442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v440, <4 x float> %v439) #8, !dbg !15
  %v443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v441, <4 x float> %v331) #8, !dbg !15
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %71, i64 13376, !dbg !15
  %v444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !15
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !15
  %v445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !15
  %v446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v444, <4 x float> %v443) #8, !dbg !15
  %v447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v445, <4 x float> %v335) #8, !dbg !15
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !15
  %v448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !15
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %71, i64 9408, !dbg !15
  %v449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !15
  %v450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v448, <4 x float> %v447) #8, !dbg !15
  %v451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v449, <4 x float> %v339) #8, !dbg !15
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %71, i64 13504, !dbg !15
  %v452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !15
  %v454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v452, <4 x float> %v451) #8, !dbg !15
  %v455 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v422, <4 x float> %v343) #8, !dbg !15
  %v458 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v424, <4 x float> %v455) #8, !dbg !15
  %v459 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v425, <4 x float> %v347) #8, !dbg !15
  %v462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v428, <4 x float> %v459) #8, !dbg !15
  %v463 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v429, <4 x float> %v351) #8, !dbg !15
  %v466 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v432, <4 x float> %v463) #8, !dbg !15
  %v467 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v433, <4 x float> %v355) #8, !dbg !15
  %v470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v436, <4 x float> %v467) #8, !dbg !15
  %v471 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v437, <4 x float> %v359) #8, !dbg !15
  %v474 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v440, <4 x float> %v471) #8, !dbg !15
  %v475 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v441, <4 x float> %v363) #8, !dbg !15
  %v478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v444, <4 x float> %v475) #8, !dbg !15
  %v479 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v445, <4 x float> %v367) #8, !dbg !15
  %v482 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v448, <4 x float> %v479) #8, !dbg !15
  %v483 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v449, <4 x float> %v370) #8, !dbg !15
  %v485 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v452, <4 x float> %v483) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %v486 = add nuw nsw i64 %v20552, 3072, !dbg !14
  %v49022 = lshr i64 %v486, 12, !dbg !14
  %v492 = add nuw nsw i64 %v49022, %v16, !dbg !14
  %v493 = add nsw i64 %v262, 96, !dbg !14
  %v494 = icmp slt i64 %v262, -96, !dbg !14
  %v495 = sub nsw i64 -97, %v262, !dbg !14
  %v496 = select i1 %v494, i64 %v495, i64 %v493, !dbg !14
  %v497 = sdiv i64 %v496, 128, !dbg !14
  %v498 = sext i1 %v494 to i64, !dbg !14
  %v499 = xor i64 %v497, %v498, !dbg !14
  %v502 = shl i64 %v492, 19, !dbg !14
  %v503 = and i64 %v502, 16252928, !dbg !14
  %81 = add i64 %v274.idx10, %v503, !dbg !14
  %82 = shl nsw i64 %v499, 7, !dbg !14
  %83 = sub i64 %81, %82, !dbg !14
  %84 = add i64 %address.local449, %83, !dbg !14
  %.tr24 = trunc i64 %84 to i32, !dbg !14
  %85 = shl i32 %.tr24, 1, !dbg !14
  %dma.buffer.offset1570 = add i32 %85, 192, !dbg !14
  %v522 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %86 = add i64 %address.local466, %83, !dbg !14
  %.tr26 = trunc i64 %86 to i32, !dbg !14
  %87 = shl i32 %.tr26, 1, !dbg !14
  %dma.buffer.offset1567 = add i32 %87, 192, !dbg !14
  %v536 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %v537 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !15
  %v538 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v537, <4 x float> %v426) #8, !dbg !15
  %v539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !15
  %v540 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !15
  %v541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v539, <4 x float> %v538) #8, !dbg !15
  %v542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v540, <4 x float> %v430) #8, !dbg !15
  %v543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !15
  %v544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !15
  %v545 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v543, <4 x float> %v542) #8, !dbg !15
  %v546 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v544, <4 x float> %v434) #8, !dbg !15
  %v547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !15
  %v548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !15
  %v549 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v547, <4 x float> %v546) #8, !dbg !15
  %v550 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v548, <4 x float> %v438) #8, !dbg !15
  %v551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !15
  %v552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !15
  %v553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v551, <4 x float> %v550) #8, !dbg !15
  %v554 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v552, <4 x float> %v442) #8, !dbg !15
  %v555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !15
  %v556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !15
  %v557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v555, <4 x float> %v554) #8, !dbg !15
  %v558 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v556, <4 x float> %v446) #8, !dbg !15
  %v559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !15
  %v560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !15
  %v561 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v559, <4 x float> %v558) #8, !dbg !15
  %v562 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v560, <4 x float> %v450) #8, !dbg !15
  %v563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !15
  %v564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !15
  %v565 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v563, <4 x float> %v562) #8, !dbg !15
  %v566 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v564, <4 x float> %v454) #8, !dbg !15
  %v567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !15
  %v569 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v567, <4 x float> %v566) #8, !dbg !15
  %v570 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v537, <4 x float> %v458) #8, !dbg !15
  %v573 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v539, <4 x float> %v570) #8, !dbg !15
  %v574 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v540, <4 x float> %v462) #8, !dbg !15
  %v577 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v543, <4 x float> %v574) #8, !dbg !15
  %v578 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v544, <4 x float> %v466) #8, !dbg !15
  %v581 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v547, <4 x float> %v578) #8, !dbg !15
  %v582 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v548, <4 x float> %v470) #8, !dbg !15
  %v585 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v551, <4 x float> %v582) #8, !dbg !15
  %v586 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v552, <4 x float> %v474) #8, !dbg !15
  %v589 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v555, <4 x float> %v586) #8, !dbg !15
  %v590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v556, <4 x float> %v478) #8, !dbg !15
  %v593 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v559, <4 x float> %v590) #8, !dbg !15
  %v594 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v560, <4 x float> %v482) #8, !dbg !15
  %v597 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v563, <4 x float> %v594) #8, !dbg !15
  %v598 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v564, <4 x float> %v485) #8, !dbg !15
  %v600 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v567, <4 x float> %v598) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %v601 = add nuw nsw i64 %v217, %v16, !dbg !17
  %v602 = shl i64 %v601, 19, !dbg !17
  %v603 = and i64 %v602, 16252928, !dbg !17
  %address.local852 = shl nuw nsw i64 %address.local434, 6, !dbg !17
  %address.local853 = and i64 %address.local852, 16128, !dbg !17
  %address.local855 = and i64 %address.local435, 24, !dbg !17
  %v604.idx28 = add i64 %v603, %v223, !dbg !17
  %88 = or disjoint i64 %address.local855, %v604.idx28, !dbg !17
  %89 = shl i64 %88, 1, !dbg !17
  %v611 = or disjoint i64 %89, %address.local853, !dbg !17
  %dma.buffer.offset1564 = trunc i64 %v611 to i32, !dbg !17
  %v619 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %address.local858 = shl nuw nsw i64 %address.local451, 6, !dbg !17
  %address.local859 = and i64 %address.local858, 32512, !dbg !17
  %v625 = or disjoint i64 %89, %address.local859, !dbg !17
  %dma.buffer.offset1561 = trunc i64 %v625 to i32, !dbg !17
  %v633 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %v634 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !15
  %v635 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v634, <4 x float> %v541) #8, !dbg !15
  %v636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !15
  %v637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !15
  %v638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v636, <4 x float> %v635) #8, !dbg !15
  %v639 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v637, <4 x float> %v545) #8, !dbg !15
  %v640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !15
  %v641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !15
  %v642 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v640, <4 x float> %v639) #8, !dbg !15
  %v643 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v641, <4 x float> %v549) #8, !dbg !15
  %v644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !15
  %v645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !15
  %v646 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v644, <4 x float> %v643) #8, !dbg !15
  %v647 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v645, <4 x float> %v553) #8, !dbg !15
  %v648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !15
  %v649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !15
  %v650 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v648, <4 x float> %v647) #8, !dbg !15
  %v651 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v649, <4 x float> %v557) #8, !dbg !15
  %v652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !15
  %v653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !15
  %v654 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v652, <4 x float> %v651) #8, !dbg !15
  %v655 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v653, <4 x float> %v561) #8, !dbg !15
  %v656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !15
  %v657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !15
  %v658 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v656, <4 x float> %v655) #8, !dbg !15
  %v659 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v657, <4 x float> %v565) #8, !dbg !15
  %v660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !15
  %v661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !15
  %v662 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v660, <4 x float> %v659) #8, !dbg !15
  %v663 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v661, <4 x float> %v569) #8, !dbg !15
  %v664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !15
  %v666 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v664, <4 x float> %v663) #8, !dbg !15
  %v667 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v634, <4 x float> %v573) #8, !dbg !15
  %v670 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v636, <4 x float> %v667) #8, !dbg !15
  %v671 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v637, <4 x float> %v577) #8, !dbg !15
  %v674 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v640, <4 x float> %v671) #8, !dbg !15
  %v675 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v641, <4 x float> %v581) #8, !dbg !15
  %v678 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v644, <4 x float> %v675) #8, !dbg !15
  %v679 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v645, <4 x float> %v585) #8, !dbg !15
  %v682 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v648, <4 x float> %v679) #8, !dbg !15
  %v683 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v649, <4 x float> %v589) #8, !dbg !15
  %v686 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v652, <4 x float> %v683) #8, !dbg !15
  %v687 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v653, <4 x float> %v593) #8, !dbg !15
  %v690 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v656, <4 x float> %v687) #8, !dbg !15
  %v691 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v657, <4 x float> %v597) #8, !dbg !15
  %v694 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v660, <4 x float> %v691) #8, !dbg !15
  %v695 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v661, <4 x float> %v600) #8, !dbg !15
  %v697 = or disjoint i64 %v214, 127, !dbg !12
  %v698 = icmp uge i64 %v25, %v697, !dbg !12
  %v699 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v664, <4 x float> %v695) #8, !dbg !15
  %v700 = and i1 %v190, %v698, !dbg !12
  br i1 %v700, label %v848, label %v702, !dbg !12

v702:                                             ; preds = %v213
  %v709 = or disjoint i64 %v214, %v708, !dbg !12
  %v712 = lshr i64 %v20552, 25, !dbg !16
  %compare.high1157 = trunc nuw i64 %v712 to i32, !dbg !16
  %compare.low.base1158 = trunc i64 %v709 to i32, !dbg !16
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1161 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1165 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #8, !dbg !16, !frisk.register_lifetimes !5
  %90 = insertelement <4 x i32> poison, i32 %compare.low.base1158, i64 0, !dbg !16
  %91 = insertelement <4 x i32> %90, i32 %compare.local.base, i64 1, !dbg !16
  %92 = insertelement <4 x i32> %91, i32 %compare.local.base1161, i64 2, !dbg !16
  %93 = insertelement <4 x i32> %92, i32 %compare.local.base1165, i64 3, !dbg !16
  %94 = or <4 x i32> %93, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %95 = icmp ugt <4 x i32> %33, %94, !dbg !16
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
  %96 = insertelement <4 x i32> poison, i32 %compare.local.base1249, i64 0, !dbg !16
  %97 = insertelement <4 x i32> %96, i32 %compare.local.base1253, i64 1, !dbg !16
  %98 = insertelement <4 x i32> %97, i32 %compare.local.base1257, i64 2, !dbg !16
  %99 = insertelement <4 x i32> %98, i32 %compare.local.base1261, i64 3, !dbg !16
  %100 = or <4 x i32> %99, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %101 = icmp ult <4 x i32> %100, %33, !dbg !16
  %102 = insertelement <4 x i32> poison, i32 %compare.local.base1233, i64 0, !dbg !16
  %103 = insertelement <4 x i32> %102, i32 %compare.local.base1237, i64 1, !dbg !16
  %104 = insertelement <4 x i32> %103, i32 %compare.local.base1241, i64 2, !dbg !16
  %105 = insertelement <4 x i32> %104, i32 %compare.local.base1245, i64 3, !dbg !16
  %106 = or <4 x i32> %105, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %107 = icmp ult <4 x i32> %106, %33, !dbg !16
  %108 = insertelement <4 x i32> poison, i32 %compare.local.base1217, i64 0, !dbg !16
  %109 = insertelement <4 x i32> %108, i32 %compare.local.base1221, i64 1, !dbg !16
  %110 = insertelement <4 x i32> %109, i32 %compare.local.base1225, i64 2, !dbg !16
  %111 = insertelement <4 x i32> %110, i32 %compare.local.base1229, i64 3, !dbg !16
  %112 = or <4 x i32> %111, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %113 = icmp ult <4 x i32> %112, %33, !dbg !16
  %114 = insertelement <4 x i32> poison, i32 %compare.local.base1201, i64 0, !dbg !16
  %115 = insertelement <4 x i32> %114, i32 %compare.local.base1205, i64 1, !dbg !16
  %116 = insertelement <4 x i32> %115, i32 %compare.local.base1209, i64 2, !dbg !16
  %117 = insertelement <4 x i32> %116, i32 %compare.local.base1213, i64 3, !dbg !16
  %118 = or <4 x i32> %117, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %119 = icmp ult <4 x i32> %118, %33, !dbg !16
  %120 = insertelement <4 x i32> poison, i32 %compare.local.base1185, i64 0, !dbg !16
  %121 = insertelement <4 x i32> %120, i32 %compare.local.base1189, i64 1, !dbg !16
  %122 = insertelement <4 x i32> %121, i32 %compare.local.base1193, i64 2, !dbg !16
  %123 = insertelement <4 x i32> %122, i32 %compare.local.base1197, i64 3, !dbg !16
  %124 = or <4 x i32> %123, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %125 = icmp ult <4 x i32> %124, %33, !dbg !16
  %126 = insertelement <4 x i32> poison, i32 %compare.local.base1169, i64 0, !dbg !16
  %127 = insertelement <4 x i32> %126, i32 %compare.local.base1173, i64 1, !dbg !16
  %128 = insertelement <4 x i32> %127, i32 %compare.local.base1177, i64 2, !dbg !16
  %129 = insertelement <4 x i32> %128, i32 %compare.local.base1181, i64 3, !dbg !16
  %130 = or <4 x i32> %129, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %131 = icmp ult <4 x i32> %130, %33, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !16
  %132 = insertelement <4 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %133 = shufflevector <4 x i1> %132, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %134 = insertelement <4 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %135 = shufflevector <4 x i1> %134, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %136 = select <4 x i1> %133, <4 x i1> %95, <4 x i1> %135, !dbg !16
  %137 = select <4 x i1> %136, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %138 = select <4 x i1> %133, <4 x i1> %131, <4 x i1> %135, !dbg !16
  %139 = select <4 x i1> %138, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %140 = select <4 x i1> %133, <4 x i1> %125, <4 x i1> %135, !dbg !16
  %141 = select <4 x i1> %140, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %142 = select <4 x i1> %133, <4 x i1> %119, <4 x i1> %135, !dbg !16
  %143 = select <4 x i1> %142, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %144 = select <4 x i1> %133, <4 x i1> %113, <4 x i1> %135, !dbg !16
  %145 = select <4 x i1> %144, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %146 = select <4 x i1> %133, <4 x i1> %107, <4 x i1> %135, !dbg !16
  %147 = select <4 x i1> %146, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %148 = select <4 x i1> %133, <4 x i1> %101, <4 x i1> %135, !dbg !16
  %149 = select <4 x i1> %148, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %150 = insertelement <4 x i32> poison, i32 %compare.local.base1265, i64 0, !dbg !16
  %151 = insertelement <4 x i32> %150, i32 %compare.local.base1269, i64 1, !dbg !16
  %152 = insertelement <4 x i32> %151, i32 %compare.local.base1273, i64 2, !dbg !16
  %153 = insertelement <4 x i32> %152, i32 %compare.local.base1277, i64 3, !dbg !16
  %154 = or <4 x i32> %153, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %155 = icmp ult <4 x i32> %154, %33, !dbg !16
  %156 = select <4 x i1> %133, <4 x i1> %155, <4 x i1> %135, !dbg !16
  %157 = select <4 x i1> %156, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %compare.local.base1289 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 32) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1293 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 33) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1297 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 34) #8, !dbg !16, !frisk.register_lifetimes !5
  %158 = insertelement <4 x i32> %90, i32 %compare.local.base1289, i64 1, !dbg !16
  %159 = insertelement <4 x i32> %158, i32 %compare.local.base1293, i64 2, !dbg !16
  %160 = insertelement <4 x i32> %159, i32 %compare.local.base1297, i64 3, !dbg !16
  %161 = or <4 x i32> %160, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %162 = icmp ugt <4 x i32> %31, %161, !dbg !16
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
  %163 = insertelement <4 x i32> poison, i32 %compare.local.base1381, i64 0, !dbg !16
  %164 = insertelement <4 x i32> %163, i32 %compare.local.base1385, i64 1, !dbg !16
  %165 = insertelement <4 x i32> %164, i32 %compare.local.base1389, i64 2, !dbg !16
  %166 = insertelement <4 x i32> %165, i32 %compare.local.base1393, i64 3, !dbg !16
  %167 = or <4 x i32> %166, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %168 = icmp ult <4 x i32> %167, %31, !dbg !16
  %169 = insertelement <4 x i32> poison, i32 %compare.local.base1365, i64 0, !dbg !16
  %170 = insertelement <4 x i32> %169, i32 %compare.local.base1369, i64 1, !dbg !16
  %171 = insertelement <4 x i32> %170, i32 %compare.local.base1373, i64 2, !dbg !16
  %172 = insertelement <4 x i32> %171, i32 %compare.local.base1377, i64 3, !dbg !16
  %173 = or <4 x i32> %172, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %174 = icmp ult <4 x i32> %173, %31, !dbg !16
  %175 = insertelement <4 x i32> poison, i32 %compare.local.base1349, i64 0, !dbg !16
  %176 = insertelement <4 x i32> %175, i32 %compare.local.base1353, i64 1, !dbg !16
  %177 = insertelement <4 x i32> %176, i32 %compare.local.base1357, i64 2, !dbg !16
  %178 = insertelement <4 x i32> %177, i32 %compare.local.base1361, i64 3, !dbg !16
  %179 = or <4 x i32> %178, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %180 = icmp ult <4 x i32> %179, %31, !dbg !16
  %181 = insertelement <4 x i32> poison, i32 %compare.local.base1333, i64 0, !dbg !16
  %182 = insertelement <4 x i32> %181, i32 %compare.local.base1337, i64 1, !dbg !16
  %183 = insertelement <4 x i32> %182, i32 %compare.local.base1341, i64 2, !dbg !16
  %184 = insertelement <4 x i32> %183, i32 %compare.local.base1345, i64 3, !dbg !16
  %185 = or <4 x i32> %184, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %186 = icmp ult <4 x i32> %185, %31, !dbg !16
  %187 = insertelement <4 x i32> poison, i32 %compare.local.base1317, i64 0, !dbg !16
  %188 = insertelement <4 x i32> %187, i32 %compare.local.base1321, i64 1, !dbg !16
  %189 = insertelement <4 x i32> %188, i32 %compare.local.base1325, i64 2, !dbg !16
  %190 = insertelement <4 x i32> %189, i32 %compare.local.base1329, i64 3, !dbg !16
  %191 = or <4 x i32> %190, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %192 = icmp ult <4 x i32> %191, %31, !dbg !16
  %193 = insertelement <4 x i32> poison, i32 %compare.local.base1301, i64 0, !dbg !16
  %194 = insertelement <4 x i32> %193, i32 %compare.local.base1305, i64 1, !dbg !16
  %195 = insertelement <4 x i32> %194, i32 %compare.local.base1309, i64 2, !dbg !16
  %196 = insertelement <4 x i32> %195, i32 %compare.local.base1313, i64 3, !dbg !16
  %197 = or <4 x i32> %196, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %198 = icmp ult <4 x i32> %197, %31, !dbg !16
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1157, !dbg !16
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1157, !dbg !16
  %199 = insertelement <4 x i1> poison, i1 %compare.high.equal1283, i64 0, !dbg !16
  %200 = shufflevector <4 x i1> %199, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %201 = insertelement <4 x i1> poison, i1 %compare.high.order1284, i64 0, !dbg !16
  %202 = shufflevector <4 x i1> %201, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %203 = select <4 x i1> %200, <4 x i1> %162, <4 x i1> %202, !dbg !16
  %204 = select <4 x i1> %203, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %205 = select <4 x i1> %200, <4 x i1> %198, <4 x i1> %202, !dbg !16
  %206 = select <4 x i1> %205, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %207 = select <4 x i1> %200, <4 x i1> %192, <4 x i1> %202, !dbg !16
  %208 = select <4 x i1> %207, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %209 = select <4 x i1> %200, <4 x i1> %186, <4 x i1> %202, !dbg !16
  %210 = select <4 x i1> %209, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %211 = select <4 x i1> %200, <4 x i1> %180, <4 x i1> %202, !dbg !16
  %212 = select <4 x i1> %211, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %213 = select <4 x i1> %200, <4 x i1> %174, <4 x i1> %202, !dbg !16
  %214 = select <4 x i1> %213, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %215 = select <4 x i1> %200, <4 x i1> %168, <4 x i1> %202, !dbg !16
  %216 = select <4 x i1> %215, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %217 = insertelement <4 x i32> poison, i32 %compare.local.base1397, i64 0, !dbg !16
  %218 = insertelement <4 x i32> %217, i32 %compare.local.base1401, i64 1, !dbg !16
  %219 = insertelement <4 x i32> %218, i32 %compare.local.base1405, i64 2, !dbg !16
  %220 = insertelement <4 x i32> %219, i32 %compare.local.base1409, i64 3, !dbg !16
  %221 = or <4 x i32> %220, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %222 = icmp ult <4 x i32> %221, %31, !dbg !16
  %223 = select <4 x i1> %200, <4 x i1> %222, <4 x i1> %202, !dbg !16
  %224 = select <4 x i1> %223, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %v851.masked = fadd <4 x float> %v638, %137, !dbg !19
  %v857.masked = fadd <4 x float> %v642, %139, !dbg !19
  %v863.masked = fadd <4 x float> %v646, %141, !dbg !19
  %v869.masked = fadd <4 x float> %v650, %143, !dbg !19
  %v875.masked = fadd <4 x float> %v654, %145, !dbg !19
  %v881.masked = fadd <4 x float> %v658, %147, !dbg !19
  %v887.masked = fadd <4 x float> %v662, %149, !dbg !19
  %v893.masked = fadd <4 x float> %v666, %157, !dbg !19
  %v900.masked = fadd <4 x float> %v670, %204, !dbg !19
  %v906.masked = fadd <4 x float> %v674, %206, !dbg !19
  %v912.masked = fadd <4 x float> %v678, %208, !dbg !19
  %v918.masked = fadd <4 x float> %v682, %210, !dbg !19
  %v924.masked = fadd <4 x float> %v686, %212, !dbg !19
  %v930.masked = fadd <4 x float> %v690, %214, !dbg !19
  %v936.masked = fadd <4 x float> %v694, %216, !dbg !19
  %v942.masked = fadd <4 x float> %v699, %224, !dbg !19
  br label %v848, !dbg !12

v848:                                             ; preds = %v702, %v213
  %v851 = phi <4 x float> [ %v851.masked, %v702 ], [ %v638, %v213 ], !dbg !19
  %v857 = phi <4 x float> [ %v857.masked, %v702 ], [ %v642, %v213 ], !dbg !19
  %v863 = phi <4 x float> [ %v863.masked, %v702 ], [ %v646, %v213 ], !dbg !19
  %v869 = phi <4 x float> [ %v869.masked, %v702 ], [ %v650, %v213 ], !dbg !19
  %v875 = phi <4 x float> [ %v875.masked, %v702 ], [ %v654, %v213 ], !dbg !19
  %v881 = phi <4 x float> [ %v881.masked, %v702 ], [ %v658, %v213 ], !dbg !19
  %v887 = phi <4 x float> [ %v887.masked, %v702 ], [ %v662, %v213 ], !dbg !19
  %v893 = phi <4 x float> [ %v893.masked, %v702 ], [ %v666, %v213 ], !dbg !19
  %v900 = phi <4 x float> [ %v900.masked, %v702 ], [ %v670, %v213 ], !dbg !19
  %v906 = phi <4 x float> [ %v906.masked, %v702 ], [ %v674, %v213 ], !dbg !19
  %v912 = phi <4 x float> [ %v912.masked, %v702 ], [ %v678, %v213 ], !dbg !19
  %v918 = phi <4 x float> [ %v918.masked, %v702 ], [ %v682, %v213 ], !dbg !19
  %v924 = phi <4 x float> [ %v924.masked, %v702 ], [ %v686, %v213 ], !dbg !19
  %v930 = phi <4 x float> [ %v930.masked, %v702 ], [ %v690, %v213 ], !dbg !19
  %v936 = phi <4 x float> [ %v936.masked, %v702 ], [ %v694, %v213 ], !dbg !19
  %v942 = phi <4 x float> [ %v942.masked, %v702 ], [ %v699, %v213 ], !dbg !19
  %v852 = extractelement <4 x float> %v851, i64 0, !dbg !19
  %v853 = extractelement <4 x float> %v851, i64 1, !dbg !19
  %v854 = extractelement <4 x float> %v851, i64 2, !dbg !19
  %v855 = extractelement <4 x float> %v851, i64 3, !dbg !19
  %v858 = extractelement <4 x float> %v857, i64 0, !dbg !19
  %v859 = extractelement <4 x float> %v857, i64 1, !dbg !19
  %v860 = extractelement <4 x float> %v857, i64 2, !dbg !19
  %v861 = extractelement <4 x float> %v857, i64 3, !dbg !19
  %v864 = extractelement <4 x float> %v863, i64 0, !dbg !19
  %v865 = extractelement <4 x float> %v863, i64 1, !dbg !19
  %v866 = extractelement <4 x float> %v863, i64 2, !dbg !19
  %v867 = extractelement <4 x float> %v863, i64 3, !dbg !19
  %v870 = extractelement <4 x float> %v869, i64 0, !dbg !19
  %v871 = extractelement <4 x float> %v869, i64 1, !dbg !19
  %v872 = extractelement <4 x float> %v869, i64 2, !dbg !19
  %v873 = extractelement <4 x float> %v869, i64 3, !dbg !19
  %v876 = extractelement <4 x float> %v875, i64 0, !dbg !19
  %v877 = extractelement <4 x float> %v875, i64 1, !dbg !19
  %v878 = extractelement <4 x float> %v875, i64 2, !dbg !19
  %v879 = extractelement <4 x float> %v875, i64 3, !dbg !19
  %v882 = extractelement <4 x float> %v881, i64 0, !dbg !19
  %v883 = extractelement <4 x float> %v881, i64 1, !dbg !19
  %v884 = extractelement <4 x float> %v881, i64 2, !dbg !19
  %v885 = extractelement <4 x float> %v881, i64 3, !dbg !19
  %v888 = extractelement <4 x float> %v887, i64 0, !dbg !19
  %v889 = extractelement <4 x float> %v887, i64 1, !dbg !19
  %v890 = extractelement <4 x float> %v887, i64 2, !dbg !19
  %v891 = extractelement <4 x float> %v887, i64 3, !dbg !19
  %v894 = extractelement <4 x float> %v893, i64 0, !dbg !19
  %v895 = extractelement <4 x float> %v893, i64 1, !dbg !19
  %v896 = extractelement <4 x float> %v893, i64 2, !dbg !19
  %v897 = extractelement <4 x float> %v893, i64 3, !dbg !19
  %v901 = extractelement <4 x float> %v900, i64 0, !dbg !19
  %v902 = extractelement <4 x float> %v900, i64 1, !dbg !19
  %v903 = extractelement <4 x float> %v900, i64 2, !dbg !19
  %v904 = extractelement <4 x float> %v900, i64 3, !dbg !19
  %v907 = extractelement <4 x float> %v906, i64 0, !dbg !19
  %v908 = extractelement <4 x float> %v906, i64 1, !dbg !19
  %v909 = extractelement <4 x float> %v906, i64 2, !dbg !19
  %v910 = extractelement <4 x float> %v906, i64 3, !dbg !19
  %v913 = extractelement <4 x float> %v912, i64 0, !dbg !19
  %v914 = extractelement <4 x float> %v912, i64 1, !dbg !19
  %v915 = extractelement <4 x float> %v912, i64 2, !dbg !19
  %v916 = extractelement <4 x float> %v912, i64 3, !dbg !19
  %v919 = extractelement <4 x float> %v918, i64 0, !dbg !19
  %v920 = extractelement <4 x float> %v918, i64 1, !dbg !19
  %v921 = extractelement <4 x float> %v918, i64 2, !dbg !19
  %v922 = extractelement <4 x float> %v918, i64 3, !dbg !19
  %v925 = extractelement <4 x float> %v924, i64 0, !dbg !19
  %v926 = extractelement <4 x float> %v924, i64 1, !dbg !19
  %v927 = extractelement <4 x float> %v924, i64 2, !dbg !19
  %v928 = extractelement <4 x float> %v924, i64 3, !dbg !19
  %v931 = extractelement <4 x float> %v930, i64 0, !dbg !19
  %v932 = extractelement <4 x float> %v930, i64 1, !dbg !19
  %v933 = extractelement <4 x float> %v930, i64 2, !dbg !19
  %v934 = extractelement <4 x float> %v930, i64 3, !dbg !19
  %v937 = extractelement <4 x float> %v936, i64 0, !dbg !19
  %v938 = extractelement <4 x float> %v936, i64 1, !dbg !19
  %v939 = extractelement <4 x float> %v936, i64 2, !dbg !19
  %v940 = extractelement <4 x float> %v936, i64 3, !dbg !19
  %v943 = extractelement <4 x float> %v942, i64 0, !dbg !19
  %v944 = extractelement <4 x float> %v942, i64 1, !dbg !19
  %v945 = extractelement <4 x float> %v942, i64 2, !dbg !19
  %v946 = extractelement <4 x float> %v942, i64 3, !dbg !19
  %v947 = tail call float @llvm.maxnum.f32(float %v852, float -inf), !dbg !20
  %v948 = tail call float @llvm.maxnum.f32(float %v947, float %v853), !dbg !20
  %v949 = tail call float @llvm.maxnum.f32(float %v948, float %v854), !dbg !20
  %v950 = tail call float @llvm.maxnum.f32(float %v949, float %v855), !dbg !20
  %v951 = tail call float @llvm.maxnum.f32(float %v950, float %v858), !dbg !20
  %v952 = tail call float @llvm.maxnum.f32(float %v951, float %v859), !dbg !20
  %v953 = tail call float @llvm.maxnum.f32(float %v952, float %v860), !dbg !20
  %v954 = tail call float @llvm.maxnum.f32(float %v953, float %v861), !dbg !20
  %v955 = tail call float @llvm.maxnum.f32(float %v954, float %v864), !dbg !20
  %v956 = tail call float @llvm.maxnum.f32(float %v955, float %v865), !dbg !20
  %v957 = tail call float @llvm.maxnum.f32(float %v956, float %v866), !dbg !20
  %v958 = tail call float @llvm.maxnum.f32(float %v957, float %v867), !dbg !20
  %v959 = tail call float @llvm.maxnum.f32(float %v958, float %v870), !dbg !20
  %v960 = tail call float @llvm.maxnum.f32(float %v959, float %v871), !dbg !20
  %v961 = tail call float @llvm.maxnum.f32(float %v960, float %v872), !dbg !20
  %v962 = tail call float @llvm.maxnum.f32(float %v961, float %v873), !dbg !20
  %v963 = tail call float @llvm.maxnum.f32(float %v962, float %v876), !dbg !20
  %v964 = tail call float @llvm.maxnum.f32(float %v963, float %v877), !dbg !20
  %v965 = tail call float @llvm.maxnum.f32(float %v964, float %v878), !dbg !20
  %v966 = tail call float @llvm.maxnum.f32(float %v965, float %v879), !dbg !20
  %v967 = tail call float @llvm.maxnum.f32(float %v966, float %v882), !dbg !20
  %v968 = tail call float @llvm.maxnum.f32(float %v967, float %v883), !dbg !20
  %v969 = tail call float @llvm.maxnum.f32(float %v968, float %v884), !dbg !20
  %v970 = tail call float @llvm.maxnum.f32(float %v969, float %v885), !dbg !20
  %v971 = tail call float @llvm.maxnum.f32(float %v970, float %v888), !dbg !20
  %v972 = tail call float @llvm.maxnum.f32(float %v971, float %v889), !dbg !20
  %v973 = tail call float @llvm.maxnum.f32(float %v972, float %v890), !dbg !20
  %v974 = tail call float @llvm.maxnum.f32(float %v973, float %v891), !dbg !20
  %v975 = tail call float @llvm.maxnum.f32(float %v974, float %v894), !dbg !20
  %v976 = tail call float @llvm.maxnum.f32(float %v975, float %v895), !dbg !20
  %v977 = tail call float @llvm.maxnum.f32(float %v976, float %v896), !dbg !20
  %v978 = tail call float @llvm.maxnum.f32(float %v977, float %v897), !dbg !20
  %v979 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %v980 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %v979), !dbg !20
  %v981 = add nuw nsw i32 %v980, 64, !dbg !20
  %v982 = xor i32 %v980, 16, !dbg !20
  %v983 = and i32 %v981, 192, !dbg !20
  %v984 = icmp samesign ult i32 %v982, %v983, !dbg !20
  %v985 = select i1 %v984, i32 %v982, i32 %v980, !dbg !20
  %v986 = shl nuw nsw i32 %v985, 2, !dbg !20
  %v987 = bitcast float %v978 to i32, !dbg !20
  %v988 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v987), !dbg !20
  %v989 = bitcast i32 %v988 to float, !dbg !20
  %v990 = tail call float @llvm.maxnum.f32(float %v978, float %v989), !dbg !20
  %v991 = xor i32 %v980, 32, !dbg !20
  %v992 = icmp samesign ult i32 %v991, %v983, !dbg !20
  %v993 = select i1 %v992, i32 %v991, i32 %v980, !dbg !20
  %v994 = shl nuw nsw i32 %v993, 2, !dbg !20
  %v995 = bitcast float %v990 to i32, !dbg !20
  %v996 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v995), !dbg !20
  %v997 = bitcast i32 %v996 to float, !dbg !20
  %v998 = tail call float @llvm.maxnum.f32(float %v990, float %v997), !dbg !20
  %v999 = extractvalue [2 x <1 x float>] %v20855, 0, !dbg !20
  %v1000 = insertelement <1 x float> poison, float %v998, i64 0, !dbg !20
  %v1001 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %v1000, <1 x float> %v999), !dbg !20
  %v1002 = tail call float @llvm.maxnum.f32(float %v901, float -inf), !dbg !20
  %v1003 = tail call float @llvm.maxnum.f32(float %v1002, float %v902), !dbg !20
  %v1004 = tail call float @llvm.maxnum.f32(float %v1003, float %v903), !dbg !20
  %v1005 = tail call float @llvm.maxnum.f32(float %v1004, float %v904), !dbg !20
  %v1006 = tail call float @llvm.maxnum.f32(float %v1005, float %v907), !dbg !20
  %v1007 = tail call float @llvm.maxnum.f32(float %v1006, float %v908), !dbg !20
  %v1008 = tail call float @llvm.maxnum.f32(float %v1007, float %v909), !dbg !20
  %v1009 = tail call float @llvm.maxnum.f32(float %v1008, float %v910), !dbg !20
  %v1010 = tail call float @llvm.maxnum.f32(float %v1009, float %v913), !dbg !20
  %v1011 = tail call float @llvm.maxnum.f32(float %v1010, float %v914), !dbg !20
  %v1012 = tail call float @llvm.maxnum.f32(float %v1011, float %v915), !dbg !20
  %v1013 = tail call float @llvm.maxnum.f32(float %v1012, float %v916), !dbg !20
  %v1014 = tail call float @llvm.maxnum.f32(float %v1013, float %v919), !dbg !20
  %v1015 = tail call float @llvm.maxnum.f32(float %v1014, float %v920), !dbg !20
  %v1016 = tail call float @llvm.maxnum.f32(float %v1015, float %v921), !dbg !20
  %v1017 = tail call float @llvm.maxnum.f32(float %v1016, float %v922), !dbg !20
  %v1018 = tail call float @llvm.maxnum.f32(float %v1017, float %v925), !dbg !20
  %v1019 = tail call float @llvm.maxnum.f32(float %v1018, float %v926), !dbg !20
  %v1020 = tail call float @llvm.maxnum.f32(float %v1019, float %v927), !dbg !20
  %v1021 = tail call float @llvm.maxnum.f32(float %v1020, float %v928), !dbg !20
  %v1022 = tail call float @llvm.maxnum.f32(float %v1021, float %v931), !dbg !20
  %v1023 = tail call float @llvm.maxnum.f32(float %v1022, float %v932), !dbg !20
  %v1024 = tail call float @llvm.maxnum.f32(float %v1023, float %v933), !dbg !20
  %v1025 = tail call float @llvm.maxnum.f32(float %v1024, float %v934), !dbg !20
  %v1026 = tail call float @llvm.maxnum.f32(float %v1025, float %v937), !dbg !20
  %v1027 = tail call float @llvm.maxnum.f32(float %v1026, float %v938), !dbg !20
  %v1028 = tail call float @llvm.maxnum.f32(float %v1027, float %v939), !dbg !20
  %v1029 = tail call float @llvm.maxnum.f32(float %v1028, float %v940), !dbg !20
  %v1030 = tail call float @llvm.maxnum.f32(float %v1029, float %v943), !dbg !20
  %v1031 = tail call float @llvm.maxnum.f32(float %v1030, float %v944), !dbg !20
  %v1032 = tail call float @llvm.maxnum.f32(float %v1031, float %v945), !dbg !20
  %v1033 = tail call float @llvm.maxnum.f32(float %v1032, float %v946), !dbg !20
  %v1034 = bitcast float %v1033 to i32, !dbg !20
  %v1035 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v1034), !dbg !20
  %v1036 = bitcast i32 %v1035 to float, !dbg !20
  %v1037 = tail call float @llvm.maxnum.f32(float %v1033, float %v1036), !dbg !20
  %v1038 = bitcast float %v1037 to i32, !dbg !20
  %v1039 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v1038), !dbg !20
  %v1040 = bitcast i32 %v1039 to float, !dbg !20
  %v1041 = tail call float @llvm.maxnum.f32(float %v1037, float %v1040), !dbg !20
  %v1042 = extractvalue [2 x <1 x float>] %v20855, 1, !dbg !20
  %v1043 = insertelement <1 x float> poison, float %v1041, i64 0, !dbg !20
  %v1044 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %v1043, <1 x float> %v1042), !dbg !20
  %v1045 = shufflevector <1 x float> %v1001, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %v1046 = fsub <4 x float> %v851, %v1045, !dbg !21
  %v1047 = fsub <4 x float> %v857, %v1045, !dbg !21
  %v1048 = fsub <4 x float> %v863, %v1045, !dbg !21
  %v1049 = fsub <4 x float> %v869, %v1045, !dbg !21
  %v1050 = fsub <4 x float> %v875, %v1045, !dbg !21
  %v1051 = fsub <4 x float> %v881, %v1045, !dbg !21
  %v1052 = fsub <4 x float> %v887, %v1045, !dbg !21
  %v1053 = fsub <4 x float> %v893, %v1045, !dbg !21
  %v1054 = shufflevector <1 x float> %v1044, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %v1055 = fsub <4 x float> %v900, %v1054, !dbg !21
  %v1056 = fsub <4 x float> %v906, %v1054, !dbg !21
  %v1057 = fsub <4 x float> %v912, %v1054, !dbg !21
  %v1058 = fsub <4 x float> %v918, %v1054, !dbg !21
  %v1059 = fsub <4 x float> %v924, %v1054, !dbg !21
  %v1060 = fsub <4 x float> %v930, %v1054, !dbg !21
  %v1061 = fsub <4 x float> %v936, %v1054, !dbg !21
  %v1062 = fsub <4 x float> %v942, %v1054, !dbg !21
  %v1063 = fsub <1 x float> %v999, %v1001, !dbg !22
  %v1064 = extractelement <1 x float> %v1063, i64 0, !dbg !23
  %v1065 = tail call float @llvm.amdgcn.exp2.f32(float %v1064), !dbg !23
  %v1066 = insertelement <4 x float> poison, float %v1065, i64 0, !dbg !24
  %v1067 = shufflevector <4 x float> %v1066, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %v1068 = extractvalue [2 x <1 x float>] %v20754, 0, !dbg !25
  %v1069 = extractelement <1 x float> %v1068, i64 0, !dbg !25
  %v1070 = fmul float %v1069, %v1065, !dbg !25
  %v1071 = fsub <1 x float> %v1042, %v1044, !dbg !22
  %v1072 = extractelement <1 x float> %v1071, i64 0, !dbg !23
  %v1073 = tail call float @llvm.amdgcn.exp2.f32(float %v1072), !dbg !23
  %v1074 = insertelement <4 x float> poison, float %v1073, i64 0, !dbg !24
  %v1075 = shufflevector <4 x float> %v1074, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %v1076 = extractvalue [2 x <1 x float>] %v20754, 1, !dbg !25
  %v1077 = extractelement <1 x float> %v1076, i64 0, !dbg !25
  %v1078 = fmul float %v1077, %v1073, !dbg !25
  %v1079 = extractelement <4 x float> %v1046, i64 0, !dbg !26
  %v1080 = tail call float @llvm.amdgcn.exp2.f32(float %v1079), !dbg !26
  %v1081 = insertelement <4 x float> poison, float %v1080, i64 0, !dbg !27
  %v1082 = extractelement <4 x float> %v1046, i64 1, !dbg !26
  %v1083 = tail call float @llvm.amdgcn.exp2.f32(float %v1082), !dbg !26
  %v1084 = insertelement <4 x float> %v1081, float %v1083, i64 1, !dbg !27
  %v1085 = fadd float %v1080, %v1083, !dbg !28
  %v1086 = extractelement <4 x float> %v1046, i64 2, !dbg !26
  %v1087 = tail call float @llvm.amdgcn.exp2.f32(float %v1086), !dbg !26
  %v1088 = insertelement <4 x float> %v1084, float %v1087, i64 2, !dbg !27
  %v1089 = fadd float %v1087, %v1085, !dbg !28
  %v1090 = extractelement <4 x float> %v1046, i64 3, !dbg !26
  %v1091 = tail call float @llvm.amdgcn.exp2.f32(float %v1090), !dbg !26
  %v1092 = insertelement <4 x float> %v1088, float %v1091, i64 3, !dbg !27
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1092) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1093 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1093) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1094 = fadd float %v1091, %v1089, !dbg !28
  %v1095 = extractelement <4 x float> %v1047, i64 0, !dbg !26
  %v1096 = tail call float @llvm.amdgcn.exp2.f32(float %v1095), !dbg !26
  %v1097 = insertelement <4 x float> poison, float %v1096, i64 0, !dbg !27
  %v1098 = fadd float %v1096, %v1094, !dbg !28
  %v1099 = extractelement <4 x float> %v1047, i64 1, !dbg !26
  %v1100 = tail call float @llvm.amdgcn.exp2.f32(float %v1099), !dbg !26
  %v1101 = insertelement <4 x float> %v1097, float %v1100, i64 1, !dbg !27
  %v1102 = fadd float %v1100, %v1098, !dbg !28
  %v1103 = extractelement <4 x float> %v1047, i64 2, !dbg !26
  %v1104 = tail call float @llvm.amdgcn.exp2.f32(float %v1103), !dbg !26
  %v1105 = insertelement <4 x float> %v1101, float %v1104, i64 2, !dbg !27
  %v1106 = fadd float %v1104, %v1102, !dbg !28
  %v1107 = extractelement <4 x float> %v1047, i64 3, !dbg !26
  %v1108 = tail call float @llvm.amdgcn.exp2.f32(float %v1107), !dbg !26
  %v1109 = insertelement <4 x float> %v1105, float %v1108, i64 3, !dbg !27
  %math.result309 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1109) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1110 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !27
  %math.packed310 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1110) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1111 = fadd float %v1108, %v1106, !dbg !28
  %v1112 = extractelement <4 x float> %v1048, i64 0, !dbg !26
  %v1113 = tail call float @llvm.amdgcn.exp2.f32(float %v1112), !dbg !26
  %v1114 = insertelement <4 x float> poison, float %v1113, i64 0, !dbg !27
  %v1115 = fadd float %v1113, %v1111, !dbg !28
  %v1116 = extractelement <4 x float> %v1048, i64 1, !dbg !26
  %v1117 = tail call float @llvm.amdgcn.exp2.f32(float %v1116), !dbg !26
  %v1118 = insertelement <4 x float> %v1114, float %v1117, i64 1, !dbg !27
  %v1119 = fadd float %v1117, %v1115, !dbg !28
  %v1120 = extractelement <4 x float> %v1048, i64 2, !dbg !26
  %v1121 = tail call float @llvm.amdgcn.exp2.f32(float %v1120), !dbg !26
  %v1122 = insertelement <4 x float> %v1118, float %v1121, i64 2, !dbg !27
  %v1123 = fadd float %v1121, %v1119, !dbg !28
  %v1124 = extractelement <4 x float> %v1048, i64 3, !dbg !26
  %v1125 = tail call float @llvm.amdgcn.exp2.f32(float %v1124), !dbg !26
  %v1126 = insertelement <4 x float> %v1122, float %v1125, i64 3, !dbg !27
  %math.result311 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1126) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1127 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !27
  %math.packed312 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1127) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1128 = fadd float %v1125, %v1123, !dbg !28
  %v1129 = extractelement <4 x float> %v1049, i64 0, !dbg !26
  %v1130 = tail call float @llvm.amdgcn.exp2.f32(float %v1129), !dbg !26
  %v1131 = insertelement <4 x float> poison, float %v1130, i64 0, !dbg !27
  %v1132 = fadd float %v1130, %v1128, !dbg !28
  %v1133 = extractelement <4 x float> %v1049, i64 1, !dbg !26
  %v1134 = tail call float @llvm.amdgcn.exp2.f32(float %v1133), !dbg !26
  %v1135 = insertelement <4 x float> %v1131, float %v1134, i64 1, !dbg !27
  %v1136 = fadd float %v1134, %v1132, !dbg !28
  %v1137 = extractelement <4 x float> %v1049, i64 2, !dbg !26
  %v1138 = tail call float @llvm.amdgcn.exp2.f32(float %v1137), !dbg !26
  %v1139 = insertelement <4 x float> %v1135, float %v1138, i64 2, !dbg !27
  %v1140 = fadd float %v1138, %v1136, !dbg !28
  %v1141 = extractelement <4 x float> %v1049, i64 3, !dbg !26
  %v1142 = tail call float @llvm.amdgcn.exp2.f32(float %v1141), !dbg !26
  %v1143 = insertelement <4 x float> %v1139, float %v1142, i64 3, !dbg !27
  %math.result313 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1143) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1144 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !27
  %math.packed314 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1144) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1145 = fadd float %v1142, %v1140, !dbg !28
  %v1146 = extractelement <4 x float> %v1050, i64 0, !dbg !26
  %v1147 = tail call float @llvm.amdgcn.exp2.f32(float %v1146), !dbg !26
  %v1148 = insertelement <4 x float> poison, float %v1147, i64 0, !dbg !27
  %v1149 = fadd float %v1147, %v1145, !dbg !28
  %v1150 = extractelement <4 x float> %v1050, i64 1, !dbg !26
  %v1151 = tail call float @llvm.amdgcn.exp2.f32(float %v1150), !dbg !26
  %v1152 = insertelement <4 x float> %v1148, float %v1151, i64 1, !dbg !27
  %v1153 = fadd float %v1151, %v1149, !dbg !28
  %v1154 = extractelement <4 x float> %v1050, i64 2, !dbg !26
  %v1155 = tail call float @llvm.amdgcn.exp2.f32(float %v1154), !dbg !26
  %v1156 = insertelement <4 x float> %v1152, float %v1155, i64 2, !dbg !27
  %v1157 = fadd float %v1155, %v1153, !dbg !28
  %v1158 = extractelement <4 x float> %v1050, i64 3, !dbg !26
  %v1159 = tail call float @llvm.amdgcn.exp2.f32(float %v1158), !dbg !26
  %v1160 = insertelement <4 x float> %v1156, float %v1159, i64 3, !dbg !27
  %math.result315 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1160) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1161 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !27
  %math.packed316 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1161) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1162 = fadd float %v1159, %v1157, !dbg !28
  %v1163 = extractelement <4 x float> %v1051, i64 0, !dbg !26
  %v1164 = tail call float @llvm.amdgcn.exp2.f32(float %v1163), !dbg !26
  %v1165 = insertelement <4 x float> poison, float %v1164, i64 0, !dbg !27
  %v1166 = fadd float %v1164, %v1162, !dbg !28
  %v1167 = extractelement <4 x float> %v1051, i64 1, !dbg !26
  %v1168 = tail call float @llvm.amdgcn.exp2.f32(float %v1167), !dbg !26
  %v1169 = insertelement <4 x float> %v1165, float %v1168, i64 1, !dbg !27
  %v1170 = fadd float %v1168, %v1166, !dbg !28
  %v1171 = extractelement <4 x float> %v1051, i64 2, !dbg !26
  %v1172 = tail call float @llvm.amdgcn.exp2.f32(float %v1171), !dbg !26
  %v1173 = insertelement <4 x float> %v1169, float %v1172, i64 2, !dbg !27
  %v1174 = fadd float %v1172, %v1170, !dbg !28
  %v1175 = extractelement <4 x float> %v1051, i64 3, !dbg !26
  %v1176 = tail call float @llvm.amdgcn.exp2.f32(float %v1175), !dbg !26
  %v1177 = insertelement <4 x float> %v1173, float %v1176, i64 3, !dbg !27
  %math.result317 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1177) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1178 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !27
  %math.packed318 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1178) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1179 = fadd float %v1176, %v1174, !dbg !28
  %v1180 = extractelement <4 x float> %v1052, i64 0, !dbg !26
  %v1181 = tail call float @llvm.amdgcn.exp2.f32(float %v1180), !dbg !26
  %v1182 = insertelement <4 x float> poison, float %v1181, i64 0, !dbg !27
  %v1183 = fadd float %v1181, %v1179, !dbg !28
  %v1184 = extractelement <4 x float> %v1052, i64 1, !dbg !26
  %v1185 = tail call float @llvm.amdgcn.exp2.f32(float %v1184), !dbg !26
  %v1186 = insertelement <4 x float> %v1182, float %v1185, i64 1, !dbg !27
  %v1187 = fadd float %v1185, %v1183, !dbg !28
  %v1188 = extractelement <4 x float> %v1052, i64 2, !dbg !26
  %v1189 = tail call float @llvm.amdgcn.exp2.f32(float %v1188), !dbg !26
  %v1190 = insertelement <4 x float> %v1186, float %v1189, i64 2, !dbg !27
  %v1191 = fadd float %v1189, %v1187, !dbg !28
  %v1192 = extractelement <4 x float> %v1052, i64 3, !dbg !26
  %v1193 = tail call float @llvm.amdgcn.exp2.f32(float %v1192), !dbg !26
  %v1194 = insertelement <4 x float> %v1190, float %v1193, i64 3, !dbg !27
  %math.result319 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1194) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1195 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !27
  %math.packed320 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1195) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1196 = fadd float %v1193, %v1191, !dbg !28
  %v1197 = extractelement <4 x float> %v1053, i64 0, !dbg !26
  %v1198 = tail call float @llvm.amdgcn.exp2.f32(float %v1197), !dbg !26
  %v1199 = insertelement <4 x float> poison, float %v1198, i64 0, !dbg !27
  %v1200 = fadd float %v1198, %v1196, !dbg !28
  %v1201 = extractelement <4 x float> %v1053, i64 1, !dbg !26
  %v1202 = tail call float @llvm.amdgcn.exp2.f32(float %v1201), !dbg !26
  %v1203 = insertelement <4 x float> %v1199, float %v1202, i64 1, !dbg !27
  %v1204 = fadd float %v1202, %v1200, !dbg !28
  %v1205 = extractelement <4 x float> %v1053, i64 2, !dbg !26
  %v1206 = tail call float @llvm.amdgcn.exp2.f32(float %v1205), !dbg !26
  %v1207 = insertelement <4 x float> %v1203, float %v1206, i64 2, !dbg !27
  %v1208 = fadd float %v1206, %v1204, !dbg !28
  %v1209 = extractelement <4 x float> %v1053, i64 3, !dbg !26
  %v1210 = tail call float @llvm.amdgcn.exp2.f32(float %v1209), !dbg !26
  %v1211 = insertelement <4 x float> %v1207, float %v1210, i64 3, !dbg !27
  %math.result321 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1211) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1212 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !27
  %math.packed322 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1212) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1213 = fadd float %v1210, %v1208, !dbg !28
  %v1214 = extractelement <4 x float> %v1055, i64 0, !dbg !26
  %v1215 = tail call float @llvm.amdgcn.exp2.f32(float %v1214), !dbg !26
  %v1216 = insertelement <4 x float> poison, float %v1215, i64 0, !dbg !27
  %v1217 = extractelement <4 x float> %v1055, i64 1, !dbg !26
  %v1218 = tail call float @llvm.amdgcn.exp2.f32(float %v1217), !dbg !26
  %v1219 = insertelement <4 x float> %v1216, float %v1218, i64 1, !dbg !27
  %v1220 = fadd float %v1215, %v1218, !dbg !28
  %v1221 = extractelement <4 x float> %v1055, i64 2, !dbg !26
  %v1222 = tail call float @llvm.amdgcn.exp2.f32(float %v1221), !dbg !26
  %v1223 = insertelement <4 x float> %v1219, float %v1222, i64 2, !dbg !27
  %v1224 = fadd float %v1222, %v1220, !dbg !28
  %v1225 = extractelement <4 x float> %v1055, i64 3, !dbg !26
  %v1226 = tail call float @llvm.amdgcn.exp2.f32(float %v1225), !dbg !26
  %v1227 = insertelement <4 x float> %v1223, float %v1226, i64 3, !dbg !27
  %math.result323 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1227) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1228 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !27
  %math.packed324 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1228) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1229 = fadd float %v1226, %v1224, !dbg !28
  %v1230 = extractelement <4 x float> %v1056, i64 0, !dbg !26
  %v1231 = tail call float @llvm.amdgcn.exp2.f32(float %v1230), !dbg !26
  %v1232 = insertelement <4 x float> poison, float %v1231, i64 0, !dbg !27
  %v1233 = fadd float %v1231, %v1229, !dbg !28
  %v1234 = extractelement <4 x float> %v1056, i64 1, !dbg !26
  %v1235 = tail call float @llvm.amdgcn.exp2.f32(float %v1234), !dbg !26
  %v1236 = insertelement <4 x float> %v1232, float %v1235, i64 1, !dbg !27
  %v1237 = fadd float %v1235, %v1233, !dbg !28
  %v1238 = extractelement <4 x float> %v1056, i64 2, !dbg !26
  %v1239 = tail call float @llvm.amdgcn.exp2.f32(float %v1238), !dbg !26
  %v1240 = insertelement <4 x float> %v1236, float %v1239, i64 2, !dbg !27
  %v1241 = fadd float %v1239, %v1237, !dbg !28
  %v1242 = extractelement <4 x float> %v1056, i64 3, !dbg !26
  %v1243 = tail call float @llvm.amdgcn.exp2.f32(float %v1242), !dbg !26
  %v1244 = insertelement <4 x float> %v1240, float %v1243, i64 3, !dbg !27
  %math.result325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1244) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1245 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !27
  %math.packed326 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1245) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1246 = fadd float %v1243, %v1241, !dbg !28
  %v1247 = extractelement <4 x float> %v1057, i64 0, !dbg !26
  %v1248 = tail call float @llvm.amdgcn.exp2.f32(float %v1247), !dbg !26
  %v1249 = insertelement <4 x float> poison, float %v1248, i64 0, !dbg !27
  %v1250 = fadd float %v1248, %v1246, !dbg !28
  %v1251 = extractelement <4 x float> %v1057, i64 1, !dbg !26
  %v1252 = tail call float @llvm.amdgcn.exp2.f32(float %v1251), !dbg !26
  %v1253 = insertelement <4 x float> %v1249, float %v1252, i64 1, !dbg !27
  %v1254 = fadd float %v1252, %v1250, !dbg !28
  %v1255 = extractelement <4 x float> %v1057, i64 2, !dbg !26
  %v1256 = tail call float @llvm.amdgcn.exp2.f32(float %v1255), !dbg !26
  %v1257 = insertelement <4 x float> %v1253, float %v1256, i64 2, !dbg !27
  %v1258 = fadd float %v1256, %v1254, !dbg !28
  %v1259 = extractelement <4 x float> %v1057, i64 3, !dbg !26
  %v1260 = tail call float @llvm.amdgcn.exp2.f32(float %v1259), !dbg !26
  %v1261 = insertelement <4 x float> %v1257, float %v1260, i64 3, !dbg !27
  %math.result327 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1261) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1262 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !27
  %math.packed328 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1262) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1263 = fadd float %v1260, %v1258, !dbg !28
  %v1264 = extractelement <4 x float> %v1058, i64 0, !dbg !26
  %v1265 = tail call float @llvm.amdgcn.exp2.f32(float %v1264), !dbg !26
  %v1266 = insertelement <4 x float> poison, float %v1265, i64 0, !dbg !27
  %v1267 = fadd float %v1265, %v1263, !dbg !28
  %v1268 = extractelement <4 x float> %v1058, i64 1, !dbg !26
  %v1269 = tail call float @llvm.amdgcn.exp2.f32(float %v1268), !dbg !26
  %v1270 = insertelement <4 x float> %v1266, float %v1269, i64 1, !dbg !27
  %v1271 = fadd float %v1269, %v1267, !dbg !28
  %v1272 = extractelement <4 x float> %v1058, i64 2, !dbg !26
  %v1273 = tail call float @llvm.amdgcn.exp2.f32(float %v1272), !dbg !26
  %v1274 = insertelement <4 x float> %v1270, float %v1273, i64 2, !dbg !27
  %v1275 = fadd float %v1273, %v1271, !dbg !28
  %v1276 = extractelement <4 x float> %v1058, i64 3, !dbg !26
  %v1277 = tail call float @llvm.amdgcn.exp2.f32(float %v1276), !dbg !26
  %v1278 = insertelement <4 x float> %v1274, float %v1277, i64 3, !dbg !27
  %math.result329 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1278) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1279 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !27
  %math.packed330 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1279) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1280 = fadd float %v1277, %v1275, !dbg !28
  %v1281 = extractelement <4 x float> %v1059, i64 0, !dbg !26
  %v1282 = tail call float @llvm.amdgcn.exp2.f32(float %v1281), !dbg !26
  %v1283 = insertelement <4 x float> poison, float %v1282, i64 0, !dbg !27
  %v1284 = fadd float %v1282, %v1280, !dbg !28
  %v1285 = extractelement <4 x float> %v1059, i64 1, !dbg !26
  %v1286 = tail call float @llvm.amdgcn.exp2.f32(float %v1285), !dbg !26
  %v1287 = insertelement <4 x float> %v1283, float %v1286, i64 1, !dbg !27
  %v1288 = fadd float %v1286, %v1284, !dbg !28
  %v1289 = extractelement <4 x float> %v1059, i64 2, !dbg !26
  %v1290 = tail call float @llvm.amdgcn.exp2.f32(float %v1289), !dbg !26
  %v1291 = insertelement <4 x float> %v1287, float %v1290, i64 2, !dbg !27
  %v1292 = fadd float %v1290, %v1288, !dbg !28
  %v1293 = extractelement <4 x float> %v1059, i64 3, !dbg !26
  %v1294 = tail call float @llvm.amdgcn.exp2.f32(float %v1293), !dbg !26
  %v1295 = insertelement <4 x float> %v1291, float %v1294, i64 3, !dbg !27
  %math.result331 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1295) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1296 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !27
  %math.packed332 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1296) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1297 = fadd float %v1294, %v1292, !dbg !28
  %v1298 = extractelement <4 x float> %v1060, i64 0, !dbg !26
  %v1299 = tail call float @llvm.amdgcn.exp2.f32(float %v1298), !dbg !26
  %v1300 = insertelement <4 x float> poison, float %v1299, i64 0, !dbg !27
  %v1301 = fadd float %v1299, %v1297, !dbg !28
  %v1302 = extractelement <4 x float> %v1060, i64 1, !dbg !26
  %v1303 = tail call float @llvm.amdgcn.exp2.f32(float %v1302), !dbg !26
  %v1304 = insertelement <4 x float> %v1300, float %v1303, i64 1, !dbg !27
  %v1305 = fadd float %v1303, %v1301, !dbg !28
  %v1306 = extractelement <4 x float> %v1060, i64 2, !dbg !26
  %v1307 = tail call float @llvm.amdgcn.exp2.f32(float %v1306), !dbg !26
  %v1308 = insertelement <4 x float> %v1304, float %v1307, i64 2, !dbg !27
  %v1309 = fadd float %v1307, %v1305, !dbg !28
  %v1310 = extractelement <4 x float> %v1060, i64 3, !dbg !26
  %v1311 = tail call float @llvm.amdgcn.exp2.f32(float %v1310), !dbg !26
  %v1312 = insertelement <4 x float> %v1308, float %v1311, i64 3, !dbg !27
  %math.result333 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1312) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1313 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !27
  %math.packed334 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1313) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1314 = fadd float %v1311, %v1309, !dbg !28
  %v1315 = extractelement <4 x float> %v1061, i64 0, !dbg !26
  %v1316 = tail call float @llvm.amdgcn.exp2.f32(float %v1315), !dbg !26
  %v1317 = insertelement <4 x float> poison, float %v1316, i64 0, !dbg !27
  %v1318 = fadd float %v1316, %v1314, !dbg !28
  %v1319 = extractelement <4 x float> %v1061, i64 1, !dbg !26
  %v1320 = tail call float @llvm.amdgcn.exp2.f32(float %v1319), !dbg !26
  %v1321 = insertelement <4 x float> %v1317, float %v1320, i64 1, !dbg !27
  %v1322 = fadd float %v1320, %v1318, !dbg !28
  %v1323 = extractelement <4 x float> %v1061, i64 2, !dbg !26
  %v1324 = tail call float @llvm.amdgcn.exp2.f32(float %v1323), !dbg !26
  %v1325 = insertelement <4 x float> %v1321, float %v1324, i64 2, !dbg !27
  %v1326 = fadd float %v1324, %v1322, !dbg !28
  %v1327 = extractelement <4 x float> %v1061, i64 3, !dbg !26
  %v1328 = tail call float @llvm.amdgcn.exp2.f32(float %v1327), !dbg !26
  %v1329 = insertelement <4 x float> %v1325, float %v1328, i64 3, !dbg !27
  %math.result335 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1329) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1330 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !27
  %math.packed336 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1330) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1331 = fadd float %v1328, %v1326, !dbg !28
  %v1332 = extractelement <4 x float> %v1062, i64 0, !dbg !26
  %v1333 = tail call float @llvm.amdgcn.exp2.f32(float %v1332), !dbg !26
  %v1334 = insertelement <4 x float> poison, float %v1333, i64 0, !dbg !27
  %v1335 = fadd float %v1333, %v1331, !dbg !28
  %v1336 = extractelement <4 x float> %v1062, i64 1, !dbg !26
  %v1337 = tail call float @llvm.amdgcn.exp2.f32(float %v1336), !dbg !26
  %v1338 = insertelement <4 x float> %v1334, float %v1337, i64 1, !dbg !27
  %v1339 = fadd float %v1337, %v1335, !dbg !28
  %v1340 = extractelement <4 x float> %v1062, i64 2, !dbg !26
  %v1341 = tail call float @llvm.amdgcn.exp2.f32(float %v1340), !dbg !26
  %v1342 = insertelement <4 x float> %v1338, float %v1341, i64 2, !dbg !27
  %v1343 = fadd float %v1341, %v1339, !dbg !28
  %v1344 = extractelement <4 x float> %v1062, i64 3, !dbg !26
  %v1345 = tail call float @llvm.amdgcn.exp2.f32(float %v1344), !dbg !26
  %v1346 = insertelement <4 x float> %v1342, float %v1345, i64 3, !dbg !27
  %math.result337 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1346) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1347 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !27
  %math.packed338 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1347) #8, !dbg !28, !frisk.register_lifetimes !5
  %v1348 = fadd float %v1345, %v1343, !dbg !28
  %v1349 = bitcast float %v1213 to i32, !dbg !28
  %v1350 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v1349), !dbg !28
  %v1351 = bitcast i32 %v1350 to float, !dbg !28
  %v1352 = fadd float %v1213, %v1351, !dbg !28
  %v1353 = bitcast float %v1352 to i32, !dbg !28
  %v1354 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v1353), !dbg !28
  %v1355 = bitcast i32 %v1354 to float, !dbg !28
  %v1356 = fadd float %v1352, %v1355, !dbg !28
  %v1357 = fadd float %v1070, %v1356, !dbg !28
  %v1358 = insertelement <1 x float> poison, float %v1357, i64 0, !dbg !11
  %v1359 = insertvalue [2 x <1 x float>] poison, <1 x float> %v1358, 0, !dbg !11
  %v1360 = bitcast float %v1348 to i32, !dbg !28
  %v1361 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v1360), !dbg !28
  %v1362 = bitcast i32 %v1361 to float, !dbg !28
  %v1363 = fadd float %v1348, %v1362, !dbg !28
  %v1364 = bitcast float %v1363 to i32, !dbg !28
  %v1365 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v1364), !dbg !28
  %v1366 = bitcast i32 %v1365 to float, !dbg !28
  %v1367 = fadd float %v1363, %v1366, !dbg !28
  %v1368 = fadd float %v1078, %v1367, !dbg !28
  %v1369 = insertelement <1 x float> poison, float %v1368, i64 0, !dbg !11
  %v1370 = insertvalue [2 x <1 x float>] %v1359, <1 x float> %v1369, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread864 = tail call i32 asm sideeffect "", "=v,0"(i32 %v31) #8, !dbg !29, !range !9, !frisk.register_lifetimes !5
  %address.local865 = shl nuw nsw i32 %address.thread864, 3, !dbg !29
  %address.local866 = and i32 %address.local865, 120, !dbg !29
  %address.local867 = shl nuw nsw i32 %address.thread864, 5, !dbg !29
  %address.local868 = and i32 %address.local867, 1536, !dbg !29
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !29
  %address.local870 = zext nneg i32 %address.local869 to i64, !dbg !29
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !29
  %v1371 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #10, !dbg !29
  %v1372 = bitcast <4 x i32> %v1371 to <8 x half>, !dbg !29
  %address.local872 = zext nneg i32 %address.thread864 to i64, !dbg !8
  %address.local873 = and i64 %address.local872, 60, !dbg !29
  %address.local874 = add nuw nsw i64 %address.local873, %address.local872, !dbg !29
  %address.local876.idx = shl nuw nsw i64 %address.local874, 4, !dbg !29
  %address.local876 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local876.idx, !dbg !29
  %address.local877 = lshr i64 %address.local872, 4, !dbg !29
  %address.local878 = and i64 %address.local877, 3, !dbg !29
  %address.local880.idx = mul nuw nsw i64 %address.local878, 768, !dbg !29
  %address.local880 = getelementptr i8, ptr addrspace(3) %address.local876, i64 %address.local880.idx, !dbg !29
  %address.local881 = lshr i64 %address.local872, 2, !dbg !15
  %address.local883.idx = mul nsw i64 %address.local881, -64, !dbg !29
  %address.local883 = getelementptr i8, ptr addrspace(3) %address.local880, i64 %address.local883.idx, !dbg !29
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 256, !dbg !29
  %v1373 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #10, !dbg !29
  %v1374 = bitcast <4 x i32> %v1373 to <8 x half>, !dbg !29
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local883, i64 512, !dbg !29
  %v1375 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #10, !dbg !29
  %v1376 = bitcast <4 x i32> %v1375 to <8 x half>, !dbg !29
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local883, i64 768, !dbg !29
  %v1377 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #10, !dbg !29
  %v1378 = bitcast <4 x i32> %v1377 to <8 x half>, !dbg !29
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local871, i64 4096, !dbg !29
  %v1379 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #10, !dbg !29
  %v1380 = bitcast <4 x i32> %v1379 to <8 x half>, !dbg !29
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4352, !dbg !29
  %v1381 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #10, !dbg !29
  %v1382 = bitcast <4 x i32> %v1381 to <8 x half>, !dbg !29
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4608, !dbg !29
  %v1383 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #10, !dbg !29
  %v1384 = bitcast <4 x i32> %v1383 to <8 x half>, !dbg !29
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4864, !dbg !29
  %v1385 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #10, !dbg !29
  %v1386 = bitcast <4 x i32> %v1385 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !8
  %address.local908 = and i64 %address.local907, 1536, !dbg !8
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !8
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !8
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !8
  %v1393 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !17
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !17
  %address.local913 = and i64 %address.local912, 8064, !dbg !17
  %address.local915 = and i64 %address.local907, 24, !dbg !17
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !17
  %225 = or disjoint i64 %v604.idx28, %address.local916, !dbg !17
  %226 = trunc i64 %225 to i32, !dbg !17
  %.tr33 = shl i32 %226, 1, !dbg !17
  %dma.buffer.offset1558 = or i32 %.tr33, 64, !dbg !17
  %dma.buffer.lds1560 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1393), !dbg !17
  %v1406 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !8
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !8
  %address.local920 = and i64 %address.local919, 3584, !dbg !8
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !8
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !8
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !8
  %v1407 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !17
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !17
  %address.local925 = and i64 %address.local924, 16256, !dbg !17
  %address.local928 = or disjoint i64 %address.local925, %address.local915, !dbg !17
  %227 = or disjoint i64 %v604.idx28, %address.local928, !dbg !17
  %228 = trunc i64 %227 to i32, !dbg !17
  %.tr36 = shl i32 %228, 1, !dbg !17
  %dma.buffer.offset1555 = or i32 %.tr36, 64, !dbg !17
  %dma.buffer.lds1557 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1407), !dbg !17
  %v1420 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %v1421 = shufflevector <8 x half> %v1372, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1422 = extractvalue [2 x <8 x float>] %v20653, 0, !dbg !24
  %v1423 = shufflevector <8 x float> %v1422, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1424 = fmul <4 x float> %v1423, %v1067, !dbg !24
  %v1425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1421, <4 x float> %v1424) #8, !dbg !29
  %v1426 = shufflevector <8 x half> %v1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1427 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1426, <4 x float> %v1425) #8, !dbg !29
  %v1428 = shufflevector <8 x half> %v1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1428, <4 x float> %v1427) #8, !dbg !29
  %v1430 = shufflevector <8 x half> %v1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1431 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1430, <4 x float> %v1429) #8, !dbg !29
  %v1432 = shufflevector <8 x half> %v1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1432, <4 x float> %v1431) #8, !dbg !29
  %v1434 = shufflevector <8 x half> %v1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1434, <4 x float> %v1433) #8, !dbg !29
  %v1436 = shufflevector <8 x half> %v1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1436, <4 x float> %v1435) #8, !dbg !29
  %v1438 = shufflevector <8 x half> %v1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1438, <4 x float> %v1437) #8, !dbg !29
  %v1440 = shufflevector <8 x half> %v1372, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1441 = shufflevector <8 x float> %v1422, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1442 = fmul <4 x float> %v1441, %v1067, !dbg !24
  %v1443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1440, <4 x float> %v1442) #8, !dbg !29
  %v1444 = shufflevector <8 x half> %v1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1444, <4 x float> %v1443) #8, !dbg !29
  %v1446 = shufflevector <8 x half> %v1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1446, <4 x float> %v1445) #8, !dbg !29
  %v1448 = shufflevector <8 x half> %v1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1448, <4 x float> %v1447) #8, !dbg !29
  %v1450 = shufflevector <8 x half> %v1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1450, <4 x float> %v1449) #8, !dbg !29
  %v1452 = shufflevector <8 x half> %v1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1452, <4 x float> %v1451) #8, !dbg !29
  %v1454 = shufflevector <8 x half> %v1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1455 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1454, <4 x float> %v1453) #8, !dbg !29
  %v1456 = shufflevector <8 x half> %v1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1456, <4 x float> %v1455) #8, !dbg !29
  %v1458 = extractvalue [2 x <8 x float>] %v20653, 1, !dbg !24
  %v1459 = shufflevector <8 x float> %v1458, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1460 = fmul <4 x float> %v1459, %v1075, !dbg !24
  %v1461 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1421, <4 x float> %v1460) #8, !dbg !29
  %v1462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1426, <4 x float> %v1461) #8, !dbg !29
  %v1463 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1428, <4 x float> %v1462) #8, !dbg !29
  %v1464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1430, <4 x float> %v1463) #8, !dbg !29
  %v1465 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1432, <4 x float> %v1464) #8, !dbg !29
  %v1466 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1434, <4 x float> %v1465) #8, !dbg !29
  %v1467 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1436, <4 x float> %v1466) #8, !dbg !29
  %v1468 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1438, <4 x float> %v1467) #8, !dbg !29
  %v1469 = shufflevector <8 x float> %v1458, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1470 = fmul <4 x float> %v1469, %v1075, !dbg !24
  %v1471 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1440, <4 x float> %v1470) #8, !dbg !29
  %v1472 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1444, <4 x float> %v1471) #8, !dbg !29
  %v1473 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1446, <4 x float> %v1472) #8, !dbg !29
  %v1474 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1448, <4 x float> %v1473) #8, !dbg !29
  %v1475 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1450, <4 x float> %v1474) #8, !dbg !29
  %v1476 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1452, <4 x float> %v1475) #8, !dbg !29
  %v1477 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1454, <4 x float> %v1476) #8, !dbg !29
  %v1478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1456, <4 x float> %v1477) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %v1479 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #10, !dbg !29
  %v1480 = bitcast <4 x i32> %v1479 to <8 x half>, !dbg !29
  %v1481 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #10, !dbg !29
  %v1482 = bitcast <4 x i32> %v1481 to <8 x half>, !dbg !29
  %v1483 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #10, !dbg !29
  %v1484 = bitcast <4 x i32> %v1483 to <8 x half>, !dbg !29
  %v1485 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #10, !dbg !29
  %v1486 = bitcast <4 x i32> %v1485 to <8 x half>, !dbg !29
  %v1487 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #10, !dbg !29
  %v1488 = bitcast <4 x i32> %v1487 to <8 x half>, !dbg !29
  %v1489 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #10, !dbg !29
  %v1490 = bitcast <4 x i32> %v1489 to <8 x half>, !dbg !29
  %v1491 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #10, !dbg !29
  %v1492 = bitcast <4 x i32> %v1491 to <8 x half>, !dbg !29
  %v1493 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #10, !dbg !29
  %v1494 = bitcast <4 x i32> %v1493 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %dma.buffer.offset1552 = or i32 %.tr33, 128, !dbg !17
  %v1510 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %dma.buffer.offset1549 = or i32 %.tr36, 128, !dbg !17
  %v1524 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %v1525 = shufflevector <8 x half> %v1480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1526 = extractvalue [2 x <8 x float>] %v20956, 0, !dbg !24
  %v1527 = shufflevector <8 x float> %v1526, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1528 = fmul <4 x float> %v1527, %v1067, !dbg !24
  %v1529 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1525, <4 x float> %v1528) #8, !dbg !29
  %v1530 = shufflevector <8 x half> %v1482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1530, <4 x float> %v1529) #8, !dbg !29
  %v1532 = shufflevector <8 x half> %v1484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1533 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1532, <4 x float> %v1531) #8, !dbg !29
  %v1534 = shufflevector <8 x half> %v1486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1535 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1534, <4 x float> %v1533) #8, !dbg !29
  %v1536 = shufflevector <8 x half> %v1488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1537 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1536, <4 x float> %v1535) #8, !dbg !29
  %v1538 = shufflevector <8 x half> %v1490, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1539 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1538, <4 x float> %v1537) #8, !dbg !29
  %v1540 = shufflevector <8 x half> %v1492, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1540, <4 x float> %v1539) #8, !dbg !29
  %v1542 = shufflevector <8 x half> %v1494, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1543 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1542, <4 x float> %v1541) #8, !dbg !29
  %v1544 = shufflevector <8 x half> %v1480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1545 = shufflevector <8 x float> %v1526, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1546 = fmul <4 x float> %v1545, %v1067, !dbg !24
  %v1547 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1544, <4 x float> %v1546) #8, !dbg !29
  %v1548 = shufflevector <8 x half> %v1482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1549 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1548, <4 x float> %v1547) #8, !dbg !29
  %v1550 = shufflevector <8 x half> %v1484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1551 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1550, <4 x float> %v1549) #8, !dbg !29
  %v1552 = shufflevector <8 x half> %v1486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1552, <4 x float> %v1551) #8, !dbg !29
  %v1554 = shufflevector <8 x half> %v1488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1554, <4 x float> %v1553) #8, !dbg !29
  %v1556 = shufflevector <8 x half> %v1490, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1556, <4 x float> %v1555) #8, !dbg !29
  %v1558 = shufflevector <8 x half> %v1492, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1559 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1558, <4 x float> %v1557) #8, !dbg !29
  %v1560 = shufflevector <8 x half> %v1494, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1561 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1560, <4 x float> %v1559) #8, !dbg !29
  %v1562 = extractvalue [2 x <8 x float>] %v20956, 1, !dbg !24
  %v1563 = shufflevector <8 x float> %v1562, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1564 = fmul <4 x float> %v1563, %v1075, !dbg !24
  %v1565 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1525, <4 x float> %v1564) #8, !dbg !29
  %v1566 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1530, <4 x float> %v1565) #8, !dbg !29
  %v1567 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1532, <4 x float> %v1566) #8, !dbg !29
  %v1568 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1534, <4 x float> %v1567) #8, !dbg !29
  %v1569 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1536, <4 x float> %v1568) #8, !dbg !29
  %v1570 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1538, <4 x float> %v1569) #8, !dbg !29
  %v1571 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1540, <4 x float> %v1570) #8, !dbg !29
  %v1572 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1542, <4 x float> %v1571) #8, !dbg !29
  %v1573 = shufflevector <8 x float> %v1562, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1574 = fmul <4 x float> %v1573, %v1075, !dbg !24
  %v1575 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1544, <4 x float> %v1574) #8, !dbg !29
  %v1576 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1548, <4 x float> %v1575) #8, !dbg !29
  %v1577 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1550, <4 x float> %v1576) #8, !dbg !29
  %v1578 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1552, <4 x float> %v1577) #8, !dbg !29
  %v1579 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1554, <4 x float> %v1578) #8, !dbg !29
  %v1580 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1556, <4 x float> %v1579) #8, !dbg !29
  %v1581 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1558, <4 x float> %v1580) #8, !dbg !29
  %v1582 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1560, <4 x float> %v1581) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %v1583 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #10, !dbg !29
  %v1584 = bitcast <4 x i32> %v1583 to <8 x half>, !dbg !29
  %v1585 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #10, !dbg !29
  %v1586 = bitcast <4 x i32> %v1585 to <8 x half>, !dbg !29
  %v1587 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #10, !dbg !29
  %v1588 = bitcast <4 x i32> %v1587 to <8 x half>, !dbg !29
  %v1589 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #10, !dbg !29
  %v1590 = bitcast <4 x i32> %v1589 to <8 x half>, !dbg !29
  %v1591 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #10, !dbg !29
  %v1592 = bitcast <4 x i32> %v1591 to <8 x half>, !dbg !29
  %v1593 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #10, !dbg !29
  %v1594 = bitcast <4 x i32> %v1593 to <8 x half>, !dbg !29
  %v1595 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #10, !dbg !29
  %v1596 = bitcast <4 x i32> %v1595 to <8 x half>, !dbg !29
  %v1597 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #10, !dbg !29
  %v1598 = bitcast <4 x i32> %v1597 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %229 = or disjoint i64 %v604.idx28, 96, !dbg !17
  %230 = or disjoint i64 %address.local916, %229, !dbg !17
  %.tr45 = trunc i64 %230 to i32, !dbg !17
  %dma.buffer.offset1546 = shl i32 %.tr45, 1, !dbg !17
  %v1614 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %231 = or disjoint i64 %address.local928, %229, !dbg !17
  %.tr48 = trunc i64 %231 to i32, !dbg !17
  %dma.buffer.offset = shl i32 %.tr48, 1, !dbg !17
  %v1628 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %v1629 = shufflevector <8 x half> %v1584, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1630 = extractvalue [2 x <8 x float>] %v21057, 0, !dbg !24
  %v1631 = shufflevector <8 x float> %v1630, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1632 = fmul <4 x float> %v1631, %v1067, !dbg !24
  %v1633 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1629, <4 x float> %v1632) #8, !dbg !29
  %v1634 = shufflevector <8 x half> %v1586, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1635 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1634, <4 x float> %v1633) #8, !dbg !29
  %v1636 = shufflevector <8 x half> %v1588, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1636, <4 x float> %v1635) #8, !dbg !29
  %v1638 = shufflevector <8 x half> %v1590, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1639 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1638, <4 x float> %v1637) #8, !dbg !29
  %v1640 = shufflevector <8 x half> %v1592, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1641 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1640, <4 x float> %v1639) #8, !dbg !29
  %v1642 = shufflevector <8 x half> %v1594, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1643 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1642, <4 x float> %v1641) #8, !dbg !29
  %v1644 = shufflevector <8 x half> %v1596, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1645 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1644, <4 x float> %v1643) #8, !dbg !29
  %v1646 = shufflevector <8 x half> %v1598, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1647 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1646, <4 x float> %v1645) #8, !dbg !29
  %v1648 = shufflevector <8 x half> %v1584, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1649 = shufflevector <8 x float> %v1630, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1650 = fmul <4 x float> %v1649, %v1067, !dbg !24
  %v1651 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1648, <4 x float> %v1650) #8, !dbg !29
  %v1652 = shufflevector <8 x half> %v1586, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1653 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1652, <4 x float> %v1651) #8, !dbg !29
  %v1654 = shufflevector <8 x half> %v1588, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1655 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1654, <4 x float> %v1653) #8, !dbg !29
  %v1656 = shufflevector <8 x half> %v1590, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1657 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1656, <4 x float> %v1655) #8, !dbg !29
  %v1658 = shufflevector <8 x half> %v1592, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1659 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1658, <4 x float> %v1657) #8, !dbg !29
  %v1660 = shufflevector <8 x half> %v1594, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1661 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1660, <4 x float> %v1659) #8, !dbg !29
  %v1662 = shufflevector <8 x half> %v1596, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1663 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1662, <4 x float> %v1661) #8, !dbg !29
  %v1664 = shufflevector <8 x half> %v1598, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1665 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1664, <4 x float> %v1663) #8, !dbg !29
  %v1666 = extractvalue [2 x <8 x float>] %v21057, 1, !dbg !24
  %v1667 = shufflevector <8 x float> %v1666, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1668 = fmul <4 x float> %v1667, %v1075, !dbg !24
  %v1669 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1629, <4 x float> %v1668) #8, !dbg !29
  %v1670 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1634, <4 x float> %v1669) #8, !dbg !29
  %v1671 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1636, <4 x float> %v1670) #8, !dbg !29
  %v1672 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1638, <4 x float> %v1671) #8, !dbg !29
  %v1673 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1640, <4 x float> %v1672) #8, !dbg !29
  %v1674 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1642, <4 x float> %v1673) #8, !dbg !29
  %v1675 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1644, <4 x float> %v1674) #8, !dbg !29
  %v1676 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1646, <4 x float> %v1675) #8, !dbg !29
  %v1677 = shufflevector <8 x float> %v1666, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1678 = fmul <4 x float> %v1677, %v1075, !dbg !24
  %v1679 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1648, <4 x float> %v1678) #8, !dbg !29
  %v1680 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1652, <4 x float> %v1679) #8, !dbg !29
  %v1681 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1654, <4 x float> %v1680) #8, !dbg !29
  %v1682 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1656, <4 x float> %v1681) #8, !dbg !29
  %v1683 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1658, <4 x float> %v1682) #8, !dbg !29
  %v1684 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1660, <4 x float> %v1683) #8, !dbg !29
  %v1685 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1662, <4 x float> %v1684) #8, !dbg !29
  %v1686 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1664, <4 x float> %v1685) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %v1687 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #10, !dbg !29
  %v1688 = bitcast <4 x i32> %v1687 to <8 x half>, !dbg !29
  %v1689 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #10, !dbg !29
  %v1690 = bitcast <4 x i32> %v1689 to <8 x half>, !dbg !29
  %v1691 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #10, !dbg !29
  %v1692 = bitcast <4 x i32> %v1691 to <8 x half>, !dbg !29
  %v1693 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #10, !dbg !29
  %v1694 = bitcast <4 x i32> %v1693 to <8 x half>, !dbg !29
  %v1695 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #10, !dbg !29
  %v1696 = bitcast <4 x i32> %v1695 to <8 x half>, !dbg !29
  %v1697 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #10, !dbg !29
  %v1698 = bitcast <4 x i32> %v1697 to <8 x half>, !dbg !29
  %v1699 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #10, !dbg !29
  %v1700 = bitcast <4 x i32> %v1699 to <8 x half>, !dbg !29
  %v1701 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #10, !dbg !29
  %v1702 = bitcast <4 x i32> %v1701 to <8 x half>, !dbg !29
  %v1703 = shufflevector <8 x half> %v1688, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1704 = extractvalue [2 x <8 x float>] %v21158, 0, !dbg !24
  %v1705 = shufflevector <8 x float> %v1704, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1706 = fmul <4 x float> %v1705, %v1067, !dbg !24
  %v1707 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1703, <4 x float> %v1706) #8, !dbg !29
  %v1708 = shufflevector <8 x half> %v1690, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1709 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1708, <4 x float> %v1707) #8, !dbg !29
  %v1710 = shufflevector <8 x half> %v1692, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1711 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1710, <4 x float> %v1709) #8, !dbg !29
  %v1712 = shufflevector <8 x half> %v1694, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1713 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1712, <4 x float> %v1711) #8, !dbg !29
  %v1714 = shufflevector <8 x half> %v1696, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1715 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1714, <4 x float> %v1713) #8, !dbg !29
  %v1716 = shufflevector <8 x half> %v1698, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1717 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1716, <4 x float> %v1715) #8, !dbg !29
  %v1718 = shufflevector <8 x half> %v1700, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1719 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1718, <4 x float> %v1717) #8, !dbg !29
  %v1720 = shufflevector <8 x half> %v1702, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1721 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1720, <4 x float> %v1719) #8, !dbg !29
  %v1722 = shufflevector <8 x half> %v1688, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1723 = shufflevector <8 x float> %v1704, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1724 = fmul <4 x float> %v1723, %v1067, !dbg !24
  %v1725 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1722, <4 x float> %v1724) #8, !dbg !29
  %v1726 = shufflevector <8 x half> %v1690, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1727 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1726, <4 x float> %v1725) #8, !dbg !29
  %v1728 = shufflevector <8 x half> %v1692, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1729 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1728, <4 x float> %v1727) #8, !dbg !29
  %v1730 = shufflevector <8 x half> %v1694, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1731 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1730, <4 x float> %v1729) #8, !dbg !29
  %v1732 = shufflevector <8 x half> %v1696, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1733 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1732, <4 x float> %v1731) #8, !dbg !29
  %v1734 = shufflevector <8 x half> %v1698, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1735 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1734, <4 x float> %v1733) #8, !dbg !29
  %v1736 = shufflevector <8 x half> %v1700, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1737 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1736, <4 x float> %v1735) #8, !dbg !29
  %v1738 = shufflevector <8 x half> %v1702, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1739 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1738, <4 x float> %v1737) #8, !dbg !29
  %v1740 = extractvalue [2 x <8 x float>] %v21158, 1, !dbg !24
  %v1741 = shufflevector <8 x float> %v1740, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1742 = fmul <4 x float> %v1741, %v1075, !dbg !24
  %v1743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1703, <4 x float> %v1742) #8, !dbg !29
  %v1744 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1708, <4 x float> %v1743) #8, !dbg !29
  %v1745 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1710, <4 x float> %v1744) #8, !dbg !29
  %v1746 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1712, <4 x float> %v1745) #8, !dbg !29
  %v1747 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1714, <4 x float> %v1746) #8, !dbg !29
  %v1748 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1716, <4 x float> %v1747) #8, !dbg !29
  %v1749 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1718, <4 x float> %v1748) #8, !dbg !29
  %v1750 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1720, <4 x float> %v1749) #8, !dbg !29
  %v1751 = shufflevector <8 x float> %v1740, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1752 = fmul <4 x float> %v1751, %v1075, !dbg !24
  %v1753 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1722, <4 x float> %v1752) #8, !dbg !29
  %v1754 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1726, <4 x float> %v1753) #8, !dbg !29
  %v1755 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1728, <4 x float> %v1754) #8, !dbg !29
  %v1756 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1730, <4 x float> %v1755) #8, !dbg !29
  %v1757 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1732, <4 x float> %v1756) #8, !dbg !29
  %v1758 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1734, <4 x float> %v1757) #8, !dbg !29
  %v1759 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1736, <4 x float> %v1758) #8, !dbg !29
  %v1760 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1738, <4 x float> %v1759) #8, !dbg !29
  %v1761 = shufflevector <4 x float> %v1439, <4 x float> %v1457, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1762 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1761, 0, !dbg !11
  %v1763 = shufflevector <4 x float> %v1468, <4 x float> %v1478, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1764 = insertvalue [2 x <8 x float>] %v1762, <8 x float> %v1763, 1, !dbg !11
  %v1765 = insertvalue [2 x <1 x float>] poison, <1 x float> %v1001, 0, !dbg !11
  %v1766 = insertvalue [2 x <1 x float>] %v1765, <1 x float> %v1044, 1, !dbg !11
  %v1767 = shufflevector <4 x float> %v1543, <4 x float> %v1561, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1768 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1767, 0, !dbg !11
  %v1769 = shufflevector <4 x float> %v1572, <4 x float> %v1582, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1770 = insertvalue [2 x <8 x float>] %v1768, <8 x float> %v1769, 1, !dbg !11
  %v1771 = shufflevector <4 x float> %v1647, <4 x float> %v1665, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1772 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1771, 0, !dbg !11
  %v1773 = shufflevector <4 x float> %v1676, <4 x float> %v1686, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1774 = insertvalue [2 x <8 x float>] %v1772, <8 x float> %v1773, 1, !dbg !11
  %v1775 = shufflevector <4 x float> %v1721, <4 x float> %v1739, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1776 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1775, 0, !dbg !11
  %v1777 = shufflevector <4 x float> %v1750, <4 x float> %v1760, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1778 = insertvalue [2 x <8 x float>] %v1776, <8 x float> %v1777, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %v1779 = add nuw nsw i64 %v20552, 1, !dbg !11
  %v212 = icmp slt i64 %v1779, %v191, !dbg !11
  br i1 %v212, label %v213, label %v1780, !dbg !11

v1780:                                            ; preds = %v848, %.v1780_crit_edge
  %v1809.pre-phi = phi i32 [ %.pre70, %.v1780_crit_edge ], [ %v991, %v848 ], !dbg !11
  %v1793.pre-phi = phi i32 [ %.pre69, %.v1780_crit_edge ], [ %v983, %v848 ], !dbg !11
  %v1792.pre-phi = phi i32 [ %.pre68, %.v1780_crit_edge ], [ %v982, %v848 ], !dbg !11
  %v1790.pre-phi = phi i32 [ %.pre66, %.v1780_crit_edge ], [ %v980, %v848 ], !dbg !11
  %v206.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.v1780_crit_edge ], [ %v1764, %v848 ], !dbg !11
  %v207.lcssa = phi [2 x <1 x float>] [ zeroinitializer, %.v1780_crit_edge ], [ %v1370, %v848 ], !dbg !11
  %v209.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.v1780_crit_edge ], [ %v1770, %v848 ], !dbg !11
  %v210.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.v1780_crit_edge ], [ %v1774, %v848 ], !dbg !11
  %v211.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.v1780_crit_edge ], [ %v1778, %v848 ], !dbg !11
  %v1781 = and i32 %v31, 16, !dbg !11
  %.not = icmp eq i32 %v1781, 0, !dbg !11
  %v1782 = and i32 %v31, 32, !dbg !11
  %.not36 = icmp eq i32 %v1782, 0, !dbg !11
  %v1783 = extractvalue [2 x <8 x float>] %v211.lcssa, 0, !dbg !11
  %v1784 = extractelement <8 x float> %v1783, i64 0, !dbg !11
  %v1785 = extractelement <8 x float> %v1783, i64 1, !dbg !11
  %v1786 = extractelement <8 x float> %v1783, i64 2, !dbg !11
  %v1787 = extractelement <8 x float> %v1783, i64 3, !dbg !11
  %v1788 = select i1 %.not, float %v1785, float %v1784, !dbg !11
  %v1794 = icmp samesign ult i32 %v1792.pre-phi, %v1793.pre-phi, !dbg !11
  %v1795 = select i1 %v1794, i32 %v1792.pre-phi, i32 %v1790.pre-phi, !dbg !11
  %v1796 = shl nuw nsw i32 %v1795, 2, !dbg !11
  %v1797 = bitcast float %v1788 to i32, !dbg !11
  %v1798 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1797), !dbg !11
  %v1799 = bitcast i32 %v1798 to float, !dbg !11
  %v1800 = select i1 %.not, float %v1784, float %v1799, !dbg !11
  %v1801 = select i1 %.not, float %v1799, float %v1785, !dbg !11
  %v1802 = select i1 %.not, float %v1787, float %v1786, !dbg !11
  %v1803 = bitcast float %v1802 to i32, !dbg !11
  %v1804 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1803), !dbg !11
  %v1805 = bitcast i32 %v1804 to float, !dbg !11
  %v1806 = select i1 %.not, float %v1786, float %v1805, !dbg !11
  %v1807 = select i1 %.not, float %v1805, float %v1787, !dbg !11
  %v1808 = select i1 %.not36, float %v1806, float %v1800, !dbg !11
  %v1810 = icmp samesign ult i32 %v1809.pre-phi, %v1793.pre-phi, !dbg !11
  %v1811 = select i1 %v1810, i32 %v1809.pre-phi, i32 %v1790.pre-phi, !dbg !11
  %v1812 = shl nuw nsw i32 %v1811, 2, !dbg !11
  %v1813 = bitcast float %v1808 to i32, !dbg !11
  %v1814 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1813), !dbg !11
  %v1815 = bitcast i32 %v1814 to float, !dbg !11
  %v1818 = select i1 %.not36, float %v1807, float %v1801, !dbg !11
  %v1819 = bitcast float %v1818 to i32, !dbg !11
  %v1820 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1819), !dbg !11
  %v1821 = bitcast i32 %v1820 to float, !dbg !11
  %232 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %233 = shufflevector <4 x i1> %232, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %234 = insertelement <4 x float> poison, float %v1800, i64 0, !dbg !11
  %235 = insertelement <4 x float> %234, float %v1801, i64 1, !dbg !11
  %236 = insertelement <4 x float> %235, float %v1815, i64 2, !dbg !11
  %237 = insertelement <4 x float> %236, float %v1821, i64 3, !dbg !11
  %238 = insertelement <4 x float> poison, float %v1815, i64 0, !dbg !11
  %239 = insertelement <4 x float> %238, float %v1821, i64 1, !dbg !11
  %240 = insertelement <4 x float> %239, float %v1806, i64 2, !dbg !11
  %241 = insertelement <4 x float> %240, float %v1807, i64 3, !dbg !11
  %242 = select <4 x i1> %233, <4 x float> %237, <4 x float> %241, !dbg !11
  %v1824 = extractvalue [2 x <8 x float>] %v210.lcssa, 0, !dbg !11
  %v1825 = extractelement <8 x float> %v1824, i64 0, !dbg !11
  %v1826 = extractelement <8 x float> %v1824, i64 1, !dbg !11
  %v1827 = extractelement <8 x float> %v1824, i64 2, !dbg !11
  %v1828 = extractelement <8 x float> %v1824, i64 3, !dbg !11
  %v1829 = select i1 %.not, float %v1826, float %v1825, !dbg !11
  %v1830 = bitcast float %v1829 to i32, !dbg !11
  %v1831 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1830), !dbg !11
  %v1832 = bitcast i32 %v1831 to float, !dbg !11
  %v1833 = select i1 %.not, float %v1825, float %v1832, !dbg !11
  %v1834 = select i1 %.not, float %v1832, float %v1826, !dbg !11
  %v1835 = select i1 %.not, float %v1828, float %v1827, !dbg !11
  %v1836 = bitcast float %v1835 to i32, !dbg !11
  %v1837 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1836), !dbg !11
  %v1838 = bitcast i32 %v1837 to float, !dbg !11
  %v1839 = select i1 %.not, float %v1827, float %v1838, !dbg !11
  %v1840 = select i1 %.not, float %v1838, float %v1828, !dbg !11
  %v1841 = select i1 %.not36, float %v1839, float %v1833, !dbg !11
  %v1842 = bitcast float %v1841 to i32, !dbg !11
  %v1843 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1842), !dbg !11
  %v1844 = bitcast i32 %v1843 to float, !dbg !11
  %v1847 = select i1 %.not36, float %v1840, float %v1834, !dbg !11
  %v1848 = bitcast float %v1847 to i32, !dbg !11
  %v1849 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1848), !dbg !11
  %v1850 = bitcast i32 %v1849 to float, !dbg !11
  %243 = insertelement <4 x float> poison, float %v1833, i64 0, !dbg !11
  %244 = insertelement <4 x float> %243, float %v1834, i64 1, !dbg !11
  %245 = insertelement <4 x float> %244, float %v1844, i64 2, !dbg !11
  %246 = insertelement <4 x float> %245, float %v1850, i64 3, !dbg !11
  %247 = insertelement <4 x float> poison, float %v1844, i64 0, !dbg !11
  %248 = insertelement <4 x float> %247, float %v1850, i64 1, !dbg !11
  %249 = insertelement <4 x float> %248, float %v1839, i64 2, !dbg !11
  %250 = insertelement <4 x float> %249, float %v1840, i64 3, !dbg !11
  %251 = select <4 x i1> %233, <4 x float> %246, <4 x float> %250, !dbg !11
  %v1853 = extractvalue [2 x <8 x float>] %v209.lcssa, 0, !dbg !11
  %v1854 = extractelement <8 x float> %v1853, i64 0, !dbg !11
  %v1855 = extractelement <8 x float> %v1853, i64 1, !dbg !11
  %v1856 = extractelement <8 x float> %v1853, i64 2, !dbg !11
  %v1857 = extractelement <8 x float> %v1853, i64 3, !dbg !11
  %v1858 = select i1 %.not, float %v1855, float %v1854, !dbg !11
  %v1859 = bitcast float %v1858 to i32, !dbg !11
  %v1860 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1859), !dbg !11
  %v1861 = bitcast i32 %v1860 to float, !dbg !11
  %v1862 = select i1 %.not, float %v1854, float %v1861, !dbg !11
  %v1863 = select i1 %.not, float %v1861, float %v1855, !dbg !11
  %v1864 = select i1 %.not, float %v1857, float %v1856, !dbg !11
  %v1865 = bitcast float %v1864 to i32, !dbg !11
  %v1866 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1865), !dbg !11
  %v1867 = bitcast i32 %v1866 to float, !dbg !11
  %v1868 = select i1 %.not, float %v1856, float %v1867, !dbg !11
  %v1869 = select i1 %.not, float %v1867, float %v1857, !dbg !11
  %v1870 = select i1 %.not36, float %v1868, float %v1862, !dbg !11
  %v1871 = bitcast float %v1870 to i32, !dbg !11
  %v1872 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1871), !dbg !11
  %v1873 = bitcast i32 %v1872 to float, !dbg !11
  %v1876 = select i1 %.not36, float %v1869, float %v1863, !dbg !11
  %v1877 = bitcast float %v1876 to i32, !dbg !11
  %v1878 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1877), !dbg !11
  %v1879 = bitcast i32 %v1878 to float, !dbg !11
  %252 = insertelement <4 x float> poison, float %v1862, i64 0, !dbg !11
  %253 = insertelement <4 x float> %252, float %v1863, i64 1, !dbg !11
  %254 = insertelement <4 x float> %253, float %v1873, i64 2, !dbg !11
  %255 = insertelement <4 x float> %254, float %v1879, i64 3, !dbg !11
  %256 = insertelement <4 x float> poison, float %v1873, i64 0, !dbg !11
  %257 = insertelement <4 x float> %256, float %v1879, i64 1, !dbg !11
  %258 = insertelement <4 x float> %257, float %v1868, i64 2, !dbg !11
  %259 = insertelement <4 x float> %258, float %v1869, i64 3, !dbg !11
  %260 = select <4 x i1> %233, <4 x float> %255, <4 x float> %259, !dbg !11
  %v1882 = extractvalue [2 x <8 x float>] %v206.lcssa, 0, !dbg !11
  %v1883 = extractelement <8 x float> %v1882, i64 0, !dbg !11
  %v1884 = extractelement <8 x float> %v1882, i64 1, !dbg !11
  %v1885 = extractelement <8 x float> %v1882, i64 2, !dbg !11
  %v1886 = extractelement <8 x float> %v1882, i64 3, !dbg !11
  %v1887 = select i1 %.not, float %v1884, float %v1883, !dbg !11
  %v1888 = bitcast float %v1887 to i32, !dbg !11
  %v1889 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1888), !dbg !11
  %v1890 = bitcast i32 %v1889 to float, !dbg !11
  %v1891 = select i1 %.not, float %v1883, float %v1890, !dbg !11
  %v1892 = select i1 %.not, float %v1890, float %v1884, !dbg !11
  %v1893 = select i1 %.not, float %v1886, float %v1885, !dbg !11
  %v1894 = bitcast float %v1893 to i32, !dbg !11
  %v1895 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1894), !dbg !11
  %v1896 = bitcast i32 %v1895 to float, !dbg !11
  %v1897 = select i1 %.not, float %v1885, float %v1896, !dbg !11
  %v1898 = select i1 %.not, float %v1896, float %v1886, !dbg !11
  %v1899 = select i1 %.not36, float %v1897, float %v1891, !dbg !11
  %v1900 = bitcast float %v1899 to i32, !dbg !11
  %v1901 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1900), !dbg !11
  %v1902 = select i1 %.not36, float %v1898, float %v1892, !dbg !11
  %v1903 = bitcast float %v1902 to i32, !dbg !11
  %v1904 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1903), !dbg !11
  %v1905 = extractvalue [2 x <1 x float>] %v207.lcssa, 0, !dbg !30
  %v1906 = shufflevector <1 x float> %v1905, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %v1907 = extractelement <8 x float> %v1783, i64 4, !dbg !11
  %v1908 = extractelement <8 x float> %v1783, i64 5, !dbg !11
  %v1909 = extractelement <8 x float> %v1783, i64 6, !dbg !11
  %v1910 = extractelement <8 x float> %v1783, i64 7, !dbg !11
  %v1911 = select i1 %.not, float %v1908, float %v1907, !dbg !11
  %v1912 = bitcast float %v1911 to i32, !dbg !11
  %v1913 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1912), !dbg !11
  %v1914 = bitcast i32 %v1913 to float, !dbg !11
  %v1915 = select i1 %.not, float %v1907, float %v1914, !dbg !11
  %v1916 = select i1 %.not, float %v1914, float %v1908, !dbg !11
  %v1917 = select i1 %.not, float %v1910, float %v1909, !dbg !11
  %v1918 = bitcast float %v1917 to i32, !dbg !11
  %v1919 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1918), !dbg !11
  %v1920 = bitcast i32 %v1919 to float, !dbg !11
  %v1921 = select i1 %.not, float %v1909, float %v1920, !dbg !11
  %v1922 = select i1 %.not, float %v1920, float %v1910, !dbg !11
  %v1923 = select i1 %.not36, float %v1921, float %v1915, !dbg !11
  %v1924 = bitcast float %v1923 to i32, !dbg !11
  %v1925 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1924), !dbg !11
  %v1926 = bitcast i32 %v1925 to float, !dbg !11
  %v1929 = select i1 %.not36, float %v1922, float %v1916, !dbg !11
  %v1930 = bitcast float %v1929 to i32, !dbg !11
  %v1931 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1930), !dbg !11
  %v1932 = bitcast i32 %v1931 to float, !dbg !11
  %261 = insertelement <4 x float> poison, float %v1915, i64 0, !dbg !11
  %262 = insertelement <4 x float> %261, float %v1916, i64 1, !dbg !11
  %263 = insertelement <4 x float> %262, float %v1926, i64 2, !dbg !11
  %264 = insertelement <4 x float> %263, float %v1932, i64 3, !dbg !11
  %265 = insertelement <4 x float> poison, float %v1926, i64 0, !dbg !11
  %266 = insertelement <4 x float> %265, float %v1932, i64 1, !dbg !11
  %267 = insertelement <4 x float> %266, float %v1921, i64 2, !dbg !11
  %268 = insertelement <4 x float> %267, float %v1922, i64 3, !dbg !11
  %269 = select <4 x i1> %233, <4 x float> %264, <4 x float> %268, !dbg !11
  %v1935 = extractelement <8 x float> %v1824, i64 4, !dbg !11
  %v1936 = extractelement <8 x float> %v1824, i64 5, !dbg !11
  %v1937 = extractelement <8 x float> %v1824, i64 6, !dbg !11
  %v1938 = extractelement <8 x float> %v1824, i64 7, !dbg !11
  %v1939 = select i1 %.not, float %v1936, float %v1935, !dbg !11
  %v1940 = bitcast float %v1939 to i32, !dbg !11
  %v1941 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1940), !dbg !11
  %v1942 = bitcast i32 %v1941 to float, !dbg !11
  %v1943 = select i1 %.not, float %v1935, float %v1942, !dbg !11
  %v1944 = select i1 %.not, float %v1942, float %v1936, !dbg !11
  %v1945 = select i1 %.not, float %v1938, float %v1937, !dbg !11
  %v1946 = bitcast float %v1945 to i32, !dbg !11
  %v1947 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1946), !dbg !11
  %v1948 = bitcast i32 %v1947 to float, !dbg !11
  %v1949 = select i1 %.not, float %v1937, float %v1948, !dbg !11
  %v1950 = select i1 %.not, float %v1948, float %v1938, !dbg !11
  %v1951 = select i1 %.not36, float %v1949, float %v1943, !dbg !11
  %v1952 = bitcast float %v1951 to i32, !dbg !11
  %v1953 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1952), !dbg !11
  %v1954 = bitcast i32 %v1953 to float, !dbg !11
  %v1957 = select i1 %.not36, float %v1950, float %v1944, !dbg !11
  %v1958 = bitcast float %v1957 to i32, !dbg !11
  %v1959 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1958), !dbg !11
  %v1960 = bitcast i32 %v1959 to float, !dbg !11
  %270 = insertelement <4 x float> poison, float %v1943, i64 0, !dbg !11
  %271 = insertelement <4 x float> %270, float %v1944, i64 1, !dbg !11
  %272 = insertelement <4 x float> %271, float %v1954, i64 2, !dbg !11
  %273 = insertelement <4 x float> %272, float %v1960, i64 3, !dbg !11
  %274 = insertelement <4 x float> poison, float %v1954, i64 0, !dbg !11
  %275 = insertelement <4 x float> %274, float %v1960, i64 1, !dbg !11
  %276 = insertelement <4 x float> %275, float %v1949, i64 2, !dbg !11
  %277 = insertelement <4 x float> %276, float %v1950, i64 3, !dbg !11
  %278 = select <4 x i1> %233, <4 x float> %273, <4 x float> %277, !dbg !11
  %v1963 = extractelement <8 x float> %v1853, i64 4, !dbg !11
  %v1964 = extractelement <8 x float> %v1853, i64 5, !dbg !11
  %v1965 = extractelement <8 x float> %v1853, i64 6, !dbg !11
  %v1966 = extractelement <8 x float> %v1853, i64 7, !dbg !11
  %v1967 = select i1 %.not, float %v1964, float %v1963, !dbg !11
  %v1968 = bitcast float %v1967 to i32, !dbg !11
  %v1969 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1968), !dbg !11
  %v1970 = bitcast i32 %v1969 to float, !dbg !11
  %v1971 = select i1 %.not, float %v1963, float %v1970, !dbg !11
  %v1972 = select i1 %.not, float %v1970, float %v1964, !dbg !11
  %v1973 = select i1 %.not, float %v1966, float %v1965, !dbg !11
  %v1974 = bitcast float %v1973 to i32, !dbg !11
  %v1975 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1974), !dbg !11
  %v1976 = bitcast i32 %v1975 to float, !dbg !11
  %v1977 = select i1 %.not, float %v1965, float %v1976, !dbg !11
  %v1978 = select i1 %.not, float %v1976, float %v1966, !dbg !11
  %v1979 = select i1 %.not36, float %v1977, float %v1971, !dbg !11
  %v1980 = bitcast float %v1979 to i32, !dbg !11
  %v1981 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1980), !dbg !11
  %v1982 = bitcast i32 %v1981 to float, !dbg !11
  %v1985 = select i1 %.not36, float %v1978, float %v1972, !dbg !11
  %v1986 = bitcast float %v1985 to i32, !dbg !11
  %v1987 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1986), !dbg !11
  %v1988 = bitcast i32 %v1987 to float, !dbg !11
  %279 = insertelement <4 x float> poison, float %v1971, i64 0, !dbg !11
  %280 = insertelement <4 x float> %279, float %v1972, i64 1, !dbg !11
  %281 = insertelement <4 x float> %280, float %v1982, i64 2, !dbg !11
  %282 = insertelement <4 x float> %281, float %v1988, i64 3, !dbg !11
  %283 = insertelement <4 x float> poison, float %v1982, i64 0, !dbg !11
  %284 = insertelement <4 x float> %283, float %v1988, i64 1, !dbg !11
  %285 = insertelement <4 x float> %284, float %v1977, i64 2, !dbg !11
  %286 = insertelement <4 x float> %285, float %v1978, i64 3, !dbg !11
  %287 = select <4 x i1> %233, <4 x float> %282, <4 x float> %286, !dbg !11
  %v1991 = extractelement <8 x float> %v1882, i64 4, !dbg !11
  %v1992 = extractelement <8 x float> %v1882, i64 5, !dbg !11
  %v1993 = extractelement <8 x float> %v1882, i64 6, !dbg !11
  %v1994 = extractelement <8 x float> %v1882, i64 7, !dbg !11
  %v1995 = select i1 %.not, float %v1992, float %v1991, !dbg !11
  %v1996 = bitcast float %v1995 to i32, !dbg !11
  %v1997 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1996), !dbg !11
  %v1998 = bitcast i32 %v1997 to float, !dbg !11
  %v1999 = select i1 %.not, float %v1991, float %v1998, !dbg !11
  %v2000 = select i1 %.not, float %v1998, float %v1992, !dbg !11
  %v2001 = select i1 %.not, float %v1994, float %v1993, !dbg !11
  %v2002 = bitcast float %v2001 to i32, !dbg !11
  %v2003 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2002), !dbg !11
  %v2004 = bitcast i32 %v2003 to float, !dbg !11
  %v2005 = select i1 %.not, float %v1993, float %v2004, !dbg !11
  %v2006 = select i1 %.not, float %v2004, float %v1994, !dbg !11
  %v2007 = select i1 %.not36, float %v2005, float %v1999, !dbg !11
  %v2008 = bitcast float %v2007 to i32, !dbg !11
  %v2009 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2008), !dbg !11
  %v2010 = select i1 %.not36, float %v2006, float %v2000, !dbg !11
  %v2011 = bitcast float %v2010 to i32, !dbg !11
  %v2012 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2011), !dbg !11
  %v2013 = extractvalue [2 x <8 x float>] %v211.lcssa, 1, !dbg !11
  %v2014 = extractelement <8 x float> %v2013, i64 0, !dbg !11
  %v2015 = extractelement <8 x float> %v2013, i64 1, !dbg !11
  %v2016 = extractelement <8 x float> %v2013, i64 2, !dbg !11
  %v2017 = extractelement <8 x float> %v2013, i64 3, !dbg !11
  %v2018 = select i1 %.not, float %v2015, float %v2014, !dbg !11
  %v2019 = bitcast float %v2018 to i32, !dbg !11
  %v2020 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2019), !dbg !11
  %v2021 = bitcast i32 %v2020 to float, !dbg !11
  %v2022 = select i1 %.not, float %v2014, float %v2021, !dbg !11
  %v2023 = select i1 %.not, float %v2021, float %v2015, !dbg !11
  %v2024 = select i1 %.not, float %v2017, float %v2016, !dbg !11
  %v2025 = bitcast float %v2024 to i32, !dbg !11
  %v2026 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2025), !dbg !11
  %v2027 = bitcast i32 %v2026 to float, !dbg !11
  %v2028 = select i1 %.not, float %v2016, float %v2027, !dbg !11
  %v2029 = select i1 %.not, float %v2027, float %v2017, !dbg !11
  %v2030 = select i1 %.not36, float %v2028, float %v2022, !dbg !11
  %v2031 = bitcast float %v2030 to i32, !dbg !11
  %v2032 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2031), !dbg !11
  %v2033 = bitcast i32 %v2032 to float, !dbg !11
  %v2036 = select i1 %.not36, float %v2029, float %v2023, !dbg !11
  %v2037 = bitcast float %v2036 to i32, !dbg !11
  %v2038 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2037), !dbg !11
  %v2039 = bitcast i32 %v2038 to float, !dbg !11
  %288 = insertelement <4 x float> poison, float %v2022, i64 0, !dbg !11
  %289 = insertelement <4 x float> %288, float %v2023, i64 1, !dbg !11
  %290 = insertelement <4 x float> %289, float %v2033, i64 2, !dbg !11
  %291 = insertelement <4 x float> %290, float %v2039, i64 3, !dbg !11
  %292 = insertelement <4 x float> poison, float %v2033, i64 0, !dbg !11
  %293 = insertelement <4 x float> %292, float %v2039, i64 1, !dbg !11
  %294 = insertelement <4 x float> %293, float %v2028, i64 2, !dbg !11
  %295 = insertelement <4 x float> %294, float %v2029, i64 3, !dbg !11
  %296 = select <4 x i1> %233, <4 x float> %291, <4 x float> %295, !dbg !11
  %v2042 = extractvalue [2 x <8 x float>] %v210.lcssa, 1, !dbg !11
  %v2043 = extractelement <8 x float> %v2042, i64 0, !dbg !11
  %v2044 = extractelement <8 x float> %v2042, i64 1, !dbg !11
  %v2045 = extractelement <8 x float> %v2042, i64 2, !dbg !11
  %v2046 = extractelement <8 x float> %v2042, i64 3, !dbg !11
  %v2047 = select i1 %.not, float %v2044, float %v2043, !dbg !11
  %v2048 = bitcast float %v2047 to i32, !dbg !11
  %v2049 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2048), !dbg !11
  %v2050 = bitcast i32 %v2049 to float, !dbg !11
  %v2051 = select i1 %.not, float %v2043, float %v2050, !dbg !11
  %v2052 = select i1 %.not, float %v2050, float %v2044, !dbg !11
  %v2053 = select i1 %.not, float %v2046, float %v2045, !dbg !11
  %v2054 = bitcast float %v2053 to i32, !dbg !11
  %v2055 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2054), !dbg !11
  %v2056 = bitcast i32 %v2055 to float, !dbg !11
  %v2057 = select i1 %.not, float %v2045, float %v2056, !dbg !11
  %v2058 = select i1 %.not, float %v2056, float %v2046, !dbg !11
  %v2059 = select i1 %.not36, float %v2057, float %v2051, !dbg !11
  %v2060 = bitcast float %v2059 to i32, !dbg !11
  %v2061 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2060), !dbg !11
  %v2062 = bitcast i32 %v2061 to float, !dbg !11
  %v2065 = select i1 %.not36, float %v2058, float %v2052, !dbg !11
  %v2066 = bitcast float %v2065 to i32, !dbg !11
  %v2067 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2066), !dbg !11
  %v2068 = bitcast i32 %v2067 to float, !dbg !11
  %297 = insertelement <4 x float> poison, float %v2051, i64 0, !dbg !11
  %298 = insertelement <4 x float> %297, float %v2052, i64 1, !dbg !11
  %299 = insertelement <4 x float> %298, float %v2062, i64 2, !dbg !11
  %300 = insertelement <4 x float> %299, float %v2068, i64 3, !dbg !11
  %301 = insertelement <4 x float> poison, float %v2062, i64 0, !dbg !11
  %302 = insertelement <4 x float> %301, float %v2068, i64 1, !dbg !11
  %303 = insertelement <4 x float> %302, float %v2057, i64 2, !dbg !11
  %304 = insertelement <4 x float> %303, float %v2058, i64 3, !dbg !11
  %305 = select <4 x i1> %233, <4 x float> %300, <4 x float> %304, !dbg !11
  %v2071 = extractvalue [2 x <8 x float>] %v209.lcssa, 1, !dbg !11
  %v2072 = extractelement <8 x float> %v2071, i64 0, !dbg !11
  %v2073 = extractelement <8 x float> %v2071, i64 1, !dbg !11
  %v2074 = extractelement <8 x float> %v2071, i64 2, !dbg !11
  %v2075 = extractelement <8 x float> %v2071, i64 3, !dbg !11
  %v2076 = select i1 %.not, float %v2073, float %v2072, !dbg !11
  %v2077 = bitcast float %v2076 to i32, !dbg !11
  %v2078 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2077), !dbg !11
  %v2079 = bitcast i32 %v2078 to float, !dbg !11
  %v2080 = select i1 %.not, float %v2072, float %v2079, !dbg !11
  %v2081 = select i1 %.not, float %v2079, float %v2073, !dbg !11
  %v2082 = select i1 %.not, float %v2075, float %v2074, !dbg !11
  %v2083 = bitcast float %v2082 to i32, !dbg !11
  %v2084 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2083), !dbg !11
  %v2085 = bitcast i32 %v2084 to float, !dbg !11
  %v2086 = select i1 %.not, float %v2074, float %v2085, !dbg !11
  %v2087 = select i1 %.not, float %v2085, float %v2075, !dbg !11
  %v2088 = select i1 %.not36, float %v2086, float %v2080, !dbg !11
  %v2089 = bitcast float %v2088 to i32, !dbg !11
  %v2090 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2089), !dbg !11
  %v2091 = bitcast i32 %v2090 to float, !dbg !11
  %v2094 = select i1 %.not36, float %v2087, float %v2081, !dbg !11
  %v2095 = bitcast float %v2094 to i32, !dbg !11
  %v2096 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2095), !dbg !11
  %v2097 = bitcast i32 %v2096 to float, !dbg !11
  %306 = insertelement <4 x float> poison, float %v2080, i64 0, !dbg !11
  %307 = insertelement <4 x float> %306, float %v2081, i64 1, !dbg !11
  %308 = insertelement <4 x float> %307, float %v2091, i64 2, !dbg !11
  %309 = insertelement <4 x float> %308, float %v2097, i64 3, !dbg !11
  %310 = insertelement <4 x float> poison, float %v2091, i64 0, !dbg !11
  %311 = insertelement <4 x float> %310, float %v2097, i64 1, !dbg !11
  %312 = insertelement <4 x float> %311, float %v2086, i64 2, !dbg !11
  %313 = insertelement <4 x float> %312, float %v2087, i64 3, !dbg !11
  %314 = select <4 x i1> %233, <4 x float> %309, <4 x float> %313, !dbg !11
  %v2100 = extractvalue [2 x <8 x float>] %v206.lcssa, 1, !dbg !11
  %v2101 = extractelement <8 x float> %v2100, i64 0, !dbg !11
  %v2102 = extractelement <8 x float> %v2100, i64 1, !dbg !11
  %v2103 = extractelement <8 x float> %v2100, i64 2, !dbg !11
  %v2104 = extractelement <8 x float> %v2100, i64 3, !dbg !11
  %v2105 = select i1 %.not, float %v2102, float %v2101, !dbg !11
  %v2106 = bitcast float %v2105 to i32, !dbg !11
  %v2107 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2106), !dbg !11
  %v2108 = bitcast i32 %v2107 to float, !dbg !11
  %v2109 = select i1 %.not, float %v2101, float %v2108, !dbg !11
  %v2110 = select i1 %.not, float %v2108, float %v2102, !dbg !11
  %v2111 = select i1 %.not, float %v2104, float %v2103, !dbg !11
  %v2112 = bitcast float %v2111 to i32, !dbg !11
  %v2113 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2112), !dbg !11
  %v2114 = bitcast i32 %v2113 to float, !dbg !11
  %v2115 = select i1 %.not, float %v2103, float %v2114, !dbg !11
  %v2116 = select i1 %.not, float %v2114, float %v2104, !dbg !11
  %v2117 = select i1 %.not36, float %v2115, float %v2109, !dbg !11
  %v2118 = bitcast float %v2117 to i32, !dbg !11
  %v2119 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2118), !dbg !11
  %v2120 = select i1 %.not36, float %v2116, float %v2110, !dbg !11
  %v2121 = bitcast float %v2120 to i32, !dbg !11
  %v2122 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2121), !dbg !11
  %v2123 = extractvalue [2 x <1 x float>] %v207.lcssa, 1, !dbg !30
  %v2124 = shufflevector <1 x float> %v2123, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %v2125 = extractelement <8 x float> %v2013, i64 4, !dbg !11
  %v2126 = extractelement <8 x float> %v2013, i64 5, !dbg !11
  %v2127 = extractelement <8 x float> %v2013, i64 6, !dbg !11
  %v2128 = extractelement <8 x float> %v2013, i64 7, !dbg !11
  %v2129 = select i1 %.not, float %v2126, float %v2125, !dbg !11
  %v2130 = bitcast float %v2129 to i32, !dbg !11
  %v2131 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2130), !dbg !11
  %v2132 = bitcast i32 %v2131 to float, !dbg !11
  %v2133 = select i1 %.not, float %v2125, float %v2132, !dbg !11
  %v2134 = select i1 %.not, float %v2132, float %v2126, !dbg !11
  %v2135 = select i1 %.not, float %v2128, float %v2127, !dbg !11
  %v2136 = bitcast float %v2135 to i32, !dbg !11
  %v2137 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2136), !dbg !11
  %v2138 = bitcast i32 %v2137 to float, !dbg !11
  %v2139 = select i1 %.not, float %v2127, float %v2138, !dbg !11
  %v2140 = select i1 %.not, float %v2138, float %v2128, !dbg !11
  %v2141 = select i1 %.not36, float %v2139, float %v2133, !dbg !11
  %v2142 = bitcast float %v2141 to i32, !dbg !11
  %v2143 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2142), !dbg !11
  %v2144 = bitcast i32 %v2143 to float, !dbg !11
  %v2147 = select i1 %.not36, float %v2140, float %v2134, !dbg !11
  %v2148 = bitcast float %v2147 to i32, !dbg !11
  %v2149 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2148), !dbg !11
  %v2150 = bitcast i32 %v2149 to float, !dbg !11
  %315 = insertelement <4 x float> poison, float %v2133, i64 0, !dbg !11
  %316 = insertelement <4 x float> %315, float %v2134, i64 1, !dbg !11
  %317 = insertelement <4 x float> %316, float %v2144, i64 2, !dbg !11
  %318 = insertelement <4 x float> %317, float %v2150, i64 3, !dbg !11
  %319 = insertelement <4 x float> poison, float %v2144, i64 0, !dbg !11
  %320 = insertelement <4 x float> %319, float %v2150, i64 1, !dbg !11
  %321 = insertelement <4 x float> %320, float %v2139, i64 2, !dbg !11
  %322 = insertelement <4 x float> %321, float %v2140, i64 3, !dbg !11
  %323 = select <4 x i1> %233, <4 x float> %318, <4 x float> %322, !dbg !11
  %v2153 = extractelement <8 x float> %v2042, i64 4, !dbg !11
  %v2154 = extractelement <8 x float> %v2042, i64 5, !dbg !11
  %v2155 = extractelement <8 x float> %v2042, i64 6, !dbg !11
  %v2156 = extractelement <8 x float> %v2042, i64 7, !dbg !11
  %v2157 = select i1 %.not, float %v2154, float %v2153, !dbg !11
  %v2158 = bitcast float %v2157 to i32, !dbg !11
  %v2159 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2158), !dbg !11
  %v2160 = bitcast i32 %v2159 to float, !dbg !11
  %v2161 = select i1 %.not, float %v2153, float %v2160, !dbg !11
  %v2162 = select i1 %.not, float %v2160, float %v2154, !dbg !11
  %v2163 = select i1 %.not, float %v2156, float %v2155, !dbg !11
  %v2164 = bitcast float %v2163 to i32, !dbg !11
  %v2165 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2164), !dbg !11
  %v2166 = bitcast i32 %v2165 to float, !dbg !11
  %v2167 = select i1 %.not, float %v2155, float %v2166, !dbg !11
  %v2168 = select i1 %.not, float %v2166, float %v2156, !dbg !11
  %v2169 = select i1 %.not36, float %v2167, float %v2161, !dbg !11
  %v2170 = bitcast float %v2169 to i32, !dbg !11
  %v2171 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2170), !dbg !11
  %v2172 = bitcast i32 %v2171 to float, !dbg !11
  %v2175 = select i1 %.not36, float %v2168, float %v2162, !dbg !11
  %v2176 = bitcast float %v2175 to i32, !dbg !11
  %v2177 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2176), !dbg !11
  %v2178 = bitcast i32 %v2177 to float, !dbg !11
  %324 = insertelement <4 x float> poison, float %v2161, i64 0, !dbg !11
  %325 = insertelement <4 x float> %324, float %v2162, i64 1, !dbg !11
  %326 = insertelement <4 x float> %325, float %v2172, i64 2, !dbg !11
  %327 = insertelement <4 x float> %326, float %v2178, i64 3, !dbg !11
  %328 = insertelement <4 x float> poison, float %v2172, i64 0, !dbg !11
  %329 = insertelement <4 x float> %328, float %v2178, i64 1, !dbg !11
  %330 = insertelement <4 x float> %329, float %v2167, i64 2, !dbg !11
  %331 = insertelement <4 x float> %330, float %v2168, i64 3, !dbg !11
  %332 = select <4 x i1> %233, <4 x float> %327, <4 x float> %331, !dbg !11
  %v2181 = extractelement <8 x float> %v2071, i64 4, !dbg !11
  %v2182 = extractelement <8 x float> %v2071, i64 5, !dbg !11
  %v2183 = extractelement <8 x float> %v2071, i64 6, !dbg !11
  %v2184 = extractelement <8 x float> %v2071, i64 7, !dbg !11
  %v2185 = select i1 %.not, float %v2182, float %v2181, !dbg !11
  %v2186 = bitcast float %v2185 to i32, !dbg !11
  %v2187 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2186), !dbg !11
  %v2188 = bitcast i32 %v2187 to float, !dbg !11
  %v2189 = select i1 %.not, float %v2181, float %v2188, !dbg !11
  %v2190 = select i1 %.not, float %v2188, float %v2182, !dbg !11
  %v2191 = select i1 %.not, float %v2184, float %v2183, !dbg !11
  %v2192 = bitcast float %v2191 to i32, !dbg !11
  %v2193 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2192), !dbg !11
  %v2194 = bitcast i32 %v2193 to float, !dbg !11
  %v2195 = select i1 %.not, float %v2183, float %v2194, !dbg !11
  %v2196 = select i1 %.not, float %v2194, float %v2184, !dbg !11
  %v2197 = select i1 %.not36, float %v2195, float %v2189, !dbg !11
  %v2198 = bitcast float %v2197 to i32, !dbg !11
  %v2199 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2198), !dbg !11
  %v2200 = bitcast i32 %v2199 to float, !dbg !11
  %v2203 = select i1 %.not36, float %v2196, float %v2190, !dbg !11
  %v2204 = bitcast float %v2203 to i32, !dbg !11
  %v2205 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2204), !dbg !11
  %v2206 = bitcast i32 %v2205 to float, !dbg !11
  %333 = insertelement <4 x float> poison, float %v2189, i64 0, !dbg !11
  %334 = insertelement <4 x float> %333, float %v2190, i64 1, !dbg !11
  %335 = insertelement <4 x float> %334, float %v2200, i64 2, !dbg !11
  %336 = insertelement <4 x float> %335, float %v2206, i64 3, !dbg !11
  %337 = insertelement <4 x float> poison, float %v2200, i64 0, !dbg !11
  %338 = insertelement <4 x float> %337, float %v2206, i64 1, !dbg !11
  %339 = insertelement <4 x float> %338, float %v2195, i64 2, !dbg !11
  %340 = insertelement <4 x float> %339, float %v2196, i64 3, !dbg !11
  %341 = select <4 x i1> %233, <4 x float> %336, <4 x float> %340, !dbg !11
  %v2209 = extractelement <8 x float> %v2100, i64 4, !dbg !11
  %v2210 = extractelement <8 x float> %v2100, i64 5, !dbg !11
  %v2211 = extractelement <8 x float> %v2100, i64 6, !dbg !11
  %v2212 = extractelement <8 x float> %v2100, i64 7, !dbg !11
  %v2213 = select i1 %.not, float %v2210, float %v2209, !dbg !11
  %v2214 = bitcast float %v2213 to i32, !dbg !11
  %v2215 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2214), !dbg !11
  %v2216 = bitcast i32 %v2215 to float, !dbg !11
  %v2217 = select i1 %.not, float %v2209, float %v2216, !dbg !11
  %v2218 = select i1 %.not, float %v2216, float %v2210, !dbg !11
  %v2219 = select i1 %.not, float %v2212, float %v2211, !dbg !11
  %v2220 = bitcast float %v2219 to i32, !dbg !11
  %v2221 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2220), !dbg !11
  %v2222 = bitcast i32 %v2221 to float, !dbg !11
  %v2223 = select i1 %.not, float %v2211, float %v2222, !dbg !11
  %v2224 = select i1 %.not, float %v2222, float %v2212, !dbg !11
  %v2225 = select i1 %.not36, float %v2223, float %v2217, !dbg !11
  %v2226 = bitcast float %v2225 to i32, !dbg !11
  %v2227 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2226), !dbg !11
  %v2228 = select i1 %.not36, float %v2224, float %v2218, !dbg !11
  %v2229 = bitcast float %v2228 to i32, !dbg !11
  %v2230 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2229), !dbg !11
  %v2232 = bitcast i32 %v2227 to float, !dbg !11
  %v2235 = bitcast i32 %v2230 to float, !dbg !11
  %342 = insertelement <4 x float> poison, float %v2217, i64 0, !dbg !11
  %343 = insertelement <4 x float> %342, float %v2218, i64 1, !dbg !11
  %344 = insertelement <4 x float> %343, float %v2232, i64 2, !dbg !11
  %345 = insertelement <4 x float> %344, float %v2235, i64 3, !dbg !11
  %346 = insertelement <4 x float> poison, float %v2232, i64 0, !dbg !11
  %347 = insertelement <4 x float> %346, float %v2235, i64 1, !dbg !11
  %348 = insertelement <4 x float> %347, float %v2223, i64 2, !dbg !11
  %349 = insertelement <4 x float> %348, float %v2224, i64 3, !dbg !11
  %350 = select <4 x i1> %233, <4 x float> %345, <4 x float> %349, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %v2124, !dbg !30
  %normalized = fmul <4 x float> %row.reciprocal, %350, !dbg !30
  %v2242 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !31
  %v2243 = extractelement <4 x half> %v2242, i64 3, !dbg !31
  %v2244 = extractelement <4 x half> %v2242, i64 2, !dbg !31
  %v2245 = extractelement <4 x half> %v2242, i64 1, !dbg !31
  %v2246 = extractelement <4 x half> %v2242, i64 0, !dbg !31
  %v2247 = bitcast i32 %v2119 to float, !dbg !11
  %v2250 = bitcast i32 %v2122 to float, !dbg !11
  %351 = insertelement <4 x float> poison, float %v2109, i64 0, !dbg !11
  %352 = insertelement <4 x float> %351, float %v2110, i64 1, !dbg !11
  %353 = insertelement <4 x float> %352, float %v2247, i64 2, !dbg !11
  %354 = insertelement <4 x float> %353, float %v2250, i64 3, !dbg !11
  %355 = insertelement <4 x float> poison, float %v2247, i64 0, !dbg !11
  %356 = insertelement <4 x float> %355, float %v2250, i64 1, !dbg !11
  %357 = insertelement <4 x float> %356, float %v2115, i64 2, !dbg !11
  %358 = insertelement <4 x float> %357, float %v2116, i64 3, !dbg !11
  %359 = select <4 x i1> %233, <4 x float> %354, <4 x float> %358, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %359, !dbg !30
  %v2257 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !31
  %v2258 = extractelement <4 x half> %v2257, i64 3, !dbg !31
  %v2259 = extractelement <4 x half> %v2257, i64 2, !dbg !31
  %v2260 = extractelement <4 x half> %v2257, i64 1, !dbg !31
  %v2261 = extractelement <4 x half> %v2257, i64 0, !dbg !31
  %v2262 = bitcast i32 %v2009 to float, !dbg !11
  %v2265 = bitcast i32 %v2012 to float, !dbg !11
  %360 = insertelement <4 x float> poison, float %v1999, i64 0, !dbg !11
  %361 = insertelement <4 x float> %360, float %v2000, i64 1, !dbg !11
  %362 = insertelement <4 x float> %361, float %v2262, i64 2, !dbg !11
  %363 = insertelement <4 x float> %362, float %v2265, i64 3, !dbg !11
  %364 = insertelement <4 x float> poison, float %v2262, i64 0, !dbg !11
  %365 = insertelement <4 x float> %364, float %v2265, i64 1, !dbg !11
  %366 = insertelement <4 x float> %365, float %v2005, i64 2, !dbg !11
  %367 = insertelement <4 x float> %366, float %v2006, i64 3, !dbg !11
  %368 = select <4 x i1> %233, <4 x float> %363, <4 x float> %367, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %v1906, !dbg !30
  %normalized106 = fmul <4 x float> %row.reciprocal105, %368, !dbg !30
  %v2272 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !31
  %v2273 = extractelement <4 x half> %v2272, i64 3, !dbg !31
  %v2274 = extractelement <4 x half> %v2272, i64 2, !dbg !31
  %v2275 = extractelement <4 x half> %v2272, i64 1, !dbg !31
  %v2276 = extractelement <4 x half> %v2272, i64 0, !dbg !31
  %v2277 = bitcast i32 %v1901 to float, !dbg !11
  %v2280 = bitcast i32 %v1904 to float, !dbg !11
  %369 = insertelement <4 x float> poison, float %v1891, i64 0, !dbg !11
  %370 = insertelement <4 x float> %369, float %v1892, i64 1, !dbg !11
  %371 = insertelement <4 x float> %370, float %v2277, i64 2, !dbg !11
  %372 = insertelement <4 x float> %371, float %v2280, i64 3, !dbg !11
  %373 = insertelement <4 x float> poison, float %v2277, i64 0, !dbg !11
  %374 = insertelement <4 x float> %373, float %v2280, i64 1, !dbg !11
  %375 = insertelement <4 x float> %374, float %v1897, i64 2, !dbg !11
  %376 = insertelement <4 x float> %375, float %v1898, i64 3, !dbg !11
  %377 = select <4 x i1> %233, <4 x float> %372, <4 x float> %376, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %377, !dbg !30
  %v2287 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !31
  %v2288 = extractelement <4 x half> %v2287, i64 3, !dbg !31
  %v2289 = extractelement <4 x half> %v2287, i64 2, !dbg !31
  %v2290 = extractelement <4 x half> %v2287, i64 1, !dbg !31
  %v2291 = extractelement <4 x half> %v2287, i64 0, !dbg !31
  %address.thread934 = tail call i32 asm sideeffect "", "=v,0"(i32 %v31) #8, !dbg !32, !range !9, !frisk.register_lifetimes !5
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !8
  %378 = lshr i64 %address.local935, 2, !dbg !32
  %address.local938 = and i64 %378, 48, !dbg !32
  %address.local940 = and i64 %address.local935, 240, !dbg !32
  %address.local939 = add i64 %v27, %address.local935, !dbg !32
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !32
  %address.local942 = add i64 %address.local941, %address.local938, !dbg !32
  %379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %v28, !dbg !32
  %address.local945.idx = shl nuw nsw i64 %address.local942, 8, !dbg !32
  %address.local945 = getelementptr inbounds nuw i8, ptr addrspace(1) %379, i64 %address.local945.idx, !dbg !32
  %address.local947 = and i64 %378, 12, !dbg !10
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !32
  store half %v2291, ptr addrspace(1) %address.local948, align 2, !dbg !32
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !32
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local950, !dbg !32
  store half %v2290, ptr addrspace(1) %address.local951, align 2, !dbg !32
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !32
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local953, !dbg !32
  store half %v2289, ptr addrspace(1) %address.local954, align 2, !dbg !32
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !32
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local956, !dbg !32
  store half %v2288, ptr addrspace(1) %address.local957, align 2, !dbg !32
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !32
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local959, !dbg !32
  store half %v2276, ptr addrspace(1) %address.local960, align 2, !dbg !32
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !32
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local962, !dbg !32
  store half %v2275, ptr addrspace(1) %address.local963, align 2, !dbg !32
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !32
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local965, !dbg !32
  store half %v2274, ptr addrspace(1) %address.local966, align 2, !dbg !32
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !32
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local968, !dbg !32
  store half %v2273, ptr addrspace(1) %address.local969, align 2, !dbg !32
  %address.local973 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local945, i64 16384, !dbg !32
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !32
  store half %v2261, ptr addrspace(1) %address.local974, align 2, !dbg !32
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local950, !dbg !32
  store half %v2260, ptr addrspace(1) %address.local976, align 2, !dbg !32
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local953, !dbg !32
  store half %v2259, ptr addrspace(1) %address.local978, align 2, !dbg !32
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local956, !dbg !32
  store half %v2258, ptr addrspace(1) %address.local980, align 2, !dbg !32
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local959, !dbg !32
  store half %v2246, ptr addrspace(1) %address.local982, align 2, !dbg !32
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local962, !dbg !32
  store half %v2245, ptr addrspace(1) %address.local984, align 2, !dbg !32
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local965, !dbg !32
  store half %v2244, ptr addrspace(1) %address.local986, align 2, !dbg !32
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local968, !dbg !32
  store half %v2243, ptr addrspace(1) %address.local988, align 2, !dbg !32
  %normalized109 = fmul <4 x float> %row.reciprocal, %341, !dbg !30
  %v2301 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !31
  %v2302 = extractelement <4 x half> %v2301, i64 3, !dbg !31
  %v2303 = extractelement <4 x half> %v2301, i64 2, !dbg !31
  %v2304 = extractelement <4 x half> %v2301, i64 1, !dbg !31
  %v2305 = extractelement <4 x half> %v2301, i64 0, !dbg !31
  %normalized110 = fmul <4 x float> %row.reciprocal, %314, !dbg !30
  %v2310 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !31
  %v2311 = extractelement <4 x half> %v2310, i64 3, !dbg !31
  %v2312 = extractelement <4 x half> %v2310, i64 2, !dbg !31
  %v2313 = extractelement <4 x half> %v2310, i64 1, !dbg !31
  %v2314 = extractelement <4 x half> %v2310, i64 0, !dbg !31
  %normalized112 = fmul <4 x float> %row.reciprocal105, %287, !dbg !30
  %v2319 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !31
  %v2320 = extractelement <4 x half> %v2319, i64 3, !dbg !31
  %v2321 = extractelement <4 x half> %v2319, i64 2, !dbg !31
  %v2322 = extractelement <4 x half> %v2319, i64 1, !dbg !31
  %v2323 = extractelement <4 x half> %v2319, i64 0, !dbg !31
  %normalized113 = fmul <4 x float> %row.reciprocal105, %260, !dbg !30
  %v2328 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !31
  %v2329 = extractelement <4 x half> %v2328, i64 3, !dbg !31
  %v2330 = extractelement <4 x half> %v2328, i64 2, !dbg !31
  %v2331 = extractelement <4 x half> %v2328, i64 1, !dbg !31
  %v2332 = extractelement <4 x half> %v2328, i64 0, !dbg !31
  %address.local1003 = or disjoint i64 %address.local947, 32, !dbg !32
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1003, !dbg !32
  store half %v2332, ptr addrspace(1) %address.local1004, align 2, !dbg !32
  %address.local1006 = or disjoint i64 %address.local947, 33, !dbg !32
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1006, !dbg !32
  store half %v2331, ptr addrspace(1) %address.local1007, align 2, !dbg !32
  %address.local1009 = or disjoint i64 %address.local947, 34, !dbg !32
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1009, !dbg !32
  store half %v2330, ptr addrspace(1) %address.local1010, align 2, !dbg !32
  %address.local1012 = or disjoint i64 %address.local947, 35, !dbg !32
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1012, !dbg !32
  store half %v2329, ptr addrspace(1) %address.local1013, align 2, !dbg !32
  %address.local1015 = or disjoint i64 %address.local947, 48, !dbg !32
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1015, !dbg !32
  store half %v2323, ptr addrspace(1) %address.local1016, align 2, !dbg !32
  %address.local1018 = or disjoint i64 %address.local947, 49, !dbg !32
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1018, !dbg !32
  store half %v2322, ptr addrspace(1) %address.local1019, align 2, !dbg !32
  %address.local1021 = or disjoint i64 %address.local947, 50, !dbg !32
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1021, !dbg !32
  store half %v2321, ptr addrspace(1) %address.local1022, align 2, !dbg !32
  %address.local1024 = or i64 %378, 51, !dbg !32
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1024, !dbg !32
  store half %v2320, ptr addrspace(1) %address.local1025, align 2, !dbg !32
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1003, !dbg !32
  store half %v2314, ptr addrspace(1) %address.local1030, align 2, !dbg !32
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1006, !dbg !32
  store half %v2313, ptr addrspace(1) %address.local1032, align 2, !dbg !32
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1009, !dbg !32
  store half %v2312, ptr addrspace(1) %address.local1034, align 2, !dbg !32
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1012, !dbg !32
  store half %v2311, ptr addrspace(1) %address.local1036, align 2, !dbg !32
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1015, !dbg !32
  store half %v2305, ptr addrspace(1) %address.local1038, align 2, !dbg !32
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1018, !dbg !32
  store half %v2304, ptr addrspace(1) %address.local1040, align 2, !dbg !32
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1021, !dbg !32
  store half %v2303, ptr addrspace(1) %address.local1042, align 2, !dbg !32
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1024, !dbg !32
  store half %v2302, ptr addrspace(1) %address.local1044, align 2, !dbg !32
  %normalized115 = fmul <4 x float> %row.reciprocal, %332, !dbg !30
  %v2340 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !31
  %v2341 = extractelement <4 x half> %v2340, i64 3, !dbg !31
  %v2342 = extractelement <4 x half> %v2340, i64 2, !dbg !31
  %v2343 = extractelement <4 x half> %v2340, i64 1, !dbg !31
  %v2344 = extractelement <4 x half> %v2340, i64 0, !dbg !31
  %normalized116 = fmul <4 x float> %row.reciprocal, %305, !dbg !30
  %v2349 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !31
  %v2350 = extractelement <4 x half> %v2349, i64 3, !dbg !31
  %v2351 = extractelement <4 x half> %v2349, i64 2, !dbg !31
  %v2352 = extractelement <4 x half> %v2349, i64 1, !dbg !31
  %v2353 = extractelement <4 x half> %v2349, i64 0, !dbg !31
  %normalized118 = fmul <4 x float> %row.reciprocal105, %278, !dbg !30
  %v2358 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !31
  %v2359 = extractelement <4 x half> %v2358, i64 3, !dbg !31
  %v2360 = extractelement <4 x half> %v2358, i64 2, !dbg !31
  %v2361 = extractelement <4 x half> %v2358, i64 1, !dbg !31
  %v2362 = extractelement <4 x half> %v2358, i64 0, !dbg !31
  %normalized119 = fmul <4 x float> %row.reciprocal105, %251, !dbg !30
  %v2367 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !31
  %v2368 = extractelement <4 x half> %v2367, i64 3, !dbg !31
  %v2369 = extractelement <4 x half> %v2367, i64 2, !dbg !31
  %v2370 = extractelement <4 x half> %v2367, i64 1, !dbg !31
  %v2371 = extractelement <4 x half> %v2367, i64 0, !dbg !31
  %address.local1059 = or disjoint i64 %address.local947, 64, !dbg !32
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1059, !dbg !32
  store half %v2371, ptr addrspace(1) %address.local1060, align 2, !dbg !32
  %address.local1062 = or disjoint i64 %address.local947, 65, !dbg !32
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1062, !dbg !32
  store half %v2370, ptr addrspace(1) %address.local1063, align 2, !dbg !32
  %address.local1065 = or disjoint i64 %address.local947, 66, !dbg !32
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1065, !dbg !32
  store half %v2369, ptr addrspace(1) %address.local1066, align 2, !dbg !32
  %address.local1068 = or disjoint i64 %address.local947, 67, !dbg !32
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1068, !dbg !32
  store half %v2368, ptr addrspace(1) %address.local1069, align 2, !dbg !32
  %address.local1071 = or disjoint i64 %address.local947, 80, !dbg !32
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1071, !dbg !32
  store half %v2362, ptr addrspace(1) %address.local1072, align 2, !dbg !32
  %address.local1074 = or disjoint i64 %address.local947, 81, !dbg !32
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1074, !dbg !32
  store half %v2361, ptr addrspace(1) %address.local1075, align 2, !dbg !32
  %address.local1077 = or disjoint i64 %address.local947, 82, !dbg !32
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1077, !dbg !32
  store half %v2360, ptr addrspace(1) %address.local1078, align 2, !dbg !32
  %address.local1080 = or disjoint i64 %address.local947, 83, !dbg !32
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1080, !dbg !32
  store half %v2359, ptr addrspace(1) %address.local1081, align 2, !dbg !32
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1059, !dbg !32
  store half %v2353, ptr addrspace(1) %address.local1086, align 2, !dbg !32
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1062, !dbg !32
  store half %v2352, ptr addrspace(1) %address.local1088, align 2, !dbg !32
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1065, !dbg !32
  store half %v2351, ptr addrspace(1) %address.local1090, align 2, !dbg !32
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1068, !dbg !32
  store half %v2350, ptr addrspace(1) %address.local1092, align 2, !dbg !32
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1071, !dbg !32
  store half %v2344, ptr addrspace(1) %address.local1094, align 2, !dbg !32
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1074, !dbg !32
  store half %v2343, ptr addrspace(1) %address.local1096, align 2, !dbg !32
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1077, !dbg !32
  store half %v2342, ptr addrspace(1) %address.local1098, align 2, !dbg !32
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1080, !dbg !32
  store half %v2341, ptr addrspace(1) %address.local1100, align 2, !dbg !32
  %normalized121 = fmul <4 x float> %row.reciprocal, %323, !dbg !30
  %v2379 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !31
  %v2380 = extractelement <4 x half> %v2379, i64 3, !dbg !31
  %v2381 = extractelement <4 x half> %v2379, i64 2, !dbg !31
  %v2382 = extractelement <4 x half> %v2379, i64 1, !dbg !31
  %v2383 = extractelement <4 x half> %v2379, i64 0, !dbg !31
  %normalized122 = fmul <4 x float> %row.reciprocal, %296, !dbg !30
  %v2388 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !31
  %v2389 = extractelement <4 x half> %v2388, i64 3, !dbg !31
  %v2390 = extractelement <4 x half> %v2388, i64 2, !dbg !31
  %v2391 = extractelement <4 x half> %v2388, i64 1, !dbg !31
  %v2392 = extractelement <4 x half> %v2388, i64 0, !dbg !31
  %normalized124 = fmul <4 x float> %row.reciprocal105, %269, !dbg !30
  %v2397 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !31
  %v2398 = extractelement <4 x half> %v2397, i64 3, !dbg !31
  %v2399 = extractelement <4 x half> %v2397, i64 2, !dbg !31
  %v2400 = extractelement <4 x half> %v2397, i64 1, !dbg !31
  %v2401 = extractelement <4 x half> %v2397, i64 0, !dbg !31
  %normalized125 = fmul <4 x float> %row.reciprocal105, %242, !dbg !30
  %v2406 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !31
  %v2407 = extractelement <4 x half> %v2406, i64 3, !dbg !31
  %v2408 = extractelement <4 x half> %v2406, i64 2, !dbg !31
  %v2409 = extractelement <4 x half> %v2406, i64 1, !dbg !31
  %v2410 = extractelement <4 x half> %v2406, i64 0, !dbg !31
  %address.local1115 = or disjoint i64 %address.local947, 96, !dbg !32
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1115, !dbg !32
  store half %v2410, ptr addrspace(1) %address.local1116, align 2, !dbg !32
  %address.local1118 = or disjoint i64 %address.local947, 97, !dbg !32
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1118, !dbg !32
  store half %v2409, ptr addrspace(1) %address.local1119, align 2, !dbg !32
  %address.local1121 = or disjoint i64 %address.local947, 98, !dbg !32
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1121, !dbg !32
  store half %v2408, ptr addrspace(1) %address.local1122, align 2, !dbg !32
  %address.local1124 = or disjoint i64 %address.local947, 99, !dbg !32
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1124, !dbg !32
  store half %v2407, ptr addrspace(1) %address.local1125, align 2, !dbg !32
  %address.local1127 = or disjoint i64 %address.local947, 112, !dbg !32
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1127, !dbg !32
  store half %v2401, ptr addrspace(1) %address.local1128, align 2, !dbg !32
  %address.local1130 = or disjoint i64 %address.local947, 113, !dbg !32
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1130, !dbg !32
  store half %v2400, ptr addrspace(1) %address.local1131, align 2, !dbg !32
  %address.local1133 = or disjoint i64 %address.local947, 114, !dbg !32
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1133, !dbg !32
  store half %v2399, ptr addrspace(1) %address.local1134, align 2, !dbg !32
  %address.local1136 = or i64 %378, 115, !dbg !32
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1136, !dbg !32
  store half %v2398, ptr addrspace(1) %address.local1137, align 2, !dbg !32
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1115, !dbg !32
  store half %v2392, ptr addrspace(1) %address.local1142, align 2, !dbg !32
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1118, !dbg !32
  store half %v2391, ptr addrspace(1) %address.local1144, align 2, !dbg !32
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1121, !dbg !32
  store half %v2390, ptr addrspace(1) %address.local1146, align 2, !dbg !32
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1124, !dbg !32
  store half %v2389, ptr addrspace(1) %address.local1148, align 2, !dbg !32
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1127, !dbg !32
  store half %v2383, ptr addrspace(1) %address.local1150, align 2, !dbg !32
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1130, !dbg !32
  store half %v2382, ptr addrspace(1) %address.local1152, align 2, !dbg !32
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1133, !dbg !32
  store half %v2381, ptr addrspace(1) %address.local1154, align 2, !dbg !32
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1136, !dbg !32
  store half %v2380, ptr addrspace(1) %address.local1156, align 2, !dbg !32
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
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 70, column: 22, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 72, column: 24, scope: !3)
!11 = !DILocation(line: 60, column: 24, scope: !3)
!12 = !DILocation(line: 77, column: 19, scope: !3)
!13 = !DILocation(line: 0, scope: !3)
!14 = !DILocation(line: 71, column: 22, scope: !3)
!15 = !DILocation(line: 73, column: 26, scope: !3)
!16 = !DILocation(line: 80, column: 23, scope: !3)
!17 = !DILocation(line: 101, column: 20, scope: !3)
!18 = !DILocation(line: 81, column: 26, scope: !3)
!19 = !DILocation(line: 88, column: 21, scope: !3)
!20 = !DILocation(line: 89, column: 20, scope: !3)
!21 = !DILocation(line: 90, column: 22, scope: !3)
!22 = !DILocation(line: 91, column: 23, scope: !3)
!23 = !DILocation(line: 92, column: 20, scope: !3)
!24 = !DILocation(line: 103, column: 22, scope: !3)
!25 = !DILocation(line: 93, column: 22, scope: !3)
!26 = !DILocation(line: 94, column: 22, scope: !3)
!27 = !DILocation(line: 96, column: 16, scope: !3)
!28 = !DILocation(line: 95, column: 20, scope: !3)
!29 = !DILocation(line: 102, column: 23, scope: !3)
!30 = !DILocation(line: 107, column: 23, scope: !3)
!31 = !DILocation(line: 108, column: 19, scope: !3)
!32 = !DILocation(line: 109, column: 9, scope: !3)
