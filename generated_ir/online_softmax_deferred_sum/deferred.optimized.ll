; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/online_softmax_deferred_sum/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/local_reduction_pass/compiler-bn128/run-20261026-112711-jne7_i1s/grid-000003/input.mlir"

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
  %address.local495 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local496 = and i64 %address.local495, 1536, !dbg !8
  %address.local497 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local496, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local501 = and i64 %address.local500, 16256, !dbg !8
  %address.local502 = lshr i64 %address.local, 4, !dbg !8
  %address.local503 = and i64 %address.local502, 8, !dbg !8
  %address.local504 = or disjoint i64 %address.local501, %address.local503, !dbg !8
  %v37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %6 = or disjoint i64 %address.local504, %v28, !dbg !8
  %7 = shl nuw nsw i64 %6, 1, !dbg !8
  %v39 = add i64 %7, %v33.idx, !dbg !8
  %dma.buffer.offset1768 = trunc i64 %v39 to i32, !dbg !8
  %8 = insertelement <2 x i64> poison, i64 %v37, i64 0, !dbg !8
  %v44 = bitcast <2 x i64> %8 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1769 = shufflevector <4 x i32> %v44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local506 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local507 = shl nuw nsw i64 %address.local506, 3, !dbg !8
  %address.local508 = and i64 %address.local507, 3584, !dbg !8
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local508, !dbg !8
  %address.local514 = lshr i64 %address.local506, 4, !dbg !8
  %address.local515 = and i64 %address.local514, 24, !dbg !8
  %address.local516 = or disjoint i64 %address.local515, %address.local501, !dbg !8
  %9 = or disjoint i64 %address.local516, %v28, !dbg !8
  %10 = shl nuw nsw i64 %9, 1, !dbg !8
  %v53 = add i64 %10, %v33.idx, !dbg !8
  %dma.buffer.offset1765 = trunc i64 %v53 to i32, !dbg !8
  %v62 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %v63 = sext i32 %v62 to i64, !dbg !7
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local496, !dbg !8
  %.idx = shl nuw nsw i64 %v28, 1, !dbg !8
  %11 = add i64 %.idx, %v33.idx, !dbg !8
  %address.local521.idx = shl nuw nsw i64 %address.local504, 1, !dbg !8
  %12 = or disjoint i64 %address.local521.idx, %11, !dbg !8
  %13 = trunc i64 %12 to i32, !dbg !8
  %dma.buffer.offset1762 = or disjoint i32 %13, 64, !dbg !8
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local508, !dbg !8
  %14 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local497, i64 0, !dbg !8
  %15 = insertelement <4 x ptr addrspace(3)> %14, ptr addrspace(3) %address.local509, i64 1, !dbg !8
  %16 = insertelement <4 x ptr addrspace(3)> %15, ptr addrspace(3) %address.local518, i64 2, !dbg !8
  %17 = insertelement <4 x ptr addrspace(3)> %16, ptr addrspace(3) %address.local522, i64 3, !dbg !8
  %18 = ptrtoint <4 x ptr addrspace(3)> %17 to <4 x i64>, !dbg !8
  %19 = trunc <4 x i64> %18 to <4 x i32>, !dbg !8
  %20 = extractelement <4 x i32> %19, i64 0, !dbg !8
  %v34 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %20), !dbg !8
  %dma.buffer.lds1770 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v34), !dbg !8
  %v47 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1770, i32 %dma.buffer.offset1768, <4 x i32> %dma.buffer.rsrc1769) #9, !dbg !8
  %21 = extractelement <4 x i32> %19, i64 1, !dbg !8
  %v48 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %21), !dbg !8
  %dma.buffer.lds1767 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v48), !dbg !8
  %v61 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1767, i32 %dma.buffer.offset1765, <4 x i32> %dma.buffer.rsrc1769) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %22 = extractelement <4 x i32> %19, i64 2, !dbg !8
  %v68 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %22), !dbg !8
  %dma.buffer.lds1764 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v68), !dbg !8
  %v81 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1764, i32 %dma.buffer.offset1762, <4 x i32> %dma.buffer.rsrc1769) #9, !dbg !8
  %23 = extractelement <4 x i32> %19, i64 3, !dbg !8
  %v82 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %23), !dbg !8
  %address.local525.idx = shl nuw nsw i64 %address.local516, 1, !dbg !8
  %24 = or disjoint i64 %address.local525.idx, %11, !dbg !8
  %25 = trunc i64 %24 to i32, !dbg !8
  %dma.buffer.offset1759 = or disjoint i32 %25, 64, !dbg !8
  %dma.buffer.lds1761 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v82), !dbg !8
  %v95 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1761, i32 %dma.buffer.offset1759, <4 x i32> %dma.buffer.rsrc1769) #9, !dbg !8
  %address.local526 = lshr i32 %address.thread, 2, !dbg !10
  %address.local527 = and i32 %address.local526, 4, !dbg !10
  %address.local528 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local529 = and i32 %address.local528, 384, !dbg !10
  %address.local530 = or disjoint i32 %address.local527, %address.local529, !dbg !10
  %address.local531 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local532 = and i32 %address.local531, 120, !dbg !10
  %address.local533 = or disjoint i32 %address.local530, %address.local532, !dbg !10
  %address.local534 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local535 = and i32 %address.local534, 1024, !dbg !10
  %address.local536 = or disjoint i32 %address.local533, %address.local535, !dbg !10
  %address.local537 = zext nneg i32 %address.local536 to i64, !dbg !10
  %address.local538 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local537, !dbg !10
  %v96 = load <4 x half>, ptr addrspace(3) %address.local538, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local538, i64 4096, !dbg !10
  %v97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %v98 = fmul <4 x half> %v96, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local538, i64 1024, !dbg !10
  %v99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !10
  %v100 = fmul <4 x half> %v97, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local538, i64 5120, !dbg !10
  %v101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !10
  %v102 = fmul <4 x half> %v99, splat (half 1.275630e-01), !dbg !10
  %v103 = fmul <4 x half> %v101, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1756 = or disjoint i32 %13, 128, !dbg !8
  %v119 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1770, i32 %dma.buffer.offset1756, <4 x i32> %dma.buffer.rsrc1769) #9, !dbg !8
  %dma.buffer.offset1753 = or disjoint i32 %25, 128, !dbg !8
  %v133 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1767, i32 %dma.buffer.offset1753, <4 x i32> %dma.buffer.rsrc1769) #9, !dbg !8
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local538, i64 8192, !dbg !10
  %v134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !10
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local538, i64 12288, !dbg !10
  %v135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !10
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local538, i64 9216, !dbg !10
  %v136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !10
  %v137 = fmul <4 x half> %v134, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local538, i64 13312, !dbg !10
  %v138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !10
  %v139 = fmul <4 x half> %v135, splat (half 1.275630e-01), !dbg !10
  %v140 = fmul <4 x half> %v136, splat (half 1.275630e-01), !dbg !10
  %v141 = fmul <4 x half> %v138, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1750 = or disjoint i32 %13, 192, !dbg !8
  %v157 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1764, i32 %dma.buffer.offset1750, <4 x i32> %dma.buffer.rsrc1769) #9, !dbg !8
  %dma.buffer.offset1747 = or disjoint i32 %25, 192, !dbg !8
  %v171 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1761, i32 %dma.buffer.offset1747, <4 x i32> %dma.buffer.rsrc1769) #9, !dbg !8
  %v172 = load <4 x half>, ptr addrspace(3) %address.local538, align 8, !dbg !10
  %v173 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %v174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !10
  %v175 = fmul <4 x half> %v172, splat (half 1.275630e-01), !dbg !10
  %v176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !10
  %v177 = fmul <4 x half> %v173, splat (half 1.275630e-01), !dbg !10
  %v178 = fmul <4 x half> %v174, splat (half 1.275630e-01), !dbg !10
  %v179 = fmul <4 x half> %v176, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %v180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !10
  %v181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !10
  %v182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !10
  %v183 = fmul <4 x half> %v180, splat (half 1.275630e-01), !dbg !10
  %v184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !10
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
  br i1 %v21251, label %v213.lr.ph, label %.v1623_crit_edge, !dbg !11

.v1623_crit_edge:                                 ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre66 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre67 = add nuw nsw i32 %.pre66, 64, !dbg !11
  %.pre68 = xor i32 %.pre66, 16, !dbg !11
  %.pre69 = and i32 %.pre67, 192, !dbg !11
  %.pre70 = xor i32 %.pre66, 32, !dbg !11
  br label %v1623, !dbg !11

v213.lr.ph:                                       ; preds = %4
  %v32 = zext nneg i32 %v31 to i64, !dbg !8
  %v230 = ptrtoint ptr addrspace(1) %1 to i64
  %27 = insertelement <2 x i64> poison, i64 %v230, i64 0
  %v237 = bitcast <2 x i64> %27 to <4 x i32>
  %dma.buffer.rsrc1745 = shufflevector <4 x i32> %v237, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %v609 = ptrtoint ptr addrspace(1) %2 to i64
  %28 = insertelement <2 x i64> poison, i64 %v609, i64 0
  %v616 = bitcast <2 x i64> %28 to <4 x i32>
  %dma.buffer.rsrc1721 = shufflevector <4 x i32> %v616, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
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
  %compare.high1437 = trunc nuw i64 %v746 to i32
  %compare.low.base1441 = trunc i64 %v745 to i32
  %30 = insertelement <4 x i32> poison, i32 %compare.low.base1441, i64 0, !dbg !16
  %31 = shufflevector <4 x i32> %30, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  %32 = insertelement <4 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %33 = shufflevector <4 x i32> %32, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  br label %v213, !dbg !11

v213:                                             ; preds = %v213.lr.ph, %v845
  %v21158 = phi [2 x <8 x float>] [ zeroinitializer, %v213.lr.ph ], [ %v1621, %v845 ]
  %v21057 = phi [2 x <8 x float>] [ zeroinitializer, %v213.lr.ph ], [ %v1617, %v845 ]
  %v20956 = phi [2 x <8 x float>] [ zeroinitializer, %v213.lr.ph ], [ %v1613, %v845 ]
  %v20855 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %v213.lr.ph ], [ %v1609, %v845 ]
  %v20754 = phi [2 x <1 x float>] [ zeroinitializer, %v213.lr.ph ], [ %v1213, %v845 ]
  %v20653 = phi [2 x <8 x float>] [ zeroinitializer, %v213.lr.ph ], [ %v1607, %v845 ]
  %v20552 = phi i64 [ 0, %v213.lr.ph ], [ %v1622, %v845 ]
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
  %address.thread589 = tail call i32 asm sideeffect "", "=v,0"(i32 %v31) #8, !dbg !14, !range !9, !frisk.register_lifetimes !5
  %address.local590 = zext nneg i32 %address.thread589 to i64, !dbg !8
  %address.local591 = shl nuw nsw i64 %address.local590, 3, !dbg !8
  %address.local592 = and i64 %address.local591, 1536, !dbg !8
  %address.local593 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local592, !dbg !8
  %address.local596 = lshr i64 %address.local590, 4, !dbg !14
  %address.local597 = and i64 %address.local596, 8, !dbg !14
  %address.local598 = and i64 %address.local590, 123, !dbg !14
  %address.local599 = lshr i64 %address.local590, 2, !dbg !14
  %address.local600 = and i64 %address.local599, 4, !dbg !14
  %address.local601 = add nuw nsw i64 %address.local600, %address.local590, !dbg !14
  %address.local602 = and i64 %address.local601, 4, !dbg !14
  %address.local603 = or disjoint i64 %address.local602, %address.local598, !dbg !14
  %address.local604 = shl nuw nsw i64 %address.local603, 7, !dbg !14
  %address.local605 = or disjoint i64 %address.local604, %address.local597, !dbg !14
  %v224.idx6 = or disjoint i64 %v220, %v218, !dbg !14
  %34 = add i64 %v224.idx6, %v223, !dbg !14
  %35 = add i64 %address.local605, %34, !dbg !14
  %.tr = trunc i64 %35 to i32, !dbg !14
  %dma.buffer.offset1744 = shl i32 %.tr, 1, !dbg !14
  %address.local607 = or disjoint i64 %address.local590, 256, !dbg !8
  %address.local608 = shl nuw nsw i64 %address.local607, 3, !dbg !8
  %address.local609 = and i64 %address.local608, 3584, !dbg !8
  %address.local610 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local609, !dbg !8
  %address.local613 = lshr i64 %address.local607, 4, !dbg !14
  %address.local614 = and i64 %address.local613, 24, !dbg !14
  %address.local622 = or disjoint i64 %address.local604, %address.local614, !dbg !14
  %36 = add i64 %address.local622, %34, !dbg !14
  %.tr8 = trunc i64 %36 to i32, !dbg !14
  %dma.buffer.offset1741 = shl i32 %.tr8, 1, !dbg !14
  %v255 = add nuw nsw i64 %v20552, 1024, !dbg !14
  %v2599 = lshr i64 %v255, 12, !dbg !14
  %v261 = add nuw nsw i64 %v2599, %v16, !dbg !14
  %v262 = add nsw i64 %v217, %v195, !dbg !14
  %v272 = shl i64 %v261, 19, !dbg !14
  %v273 = and i64 %v272, 16252928, !dbg !14
  %address.local624 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local592, !dbg !8
  %v271 = add i64 %v223, %v217, !dbg !14
  %v274.idx10 = add i64 %v271, %v195, !dbg !14
  %37 = add i64 %v274.idx10, %v273, !dbg !14
  %address.local628 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local609, !dbg !8
  %38 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local593, i64 0, !dbg !8
  %39 = insertelement <4 x ptr addrspace(3)> %38, ptr addrspace(3) %address.local610, i64 1, !dbg !8
  %40 = insertelement <4 x ptr addrspace(3)> %39, ptr addrspace(3) %address.local624, i64 2, !dbg !8
  %41 = insertelement <4 x ptr addrspace(3)> %40, ptr addrspace(3) %address.local628, i64 3, !dbg !8
  %42 = ptrtoint <4 x ptr addrspace(3)> %41 to <4 x i64>, !dbg !8
  %43 = trunc <4 x i64> %42 to <4 x i32>, !dbg !8
  %44 = extractelement <4 x i32> %43, i64 0, !dbg !14
  %v227 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44), !dbg !14
  %dma.buffer.lds1746 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v227), !dbg !14
  %v240 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1746, i32 %dma.buffer.offset1744, <4 x i32> %dma.buffer.rsrc1745) #9, !dbg !14
  %45 = extractelement <4 x i32> %43, i64 1, !dbg !14
  %v241 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %45), !dbg !14
  %dma.buffer.lds1743 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v241), !dbg !14
  %v254 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1743, i32 %dma.buffer.offset1741, <4 x i32> %dma.buffer.rsrc1745) #9, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %46 = extractelement <4 x i32> %43, i64 2, !dbg !14
  %v279 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !14
  %dma.buffer.lds1740 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v279), !dbg !14
  %47 = extractelement <4 x i32> %43, i64 3, !dbg !14
  %address.local636 = shl nuw nsw i64 %address.local590, 1, !dbg !15
  %address.local637 = and i64 %address.local636, 24, !dbg !15
  %address.local639 = shl i64 %address.local590, 62, !dbg !15
  %address.local640 = ashr exact i64 %address.local639, 57, !dbg !15
  %address.local641 = and i64 %address.local640, -64, !dbg !15
  %address.local644 = and i64 %address.local599, 12, !dbg !10
  %48 = and i64 %address.local590, 32, !dbg !10
  %.not14 = icmp eq i64 %48, 0, !dbg !10
  %address.local648 = select i1 %.not14, i64 0, i64 1016, !dbg !10
  %.idx15 = mul nuw nsw i64 %address.local590, 320, !dbg !15
  %49 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx15, !dbg !15
  %.idx16 = mul nsw i64 %address.local599, -1280, !dbg !15
  %50 = getelementptr i8, ptr addrspace(3) %49, i64 %.idx16, !dbg !15
  %51 = getelementptr [2 x i8], ptr addrspace(3) %50, i64 %address.local637, !dbg !15
  %52 = getelementptr [2 x i8], ptr addrspace(3) %51, i64 %address.local641, !dbg !15
  %53 = getelementptr [2 x i8], ptr addrspace(3) %52, i64 %address.local644, !dbg !15
  %address.local650 = getelementptr [2 x i8], ptr addrspace(3) %53, i64 %address.local648, !dbg !15
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4096, !dbg !15
  %address.local661 = shl nuw nsw i32 %address.thread589, 1, !dbg !15
  %address.local662 = and i32 %address.local661, 24, !dbg !15
  %address.local663 = shl nuw nsw i32 %address.thread589, 8, !dbg !15
  %address.local664 = and i32 %address.local663, 768, !dbg !15
  %address.local665 = or disjoint i32 %address.local662, %address.local664, !dbg !15
  %address.local666 = shl nuw nsw i32 %address.thread589, 5, !dbg !15
  %address.local667 = and i32 %address.local666, 96, !dbg !15
  %address.local668 = or disjoint i32 %address.local665, %address.local667, !dbg !15
  %address.local669 = zext nneg i32 %address.local668 to i64, !dbg !15
  %address.local670 = lshr i32 %address.thread589, 1, !dbg !15
  %address.local671 = and i32 %address.local670, 1, !dbg !15
  %address.local672 = zext nneg i32 %address.local671 to i64, !dbg !15
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local650, i64 128, !dbg !15
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4224, !dbg !15
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1024, !dbg !15
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5120, !dbg !15
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1152, !dbg !15
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5248, !dbg !15
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
  %65 = add i64 %address.local605, %64, !dbg !14
  %.tr11 = trunc i64 %65 to i32, !dbg !14
  %66 = shl i32 %.tr11, 1, !dbg !14
  %dma.buffer.offset1738 = add i32 %66, 64, !dbg !14
  %v292 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1740, i32 %dma.buffer.offset1738, <4 x i32> %dma.buffer.rsrc1745) #9, !dbg !14
  %v293 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47), !dbg !14
  %67 = add i64 %address.local622, %64, !dbg !14
  %.tr13 = trunc i64 %67 to i32, !dbg !14
  %68 = shl i32 %.tr13, 1, !dbg !14
  %dma.buffer.offset1735 = add i32 %68, 64, !dbg !14
  %dma.buffer.lds1737 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v293), !dbg !14
  %v306 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1737, i32 %dma.buffer.offset1735, <4 x i32> %dma.buffer.rsrc1745) #9, !dbg !14
  %v307 = load <4 x half>, ptr addrspace(3) %address.local650, align 8, !dbg !15
  %v308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v307, <4 x float> zeroinitializer) #8, !dbg !15
  %v309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %v203.sext) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local673 = add nsw i64 %address.base, %address.local672, !dbg !15
  %address.local674 = shl i64 %address.local673, 6, !dbg !15
  %address.local675 = sub i64 %address.local669, %address.local674, !dbg !15
  %69 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local675, !dbg !15
  %70 = getelementptr [2 x i8], ptr addrspace(3) %69, i64 %address.local644, !dbg !15
  %71 = getelementptr [2 x i8], ptr addrspace(3) %70, i64 %address.local648, !dbg !15
  %address.local679 = getelementptr i8, ptr addrspace(3) %71, i64 64, !dbg !15
  %v310 = load <4 x half>, ptr addrspace(3) %address.local679, align 8, !dbg !15
  %v311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v309, <4 x float> %v308) #8, !dbg !15
  %v312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v310, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %71, i64 4160, !dbg !15
  %v313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !15
  %v314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !15
  %v315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v313, <4 x float> %v312) #8, !dbg !15
  %v316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v314, <4 x float> zeroinitializer) #8, !dbg !15
  %v317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !15
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %71, i64 192, !dbg !15
  %v318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !15
  %v319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v317, <4 x float> %v316) #8, !dbg !15
  %v320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v318, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %71, i64 4288, !dbg !15
  %v321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !15
  %v322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !15
  %v323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v321, <4 x float> %v320) #8, !dbg !15
  %v324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v322, <4 x float> zeroinitializer) #8, !dbg !15
  %v325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !15
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %71, i64 1088, !dbg !15
  %v326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !15
  %v327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v325, <4 x float> %v324) #8, !dbg !15
  %v328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v326, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %71, i64 5184, !dbg !15
  %v329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !15
  %v330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !15
  %v331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v329, <4 x float> %v328) #8, !dbg !15
  %v332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v330, <4 x float> zeroinitializer) #8, !dbg !15
  %v333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !15
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %71, i64 1216, !dbg !15
  %v334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !15
  %v335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v333, <4 x float> %v332) #8, !dbg !15
  %v336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v334, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %71, i64 5312, !dbg !15
  %v337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !15
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
  %77 = add i64 %address.local605, %76, !dbg !14
  %.tr19 = trunc i64 %77 to i32, !dbg !14
  %78 = shl i32 %.tr19, 1, !dbg !14
  %dma.buffer.offset1732 = add i32 %78, 128, !dbg !14
  %v407 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1746, i32 %dma.buffer.offset1732, <4 x i32> %dma.buffer.rsrc1745) #9, !dbg !14
  %79 = add i64 %address.local622, %76, !dbg !14
  %.tr21 = trunc i64 %79 to i32, !dbg !14
  %80 = shl i32 %.tr21, 1, !dbg !14
  %dma.buffer.offset1729 = add i32 %80, 128, !dbg !14
  %v421 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1743, i32 %dma.buffer.offset1729, <4 x i32> %dma.buffer.rsrc1745) #9, !dbg !14
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8192, !dbg !15
  %v422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !15
  %v423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v422, <4 x float> %v311) #8, !dbg !15
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12288, !dbg !15
  %v424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !15
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %71, i64 8256, !dbg !15
  %v425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !15
  %v426 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v424, <4 x float> %v423) #8, !dbg !15
  %v427 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v425, <4 x float> %v315) #8, !dbg !15
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %71, i64 12352, !dbg !15
  %v428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !15
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8320, !dbg !15
  %v429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !15
  %v430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v428, <4 x float> %v427) #8, !dbg !15
  %v431 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v429, <4 x float> %v319) #8, !dbg !15
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12416, !dbg !15
  %v432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !15
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %71, i64 8384, !dbg !15
  %v433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !15
  %v434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v432, <4 x float> %v431) #8, !dbg !15
  %v435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v433, <4 x float> %v323) #8, !dbg !15
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %71, i64 12480, !dbg !15
  %v436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !15
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9216, !dbg !15
  %v437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !15
  %v438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v436, <4 x float> %v435) #8, !dbg !15
  %v439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v437, <4 x float> %v327) #8, !dbg !15
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13312, !dbg !15
  %v440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !15
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %71, i64 9280, !dbg !15
  %v441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !15
  %v442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v440, <4 x float> %v439) #8, !dbg !15
  %v443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v441, <4 x float> %v331) #8, !dbg !15
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %71, i64 13376, !dbg !15
  %v444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !15
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9344, !dbg !15
  %v445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !15
  %v446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v444, <4 x float> %v443) #8, !dbg !15
  %v447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v445, <4 x float> %v335) #8, !dbg !15
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13440, !dbg !15
  %v448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !15
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %71, i64 9408, !dbg !15
  %v449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !15
  %v450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v448, <4 x float> %v447) #8, !dbg !15
  %v451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v449, <4 x float> %v339) #8, !dbg !15
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %71, i64 13504, !dbg !15
  %v452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !15
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
  %84 = add i64 %address.local605, %83, !dbg !14
  %.tr24 = trunc i64 %84 to i32, !dbg !14
  %85 = shl i32 %.tr24, 1, !dbg !14
  %dma.buffer.offset1726 = add i32 %85, 192, !dbg !14
  %v522 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1740, i32 %dma.buffer.offset1726, <4 x i32> %dma.buffer.rsrc1745) #9, !dbg !14
  %86 = add i64 %address.local622, %83, !dbg !14
  %.tr26 = trunc i64 %86 to i32, !dbg !14
  %87 = shl i32 %.tr26, 1, !dbg !14
  %dma.buffer.offset1723 = add i32 %87, 192, !dbg !14
  %v536 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1737, i32 %dma.buffer.offset1723, <4 x i32> %dma.buffer.rsrc1745) #9, !dbg !14
  %v537 = load <4 x half>, ptr addrspace(3) %address.local650, align 8, !dbg !15
  %v538 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v537, <4 x float> %v426) #8, !dbg !15
  %v539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !15
  %v540 = load <4 x half>, ptr addrspace(3) %address.local679, align 8, !dbg !15
  %v541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v539, <4 x float> %v538) #8, !dbg !15
  %v542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v540, <4 x float> %v430) #8, !dbg !15
  %v543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !15
  %v544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !15
  %v545 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v543, <4 x float> %v542) #8, !dbg !15
  %v546 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v544, <4 x float> %v434) #8, !dbg !15
  %v547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !15
  %v548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !15
  %v549 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v547, <4 x float> %v546) #8, !dbg !15
  %v550 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v548, <4 x float> %v438) #8, !dbg !15
  %v551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !15
  %v552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !15
  %v553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v551, <4 x float> %v550) #8, !dbg !15
  %v554 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v552, <4 x float> %v442) #8, !dbg !15
  %v555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !15
  %v556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !15
  %v557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v555, <4 x float> %v554) #8, !dbg !15
  %v558 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v556, <4 x float> %v446) #8, !dbg !15
  %v559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !15
  %v560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !15
  %v561 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v559, <4 x float> %v558) #8, !dbg !15
  %v562 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v560, <4 x float> %v450) #8, !dbg !15
  %v563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !15
  %v564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !15
  %v565 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v563, <4 x float> %v562) #8, !dbg !15
  %v566 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v564, <4 x float> %v454) #8, !dbg !15
  %v567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !15
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
  %address.local1008 = shl nuw nsw i64 %address.local590, 6, !dbg !17
  %address.local1009 = and i64 %address.local1008, 16128, !dbg !17
  %address.local1011 = and i64 %address.local591, 24, !dbg !17
  %v604.idx28 = add i64 %v603, %v223, !dbg !17
  %88 = or disjoint i64 %address.local1011, %v604.idx28, !dbg !17
  %89 = shl i64 %88, 1, !dbg !17
  %v611 = or disjoint i64 %89, %address.local1009, !dbg !17
  %dma.buffer.offset1720 = trunc i64 %v611 to i32, !dbg !17
  %v619 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1746, i32 %dma.buffer.offset1720, <4 x i32> %dma.buffer.rsrc1721) #9, !dbg !17
  %address.local1014 = shl nuw nsw i64 %address.local607, 6, !dbg !17
  %address.local1015 = and i64 %address.local1014, 32512, !dbg !17
  %v625 = or disjoint i64 %89, %address.local1015, !dbg !17
  %dma.buffer.offset1717 = trunc i64 %v625 to i32, !dbg !17
  %v633 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1743, i32 %dma.buffer.offset1717, <4 x i32> %dma.buffer.rsrc1721) #9, !dbg !17
  %v634 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !15
  %v635 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v634, <4 x float> %v541) #8, !dbg !15
  %v636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !15
  %v637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !15
  %v638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v636, <4 x float> %v635) #8, !dbg !15
  %v639 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v637, <4 x float> %v545) #8, !dbg !15
  %v640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !15
  %v641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !15
  %v642 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v640, <4 x float> %v639) #8, !dbg !15
  %v643 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v641, <4 x float> %v549) #8, !dbg !15
  %v644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !15
  %v645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !15
  %v646 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v644, <4 x float> %v643) #8, !dbg !15
  %v647 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v645, <4 x float> %v553) #8, !dbg !15
  %v648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !15
  %v649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !15
  %v650 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v648, <4 x float> %v647) #8, !dbg !15
  %v651 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v649, <4 x float> %v557) #8, !dbg !15
  %v652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !15
  %v653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !15
  %v654 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v652, <4 x float> %v651) #8, !dbg !15
  %v655 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v653, <4 x float> %v561) #8, !dbg !15
  %v656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !15
  %v657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !15
  %v658 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v656, <4 x float> %v655) #8, !dbg !15
  %v659 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v657, <4 x float> %v565) #8, !dbg !15
  %v660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !15
  %v661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !15
  %v662 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v660, <4 x float> %v659) #8, !dbg !15
  %v663 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v661, <4 x float> %v569) #8, !dbg !15
  %v664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !15
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
  br i1 %v700, label %v845, label %v702, !dbg !12

v702:                                             ; preds = %v213
  %v709 = or disjoint i64 %v214, %v708, !dbg !12
  %v712 = lshr i64 %v20552, 25, !dbg !16
  %compare.high1313 = trunc nuw i64 %v712 to i32, !dbg !16
  %compare.low.base1314 = trunc i64 %v709 to i32, !dbg !16
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 1) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1317 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 2) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1321 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 3) #8, !dbg !16, !frisk.register_lifetimes !5
  %90 = insertelement <4 x i32> poison, i32 %compare.low.base1314, i64 0, !dbg !16
  %91 = insertelement <4 x i32> %90, i32 %compare.local.base, i64 1, !dbg !16
  %92 = insertelement <4 x i32> %91, i32 %compare.local.base1317, i64 2, !dbg !16
  %93 = insertelement <4 x i32> %92, i32 %compare.local.base1321, i64 3, !dbg !16
  %94 = or <4 x i32> %93, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %95 = icmp ugt <4 x i32> %33, %94, !dbg !16
  %compare.local.base1325 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 4) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1329 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 5) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1333 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 6) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1337 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 7) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1341 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 8) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1345 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 9) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1349 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 10) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1353 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 11) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1357 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 12) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1361 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 13) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1365 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 14) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1369 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 15) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1373 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 16) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1377 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 17) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1381 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 18) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1385 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 19) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1389 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 20) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1393 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 21) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1397 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 22) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1401 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 23) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1405 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 24) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1409 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 25) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1413 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 26) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1417 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 27) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1421 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 28) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1425 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 29) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1429 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 30) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1433 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 31) #8, !dbg !16, !frisk.register_lifetimes !5
  %96 = insertelement <4 x i32> poison, i32 %compare.local.base1405, i64 0, !dbg !16
  %97 = insertelement <4 x i32> %96, i32 %compare.local.base1409, i64 1, !dbg !16
  %98 = insertelement <4 x i32> %97, i32 %compare.local.base1413, i64 2, !dbg !16
  %99 = insertelement <4 x i32> %98, i32 %compare.local.base1417, i64 3, !dbg !16
  %100 = or <4 x i32> %99, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %101 = icmp ult <4 x i32> %100, %33, !dbg !16
  %102 = insertelement <4 x i32> poison, i32 %compare.local.base1389, i64 0, !dbg !16
  %103 = insertelement <4 x i32> %102, i32 %compare.local.base1393, i64 1, !dbg !16
  %104 = insertelement <4 x i32> %103, i32 %compare.local.base1397, i64 2, !dbg !16
  %105 = insertelement <4 x i32> %104, i32 %compare.local.base1401, i64 3, !dbg !16
  %106 = or <4 x i32> %105, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %107 = icmp ult <4 x i32> %106, %33, !dbg !16
  %108 = insertelement <4 x i32> poison, i32 %compare.local.base1373, i64 0, !dbg !16
  %109 = insertelement <4 x i32> %108, i32 %compare.local.base1377, i64 1, !dbg !16
  %110 = insertelement <4 x i32> %109, i32 %compare.local.base1381, i64 2, !dbg !16
  %111 = insertelement <4 x i32> %110, i32 %compare.local.base1385, i64 3, !dbg !16
  %112 = or <4 x i32> %111, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %113 = icmp ult <4 x i32> %112, %33, !dbg !16
  %114 = insertelement <4 x i32> poison, i32 %compare.local.base1357, i64 0, !dbg !16
  %115 = insertelement <4 x i32> %114, i32 %compare.local.base1361, i64 1, !dbg !16
  %116 = insertelement <4 x i32> %115, i32 %compare.local.base1365, i64 2, !dbg !16
  %117 = insertelement <4 x i32> %116, i32 %compare.local.base1369, i64 3, !dbg !16
  %118 = or <4 x i32> %117, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %119 = icmp ult <4 x i32> %118, %33, !dbg !16
  %120 = insertelement <4 x i32> poison, i32 %compare.local.base1341, i64 0, !dbg !16
  %121 = insertelement <4 x i32> %120, i32 %compare.local.base1345, i64 1, !dbg !16
  %122 = insertelement <4 x i32> %121, i32 %compare.local.base1349, i64 2, !dbg !16
  %123 = insertelement <4 x i32> %122, i32 %compare.local.base1353, i64 3, !dbg !16
  %124 = or <4 x i32> %123, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %125 = icmp ult <4 x i32> %124, %33, !dbg !16
  %126 = insertelement <4 x i32> poison, i32 %compare.local.base1325, i64 0, !dbg !16
  %127 = insertelement <4 x i32> %126, i32 %compare.local.base1329, i64 1, !dbg !16
  %128 = insertelement <4 x i32> %127, i32 %compare.local.base1333, i64 2, !dbg !16
  %129 = insertelement <4 x i32> %128, i32 %compare.local.base1337, i64 3, !dbg !16
  %130 = or <4 x i32> %129, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %131 = icmp ult <4 x i32> %130, %33, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1313, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1313, !dbg !16
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
  %150 = insertelement <4 x i32> poison, i32 %compare.local.base1421, i64 0, !dbg !16
  %151 = insertelement <4 x i32> %150, i32 %compare.local.base1425, i64 1, !dbg !16
  %152 = insertelement <4 x i32> %151, i32 %compare.local.base1429, i64 2, !dbg !16
  %153 = insertelement <4 x i32> %152, i32 %compare.local.base1433, i64 3, !dbg !16
  %154 = or <4 x i32> %153, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %155 = icmp ult <4 x i32> %154, %33, !dbg !16
  %156 = select <4 x i1> %133, <4 x i1> %155, <4 x i1> %135, !dbg !16
  %157 = select <4 x i1> %156, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %compare.local.base1445 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 32) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1449 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 33) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1453 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 34) #8, !dbg !16, !frisk.register_lifetimes !5
  %158 = insertelement <4 x i32> %90, i32 %compare.local.base1445, i64 1, !dbg !16
  %159 = insertelement <4 x i32> %158, i32 %compare.local.base1449, i64 2, !dbg !16
  %160 = insertelement <4 x i32> %159, i32 %compare.local.base1453, i64 3, !dbg !16
  %161 = or <4 x i32> %160, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %162 = icmp ugt <4 x i32> %31, %161, !dbg !16
  %compare.local.base1457 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 35) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1461 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 36) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1465 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 37) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1469 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 38) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1473 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 39) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1477 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 40) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1481 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 41) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1485 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 42) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1489 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 43) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1493 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 44) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1497 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 45) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1501 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 46) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1505 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 47) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1509 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 48) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1513 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 49) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1517 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 50) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1521 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 51) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1525 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 52) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1529 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 53) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1533 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 54) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1537 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 55) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1541 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 56) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1545 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 57) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1549 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 58) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1553 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 59) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1557 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 60) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1561 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 61) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1565 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 62) #8, !dbg !16, !frisk.register_lifetimes !5
  %163 = insertelement <4 x i32> poison, i32 %compare.local.base1537, i64 0, !dbg !16
  %164 = insertelement <4 x i32> %163, i32 %compare.local.base1541, i64 1, !dbg !16
  %165 = insertelement <4 x i32> %164, i32 %compare.local.base1545, i64 2, !dbg !16
  %166 = insertelement <4 x i32> %165, i32 %compare.local.base1549, i64 3, !dbg !16
  %167 = or <4 x i32> %166, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %168 = icmp ult <4 x i32> %167, %31, !dbg !16
  %169 = insertelement <4 x i32> poison, i32 %compare.local.base1521, i64 0, !dbg !16
  %170 = insertelement <4 x i32> %169, i32 %compare.local.base1525, i64 1, !dbg !16
  %171 = insertelement <4 x i32> %170, i32 %compare.local.base1529, i64 2, !dbg !16
  %172 = insertelement <4 x i32> %171, i32 %compare.local.base1533, i64 3, !dbg !16
  %173 = or <4 x i32> %172, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %174 = icmp ult <4 x i32> %173, %31, !dbg !16
  %175 = insertelement <4 x i32> poison, i32 %compare.local.base1505, i64 0, !dbg !16
  %176 = insertelement <4 x i32> %175, i32 %compare.local.base1509, i64 1, !dbg !16
  %177 = insertelement <4 x i32> %176, i32 %compare.local.base1513, i64 2, !dbg !16
  %178 = insertelement <4 x i32> %177, i32 %compare.local.base1517, i64 3, !dbg !16
  %179 = or <4 x i32> %178, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %180 = icmp ult <4 x i32> %179, %31, !dbg !16
  %181 = insertelement <4 x i32> poison, i32 %compare.local.base1489, i64 0, !dbg !16
  %182 = insertelement <4 x i32> %181, i32 %compare.local.base1493, i64 1, !dbg !16
  %183 = insertelement <4 x i32> %182, i32 %compare.local.base1497, i64 2, !dbg !16
  %184 = insertelement <4 x i32> %183, i32 %compare.local.base1501, i64 3, !dbg !16
  %185 = or <4 x i32> %184, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %186 = icmp ult <4 x i32> %185, %31, !dbg !16
  %187 = insertelement <4 x i32> poison, i32 %compare.local.base1473, i64 0, !dbg !16
  %188 = insertelement <4 x i32> %187, i32 %compare.local.base1477, i64 1, !dbg !16
  %189 = insertelement <4 x i32> %188, i32 %compare.local.base1481, i64 2, !dbg !16
  %190 = insertelement <4 x i32> %189, i32 %compare.local.base1485, i64 3, !dbg !16
  %191 = or <4 x i32> %190, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %192 = icmp ult <4 x i32> %191, %31, !dbg !16
  %193 = insertelement <4 x i32> poison, i32 %compare.local.base1457, i64 0, !dbg !16
  %194 = insertelement <4 x i32> %193, i32 %compare.local.base1461, i64 1, !dbg !16
  %195 = insertelement <4 x i32> %194, i32 %compare.local.base1465, i64 2, !dbg !16
  %196 = insertelement <4 x i32> %195, i32 %compare.local.base1469, i64 3, !dbg !16
  %197 = or <4 x i32> %196, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %198 = icmp ult <4 x i32> %197, %31, !dbg !16
  %compare.high.equal1439 = icmp eq i32 %compare.high1437, %compare.high1313, !dbg !16
  %compare.high.order1440 = icmp ugt i32 %compare.high1437, %compare.high1313, !dbg !16
  %199 = insertelement <4 x i1> poison, i1 %compare.high.equal1439, i64 0, !dbg !16
  %200 = shufflevector <4 x i1> %199, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %201 = insertelement <4 x i1> poison, i1 %compare.high.order1440, i64 0, !dbg !16
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
  %217 = insertelement <4 x i32> poison, i32 %compare.local.base1553, i64 0, !dbg !16
  %218 = insertelement <4 x i32> %217, i32 %compare.local.base1557, i64 1, !dbg !16
  %219 = insertelement <4 x i32> %218, i32 %compare.local.base1561, i64 2, !dbg !16
  %220 = insertelement <4 x i32> %219, i32 %compare.local.base1565, i64 3, !dbg !16
  %221 = or <4 x i32> %220, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %222 = icmp ult <4 x i32> %221, %31, !dbg !16
  %223 = select <4 x i1> %200, <4 x i1> %222, <4 x i1> %202, !dbg !16
  %224 = select <4 x i1> %223, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %.masked127 = fadd <4 x float> %v638, %137, !dbg !19
  %.masked129 = fadd <4 x float> %v642, %139, !dbg !19
  %.masked131 = fadd <4 x float> %v646, %141, !dbg !19
  %.masked133 = fadd <4 x float> %v650, %143, !dbg !19
  %.masked135 = fadd <4 x float> %v654, %145, !dbg !19
  %.masked137 = fadd <4 x float> %v658, %147, !dbg !19
  %.masked139 = fadd <4 x float> %v662, %149, !dbg !19
  %.masked141 = fadd <4 x float> %v666, %157, !dbg !19
  %.masked144 = fadd <4 x float> %v670, %204, !dbg !19
  %.masked146 = fadd <4 x float> %v674, %206, !dbg !19
  %.masked148 = fadd <4 x float> %v678, %208, !dbg !19
  %.masked150 = fadd <4 x float> %v682, %210, !dbg !19
  %.masked152 = fadd <4 x float> %v686, %212, !dbg !19
  %.masked154 = fadd <4 x float> %v690, %214, !dbg !19
  %.masked156 = fadd <4 x float> %v694, %216, !dbg !19
  %.masked158 = fadd <4 x float> %v699, %224, !dbg !19
  br label %v845, !dbg !12

v845:                                             ; preds = %v702, %v213
  %v829 = phi <4 x float> [ %.masked127, %v702 ], [ %v638, %v213 ], !dbg !19
  %v830 = phi <4 x float> [ %.masked129, %v702 ], [ %v642, %v213 ], !dbg !19
  %v831 = phi <4 x float> [ %.masked131, %v702 ], [ %v646, %v213 ], !dbg !19
  %v832 = phi <4 x float> [ %.masked133, %v702 ], [ %v650, %v213 ], !dbg !19
  %v833 = phi <4 x float> [ %.masked135, %v702 ], [ %v654, %v213 ], !dbg !19
  %v834 = phi <4 x float> [ %.masked137, %v702 ], [ %v658, %v213 ], !dbg !19
  %v835 = phi <4 x float> [ %.masked139, %v702 ], [ %v662, %v213 ], !dbg !19
  %v836 = phi <4 x float> [ %.masked141, %v702 ], [ %v666, %v213 ], !dbg !19
  %v837 = phi <4 x float> [ %.masked144, %v702 ], [ %v670, %v213 ], !dbg !19
  %v838 = phi <4 x float> [ %.masked146, %v702 ], [ %v674, %v213 ], !dbg !19
  %v839 = phi <4 x float> [ %.masked148, %v702 ], [ %v678, %v213 ], !dbg !19
  %v840 = phi <4 x float> [ %.masked150, %v702 ], [ %v682, %v213 ], !dbg !19
  %v841 = phi <4 x float> [ %.masked152, %v702 ], [ %v686, %v213 ], !dbg !19
  %v842 = phi <4 x float> [ %.masked154, %v702 ], [ %v690, %v213 ], !dbg !19
  %v843 = phi <4 x float> [ %.masked156, %v702 ], [ %v694, %v213 ], !dbg !19
  %v844 = phi <4 x float> [ %.masked158, %v702 ], [ %v699, %v213 ], !dbg !19
  %v846 = extractelement <4 x float> %v829, i64 0, !dbg !19
  %v847 = extractelement <4 x float> %v829, i64 1, !dbg !19
  %v848 = extractelement <4 x float> %v829, i64 2, !dbg !19
  %v849 = extractelement <4 x float> %v829, i64 3, !dbg !19
  %v850 = extractelement <4 x float> %v830, i64 0, !dbg !19
  %v851 = extractelement <4 x float> %v830, i64 1, !dbg !19
  %v852 = extractelement <4 x float> %v830, i64 2, !dbg !19
  %v853 = extractelement <4 x float> %v830, i64 3, !dbg !19
  %v854 = extractelement <4 x float> %v831, i64 0, !dbg !19
  %v855 = extractelement <4 x float> %v831, i64 1, !dbg !19
  %v856 = extractelement <4 x float> %v831, i64 2, !dbg !19
  %v857 = extractelement <4 x float> %v831, i64 3, !dbg !19
  %v858 = extractelement <4 x float> %v832, i64 0, !dbg !19
  %v859 = extractelement <4 x float> %v832, i64 1, !dbg !19
  %v860 = extractelement <4 x float> %v832, i64 2, !dbg !19
  %v861 = extractelement <4 x float> %v832, i64 3, !dbg !19
  %v862 = extractelement <4 x float> %v833, i64 0, !dbg !19
  %v863 = extractelement <4 x float> %v833, i64 1, !dbg !19
  %v864 = extractelement <4 x float> %v833, i64 2, !dbg !19
  %v865 = extractelement <4 x float> %v833, i64 3, !dbg !19
  %v866 = extractelement <4 x float> %v834, i64 0, !dbg !19
  %v867 = extractelement <4 x float> %v834, i64 1, !dbg !19
  %v868 = extractelement <4 x float> %v834, i64 2, !dbg !19
  %v869 = extractelement <4 x float> %v834, i64 3, !dbg !19
  %v870 = extractelement <4 x float> %v835, i64 0, !dbg !19
  %v871 = extractelement <4 x float> %v835, i64 1, !dbg !19
  %v872 = extractelement <4 x float> %v835, i64 2, !dbg !19
  %v873 = extractelement <4 x float> %v835, i64 3, !dbg !19
  %v874 = extractelement <4 x float> %v836, i64 0, !dbg !19
  %v875 = extractelement <4 x float> %v836, i64 1, !dbg !19
  %v876 = extractelement <4 x float> %v836, i64 2, !dbg !19
  %v877 = extractelement <4 x float> %v836, i64 3, !dbg !19
  %v878 = extractelement <4 x float> %v837, i64 0, !dbg !19
  %v879 = extractelement <4 x float> %v837, i64 1, !dbg !19
  %v880 = extractelement <4 x float> %v837, i64 2, !dbg !19
  %v881 = extractelement <4 x float> %v837, i64 3, !dbg !19
  %v882 = extractelement <4 x float> %v838, i64 0, !dbg !19
  %v883 = extractelement <4 x float> %v838, i64 1, !dbg !19
  %v884 = extractelement <4 x float> %v838, i64 2, !dbg !19
  %v885 = extractelement <4 x float> %v838, i64 3, !dbg !19
  %v886 = extractelement <4 x float> %v839, i64 0, !dbg !19
  %v887 = extractelement <4 x float> %v839, i64 1, !dbg !19
  %v888 = extractelement <4 x float> %v839, i64 2, !dbg !19
  %v889 = extractelement <4 x float> %v839, i64 3, !dbg !19
  %v890 = extractelement <4 x float> %v840, i64 0, !dbg !19
  %v891 = extractelement <4 x float> %v840, i64 1, !dbg !19
  %v892 = extractelement <4 x float> %v840, i64 2, !dbg !19
  %v893 = extractelement <4 x float> %v840, i64 3, !dbg !19
  %v894 = extractelement <4 x float> %v841, i64 0, !dbg !19
  %v895 = extractelement <4 x float> %v841, i64 1, !dbg !19
  %v896 = extractelement <4 x float> %v841, i64 2, !dbg !19
  %v897 = extractelement <4 x float> %v841, i64 3, !dbg !19
  %v898 = extractelement <4 x float> %v842, i64 0, !dbg !19
  %v899 = extractelement <4 x float> %v842, i64 1, !dbg !19
  %v900 = extractelement <4 x float> %v842, i64 2, !dbg !19
  %v901 = extractelement <4 x float> %v842, i64 3, !dbg !19
  %v902 = extractelement <4 x float> %v843, i64 0, !dbg !19
  %v903 = extractelement <4 x float> %v843, i64 1, !dbg !19
  %v904 = extractelement <4 x float> %v843, i64 2, !dbg !19
  %v905 = extractelement <4 x float> %v843, i64 3, !dbg !19
  %v906 = extractelement <4 x float> %v844, i64 0, !dbg !19
  %v907 = extractelement <4 x float> %v844, i64 1, !dbg !19
  %v908 = extractelement <4 x float> %v844, i64 2, !dbg !19
  %v909 = extractelement <4 x float> %v844, i64 3, !dbg !19
  %v910 = tail call float @llvm.maxnum.f32(float %v846, float -inf), !dbg !20, !frisk.local_reduction !5
  %local.reduce251 = tail call float @llvm.maxnum.f32(float %v910, float %v850), !dbg !20, !frisk.local_reduction !5
  %local.reduce252 = tail call float @llvm.maxnum.f32(float %v847, float %v851), !dbg !20, !frisk.local_reduction !5
  %local.reduce253 = tail call float @llvm.maxnum.f32(float %v848, float %v852), !dbg !20, !frisk.local_reduction !5
  %local.reduce254 = tail call float @llvm.maxnum.f32(float %v849, float %v853), !dbg !20, !frisk.local_reduction !5
  %local.reduce255 = tail call float @llvm.maxnum.f32(float %local.reduce251, float %v854), !dbg !20, !frisk.local_reduction !5
  %local.reduce256 = tail call float @llvm.maxnum.f32(float %local.reduce252, float %v855), !dbg !20, !frisk.local_reduction !5
  %local.reduce257 = tail call float @llvm.maxnum.f32(float %local.reduce253, float %v856), !dbg !20, !frisk.local_reduction !5
  %local.reduce258 = tail call float @llvm.maxnum.f32(float %local.reduce254, float %v857), !dbg !20, !frisk.local_reduction !5
  %local.reduce259 = tail call float @llvm.maxnum.f32(float %local.reduce255, float %v858), !dbg !20, !frisk.local_reduction !5
  %local.reduce260 = tail call float @llvm.maxnum.f32(float %local.reduce256, float %v859), !dbg !20, !frisk.local_reduction !5
  %local.reduce261 = tail call float @llvm.maxnum.f32(float %local.reduce257, float %v860), !dbg !20, !frisk.local_reduction !5
  %local.reduce262 = tail call float @llvm.maxnum.f32(float %local.reduce258, float %v861), !dbg !20, !frisk.local_reduction !5
  %local.reduce263 = tail call float @llvm.maxnum.f32(float %local.reduce259, float %v862), !dbg !20, !frisk.local_reduction !5
  %local.reduce264 = tail call float @llvm.maxnum.f32(float %local.reduce260, float %v863), !dbg !20, !frisk.local_reduction !5
  %local.reduce265 = tail call float @llvm.maxnum.f32(float %local.reduce261, float %v864), !dbg !20, !frisk.local_reduction !5
  %local.reduce266 = tail call float @llvm.maxnum.f32(float %local.reduce262, float %v865), !dbg !20, !frisk.local_reduction !5
  %local.reduce267 = tail call float @llvm.maxnum.f32(float %local.reduce263, float %v866), !dbg !20, !frisk.local_reduction !5
  %local.reduce268 = tail call float @llvm.maxnum.f32(float %local.reduce264, float %v867), !dbg !20, !frisk.local_reduction !5
  %local.reduce269 = tail call float @llvm.maxnum.f32(float %local.reduce265, float %v868), !dbg !20, !frisk.local_reduction !5
  %local.reduce270 = tail call float @llvm.maxnum.f32(float %local.reduce266, float %v869), !dbg !20, !frisk.local_reduction !5
  %local.reduce271 = tail call float @llvm.maxnum.f32(float %local.reduce267, float %v870), !dbg !20, !frisk.local_reduction !5
  %local.reduce272 = tail call float @llvm.maxnum.f32(float %local.reduce268, float %v871), !dbg !20, !frisk.local_reduction !5
  %local.reduce273 = tail call float @llvm.maxnum.f32(float %local.reduce269, float %v872), !dbg !20, !frisk.local_reduction !5
  %local.reduce274 = tail call float @llvm.maxnum.f32(float %local.reduce270, float %v873), !dbg !20, !frisk.local_reduction !5
  %local.reduce275 = tail call float @llvm.maxnum.f32(float %local.reduce271, float %v874), !dbg !20, !frisk.local_reduction !5
  %local.reduce276 = tail call float @llvm.maxnum.f32(float %local.reduce272, float %v875), !dbg !20, !frisk.local_reduction !5
  %local.reduce277 = tail call float @llvm.maxnum.f32(float %local.reduce273, float %v876), !dbg !20, !frisk.local_reduction !5
  %local.reduce278 = tail call float @llvm.maxnum.f32(float %local.reduce274, float %v877), !dbg !20, !frisk.local_reduction !5
  %local.reduce279 = tail call float @llvm.maxnum.f32(float %local.reduce275, float %local.reduce276), !dbg !20, !frisk.local_reduction !5
  %local.reduce280 = tail call float @llvm.maxnum.f32(float %local.reduce277, float %local.reduce278), !dbg !20, !frisk.local_reduction !5
  %v911 = tail call float @llvm.maxnum.f32(float %local.reduce279, float %local.reduce280), !dbg !20, !frisk.local_reduction !5
  %v912 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %v913 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %v912), !dbg !20
  %v914 = add nuw nsw i32 %v913, 64, !dbg !20
  %v915 = xor i32 %v913, 16, !dbg !20
  %v916 = and i32 %v914, 192, !dbg !20
  %v917 = icmp samesign ult i32 %v915, %v916, !dbg !20
  %v918 = select i1 %v917, i32 %v915, i32 %v913, !dbg !20
  %v919 = shl nuw nsw i32 %v918, 2, !dbg !20
  %v920 = bitcast float %v911 to i32, !dbg !20
  %v921 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v919, i32 %v920), !dbg !20
  %v922 = bitcast i32 %v921 to float, !dbg !20
  %v923 = tail call float @llvm.maxnum.f32(float %v911, float %v922), !dbg !20
  %v924 = xor i32 %v913, 32, !dbg !20
  %v925 = icmp samesign ult i32 %v924, %v916, !dbg !20
  %v926 = select i1 %v925, i32 %v924, i32 %v913, !dbg !20
  %v927 = shl nuw nsw i32 %v926, 2, !dbg !20
  %v928 = bitcast float %v923 to i32, !dbg !20
  %v929 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v927, i32 %v928), !dbg !20
  %v930 = bitcast i32 %v929 to float, !dbg !20
  %v931 = tail call float @llvm.maxnum.f32(float %v923, float %v930), !dbg !20
  %v932 = extractvalue [2 x <1 x float>] %v20855, 0, !dbg !20
  %v933 = insertelement <1 x float> poison, float %v931, i64 0, !dbg !20
  %v934 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %v933, <1 x float> %v932), !dbg !20
  %v935 = tail call float @llvm.maxnum.f32(float %v878, float -inf), !dbg !20, !frisk.local_reduction !5
  %local.reduce220 = tail call float @llvm.maxnum.f32(float %v935, float %v882), !dbg !20, !frisk.local_reduction !5
  %local.reduce221 = tail call float @llvm.maxnum.f32(float %v879, float %v883), !dbg !20, !frisk.local_reduction !5
  %local.reduce222 = tail call float @llvm.maxnum.f32(float %v880, float %v884), !dbg !20, !frisk.local_reduction !5
  %local.reduce223 = tail call float @llvm.maxnum.f32(float %v881, float %v885), !dbg !20, !frisk.local_reduction !5
  %local.reduce224 = tail call float @llvm.maxnum.f32(float %local.reduce220, float %v886), !dbg !20, !frisk.local_reduction !5
  %local.reduce225 = tail call float @llvm.maxnum.f32(float %local.reduce221, float %v887), !dbg !20, !frisk.local_reduction !5
  %local.reduce226 = tail call float @llvm.maxnum.f32(float %local.reduce222, float %v888), !dbg !20, !frisk.local_reduction !5
  %local.reduce227 = tail call float @llvm.maxnum.f32(float %local.reduce223, float %v889), !dbg !20, !frisk.local_reduction !5
  %local.reduce228 = tail call float @llvm.maxnum.f32(float %local.reduce224, float %v890), !dbg !20, !frisk.local_reduction !5
  %local.reduce229 = tail call float @llvm.maxnum.f32(float %local.reduce225, float %v891), !dbg !20, !frisk.local_reduction !5
  %local.reduce230 = tail call float @llvm.maxnum.f32(float %local.reduce226, float %v892), !dbg !20, !frisk.local_reduction !5
  %local.reduce231 = tail call float @llvm.maxnum.f32(float %local.reduce227, float %v893), !dbg !20, !frisk.local_reduction !5
  %local.reduce232 = tail call float @llvm.maxnum.f32(float %local.reduce228, float %v894), !dbg !20, !frisk.local_reduction !5
  %local.reduce233 = tail call float @llvm.maxnum.f32(float %local.reduce229, float %v895), !dbg !20, !frisk.local_reduction !5
  %local.reduce234 = tail call float @llvm.maxnum.f32(float %local.reduce230, float %v896), !dbg !20, !frisk.local_reduction !5
  %local.reduce235 = tail call float @llvm.maxnum.f32(float %local.reduce231, float %v897), !dbg !20, !frisk.local_reduction !5
  %local.reduce236 = tail call float @llvm.maxnum.f32(float %local.reduce232, float %v898), !dbg !20, !frisk.local_reduction !5
  %local.reduce237 = tail call float @llvm.maxnum.f32(float %local.reduce233, float %v899), !dbg !20, !frisk.local_reduction !5
  %local.reduce238 = tail call float @llvm.maxnum.f32(float %local.reduce234, float %v900), !dbg !20, !frisk.local_reduction !5
  %local.reduce239 = tail call float @llvm.maxnum.f32(float %local.reduce235, float %v901), !dbg !20, !frisk.local_reduction !5
  %local.reduce240 = tail call float @llvm.maxnum.f32(float %local.reduce236, float %v902), !dbg !20, !frisk.local_reduction !5
  %local.reduce241 = tail call float @llvm.maxnum.f32(float %local.reduce237, float %v903), !dbg !20, !frisk.local_reduction !5
  %local.reduce242 = tail call float @llvm.maxnum.f32(float %local.reduce238, float %v904), !dbg !20, !frisk.local_reduction !5
  %local.reduce243 = tail call float @llvm.maxnum.f32(float %local.reduce239, float %v905), !dbg !20, !frisk.local_reduction !5
  %local.reduce244 = tail call float @llvm.maxnum.f32(float %local.reduce240, float %v906), !dbg !20, !frisk.local_reduction !5
  %local.reduce245 = tail call float @llvm.maxnum.f32(float %local.reduce241, float %v907), !dbg !20, !frisk.local_reduction !5
  %local.reduce246 = tail call float @llvm.maxnum.f32(float %local.reduce242, float %v908), !dbg !20, !frisk.local_reduction !5
  %local.reduce247 = tail call float @llvm.maxnum.f32(float %local.reduce243, float %v909), !dbg !20, !frisk.local_reduction !5
  %local.reduce248 = tail call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !20, !frisk.local_reduction !5
  %local.reduce249 = tail call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !20, !frisk.local_reduction !5
  %v936 = tail call float @llvm.maxnum.f32(float %local.reduce248, float %local.reduce249), !dbg !20, !frisk.local_reduction !5
  %v937 = bitcast float %v936 to i32, !dbg !20
  %v938 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v919, i32 %v937), !dbg !20
  %v939 = bitcast i32 %v938 to float, !dbg !20
  %v940 = tail call float @llvm.maxnum.f32(float %v936, float %v939), !dbg !20
  %v941 = bitcast float %v940 to i32, !dbg !20
  %v942 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v927, i32 %v941), !dbg !20
  %v943 = bitcast i32 %v942 to float, !dbg !20
  %v944 = tail call float @llvm.maxnum.f32(float %v940, float %v943), !dbg !20
  %v945 = extractvalue [2 x <1 x float>] %v20855, 1, !dbg !20
  %v946 = insertelement <1 x float> poison, float %v944, i64 0, !dbg !20
  %v947 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %v946, <1 x float> %v945), !dbg !20
  %v948 = shufflevector <1 x float> %v934, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %v949 = fsub <4 x float> %v829, %v948, !dbg !21
  %v950 = fsub <4 x float> %v830, %v948, !dbg !21
  %v951 = fsub <4 x float> %v831, %v948, !dbg !21
  %v952 = fsub <4 x float> %v832, %v948, !dbg !21
  %v953 = fsub <4 x float> %v833, %v948, !dbg !21
  %v954 = fsub <4 x float> %v834, %v948, !dbg !21
  %v955 = fsub <4 x float> %v835, %v948, !dbg !21
  %v956 = fsub <4 x float> %v836, %v948, !dbg !21
  %v957 = shufflevector <1 x float> %v947, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %v958 = fsub <4 x float> %v837, %v957, !dbg !21
  %v959 = fsub <4 x float> %v838, %v957, !dbg !21
  %v960 = fsub <4 x float> %v839, %v957, !dbg !21
  %v961 = fsub <4 x float> %v840, %v957, !dbg !21
  %v962 = fsub <4 x float> %v841, %v957, !dbg !21
  %v963 = fsub <4 x float> %v842, %v957, !dbg !21
  %v964 = fsub <4 x float> %v843, %v957, !dbg !21
  %v965 = fsub <4 x float> %v844, %v957, !dbg !21
  %v966 = fsub <1 x float> %v932, %v934, !dbg !22
  %v967 = extractelement <1 x float> %v966, i64 0, !dbg !23
  %v968 = tail call float @llvm.amdgcn.exp2.f32(float %v967), !dbg !23
  %v969 = insertelement <4 x float> poison, float %v968, i64 0, !dbg !24
  %v970 = shufflevector <4 x float> %v969, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %v971 = extractvalue [2 x <1 x float>] %v20754, 0, !dbg !25
  %v972 = extractelement <1 x float> %v971, i64 0, !dbg !25
  %v973 = fmul float %v972, %v968, !dbg !25
  %v974 = fsub <1 x float> %v945, %v947, !dbg !22
  %v975 = extractelement <1 x float> %v974, i64 0, !dbg !23
  %v976 = tail call float @llvm.amdgcn.exp2.f32(float %v975), !dbg !23
  %v977 = insertelement <4 x float> poison, float %v976, i64 0, !dbg !24
  %v978 = shufflevector <4 x float> %v977, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %v979 = extractvalue [2 x <1 x float>] %v20754, 1, !dbg !25
  %v980 = extractelement <1 x float> %v979, i64 0, !dbg !25
  %v981 = fmul float %v980, %v976, !dbg !25
  %v982 = extractelement <4 x float> %v949, i64 0, !dbg !26
  %v983 = tail call float @llvm.amdgcn.exp2.f32(float %v982), !dbg !26
  %v984 = insertelement <4 x float> poison, float %v983, i64 0, !dbg !27
  %v985 = extractelement <4 x float> %v949, i64 1, !dbg !26
  %v986 = tail call float @llvm.amdgcn.exp2.f32(float %v985), !dbg !26
  %v987 = insertelement <4 x float> %v984, float %v986, i64 1, !dbg !27
  %v988 = extractelement <4 x float> %v949, i64 2, !dbg !26
  %v989 = tail call float @llvm.amdgcn.exp2.f32(float %v988), !dbg !26
  %v990 = insertelement <4 x float> %v987, float %v989, i64 2, !dbg !27
  %v991 = extractelement <4 x float> %v949, i64 3, !dbg !26
  %v992 = tail call float @llvm.amdgcn.exp2.f32(float %v991), !dbg !26
  %v993 = insertelement <4 x float> %v990, float %v992, i64 3, !dbg !27
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v993) #8, !dbg !27, !frisk.register_lifetimes !5
  %v994 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v994) #8, !dbg !26, !frisk.register_lifetimes !5
  %v995 = extractelement <4 x float> %v950, i64 0, !dbg !26
  %v996 = tail call float @llvm.amdgcn.exp2.f32(float %v995), !dbg !26
  %v997 = insertelement <4 x float> poison, float %v996, i64 0, !dbg !27
  %local.reduce189 = fadd float %v983, %v996, !dbg !28, !frisk.local_reduction !5
  %v998 = extractelement <4 x float> %v950, i64 1, !dbg !26
  %v999 = tail call float @llvm.amdgcn.exp2.f32(float %v998), !dbg !26
  %v1000 = insertelement <4 x float> %v997, float %v999, i64 1, !dbg !27
  %local.reduce190 = fadd float %v986, %v999, !dbg !28, !frisk.local_reduction !5
  %v1001 = extractelement <4 x float> %v950, i64 2, !dbg !26
  %v1002 = tail call float @llvm.amdgcn.exp2.f32(float %v1001), !dbg !26
  %v1003 = insertelement <4 x float> %v1000, float %v1002, i64 2, !dbg !27
  %local.reduce191 = fadd float %v989, %v1002, !dbg !28, !frisk.local_reduction !5
  %v1004 = extractelement <4 x float> %v950, i64 3, !dbg !26
  %v1005 = tail call float @llvm.amdgcn.exp2.f32(float %v1004), !dbg !26
  %v1006 = insertelement <4 x float> %v1003, float %v1005, i64 3, !dbg !27
  %math.result465 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1006) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1007 = fptrunc <4 x float> %math.result465 to <4 x half>, !dbg !27
  %math.packed466 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1007) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce192 = fadd float %v992, %v1005, !dbg !28, !frisk.local_reduction !5
  %v1008 = extractelement <4 x float> %v951, i64 0, !dbg !26
  %v1009 = tail call float @llvm.amdgcn.exp2.f32(float %v1008), !dbg !26
  %v1010 = insertelement <4 x float> poison, float %v1009, i64 0, !dbg !27
  %local.reduce193 = fadd float %v1009, %local.reduce189, !dbg !28, !frisk.local_reduction !5
  %v1011 = extractelement <4 x float> %v951, i64 1, !dbg !26
  %v1012 = tail call float @llvm.amdgcn.exp2.f32(float %v1011), !dbg !26
  %v1013 = insertelement <4 x float> %v1010, float %v1012, i64 1, !dbg !27
  %local.reduce194 = fadd float %v1012, %local.reduce190, !dbg !28, !frisk.local_reduction !5
  %v1014 = extractelement <4 x float> %v951, i64 2, !dbg !26
  %v1015 = tail call float @llvm.amdgcn.exp2.f32(float %v1014), !dbg !26
  %v1016 = insertelement <4 x float> %v1013, float %v1015, i64 2, !dbg !27
  %local.reduce195 = fadd float %v1015, %local.reduce191, !dbg !28, !frisk.local_reduction !5
  %v1017 = extractelement <4 x float> %v951, i64 3, !dbg !26
  %v1018 = tail call float @llvm.amdgcn.exp2.f32(float %v1017), !dbg !26
  %v1019 = insertelement <4 x float> %v1016, float %v1018, i64 3, !dbg !27
  %math.result467 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1019) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1020 = fptrunc <4 x float> %math.result467 to <4 x half>, !dbg !27
  %math.packed468 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1020) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce196 = fadd float %v1018, %local.reduce192, !dbg !28, !frisk.local_reduction !5
  %v1021 = extractelement <4 x float> %v952, i64 0, !dbg !26
  %v1022 = tail call float @llvm.amdgcn.exp2.f32(float %v1021), !dbg !26
  %v1023 = insertelement <4 x float> poison, float %v1022, i64 0, !dbg !27
  %local.reduce197 = fadd float %v1022, %local.reduce193, !dbg !28, !frisk.local_reduction !5
  %v1024 = extractelement <4 x float> %v952, i64 1, !dbg !26
  %v1025 = tail call float @llvm.amdgcn.exp2.f32(float %v1024), !dbg !26
  %v1026 = insertelement <4 x float> %v1023, float %v1025, i64 1, !dbg !27
  %local.reduce198 = fadd float %v1025, %local.reduce194, !dbg !28, !frisk.local_reduction !5
  %v1027 = extractelement <4 x float> %v952, i64 2, !dbg !26
  %v1028 = tail call float @llvm.amdgcn.exp2.f32(float %v1027), !dbg !26
  %v1029 = insertelement <4 x float> %v1026, float %v1028, i64 2, !dbg !27
  %local.reduce199 = fadd float %v1028, %local.reduce195, !dbg !28, !frisk.local_reduction !5
  %v1030 = extractelement <4 x float> %v952, i64 3, !dbg !26
  %v1031 = tail call float @llvm.amdgcn.exp2.f32(float %v1030), !dbg !26
  %v1032 = insertelement <4 x float> %v1029, float %v1031, i64 3, !dbg !27
  %math.result469 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1032) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1033 = fptrunc <4 x float> %math.result469 to <4 x half>, !dbg !27
  %math.packed470 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1033) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce200 = fadd float %v1031, %local.reduce196, !dbg !28, !frisk.local_reduction !5
  %v1034 = extractelement <4 x float> %v953, i64 0, !dbg !26
  %v1035 = tail call float @llvm.amdgcn.exp2.f32(float %v1034), !dbg !26
  %v1036 = insertelement <4 x float> poison, float %v1035, i64 0, !dbg !27
  %local.reduce201 = fadd float %v1035, %local.reduce197, !dbg !28, !frisk.local_reduction !5
  %v1037 = extractelement <4 x float> %v953, i64 1, !dbg !26
  %v1038 = tail call float @llvm.amdgcn.exp2.f32(float %v1037), !dbg !26
  %v1039 = insertelement <4 x float> %v1036, float %v1038, i64 1, !dbg !27
  %local.reduce202 = fadd float %v1038, %local.reduce198, !dbg !28, !frisk.local_reduction !5
  %v1040 = extractelement <4 x float> %v953, i64 2, !dbg !26
  %v1041 = tail call float @llvm.amdgcn.exp2.f32(float %v1040), !dbg !26
  %v1042 = insertelement <4 x float> %v1039, float %v1041, i64 2, !dbg !27
  %local.reduce203 = fadd float %v1041, %local.reduce199, !dbg !28, !frisk.local_reduction !5
  %v1043 = extractelement <4 x float> %v953, i64 3, !dbg !26
  %v1044 = tail call float @llvm.amdgcn.exp2.f32(float %v1043), !dbg !26
  %v1045 = insertelement <4 x float> %v1042, float %v1044, i64 3, !dbg !27
  %math.result471 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1045) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1046 = fptrunc <4 x float> %math.result471 to <4 x half>, !dbg !27
  %math.packed472 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1046) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce204 = fadd float %v1044, %local.reduce200, !dbg !28, !frisk.local_reduction !5
  %v1047 = extractelement <4 x float> %v954, i64 0, !dbg !26
  %v1048 = tail call float @llvm.amdgcn.exp2.f32(float %v1047), !dbg !26
  %v1049 = insertelement <4 x float> poison, float %v1048, i64 0, !dbg !27
  %local.reduce205 = fadd float %v1048, %local.reduce201, !dbg !28, !frisk.local_reduction !5
  %v1050 = extractelement <4 x float> %v954, i64 1, !dbg !26
  %v1051 = tail call float @llvm.amdgcn.exp2.f32(float %v1050), !dbg !26
  %v1052 = insertelement <4 x float> %v1049, float %v1051, i64 1, !dbg !27
  %local.reduce206 = fadd float %v1051, %local.reduce202, !dbg !28, !frisk.local_reduction !5
  %v1053 = extractelement <4 x float> %v954, i64 2, !dbg !26
  %v1054 = tail call float @llvm.amdgcn.exp2.f32(float %v1053), !dbg !26
  %v1055 = insertelement <4 x float> %v1052, float %v1054, i64 2, !dbg !27
  %local.reduce207 = fadd float %v1054, %local.reduce203, !dbg !28, !frisk.local_reduction !5
  %v1056 = extractelement <4 x float> %v954, i64 3, !dbg !26
  %v1057 = tail call float @llvm.amdgcn.exp2.f32(float %v1056), !dbg !26
  %v1058 = insertelement <4 x float> %v1055, float %v1057, i64 3, !dbg !27
  %math.result473 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1058) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1059 = fptrunc <4 x float> %math.result473 to <4 x half>, !dbg !27
  %math.packed474 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1059) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce208 = fadd float %v1057, %local.reduce204, !dbg !28, !frisk.local_reduction !5
  %v1060 = extractelement <4 x float> %v955, i64 0, !dbg !26
  %v1061 = tail call float @llvm.amdgcn.exp2.f32(float %v1060), !dbg !26
  %v1062 = insertelement <4 x float> poison, float %v1061, i64 0, !dbg !27
  %local.reduce209 = fadd float %v1061, %local.reduce205, !dbg !28, !frisk.local_reduction !5
  %v1063 = extractelement <4 x float> %v955, i64 1, !dbg !26
  %v1064 = tail call float @llvm.amdgcn.exp2.f32(float %v1063), !dbg !26
  %v1065 = insertelement <4 x float> %v1062, float %v1064, i64 1, !dbg !27
  %local.reduce210 = fadd float %v1064, %local.reduce206, !dbg !28, !frisk.local_reduction !5
  %v1066 = extractelement <4 x float> %v955, i64 2, !dbg !26
  %v1067 = tail call float @llvm.amdgcn.exp2.f32(float %v1066), !dbg !26
  %v1068 = insertelement <4 x float> %v1065, float %v1067, i64 2, !dbg !27
  %local.reduce211 = fadd float %v1067, %local.reduce207, !dbg !28, !frisk.local_reduction !5
  %v1069 = extractelement <4 x float> %v955, i64 3, !dbg !26
  %v1070 = tail call float @llvm.amdgcn.exp2.f32(float %v1069), !dbg !26
  %v1071 = insertelement <4 x float> %v1068, float %v1070, i64 3, !dbg !27
  %math.result475 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1071) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1072 = fptrunc <4 x float> %math.result475 to <4 x half>, !dbg !27
  %math.packed476 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1072) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce212 = fadd float %v1070, %local.reduce208, !dbg !28, !frisk.local_reduction !5
  %v1073 = extractelement <4 x float> %v956, i64 0, !dbg !26
  %v1074 = tail call float @llvm.amdgcn.exp2.f32(float %v1073), !dbg !26
  %v1075 = insertelement <4 x float> poison, float %v1074, i64 0, !dbg !27
  %local.reduce213 = fadd float %v1074, %local.reduce209, !dbg !28, !frisk.local_reduction !5
  %v1076 = extractelement <4 x float> %v956, i64 1, !dbg !26
  %v1077 = tail call float @llvm.amdgcn.exp2.f32(float %v1076), !dbg !26
  %v1078 = insertelement <4 x float> %v1075, float %v1077, i64 1, !dbg !27
  %local.reduce214 = fadd float %v1077, %local.reduce210, !dbg !28, !frisk.local_reduction !5
  %local.reduce217 = fadd float %local.reduce213, %local.reduce214, !dbg !28, !frisk.local_reduction !5
  %v1079 = extractelement <4 x float> %v956, i64 2, !dbg !26
  %v1080 = tail call float @llvm.amdgcn.exp2.f32(float %v1079), !dbg !26
  %v1081 = insertelement <4 x float> %v1078, float %v1080, i64 2, !dbg !27
  %local.reduce215 = fadd float %v1080, %local.reduce211, !dbg !28, !frisk.local_reduction !5
  %v1082 = extractelement <4 x float> %v956, i64 3, !dbg !26
  %v1083 = tail call float @llvm.amdgcn.exp2.f32(float %v1082), !dbg !26
  %v1084 = insertelement <4 x float> %v1081, float %v1083, i64 3, !dbg !27
  %math.result477 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1084) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1085 = fptrunc <4 x float> %math.result477 to <4 x half>, !dbg !27
  %math.packed478 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1085) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce216 = fadd float %v1083, %local.reduce212, !dbg !28, !frisk.local_reduction !5
  %local.reduce218 = fadd float %local.reduce215, %local.reduce216, !dbg !28, !frisk.local_reduction !5
  %v1086 = fadd float %local.reduce217, %local.reduce218, !dbg !28, !frisk.local_reduction !5
  %v1087 = extractelement <4 x float> %v958, i64 0, !dbg !26
  %v1088 = tail call float @llvm.amdgcn.exp2.f32(float %v1087), !dbg !26
  %v1089 = insertelement <4 x float> poison, float %v1088, i64 0, !dbg !27
  %v1090 = extractelement <4 x float> %v958, i64 1, !dbg !26
  %v1091 = tail call float @llvm.amdgcn.exp2.f32(float %v1090), !dbg !26
  %v1092 = insertelement <4 x float> %v1089, float %v1091, i64 1, !dbg !27
  %v1093 = extractelement <4 x float> %v958, i64 2, !dbg !26
  %v1094 = tail call float @llvm.amdgcn.exp2.f32(float %v1093), !dbg !26
  %v1095 = insertelement <4 x float> %v1092, float %v1094, i64 2, !dbg !27
  %v1096 = extractelement <4 x float> %v958, i64 3, !dbg !26
  %v1097 = tail call float @llvm.amdgcn.exp2.f32(float %v1096), !dbg !26
  %v1098 = insertelement <4 x float> %v1095, float %v1097, i64 3, !dbg !27
  %math.result479 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1098) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1099 = fptrunc <4 x float> %math.result479 to <4 x half>, !dbg !27
  %math.packed480 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1099) #8, !dbg !26, !frisk.register_lifetimes !5
  %v1100 = extractelement <4 x float> %v959, i64 0, !dbg !26
  %v1101 = tail call float @llvm.amdgcn.exp2.f32(float %v1100), !dbg !26
  %v1102 = insertelement <4 x float> poison, float %v1101, i64 0, !dbg !27
  %local.reduce = fadd float %v1088, %v1101, !dbg !28, !frisk.local_reduction !5
  %v1103 = extractelement <4 x float> %v959, i64 1, !dbg !26
  %v1104 = tail call float @llvm.amdgcn.exp2.f32(float %v1103), !dbg !26
  %v1105 = insertelement <4 x float> %v1102, float %v1104, i64 1, !dbg !27
  %local.reduce159 = fadd float %v1091, %v1104, !dbg !28, !frisk.local_reduction !5
  %v1106 = extractelement <4 x float> %v959, i64 2, !dbg !26
  %v1107 = tail call float @llvm.amdgcn.exp2.f32(float %v1106), !dbg !26
  %v1108 = insertelement <4 x float> %v1105, float %v1107, i64 2, !dbg !27
  %local.reduce160 = fadd float %v1094, %v1107, !dbg !28, !frisk.local_reduction !5
  %v1109 = extractelement <4 x float> %v959, i64 3, !dbg !26
  %v1110 = tail call float @llvm.amdgcn.exp2.f32(float %v1109), !dbg !26
  %v1111 = insertelement <4 x float> %v1108, float %v1110, i64 3, !dbg !27
  %math.result481 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1111) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1112 = fptrunc <4 x float> %math.result481 to <4 x half>, !dbg !27
  %math.packed482 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1112) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce161 = fadd float %v1097, %v1110, !dbg !28, !frisk.local_reduction !5
  %v1113 = extractelement <4 x float> %v960, i64 0, !dbg !26
  %v1114 = tail call float @llvm.amdgcn.exp2.f32(float %v1113), !dbg !26
  %v1115 = insertelement <4 x float> poison, float %v1114, i64 0, !dbg !27
  %local.reduce162 = fadd float %v1114, %local.reduce, !dbg !28, !frisk.local_reduction !5
  %v1116 = extractelement <4 x float> %v960, i64 1, !dbg !26
  %v1117 = tail call float @llvm.amdgcn.exp2.f32(float %v1116), !dbg !26
  %v1118 = insertelement <4 x float> %v1115, float %v1117, i64 1, !dbg !27
  %local.reduce163 = fadd float %v1117, %local.reduce159, !dbg !28, !frisk.local_reduction !5
  %v1119 = extractelement <4 x float> %v960, i64 2, !dbg !26
  %v1120 = tail call float @llvm.amdgcn.exp2.f32(float %v1119), !dbg !26
  %v1121 = insertelement <4 x float> %v1118, float %v1120, i64 2, !dbg !27
  %local.reduce164 = fadd float %v1120, %local.reduce160, !dbg !28, !frisk.local_reduction !5
  %v1122 = extractelement <4 x float> %v960, i64 3, !dbg !26
  %v1123 = tail call float @llvm.amdgcn.exp2.f32(float %v1122), !dbg !26
  %v1124 = insertelement <4 x float> %v1121, float %v1123, i64 3, !dbg !27
  %math.result483 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1124) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1125 = fptrunc <4 x float> %math.result483 to <4 x half>, !dbg !27
  %math.packed484 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1125) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce165 = fadd float %v1123, %local.reduce161, !dbg !28, !frisk.local_reduction !5
  %v1126 = extractelement <4 x float> %v961, i64 0, !dbg !26
  %v1127 = tail call float @llvm.amdgcn.exp2.f32(float %v1126), !dbg !26
  %v1128 = insertelement <4 x float> poison, float %v1127, i64 0, !dbg !27
  %local.reduce166 = fadd float %v1127, %local.reduce162, !dbg !28, !frisk.local_reduction !5
  %v1129 = extractelement <4 x float> %v961, i64 1, !dbg !26
  %v1130 = tail call float @llvm.amdgcn.exp2.f32(float %v1129), !dbg !26
  %v1131 = insertelement <4 x float> %v1128, float %v1130, i64 1, !dbg !27
  %local.reduce167 = fadd float %v1130, %local.reduce163, !dbg !28, !frisk.local_reduction !5
  %v1132 = extractelement <4 x float> %v961, i64 2, !dbg !26
  %v1133 = tail call float @llvm.amdgcn.exp2.f32(float %v1132), !dbg !26
  %v1134 = insertelement <4 x float> %v1131, float %v1133, i64 2, !dbg !27
  %local.reduce168 = fadd float %v1133, %local.reduce164, !dbg !28, !frisk.local_reduction !5
  %v1135 = extractelement <4 x float> %v961, i64 3, !dbg !26
  %v1136 = tail call float @llvm.amdgcn.exp2.f32(float %v1135), !dbg !26
  %v1137 = insertelement <4 x float> %v1134, float %v1136, i64 3, !dbg !27
  %math.result485 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1137) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1138 = fptrunc <4 x float> %math.result485 to <4 x half>, !dbg !27
  %math.packed486 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1138) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce169 = fadd float %v1136, %local.reduce165, !dbg !28, !frisk.local_reduction !5
  %v1139 = extractelement <4 x float> %v962, i64 0, !dbg !26
  %v1140 = tail call float @llvm.amdgcn.exp2.f32(float %v1139), !dbg !26
  %v1141 = insertelement <4 x float> poison, float %v1140, i64 0, !dbg !27
  %local.reduce170 = fadd float %v1140, %local.reduce166, !dbg !28, !frisk.local_reduction !5
  %v1142 = extractelement <4 x float> %v962, i64 1, !dbg !26
  %v1143 = tail call float @llvm.amdgcn.exp2.f32(float %v1142), !dbg !26
  %v1144 = insertelement <4 x float> %v1141, float %v1143, i64 1, !dbg !27
  %local.reduce171 = fadd float %v1143, %local.reduce167, !dbg !28, !frisk.local_reduction !5
  %v1145 = extractelement <4 x float> %v962, i64 2, !dbg !26
  %v1146 = tail call float @llvm.amdgcn.exp2.f32(float %v1145), !dbg !26
  %v1147 = insertelement <4 x float> %v1144, float %v1146, i64 2, !dbg !27
  %local.reduce172 = fadd float %v1146, %local.reduce168, !dbg !28, !frisk.local_reduction !5
  %v1148 = extractelement <4 x float> %v962, i64 3, !dbg !26
  %v1149 = tail call float @llvm.amdgcn.exp2.f32(float %v1148), !dbg !26
  %v1150 = insertelement <4 x float> %v1147, float %v1149, i64 3, !dbg !27
  %math.result487 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1150) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1151 = fptrunc <4 x float> %math.result487 to <4 x half>, !dbg !27
  %math.packed488 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1151) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce173 = fadd float %v1149, %local.reduce169, !dbg !28, !frisk.local_reduction !5
  %v1152 = extractelement <4 x float> %v963, i64 0, !dbg !26
  %v1153 = tail call float @llvm.amdgcn.exp2.f32(float %v1152), !dbg !26
  %v1154 = insertelement <4 x float> poison, float %v1153, i64 0, !dbg !27
  %local.reduce174 = fadd float %v1153, %local.reduce170, !dbg !28, !frisk.local_reduction !5
  %v1155 = extractelement <4 x float> %v963, i64 1, !dbg !26
  %v1156 = tail call float @llvm.amdgcn.exp2.f32(float %v1155), !dbg !26
  %v1157 = insertelement <4 x float> %v1154, float %v1156, i64 1, !dbg !27
  %local.reduce175 = fadd float %v1156, %local.reduce171, !dbg !28, !frisk.local_reduction !5
  %v1158 = extractelement <4 x float> %v963, i64 2, !dbg !26
  %v1159 = tail call float @llvm.amdgcn.exp2.f32(float %v1158), !dbg !26
  %v1160 = insertelement <4 x float> %v1157, float %v1159, i64 2, !dbg !27
  %local.reduce176 = fadd float %v1159, %local.reduce172, !dbg !28, !frisk.local_reduction !5
  %v1161 = extractelement <4 x float> %v963, i64 3, !dbg !26
  %v1162 = tail call float @llvm.amdgcn.exp2.f32(float %v1161), !dbg !26
  %v1163 = insertelement <4 x float> %v1160, float %v1162, i64 3, !dbg !27
  %math.result489 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1163) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1164 = fptrunc <4 x float> %math.result489 to <4 x half>, !dbg !27
  %math.packed490 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1164) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce177 = fadd float %v1162, %local.reduce173, !dbg !28, !frisk.local_reduction !5
  %v1165 = extractelement <4 x float> %v964, i64 0, !dbg !26
  %v1166 = tail call float @llvm.amdgcn.exp2.f32(float %v1165), !dbg !26
  %v1167 = insertelement <4 x float> poison, float %v1166, i64 0, !dbg !27
  %local.reduce178 = fadd float %v1166, %local.reduce174, !dbg !28, !frisk.local_reduction !5
  %v1168 = extractelement <4 x float> %v964, i64 1, !dbg !26
  %v1169 = tail call float @llvm.amdgcn.exp2.f32(float %v1168), !dbg !26
  %v1170 = insertelement <4 x float> %v1167, float %v1169, i64 1, !dbg !27
  %local.reduce179 = fadd float %v1169, %local.reduce175, !dbg !28, !frisk.local_reduction !5
  %v1171 = extractelement <4 x float> %v964, i64 2, !dbg !26
  %v1172 = tail call float @llvm.amdgcn.exp2.f32(float %v1171), !dbg !26
  %v1173 = insertelement <4 x float> %v1170, float %v1172, i64 2, !dbg !27
  %local.reduce180 = fadd float %v1172, %local.reduce176, !dbg !28, !frisk.local_reduction !5
  %v1174 = extractelement <4 x float> %v964, i64 3, !dbg !26
  %v1175 = tail call float @llvm.amdgcn.exp2.f32(float %v1174), !dbg !26
  %v1176 = insertelement <4 x float> %v1173, float %v1175, i64 3, !dbg !27
  %math.result491 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1176) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1177 = fptrunc <4 x float> %math.result491 to <4 x half>, !dbg !27
  %math.packed492 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1177) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce181 = fadd float %v1175, %local.reduce177, !dbg !28, !frisk.local_reduction !5
  %v1178 = extractelement <4 x float> %v965, i64 0, !dbg !26
  %v1179 = tail call float @llvm.amdgcn.exp2.f32(float %v1178), !dbg !26
  %v1180 = insertelement <4 x float> poison, float %v1179, i64 0, !dbg !27
  %local.reduce182 = fadd float %v1179, %local.reduce178, !dbg !28, !frisk.local_reduction !5
  %v1181 = extractelement <4 x float> %v965, i64 1, !dbg !26
  %v1182 = tail call float @llvm.amdgcn.exp2.f32(float %v1181), !dbg !26
  %v1183 = insertelement <4 x float> %v1180, float %v1182, i64 1, !dbg !27
  %local.reduce183 = fadd float %v1182, %local.reduce179, !dbg !28, !frisk.local_reduction !5
  %local.reduce186 = fadd float %local.reduce182, %local.reduce183, !dbg !28, !frisk.local_reduction !5
  %v1184 = extractelement <4 x float> %v965, i64 2, !dbg !26
  %v1185 = tail call float @llvm.amdgcn.exp2.f32(float %v1184), !dbg !26
  %v1186 = insertelement <4 x float> %v1183, float %v1185, i64 2, !dbg !27
  %local.reduce184 = fadd float %v1185, %local.reduce180, !dbg !28, !frisk.local_reduction !5
  %v1187 = extractelement <4 x float> %v965, i64 3, !dbg !26
  %v1188 = tail call float @llvm.amdgcn.exp2.f32(float %v1187), !dbg !26
  %v1189 = insertelement <4 x float> %v1186, float %v1188, i64 3, !dbg !27
  %math.result493 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1189) #8, !dbg !27, !frisk.register_lifetimes !5
  %v1190 = fptrunc <4 x float> %math.result493 to <4 x half>, !dbg !27
  %math.packed494 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1190) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce185 = fadd float %v1188, %local.reduce181, !dbg !28, !frisk.local_reduction !5
  %local.reduce187 = fadd float %local.reduce184, %local.reduce185, !dbg !28, !frisk.local_reduction !5
  %v1191 = fadd float %local.reduce186, %local.reduce187, !dbg !28, !frisk.local_reduction !5
  %v1200 = fadd float %v973, %v1086, !dbg !28
  %v1201 = insertelement <1 x float> poison, float %v1200, i64 0, !dbg !11
  %v1202 = insertvalue [2 x <1 x float>] poison, <1 x float> %v1201, 0, !dbg !11
  %v1211 = fadd float %v981, %v1191, !dbg !28
  %v1212 = insertelement <1 x float> poison, float %v1211, i64 0, !dbg !11
  %v1213 = insertvalue [2 x <1 x float>] %v1202, <1 x float> %v1212, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread1020 = tail call i32 asm sideeffect "", "=v,0"(i32 %v31) #8, !dbg !29, !range !9, !frisk.register_lifetimes !5
  %address.local1021 = shl nuw nsw i32 %address.thread1020, 3, !dbg !29
  %address.local1022 = and i32 %address.local1021, 120, !dbg !29
  %address.local1023 = shl nuw nsw i32 %address.thread1020, 5, !dbg !29
  %address.local1024 = and i32 %address.local1023, 1536, !dbg !29
  %address.local1025 = or disjoint i32 %address.local1022, %address.local1024, !dbg !29
  %address.local1026 = zext nneg i32 %address.local1025 to i64, !dbg !29
  %address.local1027 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1026, !dbg !29
  %v1214 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #10, !dbg !29
  %v1215 = bitcast <4 x i32> %v1214 to <8 x half>, !dbg !29
  %address.local1028 = zext nneg i32 %address.thread1020 to i64, !dbg !8
  %address.local1029 = and i64 %address.local1028, 60, !dbg !29
  %address.local1030 = add nuw nsw i64 %address.local1029, %address.local1028, !dbg !29
  %address.local1032.idx = shl nuw nsw i64 %address.local1030, 4, !dbg !29
  %address.local1032 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1032.idx, !dbg !29
  %address.local1033 = lshr i64 %address.local1028, 4, !dbg !29
  %address.local1034 = and i64 %address.local1033, 3, !dbg !29
  %address.local1036.idx = mul nuw nsw i64 %address.local1034, 768, !dbg !29
  %address.local1036 = getelementptr i8, ptr addrspace(3) %address.local1032, i64 %address.local1036.idx, !dbg !29
  %address.local1037 = lshr i64 %address.local1028, 2, !dbg !15
  %address.local1039.idx = mul nsw i64 %address.local1037, -64, !dbg !29
  %address.local1039 = getelementptr i8, ptr addrspace(3) %address.local1036, i64 %address.local1039.idx, !dbg !29
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 256, !dbg !29
  %v1216 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #10, !dbg !29
  %v1217 = bitcast <4 x i32> %v1216 to <8 x half>, !dbg !29
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 512, !dbg !29
  %v1218 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #10, !dbg !29
  %v1219 = bitcast <4 x i32> %v1218 to <8 x half>, !dbg !29
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 768, !dbg !29
  %v1220 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #10, !dbg !29
  %v1221 = bitcast <4 x i32> %v1220 to <8 x half>, !dbg !29
  %address.local1050 = getelementptr i8, ptr addrspace(3) %address.local1027, i64 4096, !dbg !29
  %v1222 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %v1223 = bitcast <4 x i32> %v1222 to <8 x half>, !dbg !29
  %address.local1054 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 4352, !dbg !29
  %v1224 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %v1225 = bitcast <4 x i32> %v1224 to <8 x half>, !dbg !29
  %address.local1058 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 4608, !dbg !29
  %v1226 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %v1227 = bitcast <4 x i32> %v1226 to <8 x half>, !dbg !29
  %address.local1062 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 4864, !dbg !29
  %v1228 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #10, !dbg !29
  %v1229 = bitcast <4 x i32> %v1228 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local1063 = shl nuw nsw i64 %address.local1028, 3, !dbg !8
  %address.local1064 = and i64 %address.local1063, 1536, !dbg !8
  %address.local1065 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1064, !dbg !8
  %address.local1066 = ptrtoint ptr addrspace(3) %address.local1065 to i64, !dbg !8
  %address.local1067 = trunc i64 %address.local1066 to i32, !dbg !8
  %v1236 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1067), !dbg !17
  %address.local1068 = shl nuw nsw i64 %address.local1028, 5, !dbg !17
  %address.local1069 = and i64 %address.local1068, 8064, !dbg !17
  %address.local1071 = and i64 %address.local1063, 24, !dbg !17
  %address.local1072 = or disjoint i64 %address.local1069, %address.local1071, !dbg !17
  %225 = or disjoint i64 %v604.idx28, %address.local1072, !dbg !17
  %226 = trunc i64 %225 to i32, !dbg !17
  %.tr33 = shl i32 %226, 1, !dbg !17
  %dma.buffer.offset1714 = or i32 %.tr33, 64, !dbg !17
  %dma.buffer.lds1716 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1236), !dbg !17
  %v1249 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1716, i32 %dma.buffer.offset1714, <4 x i32> %dma.buffer.rsrc1721) #9, !dbg !17
  %address.local1074 = or disjoint i64 %address.local1028, 256, !dbg !8
  %address.local1075 = shl nuw nsw i64 %address.local1074, 3, !dbg !8
  %address.local1076 = and i64 %address.local1075, 3584, !dbg !8
  %address.local1077 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1076, !dbg !8
  %address.local1078 = ptrtoint ptr addrspace(3) %address.local1077 to i64, !dbg !8
  %address.local1079 = trunc i64 %address.local1078 to i32, !dbg !8
  %v1250 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1079), !dbg !17
  %address.local1080 = shl nuw nsw i64 %address.local1074, 5, !dbg !17
  %address.local1081 = and i64 %address.local1080, 16256, !dbg !17
  %address.local1084 = or disjoint i64 %address.local1081, %address.local1071, !dbg !17
  %227 = or disjoint i64 %v604.idx28, %address.local1084, !dbg !17
  %228 = trunc i64 %227 to i32, !dbg !17
  %.tr36 = shl i32 %228, 1, !dbg !17
  %dma.buffer.offset1711 = or i32 %.tr36, 64, !dbg !17
  %dma.buffer.lds1713 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1250), !dbg !17
  %v1263 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1713, i32 %dma.buffer.offset1711, <4 x i32> %dma.buffer.rsrc1721) #9, !dbg !17
  %v1264 = shufflevector <8 x half> %v1215, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1265 = extractvalue [2 x <8 x float>] %v20653, 0, !dbg !24
  %v1266 = shufflevector <8 x float> %v1265, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1267 = fmul <4 x float> %v1266, %v970, !dbg !24
  %v1268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1264, <4 x float> %v1267) #8, !dbg !29
  %v1269 = shufflevector <8 x half> %v1217, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1269, <4 x float> %v1268) #8, !dbg !29
  %v1271 = shufflevector <8 x half> %v1219, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1271, <4 x float> %v1270) #8, !dbg !29
  %v1273 = shufflevector <8 x half> %v1221, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1273, <4 x float> %v1272) #8, !dbg !29
  %v1275 = shufflevector <8 x half> %v1223, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1275, <4 x float> %v1274) #8, !dbg !29
  %v1277 = shufflevector <8 x half> %v1225, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1277, <4 x float> %v1276) #8, !dbg !29
  %v1279 = shufflevector <8 x half> %v1227, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1279, <4 x float> %v1278) #8, !dbg !29
  %v1281 = shufflevector <8 x half> %v1229, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1281, <4 x float> %v1280) #8, !dbg !29
  %v1283 = shufflevector <8 x half> %v1215, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1284 = shufflevector <8 x float> %v1265, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1285 = fmul <4 x float> %v1284, %v970, !dbg !24
  %v1286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1283, <4 x float> %v1285) #8, !dbg !29
  %v1287 = shufflevector <8 x half> %v1217, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1287, <4 x float> %v1286) #8, !dbg !29
  %v1289 = shufflevector <8 x half> %v1219, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1289, <4 x float> %v1288) #8, !dbg !29
  %v1291 = shufflevector <8 x half> %v1221, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1291, <4 x float> %v1290) #8, !dbg !29
  %v1293 = shufflevector <8 x half> %v1223, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1293, <4 x float> %v1292) #8, !dbg !29
  %v1295 = shufflevector <8 x half> %v1225, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1295, <4 x float> %v1294) #8, !dbg !29
  %v1297 = shufflevector <8 x half> %v1227, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1297, <4 x float> %v1296) #8, !dbg !29
  %v1299 = shufflevector <8 x half> %v1229, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1299, <4 x float> %v1298) #8, !dbg !29
  %v1301 = extractvalue [2 x <8 x float>] %v20653, 1, !dbg !24
  %v1302 = shufflevector <8 x float> %v1301, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1303 = fmul <4 x float> %v1302, %v978, !dbg !24
  %v1304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1264, <4 x float> %v1303) #8, !dbg !29
  %v1305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1269, <4 x float> %v1304) #8, !dbg !29
  %v1306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1271, <4 x float> %v1305) #8, !dbg !29
  %v1307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1273, <4 x float> %v1306) #8, !dbg !29
  %v1308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1275, <4 x float> %v1307) #8, !dbg !29
  %v1309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1277, <4 x float> %v1308) #8, !dbg !29
  %v1310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1279, <4 x float> %v1309) #8, !dbg !29
  %v1311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1281, <4 x float> %v1310) #8, !dbg !29
  %v1312 = shufflevector <8 x float> %v1301, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1313 = fmul <4 x float> %v1312, %v978, !dbg !24
  %v1314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1283, <4 x float> %v1313) #8, !dbg !29
  %v1315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1287, <4 x float> %v1314) #8, !dbg !29
  %v1316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1289, <4 x float> %v1315) #8, !dbg !29
  %v1317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1291, <4 x float> %v1316) #8, !dbg !29
  %v1318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1293, <4 x float> %v1317) #8, !dbg !29
  %v1319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1295, <4 x float> %v1318) #8, !dbg !29
  %v1320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1297, <4 x float> %v1319) #8, !dbg !29
  %v1321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1299, <4 x float> %v1320) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %v1322 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #10, !dbg !29
  %v1323 = bitcast <4 x i32> %v1322 to <8 x half>, !dbg !29
  %v1324 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #10, !dbg !29
  %v1325 = bitcast <4 x i32> %v1324 to <8 x half>, !dbg !29
  %v1326 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #10, !dbg !29
  %v1327 = bitcast <4 x i32> %v1326 to <8 x half>, !dbg !29
  %v1328 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #10, !dbg !29
  %v1329 = bitcast <4 x i32> %v1328 to <8 x half>, !dbg !29
  %v1330 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %v1331 = bitcast <4 x i32> %v1330 to <8 x half>, !dbg !29
  %v1332 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %v1333 = bitcast <4 x i32> %v1332 to <8 x half>, !dbg !29
  %v1334 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %v1335 = bitcast <4 x i32> %v1334 to <8 x half>, !dbg !29
  %v1336 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #10, !dbg !29
  %v1337 = bitcast <4 x i32> %v1336 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %dma.buffer.offset1708 = or i32 %.tr33, 128, !dbg !17
  %v1353 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1716, i32 %dma.buffer.offset1708, <4 x i32> %dma.buffer.rsrc1721) #9, !dbg !17
  %dma.buffer.offset1705 = or i32 %.tr36, 128, !dbg !17
  %v1367 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1713, i32 %dma.buffer.offset1705, <4 x i32> %dma.buffer.rsrc1721) #9, !dbg !17
  %v1368 = shufflevector <8 x half> %v1323, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1369 = extractvalue [2 x <8 x float>] %v20956, 0, !dbg !24
  %v1370 = shufflevector <8 x float> %v1369, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1371 = fmul <4 x float> %v1370, %v970, !dbg !24
  %v1372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1368, <4 x float> %v1371) #8, !dbg !29
  %v1373 = shufflevector <8 x half> %v1325, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1373, <4 x float> %v1372) #8, !dbg !29
  %v1375 = shufflevector <8 x half> %v1327, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1376 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1375, <4 x float> %v1374) #8, !dbg !29
  %v1377 = shufflevector <8 x half> %v1329, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1377, <4 x float> %v1376) #8, !dbg !29
  %v1379 = shufflevector <8 x half> %v1331, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1379, <4 x float> %v1378) #8, !dbg !29
  %v1381 = shufflevector <8 x half> %v1333, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1381, <4 x float> %v1380) #8, !dbg !29
  %v1383 = shufflevector <8 x half> %v1335, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1383, <4 x float> %v1382) #8, !dbg !29
  %v1385 = shufflevector <8 x half> %v1337, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1385, <4 x float> %v1384) #8, !dbg !29
  %v1387 = shufflevector <8 x half> %v1323, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1388 = shufflevector <8 x float> %v1369, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1389 = fmul <4 x float> %v1388, %v970, !dbg !24
  %v1390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1387, <4 x float> %v1389) #8, !dbg !29
  %v1391 = shufflevector <8 x half> %v1325, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1391, <4 x float> %v1390) #8, !dbg !29
  %v1393 = shufflevector <8 x half> %v1327, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1393, <4 x float> %v1392) #8, !dbg !29
  %v1395 = shufflevector <8 x half> %v1329, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1396 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1395, <4 x float> %v1394) #8, !dbg !29
  %v1397 = shufflevector <8 x half> %v1331, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1397, <4 x float> %v1396) #8, !dbg !29
  %v1399 = shufflevector <8 x half> %v1333, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1399, <4 x float> %v1398) #8, !dbg !29
  %v1401 = shufflevector <8 x half> %v1335, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1401, <4 x float> %v1400) #8, !dbg !29
  %v1403 = shufflevector <8 x half> %v1337, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1403, <4 x float> %v1402) #8, !dbg !29
  %v1405 = extractvalue [2 x <8 x float>] %v20956, 1, !dbg !24
  %v1406 = shufflevector <8 x float> %v1405, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1407 = fmul <4 x float> %v1406, %v978, !dbg !24
  %v1408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1368, <4 x float> %v1407) #8, !dbg !29
  %v1409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1373, <4 x float> %v1408) #8, !dbg !29
  %v1410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1375, <4 x float> %v1409) #8, !dbg !29
  %v1411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1377, <4 x float> %v1410) #8, !dbg !29
  %v1412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1379, <4 x float> %v1411) #8, !dbg !29
  %v1413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1381, <4 x float> %v1412) #8, !dbg !29
  %v1414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1383, <4 x float> %v1413) #8, !dbg !29
  %v1415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1385, <4 x float> %v1414) #8, !dbg !29
  %v1416 = shufflevector <8 x float> %v1405, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1417 = fmul <4 x float> %v1416, %v978, !dbg !24
  %v1418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1387, <4 x float> %v1417) #8, !dbg !29
  %v1419 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1391, <4 x float> %v1418) #8, !dbg !29
  %v1420 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1393, <4 x float> %v1419) #8, !dbg !29
  %v1421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1395, <4 x float> %v1420) #8, !dbg !29
  %v1422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1397, <4 x float> %v1421) #8, !dbg !29
  %v1423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1399, <4 x float> %v1422) #8, !dbg !29
  %v1424 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1401, <4 x float> %v1423) #8, !dbg !29
  %v1425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1403, <4 x float> %v1424) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %v1426 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #10, !dbg !29
  %v1427 = bitcast <4 x i32> %v1426 to <8 x half>, !dbg !29
  %v1428 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #10, !dbg !29
  %v1429 = bitcast <4 x i32> %v1428 to <8 x half>, !dbg !29
  %v1430 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #10, !dbg !29
  %v1431 = bitcast <4 x i32> %v1430 to <8 x half>, !dbg !29
  %v1432 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #10, !dbg !29
  %v1433 = bitcast <4 x i32> %v1432 to <8 x half>, !dbg !29
  %v1434 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %v1435 = bitcast <4 x i32> %v1434 to <8 x half>, !dbg !29
  %v1436 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %v1437 = bitcast <4 x i32> %v1436 to <8 x half>, !dbg !29
  %v1438 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %v1439 = bitcast <4 x i32> %v1438 to <8 x half>, !dbg !29
  %v1440 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #10, !dbg !29
  %v1441 = bitcast <4 x i32> %v1440 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %229 = or disjoint i64 %v604.idx28, 96, !dbg !17
  %230 = or disjoint i64 %address.local1072, %229, !dbg !17
  %.tr45 = trunc i64 %230 to i32, !dbg !17
  %dma.buffer.offset1702 = shl i32 %.tr45, 1, !dbg !17
  %v1457 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1716, i32 %dma.buffer.offset1702, <4 x i32> %dma.buffer.rsrc1721) #9, !dbg !17
  %231 = or disjoint i64 %address.local1084, %229, !dbg !17
  %.tr48 = trunc i64 %231 to i32, !dbg !17
  %dma.buffer.offset = shl i32 %.tr48, 1, !dbg !17
  %v1471 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1713, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1721) #9, !dbg !17
  %v1472 = shufflevector <8 x half> %v1427, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1473 = extractvalue [2 x <8 x float>] %v21057, 0, !dbg !24
  %v1474 = shufflevector <8 x float> %v1473, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1475 = fmul <4 x float> %v1474, %v970, !dbg !24
  %v1476 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1472, <4 x float> %v1475) #8, !dbg !29
  %v1477 = shufflevector <8 x half> %v1429, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1477, <4 x float> %v1476) #8, !dbg !29
  %v1479 = shufflevector <8 x half> %v1431, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1480 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1479, <4 x float> %v1478) #8, !dbg !29
  %v1481 = shufflevector <8 x half> %v1433, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1482 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1481, <4 x float> %v1480) #8, !dbg !29
  %v1483 = shufflevector <8 x half> %v1435, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1484 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1483, <4 x float> %v1482) #8, !dbg !29
  %v1485 = shufflevector <8 x half> %v1437, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1486 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1485, <4 x float> %v1484) #8, !dbg !29
  %v1487 = shufflevector <8 x half> %v1439, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1488 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1487, <4 x float> %v1486) #8, !dbg !29
  %v1489 = shufflevector <8 x half> %v1441, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1490 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1489, <4 x float> %v1488) #8, !dbg !29
  %v1491 = shufflevector <8 x half> %v1427, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1492 = shufflevector <8 x float> %v1473, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1493 = fmul <4 x float> %v1492, %v970, !dbg !24
  %v1494 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1491, <4 x float> %v1493) #8, !dbg !29
  %v1495 = shufflevector <8 x half> %v1429, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1496 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1495, <4 x float> %v1494) #8, !dbg !29
  %v1497 = shufflevector <8 x half> %v1431, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1498 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1497, <4 x float> %v1496) #8, !dbg !29
  %v1499 = shufflevector <8 x half> %v1433, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1500 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1499, <4 x float> %v1498) #8, !dbg !29
  %v1501 = shufflevector <8 x half> %v1435, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1502 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1501, <4 x float> %v1500) #8, !dbg !29
  %v1503 = shufflevector <8 x half> %v1437, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1504 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1503, <4 x float> %v1502) #8, !dbg !29
  %v1505 = shufflevector <8 x half> %v1439, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1506 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1505, <4 x float> %v1504) #8, !dbg !29
  %v1507 = shufflevector <8 x half> %v1441, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1508 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1507, <4 x float> %v1506) #8, !dbg !29
  %v1509 = extractvalue [2 x <8 x float>] %v21057, 1, !dbg !24
  %v1510 = shufflevector <8 x float> %v1509, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1511 = fmul <4 x float> %v1510, %v978, !dbg !24
  %v1512 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1472, <4 x float> %v1511) #8, !dbg !29
  %v1513 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1477, <4 x float> %v1512) #8, !dbg !29
  %v1514 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1479, <4 x float> %v1513) #8, !dbg !29
  %v1515 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1481, <4 x float> %v1514) #8, !dbg !29
  %v1516 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1483, <4 x float> %v1515) #8, !dbg !29
  %v1517 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1485, <4 x float> %v1516) #8, !dbg !29
  %v1518 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1487, <4 x float> %v1517) #8, !dbg !29
  %v1519 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1489, <4 x float> %v1518) #8, !dbg !29
  %v1520 = shufflevector <8 x float> %v1509, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1521 = fmul <4 x float> %v1520, %v978, !dbg !24
  %v1522 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1491, <4 x float> %v1521) #8, !dbg !29
  %v1523 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1495, <4 x float> %v1522) #8, !dbg !29
  %v1524 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1497, <4 x float> %v1523) #8, !dbg !29
  %v1525 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1499, <4 x float> %v1524) #8, !dbg !29
  %v1526 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1501, <4 x float> %v1525) #8, !dbg !29
  %v1527 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1503, <4 x float> %v1526) #8, !dbg !29
  %v1528 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1505, <4 x float> %v1527) #8, !dbg !29
  %v1529 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1507, <4 x float> %v1528) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %v1530 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #10, !dbg !29
  %v1531 = bitcast <4 x i32> %v1530 to <8 x half>, !dbg !29
  %v1532 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #10, !dbg !29
  %v1533 = bitcast <4 x i32> %v1532 to <8 x half>, !dbg !29
  %v1534 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #10, !dbg !29
  %v1535 = bitcast <4 x i32> %v1534 to <8 x half>, !dbg !29
  %v1536 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #10, !dbg !29
  %v1537 = bitcast <4 x i32> %v1536 to <8 x half>, !dbg !29
  %v1538 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %v1539 = bitcast <4 x i32> %v1538 to <8 x half>, !dbg !29
  %v1540 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %v1541 = bitcast <4 x i32> %v1540 to <8 x half>, !dbg !29
  %v1542 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %v1543 = bitcast <4 x i32> %v1542 to <8 x half>, !dbg !29
  %v1544 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #10, !dbg !29
  %v1545 = bitcast <4 x i32> %v1544 to <8 x half>, !dbg !29
  %v1546 = shufflevector <8 x half> %v1531, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1547 = extractvalue [2 x <8 x float>] %v21158, 0, !dbg !24
  %v1548 = shufflevector <8 x float> %v1547, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1549 = fmul <4 x float> %v1548, %v970, !dbg !24
  %v1550 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1546, <4 x float> %v1549) #8, !dbg !29
  %v1551 = shufflevector <8 x half> %v1533, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1552 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1551, <4 x float> %v1550) #8, !dbg !29
  %v1553 = shufflevector <8 x half> %v1535, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1554 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1553, <4 x float> %v1552) #8, !dbg !29
  %v1555 = shufflevector <8 x half> %v1537, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1556 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1555, <4 x float> %v1554) #8, !dbg !29
  %v1557 = shufflevector <8 x half> %v1539, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1558 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1557, <4 x float> %v1556) #8, !dbg !29
  %v1559 = shufflevector <8 x half> %v1541, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1560 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1559, <4 x float> %v1558) #8, !dbg !29
  %v1561 = shufflevector <8 x half> %v1543, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1562 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1561, <4 x float> %v1560) #8, !dbg !29
  %v1563 = shufflevector <8 x half> %v1545, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %v1564 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1563, <4 x float> %v1562) #8, !dbg !29
  %v1565 = shufflevector <8 x half> %v1531, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1566 = shufflevector <8 x float> %v1547, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1567 = fmul <4 x float> %v1566, %v970, !dbg !24
  %v1568 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1565, <4 x float> %v1567) #8, !dbg !29
  %v1569 = shufflevector <8 x half> %v1533, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1570 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1569, <4 x float> %v1568) #8, !dbg !29
  %v1571 = shufflevector <8 x half> %v1535, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1572 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1571, <4 x float> %v1570) #8, !dbg !29
  %v1573 = shufflevector <8 x half> %v1537, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1574 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1573, <4 x float> %v1572) #8, !dbg !29
  %v1575 = shufflevector <8 x half> %v1539, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1576 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1575, <4 x float> %v1574) #8, !dbg !29
  %v1577 = shufflevector <8 x half> %v1541, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1578 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1577, <4 x float> %v1576) #8, !dbg !29
  %v1579 = shufflevector <8 x half> %v1543, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1580 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1579, <4 x float> %v1578) #8, !dbg !29
  %v1581 = shufflevector <8 x half> %v1545, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %v1582 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1581, <4 x float> %v1580) #8, !dbg !29
  %v1583 = extractvalue [2 x <8 x float>] %v21158, 1, !dbg !24
  %v1584 = shufflevector <8 x float> %v1583, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %v1585 = fmul <4 x float> %v1584, %v978, !dbg !24
  %v1586 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1546, <4 x float> %v1585) #8, !dbg !29
  %v1587 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1551, <4 x float> %v1586) #8, !dbg !29
  %v1588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1553, <4 x float> %v1587) #8, !dbg !29
  %v1589 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1555, <4 x float> %v1588) #8, !dbg !29
  %v1590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1557, <4 x float> %v1589) #8, !dbg !29
  %v1591 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1559, <4 x float> %v1590) #8, !dbg !29
  %v1592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1561, <4 x float> %v1591) #8, !dbg !29
  %v1593 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1563, <4 x float> %v1592) #8, !dbg !29
  %v1594 = shufflevector <8 x float> %v1583, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %v1595 = fmul <4 x float> %v1594, %v978, !dbg !24
  %v1596 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1565, <4 x float> %v1595) #8, !dbg !29
  %v1597 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1569, <4 x float> %v1596) #8, !dbg !29
  %v1598 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1571, <4 x float> %v1597) #8, !dbg !29
  %v1599 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1573, <4 x float> %v1598) #8, !dbg !29
  %v1600 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1575, <4 x float> %v1599) #8, !dbg !29
  %v1601 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1577, <4 x float> %v1600) #8, !dbg !29
  %v1602 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1579, <4 x float> %v1601) #8, !dbg !29
  %v1603 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1581, <4 x float> %v1602) #8, !dbg !29
  %v1604 = shufflevector <4 x float> %v1282, <4 x float> %v1300, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1605 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1604, 0, !dbg !11
  %v1606 = shufflevector <4 x float> %v1311, <4 x float> %v1321, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1607 = insertvalue [2 x <8 x float>] %v1605, <8 x float> %v1606, 1, !dbg !11
  %v1608 = insertvalue [2 x <1 x float>] poison, <1 x float> %v934, 0, !dbg !11
  %v1609 = insertvalue [2 x <1 x float>] %v1608, <1 x float> %v947, 1, !dbg !11
  %v1610 = shufflevector <4 x float> %v1386, <4 x float> %v1404, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1611 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1610, 0, !dbg !11
  %v1612 = shufflevector <4 x float> %v1415, <4 x float> %v1425, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1613 = insertvalue [2 x <8 x float>] %v1611, <8 x float> %v1612, 1, !dbg !11
  %v1614 = shufflevector <4 x float> %v1490, <4 x float> %v1508, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1615 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1614, 0, !dbg !11
  %v1616 = shufflevector <4 x float> %v1519, <4 x float> %v1529, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1617 = insertvalue [2 x <8 x float>] %v1615, <8 x float> %v1616, 1, !dbg !11
  %v1618 = shufflevector <4 x float> %v1564, <4 x float> %v1582, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1619 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1618, 0, !dbg !11
  %v1620 = shufflevector <4 x float> %v1593, <4 x float> %v1603, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %v1621 = insertvalue [2 x <8 x float>] %v1619, <8 x float> %v1620, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %v1622 = add nuw nsw i64 %v20552, 1, !dbg !11
  %v212 = icmp slt i64 %v1622, %v191, !dbg !11
  br i1 %v212, label %v213, label %v1623, !dbg !11

v1623:                                            ; preds = %v845, %.v1623_crit_edge
  %v1652.pre-phi = phi i32 [ %.pre70, %.v1623_crit_edge ], [ %v924, %v845 ], !dbg !11
  %v1636.pre-phi = phi i32 [ %.pre69, %.v1623_crit_edge ], [ %v916, %v845 ], !dbg !11
  %v1635.pre-phi = phi i32 [ %.pre68, %.v1623_crit_edge ], [ %v915, %v845 ], !dbg !11
  %v1633.pre-phi = phi i32 [ %.pre66, %.v1623_crit_edge ], [ %v913, %v845 ], !dbg !11
  %v206.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.v1623_crit_edge ], [ %v1607, %v845 ], !dbg !11
  %v207.lcssa = phi [2 x <1 x float>] [ zeroinitializer, %.v1623_crit_edge ], [ %v1213, %v845 ], !dbg !11
  %v209.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.v1623_crit_edge ], [ %v1613, %v845 ], !dbg !11
  %v210.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.v1623_crit_edge ], [ %v1617, %v845 ], !dbg !11
  %v211.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.v1623_crit_edge ], [ %v1621, %v845 ], !dbg !11
  %v1624 = and i32 %v31, 16, !dbg !11
  %.not = icmp eq i32 %v1624, 0, !dbg !11
  %v1625 = and i32 %v31, 32, !dbg !11
  %.not36 = icmp eq i32 %v1625, 0, !dbg !11
  %v1626 = extractvalue [2 x <8 x float>] %v211.lcssa, 0, !dbg !11
  %v1627 = extractelement <8 x float> %v1626, i64 0, !dbg !11
  %v1628 = extractelement <8 x float> %v1626, i64 1, !dbg !11
  %v1629 = extractelement <8 x float> %v1626, i64 2, !dbg !11
  %v1630 = extractelement <8 x float> %v1626, i64 3, !dbg !11
  %v1631 = select i1 %.not, float %v1628, float %v1627, !dbg !11
  %v1637 = icmp samesign ult i32 %v1635.pre-phi, %v1636.pre-phi, !dbg !11
  %v1638 = select i1 %v1637, i32 %v1635.pre-phi, i32 %v1633.pre-phi, !dbg !11
  %v1639 = shl nuw nsw i32 %v1638, 2, !dbg !11
  %v1640 = bitcast float %v1631 to i32, !dbg !11
  %v1641 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1640), !dbg !11
  %v1642 = bitcast i32 %v1641 to float, !dbg !11
  %v1643 = select i1 %.not, float %v1627, float %v1642, !dbg !11
  %v1644 = select i1 %.not, float %v1642, float %v1628, !dbg !11
  %v1645 = select i1 %.not, float %v1630, float %v1629, !dbg !11
  %v1646 = bitcast float %v1645 to i32, !dbg !11
  %v1647 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1646), !dbg !11
  %v1648 = bitcast i32 %v1647 to float, !dbg !11
  %v1649 = select i1 %.not, float %v1629, float %v1648, !dbg !11
  %v1650 = select i1 %.not, float %v1648, float %v1630, !dbg !11
  %v1651 = select i1 %.not36, float %v1649, float %v1643, !dbg !11
  %v1653 = icmp samesign ult i32 %v1652.pre-phi, %v1636.pre-phi, !dbg !11
  %v1654 = select i1 %v1653, i32 %v1652.pre-phi, i32 %v1633.pre-phi, !dbg !11
  %v1655 = shl nuw nsw i32 %v1654, 2, !dbg !11
  %v1656 = bitcast float %v1651 to i32, !dbg !11
  %v1657 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1656), !dbg !11
  %v1658 = bitcast i32 %v1657 to float, !dbg !11
  %v1661 = select i1 %.not36, float %v1650, float %v1644, !dbg !11
  %v1662 = bitcast float %v1661 to i32, !dbg !11
  %v1663 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1662), !dbg !11
  %v1664 = bitcast i32 %v1663 to float, !dbg !11
  %232 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %233 = shufflevector <4 x i1> %232, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %234 = insertelement <4 x float> poison, float %v1643, i64 0, !dbg !11
  %235 = insertelement <4 x float> %234, float %v1644, i64 1, !dbg !11
  %236 = insertelement <4 x float> %235, float %v1658, i64 2, !dbg !11
  %237 = insertelement <4 x float> %236, float %v1664, i64 3, !dbg !11
  %238 = insertelement <4 x float> poison, float %v1658, i64 0, !dbg !11
  %239 = insertelement <4 x float> %238, float %v1664, i64 1, !dbg !11
  %240 = insertelement <4 x float> %239, float %v1649, i64 2, !dbg !11
  %241 = insertelement <4 x float> %240, float %v1650, i64 3, !dbg !11
  %242 = select <4 x i1> %233, <4 x float> %237, <4 x float> %241, !dbg !11
  %v1667 = extractvalue [2 x <8 x float>] %v210.lcssa, 0, !dbg !11
  %v1668 = extractelement <8 x float> %v1667, i64 0, !dbg !11
  %v1669 = extractelement <8 x float> %v1667, i64 1, !dbg !11
  %v1670 = extractelement <8 x float> %v1667, i64 2, !dbg !11
  %v1671 = extractelement <8 x float> %v1667, i64 3, !dbg !11
  %v1672 = select i1 %.not, float %v1669, float %v1668, !dbg !11
  %v1673 = bitcast float %v1672 to i32, !dbg !11
  %v1674 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1673), !dbg !11
  %v1675 = bitcast i32 %v1674 to float, !dbg !11
  %v1676 = select i1 %.not, float %v1668, float %v1675, !dbg !11
  %v1677 = select i1 %.not, float %v1675, float %v1669, !dbg !11
  %v1678 = select i1 %.not, float %v1671, float %v1670, !dbg !11
  %v1679 = bitcast float %v1678 to i32, !dbg !11
  %v1680 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1679), !dbg !11
  %v1681 = bitcast i32 %v1680 to float, !dbg !11
  %v1682 = select i1 %.not, float %v1670, float %v1681, !dbg !11
  %v1683 = select i1 %.not, float %v1681, float %v1671, !dbg !11
  %v1684 = select i1 %.not36, float %v1682, float %v1676, !dbg !11
  %v1685 = bitcast float %v1684 to i32, !dbg !11
  %v1686 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1685), !dbg !11
  %v1687 = bitcast i32 %v1686 to float, !dbg !11
  %v1690 = select i1 %.not36, float %v1683, float %v1677, !dbg !11
  %v1691 = bitcast float %v1690 to i32, !dbg !11
  %v1692 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1691), !dbg !11
  %v1693 = bitcast i32 %v1692 to float, !dbg !11
  %243 = insertelement <4 x float> poison, float %v1676, i64 0, !dbg !11
  %244 = insertelement <4 x float> %243, float %v1677, i64 1, !dbg !11
  %245 = insertelement <4 x float> %244, float %v1687, i64 2, !dbg !11
  %246 = insertelement <4 x float> %245, float %v1693, i64 3, !dbg !11
  %247 = insertelement <4 x float> poison, float %v1687, i64 0, !dbg !11
  %248 = insertelement <4 x float> %247, float %v1693, i64 1, !dbg !11
  %249 = insertelement <4 x float> %248, float %v1682, i64 2, !dbg !11
  %250 = insertelement <4 x float> %249, float %v1683, i64 3, !dbg !11
  %251 = select <4 x i1> %233, <4 x float> %246, <4 x float> %250, !dbg !11
  %v1696 = extractvalue [2 x <8 x float>] %v209.lcssa, 0, !dbg !11
  %v1697 = extractelement <8 x float> %v1696, i64 0, !dbg !11
  %v1698 = extractelement <8 x float> %v1696, i64 1, !dbg !11
  %v1699 = extractelement <8 x float> %v1696, i64 2, !dbg !11
  %v1700 = extractelement <8 x float> %v1696, i64 3, !dbg !11
  %v1701 = select i1 %.not, float %v1698, float %v1697, !dbg !11
  %v1702 = bitcast float %v1701 to i32, !dbg !11
  %v1703 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1702), !dbg !11
  %v1704 = bitcast i32 %v1703 to float, !dbg !11
  %v1705 = select i1 %.not, float %v1697, float %v1704, !dbg !11
  %v1706 = select i1 %.not, float %v1704, float %v1698, !dbg !11
  %v1707 = select i1 %.not, float %v1700, float %v1699, !dbg !11
  %v1708 = bitcast float %v1707 to i32, !dbg !11
  %v1709 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1708), !dbg !11
  %v1710 = bitcast i32 %v1709 to float, !dbg !11
  %v1711 = select i1 %.not, float %v1699, float %v1710, !dbg !11
  %v1712 = select i1 %.not, float %v1710, float %v1700, !dbg !11
  %v1713 = select i1 %.not36, float %v1711, float %v1705, !dbg !11
  %v1714 = bitcast float %v1713 to i32, !dbg !11
  %v1715 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1714), !dbg !11
  %v1716 = bitcast i32 %v1715 to float, !dbg !11
  %v1719 = select i1 %.not36, float %v1712, float %v1706, !dbg !11
  %v1720 = bitcast float %v1719 to i32, !dbg !11
  %v1721 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1720), !dbg !11
  %v1722 = bitcast i32 %v1721 to float, !dbg !11
  %252 = insertelement <4 x float> poison, float %v1705, i64 0, !dbg !11
  %253 = insertelement <4 x float> %252, float %v1706, i64 1, !dbg !11
  %254 = insertelement <4 x float> %253, float %v1716, i64 2, !dbg !11
  %255 = insertelement <4 x float> %254, float %v1722, i64 3, !dbg !11
  %256 = insertelement <4 x float> poison, float %v1716, i64 0, !dbg !11
  %257 = insertelement <4 x float> %256, float %v1722, i64 1, !dbg !11
  %258 = insertelement <4 x float> %257, float %v1711, i64 2, !dbg !11
  %259 = insertelement <4 x float> %258, float %v1712, i64 3, !dbg !11
  %260 = select <4 x i1> %233, <4 x float> %255, <4 x float> %259, !dbg !11
  %v1725 = extractvalue [2 x <8 x float>] %v206.lcssa, 0, !dbg !11
  %v1726 = extractelement <8 x float> %v1725, i64 0, !dbg !11
  %v1727 = extractelement <8 x float> %v1725, i64 1, !dbg !11
  %v1728 = extractelement <8 x float> %v1725, i64 2, !dbg !11
  %v1729 = extractelement <8 x float> %v1725, i64 3, !dbg !11
  %v1730 = select i1 %.not, float %v1727, float %v1726, !dbg !11
  %v1731 = bitcast float %v1730 to i32, !dbg !11
  %v1732 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1731), !dbg !11
  %v1733 = bitcast i32 %v1732 to float, !dbg !11
  %v1734 = select i1 %.not, float %v1726, float %v1733, !dbg !11
  %v1735 = select i1 %.not, float %v1733, float %v1727, !dbg !11
  %v1736 = select i1 %.not, float %v1729, float %v1728, !dbg !11
  %v1737 = bitcast float %v1736 to i32, !dbg !11
  %v1738 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1737), !dbg !11
  %v1739 = bitcast i32 %v1738 to float, !dbg !11
  %v1740 = select i1 %.not, float %v1728, float %v1739, !dbg !11
  %v1741 = select i1 %.not, float %v1739, float %v1729, !dbg !11
  %v1742 = select i1 %.not36, float %v1740, float %v1734, !dbg !11
  %v1743 = bitcast float %v1742 to i32, !dbg !11
  %v1744 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1743), !dbg !11
  %v1745 = select i1 %.not36, float %v1741, float %v1735, !dbg !11
  %v1746 = bitcast float %v1745 to i32, !dbg !11
  %v1747 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1746), !dbg !11
  %denom.exit.r0.packed = extractvalue [2 x <1 x float>] %v207.lcssa, 0, !dbg !30
  %denom.exit.r0.local = extractelement <1 x float> %denom.exit.r0.packed, i64 0, !dbg !30
  %denom.exit.r0.bits16 = bitcast float %denom.exit.r0.local to i32, !dbg !30
  %denom.exit.r0.peer16 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %denom.exit.r0.bits16), !dbg !30
  %denom.exit.r0.float16 = bitcast i32 %denom.exit.r0.peer16 to float, !dbg !30
  %denom.exit.r0.sum16 = fadd float %denom.exit.r0.local, %denom.exit.r0.float16, !dbg !30
  %denom.exit.r0.bits32 = bitcast float %denom.exit.r0.sum16 to i32, !dbg !30
  %denom.exit.r0.peer32 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %denom.exit.r0.bits32), !dbg !30
  %denom.exit.r0.float32 = bitcast i32 %denom.exit.r0.peer32 to float, !dbg !30
  %denom.exit.r0.sum32 = fadd float %denom.exit.r0.sum16, %denom.exit.r0.float32, !dbg !30
  %v1748 = insertelement <1 x float> poison, float %denom.exit.r0.sum32, i64 0, !dbg !30
  %v1749 = shufflevector <1 x float> %v1748, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %v1750 = extractelement <8 x float> %v1626, i64 4, !dbg !11
  %v1751 = extractelement <8 x float> %v1626, i64 5, !dbg !11
  %v1752 = extractelement <8 x float> %v1626, i64 6, !dbg !11
  %v1753 = extractelement <8 x float> %v1626, i64 7, !dbg !11
  %v1754 = select i1 %.not, float %v1751, float %v1750, !dbg !11
  %v1755 = bitcast float %v1754 to i32, !dbg !11
  %v1756 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1755), !dbg !11
  %v1757 = bitcast i32 %v1756 to float, !dbg !11
  %v1758 = select i1 %.not, float %v1750, float %v1757, !dbg !11
  %v1759 = select i1 %.not, float %v1757, float %v1751, !dbg !11
  %v1760 = select i1 %.not, float %v1753, float %v1752, !dbg !11
  %v1761 = bitcast float %v1760 to i32, !dbg !11
  %v1762 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1761), !dbg !11
  %v1763 = bitcast i32 %v1762 to float, !dbg !11
  %v1764 = select i1 %.not, float %v1752, float %v1763, !dbg !11
  %v1765 = select i1 %.not, float %v1763, float %v1753, !dbg !11
  %v1766 = select i1 %.not36, float %v1764, float %v1758, !dbg !11
  %v1767 = bitcast float %v1766 to i32, !dbg !11
  %v1768 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1767), !dbg !11
  %v1769 = bitcast i32 %v1768 to float, !dbg !11
  %v1772 = select i1 %.not36, float %v1765, float %v1759, !dbg !11
  %v1773 = bitcast float %v1772 to i32, !dbg !11
  %v1774 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1773), !dbg !11
  %v1775 = bitcast i32 %v1774 to float, !dbg !11
  %261 = insertelement <4 x float> poison, float %v1758, i64 0, !dbg !11
  %262 = insertelement <4 x float> %261, float %v1759, i64 1, !dbg !11
  %263 = insertelement <4 x float> %262, float %v1769, i64 2, !dbg !11
  %264 = insertelement <4 x float> %263, float %v1775, i64 3, !dbg !11
  %265 = insertelement <4 x float> poison, float %v1769, i64 0, !dbg !11
  %266 = insertelement <4 x float> %265, float %v1775, i64 1, !dbg !11
  %267 = insertelement <4 x float> %266, float %v1764, i64 2, !dbg !11
  %268 = insertelement <4 x float> %267, float %v1765, i64 3, !dbg !11
  %269 = select <4 x i1> %233, <4 x float> %264, <4 x float> %268, !dbg !11
  %v1778 = extractelement <8 x float> %v1667, i64 4, !dbg !11
  %v1779 = extractelement <8 x float> %v1667, i64 5, !dbg !11
  %v1780 = extractelement <8 x float> %v1667, i64 6, !dbg !11
  %v1781 = extractelement <8 x float> %v1667, i64 7, !dbg !11
  %v1782 = select i1 %.not, float %v1779, float %v1778, !dbg !11
  %v1783 = bitcast float %v1782 to i32, !dbg !11
  %v1784 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1783), !dbg !11
  %v1785 = bitcast i32 %v1784 to float, !dbg !11
  %v1786 = select i1 %.not, float %v1778, float %v1785, !dbg !11
  %v1787 = select i1 %.not, float %v1785, float %v1779, !dbg !11
  %v1788 = select i1 %.not, float %v1781, float %v1780, !dbg !11
  %v1789 = bitcast float %v1788 to i32, !dbg !11
  %v1790 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1789), !dbg !11
  %v1791 = bitcast i32 %v1790 to float, !dbg !11
  %v1792 = select i1 %.not, float %v1780, float %v1791, !dbg !11
  %v1793 = select i1 %.not, float %v1791, float %v1781, !dbg !11
  %v1794 = select i1 %.not36, float %v1792, float %v1786, !dbg !11
  %v1795 = bitcast float %v1794 to i32, !dbg !11
  %v1796 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1795), !dbg !11
  %v1797 = bitcast i32 %v1796 to float, !dbg !11
  %v1800 = select i1 %.not36, float %v1793, float %v1787, !dbg !11
  %v1801 = bitcast float %v1800 to i32, !dbg !11
  %v1802 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1801), !dbg !11
  %v1803 = bitcast i32 %v1802 to float, !dbg !11
  %270 = insertelement <4 x float> poison, float %v1786, i64 0, !dbg !11
  %271 = insertelement <4 x float> %270, float %v1787, i64 1, !dbg !11
  %272 = insertelement <4 x float> %271, float %v1797, i64 2, !dbg !11
  %273 = insertelement <4 x float> %272, float %v1803, i64 3, !dbg !11
  %274 = insertelement <4 x float> poison, float %v1797, i64 0, !dbg !11
  %275 = insertelement <4 x float> %274, float %v1803, i64 1, !dbg !11
  %276 = insertelement <4 x float> %275, float %v1792, i64 2, !dbg !11
  %277 = insertelement <4 x float> %276, float %v1793, i64 3, !dbg !11
  %278 = select <4 x i1> %233, <4 x float> %273, <4 x float> %277, !dbg !11
  %v1806 = extractelement <8 x float> %v1696, i64 4, !dbg !11
  %v1807 = extractelement <8 x float> %v1696, i64 5, !dbg !11
  %v1808 = extractelement <8 x float> %v1696, i64 6, !dbg !11
  %v1809 = extractelement <8 x float> %v1696, i64 7, !dbg !11
  %v1810 = select i1 %.not, float %v1807, float %v1806, !dbg !11
  %v1811 = bitcast float %v1810 to i32, !dbg !11
  %v1812 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1811), !dbg !11
  %v1813 = bitcast i32 %v1812 to float, !dbg !11
  %v1814 = select i1 %.not, float %v1806, float %v1813, !dbg !11
  %v1815 = select i1 %.not, float %v1813, float %v1807, !dbg !11
  %v1816 = select i1 %.not, float %v1809, float %v1808, !dbg !11
  %v1817 = bitcast float %v1816 to i32, !dbg !11
  %v1818 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1817), !dbg !11
  %v1819 = bitcast i32 %v1818 to float, !dbg !11
  %v1820 = select i1 %.not, float %v1808, float %v1819, !dbg !11
  %v1821 = select i1 %.not, float %v1819, float %v1809, !dbg !11
  %v1822 = select i1 %.not36, float %v1820, float %v1814, !dbg !11
  %v1823 = bitcast float %v1822 to i32, !dbg !11
  %v1824 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1823), !dbg !11
  %v1825 = bitcast i32 %v1824 to float, !dbg !11
  %v1828 = select i1 %.not36, float %v1821, float %v1815, !dbg !11
  %v1829 = bitcast float %v1828 to i32, !dbg !11
  %v1830 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1829), !dbg !11
  %v1831 = bitcast i32 %v1830 to float, !dbg !11
  %279 = insertelement <4 x float> poison, float %v1814, i64 0, !dbg !11
  %280 = insertelement <4 x float> %279, float %v1815, i64 1, !dbg !11
  %281 = insertelement <4 x float> %280, float %v1825, i64 2, !dbg !11
  %282 = insertelement <4 x float> %281, float %v1831, i64 3, !dbg !11
  %283 = insertelement <4 x float> poison, float %v1825, i64 0, !dbg !11
  %284 = insertelement <4 x float> %283, float %v1831, i64 1, !dbg !11
  %285 = insertelement <4 x float> %284, float %v1820, i64 2, !dbg !11
  %286 = insertelement <4 x float> %285, float %v1821, i64 3, !dbg !11
  %287 = select <4 x i1> %233, <4 x float> %282, <4 x float> %286, !dbg !11
  %v1834 = extractelement <8 x float> %v1725, i64 4, !dbg !11
  %v1835 = extractelement <8 x float> %v1725, i64 5, !dbg !11
  %v1836 = extractelement <8 x float> %v1725, i64 6, !dbg !11
  %v1837 = extractelement <8 x float> %v1725, i64 7, !dbg !11
  %v1838 = select i1 %.not, float %v1835, float %v1834, !dbg !11
  %v1839 = bitcast float %v1838 to i32, !dbg !11
  %v1840 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1839), !dbg !11
  %v1841 = bitcast i32 %v1840 to float, !dbg !11
  %v1842 = select i1 %.not, float %v1834, float %v1841, !dbg !11
  %v1843 = select i1 %.not, float %v1841, float %v1835, !dbg !11
  %v1844 = select i1 %.not, float %v1837, float %v1836, !dbg !11
  %v1845 = bitcast float %v1844 to i32, !dbg !11
  %v1846 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1845), !dbg !11
  %v1847 = bitcast i32 %v1846 to float, !dbg !11
  %v1848 = select i1 %.not, float %v1836, float %v1847, !dbg !11
  %v1849 = select i1 %.not, float %v1847, float %v1837, !dbg !11
  %v1850 = select i1 %.not36, float %v1848, float %v1842, !dbg !11
  %v1851 = bitcast float %v1850 to i32, !dbg !11
  %v1852 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1851), !dbg !11
  %v1853 = select i1 %.not36, float %v1849, float %v1843, !dbg !11
  %v1854 = bitcast float %v1853 to i32, !dbg !11
  %v1855 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1854), !dbg !11
  %v1856 = extractvalue [2 x <8 x float>] %v211.lcssa, 1, !dbg !11
  %v1857 = extractelement <8 x float> %v1856, i64 0, !dbg !11
  %v1858 = extractelement <8 x float> %v1856, i64 1, !dbg !11
  %v1859 = extractelement <8 x float> %v1856, i64 2, !dbg !11
  %v1860 = extractelement <8 x float> %v1856, i64 3, !dbg !11
  %v1861 = select i1 %.not, float %v1858, float %v1857, !dbg !11
  %v1862 = bitcast float %v1861 to i32, !dbg !11
  %v1863 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1862), !dbg !11
  %v1864 = bitcast i32 %v1863 to float, !dbg !11
  %v1865 = select i1 %.not, float %v1857, float %v1864, !dbg !11
  %v1866 = select i1 %.not, float %v1864, float %v1858, !dbg !11
  %v1867 = select i1 %.not, float %v1860, float %v1859, !dbg !11
  %v1868 = bitcast float %v1867 to i32, !dbg !11
  %v1869 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1868), !dbg !11
  %v1870 = bitcast i32 %v1869 to float, !dbg !11
  %v1871 = select i1 %.not, float %v1859, float %v1870, !dbg !11
  %v1872 = select i1 %.not, float %v1870, float %v1860, !dbg !11
  %v1873 = select i1 %.not36, float %v1871, float %v1865, !dbg !11
  %v1874 = bitcast float %v1873 to i32, !dbg !11
  %v1875 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1874), !dbg !11
  %v1876 = bitcast i32 %v1875 to float, !dbg !11
  %v1879 = select i1 %.not36, float %v1872, float %v1866, !dbg !11
  %v1880 = bitcast float %v1879 to i32, !dbg !11
  %v1881 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1880), !dbg !11
  %v1882 = bitcast i32 %v1881 to float, !dbg !11
  %288 = insertelement <4 x float> poison, float %v1865, i64 0, !dbg !11
  %289 = insertelement <4 x float> %288, float %v1866, i64 1, !dbg !11
  %290 = insertelement <4 x float> %289, float %v1876, i64 2, !dbg !11
  %291 = insertelement <4 x float> %290, float %v1882, i64 3, !dbg !11
  %292 = insertelement <4 x float> poison, float %v1876, i64 0, !dbg !11
  %293 = insertelement <4 x float> %292, float %v1882, i64 1, !dbg !11
  %294 = insertelement <4 x float> %293, float %v1871, i64 2, !dbg !11
  %295 = insertelement <4 x float> %294, float %v1872, i64 3, !dbg !11
  %296 = select <4 x i1> %233, <4 x float> %291, <4 x float> %295, !dbg !11
  %v1885 = extractvalue [2 x <8 x float>] %v210.lcssa, 1, !dbg !11
  %v1886 = extractelement <8 x float> %v1885, i64 0, !dbg !11
  %v1887 = extractelement <8 x float> %v1885, i64 1, !dbg !11
  %v1888 = extractelement <8 x float> %v1885, i64 2, !dbg !11
  %v1889 = extractelement <8 x float> %v1885, i64 3, !dbg !11
  %v1890 = select i1 %.not, float %v1887, float %v1886, !dbg !11
  %v1891 = bitcast float %v1890 to i32, !dbg !11
  %v1892 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1891), !dbg !11
  %v1893 = bitcast i32 %v1892 to float, !dbg !11
  %v1894 = select i1 %.not, float %v1886, float %v1893, !dbg !11
  %v1895 = select i1 %.not, float %v1893, float %v1887, !dbg !11
  %v1896 = select i1 %.not, float %v1889, float %v1888, !dbg !11
  %v1897 = bitcast float %v1896 to i32, !dbg !11
  %v1898 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1897), !dbg !11
  %v1899 = bitcast i32 %v1898 to float, !dbg !11
  %v1900 = select i1 %.not, float %v1888, float %v1899, !dbg !11
  %v1901 = select i1 %.not, float %v1899, float %v1889, !dbg !11
  %v1902 = select i1 %.not36, float %v1900, float %v1894, !dbg !11
  %v1903 = bitcast float %v1902 to i32, !dbg !11
  %v1904 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1903), !dbg !11
  %v1905 = bitcast i32 %v1904 to float, !dbg !11
  %v1908 = select i1 %.not36, float %v1901, float %v1895, !dbg !11
  %v1909 = bitcast float %v1908 to i32, !dbg !11
  %v1910 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1909), !dbg !11
  %v1911 = bitcast i32 %v1910 to float, !dbg !11
  %297 = insertelement <4 x float> poison, float %v1894, i64 0, !dbg !11
  %298 = insertelement <4 x float> %297, float %v1895, i64 1, !dbg !11
  %299 = insertelement <4 x float> %298, float %v1905, i64 2, !dbg !11
  %300 = insertelement <4 x float> %299, float %v1911, i64 3, !dbg !11
  %301 = insertelement <4 x float> poison, float %v1905, i64 0, !dbg !11
  %302 = insertelement <4 x float> %301, float %v1911, i64 1, !dbg !11
  %303 = insertelement <4 x float> %302, float %v1900, i64 2, !dbg !11
  %304 = insertelement <4 x float> %303, float %v1901, i64 3, !dbg !11
  %305 = select <4 x i1> %233, <4 x float> %300, <4 x float> %304, !dbg !11
  %v1914 = extractvalue [2 x <8 x float>] %v209.lcssa, 1, !dbg !11
  %v1915 = extractelement <8 x float> %v1914, i64 0, !dbg !11
  %v1916 = extractelement <8 x float> %v1914, i64 1, !dbg !11
  %v1917 = extractelement <8 x float> %v1914, i64 2, !dbg !11
  %v1918 = extractelement <8 x float> %v1914, i64 3, !dbg !11
  %v1919 = select i1 %.not, float %v1916, float %v1915, !dbg !11
  %v1920 = bitcast float %v1919 to i32, !dbg !11
  %v1921 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1920), !dbg !11
  %v1922 = bitcast i32 %v1921 to float, !dbg !11
  %v1923 = select i1 %.not, float %v1915, float %v1922, !dbg !11
  %v1924 = select i1 %.not, float %v1922, float %v1916, !dbg !11
  %v1925 = select i1 %.not, float %v1918, float %v1917, !dbg !11
  %v1926 = bitcast float %v1925 to i32, !dbg !11
  %v1927 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1926), !dbg !11
  %v1928 = bitcast i32 %v1927 to float, !dbg !11
  %v1929 = select i1 %.not, float %v1917, float %v1928, !dbg !11
  %v1930 = select i1 %.not, float %v1928, float %v1918, !dbg !11
  %v1931 = select i1 %.not36, float %v1929, float %v1923, !dbg !11
  %v1932 = bitcast float %v1931 to i32, !dbg !11
  %v1933 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1932), !dbg !11
  %v1934 = bitcast i32 %v1933 to float, !dbg !11
  %v1937 = select i1 %.not36, float %v1930, float %v1924, !dbg !11
  %v1938 = bitcast float %v1937 to i32, !dbg !11
  %v1939 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1938), !dbg !11
  %v1940 = bitcast i32 %v1939 to float, !dbg !11
  %306 = insertelement <4 x float> poison, float %v1923, i64 0, !dbg !11
  %307 = insertelement <4 x float> %306, float %v1924, i64 1, !dbg !11
  %308 = insertelement <4 x float> %307, float %v1934, i64 2, !dbg !11
  %309 = insertelement <4 x float> %308, float %v1940, i64 3, !dbg !11
  %310 = insertelement <4 x float> poison, float %v1934, i64 0, !dbg !11
  %311 = insertelement <4 x float> %310, float %v1940, i64 1, !dbg !11
  %312 = insertelement <4 x float> %311, float %v1929, i64 2, !dbg !11
  %313 = insertelement <4 x float> %312, float %v1930, i64 3, !dbg !11
  %314 = select <4 x i1> %233, <4 x float> %309, <4 x float> %313, !dbg !11
  %v1943 = extractvalue [2 x <8 x float>] %v206.lcssa, 1, !dbg !11
  %v1944 = extractelement <8 x float> %v1943, i64 0, !dbg !11
  %v1945 = extractelement <8 x float> %v1943, i64 1, !dbg !11
  %v1946 = extractelement <8 x float> %v1943, i64 2, !dbg !11
  %v1947 = extractelement <8 x float> %v1943, i64 3, !dbg !11
  %v1948 = select i1 %.not, float %v1945, float %v1944, !dbg !11
  %v1949 = bitcast float %v1948 to i32, !dbg !11
  %v1950 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1949), !dbg !11
  %v1951 = bitcast i32 %v1950 to float, !dbg !11
  %v1952 = select i1 %.not, float %v1944, float %v1951, !dbg !11
  %v1953 = select i1 %.not, float %v1951, float %v1945, !dbg !11
  %v1954 = select i1 %.not, float %v1947, float %v1946, !dbg !11
  %v1955 = bitcast float %v1954 to i32, !dbg !11
  %v1956 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1955), !dbg !11
  %v1957 = bitcast i32 %v1956 to float, !dbg !11
  %v1958 = select i1 %.not, float %v1946, float %v1957, !dbg !11
  %v1959 = select i1 %.not, float %v1957, float %v1947, !dbg !11
  %v1960 = select i1 %.not36, float %v1958, float %v1952, !dbg !11
  %v1961 = bitcast float %v1960 to i32, !dbg !11
  %v1962 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1961), !dbg !11
  %v1963 = select i1 %.not36, float %v1959, float %v1953, !dbg !11
  %v1964 = bitcast float %v1963 to i32, !dbg !11
  %v1965 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1964), !dbg !11
  %denom.exit.r1.packed = extractvalue [2 x <1 x float>] %v207.lcssa, 1, !dbg !30
  %denom.exit.r1.local = extractelement <1 x float> %denom.exit.r1.packed, i64 0, !dbg !30
  %denom.exit.r1.bits16 = bitcast float %denom.exit.r1.local to i32, !dbg !30
  %denom.exit.r1.peer16 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %denom.exit.r1.bits16), !dbg !30
  %denom.exit.r1.float16 = bitcast i32 %denom.exit.r1.peer16 to float, !dbg !30
  %denom.exit.r1.sum16 = fadd float %denom.exit.r1.local, %denom.exit.r1.float16, !dbg !30
  %denom.exit.r1.bits32 = bitcast float %denom.exit.r1.sum16 to i32, !dbg !30
  %denom.exit.r1.peer32 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %denom.exit.r1.bits32), !dbg !30
  %denom.exit.r1.float32 = bitcast i32 %denom.exit.r1.peer32 to float, !dbg !30
  %denom.exit.r1.sum32 = fadd float %denom.exit.r1.sum16, %denom.exit.r1.float32, !dbg !30
  %v1966 = insertelement <1 x float> poison, float %denom.exit.r1.sum32, i64 0, !dbg !30
  %v1967 = shufflevector <1 x float> %v1966, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %v1968 = extractelement <8 x float> %v1856, i64 4, !dbg !11
  %v1969 = extractelement <8 x float> %v1856, i64 5, !dbg !11
  %v1970 = extractelement <8 x float> %v1856, i64 6, !dbg !11
  %v1971 = extractelement <8 x float> %v1856, i64 7, !dbg !11
  %v1972 = select i1 %.not, float %v1969, float %v1968, !dbg !11
  %v1973 = bitcast float %v1972 to i32, !dbg !11
  %v1974 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1973), !dbg !11
  %v1975 = bitcast i32 %v1974 to float, !dbg !11
  %v1976 = select i1 %.not, float %v1968, float %v1975, !dbg !11
  %v1977 = select i1 %.not, float %v1975, float %v1969, !dbg !11
  %v1978 = select i1 %.not, float %v1971, float %v1970, !dbg !11
  %v1979 = bitcast float %v1978 to i32, !dbg !11
  %v1980 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1979), !dbg !11
  %v1981 = bitcast i32 %v1980 to float, !dbg !11
  %v1982 = select i1 %.not, float %v1970, float %v1981, !dbg !11
  %v1983 = select i1 %.not, float %v1981, float %v1971, !dbg !11
  %v1984 = select i1 %.not36, float %v1982, float %v1976, !dbg !11
  %v1985 = bitcast float %v1984 to i32, !dbg !11
  %v1986 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1985), !dbg !11
  %v1987 = bitcast i32 %v1986 to float, !dbg !11
  %v1990 = select i1 %.not36, float %v1983, float %v1977, !dbg !11
  %v1991 = bitcast float %v1990 to i32, !dbg !11
  %v1992 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1991), !dbg !11
  %v1993 = bitcast i32 %v1992 to float, !dbg !11
  %315 = insertelement <4 x float> poison, float %v1976, i64 0, !dbg !11
  %316 = insertelement <4 x float> %315, float %v1977, i64 1, !dbg !11
  %317 = insertelement <4 x float> %316, float %v1987, i64 2, !dbg !11
  %318 = insertelement <4 x float> %317, float %v1993, i64 3, !dbg !11
  %319 = insertelement <4 x float> poison, float %v1987, i64 0, !dbg !11
  %320 = insertelement <4 x float> %319, float %v1993, i64 1, !dbg !11
  %321 = insertelement <4 x float> %320, float %v1982, i64 2, !dbg !11
  %322 = insertelement <4 x float> %321, float %v1983, i64 3, !dbg !11
  %323 = select <4 x i1> %233, <4 x float> %318, <4 x float> %322, !dbg !11
  %v1996 = extractelement <8 x float> %v1885, i64 4, !dbg !11
  %v1997 = extractelement <8 x float> %v1885, i64 5, !dbg !11
  %v1998 = extractelement <8 x float> %v1885, i64 6, !dbg !11
  %v1999 = extractelement <8 x float> %v1885, i64 7, !dbg !11
  %v2000 = select i1 %.not, float %v1997, float %v1996, !dbg !11
  %v2001 = bitcast float %v2000 to i32, !dbg !11
  %v2002 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2001), !dbg !11
  %v2003 = bitcast i32 %v2002 to float, !dbg !11
  %v2004 = select i1 %.not, float %v1996, float %v2003, !dbg !11
  %v2005 = select i1 %.not, float %v2003, float %v1997, !dbg !11
  %v2006 = select i1 %.not, float %v1999, float %v1998, !dbg !11
  %v2007 = bitcast float %v2006 to i32, !dbg !11
  %v2008 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2007), !dbg !11
  %v2009 = bitcast i32 %v2008 to float, !dbg !11
  %v2010 = select i1 %.not, float %v1998, float %v2009, !dbg !11
  %v2011 = select i1 %.not, float %v2009, float %v1999, !dbg !11
  %v2012 = select i1 %.not36, float %v2010, float %v2004, !dbg !11
  %v2013 = bitcast float %v2012 to i32, !dbg !11
  %v2014 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2013), !dbg !11
  %v2015 = bitcast i32 %v2014 to float, !dbg !11
  %v2018 = select i1 %.not36, float %v2011, float %v2005, !dbg !11
  %v2019 = bitcast float %v2018 to i32, !dbg !11
  %v2020 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2019), !dbg !11
  %v2021 = bitcast i32 %v2020 to float, !dbg !11
  %324 = insertelement <4 x float> poison, float %v2004, i64 0, !dbg !11
  %325 = insertelement <4 x float> %324, float %v2005, i64 1, !dbg !11
  %326 = insertelement <4 x float> %325, float %v2015, i64 2, !dbg !11
  %327 = insertelement <4 x float> %326, float %v2021, i64 3, !dbg !11
  %328 = insertelement <4 x float> poison, float %v2015, i64 0, !dbg !11
  %329 = insertelement <4 x float> %328, float %v2021, i64 1, !dbg !11
  %330 = insertelement <4 x float> %329, float %v2010, i64 2, !dbg !11
  %331 = insertelement <4 x float> %330, float %v2011, i64 3, !dbg !11
  %332 = select <4 x i1> %233, <4 x float> %327, <4 x float> %331, !dbg !11
  %v2024 = extractelement <8 x float> %v1914, i64 4, !dbg !11
  %v2025 = extractelement <8 x float> %v1914, i64 5, !dbg !11
  %v2026 = extractelement <8 x float> %v1914, i64 6, !dbg !11
  %v2027 = extractelement <8 x float> %v1914, i64 7, !dbg !11
  %v2028 = select i1 %.not, float %v2025, float %v2024, !dbg !11
  %v2029 = bitcast float %v2028 to i32, !dbg !11
  %v2030 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2029), !dbg !11
  %v2031 = bitcast i32 %v2030 to float, !dbg !11
  %v2032 = select i1 %.not, float %v2024, float %v2031, !dbg !11
  %v2033 = select i1 %.not, float %v2031, float %v2025, !dbg !11
  %v2034 = select i1 %.not, float %v2027, float %v2026, !dbg !11
  %v2035 = bitcast float %v2034 to i32, !dbg !11
  %v2036 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2035), !dbg !11
  %v2037 = bitcast i32 %v2036 to float, !dbg !11
  %v2038 = select i1 %.not, float %v2026, float %v2037, !dbg !11
  %v2039 = select i1 %.not, float %v2037, float %v2027, !dbg !11
  %v2040 = select i1 %.not36, float %v2038, float %v2032, !dbg !11
  %v2041 = bitcast float %v2040 to i32, !dbg !11
  %v2042 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2041), !dbg !11
  %v2043 = bitcast i32 %v2042 to float, !dbg !11
  %v2046 = select i1 %.not36, float %v2039, float %v2033, !dbg !11
  %v2047 = bitcast float %v2046 to i32, !dbg !11
  %v2048 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2047), !dbg !11
  %v2049 = bitcast i32 %v2048 to float, !dbg !11
  %333 = insertelement <4 x float> poison, float %v2032, i64 0, !dbg !11
  %334 = insertelement <4 x float> %333, float %v2033, i64 1, !dbg !11
  %335 = insertelement <4 x float> %334, float %v2043, i64 2, !dbg !11
  %336 = insertelement <4 x float> %335, float %v2049, i64 3, !dbg !11
  %337 = insertelement <4 x float> poison, float %v2043, i64 0, !dbg !11
  %338 = insertelement <4 x float> %337, float %v2049, i64 1, !dbg !11
  %339 = insertelement <4 x float> %338, float %v2038, i64 2, !dbg !11
  %340 = insertelement <4 x float> %339, float %v2039, i64 3, !dbg !11
  %341 = select <4 x i1> %233, <4 x float> %336, <4 x float> %340, !dbg !11
  %v2052 = extractelement <8 x float> %v1943, i64 4, !dbg !11
  %v2053 = extractelement <8 x float> %v1943, i64 5, !dbg !11
  %v2054 = extractelement <8 x float> %v1943, i64 6, !dbg !11
  %v2055 = extractelement <8 x float> %v1943, i64 7, !dbg !11
  %v2056 = select i1 %.not, float %v2053, float %v2052, !dbg !11
  %v2057 = bitcast float %v2056 to i32, !dbg !11
  %v2058 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2057), !dbg !11
  %v2059 = bitcast i32 %v2058 to float, !dbg !11
  %v2060 = select i1 %.not, float %v2052, float %v2059, !dbg !11
  %v2061 = select i1 %.not, float %v2059, float %v2053, !dbg !11
  %v2062 = select i1 %.not, float %v2055, float %v2054, !dbg !11
  %v2063 = bitcast float %v2062 to i32, !dbg !11
  %v2064 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2063), !dbg !11
  %v2065 = bitcast i32 %v2064 to float, !dbg !11
  %v2066 = select i1 %.not, float %v2054, float %v2065, !dbg !11
  %v2067 = select i1 %.not, float %v2065, float %v2055, !dbg !11
  %v2068 = select i1 %.not36, float %v2066, float %v2060, !dbg !11
  %v2069 = bitcast float %v2068 to i32, !dbg !11
  %v2070 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2069), !dbg !11
  %v2071 = select i1 %.not36, float %v2067, float %v2061, !dbg !11
  %v2072 = bitcast float %v2071 to i32, !dbg !11
  %v2073 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2072), !dbg !11
  %v2075 = bitcast i32 %v2070 to float, !dbg !11
  %v2078 = bitcast i32 %v2073 to float, !dbg !11
  %342 = insertelement <4 x float> poison, float %v2060, i64 0, !dbg !11
  %343 = insertelement <4 x float> %342, float %v2061, i64 1, !dbg !11
  %344 = insertelement <4 x float> %343, float %v2075, i64 2, !dbg !11
  %345 = insertelement <4 x float> %344, float %v2078, i64 3, !dbg !11
  %346 = insertelement <4 x float> poison, float %v2075, i64 0, !dbg !11
  %347 = insertelement <4 x float> %346, float %v2078, i64 1, !dbg !11
  %348 = insertelement <4 x float> %347, float %v2066, i64 2, !dbg !11
  %349 = insertelement <4 x float> %348, float %v2067, i64 3, !dbg !11
  %350 = select <4 x i1> %233, <4 x float> %345, <4 x float> %349, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %v1967, !dbg !30
  %normalized = fmul <4 x float> %row.reciprocal, %350, !dbg !30
  %v2085 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !31
  %v2086 = extractelement <4 x half> %v2085, i64 3, !dbg !31
  %v2087 = extractelement <4 x half> %v2085, i64 2, !dbg !31
  %v2088 = extractelement <4 x half> %v2085, i64 1, !dbg !31
  %v2089 = extractelement <4 x half> %v2085, i64 0, !dbg !31
  %v2090 = bitcast i32 %v1962 to float, !dbg !11
  %v2093 = bitcast i32 %v1965 to float, !dbg !11
  %351 = insertelement <4 x float> poison, float %v1952, i64 0, !dbg !11
  %352 = insertelement <4 x float> %351, float %v1953, i64 1, !dbg !11
  %353 = insertelement <4 x float> %352, float %v2090, i64 2, !dbg !11
  %354 = insertelement <4 x float> %353, float %v2093, i64 3, !dbg !11
  %355 = insertelement <4 x float> poison, float %v2090, i64 0, !dbg !11
  %356 = insertelement <4 x float> %355, float %v2093, i64 1, !dbg !11
  %357 = insertelement <4 x float> %356, float %v1958, i64 2, !dbg !11
  %358 = insertelement <4 x float> %357, float %v1959, i64 3, !dbg !11
  %359 = select <4 x i1> %233, <4 x float> %354, <4 x float> %358, !dbg !11
  %normalized104 = fmul <4 x float> %359, %row.reciprocal, !dbg !30
  %v2100 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !31
  %v2101 = extractelement <4 x half> %v2100, i64 3, !dbg !31
  %v2102 = extractelement <4 x half> %v2100, i64 2, !dbg !31
  %v2103 = extractelement <4 x half> %v2100, i64 1, !dbg !31
  %v2104 = extractelement <4 x half> %v2100, i64 0, !dbg !31
  %v2105 = bitcast i32 %v1852 to float, !dbg !11
  %v2108 = bitcast i32 %v1855 to float, !dbg !11
  %360 = insertelement <4 x float> poison, float %v1842, i64 0, !dbg !11
  %361 = insertelement <4 x float> %360, float %v1843, i64 1, !dbg !11
  %362 = insertelement <4 x float> %361, float %v2105, i64 2, !dbg !11
  %363 = insertelement <4 x float> %362, float %v2108, i64 3, !dbg !11
  %364 = insertelement <4 x float> poison, float %v2105, i64 0, !dbg !11
  %365 = insertelement <4 x float> %364, float %v2108, i64 1, !dbg !11
  %366 = insertelement <4 x float> %365, float %v1848, i64 2, !dbg !11
  %367 = insertelement <4 x float> %366, float %v1849, i64 3, !dbg !11
  %368 = select <4 x i1> %233, <4 x float> %363, <4 x float> %367, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %v1749, !dbg !30
  %normalized106 = fmul <4 x float> %row.reciprocal105, %368, !dbg !30
  %v2115 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !31
  %v2116 = extractelement <4 x half> %v2115, i64 3, !dbg !31
  %v2117 = extractelement <4 x half> %v2115, i64 2, !dbg !31
  %v2118 = extractelement <4 x half> %v2115, i64 1, !dbg !31
  %v2119 = extractelement <4 x half> %v2115, i64 0, !dbg !31
  %v2120 = bitcast i32 %v1744 to float, !dbg !11
  %v2123 = bitcast i32 %v1747 to float, !dbg !11
  %369 = insertelement <4 x float> poison, float %v1734, i64 0, !dbg !11
  %370 = insertelement <4 x float> %369, float %v1735, i64 1, !dbg !11
  %371 = insertelement <4 x float> %370, float %v2120, i64 2, !dbg !11
  %372 = insertelement <4 x float> %371, float %v2123, i64 3, !dbg !11
  %373 = insertelement <4 x float> poison, float %v2120, i64 0, !dbg !11
  %374 = insertelement <4 x float> %373, float %v2123, i64 1, !dbg !11
  %375 = insertelement <4 x float> %374, float %v1740, i64 2, !dbg !11
  %376 = insertelement <4 x float> %375, float %v1741, i64 3, !dbg !11
  %377 = select <4 x i1> %233, <4 x float> %372, <4 x float> %376, !dbg !11
  %normalized107 = fmul <4 x float> %377, %row.reciprocal105, !dbg !30
  %v2130 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !31
  %v2131 = extractelement <4 x half> %v2130, i64 3, !dbg !31
  %v2132 = extractelement <4 x half> %v2130, i64 2, !dbg !31
  %v2133 = extractelement <4 x half> %v2130, i64 1, !dbg !31
  %v2134 = extractelement <4 x half> %v2130, i64 0, !dbg !31
  %address.thread1090 = tail call i32 asm sideeffect "", "=v,0"(i32 %v31) #8, !dbg !32, !range !9, !frisk.register_lifetimes !5
  %address.local1091 = zext nneg i32 %address.thread1090 to i64, !dbg !8
  %378 = lshr i64 %address.local1091, 2, !dbg !32
  %address.local1094 = and i64 %378, 48, !dbg !32
  %address.local1096 = and i64 %address.local1091, 240, !dbg !32
  %address.local1095 = add i64 %v27, %address.local1091, !dbg !32
  %address.local1097 = sub i64 %address.local1095, %address.local1096, !dbg !32
  %address.local1098 = add i64 %address.local1097, %address.local1094, !dbg !32
  %379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %v28, !dbg !32
  %address.local1101.idx = shl nuw nsw i64 %address.local1098, 8, !dbg !32
  %address.local1101 = getelementptr inbounds nuw i8, ptr addrspace(1) %379, i64 %address.local1101.idx, !dbg !32
  %address.local1103 = and i64 %378, 12, !dbg !10
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1103, !dbg !32
  store half %v2134, ptr addrspace(1) %address.local1104, align 2, !dbg !32
  %address.local1106 = or disjoint i64 %address.local1103, 1, !dbg !32
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1106, !dbg !32
  store half %v2133, ptr addrspace(1) %address.local1107, align 2, !dbg !32
  %address.local1109 = or disjoint i64 %address.local1103, 2, !dbg !32
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1109, !dbg !32
  store half %v2132, ptr addrspace(1) %address.local1110, align 2, !dbg !32
  %address.local1112 = or disjoint i64 %address.local1103, 3, !dbg !32
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1112, !dbg !32
  store half %v2131, ptr addrspace(1) %address.local1113, align 2, !dbg !32
  %address.local1115 = or disjoint i64 %address.local1103, 16, !dbg !32
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1115, !dbg !32
  store half %v2119, ptr addrspace(1) %address.local1116, align 2, !dbg !32
  %address.local1118 = or disjoint i64 %address.local1103, 17, !dbg !32
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1118, !dbg !32
  store half %v2118, ptr addrspace(1) %address.local1119, align 2, !dbg !32
  %address.local1121 = or disjoint i64 %address.local1103, 18, !dbg !32
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1121, !dbg !32
  store half %v2117, ptr addrspace(1) %address.local1122, align 2, !dbg !32
  %address.local1124 = or disjoint i64 %address.local1103, 19, !dbg !32
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1124, !dbg !32
  store half %v2116, ptr addrspace(1) %address.local1125, align 2, !dbg !32
  %address.local1129 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1101, i64 16384, !dbg !32
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1103, !dbg !32
  store half %v2104, ptr addrspace(1) %address.local1130, align 2, !dbg !32
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1106, !dbg !32
  store half %v2103, ptr addrspace(1) %address.local1132, align 2, !dbg !32
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1109, !dbg !32
  store half %v2102, ptr addrspace(1) %address.local1134, align 2, !dbg !32
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1112, !dbg !32
  store half %v2101, ptr addrspace(1) %address.local1136, align 2, !dbg !32
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1115, !dbg !32
  store half %v2089, ptr addrspace(1) %address.local1138, align 2, !dbg !32
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1118, !dbg !32
  store half %v2088, ptr addrspace(1) %address.local1140, align 2, !dbg !32
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1121, !dbg !32
  store half %v2087, ptr addrspace(1) %address.local1142, align 2, !dbg !32
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1124, !dbg !32
  store half %v2086, ptr addrspace(1) %address.local1144, align 2, !dbg !32
  %normalized109 = fmul <4 x float> %row.reciprocal, %341, !dbg !30
  %v2144 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !31
  %v2145 = extractelement <4 x half> %v2144, i64 3, !dbg !31
  %v2146 = extractelement <4 x half> %v2144, i64 2, !dbg !31
  %v2147 = extractelement <4 x half> %v2144, i64 1, !dbg !31
  %v2148 = extractelement <4 x half> %v2144, i64 0, !dbg !31
  %normalized110 = fmul <4 x float> %314, %row.reciprocal, !dbg !30
  %v2153 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !31
  %v2154 = extractelement <4 x half> %v2153, i64 3, !dbg !31
  %v2155 = extractelement <4 x half> %v2153, i64 2, !dbg !31
  %v2156 = extractelement <4 x half> %v2153, i64 1, !dbg !31
  %v2157 = extractelement <4 x half> %v2153, i64 0, !dbg !31
  %normalized112 = fmul <4 x float> %row.reciprocal105, %287, !dbg !30
  %v2162 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !31
  %v2163 = extractelement <4 x half> %v2162, i64 3, !dbg !31
  %v2164 = extractelement <4 x half> %v2162, i64 2, !dbg !31
  %v2165 = extractelement <4 x half> %v2162, i64 1, !dbg !31
  %v2166 = extractelement <4 x half> %v2162, i64 0, !dbg !31
  %normalized113 = fmul <4 x float> %260, %row.reciprocal105, !dbg !30
  %v2171 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !31
  %v2172 = extractelement <4 x half> %v2171, i64 3, !dbg !31
  %v2173 = extractelement <4 x half> %v2171, i64 2, !dbg !31
  %v2174 = extractelement <4 x half> %v2171, i64 1, !dbg !31
  %v2175 = extractelement <4 x half> %v2171, i64 0, !dbg !31
  %address.local1159 = or disjoint i64 %address.local1103, 32, !dbg !32
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1159, !dbg !32
  store half %v2175, ptr addrspace(1) %address.local1160, align 2, !dbg !32
  %address.local1162 = or disjoint i64 %address.local1103, 33, !dbg !32
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1162, !dbg !32
  store half %v2174, ptr addrspace(1) %address.local1163, align 2, !dbg !32
  %address.local1165 = or disjoint i64 %address.local1103, 34, !dbg !32
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1165, !dbg !32
  store half %v2173, ptr addrspace(1) %address.local1166, align 2, !dbg !32
  %address.local1168 = or disjoint i64 %address.local1103, 35, !dbg !32
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1168, !dbg !32
  store half %v2172, ptr addrspace(1) %address.local1169, align 2, !dbg !32
  %address.local1171 = or disjoint i64 %address.local1103, 48, !dbg !32
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1171, !dbg !32
  store half %v2166, ptr addrspace(1) %address.local1172, align 2, !dbg !32
  %address.local1174 = or disjoint i64 %address.local1103, 49, !dbg !32
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1174, !dbg !32
  store half %v2165, ptr addrspace(1) %address.local1175, align 2, !dbg !32
  %address.local1177 = or disjoint i64 %address.local1103, 50, !dbg !32
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1177, !dbg !32
  store half %v2164, ptr addrspace(1) %address.local1178, align 2, !dbg !32
  %address.local1180 = or i64 %378, 51, !dbg !32
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1180, !dbg !32
  store half %v2163, ptr addrspace(1) %address.local1181, align 2, !dbg !32
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1159, !dbg !32
  store half %v2157, ptr addrspace(1) %address.local1186, align 2, !dbg !32
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1162, !dbg !32
  store half %v2156, ptr addrspace(1) %address.local1188, align 2, !dbg !32
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1165, !dbg !32
  store half %v2155, ptr addrspace(1) %address.local1190, align 2, !dbg !32
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1168, !dbg !32
  store half %v2154, ptr addrspace(1) %address.local1192, align 2, !dbg !32
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1171, !dbg !32
  store half %v2148, ptr addrspace(1) %address.local1194, align 2, !dbg !32
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1174, !dbg !32
  store half %v2147, ptr addrspace(1) %address.local1196, align 2, !dbg !32
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1177, !dbg !32
  store half %v2146, ptr addrspace(1) %address.local1198, align 2, !dbg !32
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1180, !dbg !32
  store half %v2145, ptr addrspace(1) %address.local1200, align 2, !dbg !32
  %normalized115 = fmul <4 x float> %row.reciprocal, %332, !dbg !30
  %v2183 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !31
  %v2184 = extractelement <4 x half> %v2183, i64 3, !dbg !31
  %v2185 = extractelement <4 x half> %v2183, i64 2, !dbg !31
  %v2186 = extractelement <4 x half> %v2183, i64 1, !dbg !31
  %v2187 = extractelement <4 x half> %v2183, i64 0, !dbg !31
  %normalized116 = fmul <4 x float> %305, %row.reciprocal, !dbg !30
  %v2192 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !31
  %v2193 = extractelement <4 x half> %v2192, i64 3, !dbg !31
  %v2194 = extractelement <4 x half> %v2192, i64 2, !dbg !31
  %v2195 = extractelement <4 x half> %v2192, i64 1, !dbg !31
  %v2196 = extractelement <4 x half> %v2192, i64 0, !dbg !31
  %normalized118 = fmul <4 x float> %row.reciprocal105, %278, !dbg !30
  %v2201 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !31
  %v2202 = extractelement <4 x half> %v2201, i64 3, !dbg !31
  %v2203 = extractelement <4 x half> %v2201, i64 2, !dbg !31
  %v2204 = extractelement <4 x half> %v2201, i64 1, !dbg !31
  %v2205 = extractelement <4 x half> %v2201, i64 0, !dbg !31
  %normalized119 = fmul <4 x float> %251, %row.reciprocal105, !dbg !30
  %v2210 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !31
  %v2211 = extractelement <4 x half> %v2210, i64 3, !dbg !31
  %v2212 = extractelement <4 x half> %v2210, i64 2, !dbg !31
  %v2213 = extractelement <4 x half> %v2210, i64 1, !dbg !31
  %v2214 = extractelement <4 x half> %v2210, i64 0, !dbg !31
  %address.local1215 = or disjoint i64 %address.local1103, 64, !dbg !32
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1215, !dbg !32
  store half %v2214, ptr addrspace(1) %address.local1216, align 2, !dbg !32
  %address.local1218 = or disjoint i64 %address.local1103, 65, !dbg !32
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1218, !dbg !32
  store half %v2213, ptr addrspace(1) %address.local1219, align 2, !dbg !32
  %address.local1221 = or disjoint i64 %address.local1103, 66, !dbg !32
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1221, !dbg !32
  store half %v2212, ptr addrspace(1) %address.local1222, align 2, !dbg !32
  %address.local1224 = or disjoint i64 %address.local1103, 67, !dbg !32
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1224, !dbg !32
  store half %v2211, ptr addrspace(1) %address.local1225, align 2, !dbg !32
  %address.local1227 = or disjoint i64 %address.local1103, 80, !dbg !32
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1227, !dbg !32
  store half %v2205, ptr addrspace(1) %address.local1228, align 2, !dbg !32
  %address.local1230 = or disjoint i64 %address.local1103, 81, !dbg !32
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1230, !dbg !32
  store half %v2204, ptr addrspace(1) %address.local1231, align 2, !dbg !32
  %address.local1233 = or disjoint i64 %address.local1103, 82, !dbg !32
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1233, !dbg !32
  store half %v2203, ptr addrspace(1) %address.local1234, align 2, !dbg !32
  %address.local1236 = or disjoint i64 %address.local1103, 83, !dbg !32
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1236, !dbg !32
  store half %v2202, ptr addrspace(1) %address.local1237, align 2, !dbg !32
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1215, !dbg !32
  store half %v2196, ptr addrspace(1) %address.local1242, align 2, !dbg !32
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1218, !dbg !32
  store half %v2195, ptr addrspace(1) %address.local1244, align 2, !dbg !32
  %address.local1246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1221, !dbg !32
  store half %v2194, ptr addrspace(1) %address.local1246, align 2, !dbg !32
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1224, !dbg !32
  store half %v2193, ptr addrspace(1) %address.local1248, align 2, !dbg !32
  %address.local1250 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1227, !dbg !32
  store half %v2187, ptr addrspace(1) %address.local1250, align 2, !dbg !32
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1230, !dbg !32
  store half %v2186, ptr addrspace(1) %address.local1252, align 2, !dbg !32
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1233, !dbg !32
  store half %v2185, ptr addrspace(1) %address.local1254, align 2, !dbg !32
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1236, !dbg !32
  store half %v2184, ptr addrspace(1) %address.local1256, align 2, !dbg !32
  %normalized121 = fmul <4 x float> %row.reciprocal, %323, !dbg !30
  %v2222 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !31
  %v2223 = extractelement <4 x half> %v2222, i64 3, !dbg !31
  %v2224 = extractelement <4 x half> %v2222, i64 2, !dbg !31
  %v2225 = extractelement <4 x half> %v2222, i64 1, !dbg !31
  %v2226 = extractelement <4 x half> %v2222, i64 0, !dbg !31
  %normalized122 = fmul <4 x float> %296, %row.reciprocal, !dbg !30
  %v2231 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !31
  %v2232 = extractelement <4 x half> %v2231, i64 3, !dbg !31
  %v2233 = extractelement <4 x half> %v2231, i64 2, !dbg !31
  %v2234 = extractelement <4 x half> %v2231, i64 1, !dbg !31
  %v2235 = extractelement <4 x half> %v2231, i64 0, !dbg !31
  %normalized124 = fmul <4 x float> %row.reciprocal105, %269, !dbg !30
  %v2240 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !31
  %v2241 = extractelement <4 x half> %v2240, i64 3, !dbg !31
  %v2242 = extractelement <4 x half> %v2240, i64 2, !dbg !31
  %v2243 = extractelement <4 x half> %v2240, i64 1, !dbg !31
  %v2244 = extractelement <4 x half> %v2240, i64 0, !dbg !31
  %normalized125 = fmul <4 x float> %242, %row.reciprocal105, !dbg !30
  %v2249 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !31
  %v2250 = extractelement <4 x half> %v2249, i64 3, !dbg !31
  %v2251 = extractelement <4 x half> %v2249, i64 2, !dbg !31
  %v2252 = extractelement <4 x half> %v2249, i64 1, !dbg !31
  %v2253 = extractelement <4 x half> %v2249, i64 0, !dbg !31
  %address.local1271 = or disjoint i64 %address.local1103, 96, !dbg !32
  %address.local1272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1271, !dbg !32
  store half %v2253, ptr addrspace(1) %address.local1272, align 2, !dbg !32
  %address.local1274 = or disjoint i64 %address.local1103, 97, !dbg !32
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1274, !dbg !32
  store half %v2252, ptr addrspace(1) %address.local1275, align 2, !dbg !32
  %address.local1277 = or disjoint i64 %address.local1103, 98, !dbg !32
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1277, !dbg !32
  store half %v2251, ptr addrspace(1) %address.local1278, align 2, !dbg !32
  %address.local1280 = or disjoint i64 %address.local1103, 99, !dbg !32
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1280, !dbg !32
  store half %v2250, ptr addrspace(1) %address.local1281, align 2, !dbg !32
  %address.local1283 = or disjoint i64 %address.local1103, 112, !dbg !32
  %address.local1284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1283, !dbg !32
  store half %v2244, ptr addrspace(1) %address.local1284, align 2, !dbg !32
  %address.local1286 = or disjoint i64 %address.local1103, 113, !dbg !32
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1286, !dbg !32
  store half %v2243, ptr addrspace(1) %address.local1287, align 2, !dbg !32
  %address.local1289 = or disjoint i64 %address.local1103, 114, !dbg !32
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1289, !dbg !32
  store half %v2242, ptr addrspace(1) %address.local1290, align 2, !dbg !32
  %address.local1292 = or i64 %378, 115, !dbg !32
  %address.local1293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1292, !dbg !32
  store half %v2241, ptr addrspace(1) %address.local1293, align 2, !dbg !32
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1271, !dbg !32
  store half %v2235, ptr addrspace(1) %address.local1298, align 2, !dbg !32
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1274, !dbg !32
  store half %v2234, ptr addrspace(1) %address.local1300, align 2, !dbg !32
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1277, !dbg !32
  store half %v2233, ptr addrspace(1) %address.local1302, align 2, !dbg !32
  %address.local1304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1280, !dbg !32
  store half %v2232, ptr addrspace(1) %address.local1304, align 2, !dbg !32
  %address.local1306 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1283, !dbg !32
  store half %v2226, ptr addrspace(1) %address.local1306, align 2, !dbg !32
  %address.local1308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1286, !dbg !32
  store half %v2225, ptr addrspace(1) %address.local1308, align 2, !dbg !32
  %address.local1310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1289, !dbg !32
  store half %v2224, ptr addrspace(1) %address.local1310, align 2, !dbg !32
  %address.local1312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1292, !dbg !32
  store half %v2223, ptr addrspace(1) %address.local1312, align 2, !dbg !32
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
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/local_reduction_pass/compiler-bn128/run-20261026-112711-jne7_i1s/grid-000003")
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
