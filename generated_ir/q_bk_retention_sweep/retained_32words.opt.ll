; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/q_bk_retention_sweep/retained_32words.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = shl nsw i64 %8, 5, !dbg !7
  %10 = add nsw i64 %9, %6, !dbg !7
  %11 = lshr i64 %10, 5, !dbg !7
  %12 = and i64 %11, 576460752303423480, !dbg !7
  %13 = and i64 %6, 7, !dbg !7
  %14 = or disjoint i64 %12, %13, !dbg !7
  %15 = sub nsw i64 31, %14, !dbg !7
  %16 = lshr i64 %10, 3, !dbg !7
  %17 = icmp ugt i64 %10, 1023, !dbg !8
  %18 = add nsw i64 %14, -32, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 32, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %22, %16, !dbg !8
  %24 = shl i64 %15, 7, !dbg !8
  %25 = shl i64 %22, 12, !dbg !8
  %26 = sub i64 %24, %25, !dbg !8
  %27 = shl i64 %23, 19, !dbg !8
  %28 = and i64 %27, 16252928, !dbg !8
  %29 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %.idx = shl i64 %26, 8, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local339 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local340 = and i64 %address.local339, 1536, !dbg !8
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !8
  %address.local344 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local345 = and i64 %address.local344, 16256, !dbg !8
  %address.local346 = lshr i64 %address.local, 4, !dbg !8
  %address.local347 = and i64 %address.local346, 8, !dbg !8
  %address.local348 = or disjoint i64 %address.local345, %address.local347, !dbg !8
  %30 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %31 = or disjoint i64 %address.local348, %28, !dbg !8
  %32 = shl nuw nsw i64 %31, 1, !dbg !8
  %33 = add i64 %32, %.idx, !dbg !8
  %dma.buffer.offset1612 = trunc i64 %33 to i32, !dbg !8
  %34 = insertelement <2 x i64> poison, i64 %30, i64 0, !dbg !8
  %35 = bitcast <2 x i64> %34 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1613 = shufflevector <4 x i32> %35, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local350 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local351 = shl nuw nsw i64 %address.local350, 3, !dbg !8
  %address.local352 = and i64 %address.local351, 3584, !dbg !8
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !8
  %address.local358 = lshr i64 %address.local350, 4, !dbg !8
  %address.local359 = and i64 %address.local358, 24, !dbg !8
  %address.local360 = or disjoint i64 %address.local359, %address.local345, !dbg !8
  %36 = or disjoint i64 %address.local360, %28, !dbg !8
  %37 = shl nuw nsw i64 %36, 1, !dbg !8
  %38 = add i64 %37, %.idx, !dbg !8
  %dma.buffer.offset1609 = trunc i64 %38 to i32, !dbg !8
  %39 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %40 = sext i32 %39 to i64, !dbg !7
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !8
  %.idx3 = shl nuw nsw i64 %28, 1, !dbg !8
  %41 = add i64 %.idx3, %.idx, !dbg !8
  %address.local365.idx = shl nuw nsw i64 %address.local348, 1, !dbg !8
  %42 = or disjoint i64 %address.local365.idx, %41, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %dma.buffer.offset1606 = or disjoint i32 %43, 64, !dbg !8
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !8
  %44 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local341, i64 0, !dbg !8
  %45 = insertelement <4 x ptr addrspace(3)> %44, ptr addrspace(3) %address.local353, i64 1, !dbg !8
  %46 = insertelement <4 x ptr addrspace(3)> %45, ptr addrspace(3) %address.local362, i64 2, !dbg !8
  %47 = insertelement <4 x ptr addrspace(3)> %46, ptr addrspace(3) %address.local366, i64 3, !dbg !8
  %48 = ptrtoint <4 x ptr addrspace(3)> %47 to <4 x i64>, !dbg !8
  %49 = trunc <4 x i64> %48 to <4 x i32>, !dbg !8
  %50 = extractelement <4 x i32> %49, i64 0, !dbg !8
  %51 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %dma.buffer.lds1614 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %51), !dbg !8
  %52 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1612, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %53 = extractelement <4 x i32> %49, i64 1, !dbg !8
  %54 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %dma.buffer.lds1611 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %54), !dbg !8
  %55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1609, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %56 = extractelement <4 x i32> %49, i64 2, !dbg !8
  %57 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %56), !dbg !8
  %dma.buffer.lds1608 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %57), !dbg !8
  %58 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1606, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %59 = extractelement <4 x i32> %49, i64 3, !dbg !8
  %60 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %59), !dbg !8
  %address.local369.idx = shl nuw nsw i64 %address.local360, 1, !dbg !8
  %61 = or disjoint i64 %address.local369.idx, %41, !dbg !8
  %62 = trunc i64 %61 to i32, !dbg !8
  %dma.buffer.offset1603 = or disjoint i32 %62, 64, !dbg !8
  %dma.buffer.lds1605 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1603, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
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
  %64 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !10
  %65 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %66 = fmul <4 x half> %64, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !10
  %67 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !10
  %68 = fmul <4 x half> %65, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !10
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !10
  %70 = fmul <4 x half> %67, splat (half 1.275630e-01), !dbg !10
  %71 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1600 = or disjoint i32 %43, 128, !dbg !8
  %72 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1600, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %dma.buffer.offset1597 = or disjoint i32 %62, 128, !dbg !8
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !10
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !10
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !10
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !10
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !10
  %77 = fmul <4 x half> %74, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !10
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !10
  %79 = fmul <4 x half> %75, splat (half 1.275630e-01), !dbg !10
  %80 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1594 = or disjoint i32 %43, 192, !dbg !8
  %82 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %dma.buffer.offset1591 = or disjoint i32 %62, 192, !dbg !8
  %83 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1613) #9, !dbg !8
  %84 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !10
  %87 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !10
  %89 = fmul <4 x half> %85, splat (half 1.275630e-01), !dbg !10
  %90 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !10
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !10
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !10
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !10
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !10
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !10
  %97 = fmul <4 x half> %93, splat (half 1.275630e-01), !dbg !10
  %98 = fmul <4 x half> %94, splat (half 1.275630e-01), !dbg !10
  %99 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %100 = or disjoint i64 %24, 127, !dbg !12
  %101 = add i64 %24, 128, !dbg !12
  %102 = icmp uge i64 %101, %100, !dbg !12
  %103 = sub nsw i64 32, %14, !dbg !13
  %104 = shl i64 %16, 7, !dbg !14
  %105 = and i64 %104, 3968, !dbg !14
  %106 = shl nsw i64 %40, 12, !dbg !14
  %107 = or disjoint i64 %105, %106, !dbg !14
  %108 = shl nuw nsw i32 %29, 2, !dbg !15
  %109 = and i32 %108, 12, !dbg !15
  %110 = and i32 %29, 1, !dbg !15
  %111 = or disjoint i32 %109, %110, !dbg !15
  %112 = trunc nuw nsw i32 %111 to i16, !dbg !15
  %.lhs.trunc = add nuw nsw i16 %112, 1, !dbg !15
  %113 = lshr i16 %.lhs.trunc, 1, !dbg !15
  %.sext = zext nneg i16 %113 to i64, !dbg !15
  %114 = icmp ult i64 %10, 1024, !dbg !11
  br i1 %114, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !11

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre105 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre107 = add nuw nsw i32 %.pre105, 64, !dbg !11
  %.pre109 = xor i32 %.pre105, 16, !dbg !11
  %.pre111 = and i32 %.pre107, 192, !dbg !11
  %.pre113 = xor i32 %.pre105, 32, !dbg !11
  br label %._crit_edge, !dbg !11

.lr.ph:                                           ; preds = %4
  %115 = zext nneg i32 %29 to i64, !dbg !8
  %116 = ptrtoint ptr addrspace(1) %1 to i64
  %117 = insertelement <2 x i64> poison, i64 %116, i64 0
  %118 = bitcast <2 x i64> %117 to <4 x i32>
  %dma.buffer.rsrc1589 = shufflevector <4 x i32> %118, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %119 = ptrtoint ptr addrspace(1) %2 to i64
  %120 = insertelement <2 x i64> poison, i64 %119, i64 0
  %121 = bitcast <2 x i64> %120 to <4 x i32>
  %dma.buffer.rsrc1565 = shufflevector <4 x i32> %121, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %122 = lshr i64 %115, 2
  %123 = and i64 %122, 48
  %124 = and i32 %29, 240
  %index.compact = zext nneg i32 %124 to i64
  %125 = sub nsw i64 %115, %index.compact
  %126 = add nsw i64 %125, %123
  %127 = add nsw i64 %126, %24
  %128 = and i64 %115, 48
  %129 = add i64 %127, 1
  %130 = lshr i64 %129, 32
  %compare.high = trunc nuw i64 %130 to i32
  %compare.low.base = trunc i64 %129 to i32
  %131 = add i64 %127, 65
  %132 = lshr i64 %131, 32
  %compare.high1281 = trunc nuw i64 %132 to i32
  %compare.low.base1285 = trunc i64 %131 to i32
  %133 = insertelement <32 x i32> poison, i32 %compare.low.base1285, i64 0, !dbg !16
  %134 = shufflevector <32 x i32> %133, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  %135 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %136 = shufflevector <32 x i32> %135, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  br label %137, !dbg !11

137:                                              ; preds = %.lr.ph, %536
  %138 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1190, %536 ]
  %139 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1186, %536 ]
  %140 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1182, %536 ]
  %141 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %879, %536 ]
  %142 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1178, %536 ]
  %143 = phi i64 [ 0, %.lr.ph ], [ %1191, %536 ]
  %144 = shl nuw nsw i64 %143, 7, !dbg !11
  %145 = lshr i64 %143, 12, !dbg !14
  %146 = add nuw nsw i64 %145, %16, !dbg !14
  %147 = lshr i64 %143, 5, !dbg !14
  %148 = and i64 %147, 127, !dbg !14
  %149 = shl i64 %146, 19, !dbg !14
  %150 = and i64 %149, 16252928, !dbg !14
  %151 = shl i64 %147, 19, !dbg !14
  %152 = shl nuw nsw i64 %143, 14, !dbg !14
  %153 = sub i64 %152, %151, !dbg !14
  %address.thread433 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !14, !range !9, !frisk.register_lifetimes !5
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
  %.idx912 = or disjoint i64 %150, %148, !dbg !14
  %154 = add i64 %.idx912, %153, !dbg !14
  %155 = add i64 %address.local449, %154, !dbg !14
  %.tr = trunc i64 %155 to i32, !dbg !14
  %dma.buffer.offset1588 = shl i32 %.tr, 1, !dbg !14
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !8
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !8
  %address.local453 = and i64 %address.local452, 3584, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !8
  %address.local457 = lshr i64 %address.local451, 4, !dbg !14
  %address.local458 = and i64 %address.local457, 24, !dbg !14
  %address.local466 = or disjoint i64 %address.local448, %address.local458, !dbg !14
  %156 = add i64 %address.local466, %154, !dbg !14
  %.tr17 = trunc i64 %156 to i32, !dbg !14
  %dma.buffer.offset1585 = shl i32 %.tr17, 1, !dbg !14
  %157 = add nuw nsw i64 %143, 1024, !dbg !14
  %158 = lshr i64 %157, 12, !dbg !14
  %159 = add nuw nsw i64 %158, %16, !dbg !14
  %160 = add nsw i64 %147, %107, !dbg !14
  %161 = shl i64 %159, 19, !dbg !14
  %162 = and i64 %161, 16252928, !dbg !14
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !8
  %163 = add i64 %153, %147, !dbg !14
  %.idx1822 = add i64 %163, %107, !dbg !14
  %164 = add i64 %.idx1822, %162, !dbg !14
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !8
  %165 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local437, i64 0, !dbg !8
  %166 = insertelement <4 x ptr addrspace(3)> %165, ptr addrspace(3) %address.local454, i64 1, !dbg !8
  %167 = insertelement <4 x ptr addrspace(3)> %166, ptr addrspace(3) %address.local468, i64 2, !dbg !8
  %168 = insertelement <4 x ptr addrspace(3)> %167, ptr addrspace(3) %address.local472, i64 3, !dbg !8
  %169 = ptrtoint <4 x ptr addrspace(3)> %168 to <4 x i64>, !dbg !8
  %170 = trunc <4 x i64> %169 to <4 x i32>, !dbg !8
  %171 = extractelement <4 x i32> %170, i64 0, !dbg !14
  %172 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %171), !dbg !14
  %dma.buffer.lds1590 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %172), !dbg !14
  %173 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %174 = extractelement <4 x i32> %170, i64 1, !dbg !14
  %175 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %174), !dbg !14
  %dma.buffer.lds1587 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %175), !dbg !14
  %176 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %177 = extractelement <4 x i32> %170, i64 2, !dbg !14
  %178 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %177), !dbg !14
  %dma.buffer.lds1584 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %178), !dbg !14
  %179 = extractelement <4 x i32> %170, i64 3, !dbg !14
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !15
  %address.local481 = and i64 %address.local480, 24, !dbg !15
  %address.local483 = shl i64 %address.local434, 62, !dbg !15
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !15
  %address.local485 = and i64 %address.local484, -64, !dbg !15
  %address.local488 = and i64 %address.local443, 12, !dbg !10
  %180 = and i64 %address.local434, 32, !dbg !10
  %.not30 = icmp eq i64 %180, 0, !dbg !10
  %address.local492 = select i1 %.not30, i64 0, i64 1016, !dbg !10
  %.idx31 = mul nuw nsw i64 %address.local434, 320, !dbg !15
  %181 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx31, !dbg !15
  %.idx32 = mul nsw i64 %address.local443, -1280, !dbg !15
  %182 = getelementptr i8, ptr addrspace(3) %181, i64 %.idx32, !dbg !15
  %183 = getelementptr [2 x i8], ptr addrspace(3) %182, i64 %address.local481, !dbg !15
  %184 = getelementptr [2 x i8], ptr addrspace(3) %183, i64 %address.local485, !dbg !15
  %185 = getelementptr [2 x i8], ptr addrspace(3) %184, i64 %address.local488, !dbg !15
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) %185, i64 %address.local492, !dbg !15
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
  %186 = add nuw nsw i64 %143, 2048, !dbg !14
  %187 = lshr i64 %186, 12, !dbg !14
  %188 = add nuw nsw i64 %187, %16, !dbg !14
  %189 = insertelement <2 x i64> poison, i64 %160, i64 0, !dbg !14
  %190 = shufflevector <2 x i64> %189, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !14
  %191 = icmp slt <2 x i64> %190, <i64 -32, i64 -64>, !dbg !14
  %192 = extractelement <2 x i1> %191, i64 0, !dbg !14
  %193 = sext i1 %192 to i64, !dbg !14
  %194 = add nsw <2 x i64> %190, <i64 32, i64 64>, !dbg !14
  %195 = sub nsw <2 x i64> <i64 -33, i64 -65>, %190, !dbg !14
  %196 = select <2 x i1> %191, <2 x i64> %195, <2 x i64> %194, !dbg !14
  %197 = sdiv <2 x i64> %196, splat (i64 128), !dbg !14
  %198 = extractelement <2 x i64> %197, i64 0, !dbg !14
  %199 = xor i64 %198, %193, !dbg !14
  %200 = shl nsw i64 %199, 7, !dbg !14
  %201 = sub i64 %164, %200, !dbg !14
  %202 = add i64 %address.local449, %201, !dbg !14
  %.tr23 = trunc i64 %202 to i32, !dbg !14
  %203 = shl i32 %.tr23, 1, !dbg !14
  %dma.buffer.offset1582 = add i32 %203, 64, !dbg !14
  %204 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %205 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %179), !dbg !14
  %206 = add i64 %address.local466, %201, !dbg !14
  %.tr29 = trunc i64 %206 to i32, !dbg !14
  %207 = shl i32 %.tr29, 1, !dbg !14
  %dma.buffer.offset1579 = add i32 %207, 64, !dbg !14
  %dma.buffer.lds1581 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %205), !dbg !14
  %208 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %209 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !15
  %210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %209, <4 x float> zeroinitializer) #8, !dbg !15
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %.sext) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !15
  %address.local518 = shl i64 %address.local517, 6, !dbg !15
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !15
  %212 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local519, !dbg !15
  %213 = getelementptr [2 x i8], ptr addrspace(3) %212, i64 %address.local488, !dbg !15
  %214 = getelementptr [2 x i8], ptr addrspace(3) %213, i64 %address.local492, !dbg !15
  %address.local523 = getelementptr i8, ptr addrspace(3) %214, i64 64, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !15
  %216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %211, <4 x float> %210) #8, !dbg !15
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %215, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %214, i64 4160, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !15
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !15
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %218, <4 x float> %217) #8, !dbg !15
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %219, <4 x float> zeroinitializer) #8, !dbg !15
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !15
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %214, i64 192, !dbg !15
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !15
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %222, <4 x float> %221) #8, !dbg !15
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %214, i64 4288, !dbg !15
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !15
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !15
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %226, <4 x float> %225) #8, !dbg !15
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %227, <4 x float> zeroinitializer) #8, !dbg !15
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !15
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %214, i64 1088, !dbg !15
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !15
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %230, <4 x float> %229) #8, !dbg !15
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %231, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %214, i64 5184, !dbg !15
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !15
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !15
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %234, <4 x float> %233) #8, !dbg !15
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %235, <4 x float> zeroinitializer) #8, !dbg !15
  %238 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !15
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %214, i64 1216, !dbg !15
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !15
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %238, <4 x float> %237) #8, !dbg !15
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %239, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %214, i64 5312, !dbg !15
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !15
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %242, <4 x float> %241) #8, !dbg !15
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %209, <4 x float> zeroinitializer) #8, !dbg !15
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %211, <4 x float> %244) #8, !dbg !15
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %215, <4 x float> zeroinitializer) #8, !dbg !15
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %218, <4 x float> %246) #8, !dbg !15
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %219, <4 x float> zeroinitializer) #8, !dbg !15
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %222, <4 x float> %248) #8, !dbg !15
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !15
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %226, <4 x float> %250) #8, !dbg !15
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %227, <4 x float> zeroinitializer) #8, !dbg !15
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %230, <4 x float> %252) #8, !dbg !15
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %231, <4 x float> zeroinitializer) #8, !dbg !15
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %234, <4 x float> %254) #8, !dbg !15
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %235, <4 x float> zeroinitializer) #8, !dbg !15
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %238, <4 x float> %256) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %239, <4 x float> zeroinitializer) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %242, <4 x float> %258) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %260 = extractelement <2 x i1> %191, i64 1, !dbg !14
  %261 = sext i1 %260 to i64, !dbg !14
  %262 = extractelement <2 x i64> %197, i64 1, !dbg !14
  %263 = xor i64 %262, %261, !dbg !14
  %264 = shl i64 %188, 19, !dbg !14
  %265 = and i64 %264, 16252928, !dbg !14
  %266 = add i64 %.idx1822, %265, !dbg !14
  %267 = shl nsw i64 %263, 7, !dbg !14
  %268 = sub i64 %266, %267, !dbg !14
  %269 = add i64 %address.local449, %268, !dbg !14
  %.tr38 = trunc i64 %269 to i32, !dbg !14
  %270 = shl i32 %.tr38, 1, !dbg !14
  %dma.buffer.offset1576 = add i32 %270, 128, !dbg !14
  %271 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %272 = add i64 %address.local466, %268, !dbg !14
  %.tr44 = trunc i64 %272 to i32, !dbg !14
  %273 = shl i32 %.tr44, 1, !dbg !14
  %dma.buffer.offset1573 = add i32 %273, 128, !dbg !14
  %274 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !15
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !15
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %275, <4 x float> %216) #8, !dbg !15
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !15
  %277 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !15
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %214, i64 8256, !dbg !15
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !15
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %277, <4 x float> %276) #8, !dbg !15
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %278, <4 x float> %220) #8, !dbg !15
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %214, i64 12352, !dbg !15
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !15
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !15
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !15
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %281, <4 x float> %280) #8, !dbg !15
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %282, <4 x float> %224) #8, !dbg !15
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !15
  %285 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !15
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %214, i64 8384, !dbg !15
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !15
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %285, <4 x float> %284) #8, !dbg !15
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %286, <4 x float> %228) #8, !dbg !15
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %214, i64 12480, !dbg !15
  %289 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !15
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !15
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !15
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %289, <4 x float> %288) #8, !dbg !15
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %290, <4 x float> %232) #8, !dbg !15
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !15
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !15
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %214, i64 9280, !dbg !15
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !15
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %293, <4 x float> %292) #8, !dbg !15
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %294, <4 x float> %236) #8, !dbg !15
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %214, i64 13376, !dbg !15
  %297 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !15
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !15
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !15
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %297, <4 x float> %296) #8, !dbg !15
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %298, <4 x float> %240) #8, !dbg !15
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !15
  %301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !15
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %214, i64 9408, !dbg !15
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !15
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %301, <4 x float> %300) #8, !dbg !15
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %302, <4 x float> %243) #8, !dbg !15
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %214, i64 13504, !dbg !15
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !15
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %305, <4 x float> %304) #8, !dbg !15
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %275, <4 x float> %245) #8, !dbg !15
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %277, <4 x float> %307) #8, !dbg !15
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %278, <4 x float> %247) #8, !dbg !15
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %281, <4 x float> %309) #8, !dbg !15
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %282, <4 x float> %249) #8, !dbg !15
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %285, <4 x float> %311) #8, !dbg !15
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %286, <4 x float> %251) #8, !dbg !15
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %289, <4 x float> %313) #8, !dbg !15
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %290, <4 x float> %253) #8, !dbg !15
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %293, <4 x float> %315) #8, !dbg !15
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %294, <4 x float> %255) #8, !dbg !15
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %297, <4 x float> %317) #8, !dbg !15
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %298, <4 x float> %257) #8, !dbg !15
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %301, <4 x float> %319) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %302, <4 x float> %259) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %305, <4 x float> %321) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %323 = add nuw nsw i64 %143, 3072, !dbg !14
  %324 = lshr i64 %323, 12, !dbg !14
  %325 = add nuw nsw i64 %324, %16, !dbg !14
  %326 = add nsw i64 %160, 96, !dbg !14
  %327 = icmp slt i64 %160, -96, !dbg !14
  %328 = sub nsw i64 -97, %160, !dbg !14
  %329 = select i1 %327, i64 %328, i64 %326, !dbg !14
  %330 = sdiv i64 %329, 128, !dbg !14
  %331 = sext i1 %327 to i64, !dbg !14
  %332 = xor i64 %330, %331, !dbg !14
  %333 = shl i64 %325, 19, !dbg !14
  %334 = and i64 %333, 16252928, !dbg !14
  %335 = add i64 %.idx1822, %334, !dbg !14
  %336 = shl nsw i64 %332, 7, !dbg !14
  %337 = sub i64 %335, %336, !dbg !14
  %338 = add i64 %address.local449, %337, !dbg !14
  %.tr50 = trunc i64 %338 to i32, !dbg !14
  %339 = shl i32 %.tr50, 1, !dbg !14
  %dma.buffer.offset1570 = add i32 %339, 192, !dbg !14
  %340 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %341 = add i64 %address.local466, %337, !dbg !14
  %.tr56 = trunc i64 %341 to i32, !dbg !14
  %342 = shl i32 %.tr56, 1, !dbg !14
  %dma.buffer.offset1567 = add i32 %342, 192, !dbg !14
  %343 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !14
  %344 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !15
  %345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %344, <4 x float> %279) #8, !dbg !15
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !15
  %347 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !15
  %348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %346, <4 x float> %345) #8, !dbg !15
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %347, <4 x float> %283) #8, !dbg !15
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !15
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !15
  %352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %350, <4 x float> %349) #8, !dbg !15
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %351, <4 x float> %287) #8, !dbg !15
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !15
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !15
  %356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %354, <4 x float> %353) #8, !dbg !15
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %355, <4 x float> %291) #8, !dbg !15
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !15
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !15
  %360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %358, <4 x float> %357) #8, !dbg !15
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %359, <4 x float> %295) #8, !dbg !15
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !15
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !15
  %364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %362, <4 x float> %361) #8, !dbg !15
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %363, <4 x float> %299) #8, !dbg !15
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !15
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !15
  %368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %366, <4 x float> %365) #8, !dbg !15
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %367, <4 x float> %303) #8, !dbg !15
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !15
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !15
  %372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %370, <4 x float> %369) #8, !dbg !15
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %371, <4 x float> %306) #8, !dbg !15
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !15
  %375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %374, <4 x float> %373) #8, !dbg !15
  %376 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %344, <4 x float> %308) #8, !dbg !15
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %346, <4 x float> %376) #8, !dbg !15
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %347, <4 x float> %310) #8, !dbg !15
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %350, <4 x float> %378) #8, !dbg !15
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %351, <4 x float> %312) #8, !dbg !15
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %354, <4 x float> %380) #8, !dbg !15
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %355, <4 x float> %314) #8, !dbg !15
  %383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %358, <4 x float> %382) #8, !dbg !15
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %359, <4 x float> %316) #8, !dbg !15
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %362, <4 x float> %384) #8, !dbg !15
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %363, <4 x float> %318) #8, !dbg !15
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %366, <4 x float> %386) #8, !dbg !15
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %367, <4 x float> %320) #8, !dbg !15
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %370, <4 x float> %388) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %371, <4 x float> %322) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %374, <4 x float> %390) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %392 = add nuw nsw i64 %147, %16, !dbg !17
  %393 = shl i64 %392, 19, !dbg !17
  %394 = and i64 %393, 16252928, !dbg !17
  %address.local852 = shl nuw nsw i64 %address.local434, 6, !dbg !17
  %address.local853 = and i64 %address.local852, 16128, !dbg !17
  %address.local855 = and i64 %address.local435, 24, !dbg !17
  %.idx5760 = add i64 %394, %153, !dbg !17
  %395 = or disjoint i64 %address.local855, %.idx5760, !dbg !17
  %396 = shl i64 %395, 1, !dbg !17
  %397 = or disjoint i64 %396, %address.local853, !dbg !17
  %dma.buffer.offset1564 = trunc i64 %397 to i32, !dbg !17
  %398 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %address.local858 = shl nuw nsw i64 %address.local451, 6, !dbg !17
  %address.local859 = and i64 %address.local858, 32512, !dbg !17
  %399 = or disjoint i64 %396, %address.local859, !dbg !17
  %dma.buffer.offset1561 = trunc i64 %399 to i32, !dbg !17
  %400 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %401 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !15
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %401, <4 x float> %348) #8, !dbg !15
  %403 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !15
  %404 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !15
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %403, <4 x float> %402) #8, !dbg !15
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %404, <4 x float> %352) #8, !dbg !15
  %407 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !15
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !15
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %407, <4 x float> %406) #8, !dbg !15
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %408, <4 x float> %356) #8, !dbg !15
  %411 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !15
  %412 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !15
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %411, <4 x float> %410) #8, !dbg !15
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %412, <4 x float> %360) #8, !dbg !15
  %415 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !15
  %416 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !15
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %415, <4 x float> %414) #8, !dbg !15
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %416, <4 x float> %364) #8, !dbg !15
  %419 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !15
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !15
  %421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %419, <4 x float> %418) #8, !dbg !15
  %422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %420, <4 x float> %368) #8, !dbg !15
  %423 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !15
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !15
  %425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %423, <4 x float> %422) #8, !dbg !15
  %426 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %424, <4 x float> %372) #8, !dbg !15
  %427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !15
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !15
  %429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %427, <4 x float> %426) #8, !dbg !15
  %430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %428, <4 x float> %375) #8, !dbg !15
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !15
  %432 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %431, <4 x float> %430) #8, !dbg !15
  %433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %401, <4 x float> %377) #8, !dbg !15
  %434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %403, <4 x float> %433) #8, !dbg !15
  %435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %404, <4 x float> %379) #8, !dbg !15
  %436 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %407, <4 x float> %435) #8, !dbg !15
  %437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %408, <4 x float> %381) #8, !dbg !15
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %411, <4 x float> %437) #8, !dbg !15
  %439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %412, <4 x float> %383) #8, !dbg !15
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %415, <4 x float> %439) #8, !dbg !15
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %416, <4 x float> %385) #8, !dbg !15
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %419, <4 x float> %441) #8, !dbg !15
  %443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %420, <4 x float> %387) #8, !dbg !15
  %444 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %423, <4 x float> %443) #8, !dbg !15
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %424, <4 x float> %389) #8, !dbg !15
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %427, <4 x float> %445) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %428, <4 x float> %391) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %448 = or disjoint i64 %144, 127, !dbg !12
  %449 = icmp uge i64 %24, %448, !dbg !12
  %450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %431, <4 x float> %447) #8, !dbg !15
  %451 = and i1 %102, %449, !dbg !12
  br i1 %451, label %536, label %452, !dbg !12

452:                                              ; preds = %137
  %453 = or disjoint i64 %144, %128, !dbg !12
  %454 = lshr i64 %143, 25, !dbg !16
  %compare.high1157 = trunc nuw i64 %454 to i32, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !16
  %compare.low.base1158 = trunc i64 %453 to i32, !dbg !16
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
  %455 = insertelement <32 x i32> poison, i32 %compare.low.base1158, i64 0, !dbg !16
  %456 = insertelement <32 x i32> %455, i32 %compare.local.base, i64 1, !dbg !16
  %457 = insertelement <32 x i32> %456, i32 %compare.local.base1161, i64 2, !dbg !16
  %458 = insertelement <32 x i32> %457, i32 %compare.local.base1165, i64 3, !dbg !16
  %459 = insertelement <32 x i32> %458, i32 %compare.local.base1169, i64 4, !dbg !16
  %460 = insertelement <32 x i32> %459, i32 %compare.local.base1173, i64 5, !dbg !16
  %461 = insertelement <32 x i32> %460, i32 %compare.local.base1177, i64 6, !dbg !16
  %462 = insertelement <32 x i32> %461, i32 %compare.local.base1181, i64 7, !dbg !16
  %463 = insertelement <32 x i32> %462, i32 %compare.local.base1185, i64 8, !dbg !16
  %464 = insertelement <32 x i32> %463, i32 %compare.local.base1189, i64 9, !dbg !16
  %465 = insertelement <32 x i32> %464, i32 %compare.local.base1193, i64 10, !dbg !16
  %466 = insertelement <32 x i32> %465, i32 %compare.local.base1197, i64 11, !dbg !16
  %467 = insertelement <32 x i32> %466, i32 %compare.local.base1201, i64 12, !dbg !16
  %468 = insertelement <32 x i32> %467, i32 %compare.local.base1205, i64 13, !dbg !16
  %469 = insertelement <32 x i32> %468, i32 %compare.local.base1209, i64 14, !dbg !16
  %470 = insertelement <32 x i32> %469, i32 %compare.local.base1213, i64 15, !dbg !16
  %471 = insertelement <32 x i32> %470, i32 %compare.local.base1217, i64 16, !dbg !16
  %472 = insertelement <32 x i32> %471, i32 %compare.local.base1221, i64 17, !dbg !16
  %473 = insertelement <32 x i32> %472, i32 %compare.local.base1225, i64 18, !dbg !16
  %474 = insertelement <32 x i32> %473, i32 %compare.local.base1229, i64 19, !dbg !16
  %475 = insertelement <32 x i32> %474, i32 %compare.local.base1233, i64 20, !dbg !16
  %476 = insertelement <32 x i32> %475, i32 %compare.local.base1237, i64 21, !dbg !16
  %477 = insertelement <32 x i32> %476, i32 %compare.local.base1241, i64 22, !dbg !16
  %478 = insertelement <32 x i32> %477, i32 %compare.local.base1245, i64 23, !dbg !16
  %479 = insertelement <32 x i32> %478, i32 %compare.local.base1249, i64 24, !dbg !16
  %480 = insertelement <32 x i32> %479, i32 %compare.local.base1253, i64 25, !dbg !16
  %481 = insertelement <32 x i32> %480, i32 %compare.local.base1257, i64 26, !dbg !16
  %482 = insertelement <32 x i32> %481, i32 %compare.local.base1261, i64 27, !dbg !16
  %483 = insertelement <32 x i32> %482, i32 %compare.local.base1265, i64 28, !dbg !16
  %484 = insertelement <32 x i32> %483, i32 %compare.local.base1269, i64 29, !dbg !16
  %485 = insertelement <32 x i32> %484, i32 %compare.local.base1273, i64 30, !dbg !16
  %486 = insertelement <32 x i32> %485, i32 %compare.local.base1277, i64 31, !dbg !16
  %487 = or <32 x i32> %486, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %488 = icmp ugt <32 x i32> %136, %487, !dbg !16
  %489 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %490 = shufflevector <32 x i1> %489, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %491 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %492 = shufflevector <32 x i1> %491, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %493 = select <32 x i1> %490, <32 x i1> %488, <32 x i1> %492, !dbg !16
  %494 = select <32 x i1> %493, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
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
  %495 = insertelement <32 x i32> %455, i32 %compare.local.base1289, i64 1, !dbg !16
  %496 = insertelement <32 x i32> %495, i32 %compare.local.base1293, i64 2, !dbg !16
  %497 = insertelement <32 x i32> %496, i32 %compare.local.base1297, i64 3, !dbg !16
  %498 = insertelement <32 x i32> %497, i32 %compare.local.base1301, i64 4, !dbg !16
  %499 = insertelement <32 x i32> %498, i32 %compare.local.base1305, i64 5, !dbg !16
  %500 = insertelement <32 x i32> %499, i32 %compare.local.base1309, i64 6, !dbg !16
  %501 = insertelement <32 x i32> %500, i32 %compare.local.base1313, i64 7, !dbg !16
  %502 = insertelement <32 x i32> %501, i32 %compare.local.base1317, i64 8, !dbg !16
  %503 = insertelement <32 x i32> %502, i32 %compare.local.base1321, i64 9, !dbg !16
  %504 = insertelement <32 x i32> %503, i32 %compare.local.base1325, i64 10, !dbg !16
  %505 = insertelement <32 x i32> %504, i32 %compare.local.base1329, i64 11, !dbg !16
  %506 = insertelement <32 x i32> %505, i32 %compare.local.base1333, i64 12, !dbg !16
  %507 = insertelement <32 x i32> %506, i32 %compare.local.base1337, i64 13, !dbg !16
  %508 = insertelement <32 x i32> %507, i32 %compare.local.base1341, i64 14, !dbg !16
  %509 = insertelement <32 x i32> %508, i32 %compare.local.base1345, i64 15, !dbg !16
  %510 = insertelement <32 x i32> %509, i32 %compare.local.base1349, i64 16, !dbg !16
  %511 = insertelement <32 x i32> %510, i32 %compare.local.base1353, i64 17, !dbg !16
  %512 = insertelement <32 x i32> %511, i32 %compare.local.base1357, i64 18, !dbg !16
  %513 = insertelement <32 x i32> %512, i32 %compare.local.base1361, i64 19, !dbg !16
  %514 = insertelement <32 x i32> %513, i32 %compare.local.base1365, i64 20, !dbg !16
  %515 = insertelement <32 x i32> %514, i32 %compare.local.base1369, i64 21, !dbg !16
  %516 = insertelement <32 x i32> %515, i32 %compare.local.base1373, i64 22, !dbg !16
  %517 = insertelement <32 x i32> %516, i32 %compare.local.base1377, i64 23, !dbg !16
  %518 = insertelement <32 x i32> %517, i32 %compare.local.base1381, i64 24, !dbg !16
  %519 = insertelement <32 x i32> %518, i32 %compare.local.base1385, i64 25, !dbg !16
  %520 = insertelement <32 x i32> %519, i32 %compare.local.base1389, i64 26, !dbg !16
  %521 = insertelement <32 x i32> %520, i32 %compare.local.base1393, i64 27, !dbg !16
  %522 = insertelement <32 x i32> %521, i32 %compare.local.base1397, i64 28, !dbg !16
  %523 = insertelement <32 x i32> %522, i32 %compare.local.base1401, i64 29, !dbg !16
  %524 = insertelement <32 x i32> %523, i32 %compare.local.base1405, i64 30, !dbg !16
  %525 = insertelement <32 x i32> %524, i32 %compare.local.base1409, i64 31, !dbg !16
  %526 = or <32 x i32> %525, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %527 = icmp ugt <32 x i32> %134, %526, !dbg !16
  %528 = insertelement <32 x i1> poison, i1 %compare.high.equal1283, i64 0, !dbg !16
  %529 = shufflevector <32 x i1> %528, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %530 = insertelement <32 x i1> poison, i1 %compare.high.order1284, i64 0, !dbg !16
  %531 = shufflevector <32 x i1> %530, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %532 = select <32 x i1> %529, <32 x i1> %527, <32 x i1> %531, !dbg !16
  %533 = select <32 x i1> %532, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %534 = insertvalue [2 x <32 x float>] poison, <32 x float> %494, 0, !dbg !12
  %535 = insertvalue [2 x <32 x float>] %534, <32 x float> %533, 1, !dbg !12
  br label %536, !dbg !12

536:                                              ; preds = %452, %137
  %537 = phi [2 x <32 x float>] [ %535, %452 ], [ zeroinitializer, %137 ], !dbg !12
  %538 = extractvalue [2 x <32 x float>] %537, 0, !dbg !19
  %539 = shufflevector <32 x float> %538, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %540 = fadd <4 x float> %405, %539, !dbg !19
  %541 = extractelement <4 x float> %540, i64 0, !dbg !20
  %542 = tail call float @llvm.amdgcn.exp2.f32(float %541), !dbg !20
  %543 = insertelement <4 x float> poison, float %542, i64 0, !dbg !21
  %544 = extractelement <4 x float> %540, i64 1, !dbg !20
  %545 = tail call float @llvm.amdgcn.exp2.f32(float %544), !dbg !20
  %546 = insertelement <4 x float> %543, float %545, i64 1, !dbg !21
  %547 = fadd float %542, %545, !dbg !22
  %548 = extractelement <4 x float> %540, i64 2, !dbg !20
  %549 = tail call float @llvm.amdgcn.exp2.f32(float %548), !dbg !20
  %550 = insertelement <4 x float> %546, float %549, i64 2, !dbg !21
  %551 = fadd float %549, %547, !dbg !22
  %552 = extractelement <4 x float> %540, i64 3, !dbg !20
  %553 = tail call float @llvm.amdgcn.exp2.f32(float %552), !dbg !20
  %554 = insertelement <4 x float> %550, float %553, i64 3, !dbg !21
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %554) #8, !dbg !21, !frisk.register_lifetimes !5
  %555 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !21
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %555) #8, !dbg !22, !frisk.register_lifetimes !5
  %556 = fadd float %553, %551, !dbg !22
  %557 = shufflevector <32 x float> %538, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %558 = fadd <4 x float> %409, %557, !dbg !19
  %559 = extractelement <4 x float> %558, i64 0, !dbg !20
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %559), !dbg !20
  %561 = insertelement <4 x float> poison, float %560, i64 0, !dbg !21
  %562 = fadd float %560, %556, !dbg !22
  %563 = extractelement <4 x float> %558, i64 1, !dbg !20
  %564 = tail call float @llvm.amdgcn.exp2.f32(float %563), !dbg !20
  %565 = insertelement <4 x float> %561, float %564, i64 1, !dbg !21
  %566 = fadd float %564, %562, !dbg !22
  %567 = extractelement <4 x float> %558, i64 2, !dbg !20
  %568 = tail call float @llvm.amdgcn.exp2.f32(float %567), !dbg !20
  %569 = insertelement <4 x float> %565, float %568, i64 2, !dbg !21
  %570 = fadd float %568, %566, !dbg !22
  %571 = extractelement <4 x float> %558, i64 3, !dbg !20
  %572 = tail call float @llvm.amdgcn.exp2.f32(float %571), !dbg !20
  %573 = insertelement <4 x float> %569, float %572, i64 3, !dbg !21
  %math.result309 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %573) #8, !dbg !21, !frisk.register_lifetimes !5
  %574 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !21
  %math.packed310 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %574) #8, !dbg !22, !frisk.register_lifetimes !5
  %575 = fadd float %572, %570, !dbg !22
  %576 = shufflevector <32 x float> %538, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %577 = fadd <4 x float> %413, %576, !dbg !19
  %578 = extractelement <4 x float> %577, i64 0, !dbg !20
  %579 = tail call float @llvm.amdgcn.exp2.f32(float %578), !dbg !20
  %580 = insertelement <4 x float> poison, float %579, i64 0, !dbg !21
  %581 = fadd float %579, %575, !dbg !22
  %582 = extractelement <4 x float> %577, i64 1, !dbg !20
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %582), !dbg !20
  %584 = insertelement <4 x float> %580, float %583, i64 1, !dbg !21
  %585 = fadd float %583, %581, !dbg !22
  %586 = extractelement <4 x float> %577, i64 2, !dbg !20
  %587 = tail call float @llvm.amdgcn.exp2.f32(float %586), !dbg !20
  %588 = insertelement <4 x float> %584, float %587, i64 2, !dbg !21
  %589 = fadd float %587, %585, !dbg !22
  %590 = extractelement <4 x float> %577, i64 3, !dbg !20
  %591 = tail call float @llvm.amdgcn.exp2.f32(float %590), !dbg !20
  %592 = insertelement <4 x float> %588, float %591, i64 3, !dbg !21
  %math.result311 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %592) #8, !dbg !21, !frisk.register_lifetimes !5
  %593 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !21
  %math.packed312 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %593) #8, !dbg !22, !frisk.register_lifetimes !5
  %594 = fadd float %591, %589, !dbg !22
  %595 = shufflevector <32 x float> %538, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %596 = fadd <4 x float> %417, %595, !dbg !19
  %597 = extractelement <4 x float> %596, i64 0, !dbg !20
  %598 = tail call float @llvm.amdgcn.exp2.f32(float %597), !dbg !20
  %599 = insertelement <4 x float> poison, float %598, i64 0, !dbg !21
  %600 = fadd float %598, %594, !dbg !22
  %601 = extractelement <4 x float> %596, i64 1, !dbg !20
  %602 = tail call float @llvm.amdgcn.exp2.f32(float %601), !dbg !20
  %603 = insertelement <4 x float> %599, float %602, i64 1, !dbg !21
  %604 = fadd float %602, %600, !dbg !22
  %605 = extractelement <4 x float> %596, i64 2, !dbg !20
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %605), !dbg !20
  %607 = insertelement <4 x float> %603, float %606, i64 2, !dbg !21
  %608 = fadd float %606, %604, !dbg !22
  %609 = extractelement <4 x float> %596, i64 3, !dbg !20
  %610 = tail call float @llvm.amdgcn.exp2.f32(float %609), !dbg !20
  %611 = insertelement <4 x float> %607, float %610, i64 3, !dbg !21
  %math.result313 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %611) #8, !dbg !21, !frisk.register_lifetimes !5
  %612 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !21
  %math.packed314 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %612) #8, !dbg !22, !frisk.register_lifetimes !5
  %613 = fadd float %610, %608, !dbg !22
  %614 = shufflevector <32 x float> %538, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %615 = fadd <4 x float> %421, %614, !dbg !19
  %616 = extractelement <4 x float> %615, i64 0, !dbg !20
  %617 = tail call float @llvm.amdgcn.exp2.f32(float %616), !dbg !20
  %618 = insertelement <4 x float> poison, float %617, i64 0, !dbg !21
  %619 = fadd float %617, %613, !dbg !22
  %620 = extractelement <4 x float> %615, i64 1, !dbg !20
  %621 = tail call float @llvm.amdgcn.exp2.f32(float %620), !dbg !20
  %622 = insertelement <4 x float> %618, float %621, i64 1, !dbg !21
  %623 = fadd float %621, %619, !dbg !22
  %624 = extractelement <4 x float> %615, i64 2, !dbg !20
  %625 = tail call float @llvm.amdgcn.exp2.f32(float %624), !dbg !20
  %626 = insertelement <4 x float> %622, float %625, i64 2, !dbg !21
  %627 = fadd float %625, %623, !dbg !22
  %628 = extractelement <4 x float> %615, i64 3, !dbg !20
  %629 = tail call float @llvm.amdgcn.exp2.f32(float %628), !dbg !20
  %630 = insertelement <4 x float> %626, float %629, i64 3, !dbg !21
  %math.result315 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %630) #8, !dbg !21, !frisk.register_lifetimes !5
  %631 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !21
  %math.packed316 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %631) #8, !dbg !22, !frisk.register_lifetimes !5
  %632 = fadd float %629, %627, !dbg !22
  %633 = shufflevector <32 x float> %538, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %634 = fadd <4 x float> %425, %633, !dbg !19
  %635 = extractelement <4 x float> %634, i64 0, !dbg !20
  %636 = tail call float @llvm.amdgcn.exp2.f32(float %635), !dbg !20
  %637 = insertelement <4 x float> poison, float %636, i64 0, !dbg !21
  %638 = fadd float %636, %632, !dbg !22
  %639 = extractelement <4 x float> %634, i64 1, !dbg !20
  %640 = tail call float @llvm.amdgcn.exp2.f32(float %639), !dbg !20
  %641 = insertelement <4 x float> %637, float %640, i64 1, !dbg !21
  %642 = fadd float %640, %638, !dbg !22
  %643 = extractelement <4 x float> %634, i64 2, !dbg !20
  %644 = tail call float @llvm.amdgcn.exp2.f32(float %643), !dbg !20
  %645 = insertelement <4 x float> %641, float %644, i64 2, !dbg !21
  %646 = fadd float %644, %642, !dbg !22
  %647 = extractelement <4 x float> %634, i64 3, !dbg !20
  %648 = tail call float @llvm.amdgcn.exp2.f32(float %647), !dbg !20
  %649 = insertelement <4 x float> %645, float %648, i64 3, !dbg !21
  %math.result317 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %649) #8, !dbg !21, !frisk.register_lifetimes !5
  %650 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !21
  %math.packed318 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %650) #8, !dbg !22, !frisk.register_lifetimes !5
  %651 = fadd float %648, %646, !dbg !22
  %652 = shufflevector <32 x float> %538, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %653 = fadd <4 x float> %429, %652, !dbg !19
  %654 = extractelement <4 x float> %653, i64 0, !dbg !20
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %654), !dbg !20
  %656 = insertelement <4 x float> poison, float %655, i64 0, !dbg !21
  %657 = fadd float %655, %651, !dbg !22
  %658 = extractelement <4 x float> %653, i64 1, !dbg !20
  %659 = tail call float @llvm.amdgcn.exp2.f32(float %658), !dbg !20
  %660 = insertelement <4 x float> %656, float %659, i64 1, !dbg !21
  %661 = fadd float %659, %657, !dbg !22
  %662 = extractelement <4 x float> %653, i64 2, !dbg !20
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %662), !dbg !20
  %664 = insertelement <4 x float> %660, float %663, i64 2, !dbg !21
  %665 = fadd float %663, %661, !dbg !22
  %666 = extractelement <4 x float> %653, i64 3, !dbg !20
  %667 = tail call float @llvm.amdgcn.exp2.f32(float %666), !dbg !20
  %668 = insertelement <4 x float> %664, float %667, i64 3, !dbg !21
  %math.result319 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %668) #8, !dbg !21, !frisk.register_lifetimes !5
  %669 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !21
  %math.packed320 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %669) #8, !dbg !22, !frisk.register_lifetimes !5
  %670 = fadd float %667, %665, !dbg !22
  %671 = shufflevector <32 x float> %538, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %672 = fadd <4 x float> %432, %671, !dbg !19
  %673 = extractelement <4 x float> %672, i64 0, !dbg !20
  %674 = tail call float @llvm.amdgcn.exp2.f32(float %673), !dbg !20
  %675 = insertelement <4 x float> poison, float %674, i64 0, !dbg !21
  %676 = fadd float %674, %670, !dbg !22
  %677 = extractelement <4 x float> %672, i64 1, !dbg !20
  %678 = tail call float @llvm.amdgcn.exp2.f32(float %677), !dbg !20
  %679 = insertelement <4 x float> %675, float %678, i64 1, !dbg !21
  %680 = fadd float %678, %676, !dbg !22
  %681 = extractelement <4 x float> %672, i64 2, !dbg !20
  %682 = tail call float @llvm.amdgcn.exp2.f32(float %681), !dbg !20
  %683 = insertelement <4 x float> %679, float %682, i64 2, !dbg !21
  %684 = fadd float %682, %680, !dbg !22
  %685 = extractelement <4 x float> %672, i64 3, !dbg !20
  %686 = tail call float @llvm.amdgcn.exp2.f32(float %685), !dbg !20
  %687 = insertelement <4 x float> %683, float %686, i64 3, !dbg !21
  %math.result321 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %687) #8, !dbg !21, !frisk.register_lifetimes !5
  %688 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !21
  %math.packed322 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %688) #8, !dbg !22, !frisk.register_lifetimes !5
  %689 = fadd float %686, %684, !dbg !22
  %690 = extractvalue [2 x <32 x float>] %537, 1, !dbg !19
  %691 = shufflevector <32 x float> %690, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %692 = fadd <4 x float> %434, %691, !dbg !19
  %693 = extractelement <4 x float> %692, i64 0, !dbg !20
  %694 = tail call float @llvm.amdgcn.exp2.f32(float %693), !dbg !20
  %695 = insertelement <4 x float> poison, float %694, i64 0, !dbg !21
  %696 = extractelement <4 x float> %692, i64 1, !dbg !20
  %697 = tail call float @llvm.amdgcn.exp2.f32(float %696), !dbg !20
  %698 = insertelement <4 x float> %695, float %697, i64 1, !dbg !21
  %699 = fadd float %694, %697, !dbg !22
  %700 = extractelement <4 x float> %692, i64 2, !dbg !20
  %701 = tail call float @llvm.amdgcn.exp2.f32(float %700), !dbg !20
  %702 = insertelement <4 x float> %698, float %701, i64 2, !dbg !21
  %703 = fadd float %701, %699, !dbg !22
  %704 = extractelement <4 x float> %692, i64 3, !dbg !20
  %705 = tail call float @llvm.amdgcn.exp2.f32(float %704), !dbg !20
  %706 = insertelement <4 x float> %702, float %705, i64 3, !dbg !21
  %math.result323 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %706) #8, !dbg !21, !frisk.register_lifetimes !5
  %707 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !21
  %math.packed324 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %707) #8, !dbg !22, !frisk.register_lifetimes !5
  %708 = fadd float %705, %703, !dbg !22
  %709 = shufflevector <32 x float> %690, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %710 = fadd <4 x float> %436, %709, !dbg !19
  %711 = extractelement <4 x float> %710, i64 0, !dbg !20
  %712 = tail call float @llvm.amdgcn.exp2.f32(float %711), !dbg !20
  %713 = insertelement <4 x float> poison, float %712, i64 0, !dbg !21
  %714 = fadd float %712, %708, !dbg !22
  %715 = extractelement <4 x float> %710, i64 1, !dbg !20
  %716 = tail call float @llvm.amdgcn.exp2.f32(float %715), !dbg !20
  %717 = insertelement <4 x float> %713, float %716, i64 1, !dbg !21
  %718 = fadd float %716, %714, !dbg !22
  %719 = extractelement <4 x float> %710, i64 2, !dbg !20
  %720 = tail call float @llvm.amdgcn.exp2.f32(float %719), !dbg !20
  %721 = insertelement <4 x float> %717, float %720, i64 2, !dbg !21
  %722 = fadd float %720, %718, !dbg !22
  %723 = extractelement <4 x float> %710, i64 3, !dbg !20
  %724 = tail call float @llvm.amdgcn.exp2.f32(float %723), !dbg !20
  %725 = insertelement <4 x float> %721, float %724, i64 3, !dbg !21
  %math.result325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %725) #8, !dbg !21, !frisk.register_lifetimes !5
  %726 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !21
  %math.packed326 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %726) #8, !dbg !22, !frisk.register_lifetimes !5
  %727 = fadd float %724, %722, !dbg !22
  %728 = shufflevector <32 x float> %690, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %729 = fadd <4 x float> %438, %728, !dbg !19
  %730 = extractelement <4 x float> %729, i64 0, !dbg !20
  %731 = tail call float @llvm.amdgcn.exp2.f32(float %730), !dbg !20
  %732 = insertelement <4 x float> poison, float %731, i64 0, !dbg !21
  %733 = fadd float %731, %727, !dbg !22
  %734 = extractelement <4 x float> %729, i64 1, !dbg !20
  %735 = tail call float @llvm.amdgcn.exp2.f32(float %734), !dbg !20
  %736 = insertelement <4 x float> %732, float %735, i64 1, !dbg !21
  %737 = fadd float %735, %733, !dbg !22
  %738 = extractelement <4 x float> %729, i64 2, !dbg !20
  %739 = tail call float @llvm.amdgcn.exp2.f32(float %738), !dbg !20
  %740 = insertelement <4 x float> %736, float %739, i64 2, !dbg !21
  %741 = fadd float %739, %737, !dbg !22
  %742 = extractelement <4 x float> %729, i64 3, !dbg !20
  %743 = tail call float @llvm.amdgcn.exp2.f32(float %742), !dbg !20
  %744 = insertelement <4 x float> %740, float %743, i64 3, !dbg !21
  %math.result327 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %744) #8, !dbg !21, !frisk.register_lifetimes !5
  %745 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !21
  %math.packed328 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %745) #8, !dbg !22, !frisk.register_lifetimes !5
  %746 = fadd float %743, %741, !dbg !22
  %747 = shufflevector <32 x float> %690, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %748 = fadd <4 x float> %440, %747, !dbg !19
  %749 = extractelement <4 x float> %748, i64 0, !dbg !20
  %750 = tail call float @llvm.amdgcn.exp2.f32(float %749), !dbg !20
  %751 = insertelement <4 x float> poison, float %750, i64 0, !dbg !21
  %752 = fadd float %750, %746, !dbg !22
  %753 = extractelement <4 x float> %748, i64 1, !dbg !20
  %754 = tail call float @llvm.amdgcn.exp2.f32(float %753), !dbg !20
  %755 = insertelement <4 x float> %751, float %754, i64 1, !dbg !21
  %756 = fadd float %754, %752, !dbg !22
  %757 = extractelement <4 x float> %748, i64 2, !dbg !20
  %758 = tail call float @llvm.amdgcn.exp2.f32(float %757), !dbg !20
  %759 = insertelement <4 x float> %755, float %758, i64 2, !dbg !21
  %760 = fadd float %758, %756, !dbg !22
  %761 = extractelement <4 x float> %748, i64 3, !dbg !20
  %762 = tail call float @llvm.amdgcn.exp2.f32(float %761), !dbg !20
  %763 = insertelement <4 x float> %759, float %762, i64 3, !dbg !21
  %math.result329 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %763) #8, !dbg !21, !frisk.register_lifetimes !5
  %764 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !21
  %math.packed330 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %764) #8, !dbg !22, !frisk.register_lifetimes !5
  %765 = fadd float %762, %760, !dbg !22
  %766 = shufflevector <32 x float> %690, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %767 = fadd <4 x float> %442, %766, !dbg !19
  %768 = extractelement <4 x float> %767, i64 0, !dbg !20
  %769 = tail call float @llvm.amdgcn.exp2.f32(float %768), !dbg !20
  %770 = insertelement <4 x float> poison, float %769, i64 0, !dbg !21
  %771 = fadd float %769, %765, !dbg !22
  %772 = extractelement <4 x float> %767, i64 1, !dbg !20
  %773 = tail call float @llvm.amdgcn.exp2.f32(float %772), !dbg !20
  %774 = insertelement <4 x float> %770, float %773, i64 1, !dbg !21
  %775 = fadd float %773, %771, !dbg !22
  %776 = extractelement <4 x float> %767, i64 2, !dbg !20
  %777 = tail call float @llvm.amdgcn.exp2.f32(float %776), !dbg !20
  %778 = insertelement <4 x float> %774, float %777, i64 2, !dbg !21
  %779 = fadd float %777, %775, !dbg !22
  %780 = extractelement <4 x float> %767, i64 3, !dbg !20
  %781 = tail call float @llvm.amdgcn.exp2.f32(float %780), !dbg !20
  %782 = insertelement <4 x float> %778, float %781, i64 3, !dbg !21
  %math.result331 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %782) #8, !dbg !21, !frisk.register_lifetimes !5
  %783 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !21
  %math.packed332 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %783) #8, !dbg !22, !frisk.register_lifetimes !5
  %784 = fadd float %781, %779, !dbg !22
  %785 = shufflevector <32 x float> %690, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %786 = fadd <4 x float> %444, %785, !dbg !19
  %787 = extractelement <4 x float> %786, i64 0, !dbg !20
  %788 = tail call float @llvm.amdgcn.exp2.f32(float %787), !dbg !20
  %789 = insertelement <4 x float> poison, float %788, i64 0, !dbg !21
  %790 = fadd float %788, %784, !dbg !22
  %791 = extractelement <4 x float> %786, i64 1, !dbg !20
  %792 = tail call float @llvm.amdgcn.exp2.f32(float %791), !dbg !20
  %793 = insertelement <4 x float> %789, float %792, i64 1, !dbg !21
  %794 = fadd float %792, %790, !dbg !22
  %795 = extractelement <4 x float> %786, i64 2, !dbg !20
  %796 = tail call float @llvm.amdgcn.exp2.f32(float %795), !dbg !20
  %797 = insertelement <4 x float> %793, float %796, i64 2, !dbg !21
  %798 = fadd float %796, %794, !dbg !22
  %799 = extractelement <4 x float> %786, i64 3, !dbg !20
  %800 = tail call float @llvm.amdgcn.exp2.f32(float %799), !dbg !20
  %801 = insertelement <4 x float> %797, float %800, i64 3, !dbg !21
  %math.result333 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %801) #8, !dbg !21, !frisk.register_lifetimes !5
  %802 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !21
  %math.packed334 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %802) #8, !dbg !22, !frisk.register_lifetimes !5
  %803 = fadd float %800, %798, !dbg !22
  %804 = shufflevector <32 x float> %690, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %805 = fadd <4 x float> %446, %804, !dbg !19
  %806 = extractelement <4 x float> %805, i64 0, !dbg !20
  %807 = tail call float @llvm.amdgcn.exp2.f32(float %806), !dbg !20
  %808 = insertelement <4 x float> poison, float %807, i64 0, !dbg !21
  %809 = fadd float %807, %803, !dbg !22
  %810 = extractelement <4 x float> %805, i64 1, !dbg !20
  %811 = tail call float @llvm.amdgcn.exp2.f32(float %810), !dbg !20
  %812 = insertelement <4 x float> %808, float %811, i64 1, !dbg !21
  %813 = fadd float %811, %809, !dbg !22
  %814 = extractelement <4 x float> %805, i64 2, !dbg !20
  %815 = tail call float @llvm.amdgcn.exp2.f32(float %814), !dbg !20
  %816 = insertelement <4 x float> %812, float %815, i64 2, !dbg !21
  %817 = fadd float %815, %813, !dbg !22
  %818 = extractelement <4 x float> %805, i64 3, !dbg !20
  %819 = tail call float @llvm.amdgcn.exp2.f32(float %818), !dbg !20
  %820 = insertelement <4 x float> %816, float %819, i64 3, !dbg !21
  %math.result335 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %820) #8, !dbg !21, !frisk.register_lifetimes !5
  %821 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !21
  %math.packed336 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %821) #8, !dbg !22, !frisk.register_lifetimes !5
  %822 = fadd float %819, %817, !dbg !22
  %823 = shufflevector <32 x float> %690, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %824 = fadd <4 x float> %450, %823, !dbg !19
  %825 = extractelement <4 x float> %824, i64 0, !dbg !20
  %826 = tail call float @llvm.amdgcn.exp2.f32(float %825), !dbg !20
  %827 = insertelement <4 x float> poison, float %826, i64 0, !dbg !21
  %828 = fadd float %826, %822, !dbg !22
  %829 = extractelement <4 x float> %824, i64 1, !dbg !20
  %830 = tail call float @llvm.amdgcn.exp2.f32(float %829), !dbg !20
  %831 = insertelement <4 x float> %827, float %830, i64 1, !dbg !21
  %832 = fadd float %830, %828, !dbg !22
  %833 = extractelement <4 x float> %824, i64 2, !dbg !20
  %834 = tail call float @llvm.amdgcn.exp2.f32(float %833), !dbg !20
  %835 = insertelement <4 x float> %831, float %834, i64 2, !dbg !21
  %836 = fadd float %834, %832, !dbg !22
  %837 = extractelement <4 x float> %824, i64 3, !dbg !20
  %838 = tail call float @llvm.amdgcn.exp2.f32(float %837), !dbg !20
  %839 = insertelement <4 x float> %835, float %838, i64 3, !dbg !21
  %math.result337 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %839) #8, !dbg !21, !frisk.register_lifetimes !5
  %840 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !21
  %math.packed338 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %840) #8, !dbg !22, !frisk.register_lifetimes !5
  %841 = fadd float %838, %836, !dbg !22
  %842 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !22
  %843 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %842), !dbg !22
  %844 = add nuw nsw i32 %843, 64, !dbg !22
  %845 = xor i32 %843, 16, !dbg !22
  %846 = and i32 %844, 192, !dbg !22
  %847 = icmp samesign ult i32 %845, %846, !dbg !22
  %848 = select i1 %847, i32 %845, i32 %843, !dbg !22
  %849 = shl nuw nsw i32 %848, 2, !dbg !22
  %850 = bitcast float %689 to i32, !dbg !22
  %851 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %849, i32 %850), !dbg !22
  %852 = bitcast i32 %851 to float, !dbg !22
  %853 = fadd float %689, %852, !dbg !22
  %854 = xor i32 %843, 32, !dbg !22
  %855 = icmp samesign ult i32 %854, %846, !dbg !22
  %856 = select i1 %855, i32 %854, i32 %843, !dbg !22
  %857 = shl nuw nsw i32 %856, 2, !dbg !22
  %858 = bitcast float %853 to i32, !dbg !22
  %859 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %857, i32 %858), !dbg !22
  %860 = bitcast i32 %859 to float, !dbg !22
  %861 = fadd float %853, %860, !dbg !22
  %862 = extractvalue [2 x <1 x float>] %141, 0, !dbg !22
  %863 = extractelement <1 x float> %862, i64 0, !dbg !22
  %864 = fadd float %863, %861, !dbg !22
  %865 = insertelement <1 x float> poison, float %864, i64 0, !dbg !11
  %866 = insertvalue [2 x <1 x float>] poison, <1 x float> %865, 0, !dbg !11
  %867 = bitcast float %841 to i32, !dbg !22
  %868 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %849, i32 %867), !dbg !22
  %869 = bitcast i32 %868 to float, !dbg !22
  %870 = fadd float %841, %869, !dbg !22
  %871 = bitcast float %870 to i32, !dbg !22
  %872 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %857, i32 %871), !dbg !22
  %873 = bitcast i32 %872 to float, !dbg !22
  %874 = fadd float %870, %873, !dbg !22
  %875 = extractvalue [2 x <1 x float>] %141, 1, !dbg !22
  %876 = extractelement <1 x float> %875, i64 0, !dbg !22
  %877 = fadd float %876, %874, !dbg !22
  %878 = insertelement <1 x float> poison, float %877, i64 0, !dbg !11
  %879 = insertvalue [2 x <1 x float>] %866, <1 x float> %878, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread864 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !23, !range !9, !frisk.register_lifetimes !5
  %address.local865 = shl nuw nsw i32 %address.thread864, 3, !dbg !23
  %address.local866 = and i32 %address.local865, 120, !dbg !23
  %address.local867 = shl nuw nsw i32 %address.thread864, 5, !dbg !23
  %address.local868 = and i32 %address.local867, 1536, !dbg !23
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !23
  %address.local870 = zext nneg i32 %address.local869 to i64, !dbg !23
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !23
  %880 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !23
  %881 = bitcast <4 x i32> %880 to <8 x half>, !dbg !23
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
  %882 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !23
  %883 = bitcast <4 x i32> %882 to <8 x half>, !dbg !23
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local883, i64 512, !dbg !23
  %884 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !23
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !23
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local883, i64 768, !dbg !23
  %886 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !23
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !23
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local871, i64 4096, !dbg !23
  %888 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !23
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !23
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4352, !dbg !23
  %890 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !23
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !23
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4608, !dbg !23
  %892 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !23
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !23
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4864, !dbg !23
  %894 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !23
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !8
  %address.local908 = and i64 %address.local907, 1536, !dbg !8
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !8
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !8
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !8
  %896 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !17
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !17
  %address.local913 = and i64 %address.local912, 8064, !dbg !17
  %address.local915 = and i64 %address.local907, 24, !dbg !17
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !17
  %897 = or disjoint i64 %.idx5760, %address.local916, !dbg !17
  %898 = trunc i64 %897 to i32, !dbg !17
  %.tr68 = shl i32 %898, 1, !dbg !17
  %dma.buffer.offset1558 = or i32 %.tr68, 64, !dbg !17
  %dma.buffer.lds1560 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %896), !dbg !17
  %899 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !8
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !8
  %address.local920 = and i64 %address.local919, 3584, !dbg !8
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !8
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !8
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !8
  %900 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !17
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !17
  %address.local925 = and i64 %address.local924, 16256, !dbg !17
  %address.local928 = or disjoint i64 %address.local925, %address.local915, !dbg !17
  %901 = or disjoint i64 %.idx5760, %address.local928, !dbg !17
  %902 = trunc i64 %901 to i32, !dbg !17
  %.tr72 = shl i32 %902, 1, !dbg !17
  %dma.buffer.offset1555 = or i32 %.tr72, 64, !dbg !17
  %dma.buffer.lds1557 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %900), !dbg !17
  %903 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %904 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %905 = extractvalue [2 x <8 x float>] %142, 0, !dbg !24
  %906 = shufflevector <8 x float> %905, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %904, <4 x float> %906) #8, !dbg !23
  %908 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %909 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %908, <4 x float> %907) #8, !dbg !23
  %910 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %910, <4 x float> %909) #8, !dbg !23
  %912 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %912, <4 x float> %911) #8, !dbg !23
  %914 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %915 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %914, <4 x float> %913) #8, !dbg !23
  %916 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %916, <4 x float> %915) #8, !dbg !23
  %918 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %919 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %918, <4 x float> %917) #8, !dbg !23
  %920 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %921 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %920, <4 x float> %919) #8, !dbg !23
  %922 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %923 = shufflevector <8 x float> %905, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %922, <4 x float> %923) #8, !dbg !23
  %925 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %925, <4 x float> %924) #8, !dbg !23
  %927 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %927, <4 x float> %926) #8, !dbg !23
  %929 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %929, <4 x float> %928) #8, !dbg !23
  %931 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %932 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %931, <4 x float> %930) #8, !dbg !23
  %933 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %934 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %933, <4 x float> %932) #8, !dbg !23
  %935 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %935, <4 x float> %934) #8, !dbg !23
  %937 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %937, <4 x float> %936) #8, !dbg !23
  %939 = extractvalue [2 x <8 x float>] %142, 1, !dbg !24
  %940 = shufflevector <8 x float> %939, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %941 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %904, <4 x float> %940) #8, !dbg !23
  %942 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %908, <4 x float> %941) #8, !dbg !23
  %943 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %910, <4 x float> %942) #8, !dbg !23
  %944 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %912, <4 x float> %943) #8, !dbg !23
  %945 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %914, <4 x float> %944) #8, !dbg !23
  %946 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %916, <4 x float> %945) #8, !dbg !23
  %947 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %918, <4 x float> %946) #8, !dbg !23
  %948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %920, <4 x float> %947) #8, !dbg !23
  %949 = shufflevector <8 x float> %939, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %950 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %922, <4 x float> %949) #8, !dbg !23
  %951 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %925, <4 x float> %950) #8, !dbg !23
  %952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %927, <4 x float> %951) #8, !dbg !23
  %953 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %929, <4 x float> %952) #8, !dbg !23
  %954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %931, <4 x float> %953) #8, !dbg !23
  %955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %933, <4 x float> %954) #8, !dbg !23
  %956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %935, <4 x float> %955) #8, !dbg !23
  %957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %937, <4 x float> %956) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %958 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !23
  %959 = bitcast <4 x i32> %958 to <8 x half>, !dbg !23
  %960 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !23
  %961 = bitcast <4 x i32> %960 to <8 x half>, !dbg !23
  %962 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !23
  %963 = bitcast <4 x i32> %962 to <8 x half>, !dbg !23
  %964 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !23
  %965 = bitcast <4 x i32> %964 to <8 x half>, !dbg !23
  %966 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !23
  %967 = bitcast <4 x i32> %966 to <8 x half>, !dbg !23
  %968 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !23
  %969 = bitcast <4 x i32> %968 to <8 x half>, !dbg !23
  %970 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !23
  %971 = bitcast <4 x i32> %970 to <8 x half>, !dbg !23
  %972 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !23
  %973 = bitcast <4 x i32> %972 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %dma.buffer.offset1552 = or i32 %.tr68, 128, !dbg !17
  %974 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %dma.buffer.offset1549 = or i32 %.tr72, 128, !dbg !17
  %975 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %976 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %977 = extractvalue [2 x <8 x float>] %140, 0, !dbg !24
  %978 = shufflevector <8 x float> %977, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %979 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %976, <4 x float> %978) #8, !dbg !23
  %980 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %980, <4 x float> %979) #8, !dbg !23
  %982 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %983 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %982, <4 x float> %981) #8, !dbg !23
  %984 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %985 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %984, <4 x float> %983) #8, !dbg !23
  %986 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %986, <4 x float> %985) #8, !dbg !23
  %988 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %989 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %988, <4 x float> %987) #8, !dbg !23
  %990 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %990, <4 x float> %989) #8, !dbg !23
  %992 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %992, <4 x float> %991) #8, !dbg !23
  %994 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %995 = shufflevector <8 x float> %977, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %994, <4 x float> %995) #8, !dbg !23
  %997 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %997, <4 x float> %996) #8, !dbg !23
  %999 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %999, <4 x float> %998) #8, !dbg !23
  %1001 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1001, <4 x float> %1000) #8, !dbg !23
  %1003 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1003, <4 x float> %1002) #8, !dbg !23
  %1005 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1005, <4 x float> %1004) #8, !dbg !23
  %1007 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1007, <4 x float> %1006) #8, !dbg !23
  %1009 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1009, <4 x float> %1008) #8, !dbg !23
  %1011 = extractvalue [2 x <8 x float>] %140, 1, !dbg !24
  %1012 = shufflevector <8 x float> %1011, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %976, <4 x float> %1012) #8, !dbg !23
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %980, <4 x float> %1013) #8, !dbg !23
  %1015 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %982, <4 x float> %1014) #8, !dbg !23
  %1016 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %984, <4 x float> %1015) #8, !dbg !23
  %1017 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %986, <4 x float> %1016) #8, !dbg !23
  %1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %988, <4 x float> %1017) #8, !dbg !23
  %1019 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %990, <4 x float> %1018) #8, !dbg !23
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %992, <4 x float> %1019) #8, !dbg !23
  %1021 = shufflevector <8 x float> %1011, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1022 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %994, <4 x float> %1021) #8, !dbg !23
  %1023 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %997, <4 x float> %1022) #8, !dbg !23
  %1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %999, <4 x float> %1023) #8, !dbg !23
  %1025 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1001, <4 x float> %1024) #8, !dbg !23
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1003, <4 x float> %1025) #8, !dbg !23
  %1027 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1005, <4 x float> %1026) #8, !dbg !23
  %1028 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1007, <4 x float> %1027) #8, !dbg !23
  %1029 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1009, <4 x float> %1028) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1030 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !23
  %1031 = bitcast <4 x i32> %1030 to <8 x half>, !dbg !23
  %1032 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !23
  %1033 = bitcast <4 x i32> %1032 to <8 x half>, !dbg !23
  %1034 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !23
  %1035 = bitcast <4 x i32> %1034 to <8 x half>, !dbg !23
  %1036 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !23
  %1037 = bitcast <4 x i32> %1036 to <8 x half>, !dbg !23
  %1038 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !23
  %1039 = bitcast <4 x i32> %1038 to <8 x half>, !dbg !23
  %1040 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !23
  %1041 = bitcast <4 x i32> %1040 to <8 x half>, !dbg !23
  %1042 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !23
  %1043 = bitcast <4 x i32> %1042 to <8 x half>, !dbg !23
  %1044 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !23
  %1045 = bitcast <4 x i32> %1044 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1046 = or disjoint i64 %.idx5760, 96, !dbg !17
  %1047 = or disjoint i64 %address.local916, %1046, !dbg !17
  %.tr84 = trunc i64 %1047 to i32, !dbg !17
  %dma.buffer.offset1546 = shl i32 %.tr84, 1, !dbg !17
  %1048 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %1049 = or disjoint i64 %address.local928, %1046, !dbg !17
  %.tr88 = trunc i64 %1049 to i32, !dbg !17
  %dma.buffer.offset = shl i32 %.tr88, 1, !dbg !17
  %1050 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !17
  %1051 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1052 = extractvalue [2 x <8 x float>] %139, 0, !dbg !24
  %1053 = shufflevector <8 x float> %1052, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1054 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1051, <4 x float> %1053) #8, !dbg !23
  %1055 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1056 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1055, <4 x float> %1054) #8, !dbg !23
  %1057 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1058 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1057, <4 x float> %1056) #8, !dbg !23
  %1059 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1060 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1059, <4 x float> %1058) #8, !dbg !23
  %1061 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1062 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1061, <4 x float> %1060) #8, !dbg !23
  %1063 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1064 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1063, <4 x float> %1062) #8, !dbg !23
  %1065 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1066 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1065, <4 x float> %1064) #8, !dbg !23
  %1067 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1068 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1067, <4 x float> %1066) #8, !dbg !23
  %1069 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1070 = shufflevector <8 x float> %1052, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1071 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1069, <4 x float> %1070) #8, !dbg !23
  %1072 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1073 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1072, <4 x float> %1071) #8, !dbg !23
  %1074 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1075 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1074, <4 x float> %1073) #8, !dbg !23
  %1076 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1077 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1076, <4 x float> %1075) #8, !dbg !23
  %1078 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1079 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1078, <4 x float> %1077) #8, !dbg !23
  %1080 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1081 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1080, <4 x float> %1079) #8, !dbg !23
  %1082 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1083 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1082, <4 x float> %1081) #8, !dbg !23
  %1084 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1085 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1084, <4 x float> %1083) #8, !dbg !23
  %1086 = extractvalue [2 x <8 x float>] %139, 1, !dbg !24
  %1087 = shufflevector <8 x float> %1086, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1051, <4 x float> %1087) #8, !dbg !23
  %1089 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1055, <4 x float> %1088) #8, !dbg !23
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1057, <4 x float> %1089) #8, !dbg !23
  %1091 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1059, <4 x float> %1090) #8, !dbg !23
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1061, <4 x float> %1091) #8, !dbg !23
  %1093 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1063, <4 x float> %1092) #8, !dbg !23
  %1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1065, <4 x float> %1093) #8, !dbg !23
  %1095 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1067, <4 x float> %1094) #8, !dbg !23
  %1096 = shufflevector <8 x float> %1086, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1097 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1069, <4 x float> %1096) #8, !dbg !23
  %1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1072, <4 x float> %1097) #8, !dbg !23
  %1099 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1074, <4 x float> %1098) #8, !dbg !23
  %1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1076, <4 x float> %1099) #8, !dbg !23
  %1101 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1078, <4 x float> %1100) #8, !dbg !23
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1080, <4 x float> %1101) #8, !dbg !23
  %1103 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1082, <4 x float> %1102) #8, !dbg !23
  %1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1084, <4 x float> %1103) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1105 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !23
  %1106 = bitcast <4 x i32> %1105 to <8 x half>, !dbg !23
  %1107 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !23
  %1108 = bitcast <4 x i32> %1107 to <8 x half>, !dbg !23
  %1109 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !23
  %1110 = bitcast <4 x i32> %1109 to <8 x half>, !dbg !23
  %1111 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !23
  %1112 = bitcast <4 x i32> %1111 to <8 x half>, !dbg !23
  %1113 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !23
  %1114 = bitcast <4 x i32> %1113 to <8 x half>, !dbg !23
  %1115 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !23
  %1116 = bitcast <4 x i32> %1115 to <8 x half>, !dbg !23
  %1117 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !23
  %1118 = bitcast <4 x i32> %1117 to <8 x half>, !dbg !23
  %1119 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !23
  %1120 = bitcast <4 x i32> %1119 to <8 x half>, !dbg !23
  %1121 = shufflevector <8 x half> %1106, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1122 = extractvalue [2 x <8 x float>] %138, 0, !dbg !24
  %1123 = shufflevector <8 x float> %1122, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1124 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1121, <4 x float> %1123) #8, !dbg !23
  %1125 = shufflevector <8 x half> %1108, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1126 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1125, <4 x float> %1124) #8, !dbg !23
  %1127 = shufflevector <8 x half> %1110, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1128 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1127, <4 x float> %1126) #8, !dbg !23
  %1129 = shufflevector <8 x half> %1112, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1130 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1129, <4 x float> %1128) #8, !dbg !23
  %1131 = shufflevector <8 x half> %1114, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1132 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1131, <4 x float> %1130) #8, !dbg !23
  %1133 = shufflevector <8 x half> %1116, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1134 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1133, <4 x float> %1132) #8, !dbg !23
  %1135 = shufflevector <8 x half> %1118, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1136 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1135, <4 x float> %1134) #8, !dbg !23
  %1137 = shufflevector <8 x half> %1120, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1138 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1137, <4 x float> %1136) #8, !dbg !23
  %1139 = shufflevector <8 x half> %1106, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1140 = shufflevector <8 x float> %1122, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1139, <4 x float> %1140) #8, !dbg !23
  %1142 = shufflevector <8 x half> %1108, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1142, <4 x float> %1141) #8, !dbg !23
  %1144 = shufflevector <8 x half> %1110, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1144, <4 x float> %1143) #8, !dbg !23
  %1146 = shufflevector <8 x half> %1112, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1146, <4 x float> %1145) #8, !dbg !23
  %1148 = shufflevector <8 x half> %1114, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1148, <4 x float> %1147) #8, !dbg !23
  %1150 = shufflevector <8 x half> %1116, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1150, <4 x float> %1149) #8, !dbg !23
  %1152 = shufflevector <8 x half> %1118, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1152, <4 x float> %1151) #8, !dbg !23
  %1154 = shufflevector <8 x half> %1120, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1154, <4 x float> %1153) #8, !dbg !23
  %1156 = extractvalue [2 x <8 x float>] %138, 1, !dbg !24
  %1157 = shufflevector <8 x float> %1156, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1121, <4 x float> %1157) #8, !dbg !23
  %1159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1125, <4 x float> %1158) #8, !dbg !23
  %1160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1127, <4 x float> %1159) #8, !dbg !23
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1129, <4 x float> %1160) #8, !dbg !23
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1131, <4 x float> %1161) #8, !dbg !23
  %1163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1133, <4 x float> %1162) #8, !dbg !23
  %1164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1135, <4 x float> %1163) #8, !dbg !23
  %1165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1137, <4 x float> %1164) #8, !dbg !23
  %1166 = shufflevector <8 x float> %1156, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1139, <4 x float> %1166) #8, !dbg !23
  %1168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1142, <4 x float> %1167) #8, !dbg !23
  %1169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1144, <4 x float> %1168) #8, !dbg !23
  %1170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1146, <4 x float> %1169) #8, !dbg !23
  %1171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1148, <4 x float> %1170) #8, !dbg !23
  %1172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1150, <4 x float> %1171) #8, !dbg !23
  %1173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1152, <4 x float> %1172) #8, !dbg !23
  %1174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1154, <4 x float> %1173) #8, !dbg !23
  %1175 = shufflevector <4 x float> %921, <4 x float> %938, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1176 = insertvalue [2 x <8 x float>] poison, <8 x float> %1175, 0, !dbg !11
  %1177 = shufflevector <4 x float> %948, <4 x float> %957, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1178 = insertvalue [2 x <8 x float>] %1176, <8 x float> %1177, 1, !dbg !11
  %1179 = shufflevector <4 x float> %993, <4 x float> %1010, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1180 = insertvalue [2 x <8 x float>] poison, <8 x float> %1179, 0, !dbg !11
  %1181 = shufflevector <4 x float> %1020, <4 x float> %1029, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1182 = insertvalue [2 x <8 x float>] %1180, <8 x float> %1181, 1, !dbg !11
  %1183 = shufflevector <4 x float> %1068, <4 x float> %1085, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1184 = insertvalue [2 x <8 x float>] poison, <8 x float> %1183, 0, !dbg !11
  %1185 = shufflevector <4 x float> %1095, <4 x float> %1104, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1186 = insertvalue [2 x <8 x float>] %1184, <8 x float> %1185, 1, !dbg !11
  %1187 = shufflevector <4 x float> %1138, <4 x float> %1155, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1188 = insertvalue [2 x <8 x float>] poison, <8 x float> %1187, 0, !dbg !11
  %1189 = shufflevector <4 x float> %1165, <4 x float> %1174, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1190 = insertvalue [2 x <8 x float>] %1188, <8 x float> %1189, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1191 = add nuw nsw i64 %143, 1, !dbg !11
  %1192 = icmp slt i64 %1191, %103, !dbg !11
  br i1 %1192, label %137, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %536, %.._crit_edge_crit_edge
  %.pre-phi114 = phi i32 [ %.pre113, %.._crit_edge_crit_edge ], [ %854, %536 ], !dbg !11
  %.pre-phi112 = phi i32 [ %.pre111, %.._crit_edge_crit_edge ], [ %846, %536 ], !dbg !11
  %.pre-phi110 = phi i32 [ %.pre109, %.._crit_edge_crit_edge ], [ %845, %536 ], !dbg !11
  %.pre-phi106 = phi i32 [ %.pre105, %.._crit_edge_crit_edge ], [ %843, %536 ], !dbg !11
  %.lcssa94 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1178, %536 ], !dbg !11
  %.lcssa93 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %879, %536 ], !dbg !11
  %.lcssa92 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1182, %536 ], !dbg !11
  %.lcssa91 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1186, %536 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1190, %536 ], !dbg !11
  %1193 = and i32 %29, 16, !dbg !11
  %.not = icmp eq i32 %1193, 0, !dbg !11
  %1194 = and i32 %29, 32, !dbg !11
  %.not36 = icmp eq i32 %1194, 0, !dbg !11
  %1195 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !11
  %1196 = extractelement <8 x float> %1195, i64 0, !dbg !11
  %1197 = extractelement <8 x float> %1195, i64 1, !dbg !11
  %1198 = extractelement <8 x float> %1195, i64 2, !dbg !11
  %1199 = extractelement <8 x float> %1195, i64 3, !dbg !11
  %1200 = select i1 %.not, float %1197, float %1196, !dbg !11
  %1201 = icmp samesign ult i32 %.pre-phi110, %.pre-phi112, !dbg !11
  %1202 = select i1 %1201, i32 %.pre-phi110, i32 %.pre-phi106, !dbg !11
  %1203 = shl nuw nsw i32 %1202, 2, !dbg !11
  %1204 = bitcast float %1200 to i32, !dbg !11
  %1205 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1204), !dbg !11
  %1206 = bitcast i32 %1205 to float, !dbg !11
  %1207 = select i1 %.not, float %1196, float %1206, !dbg !11
  %1208 = select i1 %.not, float %1206, float %1197, !dbg !11
  %1209 = select i1 %.not, float %1199, float %1198, !dbg !11
  %1210 = bitcast float %1209 to i32, !dbg !11
  %1211 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1210), !dbg !11
  %1212 = bitcast i32 %1211 to float, !dbg !11
  %1213 = select i1 %.not, float %1198, float %1212, !dbg !11
  %1214 = select i1 %.not, float %1212, float %1199, !dbg !11
  %1215 = select i1 %.not36, float %1213, float %1207, !dbg !11
  %1216 = icmp samesign ult i32 %.pre-phi114, %.pre-phi112, !dbg !11
  %1217 = select i1 %1216, i32 %.pre-phi114, i32 %.pre-phi106, !dbg !11
  %1218 = shl nuw nsw i32 %1217, 2, !dbg !11
  %1219 = bitcast float %1215 to i32, !dbg !11
  %1220 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1219), !dbg !11
  %1221 = bitcast i32 %1220 to float, !dbg !11
  %1222 = select i1 %.not36, float %1214, float %1208, !dbg !11
  %1223 = bitcast float %1222 to i32, !dbg !11
  %1224 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1223), !dbg !11
  %1225 = bitcast i32 %1224 to float, !dbg !11
  %1226 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1227 = shufflevector <4 x i1> %1226, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1228 = insertelement <4 x float> poison, float %1207, i64 0, !dbg !11
  %1229 = insertelement <4 x float> %1228, float %1208, i64 1, !dbg !11
  %1230 = insertelement <4 x float> %1229, float %1221, i64 2, !dbg !11
  %1231 = insertelement <4 x float> %1230, float %1225, i64 3, !dbg !11
  %1232 = insertelement <4 x float> poison, float %1221, i64 0, !dbg !11
  %1233 = insertelement <4 x float> %1232, float %1225, i64 1, !dbg !11
  %1234 = insertelement <4 x float> %1233, float %1213, i64 2, !dbg !11
  %1235 = insertelement <4 x float> %1234, float %1214, i64 3, !dbg !11
  %1236 = select <4 x i1> %1227, <4 x float> %1231, <4 x float> %1235, !dbg !11
  %1237 = extractvalue [2 x <8 x float>] %.lcssa91, 0, !dbg !11
  %1238 = extractelement <8 x float> %1237, i64 0, !dbg !11
  %1239 = extractelement <8 x float> %1237, i64 1, !dbg !11
  %1240 = extractelement <8 x float> %1237, i64 2, !dbg !11
  %1241 = extractelement <8 x float> %1237, i64 3, !dbg !11
  %1242 = select i1 %.not, float %1239, float %1238, !dbg !11
  %1243 = bitcast float %1242 to i32, !dbg !11
  %1244 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1243), !dbg !11
  %1245 = bitcast i32 %1244 to float, !dbg !11
  %1246 = select i1 %.not, float %1238, float %1245, !dbg !11
  %1247 = select i1 %.not, float %1245, float %1239, !dbg !11
  %1248 = select i1 %.not, float %1241, float %1240, !dbg !11
  %1249 = bitcast float %1248 to i32, !dbg !11
  %1250 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1249), !dbg !11
  %1251 = bitcast i32 %1250 to float, !dbg !11
  %1252 = select i1 %.not, float %1240, float %1251, !dbg !11
  %1253 = select i1 %.not, float %1251, float %1241, !dbg !11
  %1254 = select i1 %.not36, float %1252, float %1246, !dbg !11
  %1255 = bitcast float %1254 to i32, !dbg !11
  %1256 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1255), !dbg !11
  %1257 = bitcast i32 %1256 to float, !dbg !11
  %1258 = select i1 %.not36, float %1253, float %1247, !dbg !11
  %1259 = bitcast float %1258 to i32, !dbg !11
  %1260 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1259), !dbg !11
  %1261 = bitcast i32 %1260 to float, !dbg !11
  %1262 = insertelement <4 x float> poison, float %1246, i64 0, !dbg !11
  %1263 = insertelement <4 x float> %1262, float %1247, i64 1, !dbg !11
  %1264 = insertelement <4 x float> %1263, float %1257, i64 2, !dbg !11
  %1265 = insertelement <4 x float> %1264, float %1261, i64 3, !dbg !11
  %1266 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !11
  %1267 = insertelement <4 x float> %1266, float %1261, i64 1, !dbg !11
  %1268 = insertelement <4 x float> %1267, float %1252, i64 2, !dbg !11
  %1269 = insertelement <4 x float> %1268, float %1253, i64 3, !dbg !11
  %1270 = select <4 x i1> %1227, <4 x float> %1265, <4 x float> %1269, !dbg !11
  %1271 = extractvalue [2 x <8 x float>] %.lcssa92, 0, !dbg !11
  %1272 = extractelement <8 x float> %1271, i64 0, !dbg !11
  %1273 = extractelement <8 x float> %1271, i64 1, !dbg !11
  %1274 = extractelement <8 x float> %1271, i64 2, !dbg !11
  %1275 = extractelement <8 x float> %1271, i64 3, !dbg !11
  %1276 = select i1 %.not, float %1273, float %1272, !dbg !11
  %1277 = bitcast float %1276 to i32, !dbg !11
  %1278 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1277), !dbg !11
  %1279 = bitcast i32 %1278 to float, !dbg !11
  %1280 = select i1 %.not, float %1272, float %1279, !dbg !11
  %1281 = select i1 %.not, float %1279, float %1273, !dbg !11
  %1282 = select i1 %.not, float %1275, float %1274, !dbg !11
  %1283 = bitcast float %1282 to i32, !dbg !11
  %1284 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1283), !dbg !11
  %1285 = bitcast i32 %1284 to float, !dbg !11
  %1286 = select i1 %.not, float %1274, float %1285, !dbg !11
  %1287 = select i1 %.not, float %1285, float %1275, !dbg !11
  %1288 = select i1 %.not36, float %1286, float %1280, !dbg !11
  %1289 = bitcast float %1288 to i32, !dbg !11
  %1290 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1289), !dbg !11
  %1291 = bitcast i32 %1290 to float, !dbg !11
  %1292 = select i1 %.not36, float %1287, float %1281, !dbg !11
  %1293 = bitcast float %1292 to i32, !dbg !11
  %1294 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1293), !dbg !11
  %1295 = bitcast i32 %1294 to float, !dbg !11
  %1296 = insertelement <4 x float> poison, float %1280, i64 0, !dbg !11
  %1297 = insertelement <4 x float> %1296, float %1281, i64 1, !dbg !11
  %1298 = insertelement <4 x float> %1297, float %1291, i64 2, !dbg !11
  %1299 = insertelement <4 x float> %1298, float %1295, i64 3, !dbg !11
  %1300 = insertelement <4 x float> poison, float %1291, i64 0, !dbg !11
  %1301 = insertelement <4 x float> %1300, float %1295, i64 1, !dbg !11
  %1302 = insertelement <4 x float> %1301, float %1286, i64 2, !dbg !11
  %1303 = insertelement <4 x float> %1302, float %1287, i64 3, !dbg !11
  %1304 = select <4 x i1> %1227, <4 x float> %1299, <4 x float> %1303, !dbg !11
  %1305 = extractvalue [2 x <8 x float>] %.lcssa94, 0, !dbg !11
  %1306 = extractelement <8 x float> %1305, i64 0, !dbg !11
  %1307 = extractelement <8 x float> %1305, i64 1, !dbg !11
  %1308 = extractelement <8 x float> %1305, i64 2, !dbg !11
  %1309 = extractelement <8 x float> %1305, i64 3, !dbg !11
  %1310 = select i1 %.not, float %1307, float %1306, !dbg !11
  %1311 = bitcast float %1310 to i32, !dbg !11
  %1312 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1311), !dbg !11
  %1313 = bitcast i32 %1312 to float, !dbg !11
  %1314 = select i1 %.not, float %1306, float %1313, !dbg !11
  %1315 = select i1 %.not, float %1313, float %1307, !dbg !11
  %1316 = select i1 %.not, float %1309, float %1308, !dbg !11
  %1317 = bitcast float %1316 to i32, !dbg !11
  %1318 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1317), !dbg !11
  %1319 = bitcast i32 %1318 to float, !dbg !11
  %1320 = select i1 %.not, float %1308, float %1319, !dbg !11
  %1321 = select i1 %.not, float %1319, float %1309, !dbg !11
  %1322 = select i1 %.not36, float %1320, float %1314, !dbg !11
  %1323 = bitcast float %1322 to i32, !dbg !11
  %1324 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1323), !dbg !11
  %1325 = select i1 %.not36, float %1321, float %1315, !dbg !11
  %1326 = bitcast float %1325 to i32, !dbg !11
  %1327 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1326), !dbg !11
  %1328 = extractvalue [2 x <1 x float>] %.lcssa93, 0, !dbg !25
  %1329 = shufflevector <1 x float> %1328, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %1330 = extractelement <8 x float> %1195, i64 4, !dbg !11
  %1331 = extractelement <8 x float> %1195, i64 5, !dbg !11
  %1332 = extractelement <8 x float> %1195, i64 6, !dbg !11
  %1333 = extractelement <8 x float> %1195, i64 7, !dbg !11
  %1334 = select i1 %.not, float %1331, float %1330, !dbg !11
  %1335 = bitcast float %1334 to i32, !dbg !11
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1335), !dbg !11
  %1337 = bitcast i32 %1336 to float, !dbg !11
  %1338 = select i1 %.not, float %1330, float %1337, !dbg !11
  %1339 = select i1 %.not, float %1337, float %1331, !dbg !11
  %1340 = select i1 %.not, float %1333, float %1332, !dbg !11
  %1341 = bitcast float %1340 to i32, !dbg !11
  %1342 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1341), !dbg !11
  %1343 = bitcast i32 %1342 to float, !dbg !11
  %1344 = select i1 %.not, float %1332, float %1343, !dbg !11
  %1345 = select i1 %.not, float %1343, float %1333, !dbg !11
  %1346 = select i1 %.not36, float %1344, float %1338, !dbg !11
  %1347 = bitcast float %1346 to i32, !dbg !11
  %1348 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1347), !dbg !11
  %1349 = bitcast i32 %1348 to float, !dbg !11
  %1350 = select i1 %.not36, float %1345, float %1339, !dbg !11
  %1351 = bitcast float %1350 to i32, !dbg !11
  %1352 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1351), !dbg !11
  %1353 = bitcast i32 %1352 to float, !dbg !11
  %1354 = insertelement <4 x float> poison, float %1338, i64 0, !dbg !11
  %1355 = insertelement <4 x float> %1354, float %1339, i64 1, !dbg !11
  %1356 = insertelement <4 x float> %1355, float %1349, i64 2, !dbg !11
  %1357 = insertelement <4 x float> %1356, float %1353, i64 3, !dbg !11
  %1358 = insertelement <4 x float> poison, float %1349, i64 0, !dbg !11
  %1359 = insertelement <4 x float> %1358, float %1353, i64 1, !dbg !11
  %1360 = insertelement <4 x float> %1359, float %1344, i64 2, !dbg !11
  %1361 = insertelement <4 x float> %1360, float %1345, i64 3, !dbg !11
  %1362 = select <4 x i1> %1227, <4 x float> %1357, <4 x float> %1361, !dbg !11
  %1363 = extractelement <8 x float> %1237, i64 4, !dbg !11
  %1364 = extractelement <8 x float> %1237, i64 5, !dbg !11
  %1365 = extractelement <8 x float> %1237, i64 6, !dbg !11
  %1366 = extractelement <8 x float> %1237, i64 7, !dbg !11
  %1367 = select i1 %.not, float %1364, float %1363, !dbg !11
  %1368 = bitcast float %1367 to i32, !dbg !11
  %1369 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1368), !dbg !11
  %1370 = bitcast i32 %1369 to float, !dbg !11
  %1371 = select i1 %.not, float %1363, float %1370, !dbg !11
  %1372 = select i1 %.not, float %1370, float %1364, !dbg !11
  %1373 = select i1 %.not, float %1366, float %1365, !dbg !11
  %1374 = bitcast float %1373 to i32, !dbg !11
  %1375 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1374), !dbg !11
  %1376 = bitcast i32 %1375 to float, !dbg !11
  %1377 = select i1 %.not, float %1365, float %1376, !dbg !11
  %1378 = select i1 %.not, float %1376, float %1366, !dbg !11
  %1379 = select i1 %.not36, float %1377, float %1371, !dbg !11
  %1380 = bitcast float %1379 to i32, !dbg !11
  %1381 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1380), !dbg !11
  %1382 = bitcast i32 %1381 to float, !dbg !11
  %1383 = select i1 %.not36, float %1378, float %1372, !dbg !11
  %1384 = bitcast float %1383 to i32, !dbg !11
  %1385 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1384), !dbg !11
  %1386 = bitcast i32 %1385 to float, !dbg !11
  %1387 = insertelement <4 x float> poison, float %1371, i64 0, !dbg !11
  %1388 = insertelement <4 x float> %1387, float %1372, i64 1, !dbg !11
  %1389 = insertelement <4 x float> %1388, float %1382, i64 2, !dbg !11
  %1390 = insertelement <4 x float> %1389, float %1386, i64 3, !dbg !11
  %1391 = insertelement <4 x float> poison, float %1382, i64 0, !dbg !11
  %1392 = insertelement <4 x float> %1391, float %1386, i64 1, !dbg !11
  %1393 = insertelement <4 x float> %1392, float %1377, i64 2, !dbg !11
  %1394 = insertelement <4 x float> %1393, float %1378, i64 3, !dbg !11
  %1395 = select <4 x i1> %1227, <4 x float> %1390, <4 x float> %1394, !dbg !11
  %1396 = extractelement <8 x float> %1271, i64 4, !dbg !11
  %1397 = extractelement <8 x float> %1271, i64 5, !dbg !11
  %1398 = extractelement <8 x float> %1271, i64 6, !dbg !11
  %1399 = extractelement <8 x float> %1271, i64 7, !dbg !11
  %1400 = select i1 %.not, float %1397, float %1396, !dbg !11
  %1401 = bitcast float %1400 to i32, !dbg !11
  %1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1401), !dbg !11
  %1403 = bitcast i32 %1402 to float, !dbg !11
  %1404 = select i1 %.not, float %1396, float %1403, !dbg !11
  %1405 = select i1 %.not, float %1403, float %1397, !dbg !11
  %1406 = select i1 %.not, float %1399, float %1398, !dbg !11
  %1407 = bitcast float %1406 to i32, !dbg !11
  %1408 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1407), !dbg !11
  %1409 = bitcast i32 %1408 to float, !dbg !11
  %1410 = select i1 %.not, float %1398, float %1409, !dbg !11
  %1411 = select i1 %.not, float %1409, float %1399, !dbg !11
  %1412 = select i1 %.not36, float %1410, float %1404, !dbg !11
  %1413 = bitcast float %1412 to i32, !dbg !11
  %1414 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1413), !dbg !11
  %1415 = bitcast i32 %1414 to float, !dbg !11
  %1416 = select i1 %.not36, float %1411, float %1405, !dbg !11
  %1417 = bitcast float %1416 to i32, !dbg !11
  %1418 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1417), !dbg !11
  %1419 = bitcast i32 %1418 to float, !dbg !11
  %1420 = insertelement <4 x float> poison, float %1404, i64 0, !dbg !11
  %1421 = insertelement <4 x float> %1420, float %1405, i64 1, !dbg !11
  %1422 = insertelement <4 x float> %1421, float %1415, i64 2, !dbg !11
  %1423 = insertelement <4 x float> %1422, float %1419, i64 3, !dbg !11
  %1424 = insertelement <4 x float> poison, float %1415, i64 0, !dbg !11
  %1425 = insertelement <4 x float> %1424, float %1419, i64 1, !dbg !11
  %1426 = insertelement <4 x float> %1425, float %1410, i64 2, !dbg !11
  %1427 = insertelement <4 x float> %1426, float %1411, i64 3, !dbg !11
  %1428 = select <4 x i1> %1227, <4 x float> %1423, <4 x float> %1427, !dbg !11
  %1429 = extractelement <8 x float> %1305, i64 4, !dbg !11
  %1430 = extractelement <8 x float> %1305, i64 5, !dbg !11
  %1431 = extractelement <8 x float> %1305, i64 6, !dbg !11
  %1432 = extractelement <8 x float> %1305, i64 7, !dbg !11
  %1433 = select i1 %.not, float %1430, float %1429, !dbg !11
  %1434 = bitcast float %1433 to i32, !dbg !11
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1434), !dbg !11
  %1436 = bitcast i32 %1435 to float, !dbg !11
  %1437 = select i1 %.not, float %1429, float %1436, !dbg !11
  %1438 = select i1 %.not, float %1436, float %1430, !dbg !11
  %1439 = select i1 %.not, float %1432, float %1431, !dbg !11
  %1440 = bitcast float %1439 to i32, !dbg !11
  %1441 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1440), !dbg !11
  %1442 = bitcast i32 %1441 to float, !dbg !11
  %1443 = select i1 %.not, float %1431, float %1442, !dbg !11
  %1444 = select i1 %.not, float %1442, float %1432, !dbg !11
  %1445 = select i1 %.not36, float %1443, float %1437, !dbg !11
  %1446 = bitcast float %1445 to i32, !dbg !11
  %1447 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1446), !dbg !11
  %1448 = select i1 %.not36, float %1444, float %1438, !dbg !11
  %1449 = bitcast float %1448 to i32, !dbg !11
  %1450 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1449), !dbg !11
  %1451 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !11
  %1452 = extractelement <8 x float> %1451, i64 0, !dbg !11
  %1453 = extractelement <8 x float> %1451, i64 1, !dbg !11
  %1454 = extractelement <8 x float> %1451, i64 2, !dbg !11
  %1455 = extractelement <8 x float> %1451, i64 3, !dbg !11
  %1456 = select i1 %.not, float %1453, float %1452, !dbg !11
  %1457 = bitcast float %1456 to i32, !dbg !11
  %1458 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1457), !dbg !11
  %1459 = bitcast i32 %1458 to float, !dbg !11
  %1460 = select i1 %.not, float %1452, float %1459, !dbg !11
  %1461 = select i1 %.not, float %1459, float %1453, !dbg !11
  %1462 = select i1 %.not, float %1455, float %1454, !dbg !11
  %1463 = bitcast float %1462 to i32, !dbg !11
  %1464 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1463), !dbg !11
  %1465 = bitcast i32 %1464 to float, !dbg !11
  %1466 = select i1 %.not, float %1454, float %1465, !dbg !11
  %1467 = select i1 %.not, float %1465, float %1455, !dbg !11
  %1468 = select i1 %.not36, float %1466, float %1460, !dbg !11
  %1469 = bitcast float %1468 to i32, !dbg !11
  %1470 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1469), !dbg !11
  %1471 = bitcast i32 %1470 to float, !dbg !11
  %1472 = select i1 %.not36, float %1467, float %1461, !dbg !11
  %1473 = bitcast float %1472 to i32, !dbg !11
  %1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1473), !dbg !11
  %1475 = bitcast i32 %1474 to float, !dbg !11
  %1476 = insertelement <4 x float> poison, float %1460, i64 0, !dbg !11
  %1477 = insertelement <4 x float> %1476, float %1461, i64 1, !dbg !11
  %1478 = insertelement <4 x float> %1477, float %1471, i64 2, !dbg !11
  %1479 = insertelement <4 x float> %1478, float %1475, i64 3, !dbg !11
  %1480 = insertelement <4 x float> poison, float %1471, i64 0, !dbg !11
  %1481 = insertelement <4 x float> %1480, float %1475, i64 1, !dbg !11
  %1482 = insertelement <4 x float> %1481, float %1466, i64 2, !dbg !11
  %1483 = insertelement <4 x float> %1482, float %1467, i64 3, !dbg !11
  %1484 = select <4 x i1> %1227, <4 x float> %1479, <4 x float> %1483, !dbg !11
  %1485 = extractvalue [2 x <8 x float>] %.lcssa91, 1, !dbg !11
  %1486 = extractelement <8 x float> %1485, i64 0, !dbg !11
  %1487 = extractelement <8 x float> %1485, i64 1, !dbg !11
  %1488 = extractelement <8 x float> %1485, i64 2, !dbg !11
  %1489 = extractelement <8 x float> %1485, i64 3, !dbg !11
  %1490 = select i1 %.not, float %1487, float %1486, !dbg !11
  %1491 = bitcast float %1490 to i32, !dbg !11
  %1492 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1491), !dbg !11
  %1493 = bitcast i32 %1492 to float, !dbg !11
  %1494 = select i1 %.not, float %1486, float %1493, !dbg !11
  %1495 = select i1 %.not, float %1493, float %1487, !dbg !11
  %1496 = select i1 %.not, float %1489, float %1488, !dbg !11
  %1497 = bitcast float %1496 to i32, !dbg !11
  %1498 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1497), !dbg !11
  %1499 = bitcast i32 %1498 to float, !dbg !11
  %1500 = select i1 %.not, float %1488, float %1499, !dbg !11
  %1501 = select i1 %.not, float %1499, float %1489, !dbg !11
  %1502 = select i1 %.not36, float %1500, float %1494, !dbg !11
  %1503 = bitcast float %1502 to i32, !dbg !11
  %1504 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1503), !dbg !11
  %1505 = bitcast i32 %1504 to float, !dbg !11
  %1506 = select i1 %.not36, float %1501, float %1495, !dbg !11
  %1507 = bitcast float %1506 to i32, !dbg !11
  %1508 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1507), !dbg !11
  %1509 = bitcast i32 %1508 to float, !dbg !11
  %1510 = insertelement <4 x float> poison, float %1494, i64 0, !dbg !11
  %1511 = insertelement <4 x float> %1510, float %1495, i64 1, !dbg !11
  %1512 = insertelement <4 x float> %1511, float %1505, i64 2, !dbg !11
  %1513 = insertelement <4 x float> %1512, float %1509, i64 3, !dbg !11
  %1514 = insertelement <4 x float> poison, float %1505, i64 0, !dbg !11
  %1515 = insertelement <4 x float> %1514, float %1509, i64 1, !dbg !11
  %1516 = insertelement <4 x float> %1515, float %1500, i64 2, !dbg !11
  %1517 = insertelement <4 x float> %1516, float %1501, i64 3, !dbg !11
  %1518 = select <4 x i1> %1227, <4 x float> %1513, <4 x float> %1517, !dbg !11
  %1519 = extractvalue [2 x <8 x float>] %.lcssa92, 1, !dbg !11
  %1520 = extractelement <8 x float> %1519, i64 0, !dbg !11
  %1521 = extractelement <8 x float> %1519, i64 1, !dbg !11
  %1522 = extractelement <8 x float> %1519, i64 2, !dbg !11
  %1523 = extractelement <8 x float> %1519, i64 3, !dbg !11
  %1524 = select i1 %.not, float %1521, float %1520, !dbg !11
  %1525 = bitcast float %1524 to i32, !dbg !11
  %1526 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1525), !dbg !11
  %1527 = bitcast i32 %1526 to float, !dbg !11
  %1528 = select i1 %.not, float %1520, float %1527, !dbg !11
  %1529 = select i1 %.not, float %1527, float %1521, !dbg !11
  %1530 = select i1 %.not, float %1523, float %1522, !dbg !11
  %1531 = bitcast float %1530 to i32, !dbg !11
  %1532 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1531), !dbg !11
  %1533 = bitcast i32 %1532 to float, !dbg !11
  %1534 = select i1 %.not, float %1522, float %1533, !dbg !11
  %1535 = select i1 %.not, float %1533, float %1523, !dbg !11
  %1536 = select i1 %.not36, float %1534, float %1528, !dbg !11
  %1537 = bitcast float %1536 to i32, !dbg !11
  %1538 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1537), !dbg !11
  %1539 = bitcast i32 %1538 to float, !dbg !11
  %1540 = select i1 %.not36, float %1535, float %1529, !dbg !11
  %1541 = bitcast float %1540 to i32, !dbg !11
  %1542 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1541), !dbg !11
  %1543 = bitcast i32 %1542 to float, !dbg !11
  %1544 = insertelement <4 x float> poison, float %1528, i64 0, !dbg !11
  %1545 = insertelement <4 x float> %1544, float %1529, i64 1, !dbg !11
  %1546 = insertelement <4 x float> %1545, float %1539, i64 2, !dbg !11
  %1547 = insertelement <4 x float> %1546, float %1543, i64 3, !dbg !11
  %1548 = insertelement <4 x float> poison, float %1539, i64 0, !dbg !11
  %1549 = insertelement <4 x float> %1548, float %1543, i64 1, !dbg !11
  %1550 = insertelement <4 x float> %1549, float %1534, i64 2, !dbg !11
  %1551 = insertelement <4 x float> %1550, float %1535, i64 3, !dbg !11
  %1552 = select <4 x i1> %1227, <4 x float> %1547, <4 x float> %1551, !dbg !11
  %1553 = extractvalue [2 x <8 x float>] %.lcssa94, 1, !dbg !11
  %1554 = extractelement <8 x float> %1553, i64 0, !dbg !11
  %1555 = extractelement <8 x float> %1553, i64 1, !dbg !11
  %1556 = extractelement <8 x float> %1553, i64 2, !dbg !11
  %1557 = extractelement <8 x float> %1553, i64 3, !dbg !11
  %1558 = select i1 %.not, float %1555, float %1554, !dbg !11
  %1559 = bitcast float %1558 to i32, !dbg !11
  %1560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1559), !dbg !11
  %1561 = bitcast i32 %1560 to float, !dbg !11
  %1562 = select i1 %.not, float %1554, float %1561, !dbg !11
  %1563 = select i1 %.not, float %1561, float %1555, !dbg !11
  %1564 = select i1 %.not, float %1557, float %1556, !dbg !11
  %1565 = bitcast float %1564 to i32, !dbg !11
  %1566 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1565), !dbg !11
  %1567 = bitcast i32 %1566 to float, !dbg !11
  %1568 = select i1 %.not, float %1556, float %1567, !dbg !11
  %1569 = select i1 %.not, float %1567, float %1557, !dbg !11
  %1570 = select i1 %.not36, float %1568, float %1562, !dbg !11
  %1571 = bitcast float %1570 to i32, !dbg !11
  %1572 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1571), !dbg !11
  %1573 = select i1 %.not36, float %1569, float %1563, !dbg !11
  %1574 = bitcast float %1573 to i32, !dbg !11
  %1575 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1574), !dbg !11
  %1576 = extractvalue [2 x <1 x float>] %.lcssa93, 1, !dbg !25
  %1577 = shufflevector <1 x float> %1576, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %1578 = extractelement <8 x float> %1451, i64 4, !dbg !11
  %1579 = extractelement <8 x float> %1451, i64 5, !dbg !11
  %1580 = extractelement <8 x float> %1451, i64 6, !dbg !11
  %1581 = extractelement <8 x float> %1451, i64 7, !dbg !11
  %1582 = select i1 %.not, float %1579, float %1578, !dbg !11
  %1583 = bitcast float %1582 to i32, !dbg !11
  %1584 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1583), !dbg !11
  %1585 = bitcast i32 %1584 to float, !dbg !11
  %1586 = select i1 %.not, float %1578, float %1585, !dbg !11
  %1587 = select i1 %.not, float %1585, float %1579, !dbg !11
  %1588 = select i1 %.not, float %1581, float %1580, !dbg !11
  %1589 = bitcast float %1588 to i32, !dbg !11
  %1590 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1589), !dbg !11
  %1591 = bitcast i32 %1590 to float, !dbg !11
  %1592 = select i1 %.not, float %1580, float %1591, !dbg !11
  %1593 = select i1 %.not, float %1591, float %1581, !dbg !11
  %1594 = select i1 %.not36, float %1592, float %1586, !dbg !11
  %1595 = bitcast float %1594 to i32, !dbg !11
  %1596 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1595), !dbg !11
  %1597 = bitcast i32 %1596 to float, !dbg !11
  %1598 = select i1 %.not36, float %1593, float %1587, !dbg !11
  %1599 = bitcast float %1598 to i32, !dbg !11
  %1600 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1599), !dbg !11
  %1601 = bitcast i32 %1600 to float, !dbg !11
  %1602 = insertelement <4 x float> poison, float %1586, i64 0, !dbg !11
  %1603 = insertelement <4 x float> %1602, float %1587, i64 1, !dbg !11
  %1604 = insertelement <4 x float> %1603, float %1597, i64 2, !dbg !11
  %1605 = insertelement <4 x float> %1604, float %1601, i64 3, !dbg !11
  %1606 = insertelement <4 x float> poison, float %1597, i64 0, !dbg !11
  %1607 = insertelement <4 x float> %1606, float %1601, i64 1, !dbg !11
  %1608 = insertelement <4 x float> %1607, float %1592, i64 2, !dbg !11
  %1609 = insertelement <4 x float> %1608, float %1593, i64 3, !dbg !11
  %1610 = select <4 x i1> %1227, <4 x float> %1605, <4 x float> %1609, !dbg !11
  %1611 = extractelement <8 x float> %1485, i64 4, !dbg !11
  %1612 = extractelement <8 x float> %1485, i64 5, !dbg !11
  %1613 = extractelement <8 x float> %1485, i64 6, !dbg !11
  %1614 = extractelement <8 x float> %1485, i64 7, !dbg !11
  %1615 = select i1 %.not, float %1612, float %1611, !dbg !11
  %1616 = bitcast float %1615 to i32, !dbg !11
  %1617 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1616), !dbg !11
  %1618 = bitcast i32 %1617 to float, !dbg !11
  %1619 = select i1 %.not, float %1611, float %1618, !dbg !11
  %1620 = select i1 %.not, float %1618, float %1612, !dbg !11
  %1621 = select i1 %.not, float %1614, float %1613, !dbg !11
  %1622 = bitcast float %1621 to i32, !dbg !11
  %1623 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1622), !dbg !11
  %1624 = bitcast i32 %1623 to float, !dbg !11
  %1625 = select i1 %.not, float %1613, float %1624, !dbg !11
  %1626 = select i1 %.not, float %1624, float %1614, !dbg !11
  %1627 = select i1 %.not36, float %1625, float %1619, !dbg !11
  %1628 = bitcast float %1627 to i32, !dbg !11
  %1629 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1628), !dbg !11
  %1630 = bitcast i32 %1629 to float, !dbg !11
  %1631 = select i1 %.not36, float %1626, float %1620, !dbg !11
  %1632 = bitcast float %1631 to i32, !dbg !11
  %1633 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1632), !dbg !11
  %1634 = bitcast i32 %1633 to float, !dbg !11
  %1635 = insertelement <4 x float> poison, float %1619, i64 0, !dbg !11
  %1636 = insertelement <4 x float> %1635, float %1620, i64 1, !dbg !11
  %1637 = insertelement <4 x float> %1636, float %1630, i64 2, !dbg !11
  %1638 = insertelement <4 x float> %1637, float %1634, i64 3, !dbg !11
  %1639 = insertelement <4 x float> poison, float %1630, i64 0, !dbg !11
  %1640 = insertelement <4 x float> %1639, float %1634, i64 1, !dbg !11
  %1641 = insertelement <4 x float> %1640, float %1625, i64 2, !dbg !11
  %1642 = insertelement <4 x float> %1641, float %1626, i64 3, !dbg !11
  %1643 = select <4 x i1> %1227, <4 x float> %1638, <4 x float> %1642, !dbg !11
  %1644 = extractelement <8 x float> %1519, i64 4, !dbg !11
  %1645 = extractelement <8 x float> %1519, i64 5, !dbg !11
  %1646 = extractelement <8 x float> %1519, i64 6, !dbg !11
  %1647 = extractelement <8 x float> %1519, i64 7, !dbg !11
  %1648 = select i1 %.not, float %1645, float %1644, !dbg !11
  %1649 = bitcast float %1648 to i32, !dbg !11
  %1650 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1649), !dbg !11
  %1651 = bitcast i32 %1650 to float, !dbg !11
  %1652 = select i1 %.not, float %1644, float %1651, !dbg !11
  %1653 = select i1 %.not, float %1651, float %1645, !dbg !11
  %1654 = select i1 %.not, float %1647, float %1646, !dbg !11
  %1655 = bitcast float %1654 to i32, !dbg !11
  %1656 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1655), !dbg !11
  %1657 = bitcast i32 %1656 to float, !dbg !11
  %1658 = select i1 %.not, float %1646, float %1657, !dbg !11
  %1659 = select i1 %.not, float %1657, float %1647, !dbg !11
  %1660 = select i1 %.not36, float %1658, float %1652, !dbg !11
  %1661 = bitcast float %1660 to i32, !dbg !11
  %1662 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1661), !dbg !11
  %1663 = bitcast i32 %1662 to float, !dbg !11
  %1664 = select i1 %.not36, float %1659, float %1653, !dbg !11
  %1665 = bitcast float %1664 to i32, !dbg !11
  %1666 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1665), !dbg !11
  %1667 = bitcast i32 %1666 to float, !dbg !11
  %1668 = insertelement <4 x float> poison, float %1652, i64 0, !dbg !11
  %1669 = insertelement <4 x float> %1668, float %1653, i64 1, !dbg !11
  %1670 = insertelement <4 x float> %1669, float %1663, i64 2, !dbg !11
  %1671 = insertelement <4 x float> %1670, float %1667, i64 3, !dbg !11
  %1672 = insertelement <4 x float> poison, float %1663, i64 0, !dbg !11
  %1673 = insertelement <4 x float> %1672, float %1667, i64 1, !dbg !11
  %1674 = insertelement <4 x float> %1673, float %1658, i64 2, !dbg !11
  %1675 = insertelement <4 x float> %1674, float %1659, i64 3, !dbg !11
  %1676 = select <4 x i1> %1227, <4 x float> %1671, <4 x float> %1675, !dbg !11
  %1677 = extractelement <8 x float> %1553, i64 4, !dbg !11
  %1678 = extractelement <8 x float> %1553, i64 5, !dbg !11
  %1679 = extractelement <8 x float> %1553, i64 6, !dbg !11
  %1680 = extractelement <8 x float> %1553, i64 7, !dbg !11
  %1681 = select i1 %.not, float %1678, float %1677, !dbg !11
  %1682 = bitcast float %1681 to i32, !dbg !11
  %1683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1682), !dbg !11
  %1684 = bitcast i32 %1683 to float, !dbg !11
  %1685 = select i1 %.not, float %1677, float %1684, !dbg !11
  %1686 = select i1 %.not, float %1684, float %1678, !dbg !11
  %1687 = select i1 %.not, float %1680, float %1679, !dbg !11
  %1688 = bitcast float %1687 to i32, !dbg !11
  %1689 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1688), !dbg !11
  %1690 = bitcast i32 %1689 to float, !dbg !11
  %1691 = select i1 %.not, float %1679, float %1690, !dbg !11
  %1692 = select i1 %.not, float %1690, float %1680, !dbg !11
  %1693 = select i1 %.not36, float %1691, float %1685, !dbg !11
  %1694 = bitcast float %1693 to i32, !dbg !11
  %1695 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1694), !dbg !11
  %1696 = select i1 %.not36, float %1692, float %1686, !dbg !11
  %1697 = bitcast float %1696 to i32, !dbg !11
  %1698 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1218, i32 %1697), !dbg !11
  %1699 = bitcast i32 %1695 to float, !dbg !11
  %1700 = bitcast i32 %1698 to float, !dbg !11
  %1701 = insertelement <4 x float> poison, float %1685, i64 0, !dbg !11
  %1702 = insertelement <4 x float> %1701, float %1686, i64 1, !dbg !11
  %1703 = insertelement <4 x float> %1702, float %1699, i64 2, !dbg !11
  %1704 = insertelement <4 x float> %1703, float %1700, i64 3, !dbg !11
  %1705 = insertelement <4 x float> poison, float %1699, i64 0, !dbg !11
  %1706 = insertelement <4 x float> %1705, float %1700, i64 1, !dbg !11
  %1707 = insertelement <4 x float> %1706, float %1691, i64 2, !dbg !11
  %1708 = insertelement <4 x float> %1707, float %1692, i64 3, !dbg !11
  %1709 = select <4 x i1> %1227, <4 x float> %1704, <4 x float> %1708, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1577, !dbg !25
  %normalized = fmul <4 x float> %row.reciprocal, %1709, !dbg !25
  %1710 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !26
  %1711 = extractelement <4 x half> %1710, i64 3, !dbg !26
  %1712 = extractelement <4 x half> %1710, i64 2, !dbg !26
  %1713 = extractelement <4 x half> %1710, i64 1, !dbg !26
  %1714 = extractelement <4 x half> %1710, i64 0, !dbg !26
  %1715 = bitcast i32 %1572 to float, !dbg !11
  %1716 = bitcast i32 %1575 to float, !dbg !11
  %1717 = insertelement <4 x float> poison, float %1562, i64 0, !dbg !11
  %1718 = insertelement <4 x float> %1717, float %1563, i64 1, !dbg !11
  %1719 = insertelement <4 x float> %1718, float %1715, i64 2, !dbg !11
  %1720 = insertelement <4 x float> %1719, float %1716, i64 3, !dbg !11
  %1721 = insertelement <4 x float> poison, float %1715, i64 0, !dbg !11
  %1722 = insertelement <4 x float> %1721, float %1716, i64 1, !dbg !11
  %1723 = insertelement <4 x float> %1722, float %1568, i64 2, !dbg !11
  %1724 = insertelement <4 x float> %1723, float %1569, i64 3, !dbg !11
  %1725 = select <4 x i1> %1227, <4 x float> %1720, <4 x float> %1724, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %1725, !dbg !25
  %1726 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !26
  %1727 = extractelement <4 x half> %1726, i64 3, !dbg !26
  %1728 = extractelement <4 x half> %1726, i64 2, !dbg !26
  %1729 = extractelement <4 x half> %1726, i64 1, !dbg !26
  %1730 = extractelement <4 x half> %1726, i64 0, !dbg !26
  %1731 = bitcast i32 %1447 to float, !dbg !11
  %1732 = bitcast i32 %1450 to float, !dbg !11
  %1733 = insertelement <4 x float> poison, float %1437, i64 0, !dbg !11
  %1734 = insertelement <4 x float> %1733, float %1438, i64 1, !dbg !11
  %1735 = insertelement <4 x float> %1734, float %1731, i64 2, !dbg !11
  %1736 = insertelement <4 x float> %1735, float %1732, i64 3, !dbg !11
  %1737 = insertelement <4 x float> poison, float %1731, i64 0, !dbg !11
  %1738 = insertelement <4 x float> %1737, float %1732, i64 1, !dbg !11
  %1739 = insertelement <4 x float> %1738, float %1443, i64 2, !dbg !11
  %1740 = insertelement <4 x float> %1739, float %1444, i64 3, !dbg !11
  %1741 = select <4 x i1> %1227, <4 x float> %1736, <4 x float> %1740, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1329, !dbg !25
  %normalized106 = fmul <4 x float> %row.reciprocal105, %1741, !dbg !25
  %1742 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !26
  %1743 = extractelement <4 x half> %1742, i64 3, !dbg !26
  %1744 = extractelement <4 x half> %1742, i64 2, !dbg !26
  %1745 = extractelement <4 x half> %1742, i64 1, !dbg !26
  %1746 = extractelement <4 x half> %1742, i64 0, !dbg !26
  %1747 = bitcast i32 %1324 to float, !dbg !11
  %1748 = bitcast i32 %1327 to float, !dbg !11
  %1749 = insertelement <4 x float> poison, float %1314, i64 0, !dbg !11
  %1750 = insertelement <4 x float> %1749, float %1315, i64 1, !dbg !11
  %1751 = insertelement <4 x float> %1750, float %1747, i64 2, !dbg !11
  %1752 = insertelement <4 x float> %1751, float %1748, i64 3, !dbg !11
  %1753 = insertelement <4 x float> poison, float %1747, i64 0, !dbg !11
  %1754 = insertelement <4 x float> %1753, float %1748, i64 1, !dbg !11
  %1755 = insertelement <4 x float> %1754, float %1320, i64 2, !dbg !11
  %1756 = insertelement <4 x float> %1755, float %1321, i64 3, !dbg !11
  %1757 = select <4 x i1> %1227, <4 x float> %1752, <4 x float> %1756, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %1757, !dbg !25
  %1758 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !26
  %1759 = extractelement <4 x half> %1758, i64 3, !dbg !26
  %1760 = extractelement <4 x half> %1758, i64 2, !dbg !26
  %1761 = extractelement <4 x half> %1758, i64 1, !dbg !26
  %1762 = extractelement <4 x half> %1758, i64 0, !dbg !26
  %address.thread934 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !27, !range !9, !frisk.register_lifetimes !5
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !8
  %1763 = lshr i64 %address.local935, 2, !dbg !27
  %address.local938 = and i64 %1763, 48, !dbg !27
  %address.local940 = and i64 %address.local935, 240, !dbg !27
  %address.local939 = add i64 %26, %address.local935, !dbg !27
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !27
  %address.local942 = add i64 %address.local941, %address.local938, !dbg !27
  %1764 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !27
  %address.local945.idx = shl nuw nsw i64 %address.local942, 8, !dbg !27
  %address.local945 = getelementptr inbounds nuw i8, ptr addrspace(1) %1764, i64 %address.local945.idx, !dbg !27
  %address.local947 = and i64 %1763, 12, !dbg !10
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !27
  store half %1762, ptr addrspace(1) %address.local948, align 2, !dbg !27
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !27
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local950, !dbg !27
  store half %1761, ptr addrspace(1) %address.local951, align 2, !dbg !27
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !27
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local953, !dbg !27
  store half %1760, ptr addrspace(1) %address.local954, align 2, !dbg !27
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !27
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local956, !dbg !27
  store half %1759, ptr addrspace(1) %address.local957, align 2, !dbg !27
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !27
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local959, !dbg !27
  store half %1746, ptr addrspace(1) %address.local960, align 2, !dbg !27
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !27
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local962, !dbg !27
  store half %1745, ptr addrspace(1) %address.local963, align 2, !dbg !27
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !27
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local965, !dbg !27
  store half %1744, ptr addrspace(1) %address.local966, align 2, !dbg !27
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !27
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local968, !dbg !27
  store half %1743, ptr addrspace(1) %address.local969, align 2, !dbg !27
  %address.local973 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local945, i64 16384, !dbg !27
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !27
  store half %1730, ptr addrspace(1) %address.local974, align 2, !dbg !27
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local950, !dbg !27
  store half %1729, ptr addrspace(1) %address.local976, align 2, !dbg !27
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local953, !dbg !27
  store half %1728, ptr addrspace(1) %address.local978, align 2, !dbg !27
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local956, !dbg !27
  store half %1727, ptr addrspace(1) %address.local980, align 2, !dbg !27
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local959, !dbg !27
  store half %1714, ptr addrspace(1) %address.local982, align 2, !dbg !27
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local962, !dbg !27
  store half %1713, ptr addrspace(1) %address.local984, align 2, !dbg !27
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local965, !dbg !27
  store half %1712, ptr addrspace(1) %address.local986, align 2, !dbg !27
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local968, !dbg !27
  store half %1711, ptr addrspace(1) %address.local988, align 2, !dbg !27
  %normalized109 = fmul <4 x float> %row.reciprocal, %1676, !dbg !25
  %1765 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !26
  %1766 = extractelement <4 x half> %1765, i64 3, !dbg !26
  %1767 = extractelement <4 x half> %1765, i64 2, !dbg !26
  %1768 = extractelement <4 x half> %1765, i64 1, !dbg !26
  %1769 = extractelement <4 x half> %1765, i64 0, !dbg !26
  %normalized110 = fmul <4 x float> %row.reciprocal, %1552, !dbg !25
  %1770 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !26
  %1771 = extractelement <4 x half> %1770, i64 3, !dbg !26
  %1772 = extractelement <4 x half> %1770, i64 2, !dbg !26
  %1773 = extractelement <4 x half> %1770, i64 1, !dbg !26
  %1774 = extractelement <4 x half> %1770, i64 0, !dbg !26
  %normalized112 = fmul <4 x float> %row.reciprocal105, %1428, !dbg !25
  %1775 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !26
  %1776 = extractelement <4 x half> %1775, i64 3, !dbg !26
  %1777 = extractelement <4 x half> %1775, i64 2, !dbg !26
  %1778 = extractelement <4 x half> %1775, i64 1, !dbg !26
  %1779 = extractelement <4 x half> %1775, i64 0, !dbg !26
  %normalized113 = fmul <4 x float> %row.reciprocal105, %1304, !dbg !25
  %1780 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !26
  %1781 = extractelement <4 x half> %1780, i64 3, !dbg !26
  %1782 = extractelement <4 x half> %1780, i64 2, !dbg !26
  %1783 = extractelement <4 x half> %1780, i64 1, !dbg !26
  %1784 = extractelement <4 x half> %1780, i64 0, !dbg !26
  %address.local1003 = or disjoint i64 %address.local947, 32, !dbg !27
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1003, !dbg !27
  store half %1784, ptr addrspace(1) %address.local1004, align 2, !dbg !27
  %address.local1006 = or disjoint i64 %address.local947, 33, !dbg !27
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1006, !dbg !27
  store half %1783, ptr addrspace(1) %address.local1007, align 2, !dbg !27
  %address.local1009 = or disjoint i64 %address.local947, 34, !dbg !27
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1009, !dbg !27
  store half %1782, ptr addrspace(1) %address.local1010, align 2, !dbg !27
  %address.local1012 = or disjoint i64 %address.local947, 35, !dbg !27
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1012, !dbg !27
  store half %1781, ptr addrspace(1) %address.local1013, align 2, !dbg !27
  %address.local1015 = or disjoint i64 %address.local947, 48, !dbg !27
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1015, !dbg !27
  store half %1779, ptr addrspace(1) %address.local1016, align 2, !dbg !27
  %address.local1018 = or disjoint i64 %address.local947, 49, !dbg !27
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1018, !dbg !27
  store half %1778, ptr addrspace(1) %address.local1019, align 2, !dbg !27
  %address.local1021 = or disjoint i64 %address.local947, 50, !dbg !27
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1021, !dbg !27
  store half %1777, ptr addrspace(1) %address.local1022, align 2, !dbg !27
  %address.local1024 = or i64 %1763, 51, !dbg !27
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1024, !dbg !27
  store half %1776, ptr addrspace(1) %address.local1025, align 2, !dbg !27
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1003, !dbg !27
  store half %1774, ptr addrspace(1) %address.local1030, align 2, !dbg !27
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1006, !dbg !27
  store half %1773, ptr addrspace(1) %address.local1032, align 2, !dbg !27
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1009, !dbg !27
  store half %1772, ptr addrspace(1) %address.local1034, align 2, !dbg !27
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1012, !dbg !27
  store half %1771, ptr addrspace(1) %address.local1036, align 2, !dbg !27
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1015, !dbg !27
  store half %1769, ptr addrspace(1) %address.local1038, align 2, !dbg !27
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1018, !dbg !27
  store half %1768, ptr addrspace(1) %address.local1040, align 2, !dbg !27
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1021, !dbg !27
  store half %1767, ptr addrspace(1) %address.local1042, align 2, !dbg !27
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1024, !dbg !27
  store half %1766, ptr addrspace(1) %address.local1044, align 2, !dbg !27
  %normalized115 = fmul <4 x float> %row.reciprocal, %1643, !dbg !25
  %1785 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !26
  %1786 = extractelement <4 x half> %1785, i64 3, !dbg !26
  %1787 = extractelement <4 x half> %1785, i64 2, !dbg !26
  %1788 = extractelement <4 x half> %1785, i64 1, !dbg !26
  %1789 = extractelement <4 x half> %1785, i64 0, !dbg !26
  %normalized116 = fmul <4 x float> %row.reciprocal, %1518, !dbg !25
  %1790 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !26
  %1791 = extractelement <4 x half> %1790, i64 3, !dbg !26
  %1792 = extractelement <4 x half> %1790, i64 2, !dbg !26
  %1793 = extractelement <4 x half> %1790, i64 1, !dbg !26
  %1794 = extractelement <4 x half> %1790, i64 0, !dbg !26
  %normalized118 = fmul <4 x float> %row.reciprocal105, %1395, !dbg !25
  %1795 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !26
  %1796 = extractelement <4 x half> %1795, i64 3, !dbg !26
  %1797 = extractelement <4 x half> %1795, i64 2, !dbg !26
  %1798 = extractelement <4 x half> %1795, i64 1, !dbg !26
  %1799 = extractelement <4 x half> %1795, i64 0, !dbg !26
  %normalized119 = fmul <4 x float> %row.reciprocal105, %1270, !dbg !25
  %1800 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !26
  %1801 = extractelement <4 x half> %1800, i64 3, !dbg !26
  %1802 = extractelement <4 x half> %1800, i64 2, !dbg !26
  %1803 = extractelement <4 x half> %1800, i64 1, !dbg !26
  %1804 = extractelement <4 x half> %1800, i64 0, !dbg !26
  %address.local1059 = or disjoint i64 %address.local947, 64, !dbg !27
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1059, !dbg !27
  store half %1804, ptr addrspace(1) %address.local1060, align 2, !dbg !27
  %address.local1062 = or disjoint i64 %address.local947, 65, !dbg !27
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1062, !dbg !27
  store half %1803, ptr addrspace(1) %address.local1063, align 2, !dbg !27
  %address.local1065 = or disjoint i64 %address.local947, 66, !dbg !27
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1065, !dbg !27
  store half %1802, ptr addrspace(1) %address.local1066, align 2, !dbg !27
  %address.local1068 = or disjoint i64 %address.local947, 67, !dbg !27
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1068, !dbg !27
  store half %1801, ptr addrspace(1) %address.local1069, align 2, !dbg !27
  %address.local1071 = or disjoint i64 %address.local947, 80, !dbg !27
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1071, !dbg !27
  store half %1799, ptr addrspace(1) %address.local1072, align 2, !dbg !27
  %address.local1074 = or disjoint i64 %address.local947, 81, !dbg !27
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1074, !dbg !27
  store half %1798, ptr addrspace(1) %address.local1075, align 2, !dbg !27
  %address.local1077 = or disjoint i64 %address.local947, 82, !dbg !27
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1077, !dbg !27
  store half %1797, ptr addrspace(1) %address.local1078, align 2, !dbg !27
  %address.local1080 = or disjoint i64 %address.local947, 83, !dbg !27
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1080, !dbg !27
  store half %1796, ptr addrspace(1) %address.local1081, align 2, !dbg !27
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1059, !dbg !27
  store half %1794, ptr addrspace(1) %address.local1086, align 2, !dbg !27
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1062, !dbg !27
  store half %1793, ptr addrspace(1) %address.local1088, align 2, !dbg !27
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1065, !dbg !27
  store half %1792, ptr addrspace(1) %address.local1090, align 2, !dbg !27
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1068, !dbg !27
  store half %1791, ptr addrspace(1) %address.local1092, align 2, !dbg !27
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1071, !dbg !27
  store half %1789, ptr addrspace(1) %address.local1094, align 2, !dbg !27
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1074, !dbg !27
  store half %1788, ptr addrspace(1) %address.local1096, align 2, !dbg !27
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1077, !dbg !27
  store half %1787, ptr addrspace(1) %address.local1098, align 2, !dbg !27
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1080, !dbg !27
  store half %1786, ptr addrspace(1) %address.local1100, align 2, !dbg !27
  %normalized121 = fmul <4 x float> %row.reciprocal, %1610, !dbg !25
  %1805 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !26
  %1806 = extractelement <4 x half> %1805, i64 3, !dbg !26
  %1807 = extractelement <4 x half> %1805, i64 2, !dbg !26
  %1808 = extractelement <4 x half> %1805, i64 1, !dbg !26
  %1809 = extractelement <4 x half> %1805, i64 0, !dbg !26
  %normalized122 = fmul <4 x float> %row.reciprocal, %1484, !dbg !25
  %1810 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !26
  %1811 = extractelement <4 x half> %1810, i64 3, !dbg !26
  %1812 = extractelement <4 x half> %1810, i64 2, !dbg !26
  %1813 = extractelement <4 x half> %1810, i64 1, !dbg !26
  %1814 = extractelement <4 x half> %1810, i64 0, !dbg !26
  %normalized124 = fmul <4 x float> %row.reciprocal105, %1362, !dbg !25
  %1815 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !26
  %1816 = extractelement <4 x half> %1815, i64 3, !dbg !26
  %1817 = extractelement <4 x half> %1815, i64 2, !dbg !26
  %1818 = extractelement <4 x half> %1815, i64 1, !dbg !26
  %1819 = extractelement <4 x half> %1815, i64 0, !dbg !26
  %normalized125 = fmul <4 x float> %row.reciprocal105, %1236, !dbg !25
  %1820 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !26
  %1821 = extractelement <4 x half> %1820, i64 3, !dbg !26
  %1822 = extractelement <4 x half> %1820, i64 2, !dbg !26
  %1823 = extractelement <4 x half> %1820, i64 1, !dbg !26
  %1824 = extractelement <4 x half> %1820, i64 0, !dbg !26
  %address.local1115 = or disjoint i64 %address.local947, 96, !dbg !27
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1115, !dbg !27
  store half %1824, ptr addrspace(1) %address.local1116, align 2, !dbg !27
  %address.local1118 = or disjoint i64 %address.local947, 97, !dbg !27
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1118, !dbg !27
  store half %1823, ptr addrspace(1) %address.local1119, align 2, !dbg !27
  %address.local1121 = or disjoint i64 %address.local947, 98, !dbg !27
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1121, !dbg !27
  store half %1822, ptr addrspace(1) %address.local1122, align 2, !dbg !27
  %address.local1124 = or disjoint i64 %address.local947, 99, !dbg !27
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1124, !dbg !27
  store half %1821, ptr addrspace(1) %address.local1125, align 2, !dbg !27
  %address.local1127 = or disjoint i64 %address.local947, 112, !dbg !27
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1127, !dbg !27
  store half %1819, ptr addrspace(1) %address.local1128, align 2, !dbg !27
  %address.local1130 = or disjoint i64 %address.local947, 113, !dbg !27
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1130, !dbg !27
  store half %1818, ptr addrspace(1) %address.local1131, align 2, !dbg !27
  %address.local1133 = or disjoint i64 %address.local947, 114, !dbg !27
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1133, !dbg !27
  store half %1817, ptr addrspace(1) %address.local1134, align 2, !dbg !27
  %address.local1136 = or i64 %1763, 115, !dbg !27
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1136, !dbg !27
  store half %1816, ptr addrspace(1) %address.local1137, align 2, !dbg !27
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1115, !dbg !27
  store half %1814, ptr addrspace(1) %address.local1142, align 2, !dbg !27
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1118, !dbg !27
  store half %1813, ptr addrspace(1) %address.local1144, align 2, !dbg !27
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1121, !dbg !27
  store half %1812, ptr addrspace(1) %address.local1146, align 2, !dbg !27
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1124, !dbg !27
  store half %1811, ptr addrspace(1) %address.local1148, align 2, !dbg !27
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1127, !dbg !27
  store half %1809, ptr addrspace(1) %address.local1150, align 2, !dbg !27
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1130, !dbg !27
  store half %1808, ptr addrspace(1) %address.local1152, align 2, !dbg !27
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1133, !dbg !27
  store half %1807, ptr addrspace(1) %address.local1154, align 2, !dbg !27
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1136, !dbg !27
  store half %1806, ptr addrspace(1) %address.local1156, align 2, !dbg !27
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
