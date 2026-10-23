; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/q_bk_retention_sweep/retained_24words.ll'
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
  %dma.buffer.offset1677 = trunc i64 %33 to i32, !dbg !8
  %34 = insertelement <2 x i64> poison, i64 %30, i64 0, !dbg !8
  %35 = bitcast <2 x i64> %34 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1678 = shufflevector <4 x i32> %35, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
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
  %dma.buffer.offset1674 = trunc i64 %38 to i32, !dbg !8
  %39 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %40 = sext i32 %39 to i64, !dbg !7
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !8
  %.idx3 = shl nuw nsw i64 %28, 1, !dbg !8
  %41 = add i64 %.idx3, %.idx, !dbg !8
  %address.local365.idx = shl nuw nsw i64 %address.local348, 1, !dbg !8
  %42 = or disjoint i64 %address.local365.idx, %41, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %dma.buffer.offset1671 = or disjoint i32 %43, 64, !dbg !8
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !8
  %44 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local341, i64 0, !dbg !8
  %45 = insertelement <4 x ptr addrspace(3)> %44, ptr addrspace(3) %address.local353, i64 1, !dbg !8
  %46 = insertelement <4 x ptr addrspace(3)> %45, ptr addrspace(3) %address.local362, i64 2, !dbg !8
  %47 = insertelement <4 x ptr addrspace(3)> %46, ptr addrspace(3) %address.local366, i64 3, !dbg !8
  %48 = ptrtoint <4 x ptr addrspace(3)> %47 to <4 x i64>, !dbg !8
  %49 = trunc <4 x i64> %48 to <4 x i32>, !dbg !8
  %50 = extractelement <4 x i32> %49, i64 0, !dbg !8
  %51 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %dma.buffer.lds1679 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %51), !dbg !8
  %52 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1679, i32 %dma.buffer.offset1677, <4 x i32> %dma.buffer.rsrc1678) #9, !dbg !8
  %53 = extractelement <4 x i32> %49, i64 1, !dbg !8
  %54 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %dma.buffer.lds1676 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %54), !dbg !8
  %55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1676, i32 %dma.buffer.offset1674, <4 x i32> %dma.buffer.rsrc1678) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %56 = extractelement <4 x i32> %49, i64 2, !dbg !8
  %57 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %56), !dbg !8
  %dma.buffer.lds1673 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %57), !dbg !8
  %58 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1673, i32 %dma.buffer.offset1671, <4 x i32> %dma.buffer.rsrc1678) #9, !dbg !8
  %59 = extractelement <4 x i32> %49, i64 3, !dbg !8
  %60 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %59), !dbg !8
  %address.local369.idx = shl nuw nsw i64 %address.local360, 1, !dbg !8
  %61 = or disjoint i64 %address.local369.idx, %41, !dbg !8
  %62 = trunc i64 %61 to i32, !dbg !8
  %dma.buffer.offset1668 = or disjoint i32 %62, 64, !dbg !8
  %dma.buffer.lds1670 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1670, i32 %dma.buffer.offset1668, <4 x i32> %dma.buffer.rsrc1678) #9, !dbg !8
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
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !10
  %67 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !10
  %68 = fmul <4 x half> %65, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !10
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !10
  %70 = fmul <4 x half> %67, splat (half 1.275630e-01), !dbg !10
  %71 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1665 = or disjoint i32 %43, 128, !dbg !8
  %72 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1679, i32 %dma.buffer.offset1665, <4 x i32> %dma.buffer.rsrc1678) #9, !dbg !8
  %dma.buffer.offset1662 = or disjoint i32 %62, 128, !dbg !8
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1676, i32 %dma.buffer.offset1662, <4 x i32> %dma.buffer.rsrc1678) #9, !dbg !8
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !10
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !10
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !10
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !10
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !10
  %77 = fmul <4 x half> %74, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !10
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !10
  %79 = fmul <4 x half> %75, splat (half 1.275630e-01), !dbg !10
  %80 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %82 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !10
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !10
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !10
  %87 = fmul <4 x half> %83, splat (half 1.275630e-01), !dbg !10
  %88 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  %89 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %90 = or disjoint i64 %24, 127, !dbg !12
  %91 = add i64 %24, 128, !dbg !12
  %92 = icmp uge i64 %91, %90, !dbg !12
  %93 = sub nsw i64 32, %14, !dbg !13
  %94 = shl i64 %16, 7, !dbg !14
  %95 = and i64 %94, 3968, !dbg !14
  %96 = shl nsw i64 %40, 12, !dbg !14
  %97 = or disjoint i64 %95, %96, !dbg !14
  %98 = shl nuw nsw i32 %29, 2, !dbg !15
  %99 = and i32 %98, 12, !dbg !15
  %100 = and i32 %29, 1, !dbg !15
  %101 = or disjoint i32 %99, %100, !dbg !15
  %102 = trunc nuw nsw i32 %101 to i16, !dbg !15
  %.lhs.trunc = add nuw nsw i16 %102, 1, !dbg !15
  %103 = lshr i16 %.lhs.trunc, 1, !dbg !15
  %.sext = zext nneg i16 %103 to i64, !dbg !15
  %104 = icmp ult i64 %10, 1024, !dbg !11
  br i1 %104, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !11

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre108 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre110 = add nuw nsw i32 %.pre108, 64, !dbg !11
  %.pre112 = xor i32 %.pre108, 16, !dbg !11
  %.pre114 = and i32 %.pre110, 192, !dbg !11
  %.pre116 = xor i32 %.pre108, 32, !dbg !11
  br label %._crit_edge, !dbg !11

.lr.ph:                                           ; preds = %4
  %105 = zext nneg i32 %29 to i64, !dbg !8
  %106 = ptrtoint ptr addrspace(1) %1 to i64
  %107 = insertelement <2 x i64> poison, i64 %106, i64 0
  %108 = bitcast <2 x i64> %107 to <4 x i32>
  %dma.buffer.rsrc1660 = shufflevector <4 x i32> %108, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %109 = ptrtoint ptr addrspace(1) %2 to i64
  %110 = insertelement <2 x i64> poison, i64 %109, i64 0
  %111 = bitcast <2 x i64> %110 to <4 x i32>
  %dma.buffer.rsrc1630 = shufflevector <4 x i32> %111, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %112 = lshr i64 %105, 2
  %113 = and i64 %112, 48
  %114 = and i32 %29, 240
  %index.compact = zext nneg i32 %114 to i64
  %115 = sub nsw i64 %105, %index.compact
  %116 = add nsw i64 %115, %113
  %117 = add nsw i64 %116, %24
  %118 = and i64 %105, 48
  %119 = add i64 %117, 1
  %120 = lshr i64 %119, 32
  %compare.high = trunc nuw i64 %120 to i32
  %compare.low.base = trunc i64 %119 to i32
  %121 = add i64 %117, 65
  %122 = lshr i64 %121, 32
  %compare.high1346 = trunc nuw i64 %122 to i32
  %compare.low.base1350 = trunc i64 %121 to i32
  %123 = insertelement <32 x i32> poison, i32 %compare.low.base1350, i64 0, !dbg !16
  %124 = shufflevector <32 x i32> %123, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  %125 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %126 = shufflevector <32 x i32> %125, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  br label %127, !dbg !11

127:                                              ; preds = %.lr.ph, %543
  %128 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1197, %543 ]
  %129 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1193, %543 ]
  %130 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1189, %543 ]
  %131 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %886, %543 ]
  %132 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1185, %543 ]
  %133 = phi i64 [ 0, %.lr.ph ], [ %1198, %543 ]
  %134 = shl nuw nsw i64 %133, 7, !dbg !11
  %135 = lshr i64 %133, 12, !dbg !14
  %136 = add nuw nsw i64 %135, %16, !dbg !14
  %137 = lshr i64 %133, 5, !dbg !14
  %138 = and i64 %137, 127, !dbg !14
  %139 = shl i64 %136, 19, !dbg !14
  %140 = and i64 %139, 16252928, !dbg !14
  %141 = shl i64 %137, 19, !dbg !14
  %142 = shl nuw nsw i64 %133, 14, !dbg !14
  %143 = sub i64 %142, %141, !dbg !14
  %address.thread431 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !14, !range !9, !frisk.register_lifetimes !5
  %address.local432 = zext nneg i32 %address.thread431 to i64, !dbg !8
  %address.local433 = shl nuw nsw i64 %address.local432, 3, !dbg !8
  %address.local434 = and i64 %address.local433, 1536, !dbg !8
  %address.local435 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local434, !dbg !8
  %address.local438 = lshr i64 %address.local432, 4, !dbg !14
  %address.local439 = and i64 %address.local438, 8, !dbg !14
  %address.local440 = and i64 %address.local432, 123, !dbg !14
  %address.local441 = lshr i64 %address.local432, 2, !dbg !14
  %address.local442 = and i64 %address.local441, 4, !dbg !14
  %address.local443 = add nuw nsw i64 %address.local442, %address.local432, !dbg !14
  %address.local444 = and i64 %address.local443, 4, !dbg !14
  %address.local445 = or disjoint i64 %address.local444, %address.local440, !dbg !14
  %address.local446 = shl nuw nsw i64 %address.local445, 7, !dbg !14
  %address.local447 = or disjoint i64 %address.local446, %address.local439, !dbg !14
  %.idx710 = or disjoint i64 %140, %138, !dbg !14
  %144 = add i64 %.idx710, %143, !dbg !14
  %145 = add i64 %address.local447, %144, !dbg !14
  %.tr = trunc i64 %145 to i32, !dbg !14
  %dma.buffer.offset1659 = shl i32 %.tr, 1, !dbg !14
  %address.local449 = or disjoint i64 %address.local432, 256, !dbg !8
  %address.local450 = shl nuw nsw i64 %address.local449, 3, !dbg !8
  %address.local451 = and i64 %address.local450, 3584, !dbg !8
  %address.local452 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local451, !dbg !8
  %address.local455 = lshr i64 %address.local449, 4, !dbg !14
  %address.local456 = and i64 %address.local455, 24, !dbg !14
  %address.local464 = or disjoint i64 %address.local446, %address.local456, !dbg !14
  %146 = add i64 %address.local464, %144, !dbg !14
  %.tr15 = trunc i64 %146 to i32, !dbg !14
  %dma.buffer.offset1656 = shl i32 %.tr15, 1, !dbg !14
  %147 = add nuw nsw i64 %133, 1024, !dbg !14
  %148 = lshr i64 %147, 12, !dbg !14
  %149 = add nuw nsw i64 %148, %16, !dbg !14
  %150 = add nsw i64 %137, %97, !dbg !14
  %151 = shl i64 %149, 19, !dbg !14
  %152 = and i64 %151, 16252928, !dbg !14
  %address.local466 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local434, !dbg !8
  %153 = add i64 %143, %137, !dbg !14
  %.idx1620 = add i64 %153, %97, !dbg !14
  %154 = add i64 %.idx1620, %152, !dbg !14
  %address.local470 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local451, !dbg !8
  %155 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local435, i64 0, !dbg !8
  %156 = insertelement <4 x ptr addrspace(3)> %155, ptr addrspace(3) %address.local452, i64 1, !dbg !8
  %157 = insertelement <4 x ptr addrspace(3)> %156, ptr addrspace(3) %address.local466, i64 2, !dbg !8
  %158 = insertelement <4 x ptr addrspace(3)> %157, ptr addrspace(3) %address.local470, i64 3, !dbg !8
  %159 = ptrtoint <4 x ptr addrspace(3)> %158 to <4 x i64>, !dbg !8
  %160 = trunc <4 x i64> %159 to <4 x i32>, !dbg !8
  %161 = extractelement <4 x i32> %160, i64 0, !dbg !14
  %162 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %161), !dbg !14
  %dma.buffer.lds1661 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %162), !dbg !14
  %163 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1661, i32 %dma.buffer.offset1659, <4 x i32> %dma.buffer.rsrc1660) #9, !dbg !14
  %164 = extractelement <4 x i32> %160, i64 1, !dbg !14
  %165 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %164), !dbg !14
  %dma.buffer.lds1658 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %165), !dbg !14
  %166 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1658, i32 %dma.buffer.offset1656, <4 x i32> %dma.buffer.rsrc1660) #9, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %167 = extractelement <4 x i32> %160, i64 2, !dbg !14
  %168 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %167), !dbg !14
  %dma.buffer.lds1655 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %168), !dbg !14
  %169 = extractelement <4 x i32> %160, i64 3, !dbg !14
  %address.local478 = shl nuw nsw i64 %address.local432, 1, !dbg !15
  %address.local479 = and i64 %address.local478, 24, !dbg !15
  %address.local481 = shl i64 %address.local432, 62, !dbg !15
  %address.local482 = ashr exact i64 %address.local481, 57, !dbg !15
  %address.local483 = and i64 %address.local482, -64, !dbg !15
  %address.local486 = and i64 %address.local441, 12, !dbg !10
  %170 = and i64 %address.local432, 32, !dbg !10
  %.not28 = icmp eq i64 %170, 0, !dbg !10
  %address.local490 = select i1 %.not28, i64 0, i64 1016, !dbg !10
  %.idx29 = mul nuw nsw i64 %address.local432, 320, !dbg !15
  %171 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx29, !dbg !15
  %.idx30 = mul nsw i64 %address.local441, -1280, !dbg !15
  %172 = getelementptr i8, ptr addrspace(3) %171, i64 %.idx30, !dbg !15
  %173 = getelementptr [2 x i8], ptr addrspace(3) %172, i64 %address.local479, !dbg !15
  %174 = getelementptr [2 x i8], ptr addrspace(3) %173, i64 %address.local483, !dbg !15
  %175 = getelementptr [2 x i8], ptr addrspace(3) %174, i64 %address.local486, !dbg !15
  %address.local492 = getelementptr [2 x i8], ptr addrspace(3) %175, i64 %address.local490, !dbg !15
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local492, i64 4096, !dbg !15
  %address.local503 = shl nuw nsw i32 %address.thread431, 1, !dbg !15
  %address.local504 = and i32 %address.local503, 24, !dbg !15
  %address.local505 = shl nuw nsw i32 %address.thread431, 8, !dbg !15
  %address.local506 = and i32 %address.local505, 768, !dbg !15
  %address.local507 = or disjoint i32 %address.local504, %address.local506, !dbg !15
  %address.local508 = shl nuw nsw i32 %address.thread431, 5, !dbg !15
  %address.local509 = and i32 %address.local508, 96, !dbg !15
  %address.local510 = or disjoint i32 %address.local507, %address.local509, !dbg !15
  %address.local511 = zext nneg i32 %address.local510 to i64, !dbg !15
  %address.local512 = lshr i32 %address.thread431, 1, !dbg !15
  %address.local513 = and i32 %address.local512, 1, !dbg !15
  %address.local514 = zext nneg i32 %address.local513 to i64, !dbg !15
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local492, i64 128, !dbg !15
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local492, i64 4224, !dbg !15
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local492, i64 1024, !dbg !15
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local492, i64 5120, !dbg !15
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local492, i64 1152, !dbg !15
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local492, i64 5248, !dbg !15
  %176 = add nuw nsw i64 %133, 2048, !dbg !14
  %177 = lshr i64 %176, 12, !dbg !14
  %178 = add nuw nsw i64 %177, %16, !dbg !14
  %179 = insertelement <2 x i64> poison, i64 %150, i64 0, !dbg !14
  %180 = shufflevector <2 x i64> %179, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !14
  %181 = icmp slt <2 x i64> %180, <i64 -32, i64 -64>, !dbg !14
  %182 = extractelement <2 x i1> %181, i64 0, !dbg !14
  %183 = sext i1 %182 to i64, !dbg !14
  %184 = add nsw <2 x i64> %180, <i64 32, i64 64>, !dbg !14
  %185 = sub nsw <2 x i64> <i64 -33, i64 -65>, %180, !dbg !14
  %186 = select <2 x i1> %181, <2 x i64> %185, <2 x i64> %184, !dbg !14
  %187 = sdiv <2 x i64> %186, splat (i64 128), !dbg !14
  %188 = extractelement <2 x i64> %187, i64 0, !dbg !14
  %189 = xor i64 %188, %183, !dbg !14
  %190 = shl nsw i64 %189, 7, !dbg !14
  %191 = sub i64 %154, %190, !dbg !14
  %192 = add i64 %address.local447, %191, !dbg !14
  %.tr21 = trunc i64 %192 to i32, !dbg !14
  %193 = shl i32 %.tr21, 1, !dbg !14
  %dma.buffer.offset1653 = add i32 %193, 64, !dbg !14
  %194 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1655, i32 %dma.buffer.offset1653, <4 x i32> %dma.buffer.rsrc1660) #9, !dbg !14
  %195 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %169), !dbg !14
  %196 = add i64 %address.local464, %191, !dbg !14
  %.tr27 = trunc i64 %196 to i32, !dbg !14
  %197 = shl i32 %.tr27, 1, !dbg !14
  %dma.buffer.offset1650 = add i32 %197, 64, !dbg !14
  %dma.buffer.lds1652 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %195), !dbg !14
  %198 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1652, i32 %dma.buffer.offset1650, <4 x i32> %dma.buffer.rsrc1660) #9, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %address.local492, align 8, !dbg !15
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %199, <4 x float> zeroinitializer) #8, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %.sext) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local515 = add nsw i64 %address.base, %address.local514, !dbg !15
  %address.local516 = shl i64 %address.local515, 6, !dbg !15
  %address.local517 = sub i64 %address.local511, %address.local516, !dbg !15
  %202 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local517, !dbg !15
  %203 = getelementptr [2 x i8], ptr addrspace(3) %202, i64 %address.local486, !dbg !15
  %204 = getelementptr [2 x i8], ptr addrspace(3) %203, i64 %address.local490, !dbg !15
  %address.local521 = getelementptr i8, ptr addrspace(3) %204, i64 64, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %address.local521, align 8, !dbg !15
  %206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %201, <4 x float> %200) #8, !dbg !15
  %207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %205, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %204, i64 4160, !dbg !15
  %208 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !15
  %210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %208, <4 x float> %207) #8, !dbg !15
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %209, <4 x float> zeroinitializer) #8, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !15
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %204, i64 192, !dbg !15
  %213 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !15
  %214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %212, <4 x float> %211) #8, !dbg !15
  %215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %213, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %204, i64 4288, !dbg !15
  %216 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !15
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %216, <4 x float> %215) #8, !dbg !15
  %219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %217, <4 x float> zeroinitializer) #8, !dbg !15
  %220 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !15
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %204, i64 1088, !dbg !15
  %221 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !15
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %220, <4 x float> %219) #8, !dbg !15
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %221, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %204, i64 5184, !dbg !15
  %224 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !15
  %225 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !15
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %224, <4 x float> %223) #8, !dbg !15
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %225, <4 x float> zeroinitializer) #8, !dbg !15
  %228 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !15
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %204, i64 1216, !dbg !15
  %229 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !15
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %228, <4 x float> %227) #8, !dbg !15
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %229, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %204, i64 5312, !dbg !15
  %232 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !15
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %232, <4 x float> %231) #8, !dbg !15
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %199, <4 x float> zeroinitializer) #8, !dbg !15
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %201, <4 x float> %234) #8, !dbg !15
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %205, <4 x float> zeroinitializer) #8, !dbg !15
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %208, <4 x float> %236) #8, !dbg !15
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %209, <4 x float> zeroinitializer) #8, !dbg !15
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %212, <4 x float> %238) #8, !dbg !15
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %213, <4 x float> zeroinitializer) #8, !dbg !15
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %216, <4 x float> %240) #8, !dbg !15
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %217, <4 x float> zeroinitializer) #8, !dbg !15
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %220, <4 x float> %242) #8, !dbg !15
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %221, <4 x float> zeroinitializer) #8, !dbg !15
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %224, <4 x float> %244) #8, !dbg !15
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %225, <4 x float> zeroinitializer) #8, !dbg !15
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %228, <4 x float> %246) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %229, <4 x float> zeroinitializer) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %232, <4 x float> %248) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %250 = extractelement <2 x i1> %181, i64 1, !dbg !14
  %251 = sext i1 %250 to i64, !dbg !14
  %252 = extractelement <2 x i64> %187, i64 1, !dbg !14
  %253 = xor i64 %252, %251, !dbg !14
  %254 = shl i64 %178, 19, !dbg !14
  %255 = and i64 %254, 16252928, !dbg !14
  %256 = add i64 %.idx1620, %255, !dbg !14
  %257 = shl nsw i64 %253, 7, !dbg !14
  %258 = sub i64 %256, %257, !dbg !14
  %259 = add i64 %address.local447, %258, !dbg !14
  %.tr36 = trunc i64 %259 to i32, !dbg !14
  %260 = shl i32 %.tr36, 1, !dbg !14
  %dma.buffer.offset1647 = add i32 %260, 128, !dbg !14
  %261 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1661, i32 %dma.buffer.offset1647, <4 x i32> %dma.buffer.rsrc1660) #9, !dbg !14
  %262 = add i64 %address.local464, %258, !dbg !14
  %.tr42 = trunc i64 %262 to i32, !dbg !14
  %263 = shl i32 %.tr42, 1, !dbg !14
  %dma.buffer.offset1644 = add i32 %263, 128, !dbg !14
  %264 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1658, i32 %dma.buffer.offset1644, <4 x i32> %dma.buffer.rsrc1660) #9, !dbg !14
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local492, i64 8192, !dbg !15
  %265 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !15
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %265, <4 x float> %206) #8, !dbg !15
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local492, i64 12288, !dbg !15
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !15
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %204, i64 8256, !dbg !15
  %268 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !15
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %267, <4 x float> %266) #8, !dbg !15
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %268, <4 x float> %210) #8, !dbg !15
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %204, i64 12352, !dbg !15
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !15
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local492, i64 8320, !dbg !15
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !15
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %271, <4 x float> %270) #8, !dbg !15
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %272, <4 x float> %214) #8, !dbg !15
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local492, i64 12416, !dbg !15
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !15
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %204, i64 8384, !dbg !15
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !15
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %275, <4 x float> %274) #8, !dbg !15
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %276, <4 x float> %218) #8, !dbg !15
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %204, i64 12480, !dbg !15
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !15
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local492, i64 9216, !dbg !15
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !15
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %279, <4 x float> %278) #8, !dbg !15
  %282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %280, <4 x float> %222) #8, !dbg !15
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local492, i64 13312, !dbg !15
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !15
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %204, i64 9280, !dbg !15
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !15
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %283, <4 x float> %282) #8, !dbg !15
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %284, <4 x float> %226) #8, !dbg !15
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %204, i64 13376, !dbg !15
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !15
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local492, i64 9344, !dbg !15
  %288 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !15
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %287, <4 x float> %286) #8, !dbg !15
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %288, <4 x float> %230) #8, !dbg !15
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local492, i64 13440, !dbg !15
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !15
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %204, i64 9408, !dbg !15
  %292 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !15
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %291, <4 x float> %290) #8, !dbg !15
  %294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %292, <4 x float> %233) #8, !dbg !15
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %204, i64 13504, !dbg !15
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !15
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %295, <4 x float> %294) #8, !dbg !15
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %265, <4 x float> %235) #8, !dbg !15
  %298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %267, <4 x float> %297) #8, !dbg !15
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %268, <4 x float> %237) #8, !dbg !15
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %271, <4 x float> %299) #8, !dbg !15
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %272, <4 x float> %239) #8, !dbg !15
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %275, <4 x float> %301) #8, !dbg !15
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %276, <4 x float> %241) #8, !dbg !15
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %279, <4 x float> %303) #8, !dbg !15
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %280, <4 x float> %243) #8, !dbg !15
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %283, <4 x float> %305) #8, !dbg !15
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %284, <4 x float> %245) #8, !dbg !15
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %287, <4 x float> %307) #8, !dbg !15
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %288, <4 x float> %247) #8, !dbg !15
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %291, <4 x float> %309) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %292, <4 x float> %249) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %295, <4 x float> %311) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %313 = load <4 x half>, ptr addrspace(3) %address.local492, align 8, !dbg !15
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %313, <4 x float> %269) #8, !dbg !15
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !15
  %316 = load <4 x half>, ptr addrspace(3) %address.local521, align 8, !dbg !15
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %315, <4 x float> %314) #8, !dbg !15
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %316, <4 x float> %273) #8, !dbg !15
  %319 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !15
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !15
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %319, <4 x float> %318) #8, !dbg !15
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %320, <4 x float> %277) #8, !dbg !15
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !15
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !15
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %323, <4 x float> %322) #8, !dbg !15
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %324, <4 x float> %281) #8, !dbg !15
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !15
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !15
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %327, <4 x float> %326) #8, !dbg !15
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %328, <4 x float> %285) #8, !dbg !15
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !15
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !15
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %331, <4 x float> %330) #8, !dbg !15
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %332, <4 x float> %289) #8, !dbg !15
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !15
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !15
  %337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %335, <4 x float> %334) #8, !dbg !15
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %336, <4 x float> %293) #8, !dbg !15
  %339 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !15
  %340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !15
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %339, <4 x float> %338) #8, !dbg !15
  %342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %340, <4 x float> %296) #8, !dbg !15
  %343 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !15
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %343, <4 x float> %342) #8, !dbg !15
  %345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %313, <4 x float> %298) #8, !dbg !15
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %315, <4 x float> %345) #8, !dbg !15
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %316, <4 x float> %300) #8, !dbg !15
  %348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %319, <4 x float> %347) #8, !dbg !15
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %320, <4 x float> %302) #8, !dbg !15
  %350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %323, <4 x float> %349) #8, !dbg !15
  %351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %324, <4 x float> %304) #8, !dbg !15
  %352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %327, <4 x float> %351) #8, !dbg !15
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %328, <4 x float> %306) #8, !dbg !15
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %331, <4 x float> %353) #8, !dbg !15
  %355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %332, <4 x float> %308) #8, !dbg !15
  %356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %335, <4 x float> %355) #8, !dbg !15
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %336, <4 x float> %310) #8, !dbg !15
  %358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %339, <4 x float> %357) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %340, <4 x float> %312) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %343, <4 x float> %359) #8, !dbg !15
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local848 = shl nuw nsw i64 %address.local432, 7, !dbg !8
  %address.local849 = and i64 %address.local848, 16256, !dbg !8
  %.idx4445 = or disjoint i64 %address.local849, %address.local439, !dbg !8
  %361 = shl nuw nsw i64 %.idx4445, 1, !dbg !8
  %362 = or disjoint i64 %361, %41, !dbg !8
  %363 = trunc i64 %362 to i32, !dbg !8
  %dma.buffer.offset1641 = or disjoint i32 %363, 192, !dbg !8
  %364 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1661, i32 %dma.buffer.offset1641, <4 x i32> %dma.buffer.rsrc1678) #9, !dbg !8
  %365 = or disjoint i64 %address.local456, %address.local849, !dbg !8
  %366 = shl nuw nsw i64 %365, 1, !dbg !8
  %367 = or disjoint i64 %366, %41, !dbg !8
  %368 = trunc i64 %367 to i32, !dbg !8
  %dma.buffer.offset1638 = or disjoint i32 %368, 192, !dbg !8
  %369 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1658, i32 %dma.buffer.offset1638, <4 x i32> %dma.buffer.rsrc1678) #9, !dbg !8
  %370 = add nuw nsw i64 %133, 3072, !dbg !14
  %371 = lshr i64 %370, 12, !dbg !14
  %372 = add nuw nsw i64 %371, %16, !dbg !14
  %373 = add nsw i64 %150, 96, !dbg !14
  %374 = icmp slt i64 %150, -96, !dbg !14
  %375 = sub nsw i64 -97, %150, !dbg !14
  %376 = select i1 %374, i64 %375, i64 %373, !dbg !14
  %377 = sdiv i64 %376, 128, !dbg !14
  %378 = sext i1 %374 to i64, !dbg !14
  %379 = xor i64 %377, %378, !dbg !14
  %380 = shl i64 %372, 19, !dbg !14
  %381 = and i64 %380, 16252928, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %382 = add i64 %.idx1620, %381, !dbg !14
  %383 = shl nsw i64 %379, 7, !dbg !14
  %384 = sub i64 %382, %383, !dbg !14
  %385 = add i64 %address.local447, %384, !dbg !14
  %.tr53 = trunc i64 %385 to i32, !dbg !14
  %386 = shl i32 %.tr53, 1, !dbg !14
  %dma.buffer.offset1635 = add i32 %386, 192, !dbg !14
  %387 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1655, i32 %dma.buffer.offset1635, <4 x i32> %dma.buffer.rsrc1660) #9, !dbg !14
  %388 = add i64 %address.local464, %384, !dbg !14
  %.tr59 = trunc i64 %388 to i32, !dbg !14
  %389 = shl i32 %.tr59, 1, !dbg !14
  %dma.buffer.offset1632 = add i32 %389, 192, !dbg !14
  %390 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1652, i32 %dma.buffer.offset1632, <4 x i32> %dma.buffer.rsrc1660) #9, !dbg !14
  %391 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %393 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !10
  %394 = fmul <4 x half> %391, splat (half 1.275630e-01), !dbg !10
  %395 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !10
  %396 = fmul <4 x half> %392, splat (half 1.275630e-01), !dbg !10
  %397 = fmul <4 x half> %393, splat (half 1.275630e-01), !dbg !10
  %398 = fmul <4 x half> %395, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %399 = add nuw nsw i64 %137, %16, !dbg !17
  %400 = shl i64 %399, 19, !dbg !17
  %401 = and i64 %400, 16252928, !dbg !17
  %address.local917 = shl nuw nsw i64 %address.local432, 6, !dbg !17
  %address.local918 = and i64 %address.local917, 16128, !dbg !17
  %address.local920 = and i64 %address.local433, 24, !dbg !17
  %.idx6063 = add i64 %401, %143, !dbg !17
  %402 = or disjoint i64 %address.local920, %.idx6063, !dbg !17
  %403 = shl i64 %402, 1, !dbg !17
  %404 = or disjoint i64 %403, %address.local918, !dbg !17
  %dma.buffer.offset1629 = trunc i64 %404 to i32, !dbg !17
  %405 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1661, i32 %dma.buffer.offset1629, <4 x i32> %dma.buffer.rsrc1630) #9, !dbg !17
  %address.local923 = shl nuw nsw i64 %address.local449, 6, !dbg !17
  %address.local924 = and i64 %address.local923, 32512, !dbg !17
  %406 = or disjoint i64 %403, %address.local924, !dbg !17
  %dma.buffer.offset1626 = trunc i64 %406 to i32, !dbg !17
  %407 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1658, i32 %dma.buffer.offset1626, <4 x i32> %dma.buffer.rsrc1630) #9, !dbg !17
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !15
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %408, <4 x float> %317) #8, !dbg !15
  %410 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !15
  %411 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !15
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %396, <4 x half> %410, <4 x float> %409) #8, !dbg !15
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %411, <4 x float> %321) #8, !dbg !15
  %414 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !15
  %415 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !15
  %416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %396, <4 x half> %414, <4 x float> %413) #8, !dbg !15
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %415, <4 x float> %325) #8, !dbg !15
  %418 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !15
  %419 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !15
  %420 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %396, <4 x half> %418, <4 x float> %417) #8, !dbg !15
  %421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %419, <4 x float> %329) #8, !dbg !15
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !15
  %423 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !15
  %424 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %396, <4 x half> %422, <4 x float> %421) #8, !dbg !15
  %425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %423, <4 x float> %333) #8, !dbg !15
  %426 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !15
  %427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !15
  %428 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %396, <4 x half> %426, <4 x float> %425) #8, !dbg !15
  %429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %427, <4 x float> %337) #8, !dbg !15
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !15
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !15
  %432 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %396, <4 x half> %430, <4 x float> %429) #8, !dbg !15
  %433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %431, <4 x float> %341) #8, !dbg !15
  %434 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !15
  %435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !15
  %436 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %396, <4 x half> %434, <4 x float> %433) #8, !dbg !15
  %437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %435, <4 x float> %344) #8, !dbg !15
  %438 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !15
  %439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %396, <4 x half> %438, <4 x float> %437) #8, !dbg !15
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %397, <4 x half> %408, <4 x float> %346) #8, !dbg !15
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %410, <4 x float> %440) #8, !dbg !15
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %397, <4 x half> %411, <4 x float> %348) #8, !dbg !15
  %443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %414, <4 x float> %442) #8, !dbg !15
  %444 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %397, <4 x half> %415, <4 x float> %350) #8, !dbg !15
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %418, <4 x float> %444) #8, !dbg !15
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %397, <4 x half> %419, <4 x float> %352) #8, !dbg !15
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %422, <4 x float> %446) #8, !dbg !15
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %397, <4 x half> %423, <4 x float> %354) #8, !dbg !15
  %449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %426, <4 x float> %448) #8, !dbg !15
  %450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %397, <4 x half> %427, <4 x float> %356) #8, !dbg !15
  %451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %430, <4 x float> %450) #8, !dbg !15
  %452 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %397, <4 x half> %431, <4 x float> %358) #8, !dbg !15
  %453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %434, <4 x float> %452) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %397, <4 x half> %435, <4 x float> %360) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %455 = or disjoint i64 %134, 127, !dbg !12
  %456 = icmp uge i64 %24, %455, !dbg !12
  %457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %438, <4 x float> %454) #8, !dbg !15
  %458 = and i1 %92, %456, !dbg !12
  br i1 %458, label %543, label %459, !dbg !12

459:                                              ; preds = %127
  %460 = or disjoint i64 %134, %118, !dbg !12
  %461 = lshr i64 %133, 25, !dbg !16
  %compare.high1222 = trunc nuw i64 %461 to i32, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1222, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1222, !dbg !16
  %compare.low.base1223 = trunc i64 %460 to i32, !dbg !16
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 1) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1226 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 2) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1230 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 3) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1234 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 4) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1238 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 5) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1242 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 6) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1246 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 7) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1250 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 8) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1254 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 9) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1258 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 10) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1262 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 11) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1266 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 12) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1270 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 13) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1274 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 14) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1278 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 15) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1282 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 16) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1286 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 17) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1290 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 18) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1294 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 19) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1298 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 20) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1302 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 21) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1306 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 22) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1310 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 23) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1314 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 24) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1318 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 25) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1322 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 26) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1326 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 27) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1330 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 28) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1334 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 29) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1338 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 30) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1342 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 31) #8, !dbg !16, !frisk.register_lifetimes !5
  %462 = insertelement <32 x i32> poison, i32 %compare.low.base1223, i64 0, !dbg !16
  %463 = insertelement <32 x i32> %462, i32 %compare.local.base, i64 1, !dbg !16
  %464 = insertelement <32 x i32> %463, i32 %compare.local.base1226, i64 2, !dbg !16
  %465 = insertelement <32 x i32> %464, i32 %compare.local.base1230, i64 3, !dbg !16
  %466 = insertelement <32 x i32> %465, i32 %compare.local.base1234, i64 4, !dbg !16
  %467 = insertelement <32 x i32> %466, i32 %compare.local.base1238, i64 5, !dbg !16
  %468 = insertelement <32 x i32> %467, i32 %compare.local.base1242, i64 6, !dbg !16
  %469 = insertelement <32 x i32> %468, i32 %compare.local.base1246, i64 7, !dbg !16
  %470 = insertelement <32 x i32> %469, i32 %compare.local.base1250, i64 8, !dbg !16
  %471 = insertelement <32 x i32> %470, i32 %compare.local.base1254, i64 9, !dbg !16
  %472 = insertelement <32 x i32> %471, i32 %compare.local.base1258, i64 10, !dbg !16
  %473 = insertelement <32 x i32> %472, i32 %compare.local.base1262, i64 11, !dbg !16
  %474 = insertelement <32 x i32> %473, i32 %compare.local.base1266, i64 12, !dbg !16
  %475 = insertelement <32 x i32> %474, i32 %compare.local.base1270, i64 13, !dbg !16
  %476 = insertelement <32 x i32> %475, i32 %compare.local.base1274, i64 14, !dbg !16
  %477 = insertelement <32 x i32> %476, i32 %compare.local.base1278, i64 15, !dbg !16
  %478 = insertelement <32 x i32> %477, i32 %compare.local.base1282, i64 16, !dbg !16
  %479 = insertelement <32 x i32> %478, i32 %compare.local.base1286, i64 17, !dbg !16
  %480 = insertelement <32 x i32> %479, i32 %compare.local.base1290, i64 18, !dbg !16
  %481 = insertelement <32 x i32> %480, i32 %compare.local.base1294, i64 19, !dbg !16
  %482 = insertelement <32 x i32> %481, i32 %compare.local.base1298, i64 20, !dbg !16
  %483 = insertelement <32 x i32> %482, i32 %compare.local.base1302, i64 21, !dbg !16
  %484 = insertelement <32 x i32> %483, i32 %compare.local.base1306, i64 22, !dbg !16
  %485 = insertelement <32 x i32> %484, i32 %compare.local.base1310, i64 23, !dbg !16
  %486 = insertelement <32 x i32> %485, i32 %compare.local.base1314, i64 24, !dbg !16
  %487 = insertelement <32 x i32> %486, i32 %compare.local.base1318, i64 25, !dbg !16
  %488 = insertelement <32 x i32> %487, i32 %compare.local.base1322, i64 26, !dbg !16
  %489 = insertelement <32 x i32> %488, i32 %compare.local.base1326, i64 27, !dbg !16
  %490 = insertelement <32 x i32> %489, i32 %compare.local.base1330, i64 28, !dbg !16
  %491 = insertelement <32 x i32> %490, i32 %compare.local.base1334, i64 29, !dbg !16
  %492 = insertelement <32 x i32> %491, i32 %compare.local.base1338, i64 30, !dbg !16
  %493 = insertelement <32 x i32> %492, i32 %compare.local.base1342, i64 31, !dbg !16
  %494 = or <32 x i32> %493, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %495 = icmp ugt <32 x i32> %126, %494, !dbg !16
  %496 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %497 = shufflevector <32 x i1> %496, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %498 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %499 = shufflevector <32 x i1> %498, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %500 = select <32 x i1> %497, <32 x i1> %495, <32 x i1> %499, !dbg !16
  %501 = select <32 x i1> %500, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %compare.high.equal1348 = icmp eq i32 %compare.high1346, %compare.high1222, !dbg !16
  %compare.high.order1349 = icmp ugt i32 %compare.high1346, %compare.high1222, !dbg !16
  %compare.local.base1354 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 32) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1358 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 33) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1362 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 34) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1366 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 35) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1370 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 36) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1374 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 37) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1378 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 38) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1382 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 39) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1386 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 40) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1390 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 41) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1394 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 42) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1398 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 43) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1402 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 44) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1406 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 45) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1410 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 46) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1414 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 47) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1418 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 48) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1422 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 49) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1426 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 50) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1430 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 51) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1434 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 52) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1438 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 53) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1442 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 54) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1446 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 55) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1450 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 56) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1454 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 57) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1458 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 58) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1462 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 59) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1466 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 60) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1470 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 61) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1474 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1223, i32 62) #8, !dbg !16, !frisk.register_lifetimes !5
  %502 = insertelement <32 x i32> %462, i32 %compare.local.base1354, i64 1, !dbg !16
  %503 = insertelement <32 x i32> %502, i32 %compare.local.base1358, i64 2, !dbg !16
  %504 = insertelement <32 x i32> %503, i32 %compare.local.base1362, i64 3, !dbg !16
  %505 = insertelement <32 x i32> %504, i32 %compare.local.base1366, i64 4, !dbg !16
  %506 = insertelement <32 x i32> %505, i32 %compare.local.base1370, i64 5, !dbg !16
  %507 = insertelement <32 x i32> %506, i32 %compare.local.base1374, i64 6, !dbg !16
  %508 = insertelement <32 x i32> %507, i32 %compare.local.base1378, i64 7, !dbg !16
  %509 = insertelement <32 x i32> %508, i32 %compare.local.base1382, i64 8, !dbg !16
  %510 = insertelement <32 x i32> %509, i32 %compare.local.base1386, i64 9, !dbg !16
  %511 = insertelement <32 x i32> %510, i32 %compare.local.base1390, i64 10, !dbg !16
  %512 = insertelement <32 x i32> %511, i32 %compare.local.base1394, i64 11, !dbg !16
  %513 = insertelement <32 x i32> %512, i32 %compare.local.base1398, i64 12, !dbg !16
  %514 = insertelement <32 x i32> %513, i32 %compare.local.base1402, i64 13, !dbg !16
  %515 = insertelement <32 x i32> %514, i32 %compare.local.base1406, i64 14, !dbg !16
  %516 = insertelement <32 x i32> %515, i32 %compare.local.base1410, i64 15, !dbg !16
  %517 = insertelement <32 x i32> %516, i32 %compare.local.base1414, i64 16, !dbg !16
  %518 = insertelement <32 x i32> %517, i32 %compare.local.base1418, i64 17, !dbg !16
  %519 = insertelement <32 x i32> %518, i32 %compare.local.base1422, i64 18, !dbg !16
  %520 = insertelement <32 x i32> %519, i32 %compare.local.base1426, i64 19, !dbg !16
  %521 = insertelement <32 x i32> %520, i32 %compare.local.base1430, i64 20, !dbg !16
  %522 = insertelement <32 x i32> %521, i32 %compare.local.base1434, i64 21, !dbg !16
  %523 = insertelement <32 x i32> %522, i32 %compare.local.base1438, i64 22, !dbg !16
  %524 = insertelement <32 x i32> %523, i32 %compare.local.base1442, i64 23, !dbg !16
  %525 = insertelement <32 x i32> %524, i32 %compare.local.base1446, i64 24, !dbg !16
  %526 = insertelement <32 x i32> %525, i32 %compare.local.base1450, i64 25, !dbg !16
  %527 = insertelement <32 x i32> %526, i32 %compare.local.base1454, i64 26, !dbg !16
  %528 = insertelement <32 x i32> %527, i32 %compare.local.base1458, i64 27, !dbg !16
  %529 = insertelement <32 x i32> %528, i32 %compare.local.base1462, i64 28, !dbg !16
  %530 = insertelement <32 x i32> %529, i32 %compare.local.base1466, i64 29, !dbg !16
  %531 = insertelement <32 x i32> %530, i32 %compare.local.base1470, i64 30, !dbg !16
  %532 = insertelement <32 x i32> %531, i32 %compare.local.base1474, i64 31, !dbg !16
  %533 = or <32 x i32> %532, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %534 = icmp ugt <32 x i32> %124, %533, !dbg !16
  %535 = insertelement <32 x i1> poison, i1 %compare.high.equal1348, i64 0, !dbg !16
  %536 = shufflevector <32 x i1> %535, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %537 = insertelement <32 x i1> poison, i1 %compare.high.order1349, i64 0, !dbg !16
  %538 = shufflevector <32 x i1> %537, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %539 = select <32 x i1> %536, <32 x i1> %534, <32 x i1> %538, !dbg !16
  %540 = select <32 x i1> %539, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %541 = insertvalue [2 x <32 x float>] poison, <32 x float> %501, 0, !dbg !12
  %542 = insertvalue [2 x <32 x float>] %541, <32 x float> %540, 1, !dbg !12
  br label %543, !dbg !12

543:                                              ; preds = %459, %127
  %544 = phi [2 x <32 x float>] [ %542, %459 ], [ zeroinitializer, %127 ], !dbg !12
  %545 = extractvalue [2 x <32 x float>] %544, 0, !dbg !19
  %546 = shufflevector <32 x float> %545, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %547 = fadd <4 x float> %412, %546, !dbg !19
  %548 = extractelement <4 x float> %547, i64 0, !dbg !20
  %549 = tail call float @llvm.amdgcn.exp2.f32(float %548), !dbg !20
  %550 = insertelement <4 x float> poison, float %549, i64 0, !dbg !21
  %551 = extractelement <4 x float> %547, i64 1, !dbg !20
  %552 = tail call float @llvm.amdgcn.exp2.f32(float %551), !dbg !20
  %553 = insertelement <4 x float> %550, float %552, i64 1, !dbg !21
  %554 = fadd float %549, %552, !dbg !22
  %555 = extractelement <4 x float> %547, i64 2, !dbg !20
  %556 = tail call float @llvm.amdgcn.exp2.f32(float %555), !dbg !20
  %557 = insertelement <4 x float> %553, float %556, i64 2, !dbg !21
  %558 = fadd float %556, %554, !dbg !22
  %559 = extractelement <4 x float> %547, i64 3, !dbg !20
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %559), !dbg !20
  %561 = insertelement <4 x float> %557, float %560, i64 3, !dbg !21
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %561) #8, !dbg !21, !frisk.register_lifetimes !5
  %562 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !21
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %562) #8, !dbg !22, !frisk.register_lifetimes !5
  %563 = fadd float %560, %558, !dbg !22
  %564 = shufflevector <32 x float> %545, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %565 = fadd <4 x float> %416, %564, !dbg !19
  %566 = extractelement <4 x float> %565, i64 0, !dbg !20
  %567 = tail call float @llvm.amdgcn.exp2.f32(float %566), !dbg !20
  %568 = insertelement <4 x float> poison, float %567, i64 0, !dbg !21
  %569 = fadd float %567, %563, !dbg !22
  %570 = extractelement <4 x float> %565, i64 1, !dbg !20
  %571 = tail call float @llvm.amdgcn.exp2.f32(float %570), !dbg !20
  %572 = insertelement <4 x float> %568, float %571, i64 1, !dbg !21
  %573 = fadd float %571, %569, !dbg !22
  %574 = extractelement <4 x float> %565, i64 2, !dbg !20
  %575 = tail call float @llvm.amdgcn.exp2.f32(float %574), !dbg !20
  %576 = insertelement <4 x float> %572, float %575, i64 2, !dbg !21
  %577 = fadd float %575, %573, !dbg !22
  %578 = extractelement <4 x float> %565, i64 3, !dbg !20
  %579 = tail call float @llvm.amdgcn.exp2.f32(float %578), !dbg !20
  %580 = insertelement <4 x float> %576, float %579, i64 3, !dbg !21
  %math.result309 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %580) #8, !dbg !21, !frisk.register_lifetimes !5
  %581 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !21
  %math.packed310 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %581) #8, !dbg !22, !frisk.register_lifetimes !5
  %582 = fadd float %579, %577, !dbg !22
  %583 = shufflevector <32 x float> %545, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %584 = fadd <4 x float> %420, %583, !dbg !19
  %585 = extractelement <4 x float> %584, i64 0, !dbg !20
  %586 = tail call float @llvm.amdgcn.exp2.f32(float %585), !dbg !20
  %587 = insertelement <4 x float> poison, float %586, i64 0, !dbg !21
  %588 = fadd float %586, %582, !dbg !22
  %589 = extractelement <4 x float> %584, i64 1, !dbg !20
  %590 = tail call float @llvm.amdgcn.exp2.f32(float %589), !dbg !20
  %591 = insertelement <4 x float> %587, float %590, i64 1, !dbg !21
  %592 = fadd float %590, %588, !dbg !22
  %593 = extractelement <4 x float> %584, i64 2, !dbg !20
  %594 = tail call float @llvm.amdgcn.exp2.f32(float %593), !dbg !20
  %595 = insertelement <4 x float> %591, float %594, i64 2, !dbg !21
  %596 = fadd float %594, %592, !dbg !22
  %597 = extractelement <4 x float> %584, i64 3, !dbg !20
  %598 = tail call float @llvm.amdgcn.exp2.f32(float %597), !dbg !20
  %599 = insertelement <4 x float> %595, float %598, i64 3, !dbg !21
  %math.result311 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %599) #8, !dbg !21, !frisk.register_lifetimes !5
  %600 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !21
  %math.packed312 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %600) #8, !dbg !22, !frisk.register_lifetimes !5
  %601 = fadd float %598, %596, !dbg !22
  %602 = shufflevector <32 x float> %545, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %603 = fadd <4 x float> %424, %602, !dbg !19
  %604 = extractelement <4 x float> %603, i64 0, !dbg !20
  %605 = tail call float @llvm.amdgcn.exp2.f32(float %604), !dbg !20
  %606 = insertelement <4 x float> poison, float %605, i64 0, !dbg !21
  %607 = fadd float %605, %601, !dbg !22
  %608 = extractelement <4 x float> %603, i64 1, !dbg !20
  %609 = tail call float @llvm.amdgcn.exp2.f32(float %608), !dbg !20
  %610 = insertelement <4 x float> %606, float %609, i64 1, !dbg !21
  %611 = fadd float %609, %607, !dbg !22
  %612 = extractelement <4 x float> %603, i64 2, !dbg !20
  %613 = tail call float @llvm.amdgcn.exp2.f32(float %612), !dbg !20
  %614 = insertelement <4 x float> %610, float %613, i64 2, !dbg !21
  %615 = fadd float %613, %611, !dbg !22
  %616 = extractelement <4 x float> %603, i64 3, !dbg !20
  %617 = tail call float @llvm.amdgcn.exp2.f32(float %616), !dbg !20
  %618 = insertelement <4 x float> %614, float %617, i64 3, !dbg !21
  %math.result313 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %618) #8, !dbg !21, !frisk.register_lifetimes !5
  %619 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !21
  %math.packed314 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %619) #8, !dbg !22, !frisk.register_lifetimes !5
  %620 = fadd float %617, %615, !dbg !22
  %621 = shufflevector <32 x float> %545, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %622 = fadd <4 x float> %428, %621, !dbg !19
  %623 = extractelement <4 x float> %622, i64 0, !dbg !20
  %624 = tail call float @llvm.amdgcn.exp2.f32(float %623), !dbg !20
  %625 = insertelement <4 x float> poison, float %624, i64 0, !dbg !21
  %626 = fadd float %624, %620, !dbg !22
  %627 = extractelement <4 x float> %622, i64 1, !dbg !20
  %628 = tail call float @llvm.amdgcn.exp2.f32(float %627), !dbg !20
  %629 = insertelement <4 x float> %625, float %628, i64 1, !dbg !21
  %630 = fadd float %628, %626, !dbg !22
  %631 = extractelement <4 x float> %622, i64 2, !dbg !20
  %632 = tail call float @llvm.amdgcn.exp2.f32(float %631), !dbg !20
  %633 = insertelement <4 x float> %629, float %632, i64 2, !dbg !21
  %634 = fadd float %632, %630, !dbg !22
  %635 = extractelement <4 x float> %622, i64 3, !dbg !20
  %636 = tail call float @llvm.amdgcn.exp2.f32(float %635), !dbg !20
  %637 = insertelement <4 x float> %633, float %636, i64 3, !dbg !21
  %math.result315 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %637) #8, !dbg !21, !frisk.register_lifetimes !5
  %638 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !21
  %math.packed316 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %638) #8, !dbg !22, !frisk.register_lifetimes !5
  %639 = fadd float %636, %634, !dbg !22
  %640 = shufflevector <32 x float> %545, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %641 = fadd <4 x float> %432, %640, !dbg !19
  %642 = extractelement <4 x float> %641, i64 0, !dbg !20
  %643 = tail call float @llvm.amdgcn.exp2.f32(float %642), !dbg !20
  %644 = insertelement <4 x float> poison, float %643, i64 0, !dbg !21
  %645 = fadd float %643, %639, !dbg !22
  %646 = extractelement <4 x float> %641, i64 1, !dbg !20
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %646), !dbg !20
  %648 = insertelement <4 x float> %644, float %647, i64 1, !dbg !21
  %649 = fadd float %647, %645, !dbg !22
  %650 = extractelement <4 x float> %641, i64 2, !dbg !20
  %651 = tail call float @llvm.amdgcn.exp2.f32(float %650), !dbg !20
  %652 = insertelement <4 x float> %648, float %651, i64 2, !dbg !21
  %653 = fadd float %651, %649, !dbg !22
  %654 = extractelement <4 x float> %641, i64 3, !dbg !20
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %654), !dbg !20
  %656 = insertelement <4 x float> %652, float %655, i64 3, !dbg !21
  %math.result317 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %656) #8, !dbg !21, !frisk.register_lifetimes !5
  %657 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !21
  %math.packed318 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %657) #8, !dbg !22, !frisk.register_lifetimes !5
  %658 = fadd float %655, %653, !dbg !22
  %659 = shufflevector <32 x float> %545, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %660 = fadd <4 x float> %436, %659, !dbg !19
  %661 = extractelement <4 x float> %660, i64 0, !dbg !20
  %662 = tail call float @llvm.amdgcn.exp2.f32(float %661), !dbg !20
  %663 = insertelement <4 x float> poison, float %662, i64 0, !dbg !21
  %664 = fadd float %662, %658, !dbg !22
  %665 = extractelement <4 x float> %660, i64 1, !dbg !20
  %666 = tail call float @llvm.amdgcn.exp2.f32(float %665), !dbg !20
  %667 = insertelement <4 x float> %663, float %666, i64 1, !dbg !21
  %668 = fadd float %666, %664, !dbg !22
  %669 = extractelement <4 x float> %660, i64 2, !dbg !20
  %670 = tail call float @llvm.amdgcn.exp2.f32(float %669), !dbg !20
  %671 = insertelement <4 x float> %667, float %670, i64 2, !dbg !21
  %672 = fadd float %670, %668, !dbg !22
  %673 = extractelement <4 x float> %660, i64 3, !dbg !20
  %674 = tail call float @llvm.amdgcn.exp2.f32(float %673), !dbg !20
  %675 = insertelement <4 x float> %671, float %674, i64 3, !dbg !21
  %math.result319 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %675) #8, !dbg !21, !frisk.register_lifetimes !5
  %676 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !21
  %math.packed320 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %676) #8, !dbg !22, !frisk.register_lifetimes !5
  %677 = fadd float %674, %672, !dbg !22
  %678 = shufflevector <32 x float> %545, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %679 = fadd <4 x float> %439, %678, !dbg !19
  %680 = extractelement <4 x float> %679, i64 0, !dbg !20
  %681 = tail call float @llvm.amdgcn.exp2.f32(float %680), !dbg !20
  %682 = insertelement <4 x float> poison, float %681, i64 0, !dbg !21
  %683 = fadd float %681, %677, !dbg !22
  %684 = extractelement <4 x float> %679, i64 1, !dbg !20
  %685 = tail call float @llvm.amdgcn.exp2.f32(float %684), !dbg !20
  %686 = insertelement <4 x float> %682, float %685, i64 1, !dbg !21
  %687 = fadd float %685, %683, !dbg !22
  %688 = extractelement <4 x float> %679, i64 2, !dbg !20
  %689 = tail call float @llvm.amdgcn.exp2.f32(float %688), !dbg !20
  %690 = insertelement <4 x float> %686, float %689, i64 2, !dbg !21
  %691 = fadd float %689, %687, !dbg !22
  %692 = extractelement <4 x float> %679, i64 3, !dbg !20
  %693 = tail call float @llvm.amdgcn.exp2.f32(float %692), !dbg !20
  %694 = insertelement <4 x float> %690, float %693, i64 3, !dbg !21
  %math.result321 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %694) #8, !dbg !21, !frisk.register_lifetimes !5
  %695 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !21
  %math.packed322 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %695) #8, !dbg !22, !frisk.register_lifetimes !5
  %696 = fadd float %693, %691, !dbg !22
  %697 = extractvalue [2 x <32 x float>] %544, 1, !dbg !19
  %698 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %699 = fadd <4 x float> %441, %698, !dbg !19
  %700 = extractelement <4 x float> %699, i64 0, !dbg !20
  %701 = tail call float @llvm.amdgcn.exp2.f32(float %700), !dbg !20
  %702 = insertelement <4 x float> poison, float %701, i64 0, !dbg !21
  %703 = extractelement <4 x float> %699, i64 1, !dbg !20
  %704 = tail call float @llvm.amdgcn.exp2.f32(float %703), !dbg !20
  %705 = insertelement <4 x float> %702, float %704, i64 1, !dbg !21
  %706 = fadd float %701, %704, !dbg !22
  %707 = extractelement <4 x float> %699, i64 2, !dbg !20
  %708 = tail call float @llvm.amdgcn.exp2.f32(float %707), !dbg !20
  %709 = insertelement <4 x float> %705, float %708, i64 2, !dbg !21
  %710 = fadd float %708, %706, !dbg !22
  %711 = extractelement <4 x float> %699, i64 3, !dbg !20
  %712 = tail call float @llvm.amdgcn.exp2.f32(float %711), !dbg !20
  %713 = insertelement <4 x float> %709, float %712, i64 3, !dbg !21
  %math.result323 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %713) #8, !dbg !21, !frisk.register_lifetimes !5
  %714 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !21
  %math.packed324 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %714) #8, !dbg !22, !frisk.register_lifetimes !5
  %715 = fadd float %712, %710, !dbg !22
  %716 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %717 = fadd <4 x float> %443, %716, !dbg !19
  %718 = extractelement <4 x float> %717, i64 0, !dbg !20
  %719 = tail call float @llvm.amdgcn.exp2.f32(float %718), !dbg !20
  %720 = insertelement <4 x float> poison, float %719, i64 0, !dbg !21
  %721 = fadd float %719, %715, !dbg !22
  %722 = extractelement <4 x float> %717, i64 1, !dbg !20
  %723 = tail call float @llvm.amdgcn.exp2.f32(float %722), !dbg !20
  %724 = insertelement <4 x float> %720, float %723, i64 1, !dbg !21
  %725 = fadd float %723, %721, !dbg !22
  %726 = extractelement <4 x float> %717, i64 2, !dbg !20
  %727 = tail call float @llvm.amdgcn.exp2.f32(float %726), !dbg !20
  %728 = insertelement <4 x float> %724, float %727, i64 2, !dbg !21
  %729 = fadd float %727, %725, !dbg !22
  %730 = extractelement <4 x float> %717, i64 3, !dbg !20
  %731 = tail call float @llvm.amdgcn.exp2.f32(float %730), !dbg !20
  %732 = insertelement <4 x float> %728, float %731, i64 3, !dbg !21
  %math.result325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %732) #8, !dbg !21, !frisk.register_lifetimes !5
  %733 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !21
  %math.packed326 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %733) #8, !dbg !22, !frisk.register_lifetimes !5
  %734 = fadd float %731, %729, !dbg !22
  %735 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %736 = fadd <4 x float> %445, %735, !dbg !19
  %737 = extractelement <4 x float> %736, i64 0, !dbg !20
  %738 = tail call float @llvm.amdgcn.exp2.f32(float %737), !dbg !20
  %739 = insertelement <4 x float> poison, float %738, i64 0, !dbg !21
  %740 = fadd float %738, %734, !dbg !22
  %741 = extractelement <4 x float> %736, i64 1, !dbg !20
  %742 = tail call float @llvm.amdgcn.exp2.f32(float %741), !dbg !20
  %743 = insertelement <4 x float> %739, float %742, i64 1, !dbg !21
  %744 = fadd float %742, %740, !dbg !22
  %745 = extractelement <4 x float> %736, i64 2, !dbg !20
  %746 = tail call float @llvm.amdgcn.exp2.f32(float %745), !dbg !20
  %747 = insertelement <4 x float> %743, float %746, i64 2, !dbg !21
  %748 = fadd float %746, %744, !dbg !22
  %749 = extractelement <4 x float> %736, i64 3, !dbg !20
  %750 = tail call float @llvm.amdgcn.exp2.f32(float %749), !dbg !20
  %751 = insertelement <4 x float> %747, float %750, i64 3, !dbg !21
  %math.result327 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %751) #8, !dbg !21, !frisk.register_lifetimes !5
  %752 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !21
  %math.packed328 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %752) #8, !dbg !22, !frisk.register_lifetimes !5
  %753 = fadd float %750, %748, !dbg !22
  %754 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %755 = fadd <4 x float> %447, %754, !dbg !19
  %756 = extractelement <4 x float> %755, i64 0, !dbg !20
  %757 = tail call float @llvm.amdgcn.exp2.f32(float %756), !dbg !20
  %758 = insertelement <4 x float> poison, float %757, i64 0, !dbg !21
  %759 = fadd float %757, %753, !dbg !22
  %760 = extractelement <4 x float> %755, i64 1, !dbg !20
  %761 = tail call float @llvm.amdgcn.exp2.f32(float %760), !dbg !20
  %762 = insertelement <4 x float> %758, float %761, i64 1, !dbg !21
  %763 = fadd float %761, %759, !dbg !22
  %764 = extractelement <4 x float> %755, i64 2, !dbg !20
  %765 = tail call float @llvm.amdgcn.exp2.f32(float %764), !dbg !20
  %766 = insertelement <4 x float> %762, float %765, i64 2, !dbg !21
  %767 = fadd float %765, %763, !dbg !22
  %768 = extractelement <4 x float> %755, i64 3, !dbg !20
  %769 = tail call float @llvm.amdgcn.exp2.f32(float %768), !dbg !20
  %770 = insertelement <4 x float> %766, float %769, i64 3, !dbg !21
  %math.result329 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %770) #8, !dbg !21, !frisk.register_lifetimes !5
  %771 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !21
  %math.packed330 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %771) #8, !dbg !22, !frisk.register_lifetimes !5
  %772 = fadd float %769, %767, !dbg !22
  %773 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %774 = fadd <4 x float> %449, %773, !dbg !19
  %775 = extractelement <4 x float> %774, i64 0, !dbg !20
  %776 = tail call float @llvm.amdgcn.exp2.f32(float %775), !dbg !20
  %777 = insertelement <4 x float> poison, float %776, i64 0, !dbg !21
  %778 = fadd float %776, %772, !dbg !22
  %779 = extractelement <4 x float> %774, i64 1, !dbg !20
  %780 = tail call float @llvm.amdgcn.exp2.f32(float %779), !dbg !20
  %781 = insertelement <4 x float> %777, float %780, i64 1, !dbg !21
  %782 = fadd float %780, %778, !dbg !22
  %783 = extractelement <4 x float> %774, i64 2, !dbg !20
  %784 = tail call float @llvm.amdgcn.exp2.f32(float %783), !dbg !20
  %785 = insertelement <4 x float> %781, float %784, i64 2, !dbg !21
  %786 = fadd float %784, %782, !dbg !22
  %787 = extractelement <4 x float> %774, i64 3, !dbg !20
  %788 = tail call float @llvm.amdgcn.exp2.f32(float %787), !dbg !20
  %789 = insertelement <4 x float> %785, float %788, i64 3, !dbg !21
  %math.result331 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %789) #8, !dbg !21, !frisk.register_lifetimes !5
  %790 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !21
  %math.packed332 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %790) #8, !dbg !22, !frisk.register_lifetimes !5
  %791 = fadd float %788, %786, !dbg !22
  %792 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %793 = fadd <4 x float> %451, %792, !dbg !19
  %794 = extractelement <4 x float> %793, i64 0, !dbg !20
  %795 = tail call float @llvm.amdgcn.exp2.f32(float %794), !dbg !20
  %796 = insertelement <4 x float> poison, float %795, i64 0, !dbg !21
  %797 = fadd float %795, %791, !dbg !22
  %798 = extractelement <4 x float> %793, i64 1, !dbg !20
  %799 = tail call float @llvm.amdgcn.exp2.f32(float %798), !dbg !20
  %800 = insertelement <4 x float> %796, float %799, i64 1, !dbg !21
  %801 = fadd float %799, %797, !dbg !22
  %802 = extractelement <4 x float> %793, i64 2, !dbg !20
  %803 = tail call float @llvm.amdgcn.exp2.f32(float %802), !dbg !20
  %804 = insertelement <4 x float> %800, float %803, i64 2, !dbg !21
  %805 = fadd float %803, %801, !dbg !22
  %806 = extractelement <4 x float> %793, i64 3, !dbg !20
  %807 = tail call float @llvm.amdgcn.exp2.f32(float %806), !dbg !20
  %808 = insertelement <4 x float> %804, float %807, i64 3, !dbg !21
  %math.result333 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %808) #8, !dbg !21, !frisk.register_lifetimes !5
  %809 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !21
  %math.packed334 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %809) #8, !dbg !22, !frisk.register_lifetimes !5
  %810 = fadd float %807, %805, !dbg !22
  %811 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %812 = fadd <4 x float> %453, %811, !dbg !19
  %813 = extractelement <4 x float> %812, i64 0, !dbg !20
  %814 = tail call float @llvm.amdgcn.exp2.f32(float %813), !dbg !20
  %815 = insertelement <4 x float> poison, float %814, i64 0, !dbg !21
  %816 = fadd float %814, %810, !dbg !22
  %817 = extractelement <4 x float> %812, i64 1, !dbg !20
  %818 = tail call float @llvm.amdgcn.exp2.f32(float %817), !dbg !20
  %819 = insertelement <4 x float> %815, float %818, i64 1, !dbg !21
  %820 = fadd float %818, %816, !dbg !22
  %821 = extractelement <4 x float> %812, i64 2, !dbg !20
  %822 = tail call float @llvm.amdgcn.exp2.f32(float %821), !dbg !20
  %823 = insertelement <4 x float> %819, float %822, i64 2, !dbg !21
  %824 = fadd float %822, %820, !dbg !22
  %825 = extractelement <4 x float> %812, i64 3, !dbg !20
  %826 = tail call float @llvm.amdgcn.exp2.f32(float %825), !dbg !20
  %827 = insertelement <4 x float> %823, float %826, i64 3, !dbg !21
  %math.result335 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %827) #8, !dbg !21, !frisk.register_lifetimes !5
  %828 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !21
  %math.packed336 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %828) #8, !dbg !22, !frisk.register_lifetimes !5
  %829 = fadd float %826, %824, !dbg !22
  %830 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %831 = fadd <4 x float> %457, %830, !dbg !19
  %832 = extractelement <4 x float> %831, i64 0, !dbg !20
  %833 = tail call float @llvm.amdgcn.exp2.f32(float %832), !dbg !20
  %834 = insertelement <4 x float> poison, float %833, i64 0, !dbg !21
  %835 = fadd float %833, %829, !dbg !22
  %836 = extractelement <4 x float> %831, i64 1, !dbg !20
  %837 = tail call float @llvm.amdgcn.exp2.f32(float %836), !dbg !20
  %838 = insertelement <4 x float> %834, float %837, i64 1, !dbg !21
  %839 = fadd float %837, %835, !dbg !22
  %840 = extractelement <4 x float> %831, i64 2, !dbg !20
  %841 = tail call float @llvm.amdgcn.exp2.f32(float %840), !dbg !20
  %842 = insertelement <4 x float> %838, float %841, i64 2, !dbg !21
  %843 = fadd float %841, %839, !dbg !22
  %844 = extractelement <4 x float> %831, i64 3, !dbg !20
  %845 = tail call float @llvm.amdgcn.exp2.f32(float %844), !dbg !20
  %846 = insertelement <4 x float> %842, float %845, i64 3, !dbg !21
  %math.result337 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %846) #8, !dbg !21, !frisk.register_lifetimes !5
  %847 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !21
  %math.packed338 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %847) #8, !dbg !22, !frisk.register_lifetimes !5
  %848 = fadd float %845, %843, !dbg !22
  %849 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !22
  %850 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %849), !dbg !22
  %851 = add nuw nsw i32 %850, 64, !dbg !22
  %852 = xor i32 %850, 16, !dbg !22
  %853 = and i32 %851, 192, !dbg !22
  %854 = icmp samesign ult i32 %852, %853, !dbg !22
  %855 = select i1 %854, i32 %852, i32 %850, !dbg !22
  %856 = shl nuw nsw i32 %855, 2, !dbg !22
  %857 = bitcast float %696 to i32, !dbg !22
  %858 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %856, i32 %857), !dbg !22
  %859 = bitcast i32 %858 to float, !dbg !22
  %860 = fadd float %696, %859, !dbg !22
  %861 = xor i32 %850, 32, !dbg !22
  %862 = icmp samesign ult i32 %861, %853, !dbg !22
  %863 = select i1 %862, i32 %861, i32 %850, !dbg !22
  %864 = shl nuw nsw i32 %863, 2, !dbg !22
  %865 = bitcast float %860 to i32, !dbg !22
  %866 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %864, i32 %865), !dbg !22
  %867 = bitcast i32 %866 to float, !dbg !22
  %868 = fadd float %860, %867, !dbg !22
  %869 = extractvalue [2 x <1 x float>] %131, 0, !dbg !22
  %870 = extractelement <1 x float> %869, i64 0, !dbg !22
  %871 = fadd float %870, %868, !dbg !22
  %872 = insertelement <1 x float> poison, float %871, i64 0, !dbg !11
  %873 = insertvalue [2 x <1 x float>] poison, <1 x float> %872, 0, !dbg !11
  %874 = bitcast float %848 to i32, !dbg !22
  %875 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %856, i32 %874), !dbg !22
  %876 = bitcast i32 %875 to float, !dbg !22
  %877 = fadd float %848, %876, !dbg !22
  %878 = bitcast float %877 to i32, !dbg !22
  %879 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %864, i32 %878), !dbg !22
  %880 = bitcast i32 %879 to float, !dbg !22
  %881 = fadd float %877, %880, !dbg !22
  %882 = extractvalue [2 x <1 x float>] %131, 1, !dbg !22
  %883 = extractelement <1 x float> %882, i64 0, !dbg !22
  %884 = fadd float %883, %881, !dbg !22
  %885 = insertelement <1 x float> poison, float %884, i64 0, !dbg !11
  %886 = insertvalue [2 x <1 x float>] %873, <1 x float> %885, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread929 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !23, !range !9, !frisk.register_lifetimes !5
  %address.local930 = shl nuw nsw i32 %address.thread929, 3, !dbg !23
  %address.local931 = and i32 %address.local930, 120, !dbg !23
  %address.local932 = shl nuw nsw i32 %address.thread929, 5, !dbg !23
  %address.local933 = and i32 %address.local932, 1536, !dbg !23
  %address.local934 = or disjoint i32 %address.local931, %address.local933, !dbg !23
  %address.local935 = zext nneg i32 %address.local934 to i64, !dbg !23
  %address.local936 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local935, !dbg !23
  %887 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local936, i32 0) #11, !dbg !23
  %888 = bitcast <4 x i32> %887 to <8 x half>, !dbg !23
  %address.local937 = zext nneg i32 %address.thread929 to i64, !dbg !8
  %address.local938 = and i64 %address.local937, 60, !dbg !23
  %address.local939 = add nuw nsw i64 %address.local938, %address.local937, !dbg !23
  %address.local941.idx = shl nuw nsw i64 %address.local939, 4, !dbg !23
  %address.local941 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local941.idx, !dbg !23
  %address.local942 = lshr i64 %address.local937, 4, !dbg !23
  %address.local943 = and i64 %address.local942, 3, !dbg !23
  %address.local945.idx = mul nuw nsw i64 %address.local943, 768, !dbg !23
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local941, i64 %address.local945.idx, !dbg !23
  %address.local946 = lshr i64 %address.local937, 2, !dbg !15
  %address.local948.idx = mul nsw i64 %address.local946, -64, !dbg !23
  %address.local948 = getelementptr i8, ptr addrspace(3) %address.local945, i64 %address.local948.idx, !dbg !23
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local948, i64 256, !dbg !23
  %889 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !23
  %890 = bitcast <4 x i32> %889 to <8 x half>, !dbg !23
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local948, i64 512, !dbg !23
  %891 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !23
  %892 = bitcast <4 x i32> %891 to <8 x half>, !dbg !23
  %address.local957 = getelementptr i8, ptr addrspace(3) %address.local948, i64 768, !dbg !23
  %893 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local957, i32 0) #11, !dbg !23
  %894 = bitcast <4 x i32> %893 to <8 x half>, !dbg !23
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local936, i64 4096, !dbg !23
  %895 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !23
  %896 = bitcast <4 x i32> %895 to <8 x half>, !dbg !23
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local948, i64 4352, !dbg !23
  %897 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !23
  %898 = bitcast <4 x i32> %897 to <8 x half>, !dbg !23
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local948, i64 4608, !dbg !23
  %899 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !23
  %900 = bitcast <4 x i32> %899 to <8 x half>, !dbg !23
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local948, i64 4864, !dbg !23
  %901 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #11, !dbg !23
  %902 = bitcast <4 x i32> %901 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local972 = shl nuw nsw i64 %address.local937, 3, !dbg !8
  %address.local973 = and i64 %address.local972, 1536, !dbg !8
  %address.local974 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local973, !dbg !8
  %address.local975 = ptrtoint ptr addrspace(3) %address.local974 to i64, !dbg !8
  %address.local976 = trunc i64 %address.local975 to i32, !dbg !8
  %903 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local976), !dbg !17
  %address.local977 = shl nuw nsw i64 %address.local937, 5, !dbg !17
  %address.local978 = and i64 %address.local977, 8064, !dbg !17
  %address.local980 = and i64 %address.local972, 24, !dbg !17
  %address.local981 = or disjoint i64 %address.local978, %address.local980, !dbg !17
  %904 = or disjoint i64 %.idx6063, %address.local981, !dbg !17
  %905 = trunc i64 %904 to i32, !dbg !17
  %.tr71 = shl i32 %905, 1, !dbg !17
  %dma.buffer.offset1623 = or i32 %.tr71, 64, !dbg !17
  %dma.buffer.lds1625 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %903), !dbg !17
  %906 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1625, i32 %dma.buffer.offset1623, <4 x i32> %dma.buffer.rsrc1630) #9, !dbg !17
  %address.local983 = or disjoint i64 %address.local937, 256, !dbg !8
  %address.local984 = shl nuw nsw i64 %address.local983, 3, !dbg !8
  %address.local985 = and i64 %address.local984, 3584, !dbg !8
  %address.local986 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local985, !dbg !8
  %address.local987 = ptrtoint ptr addrspace(3) %address.local986 to i64, !dbg !8
  %address.local988 = trunc i64 %address.local987 to i32, !dbg !8
  %907 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local988), !dbg !17
  %address.local989 = shl nuw nsw i64 %address.local983, 5, !dbg !17
  %address.local990 = and i64 %address.local989, 16256, !dbg !17
  %address.local993 = or disjoint i64 %address.local990, %address.local980, !dbg !17
  %908 = or disjoint i64 %.idx6063, %address.local993, !dbg !17
  %909 = trunc i64 %908 to i32, !dbg !17
  %.tr75 = shl i32 %909, 1, !dbg !17
  %dma.buffer.offset1620 = or i32 %.tr75, 64, !dbg !17
  %dma.buffer.lds1622 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %907), !dbg !17
  %910 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1622, i32 %dma.buffer.offset1620, <4 x i32> %dma.buffer.rsrc1630) #9, !dbg !17
  %911 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %912 = extractvalue [2 x <8 x float>] %132, 0, !dbg !24
  %913 = shufflevector <8 x float> %912, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %911, <4 x float> %913) #8, !dbg !23
  %915 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %915, <4 x float> %914) #8, !dbg !23
  %917 = shufflevector <8 x half> %892, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %917, <4 x float> %916) #8, !dbg !23
  %919 = shufflevector <8 x half> %894, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %919, <4 x float> %918) #8, !dbg !23
  %921 = shufflevector <8 x half> %896, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %921, <4 x float> %920) #8, !dbg !23
  %923 = shufflevector <8 x half> %898, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %923, <4 x float> %922) #8, !dbg !23
  %925 = shufflevector <8 x half> %900, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %925, <4 x float> %924) #8, !dbg !23
  %927 = shufflevector <8 x half> %902, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %927, <4 x float> %926) #8, !dbg !23
  %929 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %930 = shufflevector <8 x float> %912, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %931 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %929, <4 x float> %930) #8, !dbg !23
  %932 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %932, <4 x float> %931) #8, !dbg !23
  %934 = shufflevector <8 x half> %892, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %935 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %934, <4 x float> %933) #8, !dbg !23
  %936 = shufflevector <8 x half> %894, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %936, <4 x float> %935) #8, !dbg !23
  %938 = shufflevector <8 x half> %896, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %938, <4 x float> %937) #8, !dbg !23
  %940 = shufflevector <8 x half> %898, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %941 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %940, <4 x float> %939) #8, !dbg !23
  %942 = shufflevector <8 x half> %900, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %943 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %942, <4 x float> %941) #8, !dbg !23
  %944 = shufflevector <8 x half> %902, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %945 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %944, <4 x float> %943) #8, !dbg !23
  %946 = extractvalue [2 x <8 x float>] %132, 1, !dbg !24
  %947 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %911, <4 x float> %947) #8, !dbg !23
  %949 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %915, <4 x float> %948) #8, !dbg !23
  %950 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %917, <4 x float> %949) #8, !dbg !23
  %951 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %919, <4 x float> %950) #8, !dbg !23
  %952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %921, <4 x float> %951) #8, !dbg !23
  %953 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %923, <4 x float> %952) #8, !dbg !23
  %954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %925, <4 x float> %953) #8, !dbg !23
  %955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %927, <4 x float> %954) #8, !dbg !23
  %956 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %929, <4 x float> %956) #8, !dbg !23
  %958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %932, <4 x float> %957) #8, !dbg !23
  %959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %934, <4 x float> %958) #8, !dbg !23
  %960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %936, <4 x float> %959) #8, !dbg !23
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %938, <4 x float> %960) #8, !dbg !23
  %962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %940, <4 x float> %961) #8, !dbg !23
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %942, <4 x float> %962) #8, !dbg !23
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %944, <4 x float> %963) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %965 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local936, i32 0) #11, !dbg !23
  %966 = bitcast <4 x i32> %965 to <8 x half>, !dbg !23
  %967 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !23
  %968 = bitcast <4 x i32> %967 to <8 x half>, !dbg !23
  %969 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !23
  %970 = bitcast <4 x i32> %969 to <8 x half>, !dbg !23
  %971 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local957, i32 0) #11, !dbg !23
  %972 = bitcast <4 x i32> %971 to <8 x half>, !dbg !23
  %973 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !23
  %974 = bitcast <4 x i32> %973 to <8 x half>, !dbg !23
  %975 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !23
  %976 = bitcast <4 x i32> %975 to <8 x half>, !dbg !23
  %977 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !23
  %978 = bitcast <4 x i32> %977 to <8 x half>, !dbg !23
  %979 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #11, !dbg !23
  %980 = bitcast <4 x i32> %979 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %dma.buffer.offset1617 = or i32 %.tr71, 128, !dbg !17
  %981 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1625, i32 %dma.buffer.offset1617, <4 x i32> %dma.buffer.rsrc1630) #9, !dbg !17
  %dma.buffer.offset1614 = or i32 %.tr75, 128, !dbg !17
  %982 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1622, i32 %dma.buffer.offset1614, <4 x i32> %dma.buffer.rsrc1630) #9, !dbg !17
  %983 = shufflevector <8 x half> %966, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %984 = extractvalue [2 x <8 x float>] %130, 0, !dbg !24
  %985 = shufflevector <8 x float> %984, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %983, <4 x float> %985) #8, !dbg !23
  %987 = shufflevector <8 x half> %968, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %987, <4 x float> %986) #8, !dbg !23
  %989 = shufflevector <8 x half> %970, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %989, <4 x float> %988) #8, !dbg !23
  %991 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %991, <4 x float> %990) #8, !dbg !23
  %993 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %993, <4 x float> %992) #8, !dbg !23
  %995 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %995, <4 x float> %994) #8, !dbg !23
  %997 = shufflevector <8 x half> %978, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %997, <4 x float> %996) #8, !dbg !23
  %999 = shufflevector <8 x half> %980, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %999, <4 x float> %998) #8, !dbg !23
  %1001 = shufflevector <8 x half> %966, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1002 = shufflevector <8 x float> %984, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1003 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1001, <4 x float> %1002) #8, !dbg !23
  %1004 = shufflevector <8 x half> %968, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1005 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1004, <4 x float> %1003) #8, !dbg !23
  %1006 = shufflevector <8 x half> %970, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1007 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1006, <4 x float> %1005) #8, !dbg !23
  %1008 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1009 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1008, <4 x float> %1007) #8, !dbg !23
  %1010 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1011 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1010, <4 x float> %1009) #8, !dbg !23
  %1012 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1012, <4 x float> %1011) #8, !dbg !23
  %1014 = shufflevector <8 x half> %978, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1015 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1014, <4 x float> %1013) #8, !dbg !23
  %1016 = shufflevector <8 x half> %980, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1017 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1016, <4 x float> %1015) #8, !dbg !23
  %1018 = extractvalue [2 x <8 x float>] %130, 1, !dbg !24
  %1019 = shufflevector <8 x float> %1018, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %983, <4 x float> %1019) #8, !dbg !23
  %1021 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %987, <4 x float> %1020) #8, !dbg !23
  %1022 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %989, <4 x float> %1021) #8, !dbg !23
  %1023 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %991, <4 x float> %1022) #8, !dbg !23
  %1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %993, <4 x float> %1023) #8, !dbg !23
  %1025 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %995, <4 x float> %1024) #8, !dbg !23
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %997, <4 x float> %1025) #8, !dbg !23
  %1027 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %999, <4 x float> %1026) #8, !dbg !23
  %1028 = shufflevector <8 x float> %1018, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1029 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1001, <4 x float> %1028) #8, !dbg !23
  %1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1004, <4 x float> %1029) #8, !dbg !23
  %1031 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1006, <4 x float> %1030) #8, !dbg !23
  %1032 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1008, <4 x float> %1031) #8, !dbg !23
  %1033 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1010, <4 x float> %1032) #8, !dbg !23
  %1034 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1012, <4 x float> %1033) #8, !dbg !23
  %1035 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1014, <4 x float> %1034) #8, !dbg !23
  %1036 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1016, <4 x float> %1035) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1037 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local936, i32 0) #11, !dbg !23
  %1038 = bitcast <4 x i32> %1037 to <8 x half>, !dbg !23
  %1039 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !23
  %1040 = bitcast <4 x i32> %1039 to <8 x half>, !dbg !23
  %1041 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !23
  %1042 = bitcast <4 x i32> %1041 to <8 x half>, !dbg !23
  %1043 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local957, i32 0) #11, !dbg !23
  %1044 = bitcast <4 x i32> %1043 to <8 x half>, !dbg !23
  %1045 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !23
  %1046 = bitcast <4 x i32> %1045 to <8 x half>, !dbg !23
  %1047 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !23
  %1048 = bitcast <4 x i32> %1047 to <8 x half>, !dbg !23
  %1049 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !23
  %1050 = bitcast <4 x i32> %1049 to <8 x half>, !dbg !23
  %1051 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #11, !dbg !23
  %1052 = bitcast <4 x i32> %1051 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1053 = or disjoint i64 %.idx6063, 96, !dbg !17
  %1054 = or disjoint i64 %address.local981, %1053, !dbg !17
  %.tr87 = trunc i64 %1054 to i32, !dbg !17
  %dma.buffer.offset1611 = shl i32 %.tr87, 1, !dbg !17
  %1055 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1625, i32 %dma.buffer.offset1611, <4 x i32> %dma.buffer.rsrc1630) #9, !dbg !17
  %1056 = or disjoint i64 %address.local993, %1053, !dbg !17
  %.tr91 = trunc i64 %1056 to i32, !dbg !17
  %dma.buffer.offset = shl i32 %.tr91, 1, !dbg !17
  %1057 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1622, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1630) #9, !dbg !17
  %1058 = shufflevector <8 x half> %1038, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1059 = extractvalue [2 x <8 x float>] %129, 0, !dbg !24
  %1060 = shufflevector <8 x float> %1059, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1061 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1058, <4 x float> %1060) #8, !dbg !23
  %1062 = shufflevector <8 x half> %1040, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1063 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1062, <4 x float> %1061) #8, !dbg !23
  %1064 = shufflevector <8 x half> %1042, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1065 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1064, <4 x float> %1063) #8, !dbg !23
  %1066 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1067 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1066, <4 x float> %1065) #8, !dbg !23
  %1068 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1069 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1068, <4 x float> %1067) #8, !dbg !23
  %1070 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1071 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1070, <4 x float> %1069) #8, !dbg !23
  %1072 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1073 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1072, <4 x float> %1071) #8, !dbg !23
  %1074 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1075 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1074, <4 x float> %1073) #8, !dbg !23
  %1076 = shufflevector <8 x half> %1038, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1077 = shufflevector <8 x float> %1059, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1076, <4 x float> %1077) #8, !dbg !23
  %1079 = shufflevector <8 x half> %1040, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1079, <4 x float> %1078) #8, !dbg !23
  %1081 = shufflevector <8 x half> %1042, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1081, <4 x float> %1080) #8, !dbg !23
  %1083 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1083, <4 x float> %1082) #8, !dbg !23
  %1085 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1085, <4 x float> %1084) #8, !dbg !23
  %1087 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1087, <4 x float> %1086) #8, !dbg !23
  %1089 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1089, <4 x float> %1088) #8, !dbg !23
  %1091 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1091, <4 x float> %1090) #8, !dbg !23
  %1093 = extractvalue [2 x <8 x float>] %129, 1, !dbg !24
  %1094 = shufflevector <8 x float> %1093, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1095 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1058, <4 x float> %1094) #8, !dbg !23
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1062, <4 x float> %1095) #8, !dbg !23
  %1097 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1064, <4 x float> %1096) #8, !dbg !23
  %1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1066, <4 x float> %1097) #8, !dbg !23
  %1099 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1068, <4 x float> %1098) #8, !dbg !23
  %1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1070, <4 x float> %1099) #8, !dbg !23
  %1101 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1072, <4 x float> %1100) #8, !dbg !23
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1074, <4 x float> %1101) #8, !dbg !23
  %1103 = shufflevector <8 x float> %1093, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1076, <4 x float> %1103) #8, !dbg !23
  %1105 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1079, <4 x float> %1104) #8, !dbg !23
  %1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1081, <4 x float> %1105) #8, !dbg !23
  %1107 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1083, <4 x float> %1106) #8, !dbg !23
  %1108 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1085, <4 x float> %1107) #8, !dbg !23
  %1109 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1087, <4 x float> %1108) #8, !dbg !23
  %1110 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1089, <4 x float> %1109) #8, !dbg !23
  %1111 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1091, <4 x float> %1110) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1112 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local936, i32 0) #11, !dbg !23
  %1113 = bitcast <4 x i32> %1112 to <8 x half>, !dbg !23
  %1114 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !23
  %1115 = bitcast <4 x i32> %1114 to <8 x half>, !dbg !23
  %1116 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !23
  %1117 = bitcast <4 x i32> %1116 to <8 x half>, !dbg !23
  %1118 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local957, i32 0) #11, !dbg !23
  %1119 = bitcast <4 x i32> %1118 to <8 x half>, !dbg !23
  %1120 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !23
  %1121 = bitcast <4 x i32> %1120 to <8 x half>, !dbg !23
  %1122 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !23
  %1123 = bitcast <4 x i32> %1122 to <8 x half>, !dbg !23
  %1124 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !23
  %1125 = bitcast <4 x i32> %1124 to <8 x half>, !dbg !23
  %1126 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #11, !dbg !23
  %1127 = bitcast <4 x i32> %1126 to <8 x half>, !dbg !23
  %1128 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1129 = extractvalue [2 x <8 x float>] %128, 0, !dbg !24
  %1130 = shufflevector <8 x float> %1129, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1131 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1128, <4 x float> %1130) #8, !dbg !23
  %1132 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1132, <4 x float> %1131) #8, !dbg !23
  %1134 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1135 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1134, <4 x float> %1133) #8, !dbg !23
  %1136 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1136, <4 x float> %1135) #8, !dbg !23
  %1138 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1139 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1138, <4 x float> %1137) #8, !dbg !23
  %1140 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1140, <4 x float> %1139) #8, !dbg !23
  %1142 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1142, <4 x float> %1141) #8, !dbg !23
  %1144 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1144, <4 x float> %1143) #8, !dbg !23
  %1146 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1147 = shufflevector <8 x float> %1129, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1146, <4 x float> %1147) #8, !dbg !23
  %1149 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1149, <4 x float> %1148) #8, !dbg !23
  %1151 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1151, <4 x float> %1150) #8, !dbg !23
  %1153 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1153, <4 x float> %1152) #8, !dbg !23
  %1155 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1155, <4 x float> %1154) #8, !dbg !23
  %1157 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1157, <4 x float> %1156) #8, !dbg !23
  %1159 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1159, <4 x float> %1158) #8, !dbg !23
  %1161 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1161, <4 x float> %1160) #8, !dbg !23
  %1163 = extractvalue [2 x <8 x float>] %128, 1, !dbg !24
  %1164 = shufflevector <8 x float> %1163, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1128, <4 x float> %1164) #8, !dbg !23
  %1166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1132, <4 x float> %1165) #8, !dbg !23
  %1167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1134, <4 x float> %1166) #8, !dbg !23
  %1168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1136, <4 x float> %1167) #8, !dbg !23
  %1169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1138, <4 x float> %1168) #8, !dbg !23
  %1170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1140, <4 x float> %1169) #8, !dbg !23
  %1171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1142, <4 x float> %1170) #8, !dbg !23
  %1172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1144, <4 x float> %1171) #8, !dbg !23
  %1173 = shufflevector <8 x float> %1163, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1146, <4 x float> %1173) #8, !dbg !23
  %1175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1149, <4 x float> %1174) #8, !dbg !23
  %1176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1151, <4 x float> %1175) #8, !dbg !23
  %1177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1153, <4 x float> %1176) #8, !dbg !23
  %1178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1155, <4 x float> %1177) #8, !dbg !23
  %1179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1157, <4 x float> %1178) #8, !dbg !23
  %1180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1159, <4 x float> %1179) #8, !dbg !23
  %1181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1161, <4 x float> %1180) #8, !dbg !23
  %1182 = shufflevector <4 x float> %928, <4 x float> %945, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1183 = insertvalue [2 x <8 x float>] poison, <8 x float> %1182, 0, !dbg !11
  %1184 = shufflevector <4 x float> %955, <4 x float> %964, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1185 = insertvalue [2 x <8 x float>] %1183, <8 x float> %1184, 1, !dbg !11
  %1186 = shufflevector <4 x float> %1000, <4 x float> %1017, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1187 = insertvalue [2 x <8 x float>] poison, <8 x float> %1186, 0, !dbg !11
  %1188 = shufflevector <4 x float> %1027, <4 x float> %1036, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1189 = insertvalue [2 x <8 x float>] %1187, <8 x float> %1188, 1, !dbg !11
  %1190 = shufflevector <4 x float> %1075, <4 x float> %1092, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1191 = insertvalue [2 x <8 x float>] poison, <8 x float> %1190, 0, !dbg !11
  %1192 = shufflevector <4 x float> %1102, <4 x float> %1111, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1193 = insertvalue [2 x <8 x float>] %1191, <8 x float> %1192, 1, !dbg !11
  %1194 = shufflevector <4 x float> %1145, <4 x float> %1162, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1195 = insertvalue [2 x <8 x float>] poison, <8 x float> %1194, 0, !dbg !11
  %1196 = shufflevector <4 x float> %1172, <4 x float> %1181, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1197 = insertvalue [2 x <8 x float>] %1195, <8 x float> %1196, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1198 = add nuw nsw i64 %133, 1, !dbg !11
  %1199 = icmp slt i64 %1198, %93, !dbg !11
  br i1 %1199, label %127, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %543, %.._crit_edge_crit_edge
  %.pre-phi117 = phi i32 [ %.pre116, %.._crit_edge_crit_edge ], [ %861, %543 ], !dbg !11
  %.pre-phi115 = phi i32 [ %.pre114, %.._crit_edge_crit_edge ], [ %853, %543 ], !dbg !11
  %.pre-phi113 = phi i32 [ %.pre112, %.._crit_edge_crit_edge ], [ %852, %543 ], !dbg !11
  %.pre-phi109 = phi i32 [ %.pre108, %.._crit_edge_crit_edge ], [ %850, %543 ], !dbg !11
  %.lcssa97 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1185, %543 ], !dbg !11
  %.lcssa96 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %886, %543 ], !dbg !11
  %.lcssa95 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1189, %543 ], !dbg !11
  %.lcssa94 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1193, %543 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1197, %543 ], !dbg !11
  %1200 = and i32 %29, 16, !dbg !11
  %.not = icmp eq i32 %1200, 0, !dbg !11
  %1201 = and i32 %29, 32, !dbg !11
  %.not36 = icmp eq i32 %1201, 0, !dbg !11
  %1202 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !11
  %1203 = extractelement <8 x float> %1202, i64 0, !dbg !11
  %1204 = extractelement <8 x float> %1202, i64 1, !dbg !11
  %1205 = extractelement <8 x float> %1202, i64 2, !dbg !11
  %1206 = extractelement <8 x float> %1202, i64 3, !dbg !11
  %1207 = select i1 %.not, float %1204, float %1203, !dbg !11
  %1208 = icmp samesign ult i32 %.pre-phi113, %.pre-phi115, !dbg !11
  %1209 = select i1 %1208, i32 %.pre-phi113, i32 %.pre-phi109, !dbg !11
  %1210 = shl nuw nsw i32 %1209, 2, !dbg !11
  %1211 = bitcast float %1207 to i32, !dbg !11
  %1212 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1211), !dbg !11
  %1213 = bitcast i32 %1212 to float, !dbg !11
  %1214 = select i1 %.not, float %1203, float %1213, !dbg !11
  %1215 = select i1 %.not, float %1213, float %1204, !dbg !11
  %1216 = select i1 %.not, float %1206, float %1205, !dbg !11
  %1217 = bitcast float %1216 to i32, !dbg !11
  %1218 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1217), !dbg !11
  %1219 = bitcast i32 %1218 to float, !dbg !11
  %1220 = select i1 %.not, float %1205, float %1219, !dbg !11
  %1221 = select i1 %.not, float %1219, float %1206, !dbg !11
  %1222 = select i1 %.not36, float %1220, float %1214, !dbg !11
  %1223 = icmp samesign ult i32 %.pre-phi117, %.pre-phi115, !dbg !11
  %1224 = select i1 %1223, i32 %.pre-phi117, i32 %.pre-phi109, !dbg !11
  %1225 = shl nuw nsw i32 %1224, 2, !dbg !11
  %1226 = bitcast float %1222 to i32, !dbg !11
  %1227 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1226), !dbg !11
  %1228 = bitcast i32 %1227 to float, !dbg !11
  %1229 = select i1 %.not36, float %1221, float %1215, !dbg !11
  %1230 = bitcast float %1229 to i32, !dbg !11
  %1231 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1230), !dbg !11
  %1232 = bitcast i32 %1231 to float, !dbg !11
  %1233 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1234 = shufflevector <4 x i1> %1233, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1235 = insertelement <4 x float> poison, float %1214, i64 0, !dbg !11
  %1236 = insertelement <4 x float> %1235, float %1215, i64 1, !dbg !11
  %1237 = insertelement <4 x float> %1236, float %1228, i64 2, !dbg !11
  %1238 = insertelement <4 x float> %1237, float %1232, i64 3, !dbg !11
  %1239 = insertelement <4 x float> poison, float %1228, i64 0, !dbg !11
  %1240 = insertelement <4 x float> %1239, float %1232, i64 1, !dbg !11
  %1241 = insertelement <4 x float> %1240, float %1220, i64 2, !dbg !11
  %1242 = insertelement <4 x float> %1241, float %1221, i64 3, !dbg !11
  %1243 = select <4 x i1> %1234, <4 x float> %1238, <4 x float> %1242, !dbg !11
  %1244 = extractvalue [2 x <8 x float>] %.lcssa94, 0, !dbg !11
  %1245 = extractelement <8 x float> %1244, i64 0, !dbg !11
  %1246 = extractelement <8 x float> %1244, i64 1, !dbg !11
  %1247 = extractelement <8 x float> %1244, i64 2, !dbg !11
  %1248 = extractelement <8 x float> %1244, i64 3, !dbg !11
  %1249 = select i1 %.not, float %1246, float %1245, !dbg !11
  %1250 = bitcast float %1249 to i32, !dbg !11
  %1251 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1250), !dbg !11
  %1252 = bitcast i32 %1251 to float, !dbg !11
  %1253 = select i1 %.not, float %1245, float %1252, !dbg !11
  %1254 = select i1 %.not, float %1252, float %1246, !dbg !11
  %1255 = select i1 %.not, float %1248, float %1247, !dbg !11
  %1256 = bitcast float %1255 to i32, !dbg !11
  %1257 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1256), !dbg !11
  %1258 = bitcast i32 %1257 to float, !dbg !11
  %1259 = select i1 %.not, float %1247, float %1258, !dbg !11
  %1260 = select i1 %.not, float %1258, float %1248, !dbg !11
  %1261 = select i1 %.not36, float %1259, float %1253, !dbg !11
  %1262 = bitcast float %1261 to i32, !dbg !11
  %1263 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1262), !dbg !11
  %1264 = bitcast i32 %1263 to float, !dbg !11
  %1265 = select i1 %.not36, float %1260, float %1254, !dbg !11
  %1266 = bitcast float %1265 to i32, !dbg !11
  %1267 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1266), !dbg !11
  %1268 = bitcast i32 %1267 to float, !dbg !11
  %1269 = insertelement <4 x float> poison, float %1253, i64 0, !dbg !11
  %1270 = insertelement <4 x float> %1269, float %1254, i64 1, !dbg !11
  %1271 = insertelement <4 x float> %1270, float %1264, i64 2, !dbg !11
  %1272 = insertelement <4 x float> %1271, float %1268, i64 3, !dbg !11
  %1273 = insertelement <4 x float> poison, float %1264, i64 0, !dbg !11
  %1274 = insertelement <4 x float> %1273, float %1268, i64 1, !dbg !11
  %1275 = insertelement <4 x float> %1274, float %1259, i64 2, !dbg !11
  %1276 = insertelement <4 x float> %1275, float %1260, i64 3, !dbg !11
  %1277 = select <4 x i1> %1234, <4 x float> %1272, <4 x float> %1276, !dbg !11
  %1278 = extractvalue [2 x <8 x float>] %.lcssa95, 0, !dbg !11
  %1279 = extractelement <8 x float> %1278, i64 0, !dbg !11
  %1280 = extractelement <8 x float> %1278, i64 1, !dbg !11
  %1281 = extractelement <8 x float> %1278, i64 2, !dbg !11
  %1282 = extractelement <8 x float> %1278, i64 3, !dbg !11
  %1283 = select i1 %.not, float %1280, float %1279, !dbg !11
  %1284 = bitcast float %1283 to i32, !dbg !11
  %1285 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1284), !dbg !11
  %1286 = bitcast i32 %1285 to float, !dbg !11
  %1287 = select i1 %.not, float %1279, float %1286, !dbg !11
  %1288 = select i1 %.not, float %1286, float %1280, !dbg !11
  %1289 = select i1 %.not, float %1282, float %1281, !dbg !11
  %1290 = bitcast float %1289 to i32, !dbg !11
  %1291 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1290), !dbg !11
  %1292 = bitcast i32 %1291 to float, !dbg !11
  %1293 = select i1 %.not, float %1281, float %1292, !dbg !11
  %1294 = select i1 %.not, float %1292, float %1282, !dbg !11
  %1295 = select i1 %.not36, float %1293, float %1287, !dbg !11
  %1296 = bitcast float %1295 to i32, !dbg !11
  %1297 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1296), !dbg !11
  %1298 = bitcast i32 %1297 to float, !dbg !11
  %1299 = select i1 %.not36, float %1294, float %1288, !dbg !11
  %1300 = bitcast float %1299 to i32, !dbg !11
  %1301 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1300), !dbg !11
  %1302 = bitcast i32 %1301 to float, !dbg !11
  %1303 = insertelement <4 x float> poison, float %1287, i64 0, !dbg !11
  %1304 = insertelement <4 x float> %1303, float %1288, i64 1, !dbg !11
  %1305 = insertelement <4 x float> %1304, float %1298, i64 2, !dbg !11
  %1306 = insertelement <4 x float> %1305, float %1302, i64 3, !dbg !11
  %1307 = insertelement <4 x float> poison, float %1298, i64 0, !dbg !11
  %1308 = insertelement <4 x float> %1307, float %1302, i64 1, !dbg !11
  %1309 = insertelement <4 x float> %1308, float %1293, i64 2, !dbg !11
  %1310 = insertelement <4 x float> %1309, float %1294, i64 3, !dbg !11
  %1311 = select <4 x i1> %1234, <4 x float> %1306, <4 x float> %1310, !dbg !11
  %1312 = extractvalue [2 x <8 x float>] %.lcssa97, 0, !dbg !11
  %1313 = extractelement <8 x float> %1312, i64 0, !dbg !11
  %1314 = extractelement <8 x float> %1312, i64 1, !dbg !11
  %1315 = extractelement <8 x float> %1312, i64 2, !dbg !11
  %1316 = extractelement <8 x float> %1312, i64 3, !dbg !11
  %1317 = select i1 %.not, float %1314, float %1313, !dbg !11
  %1318 = bitcast float %1317 to i32, !dbg !11
  %1319 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1318), !dbg !11
  %1320 = bitcast i32 %1319 to float, !dbg !11
  %1321 = select i1 %.not, float %1313, float %1320, !dbg !11
  %1322 = select i1 %.not, float %1320, float %1314, !dbg !11
  %1323 = select i1 %.not, float %1316, float %1315, !dbg !11
  %1324 = bitcast float %1323 to i32, !dbg !11
  %1325 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1324), !dbg !11
  %1326 = bitcast i32 %1325 to float, !dbg !11
  %1327 = select i1 %.not, float %1315, float %1326, !dbg !11
  %1328 = select i1 %.not, float %1326, float %1316, !dbg !11
  %1329 = select i1 %.not36, float %1327, float %1321, !dbg !11
  %1330 = bitcast float %1329 to i32, !dbg !11
  %1331 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1330), !dbg !11
  %1332 = select i1 %.not36, float %1328, float %1322, !dbg !11
  %1333 = bitcast float %1332 to i32, !dbg !11
  %1334 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1333), !dbg !11
  %1335 = extractvalue [2 x <1 x float>] %.lcssa96, 0, !dbg !25
  %1336 = shufflevector <1 x float> %1335, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %1337 = extractelement <8 x float> %1202, i64 4, !dbg !11
  %1338 = extractelement <8 x float> %1202, i64 5, !dbg !11
  %1339 = extractelement <8 x float> %1202, i64 6, !dbg !11
  %1340 = extractelement <8 x float> %1202, i64 7, !dbg !11
  %1341 = select i1 %.not, float %1338, float %1337, !dbg !11
  %1342 = bitcast float %1341 to i32, !dbg !11
  %1343 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1342), !dbg !11
  %1344 = bitcast i32 %1343 to float, !dbg !11
  %1345 = select i1 %.not, float %1337, float %1344, !dbg !11
  %1346 = select i1 %.not, float %1344, float %1338, !dbg !11
  %1347 = select i1 %.not, float %1340, float %1339, !dbg !11
  %1348 = bitcast float %1347 to i32, !dbg !11
  %1349 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1348), !dbg !11
  %1350 = bitcast i32 %1349 to float, !dbg !11
  %1351 = select i1 %.not, float %1339, float %1350, !dbg !11
  %1352 = select i1 %.not, float %1350, float %1340, !dbg !11
  %1353 = select i1 %.not36, float %1351, float %1345, !dbg !11
  %1354 = bitcast float %1353 to i32, !dbg !11
  %1355 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1354), !dbg !11
  %1356 = bitcast i32 %1355 to float, !dbg !11
  %1357 = select i1 %.not36, float %1352, float %1346, !dbg !11
  %1358 = bitcast float %1357 to i32, !dbg !11
  %1359 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1358), !dbg !11
  %1360 = bitcast i32 %1359 to float, !dbg !11
  %1361 = insertelement <4 x float> poison, float %1345, i64 0, !dbg !11
  %1362 = insertelement <4 x float> %1361, float %1346, i64 1, !dbg !11
  %1363 = insertelement <4 x float> %1362, float %1356, i64 2, !dbg !11
  %1364 = insertelement <4 x float> %1363, float %1360, i64 3, !dbg !11
  %1365 = insertelement <4 x float> poison, float %1356, i64 0, !dbg !11
  %1366 = insertelement <4 x float> %1365, float %1360, i64 1, !dbg !11
  %1367 = insertelement <4 x float> %1366, float %1351, i64 2, !dbg !11
  %1368 = insertelement <4 x float> %1367, float %1352, i64 3, !dbg !11
  %1369 = select <4 x i1> %1234, <4 x float> %1364, <4 x float> %1368, !dbg !11
  %1370 = extractelement <8 x float> %1244, i64 4, !dbg !11
  %1371 = extractelement <8 x float> %1244, i64 5, !dbg !11
  %1372 = extractelement <8 x float> %1244, i64 6, !dbg !11
  %1373 = extractelement <8 x float> %1244, i64 7, !dbg !11
  %1374 = select i1 %.not, float %1371, float %1370, !dbg !11
  %1375 = bitcast float %1374 to i32, !dbg !11
  %1376 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1375), !dbg !11
  %1377 = bitcast i32 %1376 to float, !dbg !11
  %1378 = select i1 %.not, float %1370, float %1377, !dbg !11
  %1379 = select i1 %.not, float %1377, float %1371, !dbg !11
  %1380 = select i1 %.not, float %1373, float %1372, !dbg !11
  %1381 = bitcast float %1380 to i32, !dbg !11
  %1382 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1381), !dbg !11
  %1383 = bitcast i32 %1382 to float, !dbg !11
  %1384 = select i1 %.not, float %1372, float %1383, !dbg !11
  %1385 = select i1 %.not, float %1383, float %1373, !dbg !11
  %1386 = select i1 %.not36, float %1384, float %1378, !dbg !11
  %1387 = bitcast float %1386 to i32, !dbg !11
  %1388 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1387), !dbg !11
  %1389 = bitcast i32 %1388 to float, !dbg !11
  %1390 = select i1 %.not36, float %1385, float %1379, !dbg !11
  %1391 = bitcast float %1390 to i32, !dbg !11
  %1392 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1391), !dbg !11
  %1393 = bitcast i32 %1392 to float, !dbg !11
  %1394 = insertelement <4 x float> poison, float %1378, i64 0, !dbg !11
  %1395 = insertelement <4 x float> %1394, float %1379, i64 1, !dbg !11
  %1396 = insertelement <4 x float> %1395, float %1389, i64 2, !dbg !11
  %1397 = insertelement <4 x float> %1396, float %1393, i64 3, !dbg !11
  %1398 = insertelement <4 x float> poison, float %1389, i64 0, !dbg !11
  %1399 = insertelement <4 x float> %1398, float %1393, i64 1, !dbg !11
  %1400 = insertelement <4 x float> %1399, float %1384, i64 2, !dbg !11
  %1401 = insertelement <4 x float> %1400, float %1385, i64 3, !dbg !11
  %1402 = select <4 x i1> %1234, <4 x float> %1397, <4 x float> %1401, !dbg !11
  %1403 = extractelement <8 x float> %1278, i64 4, !dbg !11
  %1404 = extractelement <8 x float> %1278, i64 5, !dbg !11
  %1405 = extractelement <8 x float> %1278, i64 6, !dbg !11
  %1406 = extractelement <8 x float> %1278, i64 7, !dbg !11
  %1407 = select i1 %.not, float %1404, float %1403, !dbg !11
  %1408 = bitcast float %1407 to i32, !dbg !11
  %1409 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1408), !dbg !11
  %1410 = bitcast i32 %1409 to float, !dbg !11
  %1411 = select i1 %.not, float %1403, float %1410, !dbg !11
  %1412 = select i1 %.not, float %1410, float %1404, !dbg !11
  %1413 = select i1 %.not, float %1406, float %1405, !dbg !11
  %1414 = bitcast float %1413 to i32, !dbg !11
  %1415 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1414), !dbg !11
  %1416 = bitcast i32 %1415 to float, !dbg !11
  %1417 = select i1 %.not, float %1405, float %1416, !dbg !11
  %1418 = select i1 %.not, float %1416, float %1406, !dbg !11
  %1419 = select i1 %.not36, float %1417, float %1411, !dbg !11
  %1420 = bitcast float %1419 to i32, !dbg !11
  %1421 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1420), !dbg !11
  %1422 = bitcast i32 %1421 to float, !dbg !11
  %1423 = select i1 %.not36, float %1418, float %1412, !dbg !11
  %1424 = bitcast float %1423 to i32, !dbg !11
  %1425 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1424), !dbg !11
  %1426 = bitcast i32 %1425 to float, !dbg !11
  %1427 = insertelement <4 x float> poison, float %1411, i64 0, !dbg !11
  %1428 = insertelement <4 x float> %1427, float %1412, i64 1, !dbg !11
  %1429 = insertelement <4 x float> %1428, float %1422, i64 2, !dbg !11
  %1430 = insertelement <4 x float> %1429, float %1426, i64 3, !dbg !11
  %1431 = insertelement <4 x float> poison, float %1422, i64 0, !dbg !11
  %1432 = insertelement <4 x float> %1431, float %1426, i64 1, !dbg !11
  %1433 = insertelement <4 x float> %1432, float %1417, i64 2, !dbg !11
  %1434 = insertelement <4 x float> %1433, float %1418, i64 3, !dbg !11
  %1435 = select <4 x i1> %1234, <4 x float> %1430, <4 x float> %1434, !dbg !11
  %1436 = extractelement <8 x float> %1312, i64 4, !dbg !11
  %1437 = extractelement <8 x float> %1312, i64 5, !dbg !11
  %1438 = extractelement <8 x float> %1312, i64 6, !dbg !11
  %1439 = extractelement <8 x float> %1312, i64 7, !dbg !11
  %1440 = select i1 %.not, float %1437, float %1436, !dbg !11
  %1441 = bitcast float %1440 to i32, !dbg !11
  %1442 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1441), !dbg !11
  %1443 = bitcast i32 %1442 to float, !dbg !11
  %1444 = select i1 %.not, float %1436, float %1443, !dbg !11
  %1445 = select i1 %.not, float %1443, float %1437, !dbg !11
  %1446 = select i1 %.not, float %1439, float %1438, !dbg !11
  %1447 = bitcast float %1446 to i32, !dbg !11
  %1448 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1447), !dbg !11
  %1449 = bitcast i32 %1448 to float, !dbg !11
  %1450 = select i1 %.not, float %1438, float %1449, !dbg !11
  %1451 = select i1 %.not, float %1449, float %1439, !dbg !11
  %1452 = select i1 %.not36, float %1450, float %1444, !dbg !11
  %1453 = bitcast float %1452 to i32, !dbg !11
  %1454 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1453), !dbg !11
  %1455 = select i1 %.not36, float %1451, float %1445, !dbg !11
  %1456 = bitcast float %1455 to i32, !dbg !11
  %1457 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1456), !dbg !11
  %1458 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !11
  %1459 = extractelement <8 x float> %1458, i64 0, !dbg !11
  %1460 = extractelement <8 x float> %1458, i64 1, !dbg !11
  %1461 = extractelement <8 x float> %1458, i64 2, !dbg !11
  %1462 = extractelement <8 x float> %1458, i64 3, !dbg !11
  %1463 = select i1 %.not, float %1460, float %1459, !dbg !11
  %1464 = bitcast float %1463 to i32, !dbg !11
  %1465 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1464), !dbg !11
  %1466 = bitcast i32 %1465 to float, !dbg !11
  %1467 = select i1 %.not, float %1459, float %1466, !dbg !11
  %1468 = select i1 %.not, float %1466, float %1460, !dbg !11
  %1469 = select i1 %.not, float %1462, float %1461, !dbg !11
  %1470 = bitcast float %1469 to i32, !dbg !11
  %1471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1470), !dbg !11
  %1472 = bitcast i32 %1471 to float, !dbg !11
  %1473 = select i1 %.not, float %1461, float %1472, !dbg !11
  %1474 = select i1 %.not, float %1472, float %1462, !dbg !11
  %1475 = select i1 %.not36, float %1473, float %1467, !dbg !11
  %1476 = bitcast float %1475 to i32, !dbg !11
  %1477 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1476), !dbg !11
  %1478 = bitcast i32 %1477 to float, !dbg !11
  %1479 = select i1 %.not36, float %1474, float %1468, !dbg !11
  %1480 = bitcast float %1479 to i32, !dbg !11
  %1481 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1480), !dbg !11
  %1482 = bitcast i32 %1481 to float, !dbg !11
  %1483 = insertelement <4 x float> poison, float %1467, i64 0, !dbg !11
  %1484 = insertelement <4 x float> %1483, float %1468, i64 1, !dbg !11
  %1485 = insertelement <4 x float> %1484, float %1478, i64 2, !dbg !11
  %1486 = insertelement <4 x float> %1485, float %1482, i64 3, !dbg !11
  %1487 = insertelement <4 x float> poison, float %1478, i64 0, !dbg !11
  %1488 = insertelement <4 x float> %1487, float %1482, i64 1, !dbg !11
  %1489 = insertelement <4 x float> %1488, float %1473, i64 2, !dbg !11
  %1490 = insertelement <4 x float> %1489, float %1474, i64 3, !dbg !11
  %1491 = select <4 x i1> %1234, <4 x float> %1486, <4 x float> %1490, !dbg !11
  %1492 = extractvalue [2 x <8 x float>] %.lcssa94, 1, !dbg !11
  %1493 = extractelement <8 x float> %1492, i64 0, !dbg !11
  %1494 = extractelement <8 x float> %1492, i64 1, !dbg !11
  %1495 = extractelement <8 x float> %1492, i64 2, !dbg !11
  %1496 = extractelement <8 x float> %1492, i64 3, !dbg !11
  %1497 = select i1 %.not, float %1494, float %1493, !dbg !11
  %1498 = bitcast float %1497 to i32, !dbg !11
  %1499 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1498), !dbg !11
  %1500 = bitcast i32 %1499 to float, !dbg !11
  %1501 = select i1 %.not, float %1493, float %1500, !dbg !11
  %1502 = select i1 %.not, float %1500, float %1494, !dbg !11
  %1503 = select i1 %.not, float %1496, float %1495, !dbg !11
  %1504 = bitcast float %1503 to i32, !dbg !11
  %1505 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1504), !dbg !11
  %1506 = bitcast i32 %1505 to float, !dbg !11
  %1507 = select i1 %.not, float %1495, float %1506, !dbg !11
  %1508 = select i1 %.not, float %1506, float %1496, !dbg !11
  %1509 = select i1 %.not36, float %1507, float %1501, !dbg !11
  %1510 = bitcast float %1509 to i32, !dbg !11
  %1511 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1510), !dbg !11
  %1512 = bitcast i32 %1511 to float, !dbg !11
  %1513 = select i1 %.not36, float %1508, float %1502, !dbg !11
  %1514 = bitcast float %1513 to i32, !dbg !11
  %1515 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1514), !dbg !11
  %1516 = bitcast i32 %1515 to float, !dbg !11
  %1517 = insertelement <4 x float> poison, float %1501, i64 0, !dbg !11
  %1518 = insertelement <4 x float> %1517, float %1502, i64 1, !dbg !11
  %1519 = insertelement <4 x float> %1518, float %1512, i64 2, !dbg !11
  %1520 = insertelement <4 x float> %1519, float %1516, i64 3, !dbg !11
  %1521 = insertelement <4 x float> poison, float %1512, i64 0, !dbg !11
  %1522 = insertelement <4 x float> %1521, float %1516, i64 1, !dbg !11
  %1523 = insertelement <4 x float> %1522, float %1507, i64 2, !dbg !11
  %1524 = insertelement <4 x float> %1523, float %1508, i64 3, !dbg !11
  %1525 = select <4 x i1> %1234, <4 x float> %1520, <4 x float> %1524, !dbg !11
  %1526 = extractvalue [2 x <8 x float>] %.lcssa95, 1, !dbg !11
  %1527 = extractelement <8 x float> %1526, i64 0, !dbg !11
  %1528 = extractelement <8 x float> %1526, i64 1, !dbg !11
  %1529 = extractelement <8 x float> %1526, i64 2, !dbg !11
  %1530 = extractelement <8 x float> %1526, i64 3, !dbg !11
  %1531 = select i1 %.not, float %1528, float %1527, !dbg !11
  %1532 = bitcast float %1531 to i32, !dbg !11
  %1533 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1532), !dbg !11
  %1534 = bitcast i32 %1533 to float, !dbg !11
  %1535 = select i1 %.not, float %1527, float %1534, !dbg !11
  %1536 = select i1 %.not, float %1534, float %1528, !dbg !11
  %1537 = select i1 %.not, float %1530, float %1529, !dbg !11
  %1538 = bitcast float %1537 to i32, !dbg !11
  %1539 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1538), !dbg !11
  %1540 = bitcast i32 %1539 to float, !dbg !11
  %1541 = select i1 %.not, float %1529, float %1540, !dbg !11
  %1542 = select i1 %.not, float %1540, float %1530, !dbg !11
  %1543 = select i1 %.not36, float %1541, float %1535, !dbg !11
  %1544 = bitcast float %1543 to i32, !dbg !11
  %1545 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1544), !dbg !11
  %1546 = bitcast i32 %1545 to float, !dbg !11
  %1547 = select i1 %.not36, float %1542, float %1536, !dbg !11
  %1548 = bitcast float %1547 to i32, !dbg !11
  %1549 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1548), !dbg !11
  %1550 = bitcast i32 %1549 to float, !dbg !11
  %1551 = insertelement <4 x float> poison, float %1535, i64 0, !dbg !11
  %1552 = insertelement <4 x float> %1551, float %1536, i64 1, !dbg !11
  %1553 = insertelement <4 x float> %1552, float %1546, i64 2, !dbg !11
  %1554 = insertelement <4 x float> %1553, float %1550, i64 3, !dbg !11
  %1555 = insertelement <4 x float> poison, float %1546, i64 0, !dbg !11
  %1556 = insertelement <4 x float> %1555, float %1550, i64 1, !dbg !11
  %1557 = insertelement <4 x float> %1556, float %1541, i64 2, !dbg !11
  %1558 = insertelement <4 x float> %1557, float %1542, i64 3, !dbg !11
  %1559 = select <4 x i1> %1234, <4 x float> %1554, <4 x float> %1558, !dbg !11
  %1560 = extractvalue [2 x <8 x float>] %.lcssa97, 1, !dbg !11
  %1561 = extractelement <8 x float> %1560, i64 0, !dbg !11
  %1562 = extractelement <8 x float> %1560, i64 1, !dbg !11
  %1563 = extractelement <8 x float> %1560, i64 2, !dbg !11
  %1564 = extractelement <8 x float> %1560, i64 3, !dbg !11
  %1565 = select i1 %.not, float %1562, float %1561, !dbg !11
  %1566 = bitcast float %1565 to i32, !dbg !11
  %1567 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1566), !dbg !11
  %1568 = bitcast i32 %1567 to float, !dbg !11
  %1569 = select i1 %.not, float %1561, float %1568, !dbg !11
  %1570 = select i1 %.not, float %1568, float %1562, !dbg !11
  %1571 = select i1 %.not, float %1564, float %1563, !dbg !11
  %1572 = bitcast float %1571 to i32, !dbg !11
  %1573 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1572), !dbg !11
  %1574 = bitcast i32 %1573 to float, !dbg !11
  %1575 = select i1 %.not, float %1563, float %1574, !dbg !11
  %1576 = select i1 %.not, float %1574, float %1564, !dbg !11
  %1577 = select i1 %.not36, float %1575, float %1569, !dbg !11
  %1578 = bitcast float %1577 to i32, !dbg !11
  %1579 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1578), !dbg !11
  %1580 = select i1 %.not36, float %1576, float %1570, !dbg !11
  %1581 = bitcast float %1580 to i32, !dbg !11
  %1582 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1581), !dbg !11
  %1583 = extractvalue [2 x <1 x float>] %.lcssa96, 1, !dbg !25
  %1584 = shufflevector <1 x float> %1583, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %1585 = extractelement <8 x float> %1458, i64 4, !dbg !11
  %1586 = extractelement <8 x float> %1458, i64 5, !dbg !11
  %1587 = extractelement <8 x float> %1458, i64 6, !dbg !11
  %1588 = extractelement <8 x float> %1458, i64 7, !dbg !11
  %1589 = select i1 %.not, float %1586, float %1585, !dbg !11
  %1590 = bitcast float %1589 to i32, !dbg !11
  %1591 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1590), !dbg !11
  %1592 = bitcast i32 %1591 to float, !dbg !11
  %1593 = select i1 %.not, float %1585, float %1592, !dbg !11
  %1594 = select i1 %.not, float %1592, float %1586, !dbg !11
  %1595 = select i1 %.not, float %1588, float %1587, !dbg !11
  %1596 = bitcast float %1595 to i32, !dbg !11
  %1597 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1596), !dbg !11
  %1598 = bitcast i32 %1597 to float, !dbg !11
  %1599 = select i1 %.not, float %1587, float %1598, !dbg !11
  %1600 = select i1 %.not, float %1598, float %1588, !dbg !11
  %1601 = select i1 %.not36, float %1599, float %1593, !dbg !11
  %1602 = bitcast float %1601 to i32, !dbg !11
  %1603 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1602), !dbg !11
  %1604 = bitcast i32 %1603 to float, !dbg !11
  %1605 = select i1 %.not36, float %1600, float %1594, !dbg !11
  %1606 = bitcast float %1605 to i32, !dbg !11
  %1607 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1606), !dbg !11
  %1608 = bitcast i32 %1607 to float, !dbg !11
  %1609 = insertelement <4 x float> poison, float %1593, i64 0, !dbg !11
  %1610 = insertelement <4 x float> %1609, float %1594, i64 1, !dbg !11
  %1611 = insertelement <4 x float> %1610, float %1604, i64 2, !dbg !11
  %1612 = insertelement <4 x float> %1611, float %1608, i64 3, !dbg !11
  %1613 = insertelement <4 x float> poison, float %1604, i64 0, !dbg !11
  %1614 = insertelement <4 x float> %1613, float %1608, i64 1, !dbg !11
  %1615 = insertelement <4 x float> %1614, float %1599, i64 2, !dbg !11
  %1616 = insertelement <4 x float> %1615, float %1600, i64 3, !dbg !11
  %1617 = select <4 x i1> %1234, <4 x float> %1612, <4 x float> %1616, !dbg !11
  %1618 = extractelement <8 x float> %1492, i64 4, !dbg !11
  %1619 = extractelement <8 x float> %1492, i64 5, !dbg !11
  %1620 = extractelement <8 x float> %1492, i64 6, !dbg !11
  %1621 = extractelement <8 x float> %1492, i64 7, !dbg !11
  %1622 = select i1 %.not, float %1619, float %1618, !dbg !11
  %1623 = bitcast float %1622 to i32, !dbg !11
  %1624 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1623), !dbg !11
  %1625 = bitcast i32 %1624 to float, !dbg !11
  %1626 = select i1 %.not, float %1618, float %1625, !dbg !11
  %1627 = select i1 %.not, float %1625, float %1619, !dbg !11
  %1628 = select i1 %.not, float %1621, float %1620, !dbg !11
  %1629 = bitcast float %1628 to i32, !dbg !11
  %1630 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1629), !dbg !11
  %1631 = bitcast i32 %1630 to float, !dbg !11
  %1632 = select i1 %.not, float %1620, float %1631, !dbg !11
  %1633 = select i1 %.not, float %1631, float %1621, !dbg !11
  %1634 = select i1 %.not36, float %1632, float %1626, !dbg !11
  %1635 = bitcast float %1634 to i32, !dbg !11
  %1636 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1635), !dbg !11
  %1637 = bitcast i32 %1636 to float, !dbg !11
  %1638 = select i1 %.not36, float %1633, float %1627, !dbg !11
  %1639 = bitcast float %1638 to i32, !dbg !11
  %1640 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1639), !dbg !11
  %1641 = bitcast i32 %1640 to float, !dbg !11
  %1642 = insertelement <4 x float> poison, float %1626, i64 0, !dbg !11
  %1643 = insertelement <4 x float> %1642, float %1627, i64 1, !dbg !11
  %1644 = insertelement <4 x float> %1643, float %1637, i64 2, !dbg !11
  %1645 = insertelement <4 x float> %1644, float %1641, i64 3, !dbg !11
  %1646 = insertelement <4 x float> poison, float %1637, i64 0, !dbg !11
  %1647 = insertelement <4 x float> %1646, float %1641, i64 1, !dbg !11
  %1648 = insertelement <4 x float> %1647, float %1632, i64 2, !dbg !11
  %1649 = insertelement <4 x float> %1648, float %1633, i64 3, !dbg !11
  %1650 = select <4 x i1> %1234, <4 x float> %1645, <4 x float> %1649, !dbg !11
  %1651 = extractelement <8 x float> %1526, i64 4, !dbg !11
  %1652 = extractelement <8 x float> %1526, i64 5, !dbg !11
  %1653 = extractelement <8 x float> %1526, i64 6, !dbg !11
  %1654 = extractelement <8 x float> %1526, i64 7, !dbg !11
  %1655 = select i1 %.not, float %1652, float %1651, !dbg !11
  %1656 = bitcast float %1655 to i32, !dbg !11
  %1657 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1656), !dbg !11
  %1658 = bitcast i32 %1657 to float, !dbg !11
  %1659 = select i1 %.not, float %1651, float %1658, !dbg !11
  %1660 = select i1 %.not, float %1658, float %1652, !dbg !11
  %1661 = select i1 %.not, float %1654, float %1653, !dbg !11
  %1662 = bitcast float %1661 to i32, !dbg !11
  %1663 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1662), !dbg !11
  %1664 = bitcast i32 %1663 to float, !dbg !11
  %1665 = select i1 %.not, float %1653, float %1664, !dbg !11
  %1666 = select i1 %.not, float %1664, float %1654, !dbg !11
  %1667 = select i1 %.not36, float %1665, float %1659, !dbg !11
  %1668 = bitcast float %1667 to i32, !dbg !11
  %1669 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1668), !dbg !11
  %1670 = bitcast i32 %1669 to float, !dbg !11
  %1671 = select i1 %.not36, float %1666, float %1660, !dbg !11
  %1672 = bitcast float %1671 to i32, !dbg !11
  %1673 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1672), !dbg !11
  %1674 = bitcast i32 %1673 to float, !dbg !11
  %1675 = insertelement <4 x float> poison, float %1659, i64 0, !dbg !11
  %1676 = insertelement <4 x float> %1675, float %1660, i64 1, !dbg !11
  %1677 = insertelement <4 x float> %1676, float %1670, i64 2, !dbg !11
  %1678 = insertelement <4 x float> %1677, float %1674, i64 3, !dbg !11
  %1679 = insertelement <4 x float> poison, float %1670, i64 0, !dbg !11
  %1680 = insertelement <4 x float> %1679, float %1674, i64 1, !dbg !11
  %1681 = insertelement <4 x float> %1680, float %1665, i64 2, !dbg !11
  %1682 = insertelement <4 x float> %1681, float %1666, i64 3, !dbg !11
  %1683 = select <4 x i1> %1234, <4 x float> %1678, <4 x float> %1682, !dbg !11
  %1684 = extractelement <8 x float> %1560, i64 4, !dbg !11
  %1685 = extractelement <8 x float> %1560, i64 5, !dbg !11
  %1686 = extractelement <8 x float> %1560, i64 6, !dbg !11
  %1687 = extractelement <8 x float> %1560, i64 7, !dbg !11
  %1688 = select i1 %.not, float %1685, float %1684, !dbg !11
  %1689 = bitcast float %1688 to i32, !dbg !11
  %1690 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1689), !dbg !11
  %1691 = bitcast i32 %1690 to float, !dbg !11
  %1692 = select i1 %.not, float %1684, float %1691, !dbg !11
  %1693 = select i1 %.not, float %1691, float %1685, !dbg !11
  %1694 = select i1 %.not, float %1687, float %1686, !dbg !11
  %1695 = bitcast float %1694 to i32, !dbg !11
  %1696 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1210, i32 %1695), !dbg !11
  %1697 = bitcast i32 %1696 to float, !dbg !11
  %1698 = select i1 %.not, float %1686, float %1697, !dbg !11
  %1699 = select i1 %.not, float %1697, float %1687, !dbg !11
  %1700 = select i1 %.not36, float %1698, float %1692, !dbg !11
  %1701 = bitcast float %1700 to i32, !dbg !11
  %1702 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1701), !dbg !11
  %1703 = select i1 %.not36, float %1699, float %1693, !dbg !11
  %1704 = bitcast float %1703 to i32, !dbg !11
  %1705 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1225, i32 %1704), !dbg !11
  %1706 = bitcast i32 %1702 to float, !dbg !11
  %1707 = bitcast i32 %1705 to float, !dbg !11
  %1708 = insertelement <4 x float> poison, float %1692, i64 0, !dbg !11
  %1709 = insertelement <4 x float> %1708, float %1693, i64 1, !dbg !11
  %1710 = insertelement <4 x float> %1709, float %1706, i64 2, !dbg !11
  %1711 = insertelement <4 x float> %1710, float %1707, i64 3, !dbg !11
  %1712 = insertelement <4 x float> poison, float %1706, i64 0, !dbg !11
  %1713 = insertelement <4 x float> %1712, float %1707, i64 1, !dbg !11
  %1714 = insertelement <4 x float> %1713, float %1698, i64 2, !dbg !11
  %1715 = insertelement <4 x float> %1714, float %1699, i64 3, !dbg !11
  %1716 = select <4 x i1> %1234, <4 x float> %1711, <4 x float> %1715, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1584, !dbg !25
  %normalized = fmul <4 x float> %row.reciprocal, %1716, !dbg !25
  %1717 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !26
  %1718 = extractelement <4 x half> %1717, i64 3, !dbg !26
  %1719 = extractelement <4 x half> %1717, i64 2, !dbg !26
  %1720 = extractelement <4 x half> %1717, i64 1, !dbg !26
  %1721 = extractelement <4 x half> %1717, i64 0, !dbg !26
  %1722 = bitcast i32 %1579 to float, !dbg !11
  %1723 = bitcast i32 %1582 to float, !dbg !11
  %1724 = insertelement <4 x float> poison, float %1569, i64 0, !dbg !11
  %1725 = insertelement <4 x float> %1724, float %1570, i64 1, !dbg !11
  %1726 = insertelement <4 x float> %1725, float %1722, i64 2, !dbg !11
  %1727 = insertelement <4 x float> %1726, float %1723, i64 3, !dbg !11
  %1728 = insertelement <4 x float> poison, float %1722, i64 0, !dbg !11
  %1729 = insertelement <4 x float> %1728, float %1723, i64 1, !dbg !11
  %1730 = insertelement <4 x float> %1729, float %1575, i64 2, !dbg !11
  %1731 = insertelement <4 x float> %1730, float %1576, i64 3, !dbg !11
  %1732 = select <4 x i1> %1234, <4 x float> %1727, <4 x float> %1731, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %1732, !dbg !25
  %1733 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !26
  %1734 = extractelement <4 x half> %1733, i64 3, !dbg !26
  %1735 = extractelement <4 x half> %1733, i64 2, !dbg !26
  %1736 = extractelement <4 x half> %1733, i64 1, !dbg !26
  %1737 = extractelement <4 x half> %1733, i64 0, !dbg !26
  %1738 = bitcast i32 %1454 to float, !dbg !11
  %1739 = bitcast i32 %1457 to float, !dbg !11
  %1740 = insertelement <4 x float> poison, float %1444, i64 0, !dbg !11
  %1741 = insertelement <4 x float> %1740, float %1445, i64 1, !dbg !11
  %1742 = insertelement <4 x float> %1741, float %1738, i64 2, !dbg !11
  %1743 = insertelement <4 x float> %1742, float %1739, i64 3, !dbg !11
  %1744 = insertelement <4 x float> poison, float %1738, i64 0, !dbg !11
  %1745 = insertelement <4 x float> %1744, float %1739, i64 1, !dbg !11
  %1746 = insertelement <4 x float> %1745, float %1450, i64 2, !dbg !11
  %1747 = insertelement <4 x float> %1746, float %1451, i64 3, !dbg !11
  %1748 = select <4 x i1> %1234, <4 x float> %1743, <4 x float> %1747, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1336, !dbg !25
  %normalized106 = fmul <4 x float> %row.reciprocal105, %1748, !dbg !25
  %1749 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !26
  %1750 = extractelement <4 x half> %1749, i64 3, !dbg !26
  %1751 = extractelement <4 x half> %1749, i64 2, !dbg !26
  %1752 = extractelement <4 x half> %1749, i64 1, !dbg !26
  %1753 = extractelement <4 x half> %1749, i64 0, !dbg !26
  %1754 = bitcast i32 %1331 to float, !dbg !11
  %1755 = bitcast i32 %1334 to float, !dbg !11
  %1756 = insertelement <4 x float> poison, float %1321, i64 0, !dbg !11
  %1757 = insertelement <4 x float> %1756, float %1322, i64 1, !dbg !11
  %1758 = insertelement <4 x float> %1757, float %1754, i64 2, !dbg !11
  %1759 = insertelement <4 x float> %1758, float %1755, i64 3, !dbg !11
  %1760 = insertelement <4 x float> poison, float %1754, i64 0, !dbg !11
  %1761 = insertelement <4 x float> %1760, float %1755, i64 1, !dbg !11
  %1762 = insertelement <4 x float> %1761, float %1327, i64 2, !dbg !11
  %1763 = insertelement <4 x float> %1762, float %1328, i64 3, !dbg !11
  %1764 = select <4 x i1> %1234, <4 x float> %1759, <4 x float> %1763, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %1764, !dbg !25
  %1765 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !26
  %1766 = extractelement <4 x half> %1765, i64 3, !dbg !26
  %1767 = extractelement <4 x half> %1765, i64 2, !dbg !26
  %1768 = extractelement <4 x half> %1765, i64 1, !dbg !26
  %1769 = extractelement <4 x half> %1765, i64 0, !dbg !26
  %address.thread999 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !27, !range !9, !frisk.register_lifetimes !5
  %address.local1000 = zext nneg i32 %address.thread999 to i64, !dbg !8
  %1770 = lshr i64 %address.local1000, 2, !dbg !27
  %address.local1003 = and i64 %1770, 48, !dbg !27
  %address.local1005 = and i64 %address.local1000, 240, !dbg !27
  %address.local1004 = add i64 %26, %address.local1000, !dbg !27
  %address.local1006 = sub i64 %address.local1004, %address.local1005, !dbg !27
  %address.local1007 = add i64 %address.local1006, %address.local1003, !dbg !27
  %1771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !27
  %address.local1010.idx = shl nuw nsw i64 %address.local1007, 8, !dbg !27
  %address.local1010 = getelementptr inbounds nuw i8, ptr addrspace(1) %1771, i64 %address.local1010.idx, !dbg !27
  %address.local1012 = and i64 %1770, 12, !dbg !10
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1012, !dbg !27
  store half %1769, ptr addrspace(1) %address.local1013, align 2, !dbg !27
  %address.local1015 = or disjoint i64 %address.local1012, 1, !dbg !27
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1015, !dbg !27
  store half %1768, ptr addrspace(1) %address.local1016, align 2, !dbg !27
  %address.local1018 = or disjoint i64 %address.local1012, 2, !dbg !27
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1018, !dbg !27
  store half %1767, ptr addrspace(1) %address.local1019, align 2, !dbg !27
  %address.local1021 = or disjoint i64 %address.local1012, 3, !dbg !27
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1021, !dbg !27
  store half %1766, ptr addrspace(1) %address.local1022, align 2, !dbg !27
  %address.local1024 = or disjoint i64 %address.local1012, 16, !dbg !27
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1024, !dbg !27
  store half %1753, ptr addrspace(1) %address.local1025, align 2, !dbg !27
  %address.local1027 = or disjoint i64 %address.local1012, 17, !dbg !27
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1027, !dbg !27
  store half %1752, ptr addrspace(1) %address.local1028, align 2, !dbg !27
  %address.local1030 = or disjoint i64 %address.local1012, 18, !dbg !27
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1030, !dbg !27
  store half %1751, ptr addrspace(1) %address.local1031, align 2, !dbg !27
  %address.local1033 = or disjoint i64 %address.local1012, 19, !dbg !27
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1033, !dbg !27
  store half %1750, ptr addrspace(1) %address.local1034, align 2, !dbg !27
  %address.local1038 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1010, i64 16384, !dbg !27
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1012, !dbg !27
  store half %1737, ptr addrspace(1) %address.local1039, align 2, !dbg !27
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1015, !dbg !27
  store half %1736, ptr addrspace(1) %address.local1041, align 2, !dbg !27
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1018, !dbg !27
  store half %1735, ptr addrspace(1) %address.local1043, align 2, !dbg !27
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1021, !dbg !27
  store half %1734, ptr addrspace(1) %address.local1045, align 2, !dbg !27
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1024, !dbg !27
  store half %1721, ptr addrspace(1) %address.local1047, align 2, !dbg !27
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1027, !dbg !27
  store half %1720, ptr addrspace(1) %address.local1049, align 2, !dbg !27
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1030, !dbg !27
  store half %1719, ptr addrspace(1) %address.local1051, align 2, !dbg !27
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1033, !dbg !27
  store half %1718, ptr addrspace(1) %address.local1053, align 2, !dbg !27
  %normalized109 = fmul <4 x float> %row.reciprocal, %1683, !dbg !25
  %1772 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !26
  %1773 = extractelement <4 x half> %1772, i64 3, !dbg !26
  %1774 = extractelement <4 x half> %1772, i64 2, !dbg !26
  %1775 = extractelement <4 x half> %1772, i64 1, !dbg !26
  %1776 = extractelement <4 x half> %1772, i64 0, !dbg !26
  %normalized110 = fmul <4 x float> %row.reciprocal, %1559, !dbg !25
  %1777 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !26
  %1778 = extractelement <4 x half> %1777, i64 3, !dbg !26
  %1779 = extractelement <4 x half> %1777, i64 2, !dbg !26
  %1780 = extractelement <4 x half> %1777, i64 1, !dbg !26
  %1781 = extractelement <4 x half> %1777, i64 0, !dbg !26
  %normalized112 = fmul <4 x float> %row.reciprocal105, %1435, !dbg !25
  %1782 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !26
  %1783 = extractelement <4 x half> %1782, i64 3, !dbg !26
  %1784 = extractelement <4 x half> %1782, i64 2, !dbg !26
  %1785 = extractelement <4 x half> %1782, i64 1, !dbg !26
  %1786 = extractelement <4 x half> %1782, i64 0, !dbg !26
  %normalized113 = fmul <4 x float> %row.reciprocal105, %1311, !dbg !25
  %1787 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !26
  %1788 = extractelement <4 x half> %1787, i64 3, !dbg !26
  %1789 = extractelement <4 x half> %1787, i64 2, !dbg !26
  %1790 = extractelement <4 x half> %1787, i64 1, !dbg !26
  %1791 = extractelement <4 x half> %1787, i64 0, !dbg !26
  %address.local1068 = or disjoint i64 %address.local1012, 32, !dbg !27
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1068, !dbg !27
  store half %1791, ptr addrspace(1) %address.local1069, align 2, !dbg !27
  %address.local1071 = or disjoint i64 %address.local1012, 33, !dbg !27
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1071, !dbg !27
  store half %1790, ptr addrspace(1) %address.local1072, align 2, !dbg !27
  %address.local1074 = or disjoint i64 %address.local1012, 34, !dbg !27
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1074, !dbg !27
  store half %1789, ptr addrspace(1) %address.local1075, align 2, !dbg !27
  %address.local1077 = or disjoint i64 %address.local1012, 35, !dbg !27
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1077, !dbg !27
  store half %1788, ptr addrspace(1) %address.local1078, align 2, !dbg !27
  %address.local1080 = or disjoint i64 %address.local1012, 48, !dbg !27
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1080, !dbg !27
  store half %1786, ptr addrspace(1) %address.local1081, align 2, !dbg !27
  %address.local1083 = or disjoint i64 %address.local1012, 49, !dbg !27
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1083, !dbg !27
  store half %1785, ptr addrspace(1) %address.local1084, align 2, !dbg !27
  %address.local1086 = or disjoint i64 %address.local1012, 50, !dbg !27
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1086, !dbg !27
  store half %1784, ptr addrspace(1) %address.local1087, align 2, !dbg !27
  %address.local1089 = or i64 %1770, 51, !dbg !27
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1089, !dbg !27
  store half %1783, ptr addrspace(1) %address.local1090, align 2, !dbg !27
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1068, !dbg !27
  store half %1781, ptr addrspace(1) %address.local1095, align 2, !dbg !27
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1071, !dbg !27
  store half %1780, ptr addrspace(1) %address.local1097, align 2, !dbg !27
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1074, !dbg !27
  store half %1779, ptr addrspace(1) %address.local1099, align 2, !dbg !27
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1077, !dbg !27
  store half %1778, ptr addrspace(1) %address.local1101, align 2, !dbg !27
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1080, !dbg !27
  store half %1776, ptr addrspace(1) %address.local1103, align 2, !dbg !27
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1083, !dbg !27
  store half %1775, ptr addrspace(1) %address.local1105, align 2, !dbg !27
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1086, !dbg !27
  store half %1774, ptr addrspace(1) %address.local1107, align 2, !dbg !27
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1089, !dbg !27
  store half %1773, ptr addrspace(1) %address.local1109, align 2, !dbg !27
  %normalized115 = fmul <4 x float> %row.reciprocal, %1650, !dbg !25
  %1792 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !26
  %1793 = extractelement <4 x half> %1792, i64 3, !dbg !26
  %1794 = extractelement <4 x half> %1792, i64 2, !dbg !26
  %1795 = extractelement <4 x half> %1792, i64 1, !dbg !26
  %1796 = extractelement <4 x half> %1792, i64 0, !dbg !26
  %normalized116 = fmul <4 x float> %row.reciprocal, %1525, !dbg !25
  %1797 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !26
  %1798 = extractelement <4 x half> %1797, i64 3, !dbg !26
  %1799 = extractelement <4 x half> %1797, i64 2, !dbg !26
  %1800 = extractelement <4 x half> %1797, i64 1, !dbg !26
  %1801 = extractelement <4 x half> %1797, i64 0, !dbg !26
  %normalized118 = fmul <4 x float> %row.reciprocal105, %1402, !dbg !25
  %1802 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !26
  %1803 = extractelement <4 x half> %1802, i64 3, !dbg !26
  %1804 = extractelement <4 x half> %1802, i64 2, !dbg !26
  %1805 = extractelement <4 x half> %1802, i64 1, !dbg !26
  %1806 = extractelement <4 x half> %1802, i64 0, !dbg !26
  %normalized119 = fmul <4 x float> %row.reciprocal105, %1277, !dbg !25
  %1807 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !26
  %1808 = extractelement <4 x half> %1807, i64 3, !dbg !26
  %1809 = extractelement <4 x half> %1807, i64 2, !dbg !26
  %1810 = extractelement <4 x half> %1807, i64 1, !dbg !26
  %1811 = extractelement <4 x half> %1807, i64 0, !dbg !26
  %address.local1124 = or disjoint i64 %address.local1012, 64, !dbg !27
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1124, !dbg !27
  store half %1811, ptr addrspace(1) %address.local1125, align 2, !dbg !27
  %address.local1127 = or disjoint i64 %address.local1012, 65, !dbg !27
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1127, !dbg !27
  store half %1810, ptr addrspace(1) %address.local1128, align 2, !dbg !27
  %address.local1130 = or disjoint i64 %address.local1012, 66, !dbg !27
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1130, !dbg !27
  store half %1809, ptr addrspace(1) %address.local1131, align 2, !dbg !27
  %address.local1133 = or disjoint i64 %address.local1012, 67, !dbg !27
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1133, !dbg !27
  store half %1808, ptr addrspace(1) %address.local1134, align 2, !dbg !27
  %address.local1136 = or disjoint i64 %address.local1012, 80, !dbg !27
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1136, !dbg !27
  store half %1806, ptr addrspace(1) %address.local1137, align 2, !dbg !27
  %address.local1139 = or disjoint i64 %address.local1012, 81, !dbg !27
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1139, !dbg !27
  store half %1805, ptr addrspace(1) %address.local1140, align 2, !dbg !27
  %address.local1142 = or disjoint i64 %address.local1012, 82, !dbg !27
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1142, !dbg !27
  store half %1804, ptr addrspace(1) %address.local1143, align 2, !dbg !27
  %address.local1145 = or disjoint i64 %address.local1012, 83, !dbg !27
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1145, !dbg !27
  store half %1803, ptr addrspace(1) %address.local1146, align 2, !dbg !27
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1124, !dbg !27
  store half %1801, ptr addrspace(1) %address.local1151, align 2, !dbg !27
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1127, !dbg !27
  store half %1800, ptr addrspace(1) %address.local1153, align 2, !dbg !27
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1130, !dbg !27
  store half %1799, ptr addrspace(1) %address.local1155, align 2, !dbg !27
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1133, !dbg !27
  store half %1798, ptr addrspace(1) %address.local1157, align 2, !dbg !27
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1136, !dbg !27
  store half %1796, ptr addrspace(1) %address.local1159, align 2, !dbg !27
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1139, !dbg !27
  store half %1795, ptr addrspace(1) %address.local1161, align 2, !dbg !27
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1142, !dbg !27
  store half %1794, ptr addrspace(1) %address.local1163, align 2, !dbg !27
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1145, !dbg !27
  store half %1793, ptr addrspace(1) %address.local1165, align 2, !dbg !27
  %normalized121 = fmul <4 x float> %row.reciprocal, %1617, !dbg !25
  %1812 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !26
  %1813 = extractelement <4 x half> %1812, i64 3, !dbg !26
  %1814 = extractelement <4 x half> %1812, i64 2, !dbg !26
  %1815 = extractelement <4 x half> %1812, i64 1, !dbg !26
  %1816 = extractelement <4 x half> %1812, i64 0, !dbg !26
  %normalized122 = fmul <4 x float> %row.reciprocal, %1491, !dbg !25
  %1817 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !26
  %1818 = extractelement <4 x half> %1817, i64 3, !dbg !26
  %1819 = extractelement <4 x half> %1817, i64 2, !dbg !26
  %1820 = extractelement <4 x half> %1817, i64 1, !dbg !26
  %1821 = extractelement <4 x half> %1817, i64 0, !dbg !26
  %normalized124 = fmul <4 x float> %row.reciprocal105, %1369, !dbg !25
  %1822 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !26
  %1823 = extractelement <4 x half> %1822, i64 3, !dbg !26
  %1824 = extractelement <4 x half> %1822, i64 2, !dbg !26
  %1825 = extractelement <4 x half> %1822, i64 1, !dbg !26
  %1826 = extractelement <4 x half> %1822, i64 0, !dbg !26
  %normalized125 = fmul <4 x float> %row.reciprocal105, %1243, !dbg !25
  %1827 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !26
  %1828 = extractelement <4 x half> %1827, i64 3, !dbg !26
  %1829 = extractelement <4 x half> %1827, i64 2, !dbg !26
  %1830 = extractelement <4 x half> %1827, i64 1, !dbg !26
  %1831 = extractelement <4 x half> %1827, i64 0, !dbg !26
  %address.local1180 = or disjoint i64 %address.local1012, 96, !dbg !27
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1180, !dbg !27
  store half %1831, ptr addrspace(1) %address.local1181, align 2, !dbg !27
  %address.local1183 = or disjoint i64 %address.local1012, 97, !dbg !27
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1183, !dbg !27
  store half %1830, ptr addrspace(1) %address.local1184, align 2, !dbg !27
  %address.local1186 = or disjoint i64 %address.local1012, 98, !dbg !27
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1186, !dbg !27
  store half %1829, ptr addrspace(1) %address.local1187, align 2, !dbg !27
  %address.local1189 = or disjoint i64 %address.local1012, 99, !dbg !27
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1189, !dbg !27
  store half %1828, ptr addrspace(1) %address.local1190, align 2, !dbg !27
  %address.local1192 = or disjoint i64 %address.local1012, 112, !dbg !27
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1192, !dbg !27
  store half %1826, ptr addrspace(1) %address.local1193, align 2, !dbg !27
  %address.local1195 = or disjoint i64 %address.local1012, 113, !dbg !27
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1195, !dbg !27
  store half %1825, ptr addrspace(1) %address.local1196, align 2, !dbg !27
  %address.local1198 = or disjoint i64 %address.local1012, 114, !dbg !27
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1198, !dbg !27
  store half %1824, ptr addrspace(1) %address.local1199, align 2, !dbg !27
  %address.local1201 = or i64 %1770, 115, !dbg !27
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1201, !dbg !27
  store half %1823, ptr addrspace(1) %address.local1202, align 2, !dbg !27
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1180, !dbg !27
  store half %1821, ptr addrspace(1) %address.local1207, align 2, !dbg !27
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1183, !dbg !27
  store half %1820, ptr addrspace(1) %address.local1209, align 2, !dbg !27
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1186, !dbg !27
  store half %1819, ptr addrspace(1) %address.local1211, align 2, !dbg !27
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1189, !dbg !27
  store half %1818, ptr addrspace(1) %address.local1213, align 2, !dbg !27
  %address.local1215 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1192, !dbg !27
  store half %1816, ptr addrspace(1) %address.local1215, align 2, !dbg !27
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1195, !dbg !27
  store half %1815, ptr addrspace(1) %address.local1217, align 2, !dbg !27
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1198, !dbg !27
  store half %1814, ptr addrspace(1) %address.local1219, align 2, !dbg !27
  %address.local1221 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1201, !dbg !27
  store half %1813, ptr addrspace(1) %address.local1221, align 2, !dbg !27
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
