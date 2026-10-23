; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/q_bk_retention_sweep/baseline_16words.ll'
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
  %dma.buffer.offset1682 = trunc i64 %33 to i32, !dbg !8
  %34 = insertelement <2 x i64> poison, i64 %30, i64 0, !dbg !8
  %35 = bitcast <2 x i64> %34 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1683 = shufflevector <4 x i32> %35, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
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
  %dma.buffer.offset1679 = trunc i64 %38 to i32, !dbg !8
  %39 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %40 = sext i32 %39 to i64, !dbg !7
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !8
  %.idx3 = shl nuw nsw i64 %28, 1, !dbg !8
  %41 = add i64 %.idx3, %.idx, !dbg !8
  %address.local365.idx = shl nuw nsw i64 %address.local348, 1, !dbg !8
  %42 = or disjoint i64 %address.local365.idx, %41, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %dma.buffer.offset1676 = or disjoint i32 %43, 64, !dbg !8
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !8
  %44 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local341, i64 0, !dbg !8
  %45 = insertelement <4 x ptr addrspace(3)> %44, ptr addrspace(3) %address.local353, i64 1, !dbg !8
  %46 = insertelement <4 x ptr addrspace(3)> %45, ptr addrspace(3) %address.local362, i64 2, !dbg !8
  %47 = insertelement <4 x ptr addrspace(3)> %46, ptr addrspace(3) %address.local366, i64 3, !dbg !8
  %48 = ptrtoint <4 x ptr addrspace(3)> %47 to <4 x i64>, !dbg !8
  %49 = trunc <4 x i64> %48 to <4 x i32>, !dbg !8
  %50 = extractelement <4 x i32> %49, i64 0, !dbg !8
  %51 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %dma.buffer.lds1684 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %51), !dbg !8
  %52 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1684, i32 %dma.buffer.offset1682, <4 x i32> %dma.buffer.rsrc1683) #9, !dbg !8
  %53 = extractelement <4 x i32> %49, i64 1, !dbg !8
  %54 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %dma.buffer.lds1681 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %54), !dbg !8
  %55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1681, i32 %dma.buffer.offset1679, <4 x i32> %dma.buffer.rsrc1683) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %56 = extractelement <4 x i32> %49, i64 2, !dbg !8
  %57 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %56), !dbg !8
  %dma.buffer.lds1678 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %57), !dbg !8
  %58 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1678, i32 %dma.buffer.offset1676, <4 x i32> %dma.buffer.rsrc1683) #9, !dbg !8
  %59 = extractelement <4 x i32> %49, i64 3, !dbg !8
  %60 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %59), !dbg !8
  %address.local369.idx = shl nuw nsw i64 %address.local360, 1, !dbg !8
  %61 = or disjoint i64 %address.local369.idx, %41, !dbg !8
  %62 = trunc i64 %61 to i32, !dbg !8
  %dma.buffer.offset1673 = or disjoint i32 %62, 64, !dbg !8
  %dma.buffer.lds1675 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1675, i32 %dma.buffer.offset1673, <4 x i32> %dma.buffer.rsrc1683) #9, !dbg !8
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
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !10
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !10
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !10
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !10
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !10
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !10
  %75 = fmul <4 x half> %72, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !10
  %77 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !10
  %78 = fmul <4 x half> %74, splat (half 1.275630e-01), !dbg !10
  %79 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %80 = or disjoint i64 %24, 127, !dbg !12
  %81 = add i64 %24, 128, !dbg !12
  %82 = icmp uge i64 %81, %80, !dbg !12
  %83 = sub nsw i64 32, %14, !dbg !13
  %84 = shl i64 %16, 7, !dbg !14
  %85 = and i64 %84, 3968, !dbg !14
  %86 = shl nsw i64 %40, 12, !dbg !14
  %87 = or disjoint i64 %85, %86, !dbg !14
  %88 = shl nuw nsw i32 %29, 2, !dbg !15
  %89 = and i32 %88, 12, !dbg !15
  %90 = and i32 %29, 1, !dbg !15
  %91 = or disjoint i32 %89, %90, !dbg !15
  %92 = trunc nuw nsw i32 %91 to i16, !dbg !15
  %.lhs.trunc = add nuw nsw i16 %92, 1, !dbg !15
  %93 = lshr i16 %.lhs.trunc, 1, !dbg !15
  %.sext = zext nneg i16 %93 to i64, !dbg !15
  %94 = icmp ult i64 %10, 1024, !dbg !11
  br i1 %94, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !11

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre106 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre108 = add nuw nsw i32 %.pre106, 64, !dbg !11
  %.pre110 = xor i32 %.pre106, 16, !dbg !11
  %.pre112 = and i32 %.pre108, 192, !dbg !11
  %.pre114 = xor i32 %.pre106, 32, !dbg !11
  br label %._crit_edge, !dbg !11

.lr.ph:                                           ; preds = %4
  %95 = zext nneg i32 %29 to i64, !dbg !8
  %96 = ptrtoint ptr addrspace(1) %1 to i64
  %97 = insertelement <2 x i64> poison, i64 %96, i64 0
  %98 = bitcast <2 x i64> %97 to <4 x i32>
  %dma.buffer.rsrc1671 = shufflevector <4 x i32> %98, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %99 = ptrtoint ptr addrspace(1) %2 to i64
  %100 = insertelement <2 x i64> poison, i64 %99, i64 0
  %101 = bitcast <2 x i64> %100 to <4 x i32>
  %dma.buffer.rsrc1635 = shufflevector <4 x i32> %101, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %102 = lshr i64 %95, 2
  %103 = and i64 %102, 48
  %104 = and i32 %29, 240
  %index.compact = zext nneg i32 %104 to i64
  %105 = sub nsw i64 %95, %index.compact
  %106 = add nsw i64 %105, %103
  %107 = add nsw i64 %106, %24
  %108 = and i64 %95, 48
  %109 = add i64 %107, 1
  %110 = lshr i64 %109, 32
  %compare.high = trunc nuw i64 %110 to i32
  %compare.low.base = trunc i64 %109 to i32
  %111 = add i64 %107, 65
  %112 = lshr i64 %111, 32
  %compare.high1346 = trunc nuw i64 %112 to i32
  %compare.low.base1350 = trunc i64 %111 to i32
  %113 = insertelement <32 x i32> poison, i32 %compare.low.base1350, i64 0, !dbg !16
  %114 = shufflevector <32 x i32> %113, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  %115 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %116 = shufflevector <32 x i32> %115, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  br label %117, !dbg !11

117:                                              ; preds = %.lr.ph, %540
  %118 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1194, %540 ]
  %119 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1190, %540 ]
  %120 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1186, %540 ]
  %121 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %883, %540 ]
  %122 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1182, %540 ]
  %123 = phi i64 [ 0, %.lr.ph ], [ %1195, %540 ]
  %124 = shl nuw nsw i64 %123, 7, !dbg !11
  %125 = lshr i64 %123, 12, !dbg !14
  %126 = add nuw nsw i64 %125, %16, !dbg !14
  %127 = lshr i64 %123, 5, !dbg !14
  %128 = and i64 %127, 127, !dbg !14
  %129 = shl i64 %126, 19, !dbg !14
  %130 = and i64 %129, 16252928, !dbg !14
  %131 = shl i64 %127, 19, !dbg !14
  %132 = shl nuw nsw i64 %123, 14, !dbg !14
  %133 = sub i64 %132, %131, !dbg !14
  %address.thread429 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !14, !range !9, !frisk.register_lifetimes !5
  %address.local430 = zext nneg i32 %address.thread429 to i64, !dbg !8
  %address.local431 = shl nuw nsw i64 %address.local430, 3, !dbg !8
  %address.local432 = and i64 %address.local431, 1536, !dbg !8
  %address.local433 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local432, !dbg !8
  %address.local436 = lshr i64 %address.local430, 4, !dbg !14
  %address.local437 = and i64 %address.local436, 8, !dbg !14
  %address.local438 = and i64 %address.local430, 123, !dbg !14
  %address.local439 = lshr i64 %address.local430, 2, !dbg !14
  %address.local440 = and i64 %address.local439, 4, !dbg !14
  %address.local441 = add nuw nsw i64 %address.local440, %address.local430, !dbg !14
  %address.local442 = and i64 %address.local441, 4, !dbg !14
  %address.local443 = or disjoint i64 %address.local442, %address.local438, !dbg !14
  %address.local444 = shl nuw nsw i64 %address.local443, 7, !dbg !14
  %address.local445 = or disjoint i64 %address.local444, %address.local437, !dbg !14
  %.idx58 = or disjoint i64 %130, %128, !dbg !14
  %134 = add i64 %.idx58, %133, !dbg !14
  %135 = add i64 %address.local445, %134, !dbg !14
  %.tr = trunc i64 %135 to i32, !dbg !14
  %dma.buffer.offset1670 = shl i32 %.tr, 1, !dbg !14
  %address.local447 = or disjoint i64 %address.local430, 256, !dbg !8
  %address.local448 = shl nuw nsw i64 %address.local447, 3, !dbg !8
  %address.local449 = and i64 %address.local448, 3584, !dbg !8
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local449, !dbg !8
  %address.local453 = lshr i64 %address.local447, 4, !dbg !14
  %address.local454 = and i64 %address.local453, 24, !dbg !14
  %address.local462 = or disjoint i64 %address.local444, %address.local454, !dbg !14
  %136 = add i64 %address.local462, %134, !dbg !14
  %.tr13 = trunc i64 %136 to i32, !dbg !14
  %dma.buffer.offset1667 = shl i32 %.tr13, 1, !dbg !14
  %137 = add nuw nsw i64 %123, 1024, !dbg !14
  %138 = lshr i64 %137, 12, !dbg !14
  %139 = add nuw nsw i64 %138, %16, !dbg !14
  %140 = add nsw i64 %127, %87, !dbg !14
  %141 = shl i64 %139, 19, !dbg !14
  %142 = and i64 %141, 16252928, !dbg !14
  %address.local464 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local432, !dbg !8
  %143 = add i64 %133, %127, !dbg !14
  %.idx1418 = add i64 %143, %87, !dbg !14
  %144 = add i64 %.idx1418, %142, !dbg !14
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local449, !dbg !8
  %145 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local433, i64 0, !dbg !8
  %146 = insertelement <4 x ptr addrspace(3)> %145, ptr addrspace(3) %address.local450, i64 1, !dbg !8
  %147 = insertelement <4 x ptr addrspace(3)> %146, ptr addrspace(3) %address.local464, i64 2, !dbg !8
  %148 = insertelement <4 x ptr addrspace(3)> %147, ptr addrspace(3) %address.local468, i64 3, !dbg !8
  %149 = ptrtoint <4 x ptr addrspace(3)> %148 to <4 x i64>, !dbg !8
  %150 = trunc <4 x i64> %149 to <4 x i32>, !dbg !8
  %151 = extractelement <4 x i32> %150, i64 0, !dbg !14
  %152 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %151), !dbg !14
  %dma.buffer.lds1672 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %152), !dbg !14
  %153 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1672, i32 %dma.buffer.offset1670, <4 x i32> %dma.buffer.rsrc1671) #9, !dbg !14
  %154 = extractelement <4 x i32> %150, i64 1, !dbg !14
  %155 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %154), !dbg !14
  %dma.buffer.lds1669 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %155), !dbg !14
  %156 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1669, i32 %dma.buffer.offset1667, <4 x i32> %dma.buffer.rsrc1671) #9, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %157 = extractelement <4 x i32> %150, i64 2, !dbg !14
  %158 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %157), !dbg !14
  %dma.buffer.lds1666 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %158), !dbg !14
  %159 = extractelement <4 x i32> %150, i64 3, !dbg !14
  %address.local476 = shl nuw nsw i64 %address.local430, 1, !dbg !15
  %address.local477 = and i64 %address.local476, 24, !dbg !15
  %address.local479 = shl i64 %address.local430, 62, !dbg !15
  %address.local480 = ashr exact i64 %address.local479, 57, !dbg !15
  %address.local481 = and i64 %address.local480, -64, !dbg !15
  %address.local484 = and i64 %address.local439, 12, !dbg !10
  %160 = and i64 %address.local430, 32, !dbg !10
  %.not26 = icmp eq i64 %160, 0, !dbg !10
  %address.local488 = select i1 %.not26, i64 0, i64 1016, !dbg !10
  %.idx27 = mul nuw nsw i64 %address.local430, 320, !dbg !15
  %161 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx27, !dbg !15
  %.idx28 = mul nsw i64 %address.local439, -1280, !dbg !15
  %162 = getelementptr i8, ptr addrspace(3) %161, i64 %.idx28, !dbg !15
  %163 = getelementptr [2 x i8], ptr addrspace(3) %162, i64 %address.local477, !dbg !15
  %164 = getelementptr [2 x i8], ptr addrspace(3) %163, i64 %address.local481, !dbg !15
  %165 = getelementptr [2 x i8], ptr addrspace(3) %164, i64 %address.local484, !dbg !15
  %address.local490 = getelementptr [2 x i8], ptr addrspace(3) %165, i64 %address.local488, !dbg !15
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local490, i64 4096, !dbg !15
  %address.local501 = shl nuw nsw i32 %address.thread429, 1, !dbg !15
  %address.local502 = and i32 %address.local501, 24, !dbg !15
  %address.local503 = shl nuw nsw i32 %address.thread429, 8, !dbg !15
  %address.local504 = and i32 %address.local503, 768, !dbg !15
  %address.local505 = or disjoint i32 %address.local502, %address.local504, !dbg !15
  %address.local506 = shl nuw nsw i32 %address.thread429, 5, !dbg !15
  %address.local507 = and i32 %address.local506, 96, !dbg !15
  %address.local508 = or disjoint i32 %address.local505, %address.local507, !dbg !15
  %address.local509 = zext nneg i32 %address.local508 to i64, !dbg !15
  %address.local510 = lshr i32 %address.thread429, 1, !dbg !15
  %address.local511 = and i32 %address.local510, 1, !dbg !15
  %address.local512 = zext nneg i32 %address.local511 to i64, !dbg !15
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local490, i64 128, !dbg !15
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local490, i64 4224, !dbg !15
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local490, i64 1024, !dbg !15
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local490, i64 5120, !dbg !15
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local490, i64 1152, !dbg !15
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local490, i64 5248, !dbg !15
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local490, i64 8192, !dbg !15
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local490, i64 12288, !dbg !15
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local490, i64 8320, !dbg !15
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local490, i64 12416, !dbg !15
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local490, i64 9216, !dbg !15
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local490, i64 13312, !dbg !15
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local490, i64 9344, !dbg !15
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local490, i64 13440, !dbg !15
  %address.local844 = shl nuw nsw i64 %address.local430, 7, !dbg !8
  %address.local845 = and i64 %address.local844, 16256, !dbg !8
  %address.local848 = or disjoint i64 %address.local845, %address.local437, !dbg !8
  %address.local849.idx = shl nuw nsw i64 %address.local848, 1, !dbg !8
  %166 = or disjoint i64 %address.local849.idx, %41, !dbg !8
  %167 = trunc i64 %166 to i32, !dbg !8
  %dma.buffer.offset1658 = or disjoint i32 %167, 128, !dbg !8
  %address.local854 = or disjoint i64 %address.local454, %address.local845, !dbg !8
  %address.local855.idx = shl nuw nsw i64 %address.local854, 1, !dbg !8
  %168 = or disjoint i64 %41, %address.local855.idx, !dbg !8
  %169 = trunc i64 %168 to i32, !dbg !8
  %dma.buffer.offset1655 = or disjoint i32 %169, 128, !dbg !8
  %170 = add nuw nsw i64 %123, 2048, !dbg !14
  %171 = lshr i64 %170, 12, !dbg !14
  %172 = add nuw nsw i64 %171, %16, !dbg !14
  %173 = insertelement <2 x i64> poison, i64 %140, i64 0, !dbg !14
  %174 = shufflevector <2 x i64> %173, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !14
  %175 = icmp slt <2 x i64> %174, <i64 -32, i64 -64>, !dbg !14
  %176 = extractelement <2 x i1> %175, i64 0, !dbg !14
  %177 = sext i1 %176 to i64, !dbg !14
  %178 = add nsw <2 x i64> %174, <i64 32, i64 64>, !dbg !14
  %179 = sub nsw <2 x i64> <i64 -33, i64 -65>, %174, !dbg !14
  %180 = select <2 x i1> %175, <2 x i64> %179, <2 x i64> %178, !dbg !14
  %181 = sdiv <2 x i64> %180, splat (i64 128), !dbg !14
  %182 = extractelement <2 x i64> %181, i64 0, !dbg !14
  %183 = xor i64 %182, %177, !dbg !14
  %184 = shl nsw i64 %183, 7, !dbg !14
  %185 = sub i64 %144, %184, !dbg !14
  %186 = add i64 %address.local445, %185, !dbg !14
  %.tr19 = trunc i64 %186 to i32, !dbg !14
  %187 = shl i32 %.tr19, 1, !dbg !14
  %dma.buffer.offset1664 = add i32 %187, 64, !dbg !14
  %188 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1666, i32 %dma.buffer.offset1664, <4 x i32> %dma.buffer.rsrc1671) #9, !dbg !14
  %189 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %159), !dbg !14
  %190 = add i64 %address.local462, %185, !dbg !14
  %.tr25 = trunc i64 %190 to i32, !dbg !14
  %191 = shl i32 %.tr25, 1, !dbg !14
  %dma.buffer.offset1661 = add i32 %191, 64, !dbg !14
  %dma.buffer.lds1663 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %189), !dbg !14
  %192 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1663, i32 %dma.buffer.offset1661, <4 x i32> %dma.buffer.rsrc1671) #9, !dbg !14
  %193 = load <4 x half>, ptr addrspace(3) %address.local490, align 8, !dbg !15
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %193, <4 x float> zeroinitializer) #8, !dbg !15
  %195 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %.sext) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local513 = add nsw i64 %address.base, %address.local512, !dbg !15
  %address.local514 = shl i64 %address.local513, 6, !dbg !15
  %address.local515 = sub i64 %address.local509, %address.local514, !dbg !15
  %196 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local515, !dbg !15
  %197 = getelementptr [2 x i8], ptr addrspace(3) %196, i64 %address.local484, !dbg !15
  %198 = getelementptr [2 x i8], ptr addrspace(3) %197, i64 %address.local488, !dbg !15
  %address.local519 = getelementptr i8, ptr addrspace(3) %198, i64 64, !dbg !15
  %199 = load <4 x half>, ptr addrspace(3) %address.local519, align 8, !dbg !15
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %195, <4 x float> %194) #8, !dbg !15
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %199, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %198, i64 4160, !dbg !15
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !15
  %203 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !15
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %202, <4 x float> %201) #8, !dbg !15
  %205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %203, <4 x float> zeroinitializer) #8, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !15
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %198, i64 192, !dbg !15
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !15
  %208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %206, <4 x float> %205) #8, !dbg !15
  %209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %207, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %198, i64 4288, !dbg !15
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !15
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !15
  %212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %210, <4 x float> %209) #8, !dbg !15
  %213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %211, <4 x float> zeroinitializer) #8, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !15
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %198, i64 1088, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !15
  %216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %214, <4 x float> %213) #8, !dbg !15
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %215, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %198, i64 5184, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !15
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !15
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %218, <4 x float> %217) #8, !dbg !15
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %219, <4 x float> zeroinitializer) #8, !dbg !15
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !15
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %198, i64 1216, !dbg !15
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !15
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %222, <4 x float> %221) #8, !dbg !15
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %198, i64 5312, !dbg !15
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !15
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %226, <4 x float> %225) #8, !dbg !15
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %193, <4 x float> zeroinitializer) #8, !dbg !15
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %195, <4 x float> %228) #8, !dbg !15
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %199, <4 x float> zeroinitializer) #8, !dbg !15
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %202, <4 x float> %230) #8, !dbg !15
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %203, <4 x float> zeroinitializer) #8, !dbg !15
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %206, <4 x float> %232) #8, !dbg !15
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %207, <4 x float> zeroinitializer) #8, !dbg !15
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %210, <4 x float> %234) #8, !dbg !15
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %211, <4 x float> zeroinitializer) #8, !dbg !15
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %214, <4 x float> %236) #8, !dbg !15
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %215, <4 x float> zeroinitializer) #8, !dbg !15
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %218, <4 x float> %238) #8, !dbg !15
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %219, <4 x float> zeroinitializer) #8, !dbg !15
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %222, <4 x float> %240) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %226, <4 x float> %242) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %244 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !15
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %244, <4 x float> %200) #8, !dbg !15
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !15
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %198, i64 8256, !dbg !15
  %247 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !15
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %246, <4 x float> %245) #8, !dbg !15
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %247, <4 x float> %204) #8, !dbg !15
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %198, i64 12352, !dbg !15
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !15
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !15
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %250, <4 x float> %249) #8, !dbg !15
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %251, <4 x float> %208) #8, !dbg !15
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !15
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %198, i64 8384, !dbg !15
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !15
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %254, <4 x float> %253) #8, !dbg !15
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %255, <4 x float> %212) #8, !dbg !15
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %198, i64 12480, !dbg !15
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !15
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !15
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %258, <4 x float> %257) #8, !dbg !15
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %259, <4 x float> %216) #8, !dbg !15
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !15
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %198, i64 9280, !dbg !15
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !15
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %262, <4 x float> %261) #8, !dbg !15
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %263, <4 x float> %220) #8, !dbg !15
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %198, i64 13376, !dbg !15
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !15
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !15
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %266, <4 x float> %265) #8, !dbg !15
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %267, <4 x float> %224) #8, !dbg !15
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !15
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %198, i64 9408, !dbg !15
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !15
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %270, <4 x float> %269) #8, !dbg !15
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %271, <4 x float> %227) #8, !dbg !15
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %198, i64 13504, !dbg !15
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !15
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %274, <4 x float> %273) #8, !dbg !15
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %244, <4 x float> %229) #8, !dbg !15
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %246, <4 x float> %276) #8, !dbg !15
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %247, <4 x float> %231) #8, !dbg !15
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %250, <4 x float> %278) #8, !dbg !15
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %251, <4 x float> %233) #8, !dbg !15
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %254, <4 x float> %280) #8, !dbg !15
  %282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %255, <4 x float> %235) #8, !dbg !15
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %258, <4 x float> %282) #8, !dbg !15
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %259, <4 x float> %237) #8, !dbg !15
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %262, <4 x float> %284) #8, !dbg !15
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %263, <4 x float> %239) #8, !dbg !15
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %266, <4 x float> %286) #8, !dbg !15
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %267, <4 x float> %241) #8, !dbg !15
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %270, <4 x float> %288) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %271, <4 x float> %243) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %274, <4 x float> %290) #8, !dbg !15
  %292 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1672, i32 %dma.buffer.offset1658, <4 x i32> %dma.buffer.rsrc1683) #9, !dbg !8
  %293 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1669, i32 %dma.buffer.offset1655, <4 x i32> %dma.buffer.rsrc1683) #9, !dbg !8
  %294 = extractelement <2 x i1> %175, i64 1, !dbg !14
  %295 = sext i1 %294 to i64, !dbg !14
  %296 = extractelement <2 x i64> %181, i64 1, !dbg !14
  %297 = xor i64 %296, %295, !dbg !14
  %298 = shl i64 %172, 19, !dbg !14
  %299 = and i64 %298, 16252928, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %300 = add i64 %.idx1418, %299, !dbg !14
  %301 = shl nsw i64 %297, 7, !dbg !14
  %302 = sub i64 %300, %301, !dbg !14
  %303 = add i64 %address.local445, %302, !dbg !14
  %.tr36 = trunc i64 %303 to i32, !dbg !14
  %304 = shl i32 %.tr36, 1, !dbg !14
  %dma.buffer.offset1652 = add i32 %304, 128, !dbg !14
  %305 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1666, i32 %dma.buffer.offset1652, <4 x i32> %dma.buffer.rsrc1671) #9, !dbg !14
  %306 = add i64 %address.local462, %302, !dbg !14
  %.tr42 = trunc i64 %306 to i32, !dbg !14
  %307 = shl i32 %.tr42, 1, !dbg !14
  %dma.buffer.offset1649 = add i32 %307, 128, !dbg !14
  %308 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1663, i32 %dma.buffer.offset1649, <4 x i32> %dma.buffer.rsrc1671) #9, !dbg !14
  %309 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !10
  %312 = fmul <4 x half> %309, splat (half 1.275630e-01), !dbg !10
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !10
  %314 = fmul <4 x half> %310, splat (half 1.275630e-01), !dbg !10
  %315 = fmul <4 x half> %311, splat (half 1.275630e-01), !dbg !10
  %316 = fmul <4 x half> %313, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !15
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %317, <4 x float> %248) #8, !dbg !15
  %319 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !15
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !15
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %319, <4 x float> %318) #8, !dbg !15
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %320, <4 x float> %252) #8, !dbg !15
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !15
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !15
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %323, <4 x float> %322) #8, !dbg !15
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %324, <4 x float> %256) #8, !dbg !15
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !15
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !15
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %327, <4 x float> %326) #8, !dbg !15
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %328, <4 x float> %260) #8, !dbg !15
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !15
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !15
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %331, <4 x float> %330) #8, !dbg !15
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %332, <4 x float> %264) #8, !dbg !15
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !15
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !15
  %337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %335, <4 x float> %334) #8, !dbg !15
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %336, <4 x float> %268) #8, !dbg !15
  %339 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !15
  %340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !15
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %339, <4 x float> %338) #8, !dbg !15
  %342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %340, <4 x float> %272) #8, !dbg !15
  %343 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !15
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !15
  %345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %343, <4 x float> %342) #8, !dbg !15
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %344, <4 x float> %275) #8, !dbg !15
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !15
  %348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %347, <4 x float> %346) #8, !dbg !15
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %317, <4 x float> %277) #8, !dbg !15
  %350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %316, <4 x half> %319, <4 x float> %349) #8, !dbg !15
  %351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %320, <4 x float> %279) #8, !dbg !15
  %352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %316, <4 x half> %323, <4 x float> %351) #8, !dbg !15
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %324, <4 x float> %281) #8, !dbg !15
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %316, <4 x half> %327, <4 x float> %353) #8, !dbg !15
  %355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %328, <4 x float> %283) #8, !dbg !15
  %356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %316, <4 x half> %331, <4 x float> %355) #8, !dbg !15
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %332, <4 x float> %285) #8, !dbg !15
  %358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %316, <4 x half> %335, <4 x float> %357) #8, !dbg !15
  %359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %336, <4 x float> %287) #8, !dbg !15
  %360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %316, <4 x half> %339, <4 x float> %359) #8, !dbg !15
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %340, <4 x float> %289) #8, !dbg !15
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %316, <4 x half> %343, <4 x float> %361) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %344, <4 x float> %291) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %316, <4 x half> %347, <4 x float> %363) #8, !dbg !15
  %dma.buffer.offset1646 = or disjoint i32 %167, 192, !dbg !8
  %365 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1672, i32 %dma.buffer.offset1646, <4 x i32> %dma.buffer.rsrc1683) #9, !dbg !8
  %dma.buffer.offset1643 = or disjoint i32 %169, 192, !dbg !8
  %366 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1669, i32 %dma.buffer.offset1643, <4 x i32> %dma.buffer.rsrc1683) #9, !dbg !8
  %367 = add nuw nsw i64 %123, 3072, !dbg !14
  %368 = lshr i64 %367, 12, !dbg !14
  %369 = add nuw nsw i64 %368, %16, !dbg !14
  %370 = add nsw i64 %140, 96, !dbg !14
  %371 = icmp slt i64 %140, -96, !dbg !14
  %372 = sub nsw i64 -97, %140, !dbg !14
  %373 = select i1 %371, i64 %372, i64 %370, !dbg !14
  %374 = sdiv i64 %373, 128, !dbg !14
  %375 = sext i1 %371 to i64, !dbg !14
  %376 = xor i64 %374, %375, !dbg !14
  %377 = shl i64 %369, 19, !dbg !14
  %378 = and i64 %377, 16252928, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %379 = add i64 %.idx1418, %378, !dbg !14
  %380 = shl nsw i64 %376, 7, !dbg !14
  %381 = sub i64 %379, %380, !dbg !14
  %382 = add i64 %address.local445, %381, !dbg !14
  %.tr50 = trunc i64 %382 to i32, !dbg !14
  %383 = shl i32 %.tr50, 1, !dbg !14
  %dma.buffer.offset1640 = add i32 %383, 192, !dbg !14
  %384 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1666, i32 %dma.buffer.offset1640, <4 x i32> %dma.buffer.rsrc1671) #9, !dbg !14
  %385 = add i64 %address.local462, %381, !dbg !14
  %.tr56 = trunc i64 %385 to i32, !dbg !14
  %386 = shl i32 %.tr56, 1, !dbg !14
  %dma.buffer.offset1637 = add i32 %386, 192, !dbg !14
  %387 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1663, i32 %dma.buffer.offset1637, <4 x i32> %dma.buffer.rsrc1671) #9, !dbg !14
  %388 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %389 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %390 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !10
  %391 = fmul <4 x half> %388, splat (half 1.275630e-01), !dbg !10
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !10
  %393 = fmul <4 x half> %389, splat (half 1.275630e-01), !dbg !10
  %394 = fmul <4 x half> %390, splat (half 1.275630e-01), !dbg !10
  %395 = fmul <4 x half> %392, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %396 = add nuw nsw i64 %127, %16, !dbg !17
  %397 = shl i64 %396, 19, !dbg !17
  %398 = and i64 %397, 16252928, !dbg !17
  %address.local917 = shl nuw nsw i64 %address.local430, 6, !dbg !17
  %address.local918 = and i64 %address.local917, 16128, !dbg !17
  %address.local920 = and i64 %address.local431, 24, !dbg !17
  %.idx5761 = add i64 %398, %133, !dbg !17
  %399 = or disjoint i64 %address.local920, %.idx5761, !dbg !17
  %400 = shl i64 %399, 1, !dbg !17
  %401 = or disjoint i64 %400, %address.local918, !dbg !17
  %dma.buffer.offset1634 = trunc i64 %401 to i32, !dbg !17
  %402 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1672, i32 %dma.buffer.offset1634, <4 x i32> %dma.buffer.rsrc1635) #9, !dbg !17
  %address.local923 = shl nuw nsw i64 %address.local447, 6, !dbg !17
  %address.local924 = and i64 %address.local923, 32512, !dbg !17
  %403 = or disjoint i64 %400, %address.local924, !dbg !17
  %dma.buffer.offset1631 = trunc i64 %403 to i32, !dbg !17
  %404 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1669, i32 %dma.buffer.offset1631, <4 x i32> %dma.buffer.rsrc1635) #9, !dbg !17
  %405 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !15
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %405, <4 x float> %321) #8, !dbg !15
  %407 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !15
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !15
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %393, <4 x half> %407, <4 x float> %406) #8, !dbg !15
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %408, <4 x float> %325) #8, !dbg !15
  %411 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !15
  %412 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !15
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %393, <4 x half> %411, <4 x float> %410) #8, !dbg !15
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %412, <4 x float> %329) #8, !dbg !15
  %415 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !15
  %416 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !15
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %393, <4 x half> %415, <4 x float> %414) #8, !dbg !15
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %416, <4 x float> %333) #8, !dbg !15
  %419 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !15
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !15
  %421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %393, <4 x half> %419, <4 x float> %418) #8, !dbg !15
  %422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %420, <4 x float> %337) #8, !dbg !15
  %423 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !15
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !15
  %425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %393, <4 x half> %423, <4 x float> %422) #8, !dbg !15
  %426 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %424, <4 x float> %341) #8, !dbg !15
  %427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !15
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !15
  %429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %393, <4 x half> %427, <4 x float> %426) #8, !dbg !15
  %430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %428, <4 x float> %345) #8, !dbg !15
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !15
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !15
  %433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %393, <4 x half> %431, <4 x float> %430) #8, !dbg !15
  %434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %432, <4 x float> %348) #8, !dbg !15
  %435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !15
  %436 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %393, <4 x half> %435, <4 x float> %434) #8, !dbg !15
  %437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %405, <4 x float> %350) #8, !dbg !15
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %407, <4 x float> %437) #8, !dbg !15
  %439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %408, <4 x float> %352) #8, !dbg !15
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %411, <4 x float> %439) #8, !dbg !15
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %412, <4 x float> %354) #8, !dbg !15
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %415, <4 x float> %441) #8, !dbg !15
  %443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %416, <4 x float> %356) #8, !dbg !15
  %444 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %419, <4 x float> %443) #8, !dbg !15
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %420, <4 x float> %358) #8, !dbg !15
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %423, <4 x float> %445) #8, !dbg !15
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %424, <4 x float> %360) #8, !dbg !15
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %427, <4 x float> %447) #8, !dbg !15
  %449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %428, <4 x float> %362) #8, !dbg !15
  %450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %431, <4 x float> %449) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %432, <4 x float> %364) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %452 = or disjoint i64 %124, 127, !dbg !12
  %453 = icmp uge i64 %24, %452, !dbg !12
  %454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %435, <4 x float> %451) #8, !dbg !15
  %455 = and i1 %82, %453, !dbg !12
  br i1 %455, label %540, label %456, !dbg !12

456:                                              ; preds = %117
  %457 = or disjoint i64 %124, %108, !dbg !12
  %458 = lshr i64 %123, 25, !dbg !16
  %compare.high1222 = trunc nuw i64 %458 to i32, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1222, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1222, !dbg !16
  %compare.low.base1223 = trunc i64 %457 to i32, !dbg !16
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
  %459 = insertelement <32 x i32> poison, i32 %compare.low.base1223, i64 0, !dbg !16
  %460 = insertelement <32 x i32> %459, i32 %compare.local.base, i64 1, !dbg !16
  %461 = insertelement <32 x i32> %460, i32 %compare.local.base1226, i64 2, !dbg !16
  %462 = insertelement <32 x i32> %461, i32 %compare.local.base1230, i64 3, !dbg !16
  %463 = insertelement <32 x i32> %462, i32 %compare.local.base1234, i64 4, !dbg !16
  %464 = insertelement <32 x i32> %463, i32 %compare.local.base1238, i64 5, !dbg !16
  %465 = insertelement <32 x i32> %464, i32 %compare.local.base1242, i64 6, !dbg !16
  %466 = insertelement <32 x i32> %465, i32 %compare.local.base1246, i64 7, !dbg !16
  %467 = insertelement <32 x i32> %466, i32 %compare.local.base1250, i64 8, !dbg !16
  %468 = insertelement <32 x i32> %467, i32 %compare.local.base1254, i64 9, !dbg !16
  %469 = insertelement <32 x i32> %468, i32 %compare.local.base1258, i64 10, !dbg !16
  %470 = insertelement <32 x i32> %469, i32 %compare.local.base1262, i64 11, !dbg !16
  %471 = insertelement <32 x i32> %470, i32 %compare.local.base1266, i64 12, !dbg !16
  %472 = insertelement <32 x i32> %471, i32 %compare.local.base1270, i64 13, !dbg !16
  %473 = insertelement <32 x i32> %472, i32 %compare.local.base1274, i64 14, !dbg !16
  %474 = insertelement <32 x i32> %473, i32 %compare.local.base1278, i64 15, !dbg !16
  %475 = insertelement <32 x i32> %474, i32 %compare.local.base1282, i64 16, !dbg !16
  %476 = insertelement <32 x i32> %475, i32 %compare.local.base1286, i64 17, !dbg !16
  %477 = insertelement <32 x i32> %476, i32 %compare.local.base1290, i64 18, !dbg !16
  %478 = insertelement <32 x i32> %477, i32 %compare.local.base1294, i64 19, !dbg !16
  %479 = insertelement <32 x i32> %478, i32 %compare.local.base1298, i64 20, !dbg !16
  %480 = insertelement <32 x i32> %479, i32 %compare.local.base1302, i64 21, !dbg !16
  %481 = insertelement <32 x i32> %480, i32 %compare.local.base1306, i64 22, !dbg !16
  %482 = insertelement <32 x i32> %481, i32 %compare.local.base1310, i64 23, !dbg !16
  %483 = insertelement <32 x i32> %482, i32 %compare.local.base1314, i64 24, !dbg !16
  %484 = insertelement <32 x i32> %483, i32 %compare.local.base1318, i64 25, !dbg !16
  %485 = insertelement <32 x i32> %484, i32 %compare.local.base1322, i64 26, !dbg !16
  %486 = insertelement <32 x i32> %485, i32 %compare.local.base1326, i64 27, !dbg !16
  %487 = insertelement <32 x i32> %486, i32 %compare.local.base1330, i64 28, !dbg !16
  %488 = insertelement <32 x i32> %487, i32 %compare.local.base1334, i64 29, !dbg !16
  %489 = insertelement <32 x i32> %488, i32 %compare.local.base1338, i64 30, !dbg !16
  %490 = insertelement <32 x i32> %489, i32 %compare.local.base1342, i64 31, !dbg !16
  %491 = or <32 x i32> %490, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %492 = icmp ugt <32 x i32> %116, %491, !dbg !16
  %493 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %494 = shufflevector <32 x i1> %493, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %495 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %496 = shufflevector <32 x i1> %495, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %497 = select <32 x i1> %494, <32 x i1> %492, <32 x i1> %496, !dbg !16
  %498 = select <32 x i1> %497, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
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
  %499 = insertelement <32 x i32> %459, i32 %compare.local.base1354, i64 1, !dbg !16
  %500 = insertelement <32 x i32> %499, i32 %compare.local.base1358, i64 2, !dbg !16
  %501 = insertelement <32 x i32> %500, i32 %compare.local.base1362, i64 3, !dbg !16
  %502 = insertelement <32 x i32> %501, i32 %compare.local.base1366, i64 4, !dbg !16
  %503 = insertelement <32 x i32> %502, i32 %compare.local.base1370, i64 5, !dbg !16
  %504 = insertelement <32 x i32> %503, i32 %compare.local.base1374, i64 6, !dbg !16
  %505 = insertelement <32 x i32> %504, i32 %compare.local.base1378, i64 7, !dbg !16
  %506 = insertelement <32 x i32> %505, i32 %compare.local.base1382, i64 8, !dbg !16
  %507 = insertelement <32 x i32> %506, i32 %compare.local.base1386, i64 9, !dbg !16
  %508 = insertelement <32 x i32> %507, i32 %compare.local.base1390, i64 10, !dbg !16
  %509 = insertelement <32 x i32> %508, i32 %compare.local.base1394, i64 11, !dbg !16
  %510 = insertelement <32 x i32> %509, i32 %compare.local.base1398, i64 12, !dbg !16
  %511 = insertelement <32 x i32> %510, i32 %compare.local.base1402, i64 13, !dbg !16
  %512 = insertelement <32 x i32> %511, i32 %compare.local.base1406, i64 14, !dbg !16
  %513 = insertelement <32 x i32> %512, i32 %compare.local.base1410, i64 15, !dbg !16
  %514 = insertelement <32 x i32> %513, i32 %compare.local.base1414, i64 16, !dbg !16
  %515 = insertelement <32 x i32> %514, i32 %compare.local.base1418, i64 17, !dbg !16
  %516 = insertelement <32 x i32> %515, i32 %compare.local.base1422, i64 18, !dbg !16
  %517 = insertelement <32 x i32> %516, i32 %compare.local.base1426, i64 19, !dbg !16
  %518 = insertelement <32 x i32> %517, i32 %compare.local.base1430, i64 20, !dbg !16
  %519 = insertelement <32 x i32> %518, i32 %compare.local.base1434, i64 21, !dbg !16
  %520 = insertelement <32 x i32> %519, i32 %compare.local.base1438, i64 22, !dbg !16
  %521 = insertelement <32 x i32> %520, i32 %compare.local.base1442, i64 23, !dbg !16
  %522 = insertelement <32 x i32> %521, i32 %compare.local.base1446, i64 24, !dbg !16
  %523 = insertelement <32 x i32> %522, i32 %compare.local.base1450, i64 25, !dbg !16
  %524 = insertelement <32 x i32> %523, i32 %compare.local.base1454, i64 26, !dbg !16
  %525 = insertelement <32 x i32> %524, i32 %compare.local.base1458, i64 27, !dbg !16
  %526 = insertelement <32 x i32> %525, i32 %compare.local.base1462, i64 28, !dbg !16
  %527 = insertelement <32 x i32> %526, i32 %compare.local.base1466, i64 29, !dbg !16
  %528 = insertelement <32 x i32> %527, i32 %compare.local.base1470, i64 30, !dbg !16
  %529 = insertelement <32 x i32> %528, i32 %compare.local.base1474, i64 31, !dbg !16
  %530 = or <32 x i32> %529, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %531 = icmp ugt <32 x i32> %114, %530, !dbg !16
  %532 = insertelement <32 x i1> poison, i1 %compare.high.equal1348, i64 0, !dbg !16
  %533 = shufflevector <32 x i1> %532, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %534 = insertelement <32 x i1> poison, i1 %compare.high.order1349, i64 0, !dbg !16
  %535 = shufflevector <32 x i1> %534, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %536 = select <32 x i1> %533, <32 x i1> %531, <32 x i1> %535, !dbg !16
  %537 = select <32 x i1> %536, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %538 = insertvalue [2 x <32 x float>] poison, <32 x float> %498, 0, !dbg !12
  %539 = insertvalue [2 x <32 x float>] %538, <32 x float> %537, 1, !dbg !12
  br label %540, !dbg !12

540:                                              ; preds = %456, %117
  %541 = phi [2 x <32 x float>] [ %539, %456 ], [ zeroinitializer, %117 ], !dbg !12
  %542 = extractvalue [2 x <32 x float>] %541, 0, !dbg !19
  %543 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %544 = fadd <4 x float> %409, %543, !dbg !19
  %545 = extractelement <4 x float> %544, i64 0, !dbg !20
  %546 = tail call float @llvm.amdgcn.exp2.f32(float %545), !dbg !20
  %547 = insertelement <4 x float> poison, float %546, i64 0, !dbg !21
  %548 = extractelement <4 x float> %544, i64 1, !dbg !20
  %549 = tail call float @llvm.amdgcn.exp2.f32(float %548), !dbg !20
  %550 = insertelement <4 x float> %547, float %549, i64 1, !dbg !21
  %551 = fadd float %546, %549, !dbg !22
  %552 = extractelement <4 x float> %544, i64 2, !dbg !20
  %553 = tail call float @llvm.amdgcn.exp2.f32(float %552), !dbg !20
  %554 = insertelement <4 x float> %550, float %553, i64 2, !dbg !21
  %555 = fadd float %553, %551, !dbg !22
  %556 = extractelement <4 x float> %544, i64 3, !dbg !20
  %557 = tail call float @llvm.amdgcn.exp2.f32(float %556), !dbg !20
  %558 = insertelement <4 x float> %554, float %557, i64 3, !dbg !21
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %558) #8, !dbg !21, !frisk.register_lifetimes !5
  %559 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !21
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %559) #8, !dbg !22, !frisk.register_lifetimes !5
  %560 = fadd float %557, %555, !dbg !22
  %561 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %562 = fadd <4 x float> %413, %561, !dbg !19
  %563 = extractelement <4 x float> %562, i64 0, !dbg !20
  %564 = tail call float @llvm.amdgcn.exp2.f32(float %563), !dbg !20
  %565 = insertelement <4 x float> poison, float %564, i64 0, !dbg !21
  %566 = fadd float %564, %560, !dbg !22
  %567 = extractelement <4 x float> %562, i64 1, !dbg !20
  %568 = tail call float @llvm.amdgcn.exp2.f32(float %567), !dbg !20
  %569 = insertelement <4 x float> %565, float %568, i64 1, !dbg !21
  %570 = fadd float %568, %566, !dbg !22
  %571 = extractelement <4 x float> %562, i64 2, !dbg !20
  %572 = tail call float @llvm.amdgcn.exp2.f32(float %571), !dbg !20
  %573 = insertelement <4 x float> %569, float %572, i64 2, !dbg !21
  %574 = fadd float %572, %570, !dbg !22
  %575 = extractelement <4 x float> %562, i64 3, !dbg !20
  %576 = tail call float @llvm.amdgcn.exp2.f32(float %575), !dbg !20
  %577 = insertelement <4 x float> %573, float %576, i64 3, !dbg !21
  %math.result309 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %577) #8, !dbg !21, !frisk.register_lifetimes !5
  %578 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !21
  %math.packed310 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %578) #8, !dbg !22, !frisk.register_lifetimes !5
  %579 = fadd float %576, %574, !dbg !22
  %580 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %581 = fadd <4 x float> %417, %580, !dbg !19
  %582 = extractelement <4 x float> %581, i64 0, !dbg !20
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %582), !dbg !20
  %584 = insertelement <4 x float> poison, float %583, i64 0, !dbg !21
  %585 = fadd float %583, %579, !dbg !22
  %586 = extractelement <4 x float> %581, i64 1, !dbg !20
  %587 = tail call float @llvm.amdgcn.exp2.f32(float %586), !dbg !20
  %588 = insertelement <4 x float> %584, float %587, i64 1, !dbg !21
  %589 = fadd float %587, %585, !dbg !22
  %590 = extractelement <4 x float> %581, i64 2, !dbg !20
  %591 = tail call float @llvm.amdgcn.exp2.f32(float %590), !dbg !20
  %592 = insertelement <4 x float> %588, float %591, i64 2, !dbg !21
  %593 = fadd float %591, %589, !dbg !22
  %594 = extractelement <4 x float> %581, i64 3, !dbg !20
  %595 = tail call float @llvm.amdgcn.exp2.f32(float %594), !dbg !20
  %596 = insertelement <4 x float> %592, float %595, i64 3, !dbg !21
  %math.result311 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %596) #8, !dbg !21, !frisk.register_lifetimes !5
  %597 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !21
  %math.packed312 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %597) #8, !dbg !22, !frisk.register_lifetimes !5
  %598 = fadd float %595, %593, !dbg !22
  %599 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %600 = fadd <4 x float> %421, %599, !dbg !19
  %601 = extractelement <4 x float> %600, i64 0, !dbg !20
  %602 = tail call float @llvm.amdgcn.exp2.f32(float %601), !dbg !20
  %603 = insertelement <4 x float> poison, float %602, i64 0, !dbg !21
  %604 = fadd float %602, %598, !dbg !22
  %605 = extractelement <4 x float> %600, i64 1, !dbg !20
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %605), !dbg !20
  %607 = insertelement <4 x float> %603, float %606, i64 1, !dbg !21
  %608 = fadd float %606, %604, !dbg !22
  %609 = extractelement <4 x float> %600, i64 2, !dbg !20
  %610 = tail call float @llvm.amdgcn.exp2.f32(float %609), !dbg !20
  %611 = insertelement <4 x float> %607, float %610, i64 2, !dbg !21
  %612 = fadd float %610, %608, !dbg !22
  %613 = extractelement <4 x float> %600, i64 3, !dbg !20
  %614 = tail call float @llvm.amdgcn.exp2.f32(float %613), !dbg !20
  %615 = insertelement <4 x float> %611, float %614, i64 3, !dbg !21
  %math.result313 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %615) #8, !dbg !21, !frisk.register_lifetimes !5
  %616 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !21
  %math.packed314 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %616) #8, !dbg !22, !frisk.register_lifetimes !5
  %617 = fadd float %614, %612, !dbg !22
  %618 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %619 = fadd <4 x float> %425, %618, !dbg !19
  %620 = extractelement <4 x float> %619, i64 0, !dbg !20
  %621 = tail call float @llvm.amdgcn.exp2.f32(float %620), !dbg !20
  %622 = insertelement <4 x float> poison, float %621, i64 0, !dbg !21
  %623 = fadd float %621, %617, !dbg !22
  %624 = extractelement <4 x float> %619, i64 1, !dbg !20
  %625 = tail call float @llvm.amdgcn.exp2.f32(float %624), !dbg !20
  %626 = insertelement <4 x float> %622, float %625, i64 1, !dbg !21
  %627 = fadd float %625, %623, !dbg !22
  %628 = extractelement <4 x float> %619, i64 2, !dbg !20
  %629 = tail call float @llvm.amdgcn.exp2.f32(float %628), !dbg !20
  %630 = insertelement <4 x float> %626, float %629, i64 2, !dbg !21
  %631 = fadd float %629, %627, !dbg !22
  %632 = extractelement <4 x float> %619, i64 3, !dbg !20
  %633 = tail call float @llvm.amdgcn.exp2.f32(float %632), !dbg !20
  %634 = insertelement <4 x float> %630, float %633, i64 3, !dbg !21
  %math.result315 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %634) #8, !dbg !21, !frisk.register_lifetimes !5
  %635 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !21
  %math.packed316 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %635) #8, !dbg !22, !frisk.register_lifetimes !5
  %636 = fadd float %633, %631, !dbg !22
  %637 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %638 = fadd <4 x float> %429, %637, !dbg !19
  %639 = extractelement <4 x float> %638, i64 0, !dbg !20
  %640 = tail call float @llvm.amdgcn.exp2.f32(float %639), !dbg !20
  %641 = insertelement <4 x float> poison, float %640, i64 0, !dbg !21
  %642 = fadd float %640, %636, !dbg !22
  %643 = extractelement <4 x float> %638, i64 1, !dbg !20
  %644 = tail call float @llvm.amdgcn.exp2.f32(float %643), !dbg !20
  %645 = insertelement <4 x float> %641, float %644, i64 1, !dbg !21
  %646 = fadd float %644, %642, !dbg !22
  %647 = extractelement <4 x float> %638, i64 2, !dbg !20
  %648 = tail call float @llvm.amdgcn.exp2.f32(float %647), !dbg !20
  %649 = insertelement <4 x float> %645, float %648, i64 2, !dbg !21
  %650 = fadd float %648, %646, !dbg !22
  %651 = extractelement <4 x float> %638, i64 3, !dbg !20
  %652 = tail call float @llvm.amdgcn.exp2.f32(float %651), !dbg !20
  %653 = insertelement <4 x float> %649, float %652, i64 3, !dbg !21
  %math.result317 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %653) #8, !dbg !21, !frisk.register_lifetimes !5
  %654 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !21
  %math.packed318 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %654) #8, !dbg !22, !frisk.register_lifetimes !5
  %655 = fadd float %652, %650, !dbg !22
  %656 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %657 = fadd <4 x float> %433, %656, !dbg !19
  %658 = extractelement <4 x float> %657, i64 0, !dbg !20
  %659 = tail call float @llvm.amdgcn.exp2.f32(float %658), !dbg !20
  %660 = insertelement <4 x float> poison, float %659, i64 0, !dbg !21
  %661 = fadd float %659, %655, !dbg !22
  %662 = extractelement <4 x float> %657, i64 1, !dbg !20
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %662), !dbg !20
  %664 = insertelement <4 x float> %660, float %663, i64 1, !dbg !21
  %665 = fadd float %663, %661, !dbg !22
  %666 = extractelement <4 x float> %657, i64 2, !dbg !20
  %667 = tail call float @llvm.amdgcn.exp2.f32(float %666), !dbg !20
  %668 = insertelement <4 x float> %664, float %667, i64 2, !dbg !21
  %669 = fadd float %667, %665, !dbg !22
  %670 = extractelement <4 x float> %657, i64 3, !dbg !20
  %671 = tail call float @llvm.amdgcn.exp2.f32(float %670), !dbg !20
  %672 = insertelement <4 x float> %668, float %671, i64 3, !dbg !21
  %math.result319 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %672) #8, !dbg !21, !frisk.register_lifetimes !5
  %673 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !21
  %math.packed320 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %673) #8, !dbg !22, !frisk.register_lifetimes !5
  %674 = fadd float %671, %669, !dbg !22
  %675 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %676 = fadd <4 x float> %436, %675, !dbg !19
  %677 = extractelement <4 x float> %676, i64 0, !dbg !20
  %678 = tail call float @llvm.amdgcn.exp2.f32(float %677), !dbg !20
  %679 = insertelement <4 x float> poison, float %678, i64 0, !dbg !21
  %680 = fadd float %678, %674, !dbg !22
  %681 = extractelement <4 x float> %676, i64 1, !dbg !20
  %682 = tail call float @llvm.amdgcn.exp2.f32(float %681), !dbg !20
  %683 = insertelement <4 x float> %679, float %682, i64 1, !dbg !21
  %684 = fadd float %682, %680, !dbg !22
  %685 = extractelement <4 x float> %676, i64 2, !dbg !20
  %686 = tail call float @llvm.amdgcn.exp2.f32(float %685), !dbg !20
  %687 = insertelement <4 x float> %683, float %686, i64 2, !dbg !21
  %688 = fadd float %686, %684, !dbg !22
  %689 = extractelement <4 x float> %676, i64 3, !dbg !20
  %690 = tail call float @llvm.amdgcn.exp2.f32(float %689), !dbg !20
  %691 = insertelement <4 x float> %687, float %690, i64 3, !dbg !21
  %math.result321 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %691) #8, !dbg !21, !frisk.register_lifetimes !5
  %692 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !21
  %math.packed322 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %692) #8, !dbg !22, !frisk.register_lifetimes !5
  %693 = fadd float %690, %688, !dbg !22
  %694 = extractvalue [2 x <32 x float>] %541, 1, !dbg !19
  %695 = shufflevector <32 x float> %694, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %696 = fadd <4 x float> %438, %695, !dbg !19
  %697 = extractelement <4 x float> %696, i64 0, !dbg !20
  %698 = tail call float @llvm.amdgcn.exp2.f32(float %697), !dbg !20
  %699 = insertelement <4 x float> poison, float %698, i64 0, !dbg !21
  %700 = extractelement <4 x float> %696, i64 1, !dbg !20
  %701 = tail call float @llvm.amdgcn.exp2.f32(float %700), !dbg !20
  %702 = insertelement <4 x float> %699, float %701, i64 1, !dbg !21
  %703 = fadd float %698, %701, !dbg !22
  %704 = extractelement <4 x float> %696, i64 2, !dbg !20
  %705 = tail call float @llvm.amdgcn.exp2.f32(float %704), !dbg !20
  %706 = insertelement <4 x float> %702, float %705, i64 2, !dbg !21
  %707 = fadd float %705, %703, !dbg !22
  %708 = extractelement <4 x float> %696, i64 3, !dbg !20
  %709 = tail call float @llvm.amdgcn.exp2.f32(float %708), !dbg !20
  %710 = insertelement <4 x float> %706, float %709, i64 3, !dbg !21
  %math.result323 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %710) #8, !dbg !21, !frisk.register_lifetimes !5
  %711 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !21
  %math.packed324 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %711) #8, !dbg !22, !frisk.register_lifetimes !5
  %712 = fadd float %709, %707, !dbg !22
  %713 = shufflevector <32 x float> %694, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %714 = fadd <4 x float> %440, %713, !dbg !19
  %715 = extractelement <4 x float> %714, i64 0, !dbg !20
  %716 = tail call float @llvm.amdgcn.exp2.f32(float %715), !dbg !20
  %717 = insertelement <4 x float> poison, float %716, i64 0, !dbg !21
  %718 = fadd float %716, %712, !dbg !22
  %719 = extractelement <4 x float> %714, i64 1, !dbg !20
  %720 = tail call float @llvm.amdgcn.exp2.f32(float %719), !dbg !20
  %721 = insertelement <4 x float> %717, float %720, i64 1, !dbg !21
  %722 = fadd float %720, %718, !dbg !22
  %723 = extractelement <4 x float> %714, i64 2, !dbg !20
  %724 = tail call float @llvm.amdgcn.exp2.f32(float %723), !dbg !20
  %725 = insertelement <4 x float> %721, float %724, i64 2, !dbg !21
  %726 = fadd float %724, %722, !dbg !22
  %727 = extractelement <4 x float> %714, i64 3, !dbg !20
  %728 = tail call float @llvm.amdgcn.exp2.f32(float %727), !dbg !20
  %729 = insertelement <4 x float> %725, float %728, i64 3, !dbg !21
  %math.result325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %729) #8, !dbg !21, !frisk.register_lifetimes !5
  %730 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !21
  %math.packed326 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %730) #8, !dbg !22, !frisk.register_lifetimes !5
  %731 = fadd float %728, %726, !dbg !22
  %732 = shufflevector <32 x float> %694, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %733 = fadd <4 x float> %442, %732, !dbg !19
  %734 = extractelement <4 x float> %733, i64 0, !dbg !20
  %735 = tail call float @llvm.amdgcn.exp2.f32(float %734), !dbg !20
  %736 = insertelement <4 x float> poison, float %735, i64 0, !dbg !21
  %737 = fadd float %735, %731, !dbg !22
  %738 = extractelement <4 x float> %733, i64 1, !dbg !20
  %739 = tail call float @llvm.amdgcn.exp2.f32(float %738), !dbg !20
  %740 = insertelement <4 x float> %736, float %739, i64 1, !dbg !21
  %741 = fadd float %739, %737, !dbg !22
  %742 = extractelement <4 x float> %733, i64 2, !dbg !20
  %743 = tail call float @llvm.amdgcn.exp2.f32(float %742), !dbg !20
  %744 = insertelement <4 x float> %740, float %743, i64 2, !dbg !21
  %745 = fadd float %743, %741, !dbg !22
  %746 = extractelement <4 x float> %733, i64 3, !dbg !20
  %747 = tail call float @llvm.amdgcn.exp2.f32(float %746), !dbg !20
  %748 = insertelement <4 x float> %744, float %747, i64 3, !dbg !21
  %math.result327 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %748) #8, !dbg !21, !frisk.register_lifetimes !5
  %749 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !21
  %math.packed328 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %749) #8, !dbg !22, !frisk.register_lifetimes !5
  %750 = fadd float %747, %745, !dbg !22
  %751 = shufflevector <32 x float> %694, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %752 = fadd <4 x float> %444, %751, !dbg !19
  %753 = extractelement <4 x float> %752, i64 0, !dbg !20
  %754 = tail call float @llvm.amdgcn.exp2.f32(float %753), !dbg !20
  %755 = insertelement <4 x float> poison, float %754, i64 0, !dbg !21
  %756 = fadd float %754, %750, !dbg !22
  %757 = extractelement <4 x float> %752, i64 1, !dbg !20
  %758 = tail call float @llvm.amdgcn.exp2.f32(float %757), !dbg !20
  %759 = insertelement <4 x float> %755, float %758, i64 1, !dbg !21
  %760 = fadd float %758, %756, !dbg !22
  %761 = extractelement <4 x float> %752, i64 2, !dbg !20
  %762 = tail call float @llvm.amdgcn.exp2.f32(float %761), !dbg !20
  %763 = insertelement <4 x float> %759, float %762, i64 2, !dbg !21
  %764 = fadd float %762, %760, !dbg !22
  %765 = extractelement <4 x float> %752, i64 3, !dbg !20
  %766 = tail call float @llvm.amdgcn.exp2.f32(float %765), !dbg !20
  %767 = insertelement <4 x float> %763, float %766, i64 3, !dbg !21
  %math.result329 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %767) #8, !dbg !21, !frisk.register_lifetimes !5
  %768 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !21
  %math.packed330 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %768) #8, !dbg !22, !frisk.register_lifetimes !5
  %769 = fadd float %766, %764, !dbg !22
  %770 = shufflevector <32 x float> %694, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %771 = fadd <4 x float> %446, %770, !dbg !19
  %772 = extractelement <4 x float> %771, i64 0, !dbg !20
  %773 = tail call float @llvm.amdgcn.exp2.f32(float %772), !dbg !20
  %774 = insertelement <4 x float> poison, float %773, i64 0, !dbg !21
  %775 = fadd float %773, %769, !dbg !22
  %776 = extractelement <4 x float> %771, i64 1, !dbg !20
  %777 = tail call float @llvm.amdgcn.exp2.f32(float %776), !dbg !20
  %778 = insertelement <4 x float> %774, float %777, i64 1, !dbg !21
  %779 = fadd float %777, %775, !dbg !22
  %780 = extractelement <4 x float> %771, i64 2, !dbg !20
  %781 = tail call float @llvm.amdgcn.exp2.f32(float %780), !dbg !20
  %782 = insertelement <4 x float> %778, float %781, i64 2, !dbg !21
  %783 = fadd float %781, %779, !dbg !22
  %784 = extractelement <4 x float> %771, i64 3, !dbg !20
  %785 = tail call float @llvm.amdgcn.exp2.f32(float %784), !dbg !20
  %786 = insertelement <4 x float> %782, float %785, i64 3, !dbg !21
  %math.result331 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %786) #8, !dbg !21, !frisk.register_lifetimes !5
  %787 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !21
  %math.packed332 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %787) #8, !dbg !22, !frisk.register_lifetimes !5
  %788 = fadd float %785, %783, !dbg !22
  %789 = shufflevector <32 x float> %694, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %790 = fadd <4 x float> %448, %789, !dbg !19
  %791 = extractelement <4 x float> %790, i64 0, !dbg !20
  %792 = tail call float @llvm.amdgcn.exp2.f32(float %791), !dbg !20
  %793 = insertelement <4 x float> poison, float %792, i64 0, !dbg !21
  %794 = fadd float %792, %788, !dbg !22
  %795 = extractelement <4 x float> %790, i64 1, !dbg !20
  %796 = tail call float @llvm.amdgcn.exp2.f32(float %795), !dbg !20
  %797 = insertelement <4 x float> %793, float %796, i64 1, !dbg !21
  %798 = fadd float %796, %794, !dbg !22
  %799 = extractelement <4 x float> %790, i64 2, !dbg !20
  %800 = tail call float @llvm.amdgcn.exp2.f32(float %799), !dbg !20
  %801 = insertelement <4 x float> %797, float %800, i64 2, !dbg !21
  %802 = fadd float %800, %798, !dbg !22
  %803 = extractelement <4 x float> %790, i64 3, !dbg !20
  %804 = tail call float @llvm.amdgcn.exp2.f32(float %803), !dbg !20
  %805 = insertelement <4 x float> %801, float %804, i64 3, !dbg !21
  %math.result333 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %805) #8, !dbg !21, !frisk.register_lifetimes !5
  %806 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !21
  %math.packed334 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %806) #8, !dbg !22, !frisk.register_lifetimes !5
  %807 = fadd float %804, %802, !dbg !22
  %808 = shufflevector <32 x float> %694, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %809 = fadd <4 x float> %450, %808, !dbg !19
  %810 = extractelement <4 x float> %809, i64 0, !dbg !20
  %811 = tail call float @llvm.amdgcn.exp2.f32(float %810), !dbg !20
  %812 = insertelement <4 x float> poison, float %811, i64 0, !dbg !21
  %813 = fadd float %811, %807, !dbg !22
  %814 = extractelement <4 x float> %809, i64 1, !dbg !20
  %815 = tail call float @llvm.amdgcn.exp2.f32(float %814), !dbg !20
  %816 = insertelement <4 x float> %812, float %815, i64 1, !dbg !21
  %817 = fadd float %815, %813, !dbg !22
  %818 = extractelement <4 x float> %809, i64 2, !dbg !20
  %819 = tail call float @llvm.amdgcn.exp2.f32(float %818), !dbg !20
  %820 = insertelement <4 x float> %816, float %819, i64 2, !dbg !21
  %821 = fadd float %819, %817, !dbg !22
  %822 = extractelement <4 x float> %809, i64 3, !dbg !20
  %823 = tail call float @llvm.amdgcn.exp2.f32(float %822), !dbg !20
  %824 = insertelement <4 x float> %820, float %823, i64 3, !dbg !21
  %math.result335 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %824) #8, !dbg !21, !frisk.register_lifetimes !5
  %825 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !21
  %math.packed336 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %825) #8, !dbg !22, !frisk.register_lifetimes !5
  %826 = fadd float %823, %821, !dbg !22
  %827 = shufflevector <32 x float> %694, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %828 = fadd <4 x float> %454, %827, !dbg !19
  %829 = extractelement <4 x float> %828, i64 0, !dbg !20
  %830 = tail call float @llvm.amdgcn.exp2.f32(float %829), !dbg !20
  %831 = insertelement <4 x float> poison, float %830, i64 0, !dbg !21
  %832 = fadd float %830, %826, !dbg !22
  %833 = extractelement <4 x float> %828, i64 1, !dbg !20
  %834 = tail call float @llvm.amdgcn.exp2.f32(float %833), !dbg !20
  %835 = insertelement <4 x float> %831, float %834, i64 1, !dbg !21
  %836 = fadd float %834, %832, !dbg !22
  %837 = extractelement <4 x float> %828, i64 2, !dbg !20
  %838 = tail call float @llvm.amdgcn.exp2.f32(float %837), !dbg !20
  %839 = insertelement <4 x float> %835, float %838, i64 2, !dbg !21
  %840 = fadd float %838, %836, !dbg !22
  %841 = extractelement <4 x float> %828, i64 3, !dbg !20
  %842 = tail call float @llvm.amdgcn.exp2.f32(float %841), !dbg !20
  %843 = insertelement <4 x float> %839, float %842, i64 3, !dbg !21
  %math.result337 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %843) #8, !dbg !21, !frisk.register_lifetimes !5
  %844 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !21
  %math.packed338 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %844) #8, !dbg !22, !frisk.register_lifetimes !5
  %845 = fadd float %842, %840, !dbg !22
  %846 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !22
  %847 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %846), !dbg !22
  %848 = add nuw nsw i32 %847, 64, !dbg !22
  %849 = xor i32 %847, 16, !dbg !22
  %850 = and i32 %848, 192, !dbg !22
  %851 = icmp samesign ult i32 %849, %850, !dbg !22
  %852 = select i1 %851, i32 %849, i32 %847, !dbg !22
  %853 = shl nuw nsw i32 %852, 2, !dbg !22
  %854 = bitcast float %693 to i32, !dbg !22
  %855 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %853, i32 %854), !dbg !22
  %856 = bitcast i32 %855 to float, !dbg !22
  %857 = fadd float %693, %856, !dbg !22
  %858 = xor i32 %847, 32, !dbg !22
  %859 = icmp samesign ult i32 %858, %850, !dbg !22
  %860 = select i1 %859, i32 %858, i32 %847, !dbg !22
  %861 = shl nuw nsw i32 %860, 2, !dbg !22
  %862 = bitcast float %857 to i32, !dbg !22
  %863 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %861, i32 %862), !dbg !22
  %864 = bitcast i32 %863 to float, !dbg !22
  %865 = fadd float %857, %864, !dbg !22
  %866 = extractvalue [2 x <1 x float>] %121, 0, !dbg !22
  %867 = extractelement <1 x float> %866, i64 0, !dbg !22
  %868 = fadd float %867, %865, !dbg !22
  %869 = insertelement <1 x float> poison, float %868, i64 0, !dbg !11
  %870 = insertvalue [2 x <1 x float>] poison, <1 x float> %869, 0, !dbg !11
  %871 = bitcast float %845 to i32, !dbg !22
  %872 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %853, i32 %871), !dbg !22
  %873 = bitcast i32 %872 to float, !dbg !22
  %874 = fadd float %845, %873, !dbg !22
  %875 = bitcast float %874 to i32, !dbg !22
  %876 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %861, i32 %875), !dbg !22
  %877 = bitcast i32 %876 to float, !dbg !22
  %878 = fadd float %874, %877, !dbg !22
  %879 = extractvalue [2 x <1 x float>] %121, 1, !dbg !22
  %880 = extractelement <1 x float> %879, i64 0, !dbg !22
  %881 = fadd float %880, %878, !dbg !22
  %882 = insertelement <1 x float> poison, float %881, i64 0, !dbg !11
  %883 = insertvalue [2 x <1 x float>] %870, <1 x float> %882, 1, !dbg !11
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
  %884 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local936, i32 0) #11, !dbg !23
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !23
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
  %886 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !23
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !23
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local948, i64 512, !dbg !23
  %888 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !23
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !23
  %address.local957 = getelementptr i8, ptr addrspace(3) %address.local948, i64 768, !dbg !23
  %890 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local957, i32 0) #11, !dbg !23
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !23
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local936, i64 4096, !dbg !23
  %892 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !23
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !23
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local948, i64 4352, !dbg !23
  %894 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !23
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !23
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local948, i64 4608, !dbg !23
  %896 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !23
  %897 = bitcast <4 x i32> %896 to <8 x half>, !dbg !23
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local948, i64 4864, !dbg !23
  %898 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #11, !dbg !23
  %899 = bitcast <4 x i32> %898 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local972 = shl nuw nsw i64 %address.local937, 3, !dbg !8
  %address.local973 = and i64 %address.local972, 1536, !dbg !8
  %address.local974 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local973, !dbg !8
  %address.local975 = ptrtoint ptr addrspace(3) %address.local974 to i64, !dbg !8
  %address.local976 = trunc i64 %address.local975 to i32, !dbg !8
  %900 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local976), !dbg !17
  %address.local977 = shl nuw nsw i64 %address.local937, 5, !dbg !17
  %address.local978 = and i64 %address.local977, 8064, !dbg !17
  %address.local980 = and i64 %address.local972, 24, !dbg !17
  %address.local981 = or disjoint i64 %address.local978, %address.local980, !dbg !17
  %901 = or disjoint i64 %.idx5761, %address.local981, !dbg !17
  %902 = trunc i64 %901 to i32, !dbg !17
  %.tr69 = shl i32 %902, 1, !dbg !17
  %dma.buffer.offset1628 = or i32 %.tr69, 64, !dbg !17
  %dma.buffer.lds1630 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %900), !dbg !17
  %903 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1630, i32 %dma.buffer.offset1628, <4 x i32> %dma.buffer.rsrc1635) #9, !dbg !17
  %address.local983 = or disjoint i64 %address.local937, 256, !dbg !8
  %address.local984 = shl nuw nsw i64 %address.local983, 3, !dbg !8
  %address.local985 = and i64 %address.local984, 3584, !dbg !8
  %address.local986 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local985, !dbg !8
  %address.local987 = ptrtoint ptr addrspace(3) %address.local986 to i64, !dbg !8
  %address.local988 = trunc i64 %address.local987 to i32, !dbg !8
  %904 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local988), !dbg !17
  %address.local989 = shl nuw nsw i64 %address.local983, 5, !dbg !17
  %address.local990 = and i64 %address.local989, 16256, !dbg !17
  %address.local993 = or disjoint i64 %address.local990, %address.local980, !dbg !17
  %905 = or disjoint i64 %.idx5761, %address.local993, !dbg !17
  %906 = trunc i64 %905 to i32, !dbg !17
  %.tr73 = shl i32 %906, 1, !dbg !17
  %dma.buffer.offset1625 = or i32 %.tr73, 64, !dbg !17
  %dma.buffer.lds1627 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %904), !dbg !17
  %907 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1627, i32 %dma.buffer.offset1625, <4 x i32> %dma.buffer.rsrc1635) #9, !dbg !17
  %908 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %909 = extractvalue [2 x <8 x float>] %122, 0, !dbg !24
  %910 = shufflevector <8 x float> %909, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %908, <4 x float> %910) #8, !dbg !23
  %912 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %912, <4 x float> %911) #8, !dbg !23
  %914 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %915 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %914, <4 x float> %913) #8, !dbg !23
  %916 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %916, <4 x float> %915) #8, !dbg !23
  %918 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %919 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %918, <4 x float> %917) #8, !dbg !23
  %920 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %921 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %920, <4 x float> %919) #8, !dbg !23
  %922 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %922, <4 x float> %921) #8, !dbg !23
  %924 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %925 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %924, <4 x float> %923) #8, !dbg !23
  %926 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %927 = shufflevector <8 x float> %909, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %926, <4 x float> %927) #8, !dbg !23
  %929 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %929, <4 x float> %928) #8, !dbg !23
  %931 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %932 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %931, <4 x float> %930) #8, !dbg !23
  %933 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %934 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %933, <4 x float> %932) #8, !dbg !23
  %935 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %935, <4 x float> %934) #8, !dbg !23
  %937 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %937, <4 x float> %936) #8, !dbg !23
  %939 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %940 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %939, <4 x float> %938) #8, !dbg !23
  %941 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %942 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %941, <4 x float> %940) #8, !dbg !23
  %943 = extractvalue [2 x <8 x float>] %122, 1, !dbg !24
  %944 = shufflevector <8 x float> %943, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %945 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %908, <4 x float> %944) #8, !dbg !23
  %946 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %912, <4 x float> %945) #8, !dbg !23
  %947 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %914, <4 x float> %946) #8, !dbg !23
  %948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %916, <4 x float> %947) #8, !dbg !23
  %949 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %918, <4 x float> %948) #8, !dbg !23
  %950 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %920, <4 x float> %949) #8, !dbg !23
  %951 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %922, <4 x float> %950) #8, !dbg !23
  %952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %924, <4 x float> %951) #8, !dbg !23
  %953 = shufflevector <8 x float> %943, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %926, <4 x float> %953) #8, !dbg !23
  %955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %929, <4 x float> %954) #8, !dbg !23
  %956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %931, <4 x float> %955) #8, !dbg !23
  %957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %933, <4 x float> %956) #8, !dbg !23
  %958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %935, <4 x float> %957) #8, !dbg !23
  %959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %937, <4 x float> %958) #8, !dbg !23
  %960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %939, <4 x float> %959) #8, !dbg !23
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %941, <4 x float> %960) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %962 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local936, i32 0) #11, !dbg !23
  %963 = bitcast <4 x i32> %962 to <8 x half>, !dbg !23
  %964 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !23
  %965 = bitcast <4 x i32> %964 to <8 x half>, !dbg !23
  %966 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !23
  %967 = bitcast <4 x i32> %966 to <8 x half>, !dbg !23
  %968 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local957, i32 0) #11, !dbg !23
  %969 = bitcast <4 x i32> %968 to <8 x half>, !dbg !23
  %970 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !23
  %971 = bitcast <4 x i32> %970 to <8 x half>, !dbg !23
  %972 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !23
  %973 = bitcast <4 x i32> %972 to <8 x half>, !dbg !23
  %974 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !23
  %975 = bitcast <4 x i32> %974 to <8 x half>, !dbg !23
  %976 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #11, !dbg !23
  %977 = bitcast <4 x i32> %976 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %dma.buffer.offset1622 = or i32 %.tr69, 128, !dbg !17
  %978 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1630, i32 %dma.buffer.offset1622, <4 x i32> %dma.buffer.rsrc1635) #9, !dbg !17
  %dma.buffer.offset1619 = or i32 %.tr73, 128, !dbg !17
  %979 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1627, i32 %dma.buffer.offset1619, <4 x i32> %dma.buffer.rsrc1635) #9, !dbg !17
  %980 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %981 = extractvalue [2 x <8 x float>] %120, 0, !dbg !24
  %982 = shufflevector <8 x float> %981, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %983 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %980, <4 x float> %982) #8, !dbg !23
  %984 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %985 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %984, <4 x float> %983) #8, !dbg !23
  %986 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %986, <4 x float> %985) #8, !dbg !23
  %988 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %989 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %988, <4 x float> %987) #8, !dbg !23
  %990 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %990, <4 x float> %989) #8, !dbg !23
  %992 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %992, <4 x float> %991) #8, !dbg !23
  %994 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %995 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %994, <4 x float> %993) #8, !dbg !23
  %996 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %997 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %996, <4 x float> %995) #8, !dbg !23
  %998 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %999 = shufflevector <8 x float> %981, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %998, <4 x float> %999) #8, !dbg !23
  %1001 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1001, <4 x float> %1000) #8, !dbg !23
  %1003 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1003, <4 x float> %1002) #8, !dbg !23
  %1005 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1005, <4 x float> %1004) #8, !dbg !23
  %1007 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1007, <4 x float> %1006) #8, !dbg !23
  %1009 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1009, <4 x float> %1008) #8, !dbg !23
  %1011 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1011, <4 x float> %1010) #8, !dbg !23
  %1013 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1013, <4 x float> %1012) #8, !dbg !23
  %1015 = extractvalue [2 x <8 x float>] %120, 1, !dbg !24
  %1016 = shufflevector <8 x float> %1015, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1017 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %980, <4 x float> %1016) #8, !dbg !23
  %1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %984, <4 x float> %1017) #8, !dbg !23
  %1019 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %986, <4 x float> %1018) #8, !dbg !23
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %988, <4 x float> %1019) #8, !dbg !23
  %1021 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %990, <4 x float> %1020) #8, !dbg !23
  %1022 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %992, <4 x float> %1021) #8, !dbg !23
  %1023 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %994, <4 x float> %1022) #8, !dbg !23
  %1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %996, <4 x float> %1023) #8, !dbg !23
  %1025 = shufflevector <8 x float> %1015, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %998, <4 x float> %1025) #8, !dbg !23
  %1027 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1001, <4 x float> %1026) #8, !dbg !23
  %1028 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1003, <4 x float> %1027) #8, !dbg !23
  %1029 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1005, <4 x float> %1028) #8, !dbg !23
  %1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1007, <4 x float> %1029) #8, !dbg !23
  %1031 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1009, <4 x float> %1030) #8, !dbg !23
  %1032 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1011, <4 x float> %1031) #8, !dbg !23
  %1033 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1013, <4 x float> %1032) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1034 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local936, i32 0) #11, !dbg !23
  %1035 = bitcast <4 x i32> %1034 to <8 x half>, !dbg !23
  %1036 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !23
  %1037 = bitcast <4 x i32> %1036 to <8 x half>, !dbg !23
  %1038 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !23
  %1039 = bitcast <4 x i32> %1038 to <8 x half>, !dbg !23
  %1040 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local957, i32 0) #11, !dbg !23
  %1041 = bitcast <4 x i32> %1040 to <8 x half>, !dbg !23
  %1042 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !23
  %1043 = bitcast <4 x i32> %1042 to <8 x half>, !dbg !23
  %1044 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !23
  %1045 = bitcast <4 x i32> %1044 to <8 x half>, !dbg !23
  %1046 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !23
  %1047 = bitcast <4 x i32> %1046 to <8 x half>, !dbg !23
  %1048 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #11, !dbg !23
  %1049 = bitcast <4 x i32> %1048 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1050 = or disjoint i64 %.idx5761, 96, !dbg !17
  %1051 = or disjoint i64 %address.local981, %1050, !dbg !17
  %.tr85 = trunc i64 %1051 to i32, !dbg !17
  %dma.buffer.offset1616 = shl i32 %.tr85, 1, !dbg !17
  %1052 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1630, i32 %dma.buffer.offset1616, <4 x i32> %dma.buffer.rsrc1635) #9, !dbg !17
  %1053 = or disjoint i64 %address.local993, %1050, !dbg !17
  %.tr89 = trunc i64 %1053 to i32, !dbg !17
  %dma.buffer.offset = shl i32 %.tr89, 1, !dbg !17
  %1054 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1627, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1635) #9, !dbg !17
  %1055 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1056 = extractvalue [2 x <8 x float>] %119, 0, !dbg !24
  %1057 = shufflevector <8 x float> %1056, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1058 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1055, <4 x float> %1057) #8, !dbg !23
  %1059 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1060 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1059, <4 x float> %1058) #8, !dbg !23
  %1061 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1062 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1061, <4 x float> %1060) #8, !dbg !23
  %1063 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1064 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1063, <4 x float> %1062) #8, !dbg !23
  %1065 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1066 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1065, <4 x float> %1064) #8, !dbg !23
  %1067 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1068 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1067, <4 x float> %1066) #8, !dbg !23
  %1069 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1070 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1069, <4 x float> %1068) #8, !dbg !23
  %1071 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1072 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1071, <4 x float> %1070) #8, !dbg !23
  %1073 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1074 = shufflevector <8 x float> %1056, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1075 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1073, <4 x float> %1074) #8, !dbg !23
  %1076 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1077 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1076, <4 x float> %1075) #8, !dbg !23
  %1078 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1079 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1078, <4 x float> %1077) #8, !dbg !23
  %1080 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1081 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1080, <4 x float> %1079) #8, !dbg !23
  %1082 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1083 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1082, <4 x float> %1081) #8, !dbg !23
  %1084 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1085 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1084, <4 x float> %1083) #8, !dbg !23
  %1086 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1087 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1086, <4 x float> %1085) #8, !dbg !23
  %1088 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1089 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1088, <4 x float> %1087) #8, !dbg !23
  %1090 = extractvalue [2 x <8 x float>] %119, 1, !dbg !24
  %1091 = shufflevector <8 x float> %1090, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1055, <4 x float> %1091) #8, !dbg !23
  %1093 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1059, <4 x float> %1092) #8, !dbg !23
  %1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1061, <4 x float> %1093) #8, !dbg !23
  %1095 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1063, <4 x float> %1094) #8, !dbg !23
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1065, <4 x float> %1095) #8, !dbg !23
  %1097 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1067, <4 x float> %1096) #8, !dbg !23
  %1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1069, <4 x float> %1097) #8, !dbg !23
  %1099 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1071, <4 x float> %1098) #8, !dbg !23
  %1100 = shufflevector <8 x float> %1090, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1101 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1073, <4 x float> %1100) #8, !dbg !23
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1076, <4 x float> %1101) #8, !dbg !23
  %1103 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1078, <4 x float> %1102) #8, !dbg !23
  %1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1080, <4 x float> %1103) #8, !dbg !23
  %1105 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1082, <4 x float> %1104) #8, !dbg !23
  %1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1084, <4 x float> %1105) #8, !dbg !23
  %1107 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1086, <4 x float> %1106) #8, !dbg !23
  %1108 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1088, <4 x float> %1107) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1109 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local936, i32 0) #11, !dbg !23
  %1110 = bitcast <4 x i32> %1109 to <8 x half>, !dbg !23
  %1111 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !23
  %1112 = bitcast <4 x i32> %1111 to <8 x half>, !dbg !23
  %1113 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !23
  %1114 = bitcast <4 x i32> %1113 to <8 x half>, !dbg !23
  %1115 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local957, i32 0) #11, !dbg !23
  %1116 = bitcast <4 x i32> %1115 to <8 x half>, !dbg !23
  %1117 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !23
  %1118 = bitcast <4 x i32> %1117 to <8 x half>, !dbg !23
  %1119 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !23
  %1120 = bitcast <4 x i32> %1119 to <8 x half>, !dbg !23
  %1121 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !23
  %1122 = bitcast <4 x i32> %1121 to <8 x half>, !dbg !23
  %1123 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #11, !dbg !23
  %1124 = bitcast <4 x i32> %1123 to <8 x half>, !dbg !23
  %1125 = shufflevector <8 x half> %1110, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1126 = extractvalue [2 x <8 x float>] %118, 0, !dbg !24
  %1127 = shufflevector <8 x float> %1126, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1128 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1125, <4 x float> %1127) #8, !dbg !23
  %1129 = shufflevector <8 x half> %1112, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1130 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1129, <4 x float> %1128) #8, !dbg !23
  %1131 = shufflevector <8 x half> %1114, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1132 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1131, <4 x float> %1130) #8, !dbg !23
  %1133 = shufflevector <8 x half> %1116, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1134 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1133, <4 x float> %1132) #8, !dbg !23
  %1135 = shufflevector <8 x half> %1118, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1136 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1135, <4 x float> %1134) #8, !dbg !23
  %1137 = shufflevector <8 x half> %1120, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1138 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1137, <4 x float> %1136) #8, !dbg !23
  %1139 = shufflevector <8 x half> %1122, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1140 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1139, <4 x float> %1138) #8, !dbg !23
  %1141 = shufflevector <8 x half> %1124, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1142 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1141, <4 x float> %1140) #8, !dbg !23
  %1143 = shufflevector <8 x half> %1110, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1144 = shufflevector <8 x float> %1126, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1143, <4 x float> %1144) #8, !dbg !23
  %1146 = shufflevector <8 x half> %1112, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1146, <4 x float> %1145) #8, !dbg !23
  %1148 = shufflevector <8 x half> %1114, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1148, <4 x float> %1147) #8, !dbg !23
  %1150 = shufflevector <8 x half> %1116, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1150, <4 x float> %1149) #8, !dbg !23
  %1152 = shufflevector <8 x half> %1118, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1152, <4 x float> %1151) #8, !dbg !23
  %1154 = shufflevector <8 x half> %1120, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1154, <4 x float> %1153) #8, !dbg !23
  %1156 = shufflevector <8 x half> %1122, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1156, <4 x float> %1155) #8, !dbg !23
  %1158 = shufflevector <8 x half> %1124, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1158, <4 x float> %1157) #8, !dbg !23
  %1160 = extractvalue [2 x <8 x float>] %118, 1, !dbg !24
  %1161 = shufflevector <8 x float> %1160, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1125, <4 x float> %1161) #8, !dbg !23
  %1163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1129, <4 x float> %1162) #8, !dbg !23
  %1164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1131, <4 x float> %1163) #8, !dbg !23
  %1165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1133, <4 x float> %1164) #8, !dbg !23
  %1166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1135, <4 x float> %1165) #8, !dbg !23
  %1167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1137, <4 x float> %1166) #8, !dbg !23
  %1168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1139, <4 x float> %1167) #8, !dbg !23
  %1169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1141, <4 x float> %1168) #8, !dbg !23
  %1170 = shufflevector <8 x float> %1160, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1143, <4 x float> %1170) #8, !dbg !23
  %1172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1146, <4 x float> %1171) #8, !dbg !23
  %1173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1148, <4 x float> %1172) #8, !dbg !23
  %1174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1150, <4 x float> %1173) #8, !dbg !23
  %1175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1152, <4 x float> %1174) #8, !dbg !23
  %1176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1154, <4 x float> %1175) #8, !dbg !23
  %1177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1156, <4 x float> %1176) #8, !dbg !23
  %1178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1158, <4 x float> %1177) #8, !dbg !23
  %1179 = shufflevector <4 x float> %925, <4 x float> %942, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1180 = insertvalue [2 x <8 x float>] poison, <8 x float> %1179, 0, !dbg !11
  %1181 = shufflevector <4 x float> %952, <4 x float> %961, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1182 = insertvalue [2 x <8 x float>] %1180, <8 x float> %1181, 1, !dbg !11
  %1183 = shufflevector <4 x float> %997, <4 x float> %1014, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1184 = insertvalue [2 x <8 x float>] poison, <8 x float> %1183, 0, !dbg !11
  %1185 = shufflevector <4 x float> %1024, <4 x float> %1033, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1186 = insertvalue [2 x <8 x float>] %1184, <8 x float> %1185, 1, !dbg !11
  %1187 = shufflevector <4 x float> %1072, <4 x float> %1089, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1188 = insertvalue [2 x <8 x float>] poison, <8 x float> %1187, 0, !dbg !11
  %1189 = shufflevector <4 x float> %1099, <4 x float> %1108, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1190 = insertvalue [2 x <8 x float>] %1188, <8 x float> %1189, 1, !dbg !11
  %1191 = shufflevector <4 x float> %1142, <4 x float> %1159, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1192 = insertvalue [2 x <8 x float>] poison, <8 x float> %1191, 0, !dbg !11
  %1193 = shufflevector <4 x float> %1169, <4 x float> %1178, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1194 = insertvalue [2 x <8 x float>] %1192, <8 x float> %1193, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1195 = add nuw nsw i64 %123, 1, !dbg !11
  %1196 = icmp slt i64 %1195, %83, !dbg !11
  br i1 %1196, label %117, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %540, %.._crit_edge_crit_edge
  %.pre-phi115 = phi i32 [ %.pre114, %.._crit_edge_crit_edge ], [ %858, %540 ], !dbg !11
  %.pre-phi113 = phi i32 [ %.pre112, %.._crit_edge_crit_edge ], [ %850, %540 ], !dbg !11
  %.pre-phi111 = phi i32 [ %.pre110, %.._crit_edge_crit_edge ], [ %849, %540 ], !dbg !11
  %.pre-phi107 = phi i32 [ %.pre106, %.._crit_edge_crit_edge ], [ %847, %540 ], !dbg !11
  %.lcssa95 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1182, %540 ], !dbg !11
  %.lcssa94 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %883, %540 ], !dbg !11
  %.lcssa93 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1186, %540 ], !dbg !11
  %.lcssa92 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1190, %540 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1194, %540 ], !dbg !11
  %1197 = and i32 %29, 16, !dbg !11
  %.not = icmp eq i32 %1197, 0, !dbg !11
  %1198 = and i32 %29, 32, !dbg !11
  %.not36 = icmp eq i32 %1198, 0, !dbg !11
  %1199 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !11
  %1200 = extractelement <8 x float> %1199, i64 0, !dbg !11
  %1201 = extractelement <8 x float> %1199, i64 1, !dbg !11
  %1202 = extractelement <8 x float> %1199, i64 2, !dbg !11
  %1203 = extractelement <8 x float> %1199, i64 3, !dbg !11
  %1204 = select i1 %.not, float %1201, float %1200, !dbg !11
  %1205 = icmp samesign ult i32 %.pre-phi111, %.pre-phi113, !dbg !11
  %1206 = select i1 %1205, i32 %.pre-phi111, i32 %.pre-phi107, !dbg !11
  %1207 = shl nuw nsw i32 %1206, 2, !dbg !11
  %1208 = bitcast float %1204 to i32, !dbg !11
  %1209 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1208), !dbg !11
  %1210 = bitcast i32 %1209 to float, !dbg !11
  %1211 = select i1 %.not, float %1200, float %1210, !dbg !11
  %1212 = select i1 %.not, float %1210, float %1201, !dbg !11
  %1213 = select i1 %.not, float %1203, float %1202, !dbg !11
  %1214 = bitcast float %1213 to i32, !dbg !11
  %1215 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1214), !dbg !11
  %1216 = bitcast i32 %1215 to float, !dbg !11
  %1217 = select i1 %.not, float %1202, float %1216, !dbg !11
  %1218 = select i1 %.not, float %1216, float %1203, !dbg !11
  %1219 = select i1 %.not36, float %1217, float %1211, !dbg !11
  %1220 = icmp samesign ult i32 %.pre-phi115, %.pre-phi113, !dbg !11
  %1221 = select i1 %1220, i32 %.pre-phi115, i32 %.pre-phi107, !dbg !11
  %1222 = shl nuw nsw i32 %1221, 2, !dbg !11
  %1223 = bitcast float %1219 to i32, !dbg !11
  %1224 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1223), !dbg !11
  %1225 = bitcast i32 %1224 to float, !dbg !11
  %1226 = select i1 %.not36, float %1218, float %1212, !dbg !11
  %1227 = bitcast float %1226 to i32, !dbg !11
  %1228 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1227), !dbg !11
  %1229 = bitcast i32 %1228 to float, !dbg !11
  %1230 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1231 = shufflevector <4 x i1> %1230, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1232 = insertelement <4 x float> poison, float %1211, i64 0, !dbg !11
  %1233 = insertelement <4 x float> %1232, float %1212, i64 1, !dbg !11
  %1234 = insertelement <4 x float> %1233, float %1225, i64 2, !dbg !11
  %1235 = insertelement <4 x float> %1234, float %1229, i64 3, !dbg !11
  %1236 = insertelement <4 x float> poison, float %1225, i64 0, !dbg !11
  %1237 = insertelement <4 x float> %1236, float %1229, i64 1, !dbg !11
  %1238 = insertelement <4 x float> %1237, float %1217, i64 2, !dbg !11
  %1239 = insertelement <4 x float> %1238, float %1218, i64 3, !dbg !11
  %1240 = select <4 x i1> %1231, <4 x float> %1235, <4 x float> %1239, !dbg !11
  %1241 = extractvalue [2 x <8 x float>] %.lcssa92, 0, !dbg !11
  %1242 = extractelement <8 x float> %1241, i64 0, !dbg !11
  %1243 = extractelement <8 x float> %1241, i64 1, !dbg !11
  %1244 = extractelement <8 x float> %1241, i64 2, !dbg !11
  %1245 = extractelement <8 x float> %1241, i64 3, !dbg !11
  %1246 = select i1 %.not, float %1243, float %1242, !dbg !11
  %1247 = bitcast float %1246 to i32, !dbg !11
  %1248 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1247), !dbg !11
  %1249 = bitcast i32 %1248 to float, !dbg !11
  %1250 = select i1 %.not, float %1242, float %1249, !dbg !11
  %1251 = select i1 %.not, float %1249, float %1243, !dbg !11
  %1252 = select i1 %.not, float %1245, float %1244, !dbg !11
  %1253 = bitcast float %1252 to i32, !dbg !11
  %1254 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1253), !dbg !11
  %1255 = bitcast i32 %1254 to float, !dbg !11
  %1256 = select i1 %.not, float %1244, float %1255, !dbg !11
  %1257 = select i1 %.not, float %1255, float %1245, !dbg !11
  %1258 = select i1 %.not36, float %1256, float %1250, !dbg !11
  %1259 = bitcast float %1258 to i32, !dbg !11
  %1260 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1259), !dbg !11
  %1261 = bitcast i32 %1260 to float, !dbg !11
  %1262 = select i1 %.not36, float %1257, float %1251, !dbg !11
  %1263 = bitcast float %1262 to i32, !dbg !11
  %1264 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1263), !dbg !11
  %1265 = bitcast i32 %1264 to float, !dbg !11
  %1266 = insertelement <4 x float> poison, float %1250, i64 0, !dbg !11
  %1267 = insertelement <4 x float> %1266, float %1251, i64 1, !dbg !11
  %1268 = insertelement <4 x float> %1267, float %1261, i64 2, !dbg !11
  %1269 = insertelement <4 x float> %1268, float %1265, i64 3, !dbg !11
  %1270 = insertelement <4 x float> poison, float %1261, i64 0, !dbg !11
  %1271 = insertelement <4 x float> %1270, float %1265, i64 1, !dbg !11
  %1272 = insertelement <4 x float> %1271, float %1256, i64 2, !dbg !11
  %1273 = insertelement <4 x float> %1272, float %1257, i64 3, !dbg !11
  %1274 = select <4 x i1> %1231, <4 x float> %1269, <4 x float> %1273, !dbg !11
  %1275 = extractvalue [2 x <8 x float>] %.lcssa93, 0, !dbg !11
  %1276 = extractelement <8 x float> %1275, i64 0, !dbg !11
  %1277 = extractelement <8 x float> %1275, i64 1, !dbg !11
  %1278 = extractelement <8 x float> %1275, i64 2, !dbg !11
  %1279 = extractelement <8 x float> %1275, i64 3, !dbg !11
  %1280 = select i1 %.not, float %1277, float %1276, !dbg !11
  %1281 = bitcast float %1280 to i32, !dbg !11
  %1282 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1281), !dbg !11
  %1283 = bitcast i32 %1282 to float, !dbg !11
  %1284 = select i1 %.not, float %1276, float %1283, !dbg !11
  %1285 = select i1 %.not, float %1283, float %1277, !dbg !11
  %1286 = select i1 %.not, float %1279, float %1278, !dbg !11
  %1287 = bitcast float %1286 to i32, !dbg !11
  %1288 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1287), !dbg !11
  %1289 = bitcast i32 %1288 to float, !dbg !11
  %1290 = select i1 %.not, float %1278, float %1289, !dbg !11
  %1291 = select i1 %.not, float %1289, float %1279, !dbg !11
  %1292 = select i1 %.not36, float %1290, float %1284, !dbg !11
  %1293 = bitcast float %1292 to i32, !dbg !11
  %1294 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1293), !dbg !11
  %1295 = bitcast i32 %1294 to float, !dbg !11
  %1296 = select i1 %.not36, float %1291, float %1285, !dbg !11
  %1297 = bitcast float %1296 to i32, !dbg !11
  %1298 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1297), !dbg !11
  %1299 = bitcast i32 %1298 to float, !dbg !11
  %1300 = insertelement <4 x float> poison, float %1284, i64 0, !dbg !11
  %1301 = insertelement <4 x float> %1300, float %1285, i64 1, !dbg !11
  %1302 = insertelement <4 x float> %1301, float %1295, i64 2, !dbg !11
  %1303 = insertelement <4 x float> %1302, float %1299, i64 3, !dbg !11
  %1304 = insertelement <4 x float> poison, float %1295, i64 0, !dbg !11
  %1305 = insertelement <4 x float> %1304, float %1299, i64 1, !dbg !11
  %1306 = insertelement <4 x float> %1305, float %1290, i64 2, !dbg !11
  %1307 = insertelement <4 x float> %1306, float %1291, i64 3, !dbg !11
  %1308 = select <4 x i1> %1231, <4 x float> %1303, <4 x float> %1307, !dbg !11
  %1309 = extractvalue [2 x <8 x float>] %.lcssa95, 0, !dbg !11
  %1310 = extractelement <8 x float> %1309, i64 0, !dbg !11
  %1311 = extractelement <8 x float> %1309, i64 1, !dbg !11
  %1312 = extractelement <8 x float> %1309, i64 2, !dbg !11
  %1313 = extractelement <8 x float> %1309, i64 3, !dbg !11
  %1314 = select i1 %.not, float %1311, float %1310, !dbg !11
  %1315 = bitcast float %1314 to i32, !dbg !11
  %1316 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1315), !dbg !11
  %1317 = bitcast i32 %1316 to float, !dbg !11
  %1318 = select i1 %.not, float %1310, float %1317, !dbg !11
  %1319 = select i1 %.not, float %1317, float %1311, !dbg !11
  %1320 = select i1 %.not, float %1313, float %1312, !dbg !11
  %1321 = bitcast float %1320 to i32, !dbg !11
  %1322 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1321), !dbg !11
  %1323 = bitcast i32 %1322 to float, !dbg !11
  %1324 = select i1 %.not, float %1312, float %1323, !dbg !11
  %1325 = select i1 %.not, float %1323, float %1313, !dbg !11
  %1326 = select i1 %.not36, float %1324, float %1318, !dbg !11
  %1327 = bitcast float %1326 to i32, !dbg !11
  %1328 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1327), !dbg !11
  %1329 = select i1 %.not36, float %1325, float %1319, !dbg !11
  %1330 = bitcast float %1329 to i32, !dbg !11
  %1331 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1330), !dbg !11
  %1332 = extractvalue [2 x <1 x float>] %.lcssa94, 0, !dbg !25
  %1333 = shufflevector <1 x float> %1332, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %1334 = extractelement <8 x float> %1199, i64 4, !dbg !11
  %1335 = extractelement <8 x float> %1199, i64 5, !dbg !11
  %1336 = extractelement <8 x float> %1199, i64 6, !dbg !11
  %1337 = extractelement <8 x float> %1199, i64 7, !dbg !11
  %1338 = select i1 %.not, float %1335, float %1334, !dbg !11
  %1339 = bitcast float %1338 to i32, !dbg !11
  %1340 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1339), !dbg !11
  %1341 = bitcast i32 %1340 to float, !dbg !11
  %1342 = select i1 %.not, float %1334, float %1341, !dbg !11
  %1343 = select i1 %.not, float %1341, float %1335, !dbg !11
  %1344 = select i1 %.not, float %1337, float %1336, !dbg !11
  %1345 = bitcast float %1344 to i32, !dbg !11
  %1346 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1345), !dbg !11
  %1347 = bitcast i32 %1346 to float, !dbg !11
  %1348 = select i1 %.not, float %1336, float %1347, !dbg !11
  %1349 = select i1 %.not, float %1347, float %1337, !dbg !11
  %1350 = select i1 %.not36, float %1348, float %1342, !dbg !11
  %1351 = bitcast float %1350 to i32, !dbg !11
  %1352 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1351), !dbg !11
  %1353 = bitcast i32 %1352 to float, !dbg !11
  %1354 = select i1 %.not36, float %1349, float %1343, !dbg !11
  %1355 = bitcast float %1354 to i32, !dbg !11
  %1356 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1355), !dbg !11
  %1357 = bitcast i32 %1356 to float, !dbg !11
  %1358 = insertelement <4 x float> poison, float %1342, i64 0, !dbg !11
  %1359 = insertelement <4 x float> %1358, float %1343, i64 1, !dbg !11
  %1360 = insertelement <4 x float> %1359, float %1353, i64 2, !dbg !11
  %1361 = insertelement <4 x float> %1360, float %1357, i64 3, !dbg !11
  %1362 = insertelement <4 x float> poison, float %1353, i64 0, !dbg !11
  %1363 = insertelement <4 x float> %1362, float %1357, i64 1, !dbg !11
  %1364 = insertelement <4 x float> %1363, float %1348, i64 2, !dbg !11
  %1365 = insertelement <4 x float> %1364, float %1349, i64 3, !dbg !11
  %1366 = select <4 x i1> %1231, <4 x float> %1361, <4 x float> %1365, !dbg !11
  %1367 = extractelement <8 x float> %1241, i64 4, !dbg !11
  %1368 = extractelement <8 x float> %1241, i64 5, !dbg !11
  %1369 = extractelement <8 x float> %1241, i64 6, !dbg !11
  %1370 = extractelement <8 x float> %1241, i64 7, !dbg !11
  %1371 = select i1 %.not, float %1368, float %1367, !dbg !11
  %1372 = bitcast float %1371 to i32, !dbg !11
  %1373 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1372), !dbg !11
  %1374 = bitcast i32 %1373 to float, !dbg !11
  %1375 = select i1 %.not, float %1367, float %1374, !dbg !11
  %1376 = select i1 %.not, float %1374, float %1368, !dbg !11
  %1377 = select i1 %.not, float %1370, float %1369, !dbg !11
  %1378 = bitcast float %1377 to i32, !dbg !11
  %1379 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1378), !dbg !11
  %1380 = bitcast i32 %1379 to float, !dbg !11
  %1381 = select i1 %.not, float %1369, float %1380, !dbg !11
  %1382 = select i1 %.not, float %1380, float %1370, !dbg !11
  %1383 = select i1 %.not36, float %1381, float %1375, !dbg !11
  %1384 = bitcast float %1383 to i32, !dbg !11
  %1385 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1384), !dbg !11
  %1386 = bitcast i32 %1385 to float, !dbg !11
  %1387 = select i1 %.not36, float %1382, float %1376, !dbg !11
  %1388 = bitcast float %1387 to i32, !dbg !11
  %1389 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1388), !dbg !11
  %1390 = bitcast i32 %1389 to float, !dbg !11
  %1391 = insertelement <4 x float> poison, float %1375, i64 0, !dbg !11
  %1392 = insertelement <4 x float> %1391, float %1376, i64 1, !dbg !11
  %1393 = insertelement <4 x float> %1392, float %1386, i64 2, !dbg !11
  %1394 = insertelement <4 x float> %1393, float %1390, i64 3, !dbg !11
  %1395 = insertelement <4 x float> poison, float %1386, i64 0, !dbg !11
  %1396 = insertelement <4 x float> %1395, float %1390, i64 1, !dbg !11
  %1397 = insertelement <4 x float> %1396, float %1381, i64 2, !dbg !11
  %1398 = insertelement <4 x float> %1397, float %1382, i64 3, !dbg !11
  %1399 = select <4 x i1> %1231, <4 x float> %1394, <4 x float> %1398, !dbg !11
  %1400 = extractelement <8 x float> %1275, i64 4, !dbg !11
  %1401 = extractelement <8 x float> %1275, i64 5, !dbg !11
  %1402 = extractelement <8 x float> %1275, i64 6, !dbg !11
  %1403 = extractelement <8 x float> %1275, i64 7, !dbg !11
  %1404 = select i1 %.not, float %1401, float %1400, !dbg !11
  %1405 = bitcast float %1404 to i32, !dbg !11
  %1406 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1405), !dbg !11
  %1407 = bitcast i32 %1406 to float, !dbg !11
  %1408 = select i1 %.not, float %1400, float %1407, !dbg !11
  %1409 = select i1 %.not, float %1407, float %1401, !dbg !11
  %1410 = select i1 %.not, float %1403, float %1402, !dbg !11
  %1411 = bitcast float %1410 to i32, !dbg !11
  %1412 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1411), !dbg !11
  %1413 = bitcast i32 %1412 to float, !dbg !11
  %1414 = select i1 %.not, float %1402, float %1413, !dbg !11
  %1415 = select i1 %.not, float %1413, float %1403, !dbg !11
  %1416 = select i1 %.not36, float %1414, float %1408, !dbg !11
  %1417 = bitcast float %1416 to i32, !dbg !11
  %1418 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1417), !dbg !11
  %1419 = bitcast i32 %1418 to float, !dbg !11
  %1420 = select i1 %.not36, float %1415, float %1409, !dbg !11
  %1421 = bitcast float %1420 to i32, !dbg !11
  %1422 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1421), !dbg !11
  %1423 = bitcast i32 %1422 to float, !dbg !11
  %1424 = insertelement <4 x float> poison, float %1408, i64 0, !dbg !11
  %1425 = insertelement <4 x float> %1424, float %1409, i64 1, !dbg !11
  %1426 = insertelement <4 x float> %1425, float %1419, i64 2, !dbg !11
  %1427 = insertelement <4 x float> %1426, float %1423, i64 3, !dbg !11
  %1428 = insertelement <4 x float> poison, float %1419, i64 0, !dbg !11
  %1429 = insertelement <4 x float> %1428, float %1423, i64 1, !dbg !11
  %1430 = insertelement <4 x float> %1429, float %1414, i64 2, !dbg !11
  %1431 = insertelement <4 x float> %1430, float %1415, i64 3, !dbg !11
  %1432 = select <4 x i1> %1231, <4 x float> %1427, <4 x float> %1431, !dbg !11
  %1433 = extractelement <8 x float> %1309, i64 4, !dbg !11
  %1434 = extractelement <8 x float> %1309, i64 5, !dbg !11
  %1435 = extractelement <8 x float> %1309, i64 6, !dbg !11
  %1436 = extractelement <8 x float> %1309, i64 7, !dbg !11
  %1437 = select i1 %.not, float %1434, float %1433, !dbg !11
  %1438 = bitcast float %1437 to i32, !dbg !11
  %1439 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1438), !dbg !11
  %1440 = bitcast i32 %1439 to float, !dbg !11
  %1441 = select i1 %.not, float %1433, float %1440, !dbg !11
  %1442 = select i1 %.not, float %1440, float %1434, !dbg !11
  %1443 = select i1 %.not, float %1436, float %1435, !dbg !11
  %1444 = bitcast float %1443 to i32, !dbg !11
  %1445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1444), !dbg !11
  %1446 = bitcast i32 %1445 to float, !dbg !11
  %1447 = select i1 %.not, float %1435, float %1446, !dbg !11
  %1448 = select i1 %.not, float %1446, float %1436, !dbg !11
  %1449 = select i1 %.not36, float %1447, float %1441, !dbg !11
  %1450 = bitcast float %1449 to i32, !dbg !11
  %1451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1450), !dbg !11
  %1452 = select i1 %.not36, float %1448, float %1442, !dbg !11
  %1453 = bitcast float %1452 to i32, !dbg !11
  %1454 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1453), !dbg !11
  %1455 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !11
  %1456 = extractelement <8 x float> %1455, i64 0, !dbg !11
  %1457 = extractelement <8 x float> %1455, i64 1, !dbg !11
  %1458 = extractelement <8 x float> %1455, i64 2, !dbg !11
  %1459 = extractelement <8 x float> %1455, i64 3, !dbg !11
  %1460 = select i1 %.not, float %1457, float %1456, !dbg !11
  %1461 = bitcast float %1460 to i32, !dbg !11
  %1462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1461), !dbg !11
  %1463 = bitcast i32 %1462 to float, !dbg !11
  %1464 = select i1 %.not, float %1456, float %1463, !dbg !11
  %1465 = select i1 %.not, float %1463, float %1457, !dbg !11
  %1466 = select i1 %.not, float %1459, float %1458, !dbg !11
  %1467 = bitcast float %1466 to i32, !dbg !11
  %1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1467), !dbg !11
  %1469 = bitcast i32 %1468 to float, !dbg !11
  %1470 = select i1 %.not, float %1458, float %1469, !dbg !11
  %1471 = select i1 %.not, float %1469, float %1459, !dbg !11
  %1472 = select i1 %.not36, float %1470, float %1464, !dbg !11
  %1473 = bitcast float %1472 to i32, !dbg !11
  %1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1473), !dbg !11
  %1475 = bitcast i32 %1474 to float, !dbg !11
  %1476 = select i1 %.not36, float %1471, float %1465, !dbg !11
  %1477 = bitcast float %1476 to i32, !dbg !11
  %1478 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1477), !dbg !11
  %1479 = bitcast i32 %1478 to float, !dbg !11
  %1480 = insertelement <4 x float> poison, float %1464, i64 0, !dbg !11
  %1481 = insertelement <4 x float> %1480, float %1465, i64 1, !dbg !11
  %1482 = insertelement <4 x float> %1481, float %1475, i64 2, !dbg !11
  %1483 = insertelement <4 x float> %1482, float %1479, i64 3, !dbg !11
  %1484 = insertelement <4 x float> poison, float %1475, i64 0, !dbg !11
  %1485 = insertelement <4 x float> %1484, float %1479, i64 1, !dbg !11
  %1486 = insertelement <4 x float> %1485, float %1470, i64 2, !dbg !11
  %1487 = insertelement <4 x float> %1486, float %1471, i64 3, !dbg !11
  %1488 = select <4 x i1> %1231, <4 x float> %1483, <4 x float> %1487, !dbg !11
  %1489 = extractvalue [2 x <8 x float>] %.lcssa92, 1, !dbg !11
  %1490 = extractelement <8 x float> %1489, i64 0, !dbg !11
  %1491 = extractelement <8 x float> %1489, i64 1, !dbg !11
  %1492 = extractelement <8 x float> %1489, i64 2, !dbg !11
  %1493 = extractelement <8 x float> %1489, i64 3, !dbg !11
  %1494 = select i1 %.not, float %1491, float %1490, !dbg !11
  %1495 = bitcast float %1494 to i32, !dbg !11
  %1496 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1495), !dbg !11
  %1497 = bitcast i32 %1496 to float, !dbg !11
  %1498 = select i1 %.not, float %1490, float %1497, !dbg !11
  %1499 = select i1 %.not, float %1497, float %1491, !dbg !11
  %1500 = select i1 %.not, float %1493, float %1492, !dbg !11
  %1501 = bitcast float %1500 to i32, !dbg !11
  %1502 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1501), !dbg !11
  %1503 = bitcast i32 %1502 to float, !dbg !11
  %1504 = select i1 %.not, float %1492, float %1503, !dbg !11
  %1505 = select i1 %.not, float %1503, float %1493, !dbg !11
  %1506 = select i1 %.not36, float %1504, float %1498, !dbg !11
  %1507 = bitcast float %1506 to i32, !dbg !11
  %1508 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1507), !dbg !11
  %1509 = bitcast i32 %1508 to float, !dbg !11
  %1510 = select i1 %.not36, float %1505, float %1499, !dbg !11
  %1511 = bitcast float %1510 to i32, !dbg !11
  %1512 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1511), !dbg !11
  %1513 = bitcast i32 %1512 to float, !dbg !11
  %1514 = insertelement <4 x float> poison, float %1498, i64 0, !dbg !11
  %1515 = insertelement <4 x float> %1514, float %1499, i64 1, !dbg !11
  %1516 = insertelement <4 x float> %1515, float %1509, i64 2, !dbg !11
  %1517 = insertelement <4 x float> %1516, float %1513, i64 3, !dbg !11
  %1518 = insertelement <4 x float> poison, float %1509, i64 0, !dbg !11
  %1519 = insertelement <4 x float> %1518, float %1513, i64 1, !dbg !11
  %1520 = insertelement <4 x float> %1519, float %1504, i64 2, !dbg !11
  %1521 = insertelement <4 x float> %1520, float %1505, i64 3, !dbg !11
  %1522 = select <4 x i1> %1231, <4 x float> %1517, <4 x float> %1521, !dbg !11
  %1523 = extractvalue [2 x <8 x float>] %.lcssa93, 1, !dbg !11
  %1524 = extractelement <8 x float> %1523, i64 0, !dbg !11
  %1525 = extractelement <8 x float> %1523, i64 1, !dbg !11
  %1526 = extractelement <8 x float> %1523, i64 2, !dbg !11
  %1527 = extractelement <8 x float> %1523, i64 3, !dbg !11
  %1528 = select i1 %.not, float %1525, float %1524, !dbg !11
  %1529 = bitcast float %1528 to i32, !dbg !11
  %1530 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1529), !dbg !11
  %1531 = bitcast i32 %1530 to float, !dbg !11
  %1532 = select i1 %.not, float %1524, float %1531, !dbg !11
  %1533 = select i1 %.not, float %1531, float %1525, !dbg !11
  %1534 = select i1 %.not, float %1527, float %1526, !dbg !11
  %1535 = bitcast float %1534 to i32, !dbg !11
  %1536 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1535), !dbg !11
  %1537 = bitcast i32 %1536 to float, !dbg !11
  %1538 = select i1 %.not, float %1526, float %1537, !dbg !11
  %1539 = select i1 %.not, float %1537, float %1527, !dbg !11
  %1540 = select i1 %.not36, float %1538, float %1532, !dbg !11
  %1541 = bitcast float %1540 to i32, !dbg !11
  %1542 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1541), !dbg !11
  %1543 = bitcast i32 %1542 to float, !dbg !11
  %1544 = select i1 %.not36, float %1539, float %1533, !dbg !11
  %1545 = bitcast float %1544 to i32, !dbg !11
  %1546 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1545), !dbg !11
  %1547 = bitcast i32 %1546 to float, !dbg !11
  %1548 = insertelement <4 x float> poison, float %1532, i64 0, !dbg !11
  %1549 = insertelement <4 x float> %1548, float %1533, i64 1, !dbg !11
  %1550 = insertelement <4 x float> %1549, float %1543, i64 2, !dbg !11
  %1551 = insertelement <4 x float> %1550, float %1547, i64 3, !dbg !11
  %1552 = insertelement <4 x float> poison, float %1543, i64 0, !dbg !11
  %1553 = insertelement <4 x float> %1552, float %1547, i64 1, !dbg !11
  %1554 = insertelement <4 x float> %1553, float %1538, i64 2, !dbg !11
  %1555 = insertelement <4 x float> %1554, float %1539, i64 3, !dbg !11
  %1556 = select <4 x i1> %1231, <4 x float> %1551, <4 x float> %1555, !dbg !11
  %1557 = extractvalue [2 x <8 x float>] %.lcssa95, 1, !dbg !11
  %1558 = extractelement <8 x float> %1557, i64 0, !dbg !11
  %1559 = extractelement <8 x float> %1557, i64 1, !dbg !11
  %1560 = extractelement <8 x float> %1557, i64 2, !dbg !11
  %1561 = extractelement <8 x float> %1557, i64 3, !dbg !11
  %1562 = select i1 %.not, float %1559, float %1558, !dbg !11
  %1563 = bitcast float %1562 to i32, !dbg !11
  %1564 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1563), !dbg !11
  %1565 = bitcast i32 %1564 to float, !dbg !11
  %1566 = select i1 %.not, float %1558, float %1565, !dbg !11
  %1567 = select i1 %.not, float %1565, float %1559, !dbg !11
  %1568 = select i1 %.not, float %1561, float %1560, !dbg !11
  %1569 = bitcast float %1568 to i32, !dbg !11
  %1570 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1569), !dbg !11
  %1571 = bitcast i32 %1570 to float, !dbg !11
  %1572 = select i1 %.not, float %1560, float %1571, !dbg !11
  %1573 = select i1 %.not, float %1571, float %1561, !dbg !11
  %1574 = select i1 %.not36, float %1572, float %1566, !dbg !11
  %1575 = bitcast float %1574 to i32, !dbg !11
  %1576 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1575), !dbg !11
  %1577 = select i1 %.not36, float %1573, float %1567, !dbg !11
  %1578 = bitcast float %1577 to i32, !dbg !11
  %1579 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1578), !dbg !11
  %1580 = extractvalue [2 x <1 x float>] %.lcssa94, 1, !dbg !25
  %1581 = shufflevector <1 x float> %1580, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %1582 = extractelement <8 x float> %1455, i64 4, !dbg !11
  %1583 = extractelement <8 x float> %1455, i64 5, !dbg !11
  %1584 = extractelement <8 x float> %1455, i64 6, !dbg !11
  %1585 = extractelement <8 x float> %1455, i64 7, !dbg !11
  %1586 = select i1 %.not, float %1583, float %1582, !dbg !11
  %1587 = bitcast float %1586 to i32, !dbg !11
  %1588 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1587), !dbg !11
  %1589 = bitcast i32 %1588 to float, !dbg !11
  %1590 = select i1 %.not, float %1582, float %1589, !dbg !11
  %1591 = select i1 %.not, float %1589, float %1583, !dbg !11
  %1592 = select i1 %.not, float %1585, float %1584, !dbg !11
  %1593 = bitcast float %1592 to i32, !dbg !11
  %1594 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1593), !dbg !11
  %1595 = bitcast i32 %1594 to float, !dbg !11
  %1596 = select i1 %.not, float %1584, float %1595, !dbg !11
  %1597 = select i1 %.not, float %1595, float %1585, !dbg !11
  %1598 = select i1 %.not36, float %1596, float %1590, !dbg !11
  %1599 = bitcast float %1598 to i32, !dbg !11
  %1600 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1599), !dbg !11
  %1601 = bitcast i32 %1600 to float, !dbg !11
  %1602 = select i1 %.not36, float %1597, float %1591, !dbg !11
  %1603 = bitcast float %1602 to i32, !dbg !11
  %1604 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1603), !dbg !11
  %1605 = bitcast i32 %1604 to float, !dbg !11
  %1606 = insertelement <4 x float> poison, float %1590, i64 0, !dbg !11
  %1607 = insertelement <4 x float> %1606, float %1591, i64 1, !dbg !11
  %1608 = insertelement <4 x float> %1607, float %1601, i64 2, !dbg !11
  %1609 = insertelement <4 x float> %1608, float %1605, i64 3, !dbg !11
  %1610 = insertelement <4 x float> poison, float %1601, i64 0, !dbg !11
  %1611 = insertelement <4 x float> %1610, float %1605, i64 1, !dbg !11
  %1612 = insertelement <4 x float> %1611, float %1596, i64 2, !dbg !11
  %1613 = insertelement <4 x float> %1612, float %1597, i64 3, !dbg !11
  %1614 = select <4 x i1> %1231, <4 x float> %1609, <4 x float> %1613, !dbg !11
  %1615 = extractelement <8 x float> %1489, i64 4, !dbg !11
  %1616 = extractelement <8 x float> %1489, i64 5, !dbg !11
  %1617 = extractelement <8 x float> %1489, i64 6, !dbg !11
  %1618 = extractelement <8 x float> %1489, i64 7, !dbg !11
  %1619 = select i1 %.not, float %1616, float %1615, !dbg !11
  %1620 = bitcast float %1619 to i32, !dbg !11
  %1621 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1620), !dbg !11
  %1622 = bitcast i32 %1621 to float, !dbg !11
  %1623 = select i1 %.not, float %1615, float %1622, !dbg !11
  %1624 = select i1 %.not, float %1622, float %1616, !dbg !11
  %1625 = select i1 %.not, float %1618, float %1617, !dbg !11
  %1626 = bitcast float %1625 to i32, !dbg !11
  %1627 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1626), !dbg !11
  %1628 = bitcast i32 %1627 to float, !dbg !11
  %1629 = select i1 %.not, float %1617, float %1628, !dbg !11
  %1630 = select i1 %.not, float %1628, float %1618, !dbg !11
  %1631 = select i1 %.not36, float %1629, float %1623, !dbg !11
  %1632 = bitcast float %1631 to i32, !dbg !11
  %1633 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1632), !dbg !11
  %1634 = bitcast i32 %1633 to float, !dbg !11
  %1635 = select i1 %.not36, float %1630, float %1624, !dbg !11
  %1636 = bitcast float %1635 to i32, !dbg !11
  %1637 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1636), !dbg !11
  %1638 = bitcast i32 %1637 to float, !dbg !11
  %1639 = insertelement <4 x float> poison, float %1623, i64 0, !dbg !11
  %1640 = insertelement <4 x float> %1639, float %1624, i64 1, !dbg !11
  %1641 = insertelement <4 x float> %1640, float %1634, i64 2, !dbg !11
  %1642 = insertelement <4 x float> %1641, float %1638, i64 3, !dbg !11
  %1643 = insertelement <4 x float> poison, float %1634, i64 0, !dbg !11
  %1644 = insertelement <4 x float> %1643, float %1638, i64 1, !dbg !11
  %1645 = insertelement <4 x float> %1644, float %1629, i64 2, !dbg !11
  %1646 = insertelement <4 x float> %1645, float %1630, i64 3, !dbg !11
  %1647 = select <4 x i1> %1231, <4 x float> %1642, <4 x float> %1646, !dbg !11
  %1648 = extractelement <8 x float> %1523, i64 4, !dbg !11
  %1649 = extractelement <8 x float> %1523, i64 5, !dbg !11
  %1650 = extractelement <8 x float> %1523, i64 6, !dbg !11
  %1651 = extractelement <8 x float> %1523, i64 7, !dbg !11
  %1652 = select i1 %.not, float %1649, float %1648, !dbg !11
  %1653 = bitcast float %1652 to i32, !dbg !11
  %1654 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1653), !dbg !11
  %1655 = bitcast i32 %1654 to float, !dbg !11
  %1656 = select i1 %.not, float %1648, float %1655, !dbg !11
  %1657 = select i1 %.not, float %1655, float %1649, !dbg !11
  %1658 = select i1 %.not, float %1651, float %1650, !dbg !11
  %1659 = bitcast float %1658 to i32, !dbg !11
  %1660 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1659), !dbg !11
  %1661 = bitcast i32 %1660 to float, !dbg !11
  %1662 = select i1 %.not, float %1650, float %1661, !dbg !11
  %1663 = select i1 %.not, float %1661, float %1651, !dbg !11
  %1664 = select i1 %.not36, float %1662, float %1656, !dbg !11
  %1665 = bitcast float %1664 to i32, !dbg !11
  %1666 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1665), !dbg !11
  %1667 = bitcast i32 %1666 to float, !dbg !11
  %1668 = select i1 %.not36, float %1663, float %1657, !dbg !11
  %1669 = bitcast float %1668 to i32, !dbg !11
  %1670 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1669), !dbg !11
  %1671 = bitcast i32 %1670 to float, !dbg !11
  %1672 = insertelement <4 x float> poison, float %1656, i64 0, !dbg !11
  %1673 = insertelement <4 x float> %1672, float %1657, i64 1, !dbg !11
  %1674 = insertelement <4 x float> %1673, float %1667, i64 2, !dbg !11
  %1675 = insertelement <4 x float> %1674, float %1671, i64 3, !dbg !11
  %1676 = insertelement <4 x float> poison, float %1667, i64 0, !dbg !11
  %1677 = insertelement <4 x float> %1676, float %1671, i64 1, !dbg !11
  %1678 = insertelement <4 x float> %1677, float %1662, i64 2, !dbg !11
  %1679 = insertelement <4 x float> %1678, float %1663, i64 3, !dbg !11
  %1680 = select <4 x i1> %1231, <4 x float> %1675, <4 x float> %1679, !dbg !11
  %1681 = extractelement <8 x float> %1557, i64 4, !dbg !11
  %1682 = extractelement <8 x float> %1557, i64 5, !dbg !11
  %1683 = extractelement <8 x float> %1557, i64 6, !dbg !11
  %1684 = extractelement <8 x float> %1557, i64 7, !dbg !11
  %1685 = select i1 %.not, float %1682, float %1681, !dbg !11
  %1686 = bitcast float %1685 to i32, !dbg !11
  %1687 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1686), !dbg !11
  %1688 = bitcast i32 %1687 to float, !dbg !11
  %1689 = select i1 %.not, float %1681, float %1688, !dbg !11
  %1690 = select i1 %.not, float %1688, float %1682, !dbg !11
  %1691 = select i1 %.not, float %1684, float %1683, !dbg !11
  %1692 = bitcast float %1691 to i32, !dbg !11
  %1693 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1207, i32 %1692), !dbg !11
  %1694 = bitcast i32 %1693 to float, !dbg !11
  %1695 = select i1 %.not, float %1683, float %1694, !dbg !11
  %1696 = select i1 %.not, float %1694, float %1684, !dbg !11
  %1697 = select i1 %.not36, float %1695, float %1689, !dbg !11
  %1698 = bitcast float %1697 to i32, !dbg !11
  %1699 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1698), !dbg !11
  %1700 = select i1 %.not36, float %1696, float %1690, !dbg !11
  %1701 = bitcast float %1700 to i32, !dbg !11
  %1702 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1222, i32 %1701), !dbg !11
  %1703 = bitcast i32 %1699 to float, !dbg !11
  %1704 = bitcast i32 %1702 to float, !dbg !11
  %1705 = insertelement <4 x float> poison, float %1689, i64 0, !dbg !11
  %1706 = insertelement <4 x float> %1705, float %1690, i64 1, !dbg !11
  %1707 = insertelement <4 x float> %1706, float %1703, i64 2, !dbg !11
  %1708 = insertelement <4 x float> %1707, float %1704, i64 3, !dbg !11
  %1709 = insertelement <4 x float> poison, float %1703, i64 0, !dbg !11
  %1710 = insertelement <4 x float> %1709, float %1704, i64 1, !dbg !11
  %1711 = insertelement <4 x float> %1710, float %1695, i64 2, !dbg !11
  %1712 = insertelement <4 x float> %1711, float %1696, i64 3, !dbg !11
  %1713 = select <4 x i1> %1231, <4 x float> %1708, <4 x float> %1712, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1581, !dbg !25
  %normalized = fmul <4 x float> %row.reciprocal, %1713, !dbg !25
  %1714 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !26
  %1715 = extractelement <4 x half> %1714, i64 3, !dbg !26
  %1716 = extractelement <4 x half> %1714, i64 2, !dbg !26
  %1717 = extractelement <4 x half> %1714, i64 1, !dbg !26
  %1718 = extractelement <4 x half> %1714, i64 0, !dbg !26
  %1719 = bitcast i32 %1576 to float, !dbg !11
  %1720 = bitcast i32 %1579 to float, !dbg !11
  %1721 = insertelement <4 x float> poison, float %1566, i64 0, !dbg !11
  %1722 = insertelement <4 x float> %1721, float %1567, i64 1, !dbg !11
  %1723 = insertelement <4 x float> %1722, float %1719, i64 2, !dbg !11
  %1724 = insertelement <4 x float> %1723, float %1720, i64 3, !dbg !11
  %1725 = insertelement <4 x float> poison, float %1719, i64 0, !dbg !11
  %1726 = insertelement <4 x float> %1725, float %1720, i64 1, !dbg !11
  %1727 = insertelement <4 x float> %1726, float %1572, i64 2, !dbg !11
  %1728 = insertelement <4 x float> %1727, float %1573, i64 3, !dbg !11
  %1729 = select <4 x i1> %1231, <4 x float> %1724, <4 x float> %1728, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %1729, !dbg !25
  %1730 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !26
  %1731 = extractelement <4 x half> %1730, i64 3, !dbg !26
  %1732 = extractelement <4 x half> %1730, i64 2, !dbg !26
  %1733 = extractelement <4 x half> %1730, i64 1, !dbg !26
  %1734 = extractelement <4 x half> %1730, i64 0, !dbg !26
  %1735 = bitcast i32 %1451 to float, !dbg !11
  %1736 = bitcast i32 %1454 to float, !dbg !11
  %1737 = insertelement <4 x float> poison, float %1441, i64 0, !dbg !11
  %1738 = insertelement <4 x float> %1737, float %1442, i64 1, !dbg !11
  %1739 = insertelement <4 x float> %1738, float %1735, i64 2, !dbg !11
  %1740 = insertelement <4 x float> %1739, float %1736, i64 3, !dbg !11
  %1741 = insertelement <4 x float> poison, float %1735, i64 0, !dbg !11
  %1742 = insertelement <4 x float> %1741, float %1736, i64 1, !dbg !11
  %1743 = insertelement <4 x float> %1742, float %1447, i64 2, !dbg !11
  %1744 = insertelement <4 x float> %1743, float %1448, i64 3, !dbg !11
  %1745 = select <4 x i1> %1231, <4 x float> %1740, <4 x float> %1744, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1333, !dbg !25
  %normalized106 = fmul <4 x float> %row.reciprocal105, %1745, !dbg !25
  %1746 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !26
  %1747 = extractelement <4 x half> %1746, i64 3, !dbg !26
  %1748 = extractelement <4 x half> %1746, i64 2, !dbg !26
  %1749 = extractelement <4 x half> %1746, i64 1, !dbg !26
  %1750 = extractelement <4 x half> %1746, i64 0, !dbg !26
  %1751 = bitcast i32 %1328 to float, !dbg !11
  %1752 = bitcast i32 %1331 to float, !dbg !11
  %1753 = insertelement <4 x float> poison, float %1318, i64 0, !dbg !11
  %1754 = insertelement <4 x float> %1753, float %1319, i64 1, !dbg !11
  %1755 = insertelement <4 x float> %1754, float %1751, i64 2, !dbg !11
  %1756 = insertelement <4 x float> %1755, float %1752, i64 3, !dbg !11
  %1757 = insertelement <4 x float> poison, float %1751, i64 0, !dbg !11
  %1758 = insertelement <4 x float> %1757, float %1752, i64 1, !dbg !11
  %1759 = insertelement <4 x float> %1758, float %1324, i64 2, !dbg !11
  %1760 = insertelement <4 x float> %1759, float %1325, i64 3, !dbg !11
  %1761 = select <4 x i1> %1231, <4 x float> %1756, <4 x float> %1760, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %1761, !dbg !25
  %1762 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !26
  %1763 = extractelement <4 x half> %1762, i64 3, !dbg !26
  %1764 = extractelement <4 x half> %1762, i64 2, !dbg !26
  %1765 = extractelement <4 x half> %1762, i64 1, !dbg !26
  %1766 = extractelement <4 x half> %1762, i64 0, !dbg !26
  %address.thread999 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !27, !range !9, !frisk.register_lifetimes !5
  %address.local1000 = zext nneg i32 %address.thread999 to i64, !dbg !8
  %1767 = lshr i64 %address.local1000, 2, !dbg !27
  %address.local1003 = and i64 %1767, 48, !dbg !27
  %address.local1005 = and i64 %address.local1000, 240, !dbg !27
  %address.local1004 = add i64 %26, %address.local1000, !dbg !27
  %address.local1006 = sub i64 %address.local1004, %address.local1005, !dbg !27
  %address.local1007 = add i64 %address.local1006, %address.local1003, !dbg !27
  %1768 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !27
  %address.local1010.idx = shl nuw nsw i64 %address.local1007, 8, !dbg !27
  %address.local1010 = getelementptr inbounds nuw i8, ptr addrspace(1) %1768, i64 %address.local1010.idx, !dbg !27
  %address.local1012 = and i64 %1767, 12, !dbg !10
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1012, !dbg !27
  store half %1766, ptr addrspace(1) %address.local1013, align 2, !dbg !27
  %address.local1015 = or disjoint i64 %address.local1012, 1, !dbg !27
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1015, !dbg !27
  store half %1765, ptr addrspace(1) %address.local1016, align 2, !dbg !27
  %address.local1018 = or disjoint i64 %address.local1012, 2, !dbg !27
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1018, !dbg !27
  store half %1764, ptr addrspace(1) %address.local1019, align 2, !dbg !27
  %address.local1021 = or disjoint i64 %address.local1012, 3, !dbg !27
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1021, !dbg !27
  store half %1763, ptr addrspace(1) %address.local1022, align 2, !dbg !27
  %address.local1024 = or disjoint i64 %address.local1012, 16, !dbg !27
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1024, !dbg !27
  store half %1750, ptr addrspace(1) %address.local1025, align 2, !dbg !27
  %address.local1027 = or disjoint i64 %address.local1012, 17, !dbg !27
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1027, !dbg !27
  store half %1749, ptr addrspace(1) %address.local1028, align 2, !dbg !27
  %address.local1030 = or disjoint i64 %address.local1012, 18, !dbg !27
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1030, !dbg !27
  store half %1748, ptr addrspace(1) %address.local1031, align 2, !dbg !27
  %address.local1033 = or disjoint i64 %address.local1012, 19, !dbg !27
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1033, !dbg !27
  store half %1747, ptr addrspace(1) %address.local1034, align 2, !dbg !27
  %address.local1038 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1010, i64 16384, !dbg !27
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1012, !dbg !27
  store half %1734, ptr addrspace(1) %address.local1039, align 2, !dbg !27
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1015, !dbg !27
  store half %1733, ptr addrspace(1) %address.local1041, align 2, !dbg !27
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1018, !dbg !27
  store half %1732, ptr addrspace(1) %address.local1043, align 2, !dbg !27
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1021, !dbg !27
  store half %1731, ptr addrspace(1) %address.local1045, align 2, !dbg !27
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1024, !dbg !27
  store half %1718, ptr addrspace(1) %address.local1047, align 2, !dbg !27
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1027, !dbg !27
  store half %1717, ptr addrspace(1) %address.local1049, align 2, !dbg !27
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1030, !dbg !27
  store half %1716, ptr addrspace(1) %address.local1051, align 2, !dbg !27
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1033, !dbg !27
  store half %1715, ptr addrspace(1) %address.local1053, align 2, !dbg !27
  %normalized109 = fmul <4 x float> %row.reciprocal, %1680, !dbg !25
  %1769 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !26
  %1770 = extractelement <4 x half> %1769, i64 3, !dbg !26
  %1771 = extractelement <4 x half> %1769, i64 2, !dbg !26
  %1772 = extractelement <4 x half> %1769, i64 1, !dbg !26
  %1773 = extractelement <4 x half> %1769, i64 0, !dbg !26
  %normalized110 = fmul <4 x float> %row.reciprocal, %1556, !dbg !25
  %1774 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !26
  %1775 = extractelement <4 x half> %1774, i64 3, !dbg !26
  %1776 = extractelement <4 x half> %1774, i64 2, !dbg !26
  %1777 = extractelement <4 x half> %1774, i64 1, !dbg !26
  %1778 = extractelement <4 x half> %1774, i64 0, !dbg !26
  %normalized112 = fmul <4 x float> %row.reciprocal105, %1432, !dbg !25
  %1779 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !26
  %1780 = extractelement <4 x half> %1779, i64 3, !dbg !26
  %1781 = extractelement <4 x half> %1779, i64 2, !dbg !26
  %1782 = extractelement <4 x half> %1779, i64 1, !dbg !26
  %1783 = extractelement <4 x half> %1779, i64 0, !dbg !26
  %normalized113 = fmul <4 x float> %row.reciprocal105, %1308, !dbg !25
  %1784 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !26
  %1785 = extractelement <4 x half> %1784, i64 3, !dbg !26
  %1786 = extractelement <4 x half> %1784, i64 2, !dbg !26
  %1787 = extractelement <4 x half> %1784, i64 1, !dbg !26
  %1788 = extractelement <4 x half> %1784, i64 0, !dbg !26
  %address.local1068 = or disjoint i64 %address.local1012, 32, !dbg !27
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1068, !dbg !27
  store half %1788, ptr addrspace(1) %address.local1069, align 2, !dbg !27
  %address.local1071 = or disjoint i64 %address.local1012, 33, !dbg !27
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1071, !dbg !27
  store half %1787, ptr addrspace(1) %address.local1072, align 2, !dbg !27
  %address.local1074 = or disjoint i64 %address.local1012, 34, !dbg !27
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1074, !dbg !27
  store half %1786, ptr addrspace(1) %address.local1075, align 2, !dbg !27
  %address.local1077 = or disjoint i64 %address.local1012, 35, !dbg !27
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1077, !dbg !27
  store half %1785, ptr addrspace(1) %address.local1078, align 2, !dbg !27
  %address.local1080 = or disjoint i64 %address.local1012, 48, !dbg !27
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1080, !dbg !27
  store half %1783, ptr addrspace(1) %address.local1081, align 2, !dbg !27
  %address.local1083 = or disjoint i64 %address.local1012, 49, !dbg !27
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1083, !dbg !27
  store half %1782, ptr addrspace(1) %address.local1084, align 2, !dbg !27
  %address.local1086 = or disjoint i64 %address.local1012, 50, !dbg !27
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1086, !dbg !27
  store half %1781, ptr addrspace(1) %address.local1087, align 2, !dbg !27
  %address.local1089 = or i64 %1767, 51, !dbg !27
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1089, !dbg !27
  store half %1780, ptr addrspace(1) %address.local1090, align 2, !dbg !27
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1068, !dbg !27
  store half %1778, ptr addrspace(1) %address.local1095, align 2, !dbg !27
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1071, !dbg !27
  store half %1777, ptr addrspace(1) %address.local1097, align 2, !dbg !27
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1074, !dbg !27
  store half %1776, ptr addrspace(1) %address.local1099, align 2, !dbg !27
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1077, !dbg !27
  store half %1775, ptr addrspace(1) %address.local1101, align 2, !dbg !27
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1080, !dbg !27
  store half %1773, ptr addrspace(1) %address.local1103, align 2, !dbg !27
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1083, !dbg !27
  store half %1772, ptr addrspace(1) %address.local1105, align 2, !dbg !27
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1086, !dbg !27
  store half %1771, ptr addrspace(1) %address.local1107, align 2, !dbg !27
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1089, !dbg !27
  store half %1770, ptr addrspace(1) %address.local1109, align 2, !dbg !27
  %normalized115 = fmul <4 x float> %row.reciprocal, %1647, !dbg !25
  %1789 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !26
  %1790 = extractelement <4 x half> %1789, i64 3, !dbg !26
  %1791 = extractelement <4 x half> %1789, i64 2, !dbg !26
  %1792 = extractelement <4 x half> %1789, i64 1, !dbg !26
  %1793 = extractelement <4 x half> %1789, i64 0, !dbg !26
  %normalized116 = fmul <4 x float> %row.reciprocal, %1522, !dbg !25
  %1794 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !26
  %1795 = extractelement <4 x half> %1794, i64 3, !dbg !26
  %1796 = extractelement <4 x half> %1794, i64 2, !dbg !26
  %1797 = extractelement <4 x half> %1794, i64 1, !dbg !26
  %1798 = extractelement <4 x half> %1794, i64 0, !dbg !26
  %normalized118 = fmul <4 x float> %row.reciprocal105, %1399, !dbg !25
  %1799 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !26
  %1800 = extractelement <4 x half> %1799, i64 3, !dbg !26
  %1801 = extractelement <4 x half> %1799, i64 2, !dbg !26
  %1802 = extractelement <4 x half> %1799, i64 1, !dbg !26
  %1803 = extractelement <4 x half> %1799, i64 0, !dbg !26
  %normalized119 = fmul <4 x float> %row.reciprocal105, %1274, !dbg !25
  %1804 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !26
  %1805 = extractelement <4 x half> %1804, i64 3, !dbg !26
  %1806 = extractelement <4 x half> %1804, i64 2, !dbg !26
  %1807 = extractelement <4 x half> %1804, i64 1, !dbg !26
  %1808 = extractelement <4 x half> %1804, i64 0, !dbg !26
  %address.local1124 = or disjoint i64 %address.local1012, 64, !dbg !27
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1124, !dbg !27
  store half %1808, ptr addrspace(1) %address.local1125, align 2, !dbg !27
  %address.local1127 = or disjoint i64 %address.local1012, 65, !dbg !27
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1127, !dbg !27
  store half %1807, ptr addrspace(1) %address.local1128, align 2, !dbg !27
  %address.local1130 = or disjoint i64 %address.local1012, 66, !dbg !27
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1130, !dbg !27
  store half %1806, ptr addrspace(1) %address.local1131, align 2, !dbg !27
  %address.local1133 = or disjoint i64 %address.local1012, 67, !dbg !27
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1133, !dbg !27
  store half %1805, ptr addrspace(1) %address.local1134, align 2, !dbg !27
  %address.local1136 = or disjoint i64 %address.local1012, 80, !dbg !27
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1136, !dbg !27
  store half %1803, ptr addrspace(1) %address.local1137, align 2, !dbg !27
  %address.local1139 = or disjoint i64 %address.local1012, 81, !dbg !27
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1139, !dbg !27
  store half %1802, ptr addrspace(1) %address.local1140, align 2, !dbg !27
  %address.local1142 = or disjoint i64 %address.local1012, 82, !dbg !27
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1142, !dbg !27
  store half %1801, ptr addrspace(1) %address.local1143, align 2, !dbg !27
  %address.local1145 = or disjoint i64 %address.local1012, 83, !dbg !27
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1145, !dbg !27
  store half %1800, ptr addrspace(1) %address.local1146, align 2, !dbg !27
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1124, !dbg !27
  store half %1798, ptr addrspace(1) %address.local1151, align 2, !dbg !27
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1127, !dbg !27
  store half %1797, ptr addrspace(1) %address.local1153, align 2, !dbg !27
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1130, !dbg !27
  store half %1796, ptr addrspace(1) %address.local1155, align 2, !dbg !27
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1133, !dbg !27
  store half %1795, ptr addrspace(1) %address.local1157, align 2, !dbg !27
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1136, !dbg !27
  store half %1793, ptr addrspace(1) %address.local1159, align 2, !dbg !27
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1139, !dbg !27
  store half %1792, ptr addrspace(1) %address.local1161, align 2, !dbg !27
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1142, !dbg !27
  store half %1791, ptr addrspace(1) %address.local1163, align 2, !dbg !27
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1145, !dbg !27
  store half %1790, ptr addrspace(1) %address.local1165, align 2, !dbg !27
  %normalized121 = fmul <4 x float> %row.reciprocal, %1614, !dbg !25
  %1809 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !26
  %1810 = extractelement <4 x half> %1809, i64 3, !dbg !26
  %1811 = extractelement <4 x half> %1809, i64 2, !dbg !26
  %1812 = extractelement <4 x half> %1809, i64 1, !dbg !26
  %1813 = extractelement <4 x half> %1809, i64 0, !dbg !26
  %normalized122 = fmul <4 x float> %row.reciprocal, %1488, !dbg !25
  %1814 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !26
  %1815 = extractelement <4 x half> %1814, i64 3, !dbg !26
  %1816 = extractelement <4 x half> %1814, i64 2, !dbg !26
  %1817 = extractelement <4 x half> %1814, i64 1, !dbg !26
  %1818 = extractelement <4 x half> %1814, i64 0, !dbg !26
  %normalized124 = fmul <4 x float> %row.reciprocal105, %1366, !dbg !25
  %1819 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !26
  %1820 = extractelement <4 x half> %1819, i64 3, !dbg !26
  %1821 = extractelement <4 x half> %1819, i64 2, !dbg !26
  %1822 = extractelement <4 x half> %1819, i64 1, !dbg !26
  %1823 = extractelement <4 x half> %1819, i64 0, !dbg !26
  %normalized125 = fmul <4 x float> %row.reciprocal105, %1240, !dbg !25
  %1824 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !26
  %1825 = extractelement <4 x half> %1824, i64 3, !dbg !26
  %1826 = extractelement <4 x half> %1824, i64 2, !dbg !26
  %1827 = extractelement <4 x half> %1824, i64 1, !dbg !26
  %1828 = extractelement <4 x half> %1824, i64 0, !dbg !26
  %address.local1180 = or disjoint i64 %address.local1012, 96, !dbg !27
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1180, !dbg !27
  store half %1828, ptr addrspace(1) %address.local1181, align 2, !dbg !27
  %address.local1183 = or disjoint i64 %address.local1012, 97, !dbg !27
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1183, !dbg !27
  store half %1827, ptr addrspace(1) %address.local1184, align 2, !dbg !27
  %address.local1186 = or disjoint i64 %address.local1012, 98, !dbg !27
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1186, !dbg !27
  store half %1826, ptr addrspace(1) %address.local1187, align 2, !dbg !27
  %address.local1189 = or disjoint i64 %address.local1012, 99, !dbg !27
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1189, !dbg !27
  store half %1825, ptr addrspace(1) %address.local1190, align 2, !dbg !27
  %address.local1192 = or disjoint i64 %address.local1012, 112, !dbg !27
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1192, !dbg !27
  store half %1823, ptr addrspace(1) %address.local1193, align 2, !dbg !27
  %address.local1195 = or disjoint i64 %address.local1012, 113, !dbg !27
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1195, !dbg !27
  store half %1822, ptr addrspace(1) %address.local1196, align 2, !dbg !27
  %address.local1198 = or disjoint i64 %address.local1012, 114, !dbg !27
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1198, !dbg !27
  store half %1821, ptr addrspace(1) %address.local1199, align 2, !dbg !27
  %address.local1201 = or i64 %1767, 115, !dbg !27
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1201, !dbg !27
  store half %1820, ptr addrspace(1) %address.local1202, align 2, !dbg !27
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1180, !dbg !27
  store half %1818, ptr addrspace(1) %address.local1207, align 2, !dbg !27
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1183, !dbg !27
  store half %1817, ptr addrspace(1) %address.local1209, align 2, !dbg !27
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1186, !dbg !27
  store half %1816, ptr addrspace(1) %address.local1211, align 2, !dbg !27
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1189, !dbg !27
  store half %1815, ptr addrspace(1) %address.local1213, align 2, !dbg !27
  %address.local1215 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1192, !dbg !27
  store half %1813, ptr addrspace(1) %address.local1215, align 2, !dbg !27
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1195, !dbg !27
  store half %1812, ptr addrspace(1) %address.local1217, align 2, !dbg !27
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1198, !dbg !27
  store half %1811, ptr addrspace(1) %address.local1219, align 2, !dbg !27
  %address.local1221 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1201, !dbg !27
  store half %1810, ptr addrspace(1) %address.local1221, align 2, !dbg !27
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
