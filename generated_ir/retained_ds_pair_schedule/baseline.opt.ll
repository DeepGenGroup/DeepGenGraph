; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/retained_ds_pair_schedule/baseline.ll'
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
  %address.local336 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local337 = and i64 %address.local336, 1536, !dbg !8
  %address.local338 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local337, !dbg !8
  %address.local339 = ptrtoint ptr addrspace(3) %address.local338 to i64, !dbg !8
  %address.local340 = trunc i64 %address.local339 to i32, !dbg !8
  %30 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !8
  %address.local341 = shl nuw nsw i64 %address.local, 8, !dbg !8
  %address.local342 = and i64 %address.local341, 32512, !dbg !8
  %31 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1 = shl nuw nsw i64 %28, 1, !dbg !8
  %32 = add i64 %.idx1, %.idx, !dbg !8
  %33 = lshr i64 %address.local, 3, !dbg !8
  %address.local346.idx = and i64 %33, 16, !dbg !8
  %34 = or disjoint i64 %32, %address.local346.idx, !dbg !8
  %35 = or disjoint i64 %34, %address.local342, !dbg !8
  %dma.buffer.offset1597 = trunc i64 %35 to i32, !dbg !8
  %36 = insertelement <2 x i64> poison, i64 %31, i64 0, !dbg !8
  %37 = bitcast <2 x i64> %36 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1598 = shufflevector <4 x i32> %37, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1599 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %30), !dbg !8
  %38 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1599, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %address.local347 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local348 = shl nuw nsw i64 %address.local347, 3, !dbg !8
  %address.local349 = and i64 %address.local348, 3584, !dbg !8
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local349, !dbg !8
  %address.local351 = ptrtoint ptr addrspace(3) %address.local350 to i64, !dbg !8
  %address.local352 = trunc i64 %address.local351 to i32, !dbg !8
  %39 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !8
  %40 = lshr i64 %address.local347, 3, !dbg !8
  %address.local358.idx = and i64 %40, 48, !dbg !8
  %41 = or disjoint i64 %32, %address.local358.idx, !dbg !8
  %42 = or disjoint i64 %41, %address.local342, !dbg !8
  %dma.buffer.offset1594 = trunc i64 %42 to i32, !dbg !8
  %dma.buffer.lds1596 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %39), !dbg !8
  %43 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1596, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %44 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %45 = sext i32 %44 to i64, !dbg !7
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local359 = lshr i32 %address.thread, 2, !dbg !10
  %address.local360 = and i32 %address.local359, 4, !dbg !10
  %address.local361 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local362 = and i32 %address.local361, 384, !dbg !10
  %address.local363 = or disjoint i32 %address.local360, %address.local362, !dbg !10
  %address.local364 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local365 = and i32 %address.local364, 120, !dbg !10
  %address.local366 = or disjoint i32 %address.local363, %address.local365, !dbg !10
  %address.local367 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local368 = and i32 %address.local367, 1024, !dbg !10
  %address.local369 = or disjoint i32 %address.local366, %address.local368, !dbg !10
  %address.local370 = zext nneg i32 %address.local369 to i64, !dbg !10
  %address.local371 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local370, !dbg !10
  %46 = load <4 x half>, ptr addrspace(3) %address.local371, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local371, i64 4096, !dbg !10
  %47 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %48 = fmul <4 x half> %46, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1393 = getelementptr i8, ptr addrspace(3) %address.local371, i64 1024, !dbg !10
  %49 = load <4 x half>, ptr addrspace(3) %lds.const.gep1393, align 8, !dbg !10
  %50 = fmul <4 x half> %47, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1394 = getelementptr i8, ptr addrspace(3) %address.local371, i64 5120, !dbg !10
  %51 = load <4 x half>, ptr addrspace(3) %lds.const.gep1394, align 8, !dbg !10
  %52 = fmul <4 x half> %49, splat (half 1.275630e-01), !dbg !10
  %53 = fmul <4 x half> %51, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %54 = or disjoint i64 %24, 127, !dbg !12
  %55 = add i64 %24, 128, !dbg !12
  %56 = icmp uge i64 %55, %54, !dbg !12
  %57 = sub nsw i64 32, %14, !dbg !13
  %58 = shl nuw nsw i32 %29, 2, !dbg !14
  %59 = and i32 %58, 12, !dbg !14
  %60 = and i32 %29, 1, !dbg !14
  %61 = or disjoint i32 %59, %60, !dbg !14
  %62 = trunc nuw nsw i32 %61 to i16, !dbg !14
  %.lhs.trunc = add nuw nsw i16 %62, 1, !dbg !14
  %63 = lshr i16 %.lhs.trunc, 1, !dbg !14
  %.sext = zext nneg i16 %63 to i64, !dbg !14
  %64 = shl i64 %16, 7, !dbg !15
  %65 = and i64 %64, 3968, !dbg !15
  %66 = shl nsw i64 %45, 12, !dbg !15
  %67 = or disjoint i64 %65, %66, !dbg !15
  %68 = icmp ult i64 %10, 1024, !dbg !11
  br i1 %68, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !11

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre107 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre109 = add nuw nsw i32 %.pre107, 64, !dbg !11
  %.pre111 = xor i32 %.pre107, 16, !dbg !11
  %.pre113 = and i32 %.pre109, 192, !dbg !11
  %.pre115 = xor i32 %.pre107, 32, !dbg !11
  br label %._crit_edge, !dbg !11

.lr.ph:                                           ; preds = %4
  %69 = zext nneg i32 %29 to i64, !dbg !8
  %70 = ptrtoint ptr addrspace(1) %1 to i64
  %71 = insertelement <2 x i64> poison, i64 %70, i64 0
  %72 = bitcast <2 x i64> %71 to <4 x i32>
  %dma.buffer.rsrc1592 = shufflevector <4 x i32> %72, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %73 = ptrtoint ptr addrspace(1) %2 to i64
  %74 = insertelement <2 x i64> poison, i64 %73, i64 0
  %75 = bitcast <2 x i64> %74 to <4 x i32>
  %dma.buffer.rsrc1550 = shufflevector <4 x i32> %75, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %76 = lshr i64 %69, 2
  %77 = and i64 %76, 48
  %78 = and i32 %29, 240
  %index.compact = zext nneg i32 %78 to i64
  %79 = sub nsw i64 %69, %index.compact
  %80 = add nsw i64 %79, %77
  %81 = add nsw i64 %80, %24
  %82 = and i64 %69, 48
  %83 = add i64 %81, 1
  %84 = lshr i64 %83, 32
  %compare.high = trunc nuw i64 %84 to i32
  %compare.low.base = trunc i64 %83 to i32
  %85 = add i64 %81, 65
  %86 = lshr i64 %85, 32
  %compare.high1261 = trunc nuw i64 %86 to i32
  %compare.low.base1265 = trunc i64 %85 to i32
  %87 = insertelement <32 x i32> poison, i32 %compare.low.base1265, i64 0, !dbg !16
  %88 = shufflevector <32 x i32> %87, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  %89 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %90 = shufflevector <32 x i32> %89, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  br label %91, !dbg !11

91:                                               ; preds = %.lr.ph, %524
  %92 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1178, %524 ]
  %93 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1174, %524 ]
  %94 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1170, %524 ]
  %95 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %867, %524 ]
  %96 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1166, %524 ]
  %97 = phi i64 [ 0, %.lr.ph ], [ %1179, %524 ]
  %98 = shl nuw nsw i64 %97, 7, !dbg !11
  %99 = lshr i64 %97, 12, !dbg !15
  %100 = add nuw nsw i64 %99, %16, !dbg !15
  %101 = lshr i64 %97, 5, !dbg !15
  %102 = and i64 %101, 127, !dbg !15
  %103 = shl i64 %100, 19, !dbg !15
  %104 = and i64 %103, 16252928, !dbg !15
  %105 = shl i64 %101, 19, !dbg !15
  %106 = shl nuw nsw i64 %97, 14, !dbg !15
  %107 = sub i64 %106, %105, !dbg !15
  %address.thread378 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !15, !range !9, !frisk.register_lifetimes !5
  %address.local379 = zext nneg i32 %address.thread378 to i64, !dbg !8
  %address.local380 = shl nuw nsw i64 %address.local379, 3, !dbg !8
  %address.local381 = and i64 %address.local380, 1536, !dbg !8
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !8
  %address.local385 = lshr i64 %address.local379, 4, !dbg !15
  %address.local386 = and i64 %address.local385, 8, !dbg !15
  %address.local387 = and i64 %address.local379, 123, !dbg !15
  %address.local388 = lshr i64 %address.local379, 2, !dbg !15
  %address.local389 = and i64 %address.local388, 4, !dbg !15
  %address.local390 = add nuw nsw i64 %address.local389, %address.local379, !dbg !15
  %address.local391 = and i64 %address.local390, 4, !dbg !15
  %address.local392 = or disjoint i64 %address.local391, %address.local387, !dbg !15
  %address.local393 = shl nuw nsw i64 %address.local392, 7, !dbg !15
  %address.local394 = or disjoint i64 %address.local393, %address.local386, !dbg !15
  %.idx58 = or disjoint i64 %104, %102, !dbg !15
  %108 = add i64 %.idx58, %107, !dbg !15
  %109 = add i64 %address.local394, %108, !dbg !15
  %.tr = trunc i64 %109 to i32, !dbg !15
  %dma.buffer.offset1591 = shl i32 %.tr, 1, !dbg !15
  %address.local396 = or disjoint i64 %address.local379, 256, !dbg !8
  %address.local397 = shl nuw nsw i64 %address.local396, 3, !dbg !8
  %address.local398 = and i64 %address.local397, 3584, !dbg !8
  %address.local399 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local398, !dbg !8
  %address.local402 = lshr i64 %address.local396, 4, !dbg !15
  %address.local403 = and i64 %address.local402, 24, !dbg !15
  %address.local411 = or disjoint i64 %address.local393, %address.local403, !dbg !15
  %110 = add i64 %address.local411, %108, !dbg !15
  %.tr13 = trunc i64 %110 to i32, !dbg !15
  %dma.buffer.offset1588 = shl i32 %.tr13, 1, !dbg !15
  %address.local417 = shl nuw nsw i64 %address.local379, 1, !dbg !14
  %address.local418 = and i64 %address.local417, 24, !dbg !14
  %address.local420 = shl i64 %address.local379, 62, !dbg !14
  %address.local421 = ashr exact i64 %address.local420, 57, !dbg !14
  %address.local422 = and i64 %address.local421, -64, !dbg !14
  %address.local425 = and i64 %address.local388, 12, !dbg !10
  %111 = and i64 %address.local379, 32, !dbg !10
  %.not14 = icmp eq i64 %111, 0, !dbg !10
  %address.local429 = select i1 %.not14, i64 0, i64 1016, !dbg !10
  %.idx15 = mul nuw nsw i64 %address.local379, 320, !dbg !14
  %112 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx15, !dbg !14
  %.idx16 = mul nsw i64 %address.local388, -1280, !dbg !14
  %113 = getelementptr i8, ptr addrspace(3) %112, i64 %.idx16, !dbg !14
  %114 = getelementptr [2 x i8], ptr addrspace(3) %113, i64 %address.local418, !dbg !14
  %115 = getelementptr [2 x i8], ptr addrspace(3) %114, i64 %address.local422, !dbg !14
  %116 = getelementptr [2 x i8], ptr addrspace(3) %115, i64 %address.local425, !dbg !14
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) %116, i64 %address.local429, !dbg !14
  %lds.const.gep1395 = getelementptr i8, ptr addrspace(3) %address.local431, i64 4096, !dbg !14
  %address.local442 = shl nuw nsw i32 %address.thread378, 1, !dbg !14
  %address.local443 = and i32 %address.local442, 24, !dbg !14
  %address.local444 = shl nuw nsw i32 %address.thread378, 8, !dbg !14
  %address.local445 = and i32 %address.local444, 768, !dbg !14
  %address.local446 = or disjoint i32 %address.local443, %address.local445, !dbg !14
  %address.local447 = shl nuw nsw i32 %address.thread378, 5, !dbg !14
  %address.local448 = and i32 %address.local447, 96, !dbg !14
  %address.local449 = or disjoint i32 %address.local446, %address.local448, !dbg !14
  %address.local450 = zext nneg i32 %address.local449 to i64, !dbg !14
  %address.local451 = lshr i32 %address.thread378, 1, !dbg !14
  %address.local452 = and i32 %address.local451, 1, !dbg !14
  %address.local453 = zext nneg i32 %address.local452 to i64, !dbg !14
  %lds.const.gep1397 = getelementptr i8, ptr addrspace(3) %address.local431, i64 128, !dbg !14
  %lds.const.gep1398 = getelementptr i8, ptr addrspace(3) %address.local431, i64 4224, !dbg !14
  %lds.const.gep1401 = getelementptr i8, ptr addrspace(3) %address.local431, i64 1024, !dbg !14
  %lds.const.gep1402 = getelementptr i8, ptr addrspace(3) %address.local431, i64 5120, !dbg !14
  %lds.const.gep1405 = getelementptr i8, ptr addrspace(3) %address.local431, i64 1152, !dbg !14
  %lds.const.gep1406 = getelementptr i8, ptr addrspace(3) %address.local431, i64 5248, !dbg !14
  %address.local769 = shl nuw nsw i64 %address.local379, 7, !dbg !8
  %address.local770 = and i64 %address.local769, 16256, !dbg !8
  %address.local773 = or disjoint i64 %address.local770, %address.local386, !dbg !8
  %address.local774.idx = shl nuw nsw i64 %address.local773, 1, !dbg !8
  %117 = or disjoint i64 %address.local774.idx, %32, !dbg !8
  %118 = trunc i64 %117 to i32, !dbg !8
  %dma.buffer.offset1585 = or disjoint i32 %118, 64, !dbg !8
  %address.local779 = or disjoint i64 %address.local403, %address.local770, !dbg !8
  %address.local780.idx = shl nuw nsw i64 %address.local779, 1, !dbg !8
  %119 = or disjoint i64 %32, %address.local780.idx, !dbg !8
  %120 = trunc i64 %119 to i32, !dbg !8
  %dma.buffer.offset1582 = or disjoint i32 %120, 64, !dbg !8
  %121 = add nuw nsw i64 %97, 1024, !dbg !15
  %122 = lshr i64 %121, 12, !dbg !15
  %123 = add nuw nsw i64 %122, %16, !dbg !15
  %124 = add nsw i64 %101, %67, !dbg !15
  %125 = shl i64 %123, 19, !dbg !15
  %126 = and i64 %125, 16252928, !dbg !15
  %address.local781 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local381, !dbg !15
  %127 = add i64 %107, %101, !dbg !15
  %.idx1923 = add i64 %127, %67, !dbg !15
  %128 = add i64 %.idx1923, %126, !dbg !15
  %address.local785 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local398, !dbg !15
  %129 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local382, i64 0, !dbg !8
  %130 = insertelement <4 x ptr addrspace(3)> %129, ptr addrspace(3) %address.local399, i64 1, !dbg !8
  %131 = insertelement <4 x ptr addrspace(3)> %130, ptr addrspace(3) %address.local781, i64 2, !dbg !8
  %132 = insertelement <4 x ptr addrspace(3)> %131, ptr addrspace(3) %address.local785, i64 3, !dbg !8
  %133 = ptrtoint <4 x ptr addrspace(3)> %132 to <4 x i64>, !dbg !8
  %134 = trunc <4 x i64> %133 to <4 x i32>, !dbg !8
  %135 = extractelement <4 x i32> %134, i64 0, !dbg !15
  %136 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %135), !dbg !15
  %dma.buffer.lds1593 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %136), !dbg !15
  %137 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %138 = extractelement <4 x i32> %134, i64 1, !dbg !15
  %139 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138), !dbg !15
  %dma.buffer.lds1590 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %139), !dbg !15
  %140 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %141 = load <4 x half>, ptr addrspace(3) %address.local431, align 8, !dbg !14
  %142 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %141, <4 x float> zeroinitializer) #8, !dbg !14
  %143 = load <4 x half>, ptr addrspace(3) %lds.const.gep1395, align 8, !dbg !14
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %.sext) #8, !dbg !14, !frisk.register_lifetimes !5
  %address.local454 = add nsw i64 %address.base, %address.local453, !dbg !14
  %address.local455 = shl i64 %address.local454, 6, !dbg !14
  %address.local456 = sub i64 %address.local450, %address.local455, !dbg !14
  %144 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local456, !dbg !14
  %145 = getelementptr [2 x i8], ptr addrspace(3) %144, i64 %address.local425, !dbg !14
  %146 = getelementptr [2 x i8], ptr addrspace(3) %145, i64 %address.local429, !dbg !14
  %address.local460 = getelementptr i8, ptr addrspace(3) %146, i64 64, !dbg !14
  %147 = load <4 x half>, ptr addrspace(3) %address.local460, align 8, !dbg !14
  %148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %143, <4 x float> %142) #8, !dbg !14
  %149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %147, <4 x float> zeroinitializer) #8, !dbg !14
  %lds.const.gep1396 = getelementptr i8, ptr addrspace(3) %146, i64 4160, !dbg !14
  %150 = load <4 x half>, ptr addrspace(3) %lds.const.gep1396, align 8, !dbg !14
  %151 = load <4 x half>, ptr addrspace(3) %lds.const.gep1397, align 8, !dbg !14
  %152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %150, <4 x float> %149) #8, !dbg !14
  %153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %151, <4 x float> zeroinitializer) #8, !dbg !14
  %154 = load <4 x half>, ptr addrspace(3) %lds.const.gep1398, align 8, !dbg !14
  %lds.const.gep1399 = getelementptr i8, ptr addrspace(3) %146, i64 192, !dbg !14
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep1399, align 8, !dbg !14
  %156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %154, <4 x float> %153) #8, !dbg !14
  %157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %155, <4 x float> zeroinitializer) #8, !dbg !14
  %lds.const.gep1400 = getelementptr i8, ptr addrspace(3) %146, i64 4288, !dbg !14
  %158 = load <4 x half>, ptr addrspace(3) %lds.const.gep1400, align 8, !dbg !14
  %159 = load <4 x half>, ptr addrspace(3) %lds.const.gep1401, align 8, !dbg !14
  %160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %158, <4 x float> %157) #8, !dbg !14
  %161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %159, <4 x float> zeroinitializer) #8, !dbg !14
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep1402, align 8, !dbg !14
  %lds.const.gep1403 = getelementptr i8, ptr addrspace(3) %146, i64 1088, !dbg !14
  %163 = load <4 x half>, ptr addrspace(3) %lds.const.gep1403, align 8, !dbg !14
  %164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %162, <4 x float> %161) #8, !dbg !14
  %165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %163, <4 x float> zeroinitializer) #8, !dbg !14
  %lds.const.gep1404 = getelementptr i8, ptr addrspace(3) %146, i64 5184, !dbg !14
  %166 = load <4 x half>, ptr addrspace(3) %lds.const.gep1404, align 8, !dbg !14
  %167 = load <4 x half>, ptr addrspace(3) %lds.const.gep1405, align 8, !dbg !14
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %166, <4 x float> %165) #8, !dbg !14
  %169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %167, <4 x float> zeroinitializer) #8, !dbg !14
  %170 = load <4 x half>, ptr addrspace(3) %lds.const.gep1406, align 8, !dbg !14
  %lds.const.gep1407 = getelementptr i8, ptr addrspace(3) %146, i64 1216, !dbg !14
  %171 = load <4 x half>, ptr addrspace(3) %lds.const.gep1407, align 8, !dbg !14
  %172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %170, <4 x float> %169) #8, !dbg !14
  %173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %171, <4 x float> zeroinitializer) #8, !dbg !14
  %lds.const.gep1408 = getelementptr i8, ptr addrspace(3) %146, i64 5312, !dbg !14
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1408, align 8, !dbg !14
  %175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %174, <4 x float> %173) #8, !dbg !14
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %141, <4 x float> zeroinitializer) #8, !dbg !14
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %143, <4 x float> %176) #8, !dbg !14
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %147, <4 x float> zeroinitializer) #8, !dbg !14
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %150, <4 x float> %178) #8, !dbg !14
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %151, <4 x float> zeroinitializer) #8, !dbg !14
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %154, <4 x float> %180) #8, !dbg !14
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %155, <4 x float> zeroinitializer) #8, !dbg !14
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %158, <4 x float> %182) #8, !dbg !14
  %184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %159, <4 x float> zeroinitializer) #8, !dbg !14
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %162, <4 x float> %184) #8, !dbg !14
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %163, <4 x float> zeroinitializer) #8, !dbg !14
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %166, <4 x float> %186) #8, !dbg !14
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %167, <4 x float> zeroinitializer) #8, !dbg !14
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %170, <4 x float> %188) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %171, <4 x float> zeroinitializer) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %174, <4 x float> %190) #8, !dbg !14
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %192 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %193 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %194 = extractelement <4 x i32> %134, i64 2, !dbg !15
  %195 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %194), !dbg !15
  %dma.buffer.lds1581 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %195), !dbg !15
  %196 = extractelement <4 x i32> %134, i64 3, !dbg !15
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8192, !dbg !14
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12288, !dbg !14
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %146, i64 8256, !dbg !14
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %146, i64 12352, !dbg !14
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local431, i64 8320, !dbg !14
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local431, i64 12416, !dbg !14
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %146, i64 8384, !dbg !14
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %146, i64 12480, !dbg !14
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9216, !dbg !14
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13312, !dbg !14
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %146, i64 9280, !dbg !14
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %146, i64 13376, !dbg !14
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local431, i64 9344, !dbg !14
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local431, i64 13440, !dbg !14
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %146, i64 9408, !dbg !14
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %146, i64 13504, !dbg !14
  %dma.buffer.offset1573 = or disjoint i32 %118, 128, !dbg !8
  %dma.buffer.offset1570 = or disjoint i32 %120, 128, !dbg !8
  %197 = add nuw nsw i64 %97, 2048, !dbg !15
  %198 = lshr i64 %197, 12, !dbg !15
  %199 = add nuw nsw i64 %198, %16, !dbg !15
  %200 = insertelement <2 x i64> poison, i64 %124, i64 0, !dbg !15
  %201 = shufflevector <2 x i64> %200, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !15
  %202 = icmp slt <2 x i64> %201, <i64 -32, i64 -64>, !dbg !15
  %203 = extractelement <2 x i1> %202, i64 0, !dbg !15
  %204 = sext i1 %203 to i64, !dbg !15
  %205 = add nsw <2 x i64> %201, <i64 32, i64 64>, !dbg !15
  %206 = sub nsw <2 x i64> <i64 -33, i64 -65>, %201, !dbg !15
  %207 = select <2 x i1> %202, <2 x i64> %206, <2 x i64> %205, !dbg !15
  %208 = sdiv <2 x i64> %207, splat (i64 128), !dbg !15
  %209 = extractelement <2 x i64> %208, i64 0, !dbg !15
  %210 = xor i64 %209, %204, !dbg !15
  %211 = shl nsw i64 %210, 7, !dbg !15
  %212 = sub i64 %128, %211, !dbg !15
  %213 = add i64 %address.local394, %212, !dbg !15
  %.tr24 = trunc i64 %213 to i32, !dbg !15
  %214 = shl i32 %.tr24, 1, !dbg !15
  %dma.buffer.offset1579 = add i32 %214, 64, !dbg !15
  %215 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %216 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %196), !dbg !15
  %217 = add i64 %address.local411, %212, !dbg !15
  %.tr30 = trunc i64 %217 to i32, !dbg !15
  %218 = shl i32 %.tr30, 1, !dbg !15
  %dma.buffer.offset1576 = add i32 %218, 64, !dbg !15
  %dma.buffer.lds1578 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %216), !dbg !15
  %219 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %220 = load <4 x half>, ptr addrspace(3) %address.local371, align 8, !dbg !10
  %221 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1393, align 8, !dbg !10
  %223 = fmul <4 x half> %220, splat (half 1.275630e-01), !dbg !10
  %224 = load <4 x half>, ptr addrspace(3) %lds.const.gep1394, align 8, !dbg !10
  %225 = fmul <4 x half> %221, splat (half 1.275630e-01), !dbg !10
  %226 = fmul <4 x half> %222, splat (half 1.275630e-01), !dbg !10
  %227 = fmul <4 x half> %224, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %228 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !14
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %228, <4 x float> %148) #8, !dbg !14
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !14
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !14
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %230, <4 x float> %229) #8, !dbg !14
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %231, <4 x float> %152) #8, !dbg !14
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !14
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !14
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %234, <4 x float> %233) #8, !dbg !14
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %235, <4 x float> %156) #8, !dbg !14
  %238 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !14
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !14
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %238, <4 x float> %237) #8, !dbg !14
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %239, <4 x float> %160) #8, !dbg !14
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !14
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !14
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %242, <4 x float> %241) #8, !dbg !14
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %243, <4 x float> %164) #8, !dbg !14
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !14
  %247 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !14
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %246, <4 x float> %245) #8, !dbg !14
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %247, <4 x float> %168) #8, !dbg !14
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !14
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !14
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %250, <4 x float> %249) #8, !dbg !14
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %251, <4 x float> %172) #8, !dbg !14
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !14
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !14
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %254, <4 x float> %253) #8, !dbg !14
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %255, <4 x float> %175) #8, !dbg !14
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !14
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %258, <4 x float> %257) #8, !dbg !14
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %228, <4 x float> %177) #8, !dbg !14
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %230, <4 x float> %260) #8, !dbg !14
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %231, <4 x float> %179) #8, !dbg !14
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %234, <4 x float> %262) #8, !dbg !14
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %235, <4 x float> %181) #8, !dbg !14
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %238, <4 x float> %264) #8, !dbg !14
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %239, <4 x float> %183) #8, !dbg !14
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %242, <4 x float> %266) #8, !dbg !14
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %243, <4 x float> %185) #8, !dbg !14
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %246, <4 x float> %268) #8, !dbg !14
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %247, <4 x float> %187) #8, !dbg !14
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %250, <4 x float> %270) #8, !dbg !14
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %251, <4 x float> %189) #8, !dbg !14
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %254, <4 x float> %272) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %255, <4 x float> %191) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %258, <4 x float> %274) #8, !dbg !14
  %276 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %277 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %278 = extractelement <2 x i1> %202, i64 1, !dbg !15
  %279 = sext i1 %278 to i64, !dbg !15
  %280 = extractelement <2 x i64> %208, i64 1, !dbg !15
  %281 = xor i64 %280, %279, !dbg !15
  %282 = shl i64 %199, 19, !dbg !15
  %283 = and i64 %282, 16252928, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %284 = add i64 %.idx1923, %283, !dbg !15
  %285 = shl nsw i64 %281, 7, !dbg !15
  %286 = sub i64 %284, %285, !dbg !15
  %287 = add i64 %address.local394, %286, !dbg !15
  %.tr38 = trunc i64 %287 to i32, !dbg !15
  %288 = shl i32 %.tr38, 1, !dbg !15
  %dma.buffer.offset1567 = add i32 %288, 128, !dbg !15
  %289 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %290 = add i64 %address.local411, %286, !dbg !15
  %.tr44 = trunc i64 %290 to i32, !dbg !15
  %291 = shl i32 %.tr44, 1, !dbg !15
  %dma.buffer.offset1564 = add i32 %291, 128, !dbg !15
  %292 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %293 = load <4 x half>, ptr addrspace(3) %address.local371, align 8, !dbg !10
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1393, align 8, !dbg !10
  %296 = fmul <4 x half> %293, splat (half 1.275630e-01), !dbg !10
  %297 = load <4 x half>, ptr addrspace(3) %lds.const.gep1394, align 8, !dbg !10
  %298 = fmul <4 x half> %294, splat (half 1.275630e-01), !dbg !10
  %299 = fmul <4 x half> %295, splat (half 1.275630e-01), !dbg !10
  %300 = fmul <4 x half> %297, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !14
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %301, <4 x float> %232) #8, !dbg !14
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !14
  %304 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !14
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %303, <4 x float> %302) #8, !dbg !14
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %304, <4 x float> %236) #8, !dbg !14
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !14
  %308 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !14
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %307, <4 x float> %306) #8, !dbg !14
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %308, <4 x float> %240) #8, !dbg !14
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !14
  %312 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !14
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %311, <4 x float> %310) #8, !dbg !14
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %312, <4 x float> %244) #8, !dbg !14
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !14
  %316 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !14
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %315, <4 x float> %314) #8, !dbg !14
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %316, <4 x float> %248) #8, !dbg !14
  %319 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !14
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !14
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %319, <4 x float> %318) #8, !dbg !14
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %320, <4 x float> %252) #8, !dbg !14
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !14
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !14
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %323, <4 x float> %322) #8, !dbg !14
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %324, <4 x float> %256) #8, !dbg !14
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !14
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !14
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %327, <4 x float> %326) #8, !dbg !14
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %328, <4 x float> %259) #8, !dbg !14
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !14
  %332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %331, <4 x float> %330) #8, !dbg !14
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %301, <4 x float> %261) #8, !dbg !14
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %303, <4 x float> %333) #8, !dbg !14
  %335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %304, <4 x float> %263) #8, !dbg !14
  %336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %307, <4 x float> %335) #8, !dbg !14
  %337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %308, <4 x float> %265) #8, !dbg !14
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %311, <4 x float> %337) #8, !dbg !14
  %339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %312, <4 x float> %267) #8, !dbg !14
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %315, <4 x float> %339) #8, !dbg !14
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %316, <4 x float> %269) #8, !dbg !14
  %342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %319, <4 x float> %341) #8, !dbg !14
  %343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %320, <4 x float> %271) #8, !dbg !14
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %323, <4 x float> %343) #8, !dbg !14
  %345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %324, <4 x float> %273) #8, !dbg !14
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %327, <4 x float> %345) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %328, <4 x float> %275) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %331, <4 x float> %347) #8, !dbg !14
  %dma.buffer.offset1561 = or disjoint i32 %118, 192, !dbg !8
  %349 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %dma.buffer.offset1558 = or disjoint i32 %120, 192, !dbg !8
  %350 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1598) #9, !dbg !8
  %351 = add nuw nsw i64 %97, 3072, !dbg !15
  %352 = lshr i64 %351, 12, !dbg !15
  %353 = add nuw nsw i64 %352, %16, !dbg !15
  %354 = add nsw i64 %124, 96, !dbg !15
  %355 = icmp slt i64 %124, -96, !dbg !15
  %356 = sub nsw i64 -97, %124, !dbg !15
  %357 = select i1 %355, i64 %356, i64 %354, !dbg !15
  %358 = sdiv i64 %357, 128, !dbg !15
  %359 = sext i1 %355 to i64, !dbg !15
  %360 = xor i64 %358, %359, !dbg !15
  %361 = shl i64 %353, 19, !dbg !15
  %362 = and i64 %361, 16252928, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %363 = add i64 %.idx1923, %362, !dbg !15
  %364 = shl nsw i64 %360, 7, !dbg !15
  %365 = sub i64 %363, %364, !dbg !15
  %366 = add i64 %address.local394, %365, !dbg !15
  %.tr52 = trunc i64 %366 to i32, !dbg !15
  %367 = shl i32 %.tr52, 1, !dbg !15
  %dma.buffer.offset1555 = add i32 %367, 192, !dbg !15
  %368 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %369 = add i64 %address.local411, %365, !dbg !15
  %.tr58 = trunc i64 %369 to i32, !dbg !15
  %370 = shl i32 %.tr58, 1, !dbg !15
  %dma.buffer.offset1552 = add i32 %370, 192, !dbg !15
  %371 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1592) #9, !dbg !15
  %372 = load <4 x half>, ptr addrspace(3) %address.local371, align 8, !dbg !10
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep1393, align 8, !dbg !10
  %375 = fmul <4 x half> %372, splat (half 1.275630e-01), !dbg !10
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1394, align 8, !dbg !10
  %377 = fmul <4 x half> %373, splat (half 1.275630e-01), !dbg !10
  %378 = fmul <4 x half> %374, splat (half 1.275630e-01), !dbg !10
  %379 = fmul <4 x half> %376, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %380 = add nuw nsw i64 %101, %16, !dbg !17
  %381 = shl i64 %380, 19, !dbg !17
  %382 = and i64 %381, 16252928, !dbg !17
  %address.local832 = shl nuw nsw i64 %address.local379, 6, !dbg !17
  %address.local833 = and i64 %address.local832, 16128, !dbg !17
  %address.local835 = and i64 %address.local380, 24, !dbg !17
  %.idx5962 = add i64 %382, %107, !dbg !17
  %383 = or disjoint i64 %address.local835, %.idx5962, !dbg !17
  %384 = shl i64 %383, 1, !dbg !17
  %385 = or disjoint i64 %384, %address.local833, !dbg !17
  %dma.buffer.offset1549 = trunc i64 %385 to i32, !dbg !17
  %386 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %address.local838 = shl nuw nsw i64 %address.local396, 6, !dbg !17
  %address.local839 = and i64 %address.local838, 32512, !dbg !17
  %387 = or disjoint i64 %384, %address.local839, !dbg !17
  %dma.buffer.offset1546 = trunc i64 %387 to i32, !dbg !17
  %388 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %389 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !14
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %375, <4 x half> %389, <4 x float> %305) #8, !dbg !14
  %391 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !14
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !14
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %391, <4 x float> %390) #8, !dbg !14
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %375, <4 x half> %392, <4 x float> %309) #8, !dbg !14
  %395 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !14
  %396 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !14
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %395, <4 x float> %394) #8, !dbg !14
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %375, <4 x half> %396, <4 x float> %313) #8, !dbg !14
  %399 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !14
  %400 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !14
  %401 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %399, <4 x float> %398) #8, !dbg !14
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %375, <4 x half> %400, <4 x float> %317) #8, !dbg !14
  %403 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !14
  %404 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !14
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %403, <4 x float> %402) #8, !dbg !14
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %375, <4 x half> %404, <4 x float> %321) #8, !dbg !14
  %407 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !14
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !14
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %407, <4 x float> %406) #8, !dbg !14
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %375, <4 x half> %408, <4 x float> %325) #8, !dbg !14
  %411 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !14
  %412 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !14
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %411, <4 x float> %410) #8, !dbg !14
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %375, <4 x half> %412, <4 x float> %329) #8, !dbg !14
  %415 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !14
  %416 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !14
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %415, <4 x float> %414) #8, !dbg !14
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %375, <4 x half> %416, <4 x float> %332) #8, !dbg !14
  %419 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !14
  %420 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %419, <4 x float> %418) #8, !dbg !14
  %421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %378, <4 x half> %389, <4 x float> %334) #8, !dbg !14
  %422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %379, <4 x half> %391, <4 x float> %421) #8, !dbg !14
  %423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %378, <4 x half> %392, <4 x float> %336) #8, !dbg !14
  %424 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %379, <4 x half> %395, <4 x float> %423) #8, !dbg !14
  %425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %378, <4 x half> %396, <4 x float> %338) #8, !dbg !14
  %426 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %379, <4 x half> %399, <4 x float> %425) #8, !dbg !14
  %427 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %378, <4 x half> %400, <4 x float> %340) #8, !dbg !14
  %428 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %379, <4 x half> %403, <4 x float> %427) #8, !dbg !14
  %429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %378, <4 x half> %404, <4 x float> %342) #8, !dbg !14
  %430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %379, <4 x half> %407, <4 x float> %429) #8, !dbg !14
  %431 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %378, <4 x half> %408, <4 x float> %344) #8, !dbg !14
  %432 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %379, <4 x half> %411, <4 x float> %431) #8, !dbg !14
  %433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %378, <4 x half> %412, <4 x float> %346) #8, !dbg !14
  %434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %379, <4 x half> %415, <4 x float> %433) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14
  %435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %378, <4 x half> %416, <4 x float> %348) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %436 = or disjoint i64 %98, 127, !dbg !12
  %437 = icmp uge i64 %24, %436, !dbg !12
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %379, <4 x half> %419, <4 x float> %435) #8, !dbg !14
  %439 = and i1 %56, %437, !dbg !12
  br i1 %439, label %524, label %440, !dbg !12

440:                                              ; preds = %91
  %441 = or disjoint i64 %98, %82, !dbg !12
  %442 = lshr i64 %97, 25, !dbg !16
  %compare.high1137 = trunc nuw i64 %442 to i32, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1137, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1137, !dbg !16
  %compare.low.base1138 = trunc i64 %441 to i32, !dbg !16
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 1) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1141 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 2) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1145 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 3) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1149 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 4) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1153 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 5) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1157 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 6) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1161 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 7) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1165 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 8) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1169 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 9) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1173 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 10) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1177 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 11) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1181 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 12) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1185 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 13) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1189 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 14) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1193 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 15) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1197 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 16) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1201 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 17) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1205 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 18) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1209 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 19) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1213 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 20) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1217 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 21) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1221 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 22) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1225 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 23) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1229 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 24) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1233 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 25) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1237 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 26) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1241 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 27) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1245 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 28) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1249 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 29) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1253 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 30) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1257 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 31) #8, !dbg !16, !frisk.register_lifetimes !5
  %443 = insertelement <32 x i32> poison, i32 %compare.low.base1138, i64 0, !dbg !16
  %444 = insertelement <32 x i32> %443, i32 %compare.local.base, i64 1, !dbg !16
  %445 = insertelement <32 x i32> %444, i32 %compare.local.base1141, i64 2, !dbg !16
  %446 = insertelement <32 x i32> %445, i32 %compare.local.base1145, i64 3, !dbg !16
  %447 = insertelement <32 x i32> %446, i32 %compare.local.base1149, i64 4, !dbg !16
  %448 = insertelement <32 x i32> %447, i32 %compare.local.base1153, i64 5, !dbg !16
  %449 = insertelement <32 x i32> %448, i32 %compare.local.base1157, i64 6, !dbg !16
  %450 = insertelement <32 x i32> %449, i32 %compare.local.base1161, i64 7, !dbg !16
  %451 = insertelement <32 x i32> %450, i32 %compare.local.base1165, i64 8, !dbg !16
  %452 = insertelement <32 x i32> %451, i32 %compare.local.base1169, i64 9, !dbg !16
  %453 = insertelement <32 x i32> %452, i32 %compare.local.base1173, i64 10, !dbg !16
  %454 = insertelement <32 x i32> %453, i32 %compare.local.base1177, i64 11, !dbg !16
  %455 = insertelement <32 x i32> %454, i32 %compare.local.base1181, i64 12, !dbg !16
  %456 = insertelement <32 x i32> %455, i32 %compare.local.base1185, i64 13, !dbg !16
  %457 = insertelement <32 x i32> %456, i32 %compare.local.base1189, i64 14, !dbg !16
  %458 = insertelement <32 x i32> %457, i32 %compare.local.base1193, i64 15, !dbg !16
  %459 = insertelement <32 x i32> %458, i32 %compare.local.base1197, i64 16, !dbg !16
  %460 = insertelement <32 x i32> %459, i32 %compare.local.base1201, i64 17, !dbg !16
  %461 = insertelement <32 x i32> %460, i32 %compare.local.base1205, i64 18, !dbg !16
  %462 = insertelement <32 x i32> %461, i32 %compare.local.base1209, i64 19, !dbg !16
  %463 = insertelement <32 x i32> %462, i32 %compare.local.base1213, i64 20, !dbg !16
  %464 = insertelement <32 x i32> %463, i32 %compare.local.base1217, i64 21, !dbg !16
  %465 = insertelement <32 x i32> %464, i32 %compare.local.base1221, i64 22, !dbg !16
  %466 = insertelement <32 x i32> %465, i32 %compare.local.base1225, i64 23, !dbg !16
  %467 = insertelement <32 x i32> %466, i32 %compare.local.base1229, i64 24, !dbg !16
  %468 = insertelement <32 x i32> %467, i32 %compare.local.base1233, i64 25, !dbg !16
  %469 = insertelement <32 x i32> %468, i32 %compare.local.base1237, i64 26, !dbg !16
  %470 = insertelement <32 x i32> %469, i32 %compare.local.base1241, i64 27, !dbg !16
  %471 = insertelement <32 x i32> %470, i32 %compare.local.base1245, i64 28, !dbg !16
  %472 = insertelement <32 x i32> %471, i32 %compare.local.base1249, i64 29, !dbg !16
  %473 = insertelement <32 x i32> %472, i32 %compare.local.base1253, i64 30, !dbg !16
  %474 = insertelement <32 x i32> %473, i32 %compare.local.base1257, i64 31, !dbg !16
  %475 = or <32 x i32> %474, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %476 = icmp ugt <32 x i32> %90, %475, !dbg !16
  %477 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %478 = shufflevector <32 x i1> %477, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %479 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %480 = shufflevector <32 x i1> %479, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %481 = select <32 x i1> %478, <32 x i1> %476, <32 x i1> %480, !dbg !16
  %482 = select <32 x i1> %481, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %compare.high.equal1263 = icmp eq i32 %compare.high1261, %compare.high1137, !dbg !16
  %compare.high.order1264 = icmp ugt i32 %compare.high1261, %compare.high1137, !dbg !16
  %compare.local.base1269 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 32) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1273 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 33) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1277 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 34) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1281 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 35) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1285 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 36) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1289 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 37) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1293 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 38) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1297 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 39) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1301 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 40) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1305 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 41) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1309 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 42) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1313 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 43) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1317 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 44) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1321 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 45) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1325 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 46) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1329 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 47) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1333 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 48) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1337 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 49) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1341 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 50) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1345 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 51) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1349 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 52) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1353 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 53) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1357 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 54) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1361 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 55) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1365 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 56) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1369 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 57) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1373 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 58) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1377 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 59) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1381 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 60) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1385 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 61) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1389 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1138, i32 62) #8, !dbg !16, !frisk.register_lifetimes !5
  %483 = insertelement <32 x i32> %443, i32 %compare.local.base1269, i64 1, !dbg !16
  %484 = insertelement <32 x i32> %483, i32 %compare.local.base1273, i64 2, !dbg !16
  %485 = insertelement <32 x i32> %484, i32 %compare.local.base1277, i64 3, !dbg !16
  %486 = insertelement <32 x i32> %485, i32 %compare.local.base1281, i64 4, !dbg !16
  %487 = insertelement <32 x i32> %486, i32 %compare.local.base1285, i64 5, !dbg !16
  %488 = insertelement <32 x i32> %487, i32 %compare.local.base1289, i64 6, !dbg !16
  %489 = insertelement <32 x i32> %488, i32 %compare.local.base1293, i64 7, !dbg !16
  %490 = insertelement <32 x i32> %489, i32 %compare.local.base1297, i64 8, !dbg !16
  %491 = insertelement <32 x i32> %490, i32 %compare.local.base1301, i64 9, !dbg !16
  %492 = insertelement <32 x i32> %491, i32 %compare.local.base1305, i64 10, !dbg !16
  %493 = insertelement <32 x i32> %492, i32 %compare.local.base1309, i64 11, !dbg !16
  %494 = insertelement <32 x i32> %493, i32 %compare.local.base1313, i64 12, !dbg !16
  %495 = insertelement <32 x i32> %494, i32 %compare.local.base1317, i64 13, !dbg !16
  %496 = insertelement <32 x i32> %495, i32 %compare.local.base1321, i64 14, !dbg !16
  %497 = insertelement <32 x i32> %496, i32 %compare.local.base1325, i64 15, !dbg !16
  %498 = insertelement <32 x i32> %497, i32 %compare.local.base1329, i64 16, !dbg !16
  %499 = insertelement <32 x i32> %498, i32 %compare.local.base1333, i64 17, !dbg !16
  %500 = insertelement <32 x i32> %499, i32 %compare.local.base1337, i64 18, !dbg !16
  %501 = insertelement <32 x i32> %500, i32 %compare.local.base1341, i64 19, !dbg !16
  %502 = insertelement <32 x i32> %501, i32 %compare.local.base1345, i64 20, !dbg !16
  %503 = insertelement <32 x i32> %502, i32 %compare.local.base1349, i64 21, !dbg !16
  %504 = insertelement <32 x i32> %503, i32 %compare.local.base1353, i64 22, !dbg !16
  %505 = insertelement <32 x i32> %504, i32 %compare.local.base1357, i64 23, !dbg !16
  %506 = insertelement <32 x i32> %505, i32 %compare.local.base1361, i64 24, !dbg !16
  %507 = insertelement <32 x i32> %506, i32 %compare.local.base1365, i64 25, !dbg !16
  %508 = insertelement <32 x i32> %507, i32 %compare.local.base1369, i64 26, !dbg !16
  %509 = insertelement <32 x i32> %508, i32 %compare.local.base1373, i64 27, !dbg !16
  %510 = insertelement <32 x i32> %509, i32 %compare.local.base1377, i64 28, !dbg !16
  %511 = insertelement <32 x i32> %510, i32 %compare.local.base1381, i64 29, !dbg !16
  %512 = insertelement <32 x i32> %511, i32 %compare.local.base1385, i64 30, !dbg !16
  %513 = insertelement <32 x i32> %512, i32 %compare.local.base1389, i64 31, !dbg !16
  %514 = or <32 x i32> %513, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %515 = icmp ugt <32 x i32> %88, %514, !dbg !16
  %516 = insertelement <32 x i1> poison, i1 %compare.high.equal1263, i64 0, !dbg !16
  %517 = shufflevector <32 x i1> %516, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %518 = insertelement <32 x i1> poison, i1 %compare.high.order1264, i64 0, !dbg !16
  %519 = shufflevector <32 x i1> %518, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !16
  %520 = select <32 x i1> %517, <32 x i1> %515, <32 x i1> %519, !dbg !16
  %521 = select <32 x i1> %520, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %522 = insertvalue [2 x <32 x float>] poison, <32 x float> %482, 0, !dbg !12
  %523 = insertvalue [2 x <32 x float>] %522, <32 x float> %521, 1, !dbg !12
  br label %524, !dbg !12

524:                                              ; preds = %440, %91
  %525 = phi [2 x <32 x float>] [ %523, %440 ], [ zeroinitializer, %91 ], !dbg !12
  %526 = extractvalue [2 x <32 x float>] %525, 0, !dbg !19
  %527 = shufflevector <32 x float> %526, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %528 = fadd <4 x float> %393, %527, !dbg !19
  %529 = extractelement <4 x float> %528, i64 0, !dbg !20
  %530 = tail call float @llvm.amdgcn.exp2.f32(float %529), !dbg !20
  %531 = insertelement <4 x float> poison, float %530, i64 0, !dbg !21
  %532 = extractelement <4 x float> %528, i64 1, !dbg !20
  %533 = tail call float @llvm.amdgcn.exp2.f32(float %532), !dbg !20
  %534 = insertelement <4 x float> %531, float %533, i64 1, !dbg !21
  %535 = fadd float %530, %533, !dbg !22
  %536 = extractelement <4 x float> %528, i64 2, !dbg !20
  %537 = tail call float @llvm.amdgcn.exp2.f32(float %536), !dbg !20
  %538 = insertelement <4 x float> %534, float %537, i64 2, !dbg !21
  %539 = fadd float %537, %535, !dbg !22
  %540 = extractelement <4 x float> %528, i64 3, !dbg !20
  %541 = tail call float @llvm.amdgcn.exp2.f32(float %540), !dbg !20
  %542 = insertelement <4 x float> %538, float %541, i64 3, !dbg !21
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %542) #8, !dbg !21, !frisk.register_lifetimes !5
  %543 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !21
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %543) #8, !dbg !22, !frisk.register_lifetimes !5
  %544 = fadd float %541, %539, !dbg !22
  %545 = shufflevector <32 x float> %526, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %546 = fadd <4 x float> %397, %545, !dbg !19
  %547 = extractelement <4 x float> %546, i64 0, !dbg !20
  %548 = tail call float @llvm.amdgcn.exp2.f32(float %547), !dbg !20
  %549 = insertelement <4 x float> poison, float %548, i64 0, !dbg !21
  %550 = fadd float %548, %544, !dbg !22
  %551 = extractelement <4 x float> %546, i64 1, !dbg !20
  %552 = tail call float @llvm.amdgcn.exp2.f32(float %551), !dbg !20
  %553 = insertelement <4 x float> %549, float %552, i64 1, !dbg !21
  %554 = fadd float %552, %550, !dbg !22
  %555 = extractelement <4 x float> %546, i64 2, !dbg !20
  %556 = tail call float @llvm.amdgcn.exp2.f32(float %555), !dbg !20
  %557 = insertelement <4 x float> %553, float %556, i64 2, !dbg !21
  %558 = fadd float %556, %554, !dbg !22
  %559 = extractelement <4 x float> %546, i64 3, !dbg !20
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %559), !dbg !20
  %561 = insertelement <4 x float> %557, float %560, i64 3, !dbg !21
  %math.result306 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %561) #8, !dbg !21, !frisk.register_lifetimes !5
  %562 = fptrunc <4 x float> %math.result306 to <4 x half>, !dbg !21
  %math.packed307 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %562) #8, !dbg !22, !frisk.register_lifetimes !5
  %563 = fadd float %560, %558, !dbg !22
  %564 = shufflevector <32 x float> %526, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %565 = fadd <4 x float> %401, %564, !dbg !19
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
  %math.result308 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %580) #8, !dbg !21, !frisk.register_lifetimes !5
  %581 = fptrunc <4 x float> %math.result308 to <4 x half>, !dbg !21
  %math.packed309 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %581) #8, !dbg !22, !frisk.register_lifetimes !5
  %582 = fadd float %579, %577, !dbg !22
  %583 = shufflevector <32 x float> %526, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %584 = fadd <4 x float> %405, %583, !dbg !19
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
  %math.result310 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %599) #8, !dbg !21, !frisk.register_lifetimes !5
  %600 = fptrunc <4 x float> %math.result310 to <4 x half>, !dbg !21
  %math.packed311 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %600) #8, !dbg !22, !frisk.register_lifetimes !5
  %601 = fadd float %598, %596, !dbg !22
  %602 = shufflevector <32 x float> %526, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %603 = fadd <4 x float> %409, %602, !dbg !19
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
  %math.result312 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %618) #8, !dbg !21, !frisk.register_lifetimes !5
  %619 = fptrunc <4 x float> %math.result312 to <4 x half>, !dbg !21
  %math.packed313 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %619) #8, !dbg !22, !frisk.register_lifetimes !5
  %620 = fadd float %617, %615, !dbg !22
  %621 = shufflevector <32 x float> %526, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %622 = fadd <4 x float> %413, %621, !dbg !19
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
  %math.result314 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %637) #8, !dbg !21, !frisk.register_lifetimes !5
  %638 = fptrunc <4 x float> %math.result314 to <4 x half>, !dbg !21
  %math.packed315 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %638) #8, !dbg !22, !frisk.register_lifetimes !5
  %639 = fadd float %636, %634, !dbg !22
  %640 = shufflevector <32 x float> %526, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %641 = fadd <4 x float> %417, %640, !dbg !19
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
  %math.result316 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %656) #8, !dbg !21, !frisk.register_lifetimes !5
  %657 = fptrunc <4 x float> %math.result316 to <4 x half>, !dbg !21
  %math.packed317 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %657) #8, !dbg !22, !frisk.register_lifetimes !5
  %658 = fadd float %655, %653, !dbg !22
  %659 = shufflevector <32 x float> %526, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %660 = fadd <4 x float> %420, %659, !dbg !19
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
  %math.result318 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %675) #8, !dbg !21, !frisk.register_lifetimes !5
  %676 = fptrunc <4 x float> %math.result318 to <4 x half>, !dbg !21
  %math.packed319 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %676) #8, !dbg !22, !frisk.register_lifetimes !5
  %677 = fadd float %674, %672, !dbg !22
  %678 = extractvalue [2 x <32 x float>] %525, 1, !dbg !19
  %679 = shufflevector <32 x float> %678, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %680 = fadd <4 x float> %422, %679, !dbg !19
  %681 = extractelement <4 x float> %680, i64 0, !dbg !20
  %682 = tail call float @llvm.amdgcn.exp2.f32(float %681), !dbg !20
  %683 = insertelement <4 x float> poison, float %682, i64 0, !dbg !21
  %684 = extractelement <4 x float> %680, i64 1, !dbg !20
  %685 = tail call float @llvm.amdgcn.exp2.f32(float %684), !dbg !20
  %686 = insertelement <4 x float> %683, float %685, i64 1, !dbg !21
  %687 = fadd float %682, %685, !dbg !22
  %688 = extractelement <4 x float> %680, i64 2, !dbg !20
  %689 = tail call float @llvm.amdgcn.exp2.f32(float %688), !dbg !20
  %690 = insertelement <4 x float> %686, float %689, i64 2, !dbg !21
  %691 = fadd float %689, %687, !dbg !22
  %692 = extractelement <4 x float> %680, i64 3, !dbg !20
  %693 = tail call float @llvm.amdgcn.exp2.f32(float %692), !dbg !20
  %694 = insertelement <4 x float> %690, float %693, i64 3, !dbg !21
  %math.result320 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %694) #8, !dbg !21, !frisk.register_lifetimes !5
  %695 = fptrunc <4 x float> %math.result320 to <4 x half>, !dbg !21
  %math.packed321 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %695) #8, !dbg !22, !frisk.register_lifetimes !5
  %696 = fadd float %693, %691, !dbg !22
  %697 = shufflevector <32 x float> %678, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %698 = fadd <4 x float> %424, %697, !dbg !19
  %699 = extractelement <4 x float> %698, i64 0, !dbg !20
  %700 = tail call float @llvm.amdgcn.exp2.f32(float %699), !dbg !20
  %701 = insertelement <4 x float> poison, float %700, i64 0, !dbg !21
  %702 = fadd float %700, %696, !dbg !22
  %703 = extractelement <4 x float> %698, i64 1, !dbg !20
  %704 = tail call float @llvm.amdgcn.exp2.f32(float %703), !dbg !20
  %705 = insertelement <4 x float> %701, float %704, i64 1, !dbg !21
  %706 = fadd float %704, %702, !dbg !22
  %707 = extractelement <4 x float> %698, i64 2, !dbg !20
  %708 = tail call float @llvm.amdgcn.exp2.f32(float %707), !dbg !20
  %709 = insertelement <4 x float> %705, float %708, i64 2, !dbg !21
  %710 = fadd float %708, %706, !dbg !22
  %711 = extractelement <4 x float> %698, i64 3, !dbg !20
  %712 = tail call float @llvm.amdgcn.exp2.f32(float %711), !dbg !20
  %713 = insertelement <4 x float> %709, float %712, i64 3, !dbg !21
  %math.result322 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %713) #8, !dbg !21, !frisk.register_lifetimes !5
  %714 = fptrunc <4 x float> %math.result322 to <4 x half>, !dbg !21
  %math.packed323 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %714) #8, !dbg !22, !frisk.register_lifetimes !5
  %715 = fadd float %712, %710, !dbg !22
  %716 = shufflevector <32 x float> %678, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %717 = fadd <4 x float> %426, %716, !dbg !19
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
  %math.result324 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %732) #8, !dbg !21, !frisk.register_lifetimes !5
  %733 = fptrunc <4 x float> %math.result324 to <4 x half>, !dbg !21
  %math.packed325 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %733) #8, !dbg !22, !frisk.register_lifetimes !5
  %734 = fadd float %731, %729, !dbg !22
  %735 = shufflevector <32 x float> %678, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %736 = fadd <4 x float> %428, %735, !dbg !19
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
  %math.result326 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %751) #8, !dbg !21, !frisk.register_lifetimes !5
  %752 = fptrunc <4 x float> %math.result326 to <4 x half>, !dbg !21
  %math.packed327 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %752) #8, !dbg !22, !frisk.register_lifetimes !5
  %753 = fadd float %750, %748, !dbg !22
  %754 = shufflevector <32 x float> %678, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %755 = fadd <4 x float> %430, %754, !dbg !19
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
  %math.result328 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %770) #8, !dbg !21, !frisk.register_lifetimes !5
  %771 = fptrunc <4 x float> %math.result328 to <4 x half>, !dbg !21
  %math.packed329 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %771) #8, !dbg !22, !frisk.register_lifetimes !5
  %772 = fadd float %769, %767, !dbg !22
  %773 = shufflevector <32 x float> %678, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %774 = fadd <4 x float> %432, %773, !dbg !19
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
  %math.result330 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %789) #8, !dbg !21, !frisk.register_lifetimes !5
  %790 = fptrunc <4 x float> %math.result330 to <4 x half>, !dbg !21
  %math.packed331 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %790) #8, !dbg !22, !frisk.register_lifetimes !5
  %791 = fadd float %788, %786, !dbg !22
  %792 = shufflevector <32 x float> %678, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %793 = fadd <4 x float> %434, %792, !dbg !19
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
  %math.result332 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %808) #8, !dbg !21, !frisk.register_lifetimes !5
  %809 = fptrunc <4 x float> %math.result332 to <4 x half>, !dbg !21
  %math.packed333 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %809) #8, !dbg !22, !frisk.register_lifetimes !5
  %810 = fadd float %807, %805, !dbg !22
  %811 = shufflevector <32 x float> %678, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %812 = fadd <4 x float> %438, %811, !dbg !19
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
  %math.result334 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %827) #8, !dbg !21, !frisk.register_lifetimes !5
  %828 = fptrunc <4 x float> %math.result334 to <4 x half>, !dbg !21
  %math.packed335 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %828) #8, !dbg !22, !frisk.register_lifetimes !5
  %829 = fadd float %826, %824, !dbg !22
  %830 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !22
  %831 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %830), !dbg !22
  %832 = add nuw nsw i32 %831, 64, !dbg !22
  %833 = xor i32 %831, 16, !dbg !22
  %834 = and i32 %832, 192, !dbg !22
  %835 = icmp samesign ult i32 %833, %834, !dbg !22
  %836 = select i1 %835, i32 %833, i32 %831, !dbg !22
  %837 = shl nuw nsw i32 %836, 2, !dbg !22
  %838 = bitcast float %677 to i32, !dbg !22
  %839 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %837, i32 %838), !dbg !22
  %840 = bitcast i32 %839 to float, !dbg !22
  %841 = fadd float %677, %840, !dbg !22
  %842 = xor i32 %831, 32, !dbg !22
  %843 = icmp samesign ult i32 %842, %834, !dbg !22
  %844 = select i1 %843, i32 %842, i32 %831, !dbg !22
  %845 = shl nuw nsw i32 %844, 2, !dbg !22
  %846 = bitcast float %841 to i32, !dbg !22
  %847 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %845, i32 %846), !dbg !22
  %848 = bitcast i32 %847 to float, !dbg !22
  %849 = fadd float %841, %848, !dbg !22
  %850 = extractvalue [2 x <1 x float>] %95, 0, !dbg !22
  %851 = extractelement <1 x float> %850, i64 0, !dbg !22
  %852 = fadd float %851, %849, !dbg !22
  %853 = insertelement <1 x float> poison, float %852, i64 0, !dbg !11
  %854 = insertvalue [2 x <1 x float>] poison, <1 x float> %853, 0, !dbg !11
  %855 = bitcast float %829 to i32, !dbg !22
  %856 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %837, i32 %855), !dbg !22
  %857 = bitcast i32 %856 to float, !dbg !22
  %858 = fadd float %829, %857, !dbg !22
  %859 = bitcast float %858 to i32, !dbg !22
  %860 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %845, i32 %859), !dbg !22
  %861 = bitcast i32 %860 to float, !dbg !22
  %862 = fadd float %858, %861, !dbg !22
  %863 = extractvalue [2 x <1 x float>] %95, 1, !dbg !22
  %864 = extractelement <1 x float> %863, i64 0, !dbg !22
  %865 = fadd float %864, %862, !dbg !22
  %866 = insertelement <1 x float> poison, float %865, i64 0, !dbg !11
  %867 = insertvalue [2 x <1 x float>] %854, <1 x float> %866, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread844 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !23, !range !9, !frisk.register_lifetimes !5
  %address.local845 = shl nuw nsw i32 %address.thread844, 3, !dbg !23
  %address.local846 = and i32 %address.local845, 120, !dbg !23
  %address.local847 = shl nuw nsw i32 %address.thread844, 5, !dbg !23
  %address.local848 = and i32 %address.local847, 1536, !dbg !23
  %address.local849 = or disjoint i32 %address.local846, %address.local848, !dbg !23
  %address.local850 = zext nneg i32 %address.local849 to i64, !dbg !23
  %address.local851 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local850, !dbg !23
  %868 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #11, !dbg !23
  %869 = bitcast <4 x i32> %868 to <8 x half>, !dbg !23
  %address.local852 = zext nneg i32 %address.thread844 to i64, !dbg !8
  %address.local853 = and i64 %address.local852, 60, !dbg !23
  %address.local854 = add nuw nsw i64 %address.local853, %address.local852, !dbg !23
  %address.local856.idx = shl nuw nsw i64 %address.local854, 4, !dbg !23
  %address.local856 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local856.idx, !dbg !23
  %address.local857 = lshr i64 %address.local852, 4, !dbg !23
  %address.local858 = and i64 %address.local857, 3, !dbg !23
  %address.local860.idx = mul nuw nsw i64 %address.local858, 768, !dbg !23
  %address.local860 = getelementptr i8, ptr addrspace(3) %address.local856, i64 %address.local860.idx, !dbg !23
  %address.local861 = lshr i64 %address.local852, 2, !dbg !14
  %address.local863.idx = mul nsw i64 %address.local861, -64, !dbg !23
  %address.local863 = getelementptr i8, ptr addrspace(3) %address.local860, i64 %address.local863.idx, !dbg !23
  %address.local864 = getelementptr i8, ptr addrspace(3) %address.local863, i64 256, !dbg !23
  %870 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #11, !dbg !23
  %871 = bitcast <4 x i32> %870 to <8 x half>, !dbg !23
  %address.local868 = getelementptr i8, ptr addrspace(3) %address.local863, i64 512, !dbg !23
  %872 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #11, !dbg !23
  %873 = bitcast <4 x i32> %872 to <8 x half>, !dbg !23
  %address.local872 = getelementptr i8, ptr addrspace(3) %address.local863, i64 768, !dbg !23
  %874 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #11, !dbg !23
  %875 = bitcast <4 x i32> %874 to <8 x half>, !dbg !23
  %address.local874 = getelementptr i8, ptr addrspace(3) %address.local851, i64 4096, !dbg !23
  %876 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #11, !dbg !23
  %877 = bitcast <4 x i32> %876 to <8 x half>, !dbg !23
  %address.local878 = getelementptr i8, ptr addrspace(3) %address.local863, i64 4352, !dbg !23
  %878 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #11, !dbg !23
  %879 = bitcast <4 x i32> %878 to <8 x half>, !dbg !23
  %address.local882 = getelementptr i8, ptr addrspace(3) %address.local863, i64 4608, !dbg !23
  %880 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #11, !dbg !23
  %881 = bitcast <4 x i32> %880 to <8 x half>, !dbg !23
  %address.local886 = getelementptr i8, ptr addrspace(3) %address.local863, i64 4864, !dbg !23
  %882 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #11, !dbg !23
  %883 = bitcast <4 x i32> %882 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local887 = shl nuw nsw i64 %address.local852, 3, !dbg !8
  %address.local888 = and i64 %address.local887, 1536, !dbg !8
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local888, !dbg !8
  %address.local890 = ptrtoint ptr addrspace(3) %address.local889 to i64, !dbg !8
  %address.local891 = trunc i64 %address.local890 to i32, !dbg !8
  %884 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local891), !dbg !17
  %address.local892 = shl nuw nsw i64 %address.local852, 5, !dbg !17
  %address.local893 = and i64 %address.local892, 8064, !dbg !17
  %address.local895 = and i64 %address.local887, 24, !dbg !17
  %address.local896 = or disjoint i64 %address.local893, %address.local895, !dbg !17
  %885 = or disjoint i64 %.idx5962, %address.local896, !dbg !17
  %886 = trunc i64 %885 to i32, !dbg !17
  %.tr70 = shl i32 %886, 1, !dbg !17
  %dma.buffer.offset1543 = or i32 %.tr70, 64, !dbg !17
  %dma.buffer.lds1545 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %884), !dbg !17
  %887 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1545, i32 %dma.buffer.offset1543, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %address.local898 = or disjoint i64 %address.local852, 256, !dbg !8
  %address.local899 = shl nuw nsw i64 %address.local898, 3, !dbg !8
  %address.local900 = and i64 %address.local899, 3584, !dbg !8
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local900, !dbg !8
  %address.local902 = ptrtoint ptr addrspace(3) %address.local901 to i64, !dbg !8
  %address.local903 = trunc i64 %address.local902 to i32, !dbg !8
  %888 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local903), !dbg !17
  %address.local904 = shl nuw nsw i64 %address.local898, 5, !dbg !17
  %address.local905 = and i64 %address.local904, 16256, !dbg !17
  %address.local908 = or disjoint i64 %address.local905, %address.local895, !dbg !17
  %889 = or disjoint i64 %.idx5962, %address.local908, !dbg !17
  %890 = trunc i64 %889 to i32, !dbg !17
  %.tr74 = shl i32 %890, 1, !dbg !17
  %dma.buffer.offset1540 = or i32 %.tr74, 64, !dbg !17
  %dma.buffer.lds1542 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %888), !dbg !17
  %891 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1542, i32 %dma.buffer.offset1540, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %892 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %893 = extractvalue [2 x <8 x float>] %96, 0, !dbg !24
  %894 = shufflevector <8 x float> %893, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %892, <4 x float> %894) #8, !dbg !23
  %896 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %897 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %896, <4 x float> %895) #8, !dbg !23
  %898 = shufflevector <8 x half> %873, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %898, <4 x float> %897) #8, !dbg !23
  %900 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %901 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %900, <4 x float> %899) #8, !dbg !23
  %902 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %903 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %902, <4 x float> %901) #8, !dbg !23
  %904 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %904, <4 x float> %903) #8, !dbg !23
  %906 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %906, <4 x float> %905) #8, !dbg !23
  %908 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %909 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %908, <4 x float> %907) #8, !dbg !23
  %910 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %911 = shufflevector <8 x float> %893, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %910, <4 x float> %911) #8, !dbg !23
  %913 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %913, <4 x float> %912) #8, !dbg !23
  %915 = shufflevector <8 x half> %873, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %915, <4 x float> %914) #8, !dbg !23
  %917 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %917, <4 x float> %916) #8, !dbg !23
  %919 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %919, <4 x float> %918) #8, !dbg !23
  %921 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %921, <4 x float> %920) #8, !dbg !23
  %923 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %923, <4 x float> %922) #8, !dbg !23
  %925 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %925, <4 x float> %924) #8, !dbg !23
  %927 = extractvalue [2 x <8 x float>] %96, 1, !dbg !24
  %928 = shufflevector <8 x float> %927, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %892, <4 x float> %928) #8, !dbg !23
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %896, <4 x float> %929) #8, !dbg !23
  %931 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %898, <4 x float> %930) #8, !dbg !23
  %932 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %900, <4 x float> %931) #8, !dbg !23
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %902, <4 x float> %932) #8, !dbg !23
  %934 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %904, <4 x float> %933) #8, !dbg !23
  %935 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %906, <4 x float> %934) #8, !dbg !23
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %908, <4 x float> %935) #8, !dbg !23
  %937 = shufflevector <8 x float> %927, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %910, <4 x float> %937) #8, !dbg !23
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %913, <4 x float> %938) #8, !dbg !23
  %940 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %915, <4 x float> %939) #8, !dbg !23
  %941 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %917, <4 x float> %940) #8, !dbg !23
  %942 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %919, <4 x float> %941) #8, !dbg !23
  %943 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %921, <4 x float> %942) #8, !dbg !23
  %944 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %923, <4 x float> %943) #8, !dbg !23
  %945 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %925, <4 x float> %944) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %946 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #11, !dbg !23
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !23
  %948 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #11, !dbg !23
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !23
  %950 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #11, !dbg !23
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !23
  %952 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #11, !dbg !23
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !23
  %954 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #11, !dbg !23
  %955 = bitcast <4 x i32> %954 to <8 x half>, !dbg !23
  %956 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #11, !dbg !23
  %957 = bitcast <4 x i32> %956 to <8 x half>, !dbg !23
  %958 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #11, !dbg !23
  %959 = bitcast <4 x i32> %958 to <8 x half>, !dbg !23
  %960 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #11, !dbg !23
  %961 = bitcast <4 x i32> %960 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %dma.buffer.offset1537 = or i32 %.tr70, 128, !dbg !17
  %962 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1545, i32 %dma.buffer.offset1537, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %dma.buffer.offset1534 = or i32 %.tr74, 128, !dbg !17
  %963 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1542, i32 %dma.buffer.offset1534, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %964 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %965 = extractvalue [2 x <8 x float>] %94, 0, !dbg !24
  %966 = shufflevector <8 x float> %965, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %964, <4 x float> %966) #8, !dbg !23
  %968 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %968, <4 x float> %967) #8, !dbg !23
  %970 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %971 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %970, <4 x float> %969) #8, !dbg !23
  %972 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %973 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %972, <4 x float> %971) #8, !dbg !23
  %974 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %974, <4 x float> %973) #8, !dbg !23
  %976 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %977 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %976, <4 x float> %975) #8, !dbg !23
  %978 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %979 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %978, <4 x float> %977) #8, !dbg !23
  %980 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %980, <4 x float> %979) #8, !dbg !23
  %982 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %983 = shufflevector <8 x float> %965, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %982, <4 x float> %983) #8, !dbg !23
  %985 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %985, <4 x float> %984) #8, !dbg !23
  %987 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %987, <4 x float> %986) #8, !dbg !23
  %989 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %989, <4 x float> %988) #8, !dbg !23
  %991 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %991, <4 x float> %990) #8, !dbg !23
  %993 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %993, <4 x float> %992) #8, !dbg !23
  %995 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %995, <4 x float> %994) #8, !dbg !23
  %997 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %997, <4 x float> %996) #8, !dbg !23
  %999 = extractvalue [2 x <8 x float>] %94, 1, !dbg !24
  %1000 = shufflevector <8 x float> %999, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1001 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %964, <4 x float> %1000) #8, !dbg !23
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %968, <4 x float> %1001) #8, !dbg !23
  %1003 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %970, <4 x float> %1002) #8, !dbg !23
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %972, <4 x float> %1003) #8, !dbg !23
  %1005 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %974, <4 x float> %1004) #8, !dbg !23
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %976, <4 x float> %1005) #8, !dbg !23
  %1007 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %978, <4 x float> %1006) #8, !dbg !23
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %980, <4 x float> %1007) #8, !dbg !23
  %1009 = shufflevector <8 x float> %999, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %982, <4 x float> %1009) #8, !dbg !23
  %1011 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %985, <4 x float> %1010) #8, !dbg !23
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %987, <4 x float> %1011) #8, !dbg !23
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %989, <4 x float> %1012) #8, !dbg !23
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %991, <4 x float> %1013) #8, !dbg !23
  %1015 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %993, <4 x float> %1014) #8, !dbg !23
  %1016 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %995, <4 x float> %1015) #8, !dbg !23
  %1017 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %997, <4 x float> %1016) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1018 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #11, !dbg !23
  %1019 = bitcast <4 x i32> %1018 to <8 x half>, !dbg !23
  %1020 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #11, !dbg !23
  %1021 = bitcast <4 x i32> %1020 to <8 x half>, !dbg !23
  %1022 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #11, !dbg !23
  %1023 = bitcast <4 x i32> %1022 to <8 x half>, !dbg !23
  %1024 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #11, !dbg !23
  %1025 = bitcast <4 x i32> %1024 to <8 x half>, !dbg !23
  %1026 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #11, !dbg !23
  %1027 = bitcast <4 x i32> %1026 to <8 x half>, !dbg !23
  %1028 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #11, !dbg !23
  %1029 = bitcast <4 x i32> %1028 to <8 x half>, !dbg !23
  %1030 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #11, !dbg !23
  %1031 = bitcast <4 x i32> %1030 to <8 x half>, !dbg !23
  %1032 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #11, !dbg !23
  %1033 = bitcast <4 x i32> %1032 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1034 = or disjoint i64 %.idx5962, 96, !dbg !17
  %1035 = or disjoint i64 %address.local896, %1034, !dbg !17
  %.tr86 = trunc i64 %1035 to i32, !dbg !17
  %dma.buffer.offset1531 = shl i32 %.tr86, 1, !dbg !17
  %1036 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1545, i32 %dma.buffer.offset1531, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %1037 = or disjoint i64 %address.local908, %1034, !dbg !17
  %.tr90 = trunc i64 %1037 to i32, !dbg !17
  %dma.buffer.offset = shl i32 %.tr90, 1, !dbg !17
  %1038 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1542, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1550) #9, !dbg !17
  %1039 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1040 = extractvalue [2 x <8 x float>] %93, 0, !dbg !24
  %1041 = shufflevector <8 x float> %1040, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1042 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1039, <4 x float> %1041) #8, !dbg !23
  %1043 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1044 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1043, <4 x float> %1042) #8, !dbg !23
  %1045 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1046 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1045, <4 x float> %1044) #8, !dbg !23
  %1047 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1048 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1047, <4 x float> %1046) #8, !dbg !23
  %1049 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1050 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1049, <4 x float> %1048) #8, !dbg !23
  %1051 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1052 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1051, <4 x float> %1050) #8, !dbg !23
  %1053 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1054 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1053, <4 x float> %1052) #8, !dbg !23
  %1055 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1056 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1055, <4 x float> %1054) #8, !dbg !23
  %1057 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1058 = shufflevector <8 x float> %1040, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1059 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1057, <4 x float> %1058) #8, !dbg !23
  %1060 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1061 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1060, <4 x float> %1059) #8, !dbg !23
  %1062 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1063 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1062, <4 x float> %1061) #8, !dbg !23
  %1064 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1065 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1064, <4 x float> %1063) #8, !dbg !23
  %1066 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1067 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1066, <4 x float> %1065) #8, !dbg !23
  %1068 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1069 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1068, <4 x float> %1067) #8, !dbg !23
  %1070 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1071 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1070, <4 x float> %1069) #8, !dbg !23
  %1072 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1073 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1072, <4 x float> %1071) #8, !dbg !23
  %1074 = extractvalue [2 x <8 x float>] %93, 1, !dbg !24
  %1075 = shufflevector <8 x float> %1074, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1039, <4 x float> %1075) #8, !dbg !23
  %1077 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1043, <4 x float> %1076) #8, !dbg !23
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1045, <4 x float> %1077) #8, !dbg !23
  %1079 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1047, <4 x float> %1078) #8, !dbg !23
  %1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1049, <4 x float> %1079) #8, !dbg !23
  %1081 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1051, <4 x float> %1080) #8, !dbg !23
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1053, <4 x float> %1081) #8, !dbg !23
  %1083 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1055, <4 x float> %1082) #8, !dbg !23
  %1084 = shufflevector <8 x float> %1074, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1085 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1057, <4 x float> %1084) #8, !dbg !23
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1060, <4 x float> %1085) #8, !dbg !23
  %1087 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1062, <4 x float> %1086) #8, !dbg !23
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1064, <4 x float> %1087) #8, !dbg !23
  %1089 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1066, <4 x float> %1088) #8, !dbg !23
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1068, <4 x float> %1089) #8, !dbg !23
  %1091 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1070, <4 x float> %1090) #8, !dbg !23
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1072, <4 x float> %1091) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1093 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #11, !dbg !23
  %1094 = bitcast <4 x i32> %1093 to <8 x half>, !dbg !23
  %1095 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local864, i32 0) #11, !dbg !23
  %1096 = bitcast <4 x i32> %1095 to <8 x half>, !dbg !23
  %1097 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local868, i32 0) #11, !dbg !23
  %1098 = bitcast <4 x i32> %1097 to <8 x half>, !dbg !23
  %1099 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local872, i32 0) #11, !dbg !23
  %1100 = bitcast <4 x i32> %1099 to <8 x half>, !dbg !23
  %1101 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #11, !dbg !23
  %1102 = bitcast <4 x i32> %1101 to <8 x half>, !dbg !23
  %1103 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #11, !dbg !23
  %1104 = bitcast <4 x i32> %1103 to <8 x half>, !dbg !23
  %1105 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local882, i32 0) #11, !dbg !23
  %1106 = bitcast <4 x i32> %1105 to <8 x half>, !dbg !23
  %1107 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #11, !dbg !23
  %1108 = bitcast <4 x i32> %1107 to <8 x half>, !dbg !23
  %1109 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1110 = extractvalue [2 x <8 x float>] %92, 0, !dbg !24
  %1111 = shufflevector <8 x float> %1110, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1109, <4 x float> %1111) #8, !dbg !23
  %1113 = shufflevector <8 x half> %1096, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1114 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1113, <4 x float> %1112) #8, !dbg !23
  %1115 = shufflevector <8 x half> %1098, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1115, <4 x float> %1114) #8, !dbg !23
  %1117 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1118 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1117, <4 x float> %1116) #8, !dbg !23
  %1119 = shufflevector <8 x half> %1102, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1119, <4 x float> %1118) #8, !dbg !23
  %1121 = shufflevector <8 x half> %1104, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1122 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1121, <4 x float> %1120) #8, !dbg !23
  %1123 = shufflevector <8 x half> %1106, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1124 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1123, <4 x float> %1122) #8, !dbg !23
  %1125 = shufflevector <8 x half> %1108, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1126 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1125, <4 x float> %1124) #8, !dbg !23
  %1127 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1128 = shufflevector <8 x float> %1110, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1127, <4 x float> %1128) #8, !dbg !23
  %1130 = shufflevector <8 x half> %1096, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1131 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1130, <4 x float> %1129) #8, !dbg !23
  %1132 = shufflevector <8 x half> %1098, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1132, <4 x float> %1131) #8, !dbg !23
  %1134 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1135 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1134, <4 x float> %1133) #8, !dbg !23
  %1136 = shufflevector <8 x half> %1102, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1136, <4 x float> %1135) #8, !dbg !23
  %1138 = shufflevector <8 x half> %1104, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1139 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1138, <4 x float> %1137) #8, !dbg !23
  %1140 = shufflevector <8 x half> %1106, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1140, <4 x float> %1139) #8, !dbg !23
  %1142 = shufflevector <8 x half> %1108, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1142, <4 x float> %1141) #8, !dbg !23
  %1144 = extractvalue [2 x <8 x float>] %92, 1, !dbg !24
  %1145 = shufflevector <8 x float> %1144, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1109, <4 x float> %1145) #8, !dbg !23
  %1147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1113, <4 x float> %1146) #8, !dbg !23
  %1148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1115, <4 x float> %1147) #8, !dbg !23
  %1149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1117, <4 x float> %1148) #8, !dbg !23
  %1150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1119, <4 x float> %1149) #8, !dbg !23
  %1151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1121, <4 x float> %1150) #8, !dbg !23
  %1152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1123, <4 x float> %1151) #8, !dbg !23
  %1153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1125, <4 x float> %1152) #8, !dbg !23
  %1154 = shufflevector <8 x float> %1144, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1127, <4 x float> %1154) #8, !dbg !23
  %1156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1130, <4 x float> %1155) #8, !dbg !23
  %1157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1132, <4 x float> %1156) #8, !dbg !23
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1134, <4 x float> %1157) #8, !dbg !23
  %1159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1136, <4 x float> %1158) #8, !dbg !23
  %1160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1138, <4 x float> %1159) #8, !dbg !23
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1140, <4 x float> %1160) #8, !dbg !23
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1142, <4 x float> %1161) #8, !dbg !23
  %1163 = shufflevector <4 x float> %909, <4 x float> %926, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1164 = insertvalue [2 x <8 x float>] poison, <8 x float> %1163, 0, !dbg !11
  %1165 = shufflevector <4 x float> %936, <4 x float> %945, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1166 = insertvalue [2 x <8 x float>] %1164, <8 x float> %1165, 1, !dbg !11
  %1167 = shufflevector <4 x float> %981, <4 x float> %998, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1168 = insertvalue [2 x <8 x float>] poison, <8 x float> %1167, 0, !dbg !11
  %1169 = shufflevector <4 x float> %1008, <4 x float> %1017, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1170 = insertvalue [2 x <8 x float>] %1168, <8 x float> %1169, 1, !dbg !11
  %1171 = shufflevector <4 x float> %1056, <4 x float> %1073, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1172 = insertvalue [2 x <8 x float>] poison, <8 x float> %1171, 0, !dbg !11
  %1173 = shufflevector <4 x float> %1083, <4 x float> %1092, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1174 = insertvalue [2 x <8 x float>] %1172, <8 x float> %1173, 1, !dbg !11
  %1175 = shufflevector <4 x float> %1126, <4 x float> %1143, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1176 = insertvalue [2 x <8 x float>] poison, <8 x float> %1175, 0, !dbg !11
  %1177 = shufflevector <4 x float> %1153, <4 x float> %1162, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1178 = insertvalue [2 x <8 x float>] %1176, <8 x float> %1177, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1179 = add nuw nsw i64 %97, 1, !dbg !11
  %1180 = icmp slt i64 %1179, %57, !dbg !11
  br i1 %1180, label %91, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %524, %.._crit_edge_crit_edge
  %.pre-phi116 = phi i32 [ %.pre115, %.._crit_edge_crit_edge ], [ %842, %524 ], !dbg !11
  %.pre-phi114 = phi i32 [ %.pre113, %.._crit_edge_crit_edge ], [ %834, %524 ], !dbg !11
  %.pre-phi112 = phi i32 [ %.pre111, %.._crit_edge_crit_edge ], [ %833, %524 ], !dbg !11
  %.pre-phi108 = phi i32 [ %.pre107, %.._crit_edge_crit_edge ], [ %831, %524 ], !dbg !11
  %.lcssa96 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1166, %524 ], !dbg !11
  %.lcssa95 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %867, %524 ], !dbg !11
  %.lcssa94 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1170, %524 ], !dbg !11
  %.lcssa93 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1174, %524 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1178, %524 ], !dbg !11
  %1181 = and i32 %29, 16, !dbg !11
  %.not = icmp eq i32 %1181, 0, !dbg !11
  %1182 = and i32 %29, 32, !dbg !11
  %.not36 = icmp eq i32 %1182, 0, !dbg !11
  %1183 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !11
  %1184 = extractelement <8 x float> %1183, i64 0, !dbg !11
  %1185 = extractelement <8 x float> %1183, i64 1, !dbg !11
  %1186 = extractelement <8 x float> %1183, i64 2, !dbg !11
  %1187 = extractelement <8 x float> %1183, i64 3, !dbg !11
  %1188 = select i1 %.not, float %1185, float %1184, !dbg !11
  %1189 = icmp samesign ult i32 %.pre-phi112, %.pre-phi114, !dbg !11
  %1190 = select i1 %1189, i32 %.pre-phi112, i32 %.pre-phi108, !dbg !11
  %1191 = shl nuw nsw i32 %1190, 2, !dbg !11
  %1192 = bitcast float %1188 to i32, !dbg !11
  %1193 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1192), !dbg !11
  %1194 = bitcast i32 %1193 to float, !dbg !11
  %1195 = select i1 %.not, float %1184, float %1194, !dbg !11
  %1196 = select i1 %.not, float %1194, float %1185, !dbg !11
  %1197 = select i1 %.not, float %1187, float %1186, !dbg !11
  %1198 = bitcast float %1197 to i32, !dbg !11
  %1199 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1198), !dbg !11
  %1200 = bitcast i32 %1199 to float, !dbg !11
  %1201 = select i1 %.not, float %1186, float %1200, !dbg !11
  %1202 = select i1 %.not, float %1200, float %1187, !dbg !11
  %1203 = select i1 %.not36, float %1201, float %1195, !dbg !11
  %1204 = icmp samesign ult i32 %.pre-phi116, %.pre-phi114, !dbg !11
  %1205 = select i1 %1204, i32 %.pre-phi116, i32 %.pre-phi108, !dbg !11
  %1206 = shl nuw nsw i32 %1205, 2, !dbg !11
  %1207 = bitcast float %1203 to i32, !dbg !11
  %1208 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1207), !dbg !11
  %1209 = bitcast i32 %1208 to float, !dbg !11
  %1210 = select i1 %.not36, float %1202, float %1196, !dbg !11
  %1211 = bitcast float %1210 to i32, !dbg !11
  %1212 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1211), !dbg !11
  %1213 = bitcast i32 %1212 to float, !dbg !11
  %1214 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1215 = shufflevector <4 x i1> %1214, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1216 = insertelement <4 x float> poison, float %1195, i64 0, !dbg !11
  %1217 = insertelement <4 x float> %1216, float %1196, i64 1, !dbg !11
  %1218 = insertelement <4 x float> %1217, float %1209, i64 2, !dbg !11
  %1219 = insertelement <4 x float> %1218, float %1213, i64 3, !dbg !11
  %1220 = insertelement <4 x float> poison, float %1209, i64 0, !dbg !11
  %1221 = insertelement <4 x float> %1220, float %1213, i64 1, !dbg !11
  %1222 = insertelement <4 x float> %1221, float %1201, i64 2, !dbg !11
  %1223 = insertelement <4 x float> %1222, float %1202, i64 3, !dbg !11
  %1224 = select <4 x i1> %1215, <4 x float> %1219, <4 x float> %1223, !dbg !11
  %1225 = extractvalue [2 x <8 x float>] %.lcssa93, 0, !dbg !11
  %1226 = extractelement <8 x float> %1225, i64 0, !dbg !11
  %1227 = extractelement <8 x float> %1225, i64 1, !dbg !11
  %1228 = extractelement <8 x float> %1225, i64 2, !dbg !11
  %1229 = extractelement <8 x float> %1225, i64 3, !dbg !11
  %1230 = select i1 %.not, float %1227, float %1226, !dbg !11
  %1231 = bitcast float %1230 to i32, !dbg !11
  %1232 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1231), !dbg !11
  %1233 = bitcast i32 %1232 to float, !dbg !11
  %1234 = select i1 %.not, float %1226, float %1233, !dbg !11
  %1235 = select i1 %.not, float %1233, float %1227, !dbg !11
  %1236 = select i1 %.not, float %1229, float %1228, !dbg !11
  %1237 = bitcast float %1236 to i32, !dbg !11
  %1238 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1237), !dbg !11
  %1239 = bitcast i32 %1238 to float, !dbg !11
  %1240 = select i1 %.not, float %1228, float %1239, !dbg !11
  %1241 = select i1 %.not, float %1239, float %1229, !dbg !11
  %1242 = select i1 %.not36, float %1240, float %1234, !dbg !11
  %1243 = bitcast float %1242 to i32, !dbg !11
  %1244 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1243), !dbg !11
  %1245 = bitcast i32 %1244 to float, !dbg !11
  %1246 = select i1 %.not36, float %1241, float %1235, !dbg !11
  %1247 = bitcast float %1246 to i32, !dbg !11
  %1248 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1247), !dbg !11
  %1249 = bitcast i32 %1248 to float, !dbg !11
  %1250 = insertelement <4 x float> poison, float %1234, i64 0, !dbg !11
  %1251 = insertelement <4 x float> %1250, float %1235, i64 1, !dbg !11
  %1252 = insertelement <4 x float> %1251, float %1245, i64 2, !dbg !11
  %1253 = insertelement <4 x float> %1252, float %1249, i64 3, !dbg !11
  %1254 = insertelement <4 x float> poison, float %1245, i64 0, !dbg !11
  %1255 = insertelement <4 x float> %1254, float %1249, i64 1, !dbg !11
  %1256 = insertelement <4 x float> %1255, float %1240, i64 2, !dbg !11
  %1257 = insertelement <4 x float> %1256, float %1241, i64 3, !dbg !11
  %1258 = select <4 x i1> %1215, <4 x float> %1253, <4 x float> %1257, !dbg !11
  %1259 = extractvalue [2 x <8 x float>] %.lcssa94, 0, !dbg !11
  %1260 = extractelement <8 x float> %1259, i64 0, !dbg !11
  %1261 = extractelement <8 x float> %1259, i64 1, !dbg !11
  %1262 = extractelement <8 x float> %1259, i64 2, !dbg !11
  %1263 = extractelement <8 x float> %1259, i64 3, !dbg !11
  %1264 = select i1 %.not, float %1261, float %1260, !dbg !11
  %1265 = bitcast float %1264 to i32, !dbg !11
  %1266 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1265), !dbg !11
  %1267 = bitcast i32 %1266 to float, !dbg !11
  %1268 = select i1 %.not, float %1260, float %1267, !dbg !11
  %1269 = select i1 %.not, float %1267, float %1261, !dbg !11
  %1270 = select i1 %.not, float %1263, float %1262, !dbg !11
  %1271 = bitcast float %1270 to i32, !dbg !11
  %1272 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1271), !dbg !11
  %1273 = bitcast i32 %1272 to float, !dbg !11
  %1274 = select i1 %.not, float %1262, float %1273, !dbg !11
  %1275 = select i1 %.not, float %1273, float %1263, !dbg !11
  %1276 = select i1 %.not36, float %1274, float %1268, !dbg !11
  %1277 = bitcast float %1276 to i32, !dbg !11
  %1278 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1277), !dbg !11
  %1279 = bitcast i32 %1278 to float, !dbg !11
  %1280 = select i1 %.not36, float %1275, float %1269, !dbg !11
  %1281 = bitcast float %1280 to i32, !dbg !11
  %1282 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1281), !dbg !11
  %1283 = bitcast i32 %1282 to float, !dbg !11
  %1284 = insertelement <4 x float> poison, float %1268, i64 0, !dbg !11
  %1285 = insertelement <4 x float> %1284, float %1269, i64 1, !dbg !11
  %1286 = insertelement <4 x float> %1285, float %1279, i64 2, !dbg !11
  %1287 = insertelement <4 x float> %1286, float %1283, i64 3, !dbg !11
  %1288 = insertelement <4 x float> poison, float %1279, i64 0, !dbg !11
  %1289 = insertelement <4 x float> %1288, float %1283, i64 1, !dbg !11
  %1290 = insertelement <4 x float> %1289, float %1274, i64 2, !dbg !11
  %1291 = insertelement <4 x float> %1290, float %1275, i64 3, !dbg !11
  %1292 = select <4 x i1> %1215, <4 x float> %1287, <4 x float> %1291, !dbg !11
  %1293 = extractvalue [2 x <8 x float>] %.lcssa96, 0, !dbg !11
  %1294 = extractelement <8 x float> %1293, i64 0, !dbg !11
  %1295 = extractelement <8 x float> %1293, i64 1, !dbg !11
  %1296 = extractelement <8 x float> %1293, i64 2, !dbg !11
  %1297 = extractelement <8 x float> %1293, i64 3, !dbg !11
  %1298 = select i1 %.not, float %1295, float %1294, !dbg !11
  %1299 = bitcast float %1298 to i32, !dbg !11
  %1300 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1299), !dbg !11
  %1301 = bitcast i32 %1300 to float, !dbg !11
  %1302 = select i1 %.not, float %1294, float %1301, !dbg !11
  %1303 = select i1 %.not, float %1301, float %1295, !dbg !11
  %1304 = select i1 %.not, float %1297, float %1296, !dbg !11
  %1305 = bitcast float %1304 to i32, !dbg !11
  %1306 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1305), !dbg !11
  %1307 = bitcast i32 %1306 to float, !dbg !11
  %1308 = select i1 %.not, float %1296, float %1307, !dbg !11
  %1309 = select i1 %.not, float %1307, float %1297, !dbg !11
  %1310 = select i1 %.not36, float %1308, float %1302, !dbg !11
  %1311 = bitcast float %1310 to i32, !dbg !11
  %1312 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1311), !dbg !11
  %1313 = select i1 %.not36, float %1309, float %1303, !dbg !11
  %1314 = bitcast float %1313 to i32, !dbg !11
  %1315 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1314), !dbg !11
  %1316 = extractvalue [2 x <1 x float>] %.lcssa95, 0, !dbg !25
  %1317 = shufflevector <1 x float> %1316, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %1318 = extractelement <8 x float> %1183, i64 4, !dbg !11
  %1319 = extractelement <8 x float> %1183, i64 5, !dbg !11
  %1320 = extractelement <8 x float> %1183, i64 6, !dbg !11
  %1321 = extractelement <8 x float> %1183, i64 7, !dbg !11
  %1322 = select i1 %.not, float %1319, float %1318, !dbg !11
  %1323 = bitcast float %1322 to i32, !dbg !11
  %1324 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1323), !dbg !11
  %1325 = bitcast i32 %1324 to float, !dbg !11
  %1326 = select i1 %.not, float %1318, float %1325, !dbg !11
  %1327 = select i1 %.not, float %1325, float %1319, !dbg !11
  %1328 = select i1 %.not, float %1321, float %1320, !dbg !11
  %1329 = bitcast float %1328 to i32, !dbg !11
  %1330 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1329), !dbg !11
  %1331 = bitcast i32 %1330 to float, !dbg !11
  %1332 = select i1 %.not, float %1320, float %1331, !dbg !11
  %1333 = select i1 %.not, float %1331, float %1321, !dbg !11
  %1334 = select i1 %.not36, float %1332, float %1326, !dbg !11
  %1335 = bitcast float %1334 to i32, !dbg !11
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1335), !dbg !11
  %1337 = bitcast i32 %1336 to float, !dbg !11
  %1338 = select i1 %.not36, float %1333, float %1327, !dbg !11
  %1339 = bitcast float %1338 to i32, !dbg !11
  %1340 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1339), !dbg !11
  %1341 = bitcast i32 %1340 to float, !dbg !11
  %1342 = insertelement <4 x float> poison, float %1326, i64 0, !dbg !11
  %1343 = insertelement <4 x float> %1342, float %1327, i64 1, !dbg !11
  %1344 = insertelement <4 x float> %1343, float %1337, i64 2, !dbg !11
  %1345 = insertelement <4 x float> %1344, float %1341, i64 3, !dbg !11
  %1346 = insertelement <4 x float> poison, float %1337, i64 0, !dbg !11
  %1347 = insertelement <4 x float> %1346, float %1341, i64 1, !dbg !11
  %1348 = insertelement <4 x float> %1347, float %1332, i64 2, !dbg !11
  %1349 = insertelement <4 x float> %1348, float %1333, i64 3, !dbg !11
  %1350 = select <4 x i1> %1215, <4 x float> %1345, <4 x float> %1349, !dbg !11
  %1351 = extractelement <8 x float> %1225, i64 4, !dbg !11
  %1352 = extractelement <8 x float> %1225, i64 5, !dbg !11
  %1353 = extractelement <8 x float> %1225, i64 6, !dbg !11
  %1354 = extractelement <8 x float> %1225, i64 7, !dbg !11
  %1355 = select i1 %.not, float %1352, float %1351, !dbg !11
  %1356 = bitcast float %1355 to i32, !dbg !11
  %1357 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1356), !dbg !11
  %1358 = bitcast i32 %1357 to float, !dbg !11
  %1359 = select i1 %.not, float %1351, float %1358, !dbg !11
  %1360 = select i1 %.not, float %1358, float %1352, !dbg !11
  %1361 = select i1 %.not, float %1354, float %1353, !dbg !11
  %1362 = bitcast float %1361 to i32, !dbg !11
  %1363 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1362), !dbg !11
  %1364 = bitcast i32 %1363 to float, !dbg !11
  %1365 = select i1 %.not, float %1353, float %1364, !dbg !11
  %1366 = select i1 %.not, float %1364, float %1354, !dbg !11
  %1367 = select i1 %.not36, float %1365, float %1359, !dbg !11
  %1368 = bitcast float %1367 to i32, !dbg !11
  %1369 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1368), !dbg !11
  %1370 = bitcast i32 %1369 to float, !dbg !11
  %1371 = select i1 %.not36, float %1366, float %1360, !dbg !11
  %1372 = bitcast float %1371 to i32, !dbg !11
  %1373 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1372), !dbg !11
  %1374 = bitcast i32 %1373 to float, !dbg !11
  %1375 = insertelement <4 x float> poison, float %1359, i64 0, !dbg !11
  %1376 = insertelement <4 x float> %1375, float %1360, i64 1, !dbg !11
  %1377 = insertelement <4 x float> %1376, float %1370, i64 2, !dbg !11
  %1378 = insertelement <4 x float> %1377, float %1374, i64 3, !dbg !11
  %1379 = insertelement <4 x float> poison, float %1370, i64 0, !dbg !11
  %1380 = insertelement <4 x float> %1379, float %1374, i64 1, !dbg !11
  %1381 = insertelement <4 x float> %1380, float %1365, i64 2, !dbg !11
  %1382 = insertelement <4 x float> %1381, float %1366, i64 3, !dbg !11
  %1383 = select <4 x i1> %1215, <4 x float> %1378, <4 x float> %1382, !dbg !11
  %1384 = extractelement <8 x float> %1259, i64 4, !dbg !11
  %1385 = extractelement <8 x float> %1259, i64 5, !dbg !11
  %1386 = extractelement <8 x float> %1259, i64 6, !dbg !11
  %1387 = extractelement <8 x float> %1259, i64 7, !dbg !11
  %1388 = select i1 %.not, float %1385, float %1384, !dbg !11
  %1389 = bitcast float %1388 to i32, !dbg !11
  %1390 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1389), !dbg !11
  %1391 = bitcast i32 %1390 to float, !dbg !11
  %1392 = select i1 %.not, float %1384, float %1391, !dbg !11
  %1393 = select i1 %.not, float %1391, float %1385, !dbg !11
  %1394 = select i1 %.not, float %1387, float %1386, !dbg !11
  %1395 = bitcast float %1394 to i32, !dbg !11
  %1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1395), !dbg !11
  %1397 = bitcast i32 %1396 to float, !dbg !11
  %1398 = select i1 %.not, float %1386, float %1397, !dbg !11
  %1399 = select i1 %.not, float %1397, float %1387, !dbg !11
  %1400 = select i1 %.not36, float %1398, float %1392, !dbg !11
  %1401 = bitcast float %1400 to i32, !dbg !11
  %1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1401), !dbg !11
  %1403 = bitcast i32 %1402 to float, !dbg !11
  %1404 = select i1 %.not36, float %1399, float %1393, !dbg !11
  %1405 = bitcast float %1404 to i32, !dbg !11
  %1406 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1405), !dbg !11
  %1407 = bitcast i32 %1406 to float, !dbg !11
  %1408 = insertelement <4 x float> poison, float %1392, i64 0, !dbg !11
  %1409 = insertelement <4 x float> %1408, float %1393, i64 1, !dbg !11
  %1410 = insertelement <4 x float> %1409, float %1403, i64 2, !dbg !11
  %1411 = insertelement <4 x float> %1410, float %1407, i64 3, !dbg !11
  %1412 = insertelement <4 x float> poison, float %1403, i64 0, !dbg !11
  %1413 = insertelement <4 x float> %1412, float %1407, i64 1, !dbg !11
  %1414 = insertelement <4 x float> %1413, float %1398, i64 2, !dbg !11
  %1415 = insertelement <4 x float> %1414, float %1399, i64 3, !dbg !11
  %1416 = select <4 x i1> %1215, <4 x float> %1411, <4 x float> %1415, !dbg !11
  %1417 = extractelement <8 x float> %1293, i64 4, !dbg !11
  %1418 = extractelement <8 x float> %1293, i64 5, !dbg !11
  %1419 = extractelement <8 x float> %1293, i64 6, !dbg !11
  %1420 = extractelement <8 x float> %1293, i64 7, !dbg !11
  %1421 = select i1 %.not, float %1418, float %1417, !dbg !11
  %1422 = bitcast float %1421 to i32, !dbg !11
  %1423 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1422), !dbg !11
  %1424 = bitcast i32 %1423 to float, !dbg !11
  %1425 = select i1 %.not, float %1417, float %1424, !dbg !11
  %1426 = select i1 %.not, float %1424, float %1418, !dbg !11
  %1427 = select i1 %.not, float %1420, float %1419, !dbg !11
  %1428 = bitcast float %1427 to i32, !dbg !11
  %1429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1428), !dbg !11
  %1430 = bitcast i32 %1429 to float, !dbg !11
  %1431 = select i1 %.not, float %1419, float %1430, !dbg !11
  %1432 = select i1 %.not, float %1430, float %1420, !dbg !11
  %1433 = select i1 %.not36, float %1431, float %1425, !dbg !11
  %1434 = bitcast float %1433 to i32, !dbg !11
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1434), !dbg !11
  %1436 = select i1 %.not36, float %1432, float %1426, !dbg !11
  %1437 = bitcast float %1436 to i32, !dbg !11
  %1438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1437), !dbg !11
  %1439 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !11
  %1440 = extractelement <8 x float> %1439, i64 0, !dbg !11
  %1441 = extractelement <8 x float> %1439, i64 1, !dbg !11
  %1442 = extractelement <8 x float> %1439, i64 2, !dbg !11
  %1443 = extractelement <8 x float> %1439, i64 3, !dbg !11
  %1444 = select i1 %.not, float %1441, float %1440, !dbg !11
  %1445 = bitcast float %1444 to i32, !dbg !11
  %1446 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1445), !dbg !11
  %1447 = bitcast i32 %1446 to float, !dbg !11
  %1448 = select i1 %.not, float %1440, float %1447, !dbg !11
  %1449 = select i1 %.not, float %1447, float %1441, !dbg !11
  %1450 = select i1 %.not, float %1443, float %1442, !dbg !11
  %1451 = bitcast float %1450 to i32, !dbg !11
  %1452 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1451), !dbg !11
  %1453 = bitcast i32 %1452 to float, !dbg !11
  %1454 = select i1 %.not, float %1442, float %1453, !dbg !11
  %1455 = select i1 %.not, float %1453, float %1443, !dbg !11
  %1456 = select i1 %.not36, float %1454, float %1448, !dbg !11
  %1457 = bitcast float %1456 to i32, !dbg !11
  %1458 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1457), !dbg !11
  %1459 = bitcast i32 %1458 to float, !dbg !11
  %1460 = select i1 %.not36, float %1455, float %1449, !dbg !11
  %1461 = bitcast float %1460 to i32, !dbg !11
  %1462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1461), !dbg !11
  %1463 = bitcast i32 %1462 to float, !dbg !11
  %1464 = insertelement <4 x float> poison, float %1448, i64 0, !dbg !11
  %1465 = insertelement <4 x float> %1464, float %1449, i64 1, !dbg !11
  %1466 = insertelement <4 x float> %1465, float %1459, i64 2, !dbg !11
  %1467 = insertelement <4 x float> %1466, float %1463, i64 3, !dbg !11
  %1468 = insertelement <4 x float> poison, float %1459, i64 0, !dbg !11
  %1469 = insertelement <4 x float> %1468, float %1463, i64 1, !dbg !11
  %1470 = insertelement <4 x float> %1469, float %1454, i64 2, !dbg !11
  %1471 = insertelement <4 x float> %1470, float %1455, i64 3, !dbg !11
  %1472 = select <4 x i1> %1215, <4 x float> %1467, <4 x float> %1471, !dbg !11
  %1473 = extractvalue [2 x <8 x float>] %.lcssa93, 1, !dbg !11
  %1474 = extractelement <8 x float> %1473, i64 0, !dbg !11
  %1475 = extractelement <8 x float> %1473, i64 1, !dbg !11
  %1476 = extractelement <8 x float> %1473, i64 2, !dbg !11
  %1477 = extractelement <8 x float> %1473, i64 3, !dbg !11
  %1478 = select i1 %.not, float %1475, float %1474, !dbg !11
  %1479 = bitcast float %1478 to i32, !dbg !11
  %1480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1479), !dbg !11
  %1481 = bitcast i32 %1480 to float, !dbg !11
  %1482 = select i1 %.not, float %1474, float %1481, !dbg !11
  %1483 = select i1 %.not, float %1481, float %1475, !dbg !11
  %1484 = select i1 %.not, float %1477, float %1476, !dbg !11
  %1485 = bitcast float %1484 to i32, !dbg !11
  %1486 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1485), !dbg !11
  %1487 = bitcast i32 %1486 to float, !dbg !11
  %1488 = select i1 %.not, float %1476, float %1487, !dbg !11
  %1489 = select i1 %.not, float %1487, float %1477, !dbg !11
  %1490 = select i1 %.not36, float %1488, float %1482, !dbg !11
  %1491 = bitcast float %1490 to i32, !dbg !11
  %1492 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1491), !dbg !11
  %1493 = bitcast i32 %1492 to float, !dbg !11
  %1494 = select i1 %.not36, float %1489, float %1483, !dbg !11
  %1495 = bitcast float %1494 to i32, !dbg !11
  %1496 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1495), !dbg !11
  %1497 = bitcast i32 %1496 to float, !dbg !11
  %1498 = insertelement <4 x float> poison, float %1482, i64 0, !dbg !11
  %1499 = insertelement <4 x float> %1498, float %1483, i64 1, !dbg !11
  %1500 = insertelement <4 x float> %1499, float %1493, i64 2, !dbg !11
  %1501 = insertelement <4 x float> %1500, float %1497, i64 3, !dbg !11
  %1502 = insertelement <4 x float> poison, float %1493, i64 0, !dbg !11
  %1503 = insertelement <4 x float> %1502, float %1497, i64 1, !dbg !11
  %1504 = insertelement <4 x float> %1503, float %1488, i64 2, !dbg !11
  %1505 = insertelement <4 x float> %1504, float %1489, i64 3, !dbg !11
  %1506 = select <4 x i1> %1215, <4 x float> %1501, <4 x float> %1505, !dbg !11
  %1507 = extractvalue [2 x <8 x float>] %.lcssa94, 1, !dbg !11
  %1508 = extractelement <8 x float> %1507, i64 0, !dbg !11
  %1509 = extractelement <8 x float> %1507, i64 1, !dbg !11
  %1510 = extractelement <8 x float> %1507, i64 2, !dbg !11
  %1511 = extractelement <8 x float> %1507, i64 3, !dbg !11
  %1512 = select i1 %.not, float %1509, float %1508, !dbg !11
  %1513 = bitcast float %1512 to i32, !dbg !11
  %1514 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1513), !dbg !11
  %1515 = bitcast i32 %1514 to float, !dbg !11
  %1516 = select i1 %.not, float %1508, float %1515, !dbg !11
  %1517 = select i1 %.not, float %1515, float %1509, !dbg !11
  %1518 = select i1 %.not, float %1511, float %1510, !dbg !11
  %1519 = bitcast float %1518 to i32, !dbg !11
  %1520 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1519), !dbg !11
  %1521 = bitcast i32 %1520 to float, !dbg !11
  %1522 = select i1 %.not, float %1510, float %1521, !dbg !11
  %1523 = select i1 %.not, float %1521, float %1511, !dbg !11
  %1524 = select i1 %.not36, float %1522, float %1516, !dbg !11
  %1525 = bitcast float %1524 to i32, !dbg !11
  %1526 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1525), !dbg !11
  %1527 = bitcast i32 %1526 to float, !dbg !11
  %1528 = select i1 %.not36, float %1523, float %1517, !dbg !11
  %1529 = bitcast float %1528 to i32, !dbg !11
  %1530 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1529), !dbg !11
  %1531 = bitcast i32 %1530 to float, !dbg !11
  %1532 = insertelement <4 x float> poison, float %1516, i64 0, !dbg !11
  %1533 = insertelement <4 x float> %1532, float %1517, i64 1, !dbg !11
  %1534 = insertelement <4 x float> %1533, float %1527, i64 2, !dbg !11
  %1535 = insertelement <4 x float> %1534, float %1531, i64 3, !dbg !11
  %1536 = insertelement <4 x float> poison, float %1527, i64 0, !dbg !11
  %1537 = insertelement <4 x float> %1536, float %1531, i64 1, !dbg !11
  %1538 = insertelement <4 x float> %1537, float %1522, i64 2, !dbg !11
  %1539 = insertelement <4 x float> %1538, float %1523, i64 3, !dbg !11
  %1540 = select <4 x i1> %1215, <4 x float> %1535, <4 x float> %1539, !dbg !11
  %1541 = extractvalue [2 x <8 x float>] %.lcssa96, 1, !dbg !11
  %1542 = extractelement <8 x float> %1541, i64 0, !dbg !11
  %1543 = extractelement <8 x float> %1541, i64 1, !dbg !11
  %1544 = extractelement <8 x float> %1541, i64 2, !dbg !11
  %1545 = extractelement <8 x float> %1541, i64 3, !dbg !11
  %1546 = select i1 %.not, float %1543, float %1542, !dbg !11
  %1547 = bitcast float %1546 to i32, !dbg !11
  %1548 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1547), !dbg !11
  %1549 = bitcast i32 %1548 to float, !dbg !11
  %1550 = select i1 %.not, float %1542, float %1549, !dbg !11
  %1551 = select i1 %.not, float %1549, float %1543, !dbg !11
  %1552 = select i1 %.not, float %1545, float %1544, !dbg !11
  %1553 = bitcast float %1552 to i32, !dbg !11
  %1554 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1553), !dbg !11
  %1555 = bitcast i32 %1554 to float, !dbg !11
  %1556 = select i1 %.not, float %1544, float %1555, !dbg !11
  %1557 = select i1 %.not, float %1555, float %1545, !dbg !11
  %1558 = select i1 %.not36, float %1556, float %1550, !dbg !11
  %1559 = bitcast float %1558 to i32, !dbg !11
  %1560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1559), !dbg !11
  %1561 = select i1 %.not36, float %1557, float %1551, !dbg !11
  %1562 = bitcast float %1561 to i32, !dbg !11
  %1563 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1562), !dbg !11
  %1564 = extractvalue [2 x <1 x float>] %.lcssa95, 1, !dbg !25
  %1565 = shufflevector <1 x float> %1564, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %1566 = extractelement <8 x float> %1439, i64 4, !dbg !11
  %1567 = extractelement <8 x float> %1439, i64 5, !dbg !11
  %1568 = extractelement <8 x float> %1439, i64 6, !dbg !11
  %1569 = extractelement <8 x float> %1439, i64 7, !dbg !11
  %1570 = select i1 %.not, float %1567, float %1566, !dbg !11
  %1571 = bitcast float %1570 to i32, !dbg !11
  %1572 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1571), !dbg !11
  %1573 = bitcast i32 %1572 to float, !dbg !11
  %1574 = select i1 %.not, float %1566, float %1573, !dbg !11
  %1575 = select i1 %.not, float %1573, float %1567, !dbg !11
  %1576 = select i1 %.not, float %1569, float %1568, !dbg !11
  %1577 = bitcast float %1576 to i32, !dbg !11
  %1578 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1577), !dbg !11
  %1579 = bitcast i32 %1578 to float, !dbg !11
  %1580 = select i1 %.not, float %1568, float %1579, !dbg !11
  %1581 = select i1 %.not, float %1579, float %1569, !dbg !11
  %1582 = select i1 %.not36, float %1580, float %1574, !dbg !11
  %1583 = bitcast float %1582 to i32, !dbg !11
  %1584 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1583), !dbg !11
  %1585 = bitcast i32 %1584 to float, !dbg !11
  %1586 = select i1 %.not36, float %1581, float %1575, !dbg !11
  %1587 = bitcast float %1586 to i32, !dbg !11
  %1588 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1587), !dbg !11
  %1589 = bitcast i32 %1588 to float, !dbg !11
  %1590 = insertelement <4 x float> poison, float %1574, i64 0, !dbg !11
  %1591 = insertelement <4 x float> %1590, float %1575, i64 1, !dbg !11
  %1592 = insertelement <4 x float> %1591, float %1585, i64 2, !dbg !11
  %1593 = insertelement <4 x float> %1592, float %1589, i64 3, !dbg !11
  %1594 = insertelement <4 x float> poison, float %1585, i64 0, !dbg !11
  %1595 = insertelement <4 x float> %1594, float %1589, i64 1, !dbg !11
  %1596 = insertelement <4 x float> %1595, float %1580, i64 2, !dbg !11
  %1597 = insertelement <4 x float> %1596, float %1581, i64 3, !dbg !11
  %1598 = select <4 x i1> %1215, <4 x float> %1593, <4 x float> %1597, !dbg !11
  %1599 = extractelement <8 x float> %1473, i64 4, !dbg !11
  %1600 = extractelement <8 x float> %1473, i64 5, !dbg !11
  %1601 = extractelement <8 x float> %1473, i64 6, !dbg !11
  %1602 = extractelement <8 x float> %1473, i64 7, !dbg !11
  %1603 = select i1 %.not, float %1600, float %1599, !dbg !11
  %1604 = bitcast float %1603 to i32, !dbg !11
  %1605 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1604), !dbg !11
  %1606 = bitcast i32 %1605 to float, !dbg !11
  %1607 = select i1 %.not, float %1599, float %1606, !dbg !11
  %1608 = select i1 %.not, float %1606, float %1600, !dbg !11
  %1609 = select i1 %.not, float %1602, float %1601, !dbg !11
  %1610 = bitcast float %1609 to i32, !dbg !11
  %1611 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1610), !dbg !11
  %1612 = bitcast i32 %1611 to float, !dbg !11
  %1613 = select i1 %.not, float %1601, float %1612, !dbg !11
  %1614 = select i1 %.not, float %1612, float %1602, !dbg !11
  %1615 = select i1 %.not36, float %1613, float %1607, !dbg !11
  %1616 = bitcast float %1615 to i32, !dbg !11
  %1617 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1616), !dbg !11
  %1618 = bitcast i32 %1617 to float, !dbg !11
  %1619 = select i1 %.not36, float %1614, float %1608, !dbg !11
  %1620 = bitcast float %1619 to i32, !dbg !11
  %1621 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1620), !dbg !11
  %1622 = bitcast i32 %1621 to float, !dbg !11
  %1623 = insertelement <4 x float> poison, float %1607, i64 0, !dbg !11
  %1624 = insertelement <4 x float> %1623, float %1608, i64 1, !dbg !11
  %1625 = insertelement <4 x float> %1624, float %1618, i64 2, !dbg !11
  %1626 = insertelement <4 x float> %1625, float %1622, i64 3, !dbg !11
  %1627 = insertelement <4 x float> poison, float %1618, i64 0, !dbg !11
  %1628 = insertelement <4 x float> %1627, float %1622, i64 1, !dbg !11
  %1629 = insertelement <4 x float> %1628, float %1613, i64 2, !dbg !11
  %1630 = insertelement <4 x float> %1629, float %1614, i64 3, !dbg !11
  %1631 = select <4 x i1> %1215, <4 x float> %1626, <4 x float> %1630, !dbg !11
  %1632 = extractelement <8 x float> %1507, i64 4, !dbg !11
  %1633 = extractelement <8 x float> %1507, i64 5, !dbg !11
  %1634 = extractelement <8 x float> %1507, i64 6, !dbg !11
  %1635 = extractelement <8 x float> %1507, i64 7, !dbg !11
  %1636 = select i1 %.not, float %1633, float %1632, !dbg !11
  %1637 = bitcast float %1636 to i32, !dbg !11
  %1638 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1637), !dbg !11
  %1639 = bitcast i32 %1638 to float, !dbg !11
  %1640 = select i1 %.not, float %1632, float %1639, !dbg !11
  %1641 = select i1 %.not, float %1639, float %1633, !dbg !11
  %1642 = select i1 %.not, float %1635, float %1634, !dbg !11
  %1643 = bitcast float %1642 to i32, !dbg !11
  %1644 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1643), !dbg !11
  %1645 = bitcast i32 %1644 to float, !dbg !11
  %1646 = select i1 %.not, float %1634, float %1645, !dbg !11
  %1647 = select i1 %.not, float %1645, float %1635, !dbg !11
  %1648 = select i1 %.not36, float %1646, float %1640, !dbg !11
  %1649 = bitcast float %1648 to i32, !dbg !11
  %1650 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1649), !dbg !11
  %1651 = bitcast i32 %1650 to float, !dbg !11
  %1652 = select i1 %.not36, float %1647, float %1641, !dbg !11
  %1653 = bitcast float %1652 to i32, !dbg !11
  %1654 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1653), !dbg !11
  %1655 = bitcast i32 %1654 to float, !dbg !11
  %1656 = insertelement <4 x float> poison, float %1640, i64 0, !dbg !11
  %1657 = insertelement <4 x float> %1656, float %1641, i64 1, !dbg !11
  %1658 = insertelement <4 x float> %1657, float %1651, i64 2, !dbg !11
  %1659 = insertelement <4 x float> %1658, float %1655, i64 3, !dbg !11
  %1660 = insertelement <4 x float> poison, float %1651, i64 0, !dbg !11
  %1661 = insertelement <4 x float> %1660, float %1655, i64 1, !dbg !11
  %1662 = insertelement <4 x float> %1661, float %1646, i64 2, !dbg !11
  %1663 = insertelement <4 x float> %1662, float %1647, i64 3, !dbg !11
  %1664 = select <4 x i1> %1215, <4 x float> %1659, <4 x float> %1663, !dbg !11
  %1665 = extractelement <8 x float> %1541, i64 4, !dbg !11
  %1666 = extractelement <8 x float> %1541, i64 5, !dbg !11
  %1667 = extractelement <8 x float> %1541, i64 6, !dbg !11
  %1668 = extractelement <8 x float> %1541, i64 7, !dbg !11
  %1669 = select i1 %.not, float %1666, float %1665, !dbg !11
  %1670 = bitcast float %1669 to i32, !dbg !11
  %1671 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1670), !dbg !11
  %1672 = bitcast i32 %1671 to float, !dbg !11
  %1673 = select i1 %.not, float %1665, float %1672, !dbg !11
  %1674 = select i1 %.not, float %1672, float %1666, !dbg !11
  %1675 = select i1 %.not, float %1668, float %1667, !dbg !11
  %1676 = bitcast float %1675 to i32, !dbg !11
  %1677 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1676), !dbg !11
  %1678 = bitcast i32 %1677 to float, !dbg !11
  %1679 = select i1 %.not, float %1667, float %1678, !dbg !11
  %1680 = select i1 %.not, float %1678, float %1668, !dbg !11
  %1681 = select i1 %.not36, float %1679, float %1673, !dbg !11
  %1682 = bitcast float %1681 to i32, !dbg !11
  %1683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1682), !dbg !11
  %1684 = select i1 %.not36, float %1680, float %1674, !dbg !11
  %1685 = bitcast float %1684 to i32, !dbg !11
  %1686 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1206, i32 %1685), !dbg !11
  %1687 = bitcast i32 %1683 to float, !dbg !11
  %1688 = bitcast i32 %1686 to float, !dbg !11
  %1689 = insertelement <4 x float> poison, float %1673, i64 0, !dbg !11
  %1690 = insertelement <4 x float> %1689, float %1674, i64 1, !dbg !11
  %1691 = insertelement <4 x float> %1690, float %1687, i64 2, !dbg !11
  %1692 = insertelement <4 x float> %1691, float %1688, i64 3, !dbg !11
  %1693 = insertelement <4 x float> poison, float %1687, i64 0, !dbg !11
  %1694 = insertelement <4 x float> %1693, float %1688, i64 1, !dbg !11
  %1695 = insertelement <4 x float> %1694, float %1679, i64 2, !dbg !11
  %1696 = insertelement <4 x float> %1695, float %1680, i64 3, !dbg !11
  %1697 = select <4 x i1> %1215, <4 x float> %1692, <4 x float> %1696, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1565, !dbg !25
  %normalized = fmul <4 x float> %row.reciprocal, %1697, !dbg !25
  %1698 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !26
  %1699 = extractelement <4 x half> %1698, i64 3, !dbg !26
  %1700 = extractelement <4 x half> %1698, i64 2, !dbg !26
  %1701 = extractelement <4 x half> %1698, i64 1, !dbg !26
  %1702 = extractelement <4 x half> %1698, i64 0, !dbg !26
  %1703 = bitcast i32 %1560 to float, !dbg !11
  %1704 = bitcast i32 %1563 to float, !dbg !11
  %1705 = insertelement <4 x float> poison, float %1550, i64 0, !dbg !11
  %1706 = insertelement <4 x float> %1705, float %1551, i64 1, !dbg !11
  %1707 = insertelement <4 x float> %1706, float %1703, i64 2, !dbg !11
  %1708 = insertelement <4 x float> %1707, float %1704, i64 3, !dbg !11
  %1709 = insertelement <4 x float> poison, float %1703, i64 0, !dbg !11
  %1710 = insertelement <4 x float> %1709, float %1704, i64 1, !dbg !11
  %1711 = insertelement <4 x float> %1710, float %1556, i64 2, !dbg !11
  %1712 = insertelement <4 x float> %1711, float %1557, i64 3, !dbg !11
  %1713 = select <4 x i1> %1215, <4 x float> %1708, <4 x float> %1712, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %1713, !dbg !25
  %1714 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !26
  %1715 = extractelement <4 x half> %1714, i64 3, !dbg !26
  %1716 = extractelement <4 x half> %1714, i64 2, !dbg !26
  %1717 = extractelement <4 x half> %1714, i64 1, !dbg !26
  %1718 = extractelement <4 x half> %1714, i64 0, !dbg !26
  %1719 = bitcast i32 %1435 to float, !dbg !11
  %1720 = bitcast i32 %1438 to float, !dbg !11
  %1721 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !11
  %1722 = insertelement <4 x float> %1721, float %1426, i64 1, !dbg !11
  %1723 = insertelement <4 x float> %1722, float %1719, i64 2, !dbg !11
  %1724 = insertelement <4 x float> %1723, float %1720, i64 3, !dbg !11
  %1725 = insertelement <4 x float> poison, float %1719, i64 0, !dbg !11
  %1726 = insertelement <4 x float> %1725, float %1720, i64 1, !dbg !11
  %1727 = insertelement <4 x float> %1726, float %1431, i64 2, !dbg !11
  %1728 = insertelement <4 x float> %1727, float %1432, i64 3, !dbg !11
  %1729 = select <4 x i1> %1215, <4 x float> %1724, <4 x float> %1728, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1317, !dbg !25
  %normalized106 = fmul <4 x float> %row.reciprocal105, %1729, !dbg !25
  %1730 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !26
  %1731 = extractelement <4 x half> %1730, i64 3, !dbg !26
  %1732 = extractelement <4 x half> %1730, i64 2, !dbg !26
  %1733 = extractelement <4 x half> %1730, i64 1, !dbg !26
  %1734 = extractelement <4 x half> %1730, i64 0, !dbg !26
  %1735 = bitcast i32 %1312 to float, !dbg !11
  %1736 = bitcast i32 %1315 to float, !dbg !11
  %1737 = insertelement <4 x float> poison, float %1302, i64 0, !dbg !11
  %1738 = insertelement <4 x float> %1737, float %1303, i64 1, !dbg !11
  %1739 = insertelement <4 x float> %1738, float %1735, i64 2, !dbg !11
  %1740 = insertelement <4 x float> %1739, float %1736, i64 3, !dbg !11
  %1741 = insertelement <4 x float> poison, float %1735, i64 0, !dbg !11
  %1742 = insertelement <4 x float> %1741, float %1736, i64 1, !dbg !11
  %1743 = insertelement <4 x float> %1742, float %1308, i64 2, !dbg !11
  %1744 = insertelement <4 x float> %1743, float %1309, i64 3, !dbg !11
  %1745 = select <4 x i1> %1215, <4 x float> %1740, <4 x float> %1744, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %1745, !dbg !25
  %1746 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !26
  %1747 = extractelement <4 x half> %1746, i64 3, !dbg !26
  %1748 = extractelement <4 x half> %1746, i64 2, !dbg !26
  %1749 = extractelement <4 x half> %1746, i64 1, !dbg !26
  %1750 = extractelement <4 x half> %1746, i64 0, !dbg !26
  %address.thread914 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !27, !range !9, !frisk.register_lifetimes !5
  %address.local915 = zext nneg i32 %address.thread914 to i64, !dbg !8
  %1751 = lshr i64 %address.local915, 2, !dbg !27
  %address.local918 = and i64 %1751, 48, !dbg !27
  %address.local920 = and i64 %address.local915, 240, !dbg !27
  %address.local919 = add i64 %26, %address.local915, !dbg !27
  %address.local921 = sub i64 %address.local919, %address.local920, !dbg !27
  %address.local922 = add i64 %address.local921, %address.local918, !dbg !27
  %1752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !27
  %address.local925.idx = shl nuw nsw i64 %address.local922, 8, !dbg !27
  %address.local925 = getelementptr inbounds nuw i8, ptr addrspace(1) %1752, i64 %address.local925.idx, !dbg !27
  %address.local927 = and i64 %1751, 12, !dbg !10
  %address.local928 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local927, !dbg !27
  store half %1750, ptr addrspace(1) %address.local928, align 2, !dbg !27
  %address.local930 = or disjoint i64 %address.local927, 1, !dbg !27
  %address.local931 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local930, !dbg !27
  store half %1749, ptr addrspace(1) %address.local931, align 2, !dbg !27
  %address.local933 = or disjoint i64 %address.local927, 2, !dbg !27
  %address.local934 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local933, !dbg !27
  store half %1748, ptr addrspace(1) %address.local934, align 2, !dbg !27
  %address.local936 = or disjoint i64 %address.local927, 3, !dbg !27
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local936, !dbg !27
  store half %1747, ptr addrspace(1) %address.local937, align 2, !dbg !27
  %address.local939 = or disjoint i64 %address.local927, 16, !dbg !27
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local939, !dbg !27
  store half %1734, ptr addrspace(1) %address.local940, align 2, !dbg !27
  %address.local942 = or disjoint i64 %address.local927, 17, !dbg !27
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local942, !dbg !27
  store half %1733, ptr addrspace(1) %address.local943, align 2, !dbg !27
  %address.local945 = or disjoint i64 %address.local927, 18, !dbg !27
  %address.local946 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local945, !dbg !27
  store half %1732, ptr addrspace(1) %address.local946, align 2, !dbg !27
  %address.local948 = or disjoint i64 %address.local927, 19, !dbg !27
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local948, !dbg !27
  store half %1731, ptr addrspace(1) %address.local949, align 2, !dbg !27
  %address.local953 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local925, i64 16384, !dbg !27
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local927, !dbg !27
  store half %1718, ptr addrspace(1) %address.local954, align 2, !dbg !27
  %address.local956 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local930, !dbg !27
  store half %1717, ptr addrspace(1) %address.local956, align 2, !dbg !27
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local933, !dbg !27
  store half %1716, ptr addrspace(1) %address.local958, align 2, !dbg !27
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local936, !dbg !27
  store half %1715, ptr addrspace(1) %address.local960, align 2, !dbg !27
  %address.local962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local939, !dbg !27
  store half %1702, ptr addrspace(1) %address.local962, align 2, !dbg !27
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local942, !dbg !27
  store half %1701, ptr addrspace(1) %address.local964, align 2, !dbg !27
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local945, !dbg !27
  store half %1700, ptr addrspace(1) %address.local966, align 2, !dbg !27
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local948, !dbg !27
  store half %1699, ptr addrspace(1) %address.local968, align 2, !dbg !27
  %normalized109 = fmul <4 x float> %row.reciprocal, %1664, !dbg !25
  %1753 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !26
  %1754 = extractelement <4 x half> %1753, i64 3, !dbg !26
  %1755 = extractelement <4 x half> %1753, i64 2, !dbg !26
  %1756 = extractelement <4 x half> %1753, i64 1, !dbg !26
  %1757 = extractelement <4 x half> %1753, i64 0, !dbg !26
  %normalized110 = fmul <4 x float> %row.reciprocal, %1540, !dbg !25
  %1758 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !26
  %1759 = extractelement <4 x half> %1758, i64 3, !dbg !26
  %1760 = extractelement <4 x half> %1758, i64 2, !dbg !26
  %1761 = extractelement <4 x half> %1758, i64 1, !dbg !26
  %1762 = extractelement <4 x half> %1758, i64 0, !dbg !26
  %normalized112 = fmul <4 x float> %row.reciprocal105, %1416, !dbg !25
  %1763 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !26
  %1764 = extractelement <4 x half> %1763, i64 3, !dbg !26
  %1765 = extractelement <4 x half> %1763, i64 2, !dbg !26
  %1766 = extractelement <4 x half> %1763, i64 1, !dbg !26
  %1767 = extractelement <4 x half> %1763, i64 0, !dbg !26
  %normalized113 = fmul <4 x float> %row.reciprocal105, %1292, !dbg !25
  %1768 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !26
  %1769 = extractelement <4 x half> %1768, i64 3, !dbg !26
  %1770 = extractelement <4 x half> %1768, i64 2, !dbg !26
  %1771 = extractelement <4 x half> %1768, i64 1, !dbg !26
  %1772 = extractelement <4 x half> %1768, i64 0, !dbg !26
  %address.local983 = or disjoint i64 %address.local927, 32, !dbg !27
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local983, !dbg !27
  store half %1772, ptr addrspace(1) %address.local984, align 2, !dbg !27
  %address.local986 = or disjoint i64 %address.local927, 33, !dbg !27
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local986, !dbg !27
  store half %1771, ptr addrspace(1) %address.local987, align 2, !dbg !27
  %address.local989 = or disjoint i64 %address.local927, 34, !dbg !27
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local989, !dbg !27
  store half %1770, ptr addrspace(1) %address.local990, align 2, !dbg !27
  %address.local992 = or disjoint i64 %address.local927, 35, !dbg !27
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local992, !dbg !27
  store half %1769, ptr addrspace(1) %address.local993, align 2, !dbg !27
  %address.local995 = or disjoint i64 %address.local927, 48, !dbg !27
  %address.local996 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local995, !dbg !27
  store half %1767, ptr addrspace(1) %address.local996, align 2, !dbg !27
  %address.local998 = or disjoint i64 %address.local927, 49, !dbg !27
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local998, !dbg !27
  store half %1766, ptr addrspace(1) %address.local999, align 2, !dbg !27
  %address.local1001 = or disjoint i64 %address.local927, 50, !dbg !27
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1001, !dbg !27
  store half %1765, ptr addrspace(1) %address.local1002, align 2, !dbg !27
  %address.local1004 = or i64 %1751, 51, !dbg !27
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1004, !dbg !27
  store half %1764, ptr addrspace(1) %address.local1005, align 2, !dbg !27
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local983, !dbg !27
  store half %1762, ptr addrspace(1) %address.local1010, align 2, !dbg !27
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local986, !dbg !27
  store half %1761, ptr addrspace(1) %address.local1012, align 2, !dbg !27
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local989, !dbg !27
  store half %1760, ptr addrspace(1) %address.local1014, align 2, !dbg !27
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local992, !dbg !27
  store half %1759, ptr addrspace(1) %address.local1016, align 2, !dbg !27
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local995, !dbg !27
  store half %1757, ptr addrspace(1) %address.local1018, align 2, !dbg !27
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local998, !dbg !27
  store half %1756, ptr addrspace(1) %address.local1020, align 2, !dbg !27
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1001, !dbg !27
  store half %1755, ptr addrspace(1) %address.local1022, align 2, !dbg !27
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1004, !dbg !27
  store half %1754, ptr addrspace(1) %address.local1024, align 2, !dbg !27
  %normalized115 = fmul <4 x float> %row.reciprocal, %1631, !dbg !25
  %1773 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !26
  %1774 = extractelement <4 x half> %1773, i64 3, !dbg !26
  %1775 = extractelement <4 x half> %1773, i64 2, !dbg !26
  %1776 = extractelement <4 x half> %1773, i64 1, !dbg !26
  %1777 = extractelement <4 x half> %1773, i64 0, !dbg !26
  %normalized116 = fmul <4 x float> %row.reciprocal, %1506, !dbg !25
  %1778 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !26
  %1779 = extractelement <4 x half> %1778, i64 3, !dbg !26
  %1780 = extractelement <4 x half> %1778, i64 2, !dbg !26
  %1781 = extractelement <4 x half> %1778, i64 1, !dbg !26
  %1782 = extractelement <4 x half> %1778, i64 0, !dbg !26
  %normalized118 = fmul <4 x float> %row.reciprocal105, %1383, !dbg !25
  %1783 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !26
  %1784 = extractelement <4 x half> %1783, i64 3, !dbg !26
  %1785 = extractelement <4 x half> %1783, i64 2, !dbg !26
  %1786 = extractelement <4 x half> %1783, i64 1, !dbg !26
  %1787 = extractelement <4 x half> %1783, i64 0, !dbg !26
  %normalized119 = fmul <4 x float> %row.reciprocal105, %1258, !dbg !25
  %1788 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !26
  %1789 = extractelement <4 x half> %1788, i64 3, !dbg !26
  %1790 = extractelement <4 x half> %1788, i64 2, !dbg !26
  %1791 = extractelement <4 x half> %1788, i64 1, !dbg !26
  %1792 = extractelement <4 x half> %1788, i64 0, !dbg !26
  %address.local1039 = or disjoint i64 %address.local927, 64, !dbg !27
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1039, !dbg !27
  store half %1792, ptr addrspace(1) %address.local1040, align 2, !dbg !27
  %address.local1042 = or disjoint i64 %address.local927, 65, !dbg !27
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1042, !dbg !27
  store half %1791, ptr addrspace(1) %address.local1043, align 2, !dbg !27
  %address.local1045 = or disjoint i64 %address.local927, 66, !dbg !27
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1045, !dbg !27
  store half %1790, ptr addrspace(1) %address.local1046, align 2, !dbg !27
  %address.local1048 = or disjoint i64 %address.local927, 67, !dbg !27
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1048, !dbg !27
  store half %1789, ptr addrspace(1) %address.local1049, align 2, !dbg !27
  %address.local1051 = or disjoint i64 %address.local927, 80, !dbg !27
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1051, !dbg !27
  store half %1787, ptr addrspace(1) %address.local1052, align 2, !dbg !27
  %address.local1054 = or disjoint i64 %address.local927, 81, !dbg !27
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1054, !dbg !27
  store half %1786, ptr addrspace(1) %address.local1055, align 2, !dbg !27
  %address.local1057 = or disjoint i64 %address.local927, 82, !dbg !27
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1057, !dbg !27
  store half %1785, ptr addrspace(1) %address.local1058, align 2, !dbg !27
  %address.local1060 = or disjoint i64 %address.local927, 83, !dbg !27
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1060, !dbg !27
  store half %1784, ptr addrspace(1) %address.local1061, align 2, !dbg !27
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1039, !dbg !27
  store half %1782, ptr addrspace(1) %address.local1066, align 2, !dbg !27
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1042, !dbg !27
  store half %1781, ptr addrspace(1) %address.local1068, align 2, !dbg !27
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1045, !dbg !27
  store half %1780, ptr addrspace(1) %address.local1070, align 2, !dbg !27
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1048, !dbg !27
  store half %1779, ptr addrspace(1) %address.local1072, align 2, !dbg !27
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1051, !dbg !27
  store half %1777, ptr addrspace(1) %address.local1074, align 2, !dbg !27
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1054, !dbg !27
  store half %1776, ptr addrspace(1) %address.local1076, align 2, !dbg !27
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1057, !dbg !27
  store half %1775, ptr addrspace(1) %address.local1078, align 2, !dbg !27
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1060, !dbg !27
  store half %1774, ptr addrspace(1) %address.local1080, align 2, !dbg !27
  %normalized121 = fmul <4 x float> %row.reciprocal, %1598, !dbg !25
  %1793 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !26
  %1794 = extractelement <4 x half> %1793, i64 3, !dbg !26
  %1795 = extractelement <4 x half> %1793, i64 2, !dbg !26
  %1796 = extractelement <4 x half> %1793, i64 1, !dbg !26
  %1797 = extractelement <4 x half> %1793, i64 0, !dbg !26
  %normalized122 = fmul <4 x float> %row.reciprocal, %1472, !dbg !25
  %1798 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !26
  %1799 = extractelement <4 x half> %1798, i64 3, !dbg !26
  %1800 = extractelement <4 x half> %1798, i64 2, !dbg !26
  %1801 = extractelement <4 x half> %1798, i64 1, !dbg !26
  %1802 = extractelement <4 x half> %1798, i64 0, !dbg !26
  %normalized124 = fmul <4 x float> %row.reciprocal105, %1350, !dbg !25
  %1803 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !26
  %1804 = extractelement <4 x half> %1803, i64 3, !dbg !26
  %1805 = extractelement <4 x half> %1803, i64 2, !dbg !26
  %1806 = extractelement <4 x half> %1803, i64 1, !dbg !26
  %1807 = extractelement <4 x half> %1803, i64 0, !dbg !26
  %normalized125 = fmul <4 x float> %row.reciprocal105, %1224, !dbg !25
  %1808 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !26
  %1809 = extractelement <4 x half> %1808, i64 3, !dbg !26
  %1810 = extractelement <4 x half> %1808, i64 2, !dbg !26
  %1811 = extractelement <4 x half> %1808, i64 1, !dbg !26
  %1812 = extractelement <4 x half> %1808, i64 0, !dbg !26
  %address.local1095 = or disjoint i64 %address.local927, 96, !dbg !27
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1095, !dbg !27
  store half %1812, ptr addrspace(1) %address.local1096, align 2, !dbg !27
  %address.local1098 = or disjoint i64 %address.local927, 97, !dbg !27
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1098, !dbg !27
  store half %1811, ptr addrspace(1) %address.local1099, align 2, !dbg !27
  %address.local1101 = or disjoint i64 %address.local927, 98, !dbg !27
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1101, !dbg !27
  store half %1810, ptr addrspace(1) %address.local1102, align 2, !dbg !27
  %address.local1104 = or disjoint i64 %address.local927, 99, !dbg !27
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1104, !dbg !27
  store half %1809, ptr addrspace(1) %address.local1105, align 2, !dbg !27
  %address.local1107 = or disjoint i64 %address.local927, 112, !dbg !27
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1107, !dbg !27
  store half %1807, ptr addrspace(1) %address.local1108, align 2, !dbg !27
  %address.local1110 = or disjoint i64 %address.local927, 113, !dbg !27
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1110, !dbg !27
  store half %1806, ptr addrspace(1) %address.local1111, align 2, !dbg !27
  %address.local1113 = or disjoint i64 %address.local927, 114, !dbg !27
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1113, !dbg !27
  store half %1805, ptr addrspace(1) %address.local1114, align 2, !dbg !27
  %address.local1116 = or i64 %1751, 115, !dbg !27
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local1116, !dbg !27
  store half %1804, ptr addrspace(1) %address.local1117, align 2, !dbg !27
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1095, !dbg !27
  store half %1802, ptr addrspace(1) %address.local1122, align 2, !dbg !27
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1098, !dbg !27
  store half %1801, ptr addrspace(1) %address.local1124, align 2, !dbg !27
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1101, !dbg !27
  store half %1800, ptr addrspace(1) %address.local1126, align 2, !dbg !27
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1104, !dbg !27
  store half %1799, ptr addrspace(1) %address.local1128, align 2, !dbg !27
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1107, !dbg !27
  store half %1797, ptr addrspace(1) %address.local1130, align 2, !dbg !27
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1110, !dbg !27
  store half %1796, ptr addrspace(1) %address.local1132, align 2, !dbg !27
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1113, !dbg !27
  store half %1795, ptr addrspace(1) %address.local1134, align 2, !dbg !27
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local1116, !dbg !27
  store half %1794, ptr addrspace(1) %address.local1136, align 2, !dbg !27
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
!14 = !DILocation(line: 69, column: 26, scope: !3)
!15 = !DILocation(line: 67, column: 22, scope: !3)
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
