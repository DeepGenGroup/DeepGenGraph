; ModuleID = 'generated_ir/online_softmax/export/run-20261023-171017-a0mz8rfe/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/export/run-20261023-171017-a0mz8rfe/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/input.mlir"

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
  %103 = and i32 %29, 16, !dbg !13
  %.not = icmp eq i32 %103, 0, !dbg !13
  %104 = and i32 %29, 32, !dbg !13
  %.not1 = icmp eq i32 %104, 0, !dbg !13
  %105 = sub nsw i64 32, %14, !dbg !14
  %106 = shl i64 %16, 7, !dbg !15
  %107 = and i64 %106, 3968, !dbg !15
  %108 = shl nsw i64 %40, 12, !dbg !15
  %109 = or disjoint i64 %107, %108, !dbg !15
  %110 = shl nuw nsw i32 %29, 2, !dbg !16
  %111 = and i32 %110, 12, !dbg !16
  %112 = and i32 %29, 1, !dbg !16
  %113 = or disjoint i32 %111, %112, !dbg !16
  %114 = trunc nuw nsw i32 %113 to i16, !dbg !16
  %.lhs.trunc = add nuw nsw i16 %114, 1, !dbg !16
  %115 = lshr i16 %.lhs.trunc, 1, !dbg !16
  %.sext = zext nneg i16 %115 to i64, !dbg !16
  %116 = icmp ult i64 %10, 1024, !dbg !11
  br i1 %116, label %.lr.ph, label %._crit_edge, !dbg !11

.lr.ph:                                           ; preds = %4
  %117 = zext nneg i32 %29 to i64, !dbg !8
  %118 = ptrtoint ptr addrspace(1) %1 to i64
  %119 = insertelement <2 x i64> poison, i64 %118, i64 0
  %120 = bitcast <2 x i64> %119 to <4 x i32>
  %dma.buffer.rsrc1589 = shufflevector <4 x i32> %120, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %121 = ptrtoint ptr addrspace(1) %2 to i64
  %122 = insertelement <2 x i64> poison, i64 %121, i64 0
  %123 = bitcast <2 x i64> %122 to <4 x i32>
  %dma.buffer.rsrc1565 = shufflevector <4 x i32> %123, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %124 = lshr i64 %117, 2
  %125 = and i64 %124, 48
  %126 = and i32 %29, 240
  %index.compact = zext nneg i32 %126 to i64
  %127 = sub nsw i64 %117, %index.compact
  %128 = add nsw i64 %127, %125
  %129 = add nsw i64 %128, %24
  %130 = and i64 %117, 48
  %131 = add i64 %129, 1
  %132 = lshr i64 %131, 32
  %compare.high = trunc nuw i64 %132 to i32
  %compare.low.base = trunc i64 %131 to i32
  %133 = add i64 %129, 65
  %134 = lshr i64 %133, 32
  %compare.high1281 = trunc nuw i64 %134 to i32
  %compare.low.base1285 = trunc i64 %133 to i32
  %135 = insertelement <4 x i1> poison, i1 %.not1, i64 0, !dbg !13
  %136 = shufflevector <4 x i1> %135, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !13
  %137 = insertelement <32 x i32> poison, i32 %compare.low.base1285, i64 0, !dbg !17
  %138 = shufflevector <32 x i32> %137, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !17
  %139 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !17
  %140 = shufflevector <32 x i32> %139, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !17
  br label %141, !dbg !11

141:                                              ; preds = %.lr.ph, %540
  %142 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1928, %540 ]
  %143 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1714, %540 ]
  %144 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1495, %540 ]
  %145 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1057, %540 ]
  %146 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1279, %540 ]
  %147 = phi i64 [ 0, %.lr.ph ], [ %1929, %540 ]
  %148 = shl nuw nsw i64 %147, 7, !dbg !11
  %149 = lshr i64 %147, 12, !dbg !15
  %150 = add nuw nsw i64 %149, %16, !dbg !15
  %151 = lshr i64 %147, 5, !dbg !15
  %152 = and i64 %151, 127, !dbg !15
  %153 = shl i64 %150, 19, !dbg !15
  %154 = and i64 %153, 16252928, !dbg !15
  %155 = shl i64 %151, 19, !dbg !15
  %156 = shl nuw nsw i64 %147, 14, !dbg !15
  %157 = sub i64 %156, %155, !dbg !15
  %address.thread433 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !15, !range !9, !frisk.register_lifetimes !5
  %address.local434 = zext nneg i32 %address.thread433 to i64, !dbg !8
  %address.local435 = shl nuw nsw i64 %address.local434, 3, !dbg !8
  %address.local436 = and i64 %address.local435, 1536, !dbg !8
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !8
  %address.local440 = lshr i64 %address.local434, 4, !dbg !15
  %address.local441 = and i64 %address.local440, 8, !dbg !15
  %address.local442 = and i64 %address.local434, 123, !dbg !15
  %address.local443 = lshr i64 %address.local434, 2, !dbg !15
  %address.local444 = and i64 %address.local443, 4, !dbg !15
  %address.local445 = add nuw nsw i64 %address.local444, %address.local434, !dbg !15
  %address.local446 = and i64 %address.local445, 4, !dbg !15
  %address.local447 = or disjoint i64 %address.local446, %address.local442, !dbg !15
  %address.local448 = shl nuw nsw i64 %address.local447, 7, !dbg !15
  %address.local449 = or disjoint i64 %address.local448, %address.local441, !dbg !15
  %.idx912 = or disjoint i64 %154, %152, !dbg !15
  %158 = add i64 %.idx912, %157, !dbg !15
  %159 = add i64 %address.local449, %158, !dbg !15
  %.tr = trunc i64 %159 to i32, !dbg !15
  %dma.buffer.offset1588 = shl i32 %.tr, 1, !dbg !15
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !8
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !8
  %address.local453 = and i64 %address.local452, 3584, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !8
  %address.local457 = lshr i64 %address.local451, 4, !dbg !15
  %address.local458 = and i64 %address.local457, 24, !dbg !15
  %address.local466 = or disjoint i64 %address.local448, %address.local458, !dbg !15
  %160 = add i64 %address.local466, %158, !dbg !15
  %.tr17 = trunc i64 %160 to i32, !dbg !15
  %dma.buffer.offset1585 = shl i32 %.tr17, 1, !dbg !15
  %161 = add nuw nsw i64 %147, 1024, !dbg !15
  %162 = lshr i64 %161, 12, !dbg !15
  %163 = add nuw nsw i64 %162, %16, !dbg !15
  %164 = add nsw i64 %151, %109, !dbg !15
  %165 = shl i64 %163, 19, !dbg !15
  %166 = and i64 %165, 16252928, !dbg !15
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !8
  %167 = add i64 %157, %151, !dbg !15
  %.idx1822 = add i64 %167, %109, !dbg !15
  %168 = add i64 %.idx1822, %166, !dbg !15
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !8
  %169 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local437, i64 0, !dbg !8
  %170 = insertelement <4 x ptr addrspace(3)> %169, ptr addrspace(3) %address.local454, i64 1, !dbg !8
  %171 = insertelement <4 x ptr addrspace(3)> %170, ptr addrspace(3) %address.local468, i64 2, !dbg !8
  %172 = insertelement <4 x ptr addrspace(3)> %171, ptr addrspace(3) %address.local472, i64 3, !dbg !8
  %173 = ptrtoint <4 x ptr addrspace(3)> %172 to <4 x i64>, !dbg !8
  %174 = trunc <4 x i64> %173 to <4 x i32>, !dbg !8
  %175 = extractelement <4 x i32> %174, i64 0, !dbg !15
  %176 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %175), !dbg !15
  %dma.buffer.lds1590 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %176), !dbg !15
  %177 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %178 = extractelement <4 x i32> %174, i64 1, !dbg !15
  %179 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %178), !dbg !15
  %dma.buffer.lds1587 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %179), !dbg !15
  %180 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %181 = extractelement <4 x i32> %174, i64 2, !dbg !15
  %182 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %181), !dbg !15
  %dma.buffer.lds1584 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %182), !dbg !15
  %183 = extractelement <4 x i32> %174, i64 3, !dbg !15
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !16
  %address.local481 = and i64 %address.local480, 24, !dbg !16
  %address.local483 = shl i64 %address.local434, 62, !dbg !16
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !16
  %address.local485 = and i64 %address.local484, -64, !dbg !16
  %address.local488 = and i64 %address.local443, 12, !dbg !10
  %184 = and i64 %address.local434, 32, !dbg !10
  %.not30 = icmp eq i64 %184, 0, !dbg !10
  %address.local492 = select i1 %.not30, i64 0, i64 1016, !dbg !10
  %.idx31 = mul nuw nsw i64 %address.local434, 320, !dbg !16
  %185 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx31, !dbg !16
  %.idx32 = mul nsw i64 %address.local443, -1280, !dbg !16
  %186 = getelementptr i8, ptr addrspace(3) %185, i64 %.idx32, !dbg !16
  %187 = getelementptr [2 x i8], ptr addrspace(3) %186, i64 %address.local481, !dbg !16
  %188 = getelementptr [2 x i8], ptr addrspace(3) %187, i64 %address.local485, !dbg !16
  %189 = getelementptr [2 x i8], ptr addrspace(3) %188, i64 %address.local488, !dbg !16
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) %189, i64 %address.local492, !dbg !16
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !16
  %address.local505 = shl nuw nsw i32 %address.thread433, 1, !dbg !16
  %address.local506 = and i32 %address.local505, 24, !dbg !16
  %address.local507 = shl nuw nsw i32 %address.thread433, 8, !dbg !16
  %address.local508 = and i32 %address.local507, 768, !dbg !16
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !16
  %address.local510 = shl nuw nsw i32 %address.thread433, 5, !dbg !16
  %address.local511 = and i32 %address.local510, 96, !dbg !16
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !16
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !16
  %address.local514 = lshr i32 %address.thread433, 1, !dbg !16
  %address.local515 = and i32 %address.local514, 1, !dbg !16
  %address.local516 = zext nneg i32 %address.local515 to i64, !dbg !16
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !16
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !16
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !16
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !16
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !16
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !16
  %190 = add nuw nsw i64 %147, 2048, !dbg !15
  %191 = lshr i64 %190, 12, !dbg !15
  %192 = add nuw nsw i64 %191, %16, !dbg !15
  %193 = insertelement <2 x i64> poison, i64 %164, i64 0, !dbg !15
  %194 = shufflevector <2 x i64> %193, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !15
  %195 = icmp slt <2 x i64> %194, <i64 -32, i64 -64>, !dbg !15
  %196 = extractelement <2 x i1> %195, i64 0, !dbg !15
  %197 = sext i1 %196 to i64, !dbg !15
  %198 = add nsw <2 x i64> %194, <i64 32, i64 64>, !dbg !15
  %199 = sub nsw <2 x i64> <i64 -33, i64 -65>, %194, !dbg !15
  %200 = select <2 x i1> %195, <2 x i64> %199, <2 x i64> %198, !dbg !15
  %201 = sdiv <2 x i64> %200, splat (i64 128), !dbg !15
  %202 = extractelement <2 x i64> %201, i64 0, !dbg !15
  %203 = xor i64 %202, %197, !dbg !15
  %204 = shl nsw i64 %203, 7, !dbg !15
  %205 = sub i64 %168, %204, !dbg !15
  %206 = add i64 %address.local449, %205, !dbg !15
  %.tr23 = trunc i64 %206 to i32, !dbg !15
  %207 = shl i32 %.tr23, 1, !dbg !15
  %dma.buffer.offset1582 = add i32 %207, 64, !dbg !15
  %208 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %209 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %183), !dbg !15
  %210 = add i64 %address.local466, %205, !dbg !15
  %.tr29 = trunc i64 %210 to i32, !dbg !15
  %211 = shl i32 %.tr29, 1, !dbg !15
  %dma.buffer.offset1579 = add i32 %211, 64, !dbg !15
  %dma.buffer.lds1581 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %209), !dbg !15
  %212 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %213 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !16
  %214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %213, <4 x float> zeroinitializer) #8, !dbg !16
  %215 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !16
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %.sext) #8, !dbg !16, !frisk.register_lifetimes !5
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !16
  %address.local518 = shl i64 %address.local517, 6, !dbg !16
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !16
  %216 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local519, !dbg !16
  %217 = getelementptr [2 x i8], ptr addrspace(3) %216, i64 %address.local488, !dbg !16
  %218 = getelementptr [2 x i8], ptr addrspace(3) %217, i64 %address.local492, !dbg !16
  %address.local523 = getelementptr i8, ptr addrspace(3) %218, i64 64, !dbg !16
  %219 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !16
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %215, <4 x float> %214) #8, !dbg !16
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %219, <4 x float> zeroinitializer) #8, !dbg !16
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %218, i64 4160, !dbg !16
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !16
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !16
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %222, <4 x float> %221) #8, !dbg !16
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !16
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !16
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %218, i64 192, !dbg !16
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !16
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %226, <4 x float> %225) #8, !dbg !16
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %227, <4 x float> zeroinitializer) #8, !dbg !16
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %218, i64 4288, !dbg !16
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !16
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !16
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %230, <4 x float> %229) #8, !dbg !16
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %231, <4 x float> zeroinitializer) #8, !dbg !16
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !16
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %218, i64 1088, !dbg !16
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !16
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %234, <4 x float> %233) #8, !dbg !16
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %235, <4 x float> zeroinitializer) #8, !dbg !16
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %218, i64 5184, !dbg !16
  %238 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !16
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !16
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %238, <4 x float> %237) #8, !dbg !16
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %239, <4 x float> zeroinitializer) #8, !dbg !16
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !16
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %218, i64 1216, !dbg !16
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !16
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %242, <4 x float> %241) #8, !dbg !16
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %243, <4 x float> zeroinitializer) #8, !dbg !16
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %218, i64 5312, !dbg !16
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !16
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %246, <4 x float> %245) #8, !dbg !16
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %213, <4 x float> zeroinitializer) #8, !dbg !16
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %215, <4 x float> %248) #8, !dbg !16
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %219, <4 x float> zeroinitializer) #8, !dbg !16
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %222, <4 x float> %250) #8, !dbg !16
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !16
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %226, <4 x float> %252) #8, !dbg !16
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %227, <4 x float> zeroinitializer) #8, !dbg !16
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %230, <4 x float> %254) #8, !dbg !16
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %231, <4 x float> zeroinitializer) #8, !dbg !16
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %234, <4 x float> %256) #8, !dbg !16
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %235, <4 x float> zeroinitializer) #8, !dbg !16
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %238, <4 x float> %258) #8, !dbg !16
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %239, <4 x float> zeroinitializer) #8, !dbg !16
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %242, <4 x float> %260) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %243, <4 x float> zeroinitializer) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %246, <4 x float> %262) #8, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %264 = extractelement <2 x i1> %195, i64 1, !dbg !15
  %265 = sext i1 %264 to i64, !dbg !15
  %266 = extractelement <2 x i64> %201, i64 1, !dbg !15
  %267 = xor i64 %266, %265, !dbg !15
  %268 = shl i64 %192, 19, !dbg !15
  %269 = and i64 %268, 16252928, !dbg !15
  %270 = add i64 %.idx1822, %269, !dbg !15
  %271 = shl nsw i64 %267, 7, !dbg !15
  %272 = sub i64 %270, %271, !dbg !15
  %273 = add i64 %address.local449, %272, !dbg !15
  %.tr38 = trunc i64 %273 to i32, !dbg !15
  %274 = shl i32 %.tr38, 1, !dbg !15
  %dma.buffer.offset1576 = add i32 %274, 128, !dbg !15
  %275 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %276 = add i64 %address.local466, %272, !dbg !15
  %.tr44 = trunc i64 %276 to i32, !dbg !15
  %277 = shl i32 %.tr44, 1, !dbg !15
  %dma.buffer.offset1573 = add i32 %277, 128, !dbg !15
  %278 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !16
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !16
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %279, <4 x float> %220) #8, !dbg !16
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !16
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !16
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %218, i64 8256, !dbg !16
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !16
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %281, <4 x float> %280) #8, !dbg !16
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %282, <4 x float> %224) #8, !dbg !16
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %218, i64 12352, !dbg !16
  %285 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !16
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !16
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !16
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %285, <4 x float> %284) #8, !dbg !16
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %286, <4 x float> %228) #8, !dbg !16
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !16
  %289 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !16
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %218, i64 8384, !dbg !16
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !16
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %289, <4 x float> %288) #8, !dbg !16
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %290, <4 x float> %232) #8, !dbg !16
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %218, i64 12480, !dbg !16
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !16
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !16
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !16
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %293, <4 x float> %292) #8, !dbg !16
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %294, <4 x float> %236) #8, !dbg !16
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !16
  %297 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !16
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %218, i64 9280, !dbg !16
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !16
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %297, <4 x float> %296) #8, !dbg !16
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %298, <4 x float> %240) #8, !dbg !16
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %218, i64 13376, !dbg !16
  %301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !16
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !16
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !16
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %301, <4 x float> %300) #8, !dbg !16
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %302, <4 x float> %244) #8, !dbg !16
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !16
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !16
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %218, i64 9408, !dbg !16
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !16
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %305, <4 x float> %304) #8, !dbg !16
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %306, <4 x float> %247) #8, !dbg !16
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %218, i64 13504, !dbg !16
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !16
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %309, <4 x float> %308) #8, !dbg !16
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %279, <4 x float> %249) #8, !dbg !16
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %281, <4 x float> %311) #8, !dbg !16
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %282, <4 x float> %251) #8, !dbg !16
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %285, <4 x float> %313) #8, !dbg !16
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %286, <4 x float> %253) #8, !dbg !16
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %289, <4 x float> %315) #8, !dbg !16
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %290, <4 x float> %255) #8, !dbg !16
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %293, <4 x float> %317) #8, !dbg !16
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %294, <4 x float> %257) #8, !dbg !16
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %297, <4 x float> %319) #8, !dbg !16
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %298, <4 x float> %259) #8, !dbg !16
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %301, <4 x float> %321) #8, !dbg !16
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %302, <4 x float> %261) #8, !dbg !16
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %305, <4 x float> %323) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %306, <4 x float> %263) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %309, <4 x float> %325) #8, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %327 = add nuw nsw i64 %147, 3072, !dbg !15
  %328 = lshr i64 %327, 12, !dbg !15
  %329 = add nuw nsw i64 %328, %16, !dbg !15
  %330 = add nsw i64 %164, 96, !dbg !15
  %331 = icmp slt i64 %164, -96, !dbg !15
  %332 = sub nsw i64 -97, %164, !dbg !15
  %333 = select i1 %331, i64 %332, i64 %330, !dbg !15
  %334 = sdiv i64 %333, 128, !dbg !15
  %335 = sext i1 %331 to i64, !dbg !15
  %336 = xor i64 %334, %335, !dbg !15
  %337 = shl i64 %329, 19, !dbg !15
  %338 = and i64 %337, 16252928, !dbg !15
  %339 = add i64 %.idx1822, %338, !dbg !15
  %340 = shl nsw i64 %336, 7, !dbg !15
  %341 = sub i64 %339, %340, !dbg !15
  %342 = add i64 %address.local449, %341, !dbg !15
  %.tr50 = trunc i64 %342 to i32, !dbg !15
  %343 = shl i32 %.tr50, 1, !dbg !15
  %dma.buffer.offset1570 = add i32 %343, 192, !dbg !15
  %344 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %345 = add i64 %address.local466, %341, !dbg !15
  %.tr56 = trunc i64 %345 to i32, !dbg !15
  %346 = shl i32 %.tr56, 1, !dbg !15
  %dma.buffer.offset1567 = add i32 %346, 192, !dbg !15
  %347 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %348 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !16
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %348, <4 x float> %283) #8, !dbg !16
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !16
  %351 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !16
  %352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %350, <4 x float> %349) #8, !dbg !16
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %351, <4 x float> %287) #8, !dbg !16
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !16
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !16
  %356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %354, <4 x float> %353) #8, !dbg !16
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %355, <4 x float> %291) #8, !dbg !16
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !16
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !16
  %360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %358, <4 x float> %357) #8, !dbg !16
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %359, <4 x float> %295) #8, !dbg !16
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !16
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !16
  %364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %362, <4 x float> %361) #8, !dbg !16
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %363, <4 x float> %299) #8, !dbg !16
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !16
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !16
  %368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %366, <4 x float> %365) #8, !dbg !16
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %367, <4 x float> %303) #8, !dbg !16
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !16
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !16
  %372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %370, <4 x float> %369) #8, !dbg !16
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %371, <4 x float> %307) #8, !dbg !16
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !16
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !16
  %376 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %374, <4 x float> %373) #8, !dbg !16
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %375, <4 x float> %310) #8, !dbg !16
  %378 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !16
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %378, <4 x float> %377) #8, !dbg !16
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %348, <4 x float> %312) #8, !dbg !16
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %350, <4 x float> %380) #8, !dbg !16
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %351, <4 x float> %314) #8, !dbg !16
  %383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %354, <4 x float> %382) #8, !dbg !16
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %355, <4 x float> %316) #8, !dbg !16
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %358, <4 x float> %384) #8, !dbg !16
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %359, <4 x float> %318) #8, !dbg !16
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %362, <4 x float> %386) #8, !dbg !16
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %363, <4 x float> %320) #8, !dbg !16
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %366, <4 x float> %388) #8, !dbg !16
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %367, <4 x float> %322) #8, !dbg !16
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %370, <4 x float> %390) #8, !dbg !16
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %371, <4 x float> %324) #8, !dbg !16
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %374, <4 x float> %392) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %375, <4 x float> %326) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %395 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %378, <4 x float> %394) #8, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %396 = add nuw nsw i64 %151, %16, !dbg !18
  %397 = shl i64 %396, 19, !dbg !18
  %398 = and i64 %397, 16252928, !dbg !18
  %address.local852 = shl nuw nsw i64 %address.local434, 6, !dbg !18
  %address.local853 = and i64 %address.local852, 16128, !dbg !18
  %address.local855 = and i64 %address.local435, 24, !dbg !18
  %.idx5760 = add i64 %398, %157, !dbg !18
  %399 = or disjoint i64 %address.local855, %.idx5760, !dbg !18
  %400 = shl i64 %399, 1, !dbg !18
  %401 = or disjoint i64 %400, %address.local853, !dbg !18
  %dma.buffer.offset1564 = trunc i64 %401 to i32, !dbg !18
  %402 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %address.local858 = shl nuw nsw i64 %address.local451, 6, !dbg !18
  %address.local859 = and i64 %address.local858, 32512, !dbg !18
  %403 = or disjoint i64 %400, %address.local859, !dbg !18
  %dma.buffer.offset1561 = trunc i64 %403 to i32, !dbg !18
  %404 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %405 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !16
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %405, <4 x float> %352) #8, !dbg !16
  %407 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !16
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !16
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %407, <4 x float> %406) #8, !dbg !16
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %408, <4 x float> %356) #8, !dbg !16
  %411 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !16
  %412 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !16
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %411, <4 x float> %410) #8, !dbg !16
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %412, <4 x float> %360) #8, !dbg !16
  %415 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !16
  %416 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !16
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %415, <4 x float> %414) #8, !dbg !16
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %416, <4 x float> %364) #8, !dbg !16
  %419 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !16
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !16
  %421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %419, <4 x float> %418) #8, !dbg !16
  %422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %420, <4 x float> %368) #8, !dbg !16
  %423 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !16
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !16
  %425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %423, <4 x float> %422) #8, !dbg !16
  %426 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %424, <4 x float> %372) #8, !dbg !16
  %427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !16
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !16
  %429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %427, <4 x float> %426) #8, !dbg !16
  %430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %428, <4 x float> %376) #8, !dbg !16
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !16
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !16
  %433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %431, <4 x float> %430) #8, !dbg !16
  %434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %432, <4 x float> %379) #8, !dbg !16
  %435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !16
  %436 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %435, <4 x float> %434) #8, !dbg !16
  %437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %405, <4 x float> %381) #8, !dbg !16
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %407, <4 x float> %437) #8, !dbg !16
  %439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %408, <4 x float> %383) #8, !dbg !16
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %411, <4 x float> %439) #8, !dbg !16
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %412, <4 x float> %385) #8, !dbg !16
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %415, <4 x float> %441) #8, !dbg !16
  %443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %416, <4 x float> %387) #8, !dbg !16
  %444 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %419, <4 x float> %443) #8, !dbg !16
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %420, <4 x float> %389) #8, !dbg !16
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %423, <4 x float> %445) #8, !dbg !16
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %424, <4 x float> %391) #8, !dbg !16
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %427, <4 x float> %447) #8, !dbg !16
  %449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %428, <4 x float> %393) #8, !dbg !16
  %450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %431, <4 x float> %449) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %432, <4 x float> %395) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %452 = or disjoint i64 %148, 127, !dbg !12
  %453 = icmp uge i64 %24, %452, !dbg !12
  %454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %435, <4 x float> %451) #8, !dbg !16
  %455 = and i1 %102, %453, !dbg !12
  br i1 %455, label %540, label %456, !dbg !12

456:                                              ; preds = %141
  %457 = or disjoint i64 %148, %130, !dbg !12
  %458 = lshr i64 %147, 25, !dbg !17
  %compare.high1157 = trunc nuw i64 %458 to i32, !dbg !17
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !17
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !17
  %compare.low.base1158 = trunc i64 %457 to i32, !dbg !17
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1161 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1165 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1169 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 4) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1173 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 5) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1177 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 6) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1181 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 7) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1185 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 8) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1189 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 9) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1193 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 10) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1197 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 11) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1201 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 12) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1205 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 13) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1209 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 14) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1213 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 15) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1217 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 16) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1221 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 17) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1225 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 18) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1229 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 19) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1233 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 20) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1237 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 21) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1241 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 22) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1245 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 23) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1249 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 24) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1253 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 25) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1257 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 26) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1261 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 27) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1265 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 28) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1269 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 29) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1273 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 30) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1277 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 31) #8, !dbg !17, !frisk.register_lifetimes !5
  %459 = insertelement <32 x i32> poison, i32 %compare.low.base1158, i64 0, !dbg !17
  %460 = insertelement <32 x i32> %459, i32 %compare.local.base, i64 1, !dbg !17
  %461 = insertelement <32 x i32> %460, i32 %compare.local.base1161, i64 2, !dbg !17
  %462 = insertelement <32 x i32> %461, i32 %compare.local.base1165, i64 3, !dbg !17
  %463 = insertelement <32 x i32> %462, i32 %compare.local.base1169, i64 4, !dbg !17
  %464 = insertelement <32 x i32> %463, i32 %compare.local.base1173, i64 5, !dbg !17
  %465 = insertelement <32 x i32> %464, i32 %compare.local.base1177, i64 6, !dbg !17
  %466 = insertelement <32 x i32> %465, i32 %compare.local.base1181, i64 7, !dbg !17
  %467 = insertelement <32 x i32> %466, i32 %compare.local.base1185, i64 8, !dbg !17
  %468 = insertelement <32 x i32> %467, i32 %compare.local.base1189, i64 9, !dbg !17
  %469 = insertelement <32 x i32> %468, i32 %compare.local.base1193, i64 10, !dbg !17
  %470 = insertelement <32 x i32> %469, i32 %compare.local.base1197, i64 11, !dbg !17
  %471 = insertelement <32 x i32> %470, i32 %compare.local.base1201, i64 12, !dbg !17
  %472 = insertelement <32 x i32> %471, i32 %compare.local.base1205, i64 13, !dbg !17
  %473 = insertelement <32 x i32> %472, i32 %compare.local.base1209, i64 14, !dbg !17
  %474 = insertelement <32 x i32> %473, i32 %compare.local.base1213, i64 15, !dbg !17
  %475 = insertelement <32 x i32> %474, i32 %compare.local.base1217, i64 16, !dbg !17
  %476 = insertelement <32 x i32> %475, i32 %compare.local.base1221, i64 17, !dbg !17
  %477 = insertelement <32 x i32> %476, i32 %compare.local.base1225, i64 18, !dbg !17
  %478 = insertelement <32 x i32> %477, i32 %compare.local.base1229, i64 19, !dbg !17
  %479 = insertelement <32 x i32> %478, i32 %compare.local.base1233, i64 20, !dbg !17
  %480 = insertelement <32 x i32> %479, i32 %compare.local.base1237, i64 21, !dbg !17
  %481 = insertelement <32 x i32> %480, i32 %compare.local.base1241, i64 22, !dbg !17
  %482 = insertelement <32 x i32> %481, i32 %compare.local.base1245, i64 23, !dbg !17
  %483 = insertelement <32 x i32> %482, i32 %compare.local.base1249, i64 24, !dbg !17
  %484 = insertelement <32 x i32> %483, i32 %compare.local.base1253, i64 25, !dbg !17
  %485 = insertelement <32 x i32> %484, i32 %compare.local.base1257, i64 26, !dbg !17
  %486 = insertelement <32 x i32> %485, i32 %compare.local.base1261, i64 27, !dbg !17
  %487 = insertelement <32 x i32> %486, i32 %compare.local.base1265, i64 28, !dbg !17
  %488 = insertelement <32 x i32> %487, i32 %compare.local.base1269, i64 29, !dbg !17
  %489 = insertelement <32 x i32> %488, i32 %compare.local.base1273, i64 30, !dbg !17
  %490 = insertelement <32 x i32> %489, i32 %compare.local.base1277, i64 31, !dbg !17
  %491 = or <32 x i32> %490, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !17
  %492 = icmp ugt <32 x i32> %140, %491, !dbg !17
  %493 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !17
  %494 = shufflevector <32 x i1> %493, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !17
  %495 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !17
  %496 = shufflevector <32 x i1> %495, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !17
  %497 = select <32 x i1> %494, <32 x i1> %492, <32 x i1> %496, !dbg !17
  %498 = select <32 x i1> %497, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !19
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1157, !dbg !17
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1157, !dbg !17
  %compare.local.base1289 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 32) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1293 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 33) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1297 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 34) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1301 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 35) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1305 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 36) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1309 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 37) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1313 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 38) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1317 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 39) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1321 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 40) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1325 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 41) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1329 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 42) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1333 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 43) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1337 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 44) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1341 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 45) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1345 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 46) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1349 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 47) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1353 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 48) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1357 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 49) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1361 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 50) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1365 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 51) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1369 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 52) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1373 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 53) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1377 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 54) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1381 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 55) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1385 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 56) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1389 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 57) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1393 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 58) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1397 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 59) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1401 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 60) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1405 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 61) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1409 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 62) #8, !dbg !17, !frisk.register_lifetimes !5
  %499 = insertelement <32 x i32> %459, i32 %compare.local.base1289, i64 1, !dbg !17
  %500 = insertelement <32 x i32> %499, i32 %compare.local.base1293, i64 2, !dbg !17
  %501 = insertelement <32 x i32> %500, i32 %compare.local.base1297, i64 3, !dbg !17
  %502 = insertelement <32 x i32> %501, i32 %compare.local.base1301, i64 4, !dbg !17
  %503 = insertelement <32 x i32> %502, i32 %compare.local.base1305, i64 5, !dbg !17
  %504 = insertelement <32 x i32> %503, i32 %compare.local.base1309, i64 6, !dbg !17
  %505 = insertelement <32 x i32> %504, i32 %compare.local.base1313, i64 7, !dbg !17
  %506 = insertelement <32 x i32> %505, i32 %compare.local.base1317, i64 8, !dbg !17
  %507 = insertelement <32 x i32> %506, i32 %compare.local.base1321, i64 9, !dbg !17
  %508 = insertelement <32 x i32> %507, i32 %compare.local.base1325, i64 10, !dbg !17
  %509 = insertelement <32 x i32> %508, i32 %compare.local.base1329, i64 11, !dbg !17
  %510 = insertelement <32 x i32> %509, i32 %compare.local.base1333, i64 12, !dbg !17
  %511 = insertelement <32 x i32> %510, i32 %compare.local.base1337, i64 13, !dbg !17
  %512 = insertelement <32 x i32> %511, i32 %compare.local.base1341, i64 14, !dbg !17
  %513 = insertelement <32 x i32> %512, i32 %compare.local.base1345, i64 15, !dbg !17
  %514 = insertelement <32 x i32> %513, i32 %compare.local.base1349, i64 16, !dbg !17
  %515 = insertelement <32 x i32> %514, i32 %compare.local.base1353, i64 17, !dbg !17
  %516 = insertelement <32 x i32> %515, i32 %compare.local.base1357, i64 18, !dbg !17
  %517 = insertelement <32 x i32> %516, i32 %compare.local.base1361, i64 19, !dbg !17
  %518 = insertelement <32 x i32> %517, i32 %compare.local.base1365, i64 20, !dbg !17
  %519 = insertelement <32 x i32> %518, i32 %compare.local.base1369, i64 21, !dbg !17
  %520 = insertelement <32 x i32> %519, i32 %compare.local.base1373, i64 22, !dbg !17
  %521 = insertelement <32 x i32> %520, i32 %compare.local.base1377, i64 23, !dbg !17
  %522 = insertelement <32 x i32> %521, i32 %compare.local.base1381, i64 24, !dbg !17
  %523 = insertelement <32 x i32> %522, i32 %compare.local.base1385, i64 25, !dbg !17
  %524 = insertelement <32 x i32> %523, i32 %compare.local.base1389, i64 26, !dbg !17
  %525 = insertelement <32 x i32> %524, i32 %compare.local.base1393, i64 27, !dbg !17
  %526 = insertelement <32 x i32> %525, i32 %compare.local.base1397, i64 28, !dbg !17
  %527 = insertelement <32 x i32> %526, i32 %compare.local.base1401, i64 29, !dbg !17
  %528 = insertelement <32 x i32> %527, i32 %compare.local.base1405, i64 30, !dbg !17
  %529 = insertelement <32 x i32> %528, i32 %compare.local.base1409, i64 31, !dbg !17
  %530 = or <32 x i32> %529, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !17
  %531 = icmp ugt <32 x i32> %138, %530, !dbg !17
  %532 = insertelement <32 x i1> poison, i1 %compare.high.equal1283, i64 0, !dbg !17
  %533 = shufflevector <32 x i1> %532, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !17
  %534 = insertelement <32 x i1> poison, i1 %compare.high.order1284, i64 0, !dbg !17
  %535 = shufflevector <32 x i1> %534, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !17
  %536 = select <32 x i1> %533, <32 x i1> %531, <32 x i1> %535, !dbg !17
  %537 = select <32 x i1> %536, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !19
  %538 = insertvalue [2 x <32 x float>] poison, <32 x float> %498, 0, !dbg !12
  %539 = insertvalue [2 x <32 x float>] %538, <32 x float> %537, 1, !dbg !12
  br label %540, !dbg !12

540:                                              ; preds = %456, %141
  %541 = phi [2 x <32 x float>] [ %539, %456 ], [ zeroinitializer, %141 ], !dbg !12
  %542 = extractvalue [2 x <32 x float>] %541, 0, !dbg !20
  %543 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %544 = fadd <4 x float> %409, %543, !dbg !20
  %545 = extractelement <4 x float> %544, i64 0, !dbg !20
  %546 = extractelement <4 x float> %544, i64 1, !dbg !20
  %547 = extractelement <4 x float> %544, i64 2, !dbg !20
  %548 = extractelement <4 x float> %544, i64 3, !dbg !20
  %549 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %550 = fadd <4 x float> %413, %549, !dbg !20
  %551 = extractelement <4 x float> %550, i64 0, !dbg !20
  %552 = extractelement <4 x float> %550, i64 1, !dbg !20
  %553 = extractelement <4 x float> %550, i64 2, !dbg !20
  %554 = extractelement <4 x float> %550, i64 3, !dbg !20
  %555 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !20
  %556 = fadd <4 x float> %417, %555, !dbg !20
  %557 = extractelement <4 x float> %556, i64 0, !dbg !20
  %558 = extractelement <4 x float> %556, i64 1, !dbg !20
  %559 = extractelement <4 x float> %556, i64 2, !dbg !20
  %560 = extractelement <4 x float> %556, i64 3, !dbg !20
  %561 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %562 = fadd <4 x float> %421, %561, !dbg !20
  %563 = extractelement <4 x float> %562, i64 0, !dbg !20
  %564 = extractelement <4 x float> %562, i64 1, !dbg !20
  %565 = extractelement <4 x float> %562, i64 2, !dbg !20
  %566 = extractelement <4 x float> %562, i64 3, !dbg !20
  %567 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !20
  %568 = fadd <4 x float> %425, %567, !dbg !20
  %569 = extractelement <4 x float> %568, i64 0, !dbg !20
  %570 = extractelement <4 x float> %568, i64 1, !dbg !20
  %571 = extractelement <4 x float> %568, i64 2, !dbg !20
  %572 = extractelement <4 x float> %568, i64 3, !dbg !20
  %573 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !20
  %574 = fadd <4 x float> %429, %573, !dbg !20
  %575 = extractelement <4 x float> %574, i64 0, !dbg !20
  %576 = extractelement <4 x float> %574, i64 1, !dbg !20
  %577 = extractelement <4 x float> %574, i64 2, !dbg !20
  %578 = extractelement <4 x float> %574, i64 3, !dbg !20
  %579 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !20
  %580 = fadd <4 x float> %433, %579, !dbg !20
  %581 = extractelement <4 x float> %580, i64 0, !dbg !20
  %582 = extractelement <4 x float> %580, i64 1, !dbg !20
  %583 = extractelement <4 x float> %580, i64 2, !dbg !20
  %584 = extractelement <4 x float> %580, i64 3, !dbg !20
  %585 = shufflevector <32 x float> %542, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %586 = fadd <4 x float> %436, %585, !dbg !20
  %587 = extractelement <4 x float> %586, i64 0, !dbg !20
  %588 = extractelement <4 x float> %586, i64 1, !dbg !20
  %589 = extractelement <4 x float> %586, i64 2, !dbg !20
  %590 = extractelement <4 x float> %586, i64 3, !dbg !20
  %591 = extractvalue [2 x <32 x float>] %541, 1, !dbg !20
  %592 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %593 = fadd <4 x float> %438, %592, !dbg !20
  %594 = extractelement <4 x float> %593, i64 0, !dbg !20
  %595 = extractelement <4 x float> %593, i64 1, !dbg !20
  %596 = extractelement <4 x float> %593, i64 2, !dbg !20
  %597 = extractelement <4 x float> %593, i64 3, !dbg !20
  %598 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %599 = fadd <4 x float> %440, %598, !dbg !20
  %600 = extractelement <4 x float> %599, i64 0, !dbg !20
  %601 = extractelement <4 x float> %599, i64 1, !dbg !20
  %602 = extractelement <4 x float> %599, i64 2, !dbg !20
  %603 = extractelement <4 x float> %599, i64 3, !dbg !20
  %604 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !20
  %605 = fadd <4 x float> %442, %604, !dbg !20
  %606 = extractelement <4 x float> %605, i64 0, !dbg !20
  %607 = extractelement <4 x float> %605, i64 1, !dbg !20
  %608 = extractelement <4 x float> %605, i64 2, !dbg !20
  %609 = extractelement <4 x float> %605, i64 3, !dbg !20
  %610 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %611 = fadd <4 x float> %444, %610, !dbg !20
  %612 = extractelement <4 x float> %611, i64 0, !dbg !20
  %613 = extractelement <4 x float> %611, i64 1, !dbg !20
  %614 = extractelement <4 x float> %611, i64 2, !dbg !20
  %615 = extractelement <4 x float> %611, i64 3, !dbg !20
  %616 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !20
  %617 = fadd <4 x float> %446, %616, !dbg !20
  %618 = extractelement <4 x float> %617, i64 0, !dbg !20
  %619 = extractelement <4 x float> %617, i64 1, !dbg !20
  %620 = extractelement <4 x float> %617, i64 2, !dbg !20
  %621 = extractelement <4 x float> %617, i64 3, !dbg !20
  %622 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !20
  %623 = fadd <4 x float> %448, %622, !dbg !20
  %624 = extractelement <4 x float> %623, i64 0, !dbg !20
  %625 = extractelement <4 x float> %623, i64 1, !dbg !20
  %626 = extractelement <4 x float> %623, i64 2, !dbg !20
  %627 = extractelement <4 x float> %623, i64 3, !dbg !20
  %628 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !20
  %629 = fadd <4 x float> %450, %628, !dbg !20
  %630 = extractelement <4 x float> %629, i64 0, !dbg !20
  %631 = extractelement <4 x float> %629, i64 1, !dbg !20
  %632 = extractelement <4 x float> %629, i64 2, !dbg !20
  %633 = extractelement <4 x float> %629, i64 3, !dbg !20
  %634 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %635 = fadd <4 x float> %454, %634, !dbg !20
  %636 = extractelement <4 x float> %635, i64 0, !dbg !20
  %637 = extractelement <4 x float> %635, i64 1, !dbg !20
  %638 = extractelement <4 x float> %635, i64 2, !dbg !20
  %639 = extractelement <4 x float> %635, i64 3, !dbg !20
  %640 = tail call float @llvm.maxnum.f32(float %545, float -inf), !dbg !21
  %641 = tail call float @llvm.maxnum.f32(float %640, float %546), !dbg !21
  %642 = tail call float @llvm.maxnum.f32(float %641, float %547), !dbg !21
  %643 = tail call float @llvm.maxnum.f32(float %642, float %548), !dbg !21
  %644 = tail call float @llvm.maxnum.f32(float %643, float %551), !dbg !21
  %645 = tail call float @llvm.maxnum.f32(float %644, float %552), !dbg !21
  %646 = tail call float @llvm.maxnum.f32(float %645, float %553), !dbg !21
  %647 = tail call float @llvm.maxnum.f32(float %646, float %554), !dbg !21
  %648 = tail call float @llvm.maxnum.f32(float %647, float %557), !dbg !21
  %649 = tail call float @llvm.maxnum.f32(float %648, float %558), !dbg !21
  %650 = tail call float @llvm.maxnum.f32(float %649, float %559), !dbg !21
  %651 = tail call float @llvm.maxnum.f32(float %650, float %560), !dbg !21
  %652 = tail call float @llvm.maxnum.f32(float %651, float %563), !dbg !21
  %653 = tail call float @llvm.maxnum.f32(float %652, float %564), !dbg !21
  %654 = tail call float @llvm.maxnum.f32(float %653, float %565), !dbg !21
  %655 = tail call float @llvm.maxnum.f32(float %654, float %566), !dbg !21
  %656 = tail call float @llvm.maxnum.f32(float %655, float %569), !dbg !21
  %657 = tail call float @llvm.maxnum.f32(float %656, float %570), !dbg !21
  %658 = tail call float @llvm.maxnum.f32(float %657, float %571), !dbg !21
  %659 = tail call float @llvm.maxnum.f32(float %658, float %572), !dbg !21
  %660 = tail call float @llvm.maxnum.f32(float %659, float %575), !dbg !21
  %661 = tail call float @llvm.maxnum.f32(float %660, float %576), !dbg !21
  %662 = tail call float @llvm.maxnum.f32(float %661, float %577), !dbg !21
  %663 = tail call float @llvm.maxnum.f32(float %662, float %578), !dbg !21
  %664 = tail call float @llvm.maxnum.f32(float %663, float %581), !dbg !21
  %665 = tail call float @llvm.maxnum.f32(float %664, float %582), !dbg !21
  %666 = tail call float @llvm.maxnum.f32(float %665, float %583), !dbg !21
  %667 = tail call float @llvm.maxnum.f32(float %666, float %584), !dbg !21
  %668 = tail call float @llvm.maxnum.f32(float %667, float %587), !dbg !21
  %669 = tail call float @llvm.maxnum.f32(float %668, float %588), !dbg !21
  %670 = tail call float @llvm.maxnum.f32(float %669, float %589), !dbg !21
  %671 = tail call float @llvm.maxnum.f32(float %670, float %590), !dbg !21
  %672 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !21
  %673 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %672), !dbg !21
  %674 = add nuw nsw i32 %673, 64, !dbg !21
  %675 = xor i32 %673, 16, !dbg !21
  %676 = and i32 %674, 192, !dbg !21
  %677 = icmp samesign ult i32 %675, %676, !dbg !21
  %678 = select i1 %677, i32 %675, i32 %673, !dbg !21
  %679 = shl nuw nsw i32 %678, 2, !dbg !21
  %680 = bitcast float %671 to i32, !dbg !21
  %681 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %680), !dbg !21
  %682 = bitcast i32 %681 to float, !dbg !21
  %683 = tail call float @llvm.maxnum.f32(float %671, float %682), !dbg !21
  %684 = xor i32 %673, 32, !dbg !21
  %685 = icmp samesign ult i32 %684, %676, !dbg !21
  %686 = select i1 %685, i32 %684, i32 %673, !dbg !21
  %687 = shl nuw nsw i32 %686, 2, !dbg !21
  %688 = bitcast float %683 to i32, !dbg !21
  %689 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %688), !dbg !21
  %690 = bitcast i32 %689 to float, !dbg !21
  %691 = tail call float @llvm.maxnum.f32(float %683, float %690), !dbg !21
  %.scalar = tail call float @llvm.maxnum.f32(float %691, float -inf), !dbg !21
  %692 = insertelement <1 x float> poison, float %.scalar, i64 0, !dbg !21
  %693 = tail call float @llvm.maxnum.f32(float %594, float -inf), !dbg !21
  %694 = tail call float @llvm.maxnum.f32(float %693, float %595), !dbg !21
  %695 = tail call float @llvm.maxnum.f32(float %694, float %596), !dbg !21
  %696 = tail call float @llvm.maxnum.f32(float %695, float %597), !dbg !21
  %697 = tail call float @llvm.maxnum.f32(float %696, float %600), !dbg !21
  %698 = tail call float @llvm.maxnum.f32(float %697, float %601), !dbg !21
  %699 = tail call float @llvm.maxnum.f32(float %698, float %602), !dbg !21
  %700 = tail call float @llvm.maxnum.f32(float %699, float %603), !dbg !21
  %701 = tail call float @llvm.maxnum.f32(float %700, float %606), !dbg !21
  %702 = tail call float @llvm.maxnum.f32(float %701, float %607), !dbg !21
  %703 = tail call float @llvm.maxnum.f32(float %702, float %608), !dbg !21
  %704 = tail call float @llvm.maxnum.f32(float %703, float %609), !dbg !21
  %705 = tail call float @llvm.maxnum.f32(float %704, float %612), !dbg !21
  %706 = tail call float @llvm.maxnum.f32(float %705, float %613), !dbg !21
  %707 = tail call float @llvm.maxnum.f32(float %706, float %614), !dbg !21
  %708 = tail call float @llvm.maxnum.f32(float %707, float %615), !dbg !21
  %709 = tail call float @llvm.maxnum.f32(float %708, float %618), !dbg !21
  %710 = tail call float @llvm.maxnum.f32(float %709, float %619), !dbg !21
  %711 = tail call float @llvm.maxnum.f32(float %710, float %620), !dbg !21
  %712 = tail call float @llvm.maxnum.f32(float %711, float %621), !dbg !21
  %713 = tail call float @llvm.maxnum.f32(float %712, float %624), !dbg !21
  %714 = tail call float @llvm.maxnum.f32(float %713, float %625), !dbg !21
  %715 = tail call float @llvm.maxnum.f32(float %714, float %626), !dbg !21
  %716 = tail call float @llvm.maxnum.f32(float %715, float %627), !dbg !21
  %717 = tail call float @llvm.maxnum.f32(float %716, float %630), !dbg !21
  %718 = tail call float @llvm.maxnum.f32(float %717, float %631), !dbg !21
  %719 = tail call float @llvm.maxnum.f32(float %718, float %632), !dbg !21
  %720 = tail call float @llvm.maxnum.f32(float %719, float %633), !dbg !21
  %721 = tail call float @llvm.maxnum.f32(float %720, float %636), !dbg !21
  %722 = tail call float @llvm.maxnum.f32(float %721, float %637), !dbg !21
  %723 = tail call float @llvm.maxnum.f32(float %722, float %638), !dbg !21
  %724 = tail call float @llvm.maxnum.f32(float %723, float %639), !dbg !21
  %725 = bitcast float %724 to i32, !dbg !21
  %726 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %725), !dbg !21
  %727 = bitcast i32 %726 to float, !dbg !21
  %728 = tail call float @llvm.maxnum.f32(float %724, float %727), !dbg !21
  %729 = bitcast float %728 to i32, !dbg !21
  %730 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %729), !dbg !21
  %731 = bitcast i32 %730 to float, !dbg !21
  %732 = tail call float @llvm.maxnum.f32(float %728, float %731), !dbg !21
  %.scalar105 = tail call float @llvm.maxnum.f32(float %732, float -inf), !dbg !21
  %733 = insertelement <1 x float> poison, float %.scalar105, i64 0, !dbg !21
  %734 = shufflevector <1 x float> %692, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !22
  %735 = fsub <4 x float> %544, %734, !dbg !22
  %736 = fsub <4 x float> %550, %734, !dbg !22
  %737 = fsub <4 x float> %556, %734, !dbg !22
  %738 = fsub <4 x float> %562, %734, !dbg !22
  %739 = fsub <4 x float> %568, %734, !dbg !22
  %740 = fsub <4 x float> %574, %734, !dbg !22
  %741 = fsub <4 x float> %580, %734, !dbg !22
  %742 = fsub <4 x float> %586, %734, !dbg !22
  %743 = shufflevector <1 x float> %733, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !22
  %744 = fsub <4 x float> %593, %743, !dbg !22
  %745 = fsub <4 x float> %599, %743, !dbg !22
  %746 = fsub <4 x float> %605, %743, !dbg !22
  %747 = fsub <4 x float> %611, %743, !dbg !22
  %748 = fsub <4 x float> %617, %743, !dbg !22
  %749 = fsub <4 x float> %623, %743, !dbg !22
  %750 = fsub <4 x float> %629, %743, !dbg !22
  %751 = fsub <4 x float> %635, %743, !dbg !22
  %752 = fsub float -inf, %.scalar, !dbg !23
  %753 = tail call float @llvm.amdgcn.exp2.f32(float %752), !dbg !24
  %754 = insertelement <4 x float> poison, float %753, i64 0, !dbg !25
  %755 = shufflevector <4 x float> %754, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %756 = extractvalue [2 x <1 x float>] %145, 0, !dbg !26
  %757 = extractelement <1 x float> %756, i64 0, !dbg !26
  %758 = fmul float %757, %753, !dbg !26
  %759 = fsub float -inf, %.scalar105, !dbg !23
  %760 = tail call float @llvm.amdgcn.exp2.f32(float %759), !dbg !24
  %761 = insertelement <4 x float> poison, float %760, i64 0, !dbg !25
  %762 = shufflevector <4 x float> %761, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %763 = extractvalue [2 x <1 x float>] %145, 1, !dbg !26
  %764 = extractelement <1 x float> %763, i64 0, !dbg !26
  %765 = fmul float %764, %760, !dbg !26
  %766 = extractelement <4 x float> %735, i64 0, !dbg !27
  %767 = tail call float @llvm.amdgcn.exp2.f32(float %766), !dbg !27
  %768 = insertelement <4 x float> poison, float %767, i64 0, !dbg !28
  %769 = extractelement <4 x float> %735, i64 1, !dbg !27
  %770 = tail call float @llvm.amdgcn.exp2.f32(float %769), !dbg !27
  %771 = insertelement <4 x float> %768, float %770, i64 1, !dbg !28
  %772 = fadd float %767, %770, !dbg !29
  %773 = extractelement <4 x float> %735, i64 2, !dbg !27
  %774 = tail call float @llvm.amdgcn.exp2.f32(float %773), !dbg !27
  %775 = insertelement <4 x float> %771, float %774, i64 2, !dbg !28
  %776 = fadd float %774, %772, !dbg !29
  %777 = extractelement <4 x float> %735, i64 3, !dbg !27
  %778 = tail call float @llvm.amdgcn.exp2.f32(float %777), !dbg !27
  %779 = insertelement <4 x float> %775, float %778, i64 3, !dbg !28
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %779) #8, !dbg !28, !frisk.register_lifetimes !5
  %780 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !28
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %780) #8, !dbg !29, !frisk.register_lifetimes !5
  %781 = fadd float %778, %776, !dbg !29
  %782 = extractelement <4 x float> %736, i64 0, !dbg !27
  %783 = tail call float @llvm.amdgcn.exp2.f32(float %782), !dbg !27
  %784 = insertelement <4 x float> poison, float %783, i64 0, !dbg !28
  %785 = fadd float %783, %781, !dbg !29
  %786 = extractelement <4 x float> %736, i64 1, !dbg !27
  %787 = tail call float @llvm.amdgcn.exp2.f32(float %786), !dbg !27
  %788 = insertelement <4 x float> %784, float %787, i64 1, !dbg !28
  %789 = fadd float %787, %785, !dbg !29
  %790 = extractelement <4 x float> %736, i64 2, !dbg !27
  %791 = tail call float @llvm.amdgcn.exp2.f32(float %790), !dbg !27
  %792 = insertelement <4 x float> %788, float %791, i64 2, !dbg !28
  %793 = fadd float %791, %789, !dbg !29
  %794 = extractelement <4 x float> %736, i64 3, !dbg !27
  %795 = tail call float @llvm.amdgcn.exp2.f32(float %794), !dbg !27
  %796 = insertelement <4 x float> %792, float %795, i64 3, !dbg !28
  %math.result309 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %796) #8, !dbg !28, !frisk.register_lifetimes !5
  %797 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !28
  %math.packed310 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %797) #8, !dbg !29, !frisk.register_lifetimes !5
  %798 = fadd float %795, %793, !dbg !29
  %799 = extractelement <4 x float> %737, i64 0, !dbg !27
  %800 = tail call float @llvm.amdgcn.exp2.f32(float %799), !dbg !27
  %801 = insertelement <4 x float> poison, float %800, i64 0, !dbg !28
  %802 = fadd float %800, %798, !dbg !29
  %803 = extractelement <4 x float> %737, i64 1, !dbg !27
  %804 = tail call float @llvm.amdgcn.exp2.f32(float %803), !dbg !27
  %805 = insertelement <4 x float> %801, float %804, i64 1, !dbg !28
  %806 = fadd float %804, %802, !dbg !29
  %807 = extractelement <4 x float> %737, i64 2, !dbg !27
  %808 = tail call float @llvm.amdgcn.exp2.f32(float %807), !dbg !27
  %809 = insertelement <4 x float> %805, float %808, i64 2, !dbg !28
  %810 = fadd float %808, %806, !dbg !29
  %811 = extractelement <4 x float> %737, i64 3, !dbg !27
  %812 = tail call float @llvm.amdgcn.exp2.f32(float %811), !dbg !27
  %813 = insertelement <4 x float> %809, float %812, i64 3, !dbg !28
  %math.result311 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %813) #8, !dbg !28, !frisk.register_lifetimes !5
  %814 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !28
  %math.packed312 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %814) #8, !dbg !29, !frisk.register_lifetimes !5
  %815 = fadd float %812, %810, !dbg !29
  %816 = extractelement <4 x float> %738, i64 0, !dbg !27
  %817 = tail call float @llvm.amdgcn.exp2.f32(float %816), !dbg !27
  %818 = insertelement <4 x float> poison, float %817, i64 0, !dbg !28
  %819 = fadd float %817, %815, !dbg !29
  %820 = extractelement <4 x float> %738, i64 1, !dbg !27
  %821 = tail call float @llvm.amdgcn.exp2.f32(float %820), !dbg !27
  %822 = insertelement <4 x float> %818, float %821, i64 1, !dbg !28
  %823 = fadd float %821, %819, !dbg !29
  %824 = extractelement <4 x float> %738, i64 2, !dbg !27
  %825 = tail call float @llvm.amdgcn.exp2.f32(float %824), !dbg !27
  %826 = insertelement <4 x float> %822, float %825, i64 2, !dbg !28
  %827 = fadd float %825, %823, !dbg !29
  %828 = extractelement <4 x float> %738, i64 3, !dbg !27
  %829 = tail call float @llvm.amdgcn.exp2.f32(float %828), !dbg !27
  %830 = insertelement <4 x float> %826, float %829, i64 3, !dbg !28
  %math.result313 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %830) #8, !dbg !28, !frisk.register_lifetimes !5
  %831 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !28
  %math.packed314 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %831) #8, !dbg !29, !frisk.register_lifetimes !5
  %832 = fadd float %829, %827, !dbg !29
  %833 = extractelement <4 x float> %739, i64 0, !dbg !27
  %834 = tail call float @llvm.amdgcn.exp2.f32(float %833), !dbg !27
  %835 = insertelement <4 x float> poison, float %834, i64 0, !dbg !28
  %836 = fadd float %834, %832, !dbg !29
  %837 = extractelement <4 x float> %739, i64 1, !dbg !27
  %838 = tail call float @llvm.amdgcn.exp2.f32(float %837), !dbg !27
  %839 = insertelement <4 x float> %835, float %838, i64 1, !dbg !28
  %840 = fadd float %838, %836, !dbg !29
  %841 = extractelement <4 x float> %739, i64 2, !dbg !27
  %842 = tail call float @llvm.amdgcn.exp2.f32(float %841), !dbg !27
  %843 = insertelement <4 x float> %839, float %842, i64 2, !dbg !28
  %844 = fadd float %842, %840, !dbg !29
  %845 = extractelement <4 x float> %739, i64 3, !dbg !27
  %846 = tail call float @llvm.amdgcn.exp2.f32(float %845), !dbg !27
  %847 = insertelement <4 x float> %843, float %846, i64 3, !dbg !28
  %math.result315 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %847) #8, !dbg !28, !frisk.register_lifetimes !5
  %848 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !28
  %math.packed316 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %848) #8, !dbg !29, !frisk.register_lifetimes !5
  %849 = fadd float %846, %844, !dbg !29
  %850 = extractelement <4 x float> %740, i64 0, !dbg !27
  %851 = tail call float @llvm.amdgcn.exp2.f32(float %850), !dbg !27
  %852 = insertelement <4 x float> poison, float %851, i64 0, !dbg !28
  %853 = fadd float %851, %849, !dbg !29
  %854 = extractelement <4 x float> %740, i64 1, !dbg !27
  %855 = tail call float @llvm.amdgcn.exp2.f32(float %854), !dbg !27
  %856 = insertelement <4 x float> %852, float %855, i64 1, !dbg !28
  %857 = fadd float %855, %853, !dbg !29
  %858 = extractelement <4 x float> %740, i64 2, !dbg !27
  %859 = tail call float @llvm.amdgcn.exp2.f32(float %858), !dbg !27
  %860 = insertelement <4 x float> %856, float %859, i64 2, !dbg !28
  %861 = fadd float %859, %857, !dbg !29
  %862 = extractelement <4 x float> %740, i64 3, !dbg !27
  %863 = tail call float @llvm.amdgcn.exp2.f32(float %862), !dbg !27
  %864 = insertelement <4 x float> %860, float %863, i64 3, !dbg !28
  %math.result317 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %864) #8, !dbg !28, !frisk.register_lifetimes !5
  %865 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !28
  %math.packed318 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %865) #8, !dbg !29, !frisk.register_lifetimes !5
  %866 = fadd float %863, %861, !dbg !29
  %867 = extractelement <4 x float> %741, i64 0, !dbg !27
  %868 = tail call float @llvm.amdgcn.exp2.f32(float %867), !dbg !27
  %869 = insertelement <4 x float> poison, float %868, i64 0, !dbg !28
  %870 = fadd float %868, %866, !dbg !29
  %871 = extractelement <4 x float> %741, i64 1, !dbg !27
  %872 = tail call float @llvm.amdgcn.exp2.f32(float %871), !dbg !27
  %873 = insertelement <4 x float> %869, float %872, i64 1, !dbg !28
  %874 = fadd float %872, %870, !dbg !29
  %875 = extractelement <4 x float> %741, i64 2, !dbg !27
  %876 = tail call float @llvm.amdgcn.exp2.f32(float %875), !dbg !27
  %877 = insertelement <4 x float> %873, float %876, i64 2, !dbg !28
  %878 = fadd float %876, %874, !dbg !29
  %879 = extractelement <4 x float> %741, i64 3, !dbg !27
  %880 = tail call float @llvm.amdgcn.exp2.f32(float %879), !dbg !27
  %881 = insertelement <4 x float> %877, float %880, i64 3, !dbg !28
  %math.result319 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %881) #8, !dbg !28, !frisk.register_lifetimes !5
  %882 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !28
  %math.packed320 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %882) #8, !dbg !29, !frisk.register_lifetimes !5
  %883 = fadd float %880, %878, !dbg !29
  %884 = extractelement <4 x float> %742, i64 0, !dbg !27
  %885 = tail call float @llvm.amdgcn.exp2.f32(float %884), !dbg !27
  %886 = insertelement <4 x float> poison, float %885, i64 0, !dbg !28
  %887 = fadd float %885, %883, !dbg !29
  %888 = extractelement <4 x float> %742, i64 1, !dbg !27
  %889 = tail call float @llvm.amdgcn.exp2.f32(float %888), !dbg !27
  %890 = insertelement <4 x float> %886, float %889, i64 1, !dbg !28
  %891 = fadd float %889, %887, !dbg !29
  %892 = extractelement <4 x float> %742, i64 2, !dbg !27
  %893 = tail call float @llvm.amdgcn.exp2.f32(float %892), !dbg !27
  %894 = insertelement <4 x float> %890, float %893, i64 2, !dbg !28
  %895 = fadd float %893, %891, !dbg !29
  %896 = extractelement <4 x float> %742, i64 3, !dbg !27
  %897 = tail call float @llvm.amdgcn.exp2.f32(float %896), !dbg !27
  %898 = insertelement <4 x float> %894, float %897, i64 3, !dbg !28
  %math.result321 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %898) #8, !dbg !28, !frisk.register_lifetimes !5
  %899 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !28
  %math.packed322 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %899) #8, !dbg !29, !frisk.register_lifetimes !5
  %900 = fadd float %897, %895, !dbg !29
  %901 = extractelement <4 x float> %744, i64 0, !dbg !27
  %902 = tail call float @llvm.amdgcn.exp2.f32(float %901), !dbg !27
  %903 = insertelement <4 x float> poison, float %902, i64 0, !dbg !28
  %904 = extractelement <4 x float> %744, i64 1, !dbg !27
  %905 = tail call float @llvm.amdgcn.exp2.f32(float %904), !dbg !27
  %906 = insertelement <4 x float> %903, float %905, i64 1, !dbg !28
  %907 = fadd float %902, %905, !dbg !29
  %908 = extractelement <4 x float> %744, i64 2, !dbg !27
  %909 = tail call float @llvm.amdgcn.exp2.f32(float %908), !dbg !27
  %910 = insertelement <4 x float> %906, float %909, i64 2, !dbg !28
  %911 = fadd float %909, %907, !dbg !29
  %912 = extractelement <4 x float> %744, i64 3, !dbg !27
  %913 = tail call float @llvm.amdgcn.exp2.f32(float %912), !dbg !27
  %914 = insertelement <4 x float> %910, float %913, i64 3, !dbg !28
  %math.result323 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %914) #8, !dbg !28, !frisk.register_lifetimes !5
  %915 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !28
  %math.packed324 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %915) #8, !dbg !29, !frisk.register_lifetimes !5
  %916 = fadd float %913, %911, !dbg !29
  %917 = extractelement <4 x float> %745, i64 0, !dbg !27
  %918 = tail call float @llvm.amdgcn.exp2.f32(float %917), !dbg !27
  %919 = insertelement <4 x float> poison, float %918, i64 0, !dbg !28
  %920 = fadd float %918, %916, !dbg !29
  %921 = extractelement <4 x float> %745, i64 1, !dbg !27
  %922 = tail call float @llvm.amdgcn.exp2.f32(float %921), !dbg !27
  %923 = insertelement <4 x float> %919, float %922, i64 1, !dbg !28
  %924 = fadd float %922, %920, !dbg !29
  %925 = extractelement <4 x float> %745, i64 2, !dbg !27
  %926 = tail call float @llvm.amdgcn.exp2.f32(float %925), !dbg !27
  %927 = insertelement <4 x float> %923, float %926, i64 2, !dbg !28
  %928 = fadd float %926, %924, !dbg !29
  %929 = extractelement <4 x float> %745, i64 3, !dbg !27
  %930 = tail call float @llvm.amdgcn.exp2.f32(float %929), !dbg !27
  %931 = insertelement <4 x float> %927, float %930, i64 3, !dbg !28
  %math.result325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %931) #8, !dbg !28, !frisk.register_lifetimes !5
  %932 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !28
  %math.packed326 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %932) #8, !dbg !29, !frisk.register_lifetimes !5
  %933 = fadd float %930, %928, !dbg !29
  %934 = extractelement <4 x float> %746, i64 0, !dbg !27
  %935 = tail call float @llvm.amdgcn.exp2.f32(float %934), !dbg !27
  %936 = insertelement <4 x float> poison, float %935, i64 0, !dbg !28
  %937 = fadd float %935, %933, !dbg !29
  %938 = extractelement <4 x float> %746, i64 1, !dbg !27
  %939 = tail call float @llvm.amdgcn.exp2.f32(float %938), !dbg !27
  %940 = insertelement <4 x float> %936, float %939, i64 1, !dbg !28
  %941 = fadd float %939, %937, !dbg !29
  %942 = extractelement <4 x float> %746, i64 2, !dbg !27
  %943 = tail call float @llvm.amdgcn.exp2.f32(float %942), !dbg !27
  %944 = insertelement <4 x float> %940, float %943, i64 2, !dbg !28
  %945 = fadd float %943, %941, !dbg !29
  %946 = extractelement <4 x float> %746, i64 3, !dbg !27
  %947 = tail call float @llvm.amdgcn.exp2.f32(float %946), !dbg !27
  %948 = insertelement <4 x float> %944, float %947, i64 3, !dbg !28
  %math.result327 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %948) #8, !dbg !28, !frisk.register_lifetimes !5
  %949 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !28
  %math.packed328 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %949) #8, !dbg !29, !frisk.register_lifetimes !5
  %950 = fadd float %947, %945, !dbg !29
  %951 = extractelement <4 x float> %747, i64 0, !dbg !27
  %952 = tail call float @llvm.amdgcn.exp2.f32(float %951), !dbg !27
  %953 = insertelement <4 x float> poison, float %952, i64 0, !dbg !28
  %954 = fadd float %952, %950, !dbg !29
  %955 = extractelement <4 x float> %747, i64 1, !dbg !27
  %956 = tail call float @llvm.amdgcn.exp2.f32(float %955), !dbg !27
  %957 = insertelement <4 x float> %953, float %956, i64 1, !dbg !28
  %958 = fadd float %956, %954, !dbg !29
  %959 = extractelement <4 x float> %747, i64 2, !dbg !27
  %960 = tail call float @llvm.amdgcn.exp2.f32(float %959), !dbg !27
  %961 = insertelement <4 x float> %957, float %960, i64 2, !dbg !28
  %962 = fadd float %960, %958, !dbg !29
  %963 = extractelement <4 x float> %747, i64 3, !dbg !27
  %964 = tail call float @llvm.amdgcn.exp2.f32(float %963), !dbg !27
  %965 = insertelement <4 x float> %961, float %964, i64 3, !dbg !28
  %math.result329 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %965) #8, !dbg !28, !frisk.register_lifetimes !5
  %966 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !28
  %math.packed330 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %966) #8, !dbg !29, !frisk.register_lifetimes !5
  %967 = fadd float %964, %962, !dbg !29
  %968 = extractelement <4 x float> %748, i64 0, !dbg !27
  %969 = tail call float @llvm.amdgcn.exp2.f32(float %968), !dbg !27
  %970 = insertelement <4 x float> poison, float %969, i64 0, !dbg !28
  %971 = fadd float %969, %967, !dbg !29
  %972 = extractelement <4 x float> %748, i64 1, !dbg !27
  %973 = tail call float @llvm.amdgcn.exp2.f32(float %972), !dbg !27
  %974 = insertelement <4 x float> %970, float %973, i64 1, !dbg !28
  %975 = fadd float %973, %971, !dbg !29
  %976 = extractelement <4 x float> %748, i64 2, !dbg !27
  %977 = tail call float @llvm.amdgcn.exp2.f32(float %976), !dbg !27
  %978 = insertelement <4 x float> %974, float %977, i64 2, !dbg !28
  %979 = fadd float %977, %975, !dbg !29
  %980 = extractelement <4 x float> %748, i64 3, !dbg !27
  %981 = tail call float @llvm.amdgcn.exp2.f32(float %980), !dbg !27
  %982 = insertelement <4 x float> %978, float %981, i64 3, !dbg !28
  %math.result331 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %982) #8, !dbg !28, !frisk.register_lifetimes !5
  %983 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !28
  %math.packed332 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %983) #8, !dbg !29, !frisk.register_lifetimes !5
  %984 = fadd float %981, %979, !dbg !29
  %985 = extractelement <4 x float> %749, i64 0, !dbg !27
  %986 = tail call float @llvm.amdgcn.exp2.f32(float %985), !dbg !27
  %987 = insertelement <4 x float> poison, float %986, i64 0, !dbg !28
  %988 = fadd float %986, %984, !dbg !29
  %989 = extractelement <4 x float> %749, i64 1, !dbg !27
  %990 = tail call float @llvm.amdgcn.exp2.f32(float %989), !dbg !27
  %991 = insertelement <4 x float> %987, float %990, i64 1, !dbg !28
  %992 = fadd float %990, %988, !dbg !29
  %993 = extractelement <4 x float> %749, i64 2, !dbg !27
  %994 = tail call float @llvm.amdgcn.exp2.f32(float %993), !dbg !27
  %995 = insertelement <4 x float> %991, float %994, i64 2, !dbg !28
  %996 = fadd float %994, %992, !dbg !29
  %997 = extractelement <4 x float> %749, i64 3, !dbg !27
  %998 = tail call float @llvm.amdgcn.exp2.f32(float %997), !dbg !27
  %999 = insertelement <4 x float> %995, float %998, i64 3, !dbg !28
  %math.result333 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %999) #8, !dbg !28, !frisk.register_lifetimes !5
  %1000 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !28
  %math.packed334 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1000) #8, !dbg !29, !frisk.register_lifetimes !5
  %1001 = fadd float %998, %996, !dbg !29
  %1002 = extractelement <4 x float> %750, i64 0, !dbg !27
  %1003 = tail call float @llvm.amdgcn.exp2.f32(float %1002), !dbg !27
  %1004 = insertelement <4 x float> poison, float %1003, i64 0, !dbg !28
  %1005 = fadd float %1003, %1001, !dbg !29
  %1006 = extractelement <4 x float> %750, i64 1, !dbg !27
  %1007 = tail call float @llvm.amdgcn.exp2.f32(float %1006), !dbg !27
  %1008 = insertelement <4 x float> %1004, float %1007, i64 1, !dbg !28
  %1009 = fadd float %1007, %1005, !dbg !29
  %1010 = extractelement <4 x float> %750, i64 2, !dbg !27
  %1011 = tail call float @llvm.amdgcn.exp2.f32(float %1010), !dbg !27
  %1012 = insertelement <4 x float> %1008, float %1011, i64 2, !dbg !28
  %1013 = fadd float %1011, %1009, !dbg !29
  %1014 = extractelement <4 x float> %750, i64 3, !dbg !27
  %1015 = tail call float @llvm.amdgcn.exp2.f32(float %1014), !dbg !27
  %1016 = insertelement <4 x float> %1012, float %1015, i64 3, !dbg !28
  %math.result335 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1016) #8, !dbg !28, !frisk.register_lifetimes !5
  %1017 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !28
  %math.packed336 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1017) #8, !dbg !29, !frisk.register_lifetimes !5
  %1018 = fadd float %1015, %1013, !dbg !29
  %1019 = extractelement <4 x float> %751, i64 0, !dbg !27
  %1020 = tail call float @llvm.amdgcn.exp2.f32(float %1019), !dbg !27
  %1021 = insertelement <4 x float> poison, float %1020, i64 0, !dbg !28
  %1022 = fadd float %1020, %1018, !dbg !29
  %1023 = extractelement <4 x float> %751, i64 1, !dbg !27
  %1024 = tail call float @llvm.amdgcn.exp2.f32(float %1023), !dbg !27
  %1025 = insertelement <4 x float> %1021, float %1024, i64 1, !dbg !28
  %1026 = fadd float %1024, %1022, !dbg !29
  %1027 = extractelement <4 x float> %751, i64 2, !dbg !27
  %1028 = tail call float @llvm.amdgcn.exp2.f32(float %1027), !dbg !27
  %1029 = insertelement <4 x float> %1025, float %1028, i64 2, !dbg !28
  %1030 = fadd float %1028, %1026, !dbg !29
  %1031 = extractelement <4 x float> %751, i64 3, !dbg !27
  %1032 = tail call float @llvm.amdgcn.exp2.f32(float %1031), !dbg !27
  %1033 = insertelement <4 x float> %1029, float %1032, i64 3, !dbg !28
  %math.result337 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1033) #8, !dbg !28, !frisk.register_lifetimes !5
  %1034 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !28
  %math.packed338 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1034) #8, !dbg !29, !frisk.register_lifetimes !5
  %1035 = fadd float %1032, %1030, !dbg !29
  %1036 = bitcast float %900 to i32, !dbg !29
  %1037 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1036), !dbg !29
  %1038 = bitcast i32 %1037 to float, !dbg !29
  %1039 = fadd float %900, %1038, !dbg !29
  %1040 = bitcast float %1039 to i32, !dbg !29
  %1041 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1040), !dbg !29
  %1042 = bitcast i32 %1041 to float, !dbg !29
  %1043 = fadd float %1039, %1042, !dbg !29
  %1044 = fadd float %758, %1043, !dbg !29
  %1045 = insertelement <1 x float> poison, float %1044, i64 0, !dbg !11
  %1046 = insertvalue [2 x <1 x float>] poison, <1 x float> %1045, 0, !dbg !11
  %1047 = bitcast float %1035 to i32, !dbg !29
  %1048 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1047), !dbg !29
  %1049 = bitcast i32 %1048 to float, !dbg !29
  %1050 = fadd float %1035, %1049, !dbg !29
  %1051 = bitcast float %1050 to i32, !dbg !29
  %1052 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1051), !dbg !29
  %1053 = bitcast i32 %1052 to float, !dbg !29
  %1054 = fadd float %1050, %1053, !dbg !29
  %1055 = fadd float %765, %1054, !dbg !29
  %1056 = insertelement <1 x float> poison, float %1055, i64 0, !dbg !11
  %1057 = insertvalue [2 x <1 x float>] %1046, <1 x float> %1056, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread864 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local865 = shl nuw nsw i32 %address.thread864, 3, !dbg !13
  %address.local866 = and i32 %address.local865, 120, !dbg !13
  %address.local867 = shl nuw nsw i32 %address.thread864, 5, !dbg !13
  %address.local868 = and i32 %address.local867, 1536, !dbg !13
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !13
  %address.local870 = zext nneg i32 %address.local869 to i64, !dbg !13
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !13
  %1058 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !13
  %1059 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !13
  %address.local872 = zext nneg i32 %address.thread864 to i64, !dbg !8
  %address.local873 = and i64 %address.local872, 60, !dbg !13
  %address.local874 = add nuw nsw i64 %address.local873, %address.local872, !dbg !13
  %address.local876.idx = shl nuw nsw i64 %address.local874, 4, !dbg !13
  %address.local876 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local876.idx, !dbg !13
  %address.local877 = lshr i64 %address.local872, 4, !dbg !13
  %address.local878 = and i64 %address.local877, 3, !dbg !13
  %address.local880.idx = mul nuw nsw i64 %address.local878, 768, !dbg !13
  %address.local880 = getelementptr i8, ptr addrspace(3) %address.local876, i64 %address.local880.idx, !dbg !13
  %address.local881 = lshr i64 %address.local872, 2, !dbg !16
  %address.local883.idx = mul nsw i64 %address.local881, -64, !dbg !13
  %address.local883 = getelementptr i8, ptr addrspace(3) %address.local880, i64 %address.local883.idx, !dbg !13
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 256, !dbg !13
  %1060 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !13
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !13
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local883, i64 512, !dbg !13
  %1062 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !13
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !13
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local883, i64 768, !dbg !13
  %1064 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !13
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !13
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local871, i64 4096, !dbg !13
  %1066 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !13
  %1067 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !13
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4352, !dbg !13
  %1068 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !13
  %1069 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !13
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4608, !dbg !13
  %1070 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !13
  %1071 = bitcast <4 x i32> %1070 to <8 x half>, !dbg !13
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4864, !dbg !13
  %1072 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !13
  %1073 = bitcast <4 x i32> %1072 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !8
  %address.local908 = and i64 %address.local907, 1536, !dbg !8
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !8
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !8
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !8
  %1074 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !18
  %address.local913 = and i64 %address.local912, 8064, !dbg !18
  %address.local915 = and i64 %address.local907, 24, !dbg !18
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !18
  %1075 = or disjoint i64 %.idx5760, %address.local916, !dbg !18
  %1076 = trunc i64 %1075 to i32, !dbg !18
  %.tr68 = shl i32 %1076, 1, !dbg !18
  %dma.buffer.offset1558 = or i32 %.tr68, 64, !dbg !18
  %dma.buffer.lds1560 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1074), !dbg !18
  %1077 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !8
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !8
  %address.local920 = and i64 %address.local919, 3584, !dbg !8
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !8
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !8
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !8
  %1078 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !18
  %address.local925 = and i64 %address.local924, 16256, !dbg !18
  %address.local928 = or disjoint i64 %address.local925, %address.local915, !dbg !18
  %1079 = or disjoint i64 %.idx5760, %address.local928, !dbg !18
  %1080 = trunc i64 %1079 to i32, !dbg !18
  %.tr72 = shl i32 %1080, 1, !dbg !18
  %dma.buffer.offset1555 = or i32 %.tr72, 64, !dbg !18
  %dma.buffer.lds1557 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1078), !dbg !18
  %1081 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %1082 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1083 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1082, <4 x float> zeroinitializer) #8, !dbg !13
  %1084 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1085 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1084, <4 x float> %1083) #8, !dbg !13
  %1086 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1087 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1086, <4 x float> %1085) #8, !dbg !13
  %1088 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1089 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1088, <4 x float> %1087) #8, !dbg !13
  %1090 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1091 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1090, <4 x float> %1089) #8, !dbg !13
  %1092 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1093 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1092, <4 x float> %1091) #8, !dbg !13
  %1094 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1095 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1094, <4 x float> %1093) #8, !dbg !13
  %1096 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1097 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1096, <4 x float> %1095) #8, !dbg !13
  %1098 = extractelement <4 x float> %1097, i64 0, !dbg !13
  %1099 = extractelement <4 x float> %1097, i64 1, !dbg !13
  %1100 = extractelement <4 x float> %1097, i64 2, !dbg !13
  %1101 = extractelement <4 x float> %1097, i64 3, !dbg !13
  %1102 = select i1 %.not, float %1099, float %1098, !dbg !13
  %1103 = bitcast float %1102 to i32, !dbg !13
  %1104 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1103), !dbg !13
  %1105 = bitcast i32 %1104 to float, !dbg !13
  %1106 = select i1 %.not, float %1098, float %1105, !dbg !13
  %1107 = select i1 %.not, float %1105, float %1099, !dbg !13
  %1108 = select i1 %.not, float %1101, float %1100, !dbg !13
  %1109 = bitcast float %1108 to i32, !dbg !13
  %1110 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1109), !dbg !13
  %1111 = bitcast i32 %1110 to float, !dbg !13
  %1112 = select i1 %.not, float %1100, float %1111, !dbg !13
  %1113 = select i1 %.not, float %1111, float %1101, !dbg !13
  %1114 = select i1 %.not1, float %1112, float %1106, !dbg !13
  %1115 = bitcast float %1114 to i32, !dbg !13
  %1116 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1115), !dbg !13
  %1117 = bitcast i32 %1116 to float, !dbg !13
  %1118 = select i1 %.not1, float %1113, float %1107, !dbg !13
  %1119 = bitcast float %1118 to i32, !dbg !13
  %1120 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1119), !dbg !13
  %1121 = bitcast i32 %1120 to float, !dbg !13
  %1122 = insertelement <4 x float> poison, float %1106, i64 0, !dbg !13
  %1123 = insertelement <4 x float> %1122, float %1107, i64 1, !dbg !13
  %1124 = insertelement <4 x float> %1123, float %1117, i64 2, !dbg !13
  %1125 = insertelement <4 x float> %1124, float %1121, i64 3, !dbg !13
  %1126 = insertelement <4 x float> poison, float %1117, i64 0, !dbg !13
  %1127 = insertelement <4 x float> %1126, float %1121, i64 1, !dbg !13
  %1128 = insertelement <4 x float> %1127, float %1112, i64 2, !dbg !13
  %1129 = insertelement <4 x float> %1128, float %1113, i64 3, !dbg !13
  %1130 = select <4 x i1> %136, <4 x float> %1125, <4 x float> %1129, !dbg !13
  %1131 = extractvalue [2 x <8 x float>] %146, 0, !dbg !25
  %1132 = shufflevector <8 x float> %1131, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1133 = fmul <4 x float> %1132, %755, !dbg !25
  %1134 = fadd <4 x float> %1133, %1130, !dbg !30
  %1135 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1136 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1135, <4 x float> zeroinitializer) #8, !dbg !13
  %1137 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1138 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1137, <4 x float> %1136) #8, !dbg !13
  %1139 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1140 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1139, <4 x float> %1138) #8, !dbg !13
  %1141 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1142 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1141, <4 x float> %1140) #8, !dbg !13
  %1143 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1144 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1143, <4 x float> %1142) #8, !dbg !13
  %1145 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1145, <4 x float> %1144) #8, !dbg !13
  %1147 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1147, <4 x float> %1146) #8, !dbg !13
  %1149 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1149, <4 x float> %1148) #8, !dbg !13
  %1151 = extractelement <4 x float> %1150, i64 0, !dbg !13
  %1152 = extractelement <4 x float> %1150, i64 1, !dbg !13
  %1153 = extractelement <4 x float> %1150, i64 2, !dbg !13
  %1154 = extractelement <4 x float> %1150, i64 3, !dbg !13
  %1155 = select i1 %.not, float %1152, float %1151, !dbg !13
  %1156 = bitcast float %1155 to i32, !dbg !13
  %1157 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1156), !dbg !13
  %1158 = bitcast i32 %1157 to float, !dbg !13
  %1159 = select i1 %.not, float %1151, float %1158, !dbg !13
  %1160 = select i1 %.not, float %1158, float %1152, !dbg !13
  %1161 = select i1 %.not, float %1154, float %1153, !dbg !13
  %1162 = bitcast float %1161 to i32, !dbg !13
  %1163 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1162), !dbg !13
  %1164 = bitcast i32 %1163 to float, !dbg !13
  %1165 = select i1 %.not, float %1153, float %1164, !dbg !13
  %1166 = select i1 %.not, float %1164, float %1154, !dbg !13
  %1167 = select i1 %.not1, float %1165, float %1159, !dbg !13
  %1168 = bitcast float %1167 to i32, !dbg !13
  %1169 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1168), !dbg !13
  %1170 = bitcast i32 %1169 to float, !dbg !13
  %1171 = select i1 %.not1, float %1166, float %1160, !dbg !13
  %1172 = bitcast float %1171 to i32, !dbg !13
  %1173 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1172), !dbg !13
  %1174 = bitcast i32 %1173 to float, !dbg !13
  %1175 = insertelement <4 x float> poison, float %1159, i64 0, !dbg !13
  %1176 = insertelement <4 x float> %1175, float %1160, i64 1, !dbg !13
  %1177 = insertelement <4 x float> %1176, float %1170, i64 2, !dbg !13
  %1178 = insertelement <4 x float> %1177, float %1174, i64 3, !dbg !13
  %1179 = insertelement <4 x float> poison, float %1170, i64 0, !dbg !13
  %1180 = insertelement <4 x float> %1179, float %1174, i64 1, !dbg !13
  %1181 = insertelement <4 x float> %1180, float %1165, i64 2, !dbg !13
  %1182 = insertelement <4 x float> %1181, float %1166, i64 3, !dbg !13
  %1183 = select <4 x i1> %136, <4 x float> %1178, <4 x float> %1182, !dbg !13
  %1184 = shufflevector <8 x float> %1131, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1185 = fmul <4 x float> %1184, %755, !dbg !25
  %1186 = fadd <4 x float> %1185, %1183, !dbg !30
  %1187 = shufflevector <4 x float> %1134, <4 x float> %1186, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1188 = insertvalue [2 x <8 x float>] poison, <8 x float> %1187, 0, !dbg !11
  %1189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1082, <4 x float> zeroinitializer) #8, !dbg !13
  %1190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1084, <4 x float> %1189) #8, !dbg !13
  %1191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1086, <4 x float> %1190) #8, !dbg !13
  %1192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1088, <4 x float> %1191) #8, !dbg !13
  %1193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1090, <4 x float> %1192) #8, !dbg !13
  %1194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1092, <4 x float> %1193) #8, !dbg !13
  %1195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1094, <4 x float> %1194) #8, !dbg !13
  %1196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1096, <4 x float> %1195) #8, !dbg !13
  %1197 = extractelement <4 x float> %1196, i64 0, !dbg !13
  %1198 = extractelement <4 x float> %1196, i64 1, !dbg !13
  %1199 = extractelement <4 x float> %1196, i64 2, !dbg !13
  %1200 = extractelement <4 x float> %1196, i64 3, !dbg !13
  %1201 = select i1 %.not, float %1198, float %1197, !dbg !13
  %1202 = bitcast float %1201 to i32, !dbg !13
  %1203 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1202), !dbg !13
  %1204 = bitcast i32 %1203 to float, !dbg !13
  %1205 = select i1 %.not, float %1197, float %1204, !dbg !13
  %1206 = select i1 %.not, float %1204, float %1198, !dbg !13
  %1207 = select i1 %.not, float %1200, float %1199, !dbg !13
  %1208 = bitcast float %1207 to i32, !dbg !13
  %1209 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1208), !dbg !13
  %1210 = bitcast i32 %1209 to float, !dbg !13
  %1211 = select i1 %.not, float %1199, float %1210, !dbg !13
  %1212 = select i1 %.not, float %1210, float %1200, !dbg !13
  %1213 = select i1 %.not1, float %1211, float %1205, !dbg !13
  %1214 = bitcast float %1213 to i32, !dbg !13
  %1215 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1214), !dbg !13
  %1216 = bitcast i32 %1215 to float, !dbg !13
  %1217 = select i1 %.not1, float %1212, float %1206, !dbg !13
  %1218 = bitcast float %1217 to i32, !dbg !13
  %1219 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1218), !dbg !13
  %1220 = bitcast i32 %1219 to float, !dbg !13
  %1221 = insertelement <4 x float> poison, float %1205, i64 0, !dbg !13
  %1222 = insertelement <4 x float> %1221, float %1206, i64 1, !dbg !13
  %1223 = insertelement <4 x float> %1222, float %1216, i64 2, !dbg !13
  %1224 = insertelement <4 x float> %1223, float %1220, i64 3, !dbg !13
  %1225 = insertelement <4 x float> poison, float %1216, i64 0, !dbg !13
  %1226 = insertelement <4 x float> %1225, float %1220, i64 1, !dbg !13
  %1227 = insertelement <4 x float> %1226, float %1211, i64 2, !dbg !13
  %1228 = insertelement <4 x float> %1227, float %1212, i64 3, !dbg !13
  %1229 = select <4 x i1> %136, <4 x float> %1224, <4 x float> %1228, !dbg !13
  %1230 = extractvalue [2 x <8 x float>] %146, 1, !dbg !25
  %1231 = shufflevector <8 x float> %1230, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1232 = fmul <4 x float> %1231, %762, !dbg !25
  %1233 = fadd <4 x float> %1232, %1229, !dbg !30
  %1234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1135, <4 x float> zeroinitializer) #8, !dbg !13
  %1235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1137, <4 x float> %1234) #8, !dbg !13
  %1236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1139, <4 x float> %1235) #8, !dbg !13
  %1237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1141, <4 x float> %1236) #8, !dbg !13
  %1238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1143, <4 x float> %1237) #8, !dbg !13
  %1239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1145, <4 x float> %1238) #8, !dbg !13
  %1240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1147, <4 x float> %1239) #8, !dbg !13
  %1241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1149, <4 x float> %1240) #8, !dbg !13
  %1242 = extractelement <4 x float> %1241, i64 0, !dbg !13
  %1243 = extractelement <4 x float> %1241, i64 1, !dbg !13
  %1244 = extractelement <4 x float> %1241, i64 2, !dbg !13
  %1245 = extractelement <4 x float> %1241, i64 3, !dbg !13
  %1246 = select i1 %.not, float %1243, float %1242, !dbg !13
  %1247 = bitcast float %1246 to i32, !dbg !13
  %1248 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1247), !dbg !13
  %1249 = bitcast i32 %1248 to float, !dbg !13
  %1250 = select i1 %.not, float %1242, float %1249, !dbg !13
  %1251 = select i1 %.not, float %1249, float %1243, !dbg !13
  %1252 = select i1 %.not, float %1245, float %1244, !dbg !13
  %1253 = bitcast float %1252 to i32, !dbg !13
  %1254 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1253), !dbg !13
  %1255 = bitcast i32 %1254 to float, !dbg !13
  %1256 = select i1 %.not, float %1244, float %1255, !dbg !13
  %1257 = select i1 %.not, float %1255, float %1245, !dbg !13
  %1258 = select i1 %.not1, float %1256, float %1250, !dbg !13
  %1259 = bitcast float %1258 to i32, !dbg !13
  %1260 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1259), !dbg !13
  %1261 = bitcast i32 %1260 to float, !dbg !13
  %1262 = select i1 %.not1, float %1257, float %1251, !dbg !13
  %1263 = bitcast float %1262 to i32, !dbg !13
  %1264 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1263), !dbg !13
  %1265 = bitcast i32 %1264 to float, !dbg !13
  %1266 = insertelement <4 x float> poison, float %1250, i64 0, !dbg !13
  %1267 = insertelement <4 x float> %1266, float %1251, i64 1, !dbg !13
  %1268 = insertelement <4 x float> %1267, float %1261, i64 2, !dbg !13
  %1269 = insertelement <4 x float> %1268, float %1265, i64 3, !dbg !13
  %1270 = insertelement <4 x float> poison, float %1261, i64 0, !dbg !13
  %1271 = insertelement <4 x float> %1270, float %1265, i64 1, !dbg !13
  %1272 = insertelement <4 x float> %1271, float %1256, i64 2, !dbg !13
  %1273 = insertelement <4 x float> %1272, float %1257, i64 3, !dbg !13
  %1274 = select <4 x i1> %136, <4 x float> %1269, <4 x float> %1273, !dbg !13
  %1275 = shufflevector <8 x float> %1230, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1276 = fmul <4 x float> %1275, %762, !dbg !25
  %1277 = fadd <4 x float> %1276, %1274, !dbg !30
  %1278 = shufflevector <4 x float> %1233, <4 x float> %1277, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1279 = insertvalue [2 x <8 x float>] %1188, <8 x float> %1278, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1280 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !13
  %1281 = bitcast <4 x i32> %1280 to <8 x half>, !dbg !13
  %1282 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !13
  %1283 = bitcast <4 x i32> %1282 to <8 x half>, !dbg !13
  %1284 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !13
  %1285 = bitcast <4 x i32> %1284 to <8 x half>, !dbg !13
  %1286 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !13
  %1287 = bitcast <4 x i32> %1286 to <8 x half>, !dbg !13
  %1288 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !13
  %1289 = bitcast <4 x i32> %1288 to <8 x half>, !dbg !13
  %1290 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !13
  %1291 = bitcast <4 x i32> %1290 to <8 x half>, !dbg !13
  %1292 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !13
  %1293 = bitcast <4 x i32> %1292 to <8 x half>, !dbg !13
  %1294 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !13
  %1295 = bitcast <4 x i32> %1294 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %dma.buffer.offset1552 = or i32 %.tr68, 128, !dbg !18
  %1296 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %dma.buffer.offset1549 = or i32 %.tr72, 128, !dbg !18
  %1297 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %1298 = shufflevector <8 x half> %1281, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1298, <4 x float> zeroinitializer) #8, !dbg !13
  %1300 = shufflevector <8 x half> %1283, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1300, <4 x float> %1299) #8, !dbg !13
  %1302 = shufflevector <8 x half> %1285, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1302, <4 x float> %1301) #8, !dbg !13
  %1304 = shufflevector <8 x half> %1287, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1304, <4 x float> %1303) #8, !dbg !13
  %1306 = shufflevector <8 x half> %1289, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1306, <4 x float> %1305) #8, !dbg !13
  %1308 = shufflevector <8 x half> %1291, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1308, <4 x float> %1307) #8, !dbg !13
  %1310 = shufflevector <8 x half> %1293, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1310, <4 x float> %1309) #8, !dbg !13
  %1312 = shufflevector <8 x half> %1295, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1312, <4 x float> %1311) #8, !dbg !13
  %1314 = extractelement <4 x float> %1313, i64 0, !dbg !13
  %1315 = extractelement <4 x float> %1313, i64 1, !dbg !13
  %1316 = extractelement <4 x float> %1313, i64 2, !dbg !13
  %1317 = extractelement <4 x float> %1313, i64 3, !dbg !13
  %1318 = select i1 %.not, float %1315, float %1314, !dbg !13
  %1319 = bitcast float %1318 to i32, !dbg !13
  %1320 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1319), !dbg !13
  %1321 = bitcast i32 %1320 to float, !dbg !13
  %1322 = select i1 %.not, float %1314, float %1321, !dbg !13
  %1323 = select i1 %.not, float %1321, float %1315, !dbg !13
  %1324 = select i1 %.not, float %1317, float %1316, !dbg !13
  %1325 = bitcast float %1324 to i32, !dbg !13
  %1326 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1325), !dbg !13
  %1327 = bitcast i32 %1326 to float, !dbg !13
  %1328 = select i1 %.not, float %1316, float %1327, !dbg !13
  %1329 = select i1 %.not, float %1327, float %1317, !dbg !13
  %1330 = select i1 %.not1, float %1328, float %1322, !dbg !13
  %1331 = bitcast float %1330 to i32, !dbg !13
  %1332 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1331), !dbg !13
  %1333 = bitcast i32 %1332 to float, !dbg !13
  %1334 = select i1 %.not1, float %1329, float %1323, !dbg !13
  %1335 = bitcast float %1334 to i32, !dbg !13
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1335), !dbg !13
  %1337 = bitcast i32 %1336 to float, !dbg !13
  %1338 = insertelement <4 x float> poison, float %1322, i64 0, !dbg !13
  %1339 = insertelement <4 x float> %1338, float %1323, i64 1, !dbg !13
  %1340 = insertelement <4 x float> %1339, float %1333, i64 2, !dbg !13
  %1341 = insertelement <4 x float> %1340, float %1337, i64 3, !dbg !13
  %1342 = insertelement <4 x float> poison, float %1333, i64 0, !dbg !13
  %1343 = insertelement <4 x float> %1342, float %1337, i64 1, !dbg !13
  %1344 = insertelement <4 x float> %1343, float %1328, i64 2, !dbg !13
  %1345 = insertelement <4 x float> %1344, float %1329, i64 3, !dbg !13
  %1346 = select <4 x i1> %136, <4 x float> %1341, <4 x float> %1345, !dbg !13
  %1347 = extractvalue [2 x <8 x float>] %144, 0, !dbg !25
  %1348 = shufflevector <8 x float> %1347, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1349 = fmul <4 x float> %1348, %755, !dbg !25
  %1350 = fadd <4 x float> %1349, %1346, !dbg !30
  %1351 = shufflevector <8 x half> %1281, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1351, <4 x float> zeroinitializer) #8, !dbg !13
  %1353 = shufflevector <8 x half> %1283, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1353, <4 x float> %1352) #8, !dbg !13
  %1355 = shufflevector <8 x half> %1285, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1355, <4 x float> %1354) #8, !dbg !13
  %1357 = shufflevector <8 x half> %1287, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1357, <4 x float> %1356) #8, !dbg !13
  %1359 = shufflevector <8 x half> %1289, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1359, <4 x float> %1358) #8, !dbg !13
  %1361 = shufflevector <8 x half> %1291, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1361, <4 x float> %1360) #8, !dbg !13
  %1363 = shufflevector <8 x half> %1293, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1363, <4 x float> %1362) #8, !dbg !13
  %1365 = shufflevector <8 x half> %1295, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1365, <4 x float> %1364) #8, !dbg !13
  %1367 = extractelement <4 x float> %1366, i64 0, !dbg !13
  %1368 = extractelement <4 x float> %1366, i64 1, !dbg !13
  %1369 = extractelement <4 x float> %1366, i64 2, !dbg !13
  %1370 = extractelement <4 x float> %1366, i64 3, !dbg !13
  %1371 = select i1 %.not, float %1368, float %1367, !dbg !13
  %1372 = bitcast float %1371 to i32, !dbg !13
  %1373 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1372), !dbg !13
  %1374 = bitcast i32 %1373 to float, !dbg !13
  %1375 = select i1 %.not, float %1367, float %1374, !dbg !13
  %1376 = select i1 %.not, float %1374, float %1368, !dbg !13
  %1377 = select i1 %.not, float %1370, float %1369, !dbg !13
  %1378 = bitcast float %1377 to i32, !dbg !13
  %1379 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1378), !dbg !13
  %1380 = bitcast i32 %1379 to float, !dbg !13
  %1381 = select i1 %.not, float %1369, float %1380, !dbg !13
  %1382 = select i1 %.not, float %1380, float %1370, !dbg !13
  %1383 = select i1 %.not1, float %1381, float %1375, !dbg !13
  %1384 = bitcast float %1383 to i32, !dbg !13
  %1385 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1384), !dbg !13
  %1386 = bitcast i32 %1385 to float, !dbg !13
  %1387 = select i1 %.not1, float %1382, float %1376, !dbg !13
  %1388 = bitcast float %1387 to i32, !dbg !13
  %1389 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1388), !dbg !13
  %1390 = bitcast i32 %1389 to float, !dbg !13
  %1391 = insertelement <4 x float> poison, float %1375, i64 0, !dbg !13
  %1392 = insertelement <4 x float> %1391, float %1376, i64 1, !dbg !13
  %1393 = insertelement <4 x float> %1392, float %1386, i64 2, !dbg !13
  %1394 = insertelement <4 x float> %1393, float %1390, i64 3, !dbg !13
  %1395 = insertelement <4 x float> poison, float %1386, i64 0, !dbg !13
  %1396 = insertelement <4 x float> %1395, float %1390, i64 1, !dbg !13
  %1397 = insertelement <4 x float> %1396, float %1381, i64 2, !dbg !13
  %1398 = insertelement <4 x float> %1397, float %1382, i64 3, !dbg !13
  %1399 = select <4 x i1> %136, <4 x float> %1394, <4 x float> %1398, !dbg !13
  %1400 = shufflevector <8 x float> %1347, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1401 = fmul <4 x float> %1400, %755, !dbg !25
  %1402 = fadd <4 x float> %1401, %1399, !dbg !30
  %1403 = shufflevector <4 x float> %1350, <4 x float> %1402, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1404 = insertvalue [2 x <8 x float>] poison, <8 x float> %1403, 0, !dbg !11
  %1405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1298, <4 x float> zeroinitializer) #8, !dbg !13
  %1406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1300, <4 x float> %1405) #8, !dbg !13
  %1407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1302, <4 x float> %1406) #8, !dbg !13
  %1408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1304, <4 x float> %1407) #8, !dbg !13
  %1409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1306, <4 x float> %1408) #8, !dbg !13
  %1410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1308, <4 x float> %1409) #8, !dbg !13
  %1411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1310, <4 x float> %1410) #8, !dbg !13
  %1412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1312, <4 x float> %1411) #8, !dbg !13
  %1413 = extractelement <4 x float> %1412, i64 0, !dbg !13
  %1414 = extractelement <4 x float> %1412, i64 1, !dbg !13
  %1415 = extractelement <4 x float> %1412, i64 2, !dbg !13
  %1416 = extractelement <4 x float> %1412, i64 3, !dbg !13
  %1417 = select i1 %.not, float %1414, float %1413, !dbg !13
  %1418 = bitcast float %1417 to i32, !dbg !13
  %1419 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1418), !dbg !13
  %1420 = bitcast i32 %1419 to float, !dbg !13
  %1421 = select i1 %.not, float %1413, float %1420, !dbg !13
  %1422 = select i1 %.not, float %1420, float %1414, !dbg !13
  %1423 = select i1 %.not, float %1416, float %1415, !dbg !13
  %1424 = bitcast float %1423 to i32, !dbg !13
  %1425 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1424), !dbg !13
  %1426 = bitcast i32 %1425 to float, !dbg !13
  %1427 = select i1 %.not, float %1415, float %1426, !dbg !13
  %1428 = select i1 %.not, float %1426, float %1416, !dbg !13
  %1429 = select i1 %.not1, float %1427, float %1421, !dbg !13
  %1430 = bitcast float %1429 to i32, !dbg !13
  %1431 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1430), !dbg !13
  %1432 = bitcast i32 %1431 to float, !dbg !13
  %1433 = select i1 %.not1, float %1428, float %1422, !dbg !13
  %1434 = bitcast float %1433 to i32, !dbg !13
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1434), !dbg !13
  %1436 = bitcast i32 %1435 to float, !dbg !13
  %1437 = insertelement <4 x float> poison, float %1421, i64 0, !dbg !13
  %1438 = insertelement <4 x float> %1437, float %1422, i64 1, !dbg !13
  %1439 = insertelement <4 x float> %1438, float %1432, i64 2, !dbg !13
  %1440 = insertelement <4 x float> %1439, float %1436, i64 3, !dbg !13
  %1441 = insertelement <4 x float> poison, float %1432, i64 0, !dbg !13
  %1442 = insertelement <4 x float> %1441, float %1436, i64 1, !dbg !13
  %1443 = insertelement <4 x float> %1442, float %1427, i64 2, !dbg !13
  %1444 = insertelement <4 x float> %1443, float %1428, i64 3, !dbg !13
  %1445 = select <4 x i1> %136, <4 x float> %1440, <4 x float> %1444, !dbg !13
  %1446 = extractvalue [2 x <8 x float>] %144, 1, !dbg !25
  %1447 = shufflevector <8 x float> %1446, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1448 = fmul <4 x float> %1447, %762, !dbg !25
  %1449 = fadd <4 x float> %1448, %1445, !dbg !30
  %1450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1351, <4 x float> zeroinitializer) #8, !dbg !13
  %1451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1353, <4 x float> %1450) #8, !dbg !13
  %1452 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1355, <4 x float> %1451) #8, !dbg !13
  %1453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1357, <4 x float> %1452) #8, !dbg !13
  %1454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1359, <4 x float> %1453) #8, !dbg !13
  %1455 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1361, <4 x float> %1454) #8, !dbg !13
  %1456 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1363, <4 x float> %1455) #8, !dbg !13
  %1457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1365, <4 x float> %1456) #8, !dbg !13
  %1458 = extractelement <4 x float> %1457, i64 0, !dbg !13
  %1459 = extractelement <4 x float> %1457, i64 1, !dbg !13
  %1460 = extractelement <4 x float> %1457, i64 2, !dbg !13
  %1461 = extractelement <4 x float> %1457, i64 3, !dbg !13
  %1462 = select i1 %.not, float %1459, float %1458, !dbg !13
  %1463 = bitcast float %1462 to i32, !dbg !13
  %1464 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1463), !dbg !13
  %1465 = bitcast i32 %1464 to float, !dbg !13
  %1466 = select i1 %.not, float %1458, float %1465, !dbg !13
  %1467 = select i1 %.not, float %1465, float %1459, !dbg !13
  %1468 = select i1 %.not, float %1461, float %1460, !dbg !13
  %1469 = bitcast float %1468 to i32, !dbg !13
  %1470 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1469), !dbg !13
  %1471 = bitcast i32 %1470 to float, !dbg !13
  %1472 = select i1 %.not, float %1460, float %1471, !dbg !13
  %1473 = select i1 %.not, float %1471, float %1461, !dbg !13
  %1474 = select i1 %.not1, float %1472, float %1466, !dbg !13
  %1475 = bitcast float %1474 to i32, !dbg !13
  %1476 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1475), !dbg !13
  %1477 = bitcast i32 %1476 to float, !dbg !13
  %1478 = select i1 %.not1, float %1473, float %1467, !dbg !13
  %1479 = bitcast float %1478 to i32, !dbg !13
  %1480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1479), !dbg !13
  %1481 = bitcast i32 %1480 to float, !dbg !13
  %1482 = insertelement <4 x float> poison, float %1466, i64 0, !dbg !13
  %1483 = insertelement <4 x float> %1482, float %1467, i64 1, !dbg !13
  %1484 = insertelement <4 x float> %1483, float %1477, i64 2, !dbg !13
  %1485 = insertelement <4 x float> %1484, float %1481, i64 3, !dbg !13
  %1486 = insertelement <4 x float> poison, float %1477, i64 0, !dbg !13
  %1487 = insertelement <4 x float> %1486, float %1481, i64 1, !dbg !13
  %1488 = insertelement <4 x float> %1487, float %1472, i64 2, !dbg !13
  %1489 = insertelement <4 x float> %1488, float %1473, i64 3, !dbg !13
  %1490 = select <4 x i1> %136, <4 x float> %1485, <4 x float> %1489, !dbg !13
  %1491 = shufflevector <8 x float> %1446, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1492 = fmul <4 x float> %1491, %762, !dbg !25
  %1493 = fadd <4 x float> %1492, %1490, !dbg !30
  %1494 = shufflevector <4 x float> %1449, <4 x float> %1493, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1495 = insertvalue [2 x <8 x float>] %1404, <8 x float> %1494, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1496 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !13
  %1497 = bitcast <4 x i32> %1496 to <8 x half>, !dbg !13
  %1498 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !13
  %1499 = bitcast <4 x i32> %1498 to <8 x half>, !dbg !13
  %1500 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !13
  %1501 = bitcast <4 x i32> %1500 to <8 x half>, !dbg !13
  %1502 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !13
  %1503 = bitcast <4 x i32> %1502 to <8 x half>, !dbg !13
  %1504 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !13
  %1505 = bitcast <4 x i32> %1504 to <8 x half>, !dbg !13
  %1506 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !13
  %1507 = bitcast <4 x i32> %1506 to <8 x half>, !dbg !13
  %1508 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !13
  %1509 = bitcast <4 x i32> %1508 to <8 x half>, !dbg !13
  %1510 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !13
  %1511 = bitcast <4 x i32> %1510 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1512 = or disjoint i64 %.idx5760, 96, !dbg !18
  %1513 = or disjoint i64 %address.local916, %1512, !dbg !18
  %.tr84 = trunc i64 %1513 to i32, !dbg !18
  %dma.buffer.offset1546 = shl i32 %.tr84, 1, !dbg !18
  %1514 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %1515 = or disjoint i64 %address.local928, %1512, !dbg !18
  %.tr88 = trunc i64 %1515 to i32, !dbg !18
  %dma.buffer.offset = shl i32 %.tr88, 1, !dbg !18
  %1516 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %1517 = shufflevector <8 x half> %1497, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1518 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1517, <4 x float> zeroinitializer) #8, !dbg !13
  %1519 = shufflevector <8 x half> %1499, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1520 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1519, <4 x float> %1518) #8, !dbg !13
  %1521 = shufflevector <8 x half> %1501, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1522 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1521, <4 x float> %1520) #8, !dbg !13
  %1523 = shufflevector <8 x half> %1503, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1524 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1523, <4 x float> %1522) #8, !dbg !13
  %1525 = shufflevector <8 x half> %1505, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1526 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1525, <4 x float> %1524) #8, !dbg !13
  %1527 = shufflevector <8 x half> %1507, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1528 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1527, <4 x float> %1526) #8, !dbg !13
  %1529 = shufflevector <8 x half> %1509, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1530 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1529, <4 x float> %1528) #8, !dbg !13
  %1531 = shufflevector <8 x half> %1511, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1532 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1531, <4 x float> %1530) #8, !dbg !13
  %1533 = extractelement <4 x float> %1532, i64 0, !dbg !13
  %1534 = extractelement <4 x float> %1532, i64 1, !dbg !13
  %1535 = extractelement <4 x float> %1532, i64 2, !dbg !13
  %1536 = extractelement <4 x float> %1532, i64 3, !dbg !13
  %1537 = select i1 %.not, float %1534, float %1533, !dbg !13
  %1538 = bitcast float %1537 to i32, !dbg !13
  %1539 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1538), !dbg !13
  %1540 = bitcast i32 %1539 to float, !dbg !13
  %1541 = select i1 %.not, float %1533, float %1540, !dbg !13
  %1542 = select i1 %.not, float %1540, float %1534, !dbg !13
  %1543 = select i1 %.not, float %1536, float %1535, !dbg !13
  %1544 = bitcast float %1543 to i32, !dbg !13
  %1545 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1544), !dbg !13
  %1546 = bitcast i32 %1545 to float, !dbg !13
  %1547 = select i1 %.not, float %1535, float %1546, !dbg !13
  %1548 = select i1 %.not, float %1546, float %1536, !dbg !13
  %1549 = select i1 %.not1, float %1547, float %1541, !dbg !13
  %1550 = bitcast float %1549 to i32, !dbg !13
  %1551 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1550), !dbg !13
  %1552 = bitcast i32 %1551 to float, !dbg !13
  %1553 = select i1 %.not1, float %1548, float %1542, !dbg !13
  %1554 = bitcast float %1553 to i32, !dbg !13
  %1555 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1554), !dbg !13
  %1556 = bitcast i32 %1555 to float, !dbg !13
  %1557 = insertelement <4 x float> poison, float %1541, i64 0, !dbg !13
  %1558 = insertelement <4 x float> %1557, float %1542, i64 1, !dbg !13
  %1559 = insertelement <4 x float> %1558, float %1552, i64 2, !dbg !13
  %1560 = insertelement <4 x float> %1559, float %1556, i64 3, !dbg !13
  %1561 = insertelement <4 x float> poison, float %1552, i64 0, !dbg !13
  %1562 = insertelement <4 x float> %1561, float %1556, i64 1, !dbg !13
  %1563 = insertelement <4 x float> %1562, float %1547, i64 2, !dbg !13
  %1564 = insertelement <4 x float> %1563, float %1548, i64 3, !dbg !13
  %1565 = select <4 x i1> %136, <4 x float> %1560, <4 x float> %1564, !dbg !13
  %1566 = extractvalue [2 x <8 x float>] %143, 0, !dbg !25
  %1567 = shufflevector <8 x float> %1566, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1568 = fmul <4 x float> %1567, %755, !dbg !25
  %1569 = fadd <4 x float> %1568, %1565, !dbg !30
  %1570 = shufflevector <8 x half> %1497, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1571 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1570, <4 x float> zeroinitializer) #8, !dbg !13
  %1572 = shufflevector <8 x half> %1499, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1573 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1572, <4 x float> %1571) #8, !dbg !13
  %1574 = shufflevector <8 x half> %1501, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1575 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1574, <4 x float> %1573) #8, !dbg !13
  %1576 = shufflevector <8 x half> %1503, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1577 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1576, <4 x float> %1575) #8, !dbg !13
  %1578 = shufflevector <8 x half> %1505, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1579 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1578, <4 x float> %1577) #8, !dbg !13
  %1580 = shufflevector <8 x half> %1507, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1581 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1580, <4 x float> %1579) #8, !dbg !13
  %1582 = shufflevector <8 x half> %1509, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1583 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1582, <4 x float> %1581) #8, !dbg !13
  %1584 = shufflevector <8 x half> %1511, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1585 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1584, <4 x float> %1583) #8, !dbg !13
  %1586 = extractelement <4 x float> %1585, i64 0, !dbg !13
  %1587 = extractelement <4 x float> %1585, i64 1, !dbg !13
  %1588 = extractelement <4 x float> %1585, i64 2, !dbg !13
  %1589 = extractelement <4 x float> %1585, i64 3, !dbg !13
  %1590 = select i1 %.not, float %1587, float %1586, !dbg !13
  %1591 = bitcast float %1590 to i32, !dbg !13
  %1592 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1591), !dbg !13
  %1593 = bitcast i32 %1592 to float, !dbg !13
  %1594 = select i1 %.not, float %1586, float %1593, !dbg !13
  %1595 = select i1 %.not, float %1593, float %1587, !dbg !13
  %1596 = select i1 %.not, float %1589, float %1588, !dbg !13
  %1597 = bitcast float %1596 to i32, !dbg !13
  %1598 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1597), !dbg !13
  %1599 = bitcast i32 %1598 to float, !dbg !13
  %1600 = select i1 %.not, float %1588, float %1599, !dbg !13
  %1601 = select i1 %.not, float %1599, float %1589, !dbg !13
  %1602 = select i1 %.not1, float %1600, float %1594, !dbg !13
  %1603 = bitcast float %1602 to i32, !dbg !13
  %1604 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1603), !dbg !13
  %1605 = bitcast i32 %1604 to float, !dbg !13
  %1606 = select i1 %.not1, float %1601, float %1595, !dbg !13
  %1607 = bitcast float %1606 to i32, !dbg !13
  %1608 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1607), !dbg !13
  %1609 = bitcast i32 %1608 to float, !dbg !13
  %1610 = insertelement <4 x float> poison, float %1594, i64 0, !dbg !13
  %1611 = insertelement <4 x float> %1610, float %1595, i64 1, !dbg !13
  %1612 = insertelement <4 x float> %1611, float %1605, i64 2, !dbg !13
  %1613 = insertelement <4 x float> %1612, float %1609, i64 3, !dbg !13
  %1614 = insertelement <4 x float> poison, float %1605, i64 0, !dbg !13
  %1615 = insertelement <4 x float> %1614, float %1609, i64 1, !dbg !13
  %1616 = insertelement <4 x float> %1615, float %1600, i64 2, !dbg !13
  %1617 = insertelement <4 x float> %1616, float %1601, i64 3, !dbg !13
  %1618 = select <4 x i1> %136, <4 x float> %1613, <4 x float> %1617, !dbg !13
  %1619 = shufflevector <8 x float> %1566, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1620 = fmul <4 x float> %1619, %755, !dbg !25
  %1621 = fadd <4 x float> %1620, %1618, !dbg !30
  %1622 = shufflevector <4 x float> %1569, <4 x float> %1621, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1623 = insertvalue [2 x <8 x float>] poison, <8 x float> %1622, 0, !dbg !11
  %1624 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1517, <4 x float> zeroinitializer) #8, !dbg !13
  %1625 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1519, <4 x float> %1624) #8, !dbg !13
  %1626 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1521, <4 x float> %1625) #8, !dbg !13
  %1627 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1523, <4 x float> %1626) #8, !dbg !13
  %1628 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1525, <4 x float> %1627) #8, !dbg !13
  %1629 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1527, <4 x float> %1628) #8, !dbg !13
  %1630 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1529, <4 x float> %1629) #8, !dbg !13
  %1631 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1531, <4 x float> %1630) #8, !dbg !13
  %1632 = extractelement <4 x float> %1631, i64 0, !dbg !13
  %1633 = extractelement <4 x float> %1631, i64 1, !dbg !13
  %1634 = extractelement <4 x float> %1631, i64 2, !dbg !13
  %1635 = extractelement <4 x float> %1631, i64 3, !dbg !13
  %1636 = select i1 %.not, float %1633, float %1632, !dbg !13
  %1637 = bitcast float %1636 to i32, !dbg !13
  %1638 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1637), !dbg !13
  %1639 = bitcast i32 %1638 to float, !dbg !13
  %1640 = select i1 %.not, float %1632, float %1639, !dbg !13
  %1641 = select i1 %.not, float %1639, float %1633, !dbg !13
  %1642 = select i1 %.not, float %1635, float %1634, !dbg !13
  %1643 = bitcast float %1642 to i32, !dbg !13
  %1644 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1643), !dbg !13
  %1645 = bitcast i32 %1644 to float, !dbg !13
  %1646 = select i1 %.not, float %1634, float %1645, !dbg !13
  %1647 = select i1 %.not, float %1645, float %1635, !dbg !13
  %1648 = select i1 %.not1, float %1646, float %1640, !dbg !13
  %1649 = bitcast float %1648 to i32, !dbg !13
  %1650 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1649), !dbg !13
  %1651 = bitcast i32 %1650 to float, !dbg !13
  %1652 = select i1 %.not1, float %1647, float %1641, !dbg !13
  %1653 = bitcast float %1652 to i32, !dbg !13
  %1654 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1653), !dbg !13
  %1655 = bitcast i32 %1654 to float, !dbg !13
  %1656 = insertelement <4 x float> poison, float %1640, i64 0, !dbg !13
  %1657 = insertelement <4 x float> %1656, float %1641, i64 1, !dbg !13
  %1658 = insertelement <4 x float> %1657, float %1651, i64 2, !dbg !13
  %1659 = insertelement <4 x float> %1658, float %1655, i64 3, !dbg !13
  %1660 = insertelement <4 x float> poison, float %1651, i64 0, !dbg !13
  %1661 = insertelement <4 x float> %1660, float %1655, i64 1, !dbg !13
  %1662 = insertelement <4 x float> %1661, float %1646, i64 2, !dbg !13
  %1663 = insertelement <4 x float> %1662, float %1647, i64 3, !dbg !13
  %1664 = select <4 x i1> %136, <4 x float> %1659, <4 x float> %1663, !dbg !13
  %1665 = extractvalue [2 x <8 x float>] %143, 1, !dbg !25
  %1666 = shufflevector <8 x float> %1665, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1667 = fmul <4 x float> %1666, %762, !dbg !25
  %1668 = fadd <4 x float> %1667, %1664, !dbg !30
  %1669 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1570, <4 x float> zeroinitializer) #8, !dbg !13
  %1670 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1572, <4 x float> %1669) #8, !dbg !13
  %1671 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1574, <4 x float> %1670) #8, !dbg !13
  %1672 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1576, <4 x float> %1671) #8, !dbg !13
  %1673 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1578, <4 x float> %1672) #8, !dbg !13
  %1674 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1580, <4 x float> %1673) #8, !dbg !13
  %1675 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1582, <4 x float> %1674) #8, !dbg !13
  %1676 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1584, <4 x float> %1675) #8, !dbg !13
  %1677 = extractelement <4 x float> %1676, i64 0, !dbg !13
  %1678 = extractelement <4 x float> %1676, i64 1, !dbg !13
  %1679 = extractelement <4 x float> %1676, i64 2, !dbg !13
  %1680 = extractelement <4 x float> %1676, i64 3, !dbg !13
  %1681 = select i1 %.not, float %1678, float %1677, !dbg !13
  %1682 = bitcast float %1681 to i32, !dbg !13
  %1683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1682), !dbg !13
  %1684 = bitcast i32 %1683 to float, !dbg !13
  %1685 = select i1 %.not, float %1677, float %1684, !dbg !13
  %1686 = select i1 %.not, float %1684, float %1678, !dbg !13
  %1687 = select i1 %.not, float %1680, float %1679, !dbg !13
  %1688 = bitcast float %1687 to i32, !dbg !13
  %1689 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1688), !dbg !13
  %1690 = bitcast i32 %1689 to float, !dbg !13
  %1691 = select i1 %.not, float %1679, float %1690, !dbg !13
  %1692 = select i1 %.not, float %1690, float %1680, !dbg !13
  %1693 = select i1 %.not1, float %1691, float %1685, !dbg !13
  %1694 = bitcast float %1693 to i32, !dbg !13
  %1695 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1694), !dbg !13
  %1696 = bitcast i32 %1695 to float, !dbg !13
  %1697 = select i1 %.not1, float %1692, float %1686, !dbg !13
  %1698 = bitcast float %1697 to i32, !dbg !13
  %1699 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1698), !dbg !13
  %1700 = bitcast i32 %1699 to float, !dbg !13
  %1701 = insertelement <4 x float> poison, float %1685, i64 0, !dbg !13
  %1702 = insertelement <4 x float> %1701, float %1686, i64 1, !dbg !13
  %1703 = insertelement <4 x float> %1702, float %1696, i64 2, !dbg !13
  %1704 = insertelement <4 x float> %1703, float %1700, i64 3, !dbg !13
  %1705 = insertelement <4 x float> poison, float %1696, i64 0, !dbg !13
  %1706 = insertelement <4 x float> %1705, float %1700, i64 1, !dbg !13
  %1707 = insertelement <4 x float> %1706, float %1691, i64 2, !dbg !13
  %1708 = insertelement <4 x float> %1707, float %1692, i64 3, !dbg !13
  %1709 = select <4 x i1> %136, <4 x float> %1704, <4 x float> %1708, !dbg !13
  %1710 = shufflevector <8 x float> %1665, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1711 = fmul <4 x float> %1710, %762, !dbg !25
  %1712 = fadd <4 x float> %1711, %1709, !dbg !30
  %1713 = shufflevector <4 x float> %1668, <4 x float> %1712, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1714 = insertvalue [2 x <8 x float>] %1623, <8 x float> %1713, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1715 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !13
  %1716 = bitcast <4 x i32> %1715 to <8 x half>, !dbg !13
  %1717 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !13
  %1718 = bitcast <4 x i32> %1717 to <8 x half>, !dbg !13
  %1719 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !13
  %1720 = bitcast <4 x i32> %1719 to <8 x half>, !dbg !13
  %1721 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !13
  %1722 = bitcast <4 x i32> %1721 to <8 x half>, !dbg !13
  %1723 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !13
  %1724 = bitcast <4 x i32> %1723 to <8 x half>, !dbg !13
  %1725 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !13
  %1726 = bitcast <4 x i32> %1725 to <8 x half>, !dbg !13
  %1727 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !13
  %1728 = bitcast <4 x i32> %1727 to <8 x half>, !dbg !13
  %1729 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !13
  %1730 = bitcast <4 x i32> %1729 to <8 x half>, !dbg !13
  %1731 = shufflevector <8 x half> %1716, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1731, <4 x float> zeroinitializer) #8, !dbg !13
  %1733 = shufflevector <8 x half> %1718, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1734 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1733, <4 x float> %1732) #8, !dbg !13
  %1735 = shufflevector <8 x half> %1720, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1736 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1735, <4 x float> %1734) #8, !dbg !13
  %1737 = shufflevector <8 x half> %1722, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1738 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1737, <4 x float> %1736) #8, !dbg !13
  %1739 = shufflevector <8 x half> %1724, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1740 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1739, <4 x float> %1738) #8, !dbg !13
  %1741 = shufflevector <8 x half> %1726, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1742 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1741, <4 x float> %1740) #8, !dbg !13
  %1743 = shufflevector <8 x half> %1728, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1744 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1743, <4 x float> %1742) #8, !dbg !13
  %1745 = shufflevector <8 x half> %1730, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1746 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1745, <4 x float> %1744) #8, !dbg !13
  %1747 = extractelement <4 x float> %1746, i64 0, !dbg !13
  %1748 = extractelement <4 x float> %1746, i64 1, !dbg !13
  %1749 = extractelement <4 x float> %1746, i64 2, !dbg !13
  %1750 = extractelement <4 x float> %1746, i64 3, !dbg !13
  %1751 = select i1 %.not, float %1748, float %1747, !dbg !13
  %1752 = bitcast float %1751 to i32, !dbg !13
  %1753 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1752), !dbg !13
  %1754 = bitcast i32 %1753 to float, !dbg !13
  %1755 = select i1 %.not, float %1747, float %1754, !dbg !13
  %1756 = select i1 %.not, float %1754, float %1748, !dbg !13
  %1757 = select i1 %.not, float %1750, float %1749, !dbg !13
  %1758 = bitcast float %1757 to i32, !dbg !13
  %1759 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1758), !dbg !13
  %1760 = bitcast i32 %1759 to float, !dbg !13
  %1761 = select i1 %.not, float %1749, float %1760, !dbg !13
  %1762 = select i1 %.not, float %1760, float %1750, !dbg !13
  %1763 = select i1 %.not1, float %1761, float %1755, !dbg !13
  %1764 = bitcast float %1763 to i32, !dbg !13
  %1765 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1764), !dbg !13
  %1766 = bitcast i32 %1765 to float, !dbg !13
  %1767 = select i1 %.not1, float %1762, float %1756, !dbg !13
  %1768 = bitcast float %1767 to i32, !dbg !13
  %1769 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1768), !dbg !13
  %1770 = bitcast i32 %1769 to float, !dbg !13
  %1771 = insertelement <4 x float> poison, float %1755, i64 0, !dbg !13
  %1772 = insertelement <4 x float> %1771, float %1756, i64 1, !dbg !13
  %1773 = insertelement <4 x float> %1772, float %1766, i64 2, !dbg !13
  %1774 = insertelement <4 x float> %1773, float %1770, i64 3, !dbg !13
  %1775 = insertelement <4 x float> poison, float %1766, i64 0, !dbg !13
  %1776 = insertelement <4 x float> %1775, float %1770, i64 1, !dbg !13
  %1777 = insertelement <4 x float> %1776, float %1761, i64 2, !dbg !13
  %1778 = insertelement <4 x float> %1777, float %1762, i64 3, !dbg !13
  %1779 = select <4 x i1> %136, <4 x float> %1774, <4 x float> %1778, !dbg !13
  %1780 = extractvalue [2 x <8 x float>] %142, 0, !dbg !25
  %1781 = shufflevector <8 x float> %1780, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1782 = fmul <4 x float> %1781, %755, !dbg !25
  %1783 = fadd <4 x float> %1782, %1779, !dbg !30
  %1784 = shufflevector <8 x half> %1716, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1785 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1784, <4 x float> zeroinitializer) #8, !dbg !13
  %1786 = shufflevector <8 x half> %1718, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1787 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1786, <4 x float> %1785) #8, !dbg !13
  %1788 = shufflevector <8 x half> %1720, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1789 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1788, <4 x float> %1787) #8, !dbg !13
  %1790 = shufflevector <8 x half> %1722, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1791 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1790, <4 x float> %1789) #8, !dbg !13
  %1792 = shufflevector <8 x half> %1724, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1793 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1792, <4 x float> %1791) #8, !dbg !13
  %1794 = shufflevector <8 x half> %1726, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1795 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1794, <4 x float> %1793) #8, !dbg !13
  %1796 = shufflevector <8 x half> %1728, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1797 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1796, <4 x float> %1795) #8, !dbg !13
  %1798 = shufflevector <8 x half> %1730, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1799 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1798, <4 x float> %1797) #8, !dbg !13
  %1800 = extractelement <4 x float> %1799, i64 0, !dbg !13
  %1801 = extractelement <4 x float> %1799, i64 1, !dbg !13
  %1802 = extractelement <4 x float> %1799, i64 2, !dbg !13
  %1803 = extractelement <4 x float> %1799, i64 3, !dbg !13
  %1804 = select i1 %.not, float %1801, float %1800, !dbg !13
  %1805 = bitcast float %1804 to i32, !dbg !13
  %1806 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1805), !dbg !13
  %1807 = bitcast i32 %1806 to float, !dbg !13
  %1808 = select i1 %.not, float %1800, float %1807, !dbg !13
  %1809 = select i1 %.not, float %1807, float %1801, !dbg !13
  %1810 = select i1 %.not, float %1803, float %1802, !dbg !13
  %1811 = bitcast float %1810 to i32, !dbg !13
  %1812 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1811), !dbg !13
  %1813 = bitcast i32 %1812 to float, !dbg !13
  %1814 = select i1 %.not, float %1802, float %1813, !dbg !13
  %1815 = select i1 %.not, float %1813, float %1803, !dbg !13
  %1816 = select i1 %.not1, float %1814, float %1808, !dbg !13
  %1817 = bitcast float %1816 to i32, !dbg !13
  %1818 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1817), !dbg !13
  %1819 = bitcast i32 %1818 to float, !dbg !13
  %1820 = select i1 %.not1, float %1815, float %1809, !dbg !13
  %1821 = bitcast float %1820 to i32, !dbg !13
  %1822 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1821), !dbg !13
  %1823 = bitcast i32 %1822 to float, !dbg !13
  %1824 = insertelement <4 x float> poison, float %1808, i64 0, !dbg !13
  %1825 = insertelement <4 x float> %1824, float %1809, i64 1, !dbg !13
  %1826 = insertelement <4 x float> %1825, float %1819, i64 2, !dbg !13
  %1827 = insertelement <4 x float> %1826, float %1823, i64 3, !dbg !13
  %1828 = insertelement <4 x float> poison, float %1819, i64 0, !dbg !13
  %1829 = insertelement <4 x float> %1828, float %1823, i64 1, !dbg !13
  %1830 = insertelement <4 x float> %1829, float %1814, i64 2, !dbg !13
  %1831 = insertelement <4 x float> %1830, float %1815, i64 3, !dbg !13
  %1832 = select <4 x i1> %136, <4 x float> %1827, <4 x float> %1831, !dbg !13
  %1833 = shufflevector <8 x float> %1780, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1834 = fmul <4 x float> %1833, %755, !dbg !25
  %1835 = fadd <4 x float> %1834, %1832, !dbg !30
  %1836 = shufflevector <4 x float> %1783, <4 x float> %1835, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1837 = insertvalue [2 x <8 x float>] poison, <8 x float> %1836, 0, !dbg !11
  %1838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1731, <4 x float> zeroinitializer) #8, !dbg !13
  %1839 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1733, <4 x float> %1838) #8, !dbg !13
  %1840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1735, <4 x float> %1839) #8, !dbg !13
  %1841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1737, <4 x float> %1840) #8, !dbg !13
  %1842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1739, <4 x float> %1841) #8, !dbg !13
  %1843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1741, <4 x float> %1842) #8, !dbg !13
  %1844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1743, <4 x float> %1843) #8, !dbg !13
  %1845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1745, <4 x float> %1844) #8, !dbg !13
  %1846 = extractelement <4 x float> %1845, i64 0, !dbg !13
  %1847 = extractelement <4 x float> %1845, i64 1, !dbg !13
  %1848 = extractelement <4 x float> %1845, i64 2, !dbg !13
  %1849 = extractelement <4 x float> %1845, i64 3, !dbg !13
  %1850 = select i1 %.not, float %1847, float %1846, !dbg !13
  %1851 = bitcast float %1850 to i32, !dbg !13
  %1852 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1851), !dbg !13
  %1853 = bitcast i32 %1852 to float, !dbg !13
  %1854 = select i1 %.not, float %1846, float %1853, !dbg !13
  %1855 = select i1 %.not, float %1853, float %1847, !dbg !13
  %1856 = select i1 %.not, float %1849, float %1848, !dbg !13
  %1857 = bitcast float %1856 to i32, !dbg !13
  %1858 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1857), !dbg !13
  %1859 = bitcast i32 %1858 to float, !dbg !13
  %1860 = select i1 %.not, float %1848, float %1859, !dbg !13
  %1861 = select i1 %.not, float %1859, float %1849, !dbg !13
  %1862 = select i1 %.not1, float %1860, float %1854, !dbg !13
  %1863 = bitcast float %1862 to i32, !dbg !13
  %1864 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1863), !dbg !13
  %1865 = bitcast i32 %1864 to float, !dbg !13
  %1866 = select i1 %.not1, float %1861, float %1855, !dbg !13
  %1867 = bitcast float %1866 to i32, !dbg !13
  %1868 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1867), !dbg !13
  %1869 = bitcast i32 %1868 to float, !dbg !13
  %1870 = insertelement <4 x float> poison, float %1854, i64 0, !dbg !13
  %1871 = insertelement <4 x float> %1870, float %1855, i64 1, !dbg !13
  %1872 = insertelement <4 x float> %1871, float %1865, i64 2, !dbg !13
  %1873 = insertelement <4 x float> %1872, float %1869, i64 3, !dbg !13
  %1874 = insertelement <4 x float> poison, float %1865, i64 0, !dbg !13
  %1875 = insertelement <4 x float> %1874, float %1869, i64 1, !dbg !13
  %1876 = insertelement <4 x float> %1875, float %1860, i64 2, !dbg !13
  %1877 = insertelement <4 x float> %1876, float %1861, i64 3, !dbg !13
  %1878 = select <4 x i1> %136, <4 x float> %1873, <4 x float> %1877, !dbg !13
  %1879 = extractvalue [2 x <8 x float>] %142, 1, !dbg !25
  %1880 = shufflevector <8 x float> %1879, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1881 = fmul <4 x float> %1880, %762, !dbg !25
  %1882 = fadd <4 x float> %1881, %1878, !dbg !30
  %1883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1784, <4 x float> zeroinitializer) #8, !dbg !13
  %1884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1786, <4 x float> %1883) #8, !dbg !13
  %1885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1788, <4 x float> %1884) #8, !dbg !13
  %1886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1790, <4 x float> %1885) #8, !dbg !13
  %1887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1792, <4 x float> %1886) #8, !dbg !13
  %1888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1794, <4 x float> %1887) #8, !dbg !13
  %1889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1796, <4 x float> %1888) #8, !dbg !13
  %1890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1798, <4 x float> %1889) #8, !dbg !13
  %1891 = extractelement <4 x float> %1890, i64 0, !dbg !13
  %1892 = extractelement <4 x float> %1890, i64 1, !dbg !13
  %1893 = extractelement <4 x float> %1890, i64 2, !dbg !13
  %1894 = extractelement <4 x float> %1890, i64 3, !dbg !13
  %1895 = select i1 %.not, float %1892, float %1891, !dbg !13
  %1896 = bitcast float %1895 to i32, !dbg !13
  %1897 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1896), !dbg !13
  %1898 = bitcast i32 %1897 to float, !dbg !13
  %1899 = select i1 %.not, float %1891, float %1898, !dbg !13
  %1900 = select i1 %.not, float %1898, float %1892, !dbg !13
  %1901 = select i1 %.not, float %1894, float %1893, !dbg !13
  %1902 = bitcast float %1901 to i32, !dbg !13
  %1903 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %679, i32 %1902), !dbg !13
  %1904 = bitcast i32 %1903 to float, !dbg !13
  %1905 = select i1 %.not, float %1893, float %1904, !dbg !13
  %1906 = select i1 %.not, float %1904, float %1894, !dbg !13
  %1907 = select i1 %.not1, float %1905, float %1899, !dbg !13
  %1908 = bitcast float %1907 to i32, !dbg !13
  %1909 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1908), !dbg !13
  %1910 = bitcast i32 %1909 to float, !dbg !13
  %1911 = select i1 %.not1, float %1906, float %1900, !dbg !13
  %1912 = bitcast float %1911 to i32, !dbg !13
  %1913 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %1912), !dbg !13
  %1914 = bitcast i32 %1913 to float, !dbg !13
  %1915 = insertelement <4 x float> poison, float %1899, i64 0, !dbg !13
  %1916 = insertelement <4 x float> %1915, float %1900, i64 1, !dbg !13
  %1917 = insertelement <4 x float> %1916, float %1910, i64 2, !dbg !13
  %1918 = insertelement <4 x float> %1917, float %1914, i64 3, !dbg !13
  %1919 = insertelement <4 x float> poison, float %1910, i64 0, !dbg !13
  %1920 = insertelement <4 x float> %1919, float %1914, i64 1, !dbg !13
  %1921 = insertelement <4 x float> %1920, float %1905, i64 2, !dbg !13
  %1922 = insertelement <4 x float> %1921, float %1906, i64 3, !dbg !13
  %1923 = select <4 x i1> %136, <4 x float> %1918, <4 x float> %1922, !dbg !13
  %1924 = shufflevector <8 x float> %1879, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1925 = fmul <4 x float> %1924, %762, !dbg !25
  %1926 = fadd <4 x float> %1925, %1923, !dbg !30
  %1927 = shufflevector <4 x float> %1882, <4 x float> %1926, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1928 = insertvalue [2 x <8 x float>] %1837, <8 x float> %1927, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1929 = add nuw nsw i64 %147, 1, !dbg !11
  %1930 = icmp slt i64 %1929, %105, !dbg !11
  br i1 %1930, label %141, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %540, %4
  %.lcssa94 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1279, %540 ], !dbg !11
  %.lcssa93 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1057, %540 ], !dbg !11
  %.lcssa92 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1495, %540 ], !dbg !11
  %.lcssa91 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1714, %540 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1928, %540 ], !dbg !11
  %1931 = extractvalue [2 x <1 x float>] %.lcssa93, 0, !dbg !31
  %1932 = shufflevector <1 x float> %1931, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !31
  %1933 = extractvalue [2 x <1 x float>] %.lcssa93, 1, !dbg !31
  %1934 = shufflevector <1 x float> %1933, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !31
  %1935 = extractvalue [2 x <8 x float>] %.lcssa94, 1, !dbg !31
  %1936 = shufflevector <8 x float> %1935, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1934, !dbg !31
  %normalized = fmul <4 x float> %1936, %row.reciprocal, !dbg !31
  %1937 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !32
  %1938 = extractelement <4 x half> %1937, i64 3, !dbg !32
  %1939 = extractelement <4 x half> %1937, i64 2, !dbg !32
  %1940 = extractelement <4 x half> %1937, i64 1, !dbg !32
  %1941 = extractelement <4 x half> %1937, i64 0, !dbg !32
  %1942 = shufflevector <8 x float> %1935, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized104 = fmul <4 x float> %1942, %row.reciprocal, !dbg !31
  %1943 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !32
  %1944 = extractelement <4 x half> %1943, i64 3, !dbg !32
  %1945 = extractelement <4 x half> %1943, i64 2, !dbg !32
  %1946 = extractelement <4 x half> %1943, i64 1, !dbg !32
  %1947 = extractelement <4 x half> %1943, i64 0, !dbg !32
  %1948 = extractvalue [2 x <8 x float>] %.lcssa94, 0, !dbg !31
  %1949 = shufflevector <8 x float> %1948, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1932, !dbg !31
  %normalized106 = fmul <4 x float> %1949, %row.reciprocal105, !dbg !31
  %1950 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !32
  %1951 = extractelement <4 x half> %1950, i64 3, !dbg !32
  %1952 = extractelement <4 x half> %1950, i64 2, !dbg !32
  %1953 = extractelement <4 x half> %1950, i64 1, !dbg !32
  %1954 = extractelement <4 x half> %1950, i64 0, !dbg !32
  %1955 = shufflevector <8 x float> %1948, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized107 = fmul <4 x float> %1955, %row.reciprocal105, !dbg !31
  %1956 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !32
  %1957 = extractelement <4 x half> %1956, i64 3, !dbg !32
  %1958 = extractelement <4 x half> %1956, i64 2, !dbg !32
  %1959 = extractelement <4 x half> %1956, i64 1, !dbg !32
  %1960 = extractelement <4 x half> %1956, i64 0, !dbg !32
  %address.thread934 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !33, !range !9, !frisk.register_lifetimes !5
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !8
  %1961 = lshr i64 %address.local935, 2, !dbg !33
  %address.local938 = and i64 %1961, 48, !dbg !33
  %address.local940 = and i64 %address.local935, 240, !dbg !33
  %address.local939 = add i64 %26, %address.local935, !dbg !33
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !33
  %address.local942 = add i64 %address.local941, %address.local938, !dbg !33
  %1962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !33
  %address.local945.idx = shl nuw nsw i64 %address.local942, 8, !dbg !33
  %address.local945 = getelementptr inbounds nuw i8, ptr addrspace(1) %1962, i64 %address.local945.idx, !dbg !33
  %address.local947 = and i64 %1961, 12, !dbg !10
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !33
  store half %1960, ptr addrspace(1) %address.local948, align 2, !dbg !33
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !33
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local950, !dbg !33
  store half %1959, ptr addrspace(1) %address.local951, align 2, !dbg !33
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !33
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local953, !dbg !33
  store half %1958, ptr addrspace(1) %address.local954, align 2, !dbg !33
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !33
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local956, !dbg !33
  store half %1957, ptr addrspace(1) %address.local957, align 2, !dbg !33
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !33
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local959, !dbg !33
  store half %1954, ptr addrspace(1) %address.local960, align 2, !dbg !33
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !33
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local962, !dbg !33
  store half %1953, ptr addrspace(1) %address.local963, align 2, !dbg !33
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !33
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local965, !dbg !33
  store half %1952, ptr addrspace(1) %address.local966, align 2, !dbg !33
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !33
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local968, !dbg !33
  store half %1951, ptr addrspace(1) %address.local969, align 2, !dbg !33
  %address.local973 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local945, i64 16384, !dbg !33
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !33
  store half %1947, ptr addrspace(1) %address.local974, align 2, !dbg !33
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local950, !dbg !33
  store half %1946, ptr addrspace(1) %address.local976, align 2, !dbg !33
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local953, !dbg !33
  store half %1945, ptr addrspace(1) %address.local978, align 2, !dbg !33
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local956, !dbg !33
  store half %1944, ptr addrspace(1) %address.local980, align 2, !dbg !33
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local959, !dbg !33
  store half %1941, ptr addrspace(1) %address.local982, align 2, !dbg !33
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local962, !dbg !33
  store half %1940, ptr addrspace(1) %address.local984, align 2, !dbg !33
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local965, !dbg !33
  store half %1939, ptr addrspace(1) %address.local986, align 2, !dbg !33
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local968, !dbg !33
  store half %1938, ptr addrspace(1) %address.local988, align 2, !dbg !33
  %1963 = extractvalue [2 x <8 x float>] %.lcssa92, 1, !dbg !31
  %1964 = shufflevector <8 x float> %1963, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized109 = fmul <4 x float> %1964, %row.reciprocal, !dbg !31
  %1965 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !32
  %1966 = extractelement <4 x half> %1965, i64 3, !dbg !32
  %1967 = extractelement <4 x half> %1965, i64 2, !dbg !32
  %1968 = extractelement <4 x half> %1965, i64 1, !dbg !32
  %1969 = extractelement <4 x half> %1965, i64 0, !dbg !32
  %1970 = shufflevector <8 x float> %1963, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized110 = fmul <4 x float> %1970, %row.reciprocal, !dbg !31
  %1971 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !32
  %1972 = extractelement <4 x half> %1971, i64 3, !dbg !32
  %1973 = extractelement <4 x half> %1971, i64 2, !dbg !32
  %1974 = extractelement <4 x half> %1971, i64 1, !dbg !32
  %1975 = extractelement <4 x half> %1971, i64 0, !dbg !32
  %1976 = extractvalue [2 x <8 x float>] %.lcssa92, 0, !dbg !31
  %1977 = shufflevector <8 x float> %1976, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized112 = fmul <4 x float> %1977, %row.reciprocal105, !dbg !31
  %1978 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !32
  %1979 = extractelement <4 x half> %1978, i64 3, !dbg !32
  %1980 = extractelement <4 x half> %1978, i64 2, !dbg !32
  %1981 = extractelement <4 x half> %1978, i64 1, !dbg !32
  %1982 = extractelement <4 x half> %1978, i64 0, !dbg !32
  %1983 = shufflevector <8 x float> %1976, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized113 = fmul <4 x float> %1983, %row.reciprocal105, !dbg !31
  %1984 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !32
  %1985 = extractelement <4 x half> %1984, i64 3, !dbg !32
  %1986 = extractelement <4 x half> %1984, i64 2, !dbg !32
  %1987 = extractelement <4 x half> %1984, i64 1, !dbg !32
  %1988 = extractelement <4 x half> %1984, i64 0, !dbg !32
  %address.local1003 = or disjoint i64 %address.local947, 32, !dbg !33
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1003, !dbg !33
  store half %1988, ptr addrspace(1) %address.local1004, align 2, !dbg !33
  %address.local1006 = or disjoint i64 %address.local947, 33, !dbg !33
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1006, !dbg !33
  store half %1987, ptr addrspace(1) %address.local1007, align 2, !dbg !33
  %address.local1009 = or disjoint i64 %address.local947, 34, !dbg !33
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1009, !dbg !33
  store half %1986, ptr addrspace(1) %address.local1010, align 2, !dbg !33
  %address.local1012 = or disjoint i64 %address.local947, 35, !dbg !33
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1012, !dbg !33
  store half %1985, ptr addrspace(1) %address.local1013, align 2, !dbg !33
  %address.local1015 = or disjoint i64 %address.local947, 48, !dbg !33
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1015, !dbg !33
  store half %1982, ptr addrspace(1) %address.local1016, align 2, !dbg !33
  %address.local1018 = or disjoint i64 %address.local947, 49, !dbg !33
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1018, !dbg !33
  store half %1981, ptr addrspace(1) %address.local1019, align 2, !dbg !33
  %address.local1021 = or disjoint i64 %address.local947, 50, !dbg !33
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1021, !dbg !33
  store half %1980, ptr addrspace(1) %address.local1022, align 2, !dbg !33
  %address.local1024 = or i64 %1961, 51, !dbg !33
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1024, !dbg !33
  store half %1979, ptr addrspace(1) %address.local1025, align 2, !dbg !33
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1003, !dbg !33
  store half %1975, ptr addrspace(1) %address.local1030, align 2, !dbg !33
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1006, !dbg !33
  store half %1974, ptr addrspace(1) %address.local1032, align 2, !dbg !33
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1009, !dbg !33
  store half %1973, ptr addrspace(1) %address.local1034, align 2, !dbg !33
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1012, !dbg !33
  store half %1972, ptr addrspace(1) %address.local1036, align 2, !dbg !33
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1015, !dbg !33
  store half %1969, ptr addrspace(1) %address.local1038, align 2, !dbg !33
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1018, !dbg !33
  store half %1968, ptr addrspace(1) %address.local1040, align 2, !dbg !33
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1021, !dbg !33
  store half %1967, ptr addrspace(1) %address.local1042, align 2, !dbg !33
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1024, !dbg !33
  store half %1966, ptr addrspace(1) %address.local1044, align 2, !dbg !33
  %1989 = extractvalue [2 x <8 x float>] %.lcssa91, 1, !dbg !31
  %1990 = shufflevector <8 x float> %1989, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized115 = fmul <4 x float> %row.reciprocal, %1990, !dbg !31
  %1991 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !32
  %1992 = extractelement <4 x half> %1991, i64 3, !dbg !32
  %1993 = extractelement <4 x half> %1991, i64 2, !dbg !32
  %1994 = extractelement <4 x half> %1991, i64 1, !dbg !32
  %1995 = extractelement <4 x half> %1991, i64 0, !dbg !32
  %1996 = shufflevector <8 x float> %1989, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized116 = fmul <4 x float> %row.reciprocal, %1996, !dbg !31
  %1997 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !32
  %1998 = extractelement <4 x half> %1997, i64 3, !dbg !32
  %1999 = extractelement <4 x half> %1997, i64 2, !dbg !32
  %2000 = extractelement <4 x half> %1997, i64 1, !dbg !32
  %2001 = extractelement <4 x half> %1997, i64 0, !dbg !32
  %2002 = extractvalue [2 x <8 x float>] %.lcssa91, 0, !dbg !31
  %2003 = shufflevector <8 x float> %2002, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized118 = fmul <4 x float> %row.reciprocal105, %2003, !dbg !31
  %2004 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !32
  %2005 = extractelement <4 x half> %2004, i64 3, !dbg !32
  %2006 = extractelement <4 x half> %2004, i64 2, !dbg !32
  %2007 = extractelement <4 x half> %2004, i64 1, !dbg !32
  %2008 = extractelement <4 x half> %2004, i64 0, !dbg !32
  %2009 = shufflevector <8 x float> %2002, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized119 = fmul <4 x float> %row.reciprocal105, %2009, !dbg !31
  %2010 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !32
  %2011 = extractelement <4 x half> %2010, i64 3, !dbg !32
  %2012 = extractelement <4 x half> %2010, i64 2, !dbg !32
  %2013 = extractelement <4 x half> %2010, i64 1, !dbg !32
  %2014 = extractelement <4 x half> %2010, i64 0, !dbg !32
  %address.local1059 = or disjoint i64 %address.local947, 64, !dbg !33
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1059, !dbg !33
  store half %2014, ptr addrspace(1) %address.local1060, align 2, !dbg !33
  %address.local1062 = or disjoint i64 %address.local947, 65, !dbg !33
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1062, !dbg !33
  store half %2013, ptr addrspace(1) %address.local1063, align 2, !dbg !33
  %address.local1065 = or disjoint i64 %address.local947, 66, !dbg !33
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1065, !dbg !33
  store half %2012, ptr addrspace(1) %address.local1066, align 2, !dbg !33
  %address.local1068 = or disjoint i64 %address.local947, 67, !dbg !33
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1068, !dbg !33
  store half %2011, ptr addrspace(1) %address.local1069, align 2, !dbg !33
  %address.local1071 = or disjoint i64 %address.local947, 80, !dbg !33
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1071, !dbg !33
  store half %2008, ptr addrspace(1) %address.local1072, align 2, !dbg !33
  %address.local1074 = or disjoint i64 %address.local947, 81, !dbg !33
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1074, !dbg !33
  store half %2007, ptr addrspace(1) %address.local1075, align 2, !dbg !33
  %address.local1077 = or disjoint i64 %address.local947, 82, !dbg !33
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1077, !dbg !33
  store half %2006, ptr addrspace(1) %address.local1078, align 2, !dbg !33
  %address.local1080 = or disjoint i64 %address.local947, 83, !dbg !33
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1080, !dbg !33
  store half %2005, ptr addrspace(1) %address.local1081, align 2, !dbg !33
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1059, !dbg !33
  store half %2001, ptr addrspace(1) %address.local1086, align 2, !dbg !33
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1062, !dbg !33
  store half %2000, ptr addrspace(1) %address.local1088, align 2, !dbg !33
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1065, !dbg !33
  store half %1999, ptr addrspace(1) %address.local1090, align 2, !dbg !33
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1068, !dbg !33
  store half %1998, ptr addrspace(1) %address.local1092, align 2, !dbg !33
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1071, !dbg !33
  store half %1995, ptr addrspace(1) %address.local1094, align 2, !dbg !33
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1074, !dbg !33
  store half %1994, ptr addrspace(1) %address.local1096, align 2, !dbg !33
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1077, !dbg !33
  store half %1993, ptr addrspace(1) %address.local1098, align 2, !dbg !33
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1080, !dbg !33
  store half %1992, ptr addrspace(1) %address.local1100, align 2, !dbg !33
  %2015 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !31
  %2016 = shufflevector <8 x float> %2015, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized121 = fmul <4 x float> %row.reciprocal, %2016, !dbg !31
  %2017 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !32
  %2018 = extractelement <4 x half> %2017, i64 3, !dbg !32
  %2019 = extractelement <4 x half> %2017, i64 2, !dbg !32
  %2020 = extractelement <4 x half> %2017, i64 1, !dbg !32
  %2021 = extractelement <4 x half> %2017, i64 0, !dbg !32
  %2022 = shufflevector <8 x float> %2015, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized122 = fmul <4 x float> %row.reciprocal, %2022, !dbg !31
  %2023 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !32
  %2024 = extractelement <4 x half> %2023, i64 3, !dbg !32
  %2025 = extractelement <4 x half> %2023, i64 2, !dbg !32
  %2026 = extractelement <4 x half> %2023, i64 1, !dbg !32
  %2027 = extractelement <4 x half> %2023, i64 0, !dbg !32
  %2028 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !31
  %2029 = shufflevector <8 x float> %2028, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized124 = fmul <4 x float> %row.reciprocal105, %2029, !dbg !31
  %2030 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !32
  %2031 = extractelement <4 x half> %2030, i64 3, !dbg !32
  %2032 = extractelement <4 x half> %2030, i64 2, !dbg !32
  %2033 = extractelement <4 x half> %2030, i64 1, !dbg !32
  %2034 = extractelement <4 x half> %2030, i64 0, !dbg !32
  %2035 = shufflevector <8 x float> %2028, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized125 = fmul <4 x float> %row.reciprocal105, %2035, !dbg !31
  %2036 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !32
  %2037 = extractelement <4 x half> %2036, i64 3, !dbg !32
  %2038 = extractelement <4 x half> %2036, i64 2, !dbg !32
  %2039 = extractelement <4 x half> %2036, i64 1, !dbg !32
  %2040 = extractelement <4 x half> %2036, i64 0, !dbg !32
  %address.local1115 = or disjoint i64 %address.local947, 96, !dbg !33
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1115, !dbg !33
  store half %2040, ptr addrspace(1) %address.local1116, align 2, !dbg !33
  %address.local1118 = or disjoint i64 %address.local947, 97, !dbg !33
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1118, !dbg !33
  store half %2039, ptr addrspace(1) %address.local1119, align 2, !dbg !33
  %address.local1121 = or disjoint i64 %address.local947, 98, !dbg !33
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1121, !dbg !33
  store half %2038, ptr addrspace(1) %address.local1122, align 2, !dbg !33
  %address.local1124 = or disjoint i64 %address.local947, 99, !dbg !33
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1124, !dbg !33
  store half %2037, ptr addrspace(1) %address.local1125, align 2, !dbg !33
  %address.local1127 = or disjoint i64 %address.local947, 112, !dbg !33
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1127, !dbg !33
  store half %2034, ptr addrspace(1) %address.local1128, align 2, !dbg !33
  %address.local1130 = or disjoint i64 %address.local947, 113, !dbg !33
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1130, !dbg !33
  store half %2033, ptr addrspace(1) %address.local1131, align 2, !dbg !33
  %address.local1133 = or disjoint i64 %address.local947, 114, !dbg !33
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1133, !dbg !33
  store half %2032, ptr addrspace(1) %address.local1134, align 2, !dbg !33
  %address.local1136 = or i64 %1961, 115, !dbg !33
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1136, !dbg !33
  store half %2031, ptr addrspace(1) %address.local1137, align 2, !dbg !33
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1115, !dbg !33
  store half %2027, ptr addrspace(1) %address.local1142, align 2, !dbg !33
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1118, !dbg !33
  store half %2026, ptr addrspace(1) %address.local1144, align 2, !dbg !33
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1121, !dbg !33
  store half %2025, ptr addrspace(1) %address.local1146, align 2, !dbg !33
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1124, !dbg !33
  store half %2024, ptr addrspace(1) %address.local1148, align 2, !dbg !33
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1127, !dbg !33
  store half %2021, ptr addrspace(1) %address.local1150, align 2, !dbg !33
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1130, !dbg !33
  store half %2020, ptr addrspace(1) %address.local1152, align 2, !dbg !33
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1133, !dbg !33
  store half %2019, ptr addrspace(1) %address.local1154, align 2, !dbg !33
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1136, !dbg !33
  store half %2018, ptr addrspace(1) %address.local1156, align 2, !dbg !33
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

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #4

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/export/run-20261023-171017-a0mz8rfe/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline")
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
!13 = !DILocation(line: 102, column: 23, scope: !3)
!14 = !DILocation(line: 0, scope: !3)
!15 = !DILocation(line: 71, column: 22, scope: !3)
!16 = !DILocation(line: 73, column: 26, scope: !3)
!17 = !DILocation(line: 80, column: 23, scope: !3)
!18 = !DILocation(line: 101, column: 20, scope: !3)
!19 = !DILocation(line: 81, column: 26, scope: !3)
!20 = !DILocation(line: 88, column: 21, scope: !3)
!21 = !DILocation(line: 89, column: 20, scope: !3)
!22 = !DILocation(line: 90, column: 22, scope: !3)
!23 = !DILocation(line: 91, column: 23, scope: !3)
!24 = !DILocation(line: 92, column: 20, scope: !3)
!25 = !DILocation(line: 103, column: 22, scope: !3)
!26 = !DILocation(line: 93, column: 22, scope: !3)
!27 = !DILocation(line: 94, column: 22, scope: !3)
!28 = !DILocation(line: 96, column: 16, scope: !3)
!29 = !DILocation(line: 95, column: 20, scope: !3)
!30 = !DILocation(line: 104, column: 20, scope: !3)
!31 = !DILocation(line: 107, column: 23, scope: !3)
!32 = !DILocation(line: 108, column: 19, scope: !3)
!33 = !DILocation(line: 109, column: 9, scope: !3)
