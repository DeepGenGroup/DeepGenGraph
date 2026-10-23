; ModuleID = 'generated_ir/online_softmax/retained32_online.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/release/run-20261023-172519-7itoy6x9/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/input.mlir"

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

141:                                              ; preds = %.lr.ph, %541
  %142 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1935, %541 ]
  %143 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1721, %541 ]
  %144 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1502, %541 ]
  %145 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1937, %541 ]
  %146 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1064, %541 ]
  %147 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1286, %541 ]
  %148 = phi i64 [ 0, %.lr.ph ], [ %1938, %541 ]
  %149 = shl nuw nsw i64 %148, 7, !dbg !11
  %150 = lshr i64 %148, 12, !dbg !15
  %151 = add nuw nsw i64 %150, %16, !dbg !15
  %152 = lshr i64 %148, 5, !dbg !15
  %153 = and i64 %152, 127, !dbg !15
  %154 = shl i64 %151, 19, !dbg !15
  %155 = and i64 %154, 16252928, !dbg !15
  %156 = shl i64 %152, 19, !dbg !15
  %157 = shl nuw nsw i64 %148, 14, !dbg !15
  %158 = sub i64 %157, %156, !dbg !15
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
  %.idx912 = or disjoint i64 %155, %153, !dbg !15
  %159 = add i64 %.idx912, %158, !dbg !15
  %160 = add i64 %address.local449, %159, !dbg !15
  %.tr = trunc i64 %160 to i32, !dbg !15
  %dma.buffer.offset1588 = shl i32 %.tr, 1, !dbg !15
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !8
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !8
  %address.local453 = and i64 %address.local452, 3584, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !8
  %address.local457 = lshr i64 %address.local451, 4, !dbg !15
  %address.local458 = and i64 %address.local457, 24, !dbg !15
  %address.local466 = or disjoint i64 %address.local448, %address.local458, !dbg !15
  %161 = add i64 %address.local466, %159, !dbg !15
  %.tr17 = trunc i64 %161 to i32, !dbg !15
  %dma.buffer.offset1585 = shl i32 %.tr17, 1, !dbg !15
  %162 = add nuw nsw i64 %148, 1024, !dbg !15
  %163 = lshr i64 %162, 12, !dbg !15
  %164 = add nuw nsw i64 %163, %16, !dbg !15
  %165 = add nsw i64 %152, %109, !dbg !15
  %166 = shl i64 %164, 19, !dbg !15
  %167 = and i64 %166, 16252928, !dbg !15
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !8
  %168 = add i64 %158, %152, !dbg !15
  %.idx1822 = add i64 %168, %109, !dbg !15
  %169 = add i64 %.idx1822, %167, !dbg !15
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !8
  %170 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local437, i64 0, !dbg !8
  %171 = insertelement <4 x ptr addrspace(3)> %170, ptr addrspace(3) %address.local454, i64 1, !dbg !8
  %172 = insertelement <4 x ptr addrspace(3)> %171, ptr addrspace(3) %address.local468, i64 2, !dbg !8
  %173 = insertelement <4 x ptr addrspace(3)> %172, ptr addrspace(3) %address.local472, i64 3, !dbg !8
  %174 = ptrtoint <4 x ptr addrspace(3)> %173 to <4 x i64>, !dbg !8
  %175 = trunc <4 x i64> %174 to <4 x i32>, !dbg !8
  %176 = extractelement <4 x i32> %175, i64 0, !dbg !15
  %177 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %176), !dbg !15
  %dma.buffer.lds1590 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %177), !dbg !15
  %178 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %179 = extractelement <4 x i32> %175, i64 1, !dbg !15
  %180 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %179), !dbg !15
  %dma.buffer.lds1587 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %180), !dbg !15
  %181 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %182 = extractelement <4 x i32> %175, i64 2, !dbg !15
  %183 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %182), !dbg !15
  %dma.buffer.lds1584 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %183), !dbg !15
  %184 = extractelement <4 x i32> %175, i64 3, !dbg !15
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !16
  %address.local481 = and i64 %address.local480, 24, !dbg !16
  %address.local483 = shl i64 %address.local434, 62, !dbg !16
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !16
  %address.local485 = and i64 %address.local484, -64, !dbg !16
  %address.local488 = and i64 %address.local443, 12, !dbg !10
  %185 = and i64 %address.local434, 32, !dbg !10
  %.not30 = icmp eq i64 %185, 0, !dbg !10
  %address.local492 = select i1 %.not30, i64 0, i64 1016, !dbg !10
  %.idx31 = mul nuw nsw i64 %address.local434, 320, !dbg !16
  %186 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx31, !dbg !16
  %.idx32 = mul nsw i64 %address.local443, -1280, !dbg !16
  %187 = getelementptr i8, ptr addrspace(3) %186, i64 %.idx32, !dbg !16
  %188 = getelementptr [2 x i8], ptr addrspace(3) %187, i64 %address.local481, !dbg !16
  %189 = getelementptr [2 x i8], ptr addrspace(3) %188, i64 %address.local485, !dbg !16
  %190 = getelementptr [2 x i8], ptr addrspace(3) %189, i64 %address.local488, !dbg !16
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) %190, i64 %address.local492, !dbg !16
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
  %191 = add nuw nsw i64 %148, 2048, !dbg !15
  %192 = lshr i64 %191, 12, !dbg !15
  %193 = add nuw nsw i64 %192, %16, !dbg !15
  %194 = insertelement <2 x i64> poison, i64 %165, i64 0, !dbg !15
  %195 = shufflevector <2 x i64> %194, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !15
  %196 = icmp slt <2 x i64> %195, <i64 -32, i64 -64>, !dbg !15
  %197 = extractelement <2 x i1> %196, i64 0, !dbg !15
  %198 = sext i1 %197 to i64, !dbg !15
  %199 = add nsw <2 x i64> %195, <i64 32, i64 64>, !dbg !15
  %200 = sub nsw <2 x i64> <i64 -33, i64 -65>, %195, !dbg !15
  %201 = select <2 x i1> %196, <2 x i64> %200, <2 x i64> %199, !dbg !15
  %202 = sdiv <2 x i64> %201, splat (i64 128), !dbg !15
  %203 = extractelement <2 x i64> %202, i64 0, !dbg !15
  %204 = xor i64 %203, %198, !dbg !15
  %205 = shl nsw i64 %204, 7, !dbg !15
  %206 = sub i64 %169, %205, !dbg !15
  %207 = add i64 %address.local449, %206, !dbg !15
  %.tr23 = trunc i64 %207 to i32, !dbg !15
  %208 = shl i32 %.tr23, 1, !dbg !15
  %dma.buffer.offset1582 = add i32 %208, 64, !dbg !15
  %209 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %210 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %184), !dbg !15
  %211 = add i64 %address.local466, %206, !dbg !15
  %.tr29 = trunc i64 %211 to i32, !dbg !15
  %212 = shl i32 %.tr29, 1, !dbg !15
  %dma.buffer.offset1579 = add i32 %212, 64, !dbg !15
  %dma.buffer.lds1581 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %210), !dbg !15
  %213 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !16
  %215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %214, <4 x float> zeroinitializer) #8, !dbg !16
  %216 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !16
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %.sext) #8, !dbg !16, !frisk.register_lifetimes !5
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !16
  %address.local518 = shl i64 %address.local517, 6, !dbg !16
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !16
  %217 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local519, !dbg !16
  %218 = getelementptr [2 x i8], ptr addrspace(3) %217, i64 %address.local488, !dbg !16
  %219 = getelementptr [2 x i8], ptr addrspace(3) %218, i64 %address.local492, !dbg !16
  %address.local523 = getelementptr i8, ptr addrspace(3) %219, i64 64, !dbg !16
  %220 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !16
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %216, <4 x float> %215) #8, !dbg !16
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %220, <4 x float> zeroinitializer) #8, !dbg !16
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %219, i64 4160, !dbg !16
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !16
  %224 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !16
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %223, <4 x float> %222) #8, !dbg !16
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !16
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !16
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %219, i64 192, !dbg !16
  %228 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !16
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %227, <4 x float> %226) #8, !dbg !16
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %228, <4 x float> zeroinitializer) #8, !dbg !16
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %219, i64 4288, !dbg !16
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !16
  %232 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !16
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %231, <4 x float> %230) #8, !dbg !16
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %232, <4 x float> zeroinitializer) #8, !dbg !16
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !16
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %219, i64 1088, !dbg !16
  %236 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !16
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %235, <4 x float> %234) #8, !dbg !16
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !16
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %219, i64 5184, !dbg !16
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !16
  %240 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !16
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %239, <4 x float> %238) #8, !dbg !16
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %240, <4 x float> zeroinitializer) #8, !dbg !16
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !16
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %219, i64 1216, !dbg !16
  %244 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !16
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %243, <4 x float> %242) #8, !dbg !16
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %244, <4 x float> zeroinitializer) #8, !dbg !16
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %219, i64 5312, !dbg !16
  %247 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !16
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %247, <4 x float> %246) #8, !dbg !16
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %214, <4 x float> zeroinitializer) #8, !dbg !16
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %216, <4 x float> %249) #8, !dbg !16
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %220, <4 x float> zeroinitializer) #8, !dbg !16
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %223, <4 x float> %251) #8, !dbg !16
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !16
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %227, <4 x float> %253) #8, !dbg !16
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %228, <4 x float> zeroinitializer) #8, !dbg !16
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %231, <4 x float> %255) #8, !dbg !16
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %232, <4 x float> zeroinitializer) #8, !dbg !16
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %235, <4 x float> %257) #8, !dbg !16
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !16
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %239, <4 x float> %259) #8, !dbg !16
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %240, <4 x float> zeroinitializer) #8, !dbg !16
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %243, <4 x float> %261) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %244, <4 x float> zeroinitializer) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %247, <4 x float> %263) #8, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %265 = extractelement <2 x i1> %196, i64 1, !dbg !15
  %266 = sext i1 %265 to i64, !dbg !15
  %267 = extractelement <2 x i64> %202, i64 1, !dbg !15
  %268 = xor i64 %267, %266, !dbg !15
  %269 = shl i64 %193, 19, !dbg !15
  %270 = and i64 %269, 16252928, !dbg !15
  %271 = add i64 %.idx1822, %270, !dbg !15
  %272 = shl nsw i64 %268, 7, !dbg !15
  %273 = sub i64 %271, %272, !dbg !15
  %274 = add i64 %address.local449, %273, !dbg !15
  %.tr38 = trunc i64 %274 to i32, !dbg !15
  %275 = shl i32 %.tr38, 1, !dbg !15
  %dma.buffer.offset1576 = add i32 %275, 128, !dbg !15
  %276 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %277 = add i64 %address.local466, %273, !dbg !15
  %.tr44 = trunc i64 %277 to i32, !dbg !15
  %278 = shl i32 %.tr44, 1, !dbg !15
  %dma.buffer.offset1573 = add i32 %278, 128, !dbg !15
  %279 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !16
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !16
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %280, <4 x float> %221) #8, !dbg !16
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !16
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !16
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %219, i64 8256, !dbg !16
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !16
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %282, <4 x float> %281) #8, !dbg !16
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %283, <4 x float> %225) #8, !dbg !16
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %219, i64 12352, !dbg !16
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !16
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !16
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !16
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %286, <4 x float> %285) #8, !dbg !16
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %287, <4 x float> %229) #8, !dbg !16
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !16
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !16
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %219, i64 8384, !dbg !16
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !16
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %290, <4 x float> %289) #8, !dbg !16
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %291, <4 x float> %233) #8, !dbg !16
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %219, i64 12480, !dbg !16
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !16
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !16
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !16
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %294, <4 x float> %293) #8, !dbg !16
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %295, <4 x float> %237) #8, !dbg !16
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !16
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !16
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %219, i64 9280, !dbg !16
  %299 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !16
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %298, <4 x float> %297) #8, !dbg !16
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %299, <4 x float> %241) #8, !dbg !16
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %219, i64 13376, !dbg !16
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !16
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !16
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !16
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %302, <4 x float> %301) #8, !dbg !16
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %303, <4 x float> %245) #8, !dbg !16
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !16
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !16
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %219, i64 9408, !dbg !16
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !16
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %306, <4 x float> %305) #8, !dbg !16
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %307, <4 x float> %248) #8, !dbg !16
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %219, i64 13504, !dbg !16
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !16
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %310, <4 x float> %309) #8, !dbg !16
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %280, <4 x float> %250) #8, !dbg !16
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %282, <4 x float> %312) #8, !dbg !16
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %283, <4 x float> %252) #8, !dbg !16
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %286, <4 x float> %314) #8, !dbg !16
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %287, <4 x float> %254) #8, !dbg !16
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %290, <4 x float> %316) #8, !dbg !16
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %291, <4 x float> %256) #8, !dbg !16
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %294, <4 x float> %318) #8, !dbg !16
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %295, <4 x float> %258) #8, !dbg !16
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %298, <4 x float> %320) #8, !dbg !16
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %299, <4 x float> %260) #8, !dbg !16
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %302, <4 x float> %322) #8, !dbg !16
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %303, <4 x float> %262) #8, !dbg !16
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %306, <4 x float> %324) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %307, <4 x float> %264) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %310, <4 x float> %326) #8, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %328 = add nuw nsw i64 %148, 3072, !dbg !15
  %329 = lshr i64 %328, 12, !dbg !15
  %330 = add nuw nsw i64 %329, %16, !dbg !15
  %331 = add nsw i64 %165, 96, !dbg !15
  %332 = icmp slt i64 %165, -96, !dbg !15
  %333 = sub nsw i64 -97, %165, !dbg !15
  %334 = select i1 %332, i64 %333, i64 %331, !dbg !15
  %335 = sdiv i64 %334, 128, !dbg !15
  %336 = sext i1 %332 to i64, !dbg !15
  %337 = xor i64 %335, %336, !dbg !15
  %338 = shl i64 %330, 19, !dbg !15
  %339 = and i64 %338, 16252928, !dbg !15
  %340 = add i64 %.idx1822, %339, !dbg !15
  %341 = shl nsw i64 %337, 7, !dbg !15
  %342 = sub i64 %340, %341, !dbg !15
  %343 = add i64 %address.local449, %342, !dbg !15
  %.tr50 = trunc i64 %343 to i32, !dbg !15
  %344 = shl i32 %.tr50, 1, !dbg !15
  %dma.buffer.offset1570 = add i32 %344, 192, !dbg !15
  %345 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %346 = add i64 %address.local466, %342, !dbg !15
  %.tr56 = trunc i64 %346 to i32, !dbg !15
  %347 = shl i32 %.tr56, 1, !dbg !15
  %dma.buffer.offset1567 = add i32 %347, 192, !dbg !15
  %348 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1589) #9, !dbg !15
  %349 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !16
  %350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %349, <4 x float> %284) #8, !dbg !16
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !16
  %352 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !16
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %351, <4 x float> %350) #8, !dbg !16
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %352, <4 x float> %288) #8, !dbg !16
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !16
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !16
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %355, <4 x float> %354) #8, !dbg !16
  %358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %356, <4 x float> %292) #8, !dbg !16
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !16
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !16
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %359, <4 x float> %358) #8, !dbg !16
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %360, <4 x float> %296) #8, !dbg !16
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !16
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !16
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %363, <4 x float> %362) #8, !dbg !16
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %364, <4 x float> %300) #8, !dbg !16
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !16
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !16
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %367, <4 x float> %366) #8, !dbg !16
  %370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %368, <4 x float> %304) #8, !dbg !16
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !16
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !16
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %371, <4 x float> %370) #8, !dbg !16
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %372, <4 x float> %308) #8, !dbg !16
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !16
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !16
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %375, <4 x float> %374) #8, !dbg !16
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %376, <4 x float> %311) #8, !dbg !16
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !16
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %379, <4 x float> %378) #8, !dbg !16
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %349, <4 x float> %313) #8, !dbg !16
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %351, <4 x float> %381) #8, !dbg !16
  %383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %352, <4 x float> %315) #8, !dbg !16
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %355, <4 x float> %383) #8, !dbg !16
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %356, <4 x float> %317) #8, !dbg !16
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %359, <4 x float> %385) #8, !dbg !16
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %360, <4 x float> %319) #8, !dbg !16
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %363, <4 x float> %387) #8, !dbg !16
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %364, <4 x float> %321) #8, !dbg !16
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %367, <4 x float> %389) #8, !dbg !16
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %368, <4 x float> %323) #8, !dbg !16
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %371, <4 x float> %391) #8, !dbg !16
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %372, <4 x float> %325) #8, !dbg !16
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %375, <4 x float> %393) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %395 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %376, <4 x float> %327) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %396 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %379, <4 x float> %395) #8, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %397 = add nuw nsw i64 %152, %16, !dbg !18
  %398 = shl i64 %397, 19, !dbg !18
  %399 = and i64 %398, 16252928, !dbg !18
  %address.local852 = shl nuw nsw i64 %address.local434, 6, !dbg !18
  %address.local853 = and i64 %address.local852, 16128, !dbg !18
  %address.local855 = and i64 %address.local435, 24, !dbg !18
  %.idx5760 = add i64 %399, %158, !dbg !18
  %400 = or disjoint i64 %address.local855, %.idx5760, !dbg !18
  %401 = shl i64 %400, 1, !dbg !18
  %402 = or disjoint i64 %401, %address.local853, !dbg !18
  %dma.buffer.offset1564 = trunc i64 %402 to i32, !dbg !18
  %403 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %address.local858 = shl nuw nsw i64 %address.local451, 6, !dbg !18
  %address.local859 = and i64 %address.local858, 32512, !dbg !18
  %404 = or disjoint i64 %401, %address.local859, !dbg !18
  %dma.buffer.offset1561 = trunc i64 %404 to i32, !dbg !18
  %405 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %406 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !16
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %406, <4 x float> %353) #8, !dbg !16
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !16
  %409 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !16
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %408, <4 x float> %407) #8, !dbg !16
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %409, <4 x float> %357) #8, !dbg !16
  %412 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !16
  %413 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !16
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %412, <4 x float> %411) #8, !dbg !16
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %413, <4 x float> %361) #8, !dbg !16
  %416 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !16
  %417 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !16
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %416, <4 x float> %415) #8, !dbg !16
  %419 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %417, <4 x float> %365) #8, !dbg !16
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !16
  %421 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !16
  %422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %420, <4 x float> %419) #8, !dbg !16
  %423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %421, <4 x float> %369) #8, !dbg !16
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !16
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !16
  %426 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %424, <4 x float> %423) #8, !dbg !16
  %427 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %425, <4 x float> %373) #8, !dbg !16
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !16
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !16
  %430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %428, <4 x float> %427) #8, !dbg !16
  %431 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %429, <4 x float> %377) #8, !dbg !16
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !16
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !16
  %434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %432, <4 x float> %431) #8, !dbg !16
  %435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %433, <4 x float> %380) #8, !dbg !16
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !16
  %437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %436, <4 x float> %435) #8, !dbg !16
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %406, <4 x float> %382) #8, !dbg !16
  %439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %408, <4 x float> %438) #8, !dbg !16
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %409, <4 x float> %384) #8, !dbg !16
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %412, <4 x float> %440) #8, !dbg !16
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %413, <4 x float> %386) #8, !dbg !16
  %443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %416, <4 x float> %442) #8, !dbg !16
  %444 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %417, <4 x float> %388) #8, !dbg !16
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %420, <4 x float> %444) #8, !dbg !16
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %421, <4 x float> %390) #8, !dbg !16
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %424, <4 x float> %446) #8, !dbg !16
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %425, <4 x float> %392) #8, !dbg !16
  %449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %428, <4 x float> %448) #8, !dbg !16
  %450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %429, <4 x float> %394) #8, !dbg !16
  %451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %432, <4 x float> %450) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16
  %452 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %433, <4 x float> %396) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %453 = or disjoint i64 %149, 127, !dbg !12
  %454 = icmp uge i64 %24, %453, !dbg !12
  %455 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %436, <4 x float> %452) #8, !dbg !16
  %456 = and i1 %102, %454, !dbg !12
  br i1 %456, label %541, label %457, !dbg !12

457:                                              ; preds = %141
  %458 = or disjoint i64 %149, %130, !dbg !12
  %459 = lshr i64 %148, 25, !dbg !17
  %compare.high1157 = trunc nuw i64 %459 to i32, !dbg !17
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !17
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !17
  %compare.low.base1158 = trunc i64 %458 to i32, !dbg !17
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
  %460 = insertelement <32 x i32> poison, i32 %compare.low.base1158, i64 0, !dbg !17
  %461 = insertelement <32 x i32> %460, i32 %compare.local.base, i64 1, !dbg !17
  %462 = insertelement <32 x i32> %461, i32 %compare.local.base1161, i64 2, !dbg !17
  %463 = insertelement <32 x i32> %462, i32 %compare.local.base1165, i64 3, !dbg !17
  %464 = insertelement <32 x i32> %463, i32 %compare.local.base1169, i64 4, !dbg !17
  %465 = insertelement <32 x i32> %464, i32 %compare.local.base1173, i64 5, !dbg !17
  %466 = insertelement <32 x i32> %465, i32 %compare.local.base1177, i64 6, !dbg !17
  %467 = insertelement <32 x i32> %466, i32 %compare.local.base1181, i64 7, !dbg !17
  %468 = insertelement <32 x i32> %467, i32 %compare.local.base1185, i64 8, !dbg !17
  %469 = insertelement <32 x i32> %468, i32 %compare.local.base1189, i64 9, !dbg !17
  %470 = insertelement <32 x i32> %469, i32 %compare.local.base1193, i64 10, !dbg !17
  %471 = insertelement <32 x i32> %470, i32 %compare.local.base1197, i64 11, !dbg !17
  %472 = insertelement <32 x i32> %471, i32 %compare.local.base1201, i64 12, !dbg !17
  %473 = insertelement <32 x i32> %472, i32 %compare.local.base1205, i64 13, !dbg !17
  %474 = insertelement <32 x i32> %473, i32 %compare.local.base1209, i64 14, !dbg !17
  %475 = insertelement <32 x i32> %474, i32 %compare.local.base1213, i64 15, !dbg !17
  %476 = insertelement <32 x i32> %475, i32 %compare.local.base1217, i64 16, !dbg !17
  %477 = insertelement <32 x i32> %476, i32 %compare.local.base1221, i64 17, !dbg !17
  %478 = insertelement <32 x i32> %477, i32 %compare.local.base1225, i64 18, !dbg !17
  %479 = insertelement <32 x i32> %478, i32 %compare.local.base1229, i64 19, !dbg !17
  %480 = insertelement <32 x i32> %479, i32 %compare.local.base1233, i64 20, !dbg !17
  %481 = insertelement <32 x i32> %480, i32 %compare.local.base1237, i64 21, !dbg !17
  %482 = insertelement <32 x i32> %481, i32 %compare.local.base1241, i64 22, !dbg !17
  %483 = insertelement <32 x i32> %482, i32 %compare.local.base1245, i64 23, !dbg !17
  %484 = insertelement <32 x i32> %483, i32 %compare.local.base1249, i64 24, !dbg !17
  %485 = insertelement <32 x i32> %484, i32 %compare.local.base1253, i64 25, !dbg !17
  %486 = insertelement <32 x i32> %485, i32 %compare.local.base1257, i64 26, !dbg !17
  %487 = insertelement <32 x i32> %486, i32 %compare.local.base1261, i64 27, !dbg !17
  %488 = insertelement <32 x i32> %487, i32 %compare.local.base1265, i64 28, !dbg !17
  %489 = insertelement <32 x i32> %488, i32 %compare.local.base1269, i64 29, !dbg !17
  %490 = insertelement <32 x i32> %489, i32 %compare.local.base1273, i64 30, !dbg !17
  %491 = insertelement <32 x i32> %490, i32 %compare.local.base1277, i64 31, !dbg !17
  %492 = or <32 x i32> %491, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !17
  %493 = icmp ugt <32 x i32> %140, %492, !dbg !17
  %494 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !17
  %495 = shufflevector <32 x i1> %494, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !17
  %496 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !17
  %497 = shufflevector <32 x i1> %496, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !17
  %498 = select <32 x i1> %495, <32 x i1> %493, <32 x i1> %497, !dbg !17
  %499 = select <32 x i1> %498, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !19
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
  %500 = insertelement <32 x i32> %460, i32 %compare.local.base1289, i64 1, !dbg !17
  %501 = insertelement <32 x i32> %500, i32 %compare.local.base1293, i64 2, !dbg !17
  %502 = insertelement <32 x i32> %501, i32 %compare.local.base1297, i64 3, !dbg !17
  %503 = insertelement <32 x i32> %502, i32 %compare.local.base1301, i64 4, !dbg !17
  %504 = insertelement <32 x i32> %503, i32 %compare.local.base1305, i64 5, !dbg !17
  %505 = insertelement <32 x i32> %504, i32 %compare.local.base1309, i64 6, !dbg !17
  %506 = insertelement <32 x i32> %505, i32 %compare.local.base1313, i64 7, !dbg !17
  %507 = insertelement <32 x i32> %506, i32 %compare.local.base1317, i64 8, !dbg !17
  %508 = insertelement <32 x i32> %507, i32 %compare.local.base1321, i64 9, !dbg !17
  %509 = insertelement <32 x i32> %508, i32 %compare.local.base1325, i64 10, !dbg !17
  %510 = insertelement <32 x i32> %509, i32 %compare.local.base1329, i64 11, !dbg !17
  %511 = insertelement <32 x i32> %510, i32 %compare.local.base1333, i64 12, !dbg !17
  %512 = insertelement <32 x i32> %511, i32 %compare.local.base1337, i64 13, !dbg !17
  %513 = insertelement <32 x i32> %512, i32 %compare.local.base1341, i64 14, !dbg !17
  %514 = insertelement <32 x i32> %513, i32 %compare.local.base1345, i64 15, !dbg !17
  %515 = insertelement <32 x i32> %514, i32 %compare.local.base1349, i64 16, !dbg !17
  %516 = insertelement <32 x i32> %515, i32 %compare.local.base1353, i64 17, !dbg !17
  %517 = insertelement <32 x i32> %516, i32 %compare.local.base1357, i64 18, !dbg !17
  %518 = insertelement <32 x i32> %517, i32 %compare.local.base1361, i64 19, !dbg !17
  %519 = insertelement <32 x i32> %518, i32 %compare.local.base1365, i64 20, !dbg !17
  %520 = insertelement <32 x i32> %519, i32 %compare.local.base1369, i64 21, !dbg !17
  %521 = insertelement <32 x i32> %520, i32 %compare.local.base1373, i64 22, !dbg !17
  %522 = insertelement <32 x i32> %521, i32 %compare.local.base1377, i64 23, !dbg !17
  %523 = insertelement <32 x i32> %522, i32 %compare.local.base1381, i64 24, !dbg !17
  %524 = insertelement <32 x i32> %523, i32 %compare.local.base1385, i64 25, !dbg !17
  %525 = insertelement <32 x i32> %524, i32 %compare.local.base1389, i64 26, !dbg !17
  %526 = insertelement <32 x i32> %525, i32 %compare.local.base1393, i64 27, !dbg !17
  %527 = insertelement <32 x i32> %526, i32 %compare.local.base1397, i64 28, !dbg !17
  %528 = insertelement <32 x i32> %527, i32 %compare.local.base1401, i64 29, !dbg !17
  %529 = insertelement <32 x i32> %528, i32 %compare.local.base1405, i64 30, !dbg !17
  %530 = insertelement <32 x i32> %529, i32 %compare.local.base1409, i64 31, !dbg !17
  %531 = or <32 x i32> %530, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !17
  %532 = icmp ugt <32 x i32> %138, %531, !dbg !17
  %533 = insertelement <32 x i1> poison, i1 %compare.high.equal1283, i64 0, !dbg !17
  %534 = shufflevector <32 x i1> %533, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !17
  %535 = insertelement <32 x i1> poison, i1 %compare.high.order1284, i64 0, !dbg !17
  %536 = shufflevector <32 x i1> %535, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !17
  %537 = select <32 x i1> %534, <32 x i1> %532, <32 x i1> %536, !dbg !17
  %538 = select <32 x i1> %537, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !19
  %539 = insertvalue [2 x <32 x float>] poison, <32 x float> %499, 0, !dbg !12
  %540 = insertvalue [2 x <32 x float>] %539, <32 x float> %538, 1, !dbg !12
  br label %541, !dbg !12

541:                                              ; preds = %457, %141
  %542 = phi [2 x <32 x float>] [ %540, %457 ], [ zeroinitializer, %141 ], !dbg !12
  %543 = extractvalue [2 x <32 x float>] %542, 0, !dbg !20
  %544 = shufflevector <32 x float> %543, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %545 = fadd <4 x float> %410, %544, !dbg !20
  %546 = extractelement <4 x float> %545, i64 0, !dbg !20
  %547 = extractelement <4 x float> %545, i64 1, !dbg !20
  %548 = extractelement <4 x float> %545, i64 2, !dbg !20
  %549 = extractelement <4 x float> %545, i64 3, !dbg !20
  %550 = shufflevector <32 x float> %543, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %551 = fadd <4 x float> %414, %550, !dbg !20
  %552 = extractelement <4 x float> %551, i64 0, !dbg !20
  %553 = extractelement <4 x float> %551, i64 1, !dbg !20
  %554 = extractelement <4 x float> %551, i64 2, !dbg !20
  %555 = extractelement <4 x float> %551, i64 3, !dbg !20
  %556 = shufflevector <32 x float> %543, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !20
  %557 = fadd <4 x float> %418, %556, !dbg !20
  %558 = extractelement <4 x float> %557, i64 0, !dbg !20
  %559 = extractelement <4 x float> %557, i64 1, !dbg !20
  %560 = extractelement <4 x float> %557, i64 2, !dbg !20
  %561 = extractelement <4 x float> %557, i64 3, !dbg !20
  %562 = shufflevector <32 x float> %543, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %563 = fadd <4 x float> %422, %562, !dbg !20
  %564 = extractelement <4 x float> %563, i64 0, !dbg !20
  %565 = extractelement <4 x float> %563, i64 1, !dbg !20
  %566 = extractelement <4 x float> %563, i64 2, !dbg !20
  %567 = extractelement <4 x float> %563, i64 3, !dbg !20
  %568 = shufflevector <32 x float> %543, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !20
  %569 = fadd <4 x float> %426, %568, !dbg !20
  %570 = extractelement <4 x float> %569, i64 0, !dbg !20
  %571 = extractelement <4 x float> %569, i64 1, !dbg !20
  %572 = extractelement <4 x float> %569, i64 2, !dbg !20
  %573 = extractelement <4 x float> %569, i64 3, !dbg !20
  %574 = shufflevector <32 x float> %543, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !20
  %575 = fadd <4 x float> %430, %574, !dbg !20
  %576 = extractelement <4 x float> %575, i64 0, !dbg !20
  %577 = extractelement <4 x float> %575, i64 1, !dbg !20
  %578 = extractelement <4 x float> %575, i64 2, !dbg !20
  %579 = extractelement <4 x float> %575, i64 3, !dbg !20
  %580 = shufflevector <32 x float> %543, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !20
  %581 = fadd <4 x float> %434, %580, !dbg !20
  %582 = extractelement <4 x float> %581, i64 0, !dbg !20
  %583 = extractelement <4 x float> %581, i64 1, !dbg !20
  %584 = extractelement <4 x float> %581, i64 2, !dbg !20
  %585 = extractelement <4 x float> %581, i64 3, !dbg !20
  %586 = shufflevector <32 x float> %543, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %587 = fadd <4 x float> %437, %586, !dbg !20
  %588 = extractelement <4 x float> %587, i64 0, !dbg !20
  %589 = extractelement <4 x float> %587, i64 1, !dbg !20
  %590 = extractelement <4 x float> %587, i64 2, !dbg !20
  %591 = extractelement <4 x float> %587, i64 3, !dbg !20
  %592 = extractvalue [2 x <32 x float>] %542, 1, !dbg !20
  %593 = shufflevector <32 x float> %592, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %594 = fadd <4 x float> %439, %593, !dbg !20
  %595 = extractelement <4 x float> %594, i64 0, !dbg !20
  %596 = extractelement <4 x float> %594, i64 1, !dbg !20
  %597 = extractelement <4 x float> %594, i64 2, !dbg !20
  %598 = extractelement <4 x float> %594, i64 3, !dbg !20
  %599 = shufflevector <32 x float> %592, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %600 = fadd <4 x float> %441, %599, !dbg !20
  %601 = extractelement <4 x float> %600, i64 0, !dbg !20
  %602 = extractelement <4 x float> %600, i64 1, !dbg !20
  %603 = extractelement <4 x float> %600, i64 2, !dbg !20
  %604 = extractelement <4 x float> %600, i64 3, !dbg !20
  %605 = shufflevector <32 x float> %592, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !20
  %606 = fadd <4 x float> %443, %605, !dbg !20
  %607 = extractelement <4 x float> %606, i64 0, !dbg !20
  %608 = extractelement <4 x float> %606, i64 1, !dbg !20
  %609 = extractelement <4 x float> %606, i64 2, !dbg !20
  %610 = extractelement <4 x float> %606, i64 3, !dbg !20
  %611 = shufflevector <32 x float> %592, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %612 = fadd <4 x float> %445, %611, !dbg !20
  %613 = extractelement <4 x float> %612, i64 0, !dbg !20
  %614 = extractelement <4 x float> %612, i64 1, !dbg !20
  %615 = extractelement <4 x float> %612, i64 2, !dbg !20
  %616 = extractelement <4 x float> %612, i64 3, !dbg !20
  %617 = shufflevector <32 x float> %592, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !20
  %618 = fadd <4 x float> %447, %617, !dbg !20
  %619 = extractelement <4 x float> %618, i64 0, !dbg !20
  %620 = extractelement <4 x float> %618, i64 1, !dbg !20
  %621 = extractelement <4 x float> %618, i64 2, !dbg !20
  %622 = extractelement <4 x float> %618, i64 3, !dbg !20
  %623 = shufflevector <32 x float> %592, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !20
  %624 = fadd <4 x float> %449, %623, !dbg !20
  %625 = extractelement <4 x float> %624, i64 0, !dbg !20
  %626 = extractelement <4 x float> %624, i64 1, !dbg !20
  %627 = extractelement <4 x float> %624, i64 2, !dbg !20
  %628 = extractelement <4 x float> %624, i64 3, !dbg !20
  %629 = shufflevector <32 x float> %592, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !20
  %630 = fadd <4 x float> %451, %629, !dbg !20
  %631 = extractelement <4 x float> %630, i64 0, !dbg !20
  %632 = extractelement <4 x float> %630, i64 1, !dbg !20
  %633 = extractelement <4 x float> %630, i64 2, !dbg !20
  %634 = extractelement <4 x float> %630, i64 3, !dbg !20
  %635 = shufflevector <32 x float> %592, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %636 = fadd <4 x float> %455, %635, !dbg !20
  %637 = extractelement <4 x float> %636, i64 0, !dbg !20
  %638 = extractelement <4 x float> %636, i64 1, !dbg !20
  %639 = extractelement <4 x float> %636, i64 2, !dbg !20
  %640 = extractelement <4 x float> %636, i64 3, !dbg !20
  %641 = tail call float @llvm.maxnum.f32(float %546, float -inf), !dbg !21
  %642 = tail call float @llvm.maxnum.f32(float %641, float %547), !dbg !21
  %643 = tail call float @llvm.maxnum.f32(float %642, float %548), !dbg !21
  %644 = tail call float @llvm.maxnum.f32(float %643, float %549), !dbg !21
  %645 = tail call float @llvm.maxnum.f32(float %644, float %552), !dbg !21
  %646 = tail call float @llvm.maxnum.f32(float %645, float %553), !dbg !21
  %647 = tail call float @llvm.maxnum.f32(float %646, float %554), !dbg !21
  %648 = tail call float @llvm.maxnum.f32(float %647, float %555), !dbg !21
  %649 = tail call float @llvm.maxnum.f32(float %648, float %558), !dbg !21
  %650 = tail call float @llvm.maxnum.f32(float %649, float %559), !dbg !21
  %651 = tail call float @llvm.maxnum.f32(float %650, float %560), !dbg !21
  %652 = tail call float @llvm.maxnum.f32(float %651, float %561), !dbg !21
  %653 = tail call float @llvm.maxnum.f32(float %652, float %564), !dbg !21
  %654 = tail call float @llvm.maxnum.f32(float %653, float %565), !dbg !21
  %655 = tail call float @llvm.maxnum.f32(float %654, float %566), !dbg !21
  %656 = tail call float @llvm.maxnum.f32(float %655, float %567), !dbg !21
  %657 = tail call float @llvm.maxnum.f32(float %656, float %570), !dbg !21
  %658 = tail call float @llvm.maxnum.f32(float %657, float %571), !dbg !21
  %659 = tail call float @llvm.maxnum.f32(float %658, float %572), !dbg !21
  %660 = tail call float @llvm.maxnum.f32(float %659, float %573), !dbg !21
  %661 = tail call float @llvm.maxnum.f32(float %660, float %576), !dbg !21
  %662 = tail call float @llvm.maxnum.f32(float %661, float %577), !dbg !21
  %663 = tail call float @llvm.maxnum.f32(float %662, float %578), !dbg !21
  %664 = tail call float @llvm.maxnum.f32(float %663, float %579), !dbg !21
  %665 = tail call float @llvm.maxnum.f32(float %664, float %582), !dbg !21
  %666 = tail call float @llvm.maxnum.f32(float %665, float %583), !dbg !21
  %667 = tail call float @llvm.maxnum.f32(float %666, float %584), !dbg !21
  %668 = tail call float @llvm.maxnum.f32(float %667, float %585), !dbg !21
  %669 = tail call float @llvm.maxnum.f32(float %668, float %588), !dbg !21
  %670 = tail call float @llvm.maxnum.f32(float %669, float %589), !dbg !21
  %671 = tail call float @llvm.maxnum.f32(float %670, float %590), !dbg !21
  %672 = tail call float @llvm.maxnum.f32(float %671, float %591), !dbg !21
  %673 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !21
  %674 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %673), !dbg !21
  %675 = add nuw nsw i32 %674, 64, !dbg !21
  %676 = xor i32 %674, 16, !dbg !21
  %677 = and i32 %675, 192, !dbg !21
  %678 = icmp samesign ult i32 %676, %677, !dbg !21
  %679 = select i1 %678, i32 %676, i32 %674, !dbg !21
  %680 = shl nuw nsw i32 %679, 2, !dbg !21
  %681 = bitcast float %672 to i32, !dbg !21
  %682 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %681), !dbg !21
  %683 = bitcast i32 %682 to float, !dbg !21
  %684 = tail call float @llvm.maxnum.f32(float %672, float %683), !dbg !21
  %685 = xor i32 %674, 32, !dbg !21
  %686 = icmp samesign ult i32 %685, %677, !dbg !21
  %687 = select i1 %686, i32 %685, i32 %674, !dbg !21
  %688 = shl nuw nsw i32 %687, 2, !dbg !21
  %689 = bitcast float %684 to i32, !dbg !21
  %690 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %689), !dbg !21
  %691 = bitcast i32 %690 to float, !dbg !21
  %692 = tail call float @llvm.maxnum.f32(float %684, float %691), !dbg !21
  %693 = extractvalue [2 x <1 x float>] %145, 0, !dbg !21
  %694 = insertelement <1 x float> poison, float %692, i64 0, !dbg !21
  %695 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %694, <1 x float> %693), !dbg !21
  %696 = tail call float @llvm.maxnum.f32(float %595, float -inf), !dbg !21
  %697 = tail call float @llvm.maxnum.f32(float %696, float %596), !dbg !21
  %698 = tail call float @llvm.maxnum.f32(float %697, float %597), !dbg !21
  %699 = tail call float @llvm.maxnum.f32(float %698, float %598), !dbg !21
  %700 = tail call float @llvm.maxnum.f32(float %699, float %601), !dbg !21
  %701 = tail call float @llvm.maxnum.f32(float %700, float %602), !dbg !21
  %702 = tail call float @llvm.maxnum.f32(float %701, float %603), !dbg !21
  %703 = tail call float @llvm.maxnum.f32(float %702, float %604), !dbg !21
  %704 = tail call float @llvm.maxnum.f32(float %703, float %607), !dbg !21
  %705 = tail call float @llvm.maxnum.f32(float %704, float %608), !dbg !21
  %706 = tail call float @llvm.maxnum.f32(float %705, float %609), !dbg !21
  %707 = tail call float @llvm.maxnum.f32(float %706, float %610), !dbg !21
  %708 = tail call float @llvm.maxnum.f32(float %707, float %613), !dbg !21
  %709 = tail call float @llvm.maxnum.f32(float %708, float %614), !dbg !21
  %710 = tail call float @llvm.maxnum.f32(float %709, float %615), !dbg !21
  %711 = tail call float @llvm.maxnum.f32(float %710, float %616), !dbg !21
  %712 = tail call float @llvm.maxnum.f32(float %711, float %619), !dbg !21
  %713 = tail call float @llvm.maxnum.f32(float %712, float %620), !dbg !21
  %714 = tail call float @llvm.maxnum.f32(float %713, float %621), !dbg !21
  %715 = tail call float @llvm.maxnum.f32(float %714, float %622), !dbg !21
  %716 = tail call float @llvm.maxnum.f32(float %715, float %625), !dbg !21
  %717 = tail call float @llvm.maxnum.f32(float %716, float %626), !dbg !21
  %718 = tail call float @llvm.maxnum.f32(float %717, float %627), !dbg !21
  %719 = tail call float @llvm.maxnum.f32(float %718, float %628), !dbg !21
  %720 = tail call float @llvm.maxnum.f32(float %719, float %631), !dbg !21
  %721 = tail call float @llvm.maxnum.f32(float %720, float %632), !dbg !21
  %722 = tail call float @llvm.maxnum.f32(float %721, float %633), !dbg !21
  %723 = tail call float @llvm.maxnum.f32(float %722, float %634), !dbg !21
  %724 = tail call float @llvm.maxnum.f32(float %723, float %637), !dbg !21
  %725 = tail call float @llvm.maxnum.f32(float %724, float %638), !dbg !21
  %726 = tail call float @llvm.maxnum.f32(float %725, float %639), !dbg !21
  %727 = tail call float @llvm.maxnum.f32(float %726, float %640), !dbg !21
  %728 = bitcast float %727 to i32, !dbg !21
  %729 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %728), !dbg !21
  %730 = bitcast i32 %729 to float, !dbg !21
  %731 = tail call float @llvm.maxnum.f32(float %727, float %730), !dbg !21
  %732 = bitcast float %731 to i32, !dbg !21
  %733 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %732), !dbg !21
  %734 = bitcast i32 %733 to float, !dbg !21
  %735 = tail call float @llvm.maxnum.f32(float %731, float %734), !dbg !21
  %736 = extractvalue [2 x <1 x float>] %145, 1, !dbg !21
  %737 = insertelement <1 x float> poison, float %735, i64 0, !dbg !21
  %738 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %737, <1 x float> %736), !dbg !21
  %739 = shufflevector <1 x float> %695, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !22
  %740 = fsub <4 x float> %545, %739, !dbg !22
  %741 = fsub <4 x float> %551, %739, !dbg !22
  %742 = fsub <4 x float> %557, %739, !dbg !22
  %743 = fsub <4 x float> %563, %739, !dbg !22
  %744 = fsub <4 x float> %569, %739, !dbg !22
  %745 = fsub <4 x float> %575, %739, !dbg !22
  %746 = fsub <4 x float> %581, %739, !dbg !22
  %747 = fsub <4 x float> %587, %739, !dbg !22
  %748 = shufflevector <1 x float> %738, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !22
  %749 = fsub <4 x float> %594, %748, !dbg !22
  %750 = fsub <4 x float> %600, %748, !dbg !22
  %751 = fsub <4 x float> %606, %748, !dbg !22
  %752 = fsub <4 x float> %612, %748, !dbg !22
  %753 = fsub <4 x float> %618, %748, !dbg !22
  %754 = fsub <4 x float> %624, %748, !dbg !22
  %755 = fsub <4 x float> %630, %748, !dbg !22
  %756 = fsub <4 x float> %636, %748, !dbg !22
  %757 = fsub <1 x float> %693, %695, !dbg !23
  %758 = extractelement <1 x float> %757, i64 0, !dbg !24
  %759 = tail call float @llvm.amdgcn.exp2.f32(float %758), !dbg !24
  %760 = insertelement <4 x float> poison, float %759, i64 0, !dbg !25
  %761 = shufflevector <4 x float> %760, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %762 = extractvalue [2 x <1 x float>] %146, 0, !dbg !26
  %763 = extractelement <1 x float> %762, i64 0, !dbg !26
  %764 = fmul float %763, %759, !dbg !26
  %765 = fsub <1 x float> %736, %738, !dbg !23
  %766 = extractelement <1 x float> %765, i64 0, !dbg !24
  %767 = tail call float @llvm.amdgcn.exp2.f32(float %766), !dbg !24
  %768 = insertelement <4 x float> poison, float %767, i64 0, !dbg !25
  %769 = shufflevector <4 x float> %768, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %770 = extractvalue [2 x <1 x float>] %146, 1, !dbg !26
  %771 = extractelement <1 x float> %770, i64 0, !dbg !26
  %772 = fmul float %771, %767, !dbg !26
  %773 = extractelement <4 x float> %740, i64 0, !dbg !27
  %774 = tail call float @llvm.amdgcn.exp2.f32(float %773), !dbg !27
  %775 = insertelement <4 x float> poison, float %774, i64 0, !dbg !28
  %776 = extractelement <4 x float> %740, i64 1, !dbg !27
  %777 = tail call float @llvm.amdgcn.exp2.f32(float %776), !dbg !27
  %778 = insertelement <4 x float> %775, float %777, i64 1, !dbg !28
  %779 = fadd float %774, %777, !dbg !29
  %780 = extractelement <4 x float> %740, i64 2, !dbg !27
  %781 = tail call float @llvm.amdgcn.exp2.f32(float %780), !dbg !27
  %782 = insertelement <4 x float> %778, float %781, i64 2, !dbg !28
  %783 = fadd float %781, %779, !dbg !29
  %784 = extractelement <4 x float> %740, i64 3, !dbg !27
  %785 = tail call float @llvm.amdgcn.exp2.f32(float %784), !dbg !27
  %786 = insertelement <4 x float> %782, float %785, i64 3, !dbg !28
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %786) #8, !dbg !28, !frisk.register_lifetimes !5
  %787 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !28
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %787) #8, !dbg !29, !frisk.register_lifetimes !5
  %788 = fadd float %785, %783, !dbg !29
  %789 = extractelement <4 x float> %741, i64 0, !dbg !27
  %790 = tail call float @llvm.amdgcn.exp2.f32(float %789), !dbg !27
  %791 = insertelement <4 x float> poison, float %790, i64 0, !dbg !28
  %792 = fadd float %790, %788, !dbg !29
  %793 = extractelement <4 x float> %741, i64 1, !dbg !27
  %794 = tail call float @llvm.amdgcn.exp2.f32(float %793), !dbg !27
  %795 = insertelement <4 x float> %791, float %794, i64 1, !dbg !28
  %796 = fadd float %794, %792, !dbg !29
  %797 = extractelement <4 x float> %741, i64 2, !dbg !27
  %798 = tail call float @llvm.amdgcn.exp2.f32(float %797), !dbg !27
  %799 = insertelement <4 x float> %795, float %798, i64 2, !dbg !28
  %800 = fadd float %798, %796, !dbg !29
  %801 = extractelement <4 x float> %741, i64 3, !dbg !27
  %802 = tail call float @llvm.amdgcn.exp2.f32(float %801), !dbg !27
  %803 = insertelement <4 x float> %799, float %802, i64 3, !dbg !28
  %math.result309 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %803) #8, !dbg !28, !frisk.register_lifetimes !5
  %804 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !28
  %math.packed310 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %804) #8, !dbg !29, !frisk.register_lifetimes !5
  %805 = fadd float %802, %800, !dbg !29
  %806 = extractelement <4 x float> %742, i64 0, !dbg !27
  %807 = tail call float @llvm.amdgcn.exp2.f32(float %806), !dbg !27
  %808 = insertelement <4 x float> poison, float %807, i64 0, !dbg !28
  %809 = fadd float %807, %805, !dbg !29
  %810 = extractelement <4 x float> %742, i64 1, !dbg !27
  %811 = tail call float @llvm.amdgcn.exp2.f32(float %810), !dbg !27
  %812 = insertelement <4 x float> %808, float %811, i64 1, !dbg !28
  %813 = fadd float %811, %809, !dbg !29
  %814 = extractelement <4 x float> %742, i64 2, !dbg !27
  %815 = tail call float @llvm.amdgcn.exp2.f32(float %814), !dbg !27
  %816 = insertelement <4 x float> %812, float %815, i64 2, !dbg !28
  %817 = fadd float %815, %813, !dbg !29
  %818 = extractelement <4 x float> %742, i64 3, !dbg !27
  %819 = tail call float @llvm.amdgcn.exp2.f32(float %818), !dbg !27
  %820 = insertelement <4 x float> %816, float %819, i64 3, !dbg !28
  %math.result311 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %820) #8, !dbg !28, !frisk.register_lifetimes !5
  %821 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !28
  %math.packed312 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %821) #8, !dbg !29, !frisk.register_lifetimes !5
  %822 = fadd float %819, %817, !dbg !29
  %823 = extractelement <4 x float> %743, i64 0, !dbg !27
  %824 = tail call float @llvm.amdgcn.exp2.f32(float %823), !dbg !27
  %825 = insertelement <4 x float> poison, float %824, i64 0, !dbg !28
  %826 = fadd float %824, %822, !dbg !29
  %827 = extractelement <4 x float> %743, i64 1, !dbg !27
  %828 = tail call float @llvm.amdgcn.exp2.f32(float %827), !dbg !27
  %829 = insertelement <4 x float> %825, float %828, i64 1, !dbg !28
  %830 = fadd float %828, %826, !dbg !29
  %831 = extractelement <4 x float> %743, i64 2, !dbg !27
  %832 = tail call float @llvm.amdgcn.exp2.f32(float %831), !dbg !27
  %833 = insertelement <4 x float> %829, float %832, i64 2, !dbg !28
  %834 = fadd float %832, %830, !dbg !29
  %835 = extractelement <4 x float> %743, i64 3, !dbg !27
  %836 = tail call float @llvm.amdgcn.exp2.f32(float %835), !dbg !27
  %837 = insertelement <4 x float> %833, float %836, i64 3, !dbg !28
  %math.result313 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %837) #8, !dbg !28, !frisk.register_lifetimes !5
  %838 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !28
  %math.packed314 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %838) #8, !dbg !29, !frisk.register_lifetimes !5
  %839 = fadd float %836, %834, !dbg !29
  %840 = extractelement <4 x float> %744, i64 0, !dbg !27
  %841 = tail call float @llvm.amdgcn.exp2.f32(float %840), !dbg !27
  %842 = insertelement <4 x float> poison, float %841, i64 0, !dbg !28
  %843 = fadd float %841, %839, !dbg !29
  %844 = extractelement <4 x float> %744, i64 1, !dbg !27
  %845 = tail call float @llvm.amdgcn.exp2.f32(float %844), !dbg !27
  %846 = insertelement <4 x float> %842, float %845, i64 1, !dbg !28
  %847 = fadd float %845, %843, !dbg !29
  %848 = extractelement <4 x float> %744, i64 2, !dbg !27
  %849 = tail call float @llvm.amdgcn.exp2.f32(float %848), !dbg !27
  %850 = insertelement <4 x float> %846, float %849, i64 2, !dbg !28
  %851 = fadd float %849, %847, !dbg !29
  %852 = extractelement <4 x float> %744, i64 3, !dbg !27
  %853 = tail call float @llvm.amdgcn.exp2.f32(float %852), !dbg !27
  %854 = insertelement <4 x float> %850, float %853, i64 3, !dbg !28
  %math.result315 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %854) #8, !dbg !28, !frisk.register_lifetimes !5
  %855 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !28
  %math.packed316 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %855) #8, !dbg !29, !frisk.register_lifetimes !5
  %856 = fadd float %853, %851, !dbg !29
  %857 = extractelement <4 x float> %745, i64 0, !dbg !27
  %858 = tail call float @llvm.amdgcn.exp2.f32(float %857), !dbg !27
  %859 = insertelement <4 x float> poison, float %858, i64 0, !dbg !28
  %860 = fadd float %858, %856, !dbg !29
  %861 = extractelement <4 x float> %745, i64 1, !dbg !27
  %862 = tail call float @llvm.amdgcn.exp2.f32(float %861), !dbg !27
  %863 = insertelement <4 x float> %859, float %862, i64 1, !dbg !28
  %864 = fadd float %862, %860, !dbg !29
  %865 = extractelement <4 x float> %745, i64 2, !dbg !27
  %866 = tail call float @llvm.amdgcn.exp2.f32(float %865), !dbg !27
  %867 = insertelement <4 x float> %863, float %866, i64 2, !dbg !28
  %868 = fadd float %866, %864, !dbg !29
  %869 = extractelement <4 x float> %745, i64 3, !dbg !27
  %870 = tail call float @llvm.amdgcn.exp2.f32(float %869), !dbg !27
  %871 = insertelement <4 x float> %867, float %870, i64 3, !dbg !28
  %math.result317 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %871) #8, !dbg !28, !frisk.register_lifetimes !5
  %872 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !28
  %math.packed318 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %872) #8, !dbg !29, !frisk.register_lifetimes !5
  %873 = fadd float %870, %868, !dbg !29
  %874 = extractelement <4 x float> %746, i64 0, !dbg !27
  %875 = tail call float @llvm.amdgcn.exp2.f32(float %874), !dbg !27
  %876 = insertelement <4 x float> poison, float %875, i64 0, !dbg !28
  %877 = fadd float %875, %873, !dbg !29
  %878 = extractelement <4 x float> %746, i64 1, !dbg !27
  %879 = tail call float @llvm.amdgcn.exp2.f32(float %878), !dbg !27
  %880 = insertelement <4 x float> %876, float %879, i64 1, !dbg !28
  %881 = fadd float %879, %877, !dbg !29
  %882 = extractelement <4 x float> %746, i64 2, !dbg !27
  %883 = tail call float @llvm.amdgcn.exp2.f32(float %882), !dbg !27
  %884 = insertelement <4 x float> %880, float %883, i64 2, !dbg !28
  %885 = fadd float %883, %881, !dbg !29
  %886 = extractelement <4 x float> %746, i64 3, !dbg !27
  %887 = tail call float @llvm.amdgcn.exp2.f32(float %886), !dbg !27
  %888 = insertelement <4 x float> %884, float %887, i64 3, !dbg !28
  %math.result319 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %888) #8, !dbg !28, !frisk.register_lifetimes !5
  %889 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !28
  %math.packed320 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %889) #8, !dbg !29, !frisk.register_lifetimes !5
  %890 = fadd float %887, %885, !dbg !29
  %891 = extractelement <4 x float> %747, i64 0, !dbg !27
  %892 = tail call float @llvm.amdgcn.exp2.f32(float %891), !dbg !27
  %893 = insertelement <4 x float> poison, float %892, i64 0, !dbg !28
  %894 = fadd float %892, %890, !dbg !29
  %895 = extractelement <4 x float> %747, i64 1, !dbg !27
  %896 = tail call float @llvm.amdgcn.exp2.f32(float %895), !dbg !27
  %897 = insertelement <4 x float> %893, float %896, i64 1, !dbg !28
  %898 = fadd float %896, %894, !dbg !29
  %899 = extractelement <4 x float> %747, i64 2, !dbg !27
  %900 = tail call float @llvm.amdgcn.exp2.f32(float %899), !dbg !27
  %901 = insertelement <4 x float> %897, float %900, i64 2, !dbg !28
  %902 = fadd float %900, %898, !dbg !29
  %903 = extractelement <4 x float> %747, i64 3, !dbg !27
  %904 = tail call float @llvm.amdgcn.exp2.f32(float %903), !dbg !27
  %905 = insertelement <4 x float> %901, float %904, i64 3, !dbg !28
  %math.result321 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %905) #8, !dbg !28, !frisk.register_lifetimes !5
  %906 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !28
  %math.packed322 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %906) #8, !dbg !29, !frisk.register_lifetimes !5
  %907 = fadd float %904, %902, !dbg !29
  %908 = extractelement <4 x float> %749, i64 0, !dbg !27
  %909 = tail call float @llvm.amdgcn.exp2.f32(float %908), !dbg !27
  %910 = insertelement <4 x float> poison, float %909, i64 0, !dbg !28
  %911 = extractelement <4 x float> %749, i64 1, !dbg !27
  %912 = tail call float @llvm.amdgcn.exp2.f32(float %911), !dbg !27
  %913 = insertelement <4 x float> %910, float %912, i64 1, !dbg !28
  %914 = fadd float %909, %912, !dbg !29
  %915 = extractelement <4 x float> %749, i64 2, !dbg !27
  %916 = tail call float @llvm.amdgcn.exp2.f32(float %915), !dbg !27
  %917 = insertelement <4 x float> %913, float %916, i64 2, !dbg !28
  %918 = fadd float %916, %914, !dbg !29
  %919 = extractelement <4 x float> %749, i64 3, !dbg !27
  %920 = tail call float @llvm.amdgcn.exp2.f32(float %919), !dbg !27
  %921 = insertelement <4 x float> %917, float %920, i64 3, !dbg !28
  %math.result323 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %921) #8, !dbg !28, !frisk.register_lifetimes !5
  %922 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !28
  %math.packed324 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %922) #8, !dbg !29, !frisk.register_lifetimes !5
  %923 = fadd float %920, %918, !dbg !29
  %924 = extractelement <4 x float> %750, i64 0, !dbg !27
  %925 = tail call float @llvm.amdgcn.exp2.f32(float %924), !dbg !27
  %926 = insertelement <4 x float> poison, float %925, i64 0, !dbg !28
  %927 = fadd float %925, %923, !dbg !29
  %928 = extractelement <4 x float> %750, i64 1, !dbg !27
  %929 = tail call float @llvm.amdgcn.exp2.f32(float %928), !dbg !27
  %930 = insertelement <4 x float> %926, float %929, i64 1, !dbg !28
  %931 = fadd float %929, %927, !dbg !29
  %932 = extractelement <4 x float> %750, i64 2, !dbg !27
  %933 = tail call float @llvm.amdgcn.exp2.f32(float %932), !dbg !27
  %934 = insertelement <4 x float> %930, float %933, i64 2, !dbg !28
  %935 = fadd float %933, %931, !dbg !29
  %936 = extractelement <4 x float> %750, i64 3, !dbg !27
  %937 = tail call float @llvm.amdgcn.exp2.f32(float %936), !dbg !27
  %938 = insertelement <4 x float> %934, float %937, i64 3, !dbg !28
  %math.result325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %938) #8, !dbg !28, !frisk.register_lifetimes !5
  %939 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !28
  %math.packed326 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %939) #8, !dbg !29, !frisk.register_lifetimes !5
  %940 = fadd float %937, %935, !dbg !29
  %941 = extractelement <4 x float> %751, i64 0, !dbg !27
  %942 = tail call float @llvm.amdgcn.exp2.f32(float %941), !dbg !27
  %943 = insertelement <4 x float> poison, float %942, i64 0, !dbg !28
  %944 = fadd float %942, %940, !dbg !29
  %945 = extractelement <4 x float> %751, i64 1, !dbg !27
  %946 = tail call float @llvm.amdgcn.exp2.f32(float %945), !dbg !27
  %947 = insertelement <4 x float> %943, float %946, i64 1, !dbg !28
  %948 = fadd float %946, %944, !dbg !29
  %949 = extractelement <4 x float> %751, i64 2, !dbg !27
  %950 = tail call float @llvm.amdgcn.exp2.f32(float %949), !dbg !27
  %951 = insertelement <4 x float> %947, float %950, i64 2, !dbg !28
  %952 = fadd float %950, %948, !dbg !29
  %953 = extractelement <4 x float> %751, i64 3, !dbg !27
  %954 = tail call float @llvm.amdgcn.exp2.f32(float %953), !dbg !27
  %955 = insertelement <4 x float> %951, float %954, i64 3, !dbg !28
  %math.result327 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %955) #8, !dbg !28, !frisk.register_lifetimes !5
  %956 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !28
  %math.packed328 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %956) #8, !dbg !29, !frisk.register_lifetimes !5
  %957 = fadd float %954, %952, !dbg !29
  %958 = extractelement <4 x float> %752, i64 0, !dbg !27
  %959 = tail call float @llvm.amdgcn.exp2.f32(float %958), !dbg !27
  %960 = insertelement <4 x float> poison, float %959, i64 0, !dbg !28
  %961 = fadd float %959, %957, !dbg !29
  %962 = extractelement <4 x float> %752, i64 1, !dbg !27
  %963 = tail call float @llvm.amdgcn.exp2.f32(float %962), !dbg !27
  %964 = insertelement <4 x float> %960, float %963, i64 1, !dbg !28
  %965 = fadd float %963, %961, !dbg !29
  %966 = extractelement <4 x float> %752, i64 2, !dbg !27
  %967 = tail call float @llvm.amdgcn.exp2.f32(float %966), !dbg !27
  %968 = insertelement <4 x float> %964, float %967, i64 2, !dbg !28
  %969 = fadd float %967, %965, !dbg !29
  %970 = extractelement <4 x float> %752, i64 3, !dbg !27
  %971 = tail call float @llvm.amdgcn.exp2.f32(float %970), !dbg !27
  %972 = insertelement <4 x float> %968, float %971, i64 3, !dbg !28
  %math.result329 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %972) #8, !dbg !28, !frisk.register_lifetimes !5
  %973 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !28
  %math.packed330 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %973) #8, !dbg !29, !frisk.register_lifetimes !5
  %974 = fadd float %971, %969, !dbg !29
  %975 = extractelement <4 x float> %753, i64 0, !dbg !27
  %976 = tail call float @llvm.amdgcn.exp2.f32(float %975), !dbg !27
  %977 = insertelement <4 x float> poison, float %976, i64 0, !dbg !28
  %978 = fadd float %976, %974, !dbg !29
  %979 = extractelement <4 x float> %753, i64 1, !dbg !27
  %980 = tail call float @llvm.amdgcn.exp2.f32(float %979), !dbg !27
  %981 = insertelement <4 x float> %977, float %980, i64 1, !dbg !28
  %982 = fadd float %980, %978, !dbg !29
  %983 = extractelement <4 x float> %753, i64 2, !dbg !27
  %984 = tail call float @llvm.amdgcn.exp2.f32(float %983), !dbg !27
  %985 = insertelement <4 x float> %981, float %984, i64 2, !dbg !28
  %986 = fadd float %984, %982, !dbg !29
  %987 = extractelement <4 x float> %753, i64 3, !dbg !27
  %988 = tail call float @llvm.amdgcn.exp2.f32(float %987), !dbg !27
  %989 = insertelement <4 x float> %985, float %988, i64 3, !dbg !28
  %math.result331 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %989) #8, !dbg !28, !frisk.register_lifetimes !5
  %990 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !28
  %math.packed332 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %990) #8, !dbg !29, !frisk.register_lifetimes !5
  %991 = fadd float %988, %986, !dbg !29
  %992 = extractelement <4 x float> %754, i64 0, !dbg !27
  %993 = tail call float @llvm.amdgcn.exp2.f32(float %992), !dbg !27
  %994 = insertelement <4 x float> poison, float %993, i64 0, !dbg !28
  %995 = fadd float %993, %991, !dbg !29
  %996 = extractelement <4 x float> %754, i64 1, !dbg !27
  %997 = tail call float @llvm.amdgcn.exp2.f32(float %996), !dbg !27
  %998 = insertelement <4 x float> %994, float %997, i64 1, !dbg !28
  %999 = fadd float %997, %995, !dbg !29
  %1000 = extractelement <4 x float> %754, i64 2, !dbg !27
  %1001 = tail call float @llvm.amdgcn.exp2.f32(float %1000), !dbg !27
  %1002 = insertelement <4 x float> %998, float %1001, i64 2, !dbg !28
  %1003 = fadd float %1001, %999, !dbg !29
  %1004 = extractelement <4 x float> %754, i64 3, !dbg !27
  %1005 = tail call float @llvm.amdgcn.exp2.f32(float %1004), !dbg !27
  %1006 = insertelement <4 x float> %1002, float %1005, i64 3, !dbg !28
  %math.result333 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1006) #8, !dbg !28, !frisk.register_lifetimes !5
  %1007 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !28
  %math.packed334 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1007) #8, !dbg !29, !frisk.register_lifetimes !5
  %1008 = fadd float %1005, %1003, !dbg !29
  %1009 = extractelement <4 x float> %755, i64 0, !dbg !27
  %1010 = tail call float @llvm.amdgcn.exp2.f32(float %1009), !dbg !27
  %1011 = insertelement <4 x float> poison, float %1010, i64 0, !dbg !28
  %1012 = fadd float %1010, %1008, !dbg !29
  %1013 = extractelement <4 x float> %755, i64 1, !dbg !27
  %1014 = tail call float @llvm.amdgcn.exp2.f32(float %1013), !dbg !27
  %1015 = insertelement <4 x float> %1011, float %1014, i64 1, !dbg !28
  %1016 = fadd float %1014, %1012, !dbg !29
  %1017 = extractelement <4 x float> %755, i64 2, !dbg !27
  %1018 = tail call float @llvm.amdgcn.exp2.f32(float %1017), !dbg !27
  %1019 = insertelement <4 x float> %1015, float %1018, i64 2, !dbg !28
  %1020 = fadd float %1018, %1016, !dbg !29
  %1021 = extractelement <4 x float> %755, i64 3, !dbg !27
  %1022 = tail call float @llvm.amdgcn.exp2.f32(float %1021), !dbg !27
  %1023 = insertelement <4 x float> %1019, float %1022, i64 3, !dbg !28
  %math.result335 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1023) #8, !dbg !28, !frisk.register_lifetimes !5
  %1024 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !28
  %math.packed336 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1024) #8, !dbg !29, !frisk.register_lifetimes !5
  %1025 = fadd float %1022, %1020, !dbg !29
  %1026 = extractelement <4 x float> %756, i64 0, !dbg !27
  %1027 = tail call float @llvm.amdgcn.exp2.f32(float %1026), !dbg !27
  %1028 = insertelement <4 x float> poison, float %1027, i64 0, !dbg !28
  %1029 = fadd float %1027, %1025, !dbg !29
  %1030 = extractelement <4 x float> %756, i64 1, !dbg !27
  %1031 = tail call float @llvm.amdgcn.exp2.f32(float %1030), !dbg !27
  %1032 = insertelement <4 x float> %1028, float %1031, i64 1, !dbg !28
  %1033 = fadd float %1031, %1029, !dbg !29
  %1034 = extractelement <4 x float> %756, i64 2, !dbg !27
  %1035 = tail call float @llvm.amdgcn.exp2.f32(float %1034), !dbg !27
  %1036 = insertelement <4 x float> %1032, float %1035, i64 2, !dbg !28
  %1037 = fadd float %1035, %1033, !dbg !29
  %1038 = extractelement <4 x float> %756, i64 3, !dbg !27
  %1039 = tail call float @llvm.amdgcn.exp2.f32(float %1038), !dbg !27
  %1040 = insertelement <4 x float> %1036, float %1039, i64 3, !dbg !28
  %math.result337 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1040) #8, !dbg !28, !frisk.register_lifetimes !5
  %1041 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !28
  %math.packed338 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1041) #8, !dbg !29, !frisk.register_lifetimes !5
  %1042 = fadd float %1039, %1037, !dbg !29
  %1043 = bitcast float %907 to i32, !dbg !29
  %1044 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1043), !dbg !29
  %1045 = bitcast i32 %1044 to float, !dbg !29
  %1046 = fadd float %907, %1045, !dbg !29
  %1047 = bitcast float %1046 to i32, !dbg !29
  %1048 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1047), !dbg !29
  %1049 = bitcast i32 %1048 to float, !dbg !29
  %1050 = fadd float %1046, %1049, !dbg !29
  %1051 = fadd float %764, %1050, !dbg !29
  %1052 = insertelement <1 x float> poison, float %1051, i64 0, !dbg !11
  %1053 = insertvalue [2 x <1 x float>] poison, <1 x float> %1052, 0, !dbg !11
  %1054 = bitcast float %1042 to i32, !dbg !29
  %1055 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1054), !dbg !29
  %1056 = bitcast i32 %1055 to float, !dbg !29
  %1057 = fadd float %1042, %1056, !dbg !29
  %1058 = bitcast float %1057 to i32, !dbg !29
  %1059 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1058), !dbg !29
  %1060 = bitcast i32 %1059 to float, !dbg !29
  %1061 = fadd float %1057, %1060, !dbg !29
  %1062 = fadd float %772, %1061, !dbg !29
  %1063 = insertelement <1 x float> poison, float %1062, i64 0, !dbg !11
  %1064 = insertvalue [2 x <1 x float>] %1053, <1 x float> %1063, 1, !dbg !11
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
  %1065 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !13
  %1066 = bitcast <4 x i32> %1065 to <8 x half>, !dbg !13
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
  %1067 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !13
  %1068 = bitcast <4 x i32> %1067 to <8 x half>, !dbg !13
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local883, i64 512, !dbg !13
  %1069 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !13
  %1070 = bitcast <4 x i32> %1069 to <8 x half>, !dbg !13
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local883, i64 768, !dbg !13
  %1071 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !13
  %1072 = bitcast <4 x i32> %1071 to <8 x half>, !dbg !13
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local871, i64 4096, !dbg !13
  %1073 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !13
  %1074 = bitcast <4 x i32> %1073 to <8 x half>, !dbg !13
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4352, !dbg !13
  %1075 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !13
  %1076 = bitcast <4 x i32> %1075 to <8 x half>, !dbg !13
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4608, !dbg !13
  %1077 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !13
  %1078 = bitcast <4 x i32> %1077 to <8 x half>, !dbg !13
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local883, i64 4864, !dbg !13
  %1079 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !13
  %1080 = bitcast <4 x i32> %1079 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !8
  %address.local908 = and i64 %address.local907, 1536, !dbg !8
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !8
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !8
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !8
  %1081 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !18
  %address.local913 = and i64 %address.local912, 8064, !dbg !18
  %address.local915 = and i64 %address.local907, 24, !dbg !18
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !18
  %1082 = or disjoint i64 %.idx5760, %address.local916, !dbg !18
  %1083 = trunc i64 %1082 to i32, !dbg !18
  %.tr68 = shl i32 %1083, 1, !dbg !18
  %dma.buffer.offset1558 = or i32 %.tr68, 64, !dbg !18
  %dma.buffer.lds1560 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1081), !dbg !18
  %1084 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !8
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !8
  %address.local920 = and i64 %address.local919, 3584, !dbg !8
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !8
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !8
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !8
  %1085 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !18
  %address.local925 = and i64 %address.local924, 16256, !dbg !18
  %address.local928 = or disjoint i64 %address.local925, %address.local915, !dbg !18
  %1086 = or disjoint i64 %.idx5760, %address.local928, !dbg !18
  %1087 = trunc i64 %1086 to i32, !dbg !18
  %.tr72 = shl i32 %1087, 1, !dbg !18
  %dma.buffer.offset1555 = or i32 %.tr72, 64, !dbg !18
  %dma.buffer.lds1557 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1085), !dbg !18
  %1088 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %1089 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1089, <4 x float> zeroinitializer) #8, !dbg !13
  %1091 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1091, <4 x float> %1090) #8, !dbg !13
  %1093 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1093, <4 x float> %1092) #8, !dbg !13
  %1095 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1095, <4 x float> %1094) #8, !dbg !13
  %1097 = shufflevector <8 x half> %1074, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1097, <4 x float> %1096) #8, !dbg !13
  %1099 = shufflevector <8 x half> %1076, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1099, <4 x float> %1098) #8, !dbg !13
  %1101 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1101, <4 x float> %1100) #8, !dbg !13
  %1103 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1103, <4 x float> %1102) #8, !dbg !13
  %1105 = extractelement <4 x float> %1104, i64 0, !dbg !13
  %1106 = extractelement <4 x float> %1104, i64 1, !dbg !13
  %1107 = extractelement <4 x float> %1104, i64 2, !dbg !13
  %1108 = extractelement <4 x float> %1104, i64 3, !dbg !13
  %1109 = select i1 %.not, float %1106, float %1105, !dbg !13
  %1110 = bitcast float %1109 to i32, !dbg !13
  %1111 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1110), !dbg !13
  %1112 = bitcast i32 %1111 to float, !dbg !13
  %1113 = select i1 %.not, float %1105, float %1112, !dbg !13
  %1114 = select i1 %.not, float %1112, float %1106, !dbg !13
  %1115 = select i1 %.not, float %1108, float %1107, !dbg !13
  %1116 = bitcast float %1115 to i32, !dbg !13
  %1117 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1116), !dbg !13
  %1118 = bitcast i32 %1117 to float, !dbg !13
  %1119 = select i1 %.not, float %1107, float %1118, !dbg !13
  %1120 = select i1 %.not, float %1118, float %1108, !dbg !13
  %1121 = select i1 %.not1, float %1119, float %1113, !dbg !13
  %1122 = bitcast float %1121 to i32, !dbg !13
  %1123 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1122), !dbg !13
  %1124 = bitcast i32 %1123 to float, !dbg !13
  %1125 = select i1 %.not1, float %1120, float %1114, !dbg !13
  %1126 = bitcast float %1125 to i32, !dbg !13
  %1127 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1126), !dbg !13
  %1128 = bitcast i32 %1127 to float, !dbg !13
  %1129 = insertelement <4 x float> poison, float %1113, i64 0, !dbg !13
  %1130 = insertelement <4 x float> %1129, float %1114, i64 1, !dbg !13
  %1131 = insertelement <4 x float> %1130, float %1124, i64 2, !dbg !13
  %1132 = insertelement <4 x float> %1131, float %1128, i64 3, !dbg !13
  %1133 = insertelement <4 x float> poison, float %1124, i64 0, !dbg !13
  %1134 = insertelement <4 x float> %1133, float %1128, i64 1, !dbg !13
  %1135 = insertelement <4 x float> %1134, float %1119, i64 2, !dbg !13
  %1136 = insertelement <4 x float> %1135, float %1120, i64 3, !dbg !13
  %1137 = select <4 x i1> %136, <4 x float> %1132, <4 x float> %1136, !dbg !13
  %1138 = extractvalue [2 x <8 x float>] %147, 0, !dbg !25
  %1139 = shufflevector <8 x float> %1138, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1140 = fmul <4 x float> %1139, %761, !dbg !25
  %1141 = fadd <4 x float> %1140, %1137, !dbg !30
  %1142 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1142, <4 x float> zeroinitializer) #8, !dbg !13
  %1144 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1144, <4 x float> %1143) #8, !dbg !13
  %1146 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1146, <4 x float> %1145) #8, !dbg !13
  %1148 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1148, <4 x float> %1147) #8, !dbg !13
  %1150 = shufflevector <8 x half> %1074, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1150, <4 x float> %1149) #8, !dbg !13
  %1152 = shufflevector <8 x half> %1076, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1152, <4 x float> %1151) #8, !dbg !13
  %1154 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1154, <4 x float> %1153) #8, !dbg !13
  %1156 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1156, <4 x float> %1155) #8, !dbg !13
  %1158 = extractelement <4 x float> %1157, i64 0, !dbg !13
  %1159 = extractelement <4 x float> %1157, i64 1, !dbg !13
  %1160 = extractelement <4 x float> %1157, i64 2, !dbg !13
  %1161 = extractelement <4 x float> %1157, i64 3, !dbg !13
  %1162 = select i1 %.not, float %1159, float %1158, !dbg !13
  %1163 = bitcast float %1162 to i32, !dbg !13
  %1164 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1163), !dbg !13
  %1165 = bitcast i32 %1164 to float, !dbg !13
  %1166 = select i1 %.not, float %1158, float %1165, !dbg !13
  %1167 = select i1 %.not, float %1165, float %1159, !dbg !13
  %1168 = select i1 %.not, float %1161, float %1160, !dbg !13
  %1169 = bitcast float %1168 to i32, !dbg !13
  %1170 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1169), !dbg !13
  %1171 = bitcast i32 %1170 to float, !dbg !13
  %1172 = select i1 %.not, float %1160, float %1171, !dbg !13
  %1173 = select i1 %.not, float %1171, float %1161, !dbg !13
  %1174 = select i1 %.not1, float %1172, float %1166, !dbg !13
  %1175 = bitcast float %1174 to i32, !dbg !13
  %1176 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1175), !dbg !13
  %1177 = bitcast i32 %1176 to float, !dbg !13
  %1178 = select i1 %.not1, float %1173, float %1167, !dbg !13
  %1179 = bitcast float %1178 to i32, !dbg !13
  %1180 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1179), !dbg !13
  %1181 = bitcast i32 %1180 to float, !dbg !13
  %1182 = insertelement <4 x float> poison, float %1166, i64 0, !dbg !13
  %1183 = insertelement <4 x float> %1182, float %1167, i64 1, !dbg !13
  %1184 = insertelement <4 x float> %1183, float %1177, i64 2, !dbg !13
  %1185 = insertelement <4 x float> %1184, float %1181, i64 3, !dbg !13
  %1186 = insertelement <4 x float> poison, float %1177, i64 0, !dbg !13
  %1187 = insertelement <4 x float> %1186, float %1181, i64 1, !dbg !13
  %1188 = insertelement <4 x float> %1187, float %1172, i64 2, !dbg !13
  %1189 = insertelement <4 x float> %1188, float %1173, i64 3, !dbg !13
  %1190 = select <4 x i1> %136, <4 x float> %1185, <4 x float> %1189, !dbg !13
  %1191 = shufflevector <8 x float> %1138, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1192 = fmul <4 x float> %1191, %761, !dbg !25
  %1193 = fadd <4 x float> %1192, %1190, !dbg !30
  %1194 = shufflevector <4 x float> %1141, <4 x float> %1193, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1195 = insertvalue [2 x <8 x float>] poison, <8 x float> %1194, 0, !dbg !11
  %1196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1089, <4 x float> zeroinitializer) #8, !dbg !13
  %1197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1091, <4 x float> %1196) #8, !dbg !13
  %1198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1093, <4 x float> %1197) #8, !dbg !13
  %1199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1095, <4 x float> %1198) #8, !dbg !13
  %1200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1097, <4 x float> %1199) #8, !dbg !13
  %1201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1099, <4 x float> %1200) #8, !dbg !13
  %1202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1101, <4 x float> %1201) #8, !dbg !13
  %1203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1103, <4 x float> %1202) #8, !dbg !13
  %1204 = extractelement <4 x float> %1203, i64 0, !dbg !13
  %1205 = extractelement <4 x float> %1203, i64 1, !dbg !13
  %1206 = extractelement <4 x float> %1203, i64 2, !dbg !13
  %1207 = extractelement <4 x float> %1203, i64 3, !dbg !13
  %1208 = select i1 %.not, float %1205, float %1204, !dbg !13
  %1209 = bitcast float %1208 to i32, !dbg !13
  %1210 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1209), !dbg !13
  %1211 = bitcast i32 %1210 to float, !dbg !13
  %1212 = select i1 %.not, float %1204, float %1211, !dbg !13
  %1213 = select i1 %.not, float %1211, float %1205, !dbg !13
  %1214 = select i1 %.not, float %1207, float %1206, !dbg !13
  %1215 = bitcast float %1214 to i32, !dbg !13
  %1216 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1215), !dbg !13
  %1217 = bitcast i32 %1216 to float, !dbg !13
  %1218 = select i1 %.not, float %1206, float %1217, !dbg !13
  %1219 = select i1 %.not, float %1217, float %1207, !dbg !13
  %1220 = select i1 %.not1, float %1218, float %1212, !dbg !13
  %1221 = bitcast float %1220 to i32, !dbg !13
  %1222 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1221), !dbg !13
  %1223 = bitcast i32 %1222 to float, !dbg !13
  %1224 = select i1 %.not1, float %1219, float %1213, !dbg !13
  %1225 = bitcast float %1224 to i32, !dbg !13
  %1226 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1225), !dbg !13
  %1227 = bitcast i32 %1226 to float, !dbg !13
  %1228 = insertelement <4 x float> poison, float %1212, i64 0, !dbg !13
  %1229 = insertelement <4 x float> %1228, float %1213, i64 1, !dbg !13
  %1230 = insertelement <4 x float> %1229, float %1223, i64 2, !dbg !13
  %1231 = insertelement <4 x float> %1230, float %1227, i64 3, !dbg !13
  %1232 = insertelement <4 x float> poison, float %1223, i64 0, !dbg !13
  %1233 = insertelement <4 x float> %1232, float %1227, i64 1, !dbg !13
  %1234 = insertelement <4 x float> %1233, float %1218, i64 2, !dbg !13
  %1235 = insertelement <4 x float> %1234, float %1219, i64 3, !dbg !13
  %1236 = select <4 x i1> %136, <4 x float> %1231, <4 x float> %1235, !dbg !13
  %1237 = extractvalue [2 x <8 x float>] %147, 1, !dbg !25
  %1238 = shufflevector <8 x float> %1237, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1239 = fmul <4 x float> %1238, %769, !dbg !25
  %1240 = fadd <4 x float> %1239, %1236, !dbg !30
  %1241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1142, <4 x float> zeroinitializer) #8, !dbg !13
  %1242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1144, <4 x float> %1241) #8, !dbg !13
  %1243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1146, <4 x float> %1242) #8, !dbg !13
  %1244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1148, <4 x float> %1243) #8, !dbg !13
  %1245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1150, <4 x float> %1244) #8, !dbg !13
  %1246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1152, <4 x float> %1245) #8, !dbg !13
  %1247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1154, <4 x float> %1246) #8, !dbg !13
  %1248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1156, <4 x float> %1247) #8, !dbg !13
  %1249 = extractelement <4 x float> %1248, i64 0, !dbg !13
  %1250 = extractelement <4 x float> %1248, i64 1, !dbg !13
  %1251 = extractelement <4 x float> %1248, i64 2, !dbg !13
  %1252 = extractelement <4 x float> %1248, i64 3, !dbg !13
  %1253 = select i1 %.not, float %1250, float %1249, !dbg !13
  %1254 = bitcast float %1253 to i32, !dbg !13
  %1255 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1254), !dbg !13
  %1256 = bitcast i32 %1255 to float, !dbg !13
  %1257 = select i1 %.not, float %1249, float %1256, !dbg !13
  %1258 = select i1 %.not, float %1256, float %1250, !dbg !13
  %1259 = select i1 %.not, float %1252, float %1251, !dbg !13
  %1260 = bitcast float %1259 to i32, !dbg !13
  %1261 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1260), !dbg !13
  %1262 = bitcast i32 %1261 to float, !dbg !13
  %1263 = select i1 %.not, float %1251, float %1262, !dbg !13
  %1264 = select i1 %.not, float %1262, float %1252, !dbg !13
  %1265 = select i1 %.not1, float %1263, float %1257, !dbg !13
  %1266 = bitcast float %1265 to i32, !dbg !13
  %1267 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1266), !dbg !13
  %1268 = bitcast i32 %1267 to float, !dbg !13
  %1269 = select i1 %.not1, float %1264, float %1258, !dbg !13
  %1270 = bitcast float %1269 to i32, !dbg !13
  %1271 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1270), !dbg !13
  %1272 = bitcast i32 %1271 to float, !dbg !13
  %1273 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !13
  %1274 = insertelement <4 x float> %1273, float %1258, i64 1, !dbg !13
  %1275 = insertelement <4 x float> %1274, float %1268, i64 2, !dbg !13
  %1276 = insertelement <4 x float> %1275, float %1272, i64 3, !dbg !13
  %1277 = insertelement <4 x float> poison, float %1268, i64 0, !dbg !13
  %1278 = insertelement <4 x float> %1277, float %1272, i64 1, !dbg !13
  %1279 = insertelement <4 x float> %1278, float %1263, i64 2, !dbg !13
  %1280 = insertelement <4 x float> %1279, float %1264, i64 3, !dbg !13
  %1281 = select <4 x i1> %136, <4 x float> %1276, <4 x float> %1280, !dbg !13
  %1282 = shufflevector <8 x float> %1237, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1283 = fmul <4 x float> %1282, %769, !dbg !25
  %1284 = fadd <4 x float> %1283, %1281, !dbg !30
  %1285 = shufflevector <4 x float> %1240, <4 x float> %1284, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1286 = insertvalue [2 x <8 x float>] %1195, <8 x float> %1285, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1287 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !13
  %1288 = bitcast <4 x i32> %1287 to <8 x half>, !dbg !13
  %1289 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !13
  %1290 = bitcast <4 x i32> %1289 to <8 x half>, !dbg !13
  %1291 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !13
  %1292 = bitcast <4 x i32> %1291 to <8 x half>, !dbg !13
  %1293 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !13
  %1294 = bitcast <4 x i32> %1293 to <8 x half>, !dbg !13
  %1295 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !13
  %1296 = bitcast <4 x i32> %1295 to <8 x half>, !dbg !13
  %1297 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !13
  %1298 = bitcast <4 x i32> %1297 to <8 x half>, !dbg !13
  %1299 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !13
  %1300 = bitcast <4 x i32> %1299 to <8 x half>, !dbg !13
  %1301 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !13
  %1302 = bitcast <4 x i32> %1301 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %dma.buffer.offset1552 = or i32 %.tr68, 128, !dbg !18
  %1303 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %dma.buffer.offset1549 = or i32 %.tr72, 128, !dbg !18
  %1304 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %1305 = shufflevector <8 x half> %1288, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1305, <4 x float> zeroinitializer) #8, !dbg !13
  %1307 = shufflevector <8 x half> %1290, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1307, <4 x float> %1306) #8, !dbg !13
  %1309 = shufflevector <8 x half> %1292, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1309, <4 x float> %1308) #8, !dbg !13
  %1311 = shufflevector <8 x half> %1294, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1311, <4 x float> %1310) #8, !dbg !13
  %1313 = shufflevector <8 x half> %1296, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1313, <4 x float> %1312) #8, !dbg !13
  %1315 = shufflevector <8 x half> %1298, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1315, <4 x float> %1314) #8, !dbg !13
  %1317 = shufflevector <8 x half> %1300, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1317, <4 x float> %1316) #8, !dbg !13
  %1319 = shufflevector <8 x half> %1302, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1319, <4 x float> %1318) #8, !dbg !13
  %1321 = extractelement <4 x float> %1320, i64 0, !dbg !13
  %1322 = extractelement <4 x float> %1320, i64 1, !dbg !13
  %1323 = extractelement <4 x float> %1320, i64 2, !dbg !13
  %1324 = extractelement <4 x float> %1320, i64 3, !dbg !13
  %1325 = select i1 %.not, float %1322, float %1321, !dbg !13
  %1326 = bitcast float %1325 to i32, !dbg !13
  %1327 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1326), !dbg !13
  %1328 = bitcast i32 %1327 to float, !dbg !13
  %1329 = select i1 %.not, float %1321, float %1328, !dbg !13
  %1330 = select i1 %.not, float %1328, float %1322, !dbg !13
  %1331 = select i1 %.not, float %1324, float %1323, !dbg !13
  %1332 = bitcast float %1331 to i32, !dbg !13
  %1333 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1332), !dbg !13
  %1334 = bitcast i32 %1333 to float, !dbg !13
  %1335 = select i1 %.not, float %1323, float %1334, !dbg !13
  %1336 = select i1 %.not, float %1334, float %1324, !dbg !13
  %1337 = select i1 %.not1, float %1335, float %1329, !dbg !13
  %1338 = bitcast float %1337 to i32, !dbg !13
  %1339 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1338), !dbg !13
  %1340 = bitcast i32 %1339 to float, !dbg !13
  %1341 = select i1 %.not1, float %1336, float %1330, !dbg !13
  %1342 = bitcast float %1341 to i32, !dbg !13
  %1343 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1342), !dbg !13
  %1344 = bitcast i32 %1343 to float, !dbg !13
  %1345 = insertelement <4 x float> poison, float %1329, i64 0, !dbg !13
  %1346 = insertelement <4 x float> %1345, float %1330, i64 1, !dbg !13
  %1347 = insertelement <4 x float> %1346, float %1340, i64 2, !dbg !13
  %1348 = insertelement <4 x float> %1347, float %1344, i64 3, !dbg !13
  %1349 = insertelement <4 x float> poison, float %1340, i64 0, !dbg !13
  %1350 = insertelement <4 x float> %1349, float %1344, i64 1, !dbg !13
  %1351 = insertelement <4 x float> %1350, float %1335, i64 2, !dbg !13
  %1352 = insertelement <4 x float> %1351, float %1336, i64 3, !dbg !13
  %1353 = select <4 x i1> %136, <4 x float> %1348, <4 x float> %1352, !dbg !13
  %1354 = extractvalue [2 x <8 x float>] %144, 0, !dbg !25
  %1355 = shufflevector <8 x float> %1354, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1356 = fmul <4 x float> %1355, %761, !dbg !25
  %1357 = fadd <4 x float> %1356, %1353, !dbg !30
  %1358 = shufflevector <8 x half> %1288, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1358, <4 x float> zeroinitializer) #8, !dbg !13
  %1360 = shufflevector <8 x half> %1290, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1360, <4 x float> %1359) #8, !dbg !13
  %1362 = shufflevector <8 x half> %1292, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1362, <4 x float> %1361) #8, !dbg !13
  %1364 = shufflevector <8 x half> %1294, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1364, <4 x float> %1363) #8, !dbg !13
  %1366 = shufflevector <8 x half> %1296, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1366, <4 x float> %1365) #8, !dbg !13
  %1368 = shufflevector <8 x half> %1298, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1368, <4 x float> %1367) #8, !dbg !13
  %1370 = shufflevector <8 x half> %1300, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1371 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1370, <4 x float> %1369) #8, !dbg !13
  %1372 = shufflevector <8 x half> %1302, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1372, <4 x float> %1371) #8, !dbg !13
  %1374 = extractelement <4 x float> %1373, i64 0, !dbg !13
  %1375 = extractelement <4 x float> %1373, i64 1, !dbg !13
  %1376 = extractelement <4 x float> %1373, i64 2, !dbg !13
  %1377 = extractelement <4 x float> %1373, i64 3, !dbg !13
  %1378 = select i1 %.not, float %1375, float %1374, !dbg !13
  %1379 = bitcast float %1378 to i32, !dbg !13
  %1380 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1379), !dbg !13
  %1381 = bitcast i32 %1380 to float, !dbg !13
  %1382 = select i1 %.not, float %1374, float %1381, !dbg !13
  %1383 = select i1 %.not, float %1381, float %1375, !dbg !13
  %1384 = select i1 %.not, float %1377, float %1376, !dbg !13
  %1385 = bitcast float %1384 to i32, !dbg !13
  %1386 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1385), !dbg !13
  %1387 = bitcast i32 %1386 to float, !dbg !13
  %1388 = select i1 %.not, float %1376, float %1387, !dbg !13
  %1389 = select i1 %.not, float %1387, float %1377, !dbg !13
  %1390 = select i1 %.not1, float %1388, float %1382, !dbg !13
  %1391 = bitcast float %1390 to i32, !dbg !13
  %1392 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1391), !dbg !13
  %1393 = bitcast i32 %1392 to float, !dbg !13
  %1394 = select i1 %.not1, float %1389, float %1383, !dbg !13
  %1395 = bitcast float %1394 to i32, !dbg !13
  %1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1395), !dbg !13
  %1397 = bitcast i32 %1396 to float, !dbg !13
  %1398 = insertelement <4 x float> poison, float %1382, i64 0, !dbg !13
  %1399 = insertelement <4 x float> %1398, float %1383, i64 1, !dbg !13
  %1400 = insertelement <4 x float> %1399, float %1393, i64 2, !dbg !13
  %1401 = insertelement <4 x float> %1400, float %1397, i64 3, !dbg !13
  %1402 = insertelement <4 x float> poison, float %1393, i64 0, !dbg !13
  %1403 = insertelement <4 x float> %1402, float %1397, i64 1, !dbg !13
  %1404 = insertelement <4 x float> %1403, float %1388, i64 2, !dbg !13
  %1405 = insertelement <4 x float> %1404, float %1389, i64 3, !dbg !13
  %1406 = select <4 x i1> %136, <4 x float> %1401, <4 x float> %1405, !dbg !13
  %1407 = shufflevector <8 x float> %1354, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1408 = fmul <4 x float> %1407, %761, !dbg !25
  %1409 = fadd <4 x float> %1408, %1406, !dbg !30
  %1410 = shufflevector <4 x float> %1357, <4 x float> %1409, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1411 = insertvalue [2 x <8 x float>] poison, <8 x float> %1410, 0, !dbg !11
  %1412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1305, <4 x float> zeroinitializer) #8, !dbg !13
  %1413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1307, <4 x float> %1412) #8, !dbg !13
  %1414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1309, <4 x float> %1413) #8, !dbg !13
  %1415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1311, <4 x float> %1414) #8, !dbg !13
  %1416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1313, <4 x float> %1415) #8, !dbg !13
  %1417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1315, <4 x float> %1416) #8, !dbg !13
  %1418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1317, <4 x float> %1417) #8, !dbg !13
  %1419 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1319, <4 x float> %1418) #8, !dbg !13
  %1420 = extractelement <4 x float> %1419, i64 0, !dbg !13
  %1421 = extractelement <4 x float> %1419, i64 1, !dbg !13
  %1422 = extractelement <4 x float> %1419, i64 2, !dbg !13
  %1423 = extractelement <4 x float> %1419, i64 3, !dbg !13
  %1424 = select i1 %.not, float %1421, float %1420, !dbg !13
  %1425 = bitcast float %1424 to i32, !dbg !13
  %1426 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1425), !dbg !13
  %1427 = bitcast i32 %1426 to float, !dbg !13
  %1428 = select i1 %.not, float %1420, float %1427, !dbg !13
  %1429 = select i1 %.not, float %1427, float %1421, !dbg !13
  %1430 = select i1 %.not, float %1423, float %1422, !dbg !13
  %1431 = bitcast float %1430 to i32, !dbg !13
  %1432 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1431), !dbg !13
  %1433 = bitcast i32 %1432 to float, !dbg !13
  %1434 = select i1 %.not, float %1422, float %1433, !dbg !13
  %1435 = select i1 %.not, float %1433, float %1423, !dbg !13
  %1436 = select i1 %.not1, float %1434, float %1428, !dbg !13
  %1437 = bitcast float %1436 to i32, !dbg !13
  %1438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1437), !dbg !13
  %1439 = bitcast i32 %1438 to float, !dbg !13
  %1440 = select i1 %.not1, float %1435, float %1429, !dbg !13
  %1441 = bitcast float %1440 to i32, !dbg !13
  %1442 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1441), !dbg !13
  %1443 = bitcast i32 %1442 to float, !dbg !13
  %1444 = insertelement <4 x float> poison, float %1428, i64 0, !dbg !13
  %1445 = insertelement <4 x float> %1444, float %1429, i64 1, !dbg !13
  %1446 = insertelement <4 x float> %1445, float %1439, i64 2, !dbg !13
  %1447 = insertelement <4 x float> %1446, float %1443, i64 3, !dbg !13
  %1448 = insertelement <4 x float> poison, float %1439, i64 0, !dbg !13
  %1449 = insertelement <4 x float> %1448, float %1443, i64 1, !dbg !13
  %1450 = insertelement <4 x float> %1449, float %1434, i64 2, !dbg !13
  %1451 = insertelement <4 x float> %1450, float %1435, i64 3, !dbg !13
  %1452 = select <4 x i1> %136, <4 x float> %1447, <4 x float> %1451, !dbg !13
  %1453 = extractvalue [2 x <8 x float>] %144, 1, !dbg !25
  %1454 = shufflevector <8 x float> %1453, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1455 = fmul <4 x float> %1454, %769, !dbg !25
  %1456 = fadd <4 x float> %1455, %1452, !dbg !30
  %1457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1358, <4 x float> zeroinitializer) #8, !dbg !13
  %1458 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1360, <4 x float> %1457) #8, !dbg !13
  %1459 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1362, <4 x float> %1458) #8, !dbg !13
  %1460 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1364, <4 x float> %1459) #8, !dbg !13
  %1461 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1366, <4 x float> %1460) #8, !dbg !13
  %1462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1368, <4 x float> %1461) #8, !dbg !13
  %1463 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1370, <4 x float> %1462) #8, !dbg !13
  %1464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1372, <4 x float> %1463) #8, !dbg !13
  %1465 = extractelement <4 x float> %1464, i64 0, !dbg !13
  %1466 = extractelement <4 x float> %1464, i64 1, !dbg !13
  %1467 = extractelement <4 x float> %1464, i64 2, !dbg !13
  %1468 = extractelement <4 x float> %1464, i64 3, !dbg !13
  %1469 = select i1 %.not, float %1466, float %1465, !dbg !13
  %1470 = bitcast float %1469 to i32, !dbg !13
  %1471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1470), !dbg !13
  %1472 = bitcast i32 %1471 to float, !dbg !13
  %1473 = select i1 %.not, float %1465, float %1472, !dbg !13
  %1474 = select i1 %.not, float %1472, float %1466, !dbg !13
  %1475 = select i1 %.not, float %1468, float %1467, !dbg !13
  %1476 = bitcast float %1475 to i32, !dbg !13
  %1477 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1476), !dbg !13
  %1478 = bitcast i32 %1477 to float, !dbg !13
  %1479 = select i1 %.not, float %1467, float %1478, !dbg !13
  %1480 = select i1 %.not, float %1478, float %1468, !dbg !13
  %1481 = select i1 %.not1, float %1479, float %1473, !dbg !13
  %1482 = bitcast float %1481 to i32, !dbg !13
  %1483 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1482), !dbg !13
  %1484 = bitcast i32 %1483 to float, !dbg !13
  %1485 = select i1 %.not1, float %1480, float %1474, !dbg !13
  %1486 = bitcast float %1485 to i32, !dbg !13
  %1487 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1486), !dbg !13
  %1488 = bitcast i32 %1487 to float, !dbg !13
  %1489 = insertelement <4 x float> poison, float %1473, i64 0, !dbg !13
  %1490 = insertelement <4 x float> %1489, float %1474, i64 1, !dbg !13
  %1491 = insertelement <4 x float> %1490, float %1484, i64 2, !dbg !13
  %1492 = insertelement <4 x float> %1491, float %1488, i64 3, !dbg !13
  %1493 = insertelement <4 x float> poison, float %1484, i64 0, !dbg !13
  %1494 = insertelement <4 x float> %1493, float %1488, i64 1, !dbg !13
  %1495 = insertelement <4 x float> %1494, float %1479, i64 2, !dbg !13
  %1496 = insertelement <4 x float> %1495, float %1480, i64 3, !dbg !13
  %1497 = select <4 x i1> %136, <4 x float> %1492, <4 x float> %1496, !dbg !13
  %1498 = shufflevector <8 x float> %1453, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1499 = fmul <4 x float> %1498, %769, !dbg !25
  %1500 = fadd <4 x float> %1499, %1497, !dbg !30
  %1501 = shufflevector <4 x float> %1456, <4 x float> %1500, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1502 = insertvalue [2 x <8 x float>] %1411, <8 x float> %1501, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1503 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !13
  %1504 = bitcast <4 x i32> %1503 to <8 x half>, !dbg !13
  %1505 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !13
  %1506 = bitcast <4 x i32> %1505 to <8 x half>, !dbg !13
  %1507 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !13
  %1508 = bitcast <4 x i32> %1507 to <8 x half>, !dbg !13
  %1509 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !13
  %1510 = bitcast <4 x i32> %1509 to <8 x half>, !dbg !13
  %1511 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !13
  %1512 = bitcast <4 x i32> %1511 to <8 x half>, !dbg !13
  %1513 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !13
  %1514 = bitcast <4 x i32> %1513 to <8 x half>, !dbg !13
  %1515 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !13
  %1516 = bitcast <4 x i32> %1515 to <8 x half>, !dbg !13
  %1517 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !13
  %1518 = bitcast <4 x i32> %1517 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1519 = or disjoint i64 %.idx5760, 96, !dbg !18
  %1520 = or disjoint i64 %address.local916, %1519, !dbg !18
  %.tr84 = trunc i64 %1520 to i32, !dbg !18
  %dma.buffer.offset1546 = shl i32 %.tr84, 1, !dbg !18
  %1521 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %1522 = or disjoint i64 %address.local928, %1519, !dbg !18
  %.tr88 = trunc i64 %1522 to i32, !dbg !18
  %dma.buffer.offset = shl i32 %.tr88, 1, !dbg !18
  %1523 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1565) #9, !dbg !18
  %1524 = shufflevector <8 x half> %1504, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1525 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1524, <4 x float> zeroinitializer) #8, !dbg !13
  %1526 = shufflevector <8 x half> %1506, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1527 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1526, <4 x float> %1525) #8, !dbg !13
  %1528 = shufflevector <8 x half> %1508, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1529 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1528, <4 x float> %1527) #8, !dbg !13
  %1530 = shufflevector <8 x half> %1510, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1530, <4 x float> %1529) #8, !dbg !13
  %1532 = shufflevector <8 x half> %1512, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1533 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1532, <4 x float> %1531) #8, !dbg !13
  %1534 = shufflevector <8 x half> %1514, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1535 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1534, <4 x float> %1533) #8, !dbg !13
  %1536 = shufflevector <8 x half> %1516, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1537 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1536, <4 x float> %1535) #8, !dbg !13
  %1538 = shufflevector <8 x half> %1518, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1539 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1538, <4 x float> %1537) #8, !dbg !13
  %1540 = extractelement <4 x float> %1539, i64 0, !dbg !13
  %1541 = extractelement <4 x float> %1539, i64 1, !dbg !13
  %1542 = extractelement <4 x float> %1539, i64 2, !dbg !13
  %1543 = extractelement <4 x float> %1539, i64 3, !dbg !13
  %1544 = select i1 %.not, float %1541, float %1540, !dbg !13
  %1545 = bitcast float %1544 to i32, !dbg !13
  %1546 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1545), !dbg !13
  %1547 = bitcast i32 %1546 to float, !dbg !13
  %1548 = select i1 %.not, float %1540, float %1547, !dbg !13
  %1549 = select i1 %.not, float %1547, float %1541, !dbg !13
  %1550 = select i1 %.not, float %1543, float %1542, !dbg !13
  %1551 = bitcast float %1550 to i32, !dbg !13
  %1552 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1551), !dbg !13
  %1553 = bitcast i32 %1552 to float, !dbg !13
  %1554 = select i1 %.not, float %1542, float %1553, !dbg !13
  %1555 = select i1 %.not, float %1553, float %1543, !dbg !13
  %1556 = select i1 %.not1, float %1554, float %1548, !dbg !13
  %1557 = bitcast float %1556 to i32, !dbg !13
  %1558 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1557), !dbg !13
  %1559 = bitcast i32 %1558 to float, !dbg !13
  %1560 = select i1 %.not1, float %1555, float %1549, !dbg !13
  %1561 = bitcast float %1560 to i32, !dbg !13
  %1562 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1561), !dbg !13
  %1563 = bitcast i32 %1562 to float, !dbg !13
  %1564 = insertelement <4 x float> poison, float %1548, i64 0, !dbg !13
  %1565 = insertelement <4 x float> %1564, float %1549, i64 1, !dbg !13
  %1566 = insertelement <4 x float> %1565, float %1559, i64 2, !dbg !13
  %1567 = insertelement <4 x float> %1566, float %1563, i64 3, !dbg !13
  %1568 = insertelement <4 x float> poison, float %1559, i64 0, !dbg !13
  %1569 = insertelement <4 x float> %1568, float %1563, i64 1, !dbg !13
  %1570 = insertelement <4 x float> %1569, float %1554, i64 2, !dbg !13
  %1571 = insertelement <4 x float> %1570, float %1555, i64 3, !dbg !13
  %1572 = select <4 x i1> %136, <4 x float> %1567, <4 x float> %1571, !dbg !13
  %1573 = extractvalue [2 x <8 x float>] %143, 0, !dbg !25
  %1574 = shufflevector <8 x float> %1573, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1575 = fmul <4 x float> %1574, %761, !dbg !25
  %1576 = fadd <4 x float> %1575, %1572, !dbg !30
  %1577 = shufflevector <8 x half> %1504, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1578 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1577, <4 x float> zeroinitializer) #8, !dbg !13
  %1579 = shufflevector <8 x half> %1506, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1580 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1579, <4 x float> %1578) #8, !dbg !13
  %1581 = shufflevector <8 x half> %1508, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1582 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1581, <4 x float> %1580) #8, !dbg !13
  %1583 = shufflevector <8 x half> %1510, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1584 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1583, <4 x float> %1582) #8, !dbg !13
  %1585 = shufflevector <8 x half> %1512, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1586 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1585, <4 x float> %1584) #8, !dbg !13
  %1587 = shufflevector <8 x half> %1514, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1587, <4 x float> %1586) #8, !dbg !13
  %1589 = shufflevector <8 x half> %1516, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1589, <4 x float> %1588) #8, !dbg !13
  %1591 = shufflevector <8 x half> %1518, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1591, <4 x float> %1590) #8, !dbg !13
  %1593 = extractelement <4 x float> %1592, i64 0, !dbg !13
  %1594 = extractelement <4 x float> %1592, i64 1, !dbg !13
  %1595 = extractelement <4 x float> %1592, i64 2, !dbg !13
  %1596 = extractelement <4 x float> %1592, i64 3, !dbg !13
  %1597 = select i1 %.not, float %1594, float %1593, !dbg !13
  %1598 = bitcast float %1597 to i32, !dbg !13
  %1599 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1598), !dbg !13
  %1600 = bitcast i32 %1599 to float, !dbg !13
  %1601 = select i1 %.not, float %1593, float %1600, !dbg !13
  %1602 = select i1 %.not, float %1600, float %1594, !dbg !13
  %1603 = select i1 %.not, float %1596, float %1595, !dbg !13
  %1604 = bitcast float %1603 to i32, !dbg !13
  %1605 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1604), !dbg !13
  %1606 = bitcast i32 %1605 to float, !dbg !13
  %1607 = select i1 %.not, float %1595, float %1606, !dbg !13
  %1608 = select i1 %.not, float %1606, float %1596, !dbg !13
  %1609 = select i1 %.not1, float %1607, float %1601, !dbg !13
  %1610 = bitcast float %1609 to i32, !dbg !13
  %1611 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1610), !dbg !13
  %1612 = bitcast i32 %1611 to float, !dbg !13
  %1613 = select i1 %.not1, float %1608, float %1602, !dbg !13
  %1614 = bitcast float %1613 to i32, !dbg !13
  %1615 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1614), !dbg !13
  %1616 = bitcast i32 %1615 to float, !dbg !13
  %1617 = insertelement <4 x float> poison, float %1601, i64 0, !dbg !13
  %1618 = insertelement <4 x float> %1617, float %1602, i64 1, !dbg !13
  %1619 = insertelement <4 x float> %1618, float %1612, i64 2, !dbg !13
  %1620 = insertelement <4 x float> %1619, float %1616, i64 3, !dbg !13
  %1621 = insertelement <4 x float> poison, float %1612, i64 0, !dbg !13
  %1622 = insertelement <4 x float> %1621, float %1616, i64 1, !dbg !13
  %1623 = insertelement <4 x float> %1622, float %1607, i64 2, !dbg !13
  %1624 = insertelement <4 x float> %1623, float %1608, i64 3, !dbg !13
  %1625 = select <4 x i1> %136, <4 x float> %1620, <4 x float> %1624, !dbg !13
  %1626 = shufflevector <8 x float> %1573, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1627 = fmul <4 x float> %1626, %761, !dbg !25
  %1628 = fadd <4 x float> %1627, %1625, !dbg !30
  %1629 = shufflevector <4 x float> %1576, <4 x float> %1628, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1630 = insertvalue [2 x <8 x float>] poison, <8 x float> %1629, 0, !dbg !11
  %1631 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1524, <4 x float> zeroinitializer) #8, !dbg !13
  %1632 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1526, <4 x float> %1631) #8, !dbg !13
  %1633 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1528, <4 x float> %1632) #8, !dbg !13
  %1634 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1530, <4 x float> %1633) #8, !dbg !13
  %1635 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1532, <4 x float> %1634) #8, !dbg !13
  %1636 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1534, <4 x float> %1635) #8, !dbg !13
  %1637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1536, <4 x float> %1636) #8, !dbg !13
  %1638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1538, <4 x float> %1637) #8, !dbg !13
  %1639 = extractelement <4 x float> %1638, i64 0, !dbg !13
  %1640 = extractelement <4 x float> %1638, i64 1, !dbg !13
  %1641 = extractelement <4 x float> %1638, i64 2, !dbg !13
  %1642 = extractelement <4 x float> %1638, i64 3, !dbg !13
  %1643 = select i1 %.not, float %1640, float %1639, !dbg !13
  %1644 = bitcast float %1643 to i32, !dbg !13
  %1645 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1644), !dbg !13
  %1646 = bitcast i32 %1645 to float, !dbg !13
  %1647 = select i1 %.not, float %1639, float %1646, !dbg !13
  %1648 = select i1 %.not, float %1646, float %1640, !dbg !13
  %1649 = select i1 %.not, float %1642, float %1641, !dbg !13
  %1650 = bitcast float %1649 to i32, !dbg !13
  %1651 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1650), !dbg !13
  %1652 = bitcast i32 %1651 to float, !dbg !13
  %1653 = select i1 %.not, float %1641, float %1652, !dbg !13
  %1654 = select i1 %.not, float %1652, float %1642, !dbg !13
  %1655 = select i1 %.not1, float %1653, float %1647, !dbg !13
  %1656 = bitcast float %1655 to i32, !dbg !13
  %1657 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1656), !dbg !13
  %1658 = bitcast i32 %1657 to float, !dbg !13
  %1659 = select i1 %.not1, float %1654, float %1648, !dbg !13
  %1660 = bitcast float %1659 to i32, !dbg !13
  %1661 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1660), !dbg !13
  %1662 = bitcast i32 %1661 to float, !dbg !13
  %1663 = insertelement <4 x float> poison, float %1647, i64 0, !dbg !13
  %1664 = insertelement <4 x float> %1663, float %1648, i64 1, !dbg !13
  %1665 = insertelement <4 x float> %1664, float %1658, i64 2, !dbg !13
  %1666 = insertelement <4 x float> %1665, float %1662, i64 3, !dbg !13
  %1667 = insertelement <4 x float> poison, float %1658, i64 0, !dbg !13
  %1668 = insertelement <4 x float> %1667, float %1662, i64 1, !dbg !13
  %1669 = insertelement <4 x float> %1668, float %1653, i64 2, !dbg !13
  %1670 = insertelement <4 x float> %1669, float %1654, i64 3, !dbg !13
  %1671 = select <4 x i1> %136, <4 x float> %1666, <4 x float> %1670, !dbg !13
  %1672 = extractvalue [2 x <8 x float>] %143, 1, !dbg !25
  %1673 = shufflevector <8 x float> %1672, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1674 = fmul <4 x float> %1673, %769, !dbg !25
  %1675 = fadd <4 x float> %1674, %1671, !dbg !30
  %1676 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1577, <4 x float> zeroinitializer) #8, !dbg !13
  %1677 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1579, <4 x float> %1676) #8, !dbg !13
  %1678 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1581, <4 x float> %1677) #8, !dbg !13
  %1679 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1583, <4 x float> %1678) #8, !dbg !13
  %1680 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1585, <4 x float> %1679) #8, !dbg !13
  %1681 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1587, <4 x float> %1680) #8, !dbg !13
  %1682 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1589, <4 x float> %1681) #8, !dbg !13
  %1683 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1591, <4 x float> %1682) #8, !dbg !13
  %1684 = extractelement <4 x float> %1683, i64 0, !dbg !13
  %1685 = extractelement <4 x float> %1683, i64 1, !dbg !13
  %1686 = extractelement <4 x float> %1683, i64 2, !dbg !13
  %1687 = extractelement <4 x float> %1683, i64 3, !dbg !13
  %1688 = select i1 %.not, float %1685, float %1684, !dbg !13
  %1689 = bitcast float %1688 to i32, !dbg !13
  %1690 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1689), !dbg !13
  %1691 = bitcast i32 %1690 to float, !dbg !13
  %1692 = select i1 %.not, float %1684, float %1691, !dbg !13
  %1693 = select i1 %.not, float %1691, float %1685, !dbg !13
  %1694 = select i1 %.not, float %1687, float %1686, !dbg !13
  %1695 = bitcast float %1694 to i32, !dbg !13
  %1696 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1695), !dbg !13
  %1697 = bitcast i32 %1696 to float, !dbg !13
  %1698 = select i1 %.not, float %1686, float %1697, !dbg !13
  %1699 = select i1 %.not, float %1697, float %1687, !dbg !13
  %1700 = select i1 %.not1, float %1698, float %1692, !dbg !13
  %1701 = bitcast float %1700 to i32, !dbg !13
  %1702 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1701), !dbg !13
  %1703 = bitcast i32 %1702 to float, !dbg !13
  %1704 = select i1 %.not1, float %1699, float %1693, !dbg !13
  %1705 = bitcast float %1704 to i32, !dbg !13
  %1706 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1705), !dbg !13
  %1707 = bitcast i32 %1706 to float, !dbg !13
  %1708 = insertelement <4 x float> poison, float %1692, i64 0, !dbg !13
  %1709 = insertelement <4 x float> %1708, float %1693, i64 1, !dbg !13
  %1710 = insertelement <4 x float> %1709, float %1703, i64 2, !dbg !13
  %1711 = insertelement <4 x float> %1710, float %1707, i64 3, !dbg !13
  %1712 = insertelement <4 x float> poison, float %1703, i64 0, !dbg !13
  %1713 = insertelement <4 x float> %1712, float %1707, i64 1, !dbg !13
  %1714 = insertelement <4 x float> %1713, float %1698, i64 2, !dbg !13
  %1715 = insertelement <4 x float> %1714, float %1699, i64 3, !dbg !13
  %1716 = select <4 x i1> %136, <4 x float> %1711, <4 x float> %1715, !dbg !13
  %1717 = shufflevector <8 x float> %1672, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1718 = fmul <4 x float> %1717, %769, !dbg !25
  %1719 = fadd <4 x float> %1718, %1716, !dbg !30
  %1720 = shufflevector <4 x float> %1675, <4 x float> %1719, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1721 = insertvalue [2 x <8 x float>] %1630, <8 x float> %1720, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1722 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #11, !dbg !13
  %1723 = bitcast <4 x i32> %1722 to <8 x half>, !dbg !13
  %1724 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #11, !dbg !13
  %1725 = bitcast <4 x i32> %1724 to <8 x half>, !dbg !13
  %1726 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #11, !dbg !13
  %1727 = bitcast <4 x i32> %1726 to <8 x half>, !dbg !13
  %1728 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #11, !dbg !13
  %1729 = bitcast <4 x i32> %1728 to <8 x half>, !dbg !13
  %1730 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #11, !dbg !13
  %1731 = bitcast <4 x i32> %1730 to <8 x half>, !dbg !13
  %1732 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #11, !dbg !13
  %1733 = bitcast <4 x i32> %1732 to <8 x half>, !dbg !13
  %1734 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #11, !dbg !13
  %1735 = bitcast <4 x i32> %1734 to <8 x half>, !dbg !13
  %1736 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #11, !dbg !13
  %1737 = bitcast <4 x i32> %1736 to <8 x half>, !dbg !13
  %1738 = shufflevector <8 x half> %1723, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1739 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1738, <4 x float> zeroinitializer) #8, !dbg !13
  %1740 = shufflevector <8 x half> %1725, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1741 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1740, <4 x float> %1739) #8, !dbg !13
  %1742 = shufflevector <8 x half> %1727, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1742, <4 x float> %1741) #8, !dbg !13
  %1744 = shufflevector <8 x half> %1729, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1745 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1744, <4 x float> %1743) #8, !dbg !13
  %1746 = shufflevector <8 x half> %1731, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1747 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1746, <4 x float> %1745) #8, !dbg !13
  %1748 = shufflevector <8 x half> %1733, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1749 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1748, <4 x float> %1747) #8, !dbg !13
  %1750 = shufflevector <8 x half> %1735, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1751 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1750, <4 x float> %1749) #8, !dbg !13
  %1752 = shufflevector <8 x half> %1737, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1753 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1752, <4 x float> %1751) #8, !dbg !13
  %1754 = extractelement <4 x float> %1753, i64 0, !dbg !13
  %1755 = extractelement <4 x float> %1753, i64 1, !dbg !13
  %1756 = extractelement <4 x float> %1753, i64 2, !dbg !13
  %1757 = extractelement <4 x float> %1753, i64 3, !dbg !13
  %1758 = select i1 %.not, float %1755, float %1754, !dbg !13
  %1759 = bitcast float %1758 to i32, !dbg !13
  %1760 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1759), !dbg !13
  %1761 = bitcast i32 %1760 to float, !dbg !13
  %1762 = select i1 %.not, float %1754, float %1761, !dbg !13
  %1763 = select i1 %.not, float %1761, float %1755, !dbg !13
  %1764 = select i1 %.not, float %1757, float %1756, !dbg !13
  %1765 = bitcast float %1764 to i32, !dbg !13
  %1766 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1765), !dbg !13
  %1767 = bitcast i32 %1766 to float, !dbg !13
  %1768 = select i1 %.not, float %1756, float %1767, !dbg !13
  %1769 = select i1 %.not, float %1767, float %1757, !dbg !13
  %1770 = select i1 %.not1, float %1768, float %1762, !dbg !13
  %1771 = bitcast float %1770 to i32, !dbg !13
  %1772 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1771), !dbg !13
  %1773 = bitcast i32 %1772 to float, !dbg !13
  %1774 = select i1 %.not1, float %1769, float %1763, !dbg !13
  %1775 = bitcast float %1774 to i32, !dbg !13
  %1776 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1775), !dbg !13
  %1777 = bitcast i32 %1776 to float, !dbg !13
  %1778 = insertelement <4 x float> poison, float %1762, i64 0, !dbg !13
  %1779 = insertelement <4 x float> %1778, float %1763, i64 1, !dbg !13
  %1780 = insertelement <4 x float> %1779, float %1773, i64 2, !dbg !13
  %1781 = insertelement <4 x float> %1780, float %1777, i64 3, !dbg !13
  %1782 = insertelement <4 x float> poison, float %1773, i64 0, !dbg !13
  %1783 = insertelement <4 x float> %1782, float %1777, i64 1, !dbg !13
  %1784 = insertelement <4 x float> %1783, float %1768, i64 2, !dbg !13
  %1785 = insertelement <4 x float> %1784, float %1769, i64 3, !dbg !13
  %1786 = select <4 x i1> %136, <4 x float> %1781, <4 x float> %1785, !dbg !13
  %1787 = extractvalue [2 x <8 x float>] %142, 0, !dbg !25
  %1788 = shufflevector <8 x float> %1787, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1789 = fmul <4 x float> %1788, %761, !dbg !25
  %1790 = fadd <4 x float> %1789, %1786, !dbg !30
  %1791 = shufflevector <8 x half> %1723, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1792 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1791, <4 x float> zeroinitializer) #8, !dbg !13
  %1793 = shufflevector <8 x half> %1725, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1794 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1793, <4 x float> %1792) #8, !dbg !13
  %1795 = shufflevector <8 x half> %1727, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1796 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1795, <4 x float> %1794) #8, !dbg !13
  %1797 = shufflevector <8 x half> %1729, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1798 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1797, <4 x float> %1796) #8, !dbg !13
  %1799 = shufflevector <8 x half> %1731, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1800 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1799, <4 x float> %1798) #8, !dbg !13
  %1801 = shufflevector <8 x half> %1733, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1802 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1801, <4 x float> %1800) #8, !dbg !13
  %1803 = shufflevector <8 x half> %1735, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1804 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1803, <4 x float> %1802) #8, !dbg !13
  %1805 = shufflevector <8 x half> %1737, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1806 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1805, <4 x float> %1804) #8, !dbg !13
  %1807 = extractelement <4 x float> %1806, i64 0, !dbg !13
  %1808 = extractelement <4 x float> %1806, i64 1, !dbg !13
  %1809 = extractelement <4 x float> %1806, i64 2, !dbg !13
  %1810 = extractelement <4 x float> %1806, i64 3, !dbg !13
  %1811 = select i1 %.not, float %1808, float %1807, !dbg !13
  %1812 = bitcast float %1811 to i32, !dbg !13
  %1813 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1812), !dbg !13
  %1814 = bitcast i32 %1813 to float, !dbg !13
  %1815 = select i1 %.not, float %1807, float %1814, !dbg !13
  %1816 = select i1 %.not, float %1814, float %1808, !dbg !13
  %1817 = select i1 %.not, float %1810, float %1809, !dbg !13
  %1818 = bitcast float %1817 to i32, !dbg !13
  %1819 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1818), !dbg !13
  %1820 = bitcast i32 %1819 to float, !dbg !13
  %1821 = select i1 %.not, float %1809, float %1820, !dbg !13
  %1822 = select i1 %.not, float %1820, float %1810, !dbg !13
  %1823 = select i1 %.not1, float %1821, float %1815, !dbg !13
  %1824 = bitcast float %1823 to i32, !dbg !13
  %1825 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1824), !dbg !13
  %1826 = bitcast i32 %1825 to float, !dbg !13
  %1827 = select i1 %.not1, float %1822, float %1816, !dbg !13
  %1828 = bitcast float %1827 to i32, !dbg !13
  %1829 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1828), !dbg !13
  %1830 = bitcast i32 %1829 to float, !dbg !13
  %1831 = insertelement <4 x float> poison, float %1815, i64 0, !dbg !13
  %1832 = insertelement <4 x float> %1831, float %1816, i64 1, !dbg !13
  %1833 = insertelement <4 x float> %1832, float %1826, i64 2, !dbg !13
  %1834 = insertelement <4 x float> %1833, float %1830, i64 3, !dbg !13
  %1835 = insertelement <4 x float> poison, float %1826, i64 0, !dbg !13
  %1836 = insertelement <4 x float> %1835, float %1830, i64 1, !dbg !13
  %1837 = insertelement <4 x float> %1836, float %1821, i64 2, !dbg !13
  %1838 = insertelement <4 x float> %1837, float %1822, i64 3, !dbg !13
  %1839 = select <4 x i1> %136, <4 x float> %1834, <4 x float> %1838, !dbg !13
  %1840 = shufflevector <8 x float> %1787, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1841 = fmul <4 x float> %1840, %761, !dbg !25
  %1842 = fadd <4 x float> %1841, %1839, !dbg !30
  %1843 = shufflevector <4 x float> %1790, <4 x float> %1842, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1844 = insertvalue [2 x <8 x float>] poison, <8 x float> %1843, 0, !dbg !11
  %1845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1738, <4 x float> zeroinitializer) #8, !dbg !13
  %1846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1740, <4 x float> %1845) #8, !dbg !13
  %1847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1742, <4 x float> %1846) #8, !dbg !13
  %1848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1744, <4 x float> %1847) #8, !dbg !13
  %1849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1746, <4 x float> %1848) #8, !dbg !13
  %1850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1748, <4 x float> %1849) #8, !dbg !13
  %1851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1750, <4 x float> %1850) #8, !dbg !13
  %1852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1752, <4 x float> %1851) #8, !dbg !13
  %1853 = extractelement <4 x float> %1852, i64 0, !dbg !13
  %1854 = extractelement <4 x float> %1852, i64 1, !dbg !13
  %1855 = extractelement <4 x float> %1852, i64 2, !dbg !13
  %1856 = extractelement <4 x float> %1852, i64 3, !dbg !13
  %1857 = select i1 %.not, float %1854, float %1853, !dbg !13
  %1858 = bitcast float %1857 to i32, !dbg !13
  %1859 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1858), !dbg !13
  %1860 = bitcast i32 %1859 to float, !dbg !13
  %1861 = select i1 %.not, float %1853, float %1860, !dbg !13
  %1862 = select i1 %.not, float %1860, float %1854, !dbg !13
  %1863 = select i1 %.not, float %1856, float %1855, !dbg !13
  %1864 = bitcast float %1863 to i32, !dbg !13
  %1865 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1864), !dbg !13
  %1866 = bitcast i32 %1865 to float, !dbg !13
  %1867 = select i1 %.not, float %1855, float %1866, !dbg !13
  %1868 = select i1 %.not, float %1866, float %1856, !dbg !13
  %1869 = select i1 %.not1, float %1867, float %1861, !dbg !13
  %1870 = bitcast float %1869 to i32, !dbg !13
  %1871 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1870), !dbg !13
  %1872 = bitcast i32 %1871 to float, !dbg !13
  %1873 = select i1 %.not1, float %1868, float %1862, !dbg !13
  %1874 = bitcast float %1873 to i32, !dbg !13
  %1875 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1874), !dbg !13
  %1876 = bitcast i32 %1875 to float, !dbg !13
  %1877 = insertelement <4 x float> poison, float %1861, i64 0, !dbg !13
  %1878 = insertelement <4 x float> %1877, float %1862, i64 1, !dbg !13
  %1879 = insertelement <4 x float> %1878, float %1872, i64 2, !dbg !13
  %1880 = insertelement <4 x float> %1879, float %1876, i64 3, !dbg !13
  %1881 = insertelement <4 x float> poison, float %1872, i64 0, !dbg !13
  %1882 = insertelement <4 x float> %1881, float %1876, i64 1, !dbg !13
  %1883 = insertelement <4 x float> %1882, float %1867, i64 2, !dbg !13
  %1884 = insertelement <4 x float> %1883, float %1868, i64 3, !dbg !13
  %1885 = select <4 x i1> %136, <4 x float> %1880, <4 x float> %1884, !dbg !13
  %1886 = extractvalue [2 x <8 x float>] %142, 1, !dbg !25
  %1887 = shufflevector <8 x float> %1886, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1888 = fmul <4 x float> %1887, %769, !dbg !25
  %1889 = fadd <4 x float> %1888, %1885, !dbg !30
  %1890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1791, <4 x float> zeroinitializer) #8, !dbg !13
  %1891 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1793, <4 x float> %1890) #8, !dbg !13
  %1892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1795, <4 x float> %1891) #8, !dbg !13
  %1893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1797, <4 x float> %1892) #8, !dbg !13
  %1894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1799, <4 x float> %1893) #8, !dbg !13
  %1895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1801, <4 x float> %1894) #8, !dbg !13
  %1896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1803, <4 x float> %1895) #8, !dbg !13
  %1897 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1805, <4 x float> %1896) #8, !dbg !13
  %1898 = extractelement <4 x float> %1897, i64 0, !dbg !13
  %1899 = extractelement <4 x float> %1897, i64 1, !dbg !13
  %1900 = extractelement <4 x float> %1897, i64 2, !dbg !13
  %1901 = extractelement <4 x float> %1897, i64 3, !dbg !13
  %1902 = select i1 %.not, float %1899, float %1898, !dbg !13
  %1903 = bitcast float %1902 to i32, !dbg !13
  %1904 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1903), !dbg !13
  %1905 = bitcast i32 %1904 to float, !dbg !13
  %1906 = select i1 %.not, float %1898, float %1905, !dbg !13
  %1907 = select i1 %.not, float %1905, float %1899, !dbg !13
  %1908 = select i1 %.not, float %1901, float %1900, !dbg !13
  %1909 = bitcast float %1908 to i32, !dbg !13
  %1910 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %1909), !dbg !13
  %1911 = bitcast i32 %1910 to float, !dbg !13
  %1912 = select i1 %.not, float %1900, float %1911, !dbg !13
  %1913 = select i1 %.not, float %1911, float %1901, !dbg !13
  %1914 = select i1 %.not1, float %1912, float %1906, !dbg !13
  %1915 = bitcast float %1914 to i32, !dbg !13
  %1916 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1915), !dbg !13
  %1917 = bitcast i32 %1916 to float, !dbg !13
  %1918 = select i1 %.not1, float %1913, float %1907, !dbg !13
  %1919 = bitcast float %1918 to i32, !dbg !13
  %1920 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %1919), !dbg !13
  %1921 = bitcast i32 %1920 to float, !dbg !13
  %1922 = insertelement <4 x float> poison, float %1906, i64 0, !dbg !13
  %1923 = insertelement <4 x float> %1922, float %1907, i64 1, !dbg !13
  %1924 = insertelement <4 x float> %1923, float %1917, i64 2, !dbg !13
  %1925 = insertelement <4 x float> %1924, float %1921, i64 3, !dbg !13
  %1926 = insertelement <4 x float> poison, float %1917, i64 0, !dbg !13
  %1927 = insertelement <4 x float> %1926, float %1921, i64 1, !dbg !13
  %1928 = insertelement <4 x float> %1927, float %1912, i64 2, !dbg !13
  %1929 = insertelement <4 x float> %1928, float %1913, i64 3, !dbg !13
  %1930 = select <4 x i1> %136, <4 x float> %1925, <4 x float> %1929, !dbg !13
  %1931 = shufflevector <8 x float> %1886, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1932 = fmul <4 x float> %1931, %769, !dbg !25
  %1933 = fadd <4 x float> %1932, %1930, !dbg !30
  %1934 = shufflevector <4 x float> %1889, <4 x float> %1933, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1935 = insertvalue [2 x <8 x float>] %1844, <8 x float> %1934, 1, !dbg !11
  %1936 = insertvalue [2 x <1 x float>] poison, <1 x float> %695, 0, !dbg !11
  %1937 = insertvalue [2 x <1 x float>] %1936, <1 x float> %738, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1938 = add nuw nsw i64 %148, 1, !dbg !11
  %1939 = icmp slt i64 %1938, %105, !dbg !11
  br i1 %1939, label %141, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %541, %4
  %.lcssa94 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1286, %541 ], !dbg !11
  %.lcssa93 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1064, %541 ], !dbg !11
  %.lcssa92 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1502, %541 ], !dbg !11
  %.lcssa91 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1721, %541 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1935, %541 ], !dbg !11
  %1940 = extractvalue [2 x <1 x float>] %.lcssa93, 0, !dbg !31
  %1941 = shufflevector <1 x float> %1940, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !31
  %1942 = extractvalue [2 x <1 x float>] %.lcssa93, 1, !dbg !31
  %1943 = shufflevector <1 x float> %1942, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !31
  %1944 = extractvalue [2 x <8 x float>] %.lcssa94, 1, !dbg !31
  %1945 = shufflevector <8 x float> %1944, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1943, !dbg !31
  %normalized = fmul <4 x float> %1945, %row.reciprocal, !dbg !31
  %1946 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !32
  %1947 = extractelement <4 x half> %1946, i64 3, !dbg !32
  %1948 = extractelement <4 x half> %1946, i64 2, !dbg !32
  %1949 = extractelement <4 x half> %1946, i64 1, !dbg !32
  %1950 = extractelement <4 x half> %1946, i64 0, !dbg !32
  %1951 = shufflevector <8 x float> %1944, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized104 = fmul <4 x float> %1951, %row.reciprocal, !dbg !31
  %1952 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !32
  %1953 = extractelement <4 x half> %1952, i64 3, !dbg !32
  %1954 = extractelement <4 x half> %1952, i64 2, !dbg !32
  %1955 = extractelement <4 x half> %1952, i64 1, !dbg !32
  %1956 = extractelement <4 x half> %1952, i64 0, !dbg !32
  %1957 = extractvalue [2 x <8 x float>] %.lcssa94, 0, !dbg !31
  %1958 = shufflevector <8 x float> %1957, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1941, !dbg !31
  %normalized106 = fmul <4 x float> %1958, %row.reciprocal105, !dbg !31
  %1959 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !32
  %1960 = extractelement <4 x half> %1959, i64 3, !dbg !32
  %1961 = extractelement <4 x half> %1959, i64 2, !dbg !32
  %1962 = extractelement <4 x half> %1959, i64 1, !dbg !32
  %1963 = extractelement <4 x half> %1959, i64 0, !dbg !32
  %1964 = shufflevector <8 x float> %1957, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized107 = fmul <4 x float> %1964, %row.reciprocal105, !dbg !31
  %1965 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !32
  %1966 = extractelement <4 x half> %1965, i64 3, !dbg !32
  %1967 = extractelement <4 x half> %1965, i64 2, !dbg !32
  %1968 = extractelement <4 x half> %1965, i64 1, !dbg !32
  %1969 = extractelement <4 x half> %1965, i64 0, !dbg !32
  %address.thread934 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !33, !range !9, !frisk.register_lifetimes !5
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !8
  %1970 = lshr i64 %address.local935, 2, !dbg !33
  %address.local938 = and i64 %1970, 48, !dbg !33
  %address.local940 = and i64 %address.local935, 240, !dbg !33
  %address.local939 = add i64 %26, %address.local935, !dbg !33
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !33
  %address.local942 = add i64 %address.local941, %address.local938, !dbg !33
  %1971 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !33
  %address.local945.idx = shl nuw nsw i64 %address.local942, 8, !dbg !33
  %address.local945 = getelementptr inbounds nuw i8, ptr addrspace(1) %1971, i64 %address.local945.idx, !dbg !33
  %address.local947 = and i64 %1970, 12, !dbg !10
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !33
  store half %1969, ptr addrspace(1) %address.local948, align 2, !dbg !33
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !33
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local950, !dbg !33
  store half %1968, ptr addrspace(1) %address.local951, align 2, !dbg !33
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !33
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local953, !dbg !33
  store half %1967, ptr addrspace(1) %address.local954, align 2, !dbg !33
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !33
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local956, !dbg !33
  store half %1966, ptr addrspace(1) %address.local957, align 2, !dbg !33
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !33
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local959, !dbg !33
  store half %1963, ptr addrspace(1) %address.local960, align 2, !dbg !33
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !33
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local962, !dbg !33
  store half %1962, ptr addrspace(1) %address.local963, align 2, !dbg !33
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !33
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local965, !dbg !33
  store half %1961, ptr addrspace(1) %address.local966, align 2, !dbg !33
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !33
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local968, !dbg !33
  store half %1960, ptr addrspace(1) %address.local969, align 2, !dbg !33
  %address.local973 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local945, i64 16384, !dbg !33
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !33
  store half %1956, ptr addrspace(1) %address.local974, align 2, !dbg !33
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local950, !dbg !33
  store half %1955, ptr addrspace(1) %address.local976, align 2, !dbg !33
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local953, !dbg !33
  store half %1954, ptr addrspace(1) %address.local978, align 2, !dbg !33
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local956, !dbg !33
  store half %1953, ptr addrspace(1) %address.local980, align 2, !dbg !33
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local959, !dbg !33
  store half %1950, ptr addrspace(1) %address.local982, align 2, !dbg !33
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local962, !dbg !33
  store half %1949, ptr addrspace(1) %address.local984, align 2, !dbg !33
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local965, !dbg !33
  store half %1948, ptr addrspace(1) %address.local986, align 2, !dbg !33
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local968, !dbg !33
  store half %1947, ptr addrspace(1) %address.local988, align 2, !dbg !33
  %1972 = extractvalue [2 x <8 x float>] %.lcssa92, 1, !dbg !31
  %1973 = shufflevector <8 x float> %1972, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized109 = fmul <4 x float> %row.reciprocal, %1973, !dbg !31
  %1974 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !32
  %1975 = extractelement <4 x half> %1974, i64 3, !dbg !32
  %1976 = extractelement <4 x half> %1974, i64 2, !dbg !32
  %1977 = extractelement <4 x half> %1974, i64 1, !dbg !32
  %1978 = extractelement <4 x half> %1974, i64 0, !dbg !32
  %1979 = shufflevector <8 x float> %1972, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized110 = fmul <4 x float> %row.reciprocal, %1979, !dbg !31
  %1980 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !32
  %1981 = extractelement <4 x half> %1980, i64 3, !dbg !32
  %1982 = extractelement <4 x half> %1980, i64 2, !dbg !32
  %1983 = extractelement <4 x half> %1980, i64 1, !dbg !32
  %1984 = extractelement <4 x half> %1980, i64 0, !dbg !32
  %1985 = extractvalue [2 x <8 x float>] %.lcssa92, 0, !dbg !31
  %1986 = shufflevector <8 x float> %1985, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized112 = fmul <4 x float> %row.reciprocal105, %1986, !dbg !31
  %1987 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !32
  %1988 = extractelement <4 x half> %1987, i64 3, !dbg !32
  %1989 = extractelement <4 x half> %1987, i64 2, !dbg !32
  %1990 = extractelement <4 x half> %1987, i64 1, !dbg !32
  %1991 = extractelement <4 x half> %1987, i64 0, !dbg !32
  %1992 = shufflevector <8 x float> %1985, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized113 = fmul <4 x float> %row.reciprocal105, %1992, !dbg !31
  %1993 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !32
  %1994 = extractelement <4 x half> %1993, i64 3, !dbg !32
  %1995 = extractelement <4 x half> %1993, i64 2, !dbg !32
  %1996 = extractelement <4 x half> %1993, i64 1, !dbg !32
  %1997 = extractelement <4 x half> %1993, i64 0, !dbg !32
  %address.local1003 = or disjoint i64 %address.local947, 32, !dbg !33
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1003, !dbg !33
  store half %1997, ptr addrspace(1) %address.local1004, align 2, !dbg !33
  %address.local1006 = or disjoint i64 %address.local947, 33, !dbg !33
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1006, !dbg !33
  store half %1996, ptr addrspace(1) %address.local1007, align 2, !dbg !33
  %address.local1009 = or disjoint i64 %address.local947, 34, !dbg !33
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1009, !dbg !33
  store half %1995, ptr addrspace(1) %address.local1010, align 2, !dbg !33
  %address.local1012 = or disjoint i64 %address.local947, 35, !dbg !33
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1012, !dbg !33
  store half %1994, ptr addrspace(1) %address.local1013, align 2, !dbg !33
  %address.local1015 = or disjoint i64 %address.local947, 48, !dbg !33
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1015, !dbg !33
  store half %1991, ptr addrspace(1) %address.local1016, align 2, !dbg !33
  %address.local1018 = or disjoint i64 %address.local947, 49, !dbg !33
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1018, !dbg !33
  store half %1990, ptr addrspace(1) %address.local1019, align 2, !dbg !33
  %address.local1021 = or disjoint i64 %address.local947, 50, !dbg !33
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1021, !dbg !33
  store half %1989, ptr addrspace(1) %address.local1022, align 2, !dbg !33
  %address.local1024 = or i64 %1970, 51, !dbg !33
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1024, !dbg !33
  store half %1988, ptr addrspace(1) %address.local1025, align 2, !dbg !33
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1003, !dbg !33
  store half %1984, ptr addrspace(1) %address.local1030, align 2, !dbg !33
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1006, !dbg !33
  store half %1983, ptr addrspace(1) %address.local1032, align 2, !dbg !33
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1009, !dbg !33
  store half %1982, ptr addrspace(1) %address.local1034, align 2, !dbg !33
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1012, !dbg !33
  store half %1981, ptr addrspace(1) %address.local1036, align 2, !dbg !33
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1015, !dbg !33
  store half %1978, ptr addrspace(1) %address.local1038, align 2, !dbg !33
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1018, !dbg !33
  store half %1977, ptr addrspace(1) %address.local1040, align 2, !dbg !33
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1021, !dbg !33
  store half %1976, ptr addrspace(1) %address.local1042, align 2, !dbg !33
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1024, !dbg !33
  store half %1975, ptr addrspace(1) %address.local1044, align 2, !dbg !33
  %1998 = extractvalue [2 x <8 x float>] %.lcssa91, 1, !dbg !31
  %1999 = shufflevector <8 x float> %1998, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized115 = fmul <4 x float> %row.reciprocal, %1999, !dbg !31
  %2000 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !32
  %2001 = extractelement <4 x half> %2000, i64 3, !dbg !32
  %2002 = extractelement <4 x half> %2000, i64 2, !dbg !32
  %2003 = extractelement <4 x half> %2000, i64 1, !dbg !32
  %2004 = extractelement <4 x half> %2000, i64 0, !dbg !32
  %2005 = shufflevector <8 x float> %1998, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized116 = fmul <4 x float> %row.reciprocal, %2005, !dbg !31
  %2006 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !32
  %2007 = extractelement <4 x half> %2006, i64 3, !dbg !32
  %2008 = extractelement <4 x half> %2006, i64 2, !dbg !32
  %2009 = extractelement <4 x half> %2006, i64 1, !dbg !32
  %2010 = extractelement <4 x half> %2006, i64 0, !dbg !32
  %2011 = extractvalue [2 x <8 x float>] %.lcssa91, 0, !dbg !31
  %2012 = shufflevector <8 x float> %2011, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized118 = fmul <4 x float> %row.reciprocal105, %2012, !dbg !31
  %2013 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !32
  %2014 = extractelement <4 x half> %2013, i64 3, !dbg !32
  %2015 = extractelement <4 x half> %2013, i64 2, !dbg !32
  %2016 = extractelement <4 x half> %2013, i64 1, !dbg !32
  %2017 = extractelement <4 x half> %2013, i64 0, !dbg !32
  %2018 = shufflevector <8 x float> %2011, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized119 = fmul <4 x float> %row.reciprocal105, %2018, !dbg !31
  %2019 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !32
  %2020 = extractelement <4 x half> %2019, i64 3, !dbg !32
  %2021 = extractelement <4 x half> %2019, i64 2, !dbg !32
  %2022 = extractelement <4 x half> %2019, i64 1, !dbg !32
  %2023 = extractelement <4 x half> %2019, i64 0, !dbg !32
  %address.local1059 = or disjoint i64 %address.local947, 64, !dbg !33
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1059, !dbg !33
  store half %2023, ptr addrspace(1) %address.local1060, align 2, !dbg !33
  %address.local1062 = or disjoint i64 %address.local947, 65, !dbg !33
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1062, !dbg !33
  store half %2022, ptr addrspace(1) %address.local1063, align 2, !dbg !33
  %address.local1065 = or disjoint i64 %address.local947, 66, !dbg !33
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1065, !dbg !33
  store half %2021, ptr addrspace(1) %address.local1066, align 2, !dbg !33
  %address.local1068 = or disjoint i64 %address.local947, 67, !dbg !33
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1068, !dbg !33
  store half %2020, ptr addrspace(1) %address.local1069, align 2, !dbg !33
  %address.local1071 = or disjoint i64 %address.local947, 80, !dbg !33
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1071, !dbg !33
  store half %2017, ptr addrspace(1) %address.local1072, align 2, !dbg !33
  %address.local1074 = or disjoint i64 %address.local947, 81, !dbg !33
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1074, !dbg !33
  store half %2016, ptr addrspace(1) %address.local1075, align 2, !dbg !33
  %address.local1077 = or disjoint i64 %address.local947, 82, !dbg !33
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1077, !dbg !33
  store half %2015, ptr addrspace(1) %address.local1078, align 2, !dbg !33
  %address.local1080 = or disjoint i64 %address.local947, 83, !dbg !33
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1080, !dbg !33
  store half %2014, ptr addrspace(1) %address.local1081, align 2, !dbg !33
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1059, !dbg !33
  store half %2010, ptr addrspace(1) %address.local1086, align 2, !dbg !33
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1062, !dbg !33
  store half %2009, ptr addrspace(1) %address.local1088, align 2, !dbg !33
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1065, !dbg !33
  store half %2008, ptr addrspace(1) %address.local1090, align 2, !dbg !33
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1068, !dbg !33
  store half %2007, ptr addrspace(1) %address.local1092, align 2, !dbg !33
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1071, !dbg !33
  store half %2004, ptr addrspace(1) %address.local1094, align 2, !dbg !33
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1074, !dbg !33
  store half %2003, ptr addrspace(1) %address.local1096, align 2, !dbg !33
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1077, !dbg !33
  store half %2002, ptr addrspace(1) %address.local1098, align 2, !dbg !33
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1080, !dbg !33
  store half %2001, ptr addrspace(1) %address.local1100, align 2, !dbg !33
  %2024 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !31
  %2025 = shufflevector <8 x float> %2024, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized121 = fmul <4 x float> %row.reciprocal, %2025, !dbg !31
  %2026 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !32
  %2027 = extractelement <4 x half> %2026, i64 3, !dbg !32
  %2028 = extractelement <4 x half> %2026, i64 2, !dbg !32
  %2029 = extractelement <4 x half> %2026, i64 1, !dbg !32
  %2030 = extractelement <4 x half> %2026, i64 0, !dbg !32
  %2031 = shufflevector <8 x float> %2024, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized122 = fmul <4 x float> %row.reciprocal, %2031, !dbg !31
  %2032 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !32
  %2033 = extractelement <4 x half> %2032, i64 3, !dbg !32
  %2034 = extractelement <4 x half> %2032, i64 2, !dbg !32
  %2035 = extractelement <4 x half> %2032, i64 1, !dbg !32
  %2036 = extractelement <4 x half> %2032, i64 0, !dbg !32
  %2037 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !31
  %2038 = shufflevector <8 x float> %2037, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %normalized124 = fmul <4 x float> %row.reciprocal105, %2038, !dbg !31
  %2039 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !32
  %2040 = extractelement <4 x half> %2039, i64 3, !dbg !32
  %2041 = extractelement <4 x half> %2039, i64 2, !dbg !32
  %2042 = extractelement <4 x half> %2039, i64 1, !dbg !32
  %2043 = extractelement <4 x half> %2039, i64 0, !dbg !32
  %2044 = shufflevector <8 x float> %2037, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %normalized125 = fmul <4 x float> %row.reciprocal105, %2044, !dbg !31
  %2045 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !32
  %2046 = extractelement <4 x half> %2045, i64 3, !dbg !32
  %2047 = extractelement <4 x half> %2045, i64 2, !dbg !32
  %2048 = extractelement <4 x half> %2045, i64 1, !dbg !32
  %2049 = extractelement <4 x half> %2045, i64 0, !dbg !32
  %address.local1115 = or disjoint i64 %address.local947, 96, !dbg !33
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1115, !dbg !33
  store half %2049, ptr addrspace(1) %address.local1116, align 2, !dbg !33
  %address.local1118 = or disjoint i64 %address.local947, 97, !dbg !33
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1118, !dbg !33
  store half %2048, ptr addrspace(1) %address.local1119, align 2, !dbg !33
  %address.local1121 = or disjoint i64 %address.local947, 98, !dbg !33
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1121, !dbg !33
  store half %2047, ptr addrspace(1) %address.local1122, align 2, !dbg !33
  %address.local1124 = or disjoint i64 %address.local947, 99, !dbg !33
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1124, !dbg !33
  store half %2046, ptr addrspace(1) %address.local1125, align 2, !dbg !33
  %address.local1127 = or disjoint i64 %address.local947, 112, !dbg !33
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1127, !dbg !33
  store half %2043, ptr addrspace(1) %address.local1128, align 2, !dbg !33
  %address.local1130 = or disjoint i64 %address.local947, 113, !dbg !33
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1130, !dbg !33
  store half %2042, ptr addrspace(1) %address.local1131, align 2, !dbg !33
  %address.local1133 = or disjoint i64 %address.local947, 114, !dbg !33
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1133, !dbg !33
  store half %2041, ptr addrspace(1) %address.local1134, align 2, !dbg !33
  %address.local1136 = or i64 %1970, 115, !dbg !33
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local1136, !dbg !33
  store half %2040, ptr addrspace(1) %address.local1137, align 2, !dbg !33
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1115, !dbg !33
  store half %2036, ptr addrspace(1) %address.local1142, align 2, !dbg !33
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1118, !dbg !33
  store half %2035, ptr addrspace(1) %address.local1144, align 2, !dbg !33
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1121, !dbg !33
  store half %2034, ptr addrspace(1) %address.local1146, align 2, !dbg !33
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1124, !dbg !33
  store half %2033, ptr addrspace(1) %address.local1148, align 2, !dbg !33
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1127, !dbg !33
  store half %2030, ptr addrspace(1) %address.local1150, align 2, !dbg !33
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1130, !dbg !33
  store half %2029, ptr addrspace(1) %address.local1152, align 2, !dbg !33
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1133, !dbg !33
  store half %2028, ptr addrspace(1) %address.local1154, align 2, !dbg !33
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local1136, !dbg !33
  store half %2027, ptr addrspace(1) %address.local1156, align 2, !dbg !33
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
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #5

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
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/release/run-20261023-172519-7itoy6x9/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline")
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
