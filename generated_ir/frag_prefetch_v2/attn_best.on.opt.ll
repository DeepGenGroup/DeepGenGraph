; ModuleID = '<stdin>'
source_filename = "3rd/deepgengraph/test/test_input_sd16.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = shl nsw i64 %6, 7, !dbg !8
  %.lobit = ashr i32 %5, 31, !dbg !9
  %10 = xor i32 %.lobit, %5, !dbg !9
  %11 = sdiv i32 %10, 32, !dbg !9
  %12 = xor i32 %11, %.lobit, !dbg !9
  %13 = sext i32 %12 to i64, !dbg !9
  %14 = add nsw i64 %13, %8, !dbg !9
  %15 = and i64 %14, 31, !dbg !9
  %16 = mul nsw i64 %13, -4096, !dbg !9
  %17 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !10, !range !11
  %.not75 = icmp slt i32 %5, 0, !dbg !12
  br i1 %.not75, label %.._crit_edge_crit_edge, label %.lr.ph, !dbg !12

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !12
  %.pre101 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !12
  %.pre103 = add nuw nsw i32 %.pre101, 64, !dbg !12
  %.pre105 = xor i32 %.pre101, 16, !dbg !12
  %.pre107 = and i32 %.pre103, 192, !dbg !12
  %.pre109 = xor i32 %.pre101, 32, !dbg !12
  %.pre111 = add nsw i64 %16, %9, !dbg !13
  %.pre112 = shl nuw nsw i64 %15, 20, !dbg !13
  br label %._crit_edge, !dbg !12

.lr.ph:                                           ; preds = %4
  %18 = zext nneg i32 %17 to i64, !dbg !10
  %19 = add nsw i64 %16, %9, !dbg !9
  %.idx = shl nuw nsw i64 %15, 20
  %.idx71 = shl nsw i64 %19, 8
  %20 = ptrtoint ptr addrspace(1) %0 to i64
  %21 = add nsw i64 %.idx, %.idx71
  %22 = insertelement <2 x i64> poison, i64 %20, i64 0
  %23 = bitcast <2 x i64> %22 to <4 x i32>
  %dma.buffer.rsrc624 = shufflevector <4 x i32> %23, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %24 = ptrtoint ptr addrspace(1) %1 to i64
  %25 = insertelement <2 x i64> poison, i64 %24, i64 0
  %26 = bitcast <2 x i64> %25 to <4 x i32>
  %dma.buffer.rsrc618 = shufflevector <4 x i32> %26, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %27 = ptrtoint ptr addrspace(1) %2 to i64
  %28 = insertelement <2 x i64> poison, i64 %27, i64 0
  %29 = bitcast <2 x i64> %28 to <4 x i32>
  %dma.buffer.rsrc648 = shufflevector <4 x i32> %29, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %30 = add nuw nsw i64 %9, %18
  %31 = lshr i64 %18, 2
  %32 = and i64 %31, 48
  %33 = and i32 %17, 240
  %index.compact = zext nneg i32 %33 to i64
  %34 = sub nsw i64 %30, %index.compact
  %35 = add nsw i64 %34, %32
  %36 = and i64 %18, 48
  %37 = add nsw i64 %35, 1
  %38 = add nsw i64 %35, 65
  %39 = insertelement <64 x i64> poison, i64 %37, i64 0, !dbg !14
  %40 = insertelement <64 x i64> %39, i64 %38, i64 1, !dbg !14
  %41 = shufflevector <64 x i64> %40, <64 x i64> poison, <64 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, !dbg !14
  br label %42, !dbg !12

42:                                               ; preds = %.lr.ph, %314
  %43 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %931, %314 ]
  %44 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %927, %314 ]
  %45 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %923, %314 ]
  %46 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %623, %314 ]
  %47 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %919, %314 ]
  %48 = phi i64 [ 0, %.lr.ph ], [ %932, %314 ]
  %49 = shl nuw nsw i64 %48, 7, !dbg !12
  %50 = lshr i64 %48, 5, !dbg !15
  %51 = shl i64 %50, 20
  %52 = shl i64 %48, 15
  %.idx74 = sub i64 %52, %51
  br label %53, !dbg !16

53:                                               ; preds = %42, %53
  %54 = phi [2 x <32 x float>] [ zeroinitializer, %42 ], [ %204, %53 ]
  %55 = phi i64 [ 0, %42 ], [ %205, %53 ]
  %56 = shl nuw nsw i64 %55, 5, !dbg !16
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #9, !dbg !9, !range !11, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !10
  %address.local133 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local134 = and i64 %address.local133, 1536, !dbg !9
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local134, !dbg !9
  %address.local138 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local139 = and i64 %address.local138, 16256, !dbg !9
  %address.local140 = lshr i64 %address.local, 4, !dbg !9
  %address.local141 = and i64 %address.local140, 8, !dbg !9
  %address.local142 = or disjoint i64 %address.local139, %address.local141, !dbg !9
  %.idx15 = shl nuw nsw i64 %55, 6, !dbg !9
  %57 = add nuw nsw i64 %.idx15, %21, !dbg !9
  %address.local143.idx = shl nuw nsw i64 %address.local142, 1, !dbg !9
  %58 = add nuw nsw i64 %address.local143.idx, %57, !dbg !9
  %dma.buffer.offset623 = trunc i64 %58 to i32, !dbg !9
  %address.local144 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local145 = shl nuw nsw i64 %address.local144, 3, !dbg !9
  %address.local146 = and i64 %address.local145, 3584, !dbg !9
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !9
  %address.local152 = lshr i64 %address.local144, 4, !dbg !9
  %address.local153 = and i64 %address.local152, 24, !dbg !9
  %address.local154 = or disjoint i64 %address.local153, %address.local139, !dbg !9
  %address.local155.idx = shl nuw nsw i64 %address.local154, 1, !dbg !9
  %59 = add nuw nsw i64 %address.local155.idx, %57, !dbg !9
  %dma.buffer.offset620 = trunc i64 %59 to i32, !dbg !9
  %60 = shl nuw nsw i64 %55, 10, !dbg !15
  %61 = add nuw nsw i64 %60, %48, !dbg !15
  %62 = lshr i64 %61, 12, !dbg !15
  %63 = add nsw i64 %62, %8, !dbg !15
  %64 = add nuw nsw i64 %56, %50, !dbg !15
  %65 = and i64 %64, 127, !dbg !15
  %66 = shl i64 %63, 19, !dbg !15
  %67 = and i64 %66, 16252928, !dbg !15
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local134, !dbg !15
  %.idx1719 = or disjoint i64 %67, %65, !dbg !15
  %68 = add nuw nsw i64 %address.local142, %.idx1719, !dbg !15
  %69 = shl nuw nsw i64 %68, 1, !dbg !15
  %70 = add i64 %69, %.idx74, !dbg !15
  %dma.buffer.offset617 = trunc i64 %70 to i32, !dbg !15
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local146, !dbg !15
  %71 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local135, i64 0, !dbg !9
  %72 = insertelement <4 x ptr addrspace(3)> %71, ptr addrspace(3) %address.local147, i64 1, !dbg !9
  %73 = insertelement <4 x ptr addrspace(3)> %72, ptr addrspace(3) %address.local156, i64 2, !dbg !9
  %74 = insertelement <4 x ptr addrspace(3)> %73, ptr addrspace(3) %address.local160, i64 3, !dbg !9
  %75 = ptrtoint <4 x ptr addrspace(3)> %74 to <4 x i64>, !dbg !9
  %76 = trunc <4 x i64> %75 to <4 x i32>, !dbg !9
  %77 = extractelement <4 x i32> %76, i64 0, !dbg !9
  %78 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %77), !dbg !9
  %dma.buffer.lds625 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %78), !dbg !9
  %79 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds625, i32 %dma.buffer.offset623, <4 x i32> %dma.buffer.rsrc624) #10, !dbg !9
  %80 = extractelement <4 x i32> %76, i64 1, !dbg !9
  %81 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !9
  %dma.buffer.lds622 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %81), !dbg !9
  %82 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds622, i32 %dma.buffer.offset620, <4 x i32> %dma.buffer.rsrc624) #10, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %83 = extractelement <4 x i32> %76, i64 2, !dbg !15
  %84 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !15
  %dma.buffer.lds619 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !15
  %85 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds619, i32 %dma.buffer.offset617, <4 x i32> %dma.buffer.rsrc618) #10, !dbg !15
  %86 = extractelement <4 x i32> %76, i64 3, !dbg !15
  %87 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %86), !dbg !15
  %88 = add nuw nsw i64 %address.local154, %.idx1719, !dbg !15
  %89 = shl nuw nsw i64 %88, 1, !dbg !15
  %90 = add i64 %89, %.idx74, !dbg !15
  %dma.buffer.offset = trunc i64 %90 to i32, !dbg !15
  %dma.buffer.lds = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %87), !dbg !15
  %91 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc618) #10, !dbg !15
  %address.local164 = lshr i32 %address.thread, 2, !dbg !10
  %address.local166 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local169 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local172 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %92 = insertelement <4 x i32> poison, i32 %address.local169, i64 0, !dbg !10
  %93 = insertelement <4 x i32> %92, i32 %address.local166, i64 1, !dbg !10
  %94 = insertelement <4 x i32> %93, i32 %address.local164, i64 2, !dbg !10
  %95 = insertelement <4 x i32> %94, i32 %address.local172, i64 3, !dbg !10
  %96 = and <4 x i32> %95, <i32 120, i32 384, i32 4, i32 1024>, !dbg !10
  %97 = tail call i32 @llvm.vector.reduce.or.v4i32(<4 x i32> %96), !dbg !10
  %address.local175 = zext nneg i32 %97 to i64, !dbg !10
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !10
  %98 = load <4 x half>, ptr addrspace(3) %address.local176, align 8, !dbg !10
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local176, i64 4096, !dbg !10
  %99 = load <4 x half>, ptr addrspace(3) %address.local178, align 8, !dbg !10
  %address.local180 = getelementptr i8, ptr addrspace(3) %address.local176, i64 1024, !dbg !10
  %100 = load <4 x half>, ptr addrspace(3) %address.local180, align 8, !dbg !10
  %101 = fmul <4 x half> %98, splat (half 1.275630e-01), !dbg !10
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local176, i64 5120, !dbg !10
  %102 = load <4 x half>, ptr addrspace(3) %address.local182, align 8, !dbg !10
  %103 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !10
  %104 = fmul <4 x half> %100, splat (half 1.275630e-01), !dbg !10
  %105 = fmul <4 x half> %102, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local186 = and i32 %address.local166, 24, !dbg !17
  %address.local191 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local192 = and i32 %address.local191, 384, !dbg !17
  %106 = or disjoint i32 %address.local192, %address.local186, !dbg !17
  %107 = extractelement <4 x i32> %96, i64 2, !dbg !17
  %108 = or disjoint i32 %106, %107, !dbg !17
  %109 = extractelement <4 x i32> %96, i64 3, !dbg !17
  %address.local193 = or disjoint i32 %108, %109, !dbg !17
  %address.local194 = zext nneg i32 %address.local193 to i64, !dbg !17
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local194, !dbg !17
  %110 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !17
  %111 = extractvalue [2 x <32 x float>] %54, 0, !dbg !18
  %112 = shufflevector <32 x float> %111, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %address.local209 = getelementptr i8, ptr addrspace(3) %address.local195, i64 4096, !dbg !17
  %113 = load <4 x half>, ptr addrspace(3) %address.local209, align 8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %114 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %110, <4 x float> %112) #9, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %115 = shufflevector <32 x float> %111, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %116 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %113, <4 x float> %114) #9, !dbg !17
  %117 = shufflevector <32 x float> %111, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !18
  %118 = shufflevector <32 x float> %111, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !18
  %119 = shufflevector <32 x float> %111, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !18
  %120 = shufflevector <32 x float> %111, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !18
  %121 = shufflevector <32 x float> %111, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !18
  %122 = shufflevector <32 x float> %111, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !18
  %123 = extractvalue [2 x <32 x float>] %54, 1, !dbg !18
  %124 = shufflevector <32 x float> %123, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %125 = shufflevector <32 x float> %123, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %126 = shufflevector <32 x float> %123, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !18
  %127 = shufflevector <32 x float> %123, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !18
  %128 = shufflevector <32 x float> %123, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !18
  %129 = shufflevector <32 x float> %123, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !18
  %130 = shufflevector <32 x float> %123, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !18
  %131 = shufflevector <32 x float> %123, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !18
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local138, !dbg !17
  %address.local212 = lshr i64 %address.local, 2, !dbg !17
  %address.local213 = mul nsw i64 %address.local212, -512, !dbg !17
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !17
  %address.local215 = shl nuw nsw i64 %address.local, 1, !dbg !17
  %address.local216 = and i64 %address.local215, 24, !dbg !17
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !17
  %address.local219 = and i64 %address.local212, 12, !dbg !10
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local219, !dbg !17
  %132 = and i64 %address.local, 32, !dbg !10
  %.not68 = icmp eq i64 %132, 0, !dbg !10
  %address.local223 = select i1 %.not68, i64 0, i64 1016, !dbg !10
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local223, !dbg !17
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 64, !dbg !17
  %133 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !17
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local219, !dbg !17
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local223, !dbg !17
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local213, !dbg !17
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local216, !dbg !17
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4160, !dbg !17
  %134 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !17
  %135 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %133, <4 x float> %115) #9, !dbg !17
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !17
  %136 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !17
  %137 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %134, <4 x float> %135) #9, !dbg !17
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4224, !dbg !17
  %138 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !17
  %139 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %136, <4 x float> %117) #9, !dbg !17
  %140 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %138, <4 x float> %139) #9, !dbg !17
  %address.local249 = getelementptr i8, ptr addrspace(3) %address.local224, i64 192, !dbg !17
  %141 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !17
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4288, !dbg !17
  %142 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !17
  %143 = shufflevector <4 x float> %140, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %144 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %141, <4 x float> %118) #9, !dbg !17
  %145 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %142, <4 x float> %144) #9, !dbg !17
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local195, i64 1024, !dbg !17
  %146 = load <4 x half>, ptr addrspace(3) %address.local257, align 8, !dbg !17
  %address.local259 = getelementptr i8, ptr addrspace(3) %address.local195, i64 5120, !dbg !17
  %147 = load <4 x half>, ptr addrspace(3) %address.local259, align 8, !dbg !17
  %148 = shufflevector <4 x float> %145, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %149 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %146, <4 x float> %119) #9, !dbg !17
  %150 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %147, <4 x float> %149) #9, !dbg !17
  %address.local261 = getelementptr i8, ptr addrspace(3) %address.local211, i64 1024, !dbg !17
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local213, !dbg !17
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) %address.local262, i64 %address.local216, !dbg !17
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local219, !dbg !17
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local223, !dbg !17
  %address.local266 = getelementptr i8, ptr addrspace(3) %address.local265, i64 64, !dbg !17
  %151 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !17
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local219, !dbg !17
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local223, !dbg !17
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local213, !dbg !17
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local216, !dbg !17
  %address.local272 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4160, !dbg !17
  %152 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !17
  %153 = shufflevector <4 x float> %150, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %154 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %151, <4 x float> %120) #9, !dbg !17
  %155 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %152, <4 x float> %154) #9, !dbg !17
  %address.local279 = getelementptr i8, ptr addrspace(3) %address.local265, i64 128, !dbg !17
  %156 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !17
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4224, !dbg !17
  %157 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !17
  %158 = shufflevector <4 x float> %155, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %159 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %156, <4 x float> %121) #9, !dbg !17
  %160 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %157, <4 x float> %159) #9, !dbg !17
  %address.local292 = getelementptr i8, ptr addrspace(3) %address.local265, i64 192, !dbg !17
  %161 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !17
  %address.local298 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4288, !dbg !17
  %162 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !17
  %163 = shufflevector <4 x float> %160, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %164 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %161, <4 x float> %122) #9, !dbg !17
  %165 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %162, <4 x float> %164) #9, !dbg !17
  %166 = shufflevector <4 x float> %165, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %167 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %110, <4 x float> %124) #9, !dbg !17
  %168 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %113, <4 x float> %167) #9, !dbg !17
  %169 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %133, <4 x float> %125) #9, !dbg !17
  %170 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %134, <4 x float> %169) #9, !dbg !17
  %171 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %136, <4 x float> %126) #9, !dbg !17
  %172 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %138, <4 x float> %171) #9, !dbg !17
  %173 = shufflevector <4 x float> %172, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %174 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %141, <4 x float> %127) #9, !dbg !17
  %175 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %142, <4 x float> %174) #9, !dbg !17
  %176 = shufflevector <4 x float> %175, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %177 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %146, <4 x float> %128) #9, !dbg !17
  %178 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %147, <4 x float> %177) #9, !dbg !17
  %179 = shufflevector <4 x float> %178, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %180 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %151, <4 x float> %129) #9, !dbg !17
  %181 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %152, <4 x float> %180) #9, !dbg !17
  %182 = shufflevector <4 x float> %181, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %183 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %156, <4 x float> %130) #9, !dbg !17
  %184 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %157, <4 x float> %183) #9, !dbg !17
  %185 = shufflevector <4 x float> %184, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %186 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %161, <4 x float> %131) #9, !dbg !17
  %187 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %162, <4 x float> %186) #9, !dbg !17
  %188 = shufflevector <4 x float> %187, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %189 = shufflevector <4 x float> %116, <4 x float> %137, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %190 = shufflevector <32 x float> %189, <32 x float> %143, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %191 = shufflevector <32 x float> %190, <32 x float> %148, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %192 = shufflevector <32 x float> %191, <32 x float> %153, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %193 = shufflevector <32 x float> %192, <32 x float> %158, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %194 = shufflevector <32 x float> %193, <32 x float> %163, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %195 = shufflevector <32 x float> %194, <32 x float> %166, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !16
  %196 = insertvalue [2 x <32 x float>] poison, <32 x float> %195, 0, !dbg !16
  %197 = shufflevector <4 x float> %168, <4 x float> %170, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %198 = shufflevector <32 x float> %197, <32 x float> %173, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %199 = shufflevector <32 x float> %198, <32 x float> %176, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %200 = shufflevector <32 x float> %199, <32 x float> %179, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %201 = shufflevector <32 x float> %200, <32 x float> %182, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %202 = shufflevector <32 x float> %201, <32 x float> %185, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %203 = shufflevector <32 x float> %202, <32 x float> %188, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !16
  %204 = insertvalue [2 x <32 x float>] %196, <32 x float> %203, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %205 = add nuw nsw i64 %55, 1, !dbg !16
  %206 = icmp samesign ult i64 %55, 3, !dbg !16
  br i1 %206, label %53, label %207, !dbg !16

207:                                              ; preds = %53
  %208 = add nsw i64 %50, %8, !dbg !19
  %209 = shl i64 %208, 20, !dbg !19
  %210 = and i64 %209, 32505856, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread299 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #9, !dbg !19, !range !11, !frisk.register_lifetimes !5
  %address.local300 = zext nneg i32 %address.thread299 to i64, !dbg !10
  %address.local301 = shl nuw nsw i64 %address.local300, 3, !dbg !9
  %address.local302 = and i64 %address.local301, 1536, !dbg !9
  %address.local303 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local302, !dbg !9
  %address.local304 = ptrtoint ptr addrspace(3) %address.local303 to i64, !dbg !9
  %address.local305 = trunc i64 %address.local304 to i32, !dbg !9
  %211 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local305), !dbg !19
  %address.local306 = shl nuw nsw i64 %address.local300, 6, !dbg !19
  %address.local307 = and i64 %address.local306, 16128, !dbg !19
  %212 = add i64 %210, %.idx74, !dbg !19
  %address.local309 = shl nuw nsw i64 %address.local300, 4, !dbg !19
  %address.local311.idx = and i64 %address.local309, 48, !dbg !19
  %213 = or disjoint i64 %212, %address.local307, !dbg !19
  %214 = or disjoint i64 %213, %address.local311.idx, !dbg !19
  %dma.buffer.offset647 = trunc i64 %214 to i32, !dbg !19
  %dma.buffer.lds649 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %211), !dbg !19
  %215 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds649, i32 %dma.buffer.offset647, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %address.local312 = or disjoint i64 %address.local300, 256, !dbg !9
  %address.local313 = shl nuw nsw i64 %address.local312, 3, !dbg !9
  %address.local314 = and i64 %address.local313, 3584, !dbg !9
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local314, !dbg !9
  %address.local316 = ptrtoint ptr addrspace(3) %address.local315 to i64, !dbg !9
  %address.local317 = trunc i64 %address.local316 to i32, !dbg !9
  %216 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local317), !dbg !19
  %address.local318 = shl nuw nsw i64 %address.local312, 6, !dbg !19
  %address.local319 = and i64 %address.local318, 32512, !dbg !19
  %217 = or disjoint i64 %212, %address.local319, !dbg !19
  %218 = or disjoint i64 %217, %address.local311.idx, !dbg !19
  %dma.buffer.offset644 = trunc i64 %218 to i32, !dbg !19
  %dma.buffer.lds646 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %216), !dbg !19
  %219 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds646, i32 %dma.buffer.offset644, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %220 = or disjoint i64 %49, 127, !dbg !20
  %.not80 = icmp ult i64 %9, %220, !dbg !20
  br i1 %.not80, label %254, label %221, !dbg !20

221:                                              ; preds = %207
  %222 = shufflevector <4 x float> %116, <4 x float> %137, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %223 = shufflevector <4 x float> %140, <4 x float> poison, <64 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %224 = shufflevector <64 x float> %222, <64 x float> %223, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 64, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %225 = shufflevector <4 x float> %140, <4 x float> poison, <64 x i32> <i32 poison, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %226 = shufflevector <64 x float> %224, <64 x float> %225, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %227 = shufflevector <4 x float> %145, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %228 = shufflevector <64 x float> %226, <64 x float> %227, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %229 = shufflevector <4 x float> %150, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %230 = shufflevector <64 x float> %228, <64 x float> %229, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %231 = shufflevector <4 x float> %155, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %232 = shufflevector <64 x float> %230, <64 x float> %231, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %233 = shufflevector <4 x float> %160, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %234 = shufflevector <64 x float> %232, <64 x float> %233, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %235 = shufflevector <4 x float> %165, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %236 = shufflevector <64 x float> %234, <64 x float> %235, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %237 = shufflevector <4 x float> %168, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %238 = shufflevector <64 x float> %236, <64 x float> %237, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %239 = shufflevector <4 x float> %170, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %240 = shufflevector <64 x float> %238, <64 x float> %239, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %241 = shufflevector <4 x float> %172, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %242 = shufflevector <64 x float> %240, <64 x float> %241, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %243 = shufflevector <4 x float> %175, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %244 = shufflevector <64 x float> %242, <64 x float> %243, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %245 = shufflevector <4 x float> %178, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %246 = shufflevector <64 x float> %244, <64 x float> %245, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %247 = shufflevector <4 x float> %181, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %248 = shufflevector <64 x float> %246, <64 x float> %247, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %249 = shufflevector <4 x float> %184, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %250 = shufflevector <64 x float> %248, <64 x float> %249, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %251 = shufflevector <4 x float> %187, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %252 = shufflevector <64 x float> %250, <64 x float> %251, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 64, i32 65, i32 66, i32 67>, !dbg !20
  %253 = fadd <64 x float> %252, zeroinitializer, !dbg !20
  br label %314, !dbg !20

254:                                              ; preds = %207
  %255 = or disjoint i64 %49, %36, !dbg !20
  %256 = insertelement <16 x i64> poison, i64 %255, i64 0, !dbg !20
  %257 = insertelement <8 x i64> poison, i64 %255, i64 0, !dbg !20
  %258 = or disjoint i64 %255, 73, !dbg !20
  %259 = or disjoint i64 %255, 74, !dbg !20
  %260 = or disjoint i64 %255, 75, !dbg !20
  %261 = or disjoint i64 %255, 76, !dbg !20
  %262 = or disjoint i64 %255, 77, !dbg !20
  %263 = or disjoint i64 %255, 78, !dbg !20
  %264 = or disjoint i64 %255, 79, !dbg !20
  %265 = insertelement <64 x i64> poison, i64 %255, i64 0, !dbg !14
  %266 = shufflevector <16 x i64> %256, <16 x i64> poison, <64 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %267 = or disjoint <64 x i64> %266, <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 64, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison>, !dbg !14
  %268 = shufflevector <64 x i64> %265, <64 x i64> %267, <64 x i32> <i32 0, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %269 = shufflevector <8 x i64> %257, <8 x i64> poison, <64 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %270 = or disjoint <64 x i64> %269, <i64 65, i64 66, i64 67, i64 68, i64 69, i64 70, i64 71, i64 72, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison>, !dbg !14
  %271 = shufflevector <64 x i64> %268, <64 x i64> %270, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %272 = insertelement <64 x i64> %271, i64 %258, i64 25, !dbg !14
  %273 = insertelement <64 x i64> %272, i64 %259, i64 26, !dbg !14
  %274 = insertelement <64 x i64> %273, i64 %260, i64 27, !dbg !14
  %275 = insertelement <64 x i64> %274, i64 %261, i64 28, !dbg !14
  %276 = insertelement <64 x i64> %275, i64 %262, i64 29, !dbg !14
  %277 = insertelement <64 x i64> %276, i64 %263, i64 30, !dbg !14
  %278 = insertelement <64 x i64> %277, i64 %264, i64 31, !dbg !14
  %279 = shufflevector <64 x i64> %278, <64 x i64> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !14
  %280 = icmp ugt <64 x i64> %41, %279, !dbg !14
  %281 = select <64 x i1> %280, <64 x float> zeroinitializer, <64 x float> splat (float -inf), !dbg !21
  %282 = shufflevector <4 x float> %116, <4 x float> %137, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %283 = shufflevector <4 x float> %140, <4 x float> poison, <64 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %284 = shufflevector <64 x float> %282, <64 x float> %283, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 64, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %285 = shufflevector <4 x float> %140, <4 x float> poison, <64 x i32> <i32 poison, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %286 = shufflevector <64 x float> %284, <64 x float> %285, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %287 = shufflevector <4 x float> %145, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %288 = shufflevector <64 x float> %286, <64 x float> %287, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %289 = shufflevector <4 x float> %150, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %290 = shufflevector <64 x float> %288, <64 x float> %289, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %291 = shufflevector <4 x float> %155, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %292 = shufflevector <64 x float> %290, <64 x float> %291, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %293 = shufflevector <4 x float> %160, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %294 = shufflevector <64 x float> %292, <64 x float> %293, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %295 = shufflevector <4 x float> %165, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %296 = shufflevector <64 x float> %294, <64 x float> %295, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %297 = shufflevector <4 x float> %168, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %298 = shufflevector <64 x float> %296, <64 x float> %297, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %299 = shufflevector <4 x float> %170, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %300 = shufflevector <64 x float> %298, <64 x float> %299, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %301 = shufflevector <4 x float> %172, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %302 = shufflevector <64 x float> %300, <64 x float> %301, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %303 = shufflevector <4 x float> %175, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %304 = shufflevector <64 x float> %302, <64 x float> %303, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %305 = shufflevector <4 x float> %178, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %306 = shufflevector <64 x float> %304, <64 x float> %305, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %307 = shufflevector <4 x float> %181, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %308 = shufflevector <64 x float> %306, <64 x float> %307, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %309 = shufflevector <4 x float> %184, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %310 = shufflevector <64 x float> %308, <64 x float> %309, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %311 = shufflevector <4 x float> %187, <4 x float> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %312 = shufflevector <64 x float> %310, <64 x float> %311, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 64, i32 65, i32 66, i32 67>, !dbg !20
  %313 = fadd <64 x float> %281, %312, !dbg !20
  br label %314, !dbg !20

314:                                              ; preds = %221, %254
  %315 = phi <64 x float> [ %253, %221 ], [ %313, %254 ]
  %316 = extractelement <64 x float> %315, i64 0, !dbg !22
  %317 = tail call float @llvm.amdgcn.exp2.f32(float %316), !dbg !22
  %318 = insertelement <4 x float> poison, float %317, i64 0, !dbg !22
  %319 = extractelement <64 x float> %315, i64 1, !dbg !22
  %320 = tail call float @llvm.amdgcn.exp2.f32(float %319), !dbg !22
  %321 = fadd float %317, %320, !dbg !23
  %322 = insertelement <4 x float> %318, float %320, i64 1, !dbg !22
  %323 = extractelement <64 x float> %315, i64 2, !dbg !22
  %324 = tail call float @llvm.amdgcn.exp2.f32(float %323), !dbg !22
  %325 = fadd float %324, %321, !dbg !23
  %326 = insertelement <4 x float> %322, float %324, i64 2, !dbg !22
  %327 = extractelement <64 x float> %315, i64 3, !dbg !22
  %328 = tail call float @llvm.amdgcn.exp2.f32(float %327), !dbg !22
  %329 = fadd float %328, %325, !dbg !23
  %330 = insertelement <4 x float> %326, float %328, i64 3, !dbg !22
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %330) #9, !dbg !24, !frisk.register_lifetimes !5
  %331 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !24
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %331) #9, !dbg !22, !frisk.register_lifetimes !5
  %332 = extractelement <64 x float> %315, i64 4, !dbg !22
  %333 = tail call float @llvm.amdgcn.exp2.f32(float %332), !dbg !22
  %334 = fadd float %333, %329, !dbg !23
  %335 = insertelement <4 x float> poison, float %333, i64 0, !dbg !22
  %336 = extractelement <64 x float> %315, i64 5, !dbg !22
  %337 = tail call float @llvm.amdgcn.exp2.f32(float %336), !dbg !22
  %338 = fadd float %337, %334, !dbg !23
  %339 = insertelement <4 x float> %335, float %337, i64 1, !dbg !22
  %340 = extractelement <64 x float> %315, i64 6, !dbg !22
  %341 = tail call float @llvm.amdgcn.exp2.f32(float %340), !dbg !22
  %342 = fadd float %341, %338, !dbg !23
  %343 = insertelement <4 x float> %339, float %341, i64 2, !dbg !22
  %344 = extractelement <64 x float> %315, i64 7, !dbg !22
  %345 = tail call float @llvm.amdgcn.exp2.f32(float %344), !dbg !22
  %346 = fadd float %345, %342, !dbg !23
  %347 = insertelement <4 x float> %343, float %345, i64 3, !dbg !22
  %math.result103 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %347) #9, !dbg !24, !frisk.register_lifetimes !5
  %348 = fptrunc <4 x float> %math.result103 to <4 x half>, !dbg !24
  %math.packed104 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %348) #9, !dbg !22, !frisk.register_lifetimes !5
  %349 = extractelement <64 x float> %315, i64 8, !dbg !22
  %350 = tail call float @llvm.amdgcn.exp2.f32(float %349), !dbg !22
  %351 = fadd float %350, %346, !dbg !23
  %352 = insertelement <4 x float> poison, float %350, i64 0, !dbg !22
  %353 = extractelement <64 x float> %315, i64 9, !dbg !22
  %354 = tail call float @llvm.amdgcn.exp2.f32(float %353), !dbg !22
  %355 = fadd float %354, %351, !dbg !23
  %356 = insertelement <4 x float> %352, float %354, i64 1, !dbg !22
  %357 = extractelement <64 x float> %315, i64 10, !dbg !22
  %358 = tail call float @llvm.amdgcn.exp2.f32(float %357), !dbg !22
  %359 = fadd float %358, %355, !dbg !23
  %360 = insertelement <4 x float> %356, float %358, i64 2, !dbg !22
  %361 = extractelement <64 x float> %315, i64 11, !dbg !22
  %362 = tail call float @llvm.amdgcn.exp2.f32(float %361), !dbg !22
  %363 = fadd float %362, %359, !dbg !23
  %364 = insertelement <4 x float> %360, float %362, i64 3, !dbg !22
  %math.result105 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %364) #9, !dbg !24, !frisk.register_lifetimes !5
  %365 = fptrunc <4 x float> %math.result105 to <4 x half>, !dbg !24
  %math.packed106 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %365) #9, !dbg !22, !frisk.register_lifetimes !5
  %366 = extractelement <64 x float> %315, i64 12, !dbg !22
  %367 = tail call float @llvm.amdgcn.exp2.f32(float %366), !dbg !22
  %368 = fadd float %367, %363, !dbg !23
  %369 = insertelement <4 x float> poison, float %367, i64 0, !dbg !22
  %370 = extractelement <64 x float> %315, i64 13, !dbg !22
  %371 = tail call float @llvm.amdgcn.exp2.f32(float %370), !dbg !22
  %372 = fadd float %371, %368, !dbg !23
  %373 = insertelement <4 x float> %369, float %371, i64 1, !dbg !22
  %374 = extractelement <64 x float> %315, i64 14, !dbg !22
  %375 = tail call float @llvm.amdgcn.exp2.f32(float %374), !dbg !22
  %376 = fadd float %375, %372, !dbg !23
  %377 = insertelement <4 x float> %373, float %375, i64 2, !dbg !22
  %378 = extractelement <64 x float> %315, i64 15, !dbg !22
  %379 = tail call float @llvm.amdgcn.exp2.f32(float %378), !dbg !22
  %380 = fadd float %379, %376, !dbg !23
  %381 = insertelement <4 x float> %377, float %379, i64 3, !dbg !22
  %math.result107 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %381) #9, !dbg !24, !frisk.register_lifetimes !5
  %382 = fptrunc <4 x float> %math.result107 to <4 x half>, !dbg !24
  %math.packed108 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %382) #9, !dbg !22, !frisk.register_lifetimes !5
  %383 = extractelement <64 x float> %315, i64 16, !dbg !22
  %384 = tail call float @llvm.amdgcn.exp2.f32(float %383), !dbg !22
  %385 = fadd float %384, %380, !dbg !23
  %386 = insertelement <4 x float> poison, float %384, i64 0, !dbg !22
  %387 = extractelement <64 x float> %315, i64 17, !dbg !22
  %388 = tail call float @llvm.amdgcn.exp2.f32(float %387), !dbg !22
  %389 = fadd float %388, %385, !dbg !23
  %390 = insertelement <4 x float> %386, float %388, i64 1, !dbg !22
  %391 = extractelement <64 x float> %315, i64 18, !dbg !22
  %392 = tail call float @llvm.amdgcn.exp2.f32(float %391), !dbg !22
  %393 = fadd float %392, %389, !dbg !23
  %394 = insertelement <4 x float> %390, float %392, i64 2, !dbg !22
  %395 = extractelement <64 x float> %315, i64 19, !dbg !22
  %396 = tail call float @llvm.amdgcn.exp2.f32(float %395), !dbg !22
  %397 = fadd float %396, %393, !dbg !23
  %398 = insertelement <4 x float> %394, float %396, i64 3, !dbg !22
  %math.result109 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %398) #9, !dbg !24, !frisk.register_lifetimes !5
  %399 = fptrunc <4 x float> %math.result109 to <4 x half>, !dbg !24
  %math.packed110 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %399) #9, !dbg !22, !frisk.register_lifetimes !5
  %400 = extractelement <64 x float> %315, i64 20, !dbg !22
  %401 = tail call float @llvm.amdgcn.exp2.f32(float %400), !dbg !22
  %402 = fadd float %401, %397, !dbg !23
  %403 = insertelement <4 x float> poison, float %401, i64 0, !dbg !22
  %404 = extractelement <64 x float> %315, i64 21, !dbg !22
  %405 = tail call float @llvm.amdgcn.exp2.f32(float %404), !dbg !22
  %406 = fadd float %405, %402, !dbg !23
  %407 = insertelement <4 x float> %403, float %405, i64 1, !dbg !22
  %408 = extractelement <64 x float> %315, i64 22, !dbg !22
  %409 = tail call float @llvm.amdgcn.exp2.f32(float %408), !dbg !22
  %410 = fadd float %409, %406, !dbg !23
  %411 = insertelement <4 x float> %407, float %409, i64 2, !dbg !22
  %412 = extractelement <64 x float> %315, i64 23, !dbg !22
  %413 = tail call float @llvm.amdgcn.exp2.f32(float %412), !dbg !22
  %414 = fadd float %413, %410, !dbg !23
  %415 = insertelement <4 x float> %411, float %413, i64 3, !dbg !22
  %math.result111 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %415) #9, !dbg !24, !frisk.register_lifetimes !5
  %416 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !24
  %math.packed112 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %416) #9, !dbg !22, !frisk.register_lifetimes !5
  %417 = extractelement <64 x float> %315, i64 24, !dbg !22
  %418 = tail call float @llvm.amdgcn.exp2.f32(float %417), !dbg !22
  %419 = fadd float %418, %414, !dbg !23
  %420 = insertelement <4 x float> poison, float %418, i64 0, !dbg !22
  %421 = extractelement <64 x float> %315, i64 25, !dbg !22
  %422 = tail call float @llvm.amdgcn.exp2.f32(float %421), !dbg !22
  %423 = fadd float %422, %419, !dbg !23
  %424 = insertelement <4 x float> %420, float %422, i64 1, !dbg !22
  %425 = extractelement <64 x float> %315, i64 26, !dbg !22
  %426 = tail call float @llvm.amdgcn.exp2.f32(float %425), !dbg !22
  %427 = fadd float %426, %423, !dbg !23
  %428 = insertelement <4 x float> %424, float %426, i64 2, !dbg !22
  %429 = extractelement <64 x float> %315, i64 27, !dbg !22
  %430 = tail call float @llvm.amdgcn.exp2.f32(float %429), !dbg !22
  %431 = fadd float %430, %427, !dbg !23
  %432 = insertelement <4 x float> %428, float %430, i64 3, !dbg !22
  %math.result113 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %432) #9, !dbg !24, !frisk.register_lifetimes !5
  %433 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !24
  %math.packed114 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %433) #9, !dbg !22, !frisk.register_lifetimes !5
  %434 = extractelement <64 x float> %315, i64 28, !dbg !22
  %435 = tail call float @llvm.amdgcn.exp2.f32(float %434), !dbg !22
  %436 = fadd float %435, %431, !dbg !23
  %437 = insertelement <4 x float> poison, float %435, i64 0, !dbg !22
  %438 = extractelement <64 x float> %315, i64 29, !dbg !22
  %439 = tail call float @llvm.amdgcn.exp2.f32(float %438), !dbg !22
  %440 = fadd float %439, %436, !dbg !23
  %441 = insertelement <4 x float> %437, float %439, i64 1, !dbg !22
  %442 = extractelement <64 x float> %315, i64 30, !dbg !22
  %443 = tail call float @llvm.amdgcn.exp2.f32(float %442), !dbg !22
  %444 = fadd float %443, %440, !dbg !23
  %445 = insertelement <4 x float> %441, float %443, i64 2, !dbg !22
  %446 = extractelement <64 x float> %315, i64 31, !dbg !22
  %447 = tail call float @llvm.amdgcn.exp2.f32(float %446), !dbg !22
  %448 = fadd float %447, %444, !dbg !23
  %449 = insertelement <4 x float> %445, float %447, i64 3, !dbg !22
  %math.result115 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %449) #9, !dbg !24, !frisk.register_lifetimes !5
  %450 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !24
  %math.packed116 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %450) #9, !dbg !22, !frisk.register_lifetimes !5
  %451 = extractelement <64 x float> %315, i64 32, !dbg !22
  %452 = tail call float @llvm.amdgcn.exp2.f32(float %451), !dbg !22
  %453 = insertelement <4 x float> poison, float %452, i64 0, !dbg !22
  %454 = extractelement <64 x float> %315, i64 33, !dbg !22
  %455 = tail call float @llvm.amdgcn.exp2.f32(float %454), !dbg !22
  %456 = fadd float %452, %455, !dbg !23
  %457 = insertelement <4 x float> %453, float %455, i64 1, !dbg !22
  %458 = extractelement <64 x float> %315, i64 34, !dbg !22
  %459 = tail call float @llvm.amdgcn.exp2.f32(float %458), !dbg !22
  %460 = fadd float %459, %456, !dbg !23
  %461 = insertelement <4 x float> %457, float %459, i64 2, !dbg !22
  %462 = extractelement <64 x float> %315, i64 35, !dbg !22
  %463 = tail call float @llvm.amdgcn.exp2.f32(float %462), !dbg !22
  %464 = fadd float %463, %460, !dbg !23
  %465 = insertelement <4 x float> %461, float %463, i64 3, !dbg !22
  %math.result117 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %465) #9, !dbg !24, !frisk.register_lifetimes !5
  %466 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !24
  %math.packed118 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %466) #9, !dbg !22, !frisk.register_lifetimes !5
  %467 = extractelement <64 x float> %315, i64 36, !dbg !22
  %468 = tail call float @llvm.amdgcn.exp2.f32(float %467), !dbg !22
  %469 = fadd float %468, %464, !dbg !23
  %470 = insertelement <4 x float> poison, float %468, i64 0, !dbg !22
  %471 = extractelement <64 x float> %315, i64 37, !dbg !22
  %472 = tail call float @llvm.amdgcn.exp2.f32(float %471), !dbg !22
  %473 = fadd float %472, %469, !dbg !23
  %474 = insertelement <4 x float> %470, float %472, i64 1, !dbg !22
  %475 = extractelement <64 x float> %315, i64 38, !dbg !22
  %476 = tail call float @llvm.amdgcn.exp2.f32(float %475), !dbg !22
  %477 = fadd float %476, %473, !dbg !23
  %478 = insertelement <4 x float> %474, float %476, i64 2, !dbg !22
  %479 = extractelement <64 x float> %315, i64 39, !dbg !22
  %480 = tail call float @llvm.amdgcn.exp2.f32(float %479), !dbg !22
  %481 = fadd float %480, %477, !dbg !23
  %482 = insertelement <4 x float> %478, float %480, i64 3, !dbg !22
  %math.result119 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %482) #9, !dbg !24, !frisk.register_lifetimes !5
  %483 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !24
  %math.packed120 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %483) #9, !dbg !22, !frisk.register_lifetimes !5
  %484 = extractelement <64 x float> %315, i64 40, !dbg !22
  %485 = tail call float @llvm.amdgcn.exp2.f32(float %484), !dbg !22
  %486 = fadd float %485, %481, !dbg !23
  %487 = insertelement <4 x float> poison, float %485, i64 0, !dbg !22
  %488 = extractelement <64 x float> %315, i64 41, !dbg !22
  %489 = tail call float @llvm.amdgcn.exp2.f32(float %488), !dbg !22
  %490 = fadd float %489, %486, !dbg !23
  %491 = insertelement <4 x float> %487, float %489, i64 1, !dbg !22
  %492 = extractelement <64 x float> %315, i64 42, !dbg !22
  %493 = tail call float @llvm.amdgcn.exp2.f32(float %492), !dbg !22
  %494 = fadd float %493, %490, !dbg !23
  %495 = insertelement <4 x float> %491, float %493, i64 2, !dbg !22
  %496 = extractelement <64 x float> %315, i64 43, !dbg !22
  %497 = tail call float @llvm.amdgcn.exp2.f32(float %496), !dbg !22
  %498 = fadd float %497, %494, !dbg !23
  %499 = insertelement <4 x float> %495, float %497, i64 3, !dbg !22
  %math.result121 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %499) #9, !dbg !24, !frisk.register_lifetimes !5
  %500 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !24
  %math.packed122 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %500) #9, !dbg !22, !frisk.register_lifetimes !5
  %501 = extractelement <64 x float> %315, i64 44, !dbg !22
  %502 = tail call float @llvm.amdgcn.exp2.f32(float %501), !dbg !22
  %503 = fadd float %502, %498, !dbg !23
  %504 = insertelement <4 x float> poison, float %502, i64 0, !dbg !22
  %505 = extractelement <64 x float> %315, i64 45, !dbg !22
  %506 = tail call float @llvm.amdgcn.exp2.f32(float %505), !dbg !22
  %507 = fadd float %506, %503, !dbg !23
  %508 = insertelement <4 x float> %504, float %506, i64 1, !dbg !22
  %509 = extractelement <64 x float> %315, i64 46, !dbg !22
  %510 = tail call float @llvm.amdgcn.exp2.f32(float %509), !dbg !22
  %511 = fadd float %510, %507, !dbg !23
  %512 = insertelement <4 x float> %508, float %510, i64 2, !dbg !22
  %513 = extractelement <64 x float> %315, i64 47, !dbg !22
  %514 = tail call float @llvm.amdgcn.exp2.f32(float %513), !dbg !22
  %515 = fadd float %514, %511, !dbg !23
  %516 = insertelement <4 x float> %512, float %514, i64 3, !dbg !22
  %math.result123 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %516) #9, !dbg !24, !frisk.register_lifetimes !5
  %517 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !24
  %math.packed124 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %517) #9, !dbg !22, !frisk.register_lifetimes !5
  %518 = extractelement <64 x float> %315, i64 48, !dbg !22
  %519 = tail call float @llvm.amdgcn.exp2.f32(float %518), !dbg !22
  %520 = fadd float %519, %515, !dbg !23
  %521 = insertelement <4 x float> poison, float %519, i64 0, !dbg !22
  %522 = extractelement <64 x float> %315, i64 49, !dbg !22
  %523 = tail call float @llvm.amdgcn.exp2.f32(float %522), !dbg !22
  %524 = fadd float %523, %520, !dbg !23
  %525 = insertelement <4 x float> %521, float %523, i64 1, !dbg !22
  %526 = extractelement <64 x float> %315, i64 50, !dbg !22
  %527 = tail call float @llvm.amdgcn.exp2.f32(float %526), !dbg !22
  %528 = fadd float %527, %524, !dbg !23
  %529 = insertelement <4 x float> %525, float %527, i64 2, !dbg !22
  %530 = extractelement <64 x float> %315, i64 51, !dbg !22
  %531 = tail call float @llvm.amdgcn.exp2.f32(float %530), !dbg !22
  %532 = fadd float %531, %528, !dbg !23
  %533 = insertelement <4 x float> %529, float %531, i64 3, !dbg !22
  %math.result125 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %533) #9, !dbg !24, !frisk.register_lifetimes !5
  %534 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !24
  %math.packed126 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %534) #9, !dbg !22, !frisk.register_lifetimes !5
  %535 = extractelement <64 x float> %315, i64 52, !dbg !22
  %536 = tail call float @llvm.amdgcn.exp2.f32(float %535), !dbg !22
  %537 = fadd float %536, %532, !dbg !23
  %538 = insertelement <4 x float> poison, float %536, i64 0, !dbg !22
  %539 = extractelement <64 x float> %315, i64 53, !dbg !22
  %540 = tail call float @llvm.amdgcn.exp2.f32(float %539), !dbg !22
  %541 = fadd float %540, %537, !dbg !23
  %542 = insertelement <4 x float> %538, float %540, i64 1, !dbg !22
  %543 = extractelement <64 x float> %315, i64 54, !dbg !22
  %544 = tail call float @llvm.amdgcn.exp2.f32(float %543), !dbg !22
  %545 = fadd float %544, %541, !dbg !23
  %546 = insertelement <4 x float> %542, float %544, i64 2, !dbg !22
  %547 = extractelement <64 x float> %315, i64 55, !dbg !22
  %548 = tail call float @llvm.amdgcn.exp2.f32(float %547), !dbg !22
  %549 = fadd float %548, %545, !dbg !23
  %550 = insertelement <4 x float> %546, float %548, i64 3, !dbg !22
  %math.result127 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %550) #9, !dbg !24, !frisk.register_lifetimes !5
  %551 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !24
  %math.packed128 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %551) #9, !dbg !22, !frisk.register_lifetimes !5
  %552 = extractelement <64 x float> %315, i64 56, !dbg !22
  %553 = tail call float @llvm.amdgcn.exp2.f32(float %552), !dbg !22
  %554 = fadd float %553, %549, !dbg !23
  %555 = insertelement <4 x float> poison, float %553, i64 0, !dbg !22
  %556 = extractelement <64 x float> %315, i64 57, !dbg !22
  %557 = tail call float @llvm.amdgcn.exp2.f32(float %556), !dbg !22
  %558 = fadd float %557, %554, !dbg !23
  %559 = insertelement <4 x float> %555, float %557, i64 1, !dbg !22
  %560 = extractelement <64 x float> %315, i64 58, !dbg !22
  %561 = tail call float @llvm.amdgcn.exp2.f32(float %560), !dbg !22
  %562 = fadd float %561, %558, !dbg !23
  %563 = insertelement <4 x float> %559, float %561, i64 2, !dbg !22
  %564 = extractelement <64 x float> %315, i64 59, !dbg !22
  %565 = tail call float @llvm.amdgcn.exp2.f32(float %564), !dbg !22
  %566 = fadd float %565, %562, !dbg !23
  %567 = insertelement <4 x float> %563, float %565, i64 3, !dbg !22
  %math.result129 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %567) #9, !dbg !24, !frisk.register_lifetimes !5
  %568 = fptrunc <4 x float> %math.result129 to <4 x half>, !dbg !24
  %math.packed130 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %568) #9, !dbg !22, !frisk.register_lifetimes !5
  %569 = extractelement <64 x float> %315, i64 60, !dbg !22
  %570 = tail call float @llvm.amdgcn.exp2.f32(float %569), !dbg !22
  %571 = fadd float %570, %566, !dbg !23
  %572 = insertelement <4 x float> poison, float %570, i64 0, !dbg !22
  %573 = extractelement <64 x float> %315, i64 61, !dbg !22
  %574 = tail call float @llvm.amdgcn.exp2.f32(float %573), !dbg !22
  %575 = fadd float %574, %571, !dbg !23
  %576 = insertelement <4 x float> %572, float %574, i64 1, !dbg !22
  %577 = extractelement <64 x float> %315, i64 62, !dbg !22
  %578 = tail call float @llvm.amdgcn.exp2.f32(float %577), !dbg !22
  %579 = fadd float %578, %575, !dbg !23
  %580 = insertelement <4 x float> %576, float %578, i64 2, !dbg !22
  %581 = extractelement <64 x float> %315, i64 63, !dbg !22
  %582 = tail call float @llvm.amdgcn.exp2.f32(float %581), !dbg !22
  %583 = fadd float %582, %579, !dbg !23
  %584 = insertelement <4 x float> %580, float %582, i64 3, !dbg !22
  %math.result131 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %584) #9, !dbg !24, !frisk.register_lifetimes !5
  %585 = fptrunc <4 x float> %math.result131 to <4 x half>, !dbg !24
  %math.packed132 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %585) #9, !dbg !23, !frisk.register_lifetimes !5
  %586 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !23
  %587 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %586), !dbg !23
  %588 = add nuw nsw i32 %587, 64, !dbg !23
  %589 = xor i32 %587, 16, !dbg !23
  %590 = and i32 %588, 192, !dbg !23
  %591 = icmp samesign ult i32 %589, %590, !dbg !23
  %592 = select i1 %591, i32 %589, i32 %587, !dbg !23
  %593 = shl nuw nsw i32 %592, 2, !dbg !23
  %594 = bitcast float %448 to i32, !dbg !23
  %595 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %593, i32 %594), !dbg !23
  %596 = bitcast i32 %595 to float, !dbg !23
  %597 = fadd float %448, %596, !dbg !23
  %598 = xor i32 %587, 32, !dbg !23
  %599 = icmp samesign ult i32 %598, %590, !dbg !23
  %600 = select i1 %599, i32 %598, i32 %587, !dbg !23
  %601 = shl nuw nsw i32 %600, 2, !dbg !23
  %602 = bitcast float %597 to i32, !dbg !23
  %603 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %601, i32 %602), !dbg !23
  %604 = bitcast i32 %603 to float, !dbg !23
  %605 = fadd float %597, %604, !dbg !23
  %606 = extractvalue [2 x <1 x float>] %46, 0, !dbg !23
  %607 = extractelement <1 x float> %606, i64 0, !dbg !23
  %608 = fadd float %607, %605, !dbg !23
  %609 = insertelement <1 x float> poison, float %608, i64 0, !dbg !12
  %610 = insertvalue [2 x <1 x float>] poison, <1 x float> %609, 0, !dbg !12
  %611 = bitcast float %583 to i32, !dbg !23
  %612 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %593, i32 %611), !dbg !23
  %613 = bitcast i32 %612 to float, !dbg !23
  %614 = fadd float %583, %613, !dbg !23
  %615 = bitcast float %614 to i32, !dbg !23
  %616 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %601, i32 %615), !dbg !23
  %617 = bitcast i32 %616 to float, !dbg !23
  %618 = fadd float %614, %617, !dbg !23
  %619 = extractvalue [2 x <1 x float>] %46, 1, !dbg !23
  %620 = extractelement <1 x float> %619, i64 0, !dbg !23
  %621 = fadd float %620, %618, !dbg !23
  %622 = insertelement <1 x float> poison, float %621, i64 0, !dbg !12
  %623 = insertvalue [2 x <1 x float>] %610, <1 x float> %622, 1, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread324 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #9, !dbg !25, !range !11, !frisk.register_lifetimes !5
  %address.local325 = shl nuw nsw i32 %address.thread324, 3, !dbg !25
  %address.local326 = and i32 %address.local325, 120, !dbg !25
  %address.local327 = shl nuw nsw i32 %address.thread324, 5, !dbg !25
  %address.local328 = and i32 %address.local327, 1536, !dbg !25
  %address.local329 = or disjoint i32 %address.local326, %address.local328, !dbg !25
  %address.local330 = zext nneg i32 %address.local329 to i64, !dbg !25
  %address.local331 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !25
  %624 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #12, !dbg !25
  %625 = bitcast <4 x i32> %624 to <8 x half>, !dbg !25
  %address.local332 = zext nneg i32 %address.thread324 to i64, !dbg !10
  %address.local333 = and i64 %address.local332, 60, !dbg !25
  %address.local334 = add nuw nsw i64 %address.local333, %address.local332, !dbg !25
  %address.local336.idx = shl nuw nsw i64 %address.local334, 4, !dbg !25
  %address.local336 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local336.idx, !dbg !25
  %address.local337 = lshr i64 %address.local332, 4, !dbg !25
  %address.local338 = and i64 %address.local337, 3, !dbg !25
  %address.local340.idx = mul nuw nsw i64 %address.local338, 768, !dbg !25
  %address.local340 = getelementptr i8, ptr addrspace(3) %address.local336, i64 %address.local340.idx, !dbg !25
  %address.local341 = lshr i64 %address.local332, 2, !dbg !17
  %address.local343.idx = mul nsw i64 %address.local341, -64, !dbg !25
  %address.local343 = getelementptr i8, ptr addrspace(3) %address.local340, i64 %address.local343.idx, !dbg !25
  %address.local344 = getelementptr i8, ptr addrspace(3) %address.local343, i64 256, !dbg !25
  %626 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #12, !dbg !25
  %627 = bitcast <4 x i32> %626 to <8 x half>, !dbg !25
  %address.local348 = getelementptr i8, ptr addrspace(3) %address.local343, i64 512, !dbg !25
  %628 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #12, !dbg !25
  %629 = bitcast <4 x i32> %628 to <8 x half>, !dbg !25
  %address.local352 = getelementptr i8, ptr addrspace(3) %address.local343, i64 768, !dbg !25
  %630 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #12, !dbg !25
  %631 = bitcast <4 x i32> %630 to <8 x half>, !dbg !25
  %address.local354 = getelementptr i8, ptr addrspace(3) %address.local331, i64 4096, !dbg !25
  %632 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #12, !dbg !25
  %633 = bitcast <4 x i32> %632 to <8 x half>, !dbg !25
  %address.local358 = getelementptr i8, ptr addrspace(3) %address.local343, i64 4352, !dbg !25
  %634 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #12, !dbg !25
  %635 = bitcast <4 x i32> %634 to <8 x half>, !dbg !25
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local343, i64 4608, !dbg !25
  %636 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #12, !dbg !25
  %637 = bitcast <4 x i32> %636 to <8 x half>, !dbg !25
  %address.local366 = getelementptr i8, ptr addrspace(3) %address.local343, i64 4864, !dbg !25
  %638 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #12, !dbg !25
  %639 = bitcast <4 x i32> %638 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.local367 = shl nuw nsw i64 %address.local332, 3, !dbg !9
  %address.local368 = and i64 %address.local367, 1536, !dbg !9
  %address.local369 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local368, !dbg !9
  %address.local370 = ptrtoint ptr addrspace(3) %address.local369 to i64, !dbg !9
  %address.local371 = trunc i64 %address.local370 to i32, !dbg !9
  %640 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !19
  %address.local372 = shl nuw nsw i64 %address.local332, 5, !dbg !19
  %address.local373 = and i64 %address.local372, 8064, !dbg !19
  %address.local375 = and i64 %address.local367, 24, !dbg !19
  %address.local376 = or disjoint i64 %address.local373, %address.local375, !dbg !19
  %address.local377.idx = shl nuw nsw i64 %address.local376, 1, !dbg !19
  %641 = or disjoint i64 %address.local377.idx, %212, !dbg !19
  %642 = trunc i64 %641 to i32, !dbg !19
  %dma.buffer.offset641 = or disjoint i32 %642, 64, !dbg !19
  %dma.buffer.lds643 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %640), !dbg !19
  %643 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds643, i32 %dma.buffer.offset641, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %address.local378 = or disjoint i64 %address.local332, 256, !dbg !9
  %address.local379 = shl nuw nsw i64 %address.local378, 3, !dbg !9
  %address.local380 = and i64 %address.local379, 3584, !dbg !9
  %address.local381 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local380, !dbg !9
  %address.local382 = ptrtoint ptr addrspace(3) %address.local381 to i64, !dbg !9
  %address.local383 = trunc i64 %address.local382 to i32, !dbg !9
  %644 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !19
  %address.local384 = shl nuw nsw i64 %address.local378, 5, !dbg !19
  %address.local385 = and i64 %address.local384, 16256, !dbg !19
  %address.local388 = or disjoint i64 %address.local385, %address.local375, !dbg !19
  %address.local389.idx = shl nuw nsw i64 %address.local388, 1, !dbg !19
  %645 = or disjoint i64 %212, %address.local389.idx, !dbg !19
  %646 = trunc i64 %645 to i32, !dbg !19
  %dma.buffer.offset638 = or disjoint i32 %646, 64, !dbg !19
  %dma.buffer.lds640 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %644), !dbg !19
  %647 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds640, i32 %dma.buffer.offset638, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %648 = shufflevector <8 x half> %625, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %649 = extractvalue [2 x <8 x float>] %47, 0, !dbg !26
  %650 = shufflevector <8 x float> %649, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %651 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %648, <4 x float> %650) #9, !dbg !25
  %652 = shufflevector <8 x half> %627, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %653 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %652, <4 x float> %651) #9, !dbg !25
  %654 = shufflevector <8 x half> %629, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %655 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %654, <4 x float> %653) #9, !dbg !25
  %656 = shufflevector <8 x half> %631, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %657 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %656, <4 x float> %655) #9, !dbg !25
  %658 = shufflevector <8 x half> %633, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %659 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %658, <4 x float> %657) #9, !dbg !25
  %660 = shufflevector <8 x half> %635, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %661 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %660, <4 x float> %659) #9, !dbg !25
  %662 = shufflevector <8 x half> %637, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %663 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %662, <4 x float> %661) #9, !dbg !25
  %664 = shufflevector <8 x half> %639, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %665 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %664, <4 x float> %663) #9, !dbg !25
  %666 = shufflevector <8 x half> %625, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %667 = shufflevector <8 x float> %649, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %668 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %666, <4 x float> %667) #9, !dbg !25
  %669 = shufflevector <8 x half> %627, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %670 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %669, <4 x float> %668) #9, !dbg !25
  %671 = shufflevector <8 x half> %629, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %672 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %671, <4 x float> %670) #9, !dbg !25
  %673 = shufflevector <8 x half> %631, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %674 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %673, <4 x float> %672) #9, !dbg !25
  %675 = shufflevector <8 x half> %633, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %676 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %675, <4 x float> %674) #9, !dbg !25
  %677 = shufflevector <8 x half> %635, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %678 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %677, <4 x float> %676) #9, !dbg !25
  %679 = shufflevector <8 x half> %637, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %680 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %679, <4 x float> %678) #9, !dbg !25
  %681 = shufflevector <8 x half> %639, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %682 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %681, <4 x float> %680) #9, !dbg !25
  %683 = extractvalue [2 x <8 x float>] %47, 1, !dbg !26
  %684 = shufflevector <8 x float> %683, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %685 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %648, <4 x float> %684) #9, !dbg !25
  %686 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %652, <4 x float> %685) #9, !dbg !25
  %687 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %654, <4 x float> %686) #9, !dbg !25
  %688 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %656, <4 x float> %687) #9, !dbg !25
  %689 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %658, <4 x float> %688) #9, !dbg !25
  %690 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %660, <4 x float> %689) #9, !dbg !25
  %691 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %662, <4 x float> %690) #9, !dbg !25
  %692 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %664, <4 x float> %691) #9, !dbg !25
  %693 = shufflevector <8 x float> %683, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %694 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %666, <4 x float> %693) #9, !dbg !25
  %695 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %669, <4 x float> %694) #9, !dbg !25
  %696 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %671, <4 x float> %695) #9, !dbg !25
  %697 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %673, <4 x float> %696) #9, !dbg !25
  %698 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %675, <4 x float> %697) #9, !dbg !25
  %699 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %677, <4 x float> %698) #9, !dbg !25
  %700 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %679, <4 x float> %699) #9, !dbg !25
  %701 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %681, <4 x float> %700) #9, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %702 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #12, !dbg !25
  %703 = bitcast <4 x i32> %702 to <8 x half>, !dbg !25
  %704 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #12, !dbg !25
  %705 = bitcast <4 x i32> %704 to <8 x half>, !dbg !25
  %706 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #12, !dbg !25
  %707 = bitcast <4 x i32> %706 to <8 x half>, !dbg !25
  %708 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #12, !dbg !25
  %709 = bitcast <4 x i32> %708 to <8 x half>, !dbg !25
  %710 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #12, !dbg !25
  %711 = bitcast <4 x i32> %710 to <8 x half>, !dbg !25
  %712 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #12, !dbg !25
  %713 = bitcast <4 x i32> %712 to <8 x half>, !dbg !25
  %714 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #12, !dbg !25
  %715 = bitcast <4 x i32> %714 to <8 x half>, !dbg !25
  %716 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #12, !dbg !25
  %717 = bitcast <4 x i32> %716 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %dma.buffer.offset635 = or disjoint i32 %642, 128, !dbg !19
  %718 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds643, i32 %dma.buffer.offset635, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %dma.buffer.offset632 = or disjoint i32 %646, 128, !dbg !19
  %719 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds640, i32 %dma.buffer.offset632, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %720 = shufflevector <8 x half> %703, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %721 = extractvalue [2 x <8 x float>] %45, 0, !dbg !26
  %722 = shufflevector <8 x float> %721, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %723 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %720, <4 x float> %722) #9, !dbg !25
  %724 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %725 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %724, <4 x float> %723) #9, !dbg !25
  %726 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %727 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %726, <4 x float> %725) #9, !dbg !25
  %728 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %729 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %728, <4 x float> %727) #9, !dbg !25
  %730 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %731 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %730, <4 x float> %729) #9, !dbg !25
  %732 = shufflevector <8 x half> %713, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %733 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %732, <4 x float> %731) #9, !dbg !25
  %734 = shufflevector <8 x half> %715, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %735 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %734, <4 x float> %733) #9, !dbg !25
  %736 = shufflevector <8 x half> %717, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %737 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %736, <4 x float> %735) #9, !dbg !25
  %738 = shufflevector <8 x half> %703, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %739 = shufflevector <8 x float> %721, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %740 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %738, <4 x float> %739) #9, !dbg !25
  %741 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %742 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %741, <4 x float> %740) #9, !dbg !25
  %743 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %744 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %743, <4 x float> %742) #9, !dbg !25
  %745 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %746 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %745, <4 x float> %744) #9, !dbg !25
  %747 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %748 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %747, <4 x float> %746) #9, !dbg !25
  %749 = shufflevector <8 x half> %713, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %750 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %749, <4 x float> %748) #9, !dbg !25
  %751 = shufflevector <8 x half> %715, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %752 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %751, <4 x float> %750) #9, !dbg !25
  %753 = shufflevector <8 x half> %717, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %754 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %753, <4 x float> %752) #9, !dbg !25
  %755 = extractvalue [2 x <8 x float>] %45, 1, !dbg !26
  %756 = shufflevector <8 x float> %755, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %757 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %720, <4 x float> %756) #9, !dbg !25
  %758 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %724, <4 x float> %757) #9, !dbg !25
  %759 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %726, <4 x float> %758) #9, !dbg !25
  %760 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %728, <4 x float> %759) #9, !dbg !25
  %761 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %730, <4 x float> %760) #9, !dbg !25
  %762 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %732, <4 x float> %761) #9, !dbg !25
  %763 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %734, <4 x float> %762) #9, !dbg !25
  %764 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %736, <4 x float> %763) #9, !dbg !25
  %765 = shufflevector <8 x float> %755, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %766 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %738, <4 x float> %765) #9, !dbg !25
  %767 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %741, <4 x float> %766) #9, !dbg !25
  %768 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %743, <4 x float> %767) #9, !dbg !25
  %769 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %745, <4 x float> %768) #9, !dbg !25
  %770 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %747, <4 x float> %769) #9, !dbg !25
  %771 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %749, <4 x float> %770) #9, !dbg !25
  %772 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %751, <4 x float> %771) #9, !dbg !25
  %773 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %753, <4 x float> %772) #9, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %774 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #12, !dbg !25
  %775 = bitcast <4 x i32> %774 to <8 x half>, !dbg !25
  %776 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #12, !dbg !25
  %777 = bitcast <4 x i32> %776 to <8 x half>, !dbg !25
  %778 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #12, !dbg !25
  %779 = bitcast <4 x i32> %778 to <8 x half>, !dbg !25
  %780 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #12, !dbg !25
  %781 = bitcast <4 x i32> %780 to <8 x half>, !dbg !25
  %782 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #12, !dbg !25
  %783 = bitcast <4 x i32> %782 to <8 x half>, !dbg !25
  %784 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #12, !dbg !25
  %785 = bitcast <4 x i32> %784 to <8 x half>, !dbg !25
  %786 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #12, !dbg !25
  %787 = bitcast <4 x i32> %786 to <8 x half>, !dbg !25
  %788 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #12, !dbg !25
  %789 = bitcast <4 x i32> %788 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %dma.buffer.offset629 = or disjoint i32 %642, 192, !dbg !19
  %790 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds643, i32 %dma.buffer.offset629, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %dma.buffer.offset626 = or disjoint i32 %646, 192, !dbg !19
  %791 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds640, i32 %dma.buffer.offset626, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %792 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %793 = extractvalue [2 x <8 x float>] %44, 0, !dbg !26
  %794 = shufflevector <8 x float> %793, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %795 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %792, <4 x float> %794) #9, !dbg !25
  %796 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %797 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %796, <4 x float> %795) #9, !dbg !25
  %798 = shufflevector <8 x half> %779, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %799 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %798, <4 x float> %797) #9, !dbg !25
  %800 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %801 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %800, <4 x float> %799) #9, !dbg !25
  %802 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %803 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %802, <4 x float> %801) #9, !dbg !25
  %804 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %805 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %804, <4 x float> %803) #9, !dbg !25
  %806 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %807 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %806, <4 x float> %805) #9, !dbg !25
  %808 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %809 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %808, <4 x float> %807) #9, !dbg !25
  %810 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %811 = shufflevector <8 x float> %793, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %812 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %810, <4 x float> %811) #9, !dbg !25
  %813 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %814 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %813, <4 x float> %812) #9, !dbg !25
  %815 = shufflevector <8 x half> %779, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %816 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %815, <4 x float> %814) #9, !dbg !25
  %817 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %818 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %817, <4 x float> %816) #9, !dbg !25
  %819 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %820 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %819, <4 x float> %818) #9, !dbg !25
  %821 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %822 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %821, <4 x float> %820) #9, !dbg !25
  %823 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %824 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %823, <4 x float> %822) #9, !dbg !25
  %825 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %826 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %825, <4 x float> %824) #9, !dbg !25
  %827 = extractvalue [2 x <8 x float>] %44, 1, !dbg !26
  %828 = shufflevector <8 x float> %827, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %829 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %792, <4 x float> %828) #9, !dbg !25
  %830 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %796, <4 x float> %829) #9, !dbg !25
  %831 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %798, <4 x float> %830) #9, !dbg !25
  %832 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %800, <4 x float> %831) #9, !dbg !25
  %833 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %802, <4 x float> %832) #9, !dbg !25
  %834 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %804, <4 x float> %833) #9, !dbg !25
  %835 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %806, <4 x float> %834) #9, !dbg !25
  %836 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %808, <4 x float> %835) #9, !dbg !25
  %837 = shufflevector <8 x float> %827, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %838 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %810, <4 x float> %837) #9, !dbg !25
  %839 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %813, <4 x float> %838) #9, !dbg !25
  %840 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %815, <4 x float> %839) #9, !dbg !25
  %841 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %817, <4 x float> %840) #9, !dbg !25
  %842 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %819, <4 x float> %841) #9, !dbg !25
  %843 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %821, <4 x float> %842) #9, !dbg !25
  %844 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %823, <4 x float> %843) #9, !dbg !25
  %845 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %825, <4 x float> %844) #9, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %846 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #12, !dbg !25
  %847 = bitcast <4 x i32> %846 to <8 x half>, !dbg !25
  %848 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #12, !dbg !25
  %849 = bitcast <4 x i32> %848 to <8 x half>, !dbg !25
  %850 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #12, !dbg !25
  %851 = bitcast <4 x i32> %850 to <8 x half>, !dbg !25
  %852 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #12, !dbg !25
  %853 = bitcast <4 x i32> %852 to <8 x half>, !dbg !25
  %854 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #12, !dbg !25
  %855 = bitcast <4 x i32> %854 to <8 x half>, !dbg !25
  %856 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #12, !dbg !25
  %857 = bitcast <4 x i32> %856 to <8 x half>, !dbg !25
  %858 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #12, !dbg !25
  %859 = bitcast <4 x i32> %858 to <8 x half>, !dbg !25
  %860 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #12, !dbg !25
  %861 = bitcast <4 x i32> %860 to <8 x half>, !dbg !25
  %862 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %863 = extractvalue [2 x <8 x float>] %43, 0, !dbg !26
  %864 = shufflevector <8 x float> %863, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %865 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %862, <4 x float> %864) #9, !dbg !25
  %866 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %867 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %866, <4 x float> %865) #9, !dbg !25
  %868 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %869 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %868, <4 x float> %867) #9, !dbg !25
  %870 = shufflevector <8 x half> %853, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %871 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %870, <4 x float> %869) #9, !dbg !25
  %872 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %873 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %872, <4 x float> %871) #9, !dbg !25
  %874 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %875 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %874, <4 x float> %873) #9, !dbg !25
  %876 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %877 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %876, <4 x float> %875) #9, !dbg !25
  %878 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %879 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %878, <4 x float> %877) #9, !dbg !25
  %880 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %881 = shufflevector <8 x float> %863, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %882 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %880, <4 x float> %881) #9, !dbg !25
  %883 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %884 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %883, <4 x float> %882) #9, !dbg !25
  %885 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %886 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %885, <4 x float> %884) #9, !dbg !25
  %887 = shufflevector <8 x half> %853, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %888 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %887, <4 x float> %886) #9, !dbg !25
  %889 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %890 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %889, <4 x float> %888) #9, !dbg !25
  %891 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %892 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %891, <4 x float> %890) #9, !dbg !25
  %893 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %894 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %893, <4 x float> %892) #9, !dbg !25
  %895 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %896 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %895, <4 x float> %894) #9, !dbg !25
  %897 = extractvalue [2 x <8 x float>] %43, 1, !dbg !26
  %898 = shufflevector <8 x float> %897, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %899 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %862, <4 x float> %898) #9, !dbg !25
  %900 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %866, <4 x float> %899) #9, !dbg !25
  %901 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %868, <4 x float> %900) #9, !dbg !25
  %902 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %870, <4 x float> %901) #9, !dbg !25
  %903 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %872, <4 x float> %902) #9, !dbg !25
  %904 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %874, <4 x float> %903) #9, !dbg !25
  %905 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %876, <4 x float> %904) #9, !dbg !25
  %906 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %878, <4 x float> %905) #9, !dbg !25
  %907 = shufflevector <8 x float> %897, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %908 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %880, <4 x float> %907) #9, !dbg !25
  %909 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %883, <4 x float> %908) #9, !dbg !25
  %910 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %885, <4 x float> %909) #9, !dbg !25
  %911 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %887, <4 x float> %910) #9, !dbg !25
  %912 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %889, <4 x float> %911) #9, !dbg !25
  %913 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %891, <4 x float> %912) #9, !dbg !25
  %914 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %893, <4 x float> %913) #9, !dbg !25
  %915 = tail call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %895, <4 x float> %914) #9, !dbg !25
  %916 = shufflevector <4 x float> %665, <4 x float> %682, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %917 = insertvalue [2 x <8 x float>] poison, <8 x float> %916, 0, !dbg !12
  %918 = shufflevector <4 x float> %692, <4 x float> %701, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %919 = insertvalue [2 x <8 x float>] %917, <8 x float> %918, 1, !dbg !12
  %920 = shufflevector <4 x float> %737, <4 x float> %754, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %921 = insertvalue [2 x <8 x float>] poison, <8 x float> %920, 0, !dbg !12
  %922 = shufflevector <4 x float> %764, <4 x float> %773, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %923 = insertvalue [2 x <8 x float>] %921, <8 x float> %922, 1, !dbg !12
  %924 = shufflevector <4 x float> %809, <4 x float> %826, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %925 = insertvalue [2 x <8 x float>] poison, <8 x float> %924, 0, !dbg !12
  %926 = shufflevector <4 x float> %836, <4 x float> %845, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %927 = insertvalue [2 x <8 x float>] %925, <8 x float> %926, 1, !dbg !12
  %928 = shufflevector <4 x float> %879, <4 x float> %896, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %929 = insertvalue [2 x <8 x float>] poison, <8 x float> %928, 0, !dbg !12
  %930 = shufflevector <4 x float> %906, <4 x float> %915, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %931 = insertvalue [2 x <8 x float>] %929, <8 x float> %930, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %932 = add nuw nsw i64 %48, 1, !dbg !12
  %.not.not = icmp slt i64 %48, %6, !dbg !12
  br i1 %.not.not, label %42, label %._crit_edge, !dbg !12

._crit_edge:                                      ; preds = %314, %.._crit_edge_crit_edge
  %.idx69.pre-phi = phi i64 [ %.pre112, %.._crit_edge_crit_edge ], [ %.idx, %314 ], !dbg !13
  %address.local396.pre-phi = phi i64 [ %.pre111, %.._crit_edge_crit_edge ], [ %19, %314 ], !dbg !13
  %.pre-phi110 = phi i32 [ %.pre109, %.._crit_edge_crit_edge ], [ %598, %314 ], !dbg !12
  %.pre-phi108 = phi i32 [ %.pre107, %.._crit_edge_crit_edge ], [ %590, %314 ], !dbg !12
  %.pre-phi106 = phi i32 [ %.pre105, %.._crit_edge_crit_edge ], [ %589, %314 ], !dbg !12
  %.pre-phi102 = phi i32 [ %.pre101, %.._crit_edge_crit_edge ], [ %587, %314 ], !dbg !12
  %.lcssa74 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %919, %314 ], !dbg !12
  %.lcssa73 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %623, %314 ], !dbg !12
  %.lcssa72 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %923, %314 ], !dbg !12
  %.lcssa71 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %927, %314 ], !dbg !12
  %.lcssa70 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %931, %314 ], !dbg !12
  %933 = and i32 %17, 16, !dbg !12
  %.not2 = icmp eq i32 %933, 0, !dbg !12
  %934 = and i32 %17, 32, !dbg !12
  %.not3 = icmp eq i32 %934, 0, !dbg !12
  %935 = extractvalue [2 x <8 x float>] %.lcssa70, 0, !dbg !12
  %936 = extractelement <8 x float> %935, i64 0, !dbg !12
  %937 = extractelement <8 x float> %935, i64 1, !dbg !12
  %938 = extractelement <8 x float> %935, i64 2, !dbg !12
  %939 = extractelement <8 x float> %935, i64 3, !dbg !12
  %940 = select i1 %.not2, float %937, float %936, !dbg !12
  %941 = icmp samesign ult i32 %.pre-phi106, %.pre-phi108, !dbg !12
  %942 = select i1 %941, i32 %.pre-phi106, i32 %.pre-phi102, !dbg !12
  %943 = shl nuw nsw i32 %942, 2, !dbg !12
  %944 = bitcast float %940 to i32, !dbg !12
  %945 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %944), !dbg !12
  %946 = bitcast i32 %945 to float, !dbg !12
  %947 = select i1 %.not2, float %936, float %946, !dbg !12
  %948 = select i1 %.not2, float %946, float %937, !dbg !12
  %949 = select i1 %.not2, float %939, float %938, !dbg !12
  %950 = bitcast float %949 to i32, !dbg !12
  %951 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %950), !dbg !12
  %952 = bitcast i32 %951 to float, !dbg !12
  %953 = select i1 %.not2, float %938, float %952, !dbg !12
  %954 = select i1 %.not2, float %952, float %939, !dbg !12
  %955 = select i1 %.not3, float %953, float %947, !dbg !12
  %956 = icmp samesign ult i32 %.pre-phi110, %.pre-phi108, !dbg !12
  %957 = select i1 %956, i32 %.pre-phi110, i32 %.pre-phi102, !dbg !12
  %958 = shl nuw nsw i32 %957, 2, !dbg !12
  %959 = bitcast float %955 to i32, !dbg !12
  %960 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %959), !dbg !12
  %961 = bitcast i32 %960 to float, !dbg !12
  %962 = select i1 %.not3, float %954, float %948, !dbg !12
  %963 = bitcast float %962 to i32, !dbg !12
  %964 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %963), !dbg !12
  %965 = bitcast i32 %964 to float, !dbg !12
  %966 = insertelement <4 x i1> poison, i1 %.not3, i64 0, !dbg !12
  %967 = shufflevector <4 x i1> %966, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %968 = insertelement <4 x float> poison, float %947, i64 0, !dbg !12
  %969 = insertelement <4 x float> %968, float %948, i64 1, !dbg !12
  %970 = insertelement <4 x float> %969, float %961, i64 2, !dbg !12
  %971 = insertelement <4 x float> %970, float %965, i64 3, !dbg !12
  %972 = insertelement <4 x float> poison, float %961, i64 0, !dbg !12
  %973 = insertelement <4 x float> %972, float %965, i64 1, !dbg !12
  %974 = insertelement <4 x float> %973, float %953, i64 2, !dbg !12
  %975 = insertelement <4 x float> %974, float %954, i64 3, !dbg !12
  %976 = select <4 x i1> %967, <4 x float> %971, <4 x float> %975, !dbg !12
  %977 = extractvalue [2 x <8 x float>] %.lcssa71, 0, !dbg !12
  %978 = extractelement <8 x float> %977, i64 0, !dbg !12
  %979 = extractelement <8 x float> %977, i64 1, !dbg !12
  %980 = extractelement <8 x float> %977, i64 2, !dbg !12
  %981 = extractelement <8 x float> %977, i64 3, !dbg !12
  %982 = select i1 %.not2, float %979, float %978, !dbg !12
  %983 = bitcast float %982 to i32, !dbg !12
  %984 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %983), !dbg !12
  %985 = bitcast i32 %984 to float, !dbg !12
  %986 = select i1 %.not2, float %978, float %985, !dbg !12
  %987 = select i1 %.not2, float %985, float %979, !dbg !12
  %988 = select i1 %.not2, float %981, float %980, !dbg !12
  %989 = bitcast float %988 to i32, !dbg !12
  %990 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %989), !dbg !12
  %991 = bitcast i32 %990 to float, !dbg !12
  %992 = select i1 %.not2, float %980, float %991, !dbg !12
  %993 = select i1 %.not2, float %991, float %981, !dbg !12
  %994 = select i1 %.not3, float %992, float %986, !dbg !12
  %995 = bitcast float %994 to i32, !dbg !12
  %996 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %995), !dbg !12
  %997 = bitcast i32 %996 to float, !dbg !12
  %998 = select i1 %.not3, float %993, float %987, !dbg !12
  %999 = bitcast float %998 to i32, !dbg !12
  %1000 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %999), !dbg !12
  %1001 = bitcast i32 %1000 to float, !dbg !12
  %1002 = insertelement <4 x float> poison, float %986, i64 0, !dbg !12
  %1003 = insertelement <4 x float> %1002, float %987, i64 1, !dbg !12
  %1004 = insertelement <4 x float> %1003, float %997, i64 2, !dbg !12
  %1005 = insertelement <4 x float> %1004, float %1001, i64 3, !dbg !12
  %1006 = insertelement <4 x float> poison, float %997, i64 0, !dbg !12
  %1007 = insertelement <4 x float> %1006, float %1001, i64 1, !dbg !12
  %1008 = insertelement <4 x float> %1007, float %992, i64 2, !dbg !12
  %1009 = insertelement <4 x float> %1008, float %993, i64 3, !dbg !12
  %1010 = select <4 x i1> %967, <4 x float> %1005, <4 x float> %1009, !dbg !12
  %1011 = extractvalue [2 x <8 x float>] %.lcssa72, 0, !dbg !12
  %1012 = extractelement <8 x float> %1011, i64 0, !dbg !12
  %1013 = extractelement <8 x float> %1011, i64 1, !dbg !12
  %1014 = extractelement <8 x float> %1011, i64 2, !dbg !12
  %1015 = extractelement <8 x float> %1011, i64 3, !dbg !12
  %1016 = select i1 %.not2, float %1013, float %1012, !dbg !12
  %1017 = bitcast float %1016 to i32, !dbg !12
  %1018 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1017), !dbg !12
  %1019 = bitcast i32 %1018 to float, !dbg !12
  %1020 = select i1 %.not2, float %1012, float %1019, !dbg !12
  %1021 = select i1 %.not2, float %1019, float %1013, !dbg !12
  %1022 = select i1 %.not2, float %1015, float %1014, !dbg !12
  %1023 = bitcast float %1022 to i32, !dbg !12
  %1024 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1023), !dbg !12
  %1025 = bitcast i32 %1024 to float, !dbg !12
  %1026 = select i1 %.not2, float %1014, float %1025, !dbg !12
  %1027 = select i1 %.not2, float %1025, float %1015, !dbg !12
  %1028 = select i1 %.not3, float %1026, float %1020, !dbg !12
  %1029 = bitcast float %1028 to i32, !dbg !12
  %1030 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1029), !dbg !12
  %1031 = bitcast i32 %1030 to float, !dbg !12
  %1032 = select i1 %.not3, float %1027, float %1021, !dbg !12
  %1033 = bitcast float %1032 to i32, !dbg !12
  %1034 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1033), !dbg !12
  %1035 = bitcast i32 %1034 to float, !dbg !12
  %1036 = insertelement <4 x float> poison, float %1020, i64 0, !dbg !12
  %1037 = insertelement <4 x float> %1036, float %1021, i64 1, !dbg !12
  %1038 = insertelement <4 x float> %1037, float %1031, i64 2, !dbg !12
  %1039 = insertelement <4 x float> %1038, float %1035, i64 3, !dbg !12
  %1040 = insertelement <4 x float> poison, float %1031, i64 0, !dbg !12
  %1041 = insertelement <4 x float> %1040, float %1035, i64 1, !dbg !12
  %1042 = insertelement <4 x float> %1041, float %1026, i64 2, !dbg !12
  %1043 = insertelement <4 x float> %1042, float %1027, i64 3, !dbg !12
  %1044 = select <4 x i1> %967, <4 x float> %1039, <4 x float> %1043, !dbg !12
  %1045 = extractvalue [2 x <8 x float>] %.lcssa74, 0, !dbg !12
  %1046 = extractelement <8 x float> %1045, i64 0, !dbg !12
  %1047 = extractelement <8 x float> %1045, i64 1, !dbg !12
  %1048 = extractelement <8 x float> %1045, i64 2, !dbg !12
  %1049 = extractelement <8 x float> %1045, i64 3, !dbg !12
  %1050 = select i1 %.not2, float %1047, float %1046, !dbg !12
  %1051 = bitcast float %1050 to i32, !dbg !12
  %1052 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1051), !dbg !12
  %1053 = bitcast i32 %1052 to float, !dbg !12
  %1054 = select i1 %.not2, float %1046, float %1053, !dbg !12
  %1055 = select i1 %.not2, float %1053, float %1047, !dbg !12
  %1056 = select i1 %.not2, float %1049, float %1048, !dbg !12
  %1057 = bitcast float %1056 to i32, !dbg !12
  %1058 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1057), !dbg !12
  %1059 = bitcast i32 %1058 to float, !dbg !12
  %1060 = select i1 %.not2, float %1048, float %1059, !dbg !12
  %1061 = select i1 %.not2, float %1059, float %1049, !dbg !12
  %1062 = select i1 %.not3, float %1060, float %1054, !dbg !12
  %1063 = bitcast float %1062 to i32, !dbg !12
  %1064 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1063), !dbg !12
  %1065 = select i1 %.not3, float %1061, float %1055, !dbg !12
  %1066 = bitcast float %1065 to i32, !dbg !12
  %1067 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1066), !dbg !12
  %1068 = extractvalue [2 x <1 x float>] %.lcssa73, 0, !dbg !27
  %1069 = shufflevector <1 x float> %1068, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %1070 = extractelement <8 x float> %935, i64 4, !dbg !12
  %1071 = extractelement <8 x float> %935, i64 5, !dbg !12
  %1072 = extractelement <8 x float> %935, i64 6, !dbg !12
  %1073 = extractelement <8 x float> %935, i64 7, !dbg !12
  %1074 = select i1 %.not2, float %1071, float %1070, !dbg !12
  %1075 = bitcast float %1074 to i32, !dbg !12
  %1076 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1075), !dbg !12
  %1077 = bitcast i32 %1076 to float, !dbg !12
  %1078 = select i1 %.not2, float %1070, float %1077, !dbg !12
  %1079 = select i1 %.not2, float %1077, float %1071, !dbg !12
  %1080 = select i1 %.not2, float %1073, float %1072, !dbg !12
  %1081 = bitcast float %1080 to i32, !dbg !12
  %1082 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1081), !dbg !12
  %1083 = bitcast i32 %1082 to float, !dbg !12
  %1084 = select i1 %.not2, float %1072, float %1083, !dbg !12
  %1085 = select i1 %.not2, float %1083, float %1073, !dbg !12
  %1086 = select i1 %.not3, float %1084, float %1078, !dbg !12
  %1087 = bitcast float %1086 to i32, !dbg !12
  %1088 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1087), !dbg !12
  %1089 = bitcast i32 %1088 to float, !dbg !12
  %1090 = select i1 %.not3, float %1085, float %1079, !dbg !12
  %1091 = bitcast float %1090 to i32, !dbg !12
  %1092 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1091), !dbg !12
  %1093 = bitcast i32 %1092 to float, !dbg !12
  %1094 = insertelement <4 x float> poison, float %1078, i64 0, !dbg !12
  %1095 = insertelement <4 x float> %1094, float %1079, i64 1, !dbg !12
  %1096 = insertelement <4 x float> %1095, float %1089, i64 2, !dbg !12
  %1097 = insertelement <4 x float> %1096, float %1093, i64 3, !dbg !12
  %1098 = insertelement <4 x float> poison, float %1089, i64 0, !dbg !12
  %1099 = insertelement <4 x float> %1098, float %1093, i64 1, !dbg !12
  %1100 = insertelement <4 x float> %1099, float %1084, i64 2, !dbg !12
  %1101 = insertelement <4 x float> %1100, float %1085, i64 3, !dbg !12
  %1102 = select <4 x i1> %967, <4 x float> %1097, <4 x float> %1101, !dbg !12
  %1103 = extractelement <8 x float> %977, i64 4, !dbg !12
  %1104 = extractelement <8 x float> %977, i64 5, !dbg !12
  %1105 = extractelement <8 x float> %977, i64 6, !dbg !12
  %1106 = extractelement <8 x float> %977, i64 7, !dbg !12
  %1107 = select i1 %.not2, float %1104, float %1103, !dbg !12
  %1108 = bitcast float %1107 to i32, !dbg !12
  %1109 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1108), !dbg !12
  %1110 = bitcast i32 %1109 to float, !dbg !12
  %1111 = select i1 %.not2, float %1103, float %1110, !dbg !12
  %1112 = select i1 %.not2, float %1110, float %1104, !dbg !12
  %1113 = select i1 %.not2, float %1106, float %1105, !dbg !12
  %1114 = bitcast float %1113 to i32, !dbg !12
  %1115 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1114), !dbg !12
  %1116 = bitcast i32 %1115 to float, !dbg !12
  %1117 = select i1 %.not2, float %1105, float %1116, !dbg !12
  %1118 = select i1 %.not2, float %1116, float %1106, !dbg !12
  %1119 = select i1 %.not3, float %1117, float %1111, !dbg !12
  %1120 = bitcast float %1119 to i32, !dbg !12
  %1121 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1120), !dbg !12
  %1122 = bitcast i32 %1121 to float, !dbg !12
  %1123 = select i1 %.not3, float %1118, float %1112, !dbg !12
  %1124 = bitcast float %1123 to i32, !dbg !12
  %1125 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1124), !dbg !12
  %1126 = bitcast i32 %1125 to float, !dbg !12
  %1127 = insertelement <4 x float> poison, float %1111, i64 0, !dbg !12
  %1128 = insertelement <4 x float> %1127, float %1112, i64 1, !dbg !12
  %1129 = insertelement <4 x float> %1128, float %1122, i64 2, !dbg !12
  %1130 = insertelement <4 x float> %1129, float %1126, i64 3, !dbg !12
  %1131 = insertelement <4 x float> poison, float %1122, i64 0, !dbg !12
  %1132 = insertelement <4 x float> %1131, float %1126, i64 1, !dbg !12
  %1133 = insertelement <4 x float> %1132, float %1117, i64 2, !dbg !12
  %1134 = insertelement <4 x float> %1133, float %1118, i64 3, !dbg !12
  %1135 = select <4 x i1> %967, <4 x float> %1130, <4 x float> %1134, !dbg !12
  %1136 = extractelement <8 x float> %1011, i64 4, !dbg !12
  %1137 = extractelement <8 x float> %1011, i64 5, !dbg !12
  %1138 = extractelement <8 x float> %1011, i64 6, !dbg !12
  %1139 = extractelement <8 x float> %1011, i64 7, !dbg !12
  %1140 = select i1 %.not2, float %1137, float %1136, !dbg !12
  %1141 = bitcast float %1140 to i32, !dbg !12
  %1142 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1141), !dbg !12
  %1143 = bitcast i32 %1142 to float, !dbg !12
  %1144 = select i1 %.not2, float %1136, float %1143, !dbg !12
  %1145 = select i1 %.not2, float %1143, float %1137, !dbg !12
  %1146 = select i1 %.not2, float %1139, float %1138, !dbg !12
  %1147 = bitcast float %1146 to i32, !dbg !12
  %1148 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1147), !dbg !12
  %1149 = bitcast i32 %1148 to float, !dbg !12
  %1150 = select i1 %.not2, float %1138, float %1149, !dbg !12
  %1151 = select i1 %.not2, float %1149, float %1139, !dbg !12
  %1152 = select i1 %.not3, float %1150, float %1144, !dbg !12
  %1153 = bitcast float %1152 to i32, !dbg !12
  %1154 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1153), !dbg !12
  %1155 = bitcast i32 %1154 to float, !dbg !12
  %1156 = select i1 %.not3, float %1151, float %1145, !dbg !12
  %1157 = bitcast float %1156 to i32, !dbg !12
  %1158 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1157), !dbg !12
  %1159 = bitcast i32 %1158 to float, !dbg !12
  %1160 = insertelement <4 x float> poison, float %1144, i64 0, !dbg !12
  %1161 = insertelement <4 x float> %1160, float %1145, i64 1, !dbg !12
  %1162 = insertelement <4 x float> %1161, float %1155, i64 2, !dbg !12
  %1163 = insertelement <4 x float> %1162, float %1159, i64 3, !dbg !12
  %1164 = insertelement <4 x float> poison, float %1155, i64 0, !dbg !12
  %1165 = insertelement <4 x float> %1164, float %1159, i64 1, !dbg !12
  %1166 = insertelement <4 x float> %1165, float %1150, i64 2, !dbg !12
  %1167 = insertelement <4 x float> %1166, float %1151, i64 3, !dbg !12
  %1168 = select <4 x i1> %967, <4 x float> %1163, <4 x float> %1167, !dbg !12
  %1169 = extractelement <8 x float> %1045, i64 4, !dbg !12
  %1170 = extractelement <8 x float> %1045, i64 5, !dbg !12
  %1171 = extractelement <8 x float> %1045, i64 6, !dbg !12
  %1172 = extractelement <8 x float> %1045, i64 7, !dbg !12
  %1173 = select i1 %.not2, float %1170, float %1169, !dbg !12
  %1174 = bitcast float %1173 to i32, !dbg !12
  %1175 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1174), !dbg !12
  %1176 = bitcast i32 %1175 to float, !dbg !12
  %1177 = select i1 %.not2, float %1169, float %1176, !dbg !12
  %1178 = select i1 %.not2, float %1176, float %1170, !dbg !12
  %1179 = select i1 %.not2, float %1172, float %1171, !dbg !12
  %1180 = bitcast float %1179 to i32, !dbg !12
  %1181 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1180), !dbg !12
  %1182 = bitcast i32 %1181 to float, !dbg !12
  %1183 = select i1 %.not2, float %1171, float %1182, !dbg !12
  %1184 = select i1 %.not2, float %1182, float %1172, !dbg !12
  %1185 = select i1 %.not3, float %1183, float %1177, !dbg !12
  %1186 = bitcast float %1185 to i32, !dbg !12
  %1187 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1186), !dbg !12
  %1188 = select i1 %.not3, float %1184, float %1178, !dbg !12
  %1189 = bitcast float %1188 to i32, !dbg !12
  %1190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1189), !dbg !12
  %1191 = extractvalue [2 x <8 x float>] %.lcssa70, 1, !dbg !12
  %1192 = extractelement <8 x float> %1191, i64 0, !dbg !12
  %1193 = extractelement <8 x float> %1191, i64 1, !dbg !12
  %1194 = extractelement <8 x float> %1191, i64 2, !dbg !12
  %1195 = extractelement <8 x float> %1191, i64 3, !dbg !12
  %1196 = select i1 %.not2, float %1193, float %1192, !dbg !12
  %1197 = bitcast float %1196 to i32, !dbg !12
  %1198 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1197), !dbg !12
  %1199 = bitcast i32 %1198 to float, !dbg !12
  %1200 = select i1 %.not2, float %1192, float %1199, !dbg !12
  %1201 = select i1 %.not2, float %1199, float %1193, !dbg !12
  %1202 = select i1 %.not2, float %1195, float %1194, !dbg !12
  %1203 = bitcast float %1202 to i32, !dbg !12
  %1204 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1203), !dbg !12
  %1205 = bitcast i32 %1204 to float, !dbg !12
  %1206 = select i1 %.not2, float %1194, float %1205, !dbg !12
  %1207 = select i1 %.not2, float %1205, float %1195, !dbg !12
  %1208 = select i1 %.not3, float %1206, float %1200, !dbg !12
  %1209 = bitcast float %1208 to i32, !dbg !12
  %1210 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1209), !dbg !12
  %1211 = bitcast i32 %1210 to float, !dbg !12
  %1212 = select i1 %.not3, float %1207, float %1201, !dbg !12
  %1213 = bitcast float %1212 to i32, !dbg !12
  %1214 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1213), !dbg !12
  %1215 = bitcast i32 %1214 to float, !dbg !12
  %1216 = insertelement <4 x float> poison, float %1200, i64 0, !dbg !12
  %1217 = insertelement <4 x float> %1216, float %1201, i64 1, !dbg !12
  %1218 = insertelement <4 x float> %1217, float %1211, i64 2, !dbg !12
  %1219 = insertelement <4 x float> %1218, float %1215, i64 3, !dbg !12
  %1220 = insertelement <4 x float> poison, float %1211, i64 0, !dbg !12
  %1221 = insertelement <4 x float> %1220, float %1215, i64 1, !dbg !12
  %1222 = insertelement <4 x float> %1221, float %1206, i64 2, !dbg !12
  %1223 = insertelement <4 x float> %1222, float %1207, i64 3, !dbg !12
  %1224 = select <4 x i1> %967, <4 x float> %1219, <4 x float> %1223, !dbg !12
  %1225 = extractvalue [2 x <8 x float>] %.lcssa71, 1, !dbg !12
  %1226 = extractelement <8 x float> %1225, i64 0, !dbg !12
  %1227 = extractelement <8 x float> %1225, i64 1, !dbg !12
  %1228 = extractelement <8 x float> %1225, i64 2, !dbg !12
  %1229 = extractelement <8 x float> %1225, i64 3, !dbg !12
  %1230 = select i1 %.not2, float %1227, float %1226, !dbg !12
  %1231 = bitcast float %1230 to i32, !dbg !12
  %1232 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1231), !dbg !12
  %1233 = bitcast i32 %1232 to float, !dbg !12
  %1234 = select i1 %.not2, float %1226, float %1233, !dbg !12
  %1235 = select i1 %.not2, float %1233, float %1227, !dbg !12
  %1236 = select i1 %.not2, float %1229, float %1228, !dbg !12
  %1237 = bitcast float %1236 to i32, !dbg !12
  %1238 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1237), !dbg !12
  %1239 = bitcast i32 %1238 to float, !dbg !12
  %1240 = select i1 %.not2, float %1228, float %1239, !dbg !12
  %1241 = select i1 %.not2, float %1239, float %1229, !dbg !12
  %1242 = select i1 %.not3, float %1240, float %1234, !dbg !12
  %1243 = bitcast float %1242 to i32, !dbg !12
  %1244 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1243), !dbg !12
  %1245 = bitcast i32 %1244 to float, !dbg !12
  %1246 = select i1 %.not3, float %1241, float %1235, !dbg !12
  %1247 = bitcast float %1246 to i32, !dbg !12
  %1248 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1247), !dbg !12
  %1249 = bitcast i32 %1248 to float, !dbg !12
  %1250 = insertelement <4 x float> poison, float %1234, i64 0, !dbg !12
  %1251 = insertelement <4 x float> %1250, float %1235, i64 1, !dbg !12
  %1252 = insertelement <4 x float> %1251, float %1245, i64 2, !dbg !12
  %1253 = insertelement <4 x float> %1252, float %1249, i64 3, !dbg !12
  %1254 = insertelement <4 x float> poison, float %1245, i64 0, !dbg !12
  %1255 = insertelement <4 x float> %1254, float %1249, i64 1, !dbg !12
  %1256 = insertelement <4 x float> %1255, float %1240, i64 2, !dbg !12
  %1257 = insertelement <4 x float> %1256, float %1241, i64 3, !dbg !12
  %1258 = select <4 x i1> %967, <4 x float> %1253, <4 x float> %1257, !dbg !12
  %1259 = extractvalue [2 x <8 x float>] %.lcssa72, 1, !dbg !12
  %1260 = extractelement <8 x float> %1259, i64 0, !dbg !12
  %1261 = extractelement <8 x float> %1259, i64 1, !dbg !12
  %1262 = extractelement <8 x float> %1259, i64 2, !dbg !12
  %1263 = extractelement <8 x float> %1259, i64 3, !dbg !12
  %1264 = select i1 %.not2, float %1261, float %1260, !dbg !12
  %1265 = bitcast float %1264 to i32, !dbg !12
  %1266 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1265), !dbg !12
  %1267 = bitcast i32 %1266 to float, !dbg !12
  %1268 = select i1 %.not2, float %1260, float %1267, !dbg !12
  %1269 = select i1 %.not2, float %1267, float %1261, !dbg !12
  %1270 = select i1 %.not2, float %1263, float %1262, !dbg !12
  %1271 = bitcast float %1270 to i32, !dbg !12
  %1272 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1271), !dbg !12
  %1273 = bitcast i32 %1272 to float, !dbg !12
  %1274 = select i1 %.not2, float %1262, float %1273, !dbg !12
  %1275 = select i1 %.not2, float %1273, float %1263, !dbg !12
  %1276 = select i1 %.not3, float %1274, float %1268, !dbg !12
  %1277 = bitcast float %1276 to i32, !dbg !12
  %1278 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1277), !dbg !12
  %1279 = bitcast i32 %1278 to float, !dbg !12
  %1280 = select i1 %.not3, float %1275, float %1269, !dbg !12
  %1281 = bitcast float %1280 to i32, !dbg !12
  %1282 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1281), !dbg !12
  %1283 = bitcast i32 %1282 to float, !dbg !12
  %1284 = insertelement <4 x float> poison, float %1268, i64 0, !dbg !12
  %1285 = insertelement <4 x float> %1284, float %1269, i64 1, !dbg !12
  %1286 = insertelement <4 x float> %1285, float %1279, i64 2, !dbg !12
  %1287 = insertelement <4 x float> %1286, float %1283, i64 3, !dbg !12
  %1288 = insertelement <4 x float> poison, float %1279, i64 0, !dbg !12
  %1289 = insertelement <4 x float> %1288, float %1283, i64 1, !dbg !12
  %1290 = insertelement <4 x float> %1289, float %1274, i64 2, !dbg !12
  %1291 = insertelement <4 x float> %1290, float %1275, i64 3, !dbg !12
  %1292 = select <4 x i1> %967, <4 x float> %1287, <4 x float> %1291, !dbg !12
  %1293 = extractvalue [2 x <8 x float>] %.lcssa74, 1, !dbg !12
  %1294 = extractelement <8 x float> %1293, i64 0, !dbg !12
  %1295 = extractelement <8 x float> %1293, i64 1, !dbg !12
  %1296 = extractelement <8 x float> %1293, i64 2, !dbg !12
  %1297 = extractelement <8 x float> %1293, i64 3, !dbg !12
  %1298 = select i1 %.not2, float %1295, float %1294, !dbg !12
  %1299 = bitcast float %1298 to i32, !dbg !12
  %1300 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1299), !dbg !12
  %1301 = bitcast i32 %1300 to float, !dbg !12
  %1302 = select i1 %.not2, float %1294, float %1301, !dbg !12
  %1303 = select i1 %.not2, float %1301, float %1295, !dbg !12
  %1304 = select i1 %.not2, float %1297, float %1296, !dbg !12
  %1305 = bitcast float %1304 to i32, !dbg !12
  %1306 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1305), !dbg !12
  %1307 = bitcast i32 %1306 to float, !dbg !12
  %1308 = select i1 %.not2, float %1296, float %1307, !dbg !12
  %1309 = select i1 %.not2, float %1307, float %1297, !dbg !12
  %1310 = select i1 %.not3, float %1308, float %1302, !dbg !12
  %1311 = bitcast float %1310 to i32, !dbg !12
  %1312 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1311), !dbg !12
  %1313 = select i1 %.not3, float %1309, float %1303, !dbg !12
  %1314 = bitcast float %1313 to i32, !dbg !12
  %1315 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1314), !dbg !12
  %1316 = extractvalue [2 x <1 x float>] %.lcssa73, 1, !dbg !27
  %1317 = shufflevector <1 x float> %1316, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %1318 = extractelement <8 x float> %1191, i64 4, !dbg !12
  %1319 = extractelement <8 x float> %1191, i64 5, !dbg !12
  %1320 = extractelement <8 x float> %1191, i64 6, !dbg !12
  %1321 = extractelement <8 x float> %1191, i64 7, !dbg !12
  %1322 = select i1 %.not2, float %1319, float %1318, !dbg !12
  %1323 = bitcast float %1322 to i32, !dbg !12
  %1324 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1323), !dbg !12
  %1325 = bitcast i32 %1324 to float, !dbg !12
  %1326 = select i1 %.not2, float %1318, float %1325, !dbg !12
  %1327 = select i1 %.not2, float %1325, float %1319, !dbg !12
  %1328 = select i1 %.not2, float %1321, float %1320, !dbg !12
  %1329 = bitcast float %1328 to i32, !dbg !12
  %1330 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1329), !dbg !12
  %1331 = bitcast i32 %1330 to float, !dbg !12
  %1332 = select i1 %.not2, float %1320, float %1331, !dbg !12
  %1333 = select i1 %.not2, float %1331, float %1321, !dbg !12
  %1334 = select i1 %.not3, float %1332, float %1326, !dbg !12
  %1335 = bitcast float %1334 to i32, !dbg !12
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1335), !dbg !12
  %1337 = bitcast i32 %1336 to float, !dbg !12
  %1338 = select i1 %.not3, float %1333, float %1327, !dbg !12
  %1339 = bitcast float %1338 to i32, !dbg !12
  %1340 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1339), !dbg !12
  %1341 = bitcast i32 %1340 to float, !dbg !12
  %1342 = insertelement <4 x float> poison, float %1326, i64 0, !dbg !12
  %1343 = insertelement <4 x float> %1342, float %1327, i64 1, !dbg !12
  %1344 = insertelement <4 x float> %1343, float %1337, i64 2, !dbg !12
  %1345 = insertelement <4 x float> %1344, float %1341, i64 3, !dbg !12
  %1346 = insertelement <4 x float> poison, float %1337, i64 0, !dbg !12
  %1347 = insertelement <4 x float> %1346, float %1341, i64 1, !dbg !12
  %1348 = insertelement <4 x float> %1347, float %1332, i64 2, !dbg !12
  %1349 = insertelement <4 x float> %1348, float %1333, i64 3, !dbg !12
  %1350 = select <4 x i1> %967, <4 x float> %1345, <4 x float> %1349, !dbg !12
  %1351 = extractelement <8 x float> %1225, i64 4, !dbg !12
  %1352 = extractelement <8 x float> %1225, i64 5, !dbg !12
  %1353 = extractelement <8 x float> %1225, i64 6, !dbg !12
  %1354 = extractelement <8 x float> %1225, i64 7, !dbg !12
  %1355 = select i1 %.not2, float %1352, float %1351, !dbg !12
  %1356 = bitcast float %1355 to i32, !dbg !12
  %1357 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1356), !dbg !12
  %1358 = bitcast i32 %1357 to float, !dbg !12
  %1359 = select i1 %.not2, float %1351, float %1358, !dbg !12
  %1360 = select i1 %.not2, float %1358, float %1352, !dbg !12
  %1361 = select i1 %.not2, float %1354, float %1353, !dbg !12
  %1362 = bitcast float %1361 to i32, !dbg !12
  %1363 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1362), !dbg !12
  %1364 = bitcast i32 %1363 to float, !dbg !12
  %1365 = select i1 %.not2, float %1353, float %1364, !dbg !12
  %1366 = select i1 %.not2, float %1364, float %1354, !dbg !12
  %1367 = select i1 %.not3, float %1365, float %1359, !dbg !12
  %1368 = bitcast float %1367 to i32, !dbg !12
  %1369 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1368), !dbg !12
  %1370 = bitcast i32 %1369 to float, !dbg !12
  %1371 = select i1 %.not3, float %1366, float %1360, !dbg !12
  %1372 = bitcast float %1371 to i32, !dbg !12
  %1373 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1372), !dbg !12
  %1374 = bitcast i32 %1373 to float, !dbg !12
  %1375 = insertelement <4 x float> poison, float %1359, i64 0, !dbg !12
  %1376 = insertelement <4 x float> %1375, float %1360, i64 1, !dbg !12
  %1377 = insertelement <4 x float> %1376, float %1370, i64 2, !dbg !12
  %1378 = insertelement <4 x float> %1377, float %1374, i64 3, !dbg !12
  %1379 = insertelement <4 x float> poison, float %1370, i64 0, !dbg !12
  %1380 = insertelement <4 x float> %1379, float %1374, i64 1, !dbg !12
  %1381 = insertelement <4 x float> %1380, float %1365, i64 2, !dbg !12
  %1382 = insertelement <4 x float> %1381, float %1366, i64 3, !dbg !12
  %1383 = select <4 x i1> %967, <4 x float> %1378, <4 x float> %1382, !dbg !12
  %1384 = extractelement <8 x float> %1259, i64 4, !dbg !12
  %1385 = extractelement <8 x float> %1259, i64 5, !dbg !12
  %1386 = extractelement <8 x float> %1259, i64 6, !dbg !12
  %1387 = extractelement <8 x float> %1259, i64 7, !dbg !12
  %1388 = select i1 %.not2, float %1385, float %1384, !dbg !12
  %1389 = bitcast float %1388 to i32, !dbg !12
  %1390 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1389), !dbg !12
  %1391 = bitcast i32 %1390 to float, !dbg !12
  %1392 = select i1 %.not2, float %1384, float %1391, !dbg !12
  %1393 = select i1 %.not2, float %1391, float %1385, !dbg !12
  %1394 = select i1 %.not2, float %1387, float %1386, !dbg !12
  %1395 = bitcast float %1394 to i32, !dbg !12
  %1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1395), !dbg !12
  %1397 = bitcast i32 %1396 to float, !dbg !12
  %1398 = select i1 %.not2, float %1386, float %1397, !dbg !12
  %1399 = select i1 %.not2, float %1397, float %1387, !dbg !12
  %1400 = select i1 %.not3, float %1398, float %1392, !dbg !12
  %1401 = bitcast float %1400 to i32, !dbg !12
  %1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1401), !dbg !12
  %1403 = bitcast i32 %1402 to float, !dbg !12
  %1404 = select i1 %.not3, float %1399, float %1393, !dbg !12
  %1405 = bitcast float %1404 to i32, !dbg !12
  %1406 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1405), !dbg !12
  %1407 = bitcast i32 %1406 to float, !dbg !12
  %1408 = insertelement <4 x float> poison, float %1392, i64 0, !dbg !12
  %1409 = insertelement <4 x float> %1408, float %1393, i64 1, !dbg !12
  %1410 = insertelement <4 x float> %1409, float %1403, i64 2, !dbg !12
  %1411 = insertelement <4 x float> %1410, float %1407, i64 3, !dbg !12
  %1412 = insertelement <4 x float> poison, float %1403, i64 0, !dbg !12
  %1413 = insertelement <4 x float> %1412, float %1407, i64 1, !dbg !12
  %1414 = insertelement <4 x float> %1413, float %1398, i64 2, !dbg !12
  %1415 = insertelement <4 x float> %1414, float %1399, i64 3, !dbg !12
  %1416 = select <4 x i1> %967, <4 x float> %1411, <4 x float> %1415, !dbg !12
  %1417 = extractelement <8 x float> %1293, i64 4, !dbg !12
  %1418 = extractelement <8 x float> %1293, i64 5, !dbg !12
  %1419 = extractelement <8 x float> %1293, i64 6, !dbg !12
  %1420 = extractelement <8 x float> %1293, i64 7, !dbg !12
  %1421 = select i1 %.not2, float %1418, float %1417, !dbg !12
  %1422 = bitcast float %1421 to i32, !dbg !12
  %1423 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1422), !dbg !12
  %1424 = bitcast i32 %1423 to float, !dbg !12
  %1425 = select i1 %.not2, float %1417, float %1424, !dbg !12
  %1426 = select i1 %.not2, float %1424, float %1418, !dbg !12
  %1427 = select i1 %.not2, float %1420, float %1419, !dbg !12
  %1428 = bitcast float %1427 to i32, !dbg !12
  %1429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %943, i32 %1428), !dbg !12
  %1430 = bitcast i32 %1429 to float, !dbg !12
  %1431 = select i1 %.not2, float %1419, float %1430, !dbg !12
  %1432 = select i1 %.not2, float %1430, float %1420, !dbg !12
  %1433 = select i1 %.not3, float %1431, float %1425, !dbg !12
  %1434 = bitcast float %1433 to i32, !dbg !12
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1434), !dbg !12
  %1436 = select i1 %.not3, float %1432, float %1426, !dbg !12
  %1437 = bitcast float %1436 to i32, !dbg !12
  %1438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %958, i32 %1437), !dbg !12
  %1439 = bitcast i32 %1435 to float, !dbg !12
  %1440 = bitcast i32 %1438 to float, !dbg !12
  %1441 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !12
  %1442 = insertelement <4 x float> %1441, float %1426, i64 1, !dbg !12
  %1443 = insertelement <4 x float> %1442, float %1439, i64 2, !dbg !12
  %1444 = insertelement <4 x float> %1443, float %1440, i64 3, !dbg !12
  %1445 = insertelement <4 x float> poison, float %1439, i64 0, !dbg !12
  %1446 = insertelement <4 x float> %1445, float %1440, i64 1, !dbg !12
  %1447 = insertelement <4 x float> %1446, float %1431, i64 2, !dbg !12
  %1448 = insertelement <4 x float> %1447, float %1432, i64 3, !dbg !12
  %1449 = select <4 x i1> %967, <4 x float> %1444, <4 x float> %1448, !dbg !12
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1317, !dbg !27
  %normalized = fmul <4 x float> %row.reciprocal, %1449, !dbg !27
  %1450 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !28
  %1451 = extractelement <4 x half> %1450, i64 3, !dbg !28
  %1452 = extractelement <4 x half> %1450, i64 2, !dbg !28
  %1453 = extractelement <4 x half> %1450, i64 1, !dbg !28
  %1454 = extractelement <4 x half> %1450, i64 0, !dbg !28
  %1455 = bitcast i32 %1312 to float, !dbg !12
  %1456 = bitcast i32 %1315 to float, !dbg !12
  %1457 = insertelement <4 x float> poison, float %1302, i64 0, !dbg !12
  %1458 = insertelement <4 x float> %1457, float %1303, i64 1, !dbg !12
  %1459 = insertelement <4 x float> %1458, float %1455, i64 2, !dbg !12
  %1460 = insertelement <4 x float> %1459, float %1456, i64 3, !dbg !12
  %1461 = insertelement <4 x float> poison, float %1455, i64 0, !dbg !12
  %1462 = insertelement <4 x float> %1461, float %1456, i64 1, !dbg !12
  %1463 = insertelement <4 x float> %1462, float %1308, i64 2, !dbg !12
  %1464 = insertelement <4 x float> %1463, float %1309, i64 3, !dbg !12
  %1465 = select <4 x i1> %967, <4 x float> %1460, <4 x float> %1464, !dbg !12
  %normalized75 = fmul <4 x float> %row.reciprocal, %1465, !dbg !27
  %1466 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !28
  %1467 = extractelement <4 x half> %1466, i64 3, !dbg !28
  %1468 = extractelement <4 x half> %1466, i64 2, !dbg !28
  %1469 = extractelement <4 x half> %1466, i64 1, !dbg !28
  %1470 = extractelement <4 x half> %1466, i64 0, !dbg !28
  %1471 = bitcast i32 %1187 to float, !dbg !12
  %1472 = bitcast i32 %1190 to float, !dbg !12
  %1473 = insertelement <4 x float> poison, float %1177, i64 0, !dbg !12
  %1474 = insertelement <4 x float> %1473, float %1178, i64 1, !dbg !12
  %1475 = insertelement <4 x float> %1474, float %1471, i64 2, !dbg !12
  %1476 = insertelement <4 x float> %1475, float %1472, i64 3, !dbg !12
  %1477 = insertelement <4 x float> poison, float %1471, i64 0, !dbg !12
  %1478 = insertelement <4 x float> %1477, float %1472, i64 1, !dbg !12
  %1479 = insertelement <4 x float> %1478, float %1183, i64 2, !dbg !12
  %1480 = insertelement <4 x float> %1479, float %1184, i64 3, !dbg !12
  %1481 = select <4 x i1> %967, <4 x float> %1476, <4 x float> %1480, !dbg !12
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1069, !dbg !27
  %normalized77 = fmul <4 x float> %row.reciprocal76, %1481, !dbg !27
  %1482 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !28
  %1483 = extractelement <4 x half> %1482, i64 3, !dbg !28
  %1484 = extractelement <4 x half> %1482, i64 2, !dbg !28
  %1485 = extractelement <4 x half> %1482, i64 1, !dbg !28
  %1486 = extractelement <4 x half> %1482, i64 0, !dbg !28
  %1487 = bitcast i32 %1064 to float, !dbg !12
  %1488 = bitcast i32 %1067 to float, !dbg !12
  %1489 = insertelement <4 x float> poison, float %1054, i64 0, !dbg !12
  %1490 = insertelement <4 x float> %1489, float %1055, i64 1, !dbg !12
  %1491 = insertelement <4 x float> %1490, float %1487, i64 2, !dbg !12
  %1492 = insertelement <4 x float> %1491, float %1488, i64 3, !dbg !12
  %1493 = insertelement <4 x float> poison, float %1487, i64 0, !dbg !12
  %1494 = insertelement <4 x float> %1493, float %1488, i64 1, !dbg !12
  %1495 = insertelement <4 x float> %1494, float %1060, i64 2, !dbg !12
  %1496 = insertelement <4 x float> %1495, float %1061, i64 3, !dbg !12
  %1497 = select <4 x i1> %967, <4 x float> %1492, <4 x float> %1496, !dbg !12
  %normalized78 = fmul <4 x float> %row.reciprocal76, %1497, !dbg !27
  %1498 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !28
  %1499 = extractelement <4 x half> %1498, i64 3, !dbg !28
  %1500 = extractelement <4 x half> %1498, i64 2, !dbg !28
  %1501 = extractelement <4 x half> %1498, i64 1, !dbg !28
  %1502 = extractelement <4 x half> %1498, i64 0, !dbg !28
  %1503 = shl nsw i64 %14, 19, !dbg !13
  %1504 = and i64 %1503, 16252928, !dbg !13
  %address.thread394 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #9, !dbg !13, !range !11, !frisk.register_lifetimes !5
  %address.local395 = zext nneg i32 %address.thread394 to i64, !dbg !10
  %1505 = lshr i64 %address.local395, 2, !dbg !13
  %address.local398 = and i64 %1505, 48, !dbg !13
  %address.local400 = and i64 %address.local395, 240, !dbg !13
  %address.local399 = add nsw i64 %address.local396.pre-phi, %address.local395, !dbg !13
  %address.local401 = sub nsw i64 %address.local399, %address.local400, !dbg !13
  %address.local402 = add nsw i64 %address.local401, %address.local398, !dbg !13
  %address.local403 = shl nuw nsw i64 %address.local402, 7, !dbg !13
  %1506 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %1504, !dbg !13
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1506, i64 %address.local403, !dbg !13
  %address.local407 = and i64 %1505, 12, !dbg !10
  %address.local408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local407, !dbg !13
  store half %1502, ptr addrspace(1) %address.local408, align 2, !dbg !13
  %address.local410 = or disjoint i64 %address.local407, 1, !dbg !13
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local410, !dbg !13
  store half %1501, ptr addrspace(1) %address.local411, align 2, !dbg !13
  %address.local413 = or disjoint i64 %address.local407, 2, !dbg !13
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local413, !dbg !13
  store half %1500, ptr addrspace(1) %address.local414, align 2, !dbg !13
  %address.local416 = or disjoint i64 %address.local407, 3, !dbg !13
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local416, !dbg !13
  store half %1499, ptr addrspace(1) %address.local417, align 2, !dbg !13
  %address.local419 = or disjoint i64 %address.local407, 16, !dbg !13
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local419, !dbg !13
  store half %1486, ptr addrspace(1) %address.local420, align 2, !dbg !13
  %address.local422 = or disjoint i64 %address.local407, 17, !dbg !13
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local422, !dbg !13
  store half %1485, ptr addrspace(1) %address.local423, align 2, !dbg !13
  %address.local425 = or disjoint i64 %address.local407, 18, !dbg !13
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local425, !dbg !13
  store half %1484, ptr addrspace(1) %address.local426, align 2, !dbg !13
  %address.local428 = or disjoint i64 %address.local407, 19, !dbg !13
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local428, !dbg !13
  store half %1483, ptr addrspace(1) %address.local429, align 2, !dbg !13
  %address.local431 = add nuw nsw i64 %address.local403, 8192, !dbg !13
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1506, i64 %address.local431, !dbg !13
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local407, !dbg !13
  store half %1470, ptr addrspace(1) %address.local434, align 2, !dbg !13
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local410, !dbg !13
  store half %1469, ptr addrspace(1) %address.local436, align 2, !dbg !13
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local413, !dbg !13
  store half %1468, ptr addrspace(1) %address.local438, align 2, !dbg !13
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local416, !dbg !13
  store half %1467, ptr addrspace(1) %address.local440, align 2, !dbg !13
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local419, !dbg !13
  store half %1454, ptr addrspace(1) %address.local442, align 2, !dbg !13
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local422, !dbg !13
  store half %1453, ptr addrspace(1) %address.local444, align 2, !dbg !13
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local425, !dbg !13
  store half %1452, ptr addrspace(1) %address.local446, align 2, !dbg !13
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local428, !dbg !13
  store half %1451, ptr addrspace(1) %address.local448, align 2, !dbg !13
  %normalized80 = fmul <4 x float> %row.reciprocal, %1416, !dbg !27
  %1507 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !28
  %1508 = extractelement <4 x half> %1507, i64 3, !dbg !28
  %1509 = extractelement <4 x half> %1507, i64 2, !dbg !28
  %1510 = extractelement <4 x half> %1507, i64 1, !dbg !28
  %1511 = extractelement <4 x half> %1507, i64 0, !dbg !28
  %normalized81 = fmul <4 x float> %row.reciprocal, %1292, !dbg !27
  %1512 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !28
  %1513 = extractelement <4 x half> %1512, i64 3, !dbg !28
  %1514 = extractelement <4 x half> %1512, i64 2, !dbg !28
  %1515 = extractelement <4 x half> %1512, i64 1, !dbg !28
  %1516 = extractelement <4 x half> %1512, i64 0, !dbg !28
  %normalized83 = fmul <4 x float> %row.reciprocal76, %1168, !dbg !27
  %1517 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !28
  %1518 = extractelement <4 x half> %1517, i64 3, !dbg !28
  %1519 = extractelement <4 x half> %1517, i64 2, !dbg !28
  %1520 = extractelement <4 x half> %1517, i64 1, !dbg !28
  %1521 = extractelement <4 x half> %1517, i64 0, !dbg !28
  %normalized84 = fmul <4 x float> %row.reciprocal76, %1044, !dbg !27
  %1522 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !28
  %1523 = extractelement <4 x half> %1522, i64 3, !dbg !28
  %1524 = extractelement <4 x half> %1522, i64 2, !dbg !28
  %1525 = extractelement <4 x half> %1522, i64 1, !dbg !28
  %1526 = extractelement <4 x half> %1522, i64 0, !dbg !28
  %1527 = getelementptr inbounds nuw i8, ptr addrspace(1) %3, i64 %.idx69.pre-phi, !dbg !13
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1527, i64 %address.local403, !dbg !13
  %address.local463 = or disjoint i64 %address.local407, 32, !dbg !13
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local463, !dbg !13
  store half %1526, ptr addrspace(1) %address.local464, align 2, !dbg !13
  %address.local466 = or disjoint i64 %address.local407, 33, !dbg !13
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local466, !dbg !13
  store half %1525, ptr addrspace(1) %address.local467, align 2, !dbg !13
  %address.local469 = or disjoint i64 %address.local407, 34, !dbg !13
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local469, !dbg !13
  store half %1524, ptr addrspace(1) %address.local470, align 2, !dbg !13
  %address.local472 = or disjoint i64 %address.local407, 35, !dbg !13
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local472, !dbg !13
  store half %1523, ptr addrspace(1) %address.local473, align 2, !dbg !13
  %address.local475 = or disjoint i64 %address.local407, 48, !dbg !13
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local475, !dbg !13
  store half %1521, ptr addrspace(1) %address.local476, align 2, !dbg !13
  %address.local478 = or disjoint i64 %address.local407, 49, !dbg !13
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local478, !dbg !13
  store half %1520, ptr addrspace(1) %address.local479, align 2, !dbg !13
  %address.local481 = or disjoint i64 %address.local407, 50, !dbg !13
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local481, !dbg !13
  store half %1519, ptr addrspace(1) %address.local482, align 2, !dbg !13
  %address.local484 = or i64 %1505, 51, !dbg !13
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local484, !dbg !13
  store half %1518, ptr addrspace(1) %address.local485, align 2, !dbg !13
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1527, i64 %address.local431, !dbg !13
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local463, !dbg !13
  store half %1516, ptr addrspace(1) %address.local490, align 2, !dbg !13
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local466, !dbg !13
  store half %1515, ptr addrspace(1) %address.local492, align 2, !dbg !13
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local469, !dbg !13
  store half %1514, ptr addrspace(1) %address.local494, align 2, !dbg !13
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local472, !dbg !13
  store half %1513, ptr addrspace(1) %address.local496, align 2, !dbg !13
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local475, !dbg !13
  store half %1511, ptr addrspace(1) %address.local498, align 2, !dbg !13
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local478, !dbg !13
  store half %1510, ptr addrspace(1) %address.local500, align 2, !dbg !13
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local481, !dbg !13
  store half %1509, ptr addrspace(1) %address.local502, align 2, !dbg !13
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local484, !dbg !13
  store half %1508, ptr addrspace(1) %address.local504, align 2, !dbg !13
  %normalized86 = fmul <4 x float> %row.reciprocal, %1383, !dbg !27
  %1528 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !28
  %1529 = extractelement <4 x half> %1528, i64 3, !dbg !28
  %1530 = extractelement <4 x half> %1528, i64 2, !dbg !28
  %1531 = extractelement <4 x half> %1528, i64 1, !dbg !28
  %1532 = extractelement <4 x half> %1528, i64 0, !dbg !28
  %normalized87 = fmul <4 x float> %row.reciprocal, %1258, !dbg !27
  %1533 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !28
  %1534 = extractelement <4 x half> %1533, i64 3, !dbg !28
  %1535 = extractelement <4 x half> %1533, i64 2, !dbg !28
  %1536 = extractelement <4 x half> %1533, i64 1, !dbg !28
  %1537 = extractelement <4 x half> %1533, i64 0, !dbg !28
  %normalized89 = fmul <4 x float> %row.reciprocal76, %1135, !dbg !27
  %1538 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !28
  %1539 = extractelement <4 x half> %1538, i64 3, !dbg !28
  %1540 = extractelement <4 x half> %1538, i64 2, !dbg !28
  %1541 = extractelement <4 x half> %1538, i64 1, !dbg !28
  %1542 = extractelement <4 x half> %1538, i64 0, !dbg !28
  %normalized90 = fmul <4 x float> %row.reciprocal76, %1010, !dbg !27
  %1543 = fptrunc <4 x float> %normalized90 to <4 x half>, !dbg !28
  %1544 = extractelement <4 x half> %1543, i64 3, !dbg !28
  %1545 = extractelement <4 x half> %1543, i64 2, !dbg !28
  %1546 = extractelement <4 x half> %1543, i64 1, !dbg !28
  %1547 = extractelement <4 x half> %1543, i64 0, !dbg !28
  %address.local519 = or disjoint i64 %address.local407, 64, !dbg !13
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local519, !dbg !13
  store half %1547, ptr addrspace(1) %address.local520, align 2, !dbg !13
  %address.local522 = or disjoint i64 %address.local407, 65, !dbg !13
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local522, !dbg !13
  store half %1546, ptr addrspace(1) %address.local523, align 2, !dbg !13
  %address.local525 = or disjoint i64 %address.local407, 66, !dbg !13
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local525, !dbg !13
  store half %1545, ptr addrspace(1) %address.local526, align 2, !dbg !13
  %address.local528 = or disjoint i64 %address.local407, 67, !dbg !13
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local528, !dbg !13
  store half %1544, ptr addrspace(1) %address.local529, align 2, !dbg !13
  %address.local531 = or disjoint i64 %address.local407, 80, !dbg !13
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local531, !dbg !13
  store half %1542, ptr addrspace(1) %address.local532, align 2, !dbg !13
  %address.local534 = or disjoint i64 %address.local407, 81, !dbg !13
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local534, !dbg !13
  store half %1541, ptr addrspace(1) %address.local535, align 2, !dbg !13
  %address.local537 = or disjoint i64 %address.local407, 82, !dbg !13
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local537, !dbg !13
  store half %1540, ptr addrspace(1) %address.local538, align 2, !dbg !13
  %address.local540 = or disjoint i64 %address.local407, 83, !dbg !13
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local540, !dbg !13
  store half %1539, ptr addrspace(1) %address.local541, align 2, !dbg !13
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local519, !dbg !13
  store half %1537, ptr addrspace(1) %address.local546, align 2, !dbg !13
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local522, !dbg !13
  store half %1536, ptr addrspace(1) %address.local548, align 2, !dbg !13
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local525, !dbg !13
  store half %1535, ptr addrspace(1) %address.local550, align 2, !dbg !13
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local528, !dbg !13
  store half %1534, ptr addrspace(1) %address.local552, align 2, !dbg !13
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local531, !dbg !13
  store half %1532, ptr addrspace(1) %address.local554, align 2, !dbg !13
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local534, !dbg !13
  store half %1531, ptr addrspace(1) %address.local556, align 2, !dbg !13
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local537, !dbg !13
  store half %1530, ptr addrspace(1) %address.local558, align 2, !dbg !13
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local540, !dbg !13
  store half %1529, ptr addrspace(1) %address.local560, align 2, !dbg !13
  %normalized92 = fmul <4 x float> %row.reciprocal, %1350, !dbg !27
  %1548 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !28
  %1549 = extractelement <4 x half> %1548, i64 3, !dbg !28
  %1550 = extractelement <4 x half> %1548, i64 2, !dbg !28
  %1551 = extractelement <4 x half> %1548, i64 1, !dbg !28
  %1552 = extractelement <4 x half> %1548, i64 0, !dbg !28
  %normalized93 = fmul <4 x float> %row.reciprocal, %1224, !dbg !27
  %1553 = fptrunc <4 x float> %normalized93 to <4 x half>, !dbg !28
  %1554 = extractelement <4 x half> %1553, i64 3, !dbg !28
  %1555 = extractelement <4 x half> %1553, i64 2, !dbg !28
  %1556 = extractelement <4 x half> %1553, i64 1, !dbg !28
  %1557 = extractelement <4 x half> %1553, i64 0, !dbg !28
  %normalized95 = fmul <4 x float> %row.reciprocal76, %1102, !dbg !27
  %1558 = fptrunc <4 x float> %normalized95 to <4 x half>, !dbg !28
  %1559 = extractelement <4 x half> %1558, i64 3, !dbg !28
  %1560 = extractelement <4 x half> %1558, i64 2, !dbg !28
  %1561 = extractelement <4 x half> %1558, i64 1, !dbg !28
  %1562 = extractelement <4 x half> %1558, i64 0, !dbg !28
  %normalized96 = fmul <4 x float> %row.reciprocal76, %976, !dbg !27
  %1563 = fptrunc <4 x float> %normalized96 to <4 x half>, !dbg !28
  %1564 = extractelement <4 x half> %1563, i64 3, !dbg !28
  %1565 = extractelement <4 x half> %1563, i64 2, !dbg !28
  %1566 = extractelement <4 x half> %1563, i64 1, !dbg !28
  %1567 = extractelement <4 x half> %1563, i64 0, !dbg !28
  %address.local575 = or disjoint i64 %address.local407, 96, !dbg !13
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local575, !dbg !13
  store half %1567, ptr addrspace(1) %address.local576, align 2, !dbg !13
  %address.local578 = or disjoint i64 %address.local407, 97, !dbg !13
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local578, !dbg !13
  store half %1566, ptr addrspace(1) %address.local579, align 2, !dbg !13
  %address.local581 = or disjoint i64 %address.local407, 98, !dbg !13
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local581, !dbg !13
  store half %1565, ptr addrspace(1) %address.local582, align 2, !dbg !13
  %address.local584 = or disjoint i64 %address.local407, 99, !dbg !13
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local584, !dbg !13
  store half %1564, ptr addrspace(1) %address.local585, align 2, !dbg !13
  %address.local587 = or disjoint i64 %address.local407, 112, !dbg !13
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local587, !dbg !13
  store half %1562, ptr addrspace(1) %address.local588, align 2, !dbg !13
  %address.local590 = or disjoint i64 %address.local407, 113, !dbg !13
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local590, !dbg !13
  store half %1561, ptr addrspace(1) %address.local591, align 2, !dbg !13
  %address.local593 = or disjoint i64 %address.local407, 114, !dbg !13
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local593, !dbg !13
  store half %1560, ptr addrspace(1) %address.local594, align 2, !dbg !13
  %address.local596 = or i64 %1505, 115, !dbg !13
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local596, !dbg !13
  store half %1559, ptr addrspace(1) %address.local597, align 2, !dbg !13
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local575, !dbg !13
  store half %1557, ptr addrspace(1) %address.local602, align 2, !dbg !13
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local578, !dbg !13
  store half %1556, ptr addrspace(1) %address.local604, align 2, !dbg !13
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local581, !dbg !13
  store half %1555, ptr addrspace(1) %address.local606, align 2, !dbg !13
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local584, !dbg !13
  store half %1554, ptr addrspace(1) %address.local608, align 2, !dbg !13
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local587, !dbg !13
  store half %1552, ptr addrspace(1) %address.local610, align 2, !dbg !13
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local590, !dbg !13
  store half %1551, ptr addrspace(1) %address.local612, align 2, !dbg !13
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local593, !dbg !13
  store half %1550, ptr addrspace(1) %address.local614, align 2, !dbg !13
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local596, !dbg !13
  store half %1549, ptr addrspace(1) %address.local616, align 2, !dbg !13
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.or.v4i32(<4 x i32>) #8

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #6 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }
attributes #12 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_input_sd16.mlir", directory: "3rd/deepgengraph/test")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 22, type: !4, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 22, column: 3, scope: !3)
!8 = !DILocation(line: 46, column: 17, scope: !3)
!9 = !DILocation(line: 66, column: 22, scope: !3)
!10 = !DILocation(line: 68, column: 24, scope: !3)
!11 = !{i32 0, i32 256}
!12 = !DILocation(line: 56, column: 24, scope: !3)
!13 = !DILocation(line: 99, column: 9, scope: !3)
!14 = !DILocation(line: 76, column: 23, scope: !3)
!15 = !DILocation(line: 67, column: 22, scope: !3)
!16 = !DILocation(line: 59, column: 17, scope: !3)
!17 = !DILocation(line: 69, column: 26, scope: !3)
!18 = !DILocation(line: 70, column: 23, scope: !3)
!19 = !DILocation(line: 92, column: 20, scope: !3)
!20 = !DILocation(line: 73, column: 19, scope: !3)
!21 = !DILocation(line: 77, column: 26, scope: !3)
!22 = !DILocation(line: 85, column: 22, scope: !3)
!23 = !DILocation(line: 86, column: 20, scope: !3)
!24 = !DILocation(line: 87, column: 16, scope: !3)
!25 = !DILocation(line: 93, column: 23, scope: !3)
!26 = !DILocation(line: 94, column: 20, scope: !3)
!27 = !DILocation(line: 97, column: 23, scope: !3)
!28 = !DILocation(line: 98, column: 19, scope: !3)
