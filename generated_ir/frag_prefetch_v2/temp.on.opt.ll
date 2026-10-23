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
  %.not30 = icmp slt i32 %5, 0, !dbg !12
  br i1 %.not30, label %.._crit_edge_crit_edge, label %.lr.ph, !dbg !12

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !12
  %.pre56 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !12
  %.pre58 = add nuw nsw i32 %.pre56, 64, !dbg !12
  %.pre60 = xor i32 %.pre56, 16, !dbg !12
  %.pre62 = and i32 %.pre58, 192, !dbg !12
  %.pre64 = xor i32 %.pre56, 32, !dbg !12
  %.pre66 = add nsw i64 %16, %9, !dbg !13
  %.pre67 = shl nuw nsw i64 %15, 20, !dbg !13
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
  %38 = lshr i64 %37, 32
  %compare.high = trunc nuw i64 %38 to i32
  %compare.low.base = trunc i64 %37 to i32
  %39 = add nsw i64 %35, 65
  %40 = lshr i64 %39, 32
  %compare.high125 = trunc nuw i64 %40 to i32
  %compare.low.base129 = trunc i64 %39 to i32
  %41 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !14
  %42 = shufflevector <32 x i32> %41, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !14
  %43 = insertelement <16 x i32> poison, i32 %compare.low.base129, i64 0, !dbg !14
  %44 = shufflevector <16 x i32> %43, <16 x i32> poison, <16 x i32> zeroinitializer, !dbg !14
  br label %45, !dbg !12

45:                                               ; preds = %.lr.ph, %448
  %46 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1052, %448 ]
  %47 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1048, %448 ]
  %48 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1044, %448 ]
  %49 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %744, %448 ]
  %50 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1040, %448 ]
  %51 = phi i64 [ 0, %.lr.ph ], [ %1053, %448 ]
  %52 = shl nuw nsw i64 %51, 7, !dbg !12
  %53 = lshr i64 %51, 5, !dbg !15
  %54 = shl i64 %53, 20
  %55 = shl i64 %51, 15
  %.idx74 = sub i64 %55, %54
  br label %56, !dbg !16

56:                                               ; preds = %45, %56
  %57 = phi [2 x <32 x float>] [ zeroinitializer, %45 ], [ %207, %56 ]
  %58 = phi i64 [ 0, %45 ], [ %208, %56 ]
  %59 = shl nuw nsw i64 %58, 5, !dbg !16
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
  %.idx15 = shl nuw nsw i64 %58, 6, !dbg !9
  %60 = add nuw nsw i64 %.idx15, %21, !dbg !9
  %address.local143.idx = shl nuw nsw i64 %address.local142, 1, !dbg !9
  %61 = add nuw nsw i64 %address.local143.idx, %60, !dbg !9
  %dma.buffer.offset623 = trunc i64 %61 to i32, !dbg !9
  %address.local144 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local145 = shl nuw nsw i64 %address.local144, 3, !dbg !9
  %address.local146 = and i64 %address.local145, 3584, !dbg !9
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !9
  %address.local152 = lshr i64 %address.local144, 4, !dbg !9
  %address.local153 = and i64 %address.local152, 24, !dbg !9
  %address.local154 = or disjoint i64 %address.local153, %address.local139, !dbg !9
  %address.local155.idx = shl nuw nsw i64 %address.local154, 1, !dbg !9
  %62 = add nuw nsw i64 %address.local155.idx, %60, !dbg !9
  %dma.buffer.offset620 = trunc i64 %62 to i32, !dbg !9
  %63 = shl nuw nsw i64 %58, 10, !dbg !15
  %64 = add nuw nsw i64 %63, %51, !dbg !15
  %65 = lshr i64 %64, 12, !dbg !15
  %66 = add nsw i64 %65, %8, !dbg !15
  %67 = add nuw nsw i64 %59, %53, !dbg !15
  %68 = and i64 %67, 127, !dbg !15
  %69 = shl i64 %66, 19, !dbg !15
  %70 = and i64 %69, 16252928, !dbg !15
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local134, !dbg !15
  %.idx1719 = or disjoint i64 %70, %68, !dbg !15
  %71 = add nuw nsw i64 %address.local142, %.idx1719, !dbg !15
  %72 = shl nuw nsw i64 %71, 1, !dbg !15
  %73 = add i64 %72, %.idx74, !dbg !15
  %dma.buffer.offset617 = trunc i64 %73 to i32, !dbg !15
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local146, !dbg !15
  %74 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local135, i64 0, !dbg !9
  %75 = insertelement <4 x ptr addrspace(3)> %74, ptr addrspace(3) %address.local147, i64 1, !dbg !9
  %76 = insertelement <4 x ptr addrspace(3)> %75, ptr addrspace(3) %address.local156, i64 2, !dbg !9
  %77 = insertelement <4 x ptr addrspace(3)> %76, ptr addrspace(3) %address.local160, i64 3, !dbg !9
  %78 = ptrtoint <4 x ptr addrspace(3)> %77 to <4 x i64>, !dbg !9
  %79 = trunc <4 x i64> %78 to <4 x i32>, !dbg !9
  %80 = extractelement <4 x i32> %79, i64 0, !dbg !9
  %81 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !9
  %dma.buffer.lds625 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %81), !dbg !9
  %82 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds625, i32 %dma.buffer.offset623, <4 x i32> %dma.buffer.rsrc624) #10, !dbg !9
  %83 = extractelement <4 x i32> %79, i64 1, !dbg !9
  %84 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !9
  %dma.buffer.lds622 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !9
  %85 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds622, i32 %dma.buffer.offset620, <4 x i32> %dma.buffer.rsrc624) #10, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %86 = extractelement <4 x i32> %79, i64 2, !dbg !15
  %87 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %86), !dbg !15
  %dma.buffer.lds619 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %87), !dbg !15
  %88 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds619, i32 %dma.buffer.offset617, <4 x i32> %dma.buffer.rsrc618) #10, !dbg !15
  %89 = extractelement <4 x i32> %79, i64 3, !dbg !15
  %90 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %89), !dbg !15
  %91 = add nuw nsw i64 %address.local154, %.idx1719, !dbg !15
  %92 = shl nuw nsw i64 %91, 1, !dbg !15
  %93 = add i64 %92, %.idx74, !dbg !15
  %dma.buffer.offset = trunc i64 %93 to i32, !dbg !15
  %dma.buffer.lds = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %90), !dbg !15
  %94 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc618) #10, !dbg !15
  %address.local164 = lshr i32 %address.thread, 2, !dbg !10
  %address.local166 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local169 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local172 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %95 = insertelement <4 x i32> poison, i32 %address.local169, i64 0, !dbg !10
  %96 = insertelement <4 x i32> %95, i32 %address.local166, i64 1, !dbg !10
  %97 = insertelement <4 x i32> %96, i32 %address.local164, i64 2, !dbg !10
  %98 = insertelement <4 x i32> %97, i32 %address.local172, i64 3, !dbg !10
  %99 = and <4 x i32> %98, <i32 120, i32 384, i32 4, i32 1024>, !dbg !10
  %100 = tail call i32 @llvm.vector.reduce.or.v4i32(<4 x i32> %99), !dbg !10
  %address.local175 = zext nneg i32 %100 to i64, !dbg !10
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !10
  %101 = load <4 x half>, ptr addrspace(3) %address.local176, align 8, !dbg !10
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local176, i64 4096, !dbg !10
  %102 = load <4 x half>, ptr addrspace(3) %address.local178, align 8, !dbg !10
  %address.local180 = getelementptr i8, ptr addrspace(3) %address.local176, i64 1024, !dbg !10
  %103 = load <4 x half>, ptr addrspace(3) %address.local180, align 8, !dbg !10
  %104 = fmul <4 x half> %101, splat (half 1.275630e-01), !dbg !10
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local176, i64 5120, !dbg !10
  %105 = load <4 x half>, ptr addrspace(3) %address.local182, align 8, !dbg !10
  %106 = fmul <4 x half> %102, splat (half 1.275630e-01), !dbg !10
  %107 = fmul <4 x half> %103, splat (half 1.275630e-01), !dbg !10
  %108 = fmul <4 x half> %105, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local186 = and i32 %address.local166, 24, !dbg !17
  %address.local191 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local192 = and i32 %address.local191, 384, !dbg !17
  %109 = or disjoint i32 %address.local192, %address.local186, !dbg !17
  %110 = extractelement <4 x i32> %99, i64 2, !dbg !17
  %111 = or disjoint i32 %109, %110, !dbg !17
  %112 = extractelement <4 x i32> %99, i64 3, !dbg !17
  %address.local193 = or disjoint i32 %111, %112, !dbg !17
  %address.local194 = zext nneg i32 %address.local193 to i64, !dbg !17
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local194, !dbg !17
  %113 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !17
  %114 = extractvalue [2 x <32 x float>] %57, 0, !dbg !18
  %115 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %address.local209 = getelementptr i8, ptr addrspace(3) %address.local195, i64 4096, !dbg !17
  %116 = load <4 x half>, ptr addrspace(3) %address.local209, align 8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %113, <4 x float> %115) #9, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %118 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %119 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %116, <4 x float> %117) #9, !dbg !17
  %120 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !18
  %121 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !18
  %122 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !18
  %123 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !18
  %124 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !18
  %125 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !18
  %126 = extractvalue [2 x <32 x float>] %57, 1, !dbg !18
  %127 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %128 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %129 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !18
  %130 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !18
  %131 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !18
  %132 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !18
  %133 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !18
  %134 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !18
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local138, !dbg !17
  %address.local212 = lshr i64 %address.local, 2, !dbg !17
  %address.local213 = mul nsw i64 %address.local212, -512, !dbg !17
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !17
  %address.local215 = shl nuw nsw i64 %address.local, 1, !dbg !17
  %address.local216 = and i64 %address.local215, 24, !dbg !17
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !17
  %address.local219 = and i64 %address.local212, 12, !dbg !10
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local219, !dbg !17
  %135 = and i64 %address.local, 32, !dbg !10
  %.not23 = icmp eq i64 %135, 0, !dbg !10
  %address.local223 = select i1 %.not23, i64 0, i64 1016, !dbg !10
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local223, !dbg !17
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 64, !dbg !17
  %136 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !17
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local219, !dbg !17
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local223, !dbg !17
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local213, !dbg !17
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local216, !dbg !17
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4160, !dbg !17
  %137 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !17
  %138 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %136, <4 x float> %118) #9, !dbg !17
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !17
  %139 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !17
  %140 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %137, <4 x float> %138) #9, !dbg !17
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4224, !dbg !17
  %141 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !17
  %142 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %139, <4 x float> %120) #9, !dbg !17
  %143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %141, <4 x float> %142) #9, !dbg !17
  %address.local249 = getelementptr i8, ptr addrspace(3) %address.local224, i64 192, !dbg !17
  %144 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !17
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4288, !dbg !17
  %145 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !17
  %146 = shufflevector <4 x float> %143, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %144, <4 x float> %121) #9, !dbg !17
  %148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %145, <4 x float> %147) #9, !dbg !17
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local195, i64 1024, !dbg !17
  %149 = load <4 x half>, ptr addrspace(3) %address.local257, align 8, !dbg !17
  %address.local259 = getelementptr i8, ptr addrspace(3) %address.local195, i64 5120, !dbg !17
  %150 = load <4 x half>, ptr addrspace(3) %address.local259, align 8, !dbg !17
  %151 = shufflevector <4 x float> %148, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %149, <4 x float> %122) #9, !dbg !17
  %153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %150, <4 x float> %152) #9, !dbg !17
  %address.local261 = getelementptr i8, ptr addrspace(3) %address.local211, i64 1024, !dbg !17
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local213, !dbg !17
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) %address.local262, i64 %address.local216, !dbg !17
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local219, !dbg !17
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local223, !dbg !17
  %address.local266 = getelementptr i8, ptr addrspace(3) %address.local265, i64 64, !dbg !17
  %154 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !17
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local219, !dbg !17
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local223, !dbg !17
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local213, !dbg !17
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local216, !dbg !17
  %address.local272 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4160, !dbg !17
  %155 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !17
  %156 = shufflevector <4 x float> %153, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %154, <4 x float> %123) #9, !dbg !17
  %158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %155, <4 x float> %157) #9, !dbg !17
  %address.local279 = getelementptr i8, ptr addrspace(3) %address.local265, i64 128, !dbg !17
  %159 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !17
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4224, !dbg !17
  %160 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !17
  %161 = shufflevector <4 x float> %158, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %159, <4 x float> %124) #9, !dbg !17
  %163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %160, <4 x float> %162) #9, !dbg !17
  %address.local292 = getelementptr i8, ptr addrspace(3) %address.local265, i64 192, !dbg !17
  %164 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !17
  %address.local298 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4288, !dbg !17
  %165 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !17
  %166 = shufflevector <4 x float> %163, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %164, <4 x float> %125) #9, !dbg !17
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %165, <4 x float> %167) #9, !dbg !17
  %169 = shufflevector <4 x float> %168, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %113, <4 x float> %127) #9, !dbg !17
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %116, <4 x float> %170) #9, !dbg !17
  %172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %136, <4 x float> %128) #9, !dbg !17
  %173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %137, <4 x float> %172) #9, !dbg !17
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %139, <4 x float> %129) #9, !dbg !17
  %175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %141, <4 x float> %174) #9, !dbg !17
  %176 = shufflevector <4 x float> %175, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %144, <4 x float> %130) #9, !dbg !17
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %145, <4 x float> %177) #9, !dbg !17
  %179 = shufflevector <4 x float> %178, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %149, <4 x float> %131) #9, !dbg !17
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %150, <4 x float> %180) #9, !dbg !17
  %182 = shufflevector <4 x float> %181, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %154, <4 x float> %132) #9, !dbg !17
  %184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %155, <4 x float> %183) #9, !dbg !17
  %185 = shufflevector <4 x float> %184, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %159, <4 x float> %133) #9, !dbg !17
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %160, <4 x float> %186) #9, !dbg !17
  %188 = shufflevector <4 x float> %187, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %164, <4 x float> %134) #9, !dbg !17
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %165, <4 x float> %189) #9, !dbg !17
  %191 = shufflevector <4 x float> %190, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !18
  %192 = shufflevector <4 x float> %119, <4 x float> %140, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %193 = shufflevector <32 x float> %192, <32 x float> %146, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %194 = shufflevector <32 x float> %193, <32 x float> %151, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %195 = shufflevector <32 x float> %194, <32 x float> %156, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %196 = shufflevector <32 x float> %195, <32 x float> %161, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %197 = shufflevector <32 x float> %196, <32 x float> %166, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %198 = shufflevector <32 x float> %197, <32 x float> %169, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !16
  %199 = insertvalue [2 x <32 x float>] poison, <32 x float> %198, 0, !dbg !16
  %200 = shufflevector <4 x float> %171, <4 x float> %173, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %201 = shufflevector <32 x float> %200, <32 x float> %176, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %202 = shufflevector <32 x float> %201, <32 x float> %179, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %203 = shufflevector <32 x float> %202, <32 x float> %182, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %204 = shufflevector <32 x float> %203, <32 x float> %185, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %205 = shufflevector <32 x float> %204, <32 x float> %188, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %206 = shufflevector <32 x float> %205, <32 x float> %191, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !16
  %207 = insertvalue [2 x <32 x float>] %199, <32 x float> %206, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %208 = add nuw nsw i64 %58, 1, !dbg !16
  %209 = icmp samesign ult i64 %58, 3, !dbg !16
  br i1 %209, label %56, label %210, !dbg !16

210:                                              ; preds = %56
  %211 = add nsw i64 %53, %8, !dbg !19
  %212 = shl i64 %211, 20, !dbg !19
  %213 = and i64 %212, 32505856, !dbg !19
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
  %214 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local305), !dbg !19
  %address.local306 = shl nuw nsw i64 %address.local300, 6, !dbg !19
  %address.local307 = and i64 %address.local306, 16128, !dbg !19
  %215 = add i64 %213, %.idx74, !dbg !19
  %address.local309 = shl nuw nsw i64 %address.local300, 4, !dbg !19
  %address.local311.idx = and i64 %address.local309, 48, !dbg !19
  %216 = or disjoint i64 %215, %address.local307, !dbg !19
  %217 = or disjoint i64 %216, %address.local311.idx, !dbg !19
  %dma.buffer.offset647 = trunc i64 %217 to i32, !dbg !19
  %dma.buffer.lds649 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %214), !dbg !19
  %218 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds649, i32 %dma.buffer.offset647, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %address.local312 = or disjoint i64 %address.local300, 256, !dbg !9
  %address.local313 = shl nuw nsw i64 %address.local312, 3, !dbg !9
  %address.local314 = and i64 %address.local313, 3584, !dbg !9
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local314, !dbg !9
  %address.local316 = ptrtoint ptr addrspace(3) %address.local315 to i64, !dbg !9
  %address.local317 = trunc i64 %address.local316 to i32, !dbg !9
  %219 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local317), !dbg !19
  %address.local318 = shl nuw nsw i64 %address.local312, 6, !dbg !19
  %address.local319 = and i64 %address.local318, 32512, !dbg !19
  %220 = or disjoint i64 %215, %address.local319, !dbg !19
  %221 = or disjoint i64 %220, %address.local311.idx, !dbg !19
  %dma.buffer.offset644 = trunc i64 %221 to i32, !dbg !19
  %dma.buffer.lds646 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %219), !dbg !19
  %222 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds646, i32 %dma.buffer.offset644, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %223 = or disjoint i64 %52, 127, !dbg !20
  %.not35 = icmp ult i64 %9, %223, !dbg !20
  br i1 %.not35, label %278, label %224, !dbg !20

224:                                              ; preds = %210
  %225 = shufflevector <4 x float> %143, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %226 = shufflevector <32 x float> %192, <32 x float> %225, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %227 = shufflevector <4 x float> %143, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %228 = shufflevector <32 x float> %226, <32 x float> %227, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %229 = shufflevector <4 x float> %148, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %230 = shufflevector <32 x float> %228, <32 x float> %229, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %231 = shufflevector <4 x float> %153, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %232 = shufflevector <32 x float> %230, <32 x float> %231, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %233 = shufflevector <4 x float> %158, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %234 = shufflevector <32 x float> %232, <32 x float> %233, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %235 = shufflevector <4 x float> %163, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %236 = shufflevector <32 x float> %234, <32 x float> %235, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %237 = shufflevector <4 x float> %168, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %238 = shufflevector <32 x float> %236, <32 x float> %237, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !20
  %239 = fadd <32 x float> %238, zeroinitializer, !dbg !20
  %240 = shufflevector <4 x float> %171, <4 x float> %173, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %241 = shufflevector <4 x float> %175, <4 x float> poison, <16 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %242 = shufflevector <16 x float> %240, <16 x float> %241, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %243 = shufflevector <4 x float> %175, <4 x float> poison, <16 x i32> <i32 poison, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %244 = shufflevector <16 x float> %242, <16 x float> %243, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %245 = shufflevector <4 x float> %178, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %246 = shufflevector <16 x float> %244, <16 x float> %245, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !20
  %247 = fadd <16 x float> %246, zeroinitializer, !dbg !20
  %248 = extractelement <4 x float> %181, i64 0, !dbg !20
  %249 = fadd float %248, 0.000000e+00, !dbg !20
  %250 = extractelement <4 x float> %181, i64 1, !dbg !20
  %251 = fadd float %250, 0.000000e+00, !dbg !20
  %252 = extractelement <4 x float> %181, i64 2, !dbg !20
  %253 = fadd float %252, 0.000000e+00, !dbg !20
  %254 = extractelement <4 x float> %181, i64 3, !dbg !20
  %255 = fadd float %254, 0.000000e+00, !dbg !20
  %256 = extractelement <4 x float> %184, i64 0, !dbg !20
  %257 = fadd float %256, 0.000000e+00, !dbg !20
  %258 = extractelement <4 x float> %184, i64 1, !dbg !20
  %259 = fadd float %258, 0.000000e+00, !dbg !20
  %260 = extractelement <4 x float> %184, i64 2, !dbg !20
  %261 = fadd float %260, 0.000000e+00, !dbg !20
  %262 = extractelement <4 x float> %184, i64 3, !dbg !20
  %263 = fadd float %262, 0.000000e+00, !dbg !20
  %264 = extractelement <4 x float> %187, i64 0, !dbg !20
  %265 = fadd float %264, 0.000000e+00, !dbg !20
  %266 = extractelement <4 x float> %187, i64 1, !dbg !20
  %267 = fadd float %266, 0.000000e+00, !dbg !20
  %268 = extractelement <4 x float> %187, i64 2, !dbg !20
  %269 = fadd float %268, 0.000000e+00, !dbg !20
  %270 = extractelement <4 x float> %187, i64 3, !dbg !20
  %271 = fadd float %270, 0.000000e+00, !dbg !20
  %272 = extractelement <4 x float> %190, i64 0, !dbg !20
  %273 = fadd float %272, 0.000000e+00, !dbg !20
  %274 = extractelement <4 x float> %190, i64 1, !dbg !20
  %275 = fadd float %274, 0.000000e+00, !dbg !20
  %276 = extractelement <4 x float> %190, i64 2, !dbg !20
  %277 = fadd float %276, 0.000000e+00, !dbg !20
  br label %448, !dbg !20

278:                                              ; preds = %210
  %279 = or disjoint i64 %52, %36, !dbg !20
  %280 = lshr i64 %51, 25, !dbg !14
  %compare.high1 = trunc nuw i64 %280 to i32, !dbg !14
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1, !dbg !14
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1, !dbg !14
  %281 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !14
  %282 = shufflevector <32 x i1> %281, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !14
  %283 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !14
  %284 = shufflevector <32 x i1> %283, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !14
  %285 = shufflevector <4 x float> %143, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %286 = shufflevector <32 x float> %192, <32 x float> %285, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %287 = shufflevector <4 x float> %143, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %288 = shufflevector <32 x float> %286, <32 x float> %287, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %289 = shufflevector <4 x float> %148, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %290 = shufflevector <32 x float> %288, <32 x float> %289, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %291 = shufflevector <4 x float> %153, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %292 = shufflevector <32 x float> %290, <32 x float> %291, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %293 = shufflevector <4 x float> %158, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %294 = shufflevector <32 x float> %292, <32 x float> %293, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %295 = shufflevector <4 x float> %163, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %296 = shufflevector <32 x float> %294, <32 x float> %295, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %297 = shufflevector <4 x float> %168, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %298 = shufflevector <32 x float> %296, <32 x float> %297, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !20
  %compare.low.base2 = trunc i64 %279 to i32, !dbg !14
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 1) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base5 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 2) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base9 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 3) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base13 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 4) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base17 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 5) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base21 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 6) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base25 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 7) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base29 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 8) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base33 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 9) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base37 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 10) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base41 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 11) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base45 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 12) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base49 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 13) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base53 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 14) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base57 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 15) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base61 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 16) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base65 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 17) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base69 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 18) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base73 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 19) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base77 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 20) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base81 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 21) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base85 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 22) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base89 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 23) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base93 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 24) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base97 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 25) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base101 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 26) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base105 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 27) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base109 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 28) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base113 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 29) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base117 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 30) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base121 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 31) #9, !dbg !14, !frisk.register_lifetimes !5
  %299 = insertelement <32 x i32> poison, i32 %compare.low.base2, i64 0, !dbg !14
  %300 = insertelement <32 x i32> %299, i32 %compare.local.base, i64 1, !dbg !14
  %301 = insertelement <32 x i32> %300, i32 %compare.local.base5, i64 2, !dbg !14
  %302 = insertelement <32 x i32> %301, i32 %compare.local.base9, i64 3, !dbg !14
  %303 = insertelement <32 x i32> %302, i32 %compare.local.base13, i64 4, !dbg !14
  %304 = insertelement <32 x i32> %303, i32 %compare.local.base17, i64 5, !dbg !14
  %305 = insertelement <32 x i32> %304, i32 %compare.local.base21, i64 6, !dbg !14
  %306 = insertelement <32 x i32> %305, i32 %compare.local.base25, i64 7, !dbg !14
  %307 = insertelement <32 x i32> %306, i32 %compare.local.base29, i64 8, !dbg !14
  %308 = insertelement <32 x i32> %307, i32 %compare.local.base33, i64 9, !dbg !14
  %309 = insertelement <32 x i32> %308, i32 %compare.local.base37, i64 10, !dbg !14
  %310 = insertelement <32 x i32> %309, i32 %compare.local.base41, i64 11, !dbg !14
  %311 = insertelement <32 x i32> %310, i32 %compare.local.base45, i64 12, !dbg !14
  %312 = insertelement <32 x i32> %311, i32 %compare.local.base49, i64 13, !dbg !14
  %313 = insertelement <32 x i32> %312, i32 %compare.local.base53, i64 14, !dbg !14
  %314 = insertelement <32 x i32> %313, i32 %compare.local.base57, i64 15, !dbg !14
  %315 = insertelement <32 x i32> %314, i32 %compare.local.base61, i64 16, !dbg !14
  %316 = insertelement <32 x i32> %315, i32 %compare.local.base65, i64 17, !dbg !14
  %317 = insertelement <32 x i32> %316, i32 %compare.local.base69, i64 18, !dbg !14
  %318 = insertelement <32 x i32> %317, i32 %compare.local.base73, i64 19, !dbg !14
  %319 = insertelement <32 x i32> %318, i32 %compare.local.base77, i64 20, !dbg !14
  %320 = insertelement <32 x i32> %319, i32 %compare.local.base81, i64 21, !dbg !14
  %321 = insertelement <32 x i32> %320, i32 %compare.local.base85, i64 22, !dbg !14
  %322 = insertelement <32 x i32> %321, i32 %compare.local.base89, i64 23, !dbg !14
  %323 = insertelement <32 x i32> %322, i32 %compare.local.base93, i64 24, !dbg !14
  %324 = insertelement <32 x i32> %323, i32 %compare.local.base97, i64 25, !dbg !14
  %325 = insertelement <32 x i32> %324, i32 %compare.local.base101, i64 26, !dbg !14
  %326 = insertelement <32 x i32> %325, i32 %compare.local.base105, i64 27, !dbg !14
  %327 = insertelement <32 x i32> %326, i32 %compare.local.base109, i64 28, !dbg !14
  %328 = insertelement <32 x i32> %327, i32 %compare.local.base113, i64 29, !dbg !14
  %329 = insertelement <32 x i32> %328, i32 %compare.local.base117, i64 30, !dbg !14
  %330 = insertelement <32 x i32> %329, i32 %compare.local.base121, i64 31, !dbg !14
  %331 = or <32 x i32> %330, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !14
  %332 = icmp ugt <32 x i32> %42, %331, !dbg !14
  %333 = select <32 x i1> %282, <32 x i1> %332, <32 x i1> %284, !dbg !14
  %334 = select <32 x i1> %333, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !21
  %335 = fadd <32 x float> %334, %298, !dbg !20
  %compare.high.equal127 = icmp eq i32 %compare.high125, %compare.high1, !dbg !14
  %compare.high.order128 = icmp ugt i32 %compare.high125, %compare.high1, !dbg !14
  %compare.local.base133 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 32) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base137 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 33) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base141 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 34) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base145 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 35) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base149 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 36) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base153 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 37) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base157 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 38) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base161 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 39) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base165 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 40) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base169 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 41) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base173 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 42) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base177 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 43) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base181 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 44) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base185 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 45) #9, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base189 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 46) #9, !dbg !14, !frisk.register_lifetimes !5
  %336 = insertelement <16 x i32> poison, i32 %compare.low.base2, i64 0, !dbg !14
  %337 = insertelement <16 x i32> %336, i32 %compare.local.base133, i64 1, !dbg !14
  %338 = insertelement <16 x i32> %337, i32 %compare.local.base137, i64 2, !dbg !14
  %339 = insertelement <16 x i32> %338, i32 %compare.local.base141, i64 3, !dbg !14
  %340 = insertelement <16 x i32> %339, i32 %compare.local.base145, i64 4, !dbg !14
  %341 = insertelement <16 x i32> %340, i32 %compare.local.base149, i64 5, !dbg !14
  %342 = insertelement <16 x i32> %341, i32 %compare.local.base153, i64 6, !dbg !14
  %343 = insertelement <16 x i32> %342, i32 %compare.local.base157, i64 7, !dbg !14
  %344 = insertelement <16 x i32> %343, i32 %compare.local.base161, i64 8, !dbg !14
  %345 = insertelement <16 x i32> %344, i32 %compare.local.base165, i64 9, !dbg !14
  %346 = insertelement <16 x i32> %345, i32 %compare.local.base169, i64 10, !dbg !14
  %347 = insertelement <16 x i32> %346, i32 %compare.local.base173, i64 11, !dbg !14
  %348 = insertelement <16 x i32> %347, i32 %compare.local.base177, i64 12, !dbg !14
  %349 = insertelement <16 x i32> %348, i32 %compare.local.base181, i64 13, !dbg !14
  %350 = insertelement <16 x i32> %349, i32 %compare.local.base185, i64 14, !dbg !14
  %351 = insertelement <16 x i32> %350, i32 %compare.local.base189, i64 15, !dbg !14
  %352 = or <16 x i32> %351, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !14
  %353 = icmp ugt <16 x i32> %44, %352, !dbg !14
  %354 = insertelement <16 x i1> poison, i1 %compare.high.equal127, i64 0, !dbg !14
  %355 = shufflevector <16 x i1> %354, <16 x i1> poison, <16 x i32> zeroinitializer, !dbg !14
  %356 = insertelement <16 x i1> poison, i1 %compare.high.order128, i64 0, !dbg !14
  %357 = shufflevector <16 x i1> %356, <16 x i1> poison, <16 x i32> zeroinitializer, !dbg !14
  %358 = select <16 x i1> %355, <16 x i1> %353, <16 x i1> %357, !dbg !14
  %359 = select <16 x i1> %358, <16 x float> zeroinitializer, <16 x float> splat (float -inf), !dbg !21
  %360 = shufflevector <4 x float> %171, <4 x float> %173, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %361 = shufflevector <4 x float> %175, <4 x float> poison, <16 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %362 = shufflevector <16 x float> %360, <16 x float> %361, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %363 = shufflevector <4 x float> %175, <4 x float> poison, <16 x i32> <i32 poison, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %364 = shufflevector <16 x float> %362, <16 x float> %363, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %365 = shufflevector <4 x float> %178, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !20
  %366 = shufflevector <16 x float> %364, <16 x float> %365, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !20
  %367 = fadd <16 x float> %359, %366, !dbg !20
  %compare.local.base193 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 47) #9, !dbg !14, !frisk.register_lifetimes !5
  %368 = extractelement <4 x float> %181, i64 0, !dbg !20
  %compare.local.base197 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 48) #9, !dbg !14, !frisk.register_lifetimes !5
  %369 = extractelement <4 x float> %181, i64 1, !dbg !20
  %compare.local.base201 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 49) #9, !dbg !14, !frisk.register_lifetimes !5
  %370 = extractelement <4 x float> %181, i64 2, !dbg !20
  %compare.local.base205 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 50) #9, !dbg !14, !frisk.register_lifetimes !5
  %371 = extractelement <4 x float> %181, i64 3, !dbg !20
  %compare.local.base209 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 51) #9, !dbg !14, !frisk.register_lifetimes !5
  %372 = extractelement <4 x float> %184, i64 0, !dbg !20
  %compare.local.base213 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 52) #9, !dbg !14, !frisk.register_lifetimes !5
  %373 = extractelement <4 x float> %184, i64 1, !dbg !20
  %compare.local.base217 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 53) #9, !dbg !14, !frisk.register_lifetimes !5
  %374 = extractelement <4 x float> %184, i64 2, !dbg !20
  %compare.local.base221 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 54) #9, !dbg !14, !frisk.register_lifetimes !5
  %375 = extractelement <4 x float> %184, i64 3, !dbg !20
  %compare.local.base225 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 55) #9, !dbg !14, !frisk.register_lifetimes !5
  %376 = extractelement <4 x float> %187, i64 0, !dbg !20
  %compare.local.base229 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 56) #9, !dbg !14, !frisk.register_lifetimes !5
  %377 = extractelement <4 x float> %187, i64 1, !dbg !20
  %compare.local.base233 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 57) #9, !dbg !14, !frisk.register_lifetimes !5
  %378 = extractelement <4 x float> %187, i64 2, !dbg !20
  %compare.local.base237 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 58) #9, !dbg !14, !frisk.register_lifetimes !5
  %379 = extractelement <4 x float> %187, i64 3, !dbg !20
  %compare.local.base241 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 59) #9, !dbg !14, !frisk.register_lifetimes !5
  %380 = extractelement <4 x float> %190, i64 0, !dbg !20
  %compare.local.base245 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 60) #9, !dbg !14, !frisk.register_lifetimes !5
  %381 = extractelement <4 x float> %190, i64 1, !dbg !20
  %compare.local.base249 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 61) #9, !dbg !14, !frisk.register_lifetimes !5
  %382 = extractelement <4 x float> %190, i64 2, !dbg !20
  %compare.local.base253 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 62) #9, !dbg !14, !frisk.register_lifetimes !5
  %383 = insertelement <16 x i32> poison, i32 %compare.local.base253, i64 0, !dbg !14
  %384 = insertelement <16 x i32> %383, i32 %compare.local.base249, i64 1, !dbg !14
  %385 = insertelement <16 x i32> %384, i32 %compare.local.base245, i64 2, !dbg !14
  %386 = insertelement <16 x i32> %385, i32 %compare.local.base241, i64 3, !dbg !14
  %387 = insertelement <16 x i32> %386, i32 %compare.local.base237, i64 4, !dbg !14
  %388 = insertelement <16 x i32> %387, i32 %compare.local.base233, i64 5, !dbg !14
  %389 = insertelement <16 x i32> %388, i32 %compare.local.base229, i64 6, !dbg !14
  %390 = insertelement <16 x i32> %389, i32 %compare.local.base225, i64 7, !dbg !14
  %391 = insertelement <16 x i32> %390, i32 %compare.local.base221, i64 8, !dbg !14
  %392 = insertelement <16 x i32> %391, i32 %compare.local.base217, i64 9, !dbg !14
  %393 = insertelement <16 x i32> %392, i32 %compare.local.base213, i64 10, !dbg !14
  %394 = insertelement <16 x i32> %393, i32 %compare.local.base209, i64 11, !dbg !14
  %395 = insertelement <16 x i32> %394, i32 %compare.local.base205, i64 12, !dbg !14
  %396 = insertelement <16 x i32> %395, i32 %compare.local.base201, i64 13, !dbg !14
  %397 = insertelement <16 x i32> %396, i32 %compare.local.base197, i64 14, !dbg !14
  %398 = insertelement <16 x i32> %397, i32 %compare.local.base193, i64 15, !dbg !14
  %399 = or <16 x i32> %398, <i32 79, i32 78, i32 77, i32 76, i32 75, i32 74, i32 73, i32 72, i32 71, i32 70, i32 69, i32 68, i32 67, i32 66, i32 65, i32 64>, !dbg !14
  %400 = icmp ult <16 x i32> %399, %44, !dbg !14
  %401 = extractelement <16 x i1> %400, i64 15, !dbg !14
  %compare.result196 = select i1 %compare.high.equal127, i1 %401, i1 %compare.high.order128, !dbg !14
  %402 = select i1 %compare.result196, float 0.000000e+00, float -inf, !dbg !21
  %403 = fadd float %368, %402, !dbg !20
  %404 = extractelement <16 x i1> %400, i64 14, !dbg !14
  %compare.result200 = select i1 %compare.high.equal127, i1 %404, i1 %compare.high.order128, !dbg !14
  %405 = select i1 %compare.result200, float 0.000000e+00, float -inf, !dbg !21
  %406 = fadd float %369, %405, !dbg !20
  %407 = extractelement <16 x i1> %400, i64 13, !dbg !14
  %compare.result204 = select i1 %compare.high.equal127, i1 %407, i1 %compare.high.order128, !dbg !14
  %408 = select i1 %compare.result204, float 0.000000e+00, float -inf, !dbg !21
  %409 = fadd float %370, %408, !dbg !20
  %410 = extractelement <16 x i1> %400, i64 12, !dbg !14
  %compare.result208 = select i1 %compare.high.equal127, i1 %410, i1 %compare.high.order128, !dbg !14
  %411 = select i1 %compare.result208, float 0.000000e+00, float -inf, !dbg !21
  %412 = fadd float %371, %411, !dbg !20
  %413 = extractelement <16 x i1> %400, i64 11, !dbg !14
  %compare.result212 = select i1 %compare.high.equal127, i1 %413, i1 %compare.high.order128, !dbg !14
  %414 = select i1 %compare.result212, float 0.000000e+00, float -inf, !dbg !21
  %415 = fadd float %372, %414, !dbg !20
  %416 = extractelement <16 x i1> %400, i64 10, !dbg !14
  %compare.result216 = select i1 %compare.high.equal127, i1 %416, i1 %compare.high.order128, !dbg !14
  %417 = select i1 %compare.result216, float 0.000000e+00, float -inf, !dbg !21
  %418 = fadd float %373, %417, !dbg !20
  %419 = extractelement <16 x i1> %400, i64 9, !dbg !14
  %compare.result220 = select i1 %compare.high.equal127, i1 %419, i1 %compare.high.order128, !dbg !14
  %420 = select i1 %compare.result220, float 0.000000e+00, float -inf, !dbg !21
  %421 = fadd float %374, %420, !dbg !20
  %422 = extractelement <16 x i1> %400, i64 8, !dbg !14
  %compare.result224 = select i1 %compare.high.equal127, i1 %422, i1 %compare.high.order128, !dbg !14
  %423 = select i1 %compare.result224, float 0.000000e+00, float -inf, !dbg !21
  %424 = fadd float %375, %423, !dbg !20
  %425 = extractelement <16 x i1> %400, i64 7, !dbg !14
  %compare.result228 = select i1 %compare.high.equal127, i1 %425, i1 %compare.high.order128, !dbg !14
  %426 = select i1 %compare.result228, float 0.000000e+00, float -inf, !dbg !21
  %427 = fadd float %376, %426, !dbg !20
  %428 = extractelement <16 x i1> %400, i64 6, !dbg !14
  %compare.result232 = select i1 %compare.high.equal127, i1 %428, i1 %compare.high.order128, !dbg !14
  %429 = select i1 %compare.result232, float 0.000000e+00, float -inf, !dbg !21
  %430 = fadd float %377, %429, !dbg !20
  %431 = extractelement <16 x i1> %400, i64 5, !dbg !14
  %compare.result236 = select i1 %compare.high.equal127, i1 %431, i1 %compare.high.order128, !dbg !14
  %432 = select i1 %compare.result236, float 0.000000e+00, float -inf, !dbg !21
  %433 = fadd float %378, %432, !dbg !20
  %434 = extractelement <16 x i1> %400, i64 4, !dbg !14
  %compare.result240 = select i1 %compare.high.equal127, i1 %434, i1 %compare.high.order128, !dbg !14
  %435 = select i1 %compare.result240, float 0.000000e+00, float -inf, !dbg !21
  %436 = fadd float %379, %435, !dbg !20
  %437 = extractelement <16 x i1> %400, i64 3, !dbg !14
  %compare.result244 = select i1 %compare.high.equal127, i1 %437, i1 %compare.high.order128, !dbg !14
  %438 = select i1 %compare.result244, float 0.000000e+00, float -inf, !dbg !21
  %439 = fadd float %380, %438, !dbg !20
  %440 = extractelement <16 x i1> %400, i64 2, !dbg !14
  %compare.result248 = select i1 %compare.high.equal127, i1 %440, i1 %compare.high.order128, !dbg !14
  %441 = select i1 %compare.result248, float 0.000000e+00, float -inf, !dbg !21
  %442 = fadd float %381, %441, !dbg !20
  %443 = extractelement <16 x i1> %400, i64 1, !dbg !14
  %compare.result252 = select i1 %compare.high.equal127, i1 %443, i1 %compare.high.order128, !dbg !14
  %444 = select i1 %compare.result252, float 0.000000e+00, float -inf, !dbg !21
  %445 = fadd float %382, %444, !dbg !20
  %446 = extractelement <16 x i1> %400, i64 0, !dbg !14
  %compare.result256 = select i1 %compare.high.equal127, i1 %446, i1 %compare.high.order128, !dbg !14
  %447 = select i1 %compare.result256, float 0.000000e+00, float -inf, !dbg !21
  br label %448, !dbg !20

448:                                              ; preds = %224, %278
  %.sink186 = phi float [ 0.000000e+00, %224 ], [ %447, %278 ]
  %.sink115 = phi float [ %249, %224 ], [ %403, %278 ]
  %.sink113 = phi float [ %251, %224 ], [ %406, %278 ]
  %.sink112 = phi float [ %253, %224 ], [ %409, %278 ]
  %.sink110 = phi float [ %255, %224 ], [ %412, %278 ]
  %.sink109 = phi float [ %257, %224 ], [ %415, %278 ]
  %.sink107 = phi float [ %259, %224 ], [ %418, %278 ]
  %.sink106 = phi float [ %261, %224 ], [ %421, %278 ]
  %.sink104 = phi float [ %263, %224 ], [ %424, %278 ]
  %.sink103 = phi float [ %265, %224 ], [ %427, %278 ]
  %.sink101 = phi float [ %267, %224 ], [ %430, %278 ]
  %.sink100 = phi float [ %269, %224 ], [ %433, %278 ]
  %.sink98 = phi float [ %271, %224 ], [ %436, %278 ]
  %.sink97 = phi float [ %273, %224 ], [ %439, %278 ]
  %.sink95 = phi float [ %275, %224 ], [ %442, %278 ]
  %.sink94 = phi float [ %277, %224 ], [ %445, %278 ]
  %449 = phi <32 x float> [ %239, %224 ], [ %335, %278 ]
  %450 = phi <16 x float> [ %247, %224 ], [ %367, %278 ]
  %451 = extractelement <4 x float> %190, i64 3, !dbg !20
  %452 = fadd float %451, %.sink186, !dbg !20
  %453 = extractelement <32 x float> %449, i64 0, !dbg !22
  %454 = tail call float @llvm.amdgcn.exp2.f32(float %453), !dbg !22
  %455 = insertelement <4 x float> poison, float %454, i64 0, !dbg !22
  %456 = extractelement <32 x float> %449, i64 1, !dbg !22
  %457 = tail call float @llvm.amdgcn.exp2.f32(float %456), !dbg !22
  %458 = fadd float %454, %457, !dbg !23
  %459 = insertelement <4 x float> %455, float %457, i64 1, !dbg !22
  %460 = extractelement <32 x float> %449, i64 2, !dbg !22
  %461 = tail call float @llvm.amdgcn.exp2.f32(float %460), !dbg !22
  %462 = fadd float %461, %458, !dbg !23
  %463 = insertelement <4 x float> %459, float %461, i64 2, !dbg !22
  %464 = extractelement <32 x float> %449, i64 3, !dbg !22
  %465 = tail call float @llvm.amdgcn.exp2.f32(float %464), !dbg !22
  %466 = fadd float %465, %462, !dbg !23
  %467 = insertelement <4 x float> %463, float %465, i64 3, !dbg !22
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %467) #9, !dbg !24, !frisk.register_lifetimes !5
  %468 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !24
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %468) #9, !dbg !22, !frisk.register_lifetimes !5
  %469 = extractelement <32 x float> %449, i64 4, !dbg !22
  %470 = tail call float @llvm.amdgcn.exp2.f32(float %469), !dbg !22
  %471 = fadd float %470, %466, !dbg !23
  %472 = insertelement <4 x float> poison, float %470, i64 0, !dbg !22
  %473 = extractelement <32 x float> %449, i64 5, !dbg !22
  %474 = tail call float @llvm.amdgcn.exp2.f32(float %473), !dbg !22
  %475 = fadd float %474, %471, !dbg !23
  %476 = insertelement <4 x float> %472, float %474, i64 1, !dbg !22
  %477 = extractelement <32 x float> %449, i64 6, !dbg !22
  %478 = tail call float @llvm.amdgcn.exp2.f32(float %477), !dbg !22
  %479 = fadd float %478, %475, !dbg !23
  %480 = insertelement <4 x float> %476, float %478, i64 2, !dbg !22
  %481 = extractelement <32 x float> %449, i64 7, !dbg !22
  %482 = tail call float @llvm.amdgcn.exp2.f32(float %481), !dbg !22
  %483 = fadd float %482, %479, !dbg !23
  %484 = insertelement <4 x float> %480, float %482, i64 3, !dbg !22
  %math.result103 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %484) #9, !dbg !24, !frisk.register_lifetimes !5
  %485 = fptrunc <4 x float> %math.result103 to <4 x half>, !dbg !24
  %math.packed104 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %485) #9, !dbg !22, !frisk.register_lifetimes !5
  %486 = extractelement <32 x float> %449, i64 8, !dbg !22
  %487 = tail call float @llvm.amdgcn.exp2.f32(float %486), !dbg !22
  %488 = fadd float %487, %483, !dbg !23
  %489 = insertelement <4 x float> poison, float %487, i64 0, !dbg !22
  %490 = extractelement <32 x float> %449, i64 9, !dbg !22
  %491 = tail call float @llvm.amdgcn.exp2.f32(float %490), !dbg !22
  %492 = fadd float %491, %488, !dbg !23
  %493 = insertelement <4 x float> %489, float %491, i64 1, !dbg !22
  %494 = extractelement <32 x float> %449, i64 10, !dbg !22
  %495 = tail call float @llvm.amdgcn.exp2.f32(float %494), !dbg !22
  %496 = fadd float %495, %492, !dbg !23
  %497 = insertelement <4 x float> %493, float %495, i64 2, !dbg !22
  %498 = extractelement <32 x float> %449, i64 11, !dbg !22
  %499 = tail call float @llvm.amdgcn.exp2.f32(float %498), !dbg !22
  %500 = fadd float %499, %496, !dbg !23
  %501 = insertelement <4 x float> %497, float %499, i64 3, !dbg !22
  %math.result105 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %501) #9, !dbg !24, !frisk.register_lifetimes !5
  %502 = fptrunc <4 x float> %math.result105 to <4 x half>, !dbg !24
  %math.packed106 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %502) #9, !dbg !22, !frisk.register_lifetimes !5
  %503 = extractelement <32 x float> %449, i64 12, !dbg !22
  %504 = tail call float @llvm.amdgcn.exp2.f32(float %503), !dbg !22
  %505 = fadd float %504, %500, !dbg !23
  %506 = insertelement <4 x float> poison, float %504, i64 0, !dbg !22
  %507 = extractelement <32 x float> %449, i64 13, !dbg !22
  %508 = tail call float @llvm.amdgcn.exp2.f32(float %507), !dbg !22
  %509 = fadd float %508, %505, !dbg !23
  %510 = insertelement <4 x float> %506, float %508, i64 1, !dbg !22
  %511 = extractelement <32 x float> %449, i64 14, !dbg !22
  %512 = tail call float @llvm.amdgcn.exp2.f32(float %511), !dbg !22
  %513 = fadd float %512, %509, !dbg !23
  %514 = insertelement <4 x float> %510, float %512, i64 2, !dbg !22
  %515 = extractelement <32 x float> %449, i64 15, !dbg !22
  %516 = tail call float @llvm.amdgcn.exp2.f32(float %515), !dbg !22
  %517 = fadd float %516, %513, !dbg !23
  %518 = insertelement <4 x float> %514, float %516, i64 3, !dbg !22
  %math.result107 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %518) #9, !dbg !24, !frisk.register_lifetimes !5
  %519 = fptrunc <4 x float> %math.result107 to <4 x half>, !dbg !24
  %math.packed108 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %519) #9, !dbg !22, !frisk.register_lifetimes !5
  %520 = extractelement <32 x float> %449, i64 16, !dbg !22
  %521 = tail call float @llvm.amdgcn.exp2.f32(float %520), !dbg !22
  %522 = fadd float %521, %517, !dbg !23
  %523 = insertelement <4 x float> poison, float %521, i64 0, !dbg !22
  %524 = extractelement <32 x float> %449, i64 17, !dbg !22
  %525 = tail call float @llvm.amdgcn.exp2.f32(float %524), !dbg !22
  %526 = fadd float %525, %522, !dbg !23
  %527 = insertelement <4 x float> %523, float %525, i64 1, !dbg !22
  %528 = extractelement <32 x float> %449, i64 18, !dbg !22
  %529 = tail call float @llvm.amdgcn.exp2.f32(float %528), !dbg !22
  %530 = fadd float %529, %526, !dbg !23
  %531 = insertelement <4 x float> %527, float %529, i64 2, !dbg !22
  %532 = extractelement <32 x float> %449, i64 19, !dbg !22
  %533 = tail call float @llvm.amdgcn.exp2.f32(float %532), !dbg !22
  %534 = fadd float %533, %530, !dbg !23
  %535 = insertelement <4 x float> %531, float %533, i64 3, !dbg !22
  %math.result109 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %535) #9, !dbg !24, !frisk.register_lifetimes !5
  %536 = fptrunc <4 x float> %math.result109 to <4 x half>, !dbg !24
  %math.packed110 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %536) #9, !dbg !22, !frisk.register_lifetimes !5
  %537 = extractelement <32 x float> %449, i64 20, !dbg !22
  %538 = tail call float @llvm.amdgcn.exp2.f32(float %537), !dbg !22
  %539 = fadd float %538, %534, !dbg !23
  %540 = insertelement <4 x float> poison, float %538, i64 0, !dbg !22
  %541 = extractelement <32 x float> %449, i64 21, !dbg !22
  %542 = tail call float @llvm.amdgcn.exp2.f32(float %541), !dbg !22
  %543 = fadd float %542, %539, !dbg !23
  %544 = insertelement <4 x float> %540, float %542, i64 1, !dbg !22
  %545 = extractelement <32 x float> %449, i64 22, !dbg !22
  %546 = tail call float @llvm.amdgcn.exp2.f32(float %545), !dbg !22
  %547 = fadd float %546, %543, !dbg !23
  %548 = insertelement <4 x float> %544, float %546, i64 2, !dbg !22
  %549 = extractelement <32 x float> %449, i64 23, !dbg !22
  %550 = tail call float @llvm.amdgcn.exp2.f32(float %549), !dbg !22
  %551 = fadd float %550, %547, !dbg !23
  %552 = insertelement <4 x float> %548, float %550, i64 3, !dbg !22
  %math.result111 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %552) #9, !dbg !24, !frisk.register_lifetimes !5
  %553 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !24
  %math.packed112 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %553) #9, !dbg !22, !frisk.register_lifetimes !5
  %554 = extractelement <32 x float> %449, i64 24, !dbg !22
  %555 = tail call float @llvm.amdgcn.exp2.f32(float %554), !dbg !22
  %556 = fadd float %555, %551, !dbg !23
  %557 = insertelement <4 x float> poison, float %555, i64 0, !dbg !22
  %558 = extractelement <32 x float> %449, i64 25, !dbg !22
  %559 = tail call float @llvm.amdgcn.exp2.f32(float %558), !dbg !22
  %560 = fadd float %559, %556, !dbg !23
  %561 = insertelement <4 x float> %557, float %559, i64 1, !dbg !22
  %562 = extractelement <32 x float> %449, i64 26, !dbg !22
  %563 = tail call float @llvm.amdgcn.exp2.f32(float %562), !dbg !22
  %564 = fadd float %563, %560, !dbg !23
  %565 = insertelement <4 x float> %561, float %563, i64 2, !dbg !22
  %566 = extractelement <32 x float> %449, i64 27, !dbg !22
  %567 = tail call float @llvm.amdgcn.exp2.f32(float %566), !dbg !22
  %568 = fadd float %567, %564, !dbg !23
  %569 = insertelement <4 x float> %565, float %567, i64 3, !dbg !22
  %math.result113 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %569) #9, !dbg !24, !frisk.register_lifetimes !5
  %570 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !24
  %math.packed114 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %570) #9, !dbg !22, !frisk.register_lifetimes !5
  %571 = extractelement <32 x float> %449, i64 28, !dbg !22
  %572 = tail call float @llvm.amdgcn.exp2.f32(float %571), !dbg !22
  %573 = fadd float %572, %568, !dbg !23
  %574 = insertelement <4 x float> poison, float %572, i64 0, !dbg !22
  %575 = extractelement <32 x float> %449, i64 29, !dbg !22
  %576 = tail call float @llvm.amdgcn.exp2.f32(float %575), !dbg !22
  %577 = fadd float %576, %573, !dbg !23
  %578 = insertelement <4 x float> %574, float %576, i64 1, !dbg !22
  %579 = extractelement <32 x float> %449, i64 30, !dbg !22
  %580 = tail call float @llvm.amdgcn.exp2.f32(float %579), !dbg !22
  %581 = fadd float %580, %577, !dbg !23
  %582 = insertelement <4 x float> %578, float %580, i64 2, !dbg !22
  %583 = extractelement <32 x float> %449, i64 31, !dbg !22
  %584 = tail call float @llvm.amdgcn.exp2.f32(float %583), !dbg !22
  %585 = fadd float %584, %581, !dbg !23
  %586 = insertelement <4 x float> %582, float %584, i64 3, !dbg !22
  %math.result115 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %586) #9, !dbg !24, !frisk.register_lifetimes !5
  %587 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !24
  %math.packed116 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %587) #9, !dbg !22, !frisk.register_lifetimes !5
  %588 = extractelement <16 x float> %450, i64 0, !dbg !22
  %589 = tail call float @llvm.amdgcn.exp2.f32(float %588), !dbg !22
  %590 = insertelement <4 x float> poison, float %589, i64 0, !dbg !22
  %591 = extractelement <16 x float> %450, i64 1, !dbg !22
  %592 = tail call float @llvm.amdgcn.exp2.f32(float %591), !dbg !22
  %593 = fadd float %589, %592, !dbg !23
  %594 = insertelement <4 x float> %590, float %592, i64 1, !dbg !22
  %595 = extractelement <16 x float> %450, i64 2, !dbg !22
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %595), !dbg !22
  %597 = fadd float %596, %593, !dbg !23
  %598 = insertelement <4 x float> %594, float %596, i64 2, !dbg !22
  %599 = extractelement <16 x float> %450, i64 3, !dbg !22
  %600 = tail call float @llvm.amdgcn.exp2.f32(float %599), !dbg !22
  %601 = fadd float %600, %597, !dbg !23
  %602 = insertelement <4 x float> %598, float %600, i64 3, !dbg !22
  %math.result117 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %602) #9, !dbg !24, !frisk.register_lifetimes !5
  %603 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !24
  %math.packed118 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %603) #9, !dbg !22, !frisk.register_lifetimes !5
  %604 = extractelement <16 x float> %450, i64 4, !dbg !22
  %605 = tail call float @llvm.amdgcn.exp2.f32(float %604), !dbg !22
  %606 = fadd float %605, %601, !dbg !23
  %607 = insertelement <4 x float> poison, float %605, i64 0, !dbg !22
  %608 = extractelement <16 x float> %450, i64 5, !dbg !22
  %609 = tail call float @llvm.amdgcn.exp2.f32(float %608), !dbg !22
  %610 = fadd float %609, %606, !dbg !23
  %611 = insertelement <4 x float> %607, float %609, i64 1, !dbg !22
  %612 = extractelement <16 x float> %450, i64 6, !dbg !22
  %613 = tail call float @llvm.amdgcn.exp2.f32(float %612), !dbg !22
  %614 = fadd float %613, %610, !dbg !23
  %615 = insertelement <4 x float> %611, float %613, i64 2, !dbg !22
  %616 = extractelement <16 x float> %450, i64 7, !dbg !22
  %617 = tail call float @llvm.amdgcn.exp2.f32(float %616), !dbg !22
  %618 = fadd float %617, %614, !dbg !23
  %619 = insertelement <4 x float> %615, float %617, i64 3, !dbg !22
  %math.result119 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %619) #9, !dbg !24, !frisk.register_lifetimes !5
  %620 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !24
  %math.packed120 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %620) #9, !dbg !22, !frisk.register_lifetimes !5
  %621 = extractelement <16 x float> %450, i64 8, !dbg !22
  %622 = tail call float @llvm.amdgcn.exp2.f32(float %621), !dbg !22
  %623 = fadd float %622, %618, !dbg !23
  %624 = insertelement <4 x float> poison, float %622, i64 0, !dbg !22
  %625 = extractelement <16 x float> %450, i64 9, !dbg !22
  %626 = tail call float @llvm.amdgcn.exp2.f32(float %625), !dbg !22
  %627 = fadd float %626, %623, !dbg !23
  %628 = insertelement <4 x float> %624, float %626, i64 1, !dbg !22
  %629 = extractelement <16 x float> %450, i64 10, !dbg !22
  %630 = tail call float @llvm.amdgcn.exp2.f32(float %629), !dbg !22
  %631 = fadd float %630, %627, !dbg !23
  %632 = insertelement <4 x float> %628, float %630, i64 2, !dbg !22
  %633 = extractelement <16 x float> %450, i64 11, !dbg !22
  %634 = tail call float @llvm.amdgcn.exp2.f32(float %633), !dbg !22
  %635 = fadd float %634, %631, !dbg !23
  %636 = insertelement <4 x float> %632, float %634, i64 3, !dbg !22
  %math.result121 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %636) #9, !dbg !24, !frisk.register_lifetimes !5
  %637 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !24
  %math.packed122 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %637) #9, !dbg !22, !frisk.register_lifetimes !5
  %638 = extractelement <16 x float> %450, i64 12, !dbg !22
  %639 = tail call float @llvm.amdgcn.exp2.f32(float %638), !dbg !22
  %640 = fadd float %639, %635, !dbg !23
  %641 = insertelement <4 x float> poison, float %639, i64 0, !dbg !22
  %642 = extractelement <16 x float> %450, i64 13, !dbg !22
  %643 = tail call float @llvm.amdgcn.exp2.f32(float %642), !dbg !22
  %644 = fadd float %643, %640, !dbg !23
  %645 = insertelement <4 x float> %641, float %643, i64 1, !dbg !22
  %646 = extractelement <16 x float> %450, i64 14, !dbg !22
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %646), !dbg !22
  %648 = fadd float %647, %644, !dbg !23
  %649 = insertelement <4 x float> %645, float %647, i64 2, !dbg !22
  %650 = extractelement <16 x float> %450, i64 15, !dbg !22
  %651 = tail call float @llvm.amdgcn.exp2.f32(float %650), !dbg !22
  %652 = fadd float %651, %648, !dbg !23
  %653 = insertelement <4 x float> %649, float %651, i64 3, !dbg !22
  %math.result123 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %653) #9, !dbg !24, !frisk.register_lifetimes !5
  %654 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !24
  %math.packed124 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %654) #9, !dbg !22, !frisk.register_lifetimes !5
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %.sink115), !dbg !22
  %656 = fadd float %655, %652, !dbg !23
  %657 = insertelement <4 x float> poison, float %655, i64 0, !dbg !22
  %658 = tail call float @llvm.amdgcn.exp2.f32(float %.sink113), !dbg !22
  %659 = fadd float %658, %656, !dbg !23
  %660 = insertelement <4 x float> %657, float %658, i64 1, !dbg !22
  %661 = tail call float @llvm.amdgcn.exp2.f32(float %.sink112), !dbg !22
  %662 = fadd float %661, %659, !dbg !23
  %663 = insertelement <4 x float> %660, float %661, i64 2, !dbg !22
  %664 = tail call float @llvm.amdgcn.exp2.f32(float %.sink110), !dbg !22
  %665 = fadd float %664, %662, !dbg !23
  %666 = insertelement <4 x float> %663, float %664, i64 3, !dbg !22
  %math.result125 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %666) #9, !dbg !24, !frisk.register_lifetimes !5
  %667 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !24
  %math.packed126 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %667) #9, !dbg !22, !frisk.register_lifetimes !5
  %668 = tail call float @llvm.amdgcn.exp2.f32(float %.sink109), !dbg !22
  %669 = fadd float %668, %665, !dbg !23
  %670 = insertelement <4 x float> poison, float %668, i64 0, !dbg !22
  %671 = tail call float @llvm.amdgcn.exp2.f32(float %.sink107), !dbg !22
  %672 = fadd float %671, %669, !dbg !23
  %673 = insertelement <4 x float> %670, float %671, i64 1, !dbg !22
  %674 = tail call float @llvm.amdgcn.exp2.f32(float %.sink106), !dbg !22
  %675 = fadd float %674, %672, !dbg !23
  %676 = insertelement <4 x float> %673, float %674, i64 2, !dbg !22
  %677 = tail call float @llvm.amdgcn.exp2.f32(float %.sink104), !dbg !22
  %678 = fadd float %677, %675, !dbg !23
  %679 = insertelement <4 x float> %676, float %677, i64 3, !dbg !22
  %math.result127 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %679) #9, !dbg !24, !frisk.register_lifetimes !5
  %680 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !24
  %math.packed128 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %680) #9, !dbg !22, !frisk.register_lifetimes !5
  %681 = tail call float @llvm.amdgcn.exp2.f32(float %.sink103), !dbg !22
  %682 = fadd float %681, %678, !dbg !23
  %683 = insertelement <4 x float> poison, float %681, i64 0, !dbg !22
  %684 = tail call float @llvm.amdgcn.exp2.f32(float %.sink101), !dbg !22
  %685 = fadd float %684, %682, !dbg !23
  %686 = insertelement <4 x float> %683, float %684, i64 1, !dbg !22
  %687 = tail call float @llvm.amdgcn.exp2.f32(float %.sink100), !dbg !22
  %688 = fadd float %687, %685, !dbg !23
  %689 = insertelement <4 x float> %686, float %687, i64 2, !dbg !22
  %690 = tail call float @llvm.amdgcn.exp2.f32(float %.sink98), !dbg !22
  %691 = fadd float %690, %688, !dbg !23
  %692 = insertelement <4 x float> %689, float %690, i64 3, !dbg !22
  %math.result129 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %692) #9, !dbg !24, !frisk.register_lifetimes !5
  %693 = fptrunc <4 x float> %math.result129 to <4 x half>, !dbg !24
  %math.packed130 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %693) #9, !dbg !22, !frisk.register_lifetimes !5
  %694 = tail call float @llvm.amdgcn.exp2.f32(float %.sink97), !dbg !22
  %695 = fadd float %694, %691, !dbg !23
  %696 = insertelement <4 x float> poison, float %694, i64 0, !dbg !22
  %697 = tail call float @llvm.amdgcn.exp2.f32(float %.sink95), !dbg !22
  %698 = fadd float %697, %695, !dbg !23
  %699 = insertelement <4 x float> %696, float %697, i64 1, !dbg !22
  %700 = tail call float @llvm.amdgcn.exp2.f32(float %.sink94), !dbg !22
  %701 = fadd float %700, %698, !dbg !23
  %702 = insertelement <4 x float> %699, float %700, i64 2, !dbg !22
  %703 = tail call float @llvm.amdgcn.exp2.f32(float %452), !dbg !22
  %704 = fadd float %703, %701, !dbg !23
  %705 = insertelement <4 x float> %702, float %703, i64 3, !dbg !22
  %math.result131 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %705) #9, !dbg !24, !frisk.register_lifetimes !5
  %706 = fptrunc <4 x float> %math.result131 to <4 x half>, !dbg !24
  %math.packed132 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %706) #9, !dbg !23, !frisk.register_lifetimes !5
  %707 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !23
  %708 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %707), !dbg !23
  %709 = add nuw nsw i32 %708, 64, !dbg !23
  %710 = xor i32 %708, 16, !dbg !23
  %711 = and i32 %709, 192, !dbg !23
  %712 = icmp samesign ult i32 %710, %711, !dbg !23
  %713 = select i1 %712, i32 %710, i32 %708, !dbg !23
  %714 = shl nuw nsw i32 %713, 2, !dbg !23
  %715 = bitcast float %585 to i32, !dbg !23
  %716 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %714, i32 %715), !dbg !23
  %717 = bitcast i32 %716 to float, !dbg !23
  %718 = fadd float %585, %717, !dbg !23
  %719 = xor i32 %708, 32, !dbg !23
  %720 = icmp samesign ult i32 %719, %711, !dbg !23
  %721 = select i1 %720, i32 %719, i32 %708, !dbg !23
  %722 = shl nuw nsw i32 %721, 2, !dbg !23
  %723 = bitcast float %718 to i32, !dbg !23
  %724 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %722, i32 %723), !dbg !23
  %725 = bitcast i32 %724 to float, !dbg !23
  %726 = fadd float %718, %725, !dbg !23
  %727 = extractvalue [2 x <1 x float>] %49, 0, !dbg !23
  %728 = extractelement <1 x float> %727, i64 0, !dbg !23
  %729 = fadd float %728, %726, !dbg !23
  %730 = insertelement <1 x float> poison, float %729, i64 0, !dbg !12
  %731 = insertvalue [2 x <1 x float>] poison, <1 x float> %730, 0, !dbg !12
  %732 = bitcast float %704 to i32, !dbg !23
  %733 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %714, i32 %732), !dbg !23
  %734 = bitcast i32 %733 to float, !dbg !23
  %735 = fadd float %704, %734, !dbg !23
  %736 = bitcast float %735 to i32, !dbg !23
  %737 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %722, i32 %736), !dbg !23
  %738 = bitcast i32 %737 to float, !dbg !23
  %739 = fadd float %735, %738, !dbg !23
  %740 = extractvalue [2 x <1 x float>] %49, 1, !dbg !23
  %741 = extractelement <1 x float> %740, i64 0, !dbg !23
  %742 = fadd float %741, %739, !dbg !23
  %743 = insertelement <1 x float> poison, float %742, i64 0, !dbg !12
  %744 = insertvalue [2 x <1 x float>] %731, <1 x float> %743, 1, !dbg !12
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
  %745 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #12, !dbg !25
  %746 = bitcast <4 x i32> %745 to <8 x half>, !dbg !25
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
  %747 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #12, !dbg !25
  %748 = bitcast <4 x i32> %747 to <8 x half>, !dbg !25
  %address.local348 = getelementptr i8, ptr addrspace(3) %address.local343, i64 512, !dbg !25
  %749 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #12, !dbg !25
  %750 = bitcast <4 x i32> %749 to <8 x half>, !dbg !25
  %address.local352 = getelementptr i8, ptr addrspace(3) %address.local343, i64 768, !dbg !25
  %751 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #12, !dbg !25
  %752 = bitcast <4 x i32> %751 to <8 x half>, !dbg !25
  %address.local354 = getelementptr i8, ptr addrspace(3) %address.local331, i64 4096, !dbg !25
  %753 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #12, !dbg !25
  %754 = bitcast <4 x i32> %753 to <8 x half>, !dbg !25
  %address.local358 = getelementptr i8, ptr addrspace(3) %address.local343, i64 4352, !dbg !25
  %755 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #12, !dbg !25
  %756 = bitcast <4 x i32> %755 to <8 x half>, !dbg !25
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local343, i64 4608, !dbg !25
  %757 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #12, !dbg !25
  %758 = bitcast <4 x i32> %757 to <8 x half>, !dbg !25
  %address.local366 = getelementptr i8, ptr addrspace(3) %address.local343, i64 4864, !dbg !25
  %759 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #12, !dbg !25
  %760 = bitcast <4 x i32> %759 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.local367 = shl nuw nsw i64 %address.local332, 3, !dbg !9
  %address.local368 = and i64 %address.local367, 1536, !dbg !9
  %address.local369 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local368, !dbg !9
  %address.local370 = ptrtoint ptr addrspace(3) %address.local369 to i64, !dbg !9
  %address.local371 = trunc i64 %address.local370 to i32, !dbg !9
  %761 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !19
  %address.local372 = shl nuw nsw i64 %address.local332, 5, !dbg !19
  %address.local373 = and i64 %address.local372, 8064, !dbg !19
  %address.local375 = and i64 %address.local367, 24, !dbg !19
  %address.local376 = or disjoint i64 %address.local373, %address.local375, !dbg !19
  %address.local377.idx = shl nuw nsw i64 %address.local376, 1, !dbg !19
  %762 = or disjoint i64 %address.local377.idx, %215, !dbg !19
  %763 = trunc i64 %762 to i32, !dbg !19
  %dma.buffer.offset641 = or disjoint i32 %763, 64, !dbg !19
  %dma.buffer.lds643 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %761), !dbg !19
  %764 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds643, i32 %dma.buffer.offset641, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %address.local378 = or disjoint i64 %address.local332, 256, !dbg !9
  %address.local379 = shl nuw nsw i64 %address.local378, 3, !dbg !9
  %address.local380 = and i64 %address.local379, 3584, !dbg !9
  %address.local381 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local380, !dbg !9
  %address.local382 = ptrtoint ptr addrspace(3) %address.local381 to i64, !dbg !9
  %address.local383 = trunc i64 %address.local382 to i32, !dbg !9
  %765 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !19
  %address.local384 = shl nuw nsw i64 %address.local378, 5, !dbg !19
  %address.local385 = and i64 %address.local384, 16256, !dbg !19
  %address.local388 = or disjoint i64 %address.local385, %address.local375, !dbg !19
  %address.local389.idx = shl nuw nsw i64 %address.local388, 1, !dbg !19
  %766 = or disjoint i64 %215, %address.local389.idx, !dbg !19
  %767 = trunc i64 %766 to i32, !dbg !19
  %dma.buffer.offset638 = or disjoint i32 %767, 64, !dbg !19
  %dma.buffer.lds640 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %765), !dbg !19
  %768 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds640, i32 %dma.buffer.offset638, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %769 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %770 = extractvalue [2 x <8 x float>] %50, 0, !dbg !26
  %771 = shufflevector <8 x float> %770, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %772 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %769, <4 x float> %771) #9, !dbg !25
  %773 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %773, <4 x float> %772) #9, !dbg !25
  %775 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %776 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %775, <4 x float> %774) #9, !dbg !25
  %777 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %778 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %777, <4 x float> %776) #9, !dbg !25
  %779 = shufflevector <8 x half> %754, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %780 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %779, <4 x float> %778) #9, !dbg !25
  %781 = shufflevector <8 x half> %756, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %782 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %781, <4 x float> %780) #9, !dbg !25
  %783 = shufflevector <8 x half> %758, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %784 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %783, <4 x float> %782) #9, !dbg !25
  %785 = shufflevector <8 x half> %760, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %786 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %785, <4 x float> %784) #9, !dbg !25
  %787 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %788 = shufflevector <8 x float> %770, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %789 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %787, <4 x float> %788) #9, !dbg !25
  %790 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %791 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %790, <4 x float> %789) #9, !dbg !25
  %792 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %793 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %792, <4 x float> %791) #9, !dbg !25
  %794 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %795 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %794, <4 x float> %793) #9, !dbg !25
  %796 = shufflevector <8 x half> %754, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %797 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %796, <4 x float> %795) #9, !dbg !25
  %798 = shufflevector <8 x half> %756, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %799 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %798, <4 x float> %797) #9, !dbg !25
  %800 = shufflevector <8 x half> %758, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %801 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %800, <4 x float> %799) #9, !dbg !25
  %802 = shufflevector <8 x half> %760, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %803 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %802, <4 x float> %801) #9, !dbg !25
  %804 = extractvalue [2 x <8 x float>] %50, 1, !dbg !26
  %805 = shufflevector <8 x float> %804, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %806 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %769, <4 x float> %805) #9, !dbg !25
  %807 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %773, <4 x float> %806) #9, !dbg !25
  %808 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %775, <4 x float> %807) #9, !dbg !25
  %809 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %777, <4 x float> %808) #9, !dbg !25
  %810 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %779, <4 x float> %809) #9, !dbg !25
  %811 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %781, <4 x float> %810) #9, !dbg !25
  %812 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %783, <4 x float> %811) #9, !dbg !25
  %813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %785, <4 x float> %812) #9, !dbg !25
  %814 = shufflevector <8 x float> %804, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %815 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %787, <4 x float> %814) #9, !dbg !25
  %816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %790, <4 x float> %815) #9, !dbg !25
  %817 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %792, <4 x float> %816) #9, !dbg !25
  %818 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %794, <4 x float> %817) #9, !dbg !25
  %819 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %796, <4 x float> %818) #9, !dbg !25
  %820 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %798, <4 x float> %819) #9, !dbg !25
  %821 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %800, <4 x float> %820) #9, !dbg !25
  %822 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %802, <4 x float> %821) #9, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %823 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #12, !dbg !25
  %824 = bitcast <4 x i32> %823 to <8 x half>, !dbg !25
  %825 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #12, !dbg !25
  %826 = bitcast <4 x i32> %825 to <8 x half>, !dbg !25
  %827 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #12, !dbg !25
  %828 = bitcast <4 x i32> %827 to <8 x half>, !dbg !25
  %829 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #12, !dbg !25
  %830 = bitcast <4 x i32> %829 to <8 x half>, !dbg !25
  %831 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #12, !dbg !25
  %832 = bitcast <4 x i32> %831 to <8 x half>, !dbg !25
  %833 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #12, !dbg !25
  %834 = bitcast <4 x i32> %833 to <8 x half>, !dbg !25
  %835 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #12, !dbg !25
  %836 = bitcast <4 x i32> %835 to <8 x half>, !dbg !25
  %837 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #12, !dbg !25
  %838 = bitcast <4 x i32> %837 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %dma.buffer.offset635 = or disjoint i32 %763, 128, !dbg !19
  %839 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds643, i32 %dma.buffer.offset635, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %dma.buffer.offset632 = or disjoint i32 %767, 128, !dbg !19
  %840 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds640, i32 %dma.buffer.offset632, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %841 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %842 = extractvalue [2 x <8 x float>] %48, 0, !dbg !26
  %843 = shufflevector <8 x float> %842, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %841, <4 x float> %843) #9, !dbg !25
  %845 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %845, <4 x float> %844) #9, !dbg !25
  %847 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %847, <4 x float> %846) #9, !dbg !25
  %849 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %849, <4 x float> %848) #9, !dbg !25
  %851 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %851, <4 x float> %850) #9, !dbg !25
  %853 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %853, <4 x float> %852) #9, !dbg !25
  %855 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %855, <4 x float> %854) #9, !dbg !25
  %857 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %857, <4 x float> %856) #9, !dbg !25
  %859 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %860 = shufflevector <8 x float> %842, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %859, <4 x float> %860) #9, !dbg !25
  %862 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %862, <4 x float> %861) #9, !dbg !25
  %864 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %864, <4 x float> %863) #9, !dbg !25
  %866 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %866, <4 x float> %865) #9, !dbg !25
  %868 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %868, <4 x float> %867) #9, !dbg !25
  %870 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %870, <4 x float> %869) #9, !dbg !25
  %872 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %872, <4 x float> %871) #9, !dbg !25
  %874 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %874, <4 x float> %873) #9, !dbg !25
  %876 = extractvalue [2 x <8 x float>] %48, 1, !dbg !26
  %877 = shufflevector <8 x float> %876, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %878 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %841, <4 x float> %877) #9, !dbg !25
  %879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %845, <4 x float> %878) #9, !dbg !25
  %880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %847, <4 x float> %879) #9, !dbg !25
  %881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %849, <4 x float> %880) #9, !dbg !25
  %882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %851, <4 x float> %881) #9, !dbg !25
  %883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %853, <4 x float> %882) #9, !dbg !25
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %855, <4 x float> %883) #9, !dbg !25
  %885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %857, <4 x float> %884) #9, !dbg !25
  %886 = shufflevector <8 x float> %876, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %859, <4 x float> %886) #9, !dbg !25
  %888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %862, <4 x float> %887) #9, !dbg !25
  %889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %864, <4 x float> %888) #9, !dbg !25
  %890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %866, <4 x float> %889) #9, !dbg !25
  %891 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %868, <4 x float> %890) #9, !dbg !25
  %892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %870, <4 x float> %891) #9, !dbg !25
  %893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %872, <4 x float> %892) #9, !dbg !25
  %894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %874, <4 x float> %893) #9, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %895 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #12, !dbg !25
  %896 = bitcast <4 x i32> %895 to <8 x half>, !dbg !25
  %897 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #12, !dbg !25
  %898 = bitcast <4 x i32> %897 to <8 x half>, !dbg !25
  %899 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #12, !dbg !25
  %900 = bitcast <4 x i32> %899 to <8 x half>, !dbg !25
  %901 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #12, !dbg !25
  %902 = bitcast <4 x i32> %901 to <8 x half>, !dbg !25
  %903 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #12, !dbg !25
  %904 = bitcast <4 x i32> %903 to <8 x half>, !dbg !25
  %905 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #12, !dbg !25
  %906 = bitcast <4 x i32> %905 to <8 x half>, !dbg !25
  %907 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #12, !dbg !25
  %908 = bitcast <4 x i32> %907 to <8 x half>, !dbg !25
  %909 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #12, !dbg !25
  %910 = bitcast <4 x i32> %909 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %dma.buffer.offset629 = or disjoint i32 %763, 192, !dbg !19
  %911 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds643, i32 %dma.buffer.offset629, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %dma.buffer.offset626 = or disjoint i32 %767, 192, !dbg !19
  %912 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds640, i32 %dma.buffer.offset626, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !19
  %913 = shufflevector <8 x half> %896, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %914 = extractvalue [2 x <8 x float>] %47, 0, !dbg !26
  %915 = shufflevector <8 x float> %914, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %913, <4 x float> %915) #9, !dbg !25
  %917 = shufflevector <8 x half> %898, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %917, <4 x float> %916) #9, !dbg !25
  %919 = shufflevector <8 x half> %900, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %919, <4 x float> %918) #9, !dbg !25
  %921 = shufflevector <8 x half> %902, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %921, <4 x float> %920) #9, !dbg !25
  %923 = shufflevector <8 x half> %904, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %923, <4 x float> %922) #9, !dbg !25
  %925 = shufflevector <8 x half> %906, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %925, <4 x float> %924) #9, !dbg !25
  %927 = shufflevector <8 x half> %908, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %927, <4 x float> %926) #9, !dbg !25
  %929 = shufflevector <8 x half> %910, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %929, <4 x float> %928) #9, !dbg !25
  %931 = shufflevector <8 x half> %896, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %932 = shufflevector <8 x float> %914, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %931, <4 x float> %932) #9, !dbg !25
  %934 = shufflevector <8 x half> %898, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %935 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %934, <4 x float> %933) #9, !dbg !25
  %936 = shufflevector <8 x half> %900, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %936, <4 x float> %935) #9, !dbg !25
  %938 = shufflevector <8 x half> %902, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %938, <4 x float> %937) #9, !dbg !25
  %940 = shufflevector <8 x half> %904, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %941 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %940, <4 x float> %939) #9, !dbg !25
  %942 = shufflevector <8 x half> %906, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %943 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %942, <4 x float> %941) #9, !dbg !25
  %944 = shufflevector <8 x half> %908, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %945 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %944, <4 x float> %943) #9, !dbg !25
  %946 = shufflevector <8 x half> %910, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %947 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %946, <4 x float> %945) #9, !dbg !25
  %948 = extractvalue [2 x <8 x float>] %47, 1, !dbg !26
  %949 = shufflevector <8 x float> %948, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %950 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %913, <4 x float> %949) #9, !dbg !25
  %951 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %917, <4 x float> %950) #9, !dbg !25
  %952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %919, <4 x float> %951) #9, !dbg !25
  %953 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %921, <4 x float> %952) #9, !dbg !25
  %954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %923, <4 x float> %953) #9, !dbg !25
  %955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %925, <4 x float> %954) #9, !dbg !25
  %956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %927, <4 x float> %955) #9, !dbg !25
  %957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %929, <4 x float> %956) #9, !dbg !25
  %958 = shufflevector <8 x float> %948, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %931, <4 x float> %958) #9, !dbg !25
  %960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %934, <4 x float> %959) #9, !dbg !25
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %936, <4 x float> %960) #9, !dbg !25
  %962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %938, <4 x float> %961) #9, !dbg !25
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %940, <4 x float> %962) #9, !dbg !25
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %942, <4 x float> %963) #9, !dbg !25
  %965 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %944, <4 x float> %964) #9, !dbg !25
  %966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %946, <4 x float> %965) #9, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  tail call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %967 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #12, !dbg !25
  %968 = bitcast <4 x i32> %967 to <8 x half>, !dbg !25
  %969 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #12, !dbg !25
  %970 = bitcast <4 x i32> %969 to <8 x half>, !dbg !25
  %971 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #12, !dbg !25
  %972 = bitcast <4 x i32> %971 to <8 x half>, !dbg !25
  %973 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #12, !dbg !25
  %974 = bitcast <4 x i32> %973 to <8 x half>, !dbg !25
  %975 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #12, !dbg !25
  %976 = bitcast <4 x i32> %975 to <8 x half>, !dbg !25
  %977 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #12, !dbg !25
  %978 = bitcast <4 x i32> %977 to <8 x half>, !dbg !25
  %979 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #12, !dbg !25
  %980 = bitcast <4 x i32> %979 to <8 x half>, !dbg !25
  %981 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #12, !dbg !25
  %982 = bitcast <4 x i32> %981 to <8 x half>, !dbg !25
  %983 = shufflevector <8 x half> %968, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %984 = extractvalue [2 x <8 x float>] %46, 0, !dbg !26
  %985 = shufflevector <8 x float> %984, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %983, <4 x float> %985) #9, !dbg !25
  %987 = shufflevector <8 x half> %970, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %987, <4 x float> %986) #9, !dbg !25
  %989 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %989, <4 x float> %988) #9, !dbg !25
  %991 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %991, <4 x float> %990) #9, !dbg !25
  %993 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %993, <4 x float> %992) #9, !dbg !25
  %995 = shufflevector <8 x half> %978, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %995, <4 x float> %994) #9, !dbg !25
  %997 = shufflevector <8 x half> %980, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %997, <4 x float> %996) #9, !dbg !25
  %999 = shufflevector <8 x half> %982, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %999, <4 x float> %998) #9, !dbg !25
  %1001 = shufflevector <8 x half> %968, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1002 = shufflevector <8 x float> %984, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1003 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1001, <4 x float> %1002) #9, !dbg !25
  %1004 = shufflevector <8 x half> %970, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1005 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1004, <4 x float> %1003) #9, !dbg !25
  %1006 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1007 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1006, <4 x float> %1005) #9, !dbg !25
  %1008 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1009 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1008, <4 x float> %1007) #9, !dbg !25
  %1010 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1011 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1010, <4 x float> %1009) #9, !dbg !25
  %1012 = shufflevector <8 x half> %978, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1012, <4 x float> %1011) #9, !dbg !25
  %1014 = shufflevector <8 x half> %980, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1015 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1014, <4 x float> %1013) #9, !dbg !25
  %1016 = shufflevector <8 x half> %982, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1017 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1016, <4 x float> %1015) #9, !dbg !25
  %1018 = extractvalue [2 x <8 x float>] %46, 1, !dbg !26
  %1019 = shufflevector <8 x float> %1018, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %983, <4 x float> %1019) #9, !dbg !25
  %1021 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %987, <4 x float> %1020) #9, !dbg !25
  %1022 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %989, <4 x float> %1021) #9, !dbg !25
  %1023 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %991, <4 x float> %1022) #9, !dbg !25
  %1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %993, <4 x float> %1023) #9, !dbg !25
  %1025 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %995, <4 x float> %1024) #9, !dbg !25
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %997, <4 x float> %1025) #9, !dbg !25
  %1027 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %999, <4 x float> %1026) #9, !dbg !25
  %1028 = shufflevector <8 x float> %1018, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1029 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1001, <4 x float> %1028) #9, !dbg !25
  %1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1004, <4 x float> %1029) #9, !dbg !25
  %1031 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1006, <4 x float> %1030) #9, !dbg !25
  %1032 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1008, <4 x float> %1031) #9, !dbg !25
  %1033 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1010, <4 x float> %1032) #9, !dbg !25
  %1034 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1012, <4 x float> %1033) #9, !dbg !25
  %1035 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1014, <4 x float> %1034) #9, !dbg !25
  %1036 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1016, <4 x float> %1035) #9, !dbg !25
  %1037 = shufflevector <4 x float> %786, <4 x float> %803, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1038 = insertvalue [2 x <8 x float>] poison, <8 x float> %1037, 0, !dbg !12
  %1039 = shufflevector <4 x float> %813, <4 x float> %822, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1040 = insertvalue [2 x <8 x float>] %1038, <8 x float> %1039, 1, !dbg !12
  %1041 = shufflevector <4 x float> %858, <4 x float> %875, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1042 = insertvalue [2 x <8 x float>] poison, <8 x float> %1041, 0, !dbg !12
  %1043 = shufflevector <4 x float> %885, <4 x float> %894, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1044 = insertvalue [2 x <8 x float>] %1042, <8 x float> %1043, 1, !dbg !12
  %1045 = shufflevector <4 x float> %930, <4 x float> %947, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1046 = insertvalue [2 x <8 x float>] poison, <8 x float> %1045, 0, !dbg !12
  %1047 = shufflevector <4 x float> %957, <4 x float> %966, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1048 = insertvalue [2 x <8 x float>] %1046, <8 x float> %1047, 1, !dbg !12
  %1049 = shufflevector <4 x float> %1000, <4 x float> %1017, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1050 = insertvalue [2 x <8 x float>] poison, <8 x float> %1049, 0, !dbg !12
  %1051 = shufflevector <4 x float> %1027, <4 x float> %1036, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1052 = insertvalue [2 x <8 x float>] %1050, <8 x float> %1051, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %1053 = add nuw nsw i64 %51, 1, !dbg !12
  %.not.not = icmp slt i64 %51, %6, !dbg !12
  br i1 %.not.not, label %45, label %._crit_edge, !dbg !12

._crit_edge:                                      ; preds = %448, %.._crit_edge_crit_edge
  %.idx24.pre-phi = phi i64 [ %.pre67, %.._crit_edge_crit_edge ], [ %.idx, %448 ], !dbg !13
  %address.local396.pre-phi = phi i64 [ %.pre66, %.._crit_edge_crit_edge ], [ %19, %448 ], !dbg !13
  %.pre-phi65 = phi i32 [ %.pre64, %.._crit_edge_crit_edge ], [ %719, %448 ], !dbg !12
  %.pre-phi63 = phi i32 [ %.pre62, %.._crit_edge_crit_edge ], [ %711, %448 ], !dbg !12
  %.pre-phi61 = phi i32 [ %.pre60, %.._crit_edge_crit_edge ], [ %710, %448 ], !dbg !12
  %.pre-phi57 = phi i32 [ %.pre56, %.._crit_edge_crit_edge ], [ %708, %448 ], !dbg !12
  %.lcssa29 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1040, %448 ], !dbg !12
  %.lcssa28 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %744, %448 ], !dbg !12
  %.lcssa27 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1044, %448 ], !dbg !12
  %.lcssa26 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1048, %448 ], !dbg !12
  %.lcssa25 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1052, %448 ], !dbg !12
  %1054 = and i32 %17, 16, !dbg !12
  %.not2 = icmp eq i32 %1054, 0, !dbg !12
  %1055 = and i32 %17, 32, !dbg !12
  %.not3 = icmp eq i32 %1055, 0, !dbg !12
  %1056 = extractvalue [2 x <8 x float>] %.lcssa25, 0, !dbg !12
  %1057 = extractelement <8 x float> %1056, i64 0, !dbg !12
  %1058 = extractelement <8 x float> %1056, i64 1, !dbg !12
  %1059 = extractelement <8 x float> %1056, i64 2, !dbg !12
  %1060 = extractelement <8 x float> %1056, i64 3, !dbg !12
  %1061 = select i1 %.not2, float %1058, float %1057, !dbg !12
  %1062 = icmp samesign ult i32 %.pre-phi61, %.pre-phi63, !dbg !12
  %1063 = select i1 %1062, i32 %.pre-phi61, i32 %.pre-phi57, !dbg !12
  %1064 = shl nuw nsw i32 %1063, 2, !dbg !12
  %1065 = bitcast float %1061 to i32, !dbg !12
  %1066 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1065), !dbg !12
  %1067 = bitcast i32 %1066 to float, !dbg !12
  %1068 = select i1 %.not2, float %1057, float %1067, !dbg !12
  %1069 = select i1 %.not2, float %1067, float %1058, !dbg !12
  %1070 = select i1 %.not2, float %1060, float %1059, !dbg !12
  %1071 = bitcast float %1070 to i32, !dbg !12
  %1072 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1071), !dbg !12
  %1073 = bitcast i32 %1072 to float, !dbg !12
  %1074 = select i1 %.not2, float %1059, float %1073, !dbg !12
  %1075 = select i1 %.not2, float %1073, float %1060, !dbg !12
  %1076 = select i1 %.not3, float %1074, float %1068, !dbg !12
  %1077 = icmp samesign ult i32 %.pre-phi65, %.pre-phi63, !dbg !12
  %1078 = select i1 %1077, i32 %.pre-phi65, i32 %.pre-phi57, !dbg !12
  %1079 = shl nuw nsw i32 %1078, 2, !dbg !12
  %1080 = bitcast float %1076 to i32, !dbg !12
  %1081 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1080), !dbg !12
  %1082 = bitcast i32 %1081 to float, !dbg !12
  %1083 = select i1 %.not3, float %1075, float %1069, !dbg !12
  %1084 = bitcast float %1083 to i32, !dbg !12
  %1085 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1084), !dbg !12
  %1086 = bitcast i32 %1085 to float, !dbg !12
  %1087 = insertelement <4 x i1> poison, i1 %.not3, i64 0, !dbg !12
  %1088 = shufflevector <4 x i1> %1087, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1089 = insertelement <4 x float> poison, float %1068, i64 0, !dbg !12
  %1090 = insertelement <4 x float> %1089, float %1069, i64 1, !dbg !12
  %1091 = insertelement <4 x float> %1090, float %1082, i64 2, !dbg !12
  %1092 = insertelement <4 x float> %1091, float %1086, i64 3, !dbg !12
  %1093 = insertelement <4 x float> poison, float %1082, i64 0, !dbg !12
  %1094 = insertelement <4 x float> %1093, float %1086, i64 1, !dbg !12
  %1095 = insertelement <4 x float> %1094, float %1074, i64 2, !dbg !12
  %1096 = insertelement <4 x float> %1095, float %1075, i64 3, !dbg !12
  %1097 = select <4 x i1> %1088, <4 x float> %1092, <4 x float> %1096, !dbg !12
  %1098 = extractvalue [2 x <8 x float>] %.lcssa26, 0, !dbg !12
  %1099 = extractelement <8 x float> %1098, i64 0, !dbg !12
  %1100 = extractelement <8 x float> %1098, i64 1, !dbg !12
  %1101 = extractelement <8 x float> %1098, i64 2, !dbg !12
  %1102 = extractelement <8 x float> %1098, i64 3, !dbg !12
  %1103 = select i1 %.not2, float %1100, float %1099, !dbg !12
  %1104 = bitcast float %1103 to i32, !dbg !12
  %1105 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1104), !dbg !12
  %1106 = bitcast i32 %1105 to float, !dbg !12
  %1107 = select i1 %.not2, float %1099, float %1106, !dbg !12
  %1108 = select i1 %.not2, float %1106, float %1100, !dbg !12
  %1109 = select i1 %.not2, float %1102, float %1101, !dbg !12
  %1110 = bitcast float %1109 to i32, !dbg !12
  %1111 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1110), !dbg !12
  %1112 = bitcast i32 %1111 to float, !dbg !12
  %1113 = select i1 %.not2, float %1101, float %1112, !dbg !12
  %1114 = select i1 %.not2, float %1112, float %1102, !dbg !12
  %1115 = select i1 %.not3, float %1113, float %1107, !dbg !12
  %1116 = bitcast float %1115 to i32, !dbg !12
  %1117 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1116), !dbg !12
  %1118 = bitcast i32 %1117 to float, !dbg !12
  %1119 = select i1 %.not3, float %1114, float %1108, !dbg !12
  %1120 = bitcast float %1119 to i32, !dbg !12
  %1121 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1120), !dbg !12
  %1122 = bitcast i32 %1121 to float, !dbg !12
  %1123 = insertelement <4 x float> poison, float %1107, i64 0, !dbg !12
  %1124 = insertelement <4 x float> %1123, float %1108, i64 1, !dbg !12
  %1125 = insertelement <4 x float> %1124, float %1118, i64 2, !dbg !12
  %1126 = insertelement <4 x float> %1125, float %1122, i64 3, !dbg !12
  %1127 = insertelement <4 x float> poison, float %1118, i64 0, !dbg !12
  %1128 = insertelement <4 x float> %1127, float %1122, i64 1, !dbg !12
  %1129 = insertelement <4 x float> %1128, float %1113, i64 2, !dbg !12
  %1130 = insertelement <4 x float> %1129, float %1114, i64 3, !dbg !12
  %1131 = select <4 x i1> %1088, <4 x float> %1126, <4 x float> %1130, !dbg !12
  %1132 = extractvalue [2 x <8 x float>] %.lcssa27, 0, !dbg !12
  %1133 = extractelement <8 x float> %1132, i64 0, !dbg !12
  %1134 = extractelement <8 x float> %1132, i64 1, !dbg !12
  %1135 = extractelement <8 x float> %1132, i64 2, !dbg !12
  %1136 = extractelement <8 x float> %1132, i64 3, !dbg !12
  %1137 = select i1 %.not2, float %1134, float %1133, !dbg !12
  %1138 = bitcast float %1137 to i32, !dbg !12
  %1139 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1138), !dbg !12
  %1140 = bitcast i32 %1139 to float, !dbg !12
  %1141 = select i1 %.not2, float %1133, float %1140, !dbg !12
  %1142 = select i1 %.not2, float %1140, float %1134, !dbg !12
  %1143 = select i1 %.not2, float %1136, float %1135, !dbg !12
  %1144 = bitcast float %1143 to i32, !dbg !12
  %1145 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1144), !dbg !12
  %1146 = bitcast i32 %1145 to float, !dbg !12
  %1147 = select i1 %.not2, float %1135, float %1146, !dbg !12
  %1148 = select i1 %.not2, float %1146, float %1136, !dbg !12
  %1149 = select i1 %.not3, float %1147, float %1141, !dbg !12
  %1150 = bitcast float %1149 to i32, !dbg !12
  %1151 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1150), !dbg !12
  %1152 = bitcast i32 %1151 to float, !dbg !12
  %1153 = select i1 %.not3, float %1148, float %1142, !dbg !12
  %1154 = bitcast float %1153 to i32, !dbg !12
  %1155 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1154), !dbg !12
  %1156 = bitcast i32 %1155 to float, !dbg !12
  %1157 = insertelement <4 x float> poison, float %1141, i64 0, !dbg !12
  %1158 = insertelement <4 x float> %1157, float %1142, i64 1, !dbg !12
  %1159 = insertelement <4 x float> %1158, float %1152, i64 2, !dbg !12
  %1160 = insertelement <4 x float> %1159, float %1156, i64 3, !dbg !12
  %1161 = insertelement <4 x float> poison, float %1152, i64 0, !dbg !12
  %1162 = insertelement <4 x float> %1161, float %1156, i64 1, !dbg !12
  %1163 = insertelement <4 x float> %1162, float %1147, i64 2, !dbg !12
  %1164 = insertelement <4 x float> %1163, float %1148, i64 3, !dbg !12
  %1165 = select <4 x i1> %1088, <4 x float> %1160, <4 x float> %1164, !dbg !12
  %1166 = extractvalue [2 x <8 x float>] %.lcssa29, 0, !dbg !12
  %1167 = extractelement <8 x float> %1166, i64 0, !dbg !12
  %1168 = extractelement <8 x float> %1166, i64 1, !dbg !12
  %1169 = extractelement <8 x float> %1166, i64 2, !dbg !12
  %1170 = extractelement <8 x float> %1166, i64 3, !dbg !12
  %1171 = select i1 %.not2, float %1168, float %1167, !dbg !12
  %1172 = bitcast float %1171 to i32, !dbg !12
  %1173 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1172), !dbg !12
  %1174 = bitcast i32 %1173 to float, !dbg !12
  %1175 = select i1 %.not2, float %1167, float %1174, !dbg !12
  %1176 = select i1 %.not2, float %1174, float %1168, !dbg !12
  %1177 = select i1 %.not2, float %1170, float %1169, !dbg !12
  %1178 = bitcast float %1177 to i32, !dbg !12
  %1179 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1178), !dbg !12
  %1180 = bitcast i32 %1179 to float, !dbg !12
  %1181 = select i1 %.not2, float %1169, float %1180, !dbg !12
  %1182 = select i1 %.not2, float %1180, float %1170, !dbg !12
  %1183 = select i1 %.not3, float %1181, float %1175, !dbg !12
  %1184 = bitcast float %1183 to i32, !dbg !12
  %1185 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1184), !dbg !12
  %1186 = select i1 %.not3, float %1182, float %1176, !dbg !12
  %1187 = bitcast float %1186 to i32, !dbg !12
  %1188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1187), !dbg !12
  %1189 = extractvalue [2 x <1 x float>] %.lcssa28, 0, !dbg !27
  %1190 = shufflevector <1 x float> %1189, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %1191 = extractelement <8 x float> %1056, i64 4, !dbg !12
  %1192 = extractelement <8 x float> %1056, i64 5, !dbg !12
  %1193 = extractelement <8 x float> %1056, i64 6, !dbg !12
  %1194 = extractelement <8 x float> %1056, i64 7, !dbg !12
  %1195 = select i1 %.not2, float %1192, float %1191, !dbg !12
  %1196 = bitcast float %1195 to i32, !dbg !12
  %1197 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1196), !dbg !12
  %1198 = bitcast i32 %1197 to float, !dbg !12
  %1199 = select i1 %.not2, float %1191, float %1198, !dbg !12
  %1200 = select i1 %.not2, float %1198, float %1192, !dbg !12
  %1201 = select i1 %.not2, float %1194, float %1193, !dbg !12
  %1202 = bitcast float %1201 to i32, !dbg !12
  %1203 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1202), !dbg !12
  %1204 = bitcast i32 %1203 to float, !dbg !12
  %1205 = select i1 %.not2, float %1193, float %1204, !dbg !12
  %1206 = select i1 %.not2, float %1204, float %1194, !dbg !12
  %1207 = select i1 %.not3, float %1205, float %1199, !dbg !12
  %1208 = bitcast float %1207 to i32, !dbg !12
  %1209 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1208), !dbg !12
  %1210 = bitcast i32 %1209 to float, !dbg !12
  %1211 = select i1 %.not3, float %1206, float %1200, !dbg !12
  %1212 = bitcast float %1211 to i32, !dbg !12
  %1213 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1212), !dbg !12
  %1214 = bitcast i32 %1213 to float, !dbg !12
  %1215 = insertelement <4 x float> poison, float %1199, i64 0, !dbg !12
  %1216 = insertelement <4 x float> %1215, float %1200, i64 1, !dbg !12
  %1217 = insertelement <4 x float> %1216, float %1210, i64 2, !dbg !12
  %1218 = insertelement <4 x float> %1217, float %1214, i64 3, !dbg !12
  %1219 = insertelement <4 x float> poison, float %1210, i64 0, !dbg !12
  %1220 = insertelement <4 x float> %1219, float %1214, i64 1, !dbg !12
  %1221 = insertelement <4 x float> %1220, float %1205, i64 2, !dbg !12
  %1222 = insertelement <4 x float> %1221, float %1206, i64 3, !dbg !12
  %1223 = select <4 x i1> %1088, <4 x float> %1218, <4 x float> %1222, !dbg !12
  %1224 = extractelement <8 x float> %1098, i64 4, !dbg !12
  %1225 = extractelement <8 x float> %1098, i64 5, !dbg !12
  %1226 = extractelement <8 x float> %1098, i64 6, !dbg !12
  %1227 = extractelement <8 x float> %1098, i64 7, !dbg !12
  %1228 = select i1 %.not2, float %1225, float %1224, !dbg !12
  %1229 = bitcast float %1228 to i32, !dbg !12
  %1230 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1229), !dbg !12
  %1231 = bitcast i32 %1230 to float, !dbg !12
  %1232 = select i1 %.not2, float %1224, float %1231, !dbg !12
  %1233 = select i1 %.not2, float %1231, float %1225, !dbg !12
  %1234 = select i1 %.not2, float %1227, float %1226, !dbg !12
  %1235 = bitcast float %1234 to i32, !dbg !12
  %1236 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1235), !dbg !12
  %1237 = bitcast i32 %1236 to float, !dbg !12
  %1238 = select i1 %.not2, float %1226, float %1237, !dbg !12
  %1239 = select i1 %.not2, float %1237, float %1227, !dbg !12
  %1240 = select i1 %.not3, float %1238, float %1232, !dbg !12
  %1241 = bitcast float %1240 to i32, !dbg !12
  %1242 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1241), !dbg !12
  %1243 = bitcast i32 %1242 to float, !dbg !12
  %1244 = select i1 %.not3, float %1239, float %1233, !dbg !12
  %1245 = bitcast float %1244 to i32, !dbg !12
  %1246 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1245), !dbg !12
  %1247 = bitcast i32 %1246 to float, !dbg !12
  %1248 = insertelement <4 x float> poison, float %1232, i64 0, !dbg !12
  %1249 = insertelement <4 x float> %1248, float %1233, i64 1, !dbg !12
  %1250 = insertelement <4 x float> %1249, float %1243, i64 2, !dbg !12
  %1251 = insertelement <4 x float> %1250, float %1247, i64 3, !dbg !12
  %1252 = insertelement <4 x float> poison, float %1243, i64 0, !dbg !12
  %1253 = insertelement <4 x float> %1252, float %1247, i64 1, !dbg !12
  %1254 = insertelement <4 x float> %1253, float %1238, i64 2, !dbg !12
  %1255 = insertelement <4 x float> %1254, float %1239, i64 3, !dbg !12
  %1256 = select <4 x i1> %1088, <4 x float> %1251, <4 x float> %1255, !dbg !12
  %1257 = extractelement <8 x float> %1132, i64 4, !dbg !12
  %1258 = extractelement <8 x float> %1132, i64 5, !dbg !12
  %1259 = extractelement <8 x float> %1132, i64 6, !dbg !12
  %1260 = extractelement <8 x float> %1132, i64 7, !dbg !12
  %1261 = select i1 %.not2, float %1258, float %1257, !dbg !12
  %1262 = bitcast float %1261 to i32, !dbg !12
  %1263 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1262), !dbg !12
  %1264 = bitcast i32 %1263 to float, !dbg !12
  %1265 = select i1 %.not2, float %1257, float %1264, !dbg !12
  %1266 = select i1 %.not2, float %1264, float %1258, !dbg !12
  %1267 = select i1 %.not2, float %1260, float %1259, !dbg !12
  %1268 = bitcast float %1267 to i32, !dbg !12
  %1269 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1268), !dbg !12
  %1270 = bitcast i32 %1269 to float, !dbg !12
  %1271 = select i1 %.not2, float %1259, float %1270, !dbg !12
  %1272 = select i1 %.not2, float %1270, float %1260, !dbg !12
  %1273 = select i1 %.not3, float %1271, float %1265, !dbg !12
  %1274 = bitcast float %1273 to i32, !dbg !12
  %1275 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1274), !dbg !12
  %1276 = bitcast i32 %1275 to float, !dbg !12
  %1277 = select i1 %.not3, float %1272, float %1266, !dbg !12
  %1278 = bitcast float %1277 to i32, !dbg !12
  %1279 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1278), !dbg !12
  %1280 = bitcast i32 %1279 to float, !dbg !12
  %1281 = insertelement <4 x float> poison, float %1265, i64 0, !dbg !12
  %1282 = insertelement <4 x float> %1281, float %1266, i64 1, !dbg !12
  %1283 = insertelement <4 x float> %1282, float %1276, i64 2, !dbg !12
  %1284 = insertelement <4 x float> %1283, float %1280, i64 3, !dbg !12
  %1285 = insertelement <4 x float> poison, float %1276, i64 0, !dbg !12
  %1286 = insertelement <4 x float> %1285, float %1280, i64 1, !dbg !12
  %1287 = insertelement <4 x float> %1286, float %1271, i64 2, !dbg !12
  %1288 = insertelement <4 x float> %1287, float %1272, i64 3, !dbg !12
  %1289 = select <4 x i1> %1088, <4 x float> %1284, <4 x float> %1288, !dbg !12
  %1290 = extractelement <8 x float> %1166, i64 4, !dbg !12
  %1291 = extractelement <8 x float> %1166, i64 5, !dbg !12
  %1292 = extractelement <8 x float> %1166, i64 6, !dbg !12
  %1293 = extractelement <8 x float> %1166, i64 7, !dbg !12
  %1294 = select i1 %.not2, float %1291, float %1290, !dbg !12
  %1295 = bitcast float %1294 to i32, !dbg !12
  %1296 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1295), !dbg !12
  %1297 = bitcast i32 %1296 to float, !dbg !12
  %1298 = select i1 %.not2, float %1290, float %1297, !dbg !12
  %1299 = select i1 %.not2, float %1297, float %1291, !dbg !12
  %1300 = select i1 %.not2, float %1293, float %1292, !dbg !12
  %1301 = bitcast float %1300 to i32, !dbg !12
  %1302 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1301), !dbg !12
  %1303 = bitcast i32 %1302 to float, !dbg !12
  %1304 = select i1 %.not2, float %1292, float %1303, !dbg !12
  %1305 = select i1 %.not2, float %1303, float %1293, !dbg !12
  %1306 = select i1 %.not3, float %1304, float %1298, !dbg !12
  %1307 = bitcast float %1306 to i32, !dbg !12
  %1308 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1307), !dbg !12
  %1309 = select i1 %.not3, float %1305, float %1299, !dbg !12
  %1310 = bitcast float %1309 to i32, !dbg !12
  %1311 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1310), !dbg !12
  %1312 = extractvalue [2 x <8 x float>] %.lcssa25, 1, !dbg !12
  %1313 = extractelement <8 x float> %1312, i64 0, !dbg !12
  %1314 = extractelement <8 x float> %1312, i64 1, !dbg !12
  %1315 = extractelement <8 x float> %1312, i64 2, !dbg !12
  %1316 = extractelement <8 x float> %1312, i64 3, !dbg !12
  %1317 = select i1 %.not2, float %1314, float %1313, !dbg !12
  %1318 = bitcast float %1317 to i32, !dbg !12
  %1319 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1318), !dbg !12
  %1320 = bitcast i32 %1319 to float, !dbg !12
  %1321 = select i1 %.not2, float %1313, float %1320, !dbg !12
  %1322 = select i1 %.not2, float %1320, float %1314, !dbg !12
  %1323 = select i1 %.not2, float %1316, float %1315, !dbg !12
  %1324 = bitcast float %1323 to i32, !dbg !12
  %1325 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1324), !dbg !12
  %1326 = bitcast i32 %1325 to float, !dbg !12
  %1327 = select i1 %.not2, float %1315, float %1326, !dbg !12
  %1328 = select i1 %.not2, float %1326, float %1316, !dbg !12
  %1329 = select i1 %.not3, float %1327, float %1321, !dbg !12
  %1330 = bitcast float %1329 to i32, !dbg !12
  %1331 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1330), !dbg !12
  %1332 = bitcast i32 %1331 to float, !dbg !12
  %1333 = select i1 %.not3, float %1328, float %1322, !dbg !12
  %1334 = bitcast float %1333 to i32, !dbg !12
  %1335 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1334), !dbg !12
  %1336 = bitcast i32 %1335 to float, !dbg !12
  %1337 = insertelement <4 x float> poison, float %1321, i64 0, !dbg !12
  %1338 = insertelement <4 x float> %1337, float %1322, i64 1, !dbg !12
  %1339 = insertelement <4 x float> %1338, float %1332, i64 2, !dbg !12
  %1340 = insertelement <4 x float> %1339, float %1336, i64 3, !dbg !12
  %1341 = insertelement <4 x float> poison, float %1332, i64 0, !dbg !12
  %1342 = insertelement <4 x float> %1341, float %1336, i64 1, !dbg !12
  %1343 = insertelement <4 x float> %1342, float %1327, i64 2, !dbg !12
  %1344 = insertelement <4 x float> %1343, float %1328, i64 3, !dbg !12
  %1345 = select <4 x i1> %1088, <4 x float> %1340, <4 x float> %1344, !dbg !12
  %1346 = extractvalue [2 x <8 x float>] %.lcssa26, 1, !dbg !12
  %1347 = extractelement <8 x float> %1346, i64 0, !dbg !12
  %1348 = extractelement <8 x float> %1346, i64 1, !dbg !12
  %1349 = extractelement <8 x float> %1346, i64 2, !dbg !12
  %1350 = extractelement <8 x float> %1346, i64 3, !dbg !12
  %1351 = select i1 %.not2, float %1348, float %1347, !dbg !12
  %1352 = bitcast float %1351 to i32, !dbg !12
  %1353 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1352), !dbg !12
  %1354 = bitcast i32 %1353 to float, !dbg !12
  %1355 = select i1 %.not2, float %1347, float %1354, !dbg !12
  %1356 = select i1 %.not2, float %1354, float %1348, !dbg !12
  %1357 = select i1 %.not2, float %1350, float %1349, !dbg !12
  %1358 = bitcast float %1357 to i32, !dbg !12
  %1359 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1358), !dbg !12
  %1360 = bitcast i32 %1359 to float, !dbg !12
  %1361 = select i1 %.not2, float %1349, float %1360, !dbg !12
  %1362 = select i1 %.not2, float %1360, float %1350, !dbg !12
  %1363 = select i1 %.not3, float %1361, float %1355, !dbg !12
  %1364 = bitcast float %1363 to i32, !dbg !12
  %1365 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1364), !dbg !12
  %1366 = bitcast i32 %1365 to float, !dbg !12
  %1367 = select i1 %.not3, float %1362, float %1356, !dbg !12
  %1368 = bitcast float %1367 to i32, !dbg !12
  %1369 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1368), !dbg !12
  %1370 = bitcast i32 %1369 to float, !dbg !12
  %1371 = insertelement <4 x float> poison, float %1355, i64 0, !dbg !12
  %1372 = insertelement <4 x float> %1371, float %1356, i64 1, !dbg !12
  %1373 = insertelement <4 x float> %1372, float %1366, i64 2, !dbg !12
  %1374 = insertelement <4 x float> %1373, float %1370, i64 3, !dbg !12
  %1375 = insertelement <4 x float> poison, float %1366, i64 0, !dbg !12
  %1376 = insertelement <4 x float> %1375, float %1370, i64 1, !dbg !12
  %1377 = insertelement <4 x float> %1376, float %1361, i64 2, !dbg !12
  %1378 = insertelement <4 x float> %1377, float %1362, i64 3, !dbg !12
  %1379 = select <4 x i1> %1088, <4 x float> %1374, <4 x float> %1378, !dbg !12
  %1380 = extractvalue [2 x <8 x float>] %.lcssa27, 1, !dbg !12
  %1381 = extractelement <8 x float> %1380, i64 0, !dbg !12
  %1382 = extractelement <8 x float> %1380, i64 1, !dbg !12
  %1383 = extractelement <8 x float> %1380, i64 2, !dbg !12
  %1384 = extractelement <8 x float> %1380, i64 3, !dbg !12
  %1385 = select i1 %.not2, float %1382, float %1381, !dbg !12
  %1386 = bitcast float %1385 to i32, !dbg !12
  %1387 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1386), !dbg !12
  %1388 = bitcast i32 %1387 to float, !dbg !12
  %1389 = select i1 %.not2, float %1381, float %1388, !dbg !12
  %1390 = select i1 %.not2, float %1388, float %1382, !dbg !12
  %1391 = select i1 %.not2, float %1384, float %1383, !dbg !12
  %1392 = bitcast float %1391 to i32, !dbg !12
  %1393 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1392), !dbg !12
  %1394 = bitcast i32 %1393 to float, !dbg !12
  %1395 = select i1 %.not2, float %1383, float %1394, !dbg !12
  %1396 = select i1 %.not2, float %1394, float %1384, !dbg !12
  %1397 = select i1 %.not3, float %1395, float %1389, !dbg !12
  %1398 = bitcast float %1397 to i32, !dbg !12
  %1399 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1398), !dbg !12
  %1400 = bitcast i32 %1399 to float, !dbg !12
  %1401 = select i1 %.not3, float %1396, float %1390, !dbg !12
  %1402 = bitcast float %1401 to i32, !dbg !12
  %1403 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1402), !dbg !12
  %1404 = bitcast i32 %1403 to float, !dbg !12
  %1405 = insertelement <4 x float> poison, float %1389, i64 0, !dbg !12
  %1406 = insertelement <4 x float> %1405, float %1390, i64 1, !dbg !12
  %1407 = insertelement <4 x float> %1406, float %1400, i64 2, !dbg !12
  %1408 = insertelement <4 x float> %1407, float %1404, i64 3, !dbg !12
  %1409 = insertelement <4 x float> poison, float %1400, i64 0, !dbg !12
  %1410 = insertelement <4 x float> %1409, float %1404, i64 1, !dbg !12
  %1411 = insertelement <4 x float> %1410, float %1395, i64 2, !dbg !12
  %1412 = insertelement <4 x float> %1411, float %1396, i64 3, !dbg !12
  %1413 = select <4 x i1> %1088, <4 x float> %1408, <4 x float> %1412, !dbg !12
  %1414 = extractvalue [2 x <8 x float>] %.lcssa29, 1, !dbg !12
  %1415 = extractelement <8 x float> %1414, i64 0, !dbg !12
  %1416 = extractelement <8 x float> %1414, i64 1, !dbg !12
  %1417 = extractelement <8 x float> %1414, i64 2, !dbg !12
  %1418 = extractelement <8 x float> %1414, i64 3, !dbg !12
  %1419 = select i1 %.not2, float %1416, float %1415, !dbg !12
  %1420 = bitcast float %1419 to i32, !dbg !12
  %1421 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1420), !dbg !12
  %1422 = bitcast i32 %1421 to float, !dbg !12
  %1423 = select i1 %.not2, float %1415, float %1422, !dbg !12
  %1424 = select i1 %.not2, float %1422, float %1416, !dbg !12
  %1425 = select i1 %.not2, float %1418, float %1417, !dbg !12
  %1426 = bitcast float %1425 to i32, !dbg !12
  %1427 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1426), !dbg !12
  %1428 = bitcast i32 %1427 to float, !dbg !12
  %1429 = select i1 %.not2, float %1417, float %1428, !dbg !12
  %1430 = select i1 %.not2, float %1428, float %1418, !dbg !12
  %1431 = select i1 %.not3, float %1429, float %1423, !dbg !12
  %1432 = bitcast float %1431 to i32, !dbg !12
  %1433 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1432), !dbg !12
  %1434 = select i1 %.not3, float %1430, float %1424, !dbg !12
  %1435 = bitcast float %1434 to i32, !dbg !12
  %1436 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1435), !dbg !12
  %1437 = extractvalue [2 x <1 x float>] %.lcssa28, 1, !dbg !27
  %1438 = shufflevector <1 x float> %1437, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %1439 = extractelement <8 x float> %1312, i64 4, !dbg !12
  %1440 = extractelement <8 x float> %1312, i64 5, !dbg !12
  %1441 = extractelement <8 x float> %1312, i64 6, !dbg !12
  %1442 = extractelement <8 x float> %1312, i64 7, !dbg !12
  %1443 = select i1 %.not2, float %1440, float %1439, !dbg !12
  %1444 = bitcast float %1443 to i32, !dbg !12
  %1445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1444), !dbg !12
  %1446 = bitcast i32 %1445 to float, !dbg !12
  %1447 = select i1 %.not2, float %1439, float %1446, !dbg !12
  %1448 = select i1 %.not2, float %1446, float %1440, !dbg !12
  %1449 = select i1 %.not2, float %1442, float %1441, !dbg !12
  %1450 = bitcast float %1449 to i32, !dbg !12
  %1451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1450), !dbg !12
  %1452 = bitcast i32 %1451 to float, !dbg !12
  %1453 = select i1 %.not2, float %1441, float %1452, !dbg !12
  %1454 = select i1 %.not2, float %1452, float %1442, !dbg !12
  %1455 = select i1 %.not3, float %1453, float %1447, !dbg !12
  %1456 = bitcast float %1455 to i32, !dbg !12
  %1457 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1456), !dbg !12
  %1458 = bitcast i32 %1457 to float, !dbg !12
  %1459 = select i1 %.not3, float %1454, float %1448, !dbg !12
  %1460 = bitcast float %1459 to i32, !dbg !12
  %1461 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1460), !dbg !12
  %1462 = bitcast i32 %1461 to float, !dbg !12
  %1463 = insertelement <4 x float> poison, float %1447, i64 0, !dbg !12
  %1464 = insertelement <4 x float> %1463, float %1448, i64 1, !dbg !12
  %1465 = insertelement <4 x float> %1464, float %1458, i64 2, !dbg !12
  %1466 = insertelement <4 x float> %1465, float %1462, i64 3, !dbg !12
  %1467 = insertelement <4 x float> poison, float %1458, i64 0, !dbg !12
  %1468 = insertelement <4 x float> %1467, float %1462, i64 1, !dbg !12
  %1469 = insertelement <4 x float> %1468, float %1453, i64 2, !dbg !12
  %1470 = insertelement <4 x float> %1469, float %1454, i64 3, !dbg !12
  %1471 = select <4 x i1> %1088, <4 x float> %1466, <4 x float> %1470, !dbg !12
  %1472 = extractelement <8 x float> %1346, i64 4, !dbg !12
  %1473 = extractelement <8 x float> %1346, i64 5, !dbg !12
  %1474 = extractelement <8 x float> %1346, i64 6, !dbg !12
  %1475 = extractelement <8 x float> %1346, i64 7, !dbg !12
  %1476 = select i1 %.not2, float %1473, float %1472, !dbg !12
  %1477 = bitcast float %1476 to i32, !dbg !12
  %1478 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1477), !dbg !12
  %1479 = bitcast i32 %1478 to float, !dbg !12
  %1480 = select i1 %.not2, float %1472, float %1479, !dbg !12
  %1481 = select i1 %.not2, float %1479, float %1473, !dbg !12
  %1482 = select i1 %.not2, float %1475, float %1474, !dbg !12
  %1483 = bitcast float %1482 to i32, !dbg !12
  %1484 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1483), !dbg !12
  %1485 = bitcast i32 %1484 to float, !dbg !12
  %1486 = select i1 %.not2, float %1474, float %1485, !dbg !12
  %1487 = select i1 %.not2, float %1485, float %1475, !dbg !12
  %1488 = select i1 %.not3, float %1486, float %1480, !dbg !12
  %1489 = bitcast float %1488 to i32, !dbg !12
  %1490 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1489), !dbg !12
  %1491 = bitcast i32 %1490 to float, !dbg !12
  %1492 = select i1 %.not3, float %1487, float %1481, !dbg !12
  %1493 = bitcast float %1492 to i32, !dbg !12
  %1494 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1493), !dbg !12
  %1495 = bitcast i32 %1494 to float, !dbg !12
  %1496 = insertelement <4 x float> poison, float %1480, i64 0, !dbg !12
  %1497 = insertelement <4 x float> %1496, float %1481, i64 1, !dbg !12
  %1498 = insertelement <4 x float> %1497, float %1491, i64 2, !dbg !12
  %1499 = insertelement <4 x float> %1498, float %1495, i64 3, !dbg !12
  %1500 = insertelement <4 x float> poison, float %1491, i64 0, !dbg !12
  %1501 = insertelement <4 x float> %1500, float %1495, i64 1, !dbg !12
  %1502 = insertelement <4 x float> %1501, float %1486, i64 2, !dbg !12
  %1503 = insertelement <4 x float> %1502, float %1487, i64 3, !dbg !12
  %1504 = select <4 x i1> %1088, <4 x float> %1499, <4 x float> %1503, !dbg !12
  %1505 = extractelement <8 x float> %1380, i64 4, !dbg !12
  %1506 = extractelement <8 x float> %1380, i64 5, !dbg !12
  %1507 = extractelement <8 x float> %1380, i64 6, !dbg !12
  %1508 = extractelement <8 x float> %1380, i64 7, !dbg !12
  %1509 = select i1 %.not2, float %1506, float %1505, !dbg !12
  %1510 = bitcast float %1509 to i32, !dbg !12
  %1511 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1510), !dbg !12
  %1512 = bitcast i32 %1511 to float, !dbg !12
  %1513 = select i1 %.not2, float %1505, float %1512, !dbg !12
  %1514 = select i1 %.not2, float %1512, float %1506, !dbg !12
  %1515 = select i1 %.not2, float %1508, float %1507, !dbg !12
  %1516 = bitcast float %1515 to i32, !dbg !12
  %1517 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1516), !dbg !12
  %1518 = bitcast i32 %1517 to float, !dbg !12
  %1519 = select i1 %.not2, float %1507, float %1518, !dbg !12
  %1520 = select i1 %.not2, float %1518, float %1508, !dbg !12
  %1521 = select i1 %.not3, float %1519, float %1513, !dbg !12
  %1522 = bitcast float %1521 to i32, !dbg !12
  %1523 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1522), !dbg !12
  %1524 = bitcast i32 %1523 to float, !dbg !12
  %1525 = select i1 %.not3, float %1520, float %1514, !dbg !12
  %1526 = bitcast float %1525 to i32, !dbg !12
  %1527 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1526), !dbg !12
  %1528 = bitcast i32 %1527 to float, !dbg !12
  %1529 = insertelement <4 x float> poison, float %1513, i64 0, !dbg !12
  %1530 = insertelement <4 x float> %1529, float %1514, i64 1, !dbg !12
  %1531 = insertelement <4 x float> %1530, float %1524, i64 2, !dbg !12
  %1532 = insertelement <4 x float> %1531, float %1528, i64 3, !dbg !12
  %1533 = insertelement <4 x float> poison, float %1524, i64 0, !dbg !12
  %1534 = insertelement <4 x float> %1533, float %1528, i64 1, !dbg !12
  %1535 = insertelement <4 x float> %1534, float %1519, i64 2, !dbg !12
  %1536 = insertelement <4 x float> %1535, float %1520, i64 3, !dbg !12
  %1537 = select <4 x i1> %1088, <4 x float> %1532, <4 x float> %1536, !dbg !12
  %1538 = extractelement <8 x float> %1414, i64 4, !dbg !12
  %1539 = extractelement <8 x float> %1414, i64 5, !dbg !12
  %1540 = extractelement <8 x float> %1414, i64 6, !dbg !12
  %1541 = extractelement <8 x float> %1414, i64 7, !dbg !12
  %1542 = select i1 %.not2, float %1539, float %1538, !dbg !12
  %1543 = bitcast float %1542 to i32, !dbg !12
  %1544 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1543), !dbg !12
  %1545 = bitcast i32 %1544 to float, !dbg !12
  %1546 = select i1 %.not2, float %1538, float %1545, !dbg !12
  %1547 = select i1 %.not2, float %1545, float %1539, !dbg !12
  %1548 = select i1 %.not2, float %1541, float %1540, !dbg !12
  %1549 = bitcast float %1548 to i32, !dbg !12
  %1550 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1064, i32 %1549), !dbg !12
  %1551 = bitcast i32 %1550 to float, !dbg !12
  %1552 = select i1 %.not2, float %1540, float %1551, !dbg !12
  %1553 = select i1 %.not2, float %1551, float %1541, !dbg !12
  %1554 = select i1 %.not3, float %1552, float %1546, !dbg !12
  %1555 = bitcast float %1554 to i32, !dbg !12
  %1556 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1555), !dbg !12
  %1557 = select i1 %.not3, float %1553, float %1547, !dbg !12
  %1558 = bitcast float %1557 to i32, !dbg !12
  %1559 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1079, i32 %1558), !dbg !12
  %1560 = bitcast i32 %1556 to float, !dbg !12
  %1561 = bitcast i32 %1559 to float, !dbg !12
  %1562 = insertelement <4 x float> poison, float %1546, i64 0, !dbg !12
  %1563 = insertelement <4 x float> %1562, float %1547, i64 1, !dbg !12
  %1564 = insertelement <4 x float> %1563, float %1560, i64 2, !dbg !12
  %1565 = insertelement <4 x float> %1564, float %1561, i64 3, !dbg !12
  %1566 = insertelement <4 x float> poison, float %1560, i64 0, !dbg !12
  %1567 = insertelement <4 x float> %1566, float %1561, i64 1, !dbg !12
  %1568 = insertelement <4 x float> %1567, float %1552, i64 2, !dbg !12
  %1569 = insertelement <4 x float> %1568, float %1553, i64 3, !dbg !12
  %1570 = select <4 x i1> %1088, <4 x float> %1565, <4 x float> %1569, !dbg !12
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1438, !dbg !27
  %normalized = fmul <4 x float> %row.reciprocal, %1570, !dbg !27
  %1571 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !28
  %1572 = extractelement <4 x half> %1571, i64 3, !dbg !28
  %1573 = extractelement <4 x half> %1571, i64 2, !dbg !28
  %1574 = extractelement <4 x half> %1571, i64 1, !dbg !28
  %1575 = extractelement <4 x half> %1571, i64 0, !dbg !28
  %1576 = bitcast i32 %1433 to float, !dbg !12
  %1577 = bitcast i32 %1436 to float, !dbg !12
  %1578 = insertelement <4 x float> poison, float %1423, i64 0, !dbg !12
  %1579 = insertelement <4 x float> %1578, float %1424, i64 1, !dbg !12
  %1580 = insertelement <4 x float> %1579, float %1576, i64 2, !dbg !12
  %1581 = insertelement <4 x float> %1580, float %1577, i64 3, !dbg !12
  %1582 = insertelement <4 x float> poison, float %1576, i64 0, !dbg !12
  %1583 = insertelement <4 x float> %1582, float %1577, i64 1, !dbg !12
  %1584 = insertelement <4 x float> %1583, float %1429, i64 2, !dbg !12
  %1585 = insertelement <4 x float> %1584, float %1430, i64 3, !dbg !12
  %1586 = select <4 x i1> %1088, <4 x float> %1581, <4 x float> %1585, !dbg !12
  %normalized75 = fmul <4 x float> %row.reciprocal, %1586, !dbg !27
  %1587 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !28
  %1588 = extractelement <4 x half> %1587, i64 3, !dbg !28
  %1589 = extractelement <4 x half> %1587, i64 2, !dbg !28
  %1590 = extractelement <4 x half> %1587, i64 1, !dbg !28
  %1591 = extractelement <4 x half> %1587, i64 0, !dbg !28
  %1592 = bitcast i32 %1308 to float, !dbg !12
  %1593 = bitcast i32 %1311 to float, !dbg !12
  %1594 = insertelement <4 x float> poison, float %1298, i64 0, !dbg !12
  %1595 = insertelement <4 x float> %1594, float %1299, i64 1, !dbg !12
  %1596 = insertelement <4 x float> %1595, float %1592, i64 2, !dbg !12
  %1597 = insertelement <4 x float> %1596, float %1593, i64 3, !dbg !12
  %1598 = insertelement <4 x float> poison, float %1592, i64 0, !dbg !12
  %1599 = insertelement <4 x float> %1598, float %1593, i64 1, !dbg !12
  %1600 = insertelement <4 x float> %1599, float %1304, i64 2, !dbg !12
  %1601 = insertelement <4 x float> %1600, float %1305, i64 3, !dbg !12
  %1602 = select <4 x i1> %1088, <4 x float> %1597, <4 x float> %1601, !dbg !12
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1190, !dbg !27
  %normalized77 = fmul <4 x float> %row.reciprocal76, %1602, !dbg !27
  %1603 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !28
  %1604 = extractelement <4 x half> %1603, i64 3, !dbg !28
  %1605 = extractelement <4 x half> %1603, i64 2, !dbg !28
  %1606 = extractelement <4 x half> %1603, i64 1, !dbg !28
  %1607 = extractelement <4 x half> %1603, i64 0, !dbg !28
  %1608 = bitcast i32 %1185 to float, !dbg !12
  %1609 = bitcast i32 %1188 to float, !dbg !12
  %1610 = insertelement <4 x float> poison, float %1175, i64 0, !dbg !12
  %1611 = insertelement <4 x float> %1610, float %1176, i64 1, !dbg !12
  %1612 = insertelement <4 x float> %1611, float %1608, i64 2, !dbg !12
  %1613 = insertelement <4 x float> %1612, float %1609, i64 3, !dbg !12
  %1614 = insertelement <4 x float> poison, float %1608, i64 0, !dbg !12
  %1615 = insertelement <4 x float> %1614, float %1609, i64 1, !dbg !12
  %1616 = insertelement <4 x float> %1615, float %1181, i64 2, !dbg !12
  %1617 = insertelement <4 x float> %1616, float %1182, i64 3, !dbg !12
  %1618 = select <4 x i1> %1088, <4 x float> %1613, <4 x float> %1617, !dbg !12
  %normalized78 = fmul <4 x float> %row.reciprocal76, %1618, !dbg !27
  %1619 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !28
  %1620 = extractelement <4 x half> %1619, i64 3, !dbg !28
  %1621 = extractelement <4 x half> %1619, i64 2, !dbg !28
  %1622 = extractelement <4 x half> %1619, i64 1, !dbg !28
  %1623 = extractelement <4 x half> %1619, i64 0, !dbg !28
  %1624 = shl nsw i64 %14, 19, !dbg !13
  %1625 = and i64 %1624, 16252928, !dbg !13
  %address.thread394 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #9, !dbg !13, !range !11, !frisk.register_lifetimes !5
  %address.local395 = zext nneg i32 %address.thread394 to i64, !dbg !10
  %1626 = lshr i64 %address.local395, 2, !dbg !13
  %address.local398 = and i64 %1626, 48, !dbg !13
  %address.local400 = and i64 %address.local395, 240, !dbg !13
  %address.local399 = add nsw i64 %address.local396.pre-phi, %address.local395, !dbg !13
  %address.local401 = sub nsw i64 %address.local399, %address.local400, !dbg !13
  %address.local402 = add nsw i64 %address.local401, %address.local398, !dbg !13
  %address.local403 = shl nuw nsw i64 %address.local402, 7, !dbg !13
  %1627 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %1625, !dbg !13
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1627, i64 %address.local403, !dbg !13
  %address.local407 = and i64 %1626, 12, !dbg !10
  %address.local408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local407, !dbg !13
  store half %1623, ptr addrspace(1) %address.local408, align 2, !dbg !13
  %address.local410 = or disjoint i64 %address.local407, 1, !dbg !13
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local410, !dbg !13
  store half %1622, ptr addrspace(1) %address.local411, align 2, !dbg !13
  %address.local413 = or disjoint i64 %address.local407, 2, !dbg !13
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local413, !dbg !13
  store half %1621, ptr addrspace(1) %address.local414, align 2, !dbg !13
  %address.local416 = or disjoint i64 %address.local407, 3, !dbg !13
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local416, !dbg !13
  store half %1620, ptr addrspace(1) %address.local417, align 2, !dbg !13
  %address.local419 = or disjoint i64 %address.local407, 16, !dbg !13
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local419, !dbg !13
  store half %1607, ptr addrspace(1) %address.local420, align 2, !dbg !13
  %address.local422 = or disjoint i64 %address.local407, 17, !dbg !13
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local422, !dbg !13
  store half %1606, ptr addrspace(1) %address.local423, align 2, !dbg !13
  %address.local425 = or disjoint i64 %address.local407, 18, !dbg !13
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local425, !dbg !13
  store half %1605, ptr addrspace(1) %address.local426, align 2, !dbg !13
  %address.local428 = or disjoint i64 %address.local407, 19, !dbg !13
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local428, !dbg !13
  store half %1604, ptr addrspace(1) %address.local429, align 2, !dbg !13
  %address.local431 = add nuw nsw i64 %address.local403, 8192, !dbg !13
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1627, i64 %address.local431, !dbg !13
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local407, !dbg !13
  store half %1591, ptr addrspace(1) %address.local434, align 2, !dbg !13
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local410, !dbg !13
  store half %1590, ptr addrspace(1) %address.local436, align 2, !dbg !13
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local413, !dbg !13
  store half %1589, ptr addrspace(1) %address.local438, align 2, !dbg !13
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local416, !dbg !13
  store half %1588, ptr addrspace(1) %address.local440, align 2, !dbg !13
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local419, !dbg !13
  store half %1575, ptr addrspace(1) %address.local442, align 2, !dbg !13
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local422, !dbg !13
  store half %1574, ptr addrspace(1) %address.local444, align 2, !dbg !13
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local425, !dbg !13
  store half %1573, ptr addrspace(1) %address.local446, align 2, !dbg !13
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local428, !dbg !13
  store half %1572, ptr addrspace(1) %address.local448, align 2, !dbg !13
  %normalized80 = fmul <4 x float> %row.reciprocal, %1537, !dbg !27
  %1628 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !28
  %1629 = extractelement <4 x half> %1628, i64 3, !dbg !28
  %1630 = extractelement <4 x half> %1628, i64 2, !dbg !28
  %1631 = extractelement <4 x half> %1628, i64 1, !dbg !28
  %1632 = extractelement <4 x half> %1628, i64 0, !dbg !28
  %normalized81 = fmul <4 x float> %row.reciprocal, %1413, !dbg !27
  %1633 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !28
  %1634 = extractelement <4 x half> %1633, i64 3, !dbg !28
  %1635 = extractelement <4 x half> %1633, i64 2, !dbg !28
  %1636 = extractelement <4 x half> %1633, i64 1, !dbg !28
  %1637 = extractelement <4 x half> %1633, i64 0, !dbg !28
  %normalized83 = fmul <4 x float> %row.reciprocal76, %1289, !dbg !27
  %1638 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !28
  %1639 = extractelement <4 x half> %1638, i64 3, !dbg !28
  %1640 = extractelement <4 x half> %1638, i64 2, !dbg !28
  %1641 = extractelement <4 x half> %1638, i64 1, !dbg !28
  %1642 = extractelement <4 x half> %1638, i64 0, !dbg !28
  %normalized84 = fmul <4 x float> %row.reciprocal76, %1165, !dbg !27
  %1643 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !28
  %1644 = extractelement <4 x half> %1643, i64 3, !dbg !28
  %1645 = extractelement <4 x half> %1643, i64 2, !dbg !28
  %1646 = extractelement <4 x half> %1643, i64 1, !dbg !28
  %1647 = extractelement <4 x half> %1643, i64 0, !dbg !28
  %1648 = getelementptr inbounds nuw i8, ptr addrspace(1) %3, i64 %.idx24.pre-phi, !dbg !13
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1648, i64 %address.local403, !dbg !13
  %address.local463 = or disjoint i64 %address.local407, 32, !dbg !13
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local463, !dbg !13
  store half %1647, ptr addrspace(1) %address.local464, align 2, !dbg !13
  %address.local466 = or disjoint i64 %address.local407, 33, !dbg !13
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local466, !dbg !13
  store half %1646, ptr addrspace(1) %address.local467, align 2, !dbg !13
  %address.local469 = or disjoint i64 %address.local407, 34, !dbg !13
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local469, !dbg !13
  store half %1645, ptr addrspace(1) %address.local470, align 2, !dbg !13
  %address.local472 = or disjoint i64 %address.local407, 35, !dbg !13
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local472, !dbg !13
  store half %1644, ptr addrspace(1) %address.local473, align 2, !dbg !13
  %address.local475 = or disjoint i64 %address.local407, 48, !dbg !13
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local475, !dbg !13
  store half %1642, ptr addrspace(1) %address.local476, align 2, !dbg !13
  %address.local478 = or disjoint i64 %address.local407, 49, !dbg !13
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local478, !dbg !13
  store half %1641, ptr addrspace(1) %address.local479, align 2, !dbg !13
  %address.local481 = or disjoint i64 %address.local407, 50, !dbg !13
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local481, !dbg !13
  store half %1640, ptr addrspace(1) %address.local482, align 2, !dbg !13
  %address.local484 = or i64 %1626, 51, !dbg !13
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local484, !dbg !13
  store half %1639, ptr addrspace(1) %address.local485, align 2, !dbg !13
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1648, i64 %address.local431, !dbg !13
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local463, !dbg !13
  store half %1637, ptr addrspace(1) %address.local490, align 2, !dbg !13
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local466, !dbg !13
  store half %1636, ptr addrspace(1) %address.local492, align 2, !dbg !13
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local469, !dbg !13
  store half %1635, ptr addrspace(1) %address.local494, align 2, !dbg !13
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local472, !dbg !13
  store half %1634, ptr addrspace(1) %address.local496, align 2, !dbg !13
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local475, !dbg !13
  store half %1632, ptr addrspace(1) %address.local498, align 2, !dbg !13
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local478, !dbg !13
  store half %1631, ptr addrspace(1) %address.local500, align 2, !dbg !13
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local481, !dbg !13
  store half %1630, ptr addrspace(1) %address.local502, align 2, !dbg !13
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local484, !dbg !13
  store half %1629, ptr addrspace(1) %address.local504, align 2, !dbg !13
  %normalized86 = fmul <4 x float> %row.reciprocal, %1504, !dbg !27
  %1649 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !28
  %1650 = extractelement <4 x half> %1649, i64 3, !dbg !28
  %1651 = extractelement <4 x half> %1649, i64 2, !dbg !28
  %1652 = extractelement <4 x half> %1649, i64 1, !dbg !28
  %1653 = extractelement <4 x half> %1649, i64 0, !dbg !28
  %normalized87 = fmul <4 x float> %row.reciprocal, %1379, !dbg !27
  %1654 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !28
  %1655 = extractelement <4 x half> %1654, i64 3, !dbg !28
  %1656 = extractelement <4 x half> %1654, i64 2, !dbg !28
  %1657 = extractelement <4 x half> %1654, i64 1, !dbg !28
  %1658 = extractelement <4 x half> %1654, i64 0, !dbg !28
  %normalized89 = fmul <4 x float> %row.reciprocal76, %1256, !dbg !27
  %1659 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !28
  %1660 = extractelement <4 x half> %1659, i64 3, !dbg !28
  %1661 = extractelement <4 x half> %1659, i64 2, !dbg !28
  %1662 = extractelement <4 x half> %1659, i64 1, !dbg !28
  %1663 = extractelement <4 x half> %1659, i64 0, !dbg !28
  %normalized90 = fmul <4 x float> %row.reciprocal76, %1131, !dbg !27
  %1664 = fptrunc <4 x float> %normalized90 to <4 x half>, !dbg !28
  %1665 = extractelement <4 x half> %1664, i64 3, !dbg !28
  %1666 = extractelement <4 x half> %1664, i64 2, !dbg !28
  %1667 = extractelement <4 x half> %1664, i64 1, !dbg !28
  %1668 = extractelement <4 x half> %1664, i64 0, !dbg !28
  %address.local519 = or disjoint i64 %address.local407, 64, !dbg !13
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local519, !dbg !13
  store half %1668, ptr addrspace(1) %address.local520, align 2, !dbg !13
  %address.local522 = or disjoint i64 %address.local407, 65, !dbg !13
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local522, !dbg !13
  store half %1667, ptr addrspace(1) %address.local523, align 2, !dbg !13
  %address.local525 = or disjoint i64 %address.local407, 66, !dbg !13
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local525, !dbg !13
  store half %1666, ptr addrspace(1) %address.local526, align 2, !dbg !13
  %address.local528 = or disjoint i64 %address.local407, 67, !dbg !13
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local528, !dbg !13
  store half %1665, ptr addrspace(1) %address.local529, align 2, !dbg !13
  %address.local531 = or disjoint i64 %address.local407, 80, !dbg !13
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local531, !dbg !13
  store half %1663, ptr addrspace(1) %address.local532, align 2, !dbg !13
  %address.local534 = or disjoint i64 %address.local407, 81, !dbg !13
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local534, !dbg !13
  store half %1662, ptr addrspace(1) %address.local535, align 2, !dbg !13
  %address.local537 = or disjoint i64 %address.local407, 82, !dbg !13
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local537, !dbg !13
  store half %1661, ptr addrspace(1) %address.local538, align 2, !dbg !13
  %address.local540 = or disjoint i64 %address.local407, 83, !dbg !13
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local540, !dbg !13
  store half %1660, ptr addrspace(1) %address.local541, align 2, !dbg !13
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local519, !dbg !13
  store half %1658, ptr addrspace(1) %address.local546, align 2, !dbg !13
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local522, !dbg !13
  store half %1657, ptr addrspace(1) %address.local548, align 2, !dbg !13
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local525, !dbg !13
  store half %1656, ptr addrspace(1) %address.local550, align 2, !dbg !13
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local528, !dbg !13
  store half %1655, ptr addrspace(1) %address.local552, align 2, !dbg !13
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local531, !dbg !13
  store half %1653, ptr addrspace(1) %address.local554, align 2, !dbg !13
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local534, !dbg !13
  store half %1652, ptr addrspace(1) %address.local556, align 2, !dbg !13
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local537, !dbg !13
  store half %1651, ptr addrspace(1) %address.local558, align 2, !dbg !13
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local540, !dbg !13
  store half %1650, ptr addrspace(1) %address.local560, align 2, !dbg !13
  %normalized92 = fmul <4 x float> %row.reciprocal, %1471, !dbg !27
  %1669 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !28
  %1670 = extractelement <4 x half> %1669, i64 3, !dbg !28
  %1671 = extractelement <4 x half> %1669, i64 2, !dbg !28
  %1672 = extractelement <4 x half> %1669, i64 1, !dbg !28
  %1673 = extractelement <4 x half> %1669, i64 0, !dbg !28
  %normalized93 = fmul <4 x float> %row.reciprocal, %1345, !dbg !27
  %1674 = fptrunc <4 x float> %normalized93 to <4 x half>, !dbg !28
  %1675 = extractelement <4 x half> %1674, i64 3, !dbg !28
  %1676 = extractelement <4 x half> %1674, i64 2, !dbg !28
  %1677 = extractelement <4 x half> %1674, i64 1, !dbg !28
  %1678 = extractelement <4 x half> %1674, i64 0, !dbg !28
  %normalized95 = fmul <4 x float> %row.reciprocal76, %1223, !dbg !27
  %1679 = fptrunc <4 x float> %normalized95 to <4 x half>, !dbg !28
  %1680 = extractelement <4 x half> %1679, i64 3, !dbg !28
  %1681 = extractelement <4 x half> %1679, i64 2, !dbg !28
  %1682 = extractelement <4 x half> %1679, i64 1, !dbg !28
  %1683 = extractelement <4 x half> %1679, i64 0, !dbg !28
  %normalized96 = fmul <4 x float> %row.reciprocal76, %1097, !dbg !27
  %1684 = fptrunc <4 x float> %normalized96 to <4 x half>, !dbg !28
  %1685 = extractelement <4 x half> %1684, i64 3, !dbg !28
  %1686 = extractelement <4 x half> %1684, i64 2, !dbg !28
  %1687 = extractelement <4 x half> %1684, i64 1, !dbg !28
  %1688 = extractelement <4 x half> %1684, i64 0, !dbg !28
  %address.local575 = or disjoint i64 %address.local407, 96, !dbg !13
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local575, !dbg !13
  store half %1688, ptr addrspace(1) %address.local576, align 2, !dbg !13
  %address.local578 = or disjoint i64 %address.local407, 97, !dbg !13
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local578, !dbg !13
  store half %1687, ptr addrspace(1) %address.local579, align 2, !dbg !13
  %address.local581 = or disjoint i64 %address.local407, 98, !dbg !13
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local581, !dbg !13
  store half %1686, ptr addrspace(1) %address.local582, align 2, !dbg !13
  %address.local584 = or disjoint i64 %address.local407, 99, !dbg !13
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local584, !dbg !13
  store half %1685, ptr addrspace(1) %address.local585, align 2, !dbg !13
  %address.local587 = or disjoint i64 %address.local407, 112, !dbg !13
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local587, !dbg !13
  store half %1683, ptr addrspace(1) %address.local588, align 2, !dbg !13
  %address.local590 = or disjoint i64 %address.local407, 113, !dbg !13
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local590, !dbg !13
  store half %1682, ptr addrspace(1) %address.local591, align 2, !dbg !13
  %address.local593 = or disjoint i64 %address.local407, 114, !dbg !13
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local593, !dbg !13
  store half %1681, ptr addrspace(1) %address.local594, align 2, !dbg !13
  %address.local596 = or i64 %1626, 115, !dbg !13
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local596, !dbg !13
  store half %1680, ptr addrspace(1) %address.local597, align 2, !dbg !13
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local575, !dbg !13
  store half %1678, ptr addrspace(1) %address.local602, align 2, !dbg !13
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local578, !dbg !13
  store half %1677, ptr addrspace(1) %address.local604, align 2, !dbg !13
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local581, !dbg !13
  store half %1676, ptr addrspace(1) %address.local606, align 2, !dbg !13
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local584, !dbg !13
  store half %1675, ptr addrspace(1) %address.local608, align 2, !dbg !13
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local587, !dbg !13
  store half %1673, ptr addrspace(1) %address.local610, align 2, !dbg !13
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local590, !dbg !13
  store half %1672, ptr addrspace(1) %address.local612, align 2, !dbg !13
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local593, !dbg !13
  store half %1671, ptr addrspace(1) %address.local614, align 2, !dbg !13
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local596, !dbg !13
  store half %1670, ptr addrspace(1) %address.local616, align 2, !dbg !13
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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
