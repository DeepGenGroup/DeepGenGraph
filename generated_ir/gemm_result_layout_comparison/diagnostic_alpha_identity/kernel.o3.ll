; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/diagnostic_alpha_identity/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/run-20261027-121856-x88fub13/grid-000001/input.mlir"

@shm_0 = addrspace(3) global [65536 x i8] undef, align 16

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
  %12 = and i64 %11, 576460752303423484, !dbg !7
  %13 = and i64 %6, 3, !dbg !7
  %14 = or disjoint i64 %12, %13, !dbg !7
  %15 = lshr i64 %10, 2, !dbg !7
  %16 = sub nsw i64 31, %14, !dbg !7
  %17 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %18 = sext i32 %17 to i64, !dbg !7
  %19 = shl i64 %16, 7, !dbg !8
  %20 = icmp ugt i64 %10, 1023, !dbg !9
  %21 = add nsw i64 %14, -32, !dbg !9
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !9
  %23 = sdiv i64 %22, 32, !dbg !9
  %24 = sext i1 %20 to i64, !dbg !9
  %25 = xor i64 %23, %24, !dbg !9
  %26 = add nsw i64 %25, %15, !dbg !9
  %27 = shl i64 %25, 12, !dbg !9
  %28 = sub i64 %19, %27, !dbg !9
  %29 = shl i64 %26, 19, !dbg !9
  %30 = and i64 %29, 16252928, !dbg !9
  %31 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !10, !range !11
  %32 = zext nneg i32 %31 to i64, !dbg !10
  %33 = lshr i64 %32, 4, !dbg !10
  %34 = and i64 %33, 3, !dbg !10
  %35 = shl nuw nsw i64 %34, 2, !dbg !10
  %36 = or disjoint i64 %19, 127, !dbg !12
  %37 = add i64 %19, 128, !dbg !12
  %38 = icmp uge i64 %37, %36, !dbg !12
  %39 = icmp eq i64 %34, 0, !dbg !13
  %40 = and i32 %31, 16, !dbg !14
  %.not = icmp eq i32 %40, 0, !dbg !14
  %41 = and i32 %31, 32, !dbg !14
  %.not1 = icmp eq i32 %41, 0, !dbg !14
  %42 = sub nsw i64 32, %14, !dbg !15
  %.idx = shl i64 %28, 8, !dbg !9
  %43 = shl i64 %15, 7, !dbg !16
  %44 = and i64 %43, 3968, !dbg !16
  %45 = shl nsw i64 %18, 12, !dbg !16
  %46 = or disjoint i64 %44, %45, !dbg !16
  %47 = icmp ult i64 %10, 1024, !dbg !17
  br i1 %47, label %.lr.ph, label %._crit_edge, !dbg !17

.lr.ph:                                           ; preds = %4
  %48 = lshr i64 %32, 2, !dbg !18
  %49 = and i64 %48, 48, !dbg !18
  %invariant.op = add nuw nsw i64 %49, %32, !dbg !17
  %invariant.op95 = add nuw nsw i64 %invariant.op, %19, !dbg !17
  %50 = ptrtoint ptr addrspace(1) %0 to i64
  %51 = insertelement <2 x i64> poison, i64 %50, i64 0
  %52 = bitcast <2 x i64> %51 to <4 x i32>
  %dma.buffer.rsrc1807 = shufflevector <4 x i32> %52, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %53 = ptrtoint ptr addrspace(1) %1 to i64
  %54 = insertelement <2 x i64> poison, i64 %53, i64 0
  %55 = bitcast <2 x i64> %54 to <4 x i32>
  %dma.buffer.rsrc1801 = shufflevector <4 x i32> %55, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %.idx12 = shl nuw nsw i64 %30, 1
  %56 = add i64 %.idx12, %.idx
  %invariant.op101 = or disjoint i64 %56, 64, !dbg !17
  %invariant.op102 = or disjoint i64 %56, 128, !dbg !17
  %57 = and i64 %32, 240
  %.reass96 = sub i64 %invariant.op95, %57
  %58 = add i64 %.reass96, 1
  %59 = lshr i64 %58, 32
  %compare.high = trunc nuw i64 %59 to i32
  %compare.low.base = trunc i64 %58 to i32
  %60 = add i64 %.reass96, 65
  %61 = lshr i64 %60, 32
  %compare.high1413 = trunc nuw i64 %61 to i32
  %compare.low.base1417 = trunc i64 %60 to i32
  %62 = ptrtoint ptr addrspace(1) %2 to i64
  %63 = insertelement <2 x i64> poison, i64 %62, i64 0
  %64 = bitcast <2 x i64> %63 to <4 x i32>
  %dma.buffer.rsrc1759 = shufflevector <4 x i32> %64, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %65 = insertelement <4 x i1> poison, i1 %.not1, i64 0, !dbg !14
  %66 = shufflevector <4 x i1> %65, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !14
  %67 = insertelement <32 x i32> poison, i32 %compare.low.base1417, i64 0, !dbg !19
  %68 = shufflevector <32 x i32> %67, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !19
  %69 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !19
  %70 = shufflevector <32 x i32> %69, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !19
  br label %71, !dbg !17

71:                                               ; preds = %.lr.ph, %1937
  %72 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1973, %1937 ]
  %73 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1969, %1937 ]
  %74 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1965, %1937 ]
  %75 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1961, %1937 ]
  %76 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1959, %1937 ]
  %77 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1955, %1937 ]
  %78 = phi i64 [ 0, %.lr.ph ], [ %1974, %1937 ]
  %79 = shl nuw nsw i64 %78, 7, !dbg !17
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !9, !range !11, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !10
  %address.local257 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local258 = and i64 %address.local257, 1536, !dbg !9
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local258, !dbg !9
  %address.local262 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local263 = and i64 %address.local262, 16256, !dbg !9
  %address.local264 = lshr i64 %address.local, 4, !dbg !9
  %address.local265 = and i64 %address.local264, 8, !dbg !9
  %address.local266 = or disjoint i64 %address.local263, %address.local265, !dbg !9
  %80 = or disjoint i64 %address.local266, %30, !dbg !9
  %81 = shl nuw nsw i64 %80, 1, !dbg !9
  %82 = add i64 %81, %.idx, !dbg !9
  %dma.buffer.offset1806 = trunc i64 %82 to i32, !dbg !9
  %address.local268 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local269 = shl nuw nsw i64 %address.local268, 3, !dbg !9
  %address.local270 = and i64 %address.local269, 3584, !dbg !9
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local270, !dbg !9
  %address.local276 = lshr i64 %address.local268, 4, !dbg !9
  %address.local277 = and i64 %address.local276, 24, !dbg !9
  %address.local278 = or disjoint i64 %address.local277, %address.local263, !dbg !9
  %83 = or disjoint i64 %address.local278, %30, !dbg !9
  %84 = shl nuw nsw i64 %83, 1, !dbg !9
  %85 = add i64 %84, %.idx, !dbg !9
  %dma.buffer.offset1803 = trunc i64 %85 to i32, !dbg !9
  %86 = lshr i64 %78, 12, !dbg !16
  %87 = add nuw nsw i64 %86, %15, !dbg !16
  %88 = lshr i64 %78, 5, !dbg !16
  %89 = and i64 %88, 127, !dbg !16
  %90 = shl i64 %87, 19, !dbg !16
  %91 = and i64 %90, 16252928, !dbg !16
  %92 = shl i64 %88, 19, !dbg !16
  %93 = shl nuw nsw i64 %78, 14, !dbg !16
  %94 = sub i64 %93, %92, !dbg !16
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local258, !dbg !16
  %address.local285 = and i64 %address.local, 123, !dbg !16
  %address.local286 = lshr i64 %address.local, 2, !dbg !16
  %address.local287 = and i64 %address.local286, 4, !dbg !16
  %address.local288 = add nuw nsw i64 %address.local287, %address.local, !dbg !16
  %address.local289 = and i64 %address.local288, 4, !dbg !16
  %address.local290 = or disjoint i64 %address.local289, %address.local285, !dbg !16
  %address.local291 = shl nuw nsw i64 %address.local290, 7, !dbg !16
  %address.local292 = or disjoint i64 %address.local291, %address.local265, !dbg !16
  %.idx36 = or disjoint i64 %91, %89, !dbg !16
  %95 = add i64 %.idx36, %94, !dbg !16
  %96 = add i64 %address.local292, %95, !dbg !16
  %.tr = trunc i64 %96 to i32, !dbg !16
  %dma.buffer.offset1800 = shl i32 %.tr, 1, !dbg !16
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local270, !dbg !16
  %97 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local259, i64 0, !dbg !9
  %98 = insertelement <4 x ptr addrspace(3)> %97, ptr addrspace(3) %address.local271, i64 1, !dbg !9
  %99 = insertelement <4 x ptr addrspace(3)> %98, ptr addrspace(3) %address.local280, i64 2, !dbg !9
  %100 = insertelement <4 x ptr addrspace(3)> %99, ptr addrspace(3) %address.local294, i64 3, !dbg !9
  %101 = ptrtoint <4 x ptr addrspace(3)> %100 to <4 x i64>, !dbg !9
  %102 = trunc <4 x i64> %101 to <4 x i32>, !dbg !9
  %103 = extractelement <4 x i32> %102, i64 0, !dbg !9
  %104 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %103), !dbg !9
  %dma.buffer.lds1808 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %104), !dbg !9
  %105 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1806, <4 x i32> %dma.buffer.rsrc1807) #9, !dbg !9
  %106 = extractelement <4 x i32> %102, i64 1, !dbg !9
  %107 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %106), !dbg !9
  %dma.buffer.lds1805 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %107), !dbg !9
  %108 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1803, <4 x i32> %dma.buffer.rsrc1807) #9, !dbg !9
  %109 = extractelement <4 x i32> %102, i64 2, !dbg !16
  %110 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %109), !dbg !16
  %dma.buffer.lds1802 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %110), !dbg !16
  %111 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1800, <4 x i32> %dma.buffer.rsrc1801) #9, !dbg !16
  %112 = extractelement <4 x i32> %102, i64 3, !dbg !16
  %113 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %112), !dbg !16
  %address.local306 = or disjoint i64 %address.local291, %address.local277, !dbg !16
  %114 = add i64 %address.local306, %95, !dbg !16
  %.tr11 = trunc i64 %114 to i32, !dbg !16
  %dma.buffer.offset1797 = shl i32 %.tr11, 1, !dbg !16
  %dma.buffer.lds1799 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %113), !dbg !16
  %115 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1797, <4 x i32> %dma.buffer.rsrc1801) #9, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local308 = lshr i32 %address.thread, 2, !dbg !10
  %address.local309 = and i32 %address.local308, 4, !dbg !10
  %address.local310 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local311 = and i32 %address.local310, 384, !dbg !10
  %address.local313 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local314 = and i32 %address.local313, 120, !dbg !10
  %address.local316 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local317 = and i32 %address.local316, 1024, !dbg !10
  %116 = or disjoint i32 %address.local309, %address.local311, !dbg !10
  %address.local315 = or disjoint i32 %116, %address.local314, !dbg !10
  %address.local318 = or disjoint i32 %address.local315, %address.local317, !dbg !10
  %address.local319 = zext nneg i32 %address.local318 to i64, !dbg !10
  %address.local320 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local319, !dbg !10
  %117 = load <4 x half>, ptr addrspace(3) %address.local320, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local320, i64 4096, !dbg !10
  %118 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local320, i64 1024, !dbg !10
  %119 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !10
  %120 = fmul <4 x half> %117, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1546 = getelementptr i8, ptr addrspace(3) %address.local320, i64 5120, !dbg !10
  %121 = load <4 x half>, ptr addrspace(3) %lds.const.gep1546, align 8, !dbg !10
  %122 = fmul <4 x half> %118, splat (half 1.275630e-01), !dbg !10
  %123 = fmul <4 x half> %119, splat (half 1.275630e-01), !dbg !10
  %124 = fmul <4 x half> %121, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local331 = or disjoint i32 %address.local309, %address.local314, !dbg !20
  %address.local334 = or disjoint i32 %address.local331, %address.local317, !dbg !20
  %address.local335 = zext nneg i32 %address.local334 to i64, !dbg !20
  %address.local336 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local335, !dbg !20
  %125 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !20
  %126 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %125, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1547 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4096, !dbg !20
  %127 = load <4 x half>, ptr addrspace(3) %lds.const.gep1547, align 8, !dbg !20
  %address.local351 = and i32 %address.local313, 88, !dbg !20
  %address.local352 = or disjoint i32 %address.local309, %address.local351, !dbg !20
  %address.local354 = and i32 %address.local313, 32, !dbg !20
  %address.local355 = sub nsw i32 %address.local352, %address.local354, !dbg !20
  %address.local358 = add nsw i32 %address.local355, %address.local317, !dbg !20
  %address.local359 = sext i32 %address.local358 to i64, !dbg !20
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local359, !dbg !20
  %address.local361 = getelementptr i8, ptr addrspace(3) %address.local360, i64 320, !dbg !20
  %128 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !20
  %129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %127, <4 x float> %126) #8, !dbg !20
  %130 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %128, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1548 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4416, !dbg !20
  %131 = load <4 x half>, ptr addrspace(3) %lds.const.gep1548, align 8, !dbg !20
  %lds.const.gep1549 = getelementptr i8, ptr addrspace(3) %address.local336, i64 512, !dbg !20
  %132 = load <4 x half>, ptr addrspace(3) %lds.const.gep1549, align 8, !dbg !20
  %133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %131, <4 x float> %130) #8, !dbg !20
  %134 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %132, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1550 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4608, !dbg !20
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1550, align 8, !dbg !20
  %lds.const.gep1551 = getelementptr i8, ptr addrspace(3) %address.local360, i64 832, !dbg !20
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1551, align 8, !dbg !20
  %137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %135, <4 x float> %134) #8, !dbg !20
  %138 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %136, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1552 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4928, !dbg !20
  %139 = load <4 x half>, ptr addrspace(3) %lds.const.gep1552, align 8, !dbg !20
  %lds.const.gep1553 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1024, !dbg !20
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep1553, align 8, !dbg !20
  %141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %139, <4 x float> %138) #8, !dbg !20
  %142 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %140, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1554 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5120, !dbg !20
  %143 = load <4 x half>, ptr addrspace(3) %lds.const.gep1554, align 8, !dbg !20
  %lds.const.gep1555 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1344, !dbg !20
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep1555, align 8, !dbg !20
  %145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %143, <4 x float> %142) #8, !dbg !20
  %146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %144, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1556 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5440, !dbg !20
  %147 = load <4 x half>, ptr addrspace(3) %lds.const.gep1556, align 8, !dbg !20
  %lds.const.gep1557 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1536, !dbg !20
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1557, align 8, !dbg !20
  %149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %147, <4 x float> %146) #8, !dbg !20
  %150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %148, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1558 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5632, !dbg !20
  %151 = load <4 x half>, ptr addrspace(3) %lds.const.gep1558, align 8, !dbg !20
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1856, !dbg !20
  %152 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !20
  %153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %151, <4 x float> %150) #8, !dbg !20
  %154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %152, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5952, !dbg !20
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !20
  %156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %155, <4 x float> %154) #8, !dbg !20
  %157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %125, <4 x float> zeroinitializer) #8, !dbg !20
  %158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %127, <4 x float> %157) #8, !dbg !20
  %159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %128, <4 x float> zeroinitializer) #8, !dbg !20
  %160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %131, <4 x float> %159) #8, !dbg !20
  %161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %132, <4 x float> zeroinitializer) #8, !dbg !20
  %162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %135, <4 x float> %161) #8, !dbg !20
  %163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %136, <4 x float> zeroinitializer) #8, !dbg !20
  %164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %139, <4 x float> %163) #8, !dbg !20
  %165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %140, <4 x float> zeroinitializer) #8, !dbg !20
  %166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %143, <4 x float> %165) #8, !dbg !20
  %167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %144, <4 x float> zeroinitializer) #8, !dbg !20
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %147, <4 x float> %167) #8, !dbg !20
  %169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %148, <4 x float> zeroinitializer) #8, !dbg !20
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %151, <4 x float> %169) #8, !dbg !20
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %152, <4 x float> zeroinitializer) #8, !dbg !20
  %172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %155, <4 x float> %171) #8, !dbg !20
  %address.local400.idx = shl nuw nsw i64 %address.local266, 1, !dbg !9
  %173 = or disjoint i64 %address.local400.idx, %56, !dbg !9
  %174 = trunc i64 %173 to i32, !dbg !9
  %dma.buffer.offset1794 = or disjoint i32 %174, 64, !dbg !9
  %175 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1794, <4 x i32> %dma.buffer.rsrc1807) #9, !dbg !9
  %address.local401.idx = shl nuw nsw i64 %address.local278, 1, !dbg !9
  %.reass = or disjoint i64 %address.local401.idx, %invariant.op101
  %dma.buffer.offset1791 = trunc i64 %.reass to i32, !dbg !9
  %176 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1791, <4 x i32> %dma.buffer.rsrc1807) #9, !dbg !9
  %177 = add nuw nsw i64 %78, 1024, !dbg !16
  %178 = lshr i64 %177, 12, !dbg !16
  %179 = add nuw nsw i64 %178, %15, !dbg !16
  %180 = add nsw i64 %88, %46, !dbg !16
  %181 = shl i64 %179, 19, !dbg !16
  %182 = and i64 %181, 16252928, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %183 = add i64 %94, %88, !dbg !16
  %.idx1418 = add i64 %183, %46, !dbg !16
  %184 = add i64 %.idx1418, %182, !dbg !16
  %dma.buffer.offset1782 = or disjoint i32 %174, 128, !dbg !9
  %.reass103 = or disjoint i64 %address.local401.idx, %invariant.op102
  %dma.buffer.offset1779 = trunc i64 %.reass103 to i32, !dbg !9
  %185 = add nuw nsw i64 %78, 2048, !dbg !16
  %186 = lshr i64 %185, 12, !dbg !16
  %187 = add nuw nsw i64 %186, %15, !dbg !16
  %188 = insertelement <2 x i64> poison, i64 %180, i64 0, !dbg !16
  %189 = shufflevector <2 x i64> %188, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !16
  %190 = icmp slt <2 x i64> %189, <i64 -32, i64 -64>, !dbg !16
  %191 = extractelement <2 x i1> %190, i64 0, !dbg !16
  %192 = sext i1 %191 to i64, !dbg !16
  %193 = add nsw <2 x i64> %189, <i64 32, i64 64>, !dbg !16
  %194 = sub nsw <2 x i64> <i64 -33, i64 -65>, %189, !dbg !16
  %195 = select <2 x i1> %190, <2 x i64> %194, <2 x i64> %193, !dbg !16
  %196 = sdiv <2 x i64> %195, splat (i64 128), !dbg !16
  %197 = extractelement <2 x i64> %196, i64 0, !dbg !16
  %198 = xor i64 %197, %192, !dbg !16
  %199 = shl nsw i64 %198, 7, !dbg !16
  %200 = sub i64 %184, %199, !dbg !16
  %201 = add i64 %address.local292, %200, !dbg !16
  %.tr19 = trunc i64 %201 to i32, !dbg !16
  %202 = shl i32 %.tr19, 1, !dbg !16
  %dma.buffer.offset1788 = add i32 %202, 64, !dbg !16
  %203 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1788, <4 x i32> %dma.buffer.rsrc1801) #9, !dbg !16
  %204 = add i64 %address.local306, %200, !dbg !16
  %.tr25 = trunc i64 %204 to i32, !dbg !16
  %205 = shl i32 %.tr25, 1, !dbg !16
  %dma.buffer.offset1785 = add i32 %205, 64, !dbg !16
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1785, <4 x i32> %dma.buffer.rsrc1801) #9, !dbg !16
  %207 = load <4 x half>, ptr addrspace(3) %address.local320, align 8, !dbg !10
  %208 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %209 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !10
  %210 = fmul <4 x half> %207, splat (half 1.275630e-01), !dbg !10
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1546, align 8, !dbg !10
  %212 = fmul <4 x half> %208, splat (half 1.275630e-01), !dbg !10
  %213 = fmul <4 x half> %209, splat (half 1.275630e-01), !dbg !10
  %214 = fmul <4 x half> %211, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %215 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !20
  %216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %215, <4 x float> zeroinitializer) #8, !dbg !20
  %217 = load <4 x half>, ptr addrspace(3) %lds.const.gep1547, align 8, !dbg !20
  %218 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !20
  %219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %217, <4 x float> %216) #8, !dbg !20
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %218, <4 x float> zeroinitializer) #8, !dbg !20
  %221 = load <4 x half>, ptr addrspace(3) %lds.const.gep1548, align 8, !dbg !20
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1549, align 8, !dbg !20
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %221, <4 x float> %220) #8, !dbg !20
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %222, <4 x float> zeroinitializer) #8, !dbg !20
  %225 = load <4 x half>, ptr addrspace(3) %lds.const.gep1550, align 8, !dbg !20
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1551, align 8, !dbg !20
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %225, <4 x float> %224) #8, !dbg !20
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %226, <4 x float> zeroinitializer) #8, !dbg !20
  %229 = load <4 x half>, ptr addrspace(3) %lds.const.gep1552, align 8, !dbg !20
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1553, align 8, !dbg !20
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %229, <4 x float> %228) #8, !dbg !20
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %230, <4 x float> zeroinitializer) #8, !dbg !20
  %233 = load <4 x half>, ptr addrspace(3) %lds.const.gep1554, align 8, !dbg !20
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep1555, align 8, !dbg !20
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %233, <4 x float> %232) #8, !dbg !20
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %234, <4 x float> zeroinitializer) #8, !dbg !20
  %237 = load <4 x half>, ptr addrspace(3) %lds.const.gep1556, align 8, !dbg !20
  %238 = load <4 x half>, ptr addrspace(3) %lds.const.gep1557, align 8, !dbg !20
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %237, <4 x float> %236) #8, !dbg !20
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %238, <4 x float> zeroinitializer) #8, !dbg !20
  %241 = load <4 x half>, ptr addrspace(3) %lds.const.gep1558, align 8, !dbg !20
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !20
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %241, <4 x float> %240) #8, !dbg !20
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %242, <4 x float> zeroinitializer) #8, !dbg !20
  %245 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !20
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %245, <4 x float> %244) #8, !dbg !20
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %215, <4 x float> zeroinitializer) #8, !dbg !20
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %217, <4 x float> %247) #8, !dbg !20
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %218, <4 x float> zeroinitializer) #8, !dbg !20
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %221, <4 x float> %249) #8, !dbg !20
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %222, <4 x float> zeroinitializer) #8, !dbg !20
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %225, <4 x float> %251) #8, !dbg !20
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %226, <4 x float> zeroinitializer) #8, !dbg !20
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %229, <4 x float> %253) #8, !dbg !20
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %230, <4 x float> zeroinitializer) #8, !dbg !20
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %233, <4 x float> %255) #8, !dbg !20
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %234, <4 x float> zeroinitializer) #8, !dbg !20
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %237, <4 x float> %257) #8, !dbg !20
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %238, <4 x float> zeroinitializer) #8, !dbg !20
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %241, <4 x float> %259) #8, !dbg !20
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %242, <4 x float> zeroinitializer) #8, !dbg !20
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %245, <4 x float> %261) #8, !dbg !20
  %263 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1782, <4 x i32> %dma.buffer.rsrc1807) #9, !dbg !9
  %264 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1779, <4 x i32> %dma.buffer.rsrc1807) #9, !dbg !9
  %265 = extractelement <2 x i1> %190, i64 1, !dbg !16
  %266 = sext i1 %265 to i64, !dbg !16
  %267 = extractelement <2 x i64> %196, i64 1, !dbg !16
  %268 = xor i64 %267, %266, !dbg !16
  %269 = shl i64 %187, 19, !dbg !16
  %270 = and i64 %269, 16252928, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %271 = add i64 %.idx1418, %270, !dbg !16
  %272 = shl nsw i64 %268, 7, !dbg !16
  %273 = sub i64 %271, %272, !dbg !16
  %274 = add i64 %address.local292, %273, !dbg !16
  %.tr33 = trunc i64 %274 to i32, !dbg !16
  %275 = shl i32 %.tr33, 1, !dbg !16
  %dma.buffer.offset1776 = add i32 %275, 128, !dbg !16
  %276 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1776, <4 x i32> %dma.buffer.rsrc1801) #9, !dbg !16
  %277 = add i64 %address.local306, %273, !dbg !16
  %.tr40 = trunc i64 %277 to i32, !dbg !16
  %278 = shl i32 %.tr40, 1, !dbg !16
  %dma.buffer.offset1773 = add i32 %278, 128, !dbg !16
  %279 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1773, <4 x i32> %dma.buffer.rsrc1801) #9, !dbg !16
  %280 = load <4 x half>, ptr addrspace(3) %address.local320, align 8, !dbg !10
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !10
  %283 = fmul <4 x half> %280, splat (half 1.275630e-01), !dbg !10
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep1546, align 8, !dbg !10
  %285 = fmul <4 x half> %281, splat (half 1.275630e-01), !dbg !10
  %286 = fmul <4 x half> %282, splat (half 1.275630e-01), !dbg !10
  %287 = fmul <4 x half> %284, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %288 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !20
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %288, <4 x float> zeroinitializer) #8, !dbg !20
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1547, align 8, !dbg !20
  %291 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !20
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %290, <4 x float> %289) #8, !dbg !20
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %291, <4 x float> zeroinitializer) #8, !dbg !20
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1548, align 8, !dbg !20
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1549, align 8, !dbg !20
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %294, <4 x float> %293) #8, !dbg !20
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %295, <4 x float> zeroinitializer) #8, !dbg !20
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1550, align 8, !dbg !20
  %299 = load <4 x half>, ptr addrspace(3) %lds.const.gep1551, align 8, !dbg !20
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %298, <4 x float> %297) #8, !dbg !20
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %299, <4 x float> zeroinitializer) #8, !dbg !20
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1552, align 8, !dbg !20
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep1553, align 8, !dbg !20
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %302, <4 x float> %301) #8, !dbg !20
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %303, <4 x float> zeroinitializer) #8, !dbg !20
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1554, align 8, !dbg !20
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep1555, align 8, !dbg !20
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %306, <4 x float> %305) #8, !dbg !20
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %307, <4 x float> zeroinitializer) #8, !dbg !20
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep1556, align 8, !dbg !20
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep1557, align 8, !dbg !20
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %310, <4 x float> %309) #8, !dbg !20
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %311, <4 x float> zeroinitializer) #8, !dbg !20
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1558, align 8, !dbg !20
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !20
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %314, <4 x float> %313) #8, !dbg !20
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %315, <4 x float> zeroinitializer) #8, !dbg !20
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !20
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %318, <4 x float> %317) #8, !dbg !20
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %288, <4 x float> zeroinitializer) #8, !dbg !20
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %290, <4 x float> %320) #8, !dbg !20
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %291, <4 x float> zeroinitializer) #8, !dbg !20
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %294, <4 x float> %322) #8, !dbg !20
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %295, <4 x float> zeroinitializer) #8, !dbg !20
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %298, <4 x float> %324) #8, !dbg !20
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %299, <4 x float> zeroinitializer) #8, !dbg !20
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %302, <4 x float> %326) #8, !dbg !20
  %328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %303, <4 x float> zeroinitializer) #8, !dbg !20
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %306, <4 x float> %328) #8, !dbg !20
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %307, <4 x float> zeroinitializer) #8, !dbg !20
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %310, <4 x float> %330) #8, !dbg !20
  %332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %311, <4 x float> zeroinitializer) #8, !dbg !20
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %314, <4 x float> %332) #8, !dbg !20
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %315, <4 x float> zeroinitializer) #8, !dbg !20
  %335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %318, <4 x float> %334) #8, !dbg !20
  %dma.buffer.offset1770 = or disjoint i32 %174, 192, !dbg !9
  %336 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1770, <4 x i32> %dma.buffer.rsrc1807) #9, !dbg !9
  %337 = or disjoint i64 %address.local401.idx, %56, !dbg !9
  %338 = trunc i64 %337 to i32, !dbg !9
  %dma.buffer.offset1767 = or disjoint i32 %338, 192, !dbg !9
  %339 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1767, <4 x i32> %dma.buffer.rsrc1807) #9, !dbg !9
  %340 = add nuw nsw i64 %78, 3072, !dbg !16
  %341 = lshr i64 %340, 12, !dbg !16
  %342 = add nuw nsw i64 %341, %15, !dbg !16
  %343 = add nsw i64 %180, 96, !dbg !16
  %344 = icmp slt i64 %180, -96, !dbg !16
  %345 = sub nsw i64 -97, %180, !dbg !16
  %346 = select i1 %344, i64 %345, i64 %343, !dbg !16
  %347 = sdiv i64 %346, 128, !dbg !16
  %348 = sext i1 %344 to i64, !dbg !16
  %349 = xor i64 %347, %348, !dbg !16
  %350 = shl i64 %342, 19, !dbg !16
  %351 = and i64 %350, 16252928, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %352 = add i64 %.idx1418, %351, !dbg !16
  %353 = shl nsw i64 %349, 7, !dbg !16
  %354 = sub i64 %352, %353, !dbg !16
  %355 = add i64 %address.local292, %354, !dbg !16
  %.tr48 = trunc i64 %355 to i32, !dbg !16
  %356 = shl i32 %.tr48, 1, !dbg !16
  %dma.buffer.offset1764 = add i32 %356, 192, !dbg !16
  %357 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1764, <4 x i32> %dma.buffer.rsrc1801) #9, !dbg !16
  %358 = add i64 %address.local306, %354, !dbg !16
  %.tr54 = trunc i64 %358 to i32, !dbg !16
  %359 = shl i32 %.tr54, 1, !dbg !16
  %dma.buffer.offset1761 = add i32 %359, 192, !dbg !16
  %360 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1761, <4 x i32> %dma.buffer.rsrc1801) #9, !dbg !16
  %361 = load <4 x half>, ptr addrspace(3) %address.local320, align 8, !dbg !10
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !10
  %364 = fmul <4 x half> %361, splat (half 1.275630e-01), !dbg !10
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1546, align 8, !dbg !10
  %366 = fmul <4 x half> %362, splat (half 1.275630e-01), !dbg !10
  %367 = fmul <4 x half> %363, splat (half 1.275630e-01), !dbg !10
  %368 = fmul <4 x half> %365, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %369 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !20
  %370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %369, <4 x float> zeroinitializer) #8, !dbg !20
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1547, align 8, !dbg !20
  %372 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !20
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %371, <4 x float> %370) #8, !dbg !20
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %372, <4 x float> zeroinitializer) #8, !dbg !20
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep1548, align 8, !dbg !20
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1549, align 8, !dbg !20
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %375, <4 x float> %374) #8, !dbg !20
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %376, <4 x float> zeroinitializer) #8, !dbg !20
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep1550, align 8, !dbg !20
  %380 = load <4 x half>, ptr addrspace(3) %lds.const.gep1551, align 8, !dbg !20
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %379, <4 x float> %378) #8, !dbg !20
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %380, <4 x float> zeroinitializer) #8, !dbg !20
  %383 = load <4 x half>, ptr addrspace(3) %lds.const.gep1552, align 8, !dbg !20
  %384 = load <4 x half>, ptr addrspace(3) %lds.const.gep1553, align 8, !dbg !20
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %383, <4 x float> %382) #8, !dbg !20
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %384, <4 x float> zeroinitializer) #8, !dbg !20
  %387 = load <4 x half>, ptr addrspace(3) %lds.const.gep1554, align 8, !dbg !20
  %388 = load <4 x half>, ptr addrspace(3) %lds.const.gep1555, align 8, !dbg !20
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %387, <4 x float> %386) #8, !dbg !20
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %388, <4 x float> zeroinitializer) #8, !dbg !20
  %391 = load <4 x half>, ptr addrspace(3) %lds.const.gep1556, align 8, !dbg !20
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep1557, align 8, !dbg !20
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %391, <4 x float> %390) #8, !dbg !20
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %392, <4 x float> zeroinitializer) #8, !dbg !20
  %395 = load <4 x half>, ptr addrspace(3) %lds.const.gep1558, align 8, !dbg !20
  %396 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !20
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %395, <4 x float> %394) #8, !dbg !20
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %396, <4 x float> zeroinitializer) #8, !dbg !20
  %399 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !20
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %399, <4 x float> %398) #8, !dbg !20
  %401 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %369, <4 x float> zeroinitializer) #8, !dbg !20
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %371, <4 x float> %401) #8, !dbg !20
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %372, <4 x float> zeroinitializer) #8, !dbg !20
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %375, <4 x float> %403) #8, !dbg !20
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %376, <4 x float> zeroinitializer) #8, !dbg !20
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %379, <4 x float> %405) #8, !dbg !20
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %380, <4 x float> zeroinitializer) #8, !dbg !20
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %383, <4 x float> %407) #8, !dbg !20
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %384, <4 x float> zeroinitializer) #8, !dbg !20
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %387, <4 x float> %409) #8, !dbg !20
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %388, <4 x float> zeroinitializer) #8, !dbg !20
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %391, <4 x float> %411) #8, !dbg !20
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %392, <4 x float> zeroinitializer) #8, !dbg !20
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %395, <4 x float> %413) #8, !dbg !20
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %396, <4 x float> zeroinitializer) #8, !dbg !20
  %416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %399, <4 x float> %415) #8, !dbg !20
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %417 = fadd <4 x float> %172, zeroinitializer, !dbg !21
  %418 = fadd <4 x float> %417, %262, !dbg !21
  %419 = fadd <4 x float> %418, %335, !dbg !21
  %420 = fadd <4 x float> %419, %416, !dbg !21
  %421 = extractelement <4 x float> %420, i64 3, !dbg !21
  %422 = extractelement <4 x float> %420, i64 2, !dbg !21
  %423 = extractelement <4 x float> %420, i64 1, !dbg !21
  %424 = extractelement <4 x float> %420, i64 0, !dbg !21
  %425 = fadd <4 x float> %170, zeroinitializer, !dbg !21
  %426 = fadd <4 x float> %425, %260, !dbg !21
  %427 = fadd <4 x float> %426, %333, !dbg !21
  %428 = fadd <4 x float> %427, %414, !dbg !21
  %429 = extractelement <4 x float> %428, i64 3, !dbg !21
  %430 = extractelement <4 x float> %428, i64 2, !dbg !21
  %431 = extractelement <4 x float> %428, i64 1, !dbg !21
  %432 = extractelement <4 x float> %428, i64 0, !dbg !21
  %433 = fadd <4 x float> %168, zeroinitializer, !dbg !21
  %434 = fadd <4 x float> %433, %258, !dbg !21
  %435 = fadd <4 x float> %434, %331, !dbg !21
  %436 = fadd <4 x float> %435, %412, !dbg !21
  %437 = extractelement <4 x float> %436, i64 3, !dbg !21
  %438 = extractelement <4 x float> %436, i64 2, !dbg !21
  %439 = extractelement <4 x float> %436, i64 1, !dbg !21
  %440 = extractelement <4 x float> %436, i64 0, !dbg !21
  %441 = fadd <4 x float> %166, zeroinitializer, !dbg !21
  %442 = fadd <4 x float> %441, %256, !dbg !21
  %443 = fadd <4 x float> %442, %329, !dbg !21
  %444 = fadd <4 x float> %443, %410, !dbg !21
  %445 = extractelement <4 x float> %444, i64 3, !dbg !21
  %446 = extractelement <4 x float> %444, i64 2, !dbg !21
  %447 = extractelement <4 x float> %444, i64 1, !dbg !21
  %448 = extractelement <4 x float> %444, i64 0, !dbg !21
  %449 = fadd <4 x float> %164, zeroinitializer, !dbg !21
  %450 = fadd <4 x float> %449, %254, !dbg !21
  %451 = fadd <4 x float> %450, %327, !dbg !21
  %452 = fadd <4 x float> %451, %408, !dbg !21
  %453 = extractelement <4 x float> %452, i64 3, !dbg !21
  %454 = extractelement <4 x float> %452, i64 2, !dbg !21
  %455 = extractelement <4 x float> %452, i64 1, !dbg !21
  %456 = extractelement <4 x float> %452, i64 0, !dbg !21
  %457 = fadd <4 x float> %162, zeroinitializer, !dbg !21
  %458 = fadd <4 x float> %457, %252, !dbg !21
  %459 = fadd <4 x float> %458, %325, !dbg !21
  %460 = fadd <4 x float> %459, %406, !dbg !21
  %461 = extractelement <4 x float> %460, i64 3, !dbg !21
  %462 = extractelement <4 x float> %460, i64 2, !dbg !21
  %463 = extractelement <4 x float> %460, i64 1, !dbg !21
  %464 = extractelement <4 x float> %460, i64 0, !dbg !21
  %465 = fadd <4 x float> %160, zeroinitializer, !dbg !21
  %466 = fadd <4 x float> %465, %250, !dbg !21
  %467 = fadd <4 x float> %466, %323, !dbg !21
  %468 = fadd <4 x float> %467, %404, !dbg !21
  %469 = extractelement <4 x float> %468, i64 3, !dbg !21
  %470 = extractelement <4 x float> %468, i64 2, !dbg !21
  %471 = extractelement <4 x float> %468, i64 1, !dbg !21
  %472 = extractelement <4 x float> %468, i64 0, !dbg !21
  %473 = fadd <4 x float> %158, zeroinitializer, !dbg !21
  %474 = fadd <4 x float> %473, %248, !dbg !21
  %475 = fadd <4 x float> %474, %321, !dbg !21
  %476 = fadd <4 x float> %475, %402, !dbg !21
  %477 = extractelement <4 x float> %476, i64 3, !dbg !21
  %478 = extractelement <4 x float> %476, i64 2, !dbg !21
  %479 = extractelement <4 x float> %476, i64 1, !dbg !21
  %480 = extractelement <4 x float> %476, i64 0, !dbg !21
  %481 = fadd <4 x float> %156, zeroinitializer, !dbg !21
  %482 = fadd <4 x float> %481, %246, !dbg !21
  %483 = fadd <4 x float> %482, %319, !dbg !21
  %484 = fadd <4 x float> %483, %400, !dbg !21
  %485 = extractelement <4 x float> %484, i64 3, !dbg !21
  %486 = extractelement <4 x float> %484, i64 2, !dbg !21
  %487 = extractelement <4 x float> %484, i64 1, !dbg !21
  %488 = extractelement <4 x float> %484, i64 0, !dbg !21
  %489 = fadd <4 x float> %153, zeroinitializer, !dbg !21
  %490 = fadd <4 x float> %489, %243, !dbg !21
  %491 = fadd <4 x float> %490, %316, !dbg !21
  %492 = fadd <4 x float> %491, %397, !dbg !21
  %493 = extractelement <4 x float> %492, i64 3, !dbg !21
  %494 = extractelement <4 x float> %492, i64 2, !dbg !21
  %495 = extractelement <4 x float> %492, i64 1, !dbg !21
  %496 = extractelement <4 x float> %492, i64 0, !dbg !21
  %497 = fadd <4 x float> %149, zeroinitializer, !dbg !21
  %498 = fadd <4 x float> %497, %239, !dbg !21
  %499 = fadd <4 x float> %498, %312, !dbg !21
  %500 = fadd <4 x float> %499, %393, !dbg !21
  %501 = extractelement <4 x float> %500, i64 3, !dbg !21
  %502 = extractelement <4 x float> %500, i64 2, !dbg !21
  %503 = extractelement <4 x float> %500, i64 1, !dbg !21
  %504 = extractelement <4 x float> %500, i64 0, !dbg !21
  %505 = fadd <4 x float> %145, zeroinitializer, !dbg !21
  %506 = fadd <4 x float> %505, %235, !dbg !21
  %507 = fadd <4 x float> %506, %308, !dbg !21
  %508 = fadd <4 x float> %507, %389, !dbg !21
  %509 = extractelement <4 x float> %508, i64 3, !dbg !21
  %510 = extractelement <4 x float> %508, i64 2, !dbg !21
  %511 = extractelement <4 x float> %508, i64 1, !dbg !21
  %512 = extractelement <4 x float> %508, i64 0, !dbg !21
  %513 = fadd <4 x float> %141, zeroinitializer, !dbg !21
  %514 = fadd <4 x float> %513, %231, !dbg !21
  %515 = fadd <4 x float> %514, %304, !dbg !21
  %516 = fadd <4 x float> %515, %385, !dbg !21
  %517 = extractelement <4 x float> %516, i64 3, !dbg !21
  %518 = extractelement <4 x float> %516, i64 2, !dbg !21
  %519 = extractelement <4 x float> %516, i64 1, !dbg !21
  %520 = extractelement <4 x float> %516, i64 0, !dbg !21
  %521 = fadd <4 x float> %137, zeroinitializer, !dbg !21
  %522 = fadd <4 x float> %521, %227, !dbg !21
  %523 = fadd <4 x float> %522, %300, !dbg !21
  %524 = fadd <4 x float> %523, %381, !dbg !21
  %525 = extractelement <4 x float> %524, i64 3, !dbg !21
  %526 = extractelement <4 x float> %524, i64 2, !dbg !21
  %527 = extractelement <4 x float> %524, i64 1, !dbg !21
  %528 = extractelement <4 x float> %524, i64 0, !dbg !21
  %529 = fadd <4 x float> %133, zeroinitializer, !dbg !21
  %530 = fadd <4 x float> %529, %223, !dbg !21
  %531 = fadd <4 x float> %530, %296, !dbg !21
  %532 = fadd <4 x float> %531, %377, !dbg !21
  %533 = extractelement <4 x float> %532, i64 3, !dbg !21
  %534 = extractelement <4 x float> %532, i64 2, !dbg !21
  %535 = extractelement <4 x float> %532, i64 1, !dbg !21
  %536 = extractelement <4 x float> %532, i64 0, !dbg !21
  %537 = fadd <4 x float> %129, zeroinitializer, !dbg !21
  %538 = fadd <4 x float> %537, %219, !dbg !21
  %539 = fadd <4 x float> %538, %292, !dbg !21
  %540 = fadd <4 x float> %539, %373, !dbg !21
  %541 = extractelement <4 x float> %540, i64 3, !dbg !21
  %542 = extractelement <4 x float> %540, i64 2, !dbg !21
  %543 = extractelement <4 x float> %540, i64 1, !dbg !21
  %544 = extractelement <4 x float> %540, i64 0, !dbg !21
  %address.local414 = and i32 %address.local316, 6144, !dbg !18
  %address.local415 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local416 = and i32 %address.local415, 1920, !dbg !18
  %address.local417 = or disjoint i32 %address.local414, %address.local416, !dbg !18
  %address.local418 = zext nneg i32 %address.local417 to i64, !dbg !18
  %address.local419 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !18
  %address.local422 = and i64 %address.local264, 3, !dbg !10
  %address.local423 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local422, !dbg !18
  store float %544, ptr addrspace(3) %address.local423, align 4, !dbg !18
  %address.local425 = or disjoint i64 %address.local422, 4, !dbg !18
  %address.local426 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local425, !dbg !18
  store float %543, ptr addrspace(3) %address.local426, align 4, !dbg !18
  %address.local428 = or disjoint i64 %address.local422, 8, !dbg !18
  %address.local429 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local428, !dbg !18
  store float %542, ptr addrspace(3) %address.local429, align 4, !dbg !18
  %address.local431 = or i64 %address.local264, 12, !dbg !18
  %address.local432 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local431, !dbg !18
  store float %541, ptr addrspace(3) %address.local432, align 4, !dbg !18
  %address.local434 = or disjoint i64 %address.local422, 16, !dbg !18
  %address.local435 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local434, !dbg !18
  store float %536, ptr addrspace(3) %address.local435, align 4, !dbg !18
  %address.local437 = or disjoint i64 %address.local422, 20, !dbg !18
  %address.local438 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local437, !dbg !18
  store float %535, ptr addrspace(3) %address.local438, align 4, !dbg !18
  %address.local440 = or disjoint i64 %address.local422, 24, !dbg !18
  %address.local441 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local440, !dbg !18
  store float %534, ptr addrspace(3) %address.local441, align 4, !dbg !18
  %address.local443 = or i64 %address.local264, 28, !dbg !18
  %address.local444 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local443, !dbg !18
  store float %533, ptr addrspace(3) %address.local444, align 4, !dbg !18
  %address.local446 = or disjoint i64 %address.local422, 32, !dbg !18
  %address.local447 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local446, !dbg !18
  store float %528, ptr addrspace(3) %address.local447, align 4, !dbg !18
  %address.local449 = or disjoint i64 %address.local422, 36, !dbg !18
  %address.local450 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local449, !dbg !18
  store float %527, ptr addrspace(3) %address.local450, align 4, !dbg !18
  %address.local452 = or disjoint i64 %address.local422, 40, !dbg !18
  %address.local453 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local452, !dbg !18
  store float %526, ptr addrspace(3) %address.local453, align 4, !dbg !18
  %address.local455 = or i64 %address.local264, 44, !dbg !18
  %address.local456 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local455, !dbg !18
  store float %525, ptr addrspace(3) %address.local456, align 4, !dbg !18
  %address.local458 = or disjoint i64 %address.local422, 48, !dbg !18
  %address.local459 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local458, !dbg !18
  store float %520, ptr addrspace(3) %address.local459, align 4, !dbg !18
  %address.local461 = or disjoint i64 %address.local422, 52, !dbg !18
  %address.local462 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local461, !dbg !18
  store float %519, ptr addrspace(3) %address.local462, align 4, !dbg !18
  %address.local464 = or disjoint i64 %address.local422, 56, !dbg !18
  %address.local465 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local464, !dbg !18
  store float %518, ptr addrspace(3) %address.local465, align 4, !dbg !18
  %address.local467 = or i64 %address.local264, 60, !dbg !18
  %address.local468 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local467, !dbg !18
  store float %517, ptr addrspace(3) %address.local468, align 4, !dbg !18
  %address.local470 = or disjoint i64 %address.local422, 64, !dbg !18
  %address.local471 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local470, !dbg !18
  store float %512, ptr addrspace(3) %address.local471, align 4, !dbg !18
  %address.local473 = or disjoint i64 %address.local422, 68, !dbg !18
  %address.local474 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local473, !dbg !18
  store float %511, ptr addrspace(3) %address.local474, align 4, !dbg !18
  %address.local476 = or disjoint i64 %address.local422, 72, !dbg !18
  %address.local477 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local476, !dbg !18
  store float %510, ptr addrspace(3) %address.local477, align 4, !dbg !18
  %address.local479 = or i64 %address.local264, 76, !dbg !18
  %address.local480 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local479, !dbg !18
  store float %509, ptr addrspace(3) %address.local480, align 4, !dbg !18
  %address.local482 = or disjoint i64 %address.local422, 80, !dbg !18
  %address.local483 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local482, !dbg !18
  store float %504, ptr addrspace(3) %address.local483, align 4, !dbg !18
  %address.local485 = or disjoint i64 %address.local422, 84, !dbg !18
  %address.local486 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local485, !dbg !18
  store float %503, ptr addrspace(3) %address.local486, align 4, !dbg !18
  %address.local488 = or disjoint i64 %address.local422, 88, !dbg !18
  %address.local489 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local488, !dbg !18
  store float %502, ptr addrspace(3) %address.local489, align 4, !dbg !18
  %address.local491 = or i64 %address.local264, 92, !dbg !18
  %address.local492 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local491, !dbg !18
  store float %501, ptr addrspace(3) %address.local492, align 4, !dbg !18
  %address.local494 = or disjoint i64 %address.local422, 96, !dbg !18
  %address.local495 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local494, !dbg !18
  store float %496, ptr addrspace(3) %address.local495, align 4, !dbg !18
  %address.local497 = or disjoint i64 %address.local422, 100, !dbg !18
  %address.local498 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local497, !dbg !18
  store float %495, ptr addrspace(3) %address.local498, align 4, !dbg !18
  %address.local500 = or disjoint i64 %address.local422, 104, !dbg !18
  %address.local501 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local500, !dbg !18
  store float %494, ptr addrspace(3) %address.local501, align 4, !dbg !18
  %address.local503 = or i64 %address.local264, 108, !dbg !18
  %address.local504 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local503, !dbg !18
  store float %493, ptr addrspace(3) %address.local504, align 4, !dbg !18
  %address.local506 = or disjoint i64 %address.local422, 112, !dbg !18
  %address.local507 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local506, !dbg !18
  store float %488, ptr addrspace(3) %address.local507, align 4, !dbg !18
  %address.local509 = or disjoint i64 %address.local422, 116, !dbg !18
  %address.local510 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local509, !dbg !18
  store float %487, ptr addrspace(3) %address.local510, align 4, !dbg !18
  %address.local512 = or disjoint i64 %address.local422, 120, !dbg !18
  %address.local513 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local512, !dbg !18
  store float %486, ptr addrspace(3) %address.local513, align 4, !dbg !18
  %address.local515 = or i64 %address.local264, 124, !dbg !18
  %address.local516 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local515, !dbg !18
  store float %485, ptr addrspace(3) %address.local516, align 4, !dbg !18
  %address.local524 = getelementptr inbounds nuw i8, ptr addrspace(3) %address.local419, i64 32768, !dbg !18
  %address.local525 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local422, !dbg !18
  store float %480, ptr addrspace(3) %address.local525, align 4, !dbg !18
  %address.local527 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local425, !dbg !18
  store float %479, ptr addrspace(3) %address.local527, align 4, !dbg !18
  %address.local529 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local428, !dbg !18
  store float %478, ptr addrspace(3) %address.local529, align 4, !dbg !18
  %address.local531 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local431, !dbg !18
  store float %477, ptr addrspace(3) %address.local531, align 4, !dbg !18
  %address.local533 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local434, !dbg !18
  store float %472, ptr addrspace(3) %address.local533, align 4, !dbg !18
  %address.local535 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local437, !dbg !18
  store float %471, ptr addrspace(3) %address.local535, align 4, !dbg !18
  %address.local537 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local440, !dbg !18
  store float %470, ptr addrspace(3) %address.local537, align 4, !dbg !18
  %address.local539 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local443, !dbg !18
  store float %469, ptr addrspace(3) %address.local539, align 4, !dbg !18
  %address.local541 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local446, !dbg !18
  store float %464, ptr addrspace(3) %address.local541, align 4, !dbg !18
  %address.local543 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local449, !dbg !18
  store float %463, ptr addrspace(3) %address.local543, align 4, !dbg !18
  %address.local545 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local452, !dbg !18
  store float %462, ptr addrspace(3) %address.local545, align 4, !dbg !18
  %address.local547 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local455, !dbg !18
  store float %461, ptr addrspace(3) %address.local547, align 4, !dbg !18
  %address.local549 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local458, !dbg !18
  store float %456, ptr addrspace(3) %address.local549, align 4, !dbg !18
  %address.local551 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local461, !dbg !18
  store float %455, ptr addrspace(3) %address.local551, align 4, !dbg !18
  %address.local553 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local464, !dbg !18
  store float %454, ptr addrspace(3) %address.local553, align 4, !dbg !18
  %address.local555 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local467, !dbg !18
  store float %453, ptr addrspace(3) %address.local555, align 4, !dbg !18
  %address.local557 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local470, !dbg !18
  store float %448, ptr addrspace(3) %address.local557, align 4, !dbg !18
  %address.local559 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local473, !dbg !18
  store float %447, ptr addrspace(3) %address.local559, align 4, !dbg !18
  %address.local561 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local476, !dbg !18
  store float %446, ptr addrspace(3) %address.local561, align 4, !dbg !18
  %address.local563 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local479, !dbg !18
  store float %445, ptr addrspace(3) %address.local563, align 4, !dbg !18
  %address.local565 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local482, !dbg !18
  store float %440, ptr addrspace(3) %address.local565, align 4, !dbg !18
  %address.local567 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local485, !dbg !18
  store float %439, ptr addrspace(3) %address.local567, align 4, !dbg !18
  %address.local569 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local488, !dbg !18
  store float %438, ptr addrspace(3) %address.local569, align 4, !dbg !18
  %address.local571 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local491, !dbg !18
  store float %437, ptr addrspace(3) %address.local571, align 4, !dbg !18
  %address.local573 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local494, !dbg !18
  store float %432, ptr addrspace(3) %address.local573, align 4, !dbg !18
  %address.local575 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local497, !dbg !18
  store float %431, ptr addrspace(3) %address.local575, align 4, !dbg !18
  %address.local577 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local500, !dbg !18
  store float %430, ptr addrspace(3) %address.local577, align 4, !dbg !18
  %address.local579 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local503, !dbg !18
  store float %429, ptr addrspace(3) %address.local579, align 4, !dbg !18
  %address.local581 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local506, !dbg !18
  store float %424, ptr addrspace(3) %address.local581, align 4, !dbg !18
  %address.local583 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local509, !dbg !18
  store float %423, ptr addrspace(3) %address.local583, align 4, !dbg !18
  %address.local585 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local512, !dbg !18
  store float %422, ptr addrspace(3) %address.local585, align 4, !dbg !18
  %address.local587 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local515, !dbg !18
  store float %421, ptr addrspace(3) %address.local587, align 4, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.local600.idx = shl nuw nsw i64 %address.local422, 4, !dbg !18
  %address.local600 = getelementptr inbounds nuw i8, ptr addrspace(3) %address.local419, i64 %address.local600.idx, !dbg !18
  %545 = load <4 x float>, ptr addrspace(3) %address.local600, align 16, !dbg !18
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local600, i64 64, !dbg !18
  %546 = load <4 x float>, ptr addrspace(3) %lds.const.gep1671, align 16, !dbg !18
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local600, i64 128, !dbg !18
  %547 = load <4 x float>, ptr addrspace(3) %lds.const.gep1675, align 16, !dbg !18
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local600, i64 192, !dbg !18
  %548 = load <4 x float>, ptr addrspace(3) %lds.const.gep1679, align 16, !dbg !18
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local600, i64 256, !dbg !18
  %549 = load <4 x float>, ptr addrspace(3) %lds.const.gep1683, align 16, !dbg !18
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local600, i64 320, !dbg !18
  %550 = load <4 x float>, ptr addrspace(3) %lds.const.gep1687, align 16, !dbg !18
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local600, i64 384, !dbg !18
  %551 = load <4 x float>, ptr addrspace(3) %lds.const.gep1691, align 16, !dbg !18
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local600, i64 448, !dbg !18
  %552 = load <4 x float>, ptr addrspace(3) %lds.const.gep1695, align 16, !dbg !18
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32768, !dbg !18
  %553 = load <4 x float>, ptr addrspace(3) %lds.const.gep1699, align 16, !dbg !18
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32832, !dbg !18
  %554 = load <4 x float>, ptr addrspace(3) %lds.const.gep1703, align 16, !dbg !18
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32896, !dbg !18
  %555 = load <4 x float>, ptr addrspace(3) %lds.const.gep1707, align 16, !dbg !18
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32960, !dbg !18
  %556 = load <4 x float>, ptr addrspace(3) %lds.const.gep1711, align 16, !dbg !18
  %lds.const.gep1715 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33024, !dbg !18
  %557 = load <4 x float>, ptr addrspace(3) %lds.const.gep1715, align 16, !dbg !18
  %lds.const.gep1719 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33088, !dbg !18
  %558 = load <4 x float>, ptr addrspace(3) %lds.const.gep1719, align 16, !dbg !18
  %lds.const.gep1723 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33152, !dbg !18
  %559 = load <4 x float>, ptr addrspace(3) %lds.const.gep1723, align 16, !dbg !18
  %lds.const.gep1727 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33216, !dbg !18
  %560 = load <4 x float>, ptr addrspace(3) %lds.const.gep1727, align 16, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %561 = or disjoint i64 %79, 127, !dbg !12
  %562 = icmp uge i64 %19, %561, !dbg !12
  %563 = and i1 %38, %562, !dbg !12
  br i1 %563, label %648, label %564, !dbg !12

564:                                              ; preds = %71
  %565 = or disjoint i64 %79, %35, !dbg !12
  %566 = lshr i64 %78, 25, !dbg !19
  %compare.high1289 = trunc nuw i64 %566 to i32, !dbg !19
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1289, !dbg !19
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1289, !dbg !19
  %compare.low.base1290 = trunc i64 %565 to i32, !dbg !19
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 1) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1293 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 2) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1297 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 3) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1301 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 4) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1305 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 5) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1309 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 6) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1313 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 7) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1317 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 8) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1321 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 9) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1325 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 10) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1329 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 11) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1333 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 12) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1337 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 13) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1341 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 14) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1345 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 15) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1349 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 16) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1353 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 17) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1357 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 18) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1361 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 19) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1365 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 20) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1369 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 21) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1373 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 22) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1377 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 23) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1381 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 24) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1385 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 25) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1389 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 26) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1393 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 27) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1397 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 28) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1401 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 29) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1405 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 30) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1409 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 31) #8, !dbg !19, !frisk.register_lifetimes !5
  %567 = insertelement <32 x i32> poison, i32 %compare.low.base1290, i64 0, !dbg !19
  %568 = insertelement <32 x i32> %567, i32 %compare.local.base, i64 1, !dbg !19
  %569 = insertelement <32 x i32> %568, i32 %compare.local.base1293, i64 2, !dbg !19
  %570 = insertelement <32 x i32> %569, i32 %compare.local.base1297, i64 3, !dbg !19
  %571 = insertelement <32 x i32> %570, i32 %compare.local.base1301, i64 4, !dbg !19
  %572 = insertelement <32 x i32> %571, i32 %compare.local.base1305, i64 5, !dbg !19
  %573 = insertelement <32 x i32> %572, i32 %compare.local.base1309, i64 6, !dbg !19
  %574 = insertelement <32 x i32> %573, i32 %compare.local.base1313, i64 7, !dbg !19
  %575 = insertelement <32 x i32> %574, i32 %compare.local.base1317, i64 8, !dbg !19
  %576 = insertelement <32 x i32> %575, i32 %compare.local.base1321, i64 9, !dbg !19
  %577 = insertelement <32 x i32> %576, i32 %compare.local.base1325, i64 10, !dbg !19
  %578 = insertelement <32 x i32> %577, i32 %compare.local.base1329, i64 11, !dbg !19
  %579 = insertelement <32 x i32> %578, i32 %compare.local.base1333, i64 12, !dbg !19
  %580 = insertelement <32 x i32> %579, i32 %compare.local.base1337, i64 13, !dbg !19
  %581 = insertelement <32 x i32> %580, i32 %compare.local.base1341, i64 14, !dbg !19
  %582 = insertelement <32 x i32> %581, i32 %compare.local.base1345, i64 15, !dbg !19
  %583 = insertelement <32 x i32> %582, i32 %compare.local.base1349, i64 16, !dbg !19
  %584 = insertelement <32 x i32> %583, i32 %compare.local.base1353, i64 17, !dbg !19
  %585 = insertelement <32 x i32> %584, i32 %compare.local.base1357, i64 18, !dbg !19
  %586 = insertelement <32 x i32> %585, i32 %compare.local.base1361, i64 19, !dbg !19
  %587 = insertelement <32 x i32> %586, i32 %compare.local.base1365, i64 20, !dbg !19
  %588 = insertelement <32 x i32> %587, i32 %compare.local.base1369, i64 21, !dbg !19
  %589 = insertelement <32 x i32> %588, i32 %compare.local.base1373, i64 22, !dbg !19
  %590 = insertelement <32 x i32> %589, i32 %compare.local.base1377, i64 23, !dbg !19
  %591 = insertelement <32 x i32> %590, i32 %compare.local.base1381, i64 24, !dbg !19
  %592 = insertelement <32 x i32> %591, i32 %compare.local.base1385, i64 25, !dbg !19
  %593 = insertelement <32 x i32> %592, i32 %compare.local.base1389, i64 26, !dbg !19
  %594 = insertelement <32 x i32> %593, i32 %compare.local.base1393, i64 27, !dbg !19
  %595 = insertelement <32 x i32> %594, i32 %compare.local.base1397, i64 28, !dbg !19
  %596 = insertelement <32 x i32> %595, i32 %compare.local.base1401, i64 29, !dbg !19
  %597 = insertelement <32 x i32> %596, i32 %compare.local.base1405, i64 30, !dbg !19
  %598 = insertelement <32 x i32> %597, i32 %compare.local.base1409, i64 31, !dbg !19
  %599 = or <32 x i32> %598, <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 48, i32 49, i32 50, i32 51, i32 64, i32 65, i32 66, i32 67, i32 80, i32 81, i32 82, i32 83, i32 96, i32 97, i32 98, i32 99, i32 112, i32 113, i32 114, i32 115>, !dbg !19
  %600 = icmp ugt <32 x i32> %70, %599, !dbg !19
  %601 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !19
  %602 = shufflevector <32 x i1> %601, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %603 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !19
  %604 = shufflevector <32 x i1> %603, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %605 = select <32 x i1> %602, <32 x i1> %600, <32 x i1> %604, !dbg !19
  %606 = select <32 x i1> %605, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !22
  %compare.high.equal1415 = icmp eq i32 %compare.high1413, %compare.high1289, !dbg !19
  %compare.high.order1416 = icmp ugt i32 %compare.high1413, %compare.high1289, !dbg !19
  %compare.local.base1421 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 32) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1425 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 33) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1429 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 34) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1433 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 35) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1437 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 36) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1441 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 37) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1445 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 38) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1449 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 39) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1453 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 40) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1457 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 41) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1461 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 42) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1465 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 43) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1469 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 44) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1473 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 45) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1477 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 46) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1481 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 47) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1485 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 48) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1489 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 49) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1493 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 50) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1497 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 51) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1501 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 52) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1505 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 53) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1509 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 54) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1513 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 55) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1517 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 56) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1521 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 57) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1525 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 58) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1529 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 59) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1533 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 60) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1537 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 61) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1541 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 62) #8, !dbg !19, !frisk.register_lifetimes !5
  %607 = insertelement <32 x i32> %567, i32 %compare.local.base1421, i64 1, !dbg !19
  %608 = insertelement <32 x i32> %607, i32 %compare.local.base1425, i64 2, !dbg !19
  %609 = insertelement <32 x i32> %608, i32 %compare.local.base1429, i64 3, !dbg !19
  %610 = insertelement <32 x i32> %609, i32 %compare.local.base1433, i64 4, !dbg !19
  %611 = insertelement <32 x i32> %610, i32 %compare.local.base1437, i64 5, !dbg !19
  %612 = insertelement <32 x i32> %611, i32 %compare.local.base1441, i64 6, !dbg !19
  %613 = insertelement <32 x i32> %612, i32 %compare.local.base1445, i64 7, !dbg !19
  %614 = insertelement <32 x i32> %613, i32 %compare.local.base1449, i64 8, !dbg !19
  %615 = insertelement <32 x i32> %614, i32 %compare.local.base1453, i64 9, !dbg !19
  %616 = insertelement <32 x i32> %615, i32 %compare.local.base1457, i64 10, !dbg !19
  %617 = insertelement <32 x i32> %616, i32 %compare.local.base1461, i64 11, !dbg !19
  %618 = insertelement <32 x i32> %617, i32 %compare.local.base1465, i64 12, !dbg !19
  %619 = insertelement <32 x i32> %618, i32 %compare.local.base1469, i64 13, !dbg !19
  %620 = insertelement <32 x i32> %619, i32 %compare.local.base1473, i64 14, !dbg !19
  %621 = insertelement <32 x i32> %620, i32 %compare.local.base1477, i64 15, !dbg !19
  %622 = insertelement <32 x i32> %621, i32 %compare.local.base1481, i64 16, !dbg !19
  %623 = insertelement <32 x i32> %622, i32 %compare.local.base1485, i64 17, !dbg !19
  %624 = insertelement <32 x i32> %623, i32 %compare.local.base1489, i64 18, !dbg !19
  %625 = insertelement <32 x i32> %624, i32 %compare.local.base1493, i64 19, !dbg !19
  %626 = insertelement <32 x i32> %625, i32 %compare.local.base1497, i64 20, !dbg !19
  %627 = insertelement <32 x i32> %626, i32 %compare.local.base1501, i64 21, !dbg !19
  %628 = insertelement <32 x i32> %627, i32 %compare.local.base1505, i64 22, !dbg !19
  %629 = insertelement <32 x i32> %628, i32 %compare.local.base1509, i64 23, !dbg !19
  %630 = insertelement <32 x i32> %629, i32 %compare.local.base1513, i64 24, !dbg !19
  %631 = insertelement <32 x i32> %630, i32 %compare.local.base1517, i64 25, !dbg !19
  %632 = insertelement <32 x i32> %631, i32 %compare.local.base1521, i64 26, !dbg !19
  %633 = insertelement <32 x i32> %632, i32 %compare.local.base1525, i64 27, !dbg !19
  %634 = insertelement <32 x i32> %633, i32 %compare.local.base1529, i64 28, !dbg !19
  %635 = insertelement <32 x i32> %634, i32 %compare.local.base1533, i64 29, !dbg !19
  %636 = insertelement <32 x i32> %635, i32 %compare.local.base1537, i64 30, !dbg !19
  %637 = insertelement <32 x i32> %636, i32 %compare.local.base1541, i64 31, !dbg !19
  %638 = or <32 x i32> %637, <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 48, i32 49, i32 50, i32 51, i32 64, i32 65, i32 66, i32 67, i32 80, i32 81, i32 82, i32 83, i32 96, i32 97, i32 98, i32 99, i32 112, i32 113, i32 114, i32 115>, !dbg !19
  %639 = icmp ugt <32 x i32> %68, %638, !dbg !19
  %640 = insertelement <32 x i1> poison, i1 %compare.high.equal1415, i64 0, !dbg !19
  %641 = shufflevector <32 x i1> %640, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %642 = insertelement <32 x i1> poison, i1 %compare.high.order1416, i64 0, !dbg !19
  %643 = shufflevector <32 x i1> %642, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %644 = select <32 x i1> %641, <32 x i1> %639, <32 x i1> %643, !dbg !19
  %645 = select <32 x i1> %644, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !22
  %646 = insertvalue [2 x <32 x float>] poison, <32 x float> %606, 0, !dbg !12
  %647 = insertvalue [2 x <32 x float>] %646, <32 x float> %645, 1, !dbg !12
  br label %648, !dbg !12

648:                                              ; preds = %564, %71
  %649 = phi [2 x <32 x float>] [ %647, %564 ], [ zeroinitializer, %71 ], !dbg !12
  %650 = extractvalue [2 x <32 x float>] %649, 0, !dbg !23
  %651 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %652 = fadd <4 x float> %545, %651, !dbg !23
  %653 = extractelement <4 x float> %652, i64 0, !dbg !23
  %654 = extractelement <4 x float> %652, i64 1, !dbg !23
  %655 = extractelement <4 x float> %652, i64 2, !dbg !23
  %656 = extractelement <4 x float> %652, i64 3, !dbg !23
  %657 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %658 = fadd <4 x float> %546, %657, !dbg !23
  %659 = extractelement <4 x float> %658, i64 0, !dbg !23
  %660 = extractelement <4 x float> %658, i64 1, !dbg !23
  %661 = extractelement <4 x float> %658, i64 2, !dbg !23
  %662 = extractelement <4 x float> %658, i64 3, !dbg !23
  %663 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !23
  %664 = fadd <4 x float> %547, %663, !dbg !23
  %665 = extractelement <4 x float> %664, i64 0, !dbg !23
  %666 = extractelement <4 x float> %664, i64 1, !dbg !23
  %667 = extractelement <4 x float> %664, i64 2, !dbg !23
  %668 = extractelement <4 x float> %664, i64 3, !dbg !23
  %669 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !23
  %670 = fadd <4 x float> %548, %669, !dbg !23
  %671 = extractelement <4 x float> %670, i64 0, !dbg !23
  %672 = extractelement <4 x float> %670, i64 1, !dbg !23
  %673 = extractelement <4 x float> %670, i64 2, !dbg !23
  %674 = extractelement <4 x float> %670, i64 3, !dbg !23
  %675 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !23
  %676 = fadd <4 x float> %549, %675, !dbg !23
  %677 = extractelement <4 x float> %676, i64 0, !dbg !23
  %678 = extractelement <4 x float> %676, i64 1, !dbg !23
  %679 = extractelement <4 x float> %676, i64 2, !dbg !23
  %680 = extractelement <4 x float> %676, i64 3, !dbg !23
  %681 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !23
  %682 = fadd <4 x float> %550, %681, !dbg !23
  %683 = extractelement <4 x float> %682, i64 0, !dbg !23
  %684 = extractelement <4 x float> %682, i64 1, !dbg !23
  %685 = extractelement <4 x float> %682, i64 2, !dbg !23
  %686 = extractelement <4 x float> %682, i64 3, !dbg !23
  %687 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !23
  %688 = fadd <4 x float> %551, %687, !dbg !23
  %689 = extractelement <4 x float> %688, i64 0, !dbg !23
  %690 = extractelement <4 x float> %688, i64 1, !dbg !23
  %691 = extractelement <4 x float> %688, i64 2, !dbg !23
  %692 = extractelement <4 x float> %688, i64 3, !dbg !23
  %693 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %694 = fadd <4 x float> %552, %693, !dbg !23
  %695 = extractelement <4 x float> %694, i64 0, !dbg !23
  %696 = extractelement <4 x float> %694, i64 1, !dbg !23
  %697 = extractelement <4 x float> %694, i64 2, !dbg !23
  %698 = extractelement <4 x float> %694, i64 3, !dbg !23
  %699 = extractvalue [2 x <32 x float>] %649, 1, !dbg !23
  %700 = shufflevector <32 x float> %699, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %701 = fadd <4 x float> %553, %700, !dbg !23
  %702 = extractelement <4 x float> %701, i64 0, !dbg !23
  %703 = extractelement <4 x float> %701, i64 1, !dbg !23
  %704 = extractelement <4 x float> %701, i64 2, !dbg !23
  %705 = extractelement <4 x float> %701, i64 3, !dbg !23
  %706 = shufflevector <32 x float> %699, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %707 = fadd <4 x float> %554, %706, !dbg !23
  %708 = extractelement <4 x float> %707, i64 0, !dbg !23
  %709 = extractelement <4 x float> %707, i64 1, !dbg !23
  %710 = extractelement <4 x float> %707, i64 2, !dbg !23
  %711 = extractelement <4 x float> %707, i64 3, !dbg !23
  %712 = shufflevector <32 x float> %699, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !23
  %713 = fadd <4 x float> %555, %712, !dbg !23
  %714 = extractelement <4 x float> %713, i64 0, !dbg !23
  %715 = extractelement <4 x float> %713, i64 1, !dbg !23
  %716 = extractelement <4 x float> %713, i64 2, !dbg !23
  %717 = extractelement <4 x float> %713, i64 3, !dbg !23
  %718 = shufflevector <32 x float> %699, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !23
  %719 = fadd <4 x float> %556, %718, !dbg !23
  %720 = extractelement <4 x float> %719, i64 0, !dbg !23
  %721 = extractelement <4 x float> %719, i64 1, !dbg !23
  %722 = extractelement <4 x float> %719, i64 2, !dbg !23
  %723 = extractelement <4 x float> %719, i64 3, !dbg !23
  %724 = shufflevector <32 x float> %699, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !23
  %725 = fadd <4 x float> %557, %724, !dbg !23
  %726 = extractelement <4 x float> %725, i64 0, !dbg !23
  %727 = extractelement <4 x float> %725, i64 1, !dbg !23
  %728 = extractelement <4 x float> %725, i64 2, !dbg !23
  %729 = extractelement <4 x float> %725, i64 3, !dbg !23
  %730 = shufflevector <32 x float> %699, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !23
  %731 = fadd <4 x float> %558, %730, !dbg !23
  %732 = extractelement <4 x float> %731, i64 0, !dbg !23
  %733 = extractelement <4 x float> %731, i64 1, !dbg !23
  %734 = extractelement <4 x float> %731, i64 2, !dbg !23
  %735 = extractelement <4 x float> %731, i64 3, !dbg !23
  %736 = shufflevector <32 x float> %699, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !23
  %737 = fadd <4 x float> %559, %736, !dbg !23
  %738 = extractelement <4 x float> %737, i64 0, !dbg !23
  %739 = extractelement <4 x float> %737, i64 1, !dbg !23
  %740 = extractelement <4 x float> %737, i64 2, !dbg !23
  %741 = extractelement <4 x float> %737, i64 3, !dbg !23
  %742 = shufflevector <32 x float> %699, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %743 = fadd <4 x float> %560, %742, !dbg !23
  %744 = extractelement <4 x float> %743, i64 0, !dbg !23
  %745 = extractelement <4 x float> %743, i64 1, !dbg !23
  %746 = extractelement <4 x float> %743, i64 2, !dbg !23
  %747 = extractelement <4 x float> %743, i64 3, !dbg !23
  %748 = tail call float @llvm.maxnum.f32(float %653, float -inf), !dbg !24, !frisk.local_reduction !5
  %local.reduce182 = tail call float @llvm.maxnum.f32(float %748, float %659), !dbg !24, !frisk.local_reduction !5
  %local.reduce183 = tail call float @llvm.maxnum.f32(float %654, float %660), !dbg !24, !frisk.local_reduction !5
  %local.reduce184 = tail call float @llvm.maxnum.f32(float %655, float %661), !dbg !24, !frisk.local_reduction !5
  %local.reduce185 = tail call float @llvm.maxnum.f32(float %656, float %662), !dbg !24, !frisk.local_reduction !5
  %local.reduce186 = tail call float @llvm.maxnum.f32(float %local.reduce182, float %665), !dbg !24, !frisk.local_reduction !5
  %local.reduce187 = tail call float @llvm.maxnum.f32(float %local.reduce183, float %666), !dbg !24, !frisk.local_reduction !5
  %local.reduce188 = tail call float @llvm.maxnum.f32(float %local.reduce184, float %667), !dbg !24, !frisk.local_reduction !5
  %local.reduce189 = tail call float @llvm.maxnum.f32(float %local.reduce185, float %668), !dbg !24, !frisk.local_reduction !5
  %local.reduce190 = tail call float @llvm.maxnum.f32(float %local.reduce186, float %671), !dbg !24, !frisk.local_reduction !5
  %local.reduce191 = tail call float @llvm.maxnum.f32(float %local.reduce187, float %672), !dbg !24, !frisk.local_reduction !5
  %local.reduce192 = tail call float @llvm.maxnum.f32(float %local.reduce188, float %673), !dbg !24, !frisk.local_reduction !5
  %local.reduce193 = tail call float @llvm.maxnum.f32(float %local.reduce189, float %674), !dbg !24, !frisk.local_reduction !5
  %local.reduce194 = tail call float @llvm.maxnum.f32(float %local.reduce190, float %677), !dbg !24, !frisk.local_reduction !5
  %local.reduce195 = tail call float @llvm.maxnum.f32(float %local.reduce191, float %678), !dbg !24, !frisk.local_reduction !5
  %local.reduce196 = tail call float @llvm.maxnum.f32(float %local.reduce192, float %679), !dbg !24, !frisk.local_reduction !5
  %local.reduce197 = tail call float @llvm.maxnum.f32(float %local.reduce193, float %680), !dbg !24, !frisk.local_reduction !5
  %local.reduce198 = tail call float @llvm.maxnum.f32(float %local.reduce194, float %683), !dbg !24, !frisk.local_reduction !5
  %local.reduce199 = tail call float @llvm.maxnum.f32(float %local.reduce195, float %684), !dbg !24, !frisk.local_reduction !5
  %local.reduce200 = tail call float @llvm.maxnum.f32(float %local.reduce196, float %685), !dbg !24, !frisk.local_reduction !5
  %local.reduce201 = tail call float @llvm.maxnum.f32(float %local.reduce197, float %686), !dbg !24, !frisk.local_reduction !5
  %local.reduce202 = tail call float @llvm.maxnum.f32(float %local.reduce198, float %689), !dbg !24, !frisk.local_reduction !5
  %local.reduce203 = tail call float @llvm.maxnum.f32(float %local.reduce199, float %690), !dbg !24, !frisk.local_reduction !5
  %local.reduce204 = tail call float @llvm.maxnum.f32(float %local.reduce200, float %691), !dbg !24, !frisk.local_reduction !5
  %local.reduce205 = tail call float @llvm.maxnum.f32(float %local.reduce201, float %692), !dbg !24, !frisk.local_reduction !5
  %local.reduce206 = tail call float @llvm.maxnum.f32(float %local.reduce202, float %695), !dbg !24, !frisk.local_reduction !5
  %local.reduce207 = tail call float @llvm.maxnum.f32(float %local.reduce203, float %696), !dbg !24, !frisk.local_reduction !5
  %local.reduce208 = tail call float @llvm.maxnum.f32(float %local.reduce204, float %697), !dbg !24, !frisk.local_reduction !5
  %local.reduce209 = tail call float @llvm.maxnum.f32(float %local.reduce205, float %698), !dbg !24, !frisk.local_reduction !5
  %local.reduce210 = tail call float @llvm.maxnum.f32(float %local.reduce206, float %local.reduce207), !dbg !24, !frisk.local_reduction !5
  %local.reduce211 = tail call float @llvm.maxnum.f32(float %local.reduce208, float %local.reduce209), !dbg !24, !frisk.local_reduction !5
  %749 = tail call float @llvm.maxnum.f32(float %local.reduce210, float %local.reduce211), !dbg !24, !frisk.local_reduction !5
  %750 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !24
  %751 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %750), !dbg !24
  %752 = add nuw nsw i32 %751, 64, !dbg !24
  %753 = xor i32 %751, 16, !dbg !24
  %754 = and i32 %752, 192, !dbg !24
  %755 = icmp samesign ult i32 %753, %754, !dbg !24
  %756 = select i1 %755, i32 %753, i32 %751, !dbg !24
  %757 = shl nuw nsw i32 %756, 2, !dbg !24
  %758 = bitcast float %749 to i32, !dbg !24
  %759 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %758), !dbg !24
  %760 = bitcast i32 %759 to float, !dbg !24
  %761 = tail call float @llvm.maxnum.f32(float %749, float %760), !dbg !24
  %762 = xor i32 %751, 32, !dbg !24
  %763 = icmp samesign ult i32 %762, %754, !dbg !24
  %764 = select i1 %763, i32 %762, i32 %751, !dbg !24
  %765 = shl nuw nsw i32 %764, 2, !dbg !24
  %766 = bitcast float %761 to i32, !dbg !24
  %767 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %766), !dbg !24
  %768 = bitcast i32 %767 to float, !dbg !24
  %769 = tail call float @llvm.maxnum.f32(float %761, float %768), !dbg !24
  %770 = extractvalue [2 x <1 x float>] %75, 0, !dbg !24
  %771 = insertelement <1 x float> poison, float %769, i64 0, !dbg !24
  %772 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %771, <1 x float> %770), !dbg !24
  %773 = tail call float @llvm.maxnum.f32(float %702, float -inf), !dbg !24, !frisk.local_reduction !5
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %773, float %708), !dbg !24, !frisk.local_reduction !5
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %703, float %709), !dbg !24, !frisk.local_reduction !5
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %704, float %710), !dbg !24, !frisk.local_reduction !5
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %705, float %711), !dbg !24, !frisk.local_reduction !5
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %714), !dbg !24, !frisk.local_reduction !5
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %715), !dbg !24, !frisk.local_reduction !5
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %716), !dbg !24, !frisk.local_reduction !5
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %717), !dbg !24, !frisk.local_reduction !5
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %720), !dbg !24, !frisk.local_reduction !5
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %721), !dbg !24, !frisk.local_reduction !5
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %722), !dbg !24, !frisk.local_reduction !5
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %723), !dbg !24, !frisk.local_reduction !5
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %726), !dbg !24, !frisk.local_reduction !5
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %727), !dbg !24, !frisk.local_reduction !5
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %728), !dbg !24, !frisk.local_reduction !5
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %729), !dbg !24, !frisk.local_reduction !5
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %732), !dbg !24, !frisk.local_reduction !5
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %733), !dbg !24, !frisk.local_reduction !5
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %734), !dbg !24, !frisk.local_reduction !5
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %735), !dbg !24, !frisk.local_reduction !5
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %738), !dbg !24, !frisk.local_reduction !5
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %739), !dbg !24, !frisk.local_reduction !5
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %740), !dbg !24, !frisk.local_reduction !5
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %741), !dbg !24, !frisk.local_reduction !5
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %744), !dbg !24, !frisk.local_reduction !5
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %745), !dbg !24, !frisk.local_reduction !5
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %746), !dbg !24, !frisk.local_reduction !5
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %747), !dbg !24, !frisk.local_reduction !5
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !24, !frisk.local_reduction !5
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !24, !frisk.local_reduction !5
  %774 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !24, !frisk.local_reduction !5
  %775 = bitcast float %774 to i32, !dbg !24
  %776 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %775), !dbg !24
  %777 = bitcast i32 %776 to float, !dbg !24
  %778 = tail call float @llvm.maxnum.f32(float %774, float %777), !dbg !24
  %779 = bitcast float %778 to i32, !dbg !24
  %780 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %779), !dbg !24
  %781 = bitcast i32 %780 to float, !dbg !24
  %782 = tail call float @llvm.maxnum.f32(float %778, float %781), !dbg !24
  %783 = extractvalue [2 x <1 x float>] %75, 1, !dbg !24
  %784 = insertelement <1 x float> poison, float %782, i64 0, !dbg !24
  %785 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %784, <1 x float> %783), !dbg !24
  %786 = shufflevector <1 x float> %772, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %787 = fsub <4 x float> %652, %786, !dbg !25
  %788 = fsub <4 x float> %658, %786, !dbg !25
  %789 = fsub <4 x float> %664, %786, !dbg !25
  %790 = fsub <4 x float> %670, %786, !dbg !25
  %791 = fsub <4 x float> %676, %786, !dbg !25
  %792 = fsub <4 x float> %682, %786, !dbg !25
  %793 = fsub <4 x float> %688, %786, !dbg !25
  %794 = fsub <4 x float> %694, %786, !dbg !25
  %795 = shufflevector <1 x float> %785, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %796 = fsub <4 x float> %701, %795, !dbg !25
  %797 = fsub <4 x float> %707, %795, !dbg !25
  %798 = fsub <4 x float> %713, %795, !dbg !25
  %799 = fsub <4 x float> %719, %795, !dbg !25
  %800 = fsub <4 x float> %725, %795, !dbg !25
  %801 = fsub <4 x float> %731, %795, !dbg !25
  %802 = fsub <4 x float> %737, %795, !dbg !25
  %803 = fsub <4 x float> %743, %795, !dbg !25
  %804 = fsub <1 x float> %770, %772, !dbg !26
  %805 = extractelement <1 x float> %804, i64 0, !dbg !27
  %806 = tail call float @llvm.amdgcn.exp2.f32(float %805), !dbg !27
  %807 = fsub <1 x float> %783, %785, !dbg !26
  %808 = extractelement <1 x float> %807, i64 0, !dbg !27
  %809 = tail call float @llvm.amdgcn.exp2.f32(float %808), !dbg !27
  br i1 %39, label %810, label %813, !dbg !13

810:                                              ; preds = %648
  %address.thread765 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !13, !range !11, !frisk.register_lifetimes !5
  %address.local766 = zext nneg i32 %address.thread765 to i64, !dbg !10
  %811 = lshr i64 %address.local766, 1, !dbg !13
  %address.local768 = and i64 %811, 96, !dbg !13
  %address.local769 = and i64 %address.local766, 15, !dbg !13
  %812 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local769, !dbg !13
  %address.local773 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %812, i64 %address.local768, !dbg !13
  store float %806, ptr addrspace(3) %address.local773, align 4, !dbg !13
  %address.local775 = getelementptr i8, ptr addrspace(3) %address.local773, i64 64, !dbg !13
  store float %809, ptr addrspace(3) %address.local775, align 4, !dbg !13
  br label %813, !dbg !13

813:                                              ; preds = %810, %648
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.thread776 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !13, !range !11, !frisk.register_lifetimes !5
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %814 = add nuw nsw i64 %88, %15, !dbg !28
  %815 = shl i64 %814, 19, !dbg !28
  %816 = and i64 %815, 16252928, !dbg !28
  %address.local790 = zext nneg i32 %address.thread776 to i64, !dbg !10
  %address.local791 = shl nuw nsw i64 %address.local790, 3, !dbg !9
  %address.local792 = and i64 %address.local791, 1536, !dbg !9
  %address.local793 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local792, !dbg !9
  %address.local794 = ptrtoint ptr addrspace(3) %address.local793 to i64, !dbg !9
  %address.local795 = trunc i64 %address.local794 to i32, !dbg !9
  %817 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local795), !dbg !28
  %address.local796 = shl nuw nsw i64 %address.local790, 6, !dbg !28
  %address.local797 = and i64 %address.local796, 16128, !dbg !28
  %address.local799 = and i64 %address.local791, 24, !dbg !28
  %.idx5558 = add i64 %816, %94, !dbg !28
  %818 = or disjoint i64 %address.local799, %.idx5558, !dbg !28
  %819 = shl i64 %818, 1, !dbg !28
  %820 = or disjoint i64 %819, %address.local797, !dbg !28
  %dma.buffer.offset1758 = trunc i64 %820 to i32, !dbg !28
  %dma.buffer.lds1760 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %817), !dbg !28
  %821 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1760, i32 %dma.buffer.offset1758, <4 x i32> %dma.buffer.rsrc1759) #9, !dbg !28
  %address.local802 = or disjoint i64 %address.local790, 256, !dbg !9
  %address.local803 = shl nuw nsw i64 %address.local802, 3, !dbg !9
  %address.local804 = and i64 %address.local803, 3584, !dbg !9
  %address.local805 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local804, !dbg !9
  %address.local806 = ptrtoint ptr addrspace(3) %address.local805 to i64, !dbg !9
  %address.local807 = trunc i64 %address.local806 to i32, !dbg !9
  %822 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local807), !dbg !28
  %address.local808 = shl nuw nsw i64 %address.local802, 6, !dbg !28
  %address.local809 = and i64 %address.local808, 32512, !dbg !28
  %823 = or disjoint i64 %819, %address.local809, !dbg !28
  %dma.buffer.offset1755 = trunc i64 %823 to i32, !dbg !28
  %dma.buffer.lds1757 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %822), !dbg !28
  %824 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1757, i32 %dma.buffer.offset1755, <4 x i32> %dma.buffer.rsrc1759) #9, !dbg !28
  %825 = extractvalue [2 x <1 x float>] %76, 0, !dbg !13
  %826 = extractelement <1 x float> %825, i64 0, !dbg !13
  %827 = fmul float %826, %806, !dbg !13
  %828 = extractvalue [2 x <1 x float>] %76, 1, !dbg !13
  %829 = extractelement <1 x float> %828, i64 0, !dbg !13
  %830 = fmul float %829, %809, !dbg !13
  %831 = extractelement <4 x float> %787, i64 0, !dbg !29
  %832 = tail call float @llvm.amdgcn.exp2.f32(float %831), !dbg !29
  %833 = insertelement <4 x float> poison, float %832, i64 0, !dbg !30
  %834 = extractelement <4 x float> %787, i64 1, !dbg !29
  %835 = tail call float @llvm.amdgcn.exp2.f32(float %834), !dbg !29
  %836 = insertelement <4 x float> %833, float %835, i64 1, !dbg !30
  %837 = extractelement <4 x float> %787, i64 2, !dbg !29
  %838 = tail call float @llvm.amdgcn.exp2.f32(float %837), !dbg !29
  %839 = insertelement <4 x float> %836, float %838, i64 2, !dbg !30
  %840 = extractelement <4 x float> %787, i64 3, !dbg !29
  %841 = tail call float @llvm.amdgcn.exp2.f32(float %840), !dbg !29
  %842 = insertelement <4 x float> %839, float %841, i64 3, !dbg !30
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %842) #8, !dbg !30, !frisk.register_lifetimes !5
  %843 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !30
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %843) #8, !dbg !29, !frisk.register_lifetimes !5
  %844 = extractelement <4 x float> %788, i64 0, !dbg !29
  %845 = tail call float @llvm.amdgcn.exp2.f32(float %844), !dbg !29
  %846 = insertelement <4 x float> poison, float %845, i64 0, !dbg !30
  %local.reduce120 = fadd float %832, %845, !dbg !31, !frisk.local_reduction !5
  %847 = extractelement <4 x float> %788, i64 1, !dbg !29
  %848 = tail call float @llvm.amdgcn.exp2.f32(float %847), !dbg !29
  %849 = insertelement <4 x float> %846, float %848, i64 1, !dbg !30
  %local.reduce121 = fadd float %835, %848, !dbg !31, !frisk.local_reduction !5
  %850 = extractelement <4 x float> %788, i64 2, !dbg !29
  %851 = tail call float @llvm.amdgcn.exp2.f32(float %850), !dbg !29
  %852 = insertelement <4 x float> %849, float %851, i64 2, !dbg !30
  %local.reduce122 = fadd float %838, %851, !dbg !31, !frisk.local_reduction !5
  %853 = extractelement <4 x float> %788, i64 3, !dbg !29
  %854 = tail call float @llvm.amdgcn.exp2.f32(float %853), !dbg !29
  %855 = insertelement <4 x float> %852, float %854, i64 3, !dbg !30
  %math.result227 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %855) #8, !dbg !30, !frisk.register_lifetimes !5
  %856 = fptrunc <4 x float> %math.result227 to <4 x half>, !dbg !30
  %math.packed228 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %856) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce123 = fadd float %841, %854, !dbg !31, !frisk.local_reduction !5
  %857 = extractelement <4 x float> %789, i64 0, !dbg !29
  %858 = tail call float @llvm.amdgcn.exp2.f32(float %857), !dbg !29
  %859 = insertelement <4 x float> poison, float %858, i64 0, !dbg !30
  %local.reduce124 = fadd float %858, %local.reduce120, !dbg !31, !frisk.local_reduction !5
  %860 = extractelement <4 x float> %789, i64 1, !dbg !29
  %861 = tail call float @llvm.amdgcn.exp2.f32(float %860), !dbg !29
  %862 = insertelement <4 x float> %859, float %861, i64 1, !dbg !30
  %local.reduce125 = fadd float %861, %local.reduce121, !dbg !31, !frisk.local_reduction !5
  %863 = extractelement <4 x float> %789, i64 2, !dbg !29
  %864 = tail call float @llvm.amdgcn.exp2.f32(float %863), !dbg !29
  %865 = insertelement <4 x float> %862, float %864, i64 2, !dbg !30
  %local.reduce126 = fadd float %864, %local.reduce122, !dbg !31, !frisk.local_reduction !5
  %866 = extractelement <4 x float> %789, i64 3, !dbg !29
  %867 = tail call float @llvm.amdgcn.exp2.f32(float %866), !dbg !29
  %868 = insertelement <4 x float> %865, float %867, i64 3, !dbg !30
  %math.result229 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %868) #8, !dbg !30, !frisk.register_lifetimes !5
  %869 = fptrunc <4 x float> %math.result229 to <4 x half>, !dbg !30
  %math.packed230 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %869) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce127 = fadd float %867, %local.reduce123, !dbg !31, !frisk.local_reduction !5
  %870 = extractelement <4 x float> %790, i64 0, !dbg !29
  %871 = tail call float @llvm.amdgcn.exp2.f32(float %870), !dbg !29
  %872 = insertelement <4 x float> poison, float %871, i64 0, !dbg !30
  %local.reduce128 = fadd float %871, %local.reduce124, !dbg !31, !frisk.local_reduction !5
  %873 = extractelement <4 x float> %790, i64 1, !dbg !29
  %874 = tail call float @llvm.amdgcn.exp2.f32(float %873), !dbg !29
  %875 = insertelement <4 x float> %872, float %874, i64 1, !dbg !30
  %local.reduce129 = fadd float %874, %local.reduce125, !dbg !31, !frisk.local_reduction !5
  %876 = extractelement <4 x float> %790, i64 2, !dbg !29
  %877 = tail call float @llvm.amdgcn.exp2.f32(float %876), !dbg !29
  %878 = insertelement <4 x float> %875, float %877, i64 2, !dbg !30
  %local.reduce130 = fadd float %877, %local.reduce126, !dbg !31, !frisk.local_reduction !5
  %879 = extractelement <4 x float> %790, i64 3, !dbg !29
  %880 = tail call float @llvm.amdgcn.exp2.f32(float %879), !dbg !29
  %881 = insertelement <4 x float> %878, float %880, i64 3, !dbg !30
  %math.result231 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %881) #8, !dbg !30, !frisk.register_lifetimes !5
  %882 = fptrunc <4 x float> %math.result231 to <4 x half>, !dbg !30
  %math.packed232 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %882) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce131 = fadd float %880, %local.reduce127, !dbg !31, !frisk.local_reduction !5
  %883 = extractelement <4 x float> %791, i64 0, !dbg !29
  %884 = tail call float @llvm.amdgcn.exp2.f32(float %883), !dbg !29
  %885 = insertelement <4 x float> poison, float %884, i64 0, !dbg !30
  %local.reduce132 = fadd float %884, %local.reduce128, !dbg !31, !frisk.local_reduction !5
  %886 = extractelement <4 x float> %791, i64 1, !dbg !29
  %887 = tail call float @llvm.amdgcn.exp2.f32(float %886), !dbg !29
  %888 = insertelement <4 x float> %885, float %887, i64 1, !dbg !30
  %local.reduce133 = fadd float %887, %local.reduce129, !dbg !31, !frisk.local_reduction !5
  %889 = extractelement <4 x float> %791, i64 2, !dbg !29
  %890 = tail call float @llvm.amdgcn.exp2.f32(float %889), !dbg !29
  %891 = insertelement <4 x float> %888, float %890, i64 2, !dbg !30
  %local.reduce134 = fadd float %890, %local.reduce130, !dbg !31, !frisk.local_reduction !5
  %892 = extractelement <4 x float> %791, i64 3, !dbg !29
  %893 = tail call float @llvm.amdgcn.exp2.f32(float %892), !dbg !29
  %894 = insertelement <4 x float> %891, float %893, i64 3, !dbg !30
  %math.result233 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %894) #8, !dbg !30, !frisk.register_lifetimes !5
  %895 = fptrunc <4 x float> %math.result233 to <4 x half>, !dbg !30
  %math.packed234 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %895) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce135 = fadd float %893, %local.reduce131, !dbg !31, !frisk.local_reduction !5
  %896 = extractelement <4 x float> %792, i64 0, !dbg !29
  %897 = tail call float @llvm.amdgcn.exp2.f32(float %896), !dbg !29
  %898 = insertelement <4 x float> poison, float %897, i64 0, !dbg !30
  %local.reduce136 = fadd float %897, %local.reduce132, !dbg !31, !frisk.local_reduction !5
  %899 = extractelement <4 x float> %792, i64 1, !dbg !29
  %900 = tail call float @llvm.amdgcn.exp2.f32(float %899), !dbg !29
  %901 = insertelement <4 x float> %898, float %900, i64 1, !dbg !30
  %local.reduce137 = fadd float %900, %local.reduce133, !dbg !31, !frisk.local_reduction !5
  %902 = extractelement <4 x float> %792, i64 2, !dbg !29
  %903 = tail call float @llvm.amdgcn.exp2.f32(float %902), !dbg !29
  %904 = insertelement <4 x float> %901, float %903, i64 2, !dbg !30
  %local.reduce138 = fadd float %903, %local.reduce134, !dbg !31, !frisk.local_reduction !5
  %905 = extractelement <4 x float> %792, i64 3, !dbg !29
  %906 = tail call float @llvm.amdgcn.exp2.f32(float %905), !dbg !29
  %907 = insertelement <4 x float> %904, float %906, i64 3, !dbg !30
  %math.result235 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %907) #8, !dbg !30, !frisk.register_lifetimes !5
  %908 = fptrunc <4 x float> %math.result235 to <4 x half>, !dbg !30
  %math.packed236 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %908) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce139 = fadd float %906, %local.reduce135, !dbg !31, !frisk.local_reduction !5
  %909 = extractelement <4 x float> %793, i64 0, !dbg !29
  %910 = tail call float @llvm.amdgcn.exp2.f32(float %909), !dbg !29
  %911 = insertelement <4 x float> poison, float %910, i64 0, !dbg !30
  %local.reduce140 = fadd float %910, %local.reduce136, !dbg !31, !frisk.local_reduction !5
  %912 = extractelement <4 x float> %793, i64 1, !dbg !29
  %913 = tail call float @llvm.amdgcn.exp2.f32(float %912), !dbg !29
  %914 = insertelement <4 x float> %911, float %913, i64 1, !dbg !30
  %local.reduce141 = fadd float %913, %local.reduce137, !dbg !31, !frisk.local_reduction !5
  %915 = extractelement <4 x float> %793, i64 2, !dbg !29
  %916 = tail call float @llvm.amdgcn.exp2.f32(float %915), !dbg !29
  %917 = insertelement <4 x float> %914, float %916, i64 2, !dbg !30
  %local.reduce142 = fadd float %916, %local.reduce138, !dbg !31, !frisk.local_reduction !5
  %918 = extractelement <4 x float> %793, i64 3, !dbg !29
  %919 = tail call float @llvm.amdgcn.exp2.f32(float %918), !dbg !29
  %920 = insertelement <4 x float> %917, float %919, i64 3, !dbg !30
  %math.result237 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %920) #8, !dbg !30, !frisk.register_lifetimes !5
  %921 = fptrunc <4 x float> %math.result237 to <4 x half>, !dbg !30
  %math.packed238 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %921) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce143 = fadd float %919, %local.reduce139, !dbg !31, !frisk.local_reduction !5
  %922 = extractelement <4 x float> %794, i64 0, !dbg !29
  %923 = tail call float @llvm.amdgcn.exp2.f32(float %922), !dbg !29
  %924 = insertelement <4 x float> poison, float %923, i64 0, !dbg !30
  %local.reduce144 = fadd float %923, %local.reduce140, !dbg !31, !frisk.local_reduction !5
  %925 = extractelement <4 x float> %794, i64 1, !dbg !29
  %926 = tail call float @llvm.amdgcn.exp2.f32(float %925), !dbg !29
  %927 = insertelement <4 x float> %924, float %926, i64 1, !dbg !30
  %local.reduce145 = fadd float %926, %local.reduce141, !dbg !31, !frisk.local_reduction !5
  %local.reduce148 = fadd float %local.reduce144, %local.reduce145, !dbg !31, !frisk.local_reduction !5
  %928 = extractelement <4 x float> %794, i64 2, !dbg !29
  %929 = tail call float @llvm.amdgcn.exp2.f32(float %928), !dbg !29
  %930 = insertelement <4 x float> %927, float %929, i64 2, !dbg !30
  %local.reduce146 = fadd float %929, %local.reduce142, !dbg !31, !frisk.local_reduction !5
  %931 = extractelement <4 x float> %794, i64 3, !dbg !29
  %932 = tail call float @llvm.amdgcn.exp2.f32(float %931), !dbg !29
  %933 = insertelement <4 x float> %930, float %932, i64 3, !dbg !30
  %math.result239 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %933) #8, !dbg !30, !frisk.register_lifetimes !5
  %934 = fptrunc <4 x float> %math.result239 to <4 x half>, !dbg !30
  %math.packed240 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %934) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce147 = fadd float %932, %local.reduce143, !dbg !31, !frisk.local_reduction !5
  %local.reduce149 = fadd float %local.reduce146, %local.reduce147, !dbg !31, !frisk.local_reduction !5
  %935 = fadd float %local.reduce148, %local.reduce149, !dbg !31, !frisk.local_reduction !5
  %936 = extractelement <4 x float> %796, i64 0, !dbg !29
  %937 = tail call float @llvm.amdgcn.exp2.f32(float %936), !dbg !29
  %938 = insertelement <4 x float> poison, float %937, i64 0, !dbg !30
  %939 = extractelement <4 x float> %796, i64 1, !dbg !29
  %940 = tail call float @llvm.amdgcn.exp2.f32(float %939), !dbg !29
  %941 = insertelement <4 x float> %938, float %940, i64 1, !dbg !30
  %942 = extractelement <4 x float> %796, i64 2, !dbg !29
  %943 = tail call float @llvm.amdgcn.exp2.f32(float %942), !dbg !29
  %944 = insertelement <4 x float> %941, float %943, i64 2, !dbg !30
  %945 = extractelement <4 x float> %796, i64 3, !dbg !29
  %946 = tail call float @llvm.amdgcn.exp2.f32(float %945), !dbg !29
  %947 = insertelement <4 x float> %944, float %946, i64 3, !dbg !30
  %math.result241 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %947) #8, !dbg !30, !frisk.register_lifetimes !5
  %948 = fptrunc <4 x float> %math.result241 to <4 x half>, !dbg !30
  %math.packed242 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %948) #8, !dbg !29, !frisk.register_lifetimes !5
  %949 = extractelement <4 x float> %797, i64 0, !dbg !29
  %950 = tail call float @llvm.amdgcn.exp2.f32(float %949), !dbg !29
  %951 = insertelement <4 x float> poison, float %950, i64 0, !dbg !30
  %local.reduce = fadd float %937, %950, !dbg !31, !frisk.local_reduction !5
  %952 = extractelement <4 x float> %797, i64 1, !dbg !29
  %953 = tail call float @llvm.amdgcn.exp2.f32(float %952), !dbg !29
  %954 = insertelement <4 x float> %951, float %953, i64 1, !dbg !30
  %local.reduce90 = fadd float %940, %953, !dbg !31, !frisk.local_reduction !5
  %955 = extractelement <4 x float> %797, i64 2, !dbg !29
  %956 = tail call float @llvm.amdgcn.exp2.f32(float %955), !dbg !29
  %957 = insertelement <4 x float> %954, float %956, i64 2, !dbg !30
  %local.reduce91 = fadd float %943, %956, !dbg !31, !frisk.local_reduction !5
  %958 = extractelement <4 x float> %797, i64 3, !dbg !29
  %959 = tail call float @llvm.amdgcn.exp2.f32(float %958), !dbg !29
  %960 = insertelement <4 x float> %957, float %959, i64 3, !dbg !30
  %math.result243 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %960) #8, !dbg !30, !frisk.register_lifetimes !5
  %961 = fptrunc <4 x float> %math.result243 to <4 x half>, !dbg !30
  %math.packed244 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %961) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce92 = fadd float %946, %959, !dbg !31, !frisk.local_reduction !5
  %962 = extractelement <4 x float> %798, i64 0, !dbg !29
  %963 = tail call float @llvm.amdgcn.exp2.f32(float %962), !dbg !29
  %964 = insertelement <4 x float> poison, float %963, i64 0, !dbg !30
  %local.reduce93 = fadd float %963, %local.reduce, !dbg !31, !frisk.local_reduction !5
  %965 = extractelement <4 x float> %798, i64 1, !dbg !29
  %966 = tail call float @llvm.amdgcn.exp2.f32(float %965), !dbg !29
  %967 = insertelement <4 x float> %964, float %966, i64 1, !dbg !30
  %local.reduce94 = fadd float %966, %local.reduce90, !dbg !31, !frisk.local_reduction !5
  %968 = extractelement <4 x float> %798, i64 2, !dbg !29
  %969 = tail call float @llvm.amdgcn.exp2.f32(float %968), !dbg !29
  %970 = insertelement <4 x float> %967, float %969, i64 2, !dbg !30
  %local.reduce95 = fadd float %969, %local.reduce91, !dbg !31, !frisk.local_reduction !5
  %971 = extractelement <4 x float> %798, i64 3, !dbg !29
  %972 = tail call float @llvm.amdgcn.exp2.f32(float %971), !dbg !29
  %973 = insertelement <4 x float> %970, float %972, i64 3, !dbg !30
  %math.result245 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %973) #8, !dbg !30, !frisk.register_lifetimes !5
  %974 = fptrunc <4 x float> %math.result245 to <4 x half>, !dbg !30
  %math.packed246 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %974) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce96 = fadd float %972, %local.reduce92, !dbg !31, !frisk.local_reduction !5
  %975 = extractelement <4 x float> %799, i64 0, !dbg !29
  %976 = tail call float @llvm.amdgcn.exp2.f32(float %975), !dbg !29
  %977 = insertelement <4 x float> poison, float %976, i64 0, !dbg !30
  %local.reduce97 = fadd float %976, %local.reduce93, !dbg !31, !frisk.local_reduction !5
  %978 = extractelement <4 x float> %799, i64 1, !dbg !29
  %979 = tail call float @llvm.amdgcn.exp2.f32(float %978), !dbg !29
  %980 = insertelement <4 x float> %977, float %979, i64 1, !dbg !30
  %local.reduce98 = fadd float %979, %local.reduce94, !dbg !31, !frisk.local_reduction !5
  %981 = extractelement <4 x float> %799, i64 2, !dbg !29
  %982 = tail call float @llvm.amdgcn.exp2.f32(float %981), !dbg !29
  %983 = insertelement <4 x float> %980, float %982, i64 2, !dbg !30
  %local.reduce99 = fadd float %982, %local.reduce95, !dbg !31, !frisk.local_reduction !5
  %984 = extractelement <4 x float> %799, i64 3, !dbg !29
  %985 = tail call float @llvm.amdgcn.exp2.f32(float %984), !dbg !29
  %986 = insertelement <4 x float> %983, float %985, i64 3, !dbg !30
  %math.result247 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %986) #8, !dbg !30, !frisk.register_lifetimes !5
  %987 = fptrunc <4 x float> %math.result247 to <4 x half>, !dbg !30
  %math.packed248 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %987) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce100 = fadd float %985, %local.reduce96, !dbg !31, !frisk.local_reduction !5
  %988 = extractelement <4 x float> %800, i64 0, !dbg !29
  %989 = tail call float @llvm.amdgcn.exp2.f32(float %988), !dbg !29
  %990 = insertelement <4 x float> poison, float %989, i64 0, !dbg !30
  %local.reduce101 = fadd float %989, %local.reduce97, !dbg !31, !frisk.local_reduction !5
  %991 = extractelement <4 x float> %800, i64 1, !dbg !29
  %992 = tail call float @llvm.amdgcn.exp2.f32(float %991), !dbg !29
  %993 = insertelement <4 x float> %990, float %992, i64 1, !dbg !30
  %local.reduce102 = fadd float %992, %local.reduce98, !dbg !31, !frisk.local_reduction !5
  %994 = extractelement <4 x float> %800, i64 2, !dbg !29
  %995 = tail call float @llvm.amdgcn.exp2.f32(float %994), !dbg !29
  %996 = insertelement <4 x float> %993, float %995, i64 2, !dbg !30
  %local.reduce103 = fadd float %995, %local.reduce99, !dbg !31, !frisk.local_reduction !5
  %997 = extractelement <4 x float> %800, i64 3, !dbg !29
  %998 = tail call float @llvm.amdgcn.exp2.f32(float %997), !dbg !29
  %999 = insertelement <4 x float> %996, float %998, i64 3, !dbg !30
  %math.result249 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %999) #8, !dbg !30, !frisk.register_lifetimes !5
  %1000 = fptrunc <4 x float> %math.result249 to <4 x half>, !dbg !30
  %math.packed250 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1000) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce104 = fadd float %998, %local.reduce100, !dbg !31, !frisk.local_reduction !5
  %1001 = extractelement <4 x float> %801, i64 0, !dbg !29
  %1002 = tail call float @llvm.amdgcn.exp2.f32(float %1001), !dbg !29
  %1003 = insertelement <4 x float> poison, float %1002, i64 0, !dbg !30
  %local.reduce105 = fadd float %1002, %local.reduce101, !dbg !31, !frisk.local_reduction !5
  %1004 = extractelement <4 x float> %801, i64 1, !dbg !29
  %1005 = tail call float @llvm.amdgcn.exp2.f32(float %1004), !dbg !29
  %1006 = insertelement <4 x float> %1003, float %1005, i64 1, !dbg !30
  %local.reduce106 = fadd float %1005, %local.reduce102, !dbg !31, !frisk.local_reduction !5
  %1007 = extractelement <4 x float> %801, i64 2, !dbg !29
  %1008 = tail call float @llvm.amdgcn.exp2.f32(float %1007), !dbg !29
  %1009 = insertelement <4 x float> %1006, float %1008, i64 2, !dbg !30
  %local.reduce107 = fadd float %1008, %local.reduce103, !dbg !31, !frisk.local_reduction !5
  %1010 = extractelement <4 x float> %801, i64 3, !dbg !29
  %1011 = tail call float @llvm.amdgcn.exp2.f32(float %1010), !dbg !29
  %1012 = insertelement <4 x float> %1009, float %1011, i64 3, !dbg !30
  %math.result251 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1012) #8, !dbg !30, !frisk.register_lifetimes !5
  %1013 = fptrunc <4 x float> %math.result251 to <4 x half>, !dbg !30
  %math.packed252 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1013) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce108 = fadd float %1011, %local.reduce104, !dbg !31, !frisk.local_reduction !5
  %1014 = extractelement <4 x float> %802, i64 0, !dbg !29
  %1015 = tail call float @llvm.amdgcn.exp2.f32(float %1014), !dbg !29
  %1016 = insertelement <4 x float> poison, float %1015, i64 0, !dbg !30
  %local.reduce109 = fadd float %1015, %local.reduce105, !dbg !31, !frisk.local_reduction !5
  %1017 = extractelement <4 x float> %802, i64 1, !dbg !29
  %1018 = tail call float @llvm.amdgcn.exp2.f32(float %1017), !dbg !29
  %1019 = insertelement <4 x float> %1016, float %1018, i64 1, !dbg !30
  %local.reduce110 = fadd float %1018, %local.reduce106, !dbg !31, !frisk.local_reduction !5
  %1020 = extractelement <4 x float> %802, i64 2, !dbg !29
  %1021 = tail call float @llvm.amdgcn.exp2.f32(float %1020), !dbg !29
  %1022 = insertelement <4 x float> %1019, float %1021, i64 2, !dbg !30
  %local.reduce111 = fadd float %1021, %local.reduce107, !dbg !31, !frisk.local_reduction !5
  %1023 = extractelement <4 x float> %802, i64 3, !dbg !29
  %1024 = tail call float @llvm.amdgcn.exp2.f32(float %1023), !dbg !29
  %1025 = insertelement <4 x float> %1022, float %1024, i64 3, !dbg !30
  %math.result253 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1025) #8, !dbg !30, !frisk.register_lifetimes !5
  %1026 = fptrunc <4 x float> %math.result253 to <4 x half>, !dbg !30
  %math.packed254 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1026) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce112 = fadd float %1024, %local.reduce108, !dbg !31, !frisk.local_reduction !5
  %1027 = extractelement <4 x float> %803, i64 0, !dbg !29
  %1028 = tail call float @llvm.amdgcn.exp2.f32(float %1027), !dbg !29
  %1029 = insertelement <4 x float> poison, float %1028, i64 0, !dbg !30
  %local.reduce113 = fadd float %1028, %local.reduce109, !dbg !31, !frisk.local_reduction !5
  %1030 = extractelement <4 x float> %803, i64 1, !dbg !29
  %1031 = tail call float @llvm.amdgcn.exp2.f32(float %1030), !dbg !29
  %1032 = insertelement <4 x float> %1029, float %1031, i64 1, !dbg !30
  %local.reduce114 = fadd float %1031, %local.reduce110, !dbg !31, !frisk.local_reduction !5
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !31, !frisk.local_reduction !5
  %1033 = extractelement <4 x float> %803, i64 2, !dbg !29
  %1034 = tail call float @llvm.amdgcn.exp2.f32(float %1033), !dbg !29
  %1035 = insertelement <4 x float> %1032, float %1034, i64 2, !dbg !30
  %local.reduce115 = fadd float %1034, %local.reduce111, !dbg !31, !frisk.local_reduction !5
  %1036 = extractelement <4 x float> %803, i64 3, !dbg !29
  %1037 = tail call float @llvm.amdgcn.exp2.f32(float %1036), !dbg !29
  %1038 = insertelement <4 x float> %1035, float %1037, i64 3, !dbg !30
  %math.result255 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1038) #8, !dbg !30, !frisk.register_lifetimes !5
  %1039 = fptrunc <4 x float> %math.result255 to <4 x half>, !dbg !30
  %math.packed256 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1039) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce116 = fadd float %1037, %local.reduce112, !dbg !31, !frisk.local_reduction !5
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !31, !frisk.local_reduction !5
  %1040 = fadd float %local.reduce117, %local.reduce118, !dbg !31, !frisk.local_reduction !5
  %1041 = bitcast float %935 to i32, !dbg !31
  %1042 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1041), !dbg !31
  %1043 = bitcast i32 %1042 to float, !dbg !31
  %1044 = fadd float %935, %1043, !dbg !31
  %1045 = bitcast float %1044 to i32, !dbg !31
  %1046 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1045), !dbg !31
  %1047 = bitcast i32 %1046 to float, !dbg !31
  %1048 = fadd float %1044, %1047, !dbg !31
  %1049 = fadd float %827, %1048, !dbg !31
  %1050 = bitcast float %1040 to i32, !dbg !31
  %1051 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1050), !dbg !31
  %1052 = bitcast i32 %1051 to float, !dbg !31
  %1053 = fadd float %1040, %1052, !dbg !31
  %1054 = bitcast float %1053 to i32, !dbg !31
  %1055 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1054), !dbg !31
  %1056 = bitcast i32 %1055 to float, !dbg !31
  %1057 = fadd float %1053, %1056, !dbg !31
  %1058 = fadd float %830, %1057, !dbg !31
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local814 = shl nuw nsw i32 %address.thread776, 3, !dbg !14
  %address.local815 = and i32 %address.local814, 504, !dbg !14
  %address.local816 = zext nneg i32 %address.local815 to i64, !dbg !14
  %address.local817 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local816, !dbg !14
  %1059 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local817, i32 0) #10, !dbg !14
  %1060 = bitcast <4 x i32> %1059 to <8 x half>, !dbg !14
  %address.local819 = getelementptr i8, ptr addrspace(3) %address.local817, i64 1024, !dbg !14
  %1061 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #10, !dbg !14
  %1062 = bitcast <4 x i32> %1061 to <8 x half>, !dbg !14
  %address.local821 = getelementptr i8, ptr addrspace(3) %address.local817, i64 2048, !dbg !14
  %1063 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local821, i32 0) #10, !dbg !14
  %1064 = bitcast <4 x i32> %1063 to <8 x half>, !dbg !14
  %address.local823 = getelementptr i8, ptr addrspace(3) %address.local817, i64 3072, !dbg !14
  %1065 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #10, !dbg !14
  %1066 = bitcast <4 x i32> %1065 to <8 x half>, !dbg !14
  %address.local825 = getelementptr i8, ptr addrspace(3) %address.local817, i64 4096, !dbg !14
  %1067 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local825, i32 0) #10, !dbg !14
  %1068 = bitcast <4 x i32> %1067 to <8 x half>, !dbg !14
  %address.local827 = getelementptr i8, ptr addrspace(3) %address.local817, i64 5120, !dbg !14
  %1069 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local827, i32 0) #10, !dbg !14
  %1070 = bitcast <4 x i32> %1069 to <8 x half>, !dbg !14
  %address.local829 = getelementptr i8, ptr addrspace(3) %address.local817, i64 6144, !dbg !14
  %1071 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local829, i32 0) #10, !dbg !14
  %1072 = bitcast <4 x i32> %1071 to <8 x half>, !dbg !14
  %address.local831 = getelementptr i8, ptr addrspace(3) %address.local817, i64 7168, !dbg !14
  %1073 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local831, i32 0) #10, !dbg !14
  %1074 = bitcast <4 x i32> %1073 to <8 x half>, !dbg !14
  %1075 = shufflevector <8 x half> %1060, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1075, <4 x float> zeroinitializer) #8, !dbg !14
  %1077 = shufflevector <8 x half> %1062, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1077, <4 x float> %1076) #8, !dbg !14
  %1079 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1079, <4 x float> %1078) #8, !dbg !14
  %1081 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1081, <4 x float> %1080) #8, !dbg !14
  %1083 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1083, <4 x float> %1082) #8, !dbg !14
  %1085 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1085, <4 x float> %1084) #8, !dbg !14
  %1087 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1087, <4 x float> %1086) #8, !dbg !14
  %1089 = shufflevector <8 x half> %1074, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1089, <4 x float> %1088) #8, !dbg !14
  %1091 = extractelement <4 x float> %1090, i64 0, !dbg !14
  %1092 = extractelement <4 x float> %1090, i64 1, !dbg !14
  %1093 = extractelement <4 x float> %1090, i64 2, !dbg !14
  %1094 = extractelement <4 x float> %1090, i64 3, !dbg !14
  %1095 = select i1 %.not, float %1092, float %1091, !dbg !14
  %1096 = bitcast float %1095 to i32, !dbg !14
  %1097 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1096), !dbg !14
  %1098 = bitcast i32 %1097 to float, !dbg !14
  %1099 = select i1 %.not, float %1091, float %1098, !dbg !14
  %1100 = select i1 %.not, float %1098, float %1092, !dbg !14
  %1101 = select i1 %.not, float %1094, float %1093, !dbg !14
  %1102 = bitcast float %1101 to i32, !dbg !14
  %1103 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1102), !dbg !14
  %1104 = bitcast i32 %1103 to float, !dbg !14
  %1105 = select i1 %.not, float %1093, float %1104, !dbg !14
  %1106 = select i1 %.not, float %1104, float %1094, !dbg !14
  %1107 = select i1 %.not1, float %1105, float %1099, !dbg !14
  %1108 = bitcast float %1107 to i32, !dbg !14
  %1109 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1108), !dbg !14
  %1110 = bitcast i32 %1109 to float, !dbg !14
  %1111 = select i1 %.not1, float %1106, float %1100, !dbg !14
  %1112 = bitcast float %1111 to i32, !dbg !14
  %1113 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1112), !dbg !14
  %1114 = bitcast i32 %1113 to float, !dbg !14
  %1115 = insertelement <4 x float> poison, float %1099, i64 0, !dbg !14
  %1116 = insertelement <4 x float> %1115, float %1100, i64 1, !dbg !14
  %1117 = insertelement <4 x float> %1116, float %1110, i64 2, !dbg !14
  %1118 = insertelement <4 x float> %1117, float %1114, i64 3, !dbg !14
  %1119 = insertelement <4 x float> poison, float %1110, i64 0, !dbg !14
  %1120 = insertelement <4 x float> %1119, float %1114, i64 1, !dbg !14
  %1121 = insertelement <4 x float> %1120, float %1105, i64 2, !dbg !14
  %1122 = insertelement <4 x float> %1121, float %1106, i64 3, !dbg !14
  %1123 = select <4 x i1> %66, <4 x float> %1118, <4 x float> %1122, !dbg !14
  %1124 = shufflevector <8 x half> %1060, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1125 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1124, <4 x float> zeroinitializer) #8, !dbg !14
  %1126 = shufflevector <8 x half> %1062, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1127 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1126, <4 x float> %1125) #8, !dbg !14
  %1128 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1128, <4 x float> %1127) #8, !dbg !14
  %1130 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1131 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1130, <4 x float> %1129) #8, !dbg !14
  %1132 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1132, <4 x float> %1131) #8, !dbg !14
  %1134 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1135 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1134, <4 x float> %1133) #8, !dbg !14
  %1136 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1136, <4 x float> %1135) #8, !dbg !14
  %1138 = shufflevector <8 x half> %1074, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1139 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1138, <4 x float> %1137) #8, !dbg !14
  %1140 = extractelement <4 x float> %1139, i64 0, !dbg !14
  %1141 = extractelement <4 x float> %1139, i64 1, !dbg !14
  %1142 = extractelement <4 x float> %1139, i64 2, !dbg !14
  %1143 = extractelement <4 x float> %1139, i64 3, !dbg !14
  %1144 = select i1 %.not, float %1141, float %1140, !dbg !14
  %1145 = bitcast float %1144 to i32, !dbg !14
  %1146 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1145), !dbg !14
  %1147 = bitcast i32 %1146 to float, !dbg !14
  %1148 = select i1 %.not, float %1140, float %1147, !dbg !14
  %1149 = select i1 %.not, float %1147, float %1141, !dbg !14
  %1150 = select i1 %.not, float %1143, float %1142, !dbg !14
  %1151 = bitcast float %1150 to i32, !dbg !14
  %1152 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1151), !dbg !14
  %1153 = bitcast i32 %1152 to float, !dbg !14
  %1154 = select i1 %.not, float %1142, float %1153, !dbg !14
  %1155 = select i1 %.not, float %1153, float %1143, !dbg !14
  %1156 = select i1 %.not1, float %1154, float %1148, !dbg !14
  %1157 = bitcast float %1156 to i32, !dbg !14
  %1158 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1157), !dbg !14
  %1159 = bitcast i32 %1158 to float, !dbg !14
  %1160 = select i1 %.not1, float %1155, float %1149, !dbg !14
  %1161 = bitcast float %1160 to i32, !dbg !14
  %1162 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1161), !dbg !14
  %1163 = bitcast i32 %1162 to float, !dbg !14
  %1164 = insertelement <4 x float> poison, float %1148, i64 0, !dbg !14
  %1165 = insertelement <4 x float> %1164, float %1149, i64 1, !dbg !14
  %1166 = insertelement <4 x float> %1165, float %1159, i64 2, !dbg !14
  %1167 = insertelement <4 x float> %1166, float %1163, i64 3, !dbg !14
  %1168 = insertelement <4 x float> poison, float %1159, i64 0, !dbg !14
  %1169 = insertelement <4 x float> %1168, float %1163, i64 1, !dbg !14
  %1170 = insertelement <4 x float> %1169, float %1154, i64 2, !dbg !14
  %1171 = insertelement <4 x float> %1170, float %1155, i64 3, !dbg !14
  %1172 = select <4 x i1> %66, <4 x float> %1167, <4 x float> %1171, !dbg !14
  %1173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1075, <4 x float> zeroinitializer) #8, !dbg !14
  %1174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1077, <4 x float> %1173) #8, !dbg !14
  %1175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1079, <4 x float> %1174) #8, !dbg !14
  %1176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1081, <4 x float> %1175) #8, !dbg !14
  %1177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1083, <4 x float> %1176) #8, !dbg !14
  %1178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1085, <4 x float> %1177) #8, !dbg !14
  %1179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1087, <4 x float> %1178) #8, !dbg !14
  %1180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1089, <4 x float> %1179) #8, !dbg !14
  %1181 = extractelement <4 x float> %1180, i64 0, !dbg !14
  %1182 = extractelement <4 x float> %1180, i64 1, !dbg !14
  %1183 = extractelement <4 x float> %1180, i64 2, !dbg !14
  %1184 = extractelement <4 x float> %1180, i64 3, !dbg !14
  %1185 = select i1 %.not, float %1182, float %1181, !dbg !14
  %1186 = bitcast float %1185 to i32, !dbg !14
  %1187 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1186), !dbg !14
  %1188 = bitcast i32 %1187 to float, !dbg !14
  %1189 = select i1 %.not, float %1181, float %1188, !dbg !14
  %1190 = select i1 %.not, float %1188, float %1182, !dbg !14
  %1191 = select i1 %.not, float %1184, float %1183, !dbg !14
  %1192 = bitcast float %1191 to i32, !dbg !14
  %1193 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1192), !dbg !14
  %1194 = bitcast i32 %1193 to float, !dbg !14
  %1195 = select i1 %.not, float %1183, float %1194, !dbg !14
  %1196 = select i1 %.not, float %1194, float %1184, !dbg !14
  %1197 = select i1 %.not1, float %1195, float %1189, !dbg !14
  %1198 = bitcast float %1197 to i32, !dbg !14
  %1199 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1198), !dbg !14
  %1200 = bitcast i32 %1199 to float, !dbg !14
  %1201 = select i1 %.not1, float %1196, float %1190, !dbg !14
  %1202 = bitcast float %1201 to i32, !dbg !14
  %1203 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1202), !dbg !14
  %1204 = bitcast i32 %1203 to float, !dbg !14
  %1205 = insertelement <4 x float> poison, float %1189, i64 0, !dbg !14
  %1206 = insertelement <4 x float> %1205, float %1190, i64 1, !dbg !14
  %1207 = insertelement <4 x float> %1206, float %1200, i64 2, !dbg !14
  %1208 = insertelement <4 x float> %1207, float %1204, i64 3, !dbg !14
  %1209 = insertelement <4 x float> poison, float %1200, i64 0, !dbg !14
  %1210 = insertelement <4 x float> %1209, float %1204, i64 1, !dbg !14
  %1211 = insertelement <4 x float> %1210, float %1195, i64 2, !dbg !14
  %1212 = insertelement <4 x float> %1211, float %1196, i64 3, !dbg !14
  %1213 = select <4 x i1> %66, <4 x float> %1208, <4 x float> %1212, !dbg !14
  %1214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1124, <4 x float> zeroinitializer) #8, !dbg !14
  %1215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1126, <4 x float> %1214) #8, !dbg !14
  %1216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1128, <4 x float> %1215) #8, !dbg !14
  %1217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1130, <4 x float> %1216) #8, !dbg !14
  %1218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1132, <4 x float> %1217) #8, !dbg !14
  %1219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1134, <4 x float> %1218) #8, !dbg !14
  %1220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1136, <4 x float> %1219) #8, !dbg !14
  %1221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1138, <4 x float> %1220) #8, !dbg !14
  %1222 = extractelement <4 x float> %1221, i64 0, !dbg !14
  %1223 = extractelement <4 x float> %1221, i64 1, !dbg !14
  %1224 = extractelement <4 x float> %1221, i64 2, !dbg !14
  %1225 = extractelement <4 x float> %1221, i64 3, !dbg !14
  %1226 = select i1 %.not, float %1223, float %1222, !dbg !14
  %1227 = bitcast float %1226 to i32, !dbg !14
  %1228 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1227), !dbg !14
  %1229 = bitcast i32 %1228 to float, !dbg !14
  %1230 = select i1 %.not, float %1222, float %1229, !dbg !14
  %1231 = select i1 %.not, float %1229, float %1223, !dbg !14
  %1232 = select i1 %.not, float %1225, float %1224, !dbg !14
  %1233 = bitcast float %1232 to i32, !dbg !14
  %1234 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1233), !dbg !14
  %1235 = bitcast i32 %1234 to float, !dbg !14
  %1236 = select i1 %.not, float %1224, float %1235, !dbg !14
  %1237 = select i1 %.not, float %1235, float %1225, !dbg !14
  %1238 = select i1 %.not1, float %1236, float %1230, !dbg !14
  %1239 = bitcast float %1238 to i32, !dbg !14
  %1240 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1239), !dbg !14
  %1241 = bitcast i32 %1240 to float, !dbg !14
  %1242 = select i1 %.not1, float %1237, float %1231, !dbg !14
  %1243 = bitcast float %1242 to i32, !dbg !14
  %1244 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1243), !dbg !14
  %1245 = bitcast i32 %1244 to float, !dbg !14
  %1246 = insertelement <4 x float> poison, float %1230, i64 0, !dbg !14
  %1247 = insertelement <4 x float> %1246, float %1231, i64 1, !dbg !14
  %1248 = insertelement <4 x float> %1247, float %1241, i64 2, !dbg !14
  %1249 = insertelement <4 x float> %1248, float %1245, i64 3, !dbg !14
  %1250 = insertelement <4 x float> poison, float %1241, i64 0, !dbg !14
  %1251 = insertelement <4 x float> %1250, float %1245, i64 1, !dbg !14
  %1252 = insertelement <4 x float> %1251, float %1236, i64 2, !dbg !14
  %1253 = insertelement <4 x float> %1252, float %1237, i64 3, !dbg !14
  %1254 = select <4 x i1> %66, <4 x float> %1249, <4 x float> %1253, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %39, label %1255, label %1258, !dbg !32

1255:                                             ; preds = %813
  %address.thread832 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local833 = zext nneg i32 %address.thread832 to i64, !dbg !10
  %1256 = lshr i64 %address.local833, 1, !dbg !32
  %address.local835 = and i64 %1256, 96, !dbg !32
  %address.local836 = and i64 %address.local833, 15, !dbg !32
  %1257 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local836, !dbg !32
  %address.local840 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %1257, i64 %address.local835, !dbg !32
  store float %806, ptr addrspace(3) %address.local840, align 4, !dbg !32
  %address.local842 = getelementptr i8, ptr addrspace(3) %address.local840, i64 64, !dbg !32
  store float %809, ptr addrspace(3) %address.local842, align 4, !dbg !32
  br label %1258, !dbg !32

1258:                                             ; preds = %1255, %813
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.thread843 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.local857 = zext nneg i32 %address.thread843 to i64, !dbg !10
  %address.local858 = shl nuw nsw i64 %address.local857, 3, !dbg !9
  %address.local859 = and i64 %address.local858, 1536, !dbg !9
  %address.local860 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !9
  %address.local861 = ptrtoint ptr addrspace(3) %address.local860 to i64, !dbg !9
  %address.local862 = trunc i64 %address.local861 to i32, !dbg !9
  %1259 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local862), !dbg !28
  %address.local863 = shl nuw nsw i64 %address.local857, 6, !dbg !28
  %address.local864 = and i64 %address.local863, 16128, !dbg !28
  %address.local866 = and i64 %address.local858, 24, !dbg !28
  %1260 = or disjoint i64 %address.local866, %.idx5558, !dbg !28
  %1261 = shl i64 %1260, 1, !dbg !28
  %1262 = or disjoint i64 %1261, %address.local864, !dbg !28
  %1263 = trunc i64 %1262 to i32, !dbg !28
  %dma.buffer.offset1752 = or i32 %1263, 64, !dbg !28
  %dma.buffer.lds1754 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1259), !dbg !28
  %1264 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1754, i32 %dma.buffer.offset1752, <4 x i32> %dma.buffer.rsrc1759) #9, !dbg !28
  %address.local869 = or disjoint i64 %address.local857, 256, !dbg !9
  %address.local870 = shl nuw nsw i64 %address.local869, 3, !dbg !9
  %address.local871 = and i64 %address.local870, 3584, !dbg !9
  %address.local872 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local871, !dbg !9
  %address.local873 = ptrtoint ptr addrspace(3) %address.local872 to i64, !dbg !9
  %address.local874 = trunc i64 %address.local873 to i32, !dbg !9
  %1265 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local874), !dbg !28
  %address.local875 = shl nuw nsw i64 %address.local869, 6, !dbg !28
  %address.local876 = and i64 %address.local875, 32512, !dbg !28
  %1266 = shl i64 %.idx5558, 1, !dbg !28
  %address.local880.idx = shl nuw nsw i64 %address.local866, 1, !dbg !28
  %1267 = or disjoint i64 %address.local876, %address.local880.idx, !dbg !28
  %1268 = or disjoint i64 %1267, %1266, !dbg !28
  %1269 = trunc i64 %1268 to i32, !dbg !28
  %dma.buffer.offset1749 = or i32 %1269, 64, !dbg !28
  %dma.buffer.lds1751 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1265), !dbg !28
  %1270 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1751, i32 %dma.buffer.offset1749, <4 x i32> %dma.buffer.rsrc1759) #9, !dbg !28
  %1271 = extractvalue [2 x <8 x float>] %77, 0, !dbg !32
  %1272 = shufflevector <8 x float> %1271, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1273 = insertelement <4 x float> poison, float %806, i64 0, !dbg !32
  %1274 = shufflevector <4 x float> %1273, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1275 = fmul <4 x float> %1272, %1274, !dbg !32
  %1276 = fadd <4 x float> %1275, %1123, !dbg !33
  %1277 = shufflevector <8 x float> %1271, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1278 = fmul <4 x float> %1277, %1274, !dbg !32
  %1279 = fadd <4 x float> %1278, %1172, !dbg !33
  %1280 = extractvalue [2 x <8 x float>] %77, 1, !dbg !32
  %1281 = shufflevector <8 x float> %1280, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1282 = insertelement <4 x float> poison, float %809, i64 0, !dbg !32
  %1283 = shufflevector <4 x float> %1282, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1284 = fmul <4 x float> %1281, %1283, !dbg !32
  %1285 = fadd <4 x float> %1284, %1213, !dbg !33
  %1286 = shufflevector <8 x float> %1280, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1287 = fmul <4 x float> %1286, %1283, !dbg !32
  %1288 = fadd <4 x float> %1287, %1254, !dbg !33
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local881 = shl nuw nsw i32 %address.thread843, 3, !dbg !14
  %address.local882 = and i32 %address.local881, 504, !dbg !14
  %address.local883 = zext nneg i32 %address.local882 to i64, !dbg !14
  %address.local884 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !14
  %1289 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #10, !dbg !14
  %1290 = bitcast <4 x i32> %1289 to <8 x half>, !dbg !14
  %address.local886 = getelementptr i8, ptr addrspace(3) %address.local884, i64 1024, !dbg !14
  %1291 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #10, !dbg !14
  %1292 = bitcast <4 x i32> %1291 to <8 x half>, !dbg !14
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local884, i64 2048, !dbg !14
  %1293 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #10, !dbg !14
  %1294 = bitcast <4 x i32> %1293 to <8 x half>, !dbg !14
  %address.local890 = getelementptr i8, ptr addrspace(3) %address.local884, i64 3072, !dbg !14
  %1295 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local890, i32 0) #10, !dbg !14
  %1296 = bitcast <4 x i32> %1295 to <8 x half>, !dbg !14
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local884, i64 4096, !dbg !14
  %1297 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #10, !dbg !14
  %1298 = bitcast <4 x i32> %1297 to <8 x half>, !dbg !14
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local884, i64 5120, !dbg !14
  %1299 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #10, !dbg !14
  %1300 = bitcast <4 x i32> %1299 to <8 x half>, !dbg !14
  %address.local896 = getelementptr i8, ptr addrspace(3) %address.local884, i64 6144, !dbg !14
  %1301 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local896, i32 0) #10, !dbg !14
  %1302 = bitcast <4 x i32> %1301 to <8 x half>, !dbg !14
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local884, i64 7168, !dbg !14
  %1303 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #10, !dbg !14
  %1304 = bitcast <4 x i32> %1303 to <8 x half>, !dbg !14
  %1305 = shufflevector <8 x half> %1290, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1305, <4 x float> zeroinitializer) #8, !dbg !14
  %1307 = shufflevector <8 x half> %1292, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1307, <4 x float> %1306) #8, !dbg !14
  %1309 = shufflevector <8 x half> %1294, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1309, <4 x float> %1308) #8, !dbg !14
  %1311 = shufflevector <8 x half> %1296, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1311, <4 x float> %1310) #8, !dbg !14
  %1313 = shufflevector <8 x half> %1298, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1313, <4 x float> %1312) #8, !dbg !14
  %1315 = shufflevector <8 x half> %1300, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1315, <4 x float> %1314) #8, !dbg !14
  %1317 = shufflevector <8 x half> %1302, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1317, <4 x float> %1316) #8, !dbg !14
  %1319 = shufflevector <8 x half> %1304, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1319, <4 x float> %1318) #8, !dbg !14
  %1321 = extractelement <4 x float> %1320, i64 0, !dbg !14
  %1322 = extractelement <4 x float> %1320, i64 1, !dbg !14
  %1323 = extractelement <4 x float> %1320, i64 2, !dbg !14
  %1324 = extractelement <4 x float> %1320, i64 3, !dbg !14
  %1325 = select i1 %.not, float %1322, float %1321, !dbg !14
  %1326 = bitcast float %1325 to i32, !dbg !14
  %1327 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1326), !dbg !14
  %1328 = bitcast i32 %1327 to float, !dbg !14
  %1329 = select i1 %.not, float %1321, float %1328, !dbg !14
  %1330 = select i1 %.not, float %1328, float %1322, !dbg !14
  %1331 = select i1 %.not, float %1324, float %1323, !dbg !14
  %1332 = bitcast float %1331 to i32, !dbg !14
  %1333 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1332), !dbg !14
  %1334 = bitcast i32 %1333 to float, !dbg !14
  %1335 = select i1 %.not, float %1323, float %1334, !dbg !14
  %1336 = select i1 %.not, float %1334, float %1324, !dbg !14
  %1337 = select i1 %.not1, float %1335, float %1329, !dbg !14
  %1338 = bitcast float %1337 to i32, !dbg !14
  %1339 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1338), !dbg !14
  %1340 = bitcast i32 %1339 to float, !dbg !14
  %1341 = select i1 %.not1, float %1336, float %1330, !dbg !14
  %1342 = bitcast float %1341 to i32, !dbg !14
  %1343 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1342), !dbg !14
  %1344 = bitcast i32 %1343 to float, !dbg !14
  %1345 = insertelement <4 x float> poison, float %1329, i64 0, !dbg !14
  %1346 = insertelement <4 x float> %1345, float %1330, i64 1, !dbg !14
  %1347 = insertelement <4 x float> %1346, float %1340, i64 2, !dbg !14
  %1348 = insertelement <4 x float> %1347, float %1344, i64 3, !dbg !14
  %1349 = insertelement <4 x float> poison, float %1340, i64 0, !dbg !14
  %1350 = insertelement <4 x float> %1349, float %1344, i64 1, !dbg !14
  %1351 = insertelement <4 x float> %1350, float %1335, i64 2, !dbg !14
  %1352 = insertelement <4 x float> %1351, float %1336, i64 3, !dbg !14
  %1353 = select <4 x i1> %66, <4 x float> %1348, <4 x float> %1352, !dbg !14
  %1354 = shufflevector <8 x half> %1290, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1354, <4 x float> zeroinitializer) #8, !dbg !14
  %1356 = shufflevector <8 x half> %1292, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1356, <4 x float> %1355) #8, !dbg !14
  %1358 = shufflevector <8 x half> %1294, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1358, <4 x float> %1357) #8, !dbg !14
  %1360 = shufflevector <8 x half> %1296, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1360, <4 x float> %1359) #8, !dbg !14
  %1362 = shufflevector <8 x half> %1298, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1362, <4 x float> %1361) #8, !dbg !14
  %1364 = shufflevector <8 x half> %1300, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1364, <4 x float> %1363) #8, !dbg !14
  %1366 = shufflevector <8 x half> %1302, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1366, <4 x float> %1365) #8, !dbg !14
  %1368 = shufflevector <8 x half> %1304, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1368, <4 x float> %1367) #8, !dbg !14
  %1370 = extractelement <4 x float> %1369, i64 0, !dbg !14
  %1371 = extractelement <4 x float> %1369, i64 1, !dbg !14
  %1372 = extractelement <4 x float> %1369, i64 2, !dbg !14
  %1373 = extractelement <4 x float> %1369, i64 3, !dbg !14
  %1374 = select i1 %.not, float %1371, float %1370, !dbg !14
  %1375 = bitcast float %1374 to i32, !dbg !14
  %1376 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1375), !dbg !14
  %1377 = bitcast i32 %1376 to float, !dbg !14
  %1378 = select i1 %.not, float %1370, float %1377, !dbg !14
  %1379 = select i1 %.not, float %1377, float %1371, !dbg !14
  %1380 = select i1 %.not, float %1373, float %1372, !dbg !14
  %1381 = bitcast float %1380 to i32, !dbg !14
  %1382 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1381), !dbg !14
  %1383 = bitcast i32 %1382 to float, !dbg !14
  %1384 = select i1 %.not, float %1372, float %1383, !dbg !14
  %1385 = select i1 %.not, float %1383, float %1373, !dbg !14
  %1386 = select i1 %.not1, float %1384, float %1378, !dbg !14
  %1387 = bitcast float %1386 to i32, !dbg !14
  %1388 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1387), !dbg !14
  %1389 = bitcast i32 %1388 to float, !dbg !14
  %1390 = select i1 %.not1, float %1385, float %1379, !dbg !14
  %1391 = bitcast float %1390 to i32, !dbg !14
  %1392 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1391), !dbg !14
  %1393 = bitcast i32 %1392 to float, !dbg !14
  %1394 = insertelement <4 x float> poison, float %1378, i64 0, !dbg !14
  %1395 = insertelement <4 x float> %1394, float %1379, i64 1, !dbg !14
  %1396 = insertelement <4 x float> %1395, float %1389, i64 2, !dbg !14
  %1397 = insertelement <4 x float> %1396, float %1393, i64 3, !dbg !14
  %1398 = insertelement <4 x float> poison, float %1389, i64 0, !dbg !14
  %1399 = insertelement <4 x float> %1398, float %1393, i64 1, !dbg !14
  %1400 = insertelement <4 x float> %1399, float %1384, i64 2, !dbg !14
  %1401 = insertelement <4 x float> %1400, float %1385, i64 3, !dbg !14
  %1402 = select <4 x i1> %66, <4 x float> %1397, <4 x float> %1401, !dbg !14
  %1403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1305, <4 x float> zeroinitializer) #8, !dbg !14
  %1404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1307, <4 x float> %1403) #8, !dbg !14
  %1405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1309, <4 x float> %1404) #8, !dbg !14
  %1406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1311, <4 x float> %1405) #8, !dbg !14
  %1407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1313, <4 x float> %1406) #8, !dbg !14
  %1408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1315, <4 x float> %1407) #8, !dbg !14
  %1409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1317, <4 x float> %1408) #8, !dbg !14
  %1410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1319, <4 x float> %1409) #8, !dbg !14
  %1411 = extractelement <4 x float> %1410, i64 0, !dbg !14
  %1412 = extractelement <4 x float> %1410, i64 1, !dbg !14
  %1413 = extractelement <4 x float> %1410, i64 2, !dbg !14
  %1414 = extractelement <4 x float> %1410, i64 3, !dbg !14
  %1415 = select i1 %.not, float %1412, float %1411, !dbg !14
  %1416 = bitcast float %1415 to i32, !dbg !14
  %1417 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1416), !dbg !14
  %1418 = bitcast i32 %1417 to float, !dbg !14
  %1419 = select i1 %.not, float %1411, float %1418, !dbg !14
  %1420 = select i1 %.not, float %1418, float %1412, !dbg !14
  %1421 = select i1 %.not, float %1414, float %1413, !dbg !14
  %1422 = bitcast float %1421 to i32, !dbg !14
  %1423 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1422), !dbg !14
  %1424 = bitcast i32 %1423 to float, !dbg !14
  %1425 = select i1 %.not, float %1413, float %1424, !dbg !14
  %1426 = select i1 %.not, float %1424, float %1414, !dbg !14
  %1427 = select i1 %.not1, float %1425, float %1419, !dbg !14
  %1428 = bitcast float %1427 to i32, !dbg !14
  %1429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1428), !dbg !14
  %1430 = bitcast i32 %1429 to float, !dbg !14
  %1431 = select i1 %.not1, float %1426, float %1420, !dbg !14
  %1432 = bitcast float %1431 to i32, !dbg !14
  %1433 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1432), !dbg !14
  %1434 = bitcast i32 %1433 to float, !dbg !14
  %1435 = insertelement <4 x float> poison, float %1419, i64 0, !dbg !14
  %1436 = insertelement <4 x float> %1435, float %1420, i64 1, !dbg !14
  %1437 = insertelement <4 x float> %1436, float %1430, i64 2, !dbg !14
  %1438 = insertelement <4 x float> %1437, float %1434, i64 3, !dbg !14
  %1439 = insertelement <4 x float> poison, float %1430, i64 0, !dbg !14
  %1440 = insertelement <4 x float> %1439, float %1434, i64 1, !dbg !14
  %1441 = insertelement <4 x float> %1440, float %1425, i64 2, !dbg !14
  %1442 = insertelement <4 x float> %1441, float %1426, i64 3, !dbg !14
  %1443 = select <4 x i1> %66, <4 x float> %1438, <4 x float> %1442, !dbg !14
  %1444 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1354, <4 x float> zeroinitializer) #8, !dbg !14
  %1445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1356, <4 x float> %1444) #8, !dbg !14
  %1446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1358, <4 x float> %1445) #8, !dbg !14
  %1447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1360, <4 x float> %1446) #8, !dbg !14
  %1448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1362, <4 x float> %1447) #8, !dbg !14
  %1449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1364, <4 x float> %1448) #8, !dbg !14
  %1450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1366, <4 x float> %1449) #8, !dbg !14
  %1451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1368, <4 x float> %1450) #8, !dbg !14
  %1452 = extractelement <4 x float> %1451, i64 0, !dbg !14
  %1453 = extractelement <4 x float> %1451, i64 1, !dbg !14
  %1454 = extractelement <4 x float> %1451, i64 2, !dbg !14
  %1455 = extractelement <4 x float> %1451, i64 3, !dbg !14
  %1456 = select i1 %.not, float %1453, float %1452, !dbg !14
  %1457 = bitcast float %1456 to i32, !dbg !14
  %1458 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1457), !dbg !14
  %1459 = bitcast i32 %1458 to float, !dbg !14
  %1460 = select i1 %.not, float %1452, float %1459, !dbg !14
  %1461 = select i1 %.not, float %1459, float %1453, !dbg !14
  %1462 = select i1 %.not, float %1455, float %1454, !dbg !14
  %1463 = bitcast float %1462 to i32, !dbg !14
  %1464 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1463), !dbg !14
  %1465 = bitcast i32 %1464 to float, !dbg !14
  %1466 = select i1 %.not, float %1454, float %1465, !dbg !14
  %1467 = select i1 %.not, float %1465, float %1455, !dbg !14
  %1468 = select i1 %.not1, float %1466, float %1460, !dbg !14
  %1469 = bitcast float %1468 to i32, !dbg !14
  %1470 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1469), !dbg !14
  %1471 = bitcast i32 %1470 to float, !dbg !14
  %1472 = select i1 %.not1, float %1467, float %1461, !dbg !14
  %1473 = bitcast float %1472 to i32, !dbg !14
  %1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1473), !dbg !14
  %1475 = bitcast i32 %1474 to float, !dbg !14
  %1476 = insertelement <4 x float> poison, float %1460, i64 0, !dbg !14
  %1477 = insertelement <4 x float> %1476, float %1461, i64 1, !dbg !14
  %1478 = insertelement <4 x float> %1477, float %1471, i64 2, !dbg !14
  %1479 = insertelement <4 x float> %1478, float %1475, i64 3, !dbg !14
  %1480 = insertelement <4 x float> poison, float %1471, i64 0, !dbg !14
  %1481 = insertelement <4 x float> %1480, float %1475, i64 1, !dbg !14
  %1482 = insertelement <4 x float> %1481, float %1466, i64 2, !dbg !14
  %1483 = insertelement <4 x float> %1482, float %1467, i64 3, !dbg !14
  %1484 = select <4 x i1> %66, <4 x float> %1479, <4 x float> %1483, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %39, label %1485, label %1488, !dbg !32

1485:                                             ; preds = %1258
  %address.thread899 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local900 = zext nneg i32 %address.thread899 to i64, !dbg !10
  %1486 = lshr i64 %address.local900, 1, !dbg !32
  %address.local902 = and i64 %1486, 96, !dbg !32
  %address.local903 = and i64 %address.local900, 15, !dbg !32
  %1487 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !32
  %address.local907 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %1487, i64 %address.local902, !dbg !32
  store float %806, ptr addrspace(3) %address.local907, align 4, !dbg !32
  %address.local909 = getelementptr i8, ptr addrspace(3) %address.local907, i64 64, !dbg !32
  store float %809, ptr addrspace(3) %address.local909, align 4, !dbg !32
  br label %1488, !dbg !32

1488:                                             ; preds = %1485, %1258
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.thread910 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.local924 = zext nneg i32 %address.thread910 to i64, !dbg !10
  %address.local925 = shl nuw nsw i64 %address.local924, 3, !dbg !9
  %address.local926 = and i64 %address.local925, 1536, !dbg !9
  %address.local927 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local926, !dbg !9
  %address.local928 = ptrtoint ptr addrspace(3) %address.local927 to i64, !dbg !9
  %address.local929 = trunc i64 %address.local928 to i32, !dbg !9
  %1489 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local929), !dbg !28
  %address.local930 = shl nuw nsw i64 %address.local924, 6, !dbg !28
  %address.local931 = and i64 %address.local930, 16128, !dbg !28
  %address.local933 = and i64 %address.local925, 24, !dbg !28
  %1490 = or disjoint i64 %address.local933, %.idx5558, !dbg !28
  %1491 = shl i64 %1490, 1, !dbg !28
  %1492 = or disjoint i64 %1491, %address.local931, !dbg !28
  %1493 = trunc i64 %1492 to i32, !dbg !28
  %dma.buffer.offset1746 = or i32 %1493, 128, !dbg !28
  %dma.buffer.lds1748 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1489), !dbg !28
  %1494 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1748, i32 %dma.buffer.offset1746, <4 x i32> %dma.buffer.rsrc1759) #9, !dbg !28
  %address.local936 = or disjoint i64 %address.local924, 256, !dbg !9
  %address.local937 = shl nuw nsw i64 %address.local936, 3, !dbg !9
  %address.local938 = and i64 %address.local937, 3584, !dbg !9
  %address.local939 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !9
  %address.local940 = ptrtoint ptr addrspace(3) %address.local939 to i64, !dbg !9
  %address.local941 = trunc i64 %address.local940 to i32, !dbg !9
  %1495 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local941), !dbg !28
  %address.local942 = shl nuw nsw i64 %address.local936, 6, !dbg !28
  %address.local943 = and i64 %address.local942, 32512, !dbg !28
  %address.local947.idx = shl nuw nsw i64 %address.local933, 1, !dbg !28
  %1496 = or disjoint i64 %address.local943, %address.local947.idx, !dbg !28
  %1497 = or disjoint i64 %1496, %1266, !dbg !28
  %1498 = trunc i64 %1497 to i32, !dbg !28
  %dma.buffer.offset1743 = or i32 %1498, 128, !dbg !28
  %dma.buffer.lds1745 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1495), !dbg !28
  %1499 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1745, i32 %dma.buffer.offset1743, <4 x i32> %dma.buffer.rsrc1759) #9, !dbg !28
  %1500 = extractvalue [2 x <8 x float>] %74, 0, !dbg !32
  %1501 = shufflevector <8 x float> %1500, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1502 = fmul <4 x float> %1501, %1274, !dbg !32
  %1503 = fadd <4 x float> %1502, %1353, !dbg !33
  %1504 = shufflevector <8 x float> %1500, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1505 = fmul <4 x float> %1504, %1274, !dbg !32
  %1506 = fadd <4 x float> %1505, %1402, !dbg !33
  %1507 = extractvalue [2 x <8 x float>] %74, 1, !dbg !32
  %1508 = shufflevector <8 x float> %1507, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1509 = fmul <4 x float> %1508, %1283, !dbg !32
  %1510 = fadd <4 x float> %1509, %1443, !dbg !33
  %1511 = shufflevector <8 x float> %1507, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1512 = fmul <4 x float> %1511, %1283, !dbg !32
  %1513 = fadd <4 x float> %1512, %1484, !dbg !33
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local948 = shl nuw nsw i32 %address.thread910, 3, !dbg !14
  %address.local949 = and i32 %address.local948, 504, !dbg !14
  %address.local950 = zext nneg i32 %address.local949 to i64, !dbg !14
  %address.local951 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local950, !dbg !14
  %1514 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local951, i32 0) #10, !dbg !14
  %1515 = bitcast <4 x i32> %1514 to <8 x half>, !dbg !14
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local951, i64 1024, !dbg !14
  %1516 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !14
  %1517 = bitcast <4 x i32> %1516 to <8 x half>, !dbg !14
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local951, i64 2048, !dbg !14
  %1518 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !14
  %1519 = bitcast <4 x i32> %1518 to <8 x half>, !dbg !14
  %address.local957 = getelementptr i8, ptr addrspace(3) %address.local951, i64 3072, !dbg !14
  %1520 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local957, i32 0) #10, !dbg !14
  %1521 = bitcast <4 x i32> %1520 to <8 x half>, !dbg !14
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local951, i64 4096, !dbg !14
  %1522 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !14
  %1523 = bitcast <4 x i32> %1522 to <8 x half>, !dbg !14
  %address.local961 = getelementptr i8, ptr addrspace(3) %address.local951, i64 5120, !dbg !14
  %1524 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local961, i32 0) #10, !dbg !14
  %1525 = bitcast <4 x i32> %1524 to <8 x half>, !dbg !14
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local951, i64 6144, !dbg !14
  %1526 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !14
  %1527 = bitcast <4 x i32> %1526 to <8 x half>, !dbg !14
  %address.local965 = getelementptr i8, ptr addrspace(3) %address.local951, i64 7168, !dbg !14
  %1528 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local965, i32 0) #10, !dbg !14
  %1529 = bitcast <4 x i32> %1528 to <8 x half>, !dbg !14
  %1530 = shufflevector <8 x half> %1515, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1530, <4 x float> zeroinitializer) #8, !dbg !14
  %1532 = shufflevector <8 x half> %1517, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1533 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1532, <4 x float> %1531) #8, !dbg !14
  %1534 = shufflevector <8 x half> %1519, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1535 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1534, <4 x float> %1533) #8, !dbg !14
  %1536 = shufflevector <8 x half> %1521, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1537 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1536, <4 x float> %1535) #8, !dbg !14
  %1538 = shufflevector <8 x half> %1523, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1539 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1538, <4 x float> %1537) #8, !dbg !14
  %1540 = shufflevector <8 x half> %1525, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1540, <4 x float> %1539) #8, !dbg !14
  %1542 = shufflevector <8 x half> %1527, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1543 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1542, <4 x float> %1541) #8, !dbg !14
  %1544 = shufflevector <8 x half> %1529, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1545 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1544, <4 x float> %1543) #8, !dbg !14
  %1546 = extractelement <4 x float> %1545, i64 0, !dbg !14
  %1547 = extractelement <4 x float> %1545, i64 1, !dbg !14
  %1548 = extractelement <4 x float> %1545, i64 2, !dbg !14
  %1549 = extractelement <4 x float> %1545, i64 3, !dbg !14
  %1550 = select i1 %.not, float %1547, float %1546, !dbg !14
  %1551 = bitcast float %1550 to i32, !dbg !14
  %1552 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1551), !dbg !14
  %1553 = bitcast i32 %1552 to float, !dbg !14
  %1554 = select i1 %.not, float %1546, float %1553, !dbg !14
  %1555 = select i1 %.not, float %1553, float %1547, !dbg !14
  %1556 = select i1 %.not, float %1549, float %1548, !dbg !14
  %1557 = bitcast float %1556 to i32, !dbg !14
  %1558 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1557), !dbg !14
  %1559 = bitcast i32 %1558 to float, !dbg !14
  %1560 = select i1 %.not, float %1548, float %1559, !dbg !14
  %1561 = select i1 %.not, float %1559, float %1549, !dbg !14
  %1562 = select i1 %.not1, float %1560, float %1554, !dbg !14
  %1563 = bitcast float %1562 to i32, !dbg !14
  %1564 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1563), !dbg !14
  %1565 = bitcast i32 %1564 to float, !dbg !14
  %1566 = select i1 %.not1, float %1561, float %1555, !dbg !14
  %1567 = bitcast float %1566 to i32, !dbg !14
  %1568 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1567), !dbg !14
  %1569 = bitcast i32 %1568 to float, !dbg !14
  %1570 = insertelement <4 x float> poison, float %1554, i64 0, !dbg !14
  %1571 = insertelement <4 x float> %1570, float %1555, i64 1, !dbg !14
  %1572 = insertelement <4 x float> %1571, float %1565, i64 2, !dbg !14
  %1573 = insertelement <4 x float> %1572, float %1569, i64 3, !dbg !14
  %1574 = insertelement <4 x float> poison, float %1565, i64 0, !dbg !14
  %1575 = insertelement <4 x float> %1574, float %1569, i64 1, !dbg !14
  %1576 = insertelement <4 x float> %1575, float %1560, i64 2, !dbg !14
  %1577 = insertelement <4 x float> %1576, float %1561, i64 3, !dbg !14
  %1578 = select <4 x i1> %66, <4 x float> %1573, <4 x float> %1577, !dbg !14
  %1579 = shufflevector <8 x half> %1515, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1580 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1579, <4 x float> zeroinitializer) #8, !dbg !14
  %1581 = shufflevector <8 x half> %1517, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1582 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1581, <4 x float> %1580) #8, !dbg !14
  %1583 = shufflevector <8 x half> %1519, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1584 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1583, <4 x float> %1582) #8, !dbg !14
  %1585 = shufflevector <8 x half> %1521, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1586 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1585, <4 x float> %1584) #8, !dbg !14
  %1587 = shufflevector <8 x half> %1523, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1587, <4 x float> %1586) #8, !dbg !14
  %1589 = shufflevector <8 x half> %1525, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1589, <4 x float> %1588) #8, !dbg !14
  %1591 = shufflevector <8 x half> %1527, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1591, <4 x float> %1590) #8, !dbg !14
  %1593 = shufflevector <8 x half> %1529, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1594 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1593, <4 x float> %1592) #8, !dbg !14
  %1595 = extractelement <4 x float> %1594, i64 0, !dbg !14
  %1596 = extractelement <4 x float> %1594, i64 1, !dbg !14
  %1597 = extractelement <4 x float> %1594, i64 2, !dbg !14
  %1598 = extractelement <4 x float> %1594, i64 3, !dbg !14
  %1599 = select i1 %.not, float %1596, float %1595, !dbg !14
  %1600 = bitcast float %1599 to i32, !dbg !14
  %1601 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1600), !dbg !14
  %1602 = bitcast i32 %1601 to float, !dbg !14
  %1603 = select i1 %.not, float %1595, float %1602, !dbg !14
  %1604 = select i1 %.not, float %1602, float %1596, !dbg !14
  %1605 = select i1 %.not, float %1598, float %1597, !dbg !14
  %1606 = bitcast float %1605 to i32, !dbg !14
  %1607 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1606), !dbg !14
  %1608 = bitcast i32 %1607 to float, !dbg !14
  %1609 = select i1 %.not, float %1597, float %1608, !dbg !14
  %1610 = select i1 %.not, float %1608, float %1598, !dbg !14
  %1611 = select i1 %.not1, float %1609, float %1603, !dbg !14
  %1612 = bitcast float %1611 to i32, !dbg !14
  %1613 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1612), !dbg !14
  %1614 = bitcast i32 %1613 to float, !dbg !14
  %1615 = select i1 %.not1, float %1610, float %1604, !dbg !14
  %1616 = bitcast float %1615 to i32, !dbg !14
  %1617 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1616), !dbg !14
  %1618 = bitcast i32 %1617 to float, !dbg !14
  %1619 = insertelement <4 x float> poison, float %1603, i64 0, !dbg !14
  %1620 = insertelement <4 x float> %1619, float %1604, i64 1, !dbg !14
  %1621 = insertelement <4 x float> %1620, float %1614, i64 2, !dbg !14
  %1622 = insertelement <4 x float> %1621, float %1618, i64 3, !dbg !14
  %1623 = insertelement <4 x float> poison, float %1614, i64 0, !dbg !14
  %1624 = insertelement <4 x float> %1623, float %1618, i64 1, !dbg !14
  %1625 = insertelement <4 x float> %1624, float %1609, i64 2, !dbg !14
  %1626 = insertelement <4 x float> %1625, float %1610, i64 3, !dbg !14
  %1627 = select <4 x i1> %66, <4 x float> %1622, <4 x float> %1626, !dbg !14
  %1628 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1530, <4 x float> zeroinitializer) #8, !dbg !14
  %1629 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1532, <4 x float> %1628) #8, !dbg !14
  %1630 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1534, <4 x float> %1629) #8, !dbg !14
  %1631 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1536, <4 x float> %1630) #8, !dbg !14
  %1632 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1538, <4 x float> %1631) #8, !dbg !14
  %1633 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1540, <4 x float> %1632) #8, !dbg !14
  %1634 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1542, <4 x float> %1633) #8, !dbg !14
  %1635 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1544, <4 x float> %1634) #8, !dbg !14
  %1636 = extractelement <4 x float> %1635, i64 0, !dbg !14
  %1637 = extractelement <4 x float> %1635, i64 1, !dbg !14
  %1638 = extractelement <4 x float> %1635, i64 2, !dbg !14
  %1639 = extractelement <4 x float> %1635, i64 3, !dbg !14
  %1640 = select i1 %.not, float %1637, float %1636, !dbg !14
  %1641 = bitcast float %1640 to i32, !dbg !14
  %1642 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1641), !dbg !14
  %1643 = bitcast i32 %1642 to float, !dbg !14
  %1644 = select i1 %.not, float %1636, float %1643, !dbg !14
  %1645 = select i1 %.not, float %1643, float %1637, !dbg !14
  %1646 = select i1 %.not, float %1639, float %1638, !dbg !14
  %1647 = bitcast float %1646 to i32, !dbg !14
  %1648 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1647), !dbg !14
  %1649 = bitcast i32 %1648 to float, !dbg !14
  %1650 = select i1 %.not, float %1638, float %1649, !dbg !14
  %1651 = select i1 %.not, float %1649, float %1639, !dbg !14
  %1652 = select i1 %.not1, float %1650, float %1644, !dbg !14
  %1653 = bitcast float %1652 to i32, !dbg !14
  %1654 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1653), !dbg !14
  %1655 = bitcast i32 %1654 to float, !dbg !14
  %1656 = select i1 %.not1, float %1651, float %1645, !dbg !14
  %1657 = bitcast float %1656 to i32, !dbg !14
  %1658 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1657), !dbg !14
  %1659 = bitcast i32 %1658 to float, !dbg !14
  %1660 = insertelement <4 x float> poison, float %1644, i64 0, !dbg !14
  %1661 = insertelement <4 x float> %1660, float %1645, i64 1, !dbg !14
  %1662 = insertelement <4 x float> %1661, float %1655, i64 2, !dbg !14
  %1663 = insertelement <4 x float> %1662, float %1659, i64 3, !dbg !14
  %1664 = insertelement <4 x float> poison, float %1655, i64 0, !dbg !14
  %1665 = insertelement <4 x float> %1664, float %1659, i64 1, !dbg !14
  %1666 = insertelement <4 x float> %1665, float %1650, i64 2, !dbg !14
  %1667 = insertelement <4 x float> %1666, float %1651, i64 3, !dbg !14
  %1668 = select <4 x i1> %66, <4 x float> %1663, <4 x float> %1667, !dbg !14
  %1669 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1579, <4 x float> zeroinitializer) #8, !dbg !14
  %1670 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1581, <4 x float> %1669) #8, !dbg !14
  %1671 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1583, <4 x float> %1670) #8, !dbg !14
  %1672 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1585, <4 x float> %1671) #8, !dbg !14
  %1673 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1587, <4 x float> %1672) #8, !dbg !14
  %1674 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1589, <4 x float> %1673) #8, !dbg !14
  %1675 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1591, <4 x float> %1674) #8, !dbg !14
  %1676 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1593, <4 x float> %1675) #8, !dbg !14
  %1677 = extractelement <4 x float> %1676, i64 0, !dbg !14
  %1678 = extractelement <4 x float> %1676, i64 1, !dbg !14
  %1679 = extractelement <4 x float> %1676, i64 2, !dbg !14
  %1680 = extractelement <4 x float> %1676, i64 3, !dbg !14
  %1681 = select i1 %.not, float %1678, float %1677, !dbg !14
  %1682 = bitcast float %1681 to i32, !dbg !14
  %1683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1682), !dbg !14
  %1684 = bitcast i32 %1683 to float, !dbg !14
  %1685 = select i1 %.not, float %1677, float %1684, !dbg !14
  %1686 = select i1 %.not, float %1684, float %1678, !dbg !14
  %1687 = select i1 %.not, float %1680, float %1679, !dbg !14
  %1688 = bitcast float %1687 to i32, !dbg !14
  %1689 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1688), !dbg !14
  %1690 = bitcast i32 %1689 to float, !dbg !14
  %1691 = select i1 %.not, float %1679, float %1690, !dbg !14
  %1692 = select i1 %.not, float %1690, float %1680, !dbg !14
  %1693 = select i1 %.not1, float %1691, float %1685, !dbg !14
  %1694 = bitcast float %1693 to i32, !dbg !14
  %1695 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1694), !dbg !14
  %1696 = bitcast i32 %1695 to float, !dbg !14
  %1697 = select i1 %.not1, float %1692, float %1686, !dbg !14
  %1698 = bitcast float %1697 to i32, !dbg !14
  %1699 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1698), !dbg !14
  %1700 = bitcast i32 %1699 to float, !dbg !14
  %1701 = insertelement <4 x float> poison, float %1685, i64 0, !dbg !14
  %1702 = insertelement <4 x float> %1701, float %1686, i64 1, !dbg !14
  %1703 = insertelement <4 x float> %1702, float %1696, i64 2, !dbg !14
  %1704 = insertelement <4 x float> %1703, float %1700, i64 3, !dbg !14
  %1705 = insertelement <4 x float> poison, float %1696, i64 0, !dbg !14
  %1706 = insertelement <4 x float> %1705, float %1700, i64 1, !dbg !14
  %1707 = insertelement <4 x float> %1706, float %1691, i64 2, !dbg !14
  %1708 = insertelement <4 x float> %1707, float %1692, i64 3, !dbg !14
  %1709 = select <4 x i1> %66, <4 x float> %1704, <4 x float> %1708, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %39, label %1710, label %1713, !dbg !32

1710:                                             ; preds = %1488
  %address.thread966 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local967 = zext nneg i32 %address.thread966 to i64, !dbg !10
  %1711 = lshr i64 %address.local967, 1, !dbg !32
  %address.local969 = and i64 %1711, 96, !dbg !32
  %address.local970 = and i64 %address.local967, 15, !dbg !32
  %1712 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local970, !dbg !32
  %address.local974 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %1712, i64 %address.local969, !dbg !32
  store float %806, ptr addrspace(3) %address.local974, align 4, !dbg !32
  %address.local976 = getelementptr i8, ptr addrspace(3) %address.local974, i64 64, !dbg !32
  store float %809, ptr addrspace(3) %address.local976, align 4, !dbg !32
  br label %1713, !dbg !32

1713:                                             ; preds = %1710, %1488
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.thread977 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.local991 = zext nneg i32 %address.thread977 to i64, !dbg !10
  %address.local992 = shl nuw nsw i64 %address.local991, 3, !dbg !9
  %address.local993 = and i64 %address.local992, 1536, !dbg !9
  %address.local994 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local993, !dbg !9
  %address.local995 = ptrtoint ptr addrspace(3) %address.local994 to i64, !dbg !9
  %address.local996 = trunc i64 %address.local995 to i32, !dbg !9
  %1714 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local996), !dbg !28
  %address.local997 = shl nuw nsw i64 %address.local991, 6, !dbg !28
  %address.local1000 = and i64 %address.local992, 24, !dbg !28
  %1715 = or disjoint i64 %address.local1000, %.idx5558, !dbg !28
  %1716 = shl i64 %1715, 1, !dbg !28
  %1717 = or disjoint i64 %1716, %address.local997, !dbg !28
  %1718 = trunc i64 %1717 to i32, !dbg !28
  %dma.buffer.offset1740 = or i32 %1718, 192, !dbg !28
  %dma.buffer.lds1742 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1714), !dbg !28
  %1719 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1742, i32 %dma.buffer.offset1740, <4 x i32> %dma.buffer.rsrc1759) #9, !dbg !28
  %address.local1003 = or disjoint i64 %address.local991, 256, !dbg !9
  %address.local1004 = shl nuw nsw i64 %address.local1003, 3, !dbg !9
  %address.local1005 = and i64 %address.local1004, 3584, !dbg !9
  %address.local1006 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1005, !dbg !9
  %address.local1007 = ptrtoint ptr addrspace(3) %address.local1006 to i64, !dbg !9
  %address.local1008 = trunc i64 %address.local1007 to i32, !dbg !9
  %1720 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1008), !dbg !28
  %address.local1009 = shl nuw nsw i64 %address.local1003, 6, !dbg !28
  %address.local1010 = and i64 %address.local1009, 32512, !dbg !28
  %1721 = or disjoint i64 %1716, %address.local1010, !dbg !28
  %1722 = trunc i64 %1721 to i32, !dbg !28
  %dma.buffer.offset = or i32 %1722, 192, !dbg !28
  %dma.buffer.lds = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1720), !dbg !28
  %1723 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1759) #9, !dbg !28
  %1724 = extractvalue [2 x <8 x float>] %73, 0, !dbg !32
  %1725 = shufflevector <8 x float> %1724, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1726 = fmul <4 x float> %1725, %1274, !dbg !32
  %1727 = fadd <4 x float> %1726, %1578, !dbg !33
  %1728 = shufflevector <8 x float> %1724, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1729 = fmul <4 x float> %1728, %1274, !dbg !32
  %1730 = fadd <4 x float> %1729, %1627, !dbg !33
  %1731 = extractvalue [2 x <8 x float>] %73, 1, !dbg !32
  %1732 = shufflevector <8 x float> %1731, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1733 = fmul <4 x float> %1732, %1283, !dbg !32
  %1734 = fadd <4 x float> %1733, %1668, !dbg !33
  %1735 = shufflevector <8 x float> %1731, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1736 = fmul <4 x float> %1735, %1283, !dbg !32
  %1737 = fadd <4 x float> %1736, %1709, !dbg !33
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local1015 = shl nuw nsw i32 %address.thread977, 3, !dbg !14
  %address.local1016 = and i32 %address.local1015, 504, !dbg !14
  %address.local1017 = zext nneg i32 %address.local1016 to i64, !dbg !14
  %address.local1018 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1017, !dbg !14
  %1738 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1018, i32 0) #10, !dbg !14
  %1739 = bitcast <4 x i32> %1738 to <8 x half>, !dbg !14
  %address.local1020 = getelementptr i8, ptr addrspace(3) %address.local1018, i64 1024, !dbg !14
  %1740 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #10, !dbg !14
  %1741 = bitcast <4 x i32> %1740 to <8 x half>, !dbg !14
  %address.local1022 = getelementptr i8, ptr addrspace(3) %address.local1018, i64 2048, !dbg !14
  %1742 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #10, !dbg !14
  %1743 = bitcast <4 x i32> %1742 to <8 x half>, !dbg !14
  %address.local1024 = getelementptr i8, ptr addrspace(3) %address.local1018, i64 3072, !dbg !14
  %1744 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1024, i32 0) #10, !dbg !14
  %1745 = bitcast <4 x i32> %1744 to <8 x half>, !dbg !14
  %address.local1026 = getelementptr i8, ptr addrspace(3) %address.local1018, i64 4096, !dbg !14
  %1746 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #10, !dbg !14
  %1747 = bitcast <4 x i32> %1746 to <8 x half>, !dbg !14
  %address.local1028 = getelementptr i8, ptr addrspace(3) %address.local1018, i64 5120, !dbg !14
  %1748 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1028, i32 0) #10, !dbg !14
  %1749 = bitcast <4 x i32> %1748 to <8 x half>, !dbg !14
  %address.local1030 = getelementptr i8, ptr addrspace(3) %address.local1018, i64 6144, !dbg !14
  %1750 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #10, !dbg !14
  %1751 = bitcast <4 x i32> %1750 to <8 x half>, !dbg !14
  %address.local1032 = getelementptr i8, ptr addrspace(3) %address.local1018, i64 7168, !dbg !14
  %1752 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1032, i32 0) #10, !dbg !14
  %1753 = bitcast <4 x i32> %1752 to <8 x half>, !dbg !14
  %1754 = shufflevector <8 x half> %1739, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1755 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1754, <4 x float> zeroinitializer) #8, !dbg !14
  %1756 = shufflevector <8 x half> %1741, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1757 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1756, <4 x float> %1755) #8, !dbg !14
  %1758 = shufflevector <8 x half> %1743, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1759 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1758, <4 x float> %1757) #8, !dbg !14
  %1760 = shufflevector <8 x half> %1745, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1761 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1760, <4 x float> %1759) #8, !dbg !14
  %1762 = shufflevector <8 x half> %1747, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1763 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1762, <4 x float> %1761) #8, !dbg !14
  %1764 = shufflevector <8 x half> %1749, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1765 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1764, <4 x float> %1763) #8, !dbg !14
  %1766 = shufflevector <8 x half> %1751, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1767 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1766, <4 x float> %1765) #8, !dbg !14
  %1768 = shufflevector <8 x half> %1753, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1769 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1768, <4 x float> %1767) #8, !dbg !14
  %1770 = extractelement <4 x float> %1769, i64 0, !dbg !14
  %1771 = extractelement <4 x float> %1769, i64 1, !dbg !14
  %1772 = extractelement <4 x float> %1769, i64 2, !dbg !14
  %1773 = extractelement <4 x float> %1769, i64 3, !dbg !14
  %1774 = select i1 %.not, float %1771, float %1770, !dbg !14
  %1775 = bitcast float %1774 to i32, !dbg !14
  %1776 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1775), !dbg !14
  %1777 = bitcast i32 %1776 to float, !dbg !14
  %1778 = select i1 %.not, float %1770, float %1777, !dbg !14
  %1779 = select i1 %.not, float %1777, float %1771, !dbg !14
  %1780 = select i1 %.not, float %1773, float %1772, !dbg !14
  %1781 = bitcast float %1780 to i32, !dbg !14
  %1782 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1781), !dbg !14
  %1783 = bitcast i32 %1782 to float, !dbg !14
  %1784 = select i1 %.not, float %1772, float %1783, !dbg !14
  %1785 = select i1 %.not, float %1783, float %1773, !dbg !14
  %1786 = select i1 %.not1, float %1784, float %1778, !dbg !14
  %1787 = bitcast float %1786 to i32, !dbg !14
  %1788 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1787), !dbg !14
  %1789 = bitcast i32 %1788 to float, !dbg !14
  %1790 = select i1 %.not1, float %1785, float %1779, !dbg !14
  %1791 = bitcast float %1790 to i32, !dbg !14
  %1792 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1791), !dbg !14
  %1793 = bitcast i32 %1792 to float, !dbg !14
  %1794 = insertelement <4 x float> poison, float %1778, i64 0, !dbg !14
  %1795 = insertelement <4 x float> %1794, float %1779, i64 1, !dbg !14
  %1796 = insertelement <4 x float> %1795, float %1789, i64 2, !dbg !14
  %1797 = insertelement <4 x float> %1796, float %1793, i64 3, !dbg !14
  %1798 = insertelement <4 x float> poison, float %1789, i64 0, !dbg !14
  %1799 = insertelement <4 x float> %1798, float %1793, i64 1, !dbg !14
  %1800 = insertelement <4 x float> %1799, float %1784, i64 2, !dbg !14
  %1801 = insertelement <4 x float> %1800, float %1785, i64 3, !dbg !14
  %1802 = select <4 x i1> %66, <4 x float> %1797, <4 x float> %1801, !dbg !14
  %1803 = shufflevector <8 x half> %1739, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1804 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1803, <4 x float> zeroinitializer) #8, !dbg !14
  %1805 = shufflevector <8 x half> %1741, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1806 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1805, <4 x float> %1804) #8, !dbg !14
  %1807 = shufflevector <8 x half> %1743, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1808 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1807, <4 x float> %1806) #8, !dbg !14
  %1809 = shufflevector <8 x half> %1745, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1810 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1809, <4 x float> %1808) #8, !dbg !14
  %1811 = shufflevector <8 x half> %1747, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1812 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1811, <4 x float> %1810) #8, !dbg !14
  %1813 = shufflevector <8 x half> %1749, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1814 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1813, <4 x float> %1812) #8, !dbg !14
  %1815 = shufflevector <8 x half> %1751, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1815, <4 x float> %1814) #8, !dbg !14
  %1817 = shufflevector <8 x half> %1753, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1818 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1817, <4 x float> %1816) #8, !dbg !14
  %1819 = extractelement <4 x float> %1818, i64 0, !dbg !14
  %1820 = extractelement <4 x float> %1818, i64 1, !dbg !14
  %1821 = extractelement <4 x float> %1818, i64 2, !dbg !14
  %1822 = extractelement <4 x float> %1818, i64 3, !dbg !14
  %1823 = select i1 %.not, float %1820, float %1819, !dbg !14
  %1824 = bitcast float %1823 to i32, !dbg !14
  %1825 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1824), !dbg !14
  %1826 = bitcast i32 %1825 to float, !dbg !14
  %1827 = select i1 %.not, float %1819, float %1826, !dbg !14
  %1828 = select i1 %.not, float %1826, float %1820, !dbg !14
  %1829 = select i1 %.not, float %1822, float %1821, !dbg !14
  %1830 = bitcast float %1829 to i32, !dbg !14
  %1831 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1830), !dbg !14
  %1832 = bitcast i32 %1831 to float, !dbg !14
  %1833 = select i1 %.not, float %1821, float %1832, !dbg !14
  %1834 = select i1 %.not, float %1832, float %1822, !dbg !14
  %1835 = select i1 %.not1, float %1833, float %1827, !dbg !14
  %1836 = bitcast float %1835 to i32, !dbg !14
  %1837 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1836), !dbg !14
  %1838 = bitcast i32 %1837 to float, !dbg !14
  %1839 = select i1 %.not1, float %1834, float %1828, !dbg !14
  %1840 = bitcast float %1839 to i32, !dbg !14
  %1841 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1840), !dbg !14
  %1842 = bitcast i32 %1841 to float, !dbg !14
  %1843 = insertelement <4 x float> poison, float %1827, i64 0, !dbg !14
  %1844 = insertelement <4 x float> %1843, float %1828, i64 1, !dbg !14
  %1845 = insertelement <4 x float> %1844, float %1838, i64 2, !dbg !14
  %1846 = insertelement <4 x float> %1845, float %1842, i64 3, !dbg !14
  %1847 = insertelement <4 x float> poison, float %1838, i64 0, !dbg !14
  %1848 = insertelement <4 x float> %1847, float %1842, i64 1, !dbg !14
  %1849 = insertelement <4 x float> %1848, float %1833, i64 2, !dbg !14
  %1850 = insertelement <4 x float> %1849, float %1834, i64 3, !dbg !14
  %1851 = select <4 x i1> %66, <4 x float> %1846, <4 x float> %1850, !dbg !14
  %1852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1754, <4 x float> zeroinitializer) #8, !dbg !14
  %1853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1756, <4 x float> %1852) #8, !dbg !14
  %1854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1758, <4 x float> %1853) #8, !dbg !14
  %1855 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1760, <4 x float> %1854) #8, !dbg !14
  %1856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1762, <4 x float> %1855) #8, !dbg !14
  %1857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1764, <4 x float> %1856) #8, !dbg !14
  %1858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1766, <4 x float> %1857) #8, !dbg !14
  %1859 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1768, <4 x float> %1858) #8, !dbg !14
  %1860 = extractelement <4 x float> %1859, i64 0, !dbg !14
  %1861 = extractelement <4 x float> %1859, i64 1, !dbg !14
  %1862 = extractelement <4 x float> %1859, i64 2, !dbg !14
  %1863 = extractelement <4 x float> %1859, i64 3, !dbg !14
  %1864 = select i1 %.not, float %1861, float %1860, !dbg !14
  %1865 = bitcast float %1864 to i32, !dbg !14
  %1866 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1865), !dbg !14
  %1867 = bitcast i32 %1866 to float, !dbg !14
  %1868 = select i1 %.not, float %1860, float %1867, !dbg !14
  %1869 = select i1 %.not, float %1867, float %1861, !dbg !14
  %1870 = select i1 %.not, float %1863, float %1862, !dbg !14
  %1871 = bitcast float %1870 to i32, !dbg !14
  %1872 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1871), !dbg !14
  %1873 = bitcast i32 %1872 to float, !dbg !14
  %1874 = select i1 %.not, float %1862, float %1873, !dbg !14
  %1875 = select i1 %.not, float %1873, float %1863, !dbg !14
  %1876 = select i1 %.not1, float %1874, float %1868, !dbg !14
  %1877 = bitcast float %1876 to i32, !dbg !14
  %1878 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1877), !dbg !14
  %1879 = bitcast i32 %1878 to float, !dbg !14
  %1880 = select i1 %.not1, float %1875, float %1869, !dbg !14
  %1881 = bitcast float %1880 to i32, !dbg !14
  %1882 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1881), !dbg !14
  %1883 = bitcast i32 %1882 to float, !dbg !14
  %1884 = insertelement <4 x float> poison, float %1868, i64 0, !dbg !14
  %1885 = insertelement <4 x float> %1884, float %1869, i64 1, !dbg !14
  %1886 = insertelement <4 x float> %1885, float %1879, i64 2, !dbg !14
  %1887 = insertelement <4 x float> %1886, float %1883, i64 3, !dbg !14
  %1888 = insertelement <4 x float> poison, float %1879, i64 0, !dbg !14
  %1889 = insertelement <4 x float> %1888, float %1883, i64 1, !dbg !14
  %1890 = insertelement <4 x float> %1889, float %1874, i64 2, !dbg !14
  %1891 = insertelement <4 x float> %1890, float %1875, i64 3, !dbg !14
  %1892 = select <4 x i1> %66, <4 x float> %1887, <4 x float> %1891, !dbg !14
  %1893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1803, <4 x float> zeroinitializer) #8, !dbg !14
  %1894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1805, <4 x float> %1893) #8, !dbg !14
  %1895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1807, <4 x float> %1894) #8, !dbg !14
  %1896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1809, <4 x float> %1895) #8, !dbg !14
  %1897 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1811, <4 x float> %1896) #8, !dbg !14
  %1898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1813, <4 x float> %1897) #8, !dbg !14
  %1899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1815, <4 x float> %1898) #8, !dbg !14
  %1900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1817, <4 x float> %1899) #8, !dbg !14
  %1901 = extractelement <4 x float> %1900, i64 0, !dbg !14
  %1902 = extractelement <4 x float> %1900, i64 1, !dbg !14
  %1903 = extractelement <4 x float> %1900, i64 2, !dbg !14
  %1904 = extractelement <4 x float> %1900, i64 3, !dbg !14
  %1905 = select i1 %.not, float %1902, float %1901, !dbg !14
  %1906 = bitcast float %1905 to i32, !dbg !14
  %1907 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1906), !dbg !14
  %1908 = bitcast i32 %1907 to float, !dbg !14
  %1909 = select i1 %.not, float %1901, float %1908, !dbg !14
  %1910 = select i1 %.not, float %1908, float %1902, !dbg !14
  %1911 = select i1 %.not, float %1904, float %1903, !dbg !14
  %1912 = bitcast float %1911 to i32, !dbg !14
  %1913 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1912), !dbg !14
  %1914 = bitcast i32 %1913 to float, !dbg !14
  %1915 = select i1 %.not, float %1903, float %1914, !dbg !14
  %1916 = select i1 %.not, float %1914, float %1904, !dbg !14
  %1917 = select i1 %.not1, float %1915, float %1909, !dbg !14
  %1918 = bitcast float %1917 to i32, !dbg !14
  %1919 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1918), !dbg !14
  %1920 = bitcast i32 %1919 to float, !dbg !14
  %1921 = select i1 %.not1, float %1916, float %1910, !dbg !14
  %1922 = bitcast float %1921 to i32, !dbg !14
  %1923 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1922), !dbg !14
  %1924 = bitcast i32 %1923 to float, !dbg !14
  %1925 = insertelement <4 x float> poison, float %1909, i64 0, !dbg !14
  %1926 = insertelement <4 x float> %1925, float %1910, i64 1, !dbg !14
  %1927 = insertelement <4 x float> %1926, float %1920, i64 2, !dbg !14
  %1928 = insertelement <4 x float> %1927, float %1924, i64 3, !dbg !14
  %1929 = insertelement <4 x float> poison, float %1920, i64 0, !dbg !14
  %1930 = insertelement <4 x float> %1929, float %1924, i64 1, !dbg !14
  %1931 = insertelement <4 x float> %1930, float %1915, i64 2, !dbg !14
  %1932 = insertelement <4 x float> %1931, float %1916, i64 3, !dbg !14
  %1933 = select <4 x i1> %66, <4 x float> %1928, <4 x float> %1932, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %39, label %1934, label %1937, !dbg !32

1934:                                             ; preds = %1713
  %address.thread1033 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local1034 = zext nneg i32 %address.thread1033 to i64, !dbg !10
  %1935 = lshr i64 %address.local1034, 1, !dbg !32
  %address.local1036 = and i64 %1935, 96, !dbg !32
  %address.local1037 = and i64 %address.local1034, 15, !dbg !32
  %1936 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1037, !dbg !32
  %address.local1041 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %1936, i64 %address.local1036, !dbg !32
  store float %806, ptr addrspace(3) %address.local1041, align 4, !dbg !32
  %address.local1043 = getelementptr i8, ptr addrspace(3) %address.local1041, i64 64, !dbg !32
  store float %809, ptr addrspace(3) %address.local1043, align 4, !dbg !32
  br label %1937, !dbg !32

1937:                                             ; preds = %1934, %1713
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %1938 = extractvalue [2 x <8 x float>] %72, 0, !dbg !32
  %1939 = shufflevector <8 x float> %1938, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1940 = fmul <4 x float> %1939, %1274, !dbg !32
  %1941 = fadd <4 x float> %1940, %1802, !dbg !33
  %1942 = shufflevector <8 x float> %1938, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1943 = fmul <4 x float> %1942, %1274, !dbg !32
  %1944 = fadd <4 x float> %1943, %1851, !dbg !33
  %1945 = extractvalue [2 x <8 x float>] %72, 1, !dbg !32
  %1946 = shufflevector <8 x float> %1945, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1947 = fmul <4 x float> %1946, %1283, !dbg !32
  %1948 = fadd <4 x float> %1947, %1892, !dbg !33
  %1949 = shufflevector <8 x float> %1945, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1950 = fmul <4 x float> %1949, %1283, !dbg !32
  %1951 = fadd <4 x float> %1950, %1933, !dbg !33
  %1952 = shufflevector <4 x float> %1276, <4 x float> %1279, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1953 = insertvalue [2 x <8 x float>] poison, <8 x float> %1952, 0, !dbg !17
  %1954 = shufflevector <4 x float> %1285, <4 x float> %1288, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1955 = insertvalue [2 x <8 x float>] %1953, <8 x float> %1954, 1, !dbg !17
  %1956 = insertelement <1 x float> poison, float %1049, i64 0, !dbg !17
  %1957 = insertvalue [2 x <1 x float>] poison, <1 x float> %1956, 0, !dbg !17
  %1958 = insertelement <1 x float> poison, float %1058, i64 0, !dbg !17
  %1959 = insertvalue [2 x <1 x float>] %1957, <1 x float> %1958, 1, !dbg !17
  %1960 = insertvalue [2 x <1 x float>] poison, <1 x float> %772, 0, !dbg !17
  %1961 = insertvalue [2 x <1 x float>] %1960, <1 x float> %785, 1, !dbg !17
  %1962 = shufflevector <4 x float> %1503, <4 x float> %1506, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1963 = insertvalue [2 x <8 x float>] poison, <8 x float> %1962, 0, !dbg !17
  %1964 = shufflevector <4 x float> %1510, <4 x float> %1513, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1965 = insertvalue [2 x <8 x float>] %1963, <8 x float> %1964, 1, !dbg !17
  %1966 = shufflevector <4 x float> %1727, <4 x float> %1730, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1967 = insertvalue [2 x <8 x float>] poison, <8 x float> %1966, 0, !dbg !17
  %1968 = shufflevector <4 x float> %1734, <4 x float> %1737, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1969 = insertvalue [2 x <8 x float>] %1967, <8 x float> %1968, 1, !dbg !17
  %1970 = shufflevector <4 x float> %1941, <4 x float> %1944, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1971 = insertvalue [2 x <8 x float>] poison, <8 x float> %1970, 0, !dbg !17
  %1972 = shufflevector <4 x float> %1948, <4 x float> %1951, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1973 = insertvalue [2 x <8 x float>] %1971, <8 x float> %1972, 1, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1974 = add nuw nsw i64 %78, 1, !dbg !17
  %1975 = icmp slt i64 %1974, %42, !dbg !17
  br i1 %1975, label %71, label %._crit_edge, !dbg !17

._crit_edge:                                      ; preds = %1937, %4
  %.lcssa94 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1955, %1937 ], !dbg !17
  %.lcssa93 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1959, %1937 ], !dbg !17
  %.lcssa92 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1965, %1937 ], !dbg !17
  %.lcssa91 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1969, %1937 ], !dbg !17
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1973, %1937 ], !dbg !17
  %1976 = extractvalue [2 x <1 x float>] %.lcssa93, 0, !dbg !34
  %1977 = shufflevector <1 x float> %1976, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1978 = extractvalue [2 x <1 x float>] %.lcssa93, 1, !dbg !34
  %1979 = shufflevector <1 x float> %1978, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1980 = extractvalue [2 x <8 x float>] %.lcssa94, 1, !dbg !34
  %1981 = shufflevector <8 x float> %1980, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1979, !dbg !34
  %normalized = fmul <4 x float> %1981, %row.reciprocal, !dbg !34
  %1982 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !35
  %1983 = extractelement <4 x half> %1982, i64 3, !dbg !35
  %1984 = extractelement <4 x half> %1982, i64 2, !dbg !35
  %1985 = extractelement <4 x half> %1982, i64 1, !dbg !35
  %1986 = extractelement <4 x half> %1982, i64 0, !dbg !35
  %1987 = shufflevector <8 x float> %1980, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized68 = fmul <4 x float> %1987, %row.reciprocal, !dbg !34
  %1988 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !35
  %1989 = extractelement <4 x half> %1988, i64 3, !dbg !35
  %1990 = extractelement <4 x half> %1988, i64 2, !dbg !35
  %1991 = extractelement <4 x half> %1988, i64 1, !dbg !35
  %1992 = extractelement <4 x half> %1988, i64 0, !dbg !35
  %1993 = extractvalue [2 x <8 x float>] %.lcssa94, 0, !dbg !34
  %1994 = shufflevector <8 x float> %1993, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1977, !dbg !34
  %normalized70 = fmul <4 x float> %1994, %row.reciprocal69, !dbg !34
  %1995 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !35
  %1996 = extractelement <4 x half> %1995, i64 3, !dbg !35
  %1997 = extractelement <4 x half> %1995, i64 2, !dbg !35
  %1998 = extractelement <4 x half> %1995, i64 1, !dbg !35
  %1999 = extractelement <4 x half> %1995, i64 0, !dbg !35
  %2000 = shufflevector <8 x float> %1993, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized71 = fmul <4 x float> %2000, %row.reciprocal69, !dbg !34
  %2001 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !35
  %2002 = extractelement <4 x half> %2001, i64 3, !dbg !35
  %2003 = extractelement <4 x half> %2001, i64 2, !dbg !35
  %2004 = extractelement <4 x half> %2001, i64 1, !dbg !35
  %2005 = extractelement <4 x half> %2001, i64 0, !dbg !35
  %address.thread1058 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !36, !range !11, !frisk.register_lifetimes !5
  %address.local1059 = zext nneg i32 %address.thread1058 to i64, !dbg !10
  %2006 = lshr i64 %address.local1059, 2, !dbg !18
  %address.local1062 = and i64 %2006, 48, !dbg !18
  %2007 = and i64 %address.local1059, 240, !dbg !36
  %address.local1063 = add i64 %28, %address.local1059, !dbg !36
  %address.local1066 = sub i64 %address.local1063, %2007, !dbg !36
  %address.local1067 = add i64 %address.local1066, %address.local1062, !dbg !36
  %2008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %30, !dbg !36
  %address.local1070.idx = shl nuw nsw i64 %address.local1067, 8, !dbg !36
  %address.local1070 = getelementptr inbounds nuw i8, ptr addrspace(1) %2008, i64 %address.local1070.idx, !dbg !36
  %address.local1073 = and i64 %2006, 12, !dbg !10
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1073, !dbg !36
  store half %2005, ptr addrspace(1) %address.local1074, align 2, !dbg !36
  %address.local1076 = or disjoint i64 %address.local1073, 1, !dbg !18
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1076, !dbg !36
  store half %2004, ptr addrspace(1) %address.local1077, align 2, !dbg !36
  %address.local1079 = or disjoint i64 %address.local1073, 2, !dbg !18
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1079, !dbg !36
  store half %2003, ptr addrspace(1) %address.local1080, align 2, !dbg !36
  %address.local1082 = or disjoint i64 %address.local1073, 3, !dbg !18
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1082, !dbg !36
  store half %2002, ptr addrspace(1) %address.local1083, align 2, !dbg !36
  %address.local1085 = or disjoint i64 %address.local1073, 16, !dbg !18
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1085, !dbg !36
  store half %1999, ptr addrspace(1) %address.local1086, align 2, !dbg !36
  %address.local1088 = or disjoint i64 %address.local1073, 17, !dbg !18
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1088, !dbg !36
  store half %1998, ptr addrspace(1) %address.local1089, align 2, !dbg !36
  %address.local1091 = or disjoint i64 %address.local1073, 18, !dbg !18
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1091, !dbg !36
  store half %1997, ptr addrspace(1) %address.local1092, align 2, !dbg !36
  %address.local1094 = or disjoint i64 %address.local1073, 19, !dbg !18
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1094, !dbg !36
  store half %1996, ptr addrspace(1) %address.local1095, align 2, !dbg !36
  %address.local1099 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1070, i64 16384, !dbg !36
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1073, !dbg !36
  store half %1992, ptr addrspace(1) %address.local1100, align 2, !dbg !36
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1076, !dbg !36
  store half %1991, ptr addrspace(1) %address.local1102, align 2, !dbg !36
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1079, !dbg !36
  store half %1990, ptr addrspace(1) %address.local1104, align 2, !dbg !36
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1082, !dbg !36
  store half %1989, ptr addrspace(1) %address.local1106, align 2, !dbg !36
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1085, !dbg !36
  store half %1986, ptr addrspace(1) %address.local1108, align 2, !dbg !36
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1088, !dbg !36
  store half %1985, ptr addrspace(1) %address.local1110, align 2, !dbg !36
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1091, !dbg !36
  store half %1984, ptr addrspace(1) %address.local1112, align 2, !dbg !36
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1094, !dbg !36
  store half %1983, ptr addrspace(1) %address.local1114, align 2, !dbg !36
  %2009 = extractvalue [2 x <8 x float>] %.lcssa92, 1, !dbg !34
  %2010 = shufflevector <8 x float> %2009, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized73 = fmul <4 x float> %row.reciprocal, %2010, !dbg !34
  %2011 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !35
  %2012 = extractelement <4 x half> %2011, i64 3, !dbg !35
  %2013 = extractelement <4 x half> %2011, i64 2, !dbg !35
  %2014 = extractelement <4 x half> %2011, i64 1, !dbg !35
  %2015 = extractelement <4 x half> %2011, i64 0, !dbg !35
  %2016 = shufflevector <8 x float> %2009, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized74 = fmul <4 x float> %row.reciprocal, %2016, !dbg !34
  %2017 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !35
  %2018 = extractelement <4 x half> %2017, i64 3, !dbg !35
  %2019 = extractelement <4 x half> %2017, i64 2, !dbg !35
  %2020 = extractelement <4 x half> %2017, i64 1, !dbg !35
  %2021 = extractelement <4 x half> %2017, i64 0, !dbg !35
  %2022 = extractvalue [2 x <8 x float>] %.lcssa92, 0, !dbg !34
  %2023 = shufflevector <8 x float> %2022, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized76 = fmul <4 x float> %row.reciprocal69, %2023, !dbg !34
  %2024 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !35
  %2025 = extractelement <4 x half> %2024, i64 3, !dbg !35
  %2026 = extractelement <4 x half> %2024, i64 2, !dbg !35
  %2027 = extractelement <4 x half> %2024, i64 1, !dbg !35
  %2028 = extractelement <4 x half> %2024, i64 0, !dbg !35
  %2029 = shufflevector <8 x float> %2022, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized77 = fmul <4 x float> %row.reciprocal69, %2029, !dbg !34
  %2030 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !35
  %2031 = extractelement <4 x half> %2030, i64 3, !dbg !35
  %2032 = extractelement <4 x half> %2030, i64 2, !dbg !35
  %2033 = extractelement <4 x half> %2030, i64 1, !dbg !35
  %2034 = extractelement <4 x half> %2030, i64 0, !dbg !35
  %address.local1131 = or disjoint i64 %address.local1073, 32, !dbg !18
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1131, !dbg !36
  store half %2034, ptr addrspace(1) %address.local1132, align 2, !dbg !36
  %address.local1134 = or disjoint i64 %address.local1073, 33, !dbg !18
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1134, !dbg !36
  store half %2033, ptr addrspace(1) %address.local1135, align 2, !dbg !36
  %address.local1137 = or disjoint i64 %address.local1073, 34, !dbg !18
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1137, !dbg !36
  store half %2032, ptr addrspace(1) %address.local1138, align 2, !dbg !36
  %address.local1140 = or disjoint i64 %address.local1073, 35, !dbg !18
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1140, !dbg !36
  store half %2031, ptr addrspace(1) %address.local1141, align 2, !dbg !36
  %address.local1143 = or disjoint i64 %address.local1073, 48, !dbg !18
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1143, !dbg !36
  store half %2028, ptr addrspace(1) %address.local1144, align 2, !dbg !36
  %address.local1146 = or disjoint i64 %address.local1073, 49, !dbg !18
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1146, !dbg !36
  store half %2027, ptr addrspace(1) %address.local1147, align 2, !dbg !36
  %address.local1149 = or disjoint i64 %address.local1073, 50, !dbg !18
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1149, !dbg !36
  store half %2026, ptr addrspace(1) %address.local1150, align 2, !dbg !36
  %address.local1152 = or i64 %2006, 51, !dbg !18
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1152, !dbg !36
  store half %2025, ptr addrspace(1) %address.local1153, align 2, !dbg !36
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1131, !dbg !36
  store half %2021, ptr addrspace(1) %address.local1158, align 2, !dbg !36
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1134, !dbg !36
  store half %2020, ptr addrspace(1) %address.local1160, align 2, !dbg !36
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1137, !dbg !36
  store half %2019, ptr addrspace(1) %address.local1162, align 2, !dbg !36
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1140, !dbg !36
  store half %2018, ptr addrspace(1) %address.local1164, align 2, !dbg !36
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1143, !dbg !36
  store half %2015, ptr addrspace(1) %address.local1166, align 2, !dbg !36
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1146, !dbg !36
  store half %2014, ptr addrspace(1) %address.local1168, align 2, !dbg !36
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1149, !dbg !36
  store half %2013, ptr addrspace(1) %address.local1170, align 2, !dbg !36
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1152, !dbg !36
  store half %2012, ptr addrspace(1) %address.local1172, align 2, !dbg !36
  %2035 = extractvalue [2 x <8 x float>] %.lcssa91, 1, !dbg !34
  %2036 = shufflevector <8 x float> %2035, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized79 = fmul <4 x float> %row.reciprocal, %2036, !dbg !34
  %2037 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !35
  %2038 = extractelement <4 x half> %2037, i64 3, !dbg !35
  %2039 = extractelement <4 x half> %2037, i64 2, !dbg !35
  %2040 = extractelement <4 x half> %2037, i64 1, !dbg !35
  %2041 = extractelement <4 x half> %2037, i64 0, !dbg !35
  %2042 = shufflevector <8 x float> %2035, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized80 = fmul <4 x float> %row.reciprocal, %2042, !dbg !34
  %2043 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !35
  %2044 = extractelement <4 x half> %2043, i64 3, !dbg !35
  %2045 = extractelement <4 x half> %2043, i64 2, !dbg !35
  %2046 = extractelement <4 x half> %2043, i64 1, !dbg !35
  %2047 = extractelement <4 x half> %2043, i64 0, !dbg !35
  %2048 = extractvalue [2 x <8 x float>] %.lcssa91, 0, !dbg !34
  %2049 = shufflevector <8 x float> %2048, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized82 = fmul <4 x float> %row.reciprocal69, %2049, !dbg !34
  %2050 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !35
  %2051 = extractelement <4 x half> %2050, i64 3, !dbg !35
  %2052 = extractelement <4 x half> %2050, i64 2, !dbg !35
  %2053 = extractelement <4 x half> %2050, i64 1, !dbg !35
  %2054 = extractelement <4 x half> %2050, i64 0, !dbg !35
  %2055 = shufflevector <8 x float> %2048, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized83 = fmul <4 x float> %row.reciprocal69, %2055, !dbg !34
  %2056 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !35
  %2057 = extractelement <4 x half> %2056, i64 3, !dbg !35
  %2058 = extractelement <4 x half> %2056, i64 2, !dbg !35
  %2059 = extractelement <4 x half> %2056, i64 1, !dbg !35
  %2060 = extractelement <4 x half> %2056, i64 0, !dbg !35
  %address.local1189 = or disjoint i64 %address.local1073, 64, !dbg !18
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1189, !dbg !36
  store half %2060, ptr addrspace(1) %address.local1190, align 2, !dbg !36
  %address.local1192 = or disjoint i64 %address.local1073, 65, !dbg !18
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1192, !dbg !36
  store half %2059, ptr addrspace(1) %address.local1193, align 2, !dbg !36
  %address.local1195 = or disjoint i64 %address.local1073, 66, !dbg !18
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1195, !dbg !36
  store half %2058, ptr addrspace(1) %address.local1196, align 2, !dbg !36
  %address.local1198 = or disjoint i64 %address.local1073, 67, !dbg !18
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1198, !dbg !36
  store half %2057, ptr addrspace(1) %address.local1199, align 2, !dbg !36
  %address.local1201 = or disjoint i64 %address.local1073, 80, !dbg !18
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1201, !dbg !36
  store half %2054, ptr addrspace(1) %address.local1202, align 2, !dbg !36
  %address.local1204 = or disjoint i64 %address.local1073, 81, !dbg !18
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1204, !dbg !36
  store half %2053, ptr addrspace(1) %address.local1205, align 2, !dbg !36
  %address.local1207 = or disjoint i64 %address.local1073, 82, !dbg !18
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1207, !dbg !36
  store half %2052, ptr addrspace(1) %address.local1208, align 2, !dbg !36
  %address.local1210 = or disjoint i64 %address.local1073, 83, !dbg !18
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1210, !dbg !36
  store half %2051, ptr addrspace(1) %address.local1211, align 2, !dbg !36
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1189, !dbg !36
  store half %2047, ptr addrspace(1) %address.local1216, align 2, !dbg !36
  %address.local1218 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1192, !dbg !36
  store half %2046, ptr addrspace(1) %address.local1218, align 2, !dbg !36
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1195, !dbg !36
  store half %2045, ptr addrspace(1) %address.local1220, align 2, !dbg !36
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1198, !dbg !36
  store half %2044, ptr addrspace(1) %address.local1222, align 2, !dbg !36
  %address.local1224 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1201, !dbg !36
  store half %2041, ptr addrspace(1) %address.local1224, align 2, !dbg !36
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1204, !dbg !36
  store half %2040, ptr addrspace(1) %address.local1226, align 2, !dbg !36
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1207, !dbg !36
  store half %2039, ptr addrspace(1) %address.local1228, align 2, !dbg !36
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1210, !dbg !36
  store half %2038, ptr addrspace(1) %address.local1230, align 2, !dbg !36
  %2061 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !34
  %2062 = shufflevector <8 x float> %2061, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized85 = fmul <4 x float> %row.reciprocal, %2062, !dbg !34
  %2063 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !35
  %2064 = extractelement <4 x half> %2063, i64 3, !dbg !35
  %2065 = extractelement <4 x half> %2063, i64 2, !dbg !35
  %2066 = extractelement <4 x half> %2063, i64 1, !dbg !35
  %2067 = extractelement <4 x half> %2063, i64 0, !dbg !35
  %2068 = shufflevector <8 x float> %2061, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized86 = fmul <4 x float> %row.reciprocal, %2068, !dbg !34
  %2069 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !35
  %2070 = extractelement <4 x half> %2069, i64 3, !dbg !35
  %2071 = extractelement <4 x half> %2069, i64 2, !dbg !35
  %2072 = extractelement <4 x half> %2069, i64 1, !dbg !35
  %2073 = extractelement <4 x half> %2069, i64 0, !dbg !35
  %2074 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !34
  %2075 = shufflevector <8 x float> %2074, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized88 = fmul <4 x float> %row.reciprocal69, %2075, !dbg !34
  %2076 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !35
  %2077 = extractelement <4 x half> %2076, i64 3, !dbg !35
  %2078 = extractelement <4 x half> %2076, i64 2, !dbg !35
  %2079 = extractelement <4 x half> %2076, i64 1, !dbg !35
  %2080 = extractelement <4 x half> %2076, i64 0, !dbg !35
  %2081 = shufflevector <8 x float> %2074, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized89 = fmul <4 x float> %row.reciprocal69, %2081, !dbg !34
  %2082 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !35
  %2083 = extractelement <4 x half> %2082, i64 3, !dbg !35
  %2084 = extractelement <4 x half> %2082, i64 2, !dbg !35
  %2085 = extractelement <4 x half> %2082, i64 1, !dbg !35
  %2086 = extractelement <4 x half> %2082, i64 0, !dbg !35
  %address.local1247 = or disjoint i64 %address.local1073, 96, !dbg !18
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1247, !dbg !36
  store half %2086, ptr addrspace(1) %address.local1248, align 2, !dbg !36
  %address.local1250 = or disjoint i64 %address.local1073, 97, !dbg !18
  %address.local1251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1250, !dbg !36
  store half %2085, ptr addrspace(1) %address.local1251, align 2, !dbg !36
  %address.local1253 = or disjoint i64 %address.local1073, 98, !dbg !18
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1253, !dbg !36
  store half %2084, ptr addrspace(1) %address.local1254, align 2, !dbg !36
  %address.local1256 = or disjoint i64 %address.local1073, 99, !dbg !18
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1256, !dbg !36
  store half %2083, ptr addrspace(1) %address.local1257, align 2, !dbg !36
  %address.local1259 = or disjoint i64 %address.local1073, 112, !dbg !18
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1259, !dbg !36
  store half %2080, ptr addrspace(1) %address.local1260, align 2, !dbg !36
  %address.local1262 = or disjoint i64 %address.local1073, 113, !dbg !18
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1262, !dbg !36
  store half %2079, ptr addrspace(1) %address.local1263, align 2, !dbg !36
  %address.local1265 = or disjoint i64 %address.local1073, 114, !dbg !18
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1265, !dbg !36
  store half %2078, ptr addrspace(1) %address.local1266, align 2, !dbg !36
  %address.local1268 = or i64 %2006, 115, !dbg !18
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1268, !dbg !36
  store half %2077, ptr addrspace(1) %address.local1269, align 2, !dbg !36
  %address.local1274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1247, !dbg !36
  store half %2073, ptr addrspace(1) %address.local1274, align 2, !dbg !36
  %address.local1276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1250, !dbg !36
  store half %2072, ptr addrspace(1) %address.local1276, align 2, !dbg !36
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1253, !dbg !36
  store half %2071, ptr addrspace(1) %address.local1278, align 2, !dbg !36
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1256, !dbg !36
  store half %2070, ptr addrspace(1) %address.local1280, align 2, !dbg !36
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1259, !dbg !36
  store half %2067, ptr addrspace(1) %address.local1282, align 2, !dbg !36
  %address.local1284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1262, !dbg !36
  store half %2066, ptr addrspace(1) %address.local1284, align 2, !dbg !36
  %address.local1286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1265, !dbg !36
  store half %2065, ptr addrspace(1) %address.local1286, align 2, !dbg !36
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1268, !dbg !36
  store half %2064, ptr addrspace(1) %address.local1288, align 2, !dbg !36
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

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

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/run-20261027-121856-x88fub13/grid-000001")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 48, column: 17, scope: !3)
!9 = !DILocation(line: 70, column: 22, scope: !3)
!10 = !DILocation(line: 72, column: 24, scope: !3)
!11 = !{i32 0, i32 256}
!12 = !DILocation(line: 77, column: 19, scope: !3)
!13 = !DILocation(line: 93, column: 22, scope: !3)
!14 = !DILocation(line: 102, column: 23, scope: !3)
!15 = !DILocation(line: 0, scope: !3)
!16 = !DILocation(line: 71, column: 22, scope: !3)
!17 = !DILocation(line: 60, column: 24, scope: !3)
!18 = !DILocation(line: 63, column: 17, scope: !3)
!19 = !DILocation(line: 80, column: 23, scope: !3)
!20 = !DILocation(line: 73, column: 26, scope: !3)
!21 = !DILocation(line: 74, column: 23, scope: !3)
!22 = !DILocation(line: 81, column: 26, scope: !3)
!23 = !DILocation(line: 88, column: 21, scope: !3)
!24 = !DILocation(line: 89, column: 20, scope: !3)
!25 = !DILocation(line: 90, column: 22, scope: !3)
!26 = !DILocation(line: 91, column: 23, scope: !3)
!27 = !DILocation(line: 92, column: 20, scope: !3)
!28 = !DILocation(line: 101, column: 20, scope: !3)
!29 = !DILocation(line: 94, column: 22, scope: !3)
!30 = !DILocation(line: 96, column: 16, scope: !3)
!31 = !DILocation(line: 95, column: 20, scope: !3)
!32 = !DILocation(line: 103, column: 22, scope: !3)
!33 = !DILocation(line: 104, column: 20, scope: !3)
!34 = !DILocation(line: 107, column: 23, scope: !3)
!35 = !DILocation(line: 108, column: 19, scope: !3)
!36 = !DILocation(line: 109, column: 9, scope: !3)
