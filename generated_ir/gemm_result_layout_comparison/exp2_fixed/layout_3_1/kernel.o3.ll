; ModuleID = 'generated_ir/gemm_result_layout_comparison/exp2_fixed/layout_3_1/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000001/input.mlir"

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
  %dma.buffer.rsrc1822 = shufflevector <4 x i32> %52, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %53 = ptrtoint ptr addrspace(1) %1 to i64
  %54 = insertelement <2 x i64> poison, i64 %53, i64 0
  %55 = bitcast <2 x i64> %54 to <4 x i32>
  %dma.buffer.rsrc1816 = shufflevector <4 x i32> %55, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
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
  %compare.high1427 = trunc nuw i64 %61 to i32
  %compare.low.base1431 = trunc i64 %60 to i32
  %62 = ptrtoint ptr addrspace(1) %2 to i64
  %63 = insertelement <2 x i64> poison, i64 %62, i64 0
  %64 = bitcast <2 x i64> %63 to <4 x i32>
  %dma.buffer.rsrc1774 = shufflevector <4 x i32> %64, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %65 = insertelement <4 x i1> poison, i1 %.not1, i64 0, !dbg !14
  %66 = shufflevector <4 x i1> %65, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !14
  %67 = insertelement <32 x i32> poison, i32 %compare.low.base1431, i64 0, !dbg !19
  %68 = shufflevector <32 x i32> %67, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !19
  %69 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !19
  %70 = shufflevector <32 x i32> %69, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !19
  br label %71, !dbg !17

71:                                               ; preds = %.lr.ph, %1947
  %72 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1989, %1947 ]
  %73 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1985, %1947 ]
  %74 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1981, %1947 ]
  %75 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1977, %1947 ]
  %76 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1975, %1947 ]
  %77 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1971, %1947 ]
  %78 = phi i64 [ 0, %.lr.ph ], [ %1990, %1947 ]
  %79 = shl nuw nsw i64 %78, 7, !dbg !17
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !9, !range !11, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !10
  %address.local258 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local259 = and i64 %address.local258, 1536, !dbg !9
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local259, !dbg !9
  %address.local263 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local264 = and i64 %address.local263, 16256, !dbg !9
  %address.local265 = lshr i64 %address.local, 4, !dbg !9
  %address.local266 = and i64 %address.local265, 8, !dbg !9
  %address.local267 = or disjoint i64 %address.local264, %address.local266, !dbg !9
  %80 = or disjoint i64 %address.local267, %30, !dbg !9
  %81 = shl nuw nsw i64 %80, 1, !dbg !9
  %82 = add i64 %81, %.idx, !dbg !9
  %dma.buffer.offset1821 = trunc i64 %82 to i32, !dbg !9
  %address.local269 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local270 = shl nuw nsw i64 %address.local269, 3, !dbg !9
  %address.local271 = and i64 %address.local270, 3584, !dbg !9
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local271, !dbg !9
  %address.local277 = lshr i64 %address.local269, 4, !dbg !9
  %address.local278 = and i64 %address.local277, 24, !dbg !9
  %address.local279 = or disjoint i64 %address.local278, %address.local264, !dbg !9
  %83 = or disjoint i64 %address.local279, %30, !dbg !9
  %84 = shl nuw nsw i64 %83, 1, !dbg !9
  %85 = add i64 %84, %.idx, !dbg !9
  %dma.buffer.offset1818 = trunc i64 %85 to i32, !dbg !9
  %86 = lshr i64 %78, 12, !dbg !16
  %87 = add nuw nsw i64 %86, %15, !dbg !16
  %88 = lshr i64 %78, 5, !dbg !16
  %89 = and i64 %88, 127, !dbg !16
  %90 = shl i64 %87, 19, !dbg !16
  %91 = and i64 %90, 16252928, !dbg !16
  %92 = shl i64 %88, 19, !dbg !16
  %93 = shl nuw nsw i64 %78, 14, !dbg !16
  %94 = sub i64 %93, %92, !dbg !16
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local259, !dbg !16
  %address.local286 = and i64 %address.local, 123, !dbg !16
  %address.local287 = lshr i64 %address.local, 2, !dbg !16
  %address.local288 = and i64 %address.local287, 4, !dbg !16
  %address.local289 = add nuw nsw i64 %address.local288, %address.local, !dbg !16
  %address.local290 = and i64 %address.local289, 4, !dbg !16
  %address.local291 = or disjoint i64 %address.local290, %address.local286, !dbg !16
  %address.local292 = shl nuw nsw i64 %address.local291, 7, !dbg !16
  %address.local293 = or disjoint i64 %address.local292, %address.local266, !dbg !16
  %.idx36 = or disjoint i64 %91, %89, !dbg !16
  %95 = add i64 %.idx36, %94, !dbg !16
  %96 = add i64 %address.local293, %95, !dbg !16
  %.tr = trunc i64 %96 to i32, !dbg !16
  %dma.buffer.offset1815 = shl i32 %.tr, 1, !dbg !16
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local271, !dbg !16
  %97 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local260, i64 0, !dbg !9
  %98 = insertelement <4 x ptr addrspace(3)> %97, ptr addrspace(3) %address.local272, i64 1, !dbg !9
  %99 = insertelement <4 x ptr addrspace(3)> %98, ptr addrspace(3) %address.local281, i64 2, !dbg !9
  %100 = insertelement <4 x ptr addrspace(3)> %99, ptr addrspace(3) %address.local295, i64 3, !dbg !9
  %101 = ptrtoint <4 x ptr addrspace(3)> %100 to <4 x i64>, !dbg !9
  %102 = trunc <4 x i64> %101 to <4 x i32>, !dbg !9
  %103 = extractelement <4 x i32> %102, i64 0, !dbg !9
  %104 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %103), !dbg !9
  %dma.buffer.lds1823 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %104), !dbg !9
  %105 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1821, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %106 = extractelement <4 x i32> %102, i64 1, !dbg !9
  %107 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %106), !dbg !9
  %dma.buffer.lds1820 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %107), !dbg !9
  %108 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1818, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %109 = extractelement <4 x i32> %102, i64 2, !dbg !16
  %110 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %109), !dbg !16
  %dma.buffer.lds1817 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %110), !dbg !16
  %111 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1815, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %112 = extractelement <4 x i32> %102, i64 3, !dbg !16
  %113 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %112), !dbg !16
  %address.local307 = or disjoint i64 %address.local292, %address.local278, !dbg !16
  %114 = add i64 %address.local307, %95, !dbg !16
  %.tr11 = trunc i64 %114 to i32, !dbg !16
  %dma.buffer.offset1812 = shl i32 %.tr11, 1, !dbg !16
  %dma.buffer.lds1814 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %113), !dbg !16
  %115 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1812, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local309 = lshr i32 %address.thread, 2, !dbg !10
  %address.local310 = and i32 %address.local309, 4, !dbg !10
  %address.local311 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local312 = and i32 %address.local311, 384, !dbg !10
  %address.local314 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local315 = and i32 %address.local314, 120, !dbg !10
  %address.local317 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local318 = and i32 %address.local317, 1024, !dbg !10
  %116 = or disjoint i32 %address.local310, %address.local312, !dbg !10
  %address.local316 = or disjoint i32 %116, %address.local315, !dbg !10
  %address.local319 = or disjoint i32 %address.local316, %address.local318, !dbg !10
  %address.local320 = zext nneg i32 %address.local319 to i64, !dbg !10
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local320, !dbg !10
  %117 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !10
  %118 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !10
  %119 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !10
  %120 = fmul <4 x half> %117, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !10
  %121 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !10
  %122 = fmul <4 x half> %118, splat (half 1.275630e-01), !dbg !10
  %123 = fmul <4 x half> %119, splat (half 1.275630e-01), !dbg !10
  %124 = fmul <4 x half> %121, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local332 = or disjoint i32 %address.local310, %address.local315, !dbg !20
  %address.local335 = or disjoint i32 %address.local332, %address.local318, !dbg !20
  %address.local336 = zext nneg i32 %address.local335 to i64, !dbg !20
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local336, !dbg !20
  %125 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !20
  %126 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %125, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1561 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !20
  %127 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !20
  %address.local352 = and i32 %address.local314, 88, !dbg !20
  %address.local353 = or disjoint i32 %address.local310, %address.local352, !dbg !20
  %address.local355 = and i32 %address.local314, 32, !dbg !20
  %address.local356 = sub nsw i32 %address.local353, %address.local355, !dbg !20
  %address.local359 = add nsw i32 %address.local356, %address.local318, !dbg !20
  %address.local360 = sext i32 %address.local359 to i64, !dbg !20
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local360, !dbg !20
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local361, i64 320, !dbg !20
  %128 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !20
  %129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %127, <4 x float> %126) #8, !dbg !20
  %130 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %128, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1562 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !20
  %131 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !20
  %lds.const.gep1563 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !20
  %132 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !20
  %133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %131, <4 x float> %130) #8, !dbg !20
  %134 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %132, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1564 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !20
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !20
  %lds.const.gep1565 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !20
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !20
  %137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %135, <4 x float> %134) #8, !dbg !20
  %138 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %136, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1566 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !20
  %139 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !20
  %lds.const.gep1567 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !20
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !20
  %141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %139, <4 x float> %138) #8, !dbg !20
  %142 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %140, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1568 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !20
  %143 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !20
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !20
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !20
  %145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %143, <4 x float> %142) #8, !dbg !20
  %146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %144, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !20
  %147 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !20
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !20
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !20
  %149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %147, <4 x float> %146) #8, !dbg !20
  %150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %148, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !20
  %151 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !20
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !20
  %152 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !20
  %153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %151, <4 x float> %150) #8, !dbg !20
  %154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %152, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !20
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !20
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
  %address.local401.idx = shl nuw nsw i64 %address.local267, 1, !dbg !9
  %173 = or disjoint i64 %address.local401.idx, %56, !dbg !9
  %174 = trunc i64 %173 to i32, !dbg !9
  %dma.buffer.offset1809 = or disjoint i32 %174, 64, !dbg !9
  %175 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1809, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %address.local402.idx = shl nuw nsw i64 %address.local279, 1, !dbg !9
  %.reass = or disjoint i64 %address.local402.idx, %invariant.op101
  %dma.buffer.offset1806 = trunc i64 %.reass to i32, !dbg !9
  %176 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1806, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
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
  %dma.buffer.offset1797 = or disjoint i32 %174, 128, !dbg !9
  %.reass103 = or disjoint i64 %address.local402.idx, %invariant.op102
  %dma.buffer.offset1794 = trunc i64 %.reass103 to i32, !dbg !9
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
  %201 = add i64 %address.local293, %200, !dbg !16
  %.tr19 = trunc i64 %201 to i32, !dbg !16
  %202 = shl i32 %.tr19, 1, !dbg !16
  %dma.buffer.offset1803 = add i32 %202, 64, !dbg !16
  %203 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1803, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %204 = add i64 %address.local307, %200, !dbg !16
  %.tr25 = trunc i64 %204 to i32, !dbg !16
  %205 = shl i32 %.tr25, 1, !dbg !16
  %dma.buffer.offset1800 = add i32 %205, 64, !dbg !16
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1800, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %207 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !10
  %208 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %209 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !10
  %210 = fmul <4 x half> %207, splat (half 1.275630e-01), !dbg !10
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !10
  %212 = fmul <4 x half> %208, splat (half 1.275630e-01), !dbg !10
  %213 = fmul <4 x half> %209, splat (half 1.275630e-01), !dbg !10
  %214 = fmul <4 x half> %211, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %215 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !20
  %216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %215, <4 x float> zeroinitializer) #8, !dbg !20
  %217 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !20
  %218 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !20
  %219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %217, <4 x float> %216) #8, !dbg !20
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %218, <4 x float> zeroinitializer) #8, !dbg !20
  %221 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !20
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !20
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %221, <4 x float> %220) #8, !dbg !20
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %222, <4 x float> zeroinitializer) #8, !dbg !20
  %225 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !20
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !20
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %225, <4 x float> %224) #8, !dbg !20
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %226, <4 x float> zeroinitializer) #8, !dbg !20
  %229 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !20
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !20
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %229, <4 x float> %228) #8, !dbg !20
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %230, <4 x float> zeroinitializer) #8, !dbg !20
  %233 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !20
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !20
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %233, <4 x float> %232) #8, !dbg !20
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %234, <4 x float> zeroinitializer) #8, !dbg !20
  %237 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !20
  %238 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !20
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %237, <4 x float> %236) #8, !dbg !20
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %238, <4 x float> zeroinitializer) #8, !dbg !20
  %241 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !20
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !20
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %241, <4 x float> %240) #8, !dbg !20
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %242, <4 x float> zeroinitializer) #8, !dbg !20
  %245 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !20
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
  %263 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1797, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %264 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1794, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
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
  %274 = add i64 %address.local293, %273, !dbg !16
  %.tr33 = trunc i64 %274 to i32, !dbg !16
  %275 = shl i32 %.tr33, 1, !dbg !16
  %dma.buffer.offset1791 = add i32 %275, 128, !dbg !16
  %276 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1791, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %277 = add i64 %address.local307, %273, !dbg !16
  %.tr40 = trunc i64 %277 to i32, !dbg !16
  %278 = shl i32 %.tr40, 1, !dbg !16
  %dma.buffer.offset1788 = add i32 %278, 128, !dbg !16
  %279 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1788, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %280 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !10
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !10
  %283 = fmul <4 x half> %280, splat (half 1.275630e-01), !dbg !10
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !10
  %285 = fmul <4 x half> %281, splat (half 1.275630e-01), !dbg !10
  %286 = fmul <4 x half> %282, splat (half 1.275630e-01), !dbg !10
  %287 = fmul <4 x half> %284, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %288 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !20
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %288, <4 x float> zeroinitializer) #8, !dbg !20
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !20
  %291 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !20
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %290, <4 x float> %289) #8, !dbg !20
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %291, <4 x float> zeroinitializer) #8, !dbg !20
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !20
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !20
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %294, <4 x float> %293) #8, !dbg !20
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %295, <4 x float> zeroinitializer) #8, !dbg !20
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !20
  %299 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !20
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %298, <4 x float> %297) #8, !dbg !20
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %299, <4 x float> zeroinitializer) #8, !dbg !20
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !20
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !20
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %302, <4 x float> %301) #8, !dbg !20
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %303, <4 x float> zeroinitializer) #8, !dbg !20
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !20
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !20
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %306, <4 x float> %305) #8, !dbg !20
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %307, <4 x float> zeroinitializer) #8, !dbg !20
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !20
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !20
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %310, <4 x float> %309) #8, !dbg !20
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %311, <4 x float> zeroinitializer) #8, !dbg !20
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !20
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !20
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %314, <4 x float> %313) #8, !dbg !20
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %315, <4 x float> zeroinitializer) #8, !dbg !20
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !20
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
  %dma.buffer.offset1785 = or disjoint i32 %174, 192, !dbg !9
  %336 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1785, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %337 = or disjoint i64 %address.local402.idx, %56, !dbg !9
  %338 = trunc i64 %337 to i32, !dbg !9
  %dma.buffer.offset1782 = or disjoint i32 %338, 192, !dbg !9
  %339 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1782, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
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
  %355 = add i64 %address.local293, %354, !dbg !16
  %.tr48 = trunc i64 %355 to i32, !dbg !16
  %356 = shl i32 %.tr48, 1, !dbg !16
  %dma.buffer.offset1779 = add i32 %356, 192, !dbg !16
  %357 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1779, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %358 = add i64 %address.local307, %354, !dbg !16
  %.tr54 = trunc i64 %358 to i32, !dbg !16
  %359 = shl i32 %.tr54, 1, !dbg !16
  %dma.buffer.offset1776 = add i32 %359, 192, !dbg !16
  %360 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1776, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %361 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !10
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !10
  %364 = fmul <4 x half> %361, splat (half 1.275630e-01), !dbg !10
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !10
  %366 = fmul <4 x half> %362, splat (half 1.275630e-01), !dbg !10
  %367 = fmul <4 x half> %363, splat (half 1.275630e-01), !dbg !10
  %368 = fmul <4 x half> %365, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %369 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !20
  %370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %369, <4 x float> zeroinitializer) #8, !dbg !20
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !20
  %372 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !20
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %371, <4 x float> %370) #8, !dbg !20
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %372, <4 x float> zeroinitializer) #8, !dbg !20
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !20
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !20
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %375, <4 x float> %374) #8, !dbg !20
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %376, <4 x float> zeroinitializer) #8, !dbg !20
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !20
  %380 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !20
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %379, <4 x float> %378) #8, !dbg !20
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %380, <4 x float> zeroinitializer) #8, !dbg !20
  %383 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !20
  %384 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !20
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %383, <4 x float> %382) #8, !dbg !20
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %384, <4 x float> zeroinitializer) #8, !dbg !20
  %387 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !20
  %388 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !20
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %387, <4 x float> %386) #8, !dbg !20
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %388, <4 x float> zeroinitializer) #8, !dbg !20
  %391 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !20
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !20
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %391, <4 x float> %390) #8, !dbg !20
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %392, <4 x float> zeroinitializer) #8, !dbg !20
  %395 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !20
  %396 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !20
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %395, <4 x float> %394) #8, !dbg !20
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %396, <4 x float> zeroinitializer) #8, !dbg !20
  %399 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !20
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
  %address.local415 = and i32 %address.local317, 6144, !dbg !18
  %address.local416 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local417 = and i32 %address.local416, 1920, !dbg !18
  %address.local418 = or disjoint i32 %address.local415, %address.local417, !dbg !18
  %address.local419 = zext nneg i32 %address.local418 to i64, !dbg !18
  %address.local420 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !18
  %address.local423 = and i64 %address.local265, 3, !dbg !10
  %address.local424 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local423, !dbg !18
  store float %544, ptr addrspace(3) %address.local424, align 4, !dbg !18
  %address.local426 = or disjoint i64 %address.local423, 4, !dbg !18
  %address.local427 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local426, !dbg !18
  store float %543, ptr addrspace(3) %address.local427, align 4, !dbg !18
  %address.local429 = or disjoint i64 %address.local423, 8, !dbg !18
  %address.local430 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local429, !dbg !18
  store float %542, ptr addrspace(3) %address.local430, align 4, !dbg !18
  %address.local432 = or i64 %address.local265, 12, !dbg !18
  %address.local433 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local432, !dbg !18
  store float %541, ptr addrspace(3) %address.local433, align 4, !dbg !18
  %address.local435 = or disjoint i64 %address.local423, 16, !dbg !18
  %address.local436 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local435, !dbg !18
  store float %536, ptr addrspace(3) %address.local436, align 4, !dbg !18
  %address.local438 = or disjoint i64 %address.local423, 20, !dbg !18
  %address.local439 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local438, !dbg !18
  store float %535, ptr addrspace(3) %address.local439, align 4, !dbg !18
  %address.local441 = or disjoint i64 %address.local423, 24, !dbg !18
  %address.local442 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local441, !dbg !18
  store float %534, ptr addrspace(3) %address.local442, align 4, !dbg !18
  %address.local444 = or i64 %address.local265, 28, !dbg !18
  %address.local445 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local444, !dbg !18
  store float %533, ptr addrspace(3) %address.local445, align 4, !dbg !18
  %address.local447 = or disjoint i64 %address.local423, 32, !dbg !18
  %address.local448 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local447, !dbg !18
  store float %528, ptr addrspace(3) %address.local448, align 4, !dbg !18
  %address.local450 = or disjoint i64 %address.local423, 36, !dbg !18
  %address.local451 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local450, !dbg !18
  store float %527, ptr addrspace(3) %address.local451, align 4, !dbg !18
  %address.local453 = or disjoint i64 %address.local423, 40, !dbg !18
  %address.local454 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local453, !dbg !18
  store float %526, ptr addrspace(3) %address.local454, align 4, !dbg !18
  %address.local456 = or i64 %address.local265, 44, !dbg !18
  %address.local457 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local456, !dbg !18
  store float %525, ptr addrspace(3) %address.local457, align 4, !dbg !18
  %address.local459 = or disjoint i64 %address.local423, 48, !dbg !18
  %address.local460 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local459, !dbg !18
  store float %520, ptr addrspace(3) %address.local460, align 4, !dbg !18
  %address.local462 = or disjoint i64 %address.local423, 52, !dbg !18
  %address.local463 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local462, !dbg !18
  store float %519, ptr addrspace(3) %address.local463, align 4, !dbg !18
  %address.local465 = or disjoint i64 %address.local423, 56, !dbg !18
  %address.local466 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local465, !dbg !18
  store float %518, ptr addrspace(3) %address.local466, align 4, !dbg !18
  %address.local468 = or i64 %address.local265, 60, !dbg !18
  %address.local469 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local468, !dbg !18
  store float %517, ptr addrspace(3) %address.local469, align 4, !dbg !18
  %address.local471 = or disjoint i64 %address.local423, 64, !dbg !18
  %address.local472 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local471, !dbg !18
  store float %512, ptr addrspace(3) %address.local472, align 4, !dbg !18
  %address.local474 = or disjoint i64 %address.local423, 68, !dbg !18
  %address.local475 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local474, !dbg !18
  store float %511, ptr addrspace(3) %address.local475, align 4, !dbg !18
  %address.local477 = or disjoint i64 %address.local423, 72, !dbg !18
  %address.local478 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local477, !dbg !18
  store float %510, ptr addrspace(3) %address.local478, align 4, !dbg !18
  %address.local480 = or i64 %address.local265, 76, !dbg !18
  %address.local481 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local480, !dbg !18
  store float %509, ptr addrspace(3) %address.local481, align 4, !dbg !18
  %address.local483 = or disjoint i64 %address.local423, 80, !dbg !18
  %address.local484 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local483, !dbg !18
  store float %504, ptr addrspace(3) %address.local484, align 4, !dbg !18
  %address.local486 = or disjoint i64 %address.local423, 84, !dbg !18
  %address.local487 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local486, !dbg !18
  store float %503, ptr addrspace(3) %address.local487, align 4, !dbg !18
  %address.local489 = or disjoint i64 %address.local423, 88, !dbg !18
  %address.local490 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local489, !dbg !18
  store float %502, ptr addrspace(3) %address.local490, align 4, !dbg !18
  %address.local492 = or i64 %address.local265, 92, !dbg !18
  %address.local493 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local492, !dbg !18
  store float %501, ptr addrspace(3) %address.local493, align 4, !dbg !18
  %address.local495 = or disjoint i64 %address.local423, 96, !dbg !18
  %address.local496 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local495, !dbg !18
  store float %496, ptr addrspace(3) %address.local496, align 4, !dbg !18
  %address.local498 = or disjoint i64 %address.local423, 100, !dbg !18
  %address.local499 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local498, !dbg !18
  store float %495, ptr addrspace(3) %address.local499, align 4, !dbg !18
  %address.local501 = or disjoint i64 %address.local423, 104, !dbg !18
  %address.local502 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local501, !dbg !18
  store float %494, ptr addrspace(3) %address.local502, align 4, !dbg !18
  %address.local504 = or i64 %address.local265, 108, !dbg !18
  %address.local505 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local504, !dbg !18
  store float %493, ptr addrspace(3) %address.local505, align 4, !dbg !18
  %address.local507 = or disjoint i64 %address.local423, 112, !dbg !18
  %address.local508 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local507, !dbg !18
  store float %488, ptr addrspace(3) %address.local508, align 4, !dbg !18
  %address.local510 = or disjoint i64 %address.local423, 116, !dbg !18
  %address.local511 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local510, !dbg !18
  store float %487, ptr addrspace(3) %address.local511, align 4, !dbg !18
  %address.local513 = or disjoint i64 %address.local423, 120, !dbg !18
  %address.local514 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local513, !dbg !18
  store float %486, ptr addrspace(3) %address.local514, align 4, !dbg !18
  %address.local516 = or i64 %address.local265, 124, !dbg !18
  %address.local517 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local516, !dbg !18
  store float %485, ptr addrspace(3) %address.local517, align 4, !dbg !18
  %address.local525 = getelementptr inbounds nuw i8, ptr addrspace(3) %address.local420, i64 32768, !dbg !18
  %address.local526 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local423, !dbg !18
  store float %480, ptr addrspace(3) %address.local526, align 4, !dbg !18
  %address.local528 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local426, !dbg !18
  store float %479, ptr addrspace(3) %address.local528, align 4, !dbg !18
  %address.local530 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local429, !dbg !18
  store float %478, ptr addrspace(3) %address.local530, align 4, !dbg !18
  %address.local532 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local432, !dbg !18
  store float %477, ptr addrspace(3) %address.local532, align 4, !dbg !18
  %address.local534 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local435, !dbg !18
  store float %472, ptr addrspace(3) %address.local534, align 4, !dbg !18
  %address.local536 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local438, !dbg !18
  store float %471, ptr addrspace(3) %address.local536, align 4, !dbg !18
  %address.local538 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local441, !dbg !18
  store float %470, ptr addrspace(3) %address.local538, align 4, !dbg !18
  %address.local540 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local444, !dbg !18
  store float %469, ptr addrspace(3) %address.local540, align 4, !dbg !18
  %address.local542 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local447, !dbg !18
  store float %464, ptr addrspace(3) %address.local542, align 4, !dbg !18
  %address.local544 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local450, !dbg !18
  store float %463, ptr addrspace(3) %address.local544, align 4, !dbg !18
  %address.local546 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local453, !dbg !18
  store float %462, ptr addrspace(3) %address.local546, align 4, !dbg !18
  %address.local548 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local456, !dbg !18
  store float %461, ptr addrspace(3) %address.local548, align 4, !dbg !18
  %address.local550 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local459, !dbg !18
  store float %456, ptr addrspace(3) %address.local550, align 4, !dbg !18
  %address.local552 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local462, !dbg !18
  store float %455, ptr addrspace(3) %address.local552, align 4, !dbg !18
  %address.local554 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local465, !dbg !18
  store float %454, ptr addrspace(3) %address.local554, align 4, !dbg !18
  %address.local556 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local468, !dbg !18
  store float %453, ptr addrspace(3) %address.local556, align 4, !dbg !18
  %address.local558 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local471, !dbg !18
  store float %448, ptr addrspace(3) %address.local558, align 4, !dbg !18
  %address.local560 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local474, !dbg !18
  store float %447, ptr addrspace(3) %address.local560, align 4, !dbg !18
  %address.local562 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local477, !dbg !18
  store float %446, ptr addrspace(3) %address.local562, align 4, !dbg !18
  %address.local564 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local480, !dbg !18
  store float %445, ptr addrspace(3) %address.local564, align 4, !dbg !18
  %address.local566 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local483, !dbg !18
  store float %440, ptr addrspace(3) %address.local566, align 4, !dbg !18
  %address.local568 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local486, !dbg !18
  store float %439, ptr addrspace(3) %address.local568, align 4, !dbg !18
  %address.local570 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local489, !dbg !18
  store float %438, ptr addrspace(3) %address.local570, align 4, !dbg !18
  %address.local572 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local492, !dbg !18
  store float %437, ptr addrspace(3) %address.local572, align 4, !dbg !18
  %address.local574 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local495, !dbg !18
  store float %432, ptr addrspace(3) %address.local574, align 4, !dbg !18
  %address.local576 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local498, !dbg !18
  store float %431, ptr addrspace(3) %address.local576, align 4, !dbg !18
  %address.local578 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local501, !dbg !18
  store float %430, ptr addrspace(3) %address.local578, align 4, !dbg !18
  %address.local580 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local504, !dbg !18
  store float %429, ptr addrspace(3) %address.local580, align 4, !dbg !18
  %address.local582 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local507, !dbg !18
  store float %424, ptr addrspace(3) %address.local582, align 4, !dbg !18
  %address.local584 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local510, !dbg !18
  store float %423, ptr addrspace(3) %address.local584, align 4, !dbg !18
  %address.local586 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local513, !dbg !18
  store float %422, ptr addrspace(3) %address.local586, align 4, !dbg !18
  %address.local588 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local516, !dbg !18
  store float %421, ptr addrspace(3) %address.local588, align 4, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.local601.idx = shl nuw nsw i64 %address.local423, 4, !dbg !18
  %address.local601 = getelementptr inbounds nuw i8, ptr addrspace(3) %address.local420, i64 %address.local601.idx, !dbg !18
  %545 = load <4 x float>, ptr addrspace(3) %address.local601, align 16, !dbg !18
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local601, i64 64, !dbg !18
  %546 = load <4 x float>, ptr addrspace(3) %lds.const.gep1685, align 16, !dbg !18
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local601, i64 128, !dbg !18
  %547 = load <4 x float>, ptr addrspace(3) %lds.const.gep1689, align 16, !dbg !18
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local601, i64 192, !dbg !18
  %548 = load <4 x float>, ptr addrspace(3) %lds.const.gep1693, align 16, !dbg !18
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local601, i64 256, !dbg !18
  %549 = load <4 x float>, ptr addrspace(3) %lds.const.gep1697, align 16, !dbg !18
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local601, i64 320, !dbg !18
  %550 = load <4 x float>, ptr addrspace(3) %lds.const.gep1701, align 16, !dbg !18
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local601, i64 384, !dbg !18
  %551 = load <4 x float>, ptr addrspace(3) %lds.const.gep1705, align 16, !dbg !18
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local601, i64 448, !dbg !18
  %552 = load <4 x float>, ptr addrspace(3) %lds.const.gep1709, align 16, !dbg !18
  %lds.const.gep1713 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32768, !dbg !18
  %553 = load <4 x float>, ptr addrspace(3) %lds.const.gep1713, align 16, !dbg !18
  %lds.const.gep1717 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32832, !dbg !18
  %554 = load <4 x float>, ptr addrspace(3) %lds.const.gep1717, align 16, !dbg !18
  %lds.const.gep1721 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32896, !dbg !18
  %555 = load <4 x float>, ptr addrspace(3) %lds.const.gep1721, align 16, !dbg !18
  %lds.const.gep1725 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32960, !dbg !18
  %556 = load <4 x float>, ptr addrspace(3) %lds.const.gep1725, align 16, !dbg !18
  %lds.const.gep1729 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33024, !dbg !18
  %557 = load <4 x float>, ptr addrspace(3) %lds.const.gep1729, align 16, !dbg !18
  %lds.const.gep1733 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33088, !dbg !18
  %558 = load <4 x float>, ptr addrspace(3) %lds.const.gep1733, align 16, !dbg !18
  %lds.const.gep1737 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33152, !dbg !18
  %559 = load <4 x float>, ptr addrspace(3) %lds.const.gep1737, align 16, !dbg !18
  %lds.const.gep1741 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33216, !dbg !18
  %560 = load <4 x float>, ptr addrspace(3) %lds.const.gep1741, align 16, !dbg !18
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
  %compare.high1303 = trunc nuw i64 %566 to i32, !dbg !19
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1303, !dbg !19
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1303, !dbg !19
  %compare.low.base1304 = trunc i64 %565 to i32, !dbg !19
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 1) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1307 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 2) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1311 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 3) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1315 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 4) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1319 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 5) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1323 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 6) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1327 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 7) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1331 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 8) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1335 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 9) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1339 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 10) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1343 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 11) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1347 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 12) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1351 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 13) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1355 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 14) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1359 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 15) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1363 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 16) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1367 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 17) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1371 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 18) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1375 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 19) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1379 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 20) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1383 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 21) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1387 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 22) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1391 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 23) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1395 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 24) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1399 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 25) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1403 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 26) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1407 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 27) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1411 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 28) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1415 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 29) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1419 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 30) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1423 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 31) #8, !dbg !19, !frisk.register_lifetimes !5
  %567 = insertelement <32 x i32> poison, i32 %compare.low.base1304, i64 0, !dbg !19
  %568 = insertelement <32 x i32> %567, i32 %compare.local.base, i64 1, !dbg !19
  %569 = insertelement <32 x i32> %568, i32 %compare.local.base1307, i64 2, !dbg !19
  %570 = insertelement <32 x i32> %569, i32 %compare.local.base1311, i64 3, !dbg !19
  %571 = insertelement <32 x i32> %570, i32 %compare.local.base1315, i64 4, !dbg !19
  %572 = insertelement <32 x i32> %571, i32 %compare.local.base1319, i64 5, !dbg !19
  %573 = insertelement <32 x i32> %572, i32 %compare.local.base1323, i64 6, !dbg !19
  %574 = insertelement <32 x i32> %573, i32 %compare.local.base1327, i64 7, !dbg !19
  %575 = insertelement <32 x i32> %574, i32 %compare.local.base1331, i64 8, !dbg !19
  %576 = insertelement <32 x i32> %575, i32 %compare.local.base1335, i64 9, !dbg !19
  %577 = insertelement <32 x i32> %576, i32 %compare.local.base1339, i64 10, !dbg !19
  %578 = insertelement <32 x i32> %577, i32 %compare.local.base1343, i64 11, !dbg !19
  %579 = insertelement <32 x i32> %578, i32 %compare.local.base1347, i64 12, !dbg !19
  %580 = insertelement <32 x i32> %579, i32 %compare.local.base1351, i64 13, !dbg !19
  %581 = insertelement <32 x i32> %580, i32 %compare.local.base1355, i64 14, !dbg !19
  %582 = insertelement <32 x i32> %581, i32 %compare.local.base1359, i64 15, !dbg !19
  %583 = insertelement <32 x i32> %582, i32 %compare.local.base1363, i64 16, !dbg !19
  %584 = insertelement <32 x i32> %583, i32 %compare.local.base1367, i64 17, !dbg !19
  %585 = insertelement <32 x i32> %584, i32 %compare.local.base1371, i64 18, !dbg !19
  %586 = insertelement <32 x i32> %585, i32 %compare.local.base1375, i64 19, !dbg !19
  %587 = insertelement <32 x i32> %586, i32 %compare.local.base1379, i64 20, !dbg !19
  %588 = insertelement <32 x i32> %587, i32 %compare.local.base1383, i64 21, !dbg !19
  %589 = insertelement <32 x i32> %588, i32 %compare.local.base1387, i64 22, !dbg !19
  %590 = insertelement <32 x i32> %589, i32 %compare.local.base1391, i64 23, !dbg !19
  %591 = insertelement <32 x i32> %590, i32 %compare.local.base1395, i64 24, !dbg !19
  %592 = insertelement <32 x i32> %591, i32 %compare.local.base1399, i64 25, !dbg !19
  %593 = insertelement <32 x i32> %592, i32 %compare.local.base1403, i64 26, !dbg !19
  %594 = insertelement <32 x i32> %593, i32 %compare.local.base1407, i64 27, !dbg !19
  %595 = insertelement <32 x i32> %594, i32 %compare.local.base1411, i64 28, !dbg !19
  %596 = insertelement <32 x i32> %595, i32 %compare.local.base1415, i64 29, !dbg !19
  %597 = insertelement <32 x i32> %596, i32 %compare.local.base1419, i64 30, !dbg !19
  %598 = insertelement <32 x i32> %597, i32 %compare.local.base1423, i64 31, !dbg !19
  %599 = or <32 x i32> %598, <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 48, i32 49, i32 50, i32 51, i32 64, i32 65, i32 66, i32 67, i32 80, i32 81, i32 82, i32 83, i32 96, i32 97, i32 98, i32 99, i32 112, i32 113, i32 114, i32 115>, !dbg !19
  %600 = icmp ugt <32 x i32> %70, %599, !dbg !19
  %601 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !19
  %602 = shufflevector <32 x i1> %601, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %603 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !19
  %604 = shufflevector <32 x i1> %603, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %605 = select <32 x i1> %602, <32 x i1> %600, <32 x i1> %604, !dbg !19
  %606 = select <32 x i1> %605, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !22
  %compare.high.equal1429 = icmp eq i32 %compare.high1427, %compare.high1303, !dbg !19
  %compare.high.order1430 = icmp ugt i32 %compare.high1427, %compare.high1303, !dbg !19
  %compare.local.base1435 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 32) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1439 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 33) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1443 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 34) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1447 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 35) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1451 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 36) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1455 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 37) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1459 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 38) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1463 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 39) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1467 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 40) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1471 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 41) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1475 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 42) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1479 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 43) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1483 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 44) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1487 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 45) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1491 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 46) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1495 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 47) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1499 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 48) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1503 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 49) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1507 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 50) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1511 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 51) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1515 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 52) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1519 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 53) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1523 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 54) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1527 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 55) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1531 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 56) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1535 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 57) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1539 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 58) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1543 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 59) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1547 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 60) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1551 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 61) #8, !dbg !19, !frisk.register_lifetimes !5
  %compare.local.base1555 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 62) #8, !dbg !19, !frisk.register_lifetimes !5
  %607 = insertelement <32 x i32> %567, i32 %compare.local.base1435, i64 1, !dbg !19
  %608 = insertelement <32 x i32> %607, i32 %compare.local.base1439, i64 2, !dbg !19
  %609 = insertelement <32 x i32> %608, i32 %compare.local.base1443, i64 3, !dbg !19
  %610 = insertelement <32 x i32> %609, i32 %compare.local.base1447, i64 4, !dbg !19
  %611 = insertelement <32 x i32> %610, i32 %compare.local.base1451, i64 5, !dbg !19
  %612 = insertelement <32 x i32> %611, i32 %compare.local.base1455, i64 6, !dbg !19
  %613 = insertelement <32 x i32> %612, i32 %compare.local.base1459, i64 7, !dbg !19
  %614 = insertelement <32 x i32> %613, i32 %compare.local.base1463, i64 8, !dbg !19
  %615 = insertelement <32 x i32> %614, i32 %compare.local.base1467, i64 9, !dbg !19
  %616 = insertelement <32 x i32> %615, i32 %compare.local.base1471, i64 10, !dbg !19
  %617 = insertelement <32 x i32> %616, i32 %compare.local.base1475, i64 11, !dbg !19
  %618 = insertelement <32 x i32> %617, i32 %compare.local.base1479, i64 12, !dbg !19
  %619 = insertelement <32 x i32> %618, i32 %compare.local.base1483, i64 13, !dbg !19
  %620 = insertelement <32 x i32> %619, i32 %compare.local.base1487, i64 14, !dbg !19
  %621 = insertelement <32 x i32> %620, i32 %compare.local.base1491, i64 15, !dbg !19
  %622 = insertelement <32 x i32> %621, i32 %compare.local.base1495, i64 16, !dbg !19
  %623 = insertelement <32 x i32> %622, i32 %compare.local.base1499, i64 17, !dbg !19
  %624 = insertelement <32 x i32> %623, i32 %compare.local.base1503, i64 18, !dbg !19
  %625 = insertelement <32 x i32> %624, i32 %compare.local.base1507, i64 19, !dbg !19
  %626 = insertelement <32 x i32> %625, i32 %compare.local.base1511, i64 20, !dbg !19
  %627 = insertelement <32 x i32> %626, i32 %compare.local.base1515, i64 21, !dbg !19
  %628 = insertelement <32 x i32> %627, i32 %compare.local.base1519, i64 22, !dbg !19
  %629 = insertelement <32 x i32> %628, i32 %compare.local.base1523, i64 23, !dbg !19
  %630 = insertelement <32 x i32> %629, i32 %compare.local.base1527, i64 24, !dbg !19
  %631 = insertelement <32 x i32> %630, i32 %compare.local.base1531, i64 25, !dbg !19
  %632 = insertelement <32 x i32> %631, i32 %compare.local.base1535, i64 26, !dbg !19
  %633 = insertelement <32 x i32> %632, i32 %compare.local.base1539, i64 27, !dbg !19
  %634 = insertelement <32 x i32> %633, i32 %compare.local.base1543, i64 28, !dbg !19
  %635 = insertelement <32 x i32> %634, i32 %compare.local.base1547, i64 29, !dbg !19
  %636 = insertelement <32 x i32> %635, i32 %compare.local.base1551, i64 30, !dbg !19
  %637 = insertelement <32 x i32> %636, i32 %compare.local.base1555, i64 31, !dbg !19
  %638 = or <32 x i32> %637, <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 48, i32 49, i32 50, i32 51, i32 64, i32 65, i32 66, i32 67, i32 80, i32 81, i32 82, i32 83, i32 96, i32 97, i32 98, i32 99, i32 112, i32 113, i32 114, i32 115>, !dbg !19
  %639 = icmp ugt <32 x i32> %68, %638, !dbg !19
  %640 = insertelement <32 x i1> poison, i1 %compare.high.equal1429, i64 0, !dbg !19
  %641 = shufflevector <32 x i1> %640, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %642 = insertelement <32 x i1> poison, i1 %compare.high.order1430, i64 0, !dbg !19
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
  br i1 %39, label %804, label %809, !dbg !13

804:                                              ; preds = %648
  %805 = fsub <1 x float> %783, %785, !dbg !26
  %806 = extractelement <1 x float> %805, i64 0, !dbg !26
  %807 = fsub <1 x float> %770, %772, !dbg !26
  %808 = extractelement <1 x float> %807, i64 0, !dbg !26
  %address.thread766 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !13, !range !11, !frisk.register_lifetimes !5
  %address.local767 = lshr i32 %address.thread766, 2, !dbg !18
  %address.local768 = and i32 %address.local767, 48, !dbg !18
  %address.local769 = and i32 %address.thread766, 15, !dbg !18
  %address.local770 = or disjoint i32 %address.local768, %address.local769, !dbg !18
  %address.local771 = zext nneg i32 %address.local770 to i64, !dbg !18
  %address.local772 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local771, !dbg !13
  store float %808, ptr addrspace(3) %address.local772, align 4, !dbg !13
  %address.local779 = getelementptr inbounds nuw i8, ptr addrspace(3) %address.local772, i64 256, !dbg !13
  store float %806, ptr addrspace(3) %address.local779, align 4, !dbg !13
  br label %809, !dbg !13

809:                                              ; preds = %804, %648
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.thread780 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !13, !range !11, !frisk.register_lifetimes !5
  %address.local781 = lshr i32 %address.thread780, 1, !dbg !13
  %address.local782 = and i32 %address.local781, 96, !dbg !13
  %address.local783 = and i32 %address.thread780, 15, !dbg !13
  %address.local784 = or disjoint i32 %address.local782, %address.local783, !dbg !13
  %address.local785 = zext nneg i32 %address.local784 to i64, !dbg !13
  %address.local786 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local785, !dbg !13
  %810 = load float, ptr addrspace(3) %address.local786, align 4, !dbg !13
  %lds.const.gep1745 = getelementptr i8, ptr addrspace(3) %address.local786, i64 64, !dbg !13
  %811 = load float, ptr addrspace(3) %lds.const.gep1745, align 4, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %812 = tail call float @llvm.amdgcn.exp2.f32(float %810), !dbg !13
  %813 = tail call float @llvm.amdgcn.exp2.f32(float %811), !dbg !13
  br i1 %39, label %814, label %815, !dbg !27

814:                                              ; preds = %809
  %address.thread789 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !27, !range !11, !frisk.register_lifetimes !5
  %address.local790 = lshr i32 %address.thread789, 1, !dbg !13
  %address.local791 = and i32 %address.local790, 96, !dbg !13
  %address.local792 = and i32 %address.thread789, 15, !dbg !13
  %address.local793 = or disjoint i32 %address.local791, %address.local792, !dbg !13
  %address.local794 = zext nneg i32 %address.local793 to i64, !dbg !13
  %address.local795 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !13
  store float %812, ptr addrspace(3) %address.local795, align 4, !dbg !27
  %address.local797 = getelementptr i8, ptr addrspace(3) %address.local795, i64 64, !dbg !13
  store float %813, ptr addrspace(3) %address.local797, align 4, !dbg !27
  br label %815, !dbg !27

815:                                              ; preds = %814, %809
  fence syncscope("workgroup") release, !dbg !27
  tail call void @llvm.amdgcn.s.barrier(), !dbg !27
  fence syncscope("workgroup") acquire, !dbg !27
  %address.thread798 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !27, !range !11, !frisk.register_lifetimes !5
  %address.local799 = lshr i32 %address.thread798, 2, !dbg !18
  %address.local800 = and i32 %address.local799, 48, !dbg !18
  %address.local801 = and i32 %address.thread798, 15, !dbg !18
  %address.local802 = or disjoint i32 %address.local800, %address.local801, !dbg !18
  %address.local803 = zext nneg i32 %address.local802 to i64, !dbg !18
  %address.local804 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local803, !dbg !27
  %816 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !27
  %lds.const.gep1746 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !27
  %817 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !27
  fence syncscope("workgroup") release, !dbg !27
  tail call void @llvm.amdgcn.s.barrier(), !dbg !27
  fence syncscope("workgroup") acquire, !dbg !27
  %818 = add nuw nsw i64 %88, %15, !dbg !28
  %819 = shl i64 %818, 19, !dbg !28
  %820 = and i64 %819, 16252928, !dbg !28
  %address.local812 = zext nneg i32 %address.thread798 to i64, !dbg !10
  %address.local813 = shl nuw nsw i64 %address.local812, 3, !dbg !9
  %address.local814 = and i64 %address.local813, 1536, !dbg !9
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local814, !dbg !9
  %address.local816 = ptrtoint ptr addrspace(3) %address.local815 to i64, !dbg !9
  %address.local817 = trunc i64 %address.local816 to i32, !dbg !9
  %821 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local817), !dbg !28
  %address.local818 = shl nuw nsw i64 %address.local812, 6, !dbg !28
  %address.local819 = and i64 %address.local818, 16128, !dbg !28
  %address.local821 = and i64 %address.local813, 24, !dbg !28
  %.idx5558 = add i64 %820, %94, !dbg !28
  %822 = or disjoint i64 %address.local821, %.idx5558, !dbg !28
  %823 = shl i64 %822, 1, !dbg !28
  %824 = or disjoint i64 %823, %address.local819, !dbg !28
  %dma.buffer.offset1773 = trunc i64 %824 to i32, !dbg !28
  %dma.buffer.lds1775 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %821), !dbg !28
  %825 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1775, i32 %dma.buffer.offset1773, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %address.local824 = or disjoint i64 %address.local812, 256, !dbg !9
  %address.local825 = shl nuw nsw i64 %address.local824, 3, !dbg !9
  %address.local826 = and i64 %address.local825, 3584, !dbg !9
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local826, !dbg !9
  %address.local828 = ptrtoint ptr addrspace(3) %address.local827 to i64, !dbg !9
  %address.local829 = trunc i64 %address.local828 to i32, !dbg !9
  %826 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local829), !dbg !28
  %address.local830 = shl nuw nsw i64 %address.local824, 6, !dbg !28
  %address.local831 = and i64 %address.local830, 32512, !dbg !28
  %827 = or disjoint i64 %823, %address.local831, !dbg !28
  %dma.buffer.offset1770 = trunc i64 %827 to i32, !dbg !28
  %dma.buffer.lds1772 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %826), !dbg !28
  %828 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1772, i32 %dma.buffer.offset1770, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %829 = extractvalue [2 x <1 x float>] %76, 0, !dbg !27
  %830 = extractelement <1 x float> %829, i64 0, !dbg !27
  %831 = fmul float %830, %816, !dbg !27
  %832 = extractvalue [2 x <1 x float>] %76, 1, !dbg !27
  %833 = extractelement <1 x float> %832, i64 0, !dbg !27
  %834 = fmul float %833, %817, !dbg !27
  %835 = extractelement <4 x float> %787, i64 0, !dbg !29
  %836 = tail call float @llvm.amdgcn.exp2.f32(float %835), !dbg !29
  %837 = insertelement <4 x float> poison, float %836, i64 0, !dbg !30
  %838 = extractelement <4 x float> %787, i64 1, !dbg !29
  %839 = tail call float @llvm.amdgcn.exp2.f32(float %838), !dbg !29
  %840 = insertelement <4 x float> %837, float %839, i64 1, !dbg !30
  %841 = extractelement <4 x float> %787, i64 2, !dbg !29
  %842 = tail call float @llvm.amdgcn.exp2.f32(float %841), !dbg !29
  %843 = insertelement <4 x float> %840, float %842, i64 2, !dbg !30
  %844 = extractelement <4 x float> %787, i64 3, !dbg !29
  %845 = tail call float @llvm.amdgcn.exp2.f32(float %844), !dbg !29
  %846 = insertelement <4 x float> %843, float %845, i64 3, !dbg !30
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %846) #8, !dbg !30, !frisk.register_lifetimes !5
  %847 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !30
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %847) #8, !dbg !29, !frisk.register_lifetimes !5
  %848 = extractelement <4 x float> %788, i64 0, !dbg !29
  %849 = tail call float @llvm.amdgcn.exp2.f32(float %848), !dbg !29
  %850 = insertelement <4 x float> poison, float %849, i64 0, !dbg !30
  %local.reduce120 = fadd float %836, %849, !dbg !31, !frisk.local_reduction !5
  %851 = extractelement <4 x float> %788, i64 1, !dbg !29
  %852 = tail call float @llvm.amdgcn.exp2.f32(float %851), !dbg !29
  %853 = insertelement <4 x float> %850, float %852, i64 1, !dbg !30
  %local.reduce121 = fadd float %839, %852, !dbg !31, !frisk.local_reduction !5
  %854 = extractelement <4 x float> %788, i64 2, !dbg !29
  %855 = tail call float @llvm.amdgcn.exp2.f32(float %854), !dbg !29
  %856 = insertelement <4 x float> %853, float %855, i64 2, !dbg !30
  %local.reduce122 = fadd float %842, %855, !dbg !31, !frisk.local_reduction !5
  %857 = extractelement <4 x float> %788, i64 3, !dbg !29
  %858 = tail call float @llvm.amdgcn.exp2.f32(float %857), !dbg !29
  %859 = insertelement <4 x float> %856, float %858, i64 3, !dbg !30
  %math.result228 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %859) #8, !dbg !30, !frisk.register_lifetimes !5
  %860 = fptrunc <4 x float> %math.result228 to <4 x half>, !dbg !30
  %math.packed229 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %860) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce123 = fadd float %845, %858, !dbg !31, !frisk.local_reduction !5
  %861 = extractelement <4 x float> %789, i64 0, !dbg !29
  %862 = tail call float @llvm.amdgcn.exp2.f32(float %861), !dbg !29
  %863 = insertelement <4 x float> poison, float %862, i64 0, !dbg !30
  %local.reduce124 = fadd float %862, %local.reduce120, !dbg !31, !frisk.local_reduction !5
  %864 = extractelement <4 x float> %789, i64 1, !dbg !29
  %865 = tail call float @llvm.amdgcn.exp2.f32(float %864), !dbg !29
  %866 = insertelement <4 x float> %863, float %865, i64 1, !dbg !30
  %local.reduce125 = fadd float %865, %local.reduce121, !dbg !31, !frisk.local_reduction !5
  %867 = extractelement <4 x float> %789, i64 2, !dbg !29
  %868 = tail call float @llvm.amdgcn.exp2.f32(float %867), !dbg !29
  %869 = insertelement <4 x float> %866, float %868, i64 2, !dbg !30
  %local.reduce126 = fadd float %868, %local.reduce122, !dbg !31, !frisk.local_reduction !5
  %870 = extractelement <4 x float> %789, i64 3, !dbg !29
  %871 = tail call float @llvm.amdgcn.exp2.f32(float %870), !dbg !29
  %872 = insertelement <4 x float> %869, float %871, i64 3, !dbg !30
  %math.result230 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %872) #8, !dbg !30, !frisk.register_lifetimes !5
  %873 = fptrunc <4 x float> %math.result230 to <4 x half>, !dbg !30
  %math.packed231 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %873) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce127 = fadd float %871, %local.reduce123, !dbg !31, !frisk.local_reduction !5
  %874 = extractelement <4 x float> %790, i64 0, !dbg !29
  %875 = tail call float @llvm.amdgcn.exp2.f32(float %874), !dbg !29
  %876 = insertelement <4 x float> poison, float %875, i64 0, !dbg !30
  %local.reduce128 = fadd float %875, %local.reduce124, !dbg !31, !frisk.local_reduction !5
  %877 = extractelement <4 x float> %790, i64 1, !dbg !29
  %878 = tail call float @llvm.amdgcn.exp2.f32(float %877), !dbg !29
  %879 = insertelement <4 x float> %876, float %878, i64 1, !dbg !30
  %local.reduce129 = fadd float %878, %local.reduce125, !dbg !31, !frisk.local_reduction !5
  %880 = extractelement <4 x float> %790, i64 2, !dbg !29
  %881 = tail call float @llvm.amdgcn.exp2.f32(float %880), !dbg !29
  %882 = insertelement <4 x float> %879, float %881, i64 2, !dbg !30
  %local.reduce130 = fadd float %881, %local.reduce126, !dbg !31, !frisk.local_reduction !5
  %883 = extractelement <4 x float> %790, i64 3, !dbg !29
  %884 = tail call float @llvm.amdgcn.exp2.f32(float %883), !dbg !29
  %885 = insertelement <4 x float> %882, float %884, i64 3, !dbg !30
  %math.result232 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %885) #8, !dbg !30, !frisk.register_lifetimes !5
  %886 = fptrunc <4 x float> %math.result232 to <4 x half>, !dbg !30
  %math.packed233 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %886) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce131 = fadd float %884, %local.reduce127, !dbg !31, !frisk.local_reduction !5
  %887 = extractelement <4 x float> %791, i64 0, !dbg !29
  %888 = tail call float @llvm.amdgcn.exp2.f32(float %887), !dbg !29
  %889 = insertelement <4 x float> poison, float %888, i64 0, !dbg !30
  %local.reduce132 = fadd float %888, %local.reduce128, !dbg !31, !frisk.local_reduction !5
  %890 = extractelement <4 x float> %791, i64 1, !dbg !29
  %891 = tail call float @llvm.amdgcn.exp2.f32(float %890), !dbg !29
  %892 = insertelement <4 x float> %889, float %891, i64 1, !dbg !30
  %local.reduce133 = fadd float %891, %local.reduce129, !dbg !31, !frisk.local_reduction !5
  %893 = extractelement <4 x float> %791, i64 2, !dbg !29
  %894 = tail call float @llvm.amdgcn.exp2.f32(float %893), !dbg !29
  %895 = insertelement <4 x float> %892, float %894, i64 2, !dbg !30
  %local.reduce134 = fadd float %894, %local.reduce130, !dbg !31, !frisk.local_reduction !5
  %896 = extractelement <4 x float> %791, i64 3, !dbg !29
  %897 = tail call float @llvm.amdgcn.exp2.f32(float %896), !dbg !29
  %898 = insertelement <4 x float> %895, float %897, i64 3, !dbg !30
  %math.result234 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %898) #8, !dbg !30, !frisk.register_lifetimes !5
  %899 = fptrunc <4 x float> %math.result234 to <4 x half>, !dbg !30
  %math.packed235 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %899) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce135 = fadd float %897, %local.reduce131, !dbg !31, !frisk.local_reduction !5
  %900 = extractelement <4 x float> %792, i64 0, !dbg !29
  %901 = tail call float @llvm.amdgcn.exp2.f32(float %900), !dbg !29
  %902 = insertelement <4 x float> poison, float %901, i64 0, !dbg !30
  %local.reduce136 = fadd float %901, %local.reduce132, !dbg !31, !frisk.local_reduction !5
  %903 = extractelement <4 x float> %792, i64 1, !dbg !29
  %904 = tail call float @llvm.amdgcn.exp2.f32(float %903), !dbg !29
  %905 = insertelement <4 x float> %902, float %904, i64 1, !dbg !30
  %local.reduce137 = fadd float %904, %local.reduce133, !dbg !31, !frisk.local_reduction !5
  %906 = extractelement <4 x float> %792, i64 2, !dbg !29
  %907 = tail call float @llvm.amdgcn.exp2.f32(float %906), !dbg !29
  %908 = insertelement <4 x float> %905, float %907, i64 2, !dbg !30
  %local.reduce138 = fadd float %907, %local.reduce134, !dbg !31, !frisk.local_reduction !5
  %909 = extractelement <4 x float> %792, i64 3, !dbg !29
  %910 = tail call float @llvm.amdgcn.exp2.f32(float %909), !dbg !29
  %911 = insertelement <4 x float> %908, float %910, i64 3, !dbg !30
  %math.result236 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %911) #8, !dbg !30, !frisk.register_lifetimes !5
  %912 = fptrunc <4 x float> %math.result236 to <4 x half>, !dbg !30
  %math.packed237 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %912) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce139 = fadd float %910, %local.reduce135, !dbg !31, !frisk.local_reduction !5
  %913 = extractelement <4 x float> %793, i64 0, !dbg !29
  %914 = tail call float @llvm.amdgcn.exp2.f32(float %913), !dbg !29
  %915 = insertelement <4 x float> poison, float %914, i64 0, !dbg !30
  %local.reduce140 = fadd float %914, %local.reduce136, !dbg !31, !frisk.local_reduction !5
  %916 = extractelement <4 x float> %793, i64 1, !dbg !29
  %917 = tail call float @llvm.amdgcn.exp2.f32(float %916), !dbg !29
  %918 = insertelement <4 x float> %915, float %917, i64 1, !dbg !30
  %local.reduce141 = fadd float %917, %local.reduce137, !dbg !31, !frisk.local_reduction !5
  %919 = extractelement <4 x float> %793, i64 2, !dbg !29
  %920 = tail call float @llvm.amdgcn.exp2.f32(float %919), !dbg !29
  %921 = insertelement <4 x float> %918, float %920, i64 2, !dbg !30
  %local.reduce142 = fadd float %920, %local.reduce138, !dbg !31, !frisk.local_reduction !5
  %922 = extractelement <4 x float> %793, i64 3, !dbg !29
  %923 = tail call float @llvm.amdgcn.exp2.f32(float %922), !dbg !29
  %924 = insertelement <4 x float> %921, float %923, i64 3, !dbg !30
  %math.result238 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %924) #8, !dbg !30, !frisk.register_lifetimes !5
  %925 = fptrunc <4 x float> %math.result238 to <4 x half>, !dbg !30
  %math.packed239 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %925) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce143 = fadd float %923, %local.reduce139, !dbg !31, !frisk.local_reduction !5
  %926 = extractelement <4 x float> %794, i64 0, !dbg !29
  %927 = tail call float @llvm.amdgcn.exp2.f32(float %926), !dbg !29
  %928 = insertelement <4 x float> poison, float %927, i64 0, !dbg !30
  %local.reduce144 = fadd float %927, %local.reduce140, !dbg !31, !frisk.local_reduction !5
  %929 = extractelement <4 x float> %794, i64 1, !dbg !29
  %930 = tail call float @llvm.amdgcn.exp2.f32(float %929), !dbg !29
  %931 = insertelement <4 x float> %928, float %930, i64 1, !dbg !30
  %local.reduce145 = fadd float %930, %local.reduce141, !dbg !31, !frisk.local_reduction !5
  %local.reduce148 = fadd float %local.reduce144, %local.reduce145, !dbg !31, !frisk.local_reduction !5
  %932 = extractelement <4 x float> %794, i64 2, !dbg !29
  %933 = tail call float @llvm.amdgcn.exp2.f32(float %932), !dbg !29
  %934 = insertelement <4 x float> %931, float %933, i64 2, !dbg !30
  %local.reduce146 = fadd float %933, %local.reduce142, !dbg !31, !frisk.local_reduction !5
  %935 = extractelement <4 x float> %794, i64 3, !dbg !29
  %936 = tail call float @llvm.amdgcn.exp2.f32(float %935), !dbg !29
  %937 = insertelement <4 x float> %934, float %936, i64 3, !dbg !30
  %math.result240 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %937) #8, !dbg !30, !frisk.register_lifetimes !5
  %938 = fptrunc <4 x float> %math.result240 to <4 x half>, !dbg !30
  %math.packed241 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %938) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce147 = fadd float %936, %local.reduce143, !dbg !31, !frisk.local_reduction !5
  %local.reduce149 = fadd float %local.reduce146, %local.reduce147, !dbg !31, !frisk.local_reduction !5
  %939 = fadd float %local.reduce148, %local.reduce149, !dbg !31, !frisk.local_reduction !5
  %940 = extractelement <4 x float> %796, i64 0, !dbg !29
  %941 = tail call float @llvm.amdgcn.exp2.f32(float %940), !dbg !29
  %942 = insertelement <4 x float> poison, float %941, i64 0, !dbg !30
  %943 = extractelement <4 x float> %796, i64 1, !dbg !29
  %944 = tail call float @llvm.amdgcn.exp2.f32(float %943), !dbg !29
  %945 = insertelement <4 x float> %942, float %944, i64 1, !dbg !30
  %946 = extractelement <4 x float> %796, i64 2, !dbg !29
  %947 = tail call float @llvm.amdgcn.exp2.f32(float %946), !dbg !29
  %948 = insertelement <4 x float> %945, float %947, i64 2, !dbg !30
  %949 = extractelement <4 x float> %796, i64 3, !dbg !29
  %950 = tail call float @llvm.amdgcn.exp2.f32(float %949), !dbg !29
  %951 = insertelement <4 x float> %948, float %950, i64 3, !dbg !30
  %math.result242 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %951) #8, !dbg !30, !frisk.register_lifetimes !5
  %952 = fptrunc <4 x float> %math.result242 to <4 x half>, !dbg !30
  %math.packed243 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %952) #8, !dbg !29, !frisk.register_lifetimes !5
  %953 = extractelement <4 x float> %797, i64 0, !dbg !29
  %954 = tail call float @llvm.amdgcn.exp2.f32(float %953), !dbg !29
  %955 = insertelement <4 x float> poison, float %954, i64 0, !dbg !30
  %local.reduce = fadd float %941, %954, !dbg !31, !frisk.local_reduction !5
  %956 = extractelement <4 x float> %797, i64 1, !dbg !29
  %957 = tail call float @llvm.amdgcn.exp2.f32(float %956), !dbg !29
  %958 = insertelement <4 x float> %955, float %957, i64 1, !dbg !30
  %local.reduce90 = fadd float %944, %957, !dbg !31, !frisk.local_reduction !5
  %959 = extractelement <4 x float> %797, i64 2, !dbg !29
  %960 = tail call float @llvm.amdgcn.exp2.f32(float %959), !dbg !29
  %961 = insertelement <4 x float> %958, float %960, i64 2, !dbg !30
  %local.reduce91 = fadd float %947, %960, !dbg !31, !frisk.local_reduction !5
  %962 = extractelement <4 x float> %797, i64 3, !dbg !29
  %963 = tail call float @llvm.amdgcn.exp2.f32(float %962), !dbg !29
  %964 = insertelement <4 x float> %961, float %963, i64 3, !dbg !30
  %math.result244 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %964) #8, !dbg !30, !frisk.register_lifetimes !5
  %965 = fptrunc <4 x float> %math.result244 to <4 x half>, !dbg !30
  %math.packed245 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %965) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce92 = fadd float %950, %963, !dbg !31, !frisk.local_reduction !5
  %966 = extractelement <4 x float> %798, i64 0, !dbg !29
  %967 = tail call float @llvm.amdgcn.exp2.f32(float %966), !dbg !29
  %968 = insertelement <4 x float> poison, float %967, i64 0, !dbg !30
  %local.reduce93 = fadd float %967, %local.reduce, !dbg !31, !frisk.local_reduction !5
  %969 = extractelement <4 x float> %798, i64 1, !dbg !29
  %970 = tail call float @llvm.amdgcn.exp2.f32(float %969), !dbg !29
  %971 = insertelement <4 x float> %968, float %970, i64 1, !dbg !30
  %local.reduce94 = fadd float %970, %local.reduce90, !dbg !31, !frisk.local_reduction !5
  %972 = extractelement <4 x float> %798, i64 2, !dbg !29
  %973 = tail call float @llvm.amdgcn.exp2.f32(float %972), !dbg !29
  %974 = insertelement <4 x float> %971, float %973, i64 2, !dbg !30
  %local.reduce95 = fadd float %973, %local.reduce91, !dbg !31, !frisk.local_reduction !5
  %975 = extractelement <4 x float> %798, i64 3, !dbg !29
  %976 = tail call float @llvm.amdgcn.exp2.f32(float %975), !dbg !29
  %977 = insertelement <4 x float> %974, float %976, i64 3, !dbg !30
  %math.result246 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %977) #8, !dbg !30, !frisk.register_lifetimes !5
  %978 = fptrunc <4 x float> %math.result246 to <4 x half>, !dbg !30
  %math.packed247 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %978) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce96 = fadd float %976, %local.reduce92, !dbg !31, !frisk.local_reduction !5
  %979 = extractelement <4 x float> %799, i64 0, !dbg !29
  %980 = tail call float @llvm.amdgcn.exp2.f32(float %979), !dbg !29
  %981 = insertelement <4 x float> poison, float %980, i64 0, !dbg !30
  %local.reduce97 = fadd float %980, %local.reduce93, !dbg !31, !frisk.local_reduction !5
  %982 = extractelement <4 x float> %799, i64 1, !dbg !29
  %983 = tail call float @llvm.amdgcn.exp2.f32(float %982), !dbg !29
  %984 = insertelement <4 x float> %981, float %983, i64 1, !dbg !30
  %local.reduce98 = fadd float %983, %local.reduce94, !dbg !31, !frisk.local_reduction !5
  %985 = extractelement <4 x float> %799, i64 2, !dbg !29
  %986 = tail call float @llvm.amdgcn.exp2.f32(float %985), !dbg !29
  %987 = insertelement <4 x float> %984, float %986, i64 2, !dbg !30
  %local.reduce99 = fadd float %986, %local.reduce95, !dbg !31, !frisk.local_reduction !5
  %988 = extractelement <4 x float> %799, i64 3, !dbg !29
  %989 = tail call float @llvm.amdgcn.exp2.f32(float %988), !dbg !29
  %990 = insertelement <4 x float> %987, float %989, i64 3, !dbg !30
  %math.result248 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %990) #8, !dbg !30, !frisk.register_lifetimes !5
  %991 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !30
  %math.packed249 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %991) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce100 = fadd float %989, %local.reduce96, !dbg !31, !frisk.local_reduction !5
  %992 = extractelement <4 x float> %800, i64 0, !dbg !29
  %993 = tail call float @llvm.amdgcn.exp2.f32(float %992), !dbg !29
  %994 = insertelement <4 x float> poison, float %993, i64 0, !dbg !30
  %local.reduce101 = fadd float %993, %local.reduce97, !dbg !31, !frisk.local_reduction !5
  %995 = extractelement <4 x float> %800, i64 1, !dbg !29
  %996 = tail call float @llvm.amdgcn.exp2.f32(float %995), !dbg !29
  %997 = insertelement <4 x float> %994, float %996, i64 1, !dbg !30
  %local.reduce102 = fadd float %996, %local.reduce98, !dbg !31, !frisk.local_reduction !5
  %998 = extractelement <4 x float> %800, i64 2, !dbg !29
  %999 = tail call float @llvm.amdgcn.exp2.f32(float %998), !dbg !29
  %1000 = insertelement <4 x float> %997, float %999, i64 2, !dbg !30
  %local.reduce103 = fadd float %999, %local.reduce99, !dbg !31, !frisk.local_reduction !5
  %1001 = extractelement <4 x float> %800, i64 3, !dbg !29
  %1002 = tail call float @llvm.amdgcn.exp2.f32(float %1001), !dbg !29
  %1003 = insertelement <4 x float> %1000, float %1002, i64 3, !dbg !30
  %math.result250 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1003) #8, !dbg !30, !frisk.register_lifetimes !5
  %1004 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !30
  %math.packed251 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1004) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce104 = fadd float %1002, %local.reduce100, !dbg !31, !frisk.local_reduction !5
  %1005 = extractelement <4 x float> %801, i64 0, !dbg !29
  %1006 = tail call float @llvm.amdgcn.exp2.f32(float %1005), !dbg !29
  %1007 = insertelement <4 x float> poison, float %1006, i64 0, !dbg !30
  %local.reduce105 = fadd float %1006, %local.reduce101, !dbg !31, !frisk.local_reduction !5
  %1008 = extractelement <4 x float> %801, i64 1, !dbg !29
  %1009 = tail call float @llvm.amdgcn.exp2.f32(float %1008), !dbg !29
  %1010 = insertelement <4 x float> %1007, float %1009, i64 1, !dbg !30
  %local.reduce106 = fadd float %1009, %local.reduce102, !dbg !31, !frisk.local_reduction !5
  %1011 = extractelement <4 x float> %801, i64 2, !dbg !29
  %1012 = tail call float @llvm.amdgcn.exp2.f32(float %1011), !dbg !29
  %1013 = insertelement <4 x float> %1010, float %1012, i64 2, !dbg !30
  %local.reduce107 = fadd float %1012, %local.reduce103, !dbg !31, !frisk.local_reduction !5
  %1014 = extractelement <4 x float> %801, i64 3, !dbg !29
  %1015 = tail call float @llvm.amdgcn.exp2.f32(float %1014), !dbg !29
  %1016 = insertelement <4 x float> %1013, float %1015, i64 3, !dbg !30
  %math.result252 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1016) #8, !dbg !30, !frisk.register_lifetimes !5
  %1017 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !30
  %math.packed253 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1017) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce108 = fadd float %1015, %local.reduce104, !dbg !31, !frisk.local_reduction !5
  %1018 = extractelement <4 x float> %802, i64 0, !dbg !29
  %1019 = tail call float @llvm.amdgcn.exp2.f32(float %1018), !dbg !29
  %1020 = insertelement <4 x float> poison, float %1019, i64 0, !dbg !30
  %local.reduce109 = fadd float %1019, %local.reduce105, !dbg !31, !frisk.local_reduction !5
  %1021 = extractelement <4 x float> %802, i64 1, !dbg !29
  %1022 = tail call float @llvm.amdgcn.exp2.f32(float %1021), !dbg !29
  %1023 = insertelement <4 x float> %1020, float %1022, i64 1, !dbg !30
  %local.reduce110 = fadd float %1022, %local.reduce106, !dbg !31, !frisk.local_reduction !5
  %1024 = extractelement <4 x float> %802, i64 2, !dbg !29
  %1025 = tail call float @llvm.amdgcn.exp2.f32(float %1024), !dbg !29
  %1026 = insertelement <4 x float> %1023, float %1025, i64 2, !dbg !30
  %local.reduce111 = fadd float %1025, %local.reduce107, !dbg !31, !frisk.local_reduction !5
  %1027 = extractelement <4 x float> %802, i64 3, !dbg !29
  %1028 = tail call float @llvm.amdgcn.exp2.f32(float %1027), !dbg !29
  %1029 = insertelement <4 x float> %1026, float %1028, i64 3, !dbg !30
  %math.result254 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1029) #8, !dbg !30, !frisk.register_lifetimes !5
  %1030 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !30
  %math.packed255 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1030) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce112 = fadd float %1028, %local.reduce108, !dbg !31, !frisk.local_reduction !5
  %1031 = extractelement <4 x float> %803, i64 0, !dbg !29
  %1032 = tail call float @llvm.amdgcn.exp2.f32(float %1031), !dbg !29
  %1033 = insertelement <4 x float> poison, float %1032, i64 0, !dbg !30
  %local.reduce113 = fadd float %1032, %local.reduce109, !dbg !31, !frisk.local_reduction !5
  %1034 = extractelement <4 x float> %803, i64 1, !dbg !29
  %1035 = tail call float @llvm.amdgcn.exp2.f32(float %1034), !dbg !29
  %1036 = insertelement <4 x float> %1033, float %1035, i64 1, !dbg !30
  %local.reduce114 = fadd float %1035, %local.reduce110, !dbg !31, !frisk.local_reduction !5
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !31, !frisk.local_reduction !5
  %1037 = extractelement <4 x float> %803, i64 2, !dbg !29
  %1038 = tail call float @llvm.amdgcn.exp2.f32(float %1037), !dbg !29
  %1039 = insertelement <4 x float> %1036, float %1038, i64 2, !dbg !30
  %local.reduce115 = fadd float %1038, %local.reduce111, !dbg !31, !frisk.local_reduction !5
  %1040 = extractelement <4 x float> %803, i64 3, !dbg !29
  %1041 = tail call float @llvm.amdgcn.exp2.f32(float %1040), !dbg !29
  %1042 = insertelement <4 x float> %1039, float %1041, i64 3, !dbg !30
  %math.result256 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1042) #8, !dbg !30, !frisk.register_lifetimes !5
  %1043 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !30
  %math.packed257 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1043) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce116 = fadd float %1041, %local.reduce112, !dbg !31, !frisk.local_reduction !5
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !31, !frisk.local_reduction !5
  %1044 = fadd float %local.reduce117, %local.reduce118, !dbg !31, !frisk.local_reduction !5
  %1045 = bitcast float %939 to i32, !dbg !31
  %1046 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1045), !dbg !31
  %1047 = bitcast i32 %1046 to float, !dbg !31
  %1048 = fadd float %939, %1047, !dbg !31
  %1049 = bitcast float %1048 to i32, !dbg !31
  %1050 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1049), !dbg !31
  %1051 = bitcast i32 %1050 to float, !dbg !31
  %1052 = fadd float %1048, %1051, !dbg !31
  %1053 = fadd float %831, %1052, !dbg !31
  %1054 = bitcast float %1044 to i32, !dbg !31
  %1055 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1054), !dbg !31
  %1056 = bitcast i32 %1055 to float, !dbg !31
  %1057 = fadd float %1044, %1056, !dbg !31
  %1058 = bitcast float %1057 to i32, !dbg !31
  %1059 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1058), !dbg !31
  %1060 = bitcast i32 %1059 to float, !dbg !31
  %1061 = fadd float %1057, %1060, !dbg !31
  %1062 = fadd float %834, %1061, !dbg !31
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local836 = shl nuw nsw i32 %address.thread798, 3, !dbg !14
  %address.local837 = and i32 %address.local836, 504, !dbg !14
  %address.local838 = zext nneg i32 %address.local837 to i64, !dbg !14
  %address.local839 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !14
  %1063 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local839, i32 0) #10, !dbg !14
  %1064 = bitcast <4 x i32> %1063 to <8 x half>, !dbg !14
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local839, i64 1024, !dbg !14
  %1065 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #10, !dbg !14
  %1066 = bitcast <4 x i32> %1065 to <8 x half>, !dbg !14
  %address.local843 = getelementptr i8, ptr addrspace(3) %address.local839, i64 2048, !dbg !14
  %1067 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local843, i32 0) #10, !dbg !14
  %1068 = bitcast <4 x i32> %1067 to <8 x half>, !dbg !14
  %address.local845 = getelementptr i8, ptr addrspace(3) %address.local839, i64 3072, !dbg !14
  %1069 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local845, i32 0) #10, !dbg !14
  %1070 = bitcast <4 x i32> %1069 to <8 x half>, !dbg !14
  %address.local847 = getelementptr i8, ptr addrspace(3) %address.local839, i64 4096, !dbg !14
  %1071 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local847, i32 0) #10, !dbg !14
  %1072 = bitcast <4 x i32> %1071 to <8 x half>, !dbg !14
  %address.local849 = getelementptr i8, ptr addrspace(3) %address.local839, i64 5120, !dbg !14
  %1073 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local849, i32 0) #10, !dbg !14
  %1074 = bitcast <4 x i32> %1073 to <8 x half>, !dbg !14
  %address.local851 = getelementptr i8, ptr addrspace(3) %address.local839, i64 6144, !dbg !14
  %1075 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #10, !dbg !14
  %1076 = bitcast <4 x i32> %1075 to <8 x half>, !dbg !14
  %address.local853 = getelementptr i8, ptr addrspace(3) %address.local839, i64 7168, !dbg !14
  %1077 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local853, i32 0) #10, !dbg !14
  %1078 = bitcast <4 x i32> %1077 to <8 x half>, !dbg !14
  %1079 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1079, <4 x float> zeroinitializer) #8, !dbg !14
  %1081 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1081, <4 x float> %1080) #8, !dbg !14
  %1083 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1083, <4 x float> %1082) #8, !dbg !14
  %1085 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1085, <4 x float> %1084) #8, !dbg !14
  %1087 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1087, <4 x float> %1086) #8, !dbg !14
  %1089 = shufflevector <8 x half> %1074, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1089, <4 x float> %1088) #8, !dbg !14
  %1091 = shufflevector <8 x half> %1076, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1091, <4 x float> %1090) #8, !dbg !14
  %1093 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1093, <4 x float> %1092) #8, !dbg !14
  %1095 = extractelement <4 x float> %1094, i64 0, !dbg !14
  %1096 = extractelement <4 x float> %1094, i64 1, !dbg !14
  %1097 = extractelement <4 x float> %1094, i64 2, !dbg !14
  %1098 = extractelement <4 x float> %1094, i64 3, !dbg !14
  %1099 = select i1 %.not, float %1096, float %1095, !dbg !14
  %1100 = bitcast float %1099 to i32, !dbg !14
  %1101 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1100), !dbg !14
  %1102 = bitcast i32 %1101 to float, !dbg !14
  %1103 = select i1 %.not, float %1095, float %1102, !dbg !14
  %1104 = select i1 %.not, float %1102, float %1096, !dbg !14
  %1105 = select i1 %.not, float %1098, float %1097, !dbg !14
  %1106 = bitcast float %1105 to i32, !dbg !14
  %1107 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1106), !dbg !14
  %1108 = bitcast i32 %1107 to float, !dbg !14
  %1109 = select i1 %.not, float %1097, float %1108, !dbg !14
  %1110 = select i1 %.not, float %1108, float %1098, !dbg !14
  %1111 = select i1 %.not1, float %1109, float %1103, !dbg !14
  %1112 = bitcast float %1111 to i32, !dbg !14
  %1113 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1112), !dbg !14
  %1114 = bitcast i32 %1113 to float, !dbg !14
  %1115 = select i1 %.not1, float %1110, float %1104, !dbg !14
  %1116 = bitcast float %1115 to i32, !dbg !14
  %1117 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1116), !dbg !14
  %1118 = bitcast i32 %1117 to float, !dbg !14
  %1119 = insertelement <4 x float> poison, float %1103, i64 0, !dbg !14
  %1120 = insertelement <4 x float> %1119, float %1104, i64 1, !dbg !14
  %1121 = insertelement <4 x float> %1120, float %1114, i64 2, !dbg !14
  %1122 = insertelement <4 x float> %1121, float %1118, i64 3, !dbg !14
  %1123 = insertelement <4 x float> poison, float %1114, i64 0, !dbg !14
  %1124 = insertelement <4 x float> %1123, float %1118, i64 1, !dbg !14
  %1125 = insertelement <4 x float> %1124, float %1109, i64 2, !dbg !14
  %1126 = insertelement <4 x float> %1125, float %1110, i64 3, !dbg !14
  %1127 = select <4 x i1> %66, <4 x float> %1122, <4 x float> %1126, !dbg !14
  %1128 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1128, <4 x float> zeroinitializer) #8, !dbg !14
  %1130 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1131 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1130, <4 x float> %1129) #8, !dbg !14
  %1132 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1132, <4 x float> %1131) #8, !dbg !14
  %1134 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1135 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1134, <4 x float> %1133) #8, !dbg !14
  %1136 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1136, <4 x float> %1135) #8, !dbg !14
  %1138 = shufflevector <8 x half> %1074, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1139 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1138, <4 x float> %1137) #8, !dbg !14
  %1140 = shufflevector <8 x half> %1076, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1140, <4 x float> %1139) #8, !dbg !14
  %1142 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1142, <4 x float> %1141) #8, !dbg !14
  %1144 = extractelement <4 x float> %1143, i64 0, !dbg !14
  %1145 = extractelement <4 x float> %1143, i64 1, !dbg !14
  %1146 = extractelement <4 x float> %1143, i64 2, !dbg !14
  %1147 = extractelement <4 x float> %1143, i64 3, !dbg !14
  %1148 = select i1 %.not, float %1145, float %1144, !dbg !14
  %1149 = bitcast float %1148 to i32, !dbg !14
  %1150 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1149), !dbg !14
  %1151 = bitcast i32 %1150 to float, !dbg !14
  %1152 = select i1 %.not, float %1144, float %1151, !dbg !14
  %1153 = select i1 %.not, float %1151, float %1145, !dbg !14
  %1154 = select i1 %.not, float %1147, float %1146, !dbg !14
  %1155 = bitcast float %1154 to i32, !dbg !14
  %1156 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1155), !dbg !14
  %1157 = bitcast i32 %1156 to float, !dbg !14
  %1158 = select i1 %.not, float %1146, float %1157, !dbg !14
  %1159 = select i1 %.not, float %1157, float %1147, !dbg !14
  %1160 = select i1 %.not1, float %1158, float %1152, !dbg !14
  %1161 = bitcast float %1160 to i32, !dbg !14
  %1162 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1161), !dbg !14
  %1163 = bitcast i32 %1162 to float, !dbg !14
  %1164 = select i1 %.not1, float %1159, float %1153, !dbg !14
  %1165 = bitcast float %1164 to i32, !dbg !14
  %1166 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1165), !dbg !14
  %1167 = bitcast i32 %1166 to float, !dbg !14
  %1168 = insertelement <4 x float> poison, float %1152, i64 0, !dbg !14
  %1169 = insertelement <4 x float> %1168, float %1153, i64 1, !dbg !14
  %1170 = insertelement <4 x float> %1169, float %1163, i64 2, !dbg !14
  %1171 = insertelement <4 x float> %1170, float %1167, i64 3, !dbg !14
  %1172 = insertelement <4 x float> poison, float %1163, i64 0, !dbg !14
  %1173 = insertelement <4 x float> %1172, float %1167, i64 1, !dbg !14
  %1174 = insertelement <4 x float> %1173, float %1158, i64 2, !dbg !14
  %1175 = insertelement <4 x float> %1174, float %1159, i64 3, !dbg !14
  %1176 = select <4 x i1> %66, <4 x float> %1171, <4 x float> %1175, !dbg !14
  %1177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1079, <4 x float> zeroinitializer) #8, !dbg !14
  %1178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1081, <4 x float> %1177) #8, !dbg !14
  %1179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1083, <4 x float> %1178) #8, !dbg !14
  %1180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1085, <4 x float> %1179) #8, !dbg !14
  %1181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1087, <4 x float> %1180) #8, !dbg !14
  %1182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1089, <4 x float> %1181) #8, !dbg !14
  %1183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1091, <4 x float> %1182) #8, !dbg !14
  %1184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1093, <4 x float> %1183) #8, !dbg !14
  %1185 = extractelement <4 x float> %1184, i64 0, !dbg !14
  %1186 = extractelement <4 x float> %1184, i64 1, !dbg !14
  %1187 = extractelement <4 x float> %1184, i64 2, !dbg !14
  %1188 = extractelement <4 x float> %1184, i64 3, !dbg !14
  %1189 = select i1 %.not, float %1186, float %1185, !dbg !14
  %1190 = bitcast float %1189 to i32, !dbg !14
  %1191 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1190), !dbg !14
  %1192 = bitcast i32 %1191 to float, !dbg !14
  %1193 = select i1 %.not, float %1185, float %1192, !dbg !14
  %1194 = select i1 %.not, float %1192, float %1186, !dbg !14
  %1195 = select i1 %.not, float %1188, float %1187, !dbg !14
  %1196 = bitcast float %1195 to i32, !dbg !14
  %1197 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1196), !dbg !14
  %1198 = bitcast i32 %1197 to float, !dbg !14
  %1199 = select i1 %.not, float %1187, float %1198, !dbg !14
  %1200 = select i1 %.not, float %1198, float %1188, !dbg !14
  %1201 = select i1 %.not1, float %1199, float %1193, !dbg !14
  %1202 = bitcast float %1201 to i32, !dbg !14
  %1203 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1202), !dbg !14
  %1204 = bitcast i32 %1203 to float, !dbg !14
  %1205 = select i1 %.not1, float %1200, float %1194, !dbg !14
  %1206 = bitcast float %1205 to i32, !dbg !14
  %1207 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1206), !dbg !14
  %1208 = bitcast i32 %1207 to float, !dbg !14
  %1209 = insertelement <4 x float> poison, float %1193, i64 0, !dbg !14
  %1210 = insertelement <4 x float> %1209, float %1194, i64 1, !dbg !14
  %1211 = insertelement <4 x float> %1210, float %1204, i64 2, !dbg !14
  %1212 = insertelement <4 x float> %1211, float %1208, i64 3, !dbg !14
  %1213 = insertelement <4 x float> poison, float %1204, i64 0, !dbg !14
  %1214 = insertelement <4 x float> %1213, float %1208, i64 1, !dbg !14
  %1215 = insertelement <4 x float> %1214, float %1199, i64 2, !dbg !14
  %1216 = insertelement <4 x float> %1215, float %1200, i64 3, !dbg !14
  %1217 = select <4 x i1> %66, <4 x float> %1212, <4 x float> %1216, !dbg !14
  %1218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1128, <4 x float> zeroinitializer) #8, !dbg !14
  %1219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1130, <4 x float> %1218) #8, !dbg !14
  %1220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1132, <4 x float> %1219) #8, !dbg !14
  %1221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1134, <4 x float> %1220) #8, !dbg !14
  %1222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1136, <4 x float> %1221) #8, !dbg !14
  %1223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1138, <4 x float> %1222) #8, !dbg !14
  %1224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1140, <4 x float> %1223) #8, !dbg !14
  %1225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1142, <4 x float> %1224) #8, !dbg !14
  %1226 = extractelement <4 x float> %1225, i64 0, !dbg !14
  %1227 = extractelement <4 x float> %1225, i64 1, !dbg !14
  %1228 = extractelement <4 x float> %1225, i64 2, !dbg !14
  %1229 = extractelement <4 x float> %1225, i64 3, !dbg !14
  %1230 = select i1 %.not, float %1227, float %1226, !dbg !14
  %1231 = bitcast float %1230 to i32, !dbg !14
  %1232 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1231), !dbg !14
  %1233 = bitcast i32 %1232 to float, !dbg !14
  %1234 = select i1 %.not, float %1226, float %1233, !dbg !14
  %1235 = select i1 %.not, float %1233, float %1227, !dbg !14
  %1236 = select i1 %.not, float %1229, float %1228, !dbg !14
  %1237 = bitcast float %1236 to i32, !dbg !14
  %1238 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1237), !dbg !14
  %1239 = bitcast i32 %1238 to float, !dbg !14
  %1240 = select i1 %.not, float %1228, float %1239, !dbg !14
  %1241 = select i1 %.not, float %1239, float %1229, !dbg !14
  %1242 = select i1 %.not1, float %1240, float %1234, !dbg !14
  %1243 = bitcast float %1242 to i32, !dbg !14
  %1244 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1243), !dbg !14
  %1245 = bitcast i32 %1244 to float, !dbg !14
  %1246 = select i1 %.not1, float %1241, float %1235, !dbg !14
  %1247 = bitcast float %1246 to i32, !dbg !14
  %1248 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1247), !dbg !14
  %1249 = bitcast i32 %1248 to float, !dbg !14
  %1250 = insertelement <4 x float> poison, float %1234, i64 0, !dbg !14
  %1251 = insertelement <4 x float> %1250, float %1235, i64 1, !dbg !14
  %1252 = insertelement <4 x float> %1251, float %1245, i64 2, !dbg !14
  %1253 = insertelement <4 x float> %1252, float %1249, i64 3, !dbg !14
  %1254 = insertelement <4 x float> poison, float %1245, i64 0, !dbg !14
  %1255 = insertelement <4 x float> %1254, float %1249, i64 1, !dbg !14
  %1256 = insertelement <4 x float> %1255, float %1240, i64 2, !dbg !14
  %1257 = insertelement <4 x float> %1256, float %1241, i64 3, !dbg !14
  %1258 = select <4 x i1> %66, <4 x float> %1253, <4 x float> %1257, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %39, label %1259, label %1260, !dbg !32

1259:                                             ; preds = %815
  %address.thread854 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local855 = lshr i32 %address.thread854, 1, !dbg !13
  %address.local856 = and i32 %address.local855, 96, !dbg !13
  %address.local857 = and i32 %address.thread854, 15, !dbg !13
  %address.local858 = or disjoint i32 %address.local856, %address.local857, !dbg !13
  %address.local859 = zext nneg i32 %address.local858 to i64, !dbg !13
  %address.local860 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !13
  store float %812, ptr addrspace(3) %address.local860, align 4, !dbg !32
  %address.local862 = getelementptr i8, ptr addrspace(3) %address.local860, i64 64, !dbg !13
  store float %813, ptr addrspace(3) %address.local862, align 4, !dbg !32
  br label %1260, !dbg !32

1260:                                             ; preds = %1259, %815
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.thread863 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %1261 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %1262 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.local877 = zext nneg i32 %address.thread863 to i64, !dbg !10
  %address.local878 = shl nuw nsw i64 %address.local877, 3, !dbg !9
  %address.local879 = and i64 %address.local878, 1536, !dbg !9
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local879, !dbg !9
  %address.local881 = ptrtoint ptr addrspace(3) %address.local880 to i64, !dbg !9
  %address.local882 = trunc i64 %address.local881 to i32, !dbg !9
  %1263 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local882), !dbg !28
  %address.local883 = shl nuw nsw i64 %address.local877, 6, !dbg !28
  %address.local884 = and i64 %address.local883, 16128, !dbg !28
  %address.local886 = and i64 %address.local878, 24, !dbg !28
  %1264 = or disjoint i64 %address.local886, %.idx5558, !dbg !28
  %1265 = shl i64 %1264, 1, !dbg !28
  %1266 = or disjoint i64 %1265, %address.local884, !dbg !28
  %1267 = trunc i64 %1266 to i32, !dbg !28
  %dma.buffer.offset1767 = or i32 %1267, 64, !dbg !28
  %dma.buffer.lds1769 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1263), !dbg !28
  %1268 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1769, i32 %dma.buffer.offset1767, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %address.local889 = or disjoint i64 %address.local877, 256, !dbg !9
  %address.local890 = shl nuw nsw i64 %address.local889, 3, !dbg !9
  %address.local891 = and i64 %address.local890, 3584, !dbg !9
  %address.local892 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local891, !dbg !9
  %address.local893 = ptrtoint ptr addrspace(3) %address.local892 to i64, !dbg !9
  %address.local894 = trunc i64 %address.local893 to i32, !dbg !9
  %1269 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !28
  %address.local895 = shl nuw nsw i64 %address.local889, 6, !dbg !28
  %address.local896 = and i64 %address.local895, 32512, !dbg !28
  %1270 = shl i64 %.idx5558, 1, !dbg !28
  %address.local900.idx = shl nuw nsw i64 %address.local886, 1, !dbg !28
  %1271 = or disjoint i64 %address.local896, %address.local900.idx, !dbg !28
  %1272 = or disjoint i64 %1271, %1270, !dbg !28
  %1273 = trunc i64 %1272 to i32, !dbg !28
  %dma.buffer.offset1764 = or i32 %1273, 64, !dbg !28
  %dma.buffer.lds1766 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1269), !dbg !28
  %1274 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1766, i32 %dma.buffer.offset1764, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %1275 = extractvalue [2 x <8 x float>] %77, 0, !dbg !32
  %1276 = shufflevector <8 x float> %1275, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1277 = insertelement <4 x float> poison, float %1261, i64 0, !dbg !32
  %1278 = shufflevector <4 x float> %1277, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1279 = fmul <4 x float> %1276, %1278, !dbg !32
  %1280 = fadd <4 x float> %1127, %1279, !dbg !33
  %1281 = shufflevector <8 x float> %1275, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1282 = fmul <4 x float> %1281, %1278, !dbg !32
  %1283 = fadd <4 x float> %1176, %1282, !dbg !33
  %1284 = extractvalue [2 x <8 x float>] %77, 1, !dbg !32
  %1285 = shufflevector <8 x float> %1284, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1286 = insertelement <4 x float> poison, float %1262, i64 0, !dbg !32
  %1287 = shufflevector <4 x float> %1286, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1288 = fmul <4 x float> %1285, %1287, !dbg !32
  %1289 = fadd <4 x float> %1217, %1288, !dbg !33
  %1290 = shufflevector <8 x float> %1284, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1291 = fmul <4 x float> %1290, %1287, !dbg !32
  %1292 = fadd <4 x float> %1258, %1291, !dbg !33
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local901 = shl nuw nsw i32 %address.thread863, 3, !dbg !14
  %address.local902 = and i32 %address.local901, 504, !dbg !14
  %address.local903 = zext nneg i32 %address.local902 to i64, !dbg !14
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %1293 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #10, !dbg !14
  %1294 = bitcast <4 x i32> %1293 to <8 x half>, !dbg !14
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local904, i64 1024, !dbg !14
  %1295 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #10, !dbg !14
  %1296 = bitcast <4 x i32> %1295 to <8 x half>, !dbg !14
  %address.local908 = getelementptr i8, ptr addrspace(3) %address.local904, i64 2048, !dbg !14
  %1297 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local908, i32 0) #10, !dbg !14
  %1298 = bitcast <4 x i32> %1297 to <8 x half>, !dbg !14
  %address.local910 = getelementptr i8, ptr addrspace(3) %address.local904, i64 3072, !dbg !14
  %1299 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local910, i32 0) #10, !dbg !14
  %1300 = bitcast <4 x i32> %1299 to <8 x half>, !dbg !14
  %address.local912 = getelementptr i8, ptr addrspace(3) %address.local904, i64 4096, !dbg !14
  %1301 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local912, i32 0) #10, !dbg !14
  %1302 = bitcast <4 x i32> %1301 to <8 x half>, !dbg !14
  %address.local914 = getelementptr i8, ptr addrspace(3) %address.local904, i64 5120, !dbg !14
  %1303 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local914, i32 0) #10, !dbg !14
  %1304 = bitcast <4 x i32> %1303 to <8 x half>, !dbg !14
  %address.local916 = getelementptr i8, ptr addrspace(3) %address.local904, i64 6144, !dbg !14
  %1305 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local916, i32 0) #10, !dbg !14
  %1306 = bitcast <4 x i32> %1305 to <8 x half>, !dbg !14
  %address.local918 = getelementptr i8, ptr addrspace(3) %address.local904, i64 7168, !dbg !14
  %1307 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local918, i32 0) #10, !dbg !14
  %1308 = bitcast <4 x i32> %1307 to <8 x half>, !dbg !14
  %1309 = shufflevector <8 x half> %1294, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1309, <4 x float> zeroinitializer) #8, !dbg !14
  %1311 = shufflevector <8 x half> %1296, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1311, <4 x float> %1310) #8, !dbg !14
  %1313 = shufflevector <8 x half> %1298, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1313, <4 x float> %1312) #8, !dbg !14
  %1315 = shufflevector <8 x half> %1300, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1315, <4 x float> %1314) #8, !dbg !14
  %1317 = shufflevector <8 x half> %1302, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1317, <4 x float> %1316) #8, !dbg !14
  %1319 = shufflevector <8 x half> %1304, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1319, <4 x float> %1318) #8, !dbg !14
  %1321 = shufflevector <8 x half> %1306, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1321, <4 x float> %1320) #8, !dbg !14
  %1323 = shufflevector <8 x half> %1308, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1323, <4 x float> %1322) #8, !dbg !14
  %1325 = extractelement <4 x float> %1324, i64 0, !dbg !14
  %1326 = extractelement <4 x float> %1324, i64 1, !dbg !14
  %1327 = extractelement <4 x float> %1324, i64 2, !dbg !14
  %1328 = extractelement <4 x float> %1324, i64 3, !dbg !14
  %1329 = select i1 %.not, float %1326, float %1325, !dbg !14
  %1330 = bitcast float %1329 to i32, !dbg !14
  %1331 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1330), !dbg !14
  %1332 = bitcast i32 %1331 to float, !dbg !14
  %1333 = select i1 %.not, float %1325, float %1332, !dbg !14
  %1334 = select i1 %.not, float %1332, float %1326, !dbg !14
  %1335 = select i1 %.not, float %1328, float %1327, !dbg !14
  %1336 = bitcast float %1335 to i32, !dbg !14
  %1337 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1336), !dbg !14
  %1338 = bitcast i32 %1337 to float, !dbg !14
  %1339 = select i1 %.not, float %1327, float %1338, !dbg !14
  %1340 = select i1 %.not, float %1338, float %1328, !dbg !14
  %1341 = select i1 %.not1, float %1339, float %1333, !dbg !14
  %1342 = bitcast float %1341 to i32, !dbg !14
  %1343 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1342), !dbg !14
  %1344 = bitcast i32 %1343 to float, !dbg !14
  %1345 = select i1 %.not1, float %1340, float %1334, !dbg !14
  %1346 = bitcast float %1345 to i32, !dbg !14
  %1347 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1346), !dbg !14
  %1348 = bitcast i32 %1347 to float, !dbg !14
  %1349 = insertelement <4 x float> poison, float %1333, i64 0, !dbg !14
  %1350 = insertelement <4 x float> %1349, float %1334, i64 1, !dbg !14
  %1351 = insertelement <4 x float> %1350, float %1344, i64 2, !dbg !14
  %1352 = insertelement <4 x float> %1351, float %1348, i64 3, !dbg !14
  %1353 = insertelement <4 x float> poison, float %1344, i64 0, !dbg !14
  %1354 = insertelement <4 x float> %1353, float %1348, i64 1, !dbg !14
  %1355 = insertelement <4 x float> %1354, float %1339, i64 2, !dbg !14
  %1356 = insertelement <4 x float> %1355, float %1340, i64 3, !dbg !14
  %1357 = select <4 x i1> %66, <4 x float> %1352, <4 x float> %1356, !dbg !14
  %1358 = shufflevector <8 x half> %1294, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1358, <4 x float> zeroinitializer) #8, !dbg !14
  %1360 = shufflevector <8 x half> %1296, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1360, <4 x float> %1359) #8, !dbg !14
  %1362 = shufflevector <8 x half> %1298, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1362, <4 x float> %1361) #8, !dbg !14
  %1364 = shufflevector <8 x half> %1300, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1364, <4 x float> %1363) #8, !dbg !14
  %1366 = shufflevector <8 x half> %1302, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1366, <4 x float> %1365) #8, !dbg !14
  %1368 = shufflevector <8 x half> %1304, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1368, <4 x float> %1367) #8, !dbg !14
  %1370 = shufflevector <8 x half> %1306, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1371 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1370, <4 x float> %1369) #8, !dbg !14
  %1372 = shufflevector <8 x half> %1308, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1372, <4 x float> %1371) #8, !dbg !14
  %1374 = extractelement <4 x float> %1373, i64 0, !dbg !14
  %1375 = extractelement <4 x float> %1373, i64 1, !dbg !14
  %1376 = extractelement <4 x float> %1373, i64 2, !dbg !14
  %1377 = extractelement <4 x float> %1373, i64 3, !dbg !14
  %1378 = select i1 %.not, float %1375, float %1374, !dbg !14
  %1379 = bitcast float %1378 to i32, !dbg !14
  %1380 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1379), !dbg !14
  %1381 = bitcast i32 %1380 to float, !dbg !14
  %1382 = select i1 %.not, float %1374, float %1381, !dbg !14
  %1383 = select i1 %.not, float %1381, float %1375, !dbg !14
  %1384 = select i1 %.not, float %1377, float %1376, !dbg !14
  %1385 = bitcast float %1384 to i32, !dbg !14
  %1386 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1385), !dbg !14
  %1387 = bitcast i32 %1386 to float, !dbg !14
  %1388 = select i1 %.not, float %1376, float %1387, !dbg !14
  %1389 = select i1 %.not, float %1387, float %1377, !dbg !14
  %1390 = select i1 %.not1, float %1388, float %1382, !dbg !14
  %1391 = bitcast float %1390 to i32, !dbg !14
  %1392 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1391), !dbg !14
  %1393 = bitcast i32 %1392 to float, !dbg !14
  %1394 = select i1 %.not1, float %1389, float %1383, !dbg !14
  %1395 = bitcast float %1394 to i32, !dbg !14
  %1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1395), !dbg !14
  %1397 = bitcast i32 %1396 to float, !dbg !14
  %1398 = insertelement <4 x float> poison, float %1382, i64 0, !dbg !14
  %1399 = insertelement <4 x float> %1398, float %1383, i64 1, !dbg !14
  %1400 = insertelement <4 x float> %1399, float %1393, i64 2, !dbg !14
  %1401 = insertelement <4 x float> %1400, float %1397, i64 3, !dbg !14
  %1402 = insertelement <4 x float> poison, float %1393, i64 0, !dbg !14
  %1403 = insertelement <4 x float> %1402, float %1397, i64 1, !dbg !14
  %1404 = insertelement <4 x float> %1403, float %1388, i64 2, !dbg !14
  %1405 = insertelement <4 x float> %1404, float %1389, i64 3, !dbg !14
  %1406 = select <4 x i1> %66, <4 x float> %1401, <4 x float> %1405, !dbg !14
  %1407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1309, <4 x float> zeroinitializer) #8, !dbg !14
  %1408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1311, <4 x float> %1407) #8, !dbg !14
  %1409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1313, <4 x float> %1408) #8, !dbg !14
  %1410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1315, <4 x float> %1409) #8, !dbg !14
  %1411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1317, <4 x float> %1410) #8, !dbg !14
  %1412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1319, <4 x float> %1411) #8, !dbg !14
  %1413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1321, <4 x float> %1412) #8, !dbg !14
  %1414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1323, <4 x float> %1413) #8, !dbg !14
  %1415 = extractelement <4 x float> %1414, i64 0, !dbg !14
  %1416 = extractelement <4 x float> %1414, i64 1, !dbg !14
  %1417 = extractelement <4 x float> %1414, i64 2, !dbg !14
  %1418 = extractelement <4 x float> %1414, i64 3, !dbg !14
  %1419 = select i1 %.not, float %1416, float %1415, !dbg !14
  %1420 = bitcast float %1419 to i32, !dbg !14
  %1421 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1420), !dbg !14
  %1422 = bitcast i32 %1421 to float, !dbg !14
  %1423 = select i1 %.not, float %1415, float %1422, !dbg !14
  %1424 = select i1 %.not, float %1422, float %1416, !dbg !14
  %1425 = select i1 %.not, float %1418, float %1417, !dbg !14
  %1426 = bitcast float %1425 to i32, !dbg !14
  %1427 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1426), !dbg !14
  %1428 = bitcast i32 %1427 to float, !dbg !14
  %1429 = select i1 %.not, float %1417, float %1428, !dbg !14
  %1430 = select i1 %.not, float %1428, float %1418, !dbg !14
  %1431 = select i1 %.not1, float %1429, float %1423, !dbg !14
  %1432 = bitcast float %1431 to i32, !dbg !14
  %1433 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1432), !dbg !14
  %1434 = bitcast i32 %1433 to float, !dbg !14
  %1435 = select i1 %.not1, float %1430, float %1424, !dbg !14
  %1436 = bitcast float %1435 to i32, !dbg !14
  %1437 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1436), !dbg !14
  %1438 = bitcast i32 %1437 to float, !dbg !14
  %1439 = insertelement <4 x float> poison, float %1423, i64 0, !dbg !14
  %1440 = insertelement <4 x float> %1439, float %1424, i64 1, !dbg !14
  %1441 = insertelement <4 x float> %1440, float %1434, i64 2, !dbg !14
  %1442 = insertelement <4 x float> %1441, float %1438, i64 3, !dbg !14
  %1443 = insertelement <4 x float> poison, float %1434, i64 0, !dbg !14
  %1444 = insertelement <4 x float> %1443, float %1438, i64 1, !dbg !14
  %1445 = insertelement <4 x float> %1444, float %1429, i64 2, !dbg !14
  %1446 = insertelement <4 x float> %1445, float %1430, i64 3, !dbg !14
  %1447 = select <4 x i1> %66, <4 x float> %1442, <4 x float> %1446, !dbg !14
  %1448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1358, <4 x float> zeroinitializer) #8, !dbg !14
  %1449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1360, <4 x float> %1448) #8, !dbg !14
  %1450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1362, <4 x float> %1449) #8, !dbg !14
  %1451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1364, <4 x float> %1450) #8, !dbg !14
  %1452 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1366, <4 x float> %1451) #8, !dbg !14
  %1453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1368, <4 x float> %1452) #8, !dbg !14
  %1454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1370, <4 x float> %1453) #8, !dbg !14
  %1455 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1372, <4 x float> %1454) #8, !dbg !14
  %1456 = extractelement <4 x float> %1455, i64 0, !dbg !14
  %1457 = extractelement <4 x float> %1455, i64 1, !dbg !14
  %1458 = extractelement <4 x float> %1455, i64 2, !dbg !14
  %1459 = extractelement <4 x float> %1455, i64 3, !dbg !14
  %1460 = select i1 %.not, float %1457, float %1456, !dbg !14
  %1461 = bitcast float %1460 to i32, !dbg !14
  %1462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1461), !dbg !14
  %1463 = bitcast i32 %1462 to float, !dbg !14
  %1464 = select i1 %.not, float %1456, float %1463, !dbg !14
  %1465 = select i1 %.not, float %1463, float %1457, !dbg !14
  %1466 = select i1 %.not, float %1459, float %1458, !dbg !14
  %1467 = bitcast float %1466 to i32, !dbg !14
  %1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1467), !dbg !14
  %1469 = bitcast i32 %1468 to float, !dbg !14
  %1470 = select i1 %.not, float %1458, float %1469, !dbg !14
  %1471 = select i1 %.not, float %1469, float %1459, !dbg !14
  %1472 = select i1 %.not1, float %1470, float %1464, !dbg !14
  %1473 = bitcast float %1472 to i32, !dbg !14
  %1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1473), !dbg !14
  %1475 = bitcast i32 %1474 to float, !dbg !14
  %1476 = select i1 %.not1, float %1471, float %1465, !dbg !14
  %1477 = bitcast float %1476 to i32, !dbg !14
  %1478 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1477), !dbg !14
  %1479 = bitcast i32 %1478 to float, !dbg !14
  %1480 = insertelement <4 x float> poison, float %1464, i64 0, !dbg !14
  %1481 = insertelement <4 x float> %1480, float %1465, i64 1, !dbg !14
  %1482 = insertelement <4 x float> %1481, float %1475, i64 2, !dbg !14
  %1483 = insertelement <4 x float> %1482, float %1479, i64 3, !dbg !14
  %1484 = insertelement <4 x float> poison, float %1475, i64 0, !dbg !14
  %1485 = insertelement <4 x float> %1484, float %1479, i64 1, !dbg !14
  %1486 = insertelement <4 x float> %1485, float %1470, i64 2, !dbg !14
  %1487 = insertelement <4 x float> %1486, float %1471, i64 3, !dbg !14
  %1488 = select <4 x i1> %66, <4 x float> %1483, <4 x float> %1487, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %39, label %1489, label %1490, !dbg !32

1489:                                             ; preds = %1260
  %address.thread919 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local920 = lshr i32 %address.thread919, 1, !dbg !13
  %address.local921 = and i32 %address.local920, 96, !dbg !13
  %address.local922 = and i32 %address.thread919, 15, !dbg !13
  %address.local923 = or disjoint i32 %address.local921, %address.local922, !dbg !13
  %address.local924 = zext nneg i32 %address.local923 to i64, !dbg !13
  %address.local925 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !13
  store float %812, ptr addrspace(3) %address.local925, align 4, !dbg !32
  %address.local927 = getelementptr i8, ptr addrspace(3) %address.local925, i64 64, !dbg !13
  store float %813, ptr addrspace(3) %address.local927, align 4, !dbg !32
  br label %1490, !dbg !32

1490:                                             ; preds = %1489, %1260
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.thread928 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %1491 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %1492 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.local942 = zext nneg i32 %address.thread928 to i64, !dbg !10
  %address.local943 = shl nuw nsw i64 %address.local942, 3, !dbg !9
  %address.local944 = and i64 %address.local943, 1536, !dbg !9
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local944, !dbg !9
  %address.local946 = ptrtoint ptr addrspace(3) %address.local945 to i64, !dbg !9
  %address.local947 = trunc i64 %address.local946 to i32, !dbg !9
  %1493 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local947), !dbg !28
  %address.local948 = shl nuw nsw i64 %address.local942, 6, !dbg !28
  %address.local949 = and i64 %address.local948, 16128, !dbg !28
  %address.local951 = and i64 %address.local943, 24, !dbg !28
  %1494 = or disjoint i64 %address.local951, %.idx5558, !dbg !28
  %1495 = shl i64 %1494, 1, !dbg !28
  %1496 = or disjoint i64 %1495, %address.local949, !dbg !28
  %1497 = trunc i64 %1496 to i32, !dbg !28
  %dma.buffer.offset1761 = or i32 %1497, 128, !dbg !28
  %dma.buffer.lds1763 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1493), !dbg !28
  %1498 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1763, i32 %dma.buffer.offset1761, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %address.local954 = or disjoint i64 %address.local942, 256, !dbg !9
  %address.local955 = shl nuw nsw i64 %address.local954, 3, !dbg !9
  %address.local956 = and i64 %address.local955, 3584, !dbg !9
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local956, !dbg !9
  %address.local958 = ptrtoint ptr addrspace(3) %address.local957 to i64, !dbg !9
  %address.local959 = trunc i64 %address.local958 to i32, !dbg !9
  %1499 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local959), !dbg !28
  %address.local960 = shl nuw nsw i64 %address.local954, 6, !dbg !28
  %address.local961 = and i64 %address.local960, 32512, !dbg !28
  %address.local965.idx = shl nuw nsw i64 %address.local951, 1, !dbg !28
  %1500 = or disjoint i64 %address.local961, %address.local965.idx, !dbg !28
  %1501 = or disjoint i64 %1500, %1270, !dbg !28
  %1502 = trunc i64 %1501 to i32, !dbg !28
  %dma.buffer.offset1758 = or i32 %1502, 128, !dbg !28
  %dma.buffer.lds1760 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1499), !dbg !28
  %1503 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1760, i32 %dma.buffer.offset1758, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %1504 = extractvalue [2 x <8 x float>] %74, 0, !dbg !32
  %1505 = shufflevector <8 x float> %1504, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1506 = insertelement <4 x float> poison, float %1491, i64 0, !dbg !32
  %1507 = shufflevector <4 x float> %1506, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1508 = fmul <4 x float> %1505, %1507, !dbg !32
  %1509 = fadd <4 x float> %1357, %1508, !dbg !33
  %1510 = shufflevector <8 x float> %1504, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1511 = fmul <4 x float> %1510, %1507, !dbg !32
  %1512 = fadd <4 x float> %1406, %1511, !dbg !33
  %1513 = extractvalue [2 x <8 x float>] %74, 1, !dbg !32
  %1514 = shufflevector <8 x float> %1513, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1515 = insertelement <4 x float> poison, float %1492, i64 0, !dbg !32
  %1516 = shufflevector <4 x float> %1515, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1517 = fmul <4 x float> %1514, %1516, !dbg !32
  %1518 = fadd <4 x float> %1447, %1517, !dbg !33
  %1519 = shufflevector <8 x float> %1513, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1520 = fmul <4 x float> %1519, %1516, !dbg !32
  %1521 = fadd <4 x float> %1488, %1520, !dbg !33
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local966 = shl nuw nsw i32 %address.thread928, 3, !dbg !14
  %address.local967 = and i32 %address.local966, 504, !dbg !14
  %address.local968 = zext nneg i32 %address.local967 to i64, !dbg !14
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !14
  %1522 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #10, !dbg !14
  %1523 = bitcast <4 x i32> %1522 to <8 x half>, !dbg !14
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local969, i64 1024, !dbg !14
  %1524 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #10, !dbg !14
  %1525 = bitcast <4 x i32> %1524 to <8 x half>, !dbg !14
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local969, i64 2048, !dbg !14
  %1526 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #10, !dbg !14
  %1527 = bitcast <4 x i32> %1526 to <8 x half>, !dbg !14
  %address.local975 = getelementptr i8, ptr addrspace(3) %address.local969, i64 3072, !dbg !14
  %1528 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #10, !dbg !14
  %1529 = bitcast <4 x i32> %1528 to <8 x half>, !dbg !14
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local969, i64 4096, !dbg !14
  %1530 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #10, !dbg !14
  %1531 = bitcast <4 x i32> %1530 to <8 x half>, !dbg !14
  %address.local979 = getelementptr i8, ptr addrspace(3) %address.local969, i64 5120, !dbg !14
  %1532 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #10, !dbg !14
  %1533 = bitcast <4 x i32> %1532 to <8 x half>, !dbg !14
  %address.local981 = getelementptr i8, ptr addrspace(3) %address.local969, i64 6144, !dbg !14
  %1534 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #10, !dbg !14
  %1535 = bitcast <4 x i32> %1534 to <8 x half>, !dbg !14
  %address.local983 = getelementptr i8, ptr addrspace(3) %address.local969, i64 7168, !dbg !14
  %1536 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #10, !dbg !14
  %1537 = bitcast <4 x i32> %1536 to <8 x half>, !dbg !14
  %1538 = shufflevector <8 x half> %1523, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1539 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1538, <4 x float> zeroinitializer) #8, !dbg !14
  %1540 = shufflevector <8 x half> %1525, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1540, <4 x float> %1539) #8, !dbg !14
  %1542 = shufflevector <8 x half> %1527, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1543 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1542, <4 x float> %1541) #8, !dbg !14
  %1544 = shufflevector <8 x half> %1529, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1545 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1544, <4 x float> %1543) #8, !dbg !14
  %1546 = shufflevector <8 x half> %1531, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1547 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1546, <4 x float> %1545) #8, !dbg !14
  %1548 = shufflevector <8 x half> %1533, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1549 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1548, <4 x float> %1547) #8, !dbg !14
  %1550 = shufflevector <8 x half> %1535, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1551 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1550, <4 x float> %1549) #8, !dbg !14
  %1552 = shufflevector <8 x half> %1537, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1552, <4 x float> %1551) #8, !dbg !14
  %1554 = extractelement <4 x float> %1553, i64 0, !dbg !14
  %1555 = extractelement <4 x float> %1553, i64 1, !dbg !14
  %1556 = extractelement <4 x float> %1553, i64 2, !dbg !14
  %1557 = extractelement <4 x float> %1553, i64 3, !dbg !14
  %1558 = select i1 %.not, float %1555, float %1554, !dbg !14
  %1559 = bitcast float %1558 to i32, !dbg !14
  %1560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1559), !dbg !14
  %1561 = bitcast i32 %1560 to float, !dbg !14
  %1562 = select i1 %.not, float %1554, float %1561, !dbg !14
  %1563 = select i1 %.not, float %1561, float %1555, !dbg !14
  %1564 = select i1 %.not, float %1557, float %1556, !dbg !14
  %1565 = bitcast float %1564 to i32, !dbg !14
  %1566 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1565), !dbg !14
  %1567 = bitcast i32 %1566 to float, !dbg !14
  %1568 = select i1 %.not, float %1556, float %1567, !dbg !14
  %1569 = select i1 %.not, float %1567, float %1557, !dbg !14
  %1570 = select i1 %.not1, float %1568, float %1562, !dbg !14
  %1571 = bitcast float %1570 to i32, !dbg !14
  %1572 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1571), !dbg !14
  %1573 = bitcast i32 %1572 to float, !dbg !14
  %1574 = select i1 %.not1, float %1569, float %1563, !dbg !14
  %1575 = bitcast float %1574 to i32, !dbg !14
  %1576 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1575), !dbg !14
  %1577 = bitcast i32 %1576 to float, !dbg !14
  %1578 = insertelement <4 x float> poison, float %1562, i64 0, !dbg !14
  %1579 = insertelement <4 x float> %1578, float %1563, i64 1, !dbg !14
  %1580 = insertelement <4 x float> %1579, float %1573, i64 2, !dbg !14
  %1581 = insertelement <4 x float> %1580, float %1577, i64 3, !dbg !14
  %1582 = insertelement <4 x float> poison, float %1573, i64 0, !dbg !14
  %1583 = insertelement <4 x float> %1582, float %1577, i64 1, !dbg !14
  %1584 = insertelement <4 x float> %1583, float %1568, i64 2, !dbg !14
  %1585 = insertelement <4 x float> %1584, float %1569, i64 3, !dbg !14
  %1586 = select <4 x i1> %66, <4 x float> %1581, <4 x float> %1585, !dbg !14
  %1587 = shufflevector <8 x half> %1523, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1587, <4 x float> zeroinitializer) #8, !dbg !14
  %1589 = shufflevector <8 x half> %1525, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1589, <4 x float> %1588) #8, !dbg !14
  %1591 = shufflevector <8 x half> %1527, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1591, <4 x float> %1590) #8, !dbg !14
  %1593 = shufflevector <8 x half> %1529, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1594 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1593, <4 x float> %1592) #8, !dbg !14
  %1595 = shufflevector <8 x half> %1531, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1596 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1595, <4 x float> %1594) #8, !dbg !14
  %1597 = shufflevector <8 x half> %1533, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1598 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1597, <4 x float> %1596) #8, !dbg !14
  %1599 = shufflevector <8 x half> %1535, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1600 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1599, <4 x float> %1598) #8, !dbg !14
  %1601 = shufflevector <8 x half> %1537, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1602 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1601, <4 x float> %1600) #8, !dbg !14
  %1603 = extractelement <4 x float> %1602, i64 0, !dbg !14
  %1604 = extractelement <4 x float> %1602, i64 1, !dbg !14
  %1605 = extractelement <4 x float> %1602, i64 2, !dbg !14
  %1606 = extractelement <4 x float> %1602, i64 3, !dbg !14
  %1607 = select i1 %.not, float %1604, float %1603, !dbg !14
  %1608 = bitcast float %1607 to i32, !dbg !14
  %1609 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1608), !dbg !14
  %1610 = bitcast i32 %1609 to float, !dbg !14
  %1611 = select i1 %.not, float %1603, float %1610, !dbg !14
  %1612 = select i1 %.not, float %1610, float %1604, !dbg !14
  %1613 = select i1 %.not, float %1606, float %1605, !dbg !14
  %1614 = bitcast float %1613 to i32, !dbg !14
  %1615 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1614), !dbg !14
  %1616 = bitcast i32 %1615 to float, !dbg !14
  %1617 = select i1 %.not, float %1605, float %1616, !dbg !14
  %1618 = select i1 %.not, float %1616, float %1606, !dbg !14
  %1619 = select i1 %.not1, float %1617, float %1611, !dbg !14
  %1620 = bitcast float %1619 to i32, !dbg !14
  %1621 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1620), !dbg !14
  %1622 = bitcast i32 %1621 to float, !dbg !14
  %1623 = select i1 %.not1, float %1618, float %1612, !dbg !14
  %1624 = bitcast float %1623 to i32, !dbg !14
  %1625 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1624), !dbg !14
  %1626 = bitcast i32 %1625 to float, !dbg !14
  %1627 = insertelement <4 x float> poison, float %1611, i64 0, !dbg !14
  %1628 = insertelement <4 x float> %1627, float %1612, i64 1, !dbg !14
  %1629 = insertelement <4 x float> %1628, float %1622, i64 2, !dbg !14
  %1630 = insertelement <4 x float> %1629, float %1626, i64 3, !dbg !14
  %1631 = insertelement <4 x float> poison, float %1622, i64 0, !dbg !14
  %1632 = insertelement <4 x float> %1631, float %1626, i64 1, !dbg !14
  %1633 = insertelement <4 x float> %1632, float %1617, i64 2, !dbg !14
  %1634 = insertelement <4 x float> %1633, float %1618, i64 3, !dbg !14
  %1635 = select <4 x i1> %66, <4 x float> %1630, <4 x float> %1634, !dbg !14
  %1636 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1538, <4 x float> zeroinitializer) #8, !dbg !14
  %1637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1540, <4 x float> %1636) #8, !dbg !14
  %1638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1542, <4 x float> %1637) #8, !dbg !14
  %1639 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1544, <4 x float> %1638) #8, !dbg !14
  %1640 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1546, <4 x float> %1639) #8, !dbg !14
  %1641 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1548, <4 x float> %1640) #8, !dbg !14
  %1642 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1550, <4 x float> %1641) #8, !dbg !14
  %1643 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1552, <4 x float> %1642) #8, !dbg !14
  %1644 = extractelement <4 x float> %1643, i64 0, !dbg !14
  %1645 = extractelement <4 x float> %1643, i64 1, !dbg !14
  %1646 = extractelement <4 x float> %1643, i64 2, !dbg !14
  %1647 = extractelement <4 x float> %1643, i64 3, !dbg !14
  %1648 = select i1 %.not, float %1645, float %1644, !dbg !14
  %1649 = bitcast float %1648 to i32, !dbg !14
  %1650 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1649), !dbg !14
  %1651 = bitcast i32 %1650 to float, !dbg !14
  %1652 = select i1 %.not, float %1644, float %1651, !dbg !14
  %1653 = select i1 %.not, float %1651, float %1645, !dbg !14
  %1654 = select i1 %.not, float %1647, float %1646, !dbg !14
  %1655 = bitcast float %1654 to i32, !dbg !14
  %1656 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1655), !dbg !14
  %1657 = bitcast i32 %1656 to float, !dbg !14
  %1658 = select i1 %.not, float %1646, float %1657, !dbg !14
  %1659 = select i1 %.not, float %1657, float %1647, !dbg !14
  %1660 = select i1 %.not1, float %1658, float %1652, !dbg !14
  %1661 = bitcast float %1660 to i32, !dbg !14
  %1662 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1661), !dbg !14
  %1663 = bitcast i32 %1662 to float, !dbg !14
  %1664 = select i1 %.not1, float %1659, float %1653, !dbg !14
  %1665 = bitcast float %1664 to i32, !dbg !14
  %1666 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1665), !dbg !14
  %1667 = bitcast i32 %1666 to float, !dbg !14
  %1668 = insertelement <4 x float> poison, float %1652, i64 0, !dbg !14
  %1669 = insertelement <4 x float> %1668, float %1653, i64 1, !dbg !14
  %1670 = insertelement <4 x float> %1669, float %1663, i64 2, !dbg !14
  %1671 = insertelement <4 x float> %1670, float %1667, i64 3, !dbg !14
  %1672 = insertelement <4 x float> poison, float %1663, i64 0, !dbg !14
  %1673 = insertelement <4 x float> %1672, float %1667, i64 1, !dbg !14
  %1674 = insertelement <4 x float> %1673, float %1658, i64 2, !dbg !14
  %1675 = insertelement <4 x float> %1674, float %1659, i64 3, !dbg !14
  %1676 = select <4 x i1> %66, <4 x float> %1671, <4 x float> %1675, !dbg !14
  %1677 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1587, <4 x float> zeroinitializer) #8, !dbg !14
  %1678 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1589, <4 x float> %1677) #8, !dbg !14
  %1679 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1591, <4 x float> %1678) #8, !dbg !14
  %1680 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1593, <4 x float> %1679) #8, !dbg !14
  %1681 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1595, <4 x float> %1680) #8, !dbg !14
  %1682 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1597, <4 x float> %1681) #8, !dbg !14
  %1683 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1599, <4 x float> %1682) #8, !dbg !14
  %1684 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1601, <4 x float> %1683) #8, !dbg !14
  %1685 = extractelement <4 x float> %1684, i64 0, !dbg !14
  %1686 = extractelement <4 x float> %1684, i64 1, !dbg !14
  %1687 = extractelement <4 x float> %1684, i64 2, !dbg !14
  %1688 = extractelement <4 x float> %1684, i64 3, !dbg !14
  %1689 = select i1 %.not, float %1686, float %1685, !dbg !14
  %1690 = bitcast float %1689 to i32, !dbg !14
  %1691 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1690), !dbg !14
  %1692 = bitcast i32 %1691 to float, !dbg !14
  %1693 = select i1 %.not, float %1685, float %1692, !dbg !14
  %1694 = select i1 %.not, float %1692, float %1686, !dbg !14
  %1695 = select i1 %.not, float %1688, float %1687, !dbg !14
  %1696 = bitcast float %1695 to i32, !dbg !14
  %1697 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1696), !dbg !14
  %1698 = bitcast i32 %1697 to float, !dbg !14
  %1699 = select i1 %.not, float %1687, float %1698, !dbg !14
  %1700 = select i1 %.not, float %1698, float %1688, !dbg !14
  %1701 = select i1 %.not1, float %1699, float %1693, !dbg !14
  %1702 = bitcast float %1701 to i32, !dbg !14
  %1703 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1702), !dbg !14
  %1704 = bitcast i32 %1703 to float, !dbg !14
  %1705 = select i1 %.not1, float %1700, float %1694, !dbg !14
  %1706 = bitcast float %1705 to i32, !dbg !14
  %1707 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1706), !dbg !14
  %1708 = bitcast i32 %1707 to float, !dbg !14
  %1709 = insertelement <4 x float> poison, float %1693, i64 0, !dbg !14
  %1710 = insertelement <4 x float> %1709, float %1694, i64 1, !dbg !14
  %1711 = insertelement <4 x float> %1710, float %1704, i64 2, !dbg !14
  %1712 = insertelement <4 x float> %1711, float %1708, i64 3, !dbg !14
  %1713 = insertelement <4 x float> poison, float %1704, i64 0, !dbg !14
  %1714 = insertelement <4 x float> %1713, float %1708, i64 1, !dbg !14
  %1715 = insertelement <4 x float> %1714, float %1699, i64 2, !dbg !14
  %1716 = insertelement <4 x float> %1715, float %1700, i64 3, !dbg !14
  %1717 = select <4 x i1> %66, <4 x float> %1712, <4 x float> %1716, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %39, label %1718, label %1719, !dbg !32

1718:                                             ; preds = %1490
  %address.thread984 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local985 = lshr i32 %address.thread984, 1, !dbg !13
  %address.local986 = and i32 %address.local985, 96, !dbg !13
  %address.local987 = and i32 %address.thread984, 15, !dbg !13
  %address.local988 = or disjoint i32 %address.local986, %address.local987, !dbg !13
  %address.local989 = zext nneg i32 %address.local988 to i64, !dbg !13
  %address.local990 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !13
  store float %812, ptr addrspace(3) %address.local990, align 4, !dbg !32
  %address.local992 = getelementptr i8, ptr addrspace(3) %address.local990, i64 64, !dbg !13
  store float %813, ptr addrspace(3) %address.local992, align 4, !dbg !32
  br label %1719, !dbg !32

1719:                                             ; preds = %1718, %1490
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.thread993 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %1720 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %1721 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.local1007 = zext nneg i32 %address.thread993 to i64, !dbg !10
  %address.local1008 = shl nuw nsw i64 %address.local1007, 3, !dbg !9
  %address.local1009 = and i64 %address.local1008, 1536, !dbg !9
  %address.local1010 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1009, !dbg !9
  %address.local1011 = ptrtoint ptr addrspace(3) %address.local1010 to i64, !dbg !9
  %address.local1012 = trunc i64 %address.local1011 to i32, !dbg !9
  %1722 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1012), !dbg !28
  %address.local1013 = shl nuw nsw i64 %address.local1007, 6, !dbg !28
  %address.local1016 = and i64 %address.local1008, 24, !dbg !28
  %1723 = or disjoint i64 %address.local1016, %.idx5558, !dbg !28
  %1724 = shl i64 %1723, 1, !dbg !28
  %1725 = or disjoint i64 %1724, %address.local1013, !dbg !28
  %1726 = trunc i64 %1725 to i32, !dbg !28
  %dma.buffer.offset1755 = or i32 %1726, 192, !dbg !28
  %dma.buffer.lds1757 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1722), !dbg !28
  %1727 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1757, i32 %dma.buffer.offset1755, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %address.local1019 = or disjoint i64 %address.local1007, 256, !dbg !9
  %address.local1020 = shl nuw nsw i64 %address.local1019, 3, !dbg !9
  %address.local1021 = and i64 %address.local1020, 3584, !dbg !9
  %address.local1022 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1021, !dbg !9
  %address.local1023 = ptrtoint ptr addrspace(3) %address.local1022 to i64, !dbg !9
  %address.local1024 = trunc i64 %address.local1023 to i32, !dbg !9
  %1728 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1024), !dbg !28
  %address.local1025 = shl nuw nsw i64 %address.local1019, 6, !dbg !28
  %address.local1026 = and i64 %address.local1025, 32512, !dbg !28
  %1729 = or disjoint i64 %1724, %address.local1026, !dbg !28
  %1730 = trunc i64 %1729 to i32, !dbg !28
  %dma.buffer.offset = or i32 %1730, 192, !dbg !28
  %dma.buffer.lds = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1728), !dbg !28
  %1731 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %1732 = extractvalue [2 x <8 x float>] %73, 0, !dbg !32
  %1733 = shufflevector <8 x float> %1732, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1734 = insertelement <4 x float> poison, float %1720, i64 0, !dbg !32
  %1735 = shufflevector <4 x float> %1734, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1736 = fmul <4 x float> %1733, %1735, !dbg !32
  %1737 = fadd <4 x float> %1586, %1736, !dbg !33
  %1738 = shufflevector <8 x float> %1732, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1739 = fmul <4 x float> %1738, %1735, !dbg !32
  %1740 = fadd <4 x float> %1635, %1739, !dbg !33
  %1741 = extractvalue [2 x <8 x float>] %73, 1, !dbg !32
  %1742 = shufflevector <8 x float> %1741, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1743 = insertelement <4 x float> poison, float %1721, i64 0, !dbg !32
  %1744 = shufflevector <4 x float> %1743, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1745 = fmul <4 x float> %1742, %1744, !dbg !32
  %1746 = fadd <4 x float> %1676, %1745, !dbg !33
  %1747 = shufflevector <8 x float> %1741, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1748 = fmul <4 x float> %1747, %1744, !dbg !32
  %1749 = fadd <4 x float> %1717, %1748, !dbg !33
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local1031 = shl nuw nsw i32 %address.thread993, 3, !dbg !14
  %address.local1032 = and i32 %address.local1031, 504, !dbg !14
  %address.local1033 = zext nneg i32 %address.local1032 to i64, !dbg !14
  %address.local1034 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !14
  %1750 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #10, !dbg !14
  %1751 = bitcast <4 x i32> %1750 to <8 x half>, !dbg !14
  %address.local1036 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 1024, !dbg !14
  %1752 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1036, i32 0) #10, !dbg !14
  %1753 = bitcast <4 x i32> %1752 to <8 x half>, !dbg !14
  %address.local1038 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 2048, !dbg !14
  %1754 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1038, i32 0) #10, !dbg !14
  %1755 = bitcast <4 x i32> %1754 to <8 x half>, !dbg !14
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 3072, !dbg !14
  %1756 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #10, !dbg !14
  %1757 = bitcast <4 x i32> %1756 to <8 x half>, !dbg !14
  %address.local1042 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 4096, !dbg !14
  %1758 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1042, i32 0) #10, !dbg !14
  %1759 = bitcast <4 x i32> %1758 to <8 x half>, !dbg !14
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 5120, !dbg !14
  %1760 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #10, !dbg !14
  %1761 = bitcast <4 x i32> %1760 to <8 x half>, !dbg !14
  %address.local1046 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 6144, !dbg !14
  %1762 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1046, i32 0) #10, !dbg !14
  %1763 = bitcast <4 x i32> %1762 to <8 x half>, !dbg !14
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 7168, !dbg !14
  %1764 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #10, !dbg !14
  %1765 = bitcast <4 x i32> %1764 to <8 x half>, !dbg !14
  %1766 = shufflevector <8 x half> %1751, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1767 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1766, <4 x float> zeroinitializer) #8, !dbg !14
  %1768 = shufflevector <8 x half> %1753, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1769 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1768, <4 x float> %1767) #8, !dbg !14
  %1770 = shufflevector <8 x half> %1755, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1771 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1770, <4 x float> %1769) #8, !dbg !14
  %1772 = shufflevector <8 x half> %1757, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1773 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1772, <4 x float> %1771) #8, !dbg !14
  %1774 = shufflevector <8 x half> %1759, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1775 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1774, <4 x float> %1773) #8, !dbg !14
  %1776 = shufflevector <8 x half> %1761, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1777 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1776, <4 x float> %1775) #8, !dbg !14
  %1778 = shufflevector <8 x half> %1763, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1779 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1778, <4 x float> %1777) #8, !dbg !14
  %1780 = shufflevector <8 x half> %1765, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1781 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1780, <4 x float> %1779) #8, !dbg !14
  %1782 = extractelement <4 x float> %1781, i64 0, !dbg !14
  %1783 = extractelement <4 x float> %1781, i64 1, !dbg !14
  %1784 = extractelement <4 x float> %1781, i64 2, !dbg !14
  %1785 = extractelement <4 x float> %1781, i64 3, !dbg !14
  %1786 = select i1 %.not, float %1783, float %1782, !dbg !14
  %1787 = bitcast float %1786 to i32, !dbg !14
  %1788 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1787), !dbg !14
  %1789 = bitcast i32 %1788 to float, !dbg !14
  %1790 = select i1 %.not, float %1782, float %1789, !dbg !14
  %1791 = select i1 %.not, float %1789, float %1783, !dbg !14
  %1792 = select i1 %.not, float %1785, float %1784, !dbg !14
  %1793 = bitcast float %1792 to i32, !dbg !14
  %1794 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1793), !dbg !14
  %1795 = bitcast i32 %1794 to float, !dbg !14
  %1796 = select i1 %.not, float %1784, float %1795, !dbg !14
  %1797 = select i1 %.not, float %1795, float %1785, !dbg !14
  %1798 = select i1 %.not1, float %1796, float %1790, !dbg !14
  %1799 = bitcast float %1798 to i32, !dbg !14
  %1800 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1799), !dbg !14
  %1801 = bitcast i32 %1800 to float, !dbg !14
  %1802 = select i1 %.not1, float %1797, float %1791, !dbg !14
  %1803 = bitcast float %1802 to i32, !dbg !14
  %1804 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1803), !dbg !14
  %1805 = bitcast i32 %1804 to float, !dbg !14
  %1806 = insertelement <4 x float> poison, float %1790, i64 0, !dbg !14
  %1807 = insertelement <4 x float> %1806, float %1791, i64 1, !dbg !14
  %1808 = insertelement <4 x float> %1807, float %1801, i64 2, !dbg !14
  %1809 = insertelement <4 x float> %1808, float %1805, i64 3, !dbg !14
  %1810 = insertelement <4 x float> poison, float %1801, i64 0, !dbg !14
  %1811 = insertelement <4 x float> %1810, float %1805, i64 1, !dbg !14
  %1812 = insertelement <4 x float> %1811, float %1796, i64 2, !dbg !14
  %1813 = insertelement <4 x float> %1812, float %1797, i64 3, !dbg !14
  %1814 = select <4 x i1> %66, <4 x float> %1809, <4 x float> %1813, !dbg !14
  %1815 = shufflevector <8 x half> %1751, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1815, <4 x float> zeroinitializer) #8, !dbg !14
  %1817 = shufflevector <8 x half> %1753, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1818 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1817, <4 x float> %1816) #8, !dbg !14
  %1819 = shufflevector <8 x half> %1755, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1820 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1819, <4 x float> %1818) #8, !dbg !14
  %1821 = shufflevector <8 x half> %1757, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1822 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1821, <4 x float> %1820) #8, !dbg !14
  %1823 = shufflevector <8 x half> %1759, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1824 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1823, <4 x float> %1822) #8, !dbg !14
  %1825 = shufflevector <8 x half> %1761, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1826 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1825, <4 x float> %1824) #8, !dbg !14
  %1827 = shufflevector <8 x half> %1763, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1828 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1827, <4 x float> %1826) #8, !dbg !14
  %1829 = shufflevector <8 x half> %1765, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1829, <4 x float> %1828) #8, !dbg !14
  %1831 = extractelement <4 x float> %1830, i64 0, !dbg !14
  %1832 = extractelement <4 x float> %1830, i64 1, !dbg !14
  %1833 = extractelement <4 x float> %1830, i64 2, !dbg !14
  %1834 = extractelement <4 x float> %1830, i64 3, !dbg !14
  %1835 = select i1 %.not, float %1832, float %1831, !dbg !14
  %1836 = bitcast float %1835 to i32, !dbg !14
  %1837 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1836), !dbg !14
  %1838 = bitcast i32 %1837 to float, !dbg !14
  %1839 = select i1 %.not, float %1831, float %1838, !dbg !14
  %1840 = select i1 %.not, float %1838, float %1832, !dbg !14
  %1841 = select i1 %.not, float %1834, float %1833, !dbg !14
  %1842 = bitcast float %1841 to i32, !dbg !14
  %1843 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1842), !dbg !14
  %1844 = bitcast i32 %1843 to float, !dbg !14
  %1845 = select i1 %.not, float %1833, float %1844, !dbg !14
  %1846 = select i1 %.not, float %1844, float %1834, !dbg !14
  %1847 = select i1 %.not1, float %1845, float %1839, !dbg !14
  %1848 = bitcast float %1847 to i32, !dbg !14
  %1849 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1848), !dbg !14
  %1850 = bitcast i32 %1849 to float, !dbg !14
  %1851 = select i1 %.not1, float %1846, float %1840, !dbg !14
  %1852 = bitcast float %1851 to i32, !dbg !14
  %1853 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1852), !dbg !14
  %1854 = bitcast i32 %1853 to float, !dbg !14
  %1855 = insertelement <4 x float> poison, float %1839, i64 0, !dbg !14
  %1856 = insertelement <4 x float> %1855, float %1840, i64 1, !dbg !14
  %1857 = insertelement <4 x float> %1856, float %1850, i64 2, !dbg !14
  %1858 = insertelement <4 x float> %1857, float %1854, i64 3, !dbg !14
  %1859 = insertelement <4 x float> poison, float %1850, i64 0, !dbg !14
  %1860 = insertelement <4 x float> %1859, float %1854, i64 1, !dbg !14
  %1861 = insertelement <4 x float> %1860, float %1845, i64 2, !dbg !14
  %1862 = insertelement <4 x float> %1861, float %1846, i64 3, !dbg !14
  %1863 = select <4 x i1> %66, <4 x float> %1858, <4 x float> %1862, !dbg !14
  %1864 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1766, <4 x float> zeroinitializer) #8, !dbg !14
  %1865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1768, <4 x float> %1864) #8, !dbg !14
  %1866 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1770, <4 x float> %1865) #8, !dbg !14
  %1867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1772, <4 x float> %1866) #8, !dbg !14
  %1868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1774, <4 x float> %1867) #8, !dbg !14
  %1869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1776, <4 x float> %1868) #8, !dbg !14
  %1870 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1778, <4 x float> %1869) #8, !dbg !14
  %1871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1780, <4 x float> %1870) #8, !dbg !14
  %1872 = extractelement <4 x float> %1871, i64 0, !dbg !14
  %1873 = extractelement <4 x float> %1871, i64 1, !dbg !14
  %1874 = extractelement <4 x float> %1871, i64 2, !dbg !14
  %1875 = extractelement <4 x float> %1871, i64 3, !dbg !14
  %1876 = select i1 %.not, float %1873, float %1872, !dbg !14
  %1877 = bitcast float %1876 to i32, !dbg !14
  %1878 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1877), !dbg !14
  %1879 = bitcast i32 %1878 to float, !dbg !14
  %1880 = select i1 %.not, float %1872, float %1879, !dbg !14
  %1881 = select i1 %.not, float %1879, float %1873, !dbg !14
  %1882 = select i1 %.not, float %1875, float %1874, !dbg !14
  %1883 = bitcast float %1882 to i32, !dbg !14
  %1884 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1883), !dbg !14
  %1885 = bitcast i32 %1884 to float, !dbg !14
  %1886 = select i1 %.not, float %1874, float %1885, !dbg !14
  %1887 = select i1 %.not, float %1885, float %1875, !dbg !14
  %1888 = select i1 %.not1, float %1886, float %1880, !dbg !14
  %1889 = bitcast float %1888 to i32, !dbg !14
  %1890 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1889), !dbg !14
  %1891 = bitcast i32 %1890 to float, !dbg !14
  %1892 = select i1 %.not1, float %1887, float %1881, !dbg !14
  %1893 = bitcast float %1892 to i32, !dbg !14
  %1894 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1893), !dbg !14
  %1895 = bitcast i32 %1894 to float, !dbg !14
  %1896 = insertelement <4 x float> poison, float %1880, i64 0, !dbg !14
  %1897 = insertelement <4 x float> %1896, float %1881, i64 1, !dbg !14
  %1898 = insertelement <4 x float> %1897, float %1891, i64 2, !dbg !14
  %1899 = insertelement <4 x float> %1898, float %1895, i64 3, !dbg !14
  %1900 = insertelement <4 x float> poison, float %1891, i64 0, !dbg !14
  %1901 = insertelement <4 x float> %1900, float %1895, i64 1, !dbg !14
  %1902 = insertelement <4 x float> %1901, float %1886, i64 2, !dbg !14
  %1903 = insertelement <4 x float> %1902, float %1887, i64 3, !dbg !14
  %1904 = select <4 x i1> %66, <4 x float> %1899, <4 x float> %1903, !dbg !14
  %1905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1815, <4 x float> zeroinitializer) #8, !dbg !14
  %1906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1817, <4 x float> %1905) #8, !dbg !14
  %1907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1819, <4 x float> %1906) #8, !dbg !14
  %1908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1821, <4 x float> %1907) #8, !dbg !14
  %1909 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1823, <4 x float> %1908) #8, !dbg !14
  %1910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1825, <4 x float> %1909) #8, !dbg !14
  %1911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1827, <4 x float> %1910) #8, !dbg !14
  %1912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1829, <4 x float> %1911) #8, !dbg !14
  %1913 = extractelement <4 x float> %1912, i64 0, !dbg !14
  %1914 = extractelement <4 x float> %1912, i64 1, !dbg !14
  %1915 = extractelement <4 x float> %1912, i64 2, !dbg !14
  %1916 = extractelement <4 x float> %1912, i64 3, !dbg !14
  %1917 = select i1 %.not, float %1914, float %1913, !dbg !14
  %1918 = bitcast float %1917 to i32, !dbg !14
  %1919 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1918), !dbg !14
  %1920 = bitcast i32 %1919 to float, !dbg !14
  %1921 = select i1 %.not, float %1913, float %1920, !dbg !14
  %1922 = select i1 %.not, float %1920, float %1914, !dbg !14
  %1923 = select i1 %.not, float %1916, float %1915, !dbg !14
  %1924 = bitcast float %1923 to i32, !dbg !14
  %1925 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %1924), !dbg !14
  %1926 = bitcast i32 %1925 to float, !dbg !14
  %1927 = select i1 %.not, float %1915, float %1926, !dbg !14
  %1928 = select i1 %.not, float %1926, float %1916, !dbg !14
  %1929 = select i1 %.not1, float %1927, float %1921, !dbg !14
  %1930 = bitcast float %1929 to i32, !dbg !14
  %1931 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1930), !dbg !14
  %1932 = bitcast i32 %1931 to float, !dbg !14
  %1933 = select i1 %.not1, float %1928, float %1922, !dbg !14
  %1934 = bitcast float %1933 to i32, !dbg !14
  %1935 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %1934), !dbg !14
  %1936 = bitcast i32 %1935 to float, !dbg !14
  %1937 = insertelement <4 x float> poison, float %1921, i64 0, !dbg !14
  %1938 = insertelement <4 x float> %1937, float %1922, i64 1, !dbg !14
  %1939 = insertelement <4 x float> %1938, float %1932, i64 2, !dbg !14
  %1940 = insertelement <4 x float> %1939, float %1936, i64 3, !dbg !14
  %1941 = insertelement <4 x float> poison, float %1932, i64 0, !dbg !14
  %1942 = insertelement <4 x float> %1941, float %1936, i64 1, !dbg !14
  %1943 = insertelement <4 x float> %1942, float %1927, i64 2, !dbg !14
  %1944 = insertelement <4 x float> %1943, float %1928, i64 3, !dbg !14
  %1945 = select <4 x i1> %66, <4 x float> %1940, <4 x float> %1944, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %39, label %1946, label %1947, !dbg !32

1946:                                             ; preds = %1719
  %address.thread1049 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local1050 = lshr i32 %address.thread1049, 1, !dbg !13
  %address.local1051 = and i32 %address.local1050, 96, !dbg !13
  %address.local1052 = and i32 %address.thread1049, 15, !dbg !13
  %address.local1053 = or disjoint i32 %address.local1051, %address.local1052, !dbg !13
  %address.local1054 = zext nneg i32 %address.local1053 to i64, !dbg !13
  %address.local1055 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !13
  store float %812, ptr addrspace(3) %address.local1055, align 4, !dbg !32
  %address.local1057 = getelementptr i8, ptr addrspace(3) %address.local1055, i64 64, !dbg !13
  store float %813, ptr addrspace(3) %address.local1057, align 4, !dbg !32
  br label %1947, !dbg !32

1947:                                             ; preds = %1946, %1719
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %1948 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %1949 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  %1950 = extractvalue [2 x <8 x float>] %72, 0, !dbg !32
  %1951 = shufflevector <8 x float> %1950, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1952 = insertelement <4 x float> poison, float %1948, i64 0, !dbg !32
  %1953 = shufflevector <4 x float> %1952, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1954 = fmul <4 x float> %1951, %1953, !dbg !32
  %1955 = fadd <4 x float> %1814, %1954, !dbg !33
  %1956 = shufflevector <8 x float> %1950, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1957 = fmul <4 x float> %1956, %1953, !dbg !32
  %1958 = fadd <4 x float> %1863, %1957, !dbg !33
  %1959 = extractvalue [2 x <8 x float>] %72, 1, !dbg !32
  %1960 = shufflevector <8 x float> %1959, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1961 = insertelement <4 x float> poison, float %1949, i64 0, !dbg !32
  %1962 = shufflevector <4 x float> %1961, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1963 = fmul <4 x float> %1960, %1962, !dbg !32
  %1964 = fadd <4 x float> %1904, %1963, !dbg !33
  %1965 = shufflevector <8 x float> %1959, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1966 = fmul <4 x float> %1965, %1962, !dbg !32
  %1967 = fadd <4 x float> %1945, %1966, !dbg !33
  %1968 = shufflevector <4 x float> %1280, <4 x float> %1283, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1969 = insertvalue [2 x <8 x float>] poison, <8 x float> %1968, 0, !dbg !17
  %1970 = shufflevector <4 x float> %1289, <4 x float> %1292, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1971 = insertvalue [2 x <8 x float>] %1969, <8 x float> %1970, 1, !dbg !17
  %1972 = insertelement <1 x float> poison, float %1053, i64 0, !dbg !17
  %1973 = insertvalue [2 x <1 x float>] poison, <1 x float> %1972, 0, !dbg !17
  %1974 = insertelement <1 x float> poison, float %1062, i64 0, !dbg !17
  %1975 = insertvalue [2 x <1 x float>] %1973, <1 x float> %1974, 1, !dbg !17
  %1976 = insertvalue [2 x <1 x float>] poison, <1 x float> %772, 0, !dbg !17
  %1977 = insertvalue [2 x <1 x float>] %1976, <1 x float> %785, 1, !dbg !17
  %1978 = shufflevector <4 x float> %1509, <4 x float> %1512, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1979 = insertvalue [2 x <8 x float>] poison, <8 x float> %1978, 0, !dbg !17
  %1980 = shufflevector <4 x float> %1518, <4 x float> %1521, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1981 = insertvalue [2 x <8 x float>] %1979, <8 x float> %1980, 1, !dbg !17
  %1982 = shufflevector <4 x float> %1737, <4 x float> %1740, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1983 = insertvalue [2 x <8 x float>] poison, <8 x float> %1982, 0, !dbg !17
  %1984 = shufflevector <4 x float> %1746, <4 x float> %1749, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1985 = insertvalue [2 x <8 x float>] %1983, <8 x float> %1984, 1, !dbg !17
  %1986 = shufflevector <4 x float> %1955, <4 x float> %1958, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1987 = insertvalue [2 x <8 x float>] poison, <8 x float> %1986, 0, !dbg !17
  %1988 = shufflevector <4 x float> %1964, <4 x float> %1967, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1989 = insertvalue [2 x <8 x float>] %1987, <8 x float> %1988, 1, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1990 = add nuw nsw i64 %78, 1, !dbg !17
  %1991 = icmp slt i64 %1990, %42, !dbg !17
  br i1 %1991, label %71, label %._crit_edge, !dbg !17

._crit_edge:                                      ; preds = %1947, %4
  %.lcssa94 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1971, %1947 ], !dbg !17
  %.lcssa93 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1975, %1947 ], !dbg !17
  %.lcssa92 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1981, %1947 ], !dbg !17
  %.lcssa91 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1985, %1947 ], !dbg !17
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1989, %1947 ], !dbg !17
  %1992 = extractvalue [2 x <1 x float>] %.lcssa93, 0, !dbg !34
  %1993 = shufflevector <1 x float> %1992, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1994 = extractvalue [2 x <1 x float>] %.lcssa93, 1, !dbg !34
  %1995 = shufflevector <1 x float> %1994, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1996 = extractvalue [2 x <8 x float>] %.lcssa94, 1, !dbg !34
  %1997 = shufflevector <8 x float> %1996, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1995, !dbg !34
  %normalized = fmul <4 x float> %1997, %row.reciprocal, !dbg !34
  %1998 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !35
  %1999 = extractelement <4 x half> %1998, i64 3, !dbg !35
  %2000 = extractelement <4 x half> %1998, i64 2, !dbg !35
  %2001 = extractelement <4 x half> %1998, i64 1, !dbg !35
  %2002 = extractelement <4 x half> %1998, i64 0, !dbg !35
  %2003 = shufflevector <8 x float> %1996, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized68 = fmul <4 x float> %2003, %row.reciprocal, !dbg !34
  %2004 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !35
  %2005 = extractelement <4 x half> %2004, i64 3, !dbg !35
  %2006 = extractelement <4 x half> %2004, i64 2, !dbg !35
  %2007 = extractelement <4 x half> %2004, i64 1, !dbg !35
  %2008 = extractelement <4 x half> %2004, i64 0, !dbg !35
  %2009 = extractvalue [2 x <8 x float>] %.lcssa94, 0, !dbg !34
  %2010 = shufflevector <8 x float> %2009, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1993, !dbg !34
  %normalized70 = fmul <4 x float> %2010, %row.reciprocal69, !dbg !34
  %2011 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !35
  %2012 = extractelement <4 x half> %2011, i64 3, !dbg !35
  %2013 = extractelement <4 x half> %2011, i64 2, !dbg !35
  %2014 = extractelement <4 x half> %2011, i64 1, !dbg !35
  %2015 = extractelement <4 x half> %2011, i64 0, !dbg !35
  %2016 = shufflevector <8 x float> %2009, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized71 = fmul <4 x float> %2016, %row.reciprocal69, !dbg !34
  %2017 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !35
  %2018 = extractelement <4 x half> %2017, i64 3, !dbg !35
  %2019 = extractelement <4 x half> %2017, i64 2, !dbg !35
  %2020 = extractelement <4 x half> %2017, i64 1, !dbg !35
  %2021 = extractelement <4 x half> %2017, i64 0, !dbg !35
  %address.thread1072 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !36, !range !11, !frisk.register_lifetimes !5
  %address.local1073 = zext nneg i32 %address.thread1072 to i64, !dbg !10
  %address.local1075 = lshr i64 %address.local1073, 2, !dbg !18
  %address.local1076 = and i64 %address.local1075, 48, !dbg !18
  %2022 = and i64 %address.local1073, 240, !dbg !36
  %address.local1077 = add i64 %28, %address.local1073, !dbg !36
  %address.local1080 = sub i64 %address.local1077, %2022, !dbg !36
  %address.local1081 = add i64 %address.local1080, %address.local1076, !dbg !36
  %2023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %30, !dbg !36
  %address.local1084.idx = shl nuw nsw i64 %address.local1081, 8, !dbg !36
  %address.local1084 = getelementptr inbounds nuw i8, ptr addrspace(1) %2023, i64 %address.local1084.idx, !dbg !36
  %address.local1087 = and i64 %address.local1075, 12, !dbg !10
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1087, !dbg !36
  store half %2021, ptr addrspace(1) %address.local1088, align 2, !dbg !36
  %address.local1090 = or disjoint i64 %address.local1087, 1, !dbg !18
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1090, !dbg !36
  store half %2020, ptr addrspace(1) %address.local1091, align 2, !dbg !36
  %address.local1093 = or disjoint i64 %address.local1087, 2, !dbg !18
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1093, !dbg !36
  store half %2019, ptr addrspace(1) %address.local1094, align 2, !dbg !36
  %address.local1096 = or disjoint i64 %address.local1087, 3, !dbg !18
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1096, !dbg !36
  store half %2018, ptr addrspace(1) %address.local1097, align 2, !dbg !36
  %address.local1099 = or disjoint i64 %address.local1087, 16, !dbg !18
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1099, !dbg !36
  store half %2015, ptr addrspace(1) %address.local1100, align 2, !dbg !36
  %address.local1102 = or disjoint i64 %address.local1087, 17, !dbg !18
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1102, !dbg !36
  store half %2014, ptr addrspace(1) %address.local1103, align 2, !dbg !36
  %address.local1105 = or disjoint i64 %address.local1087, 18, !dbg !18
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1105, !dbg !36
  store half %2013, ptr addrspace(1) %address.local1106, align 2, !dbg !36
  %address.local1108 = or disjoint i64 %address.local1087, 19, !dbg !18
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1108, !dbg !36
  store half %2012, ptr addrspace(1) %address.local1109, align 2, !dbg !36
  %address.local1113 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1084, i64 16384, !dbg !36
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1087, !dbg !36
  store half %2008, ptr addrspace(1) %address.local1114, align 2, !dbg !36
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1090, !dbg !36
  store half %2007, ptr addrspace(1) %address.local1116, align 2, !dbg !36
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1093, !dbg !36
  store half %2006, ptr addrspace(1) %address.local1118, align 2, !dbg !36
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1096, !dbg !36
  store half %2005, ptr addrspace(1) %address.local1120, align 2, !dbg !36
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1099, !dbg !36
  store half %2002, ptr addrspace(1) %address.local1122, align 2, !dbg !36
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1102, !dbg !36
  store half %2001, ptr addrspace(1) %address.local1124, align 2, !dbg !36
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1105, !dbg !36
  store half %2000, ptr addrspace(1) %address.local1126, align 2, !dbg !36
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1108, !dbg !36
  store half %1999, ptr addrspace(1) %address.local1128, align 2, !dbg !36
  %2024 = extractvalue [2 x <8 x float>] %.lcssa92, 1, !dbg !34
  %2025 = shufflevector <8 x float> %2024, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized73 = fmul <4 x float> %row.reciprocal, %2025, !dbg !34
  %2026 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !35
  %2027 = extractelement <4 x half> %2026, i64 3, !dbg !35
  %2028 = extractelement <4 x half> %2026, i64 2, !dbg !35
  %2029 = extractelement <4 x half> %2026, i64 1, !dbg !35
  %2030 = extractelement <4 x half> %2026, i64 0, !dbg !35
  %2031 = shufflevector <8 x float> %2024, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized74 = fmul <4 x float> %row.reciprocal, %2031, !dbg !34
  %2032 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !35
  %2033 = extractelement <4 x half> %2032, i64 3, !dbg !35
  %2034 = extractelement <4 x half> %2032, i64 2, !dbg !35
  %2035 = extractelement <4 x half> %2032, i64 1, !dbg !35
  %2036 = extractelement <4 x half> %2032, i64 0, !dbg !35
  %2037 = extractvalue [2 x <8 x float>] %.lcssa92, 0, !dbg !34
  %2038 = shufflevector <8 x float> %2037, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized76 = fmul <4 x float> %row.reciprocal69, %2038, !dbg !34
  %2039 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !35
  %2040 = extractelement <4 x half> %2039, i64 3, !dbg !35
  %2041 = extractelement <4 x half> %2039, i64 2, !dbg !35
  %2042 = extractelement <4 x half> %2039, i64 1, !dbg !35
  %2043 = extractelement <4 x half> %2039, i64 0, !dbg !35
  %2044 = shufflevector <8 x float> %2037, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized77 = fmul <4 x float> %row.reciprocal69, %2044, !dbg !34
  %2045 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !35
  %2046 = extractelement <4 x half> %2045, i64 3, !dbg !35
  %2047 = extractelement <4 x half> %2045, i64 2, !dbg !35
  %2048 = extractelement <4 x half> %2045, i64 1, !dbg !35
  %2049 = extractelement <4 x half> %2045, i64 0, !dbg !35
  %address.local1145 = or disjoint i64 %address.local1087, 32, !dbg !18
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1145, !dbg !36
  store half %2049, ptr addrspace(1) %address.local1146, align 2, !dbg !36
  %address.local1148 = or disjoint i64 %address.local1087, 33, !dbg !18
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1148, !dbg !36
  store half %2048, ptr addrspace(1) %address.local1149, align 2, !dbg !36
  %address.local1151 = or disjoint i64 %address.local1087, 34, !dbg !18
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1151, !dbg !36
  store half %2047, ptr addrspace(1) %address.local1152, align 2, !dbg !36
  %address.local1154 = or disjoint i64 %address.local1087, 35, !dbg !18
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1154, !dbg !36
  store half %2046, ptr addrspace(1) %address.local1155, align 2, !dbg !36
  %address.local1157 = or disjoint i64 %address.local1087, 48, !dbg !18
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1157, !dbg !36
  store half %2043, ptr addrspace(1) %address.local1158, align 2, !dbg !36
  %address.local1160 = or disjoint i64 %address.local1087, 49, !dbg !18
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1160, !dbg !36
  store half %2042, ptr addrspace(1) %address.local1161, align 2, !dbg !36
  %address.local1163 = or disjoint i64 %address.local1087, 50, !dbg !18
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1163, !dbg !36
  store half %2041, ptr addrspace(1) %address.local1164, align 2, !dbg !36
  %address.local1166 = or i64 %address.local1075, 51, !dbg !18
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1166, !dbg !36
  store half %2040, ptr addrspace(1) %address.local1167, align 2, !dbg !36
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1145, !dbg !36
  store half %2036, ptr addrspace(1) %address.local1172, align 2, !dbg !36
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1148, !dbg !36
  store half %2035, ptr addrspace(1) %address.local1174, align 2, !dbg !36
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1151, !dbg !36
  store half %2034, ptr addrspace(1) %address.local1176, align 2, !dbg !36
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1154, !dbg !36
  store half %2033, ptr addrspace(1) %address.local1178, align 2, !dbg !36
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1157, !dbg !36
  store half %2030, ptr addrspace(1) %address.local1180, align 2, !dbg !36
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1160, !dbg !36
  store half %2029, ptr addrspace(1) %address.local1182, align 2, !dbg !36
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1163, !dbg !36
  store half %2028, ptr addrspace(1) %address.local1184, align 2, !dbg !36
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1166, !dbg !36
  store half %2027, ptr addrspace(1) %address.local1186, align 2, !dbg !36
  %2050 = extractvalue [2 x <8 x float>] %.lcssa91, 1, !dbg !34
  %2051 = shufflevector <8 x float> %2050, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized79 = fmul <4 x float> %row.reciprocal, %2051, !dbg !34
  %2052 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !35
  %2053 = extractelement <4 x half> %2052, i64 3, !dbg !35
  %2054 = extractelement <4 x half> %2052, i64 2, !dbg !35
  %2055 = extractelement <4 x half> %2052, i64 1, !dbg !35
  %2056 = extractelement <4 x half> %2052, i64 0, !dbg !35
  %2057 = shufflevector <8 x float> %2050, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized80 = fmul <4 x float> %row.reciprocal, %2057, !dbg !34
  %2058 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !35
  %2059 = extractelement <4 x half> %2058, i64 3, !dbg !35
  %2060 = extractelement <4 x half> %2058, i64 2, !dbg !35
  %2061 = extractelement <4 x half> %2058, i64 1, !dbg !35
  %2062 = extractelement <4 x half> %2058, i64 0, !dbg !35
  %2063 = extractvalue [2 x <8 x float>] %.lcssa91, 0, !dbg !34
  %2064 = shufflevector <8 x float> %2063, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized82 = fmul <4 x float> %row.reciprocal69, %2064, !dbg !34
  %2065 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !35
  %2066 = extractelement <4 x half> %2065, i64 3, !dbg !35
  %2067 = extractelement <4 x half> %2065, i64 2, !dbg !35
  %2068 = extractelement <4 x half> %2065, i64 1, !dbg !35
  %2069 = extractelement <4 x half> %2065, i64 0, !dbg !35
  %2070 = shufflevector <8 x float> %2063, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized83 = fmul <4 x float> %row.reciprocal69, %2070, !dbg !34
  %2071 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !35
  %2072 = extractelement <4 x half> %2071, i64 3, !dbg !35
  %2073 = extractelement <4 x half> %2071, i64 2, !dbg !35
  %2074 = extractelement <4 x half> %2071, i64 1, !dbg !35
  %2075 = extractelement <4 x half> %2071, i64 0, !dbg !35
  %address.local1203 = or disjoint i64 %address.local1087, 64, !dbg !18
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1203, !dbg !36
  store half %2075, ptr addrspace(1) %address.local1204, align 2, !dbg !36
  %address.local1206 = or disjoint i64 %address.local1087, 65, !dbg !18
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1206, !dbg !36
  store half %2074, ptr addrspace(1) %address.local1207, align 2, !dbg !36
  %address.local1209 = or disjoint i64 %address.local1087, 66, !dbg !18
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1209, !dbg !36
  store half %2073, ptr addrspace(1) %address.local1210, align 2, !dbg !36
  %address.local1212 = or disjoint i64 %address.local1087, 67, !dbg !18
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1212, !dbg !36
  store half %2072, ptr addrspace(1) %address.local1213, align 2, !dbg !36
  %address.local1215 = or disjoint i64 %address.local1087, 80, !dbg !18
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1215, !dbg !36
  store half %2069, ptr addrspace(1) %address.local1216, align 2, !dbg !36
  %address.local1218 = or disjoint i64 %address.local1087, 81, !dbg !18
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1218, !dbg !36
  store half %2068, ptr addrspace(1) %address.local1219, align 2, !dbg !36
  %address.local1221 = or disjoint i64 %address.local1087, 82, !dbg !18
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1221, !dbg !36
  store half %2067, ptr addrspace(1) %address.local1222, align 2, !dbg !36
  %address.local1224 = or disjoint i64 %address.local1087, 83, !dbg !18
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1224, !dbg !36
  store half %2066, ptr addrspace(1) %address.local1225, align 2, !dbg !36
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1203, !dbg !36
  store half %2062, ptr addrspace(1) %address.local1230, align 2, !dbg !36
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1206, !dbg !36
  store half %2061, ptr addrspace(1) %address.local1232, align 2, !dbg !36
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1209, !dbg !36
  store half %2060, ptr addrspace(1) %address.local1234, align 2, !dbg !36
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1212, !dbg !36
  store half %2059, ptr addrspace(1) %address.local1236, align 2, !dbg !36
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1215, !dbg !36
  store half %2056, ptr addrspace(1) %address.local1238, align 2, !dbg !36
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1218, !dbg !36
  store half %2055, ptr addrspace(1) %address.local1240, align 2, !dbg !36
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1221, !dbg !36
  store half %2054, ptr addrspace(1) %address.local1242, align 2, !dbg !36
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1224, !dbg !36
  store half %2053, ptr addrspace(1) %address.local1244, align 2, !dbg !36
  %2076 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !34
  %2077 = shufflevector <8 x float> %2076, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized85 = fmul <4 x float> %row.reciprocal, %2077, !dbg !34
  %2078 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !35
  %2079 = extractelement <4 x half> %2078, i64 3, !dbg !35
  %2080 = extractelement <4 x half> %2078, i64 2, !dbg !35
  %2081 = extractelement <4 x half> %2078, i64 1, !dbg !35
  %2082 = extractelement <4 x half> %2078, i64 0, !dbg !35
  %2083 = shufflevector <8 x float> %2076, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized86 = fmul <4 x float> %row.reciprocal, %2083, !dbg !34
  %2084 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !35
  %2085 = extractelement <4 x half> %2084, i64 3, !dbg !35
  %2086 = extractelement <4 x half> %2084, i64 2, !dbg !35
  %2087 = extractelement <4 x half> %2084, i64 1, !dbg !35
  %2088 = extractelement <4 x half> %2084, i64 0, !dbg !35
  %2089 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !34
  %2090 = shufflevector <8 x float> %2089, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized88 = fmul <4 x float> %row.reciprocal69, %2090, !dbg !34
  %2091 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !35
  %2092 = extractelement <4 x half> %2091, i64 3, !dbg !35
  %2093 = extractelement <4 x half> %2091, i64 2, !dbg !35
  %2094 = extractelement <4 x half> %2091, i64 1, !dbg !35
  %2095 = extractelement <4 x half> %2091, i64 0, !dbg !35
  %2096 = shufflevector <8 x float> %2089, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized89 = fmul <4 x float> %row.reciprocal69, %2096, !dbg !34
  %2097 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !35
  %2098 = extractelement <4 x half> %2097, i64 3, !dbg !35
  %2099 = extractelement <4 x half> %2097, i64 2, !dbg !35
  %2100 = extractelement <4 x half> %2097, i64 1, !dbg !35
  %2101 = extractelement <4 x half> %2097, i64 0, !dbg !35
  %address.local1261 = or disjoint i64 %address.local1087, 96, !dbg !18
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1261, !dbg !36
  store half %2101, ptr addrspace(1) %address.local1262, align 2, !dbg !36
  %address.local1264 = or disjoint i64 %address.local1087, 97, !dbg !18
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1264, !dbg !36
  store half %2100, ptr addrspace(1) %address.local1265, align 2, !dbg !36
  %address.local1267 = or disjoint i64 %address.local1087, 98, !dbg !18
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1267, !dbg !36
  store half %2099, ptr addrspace(1) %address.local1268, align 2, !dbg !36
  %address.local1270 = or disjoint i64 %address.local1087, 99, !dbg !18
  %address.local1271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1270, !dbg !36
  store half %2098, ptr addrspace(1) %address.local1271, align 2, !dbg !36
  %address.local1273 = or disjoint i64 %address.local1087, 112, !dbg !18
  %address.local1274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1273, !dbg !36
  store half %2095, ptr addrspace(1) %address.local1274, align 2, !dbg !36
  %address.local1276 = or disjoint i64 %address.local1087, 113, !dbg !18
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1276, !dbg !36
  store half %2094, ptr addrspace(1) %address.local1277, align 2, !dbg !36
  %address.local1279 = or disjoint i64 %address.local1087, 114, !dbg !18
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1279, !dbg !36
  store half %2093, ptr addrspace(1) %address.local1280, align 2, !dbg !36
  %address.local1282 = or i64 %address.local1075, 115, !dbg !18
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1282, !dbg !36
  store half %2092, ptr addrspace(1) %address.local1283, align 2, !dbg !36
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1261, !dbg !36
  store half %2088, ptr addrspace(1) %address.local1288, align 2, !dbg !36
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1264, !dbg !36
  store half %2087, ptr addrspace(1) %address.local1290, align 2, !dbg !36
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1267, !dbg !36
  store half %2086, ptr addrspace(1) %address.local1292, align 2, !dbg !36
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1270, !dbg !36
  store half %2085, ptr addrspace(1) %address.local1294, align 2, !dbg !36
  %address.local1296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1273, !dbg !36
  store half %2082, ptr addrspace(1) %address.local1296, align 2, !dbg !36
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1276, !dbg !36
  store half %2081, ptr addrspace(1) %address.local1298, align 2, !dbg !36
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1279, !dbg !36
  store half %2080, ptr addrspace(1) %address.local1300, align 2, !dbg !36
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1282, !dbg !36
  store half %2079, ptr addrspace(1) %address.local1302, align 2, !dbg !36
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
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000001")
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
!13 = !DILocation(line: 92, column: 20, scope: !3)
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
!27 = !DILocation(line: 93, column: 22, scope: !3)
!28 = !DILocation(line: 101, column: 20, scope: !3)
!29 = !DILocation(line: 94, column: 22, scope: !3)
!30 = !DILocation(line: 96, column: 16, scope: !3)
!31 = !DILocation(line: 95, column: 20, scope: !3)
!32 = !DILocation(line: 103, column: 22, scope: !3)
!33 = !DILocation(line: 104, column: 20, scope: !3)
!34 = !DILocation(line: 107, column: 23, scope: !3)
!35 = !DILocation(line: 108, column: 19, scope: !3)
!36 = !DILocation(line: 109, column: 9, scope: !3)
