; ModuleID = 'generated_ir/gemm_result_layout_comparison/spill_analysis/kernel.ll'
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
  %31 = shl nsw i64 %28, 7, !dbg !9
  %32 = add i64 %30, %31, !dbg !9
  %33 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !10, !range !11
  %34 = zext nneg i32 %33 to i64, !dbg !10
  %35 = lshr i64 %34, 4, !dbg !10
  %36 = and i64 %35, 3, !dbg !10
  %37 = shl nuw nsw i64 %36, 2, !dbg !10
  %38 = or disjoint i64 %19, 127, !dbg !12
  %39 = add i64 %19, 128, !dbg !12
  %40 = icmp uge i64 %39, %38, !dbg !12
  %41 = icmp eq i64 %36, 0, !dbg !13
  %42 = and i32 %33, 16, !dbg !14
  %.not = icmp eq i32 %42, 0, !dbg !14
  %43 = and i32 %33, 32, !dbg !14
  %.not1 = icmp eq i32 %43, 0, !dbg !14
  %44 = sub nsw i64 32, %14, !dbg !15
  %45 = shl i64 %15, 7, !dbg !16
  %46 = and i64 %45, 3968, !dbg !16
  %47 = shl nsw i64 %18, 12, !dbg !16
  %48 = or disjoint i64 %46, %47, !dbg !16
  %49 = icmp ult i64 %10, 1024, !dbg !17
  br i1 %49, label %.lr.ph, label %._crit_edge, !dbg !17

.lr.ph:                                           ; preds = %4
  %50 = lshr i64 %34, 2, !dbg !18
  %51 = and i64 %50, 48, !dbg !18
  %invariant.op = add nuw nsw i64 %51, %34, !dbg !17
  %invariant.op8 = add nuw nsw i64 %invariant.op, %19, !dbg !17
  %52 = and i64 %34, 240, !dbg !12
  %53 = ptrtoint ptr addrspace(1) %0 to i64
  %54 = insertelement <2 x i64> poison, i64 %53, i64 0
  %55 = bitcast <2 x i64> %54 to <4 x i32>
  %dma.buffer.rsrc1822 = shufflevector <4 x i32> %55, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %56 = ptrtoint ptr addrspace(1) %1 to i64
  %57 = insertelement <2 x i64> poison, i64 %56, i64 0
  %58 = bitcast <2 x i64> %57 to <4 x i32>
  %dma.buffer.rsrc1816 = shufflevector <4 x i32> %58, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %59 = or disjoint i64 %32, 32
  %60 = or disjoint i64 %32, 64
  %61 = or disjoint i64 %32, 96
  %.reass9 = sub i64 %invariant.op8, %52
  %62 = add i64 %.reass9, 1
  %63 = lshr i64 %62, 32
  %compare.high = trunc nuw i64 %63 to i32
  %compare.low.base = trunc i64 %62 to i32
  %64 = add i64 %.reass9, 65
  %65 = lshr i64 %64, 32
  %compare.high1427 = trunc nuw i64 %65 to i32
  %compare.low.base1431 = trunc i64 %64 to i32
  %66 = ptrtoint ptr addrspace(1) %2 to i64
  %67 = insertelement <2 x i64> poison, i64 %66, i64 0
  %68 = bitcast <2 x i64> %67 to <4 x i32>
  %dma.buffer.rsrc1774 = shufflevector <4 x i32> %68, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %69 = insertelement <4 x i1> poison, i1 %.not1, i64 0, !dbg !14
  %70 = shufflevector <4 x i1> %69, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !14
  %71 = insertelement <32 x i32> poison, i32 %compare.low.base1431, i64 0, !dbg !19
  %72 = shufflevector <32 x i32> %71, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !19
  %73 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !19
  %74 = shufflevector <32 x i32> %73, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !19
  br label %75, !dbg !17

75:                                               ; preds = %.lr.ph, %1883
  %76 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1925, %1883 ]
  %77 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1921, %1883 ]
  %78 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1917, %1883 ]
  %79 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1913, %1883 ]
  %80 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1911, %1883 ]
  %81 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1907, %1883 ]
  %82 = phi i64 [ 0, %.lr.ph ], [ %1926, %1883 ]
  %83 = shl nuw nsw i64 %82, 7, !dbg !17
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !9, !range !11, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !10
  %address.local258 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local259 = and i64 %address.local258, 1536, !dbg !9
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local259, !dbg !9
  %address.local263 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local264 = and i64 %address.local263, 16256, !dbg !9
  %address.local265 = lshr i64 %address.local, 4, !dbg !9
  %address.local266 = and i64 %address.local265, 8, !dbg !9
  %address.local267 = or disjoint i64 %address.local264, %address.local266, !dbg !9
  %.idx1 = or disjoint i64 %address.local267, %32, !dbg !9
  %.idx1.tr = trunc i64 %.idx1 to i32, !dbg !9
  %dma.buffer.offset1821 = shl i32 %.idx1.tr, 1, !dbg !9
  %address.local269 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local270 = shl nuw nsw i64 %address.local269, 3, !dbg !9
  %address.local271 = and i64 %address.local270, 3584, !dbg !9
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local271, !dbg !9
  %address.local277 = lshr i64 %address.local269, 4, !dbg !9
  %address.local278 = and i64 %address.local277, 24, !dbg !9
  %address.local279 = or disjoint i64 %address.local278, %address.local264, !dbg !9
  %.idx2 = or disjoint i64 %address.local279, %32, !dbg !9
  %.idx2.tr = trunc i64 %.idx2 to i32, !dbg !9
  %dma.buffer.offset1818 = shl i32 %.idx2.tr, 1, !dbg !9
  %84 = lshr i64 %82, 12, !dbg !16
  %85 = add nuw nsw i64 %84, %15, !dbg !16
  %86 = lshr i64 %82, 5, !dbg !16
  %87 = and i64 %86, 127, !dbg !16
  %88 = shl i64 %85, 19, !dbg !16
  %89 = and i64 %88, 16252928, !dbg !16
  %90 = shl i64 %86, 19, !dbg !16
  %91 = shl nuw nsw i64 %82, 14, !dbg !16
  %92 = sub i64 %91, %90, !dbg !16
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local259, !dbg !16
  %address.local286 = and i64 %address.local, 123, !dbg !16
  %address.local287 = lshr i64 %address.local, 2, !dbg !16
  %address.local288 = and i64 %address.local287, 4, !dbg !16
  %address.local289 = add nuw nsw i64 %address.local288, %address.local, !dbg !16
  %address.local290 = and i64 %address.local289, 4, !dbg !16
  %address.local291 = or disjoint i64 %address.local290, %address.local286, !dbg !16
  %address.local292 = shl nuw nsw i64 %address.local291, 7, !dbg !16
  %address.local293 = or disjoint i64 %address.local292, %address.local266, !dbg !16
  %.idx5 = or disjoint i64 %89, %87, !dbg !16
  %93 = add i64 %.idx5, %92, !dbg !16
  %94 = add i64 %address.local293, %93, !dbg !16
  %.tr = trunc i64 %94 to i32, !dbg !16
  %dma.buffer.offset1815 = shl i32 %.tr, 1, !dbg !16
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local271, !dbg !16
  %95 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local260, i64 0, !dbg !9
  %96 = insertelement <4 x ptr addrspace(3)> %95, ptr addrspace(3) %address.local272, i64 1, !dbg !9
  %97 = insertelement <4 x ptr addrspace(3)> %96, ptr addrspace(3) %address.local281, i64 2, !dbg !9
  %98 = insertelement <4 x ptr addrspace(3)> %97, ptr addrspace(3) %address.local295, i64 3, !dbg !9
  %99 = ptrtoint <4 x ptr addrspace(3)> %98 to <4 x i64>, !dbg !9
  %100 = trunc <4 x i64> %99 to <4 x i32>, !dbg !9
  %101 = extractelement <4 x i32> %100, i64 0, !dbg !9
  %102 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %101), !dbg !9
  %dma.buffer.lds1823 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %102), !dbg !9
  %103 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1821, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %104 = extractelement <4 x i32> %100, i64 1, !dbg !9
  %105 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %104), !dbg !9
  %dma.buffer.lds1820 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %105), !dbg !9
  %106 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1818, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %107 = extractelement <4 x i32> %100, i64 2, !dbg !16
  %108 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %107), !dbg !16
  %dma.buffer.lds1817 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %108), !dbg !16
  %109 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1815, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %110 = extractelement <4 x i32> %100, i64 3, !dbg !16
  %111 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %110), !dbg !16
  %address.local307 = or disjoint i64 %address.local292, %address.local278, !dbg !16
  %112 = add i64 %address.local307, %93, !dbg !16
  %.tr9 = trunc i64 %112 to i32, !dbg !16
  %dma.buffer.offset1812 = shl i32 %.tr9, 1, !dbg !16
  %dma.buffer.lds1814 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %111), !dbg !16
  %113 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1812, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
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
  %114 = or disjoint i32 %address.local310, %address.local312, !dbg !10
  %address.local316 = or disjoint i32 %114, %address.local315, !dbg !10
  %address.local319 = or disjoint i32 %address.local316, %address.local318, !dbg !10
  %address.local320 = zext nneg i32 %address.local319 to i64, !dbg !10
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local320, !dbg !10
  %115 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !10
  %116 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !10
  %117 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !10
  %118 = fmul <4 x half> %115, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !10
  %119 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !10
  %120 = fmul <4 x half> %116, splat (half 1.275630e-01), !dbg !10
  %121 = fmul <4 x half> %117, splat (half 1.275630e-01), !dbg !10
  %122 = fmul <4 x half> %119, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local332 = or disjoint i32 %address.local310, %address.local315, !dbg !20
  %address.local335 = or disjoint i32 %address.local332, %address.local318, !dbg !20
  %address.local336 = zext nneg i32 %address.local335 to i64, !dbg !20
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local336, !dbg !20
  %123 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !20
  %124 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %123, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1561 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !20
  %125 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !20
  %address.local352 = and i32 %address.local314, 88, !dbg !20
  %address.local353 = or disjoint i32 %address.local310, %address.local352, !dbg !20
  %address.local355 = and i32 %address.local314, 32, !dbg !20
  %address.local356 = sub nsw i32 %address.local353, %address.local355, !dbg !20
  %address.local359 = add nsw i32 %address.local356, %address.local318, !dbg !20
  %address.local360 = sext i32 %address.local359 to i64, !dbg !20
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local360, !dbg !20
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local361, i64 320, !dbg !20
  %126 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !20
  %127 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %125, <4 x float> %124) #8, !dbg !20
  %128 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %126, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1562 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !20
  %129 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !20
  %lds.const.gep1563 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !20
  %130 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !20
  %131 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %129, <4 x float> %128) #8, !dbg !20
  %132 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %130, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1564 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !20
  %133 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !20
  %lds.const.gep1565 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !20
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !20
  %135 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %133, <4 x float> %132) #8, !dbg !20
  %136 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %134, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1566 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !20
  %137 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !20
  %lds.const.gep1567 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !20
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !20
  %139 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %137, <4 x float> %136) #8, !dbg !20
  %140 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %138, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1568 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !20
  %141 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !20
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !20
  %142 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !20
  %143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %141, <4 x float> %140) #8, !dbg !20
  %144 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %142, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !20
  %145 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !20
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !20
  %146 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !20
  %147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %145, <4 x float> %144) #8, !dbg !20
  %148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %146, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !20
  %149 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !20
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !20
  %150 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !20
  %151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %149, <4 x float> %148) #8, !dbg !20
  %152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %150, <4 x float> zeroinitializer) #8, !dbg !20
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !20
  %153 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !20
  %154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %153, <4 x float> %152) #8, !dbg !20
  %155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %123, <4 x float> zeroinitializer) #8, !dbg !20
  %156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %125, <4 x float> %155) #8, !dbg !20
  %157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %126, <4 x float> zeroinitializer) #8, !dbg !20
  %158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %129, <4 x float> %157) #8, !dbg !20
  %159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %130, <4 x float> zeroinitializer) #8, !dbg !20
  %160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %133, <4 x float> %159) #8, !dbg !20
  %161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %134, <4 x float> zeroinitializer) #8, !dbg !20
  %162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %137, <4 x float> %161) #8, !dbg !20
  %163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %138, <4 x float> zeroinitializer) #8, !dbg !20
  %164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %141, <4 x float> %163) #8, !dbg !20
  %165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %142, <4 x float> zeroinitializer) #8, !dbg !20
  %166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %145, <4 x float> %165) #8, !dbg !20
  %167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %146, <4 x float> zeroinitializer) #8, !dbg !20
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %149, <4 x float> %167) #8, !dbg !20
  %169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %150, <4 x float> zeroinitializer) #8, !dbg !20
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %153, <4 x float> %169) #8, !dbg !20
  %171 = or disjoint i64 %address.local267, %59, !dbg !9
  %.tr10 = trunc i64 %171 to i32, !dbg !9
  %dma.buffer.offset1809 = shl i32 %.tr10, 1, !dbg !9
  %172 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1809, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %173 = or disjoint i64 %address.local279, %59, !dbg !9
  %.tr11 = trunc i64 %173 to i32, !dbg !9
  %dma.buffer.offset1806 = shl i32 %.tr11, 1, !dbg !9
  %174 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1806, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %175 = add nuw nsw i64 %82, 1024, !dbg !16
  %176 = lshr i64 %175, 12, !dbg !16
  %177 = add nuw nsw i64 %176, %15, !dbg !16
  %178 = add nsw i64 %86, %48, !dbg !16
  %179 = shl i64 %177, 19, !dbg !16
  %180 = and i64 %179, 16252928, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %181 = add i64 %92, %86, !dbg !16
  %.idx15 = add i64 %181, %48, !dbg !16
  %182 = add i64 %.idx15, %180, !dbg !16
  %183 = or disjoint i64 %address.local267, %60, !dbg !9
  %.tr22 = trunc i64 %183 to i32, !dbg !9
  %dma.buffer.offset1797 = shl i32 %.tr22, 1, !dbg !9
  %184 = or disjoint i64 %address.local279, %60, !dbg !9
  %.tr23 = trunc i64 %184 to i32, !dbg !9
  %dma.buffer.offset1794 = shl i32 %.tr23, 1, !dbg !9
  %185 = add nuw nsw i64 %82, 2048, !dbg !16
  %186 = lshr i64 %185, 12, !dbg !16
  %187 = add nuw nsw i64 %186, %15, !dbg !16
  %188 = insertelement <2 x i64> poison, i64 %178, i64 0, !dbg !16
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
  %200 = sub i64 %182, %199, !dbg !16
  %201 = add i64 %address.local293, %200, !dbg !16
  %.tr16 = trunc i64 %201 to i32, !dbg !16
  %202 = shl i32 %.tr16, 1, !dbg !16
  %dma.buffer.offset1803 = add i32 %202, 64, !dbg !16
  %203 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1803, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %204 = add i64 %address.local307, %200, !dbg !16
  %.tr21 = trunc i64 %204 to i32, !dbg !16
  %205 = shl i32 %.tr21, 1, !dbg !16
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
  %216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %215, <4 x float> %127) #8, !dbg !20
  %217 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !20
  %218 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !20
  %219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %217, <4 x float> %216) #8, !dbg !20
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %218, <4 x float> %131) #8, !dbg !20
  %221 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !20
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !20
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %221, <4 x float> %220) #8, !dbg !20
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %222, <4 x float> %135) #8, !dbg !20
  %225 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !20
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !20
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %225, <4 x float> %224) #8, !dbg !20
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %226, <4 x float> %139) #8, !dbg !20
  %229 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !20
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !20
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %229, <4 x float> %228) #8, !dbg !20
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %230, <4 x float> %143) #8, !dbg !20
  %233 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !20
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !20
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %233, <4 x float> %232) #8, !dbg !20
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %234, <4 x float> %147) #8, !dbg !20
  %237 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !20
  %238 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !20
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %237, <4 x float> %236) #8, !dbg !20
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %238, <4 x float> %151) #8, !dbg !20
  %241 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !20
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !20
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %241, <4 x float> %240) #8, !dbg !20
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %242, <4 x float> %154) #8, !dbg !20
  %245 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !20
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %245, <4 x float> %244) #8, !dbg !20
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %215, <4 x float> %156) #8, !dbg !20
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %217, <4 x float> %247) #8, !dbg !20
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %218, <4 x float> %158) #8, !dbg !20
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %221, <4 x float> %249) #8, !dbg !20
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %222, <4 x float> %160) #8, !dbg !20
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %225, <4 x float> %251) #8, !dbg !20
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %226, <4 x float> %162) #8, !dbg !20
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %229, <4 x float> %253) #8, !dbg !20
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %230, <4 x float> %164) #8, !dbg !20
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %233, <4 x float> %255) #8, !dbg !20
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %234, <4 x float> %166) #8, !dbg !20
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %237, <4 x float> %257) #8, !dbg !20
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %238, <4 x float> %168) #8, !dbg !20
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %214, <4 x half> %241, <4 x float> %259) #8, !dbg !20
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %242, <4 x float> %170) #8, !dbg !20
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
  %271 = add i64 %.idx15, %270, !dbg !16
  %272 = shl nsw i64 %268, 7, !dbg !16
  %273 = sub i64 %271, %272, !dbg !16
  %274 = add i64 %address.local293, %273, !dbg !16
  %.tr28 = trunc i64 %274 to i32, !dbg !16
  %275 = shl i32 %.tr28, 1, !dbg !16
  %dma.buffer.offset1791 = add i32 %275, 128, !dbg !16
  %276 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1791, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %277 = add i64 %address.local307, %273, !dbg !16
  %.tr33 = trunc i64 %277 to i32, !dbg !16
  %278 = shl i32 %.tr33, 1, !dbg !16
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
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %288, <4 x float> %219) #8, !dbg !20
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !20
  %291 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !20
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %290, <4 x float> %289) #8, !dbg !20
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %291, <4 x float> %223) #8, !dbg !20
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !20
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !20
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %294, <4 x float> %293) #8, !dbg !20
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %295, <4 x float> %227) #8, !dbg !20
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !20
  %299 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !20
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %298, <4 x float> %297) #8, !dbg !20
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %299, <4 x float> %231) #8, !dbg !20
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !20
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !20
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %302, <4 x float> %301) #8, !dbg !20
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %303, <4 x float> %235) #8, !dbg !20
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !20
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !20
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %306, <4 x float> %305) #8, !dbg !20
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %307, <4 x float> %239) #8, !dbg !20
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !20
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !20
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %310, <4 x float> %309) #8, !dbg !20
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %311, <4 x float> %243) #8, !dbg !20
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !20
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !20
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %314, <4 x float> %313) #8, !dbg !20
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %283, <4 x half> %315, <4 x float> %246) #8, !dbg !20
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !20
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %318, <4 x float> %317) #8, !dbg !20
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %288, <4 x float> %248) #8, !dbg !20
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %290, <4 x float> %320) #8, !dbg !20
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %291, <4 x float> %250) #8, !dbg !20
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %294, <4 x float> %322) #8, !dbg !20
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %295, <4 x float> %252) #8, !dbg !20
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %298, <4 x float> %324) #8, !dbg !20
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %299, <4 x float> %254) #8, !dbg !20
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %302, <4 x float> %326) #8, !dbg !20
  %328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %303, <4 x float> %256) #8, !dbg !20
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %306, <4 x float> %328) #8, !dbg !20
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %307, <4 x float> %258) #8, !dbg !20
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %310, <4 x float> %330) #8, !dbg !20
  %332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %311, <4 x float> %260) #8, !dbg !20
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %314, <4 x float> %332) #8, !dbg !20
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %315, <4 x float> %262) #8, !dbg !20
  %335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %318, <4 x float> %334) #8, !dbg !20
  %336 = or disjoint i64 %address.local267, %61, !dbg !9
  %.tr34 = trunc i64 %336 to i32, !dbg !9
  %dma.buffer.offset1785 = shl i32 %.tr34, 1, !dbg !9
  %337 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1785, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %338 = or disjoint i64 %address.local279, %61, !dbg !9
  %.tr35 = trunc i64 %338 to i32, !dbg !9
  %dma.buffer.offset1782 = shl i32 %.tr35, 1, !dbg !9
  %339 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1782, <4 x i32> %dma.buffer.rsrc1822) #9, !dbg !9
  %340 = add nuw nsw i64 %82, 3072, !dbg !16
  %341 = lshr i64 %340, 12, !dbg !16
  %342 = add nuw nsw i64 %341, %15, !dbg !16
  %343 = add nsw i64 %178, 96, !dbg !16
  %344 = icmp slt i64 %178, -96, !dbg !16
  %345 = sub nsw i64 -97, %178, !dbg !16
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
  %352 = add i64 %.idx15, %351, !dbg !16
  %353 = shl nsw i64 %349, 7, !dbg !16
  %354 = sub i64 %352, %353, !dbg !16
  %355 = add i64 %address.local293, %354, !dbg !16
  %.tr40 = trunc i64 %355 to i32, !dbg !16
  %356 = shl i32 %.tr40, 1, !dbg !16
  %dma.buffer.offset1779 = add i32 %356, 192, !dbg !16
  %357 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1779, <4 x i32> %dma.buffer.rsrc1816) #9, !dbg !16
  %358 = add i64 %address.local307, %354, !dbg !16
  %.tr45 = trunc i64 %358 to i32, !dbg !16
  %359 = shl i32 %.tr45, 1, !dbg !16
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
  %370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %369, <4 x float> %292) #8, !dbg !20
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !20
  %372 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !20
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %371, <4 x float> %370) #8, !dbg !20
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %372, <4 x float> %296) #8, !dbg !20
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !20
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !20
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %375, <4 x float> %374) #8, !dbg !20
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %376, <4 x float> %300) #8, !dbg !20
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !20
  %380 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !20
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %379, <4 x float> %378) #8, !dbg !20
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %380, <4 x float> %304) #8, !dbg !20
  %383 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !20
  %384 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !20
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %383, <4 x float> %382) #8, !dbg !20
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %384, <4 x float> %308) #8, !dbg !20
  %387 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !20
  %388 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !20
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %387, <4 x float> %386) #8, !dbg !20
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %388, <4 x float> %312) #8, !dbg !20
  %391 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !20
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !20
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %391, <4 x float> %390) #8, !dbg !20
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %392, <4 x float> %316) #8, !dbg !20
  %395 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !20
  %396 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !20
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %395, <4 x float> %394) #8, !dbg !20
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %364, <4 x half> %396, <4 x float> %319) #8, !dbg !20
  %399 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !20
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %399, <4 x float> %398) #8, !dbg !20
  %401 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %369, <4 x float> %321) #8, !dbg !20
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %371, <4 x float> %401) #8, !dbg !20
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %372, <4 x float> %323) #8, !dbg !20
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %375, <4 x float> %403) #8, !dbg !20
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %376, <4 x float> %325) #8, !dbg !20
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %379, <4 x float> %405) #8, !dbg !20
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %380, <4 x float> %327) #8, !dbg !20
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %383, <4 x float> %407) #8, !dbg !20
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %384, <4 x float> %329) #8, !dbg !20
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %387, <4 x float> %409) #8, !dbg !20
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %388, <4 x float> %331) #8, !dbg !20
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %391, <4 x float> %411) #8, !dbg !20
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %392, <4 x float> %333) #8, !dbg !20
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %395, <4 x float> %413) #8, !dbg !20
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %396, <4 x float> %335) #8, !dbg !20
  %416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %368, <4 x half> %399, <4 x float> %415) #8, !dbg !20
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %417 = extractelement <4 x float> %416, i64 3, !dbg !21
  %418 = extractelement <4 x float> %416, i64 2, !dbg !21
  %419 = extractelement <4 x float> %416, i64 1, !dbg !21
  %420 = extractelement <4 x float> %416, i64 0, !dbg !21
  %421 = extractelement <4 x float> %414, i64 3, !dbg !21
  %422 = extractelement <4 x float> %414, i64 2, !dbg !21
  %423 = extractelement <4 x float> %414, i64 1, !dbg !21
  %424 = extractelement <4 x float> %414, i64 0, !dbg !21
  %425 = extractelement <4 x float> %412, i64 3, !dbg !21
  %426 = extractelement <4 x float> %412, i64 2, !dbg !21
  %427 = extractelement <4 x float> %412, i64 1, !dbg !21
  %428 = extractelement <4 x float> %412, i64 0, !dbg !21
  %429 = extractelement <4 x float> %410, i64 3, !dbg !21
  %430 = extractelement <4 x float> %410, i64 2, !dbg !21
  %431 = extractelement <4 x float> %410, i64 1, !dbg !21
  %432 = extractelement <4 x float> %410, i64 0, !dbg !21
  %433 = extractelement <4 x float> %408, i64 3, !dbg !21
  %434 = extractelement <4 x float> %408, i64 2, !dbg !21
  %435 = extractelement <4 x float> %408, i64 1, !dbg !21
  %436 = extractelement <4 x float> %408, i64 0, !dbg !21
  %437 = extractelement <4 x float> %406, i64 3, !dbg !21
  %438 = extractelement <4 x float> %406, i64 2, !dbg !21
  %439 = extractelement <4 x float> %406, i64 1, !dbg !21
  %440 = extractelement <4 x float> %406, i64 0, !dbg !21
  %441 = extractelement <4 x float> %404, i64 3, !dbg !21
  %442 = extractelement <4 x float> %404, i64 2, !dbg !21
  %443 = extractelement <4 x float> %404, i64 1, !dbg !21
  %444 = extractelement <4 x float> %404, i64 0, !dbg !21
  %445 = extractelement <4 x float> %402, i64 3, !dbg !21
  %446 = extractelement <4 x float> %402, i64 2, !dbg !21
  %447 = extractelement <4 x float> %402, i64 1, !dbg !21
  %448 = extractelement <4 x float> %402, i64 0, !dbg !21
  %449 = extractelement <4 x float> %400, i64 3, !dbg !21
  %450 = extractelement <4 x float> %400, i64 2, !dbg !21
  %451 = extractelement <4 x float> %400, i64 1, !dbg !21
  %452 = extractelement <4 x float> %400, i64 0, !dbg !21
  %453 = extractelement <4 x float> %397, i64 3, !dbg !21
  %454 = extractelement <4 x float> %397, i64 2, !dbg !21
  %455 = extractelement <4 x float> %397, i64 1, !dbg !21
  %456 = extractelement <4 x float> %397, i64 0, !dbg !21
  %457 = extractelement <4 x float> %393, i64 3, !dbg !21
  %458 = extractelement <4 x float> %393, i64 2, !dbg !21
  %459 = extractelement <4 x float> %393, i64 1, !dbg !21
  %460 = extractelement <4 x float> %393, i64 0, !dbg !21
  %461 = extractelement <4 x float> %389, i64 3, !dbg !21
  %462 = extractelement <4 x float> %389, i64 2, !dbg !21
  %463 = extractelement <4 x float> %389, i64 1, !dbg !21
  %464 = extractelement <4 x float> %389, i64 0, !dbg !21
  %465 = extractelement <4 x float> %385, i64 3, !dbg !21
  %466 = extractelement <4 x float> %385, i64 2, !dbg !21
  %467 = extractelement <4 x float> %385, i64 1, !dbg !21
  %468 = extractelement <4 x float> %385, i64 0, !dbg !21
  %469 = extractelement <4 x float> %381, i64 3, !dbg !21
  %470 = extractelement <4 x float> %381, i64 2, !dbg !21
  %471 = extractelement <4 x float> %381, i64 1, !dbg !21
  %472 = extractelement <4 x float> %381, i64 0, !dbg !21
  %473 = extractelement <4 x float> %377, i64 3, !dbg !21
  %474 = extractelement <4 x float> %377, i64 2, !dbg !21
  %475 = extractelement <4 x float> %377, i64 1, !dbg !21
  %476 = extractelement <4 x float> %377, i64 0, !dbg !21
  %477 = extractelement <4 x float> %373, i64 3, !dbg !21
  %478 = extractelement <4 x float> %373, i64 2, !dbg !21
  %479 = extractelement <4 x float> %373, i64 1, !dbg !21
  %480 = extractelement <4 x float> %373, i64 0, !dbg !21
  %address.local415 = and i32 %address.local317, 6144, !dbg !18
  %address.local416 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local417 = and i32 %address.local416, 1920, !dbg !18
  %address.local418 = or disjoint i32 %address.local415, %address.local417, !dbg !18
  %address.local419 = zext nneg i32 %address.local418 to i64, !dbg !18
  %address.local420 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !18
  %481 = lshr i32 %address.thread, 4, !dbg !10
  %address.local422 = zext nneg i32 %481 to i64, !dbg !10
  %address.local423 = and i64 %address.local422, 3, !dbg !10
  %address.local424 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local423, !dbg !18
  store float %480, ptr addrspace(3) %address.local424, align 4, !dbg !18
  %address.local426 = or disjoint i64 %address.local423, 4, !dbg !18
  %address.local427 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local426, !dbg !18
  store float %479, ptr addrspace(3) %address.local427, align 4, !dbg !18
  %address.local429 = or disjoint i64 %address.local423, 8, !dbg !18
  %address.local430 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local429, !dbg !18
  store float %478, ptr addrspace(3) %address.local430, align 4, !dbg !18
  %address.local432 = or i64 %address.local422, 12, !dbg !18
  %address.local433 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local432, !dbg !18
  store float %477, ptr addrspace(3) %address.local433, align 4, !dbg !18
  %address.local435 = or disjoint i64 %address.local423, 16, !dbg !18
  %address.local436 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local435, !dbg !18
  store float %476, ptr addrspace(3) %address.local436, align 4, !dbg !18
  %address.local438 = or disjoint i64 %address.local423, 20, !dbg !18
  %address.local439 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local438, !dbg !18
  store float %475, ptr addrspace(3) %address.local439, align 4, !dbg !18
  %address.local441 = or disjoint i64 %address.local423, 24, !dbg !18
  %address.local442 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local441, !dbg !18
  store float %474, ptr addrspace(3) %address.local442, align 4, !dbg !18
  %address.local444 = or i64 %address.local422, 28, !dbg !18
  %address.local445 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local444, !dbg !18
  store float %473, ptr addrspace(3) %address.local445, align 4, !dbg !18
  %address.local447 = or disjoint i64 %address.local423, 32, !dbg !18
  %address.local448 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local447, !dbg !18
  store float %472, ptr addrspace(3) %address.local448, align 4, !dbg !18
  %address.local450 = or disjoint i64 %address.local423, 36, !dbg !18
  %address.local451 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local450, !dbg !18
  store float %471, ptr addrspace(3) %address.local451, align 4, !dbg !18
  %address.local453 = or disjoint i64 %address.local423, 40, !dbg !18
  %address.local454 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local453, !dbg !18
  store float %470, ptr addrspace(3) %address.local454, align 4, !dbg !18
  %address.local456 = or i64 %address.local422, 44, !dbg !18
  %address.local457 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local456, !dbg !18
  store float %469, ptr addrspace(3) %address.local457, align 4, !dbg !18
  %address.local459 = or disjoint i64 %address.local423, 48, !dbg !18
  %address.local460 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local459, !dbg !18
  store float %468, ptr addrspace(3) %address.local460, align 4, !dbg !18
  %address.local462 = or disjoint i64 %address.local423, 52, !dbg !18
  %address.local463 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local462, !dbg !18
  store float %467, ptr addrspace(3) %address.local463, align 4, !dbg !18
  %address.local465 = or disjoint i64 %address.local423, 56, !dbg !18
  %address.local466 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local465, !dbg !18
  store float %466, ptr addrspace(3) %address.local466, align 4, !dbg !18
  %address.local468 = or i64 %address.local422, 60, !dbg !18
  %address.local469 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local468, !dbg !18
  store float %465, ptr addrspace(3) %address.local469, align 4, !dbg !18
  %address.local471 = or disjoint i64 %address.local423, 64, !dbg !18
  %address.local472 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local471, !dbg !18
  store float %464, ptr addrspace(3) %address.local472, align 4, !dbg !18
  %address.local474 = or disjoint i64 %address.local423, 68, !dbg !18
  %address.local475 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local474, !dbg !18
  store float %463, ptr addrspace(3) %address.local475, align 4, !dbg !18
  %address.local477 = or disjoint i64 %address.local423, 72, !dbg !18
  %address.local478 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local477, !dbg !18
  store float %462, ptr addrspace(3) %address.local478, align 4, !dbg !18
  %address.local480 = or i64 %address.local422, 76, !dbg !18
  %address.local481 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local480, !dbg !18
  store float %461, ptr addrspace(3) %address.local481, align 4, !dbg !18
  %address.local483 = or disjoint i64 %address.local423, 80, !dbg !18
  %address.local484 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local483, !dbg !18
  store float %460, ptr addrspace(3) %address.local484, align 4, !dbg !18
  %address.local486 = or disjoint i64 %address.local423, 84, !dbg !18
  %address.local487 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local486, !dbg !18
  store float %459, ptr addrspace(3) %address.local487, align 4, !dbg !18
  %address.local489 = or disjoint i64 %address.local423, 88, !dbg !18
  %address.local490 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local489, !dbg !18
  store float %458, ptr addrspace(3) %address.local490, align 4, !dbg !18
  %address.local492 = or i64 %address.local422, 92, !dbg !18
  %address.local493 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local492, !dbg !18
  store float %457, ptr addrspace(3) %address.local493, align 4, !dbg !18
  %address.local495 = or disjoint i64 %address.local423, 96, !dbg !18
  %address.local496 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local495, !dbg !18
  store float %456, ptr addrspace(3) %address.local496, align 4, !dbg !18
  %address.local498 = or disjoint i64 %address.local423, 100, !dbg !18
  %address.local499 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local498, !dbg !18
  store float %455, ptr addrspace(3) %address.local499, align 4, !dbg !18
  %address.local501 = or disjoint i64 %address.local423, 104, !dbg !18
  %address.local502 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local501, !dbg !18
  store float %454, ptr addrspace(3) %address.local502, align 4, !dbg !18
  %address.local504 = or i64 %address.local422, 108, !dbg !18
  %address.local505 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local504, !dbg !18
  store float %453, ptr addrspace(3) %address.local505, align 4, !dbg !18
  %address.local507 = or disjoint i64 %address.local423, 112, !dbg !18
  %address.local508 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local507, !dbg !18
  store float %452, ptr addrspace(3) %address.local508, align 4, !dbg !18
  %address.local510 = or disjoint i64 %address.local423, 116, !dbg !18
  %address.local511 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local510, !dbg !18
  store float %451, ptr addrspace(3) %address.local511, align 4, !dbg !18
  %address.local513 = or disjoint i64 %address.local423, 120, !dbg !18
  %address.local514 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local513, !dbg !18
  store float %450, ptr addrspace(3) %address.local514, align 4, !dbg !18
  %address.local516 = or i64 %address.local422, 124, !dbg !18
  %address.local517 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local516, !dbg !18
  store float %449, ptr addrspace(3) %address.local517, align 4, !dbg !18
  %address.local525 = getelementptr inbounds nuw i8, ptr addrspace(3) %address.local420, i64 32768, !dbg !18
  %address.local526 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local423, !dbg !18
  store float %448, ptr addrspace(3) %address.local526, align 4, !dbg !18
  %address.local528 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local426, !dbg !18
  store float %447, ptr addrspace(3) %address.local528, align 4, !dbg !18
  %address.local530 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local429, !dbg !18
  store float %446, ptr addrspace(3) %address.local530, align 4, !dbg !18
  %address.local532 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local432, !dbg !18
  store float %445, ptr addrspace(3) %address.local532, align 4, !dbg !18
  %address.local534 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local435, !dbg !18
  store float %444, ptr addrspace(3) %address.local534, align 4, !dbg !18
  %address.local536 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local438, !dbg !18
  store float %443, ptr addrspace(3) %address.local536, align 4, !dbg !18
  %address.local538 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local441, !dbg !18
  store float %442, ptr addrspace(3) %address.local538, align 4, !dbg !18
  %address.local540 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local444, !dbg !18
  store float %441, ptr addrspace(3) %address.local540, align 4, !dbg !18
  %address.local542 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local447, !dbg !18
  store float %440, ptr addrspace(3) %address.local542, align 4, !dbg !18
  %address.local544 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local450, !dbg !18
  store float %439, ptr addrspace(3) %address.local544, align 4, !dbg !18
  %address.local546 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local453, !dbg !18
  store float %438, ptr addrspace(3) %address.local546, align 4, !dbg !18
  %address.local548 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local456, !dbg !18
  store float %437, ptr addrspace(3) %address.local548, align 4, !dbg !18
  %address.local550 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local459, !dbg !18
  store float %436, ptr addrspace(3) %address.local550, align 4, !dbg !18
  %address.local552 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local462, !dbg !18
  store float %435, ptr addrspace(3) %address.local552, align 4, !dbg !18
  %address.local554 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local465, !dbg !18
  store float %434, ptr addrspace(3) %address.local554, align 4, !dbg !18
  %address.local556 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local468, !dbg !18
  store float %433, ptr addrspace(3) %address.local556, align 4, !dbg !18
  %address.local558 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local471, !dbg !18
  store float %432, ptr addrspace(3) %address.local558, align 4, !dbg !18
  %address.local560 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local474, !dbg !18
  store float %431, ptr addrspace(3) %address.local560, align 4, !dbg !18
  %address.local562 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local477, !dbg !18
  store float %430, ptr addrspace(3) %address.local562, align 4, !dbg !18
  %address.local564 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local480, !dbg !18
  store float %429, ptr addrspace(3) %address.local564, align 4, !dbg !18
  %address.local566 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local483, !dbg !18
  store float %428, ptr addrspace(3) %address.local566, align 4, !dbg !18
  %address.local568 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local486, !dbg !18
  store float %427, ptr addrspace(3) %address.local568, align 4, !dbg !18
  %address.local570 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local489, !dbg !18
  store float %426, ptr addrspace(3) %address.local570, align 4, !dbg !18
  %address.local572 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local492, !dbg !18
  store float %425, ptr addrspace(3) %address.local572, align 4, !dbg !18
  %address.local574 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local495, !dbg !18
  store float %424, ptr addrspace(3) %address.local574, align 4, !dbg !18
  %address.local576 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local498, !dbg !18
  store float %423, ptr addrspace(3) %address.local576, align 4, !dbg !18
  %address.local578 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local501, !dbg !18
  store float %422, ptr addrspace(3) %address.local578, align 4, !dbg !18
  %address.local580 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local504, !dbg !18
  store float %421, ptr addrspace(3) %address.local580, align 4, !dbg !18
  %address.local582 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local507, !dbg !18
  store float %420, ptr addrspace(3) %address.local582, align 4, !dbg !18
  %address.local584 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local510, !dbg !18
  store float %419, ptr addrspace(3) %address.local584, align 4, !dbg !18
  %address.local586 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local513, !dbg !18
  store float %418, ptr addrspace(3) %address.local586, align 4, !dbg !18
  %address.local588 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local516, !dbg !18
  store float %417, ptr addrspace(3) %address.local588, align 4, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %482 = and i32 %address.local309, 12, !dbg !10
  %address.local600 = zext nneg i32 %482 to i64, !dbg !10
  %address.local601 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local600, !dbg !18
  %483 = load <4 x float>, ptr addrspace(3) %address.local601, align 16, !dbg !18
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local601, i64 64, !dbg !18
  %484 = load <4 x float>, ptr addrspace(3) %lds.const.gep1685, align 16, !dbg !18
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local601, i64 128, !dbg !18
  %485 = load <4 x float>, ptr addrspace(3) %lds.const.gep1689, align 16, !dbg !18
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local601, i64 192, !dbg !18
  %486 = load <4 x float>, ptr addrspace(3) %lds.const.gep1693, align 16, !dbg !18
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local601, i64 256, !dbg !18
  %487 = load <4 x float>, ptr addrspace(3) %lds.const.gep1697, align 16, !dbg !18
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local601, i64 320, !dbg !18
  %488 = load <4 x float>, ptr addrspace(3) %lds.const.gep1701, align 16, !dbg !18
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local601, i64 384, !dbg !18
  %489 = load <4 x float>, ptr addrspace(3) %lds.const.gep1705, align 16, !dbg !18
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local601, i64 448, !dbg !18
  %490 = load <4 x float>, ptr addrspace(3) %lds.const.gep1709, align 16, !dbg !18
  %lds.const.gep1713 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32768, !dbg !18
  %491 = load <4 x float>, ptr addrspace(3) %lds.const.gep1713, align 16, !dbg !18
  %lds.const.gep1717 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32832, !dbg !18
  %492 = load <4 x float>, ptr addrspace(3) %lds.const.gep1717, align 16, !dbg !18
  %lds.const.gep1721 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32896, !dbg !18
  %493 = load <4 x float>, ptr addrspace(3) %lds.const.gep1721, align 16, !dbg !18
  %lds.const.gep1725 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32960, !dbg !18
  %494 = load <4 x float>, ptr addrspace(3) %lds.const.gep1725, align 16, !dbg !18
  %lds.const.gep1729 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33024, !dbg !18
  %495 = load <4 x float>, ptr addrspace(3) %lds.const.gep1729, align 16, !dbg !18
  %lds.const.gep1733 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33088, !dbg !18
  %496 = load <4 x float>, ptr addrspace(3) %lds.const.gep1733, align 16, !dbg !18
  %lds.const.gep1737 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33152, !dbg !18
  %497 = load <4 x float>, ptr addrspace(3) %lds.const.gep1737, align 16, !dbg !18
  %lds.const.gep1741 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33216, !dbg !18
  %498 = load <4 x float>, ptr addrspace(3) %lds.const.gep1741, align 16, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %499 = or disjoint i64 %83, 127, !dbg !12
  %500 = icmp uge i64 %19, %499, !dbg !12
  %501 = and i1 %40, %500, !dbg !12
  br i1 %501, label %586, label %502, !dbg !12

502:                                              ; preds = %75
  %503 = or disjoint i64 %83, %37, !dbg !12
  %504 = lshr i64 %82, 25, !dbg !19
  %compare.high1303 = trunc nuw i64 %504 to i32, !dbg !19
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1303, !dbg !19
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1303, !dbg !19
  %compare.low.base1304 = trunc i64 %503 to i32, !dbg !19
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
  %505 = insertelement <32 x i32> poison, i32 %compare.low.base1304, i64 0, !dbg !19
  %506 = insertelement <32 x i32> %505, i32 %compare.local.base, i64 1, !dbg !19
  %507 = insertelement <32 x i32> %506, i32 %compare.local.base1307, i64 2, !dbg !19
  %508 = insertelement <32 x i32> %507, i32 %compare.local.base1311, i64 3, !dbg !19
  %509 = insertelement <32 x i32> %508, i32 %compare.local.base1315, i64 4, !dbg !19
  %510 = insertelement <32 x i32> %509, i32 %compare.local.base1319, i64 5, !dbg !19
  %511 = insertelement <32 x i32> %510, i32 %compare.local.base1323, i64 6, !dbg !19
  %512 = insertelement <32 x i32> %511, i32 %compare.local.base1327, i64 7, !dbg !19
  %513 = insertelement <32 x i32> %512, i32 %compare.local.base1331, i64 8, !dbg !19
  %514 = insertelement <32 x i32> %513, i32 %compare.local.base1335, i64 9, !dbg !19
  %515 = insertelement <32 x i32> %514, i32 %compare.local.base1339, i64 10, !dbg !19
  %516 = insertelement <32 x i32> %515, i32 %compare.local.base1343, i64 11, !dbg !19
  %517 = insertelement <32 x i32> %516, i32 %compare.local.base1347, i64 12, !dbg !19
  %518 = insertelement <32 x i32> %517, i32 %compare.local.base1351, i64 13, !dbg !19
  %519 = insertelement <32 x i32> %518, i32 %compare.local.base1355, i64 14, !dbg !19
  %520 = insertelement <32 x i32> %519, i32 %compare.local.base1359, i64 15, !dbg !19
  %521 = insertelement <32 x i32> %520, i32 %compare.local.base1363, i64 16, !dbg !19
  %522 = insertelement <32 x i32> %521, i32 %compare.local.base1367, i64 17, !dbg !19
  %523 = insertelement <32 x i32> %522, i32 %compare.local.base1371, i64 18, !dbg !19
  %524 = insertelement <32 x i32> %523, i32 %compare.local.base1375, i64 19, !dbg !19
  %525 = insertelement <32 x i32> %524, i32 %compare.local.base1379, i64 20, !dbg !19
  %526 = insertelement <32 x i32> %525, i32 %compare.local.base1383, i64 21, !dbg !19
  %527 = insertelement <32 x i32> %526, i32 %compare.local.base1387, i64 22, !dbg !19
  %528 = insertelement <32 x i32> %527, i32 %compare.local.base1391, i64 23, !dbg !19
  %529 = insertelement <32 x i32> %528, i32 %compare.local.base1395, i64 24, !dbg !19
  %530 = insertelement <32 x i32> %529, i32 %compare.local.base1399, i64 25, !dbg !19
  %531 = insertelement <32 x i32> %530, i32 %compare.local.base1403, i64 26, !dbg !19
  %532 = insertelement <32 x i32> %531, i32 %compare.local.base1407, i64 27, !dbg !19
  %533 = insertelement <32 x i32> %532, i32 %compare.local.base1411, i64 28, !dbg !19
  %534 = insertelement <32 x i32> %533, i32 %compare.local.base1415, i64 29, !dbg !19
  %535 = insertelement <32 x i32> %534, i32 %compare.local.base1419, i64 30, !dbg !19
  %536 = insertelement <32 x i32> %535, i32 %compare.local.base1423, i64 31, !dbg !19
  %537 = or <32 x i32> %536, <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 48, i32 49, i32 50, i32 51, i32 64, i32 65, i32 66, i32 67, i32 80, i32 81, i32 82, i32 83, i32 96, i32 97, i32 98, i32 99, i32 112, i32 113, i32 114, i32 115>, !dbg !19
  %538 = icmp ugt <32 x i32> %74, %537, !dbg !19
  %539 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !19
  %540 = shufflevector <32 x i1> %539, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %541 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !19
  %542 = shufflevector <32 x i1> %541, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %543 = select <32 x i1> %540, <32 x i1> %538, <32 x i1> %542, !dbg !19
  %544 = select <32 x i1> %543, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !22
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
  %545 = insertelement <32 x i32> %505, i32 %compare.local.base1435, i64 1, !dbg !19
  %546 = insertelement <32 x i32> %545, i32 %compare.local.base1439, i64 2, !dbg !19
  %547 = insertelement <32 x i32> %546, i32 %compare.local.base1443, i64 3, !dbg !19
  %548 = insertelement <32 x i32> %547, i32 %compare.local.base1447, i64 4, !dbg !19
  %549 = insertelement <32 x i32> %548, i32 %compare.local.base1451, i64 5, !dbg !19
  %550 = insertelement <32 x i32> %549, i32 %compare.local.base1455, i64 6, !dbg !19
  %551 = insertelement <32 x i32> %550, i32 %compare.local.base1459, i64 7, !dbg !19
  %552 = insertelement <32 x i32> %551, i32 %compare.local.base1463, i64 8, !dbg !19
  %553 = insertelement <32 x i32> %552, i32 %compare.local.base1467, i64 9, !dbg !19
  %554 = insertelement <32 x i32> %553, i32 %compare.local.base1471, i64 10, !dbg !19
  %555 = insertelement <32 x i32> %554, i32 %compare.local.base1475, i64 11, !dbg !19
  %556 = insertelement <32 x i32> %555, i32 %compare.local.base1479, i64 12, !dbg !19
  %557 = insertelement <32 x i32> %556, i32 %compare.local.base1483, i64 13, !dbg !19
  %558 = insertelement <32 x i32> %557, i32 %compare.local.base1487, i64 14, !dbg !19
  %559 = insertelement <32 x i32> %558, i32 %compare.local.base1491, i64 15, !dbg !19
  %560 = insertelement <32 x i32> %559, i32 %compare.local.base1495, i64 16, !dbg !19
  %561 = insertelement <32 x i32> %560, i32 %compare.local.base1499, i64 17, !dbg !19
  %562 = insertelement <32 x i32> %561, i32 %compare.local.base1503, i64 18, !dbg !19
  %563 = insertelement <32 x i32> %562, i32 %compare.local.base1507, i64 19, !dbg !19
  %564 = insertelement <32 x i32> %563, i32 %compare.local.base1511, i64 20, !dbg !19
  %565 = insertelement <32 x i32> %564, i32 %compare.local.base1515, i64 21, !dbg !19
  %566 = insertelement <32 x i32> %565, i32 %compare.local.base1519, i64 22, !dbg !19
  %567 = insertelement <32 x i32> %566, i32 %compare.local.base1523, i64 23, !dbg !19
  %568 = insertelement <32 x i32> %567, i32 %compare.local.base1527, i64 24, !dbg !19
  %569 = insertelement <32 x i32> %568, i32 %compare.local.base1531, i64 25, !dbg !19
  %570 = insertelement <32 x i32> %569, i32 %compare.local.base1535, i64 26, !dbg !19
  %571 = insertelement <32 x i32> %570, i32 %compare.local.base1539, i64 27, !dbg !19
  %572 = insertelement <32 x i32> %571, i32 %compare.local.base1543, i64 28, !dbg !19
  %573 = insertelement <32 x i32> %572, i32 %compare.local.base1547, i64 29, !dbg !19
  %574 = insertelement <32 x i32> %573, i32 %compare.local.base1551, i64 30, !dbg !19
  %575 = insertelement <32 x i32> %574, i32 %compare.local.base1555, i64 31, !dbg !19
  %576 = or <32 x i32> %575, <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 48, i32 49, i32 50, i32 51, i32 64, i32 65, i32 66, i32 67, i32 80, i32 81, i32 82, i32 83, i32 96, i32 97, i32 98, i32 99, i32 112, i32 113, i32 114, i32 115>, !dbg !19
  %577 = icmp ugt <32 x i32> %72, %576, !dbg !19
  %578 = insertelement <32 x i1> poison, i1 %compare.high.equal1429, i64 0, !dbg !19
  %579 = shufflevector <32 x i1> %578, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %580 = insertelement <32 x i1> poison, i1 %compare.high.order1430, i64 0, !dbg !19
  %581 = shufflevector <32 x i1> %580, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !19
  %582 = select <32 x i1> %579, <32 x i1> %577, <32 x i1> %581, !dbg !19
  %583 = select <32 x i1> %582, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !22
  %584 = insertvalue [2 x <32 x float>] poison, <32 x float> %544, 0, !dbg !12
  %585 = insertvalue [2 x <32 x float>] %584, <32 x float> %583, 1, !dbg !12
  br label %586, !dbg !12

586:                                              ; preds = %502, %75
  %587 = phi [2 x <32 x float>] [ %585, %502 ], [ zeroinitializer, %75 ], !dbg !12
  %588 = extractvalue [2 x <32 x float>] %587, 0, !dbg !23
  %589 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %590 = fadd <4 x float> %483, %589, !dbg !23
  %591 = extractelement <4 x float> %590, i64 0, !dbg !23
  %592 = extractelement <4 x float> %590, i64 1, !dbg !23
  %593 = extractelement <4 x float> %590, i64 2, !dbg !23
  %594 = extractelement <4 x float> %590, i64 3, !dbg !23
  %595 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %596 = fadd <4 x float> %484, %595, !dbg !23
  %597 = extractelement <4 x float> %596, i64 0, !dbg !23
  %598 = extractelement <4 x float> %596, i64 1, !dbg !23
  %599 = extractelement <4 x float> %596, i64 2, !dbg !23
  %600 = extractelement <4 x float> %596, i64 3, !dbg !23
  %601 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !23
  %602 = fadd <4 x float> %485, %601, !dbg !23
  %603 = extractelement <4 x float> %602, i64 0, !dbg !23
  %604 = extractelement <4 x float> %602, i64 1, !dbg !23
  %605 = extractelement <4 x float> %602, i64 2, !dbg !23
  %606 = extractelement <4 x float> %602, i64 3, !dbg !23
  %607 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !23
  %608 = fadd <4 x float> %486, %607, !dbg !23
  %609 = extractelement <4 x float> %608, i64 0, !dbg !23
  %610 = extractelement <4 x float> %608, i64 1, !dbg !23
  %611 = extractelement <4 x float> %608, i64 2, !dbg !23
  %612 = extractelement <4 x float> %608, i64 3, !dbg !23
  %613 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !23
  %614 = fadd <4 x float> %487, %613, !dbg !23
  %615 = extractelement <4 x float> %614, i64 0, !dbg !23
  %616 = extractelement <4 x float> %614, i64 1, !dbg !23
  %617 = extractelement <4 x float> %614, i64 2, !dbg !23
  %618 = extractelement <4 x float> %614, i64 3, !dbg !23
  %619 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !23
  %620 = fadd <4 x float> %488, %619, !dbg !23
  %621 = extractelement <4 x float> %620, i64 0, !dbg !23
  %622 = extractelement <4 x float> %620, i64 1, !dbg !23
  %623 = extractelement <4 x float> %620, i64 2, !dbg !23
  %624 = extractelement <4 x float> %620, i64 3, !dbg !23
  %625 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !23
  %626 = fadd <4 x float> %489, %625, !dbg !23
  %627 = extractelement <4 x float> %626, i64 0, !dbg !23
  %628 = extractelement <4 x float> %626, i64 1, !dbg !23
  %629 = extractelement <4 x float> %626, i64 2, !dbg !23
  %630 = extractelement <4 x float> %626, i64 3, !dbg !23
  %631 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %632 = fadd <4 x float> %490, %631, !dbg !23
  %633 = extractelement <4 x float> %632, i64 0, !dbg !23
  %634 = extractelement <4 x float> %632, i64 1, !dbg !23
  %635 = extractelement <4 x float> %632, i64 2, !dbg !23
  %636 = extractelement <4 x float> %632, i64 3, !dbg !23
  %637 = extractvalue [2 x <32 x float>] %587, 1, !dbg !23
  %638 = shufflevector <32 x float> %637, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %639 = fadd <4 x float> %491, %638, !dbg !23
  %640 = extractelement <4 x float> %639, i64 0, !dbg !23
  %641 = extractelement <4 x float> %639, i64 1, !dbg !23
  %642 = extractelement <4 x float> %639, i64 2, !dbg !23
  %643 = extractelement <4 x float> %639, i64 3, !dbg !23
  %644 = shufflevector <32 x float> %637, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %645 = fadd <4 x float> %492, %644, !dbg !23
  %646 = extractelement <4 x float> %645, i64 0, !dbg !23
  %647 = extractelement <4 x float> %645, i64 1, !dbg !23
  %648 = extractelement <4 x float> %645, i64 2, !dbg !23
  %649 = extractelement <4 x float> %645, i64 3, !dbg !23
  %650 = shufflevector <32 x float> %637, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !23
  %651 = fadd <4 x float> %493, %650, !dbg !23
  %652 = extractelement <4 x float> %651, i64 0, !dbg !23
  %653 = extractelement <4 x float> %651, i64 1, !dbg !23
  %654 = extractelement <4 x float> %651, i64 2, !dbg !23
  %655 = extractelement <4 x float> %651, i64 3, !dbg !23
  %656 = shufflevector <32 x float> %637, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !23
  %657 = fadd <4 x float> %494, %656, !dbg !23
  %658 = extractelement <4 x float> %657, i64 0, !dbg !23
  %659 = extractelement <4 x float> %657, i64 1, !dbg !23
  %660 = extractelement <4 x float> %657, i64 2, !dbg !23
  %661 = extractelement <4 x float> %657, i64 3, !dbg !23
  %662 = shufflevector <32 x float> %637, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !23
  %663 = fadd <4 x float> %495, %662, !dbg !23
  %664 = extractelement <4 x float> %663, i64 0, !dbg !23
  %665 = extractelement <4 x float> %663, i64 1, !dbg !23
  %666 = extractelement <4 x float> %663, i64 2, !dbg !23
  %667 = extractelement <4 x float> %663, i64 3, !dbg !23
  %668 = shufflevector <32 x float> %637, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !23
  %669 = fadd <4 x float> %496, %668, !dbg !23
  %670 = extractelement <4 x float> %669, i64 0, !dbg !23
  %671 = extractelement <4 x float> %669, i64 1, !dbg !23
  %672 = extractelement <4 x float> %669, i64 2, !dbg !23
  %673 = extractelement <4 x float> %669, i64 3, !dbg !23
  %674 = shufflevector <32 x float> %637, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !23
  %675 = fadd <4 x float> %497, %674, !dbg !23
  %676 = extractelement <4 x float> %675, i64 0, !dbg !23
  %677 = extractelement <4 x float> %675, i64 1, !dbg !23
  %678 = extractelement <4 x float> %675, i64 2, !dbg !23
  %679 = extractelement <4 x float> %675, i64 3, !dbg !23
  %680 = shufflevector <32 x float> %637, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %681 = fadd <4 x float> %498, %680, !dbg !23
  %682 = extractelement <4 x float> %681, i64 0, !dbg !23
  %683 = extractelement <4 x float> %681, i64 1, !dbg !23
  %684 = extractelement <4 x float> %681, i64 2, !dbg !23
  %685 = extractelement <4 x float> %681, i64 3, !dbg !23
  %686 = tail call float @llvm.maxnum.f32(float %591, float -inf), !dbg !24, !frisk.local_reduction !5
  %local.reduce182 = tail call float @llvm.maxnum.f32(float %686, float %597), !dbg !24, !frisk.local_reduction !5
  %local.reduce183 = tail call float @llvm.maxnum.f32(float %592, float %598), !dbg !24, !frisk.local_reduction !5
  %local.reduce184 = tail call float @llvm.maxnum.f32(float %593, float %599), !dbg !24, !frisk.local_reduction !5
  %local.reduce185 = tail call float @llvm.maxnum.f32(float %594, float %600), !dbg !24, !frisk.local_reduction !5
  %local.reduce186 = tail call float @llvm.maxnum.f32(float %local.reduce182, float %603), !dbg !24, !frisk.local_reduction !5
  %local.reduce187 = tail call float @llvm.maxnum.f32(float %local.reduce183, float %604), !dbg !24, !frisk.local_reduction !5
  %local.reduce188 = tail call float @llvm.maxnum.f32(float %local.reduce184, float %605), !dbg !24, !frisk.local_reduction !5
  %local.reduce189 = tail call float @llvm.maxnum.f32(float %local.reduce185, float %606), !dbg !24, !frisk.local_reduction !5
  %local.reduce190 = tail call float @llvm.maxnum.f32(float %local.reduce186, float %609), !dbg !24, !frisk.local_reduction !5
  %local.reduce191 = tail call float @llvm.maxnum.f32(float %local.reduce187, float %610), !dbg !24, !frisk.local_reduction !5
  %local.reduce192 = tail call float @llvm.maxnum.f32(float %local.reduce188, float %611), !dbg !24, !frisk.local_reduction !5
  %local.reduce193 = tail call float @llvm.maxnum.f32(float %local.reduce189, float %612), !dbg !24, !frisk.local_reduction !5
  %local.reduce194 = tail call float @llvm.maxnum.f32(float %local.reduce190, float %615), !dbg !24, !frisk.local_reduction !5
  %local.reduce195 = tail call float @llvm.maxnum.f32(float %local.reduce191, float %616), !dbg !24, !frisk.local_reduction !5
  %local.reduce196 = tail call float @llvm.maxnum.f32(float %local.reduce192, float %617), !dbg !24, !frisk.local_reduction !5
  %local.reduce197 = tail call float @llvm.maxnum.f32(float %local.reduce193, float %618), !dbg !24, !frisk.local_reduction !5
  %local.reduce198 = tail call float @llvm.maxnum.f32(float %local.reduce194, float %621), !dbg !24, !frisk.local_reduction !5
  %local.reduce199 = tail call float @llvm.maxnum.f32(float %local.reduce195, float %622), !dbg !24, !frisk.local_reduction !5
  %local.reduce200 = tail call float @llvm.maxnum.f32(float %local.reduce196, float %623), !dbg !24, !frisk.local_reduction !5
  %local.reduce201 = tail call float @llvm.maxnum.f32(float %local.reduce197, float %624), !dbg !24, !frisk.local_reduction !5
  %local.reduce202 = tail call float @llvm.maxnum.f32(float %local.reduce198, float %627), !dbg !24, !frisk.local_reduction !5
  %local.reduce203 = tail call float @llvm.maxnum.f32(float %local.reduce199, float %628), !dbg !24, !frisk.local_reduction !5
  %local.reduce204 = tail call float @llvm.maxnum.f32(float %local.reduce200, float %629), !dbg !24, !frisk.local_reduction !5
  %local.reduce205 = tail call float @llvm.maxnum.f32(float %local.reduce201, float %630), !dbg !24, !frisk.local_reduction !5
  %local.reduce206 = tail call float @llvm.maxnum.f32(float %local.reduce202, float %633), !dbg !24, !frisk.local_reduction !5
  %local.reduce207 = tail call float @llvm.maxnum.f32(float %local.reduce203, float %634), !dbg !24, !frisk.local_reduction !5
  %local.reduce208 = tail call float @llvm.maxnum.f32(float %local.reduce204, float %635), !dbg !24, !frisk.local_reduction !5
  %local.reduce209 = tail call float @llvm.maxnum.f32(float %local.reduce205, float %636), !dbg !24, !frisk.local_reduction !5
  %local.reduce210 = tail call float @llvm.maxnum.f32(float %local.reduce206, float %local.reduce207), !dbg !24, !frisk.local_reduction !5
  %local.reduce211 = tail call float @llvm.maxnum.f32(float %local.reduce208, float %local.reduce209), !dbg !24, !frisk.local_reduction !5
  %687 = tail call float @llvm.maxnum.f32(float %local.reduce210, float %local.reduce211), !dbg !24, !frisk.local_reduction !5
  %688 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !24
  %689 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %688), !dbg !24
  %690 = add nuw nsw i32 %689, 64, !dbg !24
  %691 = xor i32 %689, 16, !dbg !24
  %692 = and i32 %690, 192, !dbg !24
  %693 = icmp samesign ult i32 %691, %692, !dbg !24
  %694 = select i1 %693, i32 %691, i32 %689, !dbg !24
  %695 = shl nuw nsw i32 %694, 2, !dbg !24
  %696 = bitcast float %687 to i32, !dbg !24
  %697 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %696), !dbg !24
  %698 = bitcast i32 %697 to float, !dbg !24
  %699 = tail call float @llvm.maxnum.f32(float %687, float %698), !dbg !24
  %700 = xor i32 %689, 32, !dbg !24
  %701 = icmp samesign ult i32 %700, %692, !dbg !24
  %702 = select i1 %701, i32 %700, i32 %689, !dbg !24
  %703 = shl nuw nsw i32 %702, 2, !dbg !24
  %704 = bitcast float %699 to i32, !dbg !24
  %705 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %704), !dbg !24
  %706 = bitcast i32 %705 to float, !dbg !24
  %707 = tail call float @llvm.maxnum.f32(float %699, float %706), !dbg !24
  %708 = extractvalue [2 x <1 x float>] %79, 0, !dbg !24
  %709 = insertelement <1 x float> poison, float %707, i64 0, !dbg !24
  %710 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %709, <1 x float> %708), !dbg !24
  %711 = tail call float @llvm.maxnum.f32(float %640, float -inf), !dbg !24, !frisk.local_reduction !5
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %711, float %646), !dbg !24, !frisk.local_reduction !5
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %641, float %647), !dbg !24, !frisk.local_reduction !5
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %642, float %648), !dbg !24, !frisk.local_reduction !5
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %643, float %649), !dbg !24, !frisk.local_reduction !5
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %652), !dbg !24, !frisk.local_reduction !5
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %653), !dbg !24, !frisk.local_reduction !5
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %654), !dbg !24, !frisk.local_reduction !5
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %655), !dbg !24, !frisk.local_reduction !5
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %658), !dbg !24, !frisk.local_reduction !5
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %659), !dbg !24, !frisk.local_reduction !5
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %660), !dbg !24, !frisk.local_reduction !5
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %661), !dbg !24, !frisk.local_reduction !5
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %664), !dbg !24, !frisk.local_reduction !5
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %665), !dbg !24, !frisk.local_reduction !5
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %666), !dbg !24, !frisk.local_reduction !5
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %667), !dbg !24, !frisk.local_reduction !5
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %670), !dbg !24, !frisk.local_reduction !5
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %671), !dbg !24, !frisk.local_reduction !5
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %672), !dbg !24, !frisk.local_reduction !5
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %673), !dbg !24, !frisk.local_reduction !5
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %676), !dbg !24, !frisk.local_reduction !5
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %677), !dbg !24, !frisk.local_reduction !5
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %678), !dbg !24, !frisk.local_reduction !5
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %679), !dbg !24, !frisk.local_reduction !5
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %682), !dbg !24, !frisk.local_reduction !5
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %683), !dbg !24, !frisk.local_reduction !5
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %684), !dbg !24, !frisk.local_reduction !5
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %685), !dbg !24, !frisk.local_reduction !5
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !24, !frisk.local_reduction !5
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !24, !frisk.local_reduction !5
  %712 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !24, !frisk.local_reduction !5
  %713 = bitcast float %712 to i32, !dbg !24
  %714 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %713), !dbg !24
  %715 = bitcast i32 %714 to float, !dbg !24
  %716 = tail call float @llvm.maxnum.f32(float %712, float %715), !dbg !24
  %717 = bitcast float %716 to i32, !dbg !24
  %718 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %717), !dbg !24
  %719 = bitcast i32 %718 to float, !dbg !24
  %720 = tail call float @llvm.maxnum.f32(float %716, float %719), !dbg !24
  %721 = extractvalue [2 x <1 x float>] %79, 1, !dbg !24
  %722 = insertelement <1 x float> poison, float %720, i64 0, !dbg !24
  %723 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %722, <1 x float> %721), !dbg !24
  %724 = shufflevector <1 x float> %710, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %725 = fsub <4 x float> %590, %724, !dbg !25
  %726 = fsub <4 x float> %596, %724, !dbg !25
  %727 = fsub <4 x float> %602, %724, !dbg !25
  %728 = fsub <4 x float> %608, %724, !dbg !25
  %729 = fsub <4 x float> %614, %724, !dbg !25
  %730 = fsub <4 x float> %620, %724, !dbg !25
  %731 = fsub <4 x float> %626, %724, !dbg !25
  %732 = fsub <4 x float> %632, %724, !dbg !25
  %733 = shufflevector <1 x float> %723, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %734 = fsub <4 x float> %639, %733, !dbg !25
  %735 = fsub <4 x float> %645, %733, !dbg !25
  %736 = fsub <4 x float> %651, %733, !dbg !25
  %737 = fsub <4 x float> %657, %733, !dbg !25
  %738 = fsub <4 x float> %663, %733, !dbg !25
  %739 = fsub <4 x float> %669, %733, !dbg !25
  %740 = fsub <4 x float> %675, %733, !dbg !25
  %741 = fsub <4 x float> %681, %733, !dbg !25
  br i1 %41, label %742, label %747, !dbg !13

742:                                              ; preds = %586
  %743 = fsub <1 x float> %721, %723, !dbg !26
  %744 = extractelement <1 x float> %743, i64 0, !dbg !26
  %745 = fsub <1 x float> %708, %710, !dbg !26
  %746 = extractelement <1 x float> %745, i64 0, !dbg !26
  %address.thread766 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !13, !range !11, !frisk.register_lifetimes !5
  %address.local767 = lshr i32 %address.thread766, 2, !dbg !18
  %address.local768 = and i32 %address.local767, 48, !dbg !18
  %address.local769 = and i32 %address.thread766, 15, !dbg !18
  %address.local770 = or disjoint i32 %address.local768, %address.local769, !dbg !18
  %address.local771 = zext nneg i32 %address.local770 to i64, !dbg !18
  %address.local772 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local771, !dbg !13
  store float %746, ptr addrspace(3) %address.local772, align 4, !dbg !13
  %address.local779 = getelementptr inbounds nuw i8, ptr addrspace(3) %address.local772, i64 256, !dbg !13
  store float %744, ptr addrspace(3) %address.local779, align 4, !dbg !13
  br label %747, !dbg !13

747:                                              ; preds = %742, %586
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.thread780 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !13, !range !11, !frisk.register_lifetimes !5
  %address.local781 = lshr i32 %address.thread780, 1, !dbg !13
  %address.local782 = and i32 %address.local781, 96, !dbg !13
  %address.local783 = and i32 %address.thread780, 15, !dbg !13
  %address.local784 = or disjoint i32 %address.local782, %address.local783, !dbg !13
  %address.local785 = zext nneg i32 %address.local784 to i64, !dbg !13
  %address.local786 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local785, !dbg !13
  %748 = load float, ptr addrspace(3) %address.local786, align 4, !dbg !13
  %lds.const.gep1745 = getelementptr i8, ptr addrspace(3) %address.local786, i64 64, !dbg !13
  %749 = load float, ptr addrspace(3) %lds.const.gep1745, align 4, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %750 = tail call float @llvm.amdgcn.exp2.f32(float %748), !dbg !13
  %751 = tail call float @llvm.amdgcn.exp2.f32(float %749), !dbg !13
  br i1 %41, label %752, label %753, !dbg !27

752:                                              ; preds = %747
  %address.thread789 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !27, !range !11, !frisk.register_lifetimes !5
  %address.local790 = lshr i32 %address.thread789, 1, !dbg !13
  %address.local791 = and i32 %address.local790, 96, !dbg !13
  %address.local792 = and i32 %address.thread789, 15, !dbg !13
  %address.local793 = or disjoint i32 %address.local791, %address.local792, !dbg !13
  %address.local794 = zext nneg i32 %address.local793 to i64, !dbg !13
  %address.local795 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !13
  store float %750, ptr addrspace(3) %address.local795, align 4, !dbg !27
  %address.local797 = getelementptr i8, ptr addrspace(3) %address.local795, i64 64, !dbg !13
  store float %751, ptr addrspace(3) %address.local797, align 4, !dbg !27
  br label %753, !dbg !27

753:                                              ; preds = %752, %747
  fence syncscope("workgroup") release, !dbg !27
  tail call void @llvm.amdgcn.s.barrier(), !dbg !27
  fence syncscope("workgroup") acquire, !dbg !27
  %address.thread798 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !27, !range !11, !frisk.register_lifetimes !5
  %address.local799 = lshr i32 %address.thread798, 2, !dbg !18
  %address.local800 = and i32 %address.local799, 48, !dbg !18
  %address.local801 = and i32 %address.thread798, 15, !dbg !18
  %address.local802 = or disjoint i32 %address.local800, %address.local801, !dbg !18
  %address.local803 = zext nneg i32 %address.local802 to i64, !dbg !18
  %address.local804 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local803, !dbg !27
  %754 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !27
  %lds.const.gep1746 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !27
  %755 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !27
  fence syncscope("workgroup") release, !dbg !27
  tail call void @llvm.amdgcn.s.barrier(), !dbg !27
  fence syncscope("workgroup") acquire, !dbg !27
  %756 = add nuw nsw i64 %86, %15, !dbg !28
  %757 = shl i64 %756, 19, !dbg !28
  %758 = and i64 %757, 16252928, !dbg !28
  %address.local812 = zext nneg i32 %address.thread798 to i64, !dbg !10
  %address.local813 = shl nuw nsw i64 %address.local812, 3, !dbg !9
  %address.local814 = and i64 %address.local813, 1536, !dbg !9
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local814, !dbg !9
  %address.local816 = ptrtoint ptr addrspace(3) %address.local815 to i64, !dbg !9
  %address.local817 = trunc i64 %address.local816 to i32, !dbg !9
  %759 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local817), !dbg !28
  %address.local818 = shl nuw nsw i64 %address.local812, 6, !dbg !28
  %address.local819 = and i64 %address.local818, 16128, !dbg !28
  %address.local820 = shl nuw nsw i64 %address.local812, 4, !dbg !28
  %address.local821 = and i64 %address.local820, 48, !dbg !28
  %.idx48 = add i64 %758, %92, !dbg !28
  %760 = shl i64 %.idx48, 1, !dbg !28
  %761 = or disjoint i64 %760, %address.local819, !dbg !28
  %762 = or disjoint i64 %761, %address.local821, !dbg !28
  %dma.buffer.offset1773 = trunc i64 %762 to i32, !dbg !28
  %dma.buffer.lds1775 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %759), !dbg !28
  %763 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1775, i32 %dma.buffer.offset1773, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %address.local824 = or disjoint i64 %address.local812, 256, !dbg !9
  %address.local825 = shl nuw nsw i64 %address.local824, 3, !dbg !9
  %address.local826 = and i64 %address.local825, 3584, !dbg !9
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local826, !dbg !9
  %address.local828 = ptrtoint ptr addrspace(3) %address.local827 to i64, !dbg !9
  %address.local829 = trunc i64 %address.local828 to i32, !dbg !9
  %764 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local829), !dbg !28
  %address.local830 = shl nuw nsw i64 %address.local824, 6, !dbg !28
  %address.local831 = and i64 %address.local830, 32512, !dbg !28
  %765 = or disjoint i64 %760, %address.local831, !dbg !28
  %766 = or disjoint i64 %765, %address.local821, !dbg !28
  %dma.buffer.offset1770 = trunc i64 %766 to i32, !dbg !28
  %dma.buffer.lds1772 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %764), !dbg !28
  %767 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1772, i32 %dma.buffer.offset1770, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %768 = extractvalue [2 x <1 x float>] %80, 0, !dbg !27
  %769 = extractelement <1 x float> %768, i64 0, !dbg !27
  %770 = fmul float %769, %754, !dbg !27
  %771 = extractvalue [2 x <1 x float>] %80, 1, !dbg !27
  %772 = extractelement <1 x float> %771, i64 0, !dbg !27
  %773 = fmul float %772, %755, !dbg !27
  %774 = extractelement <4 x float> %725, i64 0, !dbg !29
  %775 = tail call float @llvm.amdgcn.exp2.f32(float %774), !dbg !29
  %776 = insertelement <4 x float> poison, float %775, i64 0, !dbg !30
  %777 = extractelement <4 x float> %725, i64 1, !dbg !29
  %778 = tail call float @llvm.amdgcn.exp2.f32(float %777), !dbg !29
  %779 = insertelement <4 x float> %776, float %778, i64 1, !dbg !30
  %780 = extractelement <4 x float> %725, i64 2, !dbg !29
  %781 = tail call float @llvm.amdgcn.exp2.f32(float %780), !dbg !29
  %782 = insertelement <4 x float> %779, float %781, i64 2, !dbg !30
  %783 = extractelement <4 x float> %725, i64 3, !dbg !29
  %784 = tail call float @llvm.amdgcn.exp2.f32(float %783), !dbg !29
  %785 = insertelement <4 x float> %782, float %784, i64 3, !dbg !30
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %785) #8, !dbg !30, !frisk.register_lifetimes !5
  %786 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !30
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %786) #8, !dbg !29, !frisk.register_lifetimes !5
  %787 = extractelement <4 x float> %726, i64 0, !dbg !29
  %788 = tail call float @llvm.amdgcn.exp2.f32(float %787), !dbg !29
  %789 = insertelement <4 x float> poison, float %788, i64 0, !dbg !30
  %local.reduce120 = fadd float %775, %788, !dbg !31, !frisk.local_reduction !5
  %790 = extractelement <4 x float> %726, i64 1, !dbg !29
  %791 = tail call float @llvm.amdgcn.exp2.f32(float %790), !dbg !29
  %792 = insertelement <4 x float> %789, float %791, i64 1, !dbg !30
  %local.reduce121 = fadd float %778, %791, !dbg !31, !frisk.local_reduction !5
  %793 = extractelement <4 x float> %726, i64 2, !dbg !29
  %794 = tail call float @llvm.amdgcn.exp2.f32(float %793), !dbg !29
  %795 = insertelement <4 x float> %792, float %794, i64 2, !dbg !30
  %local.reduce122 = fadd float %781, %794, !dbg !31, !frisk.local_reduction !5
  %796 = extractelement <4 x float> %726, i64 3, !dbg !29
  %797 = tail call float @llvm.amdgcn.exp2.f32(float %796), !dbg !29
  %798 = insertelement <4 x float> %795, float %797, i64 3, !dbg !30
  %math.result228 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %798) #8, !dbg !30, !frisk.register_lifetimes !5
  %799 = fptrunc <4 x float> %math.result228 to <4 x half>, !dbg !30
  %math.packed229 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %799) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce123 = fadd float %784, %797, !dbg !31, !frisk.local_reduction !5
  %800 = extractelement <4 x float> %727, i64 0, !dbg !29
  %801 = tail call float @llvm.amdgcn.exp2.f32(float %800), !dbg !29
  %802 = insertelement <4 x float> poison, float %801, i64 0, !dbg !30
  %local.reduce124 = fadd float %801, %local.reduce120, !dbg !31, !frisk.local_reduction !5
  %803 = extractelement <4 x float> %727, i64 1, !dbg !29
  %804 = tail call float @llvm.amdgcn.exp2.f32(float %803), !dbg !29
  %805 = insertelement <4 x float> %802, float %804, i64 1, !dbg !30
  %local.reduce125 = fadd float %804, %local.reduce121, !dbg !31, !frisk.local_reduction !5
  %806 = extractelement <4 x float> %727, i64 2, !dbg !29
  %807 = tail call float @llvm.amdgcn.exp2.f32(float %806), !dbg !29
  %808 = insertelement <4 x float> %805, float %807, i64 2, !dbg !30
  %local.reduce126 = fadd float %807, %local.reduce122, !dbg !31, !frisk.local_reduction !5
  %809 = extractelement <4 x float> %727, i64 3, !dbg !29
  %810 = tail call float @llvm.amdgcn.exp2.f32(float %809), !dbg !29
  %811 = insertelement <4 x float> %808, float %810, i64 3, !dbg !30
  %math.result230 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %811) #8, !dbg !30, !frisk.register_lifetimes !5
  %812 = fptrunc <4 x float> %math.result230 to <4 x half>, !dbg !30
  %math.packed231 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %812) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce127 = fadd float %810, %local.reduce123, !dbg !31, !frisk.local_reduction !5
  %813 = extractelement <4 x float> %728, i64 0, !dbg !29
  %814 = tail call float @llvm.amdgcn.exp2.f32(float %813), !dbg !29
  %815 = insertelement <4 x float> poison, float %814, i64 0, !dbg !30
  %local.reduce128 = fadd float %814, %local.reduce124, !dbg !31, !frisk.local_reduction !5
  %816 = extractelement <4 x float> %728, i64 1, !dbg !29
  %817 = tail call float @llvm.amdgcn.exp2.f32(float %816), !dbg !29
  %818 = insertelement <4 x float> %815, float %817, i64 1, !dbg !30
  %local.reduce129 = fadd float %817, %local.reduce125, !dbg !31, !frisk.local_reduction !5
  %819 = extractelement <4 x float> %728, i64 2, !dbg !29
  %820 = tail call float @llvm.amdgcn.exp2.f32(float %819), !dbg !29
  %821 = insertelement <4 x float> %818, float %820, i64 2, !dbg !30
  %local.reduce130 = fadd float %820, %local.reduce126, !dbg !31, !frisk.local_reduction !5
  %822 = extractelement <4 x float> %728, i64 3, !dbg !29
  %823 = tail call float @llvm.amdgcn.exp2.f32(float %822), !dbg !29
  %824 = insertelement <4 x float> %821, float %823, i64 3, !dbg !30
  %math.result232 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %824) #8, !dbg !30, !frisk.register_lifetimes !5
  %825 = fptrunc <4 x float> %math.result232 to <4 x half>, !dbg !30
  %math.packed233 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %825) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce131 = fadd float %823, %local.reduce127, !dbg !31, !frisk.local_reduction !5
  %826 = extractelement <4 x float> %729, i64 0, !dbg !29
  %827 = tail call float @llvm.amdgcn.exp2.f32(float %826), !dbg !29
  %828 = insertelement <4 x float> poison, float %827, i64 0, !dbg !30
  %local.reduce132 = fadd float %827, %local.reduce128, !dbg !31, !frisk.local_reduction !5
  %829 = extractelement <4 x float> %729, i64 1, !dbg !29
  %830 = tail call float @llvm.amdgcn.exp2.f32(float %829), !dbg !29
  %831 = insertelement <4 x float> %828, float %830, i64 1, !dbg !30
  %local.reduce133 = fadd float %830, %local.reduce129, !dbg !31, !frisk.local_reduction !5
  %832 = extractelement <4 x float> %729, i64 2, !dbg !29
  %833 = tail call float @llvm.amdgcn.exp2.f32(float %832), !dbg !29
  %834 = insertelement <4 x float> %831, float %833, i64 2, !dbg !30
  %local.reduce134 = fadd float %833, %local.reduce130, !dbg !31, !frisk.local_reduction !5
  %835 = extractelement <4 x float> %729, i64 3, !dbg !29
  %836 = tail call float @llvm.amdgcn.exp2.f32(float %835), !dbg !29
  %837 = insertelement <4 x float> %834, float %836, i64 3, !dbg !30
  %math.result234 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %837) #8, !dbg !30, !frisk.register_lifetimes !5
  %838 = fptrunc <4 x float> %math.result234 to <4 x half>, !dbg !30
  %math.packed235 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %838) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce135 = fadd float %836, %local.reduce131, !dbg !31, !frisk.local_reduction !5
  %839 = extractelement <4 x float> %730, i64 0, !dbg !29
  %840 = tail call float @llvm.amdgcn.exp2.f32(float %839), !dbg !29
  %841 = insertelement <4 x float> poison, float %840, i64 0, !dbg !30
  %local.reduce136 = fadd float %840, %local.reduce132, !dbg !31, !frisk.local_reduction !5
  %842 = extractelement <4 x float> %730, i64 1, !dbg !29
  %843 = tail call float @llvm.amdgcn.exp2.f32(float %842), !dbg !29
  %844 = insertelement <4 x float> %841, float %843, i64 1, !dbg !30
  %local.reduce137 = fadd float %843, %local.reduce133, !dbg !31, !frisk.local_reduction !5
  %845 = extractelement <4 x float> %730, i64 2, !dbg !29
  %846 = tail call float @llvm.amdgcn.exp2.f32(float %845), !dbg !29
  %847 = insertelement <4 x float> %844, float %846, i64 2, !dbg !30
  %local.reduce138 = fadd float %846, %local.reduce134, !dbg !31, !frisk.local_reduction !5
  %848 = extractelement <4 x float> %730, i64 3, !dbg !29
  %849 = tail call float @llvm.amdgcn.exp2.f32(float %848), !dbg !29
  %850 = insertelement <4 x float> %847, float %849, i64 3, !dbg !30
  %math.result236 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %850) #8, !dbg !30, !frisk.register_lifetimes !5
  %851 = fptrunc <4 x float> %math.result236 to <4 x half>, !dbg !30
  %math.packed237 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %851) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce139 = fadd float %849, %local.reduce135, !dbg !31, !frisk.local_reduction !5
  %852 = extractelement <4 x float> %731, i64 0, !dbg !29
  %853 = tail call float @llvm.amdgcn.exp2.f32(float %852), !dbg !29
  %854 = insertelement <4 x float> poison, float %853, i64 0, !dbg !30
  %local.reduce140 = fadd float %853, %local.reduce136, !dbg !31, !frisk.local_reduction !5
  %855 = extractelement <4 x float> %731, i64 1, !dbg !29
  %856 = tail call float @llvm.amdgcn.exp2.f32(float %855), !dbg !29
  %857 = insertelement <4 x float> %854, float %856, i64 1, !dbg !30
  %local.reduce141 = fadd float %856, %local.reduce137, !dbg !31, !frisk.local_reduction !5
  %858 = extractelement <4 x float> %731, i64 2, !dbg !29
  %859 = tail call float @llvm.amdgcn.exp2.f32(float %858), !dbg !29
  %860 = insertelement <4 x float> %857, float %859, i64 2, !dbg !30
  %local.reduce142 = fadd float %859, %local.reduce138, !dbg !31, !frisk.local_reduction !5
  %861 = extractelement <4 x float> %731, i64 3, !dbg !29
  %862 = tail call float @llvm.amdgcn.exp2.f32(float %861), !dbg !29
  %863 = insertelement <4 x float> %860, float %862, i64 3, !dbg !30
  %math.result238 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %863) #8, !dbg !30, !frisk.register_lifetimes !5
  %864 = fptrunc <4 x float> %math.result238 to <4 x half>, !dbg !30
  %math.packed239 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %864) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce143 = fadd float %862, %local.reduce139, !dbg !31, !frisk.local_reduction !5
  %865 = extractelement <4 x float> %732, i64 0, !dbg !29
  %866 = tail call float @llvm.amdgcn.exp2.f32(float %865), !dbg !29
  %867 = insertelement <4 x float> poison, float %866, i64 0, !dbg !30
  %local.reduce144 = fadd float %866, %local.reduce140, !dbg !31, !frisk.local_reduction !5
  %868 = extractelement <4 x float> %732, i64 1, !dbg !29
  %869 = tail call float @llvm.amdgcn.exp2.f32(float %868), !dbg !29
  %870 = insertelement <4 x float> %867, float %869, i64 1, !dbg !30
  %local.reduce145 = fadd float %869, %local.reduce141, !dbg !31, !frisk.local_reduction !5
  %local.reduce148 = fadd float %local.reduce144, %local.reduce145, !dbg !31, !frisk.local_reduction !5
  %871 = extractelement <4 x float> %732, i64 2, !dbg !29
  %872 = tail call float @llvm.amdgcn.exp2.f32(float %871), !dbg !29
  %873 = insertelement <4 x float> %870, float %872, i64 2, !dbg !30
  %local.reduce146 = fadd float %872, %local.reduce142, !dbg !31, !frisk.local_reduction !5
  %874 = extractelement <4 x float> %732, i64 3, !dbg !29
  %875 = tail call float @llvm.amdgcn.exp2.f32(float %874), !dbg !29
  %876 = insertelement <4 x float> %873, float %875, i64 3, !dbg !30
  %math.result240 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %876) #8, !dbg !30, !frisk.register_lifetimes !5
  %877 = fptrunc <4 x float> %math.result240 to <4 x half>, !dbg !30
  %math.packed241 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %877) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce147 = fadd float %875, %local.reduce143, !dbg !31, !frisk.local_reduction !5
  %local.reduce149 = fadd float %local.reduce146, %local.reduce147, !dbg !31, !frisk.local_reduction !5
  %878 = fadd float %local.reduce148, %local.reduce149, !dbg !31, !frisk.local_reduction !5
  %879 = extractelement <4 x float> %734, i64 0, !dbg !29
  %880 = tail call float @llvm.amdgcn.exp2.f32(float %879), !dbg !29
  %881 = insertelement <4 x float> poison, float %880, i64 0, !dbg !30
  %882 = extractelement <4 x float> %734, i64 1, !dbg !29
  %883 = tail call float @llvm.amdgcn.exp2.f32(float %882), !dbg !29
  %884 = insertelement <4 x float> %881, float %883, i64 1, !dbg !30
  %885 = extractelement <4 x float> %734, i64 2, !dbg !29
  %886 = tail call float @llvm.amdgcn.exp2.f32(float %885), !dbg !29
  %887 = insertelement <4 x float> %884, float %886, i64 2, !dbg !30
  %888 = extractelement <4 x float> %734, i64 3, !dbg !29
  %889 = tail call float @llvm.amdgcn.exp2.f32(float %888), !dbg !29
  %890 = insertelement <4 x float> %887, float %889, i64 3, !dbg !30
  %math.result242 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %890) #8, !dbg !30, !frisk.register_lifetimes !5
  %891 = fptrunc <4 x float> %math.result242 to <4 x half>, !dbg !30
  %math.packed243 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %891) #8, !dbg !29, !frisk.register_lifetimes !5
  %892 = extractelement <4 x float> %735, i64 0, !dbg !29
  %893 = tail call float @llvm.amdgcn.exp2.f32(float %892), !dbg !29
  %894 = insertelement <4 x float> poison, float %893, i64 0, !dbg !30
  %local.reduce = fadd float %880, %893, !dbg !31, !frisk.local_reduction !5
  %895 = extractelement <4 x float> %735, i64 1, !dbg !29
  %896 = tail call float @llvm.amdgcn.exp2.f32(float %895), !dbg !29
  %897 = insertelement <4 x float> %894, float %896, i64 1, !dbg !30
  %local.reduce90 = fadd float %883, %896, !dbg !31, !frisk.local_reduction !5
  %898 = extractelement <4 x float> %735, i64 2, !dbg !29
  %899 = tail call float @llvm.amdgcn.exp2.f32(float %898), !dbg !29
  %900 = insertelement <4 x float> %897, float %899, i64 2, !dbg !30
  %local.reduce91 = fadd float %886, %899, !dbg !31, !frisk.local_reduction !5
  %901 = extractelement <4 x float> %735, i64 3, !dbg !29
  %902 = tail call float @llvm.amdgcn.exp2.f32(float %901), !dbg !29
  %903 = insertelement <4 x float> %900, float %902, i64 3, !dbg !30
  %math.result244 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %903) #8, !dbg !30, !frisk.register_lifetimes !5
  %904 = fptrunc <4 x float> %math.result244 to <4 x half>, !dbg !30
  %math.packed245 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %904) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce92 = fadd float %889, %902, !dbg !31, !frisk.local_reduction !5
  %905 = extractelement <4 x float> %736, i64 0, !dbg !29
  %906 = tail call float @llvm.amdgcn.exp2.f32(float %905), !dbg !29
  %907 = insertelement <4 x float> poison, float %906, i64 0, !dbg !30
  %local.reduce93 = fadd float %906, %local.reduce, !dbg !31, !frisk.local_reduction !5
  %908 = extractelement <4 x float> %736, i64 1, !dbg !29
  %909 = tail call float @llvm.amdgcn.exp2.f32(float %908), !dbg !29
  %910 = insertelement <4 x float> %907, float %909, i64 1, !dbg !30
  %local.reduce94 = fadd float %909, %local.reduce90, !dbg !31, !frisk.local_reduction !5
  %911 = extractelement <4 x float> %736, i64 2, !dbg !29
  %912 = tail call float @llvm.amdgcn.exp2.f32(float %911), !dbg !29
  %913 = insertelement <4 x float> %910, float %912, i64 2, !dbg !30
  %local.reduce95 = fadd float %912, %local.reduce91, !dbg !31, !frisk.local_reduction !5
  %914 = extractelement <4 x float> %736, i64 3, !dbg !29
  %915 = tail call float @llvm.amdgcn.exp2.f32(float %914), !dbg !29
  %916 = insertelement <4 x float> %913, float %915, i64 3, !dbg !30
  %math.result246 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %916) #8, !dbg !30, !frisk.register_lifetimes !5
  %917 = fptrunc <4 x float> %math.result246 to <4 x half>, !dbg !30
  %math.packed247 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %917) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce96 = fadd float %915, %local.reduce92, !dbg !31, !frisk.local_reduction !5
  %918 = extractelement <4 x float> %737, i64 0, !dbg !29
  %919 = tail call float @llvm.amdgcn.exp2.f32(float %918), !dbg !29
  %920 = insertelement <4 x float> poison, float %919, i64 0, !dbg !30
  %local.reduce97 = fadd float %919, %local.reduce93, !dbg !31, !frisk.local_reduction !5
  %921 = extractelement <4 x float> %737, i64 1, !dbg !29
  %922 = tail call float @llvm.amdgcn.exp2.f32(float %921), !dbg !29
  %923 = insertelement <4 x float> %920, float %922, i64 1, !dbg !30
  %local.reduce98 = fadd float %922, %local.reduce94, !dbg !31, !frisk.local_reduction !5
  %924 = extractelement <4 x float> %737, i64 2, !dbg !29
  %925 = tail call float @llvm.amdgcn.exp2.f32(float %924), !dbg !29
  %926 = insertelement <4 x float> %923, float %925, i64 2, !dbg !30
  %local.reduce99 = fadd float %925, %local.reduce95, !dbg !31, !frisk.local_reduction !5
  %927 = extractelement <4 x float> %737, i64 3, !dbg !29
  %928 = tail call float @llvm.amdgcn.exp2.f32(float %927), !dbg !29
  %929 = insertelement <4 x float> %926, float %928, i64 3, !dbg !30
  %math.result248 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %929) #8, !dbg !30, !frisk.register_lifetimes !5
  %930 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !30
  %math.packed249 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %930) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce100 = fadd float %928, %local.reduce96, !dbg !31, !frisk.local_reduction !5
  %931 = extractelement <4 x float> %738, i64 0, !dbg !29
  %932 = tail call float @llvm.amdgcn.exp2.f32(float %931), !dbg !29
  %933 = insertelement <4 x float> poison, float %932, i64 0, !dbg !30
  %local.reduce101 = fadd float %932, %local.reduce97, !dbg !31, !frisk.local_reduction !5
  %934 = extractelement <4 x float> %738, i64 1, !dbg !29
  %935 = tail call float @llvm.amdgcn.exp2.f32(float %934), !dbg !29
  %936 = insertelement <4 x float> %933, float %935, i64 1, !dbg !30
  %local.reduce102 = fadd float %935, %local.reduce98, !dbg !31, !frisk.local_reduction !5
  %937 = extractelement <4 x float> %738, i64 2, !dbg !29
  %938 = tail call float @llvm.amdgcn.exp2.f32(float %937), !dbg !29
  %939 = insertelement <4 x float> %936, float %938, i64 2, !dbg !30
  %local.reduce103 = fadd float %938, %local.reduce99, !dbg !31, !frisk.local_reduction !5
  %940 = extractelement <4 x float> %738, i64 3, !dbg !29
  %941 = tail call float @llvm.amdgcn.exp2.f32(float %940), !dbg !29
  %942 = insertelement <4 x float> %939, float %941, i64 3, !dbg !30
  %math.result250 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %942) #8, !dbg !30, !frisk.register_lifetimes !5
  %943 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !30
  %math.packed251 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %943) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce104 = fadd float %941, %local.reduce100, !dbg !31, !frisk.local_reduction !5
  %944 = extractelement <4 x float> %739, i64 0, !dbg !29
  %945 = tail call float @llvm.amdgcn.exp2.f32(float %944), !dbg !29
  %946 = insertelement <4 x float> poison, float %945, i64 0, !dbg !30
  %local.reduce105 = fadd float %945, %local.reduce101, !dbg !31, !frisk.local_reduction !5
  %947 = extractelement <4 x float> %739, i64 1, !dbg !29
  %948 = tail call float @llvm.amdgcn.exp2.f32(float %947), !dbg !29
  %949 = insertelement <4 x float> %946, float %948, i64 1, !dbg !30
  %local.reduce106 = fadd float %948, %local.reduce102, !dbg !31, !frisk.local_reduction !5
  %950 = extractelement <4 x float> %739, i64 2, !dbg !29
  %951 = tail call float @llvm.amdgcn.exp2.f32(float %950), !dbg !29
  %952 = insertelement <4 x float> %949, float %951, i64 2, !dbg !30
  %local.reduce107 = fadd float %951, %local.reduce103, !dbg !31, !frisk.local_reduction !5
  %953 = extractelement <4 x float> %739, i64 3, !dbg !29
  %954 = tail call float @llvm.amdgcn.exp2.f32(float %953), !dbg !29
  %955 = insertelement <4 x float> %952, float %954, i64 3, !dbg !30
  %math.result252 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %955) #8, !dbg !30, !frisk.register_lifetimes !5
  %956 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !30
  %math.packed253 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %956) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce108 = fadd float %954, %local.reduce104, !dbg !31, !frisk.local_reduction !5
  %957 = extractelement <4 x float> %740, i64 0, !dbg !29
  %958 = tail call float @llvm.amdgcn.exp2.f32(float %957), !dbg !29
  %959 = insertelement <4 x float> poison, float %958, i64 0, !dbg !30
  %local.reduce109 = fadd float %958, %local.reduce105, !dbg !31, !frisk.local_reduction !5
  %960 = extractelement <4 x float> %740, i64 1, !dbg !29
  %961 = tail call float @llvm.amdgcn.exp2.f32(float %960), !dbg !29
  %962 = insertelement <4 x float> %959, float %961, i64 1, !dbg !30
  %local.reduce110 = fadd float %961, %local.reduce106, !dbg !31, !frisk.local_reduction !5
  %963 = extractelement <4 x float> %740, i64 2, !dbg !29
  %964 = tail call float @llvm.amdgcn.exp2.f32(float %963), !dbg !29
  %965 = insertelement <4 x float> %962, float %964, i64 2, !dbg !30
  %local.reduce111 = fadd float %964, %local.reduce107, !dbg !31, !frisk.local_reduction !5
  %966 = extractelement <4 x float> %740, i64 3, !dbg !29
  %967 = tail call float @llvm.amdgcn.exp2.f32(float %966), !dbg !29
  %968 = insertelement <4 x float> %965, float %967, i64 3, !dbg !30
  %math.result254 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %968) #8, !dbg !30, !frisk.register_lifetimes !5
  %969 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !30
  %math.packed255 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %969) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce112 = fadd float %967, %local.reduce108, !dbg !31, !frisk.local_reduction !5
  %970 = extractelement <4 x float> %741, i64 0, !dbg !29
  %971 = tail call float @llvm.amdgcn.exp2.f32(float %970), !dbg !29
  %972 = insertelement <4 x float> poison, float %971, i64 0, !dbg !30
  %local.reduce113 = fadd float %971, %local.reduce109, !dbg !31, !frisk.local_reduction !5
  %973 = extractelement <4 x float> %741, i64 1, !dbg !29
  %974 = tail call float @llvm.amdgcn.exp2.f32(float %973), !dbg !29
  %975 = insertelement <4 x float> %972, float %974, i64 1, !dbg !30
  %local.reduce114 = fadd float %974, %local.reduce110, !dbg !31, !frisk.local_reduction !5
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !31, !frisk.local_reduction !5
  %976 = extractelement <4 x float> %741, i64 2, !dbg !29
  %977 = tail call float @llvm.amdgcn.exp2.f32(float %976), !dbg !29
  %978 = insertelement <4 x float> %975, float %977, i64 2, !dbg !30
  %local.reduce115 = fadd float %977, %local.reduce111, !dbg !31, !frisk.local_reduction !5
  %979 = extractelement <4 x float> %741, i64 3, !dbg !29
  %980 = tail call float @llvm.amdgcn.exp2.f32(float %979), !dbg !29
  %981 = insertelement <4 x float> %978, float %980, i64 3, !dbg !30
  %math.result256 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %981) #8, !dbg !30, !frisk.register_lifetimes !5
  %982 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !30
  %math.packed257 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %982) #8, !dbg !31, !frisk.register_lifetimes !5
  %local.reduce116 = fadd float %980, %local.reduce112, !dbg !31, !frisk.local_reduction !5
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !31, !frisk.local_reduction !5
  %983 = fadd float %local.reduce117, %local.reduce118, !dbg !31, !frisk.local_reduction !5
  %984 = bitcast float %878 to i32, !dbg !31
  %985 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %984), !dbg !31
  %986 = bitcast i32 %985 to float, !dbg !31
  %987 = fadd float %878, %986, !dbg !31
  %988 = bitcast float %987 to i32, !dbg !31
  %989 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %988), !dbg !31
  %990 = bitcast i32 %989 to float, !dbg !31
  %991 = fadd float %987, %990, !dbg !31
  %992 = fadd float %770, %991, !dbg !31
  %993 = bitcast float %983 to i32, !dbg !31
  %994 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %993), !dbg !31
  %995 = bitcast i32 %994 to float, !dbg !31
  %996 = fadd float %983, %995, !dbg !31
  %997 = bitcast float %996 to i32, !dbg !31
  %998 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %997), !dbg !31
  %999 = bitcast i32 %998 to float, !dbg !31
  %1000 = fadd float %996, %999, !dbg !31
  %1001 = fadd float %773, %1000, !dbg !31
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local836 = shl nuw nsw i32 %address.thread798, 3, !dbg !14
  %address.local837 = and i32 %address.local836, 504, !dbg !14
  %address.local838 = zext nneg i32 %address.local837 to i64, !dbg !14
  %address.local839 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !14
  %1002 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local839, i32 0) #10, !dbg !14
  %1003 = bitcast <4 x i32> %1002 to <8 x half>, !dbg !14
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local839, i64 1024, !dbg !14
  %1004 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #10, !dbg !14
  %1005 = bitcast <4 x i32> %1004 to <8 x half>, !dbg !14
  %address.local843 = getelementptr i8, ptr addrspace(3) %address.local839, i64 2048, !dbg !14
  %1006 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local843, i32 0) #10, !dbg !14
  %1007 = bitcast <4 x i32> %1006 to <8 x half>, !dbg !14
  %address.local845 = getelementptr i8, ptr addrspace(3) %address.local839, i64 3072, !dbg !14
  %1008 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local845, i32 0) #10, !dbg !14
  %1009 = bitcast <4 x i32> %1008 to <8 x half>, !dbg !14
  %address.local847 = getelementptr i8, ptr addrspace(3) %address.local839, i64 4096, !dbg !14
  %1010 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local847, i32 0) #10, !dbg !14
  %1011 = bitcast <4 x i32> %1010 to <8 x half>, !dbg !14
  %address.local849 = getelementptr i8, ptr addrspace(3) %address.local839, i64 5120, !dbg !14
  %1012 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local849, i32 0) #10, !dbg !14
  %1013 = bitcast <4 x i32> %1012 to <8 x half>, !dbg !14
  %address.local851 = getelementptr i8, ptr addrspace(3) %address.local839, i64 6144, !dbg !14
  %1014 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #10, !dbg !14
  %1015 = bitcast <4 x i32> %1014 to <8 x half>, !dbg !14
  %address.local853 = getelementptr i8, ptr addrspace(3) %address.local839, i64 7168, !dbg !14
  %1016 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local853, i32 0) #10, !dbg !14
  %1017 = bitcast <4 x i32> %1016 to <8 x half>, !dbg !14
  %1018 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1019 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1018, <4 x float> zeroinitializer) #8, !dbg !14
  %1020 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1021 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1020, <4 x float> %1019) #8, !dbg !14
  %1022 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1023 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1022, <4 x float> %1021) #8, !dbg !14
  %1024 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1025 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1024, <4 x float> %1023) #8, !dbg !14
  %1026 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1027 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1026, <4 x float> %1025) #8, !dbg !14
  %1028 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1029 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1028, <4 x float> %1027) #8, !dbg !14
  %1030 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1031 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1030, <4 x float> %1029) #8, !dbg !14
  %1032 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1033 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1032, <4 x float> %1031) #8, !dbg !14
  %1034 = extractelement <4 x float> %1033, i64 0, !dbg !14
  %1035 = extractelement <4 x float> %1033, i64 1, !dbg !14
  %1036 = extractelement <4 x float> %1033, i64 2, !dbg !14
  %1037 = extractelement <4 x float> %1033, i64 3, !dbg !14
  %1038 = select i1 %.not, float %1035, float %1034, !dbg !14
  %1039 = bitcast float %1038 to i32, !dbg !14
  %1040 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1039), !dbg !14
  %1041 = bitcast i32 %1040 to float, !dbg !14
  %1042 = select i1 %.not, float %1034, float %1041, !dbg !14
  %1043 = select i1 %.not, float %1041, float %1035, !dbg !14
  %1044 = select i1 %.not, float %1037, float %1036, !dbg !14
  %1045 = bitcast float %1044 to i32, !dbg !14
  %1046 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1045), !dbg !14
  %1047 = bitcast i32 %1046 to float, !dbg !14
  %1048 = select i1 %.not, float %1036, float %1047, !dbg !14
  %1049 = select i1 %.not, float %1047, float %1037, !dbg !14
  %1050 = select i1 %.not1, float %1048, float %1042, !dbg !14
  %1051 = bitcast float %1050 to i32, !dbg !14
  %1052 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1051), !dbg !14
  %1053 = bitcast i32 %1052 to float, !dbg !14
  %1054 = select i1 %.not1, float %1049, float %1043, !dbg !14
  %1055 = bitcast float %1054 to i32, !dbg !14
  %1056 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1055), !dbg !14
  %1057 = bitcast i32 %1056 to float, !dbg !14
  %1058 = insertelement <4 x float> poison, float %1042, i64 0, !dbg !14
  %1059 = insertelement <4 x float> %1058, float %1043, i64 1, !dbg !14
  %1060 = insertelement <4 x float> %1059, float %1053, i64 2, !dbg !14
  %1061 = insertelement <4 x float> %1060, float %1057, i64 3, !dbg !14
  %1062 = insertelement <4 x float> poison, float %1053, i64 0, !dbg !14
  %1063 = insertelement <4 x float> %1062, float %1057, i64 1, !dbg !14
  %1064 = insertelement <4 x float> %1063, float %1048, i64 2, !dbg !14
  %1065 = insertelement <4 x float> %1064, float %1049, i64 3, !dbg !14
  %1066 = select <4 x i1> %70, <4 x float> %1061, <4 x float> %1065, !dbg !14
  %1067 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1068 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1067, <4 x float> zeroinitializer) #8, !dbg !14
  %1069 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1070 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1069, <4 x float> %1068) #8, !dbg !14
  %1071 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1072 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1071, <4 x float> %1070) #8, !dbg !14
  %1073 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1074 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1073, <4 x float> %1072) #8, !dbg !14
  %1075 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1075, <4 x float> %1074) #8, !dbg !14
  %1077 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1077, <4 x float> %1076) #8, !dbg !14
  %1079 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1079, <4 x float> %1078) #8, !dbg !14
  %1081 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1081, <4 x float> %1080) #8, !dbg !14
  %1083 = extractelement <4 x float> %1082, i64 0, !dbg !14
  %1084 = extractelement <4 x float> %1082, i64 1, !dbg !14
  %1085 = extractelement <4 x float> %1082, i64 2, !dbg !14
  %1086 = extractelement <4 x float> %1082, i64 3, !dbg !14
  %1087 = select i1 %.not, float %1084, float %1083, !dbg !14
  %1088 = bitcast float %1087 to i32, !dbg !14
  %1089 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1088), !dbg !14
  %1090 = bitcast i32 %1089 to float, !dbg !14
  %1091 = select i1 %.not, float %1083, float %1090, !dbg !14
  %1092 = select i1 %.not, float %1090, float %1084, !dbg !14
  %1093 = select i1 %.not, float %1086, float %1085, !dbg !14
  %1094 = bitcast float %1093 to i32, !dbg !14
  %1095 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1094), !dbg !14
  %1096 = bitcast i32 %1095 to float, !dbg !14
  %1097 = select i1 %.not, float %1085, float %1096, !dbg !14
  %1098 = select i1 %.not, float %1096, float %1086, !dbg !14
  %1099 = select i1 %.not1, float %1097, float %1091, !dbg !14
  %1100 = bitcast float %1099 to i32, !dbg !14
  %1101 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1100), !dbg !14
  %1102 = bitcast i32 %1101 to float, !dbg !14
  %1103 = select i1 %.not1, float %1098, float %1092, !dbg !14
  %1104 = bitcast float %1103 to i32, !dbg !14
  %1105 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1104), !dbg !14
  %1106 = bitcast i32 %1105 to float, !dbg !14
  %1107 = insertelement <4 x float> poison, float %1091, i64 0, !dbg !14
  %1108 = insertelement <4 x float> %1107, float %1092, i64 1, !dbg !14
  %1109 = insertelement <4 x float> %1108, float %1102, i64 2, !dbg !14
  %1110 = insertelement <4 x float> %1109, float %1106, i64 3, !dbg !14
  %1111 = insertelement <4 x float> poison, float %1102, i64 0, !dbg !14
  %1112 = insertelement <4 x float> %1111, float %1106, i64 1, !dbg !14
  %1113 = insertelement <4 x float> %1112, float %1097, i64 2, !dbg !14
  %1114 = insertelement <4 x float> %1113, float %1098, i64 3, !dbg !14
  %1115 = select <4 x i1> %70, <4 x float> %1110, <4 x float> %1114, !dbg !14
  %1116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1018, <4 x float> zeroinitializer) #8, !dbg !14
  %1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1020, <4 x float> %1116) #8, !dbg !14
  %1118 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1022, <4 x float> %1117) #8, !dbg !14
  %1119 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1024, <4 x float> %1118) #8, !dbg !14
  %1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1026, <4 x float> %1119) #8, !dbg !14
  %1121 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1028, <4 x float> %1120) #8, !dbg !14
  %1122 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1030, <4 x float> %1121) #8, !dbg !14
  %1123 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1032, <4 x float> %1122) #8, !dbg !14
  %1124 = extractelement <4 x float> %1123, i64 0, !dbg !14
  %1125 = extractelement <4 x float> %1123, i64 1, !dbg !14
  %1126 = extractelement <4 x float> %1123, i64 2, !dbg !14
  %1127 = extractelement <4 x float> %1123, i64 3, !dbg !14
  %1128 = select i1 %.not, float %1125, float %1124, !dbg !14
  %1129 = bitcast float %1128 to i32, !dbg !14
  %1130 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1129), !dbg !14
  %1131 = bitcast i32 %1130 to float, !dbg !14
  %1132 = select i1 %.not, float %1124, float %1131, !dbg !14
  %1133 = select i1 %.not, float %1131, float %1125, !dbg !14
  %1134 = select i1 %.not, float %1127, float %1126, !dbg !14
  %1135 = bitcast float %1134 to i32, !dbg !14
  %1136 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1135), !dbg !14
  %1137 = bitcast i32 %1136 to float, !dbg !14
  %1138 = select i1 %.not, float %1126, float %1137, !dbg !14
  %1139 = select i1 %.not, float %1137, float %1127, !dbg !14
  %1140 = select i1 %.not1, float %1138, float %1132, !dbg !14
  %1141 = bitcast float %1140 to i32, !dbg !14
  %1142 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1141), !dbg !14
  %1143 = bitcast i32 %1142 to float, !dbg !14
  %1144 = select i1 %.not1, float %1139, float %1133, !dbg !14
  %1145 = bitcast float %1144 to i32, !dbg !14
  %1146 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1145), !dbg !14
  %1147 = bitcast i32 %1146 to float, !dbg !14
  %1148 = insertelement <4 x float> poison, float %1132, i64 0, !dbg !14
  %1149 = insertelement <4 x float> %1148, float %1133, i64 1, !dbg !14
  %1150 = insertelement <4 x float> %1149, float %1143, i64 2, !dbg !14
  %1151 = insertelement <4 x float> %1150, float %1147, i64 3, !dbg !14
  %1152 = insertelement <4 x float> poison, float %1143, i64 0, !dbg !14
  %1153 = insertelement <4 x float> %1152, float %1147, i64 1, !dbg !14
  %1154 = insertelement <4 x float> %1153, float %1138, i64 2, !dbg !14
  %1155 = insertelement <4 x float> %1154, float %1139, i64 3, !dbg !14
  %1156 = select <4 x i1> %70, <4 x float> %1151, <4 x float> %1155, !dbg !14
  %1157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1067, <4 x float> zeroinitializer) #8, !dbg !14
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1069, <4 x float> %1157) #8, !dbg !14
  %1159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1071, <4 x float> %1158) #8, !dbg !14
  %1160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1073, <4 x float> %1159) #8, !dbg !14
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1075, <4 x float> %1160) #8, !dbg !14
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1077, <4 x float> %1161) #8, !dbg !14
  %1163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1079, <4 x float> %1162) #8, !dbg !14
  %1164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1081, <4 x float> %1163) #8, !dbg !14
  %1165 = extractelement <4 x float> %1164, i64 0, !dbg !14
  %1166 = extractelement <4 x float> %1164, i64 1, !dbg !14
  %1167 = extractelement <4 x float> %1164, i64 2, !dbg !14
  %1168 = extractelement <4 x float> %1164, i64 3, !dbg !14
  %1169 = select i1 %.not, float %1166, float %1165, !dbg !14
  %1170 = bitcast float %1169 to i32, !dbg !14
  %1171 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1170), !dbg !14
  %1172 = bitcast i32 %1171 to float, !dbg !14
  %1173 = select i1 %.not, float %1165, float %1172, !dbg !14
  %1174 = select i1 %.not, float %1172, float %1166, !dbg !14
  %1175 = select i1 %.not, float %1168, float %1167, !dbg !14
  %1176 = bitcast float %1175 to i32, !dbg !14
  %1177 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1176), !dbg !14
  %1178 = bitcast i32 %1177 to float, !dbg !14
  %1179 = select i1 %.not, float %1167, float %1178, !dbg !14
  %1180 = select i1 %.not, float %1178, float %1168, !dbg !14
  %1181 = select i1 %.not1, float %1179, float %1173, !dbg !14
  %1182 = bitcast float %1181 to i32, !dbg !14
  %1183 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1182), !dbg !14
  %1184 = bitcast i32 %1183 to float, !dbg !14
  %1185 = select i1 %.not1, float %1180, float %1174, !dbg !14
  %1186 = bitcast float %1185 to i32, !dbg !14
  %1187 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1186), !dbg !14
  %1188 = bitcast i32 %1187 to float, !dbg !14
  %1189 = insertelement <4 x float> poison, float %1173, i64 0, !dbg !14
  %1190 = insertelement <4 x float> %1189, float %1174, i64 1, !dbg !14
  %1191 = insertelement <4 x float> %1190, float %1184, i64 2, !dbg !14
  %1192 = insertelement <4 x float> %1191, float %1188, i64 3, !dbg !14
  %1193 = insertelement <4 x float> poison, float %1184, i64 0, !dbg !14
  %1194 = insertelement <4 x float> %1193, float %1188, i64 1, !dbg !14
  %1195 = insertelement <4 x float> %1194, float %1179, i64 2, !dbg !14
  %1196 = insertelement <4 x float> %1195, float %1180, i64 3, !dbg !14
  %1197 = select <4 x i1> %70, <4 x float> %1192, <4 x float> %1196, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %41, label %1198, label %1199, !dbg !32

1198:                                             ; preds = %753
  %address.thread854 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local855 = lshr i32 %address.thread854, 1, !dbg !13
  %address.local856 = and i32 %address.local855, 96, !dbg !13
  %address.local857 = and i32 %address.thread854, 15, !dbg !13
  %address.local858 = or disjoint i32 %address.local856, %address.local857, !dbg !13
  %address.local859 = zext nneg i32 %address.local858 to i64, !dbg !13
  %address.local860 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !13
  store float %750, ptr addrspace(3) %address.local860, align 4, !dbg !32
  %address.local862 = getelementptr i8, ptr addrspace(3) %address.local860, i64 64, !dbg !13
  store float %751, ptr addrspace(3) %address.local862, align 4, !dbg !32
  br label %1199, !dbg !32

1199:                                             ; preds = %1198, %753
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.thread863 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %1200 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %1201 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.local877 = zext nneg i32 %address.thread863 to i64, !dbg !10
  %address.local878 = shl nuw nsw i64 %address.local877, 3, !dbg !9
  %address.local879 = and i64 %address.local878, 1536, !dbg !9
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local879, !dbg !9
  %address.local881 = ptrtoint ptr addrspace(3) %address.local880 to i64, !dbg !9
  %address.local882 = trunc i64 %address.local881 to i32, !dbg !9
  %1202 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local882), !dbg !28
  %address.local883 = shl nuw nsw i64 %address.local877, 6, !dbg !28
  %address.local884 = and i64 %address.local883, 16128, !dbg !28
  %address.local885 = shl nuw nsw i64 %address.local877, 4, !dbg !28
  %address.local886 = and i64 %address.local885, 48, !dbg !28
  %1203 = or disjoint i64 %760, %address.local884, !dbg !28
  %1204 = or disjoint i64 %1203, %address.local886, !dbg !28
  %1205 = trunc i64 %1204 to i32, !dbg !28
  %dma.buffer.offset1767 = or i32 %1205, 64, !dbg !28
  %dma.buffer.lds1769 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1202), !dbg !28
  %1206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1769, i32 %dma.buffer.offset1767, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %address.local889 = or disjoint i64 %address.local877, 256, !dbg !9
  %address.local890 = shl nuw nsw i64 %address.local889, 3, !dbg !9
  %address.local891 = and i64 %address.local890, 3584, !dbg !9
  %address.local892 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local891, !dbg !9
  %address.local893 = ptrtoint ptr addrspace(3) %address.local892 to i64, !dbg !9
  %address.local894 = trunc i64 %address.local893 to i32, !dbg !9
  %1207 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !28
  %address.local895 = shl nuw nsw i64 %address.local889, 6, !dbg !28
  %address.local896 = and i64 %address.local895, 32512, !dbg !28
  %1208 = or disjoint i64 %760, %address.local896, !dbg !28
  %1209 = or disjoint i64 %1208, %address.local886, !dbg !28
  %1210 = trunc i64 %1209 to i32, !dbg !28
  %dma.buffer.offset1764 = or i32 %1210, 64, !dbg !28
  %dma.buffer.lds1766 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1207), !dbg !28
  %1211 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1766, i32 %dma.buffer.offset1764, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %1212 = extractvalue [2 x <8 x float>] %81, 0, !dbg !32
  %1213 = shufflevector <8 x float> %1212, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1214 = insertelement <4 x float> poison, float %1200, i64 0, !dbg !32
  %1215 = shufflevector <4 x float> %1214, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1216 = fmul <4 x float> %1213, %1215, !dbg !32
  %1217 = fadd <4 x float> %1066, %1216, !dbg !33
  %1218 = shufflevector <8 x float> %1212, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1219 = fmul <4 x float> %1218, %1215, !dbg !32
  %1220 = fadd <4 x float> %1115, %1219, !dbg !33
  %1221 = extractvalue [2 x <8 x float>] %81, 1, !dbg !32
  %1222 = shufflevector <8 x float> %1221, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1223 = insertelement <4 x float> poison, float %1201, i64 0, !dbg !32
  %1224 = shufflevector <4 x float> %1223, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1225 = fmul <4 x float> %1222, %1224, !dbg !32
  %1226 = fadd <4 x float> %1156, %1225, !dbg !33
  %1227 = shufflevector <8 x float> %1221, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1228 = fmul <4 x float> %1227, %1224, !dbg !32
  %1229 = fadd <4 x float> %1197, %1228, !dbg !33
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local901 = shl nuw nsw i32 %address.thread863, 3, !dbg !14
  %address.local902 = and i32 %address.local901, 504, !dbg !14
  %address.local903 = zext nneg i32 %address.local902 to i64, !dbg !14
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %1230 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #10, !dbg !14
  %1231 = bitcast <4 x i32> %1230 to <8 x half>, !dbg !14
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local904, i64 1024, !dbg !14
  %1232 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #10, !dbg !14
  %1233 = bitcast <4 x i32> %1232 to <8 x half>, !dbg !14
  %address.local908 = getelementptr i8, ptr addrspace(3) %address.local904, i64 2048, !dbg !14
  %1234 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local908, i32 0) #10, !dbg !14
  %1235 = bitcast <4 x i32> %1234 to <8 x half>, !dbg !14
  %address.local910 = getelementptr i8, ptr addrspace(3) %address.local904, i64 3072, !dbg !14
  %1236 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local910, i32 0) #10, !dbg !14
  %1237 = bitcast <4 x i32> %1236 to <8 x half>, !dbg !14
  %address.local912 = getelementptr i8, ptr addrspace(3) %address.local904, i64 4096, !dbg !14
  %1238 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local912, i32 0) #10, !dbg !14
  %1239 = bitcast <4 x i32> %1238 to <8 x half>, !dbg !14
  %address.local914 = getelementptr i8, ptr addrspace(3) %address.local904, i64 5120, !dbg !14
  %1240 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local914, i32 0) #10, !dbg !14
  %1241 = bitcast <4 x i32> %1240 to <8 x half>, !dbg !14
  %address.local916 = getelementptr i8, ptr addrspace(3) %address.local904, i64 6144, !dbg !14
  %1242 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local916, i32 0) #10, !dbg !14
  %1243 = bitcast <4 x i32> %1242 to <8 x half>, !dbg !14
  %address.local918 = getelementptr i8, ptr addrspace(3) %address.local904, i64 7168, !dbg !14
  %1244 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local918, i32 0) #10, !dbg !14
  %1245 = bitcast <4 x i32> %1244 to <8 x half>, !dbg !14
  %1246 = shufflevector <8 x half> %1231, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1246, <4 x float> zeroinitializer) #8, !dbg !14
  %1248 = shufflevector <8 x half> %1233, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1248, <4 x float> %1247) #8, !dbg !14
  %1250 = shufflevector <8 x half> %1235, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1250, <4 x float> %1249) #8, !dbg !14
  %1252 = shufflevector <8 x half> %1237, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1252, <4 x float> %1251) #8, !dbg !14
  %1254 = shufflevector <8 x half> %1239, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1254, <4 x float> %1253) #8, !dbg !14
  %1256 = shufflevector <8 x half> %1241, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1256, <4 x float> %1255) #8, !dbg !14
  %1258 = shufflevector <8 x half> %1243, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1258, <4 x float> %1257) #8, !dbg !14
  %1260 = shufflevector <8 x half> %1245, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1260, <4 x float> %1259) #8, !dbg !14
  %1262 = extractelement <4 x float> %1261, i64 0, !dbg !14
  %1263 = extractelement <4 x float> %1261, i64 1, !dbg !14
  %1264 = extractelement <4 x float> %1261, i64 2, !dbg !14
  %1265 = extractelement <4 x float> %1261, i64 3, !dbg !14
  %1266 = select i1 %.not, float %1263, float %1262, !dbg !14
  %1267 = bitcast float %1266 to i32, !dbg !14
  %1268 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1267), !dbg !14
  %1269 = bitcast i32 %1268 to float, !dbg !14
  %1270 = select i1 %.not, float %1262, float %1269, !dbg !14
  %1271 = select i1 %.not, float %1269, float %1263, !dbg !14
  %1272 = select i1 %.not, float %1265, float %1264, !dbg !14
  %1273 = bitcast float %1272 to i32, !dbg !14
  %1274 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1273), !dbg !14
  %1275 = bitcast i32 %1274 to float, !dbg !14
  %1276 = select i1 %.not, float %1264, float %1275, !dbg !14
  %1277 = select i1 %.not, float %1275, float %1265, !dbg !14
  %1278 = select i1 %.not1, float %1276, float %1270, !dbg !14
  %1279 = bitcast float %1278 to i32, !dbg !14
  %1280 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1279), !dbg !14
  %1281 = bitcast i32 %1280 to float, !dbg !14
  %1282 = select i1 %.not1, float %1277, float %1271, !dbg !14
  %1283 = bitcast float %1282 to i32, !dbg !14
  %1284 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1283), !dbg !14
  %1285 = bitcast i32 %1284 to float, !dbg !14
  %1286 = insertelement <4 x float> poison, float %1270, i64 0, !dbg !14
  %1287 = insertelement <4 x float> %1286, float %1271, i64 1, !dbg !14
  %1288 = insertelement <4 x float> %1287, float %1281, i64 2, !dbg !14
  %1289 = insertelement <4 x float> %1288, float %1285, i64 3, !dbg !14
  %1290 = insertelement <4 x float> poison, float %1281, i64 0, !dbg !14
  %1291 = insertelement <4 x float> %1290, float %1285, i64 1, !dbg !14
  %1292 = insertelement <4 x float> %1291, float %1276, i64 2, !dbg !14
  %1293 = insertelement <4 x float> %1292, float %1277, i64 3, !dbg !14
  %1294 = select <4 x i1> %70, <4 x float> %1289, <4 x float> %1293, !dbg !14
  %1295 = shufflevector <8 x half> %1231, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1295, <4 x float> zeroinitializer) #8, !dbg !14
  %1297 = shufflevector <8 x half> %1233, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1297, <4 x float> %1296) #8, !dbg !14
  %1299 = shufflevector <8 x half> %1235, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1299, <4 x float> %1298) #8, !dbg !14
  %1301 = shufflevector <8 x half> %1237, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1301, <4 x float> %1300) #8, !dbg !14
  %1303 = shufflevector <8 x half> %1239, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1303, <4 x float> %1302) #8, !dbg !14
  %1305 = shufflevector <8 x half> %1241, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1305, <4 x float> %1304) #8, !dbg !14
  %1307 = shufflevector <8 x half> %1243, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1307, <4 x float> %1306) #8, !dbg !14
  %1309 = shufflevector <8 x half> %1245, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1309, <4 x float> %1308) #8, !dbg !14
  %1311 = extractelement <4 x float> %1310, i64 0, !dbg !14
  %1312 = extractelement <4 x float> %1310, i64 1, !dbg !14
  %1313 = extractelement <4 x float> %1310, i64 2, !dbg !14
  %1314 = extractelement <4 x float> %1310, i64 3, !dbg !14
  %1315 = select i1 %.not, float %1312, float %1311, !dbg !14
  %1316 = bitcast float %1315 to i32, !dbg !14
  %1317 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1316), !dbg !14
  %1318 = bitcast i32 %1317 to float, !dbg !14
  %1319 = select i1 %.not, float %1311, float %1318, !dbg !14
  %1320 = select i1 %.not, float %1318, float %1312, !dbg !14
  %1321 = select i1 %.not, float %1314, float %1313, !dbg !14
  %1322 = bitcast float %1321 to i32, !dbg !14
  %1323 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1322), !dbg !14
  %1324 = bitcast i32 %1323 to float, !dbg !14
  %1325 = select i1 %.not, float %1313, float %1324, !dbg !14
  %1326 = select i1 %.not, float %1324, float %1314, !dbg !14
  %1327 = select i1 %.not1, float %1325, float %1319, !dbg !14
  %1328 = bitcast float %1327 to i32, !dbg !14
  %1329 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1328), !dbg !14
  %1330 = bitcast i32 %1329 to float, !dbg !14
  %1331 = select i1 %.not1, float %1326, float %1320, !dbg !14
  %1332 = bitcast float %1331 to i32, !dbg !14
  %1333 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1332), !dbg !14
  %1334 = bitcast i32 %1333 to float, !dbg !14
  %1335 = insertelement <4 x float> poison, float %1319, i64 0, !dbg !14
  %1336 = insertelement <4 x float> %1335, float %1320, i64 1, !dbg !14
  %1337 = insertelement <4 x float> %1336, float %1330, i64 2, !dbg !14
  %1338 = insertelement <4 x float> %1337, float %1334, i64 3, !dbg !14
  %1339 = insertelement <4 x float> poison, float %1330, i64 0, !dbg !14
  %1340 = insertelement <4 x float> %1339, float %1334, i64 1, !dbg !14
  %1341 = insertelement <4 x float> %1340, float %1325, i64 2, !dbg !14
  %1342 = insertelement <4 x float> %1341, float %1326, i64 3, !dbg !14
  %1343 = select <4 x i1> %70, <4 x float> %1338, <4 x float> %1342, !dbg !14
  %1344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1246, <4 x float> zeroinitializer) #8, !dbg !14
  %1345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1248, <4 x float> %1344) #8, !dbg !14
  %1346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1250, <4 x float> %1345) #8, !dbg !14
  %1347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1252, <4 x float> %1346) #8, !dbg !14
  %1348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1254, <4 x float> %1347) #8, !dbg !14
  %1349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1256, <4 x float> %1348) #8, !dbg !14
  %1350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1258, <4 x float> %1349) #8, !dbg !14
  %1351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1260, <4 x float> %1350) #8, !dbg !14
  %1352 = extractelement <4 x float> %1351, i64 0, !dbg !14
  %1353 = extractelement <4 x float> %1351, i64 1, !dbg !14
  %1354 = extractelement <4 x float> %1351, i64 2, !dbg !14
  %1355 = extractelement <4 x float> %1351, i64 3, !dbg !14
  %1356 = select i1 %.not, float %1353, float %1352, !dbg !14
  %1357 = bitcast float %1356 to i32, !dbg !14
  %1358 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1357), !dbg !14
  %1359 = bitcast i32 %1358 to float, !dbg !14
  %1360 = select i1 %.not, float %1352, float %1359, !dbg !14
  %1361 = select i1 %.not, float %1359, float %1353, !dbg !14
  %1362 = select i1 %.not, float %1355, float %1354, !dbg !14
  %1363 = bitcast float %1362 to i32, !dbg !14
  %1364 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1363), !dbg !14
  %1365 = bitcast i32 %1364 to float, !dbg !14
  %1366 = select i1 %.not, float %1354, float %1365, !dbg !14
  %1367 = select i1 %.not, float %1365, float %1355, !dbg !14
  %1368 = select i1 %.not1, float %1366, float %1360, !dbg !14
  %1369 = bitcast float %1368 to i32, !dbg !14
  %1370 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1369), !dbg !14
  %1371 = bitcast i32 %1370 to float, !dbg !14
  %1372 = select i1 %.not1, float %1367, float %1361, !dbg !14
  %1373 = bitcast float %1372 to i32, !dbg !14
  %1374 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1373), !dbg !14
  %1375 = bitcast i32 %1374 to float, !dbg !14
  %1376 = insertelement <4 x float> poison, float %1360, i64 0, !dbg !14
  %1377 = insertelement <4 x float> %1376, float %1361, i64 1, !dbg !14
  %1378 = insertelement <4 x float> %1377, float %1371, i64 2, !dbg !14
  %1379 = insertelement <4 x float> %1378, float %1375, i64 3, !dbg !14
  %1380 = insertelement <4 x float> poison, float %1371, i64 0, !dbg !14
  %1381 = insertelement <4 x float> %1380, float %1375, i64 1, !dbg !14
  %1382 = insertelement <4 x float> %1381, float %1366, i64 2, !dbg !14
  %1383 = insertelement <4 x float> %1382, float %1367, i64 3, !dbg !14
  %1384 = select <4 x i1> %70, <4 x float> %1379, <4 x float> %1383, !dbg !14
  %1385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1295, <4 x float> zeroinitializer) #8, !dbg !14
  %1386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1297, <4 x float> %1385) #8, !dbg !14
  %1387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1299, <4 x float> %1386) #8, !dbg !14
  %1388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1301, <4 x float> %1387) #8, !dbg !14
  %1389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1303, <4 x float> %1388) #8, !dbg !14
  %1390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1305, <4 x float> %1389) #8, !dbg !14
  %1391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1307, <4 x float> %1390) #8, !dbg !14
  %1392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1309, <4 x float> %1391) #8, !dbg !14
  %1393 = extractelement <4 x float> %1392, i64 0, !dbg !14
  %1394 = extractelement <4 x float> %1392, i64 1, !dbg !14
  %1395 = extractelement <4 x float> %1392, i64 2, !dbg !14
  %1396 = extractelement <4 x float> %1392, i64 3, !dbg !14
  %1397 = select i1 %.not, float %1394, float %1393, !dbg !14
  %1398 = bitcast float %1397 to i32, !dbg !14
  %1399 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1398), !dbg !14
  %1400 = bitcast i32 %1399 to float, !dbg !14
  %1401 = select i1 %.not, float %1393, float %1400, !dbg !14
  %1402 = select i1 %.not, float %1400, float %1394, !dbg !14
  %1403 = select i1 %.not, float %1396, float %1395, !dbg !14
  %1404 = bitcast float %1403 to i32, !dbg !14
  %1405 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1404), !dbg !14
  %1406 = bitcast i32 %1405 to float, !dbg !14
  %1407 = select i1 %.not, float %1395, float %1406, !dbg !14
  %1408 = select i1 %.not, float %1406, float %1396, !dbg !14
  %1409 = select i1 %.not1, float %1407, float %1401, !dbg !14
  %1410 = bitcast float %1409 to i32, !dbg !14
  %1411 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1410), !dbg !14
  %1412 = bitcast i32 %1411 to float, !dbg !14
  %1413 = select i1 %.not1, float %1408, float %1402, !dbg !14
  %1414 = bitcast float %1413 to i32, !dbg !14
  %1415 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1414), !dbg !14
  %1416 = bitcast i32 %1415 to float, !dbg !14
  %1417 = insertelement <4 x float> poison, float %1401, i64 0, !dbg !14
  %1418 = insertelement <4 x float> %1417, float %1402, i64 1, !dbg !14
  %1419 = insertelement <4 x float> %1418, float %1412, i64 2, !dbg !14
  %1420 = insertelement <4 x float> %1419, float %1416, i64 3, !dbg !14
  %1421 = insertelement <4 x float> poison, float %1412, i64 0, !dbg !14
  %1422 = insertelement <4 x float> %1421, float %1416, i64 1, !dbg !14
  %1423 = insertelement <4 x float> %1422, float %1407, i64 2, !dbg !14
  %1424 = insertelement <4 x float> %1423, float %1408, i64 3, !dbg !14
  %1425 = select <4 x i1> %70, <4 x float> %1420, <4 x float> %1424, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %41, label %1426, label %1427, !dbg !32

1426:                                             ; preds = %1199
  %address.thread919 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local920 = lshr i32 %address.thread919, 1, !dbg !13
  %address.local921 = and i32 %address.local920, 96, !dbg !13
  %address.local922 = and i32 %address.thread919, 15, !dbg !13
  %address.local923 = or disjoint i32 %address.local921, %address.local922, !dbg !13
  %address.local924 = zext nneg i32 %address.local923 to i64, !dbg !13
  %address.local925 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !13
  store float %750, ptr addrspace(3) %address.local925, align 4, !dbg !32
  %address.local927 = getelementptr i8, ptr addrspace(3) %address.local925, i64 64, !dbg !13
  store float %751, ptr addrspace(3) %address.local927, align 4, !dbg !32
  br label %1427, !dbg !32

1427:                                             ; preds = %1426, %1199
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.thread928 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %1428 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %1429 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.local942 = zext nneg i32 %address.thread928 to i64, !dbg !10
  %address.local943 = shl nuw nsw i64 %address.local942, 3, !dbg !9
  %address.local944 = and i64 %address.local943, 1536, !dbg !9
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local944, !dbg !9
  %address.local946 = ptrtoint ptr addrspace(3) %address.local945 to i64, !dbg !9
  %address.local947 = trunc i64 %address.local946 to i32, !dbg !9
  %1430 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local947), !dbg !28
  %address.local948 = shl nuw nsw i64 %address.local942, 6, !dbg !28
  %address.local949 = and i64 %address.local948, 16128, !dbg !28
  %address.local950 = shl nuw nsw i64 %address.local942, 4, !dbg !28
  %address.local951 = and i64 %address.local950, 48, !dbg !28
  %1431 = or disjoint i64 %760, %address.local949, !dbg !28
  %1432 = or disjoint i64 %1431, %address.local951, !dbg !28
  %1433 = trunc i64 %1432 to i32, !dbg !28
  %dma.buffer.offset1761 = or i32 %1433, 128, !dbg !28
  %dma.buffer.lds1763 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1430), !dbg !28
  %1434 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1763, i32 %dma.buffer.offset1761, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %address.local954 = or disjoint i64 %address.local942, 256, !dbg !9
  %address.local955 = shl nuw nsw i64 %address.local954, 3, !dbg !9
  %address.local956 = and i64 %address.local955, 3584, !dbg !9
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local956, !dbg !9
  %address.local958 = ptrtoint ptr addrspace(3) %address.local957 to i64, !dbg !9
  %address.local959 = trunc i64 %address.local958 to i32, !dbg !9
  %1435 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local959), !dbg !28
  %address.local960 = shl nuw nsw i64 %address.local954, 6, !dbg !28
  %address.local961 = and i64 %address.local960, 32512, !dbg !28
  %1436 = or disjoint i64 %760, %address.local961, !dbg !28
  %1437 = or disjoint i64 %1436, %address.local951, !dbg !28
  %1438 = trunc i64 %1437 to i32, !dbg !28
  %dma.buffer.offset1758 = or i32 %1438, 128, !dbg !28
  %dma.buffer.lds1760 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1435), !dbg !28
  %1439 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1760, i32 %dma.buffer.offset1758, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %1440 = extractvalue [2 x <8 x float>] %78, 0, !dbg !32
  %1441 = shufflevector <8 x float> %1440, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1442 = insertelement <4 x float> poison, float %1428, i64 0, !dbg !32
  %1443 = shufflevector <4 x float> %1442, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1444 = fmul <4 x float> %1441, %1443, !dbg !32
  %1445 = fadd <4 x float> %1294, %1444, !dbg !33
  %1446 = shufflevector <8 x float> %1440, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1447 = fmul <4 x float> %1446, %1443, !dbg !32
  %1448 = fadd <4 x float> %1343, %1447, !dbg !33
  %1449 = extractvalue [2 x <8 x float>] %78, 1, !dbg !32
  %1450 = shufflevector <8 x float> %1449, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1451 = insertelement <4 x float> poison, float %1429, i64 0, !dbg !32
  %1452 = shufflevector <4 x float> %1451, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1453 = fmul <4 x float> %1450, %1452, !dbg !32
  %1454 = fadd <4 x float> %1384, %1453, !dbg !33
  %1455 = shufflevector <8 x float> %1449, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1456 = fmul <4 x float> %1455, %1452, !dbg !32
  %1457 = fadd <4 x float> %1425, %1456, !dbg !33
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local966 = shl nuw nsw i32 %address.thread928, 3, !dbg !14
  %address.local967 = and i32 %address.local966, 504, !dbg !14
  %address.local968 = zext nneg i32 %address.local967 to i64, !dbg !14
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !14
  %1458 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #10, !dbg !14
  %1459 = bitcast <4 x i32> %1458 to <8 x half>, !dbg !14
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local969, i64 1024, !dbg !14
  %1460 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #10, !dbg !14
  %1461 = bitcast <4 x i32> %1460 to <8 x half>, !dbg !14
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local969, i64 2048, !dbg !14
  %1462 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #10, !dbg !14
  %1463 = bitcast <4 x i32> %1462 to <8 x half>, !dbg !14
  %address.local975 = getelementptr i8, ptr addrspace(3) %address.local969, i64 3072, !dbg !14
  %1464 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #10, !dbg !14
  %1465 = bitcast <4 x i32> %1464 to <8 x half>, !dbg !14
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local969, i64 4096, !dbg !14
  %1466 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #10, !dbg !14
  %1467 = bitcast <4 x i32> %1466 to <8 x half>, !dbg !14
  %address.local979 = getelementptr i8, ptr addrspace(3) %address.local969, i64 5120, !dbg !14
  %1468 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #10, !dbg !14
  %1469 = bitcast <4 x i32> %1468 to <8 x half>, !dbg !14
  %address.local981 = getelementptr i8, ptr addrspace(3) %address.local969, i64 6144, !dbg !14
  %1470 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #10, !dbg !14
  %1471 = bitcast <4 x i32> %1470 to <8 x half>, !dbg !14
  %address.local983 = getelementptr i8, ptr addrspace(3) %address.local969, i64 7168, !dbg !14
  %1472 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #10, !dbg !14
  %1473 = bitcast <4 x i32> %1472 to <8 x half>, !dbg !14
  %1474 = shufflevector <8 x half> %1459, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1475 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1474, <4 x float> zeroinitializer) #8, !dbg !14
  %1476 = shufflevector <8 x half> %1461, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1477 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1476, <4 x float> %1475) #8, !dbg !14
  %1478 = shufflevector <8 x half> %1463, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1479 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1478, <4 x float> %1477) #8, !dbg !14
  %1480 = shufflevector <8 x half> %1465, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1481 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1480, <4 x float> %1479) #8, !dbg !14
  %1482 = shufflevector <8 x half> %1467, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1483 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1482, <4 x float> %1481) #8, !dbg !14
  %1484 = shufflevector <8 x half> %1469, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1485 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1484, <4 x float> %1483) #8, !dbg !14
  %1486 = shufflevector <8 x half> %1471, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1487 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1486, <4 x float> %1485) #8, !dbg !14
  %1488 = shufflevector <8 x half> %1473, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1489 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1488, <4 x float> %1487) #8, !dbg !14
  %1490 = extractelement <4 x float> %1489, i64 0, !dbg !14
  %1491 = extractelement <4 x float> %1489, i64 1, !dbg !14
  %1492 = extractelement <4 x float> %1489, i64 2, !dbg !14
  %1493 = extractelement <4 x float> %1489, i64 3, !dbg !14
  %1494 = select i1 %.not, float %1491, float %1490, !dbg !14
  %1495 = bitcast float %1494 to i32, !dbg !14
  %1496 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1495), !dbg !14
  %1497 = bitcast i32 %1496 to float, !dbg !14
  %1498 = select i1 %.not, float %1490, float %1497, !dbg !14
  %1499 = select i1 %.not, float %1497, float %1491, !dbg !14
  %1500 = select i1 %.not, float %1493, float %1492, !dbg !14
  %1501 = bitcast float %1500 to i32, !dbg !14
  %1502 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1501), !dbg !14
  %1503 = bitcast i32 %1502 to float, !dbg !14
  %1504 = select i1 %.not, float %1492, float %1503, !dbg !14
  %1505 = select i1 %.not, float %1503, float %1493, !dbg !14
  %1506 = select i1 %.not1, float %1504, float %1498, !dbg !14
  %1507 = bitcast float %1506 to i32, !dbg !14
  %1508 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1507), !dbg !14
  %1509 = bitcast i32 %1508 to float, !dbg !14
  %1510 = select i1 %.not1, float %1505, float %1499, !dbg !14
  %1511 = bitcast float %1510 to i32, !dbg !14
  %1512 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1511), !dbg !14
  %1513 = bitcast i32 %1512 to float, !dbg !14
  %1514 = insertelement <4 x float> poison, float %1498, i64 0, !dbg !14
  %1515 = insertelement <4 x float> %1514, float %1499, i64 1, !dbg !14
  %1516 = insertelement <4 x float> %1515, float %1509, i64 2, !dbg !14
  %1517 = insertelement <4 x float> %1516, float %1513, i64 3, !dbg !14
  %1518 = insertelement <4 x float> poison, float %1509, i64 0, !dbg !14
  %1519 = insertelement <4 x float> %1518, float %1513, i64 1, !dbg !14
  %1520 = insertelement <4 x float> %1519, float %1504, i64 2, !dbg !14
  %1521 = insertelement <4 x float> %1520, float %1505, i64 3, !dbg !14
  %1522 = select <4 x i1> %70, <4 x float> %1517, <4 x float> %1521, !dbg !14
  %1523 = shufflevector <8 x half> %1459, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1524 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1523, <4 x float> zeroinitializer) #8, !dbg !14
  %1525 = shufflevector <8 x half> %1461, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1526 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1525, <4 x float> %1524) #8, !dbg !14
  %1527 = shufflevector <8 x half> %1463, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1528 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1527, <4 x float> %1526) #8, !dbg !14
  %1529 = shufflevector <8 x half> %1465, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1530 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1529, <4 x float> %1528) #8, !dbg !14
  %1531 = shufflevector <8 x half> %1467, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1532 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1531, <4 x float> %1530) #8, !dbg !14
  %1533 = shufflevector <8 x half> %1469, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1534 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1533, <4 x float> %1532) #8, !dbg !14
  %1535 = shufflevector <8 x half> %1471, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1536 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1535, <4 x float> %1534) #8, !dbg !14
  %1537 = shufflevector <8 x half> %1473, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1538 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1537, <4 x float> %1536) #8, !dbg !14
  %1539 = extractelement <4 x float> %1538, i64 0, !dbg !14
  %1540 = extractelement <4 x float> %1538, i64 1, !dbg !14
  %1541 = extractelement <4 x float> %1538, i64 2, !dbg !14
  %1542 = extractelement <4 x float> %1538, i64 3, !dbg !14
  %1543 = select i1 %.not, float %1540, float %1539, !dbg !14
  %1544 = bitcast float %1543 to i32, !dbg !14
  %1545 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1544), !dbg !14
  %1546 = bitcast i32 %1545 to float, !dbg !14
  %1547 = select i1 %.not, float %1539, float %1546, !dbg !14
  %1548 = select i1 %.not, float %1546, float %1540, !dbg !14
  %1549 = select i1 %.not, float %1542, float %1541, !dbg !14
  %1550 = bitcast float %1549 to i32, !dbg !14
  %1551 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1550), !dbg !14
  %1552 = bitcast i32 %1551 to float, !dbg !14
  %1553 = select i1 %.not, float %1541, float %1552, !dbg !14
  %1554 = select i1 %.not, float %1552, float %1542, !dbg !14
  %1555 = select i1 %.not1, float %1553, float %1547, !dbg !14
  %1556 = bitcast float %1555 to i32, !dbg !14
  %1557 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1556), !dbg !14
  %1558 = bitcast i32 %1557 to float, !dbg !14
  %1559 = select i1 %.not1, float %1554, float %1548, !dbg !14
  %1560 = bitcast float %1559 to i32, !dbg !14
  %1561 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1560), !dbg !14
  %1562 = bitcast i32 %1561 to float, !dbg !14
  %1563 = insertelement <4 x float> poison, float %1547, i64 0, !dbg !14
  %1564 = insertelement <4 x float> %1563, float %1548, i64 1, !dbg !14
  %1565 = insertelement <4 x float> %1564, float %1558, i64 2, !dbg !14
  %1566 = insertelement <4 x float> %1565, float %1562, i64 3, !dbg !14
  %1567 = insertelement <4 x float> poison, float %1558, i64 0, !dbg !14
  %1568 = insertelement <4 x float> %1567, float %1562, i64 1, !dbg !14
  %1569 = insertelement <4 x float> %1568, float %1553, i64 2, !dbg !14
  %1570 = insertelement <4 x float> %1569, float %1554, i64 3, !dbg !14
  %1571 = select <4 x i1> %70, <4 x float> %1566, <4 x float> %1570, !dbg !14
  %1572 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1474, <4 x float> zeroinitializer) #8, !dbg !14
  %1573 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1476, <4 x float> %1572) #8, !dbg !14
  %1574 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1478, <4 x float> %1573) #8, !dbg !14
  %1575 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1480, <4 x float> %1574) #8, !dbg !14
  %1576 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1482, <4 x float> %1575) #8, !dbg !14
  %1577 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1484, <4 x float> %1576) #8, !dbg !14
  %1578 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1486, <4 x float> %1577) #8, !dbg !14
  %1579 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1488, <4 x float> %1578) #8, !dbg !14
  %1580 = extractelement <4 x float> %1579, i64 0, !dbg !14
  %1581 = extractelement <4 x float> %1579, i64 1, !dbg !14
  %1582 = extractelement <4 x float> %1579, i64 2, !dbg !14
  %1583 = extractelement <4 x float> %1579, i64 3, !dbg !14
  %1584 = select i1 %.not, float %1581, float %1580, !dbg !14
  %1585 = bitcast float %1584 to i32, !dbg !14
  %1586 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1585), !dbg !14
  %1587 = bitcast i32 %1586 to float, !dbg !14
  %1588 = select i1 %.not, float %1580, float %1587, !dbg !14
  %1589 = select i1 %.not, float %1587, float %1581, !dbg !14
  %1590 = select i1 %.not, float %1583, float %1582, !dbg !14
  %1591 = bitcast float %1590 to i32, !dbg !14
  %1592 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1591), !dbg !14
  %1593 = bitcast i32 %1592 to float, !dbg !14
  %1594 = select i1 %.not, float %1582, float %1593, !dbg !14
  %1595 = select i1 %.not, float %1593, float %1583, !dbg !14
  %1596 = select i1 %.not1, float %1594, float %1588, !dbg !14
  %1597 = bitcast float %1596 to i32, !dbg !14
  %1598 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1597), !dbg !14
  %1599 = bitcast i32 %1598 to float, !dbg !14
  %1600 = select i1 %.not1, float %1595, float %1589, !dbg !14
  %1601 = bitcast float %1600 to i32, !dbg !14
  %1602 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1601), !dbg !14
  %1603 = bitcast i32 %1602 to float, !dbg !14
  %1604 = insertelement <4 x float> poison, float %1588, i64 0, !dbg !14
  %1605 = insertelement <4 x float> %1604, float %1589, i64 1, !dbg !14
  %1606 = insertelement <4 x float> %1605, float %1599, i64 2, !dbg !14
  %1607 = insertelement <4 x float> %1606, float %1603, i64 3, !dbg !14
  %1608 = insertelement <4 x float> poison, float %1599, i64 0, !dbg !14
  %1609 = insertelement <4 x float> %1608, float %1603, i64 1, !dbg !14
  %1610 = insertelement <4 x float> %1609, float %1594, i64 2, !dbg !14
  %1611 = insertelement <4 x float> %1610, float %1595, i64 3, !dbg !14
  %1612 = select <4 x i1> %70, <4 x float> %1607, <4 x float> %1611, !dbg !14
  %1613 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1523, <4 x float> zeroinitializer) #8, !dbg !14
  %1614 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1525, <4 x float> %1613) #8, !dbg !14
  %1615 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1527, <4 x float> %1614) #8, !dbg !14
  %1616 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1529, <4 x float> %1615) #8, !dbg !14
  %1617 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1531, <4 x float> %1616) #8, !dbg !14
  %1618 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1533, <4 x float> %1617) #8, !dbg !14
  %1619 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1535, <4 x float> %1618) #8, !dbg !14
  %1620 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1537, <4 x float> %1619) #8, !dbg !14
  %1621 = extractelement <4 x float> %1620, i64 0, !dbg !14
  %1622 = extractelement <4 x float> %1620, i64 1, !dbg !14
  %1623 = extractelement <4 x float> %1620, i64 2, !dbg !14
  %1624 = extractelement <4 x float> %1620, i64 3, !dbg !14
  %1625 = select i1 %.not, float %1622, float %1621, !dbg !14
  %1626 = bitcast float %1625 to i32, !dbg !14
  %1627 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1626), !dbg !14
  %1628 = bitcast i32 %1627 to float, !dbg !14
  %1629 = select i1 %.not, float %1621, float %1628, !dbg !14
  %1630 = select i1 %.not, float %1628, float %1622, !dbg !14
  %1631 = select i1 %.not, float %1624, float %1623, !dbg !14
  %1632 = bitcast float %1631 to i32, !dbg !14
  %1633 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1632), !dbg !14
  %1634 = bitcast i32 %1633 to float, !dbg !14
  %1635 = select i1 %.not, float %1623, float %1634, !dbg !14
  %1636 = select i1 %.not, float %1634, float %1624, !dbg !14
  %1637 = select i1 %.not1, float %1635, float %1629, !dbg !14
  %1638 = bitcast float %1637 to i32, !dbg !14
  %1639 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1638), !dbg !14
  %1640 = bitcast i32 %1639 to float, !dbg !14
  %1641 = select i1 %.not1, float %1636, float %1630, !dbg !14
  %1642 = bitcast float %1641 to i32, !dbg !14
  %1643 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1642), !dbg !14
  %1644 = bitcast i32 %1643 to float, !dbg !14
  %1645 = insertelement <4 x float> poison, float %1629, i64 0, !dbg !14
  %1646 = insertelement <4 x float> %1645, float %1630, i64 1, !dbg !14
  %1647 = insertelement <4 x float> %1646, float %1640, i64 2, !dbg !14
  %1648 = insertelement <4 x float> %1647, float %1644, i64 3, !dbg !14
  %1649 = insertelement <4 x float> poison, float %1640, i64 0, !dbg !14
  %1650 = insertelement <4 x float> %1649, float %1644, i64 1, !dbg !14
  %1651 = insertelement <4 x float> %1650, float %1635, i64 2, !dbg !14
  %1652 = insertelement <4 x float> %1651, float %1636, i64 3, !dbg !14
  %1653 = select <4 x i1> %70, <4 x float> %1648, <4 x float> %1652, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %41, label %1654, label %1655, !dbg !32

1654:                                             ; preds = %1427
  %address.thread984 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local985 = lshr i32 %address.thread984, 1, !dbg !13
  %address.local986 = and i32 %address.local985, 96, !dbg !13
  %address.local987 = and i32 %address.thread984, 15, !dbg !13
  %address.local988 = or disjoint i32 %address.local986, %address.local987, !dbg !13
  %address.local989 = zext nneg i32 %address.local988 to i64, !dbg !13
  %address.local990 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !13
  store float %750, ptr addrspace(3) %address.local990, align 4, !dbg !32
  %address.local992 = getelementptr i8, ptr addrspace(3) %address.local990, i64 64, !dbg !13
  store float %751, ptr addrspace(3) %address.local992, align 4, !dbg !32
  br label %1655, !dbg !32

1655:                                             ; preds = %1654, %1427
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.thread993 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %1656 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %1657 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.local1007 = zext nneg i32 %address.thread993 to i64, !dbg !10
  %address.local1008 = shl nuw nsw i64 %address.local1007, 3, !dbg !9
  %address.local1009 = and i64 %address.local1008, 1536, !dbg !9
  %address.local1010 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1009, !dbg !9
  %address.local1011 = ptrtoint ptr addrspace(3) %address.local1010 to i64, !dbg !9
  %address.local1012 = trunc i64 %address.local1011 to i32, !dbg !9
  %1658 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1012), !dbg !28
  %address.local1013 = shl nuw nsw i64 %address.local1007, 6, !dbg !28
  %address.local1014 = and i64 %address.local1013, 16128, !dbg !28
  %address.local1015 = shl nuw nsw i64 %address.local1007, 4, !dbg !28
  %address.local1016 = and i64 %address.local1015, 48, !dbg !28
  %1659 = or disjoint i64 %760, %address.local1014, !dbg !28
  %1660 = or disjoint i64 %1659, %address.local1016, !dbg !28
  %1661 = trunc i64 %1660 to i32, !dbg !28
  %dma.buffer.offset1755 = or i32 %1661, 192, !dbg !28
  %dma.buffer.lds1757 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1658), !dbg !28
  %1662 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1757, i32 %dma.buffer.offset1755, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %address.local1019 = or disjoint i64 %address.local1007, 256, !dbg !9
  %address.local1020 = shl nuw nsw i64 %address.local1019, 3, !dbg !9
  %address.local1021 = and i64 %address.local1020, 3584, !dbg !9
  %address.local1022 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1021, !dbg !9
  %address.local1023 = ptrtoint ptr addrspace(3) %address.local1022 to i64, !dbg !9
  %address.local1024 = trunc i64 %address.local1023 to i32, !dbg !9
  %1663 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1024), !dbg !28
  %address.local1025 = shl nuw nsw i64 %address.local1019, 6, !dbg !28
  %address.local1026 = and i64 %address.local1025, 32512, !dbg !28
  %1664 = or disjoint i64 %760, %address.local1026, !dbg !28
  %1665 = or disjoint i64 %1664, %address.local1016, !dbg !28
  %1666 = trunc i64 %1665 to i32, !dbg !28
  %dma.buffer.offset = or i32 %1666, 192, !dbg !28
  %dma.buffer.lds = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1663), !dbg !28
  %1667 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1774) #9, !dbg !28
  %1668 = extractvalue [2 x <8 x float>] %77, 0, !dbg !32
  %1669 = shufflevector <8 x float> %1668, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1670 = insertelement <4 x float> poison, float %1656, i64 0, !dbg !32
  %1671 = shufflevector <4 x float> %1670, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1672 = fmul <4 x float> %1669, %1671, !dbg !32
  %1673 = fadd <4 x float> %1522, %1672, !dbg !33
  %1674 = shufflevector <8 x float> %1668, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1675 = fmul <4 x float> %1674, %1671, !dbg !32
  %1676 = fadd <4 x float> %1571, %1675, !dbg !33
  %1677 = extractvalue [2 x <8 x float>] %77, 1, !dbg !32
  %1678 = shufflevector <8 x float> %1677, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1679 = insertelement <4 x float> poison, float %1657, i64 0, !dbg !32
  %1680 = shufflevector <4 x float> %1679, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1681 = fmul <4 x float> %1678, %1680, !dbg !32
  %1682 = fadd <4 x float> %1612, %1681, !dbg !33
  %1683 = shufflevector <8 x float> %1677, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1684 = fmul <4 x float> %1683, %1680, !dbg !32
  %1685 = fadd <4 x float> %1653, %1684, !dbg !33
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  tail call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.local1031 = shl nuw nsw i32 %address.thread993, 3, !dbg !14
  %address.local1032 = and i32 %address.local1031, 504, !dbg !14
  %address.local1033 = zext nneg i32 %address.local1032 to i64, !dbg !14
  %address.local1034 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !14
  %1686 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #10, !dbg !14
  %1687 = bitcast <4 x i32> %1686 to <8 x half>, !dbg !14
  %address.local1036 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 1024, !dbg !14
  %1688 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1036, i32 0) #10, !dbg !14
  %1689 = bitcast <4 x i32> %1688 to <8 x half>, !dbg !14
  %address.local1038 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 2048, !dbg !14
  %1690 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1038, i32 0) #10, !dbg !14
  %1691 = bitcast <4 x i32> %1690 to <8 x half>, !dbg !14
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 3072, !dbg !14
  %1692 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #10, !dbg !14
  %1693 = bitcast <4 x i32> %1692 to <8 x half>, !dbg !14
  %address.local1042 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 4096, !dbg !14
  %1694 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1042, i32 0) #10, !dbg !14
  %1695 = bitcast <4 x i32> %1694 to <8 x half>, !dbg !14
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 5120, !dbg !14
  %1696 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #10, !dbg !14
  %1697 = bitcast <4 x i32> %1696 to <8 x half>, !dbg !14
  %address.local1046 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 6144, !dbg !14
  %1698 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1046, i32 0) #10, !dbg !14
  %1699 = bitcast <4 x i32> %1698 to <8 x half>, !dbg !14
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1034, i64 7168, !dbg !14
  %1700 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #10, !dbg !14
  %1701 = bitcast <4 x i32> %1700 to <8 x half>, !dbg !14
  %1702 = shufflevector <8 x half> %1687, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1703 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1702, <4 x float> zeroinitializer) #8, !dbg !14
  %1704 = shufflevector <8 x half> %1689, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1705 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1704, <4 x float> %1703) #8, !dbg !14
  %1706 = shufflevector <8 x half> %1691, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1707 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1706, <4 x float> %1705) #8, !dbg !14
  %1708 = shufflevector <8 x half> %1693, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1709 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1708, <4 x float> %1707) #8, !dbg !14
  %1710 = shufflevector <8 x half> %1695, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1711 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1710, <4 x float> %1709) #8, !dbg !14
  %1712 = shufflevector <8 x half> %1697, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1713 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1712, <4 x float> %1711) #8, !dbg !14
  %1714 = shufflevector <8 x half> %1699, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1715 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1714, <4 x float> %1713) #8, !dbg !14
  %1716 = shufflevector <8 x half> %1701, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1717 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1716, <4 x float> %1715) #8, !dbg !14
  %1718 = extractelement <4 x float> %1717, i64 0, !dbg !14
  %1719 = extractelement <4 x float> %1717, i64 1, !dbg !14
  %1720 = extractelement <4 x float> %1717, i64 2, !dbg !14
  %1721 = extractelement <4 x float> %1717, i64 3, !dbg !14
  %1722 = select i1 %.not, float %1719, float %1718, !dbg !14
  %1723 = bitcast float %1722 to i32, !dbg !14
  %1724 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1723), !dbg !14
  %1725 = bitcast i32 %1724 to float, !dbg !14
  %1726 = select i1 %.not, float %1718, float %1725, !dbg !14
  %1727 = select i1 %.not, float %1725, float %1719, !dbg !14
  %1728 = select i1 %.not, float %1721, float %1720, !dbg !14
  %1729 = bitcast float %1728 to i32, !dbg !14
  %1730 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1729), !dbg !14
  %1731 = bitcast i32 %1730 to float, !dbg !14
  %1732 = select i1 %.not, float %1720, float %1731, !dbg !14
  %1733 = select i1 %.not, float %1731, float %1721, !dbg !14
  %1734 = select i1 %.not1, float %1732, float %1726, !dbg !14
  %1735 = bitcast float %1734 to i32, !dbg !14
  %1736 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1735), !dbg !14
  %1737 = bitcast i32 %1736 to float, !dbg !14
  %1738 = select i1 %.not1, float %1733, float %1727, !dbg !14
  %1739 = bitcast float %1738 to i32, !dbg !14
  %1740 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1739), !dbg !14
  %1741 = bitcast i32 %1740 to float, !dbg !14
  %1742 = insertelement <4 x float> poison, float %1726, i64 0, !dbg !14
  %1743 = insertelement <4 x float> %1742, float %1727, i64 1, !dbg !14
  %1744 = insertelement <4 x float> %1743, float %1737, i64 2, !dbg !14
  %1745 = insertelement <4 x float> %1744, float %1741, i64 3, !dbg !14
  %1746 = insertelement <4 x float> poison, float %1737, i64 0, !dbg !14
  %1747 = insertelement <4 x float> %1746, float %1741, i64 1, !dbg !14
  %1748 = insertelement <4 x float> %1747, float %1732, i64 2, !dbg !14
  %1749 = insertelement <4 x float> %1748, float %1733, i64 3, !dbg !14
  %1750 = select <4 x i1> %70, <4 x float> %1745, <4 x float> %1749, !dbg !14
  %1751 = shufflevector <8 x half> %1687, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1752 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1751, <4 x float> zeroinitializer) #8, !dbg !14
  %1753 = shufflevector <8 x half> %1689, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1754 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1753, <4 x float> %1752) #8, !dbg !14
  %1755 = shufflevector <8 x half> %1691, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1756 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1755, <4 x float> %1754) #8, !dbg !14
  %1757 = shufflevector <8 x half> %1693, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1758 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1757, <4 x float> %1756) #8, !dbg !14
  %1759 = shufflevector <8 x half> %1695, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1760 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1759, <4 x float> %1758) #8, !dbg !14
  %1761 = shufflevector <8 x half> %1697, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1762 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1761, <4 x float> %1760) #8, !dbg !14
  %1763 = shufflevector <8 x half> %1699, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1764 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1763, <4 x float> %1762) #8, !dbg !14
  %1765 = shufflevector <8 x half> %1701, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1766 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1765, <4 x float> %1764) #8, !dbg !14
  %1767 = extractelement <4 x float> %1766, i64 0, !dbg !14
  %1768 = extractelement <4 x float> %1766, i64 1, !dbg !14
  %1769 = extractelement <4 x float> %1766, i64 2, !dbg !14
  %1770 = extractelement <4 x float> %1766, i64 3, !dbg !14
  %1771 = select i1 %.not, float %1768, float %1767, !dbg !14
  %1772 = bitcast float %1771 to i32, !dbg !14
  %1773 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1772), !dbg !14
  %1774 = bitcast i32 %1773 to float, !dbg !14
  %1775 = select i1 %.not, float %1767, float %1774, !dbg !14
  %1776 = select i1 %.not, float %1774, float %1768, !dbg !14
  %1777 = select i1 %.not, float %1770, float %1769, !dbg !14
  %1778 = bitcast float %1777 to i32, !dbg !14
  %1779 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1778), !dbg !14
  %1780 = bitcast i32 %1779 to float, !dbg !14
  %1781 = select i1 %.not, float %1769, float %1780, !dbg !14
  %1782 = select i1 %.not, float %1780, float %1770, !dbg !14
  %1783 = select i1 %.not1, float %1781, float %1775, !dbg !14
  %1784 = bitcast float %1783 to i32, !dbg !14
  %1785 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1784), !dbg !14
  %1786 = bitcast i32 %1785 to float, !dbg !14
  %1787 = select i1 %.not1, float %1782, float %1776, !dbg !14
  %1788 = bitcast float %1787 to i32, !dbg !14
  %1789 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1788), !dbg !14
  %1790 = bitcast i32 %1789 to float, !dbg !14
  %1791 = insertelement <4 x float> poison, float %1775, i64 0, !dbg !14
  %1792 = insertelement <4 x float> %1791, float %1776, i64 1, !dbg !14
  %1793 = insertelement <4 x float> %1792, float %1786, i64 2, !dbg !14
  %1794 = insertelement <4 x float> %1793, float %1790, i64 3, !dbg !14
  %1795 = insertelement <4 x float> poison, float %1786, i64 0, !dbg !14
  %1796 = insertelement <4 x float> %1795, float %1790, i64 1, !dbg !14
  %1797 = insertelement <4 x float> %1796, float %1781, i64 2, !dbg !14
  %1798 = insertelement <4 x float> %1797, float %1782, i64 3, !dbg !14
  %1799 = select <4 x i1> %70, <4 x float> %1794, <4 x float> %1798, !dbg !14
  %1800 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1702, <4 x float> zeroinitializer) #8, !dbg !14
  %1801 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1704, <4 x float> %1800) #8, !dbg !14
  %1802 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1706, <4 x float> %1801) #8, !dbg !14
  %1803 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1708, <4 x float> %1802) #8, !dbg !14
  %1804 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1710, <4 x float> %1803) #8, !dbg !14
  %1805 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1712, <4 x float> %1804) #8, !dbg !14
  %1806 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1714, <4 x float> %1805) #8, !dbg !14
  %1807 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1716, <4 x float> %1806) #8, !dbg !14
  %1808 = extractelement <4 x float> %1807, i64 0, !dbg !14
  %1809 = extractelement <4 x float> %1807, i64 1, !dbg !14
  %1810 = extractelement <4 x float> %1807, i64 2, !dbg !14
  %1811 = extractelement <4 x float> %1807, i64 3, !dbg !14
  %1812 = select i1 %.not, float %1809, float %1808, !dbg !14
  %1813 = bitcast float %1812 to i32, !dbg !14
  %1814 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1813), !dbg !14
  %1815 = bitcast i32 %1814 to float, !dbg !14
  %1816 = select i1 %.not, float %1808, float %1815, !dbg !14
  %1817 = select i1 %.not, float %1815, float %1809, !dbg !14
  %1818 = select i1 %.not, float %1811, float %1810, !dbg !14
  %1819 = bitcast float %1818 to i32, !dbg !14
  %1820 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1819), !dbg !14
  %1821 = bitcast i32 %1820 to float, !dbg !14
  %1822 = select i1 %.not, float %1810, float %1821, !dbg !14
  %1823 = select i1 %.not, float %1821, float %1811, !dbg !14
  %1824 = select i1 %.not1, float %1822, float %1816, !dbg !14
  %1825 = bitcast float %1824 to i32, !dbg !14
  %1826 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1825), !dbg !14
  %1827 = bitcast i32 %1826 to float, !dbg !14
  %1828 = select i1 %.not1, float %1823, float %1817, !dbg !14
  %1829 = bitcast float %1828 to i32, !dbg !14
  %1830 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1829), !dbg !14
  %1831 = bitcast i32 %1830 to float, !dbg !14
  %1832 = insertelement <4 x float> poison, float %1816, i64 0, !dbg !14
  %1833 = insertelement <4 x float> %1832, float %1817, i64 1, !dbg !14
  %1834 = insertelement <4 x float> %1833, float %1827, i64 2, !dbg !14
  %1835 = insertelement <4 x float> %1834, float %1831, i64 3, !dbg !14
  %1836 = insertelement <4 x float> poison, float %1827, i64 0, !dbg !14
  %1837 = insertelement <4 x float> %1836, float %1831, i64 1, !dbg !14
  %1838 = insertelement <4 x float> %1837, float %1822, i64 2, !dbg !14
  %1839 = insertelement <4 x float> %1838, float %1823, i64 3, !dbg !14
  %1840 = select <4 x i1> %70, <4 x float> %1835, <4 x float> %1839, !dbg !14
  %1841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1751, <4 x float> zeroinitializer) #8, !dbg !14
  %1842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1753, <4 x float> %1841) #8, !dbg !14
  %1843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1755, <4 x float> %1842) #8, !dbg !14
  %1844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1757, <4 x float> %1843) #8, !dbg !14
  %1845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1759, <4 x float> %1844) #8, !dbg !14
  %1846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1761, <4 x float> %1845) #8, !dbg !14
  %1847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1763, <4 x float> %1846) #8, !dbg !14
  %1848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1765, <4 x float> %1847) #8, !dbg !14
  %1849 = extractelement <4 x float> %1848, i64 0, !dbg !14
  %1850 = extractelement <4 x float> %1848, i64 1, !dbg !14
  %1851 = extractelement <4 x float> %1848, i64 2, !dbg !14
  %1852 = extractelement <4 x float> %1848, i64 3, !dbg !14
  %1853 = select i1 %.not, float %1850, float %1849, !dbg !14
  %1854 = bitcast float %1853 to i32, !dbg !14
  %1855 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1854), !dbg !14
  %1856 = bitcast i32 %1855 to float, !dbg !14
  %1857 = select i1 %.not, float %1849, float %1856, !dbg !14
  %1858 = select i1 %.not, float %1856, float %1850, !dbg !14
  %1859 = select i1 %.not, float %1852, float %1851, !dbg !14
  %1860 = bitcast float %1859 to i32, !dbg !14
  %1861 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %695, i32 %1860), !dbg !14
  %1862 = bitcast i32 %1861 to float, !dbg !14
  %1863 = select i1 %.not, float %1851, float %1862, !dbg !14
  %1864 = select i1 %.not, float %1862, float %1852, !dbg !14
  %1865 = select i1 %.not1, float %1863, float %1857, !dbg !14
  %1866 = bitcast float %1865 to i32, !dbg !14
  %1867 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1866), !dbg !14
  %1868 = bitcast i32 %1867 to float, !dbg !14
  %1869 = select i1 %.not1, float %1864, float %1858, !dbg !14
  %1870 = bitcast float %1869 to i32, !dbg !14
  %1871 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %1870), !dbg !14
  %1872 = bitcast i32 %1871 to float, !dbg !14
  %1873 = insertelement <4 x float> poison, float %1857, i64 0, !dbg !14
  %1874 = insertelement <4 x float> %1873, float %1858, i64 1, !dbg !14
  %1875 = insertelement <4 x float> %1874, float %1868, i64 2, !dbg !14
  %1876 = insertelement <4 x float> %1875, float %1872, i64 3, !dbg !14
  %1877 = insertelement <4 x float> poison, float %1868, i64 0, !dbg !14
  %1878 = insertelement <4 x float> %1877, float %1872, i64 1, !dbg !14
  %1879 = insertelement <4 x float> %1878, float %1863, i64 2, !dbg !14
  %1880 = insertelement <4 x float> %1879, float %1864, i64 3, !dbg !14
  %1881 = select <4 x i1> %70, <4 x float> %1876, <4 x float> %1880, !dbg !14
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  br i1 %41, label %1882, label %1883, !dbg !32

1882:                                             ; preds = %1655
  %address.thread1049 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !32, !range !11, !frisk.register_lifetimes !5
  %address.local1050 = lshr i32 %address.thread1049, 1, !dbg !13
  %address.local1051 = and i32 %address.local1050, 96, !dbg !13
  %address.local1052 = and i32 %address.thread1049, 15, !dbg !13
  %address.local1053 = or disjoint i32 %address.local1051, %address.local1052, !dbg !13
  %address.local1054 = zext nneg i32 %address.local1053 to i64, !dbg !13
  %address.local1055 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !13
  store float %750, ptr addrspace(3) %address.local1055, align 4, !dbg !32
  %address.local1057 = getelementptr i8, ptr addrspace(3) %address.local1055, i64 64, !dbg !13
  store float %751, ptr addrspace(3) %address.local1057, align 4, !dbg !32
  br label %1883, !dbg !32

1883:                                             ; preds = %1882, %1655
  fence syncscope("workgroup") release, !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %1884 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %1885 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  %1886 = extractvalue [2 x <8 x float>] %76, 0, !dbg !32
  %1887 = shufflevector <8 x float> %1886, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1888 = insertelement <4 x float> poison, float %1884, i64 0, !dbg !32
  %1889 = shufflevector <4 x float> %1888, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1890 = fmul <4 x float> %1887, %1889, !dbg !32
  %1891 = fadd <4 x float> %1750, %1890, !dbg !33
  %1892 = shufflevector <8 x float> %1886, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1893 = fmul <4 x float> %1892, %1889, !dbg !32
  %1894 = fadd <4 x float> %1799, %1893, !dbg !33
  %1895 = extractvalue [2 x <8 x float>] %76, 1, !dbg !32
  %1896 = shufflevector <8 x float> %1895, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1897 = insertelement <4 x float> poison, float %1885, i64 0, !dbg !32
  %1898 = shufflevector <4 x float> %1897, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !32
  %1899 = fmul <4 x float> %1896, %1898, !dbg !32
  %1900 = fadd <4 x float> %1840, %1899, !dbg !33
  %1901 = shufflevector <8 x float> %1895, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1902 = fmul <4 x float> %1901, %1898, !dbg !32
  %1903 = fadd <4 x float> %1881, %1902, !dbg !33
  %1904 = shufflevector <4 x float> %1217, <4 x float> %1220, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1905 = insertvalue [2 x <8 x float>] poison, <8 x float> %1904, 0, !dbg !17
  %1906 = shufflevector <4 x float> %1226, <4 x float> %1229, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1907 = insertvalue [2 x <8 x float>] %1905, <8 x float> %1906, 1, !dbg !17
  %1908 = insertelement <1 x float> poison, float %992, i64 0, !dbg !17
  %1909 = insertvalue [2 x <1 x float>] poison, <1 x float> %1908, 0, !dbg !17
  %1910 = insertelement <1 x float> poison, float %1001, i64 0, !dbg !17
  %1911 = insertvalue [2 x <1 x float>] %1909, <1 x float> %1910, 1, !dbg !17
  %1912 = insertvalue [2 x <1 x float>] poison, <1 x float> %710, 0, !dbg !17
  %1913 = insertvalue [2 x <1 x float>] %1912, <1 x float> %723, 1, !dbg !17
  %1914 = shufflevector <4 x float> %1445, <4 x float> %1448, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1915 = insertvalue [2 x <8 x float>] poison, <8 x float> %1914, 0, !dbg !17
  %1916 = shufflevector <4 x float> %1454, <4 x float> %1457, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1917 = insertvalue [2 x <8 x float>] %1915, <8 x float> %1916, 1, !dbg !17
  %1918 = shufflevector <4 x float> %1673, <4 x float> %1676, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1919 = insertvalue [2 x <8 x float>] poison, <8 x float> %1918, 0, !dbg !17
  %1920 = shufflevector <4 x float> %1682, <4 x float> %1685, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1921 = insertvalue [2 x <8 x float>] %1919, <8 x float> %1920, 1, !dbg !17
  %1922 = shufflevector <4 x float> %1891, <4 x float> %1894, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1923 = insertvalue [2 x <8 x float>] poison, <8 x float> %1922, 0, !dbg !17
  %1924 = shufflevector <4 x float> %1900, <4 x float> %1903, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1925 = insertvalue [2 x <8 x float>] %1923, <8 x float> %1924, 1, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1926 = add nuw nsw i64 %82, 1, !dbg !17
  %1927 = icmp slt i64 %1926, %44, !dbg !17
  br i1 %1927, label %75, label %._crit_edge, !dbg !17

._crit_edge:                                      ; preds = %1883, %4
  %.lcssa7 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1907, %1883 ], !dbg !17
  %.lcssa6 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1911, %1883 ], !dbg !17
  %.lcssa5 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1917, %1883 ], !dbg !17
  %.lcssa4 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1921, %1883 ], !dbg !17
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1925, %1883 ], !dbg !17
  %1928 = extractvalue [2 x <1 x float>] %.lcssa6, 0, !dbg !34
  %1929 = shufflevector <1 x float> %1928, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1930 = extractvalue [2 x <1 x float>] %.lcssa6, 1, !dbg !34
  %1931 = shufflevector <1 x float> %1930, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1932 = extractvalue [2 x <8 x float>] %.lcssa7, 1, !dbg !34
  %1933 = shufflevector <8 x float> %1932, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1931, !dbg !34
  %normalized = fmul <4 x float> %1933, %row.reciprocal, !dbg !34
  %1934 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !35
  %1935 = extractelement <4 x half> %1934, i64 3, !dbg !35
  %1936 = extractelement <4 x half> %1934, i64 2, !dbg !35
  %1937 = extractelement <4 x half> %1934, i64 1, !dbg !35
  %1938 = extractelement <4 x half> %1934, i64 0, !dbg !35
  %1939 = shufflevector <8 x float> %1932, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized68 = fmul <4 x float> %1939, %row.reciprocal, !dbg !34
  %1940 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !35
  %1941 = extractelement <4 x half> %1940, i64 3, !dbg !35
  %1942 = extractelement <4 x half> %1940, i64 2, !dbg !35
  %1943 = extractelement <4 x half> %1940, i64 1, !dbg !35
  %1944 = extractelement <4 x half> %1940, i64 0, !dbg !35
  %1945 = extractvalue [2 x <8 x float>] %.lcssa7, 0, !dbg !34
  %1946 = shufflevector <8 x float> %1945, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1929, !dbg !34
  %normalized70 = fmul <4 x float> %1946, %row.reciprocal69, !dbg !34
  %1947 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !35
  %1948 = extractelement <4 x half> %1947, i64 3, !dbg !35
  %1949 = extractelement <4 x half> %1947, i64 2, !dbg !35
  %1950 = extractelement <4 x half> %1947, i64 1, !dbg !35
  %1951 = extractelement <4 x half> %1947, i64 0, !dbg !35
  %1952 = shufflevector <8 x float> %1945, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized71 = fmul <4 x float> %1952, %row.reciprocal69, !dbg !34
  %1953 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !35
  %1954 = extractelement <4 x half> %1953, i64 3, !dbg !35
  %1955 = extractelement <4 x half> %1953, i64 2, !dbg !35
  %1956 = extractelement <4 x half> %1953, i64 1, !dbg !35
  %1957 = extractelement <4 x half> %1953, i64 0, !dbg !35
  %address.thread1072 = tail call i32 asm sideeffect "", "=v,0"(i32 %33) #8, !dbg !36, !range !11, !frisk.register_lifetimes !5
  %address.local1073 = zext nneg i32 %address.thread1072 to i64, !dbg !10
  %address.local1075 = lshr i64 %address.local1073, 2, !dbg !18
  %address.local1076 = and i64 %address.local1075, 48, !dbg !18
  %address.local1078 = and i64 %address.local1073, 240, !dbg !36
  %address.local1077 = add i64 %28, %address.local1073, !dbg !36
  %address.local1080 = sub i64 %address.local1077, %address.local1078, !dbg !36
  %address.local1081 = add i64 %address.local1080, %address.local1076, !dbg !36
  %1958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %30, !dbg !36
  %address.local1084.idx = shl nuw nsw i64 %address.local1081, 8, !dbg !36
  %address.local1084 = getelementptr inbounds nuw i8, ptr addrspace(1) %1958, i64 %address.local1084.idx, !dbg !36
  %address.local1087 = and i64 %address.local1075, 12, !dbg !10
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1087, !dbg !36
  store half %1957, ptr addrspace(1) %address.local1088, align 2, !dbg !36
  %address.local1090 = or disjoint i64 %address.local1087, 1, !dbg !18
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1090, !dbg !36
  store half %1956, ptr addrspace(1) %address.local1091, align 2, !dbg !36
  %address.local1093 = or disjoint i64 %address.local1087, 2, !dbg !18
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1093, !dbg !36
  store half %1955, ptr addrspace(1) %address.local1094, align 2, !dbg !36
  %address.local1096 = or disjoint i64 %address.local1087, 3, !dbg !18
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1096, !dbg !36
  store half %1954, ptr addrspace(1) %address.local1097, align 2, !dbg !36
  %address.local1099 = or disjoint i64 %address.local1087, 16, !dbg !18
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1099, !dbg !36
  store half %1951, ptr addrspace(1) %address.local1100, align 2, !dbg !36
  %address.local1102 = or disjoint i64 %address.local1087, 17, !dbg !18
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1102, !dbg !36
  store half %1950, ptr addrspace(1) %address.local1103, align 2, !dbg !36
  %address.local1105 = or disjoint i64 %address.local1087, 18, !dbg !18
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1105, !dbg !36
  store half %1949, ptr addrspace(1) %address.local1106, align 2, !dbg !36
  %address.local1108 = or disjoint i64 %address.local1087, 19, !dbg !18
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1108, !dbg !36
  store half %1948, ptr addrspace(1) %address.local1109, align 2, !dbg !36
  %address.local1113 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1084, i64 16384, !dbg !36
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1087, !dbg !36
  store half %1944, ptr addrspace(1) %address.local1114, align 2, !dbg !36
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1090, !dbg !36
  store half %1943, ptr addrspace(1) %address.local1116, align 2, !dbg !36
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1093, !dbg !36
  store half %1942, ptr addrspace(1) %address.local1118, align 2, !dbg !36
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1096, !dbg !36
  store half %1941, ptr addrspace(1) %address.local1120, align 2, !dbg !36
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1099, !dbg !36
  store half %1938, ptr addrspace(1) %address.local1122, align 2, !dbg !36
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1102, !dbg !36
  store half %1937, ptr addrspace(1) %address.local1124, align 2, !dbg !36
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1105, !dbg !36
  store half %1936, ptr addrspace(1) %address.local1126, align 2, !dbg !36
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1108, !dbg !36
  store half %1935, ptr addrspace(1) %address.local1128, align 2, !dbg !36
  %1959 = extractvalue [2 x <8 x float>] %.lcssa5, 1, !dbg !34
  %1960 = shufflevector <8 x float> %1959, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized73 = fmul <4 x float> %row.reciprocal, %1960, !dbg !34
  %1961 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !35
  %1962 = extractelement <4 x half> %1961, i64 3, !dbg !35
  %1963 = extractelement <4 x half> %1961, i64 2, !dbg !35
  %1964 = extractelement <4 x half> %1961, i64 1, !dbg !35
  %1965 = extractelement <4 x half> %1961, i64 0, !dbg !35
  %1966 = shufflevector <8 x float> %1959, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized74 = fmul <4 x float> %row.reciprocal, %1966, !dbg !34
  %1967 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !35
  %1968 = extractelement <4 x half> %1967, i64 3, !dbg !35
  %1969 = extractelement <4 x half> %1967, i64 2, !dbg !35
  %1970 = extractelement <4 x half> %1967, i64 1, !dbg !35
  %1971 = extractelement <4 x half> %1967, i64 0, !dbg !35
  %1972 = extractvalue [2 x <8 x float>] %.lcssa5, 0, !dbg !34
  %1973 = shufflevector <8 x float> %1972, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized76 = fmul <4 x float> %row.reciprocal69, %1973, !dbg !34
  %1974 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !35
  %1975 = extractelement <4 x half> %1974, i64 3, !dbg !35
  %1976 = extractelement <4 x half> %1974, i64 2, !dbg !35
  %1977 = extractelement <4 x half> %1974, i64 1, !dbg !35
  %1978 = extractelement <4 x half> %1974, i64 0, !dbg !35
  %1979 = shufflevector <8 x float> %1972, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized77 = fmul <4 x float> %row.reciprocal69, %1979, !dbg !34
  %1980 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !35
  %1981 = extractelement <4 x half> %1980, i64 3, !dbg !35
  %1982 = extractelement <4 x half> %1980, i64 2, !dbg !35
  %1983 = extractelement <4 x half> %1980, i64 1, !dbg !35
  %1984 = extractelement <4 x half> %1980, i64 0, !dbg !35
  %address.local1145 = or disjoint i64 %address.local1087, 32, !dbg !18
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1145, !dbg !36
  store half %1984, ptr addrspace(1) %address.local1146, align 2, !dbg !36
  %address.local1148 = or disjoint i64 %address.local1087, 33, !dbg !18
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1148, !dbg !36
  store half %1983, ptr addrspace(1) %address.local1149, align 2, !dbg !36
  %address.local1151 = or disjoint i64 %address.local1087, 34, !dbg !18
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1151, !dbg !36
  store half %1982, ptr addrspace(1) %address.local1152, align 2, !dbg !36
  %address.local1154 = or disjoint i64 %address.local1087, 35, !dbg !18
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1154, !dbg !36
  store half %1981, ptr addrspace(1) %address.local1155, align 2, !dbg !36
  %address.local1157 = or disjoint i64 %address.local1087, 48, !dbg !18
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1157, !dbg !36
  store half %1978, ptr addrspace(1) %address.local1158, align 2, !dbg !36
  %address.local1160 = or disjoint i64 %address.local1087, 49, !dbg !18
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1160, !dbg !36
  store half %1977, ptr addrspace(1) %address.local1161, align 2, !dbg !36
  %address.local1163 = or disjoint i64 %address.local1087, 50, !dbg !18
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1163, !dbg !36
  store half %1976, ptr addrspace(1) %address.local1164, align 2, !dbg !36
  %address.local1166 = or i64 %address.local1075, 51, !dbg !18
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1166, !dbg !36
  store half %1975, ptr addrspace(1) %address.local1167, align 2, !dbg !36
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1145, !dbg !36
  store half %1971, ptr addrspace(1) %address.local1172, align 2, !dbg !36
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1148, !dbg !36
  store half %1970, ptr addrspace(1) %address.local1174, align 2, !dbg !36
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1151, !dbg !36
  store half %1969, ptr addrspace(1) %address.local1176, align 2, !dbg !36
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1154, !dbg !36
  store half %1968, ptr addrspace(1) %address.local1178, align 2, !dbg !36
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1157, !dbg !36
  store half %1965, ptr addrspace(1) %address.local1180, align 2, !dbg !36
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1160, !dbg !36
  store half %1964, ptr addrspace(1) %address.local1182, align 2, !dbg !36
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1163, !dbg !36
  store half %1963, ptr addrspace(1) %address.local1184, align 2, !dbg !36
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1166, !dbg !36
  store half %1962, ptr addrspace(1) %address.local1186, align 2, !dbg !36
  %1985 = extractvalue [2 x <8 x float>] %.lcssa4, 1, !dbg !34
  %1986 = shufflevector <8 x float> %1985, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized79 = fmul <4 x float> %row.reciprocal, %1986, !dbg !34
  %1987 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !35
  %1988 = extractelement <4 x half> %1987, i64 3, !dbg !35
  %1989 = extractelement <4 x half> %1987, i64 2, !dbg !35
  %1990 = extractelement <4 x half> %1987, i64 1, !dbg !35
  %1991 = extractelement <4 x half> %1987, i64 0, !dbg !35
  %1992 = shufflevector <8 x float> %1985, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized80 = fmul <4 x float> %row.reciprocal, %1992, !dbg !34
  %1993 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !35
  %1994 = extractelement <4 x half> %1993, i64 3, !dbg !35
  %1995 = extractelement <4 x half> %1993, i64 2, !dbg !35
  %1996 = extractelement <4 x half> %1993, i64 1, !dbg !35
  %1997 = extractelement <4 x half> %1993, i64 0, !dbg !35
  %1998 = extractvalue [2 x <8 x float>] %.lcssa4, 0, !dbg !34
  %1999 = shufflevector <8 x float> %1998, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized82 = fmul <4 x float> %row.reciprocal69, %1999, !dbg !34
  %2000 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !35
  %2001 = extractelement <4 x half> %2000, i64 3, !dbg !35
  %2002 = extractelement <4 x half> %2000, i64 2, !dbg !35
  %2003 = extractelement <4 x half> %2000, i64 1, !dbg !35
  %2004 = extractelement <4 x half> %2000, i64 0, !dbg !35
  %2005 = shufflevector <8 x float> %1998, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized83 = fmul <4 x float> %row.reciprocal69, %2005, !dbg !34
  %2006 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !35
  %2007 = extractelement <4 x half> %2006, i64 3, !dbg !35
  %2008 = extractelement <4 x half> %2006, i64 2, !dbg !35
  %2009 = extractelement <4 x half> %2006, i64 1, !dbg !35
  %2010 = extractelement <4 x half> %2006, i64 0, !dbg !35
  %address.local1203 = or disjoint i64 %address.local1087, 64, !dbg !18
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1203, !dbg !36
  store half %2010, ptr addrspace(1) %address.local1204, align 2, !dbg !36
  %address.local1206 = or disjoint i64 %address.local1087, 65, !dbg !18
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1206, !dbg !36
  store half %2009, ptr addrspace(1) %address.local1207, align 2, !dbg !36
  %address.local1209 = or disjoint i64 %address.local1087, 66, !dbg !18
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1209, !dbg !36
  store half %2008, ptr addrspace(1) %address.local1210, align 2, !dbg !36
  %address.local1212 = or disjoint i64 %address.local1087, 67, !dbg !18
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1212, !dbg !36
  store half %2007, ptr addrspace(1) %address.local1213, align 2, !dbg !36
  %address.local1215 = or disjoint i64 %address.local1087, 80, !dbg !18
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1215, !dbg !36
  store half %2004, ptr addrspace(1) %address.local1216, align 2, !dbg !36
  %address.local1218 = or disjoint i64 %address.local1087, 81, !dbg !18
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1218, !dbg !36
  store half %2003, ptr addrspace(1) %address.local1219, align 2, !dbg !36
  %address.local1221 = or disjoint i64 %address.local1087, 82, !dbg !18
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1221, !dbg !36
  store half %2002, ptr addrspace(1) %address.local1222, align 2, !dbg !36
  %address.local1224 = or disjoint i64 %address.local1087, 83, !dbg !18
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1224, !dbg !36
  store half %2001, ptr addrspace(1) %address.local1225, align 2, !dbg !36
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1203, !dbg !36
  store half %1997, ptr addrspace(1) %address.local1230, align 2, !dbg !36
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1206, !dbg !36
  store half %1996, ptr addrspace(1) %address.local1232, align 2, !dbg !36
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1209, !dbg !36
  store half %1995, ptr addrspace(1) %address.local1234, align 2, !dbg !36
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1212, !dbg !36
  store half %1994, ptr addrspace(1) %address.local1236, align 2, !dbg !36
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1215, !dbg !36
  store half %1991, ptr addrspace(1) %address.local1238, align 2, !dbg !36
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1218, !dbg !36
  store half %1990, ptr addrspace(1) %address.local1240, align 2, !dbg !36
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1221, !dbg !36
  store half %1989, ptr addrspace(1) %address.local1242, align 2, !dbg !36
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1224, !dbg !36
  store half %1988, ptr addrspace(1) %address.local1244, align 2, !dbg !36
  %2011 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !34
  %2012 = shufflevector <8 x float> %2011, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized85 = fmul <4 x float> %row.reciprocal, %2012, !dbg !34
  %2013 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !35
  %2014 = extractelement <4 x half> %2013, i64 3, !dbg !35
  %2015 = extractelement <4 x half> %2013, i64 2, !dbg !35
  %2016 = extractelement <4 x half> %2013, i64 1, !dbg !35
  %2017 = extractelement <4 x half> %2013, i64 0, !dbg !35
  %2018 = shufflevector <8 x float> %2011, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized86 = fmul <4 x float> %row.reciprocal, %2018, !dbg !34
  %2019 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !35
  %2020 = extractelement <4 x half> %2019, i64 3, !dbg !35
  %2021 = extractelement <4 x half> %2019, i64 2, !dbg !35
  %2022 = extractelement <4 x half> %2019, i64 1, !dbg !35
  %2023 = extractelement <4 x half> %2019, i64 0, !dbg !35
  %2024 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !34
  %2025 = shufflevector <8 x float> %2024, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %normalized88 = fmul <4 x float> %row.reciprocal69, %2025, !dbg !34
  %2026 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !35
  %2027 = extractelement <4 x half> %2026, i64 3, !dbg !35
  %2028 = extractelement <4 x half> %2026, i64 2, !dbg !35
  %2029 = extractelement <4 x half> %2026, i64 1, !dbg !35
  %2030 = extractelement <4 x half> %2026, i64 0, !dbg !35
  %2031 = shufflevector <8 x float> %2024, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %normalized89 = fmul <4 x float> %row.reciprocal69, %2031, !dbg !34
  %2032 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !35
  %2033 = extractelement <4 x half> %2032, i64 3, !dbg !35
  %2034 = extractelement <4 x half> %2032, i64 2, !dbg !35
  %2035 = extractelement <4 x half> %2032, i64 1, !dbg !35
  %2036 = extractelement <4 x half> %2032, i64 0, !dbg !35
  %address.local1261 = or disjoint i64 %address.local1087, 96, !dbg !18
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1261, !dbg !36
  store half %2036, ptr addrspace(1) %address.local1262, align 2, !dbg !36
  %address.local1264 = or disjoint i64 %address.local1087, 97, !dbg !18
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1264, !dbg !36
  store half %2035, ptr addrspace(1) %address.local1265, align 2, !dbg !36
  %address.local1267 = or disjoint i64 %address.local1087, 98, !dbg !18
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1267, !dbg !36
  store half %2034, ptr addrspace(1) %address.local1268, align 2, !dbg !36
  %address.local1270 = or disjoint i64 %address.local1087, 99, !dbg !18
  %address.local1271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1270, !dbg !36
  store half %2033, ptr addrspace(1) %address.local1271, align 2, !dbg !36
  %address.local1273 = or disjoint i64 %address.local1087, 112, !dbg !18
  %address.local1274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1273, !dbg !36
  store half %2030, ptr addrspace(1) %address.local1274, align 2, !dbg !36
  %address.local1276 = or disjoint i64 %address.local1087, 113, !dbg !18
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1276, !dbg !36
  store half %2029, ptr addrspace(1) %address.local1277, align 2, !dbg !36
  %address.local1279 = or disjoint i64 %address.local1087, 114, !dbg !18
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1279, !dbg !36
  store half %2028, ptr addrspace(1) %address.local1280, align 2, !dbg !36
  %address.local1282 = or i64 %address.local1075, 115, !dbg !18
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1282, !dbg !36
  store half %2027, ptr addrspace(1) %address.local1283, align 2, !dbg !36
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1261, !dbg !36
  store half %2023, ptr addrspace(1) %address.local1288, align 2, !dbg !36
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1264, !dbg !36
  store half %2022, ptr addrspace(1) %address.local1290, align 2, !dbg !36
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1267, !dbg !36
  store half %2021, ptr addrspace(1) %address.local1292, align 2, !dbg !36
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1270, !dbg !36
  store half %2020, ptr addrspace(1) %address.local1294, align 2, !dbg !36
  %address.local1296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1273, !dbg !36
  store half %2017, ptr addrspace(1) %address.local1296, align 2, !dbg !36
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1276, !dbg !36
  store half %2016, ptr addrspace(1) %address.local1298, align 2, !dbg !36
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1279, !dbg !36
  store half %2015, ptr addrspace(1) %address.local1300, align 2, !dbg !36
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1282, !dbg !36
  store half %2014, ptr addrspace(1) %address.local1302, align 2, !dbg !36
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
