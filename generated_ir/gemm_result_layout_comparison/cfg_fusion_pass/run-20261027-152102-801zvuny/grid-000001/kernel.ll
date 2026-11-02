; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-152102-801zvuny/grid-000001/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-152102-801zvuny/grid-000001/input.mlir"

@shm_0 = addrspace(3) global [65536 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !38 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 5, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423484, !dbg !5
  %13 = and i64 %6, 3, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = lshr i64 %10, 2, !dbg !5
  %16 = sub nsw i64 31, %14, !dbg !5
  %17 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %18 = sext i32 %17 to i64, !dbg !5
  %19 = shl i64 %16, 7, !dbg !9
  %20 = icmp ugt i64 %10, 1023, !dbg !11
  %21 = add nsw i64 %14, -32, !dbg !11
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !11
  %23 = sdiv i64 %22, 32, !dbg !11
  %24 = sext i1 %20 to i64, !dbg !11
  %25 = xor i64 %23, %24, !dbg !11
  %26 = add nsw i64 %15, %25, !dbg !11
  %27 = and i64 %26, 31, !dbg !11
  %28 = shl nsw i64 %16, 7, !dbg !11
  %29 = mul nsw i64 %25, -4096, !dbg !11
  %30 = add i64 %28, %29, !dbg !11
  %31 = shl nuw nsw i64 %27, 19, !dbg !11
  %32 = shl nsw i64 %30, 7, !dbg !11
  %33 = add i64 %31, %32, !dbg !11
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !40
  %35 = zext nneg i32 %34 to i64, !dbg !12
  %36 = lshr i64 %35, 4, !dbg !12
  %37 = and i64 %36, 3, !dbg !12
  %38 = shl nuw nsw i64 %37, 2, !dbg !12
  %39 = and i64 %35, 240, !dbg !14
  %40 = or disjoint i64 %19, 127, !dbg !14
  %41 = add i64 %19, 128, !dbg !14
  %42 = icmp uge i64 %41, %40, !dbg !14
  %43 = icmp eq i64 %37, 0, !dbg !15
  %44 = and i32 %34, 16, !dbg !16
  %.not = icmp eq i32 %44, 0, !dbg !16
  %45 = and i32 %34, 32, !dbg !16
  %.not1 = icmp eq i32 %45, 0, !dbg !16
  %46 = sub nsw i64 32, %14, !dbg !17
  %47 = shl i64 %15, 7, !dbg !18
  %48 = and i64 %47, 3968, !dbg !18
  %49 = shl nsw i64 %18, 12, !dbg !18
  %50 = or disjoint i64 %48, %49, !dbg !18
  %51 = lshr i64 %35, 2, !dbg !13
  %52 = and i64 %51, 48, !dbg !13
  br label %53, !dbg !7

53:                                               ; preds = %2171, %4
  %54 = phi i64 [ %2230, %2171 ], [ 0, %4 ], !dbg !7
  %55 = phi [2 x <8 x float>] [ %2211, %2171 ], [ zeroinitializer, %4 ], !dbg !7
  %56 = phi [2 x <1 x float>] [ %2215, %2171 ], [ zeroinitializer, %4 ], !dbg !7
  %57 = phi [2 x <1 x float>] [ %2217, %2171 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %58 = phi [2 x <8 x float>] [ %2221, %2171 ], [ zeroinitializer, %4 ], !dbg !7
  %59 = phi [2 x <8 x float>] [ %2225, %2171 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi [2 x <8 x float>] [ %2229, %2171 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = icmp slt i64 %54, %46, !dbg !7
  br i1 %61, label %62, label %2231, !dbg !7

62:                                               ; preds = %53
  %63 = shl nsw i64 %54, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local258 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local259 = and i64 %address.local258, 1536, !dbg !11
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local259, !dbg !11
  %address.local261 = ptrtoint ptr addrspace(3) %address.local260 to i64, !dbg !11
  %address.local262 = trunc i64 %address.local261 to i32, !dbg !11
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %address.local263 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local264 = and i64 %address.local263, 16256, !dbg !11
  %address.local265 = lshr i64 %address.local, 4, !dbg !11
  %address.local266 = and i64 %address.local265, 8, !dbg !11
  %address.local267 = or disjoint i64 %address.local264, %address.local266, !dbg !11
  %65 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %.idx1824 = or disjoint i64 %33, %address.local267, !dbg !11
  %.idx1824.tr = trunc i64 %.idx1824 to i32, !dbg !11
  %dma.buffer.offset1821 = shl i32 %.idx1824.tr, 1, !dbg !11
  %66 = insertelement <2 x i64> poison, i64 %65, i64 0, !dbg !11
  %67 = bitcast <2 x i64> %66 to <4 x i32>, !dbg !11
  %dma.buffer.rsrc1822 = shufflevector <4 x i32> %67, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !11
  %dma.buffer.lds1823 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64), !dbg !11
  %68 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1821, <4 x i32> %dma.buffer.rsrc1822) #10, !dbg !11
  %address.local269 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local270 = shl nuw nsw i64 %address.local269, 3, !dbg !11
  %address.local271 = and i64 %address.local270, 3584, !dbg !11
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local271, !dbg !11
  %address.local273 = ptrtoint ptr addrspace(3) %address.local272 to i64, !dbg !11
  %address.local274 = trunc i64 %address.local273 to i32, !dbg !11
  %69 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %address.local275 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local276 = and i64 %address.local275, 16256, !dbg !11
  %address.local277 = lshr i64 %address.local269, 4, !dbg !11
  %address.local278 = and i64 %address.local277, 24, !dbg !11
  %address.local279 = or disjoint i64 %address.local276, %address.local278, !dbg !11
  %70 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %.idx1825 = or disjoint i64 %33, %address.local279, !dbg !11
  %.idx1825.tr = trunc i64 %.idx1825 to i32, !dbg !11
  %dma.buffer.offset1818 = shl i32 %.idx1825.tr, 1, !dbg !11
  %71 = insertelement <2 x i64> poison, i64 %70, i64 0, !dbg !11
  %72 = bitcast <2 x i64> %71 to <4 x i32>, !dbg !11
  %dma.buffer.rsrc1819 = shufflevector <4 x i32> %72, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !11
  %dma.buffer.lds1820 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %69), !dbg !11
  %73 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1818, <4 x i32> %dma.buffer.rsrc1819) #10, !dbg !11
  %74 = lshr i64 %54, 12, !dbg !18
  %75 = add nuw nsw i64 %74, %15, !dbg !18
  %76 = lshr i64 %54, 5, !dbg !18
  %77 = and i64 %76, 127, !dbg !18
  %78 = shl i64 %75, 19, !dbg !18
  %79 = and i64 %78, 16252928, !dbg !18
  %80 = shl i64 %76, 19, !dbg !18
  %81 = shl i64 %54, 14, !dbg !18
  %82 = sub i64 %81, %80, !dbg !18
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local259, !dbg !18
  %address.local282 = ptrtoint ptr addrspace(3) %address.local281 to i64, !dbg !18
  %address.local283 = trunc i64 %address.local282 to i32, !dbg !18
  %83 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %address.local284 = lshr i64 %address.local, 4, !dbg !18
  %address.local285 = and i64 %address.local284, 8, !dbg !18
  %address.local286 = and i64 %address.local, 123, !dbg !18
  %address.local287 = lshr i64 %address.local, 2, !dbg !18
  %address.local288 = and i64 %address.local287, 4, !dbg !18
  %address.local289 = add nuw nsw i64 %address.local288, %address.local, !dbg !18
  %address.local290 = and i64 %address.local289, 4, !dbg !18
  %address.local291 = or disjoint i64 %address.local286, %address.local290, !dbg !18
  %address.local292 = shl nuw nsw i64 %address.local291, 7, !dbg !18
  %address.local293 = or disjoint i64 %address.local285, %address.local292, !dbg !18
  %84 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %.idx1828 = or disjoint i64 %79, %77, !dbg !18
  %85 = add i64 %.idx1828, %82, !dbg !18
  %86 = add i64 %85, %address.local293, !dbg !18
  %.tr = trunc i64 %86 to i32, !dbg !18
  %dma.buffer.offset1815 = shl i32 %.tr, 1, !dbg !18
  %87 = insertelement <2 x i64> poison, i64 %84, i64 0, !dbg !18
  %88 = bitcast <2 x i64> %87 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1816 = shufflevector <4 x i32> %88, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1817 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !18
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1815, <4 x i32> %dma.buffer.rsrc1816) #10, !dbg !18
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local271, !dbg !18
  %address.local296 = ptrtoint ptr addrspace(3) %address.local295 to i64, !dbg !18
  %address.local297 = trunc i64 %address.local296 to i32, !dbg !18
  %90 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %address.local298 = lshr i64 %address.local269, 4, !dbg !18
  %address.local299 = and i64 %address.local298, 24, !dbg !18
  %address.local300 = and i64 %address.local, 123, !dbg !18
  %address.local301 = lshr i64 %address.local, 2, !dbg !18
  %address.local302 = and i64 %address.local301, 4, !dbg !18
  %address.local303 = add nuw nsw i64 %address.local302, %address.local, !dbg !18
  %address.local304 = and i64 %address.local303, 4, !dbg !18
  %address.local305 = or disjoint i64 %address.local300, %address.local304, !dbg !18
  %address.local306 = shl nuw nsw i64 %address.local305, 7, !dbg !18
  %address.local307 = or disjoint i64 %address.local299, %address.local306, !dbg !18
  %91 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %.idx1831 = or disjoint i64 %79, %77, !dbg !18
  %92 = add i64 %.idx1831, %82, !dbg !18
  %93 = add i64 %92, %address.local307, !dbg !18
  %.tr1832 = trunc i64 %93 to i32, !dbg !18
  %dma.buffer.offset1812 = shl i32 %.tr1832, 1, !dbg !18
  %94 = insertelement <2 x i64> poison, i64 %91, i64 0, !dbg !18
  %95 = bitcast <2 x i64> %94 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1813 = shufflevector <4 x i32> %95, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1814 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %90), !dbg !18
  %96 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1812, <4 x i32> %dma.buffer.rsrc1813) #10, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local309 = lshr i32 %address.thread, 2, !dbg !12
  %address.local310 = and i32 %address.local309, 4, !dbg !12
  %address.local311 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local312 = and i32 %address.local311, 384, !dbg !12
  %address.local313 = or disjoint i32 %address.local310, %address.local312, !dbg !12
  %address.local314 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local315 = and i32 %address.local314, 120, !dbg !12
  %address.local316 = or disjoint i32 %address.local313, %address.local315, !dbg !12
  %address.local317 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local318 = and i32 %address.local317, 1024, !dbg !12
  %address.local319 = or disjoint i32 %address.local316, %address.local318, !dbg !12
  %address.local320 = zext nneg i32 %address.local319 to i64, !dbg !12
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local320, !dbg !12
  %97 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !12
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !12
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !12
  %102 = fmul <4 x half> %98, splat (half 0xH3015), !dbg !12
  %103 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !12
  %104 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.local328 = lshr i32 %address.thread, 2, !dbg !19
  %address.local329 = and i32 %address.local328, 4, !dbg !19
  %address.local330 = shl nuw nsw i32 %address.thread, 3, !dbg !19
  %address.local331 = and i32 %address.local330, 120, !dbg !19
  %address.local332 = or disjoint i32 %address.local329, %address.local331, !dbg !19
  %address.local333 = shl nuw nsw i32 %address.thread, 5, !dbg !19
  %address.local334 = and i32 %address.local333, 1024, !dbg !19
  %address.local335 = or disjoint i32 %address.local332, %address.local334, !dbg !19
  %address.local336 = zext nneg i32 %address.local335 to i64, !dbg !19
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local336, !dbg !19
  %105 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %105, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1561 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %107 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !19
  %address.local349 = lshr i32 %address.thread, 2, !dbg !19
  %address.local350 = and i32 %address.local349, 4, !dbg !19
  %address.local351 = shl nuw nsw i32 %address.thread, 3, !dbg !19
  %address.local352 = and i32 %address.local351, 88, !dbg !19
  %address.local353 = or disjoint i32 %address.local350, %address.local352, !dbg !19
  %address.local354 = shl nuw nsw i32 %address.thread, 3, !dbg !19
  %address.local355 = and i32 %address.local354, 32, !dbg !19
  %address.local356 = sub nsw i32 %address.local353, %address.local355, !dbg !19
  %address.local357 = shl nuw nsw i32 %address.thread, 5, !dbg !19
  %address.local358 = and i32 %address.local357, 1024, !dbg !19
  %address.local359 = add nsw i32 %address.local356, %address.local358, !dbg !19
  %address.local360 = sext i32 %address.local359 to i64, !dbg !19
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local360, !dbg !19
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local361, i64 320, !dbg !19
  %108 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %107, <4 x float> %106) #9, !dbg !19
  %110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %108, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1562 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %111 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !19
  %lds.const.gep1563 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %112 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !19
  %113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %111, <4 x float> %110) #9, !dbg !19
  %114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %112, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1564 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %115 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !19
  %lds.const.gep1565 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %116 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !19
  %117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %115, <4 x float> %114) #9, !dbg !19
  %118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %116, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1566 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %119 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !19
  %lds.const.gep1567 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %120 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !19
  %121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %119, <4 x float> %118) #9, !dbg !19
  %122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %120, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1568 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %123 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !19
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %124 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !19
  %125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %123, <4 x float> %122) #9, !dbg !19
  %126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %124, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %127 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !19
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %128 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !19
  %129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %127, <4 x float> %126) #9, !dbg !19
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %128, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %131 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !19
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %132 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !19
  %133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %131, <4 x float> %130) #9, !dbg !19
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %132, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !19
  %136 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %135, <4 x float> %134) #9, !dbg !19
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %136, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1575 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %139 = load <4 x half>, ptr addrspace(3) %lds.const.gep1575, align 8, !dbg !19
  %140 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %139, <4 x float> %138) #9, !dbg !19
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %140, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1576 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %143 = load <4 x half>, ptr addrspace(3) %lds.const.gep1576, align 8, !dbg !19
  %lds.const.gep1577 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep1577, align 8, !dbg !19
  %145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %143, <4 x float> %142) #9, !dbg !19
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %144, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1578 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %147 = load <4 x half>, ptr addrspace(3) %lds.const.gep1578, align 8, !dbg !19
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !19
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %147, <4 x float> %146) #9, !dbg !19
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %148, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %151 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !19
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %152 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !19
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %151, <4 x float> %150) #9, !dbg !19
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %152, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !19
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %156 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !19
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %155, <4 x float> %154) #9, !dbg !19
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %156, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %159 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !19
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %160 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !19
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %159, <4 x float> %158) #9, !dbg !19
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %160, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %163 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !19
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %164 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !19
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %163, <4 x float> %162) #9, !dbg !19
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %164, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %167 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !19
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %167, <4 x float> %166) #9, !dbg !19
  %169 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %170 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %171 = or disjoint i64 %33, 32, !dbg !11
  %172 = or disjoint i64 %address.local267, %171, !dbg !11
  %.tr1833 = trunc i64 %172 to i32, !dbg !11
  %dma.buffer.offset1809 = shl i32 %.tr1833, 1, !dbg !11
  %173 = insertelement <2 x i64> poison, i64 %170, i64 0, !dbg !11
  %174 = bitcast <2 x i64> %173 to <4 x i32>, !dbg !11
  %dma.buffer.rsrc1810 = shufflevector <4 x i32> %174, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !11
  %dma.buffer.lds1811 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %169), !dbg !11
  %175 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1811, i32 %dma.buffer.offset1809, <4 x i32> %dma.buffer.rsrc1810) #10, !dbg !11
  %176 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %177 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %178 = or disjoint i64 %33, 32, !dbg !11
  %179 = or disjoint i64 %address.local279, %178, !dbg !11
  %.tr1834 = trunc i64 %179 to i32, !dbg !11
  %dma.buffer.offset1806 = shl i32 %.tr1834, 1, !dbg !11
  %180 = insertelement <2 x i64> poison, i64 %177, i64 0, !dbg !11
  %181 = bitcast <2 x i64> %180 to <4 x i32>, !dbg !11
  %dma.buffer.rsrc1807 = shufflevector <4 x i32> %181, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !11
  %dma.buffer.lds1808 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %176), !dbg !11
  %182 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1806, <4 x i32> %dma.buffer.rsrc1807) #10, !dbg !11
  %183 = add nuw i64 %54, 1024, !dbg !18
  %184 = icmp slt i64 %183, 0, !dbg !18
  %185 = sub nuw i64 -1025, %54, !dbg !18
  %186 = select i1 %184, i64 %185, i64 %183, !dbg !18
  %187 = sdiv i64 %186, 4096, !dbg !18
  %.lobit = ashr i64 %183, 63, !dbg !18
  %188 = xor i64 %187, %.lobit, !dbg !18
  %189 = add nsw i64 %188, %15, !dbg !18
  %190 = add nsw i64 %76, %50, !dbg !18
  %191 = add nsw i64 %190, 32, !dbg !18
  %192 = icmp slt i64 %190, -32, !dbg !18
  %193 = sub nsw i64 -33, %190, !dbg !18
  %194 = select i1 %192, i64 %193, i64 %191, !dbg !18
  %195 = sdiv i64 %194, 128, !dbg !18
  %196 = sext i1 %192 to i64, !dbg !18
  %197 = xor i64 %195, %196, !dbg !18
  %198 = shl nsw i64 %197, 7, !dbg !18
  %199 = sub i64 %50, %198, !dbg !18
  %200 = shl i64 %189, 19, !dbg !18
  %201 = and i64 %200, 16252928, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %202 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %203 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %.idx1838 = add nuw nsw i64 %201, %76, !dbg !18
  %204 = add i64 %.idx1838, %199, !dbg !18
  %205 = add i64 %204, %82, !dbg !18
  %206 = add i64 %205, %address.local293, !dbg !18
  %.tr1839 = trunc i64 %206 to i32, !dbg !18
  %207 = shl i32 %.tr1839, 1, !dbg !18
  %dma.buffer.offset1803 = add i32 %207, 64, !dbg !18
  %208 = insertelement <2 x i64> poison, i64 %203, i64 0, !dbg !18
  %209 = bitcast <2 x i64> %208 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1804 = shufflevector <4 x i32> %209, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1805 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %202), !dbg !18
  %210 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1803, <4 x i32> %dma.buffer.rsrc1804) #10, !dbg !18
  %211 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %212 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %.idx1843 = add nuw nsw i64 %201, %76, !dbg !18
  %213 = add i64 %.idx1843, %199, !dbg !18
  %214 = add i64 %213, %82, !dbg !18
  %215 = add i64 %214, %address.local307, !dbg !18
  %.tr1844 = trunc i64 %215 to i32, !dbg !18
  %216 = shl i32 %.tr1844, 1, !dbg !18
  %dma.buffer.offset1800 = add i32 %216, 64, !dbg !18
  %217 = insertelement <2 x i64> poison, i64 %212, i64 0, !dbg !18
  %218 = bitcast <2 x i64> %217 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1801 = shufflevector <4 x i32> %218, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1802 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %211), !dbg !18
  %219 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1800, <4 x i32> %dma.buffer.rsrc1801) #10, !dbg !18
  %220 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %221 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !12
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !12
  %223 = fmul <4 x half> %220, splat (half 0xH3015), !dbg !12
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %224 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !12
  %225 = fmul <4 x half> %221, splat (half 0xH3015), !dbg !12
  %226 = fmul <4 x half> %222, splat (half 0xH3015), !dbg !12
  %227 = fmul <4 x half> %224, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %228 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %228, <4 x float> %109) #9, !dbg !19
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !19
  %231 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %230, <4 x float> %229) #9, !dbg !19
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %231, <4 x float> %113) #9, !dbg !19
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !19
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !19
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %234, <4 x float> %233) #9, !dbg !19
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %235, <4 x float> %117) #9, !dbg !19
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %238 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !19
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !19
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %238, <4 x float> %237) #9, !dbg !19
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %239, <4 x float> %121) #9, !dbg !19
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !19
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !19
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %242, <4 x float> %241) #9, !dbg !19
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %243, <4 x float> %125) #9, !dbg !19
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !19
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %247 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !19
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %246, <4 x float> %245) #9, !dbg !19
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %247, <4 x float> %129) #9, !dbg !19
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !19
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !19
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %250, <4 x float> %249) #9, !dbg !19
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %251, <4 x float> %133) #9, !dbg !19
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !19
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !19
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %254, <4 x float> %253) #9, !dbg !19
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %223, <4 x half> %255, <4 x float> %137) #9, !dbg !19
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !19
  %259 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %258, <4 x float> %257) #9, !dbg !19
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %259, <4 x float> %141) #9, !dbg !19
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !19
  %263 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %262, <4 x float> %261) #9, !dbg !19
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %263, <4 x float> %145) #9, !dbg !19
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !19
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !19
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %266, <4 x float> %265) #9, !dbg !19
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %267, <4 x float> %149) #9, !dbg !19
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !19
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !19
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %270, <4 x float> %269) #9, !dbg !19
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %271, <4 x float> %153) #9, !dbg !19
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !19
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !19
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %274, <4 x float> %273) #9, !dbg !19
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %275, <4 x float> %157) #9, !dbg !19
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !19
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !19
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %278, <4 x float> %277) #9, !dbg !19
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %279, <4 x float> %161) #9, !dbg !19
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !19
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !19
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %282, <4 x float> %281) #9, !dbg !19
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %283, <4 x float> %165) #9, !dbg !19
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !19
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !19
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %286, <4 x float> %285) #9, !dbg !19
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %287, <4 x float> %168) #9, !dbg !19
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !19
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %290, <4 x float> %289) #9, !dbg !19
  %292 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %293 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %294 = or disjoint i64 %33, 64, !dbg !11
  %295 = or disjoint i64 %address.local267, %294, !dbg !11
  %.tr1845 = trunc i64 %295 to i32, !dbg !11
  %dma.buffer.offset1797 = shl i32 %.tr1845, 1, !dbg !11
  %296 = insertelement <2 x i64> poison, i64 %293, i64 0, !dbg !11
  %297 = bitcast <2 x i64> %296 to <4 x i32>, !dbg !11
  %dma.buffer.rsrc1798 = shufflevector <4 x i32> %297, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !11
  %dma.buffer.lds1799 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %292), !dbg !11
  %298 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1797, <4 x i32> %dma.buffer.rsrc1798) #10, !dbg !11
  %299 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %300 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %301 = or disjoint i64 %33, 64, !dbg !11
  %302 = or disjoint i64 %address.local279, %301, !dbg !11
  %.tr1846 = trunc i64 %302 to i32, !dbg !11
  %dma.buffer.offset1794 = shl i32 %.tr1846, 1, !dbg !11
  %303 = insertelement <2 x i64> poison, i64 %300, i64 0, !dbg !11
  %304 = bitcast <2 x i64> %303 to <4 x i32>, !dbg !11
  %dma.buffer.rsrc1795 = shufflevector <4 x i32> %304, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !11
  %dma.buffer.lds1796 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %299), !dbg !11
  %305 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1796, i32 %dma.buffer.offset1794, <4 x i32> %dma.buffer.rsrc1795) #10, !dbg !11
  %306 = add nuw i64 %54, 2048, !dbg !18
  %307 = icmp slt i64 %306, 0, !dbg !18
  %308 = sub nuw i64 -2049, %54, !dbg !18
  %309 = select i1 %307, i64 %308, i64 %306, !dbg !18
  %310 = sdiv i64 %309, 4096, !dbg !18
  %.lobit2 = ashr i64 %306, 63, !dbg !18
  %311 = xor i64 %310, %.lobit2, !dbg !18
  %312 = add nsw i64 %311, %15, !dbg !18
  %313 = add nsw i64 %190, 64, !dbg !18
  %314 = icmp slt i64 %190, -64, !dbg !18
  %315 = sub nsw i64 -65, %190, !dbg !18
  %316 = select i1 %314, i64 %315, i64 %313, !dbg !18
  %317 = sdiv i64 %316, 128, !dbg !18
  %318 = sext i1 %314 to i64, !dbg !18
  %319 = xor i64 %317, %318, !dbg !18
  %320 = shl nsw i64 %319, 7, !dbg !18
  %321 = sub i64 %50, %320, !dbg !18
  %322 = shl i64 %312, 19, !dbg !18
  %323 = and i64 %322, 16252928, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %324 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %325 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %.idx1850 = add nuw nsw i64 %323, %76, !dbg !18
  %326 = add i64 %.idx1850, %321, !dbg !18
  %327 = add i64 %326, %82, !dbg !18
  %328 = add i64 %327, %address.local293, !dbg !18
  %.tr1851 = trunc i64 %328 to i32, !dbg !18
  %329 = shl i32 %.tr1851, 1, !dbg !18
  %dma.buffer.offset1791 = add i32 %329, 128, !dbg !18
  %330 = insertelement <2 x i64> poison, i64 %325, i64 0, !dbg !18
  %331 = bitcast <2 x i64> %330 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1792 = shufflevector <4 x i32> %331, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1793 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %324), !dbg !18
  %332 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1793, i32 %dma.buffer.offset1791, <4 x i32> %dma.buffer.rsrc1792) #10, !dbg !18
  %333 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %334 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %.idx1855 = add nuw nsw i64 %323, %76, !dbg !18
  %335 = add i64 %.idx1855, %321, !dbg !18
  %336 = add i64 %335, %82, !dbg !18
  %337 = add i64 %336, %address.local307, !dbg !18
  %.tr1856 = trunc i64 %337 to i32, !dbg !18
  %338 = shl i32 %.tr1856, 1, !dbg !18
  %dma.buffer.offset1788 = add i32 %338, 128, !dbg !18
  %339 = insertelement <2 x i64> poison, i64 %334, i64 0, !dbg !18
  %340 = bitcast <2 x i64> %339 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1789 = shufflevector <4 x i32> %340, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1790 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %333), !dbg !18
  %341 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1790, i32 %dma.buffer.offset1788, <4 x i32> %dma.buffer.rsrc1789) #10, !dbg !18
  %342 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %343 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !12
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !12
  %345 = fmul <4 x half> %342, splat (half 0xH3015), !dbg !12
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !12
  %347 = fmul <4 x half> %343, splat (half 0xH3015), !dbg !12
  %348 = fmul <4 x half> %344, splat (half 0xH3015), !dbg !12
  %349 = fmul <4 x half> %346, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %350 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %345, <4 x half> %350, <4 x float> %232) #9, !dbg !19
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !19
  %353 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %352, <4 x float> %351) #9, !dbg !19
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %345, <4 x half> %353, <4 x float> %236) #9, !dbg !19
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !19
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !19
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %356, <4 x float> %355) #9, !dbg !19
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %345, <4 x half> %357, <4 x float> %240) #9, !dbg !19
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !19
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !19
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %360, <4 x float> %359) #9, !dbg !19
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %345, <4 x half> %361, <4 x float> %244) #9, !dbg !19
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !19
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !19
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %364, <4 x float> %363) #9, !dbg !19
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %345, <4 x half> %365, <4 x float> %248) #9, !dbg !19
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !19
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !19
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %368, <4 x float> %367) #9, !dbg !19
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %345, <4 x half> %369, <4 x float> %252) #9, !dbg !19
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !19
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !19
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %372, <4 x float> %371) #9, !dbg !19
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %345, <4 x half> %373, <4 x float> %256) #9, !dbg !19
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !19
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %377 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !19
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %376, <4 x float> %375) #9, !dbg !19
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %345, <4 x half> %377, <4 x float> %260) #9, !dbg !19
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %380 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !19
  %381 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %380, <4 x float> %379) #9, !dbg !19
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %381, <4 x float> %264) #9, !dbg !19
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %384 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !19
  %385 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %384, <4 x float> %383) #9, !dbg !19
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %385, <4 x float> %268) #9, !dbg !19
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %388 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !19
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %389 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !19
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %388, <4 x float> %387) #9, !dbg !19
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %389, <4 x float> %272) #9, !dbg !19
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !19
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %393 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !19
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %392, <4 x float> %391) #9, !dbg !19
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %393, <4 x float> %276) #9, !dbg !19
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %396 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !19
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %397 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !19
  %398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %396, <4 x float> %395) #9, !dbg !19
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %397, <4 x float> %280) #9, !dbg !19
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %400 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !19
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %401 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !19
  %402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %400, <4 x float> %399) #9, !dbg !19
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %401, <4 x float> %284) #9, !dbg !19
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %404 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !19
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %405 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !19
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %404, <4 x float> %403) #9, !dbg !19
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %405, <4 x float> %288) #9, !dbg !19
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !19
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %409 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !19
  %410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %408, <4 x float> %407) #9, !dbg !19
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %409, <4 x float> %291) #9, !dbg !19
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %412 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !19
  %413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %412, <4 x float> %411) #9, !dbg !19
  %414 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %415 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %416 = or disjoint i64 %33, 96, !dbg !11
  %417 = or disjoint i64 %address.local267, %416, !dbg !11
  %.tr1857 = trunc i64 %417 to i32, !dbg !11
  %dma.buffer.offset1785 = shl i32 %.tr1857, 1, !dbg !11
  %418 = insertelement <2 x i64> poison, i64 %415, i64 0, !dbg !11
  %419 = bitcast <2 x i64> %418 to <4 x i32>, !dbg !11
  %dma.buffer.rsrc1786 = shufflevector <4 x i32> %419, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !11
  %dma.buffer.lds1787 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %414), !dbg !11
  %420 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1787, i32 %dma.buffer.offset1785, <4 x i32> %dma.buffer.rsrc1786) #10, !dbg !11
  %421 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %422 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %423 = or disjoint i64 %33, 96, !dbg !11
  %424 = or disjoint i64 %address.local279, %423, !dbg !11
  %.tr1858 = trunc i64 %424 to i32, !dbg !11
  %dma.buffer.offset1782 = shl i32 %.tr1858, 1, !dbg !11
  %425 = insertelement <2 x i64> poison, i64 %422, i64 0, !dbg !11
  %426 = bitcast <2 x i64> %425 to <4 x i32>, !dbg !11
  %dma.buffer.rsrc1783 = shufflevector <4 x i32> %426, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !11
  %dma.buffer.lds1784 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %421), !dbg !11
  %427 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1784, i32 %dma.buffer.offset1782, <4 x i32> %dma.buffer.rsrc1783) #10, !dbg !11
  %428 = add nuw i64 %54, 3072, !dbg !18
  %429 = icmp slt i64 %428, 0, !dbg !18
  %430 = sub nuw i64 -3073, %54, !dbg !18
  %431 = select i1 %429, i64 %430, i64 %428, !dbg !18
  %432 = sdiv i64 %431, 4096, !dbg !18
  %.lobit3 = ashr i64 %428, 63, !dbg !18
  %433 = xor i64 %432, %.lobit3, !dbg !18
  %434 = add nsw i64 %433, %15, !dbg !18
  %435 = add nsw i64 %190, 96, !dbg !18
  %436 = icmp slt i64 %190, -96, !dbg !18
  %437 = sub nsw i64 -97, %190, !dbg !18
  %438 = select i1 %436, i64 %437, i64 %435, !dbg !18
  %439 = sdiv i64 %438, 128, !dbg !18
  %440 = sext i1 %436 to i64, !dbg !18
  %441 = xor i64 %439, %440, !dbg !18
  %442 = shl nsw i64 %441, 7, !dbg !18
  %443 = sub i64 %50, %442, !dbg !18
  %444 = shl i64 %434, 19, !dbg !18
  %445 = and i64 %444, 16252928, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %446 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %447 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %.idx1862 = add nuw nsw i64 %445, %76, !dbg !18
  %448 = add i64 %.idx1862, %443, !dbg !18
  %449 = add i64 %448, %82, !dbg !18
  %450 = add i64 %449, %address.local293, !dbg !18
  %.tr1863 = trunc i64 %450 to i32, !dbg !18
  %451 = shl i32 %.tr1863, 1, !dbg !18
  %dma.buffer.offset1779 = add i32 %451, 192, !dbg !18
  %452 = insertelement <2 x i64> poison, i64 %447, i64 0, !dbg !18
  %453 = bitcast <2 x i64> %452 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1780 = shufflevector <4 x i32> %453, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1781 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %446), !dbg !18
  %454 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1781, i32 %dma.buffer.offset1779, <4 x i32> %dma.buffer.rsrc1780) #10, !dbg !18
  %455 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %456 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %.idx1867 = add nuw nsw i64 %445, %76, !dbg !18
  %457 = add i64 %.idx1867, %443, !dbg !18
  %458 = add i64 %457, %82, !dbg !18
  %459 = add i64 %458, %address.local307, !dbg !18
  %.tr1868 = trunc i64 %459 to i32, !dbg !18
  %460 = shl i32 %.tr1868, 1, !dbg !18
  %dma.buffer.offset1776 = add i32 %460, 192, !dbg !18
  %461 = insertelement <2 x i64> poison, i64 %456, i64 0, !dbg !18
  %462 = bitcast <2 x i64> %461 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1777 = shufflevector <4 x i32> %462, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1778 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %455), !dbg !18
  %463 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1778, i32 %dma.buffer.offset1776, <4 x i32> %dma.buffer.rsrc1777) #10, !dbg !18
  %464 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !12
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !12
  %467 = fmul <4 x half> %464, splat (half 0xH3015), !dbg !12
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !12
  %469 = fmul <4 x half> %465, splat (half 0xH3015), !dbg !12
  %470 = fmul <4 x half> %466, splat (half 0xH3015), !dbg !12
  %471 = fmul <4 x half> %468, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %472 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %472, <4 x float> %354) #9, !dbg !19
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %474 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !19
  %475 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %469, <4 x half> %474, <4 x float> %473) #9, !dbg !19
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %475, <4 x float> %358) #9, !dbg !19
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !19
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !19
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %469, <4 x half> %478, <4 x float> %477) #9, !dbg !19
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %479, <4 x float> %362) #9, !dbg !19
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %482 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !19
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %483 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !19
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %469, <4 x half> %482, <4 x float> %481) #9, !dbg !19
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %483, <4 x float> %366) #9, !dbg !19
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %486 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !19
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %487 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !19
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %469, <4 x half> %486, <4 x float> %485) #9, !dbg !19
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %487, <4 x float> %370) #9, !dbg !19
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %490 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !19
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !19
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %469, <4 x half> %490, <4 x float> %489) #9, !dbg !19
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %491, <4 x float> %374) #9, !dbg !19
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %494 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !19
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !19
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %469, <4 x half> %494, <4 x float> %493) #9, !dbg !19
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %495, <4 x float> %378) #9, !dbg !19
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %498 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !19
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !19
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %469, <4 x half> %498, <4 x float> %497) #9, !dbg !19
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %499, <4 x float> %382) #9, !dbg !19
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %502 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !19
  %503 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %469, <4 x half> %502, <4 x float> %501) #9, !dbg !19
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %503, <4 x float> %386) #9, !dbg !19
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %506 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !19
  %507 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %506, <4 x float> %505) #9, !dbg !19
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %507, <4 x float> %390) #9, !dbg !19
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %510 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !19
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %511 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !19
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %510, <4 x float> %509) #9, !dbg !19
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %511, <4 x float> %394) #9, !dbg !19
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %514 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !19
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %515 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !19
  %516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %514, <4 x float> %513) #9, !dbg !19
  %517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %515, <4 x float> %398) #9, !dbg !19
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %518 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !19
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %519 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !19
  %520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %518, <4 x float> %517) #9, !dbg !19
  %521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %519, <4 x float> %402) #9, !dbg !19
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %522 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !19
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %523 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !19
  %524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %522, <4 x float> %521) #9, !dbg !19
  %525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %523, <4 x float> %406) #9, !dbg !19
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %526 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !19
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %527 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !19
  %528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %526, <4 x float> %525) #9, !dbg !19
  %529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %527, <4 x float> %410) #9, !dbg !19
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %530 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !19
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %531 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !19
  %532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %530, <4 x float> %529) #9, !dbg !19
  %533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %531, <4 x float> %413) #9, !dbg !19
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %534 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !19
  %535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %534, <4 x float> %533) #9, !dbg !19
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %536 = extractelement <4 x float> %535, i64 3, !dbg !24
  %537 = extractelement <4 x float> %535, i64 2, !dbg !24
  %538 = extractelement <4 x float> %535, i64 1, !dbg !24
  %539 = extractelement <4 x float> %535, i64 0, !dbg !24
  %540 = extractelement <4 x float> %532, i64 3, !dbg !24
  %541 = extractelement <4 x float> %532, i64 2, !dbg !24
  %542 = extractelement <4 x float> %532, i64 1, !dbg !24
  %543 = extractelement <4 x float> %532, i64 0, !dbg !24
  %544 = extractelement <4 x float> %528, i64 3, !dbg !24
  %545 = extractelement <4 x float> %528, i64 2, !dbg !24
  %546 = extractelement <4 x float> %528, i64 1, !dbg !24
  %547 = extractelement <4 x float> %528, i64 0, !dbg !24
  %548 = extractelement <4 x float> %524, i64 3, !dbg !24
  %549 = extractelement <4 x float> %524, i64 2, !dbg !24
  %550 = extractelement <4 x float> %524, i64 1, !dbg !24
  %551 = extractelement <4 x float> %524, i64 0, !dbg !24
  %552 = extractelement <4 x float> %520, i64 3, !dbg !24
  %553 = extractelement <4 x float> %520, i64 2, !dbg !24
  %554 = extractelement <4 x float> %520, i64 1, !dbg !24
  %555 = extractelement <4 x float> %520, i64 0, !dbg !24
  %556 = extractelement <4 x float> %516, i64 3, !dbg !24
  %557 = extractelement <4 x float> %516, i64 2, !dbg !24
  %558 = extractelement <4 x float> %516, i64 1, !dbg !24
  %559 = extractelement <4 x float> %516, i64 0, !dbg !24
  %560 = extractelement <4 x float> %512, i64 3, !dbg !24
  %561 = extractelement <4 x float> %512, i64 2, !dbg !24
  %562 = extractelement <4 x float> %512, i64 1, !dbg !24
  %563 = extractelement <4 x float> %512, i64 0, !dbg !24
  %564 = extractelement <4 x float> %508, i64 3, !dbg !24
  %565 = extractelement <4 x float> %508, i64 2, !dbg !24
  %566 = extractelement <4 x float> %508, i64 1, !dbg !24
  %567 = extractelement <4 x float> %508, i64 0, !dbg !24
  %568 = extractelement <4 x float> %504, i64 3, !dbg !24
  %569 = extractelement <4 x float> %504, i64 2, !dbg !24
  %570 = extractelement <4 x float> %504, i64 1, !dbg !24
  %571 = extractelement <4 x float> %504, i64 0, !dbg !24
  %572 = extractelement <4 x float> %500, i64 3, !dbg !24
  %573 = extractelement <4 x float> %500, i64 2, !dbg !24
  %574 = extractelement <4 x float> %500, i64 1, !dbg !24
  %575 = extractelement <4 x float> %500, i64 0, !dbg !24
  %576 = extractelement <4 x float> %496, i64 3, !dbg !24
  %577 = extractelement <4 x float> %496, i64 2, !dbg !24
  %578 = extractelement <4 x float> %496, i64 1, !dbg !24
  %579 = extractelement <4 x float> %496, i64 0, !dbg !24
  %580 = extractelement <4 x float> %492, i64 3, !dbg !24
  %581 = extractelement <4 x float> %492, i64 2, !dbg !24
  %582 = extractelement <4 x float> %492, i64 1, !dbg !24
  %583 = extractelement <4 x float> %492, i64 0, !dbg !24
  %584 = extractelement <4 x float> %488, i64 3, !dbg !24
  %585 = extractelement <4 x float> %488, i64 2, !dbg !24
  %586 = extractelement <4 x float> %488, i64 1, !dbg !24
  %587 = extractelement <4 x float> %488, i64 0, !dbg !24
  %588 = extractelement <4 x float> %484, i64 3, !dbg !24
  %589 = extractelement <4 x float> %484, i64 2, !dbg !24
  %590 = extractelement <4 x float> %484, i64 1, !dbg !24
  %591 = extractelement <4 x float> %484, i64 0, !dbg !24
  %592 = extractelement <4 x float> %480, i64 3, !dbg !24
  %593 = extractelement <4 x float> %480, i64 2, !dbg !24
  %594 = extractelement <4 x float> %480, i64 1, !dbg !24
  %595 = extractelement <4 x float> %480, i64 0, !dbg !24
  %596 = extractelement <4 x float> %476, i64 3, !dbg !24
  %597 = extractelement <4 x float> %476, i64 2, !dbg !24
  %598 = extractelement <4 x float> %476, i64 1, !dbg !24
  %599 = extractelement <4 x float> %476, i64 0, !dbg !24
  %address.thread413 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local414 = shl nuw nsw i32 %address.thread413, 5, !dbg !13
  %address.local415 = and i32 %address.local414, 6144, !dbg !13
  %address.local416 = shl nuw nsw i32 %address.thread413, 7, !dbg !13
  %address.local417 = and i32 %address.local416, 1920, !dbg !13
  %address.local418 = or disjoint i32 %address.local415, %address.local417, !dbg !13
  %address.local419 = zext nneg i32 %address.local418 to i64, !dbg !13
  %address.local420 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %600 = lshr i32 %address.thread413, 4, !dbg !12
  %address.local422 = zext nneg i32 %600 to i64, !dbg !12
  %address.local423 = and i64 %address.local422, 3, !dbg !12
  %address.local424 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local423, !dbg !13
  store float %599, ptr addrspace(3) %address.local424, align 4, !dbg !13
  %address.local425 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local426 = or disjoint i64 %address.local423, 4, !dbg !13
  %address.local427 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local425, i64 %address.local426, !dbg !13
  store float %598, ptr addrspace(3) %address.local427, align 4, !dbg !13
  %address.local428 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local429 = or disjoint i64 %address.local423, 8, !dbg !13
  %address.local430 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local428, i64 %address.local429, !dbg !13
  store float %597, ptr addrspace(3) %address.local430, align 4, !dbg !13
  %address.local431 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local432 = or i64 %address.local422, 12, !dbg !13
  %address.local433 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local431, i64 %address.local432, !dbg !13
  store float %596, ptr addrspace(3) %address.local433, align 4, !dbg !13
  %address.local434 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local435 = or disjoint i64 %address.local423, 16, !dbg !13
  %address.local436 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local434, i64 %address.local435, !dbg !13
  store float %595, ptr addrspace(3) %address.local436, align 4, !dbg !13
  %address.local437 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local438 = or disjoint i64 %address.local423, 20, !dbg !13
  %address.local439 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local437, i64 %address.local438, !dbg !13
  store float %594, ptr addrspace(3) %address.local439, align 4, !dbg !13
  %address.local440 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local441 = or disjoint i64 %address.local423, 24, !dbg !13
  %address.local442 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local440, i64 %address.local441, !dbg !13
  store float %593, ptr addrspace(3) %address.local442, align 4, !dbg !13
  %address.local443 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local444 = or i64 %address.local422, 28, !dbg !13
  %address.local445 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local443, i64 %address.local444, !dbg !13
  store float %592, ptr addrspace(3) %address.local445, align 4, !dbg !13
  %address.local446 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local447 = or disjoint i64 %address.local423, 32, !dbg !13
  %address.local448 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local446, i64 %address.local447, !dbg !13
  store float %591, ptr addrspace(3) %address.local448, align 4, !dbg !13
  %address.local449 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local450 = or disjoint i64 %address.local423, 36, !dbg !13
  %address.local451 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local449, i64 %address.local450, !dbg !13
  store float %590, ptr addrspace(3) %address.local451, align 4, !dbg !13
  %address.local452 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local453 = or disjoint i64 %address.local423, 40, !dbg !13
  %address.local454 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local452, i64 %address.local453, !dbg !13
  store float %589, ptr addrspace(3) %address.local454, align 4, !dbg !13
  %address.local455 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local456 = or i64 %address.local422, 44, !dbg !13
  %address.local457 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local455, i64 %address.local456, !dbg !13
  store float %588, ptr addrspace(3) %address.local457, align 4, !dbg !13
  %address.local458 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local459 = or disjoint i64 %address.local423, 48, !dbg !13
  %address.local460 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local458, i64 %address.local459, !dbg !13
  store float %587, ptr addrspace(3) %address.local460, align 4, !dbg !13
  %address.local461 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local462 = or disjoint i64 %address.local423, 52, !dbg !13
  %address.local463 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local461, i64 %address.local462, !dbg !13
  store float %586, ptr addrspace(3) %address.local463, align 4, !dbg !13
  %address.local464 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local465 = or disjoint i64 %address.local423, 56, !dbg !13
  %address.local466 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local464, i64 %address.local465, !dbg !13
  store float %585, ptr addrspace(3) %address.local466, align 4, !dbg !13
  %address.local467 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local468 = or i64 %address.local422, 60, !dbg !13
  %address.local469 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local467, i64 %address.local468, !dbg !13
  store float %584, ptr addrspace(3) %address.local469, align 4, !dbg !13
  %address.local470 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local471 = or disjoint i64 %address.local423, 64, !dbg !13
  %address.local472 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local470, i64 %address.local471, !dbg !13
  store float %583, ptr addrspace(3) %address.local472, align 4, !dbg !13
  %address.local473 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local474 = or disjoint i64 %address.local423, 68, !dbg !13
  %address.local475 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local473, i64 %address.local474, !dbg !13
  store float %582, ptr addrspace(3) %address.local475, align 4, !dbg !13
  %address.local476 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local477 = or disjoint i64 %address.local423, 72, !dbg !13
  %address.local478 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local476, i64 %address.local477, !dbg !13
  store float %581, ptr addrspace(3) %address.local478, align 4, !dbg !13
  %address.local479 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local480 = or i64 %address.local422, 76, !dbg !13
  %address.local481 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local479, i64 %address.local480, !dbg !13
  store float %580, ptr addrspace(3) %address.local481, align 4, !dbg !13
  %address.local482 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local483 = or disjoint i64 %address.local423, 80, !dbg !13
  %address.local484 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local482, i64 %address.local483, !dbg !13
  store float %579, ptr addrspace(3) %address.local484, align 4, !dbg !13
  %address.local485 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local486 = or disjoint i64 %address.local423, 84, !dbg !13
  %address.local487 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local485, i64 %address.local486, !dbg !13
  store float %578, ptr addrspace(3) %address.local487, align 4, !dbg !13
  %address.local488 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local489 = or disjoint i64 %address.local423, 88, !dbg !13
  %address.local490 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local488, i64 %address.local489, !dbg !13
  store float %577, ptr addrspace(3) %address.local490, align 4, !dbg !13
  %address.local491 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local492 = or i64 %address.local422, 92, !dbg !13
  %address.local493 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local491, i64 %address.local492, !dbg !13
  store float %576, ptr addrspace(3) %address.local493, align 4, !dbg !13
  %address.local494 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local495 = or disjoint i64 %address.local423, 96, !dbg !13
  %address.local496 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local494, i64 %address.local495, !dbg !13
  store float %575, ptr addrspace(3) %address.local496, align 4, !dbg !13
  %address.local497 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local498 = or disjoint i64 %address.local423, 100, !dbg !13
  %address.local499 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local497, i64 %address.local498, !dbg !13
  store float %574, ptr addrspace(3) %address.local499, align 4, !dbg !13
  %address.local500 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local501 = or disjoint i64 %address.local423, 104, !dbg !13
  %address.local502 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local500, i64 %address.local501, !dbg !13
  store float %573, ptr addrspace(3) %address.local502, align 4, !dbg !13
  %address.local503 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local504 = or i64 %address.local422, 108, !dbg !13
  %address.local505 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local503, i64 %address.local504, !dbg !13
  store float %572, ptr addrspace(3) %address.local505, align 4, !dbg !13
  %address.local506 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local507 = or disjoint i64 %address.local423, 112, !dbg !13
  %address.local508 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local506, i64 %address.local507, !dbg !13
  store float %571, ptr addrspace(3) %address.local508, align 4, !dbg !13
  %address.local509 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local510 = or disjoint i64 %address.local423, 116, !dbg !13
  %address.local511 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local509, i64 %address.local510, !dbg !13
  store float %570, ptr addrspace(3) %address.local511, align 4, !dbg !13
  %address.local512 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local513 = or disjoint i64 %address.local423, 120, !dbg !13
  %address.local514 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local512, i64 %address.local513, !dbg !13
  store float %569, ptr addrspace(3) %address.local514, align 4, !dbg !13
  %address.local515 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local516 = or i64 %address.local422, 124, !dbg !13
  %address.local517 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local515, i64 %address.local516, !dbg !13
  store float %568, ptr addrspace(3) %address.local517, align 4, !dbg !13
  %address.local518 = shl nuw nsw i32 %address.thread413, 5, !dbg !13
  %address.local519 = and i32 %address.local518, 6144, !dbg !13
  %address.local520 = shl nuw nsw i32 %address.thread413, 7, !dbg !13
  %address.local521 = and i32 %address.local520, 1920, !dbg !13
  %address.local522 = or disjoint i32 %address.local519, %address.local521, !dbg !13
  %address.local523 = or disjoint i32 %address.local522, 8192, !dbg !13
  %address.local524 = zext nneg i32 %address.local523 to i64, !dbg !13
  %address.local525 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local526 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local423, !dbg !13
  store float %567, ptr addrspace(3) %address.local526, align 4, !dbg !13
  %address.local527 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local528 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local527, i64 %address.local426, !dbg !13
  store float %566, ptr addrspace(3) %address.local528, align 4, !dbg !13
  %address.local529 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local530 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local529, i64 %address.local429, !dbg !13
  store float %565, ptr addrspace(3) %address.local530, align 4, !dbg !13
  %address.local531 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local532 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local531, i64 %address.local432, !dbg !13
  store float %564, ptr addrspace(3) %address.local532, align 4, !dbg !13
  %address.local533 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local534 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local533, i64 %address.local435, !dbg !13
  store float %563, ptr addrspace(3) %address.local534, align 4, !dbg !13
  %address.local535 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local536 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local535, i64 %address.local438, !dbg !13
  store float %562, ptr addrspace(3) %address.local536, align 4, !dbg !13
  %address.local537 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local538 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local537, i64 %address.local441, !dbg !13
  store float %561, ptr addrspace(3) %address.local538, align 4, !dbg !13
  %address.local539 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local540 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local539, i64 %address.local444, !dbg !13
  store float %560, ptr addrspace(3) %address.local540, align 4, !dbg !13
  %address.local541 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local542 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local541, i64 %address.local447, !dbg !13
  store float %559, ptr addrspace(3) %address.local542, align 4, !dbg !13
  %address.local543 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local544 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local543, i64 %address.local450, !dbg !13
  store float %558, ptr addrspace(3) %address.local544, align 4, !dbg !13
  %address.local545 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local546 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local545, i64 %address.local453, !dbg !13
  store float %557, ptr addrspace(3) %address.local546, align 4, !dbg !13
  %address.local547 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local548 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local547, i64 %address.local456, !dbg !13
  store float %556, ptr addrspace(3) %address.local548, align 4, !dbg !13
  %address.local549 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local550 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local549, i64 %address.local459, !dbg !13
  store float %555, ptr addrspace(3) %address.local550, align 4, !dbg !13
  %address.local551 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local552 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local551, i64 %address.local462, !dbg !13
  store float %554, ptr addrspace(3) %address.local552, align 4, !dbg !13
  %address.local553 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local554 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local553, i64 %address.local465, !dbg !13
  store float %553, ptr addrspace(3) %address.local554, align 4, !dbg !13
  %address.local555 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local556 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local555, i64 %address.local468, !dbg !13
  store float %552, ptr addrspace(3) %address.local556, align 4, !dbg !13
  %address.local557 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local558 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local557, i64 %address.local471, !dbg !13
  store float %551, ptr addrspace(3) %address.local558, align 4, !dbg !13
  %address.local559 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local560 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local559, i64 %address.local474, !dbg !13
  store float %550, ptr addrspace(3) %address.local560, align 4, !dbg !13
  %address.local561 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local562 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local561, i64 %address.local477, !dbg !13
  store float %549, ptr addrspace(3) %address.local562, align 4, !dbg !13
  %address.local563 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local564 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local563, i64 %address.local480, !dbg !13
  store float %548, ptr addrspace(3) %address.local564, align 4, !dbg !13
  %address.local565 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local566 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local565, i64 %address.local483, !dbg !13
  store float %547, ptr addrspace(3) %address.local566, align 4, !dbg !13
  %address.local567 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local568 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local567, i64 %address.local486, !dbg !13
  store float %546, ptr addrspace(3) %address.local568, align 4, !dbg !13
  %address.local569 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local570 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local569, i64 %address.local489, !dbg !13
  store float %545, ptr addrspace(3) %address.local570, align 4, !dbg !13
  %address.local571 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local572 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local571, i64 %address.local492, !dbg !13
  store float %544, ptr addrspace(3) %address.local572, align 4, !dbg !13
  %address.local573 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local574 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local573, i64 %address.local495, !dbg !13
  store float %543, ptr addrspace(3) %address.local574, align 4, !dbg !13
  %address.local575 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local576 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local575, i64 %address.local498, !dbg !13
  store float %542, ptr addrspace(3) %address.local576, align 4, !dbg !13
  %address.local577 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local578 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local577, i64 %address.local501, !dbg !13
  store float %541, ptr addrspace(3) %address.local578, align 4, !dbg !13
  %address.local579 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local580 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local579, i64 %address.local504, !dbg !13
  store float %540, ptr addrspace(3) %address.local580, align 4, !dbg !13
  %address.local581 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local582 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local581, i64 %address.local507, !dbg !13
  store float %539, ptr addrspace(3) %address.local582, align 4, !dbg !13
  %address.local583 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local584 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local583, i64 %address.local510, !dbg !13
  store float %538, ptr addrspace(3) %address.local584, align 4, !dbg !13
  %address.local585 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local586 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local585, i64 %address.local513, !dbg !13
  store float %537, ptr addrspace(3) %address.local586, align 4, !dbg !13
  %address.local587 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local588 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local587, i64 %address.local516, !dbg !13
  store float %536, ptr addrspace(3) %address.local588, align 4, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.thread589 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local590 = shl nuw nsw i32 %address.thread589, 5, !dbg !13
  %address.local591 = and i32 %address.local590, 6144, !dbg !13
  %address.local592 = shl nuw nsw i32 %address.thread589, 7, !dbg !13
  %address.local593 = and i32 %address.local592, 1920, !dbg !13
  %address.local594 = or disjoint i32 %address.local591, %address.local593, !dbg !13
  %address.local595 = zext nneg i32 %address.local594 to i64, !dbg !13
  %address.local596 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local595, !dbg !13
  %601 = lshr i32 %address.thread589, 2, !dbg !12
  %602 = and i32 %601, 12, !dbg !12
  %address.local600 = zext nneg i32 %602 to i64, !dbg !12
  %address.local601 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local596, i64 %address.local600, !dbg !13
  %603 = load float, ptr addrspace(3) %address.local601, align 16, !dbg !13
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local601, i64 4, !dbg !13
  %604 = load float, ptr addrspace(3) %lds.const.gep1682, align 4, !dbg !13
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local601, i64 8, !dbg !13
  %605 = load float, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !13
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local601, i64 12, !dbg !13
  %606 = load float, ptr addrspace(3) %lds.const.gep1684, align 4, !dbg !13
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local601, i64 64, !dbg !13
  %607 = load float, ptr addrspace(3) %lds.const.gep1685, align 16, !dbg !13
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local601, i64 68, !dbg !13
  %608 = load float, ptr addrspace(3) %lds.const.gep1686, align 4, !dbg !13
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local601, i64 72, !dbg !13
  %609 = load float, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !13
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local601, i64 76, !dbg !13
  %610 = load float, ptr addrspace(3) %lds.const.gep1688, align 4, !dbg !13
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local601, i64 128, !dbg !13
  %611 = load float, ptr addrspace(3) %lds.const.gep1689, align 16, !dbg !13
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local601, i64 132, !dbg !13
  %612 = load float, ptr addrspace(3) %lds.const.gep1690, align 4, !dbg !13
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local601, i64 136, !dbg !13
  %613 = load float, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !13
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local601, i64 140, !dbg !13
  %614 = load float, ptr addrspace(3) %lds.const.gep1692, align 4, !dbg !13
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local601, i64 192, !dbg !13
  %615 = load float, ptr addrspace(3) %lds.const.gep1693, align 16, !dbg !13
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local601, i64 196, !dbg !13
  %616 = load float, ptr addrspace(3) %lds.const.gep1694, align 4, !dbg !13
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local601, i64 200, !dbg !13
  %617 = load float, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !13
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local601, i64 204, !dbg !13
  %618 = load float, ptr addrspace(3) %lds.const.gep1696, align 4, !dbg !13
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local601, i64 256, !dbg !13
  %619 = load float, ptr addrspace(3) %lds.const.gep1697, align 16, !dbg !13
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local601, i64 260, !dbg !13
  %620 = load float, ptr addrspace(3) %lds.const.gep1698, align 4, !dbg !13
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local601, i64 264, !dbg !13
  %621 = load float, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !13
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local601, i64 268, !dbg !13
  %622 = load float, ptr addrspace(3) %lds.const.gep1700, align 4, !dbg !13
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local601, i64 320, !dbg !13
  %623 = load float, ptr addrspace(3) %lds.const.gep1701, align 16, !dbg !13
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %address.local601, i64 324, !dbg !13
  %624 = load float, ptr addrspace(3) %lds.const.gep1702, align 4, !dbg !13
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %address.local601, i64 328, !dbg !13
  %625 = load float, ptr addrspace(3) %lds.const.gep1703, align 8, !dbg !13
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local601, i64 332, !dbg !13
  %626 = load float, ptr addrspace(3) %lds.const.gep1704, align 4, !dbg !13
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local601, i64 384, !dbg !13
  %627 = load float, ptr addrspace(3) %lds.const.gep1705, align 16, !dbg !13
  %lds.const.gep1706 = getelementptr i8, ptr addrspace(3) %address.local601, i64 388, !dbg !13
  %628 = load float, ptr addrspace(3) %lds.const.gep1706, align 4, !dbg !13
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %address.local601, i64 392, !dbg !13
  %629 = load float, ptr addrspace(3) %lds.const.gep1707, align 8, !dbg !13
  %lds.const.gep1708 = getelementptr i8, ptr addrspace(3) %address.local601, i64 396, !dbg !13
  %630 = load float, ptr addrspace(3) %lds.const.gep1708, align 4, !dbg !13
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local601, i64 448, !dbg !13
  %631 = load float, ptr addrspace(3) %lds.const.gep1709, align 16, !dbg !13
  %lds.const.gep1710 = getelementptr i8, ptr addrspace(3) %address.local601, i64 452, !dbg !13
  %632 = load float, ptr addrspace(3) %lds.const.gep1710, align 4, !dbg !13
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %address.local601, i64 456, !dbg !13
  %633 = load float, ptr addrspace(3) %lds.const.gep1711, align 8, !dbg !13
  %lds.const.gep1712 = getelementptr i8, ptr addrspace(3) %address.local601, i64 460, !dbg !13
  %634 = load float, ptr addrspace(3) %lds.const.gep1712, align 4, !dbg !13
  %lds.const.gep1713 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32768, !dbg !13
  %635 = load float, ptr addrspace(3) %lds.const.gep1713, align 16, !dbg !13
  %lds.const.gep1714 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32772, !dbg !13
  %636 = load float, ptr addrspace(3) %lds.const.gep1714, align 4, !dbg !13
  %lds.const.gep1715 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32776, !dbg !13
  %637 = load float, ptr addrspace(3) %lds.const.gep1715, align 8, !dbg !13
  %lds.const.gep1716 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32780, !dbg !13
  %638 = load float, ptr addrspace(3) %lds.const.gep1716, align 4, !dbg !13
  %lds.const.gep1717 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32832, !dbg !13
  %639 = load float, ptr addrspace(3) %lds.const.gep1717, align 16, !dbg !13
  %lds.const.gep1718 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32836, !dbg !13
  %640 = load float, ptr addrspace(3) %lds.const.gep1718, align 4, !dbg !13
  %lds.const.gep1719 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32840, !dbg !13
  %641 = load float, ptr addrspace(3) %lds.const.gep1719, align 8, !dbg !13
  %lds.const.gep1720 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32844, !dbg !13
  %642 = load float, ptr addrspace(3) %lds.const.gep1720, align 4, !dbg !13
  %lds.const.gep1721 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32896, !dbg !13
  %643 = load float, ptr addrspace(3) %lds.const.gep1721, align 16, !dbg !13
  %lds.const.gep1722 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32900, !dbg !13
  %644 = load float, ptr addrspace(3) %lds.const.gep1722, align 4, !dbg !13
  %lds.const.gep1723 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32904, !dbg !13
  %645 = load float, ptr addrspace(3) %lds.const.gep1723, align 8, !dbg !13
  %lds.const.gep1724 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32908, !dbg !13
  %646 = load float, ptr addrspace(3) %lds.const.gep1724, align 4, !dbg !13
  %lds.const.gep1725 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32960, !dbg !13
  %647 = load float, ptr addrspace(3) %lds.const.gep1725, align 16, !dbg !13
  %lds.const.gep1726 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32964, !dbg !13
  %648 = load float, ptr addrspace(3) %lds.const.gep1726, align 4, !dbg !13
  %lds.const.gep1727 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32968, !dbg !13
  %649 = load float, ptr addrspace(3) %lds.const.gep1727, align 8, !dbg !13
  %lds.const.gep1728 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32972, !dbg !13
  %650 = load float, ptr addrspace(3) %lds.const.gep1728, align 4, !dbg !13
  %lds.const.gep1729 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33024, !dbg !13
  %651 = load float, ptr addrspace(3) %lds.const.gep1729, align 16, !dbg !13
  %lds.const.gep1730 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33028, !dbg !13
  %652 = load float, ptr addrspace(3) %lds.const.gep1730, align 4, !dbg !13
  %lds.const.gep1731 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33032, !dbg !13
  %653 = load float, ptr addrspace(3) %lds.const.gep1731, align 8, !dbg !13
  %lds.const.gep1732 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33036, !dbg !13
  %654 = load float, ptr addrspace(3) %lds.const.gep1732, align 4, !dbg !13
  %lds.const.gep1733 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33088, !dbg !13
  %655 = load float, ptr addrspace(3) %lds.const.gep1733, align 16, !dbg !13
  %lds.const.gep1734 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33092, !dbg !13
  %656 = load float, ptr addrspace(3) %lds.const.gep1734, align 4, !dbg !13
  %lds.const.gep1735 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33096, !dbg !13
  %657 = load float, ptr addrspace(3) %lds.const.gep1735, align 8, !dbg !13
  %lds.const.gep1736 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33100, !dbg !13
  %658 = load float, ptr addrspace(3) %lds.const.gep1736, align 4, !dbg !13
  %lds.const.gep1737 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33152, !dbg !13
  %659 = load float, ptr addrspace(3) %lds.const.gep1737, align 16, !dbg !13
  %lds.const.gep1738 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33156, !dbg !13
  %660 = load float, ptr addrspace(3) %lds.const.gep1738, align 4, !dbg !13
  %lds.const.gep1739 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33160, !dbg !13
  %661 = load float, ptr addrspace(3) %lds.const.gep1739, align 8, !dbg !13
  %lds.const.gep1740 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33164, !dbg !13
  %662 = load float, ptr addrspace(3) %lds.const.gep1740, align 4, !dbg !13
  %lds.const.gep1741 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33216, !dbg !13
  %663 = load float, ptr addrspace(3) %lds.const.gep1741, align 16, !dbg !13
  %lds.const.gep1742 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33220, !dbg !13
  %664 = load float, ptr addrspace(3) %lds.const.gep1742, align 4, !dbg !13
  %lds.const.gep1743 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33224, !dbg !13
  %665 = load float, ptr addrspace(3) %lds.const.gep1743, align 8, !dbg !13
  %lds.const.gep1744 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33228, !dbg !13
  %666 = load float, ptr addrspace(3) %lds.const.gep1744, align 4, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %667 = or disjoint i64 %63, 127, !dbg !14
  %668 = icmp uge i64 %19, %667, !dbg !14
  %669 = and i1 %42, %668, !dbg !14
  br i1 %669, label %811, label %670, !dbg !14

670:                                              ; preds = %62
  %671 = add i64 %19, %35, !dbg !14
  %672 = add i64 %671, %52, !dbg !14
  %673 = sub i64 %672, %39, !dbg !14
  %674 = or disjoint i64 %63, %38, !dbg !14
  %675 = add i64 %673, 1, !dbg !25
  %676 = lshr i64 %675, 32, !dbg !26
  %compare.high = trunc nuw i64 %676 to i32, !dbg !26
  %677 = lshr i64 %63, 32, !dbg !26
  %compare.high1303 = trunc nuw i64 %677 to i32, !dbg !26
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1303, !dbg !26
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1303, !dbg !26
  %compare.low.base = trunc i64 %675 to i32, !dbg !26
  %compare.low.base1304 = trunc i64 %674 to i32, !dbg !26
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1304, !dbg !26
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !26
  %678 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 1) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !26
  %compare.low1305 = icmp ult i32 %compare.local, %compare.low.base, !dbg !26
  %compare.result1306 = select i1 %compare.high.equal, i1 %compare.low1305, i1 %compare.high.order, !dbg !26
  %679 = select i1 %compare.result1306, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1307 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 2) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1308 = or i32 %compare.local.base1307, 2, !dbg !26
  %compare.low1309 = icmp ult i32 %compare.local1308, %compare.low.base, !dbg !26
  %compare.result1310 = select i1 %compare.high.equal, i1 %compare.low1309, i1 %compare.high.order, !dbg !26
  %680 = select i1 %compare.result1310, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1311 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 3) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1312 = or i32 %compare.local.base1311, 3, !dbg !26
  %compare.low1313 = icmp ult i32 %compare.local1312, %compare.low.base, !dbg !26
  %compare.result1314 = select i1 %compare.high.equal, i1 %compare.low1313, i1 %compare.high.order, !dbg !26
  %681 = select i1 %compare.result1314, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1315 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 4) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1316 = or i32 %compare.local.base1315, 16, !dbg !26
  %compare.low1317 = icmp ult i32 %compare.local1316, %compare.low.base, !dbg !26
  %compare.result1318 = select i1 %compare.high.equal, i1 %compare.low1317, i1 %compare.high.order, !dbg !26
  %682 = select i1 %compare.result1318, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1319 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 5) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1320 = or i32 %compare.local.base1319, 17, !dbg !26
  %compare.low1321 = icmp ult i32 %compare.local1320, %compare.low.base, !dbg !26
  %compare.result1322 = select i1 %compare.high.equal, i1 %compare.low1321, i1 %compare.high.order, !dbg !26
  %683 = select i1 %compare.result1322, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1323 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 6) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1324 = or i32 %compare.local.base1323, 18, !dbg !26
  %compare.low1325 = icmp ult i32 %compare.local1324, %compare.low.base, !dbg !26
  %compare.result1326 = select i1 %compare.high.equal, i1 %compare.low1325, i1 %compare.high.order, !dbg !26
  %684 = select i1 %compare.result1326, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 7) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1328 = or i32 %compare.local.base1327, 19, !dbg !26
  %compare.low1329 = icmp ult i32 %compare.local1328, %compare.low.base, !dbg !26
  %compare.result1330 = select i1 %compare.high.equal, i1 %compare.low1329, i1 %compare.high.order, !dbg !26
  %685 = select i1 %compare.result1330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 8) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1332 = or i32 %compare.local.base1331, 32, !dbg !26
  %compare.low1333 = icmp ult i32 %compare.local1332, %compare.low.base, !dbg !26
  %compare.result1334 = select i1 %compare.high.equal, i1 %compare.low1333, i1 %compare.high.order, !dbg !26
  %686 = select i1 %compare.result1334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 9) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1336 = or i32 %compare.local.base1335, 33, !dbg !26
  %compare.low1337 = icmp ult i32 %compare.local1336, %compare.low.base, !dbg !26
  %compare.result1338 = select i1 %compare.high.equal, i1 %compare.low1337, i1 %compare.high.order, !dbg !26
  %687 = select i1 %compare.result1338, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 10) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1340 = or i32 %compare.local.base1339, 34, !dbg !26
  %compare.low1341 = icmp ult i32 %compare.local1340, %compare.low.base, !dbg !26
  %compare.result1342 = select i1 %compare.high.equal, i1 %compare.low1341, i1 %compare.high.order, !dbg !26
  %688 = select i1 %compare.result1342, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 11) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1344 = or i32 %compare.local.base1343, 35, !dbg !26
  %compare.low1345 = icmp ult i32 %compare.local1344, %compare.low.base, !dbg !26
  %compare.result1346 = select i1 %compare.high.equal, i1 %compare.low1345, i1 %compare.high.order, !dbg !26
  %689 = select i1 %compare.result1346, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 12) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1348 = or i32 %compare.local.base1347, 48, !dbg !26
  %compare.low1349 = icmp ult i32 %compare.local1348, %compare.low.base, !dbg !26
  %compare.result1350 = select i1 %compare.high.equal, i1 %compare.low1349, i1 %compare.high.order, !dbg !26
  %690 = select i1 %compare.result1350, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 13) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1352 = or i32 %compare.local.base1351, 49, !dbg !26
  %compare.low1353 = icmp ult i32 %compare.local1352, %compare.low.base, !dbg !26
  %compare.result1354 = select i1 %compare.high.equal, i1 %compare.low1353, i1 %compare.high.order, !dbg !26
  %691 = select i1 %compare.result1354, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 14) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1356 = or i32 %compare.local.base1355, 50, !dbg !26
  %compare.low1357 = icmp ult i32 %compare.local1356, %compare.low.base, !dbg !26
  %compare.result1358 = select i1 %compare.high.equal, i1 %compare.low1357, i1 %compare.high.order, !dbg !26
  %692 = select i1 %compare.result1358, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 15) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1360 = or i32 %compare.local.base1359, 51, !dbg !26
  %compare.low1361 = icmp ult i32 %compare.local1360, %compare.low.base, !dbg !26
  %compare.result1362 = select i1 %compare.high.equal, i1 %compare.low1361, i1 %compare.high.order, !dbg !26
  %693 = select i1 %compare.result1362, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 16) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1364 = or i32 %compare.local.base1363, 64, !dbg !26
  %compare.low1365 = icmp ult i32 %compare.local1364, %compare.low.base, !dbg !26
  %compare.result1366 = select i1 %compare.high.equal, i1 %compare.low1365, i1 %compare.high.order, !dbg !26
  %694 = select i1 %compare.result1366, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 17) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1368 = or i32 %compare.local.base1367, 65, !dbg !26
  %compare.low1369 = icmp ult i32 %compare.local1368, %compare.low.base, !dbg !26
  %compare.result1370 = select i1 %compare.high.equal, i1 %compare.low1369, i1 %compare.high.order, !dbg !26
  %695 = select i1 %compare.result1370, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 18) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1372 = or i32 %compare.local.base1371, 66, !dbg !26
  %compare.low1373 = icmp ult i32 %compare.local1372, %compare.low.base, !dbg !26
  %compare.result1374 = select i1 %compare.high.equal, i1 %compare.low1373, i1 %compare.high.order, !dbg !26
  %696 = select i1 %compare.result1374, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 19) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1376 = or i32 %compare.local.base1375, 67, !dbg !26
  %compare.low1377 = icmp ult i32 %compare.local1376, %compare.low.base, !dbg !26
  %compare.result1378 = select i1 %compare.high.equal, i1 %compare.low1377, i1 %compare.high.order, !dbg !26
  %697 = select i1 %compare.result1378, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 20) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1380 = or i32 %compare.local.base1379, 80, !dbg !26
  %compare.low1381 = icmp ult i32 %compare.local1380, %compare.low.base, !dbg !26
  %compare.result1382 = select i1 %compare.high.equal, i1 %compare.low1381, i1 %compare.high.order, !dbg !26
  %698 = select i1 %compare.result1382, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 21) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1384 = or i32 %compare.local.base1383, 81, !dbg !26
  %compare.low1385 = icmp ult i32 %compare.local1384, %compare.low.base, !dbg !26
  %compare.result1386 = select i1 %compare.high.equal, i1 %compare.low1385, i1 %compare.high.order, !dbg !26
  %699 = select i1 %compare.result1386, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 22) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1388 = or i32 %compare.local.base1387, 82, !dbg !26
  %compare.low1389 = icmp ult i32 %compare.local1388, %compare.low.base, !dbg !26
  %compare.result1390 = select i1 %compare.high.equal, i1 %compare.low1389, i1 %compare.high.order, !dbg !26
  %700 = select i1 %compare.result1390, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 23) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1392 = or i32 %compare.local.base1391, 83, !dbg !26
  %compare.low1393 = icmp ult i32 %compare.local1392, %compare.low.base, !dbg !26
  %compare.result1394 = select i1 %compare.high.equal, i1 %compare.low1393, i1 %compare.high.order, !dbg !26
  %701 = select i1 %compare.result1394, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 24) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1396 = or i32 %compare.local.base1395, 96, !dbg !26
  %compare.low1397 = icmp ult i32 %compare.local1396, %compare.low.base, !dbg !26
  %compare.result1398 = select i1 %compare.high.equal, i1 %compare.low1397, i1 %compare.high.order, !dbg !26
  %702 = select i1 %compare.result1398, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 25) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1400 = or i32 %compare.local.base1399, 97, !dbg !26
  %compare.low1401 = icmp ult i32 %compare.local1400, %compare.low.base, !dbg !26
  %compare.result1402 = select i1 %compare.high.equal, i1 %compare.low1401, i1 %compare.high.order, !dbg !26
  %703 = select i1 %compare.result1402, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 26) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1404 = or i32 %compare.local.base1403, 98, !dbg !26
  %compare.low1405 = icmp ult i32 %compare.local1404, %compare.low.base, !dbg !26
  %compare.result1406 = select i1 %compare.high.equal, i1 %compare.low1405, i1 %compare.high.order, !dbg !26
  %704 = select i1 %compare.result1406, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 27) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1408 = or i32 %compare.local.base1407, 99, !dbg !26
  %compare.low1409 = icmp ult i32 %compare.local1408, %compare.low.base, !dbg !26
  %compare.result1410 = select i1 %compare.high.equal, i1 %compare.low1409, i1 %compare.high.order, !dbg !26
  %705 = select i1 %compare.result1410, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 28) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1412 = or i32 %compare.local.base1411, 112, !dbg !26
  %compare.low1413 = icmp ult i32 %compare.local1412, %compare.low.base, !dbg !26
  %compare.result1414 = select i1 %compare.high.equal, i1 %compare.low1413, i1 %compare.high.order, !dbg !26
  %706 = select i1 %compare.result1414, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 29) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1416 = or i32 %compare.local.base1415, 113, !dbg !26
  %compare.low1417 = icmp ult i32 %compare.local1416, %compare.low.base, !dbg !26
  %compare.result1418 = select i1 %compare.high.equal, i1 %compare.low1417, i1 %compare.high.order, !dbg !26
  %707 = select i1 %compare.result1418, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 30) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1420 = or i32 %compare.local.base1419, 114, !dbg !26
  %compare.low1421 = icmp ult i32 %compare.local1420, %compare.low.base, !dbg !26
  %compare.result1422 = select i1 %compare.high.equal, i1 %compare.low1421, i1 %compare.high.order, !dbg !26
  %708 = select i1 %compare.result1422, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 31) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1424 = or i32 %compare.local.base1423, 115, !dbg !26
  %compare.low1425 = icmp ult i32 %compare.local1424, %compare.low.base, !dbg !26
  %compare.result1426 = select i1 %compare.high.equal, i1 %compare.low1425, i1 %compare.high.order, !dbg !26
  %709 = select i1 %compare.result1426, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %710 = add i64 %673, 65, !dbg !25
  %711 = lshr i64 %710, 32, !dbg !26
  %compare.high1427 = trunc nuw i64 %711 to i32, !dbg !26
  %712 = lshr i64 %63, 32, !dbg !26
  %compare.high1428 = trunc nuw i64 %712 to i32, !dbg !26
  %compare.high.equal1429 = icmp eq i32 %compare.high1427, %compare.high1428, !dbg !26
  %compare.high.order1430 = icmp ugt i32 %compare.high1427, %compare.high1428, !dbg !26
  %compare.low.base1431 = trunc i64 %710 to i32, !dbg !26
  %compare.low.base1432 = trunc i64 %674 to i32, !dbg !26
  %compare.low1433 = icmp ugt i32 %compare.low.base1431, %compare.low.base1432, !dbg !26
  %compare.result1434 = select i1 %compare.high.equal1429, i1 %compare.low1433, i1 %compare.high.order1430, !dbg !26
  %713 = select i1 %compare.result1434, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 32) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1436 = or i32 %compare.local.base1435, 1, !dbg !26
  %compare.low1437 = icmp ult i32 %compare.local1436, %compare.low.base1431, !dbg !26
  %compare.result1438 = select i1 %compare.high.equal1429, i1 %compare.low1437, i1 %compare.high.order1430, !dbg !26
  %714 = select i1 %compare.result1438, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 33) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1440 = or i32 %compare.local.base1439, 2, !dbg !26
  %compare.low1441 = icmp ult i32 %compare.local1440, %compare.low.base1431, !dbg !26
  %compare.result1442 = select i1 %compare.high.equal1429, i1 %compare.low1441, i1 %compare.high.order1430, !dbg !26
  %715 = select i1 %compare.result1442, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 34) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1444 = or i32 %compare.local.base1443, 3, !dbg !26
  %compare.low1445 = icmp ult i32 %compare.local1444, %compare.low.base1431, !dbg !26
  %compare.result1446 = select i1 %compare.high.equal1429, i1 %compare.low1445, i1 %compare.high.order1430, !dbg !26
  %716 = select i1 %compare.result1446, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1447 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 35) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1448 = or i32 %compare.local.base1447, 16, !dbg !26
  %compare.low1449 = icmp ult i32 %compare.local1448, %compare.low.base1431, !dbg !26
  %compare.result1450 = select i1 %compare.high.equal1429, i1 %compare.low1449, i1 %compare.high.order1430, !dbg !26
  %717 = select i1 %compare.result1450, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1451 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 36) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1452 = or i32 %compare.local.base1451, 17, !dbg !26
  %compare.low1453 = icmp ult i32 %compare.local1452, %compare.low.base1431, !dbg !26
  %compare.result1454 = select i1 %compare.high.equal1429, i1 %compare.low1453, i1 %compare.high.order1430, !dbg !26
  %718 = select i1 %compare.result1454, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1455 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 37) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1456 = or i32 %compare.local.base1455, 18, !dbg !26
  %compare.low1457 = icmp ult i32 %compare.local1456, %compare.low.base1431, !dbg !26
  %compare.result1458 = select i1 %compare.high.equal1429, i1 %compare.low1457, i1 %compare.high.order1430, !dbg !26
  %719 = select i1 %compare.result1458, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 38) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1460 = or i32 %compare.local.base1459, 19, !dbg !26
  %compare.low1461 = icmp ult i32 %compare.local1460, %compare.low.base1431, !dbg !26
  %compare.result1462 = select i1 %compare.high.equal1429, i1 %compare.low1461, i1 %compare.high.order1430, !dbg !26
  %720 = select i1 %compare.result1462, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 39) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1464 = or i32 %compare.local.base1463, 32, !dbg !26
  %compare.low1465 = icmp ult i32 %compare.local1464, %compare.low.base1431, !dbg !26
  %compare.result1466 = select i1 %compare.high.equal1429, i1 %compare.low1465, i1 %compare.high.order1430, !dbg !26
  %721 = select i1 %compare.result1466, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 40) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1468 = or i32 %compare.local.base1467, 33, !dbg !26
  %compare.low1469 = icmp ult i32 %compare.local1468, %compare.low.base1431, !dbg !26
  %compare.result1470 = select i1 %compare.high.equal1429, i1 %compare.low1469, i1 %compare.high.order1430, !dbg !26
  %722 = select i1 %compare.result1470, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 41) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1472 = or i32 %compare.local.base1471, 34, !dbg !26
  %compare.low1473 = icmp ult i32 %compare.local1472, %compare.low.base1431, !dbg !26
  %compare.result1474 = select i1 %compare.high.equal1429, i1 %compare.low1473, i1 %compare.high.order1430, !dbg !26
  %723 = select i1 %compare.result1474, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 42) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1476 = or i32 %compare.local.base1475, 35, !dbg !26
  %compare.low1477 = icmp ult i32 %compare.local1476, %compare.low.base1431, !dbg !26
  %compare.result1478 = select i1 %compare.high.equal1429, i1 %compare.low1477, i1 %compare.high.order1430, !dbg !26
  %724 = select i1 %compare.result1478, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 43) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1480 = or i32 %compare.local.base1479, 48, !dbg !26
  %compare.low1481 = icmp ult i32 %compare.local1480, %compare.low.base1431, !dbg !26
  %compare.result1482 = select i1 %compare.high.equal1429, i1 %compare.low1481, i1 %compare.high.order1430, !dbg !26
  %725 = select i1 %compare.result1482, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 44) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1484 = or i32 %compare.local.base1483, 49, !dbg !26
  %compare.low1485 = icmp ult i32 %compare.local1484, %compare.low.base1431, !dbg !26
  %compare.result1486 = select i1 %compare.high.equal1429, i1 %compare.low1485, i1 %compare.high.order1430, !dbg !26
  %726 = select i1 %compare.result1486, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1487 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 45) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1488 = or i32 %compare.local.base1487, 50, !dbg !26
  %compare.low1489 = icmp ult i32 %compare.local1488, %compare.low.base1431, !dbg !26
  %compare.result1490 = select i1 %compare.high.equal1429, i1 %compare.low1489, i1 %compare.high.order1430, !dbg !26
  %727 = select i1 %compare.result1490, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1491 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 46) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1492 = or i32 %compare.local.base1491, 51, !dbg !26
  %compare.low1493 = icmp ult i32 %compare.local1492, %compare.low.base1431, !dbg !26
  %compare.result1494 = select i1 %compare.high.equal1429, i1 %compare.low1493, i1 %compare.high.order1430, !dbg !26
  %728 = select i1 %compare.result1494, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1495 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 47) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1496 = or i32 %compare.local.base1495, 64, !dbg !26
  %compare.low1497 = icmp ult i32 %compare.local1496, %compare.low.base1431, !dbg !26
  %compare.result1498 = select i1 %compare.high.equal1429, i1 %compare.low1497, i1 %compare.high.order1430, !dbg !26
  %729 = select i1 %compare.result1498, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1499 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 48) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1500 = or i32 %compare.local.base1499, 65, !dbg !26
  %compare.low1501 = icmp ult i32 %compare.local1500, %compare.low.base1431, !dbg !26
  %compare.result1502 = select i1 %compare.high.equal1429, i1 %compare.low1501, i1 %compare.high.order1430, !dbg !26
  %730 = select i1 %compare.result1502, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1503 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 49) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1504 = or i32 %compare.local.base1503, 66, !dbg !26
  %compare.low1505 = icmp ult i32 %compare.local1504, %compare.low.base1431, !dbg !26
  %compare.result1506 = select i1 %compare.high.equal1429, i1 %compare.low1505, i1 %compare.high.order1430, !dbg !26
  %731 = select i1 %compare.result1506, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1507 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 50) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1508 = or i32 %compare.local.base1507, 67, !dbg !26
  %compare.low1509 = icmp ult i32 %compare.local1508, %compare.low.base1431, !dbg !26
  %compare.result1510 = select i1 %compare.high.equal1429, i1 %compare.low1509, i1 %compare.high.order1430, !dbg !26
  %732 = select i1 %compare.result1510, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1511 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 51) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1512 = or i32 %compare.local.base1511, 80, !dbg !26
  %compare.low1513 = icmp ult i32 %compare.local1512, %compare.low.base1431, !dbg !26
  %compare.result1514 = select i1 %compare.high.equal1429, i1 %compare.low1513, i1 %compare.high.order1430, !dbg !26
  %733 = select i1 %compare.result1514, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1515 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 52) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1516 = or i32 %compare.local.base1515, 81, !dbg !26
  %compare.low1517 = icmp ult i32 %compare.local1516, %compare.low.base1431, !dbg !26
  %compare.result1518 = select i1 %compare.high.equal1429, i1 %compare.low1517, i1 %compare.high.order1430, !dbg !26
  %734 = select i1 %compare.result1518, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1519 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 53) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1520 = or i32 %compare.local.base1519, 82, !dbg !26
  %compare.low1521 = icmp ult i32 %compare.local1520, %compare.low.base1431, !dbg !26
  %compare.result1522 = select i1 %compare.high.equal1429, i1 %compare.low1521, i1 %compare.high.order1430, !dbg !26
  %735 = select i1 %compare.result1522, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 54) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1524 = or i32 %compare.local.base1523, 83, !dbg !26
  %compare.low1525 = icmp ult i32 %compare.local1524, %compare.low.base1431, !dbg !26
  %compare.result1526 = select i1 %compare.high.equal1429, i1 %compare.low1525, i1 %compare.high.order1430, !dbg !26
  %736 = select i1 %compare.result1526, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 55) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1528 = or i32 %compare.local.base1527, 96, !dbg !26
  %compare.low1529 = icmp ult i32 %compare.local1528, %compare.low.base1431, !dbg !26
  %compare.result1530 = select i1 %compare.high.equal1429, i1 %compare.low1529, i1 %compare.high.order1430, !dbg !26
  %737 = select i1 %compare.result1530, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 56) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1532 = or i32 %compare.local.base1531, 97, !dbg !26
  %compare.low1533 = icmp ult i32 %compare.local1532, %compare.low.base1431, !dbg !26
  %compare.result1534 = select i1 %compare.high.equal1429, i1 %compare.low1533, i1 %compare.high.order1430, !dbg !26
  %738 = select i1 %compare.result1534, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 57) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1536 = or i32 %compare.local.base1535, 98, !dbg !26
  %compare.low1537 = icmp ult i32 %compare.local1536, %compare.low.base1431, !dbg !26
  %compare.result1538 = select i1 %compare.high.equal1429, i1 %compare.low1537, i1 %compare.high.order1430, !dbg !26
  %739 = select i1 %compare.result1538, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 58) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1540 = or i32 %compare.local.base1539, 99, !dbg !26
  %compare.low1541 = icmp ult i32 %compare.local1540, %compare.low.base1431, !dbg !26
  %compare.result1542 = select i1 %compare.high.equal1429, i1 %compare.low1541, i1 %compare.high.order1430, !dbg !26
  %740 = select i1 %compare.result1542, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 59) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1544 = or i32 %compare.local.base1543, 112, !dbg !26
  %compare.low1545 = icmp ult i32 %compare.local1544, %compare.low.base1431, !dbg !26
  %compare.result1546 = select i1 %compare.high.equal1429, i1 %compare.low1545, i1 %compare.high.order1430, !dbg !26
  %741 = select i1 %compare.result1546, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 60) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1548 = or i32 %compare.local.base1547, 113, !dbg !26
  %compare.low1549 = icmp ult i32 %compare.local1548, %compare.low.base1431, !dbg !26
  %compare.result1550 = select i1 %compare.high.equal1429, i1 %compare.low1549, i1 %compare.high.order1430, !dbg !26
  %742 = select i1 %compare.result1550, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 61) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1552 = or i32 %compare.local.base1551, 114, !dbg !26
  %compare.low1553 = icmp ult i32 %compare.local1552, %compare.low.base1431, !dbg !26
  %compare.result1554 = select i1 %compare.high.equal1429, i1 %compare.low1553, i1 %compare.high.order1430, !dbg !26
  %743 = select i1 %compare.result1554, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 62) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1556 = or i32 %compare.local.base1555, 115, !dbg !26
  %compare.low1557 = icmp ult i32 %compare.local1556, %compare.low.base1431, !dbg !26
  %compare.result1558 = select i1 %compare.high.equal1429, i1 %compare.low1557, i1 %compare.high.order1430, !dbg !26
  %744 = select i1 %compare.result1558, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %745 = insertelement <32 x float> poison, float %678, i64 0, !dbg !14
  %746 = insertelement <32 x float> %745, float %679, i64 1, !dbg !14
  %747 = insertelement <32 x float> %746, float %680, i64 2, !dbg !14
  %748 = insertelement <32 x float> %747, float %681, i64 3, !dbg !14
  %749 = insertelement <32 x float> %748, float %682, i64 4, !dbg !14
  %750 = insertelement <32 x float> %749, float %683, i64 5, !dbg !14
  %751 = insertelement <32 x float> %750, float %684, i64 6, !dbg !14
  %752 = insertelement <32 x float> %751, float %685, i64 7, !dbg !14
  %753 = insertelement <32 x float> %752, float %686, i64 8, !dbg !14
  %754 = insertelement <32 x float> %753, float %687, i64 9, !dbg !14
  %755 = insertelement <32 x float> %754, float %688, i64 10, !dbg !14
  %756 = insertelement <32 x float> %755, float %689, i64 11, !dbg !14
  %757 = insertelement <32 x float> %756, float %690, i64 12, !dbg !14
  %758 = insertelement <32 x float> %757, float %691, i64 13, !dbg !14
  %759 = insertelement <32 x float> %758, float %692, i64 14, !dbg !14
  %760 = insertelement <32 x float> %759, float %693, i64 15, !dbg !14
  %761 = insertelement <32 x float> %760, float %694, i64 16, !dbg !14
  %762 = insertelement <32 x float> %761, float %695, i64 17, !dbg !14
  %763 = insertelement <32 x float> %762, float %696, i64 18, !dbg !14
  %764 = insertelement <32 x float> %763, float %697, i64 19, !dbg !14
  %765 = insertelement <32 x float> %764, float %698, i64 20, !dbg !14
  %766 = insertelement <32 x float> %765, float %699, i64 21, !dbg !14
  %767 = insertelement <32 x float> %766, float %700, i64 22, !dbg !14
  %768 = insertelement <32 x float> %767, float %701, i64 23, !dbg !14
  %769 = insertelement <32 x float> %768, float %702, i64 24, !dbg !14
  %770 = insertelement <32 x float> %769, float %703, i64 25, !dbg !14
  %771 = insertelement <32 x float> %770, float %704, i64 26, !dbg !14
  %772 = insertelement <32 x float> %771, float %705, i64 27, !dbg !14
  %773 = insertelement <32 x float> %772, float %706, i64 28, !dbg !14
  %774 = insertelement <32 x float> %773, float %707, i64 29, !dbg !14
  %775 = insertelement <32 x float> %774, float %708, i64 30, !dbg !14
  %776 = insertelement <32 x float> %775, float %709, i64 31, !dbg !14
  %777 = insertvalue [2 x <32 x float>] poison, <32 x float> %776, 0, !dbg !14
  %778 = insertelement <32 x float> poison, float %713, i64 0, !dbg !14
  %779 = insertelement <32 x float> %778, float %714, i64 1, !dbg !14
  %780 = insertelement <32 x float> %779, float %715, i64 2, !dbg !14
  %781 = insertelement <32 x float> %780, float %716, i64 3, !dbg !14
  %782 = insertelement <32 x float> %781, float %717, i64 4, !dbg !14
  %783 = insertelement <32 x float> %782, float %718, i64 5, !dbg !14
  %784 = insertelement <32 x float> %783, float %719, i64 6, !dbg !14
  %785 = insertelement <32 x float> %784, float %720, i64 7, !dbg !14
  %786 = insertelement <32 x float> %785, float %721, i64 8, !dbg !14
  %787 = insertelement <32 x float> %786, float %722, i64 9, !dbg !14
  %788 = insertelement <32 x float> %787, float %723, i64 10, !dbg !14
  %789 = insertelement <32 x float> %788, float %724, i64 11, !dbg !14
  %790 = insertelement <32 x float> %789, float %725, i64 12, !dbg !14
  %791 = insertelement <32 x float> %790, float %726, i64 13, !dbg !14
  %792 = insertelement <32 x float> %791, float %727, i64 14, !dbg !14
  %793 = insertelement <32 x float> %792, float %728, i64 15, !dbg !14
  %794 = insertelement <32 x float> %793, float %729, i64 16, !dbg !14
  %795 = insertelement <32 x float> %794, float %730, i64 17, !dbg !14
  %796 = insertelement <32 x float> %795, float %731, i64 18, !dbg !14
  %797 = insertelement <32 x float> %796, float %732, i64 19, !dbg !14
  %798 = insertelement <32 x float> %797, float %733, i64 20, !dbg !14
  %799 = insertelement <32 x float> %798, float %734, i64 21, !dbg !14
  %800 = insertelement <32 x float> %799, float %735, i64 22, !dbg !14
  %801 = insertelement <32 x float> %800, float %736, i64 23, !dbg !14
  %802 = insertelement <32 x float> %801, float %737, i64 24, !dbg !14
  %803 = insertelement <32 x float> %802, float %738, i64 25, !dbg !14
  %804 = insertelement <32 x float> %803, float %739, i64 26, !dbg !14
  %805 = insertelement <32 x float> %804, float %740, i64 27, !dbg !14
  %806 = insertelement <32 x float> %805, float %741, i64 28, !dbg !14
  %807 = insertelement <32 x float> %806, float %742, i64 29, !dbg !14
  %808 = insertelement <32 x float> %807, float %743, i64 30, !dbg !14
  %809 = insertelement <32 x float> %808, float %744, i64 31, !dbg !14
  %810 = insertvalue [2 x <32 x float>] %777, <32 x float> %809, 1, !dbg !14
  br label %811, !dbg !14

811:                                              ; preds = %670, %62
  %812 = phi [2 x <32 x float>] [ %810, %670 ], [ zeroinitializer, %62 ], !dbg !14
  %813 = extractvalue [2 x <32 x float>] %812, 0, !dbg !28
  %814 = insertelement <4 x float> poison, float %603, i64 0, !dbg !28
  %815 = insertelement <4 x float> %814, float %604, i64 1, !dbg !28
  %816 = insertelement <4 x float> %815, float %605, i64 2, !dbg !28
  %817 = insertelement <4 x float> %816, float %606, i64 3, !dbg !28
  %818 = shufflevector <32 x float> %813, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %819 = fadd <4 x float> %817, %818, !dbg !28
  %820 = extractelement <4 x float> %819, i64 0, !dbg !28
  %821 = extractelement <4 x float> %819, i64 1, !dbg !28
  %822 = extractelement <4 x float> %819, i64 2, !dbg !28
  %823 = extractelement <4 x float> %819, i64 3, !dbg !28
  %824 = insertelement <4 x float> poison, float %607, i64 0, !dbg !28
  %825 = insertelement <4 x float> %824, float %608, i64 1, !dbg !28
  %826 = insertelement <4 x float> %825, float %609, i64 2, !dbg !28
  %827 = insertelement <4 x float> %826, float %610, i64 3, !dbg !28
  %828 = shufflevector <32 x float> %813, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %829 = fadd <4 x float> %827, %828, !dbg !28
  %830 = extractelement <4 x float> %829, i64 0, !dbg !28
  %831 = extractelement <4 x float> %829, i64 1, !dbg !28
  %832 = extractelement <4 x float> %829, i64 2, !dbg !28
  %833 = extractelement <4 x float> %829, i64 3, !dbg !28
  %834 = insertelement <4 x float> poison, float %611, i64 0, !dbg !28
  %835 = insertelement <4 x float> %834, float %612, i64 1, !dbg !28
  %836 = insertelement <4 x float> %835, float %613, i64 2, !dbg !28
  %837 = insertelement <4 x float> %836, float %614, i64 3, !dbg !28
  %838 = shufflevector <32 x float> %813, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !28
  %839 = fadd <4 x float> %837, %838, !dbg !28
  %840 = extractelement <4 x float> %839, i64 0, !dbg !28
  %841 = extractelement <4 x float> %839, i64 1, !dbg !28
  %842 = extractelement <4 x float> %839, i64 2, !dbg !28
  %843 = extractelement <4 x float> %839, i64 3, !dbg !28
  %844 = insertelement <4 x float> poison, float %615, i64 0, !dbg !28
  %845 = insertelement <4 x float> %844, float %616, i64 1, !dbg !28
  %846 = insertelement <4 x float> %845, float %617, i64 2, !dbg !28
  %847 = insertelement <4 x float> %846, float %618, i64 3, !dbg !28
  %848 = shufflevector <32 x float> %813, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !28
  %849 = fadd <4 x float> %847, %848, !dbg !28
  %850 = extractelement <4 x float> %849, i64 0, !dbg !28
  %851 = extractelement <4 x float> %849, i64 1, !dbg !28
  %852 = extractelement <4 x float> %849, i64 2, !dbg !28
  %853 = extractelement <4 x float> %849, i64 3, !dbg !28
  %854 = insertelement <4 x float> poison, float %619, i64 0, !dbg !28
  %855 = insertelement <4 x float> %854, float %620, i64 1, !dbg !28
  %856 = insertelement <4 x float> %855, float %621, i64 2, !dbg !28
  %857 = insertelement <4 x float> %856, float %622, i64 3, !dbg !28
  %858 = shufflevector <32 x float> %813, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !28
  %859 = fadd <4 x float> %857, %858, !dbg !28
  %860 = extractelement <4 x float> %859, i64 0, !dbg !28
  %861 = extractelement <4 x float> %859, i64 1, !dbg !28
  %862 = extractelement <4 x float> %859, i64 2, !dbg !28
  %863 = extractelement <4 x float> %859, i64 3, !dbg !28
  %864 = insertelement <4 x float> poison, float %623, i64 0, !dbg !28
  %865 = insertelement <4 x float> %864, float %624, i64 1, !dbg !28
  %866 = insertelement <4 x float> %865, float %625, i64 2, !dbg !28
  %867 = insertelement <4 x float> %866, float %626, i64 3, !dbg !28
  %868 = shufflevector <32 x float> %813, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !28
  %869 = fadd <4 x float> %867, %868, !dbg !28
  %870 = extractelement <4 x float> %869, i64 0, !dbg !28
  %871 = extractelement <4 x float> %869, i64 1, !dbg !28
  %872 = extractelement <4 x float> %869, i64 2, !dbg !28
  %873 = extractelement <4 x float> %869, i64 3, !dbg !28
  %874 = insertelement <4 x float> poison, float %627, i64 0, !dbg !28
  %875 = insertelement <4 x float> %874, float %628, i64 1, !dbg !28
  %876 = insertelement <4 x float> %875, float %629, i64 2, !dbg !28
  %877 = insertelement <4 x float> %876, float %630, i64 3, !dbg !28
  %878 = shufflevector <32 x float> %813, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !28
  %879 = fadd <4 x float> %877, %878, !dbg !28
  %880 = extractelement <4 x float> %879, i64 0, !dbg !28
  %881 = extractelement <4 x float> %879, i64 1, !dbg !28
  %882 = extractelement <4 x float> %879, i64 2, !dbg !28
  %883 = extractelement <4 x float> %879, i64 3, !dbg !28
  %884 = insertelement <4 x float> poison, float %631, i64 0, !dbg !28
  %885 = insertelement <4 x float> %884, float %632, i64 1, !dbg !28
  %886 = insertelement <4 x float> %885, float %633, i64 2, !dbg !28
  %887 = insertelement <4 x float> %886, float %634, i64 3, !dbg !28
  %888 = shufflevector <32 x float> %813, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !28
  %889 = fadd <4 x float> %887, %888, !dbg !28
  %890 = extractelement <4 x float> %889, i64 0, !dbg !28
  %891 = extractelement <4 x float> %889, i64 1, !dbg !28
  %892 = extractelement <4 x float> %889, i64 2, !dbg !28
  %893 = extractelement <4 x float> %889, i64 3, !dbg !28
  %894 = extractvalue [2 x <32 x float>] %812, 1, !dbg !28
  %895 = insertelement <4 x float> poison, float %635, i64 0, !dbg !28
  %896 = insertelement <4 x float> %895, float %636, i64 1, !dbg !28
  %897 = insertelement <4 x float> %896, float %637, i64 2, !dbg !28
  %898 = insertelement <4 x float> %897, float %638, i64 3, !dbg !28
  %899 = shufflevector <32 x float> %894, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %900 = fadd <4 x float> %898, %899, !dbg !28
  %901 = extractelement <4 x float> %900, i64 0, !dbg !28
  %902 = extractelement <4 x float> %900, i64 1, !dbg !28
  %903 = extractelement <4 x float> %900, i64 2, !dbg !28
  %904 = extractelement <4 x float> %900, i64 3, !dbg !28
  %905 = insertelement <4 x float> poison, float %639, i64 0, !dbg !28
  %906 = insertelement <4 x float> %905, float %640, i64 1, !dbg !28
  %907 = insertelement <4 x float> %906, float %641, i64 2, !dbg !28
  %908 = insertelement <4 x float> %907, float %642, i64 3, !dbg !28
  %909 = shufflevector <32 x float> %894, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %910 = fadd <4 x float> %908, %909, !dbg !28
  %911 = extractelement <4 x float> %910, i64 0, !dbg !28
  %912 = extractelement <4 x float> %910, i64 1, !dbg !28
  %913 = extractelement <4 x float> %910, i64 2, !dbg !28
  %914 = extractelement <4 x float> %910, i64 3, !dbg !28
  %915 = insertelement <4 x float> poison, float %643, i64 0, !dbg !28
  %916 = insertelement <4 x float> %915, float %644, i64 1, !dbg !28
  %917 = insertelement <4 x float> %916, float %645, i64 2, !dbg !28
  %918 = insertelement <4 x float> %917, float %646, i64 3, !dbg !28
  %919 = shufflevector <32 x float> %894, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !28
  %920 = fadd <4 x float> %918, %919, !dbg !28
  %921 = extractelement <4 x float> %920, i64 0, !dbg !28
  %922 = extractelement <4 x float> %920, i64 1, !dbg !28
  %923 = extractelement <4 x float> %920, i64 2, !dbg !28
  %924 = extractelement <4 x float> %920, i64 3, !dbg !28
  %925 = insertelement <4 x float> poison, float %647, i64 0, !dbg !28
  %926 = insertelement <4 x float> %925, float %648, i64 1, !dbg !28
  %927 = insertelement <4 x float> %926, float %649, i64 2, !dbg !28
  %928 = insertelement <4 x float> %927, float %650, i64 3, !dbg !28
  %929 = shufflevector <32 x float> %894, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !28
  %930 = fadd <4 x float> %928, %929, !dbg !28
  %931 = extractelement <4 x float> %930, i64 0, !dbg !28
  %932 = extractelement <4 x float> %930, i64 1, !dbg !28
  %933 = extractelement <4 x float> %930, i64 2, !dbg !28
  %934 = extractelement <4 x float> %930, i64 3, !dbg !28
  %935 = insertelement <4 x float> poison, float %651, i64 0, !dbg !28
  %936 = insertelement <4 x float> %935, float %652, i64 1, !dbg !28
  %937 = insertelement <4 x float> %936, float %653, i64 2, !dbg !28
  %938 = insertelement <4 x float> %937, float %654, i64 3, !dbg !28
  %939 = shufflevector <32 x float> %894, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !28
  %940 = fadd <4 x float> %938, %939, !dbg !28
  %941 = extractelement <4 x float> %940, i64 0, !dbg !28
  %942 = extractelement <4 x float> %940, i64 1, !dbg !28
  %943 = extractelement <4 x float> %940, i64 2, !dbg !28
  %944 = extractelement <4 x float> %940, i64 3, !dbg !28
  %945 = insertelement <4 x float> poison, float %655, i64 0, !dbg !28
  %946 = insertelement <4 x float> %945, float %656, i64 1, !dbg !28
  %947 = insertelement <4 x float> %946, float %657, i64 2, !dbg !28
  %948 = insertelement <4 x float> %947, float %658, i64 3, !dbg !28
  %949 = shufflevector <32 x float> %894, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !28
  %950 = fadd <4 x float> %948, %949, !dbg !28
  %951 = extractelement <4 x float> %950, i64 0, !dbg !28
  %952 = extractelement <4 x float> %950, i64 1, !dbg !28
  %953 = extractelement <4 x float> %950, i64 2, !dbg !28
  %954 = extractelement <4 x float> %950, i64 3, !dbg !28
  %955 = insertelement <4 x float> poison, float %659, i64 0, !dbg !28
  %956 = insertelement <4 x float> %955, float %660, i64 1, !dbg !28
  %957 = insertelement <4 x float> %956, float %661, i64 2, !dbg !28
  %958 = insertelement <4 x float> %957, float %662, i64 3, !dbg !28
  %959 = shufflevector <32 x float> %894, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !28
  %960 = fadd <4 x float> %958, %959, !dbg !28
  %961 = extractelement <4 x float> %960, i64 0, !dbg !28
  %962 = extractelement <4 x float> %960, i64 1, !dbg !28
  %963 = extractelement <4 x float> %960, i64 2, !dbg !28
  %964 = extractelement <4 x float> %960, i64 3, !dbg !28
  %965 = insertelement <4 x float> poison, float %663, i64 0, !dbg !28
  %966 = insertelement <4 x float> %965, float %664, i64 1, !dbg !28
  %967 = insertelement <4 x float> %966, float %665, i64 2, !dbg !28
  %968 = insertelement <4 x float> %967, float %666, i64 3, !dbg !28
  %969 = shufflevector <32 x float> %894, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !28
  %970 = fadd <4 x float> %968, %969, !dbg !28
  %971 = extractelement <4 x float> %970, i64 0, !dbg !28
  %972 = extractelement <4 x float> %970, i64 1, !dbg !28
  %973 = extractelement <4 x float> %970, i64 2, !dbg !28
  %974 = extractelement <4 x float> %970, i64 3, !dbg !28
  %975 = call float @llvm.maxnum.f32(float %820, float 0xFFF0000000000000), !dbg !29, !frisk.local_reduction !1
  %local.reduce182 = call float @llvm.maxnum.f32(float %975, float %830), !dbg !29, !frisk.local_reduction !1
  %local.reduce183 = call float @llvm.maxnum.f32(float %821, float %831), !dbg !29, !frisk.local_reduction !1
  %local.reduce184 = call float @llvm.maxnum.f32(float %822, float %832), !dbg !29, !frisk.local_reduction !1
  %local.reduce185 = call float @llvm.maxnum.f32(float %823, float %833), !dbg !29, !frisk.local_reduction !1
  %local.reduce186 = call float @llvm.maxnum.f32(float %local.reduce182, float %840), !dbg !29, !frisk.local_reduction !1
  %local.reduce187 = call float @llvm.maxnum.f32(float %local.reduce183, float %841), !dbg !29, !frisk.local_reduction !1
  %local.reduce188 = call float @llvm.maxnum.f32(float %local.reduce184, float %842), !dbg !29, !frisk.local_reduction !1
  %local.reduce189 = call float @llvm.maxnum.f32(float %local.reduce185, float %843), !dbg !29, !frisk.local_reduction !1
  %local.reduce190 = call float @llvm.maxnum.f32(float %local.reduce186, float %850), !dbg !29, !frisk.local_reduction !1
  %local.reduce191 = call float @llvm.maxnum.f32(float %local.reduce187, float %851), !dbg !29, !frisk.local_reduction !1
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %852), !dbg !29, !frisk.local_reduction !1
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %853), !dbg !29, !frisk.local_reduction !1
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %860), !dbg !29, !frisk.local_reduction !1
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %861), !dbg !29, !frisk.local_reduction !1
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %862), !dbg !29, !frisk.local_reduction !1
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %863), !dbg !29, !frisk.local_reduction !1
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %870), !dbg !29, !frisk.local_reduction !1
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %871), !dbg !29, !frisk.local_reduction !1
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %872), !dbg !29, !frisk.local_reduction !1
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %873), !dbg !29, !frisk.local_reduction !1
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %880), !dbg !29, !frisk.local_reduction !1
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %881), !dbg !29, !frisk.local_reduction !1
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %882), !dbg !29, !frisk.local_reduction !1
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %883), !dbg !29, !frisk.local_reduction !1
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %890), !dbg !29, !frisk.local_reduction !1
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %891), !dbg !29, !frisk.local_reduction !1
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %892), !dbg !29, !frisk.local_reduction !1
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %893), !dbg !29, !frisk.local_reduction !1
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %local.reduce207), !dbg !29, !frisk.local_reduction !1
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce208, float %local.reduce209), !dbg !29, !frisk.local_reduction !1
  %976 = call float @llvm.maxnum.f32(float %local.reduce210, float %local.reduce211), !dbg !29, !frisk.local_reduction !1
  %977 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !29
  %978 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %977), !dbg !29
  %979 = add nuw nsw i32 %978, 64, !dbg !29
  %980 = xor i32 %978, 16, !dbg !29
  %981 = and i32 %979, 192, !dbg !29
  %982 = icmp samesign ult i32 %980, %981, !dbg !29
  %983 = select i1 %982, i32 %980, i32 %978, !dbg !29
  %984 = shl nuw nsw i32 %983, 2, !dbg !29
  %985 = bitcast float %976 to i32, !dbg !29
  %986 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %985), !dbg !29
  %987 = bitcast i32 %986 to float, !dbg !29
  %988 = call float @llvm.maxnum.f32(float %976, float %987), !dbg !29
  %989 = xor i32 %978, 32, !dbg !29
  %990 = icmp samesign ult i32 %989, %981, !dbg !29
  %991 = select i1 %990, i32 %989, i32 %978, !dbg !29
  %992 = shl nuw nsw i32 %991, 2, !dbg !29
  %993 = bitcast float %988 to i32, !dbg !29
  %994 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %993), !dbg !29
  %995 = bitcast i32 %994 to float, !dbg !29
  %996 = call float @llvm.maxnum.f32(float %988, float %995), !dbg !29
  %997 = extractvalue [2 x <1 x float>] %57, 0, !dbg !29
  %998 = insertelement <1 x float> poison, float %996, i64 0, !dbg !29
  %999 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %998, <1 x float> %997), !dbg !29
  %1000 = call float @llvm.maxnum.f32(float %901, float 0xFFF0000000000000), !dbg !29, !frisk.local_reduction !1
  %local.reduce151 = call float @llvm.maxnum.f32(float %1000, float %911), !dbg !29, !frisk.local_reduction !1
  %local.reduce152 = call float @llvm.maxnum.f32(float %902, float %912), !dbg !29, !frisk.local_reduction !1
  %local.reduce153 = call float @llvm.maxnum.f32(float %903, float %913), !dbg !29, !frisk.local_reduction !1
  %local.reduce154 = call float @llvm.maxnum.f32(float %904, float %914), !dbg !29, !frisk.local_reduction !1
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %921), !dbg !29, !frisk.local_reduction !1
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %922), !dbg !29, !frisk.local_reduction !1
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %923), !dbg !29, !frisk.local_reduction !1
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %924), !dbg !29, !frisk.local_reduction !1
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %931), !dbg !29, !frisk.local_reduction !1
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %932), !dbg !29, !frisk.local_reduction !1
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %933), !dbg !29, !frisk.local_reduction !1
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %934), !dbg !29, !frisk.local_reduction !1
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %941), !dbg !29, !frisk.local_reduction !1
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %942), !dbg !29, !frisk.local_reduction !1
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %943), !dbg !29, !frisk.local_reduction !1
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %944), !dbg !29, !frisk.local_reduction !1
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %951), !dbg !29, !frisk.local_reduction !1
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %952), !dbg !29, !frisk.local_reduction !1
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %953), !dbg !29, !frisk.local_reduction !1
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %954), !dbg !29, !frisk.local_reduction !1
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %961), !dbg !29, !frisk.local_reduction !1
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %962), !dbg !29, !frisk.local_reduction !1
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %963), !dbg !29, !frisk.local_reduction !1
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %964), !dbg !29, !frisk.local_reduction !1
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %971), !dbg !29, !frisk.local_reduction !1
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %972), !dbg !29, !frisk.local_reduction !1
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %973), !dbg !29, !frisk.local_reduction !1
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %974), !dbg !29, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !29, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !29, !frisk.local_reduction !1
  %1001 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !29, !frisk.local_reduction !1
  %1002 = bitcast float %1001 to i32, !dbg !29
  %1003 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1002), !dbg !29
  %1004 = bitcast i32 %1003 to float, !dbg !29
  %1005 = call float @llvm.maxnum.f32(float %1001, float %1004), !dbg !29
  %1006 = bitcast float %1005 to i32, !dbg !29
  %1007 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1006), !dbg !29
  %1008 = bitcast i32 %1007 to float, !dbg !29
  %1009 = call float @llvm.maxnum.f32(float %1005, float %1008), !dbg !29
  %1010 = extractvalue [2 x <1 x float>] %57, 1, !dbg !29
  %1011 = insertelement <1 x float> poison, float %1009, i64 0, !dbg !29
  %1012 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1011, <1 x float> %1010), !dbg !29
  %1013 = shufflevector <1 x float> %999, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %1014 = fsub <4 x float> %819, %1013, !dbg !30
  %1015 = fsub <4 x float> %829, %1013, !dbg !30
  %1016 = fsub <4 x float> %839, %1013, !dbg !30
  %1017 = fsub <4 x float> %849, %1013, !dbg !30
  %1018 = fsub <4 x float> %859, %1013, !dbg !30
  %1019 = fsub <4 x float> %869, %1013, !dbg !30
  %1020 = fsub <4 x float> %879, %1013, !dbg !30
  %1021 = fsub <4 x float> %889, %1013, !dbg !30
  %1022 = shufflevector <1 x float> %1012, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %1023 = fsub <4 x float> %900, %1022, !dbg !30
  %1024 = fsub <4 x float> %910, %1022, !dbg !30
  %1025 = fsub <4 x float> %920, %1022, !dbg !30
  %1026 = fsub <4 x float> %930, %1022, !dbg !30
  %1027 = fsub <4 x float> %940, %1022, !dbg !30
  %1028 = fsub <4 x float> %950, %1022, !dbg !30
  %1029 = fsub <4 x float> %960, %1022, !dbg !30
  %1030 = fsub <4 x float> %970, %1022, !dbg !30
  br i1 %43, label %1031, label %1038, !dbg !15

1031:                                             ; preds = %811
  %1032 = fsub <1 x float> %1010, %1012, !dbg !31
  %1033 = extractelement <1 x float> %1032, i64 0, !dbg !31
  %1034 = fsub <1 x float> %997, %999, !dbg !31
  %1035 = extractelement <1 x float> %1034, i64 0, !dbg !31
  %address.thread766 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !40, !frisk.register_lifetimes !1
  %address.local767 = lshr i32 %address.thread766, 2, !dbg !13
  %address.local768 = and i32 %address.local767, 48, !dbg !13
  %address.local769 = and i32 %address.thread766, 15, !dbg !13
  %address.local770 = or disjoint i32 %address.local768, %address.local769, !dbg !13
  %address.local771 = zext nneg i32 %address.local770 to i64, !dbg !13
  %address.local772 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local771, !dbg !15
  store float %1035, ptr addrspace(3) %address.local772, align 4, !dbg !15
  %address.local773 = lshr i32 %address.thread766, 2, !dbg !13
  %address.local774 = and i32 %address.local773, 48, !dbg !13
  %address.local775 = and i32 %address.thread766, 15, !dbg !13
  %address.local776 = or disjoint i32 %address.local774, %address.local775, !dbg !13
  %1036 = zext nneg i32 %address.local776 to i64, !dbg !15
  %1037 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %1036, !dbg !15
  %address.local779 = getelementptr inbounds nuw i8, ptr addrspace(3) %1037, i64 256, !dbg !15
  store float %1033, ptr addrspace(3) %address.local779, align 4, !dbg !15
  br label %1038, !dbg !15

1038:                                             ; preds = %1031, %811
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.thread780 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !40, !frisk.register_lifetimes !1
  %address.local781 = lshr i32 %address.thread780, 1, !dbg !15
  %address.local782 = and i32 %address.local781, 96, !dbg !15
  %address.local783 = and i32 %address.thread780, 15, !dbg !15
  %address.local784 = or disjoint i32 %address.local782, %address.local783, !dbg !15
  %address.local785 = zext nneg i32 %address.local784 to i64, !dbg !15
  %address.local786 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local785, !dbg !15
  %1039 = load float, ptr addrspace(3) %address.local786, align 4, !dbg !15
  %lds.const.gep1745 = getelementptr i8, ptr addrspace(3) %address.local786, i64 64, !dbg !15
  %1040 = load float, ptr addrspace(3) %lds.const.gep1745, align 4, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1041 = call float @llvm.amdgcn.exp2.f32(float %1039), !dbg !15
  %1042 = call float @llvm.amdgcn.exp2.f32(float %1040), !dbg !15
  br i1 %43, label %1043, label %1044, !dbg !32

1043:                                             ; preds = %1038
  %address.thread789 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !32, !range !40, !frisk.register_lifetimes !1
  %address.local790 = lshr i32 %address.thread789, 1, !dbg !15
  %address.local791 = and i32 %address.local790, 96, !dbg !15
  %address.local792 = and i32 %address.thread789, 15, !dbg !15
  %address.local793 = or disjoint i32 %address.local791, %address.local792, !dbg !15
  %address.local794 = zext nneg i32 %address.local793 to i64, !dbg !15
  %address.local795 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !15
  store float %1041, ptr addrspace(3) %address.local795, align 4, !dbg !32
  %address.local796 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !15
  %address.local797 = getelementptr i8, ptr addrspace(3) %address.local796, i64 64, !dbg !15
  store float %1042, ptr addrspace(3) %address.local797, align 4, !dbg !32
  br label %1044, !dbg !32

1044:                                             ; preds = %1043, %1038
  fence syncscope("workgroup") release, !dbg !32
  call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %address.thread798 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !32, !range !40, !frisk.register_lifetimes !1
  %address.local799 = lshr i32 %address.thread798, 2, !dbg !13
  %address.local800 = and i32 %address.local799, 48, !dbg !13
  %address.local801 = and i32 %address.thread798, 15, !dbg !13
  %address.local802 = or disjoint i32 %address.local800, %address.local801, !dbg !13
  %address.local803 = zext nneg i32 %address.local802 to i64, !dbg !13
  %address.local804 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local803, !dbg !32
  %1045 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %lds.const.gep1746 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !32
  %1046 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  fence syncscope("workgroup") release, !dbg !32
  call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %1047 = add nuw nsw i64 %76, %15, !dbg !20
  %1048 = shl i64 %1047, 19, !dbg !20
  %1049 = and i64 %1048, 16252928, !dbg !20
  %address.local812 = zext nneg i32 %address.thread798 to i64, !dbg !12
  %address.local813 = shl nuw nsw i64 %address.local812, 3, !dbg !11
  %address.local814 = and i64 %address.local813, 1536, !dbg !11
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local814, !dbg !11
  %address.local816 = ptrtoint ptr addrspace(3) %address.local815 to i64, !dbg !11
  %address.local817 = trunc i64 %address.local816 to i32, !dbg !11
  %1050 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local817), !dbg !20
  %address.local818 = shl nuw nsw i64 %address.local812, 6, !dbg !20
  %address.local819 = and i64 %address.local818, 16128, !dbg !20
  %address.local820 = shl nuw nsw i64 %address.local812, 4, !dbg !20
  %address.local821 = and i64 %address.local820, 48, !dbg !20
  %1051 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %.idx1871 = add i64 %1049, %82, !dbg !20
  %1052 = shl i64 %.idx1871, 1, !dbg !20
  %1053 = or disjoint i64 %address.local819, %address.local821, !dbg !20
  %1054 = or disjoint i64 %1052, %1053, !dbg !20
  %dma.buffer.offset1773 = trunc i64 %1054 to i32, !dbg !20
  %1055 = insertelement <2 x i64> poison, i64 %1051, i64 0, !dbg !20
  %1056 = bitcast <2 x i64> %1055 to <4 x i32>, !dbg !20
  %dma.buffer.rsrc1774 = shufflevector <4 x i32> %1056, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !20
  %dma.buffer.lds1775 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1050), !dbg !20
  %1057 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1775, i32 %dma.buffer.offset1773, <4 x i32> %dma.buffer.rsrc1774) #10, !dbg !20
  %address.local824 = or disjoint i64 %address.local812, 256, !dbg !11
  %address.local825 = shl nuw nsw i64 %address.local824, 3, !dbg !11
  %address.local826 = and i64 %address.local825, 3584, !dbg !11
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local826, !dbg !11
  %address.local828 = ptrtoint ptr addrspace(3) %address.local827 to i64, !dbg !11
  %address.local829 = trunc i64 %address.local828 to i32, !dbg !11
  %1058 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local829), !dbg !20
  %address.local830 = shl nuw nsw i64 %address.local824, 6, !dbg !20
  %address.local831 = and i64 %address.local830, 32512, !dbg !20
  %address.local832 = shl nuw nsw i64 %address.local812, 4, !dbg !20
  %address.local833 = and i64 %address.local832, 48, !dbg !20
  %1059 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %.idx1874 = add i64 %1049, %82, !dbg !20
  %1060 = shl i64 %.idx1874, 1, !dbg !20
  %1061 = or disjoint i64 %1060, %address.local831, !dbg !20
  %1062 = or disjoint i64 %1061, %address.local833, !dbg !20
  %dma.buffer.offset1770 = trunc i64 %1062 to i32, !dbg !20
  %1063 = insertelement <2 x i64> poison, i64 %1059, i64 0, !dbg !20
  %1064 = bitcast <2 x i64> %1063 to <4 x i32>, !dbg !20
  %dma.buffer.rsrc1771 = shufflevector <4 x i32> %1064, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !20
  %dma.buffer.lds1772 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1058), !dbg !20
  %1065 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1772, i32 %dma.buffer.offset1770, <4 x i32> %dma.buffer.rsrc1771) #10, !dbg !20
  %1066 = extractvalue [2 x <1 x float>] %56, 0, !dbg !32
  %1067 = extractelement <1 x float> %1066, i64 0, !dbg !32
  %1068 = fmul float %1067, %1045, !dbg !32
  %1069 = extractvalue [2 x <1 x float>] %56, 1, !dbg !32
  %1070 = extractelement <1 x float> %1069, i64 0, !dbg !32
  %1071 = fmul float %1070, %1046, !dbg !32
  %1072 = extractelement <4 x float> %1014, i64 0, !dbg !33
  %1073 = call float @llvm.amdgcn.exp2.f32(float %1072), !dbg !33
  %1074 = insertelement <4 x float> poison, float %1073, i64 0, !dbg !35
  %1075 = extractelement <4 x float> %1014, i64 1, !dbg !33
  %1076 = call float @llvm.amdgcn.exp2.f32(float %1075), !dbg !33
  %1077 = insertelement <4 x float> %1074, float %1076, i64 1, !dbg !35
  %1078 = extractelement <4 x float> %1014, i64 2, !dbg !33
  %1079 = call float @llvm.amdgcn.exp2.f32(float %1078), !dbg !33
  %1080 = insertelement <4 x float> %1077, float %1079, i64 2, !dbg !35
  %1081 = extractelement <4 x float> %1014, i64 3, !dbg !33
  %1082 = call float @llvm.amdgcn.exp2.f32(float %1081), !dbg !33
  %1083 = insertelement <4 x float> %1080, float %1082, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1083) #9, !dbg !35, !frisk.register_lifetimes !1
  %1084 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1084) #9, !dbg !33, !frisk.register_lifetimes !1
  %1085 = extractelement <4 x float> %1015, i64 0, !dbg !33
  %1086 = call float @llvm.amdgcn.exp2.f32(float %1085), !dbg !33
  %1087 = insertelement <4 x float> poison, float %1086, i64 0, !dbg !35
  %local.reduce120 = fadd float %1073, %1086, !dbg !34, !frisk.local_reduction !1
  %1088 = extractelement <4 x float> %1015, i64 1, !dbg !33
  %1089 = call float @llvm.amdgcn.exp2.f32(float %1088), !dbg !33
  %1090 = insertelement <4 x float> %1087, float %1089, i64 1, !dbg !35
  %local.reduce121 = fadd float %1076, %1089, !dbg !34, !frisk.local_reduction !1
  %1091 = extractelement <4 x float> %1015, i64 2, !dbg !33
  %1092 = call float @llvm.amdgcn.exp2.f32(float %1091), !dbg !33
  %1093 = insertelement <4 x float> %1090, float %1092, i64 2, !dbg !35
  %local.reduce122 = fadd float %1079, %1092, !dbg !34, !frisk.local_reduction !1
  %1094 = extractelement <4 x float> %1015, i64 3, !dbg !33
  %1095 = call float @llvm.amdgcn.exp2.f32(float %1094), !dbg !33
  %1096 = insertelement <4 x float> %1093, float %1095, i64 3, !dbg !35
  %math.result228 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1096) #9, !dbg !35, !frisk.register_lifetimes !1
  %1097 = fptrunc <4 x float> %math.result228 to <4 x half>, !dbg !35
  %math.packed229 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1097) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce123 = fadd float %1082, %1095, !dbg !34, !frisk.local_reduction !1
  %1098 = extractelement <4 x float> %1016, i64 0, !dbg !33
  %1099 = call float @llvm.amdgcn.exp2.f32(float %1098), !dbg !33
  %1100 = insertelement <4 x float> poison, float %1099, i64 0, !dbg !35
  %local.reduce124 = fadd float %local.reduce120, %1099, !dbg !34, !frisk.local_reduction !1
  %1101 = extractelement <4 x float> %1016, i64 1, !dbg !33
  %1102 = call float @llvm.amdgcn.exp2.f32(float %1101), !dbg !33
  %1103 = insertelement <4 x float> %1100, float %1102, i64 1, !dbg !35
  %local.reduce125 = fadd float %local.reduce121, %1102, !dbg !34, !frisk.local_reduction !1
  %1104 = extractelement <4 x float> %1016, i64 2, !dbg !33
  %1105 = call float @llvm.amdgcn.exp2.f32(float %1104), !dbg !33
  %1106 = insertelement <4 x float> %1103, float %1105, i64 2, !dbg !35
  %local.reduce126 = fadd float %local.reduce122, %1105, !dbg !34, !frisk.local_reduction !1
  %1107 = extractelement <4 x float> %1016, i64 3, !dbg !33
  %1108 = call float @llvm.amdgcn.exp2.f32(float %1107), !dbg !33
  %1109 = insertelement <4 x float> %1106, float %1108, i64 3, !dbg !35
  %math.result230 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1109) #9, !dbg !35, !frisk.register_lifetimes !1
  %1110 = fptrunc <4 x float> %math.result230 to <4 x half>, !dbg !35
  %math.packed231 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1110) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce127 = fadd float %local.reduce123, %1108, !dbg !34, !frisk.local_reduction !1
  %1111 = extractelement <4 x float> %1017, i64 0, !dbg !33
  %1112 = call float @llvm.amdgcn.exp2.f32(float %1111), !dbg !33
  %1113 = insertelement <4 x float> poison, float %1112, i64 0, !dbg !35
  %local.reduce128 = fadd float %local.reduce124, %1112, !dbg !34, !frisk.local_reduction !1
  %1114 = extractelement <4 x float> %1017, i64 1, !dbg !33
  %1115 = call float @llvm.amdgcn.exp2.f32(float %1114), !dbg !33
  %1116 = insertelement <4 x float> %1113, float %1115, i64 1, !dbg !35
  %local.reduce129 = fadd float %local.reduce125, %1115, !dbg !34, !frisk.local_reduction !1
  %1117 = extractelement <4 x float> %1017, i64 2, !dbg !33
  %1118 = call float @llvm.amdgcn.exp2.f32(float %1117), !dbg !33
  %1119 = insertelement <4 x float> %1116, float %1118, i64 2, !dbg !35
  %local.reduce130 = fadd float %local.reduce126, %1118, !dbg !34, !frisk.local_reduction !1
  %1120 = extractelement <4 x float> %1017, i64 3, !dbg !33
  %1121 = call float @llvm.amdgcn.exp2.f32(float %1120), !dbg !33
  %1122 = insertelement <4 x float> %1119, float %1121, i64 3, !dbg !35
  %math.result232 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1122) #9, !dbg !35, !frisk.register_lifetimes !1
  %1123 = fptrunc <4 x float> %math.result232 to <4 x half>, !dbg !35
  %math.packed233 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1123) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce131 = fadd float %local.reduce127, %1121, !dbg !34, !frisk.local_reduction !1
  %1124 = extractelement <4 x float> %1018, i64 0, !dbg !33
  %1125 = call float @llvm.amdgcn.exp2.f32(float %1124), !dbg !33
  %1126 = insertelement <4 x float> poison, float %1125, i64 0, !dbg !35
  %local.reduce132 = fadd float %local.reduce128, %1125, !dbg !34, !frisk.local_reduction !1
  %1127 = extractelement <4 x float> %1018, i64 1, !dbg !33
  %1128 = call float @llvm.amdgcn.exp2.f32(float %1127), !dbg !33
  %1129 = insertelement <4 x float> %1126, float %1128, i64 1, !dbg !35
  %local.reduce133 = fadd float %local.reduce129, %1128, !dbg !34, !frisk.local_reduction !1
  %1130 = extractelement <4 x float> %1018, i64 2, !dbg !33
  %1131 = call float @llvm.amdgcn.exp2.f32(float %1130), !dbg !33
  %1132 = insertelement <4 x float> %1129, float %1131, i64 2, !dbg !35
  %local.reduce134 = fadd float %local.reduce130, %1131, !dbg !34, !frisk.local_reduction !1
  %1133 = extractelement <4 x float> %1018, i64 3, !dbg !33
  %1134 = call float @llvm.amdgcn.exp2.f32(float %1133), !dbg !33
  %1135 = insertelement <4 x float> %1132, float %1134, i64 3, !dbg !35
  %math.result234 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1135) #9, !dbg !35, !frisk.register_lifetimes !1
  %1136 = fptrunc <4 x float> %math.result234 to <4 x half>, !dbg !35
  %math.packed235 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1136) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce135 = fadd float %local.reduce131, %1134, !dbg !34, !frisk.local_reduction !1
  %1137 = extractelement <4 x float> %1019, i64 0, !dbg !33
  %1138 = call float @llvm.amdgcn.exp2.f32(float %1137), !dbg !33
  %1139 = insertelement <4 x float> poison, float %1138, i64 0, !dbg !35
  %local.reduce136 = fadd float %local.reduce132, %1138, !dbg !34, !frisk.local_reduction !1
  %1140 = extractelement <4 x float> %1019, i64 1, !dbg !33
  %1141 = call float @llvm.amdgcn.exp2.f32(float %1140), !dbg !33
  %1142 = insertelement <4 x float> %1139, float %1141, i64 1, !dbg !35
  %local.reduce137 = fadd float %local.reduce133, %1141, !dbg !34, !frisk.local_reduction !1
  %1143 = extractelement <4 x float> %1019, i64 2, !dbg !33
  %1144 = call float @llvm.amdgcn.exp2.f32(float %1143), !dbg !33
  %1145 = insertelement <4 x float> %1142, float %1144, i64 2, !dbg !35
  %local.reduce138 = fadd float %local.reduce134, %1144, !dbg !34, !frisk.local_reduction !1
  %1146 = extractelement <4 x float> %1019, i64 3, !dbg !33
  %1147 = call float @llvm.amdgcn.exp2.f32(float %1146), !dbg !33
  %1148 = insertelement <4 x float> %1145, float %1147, i64 3, !dbg !35
  %math.result236 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1148) #9, !dbg !35, !frisk.register_lifetimes !1
  %1149 = fptrunc <4 x float> %math.result236 to <4 x half>, !dbg !35
  %math.packed237 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1149) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce139 = fadd float %local.reduce135, %1147, !dbg !34, !frisk.local_reduction !1
  %1150 = extractelement <4 x float> %1020, i64 0, !dbg !33
  %1151 = call float @llvm.amdgcn.exp2.f32(float %1150), !dbg !33
  %1152 = insertelement <4 x float> poison, float %1151, i64 0, !dbg !35
  %local.reduce140 = fadd float %local.reduce136, %1151, !dbg !34, !frisk.local_reduction !1
  %1153 = extractelement <4 x float> %1020, i64 1, !dbg !33
  %1154 = call float @llvm.amdgcn.exp2.f32(float %1153), !dbg !33
  %1155 = insertelement <4 x float> %1152, float %1154, i64 1, !dbg !35
  %local.reduce141 = fadd float %local.reduce137, %1154, !dbg !34, !frisk.local_reduction !1
  %1156 = extractelement <4 x float> %1020, i64 2, !dbg !33
  %1157 = call float @llvm.amdgcn.exp2.f32(float %1156), !dbg !33
  %1158 = insertelement <4 x float> %1155, float %1157, i64 2, !dbg !35
  %local.reduce142 = fadd float %local.reduce138, %1157, !dbg !34, !frisk.local_reduction !1
  %1159 = extractelement <4 x float> %1020, i64 3, !dbg !33
  %1160 = call float @llvm.amdgcn.exp2.f32(float %1159), !dbg !33
  %1161 = insertelement <4 x float> %1158, float %1160, i64 3, !dbg !35
  %math.result238 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1161) #9, !dbg !35, !frisk.register_lifetimes !1
  %1162 = fptrunc <4 x float> %math.result238 to <4 x half>, !dbg !35
  %math.packed239 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1162) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce143 = fadd float %local.reduce139, %1160, !dbg !34, !frisk.local_reduction !1
  %1163 = extractelement <4 x float> %1021, i64 0, !dbg !33
  %1164 = call float @llvm.amdgcn.exp2.f32(float %1163), !dbg !33
  %1165 = insertelement <4 x float> poison, float %1164, i64 0, !dbg !35
  %local.reduce144 = fadd float %local.reduce140, %1164, !dbg !34, !frisk.local_reduction !1
  %1166 = extractelement <4 x float> %1021, i64 1, !dbg !33
  %1167 = call float @llvm.amdgcn.exp2.f32(float %1166), !dbg !33
  %1168 = insertelement <4 x float> %1165, float %1167, i64 1, !dbg !35
  %local.reduce145 = fadd float %local.reduce141, %1167, !dbg !34, !frisk.local_reduction !1
  %local.reduce148 = fadd float %local.reduce144, %local.reduce145, !dbg !34, !frisk.local_reduction !1
  %1169 = extractelement <4 x float> %1021, i64 2, !dbg !33
  %1170 = call float @llvm.amdgcn.exp2.f32(float %1169), !dbg !33
  %1171 = insertelement <4 x float> %1168, float %1170, i64 2, !dbg !35
  %local.reduce146 = fadd float %local.reduce142, %1170, !dbg !34, !frisk.local_reduction !1
  %1172 = extractelement <4 x float> %1021, i64 3, !dbg !33
  %1173 = call float @llvm.amdgcn.exp2.f32(float %1172), !dbg !33
  %1174 = insertelement <4 x float> %1171, float %1173, i64 3, !dbg !35
  %math.result240 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1174) #9, !dbg !35, !frisk.register_lifetimes !1
  %1175 = fptrunc <4 x float> %math.result240 to <4 x half>, !dbg !35
  %math.packed241 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1175) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce147 = fadd float %local.reduce143, %1173, !dbg !34, !frisk.local_reduction !1
  %local.reduce149 = fadd float %local.reduce146, %local.reduce147, !dbg !34, !frisk.local_reduction !1
  %1176 = fadd float %local.reduce148, %local.reduce149, !dbg !34, !frisk.local_reduction !1
  %1177 = extractelement <4 x float> %1023, i64 0, !dbg !33
  %1178 = call float @llvm.amdgcn.exp2.f32(float %1177), !dbg !33
  %1179 = insertelement <4 x float> poison, float %1178, i64 0, !dbg !35
  %1180 = extractelement <4 x float> %1023, i64 1, !dbg !33
  %1181 = call float @llvm.amdgcn.exp2.f32(float %1180), !dbg !33
  %1182 = insertelement <4 x float> %1179, float %1181, i64 1, !dbg !35
  %1183 = extractelement <4 x float> %1023, i64 2, !dbg !33
  %1184 = call float @llvm.amdgcn.exp2.f32(float %1183), !dbg !33
  %1185 = insertelement <4 x float> %1182, float %1184, i64 2, !dbg !35
  %1186 = extractelement <4 x float> %1023, i64 3, !dbg !33
  %1187 = call float @llvm.amdgcn.exp2.f32(float %1186), !dbg !33
  %1188 = insertelement <4 x float> %1185, float %1187, i64 3, !dbg !35
  %math.result242 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1188) #9, !dbg !35, !frisk.register_lifetimes !1
  %1189 = fptrunc <4 x float> %math.result242 to <4 x half>, !dbg !35
  %math.packed243 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1189) #9, !dbg !33, !frisk.register_lifetimes !1
  %1190 = extractelement <4 x float> %1024, i64 0, !dbg !33
  %1191 = call float @llvm.amdgcn.exp2.f32(float %1190), !dbg !33
  %1192 = insertelement <4 x float> poison, float %1191, i64 0, !dbg !35
  %local.reduce = fadd float %1178, %1191, !dbg !34, !frisk.local_reduction !1
  %1193 = extractelement <4 x float> %1024, i64 1, !dbg !33
  %1194 = call float @llvm.amdgcn.exp2.f32(float %1193), !dbg !33
  %1195 = insertelement <4 x float> %1192, float %1194, i64 1, !dbg !35
  %local.reduce90 = fadd float %1181, %1194, !dbg !34, !frisk.local_reduction !1
  %1196 = extractelement <4 x float> %1024, i64 2, !dbg !33
  %1197 = call float @llvm.amdgcn.exp2.f32(float %1196), !dbg !33
  %1198 = insertelement <4 x float> %1195, float %1197, i64 2, !dbg !35
  %local.reduce91 = fadd float %1184, %1197, !dbg !34, !frisk.local_reduction !1
  %1199 = extractelement <4 x float> %1024, i64 3, !dbg !33
  %1200 = call float @llvm.amdgcn.exp2.f32(float %1199), !dbg !33
  %1201 = insertelement <4 x float> %1198, float %1200, i64 3, !dbg !35
  %math.result244 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1201) #9, !dbg !35, !frisk.register_lifetimes !1
  %1202 = fptrunc <4 x float> %math.result244 to <4 x half>, !dbg !35
  %math.packed245 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1202) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce92 = fadd float %1187, %1200, !dbg !34, !frisk.local_reduction !1
  %1203 = extractelement <4 x float> %1025, i64 0, !dbg !33
  %1204 = call float @llvm.amdgcn.exp2.f32(float %1203), !dbg !33
  %1205 = insertelement <4 x float> poison, float %1204, i64 0, !dbg !35
  %local.reduce93 = fadd float %local.reduce, %1204, !dbg !34, !frisk.local_reduction !1
  %1206 = extractelement <4 x float> %1025, i64 1, !dbg !33
  %1207 = call float @llvm.amdgcn.exp2.f32(float %1206), !dbg !33
  %1208 = insertelement <4 x float> %1205, float %1207, i64 1, !dbg !35
  %local.reduce94 = fadd float %local.reduce90, %1207, !dbg !34, !frisk.local_reduction !1
  %1209 = extractelement <4 x float> %1025, i64 2, !dbg !33
  %1210 = call float @llvm.amdgcn.exp2.f32(float %1209), !dbg !33
  %1211 = insertelement <4 x float> %1208, float %1210, i64 2, !dbg !35
  %local.reduce95 = fadd float %local.reduce91, %1210, !dbg !34, !frisk.local_reduction !1
  %1212 = extractelement <4 x float> %1025, i64 3, !dbg !33
  %1213 = call float @llvm.amdgcn.exp2.f32(float %1212), !dbg !33
  %1214 = insertelement <4 x float> %1211, float %1213, i64 3, !dbg !35
  %math.result246 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1214) #9, !dbg !35, !frisk.register_lifetimes !1
  %1215 = fptrunc <4 x float> %math.result246 to <4 x half>, !dbg !35
  %math.packed247 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1215) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce96 = fadd float %local.reduce92, %1213, !dbg !34, !frisk.local_reduction !1
  %1216 = extractelement <4 x float> %1026, i64 0, !dbg !33
  %1217 = call float @llvm.amdgcn.exp2.f32(float %1216), !dbg !33
  %1218 = insertelement <4 x float> poison, float %1217, i64 0, !dbg !35
  %local.reduce97 = fadd float %local.reduce93, %1217, !dbg !34, !frisk.local_reduction !1
  %1219 = extractelement <4 x float> %1026, i64 1, !dbg !33
  %1220 = call float @llvm.amdgcn.exp2.f32(float %1219), !dbg !33
  %1221 = insertelement <4 x float> %1218, float %1220, i64 1, !dbg !35
  %local.reduce98 = fadd float %local.reduce94, %1220, !dbg !34, !frisk.local_reduction !1
  %1222 = extractelement <4 x float> %1026, i64 2, !dbg !33
  %1223 = call float @llvm.amdgcn.exp2.f32(float %1222), !dbg !33
  %1224 = insertelement <4 x float> %1221, float %1223, i64 2, !dbg !35
  %local.reduce99 = fadd float %local.reduce95, %1223, !dbg !34, !frisk.local_reduction !1
  %1225 = extractelement <4 x float> %1026, i64 3, !dbg !33
  %1226 = call float @llvm.amdgcn.exp2.f32(float %1225), !dbg !33
  %1227 = insertelement <4 x float> %1224, float %1226, i64 3, !dbg !35
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1227) #9, !dbg !35, !frisk.register_lifetimes !1
  %1228 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !35
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1228) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce100 = fadd float %local.reduce96, %1226, !dbg !34, !frisk.local_reduction !1
  %1229 = extractelement <4 x float> %1027, i64 0, !dbg !33
  %1230 = call float @llvm.amdgcn.exp2.f32(float %1229), !dbg !33
  %1231 = insertelement <4 x float> poison, float %1230, i64 0, !dbg !35
  %local.reduce101 = fadd float %local.reduce97, %1230, !dbg !34, !frisk.local_reduction !1
  %1232 = extractelement <4 x float> %1027, i64 1, !dbg !33
  %1233 = call float @llvm.amdgcn.exp2.f32(float %1232), !dbg !33
  %1234 = insertelement <4 x float> %1231, float %1233, i64 1, !dbg !35
  %local.reduce102 = fadd float %local.reduce98, %1233, !dbg !34, !frisk.local_reduction !1
  %1235 = extractelement <4 x float> %1027, i64 2, !dbg !33
  %1236 = call float @llvm.amdgcn.exp2.f32(float %1235), !dbg !33
  %1237 = insertelement <4 x float> %1234, float %1236, i64 2, !dbg !35
  %local.reduce103 = fadd float %local.reduce99, %1236, !dbg !34, !frisk.local_reduction !1
  %1238 = extractelement <4 x float> %1027, i64 3, !dbg !33
  %1239 = call float @llvm.amdgcn.exp2.f32(float %1238), !dbg !33
  %1240 = insertelement <4 x float> %1237, float %1239, i64 3, !dbg !35
  %math.result250 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1240) #9, !dbg !35, !frisk.register_lifetimes !1
  %1241 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !35
  %math.packed251 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1241) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce104 = fadd float %local.reduce100, %1239, !dbg !34, !frisk.local_reduction !1
  %1242 = extractelement <4 x float> %1028, i64 0, !dbg !33
  %1243 = call float @llvm.amdgcn.exp2.f32(float %1242), !dbg !33
  %1244 = insertelement <4 x float> poison, float %1243, i64 0, !dbg !35
  %local.reduce105 = fadd float %local.reduce101, %1243, !dbg !34, !frisk.local_reduction !1
  %1245 = extractelement <4 x float> %1028, i64 1, !dbg !33
  %1246 = call float @llvm.amdgcn.exp2.f32(float %1245), !dbg !33
  %1247 = insertelement <4 x float> %1244, float %1246, i64 1, !dbg !35
  %local.reduce106 = fadd float %local.reduce102, %1246, !dbg !34, !frisk.local_reduction !1
  %1248 = extractelement <4 x float> %1028, i64 2, !dbg !33
  %1249 = call float @llvm.amdgcn.exp2.f32(float %1248), !dbg !33
  %1250 = insertelement <4 x float> %1247, float %1249, i64 2, !dbg !35
  %local.reduce107 = fadd float %local.reduce103, %1249, !dbg !34, !frisk.local_reduction !1
  %1251 = extractelement <4 x float> %1028, i64 3, !dbg !33
  %1252 = call float @llvm.amdgcn.exp2.f32(float %1251), !dbg !33
  %1253 = insertelement <4 x float> %1250, float %1252, i64 3, !dbg !35
  %math.result252 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1253) #9, !dbg !35, !frisk.register_lifetimes !1
  %1254 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !35
  %math.packed253 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1254) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce108 = fadd float %local.reduce104, %1252, !dbg !34, !frisk.local_reduction !1
  %1255 = extractelement <4 x float> %1029, i64 0, !dbg !33
  %1256 = call float @llvm.amdgcn.exp2.f32(float %1255), !dbg !33
  %1257 = insertelement <4 x float> poison, float %1256, i64 0, !dbg !35
  %local.reduce109 = fadd float %local.reduce105, %1256, !dbg !34, !frisk.local_reduction !1
  %1258 = extractelement <4 x float> %1029, i64 1, !dbg !33
  %1259 = call float @llvm.amdgcn.exp2.f32(float %1258), !dbg !33
  %1260 = insertelement <4 x float> %1257, float %1259, i64 1, !dbg !35
  %local.reduce110 = fadd float %local.reduce106, %1259, !dbg !34, !frisk.local_reduction !1
  %1261 = extractelement <4 x float> %1029, i64 2, !dbg !33
  %1262 = call float @llvm.amdgcn.exp2.f32(float %1261), !dbg !33
  %1263 = insertelement <4 x float> %1260, float %1262, i64 2, !dbg !35
  %local.reduce111 = fadd float %local.reduce107, %1262, !dbg !34, !frisk.local_reduction !1
  %1264 = extractelement <4 x float> %1029, i64 3, !dbg !33
  %1265 = call float @llvm.amdgcn.exp2.f32(float %1264), !dbg !33
  %1266 = insertelement <4 x float> %1263, float %1265, i64 3, !dbg !35
  %math.result254 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1266) #9, !dbg !35, !frisk.register_lifetimes !1
  %1267 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !35
  %math.packed255 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1267) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce112 = fadd float %local.reduce108, %1265, !dbg !34, !frisk.local_reduction !1
  %1268 = extractelement <4 x float> %1030, i64 0, !dbg !33
  %1269 = call float @llvm.amdgcn.exp2.f32(float %1268), !dbg !33
  %1270 = insertelement <4 x float> poison, float %1269, i64 0, !dbg !35
  %local.reduce113 = fadd float %local.reduce109, %1269, !dbg !34, !frisk.local_reduction !1
  %1271 = extractelement <4 x float> %1030, i64 1, !dbg !33
  %1272 = call float @llvm.amdgcn.exp2.f32(float %1271), !dbg !33
  %1273 = insertelement <4 x float> %1270, float %1272, i64 1, !dbg !35
  %local.reduce114 = fadd float %local.reduce110, %1272, !dbg !34, !frisk.local_reduction !1
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !34, !frisk.local_reduction !1
  %1274 = extractelement <4 x float> %1030, i64 2, !dbg !33
  %1275 = call float @llvm.amdgcn.exp2.f32(float %1274), !dbg !33
  %1276 = insertelement <4 x float> %1273, float %1275, i64 2, !dbg !35
  %local.reduce115 = fadd float %local.reduce111, %1275, !dbg !34, !frisk.local_reduction !1
  %1277 = extractelement <4 x float> %1030, i64 3, !dbg !33
  %1278 = call float @llvm.amdgcn.exp2.f32(float %1277), !dbg !33
  %1279 = insertelement <4 x float> %1276, float %1278, i64 3, !dbg !35
  %math.result256 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1279) #9, !dbg !35, !frisk.register_lifetimes !1
  %1280 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !35
  %math.packed257 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1280) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce116 = fadd float %local.reduce112, %1278, !dbg !34, !frisk.local_reduction !1
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !34, !frisk.local_reduction !1
  %1281 = fadd float %local.reduce117, %local.reduce118, !dbg !34, !frisk.local_reduction !1
  %1282 = bitcast float %1176 to i32, !dbg !34
  %1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1282), !dbg !34
  %1284 = bitcast i32 %1283 to float, !dbg !34
  %1285 = fadd float %1176, %1284, !dbg !34
  %1286 = bitcast float %1285 to i32, !dbg !34
  %1287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1286), !dbg !34
  %1288 = bitcast i32 %1287 to float, !dbg !34
  %1289 = fadd float %1285, %1288, !dbg !34
  %1290 = fadd float %1289, %1068, !dbg !34
  %1291 = bitcast float %1281 to i32, !dbg !34
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1291), !dbg !34
  %1293 = bitcast i32 %1292 to float, !dbg !34
  %1294 = fadd float %1281, %1293, !dbg !34
  %1295 = bitcast float %1294 to i32, !dbg !34
  %1296 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1295), !dbg !34
  %1297 = bitcast i32 %1296 to float, !dbg !34
  %1298 = fadd float %1294, %1297, !dbg !34
  %1299 = fadd float %1298, %1071, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local836 = shl nuw nsw i32 %address.thread798, 3, !dbg !16
  %address.local837 = and i32 %address.local836, 504, !dbg !16
  %address.local838 = zext nneg i32 %address.local837 to i64, !dbg !16
  %address.local839 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %1300 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local839, i32 0) #0, !dbg !16
  %1301 = bitcast <4 x i32> %1300 to <8 x half>, !dbg !16
  %address.local840 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local840, i64 1024, !dbg !16
  %1302 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #0, !dbg !16
  %1303 = bitcast <4 x i32> %1302 to <8 x half>, !dbg !16
  %address.local842 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local843 = getelementptr i8, ptr addrspace(3) %address.local842, i64 2048, !dbg !16
  %1304 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local843, i32 0) #0, !dbg !16
  %1305 = bitcast <4 x i32> %1304 to <8 x half>, !dbg !16
  %address.local844 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local845 = getelementptr i8, ptr addrspace(3) %address.local844, i64 3072, !dbg !16
  %1306 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local845, i32 0) #0, !dbg !16
  %1307 = bitcast <4 x i32> %1306 to <8 x half>, !dbg !16
  %address.local846 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local847 = getelementptr i8, ptr addrspace(3) %address.local846, i64 4096, !dbg !16
  %1308 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local847, i32 0) #0, !dbg !16
  %1309 = bitcast <4 x i32> %1308 to <8 x half>, !dbg !16
  %address.local848 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local849 = getelementptr i8, ptr addrspace(3) %address.local848, i64 5120, !dbg !16
  %1310 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local849, i32 0) #0, !dbg !16
  %1311 = bitcast <4 x i32> %1310 to <8 x half>, !dbg !16
  %address.local850 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local851 = getelementptr i8, ptr addrspace(3) %address.local850, i64 6144, !dbg !16
  %1312 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #0, !dbg !16
  %1313 = bitcast <4 x i32> %1312 to <8 x half>, !dbg !16
  %address.local852 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local853 = getelementptr i8, ptr addrspace(3) %address.local852, i64 7168, !dbg !16
  %1314 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local853, i32 0) #0, !dbg !16
  %1315 = bitcast <4 x i32> %1314 to <8 x half>, !dbg !16
  %1316 = shufflevector <8 x half> %1301, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1316, <4 x float> zeroinitializer) #9, !dbg !16
  %1318 = shufflevector <8 x half> %1303, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1318, <4 x float> %1317) #9, !dbg !16
  %1320 = shufflevector <8 x half> %1305, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1320, <4 x float> %1319) #9, !dbg !16
  %1322 = shufflevector <8 x half> %1307, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1322, <4 x float> %1321) #9, !dbg !16
  %1324 = shufflevector <8 x half> %1309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1324, <4 x float> %1323) #9, !dbg !16
  %1326 = shufflevector <8 x half> %1311, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1326, <4 x float> %1325) #9, !dbg !16
  %1328 = shufflevector <8 x half> %1313, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1328, <4 x float> %1327) #9, !dbg !16
  %1330 = shufflevector <8 x half> %1315, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1330, <4 x float> %1329) #9, !dbg !16
  %1332 = extractelement <4 x float> %1331, i64 0, !dbg !16
  %1333 = extractelement <4 x float> %1331, i64 1, !dbg !16
  %1334 = extractelement <4 x float> %1331, i64 2, !dbg !16
  %1335 = extractelement <4 x float> %1331, i64 3, !dbg !16
  %1336 = select i1 %.not, float %1333, float %1332, !dbg !16
  %1337 = bitcast float %1336 to i32, !dbg !16
  %1338 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1337), !dbg !16
  %1339 = bitcast i32 %1338 to float, !dbg !16
  %1340 = select i1 %.not, float %1332, float %1339, !dbg !16
  %1341 = select i1 %.not, float %1339, float %1333, !dbg !16
  %1342 = select i1 %.not, float %1335, float %1334, !dbg !16
  %1343 = bitcast float %1342 to i32, !dbg !16
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1343), !dbg !16
  %1345 = bitcast i32 %1344 to float, !dbg !16
  %1346 = select i1 %.not, float %1334, float %1345, !dbg !16
  %1347 = select i1 %.not, float %1345, float %1335, !dbg !16
  %1348 = select i1 %.not1, float %1346, float %1340, !dbg !16
  %1349 = bitcast float %1348 to i32, !dbg !16
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1349), !dbg !16
  %1351 = bitcast i32 %1350 to float, !dbg !16
  %1352 = select i1 %.not1, float %1340, float %1351, !dbg !16
  %1353 = select i1 %.not1, float %1351, float %1346, !dbg !16
  %1354 = select i1 %.not1, float %1347, float %1341, !dbg !16
  %1355 = bitcast float %1354 to i32, !dbg !16
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1355), !dbg !16
  %1357 = bitcast i32 %1356 to float, !dbg !16
  %1358 = select i1 %.not1, float %1341, float %1357, !dbg !16
  %1359 = select i1 %.not1, float %1357, float %1347, !dbg !16
  %1360 = shufflevector <8 x half> %1301, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1360, <4 x float> zeroinitializer) #9, !dbg !16
  %1362 = shufflevector <8 x half> %1303, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1362, <4 x float> %1361) #9, !dbg !16
  %1364 = shufflevector <8 x half> %1305, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1364, <4 x float> %1363) #9, !dbg !16
  %1366 = shufflevector <8 x half> %1307, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1366, <4 x float> %1365) #9, !dbg !16
  %1368 = shufflevector <8 x half> %1309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1368, <4 x float> %1367) #9, !dbg !16
  %1370 = shufflevector <8 x half> %1311, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1370, <4 x float> %1369) #9, !dbg !16
  %1372 = shufflevector <8 x half> %1313, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1372, <4 x float> %1371) #9, !dbg !16
  %1374 = shufflevector <8 x half> %1315, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1374, <4 x float> %1373) #9, !dbg !16
  %1376 = extractelement <4 x float> %1375, i64 0, !dbg !16
  %1377 = extractelement <4 x float> %1375, i64 1, !dbg !16
  %1378 = extractelement <4 x float> %1375, i64 2, !dbg !16
  %1379 = extractelement <4 x float> %1375, i64 3, !dbg !16
  %1380 = select i1 %.not, float %1377, float %1376, !dbg !16
  %1381 = bitcast float %1380 to i32, !dbg !16
  %1382 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1381), !dbg !16
  %1383 = bitcast i32 %1382 to float, !dbg !16
  %1384 = select i1 %.not, float %1376, float %1383, !dbg !16
  %1385 = select i1 %.not, float %1383, float %1377, !dbg !16
  %1386 = select i1 %.not, float %1379, float %1378, !dbg !16
  %1387 = bitcast float %1386 to i32, !dbg !16
  %1388 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1387), !dbg !16
  %1389 = bitcast i32 %1388 to float, !dbg !16
  %1390 = select i1 %.not, float %1378, float %1389, !dbg !16
  %1391 = select i1 %.not, float %1389, float %1379, !dbg !16
  %1392 = select i1 %.not1, float %1390, float %1384, !dbg !16
  %1393 = bitcast float %1392 to i32, !dbg !16
  %1394 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1393), !dbg !16
  %1395 = bitcast i32 %1394 to float, !dbg !16
  %1396 = select i1 %.not1, float %1384, float %1395, !dbg !16
  %1397 = select i1 %.not1, float %1395, float %1390, !dbg !16
  %1398 = select i1 %.not1, float %1391, float %1385, !dbg !16
  %1399 = bitcast float %1398 to i32, !dbg !16
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1399), !dbg !16
  %1401 = bitcast i32 %1400 to float, !dbg !16
  %1402 = select i1 %.not1, float %1385, float %1401, !dbg !16
  %1403 = select i1 %.not1, float %1401, float %1391, !dbg !16
  %1404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1316, <4 x float> zeroinitializer) #9, !dbg !16
  %1405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1318, <4 x float> %1404) #9, !dbg !16
  %1406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1320, <4 x float> %1405) #9, !dbg !16
  %1407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1322, <4 x float> %1406) #9, !dbg !16
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1324, <4 x float> %1407) #9, !dbg !16
  %1409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1326, <4 x float> %1408) #9, !dbg !16
  %1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1328, <4 x float> %1409) #9, !dbg !16
  %1411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1330, <4 x float> %1410) #9, !dbg !16
  %1412 = extractelement <4 x float> %1411, i64 0, !dbg !16
  %1413 = extractelement <4 x float> %1411, i64 1, !dbg !16
  %1414 = extractelement <4 x float> %1411, i64 2, !dbg !16
  %1415 = extractelement <4 x float> %1411, i64 3, !dbg !16
  %1416 = select i1 %.not, float %1413, float %1412, !dbg !16
  %1417 = bitcast float %1416 to i32, !dbg !16
  %1418 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1417), !dbg !16
  %1419 = bitcast i32 %1418 to float, !dbg !16
  %1420 = select i1 %.not, float %1412, float %1419, !dbg !16
  %1421 = select i1 %.not, float %1419, float %1413, !dbg !16
  %1422 = select i1 %.not, float %1415, float %1414, !dbg !16
  %1423 = bitcast float %1422 to i32, !dbg !16
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1423), !dbg !16
  %1425 = bitcast i32 %1424 to float, !dbg !16
  %1426 = select i1 %.not, float %1414, float %1425, !dbg !16
  %1427 = select i1 %.not, float %1425, float %1415, !dbg !16
  %1428 = select i1 %.not1, float %1426, float %1420, !dbg !16
  %1429 = bitcast float %1428 to i32, !dbg !16
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1429), !dbg !16
  %1431 = bitcast i32 %1430 to float, !dbg !16
  %1432 = select i1 %.not1, float %1420, float %1431, !dbg !16
  %1433 = select i1 %.not1, float %1431, float %1426, !dbg !16
  %1434 = select i1 %.not1, float %1427, float %1421, !dbg !16
  %1435 = bitcast float %1434 to i32, !dbg !16
  %1436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1435), !dbg !16
  %1437 = bitcast i32 %1436 to float, !dbg !16
  %1438 = select i1 %.not1, float %1421, float %1437, !dbg !16
  %1439 = select i1 %.not1, float %1437, float %1427, !dbg !16
  %1440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1360, <4 x float> zeroinitializer) #9, !dbg !16
  %1441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1362, <4 x float> %1440) #9, !dbg !16
  %1442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1364, <4 x float> %1441) #9, !dbg !16
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1366, <4 x float> %1442) #9, !dbg !16
  %1444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1368, <4 x float> %1443) #9, !dbg !16
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1370, <4 x float> %1444) #9, !dbg !16
  %1446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1372, <4 x float> %1445) #9, !dbg !16
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1374, <4 x float> %1446) #9, !dbg !16
  %1448 = extractelement <4 x float> %1447, i64 0, !dbg !16
  %1449 = extractelement <4 x float> %1447, i64 1, !dbg !16
  %1450 = extractelement <4 x float> %1447, i64 2, !dbg !16
  %1451 = extractelement <4 x float> %1447, i64 3, !dbg !16
  %1452 = select i1 %.not, float %1449, float %1448, !dbg !16
  %1453 = bitcast float %1452 to i32, !dbg !16
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1453), !dbg !16
  %1455 = bitcast i32 %1454 to float, !dbg !16
  %1456 = select i1 %.not, float %1448, float %1455, !dbg !16
  %1457 = select i1 %.not, float %1455, float %1449, !dbg !16
  %1458 = select i1 %.not, float %1451, float %1450, !dbg !16
  %1459 = bitcast float %1458 to i32, !dbg !16
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1459), !dbg !16
  %1461 = bitcast i32 %1460 to float, !dbg !16
  %1462 = select i1 %.not, float %1450, float %1461, !dbg !16
  %1463 = select i1 %.not, float %1461, float %1451, !dbg !16
  %1464 = select i1 %.not1, float %1462, float %1456, !dbg !16
  %1465 = bitcast float %1464 to i32, !dbg !16
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1465), !dbg !16
  %1467 = bitcast i32 %1466 to float, !dbg !16
  %1468 = select i1 %.not1, float %1456, float %1467, !dbg !16
  %1469 = select i1 %.not1, float %1467, float %1462, !dbg !16
  %1470 = select i1 %.not1, float %1463, float %1457, !dbg !16
  %1471 = bitcast float %1470 to i32, !dbg !16
  %1472 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1471), !dbg !16
  %1473 = bitcast i32 %1472 to float, !dbg !16
  %1474 = select i1 %.not1, float %1457, float %1473, !dbg !16
  %1475 = select i1 %.not1, float %1473, float %1463, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %1476, label %1477, !dbg !36

1476:                                             ; preds = %1044
  %address.thread854 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local855 = lshr i32 %address.thread854, 1, !dbg !15
  %address.local856 = and i32 %address.local855, 96, !dbg !15
  %address.local857 = and i32 %address.thread854, 15, !dbg !15
  %address.local858 = or disjoint i32 %address.local856, %address.local857, !dbg !15
  %address.local859 = zext nneg i32 %address.local858 to i64, !dbg !15
  %address.local860 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !15
  store float %1041, ptr addrspace(3) %address.local860, align 4, !dbg !36
  %address.local861 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !15
  %address.local862 = getelementptr i8, ptr addrspace(3) %address.local861, i64 64, !dbg !15
  store float %1042, ptr addrspace(3) %address.local862, align 4, !dbg !36
  br label %1477, !dbg !36

1477:                                             ; preds = %1476, %1044
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread863 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %1478 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !36
  %lds.const.gep1748 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %1479 = load float, ptr addrspace(3) %lds.const.gep1748, align 4, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %1480 = add nuw nsw i64 %76, %15, !dbg !20
  %1481 = shl i64 %1480, 19, !dbg !20
  %1482 = and i64 %1481, 16252928, !dbg !20
  %1483 = add i64 %1482, %82, !dbg !20
  %address.local877 = zext nneg i32 %address.thread863 to i64, !dbg !12
  %address.local878 = shl nuw nsw i64 %address.local877, 3, !dbg !11
  %address.local879 = and i64 %address.local878, 1536, !dbg !11
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local879, !dbg !11
  %address.local881 = ptrtoint ptr addrspace(3) %address.local880 to i64, !dbg !11
  %address.local882 = trunc i64 %address.local881 to i32, !dbg !11
  %1484 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local882), !dbg !20
  %address.local883 = shl nuw nsw i64 %address.local877, 6, !dbg !20
  %address.local884 = and i64 %address.local883, 16128, !dbg !20
  %address.local885 = shl nuw nsw i64 %address.local877, 4, !dbg !20
  %address.local886 = and i64 %address.local885, 48, !dbg !20
  %1485 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %.idx = shl i64 %1483, 1, !dbg !20
  %1486 = or disjoint i64 %address.local884, %address.local886, !dbg !20
  %1487 = or disjoint i64 %.idx, %1486, !dbg !20
  %1488 = trunc i64 %1487 to i32, !dbg !20
  %dma.buffer.offset1767 = or i32 %1488, 64, !dbg !20
  %1489 = insertelement <2 x i64> poison, i64 %1485, i64 0, !dbg !20
  %1490 = bitcast <2 x i64> %1489 to <4 x i32>, !dbg !20
  %dma.buffer.rsrc1768 = shufflevector <4 x i32> %1490, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !20
  %dma.buffer.lds1769 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1484), !dbg !20
  %1491 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1769, i32 %dma.buffer.offset1767, <4 x i32> %dma.buffer.rsrc1768) #10, !dbg !20
  %address.local889 = or disjoint i64 %address.local877, 256, !dbg !11
  %address.local890 = shl nuw nsw i64 %address.local889, 3, !dbg !11
  %address.local891 = and i64 %address.local890, 3584, !dbg !11
  %address.local892 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local891, !dbg !11
  %address.local893 = ptrtoint ptr addrspace(3) %address.local892 to i64, !dbg !11
  %address.local894 = trunc i64 %address.local893 to i32, !dbg !11
  %1492 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !20
  %address.local895 = shl nuw nsw i64 %address.local889, 6, !dbg !20
  %address.local896 = and i64 %address.local895, 32512, !dbg !20
  %address.local897 = shl nuw nsw i64 %address.local877, 4, !dbg !20
  %address.local898 = and i64 %address.local897, 48, !dbg !20
  %1493 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %.idx1876 = shl i64 %1483, 1, !dbg !20
  %1494 = or disjoint i64 %.idx1876, %address.local896, !dbg !20
  %1495 = or disjoint i64 %1494, %address.local898, !dbg !20
  %1496 = trunc i64 %1495 to i32, !dbg !20
  %dma.buffer.offset1764 = or i32 %1496, 64, !dbg !20
  %1497 = insertelement <2 x i64> poison, i64 %1493, i64 0, !dbg !20
  %1498 = bitcast <2 x i64> %1497 to <4 x i32>, !dbg !20
  %dma.buffer.rsrc1765 = shufflevector <4 x i32> %1498, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !20
  %dma.buffer.lds1766 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1492), !dbg !20
  %1499 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1766, i32 %dma.buffer.offset1764, <4 x i32> %dma.buffer.rsrc1765) #10, !dbg !20
  %1500 = extractvalue [2 x <8 x float>] %55, 0, !dbg !36
  %1501 = shufflevector <8 x float> %1500, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1502 = insertelement <4 x float> poison, float %1478, i64 0, !dbg !36
  %1503 = shufflevector <4 x float> %1502, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1504 = fmul <4 x float> %1501, %1503, !dbg !36
  %1505 = insertelement <4 x float> poison, float %1352, i64 0, !dbg !37
  %1506 = insertelement <4 x float> %1505, float %1358, i64 1, !dbg !37
  %1507 = insertelement <4 x float> %1506, float %1353, i64 2, !dbg !37
  %1508 = insertelement <4 x float> %1507, float %1359, i64 3, !dbg !37
  %1509 = fadd <4 x float> %1504, %1508, !dbg !37
  %1510 = shufflevector <8 x float> %1500, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1511 = fmul <4 x float> %1510, %1503, !dbg !36
  %1512 = insertelement <4 x float> poison, float %1396, i64 0, !dbg !37
  %1513 = insertelement <4 x float> %1512, float %1402, i64 1, !dbg !37
  %1514 = insertelement <4 x float> %1513, float %1397, i64 2, !dbg !37
  %1515 = insertelement <4 x float> %1514, float %1403, i64 3, !dbg !37
  %1516 = fadd <4 x float> %1511, %1515, !dbg !37
  %1517 = extractvalue [2 x <8 x float>] %55, 1, !dbg !36
  %1518 = shufflevector <8 x float> %1517, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1519 = insertelement <4 x float> poison, float %1479, i64 0, !dbg !36
  %1520 = shufflevector <4 x float> %1519, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1521 = fmul <4 x float> %1518, %1520, !dbg !36
  %1522 = insertelement <4 x float> poison, float %1432, i64 0, !dbg !37
  %1523 = insertelement <4 x float> %1522, float %1438, i64 1, !dbg !37
  %1524 = insertelement <4 x float> %1523, float %1433, i64 2, !dbg !37
  %1525 = insertelement <4 x float> %1524, float %1439, i64 3, !dbg !37
  %1526 = fadd <4 x float> %1521, %1525, !dbg !37
  %1527 = shufflevector <8 x float> %1517, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1528 = fmul <4 x float> %1527, %1520, !dbg !36
  %1529 = insertelement <4 x float> poison, float %1468, i64 0, !dbg !37
  %1530 = insertelement <4 x float> %1529, float %1474, i64 1, !dbg !37
  %1531 = insertelement <4 x float> %1530, float %1469, i64 2, !dbg !37
  %1532 = insertelement <4 x float> %1531, float %1475, i64 3, !dbg !37
  %1533 = fadd <4 x float> %1528, %1532, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local901 = shl nuw nsw i32 %address.thread863, 3, !dbg !16
  %address.local902 = and i32 %address.local901, 504, !dbg !16
  %address.local903 = zext nneg i32 %address.local902 to i64, !dbg !16
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %1534 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #0, !dbg !16
  %1535 = bitcast <4 x i32> %1534 to <8 x half>, !dbg !16
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 1024, !dbg !16
  %1536 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !16
  %1537 = bitcast <4 x i32> %1536 to <8 x half>, !dbg !16
  %address.local907 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local908 = getelementptr i8, ptr addrspace(3) %address.local907, i64 2048, !dbg !16
  %1538 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local908, i32 0) #0, !dbg !16
  %1539 = bitcast <4 x i32> %1538 to <8 x half>, !dbg !16
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local910 = getelementptr i8, ptr addrspace(3) %address.local909, i64 3072, !dbg !16
  %1540 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local910, i32 0) #0, !dbg !16
  %1541 = bitcast <4 x i32> %1540 to <8 x half>, !dbg !16
  %address.local911 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local912 = getelementptr i8, ptr addrspace(3) %address.local911, i64 4096, !dbg !16
  %1542 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local912, i32 0) #0, !dbg !16
  %1543 = bitcast <4 x i32> %1542 to <8 x half>, !dbg !16
  %address.local913 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local914 = getelementptr i8, ptr addrspace(3) %address.local913, i64 5120, !dbg !16
  %1544 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local914, i32 0) #0, !dbg !16
  %1545 = bitcast <4 x i32> %1544 to <8 x half>, !dbg !16
  %address.local915 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local916 = getelementptr i8, ptr addrspace(3) %address.local915, i64 6144, !dbg !16
  %1546 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local916, i32 0) #0, !dbg !16
  %1547 = bitcast <4 x i32> %1546 to <8 x half>, !dbg !16
  %address.local917 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local918 = getelementptr i8, ptr addrspace(3) %address.local917, i64 7168, !dbg !16
  %1548 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local918, i32 0) #0, !dbg !16
  %1549 = bitcast <4 x i32> %1548 to <8 x half>, !dbg !16
  %1550 = shufflevector <8 x half> %1535, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1550, <4 x float> zeroinitializer) #9, !dbg !16
  %1552 = shufflevector <8 x half> %1537, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1552, <4 x float> %1551) #9, !dbg !16
  %1554 = shufflevector <8 x half> %1539, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1554, <4 x float> %1553) #9, !dbg !16
  %1556 = shufflevector <8 x half> %1541, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1556, <4 x float> %1555) #9, !dbg !16
  %1558 = shufflevector <8 x half> %1543, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1558, <4 x float> %1557) #9, !dbg !16
  %1560 = shufflevector <8 x half> %1545, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1560, <4 x float> %1559) #9, !dbg !16
  %1562 = shufflevector <8 x half> %1547, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1562, <4 x float> %1561) #9, !dbg !16
  %1564 = shufflevector <8 x half> %1549, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1564, <4 x float> %1563) #9, !dbg !16
  %1566 = extractelement <4 x float> %1565, i64 0, !dbg !16
  %1567 = extractelement <4 x float> %1565, i64 1, !dbg !16
  %1568 = extractelement <4 x float> %1565, i64 2, !dbg !16
  %1569 = extractelement <4 x float> %1565, i64 3, !dbg !16
  %1570 = select i1 %.not, float %1567, float %1566, !dbg !16
  %1571 = bitcast float %1570 to i32, !dbg !16
  %1572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1571), !dbg !16
  %1573 = bitcast i32 %1572 to float, !dbg !16
  %1574 = select i1 %.not, float %1566, float %1573, !dbg !16
  %1575 = select i1 %.not, float %1573, float %1567, !dbg !16
  %1576 = select i1 %.not, float %1569, float %1568, !dbg !16
  %1577 = bitcast float %1576 to i32, !dbg !16
  %1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1577), !dbg !16
  %1579 = bitcast i32 %1578 to float, !dbg !16
  %1580 = select i1 %.not, float %1568, float %1579, !dbg !16
  %1581 = select i1 %.not, float %1579, float %1569, !dbg !16
  %1582 = select i1 %.not1, float %1580, float %1574, !dbg !16
  %1583 = bitcast float %1582 to i32, !dbg !16
  %1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1583), !dbg !16
  %1585 = bitcast i32 %1584 to float, !dbg !16
  %1586 = select i1 %.not1, float %1574, float %1585, !dbg !16
  %1587 = select i1 %.not1, float %1585, float %1580, !dbg !16
  %1588 = select i1 %.not1, float %1581, float %1575, !dbg !16
  %1589 = bitcast float %1588 to i32, !dbg !16
  %1590 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1589), !dbg !16
  %1591 = bitcast i32 %1590 to float, !dbg !16
  %1592 = select i1 %.not1, float %1575, float %1591, !dbg !16
  %1593 = select i1 %.not1, float %1591, float %1581, !dbg !16
  %1594 = shufflevector <8 x half> %1535, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1595 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1594, <4 x float> zeroinitializer) #9, !dbg !16
  %1596 = shufflevector <8 x half> %1537, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1596, <4 x float> %1595) #9, !dbg !16
  %1598 = shufflevector <8 x half> %1539, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1598, <4 x float> %1597) #9, !dbg !16
  %1600 = shufflevector <8 x half> %1541, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1600, <4 x float> %1599) #9, !dbg !16
  %1602 = shufflevector <8 x half> %1543, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1603 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1602, <4 x float> %1601) #9, !dbg !16
  %1604 = shufflevector <8 x half> %1545, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1605 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1604, <4 x float> %1603) #9, !dbg !16
  %1606 = shufflevector <8 x half> %1547, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1606, <4 x float> %1605) #9, !dbg !16
  %1608 = shufflevector <8 x half> %1549, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1609 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1608, <4 x float> %1607) #9, !dbg !16
  %1610 = extractelement <4 x float> %1609, i64 0, !dbg !16
  %1611 = extractelement <4 x float> %1609, i64 1, !dbg !16
  %1612 = extractelement <4 x float> %1609, i64 2, !dbg !16
  %1613 = extractelement <4 x float> %1609, i64 3, !dbg !16
  %1614 = select i1 %.not, float %1611, float %1610, !dbg !16
  %1615 = bitcast float %1614 to i32, !dbg !16
  %1616 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1615), !dbg !16
  %1617 = bitcast i32 %1616 to float, !dbg !16
  %1618 = select i1 %.not, float %1610, float %1617, !dbg !16
  %1619 = select i1 %.not, float %1617, float %1611, !dbg !16
  %1620 = select i1 %.not, float %1613, float %1612, !dbg !16
  %1621 = bitcast float %1620 to i32, !dbg !16
  %1622 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1621), !dbg !16
  %1623 = bitcast i32 %1622 to float, !dbg !16
  %1624 = select i1 %.not, float %1612, float %1623, !dbg !16
  %1625 = select i1 %.not, float %1623, float %1613, !dbg !16
  %1626 = select i1 %.not1, float %1624, float %1618, !dbg !16
  %1627 = bitcast float %1626 to i32, !dbg !16
  %1628 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1627), !dbg !16
  %1629 = bitcast i32 %1628 to float, !dbg !16
  %1630 = select i1 %.not1, float %1618, float %1629, !dbg !16
  %1631 = select i1 %.not1, float %1629, float %1624, !dbg !16
  %1632 = select i1 %.not1, float %1625, float %1619, !dbg !16
  %1633 = bitcast float %1632 to i32, !dbg !16
  %1634 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1633), !dbg !16
  %1635 = bitcast i32 %1634 to float, !dbg !16
  %1636 = select i1 %.not1, float %1619, float %1635, !dbg !16
  %1637 = select i1 %.not1, float %1635, float %1625, !dbg !16
  %1638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1550, <4 x float> zeroinitializer) #9, !dbg !16
  %1639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1552, <4 x float> %1638) #9, !dbg !16
  %1640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1554, <4 x float> %1639) #9, !dbg !16
  %1641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1556, <4 x float> %1640) #9, !dbg !16
  %1642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1558, <4 x float> %1641) #9, !dbg !16
  %1643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1560, <4 x float> %1642) #9, !dbg !16
  %1644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1562, <4 x float> %1643) #9, !dbg !16
  %1645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1564, <4 x float> %1644) #9, !dbg !16
  %1646 = extractelement <4 x float> %1645, i64 0, !dbg !16
  %1647 = extractelement <4 x float> %1645, i64 1, !dbg !16
  %1648 = extractelement <4 x float> %1645, i64 2, !dbg !16
  %1649 = extractelement <4 x float> %1645, i64 3, !dbg !16
  %1650 = select i1 %.not, float %1647, float %1646, !dbg !16
  %1651 = bitcast float %1650 to i32, !dbg !16
  %1652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1651), !dbg !16
  %1653 = bitcast i32 %1652 to float, !dbg !16
  %1654 = select i1 %.not, float %1646, float %1653, !dbg !16
  %1655 = select i1 %.not, float %1653, float %1647, !dbg !16
  %1656 = select i1 %.not, float %1649, float %1648, !dbg !16
  %1657 = bitcast float %1656 to i32, !dbg !16
  %1658 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1657), !dbg !16
  %1659 = bitcast i32 %1658 to float, !dbg !16
  %1660 = select i1 %.not, float %1648, float %1659, !dbg !16
  %1661 = select i1 %.not, float %1659, float %1649, !dbg !16
  %1662 = select i1 %.not1, float %1660, float %1654, !dbg !16
  %1663 = bitcast float %1662 to i32, !dbg !16
  %1664 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1663), !dbg !16
  %1665 = bitcast i32 %1664 to float, !dbg !16
  %1666 = select i1 %.not1, float %1654, float %1665, !dbg !16
  %1667 = select i1 %.not1, float %1665, float %1660, !dbg !16
  %1668 = select i1 %.not1, float %1661, float %1655, !dbg !16
  %1669 = bitcast float %1668 to i32, !dbg !16
  %1670 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1669), !dbg !16
  %1671 = bitcast i32 %1670 to float, !dbg !16
  %1672 = select i1 %.not1, float %1655, float %1671, !dbg !16
  %1673 = select i1 %.not1, float %1671, float %1661, !dbg !16
  %1674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1594, <4 x float> zeroinitializer) #9, !dbg !16
  %1675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1596, <4 x float> %1674) #9, !dbg !16
  %1676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1598, <4 x float> %1675) #9, !dbg !16
  %1677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1600, <4 x float> %1676) #9, !dbg !16
  %1678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1602, <4 x float> %1677) #9, !dbg !16
  %1679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1604, <4 x float> %1678) #9, !dbg !16
  %1680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1606, <4 x float> %1679) #9, !dbg !16
  %1681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1608, <4 x float> %1680) #9, !dbg !16
  %1682 = extractelement <4 x float> %1681, i64 0, !dbg !16
  %1683 = extractelement <4 x float> %1681, i64 1, !dbg !16
  %1684 = extractelement <4 x float> %1681, i64 2, !dbg !16
  %1685 = extractelement <4 x float> %1681, i64 3, !dbg !16
  %1686 = select i1 %.not, float %1683, float %1682, !dbg !16
  %1687 = bitcast float %1686 to i32, !dbg !16
  %1688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1687), !dbg !16
  %1689 = bitcast i32 %1688 to float, !dbg !16
  %1690 = select i1 %.not, float %1682, float %1689, !dbg !16
  %1691 = select i1 %.not, float %1689, float %1683, !dbg !16
  %1692 = select i1 %.not, float %1685, float %1684, !dbg !16
  %1693 = bitcast float %1692 to i32, !dbg !16
  %1694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1693), !dbg !16
  %1695 = bitcast i32 %1694 to float, !dbg !16
  %1696 = select i1 %.not, float %1684, float %1695, !dbg !16
  %1697 = select i1 %.not, float %1695, float %1685, !dbg !16
  %1698 = select i1 %.not1, float %1696, float %1690, !dbg !16
  %1699 = bitcast float %1698 to i32, !dbg !16
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1699), !dbg !16
  %1701 = bitcast i32 %1700 to float, !dbg !16
  %1702 = select i1 %.not1, float %1690, float %1701, !dbg !16
  %1703 = select i1 %.not1, float %1701, float %1696, !dbg !16
  %1704 = select i1 %.not1, float %1697, float %1691, !dbg !16
  %1705 = bitcast float %1704 to i32, !dbg !16
  %1706 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1705), !dbg !16
  %1707 = bitcast i32 %1706 to float, !dbg !16
  %1708 = select i1 %.not1, float %1691, float %1707, !dbg !16
  %1709 = select i1 %.not1, float %1707, float %1697, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %1710, label %1711, !dbg !36

1710:                                             ; preds = %1477
  %address.thread919 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local920 = lshr i32 %address.thread919, 1, !dbg !15
  %address.local921 = and i32 %address.local920, 96, !dbg !15
  %address.local922 = and i32 %address.thread919, 15, !dbg !15
  %address.local923 = or disjoint i32 %address.local921, %address.local922, !dbg !15
  %address.local924 = zext nneg i32 %address.local923 to i64, !dbg !15
  %address.local925 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !15
  store float %1041, ptr addrspace(3) %address.local925, align 4, !dbg !36
  %address.local926 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !15
  %address.local927 = getelementptr i8, ptr addrspace(3) %address.local926, i64 64, !dbg !15
  store float %1042, ptr addrspace(3) %address.local927, align 4, !dbg !36
  br label %1711, !dbg !36

1711:                                             ; preds = %1710, %1477
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread928 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %1712 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !36
  %lds.const.gep1750 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %1713 = load float, ptr addrspace(3) %lds.const.gep1750, align 4, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.local942 = zext nneg i32 %address.thread928 to i64, !dbg !12
  %address.local943 = shl nuw nsw i64 %address.local942, 3, !dbg !11
  %address.local944 = and i64 %address.local943, 1536, !dbg !11
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local944, !dbg !11
  %address.local946 = ptrtoint ptr addrspace(3) %address.local945 to i64, !dbg !11
  %address.local947 = trunc i64 %address.local946 to i32, !dbg !11
  %1714 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local947), !dbg !20
  %address.local948 = shl nuw nsw i64 %address.local942, 6, !dbg !20
  %address.local949 = and i64 %address.local948, 16128, !dbg !20
  %address.local950 = shl nuw nsw i64 %address.local942, 4, !dbg !20
  %address.local951 = and i64 %address.local950, 48, !dbg !20
  %1715 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %.idx1878 = shl i64 %1483, 1, !dbg !20
  %1716 = or disjoint i64 %address.local949, %address.local951, !dbg !20
  %1717 = or disjoint i64 %.idx1878, %1716, !dbg !20
  %1718 = trunc i64 %1717 to i32, !dbg !20
  %dma.buffer.offset1761 = or i32 %1718, 128, !dbg !20
  %1719 = insertelement <2 x i64> poison, i64 %1715, i64 0, !dbg !20
  %1720 = bitcast <2 x i64> %1719 to <4 x i32>, !dbg !20
  %dma.buffer.rsrc1762 = shufflevector <4 x i32> %1720, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !20
  %dma.buffer.lds1763 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1714), !dbg !20
  %1721 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1763, i32 %dma.buffer.offset1761, <4 x i32> %dma.buffer.rsrc1762) #10, !dbg !20
  %address.local954 = or disjoint i64 %address.local942, 256, !dbg !11
  %address.local955 = shl nuw nsw i64 %address.local954, 3, !dbg !11
  %address.local956 = and i64 %address.local955, 3584, !dbg !11
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local956, !dbg !11
  %address.local958 = ptrtoint ptr addrspace(3) %address.local957 to i64, !dbg !11
  %address.local959 = trunc i64 %address.local958 to i32, !dbg !11
  %1722 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local959), !dbg !20
  %address.local960 = shl nuw nsw i64 %address.local954, 6, !dbg !20
  %address.local961 = and i64 %address.local960, 32512, !dbg !20
  %address.local962 = shl nuw nsw i64 %address.local942, 4, !dbg !20
  %address.local963 = and i64 %address.local962, 48, !dbg !20
  %1723 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %.idx1880 = shl i64 %1483, 1, !dbg !20
  %1724 = or disjoint i64 %.idx1880, %address.local961, !dbg !20
  %1725 = or disjoint i64 %1724, %address.local963, !dbg !20
  %1726 = trunc i64 %1725 to i32, !dbg !20
  %dma.buffer.offset1758 = or i32 %1726, 128, !dbg !20
  %1727 = insertelement <2 x i64> poison, i64 %1723, i64 0, !dbg !20
  %1728 = bitcast <2 x i64> %1727 to <4 x i32>, !dbg !20
  %dma.buffer.rsrc1759 = shufflevector <4 x i32> %1728, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !20
  %dma.buffer.lds1760 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1722), !dbg !20
  %1729 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1760, i32 %dma.buffer.offset1758, <4 x i32> %dma.buffer.rsrc1759) #10, !dbg !20
  %1730 = extractvalue [2 x <8 x float>] %58, 0, !dbg !36
  %1731 = shufflevector <8 x float> %1730, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1732 = insertelement <4 x float> poison, float %1712, i64 0, !dbg !36
  %1733 = shufflevector <4 x float> %1732, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1734 = fmul <4 x float> %1731, %1733, !dbg !36
  %1735 = insertelement <4 x float> poison, float %1586, i64 0, !dbg !37
  %1736 = insertelement <4 x float> %1735, float %1592, i64 1, !dbg !37
  %1737 = insertelement <4 x float> %1736, float %1587, i64 2, !dbg !37
  %1738 = insertelement <4 x float> %1737, float %1593, i64 3, !dbg !37
  %1739 = fadd <4 x float> %1734, %1738, !dbg !37
  %1740 = shufflevector <8 x float> %1730, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1741 = fmul <4 x float> %1740, %1733, !dbg !36
  %1742 = insertelement <4 x float> poison, float %1630, i64 0, !dbg !37
  %1743 = insertelement <4 x float> %1742, float %1636, i64 1, !dbg !37
  %1744 = insertelement <4 x float> %1743, float %1631, i64 2, !dbg !37
  %1745 = insertelement <4 x float> %1744, float %1637, i64 3, !dbg !37
  %1746 = fadd <4 x float> %1741, %1745, !dbg !37
  %1747 = extractvalue [2 x <8 x float>] %58, 1, !dbg !36
  %1748 = shufflevector <8 x float> %1747, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1749 = insertelement <4 x float> poison, float %1713, i64 0, !dbg !36
  %1750 = shufflevector <4 x float> %1749, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1751 = fmul <4 x float> %1748, %1750, !dbg !36
  %1752 = insertelement <4 x float> poison, float %1666, i64 0, !dbg !37
  %1753 = insertelement <4 x float> %1752, float %1672, i64 1, !dbg !37
  %1754 = insertelement <4 x float> %1753, float %1667, i64 2, !dbg !37
  %1755 = insertelement <4 x float> %1754, float %1673, i64 3, !dbg !37
  %1756 = fadd <4 x float> %1751, %1755, !dbg !37
  %1757 = shufflevector <8 x float> %1747, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1758 = fmul <4 x float> %1757, %1750, !dbg !36
  %1759 = insertelement <4 x float> poison, float %1702, i64 0, !dbg !37
  %1760 = insertelement <4 x float> %1759, float %1708, i64 1, !dbg !37
  %1761 = insertelement <4 x float> %1760, float %1703, i64 2, !dbg !37
  %1762 = insertelement <4 x float> %1761, float %1709, i64 3, !dbg !37
  %1763 = fadd <4 x float> %1758, %1762, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local966 = shl nuw nsw i32 %address.thread928, 3, !dbg !16
  %address.local967 = and i32 %address.local966, 504, !dbg !16
  %address.local968 = zext nneg i32 %address.local967 to i64, !dbg !16
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %1764 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #0, !dbg !16
  %1765 = bitcast <4 x i32> %1764 to <8 x half>, !dbg !16
  %address.local970 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local970, i64 1024, !dbg !16
  %1766 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #0, !dbg !16
  %1767 = bitcast <4 x i32> %1766 to <8 x half>, !dbg !16
  %address.local972 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local972, i64 2048, !dbg !16
  %1768 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !16
  %1769 = bitcast <4 x i32> %1768 to <8 x half>, !dbg !16
  %address.local974 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local975 = getelementptr i8, ptr addrspace(3) %address.local974, i64 3072, !dbg !16
  %1770 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #0, !dbg !16
  %1771 = bitcast <4 x i32> %1770 to <8 x half>, !dbg !16
  %address.local976 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local976, i64 4096, !dbg !16
  %1772 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !16
  %1773 = bitcast <4 x i32> %1772 to <8 x half>, !dbg !16
  %address.local978 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local979 = getelementptr i8, ptr addrspace(3) %address.local978, i64 5120, !dbg !16
  %1774 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #0, !dbg !16
  %1775 = bitcast <4 x i32> %1774 to <8 x half>, !dbg !16
  %address.local980 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local981 = getelementptr i8, ptr addrspace(3) %address.local980, i64 6144, !dbg !16
  %1776 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #0, !dbg !16
  %1777 = bitcast <4 x i32> %1776 to <8 x half>, !dbg !16
  %address.local982 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local983 = getelementptr i8, ptr addrspace(3) %address.local982, i64 7168, !dbg !16
  %1778 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #0, !dbg !16
  %1779 = bitcast <4 x i32> %1778 to <8 x half>, !dbg !16
  %1780 = shufflevector <8 x half> %1765, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1781 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1780, <4 x float> zeroinitializer) #9, !dbg !16
  %1782 = shufflevector <8 x half> %1767, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1783 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1782, <4 x float> %1781) #9, !dbg !16
  %1784 = shufflevector <8 x half> %1769, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1784, <4 x float> %1783) #9, !dbg !16
  %1786 = shufflevector <8 x half> %1771, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1787 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1786, <4 x float> %1785) #9, !dbg !16
  %1788 = shufflevector <8 x half> %1773, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1789 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1788, <4 x float> %1787) #9, !dbg !16
  %1790 = shufflevector <8 x half> %1775, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1791 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1790, <4 x float> %1789) #9, !dbg !16
  %1792 = shufflevector <8 x half> %1777, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1793 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1792, <4 x float> %1791) #9, !dbg !16
  %1794 = shufflevector <8 x half> %1779, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1795 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1794, <4 x float> %1793) #9, !dbg !16
  %1796 = extractelement <4 x float> %1795, i64 0, !dbg !16
  %1797 = extractelement <4 x float> %1795, i64 1, !dbg !16
  %1798 = extractelement <4 x float> %1795, i64 2, !dbg !16
  %1799 = extractelement <4 x float> %1795, i64 3, !dbg !16
  %1800 = select i1 %.not, float %1797, float %1796, !dbg !16
  %1801 = bitcast float %1800 to i32, !dbg !16
  %1802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1801), !dbg !16
  %1803 = bitcast i32 %1802 to float, !dbg !16
  %1804 = select i1 %.not, float %1796, float %1803, !dbg !16
  %1805 = select i1 %.not, float %1803, float %1797, !dbg !16
  %1806 = select i1 %.not, float %1799, float %1798, !dbg !16
  %1807 = bitcast float %1806 to i32, !dbg !16
  %1808 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1807), !dbg !16
  %1809 = bitcast i32 %1808 to float, !dbg !16
  %1810 = select i1 %.not, float %1798, float %1809, !dbg !16
  %1811 = select i1 %.not, float %1809, float %1799, !dbg !16
  %1812 = select i1 %.not1, float %1810, float %1804, !dbg !16
  %1813 = bitcast float %1812 to i32, !dbg !16
  %1814 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1813), !dbg !16
  %1815 = bitcast i32 %1814 to float, !dbg !16
  %1816 = select i1 %.not1, float %1804, float %1815, !dbg !16
  %1817 = select i1 %.not1, float %1815, float %1810, !dbg !16
  %1818 = select i1 %.not1, float %1811, float %1805, !dbg !16
  %1819 = bitcast float %1818 to i32, !dbg !16
  %1820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1819), !dbg !16
  %1821 = bitcast i32 %1820 to float, !dbg !16
  %1822 = select i1 %.not1, float %1805, float %1821, !dbg !16
  %1823 = select i1 %.not1, float %1821, float %1811, !dbg !16
  %1824 = shufflevector <8 x half> %1765, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1824, <4 x float> zeroinitializer) #9, !dbg !16
  %1826 = shufflevector <8 x half> %1767, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1826, <4 x float> %1825) #9, !dbg !16
  %1828 = shufflevector <8 x half> %1769, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1828, <4 x float> %1827) #9, !dbg !16
  %1830 = shufflevector <8 x half> %1771, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1830, <4 x float> %1829) #9, !dbg !16
  %1832 = shufflevector <8 x half> %1773, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1832, <4 x float> %1831) #9, !dbg !16
  %1834 = shufflevector <8 x half> %1775, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1834, <4 x float> %1833) #9, !dbg !16
  %1836 = shufflevector <8 x half> %1777, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1836, <4 x float> %1835) #9, !dbg !16
  %1838 = shufflevector <8 x half> %1779, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1838, <4 x float> %1837) #9, !dbg !16
  %1840 = extractelement <4 x float> %1839, i64 0, !dbg !16
  %1841 = extractelement <4 x float> %1839, i64 1, !dbg !16
  %1842 = extractelement <4 x float> %1839, i64 2, !dbg !16
  %1843 = extractelement <4 x float> %1839, i64 3, !dbg !16
  %1844 = select i1 %.not, float %1841, float %1840, !dbg !16
  %1845 = bitcast float %1844 to i32, !dbg !16
  %1846 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1845), !dbg !16
  %1847 = bitcast i32 %1846 to float, !dbg !16
  %1848 = select i1 %.not, float %1840, float %1847, !dbg !16
  %1849 = select i1 %.not, float %1847, float %1841, !dbg !16
  %1850 = select i1 %.not, float %1843, float %1842, !dbg !16
  %1851 = bitcast float %1850 to i32, !dbg !16
  %1852 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1851), !dbg !16
  %1853 = bitcast i32 %1852 to float, !dbg !16
  %1854 = select i1 %.not, float %1842, float %1853, !dbg !16
  %1855 = select i1 %.not, float %1853, float %1843, !dbg !16
  %1856 = select i1 %.not1, float %1854, float %1848, !dbg !16
  %1857 = bitcast float %1856 to i32, !dbg !16
  %1858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1857), !dbg !16
  %1859 = bitcast i32 %1858 to float, !dbg !16
  %1860 = select i1 %.not1, float %1848, float %1859, !dbg !16
  %1861 = select i1 %.not1, float %1859, float %1854, !dbg !16
  %1862 = select i1 %.not1, float %1855, float %1849, !dbg !16
  %1863 = bitcast float %1862 to i32, !dbg !16
  %1864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1863), !dbg !16
  %1865 = bitcast i32 %1864 to float, !dbg !16
  %1866 = select i1 %.not1, float %1849, float %1865, !dbg !16
  %1867 = select i1 %.not1, float %1865, float %1855, !dbg !16
  %1868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1780, <4 x float> zeroinitializer) #9, !dbg !16
  %1869 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1782, <4 x float> %1868) #9, !dbg !16
  %1870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1784, <4 x float> %1869) #9, !dbg !16
  %1871 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1786, <4 x float> %1870) #9, !dbg !16
  %1872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1788, <4 x float> %1871) #9, !dbg !16
  %1873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1790, <4 x float> %1872) #9, !dbg !16
  %1874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1792, <4 x float> %1873) #9, !dbg !16
  %1875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1794, <4 x float> %1874) #9, !dbg !16
  %1876 = extractelement <4 x float> %1875, i64 0, !dbg !16
  %1877 = extractelement <4 x float> %1875, i64 1, !dbg !16
  %1878 = extractelement <4 x float> %1875, i64 2, !dbg !16
  %1879 = extractelement <4 x float> %1875, i64 3, !dbg !16
  %1880 = select i1 %.not, float %1877, float %1876, !dbg !16
  %1881 = bitcast float %1880 to i32, !dbg !16
  %1882 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1881), !dbg !16
  %1883 = bitcast i32 %1882 to float, !dbg !16
  %1884 = select i1 %.not, float %1876, float %1883, !dbg !16
  %1885 = select i1 %.not, float %1883, float %1877, !dbg !16
  %1886 = select i1 %.not, float %1879, float %1878, !dbg !16
  %1887 = bitcast float %1886 to i32, !dbg !16
  %1888 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1887), !dbg !16
  %1889 = bitcast i32 %1888 to float, !dbg !16
  %1890 = select i1 %.not, float %1878, float %1889, !dbg !16
  %1891 = select i1 %.not, float %1889, float %1879, !dbg !16
  %1892 = select i1 %.not1, float %1890, float %1884, !dbg !16
  %1893 = bitcast float %1892 to i32, !dbg !16
  %1894 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1893), !dbg !16
  %1895 = bitcast i32 %1894 to float, !dbg !16
  %1896 = select i1 %.not1, float %1884, float %1895, !dbg !16
  %1897 = select i1 %.not1, float %1895, float %1890, !dbg !16
  %1898 = select i1 %.not1, float %1891, float %1885, !dbg !16
  %1899 = bitcast float %1898 to i32, !dbg !16
  %1900 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1899), !dbg !16
  %1901 = bitcast i32 %1900 to float, !dbg !16
  %1902 = select i1 %.not1, float %1885, float %1901, !dbg !16
  %1903 = select i1 %.not1, float %1901, float %1891, !dbg !16
  %1904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1824, <4 x float> zeroinitializer) #9, !dbg !16
  %1905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1826, <4 x float> %1904) #9, !dbg !16
  %1906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1828, <4 x float> %1905) #9, !dbg !16
  %1907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1830, <4 x float> %1906) #9, !dbg !16
  %1908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1832, <4 x float> %1907) #9, !dbg !16
  %1909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1834, <4 x float> %1908) #9, !dbg !16
  %1910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1836, <4 x float> %1909) #9, !dbg !16
  %1911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1838, <4 x float> %1910) #9, !dbg !16
  %1912 = extractelement <4 x float> %1911, i64 0, !dbg !16
  %1913 = extractelement <4 x float> %1911, i64 1, !dbg !16
  %1914 = extractelement <4 x float> %1911, i64 2, !dbg !16
  %1915 = extractelement <4 x float> %1911, i64 3, !dbg !16
  %1916 = select i1 %.not, float %1913, float %1912, !dbg !16
  %1917 = bitcast float %1916 to i32, !dbg !16
  %1918 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1917), !dbg !16
  %1919 = bitcast i32 %1918 to float, !dbg !16
  %1920 = select i1 %.not, float %1912, float %1919, !dbg !16
  %1921 = select i1 %.not, float %1919, float %1913, !dbg !16
  %1922 = select i1 %.not, float %1915, float %1914, !dbg !16
  %1923 = bitcast float %1922 to i32, !dbg !16
  %1924 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %1923), !dbg !16
  %1925 = bitcast i32 %1924 to float, !dbg !16
  %1926 = select i1 %.not, float %1914, float %1925, !dbg !16
  %1927 = select i1 %.not, float %1925, float %1915, !dbg !16
  %1928 = select i1 %.not1, float %1926, float %1920, !dbg !16
  %1929 = bitcast float %1928 to i32, !dbg !16
  %1930 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1929), !dbg !16
  %1931 = bitcast i32 %1930 to float, !dbg !16
  %1932 = select i1 %.not1, float %1920, float %1931, !dbg !16
  %1933 = select i1 %.not1, float %1931, float %1926, !dbg !16
  %1934 = select i1 %.not1, float %1927, float %1921, !dbg !16
  %1935 = bitcast float %1934 to i32, !dbg !16
  %1936 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %1935), !dbg !16
  %1937 = bitcast i32 %1936 to float, !dbg !16
  %1938 = select i1 %.not1, float %1921, float %1937, !dbg !16
  %1939 = select i1 %.not1, float %1937, float %1927, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %1940, label %1941, !dbg !36

1940:                                             ; preds = %1711
  %address.thread984 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local985 = lshr i32 %address.thread984, 1, !dbg !15
  %address.local986 = and i32 %address.local985, 96, !dbg !15
  %address.local987 = and i32 %address.thread984, 15, !dbg !15
  %address.local988 = or disjoint i32 %address.local986, %address.local987, !dbg !15
  %address.local989 = zext nneg i32 %address.local988 to i64, !dbg !15
  %address.local990 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !15
  store float %1041, ptr addrspace(3) %address.local990, align 4, !dbg !36
  %address.local991 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !15
  %address.local992 = getelementptr i8, ptr addrspace(3) %address.local991, i64 64, !dbg !15
  store float %1042, ptr addrspace(3) %address.local992, align 4, !dbg !36
  br label %1941, !dbg !36

1941:                                             ; preds = %1940, %1711
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread993 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %1942 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !36
  %lds.const.gep1752 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %1943 = load float, ptr addrspace(3) %lds.const.gep1752, align 4, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.local1007 = zext nneg i32 %address.thread993 to i64, !dbg !12
  %address.local1008 = shl nuw nsw i64 %address.local1007, 3, !dbg !11
  %address.local1009 = and i64 %address.local1008, 1536, !dbg !11
  %address.local1010 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1009, !dbg !11
  %address.local1011 = ptrtoint ptr addrspace(3) %address.local1010 to i64, !dbg !11
  %address.local1012 = trunc i64 %address.local1011 to i32, !dbg !11
  %1944 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1012), !dbg !20
  %address.local1013 = shl nuw nsw i64 %address.local1007, 6, !dbg !20
  %address.local1014 = and i64 %address.local1013, 16128, !dbg !20
  %address.local1015 = shl nuw nsw i64 %address.local1007, 4, !dbg !20
  %address.local1016 = and i64 %address.local1015, 48, !dbg !20
  %1945 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %.idx1882 = shl i64 %1483, 1, !dbg !20
  %1946 = or disjoint i64 %address.local1014, %address.local1016, !dbg !20
  %1947 = or disjoint i64 %.idx1882, %1946, !dbg !20
  %1948 = trunc i64 %1947 to i32, !dbg !20
  %dma.buffer.offset1755 = or i32 %1948, 192, !dbg !20
  %1949 = insertelement <2 x i64> poison, i64 %1945, i64 0, !dbg !20
  %1950 = bitcast <2 x i64> %1949 to <4 x i32>, !dbg !20
  %dma.buffer.rsrc1756 = shufflevector <4 x i32> %1950, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !20
  %dma.buffer.lds1757 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1944), !dbg !20
  %1951 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1757, i32 %dma.buffer.offset1755, <4 x i32> %dma.buffer.rsrc1756) #10, !dbg !20
  %address.local1019 = or disjoint i64 %address.local1007, 256, !dbg !11
  %address.local1020 = shl nuw nsw i64 %address.local1019, 3, !dbg !11
  %address.local1021 = and i64 %address.local1020, 3584, !dbg !11
  %address.local1022 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1021, !dbg !11
  %address.local1023 = ptrtoint ptr addrspace(3) %address.local1022 to i64, !dbg !11
  %address.local1024 = trunc i64 %address.local1023 to i32, !dbg !11
  %1952 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1024), !dbg !20
  %address.local1025 = shl nuw nsw i64 %address.local1019, 6, !dbg !20
  %address.local1026 = and i64 %address.local1025, 32512, !dbg !20
  %address.local1027 = shl nuw nsw i64 %address.local1007, 4, !dbg !20
  %address.local1028 = and i64 %address.local1027, 48, !dbg !20
  %1953 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %.idx1884 = shl i64 %1483, 1, !dbg !20
  %1954 = or disjoint i64 %.idx1884, %address.local1026, !dbg !20
  %1955 = or disjoint i64 %1954, %address.local1028, !dbg !20
  %1956 = trunc i64 %1955 to i32, !dbg !20
  %dma.buffer.offset = or i32 %1956, 192, !dbg !20
  %1957 = insertelement <2 x i64> poison, i64 %1953, i64 0, !dbg !20
  %1958 = bitcast <2 x i64> %1957 to <4 x i32>, !dbg !20
  %dma.buffer.rsrc = shufflevector <4 x i32> %1958, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !20
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1952), !dbg !20
  %1959 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !20
  %1960 = extractvalue [2 x <8 x float>] %59, 0, !dbg !36
  %1961 = shufflevector <8 x float> %1960, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1962 = insertelement <4 x float> poison, float %1942, i64 0, !dbg !36
  %1963 = shufflevector <4 x float> %1962, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1964 = fmul <4 x float> %1961, %1963, !dbg !36
  %1965 = insertelement <4 x float> poison, float %1816, i64 0, !dbg !37
  %1966 = insertelement <4 x float> %1965, float %1822, i64 1, !dbg !37
  %1967 = insertelement <4 x float> %1966, float %1817, i64 2, !dbg !37
  %1968 = insertelement <4 x float> %1967, float %1823, i64 3, !dbg !37
  %1969 = fadd <4 x float> %1964, %1968, !dbg !37
  %1970 = shufflevector <8 x float> %1960, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1971 = fmul <4 x float> %1970, %1963, !dbg !36
  %1972 = insertelement <4 x float> poison, float %1860, i64 0, !dbg !37
  %1973 = insertelement <4 x float> %1972, float %1866, i64 1, !dbg !37
  %1974 = insertelement <4 x float> %1973, float %1861, i64 2, !dbg !37
  %1975 = insertelement <4 x float> %1974, float %1867, i64 3, !dbg !37
  %1976 = fadd <4 x float> %1971, %1975, !dbg !37
  %1977 = extractvalue [2 x <8 x float>] %59, 1, !dbg !36
  %1978 = shufflevector <8 x float> %1977, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1979 = insertelement <4 x float> poison, float %1943, i64 0, !dbg !36
  %1980 = shufflevector <4 x float> %1979, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1981 = fmul <4 x float> %1978, %1980, !dbg !36
  %1982 = insertelement <4 x float> poison, float %1896, i64 0, !dbg !37
  %1983 = insertelement <4 x float> %1982, float %1902, i64 1, !dbg !37
  %1984 = insertelement <4 x float> %1983, float %1897, i64 2, !dbg !37
  %1985 = insertelement <4 x float> %1984, float %1903, i64 3, !dbg !37
  %1986 = fadd <4 x float> %1981, %1985, !dbg !37
  %1987 = shufflevector <8 x float> %1977, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1988 = fmul <4 x float> %1987, %1980, !dbg !36
  %1989 = insertelement <4 x float> poison, float %1932, i64 0, !dbg !37
  %1990 = insertelement <4 x float> %1989, float %1938, i64 1, !dbg !37
  %1991 = insertelement <4 x float> %1990, float %1933, i64 2, !dbg !37
  %1992 = insertelement <4 x float> %1991, float %1939, i64 3, !dbg !37
  %1993 = fadd <4 x float> %1988, %1992, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local1031 = shl nuw nsw i32 %address.thread993, 3, !dbg !16
  %address.local1032 = and i32 %address.local1031, 504, !dbg !16
  %address.local1033 = zext nneg i32 %address.local1032 to i64, !dbg !16
  %address.local1034 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %1994 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !16
  %1995 = bitcast <4 x i32> %1994 to <8 x half>, !dbg !16
  %address.local1035 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1036 = getelementptr i8, ptr addrspace(3) %address.local1035, i64 1024, !dbg !16
  %1996 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1036, i32 0) #0, !dbg !16
  %1997 = bitcast <4 x i32> %1996 to <8 x half>, !dbg !16
  %address.local1037 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1038 = getelementptr i8, ptr addrspace(3) %address.local1037, i64 2048, !dbg !16
  %1998 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1038, i32 0) #0, !dbg !16
  %1999 = bitcast <4 x i32> %1998 to <8 x half>, !dbg !16
  %address.local1039 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 3072, !dbg !16
  %2000 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !16
  %2001 = bitcast <4 x i32> %2000 to <8 x half>, !dbg !16
  %address.local1041 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1042 = getelementptr i8, ptr addrspace(3) %address.local1041, i64 4096, !dbg !16
  %2002 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1042, i32 0) #0, !dbg !16
  %2003 = bitcast <4 x i32> %2002 to <8 x half>, !dbg !16
  %address.local1043 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1043, i64 5120, !dbg !16
  %2004 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !16
  %2005 = bitcast <4 x i32> %2004 to <8 x half>, !dbg !16
  %address.local1045 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1046 = getelementptr i8, ptr addrspace(3) %address.local1045, i64 6144, !dbg !16
  %2006 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1046, i32 0) #0, !dbg !16
  %2007 = bitcast <4 x i32> %2006 to <8 x half>, !dbg !16
  %address.local1047 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1047, i64 7168, !dbg !16
  %2008 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !16
  %2009 = bitcast <4 x i32> %2008 to <8 x half>, !dbg !16
  %2010 = shufflevector <8 x half> %1995, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2010, <4 x float> zeroinitializer) #9, !dbg !16
  %2012 = shufflevector <8 x half> %1997, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2012, <4 x float> %2011) #9, !dbg !16
  %2014 = shufflevector <8 x half> %1999, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2014, <4 x float> %2013) #9, !dbg !16
  %2016 = shufflevector <8 x half> %2001, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2017 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2016, <4 x float> %2015) #9, !dbg !16
  %2018 = shufflevector <8 x half> %2003, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2018, <4 x float> %2017) #9, !dbg !16
  %2020 = shufflevector <8 x half> %2005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2020, <4 x float> %2019) #9, !dbg !16
  %2022 = shufflevector <8 x half> %2007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2022, <4 x float> %2021) #9, !dbg !16
  %2024 = shufflevector <8 x half> %2009, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2024, <4 x float> %2023) #9, !dbg !16
  %2026 = extractelement <4 x float> %2025, i64 0, !dbg !16
  %2027 = extractelement <4 x float> %2025, i64 1, !dbg !16
  %2028 = extractelement <4 x float> %2025, i64 2, !dbg !16
  %2029 = extractelement <4 x float> %2025, i64 3, !dbg !16
  %2030 = select i1 %.not, float %2027, float %2026, !dbg !16
  %2031 = bitcast float %2030 to i32, !dbg !16
  %2032 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %2031), !dbg !16
  %2033 = bitcast i32 %2032 to float, !dbg !16
  %2034 = select i1 %.not, float %2026, float %2033, !dbg !16
  %2035 = select i1 %.not, float %2033, float %2027, !dbg !16
  %2036 = select i1 %.not, float %2029, float %2028, !dbg !16
  %2037 = bitcast float %2036 to i32, !dbg !16
  %2038 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %2037), !dbg !16
  %2039 = bitcast i32 %2038 to float, !dbg !16
  %2040 = select i1 %.not, float %2028, float %2039, !dbg !16
  %2041 = select i1 %.not, float %2039, float %2029, !dbg !16
  %2042 = select i1 %.not1, float %2040, float %2034, !dbg !16
  %2043 = bitcast float %2042 to i32, !dbg !16
  %2044 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %2043), !dbg !16
  %2045 = bitcast i32 %2044 to float, !dbg !16
  %2046 = select i1 %.not1, float %2034, float %2045, !dbg !16
  %2047 = select i1 %.not1, float %2045, float %2040, !dbg !16
  %2048 = select i1 %.not1, float %2041, float %2035, !dbg !16
  %2049 = bitcast float %2048 to i32, !dbg !16
  %2050 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %2049), !dbg !16
  %2051 = bitcast i32 %2050 to float, !dbg !16
  %2052 = select i1 %.not1, float %2035, float %2051, !dbg !16
  %2053 = select i1 %.not1, float %2051, float %2041, !dbg !16
  %2054 = shufflevector <8 x half> %1995, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2054, <4 x float> zeroinitializer) #9, !dbg !16
  %2056 = shufflevector <8 x half> %1997, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2056, <4 x float> %2055) #9, !dbg !16
  %2058 = shufflevector <8 x half> %1999, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2058, <4 x float> %2057) #9, !dbg !16
  %2060 = shufflevector <8 x half> %2001, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2060, <4 x float> %2059) #9, !dbg !16
  %2062 = shufflevector <8 x half> %2003, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2062, <4 x float> %2061) #9, !dbg !16
  %2064 = shufflevector <8 x half> %2005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2064, <4 x float> %2063) #9, !dbg !16
  %2066 = shufflevector <8 x half> %2007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2066, <4 x float> %2065) #9, !dbg !16
  %2068 = shufflevector <8 x half> %2009, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2068, <4 x float> %2067) #9, !dbg !16
  %2070 = extractelement <4 x float> %2069, i64 0, !dbg !16
  %2071 = extractelement <4 x float> %2069, i64 1, !dbg !16
  %2072 = extractelement <4 x float> %2069, i64 2, !dbg !16
  %2073 = extractelement <4 x float> %2069, i64 3, !dbg !16
  %2074 = select i1 %.not, float %2071, float %2070, !dbg !16
  %2075 = bitcast float %2074 to i32, !dbg !16
  %2076 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %2075), !dbg !16
  %2077 = bitcast i32 %2076 to float, !dbg !16
  %2078 = select i1 %.not, float %2070, float %2077, !dbg !16
  %2079 = select i1 %.not, float %2077, float %2071, !dbg !16
  %2080 = select i1 %.not, float %2073, float %2072, !dbg !16
  %2081 = bitcast float %2080 to i32, !dbg !16
  %2082 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %2081), !dbg !16
  %2083 = bitcast i32 %2082 to float, !dbg !16
  %2084 = select i1 %.not, float %2072, float %2083, !dbg !16
  %2085 = select i1 %.not, float %2083, float %2073, !dbg !16
  %2086 = select i1 %.not1, float %2084, float %2078, !dbg !16
  %2087 = bitcast float %2086 to i32, !dbg !16
  %2088 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %2087), !dbg !16
  %2089 = bitcast i32 %2088 to float, !dbg !16
  %2090 = select i1 %.not1, float %2078, float %2089, !dbg !16
  %2091 = select i1 %.not1, float %2089, float %2084, !dbg !16
  %2092 = select i1 %.not1, float %2085, float %2079, !dbg !16
  %2093 = bitcast float %2092 to i32, !dbg !16
  %2094 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %2093), !dbg !16
  %2095 = bitcast i32 %2094 to float, !dbg !16
  %2096 = select i1 %.not1, float %2079, float %2095, !dbg !16
  %2097 = select i1 %.not1, float %2095, float %2085, !dbg !16
  %2098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2010, <4 x float> zeroinitializer) #9, !dbg !16
  %2099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2012, <4 x float> %2098) #9, !dbg !16
  %2100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2014, <4 x float> %2099) #9, !dbg !16
  %2101 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2016, <4 x float> %2100) #9, !dbg !16
  %2102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2018, <4 x float> %2101) #9, !dbg !16
  %2103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2020, <4 x float> %2102) #9, !dbg !16
  %2104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2022, <4 x float> %2103) #9, !dbg !16
  %2105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2024, <4 x float> %2104) #9, !dbg !16
  %2106 = extractelement <4 x float> %2105, i64 0, !dbg !16
  %2107 = extractelement <4 x float> %2105, i64 1, !dbg !16
  %2108 = extractelement <4 x float> %2105, i64 2, !dbg !16
  %2109 = extractelement <4 x float> %2105, i64 3, !dbg !16
  %2110 = select i1 %.not, float %2107, float %2106, !dbg !16
  %2111 = bitcast float %2110 to i32, !dbg !16
  %2112 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %2111), !dbg !16
  %2113 = bitcast i32 %2112 to float, !dbg !16
  %2114 = select i1 %.not, float %2106, float %2113, !dbg !16
  %2115 = select i1 %.not, float %2113, float %2107, !dbg !16
  %2116 = select i1 %.not, float %2109, float %2108, !dbg !16
  %2117 = bitcast float %2116 to i32, !dbg !16
  %2118 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %2117), !dbg !16
  %2119 = bitcast i32 %2118 to float, !dbg !16
  %2120 = select i1 %.not, float %2108, float %2119, !dbg !16
  %2121 = select i1 %.not, float %2119, float %2109, !dbg !16
  %2122 = select i1 %.not1, float %2120, float %2114, !dbg !16
  %2123 = bitcast float %2122 to i32, !dbg !16
  %2124 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %2123), !dbg !16
  %2125 = bitcast i32 %2124 to float, !dbg !16
  %2126 = select i1 %.not1, float %2114, float %2125, !dbg !16
  %2127 = select i1 %.not1, float %2125, float %2120, !dbg !16
  %2128 = select i1 %.not1, float %2121, float %2115, !dbg !16
  %2129 = bitcast float %2128 to i32, !dbg !16
  %2130 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %2129), !dbg !16
  %2131 = bitcast i32 %2130 to float, !dbg !16
  %2132 = select i1 %.not1, float %2115, float %2131, !dbg !16
  %2133 = select i1 %.not1, float %2131, float %2121, !dbg !16
  %2134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2054, <4 x float> zeroinitializer) #9, !dbg !16
  %2135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2056, <4 x float> %2134) #9, !dbg !16
  %2136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2058, <4 x float> %2135) #9, !dbg !16
  %2137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2060, <4 x float> %2136) #9, !dbg !16
  %2138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2062, <4 x float> %2137) #9, !dbg !16
  %2139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2064, <4 x float> %2138) #9, !dbg !16
  %2140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2066, <4 x float> %2139) #9, !dbg !16
  %2141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2068, <4 x float> %2140) #9, !dbg !16
  %2142 = extractelement <4 x float> %2141, i64 0, !dbg !16
  %2143 = extractelement <4 x float> %2141, i64 1, !dbg !16
  %2144 = extractelement <4 x float> %2141, i64 2, !dbg !16
  %2145 = extractelement <4 x float> %2141, i64 3, !dbg !16
  %2146 = select i1 %.not, float %2143, float %2142, !dbg !16
  %2147 = bitcast float %2146 to i32, !dbg !16
  %2148 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %2147), !dbg !16
  %2149 = bitcast i32 %2148 to float, !dbg !16
  %2150 = select i1 %.not, float %2142, float %2149, !dbg !16
  %2151 = select i1 %.not, float %2149, float %2143, !dbg !16
  %2152 = select i1 %.not, float %2145, float %2144, !dbg !16
  %2153 = bitcast float %2152 to i32, !dbg !16
  %2154 = call i32 @llvm.amdgcn.ds.bpermute(i32 %984, i32 %2153), !dbg !16
  %2155 = bitcast i32 %2154 to float, !dbg !16
  %2156 = select i1 %.not, float %2144, float %2155, !dbg !16
  %2157 = select i1 %.not, float %2155, float %2145, !dbg !16
  %2158 = select i1 %.not1, float %2156, float %2150, !dbg !16
  %2159 = bitcast float %2158 to i32, !dbg !16
  %2160 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %2159), !dbg !16
  %2161 = bitcast i32 %2160 to float, !dbg !16
  %2162 = select i1 %.not1, float %2150, float %2161, !dbg !16
  %2163 = select i1 %.not1, float %2161, float %2156, !dbg !16
  %2164 = select i1 %.not1, float %2157, float %2151, !dbg !16
  %2165 = bitcast float %2164 to i32, !dbg !16
  %2166 = call i32 @llvm.amdgcn.ds.bpermute(i32 %992, i32 %2165), !dbg !16
  %2167 = bitcast i32 %2166 to float, !dbg !16
  %2168 = select i1 %.not1, float %2151, float %2167, !dbg !16
  %2169 = select i1 %.not1, float %2167, float %2157, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %2170, label %2171, !dbg !36

2170:                                             ; preds = %1941
  %address.thread1049 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local1050 = lshr i32 %address.thread1049, 1, !dbg !15
  %address.local1051 = and i32 %address.local1050, 96, !dbg !15
  %address.local1052 = and i32 %address.thread1049, 15, !dbg !15
  %address.local1053 = or disjoint i32 %address.local1051, %address.local1052, !dbg !15
  %address.local1054 = zext nneg i32 %address.local1053 to i64, !dbg !15
  %address.local1055 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !15
  store float %1041, ptr addrspace(3) %address.local1055, align 4, !dbg !36
  %address.local1056 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !15
  %address.local1057 = getelementptr i8, ptr addrspace(3) %address.local1056, i64 64, !dbg !15
  store float %1042, ptr addrspace(3) %address.local1057, align 4, !dbg !36
  br label %2171, !dbg !36

2171:                                             ; preds = %2170, %1941
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %2172 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !36
  %lds.const.gep1754 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %2173 = load float, ptr addrspace(3) %lds.const.gep1754, align 4, !dbg !36
  %2174 = extractvalue [2 x <8 x float>] %60, 0, !dbg !36
  %2175 = shufflevector <8 x float> %2174, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2176 = insertelement <4 x float> poison, float %2172, i64 0, !dbg !36
  %2177 = shufflevector <4 x float> %2176, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2178 = fmul <4 x float> %2175, %2177, !dbg !36
  %2179 = insertelement <4 x float> poison, float %2046, i64 0, !dbg !37
  %2180 = insertelement <4 x float> %2179, float %2052, i64 1, !dbg !37
  %2181 = insertelement <4 x float> %2180, float %2047, i64 2, !dbg !37
  %2182 = insertelement <4 x float> %2181, float %2053, i64 3, !dbg !37
  %2183 = fadd <4 x float> %2178, %2182, !dbg !37
  %2184 = shufflevector <8 x float> %2174, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2185 = fmul <4 x float> %2184, %2177, !dbg !36
  %2186 = insertelement <4 x float> poison, float %2090, i64 0, !dbg !37
  %2187 = insertelement <4 x float> %2186, float %2096, i64 1, !dbg !37
  %2188 = insertelement <4 x float> %2187, float %2091, i64 2, !dbg !37
  %2189 = insertelement <4 x float> %2188, float %2097, i64 3, !dbg !37
  %2190 = fadd <4 x float> %2185, %2189, !dbg !37
  %2191 = extractvalue [2 x <8 x float>] %60, 1, !dbg !36
  %2192 = shufflevector <8 x float> %2191, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2193 = insertelement <4 x float> poison, float %2173, i64 0, !dbg !36
  %2194 = shufflevector <4 x float> %2193, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2195 = fmul <4 x float> %2192, %2194, !dbg !36
  %2196 = insertelement <4 x float> poison, float %2126, i64 0, !dbg !37
  %2197 = insertelement <4 x float> %2196, float %2132, i64 1, !dbg !37
  %2198 = insertelement <4 x float> %2197, float %2127, i64 2, !dbg !37
  %2199 = insertelement <4 x float> %2198, float %2133, i64 3, !dbg !37
  %2200 = fadd <4 x float> %2195, %2199, !dbg !37
  %2201 = shufflevector <8 x float> %2191, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2202 = fmul <4 x float> %2201, %2194, !dbg !36
  %2203 = insertelement <4 x float> poison, float %2162, i64 0, !dbg !37
  %2204 = insertelement <4 x float> %2203, float %2168, i64 1, !dbg !37
  %2205 = insertelement <4 x float> %2204, float %2163, i64 2, !dbg !37
  %2206 = insertelement <4 x float> %2205, float %2169, i64 3, !dbg !37
  %2207 = fadd <4 x float> %2202, %2206, !dbg !37
  %2208 = shufflevector <4 x float> %1509, <4 x float> %1516, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2209 = insertvalue [2 x <8 x float>] poison, <8 x float> %2208, 0, !dbg !7
  %2210 = shufflevector <4 x float> %1526, <4 x float> %1533, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2211 = insertvalue [2 x <8 x float>] %2209, <8 x float> %2210, 1, !dbg !7
  %2212 = insertelement <1 x float> poison, float %1290, i64 0, !dbg !7
  %2213 = insertvalue [2 x <1 x float>] poison, <1 x float> %2212, 0, !dbg !7
  %2214 = insertelement <1 x float> poison, float %1299, i64 0, !dbg !7
  %2215 = insertvalue [2 x <1 x float>] %2213, <1 x float> %2214, 1, !dbg !7
  %2216 = insertvalue [2 x <1 x float>] poison, <1 x float> %999, 0, !dbg !7
  %2217 = insertvalue [2 x <1 x float>] %2216, <1 x float> %1012, 1, !dbg !7
  %2218 = shufflevector <4 x float> %1739, <4 x float> %1746, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2219 = insertvalue [2 x <8 x float>] poison, <8 x float> %2218, 0, !dbg !7
  %2220 = shufflevector <4 x float> %1756, <4 x float> %1763, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2221 = insertvalue [2 x <8 x float>] %2219, <8 x float> %2220, 1, !dbg !7
  %2222 = shufflevector <4 x float> %1969, <4 x float> %1976, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2223 = insertvalue [2 x <8 x float>] poison, <8 x float> %2222, 0, !dbg !7
  %2224 = shufflevector <4 x float> %1986, <4 x float> %1993, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2225 = insertvalue [2 x <8 x float>] %2223, <8 x float> %2224, 1, !dbg !7
  %2226 = shufflevector <4 x float> %2183, <4 x float> %2190, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2227 = insertvalue [2 x <8 x float>] poison, <8 x float> %2226, 0, !dbg !7
  %2228 = shufflevector <4 x float> %2200, <4 x float> %2207, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2229 = insertvalue [2 x <8 x float>] %2227, <8 x float> %2228, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %2230 = add nuw nsw i64 %54, 1, !dbg !7
  br label %53, !dbg !7

2231:                                             ; preds = %53
  %2232 = extractvalue [2 x <1 x float>] %56, 0, !dbg !21
  %2233 = shufflevector <1 x float> %2232, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %2234 = extractvalue [2 x <1 x float>] %56, 1, !dbg !21
  %2235 = shufflevector <1 x float> %2234, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %2236 = extractvalue [2 x <8 x float>] %55, 1, !dbg !21
  %2237 = shufflevector <8 x float> %2236, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2235, !dbg !21
  %normalized = fmul <4 x float> %2237, %row.reciprocal, !dbg !21
  %2238 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !22
  %2239 = extractelement <4 x half> %2238, i64 3, !dbg !22
  %2240 = extractelement <4 x half> %2238, i64 2, !dbg !22
  %2241 = extractelement <4 x half> %2238, i64 1, !dbg !22
  %2242 = extractelement <4 x half> %2238, i64 0, !dbg !22
  %2243 = shufflevector <8 x float> %2236, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized68 = fmul <4 x float> %2243, %row.reciprocal, !dbg !21
  %2244 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !22
  %2245 = extractelement <4 x half> %2244, i64 3, !dbg !22
  %2246 = extractelement <4 x half> %2244, i64 2, !dbg !22
  %2247 = extractelement <4 x half> %2244, i64 1, !dbg !22
  %2248 = extractelement <4 x half> %2244, i64 0, !dbg !22
  %2249 = extractvalue [2 x <8 x float>] %55, 0, !dbg !21
  %2250 = shufflevector <8 x float> %2249, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2233, !dbg !21
  %normalized70 = fmul <4 x float> %2250, %row.reciprocal69, !dbg !21
  %2251 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !22
  %2252 = extractelement <4 x half> %2251, i64 3, !dbg !22
  %2253 = extractelement <4 x half> %2251, i64 2, !dbg !22
  %2254 = extractelement <4 x half> %2251, i64 1, !dbg !22
  %2255 = extractelement <4 x half> %2251, i64 0, !dbg !22
  %2256 = shufflevector <8 x float> %2249, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized71 = fmul <4 x float> %2256, %row.reciprocal69, !dbg !21
  %2257 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !22
  %2258 = extractelement <4 x half> %2257, i64 3, !dbg !22
  %2259 = extractelement <4 x half> %2257, i64 2, !dbg !22
  %2260 = extractelement <4 x half> %2257, i64 1, !dbg !22
  %2261 = extractelement <4 x half> %2257, i64 0, !dbg !22
  %2262 = add nsw i64 %15, %25, !dbg !23
  %2263 = shl i64 %2262, 19, !dbg !23
  %2264 = and i64 %2263, 16252928, !dbg !23
  %address.thread1072 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1073 = zext nneg i32 %address.thread1072 to i64, !dbg !12
  %address.local1074 = add i64 %28, %address.local1073, !dbg !23
  %address.local1075 = lshr i64 %address.local1073, 2, !dbg !13
  %address.local1076 = and i64 %address.local1075, 48, !dbg !13
  %address.local1077 = add i64 %address.local1074, %address.local1076, !dbg !23
  %address.local1078 = and i64 %address.local1073, 240, !dbg !23
  %address.local1080 = sub i64 %address.local1077, %address.local1078, !dbg !23
  %address.local1081 = add i64 %address.local1080, %29, !dbg !23
  %address.local1082 = shl nuw nsw i64 %address.local1081, 7, !dbg !23
  %address.local1083 = add nuw nsw i64 %2264, %address.local1082, !dbg !23
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %2265 = lshr i64 %address.local1073, 2, !dbg !12
  %address.local1087 = and i64 %2265, 12, !dbg !12
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1087, !dbg !23
  store half %2261, ptr addrspace(1) %address.local1088, align 2, !dbg !23
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1090 = or disjoint i64 %address.local1087, 1, !dbg !13
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1090, !dbg !23
  store half %2260, ptr addrspace(1) %address.local1091, align 2, !dbg !23
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1093 = or disjoint i64 %address.local1087, 2, !dbg !13
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1093, !dbg !23
  store half %2259, ptr addrspace(1) %address.local1094, align 2, !dbg !23
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1096 = or disjoint i64 %address.local1087, 3, !dbg !13
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1096, !dbg !23
  store half %2258, ptr addrspace(1) %address.local1097, align 2, !dbg !23
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1099 = or disjoint i64 %address.local1087, 16, !dbg !13
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1098, i64 %address.local1099, !dbg !23
  store half %2255, ptr addrspace(1) %address.local1100, align 2, !dbg !23
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1102 = or disjoint i64 %address.local1087, 17, !dbg !13
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !23
  store half %2254, ptr addrspace(1) %address.local1103, align 2, !dbg !23
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1105 = or disjoint i64 %address.local1087, 18, !dbg !13
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !23
  store half %2253, ptr addrspace(1) %address.local1106, align 2, !dbg !23
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1108 = or disjoint i64 %address.local1087, 19, !dbg !13
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !23
  store half %2252, ptr addrspace(1) %address.local1109, align 2, !dbg !23
  %address.local1110 = shl i64 %address.local1081, 7, !dbg !23
  %address.local1111 = add i64 %address.local1110, 8192, !dbg !23
  %address.local1112 = add nuw nsw i64 %2264, %address.local1111, !dbg !23
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1087, !dbg !23
  store half %2248, ptr addrspace(1) %address.local1114, align 2, !dbg !23
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1090, !dbg !23
  store half %2247, ptr addrspace(1) %address.local1116, align 2, !dbg !23
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1093, !dbg !23
  store half %2246, ptr addrspace(1) %address.local1118, align 2, !dbg !23
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1096, !dbg !23
  store half %2245, ptr addrspace(1) %address.local1120, align 2, !dbg !23
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1099, !dbg !23
  store half %2242, ptr addrspace(1) %address.local1122, align 2, !dbg !23
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1102, !dbg !23
  store half %2241, ptr addrspace(1) %address.local1124, align 2, !dbg !23
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1105, !dbg !23
  store half %2240, ptr addrspace(1) %address.local1126, align 2, !dbg !23
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1108, !dbg !23
  store half %2239, ptr addrspace(1) %address.local1128, align 2, !dbg !23
  %2266 = extractvalue [2 x <8 x float>] %58, 1, !dbg !21
  %2267 = shufflevector <8 x float> %2266, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2235, !dbg !21
  %normalized73 = fmul <4 x float> %2267, %row.reciprocal72, !dbg !21
  %2268 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !22
  %2269 = extractelement <4 x half> %2268, i64 3, !dbg !22
  %2270 = extractelement <4 x half> %2268, i64 2, !dbg !22
  %2271 = extractelement <4 x half> %2268, i64 1, !dbg !22
  %2272 = extractelement <4 x half> %2268, i64 0, !dbg !22
  %2273 = shufflevector <8 x float> %2266, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized74 = fmul <4 x float> %2273, %row.reciprocal72, !dbg !21
  %2274 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !22
  %2275 = extractelement <4 x half> %2274, i64 3, !dbg !22
  %2276 = extractelement <4 x half> %2274, i64 2, !dbg !22
  %2277 = extractelement <4 x half> %2274, i64 1, !dbg !22
  %2278 = extractelement <4 x half> %2274, i64 0, !dbg !22
  %2279 = extractvalue [2 x <8 x float>] %58, 0, !dbg !21
  %2280 = shufflevector <8 x float> %2279, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2233, !dbg !21
  %normalized76 = fmul <4 x float> %2280, %row.reciprocal75, !dbg !21
  %2281 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !22
  %2282 = extractelement <4 x half> %2281, i64 3, !dbg !22
  %2283 = extractelement <4 x half> %2281, i64 2, !dbg !22
  %2284 = extractelement <4 x half> %2281, i64 1, !dbg !22
  %2285 = extractelement <4 x half> %2281, i64 0, !dbg !22
  %2286 = shufflevector <8 x float> %2279, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized77 = fmul <4 x float> %2286, %row.reciprocal75, !dbg !21
  %2287 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !22
  %2288 = extractelement <4 x half> %2287, i64 3, !dbg !22
  %2289 = extractelement <4 x half> %2287, i64 2, !dbg !22
  %2290 = extractelement <4 x half> %2287, i64 1, !dbg !22
  %2291 = extractelement <4 x half> %2287, i64 0, !dbg !22
  %2292 = shl nuw nsw i64 %27, 19, !dbg !23
  %address.thread1129 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1130 = zext nneg i32 %address.thread1129 to i64, !dbg !12
  %address.local1131 = add i64 %28, %address.local1130, !dbg !23
  %address.local1132 = lshr i64 %address.local1130, 2, !dbg !13
  %address.local1133 = and i64 %address.local1132, 48, !dbg !13
  %address.local1134 = add i64 %address.local1131, %address.local1133, !dbg !23
  %address.local1135 = and i64 %address.local1130, 240, !dbg !23
  %address.local1137 = sub i64 %address.local1134, %address.local1135, !dbg !23
  %address.local1138 = add i64 %address.local1137, %29, !dbg !23
  %address.local1139 = shl nuw nsw i64 %address.local1138, 7, !dbg !23
  %address.local1140 = add nuw nsw i64 %2292, %address.local1139, !dbg !23
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %2293 = lshr i64 %address.local1130, 2, !dbg !12
  %address.local1144 = and i64 %2293, 12, !dbg !12
  %address.local1145 = or disjoint i64 %address.local1144, 32, !dbg !13
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1145, !dbg !23
  store half %2291, ptr addrspace(1) %address.local1146, align 2, !dbg !23
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1148 = or disjoint i64 %address.local1144, 33, !dbg !13
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1148, !dbg !23
  store half %2290, ptr addrspace(1) %address.local1149, align 2, !dbg !23
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1151 = or disjoint i64 %address.local1144, 34, !dbg !13
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1150, i64 %address.local1151, !dbg !23
  store half %2289, ptr addrspace(1) %address.local1152, align 2, !dbg !23
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1154 = or disjoint i64 %address.local1144, 35, !dbg !13
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1154, !dbg !23
  store half %2288, ptr addrspace(1) %address.local1155, align 2, !dbg !23
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1157 = or disjoint i64 %address.local1144, 48, !dbg !13
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1156, i64 %address.local1157, !dbg !23
  store half %2285, ptr addrspace(1) %address.local1158, align 2, !dbg !23
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1160 = or disjoint i64 %address.local1144, 49, !dbg !13
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1160, !dbg !23
  store half %2284, ptr addrspace(1) %address.local1161, align 2, !dbg !23
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1163 = or disjoint i64 %address.local1144, 50, !dbg !13
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1162, i64 %address.local1163, !dbg !23
  store half %2283, ptr addrspace(1) %address.local1164, align 2, !dbg !23
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1166 = or i64 %2293, 51, !dbg !13
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1166, !dbg !23
  store half %2282, ptr addrspace(1) %address.local1167, align 2, !dbg !23
  %address.local1168 = shl i64 %address.local1138, 7, !dbg !23
  %address.local1169 = add i64 %address.local1168, 8192, !dbg !23
  %address.local1170 = add nuw nsw i64 %2292, %address.local1169, !dbg !23
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1145, !dbg !23
  store half %2278, ptr addrspace(1) %address.local1172, align 2, !dbg !23
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1148, !dbg !23
  store half %2277, ptr addrspace(1) %address.local1174, align 2, !dbg !23
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1151, !dbg !23
  store half %2276, ptr addrspace(1) %address.local1176, align 2, !dbg !23
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1154, !dbg !23
  store half %2275, ptr addrspace(1) %address.local1178, align 2, !dbg !23
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1157, !dbg !23
  store half %2272, ptr addrspace(1) %address.local1180, align 2, !dbg !23
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1160, !dbg !23
  store half %2271, ptr addrspace(1) %address.local1182, align 2, !dbg !23
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1163, !dbg !23
  store half %2270, ptr addrspace(1) %address.local1184, align 2, !dbg !23
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1166, !dbg !23
  store half %2269, ptr addrspace(1) %address.local1186, align 2, !dbg !23
  %2294 = extractvalue [2 x <8 x float>] %59, 1, !dbg !21
  %2295 = shufflevector <8 x float> %2294, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2235, !dbg !21
  %normalized79 = fmul <4 x float> %2295, %row.reciprocal78, !dbg !21
  %2296 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !22
  %2297 = extractelement <4 x half> %2296, i64 3, !dbg !22
  %2298 = extractelement <4 x half> %2296, i64 2, !dbg !22
  %2299 = extractelement <4 x half> %2296, i64 1, !dbg !22
  %2300 = extractelement <4 x half> %2296, i64 0, !dbg !22
  %2301 = shufflevector <8 x float> %2294, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized80 = fmul <4 x float> %2301, %row.reciprocal78, !dbg !21
  %2302 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !22
  %2303 = extractelement <4 x half> %2302, i64 3, !dbg !22
  %2304 = extractelement <4 x half> %2302, i64 2, !dbg !22
  %2305 = extractelement <4 x half> %2302, i64 1, !dbg !22
  %2306 = extractelement <4 x half> %2302, i64 0, !dbg !22
  %2307 = extractvalue [2 x <8 x float>] %59, 0, !dbg !21
  %2308 = shufflevector <8 x float> %2307, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2233, !dbg !21
  %normalized82 = fmul <4 x float> %2308, %row.reciprocal81, !dbg !21
  %2309 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !22
  %2310 = extractelement <4 x half> %2309, i64 3, !dbg !22
  %2311 = extractelement <4 x half> %2309, i64 2, !dbg !22
  %2312 = extractelement <4 x half> %2309, i64 1, !dbg !22
  %2313 = extractelement <4 x half> %2309, i64 0, !dbg !22
  %2314 = shufflevector <8 x float> %2307, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized83 = fmul <4 x float> %2314, %row.reciprocal81, !dbg !21
  %2315 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !22
  %2316 = extractelement <4 x half> %2315, i64 3, !dbg !22
  %2317 = extractelement <4 x half> %2315, i64 2, !dbg !22
  %2318 = extractelement <4 x half> %2315, i64 1, !dbg !22
  %2319 = extractelement <4 x half> %2315, i64 0, !dbg !22
  %2320 = shl nuw nsw i64 %27, 19, !dbg !23
  %address.thread1187 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1188 = zext nneg i32 %address.thread1187 to i64, !dbg !12
  %address.local1189 = add i64 %28, %address.local1188, !dbg !23
  %address.local1190 = lshr i64 %address.local1188, 2, !dbg !13
  %address.local1191 = and i64 %address.local1190, 48, !dbg !13
  %address.local1192 = add i64 %address.local1189, %address.local1191, !dbg !23
  %address.local1193 = and i64 %address.local1188, 240, !dbg !23
  %address.local1195 = sub i64 %address.local1192, %address.local1193, !dbg !23
  %address.local1196 = add i64 %address.local1195, %29, !dbg !23
  %address.local1197 = shl nuw nsw i64 %address.local1196, 7, !dbg !23
  %address.local1198 = add nuw nsw i64 %2320, %address.local1197, !dbg !23
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %2321 = lshr i64 %address.local1188, 2, !dbg !12
  %address.local1202 = and i64 %2321, 12, !dbg !12
  %address.local1203 = or disjoint i64 %address.local1202, 64, !dbg !13
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1203, !dbg !23
  store half %2319, ptr addrspace(1) %address.local1204, align 2, !dbg !23
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1206 = or disjoint i64 %address.local1202, 65, !dbg !13
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1206, !dbg !23
  store half %2318, ptr addrspace(1) %address.local1207, align 2, !dbg !23
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1209 = or disjoint i64 %address.local1202, 66, !dbg !13
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1208, i64 %address.local1209, !dbg !23
  store half %2317, ptr addrspace(1) %address.local1210, align 2, !dbg !23
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1212 = or disjoint i64 %address.local1202, 67, !dbg !13
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1211, i64 %address.local1212, !dbg !23
  store half %2316, ptr addrspace(1) %address.local1213, align 2, !dbg !23
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1215 = or disjoint i64 %address.local1202, 80, !dbg !13
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1214, i64 %address.local1215, !dbg !23
  store half %2313, ptr addrspace(1) %address.local1216, align 2, !dbg !23
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1218 = or disjoint i64 %address.local1202, 81, !dbg !13
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1217, i64 %address.local1218, !dbg !23
  store half %2312, ptr addrspace(1) %address.local1219, align 2, !dbg !23
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1221 = or disjoint i64 %address.local1202, 82, !dbg !13
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1221, !dbg !23
  store half %2311, ptr addrspace(1) %address.local1222, align 2, !dbg !23
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1224 = or disjoint i64 %address.local1202, 83, !dbg !13
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1223, i64 %address.local1224, !dbg !23
  store half %2310, ptr addrspace(1) %address.local1225, align 2, !dbg !23
  %address.local1226 = shl i64 %address.local1196, 7, !dbg !23
  %address.local1227 = add i64 %address.local1226, 8192, !dbg !23
  %address.local1228 = add nuw nsw i64 %2320, %address.local1227, !dbg !23
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1229, i64 %address.local1203, !dbg !23
  store half %2306, ptr addrspace(1) %address.local1230, align 2, !dbg !23
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1231, i64 %address.local1206, !dbg !23
  store half %2305, ptr addrspace(1) %address.local1232, align 2, !dbg !23
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1233, i64 %address.local1209, !dbg !23
  store half %2304, ptr addrspace(1) %address.local1234, align 2, !dbg !23
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1235, i64 %address.local1212, !dbg !23
  store half %2303, ptr addrspace(1) %address.local1236, align 2, !dbg !23
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1237, i64 %address.local1215, !dbg !23
  store half %2300, ptr addrspace(1) %address.local1238, align 2, !dbg !23
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1239, i64 %address.local1218, !dbg !23
  store half %2299, ptr addrspace(1) %address.local1240, align 2, !dbg !23
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1241, i64 %address.local1221, !dbg !23
  store half %2298, ptr addrspace(1) %address.local1242, align 2, !dbg !23
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1243, i64 %address.local1224, !dbg !23
  store half %2297, ptr addrspace(1) %address.local1244, align 2, !dbg !23
  %2322 = extractvalue [2 x <8 x float>] %60, 1, !dbg !21
  %2323 = shufflevector <8 x float> %2322, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2235, !dbg !21
  %normalized85 = fmul <4 x float> %2323, %row.reciprocal84, !dbg !21
  %2324 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !22
  %2325 = extractelement <4 x half> %2324, i64 3, !dbg !22
  %2326 = extractelement <4 x half> %2324, i64 2, !dbg !22
  %2327 = extractelement <4 x half> %2324, i64 1, !dbg !22
  %2328 = extractelement <4 x half> %2324, i64 0, !dbg !22
  %2329 = shufflevector <8 x float> %2322, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized86 = fmul <4 x float> %2329, %row.reciprocal84, !dbg !21
  %2330 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !22
  %2331 = extractelement <4 x half> %2330, i64 3, !dbg !22
  %2332 = extractelement <4 x half> %2330, i64 2, !dbg !22
  %2333 = extractelement <4 x half> %2330, i64 1, !dbg !22
  %2334 = extractelement <4 x half> %2330, i64 0, !dbg !22
  %2335 = extractvalue [2 x <8 x float>] %60, 0, !dbg !21
  %2336 = shufflevector <8 x float> %2335, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2233, !dbg !21
  %normalized88 = fmul <4 x float> %2336, %row.reciprocal87, !dbg !21
  %2337 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !22
  %2338 = extractelement <4 x half> %2337, i64 3, !dbg !22
  %2339 = extractelement <4 x half> %2337, i64 2, !dbg !22
  %2340 = extractelement <4 x half> %2337, i64 1, !dbg !22
  %2341 = extractelement <4 x half> %2337, i64 0, !dbg !22
  %2342 = shufflevector <8 x float> %2335, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized89 = fmul <4 x float> %2342, %row.reciprocal87, !dbg !21
  %2343 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !22
  %2344 = extractelement <4 x half> %2343, i64 3, !dbg !22
  %2345 = extractelement <4 x half> %2343, i64 2, !dbg !22
  %2346 = extractelement <4 x half> %2343, i64 1, !dbg !22
  %2347 = extractelement <4 x half> %2343, i64 0, !dbg !22
  %2348 = shl nuw nsw i64 %27, 19, !dbg !23
  %address.thread1245 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1246 = zext nneg i32 %address.thread1245 to i64, !dbg !12
  %address.local1247 = add i64 %28, %address.local1246, !dbg !23
  %address.local1248 = lshr i64 %address.local1246, 2, !dbg !13
  %address.local1249 = and i64 %address.local1248, 48, !dbg !13
  %address.local1250 = add i64 %address.local1247, %address.local1249, !dbg !23
  %address.local1251 = and i64 %address.local1246, 240, !dbg !23
  %address.local1253 = sub i64 %address.local1250, %address.local1251, !dbg !23
  %address.local1254 = add i64 %address.local1253, %29, !dbg !23
  %address.local1255 = shl nuw nsw i64 %address.local1254, 7, !dbg !23
  %address.local1256 = add nuw nsw i64 %2348, %address.local1255, !dbg !23
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %2349 = lshr i64 %address.local1246, 2, !dbg !12
  %address.local1260 = and i64 %2349, 12, !dbg !12
  %address.local1261 = or disjoint i64 %address.local1260, 96, !dbg !13
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1261, !dbg !23
  store half %2347, ptr addrspace(1) %address.local1262, align 2, !dbg !23
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1264 = or disjoint i64 %address.local1260, 97, !dbg !13
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1264, !dbg !23
  store half %2346, ptr addrspace(1) %address.local1265, align 2, !dbg !23
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1267 = or disjoint i64 %address.local1260, 98, !dbg !13
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1266, i64 %address.local1267, !dbg !23
  store half %2345, ptr addrspace(1) %address.local1268, align 2, !dbg !23
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1270 = or disjoint i64 %address.local1260, 99, !dbg !13
  %address.local1271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1269, i64 %address.local1270, !dbg !23
  store half %2344, ptr addrspace(1) %address.local1271, align 2, !dbg !23
  %address.local1272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1273 = or disjoint i64 %address.local1260, 112, !dbg !13
  %address.local1274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1272, i64 %address.local1273, !dbg !23
  store half %2341, ptr addrspace(1) %address.local1274, align 2, !dbg !23
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1276 = or disjoint i64 %address.local1260, 113, !dbg !13
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1275, i64 %address.local1276, !dbg !23
  store half %2340, ptr addrspace(1) %address.local1277, align 2, !dbg !23
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1279 = or disjoint i64 %address.local1260, 114, !dbg !13
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1278, i64 %address.local1279, !dbg !23
  store half %2339, ptr addrspace(1) %address.local1280, align 2, !dbg !23
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1282 = or i64 %2349, 115, !dbg !13
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1281, i64 %address.local1282, !dbg !23
  store half %2338, ptr addrspace(1) %address.local1283, align 2, !dbg !23
  %address.local1284 = shl i64 %address.local1254, 7, !dbg !23
  %address.local1285 = add i64 %address.local1284, 8192, !dbg !23
  %address.local1286 = add nuw nsw i64 %2348, %address.local1285, !dbg !23
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1287, i64 %address.local1261, !dbg !23
  store half %2334, ptr addrspace(1) %address.local1288, align 2, !dbg !23
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1264, !dbg !23
  store half %2333, ptr addrspace(1) %address.local1290, align 2, !dbg !23
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1291, i64 %address.local1267, !dbg !23
  store half %2332, ptr addrspace(1) %address.local1292, align 2, !dbg !23
  %address.local1293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1293, i64 %address.local1270, !dbg !23
  store half %2331, ptr addrspace(1) %address.local1294, align 2, !dbg !23
  %address.local1295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1295, i64 %address.local1273, !dbg !23
  store half %2328, ptr addrspace(1) %address.local1296, align 2, !dbg !23
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1297, i64 %address.local1276, !dbg !23
  store half %2327, ptr addrspace(1) %address.local1298, align 2, !dbg !23
  %address.local1299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1299, i64 %address.local1279, !dbg !23
  store half %2326, ptr addrspace(1) %address.local1300, align 2, !dbg !23
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1282, !dbg !23
  store half %2325, ptr addrspace(1) %address.local1302, align 2, !dbg !23
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #6

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #8 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!39}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-152102-801zvuny/grid-000001")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!9 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 70, column: 22, scope: !4)
!12 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 63, column: 17, scope: !4)
!14 = !DILocation(line: 77, column: 19, scope: !4)
!15 = !DILocation(line: 92, column: 20, scope: !4)
!16 = !DILocation(line: 102, column: 23, scope: !4)
!17 = !DILocation(line: 0, scope: !4)
!18 = !DILocation(line: 71, column: 22, scope: !4)
!19 = !DILocation(line: 73, column: 26, scope: !4)
!20 = !DILocation(line: 101, column: 20, scope: !4)
!21 = !DILocation(line: 107, column: 23, scope: !4)
!22 = !DILocation(line: 108, column: 19, scope: !4)
!23 = !DILocation(line: 109, column: 9, scope: !4)
!24 = !DILocation(line: 74, column: 23, scope: !4)
!25 = !DILocation(line: 79, column: 24, scope: !4)
!26 = !DILocation(line: 80, column: 23, scope: !4)
!27 = !DILocation(line: 81, column: 26, scope: !4)
!28 = !DILocation(line: 88, column: 21, scope: !4)
!29 = !DILocation(line: 89, column: 20, scope: !4)
!30 = !DILocation(line: 90, column: 22, scope: !4)
!31 = !DILocation(line: 91, column: 23, scope: !4)
!32 = !DILocation(line: 93, column: 22, scope: !4)
!33 = !DILocation(line: 94, column: 22, scope: !4)
!34 = !DILocation(line: 95, column: 20, scope: !4)
!35 = !DILocation(line: 96, column: 16, scope: !4)
!36 = !DILocation(line: 103, column: 22, scope: !4)
!37 = !DILocation(line: 104, column: 20, scope: !4)
!38 = !{i32 256, i32 1, i32 1}
!39 = !{i32 2, !"Debug Info Version", i32 3}
!40 = !{i32 0, i32 256}
