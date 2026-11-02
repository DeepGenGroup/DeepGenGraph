; ModuleID = 'generated_ir/gemm_result_layout_comparison/exp2_fixed/layout_3_1/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000001/input.mlir"

@shm_0 = addrspace(3) global [65536 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !40 !reqd_work_group_size !71 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !41
  %6 = sext i32 %5 to i64, !dbg !41
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !41
  %8 = sext i32 %7 to i64, !dbg !41
  %9 = shl nsw i64 %8, 5, !dbg !41
  %10 = add nsw i64 %9, %6, !dbg !41
  %11 = lshr i64 %10, 5, !dbg !41
  %12 = and i64 %11, 576460752303423484, !dbg !41
  %13 = and i64 %6, 3, !dbg !41
  %14 = or disjoint i64 %12, %13, !dbg !41
  %15 = lshr i64 %10, 2, !dbg !41
  %16 = sub nsw i64 31, %14, !dbg !41
  %17 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !41
  %18 = sext i32 %17 to i64, !dbg !41
  %19 = shl i64 %16, 7, !dbg !43
  %20 = icmp ugt i64 %10, 1023, !dbg !44
  %21 = add nsw i64 %14, -32, !dbg !44
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !44
  %23 = sdiv i64 %22, 32, !dbg !44
  %24 = sext i1 %20 to i64, !dbg !44
  %25 = xor i64 %23, %24, !dbg !44
  %26 = add nsw i64 %15, %25, !dbg !44
  %27 = and i64 %26, 31, !dbg !44
  %28 = shl nsw i64 %16, 7, !dbg !44
  %29 = mul nsw i64 %25, -4096, !dbg !44
  %30 = add i64 %28, %29, !dbg !44
  %31 = shl nuw nsw i64 %27, 19, !dbg !44
  %32 = shl nsw i64 %30, 7, !dbg !44
  %33 = add i64 %31, %32, !dbg !44
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !45, !range !73
  %35 = zext nneg i32 %34 to i64, !dbg !45
  %36 = lshr i64 %35, 4, !dbg !45
  %37 = and i64 %36, 3, !dbg !45
  %38 = shl nuw nsw i64 %37, 2, !dbg !45
  %39 = and i64 %35, 240, !dbg !47
  %40 = or disjoint i64 %19, 127, !dbg !47
  %41 = add i64 %19, 128, !dbg !47
  %42 = icmp uge i64 %41, %40, !dbg !47
  %43 = icmp eq i64 %37, 0, !dbg !48
  %44 = and i32 %34, 16, !dbg !49
  %.not = icmp eq i32 %44, 0, !dbg !49
  %45 = and i32 %34, 32, !dbg !49
  %.not1 = icmp eq i32 %45, 0, !dbg !49
  %46 = sub nsw i64 32, %14, !dbg !50
  %47 = shl i64 %15, 7, !dbg !51
  %48 = and i64 %47, 3968, !dbg !51
  %49 = shl nsw i64 %18, 12, !dbg !51
  %50 = or disjoint i64 %48, %49, !dbg !51
  %51 = lshr i64 %35, 2, !dbg !46
  %52 = and i64 %51, 48, !dbg !46
  br label %53, !dbg !42

53:                                               ; preds = %2240, %4
  %54 = phi i64 [ %2299, %2240 ], [ 0, %4 ], !dbg !42
  %55 = phi [2 x <8 x float>] [ %2280, %2240 ], [ zeroinitializer, %4 ], !dbg !42
  %56 = phi [2 x <1 x float>] [ %2284, %2240 ], [ zeroinitializer, %4 ], !dbg !42
  %57 = phi [2 x <1 x float>] [ %2286, %2240 ], [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], !dbg !42
  %58 = phi [2 x <8 x float>] [ %2290, %2240 ], [ zeroinitializer, %4 ], !dbg !42
  %59 = phi [2 x <8 x float>] [ %2294, %2240 ], [ zeroinitializer, %4 ], !dbg !42
  %60 = phi [2 x <8 x float>] [ %2298, %2240 ], [ zeroinitializer, %4 ], !dbg !42
  %61 = icmp slt i64 %54, %46, !dbg !42
  br i1 %61, label %62, label %2300, !dbg !42

62:                                               ; preds = %53
  %63 = shl nsw i64 %54, 7, !dbg !42
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !44, !range !73, !frisk.register_lifetimes !37
  %address.local = zext nneg i32 %address.thread to i64, !dbg !45
  %address.local258 = shl nuw nsw i64 %address.local, 3, !dbg !44
  %address.local259 = and i64 %address.local258, 1536, !dbg !44
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local259, !dbg !44
  %address.local261 = ptrtoint ptr addrspace(3) %address.local260 to i64, !dbg !44
  %address.local262 = trunc i64 %address.local261 to i32, !dbg !44
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !44
  %address.local263 = shl nuw nsw i64 %address.local, 7, !dbg !44
  %address.local264 = and i64 %address.local263, 16256, !dbg !44
  %address.local265 = lshr i64 %address.local, 4, !dbg !44
  %address.local266 = and i64 %address.local265, 8, !dbg !44
  %address.local267 = or disjoint i64 %address.local264, %address.local266, !dbg !44
  %65 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !44
  %.idx1 = or disjoint i64 %33, %address.local267, !dbg !44
  %.idx1.tr = trunc i64 %.idx1 to i32, !dbg !44
  %dma.buffer.offset1821 = shl i32 %.idx1.tr, 1, !dbg !44
  %66 = insertelement <2 x i64> poison, i64 %65, i64 0, !dbg !44
  %67 = bitcast <2 x i64> %66 to <4 x i32>, !dbg !44
  %dma.buffer.rsrc1822 = shufflevector <4 x i32> %67, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !44
  %dma.buffer.lds1823 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64), !dbg !44
  %68 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1821, <4 x i32> %dma.buffer.rsrc1822) #10, !dbg !44
  %address.local269 = or disjoint i64 %address.local, 256, !dbg !44
  %address.local270 = shl nuw nsw i64 %address.local269, 3, !dbg !44
  %address.local271 = and i64 %address.local270, 3584, !dbg !44
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local271, !dbg !44
  %address.local273 = ptrtoint ptr addrspace(3) %address.local272 to i64, !dbg !44
  %address.local274 = trunc i64 %address.local273 to i32, !dbg !44
  %69 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !44
  %address.local275 = shl nuw nsw i64 %address.local, 7, !dbg !44
  %address.local276 = and i64 %address.local275, 16256, !dbg !44
  %address.local277 = lshr i64 %address.local269, 4, !dbg !44
  %address.local278 = and i64 %address.local277, 24, !dbg !44
  %address.local279 = or disjoint i64 %address.local276, %address.local278, !dbg !44
  %70 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !44
  %.idx2 = or disjoint i64 %33, %address.local279, !dbg !44
  %.idx2.tr = trunc i64 %.idx2 to i32, !dbg !44
  %dma.buffer.offset1818 = shl i32 %.idx2.tr, 1, !dbg !44
  %71 = insertelement <2 x i64> poison, i64 %70, i64 0, !dbg !44
  %72 = bitcast <2 x i64> %71 to <4 x i32>, !dbg !44
  %dma.buffer.rsrc1819 = shufflevector <4 x i32> %72, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !44
  %dma.buffer.lds1820 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %69), !dbg !44
  %73 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1818, <4 x i32> %dma.buffer.rsrc1819) #10, !dbg !44
  %74 = lshr i64 %54, 12, !dbg !51
  %75 = add nuw nsw i64 %74, %15, !dbg !51
  %76 = lshr i64 %54, 5, !dbg !51
  %77 = and i64 %76, 127, !dbg !51
  %78 = shl i64 %75, 19, !dbg !51
  %79 = and i64 %78, 16252928, !dbg !51
  %80 = shl i64 %76, 19, !dbg !51
  %81 = shl i64 %54, 14, !dbg !51
  %82 = sub i64 %81, %80, !dbg !51
  %83 = getelementptr inbounds nuw i8, ptr addrspace(3) @shm_0, i64 8192, !dbg !52
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %83, i64 %address.local259, !dbg !51
  %address.local282 = ptrtoint ptr addrspace(3) %address.local281 to i64, !dbg !51
  %address.local283 = trunc i64 %address.local282 to i32, !dbg !51
  %84 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !51
  %address.local284 = lshr i64 %address.local, 4, !dbg !51
  %address.local285 = and i64 %address.local284, 8, !dbg !51
  %address.local286 = and i64 %address.local, 123, !dbg !51
  %address.local287 = lshr i64 %address.local, 2, !dbg !51
  %address.local288 = and i64 %address.local287, 4, !dbg !51
  %address.local289 = add nuw nsw i64 %address.local288, %address.local, !dbg !51
  %address.local290 = and i64 %address.local289, 4, !dbg !51
  %address.local291 = or disjoint i64 %address.local286, %address.local290, !dbg !51
  %address.local292 = shl nuw nsw i64 %address.local291, 7, !dbg !51
  %address.local293 = or disjoint i64 %address.local285, %address.local292, !dbg !51
  %85 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !51
  %.idx5 = or disjoint i64 %79, %77, !dbg !51
  %86 = add i64 %.idx5, %82, !dbg !51
  %87 = add i64 %86, %address.local293, !dbg !51
  %.tr = trunc i64 %87 to i32, !dbg !51
  %dma.buffer.offset1815 = shl i32 %.tr, 1, !dbg !51
  %88 = insertelement <2 x i64> poison, i64 %85, i64 0, !dbg !51
  %89 = bitcast <2 x i64> %88 to <4 x i32>, !dbg !51
  %dma.buffer.rsrc1816 = shufflevector <4 x i32> %89, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !51
  %dma.buffer.lds1817 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !51
  %90 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1815, <4 x i32> %dma.buffer.rsrc1816) #10, !dbg !51
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) %83, i64 %address.local271, !dbg !51
  %address.local296 = ptrtoint ptr addrspace(3) %address.local295 to i64, !dbg !51
  %address.local297 = trunc i64 %address.local296 to i32, !dbg !51
  %91 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !51
  %address.local298 = lshr i64 %address.local269, 4, !dbg !51
  %address.local299 = and i64 %address.local298, 24, !dbg !51
  %address.local300 = and i64 %address.local, 123, !dbg !51
  %address.local301 = lshr i64 %address.local, 2, !dbg !51
  %address.local302 = and i64 %address.local301, 4, !dbg !51
  %address.local303 = add nuw nsw i64 %address.local302, %address.local, !dbg !51
  %address.local304 = and i64 %address.local303, 4, !dbg !51
  %address.local305 = or disjoint i64 %address.local300, %address.local304, !dbg !51
  %address.local306 = shl nuw nsw i64 %address.local305, 7, !dbg !51
  %address.local307 = or disjoint i64 %address.local299, %address.local306, !dbg !51
  %92 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !51
  %.idx8 = or disjoint i64 %79, %77, !dbg !51
  %93 = add i64 %.idx8, %82, !dbg !51
  %94 = add i64 %93, %address.local307, !dbg !51
  %.tr9 = trunc i64 %94 to i32, !dbg !51
  %dma.buffer.offset1812 = shl i32 %.tr9, 1, !dbg !51
  %95 = insertelement <2 x i64> poison, i64 %92, i64 0, !dbg !51
  %96 = bitcast <2 x i64> %95 to <4 x i32>, !dbg !51
  %dma.buffer.rsrc1813 = shufflevector <4 x i32> %96, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !51
  %dma.buffer.lds1814 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %91), !dbg !51
  %97 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1812, <4 x i32> %dma.buffer.rsrc1813) #10, !dbg !51
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !44
  fence syncscope("workgroup") release, !dbg !44
  call void @llvm.amdgcn.s.barrier(), !dbg !44
  fence syncscope("workgroup") acquire, !dbg !44
  %address.local309 = lshr i32 %address.thread, 2, !dbg !45
  %address.local310 = and i32 %address.local309, 4, !dbg !45
  %address.local311 = shl nuw nsw i32 %address.thread, 1, !dbg !45
  %address.local312 = and i32 %address.local311, 384, !dbg !45
  %address.local313 = or disjoint i32 %address.local310, %address.local312, !dbg !45
  %address.local314 = shl nuw nsw i32 %address.thread, 3, !dbg !45
  %address.local315 = and i32 %address.local314, 120, !dbg !45
  %address.local316 = or disjoint i32 %address.local313, %address.local315, !dbg !45
  %address.local317 = shl nuw nsw i32 %address.thread, 5, !dbg !45
  %address.local318 = and i32 %address.local317, 1024, !dbg !45
  %address.local319 = or disjoint i32 %address.local316, %address.local318, !dbg !45
  %address.local320 = zext nneg i32 %address.local319 to i64, !dbg !45
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local320, !dbg !45
  %98 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !45
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !45
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !45
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !45
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !45
  %101 = fmul <4 x half> %98, splat (half 1.275630e-01), !dbg !45
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !45
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !45
  %103 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !45
  %104 = fmul <4 x half> %100, splat (half 1.275630e-01), !dbg !45
  %105 = fmul <4 x half> %102, splat (half 1.275630e-01), !dbg !45
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !51
  fence syncscope("workgroup") release, !dbg !51
  call void @llvm.amdgcn.s.barrier(), !dbg !51
  fence syncscope("workgroup") acquire, !dbg !51
  %address.local328 = lshr i32 %address.thread, 2, !dbg !52
  %address.local329 = and i32 %address.local328, 4, !dbg !52
  %address.local330 = shl nuw nsw i32 %address.thread, 3, !dbg !52
  %address.local331 = and i32 %address.local330, 120, !dbg !52
  %address.local332 = or disjoint i32 %address.local329, %address.local331, !dbg !52
  %address.local333 = shl nuw nsw i32 %address.thread, 5, !dbg !52
  %address.local334 = and i32 %address.local333, 1024, !dbg !52
  %address.local335 = or disjoint i32 %address.local332, %address.local334, !dbg !52
  %address.local336 = zext nneg i32 %address.local335 to i64, !dbg !52
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) %83, i64 %address.local336, !dbg !52
  %106 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !52
  %107 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %106, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1561 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !52
  %108 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !52
  %address.local349 = lshr i32 %address.thread, 2, !dbg !52
  %address.local350 = and i32 %address.local349, 4, !dbg !52
  %address.local351 = shl nuw nsw i32 %address.thread, 3, !dbg !52
  %address.local352 = and i32 %address.local351, 88, !dbg !52
  %address.local353 = or disjoint i32 %address.local350, %address.local352, !dbg !52
  %address.local354 = shl nuw nsw i32 %address.thread, 3, !dbg !52
  %address.local355 = and i32 %address.local354, 32, !dbg !52
  %address.local356 = sub nsw i32 %address.local353, %address.local355, !dbg !52
  %address.local357 = shl nuw nsw i32 %address.thread, 5, !dbg !52
  %address.local358 = and i32 %address.local357, 1024, !dbg !52
  %address.local359 = add nsw i32 %address.local356, %address.local358, !dbg !52
  %address.local360 = sext i32 %address.local359 to i64, !dbg !52
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) %83, i64 %address.local360, !dbg !52
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local361, i64 320, !dbg !52
  %109 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !52
  %110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %108, <4 x float> %107) #9, !dbg !52
  %111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %109, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1562 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !52
  %112 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !52
  %lds.const.gep1563 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !52
  %113 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !52
  %114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %112, <4 x float> %111) #9, !dbg !52
  %115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %113, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1564 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !52
  %116 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !52
  %lds.const.gep1565 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !52
  %117 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !52
  %118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %116, <4 x float> %115) #9, !dbg !52
  %119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %117, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1566 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !52
  %120 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !52
  %lds.const.gep1567 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !52
  %121 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !52
  %122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %120, <4 x float> %119) #9, !dbg !52
  %123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %121, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1568 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !52
  %124 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !52
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !52
  %125 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !52
  %126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %124, <4 x float> %123) #9, !dbg !52
  %127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %125, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !52
  %128 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !52
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !52
  %129 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !52
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %128, <4 x float> %127) #9, !dbg !52
  %131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %129, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !52
  %132 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !52
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !52
  %133 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !52
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %132, <4 x float> %131) #9, !dbg !52
  %135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %133, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !52
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !52
  %137 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !52
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %136, <4 x float> %135) #9, !dbg !52
  %139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %137, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1575 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !52
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep1575, align 8, !dbg !52
  %141 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !52
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %140, <4 x float> %139) #9, !dbg !52
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %141, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1576 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !52
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep1576, align 8, !dbg !52
  %lds.const.gep1577 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !52
  %145 = load <4 x half>, ptr addrspace(3) %lds.const.gep1577, align 8, !dbg !52
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %144, <4 x float> %143) #9, !dbg !52
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %145, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1578 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !52
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1578, align 8, !dbg !52
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !52
  %149 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !52
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %148, <4 x float> %147) #9, !dbg !52
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %149, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !52
  %152 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !52
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !52
  %153 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !52
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %152, <4 x float> %151) #9, !dbg !52
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !52
  %156 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !52
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !52
  %157 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !52
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %156, <4 x float> %155) #9, !dbg !52
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !52
  %160 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !52
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !52
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !52
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %160, <4 x float> %159) #9, !dbg !52
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %161, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !52
  %164 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !52
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !52
  %165 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !52
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %164, <4 x float> %163) #9, !dbg !52
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !52
  %168 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !52
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %168, <4 x float> %167) #9, !dbg !52
  %170 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !44
  %171 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !44
  %172 = or disjoint i64 %33, 32, !dbg !44
  %173 = or disjoint i64 %address.local267, %172, !dbg !44
  %.tr10 = trunc i64 %173 to i32, !dbg !44
  %dma.buffer.offset1809 = shl i32 %.tr10, 1, !dbg !44
  %174 = insertelement <2 x i64> poison, i64 %171, i64 0, !dbg !44
  %175 = bitcast <2 x i64> %174 to <4 x i32>, !dbg !44
  %dma.buffer.rsrc1810 = shufflevector <4 x i32> %175, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !44
  %dma.buffer.lds1811 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !44
  %176 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1811, i32 %dma.buffer.offset1809, <4 x i32> %dma.buffer.rsrc1810) #10, !dbg !44
  %177 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !44
  %178 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !44
  %179 = or disjoint i64 %33, 32, !dbg !44
  %180 = or disjoint i64 %address.local279, %179, !dbg !44
  %.tr11 = trunc i64 %180 to i32, !dbg !44
  %dma.buffer.offset1806 = shl i32 %.tr11, 1, !dbg !44
  %181 = insertelement <2 x i64> poison, i64 %178, i64 0, !dbg !44
  %182 = bitcast <2 x i64> %181 to <4 x i32>, !dbg !44
  %dma.buffer.rsrc1807 = shufflevector <4 x i32> %182, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !44
  %dma.buffer.lds1808 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %177), !dbg !44
  %183 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1806, <4 x i32> %dma.buffer.rsrc1807) #10, !dbg !44
  %184 = add nuw i64 %54, 1024, !dbg !51
  %185 = icmp slt i64 %184, 0, !dbg !51
  %186 = sub nuw i64 -1025, %54, !dbg !51
  %187 = select i1 %185, i64 %186, i64 %184, !dbg !51
  %188 = sdiv i64 %187, 4096, !dbg !51
  %.lobit = ashr i64 %184, 63, !dbg !51
  %189 = xor i64 %188, %.lobit, !dbg !51
  %190 = add nsw i64 %189, %15, !dbg !51
  %191 = add nsw i64 %76, %50, !dbg !51
  %192 = add nsw i64 %191, 32, !dbg !51
  %193 = icmp slt i64 %191, -32, !dbg !51
  %194 = sub nsw i64 -33, %191, !dbg !51
  %195 = select i1 %193, i64 %194, i64 %192, !dbg !51
  %196 = sdiv i64 %195, 128, !dbg !51
  %197 = sext i1 %193 to i64, !dbg !51
  %198 = xor i64 %196, %197, !dbg !51
  %199 = shl nsw i64 %198, 7, !dbg !51
  %200 = sub i64 %50, %199, !dbg !51
  %201 = shl i64 %190, 19, !dbg !51
  %202 = and i64 %201, 16252928, !dbg !51
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !44
  fence syncscope("workgroup") release, !dbg !44
  call void @llvm.amdgcn.s.barrier(), !dbg !44
  fence syncscope("workgroup") acquire, !dbg !44
  %203 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !51
  %204 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !51
  %.idx15 = add nuw nsw i64 %202, %76, !dbg !51
  %205 = add i64 %.idx15, %200, !dbg !51
  %206 = add i64 %205, %82, !dbg !51
  %207 = add i64 %206, %address.local293, !dbg !51
  %.tr16 = trunc i64 %207 to i32, !dbg !51
  %208 = shl i32 %.tr16, 1, !dbg !51
  %dma.buffer.offset1803 = add i32 %208, 64, !dbg !51
  %209 = insertelement <2 x i64> poison, i64 %204, i64 0, !dbg !51
  %210 = bitcast <2 x i64> %209 to <4 x i32>, !dbg !51
  %dma.buffer.rsrc1804 = shufflevector <4 x i32> %210, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !51
  %dma.buffer.lds1805 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %203), !dbg !51
  %211 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1803, <4 x i32> %dma.buffer.rsrc1804) #10, !dbg !51
  %212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !51
  %213 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !51
  %.idx20 = add nuw nsw i64 %202, %76, !dbg !51
  %214 = add i64 %.idx20, %200, !dbg !51
  %215 = add i64 %214, %82, !dbg !51
  %216 = add i64 %215, %address.local307, !dbg !51
  %.tr21 = trunc i64 %216 to i32, !dbg !51
  %217 = shl i32 %.tr21, 1, !dbg !51
  %dma.buffer.offset1800 = add i32 %217, 64, !dbg !51
  %218 = insertelement <2 x i64> poison, i64 %213, i64 0, !dbg !51
  %219 = bitcast <2 x i64> %218 to <4 x i32>, !dbg !51
  %dma.buffer.rsrc1801 = shufflevector <4 x i32> %219, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !51
  %dma.buffer.lds1802 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %212), !dbg !51
  %220 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1800, <4 x i32> %dma.buffer.rsrc1801) #10, !dbg !51
  %221 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !45
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !45
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !45
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !45
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !45
  %224 = fmul <4 x half> %221, splat (half 1.275630e-01), !dbg !45
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !45
  %225 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !45
  %226 = fmul <4 x half> %222, splat (half 1.275630e-01), !dbg !45
  %227 = fmul <4 x half> %223, splat (half 1.275630e-01), !dbg !45
  %228 = fmul <4 x half> %225, splat (half 1.275630e-01), !dbg !45
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !51
  fence syncscope("workgroup") release, !dbg !51
  call void @llvm.amdgcn.s.barrier(), !dbg !51
  fence syncscope("workgroup") acquire, !dbg !51
  %229 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !52
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %229, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !52
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !52
  %232 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !52
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %231, <4 x float> %230) #9, !dbg !52
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %232, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !52
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !52
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !52
  %236 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !52
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %235, <4 x float> %234) #9, !dbg !52
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %236, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !52
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !52
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !52
  %240 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !52
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %239, <4 x float> %238) #9, !dbg !52
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %240, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !52
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !52
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !52
  %244 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !52
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %243, <4 x float> %242) #9, !dbg !52
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %244, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !52
  %247 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !52
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !52
  %248 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !52
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %247, <4 x float> %246) #9, !dbg !52
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %248, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !52
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !52
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !52
  %252 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !52
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %251, <4 x float> %250) #9, !dbg !52
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %252, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !52
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !52
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !52
  %256 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !52
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %255, <4 x float> %254) #9, !dbg !52
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %256, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !52
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !52
  %260 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !52
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %259, <4 x float> %258) #9, !dbg !52
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %260, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !52
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !52
  %264 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !52
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %263, <4 x float> %262) #9, !dbg !52
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %264, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !52
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !52
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !52
  %268 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !52
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %267, <4 x float> %266) #9, !dbg !52
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %268, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !52
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !52
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !52
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !52
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %271, <4 x float> %270) #9, !dbg !52
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %272, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !52
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !52
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !52
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !52
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %275, <4 x float> %274) #9, !dbg !52
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %276, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !52
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !52
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !52
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !52
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %279, <4 x float> %278) #9, !dbg !52
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %280, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !52
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !52
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !52
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !52
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %283, <4 x float> %282) #9, !dbg !52
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %284, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !52
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !52
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !52
  %288 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !52
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %287, <4 x float> %286) #9, !dbg !52
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %288, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !52
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !52
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %291, <4 x float> %290) #9, !dbg !52
  %293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !44
  %294 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !44
  %295 = or disjoint i64 %33, 64, !dbg !44
  %296 = or disjoint i64 %address.local267, %295, !dbg !44
  %.tr22 = trunc i64 %296 to i32, !dbg !44
  %dma.buffer.offset1797 = shl i32 %.tr22, 1, !dbg !44
  %297 = insertelement <2 x i64> poison, i64 %294, i64 0, !dbg !44
  %298 = bitcast <2 x i64> %297 to <4 x i32>, !dbg !44
  %dma.buffer.rsrc1798 = shufflevector <4 x i32> %298, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !44
  %dma.buffer.lds1799 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %293), !dbg !44
  %299 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1797, <4 x i32> %dma.buffer.rsrc1798) #10, !dbg !44
  %300 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !44
  %301 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !44
  %302 = or disjoint i64 %33, 64, !dbg !44
  %303 = or disjoint i64 %address.local279, %302, !dbg !44
  %.tr23 = trunc i64 %303 to i32, !dbg !44
  %dma.buffer.offset1794 = shl i32 %.tr23, 1, !dbg !44
  %304 = insertelement <2 x i64> poison, i64 %301, i64 0, !dbg !44
  %305 = bitcast <2 x i64> %304 to <4 x i32>, !dbg !44
  %dma.buffer.rsrc1795 = shufflevector <4 x i32> %305, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !44
  %dma.buffer.lds1796 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %300), !dbg !44
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1796, i32 %dma.buffer.offset1794, <4 x i32> %dma.buffer.rsrc1795) #10, !dbg !44
  %307 = add nuw i64 %54, 2048, !dbg !51
  %308 = icmp slt i64 %307, 0, !dbg !51
  %309 = sub nuw i64 -2049, %54, !dbg !51
  %310 = select i1 %308, i64 %309, i64 %307, !dbg !51
  %311 = sdiv i64 %310, 4096, !dbg !51
  %.lobit2 = ashr i64 %307, 63, !dbg !51
  %312 = xor i64 %311, %.lobit2, !dbg !51
  %313 = add nsw i64 %312, %15, !dbg !51
  %314 = add nsw i64 %191, 64, !dbg !51
  %315 = icmp slt i64 %191, -64, !dbg !51
  %316 = sub nsw i64 -65, %191, !dbg !51
  %317 = select i1 %315, i64 %316, i64 %314, !dbg !51
  %318 = sdiv i64 %317, 128, !dbg !51
  %319 = sext i1 %315 to i64, !dbg !51
  %320 = xor i64 %318, %319, !dbg !51
  %321 = shl nsw i64 %320, 7, !dbg !51
  %322 = sub i64 %50, %321, !dbg !51
  %323 = shl i64 %313, 19, !dbg !51
  %324 = and i64 %323, 16252928, !dbg !51
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !44
  fence syncscope("workgroup") release, !dbg !44
  call void @llvm.amdgcn.s.barrier(), !dbg !44
  fence syncscope("workgroup") acquire, !dbg !44
  %325 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !51
  %326 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !51
  %.idx27 = add nuw nsw i64 %324, %76, !dbg !51
  %327 = add i64 %.idx27, %322, !dbg !51
  %328 = add i64 %327, %82, !dbg !51
  %329 = add i64 %328, %address.local293, !dbg !51
  %.tr28 = trunc i64 %329 to i32, !dbg !51
  %330 = shl i32 %.tr28, 1, !dbg !51
  %dma.buffer.offset1791 = add i32 %330, 128, !dbg !51
  %331 = insertelement <2 x i64> poison, i64 %326, i64 0, !dbg !51
  %332 = bitcast <2 x i64> %331 to <4 x i32>, !dbg !51
  %dma.buffer.rsrc1792 = shufflevector <4 x i32> %332, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !51
  %dma.buffer.lds1793 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %325), !dbg !51
  %333 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1793, i32 %dma.buffer.offset1791, <4 x i32> %dma.buffer.rsrc1792) #10, !dbg !51
  %334 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !51
  %335 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !51
  %.idx32 = add nuw nsw i64 %324, %76, !dbg !51
  %336 = add i64 %.idx32, %322, !dbg !51
  %337 = add i64 %336, %82, !dbg !51
  %338 = add i64 %337, %address.local307, !dbg !51
  %.tr33 = trunc i64 %338 to i32, !dbg !51
  %339 = shl i32 %.tr33, 1, !dbg !51
  %dma.buffer.offset1788 = add i32 %339, 128, !dbg !51
  %340 = insertelement <2 x i64> poison, i64 %335, i64 0, !dbg !51
  %341 = bitcast <2 x i64> %340 to <4 x i32>, !dbg !51
  %dma.buffer.rsrc1789 = shufflevector <4 x i32> %341, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !51
  %dma.buffer.lds1790 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %334), !dbg !51
  %342 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1790, i32 %dma.buffer.offset1788, <4 x i32> %dma.buffer.rsrc1789) #10, !dbg !51
  %343 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !45
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !45
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !45
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !45
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !45
  %346 = fmul <4 x half> %343, splat (half 1.275630e-01), !dbg !45
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !45
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !45
  %348 = fmul <4 x half> %344, splat (half 1.275630e-01), !dbg !45
  %349 = fmul <4 x half> %345, splat (half 1.275630e-01), !dbg !45
  %350 = fmul <4 x half> %347, splat (half 1.275630e-01), !dbg !45
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !51
  fence syncscope("workgroup") release, !dbg !51
  call void @llvm.amdgcn.s.barrier(), !dbg !51
  fence syncscope("workgroup") acquire, !dbg !51
  %351 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !52
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %351, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !52
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !52
  %354 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !52
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %353, <4 x float> %352) #9, !dbg !52
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %354, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !52
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !52
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !52
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !52
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %357, <4 x float> %356) #9, !dbg !52
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %358, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !52
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !52
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !52
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !52
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %361, <4 x float> %360) #9, !dbg !52
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %362, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !52
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !52
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !52
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !52
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %365, <4 x float> %364) #9, !dbg !52
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %366, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !52
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !52
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !52
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !52
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %369, <4 x float> %368) #9, !dbg !52
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %370, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !52
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !52
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !52
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !52
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %373, <4 x float> %372) #9, !dbg !52
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %374, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !52
  %377 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !52
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !52
  %378 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !52
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %377, <4 x float> %376) #9, !dbg !52
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %378, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !52
  %381 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !52
  %382 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !52
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %381, <4 x float> %380) #9, !dbg !52
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %382, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !52
  %385 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !52
  %386 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !52
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %385, <4 x float> %384) #9, !dbg !52
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %386, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !52
  %389 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !52
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !52
  %390 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !52
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %389, <4 x float> %388) #9, !dbg !52
  %392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %390, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !52
  %393 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !52
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !52
  %394 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !52
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %393, <4 x float> %392) #9, !dbg !52
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %394, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !52
  %397 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !52
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !52
  %398 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !52
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %397, <4 x float> %396) #9, !dbg !52
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %398, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !52
  %401 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !52
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !52
  %402 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !52
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %401, <4 x float> %400) #9, !dbg !52
  %404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %402, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !52
  %405 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !52
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !52
  %406 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !52
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %405, <4 x float> %404) #9, !dbg !52
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %406, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !52
  %409 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !52
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !52
  %410 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !52
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %409, <4 x float> %408) #9, !dbg !52
  %412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %410, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !52
  %413 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !52
  %414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %413, <4 x float> %412) #9, !dbg !52
  %415 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !44
  %416 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !44
  %417 = or disjoint i64 %33, 96, !dbg !44
  %418 = or disjoint i64 %address.local267, %417, !dbg !44
  %.tr34 = trunc i64 %418 to i32, !dbg !44
  %dma.buffer.offset1785 = shl i32 %.tr34, 1, !dbg !44
  %419 = insertelement <2 x i64> poison, i64 %416, i64 0, !dbg !44
  %420 = bitcast <2 x i64> %419 to <4 x i32>, !dbg !44
  %dma.buffer.rsrc1786 = shufflevector <4 x i32> %420, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !44
  %dma.buffer.lds1787 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %415), !dbg !44
  %421 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1787, i32 %dma.buffer.offset1785, <4 x i32> %dma.buffer.rsrc1786) #10, !dbg !44
  %422 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !44
  %423 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !44
  %424 = or disjoint i64 %33, 96, !dbg !44
  %425 = or disjoint i64 %address.local279, %424, !dbg !44
  %.tr35 = trunc i64 %425 to i32, !dbg !44
  %dma.buffer.offset1782 = shl i32 %.tr35, 1, !dbg !44
  %426 = insertelement <2 x i64> poison, i64 %423, i64 0, !dbg !44
  %427 = bitcast <2 x i64> %426 to <4 x i32>, !dbg !44
  %dma.buffer.rsrc1783 = shufflevector <4 x i32> %427, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !44
  %dma.buffer.lds1784 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %422), !dbg !44
  %428 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1784, i32 %dma.buffer.offset1782, <4 x i32> %dma.buffer.rsrc1783) #10, !dbg !44
  %429 = add nuw i64 %54, 3072, !dbg !51
  %430 = icmp slt i64 %429, 0, !dbg !51
  %431 = sub nuw i64 -3073, %54, !dbg !51
  %432 = select i1 %430, i64 %431, i64 %429, !dbg !51
  %433 = sdiv i64 %432, 4096, !dbg !51
  %.lobit3 = ashr i64 %429, 63, !dbg !51
  %434 = xor i64 %433, %.lobit3, !dbg !51
  %435 = add nsw i64 %434, %15, !dbg !51
  %436 = add nsw i64 %191, 96, !dbg !51
  %437 = icmp slt i64 %191, -96, !dbg !51
  %438 = sub nsw i64 -97, %191, !dbg !51
  %439 = select i1 %437, i64 %438, i64 %436, !dbg !51
  %440 = sdiv i64 %439, 128, !dbg !51
  %441 = sext i1 %437 to i64, !dbg !51
  %442 = xor i64 %440, %441, !dbg !51
  %443 = shl nsw i64 %442, 7, !dbg !51
  %444 = sub i64 %50, %443, !dbg !51
  %445 = shl i64 %435, 19, !dbg !51
  %446 = and i64 %445, 16252928, !dbg !51
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !44
  fence syncscope("workgroup") release, !dbg !44
  call void @llvm.amdgcn.s.barrier(), !dbg !44
  fence syncscope("workgroup") acquire, !dbg !44
  %447 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !51
  %448 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !51
  %.idx39 = add nuw nsw i64 %446, %76, !dbg !51
  %449 = add i64 %.idx39, %444, !dbg !51
  %450 = add i64 %449, %82, !dbg !51
  %451 = add i64 %450, %address.local293, !dbg !51
  %.tr40 = trunc i64 %451 to i32, !dbg !51
  %452 = shl i32 %.tr40, 1, !dbg !51
  %dma.buffer.offset1779 = add i32 %452, 192, !dbg !51
  %453 = insertelement <2 x i64> poison, i64 %448, i64 0, !dbg !51
  %454 = bitcast <2 x i64> %453 to <4 x i32>, !dbg !51
  %dma.buffer.rsrc1780 = shufflevector <4 x i32> %454, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !51
  %dma.buffer.lds1781 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %447), !dbg !51
  %455 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1781, i32 %dma.buffer.offset1779, <4 x i32> %dma.buffer.rsrc1780) #10, !dbg !51
  %456 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !51
  %457 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !51
  %.idx44 = add nuw nsw i64 %446, %76, !dbg !51
  %458 = add i64 %.idx44, %444, !dbg !51
  %459 = add i64 %458, %82, !dbg !51
  %460 = add i64 %459, %address.local307, !dbg !51
  %.tr45 = trunc i64 %460 to i32, !dbg !51
  %461 = shl i32 %.tr45, 1, !dbg !51
  %dma.buffer.offset1776 = add i32 %461, 192, !dbg !51
  %462 = insertelement <2 x i64> poison, i64 %457, i64 0, !dbg !51
  %463 = bitcast <2 x i64> %462 to <4 x i32>, !dbg !51
  %dma.buffer.rsrc1777 = shufflevector <4 x i32> %463, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !51
  %dma.buffer.lds1778 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %456), !dbg !51
  %464 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1778, i32 %dma.buffer.offset1776, <4 x i32> %dma.buffer.rsrc1777) #10, !dbg !51
  %465 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !45
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !45
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !45
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !45
  %467 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !45
  %468 = fmul <4 x half> %465, splat (half 1.275630e-01), !dbg !45
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !45
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !45
  %470 = fmul <4 x half> %466, splat (half 1.275630e-01), !dbg !45
  %471 = fmul <4 x half> %467, splat (half 1.275630e-01), !dbg !45
  %472 = fmul <4 x half> %469, splat (half 1.275630e-01), !dbg !45
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !51
  fence syncscope("workgroup") release, !dbg !51
  call void @llvm.amdgcn.s.barrier(), !dbg !51
  fence syncscope("workgroup") acquire, !dbg !51
  %473 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !52
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %473, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !52
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !52
  %476 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !52
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %475, <4 x float> %474) #9, !dbg !52
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %476, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !52
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !52
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !52
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !52
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %479, <4 x float> %478) #9, !dbg !52
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %480, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !52
  %483 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !52
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !52
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !52
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %483, <4 x float> %482) #9, !dbg !52
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %484, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !52
  %487 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !52
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !52
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !52
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %487, <4 x float> %486) #9, !dbg !52
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %488, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !52
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !52
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !52
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !52
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %491, <4 x float> %490) #9, !dbg !52
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %492, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !52
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !52
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !52
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !52
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %495, <4 x float> %494) #9, !dbg !52
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %496, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !52
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !52
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !52
  %500 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !52
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %499, <4 x float> %498) #9, !dbg !52
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %500, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !52
  %503 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !52
  %504 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !52
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %503, <4 x float> %502) #9, !dbg !52
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %504, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !52
  %507 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !52
  %508 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !52
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %507, <4 x float> %506) #9, !dbg !52
  %510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %508, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !52
  %511 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !52
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !52
  %512 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !52
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %511, <4 x float> %510) #9, !dbg !52
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %512, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !52
  %515 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !52
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !52
  %516 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !52
  %517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %515, <4 x float> %514) #9, !dbg !52
  %518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %516, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !52
  %519 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !52
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !52
  %520 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !52
  %521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %519, <4 x float> %518) #9, !dbg !52
  %522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %520, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !52
  %523 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !52
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !52
  %524 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !52
  %525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %523, <4 x float> %522) #9, !dbg !52
  %526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %524, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !52
  %527 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !52
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !52
  %528 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !52
  %529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %527, <4 x float> %526) #9, !dbg !52
  %530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %528, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !52
  %531 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !52
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !52
  %532 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !52
  %533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %531, <4 x float> %530) #9, !dbg !52
  %534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %532, <4 x float> zeroinitializer) #9, !dbg !52
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !52
  %535 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !52
  %536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %535, <4 x float> %534) #9, !dbg !52
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  br i1 true, label %537, label %667, !dbg !46

537:                                              ; preds = %62
  %538 = fadd <4 x float> %169, zeroinitializer, !dbg !57
  %539 = fadd <4 x float> %538, %292, !dbg !57
  %540 = fadd <4 x float> %539, %414, !dbg !57
  %541 = fadd <4 x float> %540, %536, !dbg !57
  %542 = extractelement <4 x float> %541, i64 3, !dbg !57
  %543 = extractelement <4 x float> %541, i64 2, !dbg !57
  %544 = extractelement <4 x float> %541, i64 1, !dbg !57
  %545 = extractelement <4 x float> %541, i64 0, !dbg !57
  %546 = fadd <4 x float> %166, zeroinitializer, !dbg !57
  %547 = fadd <4 x float> %546, %289, !dbg !57
  %548 = fadd <4 x float> %547, %411, !dbg !57
  %549 = fadd <4 x float> %548, %533, !dbg !57
  %550 = extractelement <4 x float> %549, i64 3, !dbg !57
  %551 = extractelement <4 x float> %549, i64 2, !dbg !57
  %552 = extractelement <4 x float> %549, i64 1, !dbg !57
  %553 = extractelement <4 x float> %549, i64 0, !dbg !57
  %554 = fadd <4 x float> %162, zeroinitializer, !dbg !57
  %555 = fadd <4 x float> %554, %285, !dbg !57
  %556 = fadd <4 x float> %555, %407, !dbg !57
  %557 = fadd <4 x float> %556, %529, !dbg !57
  %558 = extractelement <4 x float> %557, i64 3, !dbg !57
  %559 = extractelement <4 x float> %557, i64 2, !dbg !57
  %560 = extractelement <4 x float> %557, i64 1, !dbg !57
  %561 = extractelement <4 x float> %557, i64 0, !dbg !57
  %562 = fadd <4 x float> %158, zeroinitializer, !dbg !57
  %563 = fadd <4 x float> %562, %281, !dbg !57
  %564 = fadd <4 x float> %563, %403, !dbg !57
  %565 = fadd <4 x float> %564, %525, !dbg !57
  %566 = extractelement <4 x float> %565, i64 3, !dbg !57
  %567 = extractelement <4 x float> %565, i64 2, !dbg !57
  %568 = extractelement <4 x float> %565, i64 1, !dbg !57
  %569 = extractelement <4 x float> %565, i64 0, !dbg !57
  %570 = fadd <4 x float> %154, zeroinitializer, !dbg !57
  %571 = fadd <4 x float> %570, %277, !dbg !57
  %572 = fadd <4 x float> %571, %399, !dbg !57
  %573 = fadd <4 x float> %572, %521, !dbg !57
  %574 = extractelement <4 x float> %573, i64 3, !dbg !57
  %575 = extractelement <4 x float> %573, i64 2, !dbg !57
  %576 = extractelement <4 x float> %573, i64 1, !dbg !57
  %577 = extractelement <4 x float> %573, i64 0, !dbg !57
  %578 = fadd <4 x float> %150, zeroinitializer, !dbg !57
  %579 = fadd <4 x float> %578, %273, !dbg !57
  %580 = fadd <4 x float> %579, %395, !dbg !57
  %581 = fadd <4 x float> %580, %517, !dbg !57
  %582 = extractelement <4 x float> %581, i64 3, !dbg !57
  %583 = extractelement <4 x float> %581, i64 2, !dbg !57
  %584 = extractelement <4 x float> %581, i64 1, !dbg !57
  %585 = extractelement <4 x float> %581, i64 0, !dbg !57
  %586 = fadd <4 x float> %146, zeroinitializer, !dbg !57
  %587 = fadd <4 x float> %586, %269, !dbg !57
  %588 = fadd <4 x float> %587, %391, !dbg !57
  %589 = fadd <4 x float> %588, %513, !dbg !57
  %590 = extractelement <4 x float> %589, i64 3, !dbg !57
  %591 = extractelement <4 x float> %589, i64 2, !dbg !57
  %592 = extractelement <4 x float> %589, i64 1, !dbg !57
  %593 = extractelement <4 x float> %589, i64 0, !dbg !57
  %594 = fadd <4 x float> %142, zeroinitializer, !dbg !57
  %595 = fadd <4 x float> %594, %265, !dbg !57
  %596 = fadd <4 x float> %595, %387, !dbg !57
  %597 = fadd <4 x float> %596, %509, !dbg !57
  %598 = extractelement <4 x float> %597, i64 3, !dbg !57
  %599 = extractelement <4 x float> %597, i64 2, !dbg !57
  %600 = extractelement <4 x float> %597, i64 1, !dbg !57
  %601 = extractelement <4 x float> %597, i64 0, !dbg !57
  %602 = fadd <4 x float> %138, zeroinitializer, !dbg !57
  %603 = fadd <4 x float> %602, %261, !dbg !57
  %604 = fadd <4 x float> %603, %383, !dbg !57
  %605 = fadd <4 x float> %604, %505, !dbg !57
  %606 = extractelement <4 x float> %605, i64 3, !dbg !57
  %607 = extractelement <4 x float> %605, i64 2, !dbg !57
  %608 = extractelement <4 x float> %605, i64 1, !dbg !57
  %609 = extractelement <4 x float> %605, i64 0, !dbg !57
  %610 = fadd <4 x float> %134, zeroinitializer, !dbg !57
  %611 = fadd <4 x float> %610, %257, !dbg !57
  %612 = fadd <4 x float> %611, %379, !dbg !57
  %613 = fadd <4 x float> %612, %501, !dbg !57
  %614 = extractelement <4 x float> %613, i64 3, !dbg !57
  %615 = extractelement <4 x float> %613, i64 2, !dbg !57
  %616 = extractelement <4 x float> %613, i64 1, !dbg !57
  %617 = extractelement <4 x float> %613, i64 0, !dbg !57
  %618 = fadd <4 x float> %130, zeroinitializer, !dbg !57
  %619 = fadd <4 x float> %618, %253, !dbg !57
  %620 = fadd <4 x float> %619, %375, !dbg !57
  %621 = fadd <4 x float> %620, %497, !dbg !57
  %622 = extractelement <4 x float> %621, i64 3, !dbg !57
  %623 = extractelement <4 x float> %621, i64 2, !dbg !57
  %624 = extractelement <4 x float> %621, i64 1, !dbg !57
  %625 = extractelement <4 x float> %621, i64 0, !dbg !57
  %626 = fadd <4 x float> %126, zeroinitializer, !dbg !57
  %627 = fadd <4 x float> %626, %249, !dbg !57
  %628 = fadd <4 x float> %627, %371, !dbg !57
  %629 = fadd <4 x float> %628, %493, !dbg !57
  %630 = extractelement <4 x float> %629, i64 3, !dbg !57
  %631 = extractelement <4 x float> %629, i64 2, !dbg !57
  %632 = extractelement <4 x float> %629, i64 1, !dbg !57
  %633 = extractelement <4 x float> %629, i64 0, !dbg !57
  %634 = fadd <4 x float> %122, zeroinitializer, !dbg !57
  %635 = fadd <4 x float> %634, %245, !dbg !57
  %636 = fadd <4 x float> %635, %367, !dbg !57
  %637 = fadd <4 x float> %636, %489, !dbg !57
  %638 = extractelement <4 x float> %637, i64 3, !dbg !57
  %639 = extractelement <4 x float> %637, i64 2, !dbg !57
  %640 = extractelement <4 x float> %637, i64 1, !dbg !57
  %641 = extractelement <4 x float> %637, i64 0, !dbg !57
  %642 = fadd <4 x float> %118, zeroinitializer, !dbg !57
  %643 = fadd <4 x float> %642, %241, !dbg !57
  %644 = fadd <4 x float> %643, %363, !dbg !57
  %645 = fadd <4 x float> %644, %485, !dbg !57
  %646 = extractelement <4 x float> %645, i64 3, !dbg !57
  %647 = extractelement <4 x float> %645, i64 2, !dbg !57
  %648 = extractelement <4 x float> %645, i64 1, !dbg !57
  %649 = extractelement <4 x float> %645, i64 0, !dbg !57
  %650 = fadd <4 x float> %114, zeroinitializer, !dbg !57
  %651 = fadd <4 x float> %650, %237, !dbg !57
  %652 = fadd <4 x float> %651, %359, !dbg !57
  %653 = fadd <4 x float> %652, %481, !dbg !57
  %654 = extractelement <4 x float> %653, i64 3, !dbg !57
  %655 = extractelement <4 x float> %653, i64 2, !dbg !57
  %656 = extractelement <4 x float> %653, i64 1, !dbg !57
  %657 = extractelement <4 x float> %653, i64 0, !dbg !57
  %658 = fadd <4 x float> %110, zeroinitializer, !dbg !57
  %659 = fadd <4 x float> %658, %233, !dbg !57
  %660 = fadd <4 x float> %659, %355, !dbg !57
  %661 = fadd <4 x float> %660, %477, !dbg !57
  %662 = extractelement <4 x float> %661, i64 3, !dbg !57
  %663 = extractelement <4 x float> %661, i64 2, !dbg !57
  %664 = extractelement <4 x float> %661, i64 1, !dbg !57
  %665 = extractelement <4 x float> %661, i64 0, !dbg !57
  %address.thread413 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !46, !range !73, !frisk.register_lifetimes !37
  %address.local414 = shl nuw nsw i32 %address.thread413, 5, !dbg !46
  %address.local415 = and i32 %address.local414, 6144, !dbg !46
  %address.local416 = shl nuw nsw i32 %address.thread413, 7, !dbg !46
  %address.local417 = and i32 %address.local416, 1920, !dbg !46
  %address.local418 = or disjoint i32 %address.local415, %address.local417, !dbg !46
  %address.local419 = zext nneg i32 %address.local418 to i64, !dbg !46
  %address.local420 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %666 = lshr i32 %address.thread413, 4, !dbg !45
  %address.local422 = zext nneg i32 %666 to i64, !dbg !45
  %address.local423 = and i64 %address.local422, 3, !dbg !45
  %address.local424 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local423, !dbg !46
  store float %665, ptr addrspace(3) %address.local424, align 4, !dbg !46
  %address.local425 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local426 = or disjoint i64 %address.local423, 4, !dbg !46
  %address.local427 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local425, i64 %address.local426, !dbg !46
  store float %664, ptr addrspace(3) %address.local427, align 4, !dbg !46
  %address.local428 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local429 = or disjoint i64 %address.local423, 8, !dbg !46
  %address.local430 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local428, i64 %address.local429, !dbg !46
  store float %663, ptr addrspace(3) %address.local430, align 4, !dbg !46
  %address.local431 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local432 = or i64 %address.local422, 12, !dbg !46
  %address.local433 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local431, i64 %address.local432, !dbg !46
  store float %662, ptr addrspace(3) %address.local433, align 4, !dbg !46
  %address.local434 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local435 = or disjoint i64 %address.local423, 16, !dbg !46
  %address.local436 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local434, i64 %address.local435, !dbg !46
  store float %657, ptr addrspace(3) %address.local436, align 4, !dbg !46
  %address.local437 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local438 = or disjoint i64 %address.local423, 20, !dbg !46
  %address.local439 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local437, i64 %address.local438, !dbg !46
  store float %656, ptr addrspace(3) %address.local439, align 4, !dbg !46
  %address.local440 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local441 = or disjoint i64 %address.local423, 24, !dbg !46
  %address.local442 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local440, i64 %address.local441, !dbg !46
  store float %655, ptr addrspace(3) %address.local442, align 4, !dbg !46
  %address.local443 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local444 = or i64 %address.local422, 28, !dbg !46
  %address.local445 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local443, i64 %address.local444, !dbg !46
  store float %654, ptr addrspace(3) %address.local445, align 4, !dbg !46
  %address.local446 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local447 = or disjoint i64 %address.local423, 32, !dbg !46
  %address.local448 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local446, i64 %address.local447, !dbg !46
  store float %649, ptr addrspace(3) %address.local448, align 4, !dbg !46
  %address.local449 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local450 = or disjoint i64 %address.local423, 36, !dbg !46
  %address.local451 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local449, i64 %address.local450, !dbg !46
  store float %648, ptr addrspace(3) %address.local451, align 4, !dbg !46
  %address.local452 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local453 = or disjoint i64 %address.local423, 40, !dbg !46
  %address.local454 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local452, i64 %address.local453, !dbg !46
  store float %647, ptr addrspace(3) %address.local454, align 4, !dbg !46
  %address.local455 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local456 = or i64 %address.local422, 44, !dbg !46
  %address.local457 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local455, i64 %address.local456, !dbg !46
  store float %646, ptr addrspace(3) %address.local457, align 4, !dbg !46
  %address.local458 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local459 = or disjoint i64 %address.local423, 48, !dbg !46
  %address.local460 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local458, i64 %address.local459, !dbg !46
  store float %641, ptr addrspace(3) %address.local460, align 4, !dbg !46
  %address.local461 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local462 = or disjoint i64 %address.local423, 52, !dbg !46
  %address.local463 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local461, i64 %address.local462, !dbg !46
  store float %640, ptr addrspace(3) %address.local463, align 4, !dbg !46
  %address.local464 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local465 = or disjoint i64 %address.local423, 56, !dbg !46
  %address.local466 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local464, i64 %address.local465, !dbg !46
  store float %639, ptr addrspace(3) %address.local466, align 4, !dbg !46
  %address.local467 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local468 = or i64 %address.local422, 60, !dbg !46
  %address.local469 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local467, i64 %address.local468, !dbg !46
  store float %638, ptr addrspace(3) %address.local469, align 4, !dbg !46
  %address.local470 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local471 = or disjoint i64 %address.local423, 64, !dbg !46
  %address.local472 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local470, i64 %address.local471, !dbg !46
  store float %633, ptr addrspace(3) %address.local472, align 4, !dbg !46
  %address.local473 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local474 = or disjoint i64 %address.local423, 68, !dbg !46
  %address.local475 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local473, i64 %address.local474, !dbg !46
  store float %632, ptr addrspace(3) %address.local475, align 4, !dbg !46
  %address.local476 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local477 = or disjoint i64 %address.local423, 72, !dbg !46
  %address.local478 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local476, i64 %address.local477, !dbg !46
  store float %631, ptr addrspace(3) %address.local478, align 4, !dbg !46
  %address.local479 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local480 = or i64 %address.local422, 76, !dbg !46
  %address.local481 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local479, i64 %address.local480, !dbg !46
  store float %630, ptr addrspace(3) %address.local481, align 4, !dbg !46
  %address.local482 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local483 = or disjoint i64 %address.local423, 80, !dbg !46
  %address.local484 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local482, i64 %address.local483, !dbg !46
  store float %625, ptr addrspace(3) %address.local484, align 4, !dbg !46
  %address.local485 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local486 = or disjoint i64 %address.local423, 84, !dbg !46
  %address.local487 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local485, i64 %address.local486, !dbg !46
  store float %624, ptr addrspace(3) %address.local487, align 4, !dbg !46
  %address.local488 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local489 = or disjoint i64 %address.local423, 88, !dbg !46
  %address.local490 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local488, i64 %address.local489, !dbg !46
  store float %623, ptr addrspace(3) %address.local490, align 4, !dbg !46
  %address.local491 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local492 = or i64 %address.local422, 92, !dbg !46
  %address.local493 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local491, i64 %address.local492, !dbg !46
  store float %622, ptr addrspace(3) %address.local493, align 4, !dbg !46
  %address.local494 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local495 = or disjoint i64 %address.local423, 96, !dbg !46
  %address.local496 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local494, i64 %address.local495, !dbg !46
  store float %617, ptr addrspace(3) %address.local496, align 4, !dbg !46
  %address.local497 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local498 = or disjoint i64 %address.local423, 100, !dbg !46
  %address.local499 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local497, i64 %address.local498, !dbg !46
  store float %616, ptr addrspace(3) %address.local499, align 4, !dbg !46
  %address.local500 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local501 = or disjoint i64 %address.local423, 104, !dbg !46
  %address.local502 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local500, i64 %address.local501, !dbg !46
  store float %615, ptr addrspace(3) %address.local502, align 4, !dbg !46
  %address.local503 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local504 = or i64 %address.local422, 108, !dbg !46
  %address.local505 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local503, i64 %address.local504, !dbg !46
  store float %614, ptr addrspace(3) %address.local505, align 4, !dbg !46
  %address.local506 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local507 = or disjoint i64 %address.local423, 112, !dbg !46
  %address.local508 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local506, i64 %address.local507, !dbg !46
  store float %609, ptr addrspace(3) %address.local508, align 4, !dbg !46
  %address.local509 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local510 = or disjoint i64 %address.local423, 116, !dbg !46
  %address.local511 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local509, i64 %address.local510, !dbg !46
  store float %608, ptr addrspace(3) %address.local511, align 4, !dbg !46
  %address.local512 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local513 = or disjoint i64 %address.local423, 120, !dbg !46
  %address.local514 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local512, i64 %address.local513, !dbg !46
  store float %607, ptr addrspace(3) %address.local514, align 4, !dbg !46
  %address.local515 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !46
  %address.local516 = or i64 %address.local422, 124, !dbg !46
  %address.local517 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local515, i64 %address.local516, !dbg !46
  store float %606, ptr addrspace(3) %address.local517, align 4, !dbg !46
  %address.local518 = shl nuw nsw i32 %address.thread413, 5, !dbg !46
  %address.local519 = and i32 %address.local518, 6144, !dbg !46
  %address.local520 = shl nuw nsw i32 %address.thread413, 7, !dbg !46
  %address.local521 = and i32 %address.local520, 1920, !dbg !46
  %address.local522 = or disjoint i32 %address.local519, %address.local521, !dbg !46
  %address.local523 = or disjoint i32 %address.local522, 8192, !dbg !46
  %address.local524 = zext nneg i32 %address.local523 to i64, !dbg !46
  %address.local525 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local526 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local423, !dbg !46
  store float %601, ptr addrspace(3) %address.local526, align 4, !dbg !46
  %address.local527 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local528 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local527, i64 %address.local426, !dbg !46
  store float %600, ptr addrspace(3) %address.local528, align 4, !dbg !46
  %address.local529 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local530 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local529, i64 %address.local429, !dbg !46
  store float %599, ptr addrspace(3) %address.local530, align 4, !dbg !46
  %address.local531 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local532 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local531, i64 %address.local432, !dbg !46
  store float %598, ptr addrspace(3) %address.local532, align 4, !dbg !46
  %address.local533 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local534 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local533, i64 %address.local435, !dbg !46
  store float %593, ptr addrspace(3) %address.local534, align 4, !dbg !46
  %address.local535 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local536 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local535, i64 %address.local438, !dbg !46
  store float %592, ptr addrspace(3) %address.local536, align 4, !dbg !46
  %address.local537 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local538 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local537, i64 %address.local441, !dbg !46
  store float %591, ptr addrspace(3) %address.local538, align 4, !dbg !46
  %address.local539 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local540 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local539, i64 %address.local444, !dbg !46
  store float %590, ptr addrspace(3) %address.local540, align 4, !dbg !46
  %address.local541 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local542 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local541, i64 %address.local447, !dbg !46
  store float %585, ptr addrspace(3) %address.local542, align 4, !dbg !46
  %address.local543 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local544 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local543, i64 %address.local450, !dbg !46
  store float %584, ptr addrspace(3) %address.local544, align 4, !dbg !46
  %address.local545 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local546 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local545, i64 %address.local453, !dbg !46
  store float %583, ptr addrspace(3) %address.local546, align 4, !dbg !46
  %address.local547 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local548 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local547, i64 %address.local456, !dbg !46
  store float %582, ptr addrspace(3) %address.local548, align 4, !dbg !46
  %address.local549 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local550 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local549, i64 %address.local459, !dbg !46
  store float %577, ptr addrspace(3) %address.local550, align 4, !dbg !46
  %address.local551 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local552 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local551, i64 %address.local462, !dbg !46
  store float %576, ptr addrspace(3) %address.local552, align 4, !dbg !46
  %address.local553 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local554 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local553, i64 %address.local465, !dbg !46
  store float %575, ptr addrspace(3) %address.local554, align 4, !dbg !46
  %address.local555 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local556 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local555, i64 %address.local468, !dbg !46
  store float %574, ptr addrspace(3) %address.local556, align 4, !dbg !46
  %address.local557 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local558 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local557, i64 %address.local471, !dbg !46
  store float %569, ptr addrspace(3) %address.local558, align 4, !dbg !46
  %address.local559 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local560 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local559, i64 %address.local474, !dbg !46
  store float %568, ptr addrspace(3) %address.local560, align 4, !dbg !46
  %address.local561 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local562 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local561, i64 %address.local477, !dbg !46
  store float %567, ptr addrspace(3) %address.local562, align 4, !dbg !46
  %address.local563 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local564 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local563, i64 %address.local480, !dbg !46
  store float %566, ptr addrspace(3) %address.local564, align 4, !dbg !46
  %address.local565 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local566 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local565, i64 %address.local483, !dbg !46
  store float %561, ptr addrspace(3) %address.local566, align 4, !dbg !46
  %address.local567 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local568 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local567, i64 %address.local486, !dbg !46
  store float %560, ptr addrspace(3) %address.local568, align 4, !dbg !46
  %address.local569 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local570 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local569, i64 %address.local489, !dbg !46
  store float %559, ptr addrspace(3) %address.local570, align 4, !dbg !46
  %address.local571 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local572 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local571, i64 %address.local492, !dbg !46
  store float %558, ptr addrspace(3) %address.local572, align 4, !dbg !46
  %address.local573 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local574 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local573, i64 %address.local495, !dbg !46
  store float %553, ptr addrspace(3) %address.local574, align 4, !dbg !46
  %address.local575 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local576 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local575, i64 %address.local498, !dbg !46
  store float %552, ptr addrspace(3) %address.local576, align 4, !dbg !46
  %address.local577 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local578 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local577, i64 %address.local501, !dbg !46
  store float %551, ptr addrspace(3) %address.local578, align 4, !dbg !46
  %address.local579 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local580 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local579, i64 %address.local504, !dbg !46
  store float %550, ptr addrspace(3) %address.local580, align 4, !dbg !46
  %address.local581 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local582 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local581, i64 %address.local507, !dbg !46
  store float %545, ptr addrspace(3) %address.local582, align 4, !dbg !46
  %address.local583 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local584 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local583, i64 %address.local510, !dbg !46
  store float %544, ptr addrspace(3) %address.local584, align 4, !dbg !46
  %address.local585 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local586 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local585, i64 %address.local513, !dbg !46
  store float %543, ptr addrspace(3) %address.local586, align 4, !dbg !46
  %address.local587 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !46
  %address.local588 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local587, i64 %address.local516, !dbg !46
  store float %542, ptr addrspace(3) %address.local588, align 4, !dbg !46
  br label %667, !dbg !46

667:                                              ; preds = %537, %62
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %address.thread589 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !46, !range !73, !frisk.register_lifetimes !37
  %address.local590 = shl nuw nsw i32 %address.thread589, 5, !dbg !46
  %address.local591 = and i32 %address.local590, 6144, !dbg !46
  %address.local592 = shl nuw nsw i32 %address.thread589, 7, !dbg !46
  %address.local593 = and i32 %address.local592, 1920, !dbg !46
  %address.local594 = or disjoint i32 %address.local591, %address.local593, !dbg !46
  %address.local595 = zext nneg i32 %address.local594 to i64, !dbg !46
  %address.local596 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local595, !dbg !46
  %668 = lshr i32 %address.thread589, 2, !dbg !45
  %669 = and i32 %668, 12, !dbg !45
  %address.local600 = zext nneg i32 %669 to i64, !dbg !45
  %address.local601 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local596, i64 %address.local600, !dbg !46
  %670 = load float, ptr addrspace(3) %address.local601, align 16, !dbg !46
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local601, i64 4, !dbg !46
  %671 = load float, ptr addrspace(3) %lds.const.gep1682, align 4, !dbg !46
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local601, i64 8, !dbg !46
  %672 = load float, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !46
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local601, i64 12, !dbg !46
  %673 = load float, ptr addrspace(3) %lds.const.gep1684, align 4, !dbg !46
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local601, i64 64, !dbg !46
  %674 = load float, ptr addrspace(3) %lds.const.gep1685, align 16, !dbg !46
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local601, i64 68, !dbg !46
  %675 = load float, ptr addrspace(3) %lds.const.gep1686, align 4, !dbg !46
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local601, i64 72, !dbg !46
  %676 = load float, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !46
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local601, i64 76, !dbg !46
  %677 = load float, ptr addrspace(3) %lds.const.gep1688, align 4, !dbg !46
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local601, i64 128, !dbg !46
  %678 = load float, ptr addrspace(3) %lds.const.gep1689, align 16, !dbg !46
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local601, i64 132, !dbg !46
  %679 = load float, ptr addrspace(3) %lds.const.gep1690, align 4, !dbg !46
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local601, i64 136, !dbg !46
  %680 = load float, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !46
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local601, i64 140, !dbg !46
  %681 = load float, ptr addrspace(3) %lds.const.gep1692, align 4, !dbg !46
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local601, i64 192, !dbg !46
  %682 = load float, ptr addrspace(3) %lds.const.gep1693, align 16, !dbg !46
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local601, i64 196, !dbg !46
  %683 = load float, ptr addrspace(3) %lds.const.gep1694, align 4, !dbg !46
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local601, i64 200, !dbg !46
  %684 = load float, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !46
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local601, i64 204, !dbg !46
  %685 = load float, ptr addrspace(3) %lds.const.gep1696, align 4, !dbg !46
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local601, i64 256, !dbg !46
  %686 = load float, ptr addrspace(3) %lds.const.gep1697, align 16, !dbg !46
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local601, i64 260, !dbg !46
  %687 = load float, ptr addrspace(3) %lds.const.gep1698, align 4, !dbg !46
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local601, i64 264, !dbg !46
  %688 = load float, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !46
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local601, i64 268, !dbg !46
  %689 = load float, ptr addrspace(3) %lds.const.gep1700, align 4, !dbg !46
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local601, i64 320, !dbg !46
  %690 = load float, ptr addrspace(3) %lds.const.gep1701, align 16, !dbg !46
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %address.local601, i64 324, !dbg !46
  %691 = load float, ptr addrspace(3) %lds.const.gep1702, align 4, !dbg !46
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %address.local601, i64 328, !dbg !46
  %692 = load float, ptr addrspace(3) %lds.const.gep1703, align 8, !dbg !46
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local601, i64 332, !dbg !46
  %693 = load float, ptr addrspace(3) %lds.const.gep1704, align 4, !dbg !46
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local601, i64 384, !dbg !46
  %694 = load float, ptr addrspace(3) %lds.const.gep1705, align 16, !dbg !46
  %lds.const.gep1706 = getelementptr i8, ptr addrspace(3) %address.local601, i64 388, !dbg !46
  %695 = load float, ptr addrspace(3) %lds.const.gep1706, align 4, !dbg !46
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %address.local601, i64 392, !dbg !46
  %696 = load float, ptr addrspace(3) %lds.const.gep1707, align 8, !dbg !46
  %lds.const.gep1708 = getelementptr i8, ptr addrspace(3) %address.local601, i64 396, !dbg !46
  %697 = load float, ptr addrspace(3) %lds.const.gep1708, align 4, !dbg !46
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local601, i64 448, !dbg !46
  %698 = load float, ptr addrspace(3) %lds.const.gep1709, align 16, !dbg !46
  %lds.const.gep1710 = getelementptr i8, ptr addrspace(3) %address.local601, i64 452, !dbg !46
  %699 = load float, ptr addrspace(3) %lds.const.gep1710, align 4, !dbg !46
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %address.local601, i64 456, !dbg !46
  %700 = load float, ptr addrspace(3) %lds.const.gep1711, align 8, !dbg !46
  %lds.const.gep1712 = getelementptr i8, ptr addrspace(3) %address.local601, i64 460, !dbg !46
  %701 = load float, ptr addrspace(3) %lds.const.gep1712, align 4, !dbg !46
  %lds.const.gep1713 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32768, !dbg !46
  %702 = load float, ptr addrspace(3) %lds.const.gep1713, align 16, !dbg !46
  %lds.const.gep1714 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32772, !dbg !46
  %703 = load float, ptr addrspace(3) %lds.const.gep1714, align 4, !dbg !46
  %lds.const.gep1715 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32776, !dbg !46
  %704 = load float, ptr addrspace(3) %lds.const.gep1715, align 8, !dbg !46
  %lds.const.gep1716 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32780, !dbg !46
  %705 = load float, ptr addrspace(3) %lds.const.gep1716, align 4, !dbg !46
  %lds.const.gep1717 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32832, !dbg !46
  %706 = load float, ptr addrspace(3) %lds.const.gep1717, align 16, !dbg !46
  %lds.const.gep1718 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32836, !dbg !46
  %707 = load float, ptr addrspace(3) %lds.const.gep1718, align 4, !dbg !46
  %lds.const.gep1719 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32840, !dbg !46
  %708 = load float, ptr addrspace(3) %lds.const.gep1719, align 8, !dbg !46
  %lds.const.gep1720 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32844, !dbg !46
  %709 = load float, ptr addrspace(3) %lds.const.gep1720, align 4, !dbg !46
  %lds.const.gep1721 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32896, !dbg !46
  %710 = load float, ptr addrspace(3) %lds.const.gep1721, align 16, !dbg !46
  %lds.const.gep1722 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32900, !dbg !46
  %711 = load float, ptr addrspace(3) %lds.const.gep1722, align 4, !dbg !46
  %lds.const.gep1723 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32904, !dbg !46
  %712 = load float, ptr addrspace(3) %lds.const.gep1723, align 8, !dbg !46
  %lds.const.gep1724 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32908, !dbg !46
  %713 = load float, ptr addrspace(3) %lds.const.gep1724, align 4, !dbg !46
  %lds.const.gep1725 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32960, !dbg !46
  %714 = load float, ptr addrspace(3) %lds.const.gep1725, align 16, !dbg !46
  %lds.const.gep1726 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32964, !dbg !46
  %715 = load float, ptr addrspace(3) %lds.const.gep1726, align 4, !dbg !46
  %lds.const.gep1727 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32968, !dbg !46
  %716 = load float, ptr addrspace(3) %lds.const.gep1727, align 8, !dbg !46
  %lds.const.gep1728 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32972, !dbg !46
  %717 = load float, ptr addrspace(3) %lds.const.gep1728, align 4, !dbg !46
  %lds.const.gep1729 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33024, !dbg !46
  %718 = load float, ptr addrspace(3) %lds.const.gep1729, align 16, !dbg !46
  %lds.const.gep1730 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33028, !dbg !46
  %719 = load float, ptr addrspace(3) %lds.const.gep1730, align 4, !dbg !46
  %lds.const.gep1731 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33032, !dbg !46
  %720 = load float, ptr addrspace(3) %lds.const.gep1731, align 8, !dbg !46
  %lds.const.gep1732 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33036, !dbg !46
  %721 = load float, ptr addrspace(3) %lds.const.gep1732, align 4, !dbg !46
  %lds.const.gep1733 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33088, !dbg !46
  %722 = load float, ptr addrspace(3) %lds.const.gep1733, align 16, !dbg !46
  %lds.const.gep1734 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33092, !dbg !46
  %723 = load float, ptr addrspace(3) %lds.const.gep1734, align 4, !dbg !46
  %lds.const.gep1735 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33096, !dbg !46
  %724 = load float, ptr addrspace(3) %lds.const.gep1735, align 8, !dbg !46
  %lds.const.gep1736 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33100, !dbg !46
  %725 = load float, ptr addrspace(3) %lds.const.gep1736, align 4, !dbg !46
  %lds.const.gep1737 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33152, !dbg !46
  %726 = load float, ptr addrspace(3) %lds.const.gep1737, align 16, !dbg !46
  %lds.const.gep1738 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33156, !dbg !46
  %727 = load float, ptr addrspace(3) %lds.const.gep1738, align 4, !dbg !46
  %lds.const.gep1739 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33160, !dbg !46
  %728 = load float, ptr addrspace(3) %lds.const.gep1739, align 8, !dbg !46
  %lds.const.gep1740 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33164, !dbg !46
  %729 = load float, ptr addrspace(3) %lds.const.gep1740, align 4, !dbg !46
  %lds.const.gep1741 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33216, !dbg !46
  %730 = load float, ptr addrspace(3) %lds.const.gep1741, align 16, !dbg !46
  %lds.const.gep1742 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33220, !dbg !46
  %731 = load float, ptr addrspace(3) %lds.const.gep1742, align 4, !dbg !46
  %lds.const.gep1743 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33224, !dbg !46
  %732 = load float, ptr addrspace(3) %lds.const.gep1743, align 8, !dbg !46
  %lds.const.gep1744 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33228, !dbg !46
  %733 = load float, ptr addrspace(3) %lds.const.gep1744, align 4, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %734 = or disjoint i64 %63, 127, !dbg !47
  %735 = icmp uge i64 %19, %734, !dbg !47
  %736 = and i1 %42, %735, !dbg !47
  br i1 %736, label %737, label %738, !dbg !47

737:                                              ; preds = %667
  br label %879, !dbg !47

738:                                              ; preds = %667
  %739 = add i64 %19, %35, !dbg !47
  %740 = add i64 %739, %52, !dbg !47
  %741 = sub i64 %740, %39, !dbg !47
  %742 = or disjoint i64 %63, %38, !dbg !47
  %743 = add i64 %741, 1, !dbg !58
  %744 = lshr i64 %743, 32, !dbg !59
  %compare.high = trunc nuw i64 %744 to i32, !dbg !59
  %745 = lshr i64 %63, 32, !dbg !59
  %compare.high1303 = trunc nuw i64 %745 to i32, !dbg !59
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1303, !dbg !59
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1303, !dbg !59
  %compare.low.base = trunc i64 %743 to i32, !dbg !59
  %compare.low.base1304 = trunc i64 %742 to i32, !dbg !59
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1304, !dbg !59
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !59
  %746 = select i1 %compare.result, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 1) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local = or i32 %compare.local.base, 1, !dbg !59
  %compare.low1305 = icmp ult i32 %compare.local, %compare.low.base, !dbg !59
  %compare.result1306 = select i1 %compare.high.equal, i1 %compare.low1305, i1 %compare.high.order, !dbg !59
  %747 = select i1 %compare.result1306, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1307 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 2) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1308 = or i32 %compare.local.base1307, 2, !dbg !59
  %compare.low1309 = icmp ult i32 %compare.local1308, %compare.low.base, !dbg !59
  %compare.result1310 = select i1 %compare.high.equal, i1 %compare.low1309, i1 %compare.high.order, !dbg !59
  %748 = select i1 %compare.result1310, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1311 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 3) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1312 = or i32 %compare.local.base1311, 3, !dbg !59
  %compare.low1313 = icmp ult i32 %compare.local1312, %compare.low.base, !dbg !59
  %compare.result1314 = select i1 %compare.high.equal, i1 %compare.low1313, i1 %compare.high.order, !dbg !59
  %749 = select i1 %compare.result1314, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1315 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 4) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1316 = or i32 %compare.local.base1315, 16, !dbg !59
  %compare.low1317 = icmp ult i32 %compare.local1316, %compare.low.base, !dbg !59
  %compare.result1318 = select i1 %compare.high.equal, i1 %compare.low1317, i1 %compare.high.order, !dbg !59
  %750 = select i1 %compare.result1318, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1319 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 5) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1320 = or i32 %compare.local.base1319, 17, !dbg !59
  %compare.low1321 = icmp ult i32 %compare.local1320, %compare.low.base, !dbg !59
  %compare.result1322 = select i1 %compare.high.equal, i1 %compare.low1321, i1 %compare.high.order, !dbg !59
  %751 = select i1 %compare.result1322, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1323 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 6) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1324 = or i32 %compare.local.base1323, 18, !dbg !59
  %compare.low1325 = icmp ult i32 %compare.local1324, %compare.low.base, !dbg !59
  %compare.result1326 = select i1 %compare.high.equal, i1 %compare.low1325, i1 %compare.high.order, !dbg !59
  %752 = select i1 %compare.result1326, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 7) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1328 = or i32 %compare.local.base1327, 19, !dbg !59
  %compare.low1329 = icmp ult i32 %compare.local1328, %compare.low.base, !dbg !59
  %compare.result1330 = select i1 %compare.high.equal, i1 %compare.low1329, i1 %compare.high.order, !dbg !59
  %753 = select i1 %compare.result1330, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 8) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1332 = or i32 %compare.local.base1331, 32, !dbg !59
  %compare.low1333 = icmp ult i32 %compare.local1332, %compare.low.base, !dbg !59
  %compare.result1334 = select i1 %compare.high.equal, i1 %compare.low1333, i1 %compare.high.order, !dbg !59
  %754 = select i1 %compare.result1334, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 9) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1336 = or i32 %compare.local.base1335, 33, !dbg !59
  %compare.low1337 = icmp ult i32 %compare.local1336, %compare.low.base, !dbg !59
  %compare.result1338 = select i1 %compare.high.equal, i1 %compare.low1337, i1 %compare.high.order, !dbg !59
  %755 = select i1 %compare.result1338, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 10) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1340 = or i32 %compare.local.base1339, 34, !dbg !59
  %compare.low1341 = icmp ult i32 %compare.local1340, %compare.low.base, !dbg !59
  %compare.result1342 = select i1 %compare.high.equal, i1 %compare.low1341, i1 %compare.high.order, !dbg !59
  %756 = select i1 %compare.result1342, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 11) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1344 = or i32 %compare.local.base1343, 35, !dbg !59
  %compare.low1345 = icmp ult i32 %compare.local1344, %compare.low.base, !dbg !59
  %compare.result1346 = select i1 %compare.high.equal, i1 %compare.low1345, i1 %compare.high.order, !dbg !59
  %757 = select i1 %compare.result1346, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 12) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1348 = or i32 %compare.local.base1347, 48, !dbg !59
  %compare.low1349 = icmp ult i32 %compare.local1348, %compare.low.base, !dbg !59
  %compare.result1350 = select i1 %compare.high.equal, i1 %compare.low1349, i1 %compare.high.order, !dbg !59
  %758 = select i1 %compare.result1350, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 13) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1352 = or i32 %compare.local.base1351, 49, !dbg !59
  %compare.low1353 = icmp ult i32 %compare.local1352, %compare.low.base, !dbg !59
  %compare.result1354 = select i1 %compare.high.equal, i1 %compare.low1353, i1 %compare.high.order, !dbg !59
  %759 = select i1 %compare.result1354, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 14) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1356 = or i32 %compare.local.base1355, 50, !dbg !59
  %compare.low1357 = icmp ult i32 %compare.local1356, %compare.low.base, !dbg !59
  %compare.result1358 = select i1 %compare.high.equal, i1 %compare.low1357, i1 %compare.high.order, !dbg !59
  %760 = select i1 %compare.result1358, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 15) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1360 = or i32 %compare.local.base1359, 51, !dbg !59
  %compare.low1361 = icmp ult i32 %compare.local1360, %compare.low.base, !dbg !59
  %compare.result1362 = select i1 %compare.high.equal, i1 %compare.low1361, i1 %compare.high.order, !dbg !59
  %761 = select i1 %compare.result1362, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 16) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1364 = or i32 %compare.local.base1363, 64, !dbg !59
  %compare.low1365 = icmp ult i32 %compare.local1364, %compare.low.base, !dbg !59
  %compare.result1366 = select i1 %compare.high.equal, i1 %compare.low1365, i1 %compare.high.order, !dbg !59
  %762 = select i1 %compare.result1366, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 17) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1368 = or i32 %compare.local.base1367, 65, !dbg !59
  %compare.low1369 = icmp ult i32 %compare.local1368, %compare.low.base, !dbg !59
  %compare.result1370 = select i1 %compare.high.equal, i1 %compare.low1369, i1 %compare.high.order, !dbg !59
  %763 = select i1 %compare.result1370, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 18) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1372 = or i32 %compare.local.base1371, 66, !dbg !59
  %compare.low1373 = icmp ult i32 %compare.local1372, %compare.low.base, !dbg !59
  %compare.result1374 = select i1 %compare.high.equal, i1 %compare.low1373, i1 %compare.high.order, !dbg !59
  %764 = select i1 %compare.result1374, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 19) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1376 = or i32 %compare.local.base1375, 67, !dbg !59
  %compare.low1377 = icmp ult i32 %compare.local1376, %compare.low.base, !dbg !59
  %compare.result1378 = select i1 %compare.high.equal, i1 %compare.low1377, i1 %compare.high.order, !dbg !59
  %765 = select i1 %compare.result1378, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 20) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1380 = or i32 %compare.local.base1379, 80, !dbg !59
  %compare.low1381 = icmp ult i32 %compare.local1380, %compare.low.base, !dbg !59
  %compare.result1382 = select i1 %compare.high.equal, i1 %compare.low1381, i1 %compare.high.order, !dbg !59
  %766 = select i1 %compare.result1382, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 21) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1384 = or i32 %compare.local.base1383, 81, !dbg !59
  %compare.low1385 = icmp ult i32 %compare.local1384, %compare.low.base, !dbg !59
  %compare.result1386 = select i1 %compare.high.equal, i1 %compare.low1385, i1 %compare.high.order, !dbg !59
  %767 = select i1 %compare.result1386, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 22) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1388 = or i32 %compare.local.base1387, 82, !dbg !59
  %compare.low1389 = icmp ult i32 %compare.local1388, %compare.low.base, !dbg !59
  %compare.result1390 = select i1 %compare.high.equal, i1 %compare.low1389, i1 %compare.high.order, !dbg !59
  %768 = select i1 %compare.result1390, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 23) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1392 = or i32 %compare.local.base1391, 83, !dbg !59
  %compare.low1393 = icmp ult i32 %compare.local1392, %compare.low.base, !dbg !59
  %compare.result1394 = select i1 %compare.high.equal, i1 %compare.low1393, i1 %compare.high.order, !dbg !59
  %769 = select i1 %compare.result1394, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 24) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1396 = or i32 %compare.local.base1395, 96, !dbg !59
  %compare.low1397 = icmp ult i32 %compare.local1396, %compare.low.base, !dbg !59
  %compare.result1398 = select i1 %compare.high.equal, i1 %compare.low1397, i1 %compare.high.order, !dbg !59
  %770 = select i1 %compare.result1398, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 25) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1400 = or i32 %compare.local.base1399, 97, !dbg !59
  %compare.low1401 = icmp ult i32 %compare.local1400, %compare.low.base, !dbg !59
  %compare.result1402 = select i1 %compare.high.equal, i1 %compare.low1401, i1 %compare.high.order, !dbg !59
  %771 = select i1 %compare.result1402, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 26) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1404 = or i32 %compare.local.base1403, 98, !dbg !59
  %compare.low1405 = icmp ult i32 %compare.local1404, %compare.low.base, !dbg !59
  %compare.result1406 = select i1 %compare.high.equal, i1 %compare.low1405, i1 %compare.high.order, !dbg !59
  %772 = select i1 %compare.result1406, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 27) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1408 = or i32 %compare.local.base1407, 99, !dbg !59
  %compare.low1409 = icmp ult i32 %compare.local1408, %compare.low.base, !dbg !59
  %compare.result1410 = select i1 %compare.high.equal, i1 %compare.low1409, i1 %compare.high.order, !dbg !59
  %773 = select i1 %compare.result1410, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 28) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1412 = or i32 %compare.local.base1411, 112, !dbg !59
  %compare.low1413 = icmp ult i32 %compare.local1412, %compare.low.base, !dbg !59
  %compare.result1414 = select i1 %compare.high.equal, i1 %compare.low1413, i1 %compare.high.order, !dbg !59
  %774 = select i1 %compare.result1414, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 29) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1416 = or i32 %compare.local.base1415, 113, !dbg !59
  %compare.low1417 = icmp ult i32 %compare.local1416, %compare.low.base, !dbg !59
  %compare.result1418 = select i1 %compare.high.equal, i1 %compare.low1417, i1 %compare.high.order, !dbg !59
  %775 = select i1 %compare.result1418, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 30) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1420 = or i32 %compare.local.base1419, 114, !dbg !59
  %compare.low1421 = icmp ult i32 %compare.local1420, %compare.low.base, !dbg !59
  %compare.result1422 = select i1 %compare.high.equal, i1 %compare.low1421, i1 %compare.high.order, !dbg !59
  %776 = select i1 %compare.result1422, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 31) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1424 = or i32 %compare.local.base1423, 115, !dbg !59
  %compare.low1425 = icmp ult i32 %compare.local1424, %compare.low.base, !dbg !59
  %compare.result1426 = select i1 %compare.high.equal, i1 %compare.low1425, i1 %compare.high.order, !dbg !59
  %777 = select i1 %compare.result1426, float 0.000000e+00, float -inf, !dbg !60
  %778 = add i64 %741, 65, !dbg !58
  %779 = lshr i64 %778, 32, !dbg !59
  %compare.high1427 = trunc nuw i64 %779 to i32, !dbg !59
  %780 = lshr i64 %63, 32, !dbg !59
  %compare.high1428 = trunc nuw i64 %780 to i32, !dbg !59
  %compare.high.equal1429 = icmp eq i32 %compare.high1427, %compare.high1428, !dbg !59
  %compare.high.order1430 = icmp ugt i32 %compare.high1427, %compare.high1428, !dbg !59
  %compare.low.base1431 = trunc i64 %778 to i32, !dbg !59
  %compare.low.base1432 = trunc i64 %742 to i32, !dbg !59
  %compare.low1433 = icmp ugt i32 %compare.low.base1431, %compare.low.base1432, !dbg !59
  %compare.result1434 = select i1 %compare.high.equal1429, i1 %compare.low1433, i1 %compare.high.order1430, !dbg !59
  %781 = select i1 %compare.result1434, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 32) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1436 = or i32 %compare.local.base1435, 1, !dbg !59
  %compare.low1437 = icmp ult i32 %compare.local1436, %compare.low.base1431, !dbg !59
  %compare.result1438 = select i1 %compare.high.equal1429, i1 %compare.low1437, i1 %compare.high.order1430, !dbg !59
  %782 = select i1 %compare.result1438, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 33) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1440 = or i32 %compare.local.base1439, 2, !dbg !59
  %compare.low1441 = icmp ult i32 %compare.local1440, %compare.low.base1431, !dbg !59
  %compare.result1442 = select i1 %compare.high.equal1429, i1 %compare.low1441, i1 %compare.high.order1430, !dbg !59
  %783 = select i1 %compare.result1442, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 34) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1444 = or i32 %compare.local.base1443, 3, !dbg !59
  %compare.low1445 = icmp ult i32 %compare.local1444, %compare.low.base1431, !dbg !59
  %compare.result1446 = select i1 %compare.high.equal1429, i1 %compare.low1445, i1 %compare.high.order1430, !dbg !59
  %784 = select i1 %compare.result1446, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1447 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 35) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1448 = or i32 %compare.local.base1447, 16, !dbg !59
  %compare.low1449 = icmp ult i32 %compare.local1448, %compare.low.base1431, !dbg !59
  %compare.result1450 = select i1 %compare.high.equal1429, i1 %compare.low1449, i1 %compare.high.order1430, !dbg !59
  %785 = select i1 %compare.result1450, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1451 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 36) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1452 = or i32 %compare.local.base1451, 17, !dbg !59
  %compare.low1453 = icmp ult i32 %compare.local1452, %compare.low.base1431, !dbg !59
  %compare.result1454 = select i1 %compare.high.equal1429, i1 %compare.low1453, i1 %compare.high.order1430, !dbg !59
  %786 = select i1 %compare.result1454, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1455 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 37) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1456 = or i32 %compare.local.base1455, 18, !dbg !59
  %compare.low1457 = icmp ult i32 %compare.local1456, %compare.low.base1431, !dbg !59
  %compare.result1458 = select i1 %compare.high.equal1429, i1 %compare.low1457, i1 %compare.high.order1430, !dbg !59
  %787 = select i1 %compare.result1458, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 38) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1460 = or i32 %compare.local.base1459, 19, !dbg !59
  %compare.low1461 = icmp ult i32 %compare.local1460, %compare.low.base1431, !dbg !59
  %compare.result1462 = select i1 %compare.high.equal1429, i1 %compare.low1461, i1 %compare.high.order1430, !dbg !59
  %788 = select i1 %compare.result1462, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 39) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1464 = or i32 %compare.local.base1463, 32, !dbg !59
  %compare.low1465 = icmp ult i32 %compare.local1464, %compare.low.base1431, !dbg !59
  %compare.result1466 = select i1 %compare.high.equal1429, i1 %compare.low1465, i1 %compare.high.order1430, !dbg !59
  %789 = select i1 %compare.result1466, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 40) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1468 = or i32 %compare.local.base1467, 33, !dbg !59
  %compare.low1469 = icmp ult i32 %compare.local1468, %compare.low.base1431, !dbg !59
  %compare.result1470 = select i1 %compare.high.equal1429, i1 %compare.low1469, i1 %compare.high.order1430, !dbg !59
  %790 = select i1 %compare.result1470, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 41) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1472 = or i32 %compare.local.base1471, 34, !dbg !59
  %compare.low1473 = icmp ult i32 %compare.local1472, %compare.low.base1431, !dbg !59
  %compare.result1474 = select i1 %compare.high.equal1429, i1 %compare.low1473, i1 %compare.high.order1430, !dbg !59
  %791 = select i1 %compare.result1474, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 42) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1476 = or i32 %compare.local.base1475, 35, !dbg !59
  %compare.low1477 = icmp ult i32 %compare.local1476, %compare.low.base1431, !dbg !59
  %compare.result1478 = select i1 %compare.high.equal1429, i1 %compare.low1477, i1 %compare.high.order1430, !dbg !59
  %792 = select i1 %compare.result1478, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 43) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1480 = or i32 %compare.local.base1479, 48, !dbg !59
  %compare.low1481 = icmp ult i32 %compare.local1480, %compare.low.base1431, !dbg !59
  %compare.result1482 = select i1 %compare.high.equal1429, i1 %compare.low1481, i1 %compare.high.order1430, !dbg !59
  %793 = select i1 %compare.result1482, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 44) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1484 = or i32 %compare.local.base1483, 49, !dbg !59
  %compare.low1485 = icmp ult i32 %compare.local1484, %compare.low.base1431, !dbg !59
  %compare.result1486 = select i1 %compare.high.equal1429, i1 %compare.low1485, i1 %compare.high.order1430, !dbg !59
  %794 = select i1 %compare.result1486, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1487 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 45) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1488 = or i32 %compare.local.base1487, 50, !dbg !59
  %compare.low1489 = icmp ult i32 %compare.local1488, %compare.low.base1431, !dbg !59
  %compare.result1490 = select i1 %compare.high.equal1429, i1 %compare.low1489, i1 %compare.high.order1430, !dbg !59
  %795 = select i1 %compare.result1490, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1491 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 46) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1492 = or i32 %compare.local.base1491, 51, !dbg !59
  %compare.low1493 = icmp ult i32 %compare.local1492, %compare.low.base1431, !dbg !59
  %compare.result1494 = select i1 %compare.high.equal1429, i1 %compare.low1493, i1 %compare.high.order1430, !dbg !59
  %796 = select i1 %compare.result1494, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1495 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 47) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1496 = or i32 %compare.local.base1495, 64, !dbg !59
  %compare.low1497 = icmp ult i32 %compare.local1496, %compare.low.base1431, !dbg !59
  %compare.result1498 = select i1 %compare.high.equal1429, i1 %compare.low1497, i1 %compare.high.order1430, !dbg !59
  %797 = select i1 %compare.result1498, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1499 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 48) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1500 = or i32 %compare.local.base1499, 65, !dbg !59
  %compare.low1501 = icmp ult i32 %compare.local1500, %compare.low.base1431, !dbg !59
  %compare.result1502 = select i1 %compare.high.equal1429, i1 %compare.low1501, i1 %compare.high.order1430, !dbg !59
  %798 = select i1 %compare.result1502, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1503 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 49) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1504 = or i32 %compare.local.base1503, 66, !dbg !59
  %compare.low1505 = icmp ult i32 %compare.local1504, %compare.low.base1431, !dbg !59
  %compare.result1506 = select i1 %compare.high.equal1429, i1 %compare.low1505, i1 %compare.high.order1430, !dbg !59
  %799 = select i1 %compare.result1506, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1507 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 50) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1508 = or i32 %compare.local.base1507, 67, !dbg !59
  %compare.low1509 = icmp ult i32 %compare.local1508, %compare.low.base1431, !dbg !59
  %compare.result1510 = select i1 %compare.high.equal1429, i1 %compare.low1509, i1 %compare.high.order1430, !dbg !59
  %800 = select i1 %compare.result1510, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1511 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 51) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1512 = or i32 %compare.local.base1511, 80, !dbg !59
  %compare.low1513 = icmp ult i32 %compare.local1512, %compare.low.base1431, !dbg !59
  %compare.result1514 = select i1 %compare.high.equal1429, i1 %compare.low1513, i1 %compare.high.order1430, !dbg !59
  %801 = select i1 %compare.result1514, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1515 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 52) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1516 = or i32 %compare.local.base1515, 81, !dbg !59
  %compare.low1517 = icmp ult i32 %compare.local1516, %compare.low.base1431, !dbg !59
  %compare.result1518 = select i1 %compare.high.equal1429, i1 %compare.low1517, i1 %compare.high.order1430, !dbg !59
  %802 = select i1 %compare.result1518, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1519 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 53) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1520 = or i32 %compare.local.base1519, 82, !dbg !59
  %compare.low1521 = icmp ult i32 %compare.local1520, %compare.low.base1431, !dbg !59
  %compare.result1522 = select i1 %compare.high.equal1429, i1 %compare.low1521, i1 %compare.high.order1430, !dbg !59
  %803 = select i1 %compare.result1522, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 54) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1524 = or i32 %compare.local.base1523, 83, !dbg !59
  %compare.low1525 = icmp ult i32 %compare.local1524, %compare.low.base1431, !dbg !59
  %compare.result1526 = select i1 %compare.high.equal1429, i1 %compare.low1525, i1 %compare.high.order1430, !dbg !59
  %804 = select i1 %compare.result1526, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 55) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1528 = or i32 %compare.local.base1527, 96, !dbg !59
  %compare.low1529 = icmp ult i32 %compare.local1528, %compare.low.base1431, !dbg !59
  %compare.result1530 = select i1 %compare.high.equal1429, i1 %compare.low1529, i1 %compare.high.order1430, !dbg !59
  %805 = select i1 %compare.result1530, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 56) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1532 = or i32 %compare.local.base1531, 97, !dbg !59
  %compare.low1533 = icmp ult i32 %compare.local1532, %compare.low.base1431, !dbg !59
  %compare.result1534 = select i1 %compare.high.equal1429, i1 %compare.low1533, i1 %compare.high.order1430, !dbg !59
  %806 = select i1 %compare.result1534, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 57) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1536 = or i32 %compare.local.base1535, 98, !dbg !59
  %compare.low1537 = icmp ult i32 %compare.local1536, %compare.low.base1431, !dbg !59
  %compare.result1538 = select i1 %compare.high.equal1429, i1 %compare.low1537, i1 %compare.high.order1430, !dbg !59
  %807 = select i1 %compare.result1538, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 58) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1540 = or i32 %compare.local.base1539, 99, !dbg !59
  %compare.low1541 = icmp ult i32 %compare.local1540, %compare.low.base1431, !dbg !59
  %compare.result1542 = select i1 %compare.high.equal1429, i1 %compare.low1541, i1 %compare.high.order1430, !dbg !59
  %808 = select i1 %compare.result1542, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 59) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1544 = or i32 %compare.local.base1543, 112, !dbg !59
  %compare.low1545 = icmp ult i32 %compare.local1544, %compare.low.base1431, !dbg !59
  %compare.result1546 = select i1 %compare.high.equal1429, i1 %compare.low1545, i1 %compare.high.order1430, !dbg !59
  %809 = select i1 %compare.result1546, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 60) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1548 = or i32 %compare.local.base1547, 113, !dbg !59
  %compare.low1549 = icmp ult i32 %compare.local1548, %compare.low.base1431, !dbg !59
  %compare.result1550 = select i1 %compare.high.equal1429, i1 %compare.low1549, i1 %compare.high.order1430, !dbg !59
  %810 = select i1 %compare.result1550, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 61) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1552 = or i32 %compare.local.base1551, 114, !dbg !59
  %compare.low1553 = icmp ult i32 %compare.local1552, %compare.low.base1431, !dbg !59
  %compare.result1554 = select i1 %compare.high.equal1429, i1 %compare.low1553, i1 %compare.high.order1430, !dbg !59
  %811 = select i1 %compare.result1554, float 0.000000e+00, float -inf, !dbg !60
  %compare.local.base1555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 62) #9, !dbg !59, !frisk.register_lifetimes !37
  %compare.local1556 = or i32 %compare.local.base1555, 115, !dbg !59
  %compare.low1557 = icmp ult i32 %compare.local1556, %compare.low.base1431, !dbg !59
  %compare.result1558 = select i1 %compare.high.equal1429, i1 %compare.low1557, i1 %compare.high.order1430, !dbg !59
  %812 = select i1 %compare.result1558, float 0.000000e+00, float -inf, !dbg !60
  %813 = insertelement <32 x float> poison, float %746, i64 0, !dbg !47
  %814 = insertelement <32 x float> %813, float %747, i64 1, !dbg !47
  %815 = insertelement <32 x float> %814, float %748, i64 2, !dbg !47
  %816 = insertelement <32 x float> %815, float %749, i64 3, !dbg !47
  %817 = insertelement <32 x float> %816, float %750, i64 4, !dbg !47
  %818 = insertelement <32 x float> %817, float %751, i64 5, !dbg !47
  %819 = insertelement <32 x float> %818, float %752, i64 6, !dbg !47
  %820 = insertelement <32 x float> %819, float %753, i64 7, !dbg !47
  %821 = insertelement <32 x float> %820, float %754, i64 8, !dbg !47
  %822 = insertelement <32 x float> %821, float %755, i64 9, !dbg !47
  %823 = insertelement <32 x float> %822, float %756, i64 10, !dbg !47
  %824 = insertelement <32 x float> %823, float %757, i64 11, !dbg !47
  %825 = insertelement <32 x float> %824, float %758, i64 12, !dbg !47
  %826 = insertelement <32 x float> %825, float %759, i64 13, !dbg !47
  %827 = insertelement <32 x float> %826, float %760, i64 14, !dbg !47
  %828 = insertelement <32 x float> %827, float %761, i64 15, !dbg !47
  %829 = insertelement <32 x float> %828, float %762, i64 16, !dbg !47
  %830 = insertelement <32 x float> %829, float %763, i64 17, !dbg !47
  %831 = insertelement <32 x float> %830, float %764, i64 18, !dbg !47
  %832 = insertelement <32 x float> %831, float %765, i64 19, !dbg !47
  %833 = insertelement <32 x float> %832, float %766, i64 20, !dbg !47
  %834 = insertelement <32 x float> %833, float %767, i64 21, !dbg !47
  %835 = insertelement <32 x float> %834, float %768, i64 22, !dbg !47
  %836 = insertelement <32 x float> %835, float %769, i64 23, !dbg !47
  %837 = insertelement <32 x float> %836, float %770, i64 24, !dbg !47
  %838 = insertelement <32 x float> %837, float %771, i64 25, !dbg !47
  %839 = insertelement <32 x float> %838, float %772, i64 26, !dbg !47
  %840 = insertelement <32 x float> %839, float %773, i64 27, !dbg !47
  %841 = insertelement <32 x float> %840, float %774, i64 28, !dbg !47
  %842 = insertelement <32 x float> %841, float %775, i64 29, !dbg !47
  %843 = insertelement <32 x float> %842, float %776, i64 30, !dbg !47
  %844 = insertelement <32 x float> %843, float %777, i64 31, !dbg !47
  %845 = insertvalue [2 x <32 x float>] poison, <32 x float> %844, 0, !dbg !47
  %846 = insertelement <32 x float> poison, float %781, i64 0, !dbg !47
  %847 = insertelement <32 x float> %846, float %782, i64 1, !dbg !47
  %848 = insertelement <32 x float> %847, float %783, i64 2, !dbg !47
  %849 = insertelement <32 x float> %848, float %784, i64 3, !dbg !47
  %850 = insertelement <32 x float> %849, float %785, i64 4, !dbg !47
  %851 = insertelement <32 x float> %850, float %786, i64 5, !dbg !47
  %852 = insertelement <32 x float> %851, float %787, i64 6, !dbg !47
  %853 = insertelement <32 x float> %852, float %788, i64 7, !dbg !47
  %854 = insertelement <32 x float> %853, float %789, i64 8, !dbg !47
  %855 = insertelement <32 x float> %854, float %790, i64 9, !dbg !47
  %856 = insertelement <32 x float> %855, float %791, i64 10, !dbg !47
  %857 = insertelement <32 x float> %856, float %792, i64 11, !dbg !47
  %858 = insertelement <32 x float> %857, float %793, i64 12, !dbg !47
  %859 = insertelement <32 x float> %858, float %794, i64 13, !dbg !47
  %860 = insertelement <32 x float> %859, float %795, i64 14, !dbg !47
  %861 = insertelement <32 x float> %860, float %796, i64 15, !dbg !47
  %862 = insertelement <32 x float> %861, float %797, i64 16, !dbg !47
  %863 = insertelement <32 x float> %862, float %798, i64 17, !dbg !47
  %864 = insertelement <32 x float> %863, float %799, i64 18, !dbg !47
  %865 = insertelement <32 x float> %864, float %800, i64 19, !dbg !47
  %866 = insertelement <32 x float> %865, float %801, i64 20, !dbg !47
  %867 = insertelement <32 x float> %866, float %802, i64 21, !dbg !47
  %868 = insertelement <32 x float> %867, float %803, i64 22, !dbg !47
  %869 = insertelement <32 x float> %868, float %804, i64 23, !dbg !47
  %870 = insertelement <32 x float> %869, float %805, i64 24, !dbg !47
  %871 = insertelement <32 x float> %870, float %806, i64 25, !dbg !47
  %872 = insertelement <32 x float> %871, float %807, i64 26, !dbg !47
  %873 = insertelement <32 x float> %872, float %808, i64 27, !dbg !47
  %874 = insertelement <32 x float> %873, float %809, i64 28, !dbg !47
  %875 = insertelement <32 x float> %874, float %810, i64 29, !dbg !47
  %876 = insertelement <32 x float> %875, float %811, i64 30, !dbg !47
  %877 = insertelement <32 x float> %876, float %812, i64 31, !dbg !47
  %878 = insertvalue [2 x <32 x float>] %845, <32 x float> %877, 1, !dbg !47
  br label %879, !dbg !47

879:                                              ; preds = %738, %737
  %880 = phi [2 x <32 x float>] [ %878, %738 ], [ zeroinitializer, %737 ], !dbg !47
  br label %881, !dbg !47

881:                                              ; preds = %879
  %882 = extractvalue [2 x <32 x float>] %880, 0, !dbg !61
  %883 = insertelement <4 x float> poison, float %670, i64 0, !dbg !61
  %884 = insertelement <4 x float> %883, float %671, i64 1, !dbg !61
  %885 = insertelement <4 x float> %884, float %672, i64 2, !dbg !61
  %886 = insertelement <4 x float> %885, float %673, i64 3, !dbg !61
  %887 = shufflevector <32 x float> %882, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !61
  %888 = fadd <4 x float> %886, %887, !dbg !61
  %889 = extractelement <4 x float> %888, i64 0, !dbg !61
  %890 = extractelement <4 x float> %888, i64 1, !dbg !61
  %891 = extractelement <4 x float> %888, i64 2, !dbg !61
  %892 = extractelement <4 x float> %888, i64 3, !dbg !61
  %893 = insertelement <4 x float> poison, float %674, i64 0, !dbg !61
  %894 = insertelement <4 x float> %893, float %675, i64 1, !dbg !61
  %895 = insertelement <4 x float> %894, float %676, i64 2, !dbg !61
  %896 = insertelement <4 x float> %895, float %677, i64 3, !dbg !61
  %897 = shufflevector <32 x float> %882, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !61
  %898 = fadd <4 x float> %896, %897, !dbg !61
  %899 = extractelement <4 x float> %898, i64 0, !dbg !61
  %900 = extractelement <4 x float> %898, i64 1, !dbg !61
  %901 = extractelement <4 x float> %898, i64 2, !dbg !61
  %902 = extractelement <4 x float> %898, i64 3, !dbg !61
  %903 = insertelement <4 x float> poison, float %678, i64 0, !dbg !61
  %904 = insertelement <4 x float> %903, float %679, i64 1, !dbg !61
  %905 = insertelement <4 x float> %904, float %680, i64 2, !dbg !61
  %906 = insertelement <4 x float> %905, float %681, i64 3, !dbg !61
  %907 = shufflevector <32 x float> %882, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !61
  %908 = fadd <4 x float> %906, %907, !dbg !61
  %909 = extractelement <4 x float> %908, i64 0, !dbg !61
  %910 = extractelement <4 x float> %908, i64 1, !dbg !61
  %911 = extractelement <4 x float> %908, i64 2, !dbg !61
  %912 = extractelement <4 x float> %908, i64 3, !dbg !61
  %913 = insertelement <4 x float> poison, float %682, i64 0, !dbg !61
  %914 = insertelement <4 x float> %913, float %683, i64 1, !dbg !61
  %915 = insertelement <4 x float> %914, float %684, i64 2, !dbg !61
  %916 = insertelement <4 x float> %915, float %685, i64 3, !dbg !61
  %917 = shufflevector <32 x float> %882, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !61
  %918 = fadd <4 x float> %916, %917, !dbg !61
  %919 = extractelement <4 x float> %918, i64 0, !dbg !61
  %920 = extractelement <4 x float> %918, i64 1, !dbg !61
  %921 = extractelement <4 x float> %918, i64 2, !dbg !61
  %922 = extractelement <4 x float> %918, i64 3, !dbg !61
  %923 = insertelement <4 x float> poison, float %686, i64 0, !dbg !61
  %924 = insertelement <4 x float> %923, float %687, i64 1, !dbg !61
  %925 = insertelement <4 x float> %924, float %688, i64 2, !dbg !61
  %926 = insertelement <4 x float> %925, float %689, i64 3, !dbg !61
  %927 = shufflevector <32 x float> %882, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !61
  %928 = fadd <4 x float> %926, %927, !dbg !61
  %929 = extractelement <4 x float> %928, i64 0, !dbg !61
  %930 = extractelement <4 x float> %928, i64 1, !dbg !61
  %931 = extractelement <4 x float> %928, i64 2, !dbg !61
  %932 = extractelement <4 x float> %928, i64 3, !dbg !61
  %933 = insertelement <4 x float> poison, float %690, i64 0, !dbg !61
  %934 = insertelement <4 x float> %933, float %691, i64 1, !dbg !61
  %935 = insertelement <4 x float> %934, float %692, i64 2, !dbg !61
  %936 = insertelement <4 x float> %935, float %693, i64 3, !dbg !61
  %937 = shufflevector <32 x float> %882, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !61
  %938 = fadd <4 x float> %936, %937, !dbg !61
  %939 = extractelement <4 x float> %938, i64 0, !dbg !61
  %940 = extractelement <4 x float> %938, i64 1, !dbg !61
  %941 = extractelement <4 x float> %938, i64 2, !dbg !61
  %942 = extractelement <4 x float> %938, i64 3, !dbg !61
  %943 = insertelement <4 x float> poison, float %694, i64 0, !dbg !61
  %944 = insertelement <4 x float> %943, float %695, i64 1, !dbg !61
  %945 = insertelement <4 x float> %944, float %696, i64 2, !dbg !61
  %946 = insertelement <4 x float> %945, float %697, i64 3, !dbg !61
  %947 = shufflevector <32 x float> %882, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !61
  %948 = fadd <4 x float> %946, %947, !dbg !61
  %949 = extractelement <4 x float> %948, i64 0, !dbg !61
  %950 = extractelement <4 x float> %948, i64 1, !dbg !61
  %951 = extractelement <4 x float> %948, i64 2, !dbg !61
  %952 = extractelement <4 x float> %948, i64 3, !dbg !61
  %953 = insertelement <4 x float> poison, float %698, i64 0, !dbg !61
  %954 = insertelement <4 x float> %953, float %699, i64 1, !dbg !61
  %955 = insertelement <4 x float> %954, float %700, i64 2, !dbg !61
  %956 = insertelement <4 x float> %955, float %701, i64 3, !dbg !61
  %957 = shufflevector <32 x float> %882, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !61
  %958 = fadd <4 x float> %956, %957, !dbg !61
  %959 = extractelement <4 x float> %958, i64 0, !dbg !61
  %960 = extractelement <4 x float> %958, i64 1, !dbg !61
  %961 = extractelement <4 x float> %958, i64 2, !dbg !61
  %962 = extractelement <4 x float> %958, i64 3, !dbg !61
  %963 = extractvalue [2 x <32 x float>] %880, 1, !dbg !61
  %964 = insertelement <4 x float> poison, float %702, i64 0, !dbg !61
  %965 = insertelement <4 x float> %964, float %703, i64 1, !dbg !61
  %966 = insertelement <4 x float> %965, float %704, i64 2, !dbg !61
  %967 = insertelement <4 x float> %966, float %705, i64 3, !dbg !61
  %968 = shufflevector <32 x float> %963, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !61
  %969 = fadd <4 x float> %967, %968, !dbg !61
  %970 = extractelement <4 x float> %969, i64 0, !dbg !61
  %971 = extractelement <4 x float> %969, i64 1, !dbg !61
  %972 = extractelement <4 x float> %969, i64 2, !dbg !61
  %973 = extractelement <4 x float> %969, i64 3, !dbg !61
  %974 = insertelement <4 x float> poison, float %706, i64 0, !dbg !61
  %975 = insertelement <4 x float> %974, float %707, i64 1, !dbg !61
  %976 = insertelement <4 x float> %975, float %708, i64 2, !dbg !61
  %977 = insertelement <4 x float> %976, float %709, i64 3, !dbg !61
  %978 = shufflevector <32 x float> %963, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !61
  %979 = fadd <4 x float> %977, %978, !dbg !61
  %980 = extractelement <4 x float> %979, i64 0, !dbg !61
  %981 = extractelement <4 x float> %979, i64 1, !dbg !61
  %982 = extractelement <4 x float> %979, i64 2, !dbg !61
  %983 = extractelement <4 x float> %979, i64 3, !dbg !61
  %984 = insertelement <4 x float> poison, float %710, i64 0, !dbg !61
  %985 = insertelement <4 x float> %984, float %711, i64 1, !dbg !61
  %986 = insertelement <4 x float> %985, float %712, i64 2, !dbg !61
  %987 = insertelement <4 x float> %986, float %713, i64 3, !dbg !61
  %988 = shufflevector <32 x float> %963, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !61
  %989 = fadd <4 x float> %987, %988, !dbg !61
  %990 = extractelement <4 x float> %989, i64 0, !dbg !61
  %991 = extractelement <4 x float> %989, i64 1, !dbg !61
  %992 = extractelement <4 x float> %989, i64 2, !dbg !61
  %993 = extractelement <4 x float> %989, i64 3, !dbg !61
  %994 = insertelement <4 x float> poison, float %714, i64 0, !dbg !61
  %995 = insertelement <4 x float> %994, float %715, i64 1, !dbg !61
  %996 = insertelement <4 x float> %995, float %716, i64 2, !dbg !61
  %997 = insertelement <4 x float> %996, float %717, i64 3, !dbg !61
  %998 = shufflevector <32 x float> %963, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !61
  %999 = fadd <4 x float> %997, %998, !dbg !61
  %1000 = extractelement <4 x float> %999, i64 0, !dbg !61
  %1001 = extractelement <4 x float> %999, i64 1, !dbg !61
  %1002 = extractelement <4 x float> %999, i64 2, !dbg !61
  %1003 = extractelement <4 x float> %999, i64 3, !dbg !61
  %1004 = insertelement <4 x float> poison, float %718, i64 0, !dbg !61
  %1005 = insertelement <4 x float> %1004, float %719, i64 1, !dbg !61
  %1006 = insertelement <4 x float> %1005, float %720, i64 2, !dbg !61
  %1007 = insertelement <4 x float> %1006, float %721, i64 3, !dbg !61
  %1008 = shufflevector <32 x float> %963, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !61
  %1009 = fadd <4 x float> %1007, %1008, !dbg !61
  %1010 = extractelement <4 x float> %1009, i64 0, !dbg !61
  %1011 = extractelement <4 x float> %1009, i64 1, !dbg !61
  %1012 = extractelement <4 x float> %1009, i64 2, !dbg !61
  %1013 = extractelement <4 x float> %1009, i64 3, !dbg !61
  %1014 = insertelement <4 x float> poison, float %722, i64 0, !dbg !61
  %1015 = insertelement <4 x float> %1014, float %723, i64 1, !dbg !61
  %1016 = insertelement <4 x float> %1015, float %724, i64 2, !dbg !61
  %1017 = insertelement <4 x float> %1016, float %725, i64 3, !dbg !61
  %1018 = shufflevector <32 x float> %963, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !61
  %1019 = fadd <4 x float> %1017, %1018, !dbg !61
  %1020 = extractelement <4 x float> %1019, i64 0, !dbg !61
  %1021 = extractelement <4 x float> %1019, i64 1, !dbg !61
  %1022 = extractelement <4 x float> %1019, i64 2, !dbg !61
  %1023 = extractelement <4 x float> %1019, i64 3, !dbg !61
  %1024 = insertelement <4 x float> poison, float %726, i64 0, !dbg !61
  %1025 = insertelement <4 x float> %1024, float %727, i64 1, !dbg !61
  %1026 = insertelement <4 x float> %1025, float %728, i64 2, !dbg !61
  %1027 = insertelement <4 x float> %1026, float %729, i64 3, !dbg !61
  %1028 = shufflevector <32 x float> %963, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !61
  %1029 = fadd <4 x float> %1027, %1028, !dbg !61
  %1030 = extractelement <4 x float> %1029, i64 0, !dbg !61
  %1031 = extractelement <4 x float> %1029, i64 1, !dbg !61
  %1032 = extractelement <4 x float> %1029, i64 2, !dbg !61
  %1033 = extractelement <4 x float> %1029, i64 3, !dbg !61
  %1034 = insertelement <4 x float> poison, float %730, i64 0, !dbg !61
  %1035 = insertelement <4 x float> %1034, float %731, i64 1, !dbg !61
  %1036 = insertelement <4 x float> %1035, float %732, i64 2, !dbg !61
  %1037 = insertelement <4 x float> %1036, float %733, i64 3, !dbg !61
  %1038 = shufflevector <32 x float> %963, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !61
  %1039 = fadd <4 x float> %1037, %1038, !dbg !61
  %1040 = extractelement <4 x float> %1039, i64 0, !dbg !61
  %1041 = extractelement <4 x float> %1039, i64 1, !dbg !61
  %1042 = extractelement <4 x float> %1039, i64 2, !dbg !61
  %1043 = extractelement <4 x float> %1039, i64 3, !dbg !61
  %1044 = call float @llvm.maxnum.f32(float %889, float -inf), !dbg !62, !frisk.local_reduction !37
  %local.reduce182 = call float @llvm.maxnum.f32(float %1044, float %899), !dbg !62, !frisk.local_reduction !37
  %local.reduce183 = call float @llvm.maxnum.f32(float %890, float %900), !dbg !62, !frisk.local_reduction !37
  %local.reduce184 = call float @llvm.maxnum.f32(float %891, float %901), !dbg !62, !frisk.local_reduction !37
  %local.reduce185 = call float @llvm.maxnum.f32(float %892, float %902), !dbg !62, !frisk.local_reduction !37
  %local.reduce186 = call float @llvm.maxnum.f32(float %local.reduce182, float %909), !dbg !62, !frisk.local_reduction !37
  %local.reduce187 = call float @llvm.maxnum.f32(float %local.reduce183, float %910), !dbg !62, !frisk.local_reduction !37
  %local.reduce188 = call float @llvm.maxnum.f32(float %local.reduce184, float %911), !dbg !62, !frisk.local_reduction !37
  %local.reduce189 = call float @llvm.maxnum.f32(float %local.reduce185, float %912), !dbg !62, !frisk.local_reduction !37
  %local.reduce190 = call float @llvm.maxnum.f32(float %local.reduce186, float %919), !dbg !62, !frisk.local_reduction !37
  %local.reduce191 = call float @llvm.maxnum.f32(float %local.reduce187, float %920), !dbg !62, !frisk.local_reduction !37
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %921), !dbg !62, !frisk.local_reduction !37
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %922), !dbg !62, !frisk.local_reduction !37
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %929), !dbg !62, !frisk.local_reduction !37
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %930), !dbg !62, !frisk.local_reduction !37
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %931), !dbg !62, !frisk.local_reduction !37
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %932), !dbg !62, !frisk.local_reduction !37
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %939), !dbg !62, !frisk.local_reduction !37
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %940), !dbg !62, !frisk.local_reduction !37
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %941), !dbg !62, !frisk.local_reduction !37
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %942), !dbg !62, !frisk.local_reduction !37
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %949), !dbg !62, !frisk.local_reduction !37
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %950), !dbg !62, !frisk.local_reduction !37
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %951), !dbg !62, !frisk.local_reduction !37
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %952), !dbg !62, !frisk.local_reduction !37
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %959), !dbg !62, !frisk.local_reduction !37
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %960), !dbg !62, !frisk.local_reduction !37
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %961), !dbg !62, !frisk.local_reduction !37
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %962), !dbg !62, !frisk.local_reduction !37
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %local.reduce207), !dbg !62, !frisk.local_reduction !37
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce208, float %local.reduce209), !dbg !62, !frisk.local_reduction !37
  %1045 = call float @llvm.maxnum.f32(float %local.reduce210, float %local.reduce211), !dbg !62, !frisk.local_reduction !37
  %1046 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !62
  %1047 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1046), !dbg !62
  %1048 = add nuw nsw i32 %1047, 64, !dbg !62
  %1049 = xor i32 %1047, 16, !dbg !62
  %1050 = and i32 %1048, 192, !dbg !62
  %1051 = icmp samesign ult i32 %1049, %1050, !dbg !62
  %1052 = select i1 %1051, i32 %1049, i32 %1047, !dbg !62
  %1053 = shl nuw nsw i32 %1052, 2, !dbg !62
  %1054 = bitcast float %1045 to i32, !dbg !62
  %1055 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1054), !dbg !62
  %1056 = bitcast i32 %1055 to float, !dbg !62
  %1057 = call float @llvm.maxnum.f32(float %1045, float %1056), !dbg !62
  %1058 = xor i32 %1047, 32, !dbg !62
  %1059 = icmp samesign ult i32 %1058, %1050, !dbg !62
  %1060 = select i1 %1059, i32 %1058, i32 %1047, !dbg !62
  %1061 = shl nuw nsw i32 %1060, 2, !dbg !62
  %1062 = bitcast float %1057 to i32, !dbg !62
  %1063 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1062), !dbg !62
  %1064 = bitcast i32 %1063 to float, !dbg !62
  %1065 = call float @llvm.maxnum.f32(float %1057, float %1064), !dbg !62
  %1066 = extractvalue [2 x <1 x float>] %57, 0, !dbg !62
  %1067 = insertelement <1 x float> poison, float %1065, i64 0, !dbg !62
  %1068 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1067, <1 x float> %1066), !dbg !62
  %1069 = call float @llvm.maxnum.f32(float %970, float -inf), !dbg !62, !frisk.local_reduction !37
  %local.reduce151 = call float @llvm.maxnum.f32(float %1069, float %980), !dbg !62, !frisk.local_reduction !37
  %local.reduce152 = call float @llvm.maxnum.f32(float %971, float %981), !dbg !62, !frisk.local_reduction !37
  %local.reduce153 = call float @llvm.maxnum.f32(float %972, float %982), !dbg !62, !frisk.local_reduction !37
  %local.reduce154 = call float @llvm.maxnum.f32(float %973, float %983), !dbg !62, !frisk.local_reduction !37
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %990), !dbg !62, !frisk.local_reduction !37
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %991), !dbg !62, !frisk.local_reduction !37
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %992), !dbg !62, !frisk.local_reduction !37
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %993), !dbg !62, !frisk.local_reduction !37
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %1000), !dbg !62, !frisk.local_reduction !37
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %1001), !dbg !62, !frisk.local_reduction !37
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %1002), !dbg !62, !frisk.local_reduction !37
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %1003), !dbg !62, !frisk.local_reduction !37
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %1010), !dbg !62, !frisk.local_reduction !37
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %1011), !dbg !62, !frisk.local_reduction !37
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %1012), !dbg !62, !frisk.local_reduction !37
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %1013), !dbg !62, !frisk.local_reduction !37
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %1020), !dbg !62, !frisk.local_reduction !37
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %1021), !dbg !62, !frisk.local_reduction !37
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %1022), !dbg !62, !frisk.local_reduction !37
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %1023), !dbg !62, !frisk.local_reduction !37
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %1030), !dbg !62, !frisk.local_reduction !37
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %1031), !dbg !62, !frisk.local_reduction !37
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %1032), !dbg !62, !frisk.local_reduction !37
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %1033), !dbg !62, !frisk.local_reduction !37
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %1040), !dbg !62, !frisk.local_reduction !37
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %1041), !dbg !62, !frisk.local_reduction !37
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %1042), !dbg !62, !frisk.local_reduction !37
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %1043), !dbg !62, !frisk.local_reduction !37
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !62, !frisk.local_reduction !37
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !62, !frisk.local_reduction !37
  %1070 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !62, !frisk.local_reduction !37
  %1071 = bitcast float %1070 to i32, !dbg !62
  %1072 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1071), !dbg !62
  %1073 = bitcast i32 %1072 to float, !dbg !62
  %1074 = call float @llvm.maxnum.f32(float %1070, float %1073), !dbg !62
  %1075 = bitcast float %1074 to i32, !dbg !62
  %1076 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1075), !dbg !62
  %1077 = bitcast i32 %1076 to float, !dbg !62
  %1078 = call float @llvm.maxnum.f32(float %1074, float %1077), !dbg !62
  %1079 = extractvalue [2 x <1 x float>] %57, 1, !dbg !62
  %1080 = insertelement <1 x float> poison, float %1078, i64 0, !dbg !62
  %1081 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1080, <1 x float> %1079), !dbg !62
  %1082 = shufflevector <1 x float> %1068, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !63
  %1083 = fsub <4 x float> %888, %1082, !dbg !63
  %1084 = fsub <4 x float> %898, %1082, !dbg !63
  %1085 = fsub <4 x float> %908, %1082, !dbg !63
  %1086 = fsub <4 x float> %918, %1082, !dbg !63
  %1087 = fsub <4 x float> %928, %1082, !dbg !63
  %1088 = fsub <4 x float> %938, %1082, !dbg !63
  %1089 = fsub <4 x float> %948, %1082, !dbg !63
  %1090 = fsub <4 x float> %958, %1082, !dbg !63
  %1091 = shufflevector <1 x float> %1081, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !63
  %1092 = fsub <4 x float> %969, %1091, !dbg !63
  %1093 = fsub <4 x float> %979, %1091, !dbg !63
  %1094 = fsub <4 x float> %989, %1091, !dbg !63
  %1095 = fsub <4 x float> %999, %1091, !dbg !63
  %1096 = fsub <4 x float> %1009, %1091, !dbg !63
  %1097 = fsub <4 x float> %1019, %1091, !dbg !63
  %1098 = fsub <4 x float> %1029, %1091, !dbg !63
  %1099 = fsub <4 x float> %1039, %1091, !dbg !63
  br i1 %43, label %1100, label %1107, !dbg !48

1100:                                             ; preds = %881
  %1101 = fsub <1 x float> %1079, %1081, !dbg !64
  %1102 = extractelement <1 x float> %1101, i64 0, !dbg !64
  %1103 = fsub <1 x float> %1066, %1068, !dbg !64
  %1104 = extractelement <1 x float> %1103, i64 0, !dbg !64
  %address.thread766 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !48, !range !73, !frisk.register_lifetimes !37
  %address.local767 = lshr i32 %address.thread766, 2, !dbg !46
  %address.local768 = and i32 %address.local767, 48, !dbg !46
  %address.local769 = and i32 %address.thread766, 15, !dbg !46
  %address.local770 = or disjoint i32 %address.local768, %address.local769, !dbg !46
  %address.local771 = zext nneg i32 %address.local770 to i64, !dbg !46
  %address.local772 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local771, !dbg !48
  store float %1104, ptr addrspace(3) %address.local772, align 4, !dbg !48
  %address.local773 = lshr i32 %address.thread766, 2, !dbg !46
  %address.local774 = and i32 %address.local773, 48, !dbg !46
  %address.local775 = and i32 %address.thread766, 15, !dbg !46
  %address.local776 = or disjoint i32 %address.local774, %address.local775, !dbg !46
  %1105 = zext nneg i32 %address.local776 to i64, !dbg !48
  %1106 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %1105, !dbg !48
  %address.local779 = getelementptr inbounds nuw i8, ptr addrspace(3) %1106, i64 256, !dbg !48
  store float %1102, ptr addrspace(3) %address.local779, align 4, !dbg !48
  br label %1107, !dbg !48

1107:                                             ; preds = %1100, %881
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %address.thread780 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !48, !range !73, !frisk.register_lifetimes !37
  %address.local781 = lshr i32 %address.thread780, 1, !dbg !48
  %address.local782 = and i32 %address.local781, 96, !dbg !48
  %address.local783 = and i32 %address.thread780, 15, !dbg !48
  %address.local784 = or disjoint i32 %address.local782, %address.local783, !dbg !48
  %address.local785 = zext nneg i32 %address.local784 to i64, !dbg !48
  %address.local786 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local785, !dbg !48
  %1108 = load float, ptr addrspace(3) %address.local786, align 4, !dbg !48
  %lds.const.gep1745 = getelementptr i8, ptr addrspace(3) %address.local786, i64 64, !dbg !48
  %1109 = load float, ptr addrspace(3) %lds.const.gep1745, align 4, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %1110 = call float @llvm.amdgcn.exp2.f32(float %1108), !dbg !48
  %1111 = call float @llvm.amdgcn.exp2.f32(float %1109), !dbg !48
  br i1 %43, label %1112, label %1113, !dbg !65

1112:                                             ; preds = %1107
  %address.thread789 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !65, !range !73, !frisk.register_lifetimes !37
  %address.local790 = lshr i32 %address.thread789, 1, !dbg !48
  %address.local791 = and i32 %address.local790, 96, !dbg !48
  %address.local792 = and i32 %address.thread789, 15, !dbg !48
  %address.local793 = or disjoint i32 %address.local791, %address.local792, !dbg !48
  %address.local794 = zext nneg i32 %address.local793 to i64, !dbg !48
  %address.local795 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !48
  store float %1110, ptr addrspace(3) %address.local795, align 4, !dbg !65
  %address.local796 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !48
  %address.local797 = getelementptr i8, ptr addrspace(3) %address.local796, i64 64, !dbg !48
  store float %1111, ptr addrspace(3) %address.local797, align 4, !dbg !65
  br label %1113, !dbg !65

1113:                                             ; preds = %1112, %1107
  fence syncscope("workgroup") release, !dbg !65
  call void @llvm.amdgcn.s.barrier(), !dbg !65
  fence syncscope("workgroup") acquire, !dbg !65
  %address.thread798 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !65, !range !73, !frisk.register_lifetimes !37
  %address.local799 = lshr i32 %address.thread798, 2, !dbg !46
  %address.local800 = and i32 %address.local799, 48, !dbg !46
  %address.local801 = and i32 %address.thread798, 15, !dbg !46
  %address.local802 = or disjoint i32 %address.local800, %address.local801, !dbg !46
  %address.local803 = zext nneg i32 %address.local802 to i64, !dbg !46
  %address.local804 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local803, !dbg !65
  %1114 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !65
  %lds.const.gep1746 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !65
  %1115 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !65
  fence syncscope("workgroup") release, !dbg !65
  call void @llvm.amdgcn.s.barrier(), !dbg !65
  fence syncscope("workgroup") acquire, !dbg !65
  %1116 = add nuw nsw i64 %76, %15, !dbg !53
  %1117 = shl i64 %1116, 19, !dbg !53
  %1118 = and i64 %1117, 16252928, !dbg !53
  %address.local812 = zext nneg i32 %address.thread798 to i64, !dbg !45
  %address.local813 = shl nuw nsw i64 %address.local812, 3, !dbg !44
  %address.local814 = and i64 %address.local813, 1536, !dbg !44
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local814, !dbg !44
  %address.local816 = ptrtoint ptr addrspace(3) %address.local815 to i64, !dbg !44
  %address.local817 = trunc i64 %address.local816 to i32, !dbg !44
  %1119 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local817), !dbg !53
  %address.local818 = shl nuw nsw i64 %address.local812, 6, !dbg !53
  %address.local819 = and i64 %address.local818, 16128, !dbg !53
  %address.local820 = shl nuw nsw i64 %address.local812, 4, !dbg !53
  %address.local821 = and i64 %address.local820, 48, !dbg !53
  %1120 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !53
  %.idx48 = add i64 %1118, %82, !dbg !53
  %1121 = shl i64 %.idx48, 1, !dbg !53
  %1122 = or disjoint i64 %address.local819, %address.local821, !dbg !53
  %1123 = or disjoint i64 %1121, %1122, !dbg !53
  %dma.buffer.offset1773 = trunc i64 %1123 to i32, !dbg !53
  %1124 = insertelement <2 x i64> poison, i64 %1120, i64 0, !dbg !53
  %1125 = bitcast <2 x i64> %1124 to <4 x i32>, !dbg !53
  %dma.buffer.rsrc1774 = shufflevector <4 x i32> %1125, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !53
  %dma.buffer.lds1775 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1119), !dbg !53
  %1126 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1775, i32 %dma.buffer.offset1773, <4 x i32> %dma.buffer.rsrc1774) #10, !dbg !53
  %address.local824 = or disjoint i64 %address.local812, 256, !dbg !44
  %address.local825 = shl nuw nsw i64 %address.local824, 3, !dbg !44
  %address.local826 = and i64 %address.local825, 3584, !dbg !44
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local826, !dbg !44
  %address.local828 = ptrtoint ptr addrspace(3) %address.local827 to i64, !dbg !44
  %address.local829 = trunc i64 %address.local828 to i32, !dbg !44
  %1127 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local829), !dbg !53
  %address.local830 = shl nuw nsw i64 %address.local824, 6, !dbg !53
  %address.local831 = and i64 %address.local830, 32512, !dbg !53
  %address.local832 = shl nuw nsw i64 %address.local812, 4, !dbg !53
  %address.local833 = and i64 %address.local832, 48, !dbg !53
  %1128 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !53
  %.idx51 = add i64 %1118, %82, !dbg !53
  %1129 = shl i64 %.idx51, 1, !dbg !53
  %1130 = or disjoint i64 %1129, %address.local831, !dbg !53
  %1131 = or disjoint i64 %1130, %address.local833, !dbg !53
  %dma.buffer.offset1770 = trunc i64 %1131 to i32, !dbg !53
  %1132 = insertelement <2 x i64> poison, i64 %1128, i64 0, !dbg !53
  %1133 = bitcast <2 x i64> %1132 to <4 x i32>, !dbg !53
  %dma.buffer.rsrc1771 = shufflevector <4 x i32> %1133, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !53
  %dma.buffer.lds1772 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1127), !dbg !53
  %1134 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1772, i32 %dma.buffer.offset1770, <4 x i32> %dma.buffer.rsrc1771) #10, !dbg !53
  %1135 = extractvalue [2 x <1 x float>] %56, 0, !dbg !65
  %1136 = extractelement <1 x float> %1135, i64 0, !dbg !65
  %1137 = fmul float %1136, %1114, !dbg !65
  %1138 = extractvalue [2 x <1 x float>] %56, 1, !dbg !65
  %1139 = extractelement <1 x float> %1138, i64 0, !dbg !65
  %1140 = fmul float %1139, %1115, !dbg !65
  %1141 = extractelement <4 x float> %1083, i64 0, !dbg !66
  %1142 = call float @llvm.amdgcn.exp2.f32(float %1141), !dbg !66
  %1143 = insertelement <4 x float> poison, float %1142, i64 0, !dbg !68
  %1144 = extractelement <4 x float> %1083, i64 1, !dbg !66
  %1145 = call float @llvm.amdgcn.exp2.f32(float %1144), !dbg !66
  %1146 = insertelement <4 x float> %1143, float %1145, i64 1, !dbg !68
  %1147 = extractelement <4 x float> %1083, i64 2, !dbg !66
  %1148 = call float @llvm.amdgcn.exp2.f32(float %1147), !dbg !66
  %1149 = insertelement <4 x float> %1146, float %1148, i64 2, !dbg !68
  %1150 = extractelement <4 x float> %1083, i64 3, !dbg !66
  %1151 = call float @llvm.amdgcn.exp2.f32(float %1150), !dbg !66
  %1152 = insertelement <4 x float> %1149, float %1151, i64 3, !dbg !68
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1152) #9, !dbg !68, !frisk.register_lifetimes !37
  %1153 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !68
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1153) #9, !dbg !66, !frisk.register_lifetimes !37
  %1154 = extractelement <4 x float> %1084, i64 0, !dbg !66
  %1155 = call float @llvm.amdgcn.exp2.f32(float %1154), !dbg !66
  %1156 = insertelement <4 x float> poison, float %1155, i64 0, !dbg !68
  %local.reduce120 = fadd float %1142, %1155, !dbg !67, !frisk.local_reduction !37
  %1157 = extractelement <4 x float> %1084, i64 1, !dbg !66
  %1158 = call float @llvm.amdgcn.exp2.f32(float %1157), !dbg !66
  %1159 = insertelement <4 x float> %1156, float %1158, i64 1, !dbg !68
  %local.reduce121 = fadd float %1145, %1158, !dbg !67, !frisk.local_reduction !37
  %1160 = extractelement <4 x float> %1084, i64 2, !dbg !66
  %1161 = call float @llvm.amdgcn.exp2.f32(float %1160), !dbg !66
  %1162 = insertelement <4 x float> %1159, float %1161, i64 2, !dbg !68
  %local.reduce122 = fadd float %1148, %1161, !dbg !67, !frisk.local_reduction !37
  %1163 = extractelement <4 x float> %1084, i64 3, !dbg !66
  %1164 = call float @llvm.amdgcn.exp2.f32(float %1163), !dbg !66
  %1165 = insertelement <4 x float> %1162, float %1164, i64 3, !dbg !68
  %math.result228 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1165) #9, !dbg !68, !frisk.register_lifetimes !37
  %1166 = fptrunc <4 x float> %math.result228 to <4 x half>, !dbg !68
  %math.packed229 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1166) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce123 = fadd float %1151, %1164, !dbg !67, !frisk.local_reduction !37
  %1167 = extractelement <4 x float> %1085, i64 0, !dbg !66
  %1168 = call float @llvm.amdgcn.exp2.f32(float %1167), !dbg !66
  %1169 = insertelement <4 x float> poison, float %1168, i64 0, !dbg !68
  %local.reduce124 = fadd float %local.reduce120, %1168, !dbg !67, !frisk.local_reduction !37
  %1170 = extractelement <4 x float> %1085, i64 1, !dbg !66
  %1171 = call float @llvm.amdgcn.exp2.f32(float %1170), !dbg !66
  %1172 = insertelement <4 x float> %1169, float %1171, i64 1, !dbg !68
  %local.reduce125 = fadd float %local.reduce121, %1171, !dbg !67, !frisk.local_reduction !37
  %1173 = extractelement <4 x float> %1085, i64 2, !dbg !66
  %1174 = call float @llvm.amdgcn.exp2.f32(float %1173), !dbg !66
  %1175 = insertelement <4 x float> %1172, float %1174, i64 2, !dbg !68
  %local.reduce126 = fadd float %local.reduce122, %1174, !dbg !67, !frisk.local_reduction !37
  %1176 = extractelement <4 x float> %1085, i64 3, !dbg !66
  %1177 = call float @llvm.amdgcn.exp2.f32(float %1176), !dbg !66
  %1178 = insertelement <4 x float> %1175, float %1177, i64 3, !dbg !68
  %math.result230 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1178) #9, !dbg !68, !frisk.register_lifetimes !37
  %1179 = fptrunc <4 x float> %math.result230 to <4 x half>, !dbg !68
  %math.packed231 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1179) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce127 = fadd float %local.reduce123, %1177, !dbg !67, !frisk.local_reduction !37
  %1180 = extractelement <4 x float> %1086, i64 0, !dbg !66
  %1181 = call float @llvm.amdgcn.exp2.f32(float %1180), !dbg !66
  %1182 = insertelement <4 x float> poison, float %1181, i64 0, !dbg !68
  %local.reduce128 = fadd float %local.reduce124, %1181, !dbg !67, !frisk.local_reduction !37
  %1183 = extractelement <4 x float> %1086, i64 1, !dbg !66
  %1184 = call float @llvm.amdgcn.exp2.f32(float %1183), !dbg !66
  %1185 = insertelement <4 x float> %1182, float %1184, i64 1, !dbg !68
  %local.reduce129 = fadd float %local.reduce125, %1184, !dbg !67, !frisk.local_reduction !37
  %1186 = extractelement <4 x float> %1086, i64 2, !dbg !66
  %1187 = call float @llvm.amdgcn.exp2.f32(float %1186), !dbg !66
  %1188 = insertelement <4 x float> %1185, float %1187, i64 2, !dbg !68
  %local.reduce130 = fadd float %local.reduce126, %1187, !dbg !67, !frisk.local_reduction !37
  %1189 = extractelement <4 x float> %1086, i64 3, !dbg !66
  %1190 = call float @llvm.amdgcn.exp2.f32(float %1189), !dbg !66
  %1191 = insertelement <4 x float> %1188, float %1190, i64 3, !dbg !68
  %math.result232 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1191) #9, !dbg !68, !frisk.register_lifetimes !37
  %1192 = fptrunc <4 x float> %math.result232 to <4 x half>, !dbg !68
  %math.packed233 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1192) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce131 = fadd float %local.reduce127, %1190, !dbg !67, !frisk.local_reduction !37
  %1193 = extractelement <4 x float> %1087, i64 0, !dbg !66
  %1194 = call float @llvm.amdgcn.exp2.f32(float %1193), !dbg !66
  %1195 = insertelement <4 x float> poison, float %1194, i64 0, !dbg !68
  %local.reduce132 = fadd float %local.reduce128, %1194, !dbg !67, !frisk.local_reduction !37
  %1196 = extractelement <4 x float> %1087, i64 1, !dbg !66
  %1197 = call float @llvm.amdgcn.exp2.f32(float %1196), !dbg !66
  %1198 = insertelement <4 x float> %1195, float %1197, i64 1, !dbg !68
  %local.reduce133 = fadd float %local.reduce129, %1197, !dbg !67, !frisk.local_reduction !37
  %1199 = extractelement <4 x float> %1087, i64 2, !dbg !66
  %1200 = call float @llvm.amdgcn.exp2.f32(float %1199), !dbg !66
  %1201 = insertelement <4 x float> %1198, float %1200, i64 2, !dbg !68
  %local.reduce134 = fadd float %local.reduce130, %1200, !dbg !67, !frisk.local_reduction !37
  %1202 = extractelement <4 x float> %1087, i64 3, !dbg !66
  %1203 = call float @llvm.amdgcn.exp2.f32(float %1202), !dbg !66
  %1204 = insertelement <4 x float> %1201, float %1203, i64 3, !dbg !68
  %math.result234 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1204) #9, !dbg !68, !frisk.register_lifetimes !37
  %1205 = fptrunc <4 x float> %math.result234 to <4 x half>, !dbg !68
  %math.packed235 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1205) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce135 = fadd float %local.reduce131, %1203, !dbg !67, !frisk.local_reduction !37
  %1206 = extractelement <4 x float> %1088, i64 0, !dbg !66
  %1207 = call float @llvm.amdgcn.exp2.f32(float %1206), !dbg !66
  %1208 = insertelement <4 x float> poison, float %1207, i64 0, !dbg !68
  %local.reduce136 = fadd float %local.reduce132, %1207, !dbg !67, !frisk.local_reduction !37
  %1209 = extractelement <4 x float> %1088, i64 1, !dbg !66
  %1210 = call float @llvm.amdgcn.exp2.f32(float %1209), !dbg !66
  %1211 = insertelement <4 x float> %1208, float %1210, i64 1, !dbg !68
  %local.reduce137 = fadd float %local.reduce133, %1210, !dbg !67, !frisk.local_reduction !37
  %1212 = extractelement <4 x float> %1088, i64 2, !dbg !66
  %1213 = call float @llvm.amdgcn.exp2.f32(float %1212), !dbg !66
  %1214 = insertelement <4 x float> %1211, float %1213, i64 2, !dbg !68
  %local.reduce138 = fadd float %local.reduce134, %1213, !dbg !67, !frisk.local_reduction !37
  %1215 = extractelement <4 x float> %1088, i64 3, !dbg !66
  %1216 = call float @llvm.amdgcn.exp2.f32(float %1215), !dbg !66
  %1217 = insertelement <4 x float> %1214, float %1216, i64 3, !dbg !68
  %math.result236 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1217) #9, !dbg !68, !frisk.register_lifetimes !37
  %1218 = fptrunc <4 x float> %math.result236 to <4 x half>, !dbg !68
  %math.packed237 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1218) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce139 = fadd float %local.reduce135, %1216, !dbg !67, !frisk.local_reduction !37
  %1219 = extractelement <4 x float> %1089, i64 0, !dbg !66
  %1220 = call float @llvm.amdgcn.exp2.f32(float %1219), !dbg !66
  %1221 = insertelement <4 x float> poison, float %1220, i64 0, !dbg !68
  %local.reduce140 = fadd float %local.reduce136, %1220, !dbg !67, !frisk.local_reduction !37
  %1222 = extractelement <4 x float> %1089, i64 1, !dbg !66
  %1223 = call float @llvm.amdgcn.exp2.f32(float %1222), !dbg !66
  %1224 = insertelement <4 x float> %1221, float %1223, i64 1, !dbg !68
  %local.reduce141 = fadd float %local.reduce137, %1223, !dbg !67, !frisk.local_reduction !37
  %1225 = extractelement <4 x float> %1089, i64 2, !dbg !66
  %1226 = call float @llvm.amdgcn.exp2.f32(float %1225), !dbg !66
  %1227 = insertelement <4 x float> %1224, float %1226, i64 2, !dbg !68
  %local.reduce142 = fadd float %local.reduce138, %1226, !dbg !67, !frisk.local_reduction !37
  %1228 = extractelement <4 x float> %1089, i64 3, !dbg !66
  %1229 = call float @llvm.amdgcn.exp2.f32(float %1228), !dbg !66
  %1230 = insertelement <4 x float> %1227, float %1229, i64 3, !dbg !68
  %math.result238 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1230) #9, !dbg !68, !frisk.register_lifetimes !37
  %1231 = fptrunc <4 x float> %math.result238 to <4 x half>, !dbg !68
  %math.packed239 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1231) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce143 = fadd float %local.reduce139, %1229, !dbg !67, !frisk.local_reduction !37
  %1232 = extractelement <4 x float> %1090, i64 0, !dbg !66
  %1233 = call float @llvm.amdgcn.exp2.f32(float %1232), !dbg !66
  %1234 = insertelement <4 x float> poison, float %1233, i64 0, !dbg !68
  %local.reduce144 = fadd float %local.reduce140, %1233, !dbg !67, !frisk.local_reduction !37
  %1235 = extractelement <4 x float> %1090, i64 1, !dbg !66
  %1236 = call float @llvm.amdgcn.exp2.f32(float %1235), !dbg !66
  %1237 = insertelement <4 x float> %1234, float %1236, i64 1, !dbg !68
  %local.reduce145 = fadd float %local.reduce141, %1236, !dbg !67, !frisk.local_reduction !37
  %local.reduce148 = fadd float %local.reduce144, %local.reduce145, !dbg !67, !frisk.local_reduction !37
  %1238 = extractelement <4 x float> %1090, i64 2, !dbg !66
  %1239 = call float @llvm.amdgcn.exp2.f32(float %1238), !dbg !66
  %1240 = insertelement <4 x float> %1237, float %1239, i64 2, !dbg !68
  %local.reduce146 = fadd float %local.reduce142, %1239, !dbg !67, !frisk.local_reduction !37
  %1241 = extractelement <4 x float> %1090, i64 3, !dbg !66
  %1242 = call float @llvm.amdgcn.exp2.f32(float %1241), !dbg !66
  %1243 = insertelement <4 x float> %1240, float %1242, i64 3, !dbg !68
  %math.result240 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1243) #9, !dbg !68, !frisk.register_lifetimes !37
  %1244 = fptrunc <4 x float> %math.result240 to <4 x half>, !dbg !68
  %math.packed241 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1244) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce147 = fadd float %local.reduce143, %1242, !dbg !67, !frisk.local_reduction !37
  %local.reduce149 = fadd float %local.reduce146, %local.reduce147, !dbg !67, !frisk.local_reduction !37
  %1245 = fadd float %local.reduce148, %local.reduce149, !dbg !67, !frisk.local_reduction !37
  %1246 = extractelement <4 x float> %1092, i64 0, !dbg !66
  %1247 = call float @llvm.amdgcn.exp2.f32(float %1246), !dbg !66
  %1248 = insertelement <4 x float> poison, float %1247, i64 0, !dbg !68
  %1249 = extractelement <4 x float> %1092, i64 1, !dbg !66
  %1250 = call float @llvm.amdgcn.exp2.f32(float %1249), !dbg !66
  %1251 = insertelement <4 x float> %1248, float %1250, i64 1, !dbg !68
  %1252 = extractelement <4 x float> %1092, i64 2, !dbg !66
  %1253 = call float @llvm.amdgcn.exp2.f32(float %1252), !dbg !66
  %1254 = insertelement <4 x float> %1251, float %1253, i64 2, !dbg !68
  %1255 = extractelement <4 x float> %1092, i64 3, !dbg !66
  %1256 = call float @llvm.amdgcn.exp2.f32(float %1255), !dbg !66
  %1257 = insertelement <4 x float> %1254, float %1256, i64 3, !dbg !68
  %math.result242 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1257) #9, !dbg !68, !frisk.register_lifetimes !37
  %1258 = fptrunc <4 x float> %math.result242 to <4 x half>, !dbg !68
  %math.packed243 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1258) #9, !dbg !66, !frisk.register_lifetimes !37
  %1259 = extractelement <4 x float> %1093, i64 0, !dbg !66
  %1260 = call float @llvm.amdgcn.exp2.f32(float %1259), !dbg !66
  %1261 = insertelement <4 x float> poison, float %1260, i64 0, !dbg !68
  %local.reduce = fadd float %1247, %1260, !dbg !67, !frisk.local_reduction !37
  %1262 = extractelement <4 x float> %1093, i64 1, !dbg !66
  %1263 = call float @llvm.amdgcn.exp2.f32(float %1262), !dbg !66
  %1264 = insertelement <4 x float> %1261, float %1263, i64 1, !dbg !68
  %local.reduce90 = fadd float %1250, %1263, !dbg !67, !frisk.local_reduction !37
  %1265 = extractelement <4 x float> %1093, i64 2, !dbg !66
  %1266 = call float @llvm.amdgcn.exp2.f32(float %1265), !dbg !66
  %1267 = insertelement <4 x float> %1264, float %1266, i64 2, !dbg !68
  %local.reduce91 = fadd float %1253, %1266, !dbg !67, !frisk.local_reduction !37
  %1268 = extractelement <4 x float> %1093, i64 3, !dbg !66
  %1269 = call float @llvm.amdgcn.exp2.f32(float %1268), !dbg !66
  %1270 = insertelement <4 x float> %1267, float %1269, i64 3, !dbg !68
  %math.result244 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1270) #9, !dbg !68, !frisk.register_lifetimes !37
  %1271 = fptrunc <4 x float> %math.result244 to <4 x half>, !dbg !68
  %math.packed245 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1271) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce92 = fadd float %1256, %1269, !dbg !67, !frisk.local_reduction !37
  %1272 = extractelement <4 x float> %1094, i64 0, !dbg !66
  %1273 = call float @llvm.amdgcn.exp2.f32(float %1272), !dbg !66
  %1274 = insertelement <4 x float> poison, float %1273, i64 0, !dbg !68
  %local.reduce93 = fadd float %local.reduce, %1273, !dbg !67, !frisk.local_reduction !37
  %1275 = extractelement <4 x float> %1094, i64 1, !dbg !66
  %1276 = call float @llvm.amdgcn.exp2.f32(float %1275), !dbg !66
  %1277 = insertelement <4 x float> %1274, float %1276, i64 1, !dbg !68
  %local.reduce94 = fadd float %local.reduce90, %1276, !dbg !67, !frisk.local_reduction !37
  %1278 = extractelement <4 x float> %1094, i64 2, !dbg !66
  %1279 = call float @llvm.amdgcn.exp2.f32(float %1278), !dbg !66
  %1280 = insertelement <4 x float> %1277, float %1279, i64 2, !dbg !68
  %local.reduce95 = fadd float %local.reduce91, %1279, !dbg !67, !frisk.local_reduction !37
  %1281 = extractelement <4 x float> %1094, i64 3, !dbg !66
  %1282 = call float @llvm.amdgcn.exp2.f32(float %1281), !dbg !66
  %1283 = insertelement <4 x float> %1280, float %1282, i64 3, !dbg !68
  %math.result246 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1283) #9, !dbg !68, !frisk.register_lifetimes !37
  %1284 = fptrunc <4 x float> %math.result246 to <4 x half>, !dbg !68
  %math.packed247 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1284) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce96 = fadd float %local.reduce92, %1282, !dbg !67, !frisk.local_reduction !37
  %1285 = extractelement <4 x float> %1095, i64 0, !dbg !66
  %1286 = call float @llvm.amdgcn.exp2.f32(float %1285), !dbg !66
  %1287 = insertelement <4 x float> poison, float %1286, i64 0, !dbg !68
  %local.reduce97 = fadd float %local.reduce93, %1286, !dbg !67, !frisk.local_reduction !37
  %1288 = extractelement <4 x float> %1095, i64 1, !dbg !66
  %1289 = call float @llvm.amdgcn.exp2.f32(float %1288), !dbg !66
  %1290 = insertelement <4 x float> %1287, float %1289, i64 1, !dbg !68
  %local.reduce98 = fadd float %local.reduce94, %1289, !dbg !67, !frisk.local_reduction !37
  %1291 = extractelement <4 x float> %1095, i64 2, !dbg !66
  %1292 = call float @llvm.amdgcn.exp2.f32(float %1291), !dbg !66
  %1293 = insertelement <4 x float> %1290, float %1292, i64 2, !dbg !68
  %local.reduce99 = fadd float %local.reduce95, %1292, !dbg !67, !frisk.local_reduction !37
  %1294 = extractelement <4 x float> %1095, i64 3, !dbg !66
  %1295 = call float @llvm.amdgcn.exp2.f32(float %1294), !dbg !66
  %1296 = insertelement <4 x float> %1293, float %1295, i64 3, !dbg !68
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1296) #9, !dbg !68, !frisk.register_lifetimes !37
  %1297 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !68
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1297) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce100 = fadd float %local.reduce96, %1295, !dbg !67, !frisk.local_reduction !37
  %1298 = extractelement <4 x float> %1096, i64 0, !dbg !66
  %1299 = call float @llvm.amdgcn.exp2.f32(float %1298), !dbg !66
  %1300 = insertelement <4 x float> poison, float %1299, i64 0, !dbg !68
  %local.reduce101 = fadd float %local.reduce97, %1299, !dbg !67, !frisk.local_reduction !37
  %1301 = extractelement <4 x float> %1096, i64 1, !dbg !66
  %1302 = call float @llvm.amdgcn.exp2.f32(float %1301), !dbg !66
  %1303 = insertelement <4 x float> %1300, float %1302, i64 1, !dbg !68
  %local.reduce102 = fadd float %local.reduce98, %1302, !dbg !67, !frisk.local_reduction !37
  %1304 = extractelement <4 x float> %1096, i64 2, !dbg !66
  %1305 = call float @llvm.amdgcn.exp2.f32(float %1304), !dbg !66
  %1306 = insertelement <4 x float> %1303, float %1305, i64 2, !dbg !68
  %local.reduce103 = fadd float %local.reduce99, %1305, !dbg !67, !frisk.local_reduction !37
  %1307 = extractelement <4 x float> %1096, i64 3, !dbg !66
  %1308 = call float @llvm.amdgcn.exp2.f32(float %1307), !dbg !66
  %1309 = insertelement <4 x float> %1306, float %1308, i64 3, !dbg !68
  %math.result250 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1309) #9, !dbg !68, !frisk.register_lifetimes !37
  %1310 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !68
  %math.packed251 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1310) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce104 = fadd float %local.reduce100, %1308, !dbg !67, !frisk.local_reduction !37
  %1311 = extractelement <4 x float> %1097, i64 0, !dbg !66
  %1312 = call float @llvm.amdgcn.exp2.f32(float %1311), !dbg !66
  %1313 = insertelement <4 x float> poison, float %1312, i64 0, !dbg !68
  %local.reduce105 = fadd float %local.reduce101, %1312, !dbg !67, !frisk.local_reduction !37
  %1314 = extractelement <4 x float> %1097, i64 1, !dbg !66
  %1315 = call float @llvm.amdgcn.exp2.f32(float %1314), !dbg !66
  %1316 = insertelement <4 x float> %1313, float %1315, i64 1, !dbg !68
  %local.reduce106 = fadd float %local.reduce102, %1315, !dbg !67, !frisk.local_reduction !37
  %1317 = extractelement <4 x float> %1097, i64 2, !dbg !66
  %1318 = call float @llvm.amdgcn.exp2.f32(float %1317), !dbg !66
  %1319 = insertelement <4 x float> %1316, float %1318, i64 2, !dbg !68
  %local.reduce107 = fadd float %local.reduce103, %1318, !dbg !67, !frisk.local_reduction !37
  %1320 = extractelement <4 x float> %1097, i64 3, !dbg !66
  %1321 = call float @llvm.amdgcn.exp2.f32(float %1320), !dbg !66
  %1322 = insertelement <4 x float> %1319, float %1321, i64 3, !dbg !68
  %math.result252 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1322) #9, !dbg !68, !frisk.register_lifetimes !37
  %1323 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !68
  %math.packed253 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1323) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce108 = fadd float %local.reduce104, %1321, !dbg !67, !frisk.local_reduction !37
  %1324 = extractelement <4 x float> %1098, i64 0, !dbg !66
  %1325 = call float @llvm.amdgcn.exp2.f32(float %1324), !dbg !66
  %1326 = insertelement <4 x float> poison, float %1325, i64 0, !dbg !68
  %local.reduce109 = fadd float %local.reduce105, %1325, !dbg !67, !frisk.local_reduction !37
  %1327 = extractelement <4 x float> %1098, i64 1, !dbg !66
  %1328 = call float @llvm.amdgcn.exp2.f32(float %1327), !dbg !66
  %1329 = insertelement <4 x float> %1326, float %1328, i64 1, !dbg !68
  %local.reduce110 = fadd float %local.reduce106, %1328, !dbg !67, !frisk.local_reduction !37
  %1330 = extractelement <4 x float> %1098, i64 2, !dbg !66
  %1331 = call float @llvm.amdgcn.exp2.f32(float %1330), !dbg !66
  %1332 = insertelement <4 x float> %1329, float %1331, i64 2, !dbg !68
  %local.reduce111 = fadd float %local.reduce107, %1331, !dbg !67, !frisk.local_reduction !37
  %1333 = extractelement <4 x float> %1098, i64 3, !dbg !66
  %1334 = call float @llvm.amdgcn.exp2.f32(float %1333), !dbg !66
  %1335 = insertelement <4 x float> %1332, float %1334, i64 3, !dbg !68
  %math.result254 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1335) #9, !dbg !68, !frisk.register_lifetimes !37
  %1336 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !68
  %math.packed255 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1336) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce112 = fadd float %local.reduce108, %1334, !dbg !67, !frisk.local_reduction !37
  %1337 = extractelement <4 x float> %1099, i64 0, !dbg !66
  %1338 = call float @llvm.amdgcn.exp2.f32(float %1337), !dbg !66
  %1339 = insertelement <4 x float> poison, float %1338, i64 0, !dbg !68
  %local.reduce113 = fadd float %local.reduce109, %1338, !dbg !67, !frisk.local_reduction !37
  %1340 = extractelement <4 x float> %1099, i64 1, !dbg !66
  %1341 = call float @llvm.amdgcn.exp2.f32(float %1340), !dbg !66
  %1342 = insertelement <4 x float> %1339, float %1341, i64 1, !dbg !68
  %local.reduce114 = fadd float %local.reduce110, %1341, !dbg !67, !frisk.local_reduction !37
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !67, !frisk.local_reduction !37
  %1343 = extractelement <4 x float> %1099, i64 2, !dbg !66
  %1344 = call float @llvm.amdgcn.exp2.f32(float %1343), !dbg !66
  %1345 = insertelement <4 x float> %1342, float %1344, i64 2, !dbg !68
  %local.reduce115 = fadd float %local.reduce111, %1344, !dbg !67, !frisk.local_reduction !37
  %1346 = extractelement <4 x float> %1099, i64 3, !dbg !66
  %1347 = call float @llvm.amdgcn.exp2.f32(float %1346), !dbg !66
  %1348 = insertelement <4 x float> %1345, float %1347, i64 3, !dbg !68
  %math.result256 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1348) #9, !dbg !68, !frisk.register_lifetimes !37
  %1349 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !68
  %math.packed257 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1349) #9, !dbg !67, !frisk.register_lifetimes !37
  %local.reduce116 = fadd float %local.reduce112, %1347, !dbg !67, !frisk.local_reduction !37
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !67, !frisk.local_reduction !37
  %1350 = fadd float %local.reduce117, %local.reduce118, !dbg !67, !frisk.local_reduction !37
  %1351 = bitcast float %1245 to i32, !dbg !67
  %1352 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1351), !dbg !67
  %1353 = bitcast i32 %1352 to float, !dbg !67
  %1354 = fadd float %1245, %1353, !dbg !67
  %1355 = bitcast float %1354 to i32, !dbg !67
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1355), !dbg !67
  %1357 = bitcast i32 %1356 to float, !dbg !67
  %1358 = fadd float %1354, %1357, !dbg !67
  %1359 = fadd float %1358, %1137, !dbg !67
  %1360 = bitcast float %1350 to i32, !dbg !67
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1360), !dbg !67
  %1362 = bitcast i32 %1361 to float, !dbg !67
  %1363 = fadd float %1350, %1362, !dbg !67
  %1364 = bitcast float %1363 to i32, !dbg !67
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1364), !dbg !67
  %1366 = bitcast i32 %1365 to float, !dbg !67
  %1367 = fadd float %1363, %1366, !dbg !67
  %1368 = fadd float %1367, %1140, !dbg !67
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !53
  fence syncscope("workgroup") release, !dbg !53
  call void @llvm.amdgcn.s.barrier(), !dbg !53
  fence syncscope("workgroup") acquire, !dbg !53
  %address.local836 = shl nuw nsw i32 %address.thread798, 3, !dbg !49
  %address.local837 = and i32 %address.local836, 504, !dbg !49
  %address.local838 = zext nneg i32 %address.local837 to i64, !dbg !49
  %address.local839 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !49
  %1369 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local839, i32 0) #0, !dbg !49
  %1370 = bitcast <4 x i32> %1369 to <8 x half>, !dbg !49
  %address.local840 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !49
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local840, i64 1024, !dbg !49
  %1371 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #0, !dbg !49
  %1372 = bitcast <4 x i32> %1371 to <8 x half>, !dbg !49
  %address.local842 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !49
  %address.local843 = getelementptr i8, ptr addrspace(3) %address.local842, i64 2048, !dbg !49
  %1373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local843, i32 0) #0, !dbg !49
  %1374 = bitcast <4 x i32> %1373 to <8 x half>, !dbg !49
  %address.local844 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !49
  %address.local845 = getelementptr i8, ptr addrspace(3) %address.local844, i64 3072, !dbg !49
  %1375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local845, i32 0) #0, !dbg !49
  %1376 = bitcast <4 x i32> %1375 to <8 x half>, !dbg !49
  %address.local846 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !49
  %address.local847 = getelementptr i8, ptr addrspace(3) %address.local846, i64 4096, !dbg !49
  %1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local847, i32 0) #0, !dbg !49
  %1378 = bitcast <4 x i32> %1377 to <8 x half>, !dbg !49
  %address.local848 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !49
  %address.local849 = getelementptr i8, ptr addrspace(3) %address.local848, i64 5120, !dbg !49
  %1379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local849, i32 0) #0, !dbg !49
  %1380 = bitcast <4 x i32> %1379 to <8 x half>, !dbg !49
  %address.local850 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !49
  %address.local851 = getelementptr i8, ptr addrspace(3) %address.local850, i64 6144, !dbg !49
  %1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #0, !dbg !49
  %1382 = bitcast <4 x i32> %1381 to <8 x half>, !dbg !49
  %address.local852 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !49
  %address.local853 = getelementptr i8, ptr addrspace(3) %address.local852, i64 7168, !dbg !49
  %1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local853, i32 0) #0, !dbg !49
  %1384 = bitcast <4 x i32> %1383 to <8 x half>, !dbg !49
  %1385 = shufflevector <8 x half> %1370, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1385, <4 x float> zeroinitializer) #9, !dbg !49
  %1387 = shufflevector <8 x half> %1372, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1387, <4 x float> %1386) #9, !dbg !49
  %1389 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1389, <4 x float> %1388) #9, !dbg !49
  %1391 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1391, <4 x float> %1390) #9, !dbg !49
  %1393 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1393, <4 x float> %1392) #9, !dbg !49
  %1395 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1395, <4 x float> %1394) #9, !dbg !49
  %1397 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1397, <4 x float> %1396) #9, !dbg !49
  %1399 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1399, <4 x float> %1398) #9, !dbg !49
  %1401 = extractelement <4 x float> %1400, i64 0, !dbg !49
  %1402 = extractelement <4 x float> %1400, i64 1, !dbg !49
  %1403 = extractelement <4 x float> %1400, i64 2, !dbg !49
  %1404 = extractelement <4 x float> %1400, i64 3, !dbg !49
  %1405 = select i1 %.not, float %1402, float %1401, !dbg !49
  %1406 = bitcast float %1405 to i32, !dbg !49
  %1407 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1406), !dbg !49
  %1408 = bitcast i32 %1407 to float, !dbg !49
  %1409 = select i1 %.not, float %1401, float %1408, !dbg !49
  %1410 = select i1 %.not, float %1408, float %1402, !dbg !49
  %1411 = select i1 %.not, float %1404, float %1403, !dbg !49
  %1412 = bitcast float %1411 to i32, !dbg !49
  %1413 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1412), !dbg !49
  %1414 = bitcast i32 %1413 to float, !dbg !49
  %1415 = select i1 %.not, float %1403, float %1414, !dbg !49
  %1416 = select i1 %.not, float %1414, float %1404, !dbg !49
  %1417 = select i1 %.not1, float %1415, float %1409, !dbg !49
  %1418 = bitcast float %1417 to i32, !dbg !49
  %1419 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1418), !dbg !49
  %1420 = bitcast i32 %1419 to float, !dbg !49
  %1421 = select i1 %.not1, float %1409, float %1420, !dbg !49
  %1422 = select i1 %.not1, float %1420, float %1415, !dbg !49
  %1423 = select i1 %.not1, float %1416, float %1410, !dbg !49
  %1424 = bitcast float %1423 to i32, !dbg !49
  %1425 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1424), !dbg !49
  %1426 = bitcast i32 %1425 to float, !dbg !49
  %1427 = select i1 %.not1, float %1410, float %1426, !dbg !49
  %1428 = select i1 %.not1, float %1426, float %1416, !dbg !49
  %1429 = shufflevector <8 x half> %1370, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1429, <4 x float> zeroinitializer) #9, !dbg !49
  %1431 = shufflevector <8 x half> %1372, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1431, <4 x float> %1430) #9, !dbg !49
  %1433 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1433, <4 x float> %1432) #9, !dbg !49
  %1435 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1435, <4 x float> %1434) #9, !dbg !49
  %1437 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1437, <4 x float> %1436) #9, !dbg !49
  %1439 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1439, <4 x float> %1438) #9, !dbg !49
  %1441 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1441, <4 x float> %1440) #9, !dbg !49
  %1443 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1443, <4 x float> %1442) #9, !dbg !49
  %1445 = extractelement <4 x float> %1444, i64 0, !dbg !49
  %1446 = extractelement <4 x float> %1444, i64 1, !dbg !49
  %1447 = extractelement <4 x float> %1444, i64 2, !dbg !49
  %1448 = extractelement <4 x float> %1444, i64 3, !dbg !49
  %1449 = select i1 %.not, float %1446, float %1445, !dbg !49
  %1450 = bitcast float %1449 to i32, !dbg !49
  %1451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1450), !dbg !49
  %1452 = bitcast i32 %1451 to float, !dbg !49
  %1453 = select i1 %.not, float %1445, float %1452, !dbg !49
  %1454 = select i1 %.not, float %1452, float %1446, !dbg !49
  %1455 = select i1 %.not, float %1448, float %1447, !dbg !49
  %1456 = bitcast float %1455 to i32, !dbg !49
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1456), !dbg !49
  %1458 = bitcast i32 %1457 to float, !dbg !49
  %1459 = select i1 %.not, float %1447, float %1458, !dbg !49
  %1460 = select i1 %.not, float %1458, float %1448, !dbg !49
  %1461 = select i1 %.not1, float %1459, float %1453, !dbg !49
  %1462 = bitcast float %1461 to i32, !dbg !49
  %1463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1462), !dbg !49
  %1464 = bitcast i32 %1463 to float, !dbg !49
  %1465 = select i1 %.not1, float %1453, float %1464, !dbg !49
  %1466 = select i1 %.not1, float %1464, float %1459, !dbg !49
  %1467 = select i1 %.not1, float %1460, float %1454, !dbg !49
  %1468 = bitcast float %1467 to i32, !dbg !49
  %1469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1468), !dbg !49
  %1470 = bitcast i32 %1469 to float, !dbg !49
  %1471 = select i1 %.not1, float %1454, float %1470, !dbg !49
  %1472 = select i1 %.not1, float %1470, float %1460, !dbg !49
  %1473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1385, <4 x float> zeroinitializer) #9, !dbg !49
  %1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1387, <4 x float> %1473) #9, !dbg !49
  %1475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1389, <4 x float> %1474) #9, !dbg !49
  %1476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1391, <4 x float> %1475) #9, !dbg !49
  %1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1393, <4 x float> %1476) #9, !dbg !49
  %1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1395, <4 x float> %1477) #9, !dbg !49
  %1479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1397, <4 x float> %1478) #9, !dbg !49
  %1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1399, <4 x float> %1479) #9, !dbg !49
  %1481 = extractelement <4 x float> %1480, i64 0, !dbg !49
  %1482 = extractelement <4 x float> %1480, i64 1, !dbg !49
  %1483 = extractelement <4 x float> %1480, i64 2, !dbg !49
  %1484 = extractelement <4 x float> %1480, i64 3, !dbg !49
  %1485 = select i1 %.not, float %1482, float %1481, !dbg !49
  %1486 = bitcast float %1485 to i32, !dbg !49
  %1487 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1486), !dbg !49
  %1488 = bitcast i32 %1487 to float, !dbg !49
  %1489 = select i1 %.not, float %1481, float %1488, !dbg !49
  %1490 = select i1 %.not, float %1488, float %1482, !dbg !49
  %1491 = select i1 %.not, float %1484, float %1483, !dbg !49
  %1492 = bitcast float %1491 to i32, !dbg !49
  %1493 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1492), !dbg !49
  %1494 = bitcast i32 %1493 to float, !dbg !49
  %1495 = select i1 %.not, float %1483, float %1494, !dbg !49
  %1496 = select i1 %.not, float %1494, float %1484, !dbg !49
  %1497 = select i1 %.not1, float %1495, float %1489, !dbg !49
  %1498 = bitcast float %1497 to i32, !dbg !49
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1498), !dbg !49
  %1500 = bitcast i32 %1499 to float, !dbg !49
  %1501 = select i1 %.not1, float %1489, float %1500, !dbg !49
  %1502 = select i1 %.not1, float %1500, float %1495, !dbg !49
  %1503 = select i1 %.not1, float %1496, float %1490, !dbg !49
  %1504 = bitcast float %1503 to i32, !dbg !49
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1504), !dbg !49
  %1506 = bitcast i32 %1505 to float, !dbg !49
  %1507 = select i1 %.not1, float %1490, float %1506, !dbg !49
  %1508 = select i1 %.not1, float %1506, float %1496, !dbg !49
  %1509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1429, <4 x float> zeroinitializer) #9, !dbg !49
  %1510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1431, <4 x float> %1509) #9, !dbg !49
  %1511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1433, <4 x float> %1510) #9, !dbg !49
  %1512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1435, <4 x float> %1511) #9, !dbg !49
  %1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1437, <4 x float> %1512) #9, !dbg !49
  %1514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1439, <4 x float> %1513) #9, !dbg !49
  %1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1441, <4 x float> %1514) #9, !dbg !49
  %1516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1443, <4 x float> %1515) #9, !dbg !49
  %1517 = extractelement <4 x float> %1516, i64 0, !dbg !49
  %1518 = extractelement <4 x float> %1516, i64 1, !dbg !49
  %1519 = extractelement <4 x float> %1516, i64 2, !dbg !49
  %1520 = extractelement <4 x float> %1516, i64 3, !dbg !49
  %1521 = select i1 %.not, float %1518, float %1517, !dbg !49
  %1522 = bitcast float %1521 to i32, !dbg !49
  %1523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1522), !dbg !49
  %1524 = bitcast i32 %1523 to float, !dbg !49
  %1525 = select i1 %.not, float %1517, float %1524, !dbg !49
  %1526 = select i1 %.not, float %1524, float %1518, !dbg !49
  %1527 = select i1 %.not, float %1520, float %1519, !dbg !49
  %1528 = bitcast float %1527 to i32, !dbg !49
  %1529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1528), !dbg !49
  %1530 = bitcast i32 %1529 to float, !dbg !49
  %1531 = select i1 %.not, float %1519, float %1530, !dbg !49
  %1532 = select i1 %.not, float %1530, float %1520, !dbg !49
  %1533 = select i1 %.not1, float %1531, float %1525, !dbg !49
  %1534 = bitcast float %1533 to i32, !dbg !49
  %1535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1534), !dbg !49
  %1536 = bitcast i32 %1535 to float, !dbg !49
  %1537 = select i1 %.not1, float %1525, float %1536, !dbg !49
  %1538 = select i1 %.not1, float %1536, float %1531, !dbg !49
  %1539 = select i1 %.not1, float %1532, float %1526, !dbg !49
  %1540 = bitcast float %1539 to i32, !dbg !49
  %1541 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1540), !dbg !49
  %1542 = bitcast i32 %1541 to float, !dbg !49
  %1543 = select i1 %.not1, float %1526, float %1542, !dbg !49
  %1544 = select i1 %.not1, float %1542, float %1532, !dbg !49
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  br i1 %43, label %1545, label %1546, !dbg !69

1545:                                             ; preds = %1113
  %address.thread854 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !69, !range !73, !frisk.register_lifetimes !37
  %address.local855 = lshr i32 %address.thread854, 1, !dbg !48
  %address.local856 = and i32 %address.local855, 96, !dbg !48
  %address.local857 = and i32 %address.thread854, 15, !dbg !48
  %address.local858 = or disjoint i32 %address.local856, %address.local857, !dbg !48
  %address.local859 = zext nneg i32 %address.local858 to i64, !dbg !48
  %address.local860 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !48
  store float %1110, ptr addrspace(3) %address.local860, align 4, !dbg !69
  %address.local861 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !48
  %address.local862 = getelementptr i8, ptr addrspace(3) %address.local861, i64 64, !dbg !48
  store float %1111, ptr addrspace(3) %address.local862, align 4, !dbg !69
  br label %1546, !dbg !69

1546:                                             ; preds = %1545, %1113
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  %address.thread863 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !69, !range !73, !frisk.register_lifetimes !37
  %1547 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !69
  %lds.const.gep1748 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !69
  %1548 = load float, ptr addrspace(3) %lds.const.gep1748, align 4, !dbg !69
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  %1549 = add nuw nsw i64 %76, %15, !dbg !53
  %1550 = shl i64 %1549, 19, !dbg !53
  %1551 = and i64 %1550, 16252928, !dbg !53
  %1552 = add i64 %1551, %82, !dbg !53
  %address.local877 = zext nneg i32 %address.thread863 to i64, !dbg !45
  %address.local878 = shl nuw nsw i64 %address.local877, 3, !dbg !44
  %address.local879 = and i64 %address.local878, 1536, !dbg !44
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local879, !dbg !44
  %address.local881 = ptrtoint ptr addrspace(3) %address.local880 to i64, !dbg !44
  %address.local882 = trunc i64 %address.local881 to i32, !dbg !44
  %1553 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local882), !dbg !53
  %address.local883 = shl nuw nsw i64 %address.local877, 6, !dbg !53
  %address.local884 = and i64 %address.local883, 16128, !dbg !53
  %address.local885 = shl nuw nsw i64 %address.local877, 4, !dbg !53
  %address.local886 = and i64 %address.local885, 48, !dbg !53
  %1554 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !53
  %.idx = shl i64 %1552, 1, !dbg !53
  %1555 = or disjoint i64 %address.local884, %address.local886, !dbg !53
  %1556 = or disjoint i64 %.idx, %1555, !dbg !53
  %1557 = trunc i64 %1556 to i32, !dbg !53
  %dma.buffer.offset1767 = or i32 %1557, 64, !dbg !53
  %1558 = insertelement <2 x i64> poison, i64 %1554, i64 0, !dbg !53
  %1559 = bitcast <2 x i64> %1558 to <4 x i32>, !dbg !53
  %dma.buffer.rsrc1768 = shufflevector <4 x i32> %1559, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !53
  %dma.buffer.lds1769 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1553), !dbg !53
  %1560 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1769, i32 %dma.buffer.offset1767, <4 x i32> %dma.buffer.rsrc1768) #10, !dbg !53
  %address.local889 = or disjoint i64 %address.local877, 256, !dbg !44
  %address.local890 = shl nuw nsw i64 %address.local889, 3, !dbg !44
  %address.local891 = and i64 %address.local890, 3584, !dbg !44
  %address.local892 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local891, !dbg !44
  %address.local893 = ptrtoint ptr addrspace(3) %address.local892 to i64, !dbg !44
  %address.local894 = trunc i64 %address.local893 to i32, !dbg !44
  %1561 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !53
  %address.local895 = shl nuw nsw i64 %address.local889, 6, !dbg !53
  %address.local896 = and i64 %address.local895, 32512, !dbg !53
  %address.local897 = shl nuw nsw i64 %address.local877, 4, !dbg !53
  %address.local898 = and i64 %address.local897, 48, !dbg !53
  %1562 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !53
  %.idx53 = shl i64 %1552, 1, !dbg !53
  %1563 = or disjoint i64 %.idx53, %address.local896, !dbg !53
  %1564 = or disjoint i64 %1563, %address.local898, !dbg !53
  %1565 = trunc i64 %1564 to i32, !dbg !53
  %dma.buffer.offset1764 = or i32 %1565, 64, !dbg !53
  %1566 = insertelement <2 x i64> poison, i64 %1562, i64 0, !dbg !53
  %1567 = bitcast <2 x i64> %1566 to <4 x i32>, !dbg !53
  %dma.buffer.rsrc1765 = shufflevector <4 x i32> %1567, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !53
  %dma.buffer.lds1766 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1561), !dbg !53
  %1568 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1766, i32 %dma.buffer.offset1764, <4 x i32> %dma.buffer.rsrc1765) #10, !dbg !53
  %1569 = extractvalue [2 x <8 x float>] %55, 0, !dbg !69
  %1570 = shufflevector <8 x float> %1569, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !69
  %1571 = insertelement <4 x float> poison, float %1547, i64 0, !dbg !69
  %1572 = shufflevector <4 x float> %1571, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !69
  %1573 = fmul <4 x float> %1570, %1572, !dbg !69
  %1574 = insertelement <4 x float> poison, float %1421, i64 0, !dbg !70
  %1575 = insertelement <4 x float> %1574, float %1427, i64 1, !dbg !70
  %1576 = insertelement <4 x float> %1575, float %1422, i64 2, !dbg !70
  %1577 = insertelement <4 x float> %1576, float %1428, i64 3, !dbg !70
  %1578 = fadd <4 x float> %1573, %1577, !dbg !70
  %1579 = shufflevector <8 x float> %1569, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !69
  %1580 = fmul <4 x float> %1579, %1572, !dbg !69
  %1581 = insertelement <4 x float> poison, float %1465, i64 0, !dbg !70
  %1582 = insertelement <4 x float> %1581, float %1471, i64 1, !dbg !70
  %1583 = insertelement <4 x float> %1582, float %1466, i64 2, !dbg !70
  %1584 = insertelement <4 x float> %1583, float %1472, i64 3, !dbg !70
  %1585 = fadd <4 x float> %1580, %1584, !dbg !70
  %1586 = extractvalue [2 x <8 x float>] %55, 1, !dbg !69
  %1587 = shufflevector <8 x float> %1586, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !69
  %1588 = insertelement <4 x float> poison, float %1548, i64 0, !dbg !69
  %1589 = shufflevector <4 x float> %1588, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !69
  %1590 = fmul <4 x float> %1587, %1589, !dbg !69
  %1591 = insertelement <4 x float> poison, float %1501, i64 0, !dbg !70
  %1592 = insertelement <4 x float> %1591, float %1507, i64 1, !dbg !70
  %1593 = insertelement <4 x float> %1592, float %1502, i64 2, !dbg !70
  %1594 = insertelement <4 x float> %1593, float %1508, i64 3, !dbg !70
  %1595 = fadd <4 x float> %1590, %1594, !dbg !70
  %1596 = shufflevector <8 x float> %1586, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !69
  %1597 = fmul <4 x float> %1596, %1589, !dbg !69
  %1598 = insertelement <4 x float> poison, float %1537, i64 0, !dbg !70
  %1599 = insertelement <4 x float> %1598, float %1543, i64 1, !dbg !70
  %1600 = insertelement <4 x float> %1599, float %1538, i64 2, !dbg !70
  %1601 = insertelement <4 x float> %1600, float %1544, i64 3, !dbg !70
  %1602 = fadd <4 x float> %1597, %1601, !dbg !70
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !53
  fence syncscope("workgroup") release, !dbg !53
  call void @llvm.amdgcn.s.barrier(), !dbg !53
  fence syncscope("workgroup") acquire, !dbg !53
  %address.local901 = shl nuw nsw i32 %address.thread863, 3, !dbg !49
  %address.local902 = and i32 %address.local901, 504, !dbg !49
  %address.local903 = zext nneg i32 %address.local902 to i64, !dbg !49
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !49
  %1603 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #0, !dbg !49
  %1604 = bitcast <4 x i32> %1603 to <8 x half>, !dbg !49
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !49
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 1024, !dbg !49
  %1605 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !49
  %1606 = bitcast <4 x i32> %1605 to <8 x half>, !dbg !49
  %address.local907 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !49
  %address.local908 = getelementptr i8, ptr addrspace(3) %address.local907, i64 2048, !dbg !49
  %1607 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local908, i32 0) #0, !dbg !49
  %1608 = bitcast <4 x i32> %1607 to <8 x half>, !dbg !49
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !49
  %address.local910 = getelementptr i8, ptr addrspace(3) %address.local909, i64 3072, !dbg !49
  %1609 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local910, i32 0) #0, !dbg !49
  %1610 = bitcast <4 x i32> %1609 to <8 x half>, !dbg !49
  %address.local911 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !49
  %address.local912 = getelementptr i8, ptr addrspace(3) %address.local911, i64 4096, !dbg !49
  %1611 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local912, i32 0) #0, !dbg !49
  %1612 = bitcast <4 x i32> %1611 to <8 x half>, !dbg !49
  %address.local913 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !49
  %address.local914 = getelementptr i8, ptr addrspace(3) %address.local913, i64 5120, !dbg !49
  %1613 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local914, i32 0) #0, !dbg !49
  %1614 = bitcast <4 x i32> %1613 to <8 x half>, !dbg !49
  %address.local915 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !49
  %address.local916 = getelementptr i8, ptr addrspace(3) %address.local915, i64 6144, !dbg !49
  %1615 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local916, i32 0) #0, !dbg !49
  %1616 = bitcast <4 x i32> %1615 to <8 x half>, !dbg !49
  %address.local917 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !49
  %address.local918 = getelementptr i8, ptr addrspace(3) %address.local917, i64 7168, !dbg !49
  %1617 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local918, i32 0) #0, !dbg !49
  %1618 = bitcast <4 x i32> %1617 to <8 x half>, !dbg !49
  %1619 = shufflevector <8 x half> %1604, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1619, <4 x float> zeroinitializer) #9, !dbg !49
  %1621 = shufflevector <8 x half> %1606, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1621, <4 x float> %1620) #9, !dbg !49
  %1623 = shufflevector <8 x half> %1608, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1623, <4 x float> %1622) #9, !dbg !49
  %1625 = shufflevector <8 x half> %1610, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1625, <4 x float> %1624) #9, !dbg !49
  %1627 = shufflevector <8 x half> %1612, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1627, <4 x float> %1626) #9, !dbg !49
  %1629 = shufflevector <8 x half> %1614, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1629, <4 x float> %1628) #9, !dbg !49
  %1631 = shufflevector <8 x half> %1616, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1631, <4 x float> %1630) #9, !dbg !49
  %1633 = shufflevector <8 x half> %1618, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1633, <4 x float> %1632) #9, !dbg !49
  %1635 = extractelement <4 x float> %1634, i64 0, !dbg !49
  %1636 = extractelement <4 x float> %1634, i64 1, !dbg !49
  %1637 = extractelement <4 x float> %1634, i64 2, !dbg !49
  %1638 = extractelement <4 x float> %1634, i64 3, !dbg !49
  %1639 = select i1 %.not, float %1636, float %1635, !dbg !49
  %1640 = bitcast float %1639 to i32, !dbg !49
  %1641 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1640), !dbg !49
  %1642 = bitcast i32 %1641 to float, !dbg !49
  %1643 = select i1 %.not, float %1635, float %1642, !dbg !49
  %1644 = select i1 %.not, float %1642, float %1636, !dbg !49
  %1645 = select i1 %.not, float %1638, float %1637, !dbg !49
  %1646 = bitcast float %1645 to i32, !dbg !49
  %1647 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1646), !dbg !49
  %1648 = bitcast i32 %1647 to float, !dbg !49
  %1649 = select i1 %.not, float %1637, float %1648, !dbg !49
  %1650 = select i1 %.not, float %1648, float %1638, !dbg !49
  %1651 = select i1 %.not1, float %1649, float %1643, !dbg !49
  %1652 = bitcast float %1651 to i32, !dbg !49
  %1653 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1652), !dbg !49
  %1654 = bitcast i32 %1653 to float, !dbg !49
  %1655 = select i1 %.not1, float %1643, float %1654, !dbg !49
  %1656 = select i1 %.not1, float %1654, float %1649, !dbg !49
  %1657 = select i1 %.not1, float %1650, float %1644, !dbg !49
  %1658 = bitcast float %1657 to i32, !dbg !49
  %1659 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1658), !dbg !49
  %1660 = bitcast i32 %1659 to float, !dbg !49
  %1661 = select i1 %.not1, float %1644, float %1660, !dbg !49
  %1662 = select i1 %.not1, float %1660, float %1650, !dbg !49
  %1663 = shufflevector <8 x half> %1604, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1663, <4 x float> zeroinitializer) #9, !dbg !49
  %1665 = shufflevector <8 x half> %1606, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1665, <4 x float> %1664) #9, !dbg !49
  %1667 = shufflevector <8 x half> %1608, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1667, <4 x float> %1666) #9, !dbg !49
  %1669 = shufflevector <8 x half> %1610, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1669, <4 x float> %1668) #9, !dbg !49
  %1671 = shufflevector <8 x half> %1612, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1671, <4 x float> %1670) #9, !dbg !49
  %1673 = shufflevector <8 x half> %1614, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1673, <4 x float> %1672) #9, !dbg !49
  %1675 = shufflevector <8 x half> %1616, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1675, <4 x float> %1674) #9, !dbg !49
  %1677 = shufflevector <8 x half> %1618, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1677, <4 x float> %1676) #9, !dbg !49
  %1679 = extractelement <4 x float> %1678, i64 0, !dbg !49
  %1680 = extractelement <4 x float> %1678, i64 1, !dbg !49
  %1681 = extractelement <4 x float> %1678, i64 2, !dbg !49
  %1682 = extractelement <4 x float> %1678, i64 3, !dbg !49
  %1683 = select i1 %.not, float %1680, float %1679, !dbg !49
  %1684 = bitcast float %1683 to i32, !dbg !49
  %1685 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1684), !dbg !49
  %1686 = bitcast i32 %1685 to float, !dbg !49
  %1687 = select i1 %.not, float %1679, float %1686, !dbg !49
  %1688 = select i1 %.not, float %1686, float %1680, !dbg !49
  %1689 = select i1 %.not, float %1682, float %1681, !dbg !49
  %1690 = bitcast float %1689 to i32, !dbg !49
  %1691 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1690), !dbg !49
  %1692 = bitcast i32 %1691 to float, !dbg !49
  %1693 = select i1 %.not, float %1681, float %1692, !dbg !49
  %1694 = select i1 %.not, float %1692, float %1682, !dbg !49
  %1695 = select i1 %.not1, float %1693, float %1687, !dbg !49
  %1696 = bitcast float %1695 to i32, !dbg !49
  %1697 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1696), !dbg !49
  %1698 = bitcast i32 %1697 to float, !dbg !49
  %1699 = select i1 %.not1, float %1687, float %1698, !dbg !49
  %1700 = select i1 %.not1, float %1698, float %1693, !dbg !49
  %1701 = select i1 %.not1, float %1694, float %1688, !dbg !49
  %1702 = bitcast float %1701 to i32, !dbg !49
  %1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1702), !dbg !49
  %1704 = bitcast i32 %1703 to float, !dbg !49
  %1705 = select i1 %.not1, float %1688, float %1704, !dbg !49
  %1706 = select i1 %.not1, float %1704, float %1694, !dbg !49
  %1707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1619, <4 x float> zeroinitializer) #9, !dbg !49
  %1708 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1621, <4 x float> %1707) #9, !dbg !49
  %1709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1623, <4 x float> %1708) #9, !dbg !49
  %1710 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1625, <4 x float> %1709) #9, !dbg !49
  %1711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1627, <4 x float> %1710) #9, !dbg !49
  %1712 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1629, <4 x float> %1711) #9, !dbg !49
  %1713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1631, <4 x float> %1712) #9, !dbg !49
  %1714 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1633, <4 x float> %1713) #9, !dbg !49
  %1715 = extractelement <4 x float> %1714, i64 0, !dbg !49
  %1716 = extractelement <4 x float> %1714, i64 1, !dbg !49
  %1717 = extractelement <4 x float> %1714, i64 2, !dbg !49
  %1718 = extractelement <4 x float> %1714, i64 3, !dbg !49
  %1719 = select i1 %.not, float %1716, float %1715, !dbg !49
  %1720 = bitcast float %1719 to i32, !dbg !49
  %1721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1720), !dbg !49
  %1722 = bitcast i32 %1721 to float, !dbg !49
  %1723 = select i1 %.not, float %1715, float %1722, !dbg !49
  %1724 = select i1 %.not, float %1722, float %1716, !dbg !49
  %1725 = select i1 %.not, float %1718, float %1717, !dbg !49
  %1726 = bitcast float %1725 to i32, !dbg !49
  %1727 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1726), !dbg !49
  %1728 = bitcast i32 %1727 to float, !dbg !49
  %1729 = select i1 %.not, float %1717, float %1728, !dbg !49
  %1730 = select i1 %.not, float %1728, float %1718, !dbg !49
  %1731 = select i1 %.not1, float %1729, float %1723, !dbg !49
  %1732 = bitcast float %1731 to i32, !dbg !49
  %1733 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1732), !dbg !49
  %1734 = bitcast i32 %1733 to float, !dbg !49
  %1735 = select i1 %.not1, float %1723, float %1734, !dbg !49
  %1736 = select i1 %.not1, float %1734, float %1729, !dbg !49
  %1737 = select i1 %.not1, float %1730, float %1724, !dbg !49
  %1738 = bitcast float %1737 to i32, !dbg !49
  %1739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1738), !dbg !49
  %1740 = bitcast i32 %1739 to float, !dbg !49
  %1741 = select i1 %.not1, float %1724, float %1740, !dbg !49
  %1742 = select i1 %.not1, float %1740, float %1730, !dbg !49
  %1743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1663, <4 x float> zeroinitializer) #9, !dbg !49
  %1744 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1665, <4 x float> %1743) #9, !dbg !49
  %1745 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1667, <4 x float> %1744) #9, !dbg !49
  %1746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1669, <4 x float> %1745) #9, !dbg !49
  %1747 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1671, <4 x float> %1746) #9, !dbg !49
  %1748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1673, <4 x float> %1747) #9, !dbg !49
  %1749 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1675, <4 x float> %1748) #9, !dbg !49
  %1750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1677, <4 x float> %1749) #9, !dbg !49
  %1751 = extractelement <4 x float> %1750, i64 0, !dbg !49
  %1752 = extractelement <4 x float> %1750, i64 1, !dbg !49
  %1753 = extractelement <4 x float> %1750, i64 2, !dbg !49
  %1754 = extractelement <4 x float> %1750, i64 3, !dbg !49
  %1755 = select i1 %.not, float %1752, float %1751, !dbg !49
  %1756 = bitcast float %1755 to i32, !dbg !49
  %1757 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1756), !dbg !49
  %1758 = bitcast i32 %1757 to float, !dbg !49
  %1759 = select i1 %.not, float %1751, float %1758, !dbg !49
  %1760 = select i1 %.not, float %1758, float %1752, !dbg !49
  %1761 = select i1 %.not, float %1754, float %1753, !dbg !49
  %1762 = bitcast float %1761 to i32, !dbg !49
  %1763 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1762), !dbg !49
  %1764 = bitcast i32 %1763 to float, !dbg !49
  %1765 = select i1 %.not, float %1753, float %1764, !dbg !49
  %1766 = select i1 %.not, float %1764, float %1754, !dbg !49
  %1767 = select i1 %.not1, float %1765, float %1759, !dbg !49
  %1768 = bitcast float %1767 to i32, !dbg !49
  %1769 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1768), !dbg !49
  %1770 = bitcast i32 %1769 to float, !dbg !49
  %1771 = select i1 %.not1, float %1759, float %1770, !dbg !49
  %1772 = select i1 %.not1, float %1770, float %1765, !dbg !49
  %1773 = select i1 %.not1, float %1766, float %1760, !dbg !49
  %1774 = bitcast float %1773 to i32, !dbg !49
  %1775 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1774), !dbg !49
  %1776 = bitcast i32 %1775 to float, !dbg !49
  %1777 = select i1 %.not1, float %1760, float %1776, !dbg !49
  %1778 = select i1 %.not1, float %1776, float %1766, !dbg !49
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  br i1 %43, label %1779, label %1780, !dbg !69

1779:                                             ; preds = %1546
  %address.thread919 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !69, !range !73, !frisk.register_lifetimes !37
  %address.local920 = lshr i32 %address.thread919, 1, !dbg !48
  %address.local921 = and i32 %address.local920, 96, !dbg !48
  %address.local922 = and i32 %address.thread919, 15, !dbg !48
  %address.local923 = or disjoint i32 %address.local921, %address.local922, !dbg !48
  %address.local924 = zext nneg i32 %address.local923 to i64, !dbg !48
  %address.local925 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !48
  store float %1110, ptr addrspace(3) %address.local925, align 4, !dbg !69
  %address.local926 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !48
  %address.local927 = getelementptr i8, ptr addrspace(3) %address.local926, i64 64, !dbg !48
  store float %1111, ptr addrspace(3) %address.local927, align 4, !dbg !69
  br label %1780, !dbg !69

1780:                                             ; preds = %1779, %1546
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  %address.thread928 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !69, !range !73, !frisk.register_lifetimes !37
  %1781 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !69
  %lds.const.gep1750 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !69
  %1782 = load float, ptr addrspace(3) %lds.const.gep1750, align 4, !dbg !69
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  %address.local942 = zext nneg i32 %address.thread928 to i64, !dbg !45
  %address.local943 = shl nuw nsw i64 %address.local942, 3, !dbg !44
  %address.local944 = and i64 %address.local943, 1536, !dbg !44
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local944, !dbg !44
  %address.local946 = ptrtoint ptr addrspace(3) %address.local945 to i64, !dbg !44
  %address.local947 = trunc i64 %address.local946 to i32, !dbg !44
  %1783 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local947), !dbg !53
  %address.local948 = shl nuw nsw i64 %address.local942, 6, !dbg !53
  %address.local949 = and i64 %address.local948, 16128, !dbg !53
  %address.local950 = shl nuw nsw i64 %address.local942, 4, !dbg !53
  %address.local951 = and i64 %address.local950, 48, !dbg !53
  %1784 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !53
  %.idx55 = shl i64 %1552, 1, !dbg !53
  %1785 = or disjoint i64 %address.local949, %address.local951, !dbg !53
  %1786 = or disjoint i64 %.idx55, %1785, !dbg !53
  %1787 = trunc i64 %1786 to i32, !dbg !53
  %dma.buffer.offset1761 = or i32 %1787, 128, !dbg !53
  %1788 = insertelement <2 x i64> poison, i64 %1784, i64 0, !dbg !53
  %1789 = bitcast <2 x i64> %1788 to <4 x i32>, !dbg !53
  %dma.buffer.rsrc1762 = shufflevector <4 x i32> %1789, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !53
  %dma.buffer.lds1763 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1783), !dbg !53
  %1790 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1763, i32 %dma.buffer.offset1761, <4 x i32> %dma.buffer.rsrc1762) #10, !dbg !53
  %address.local954 = or disjoint i64 %address.local942, 256, !dbg !44
  %address.local955 = shl nuw nsw i64 %address.local954, 3, !dbg !44
  %address.local956 = and i64 %address.local955, 3584, !dbg !44
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local956, !dbg !44
  %address.local958 = ptrtoint ptr addrspace(3) %address.local957 to i64, !dbg !44
  %address.local959 = trunc i64 %address.local958 to i32, !dbg !44
  %1791 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local959), !dbg !53
  %address.local960 = shl nuw nsw i64 %address.local954, 6, !dbg !53
  %address.local961 = and i64 %address.local960, 32512, !dbg !53
  %address.local962 = shl nuw nsw i64 %address.local942, 4, !dbg !53
  %address.local963 = and i64 %address.local962, 48, !dbg !53
  %1792 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !53
  %.idx57 = shl i64 %1552, 1, !dbg !53
  %1793 = or disjoint i64 %.idx57, %address.local961, !dbg !53
  %1794 = or disjoint i64 %1793, %address.local963, !dbg !53
  %1795 = trunc i64 %1794 to i32, !dbg !53
  %dma.buffer.offset1758 = or i32 %1795, 128, !dbg !53
  %1796 = insertelement <2 x i64> poison, i64 %1792, i64 0, !dbg !53
  %1797 = bitcast <2 x i64> %1796 to <4 x i32>, !dbg !53
  %dma.buffer.rsrc1759 = shufflevector <4 x i32> %1797, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !53
  %dma.buffer.lds1760 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1791), !dbg !53
  %1798 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1760, i32 %dma.buffer.offset1758, <4 x i32> %dma.buffer.rsrc1759) #10, !dbg !53
  %1799 = extractvalue [2 x <8 x float>] %58, 0, !dbg !69
  %1800 = shufflevector <8 x float> %1799, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !69
  %1801 = insertelement <4 x float> poison, float %1781, i64 0, !dbg !69
  %1802 = shufflevector <4 x float> %1801, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !69
  %1803 = fmul <4 x float> %1800, %1802, !dbg !69
  %1804 = insertelement <4 x float> poison, float %1655, i64 0, !dbg !70
  %1805 = insertelement <4 x float> %1804, float %1661, i64 1, !dbg !70
  %1806 = insertelement <4 x float> %1805, float %1656, i64 2, !dbg !70
  %1807 = insertelement <4 x float> %1806, float %1662, i64 3, !dbg !70
  %1808 = fadd <4 x float> %1803, %1807, !dbg !70
  %1809 = shufflevector <8 x float> %1799, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !69
  %1810 = fmul <4 x float> %1809, %1802, !dbg !69
  %1811 = insertelement <4 x float> poison, float %1699, i64 0, !dbg !70
  %1812 = insertelement <4 x float> %1811, float %1705, i64 1, !dbg !70
  %1813 = insertelement <4 x float> %1812, float %1700, i64 2, !dbg !70
  %1814 = insertelement <4 x float> %1813, float %1706, i64 3, !dbg !70
  %1815 = fadd <4 x float> %1810, %1814, !dbg !70
  %1816 = extractvalue [2 x <8 x float>] %58, 1, !dbg !69
  %1817 = shufflevector <8 x float> %1816, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !69
  %1818 = insertelement <4 x float> poison, float %1782, i64 0, !dbg !69
  %1819 = shufflevector <4 x float> %1818, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !69
  %1820 = fmul <4 x float> %1817, %1819, !dbg !69
  %1821 = insertelement <4 x float> poison, float %1735, i64 0, !dbg !70
  %1822 = insertelement <4 x float> %1821, float %1741, i64 1, !dbg !70
  %1823 = insertelement <4 x float> %1822, float %1736, i64 2, !dbg !70
  %1824 = insertelement <4 x float> %1823, float %1742, i64 3, !dbg !70
  %1825 = fadd <4 x float> %1820, %1824, !dbg !70
  %1826 = shufflevector <8 x float> %1816, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !69
  %1827 = fmul <4 x float> %1826, %1819, !dbg !69
  %1828 = insertelement <4 x float> poison, float %1771, i64 0, !dbg !70
  %1829 = insertelement <4 x float> %1828, float %1777, i64 1, !dbg !70
  %1830 = insertelement <4 x float> %1829, float %1772, i64 2, !dbg !70
  %1831 = insertelement <4 x float> %1830, float %1778, i64 3, !dbg !70
  %1832 = fadd <4 x float> %1827, %1831, !dbg !70
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !53
  fence syncscope("workgroup") release, !dbg !53
  call void @llvm.amdgcn.s.barrier(), !dbg !53
  fence syncscope("workgroup") acquire, !dbg !53
  %address.local966 = shl nuw nsw i32 %address.thread928, 3, !dbg !49
  %address.local967 = and i32 %address.local966, 504, !dbg !49
  %address.local968 = zext nneg i32 %address.local967 to i64, !dbg !49
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !49
  %1833 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #0, !dbg !49
  %1834 = bitcast <4 x i32> %1833 to <8 x half>, !dbg !49
  %address.local970 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !49
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local970, i64 1024, !dbg !49
  %1835 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #0, !dbg !49
  %1836 = bitcast <4 x i32> %1835 to <8 x half>, !dbg !49
  %address.local972 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !49
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local972, i64 2048, !dbg !49
  %1837 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !49
  %1838 = bitcast <4 x i32> %1837 to <8 x half>, !dbg !49
  %address.local974 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !49
  %address.local975 = getelementptr i8, ptr addrspace(3) %address.local974, i64 3072, !dbg !49
  %1839 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #0, !dbg !49
  %1840 = bitcast <4 x i32> %1839 to <8 x half>, !dbg !49
  %address.local976 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !49
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local976, i64 4096, !dbg !49
  %1841 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !49
  %1842 = bitcast <4 x i32> %1841 to <8 x half>, !dbg !49
  %address.local978 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !49
  %address.local979 = getelementptr i8, ptr addrspace(3) %address.local978, i64 5120, !dbg !49
  %1843 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #0, !dbg !49
  %1844 = bitcast <4 x i32> %1843 to <8 x half>, !dbg !49
  %address.local980 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !49
  %address.local981 = getelementptr i8, ptr addrspace(3) %address.local980, i64 6144, !dbg !49
  %1845 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #0, !dbg !49
  %1846 = bitcast <4 x i32> %1845 to <8 x half>, !dbg !49
  %address.local982 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !49
  %address.local983 = getelementptr i8, ptr addrspace(3) %address.local982, i64 7168, !dbg !49
  %1847 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #0, !dbg !49
  %1848 = bitcast <4 x i32> %1847 to <8 x half>, !dbg !49
  %1849 = shufflevector <8 x half> %1834, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1849, <4 x float> zeroinitializer) #9, !dbg !49
  %1851 = shufflevector <8 x half> %1836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1851, <4 x float> %1850) #9, !dbg !49
  %1853 = shufflevector <8 x half> %1838, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1854 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1853, <4 x float> %1852) #9, !dbg !49
  %1855 = shufflevector <8 x half> %1840, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1855, <4 x float> %1854) #9, !dbg !49
  %1857 = shufflevector <8 x half> %1842, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1857, <4 x float> %1856) #9, !dbg !49
  %1859 = shufflevector <8 x half> %1844, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1859, <4 x float> %1858) #9, !dbg !49
  %1861 = shufflevector <8 x half> %1846, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1861, <4 x float> %1860) #9, !dbg !49
  %1863 = shufflevector <8 x half> %1848, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1863, <4 x float> %1862) #9, !dbg !49
  %1865 = extractelement <4 x float> %1864, i64 0, !dbg !49
  %1866 = extractelement <4 x float> %1864, i64 1, !dbg !49
  %1867 = extractelement <4 x float> %1864, i64 2, !dbg !49
  %1868 = extractelement <4 x float> %1864, i64 3, !dbg !49
  %1869 = select i1 %.not, float %1866, float %1865, !dbg !49
  %1870 = bitcast float %1869 to i32, !dbg !49
  %1871 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1870), !dbg !49
  %1872 = bitcast i32 %1871 to float, !dbg !49
  %1873 = select i1 %.not, float %1865, float %1872, !dbg !49
  %1874 = select i1 %.not, float %1872, float %1866, !dbg !49
  %1875 = select i1 %.not, float %1868, float %1867, !dbg !49
  %1876 = bitcast float %1875 to i32, !dbg !49
  %1877 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1876), !dbg !49
  %1878 = bitcast i32 %1877 to float, !dbg !49
  %1879 = select i1 %.not, float %1867, float %1878, !dbg !49
  %1880 = select i1 %.not, float %1878, float %1868, !dbg !49
  %1881 = select i1 %.not1, float %1879, float %1873, !dbg !49
  %1882 = bitcast float %1881 to i32, !dbg !49
  %1883 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1882), !dbg !49
  %1884 = bitcast i32 %1883 to float, !dbg !49
  %1885 = select i1 %.not1, float %1873, float %1884, !dbg !49
  %1886 = select i1 %.not1, float %1884, float %1879, !dbg !49
  %1887 = select i1 %.not1, float %1880, float %1874, !dbg !49
  %1888 = bitcast float %1887 to i32, !dbg !49
  %1889 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1888), !dbg !49
  %1890 = bitcast i32 %1889 to float, !dbg !49
  %1891 = select i1 %.not1, float %1874, float %1890, !dbg !49
  %1892 = select i1 %.not1, float %1890, float %1880, !dbg !49
  %1893 = shufflevector <8 x half> %1834, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1893, <4 x float> zeroinitializer) #9, !dbg !49
  %1895 = shufflevector <8 x half> %1836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1895, <4 x float> %1894) #9, !dbg !49
  %1897 = shufflevector <8 x half> %1838, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1897, <4 x float> %1896) #9, !dbg !49
  %1899 = shufflevector <8 x half> %1840, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1899, <4 x float> %1898) #9, !dbg !49
  %1901 = shufflevector <8 x half> %1842, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1901, <4 x float> %1900) #9, !dbg !49
  %1903 = shufflevector <8 x half> %1844, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1903, <4 x float> %1902) #9, !dbg !49
  %1905 = shufflevector <8 x half> %1846, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1905, <4 x float> %1904) #9, !dbg !49
  %1907 = shufflevector <8 x half> %1848, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1907, <4 x float> %1906) #9, !dbg !49
  %1909 = extractelement <4 x float> %1908, i64 0, !dbg !49
  %1910 = extractelement <4 x float> %1908, i64 1, !dbg !49
  %1911 = extractelement <4 x float> %1908, i64 2, !dbg !49
  %1912 = extractelement <4 x float> %1908, i64 3, !dbg !49
  %1913 = select i1 %.not, float %1910, float %1909, !dbg !49
  %1914 = bitcast float %1913 to i32, !dbg !49
  %1915 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1914), !dbg !49
  %1916 = bitcast i32 %1915 to float, !dbg !49
  %1917 = select i1 %.not, float %1909, float %1916, !dbg !49
  %1918 = select i1 %.not, float %1916, float %1910, !dbg !49
  %1919 = select i1 %.not, float %1912, float %1911, !dbg !49
  %1920 = bitcast float %1919 to i32, !dbg !49
  %1921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1920), !dbg !49
  %1922 = bitcast i32 %1921 to float, !dbg !49
  %1923 = select i1 %.not, float %1911, float %1922, !dbg !49
  %1924 = select i1 %.not, float %1922, float %1912, !dbg !49
  %1925 = select i1 %.not1, float %1923, float %1917, !dbg !49
  %1926 = bitcast float %1925 to i32, !dbg !49
  %1927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1926), !dbg !49
  %1928 = bitcast i32 %1927 to float, !dbg !49
  %1929 = select i1 %.not1, float %1917, float %1928, !dbg !49
  %1930 = select i1 %.not1, float %1928, float %1923, !dbg !49
  %1931 = select i1 %.not1, float %1924, float %1918, !dbg !49
  %1932 = bitcast float %1931 to i32, !dbg !49
  %1933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1932), !dbg !49
  %1934 = bitcast i32 %1933 to float, !dbg !49
  %1935 = select i1 %.not1, float %1918, float %1934, !dbg !49
  %1936 = select i1 %.not1, float %1934, float %1924, !dbg !49
  %1937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1849, <4 x float> zeroinitializer) #9, !dbg !49
  %1938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1851, <4 x float> %1937) #9, !dbg !49
  %1939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1853, <4 x float> %1938) #9, !dbg !49
  %1940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1855, <4 x float> %1939) #9, !dbg !49
  %1941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1857, <4 x float> %1940) #9, !dbg !49
  %1942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1859, <4 x float> %1941) #9, !dbg !49
  %1943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1861, <4 x float> %1942) #9, !dbg !49
  %1944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1863, <4 x float> %1943) #9, !dbg !49
  %1945 = extractelement <4 x float> %1944, i64 0, !dbg !49
  %1946 = extractelement <4 x float> %1944, i64 1, !dbg !49
  %1947 = extractelement <4 x float> %1944, i64 2, !dbg !49
  %1948 = extractelement <4 x float> %1944, i64 3, !dbg !49
  %1949 = select i1 %.not, float %1946, float %1945, !dbg !49
  %1950 = bitcast float %1949 to i32, !dbg !49
  %1951 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1950), !dbg !49
  %1952 = bitcast i32 %1951 to float, !dbg !49
  %1953 = select i1 %.not, float %1945, float %1952, !dbg !49
  %1954 = select i1 %.not, float %1952, float %1946, !dbg !49
  %1955 = select i1 %.not, float %1948, float %1947, !dbg !49
  %1956 = bitcast float %1955 to i32, !dbg !49
  %1957 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1956), !dbg !49
  %1958 = bitcast i32 %1957 to float, !dbg !49
  %1959 = select i1 %.not, float %1947, float %1958, !dbg !49
  %1960 = select i1 %.not, float %1958, float %1948, !dbg !49
  %1961 = select i1 %.not1, float %1959, float %1953, !dbg !49
  %1962 = bitcast float %1961 to i32, !dbg !49
  %1963 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1962), !dbg !49
  %1964 = bitcast i32 %1963 to float, !dbg !49
  %1965 = select i1 %.not1, float %1953, float %1964, !dbg !49
  %1966 = select i1 %.not1, float %1964, float %1959, !dbg !49
  %1967 = select i1 %.not1, float %1960, float %1954, !dbg !49
  %1968 = bitcast float %1967 to i32, !dbg !49
  %1969 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1968), !dbg !49
  %1970 = bitcast i32 %1969 to float, !dbg !49
  %1971 = select i1 %.not1, float %1954, float %1970, !dbg !49
  %1972 = select i1 %.not1, float %1970, float %1960, !dbg !49
  %1973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1893, <4 x float> zeroinitializer) #9, !dbg !49
  %1974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1895, <4 x float> %1973) #9, !dbg !49
  %1975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1897, <4 x float> %1974) #9, !dbg !49
  %1976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1899, <4 x float> %1975) #9, !dbg !49
  %1977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1901, <4 x float> %1976) #9, !dbg !49
  %1978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1903, <4 x float> %1977) #9, !dbg !49
  %1979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1905, <4 x float> %1978) #9, !dbg !49
  %1980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1907, <4 x float> %1979) #9, !dbg !49
  %1981 = extractelement <4 x float> %1980, i64 0, !dbg !49
  %1982 = extractelement <4 x float> %1980, i64 1, !dbg !49
  %1983 = extractelement <4 x float> %1980, i64 2, !dbg !49
  %1984 = extractelement <4 x float> %1980, i64 3, !dbg !49
  %1985 = select i1 %.not, float %1982, float %1981, !dbg !49
  %1986 = bitcast float %1985 to i32, !dbg !49
  %1987 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1986), !dbg !49
  %1988 = bitcast i32 %1987 to float, !dbg !49
  %1989 = select i1 %.not, float %1981, float %1988, !dbg !49
  %1990 = select i1 %.not, float %1988, float %1982, !dbg !49
  %1991 = select i1 %.not, float %1984, float %1983, !dbg !49
  %1992 = bitcast float %1991 to i32, !dbg !49
  %1993 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1992), !dbg !49
  %1994 = bitcast i32 %1993 to float, !dbg !49
  %1995 = select i1 %.not, float %1983, float %1994, !dbg !49
  %1996 = select i1 %.not, float %1994, float %1984, !dbg !49
  %1997 = select i1 %.not1, float %1995, float %1989, !dbg !49
  %1998 = bitcast float %1997 to i32, !dbg !49
  %1999 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %1998), !dbg !49
  %2000 = bitcast i32 %1999 to float, !dbg !49
  %2001 = select i1 %.not1, float %1989, float %2000, !dbg !49
  %2002 = select i1 %.not1, float %2000, float %1995, !dbg !49
  %2003 = select i1 %.not1, float %1996, float %1990, !dbg !49
  %2004 = bitcast float %2003 to i32, !dbg !49
  %2005 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %2004), !dbg !49
  %2006 = bitcast i32 %2005 to float, !dbg !49
  %2007 = select i1 %.not1, float %1990, float %2006, !dbg !49
  %2008 = select i1 %.not1, float %2006, float %1996, !dbg !49
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  br i1 %43, label %2009, label %2010, !dbg !69

2009:                                             ; preds = %1780
  %address.thread984 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !69, !range !73, !frisk.register_lifetimes !37
  %address.local985 = lshr i32 %address.thread984, 1, !dbg !48
  %address.local986 = and i32 %address.local985, 96, !dbg !48
  %address.local987 = and i32 %address.thread984, 15, !dbg !48
  %address.local988 = or disjoint i32 %address.local986, %address.local987, !dbg !48
  %address.local989 = zext nneg i32 %address.local988 to i64, !dbg !48
  %address.local990 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !48
  store float %1110, ptr addrspace(3) %address.local990, align 4, !dbg !69
  %address.local991 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !48
  %address.local992 = getelementptr i8, ptr addrspace(3) %address.local991, i64 64, !dbg !48
  store float %1111, ptr addrspace(3) %address.local992, align 4, !dbg !69
  br label %2010, !dbg !69

2010:                                             ; preds = %2009, %1780
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  %address.thread993 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !69, !range !73, !frisk.register_lifetimes !37
  %2011 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !69
  %lds.const.gep1752 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !69
  %2012 = load float, ptr addrspace(3) %lds.const.gep1752, align 4, !dbg !69
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  %address.local1007 = zext nneg i32 %address.thread993 to i64, !dbg !45
  %address.local1008 = shl nuw nsw i64 %address.local1007, 3, !dbg !44
  %address.local1009 = and i64 %address.local1008, 1536, !dbg !44
  %address.local1010 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1009, !dbg !44
  %address.local1011 = ptrtoint ptr addrspace(3) %address.local1010 to i64, !dbg !44
  %address.local1012 = trunc i64 %address.local1011 to i32, !dbg !44
  %2013 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1012), !dbg !53
  %address.local1013 = shl nuw nsw i64 %address.local1007, 6, !dbg !53
  %address.local1014 = and i64 %address.local1013, 16128, !dbg !53
  %address.local1015 = shl nuw nsw i64 %address.local1007, 4, !dbg !53
  %address.local1016 = and i64 %address.local1015, 48, !dbg !53
  %2014 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !53
  %.idx59 = shl i64 %1552, 1, !dbg !53
  %2015 = or disjoint i64 %address.local1014, %address.local1016, !dbg !53
  %2016 = or disjoint i64 %.idx59, %2015, !dbg !53
  %2017 = trunc i64 %2016 to i32, !dbg !53
  %dma.buffer.offset1755 = or i32 %2017, 192, !dbg !53
  %2018 = insertelement <2 x i64> poison, i64 %2014, i64 0, !dbg !53
  %2019 = bitcast <2 x i64> %2018 to <4 x i32>, !dbg !53
  %dma.buffer.rsrc1756 = shufflevector <4 x i32> %2019, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !53
  %dma.buffer.lds1757 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %2013), !dbg !53
  %2020 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1757, i32 %dma.buffer.offset1755, <4 x i32> %dma.buffer.rsrc1756) #10, !dbg !53
  %address.local1019 = or disjoint i64 %address.local1007, 256, !dbg !44
  %address.local1020 = shl nuw nsw i64 %address.local1019, 3, !dbg !44
  %address.local1021 = and i64 %address.local1020, 3584, !dbg !44
  %address.local1022 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1021, !dbg !44
  %address.local1023 = ptrtoint ptr addrspace(3) %address.local1022 to i64, !dbg !44
  %address.local1024 = trunc i64 %address.local1023 to i32, !dbg !44
  %2021 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1024), !dbg !53
  %address.local1025 = shl nuw nsw i64 %address.local1019, 6, !dbg !53
  %address.local1026 = and i64 %address.local1025, 32512, !dbg !53
  %address.local1027 = shl nuw nsw i64 %address.local1007, 4, !dbg !53
  %address.local1028 = and i64 %address.local1027, 48, !dbg !53
  %2022 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !53
  %.idx61 = shl i64 %1552, 1, !dbg !53
  %2023 = or disjoint i64 %.idx61, %address.local1026, !dbg !53
  %2024 = or disjoint i64 %2023, %address.local1028, !dbg !53
  %2025 = trunc i64 %2024 to i32, !dbg !53
  %dma.buffer.offset = or i32 %2025, 192, !dbg !53
  %2026 = insertelement <2 x i64> poison, i64 %2022, i64 0, !dbg !53
  %2027 = bitcast <2 x i64> %2026 to <4 x i32>, !dbg !53
  %dma.buffer.rsrc = shufflevector <4 x i32> %2027, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !53
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %2021), !dbg !53
  %2028 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !53
  %2029 = extractvalue [2 x <8 x float>] %59, 0, !dbg !69
  %2030 = shufflevector <8 x float> %2029, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !69
  %2031 = insertelement <4 x float> poison, float %2011, i64 0, !dbg !69
  %2032 = shufflevector <4 x float> %2031, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !69
  %2033 = fmul <4 x float> %2030, %2032, !dbg !69
  %2034 = insertelement <4 x float> poison, float %1885, i64 0, !dbg !70
  %2035 = insertelement <4 x float> %2034, float %1891, i64 1, !dbg !70
  %2036 = insertelement <4 x float> %2035, float %1886, i64 2, !dbg !70
  %2037 = insertelement <4 x float> %2036, float %1892, i64 3, !dbg !70
  %2038 = fadd <4 x float> %2033, %2037, !dbg !70
  %2039 = shufflevector <8 x float> %2029, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !69
  %2040 = fmul <4 x float> %2039, %2032, !dbg !69
  %2041 = insertelement <4 x float> poison, float %1929, i64 0, !dbg !70
  %2042 = insertelement <4 x float> %2041, float %1935, i64 1, !dbg !70
  %2043 = insertelement <4 x float> %2042, float %1930, i64 2, !dbg !70
  %2044 = insertelement <4 x float> %2043, float %1936, i64 3, !dbg !70
  %2045 = fadd <4 x float> %2040, %2044, !dbg !70
  %2046 = extractvalue [2 x <8 x float>] %59, 1, !dbg !69
  %2047 = shufflevector <8 x float> %2046, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !69
  %2048 = insertelement <4 x float> poison, float %2012, i64 0, !dbg !69
  %2049 = shufflevector <4 x float> %2048, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !69
  %2050 = fmul <4 x float> %2047, %2049, !dbg !69
  %2051 = insertelement <4 x float> poison, float %1965, i64 0, !dbg !70
  %2052 = insertelement <4 x float> %2051, float %1971, i64 1, !dbg !70
  %2053 = insertelement <4 x float> %2052, float %1966, i64 2, !dbg !70
  %2054 = insertelement <4 x float> %2053, float %1972, i64 3, !dbg !70
  %2055 = fadd <4 x float> %2050, %2054, !dbg !70
  %2056 = shufflevector <8 x float> %2046, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !69
  %2057 = fmul <4 x float> %2056, %2049, !dbg !69
  %2058 = insertelement <4 x float> poison, float %2001, i64 0, !dbg !70
  %2059 = insertelement <4 x float> %2058, float %2007, i64 1, !dbg !70
  %2060 = insertelement <4 x float> %2059, float %2002, i64 2, !dbg !70
  %2061 = insertelement <4 x float> %2060, float %2008, i64 3, !dbg !70
  %2062 = fadd <4 x float> %2057, %2061, !dbg !70
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !53
  fence syncscope("workgroup") release, !dbg !53
  call void @llvm.amdgcn.s.barrier(), !dbg !53
  fence syncscope("workgroup") acquire, !dbg !53
  %address.local1031 = shl nuw nsw i32 %address.thread993, 3, !dbg !49
  %address.local1032 = and i32 %address.local1031, 504, !dbg !49
  %address.local1033 = zext nneg i32 %address.local1032 to i64, !dbg !49
  %address.local1034 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !49
  %2063 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !49
  %2064 = bitcast <4 x i32> %2063 to <8 x half>, !dbg !49
  %address.local1035 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !49
  %address.local1036 = getelementptr i8, ptr addrspace(3) %address.local1035, i64 1024, !dbg !49
  %2065 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1036, i32 0) #0, !dbg !49
  %2066 = bitcast <4 x i32> %2065 to <8 x half>, !dbg !49
  %address.local1037 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !49
  %address.local1038 = getelementptr i8, ptr addrspace(3) %address.local1037, i64 2048, !dbg !49
  %2067 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1038, i32 0) #0, !dbg !49
  %2068 = bitcast <4 x i32> %2067 to <8 x half>, !dbg !49
  %address.local1039 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !49
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 3072, !dbg !49
  %2069 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !49
  %2070 = bitcast <4 x i32> %2069 to <8 x half>, !dbg !49
  %address.local1041 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !49
  %address.local1042 = getelementptr i8, ptr addrspace(3) %address.local1041, i64 4096, !dbg !49
  %2071 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1042, i32 0) #0, !dbg !49
  %2072 = bitcast <4 x i32> %2071 to <8 x half>, !dbg !49
  %address.local1043 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !49
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1043, i64 5120, !dbg !49
  %2073 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !49
  %2074 = bitcast <4 x i32> %2073 to <8 x half>, !dbg !49
  %address.local1045 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !49
  %address.local1046 = getelementptr i8, ptr addrspace(3) %address.local1045, i64 6144, !dbg !49
  %2075 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1046, i32 0) #0, !dbg !49
  %2076 = bitcast <4 x i32> %2075 to <8 x half>, !dbg !49
  %address.local1047 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !49
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1047, i64 7168, !dbg !49
  %2077 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !49
  %2078 = bitcast <4 x i32> %2077 to <8 x half>, !dbg !49
  %2079 = shufflevector <8 x half> %2064, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %2080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2079, <4 x float> zeroinitializer) #9, !dbg !49
  %2081 = shufflevector <8 x half> %2066, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %2082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2081, <4 x float> %2080) #9, !dbg !49
  %2083 = shufflevector <8 x half> %2068, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %2084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2083, <4 x float> %2082) #9, !dbg !49
  %2085 = shufflevector <8 x half> %2070, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %2086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2085, <4 x float> %2084) #9, !dbg !49
  %2087 = shufflevector <8 x half> %2072, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %2088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2087, <4 x float> %2086) #9, !dbg !49
  %2089 = shufflevector <8 x half> %2074, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %2090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2089, <4 x float> %2088) #9, !dbg !49
  %2091 = shufflevector <8 x half> %2076, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %2092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2091, <4 x float> %2090) #9, !dbg !49
  %2093 = shufflevector <8 x half> %2078, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %2094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2093, <4 x float> %2092) #9, !dbg !49
  %2095 = extractelement <4 x float> %2094, i64 0, !dbg !49
  %2096 = extractelement <4 x float> %2094, i64 1, !dbg !49
  %2097 = extractelement <4 x float> %2094, i64 2, !dbg !49
  %2098 = extractelement <4 x float> %2094, i64 3, !dbg !49
  %2099 = select i1 %.not, float %2096, float %2095, !dbg !49
  %2100 = bitcast float %2099 to i32, !dbg !49
  %2101 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %2100), !dbg !49
  %2102 = bitcast i32 %2101 to float, !dbg !49
  %2103 = select i1 %.not, float %2095, float %2102, !dbg !49
  %2104 = select i1 %.not, float %2102, float %2096, !dbg !49
  %2105 = select i1 %.not, float %2098, float %2097, !dbg !49
  %2106 = bitcast float %2105 to i32, !dbg !49
  %2107 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %2106), !dbg !49
  %2108 = bitcast i32 %2107 to float, !dbg !49
  %2109 = select i1 %.not, float %2097, float %2108, !dbg !49
  %2110 = select i1 %.not, float %2108, float %2098, !dbg !49
  %2111 = select i1 %.not1, float %2109, float %2103, !dbg !49
  %2112 = bitcast float %2111 to i32, !dbg !49
  %2113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %2112), !dbg !49
  %2114 = bitcast i32 %2113 to float, !dbg !49
  %2115 = select i1 %.not1, float %2103, float %2114, !dbg !49
  %2116 = select i1 %.not1, float %2114, float %2109, !dbg !49
  %2117 = select i1 %.not1, float %2110, float %2104, !dbg !49
  %2118 = bitcast float %2117 to i32, !dbg !49
  %2119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %2118), !dbg !49
  %2120 = bitcast i32 %2119 to float, !dbg !49
  %2121 = select i1 %.not1, float %2104, float %2120, !dbg !49
  %2122 = select i1 %.not1, float %2120, float %2110, !dbg !49
  %2123 = shufflevector <8 x half> %2064, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %2124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2123, <4 x float> zeroinitializer) #9, !dbg !49
  %2125 = shufflevector <8 x half> %2066, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %2126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2125, <4 x float> %2124) #9, !dbg !49
  %2127 = shufflevector <8 x half> %2068, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %2128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2127, <4 x float> %2126) #9, !dbg !49
  %2129 = shufflevector <8 x half> %2070, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %2130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2129, <4 x float> %2128) #9, !dbg !49
  %2131 = shufflevector <8 x half> %2072, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %2132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2131, <4 x float> %2130) #9, !dbg !49
  %2133 = shufflevector <8 x half> %2074, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %2134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2133, <4 x float> %2132) #9, !dbg !49
  %2135 = shufflevector <8 x half> %2076, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %2136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2135, <4 x float> %2134) #9, !dbg !49
  %2137 = shufflevector <8 x half> %2078, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %2138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2137, <4 x float> %2136) #9, !dbg !49
  %2139 = extractelement <4 x float> %2138, i64 0, !dbg !49
  %2140 = extractelement <4 x float> %2138, i64 1, !dbg !49
  %2141 = extractelement <4 x float> %2138, i64 2, !dbg !49
  %2142 = extractelement <4 x float> %2138, i64 3, !dbg !49
  %2143 = select i1 %.not, float %2140, float %2139, !dbg !49
  %2144 = bitcast float %2143 to i32, !dbg !49
  %2145 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %2144), !dbg !49
  %2146 = bitcast i32 %2145 to float, !dbg !49
  %2147 = select i1 %.not, float %2139, float %2146, !dbg !49
  %2148 = select i1 %.not, float %2146, float %2140, !dbg !49
  %2149 = select i1 %.not, float %2142, float %2141, !dbg !49
  %2150 = bitcast float %2149 to i32, !dbg !49
  %2151 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %2150), !dbg !49
  %2152 = bitcast i32 %2151 to float, !dbg !49
  %2153 = select i1 %.not, float %2141, float %2152, !dbg !49
  %2154 = select i1 %.not, float %2152, float %2142, !dbg !49
  %2155 = select i1 %.not1, float %2153, float %2147, !dbg !49
  %2156 = bitcast float %2155 to i32, !dbg !49
  %2157 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %2156), !dbg !49
  %2158 = bitcast i32 %2157 to float, !dbg !49
  %2159 = select i1 %.not1, float %2147, float %2158, !dbg !49
  %2160 = select i1 %.not1, float %2158, float %2153, !dbg !49
  %2161 = select i1 %.not1, float %2154, float %2148, !dbg !49
  %2162 = bitcast float %2161 to i32, !dbg !49
  %2163 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %2162), !dbg !49
  %2164 = bitcast i32 %2163 to float, !dbg !49
  %2165 = select i1 %.not1, float %2148, float %2164, !dbg !49
  %2166 = select i1 %.not1, float %2164, float %2154, !dbg !49
  %2167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2079, <4 x float> zeroinitializer) #9, !dbg !49
  %2168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2081, <4 x float> %2167) #9, !dbg !49
  %2169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2083, <4 x float> %2168) #9, !dbg !49
  %2170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2085, <4 x float> %2169) #9, !dbg !49
  %2171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2087, <4 x float> %2170) #9, !dbg !49
  %2172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2089, <4 x float> %2171) #9, !dbg !49
  %2173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2091, <4 x float> %2172) #9, !dbg !49
  %2174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2093, <4 x float> %2173) #9, !dbg !49
  %2175 = extractelement <4 x float> %2174, i64 0, !dbg !49
  %2176 = extractelement <4 x float> %2174, i64 1, !dbg !49
  %2177 = extractelement <4 x float> %2174, i64 2, !dbg !49
  %2178 = extractelement <4 x float> %2174, i64 3, !dbg !49
  %2179 = select i1 %.not, float %2176, float %2175, !dbg !49
  %2180 = bitcast float %2179 to i32, !dbg !49
  %2181 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %2180), !dbg !49
  %2182 = bitcast i32 %2181 to float, !dbg !49
  %2183 = select i1 %.not, float %2175, float %2182, !dbg !49
  %2184 = select i1 %.not, float %2182, float %2176, !dbg !49
  %2185 = select i1 %.not, float %2178, float %2177, !dbg !49
  %2186 = bitcast float %2185 to i32, !dbg !49
  %2187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %2186), !dbg !49
  %2188 = bitcast i32 %2187 to float, !dbg !49
  %2189 = select i1 %.not, float %2177, float %2188, !dbg !49
  %2190 = select i1 %.not, float %2188, float %2178, !dbg !49
  %2191 = select i1 %.not1, float %2189, float %2183, !dbg !49
  %2192 = bitcast float %2191 to i32, !dbg !49
  %2193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %2192), !dbg !49
  %2194 = bitcast i32 %2193 to float, !dbg !49
  %2195 = select i1 %.not1, float %2183, float %2194, !dbg !49
  %2196 = select i1 %.not1, float %2194, float %2189, !dbg !49
  %2197 = select i1 %.not1, float %2190, float %2184, !dbg !49
  %2198 = bitcast float %2197 to i32, !dbg !49
  %2199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %2198), !dbg !49
  %2200 = bitcast i32 %2199 to float, !dbg !49
  %2201 = select i1 %.not1, float %2184, float %2200, !dbg !49
  %2202 = select i1 %.not1, float %2200, float %2190, !dbg !49
  %2203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2123, <4 x float> zeroinitializer) #9, !dbg !49
  %2204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2125, <4 x float> %2203) #9, !dbg !49
  %2205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2127, <4 x float> %2204) #9, !dbg !49
  %2206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2129, <4 x float> %2205) #9, !dbg !49
  %2207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2131, <4 x float> %2206) #9, !dbg !49
  %2208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2133, <4 x float> %2207) #9, !dbg !49
  %2209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2135, <4 x float> %2208) #9, !dbg !49
  %2210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2137, <4 x float> %2209) #9, !dbg !49
  %2211 = extractelement <4 x float> %2210, i64 0, !dbg !49
  %2212 = extractelement <4 x float> %2210, i64 1, !dbg !49
  %2213 = extractelement <4 x float> %2210, i64 2, !dbg !49
  %2214 = extractelement <4 x float> %2210, i64 3, !dbg !49
  %2215 = select i1 %.not, float %2212, float %2211, !dbg !49
  %2216 = bitcast float %2215 to i32, !dbg !49
  %2217 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %2216), !dbg !49
  %2218 = bitcast i32 %2217 to float, !dbg !49
  %2219 = select i1 %.not, float %2211, float %2218, !dbg !49
  %2220 = select i1 %.not, float %2218, float %2212, !dbg !49
  %2221 = select i1 %.not, float %2214, float %2213, !dbg !49
  %2222 = bitcast float %2221 to i32, !dbg !49
  %2223 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %2222), !dbg !49
  %2224 = bitcast i32 %2223 to float, !dbg !49
  %2225 = select i1 %.not, float %2213, float %2224, !dbg !49
  %2226 = select i1 %.not, float %2224, float %2214, !dbg !49
  %2227 = select i1 %.not1, float %2225, float %2219, !dbg !49
  %2228 = bitcast float %2227 to i32, !dbg !49
  %2229 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %2228), !dbg !49
  %2230 = bitcast i32 %2229 to float, !dbg !49
  %2231 = select i1 %.not1, float %2219, float %2230, !dbg !49
  %2232 = select i1 %.not1, float %2230, float %2225, !dbg !49
  %2233 = select i1 %.not1, float %2226, float %2220, !dbg !49
  %2234 = bitcast float %2233 to i32, !dbg !49
  %2235 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1061, i32 %2234), !dbg !49
  %2236 = bitcast i32 %2235 to float, !dbg !49
  %2237 = select i1 %.not1, float %2220, float %2236, !dbg !49
  %2238 = select i1 %.not1, float %2236, float %2226, !dbg !49
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  br i1 %43, label %2239, label %2240, !dbg !69

2239:                                             ; preds = %2010
  %address.thread1049 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !69, !range !73, !frisk.register_lifetimes !37
  %address.local1050 = lshr i32 %address.thread1049, 1, !dbg !48
  %address.local1051 = and i32 %address.local1050, 96, !dbg !48
  %address.local1052 = and i32 %address.thread1049, 15, !dbg !48
  %address.local1053 = or disjoint i32 %address.local1051, %address.local1052, !dbg !48
  %address.local1054 = zext nneg i32 %address.local1053 to i64, !dbg !48
  %address.local1055 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !48
  store float %1110, ptr addrspace(3) %address.local1055, align 4, !dbg !69
  %address.local1056 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !48
  %address.local1057 = getelementptr i8, ptr addrspace(3) %address.local1056, i64 64, !dbg !48
  store float %1111, ptr addrspace(3) %address.local1057, align 4, !dbg !69
  br label %2240, !dbg !69

2240:                                             ; preds = %2239, %2010
  fence syncscope("workgroup") release, !dbg !69
  call void @llvm.amdgcn.s.barrier(), !dbg !69
  fence syncscope("workgroup") acquire, !dbg !69
  %2241 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !69
  %lds.const.gep1754 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !69
  %2242 = load float, ptr addrspace(3) %lds.const.gep1754, align 4, !dbg !69
  %2243 = extractvalue [2 x <8 x float>] %60, 0, !dbg !69
  %2244 = shufflevector <8 x float> %2243, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !69
  %2245 = insertelement <4 x float> poison, float %2241, i64 0, !dbg !69
  %2246 = shufflevector <4 x float> %2245, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !69
  %2247 = fmul <4 x float> %2244, %2246, !dbg !69
  %2248 = insertelement <4 x float> poison, float %2115, i64 0, !dbg !70
  %2249 = insertelement <4 x float> %2248, float %2121, i64 1, !dbg !70
  %2250 = insertelement <4 x float> %2249, float %2116, i64 2, !dbg !70
  %2251 = insertelement <4 x float> %2250, float %2122, i64 3, !dbg !70
  %2252 = fadd <4 x float> %2247, %2251, !dbg !70
  %2253 = shufflevector <8 x float> %2243, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !69
  %2254 = fmul <4 x float> %2253, %2246, !dbg !69
  %2255 = insertelement <4 x float> poison, float %2159, i64 0, !dbg !70
  %2256 = insertelement <4 x float> %2255, float %2165, i64 1, !dbg !70
  %2257 = insertelement <4 x float> %2256, float %2160, i64 2, !dbg !70
  %2258 = insertelement <4 x float> %2257, float %2166, i64 3, !dbg !70
  %2259 = fadd <4 x float> %2254, %2258, !dbg !70
  %2260 = extractvalue [2 x <8 x float>] %60, 1, !dbg !69
  %2261 = shufflevector <8 x float> %2260, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !69
  %2262 = insertelement <4 x float> poison, float %2242, i64 0, !dbg !69
  %2263 = shufflevector <4 x float> %2262, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !69
  %2264 = fmul <4 x float> %2261, %2263, !dbg !69
  %2265 = insertelement <4 x float> poison, float %2195, i64 0, !dbg !70
  %2266 = insertelement <4 x float> %2265, float %2201, i64 1, !dbg !70
  %2267 = insertelement <4 x float> %2266, float %2196, i64 2, !dbg !70
  %2268 = insertelement <4 x float> %2267, float %2202, i64 3, !dbg !70
  %2269 = fadd <4 x float> %2264, %2268, !dbg !70
  %2270 = shufflevector <8 x float> %2260, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !69
  %2271 = fmul <4 x float> %2270, %2263, !dbg !69
  %2272 = insertelement <4 x float> poison, float %2231, i64 0, !dbg !70
  %2273 = insertelement <4 x float> %2272, float %2237, i64 1, !dbg !70
  %2274 = insertelement <4 x float> %2273, float %2232, i64 2, !dbg !70
  %2275 = insertelement <4 x float> %2274, float %2238, i64 3, !dbg !70
  %2276 = fadd <4 x float> %2271, %2275, !dbg !70
  %2277 = shufflevector <4 x float> %1578, <4 x float> %1585, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %2278 = insertvalue [2 x <8 x float>] poison, <8 x float> %2277, 0, !dbg !42
  %2279 = shufflevector <4 x float> %1595, <4 x float> %1602, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %2280 = insertvalue [2 x <8 x float>] %2278, <8 x float> %2279, 1, !dbg !42
  %2281 = insertelement <1 x float> poison, float %1359, i64 0, !dbg !42
  %2282 = insertvalue [2 x <1 x float>] poison, <1 x float> %2281, 0, !dbg !42
  %2283 = insertelement <1 x float> poison, float %1368, i64 0, !dbg !42
  %2284 = insertvalue [2 x <1 x float>] %2282, <1 x float> %2283, 1, !dbg !42
  %2285 = insertvalue [2 x <1 x float>] poison, <1 x float> %1068, 0, !dbg !42
  %2286 = insertvalue [2 x <1 x float>] %2285, <1 x float> %1081, 1, !dbg !42
  %2287 = shufflevector <4 x float> %1808, <4 x float> %1815, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %2288 = insertvalue [2 x <8 x float>] poison, <8 x float> %2287, 0, !dbg !42
  %2289 = shufflevector <4 x float> %1825, <4 x float> %1832, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %2290 = insertvalue [2 x <8 x float>] %2288, <8 x float> %2289, 1, !dbg !42
  %2291 = shufflevector <4 x float> %2038, <4 x float> %2045, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %2292 = insertvalue [2 x <8 x float>] poison, <8 x float> %2291, 0, !dbg !42
  %2293 = shufflevector <4 x float> %2055, <4 x float> %2062, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %2294 = insertvalue [2 x <8 x float>] %2292, <8 x float> %2293, 1, !dbg !42
  %2295 = shufflevector <4 x float> %2252, <4 x float> %2259, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %2296 = insertvalue [2 x <8 x float>] poison, <8 x float> %2295, 0, !dbg !42
  %2297 = shufflevector <4 x float> %2269, <4 x float> %2276, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %2298 = insertvalue [2 x <8 x float>] %2296, <8 x float> %2297, 1, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %2299 = add nuw nsw i64 %54, 1, !dbg !42
  br label %53, !dbg !42

2300:                                             ; preds = %53
  %2301 = extractvalue [2 x <1 x float>] %56, 0, !dbg !54
  %2302 = shufflevector <1 x float> %2301, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !54
  %2303 = extractvalue [2 x <1 x float>] %56, 1, !dbg !54
  %2304 = shufflevector <1 x float> %2303, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !54
  br i1 true, label %2305, label %2336, !dbg !56

2305:                                             ; preds = %2300
  %2306 = extractvalue [2 x <8 x float>] %55, 1, !dbg !54
  %2307 = shufflevector <8 x float> %2306, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %2304, !dbg !54
  %normalized = fmul <4 x float> %2307, %row.reciprocal, !dbg !54
  %2308 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !55
  %2309 = extractelement <4 x half> %2308, i64 3, !dbg !55
  %2310 = extractelement <4 x half> %2308, i64 2, !dbg !55
  %2311 = extractelement <4 x half> %2308, i64 1, !dbg !55
  %2312 = extractelement <4 x half> %2308, i64 0, !dbg !55
  %2313 = shufflevector <8 x float> %2306, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %normalized68 = fmul <4 x float> %2313, %row.reciprocal, !dbg !54
  %2314 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !55
  %2315 = extractelement <4 x half> %2314, i64 3, !dbg !55
  %2316 = extractelement <4 x half> %2314, i64 2, !dbg !55
  %2317 = extractelement <4 x half> %2314, i64 1, !dbg !55
  %2318 = extractelement <4 x half> %2314, i64 0, !dbg !55
  %2319 = extractvalue [2 x <8 x float>] %55, 0, !dbg !54
  %2320 = shufflevector <8 x float> %2319, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2302, !dbg !54
  %normalized70 = fmul <4 x float> %2320, %row.reciprocal69, !dbg !54
  %2321 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !55
  %2322 = extractelement <4 x half> %2321, i64 3, !dbg !55
  %2323 = extractelement <4 x half> %2321, i64 2, !dbg !55
  %2324 = extractelement <4 x half> %2321, i64 1, !dbg !55
  %2325 = extractelement <4 x half> %2321, i64 0, !dbg !55
  %2326 = shufflevector <8 x float> %2319, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %normalized71 = fmul <4 x float> %2326, %row.reciprocal69, !dbg !54
  %2327 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !55
  %2328 = extractelement <4 x half> %2327, i64 3, !dbg !55
  %2329 = extractelement <4 x half> %2327, i64 2, !dbg !55
  %2330 = extractelement <4 x half> %2327, i64 1, !dbg !55
  %2331 = extractelement <4 x half> %2327, i64 0, !dbg !55
  %2332 = add nsw i64 %15, %25, !dbg !56
  %2333 = shl i64 %2332, 19, !dbg !56
  %2334 = and i64 %2333, 16252928, !dbg !56
  %address.thread1072 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !56, !range !73, !frisk.register_lifetimes !37
  %address.local1073 = zext nneg i32 %address.thread1072 to i64, !dbg !45
  %address.local1074 = add i64 %28, %address.local1073, !dbg !56
  %address.local1075 = lshr i64 %address.local1073, 2, !dbg !46
  %address.local1076 = and i64 %address.local1075, 48, !dbg !46
  %address.local1077 = add i64 %address.local1074, %address.local1076, !dbg !56
  %address.local1078 = and i64 %address.local1073, 240, !dbg !56
  %address.local1080 = sub i64 %address.local1077, %address.local1078, !dbg !56
  %address.local1081 = add i64 %address.local1080, %29, !dbg !56
  %address.local1082 = shl nuw nsw i64 %address.local1081, 7, !dbg !56
  %address.local1083 = add nuw nsw i64 %2334, %address.local1082, !dbg !56
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !56
  %2335 = lshr i64 %address.local1073, 2, !dbg !45
  %address.local1087 = and i64 %2335, 12, !dbg !45
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1087, !dbg !56
  store half %2331, ptr addrspace(1) %address.local1088, align 2, !dbg !56
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !56
  %address.local1090 = or disjoint i64 %address.local1087, 1, !dbg !46
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1090, !dbg !56
  store half %2330, ptr addrspace(1) %address.local1091, align 2, !dbg !56
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !56
  %address.local1093 = or disjoint i64 %address.local1087, 2, !dbg !46
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1093, !dbg !56
  store half %2329, ptr addrspace(1) %address.local1094, align 2, !dbg !56
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !56
  %address.local1096 = or disjoint i64 %address.local1087, 3, !dbg !46
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1096, !dbg !56
  store half %2328, ptr addrspace(1) %address.local1097, align 2, !dbg !56
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !56
  %address.local1099 = or disjoint i64 %address.local1087, 16, !dbg !46
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1098, i64 %address.local1099, !dbg !56
  store half %2325, ptr addrspace(1) %address.local1100, align 2, !dbg !56
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !56
  %address.local1102 = or disjoint i64 %address.local1087, 17, !dbg !46
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !56
  store half %2324, ptr addrspace(1) %address.local1103, align 2, !dbg !56
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !56
  %address.local1105 = or disjoint i64 %address.local1087, 18, !dbg !46
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !56
  store half %2323, ptr addrspace(1) %address.local1106, align 2, !dbg !56
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !56
  %address.local1108 = or disjoint i64 %address.local1087, 19, !dbg !46
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !56
  store half %2322, ptr addrspace(1) %address.local1109, align 2, !dbg !56
  %address.local1110 = shl i64 %address.local1081, 7, !dbg !56
  %address.local1111 = add i64 %address.local1110, 8192, !dbg !56
  %address.local1112 = add nuw nsw i64 %2334, %address.local1111, !dbg !56
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !56
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1087, !dbg !56
  store half %2318, ptr addrspace(1) %address.local1114, align 2, !dbg !56
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !56
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1090, !dbg !56
  store half %2317, ptr addrspace(1) %address.local1116, align 2, !dbg !56
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !56
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1093, !dbg !56
  store half %2316, ptr addrspace(1) %address.local1118, align 2, !dbg !56
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !56
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1096, !dbg !56
  store half %2315, ptr addrspace(1) %address.local1120, align 2, !dbg !56
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !56
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1099, !dbg !56
  store half %2312, ptr addrspace(1) %address.local1122, align 2, !dbg !56
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !56
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1102, !dbg !56
  store half %2311, ptr addrspace(1) %address.local1124, align 2, !dbg !56
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !56
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1105, !dbg !56
  store half %2310, ptr addrspace(1) %address.local1126, align 2, !dbg !56
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !56
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1108, !dbg !56
  store half %2309, ptr addrspace(1) %address.local1128, align 2, !dbg !56
  br label %2336, !dbg !56

2336:                                             ; preds = %2305, %2300
  br i1 true, label %2337, label %2366, !dbg !56

2337:                                             ; preds = %2336
  %2338 = extractvalue [2 x <8 x float>] %58, 1, !dbg !54
  %2339 = shufflevector <8 x float> %2338, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2304, !dbg !54
  %normalized73 = fmul <4 x float> %2339, %row.reciprocal72, !dbg !54
  %2340 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !55
  %2341 = extractelement <4 x half> %2340, i64 3, !dbg !55
  %2342 = extractelement <4 x half> %2340, i64 2, !dbg !55
  %2343 = extractelement <4 x half> %2340, i64 1, !dbg !55
  %2344 = extractelement <4 x half> %2340, i64 0, !dbg !55
  %2345 = shufflevector <8 x float> %2338, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %normalized74 = fmul <4 x float> %2345, %row.reciprocal72, !dbg !54
  %2346 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !55
  %2347 = extractelement <4 x half> %2346, i64 3, !dbg !55
  %2348 = extractelement <4 x half> %2346, i64 2, !dbg !55
  %2349 = extractelement <4 x half> %2346, i64 1, !dbg !55
  %2350 = extractelement <4 x half> %2346, i64 0, !dbg !55
  %2351 = extractvalue [2 x <8 x float>] %58, 0, !dbg !54
  %2352 = shufflevector <8 x float> %2351, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2302, !dbg !54
  %normalized76 = fmul <4 x float> %2352, %row.reciprocal75, !dbg !54
  %2353 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !55
  %2354 = extractelement <4 x half> %2353, i64 3, !dbg !55
  %2355 = extractelement <4 x half> %2353, i64 2, !dbg !55
  %2356 = extractelement <4 x half> %2353, i64 1, !dbg !55
  %2357 = extractelement <4 x half> %2353, i64 0, !dbg !55
  %2358 = shufflevector <8 x float> %2351, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %normalized77 = fmul <4 x float> %2358, %row.reciprocal75, !dbg !54
  %2359 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !55
  %2360 = extractelement <4 x half> %2359, i64 3, !dbg !55
  %2361 = extractelement <4 x half> %2359, i64 2, !dbg !55
  %2362 = extractelement <4 x half> %2359, i64 1, !dbg !55
  %2363 = extractelement <4 x half> %2359, i64 0, !dbg !55
  %2364 = shl nuw nsw i64 %27, 19, !dbg !56
  %address.thread1129 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !56, !range !73, !frisk.register_lifetimes !37
  %address.local1130 = zext nneg i32 %address.thread1129 to i64, !dbg !45
  %address.local1131 = add i64 %28, %address.local1130, !dbg !56
  %address.local1132 = lshr i64 %address.local1130, 2, !dbg !46
  %address.local1133 = and i64 %address.local1132, 48, !dbg !46
  %address.local1134 = add i64 %address.local1131, %address.local1133, !dbg !56
  %address.local1135 = and i64 %address.local1130, 240, !dbg !56
  %address.local1137 = sub i64 %address.local1134, %address.local1135, !dbg !56
  %address.local1138 = add i64 %address.local1137, %29, !dbg !56
  %address.local1139 = shl nuw nsw i64 %address.local1138, 7, !dbg !56
  %address.local1140 = add nuw nsw i64 %2364, %address.local1139, !dbg !56
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !56
  %2365 = lshr i64 %address.local1130, 2, !dbg !45
  %address.local1144 = and i64 %2365, 12, !dbg !45
  %address.local1145 = or disjoint i64 %address.local1144, 32, !dbg !46
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1145, !dbg !56
  store half %2363, ptr addrspace(1) %address.local1146, align 2, !dbg !56
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !56
  %address.local1148 = or disjoint i64 %address.local1144, 33, !dbg !46
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1148, !dbg !56
  store half %2362, ptr addrspace(1) %address.local1149, align 2, !dbg !56
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !56
  %address.local1151 = or disjoint i64 %address.local1144, 34, !dbg !46
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1150, i64 %address.local1151, !dbg !56
  store half %2361, ptr addrspace(1) %address.local1152, align 2, !dbg !56
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !56
  %address.local1154 = or disjoint i64 %address.local1144, 35, !dbg !46
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1154, !dbg !56
  store half %2360, ptr addrspace(1) %address.local1155, align 2, !dbg !56
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !56
  %address.local1157 = or disjoint i64 %address.local1144, 48, !dbg !46
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1156, i64 %address.local1157, !dbg !56
  store half %2357, ptr addrspace(1) %address.local1158, align 2, !dbg !56
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !56
  %address.local1160 = or disjoint i64 %address.local1144, 49, !dbg !46
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1160, !dbg !56
  store half %2356, ptr addrspace(1) %address.local1161, align 2, !dbg !56
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !56
  %address.local1163 = or disjoint i64 %address.local1144, 50, !dbg !46
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1162, i64 %address.local1163, !dbg !56
  store half %2355, ptr addrspace(1) %address.local1164, align 2, !dbg !56
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !56
  %address.local1166 = or i64 %2365, 51, !dbg !46
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1166, !dbg !56
  store half %2354, ptr addrspace(1) %address.local1167, align 2, !dbg !56
  %address.local1168 = shl i64 %address.local1138, 7, !dbg !56
  %address.local1169 = add i64 %address.local1168, 8192, !dbg !56
  %address.local1170 = add nuw nsw i64 %2364, %address.local1169, !dbg !56
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !56
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1145, !dbg !56
  store half %2350, ptr addrspace(1) %address.local1172, align 2, !dbg !56
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !56
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1148, !dbg !56
  store half %2349, ptr addrspace(1) %address.local1174, align 2, !dbg !56
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !56
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1151, !dbg !56
  store half %2348, ptr addrspace(1) %address.local1176, align 2, !dbg !56
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !56
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1154, !dbg !56
  store half %2347, ptr addrspace(1) %address.local1178, align 2, !dbg !56
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !56
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1157, !dbg !56
  store half %2344, ptr addrspace(1) %address.local1180, align 2, !dbg !56
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !56
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1160, !dbg !56
  store half %2343, ptr addrspace(1) %address.local1182, align 2, !dbg !56
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !56
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1163, !dbg !56
  store half %2342, ptr addrspace(1) %address.local1184, align 2, !dbg !56
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !56
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1166, !dbg !56
  store half %2341, ptr addrspace(1) %address.local1186, align 2, !dbg !56
  br label %2366, !dbg !56

2366:                                             ; preds = %2337, %2336
  br i1 true, label %2367, label %2396, !dbg !56

2367:                                             ; preds = %2366
  %2368 = extractvalue [2 x <8 x float>] %59, 1, !dbg !54
  %2369 = shufflevector <8 x float> %2368, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2304, !dbg !54
  %normalized79 = fmul <4 x float> %2369, %row.reciprocal78, !dbg !54
  %2370 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !55
  %2371 = extractelement <4 x half> %2370, i64 3, !dbg !55
  %2372 = extractelement <4 x half> %2370, i64 2, !dbg !55
  %2373 = extractelement <4 x half> %2370, i64 1, !dbg !55
  %2374 = extractelement <4 x half> %2370, i64 0, !dbg !55
  %2375 = shufflevector <8 x float> %2368, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %normalized80 = fmul <4 x float> %2375, %row.reciprocal78, !dbg !54
  %2376 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !55
  %2377 = extractelement <4 x half> %2376, i64 3, !dbg !55
  %2378 = extractelement <4 x half> %2376, i64 2, !dbg !55
  %2379 = extractelement <4 x half> %2376, i64 1, !dbg !55
  %2380 = extractelement <4 x half> %2376, i64 0, !dbg !55
  %2381 = extractvalue [2 x <8 x float>] %59, 0, !dbg !54
  %2382 = shufflevector <8 x float> %2381, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2302, !dbg !54
  %normalized82 = fmul <4 x float> %2382, %row.reciprocal81, !dbg !54
  %2383 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !55
  %2384 = extractelement <4 x half> %2383, i64 3, !dbg !55
  %2385 = extractelement <4 x half> %2383, i64 2, !dbg !55
  %2386 = extractelement <4 x half> %2383, i64 1, !dbg !55
  %2387 = extractelement <4 x half> %2383, i64 0, !dbg !55
  %2388 = shufflevector <8 x float> %2381, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %normalized83 = fmul <4 x float> %2388, %row.reciprocal81, !dbg !54
  %2389 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !55
  %2390 = extractelement <4 x half> %2389, i64 3, !dbg !55
  %2391 = extractelement <4 x half> %2389, i64 2, !dbg !55
  %2392 = extractelement <4 x half> %2389, i64 1, !dbg !55
  %2393 = extractelement <4 x half> %2389, i64 0, !dbg !55
  %2394 = shl nuw nsw i64 %27, 19, !dbg !56
  %address.thread1187 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !56, !range !73, !frisk.register_lifetimes !37
  %address.local1188 = zext nneg i32 %address.thread1187 to i64, !dbg !45
  %address.local1189 = add i64 %28, %address.local1188, !dbg !56
  %address.local1190 = lshr i64 %address.local1188, 2, !dbg !46
  %address.local1191 = and i64 %address.local1190, 48, !dbg !46
  %address.local1192 = add i64 %address.local1189, %address.local1191, !dbg !56
  %address.local1193 = and i64 %address.local1188, 240, !dbg !56
  %address.local1195 = sub i64 %address.local1192, %address.local1193, !dbg !56
  %address.local1196 = add i64 %address.local1195, %29, !dbg !56
  %address.local1197 = shl nuw nsw i64 %address.local1196, 7, !dbg !56
  %address.local1198 = add nuw nsw i64 %2394, %address.local1197, !dbg !56
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !56
  %2395 = lshr i64 %address.local1188, 2, !dbg !45
  %address.local1202 = and i64 %2395, 12, !dbg !45
  %address.local1203 = or disjoint i64 %address.local1202, 64, !dbg !46
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1203, !dbg !56
  store half %2393, ptr addrspace(1) %address.local1204, align 2, !dbg !56
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !56
  %address.local1206 = or disjoint i64 %address.local1202, 65, !dbg !46
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1206, !dbg !56
  store half %2392, ptr addrspace(1) %address.local1207, align 2, !dbg !56
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !56
  %address.local1209 = or disjoint i64 %address.local1202, 66, !dbg !46
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1208, i64 %address.local1209, !dbg !56
  store half %2391, ptr addrspace(1) %address.local1210, align 2, !dbg !56
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !56
  %address.local1212 = or disjoint i64 %address.local1202, 67, !dbg !46
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1211, i64 %address.local1212, !dbg !56
  store half %2390, ptr addrspace(1) %address.local1213, align 2, !dbg !56
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !56
  %address.local1215 = or disjoint i64 %address.local1202, 80, !dbg !46
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1214, i64 %address.local1215, !dbg !56
  store half %2387, ptr addrspace(1) %address.local1216, align 2, !dbg !56
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !56
  %address.local1218 = or disjoint i64 %address.local1202, 81, !dbg !46
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1217, i64 %address.local1218, !dbg !56
  store half %2386, ptr addrspace(1) %address.local1219, align 2, !dbg !56
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !56
  %address.local1221 = or disjoint i64 %address.local1202, 82, !dbg !46
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1221, !dbg !56
  store half %2385, ptr addrspace(1) %address.local1222, align 2, !dbg !56
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !56
  %address.local1224 = or disjoint i64 %address.local1202, 83, !dbg !46
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1223, i64 %address.local1224, !dbg !56
  store half %2384, ptr addrspace(1) %address.local1225, align 2, !dbg !56
  %address.local1226 = shl i64 %address.local1196, 7, !dbg !56
  %address.local1227 = add i64 %address.local1226, 8192, !dbg !56
  %address.local1228 = add nuw nsw i64 %2394, %address.local1227, !dbg !56
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !56
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1229, i64 %address.local1203, !dbg !56
  store half %2380, ptr addrspace(1) %address.local1230, align 2, !dbg !56
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !56
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1231, i64 %address.local1206, !dbg !56
  store half %2379, ptr addrspace(1) %address.local1232, align 2, !dbg !56
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !56
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1233, i64 %address.local1209, !dbg !56
  store half %2378, ptr addrspace(1) %address.local1234, align 2, !dbg !56
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !56
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1235, i64 %address.local1212, !dbg !56
  store half %2377, ptr addrspace(1) %address.local1236, align 2, !dbg !56
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !56
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1237, i64 %address.local1215, !dbg !56
  store half %2374, ptr addrspace(1) %address.local1238, align 2, !dbg !56
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !56
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1239, i64 %address.local1218, !dbg !56
  store half %2373, ptr addrspace(1) %address.local1240, align 2, !dbg !56
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !56
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1241, i64 %address.local1221, !dbg !56
  store half %2372, ptr addrspace(1) %address.local1242, align 2, !dbg !56
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !56
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1243, i64 %address.local1224, !dbg !56
  store half %2371, ptr addrspace(1) %address.local1244, align 2, !dbg !56
  br label %2396, !dbg !56

2396:                                             ; preds = %2367, %2366
  br i1 true, label %2397, label %2426, !dbg !56

2397:                                             ; preds = %2396
  %2398 = extractvalue [2 x <8 x float>] %60, 1, !dbg !54
  %2399 = shufflevector <8 x float> %2398, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2304, !dbg !54
  %normalized85 = fmul <4 x float> %2399, %row.reciprocal84, !dbg !54
  %2400 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !55
  %2401 = extractelement <4 x half> %2400, i64 3, !dbg !55
  %2402 = extractelement <4 x half> %2400, i64 2, !dbg !55
  %2403 = extractelement <4 x half> %2400, i64 1, !dbg !55
  %2404 = extractelement <4 x half> %2400, i64 0, !dbg !55
  %2405 = shufflevector <8 x float> %2398, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %normalized86 = fmul <4 x float> %2405, %row.reciprocal84, !dbg !54
  %2406 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !55
  %2407 = extractelement <4 x half> %2406, i64 3, !dbg !55
  %2408 = extractelement <4 x half> %2406, i64 2, !dbg !55
  %2409 = extractelement <4 x half> %2406, i64 1, !dbg !55
  %2410 = extractelement <4 x half> %2406, i64 0, !dbg !55
  %2411 = extractvalue [2 x <8 x float>] %60, 0, !dbg !54
  %2412 = shufflevector <8 x float> %2411, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2302, !dbg !54
  %normalized88 = fmul <4 x float> %2412, %row.reciprocal87, !dbg !54
  %2413 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !55
  %2414 = extractelement <4 x half> %2413, i64 3, !dbg !55
  %2415 = extractelement <4 x half> %2413, i64 2, !dbg !55
  %2416 = extractelement <4 x half> %2413, i64 1, !dbg !55
  %2417 = extractelement <4 x half> %2413, i64 0, !dbg !55
  %2418 = shufflevector <8 x float> %2411, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %normalized89 = fmul <4 x float> %2418, %row.reciprocal87, !dbg !54
  %2419 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !55
  %2420 = extractelement <4 x half> %2419, i64 3, !dbg !55
  %2421 = extractelement <4 x half> %2419, i64 2, !dbg !55
  %2422 = extractelement <4 x half> %2419, i64 1, !dbg !55
  %2423 = extractelement <4 x half> %2419, i64 0, !dbg !55
  %2424 = shl nuw nsw i64 %27, 19, !dbg !56
  %address.thread1245 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !56, !range !73, !frisk.register_lifetimes !37
  %address.local1246 = zext nneg i32 %address.thread1245 to i64, !dbg !45
  %address.local1247 = add i64 %28, %address.local1246, !dbg !56
  %address.local1248 = lshr i64 %address.local1246, 2, !dbg !46
  %address.local1249 = and i64 %address.local1248, 48, !dbg !46
  %address.local1250 = add i64 %address.local1247, %address.local1249, !dbg !56
  %address.local1251 = and i64 %address.local1246, 240, !dbg !56
  %address.local1253 = sub i64 %address.local1250, %address.local1251, !dbg !56
  %address.local1254 = add i64 %address.local1253, %29, !dbg !56
  %address.local1255 = shl nuw nsw i64 %address.local1254, 7, !dbg !56
  %address.local1256 = add nuw nsw i64 %2424, %address.local1255, !dbg !56
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !56
  %2425 = lshr i64 %address.local1246, 2, !dbg !45
  %address.local1260 = and i64 %2425, 12, !dbg !45
  %address.local1261 = or disjoint i64 %address.local1260, 96, !dbg !46
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1261, !dbg !56
  store half %2423, ptr addrspace(1) %address.local1262, align 2, !dbg !56
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !56
  %address.local1264 = or disjoint i64 %address.local1260, 97, !dbg !46
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1264, !dbg !56
  store half %2422, ptr addrspace(1) %address.local1265, align 2, !dbg !56
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !56
  %address.local1267 = or disjoint i64 %address.local1260, 98, !dbg !46
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1266, i64 %address.local1267, !dbg !56
  store half %2421, ptr addrspace(1) %address.local1268, align 2, !dbg !56
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !56
  %address.local1270 = or disjoint i64 %address.local1260, 99, !dbg !46
  %address.local1271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1269, i64 %address.local1270, !dbg !56
  store half %2420, ptr addrspace(1) %address.local1271, align 2, !dbg !56
  %address.local1272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !56
  %address.local1273 = or disjoint i64 %address.local1260, 112, !dbg !46
  %address.local1274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1272, i64 %address.local1273, !dbg !56
  store half %2417, ptr addrspace(1) %address.local1274, align 2, !dbg !56
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !56
  %address.local1276 = or disjoint i64 %address.local1260, 113, !dbg !46
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1275, i64 %address.local1276, !dbg !56
  store half %2416, ptr addrspace(1) %address.local1277, align 2, !dbg !56
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !56
  %address.local1279 = or disjoint i64 %address.local1260, 114, !dbg !46
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1278, i64 %address.local1279, !dbg !56
  store half %2415, ptr addrspace(1) %address.local1280, align 2, !dbg !56
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !56
  %address.local1282 = or i64 %2425, 115, !dbg !46
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1281, i64 %address.local1282, !dbg !56
  store half %2414, ptr addrspace(1) %address.local1283, align 2, !dbg !56
  %address.local1284 = shl i64 %address.local1254, 7, !dbg !56
  %address.local1285 = add i64 %address.local1284, 8192, !dbg !56
  %address.local1286 = add nuw nsw i64 %2424, %address.local1285, !dbg !56
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !56
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1287, i64 %address.local1261, !dbg !56
  store half %2410, ptr addrspace(1) %address.local1288, align 2, !dbg !56
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !56
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1264, !dbg !56
  store half %2409, ptr addrspace(1) %address.local1290, align 2, !dbg !56
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !56
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1291, i64 %address.local1267, !dbg !56
  store half %2408, ptr addrspace(1) %address.local1292, align 2, !dbg !56
  %address.local1293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !56
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1293, i64 %address.local1270, !dbg !56
  store half %2407, ptr addrspace(1) %address.local1294, align 2, !dbg !56
  %address.local1295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !56
  %address.local1296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1295, i64 %address.local1273, !dbg !56
  store half %2404, ptr addrspace(1) %address.local1296, align 2, !dbg !56
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !56
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1297, i64 %address.local1276, !dbg !56
  store half %2403, ptr addrspace(1) %address.local1298, align 2, !dbg !56
  %address.local1299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !56
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1299, i64 %address.local1279, !dbg !56
  store half %2402, ptr addrspace(1) %address.local1300, align 2, !dbg !56
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !56
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1282, !dbg !56
  store half %2401, ptr addrspace(1) %address.local1302, align 2, !dbg !56
  br label %2426, !dbg !56

2426:                                             ; preds = %2397, %2396
  ret void, !dbg !41
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

!llvm.dbg.cu = !{!39}
!llvm.module.flags = !{!72}

!36 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000001")
!37 = !{}
!38 = !DISubroutineType(types: !37)
!39 = distinct !DICompileUnit(language: DW_LANG_C, file: !36, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!40 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !36, file: !36, line: 23, type: !38, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !39)
!41 = !DILocation(line: 23, column: 3, scope: !40)
!42 = !DILocation(line: 60, column: 24, scope: !40)
!43 = !DILocation(line: 48, column: 17, scope: !40)
!44 = !DILocation(line: 70, column: 22, scope: !40)
!45 = !DILocation(line: 72, column: 24, scope: !40)
!46 = !DILocation(line: 63, column: 17, scope: !40)
!47 = !DILocation(line: 77, column: 19, scope: !40)
!48 = !DILocation(line: 92, column: 20, scope: !40)
!49 = !DILocation(line: 102, column: 23, scope: !40)
!50 = !DILocation(line: 0, scope: !40)
!51 = !DILocation(line: 71, column: 22, scope: !40)
!52 = !DILocation(line: 73, column: 26, scope: !40)
!53 = !DILocation(line: 101, column: 20, scope: !40)
!54 = !DILocation(line: 107, column: 23, scope: !40)
!55 = !DILocation(line: 108, column: 19, scope: !40)
!56 = !DILocation(line: 109, column: 9, scope: !40)
!57 = !DILocation(line: 74, column: 23, scope: !40)
!58 = !DILocation(line: 79, column: 24, scope: !40)
!59 = !DILocation(line: 80, column: 23, scope: !40)
!60 = !DILocation(line: 81, column: 26, scope: !40)
!61 = !DILocation(line: 88, column: 21, scope: !40)
!62 = !DILocation(line: 89, column: 20, scope: !40)
!63 = !DILocation(line: 90, column: 22, scope: !40)
!64 = !DILocation(line: 91, column: 23, scope: !40)
!65 = !DILocation(line: 93, column: 22, scope: !40)
!66 = !DILocation(line: 94, column: 22, scope: !40)
!67 = !DILocation(line: 95, column: 20, scope: !40)
!68 = !DILocation(line: 96, column: 16, scope: !40)
!69 = !DILocation(line: 103, column: 22, scope: !40)
!70 = !DILocation(line: 104, column: 20, scope: !40)
!71 = !{i32 256, i32 1, i32 1}
!72 = !{i32 2, !"Debug Info Version", i32 3}
!73 = !{i32 0, i32 256}
