; ModuleID = 'generated_ir/gemm_result_layout_comparison/spill_analysis/cfg_simplified.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000001/input.mlir"

@shm_0 = addrspace(3) global [65536 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !41 !reqd_work_group_size !44 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !45
  %6 = sext i32 %5 to i64, !dbg !45
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !45
  %8 = sext i32 %7 to i64, !dbg !45
  %9 = shl nsw i64 %8, 5, !dbg !45
  %10 = add nsw i64 %9, %6, !dbg !45
  %11 = lshr i64 %10, 5, !dbg !45
  %12 = and i64 %11, 576460752303423484, !dbg !45
  %13 = and i64 %6, 3, !dbg !45
  %14 = or disjoint i64 %12, %13, !dbg !45
  %15 = lshr i64 %10, 2, !dbg !45
  %16 = sub nsw i64 31, %14, !dbg !45
  %17 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !45
  %18 = sext i32 %17 to i64, !dbg !45
  %19 = shl i64 %16, 7, !dbg !46
  %20 = icmp ugt i64 %10, 1023, !dbg !47
  %21 = add nsw i64 %14, -32, !dbg !47
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !47
  %23 = sdiv i64 %22, 32, !dbg !47
  %24 = sext i1 %20 to i64, !dbg !47
  %25 = xor i64 %23, %24, !dbg !47
  %26 = add nsw i64 %15, %25, !dbg !47
  %27 = and i64 %26, 31, !dbg !47
  %28 = shl nsw i64 %16, 7, !dbg !47
  %29 = mul nsw i64 %25, -4096, !dbg !47
  %30 = add i64 %28, %29, !dbg !47
  %31 = shl nuw nsw i64 %27, 19, !dbg !47
  %32 = shl nsw i64 %30, 7, !dbg !47
  %33 = add i64 %31, %32, !dbg !47
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !48, !range !49
  %35 = zext nneg i32 %34 to i64, !dbg !48
  %36 = lshr i64 %35, 4, !dbg !48
  %37 = and i64 %36, 3, !dbg !48
  %38 = shl nuw nsw i64 %37, 2, !dbg !48
  %39 = and i64 %35, 240, !dbg !50
  %40 = or disjoint i64 %19, 127, !dbg !50
  %41 = add i64 %19, 128, !dbg !50
  %42 = icmp uge i64 %41, %40, !dbg !50
  %43 = icmp eq i64 %37, 0, !dbg !51
  %44 = and i32 %34, 16, !dbg !52
  %.not = icmp eq i32 %44, 0, !dbg !52
  %45 = and i32 %34, 32, !dbg !52
  %.not1 = icmp eq i32 %45, 0, !dbg !52
  %46 = sub nsw i64 32, %14, !dbg !53
  %47 = shl i64 %15, 7, !dbg !54
  %48 = and i64 %47, 3968, !dbg !54
  %49 = shl nsw i64 %18, 12, !dbg !54
  %50 = or disjoint i64 %48, %49, !dbg !54
  %51 = lshr i64 %35, 2, !dbg !55
  %52 = and i64 %51, 48, !dbg !55
  br label %53, !dbg !56

53:                                               ; preds = %2172, %4
  %54 = phi i64 [ %2231, %2172 ], [ 0, %4 ], !dbg !56
  %55 = phi [2 x <8 x float>] [ %2212, %2172 ], [ zeroinitializer, %4 ], !dbg !56
  %56 = phi [2 x <1 x float>] [ %2216, %2172 ], [ zeroinitializer, %4 ], !dbg !56
  %57 = phi [2 x <1 x float>] [ %2218, %2172 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !56
  %58 = phi [2 x <8 x float>] [ %2222, %2172 ], [ zeroinitializer, %4 ], !dbg !56
  %59 = phi [2 x <8 x float>] [ %2226, %2172 ], [ zeroinitializer, %4 ], !dbg !56
  %60 = phi [2 x <8 x float>] [ %2230, %2172 ], [ zeroinitializer, %4 ], !dbg !56
  %61 = icmp slt i64 %54, %46, !dbg !56
  br i1 %61, label %62, label %2232, !dbg !56

62:                                               ; preds = %53
  %63 = shl nsw i64 %54, 7, !dbg !56
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !47, !range !49, !frisk.register_lifetimes !43
  %address.local = zext nneg i32 %address.thread to i64, !dbg !48
  %address.local258 = shl nuw nsw i64 %address.local, 3, !dbg !47
  %address.local259 = and i64 %address.local258, 1536, !dbg !47
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local259, !dbg !47
  %address.local261 = ptrtoint ptr addrspace(3) %address.local260 to i64, !dbg !47
  %address.local262 = trunc i64 %address.local261 to i32, !dbg !47
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !47
  %address.local263 = shl nuw nsw i64 %address.local, 7, !dbg !47
  %address.local264 = and i64 %address.local263, 16256, !dbg !47
  %address.local265 = lshr i64 %address.local, 4, !dbg !47
  %address.local266 = and i64 %address.local265, 8, !dbg !47
  %address.local267 = or disjoint i64 %address.local264, %address.local266, !dbg !47
  %65 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !47
  %.idx1 = or disjoint i64 %33, %address.local267, !dbg !47
  %.idx1.tr = trunc i64 %.idx1 to i32, !dbg !47
  %dma.buffer.offset1821 = shl i32 %.idx1.tr, 1, !dbg !47
  %66 = insertelement <2 x i64> poison, i64 %65, i64 0, !dbg !47
  %67 = bitcast <2 x i64> %66 to <4 x i32>, !dbg !47
  %dma.buffer.rsrc1822 = shufflevector <4 x i32> %67, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !47
  %dma.buffer.lds1823 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64), !dbg !47
  %68 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1821, <4 x i32> %dma.buffer.rsrc1822) #10, !dbg !47
  %address.local269 = or disjoint i64 %address.local, 256, !dbg !47
  %address.local270 = shl nuw nsw i64 %address.local269, 3, !dbg !47
  %address.local271 = and i64 %address.local270, 3584, !dbg !47
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local271, !dbg !47
  %address.local273 = ptrtoint ptr addrspace(3) %address.local272 to i64, !dbg !47
  %address.local274 = trunc i64 %address.local273 to i32, !dbg !47
  %69 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !47
  %address.local275 = shl nuw nsw i64 %address.local, 7, !dbg !47
  %address.local276 = and i64 %address.local275, 16256, !dbg !47
  %address.local277 = lshr i64 %address.local269, 4, !dbg !47
  %address.local278 = and i64 %address.local277, 24, !dbg !47
  %address.local279 = or disjoint i64 %address.local276, %address.local278, !dbg !47
  %70 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !47
  %.idx2 = or disjoint i64 %33, %address.local279, !dbg !47
  %.idx2.tr = trunc i64 %.idx2 to i32, !dbg !47
  %dma.buffer.offset1818 = shl i32 %.idx2.tr, 1, !dbg !47
  %71 = insertelement <2 x i64> poison, i64 %70, i64 0, !dbg !47
  %72 = bitcast <2 x i64> %71 to <4 x i32>, !dbg !47
  %dma.buffer.rsrc1819 = shufflevector <4 x i32> %72, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !47
  %dma.buffer.lds1820 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %69), !dbg !47
  %73 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1818, <4 x i32> %dma.buffer.rsrc1819) #10, !dbg !47
  %74 = lshr i64 %54, 12, !dbg !54
  %75 = add nuw nsw i64 %74, %15, !dbg !54
  %76 = lshr i64 %54, 5, !dbg !54
  %77 = and i64 %76, 127, !dbg !54
  %78 = shl i64 %75, 19, !dbg !54
  %79 = and i64 %78, 16252928, !dbg !54
  %80 = shl i64 %76, 19, !dbg !54
  %81 = shl i64 %54, 14, !dbg !54
  %82 = sub i64 %81, %80, !dbg !54
  %83 = getelementptr inbounds nuw i8, ptr addrspace(3) @shm_0, i64 8192, !dbg !57
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %83, i64 %address.local259, !dbg !54
  %address.local282 = ptrtoint ptr addrspace(3) %address.local281 to i64, !dbg !54
  %address.local283 = trunc i64 %address.local282 to i32, !dbg !54
  %84 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !54
  %address.local284 = lshr i64 %address.local, 4, !dbg !54
  %address.local285 = and i64 %address.local284, 8, !dbg !54
  %address.local286 = and i64 %address.local, 123, !dbg !54
  %address.local287 = lshr i64 %address.local, 2, !dbg !54
  %address.local288 = and i64 %address.local287, 4, !dbg !54
  %address.local289 = add nuw nsw i64 %address.local288, %address.local, !dbg !54
  %address.local290 = and i64 %address.local289, 4, !dbg !54
  %address.local291 = or disjoint i64 %address.local286, %address.local290, !dbg !54
  %address.local292 = shl nuw nsw i64 %address.local291, 7, !dbg !54
  %address.local293 = or disjoint i64 %address.local285, %address.local292, !dbg !54
  %85 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !54
  %.idx5 = or disjoint i64 %79, %77, !dbg !54
  %86 = add i64 %.idx5, %82, !dbg !54
  %87 = add i64 %86, %address.local293, !dbg !54
  %.tr = trunc i64 %87 to i32, !dbg !54
  %dma.buffer.offset1815 = shl i32 %.tr, 1, !dbg !54
  %88 = insertelement <2 x i64> poison, i64 %85, i64 0, !dbg !54
  %89 = bitcast <2 x i64> %88 to <4 x i32>, !dbg !54
  %dma.buffer.rsrc1816 = shufflevector <4 x i32> %89, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !54
  %dma.buffer.lds1817 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !54
  %90 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1815, <4 x i32> %dma.buffer.rsrc1816) #10, !dbg !54
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) %83, i64 %address.local271, !dbg !54
  %address.local296 = ptrtoint ptr addrspace(3) %address.local295 to i64, !dbg !54
  %address.local297 = trunc i64 %address.local296 to i32, !dbg !54
  %91 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !54
  %address.local298 = lshr i64 %address.local269, 4, !dbg !54
  %address.local299 = and i64 %address.local298, 24, !dbg !54
  %address.local300 = and i64 %address.local, 123, !dbg !54
  %address.local301 = lshr i64 %address.local, 2, !dbg !54
  %address.local302 = and i64 %address.local301, 4, !dbg !54
  %address.local303 = add nuw nsw i64 %address.local302, %address.local, !dbg !54
  %address.local304 = and i64 %address.local303, 4, !dbg !54
  %address.local305 = or disjoint i64 %address.local300, %address.local304, !dbg !54
  %address.local306 = shl nuw nsw i64 %address.local305, 7, !dbg !54
  %address.local307 = or disjoint i64 %address.local299, %address.local306, !dbg !54
  %92 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !54
  %.idx8 = or disjoint i64 %79, %77, !dbg !54
  %93 = add i64 %.idx8, %82, !dbg !54
  %94 = add i64 %93, %address.local307, !dbg !54
  %.tr9 = trunc i64 %94 to i32, !dbg !54
  %dma.buffer.offset1812 = shl i32 %.tr9, 1, !dbg !54
  %95 = insertelement <2 x i64> poison, i64 %92, i64 0, !dbg !54
  %96 = bitcast <2 x i64> %95 to <4 x i32>, !dbg !54
  %dma.buffer.rsrc1813 = shufflevector <4 x i32> %96, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !54
  %dma.buffer.lds1814 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %91), !dbg !54
  %97 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1812, <4 x i32> %dma.buffer.rsrc1813) #10, !dbg !54
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !47
  fence syncscope("workgroup") release, !dbg !47
  call void @llvm.amdgcn.s.barrier(), !dbg !47
  fence syncscope("workgroup") acquire, !dbg !47
  %address.local309 = lshr i32 %address.thread, 2, !dbg !48
  %address.local310 = and i32 %address.local309, 4, !dbg !48
  %address.local311 = shl nuw nsw i32 %address.thread, 1, !dbg !48
  %address.local312 = and i32 %address.local311, 384, !dbg !48
  %address.local313 = or disjoint i32 %address.local310, %address.local312, !dbg !48
  %address.local314 = shl nuw nsw i32 %address.thread, 3, !dbg !48
  %address.local315 = and i32 %address.local314, 120, !dbg !48
  %address.local316 = or disjoint i32 %address.local313, %address.local315, !dbg !48
  %address.local317 = shl nuw nsw i32 %address.thread, 5, !dbg !48
  %address.local318 = and i32 %address.local317, 1024, !dbg !48
  %address.local319 = or disjoint i32 %address.local316, %address.local318, !dbg !48
  %address.local320 = zext nneg i32 %address.local319 to i64, !dbg !48
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local320, !dbg !48
  %98 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !48
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !48
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !48
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !48
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !48
  %101 = fmul <4 x half> %98, splat (half 0xH3015), !dbg !48
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !48
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !48
  %103 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !48
  %104 = fmul <4 x half> %100, splat (half 0xH3015), !dbg !48
  %105 = fmul <4 x half> %102, splat (half 0xH3015), !dbg !48
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !54
  fence syncscope("workgroup") release, !dbg !54
  call void @llvm.amdgcn.s.barrier(), !dbg !54
  fence syncscope("workgroup") acquire, !dbg !54
  %address.local328 = lshr i32 %address.thread, 2, !dbg !57
  %address.local329 = and i32 %address.local328, 4, !dbg !57
  %address.local330 = shl nuw nsw i32 %address.thread, 3, !dbg !57
  %address.local331 = and i32 %address.local330, 120, !dbg !57
  %address.local332 = or disjoint i32 %address.local329, %address.local331, !dbg !57
  %address.local333 = shl nuw nsw i32 %address.thread, 5, !dbg !57
  %address.local334 = and i32 %address.local333, 1024, !dbg !57
  %address.local335 = or disjoint i32 %address.local332, %address.local334, !dbg !57
  %address.local336 = zext nneg i32 %address.local335 to i64, !dbg !57
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) %83, i64 %address.local336, !dbg !57
  %106 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !57
  %107 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %106, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1561 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !57
  %108 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !57
  %address.local349 = lshr i32 %address.thread, 2, !dbg !57
  %address.local350 = and i32 %address.local349, 4, !dbg !57
  %address.local351 = shl nuw nsw i32 %address.thread, 3, !dbg !57
  %address.local352 = and i32 %address.local351, 88, !dbg !57
  %address.local353 = or disjoint i32 %address.local350, %address.local352, !dbg !57
  %address.local354 = shl nuw nsw i32 %address.thread, 3, !dbg !57
  %address.local355 = and i32 %address.local354, 32, !dbg !57
  %address.local356 = sub nsw i32 %address.local353, %address.local355, !dbg !57
  %address.local357 = shl nuw nsw i32 %address.thread, 5, !dbg !57
  %address.local358 = and i32 %address.local357, 1024, !dbg !57
  %address.local359 = add nsw i32 %address.local356, %address.local358, !dbg !57
  %address.local360 = sext i32 %address.local359 to i64, !dbg !57
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) %83, i64 %address.local360, !dbg !57
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local361, i64 320, !dbg !57
  %109 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !57
  %110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %108, <4 x float> %107) #9, !dbg !57
  %111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %109, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1562 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !57
  %112 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !57
  %lds.const.gep1563 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !57
  %113 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !57
  %114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %112, <4 x float> %111) #9, !dbg !57
  %115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %113, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1564 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !57
  %116 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !57
  %lds.const.gep1565 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !57
  %117 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !57
  %118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %116, <4 x float> %115) #9, !dbg !57
  %119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %117, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1566 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !57
  %120 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !57
  %lds.const.gep1567 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !57
  %121 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !57
  %122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %120, <4 x float> %119) #9, !dbg !57
  %123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %121, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1568 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !57
  %124 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !57
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !57
  %125 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !57
  %126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %124, <4 x float> %123) #9, !dbg !57
  %127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %125, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !57
  %128 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !57
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !57
  %129 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !57
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %128, <4 x float> %127) #9, !dbg !57
  %131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %129, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !57
  %132 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !57
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !57
  %133 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !57
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %132, <4 x float> %131) #9, !dbg !57
  %135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %133, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !57
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !57
  %137 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !57
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %136, <4 x float> %135) #9, !dbg !57
  %139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %137, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1575 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !57
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep1575, align 8, !dbg !57
  %141 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !57
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %140, <4 x float> %139) #9, !dbg !57
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %141, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1576 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !57
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep1576, align 8, !dbg !57
  %lds.const.gep1577 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !57
  %145 = load <4 x half>, ptr addrspace(3) %lds.const.gep1577, align 8, !dbg !57
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %144, <4 x float> %143) #9, !dbg !57
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %145, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1578 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !57
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1578, align 8, !dbg !57
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !57
  %149 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !57
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %148, <4 x float> %147) #9, !dbg !57
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %149, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !57
  %152 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !57
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !57
  %153 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !57
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %152, <4 x float> %151) #9, !dbg !57
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !57
  %156 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !57
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !57
  %157 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !57
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %156, <4 x float> %155) #9, !dbg !57
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !57
  %160 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !57
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !57
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !57
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %160, <4 x float> %159) #9, !dbg !57
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %161, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !57
  %164 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !57
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !57
  %165 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !57
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %164, <4 x float> %163) #9, !dbg !57
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !57
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !57
  %168 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !57
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %168, <4 x float> %167) #9, !dbg !57
  %170 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !47
  %171 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !47
  %172 = or disjoint i64 %33, 32, !dbg !47
  %173 = or disjoint i64 %address.local267, %172, !dbg !47
  %.tr10 = trunc i64 %173 to i32, !dbg !47
  %dma.buffer.offset1809 = shl i32 %.tr10, 1, !dbg !47
  %174 = insertelement <2 x i64> poison, i64 %171, i64 0, !dbg !47
  %175 = bitcast <2 x i64> %174 to <4 x i32>, !dbg !47
  %dma.buffer.rsrc1810 = shufflevector <4 x i32> %175, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !47
  %dma.buffer.lds1811 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !47
  %176 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1811, i32 %dma.buffer.offset1809, <4 x i32> %dma.buffer.rsrc1810) #10, !dbg !47
  %177 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !47
  %178 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !47
  %179 = or disjoint i64 %33, 32, !dbg !47
  %180 = or disjoint i64 %address.local279, %179, !dbg !47
  %.tr11 = trunc i64 %180 to i32, !dbg !47
  %dma.buffer.offset1806 = shl i32 %.tr11, 1, !dbg !47
  %181 = insertelement <2 x i64> poison, i64 %178, i64 0, !dbg !47
  %182 = bitcast <2 x i64> %181 to <4 x i32>, !dbg !47
  %dma.buffer.rsrc1807 = shufflevector <4 x i32> %182, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !47
  %dma.buffer.lds1808 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %177), !dbg !47
  %183 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1806, <4 x i32> %dma.buffer.rsrc1807) #10, !dbg !47
  %184 = add nuw i64 %54, 1024, !dbg !54
  %185 = icmp slt i64 %184, 0, !dbg !54
  %186 = sub nuw i64 -1025, %54, !dbg !54
  %187 = select i1 %185, i64 %186, i64 %184, !dbg !54
  %188 = sdiv i64 %187, 4096, !dbg !54
  %.lobit = ashr i64 %184, 63, !dbg !54
  %189 = xor i64 %188, %.lobit, !dbg !54
  %190 = add nsw i64 %189, %15, !dbg !54
  %191 = add nsw i64 %76, %50, !dbg !54
  %192 = add nsw i64 %191, 32, !dbg !54
  %193 = icmp slt i64 %191, -32, !dbg !54
  %194 = sub nsw i64 -33, %191, !dbg !54
  %195 = select i1 %193, i64 %194, i64 %192, !dbg !54
  %196 = sdiv i64 %195, 128, !dbg !54
  %197 = sext i1 %193 to i64, !dbg !54
  %198 = xor i64 %196, %197, !dbg !54
  %199 = shl nsw i64 %198, 7, !dbg !54
  %200 = sub i64 %50, %199, !dbg !54
  %201 = shl i64 %190, 19, !dbg !54
  %202 = and i64 %201, 16252928, !dbg !54
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !47
  fence syncscope("workgroup") release, !dbg !47
  call void @llvm.amdgcn.s.barrier(), !dbg !47
  fence syncscope("workgroup") acquire, !dbg !47
  %203 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !54
  %204 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !54
  %.idx15 = add nuw nsw i64 %202, %76, !dbg !54
  %205 = add i64 %.idx15, %200, !dbg !54
  %206 = add i64 %205, %82, !dbg !54
  %207 = add i64 %206, %address.local293, !dbg !54
  %.tr16 = trunc i64 %207 to i32, !dbg !54
  %208 = shl i32 %.tr16, 1, !dbg !54
  %dma.buffer.offset1803 = add i32 %208, 64, !dbg !54
  %209 = insertelement <2 x i64> poison, i64 %204, i64 0, !dbg !54
  %210 = bitcast <2 x i64> %209 to <4 x i32>, !dbg !54
  %dma.buffer.rsrc1804 = shufflevector <4 x i32> %210, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !54
  %dma.buffer.lds1805 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %203), !dbg !54
  %211 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1803, <4 x i32> %dma.buffer.rsrc1804) #10, !dbg !54
  %212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !54
  %213 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !54
  %.idx20 = add nuw nsw i64 %202, %76, !dbg !54
  %214 = add i64 %.idx20, %200, !dbg !54
  %215 = add i64 %214, %82, !dbg !54
  %216 = add i64 %215, %address.local307, !dbg !54
  %.tr21 = trunc i64 %216 to i32, !dbg !54
  %217 = shl i32 %.tr21, 1, !dbg !54
  %dma.buffer.offset1800 = add i32 %217, 64, !dbg !54
  %218 = insertelement <2 x i64> poison, i64 %213, i64 0, !dbg !54
  %219 = bitcast <2 x i64> %218 to <4 x i32>, !dbg !54
  %dma.buffer.rsrc1801 = shufflevector <4 x i32> %219, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !54
  %dma.buffer.lds1802 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %212), !dbg !54
  %220 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1800, <4 x i32> %dma.buffer.rsrc1801) #10, !dbg !54
  %221 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !48
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !48
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !48
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !48
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !48
  %224 = fmul <4 x half> %221, splat (half 0xH3015), !dbg !48
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !48
  %225 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !48
  %226 = fmul <4 x half> %222, splat (half 0xH3015), !dbg !48
  %227 = fmul <4 x half> %223, splat (half 0xH3015), !dbg !48
  %228 = fmul <4 x half> %225, splat (half 0xH3015), !dbg !48
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !54
  fence syncscope("workgroup") release, !dbg !54
  call void @llvm.amdgcn.s.barrier(), !dbg !54
  fence syncscope("workgroup") acquire, !dbg !54
  %229 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !57
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %229, <4 x float> %110) #9, !dbg !57
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !57
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !57
  %232 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !57
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %231, <4 x float> %230) #9, !dbg !57
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %232, <4 x float> %114) #9, !dbg !57
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !57
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !57
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !57
  %236 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !57
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %235, <4 x float> %234) #9, !dbg !57
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %236, <4 x float> %118) #9, !dbg !57
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !57
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !57
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !57
  %240 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !57
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %239, <4 x float> %238) #9, !dbg !57
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %240, <4 x float> %122) #9, !dbg !57
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !57
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !57
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !57
  %244 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !57
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %243, <4 x float> %242) #9, !dbg !57
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %244, <4 x float> %126) #9, !dbg !57
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !57
  %247 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !57
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !57
  %248 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !57
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %247, <4 x float> %246) #9, !dbg !57
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %248, <4 x float> %130) #9, !dbg !57
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !57
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !57
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !57
  %252 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !57
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %251, <4 x float> %250) #9, !dbg !57
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %252, <4 x float> %134) #9, !dbg !57
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !57
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !57
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !57
  %256 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !57
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %255, <4 x float> %254) #9, !dbg !57
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %256, <4 x float> %138) #9, !dbg !57
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !57
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !57
  %260 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !57
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %259, <4 x float> %258) #9, !dbg !57
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %260, <4 x float> %142) #9, !dbg !57
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !57
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !57
  %264 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !57
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %263, <4 x float> %262) #9, !dbg !57
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %264, <4 x float> %146) #9, !dbg !57
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !57
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !57
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !57
  %268 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !57
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %267, <4 x float> %266) #9, !dbg !57
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %268, <4 x float> %150) #9, !dbg !57
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !57
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !57
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !57
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !57
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %271, <4 x float> %270) #9, !dbg !57
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %272, <4 x float> %154) #9, !dbg !57
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !57
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !57
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !57
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !57
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %275, <4 x float> %274) #9, !dbg !57
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %276, <4 x float> %158) #9, !dbg !57
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !57
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !57
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !57
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !57
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %279, <4 x float> %278) #9, !dbg !57
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %280, <4 x float> %162) #9, !dbg !57
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !57
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !57
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !57
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !57
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %283, <4 x float> %282) #9, !dbg !57
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %284, <4 x float> %166) #9, !dbg !57
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !57
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !57
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !57
  %288 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !57
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %287, <4 x float> %286) #9, !dbg !57
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %288, <4 x float> %169) #9, !dbg !57
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !57
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !57
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %291, <4 x float> %290) #9, !dbg !57
  %293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !47
  %294 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !47
  %295 = or disjoint i64 %33, 64, !dbg !47
  %296 = or disjoint i64 %address.local267, %295, !dbg !47
  %.tr22 = trunc i64 %296 to i32, !dbg !47
  %dma.buffer.offset1797 = shl i32 %.tr22, 1, !dbg !47
  %297 = insertelement <2 x i64> poison, i64 %294, i64 0, !dbg !47
  %298 = bitcast <2 x i64> %297 to <4 x i32>, !dbg !47
  %dma.buffer.rsrc1798 = shufflevector <4 x i32> %298, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !47
  %dma.buffer.lds1799 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %293), !dbg !47
  %299 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1797, <4 x i32> %dma.buffer.rsrc1798) #10, !dbg !47
  %300 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !47
  %301 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !47
  %302 = or disjoint i64 %33, 64, !dbg !47
  %303 = or disjoint i64 %address.local279, %302, !dbg !47
  %.tr23 = trunc i64 %303 to i32, !dbg !47
  %dma.buffer.offset1794 = shl i32 %.tr23, 1, !dbg !47
  %304 = insertelement <2 x i64> poison, i64 %301, i64 0, !dbg !47
  %305 = bitcast <2 x i64> %304 to <4 x i32>, !dbg !47
  %dma.buffer.rsrc1795 = shufflevector <4 x i32> %305, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !47
  %dma.buffer.lds1796 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %300), !dbg !47
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1796, i32 %dma.buffer.offset1794, <4 x i32> %dma.buffer.rsrc1795) #10, !dbg !47
  %307 = add nuw i64 %54, 2048, !dbg !54
  %308 = icmp slt i64 %307, 0, !dbg !54
  %309 = sub nuw i64 -2049, %54, !dbg !54
  %310 = select i1 %308, i64 %309, i64 %307, !dbg !54
  %311 = sdiv i64 %310, 4096, !dbg !54
  %.lobit2 = ashr i64 %307, 63, !dbg !54
  %312 = xor i64 %311, %.lobit2, !dbg !54
  %313 = add nsw i64 %312, %15, !dbg !54
  %314 = add nsw i64 %191, 64, !dbg !54
  %315 = icmp slt i64 %191, -64, !dbg !54
  %316 = sub nsw i64 -65, %191, !dbg !54
  %317 = select i1 %315, i64 %316, i64 %314, !dbg !54
  %318 = sdiv i64 %317, 128, !dbg !54
  %319 = sext i1 %315 to i64, !dbg !54
  %320 = xor i64 %318, %319, !dbg !54
  %321 = shl nsw i64 %320, 7, !dbg !54
  %322 = sub i64 %50, %321, !dbg !54
  %323 = shl i64 %313, 19, !dbg !54
  %324 = and i64 %323, 16252928, !dbg !54
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !47
  fence syncscope("workgroup") release, !dbg !47
  call void @llvm.amdgcn.s.barrier(), !dbg !47
  fence syncscope("workgroup") acquire, !dbg !47
  %325 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !54
  %326 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !54
  %.idx27 = add nuw nsw i64 %324, %76, !dbg !54
  %327 = add i64 %.idx27, %322, !dbg !54
  %328 = add i64 %327, %82, !dbg !54
  %329 = add i64 %328, %address.local293, !dbg !54
  %.tr28 = trunc i64 %329 to i32, !dbg !54
  %330 = shl i32 %.tr28, 1, !dbg !54
  %dma.buffer.offset1791 = add i32 %330, 128, !dbg !54
  %331 = insertelement <2 x i64> poison, i64 %326, i64 0, !dbg !54
  %332 = bitcast <2 x i64> %331 to <4 x i32>, !dbg !54
  %dma.buffer.rsrc1792 = shufflevector <4 x i32> %332, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !54
  %dma.buffer.lds1793 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %325), !dbg !54
  %333 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1793, i32 %dma.buffer.offset1791, <4 x i32> %dma.buffer.rsrc1792) #10, !dbg !54
  %334 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !54
  %335 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !54
  %.idx32 = add nuw nsw i64 %324, %76, !dbg !54
  %336 = add i64 %.idx32, %322, !dbg !54
  %337 = add i64 %336, %82, !dbg !54
  %338 = add i64 %337, %address.local307, !dbg !54
  %.tr33 = trunc i64 %338 to i32, !dbg !54
  %339 = shl i32 %.tr33, 1, !dbg !54
  %dma.buffer.offset1788 = add i32 %339, 128, !dbg !54
  %340 = insertelement <2 x i64> poison, i64 %335, i64 0, !dbg !54
  %341 = bitcast <2 x i64> %340 to <4 x i32>, !dbg !54
  %dma.buffer.rsrc1789 = shufflevector <4 x i32> %341, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !54
  %dma.buffer.lds1790 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %334), !dbg !54
  %342 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1790, i32 %dma.buffer.offset1788, <4 x i32> %dma.buffer.rsrc1789) #10, !dbg !54
  %343 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !48
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !48
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !48
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !48
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !48
  %346 = fmul <4 x half> %343, splat (half 0xH3015), !dbg !48
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !48
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !48
  %348 = fmul <4 x half> %344, splat (half 0xH3015), !dbg !48
  %349 = fmul <4 x half> %345, splat (half 0xH3015), !dbg !48
  %350 = fmul <4 x half> %347, splat (half 0xH3015), !dbg !48
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !54
  fence syncscope("workgroup") release, !dbg !54
  call void @llvm.amdgcn.s.barrier(), !dbg !54
  fence syncscope("workgroup") acquire, !dbg !54
  %351 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !57
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %351, <4 x float> %233) #9, !dbg !57
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !57
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !57
  %354 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !57
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %353, <4 x float> %352) #9, !dbg !57
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %354, <4 x float> %237) #9, !dbg !57
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !57
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !57
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !57
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !57
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %357, <4 x float> %356) #9, !dbg !57
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %358, <4 x float> %241) #9, !dbg !57
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !57
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !57
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !57
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !57
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %361, <4 x float> %360) #9, !dbg !57
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %362, <4 x float> %245) #9, !dbg !57
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !57
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !57
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !57
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !57
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %365, <4 x float> %364) #9, !dbg !57
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %366, <4 x float> %249) #9, !dbg !57
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !57
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !57
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !57
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !57
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %369, <4 x float> %368) #9, !dbg !57
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %370, <4 x float> %253) #9, !dbg !57
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !57
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !57
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !57
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !57
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %373, <4 x float> %372) #9, !dbg !57
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %374, <4 x float> %257) #9, !dbg !57
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !57
  %377 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !57
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !57
  %378 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !57
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %377, <4 x float> %376) #9, !dbg !57
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %378, <4 x float> %261) #9, !dbg !57
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !57
  %381 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !57
  %382 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !57
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %381, <4 x float> %380) #9, !dbg !57
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %382, <4 x float> %265) #9, !dbg !57
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !57
  %385 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !57
  %386 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !57
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %385, <4 x float> %384) #9, !dbg !57
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %386, <4 x float> %269) #9, !dbg !57
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !57
  %389 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !57
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !57
  %390 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !57
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %389, <4 x float> %388) #9, !dbg !57
  %392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %390, <4 x float> %273) #9, !dbg !57
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !57
  %393 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !57
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !57
  %394 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !57
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %393, <4 x float> %392) #9, !dbg !57
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %394, <4 x float> %277) #9, !dbg !57
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !57
  %397 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !57
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !57
  %398 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !57
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %397, <4 x float> %396) #9, !dbg !57
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %398, <4 x float> %281) #9, !dbg !57
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !57
  %401 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !57
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !57
  %402 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !57
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %401, <4 x float> %400) #9, !dbg !57
  %404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %402, <4 x float> %285) #9, !dbg !57
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !57
  %405 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !57
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !57
  %406 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !57
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %405, <4 x float> %404) #9, !dbg !57
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %406, <4 x float> %289) #9, !dbg !57
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !57
  %409 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !57
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !57
  %410 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !57
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %409, <4 x float> %408) #9, !dbg !57
  %412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %410, <4 x float> %292) #9, !dbg !57
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !57
  %413 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !57
  %414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %413, <4 x float> %412) #9, !dbg !57
  %415 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !47
  %416 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !47
  %417 = or disjoint i64 %33, 96, !dbg !47
  %418 = or disjoint i64 %address.local267, %417, !dbg !47
  %.tr34 = trunc i64 %418 to i32, !dbg !47
  %dma.buffer.offset1785 = shl i32 %.tr34, 1, !dbg !47
  %419 = insertelement <2 x i64> poison, i64 %416, i64 0, !dbg !47
  %420 = bitcast <2 x i64> %419 to <4 x i32>, !dbg !47
  %dma.buffer.rsrc1786 = shufflevector <4 x i32> %420, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !47
  %dma.buffer.lds1787 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %415), !dbg !47
  %421 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1787, i32 %dma.buffer.offset1785, <4 x i32> %dma.buffer.rsrc1786) #10, !dbg !47
  %422 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !47
  %423 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !47
  %424 = or disjoint i64 %33, 96, !dbg !47
  %425 = or disjoint i64 %address.local279, %424, !dbg !47
  %.tr35 = trunc i64 %425 to i32, !dbg !47
  %dma.buffer.offset1782 = shl i32 %.tr35, 1, !dbg !47
  %426 = insertelement <2 x i64> poison, i64 %423, i64 0, !dbg !47
  %427 = bitcast <2 x i64> %426 to <4 x i32>, !dbg !47
  %dma.buffer.rsrc1783 = shufflevector <4 x i32> %427, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !47
  %dma.buffer.lds1784 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %422), !dbg !47
  %428 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1784, i32 %dma.buffer.offset1782, <4 x i32> %dma.buffer.rsrc1783) #10, !dbg !47
  %429 = add nuw i64 %54, 3072, !dbg !54
  %430 = icmp slt i64 %429, 0, !dbg !54
  %431 = sub nuw i64 -3073, %54, !dbg !54
  %432 = select i1 %430, i64 %431, i64 %429, !dbg !54
  %433 = sdiv i64 %432, 4096, !dbg !54
  %.lobit3 = ashr i64 %429, 63, !dbg !54
  %434 = xor i64 %433, %.lobit3, !dbg !54
  %435 = add nsw i64 %434, %15, !dbg !54
  %436 = add nsw i64 %191, 96, !dbg !54
  %437 = icmp slt i64 %191, -96, !dbg !54
  %438 = sub nsw i64 -97, %191, !dbg !54
  %439 = select i1 %437, i64 %438, i64 %436, !dbg !54
  %440 = sdiv i64 %439, 128, !dbg !54
  %441 = sext i1 %437 to i64, !dbg !54
  %442 = xor i64 %440, %441, !dbg !54
  %443 = shl nsw i64 %442, 7, !dbg !54
  %444 = sub i64 %50, %443, !dbg !54
  %445 = shl i64 %435, 19, !dbg !54
  %446 = and i64 %445, 16252928, !dbg !54
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !47
  fence syncscope("workgroup") release, !dbg !47
  call void @llvm.amdgcn.s.barrier(), !dbg !47
  fence syncscope("workgroup") acquire, !dbg !47
  %447 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !54
  %448 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !54
  %.idx39 = add nuw nsw i64 %446, %76, !dbg !54
  %449 = add i64 %.idx39, %444, !dbg !54
  %450 = add i64 %449, %82, !dbg !54
  %451 = add i64 %450, %address.local293, !dbg !54
  %.tr40 = trunc i64 %451 to i32, !dbg !54
  %452 = shl i32 %.tr40, 1, !dbg !54
  %dma.buffer.offset1779 = add i32 %452, 192, !dbg !54
  %453 = insertelement <2 x i64> poison, i64 %448, i64 0, !dbg !54
  %454 = bitcast <2 x i64> %453 to <4 x i32>, !dbg !54
  %dma.buffer.rsrc1780 = shufflevector <4 x i32> %454, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !54
  %dma.buffer.lds1781 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %447), !dbg !54
  %455 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1781, i32 %dma.buffer.offset1779, <4 x i32> %dma.buffer.rsrc1780) #10, !dbg !54
  %456 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !54
  %457 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !54
  %.idx44 = add nuw nsw i64 %446, %76, !dbg !54
  %458 = add i64 %.idx44, %444, !dbg !54
  %459 = add i64 %458, %82, !dbg !54
  %460 = add i64 %459, %address.local307, !dbg !54
  %.tr45 = trunc i64 %460 to i32, !dbg !54
  %461 = shl i32 %.tr45, 1, !dbg !54
  %dma.buffer.offset1776 = add i32 %461, 192, !dbg !54
  %462 = insertelement <2 x i64> poison, i64 %457, i64 0, !dbg !54
  %463 = bitcast <2 x i64> %462 to <4 x i32>, !dbg !54
  %dma.buffer.rsrc1777 = shufflevector <4 x i32> %463, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !54
  %dma.buffer.lds1778 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %456), !dbg !54
  %464 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1778, i32 %dma.buffer.offset1776, <4 x i32> %dma.buffer.rsrc1777) #10, !dbg !54
  %465 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !48
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !48
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !48
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !48
  %467 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !48
  %468 = fmul <4 x half> %465, splat (half 0xH3015), !dbg !48
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !48
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !48
  %470 = fmul <4 x half> %466, splat (half 0xH3015), !dbg !48
  %471 = fmul <4 x half> %467, splat (half 0xH3015), !dbg !48
  %472 = fmul <4 x half> %469, splat (half 0xH3015), !dbg !48
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !54
  fence syncscope("workgroup") release, !dbg !54
  call void @llvm.amdgcn.s.barrier(), !dbg !54
  fence syncscope("workgroup") acquire, !dbg !54
  %473 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !57
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %473, <4 x float> %355) #9, !dbg !57
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !57
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !57
  %476 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !57
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %475, <4 x float> %474) #9, !dbg !57
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %476, <4 x float> %359) #9, !dbg !57
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !57
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !57
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !57
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !57
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %479, <4 x float> %478) #9, !dbg !57
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %480, <4 x float> %363) #9, !dbg !57
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !57
  %483 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !57
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !57
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !57
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %483, <4 x float> %482) #9, !dbg !57
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %484, <4 x float> %367) #9, !dbg !57
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !57
  %487 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !57
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !57
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !57
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %487, <4 x float> %486) #9, !dbg !57
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %488, <4 x float> %371) #9, !dbg !57
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !57
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !57
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !57
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !57
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %491, <4 x float> %490) #9, !dbg !57
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %492, <4 x float> %375) #9, !dbg !57
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !57
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !57
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !57
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !57
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %495, <4 x float> %494) #9, !dbg !57
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %496, <4 x float> %379) #9, !dbg !57
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !57
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !57
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !57
  %500 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !57
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %499, <4 x float> %498) #9, !dbg !57
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %468, <4 x half> %500, <4 x float> %383) #9, !dbg !57
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !57
  %503 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !57
  %504 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !57
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %470, <4 x half> %503, <4 x float> %502) #9, !dbg !57
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %504, <4 x float> %387) #9, !dbg !57
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !57
  %507 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !57
  %508 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !57
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %507, <4 x float> %506) #9, !dbg !57
  %510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %508, <4 x float> %391) #9, !dbg !57
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !57
  %511 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !57
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !57
  %512 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !57
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %511, <4 x float> %510) #9, !dbg !57
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %512, <4 x float> %395) #9, !dbg !57
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !57
  %515 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !57
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !57
  %516 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !57
  %517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %515, <4 x float> %514) #9, !dbg !57
  %518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %516, <4 x float> %399) #9, !dbg !57
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !57
  %519 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !57
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !57
  %520 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !57
  %521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %519, <4 x float> %518) #9, !dbg !57
  %522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %520, <4 x float> %403) #9, !dbg !57
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !57
  %523 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !57
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !57
  %524 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !57
  %525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %523, <4 x float> %522) #9, !dbg !57
  %526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %524, <4 x float> %407) #9, !dbg !57
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !57
  %527 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !57
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !57
  %528 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !57
  %529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %527, <4 x float> %526) #9, !dbg !57
  %530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %528, <4 x float> %411) #9, !dbg !57
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !57
  %531 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !57
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !57
  %532 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !57
  %533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %531, <4 x float> %530) #9, !dbg !57
  %534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %471, <4 x half> %532, <4 x float> %414) #9, !dbg !57
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !57
  %535 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !57
  %536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %472, <4 x half> %535, <4 x float> %534) #9, !dbg !57
  fence syncscope("workgroup") release, !dbg !55
  call void @llvm.amdgcn.s.barrier(), !dbg !55
  fence syncscope("workgroup") acquire, !dbg !55
  %537 = extractelement <4 x float> %536, i64 3, !dbg !58
  %538 = extractelement <4 x float> %536, i64 2, !dbg !58
  %539 = extractelement <4 x float> %536, i64 1, !dbg !58
  %540 = extractelement <4 x float> %536, i64 0, !dbg !58
  %541 = extractelement <4 x float> %533, i64 3, !dbg !58
  %542 = extractelement <4 x float> %533, i64 2, !dbg !58
  %543 = extractelement <4 x float> %533, i64 1, !dbg !58
  %544 = extractelement <4 x float> %533, i64 0, !dbg !58
  %545 = extractelement <4 x float> %529, i64 3, !dbg !58
  %546 = extractelement <4 x float> %529, i64 2, !dbg !58
  %547 = extractelement <4 x float> %529, i64 1, !dbg !58
  %548 = extractelement <4 x float> %529, i64 0, !dbg !58
  %549 = extractelement <4 x float> %525, i64 3, !dbg !58
  %550 = extractelement <4 x float> %525, i64 2, !dbg !58
  %551 = extractelement <4 x float> %525, i64 1, !dbg !58
  %552 = extractelement <4 x float> %525, i64 0, !dbg !58
  %553 = extractelement <4 x float> %521, i64 3, !dbg !58
  %554 = extractelement <4 x float> %521, i64 2, !dbg !58
  %555 = extractelement <4 x float> %521, i64 1, !dbg !58
  %556 = extractelement <4 x float> %521, i64 0, !dbg !58
  %557 = extractelement <4 x float> %517, i64 3, !dbg !58
  %558 = extractelement <4 x float> %517, i64 2, !dbg !58
  %559 = extractelement <4 x float> %517, i64 1, !dbg !58
  %560 = extractelement <4 x float> %517, i64 0, !dbg !58
  %561 = extractelement <4 x float> %513, i64 3, !dbg !58
  %562 = extractelement <4 x float> %513, i64 2, !dbg !58
  %563 = extractelement <4 x float> %513, i64 1, !dbg !58
  %564 = extractelement <4 x float> %513, i64 0, !dbg !58
  %565 = extractelement <4 x float> %509, i64 3, !dbg !58
  %566 = extractelement <4 x float> %509, i64 2, !dbg !58
  %567 = extractelement <4 x float> %509, i64 1, !dbg !58
  %568 = extractelement <4 x float> %509, i64 0, !dbg !58
  %569 = extractelement <4 x float> %505, i64 3, !dbg !58
  %570 = extractelement <4 x float> %505, i64 2, !dbg !58
  %571 = extractelement <4 x float> %505, i64 1, !dbg !58
  %572 = extractelement <4 x float> %505, i64 0, !dbg !58
  %573 = extractelement <4 x float> %501, i64 3, !dbg !58
  %574 = extractelement <4 x float> %501, i64 2, !dbg !58
  %575 = extractelement <4 x float> %501, i64 1, !dbg !58
  %576 = extractelement <4 x float> %501, i64 0, !dbg !58
  %577 = extractelement <4 x float> %497, i64 3, !dbg !58
  %578 = extractelement <4 x float> %497, i64 2, !dbg !58
  %579 = extractelement <4 x float> %497, i64 1, !dbg !58
  %580 = extractelement <4 x float> %497, i64 0, !dbg !58
  %581 = extractelement <4 x float> %493, i64 3, !dbg !58
  %582 = extractelement <4 x float> %493, i64 2, !dbg !58
  %583 = extractelement <4 x float> %493, i64 1, !dbg !58
  %584 = extractelement <4 x float> %493, i64 0, !dbg !58
  %585 = extractelement <4 x float> %489, i64 3, !dbg !58
  %586 = extractelement <4 x float> %489, i64 2, !dbg !58
  %587 = extractelement <4 x float> %489, i64 1, !dbg !58
  %588 = extractelement <4 x float> %489, i64 0, !dbg !58
  %589 = extractelement <4 x float> %485, i64 3, !dbg !58
  %590 = extractelement <4 x float> %485, i64 2, !dbg !58
  %591 = extractelement <4 x float> %485, i64 1, !dbg !58
  %592 = extractelement <4 x float> %485, i64 0, !dbg !58
  %593 = extractelement <4 x float> %481, i64 3, !dbg !58
  %594 = extractelement <4 x float> %481, i64 2, !dbg !58
  %595 = extractelement <4 x float> %481, i64 1, !dbg !58
  %596 = extractelement <4 x float> %481, i64 0, !dbg !58
  %597 = extractelement <4 x float> %477, i64 3, !dbg !58
  %598 = extractelement <4 x float> %477, i64 2, !dbg !58
  %599 = extractelement <4 x float> %477, i64 1, !dbg !58
  %600 = extractelement <4 x float> %477, i64 0, !dbg !58
  %address.thread413 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !55, !range !49, !frisk.register_lifetimes !43
  %address.local414 = shl nuw nsw i32 %address.thread413, 5, !dbg !55
  %address.local415 = and i32 %address.local414, 6144, !dbg !55
  %address.local416 = shl nuw nsw i32 %address.thread413, 7, !dbg !55
  %address.local417 = and i32 %address.local416, 1920, !dbg !55
  %address.local418 = or disjoint i32 %address.local415, %address.local417, !dbg !55
  %address.local419 = zext nneg i32 %address.local418 to i64, !dbg !55
  %address.local420 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %601 = lshr i32 %address.thread413, 4, !dbg !48
  %address.local422 = zext nneg i32 %601 to i64, !dbg !48
  %address.local423 = and i64 %address.local422, 3, !dbg !48
  %address.local424 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local423, !dbg !55
  store float %600, ptr addrspace(3) %address.local424, align 4, !dbg !55
  %address.local425 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local426 = or disjoint i64 %address.local423, 4, !dbg !55
  %address.local427 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local425, i64 %address.local426, !dbg !55
  store float %599, ptr addrspace(3) %address.local427, align 4, !dbg !55
  %address.local428 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local429 = or disjoint i64 %address.local423, 8, !dbg !55
  %address.local430 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local428, i64 %address.local429, !dbg !55
  store float %598, ptr addrspace(3) %address.local430, align 4, !dbg !55
  %address.local431 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local432 = or i64 %address.local422, 12, !dbg !55
  %address.local433 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local431, i64 %address.local432, !dbg !55
  store float %597, ptr addrspace(3) %address.local433, align 4, !dbg !55
  %address.local434 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local435 = or disjoint i64 %address.local423, 16, !dbg !55
  %address.local436 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local434, i64 %address.local435, !dbg !55
  store float %596, ptr addrspace(3) %address.local436, align 4, !dbg !55
  %address.local437 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local438 = or disjoint i64 %address.local423, 20, !dbg !55
  %address.local439 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local437, i64 %address.local438, !dbg !55
  store float %595, ptr addrspace(3) %address.local439, align 4, !dbg !55
  %address.local440 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local441 = or disjoint i64 %address.local423, 24, !dbg !55
  %address.local442 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local440, i64 %address.local441, !dbg !55
  store float %594, ptr addrspace(3) %address.local442, align 4, !dbg !55
  %address.local443 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local444 = or i64 %address.local422, 28, !dbg !55
  %address.local445 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local443, i64 %address.local444, !dbg !55
  store float %593, ptr addrspace(3) %address.local445, align 4, !dbg !55
  %address.local446 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local447 = or disjoint i64 %address.local423, 32, !dbg !55
  %address.local448 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local446, i64 %address.local447, !dbg !55
  store float %592, ptr addrspace(3) %address.local448, align 4, !dbg !55
  %address.local449 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local450 = or disjoint i64 %address.local423, 36, !dbg !55
  %address.local451 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local449, i64 %address.local450, !dbg !55
  store float %591, ptr addrspace(3) %address.local451, align 4, !dbg !55
  %address.local452 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local453 = or disjoint i64 %address.local423, 40, !dbg !55
  %address.local454 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local452, i64 %address.local453, !dbg !55
  store float %590, ptr addrspace(3) %address.local454, align 4, !dbg !55
  %address.local455 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local456 = or i64 %address.local422, 44, !dbg !55
  %address.local457 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local455, i64 %address.local456, !dbg !55
  store float %589, ptr addrspace(3) %address.local457, align 4, !dbg !55
  %address.local458 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local459 = or disjoint i64 %address.local423, 48, !dbg !55
  %address.local460 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local458, i64 %address.local459, !dbg !55
  store float %588, ptr addrspace(3) %address.local460, align 4, !dbg !55
  %address.local461 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local462 = or disjoint i64 %address.local423, 52, !dbg !55
  %address.local463 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local461, i64 %address.local462, !dbg !55
  store float %587, ptr addrspace(3) %address.local463, align 4, !dbg !55
  %address.local464 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local465 = or disjoint i64 %address.local423, 56, !dbg !55
  %address.local466 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local464, i64 %address.local465, !dbg !55
  store float %586, ptr addrspace(3) %address.local466, align 4, !dbg !55
  %address.local467 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local468 = or i64 %address.local422, 60, !dbg !55
  %address.local469 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local467, i64 %address.local468, !dbg !55
  store float %585, ptr addrspace(3) %address.local469, align 4, !dbg !55
  %address.local470 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local471 = or disjoint i64 %address.local423, 64, !dbg !55
  %address.local472 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local470, i64 %address.local471, !dbg !55
  store float %584, ptr addrspace(3) %address.local472, align 4, !dbg !55
  %address.local473 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local474 = or disjoint i64 %address.local423, 68, !dbg !55
  %address.local475 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local473, i64 %address.local474, !dbg !55
  store float %583, ptr addrspace(3) %address.local475, align 4, !dbg !55
  %address.local476 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local477 = or disjoint i64 %address.local423, 72, !dbg !55
  %address.local478 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local476, i64 %address.local477, !dbg !55
  store float %582, ptr addrspace(3) %address.local478, align 4, !dbg !55
  %address.local479 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local480 = or i64 %address.local422, 76, !dbg !55
  %address.local481 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local479, i64 %address.local480, !dbg !55
  store float %581, ptr addrspace(3) %address.local481, align 4, !dbg !55
  %address.local482 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local483 = or disjoint i64 %address.local423, 80, !dbg !55
  %address.local484 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local482, i64 %address.local483, !dbg !55
  store float %580, ptr addrspace(3) %address.local484, align 4, !dbg !55
  %address.local485 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local486 = or disjoint i64 %address.local423, 84, !dbg !55
  %address.local487 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local485, i64 %address.local486, !dbg !55
  store float %579, ptr addrspace(3) %address.local487, align 4, !dbg !55
  %address.local488 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local489 = or disjoint i64 %address.local423, 88, !dbg !55
  %address.local490 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local488, i64 %address.local489, !dbg !55
  store float %578, ptr addrspace(3) %address.local490, align 4, !dbg !55
  %address.local491 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local492 = or i64 %address.local422, 92, !dbg !55
  %address.local493 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local491, i64 %address.local492, !dbg !55
  store float %577, ptr addrspace(3) %address.local493, align 4, !dbg !55
  %address.local494 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local495 = or disjoint i64 %address.local423, 96, !dbg !55
  %address.local496 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local494, i64 %address.local495, !dbg !55
  store float %576, ptr addrspace(3) %address.local496, align 4, !dbg !55
  %address.local497 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local498 = or disjoint i64 %address.local423, 100, !dbg !55
  %address.local499 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local497, i64 %address.local498, !dbg !55
  store float %575, ptr addrspace(3) %address.local499, align 4, !dbg !55
  %address.local500 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local501 = or disjoint i64 %address.local423, 104, !dbg !55
  %address.local502 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local500, i64 %address.local501, !dbg !55
  store float %574, ptr addrspace(3) %address.local502, align 4, !dbg !55
  %address.local503 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local504 = or i64 %address.local422, 108, !dbg !55
  %address.local505 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local503, i64 %address.local504, !dbg !55
  store float %573, ptr addrspace(3) %address.local505, align 4, !dbg !55
  %address.local506 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local507 = or disjoint i64 %address.local423, 112, !dbg !55
  %address.local508 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local506, i64 %address.local507, !dbg !55
  store float %572, ptr addrspace(3) %address.local508, align 4, !dbg !55
  %address.local509 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local510 = or disjoint i64 %address.local423, 116, !dbg !55
  %address.local511 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local509, i64 %address.local510, !dbg !55
  store float %571, ptr addrspace(3) %address.local511, align 4, !dbg !55
  %address.local512 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local513 = or disjoint i64 %address.local423, 120, !dbg !55
  %address.local514 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local512, i64 %address.local513, !dbg !55
  store float %570, ptr addrspace(3) %address.local514, align 4, !dbg !55
  %address.local515 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !55
  %address.local516 = or i64 %address.local422, 124, !dbg !55
  %address.local517 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local515, i64 %address.local516, !dbg !55
  store float %569, ptr addrspace(3) %address.local517, align 4, !dbg !55
  %address.local518 = shl nuw nsw i32 %address.thread413, 5, !dbg !55
  %address.local519 = and i32 %address.local518, 6144, !dbg !55
  %address.local520 = shl nuw nsw i32 %address.thread413, 7, !dbg !55
  %address.local521 = and i32 %address.local520, 1920, !dbg !55
  %address.local522 = or disjoint i32 %address.local519, %address.local521, !dbg !55
  %address.local523 = or disjoint i32 %address.local522, 8192, !dbg !55
  %address.local524 = zext nneg i32 %address.local523 to i64, !dbg !55
  %address.local525 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local526 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local423, !dbg !55
  store float %568, ptr addrspace(3) %address.local526, align 4, !dbg !55
  %address.local527 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local528 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local527, i64 %address.local426, !dbg !55
  store float %567, ptr addrspace(3) %address.local528, align 4, !dbg !55
  %address.local529 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local530 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local529, i64 %address.local429, !dbg !55
  store float %566, ptr addrspace(3) %address.local530, align 4, !dbg !55
  %address.local531 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local532 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local531, i64 %address.local432, !dbg !55
  store float %565, ptr addrspace(3) %address.local532, align 4, !dbg !55
  %address.local533 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local534 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local533, i64 %address.local435, !dbg !55
  store float %564, ptr addrspace(3) %address.local534, align 4, !dbg !55
  %address.local535 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local536 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local535, i64 %address.local438, !dbg !55
  store float %563, ptr addrspace(3) %address.local536, align 4, !dbg !55
  %address.local537 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local538 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local537, i64 %address.local441, !dbg !55
  store float %562, ptr addrspace(3) %address.local538, align 4, !dbg !55
  %address.local539 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local540 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local539, i64 %address.local444, !dbg !55
  store float %561, ptr addrspace(3) %address.local540, align 4, !dbg !55
  %address.local541 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local542 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local541, i64 %address.local447, !dbg !55
  store float %560, ptr addrspace(3) %address.local542, align 4, !dbg !55
  %address.local543 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local544 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local543, i64 %address.local450, !dbg !55
  store float %559, ptr addrspace(3) %address.local544, align 4, !dbg !55
  %address.local545 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local546 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local545, i64 %address.local453, !dbg !55
  store float %558, ptr addrspace(3) %address.local546, align 4, !dbg !55
  %address.local547 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local548 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local547, i64 %address.local456, !dbg !55
  store float %557, ptr addrspace(3) %address.local548, align 4, !dbg !55
  %address.local549 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local550 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local549, i64 %address.local459, !dbg !55
  store float %556, ptr addrspace(3) %address.local550, align 4, !dbg !55
  %address.local551 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local552 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local551, i64 %address.local462, !dbg !55
  store float %555, ptr addrspace(3) %address.local552, align 4, !dbg !55
  %address.local553 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local554 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local553, i64 %address.local465, !dbg !55
  store float %554, ptr addrspace(3) %address.local554, align 4, !dbg !55
  %address.local555 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local556 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local555, i64 %address.local468, !dbg !55
  store float %553, ptr addrspace(3) %address.local556, align 4, !dbg !55
  %address.local557 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local558 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local557, i64 %address.local471, !dbg !55
  store float %552, ptr addrspace(3) %address.local558, align 4, !dbg !55
  %address.local559 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local560 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local559, i64 %address.local474, !dbg !55
  store float %551, ptr addrspace(3) %address.local560, align 4, !dbg !55
  %address.local561 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local562 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local561, i64 %address.local477, !dbg !55
  store float %550, ptr addrspace(3) %address.local562, align 4, !dbg !55
  %address.local563 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local564 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local563, i64 %address.local480, !dbg !55
  store float %549, ptr addrspace(3) %address.local564, align 4, !dbg !55
  %address.local565 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local566 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local565, i64 %address.local483, !dbg !55
  store float %548, ptr addrspace(3) %address.local566, align 4, !dbg !55
  %address.local567 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local568 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local567, i64 %address.local486, !dbg !55
  store float %547, ptr addrspace(3) %address.local568, align 4, !dbg !55
  %address.local569 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local570 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local569, i64 %address.local489, !dbg !55
  store float %546, ptr addrspace(3) %address.local570, align 4, !dbg !55
  %address.local571 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local572 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local571, i64 %address.local492, !dbg !55
  store float %545, ptr addrspace(3) %address.local572, align 4, !dbg !55
  %address.local573 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local574 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local573, i64 %address.local495, !dbg !55
  store float %544, ptr addrspace(3) %address.local574, align 4, !dbg !55
  %address.local575 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local576 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local575, i64 %address.local498, !dbg !55
  store float %543, ptr addrspace(3) %address.local576, align 4, !dbg !55
  %address.local577 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local578 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local577, i64 %address.local501, !dbg !55
  store float %542, ptr addrspace(3) %address.local578, align 4, !dbg !55
  %address.local579 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local580 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local579, i64 %address.local504, !dbg !55
  store float %541, ptr addrspace(3) %address.local580, align 4, !dbg !55
  %address.local581 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local582 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local581, i64 %address.local507, !dbg !55
  store float %540, ptr addrspace(3) %address.local582, align 4, !dbg !55
  %address.local583 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local584 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local583, i64 %address.local510, !dbg !55
  store float %539, ptr addrspace(3) %address.local584, align 4, !dbg !55
  %address.local585 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local586 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local585, i64 %address.local513, !dbg !55
  store float %538, ptr addrspace(3) %address.local586, align 4, !dbg !55
  %address.local587 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !55
  %address.local588 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local587, i64 %address.local516, !dbg !55
  store float %537, ptr addrspace(3) %address.local588, align 4, !dbg !55
  fence syncscope("workgroup") release, !dbg !55
  call void @llvm.amdgcn.s.barrier(), !dbg !55
  fence syncscope("workgroup") acquire, !dbg !55
  %address.thread589 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !55, !range !49, !frisk.register_lifetimes !43
  %address.local590 = shl nuw nsw i32 %address.thread589, 5, !dbg !55
  %address.local591 = and i32 %address.local590, 6144, !dbg !55
  %address.local592 = shl nuw nsw i32 %address.thread589, 7, !dbg !55
  %address.local593 = and i32 %address.local592, 1920, !dbg !55
  %address.local594 = or disjoint i32 %address.local591, %address.local593, !dbg !55
  %address.local595 = zext nneg i32 %address.local594 to i64, !dbg !55
  %address.local596 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local595, !dbg !55
  %602 = lshr i32 %address.thread589, 2, !dbg !48
  %603 = and i32 %602, 12, !dbg !48
  %address.local600 = zext nneg i32 %603 to i64, !dbg !48
  %address.local601 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local596, i64 %address.local600, !dbg !55
  %604 = load float, ptr addrspace(3) %address.local601, align 16, !dbg !55
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local601, i64 4, !dbg !55
  %605 = load float, ptr addrspace(3) %lds.const.gep1682, align 4, !dbg !55
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local601, i64 8, !dbg !55
  %606 = load float, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !55
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local601, i64 12, !dbg !55
  %607 = load float, ptr addrspace(3) %lds.const.gep1684, align 4, !dbg !55
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local601, i64 64, !dbg !55
  %608 = load float, ptr addrspace(3) %lds.const.gep1685, align 16, !dbg !55
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local601, i64 68, !dbg !55
  %609 = load float, ptr addrspace(3) %lds.const.gep1686, align 4, !dbg !55
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local601, i64 72, !dbg !55
  %610 = load float, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !55
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local601, i64 76, !dbg !55
  %611 = load float, ptr addrspace(3) %lds.const.gep1688, align 4, !dbg !55
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local601, i64 128, !dbg !55
  %612 = load float, ptr addrspace(3) %lds.const.gep1689, align 16, !dbg !55
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local601, i64 132, !dbg !55
  %613 = load float, ptr addrspace(3) %lds.const.gep1690, align 4, !dbg !55
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local601, i64 136, !dbg !55
  %614 = load float, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !55
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local601, i64 140, !dbg !55
  %615 = load float, ptr addrspace(3) %lds.const.gep1692, align 4, !dbg !55
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local601, i64 192, !dbg !55
  %616 = load float, ptr addrspace(3) %lds.const.gep1693, align 16, !dbg !55
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local601, i64 196, !dbg !55
  %617 = load float, ptr addrspace(3) %lds.const.gep1694, align 4, !dbg !55
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local601, i64 200, !dbg !55
  %618 = load float, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !55
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local601, i64 204, !dbg !55
  %619 = load float, ptr addrspace(3) %lds.const.gep1696, align 4, !dbg !55
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local601, i64 256, !dbg !55
  %620 = load float, ptr addrspace(3) %lds.const.gep1697, align 16, !dbg !55
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local601, i64 260, !dbg !55
  %621 = load float, ptr addrspace(3) %lds.const.gep1698, align 4, !dbg !55
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local601, i64 264, !dbg !55
  %622 = load float, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !55
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local601, i64 268, !dbg !55
  %623 = load float, ptr addrspace(3) %lds.const.gep1700, align 4, !dbg !55
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local601, i64 320, !dbg !55
  %624 = load float, ptr addrspace(3) %lds.const.gep1701, align 16, !dbg !55
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %address.local601, i64 324, !dbg !55
  %625 = load float, ptr addrspace(3) %lds.const.gep1702, align 4, !dbg !55
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %address.local601, i64 328, !dbg !55
  %626 = load float, ptr addrspace(3) %lds.const.gep1703, align 8, !dbg !55
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local601, i64 332, !dbg !55
  %627 = load float, ptr addrspace(3) %lds.const.gep1704, align 4, !dbg !55
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local601, i64 384, !dbg !55
  %628 = load float, ptr addrspace(3) %lds.const.gep1705, align 16, !dbg !55
  %lds.const.gep1706 = getelementptr i8, ptr addrspace(3) %address.local601, i64 388, !dbg !55
  %629 = load float, ptr addrspace(3) %lds.const.gep1706, align 4, !dbg !55
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %address.local601, i64 392, !dbg !55
  %630 = load float, ptr addrspace(3) %lds.const.gep1707, align 8, !dbg !55
  %lds.const.gep1708 = getelementptr i8, ptr addrspace(3) %address.local601, i64 396, !dbg !55
  %631 = load float, ptr addrspace(3) %lds.const.gep1708, align 4, !dbg !55
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local601, i64 448, !dbg !55
  %632 = load float, ptr addrspace(3) %lds.const.gep1709, align 16, !dbg !55
  %lds.const.gep1710 = getelementptr i8, ptr addrspace(3) %address.local601, i64 452, !dbg !55
  %633 = load float, ptr addrspace(3) %lds.const.gep1710, align 4, !dbg !55
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %address.local601, i64 456, !dbg !55
  %634 = load float, ptr addrspace(3) %lds.const.gep1711, align 8, !dbg !55
  %lds.const.gep1712 = getelementptr i8, ptr addrspace(3) %address.local601, i64 460, !dbg !55
  %635 = load float, ptr addrspace(3) %lds.const.gep1712, align 4, !dbg !55
  %lds.const.gep1713 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32768, !dbg !55
  %636 = load float, ptr addrspace(3) %lds.const.gep1713, align 16, !dbg !55
  %lds.const.gep1714 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32772, !dbg !55
  %637 = load float, ptr addrspace(3) %lds.const.gep1714, align 4, !dbg !55
  %lds.const.gep1715 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32776, !dbg !55
  %638 = load float, ptr addrspace(3) %lds.const.gep1715, align 8, !dbg !55
  %lds.const.gep1716 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32780, !dbg !55
  %639 = load float, ptr addrspace(3) %lds.const.gep1716, align 4, !dbg !55
  %lds.const.gep1717 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32832, !dbg !55
  %640 = load float, ptr addrspace(3) %lds.const.gep1717, align 16, !dbg !55
  %lds.const.gep1718 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32836, !dbg !55
  %641 = load float, ptr addrspace(3) %lds.const.gep1718, align 4, !dbg !55
  %lds.const.gep1719 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32840, !dbg !55
  %642 = load float, ptr addrspace(3) %lds.const.gep1719, align 8, !dbg !55
  %lds.const.gep1720 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32844, !dbg !55
  %643 = load float, ptr addrspace(3) %lds.const.gep1720, align 4, !dbg !55
  %lds.const.gep1721 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32896, !dbg !55
  %644 = load float, ptr addrspace(3) %lds.const.gep1721, align 16, !dbg !55
  %lds.const.gep1722 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32900, !dbg !55
  %645 = load float, ptr addrspace(3) %lds.const.gep1722, align 4, !dbg !55
  %lds.const.gep1723 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32904, !dbg !55
  %646 = load float, ptr addrspace(3) %lds.const.gep1723, align 8, !dbg !55
  %lds.const.gep1724 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32908, !dbg !55
  %647 = load float, ptr addrspace(3) %lds.const.gep1724, align 4, !dbg !55
  %lds.const.gep1725 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32960, !dbg !55
  %648 = load float, ptr addrspace(3) %lds.const.gep1725, align 16, !dbg !55
  %lds.const.gep1726 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32964, !dbg !55
  %649 = load float, ptr addrspace(3) %lds.const.gep1726, align 4, !dbg !55
  %lds.const.gep1727 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32968, !dbg !55
  %650 = load float, ptr addrspace(3) %lds.const.gep1727, align 8, !dbg !55
  %lds.const.gep1728 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32972, !dbg !55
  %651 = load float, ptr addrspace(3) %lds.const.gep1728, align 4, !dbg !55
  %lds.const.gep1729 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33024, !dbg !55
  %652 = load float, ptr addrspace(3) %lds.const.gep1729, align 16, !dbg !55
  %lds.const.gep1730 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33028, !dbg !55
  %653 = load float, ptr addrspace(3) %lds.const.gep1730, align 4, !dbg !55
  %lds.const.gep1731 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33032, !dbg !55
  %654 = load float, ptr addrspace(3) %lds.const.gep1731, align 8, !dbg !55
  %lds.const.gep1732 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33036, !dbg !55
  %655 = load float, ptr addrspace(3) %lds.const.gep1732, align 4, !dbg !55
  %lds.const.gep1733 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33088, !dbg !55
  %656 = load float, ptr addrspace(3) %lds.const.gep1733, align 16, !dbg !55
  %lds.const.gep1734 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33092, !dbg !55
  %657 = load float, ptr addrspace(3) %lds.const.gep1734, align 4, !dbg !55
  %lds.const.gep1735 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33096, !dbg !55
  %658 = load float, ptr addrspace(3) %lds.const.gep1735, align 8, !dbg !55
  %lds.const.gep1736 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33100, !dbg !55
  %659 = load float, ptr addrspace(3) %lds.const.gep1736, align 4, !dbg !55
  %lds.const.gep1737 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33152, !dbg !55
  %660 = load float, ptr addrspace(3) %lds.const.gep1737, align 16, !dbg !55
  %lds.const.gep1738 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33156, !dbg !55
  %661 = load float, ptr addrspace(3) %lds.const.gep1738, align 4, !dbg !55
  %lds.const.gep1739 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33160, !dbg !55
  %662 = load float, ptr addrspace(3) %lds.const.gep1739, align 8, !dbg !55
  %lds.const.gep1740 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33164, !dbg !55
  %663 = load float, ptr addrspace(3) %lds.const.gep1740, align 4, !dbg !55
  %lds.const.gep1741 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33216, !dbg !55
  %664 = load float, ptr addrspace(3) %lds.const.gep1741, align 16, !dbg !55
  %lds.const.gep1742 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33220, !dbg !55
  %665 = load float, ptr addrspace(3) %lds.const.gep1742, align 4, !dbg !55
  %lds.const.gep1743 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33224, !dbg !55
  %666 = load float, ptr addrspace(3) %lds.const.gep1743, align 8, !dbg !55
  %lds.const.gep1744 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33228, !dbg !55
  %667 = load float, ptr addrspace(3) %lds.const.gep1744, align 4, !dbg !55
  fence syncscope("workgroup") release, !dbg !55
  call void @llvm.amdgcn.s.barrier(), !dbg !55
  fence syncscope("workgroup") acquire, !dbg !55
  %668 = or disjoint i64 %63, 127, !dbg !50
  %669 = icmp uge i64 %19, %668, !dbg !50
  %670 = and i1 %42, %669, !dbg !50
  br i1 %670, label %812, label %671, !dbg !50

671:                                              ; preds = %62
  %672 = add i64 %19, %35, !dbg !50
  %673 = add i64 %672, %52, !dbg !50
  %674 = sub i64 %673, %39, !dbg !50
  %675 = or disjoint i64 %63, %38, !dbg !50
  %676 = add i64 %674, 1, !dbg !59
  %677 = lshr i64 %676, 32, !dbg !60
  %compare.high = trunc nuw i64 %677 to i32, !dbg !60
  %678 = lshr i64 %63, 32, !dbg !60
  %compare.high1303 = trunc nuw i64 %678 to i32, !dbg !60
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1303, !dbg !60
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1303, !dbg !60
  %compare.low.base = trunc i64 %676 to i32, !dbg !60
  %compare.low.base1304 = trunc i64 %675 to i32, !dbg !60
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1304, !dbg !60
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !60
  %679 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 1) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local = or i32 %compare.local.base, 1, !dbg !60
  %compare.low1305 = icmp ult i32 %compare.local, %compare.low.base, !dbg !60
  %compare.result1306 = select i1 %compare.high.equal, i1 %compare.low1305, i1 %compare.high.order, !dbg !60
  %680 = select i1 %compare.result1306, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1307 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 2) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1308 = or i32 %compare.local.base1307, 2, !dbg !60
  %compare.low1309 = icmp ult i32 %compare.local1308, %compare.low.base, !dbg !60
  %compare.result1310 = select i1 %compare.high.equal, i1 %compare.low1309, i1 %compare.high.order, !dbg !60
  %681 = select i1 %compare.result1310, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1311 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 3) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1312 = or i32 %compare.local.base1311, 3, !dbg !60
  %compare.low1313 = icmp ult i32 %compare.local1312, %compare.low.base, !dbg !60
  %compare.result1314 = select i1 %compare.high.equal, i1 %compare.low1313, i1 %compare.high.order, !dbg !60
  %682 = select i1 %compare.result1314, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1315 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 4) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1316 = or i32 %compare.local.base1315, 16, !dbg !60
  %compare.low1317 = icmp ult i32 %compare.local1316, %compare.low.base, !dbg !60
  %compare.result1318 = select i1 %compare.high.equal, i1 %compare.low1317, i1 %compare.high.order, !dbg !60
  %683 = select i1 %compare.result1318, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1319 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 5) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1320 = or i32 %compare.local.base1319, 17, !dbg !60
  %compare.low1321 = icmp ult i32 %compare.local1320, %compare.low.base, !dbg !60
  %compare.result1322 = select i1 %compare.high.equal, i1 %compare.low1321, i1 %compare.high.order, !dbg !60
  %684 = select i1 %compare.result1322, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1323 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 6) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1324 = or i32 %compare.local.base1323, 18, !dbg !60
  %compare.low1325 = icmp ult i32 %compare.local1324, %compare.low.base, !dbg !60
  %compare.result1326 = select i1 %compare.high.equal, i1 %compare.low1325, i1 %compare.high.order, !dbg !60
  %685 = select i1 %compare.result1326, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 7) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1328 = or i32 %compare.local.base1327, 19, !dbg !60
  %compare.low1329 = icmp ult i32 %compare.local1328, %compare.low.base, !dbg !60
  %compare.result1330 = select i1 %compare.high.equal, i1 %compare.low1329, i1 %compare.high.order, !dbg !60
  %686 = select i1 %compare.result1330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 8) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1332 = or i32 %compare.local.base1331, 32, !dbg !60
  %compare.low1333 = icmp ult i32 %compare.local1332, %compare.low.base, !dbg !60
  %compare.result1334 = select i1 %compare.high.equal, i1 %compare.low1333, i1 %compare.high.order, !dbg !60
  %687 = select i1 %compare.result1334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 9) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1336 = or i32 %compare.local.base1335, 33, !dbg !60
  %compare.low1337 = icmp ult i32 %compare.local1336, %compare.low.base, !dbg !60
  %compare.result1338 = select i1 %compare.high.equal, i1 %compare.low1337, i1 %compare.high.order, !dbg !60
  %688 = select i1 %compare.result1338, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 10) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1340 = or i32 %compare.local.base1339, 34, !dbg !60
  %compare.low1341 = icmp ult i32 %compare.local1340, %compare.low.base, !dbg !60
  %compare.result1342 = select i1 %compare.high.equal, i1 %compare.low1341, i1 %compare.high.order, !dbg !60
  %689 = select i1 %compare.result1342, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 11) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1344 = or i32 %compare.local.base1343, 35, !dbg !60
  %compare.low1345 = icmp ult i32 %compare.local1344, %compare.low.base, !dbg !60
  %compare.result1346 = select i1 %compare.high.equal, i1 %compare.low1345, i1 %compare.high.order, !dbg !60
  %690 = select i1 %compare.result1346, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 12) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1348 = or i32 %compare.local.base1347, 48, !dbg !60
  %compare.low1349 = icmp ult i32 %compare.local1348, %compare.low.base, !dbg !60
  %compare.result1350 = select i1 %compare.high.equal, i1 %compare.low1349, i1 %compare.high.order, !dbg !60
  %691 = select i1 %compare.result1350, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 13) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1352 = or i32 %compare.local.base1351, 49, !dbg !60
  %compare.low1353 = icmp ult i32 %compare.local1352, %compare.low.base, !dbg !60
  %compare.result1354 = select i1 %compare.high.equal, i1 %compare.low1353, i1 %compare.high.order, !dbg !60
  %692 = select i1 %compare.result1354, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 14) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1356 = or i32 %compare.local.base1355, 50, !dbg !60
  %compare.low1357 = icmp ult i32 %compare.local1356, %compare.low.base, !dbg !60
  %compare.result1358 = select i1 %compare.high.equal, i1 %compare.low1357, i1 %compare.high.order, !dbg !60
  %693 = select i1 %compare.result1358, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 15) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1360 = or i32 %compare.local.base1359, 51, !dbg !60
  %compare.low1361 = icmp ult i32 %compare.local1360, %compare.low.base, !dbg !60
  %compare.result1362 = select i1 %compare.high.equal, i1 %compare.low1361, i1 %compare.high.order, !dbg !60
  %694 = select i1 %compare.result1362, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 16) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1364 = or i32 %compare.local.base1363, 64, !dbg !60
  %compare.low1365 = icmp ult i32 %compare.local1364, %compare.low.base, !dbg !60
  %compare.result1366 = select i1 %compare.high.equal, i1 %compare.low1365, i1 %compare.high.order, !dbg !60
  %695 = select i1 %compare.result1366, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 17) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1368 = or i32 %compare.local.base1367, 65, !dbg !60
  %compare.low1369 = icmp ult i32 %compare.local1368, %compare.low.base, !dbg !60
  %compare.result1370 = select i1 %compare.high.equal, i1 %compare.low1369, i1 %compare.high.order, !dbg !60
  %696 = select i1 %compare.result1370, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 18) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1372 = or i32 %compare.local.base1371, 66, !dbg !60
  %compare.low1373 = icmp ult i32 %compare.local1372, %compare.low.base, !dbg !60
  %compare.result1374 = select i1 %compare.high.equal, i1 %compare.low1373, i1 %compare.high.order, !dbg !60
  %697 = select i1 %compare.result1374, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 19) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1376 = or i32 %compare.local.base1375, 67, !dbg !60
  %compare.low1377 = icmp ult i32 %compare.local1376, %compare.low.base, !dbg !60
  %compare.result1378 = select i1 %compare.high.equal, i1 %compare.low1377, i1 %compare.high.order, !dbg !60
  %698 = select i1 %compare.result1378, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 20) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1380 = or i32 %compare.local.base1379, 80, !dbg !60
  %compare.low1381 = icmp ult i32 %compare.local1380, %compare.low.base, !dbg !60
  %compare.result1382 = select i1 %compare.high.equal, i1 %compare.low1381, i1 %compare.high.order, !dbg !60
  %699 = select i1 %compare.result1382, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 21) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1384 = or i32 %compare.local.base1383, 81, !dbg !60
  %compare.low1385 = icmp ult i32 %compare.local1384, %compare.low.base, !dbg !60
  %compare.result1386 = select i1 %compare.high.equal, i1 %compare.low1385, i1 %compare.high.order, !dbg !60
  %700 = select i1 %compare.result1386, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 22) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1388 = or i32 %compare.local.base1387, 82, !dbg !60
  %compare.low1389 = icmp ult i32 %compare.local1388, %compare.low.base, !dbg !60
  %compare.result1390 = select i1 %compare.high.equal, i1 %compare.low1389, i1 %compare.high.order, !dbg !60
  %701 = select i1 %compare.result1390, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 23) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1392 = or i32 %compare.local.base1391, 83, !dbg !60
  %compare.low1393 = icmp ult i32 %compare.local1392, %compare.low.base, !dbg !60
  %compare.result1394 = select i1 %compare.high.equal, i1 %compare.low1393, i1 %compare.high.order, !dbg !60
  %702 = select i1 %compare.result1394, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 24) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1396 = or i32 %compare.local.base1395, 96, !dbg !60
  %compare.low1397 = icmp ult i32 %compare.local1396, %compare.low.base, !dbg !60
  %compare.result1398 = select i1 %compare.high.equal, i1 %compare.low1397, i1 %compare.high.order, !dbg !60
  %703 = select i1 %compare.result1398, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 25) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1400 = or i32 %compare.local.base1399, 97, !dbg !60
  %compare.low1401 = icmp ult i32 %compare.local1400, %compare.low.base, !dbg !60
  %compare.result1402 = select i1 %compare.high.equal, i1 %compare.low1401, i1 %compare.high.order, !dbg !60
  %704 = select i1 %compare.result1402, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 26) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1404 = or i32 %compare.local.base1403, 98, !dbg !60
  %compare.low1405 = icmp ult i32 %compare.local1404, %compare.low.base, !dbg !60
  %compare.result1406 = select i1 %compare.high.equal, i1 %compare.low1405, i1 %compare.high.order, !dbg !60
  %705 = select i1 %compare.result1406, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 27) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1408 = or i32 %compare.local.base1407, 99, !dbg !60
  %compare.low1409 = icmp ult i32 %compare.local1408, %compare.low.base, !dbg !60
  %compare.result1410 = select i1 %compare.high.equal, i1 %compare.low1409, i1 %compare.high.order, !dbg !60
  %706 = select i1 %compare.result1410, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 28) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1412 = or i32 %compare.local.base1411, 112, !dbg !60
  %compare.low1413 = icmp ult i32 %compare.local1412, %compare.low.base, !dbg !60
  %compare.result1414 = select i1 %compare.high.equal, i1 %compare.low1413, i1 %compare.high.order, !dbg !60
  %707 = select i1 %compare.result1414, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 29) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1416 = or i32 %compare.local.base1415, 113, !dbg !60
  %compare.low1417 = icmp ult i32 %compare.local1416, %compare.low.base, !dbg !60
  %compare.result1418 = select i1 %compare.high.equal, i1 %compare.low1417, i1 %compare.high.order, !dbg !60
  %708 = select i1 %compare.result1418, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 30) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1420 = or i32 %compare.local.base1419, 114, !dbg !60
  %compare.low1421 = icmp ult i32 %compare.local1420, %compare.low.base, !dbg !60
  %compare.result1422 = select i1 %compare.high.equal, i1 %compare.low1421, i1 %compare.high.order, !dbg !60
  %709 = select i1 %compare.result1422, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 31) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1424 = or i32 %compare.local.base1423, 115, !dbg !60
  %compare.low1425 = icmp ult i32 %compare.local1424, %compare.low.base, !dbg !60
  %compare.result1426 = select i1 %compare.high.equal, i1 %compare.low1425, i1 %compare.high.order, !dbg !60
  %710 = select i1 %compare.result1426, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %711 = add i64 %674, 65, !dbg !59
  %712 = lshr i64 %711, 32, !dbg !60
  %compare.high1427 = trunc nuw i64 %712 to i32, !dbg !60
  %713 = lshr i64 %63, 32, !dbg !60
  %compare.high1428 = trunc nuw i64 %713 to i32, !dbg !60
  %compare.high.equal1429 = icmp eq i32 %compare.high1427, %compare.high1428, !dbg !60
  %compare.high.order1430 = icmp ugt i32 %compare.high1427, %compare.high1428, !dbg !60
  %compare.low.base1431 = trunc i64 %711 to i32, !dbg !60
  %compare.low.base1432 = trunc i64 %675 to i32, !dbg !60
  %compare.low1433 = icmp ugt i32 %compare.low.base1431, %compare.low.base1432, !dbg !60
  %compare.result1434 = select i1 %compare.high.equal1429, i1 %compare.low1433, i1 %compare.high.order1430, !dbg !60
  %714 = select i1 %compare.result1434, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 32) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1436 = or i32 %compare.local.base1435, 1, !dbg !60
  %compare.low1437 = icmp ult i32 %compare.local1436, %compare.low.base1431, !dbg !60
  %compare.result1438 = select i1 %compare.high.equal1429, i1 %compare.low1437, i1 %compare.high.order1430, !dbg !60
  %715 = select i1 %compare.result1438, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 33) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1440 = or i32 %compare.local.base1439, 2, !dbg !60
  %compare.low1441 = icmp ult i32 %compare.local1440, %compare.low.base1431, !dbg !60
  %compare.result1442 = select i1 %compare.high.equal1429, i1 %compare.low1441, i1 %compare.high.order1430, !dbg !60
  %716 = select i1 %compare.result1442, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 34) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1444 = or i32 %compare.local.base1443, 3, !dbg !60
  %compare.low1445 = icmp ult i32 %compare.local1444, %compare.low.base1431, !dbg !60
  %compare.result1446 = select i1 %compare.high.equal1429, i1 %compare.low1445, i1 %compare.high.order1430, !dbg !60
  %717 = select i1 %compare.result1446, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1447 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 35) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1448 = or i32 %compare.local.base1447, 16, !dbg !60
  %compare.low1449 = icmp ult i32 %compare.local1448, %compare.low.base1431, !dbg !60
  %compare.result1450 = select i1 %compare.high.equal1429, i1 %compare.low1449, i1 %compare.high.order1430, !dbg !60
  %718 = select i1 %compare.result1450, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1451 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 36) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1452 = or i32 %compare.local.base1451, 17, !dbg !60
  %compare.low1453 = icmp ult i32 %compare.local1452, %compare.low.base1431, !dbg !60
  %compare.result1454 = select i1 %compare.high.equal1429, i1 %compare.low1453, i1 %compare.high.order1430, !dbg !60
  %719 = select i1 %compare.result1454, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1455 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 37) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1456 = or i32 %compare.local.base1455, 18, !dbg !60
  %compare.low1457 = icmp ult i32 %compare.local1456, %compare.low.base1431, !dbg !60
  %compare.result1458 = select i1 %compare.high.equal1429, i1 %compare.low1457, i1 %compare.high.order1430, !dbg !60
  %720 = select i1 %compare.result1458, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 38) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1460 = or i32 %compare.local.base1459, 19, !dbg !60
  %compare.low1461 = icmp ult i32 %compare.local1460, %compare.low.base1431, !dbg !60
  %compare.result1462 = select i1 %compare.high.equal1429, i1 %compare.low1461, i1 %compare.high.order1430, !dbg !60
  %721 = select i1 %compare.result1462, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 39) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1464 = or i32 %compare.local.base1463, 32, !dbg !60
  %compare.low1465 = icmp ult i32 %compare.local1464, %compare.low.base1431, !dbg !60
  %compare.result1466 = select i1 %compare.high.equal1429, i1 %compare.low1465, i1 %compare.high.order1430, !dbg !60
  %722 = select i1 %compare.result1466, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 40) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1468 = or i32 %compare.local.base1467, 33, !dbg !60
  %compare.low1469 = icmp ult i32 %compare.local1468, %compare.low.base1431, !dbg !60
  %compare.result1470 = select i1 %compare.high.equal1429, i1 %compare.low1469, i1 %compare.high.order1430, !dbg !60
  %723 = select i1 %compare.result1470, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 41) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1472 = or i32 %compare.local.base1471, 34, !dbg !60
  %compare.low1473 = icmp ult i32 %compare.local1472, %compare.low.base1431, !dbg !60
  %compare.result1474 = select i1 %compare.high.equal1429, i1 %compare.low1473, i1 %compare.high.order1430, !dbg !60
  %724 = select i1 %compare.result1474, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 42) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1476 = or i32 %compare.local.base1475, 35, !dbg !60
  %compare.low1477 = icmp ult i32 %compare.local1476, %compare.low.base1431, !dbg !60
  %compare.result1478 = select i1 %compare.high.equal1429, i1 %compare.low1477, i1 %compare.high.order1430, !dbg !60
  %725 = select i1 %compare.result1478, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 43) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1480 = or i32 %compare.local.base1479, 48, !dbg !60
  %compare.low1481 = icmp ult i32 %compare.local1480, %compare.low.base1431, !dbg !60
  %compare.result1482 = select i1 %compare.high.equal1429, i1 %compare.low1481, i1 %compare.high.order1430, !dbg !60
  %726 = select i1 %compare.result1482, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 44) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1484 = or i32 %compare.local.base1483, 49, !dbg !60
  %compare.low1485 = icmp ult i32 %compare.local1484, %compare.low.base1431, !dbg !60
  %compare.result1486 = select i1 %compare.high.equal1429, i1 %compare.low1485, i1 %compare.high.order1430, !dbg !60
  %727 = select i1 %compare.result1486, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1487 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 45) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1488 = or i32 %compare.local.base1487, 50, !dbg !60
  %compare.low1489 = icmp ult i32 %compare.local1488, %compare.low.base1431, !dbg !60
  %compare.result1490 = select i1 %compare.high.equal1429, i1 %compare.low1489, i1 %compare.high.order1430, !dbg !60
  %728 = select i1 %compare.result1490, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1491 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 46) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1492 = or i32 %compare.local.base1491, 51, !dbg !60
  %compare.low1493 = icmp ult i32 %compare.local1492, %compare.low.base1431, !dbg !60
  %compare.result1494 = select i1 %compare.high.equal1429, i1 %compare.low1493, i1 %compare.high.order1430, !dbg !60
  %729 = select i1 %compare.result1494, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1495 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 47) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1496 = or i32 %compare.local.base1495, 64, !dbg !60
  %compare.low1497 = icmp ult i32 %compare.local1496, %compare.low.base1431, !dbg !60
  %compare.result1498 = select i1 %compare.high.equal1429, i1 %compare.low1497, i1 %compare.high.order1430, !dbg !60
  %730 = select i1 %compare.result1498, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1499 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 48) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1500 = or i32 %compare.local.base1499, 65, !dbg !60
  %compare.low1501 = icmp ult i32 %compare.local1500, %compare.low.base1431, !dbg !60
  %compare.result1502 = select i1 %compare.high.equal1429, i1 %compare.low1501, i1 %compare.high.order1430, !dbg !60
  %731 = select i1 %compare.result1502, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1503 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 49) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1504 = or i32 %compare.local.base1503, 66, !dbg !60
  %compare.low1505 = icmp ult i32 %compare.local1504, %compare.low.base1431, !dbg !60
  %compare.result1506 = select i1 %compare.high.equal1429, i1 %compare.low1505, i1 %compare.high.order1430, !dbg !60
  %732 = select i1 %compare.result1506, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1507 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 50) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1508 = or i32 %compare.local.base1507, 67, !dbg !60
  %compare.low1509 = icmp ult i32 %compare.local1508, %compare.low.base1431, !dbg !60
  %compare.result1510 = select i1 %compare.high.equal1429, i1 %compare.low1509, i1 %compare.high.order1430, !dbg !60
  %733 = select i1 %compare.result1510, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1511 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 51) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1512 = or i32 %compare.local.base1511, 80, !dbg !60
  %compare.low1513 = icmp ult i32 %compare.local1512, %compare.low.base1431, !dbg !60
  %compare.result1514 = select i1 %compare.high.equal1429, i1 %compare.low1513, i1 %compare.high.order1430, !dbg !60
  %734 = select i1 %compare.result1514, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1515 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 52) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1516 = or i32 %compare.local.base1515, 81, !dbg !60
  %compare.low1517 = icmp ult i32 %compare.local1516, %compare.low.base1431, !dbg !60
  %compare.result1518 = select i1 %compare.high.equal1429, i1 %compare.low1517, i1 %compare.high.order1430, !dbg !60
  %735 = select i1 %compare.result1518, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1519 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 53) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1520 = or i32 %compare.local.base1519, 82, !dbg !60
  %compare.low1521 = icmp ult i32 %compare.local1520, %compare.low.base1431, !dbg !60
  %compare.result1522 = select i1 %compare.high.equal1429, i1 %compare.low1521, i1 %compare.high.order1430, !dbg !60
  %736 = select i1 %compare.result1522, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 54) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1524 = or i32 %compare.local.base1523, 83, !dbg !60
  %compare.low1525 = icmp ult i32 %compare.local1524, %compare.low.base1431, !dbg !60
  %compare.result1526 = select i1 %compare.high.equal1429, i1 %compare.low1525, i1 %compare.high.order1430, !dbg !60
  %737 = select i1 %compare.result1526, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 55) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1528 = or i32 %compare.local.base1527, 96, !dbg !60
  %compare.low1529 = icmp ult i32 %compare.local1528, %compare.low.base1431, !dbg !60
  %compare.result1530 = select i1 %compare.high.equal1429, i1 %compare.low1529, i1 %compare.high.order1430, !dbg !60
  %738 = select i1 %compare.result1530, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 56) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1532 = or i32 %compare.local.base1531, 97, !dbg !60
  %compare.low1533 = icmp ult i32 %compare.local1532, %compare.low.base1431, !dbg !60
  %compare.result1534 = select i1 %compare.high.equal1429, i1 %compare.low1533, i1 %compare.high.order1430, !dbg !60
  %739 = select i1 %compare.result1534, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 57) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1536 = or i32 %compare.local.base1535, 98, !dbg !60
  %compare.low1537 = icmp ult i32 %compare.local1536, %compare.low.base1431, !dbg !60
  %compare.result1538 = select i1 %compare.high.equal1429, i1 %compare.low1537, i1 %compare.high.order1430, !dbg !60
  %740 = select i1 %compare.result1538, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 58) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1540 = or i32 %compare.local.base1539, 99, !dbg !60
  %compare.low1541 = icmp ult i32 %compare.local1540, %compare.low.base1431, !dbg !60
  %compare.result1542 = select i1 %compare.high.equal1429, i1 %compare.low1541, i1 %compare.high.order1430, !dbg !60
  %741 = select i1 %compare.result1542, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 59) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1544 = or i32 %compare.local.base1543, 112, !dbg !60
  %compare.low1545 = icmp ult i32 %compare.local1544, %compare.low.base1431, !dbg !60
  %compare.result1546 = select i1 %compare.high.equal1429, i1 %compare.low1545, i1 %compare.high.order1430, !dbg !60
  %742 = select i1 %compare.result1546, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 60) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1548 = or i32 %compare.local.base1547, 113, !dbg !60
  %compare.low1549 = icmp ult i32 %compare.local1548, %compare.low.base1431, !dbg !60
  %compare.result1550 = select i1 %compare.high.equal1429, i1 %compare.low1549, i1 %compare.high.order1430, !dbg !60
  %743 = select i1 %compare.result1550, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 61) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1552 = or i32 %compare.local.base1551, 114, !dbg !60
  %compare.low1553 = icmp ult i32 %compare.local1552, %compare.low.base1431, !dbg !60
  %compare.result1554 = select i1 %compare.high.equal1429, i1 %compare.low1553, i1 %compare.high.order1430, !dbg !60
  %744 = select i1 %compare.result1554, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %compare.local.base1555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 62) #9, !dbg !60, !frisk.register_lifetimes !43
  %compare.local1556 = or i32 %compare.local.base1555, 115, !dbg !60
  %compare.low1557 = icmp ult i32 %compare.local1556, %compare.low.base1431, !dbg !60
  %compare.result1558 = select i1 %compare.high.equal1429, i1 %compare.low1557, i1 %compare.high.order1430, !dbg !60
  %745 = select i1 %compare.result1558, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !61
  %746 = insertelement <32 x float> poison, float %679, i64 0, !dbg !50
  %747 = insertelement <32 x float> %746, float %680, i64 1, !dbg !50
  %748 = insertelement <32 x float> %747, float %681, i64 2, !dbg !50
  %749 = insertelement <32 x float> %748, float %682, i64 3, !dbg !50
  %750 = insertelement <32 x float> %749, float %683, i64 4, !dbg !50
  %751 = insertelement <32 x float> %750, float %684, i64 5, !dbg !50
  %752 = insertelement <32 x float> %751, float %685, i64 6, !dbg !50
  %753 = insertelement <32 x float> %752, float %686, i64 7, !dbg !50
  %754 = insertelement <32 x float> %753, float %687, i64 8, !dbg !50
  %755 = insertelement <32 x float> %754, float %688, i64 9, !dbg !50
  %756 = insertelement <32 x float> %755, float %689, i64 10, !dbg !50
  %757 = insertelement <32 x float> %756, float %690, i64 11, !dbg !50
  %758 = insertelement <32 x float> %757, float %691, i64 12, !dbg !50
  %759 = insertelement <32 x float> %758, float %692, i64 13, !dbg !50
  %760 = insertelement <32 x float> %759, float %693, i64 14, !dbg !50
  %761 = insertelement <32 x float> %760, float %694, i64 15, !dbg !50
  %762 = insertelement <32 x float> %761, float %695, i64 16, !dbg !50
  %763 = insertelement <32 x float> %762, float %696, i64 17, !dbg !50
  %764 = insertelement <32 x float> %763, float %697, i64 18, !dbg !50
  %765 = insertelement <32 x float> %764, float %698, i64 19, !dbg !50
  %766 = insertelement <32 x float> %765, float %699, i64 20, !dbg !50
  %767 = insertelement <32 x float> %766, float %700, i64 21, !dbg !50
  %768 = insertelement <32 x float> %767, float %701, i64 22, !dbg !50
  %769 = insertelement <32 x float> %768, float %702, i64 23, !dbg !50
  %770 = insertelement <32 x float> %769, float %703, i64 24, !dbg !50
  %771 = insertelement <32 x float> %770, float %704, i64 25, !dbg !50
  %772 = insertelement <32 x float> %771, float %705, i64 26, !dbg !50
  %773 = insertelement <32 x float> %772, float %706, i64 27, !dbg !50
  %774 = insertelement <32 x float> %773, float %707, i64 28, !dbg !50
  %775 = insertelement <32 x float> %774, float %708, i64 29, !dbg !50
  %776 = insertelement <32 x float> %775, float %709, i64 30, !dbg !50
  %777 = insertelement <32 x float> %776, float %710, i64 31, !dbg !50
  %778 = insertvalue [2 x <32 x float>] poison, <32 x float> %777, 0, !dbg !50
  %779 = insertelement <32 x float> poison, float %714, i64 0, !dbg !50
  %780 = insertelement <32 x float> %779, float %715, i64 1, !dbg !50
  %781 = insertelement <32 x float> %780, float %716, i64 2, !dbg !50
  %782 = insertelement <32 x float> %781, float %717, i64 3, !dbg !50
  %783 = insertelement <32 x float> %782, float %718, i64 4, !dbg !50
  %784 = insertelement <32 x float> %783, float %719, i64 5, !dbg !50
  %785 = insertelement <32 x float> %784, float %720, i64 6, !dbg !50
  %786 = insertelement <32 x float> %785, float %721, i64 7, !dbg !50
  %787 = insertelement <32 x float> %786, float %722, i64 8, !dbg !50
  %788 = insertelement <32 x float> %787, float %723, i64 9, !dbg !50
  %789 = insertelement <32 x float> %788, float %724, i64 10, !dbg !50
  %790 = insertelement <32 x float> %789, float %725, i64 11, !dbg !50
  %791 = insertelement <32 x float> %790, float %726, i64 12, !dbg !50
  %792 = insertelement <32 x float> %791, float %727, i64 13, !dbg !50
  %793 = insertelement <32 x float> %792, float %728, i64 14, !dbg !50
  %794 = insertelement <32 x float> %793, float %729, i64 15, !dbg !50
  %795 = insertelement <32 x float> %794, float %730, i64 16, !dbg !50
  %796 = insertelement <32 x float> %795, float %731, i64 17, !dbg !50
  %797 = insertelement <32 x float> %796, float %732, i64 18, !dbg !50
  %798 = insertelement <32 x float> %797, float %733, i64 19, !dbg !50
  %799 = insertelement <32 x float> %798, float %734, i64 20, !dbg !50
  %800 = insertelement <32 x float> %799, float %735, i64 21, !dbg !50
  %801 = insertelement <32 x float> %800, float %736, i64 22, !dbg !50
  %802 = insertelement <32 x float> %801, float %737, i64 23, !dbg !50
  %803 = insertelement <32 x float> %802, float %738, i64 24, !dbg !50
  %804 = insertelement <32 x float> %803, float %739, i64 25, !dbg !50
  %805 = insertelement <32 x float> %804, float %740, i64 26, !dbg !50
  %806 = insertelement <32 x float> %805, float %741, i64 27, !dbg !50
  %807 = insertelement <32 x float> %806, float %742, i64 28, !dbg !50
  %808 = insertelement <32 x float> %807, float %743, i64 29, !dbg !50
  %809 = insertelement <32 x float> %808, float %744, i64 30, !dbg !50
  %810 = insertelement <32 x float> %809, float %745, i64 31, !dbg !50
  %811 = insertvalue [2 x <32 x float>] %778, <32 x float> %810, 1, !dbg !50
  br label %812, !dbg !50

812:                                              ; preds = %671, %62
  %813 = phi [2 x <32 x float>] [ %811, %671 ], [ zeroinitializer, %62 ], !dbg !50
  %814 = extractvalue [2 x <32 x float>] %813, 0, !dbg !62
  %815 = insertelement <4 x float> poison, float %604, i64 0, !dbg !62
  %816 = insertelement <4 x float> %815, float %605, i64 1, !dbg !62
  %817 = insertelement <4 x float> %816, float %606, i64 2, !dbg !62
  %818 = insertelement <4 x float> %817, float %607, i64 3, !dbg !62
  %819 = shufflevector <32 x float> %814, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %820 = fadd <4 x float> %818, %819, !dbg !62
  %821 = extractelement <4 x float> %820, i64 0, !dbg !62
  %822 = extractelement <4 x float> %820, i64 1, !dbg !62
  %823 = extractelement <4 x float> %820, i64 2, !dbg !62
  %824 = extractelement <4 x float> %820, i64 3, !dbg !62
  %825 = insertelement <4 x float> poison, float %608, i64 0, !dbg !62
  %826 = insertelement <4 x float> %825, float %609, i64 1, !dbg !62
  %827 = insertelement <4 x float> %826, float %610, i64 2, !dbg !62
  %828 = insertelement <4 x float> %827, float %611, i64 3, !dbg !62
  %829 = shufflevector <32 x float> %814, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %830 = fadd <4 x float> %828, %829, !dbg !62
  %831 = extractelement <4 x float> %830, i64 0, !dbg !62
  %832 = extractelement <4 x float> %830, i64 1, !dbg !62
  %833 = extractelement <4 x float> %830, i64 2, !dbg !62
  %834 = extractelement <4 x float> %830, i64 3, !dbg !62
  %835 = insertelement <4 x float> poison, float %612, i64 0, !dbg !62
  %836 = insertelement <4 x float> %835, float %613, i64 1, !dbg !62
  %837 = insertelement <4 x float> %836, float %614, i64 2, !dbg !62
  %838 = insertelement <4 x float> %837, float %615, i64 3, !dbg !62
  %839 = shufflevector <32 x float> %814, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !62
  %840 = fadd <4 x float> %838, %839, !dbg !62
  %841 = extractelement <4 x float> %840, i64 0, !dbg !62
  %842 = extractelement <4 x float> %840, i64 1, !dbg !62
  %843 = extractelement <4 x float> %840, i64 2, !dbg !62
  %844 = extractelement <4 x float> %840, i64 3, !dbg !62
  %845 = insertelement <4 x float> poison, float %616, i64 0, !dbg !62
  %846 = insertelement <4 x float> %845, float %617, i64 1, !dbg !62
  %847 = insertelement <4 x float> %846, float %618, i64 2, !dbg !62
  %848 = insertelement <4 x float> %847, float %619, i64 3, !dbg !62
  %849 = shufflevector <32 x float> %814, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !62
  %850 = fadd <4 x float> %848, %849, !dbg !62
  %851 = extractelement <4 x float> %850, i64 0, !dbg !62
  %852 = extractelement <4 x float> %850, i64 1, !dbg !62
  %853 = extractelement <4 x float> %850, i64 2, !dbg !62
  %854 = extractelement <4 x float> %850, i64 3, !dbg !62
  %855 = insertelement <4 x float> poison, float %620, i64 0, !dbg !62
  %856 = insertelement <4 x float> %855, float %621, i64 1, !dbg !62
  %857 = insertelement <4 x float> %856, float %622, i64 2, !dbg !62
  %858 = insertelement <4 x float> %857, float %623, i64 3, !dbg !62
  %859 = shufflevector <32 x float> %814, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !62
  %860 = fadd <4 x float> %858, %859, !dbg !62
  %861 = extractelement <4 x float> %860, i64 0, !dbg !62
  %862 = extractelement <4 x float> %860, i64 1, !dbg !62
  %863 = extractelement <4 x float> %860, i64 2, !dbg !62
  %864 = extractelement <4 x float> %860, i64 3, !dbg !62
  %865 = insertelement <4 x float> poison, float %624, i64 0, !dbg !62
  %866 = insertelement <4 x float> %865, float %625, i64 1, !dbg !62
  %867 = insertelement <4 x float> %866, float %626, i64 2, !dbg !62
  %868 = insertelement <4 x float> %867, float %627, i64 3, !dbg !62
  %869 = shufflevector <32 x float> %814, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !62
  %870 = fadd <4 x float> %868, %869, !dbg !62
  %871 = extractelement <4 x float> %870, i64 0, !dbg !62
  %872 = extractelement <4 x float> %870, i64 1, !dbg !62
  %873 = extractelement <4 x float> %870, i64 2, !dbg !62
  %874 = extractelement <4 x float> %870, i64 3, !dbg !62
  %875 = insertelement <4 x float> poison, float %628, i64 0, !dbg !62
  %876 = insertelement <4 x float> %875, float %629, i64 1, !dbg !62
  %877 = insertelement <4 x float> %876, float %630, i64 2, !dbg !62
  %878 = insertelement <4 x float> %877, float %631, i64 3, !dbg !62
  %879 = shufflevector <32 x float> %814, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !62
  %880 = fadd <4 x float> %878, %879, !dbg !62
  %881 = extractelement <4 x float> %880, i64 0, !dbg !62
  %882 = extractelement <4 x float> %880, i64 1, !dbg !62
  %883 = extractelement <4 x float> %880, i64 2, !dbg !62
  %884 = extractelement <4 x float> %880, i64 3, !dbg !62
  %885 = insertelement <4 x float> poison, float %632, i64 0, !dbg !62
  %886 = insertelement <4 x float> %885, float %633, i64 1, !dbg !62
  %887 = insertelement <4 x float> %886, float %634, i64 2, !dbg !62
  %888 = insertelement <4 x float> %887, float %635, i64 3, !dbg !62
  %889 = shufflevector <32 x float> %814, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !62
  %890 = fadd <4 x float> %888, %889, !dbg !62
  %891 = extractelement <4 x float> %890, i64 0, !dbg !62
  %892 = extractelement <4 x float> %890, i64 1, !dbg !62
  %893 = extractelement <4 x float> %890, i64 2, !dbg !62
  %894 = extractelement <4 x float> %890, i64 3, !dbg !62
  %895 = extractvalue [2 x <32 x float>] %813, 1, !dbg !62
  %896 = insertelement <4 x float> poison, float %636, i64 0, !dbg !62
  %897 = insertelement <4 x float> %896, float %637, i64 1, !dbg !62
  %898 = insertelement <4 x float> %897, float %638, i64 2, !dbg !62
  %899 = insertelement <4 x float> %898, float %639, i64 3, !dbg !62
  %900 = shufflevector <32 x float> %895, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %901 = fadd <4 x float> %899, %900, !dbg !62
  %902 = extractelement <4 x float> %901, i64 0, !dbg !62
  %903 = extractelement <4 x float> %901, i64 1, !dbg !62
  %904 = extractelement <4 x float> %901, i64 2, !dbg !62
  %905 = extractelement <4 x float> %901, i64 3, !dbg !62
  %906 = insertelement <4 x float> poison, float %640, i64 0, !dbg !62
  %907 = insertelement <4 x float> %906, float %641, i64 1, !dbg !62
  %908 = insertelement <4 x float> %907, float %642, i64 2, !dbg !62
  %909 = insertelement <4 x float> %908, float %643, i64 3, !dbg !62
  %910 = shufflevector <32 x float> %895, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %911 = fadd <4 x float> %909, %910, !dbg !62
  %912 = extractelement <4 x float> %911, i64 0, !dbg !62
  %913 = extractelement <4 x float> %911, i64 1, !dbg !62
  %914 = extractelement <4 x float> %911, i64 2, !dbg !62
  %915 = extractelement <4 x float> %911, i64 3, !dbg !62
  %916 = insertelement <4 x float> poison, float %644, i64 0, !dbg !62
  %917 = insertelement <4 x float> %916, float %645, i64 1, !dbg !62
  %918 = insertelement <4 x float> %917, float %646, i64 2, !dbg !62
  %919 = insertelement <4 x float> %918, float %647, i64 3, !dbg !62
  %920 = shufflevector <32 x float> %895, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !62
  %921 = fadd <4 x float> %919, %920, !dbg !62
  %922 = extractelement <4 x float> %921, i64 0, !dbg !62
  %923 = extractelement <4 x float> %921, i64 1, !dbg !62
  %924 = extractelement <4 x float> %921, i64 2, !dbg !62
  %925 = extractelement <4 x float> %921, i64 3, !dbg !62
  %926 = insertelement <4 x float> poison, float %648, i64 0, !dbg !62
  %927 = insertelement <4 x float> %926, float %649, i64 1, !dbg !62
  %928 = insertelement <4 x float> %927, float %650, i64 2, !dbg !62
  %929 = insertelement <4 x float> %928, float %651, i64 3, !dbg !62
  %930 = shufflevector <32 x float> %895, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !62
  %931 = fadd <4 x float> %929, %930, !dbg !62
  %932 = extractelement <4 x float> %931, i64 0, !dbg !62
  %933 = extractelement <4 x float> %931, i64 1, !dbg !62
  %934 = extractelement <4 x float> %931, i64 2, !dbg !62
  %935 = extractelement <4 x float> %931, i64 3, !dbg !62
  %936 = insertelement <4 x float> poison, float %652, i64 0, !dbg !62
  %937 = insertelement <4 x float> %936, float %653, i64 1, !dbg !62
  %938 = insertelement <4 x float> %937, float %654, i64 2, !dbg !62
  %939 = insertelement <4 x float> %938, float %655, i64 3, !dbg !62
  %940 = shufflevector <32 x float> %895, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !62
  %941 = fadd <4 x float> %939, %940, !dbg !62
  %942 = extractelement <4 x float> %941, i64 0, !dbg !62
  %943 = extractelement <4 x float> %941, i64 1, !dbg !62
  %944 = extractelement <4 x float> %941, i64 2, !dbg !62
  %945 = extractelement <4 x float> %941, i64 3, !dbg !62
  %946 = insertelement <4 x float> poison, float %656, i64 0, !dbg !62
  %947 = insertelement <4 x float> %946, float %657, i64 1, !dbg !62
  %948 = insertelement <4 x float> %947, float %658, i64 2, !dbg !62
  %949 = insertelement <4 x float> %948, float %659, i64 3, !dbg !62
  %950 = shufflevector <32 x float> %895, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !62
  %951 = fadd <4 x float> %949, %950, !dbg !62
  %952 = extractelement <4 x float> %951, i64 0, !dbg !62
  %953 = extractelement <4 x float> %951, i64 1, !dbg !62
  %954 = extractelement <4 x float> %951, i64 2, !dbg !62
  %955 = extractelement <4 x float> %951, i64 3, !dbg !62
  %956 = insertelement <4 x float> poison, float %660, i64 0, !dbg !62
  %957 = insertelement <4 x float> %956, float %661, i64 1, !dbg !62
  %958 = insertelement <4 x float> %957, float %662, i64 2, !dbg !62
  %959 = insertelement <4 x float> %958, float %663, i64 3, !dbg !62
  %960 = shufflevector <32 x float> %895, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !62
  %961 = fadd <4 x float> %959, %960, !dbg !62
  %962 = extractelement <4 x float> %961, i64 0, !dbg !62
  %963 = extractelement <4 x float> %961, i64 1, !dbg !62
  %964 = extractelement <4 x float> %961, i64 2, !dbg !62
  %965 = extractelement <4 x float> %961, i64 3, !dbg !62
  %966 = insertelement <4 x float> poison, float %664, i64 0, !dbg !62
  %967 = insertelement <4 x float> %966, float %665, i64 1, !dbg !62
  %968 = insertelement <4 x float> %967, float %666, i64 2, !dbg !62
  %969 = insertelement <4 x float> %968, float %667, i64 3, !dbg !62
  %970 = shufflevector <32 x float> %895, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !62
  %971 = fadd <4 x float> %969, %970, !dbg !62
  %972 = extractelement <4 x float> %971, i64 0, !dbg !62
  %973 = extractelement <4 x float> %971, i64 1, !dbg !62
  %974 = extractelement <4 x float> %971, i64 2, !dbg !62
  %975 = extractelement <4 x float> %971, i64 3, !dbg !62
  %976 = call float @llvm.maxnum.f32(float %821, float 0xFFF0000000000000), !dbg !63, !frisk.local_reduction !43
  %local.reduce182 = call float @llvm.maxnum.f32(float %976, float %831), !dbg !63, !frisk.local_reduction !43
  %local.reduce183 = call float @llvm.maxnum.f32(float %822, float %832), !dbg !63, !frisk.local_reduction !43
  %local.reduce184 = call float @llvm.maxnum.f32(float %823, float %833), !dbg !63, !frisk.local_reduction !43
  %local.reduce185 = call float @llvm.maxnum.f32(float %824, float %834), !dbg !63, !frisk.local_reduction !43
  %local.reduce186 = call float @llvm.maxnum.f32(float %local.reduce182, float %841), !dbg !63, !frisk.local_reduction !43
  %local.reduce187 = call float @llvm.maxnum.f32(float %local.reduce183, float %842), !dbg !63, !frisk.local_reduction !43
  %local.reduce188 = call float @llvm.maxnum.f32(float %local.reduce184, float %843), !dbg !63, !frisk.local_reduction !43
  %local.reduce189 = call float @llvm.maxnum.f32(float %local.reduce185, float %844), !dbg !63, !frisk.local_reduction !43
  %local.reduce190 = call float @llvm.maxnum.f32(float %local.reduce186, float %851), !dbg !63, !frisk.local_reduction !43
  %local.reduce191 = call float @llvm.maxnum.f32(float %local.reduce187, float %852), !dbg !63, !frisk.local_reduction !43
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %853), !dbg !63, !frisk.local_reduction !43
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %854), !dbg !63, !frisk.local_reduction !43
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %861), !dbg !63, !frisk.local_reduction !43
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %862), !dbg !63, !frisk.local_reduction !43
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %863), !dbg !63, !frisk.local_reduction !43
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %864), !dbg !63, !frisk.local_reduction !43
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %871), !dbg !63, !frisk.local_reduction !43
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %872), !dbg !63, !frisk.local_reduction !43
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %873), !dbg !63, !frisk.local_reduction !43
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %874), !dbg !63, !frisk.local_reduction !43
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %881), !dbg !63, !frisk.local_reduction !43
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %882), !dbg !63, !frisk.local_reduction !43
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %883), !dbg !63, !frisk.local_reduction !43
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %884), !dbg !63, !frisk.local_reduction !43
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %891), !dbg !63, !frisk.local_reduction !43
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %892), !dbg !63, !frisk.local_reduction !43
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %893), !dbg !63, !frisk.local_reduction !43
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %894), !dbg !63, !frisk.local_reduction !43
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %local.reduce207), !dbg !63, !frisk.local_reduction !43
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce208, float %local.reduce209), !dbg !63, !frisk.local_reduction !43
  %977 = call float @llvm.maxnum.f32(float %local.reduce210, float %local.reduce211), !dbg !63, !frisk.local_reduction !43
  %978 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !63
  %979 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %978), !dbg !63
  %980 = add nuw nsw i32 %979, 64, !dbg !63
  %981 = xor i32 %979, 16, !dbg !63
  %982 = and i32 %980, 192, !dbg !63
  %983 = icmp samesign ult i32 %981, %982, !dbg !63
  %984 = select i1 %983, i32 %981, i32 %979, !dbg !63
  %985 = shl nuw nsw i32 %984, 2, !dbg !63
  %986 = bitcast float %977 to i32, !dbg !63
  %987 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %986), !dbg !63
  %988 = bitcast i32 %987 to float, !dbg !63
  %989 = call float @llvm.maxnum.f32(float %977, float %988), !dbg !63
  %990 = xor i32 %979, 32, !dbg !63
  %991 = icmp samesign ult i32 %990, %982, !dbg !63
  %992 = select i1 %991, i32 %990, i32 %979, !dbg !63
  %993 = shl nuw nsw i32 %992, 2, !dbg !63
  %994 = bitcast float %989 to i32, !dbg !63
  %995 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %994), !dbg !63
  %996 = bitcast i32 %995 to float, !dbg !63
  %997 = call float @llvm.maxnum.f32(float %989, float %996), !dbg !63
  %998 = extractvalue [2 x <1 x float>] %57, 0, !dbg !63
  %999 = insertelement <1 x float> poison, float %997, i64 0, !dbg !63
  %1000 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %999, <1 x float> %998), !dbg !63
  %1001 = call float @llvm.maxnum.f32(float %902, float 0xFFF0000000000000), !dbg !63, !frisk.local_reduction !43
  %local.reduce151 = call float @llvm.maxnum.f32(float %1001, float %912), !dbg !63, !frisk.local_reduction !43
  %local.reduce152 = call float @llvm.maxnum.f32(float %903, float %913), !dbg !63, !frisk.local_reduction !43
  %local.reduce153 = call float @llvm.maxnum.f32(float %904, float %914), !dbg !63, !frisk.local_reduction !43
  %local.reduce154 = call float @llvm.maxnum.f32(float %905, float %915), !dbg !63, !frisk.local_reduction !43
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %922), !dbg !63, !frisk.local_reduction !43
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %923), !dbg !63, !frisk.local_reduction !43
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %924), !dbg !63, !frisk.local_reduction !43
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %925), !dbg !63, !frisk.local_reduction !43
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %932), !dbg !63, !frisk.local_reduction !43
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %933), !dbg !63, !frisk.local_reduction !43
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %934), !dbg !63, !frisk.local_reduction !43
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %935), !dbg !63, !frisk.local_reduction !43
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %942), !dbg !63, !frisk.local_reduction !43
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %943), !dbg !63, !frisk.local_reduction !43
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %944), !dbg !63, !frisk.local_reduction !43
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %945), !dbg !63, !frisk.local_reduction !43
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %952), !dbg !63, !frisk.local_reduction !43
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %953), !dbg !63, !frisk.local_reduction !43
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %954), !dbg !63, !frisk.local_reduction !43
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %955), !dbg !63, !frisk.local_reduction !43
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %962), !dbg !63, !frisk.local_reduction !43
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %963), !dbg !63, !frisk.local_reduction !43
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %964), !dbg !63, !frisk.local_reduction !43
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %965), !dbg !63, !frisk.local_reduction !43
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %972), !dbg !63, !frisk.local_reduction !43
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %973), !dbg !63, !frisk.local_reduction !43
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %974), !dbg !63, !frisk.local_reduction !43
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %975), !dbg !63, !frisk.local_reduction !43
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !63, !frisk.local_reduction !43
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !63, !frisk.local_reduction !43
  %1002 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !63, !frisk.local_reduction !43
  %1003 = bitcast float %1002 to i32, !dbg !63
  %1004 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1003), !dbg !63
  %1005 = bitcast i32 %1004 to float, !dbg !63
  %1006 = call float @llvm.maxnum.f32(float %1002, float %1005), !dbg !63
  %1007 = bitcast float %1006 to i32, !dbg !63
  %1008 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1007), !dbg !63
  %1009 = bitcast i32 %1008 to float, !dbg !63
  %1010 = call float @llvm.maxnum.f32(float %1006, float %1009), !dbg !63
  %1011 = extractvalue [2 x <1 x float>] %57, 1, !dbg !63
  %1012 = insertelement <1 x float> poison, float %1010, i64 0, !dbg !63
  %1013 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1012, <1 x float> %1011), !dbg !63
  %1014 = shufflevector <1 x float> %1000, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !64
  %1015 = fsub <4 x float> %820, %1014, !dbg !64
  %1016 = fsub <4 x float> %830, %1014, !dbg !64
  %1017 = fsub <4 x float> %840, %1014, !dbg !64
  %1018 = fsub <4 x float> %850, %1014, !dbg !64
  %1019 = fsub <4 x float> %860, %1014, !dbg !64
  %1020 = fsub <4 x float> %870, %1014, !dbg !64
  %1021 = fsub <4 x float> %880, %1014, !dbg !64
  %1022 = fsub <4 x float> %890, %1014, !dbg !64
  %1023 = shufflevector <1 x float> %1013, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !64
  %1024 = fsub <4 x float> %901, %1023, !dbg !64
  %1025 = fsub <4 x float> %911, %1023, !dbg !64
  %1026 = fsub <4 x float> %921, %1023, !dbg !64
  %1027 = fsub <4 x float> %931, %1023, !dbg !64
  %1028 = fsub <4 x float> %941, %1023, !dbg !64
  %1029 = fsub <4 x float> %951, %1023, !dbg !64
  %1030 = fsub <4 x float> %961, %1023, !dbg !64
  %1031 = fsub <4 x float> %971, %1023, !dbg !64
  br i1 %43, label %1032, label %1039, !dbg !51

1032:                                             ; preds = %812
  %1033 = fsub <1 x float> %1011, %1013, !dbg !65
  %1034 = extractelement <1 x float> %1033, i64 0, !dbg !65
  %1035 = fsub <1 x float> %998, %1000, !dbg !65
  %1036 = extractelement <1 x float> %1035, i64 0, !dbg !65
  %address.thread766 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !51, !range !49, !frisk.register_lifetimes !43
  %address.local767 = lshr i32 %address.thread766, 2, !dbg !55
  %address.local768 = and i32 %address.local767, 48, !dbg !55
  %address.local769 = and i32 %address.thread766, 15, !dbg !55
  %address.local770 = or disjoint i32 %address.local768, %address.local769, !dbg !55
  %address.local771 = zext nneg i32 %address.local770 to i64, !dbg !55
  %address.local772 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local771, !dbg !51
  store float %1036, ptr addrspace(3) %address.local772, align 4, !dbg !51
  %address.local773 = lshr i32 %address.thread766, 2, !dbg !55
  %address.local774 = and i32 %address.local773, 48, !dbg !55
  %address.local775 = and i32 %address.thread766, 15, !dbg !55
  %address.local776 = or disjoint i32 %address.local774, %address.local775, !dbg !55
  %1037 = zext nneg i32 %address.local776 to i64, !dbg !51
  %1038 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %1037, !dbg !51
  %address.local779 = getelementptr inbounds nuw i8, ptr addrspace(3) %1038, i64 256, !dbg !51
  store float %1034, ptr addrspace(3) %address.local779, align 4, !dbg !51
  br label %1039, !dbg !51

1039:                                             ; preds = %1032, %812
  fence syncscope("workgroup") release, !dbg !51
  call void @llvm.amdgcn.s.barrier(), !dbg !51
  fence syncscope("workgroup") acquire, !dbg !51
  %address.thread780 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !51, !range !49, !frisk.register_lifetimes !43
  %address.local781 = lshr i32 %address.thread780, 1, !dbg !51
  %address.local782 = and i32 %address.local781, 96, !dbg !51
  %address.local783 = and i32 %address.thread780, 15, !dbg !51
  %address.local784 = or disjoint i32 %address.local782, %address.local783, !dbg !51
  %address.local785 = zext nneg i32 %address.local784 to i64, !dbg !51
  %address.local786 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local785, !dbg !51
  %1040 = load float, ptr addrspace(3) %address.local786, align 4, !dbg !51
  %lds.const.gep1745 = getelementptr i8, ptr addrspace(3) %address.local786, i64 64, !dbg !51
  %1041 = load float, ptr addrspace(3) %lds.const.gep1745, align 4, !dbg !51
  fence syncscope("workgroup") release, !dbg !51
  call void @llvm.amdgcn.s.barrier(), !dbg !51
  fence syncscope("workgroup") acquire, !dbg !51
  %1042 = call float @llvm.amdgcn.exp2.f32(float %1040), !dbg !51
  %1043 = call float @llvm.amdgcn.exp2.f32(float %1041), !dbg !51
  br i1 %43, label %1044, label %1045, !dbg !66

1044:                                             ; preds = %1039
  %address.thread789 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !66, !range !49, !frisk.register_lifetimes !43
  %address.local790 = lshr i32 %address.thread789, 1, !dbg !51
  %address.local791 = and i32 %address.local790, 96, !dbg !51
  %address.local792 = and i32 %address.thread789, 15, !dbg !51
  %address.local793 = or disjoint i32 %address.local791, %address.local792, !dbg !51
  %address.local794 = zext nneg i32 %address.local793 to i64, !dbg !51
  %address.local795 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !51
  store float %1042, ptr addrspace(3) %address.local795, align 4, !dbg !66
  %address.local796 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !51
  %address.local797 = getelementptr i8, ptr addrspace(3) %address.local796, i64 64, !dbg !51
  store float %1043, ptr addrspace(3) %address.local797, align 4, !dbg !66
  br label %1045, !dbg !66

1045:                                             ; preds = %1044, %1039
  fence syncscope("workgroup") release, !dbg !66
  call void @llvm.amdgcn.s.barrier(), !dbg !66
  fence syncscope("workgroup") acquire, !dbg !66
  %address.thread798 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !66, !range !49, !frisk.register_lifetimes !43
  %address.local799 = lshr i32 %address.thread798, 2, !dbg !55
  %address.local800 = and i32 %address.local799, 48, !dbg !55
  %address.local801 = and i32 %address.thread798, 15, !dbg !55
  %address.local802 = or disjoint i32 %address.local800, %address.local801, !dbg !55
  %address.local803 = zext nneg i32 %address.local802 to i64, !dbg !55
  %address.local804 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local803, !dbg !66
  %1046 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !66
  %lds.const.gep1746 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !66
  %1047 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !66
  fence syncscope("workgroup") release, !dbg !66
  call void @llvm.amdgcn.s.barrier(), !dbg !66
  fence syncscope("workgroup") acquire, !dbg !66
  %1048 = add nuw nsw i64 %76, %15, !dbg !67
  %1049 = shl i64 %1048, 19, !dbg !67
  %1050 = and i64 %1049, 16252928, !dbg !67
  %address.local812 = zext nneg i32 %address.thread798 to i64, !dbg !48
  %address.local813 = shl nuw nsw i64 %address.local812, 3, !dbg !47
  %address.local814 = and i64 %address.local813, 1536, !dbg !47
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local814, !dbg !47
  %address.local816 = ptrtoint ptr addrspace(3) %address.local815 to i64, !dbg !47
  %address.local817 = trunc i64 %address.local816 to i32, !dbg !47
  %1051 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local817), !dbg !67
  %address.local818 = shl nuw nsw i64 %address.local812, 6, !dbg !67
  %address.local819 = and i64 %address.local818, 16128, !dbg !67
  %address.local820 = shl nuw nsw i64 %address.local812, 4, !dbg !67
  %address.local821 = and i64 %address.local820, 48, !dbg !67
  %1052 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !67
  %.idx48 = add i64 %1050, %82, !dbg !67
  %1053 = shl i64 %.idx48, 1, !dbg !67
  %1054 = or disjoint i64 %address.local819, %address.local821, !dbg !67
  %1055 = or disjoint i64 %1053, %1054, !dbg !67
  %dma.buffer.offset1773 = trunc i64 %1055 to i32, !dbg !67
  %1056 = insertelement <2 x i64> poison, i64 %1052, i64 0, !dbg !67
  %1057 = bitcast <2 x i64> %1056 to <4 x i32>, !dbg !67
  %dma.buffer.rsrc1774 = shufflevector <4 x i32> %1057, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !67
  %dma.buffer.lds1775 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1051), !dbg !67
  %1058 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1775, i32 %dma.buffer.offset1773, <4 x i32> %dma.buffer.rsrc1774) #10, !dbg !67
  %address.local824 = or disjoint i64 %address.local812, 256, !dbg !47
  %address.local825 = shl nuw nsw i64 %address.local824, 3, !dbg !47
  %address.local826 = and i64 %address.local825, 3584, !dbg !47
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local826, !dbg !47
  %address.local828 = ptrtoint ptr addrspace(3) %address.local827 to i64, !dbg !47
  %address.local829 = trunc i64 %address.local828 to i32, !dbg !47
  %1059 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local829), !dbg !67
  %address.local830 = shl nuw nsw i64 %address.local824, 6, !dbg !67
  %address.local831 = and i64 %address.local830, 32512, !dbg !67
  %address.local832 = shl nuw nsw i64 %address.local812, 4, !dbg !67
  %address.local833 = and i64 %address.local832, 48, !dbg !67
  %1060 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !67
  %.idx51 = add i64 %1050, %82, !dbg !67
  %1061 = shl i64 %.idx51, 1, !dbg !67
  %1062 = or disjoint i64 %1061, %address.local831, !dbg !67
  %1063 = or disjoint i64 %1062, %address.local833, !dbg !67
  %dma.buffer.offset1770 = trunc i64 %1063 to i32, !dbg !67
  %1064 = insertelement <2 x i64> poison, i64 %1060, i64 0, !dbg !67
  %1065 = bitcast <2 x i64> %1064 to <4 x i32>, !dbg !67
  %dma.buffer.rsrc1771 = shufflevector <4 x i32> %1065, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !67
  %dma.buffer.lds1772 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1059), !dbg !67
  %1066 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1772, i32 %dma.buffer.offset1770, <4 x i32> %dma.buffer.rsrc1771) #10, !dbg !67
  %1067 = extractvalue [2 x <1 x float>] %56, 0, !dbg !66
  %1068 = extractelement <1 x float> %1067, i64 0, !dbg !66
  %1069 = fmul float %1068, %1046, !dbg !66
  %1070 = extractvalue [2 x <1 x float>] %56, 1, !dbg !66
  %1071 = extractelement <1 x float> %1070, i64 0, !dbg !66
  %1072 = fmul float %1071, %1047, !dbg !66
  %1073 = extractelement <4 x float> %1015, i64 0, !dbg !68
  %1074 = call float @llvm.amdgcn.exp2.f32(float %1073), !dbg !68
  %1075 = insertelement <4 x float> poison, float %1074, i64 0, !dbg !69
  %1076 = extractelement <4 x float> %1015, i64 1, !dbg !68
  %1077 = call float @llvm.amdgcn.exp2.f32(float %1076), !dbg !68
  %1078 = insertelement <4 x float> %1075, float %1077, i64 1, !dbg !69
  %1079 = extractelement <4 x float> %1015, i64 2, !dbg !68
  %1080 = call float @llvm.amdgcn.exp2.f32(float %1079), !dbg !68
  %1081 = insertelement <4 x float> %1078, float %1080, i64 2, !dbg !69
  %1082 = extractelement <4 x float> %1015, i64 3, !dbg !68
  %1083 = call float @llvm.amdgcn.exp2.f32(float %1082), !dbg !68
  %1084 = insertelement <4 x float> %1081, float %1083, i64 3, !dbg !69
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1084) #9, !dbg !69, !frisk.register_lifetimes !43
  %1085 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !69
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1085) #9, !dbg !68, !frisk.register_lifetimes !43
  %1086 = extractelement <4 x float> %1016, i64 0, !dbg !68
  %1087 = call float @llvm.amdgcn.exp2.f32(float %1086), !dbg !68
  %1088 = insertelement <4 x float> poison, float %1087, i64 0, !dbg !69
  %local.reduce120 = fadd float %1074, %1087, !dbg !70, !frisk.local_reduction !43
  %1089 = extractelement <4 x float> %1016, i64 1, !dbg !68
  %1090 = call float @llvm.amdgcn.exp2.f32(float %1089), !dbg !68
  %1091 = insertelement <4 x float> %1088, float %1090, i64 1, !dbg !69
  %local.reduce121 = fadd float %1077, %1090, !dbg !70, !frisk.local_reduction !43
  %1092 = extractelement <4 x float> %1016, i64 2, !dbg !68
  %1093 = call float @llvm.amdgcn.exp2.f32(float %1092), !dbg !68
  %1094 = insertelement <4 x float> %1091, float %1093, i64 2, !dbg !69
  %local.reduce122 = fadd float %1080, %1093, !dbg !70, !frisk.local_reduction !43
  %1095 = extractelement <4 x float> %1016, i64 3, !dbg !68
  %1096 = call float @llvm.amdgcn.exp2.f32(float %1095), !dbg !68
  %1097 = insertelement <4 x float> %1094, float %1096, i64 3, !dbg !69
  %math.result228 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1097) #9, !dbg !69, !frisk.register_lifetimes !43
  %1098 = fptrunc <4 x float> %math.result228 to <4 x half>, !dbg !69
  %math.packed229 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1098) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce123 = fadd float %1083, %1096, !dbg !70, !frisk.local_reduction !43
  %1099 = extractelement <4 x float> %1017, i64 0, !dbg !68
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !68
  %1101 = insertelement <4 x float> poison, float %1100, i64 0, !dbg !69
  %local.reduce124 = fadd float %local.reduce120, %1100, !dbg !70, !frisk.local_reduction !43
  %1102 = extractelement <4 x float> %1017, i64 1, !dbg !68
  %1103 = call float @llvm.amdgcn.exp2.f32(float %1102), !dbg !68
  %1104 = insertelement <4 x float> %1101, float %1103, i64 1, !dbg !69
  %local.reduce125 = fadd float %local.reduce121, %1103, !dbg !70, !frisk.local_reduction !43
  %1105 = extractelement <4 x float> %1017, i64 2, !dbg !68
  %1106 = call float @llvm.amdgcn.exp2.f32(float %1105), !dbg !68
  %1107 = insertelement <4 x float> %1104, float %1106, i64 2, !dbg !69
  %local.reduce126 = fadd float %local.reduce122, %1106, !dbg !70, !frisk.local_reduction !43
  %1108 = extractelement <4 x float> %1017, i64 3, !dbg !68
  %1109 = call float @llvm.amdgcn.exp2.f32(float %1108), !dbg !68
  %1110 = insertelement <4 x float> %1107, float %1109, i64 3, !dbg !69
  %math.result230 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1110) #9, !dbg !69, !frisk.register_lifetimes !43
  %1111 = fptrunc <4 x float> %math.result230 to <4 x half>, !dbg !69
  %math.packed231 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1111) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce127 = fadd float %local.reduce123, %1109, !dbg !70, !frisk.local_reduction !43
  %1112 = extractelement <4 x float> %1018, i64 0, !dbg !68
  %1113 = call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !68
  %1114 = insertelement <4 x float> poison, float %1113, i64 0, !dbg !69
  %local.reduce128 = fadd float %local.reduce124, %1113, !dbg !70, !frisk.local_reduction !43
  %1115 = extractelement <4 x float> %1018, i64 1, !dbg !68
  %1116 = call float @llvm.amdgcn.exp2.f32(float %1115), !dbg !68
  %1117 = insertelement <4 x float> %1114, float %1116, i64 1, !dbg !69
  %local.reduce129 = fadd float %local.reduce125, %1116, !dbg !70, !frisk.local_reduction !43
  %1118 = extractelement <4 x float> %1018, i64 2, !dbg !68
  %1119 = call float @llvm.amdgcn.exp2.f32(float %1118), !dbg !68
  %1120 = insertelement <4 x float> %1117, float %1119, i64 2, !dbg !69
  %local.reduce130 = fadd float %local.reduce126, %1119, !dbg !70, !frisk.local_reduction !43
  %1121 = extractelement <4 x float> %1018, i64 3, !dbg !68
  %1122 = call float @llvm.amdgcn.exp2.f32(float %1121), !dbg !68
  %1123 = insertelement <4 x float> %1120, float %1122, i64 3, !dbg !69
  %math.result232 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1123) #9, !dbg !69, !frisk.register_lifetimes !43
  %1124 = fptrunc <4 x float> %math.result232 to <4 x half>, !dbg !69
  %math.packed233 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1124) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce131 = fadd float %local.reduce127, %1122, !dbg !70, !frisk.local_reduction !43
  %1125 = extractelement <4 x float> %1019, i64 0, !dbg !68
  %1126 = call float @llvm.amdgcn.exp2.f32(float %1125), !dbg !68
  %1127 = insertelement <4 x float> poison, float %1126, i64 0, !dbg !69
  %local.reduce132 = fadd float %local.reduce128, %1126, !dbg !70, !frisk.local_reduction !43
  %1128 = extractelement <4 x float> %1019, i64 1, !dbg !68
  %1129 = call float @llvm.amdgcn.exp2.f32(float %1128), !dbg !68
  %1130 = insertelement <4 x float> %1127, float %1129, i64 1, !dbg !69
  %local.reduce133 = fadd float %local.reduce129, %1129, !dbg !70, !frisk.local_reduction !43
  %1131 = extractelement <4 x float> %1019, i64 2, !dbg !68
  %1132 = call float @llvm.amdgcn.exp2.f32(float %1131), !dbg !68
  %1133 = insertelement <4 x float> %1130, float %1132, i64 2, !dbg !69
  %local.reduce134 = fadd float %local.reduce130, %1132, !dbg !70, !frisk.local_reduction !43
  %1134 = extractelement <4 x float> %1019, i64 3, !dbg !68
  %1135 = call float @llvm.amdgcn.exp2.f32(float %1134), !dbg !68
  %1136 = insertelement <4 x float> %1133, float %1135, i64 3, !dbg !69
  %math.result234 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1136) #9, !dbg !69, !frisk.register_lifetimes !43
  %1137 = fptrunc <4 x float> %math.result234 to <4 x half>, !dbg !69
  %math.packed235 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1137) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce135 = fadd float %local.reduce131, %1135, !dbg !70, !frisk.local_reduction !43
  %1138 = extractelement <4 x float> %1020, i64 0, !dbg !68
  %1139 = call float @llvm.amdgcn.exp2.f32(float %1138), !dbg !68
  %1140 = insertelement <4 x float> poison, float %1139, i64 0, !dbg !69
  %local.reduce136 = fadd float %local.reduce132, %1139, !dbg !70, !frisk.local_reduction !43
  %1141 = extractelement <4 x float> %1020, i64 1, !dbg !68
  %1142 = call float @llvm.amdgcn.exp2.f32(float %1141), !dbg !68
  %1143 = insertelement <4 x float> %1140, float %1142, i64 1, !dbg !69
  %local.reduce137 = fadd float %local.reduce133, %1142, !dbg !70, !frisk.local_reduction !43
  %1144 = extractelement <4 x float> %1020, i64 2, !dbg !68
  %1145 = call float @llvm.amdgcn.exp2.f32(float %1144), !dbg !68
  %1146 = insertelement <4 x float> %1143, float %1145, i64 2, !dbg !69
  %local.reduce138 = fadd float %local.reduce134, %1145, !dbg !70, !frisk.local_reduction !43
  %1147 = extractelement <4 x float> %1020, i64 3, !dbg !68
  %1148 = call float @llvm.amdgcn.exp2.f32(float %1147), !dbg !68
  %1149 = insertelement <4 x float> %1146, float %1148, i64 3, !dbg !69
  %math.result236 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1149) #9, !dbg !69, !frisk.register_lifetimes !43
  %1150 = fptrunc <4 x float> %math.result236 to <4 x half>, !dbg !69
  %math.packed237 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1150) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce139 = fadd float %local.reduce135, %1148, !dbg !70, !frisk.local_reduction !43
  %1151 = extractelement <4 x float> %1021, i64 0, !dbg !68
  %1152 = call float @llvm.amdgcn.exp2.f32(float %1151), !dbg !68
  %1153 = insertelement <4 x float> poison, float %1152, i64 0, !dbg !69
  %local.reduce140 = fadd float %local.reduce136, %1152, !dbg !70, !frisk.local_reduction !43
  %1154 = extractelement <4 x float> %1021, i64 1, !dbg !68
  %1155 = call float @llvm.amdgcn.exp2.f32(float %1154), !dbg !68
  %1156 = insertelement <4 x float> %1153, float %1155, i64 1, !dbg !69
  %local.reduce141 = fadd float %local.reduce137, %1155, !dbg !70, !frisk.local_reduction !43
  %1157 = extractelement <4 x float> %1021, i64 2, !dbg !68
  %1158 = call float @llvm.amdgcn.exp2.f32(float %1157), !dbg !68
  %1159 = insertelement <4 x float> %1156, float %1158, i64 2, !dbg !69
  %local.reduce142 = fadd float %local.reduce138, %1158, !dbg !70, !frisk.local_reduction !43
  %1160 = extractelement <4 x float> %1021, i64 3, !dbg !68
  %1161 = call float @llvm.amdgcn.exp2.f32(float %1160), !dbg !68
  %1162 = insertelement <4 x float> %1159, float %1161, i64 3, !dbg !69
  %math.result238 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1162) #9, !dbg !69, !frisk.register_lifetimes !43
  %1163 = fptrunc <4 x float> %math.result238 to <4 x half>, !dbg !69
  %math.packed239 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1163) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce143 = fadd float %local.reduce139, %1161, !dbg !70, !frisk.local_reduction !43
  %1164 = extractelement <4 x float> %1022, i64 0, !dbg !68
  %1165 = call float @llvm.amdgcn.exp2.f32(float %1164), !dbg !68
  %1166 = insertelement <4 x float> poison, float %1165, i64 0, !dbg !69
  %local.reduce144 = fadd float %local.reduce140, %1165, !dbg !70, !frisk.local_reduction !43
  %1167 = extractelement <4 x float> %1022, i64 1, !dbg !68
  %1168 = call float @llvm.amdgcn.exp2.f32(float %1167), !dbg !68
  %1169 = insertelement <4 x float> %1166, float %1168, i64 1, !dbg !69
  %local.reduce145 = fadd float %local.reduce141, %1168, !dbg !70, !frisk.local_reduction !43
  %local.reduce148 = fadd float %local.reduce144, %local.reduce145, !dbg !70, !frisk.local_reduction !43
  %1170 = extractelement <4 x float> %1022, i64 2, !dbg !68
  %1171 = call float @llvm.amdgcn.exp2.f32(float %1170), !dbg !68
  %1172 = insertelement <4 x float> %1169, float %1171, i64 2, !dbg !69
  %local.reduce146 = fadd float %local.reduce142, %1171, !dbg !70, !frisk.local_reduction !43
  %1173 = extractelement <4 x float> %1022, i64 3, !dbg !68
  %1174 = call float @llvm.amdgcn.exp2.f32(float %1173), !dbg !68
  %1175 = insertelement <4 x float> %1172, float %1174, i64 3, !dbg !69
  %math.result240 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1175) #9, !dbg !69, !frisk.register_lifetimes !43
  %1176 = fptrunc <4 x float> %math.result240 to <4 x half>, !dbg !69
  %math.packed241 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1176) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce147 = fadd float %local.reduce143, %1174, !dbg !70, !frisk.local_reduction !43
  %local.reduce149 = fadd float %local.reduce146, %local.reduce147, !dbg !70, !frisk.local_reduction !43
  %1177 = fadd float %local.reduce148, %local.reduce149, !dbg !70, !frisk.local_reduction !43
  %1178 = extractelement <4 x float> %1024, i64 0, !dbg !68
  %1179 = call float @llvm.amdgcn.exp2.f32(float %1178), !dbg !68
  %1180 = insertelement <4 x float> poison, float %1179, i64 0, !dbg !69
  %1181 = extractelement <4 x float> %1024, i64 1, !dbg !68
  %1182 = call float @llvm.amdgcn.exp2.f32(float %1181), !dbg !68
  %1183 = insertelement <4 x float> %1180, float %1182, i64 1, !dbg !69
  %1184 = extractelement <4 x float> %1024, i64 2, !dbg !68
  %1185 = call float @llvm.amdgcn.exp2.f32(float %1184), !dbg !68
  %1186 = insertelement <4 x float> %1183, float %1185, i64 2, !dbg !69
  %1187 = extractelement <4 x float> %1024, i64 3, !dbg !68
  %1188 = call float @llvm.amdgcn.exp2.f32(float %1187), !dbg !68
  %1189 = insertelement <4 x float> %1186, float %1188, i64 3, !dbg !69
  %math.result242 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1189) #9, !dbg !69, !frisk.register_lifetimes !43
  %1190 = fptrunc <4 x float> %math.result242 to <4 x half>, !dbg !69
  %math.packed243 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1190) #9, !dbg !68, !frisk.register_lifetimes !43
  %1191 = extractelement <4 x float> %1025, i64 0, !dbg !68
  %1192 = call float @llvm.amdgcn.exp2.f32(float %1191), !dbg !68
  %1193 = insertelement <4 x float> poison, float %1192, i64 0, !dbg !69
  %local.reduce = fadd float %1179, %1192, !dbg !70, !frisk.local_reduction !43
  %1194 = extractelement <4 x float> %1025, i64 1, !dbg !68
  %1195 = call float @llvm.amdgcn.exp2.f32(float %1194), !dbg !68
  %1196 = insertelement <4 x float> %1193, float %1195, i64 1, !dbg !69
  %local.reduce90 = fadd float %1182, %1195, !dbg !70, !frisk.local_reduction !43
  %1197 = extractelement <4 x float> %1025, i64 2, !dbg !68
  %1198 = call float @llvm.amdgcn.exp2.f32(float %1197), !dbg !68
  %1199 = insertelement <4 x float> %1196, float %1198, i64 2, !dbg !69
  %local.reduce91 = fadd float %1185, %1198, !dbg !70, !frisk.local_reduction !43
  %1200 = extractelement <4 x float> %1025, i64 3, !dbg !68
  %1201 = call float @llvm.amdgcn.exp2.f32(float %1200), !dbg !68
  %1202 = insertelement <4 x float> %1199, float %1201, i64 3, !dbg !69
  %math.result244 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1202) #9, !dbg !69, !frisk.register_lifetimes !43
  %1203 = fptrunc <4 x float> %math.result244 to <4 x half>, !dbg !69
  %math.packed245 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1203) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce92 = fadd float %1188, %1201, !dbg !70, !frisk.local_reduction !43
  %1204 = extractelement <4 x float> %1026, i64 0, !dbg !68
  %1205 = call float @llvm.amdgcn.exp2.f32(float %1204), !dbg !68
  %1206 = insertelement <4 x float> poison, float %1205, i64 0, !dbg !69
  %local.reduce93 = fadd float %local.reduce, %1205, !dbg !70, !frisk.local_reduction !43
  %1207 = extractelement <4 x float> %1026, i64 1, !dbg !68
  %1208 = call float @llvm.amdgcn.exp2.f32(float %1207), !dbg !68
  %1209 = insertelement <4 x float> %1206, float %1208, i64 1, !dbg !69
  %local.reduce94 = fadd float %local.reduce90, %1208, !dbg !70, !frisk.local_reduction !43
  %1210 = extractelement <4 x float> %1026, i64 2, !dbg !68
  %1211 = call float @llvm.amdgcn.exp2.f32(float %1210), !dbg !68
  %1212 = insertelement <4 x float> %1209, float %1211, i64 2, !dbg !69
  %local.reduce95 = fadd float %local.reduce91, %1211, !dbg !70, !frisk.local_reduction !43
  %1213 = extractelement <4 x float> %1026, i64 3, !dbg !68
  %1214 = call float @llvm.amdgcn.exp2.f32(float %1213), !dbg !68
  %1215 = insertelement <4 x float> %1212, float %1214, i64 3, !dbg !69
  %math.result246 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1215) #9, !dbg !69, !frisk.register_lifetimes !43
  %1216 = fptrunc <4 x float> %math.result246 to <4 x half>, !dbg !69
  %math.packed247 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1216) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce96 = fadd float %local.reduce92, %1214, !dbg !70, !frisk.local_reduction !43
  %1217 = extractelement <4 x float> %1027, i64 0, !dbg !68
  %1218 = call float @llvm.amdgcn.exp2.f32(float %1217), !dbg !68
  %1219 = insertelement <4 x float> poison, float %1218, i64 0, !dbg !69
  %local.reduce97 = fadd float %local.reduce93, %1218, !dbg !70, !frisk.local_reduction !43
  %1220 = extractelement <4 x float> %1027, i64 1, !dbg !68
  %1221 = call float @llvm.amdgcn.exp2.f32(float %1220), !dbg !68
  %1222 = insertelement <4 x float> %1219, float %1221, i64 1, !dbg !69
  %local.reduce98 = fadd float %local.reduce94, %1221, !dbg !70, !frisk.local_reduction !43
  %1223 = extractelement <4 x float> %1027, i64 2, !dbg !68
  %1224 = call float @llvm.amdgcn.exp2.f32(float %1223), !dbg !68
  %1225 = insertelement <4 x float> %1222, float %1224, i64 2, !dbg !69
  %local.reduce99 = fadd float %local.reduce95, %1224, !dbg !70, !frisk.local_reduction !43
  %1226 = extractelement <4 x float> %1027, i64 3, !dbg !68
  %1227 = call float @llvm.amdgcn.exp2.f32(float %1226), !dbg !68
  %1228 = insertelement <4 x float> %1225, float %1227, i64 3, !dbg !69
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1228) #9, !dbg !69, !frisk.register_lifetimes !43
  %1229 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !69
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1229) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce100 = fadd float %local.reduce96, %1227, !dbg !70, !frisk.local_reduction !43
  %1230 = extractelement <4 x float> %1028, i64 0, !dbg !68
  %1231 = call float @llvm.amdgcn.exp2.f32(float %1230), !dbg !68
  %1232 = insertelement <4 x float> poison, float %1231, i64 0, !dbg !69
  %local.reduce101 = fadd float %local.reduce97, %1231, !dbg !70, !frisk.local_reduction !43
  %1233 = extractelement <4 x float> %1028, i64 1, !dbg !68
  %1234 = call float @llvm.amdgcn.exp2.f32(float %1233), !dbg !68
  %1235 = insertelement <4 x float> %1232, float %1234, i64 1, !dbg !69
  %local.reduce102 = fadd float %local.reduce98, %1234, !dbg !70, !frisk.local_reduction !43
  %1236 = extractelement <4 x float> %1028, i64 2, !dbg !68
  %1237 = call float @llvm.amdgcn.exp2.f32(float %1236), !dbg !68
  %1238 = insertelement <4 x float> %1235, float %1237, i64 2, !dbg !69
  %local.reduce103 = fadd float %local.reduce99, %1237, !dbg !70, !frisk.local_reduction !43
  %1239 = extractelement <4 x float> %1028, i64 3, !dbg !68
  %1240 = call float @llvm.amdgcn.exp2.f32(float %1239), !dbg !68
  %1241 = insertelement <4 x float> %1238, float %1240, i64 3, !dbg !69
  %math.result250 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1241) #9, !dbg !69, !frisk.register_lifetimes !43
  %1242 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !69
  %math.packed251 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1242) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce104 = fadd float %local.reduce100, %1240, !dbg !70, !frisk.local_reduction !43
  %1243 = extractelement <4 x float> %1029, i64 0, !dbg !68
  %1244 = call float @llvm.amdgcn.exp2.f32(float %1243), !dbg !68
  %1245 = insertelement <4 x float> poison, float %1244, i64 0, !dbg !69
  %local.reduce105 = fadd float %local.reduce101, %1244, !dbg !70, !frisk.local_reduction !43
  %1246 = extractelement <4 x float> %1029, i64 1, !dbg !68
  %1247 = call float @llvm.amdgcn.exp2.f32(float %1246), !dbg !68
  %1248 = insertelement <4 x float> %1245, float %1247, i64 1, !dbg !69
  %local.reduce106 = fadd float %local.reduce102, %1247, !dbg !70, !frisk.local_reduction !43
  %1249 = extractelement <4 x float> %1029, i64 2, !dbg !68
  %1250 = call float @llvm.amdgcn.exp2.f32(float %1249), !dbg !68
  %1251 = insertelement <4 x float> %1248, float %1250, i64 2, !dbg !69
  %local.reduce107 = fadd float %local.reduce103, %1250, !dbg !70, !frisk.local_reduction !43
  %1252 = extractelement <4 x float> %1029, i64 3, !dbg !68
  %1253 = call float @llvm.amdgcn.exp2.f32(float %1252), !dbg !68
  %1254 = insertelement <4 x float> %1251, float %1253, i64 3, !dbg !69
  %math.result252 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1254) #9, !dbg !69, !frisk.register_lifetimes !43
  %1255 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !69
  %math.packed253 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1255) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce108 = fadd float %local.reduce104, %1253, !dbg !70, !frisk.local_reduction !43
  %1256 = extractelement <4 x float> %1030, i64 0, !dbg !68
  %1257 = call float @llvm.amdgcn.exp2.f32(float %1256), !dbg !68
  %1258 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !69
  %local.reduce109 = fadd float %local.reduce105, %1257, !dbg !70, !frisk.local_reduction !43
  %1259 = extractelement <4 x float> %1030, i64 1, !dbg !68
  %1260 = call float @llvm.amdgcn.exp2.f32(float %1259), !dbg !68
  %1261 = insertelement <4 x float> %1258, float %1260, i64 1, !dbg !69
  %local.reduce110 = fadd float %local.reduce106, %1260, !dbg !70, !frisk.local_reduction !43
  %1262 = extractelement <4 x float> %1030, i64 2, !dbg !68
  %1263 = call float @llvm.amdgcn.exp2.f32(float %1262), !dbg !68
  %1264 = insertelement <4 x float> %1261, float %1263, i64 2, !dbg !69
  %local.reduce111 = fadd float %local.reduce107, %1263, !dbg !70, !frisk.local_reduction !43
  %1265 = extractelement <4 x float> %1030, i64 3, !dbg !68
  %1266 = call float @llvm.amdgcn.exp2.f32(float %1265), !dbg !68
  %1267 = insertelement <4 x float> %1264, float %1266, i64 3, !dbg !69
  %math.result254 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1267) #9, !dbg !69, !frisk.register_lifetimes !43
  %1268 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !69
  %math.packed255 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1268) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce112 = fadd float %local.reduce108, %1266, !dbg !70, !frisk.local_reduction !43
  %1269 = extractelement <4 x float> %1031, i64 0, !dbg !68
  %1270 = call float @llvm.amdgcn.exp2.f32(float %1269), !dbg !68
  %1271 = insertelement <4 x float> poison, float %1270, i64 0, !dbg !69
  %local.reduce113 = fadd float %local.reduce109, %1270, !dbg !70, !frisk.local_reduction !43
  %1272 = extractelement <4 x float> %1031, i64 1, !dbg !68
  %1273 = call float @llvm.amdgcn.exp2.f32(float %1272), !dbg !68
  %1274 = insertelement <4 x float> %1271, float %1273, i64 1, !dbg !69
  %local.reduce114 = fadd float %local.reduce110, %1273, !dbg !70, !frisk.local_reduction !43
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !70, !frisk.local_reduction !43
  %1275 = extractelement <4 x float> %1031, i64 2, !dbg !68
  %1276 = call float @llvm.amdgcn.exp2.f32(float %1275), !dbg !68
  %1277 = insertelement <4 x float> %1274, float %1276, i64 2, !dbg !69
  %local.reduce115 = fadd float %local.reduce111, %1276, !dbg !70, !frisk.local_reduction !43
  %1278 = extractelement <4 x float> %1031, i64 3, !dbg !68
  %1279 = call float @llvm.amdgcn.exp2.f32(float %1278), !dbg !68
  %1280 = insertelement <4 x float> %1277, float %1279, i64 3, !dbg !69
  %math.result256 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1280) #9, !dbg !69, !frisk.register_lifetimes !43
  %1281 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !69
  %math.packed257 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1281) #9, !dbg !70, !frisk.register_lifetimes !43
  %local.reduce116 = fadd float %local.reduce112, %1279, !dbg !70, !frisk.local_reduction !43
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !70, !frisk.local_reduction !43
  %1282 = fadd float %local.reduce117, %local.reduce118, !dbg !70, !frisk.local_reduction !43
  %1283 = bitcast float %1177 to i32, !dbg !70
  %1284 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1283), !dbg !70
  %1285 = bitcast i32 %1284 to float, !dbg !70
  %1286 = fadd float %1177, %1285, !dbg !70
  %1287 = bitcast float %1286 to i32, !dbg !70
  %1288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1287), !dbg !70
  %1289 = bitcast i32 %1288 to float, !dbg !70
  %1290 = fadd float %1286, %1289, !dbg !70
  %1291 = fadd float %1290, %1069, !dbg !70
  %1292 = bitcast float %1282 to i32, !dbg !70
  %1293 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1292), !dbg !70
  %1294 = bitcast i32 %1293 to float, !dbg !70
  %1295 = fadd float %1282, %1294, !dbg !70
  %1296 = bitcast float %1295 to i32, !dbg !70
  %1297 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1296), !dbg !70
  %1298 = bitcast i32 %1297 to float, !dbg !70
  %1299 = fadd float %1295, %1298, !dbg !70
  %1300 = fadd float %1299, %1072, !dbg !70
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !67
  fence syncscope("workgroup") release, !dbg !67
  call void @llvm.amdgcn.s.barrier(), !dbg !67
  fence syncscope("workgroup") acquire, !dbg !67
  %address.local836 = shl nuw nsw i32 %address.thread798, 3, !dbg !52
  %address.local837 = and i32 %address.local836, 504, !dbg !52
  %address.local838 = zext nneg i32 %address.local837 to i64, !dbg !52
  %address.local839 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !52
  %1301 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local839, i32 0) #0, !dbg !52
  %1302 = bitcast <4 x i32> %1301 to <8 x half>, !dbg !52
  %address.local840 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !52
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local840, i64 1024, !dbg !52
  %1303 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #0, !dbg !52
  %1304 = bitcast <4 x i32> %1303 to <8 x half>, !dbg !52
  %address.local842 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !52
  %address.local843 = getelementptr i8, ptr addrspace(3) %address.local842, i64 2048, !dbg !52
  %1305 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local843, i32 0) #0, !dbg !52
  %1306 = bitcast <4 x i32> %1305 to <8 x half>, !dbg !52
  %address.local844 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !52
  %address.local845 = getelementptr i8, ptr addrspace(3) %address.local844, i64 3072, !dbg !52
  %1307 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local845, i32 0) #0, !dbg !52
  %1308 = bitcast <4 x i32> %1307 to <8 x half>, !dbg !52
  %address.local846 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !52
  %address.local847 = getelementptr i8, ptr addrspace(3) %address.local846, i64 4096, !dbg !52
  %1309 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local847, i32 0) #0, !dbg !52
  %1310 = bitcast <4 x i32> %1309 to <8 x half>, !dbg !52
  %address.local848 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !52
  %address.local849 = getelementptr i8, ptr addrspace(3) %address.local848, i64 5120, !dbg !52
  %1311 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local849, i32 0) #0, !dbg !52
  %1312 = bitcast <4 x i32> %1311 to <8 x half>, !dbg !52
  %address.local850 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !52
  %address.local851 = getelementptr i8, ptr addrspace(3) %address.local850, i64 6144, !dbg !52
  %1313 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #0, !dbg !52
  %1314 = bitcast <4 x i32> %1313 to <8 x half>, !dbg !52
  %address.local852 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !52
  %address.local853 = getelementptr i8, ptr addrspace(3) %address.local852, i64 7168, !dbg !52
  %1315 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local853, i32 0) #0, !dbg !52
  %1316 = bitcast <4 x i32> %1315 to <8 x half>, !dbg !52
  %1317 = shufflevector <8 x half> %1302, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1317, <4 x float> zeroinitializer) #9, !dbg !52
  %1319 = shufflevector <8 x half> %1304, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1319, <4 x float> %1318) #9, !dbg !52
  %1321 = shufflevector <8 x half> %1306, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1321, <4 x float> %1320) #9, !dbg !52
  %1323 = shufflevector <8 x half> %1308, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1323, <4 x float> %1322) #9, !dbg !52
  %1325 = shufflevector <8 x half> %1310, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1325, <4 x float> %1324) #9, !dbg !52
  %1327 = shufflevector <8 x half> %1312, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1327, <4 x float> %1326) #9, !dbg !52
  %1329 = shufflevector <8 x half> %1314, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1329, <4 x float> %1328) #9, !dbg !52
  %1331 = shufflevector <8 x half> %1316, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1331, <4 x float> %1330) #9, !dbg !52
  %1333 = extractelement <4 x float> %1332, i64 0, !dbg !52
  %1334 = extractelement <4 x float> %1332, i64 1, !dbg !52
  %1335 = extractelement <4 x float> %1332, i64 2, !dbg !52
  %1336 = extractelement <4 x float> %1332, i64 3, !dbg !52
  %1337 = select i1 %.not, float %1334, float %1333, !dbg !52
  %1338 = bitcast float %1337 to i32, !dbg !52
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1338), !dbg !52
  %1340 = bitcast i32 %1339 to float, !dbg !52
  %1341 = select i1 %.not, float %1333, float %1340, !dbg !52
  %1342 = select i1 %.not, float %1340, float %1334, !dbg !52
  %1343 = select i1 %.not, float %1336, float %1335, !dbg !52
  %1344 = bitcast float %1343 to i32, !dbg !52
  %1345 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1344), !dbg !52
  %1346 = bitcast i32 %1345 to float, !dbg !52
  %1347 = select i1 %.not, float %1335, float %1346, !dbg !52
  %1348 = select i1 %.not, float %1346, float %1336, !dbg !52
  %1349 = select i1 %.not1, float %1347, float %1341, !dbg !52
  %1350 = bitcast float %1349 to i32, !dbg !52
  %1351 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1350), !dbg !52
  %1352 = bitcast i32 %1351 to float, !dbg !52
  %1353 = select i1 %.not1, float %1341, float %1352, !dbg !52
  %1354 = select i1 %.not1, float %1352, float %1347, !dbg !52
  %1355 = select i1 %.not1, float %1348, float %1342, !dbg !52
  %1356 = bitcast float %1355 to i32, !dbg !52
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1356), !dbg !52
  %1358 = bitcast i32 %1357 to float, !dbg !52
  %1359 = select i1 %.not1, float %1342, float %1358, !dbg !52
  %1360 = select i1 %.not1, float %1358, float %1348, !dbg !52
  %1361 = shufflevector <8 x half> %1302, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1361, <4 x float> zeroinitializer) #9, !dbg !52
  %1363 = shufflevector <8 x half> %1304, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1363, <4 x float> %1362) #9, !dbg !52
  %1365 = shufflevector <8 x half> %1306, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1365, <4 x float> %1364) #9, !dbg !52
  %1367 = shufflevector <8 x half> %1308, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1367, <4 x float> %1366) #9, !dbg !52
  %1369 = shufflevector <8 x half> %1310, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1369, <4 x float> %1368) #9, !dbg !52
  %1371 = shufflevector <8 x half> %1312, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1371, <4 x float> %1370) #9, !dbg !52
  %1373 = shufflevector <8 x half> %1314, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1373, <4 x float> %1372) #9, !dbg !52
  %1375 = shufflevector <8 x half> %1316, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1375, <4 x float> %1374) #9, !dbg !52
  %1377 = extractelement <4 x float> %1376, i64 0, !dbg !52
  %1378 = extractelement <4 x float> %1376, i64 1, !dbg !52
  %1379 = extractelement <4 x float> %1376, i64 2, !dbg !52
  %1380 = extractelement <4 x float> %1376, i64 3, !dbg !52
  %1381 = select i1 %.not, float %1378, float %1377, !dbg !52
  %1382 = bitcast float %1381 to i32, !dbg !52
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1382), !dbg !52
  %1384 = bitcast i32 %1383 to float, !dbg !52
  %1385 = select i1 %.not, float %1377, float %1384, !dbg !52
  %1386 = select i1 %.not, float %1384, float %1378, !dbg !52
  %1387 = select i1 %.not, float %1380, float %1379, !dbg !52
  %1388 = bitcast float %1387 to i32, !dbg !52
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1388), !dbg !52
  %1390 = bitcast i32 %1389 to float, !dbg !52
  %1391 = select i1 %.not, float %1379, float %1390, !dbg !52
  %1392 = select i1 %.not, float %1390, float %1380, !dbg !52
  %1393 = select i1 %.not1, float %1391, float %1385, !dbg !52
  %1394 = bitcast float %1393 to i32, !dbg !52
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1394), !dbg !52
  %1396 = bitcast i32 %1395 to float, !dbg !52
  %1397 = select i1 %.not1, float %1385, float %1396, !dbg !52
  %1398 = select i1 %.not1, float %1396, float %1391, !dbg !52
  %1399 = select i1 %.not1, float %1392, float %1386, !dbg !52
  %1400 = bitcast float %1399 to i32, !dbg !52
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1400), !dbg !52
  %1402 = bitcast i32 %1401 to float, !dbg !52
  %1403 = select i1 %.not1, float %1386, float %1402, !dbg !52
  %1404 = select i1 %.not1, float %1402, float %1392, !dbg !52
  %1405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1317, <4 x float> zeroinitializer) #9, !dbg !52
  %1406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1319, <4 x float> %1405) #9, !dbg !52
  %1407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1321, <4 x float> %1406) #9, !dbg !52
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1323, <4 x float> %1407) #9, !dbg !52
  %1409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1325, <4 x float> %1408) #9, !dbg !52
  %1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1327, <4 x float> %1409) #9, !dbg !52
  %1411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1329, <4 x float> %1410) #9, !dbg !52
  %1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1331, <4 x float> %1411) #9, !dbg !52
  %1413 = extractelement <4 x float> %1412, i64 0, !dbg !52
  %1414 = extractelement <4 x float> %1412, i64 1, !dbg !52
  %1415 = extractelement <4 x float> %1412, i64 2, !dbg !52
  %1416 = extractelement <4 x float> %1412, i64 3, !dbg !52
  %1417 = select i1 %.not, float %1414, float %1413, !dbg !52
  %1418 = bitcast float %1417 to i32, !dbg !52
  %1419 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1418), !dbg !52
  %1420 = bitcast i32 %1419 to float, !dbg !52
  %1421 = select i1 %.not, float %1413, float %1420, !dbg !52
  %1422 = select i1 %.not, float %1420, float %1414, !dbg !52
  %1423 = select i1 %.not, float %1416, float %1415, !dbg !52
  %1424 = bitcast float %1423 to i32, !dbg !52
  %1425 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1424), !dbg !52
  %1426 = bitcast i32 %1425 to float, !dbg !52
  %1427 = select i1 %.not, float %1415, float %1426, !dbg !52
  %1428 = select i1 %.not, float %1426, float %1416, !dbg !52
  %1429 = select i1 %.not1, float %1427, float %1421, !dbg !52
  %1430 = bitcast float %1429 to i32, !dbg !52
  %1431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1430), !dbg !52
  %1432 = bitcast i32 %1431 to float, !dbg !52
  %1433 = select i1 %.not1, float %1421, float %1432, !dbg !52
  %1434 = select i1 %.not1, float %1432, float %1427, !dbg !52
  %1435 = select i1 %.not1, float %1428, float %1422, !dbg !52
  %1436 = bitcast float %1435 to i32, !dbg !52
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1436), !dbg !52
  %1438 = bitcast i32 %1437 to float, !dbg !52
  %1439 = select i1 %.not1, float %1422, float %1438, !dbg !52
  %1440 = select i1 %.not1, float %1438, float %1428, !dbg !52
  %1441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1361, <4 x float> zeroinitializer) #9, !dbg !52
  %1442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1363, <4 x float> %1441) #9, !dbg !52
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1365, <4 x float> %1442) #9, !dbg !52
  %1444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1367, <4 x float> %1443) #9, !dbg !52
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1369, <4 x float> %1444) #9, !dbg !52
  %1446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1371, <4 x float> %1445) #9, !dbg !52
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1373, <4 x float> %1446) #9, !dbg !52
  %1448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1375, <4 x float> %1447) #9, !dbg !52
  %1449 = extractelement <4 x float> %1448, i64 0, !dbg !52
  %1450 = extractelement <4 x float> %1448, i64 1, !dbg !52
  %1451 = extractelement <4 x float> %1448, i64 2, !dbg !52
  %1452 = extractelement <4 x float> %1448, i64 3, !dbg !52
  %1453 = select i1 %.not, float %1450, float %1449, !dbg !52
  %1454 = bitcast float %1453 to i32, !dbg !52
  %1455 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1454), !dbg !52
  %1456 = bitcast i32 %1455 to float, !dbg !52
  %1457 = select i1 %.not, float %1449, float %1456, !dbg !52
  %1458 = select i1 %.not, float %1456, float %1450, !dbg !52
  %1459 = select i1 %.not, float %1452, float %1451, !dbg !52
  %1460 = bitcast float %1459 to i32, !dbg !52
  %1461 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1460), !dbg !52
  %1462 = bitcast i32 %1461 to float, !dbg !52
  %1463 = select i1 %.not, float %1451, float %1462, !dbg !52
  %1464 = select i1 %.not, float %1462, float %1452, !dbg !52
  %1465 = select i1 %.not1, float %1463, float %1457, !dbg !52
  %1466 = bitcast float %1465 to i32, !dbg !52
  %1467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1466), !dbg !52
  %1468 = bitcast i32 %1467 to float, !dbg !52
  %1469 = select i1 %.not1, float %1457, float %1468, !dbg !52
  %1470 = select i1 %.not1, float %1468, float %1463, !dbg !52
  %1471 = select i1 %.not1, float %1464, float %1458, !dbg !52
  %1472 = bitcast float %1471 to i32, !dbg !52
  %1473 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1472), !dbg !52
  %1474 = bitcast i32 %1473 to float, !dbg !52
  %1475 = select i1 %.not1, float %1458, float %1474, !dbg !52
  %1476 = select i1 %.not1, float %1474, float %1464, !dbg !52
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  br i1 %43, label %1477, label %1478, !dbg !71

1477:                                             ; preds = %1045
  %address.thread854 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !71, !range !49, !frisk.register_lifetimes !43
  %address.local855 = lshr i32 %address.thread854, 1, !dbg !51
  %address.local856 = and i32 %address.local855, 96, !dbg !51
  %address.local857 = and i32 %address.thread854, 15, !dbg !51
  %address.local858 = or disjoint i32 %address.local856, %address.local857, !dbg !51
  %address.local859 = zext nneg i32 %address.local858 to i64, !dbg !51
  %address.local860 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !51
  store float %1042, ptr addrspace(3) %address.local860, align 4, !dbg !71
  %address.local861 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !51
  %address.local862 = getelementptr i8, ptr addrspace(3) %address.local861, i64 64, !dbg !51
  store float %1043, ptr addrspace(3) %address.local862, align 4, !dbg !71
  br label %1478, !dbg !71

1478:                                             ; preds = %1477, %1045
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  %address.thread863 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !71, !range !49, !frisk.register_lifetimes !43
  %1479 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !71
  %lds.const.gep1748 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !71
  %1480 = load float, ptr addrspace(3) %lds.const.gep1748, align 4, !dbg !71
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  %1481 = add nuw nsw i64 %76, %15, !dbg !67
  %1482 = shl i64 %1481, 19, !dbg !67
  %1483 = and i64 %1482, 16252928, !dbg !67
  %1484 = add i64 %1483, %82, !dbg !67
  %address.local877 = zext nneg i32 %address.thread863 to i64, !dbg !48
  %address.local878 = shl nuw nsw i64 %address.local877, 3, !dbg !47
  %address.local879 = and i64 %address.local878, 1536, !dbg !47
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local879, !dbg !47
  %address.local881 = ptrtoint ptr addrspace(3) %address.local880 to i64, !dbg !47
  %address.local882 = trunc i64 %address.local881 to i32, !dbg !47
  %1485 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local882), !dbg !67
  %address.local883 = shl nuw nsw i64 %address.local877, 6, !dbg !67
  %address.local884 = and i64 %address.local883, 16128, !dbg !67
  %address.local885 = shl nuw nsw i64 %address.local877, 4, !dbg !67
  %address.local886 = and i64 %address.local885, 48, !dbg !67
  %1486 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !67
  %.idx = shl i64 %1484, 1, !dbg !67
  %1487 = or disjoint i64 %address.local884, %address.local886, !dbg !67
  %1488 = or disjoint i64 %.idx, %1487, !dbg !67
  %1489 = trunc i64 %1488 to i32, !dbg !67
  %dma.buffer.offset1767 = or i32 %1489, 64, !dbg !67
  %1490 = insertelement <2 x i64> poison, i64 %1486, i64 0, !dbg !67
  %1491 = bitcast <2 x i64> %1490 to <4 x i32>, !dbg !67
  %dma.buffer.rsrc1768 = shufflevector <4 x i32> %1491, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !67
  %dma.buffer.lds1769 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1485), !dbg !67
  %1492 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1769, i32 %dma.buffer.offset1767, <4 x i32> %dma.buffer.rsrc1768) #10, !dbg !67
  %address.local889 = or disjoint i64 %address.local877, 256, !dbg !47
  %address.local890 = shl nuw nsw i64 %address.local889, 3, !dbg !47
  %address.local891 = and i64 %address.local890, 3584, !dbg !47
  %address.local892 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local891, !dbg !47
  %address.local893 = ptrtoint ptr addrspace(3) %address.local892 to i64, !dbg !47
  %address.local894 = trunc i64 %address.local893 to i32, !dbg !47
  %1493 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !67
  %address.local895 = shl nuw nsw i64 %address.local889, 6, !dbg !67
  %address.local896 = and i64 %address.local895, 32512, !dbg !67
  %address.local897 = shl nuw nsw i64 %address.local877, 4, !dbg !67
  %address.local898 = and i64 %address.local897, 48, !dbg !67
  %1494 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !67
  %.idx53 = shl i64 %1484, 1, !dbg !67
  %1495 = or disjoint i64 %.idx53, %address.local896, !dbg !67
  %1496 = or disjoint i64 %1495, %address.local898, !dbg !67
  %1497 = trunc i64 %1496 to i32, !dbg !67
  %dma.buffer.offset1764 = or i32 %1497, 64, !dbg !67
  %1498 = insertelement <2 x i64> poison, i64 %1494, i64 0, !dbg !67
  %1499 = bitcast <2 x i64> %1498 to <4 x i32>, !dbg !67
  %dma.buffer.rsrc1765 = shufflevector <4 x i32> %1499, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !67
  %dma.buffer.lds1766 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1493), !dbg !67
  %1500 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1766, i32 %dma.buffer.offset1764, <4 x i32> %dma.buffer.rsrc1765) #10, !dbg !67
  %1501 = extractvalue [2 x <8 x float>] %55, 0, !dbg !71
  %1502 = shufflevector <8 x float> %1501, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !71
  %1503 = insertelement <4 x float> poison, float %1479, i64 0, !dbg !71
  %1504 = shufflevector <4 x float> %1503, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !71
  %1505 = fmul <4 x float> %1502, %1504, !dbg !71
  %1506 = insertelement <4 x float> poison, float %1353, i64 0, !dbg !72
  %1507 = insertelement <4 x float> %1506, float %1359, i64 1, !dbg !72
  %1508 = insertelement <4 x float> %1507, float %1354, i64 2, !dbg !72
  %1509 = insertelement <4 x float> %1508, float %1360, i64 3, !dbg !72
  %1510 = fadd <4 x float> %1505, %1509, !dbg !72
  %1511 = shufflevector <8 x float> %1501, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !71
  %1512 = fmul <4 x float> %1511, %1504, !dbg !71
  %1513 = insertelement <4 x float> poison, float %1397, i64 0, !dbg !72
  %1514 = insertelement <4 x float> %1513, float %1403, i64 1, !dbg !72
  %1515 = insertelement <4 x float> %1514, float %1398, i64 2, !dbg !72
  %1516 = insertelement <4 x float> %1515, float %1404, i64 3, !dbg !72
  %1517 = fadd <4 x float> %1512, %1516, !dbg !72
  %1518 = extractvalue [2 x <8 x float>] %55, 1, !dbg !71
  %1519 = shufflevector <8 x float> %1518, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !71
  %1520 = insertelement <4 x float> poison, float %1480, i64 0, !dbg !71
  %1521 = shufflevector <4 x float> %1520, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !71
  %1522 = fmul <4 x float> %1519, %1521, !dbg !71
  %1523 = insertelement <4 x float> poison, float %1433, i64 0, !dbg !72
  %1524 = insertelement <4 x float> %1523, float %1439, i64 1, !dbg !72
  %1525 = insertelement <4 x float> %1524, float %1434, i64 2, !dbg !72
  %1526 = insertelement <4 x float> %1525, float %1440, i64 3, !dbg !72
  %1527 = fadd <4 x float> %1522, %1526, !dbg !72
  %1528 = shufflevector <8 x float> %1518, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !71
  %1529 = fmul <4 x float> %1528, %1521, !dbg !71
  %1530 = insertelement <4 x float> poison, float %1469, i64 0, !dbg !72
  %1531 = insertelement <4 x float> %1530, float %1475, i64 1, !dbg !72
  %1532 = insertelement <4 x float> %1531, float %1470, i64 2, !dbg !72
  %1533 = insertelement <4 x float> %1532, float %1476, i64 3, !dbg !72
  %1534 = fadd <4 x float> %1529, %1533, !dbg !72
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !67
  fence syncscope("workgroup") release, !dbg !67
  call void @llvm.amdgcn.s.barrier(), !dbg !67
  fence syncscope("workgroup") acquire, !dbg !67
  %address.local901 = shl nuw nsw i32 %address.thread863, 3, !dbg !52
  %address.local902 = and i32 %address.local901, 504, !dbg !52
  %address.local903 = zext nneg i32 %address.local902 to i64, !dbg !52
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !52
  %1535 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #0, !dbg !52
  %1536 = bitcast <4 x i32> %1535 to <8 x half>, !dbg !52
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !52
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 1024, !dbg !52
  %1537 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !52
  %1538 = bitcast <4 x i32> %1537 to <8 x half>, !dbg !52
  %address.local907 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !52
  %address.local908 = getelementptr i8, ptr addrspace(3) %address.local907, i64 2048, !dbg !52
  %1539 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local908, i32 0) #0, !dbg !52
  %1540 = bitcast <4 x i32> %1539 to <8 x half>, !dbg !52
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !52
  %address.local910 = getelementptr i8, ptr addrspace(3) %address.local909, i64 3072, !dbg !52
  %1541 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local910, i32 0) #0, !dbg !52
  %1542 = bitcast <4 x i32> %1541 to <8 x half>, !dbg !52
  %address.local911 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !52
  %address.local912 = getelementptr i8, ptr addrspace(3) %address.local911, i64 4096, !dbg !52
  %1543 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local912, i32 0) #0, !dbg !52
  %1544 = bitcast <4 x i32> %1543 to <8 x half>, !dbg !52
  %address.local913 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !52
  %address.local914 = getelementptr i8, ptr addrspace(3) %address.local913, i64 5120, !dbg !52
  %1545 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local914, i32 0) #0, !dbg !52
  %1546 = bitcast <4 x i32> %1545 to <8 x half>, !dbg !52
  %address.local915 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !52
  %address.local916 = getelementptr i8, ptr addrspace(3) %address.local915, i64 6144, !dbg !52
  %1547 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local916, i32 0) #0, !dbg !52
  %1548 = bitcast <4 x i32> %1547 to <8 x half>, !dbg !52
  %address.local917 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !52
  %address.local918 = getelementptr i8, ptr addrspace(3) %address.local917, i64 7168, !dbg !52
  %1549 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local918, i32 0) #0, !dbg !52
  %1550 = bitcast <4 x i32> %1549 to <8 x half>, !dbg !52
  %1551 = shufflevector <8 x half> %1536, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1551, <4 x float> zeroinitializer) #9, !dbg !52
  %1553 = shufflevector <8 x half> %1538, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1553, <4 x float> %1552) #9, !dbg !52
  %1555 = shufflevector <8 x half> %1540, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1555, <4 x float> %1554) #9, !dbg !52
  %1557 = shufflevector <8 x half> %1542, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1557, <4 x float> %1556) #9, !dbg !52
  %1559 = shufflevector <8 x half> %1544, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1559, <4 x float> %1558) #9, !dbg !52
  %1561 = shufflevector <8 x half> %1546, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1561, <4 x float> %1560) #9, !dbg !52
  %1563 = shufflevector <8 x half> %1548, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1563, <4 x float> %1562) #9, !dbg !52
  %1565 = shufflevector <8 x half> %1550, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1565, <4 x float> %1564) #9, !dbg !52
  %1567 = extractelement <4 x float> %1566, i64 0, !dbg !52
  %1568 = extractelement <4 x float> %1566, i64 1, !dbg !52
  %1569 = extractelement <4 x float> %1566, i64 2, !dbg !52
  %1570 = extractelement <4 x float> %1566, i64 3, !dbg !52
  %1571 = select i1 %.not, float %1568, float %1567, !dbg !52
  %1572 = bitcast float %1571 to i32, !dbg !52
  %1573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1572), !dbg !52
  %1574 = bitcast i32 %1573 to float, !dbg !52
  %1575 = select i1 %.not, float %1567, float %1574, !dbg !52
  %1576 = select i1 %.not, float %1574, float %1568, !dbg !52
  %1577 = select i1 %.not, float %1570, float %1569, !dbg !52
  %1578 = bitcast float %1577 to i32, !dbg !52
  %1579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1578), !dbg !52
  %1580 = bitcast i32 %1579 to float, !dbg !52
  %1581 = select i1 %.not, float %1569, float %1580, !dbg !52
  %1582 = select i1 %.not, float %1580, float %1570, !dbg !52
  %1583 = select i1 %.not1, float %1581, float %1575, !dbg !52
  %1584 = bitcast float %1583 to i32, !dbg !52
  %1585 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1584), !dbg !52
  %1586 = bitcast i32 %1585 to float, !dbg !52
  %1587 = select i1 %.not1, float %1575, float %1586, !dbg !52
  %1588 = select i1 %.not1, float %1586, float %1581, !dbg !52
  %1589 = select i1 %.not1, float %1582, float %1576, !dbg !52
  %1590 = bitcast float %1589 to i32, !dbg !52
  %1591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1590), !dbg !52
  %1592 = bitcast i32 %1591 to float, !dbg !52
  %1593 = select i1 %.not1, float %1576, float %1592, !dbg !52
  %1594 = select i1 %.not1, float %1592, float %1582, !dbg !52
  %1595 = shufflevector <8 x half> %1536, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1595, <4 x float> zeroinitializer) #9, !dbg !52
  %1597 = shufflevector <8 x half> %1538, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1597, <4 x float> %1596) #9, !dbg !52
  %1599 = shufflevector <8 x half> %1540, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1599, <4 x float> %1598) #9, !dbg !52
  %1601 = shufflevector <8 x half> %1542, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1601, <4 x float> %1600) #9, !dbg !52
  %1603 = shufflevector <8 x half> %1544, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1604 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1603, <4 x float> %1602) #9, !dbg !52
  %1605 = shufflevector <8 x half> %1546, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1606 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1605, <4 x float> %1604) #9, !dbg !52
  %1607 = shufflevector <8 x half> %1548, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1607, <4 x float> %1606) #9, !dbg !52
  %1609 = shufflevector <8 x half> %1550, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1609, <4 x float> %1608) #9, !dbg !52
  %1611 = extractelement <4 x float> %1610, i64 0, !dbg !52
  %1612 = extractelement <4 x float> %1610, i64 1, !dbg !52
  %1613 = extractelement <4 x float> %1610, i64 2, !dbg !52
  %1614 = extractelement <4 x float> %1610, i64 3, !dbg !52
  %1615 = select i1 %.not, float %1612, float %1611, !dbg !52
  %1616 = bitcast float %1615 to i32, !dbg !52
  %1617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1616), !dbg !52
  %1618 = bitcast i32 %1617 to float, !dbg !52
  %1619 = select i1 %.not, float %1611, float %1618, !dbg !52
  %1620 = select i1 %.not, float %1618, float %1612, !dbg !52
  %1621 = select i1 %.not, float %1614, float %1613, !dbg !52
  %1622 = bitcast float %1621 to i32, !dbg !52
  %1623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1622), !dbg !52
  %1624 = bitcast i32 %1623 to float, !dbg !52
  %1625 = select i1 %.not, float %1613, float %1624, !dbg !52
  %1626 = select i1 %.not, float %1624, float %1614, !dbg !52
  %1627 = select i1 %.not1, float %1625, float %1619, !dbg !52
  %1628 = bitcast float %1627 to i32, !dbg !52
  %1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1628), !dbg !52
  %1630 = bitcast i32 %1629 to float, !dbg !52
  %1631 = select i1 %.not1, float %1619, float %1630, !dbg !52
  %1632 = select i1 %.not1, float %1630, float %1625, !dbg !52
  %1633 = select i1 %.not1, float %1626, float %1620, !dbg !52
  %1634 = bitcast float %1633 to i32, !dbg !52
  %1635 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1634), !dbg !52
  %1636 = bitcast i32 %1635 to float, !dbg !52
  %1637 = select i1 %.not1, float %1620, float %1636, !dbg !52
  %1638 = select i1 %.not1, float %1636, float %1626, !dbg !52
  %1639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1551, <4 x float> zeroinitializer) #9, !dbg !52
  %1640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1553, <4 x float> %1639) #9, !dbg !52
  %1641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1555, <4 x float> %1640) #9, !dbg !52
  %1642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1557, <4 x float> %1641) #9, !dbg !52
  %1643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1559, <4 x float> %1642) #9, !dbg !52
  %1644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1561, <4 x float> %1643) #9, !dbg !52
  %1645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1563, <4 x float> %1644) #9, !dbg !52
  %1646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1565, <4 x float> %1645) #9, !dbg !52
  %1647 = extractelement <4 x float> %1646, i64 0, !dbg !52
  %1648 = extractelement <4 x float> %1646, i64 1, !dbg !52
  %1649 = extractelement <4 x float> %1646, i64 2, !dbg !52
  %1650 = extractelement <4 x float> %1646, i64 3, !dbg !52
  %1651 = select i1 %.not, float %1648, float %1647, !dbg !52
  %1652 = bitcast float %1651 to i32, !dbg !52
  %1653 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1652), !dbg !52
  %1654 = bitcast i32 %1653 to float, !dbg !52
  %1655 = select i1 %.not, float %1647, float %1654, !dbg !52
  %1656 = select i1 %.not, float %1654, float %1648, !dbg !52
  %1657 = select i1 %.not, float %1650, float %1649, !dbg !52
  %1658 = bitcast float %1657 to i32, !dbg !52
  %1659 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1658), !dbg !52
  %1660 = bitcast i32 %1659 to float, !dbg !52
  %1661 = select i1 %.not, float %1649, float %1660, !dbg !52
  %1662 = select i1 %.not, float %1660, float %1650, !dbg !52
  %1663 = select i1 %.not1, float %1661, float %1655, !dbg !52
  %1664 = bitcast float %1663 to i32, !dbg !52
  %1665 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1664), !dbg !52
  %1666 = bitcast i32 %1665 to float, !dbg !52
  %1667 = select i1 %.not1, float %1655, float %1666, !dbg !52
  %1668 = select i1 %.not1, float %1666, float %1661, !dbg !52
  %1669 = select i1 %.not1, float %1662, float %1656, !dbg !52
  %1670 = bitcast float %1669 to i32, !dbg !52
  %1671 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1670), !dbg !52
  %1672 = bitcast i32 %1671 to float, !dbg !52
  %1673 = select i1 %.not1, float %1656, float %1672, !dbg !52
  %1674 = select i1 %.not1, float %1672, float %1662, !dbg !52
  %1675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1595, <4 x float> zeroinitializer) #9, !dbg !52
  %1676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1597, <4 x float> %1675) #9, !dbg !52
  %1677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1599, <4 x float> %1676) #9, !dbg !52
  %1678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1601, <4 x float> %1677) #9, !dbg !52
  %1679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1603, <4 x float> %1678) #9, !dbg !52
  %1680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1605, <4 x float> %1679) #9, !dbg !52
  %1681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1607, <4 x float> %1680) #9, !dbg !52
  %1682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1609, <4 x float> %1681) #9, !dbg !52
  %1683 = extractelement <4 x float> %1682, i64 0, !dbg !52
  %1684 = extractelement <4 x float> %1682, i64 1, !dbg !52
  %1685 = extractelement <4 x float> %1682, i64 2, !dbg !52
  %1686 = extractelement <4 x float> %1682, i64 3, !dbg !52
  %1687 = select i1 %.not, float %1684, float %1683, !dbg !52
  %1688 = bitcast float %1687 to i32, !dbg !52
  %1689 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1688), !dbg !52
  %1690 = bitcast i32 %1689 to float, !dbg !52
  %1691 = select i1 %.not, float %1683, float %1690, !dbg !52
  %1692 = select i1 %.not, float %1690, float %1684, !dbg !52
  %1693 = select i1 %.not, float %1686, float %1685, !dbg !52
  %1694 = bitcast float %1693 to i32, !dbg !52
  %1695 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1694), !dbg !52
  %1696 = bitcast i32 %1695 to float, !dbg !52
  %1697 = select i1 %.not, float %1685, float %1696, !dbg !52
  %1698 = select i1 %.not, float %1696, float %1686, !dbg !52
  %1699 = select i1 %.not1, float %1697, float %1691, !dbg !52
  %1700 = bitcast float %1699 to i32, !dbg !52
  %1701 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1700), !dbg !52
  %1702 = bitcast i32 %1701 to float, !dbg !52
  %1703 = select i1 %.not1, float %1691, float %1702, !dbg !52
  %1704 = select i1 %.not1, float %1702, float %1697, !dbg !52
  %1705 = select i1 %.not1, float %1698, float %1692, !dbg !52
  %1706 = bitcast float %1705 to i32, !dbg !52
  %1707 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1706), !dbg !52
  %1708 = bitcast i32 %1707 to float, !dbg !52
  %1709 = select i1 %.not1, float %1692, float %1708, !dbg !52
  %1710 = select i1 %.not1, float %1708, float %1698, !dbg !52
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  br i1 %43, label %1711, label %1712, !dbg !71

1711:                                             ; preds = %1478
  %address.thread919 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !71, !range !49, !frisk.register_lifetimes !43
  %address.local920 = lshr i32 %address.thread919, 1, !dbg !51
  %address.local921 = and i32 %address.local920, 96, !dbg !51
  %address.local922 = and i32 %address.thread919, 15, !dbg !51
  %address.local923 = or disjoint i32 %address.local921, %address.local922, !dbg !51
  %address.local924 = zext nneg i32 %address.local923 to i64, !dbg !51
  %address.local925 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !51
  store float %1042, ptr addrspace(3) %address.local925, align 4, !dbg !71
  %address.local926 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !51
  %address.local927 = getelementptr i8, ptr addrspace(3) %address.local926, i64 64, !dbg !51
  store float %1043, ptr addrspace(3) %address.local927, align 4, !dbg !71
  br label %1712, !dbg !71

1712:                                             ; preds = %1711, %1478
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  %address.thread928 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !71, !range !49, !frisk.register_lifetimes !43
  %1713 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !71
  %lds.const.gep1750 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !71
  %1714 = load float, ptr addrspace(3) %lds.const.gep1750, align 4, !dbg !71
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  %address.local942 = zext nneg i32 %address.thread928 to i64, !dbg !48
  %address.local943 = shl nuw nsw i64 %address.local942, 3, !dbg !47
  %address.local944 = and i64 %address.local943, 1536, !dbg !47
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local944, !dbg !47
  %address.local946 = ptrtoint ptr addrspace(3) %address.local945 to i64, !dbg !47
  %address.local947 = trunc i64 %address.local946 to i32, !dbg !47
  %1715 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local947), !dbg !67
  %address.local948 = shl nuw nsw i64 %address.local942, 6, !dbg !67
  %address.local949 = and i64 %address.local948, 16128, !dbg !67
  %address.local950 = shl nuw nsw i64 %address.local942, 4, !dbg !67
  %address.local951 = and i64 %address.local950, 48, !dbg !67
  %1716 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !67
  %.idx55 = shl i64 %1484, 1, !dbg !67
  %1717 = or disjoint i64 %address.local949, %address.local951, !dbg !67
  %1718 = or disjoint i64 %.idx55, %1717, !dbg !67
  %1719 = trunc i64 %1718 to i32, !dbg !67
  %dma.buffer.offset1761 = or i32 %1719, 128, !dbg !67
  %1720 = insertelement <2 x i64> poison, i64 %1716, i64 0, !dbg !67
  %1721 = bitcast <2 x i64> %1720 to <4 x i32>, !dbg !67
  %dma.buffer.rsrc1762 = shufflevector <4 x i32> %1721, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !67
  %dma.buffer.lds1763 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1715), !dbg !67
  %1722 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1763, i32 %dma.buffer.offset1761, <4 x i32> %dma.buffer.rsrc1762) #10, !dbg !67
  %address.local954 = or disjoint i64 %address.local942, 256, !dbg !47
  %address.local955 = shl nuw nsw i64 %address.local954, 3, !dbg !47
  %address.local956 = and i64 %address.local955, 3584, !dbg !47
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local956, !dbg !47
  %address.local958 = ptrtoint ptr addrspace(3) %address.local957 to i64, !dbg !47
  %address.local959 = trunc i64 %address.local958 to i32, !dbg !47
  %1723 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local959), !dbg !67
  %address.local960 = shl nuw nsw i64 %address.local954, 6, !dbg !67
  %address.local961 = and i64 %address.local960, 32512, !dbg !67
  %address.local962 = shl nuw nsw i64 %address.local942, 4, !dbg !67
  %address.local963 = and i64 %address.local962, 48, !dbg !67
  %1724 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !67
  %.idx57 = shl i64 %1484, 1, !dbg !67
  %1725 = or disjoint i64 %.idx57, %address.local961, !dbg !67
  %1726 = or disjoint i64 %1725, %address.local963, !dbg !67
  %1727 = trunc i64 %1726 to i32, !dbg !67
  %dma.buffer.offset1758 = or i32 %1727, 128, !dbg !67
  %1728 = insertelement <2 x i64> poison, i64 %1724, i64 0, !dbg !67
  %1729 = bitcast <2 x i64> %1728 to <4 x i32>, !dbg !67
  %dma.buffer.rsrc1759 = shufflevector <4 x i32> %1729, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !67
  %dma.buffer.lds1760 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1723), !dbg !67
  %1730 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1760, i32 %dma.buffer.offset1758, <4 x i32> %dma.buffer.rsrc1759) #10, !dbg !67
  %1731 = extractvalue [2 x <8 x float>] %58, 0, !dbg !71
  %1732 = shufflevector <8 x float> %1731, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !71
  %1733 = insertelement <4 x float> poison, float %1713, i64 0, !dbg !71
  %1734 = shufflevector <4 x float> %1733, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !71
  %1735 = fmul <4 x float> %1732, %1734, !dbg !71
  %1736 = insertelement <4 x float> poison, float %1587, i64 0, !dbg !72
  %1737 = insertelement <4 x float> %1736, float %1593, i64 1, !dbg !72
  %1738 = insertelement <4 x float> %1737, float %1588, i64 2, !dbg !72
  %1739 = insertelement <4 x float> %1738, float %1594, i64 3, !dbg !72
  %1740 = fadd <4 x float> %1735, %1739, !dbg !72
  %1741 = shufflevector <8 x float> %1731, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !71
  %1742 = fmul <4 x float> %1741, %1734, !dbg !71
  %1743 = insertelement <4 x float> poison, float %1631, i64 0, !dbg !72
  %1744 = insertelement <4 x float> %1743, float %1637, i64 1, !dbg !72
  %1745 = insertelement <4 x float> %1744, float %1632, i64 2, !dbg !72
  %1746 = insertelement <4 x float> %1745, float %1638, i64 3, !dbg !72
  %1747 = fadd <4 x float> %1742, %1746, !dbg !72
  %1748 = extractvalue [2 x <8 x float>] %58, 1, !dbg !71
  %1749 = shufflevector <8 x float> %1748, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !71
  %1750 = insertelement <4 x float> poison, float %1714, i64 0, !dbg !71
  %1751 = shufflevector <4 x float> %1750, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !71
  %1752 = fmul <4 x float> %1749, %1751, !dbg !71
  %1753 = insertelement <4 x float> poison, float %1667, i64 0, !dbg !72
  %1754 = insertelement <4 x float> %1753, float %1673, i64 1, !dbg !72
  %1755 = insertelement <4 x float> %1754, float %1668, i64 2, !dbg !72
  %1756 = insertelement <4 x float> %1755, float %1674, i64 3, !dbg !72
  %1757 = fadd <4 x float> %1752, %1756, !dbg !72
  %1758 = shufflevector <8 x float> %1748, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !71
  %1759 = fmul <4 x float> %1758, %1751, !dbg !71
  %1760 = insertelement <4 x float> poison, float %1703, i64 0, !dbg !72
  %1761 = insertelement <4 x float> %1760, float %1709, i64 1, !dbg !72
  %1762 = insertelement <4 x float> %1761, float %1704, i64 2, !dbg !72
  %1763 = insertelement <4 x float> %1762, float %1710, i64 3, !dbg !72
  %1764 = fadd <4 x float> %1759, %1763, !dbg !72
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !67
  fence syncscope("workgroup") release, !dbg !67
  call void @llvm.amdgcn.s.barrier(), !dbg !67
  fence syncscope("workgroup") acquire, !dbg !67
  %address.local966 = shl nuw nsw i32 %address.thread928, 3, !dbg !52
  %address.local967 = and i32 %address.local966, 504, !dbg !52
  %address.local968 = zext nneg i32 %address.local967 to i64, !dbg !52
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !52
  %1765 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #0, !dbg !52
  %1766 = bitcast <4 x i32> %1765 to <8 x half>, !dbg !52
  %address.local970 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !52
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local970, i64 1024, !dbg !52
  %1767 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #0, !dbg !52
  %1768 = bitcast <4 x i32> %1767 to <8 x half>, !dbg !52
  %address.local972 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !52
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local972, i64 2048, !dbg !52
  %1769 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !52
  %1770 = bitcast <4 x i32> %1769 to <8 x half>, !dbg !52
  %address.local974 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !52
  %address.local975 = getelementptr i8, ptr addrspace(3) %address.local974, i64 3072, !dbg !52
  %1771 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #0, !dbg !52
  %1772 = bitcast <4 x i32> %1771 to <8 x half>, !dbg !52
  %address.local976 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !52
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local976, i64 4096, !dbg !52
  %1773 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !52
  %1774 = bitcast <4 x i32> %1773 to <8 x half>, !dbg !52
  %address.local978 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !52
  %address.local979 = getelementptr i8, ptr addrspace(3) %address.local978, i64 5120, !dbg !52
  %1775 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #0, !dbg !52
  %1776 = bitcast <4 x i32> %1775 to <8 x half>, !dbg !52
  %address.local980 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !52
  %address.local981 = getelementptr i8, ptr addrspace(3) %address.local980, i64 6144, !dbg !52
  %1777 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #0, !dbg !52
  %1778 = bitcast <4 x i32> %1777 to <8 x half>, !dbg !52
  %address.local982 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !52
  %address.local983 = getelementptr i8, ptr addrspace(3) %address.local982, i64 7168, !dbg !52
  %1779 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #0, !dbg !52
  %1780 = bitcast <4 x i32> %1779 to <8 x half>, !dbg !52
  %1781 = shufflevector <8 x half> %1766, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1781, <4 x float> zeroinitializer) #9, !dbg !52
  %1783 = shufflevector <8 x half> %1768, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1784 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1783, <4 x float> %1782) #9, !dbg !52
  %1785 = shufflevector <8 x half> %1770, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1786 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1785, <4 x float> %1784) #9, !dbg !52
  %1787 = shufflevector <8 x half> %1772, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1788 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1787, <4 x float> %1786) #9, !dbg !52
  %1789 = shufflevector <8 x half> %1774, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1790 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1789, <4 x float> %1788) #9, !dbg !52
  %1791 = shufflevector <8 x half> %1776, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1792 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1791, <4 x float> %1790) #9, !dbg !52
  %1793 = shufflevector <8 x half> %1778, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1793, <4 x float> %1792) #9, !dbg !52
  %1795 = shufflevector <8 x half> %1780, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %1796 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1795, <4 x float> %1794) #9, !dbg !52
  %1797 = extractelement <4 x float> %1796, i64 0, !dbg !52
  %1798 = extractelement <4 x float> %1796, i64 1, !dbg !52
  %1799 = extractelement <4 x float> %1796, i64 2, !dbg !52
  %1800 = extractelement <4 x float> %1796, i64 3, !dbg !52
  %1801 = select i1 %.not, float %1798, float %1797, !dbg !52
  %1802 = bitcast float %1801 to i32, !dbg !52
  %1803 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1802), !dbg !52
  %1804 = bitcast i32 %1803 to float, !dbg !52
  %1805 = select i1 %.not, float %1797, float %1804, !dbg !52
  %1806 = select i1 %.not, float %1804, float %1798, !dbg !52
  %1807 = select i1 %.not, float %1800, float %1799, !dbg !52
  %1808 = bitcast float %1807 to i32, !dbg !52
  %1809 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1808), !dbg !52
  %1810 = bitcast i32 %1809 to float, !dbg !52
  %1811 = select i1 %.not, float %1799, float %1810, !dbg !52
  %1812 = select i1 %.not, float %1810, float %1800, !dbg !52
  %1813 = select i1 %.not1, float %1811, float %1805, !dbg !52
  %1814 = bitcast float %1813 to i32, !dbg !52
  %1815 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1814), !dbg !52
  %1816 = bitcast i32 %1815 to float, !dbg !52
  %1817 = select i1 %.not1, float %1805, float %1816, !dbg !52
  %1818 = select i1 %.not1, float %1816, float %1811, !dbg !52
  %1819 = select i1 %.not1, float %1812, float %1806, !dbg !52
  %1820 = bitcast float %1819 to i32, !dbg !52
  %1821 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1820), !dbg !52
  %1822 = bitcast i32 %1821 to float, !dbg !52
  %1823 = select i1 %.not1, float %1806, float %1822, !dbg !52
  %1824 = select i1 %.not1, float %1822, float %1812, !dbg !52
  %1825 = shufflevector <8 x half> %1766, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1825, <4 x float> zeroinitializer) #9, !dbg !52
  %1827 = shufflevector <8 x half> %1768, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1827, <4 x float> %1826) #9, !dbg !52
  %1829 = shufflevector <8 x half> %1770, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1830 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1829, <4 x float> %1828) #9, !dbg !52
  %1831 = shufflevector <8 x half> %1772, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1832 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1831, <4 x float> %1830) #9, !dbg !52
  %1833 = shufflevector <8 x half> %1774, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1833, <4 x float> %1832) #9, !dbg !52
  %1835 = shufflevector <8 x half> %1776, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1835, <4 x float> %1834) #9, !dbg !52
  %1837 = shufflevector <8 x half> %1778, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1837, <4 x float> %1836) #9, !dbg !52
  %1839 = shufflevector <8 x half> %1780, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %1840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1839, <4 x float> %1838) #9, !dbg !52
  %1841 = extractelement <4 x float> %1840, i64 0, !dbg !52
  %1842 = extractelement <4 x float> %1840, i64 1, !dbg !52
  %1843 = extractelement <4 x float> %1840, i64 2, !dbg !52
  %1844 = extractelement <4 x float> %1840, i64 3, !dbg !52
  %1845 = select i1 %.not, float %1842, float %1841, !dbg !52
  %1846 = bitcast float %1845 to i32, !dbg !52
  %1847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1846), !dbg !52
  %1848 = bitcast i32 %1847 to float, !dbg !52
  %1849 = select i1 %.not, float %1841, float %1848, !dbg !52
  %1850 = select i1 %.not, float %1848, float %1842, !dbg !52
  %1851 = select i1 %.not, float %1844, float %1843, !dbg !52
  %1852 = bitcast float %1851 to i32, !dbg !52
  %1853 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1852), !dbg !52
  %1854 = bitcast i32 %1853 to float, !dbg !52
  %1855 = select i1 %.not, float %1843, float %1854, !dbg !52
  %1856 = select i1 %.not, float %1854, float %1844, !dbg !52
  %1857 = select i1 %.not1, float %1855, float %1849, !dbg !52
  %1858 = bitcast float %1857 to i32, !dbg !52
  %1859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1858), !dbg !52
  %1860 = bitcast i32 %1859 to float, !dbg !52
  %1861 = select i1 %.not1, float %1849, float %1860, !dbg !52
  %1862 = select i1 %.not1, float %1860, float %1855, !dbg !52
  %1863 = select i1 %.not1, float %1856, float %1850, !dbg !52
  %1864 = bitcast float %1863 to i32, !dbg !52
  %1865 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1864), !dbg !52
  %1866 = bitcast i32 %1865 to float, !dbg !52
  %1867 = select i1 %.not1, float %1850, float %1866, !dbg !52
  %1868 = select i1 %.not1, float %1866, float %1856, !dbg !52
  %1869 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1781, <4 x float> zeroinitializer) #9, !dbg !52
  %1870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1783, <4 x float> %1869) #9, !dbg !52
  %1871 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1785, <4 x float> %1870) #9, !dbg !52
  %1872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1787, <4 x float> %1871) #9, !dbg !52
  %1873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1789, <4 x float> %1872) #9, !dbg !52
  %1874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1791, <4 x float> %1873) #9, !dbg !52
  %1875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1793, <4 x float> %1874) #9, !dbg !52
  %1876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1795, <4 x float> %1875) #9, !dbg !52
  %1877 = extractelement <4 x float> %1876, i64 0, !dbg !52
  %1878 = extractelement <4 x float> %1876, i64 1, !dbg !52
  %1879 = extractelement <4 x float> %1876, i64 2, !dbg !52
  %1880 = extractelement <4 x float> %1876, i64 3, !dbg !52
  %1881 = select i1 %.not, float %1878, float %1877, !dbg !52
  %1882 = bitcast float %1881 to i32, !dbg !52
  %1883 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1882), !dbg !52
  %1884 = bitcast i32 %1883 to float, !dbg !52
  %1885 = select i1 %.not, float %1877, float %1884, !dbg !52
  %1886 = select i1 %.not, float %1884, float %1878, !dbg !52
  %1887 = select i1 %.not, float %1880, float %1879, !dbg !52
  %1888 = bitcast float %1887 to i32, !dbg !52
  %1889 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1888), !dbg !52
  %1890 = bitcast i32 %1889 to float, !dbg !52
  %1891 = select i1 %.not, float %1879, float %1890, !dbg !52
  %1892 = select i1 %.not, float %1890, float %1880, !dbg !52
  %1893 = select i1 %.not1, float %1891, float %1885, !dbg !52
  %1894 = bitcast float %1893 to i32, !dbg !52
  %1895 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1894), !dbg !52
  %1896 = bitcast i32 %1895 to float, !dbg !52
  %1897 = select i1 %.not1, float %1885, float %1896, !dbg !52
  %1898 = select i1 %.not1, float %1896, float %1891, !dbg !52
  %1899 = select i1 %.not1, float %1892, float %1886, !dbg !52
  %1900 = bitcast float %1899 to i32, !dbg !52
  %1901 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1900), !dbg !52
  %1902 = bitcast i32 %1901 to float, !dbg !52
  %1903 = select i1 %.not1, float %1886, float %1902, !dbg !52
  %1904 = select i1 %.not1, float %1902, float %1892, !dbg !52
  %1905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1825, <4 x float> zeroinitializer) #9, !dbg !52
  %1906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1827, <4 x float> %1905) #9, !dbg !52
  %1907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1829, <4 x float> %1906) #9, !dbg !52
  %1908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1831, <4 x float> %1907) #9, !dbg !52
  %1909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1833, <4 x float> %1908) #9, !dbg !52
  %1910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1835, <4 x float> %1909) #9, !dbg !52
  %1911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1837, <4 x float> %1910) #9, !dbg !52
  %1912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1839, <4 x float> %1911) #9, !dbg !52
  %1913 = extractelement <4 x float> %1912, i64 0, !dbg !52
  %1914 = extractelement <4 x float> %1912, i64 1, !dbg !52
  %1915 = extractelement <4 x float> %1912, i64 2, !dbg !52
  %1916 = extractelement <4 x float> %1912, i64 3, !dbg !52
  %1917 = select i1 %.not, float %1914, float %1913, !dbg !52
  %1918 = bitcast float %1917 to i32, !dbg !52
  %1919 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1918), !dbg !52
  %1920 = bitcast i32 %1919 to float, !dbg !52
  %1921 = select i1 %.not, float %1913, float %1920, !dbg !52
  %1922 = select i1 %.not, float %1920, float %1914, !dbg !52
  %1923 = select i1 %.not, float %1916, float %1915, !dbg !52
  %1924 = bitcast float %1923 to i32, !dbg !52
  %1925 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %1924), !dbg !52
  %1926 = bitcast i32 %1925 to float, !dbg !52
  %1927 = select i1 %.not, float %1915, float %1926, !dbg !52
  %1928 = select i1 %.not, float %1926, float %1916, !dbg !52
  %1929 = select i1 %.not1, float %1927, float %1921, !dbg !52
  %1930 = bitcast float %1929 to i32, !dbg !52
  %1931 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1930), !dbg !52
  %1932 = bitcast i32 %1931 to float, !dbg !52
  %1933 = select i1 %.not1, float %1921, float %1932, !dbg !52
  %1934 = select i1 %.not1, float %1932, float %1927, !dbg !52
  %1935 = select i1 %.not1, float %1928, float %1922, !dbg !52
  %1936 = bitcast float %1935 to i32, !dbg !52
  %1937 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %1936), !dbg !52
  %1938 = bitcast i32 %1937 to float, !dbg !52
  %1939 = select i1 %.not1, float %1922, float %1938, !dbg !52
  %1940 = select i1 %.not1, float %1938, float %1928, !dbg !52
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  br i1 %43, label %1941, label %1942, !dbg !71

1941:                                             ; preds = %1712
  %address.thread984 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !71, !range !49, !frisk.register_lifetimes !43
  %address.local985 = lshr i32 %address.thread984, 1, !dbg !51
  %address.local986 = and i32 %address.local985, 96, !dbg !51
  %address.local987 = and i32 %address.thread984, 15, !dbg !51
  %address.local988 = or disjoint i32 %address.local986, %address.local987, !dbg !51
  %address.local989 = zext nneg i32 %address.local988 to i64, !dbg !51
  %address.local990 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !51
  store float %1042, ptr addrspace(3) %address.local990, align 4, !dbg !71
  %address.local991 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !51
  %address.local992 = getelementptr i8, ptr addrspace(3) %address.local991, i64 64, !dbg !51
  store float %1043, ptr addrspace(3) %address.local992, align 4, !dbg !71
  br label %1942, !dbg !71

1942:                                             ; preds = %1941, %1712
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  %address.thread993 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !71, !range !49, !frisk.register_lifetimes !43
  %1943 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !71
  %lds.const.gep1752 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !71
  %1944 = load float, ptr addrspace(3) %lds.const.gep1752, align 4, !dbg !71
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  %address.local1007 = zext nneg i32 %address.thread993 to i64, !dbg !48
  %address.local1008 = shl nuw nsw i64 %address.local1007, 3, !dbg !47
  %address.local1009 = and i64 %address.local1008, 1536, !dbg !47
  %address.local1010 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1009, !dbg !47
  %address.local1011 = ptrtoint ptr addrspace(3) %address.local1010 to i64, !dbg !47
  %address.local1012 = trunc i64 %address.local1011 to i32, !dbg !47
  %1945 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1012), !dbg !67
  %address.local1013 = shl nuw nsw i64 %address.local1007, 6, !dbg !67
  %address.local1014 = and i64 %address.local1013, 16128, !dbg !67
  %address.local1015 = shl nuw nsw i64 %address.local1007, 4, !dbg !67
  %address.local1016 = and i64 %address.local1015, 48, !dbg !67
  %1946 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !67
  %.idx59 = shl i64 %1484, 1, !dbg !67
  %1947 = or disjoint i64 %address.local1014, %address.local1016, !dbg !67
  %1948 = or disjoint i64 %.idx59, %1947, !dbg !67
  %1949 = trunc i64 %1948 to i32, !dbg !67
  %dma.buffer.offset1755 = or i32 %1949, 192, !dbg !67
  %1950 = insertelement <2 x i64> poison, i64 %1946, i64 0, !dbg !67
  %1951 = bitcast <2 x i64> %1950 to <4 x i32>, !dbg !67
  %dma.buffer.rsrc1756 = shufflevector <4 x i32> %1951, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !67
  %dma.buffer.lds1757 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1945), !dbg !67
  %1952 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1757, i32 %dma.buffer.offset1755, <4 x i32> %dma.buffer.rsrc1756) #10, !dbg !67
  %address.local1019 = or disjoint i64 %address.local1007, 256, !dbg !47
  %address.local1020 = shl nuw nsw i64 %address.local1019, 3, !dbg !47
  %address.local1021 = and i64 %address.local1020, 3584, !dbg !47
  %address.local1022 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1021, !dbg !47
  %address.local1023 = ptrtoint ptr addrspace(3) %address.local1022 to i64, !dbg !47
  %address.local1024 = trunc i64 %address.local1023 to i32, !dbg !47
  %1953 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1024), !dbg !67
  %address.local1025 = shl nuw nsw i64 %address.local1019, 6, !dbg !67
  %address.local1026 = and i64 %address.local1025, 32512, !dbg !67
  %address.local1027 = shl nuw nsw i64 %address.local1007, 4, !dbg !67
  %address.local1028 = and i64 %address.local1027, 48, !dbg !67
  %1954 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !67
  %.idx61 = shl i64 %1484, 1, !dbg !67
  %1955 = or disjoint i64 %.idx61, %address.local1026, !dbg !67
  %1956 = or disjoint i64 %1955, %address.local1028, !dbg !67
  %1957 = trunc i64 %1956 to i32, !dbg !67
  %dma.buffer.offset = or i32 %1957, 192, !dbg !67
  %1958 = insertelement <2 x i64> poison, i64 %1954, i64 0, !dbg !67
  %1959 = bitcast <2 x i64> %1958 to <4 x i32>, !dbg !67
  %dma.buffer.rsrc = shufflevector <4 x i32> %1959, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !67
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1953), !dbg !67
  %1960 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !67
  %1961 = extractvalue [2 x <8 x float>] %59, 0, !dbg !71
  %1962 = shufflevector <8 x float> %1961, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !71
  %1963 = insertelement <4 x float> poison, float %1943, i64 0, !dbg !71
  %1964 = shufflevector <4 x float> %1963, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !71
  %1965 = fmul <4 x float> %1962, %1964, !dbg !71
  %1966 = insertelement <4 x float> poison, float %1817, i64 0, !dbg !72
  %1967 = insertelement <4 x float> %1966, float %1823, i64 1, !dbg !72
  %1968 = insertelement <4 x float> %1967, float %1818, i64 2, !dbg !72
  %1969 = insertelement <4 x float> %1968, float %1824, i64 3, !dbg !72
  %1970 = fadd <4 x float> %1965, %1969, !dbg !72
  %1971 = shufflevector <8 x float> %1961, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !71
  %1972 = fmul <4 x float> %1971, %1964, !dbg !71
  %1973 = insertelement <4 x float> poison, float %1861, i64 0, !dbg !72
  %1974 = insertelement <4 x float> %1973, float %1867, i64 1, !dbg !72
  %1975 = insertelement <4 x float> %1974, float %1862, i64 2, !dbg !72
  %1976 = insertelement <4 x float> %1975, float %1868, i64 3, !dbg !72
  %1977 = fadd <4 x float> %1972, %1976, !dbg !72
  %1978 = extractvalue [2 x <8 x float>] %59, 1, !dbg !71
  %1979 = shufflevector <8 x float> %1978, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !71
  %1980 = insertelement <4 x float> poison, float %1944, i64 0, !dbg !71
  %1981 = shufflevector <4 x float> %1980, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !71
  %1982 = fmul <4 x float> %1979, %1981, !dbg !71
  %1983 = insertelement <4 x float> poison, float %1897, i64 0, !dbg !72
  %1984 = insertelement <4 x float> %1983, float %1903, i64 1, !dbg !72
  %1985 = insertelement <4 x float> %1984, float %1898, i64 2, !dbg !72
  %1986 = insertelement <4 x float> %1985, float %1904, i64 3, !dbg !72
  %1987 = fadd <4 x float> %1982, %1986, !dbg !72
  %1988 = shufflevector <8 x float> %1978, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !71
  %1989 = fmul <4 x float> %1988, %1981, !dbg !71
  %1990 = insertelement <4 x float> poison, float %1933, i64 0, !dbg !72
  %1991 = insertelement <4 x float> %1990, float %1939, i64 1, !dbg !72
  %1992 = insertelement <4 x float> %1991, float %1934, i64 2, !dbg !72
  %1993 = insertelement <4 x float> %1992, float %1940, i64 3, !dbg !72
  %1994 = fadd <4 x float> %1989, %1993, !dbg !72
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !67
  fence syncscope("workgroup") release, !dbg !67
  call void @llvm.amdgcn.s.barrier(), !dbg !67
  fence syncscope("workgroup") acquire, !dbg !67
  %address.local1031 = shl nuw nsw i32 %address.thread993, 3, !dbg !52
  %address.local1032 = and i32 %address.local1031, 504, !dbg !52
  %address.local1033 = zext nneg i32 %address.local1032 to i64, !dbg !52
  %address.local1034 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !52
  %1995 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !52
  %1996 = bitcast <4 x i32> %1995 to <8 x half>, !dbg !52
  %address.local1035 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !52
  %address.local1036 = getelementptr i8, ptr addrspace(3) %address.local1035, i64 1024, !dbg !52
  %1997 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1036, i32 0) #0, !dbg !52
  %1998 = bitcast <4 x i32> %1997 to <8 x half>, !dbg !52
  %address.local1037 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !52
  %address.local1038 = getelementptr i8, ptr addrspace(3) %address.local1037, i64 2048, !dbg !52
  %1999 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1038, i32 0) #0, !dbg !52
  %2000 = bitcast <4 x i32> %1999 to <8 x half>, !dbg !52
  %address.local1039 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !52
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 3072, !dbg !52
  %2001 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !52
  %2002 = bitcast <4 x i32> %2001 to <8 x half>, !dbg !52
  %address.local1041 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !52
  %address.local1042 = getelementptr i8, ptr addrspace(3) %address.local1041, i64 4096, !dbg !52
  %2003 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1042, i32 0) #0, !dbg !52
  %2004 = bitcast <4 x i32> %2003 to <8 x half>, !dbg !52
  %address.local1043 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !52
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1043, i64 5120, !dbg !52
  %2005 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !52
  %2006 = bitcast <4 x i32> %2005 to <8 x half>, !dbg !52
  %address.local1045 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !52
  %address.local1046 = getelementptr i8, ptr addrspace(3) %address.local1045, i64 6144, !dbg !52
  %2007 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1046, i32 0) #0, !dbg !52
  %2008 = bitcast <4 x i32> %2007 to <8 x half>, !dbg !52
  %address.local1047 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !52
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1047, i64 7168, !dbg !52
  %2009 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !52
  %2010 = bitcast <4 x i32> %2009 to <8 x half>, !dbg !52
  %2011 = shufflevector <8 x half> %1996, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %2012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2011, <4 x float> zeroinitializer) #9, !dbg !52
  %2013 = shufflevector <8 x half> %1998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %2014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2013, <4 x float> %2012) #9, !dbg !52
  %2015 = shufflevector <8 x half> %2000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %2016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2015, <4 x float> %2014) #9, !dbg !52
  %2017 = shufflevector <8 x half> %2002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %2018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2017, <4 x float> %2016) #9, !dbg !52
  %2019 = shufflevector <8 x half> %2004, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %2020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2019, <4 x float> %2018) #9, !dbg !52
  %2021 = shufflevector <8 x half> %2006, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %2022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2021, <4 x float> %2020) #9, !dbg !52
  %2023 = shufflevector <8 x half> %2008, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %2024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2023, <4 x float> %2022) #9, !dbg !52
  %2025 = shufflevector <8 x half> %2010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !52
  %2026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2025, <4 x float> %2024) #9, !dbg !52
  %2027 = extractelement <4 x float> %2026, i64 0, !dbg !52
  %2028 = extractelement <4 x float> %2026, i64 1, !dbg !52
  %2029 = extractelement <4 x float> %2026, i64 2, !dbg !52
  %2030 = extractelement <4 x float> %2026, i64 3, !dbg !52
  %2031 = select i1 %.not, float %2028, float %2027, !dbg !52
  %2032 = bitcast float %2031 to i32, !dbg !52
  %2033 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %2032), !dbg !52
  %2034 = bitcast i32 %2033 to float, !dbg !52
  %2035 = select i1 %.not, float %2027, float %2034, !dbg !52
  %2036 = select i1 %.not, float %2034, float %2028, !dbg !52
  %2037 = select i1 %.not, float %2030, float %2029, !dbg !52
  %2038 = bitcast float %2037 to i32, !dbg !52
  %2039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %2038), !dbg !52
  %2040 = bitcast i32 %2039 to float, !dbg !52
  %2041 = select i1 %.not, float %2029, float %2040, !dbg !52
  %2042 = select i1 %.not, float %2040, float %2030, !dbg !52
  %2043 = select i1 %.not1, float %2041, float %2035, !dbg !52
  %2044 = bitcast float %2043 to i32, !dbg !52
  %2045 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %2044), !dbg !52
  %2046 = bitcast i32 %2045 to float, !dbg !52
  %2047 = select i1 %.not1, float %2035, float %2046, !dbg !52
  %2048 = select i1 %.not1, float %2046, float %2041, !dbg !52
  %2049 = select i1 %.not1, float %2042, float %2036, !dbg !52
  %2050 = bitcast float %2049 to i32, !dbg !52
  %2051 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %2050), !dbg !52
  %2052 = bitcast i32 %2051 to float, !dbg !52
  %2053 = select i1 %.not1, float %2036, float %2052, !dbg !52
  %2054 = select i1 %.not1, float %2052, float %2042, !dbg !52
  %2055 = shufflevector <8 x half> %1996, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %2056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2055, <4 x float> zeroinitializer) #9, !dbg !52
  %2057 = shufflevector <8 x half> %1998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %2058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2057, <4 x float> %2056) #9, !dbg !52
  %2059 = shufflevector <8 x half> %2000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %2060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2059, <4 x float> %2058) #9, !dbg !52
  %2061 = shufflevector <8 x half> %2002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %2062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2061, <4 x float> %2060) #9, !dbg !52
  %2063 = shufflevector <8 x half> %2004, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %2064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2063, <4 x float> %2062) #9, !dbg !52
  %2065 = shufflevector <8 x half> %2006, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %2066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2065, <4 x float> %2064) #9, !dbg !52
  %2067 = shufflevector <8 x half> %2008, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %2068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2067, <4 x float> %2066) #9, !dbg !52
  %2069 = shufflevector <8 x half> %2010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !52
  %2070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2069, <4 x float> %2068) #9, !dbg !52
  %2071 = extractelement <4 x float> %2070, i64 0, !dbg !52
  %2072 = extractelement <4 x float> %2070, i64 1, !dbg !52
  %2073 = extractelement <4 x float> %2070, i64 2, !dbg !52
  %2074 = extractelement <4 x float> %2070, i64 3, !dbg !52
  %2075 = select i1 %.not, float %2072, float %2071, !dbg !52
  %2076 = bitcast float %2075 to i32, !dbg !52
  %2077 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %2076), !dbg !52
  %2078 = bitcast i32 %2077 to float, !dbg !52
  %2079 = select i1 %.not, float %2071, float %2078, !dbg !52
  %2080 = select i1 %.not, float %2078, float %2072, !dbg !52
  %2081 = select i1 %.not, float %2074, float %2073, !dbg !52
  %2082 = bitcast float %2081 to i32, !dbg !52
  %2083 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %2082), !dbg !52
  %2084 = bitcast i32 %2083 to float, !dbg !52
  %2085 = select i1 %.not, float %2073, float %2084, !dbg !52
  %2086 = select i1 %.not, float %2084, float %2074, !dbg !52
  %2087 = select i1 %.not1, float %2085, float %2079, !dbg !52
  %2088 = bitcast float %2087 to i32, !dbg !52
  %2089 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %2088), !dbg !52
  %2090 = bitcast i32 %2089 to float, !dbg !52
  %2091 = select i1 %.not1, float %2079, float %2090, !dbg !52
  %2092 = select i1 %.not1, float %2090, float %2085, !dbg !52
  %2093 = select i1 %.not1, float %2086, float %2080, !dbg !52
  %2094 = bitcast float %2093 to i32, !dbg !52
  %2095 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %2094), !dbg !52
  %2096 = bitcast i32 %2095 to float, !dbg !52
  %2097 = select i1 %.not1, float %2080, float %2096, !dbg !52
  %2098 = select i1 %.not1, float %2096, float %2086, !dbg !52
  %2099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2011, <4 x float> zeroinitializer) #9, !dbg !52
  %2100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2013, <4 x float> %2099) #9, !dbg !52
  %2101 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2015, <4 x float> %2100) #9, !dbg !52
  %2102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2017, <4 x float> %2101) #9, !dbg !52
  %2103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2019, <4 x float> %2102) #9, !dbg !52
  %2104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2021, <4 x float> %2103) #9, !dbg !52
  %2105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2023, <4 x float> %2104) #9, !dbg !52
  %2106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2025, <4 x float> %2105) #9, !dbg !52
  %2107 = extractelement <4 x float> %2106, i64 0, !dbg !52
  %2108 = extractelement <4 x float> %2106, i64 1, !dbg !52
  %2109 = extractelement <4 x float> %2106, i64 2, !dbg !52
  %2110 = extractelement <4 x float> %2106, i64 3, !dbg !52
  %2111 = select i1 %.not, float %2108, float %2107, !dbg !52
  %2112 = bitcast float %2111 to i32, !dbg !52
  %2113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %2112), !dbg !52
  %2114 = bitcast i32 %2113 to float, !dbg !52
  %2115 = select i1 %.not, float %2107, float %2114, !dbg !52
  %2116 = select i1 %.not, float %2114, float %2108, !dbg !52
  %2117 = select i1 %.not, float %2110, float %2109, !dbg !52
  %2118 = bitcast float %2117 to i32, !dbg !52
  %2119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %2118), !dbg !52
  %2120 = bitcast i32 %2119 to float, !dbg !52
  %2121 = select i1 %.not, float %2109, float %2120, !dbg !52
  %2122 = select i1 %.not, float %2120, float %2110, !dbg !52
  %2123 = select i1 %.not1, float %2121, float %2115, !dbg !52
  %2124 = bitcast float %2123 to i32, !dbg !52
  %2125 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %2124), !dbg !52
  %2126 = bitcast i32 %2125 to float, !dbg !52
  %2127 = select i1 %.not1, float %2115, float %2126, !dbg !52
  %2128 = select i1 %.not1, float %2126, float %2121, !dbg !52
  %2129 = select i1 %.not1, float %2122, float %2116, !dbg !52
  %2130 = bitcast float %2129 to i32, !dbg !52
  %2131 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %2130), !dbg !52
  %2132 = bitcast i32 %2131 to float, !dbg !52
  %2133 = select i1 %.not1, float %2116, float %2132, !dbg !52
  %2134 = select i1 %.not1, float %2132, float %2122, !dbg !52
  %2135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2055, <4 x float> zeroinitializer) #9, !dbg !52
  %2136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2057, <4 x float> %2135) #9, !dbg !52
  %2137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2059, <4 x float> %2136) #9, !dbg !52
  %2138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2061, <4 x float> %2137) #9, !dbg !52
  %2139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2063, <4 x float> %2138) #9, !dbg !52
  %2140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2065, <4 x float> %2139) #9, !dbg !52
  %2141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2067, <4 x float> %2140) #9, !dbg !52
  %2142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2069, <4 x float> %2141) #9, !dbg !52
  %2143 = extractelement <4 x float> %2142, i64 0, !dbg !52
  %2144 = extractelement <4 x float> %2142, i64 1, !dbg !52
  %2145 = extractelement <4 x float> %2142, i64 2, !dbg !52
  %2146 = extractelement <4 x float> %2142, i64 3, !dbg !52
  %2147 = select i1 %.not, float %2144, float %2143, !dbg !52
  %2148 = bitcast float %2147 to i32, !dbg !52
  %2149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %2148), !dbg !52
  %2150 = bitcast i32 %2149 to float, !dbg !52
  %2151 = select i1 %.not, float %2143, float %2150, !dbg !52
  %2152 = select i1 %.not, float %2150, float %2144, !dbg !52
  %2153 = select i1 %.not, float %2146, float %2145, !dbg !52
  %2154 = bitcast float %2153 to i32, !dbg !52
  %2155 = call i32 @llvm.amdgcn.ds.bpermute(i32 %985, i32 %2154), !dbg !52
  %2156 = bitcast i32 %2155 to float, !dbg !52
  %2157 = select i1 %.not, float %2145, float %2156, !dbg !52
  %2158 = select i1 %.not, float %2156, float %2146, !dbg !52
  %2159 = select i1 %.not1, float %2157, float %2151, !dbg !52
  %2160 = bitcast float %2159 to i32, !dbg !52
  %2161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %2160), !dbg !52
  %2162 = bitcast i32 %2161 to float, !dbg !52
  %2163 = select i1 %.not1, float %2151, float %2162, !dbg !52
  %2164 = select i1 %.not1, float %2162, float %2157, !dbg !52
  %2165 = select i1 %.not1, float %2158, float %2152, !dbg !52
  %2166 = bitcast float %2165 to i32, !dbg !52
  %2167 = call i32 @llvm.amdgcn.ds.bpermute(i32 %993, i32 %2166), !dbg !52
  %2168 = bitcast i32 %2167 to float, !dbg !52
  %2169 = select i1 %.not1, float %2152, float %2168, !dbg !52
  %2170 = select i1 %.not1, float %2168, float %2158, !dbg !52
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  br i1 %43, label %2171, label %2172, !dbg !71

2171:                                             ; preds = %1942
  %address.thread1049 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !71, !range !49, !frisk.register_lifetimes !43
  %address.local1050 = lshr i32 %address.thread1049, 1, !dbg !51
  %address.local1051 = and i32 %address.local1050, 96, !dbg !51
  %address.local1052 = and i32 %address.thread1049, 15, !dbg !51
  %address.local1053 = or disjoint i32 %address.local1051, %address.local1052, !dbg !51
  %address.local1054 = zext nneg i32 %address.local1053 to i64, !dbg !51
  %address.local1055 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !51
  store float %1042, ptr addrspace(3) %address.local1055, align 4, !dbg !71
  %address.local1056 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !51
  %address.local1057 = getelementptr i8, ptr addrspace(3) %address.local1056, i64 64, !dbg !51
  store float %1043, ptr addrspace(3) %address.local1057, align 4, !dbg !71
  br label %2172, !dbg !71

2172:                                             ; preds = %2171, %1942
  fence syncscope("workgroup") release, !dbg !71
  call void @llvm.amdgcn.s.barrier(), !dbg !71
  fence syncscope("workgroup") acquire, !dbg !71
  %2173 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !71
  %lds.const.gep1754 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !71
  %2174 = load float, ptr addrspace(3) %lds.const.gep1754, align 4, !dbg !71
  %2175 = extractvalue [2 x <8 x float>] %60, 0, !dbg !71
  %2176 = shufflevector <8 x float> %2175, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !71
  %2177 = insertelement <4 x float> poison, float %2173, i64 0, !dbg !71
  %2178 = shufflevector <4 x float> %2177, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !71
  %2179 = fmul <4 x float> %2176, %2178, !dbg !71
  %2180 = insertelement <4 x float> poison, float %2047, i64 0, !dbg !72
  %2181 = insertelement <4 x float> %2180, float %2053, i64 1, !dbg !72
  %2182 = insertelement <4 x float> %2181, float %2048, i64 2, !dbg !72
  %2183 = insertelement <4 x float> %2182, float %2054, i64 3, !dbg !72
  %2184 = fadd <4 x float> %2179, %2183, !dbg !72
  %2185 = shufflevector <8 x float> %2175, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !71
  %2186 = fmul <4 x float> %2185, %2178, !dbg !71
  %2187 = insertelement <4 x float> poison, float %2091, i64 0, !dbg !72
  %2188 = insertelement <4 x float> %2187, float %2097, i64 1, !dbg !72
  %2189 = insertelement <4 x float> %2188, float %2092, i64 2, !dbg !72
  %2190 = insertelement <4 x float> %2189, float %2098, i64 3, !dbg !72
  %2191 = fadd <4 x float> %2186, %2190, !dbg !72
  %2192 = extractvalue [2 x <8 x float>] %60, 1, !dbg !71
  %2193 = shufflevector <8 x float> %2192, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !71
  %2194 = insertelement <4 x float> poison, float %2174, i64 0, !dbg !71
  %2195 = shufflevector <4 x float> %2194, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !71
  %2196 = fmul <4 x float> %2193, %2195, !dbg !71
  %2197 = insertelement <4 x float> poison, float %2127, i64 0, !dbg !72
  %2198 = insertelement <4 x float> %2197, float %2133, i64 1, !dbg !72
  %2199 = insertelement <4 x float> %2198, float %2128, i64 2, !dbg !72
  %2200 = insertelement <4 x float> %2199, float %2134, i64 3, !dbg !72
  %2201 = fadd <4 x float> %2196, %2200, !dbg !72
  %2202 = shufflevector <8 x float> %2192, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !71
  %2203 = fmul <4 x float> %2202, %2195, !dbg !71
  %2204 = insertelement <4 x float> poison, float %2163, i64 0, !dbg !72
  %2205 = insertelement <4 x float> %2204, float %2169, i64 1, !dbg !72
  %2206 = insertelement <4 x float> %2205, float %2164, i64 2, !dbg !72
  %2207 = insertelement <4 x float> %2206, float %2170, i64 3, !dbg !72
  %2208 = fadd <4 x float> %2203, %2207, !dbg !72
  %2209 = shufflevector <4 x float> %1510, <4 x float> %1517, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !56
  %2210 = insertvalue [2 x <8 x float>] poison, <8 x float> %2209, 0, !dbg !56
  %2211 = shufflevector <4 x float> %1527, <4 x float> %1534, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !56
  %2212 = insertvalue [2 x <8 x float>] %2210, <8 x float> %2211, 1, !dbg !56
  %2213 = insertelement <1 x float> poison, float %1291, i64 0, !dbg !56
  %2214 = insertvalue [2 x <1 x float>] poison, <1 x float> %2213, 0, !dbg !56
  %2215 = insertelement <1 x float> poison, float %1300, i64 0, !dbg !56
  %2216 = insertvalue [2 x <1 x float>] %2214, <1 x float> %2215, 1, !dbg !56
  %2217 = insertvalue [2 x <1 x float>] poison, <1 x float> %1000, 0, !dbg !56
  %2218 = insertvalue [2 x <1 x float>] %2217, <1 x float> %1013, 1, !dbg !56
  %2219 = shufflevector <4 x float> %1740, <4 x float> %1747, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !56
  %2220 = insertvalue [2 x <8 x float>] poison, <8 x float> %2219, 0, !dbg !56
  %2221 = shufflevector <4 x float> %1757, <4 x float> %1764, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !56
  %2222 = insertvalue [2 x <8 x float>] %2220, <8 x float> %2221, 1, !dbg !56
  %2223 = shufflevector <4 x float> %1970, <4 x float> %1977, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !56
  %2224 = insertvalue [2 x <8 x float>] poison, <8 x float> %2223, 0, !dbg !56
  %2225 = shufflevector <4 x float> %1987, <4 x float> %1994, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !56
  %2226 = insertvalue [2 x <8 x float>] %2224, <8 x float> %2225, 1, !dbg !56
  %2227 = shufflevector <4 x float> %2184, <4 x float> %2191, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !56
  %2228 = insertvalue [2 x <8 x float>] poison, <8 x float> %2227, 0, !dbg !56
  %2229 = shufflevector <4 x float> %2201, <4 x float> %2208, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !56
  %2230 = insertvalue [2 x <8 x float>] %2228, <8 x float> %2229, 1, !dbg !56
  fence syncscope("workgroup") release, !dbg !56
  call void @llvm.amdgcn.s.barrier(), !dbg !56
  fence syncscope("workgroup") acquire, !dbg !56
  %2231 = add nuw nsw i64 %54, 1, !dbg !56
  br label %53, !dbg !56

2232:                                             ; preds = %53
  %2233 = extractvalue [2 x <1 x float>] %56, 0, !dbg !73
  %2234 = shufflevector <1 x float> %2233, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !73
  %2235 = extractvalue [2 x <1 x float>] %56, 1, !dbg !73
  %2236 = shufflevector <1 x float> %2235, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !73
  %2237 = extractvalue [2 x <8 x float>] %55, 1, !dbg !73
  %2238 = shufflevector <8 x float> %2237, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !73
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2236, !dbg !73
  %normalized = fmul <4 x float> %2238, %row.reciprocal, !dbg !73
  %2239 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !74
  %2240 = extractelement <4 x half> %2239, i64 3, !dbg !74
  %2241 = extractelement <4 x half> %2239, i64 2, !dbg !74
  %2242 = extractelement <4 x half> %2239, i64 1, !dbg !74
  %2243 = extractelement <4 x half> %2239, i64 0, !dbg !74
  %2244 = shufflevector <8 x float> %2237, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !73
  %normalized68 = fmul <4 x float> %2244, %row.reciprocal, !dbg !73
  %2245 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !74
  %2246 = extractelement <4 x half> %2245, i64 3, !dbg !74
  %2247 = extractelement <4 x half> %2245, i64 2, !dbg !74
  %2248 = extractelement <4 x half> %2245, i64 1, !dbg !74
  %2249 = extractelement <4 x half> %2245, i64 0, !dbg !74
  %2250 = extractvalue [2 x <8 x float>] %55, 0, !dbg !73
  %2251 = shufflevector <8 x float> %2250, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !73
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2234, !dbg !73
  %normalized70 = fmul <4 x float> %2251, %row.reciprocal69, !dbg !73
  %2252 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !74
  %2253 = extractelement <4 x half> %2252, i64 3, !dbg !74
  %2254 = extractelement <4 x half> %2252, i64 2, !dbg !74
  %2255 = extractelement <4 x half> %2252, i64 1, !dbg !74
  %2256 = extractelement <4 x half> %2252, i64 0, !dbg !74
  %2257 = shufflevector <8 x float> %2250, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !73
  %normalized71 = fmul <4 x float> %2257, %row.reciprocal69, !dbg !73
  %2258 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !74
  %2259 = extractelement <4 x half> %2258, i64 3, !dbg !74
  %2260 = extractelement <4 x half> %2258, i64 2, !dbg !74
  %2261 = extractelement <4 x half> %2258, i64 1, !dbg !74
  %2262 = extractelement <4 x half> %2258, i64 0, !dbg !74
  %2263 = add nsw i64 %15, %25, !dbg !75
  %2264 = shl i64 %2263, 19, !dbg !75
  %2265 = and i64 %2264, 16252928, !dbg !75
  %address.thread1072 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !75, !range !49, !frisk.register_lifetimes !43
  %address.local1073 = zext nneg i32 %address.thread1072 to i64, !dbg !48
  %address.local1074 = add i64 %28, %address.local1073, !dbg !75
  %address.local1075 = lshr i64 %address.local1073, 2, !dbg !55
  %address.local1076 = and i64 %address.local1075, 48, !dbg !55
  %address.local1077 = add i64 %address.local1074, %address.local1076, !dbg !75
  %address.local1078 = and i64 %address.local1073, 240, !dbg !75
  %address.local1080 = sub i64 %address.local1077, %address.local1078, !dbg !75
  %address.local1081 = add i64 %address.local1080, %29, !dbg !75
  %address.local1082 = shl nuw nsw i64 %address.local1081, 7, !dbg !75
  %address.local1083 = add nuw nsw i64 %2265, %address.local1082, !dbg !75
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !75
  %2266 = lshr i64 %address.local1073, 2, !dbg !48
  %address.local1087 = and i64 %2266, 12, !dbg !48
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1087, !dbg !75
  store half %2262, ptr addrspace(1) %address.local1088, align 2, !dbg !75
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !75
  %address.local1090 = or disjoint i64 %address.local1087, 1, !dbg !55
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1090, !dbg !75
  store half %2261, ptr addrspace(1) %address.local1091, align 2, !dbg !75
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !75
  %address.local1093 = or disjoint i64 %address.local1087, 2, !dbg !55
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1093, !dbg !75
  store half %2260, ptr addrspace(1) %address.local1094, align 2, !dbg !75
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !75
  %address.local1096 = or disjoint i64 %address.local1087, 3, !dbg !55
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1096, !dbg !75
  store half %2259, ptr addrspace(1) %address.local1097, align 2, !dbg !75
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !75
  %address.local1099 = or disjoint i64 %address.local1087, 16, !dbg !55
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1098, i64 %address.local1099, !dbg !75
  store half %2256, ptr addrspace(1) %address.local1100, align 2, !dbg !75
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !75
  %address.local1102 = or disjoint i64 %address.local1087, 17, !dbg !55
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !75
  store half %2255, ptr addrspace(1) %address.local1103, align 2, !dbg !75
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !75
  %address.local1105 = or disjoint i64 %address.local1087, 18, !dbg !55
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !75
  store half %2254, ptr addrspace(1) %address.local1106, align 2, !dbg !75
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !75
  %address.local1108 = or disjoint i64 %address.local1087, 19, !dbg !55
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !75
  store half %2253, ptr addrspace(1) %address.local1109, align 2, !dbg !75
  %address.local1110 = shl i64 %address.local1081, 7, !dbg !75
  %address.local1111 = add i64 %address.local1110, 8192, !dbg !75
  %address.local1112 = add nuw nsw i64 %2265, %address.local1111, !dbg !75
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !75
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1087, !dbg !75
  store half %2249, ptr addrspace(1) %address.local1114, align 2, !dbg !75
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !75
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1090, !dbg !75
  store half %2248, ptr addrspace(1) %address.local1116, align 2, !dbg !75
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !75
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1093, !dbg !75
  store half %2247, ptr addrspace(1) %address.local1118, align 2, !dbg !75
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !75
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1096, !dbg !75
  store half %2246, ptr addrspace(1) %address.local1120, align 2, !dbg !75
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !75
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1099, !dbg !75
  store half %2243, ptr addrspace(1) %address.local1122, align 2, !dbg !75
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !75
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1102, !dbg !75
  store half %2242, ptr addrspace(1) %address.local1124, align 2, !dbg !75
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !75
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1105, !dbg !75
  store half %2241, ptr addrspace(1) %address.local1126, align 2, !dbg !75
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !75
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1108, !dbg !75
  store half %2240, ptr addrspace(1) %address.local1128, align 2, !dbg !75
  %2267 = extractvalue [2 x <8 x float>] %58, 1, !dbg !73
  %2268 = shufflevector <8 x float> %2267, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !73
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2236, !dbg !73
  %normalized73 = fmul <4 x float> %2268, %row.reciprocal72, !dbg !73
  %2269 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !74
  %2270 = extractelement <4 x half> %2269, i64 3, !dbg !74
  %2271 = extractelement <4 x half> %2269, i64 2, !dbg !74
  %2272 = extractelement <4 x half> %2269, i64 1, !dbg !74
  %2273 = extractelement <4 x half> %2269, i64 0, !dbg !74
  %2274 = shufflevector <8 x float> %2267, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !73
  %normalized74 = fmul <4 x float> %2274, %row.reciprocal72, !dbg !73
  %2275 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !74
  %2276 = extractelement <4 x half> %2275, i64 3, !dbg !74
  %2277 = extractelement <4 x half> %2275, i64 2, !dbg !74
  %2278 = extractelement <4 x half> %2275, i64 1, !dbg !74
  %2279 = extractelement <4 x half> %2275, i64 0, !dbg !74
  %2280 = extractvalue [2 x <8 x float>] %58, 0, !dbg !73
  %2281 = shufflevector <8 x float> %2280, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !73
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2234, !dbg !73
  %normalized76 = fmul <4 x float> %2281, %row.reciprocal75, !dbg !73
  %2282 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !74
  %2283 = extractelement <4 x half> %2282, i64 3, !dbg !74
  %2284 = extractelement <4 x half> %2282, i64 2, !dbg !74
  %2285 = extractelement <4 x half> %2282, i64 1, !dbg !74
  %2286 = extractelement <4 x half> %2282, i64 0, !dbg !74
  %2287 = shufflevector <8 x float> %2280, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !73
  %normalized77 = fmul <4 x float> %2287, %row.reciprocal75, !dbg !73
  %2288 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !74
  %2289 = extractelement <4 x half> %2288, i64 3, !dbg !74
  %2290 = extractelement <4 x half> %2288, i64 2, !dbg !74
  %2291 = extractelement <4 x half> %2288, i64 1, !dbg !74
  %2292 = extractelement <4 x half> %2288, i64 0, !dbg !74
  %2293 = shl nuw nsw i64 %27, 19, !dbg !75
  %address.thread1129 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !75, !range !49, !frisk.register_lifetimes !43
  %address.local1130 = zext nneg i32 %address.thread1129 to i64, !dbg !48
  %address.local1131 = add i64 %28, %address.local1130, !dbg !75
  %address.local1132 = lshr i64 %address.local1130, 2, !dbg !55
  %address.local1133 = and i64 %address.local1132, 48, !dbg !55
  %address.local1134 = add i64 %address.local1131, %address.local1133, !dbg !75
  %address.local1135 = and i64 %address.local1130, 240, !dbg !75
  %address.local1137 = sub i64 %address.local1134, %address.local1135, !dbg !75
  %address.local1138 = add i64 %address.local1137, %29, !dbg !75
  %address.local1139 = shl nuw nsw i64 %address.local1138, 7, !dbg !75
  %address.local1140 = add nuw nsw i64 %2293, %address.local1139, !dbg !75
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !75
  %2294 = lshr i64 %address.local1130, 2, !dbg !48
  %address.local1144 = and i64 %2294, 12, !dbg !48
  %address.local1145 = or disjoint i64 %address.local1144, 32, !dbg !55
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1145, !dbg !75
  store half %2292, ptr addrspace(1) %address.local1146, align 2, !dbg !75
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !75
  %address.local1148 = or disjoint i64 %address.local1144, 33, !dbg !55
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1148, !dbg !75
  store half %2291, ptr addrspace(1) %address.local1149, align 2, !dbg !75
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !75
  %address.local1151 = or disjoint i64 %address.local1144, 34, !dbg !55
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1150, i64 %address.local1151, !dbg !75
  store half %2290, ptr addrspace(1) %address.local1152, align 2, !dbg !75
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !75
  %address.local1154 = or disjoint i64 %address.local1144, 35, !dbg !55
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1154, !dbg !75
  store half %2289, ptr addrspace(1) %address.local1155, align 2, !dbg !75
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !75
  %address.local1157 = or disjoint i64 %address.local1144, 48, !dbg !55
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1156, i64 %address.local1157, !dbg !75
  store half %2286, ptr addrspace(1) %address.local1158, align 2, !dbg !75
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !75
  %address.local1160 = or disjoint i64 %address.local1144, 49, !dbg !55
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1160, !dbg !75
  store half %2285, ptr addrspace(1) %address.local1161, align 2, !dbg !75
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !75
  %address.local1163 = or disjoint i64 %address.local1144, 50, !dbg !55
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1162, i64 %address.local1163, !dbg !75
  store half %2284, ptr addrspace(1) %address.local1164, align 2, !dbg !75
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !75
  %address.local1166 = or i64 %2294, 51, !dbg !55
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1166, !dbg !75
  store half %2283, ptr addrspace(1) %address.local1167, align 2, !dbg !75
  %address.local1168 = shl i64 %address.local1138, 7, !dbg !75
  %address.local1169 = add i64 %address.local1168, 8192, !dbg !75
  %address.local1170 = add nuw nsw i64 %2293, %address.local1169, !dbg !75
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !75
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1145, !dbg !75
  store half %2279, ptr addrspace(1) %address.local1172, align 2, !dbg !75
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !75
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1148, !dbg !75
  store half %2278, ptr addrspace(1) %address.local1174, align 2, !dbg !75
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !75
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1151, !dbg !75
  store half %2277, ptr addrspace(1) %address.local1176, align 2, !dbg !75
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !75
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1154, !dbg !75
  store half %2276, ptr addrspace(1) %address.local1178, align 2, !dbg !75
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !75
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1157, !dbg !75
  store half %2273, ptr addrspace(1) %address.local1180, align 2, !dbg !75
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !75
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1160, !dbg !75
  store half %2272, ptr addrspace(1) %address.local1182, align 2, !dbg !75
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !75
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1163, !dbg !75
  store half %2271, ptr addrspace(1) %address.local1184, align 2, !dbg !75
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !75
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1166, !dbg !75
  store half %2270, ptr addrspace(1) %address.local1186, align 2, !dbg !75
  %2295 = extractvalue [2 x <8 x float>] %59, 1, !dbg !73
  %2296 = shufflevector <8 x float> %2295, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !73
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2236, !dbg !73
  %normalized79 = fmul <4 x float> %2296, %row.reciprocal78, !dbg !73
  %2297 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !74
  %2298 = extractelement <4 x half> %2297, i64 3, !dbg !74
  %2299 = extractelement <4 x half> %2297, i64 2, !dbg !74
  %2300 = extractelement <4 x half> %2297, i64 1, !dbg !74
  %2301 = extractelement <4 x half> %2297, i64 0, !dbg !74
  %2302 = shufflevector <8 x float> %2295, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !73
  %normalized80 = fmul <4 x float> %2302, %row.reciprocal78, !dbg !73
  %2303 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !74
  %2304 = extractelement <4 x half> %2303, i64 3, !dbg !74
  %2305 = extractelement <4 x half> %2303, i64 2, !dbg !74
  %2306 = extractelement <4 x half> %2303, i64 1, !dbg !74
  %2307 = extractelement <4 x half> %2303, i64 0, !dbg !74
  %2308 = extractvalue [2 x <8 x float>] %59, 0, !dbg !73
  %2309 = shufflevector <8 x float> %2308, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !73
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2234, !dbg !73
  %normalized82 = fmul <4 x float> %2309, %row.reciprocal81, !dbg !73
  %2310 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !74
  %2311 = extractelement <4 x half> %2310, i64 3, !dbg !74
  %2312 = extractelement <4 x half> %2310, i64 2, !dbg !74
  %2313 = extractelement <4 x half> %2310, i64 1, !dbg !74
  %2314 = extractelement <4 x half> %2310, i64 0, !dbg !74
  %2315 = shufflevector <8 x float> %2308, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !73
  %normalized83 = fmul <4 x float> %2315, %row.reciprocal81, !dbg !73
  %2316 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !74
  %2317 = extractelement <4 x half> %2316, i64 3, !dbg !74
  %2318 = extractelement <4 x half> %2316, i64 2, !dbg !74
  %2319 = extractelement <4 x half> %2316, i64 1, !dbg !74
  %2320 = extractelement <4 x half> %2316, i64 0, !dbg !74
  %2321 = shl nuw nsw i64 %27, 19, !dbg !75
  %address.thread1187 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !75, !range !49, !frisk.register_lifetimes !43
  %address.local1188 = zext nneg i32 %address.thread1187 to i64, !dbg !48
  %address.local1189 = add i64 %28, %address.local1188, !dbg !75
  %address.local1190 = lshr i64 %address.local1188, 2, !dbg !55
  %address.local1191 = and i64 %address.local1190, 48, !dbg !55
  %address.local1192 = add i64 %address.local1189, %address.local1191, !dbg !75
  %address.local1193 = and i64 %address.local1188, 240, !dbg !75
  %address.local1195 = sub i64 %address.local1192, %address.local1193, !dbg !75
  %address.local1196 = add i64 %address.local1195, %29, !dbg !75
  %address.local1197 = shl nuw nsw i64 %address.local1196, 7, !dbg !75
  %address.local1198 = add nuw nsw i64 %2321, %address.local1197, !dbg !75
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !75
  %2322 = lshr i64 %address.local1188, 2, !dbg !48
  %address.local1202 = and i64 %2322, 12, !dbg !48
  %address.local1203 = or disjoint i64 %address.local1202, 64, !dbg !55
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1203, !dbg !75
  store half %2320, ptr addrspace(1) %address.local1204, align 2, !dbg !75
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !75
  %address.local1206 = or disjoint i64 %address.local1202, 65, !dbg !55
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1206, !dbg !75
  store half %2319, ptr addrspace(1) %address.local1207, align 2, !dbg !75
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !75
  %address.local1209 = or disjoint i64 %address.local1202, 66, !dbg !55
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1208, i64 %address.local1209, !dbg !75
  store half %2318, ptr addrspace(1) %address.local1210, align 2, !dbg !75
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !75
  %address.local1212 = or disjoint i64 %address.local1202, 67, !dbg !55
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1211, i64 %address.local1212, !dbg !75
  store half %2317, ptr addrspace(1) %address.local1213, align 2, !dbg !75
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !75
  %address.local1215 = or disjoint i64 %address.local1202, 80, !dbg !55
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1214, i64 %address.local1215, !dbg !75
  store half %2314, ptr addrspace(1) %address.local1216, align 2, !dbg !75
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !75
  %address.local1218 = or disjoint i64 %address.local1202, 81, !dbg !55
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1217, i64 %address.local1218, !dbg !75
  store half %2313, ptr addrspace(1) %address.local1219, align 2, !dbg !75
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !75
  %address.local1221 = or disjoint i64 %address.local1202, 82, !dbg !55
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1221, !dbg !75
  store half %2312, ptr addrspace(1) %address.local1222, align 2, !dbg !75
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !75
  %address.local1224 = or disjoint i64 %address.local1202, 83, !dbg !55
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1223, i64 %address.local1224, !dbg !75
  store half %2311, ptr addrspace(1) %address.local1225, align 2, !dbg !75
  %address.local1226 = shl i64 %address.local1196, 7, !dbg !75
  %address.local1227 = add i64 %address.local1226, 8192, !dbg !75
  %address.local1228 = add nuw nsw i64 %2321, %address.local1227, !dbg !75
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !75
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1229, i64 %address.local1203, !dbg !75
  store half %2307, ptr addrspace(1) %address.local1230, align 2, !dbg !75
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !75
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1231, i64 %address.local1206, !dbg !75
  store half %2306, ptr addrspace(1) %address.local1232, align 2, !dbg !75
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !75
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1233, i64 %address.local1209, !dbg !75
  store half %2305, ptr addrspace(1) %address.local1234, align 2, !dbg !75
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !75
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1235, i64 %address.local1212, !dbg !75
  store half %2304, ptr addrspace(1) %address.local1236, align 2, !dbg !75
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !75
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1237, i64 %address.local1215, !dbg !75
  store half %2301, ptr addrspace(1) %address.local1238, align 2, !dbg !75
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !75
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1239, i64 %address.local1218, !dbg !75
  store half %2300, ptr addrspace(1) %address.local1240, align 2, !dbg !75
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !75
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1241, i64 %address.local1221, !dbg !75
  store half %2299, ptr addrspace(1) %address.local1242, align 2, !dbg !75
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !75
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1243, i64 %address.local1224, !dbg !75
  store half %2298, ptr addrspace(1) %address.local1244, align 2, !dbg !75
  %2323 = extractvalue [2 x <8 x float>] %60, 1, !dbg !73
  %2324 = shufflevector <8 x float> %2323, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !73
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2236, !dbg !73
  %normalized85 = fmul <4 x float> %2324, %row.reciprocal84, !dbg !73
  %2325 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !74
  %2326 = extractelement <4 x half> %2325, i64 3, !dbg !74
  %2327 = extractelement <4 x half> %2325, i64 2, !dbg !74
  %2328 = extractelement <4 x half> %2325, i64 1, !dbg !74
  %2329 = extractelement <4 x half> %2325, i64 0, !dbg !74
  %2330 = shufflevector <8 x float> %2323, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !73
  %normalized86 = fmul <4 x float> %2330, %row.reciprocal84, !dbg !73
  %2331 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !74
  %2332 = extractelement <4 x half> %2331, i64 3, !dbg !74
  %2333 = extractelement <4 x half> %2331, i64 2, !dbg !74
  %2334 = extractelement <4 x half> %2331, i64 1, !dbg !74
  %2335 = extractelement <4 x half> %2331, i64 0, !dbg !74
  %2336 = extractvalue [2 x <8 x float>] %60, 0, !dbg !73
  %2337 = shufflevector <8 x float> %2336, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !73
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2234, !dbg !73
  %normalized88 = fmul <4 x float> %2337, %row.reciprocal87, !dbg !73
  %2338 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !74
  %2339 = extractelement <4 x half> %2338, i64 3, !dbg !74
  %2340 = extractelement <4 x half> %2338, i64 2, !dbg !74
  %2341 = extractelement <4 x half> %2338, i64 1, !dbg !74
  %2342 = extractelement <4 x half> %2338, i64 0, !dbg !74
  %2343 = shufflevector <8 x float> %2336, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !73
  %normalized89 = fmul <4 x float> %2343, %row.reciprocal87, !dbg !73
  %2344 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !74
  %2345 = extractelement <4 x half> %2344, i64 3, !dbg !74
  %2346 = extractelement <4 x half> %2344, i64 2, !dbg !74
  %2347 = extractelement <4 x half> %2344, i64 1, !dbg !74
  %2348 = extractelement <4 x half> %2344, i64 0, !dbg !74
  %2349 = shl nuw nsw i64 %27, 19, !dbg !75
  %address.thread1245 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !75, !range !49, !frisk.register_lifetimes !43
  %address.local1246 = zext nneg i32 %address.thread1245 to i64, !dbg !48
  %address.local1247 = add i64 %28, %address.local1246, !dbg !75
  %address.local1248 = lshr i64 %address.local1246, 2, !dbg !55
  %address.local1249 = and i64 %address.local1248, 48, !dbg !55
  %address.local1250 = add i64 %address.local1247, %address.local1249, !dbg !75
  %address.local1251 = and i64 %address.local1246, 240, !dbg !75
  %address.local1253 = sub i64 %address.local1250, %address.local1251, !dbg !75
  %address.local1254 = add i64 %address.local1253, %29, !dbg !75
  %address.local1255 = shl nuw nsw i64 %address.local1254, 7, !dbg !75
  %address.local1256 = add nuw nsw i64 %2349, %address.local1255, !dbg !75
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !75
  %2350 = lshr i64 %address.local1246, 2, !dbg !48
  %address.local1260 = and i64 %2350, 12, !dbg !48
  %address.local1261 = or disjoint i64 %address.local1260, 96, !dbg !55
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1261, !dbg !75
  store half %2348, ptr addrspace(1) %address.local1262, align 2, !dbg !75
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !75
  %address.local1264 = or disjoint i64 %address.local1260, 97, !dbg !55
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1264, !dbg !75
  store half %2347, ptr addrspace(1) %address.local1265, align 2, !dbg !75
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !75
  %address.local1267 = or disjoint i64 %address.local1260, 98, !dbg !55
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1266, i64 %address.local1267, !dbg !75
  store half %2346, ptr addrspace(1) %address.local1268, align 2, !dbg !75
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !75
  %address.local1270 = or disjoint i64 %address.local1260, 99, !dbg !55
  %address.local1271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1269, i64 %address.local1270, !dbg !75
  store half %2345, ptr addrspace(1) %address.local1271, align 2, !dbg !75
  %address.local1272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !75
  %address.local1273 = or disjoint i64 %address.local1260, 112, !dbg !55
  %address.local1274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1272, i64 %address.local1273, !dbg !75
  store half %2342, ptr addrspace(1) %address.local1274, align 2, !dbg !75
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !75
  %address.local1276 = or disjoint i64 %address.local1260, 113, !dbg !55
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1275, i64 %address.local1276, !dbg !75
  store half %2341, ptr addrspace(1) %address.local1277, align 2, !dbg !75
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !75
  %address.local1279 = or disjoint i64 %address.local1260, 114, !dbg !55
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1278, i64 %address.local1279, !dbg !75
  store half %2340, ptr addrspace(1) %address.local1280, align 2, !dbg !75
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !75
  %address.local1282 = or i64 %2350, 115, !dbg !55
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1281, i64 %address.local1282, !dbg !75
  store half %2339, ptr addrspace(1) %address.local1283, align 2, !dbg !75
  %address.local1284 = shl i64 %address.local1254, 7, !dbg !75
  %address.local1285 = add i64 %address.local1284, 8192, !dbg !75
  %address.local1286 = add nuw nsw i64 %2349, %address.local1285, !dbg !75
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !75
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1287, i64 %address.local1261, !dbg !75
  store half %2335, ptr addrspace(1) %address.local1288, align 2, !dbg !75
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !75
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1264, !dbg !75
  store half %2334, ptr addrspace(1) %address.local1290, align 2, !dbg !75
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !75
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1291, i64 %address.local1267, !dbg !75
  store half %2333, ptr addrspace(1) %address.local1292, align 2, !dbg !75
  %address.local1293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !75
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1293, i64 %address.local1270, !dbg !75
  store half %2332, ptr addrspace(1) %address.local1294, align 2, !dbg !75
  %address.local1295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !75
  %address.local1296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1295, i64 %address.local1273, !dbg !75
  store half %2329, ptr addrspace(1) %address.local1296, align 2, !dbg !75
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !75
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1297, i64 %address.local1276, !dbg !75
  store half %2328, ptr addrspace(1) %address.local1298, align 2, !dbg !75
  %address.local1299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !75
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1299, i64 %address.local1279, !dbg !75
  store half %2327, ptr addrspace(1) %address.local1300, align 2, !dbg !75
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !75
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1282, !dbg !75
  store half %2326, ptr addrspace(1) %address.local1302, align 2, !dbg !75
  ret void, !dbg !45
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

!llvm.dbg.cu = !{!37}
!llvm.module.flags = !{!39}

!37 = distinct !DICompileUnit(language: DW_LANG_C, file: !38, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!38 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000001")
!39 = !{i32 2, !"Debug Info Version", i32 3}
!41 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !38, file: !38, line: 23, type: !42, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !37)
!42 = !DISubroutineType(types: !43)
!43 = !{}
!44 = !{i32 256, i32 1, i32 1}
!45 = !DILocation(line: 23, column: 3, scope: !41)
!46 = !DILocation(line: 48, column: 17, scope: !41)
!47 = !DILocation(line: 70, column: 22, scope: !41)
!48 = !DILocation(line: 72, column: 24, scope: !41)
!49 = !{i32 0, i32 256}
!50 = !DILocation(line: 77, column: 19, scope: !41)
!51 = !DILocation(line: 92, column: 20, scope: !41)
!52 = !DILocation(line: 102, column: 23, scope: !41)
!53 = !DILocation(line: 0, scope: !41)
!54 = !DILocation(line: 71, column: 22, scope: !41)
!55 = !DILocation(line: 63, column: 17, scope: !41)
!56 = !DILocation(line: 60, column: 24, scope: !41)
!57 = !DILocation(line: 73, column: 26, scope: !41)
!58 = !DILocation(line: 74, column: 23, scope: !41)
!59 = !DILocation(line: 79, column: 24, scope: !41)
!60 = !DILocation(line: 80, column: 23, scope: !41)
!61 = !DILocation(line: 81, column: 26, scope: !41)
!62 = !DILocation(line: 88, column: 21, scope: !41)
!63 = !DILocation(line: 89, column: 20, scope: !41)
!64 = !DILocation(line: 90, column: 22, scope: !41)
!65 = !DILocation(line: 91, column: 23, scope: !41)
!66 = !DILocation(line: 93, column: 22, scope: !41)
!67 = !DILocation(line: 101, column: 20, scope: !41)
!68 = !DILocation(line: 94, column: 22, scope: !41)
!69 = !DILocation(line: 96, column: 16, scope: !41)
!70 = !DILocation(line: 95, column: 20, scope: !41)
!71 = !DILocation(line: 103, column: 22, scope: !41)
!72 = !DILocation(line: 104, column: 20, scope: !41)
!73 = !DILocation(line: 107, column: 23, scope: !41)
!74 = !DILocation(line: 108, column: 19, scope: !41)
!75 = !DILocation(line: 109, column: 9, scope: !41)
