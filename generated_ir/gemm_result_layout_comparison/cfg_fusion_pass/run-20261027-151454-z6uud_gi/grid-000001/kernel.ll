; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-151454-z6uud_gi/grid-000001/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-151454-z6uud_gi/grid-000001/input.mlir"

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
  %47 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %48 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %49 = getelementptr i8, ptr addrspace(1) %48, i64 64, !dbg !11
  %50 = shl i64 %15, 7, !dbg !18
  %51 = and i64 %50, 3968, !dbg !18
  %52 = shl nsw i64 %18, 12, !dbg !18
  %53 = or disjoint i64 %51, %52, !dbg !18
  %54 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %55 = getelementptr i8, ptr addrspace(1) %54, i64 128, !dbg !11
  %56 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %57 = getelementptr i8, ptr addrspace(1) %56, i64 192, !dbg !11
  %58 = lshr i64 %35, 2, !dbg !13
  %59 = and i64 %58, 48, !dbg !13
  br label %60, !dbg !7

60:                                               ; preds = %2364, %4
  %61 = phi i64 [ %2423, %2364 ], [ 0, %4 ], !dbg !7
  %62 = phi [2 x <8 x float>] [ %2404, %2364 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi [2 x <1 x float>] [ %2408, %2364 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi [2 x <1 x float>] [ %2410, %2364 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %65 = phi [2 x <8 x float>] [ %2414, %2364 ], [ zeroinitializer, %4 ], !dbg !7
  %66 = phi [2 x <8 x float>] [ %2418, %2364 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = phi [2 x <8 x float>] [ %2422, %2364 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = icmp slt i64 %61, %46, !dbg !7
  br i1 %68, label %69, label %2424, !dbg !7

69:                                               ; preds = %60
  %70 = shl nsw i64 %61, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local258 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local259 = and i64 %address.local258, 1536, !dbg !11
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local259, !dbg !11
  %address.local261 = ptrtoint ptr addrspace(3) %address.local260 to i64, !dbg !11
  %address.local262 = trunc i64 %address.local261 to i32, !dbg !11
  %71 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %72 = zext i32 %71 to i64, !dbg !11
  %73 = inttoptr i64 %72 to ptr addrspace(3), !dbg !11
  %address.local263 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local264 = and i64 %address.local263, 16256, !dbg !11
  %address.local265 = lshr i64 %address.local, 4, !dbg !11
  %address.local266 = and i64 %address.local265, 8, !dbg !11
  %address.local267 = or disjoint i64 %address.local264, %address.local266, !dbg !11
  %address.local268 = getelementptr [2 x i8], ptr addrspace(1) %47, i64 %address.local267, !dbg !11
  %74 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %75 = ptrtoint ptr addrspace(1) %address.local268 to i64, !dbg !11
  %76 = sub i64 %75, %74, !dbg !11
  %dma.buffer.offset1821 = trunc i64 %76 to i32, !dbg !11
  %77 = trunc i64 %74 to i32, !dbg !11
  %78 = insertelement <4 x i32> poison, i32 %77, i64 0, !dbg !11
  %79 = lshr i64 %74, 32, !dbg !11
  %80 = trunc i64 %79 to i32, !dbg !11
  %81 = insertelement <4 x i32> %78, i32 %80, i64 1, !dbg !11
  %82 = insertelement <4 x i32> %81, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1822 = insertelement <4 x i32> %82, i32 131072, i64 3, !dbg !11
  %83 = ptrtoint ptr addrspace(3) %73 to i32, !dbg !11
  %dma.buffer.lds1823 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !11
  %84 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1821, <4 x i32> %dma.buffer.rsrc1822) #10, !dbg !11
  %address.local269 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local270 = shl nuw nsw i64 %address.local269, 3, !dbg !11
  %address.local271 = and i64 %address.local270, 3584, !dbg !11
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local271, !dbg !11
  %address.local273 = ptrtoint ptr addrspace(3) %address.local272 to i64, !dbg !11
  %address.local274 = trunc i64 %address.local273 to i32, !dbg !11
  %85 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %86 = zext i32 %85 to i64, !dbg !11
  %87 = inttoptr i64 %86 to ptr addrspace(3), !dbg !11
  %address.local275 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local276 = and i64 %address.local275, 16256, !dbg !11
  %address.local277 = lshr i64 %address.local269, 4, !dbg !11
  %address.local278 = and i64 %address.local277, 24, !dbg !11
  %address.local279 = or disjoint i64 %address.local276, %address.local278, !dbg !11
  %address.local280 = getelementptr [2 x i8], ptr addrspace(1) %47, i64 %address.local279, !dbg !11
  %88 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %89 = ptrtoint ptr addrspace(1) %address.local280 to i64, !dbg !11
  %90 = sub i64 %89, %88, !dbg !11
  %dma.buffer.offset1818 = trunc i64 %90 to i32, !dbg !11
  %91 = trunc i64 %88 to i32, !dbg !11
  %92 = insertelement <4 x i32> poison, i32 %91, i64 0, !dbg !11
  %93 = lshr i64 %88, 32, !dbg !11
  %94 = trunc i64 %93 to i32, !dbg !11
  %95 = insertelement <4 x i32> %92, i32 %94, i64 1, !dbg !11
  %96 = insertelement <4 x i32> %95, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1819 = insertelement <4 x i32> %96, i32 131072, i64 3, !dbg !11
  %97 = ptrtoint ptr addrspace(3) %87 to i32, !dbg !11
  %dma.buffer.lds1820 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %97), !dbg !11
  %98 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1818, <4 x i32> %dma.buffer.rsrc1819) #10, !dbg !11
  %99 = lshr i64 %61, 12, !dbg !18
  %100 = add nuw nsw i64 %99, %15, !dbg !18
  %101 = lshr i64 %61, 5, !dbg !18
  %102 = and i64 %101, 127, !dbg !18
  %103 = shl i64 %100, 19, !dbg !18
  %104 = and i64 %103, 16252928, !dbg !18
  %105 = shl i64 %101, 19, !dbg !18
  %106 = shl i64 %61, 14, !dbg !18
  %107 = sub i64 %106, %105, !dbg !18
  %108 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %104, !dbg !18
  %109 = getelementptr [2 x i8], ptr addrspace(1) %108, i64 %102, !dbg !18
  %110 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %107, !dbg !18
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local259, !dbg !18
  %address.local282 = ptrtoint ptr addrspace(3) %address.local281 to i64, !dbg !18
  %address.local283 = trunc i64 %address.local282 to i32, !dbg !18
  %111 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %112 = zext i32 %111 to i64, !dbg !18
  %113 = inttoptr i64 %112 to ptr addrspace(3), !dbg !18
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
  %address.local294 = getelementptr [2 x i8], ptr addrspace(1) %110, i64 %address.local293, !dbg !18
  %114 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %115 = ptrtoint ptr addrspace(1) %address.local294 to i64, !dbg !18
  %116 = sub i64 %115, %114, !dbg !18
  %dma.buffer.offset1815 = trunc i64 %116 to i32, !dbg !18
  %117 = trunc i64 %114 to i32, !dbg !18
  %118 = insertelement <4 x i32> poison, i32 %117, i64 0, !dbg !18
  %119 = lshr i64 %114, 32, !dbg !18
  %120 = trunc i64 %119 to i32, !dbg !18
  %121 = insertelement <4 x i32> %118, i32 %120, i64 1, !dbg !18
  %122 = insertelement <4 x i32> %121, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1816 = insertelement <4 x i32> %122, i32 131072, i64 3, !dbg !18
  %123 = ptrtoint ptr addrspace(3) %113 to i32, !dbg !18
  %dma.buffer.lds1817 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %123), !dbg !18
  %124 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1815, <4 x i32> %dma.buffer.rsrc1816) #10, !dbg !18
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local271, !dbg !18
  %address.local296 = ptrtoint ptr addrspace(3) %address.local295 to i64, !dbg !18
  %address.local297 = trunc i64 %address.local296 to i32, !dbg !18
  %125 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %126 = zext i32 %125 to i64, !dbg !18
  %127 = inttoptr i64 %126 to ptr addrspace(3), !dbg !18
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
  %address.local308 = getelementptr [2 x i8], ptr addrspace(1) %110, i64 %address.local307, !dbg !18
  %128 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %129 = ptrtoint ptr addrspace(1) %address.local308 to i64, !dbg !18
  %130 = sub i64 %129, %128, !dbg !18
  %dma.buffer.offset1812 = trunc i64 %130 to i32, !dbg !18
  %131 = trunc i64 %128 to i32, !dbg !18
  %132 = insertelement <4 x i32> poison, i32 %131, i64 0, !dbg !18
  %133 = lshr i64 %128, 32, !dbg !18
  %134 = trunc i64 %133 to i32, !dbg !18
  %135 = insertelement <4 x i32> %132, i32 %134, i64 1, !dbg !18
  %136 = insertelement <4 x i32> %135, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1813 = insertelement <4 x i32> %136, i32 131072, i64 3, !dbg !18
  %137 = ptrtoint ptr addrspace(3) %127 to i32, !dbg !18
  %dma.buffer.lds1814 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %137), !dbg !18
  %138 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1812, <4 x i32> %dma.buffer.rsrc1813) #10, !dbg !18
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
  %139 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %141 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !12
  %142 = fmul <4 x half> %139, splat (half 0xH3015), !dbg !12
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %143 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !12
  %144 = fmul <4 x half> %140, splat (half 0xH3015), !dbg !12
  %145 = fmul <4 x half> %141, splat (half 0xH3015), !dbg !12
  %146 = fmul <4 x half> %143, splat (half 0xH3015), !dbg !12
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
  %147 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1561 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %149 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !19
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
  %150 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %149, <4 x float> %148) #9, !dbg !19
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1562 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %153 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !19
  %lds.const.gep1563 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %154 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !19
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %153, <4 x float> %152) #9, !dbg !19
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1564 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %157 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !19
  %lds.const.gep1565 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %158 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !19
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %157, <4 x float> %156) #9, !dbg !19
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1566 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !19
  %lds.const.gep1567 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !19
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %161, <4 x float> %160) #9, !dbg !19
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1568 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %165 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !19
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %166 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !19
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %165, <4 x float> %164) #9, !dbg !19
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %166, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %169 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !19
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %170 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !19
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %169, <4 x float> %168) #9, !dbg !19
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %170, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !19
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !19
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %173, <4 x float> %172) #9, !dbg !19
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %174, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %177 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !19
  %178 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %177, <4 x float> %176) #9, !dbg !19
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %178, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1575 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1575, align 8, !dbg !19
  %182 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %181, <4 x float> %180) #9, !dbg !19
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %182, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1576 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %185 = load <4 x half>, ptr addrspace(3) %lds.const.gep1576, align 8, !dbg !19
  %lds.const.gep1577 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %186 = load <4 x half>, ptr addrspace(3) %lds.const.gep1577, align 8, !dbg !19
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %185, <4 x float> %184) #9, !dbg !19
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %186, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1578 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %189 = load <4 x half>, ptr addrspace(3) %lds.const.gep1578, align 8, !dbg !19
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %190 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !19
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %189, <4 x float> %188) #9, !dbg !19
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %190, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %193 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !19
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %194 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !19
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %193, <4 x float> %192) #9, !dbg !19
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %194, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %197 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !19
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %198 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !19
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %197, <4 x float> %196) #9, !dbg !19
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %198, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !19
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !19
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %201, <4 x float> %200) #9, !dbg !19
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %202, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %205 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !19
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !19
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %205, <4 x float> %204) #9, !dbg !19
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %206, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %209 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !19
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %209, <4 x float> %208) #9, !dbg !19
  %211 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %212 = zext i32 %211 to i64, !dbg !11
  %213 = inttoptr i64 %212 to ptr addrspace(3), !dbg !11
  %address.local401 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local267, !dbg !11
  %214 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %215 = ptrtoint ptr addrspace(1) %address.local401 to i64, !dbg !11
  %216 = sub i64 %215, %214, !dbg !11
  %dma.buffer.offset1809 = trunc i64 %216 to i32, !dbg !11
  %217 = trunc i64 %214 to i32, !dbg !11
  %218 = insertelement <4 x i32> poison, i32 %217, i64 0, !dbg !11
  %219 = lshr i64 %214, 32, !dbg !11
  %220 = trunc i64 %219 to i32, !dbg !11
  %221 = insertelement <4 x i32> %218, i32 %220, i64 1, !dbg !11
  %222 = insertelement <4 x i32> %221, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1810 = insertelement <4 x i32> %222, i32 131072, i64 3, !dbg !11
  %223 = ptrtoint ptr addrspace(3) %213 to i32, !dbg !11
  %dma.buffer.lds1811 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %223), !dbg !11
  %224 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1811, i32 %dma.buffer.offset1809, <4 x i32> %dma.buffer.rsrc1810) #10, !dbg !11
  %225 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %226 = zext i32 %225 to i64, !dbg !11
  %227 = inttoptr i64 %226 to ptr addrspace(3), !dbg !11
  %address.local402 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local279, !dbg !11
  %228 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %229 = ptrtoint ptr addrspace(1) %address.local402 to i64, !dbg !11
  %230 = sub i64 %229, %228, !dbg !11
  %dma.buffer.offset1806 = trunc i64 %230 to i32, !dbg !11
  %231 = trunc i64 %228 to i32, !dbg !11
  %232 = insertelement <4 x i32> poison, i32 %231, i64 0, !dbg !11
  %233 = lshr i64 %228, 32, !dbg !11
  %234 = trunc i64 %233 to i32, !dbg !11
  %235 = insertelement <4 x i32> %232, i32 %234, i64 1, !dbg !11
  %236 = insertelement <4 x i32> %235, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1807 = insertelement <4 x i32> %236, i32 131072, i64 3, !dbg !11
  %237 = ptrtoint ptr addrspace(3) %227 to i32, !dbg !11
  %dma.buffer.lds1808 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %237), !dbg !11
  %238 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1806, <4 x i32> %dma.buffer.rsrc1807) #10, !dbg !11
  %239 = add nuw i64 %61, 1024, !dbg !18
  %240 = icmp slt i64 %239, 0, !dbg !18
  %241 = sub nuw i64 -1025, %61, !dbg !18
  %242 = select i1 %240, i64 %241, i64 %239, !dbg !18
  %243 = sdiv i64 %242, 4096, !dbg !18
  %.lobit = ashr i64 %239, 63, !dbg !18
  %244 = xor i64 %243, %.lobit, !dbg !18
  %245 = add nsw i64 %244, %15, !dbg !18
  %246 = add nsw i64 %101, %53, !dbg !18
  %247 = add nsw i64 %246, 32, !dbg !18
  %248 = icmp slt i64 %246, -32, !dbg !18
  %249 = sub nsw i64 -33, %246, !dbg !18
  %250 = select i1 %248, i64 %249, i64 %247, !dbg !18
  %251 = sdiv i64 %250, 128, !dbg !18
  %252 = sext i1 %248 to i64, !dbg !18
  %253 = xor i64 %251, %252, !dbg !18
  %254 = shl nsw i64 %253, 7, !dbg !18
  %255 = sub i64 %53, %254, !dbg !18
  %256 = shl i64 %245, 19, !dbg !18
  %257 = and i64 %256, 16252928, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %258 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %257, !dbg !18
  %259 = getelementptr [2 x i8], ptr addrspace(1) %258, i64 %101, !dbg !18
  %260 = getelementptr [2 x i8], ptr addrspace(1) %259, i64 %255, !dbg !18
  %261 = getelementptr i8, ptr addrspace(1) %260, i64 64, !dbg !18
  %262 = getelementptr [2 x i8], ptr addrspace(1) %261, i64 %107, !dbg !18
  %263 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %264 = zext i32 %263 to i64, !dbg !18
  %265 = inttoptr i64 %264 to ptr addrspace(3), !dbg !18
  %address.local403 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %address.local293, !dbg !18
  %266 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %267 = ptrtoint ptr addrspace(1) %address.local403 to i64, !dbg !18
  %268 = sub i64 %267, %266, !dbg !18
  %dma.buffer.offset1803 = trunc i64 %268 to i32, !dbg !18
  %269 = trunc i64 %266 to i32, !dbg !18
  %270 = insertelement <4 x i32> poison, i32 %269, i64 0, !dbg !18
  %271 = lshr i64 %266, 32, !dbg !18
  %272 = trunc i64 %271 to i32, !dbg !18
  %273 = insertelement <4 x i32> %270, i32 %272, i64 1, !dbg !18
  %274 = insertelement <4 x i32> %273, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1804 = insertelement <4 x i32> %274, i32 131072, i64 3, !dbg !18
  %275 = ptrtoint ptr addrspace(3) %265 to i32, !dbg !18
  %dma.buffer.lds1805 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %275), !dbg !18
  %276 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1803, <4 x i32> %dma.buffer.rsrc1804) #10, !dbg !18
  %277 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %278 = zext i32 %277 to i64, !dbg !18
  %279 = inttoptr i64 %278 to ptr addrspace(3), !dbg !18
  %address.local404 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %address.local307, !dbg !18
  %280 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %281 = ptrtoint ptr addrspace(1) %address.local404 to i64, !dbg !18
  %282 = sub i64 %281, %280, !dbg !18
  %dma.buffer.offset1800 = trunc i64 %282 to i32, !dbg !18
  %283 = trunc i64 %280 to i32, !dbg !18
  %284 = insertelement <4 x i32> poison, i32 %283, i64 0, !dbg !18
  %285 = lshr i64 %280, 32, !dbg !18
  %286 = trunc i64 %285 to i32, !dbg !18
  %287 = insertelement <4 x i32> %284, i32 %286, i64 1, !dbg !18
  %288 = insertelement <4 x i32> %287, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1801 = insertelement <4 x i32> %288, i32 131072, i64 3, !dbg !18
  %289 = ptrtoint ptr addrspace(3) %279 to i32, !dbg !18
  %dma.buffer.lds1802 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %289), !dbg !18
  %290 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1800, <4 x i32> %dma.buffer.rsrc1801) #10, !dbg !18
  %291 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %292 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !12
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !12
  %294 = fmul <4 x half> %291, splat (half 0xH3015), !dbg !12
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !12
  %296 = fmul <4 x half> %292, splat (half 0xH3015), !dbg !12
  %297 = fmul <4 x half> %293, splat (half 0xH3015), !dbg !12
  %298 = fmul <4 x half> %295, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %299 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %299, <4 x float> %151) #9, !dbg !19
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !19
  %302 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %301, <4 x float> %300) #9, !dbg !19
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %302, <4 x float> %155) #9, !dbg !19
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !19
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !19
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %305, <4 x float> %304) #9, !dbg !19
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %306, <4 x float> %159) #9, !dbg !19
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !19
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !19
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %309, <4 x float> %308) #9, !dbg !19
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %310, <4 x float> %163) #9, !dbg !19
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !19
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !19
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %313, <4 x float> %312) #9, !dbg !19
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %314, <4 x float> %167) #9, !dbg !19
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !19
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !19
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %317, <4 x float> %316) #9, !dbg !19
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %318, <4 x float> %171) #9, !dbg !19
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !19
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !19
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %321, <4 x float> %320) #9, !dbg !19
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %322, <4 x float> %175) #9, !dbg !19
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !19
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !19
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %325, <4 x float> %324) #9, !dbg !19
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %326, <4 x float> %179) #9, !dbg !19
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !19
  %330 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %329, <4 x float> %328) #9, !dbg !19
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %330, <4 x float> %183) #9, !dbg !19
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !19
  %334 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %333, <4 x float> %332) #9, !dbg !19
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %334, <4 x float> %187) #9, !dbg !19
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !19
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !19
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %337, <4 x float> %336) #9, !dbg !19
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %338, <4 x float> %191) #9, !dbg !19
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !19
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !19
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %341, <4 x float> %340) #9, !dbg !19
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %342, <4 x float> %195) #9, !dbg !19
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !19
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !19
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %345, <4 x float> %344) #9, !dbg !19
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %346, <4 x float> %199) #9, !dbg !19
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !19
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !19
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %349, <4 x float> %348) #9, !dbg !19
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %350, <4 x float> %203) #9, !dbg !19
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !19
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !19
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %353, <4 x float> %352) #9, !dbg !19
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %354, <4 x float> %207) #9, !dbg !19
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !19
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !19
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %357, <4 x float> %356) #9, !dbg !19
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %358, <4 x float> %210) #9, !dbg !19
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !19
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %361, <4 x float> %360) #9, !dbg !19
  %363 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %364 = zext i32 %363 to i64, !dbg !11
  %365 = inttoptr i64 %364 to ptr addrspace(3), !dbg !11
  %address.local405 = getelementptr [2 x i8], ptr addrspace(1) %55, i64 %address.local267, !dbg !11
  %366 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %367 = ptrtoint ptr addrspace(1) %address.local405 to i64, !dbg !11
  %368 = sub i64 %367, %366, !dbg !11
  %dma.buffer.offset1797 = trunc i64 %368 to i32, !dbg !11
  %369 = trunc i64 %366 to i32, !dbg !11
  %370 = insertelement <4 x i32> poison, i32 %369, i64 0, !dbg !11
  %371 = lshr i64 %366, 32, !dbg !11
  %372 = trunc i64 %371 to i32, !dbg !11
  %373 = insertelement <4 x i32> %370, i32 %372, i64 1, !dbg !11
  %374 = insertelement <4 x i32> %373, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1798 = insertelement <4 x i32> %374, i32 131072, i64 3, !dbg !11
  %375 = ptrtoint ptr addrspace(3) %365 to i32, !dbg !11
  %dma.buffer.lds1799 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %375), !dbg !11
  %376 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1797, <4 x i32> %dma.buffer.rsrc1798) #10, !dbg !11
  %377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %378 = zext i32 %377 to i64, !dbg !11
  %379 = inttoptr i64 %378 to ptr addrspace(3), !dbg !11
  %address.local406 = getelementptr [2 x i8], ptr addrspace(1) %55, i64 %address.local279, !dbg !11
  %380 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %381 = ptrtoint ptr addrspace(1) %address.local406 to i64, !dbg !11
  %382 = sub i64 %381, %380, !dbg !11
  %dma.buffer.offset1794 = trunc i64 %382 to i32, !dbg !11
  %383 = trunc i64 %380 to i32, !dbg !11
  %384 = insertelement <4 x i32> poison, i32 %383, i64 0, !dbg !11
  %385 = lshr i64 %380, 32, !dbg !11
  %386 = trunc i64 %385 to i32, !dbg !11
  %387 = insertelement <4 x i32> %384, i32 %386, i64 1, !dbg !11
  %388 = insertelement <4 x i32> %387, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1795 = insertelement <4 x i32> %388, i32 131072, i64 3, !dbg !11
  %389 = ptrtoint ptr addrspace(3) %379 to i32, !dbg !11
  %dma.buffer.lds1796 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %389), !dbg !11
  %390 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1796, i32 %dma.buffer.offset1794, <4 x i32> %dma.buffer.rsrc1795) #10, !dbg !11
  %391 = add nuw i64 %61, 2048, !dbg !18
  %392 = icmp slt i64 %391, 0, !dbg !18
  %393 = sub nuw i64 -2049, %61, !dbg !18
  %394 = select i1 %392, i64 %393, i64 %391, !dbg !18
  %395 = sdiv i64 %394, 4096, !dbg !18
  %.lobit2 = ashr i64 %391, 63, !dbg !18
  %396 = xor i64 %395, %.lobit2, !dbg !18
  %397 = add nsw i64 %396, %15, !dbg !18
  %398 = add nsw i64 %246, 64, !dbg !18
  %399 = icmp slt i64 %246, -64, !dbg !18
  %400 = sub nsw i64 -65, %246, !dbg !18
  %401 = select i1 %399, i64 %400, i64 %398, !dbg !18
  %402 = sdiv i64 %401, 128, !dbg !18
  %403 = sext i1 %399 to i64, !dbg !18
  %404 = xor i64 %402, %403, !dbg !18
  %405 = shl nsw i64 %404, 7, !dbg !18
  %406 = sub i64 %53, %405, !dbg !18
  %407 = shl i64 %397, 19, !dbg !18
  %408 = and i64 %407, 16252928, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %409 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %408, !dbg !18
  %410 = getelementptr [2 x i8], ptr addrspace(1) %409, i64 %101, !dbg !18
  %411 = getelementptr [2 x i8], ptr addrspace(1) %410, i64 %406, !dbg !18
  %412 = getelementptr i8, ptr addrspace(1) %411, i64 128, !dbg !18
  %413 = getelementptr [2 x i8], ptr addrspace(1) %412, i64 %107, !dbg !18
  %414 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %415 = zext i32 %414 to i64, !dbg !18
  %416 = inttoptr i64 %415 to ptr addrspace(3), !dbg !18
  %address.local407 = getelementptr [2 x i8], ptr addrspace(1) %413, i64 %address.local293, !dbg !18
  %417 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %418 = ptrtoint ptr addrspace(1) %address.local407 to i64, !dbg !18
  %419 = sub i64 %418, %417, !dbg !18
  %dma.buffer.offset1791 = trunc i64 %419 to i32, !dbg !18
  %420 = trunc i64 %417 to i32, !dbg !18
  %421 = insertelement <4 x i32> poison, i32 %420, i64 0, !dbg !18
  %422 = lshr i64 %417, 32, !dbg !18
  %423 = trunc i64 %422 to i32, !dbg !18
  %424 = insertelement <4 x i32> %421, i32 %423, i64 1, !dbg !18
  %425 = insertelement <4 x i32> %424, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1792 = insertelement <4 x i32> %425, i32 131072, i64 3, !dbg !18
  %426 = ptrtoint ptr addrspace(3) %416 to i32, !dbg !18
  %dma.buffer.lds1793 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %426), !dbg !18
  %427 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1793, i32 %dma.buffer.offset1791, <4 x i32> %dma.buffer.rsrc1792) #10, !dbg !18
  %428 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %429 = zext i32 %428 to i64, !dbg !18
  %430 = inttoptr i64 %429 to ptr addrspace(3), !dbg !18
  %address.local408 = getelementptr [2 x i8], ptr addrspace(1) %413, i64 %address.local307, !dbg !18
  %431 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %432 = ptrtoint ptr addrspace(1) %address.local408 to i64, !dbg !18
  %433 = sub i64 %432, %431, !dbg !18
  %dma.buffer.offset1788 = trunc i64 %433 to i32, !dbg !18
  %434 = trunc i64 %431 to i32, !dbg !18
  %435 = insertelement <4 x i32> poison, i32 %434, i64 0, !dbg !18
  %436 = lshr i64 %431, 32, !dbg !18
  %437 = trunc i64 %436 to i32, !dbg !18
  %438 = insertelement <4 x i32> %435, i32 %437, i64 1, !dbg !18
  %439 = insertelement <4 x i32> %438, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1789 = insertelement <4 x i32> %439, i32 131072, i64 3, !dbg !18
  %440 = ptrtoint ptr addrspace(3) %430 to i32, !dbg !18
  %dma.buffer.lds1790 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %440), !dbg !18
  %441 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1790, i32 %dma.buffer.offset1788, <4 x i32> %dma.buffer.rsrc1789) #10, !dbg !18
  %442 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !12
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !12
  %445 = fmul <4 x half> %442, splat (half 0xH3015), !dbg !12
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %446 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !12
  %447 = fmul <4 x half> %443, splat (half 0xH3015), !dbg !12
  %448 = fmul <4 x half> %444, splat (half 0xH3015), !dbg !12
  %449 = fmul <4 x half> %446, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %450 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %450, <4 x float> %303) #9, !dbg !19
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !19
  %453 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %452, <4 x float> %451) #9, !dbg !19
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %453, <4 x float> %307) #9, !dbg !19
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !19
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !19
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %456, <4 x float> %455) #9, !dbg !19
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %457, <4 x float> %311) #9, !dbg !19
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !19
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !19
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %460, <4 x float> %459) #9, !dbg !19
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %461, <4 x float> %315) #9, !dbg !19
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !19
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !19
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %464, <4 x float> %463) #9, !dbg !19
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %465, <4 x float> %319) #9, !dbg !19
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !19
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !19
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %468, <4 x float> %467) #9, !dbg !19
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %469, <4 x float> %323) #9, !dbg !19
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !19
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !19
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %472, <4 x float> %471) #9, !dbg !19
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %473, <4 x float> %327) #9, !dbg !19
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !19
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !19
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %476, <4 x float> %475) #9, !dbg !19
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %477, <4 x float> %331) #9, !dbg !19
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !19
  %481 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %480, <4 x float> %479) #9, !dbg !19
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %481, <4 x float> %335) #9, !dbg !19
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !19
  %485 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %484, <4 x float> %483) #9, !dbg !19
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %485, <4 x float> %339) #9, !dbg !19
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !19
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !19
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %488, <4 x float> %487) #9, !dbg !19
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %489, <4 x float> %343) #9, !dbg !19
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !19
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %493 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !19
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %492, <4 x float> %491) #9, !dbg !19
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %493, <4 x float> %347) #9, !dbg !19
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !19
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !19
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %496, <4 x float> %495) #9, !dbg !19
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %497, <4 x float> %351) #9, !dbg !19
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %500 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !19
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !19
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %500, <4 x float> %499) #9, !dbg !19
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %501, <4 x float> %355) #9, !dbg !19
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %504 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !19
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %505 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !19
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %504, <4 x float> %503) #9, !dbg !19
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %505, <4 x float> %359) #9, !dbg !19
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %508 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !19
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %509 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !19
  %510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %508, <4 x float> %507) #9, !dbg !19
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %509, <4 x float> %362) #9, !dbg !19
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %512 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !19
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %512, <4 x float> %511) #9, !dbg !19
  %514 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %515 = zext i32 %514 to i64, !dbg !11
  %516 = inttoptr i64 %515 to ptr addrspace(3), !dbg !11
  %address.local409 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local267, !dbg !11
  %517 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %518 = ptrtoint ptr addrspace(1) %address.local409 to i64, !dbg !11
  %519 = sub i64 %518, %517, !dbg !11
  %dma.buffer.offset1785 = trunc i64 %519 to i32, !dbg !11
  %520 = trunc i64 %517 to i32, !dbg !11
  %521 = insertelement <4 x i32> poison, i32 %520, i64 0, !dbg !11
  %522 = lshr i64 %517, 32, !dbg !11
  %523 = trunc i64 %522 to i32, !dbg !11
  %524 = insertelement <4 x i32> %521, i32 %523, i64 1, !dbg !11
  %525 = insertelement <4 x i32> %524, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1786 = insertelement <4 x i32> %525, i32 131072, i64 3, !dbg !11
  %526 = ptrtoint ptr addrspace(3) %516 to i32, !dbg !11
  %dma.buffer.lds1787 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %526), !dbg !11
  %527 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1787, i32 %dma.buffer.offset1785, <4 x i32> %dma.buffer.rsrc1786) #10, !dbg !11
  %528 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %529 = zext i32 %528 to i64, !dbg !11
  %530 = inttoptr i64 %529 to ptr addrspace(3), !dbg !11
  %address.local410 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local279, !dbg !11
  %531 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %532 = ptrtoint ptr addrspace(1) %address.local410 to i64, !dbg !11
  %533 = sub i64 %532, %531, !dbg !11
  %dma.buffer.offset1782 = trunc i64 %533 to i32, !dbg !11
  %534 = trunc i64 %531 to i32, !dbg !11
  %535 = insertelement <4 x i32> poison, i32 %534, i64 0, !dbg !11
  %536 = lshr i64 %531, 32, !dbg !11
  %537 = trunc i64 %536 to i32, !dbg !11
  %538 = insertelement <4 x i32> %535, i32 %537, i64 1, !dbg !11
  %539 = insertelement <4 x i32> %538, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1783 = insertelement <4 x i32> %539, i32 131072, i64 3, !dbg !11
  %540 = ptrtoint ptr addrspace(3) %530 to i32, !dbg !11
  %dma.buffer.lds1784 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %540), !dbg !11
  %541 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1784, i32 %dma.buffer.offset1782, <4 x i32> %dma.buffer.rsrc1783) #10, !dbg !11
  %542 = add nuw i64 %61, 3072, !dbg !18
  %543 = icmp slt i64 %542, 0, !dbg !18
  %544 = sub nuw i64 -3073, %61, !dbg !18
  %545 = select i1 %543, i64 %544, i64 %542, !dbg !18
  %546 = sdiv i64 %545, 4096, !dbg !18
  %.lobit3 = ashr i64 %542, 63, !dbg !18
  %547 = xor i64 %546, %.lobit3, !dbg !18
  %548 = add nsw i64 %547, %15, !dbg !18
  %549 = add nsw i64 %246, 96, !dbg !18
  %550 = icmp slt i64 %246, -96, !dbg !18
  %551 = sub nsw i64 -97, %246, !dbg !18
  %552 = select i1 %550, i64 %551, i64 %549, !dbg !18
  %553 = sdiv i64 %552, 128, !dbg !18
  %554 = sext i1 %550 to i64, !dbg !18
  %555 = xor i64 %553, %554, !dbg !18
  %556 = shl nsw i64 %555, 7, !dbg !18
  %557 = sub i64 %53, %556, !dbg !18
  %558 = shl i64 %548, 19, !dbg !18
  %559 = and i64 %558, 16252928, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %560 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %559, !dbg !18
  %561 = getelementptr [2 x i8], ptr addrspace(1) %560, i64 %101, !dbg !18
  %562 = getelementptr [2 x i8], ptr addrspace(1) %561, i64 %557, !dbg !18
  %563 = getelementptr i8, ptr addrspace(1) %562, i64 192, !dbg !18
  %564 = getelementptr [2 x i8], ptr addrspace(1) %563, i64 %107, !dbg !18
  %565 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %566 = zext i32 %565 to i64, !dbg !18
  %567 = inttoptr i64 %566 to ptr addrspace(3), !dbg !18
  %address.local411 = getelementptr [2 x i8], ptr addrspace(1) %564, i64 %address.local293, !dbg !18
  %568 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %569 = ptrtoint ptr addrspace(1) %address.local411 to i64, !dbg !18
  %570 = sub i64 %569, %568, !dbg !18
  %dma.buffer.offset1779 = trunc i64 %570 to i32, !dbg !18
  %571 = trunc i64 %568 to i32, !dbg !18
  %572 = insertelement <4 x i32> poison, i32 %571, i64 0, !dbg !18
  %573 = lshr i64 %568, 32, !dbg !18
  %574 = trunc i64 %573 to i32, !dbg !18
  %575 = insertelement <4 x i32> %572, i32 %574, i64 1, !dbg !18
  %576 = insertelement <4 x i32> %575, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1780 = insertelement <4 x i32> %576, i32 131072, i64 3, !dbg !18
  %577 = ptrtoint ptr addrspace(3) %567 to i32, !dbg !18
  %dma.buffer.lds1781 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %577), !dbg !18
  %578 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1781, i32 %dma.buffer.offset1779, <4 x i32> %dma.buffer.rsrc1780) #10, !dbg !18
  %579 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %580 = zext i32 %579 to i64, !dbg !18
  %581 = inttoptr i64 %580 to ptr addrspace(3), !dbg !18
  %address.local412 = getelementptr [2 x i8], ptr addrspace(1) %564, i64 %address.local307, !dbg !18
  %582 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %583 = ptrtoint ptr addrspace(1) %address.local412 to i64, !dbg !18
  %584 = sub i64 %583, %582, !dbg !18
  %dma.buffer.offset1776 = trunc i64 %584 to i32, !dbg !18
  %585 = trunc i64 %582 to i32, !dbg !18
  %586 = insertelement <4 x i32> poison, i32 %585, i64 0, !dbg !18
  %587 = lshr i64 %582, 32, !dbg !18
  %588 = trunc i64 %587 to i32, !dbg !18
  %589 = insertelement <4 x i32> %586, i32 %588, i64 1, !dbg !18
  %590 = insertelement <4 x i32> %589, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1777 = insertelement <4 x i32> %590, i32 131072, i64 3, !dbg !18
  %591 = ptrtoint ptr addrspace(3) %581 to i32, !dbg !18
  %dma.buffer.lds1778 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %591), !dbg !18
  %592 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1778, i32 %dma.buffer.offset1776, <4 x i32> %dma.buffer.rsrc1777) #10, !dbg !18
  %593 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %594 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !12
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !12
  %596 = fmul <4 x half> %593, splat (half 0xH3015), !dbg !12
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %597 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !12
  %598 = fmul <4 x half> %594, splat (half 0xH3015), !dbg !12
  %599 = fmul <4 x half> %595, splat (half 0xH3015), !dbg !12
  %600 = fmul <4 x half> %597, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %601 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %601, <4 x float> %454) #9, !dbg !19
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %603 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !19
  %604 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %605 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %603, <4 x float> %602) #9, !dbg !19
  %606 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %604, <4 x float> %458) #9, !dbg !19
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %607 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !19
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %608 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !19
  %609 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %607, <4 x float> %606) #9, !dbg !19
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %608, <4 x float> %462) #9, !dbg !19
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %611 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !19
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %612 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !19
  %613 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %611, <4 x float> %610) #9, !dbg !19
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %612, <4 x float> %466) #9, !dbg !19
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %615 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !19
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %616 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !19
  %617 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %615, <4 x float> %614) #9, !dbg !19
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %616, <4 x float> %470) #9, !dbg !19
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %619 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !19
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %620 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !19
  %621 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %619, <4 x float> %618) #9, !dbg !19
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %620, <4 x float> %474) #9, !dbg !19
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %623 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !19
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %624 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !19
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %623, <4 x float> %622) #9, !dbg !19
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %624, <4 x float> %478) #9, !dbg !19
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %627 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !19
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %628 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !19
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %627, <4 x float> %626) #9, !dbg !19
  %630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %628, <4 x float> %482) #9, !dbg !19
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %631 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !19
  %632 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %631, <4 x float> %630) #9, !dbg !19
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %632, <4 x float> %486) #9, !dbg !19
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %635 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !19
  %636 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %635, <4 x float> %634) #9, !dbg !19
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %636, <4 x float> %490) #9, !dbg !19
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4416, !dbg !19
  %639 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !19
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !19
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %639, <4 x float> %638) #9, !dbg !19
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %640, <4 x float> %494) #9, !dbg !19
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %643 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !19
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local361, i64 832, !dbg !19
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !19
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %643, <4 x float> %642) #9, !dbg !19
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %644, <4 x float> %498) #9, !dbg !19
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4928, !dbg !19
  %647 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !19
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !19
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %647, <4 x float> %646) #9, !dbg !19
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %648, <4 x float> %502) #9, !dbg !19
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %651 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !19
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1344, !dbg !19
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !19
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %651, <4 x float> %650) #9, !dbg !19
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %652, <4 x float> %506) #9, !dbg !19
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5440, !dbg !19
  %655 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !19
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !19
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %655, <4 x float> %654) #9, !dbg !19
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %656, <4 x float> %510) #9, !dbg !19
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %659 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !19
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1856, !dbg !19
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !19
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %659, <4 x float> %658) #9, !dbg !19
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %660, <4 x float> %513) #9, !dbg !19
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5952, !dbg !19
  %663 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !19
  %664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %663, <4 x float> %662) #9, !dbg !19
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %665 = extractelement <4 x float> %664, i64 3, !dbg !24
  %666 = extractelement <4 x float> %664, i64 2, !dbg !24
  %667 = extractelement <4 x float> %664, i64 1, !dbg !24
  %668 = extractelement <4 x float> %664, i64 0, !dbg !24
  %669 = extractelement <4 x float> %661, i64 3, !dbg !24
  %670 = extractelement <4 x float> %661, i64 2, !dbg !24
  %671 = extractelement <4 x float> %661, i64 1, !dbg !24
  %672 = extractelement <4 x float> %661, i64 0, !dbg !24
  %673 = extractelement <4 x float> %657, i64 3, !dbg !24
  %674 = extractelement <4 x float> %657, i64 2, !dbg !24
  %675 = extractelement <4 x float> %657, i64 1, !dbg !24
  %676 = extractelement <4 x float> %657, i64 0, !dbg !24
  %677 = extractelement <4 x float> %653, i64 3, !dbg !24
  %678 = extractelement <4 x float> %653, i64 2, !dbg !24
  %679 = extractelement <4 x float> %653, i64 1, !dbg !24
  %680 = extractelement <4 x float> %653, i64 0, !dbg !24
  %681 = extractelement <4 x float> %649, i64 3, !dbg !24
  %682 = extractelement <4 x float> %649, i64 2, !dbg !24
  %683 = extractelement <4 x float> %649, i64 1, !dbg !24
  %684 = extractelement <4 x float> %649, i64 0, !dbg !24
  %685 = extractelement <4 x float> %645, i64 3, !dbg !24
  %686 = extractelement <4 x float> %645, i64 2, !dbg !24
  %687 = extractelement <4 x float> %645, i64 1, !dbg !24
  %688 = extractelement <4 x float> %645, i64 0, !dbg !24
  %689 = extractelement <4 x float> %641, i64 3, !dbg !24
  %690 = extractelement <4 x float> %641, i64 2, !dbg !24
  %691 = extractelement <4 x float> %641, i64 1, !dbg !24
  %692 = extractelement <4 x float> %641, i64 0, !dbg !24
  %693 = extractelement <4 x float> %637, i64 3, !dbg !24
  %694 = extractelement <4 x float> %637, i64 2, !dbg !24
  %695 = extractelement <4 x float> %637, i64 1, !dbg !24
  %696 = extractelement <4 x float> %637, i64 0, !dbg !24
  %697 = extractelement <4 x float> %633, i64 3, !dbg !24
  %698 = extractelement <4 x float> %633, i64 2, !dbg !24
  %699 = extractelement <4 x float> %633, i64 1, !dbg !24
  %700 = extractelement <4 x float> %633, i64 0, !dbg !24
  %701 = extractelement <4 x float> %629, i64 3, !dbg !24
  %702 = extractelement <4 x float> %629, i64 2, !dbg !24
  %703 = extractelement <4 x float> %629, i64 1, !dbg !24
  %704 = extractelement <4 x float> %629, i64 0, !dbg !24
  %705 = extractelement <4 x float> %625, i64 3, !dbg !24
  %706 = extractelement <4 x float> %625, i64 2, !dbg !24
  %707 = extractelement <4 x float> %625, i64 1, !dbg !24
  %708 = extractelement <4 x float> %625, i64 0, !dbg !24
  %709 = extractelement <4 x float> %621, i64 3, !dbg !24
  %710 = extractelement <4 x float> %621, i64 2, !dbg !24
  %711 = extractelement <4 x float> %621, i64 1, !dbg !24
  %712 = extractelement <4 x float> %621, i64 0, !dbg !24
  %713 = extractelement <4 x float> %617, i64 3, !dbg !24
  %714 = extractelement <4 x float> %617, i64 2, !dbg !24
  %715 = extractelement <4 x float> %617, i64 1, !dbg !24
  %716 = extractelement <4 x float> %617, i64 0, !dbg !24
  %717 = extractelement <4 x float> %613, i64 3, !dbg !24
  %718 = extractelement <4 x float> %613, i64 2, !dbg !24
  %719 = extractelement <4 x float> %613, i64 1, !dbg !24
  %720 = extractelement <4 x float> %613, i64 0, !dbg !24
  %721 = extractelement <4 x float> %609, i64 3, !dbg !24
  %722 = extractelement <4 x float> %609, i64 2, !dbg !24
  %723 = extractelement <4 x float> %609, i64 1, !dbg !24
  %724 = extractelement <4 x float> %609, i64 0, !dbg !24
  %725 = extractelement <4 x float> %605, i64 3, !dbg !24
  %726 = extractelement <4 x float> %605, i64 2, !dbg !24
  %727 = extractelement <4 x float> %605, i64 1, !dbg !24
  %728 = extractelement <4 x float> %605, i64 0, !dbg !24
  %address.thread413 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local414 = shl nuw nsw i32 %address.thread413, 5, !dbg !13
  %address.local415 = and i32 %address.local414, 6144, !dbg !13
  %address.local416 = shl nuw nsw i32 %address.thread413, 7, !dbg !13
  %address.local417 = and i32 %address.local416, 1920, !dbg !13
  %address.local418 = or disjoint i32 %address.local415, %address.local417, !dbg !13
  %address.local419 = zext nneg i32 %address.local418 to i64, !dbg !13
  %address.local420 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %729 = lshr i32 %address.thread413, 4, !dbg !12
  %address.local422 = zext nneg i32 %729 to i64, !dbg !12
  %address.local423 = and i64 %address.local422, 3, !dbg !12
  %address.local424 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local423, !dbg !13
  store float %728, ptr addrspace(3) %address.local424, align 4, !dbg !13
  %address.local425 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local426 = or disjoint i64 %address.local423, 4, !dbg !13
  %address.local427 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local425, i64 %address.local426, !dbg !13
  store float %727, ptr addrspace(3) %address.local427, align 4, !dbg !13
  %address.local428 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local429 = or disjoint i64 %address.local423, 8, !dbg !13
  %address.local430 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local428, i64 %address.local429, !dbg !13
  store float %726, ptr addrspace(3) %address.local430, align 4, !dbg !13
  %address.local431 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local432 = or i64 %address.local422, 12, !dbg !13
  %address.local433 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local431, i64 %address.local432, !dbg !13
  store float %725, ptr addrspace(3) %address.local433, align 4, !dbg !13
  %address.local434 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local435 = or disjoint i64 %address.local423, 16, !dbg !13
  %address.local436 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local434, i64 %address.local435, !dbg !13
  store float %724, ptr addrspace(3) %address.local436, align 4, !dbg !13
  %address.local437 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local438 = or disjoint i64 %address.local423, 20, !dbg !13
  %address.local439 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local437, i64 %address.local438, !dbg !13
  store float %723, ptr addrspace(3) %address.local439, align 4, !dbg !13
  %address.local440 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local441 = or disjoint i64 %address.local423, 24, !dbg !13
  %address.local442 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local440, i64 %address.local441, !dbg !13
  store float %722, ptr addrspace(3) %address.local442, align 4, !dbg !13
  %address.local443 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local444 = or i64 %address.local422, 28, !dbg !13
  %address.local445 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local443, i64 %address.local444, !dbg !13
  store float %721, ptr addrspace(3) %address.local445, align 4, !dbg !13
  %address.local446 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local447 = or disjoint i64 %address.local423, 32, !dbg !13
  %address.local448 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local446, i64 %address.local447, !dbg !13
  store float %720, ptr addrspace(3) %address.local448, align 4, !dbg !13
  %address.local449 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local450 = or disjoint i64 %address.local423, 36, !dbg !13
  %address.local451 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local449, i64 %address.local450, !dbg !13
  store float %719, ptr addrspace(3) %address.local451, align 4, !dbg !13
  %address.local452 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local453 = or disjoint i64 %address.local423, 40, !dbg !13
  %address.local454 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local452, i64 %address.local453, !dbg !13
  store float %718, ptr addrspace(3) %address.local454, align 4, !dbg !13
  %address.local455 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local456 = or i64 %address.local422, 44, !dbg !13
  %address.local457 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local455, i64 %address.local456, !dbg !13
  store float %717, ptr addrspace(3) %address.local457, align 4, !dbg !13
  %address.local458 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local459 = or disjoint i64 %address.local423, 48, !dbg !13
  %address.local460 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local458, i64 %address.local459, !dbg !13
  store float %716, ptr addrspace(3) %address.local460, align 4, !dbg !13
  %address.local461 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local462 = or disjoint i64 %address.local423, 52, !dbg !13
  %address.local463 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local461, i64 %address.local462, !dbg !13
  store float %715, ptr addrspace(3) %address.local463, align 4, !dbg !13
  %address.local464 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local465 = or disjoint i64 %address.local423, 56, !dbg !13
  %address.local466 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local464, i64 %address.local465, !dbg !13
  store float %714, ptr addrspace(3) %address.local466, align 4, !dbg !13
  %address.local467 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local468 = or i64 %address.local422, 60, !dbg !13
  %address.local469 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local467, i64 %address.local468, !dbg !13
  store float %713, ptr addrspace(3) %address.local469, align 4, !dbg !13
  %address.local470 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local471 = or disjoint i64 %address.local423, 64, !dbg !13
  %address.local472 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local470, i64 %address.local471, !dbg !13
  store float %712, ptr addrspace(3) %address.local472, align 4, !dbg !13
  %address.local473 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local474 = or disjoint i64 %address.local423, 68, !dbg !13
  %address.local475 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local473, i64 %address.local474, !dbg !13
  store float %711, ptr addrspace(3) %address.local475, align 4, !dbg !13
  %address.local476 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local477 = or disjoint i64 %address.local423, 72, !dbg !13
  %address.local478 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local476, i64 %address.local477, !dbg !13
  store float %710, ptr addrspace(3) %address.local478, align 4, !dbg !13
  %address.local479 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local480 = or i64 %address.local422, 76, !dbg !13
  %address.local481 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local479, i64 %address.local480, !dbg !13
  store float %709, ptr addrspace(3) %address.local481, align 4, !dbg !13
  %address.local482 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local483 = or disjoint i64 %address.local423, 80, !dbg !13
  %address.local484 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local482, i64 %address.local483, !dbg !13
  store float %708, ptr addrspace(3) %address.local484, align 4, !dbg !13
  %address.local485 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local486 = or disjoint i64 %address.local423, 84, !dbg !13
  %address.local487 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local485, i64 %address.local486, !dbg !13
  store float %707, ptr addrspace(3) %address.local487, align 4, !dbg !13
  %address.local488 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local489 = or disjoint i64 %address.local423, 88, !dbg !13
  %address.local490 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local488, i64 %address.local489, !dbg !13
  store float %706, ptr addrspace(3) %address.local490, align 4, !dbg !13
  %address.local491 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local492 = or i64 %address.local422, 92, !dbg !13
  %address.local493 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local491, i64 %address.local492, !dbg !13
  store float %705, ptr addrspace(3) %address.local493, align 4, !dbg !13
  %address.local494 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local495 = or disjoint i64 %address.local423, 96, !dbg !13
  %address.local496 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local494, i64 %address.local495, !dbg !13
  store float %704, ptr addrspace(3) %address.local496, align 4, !dbg !13
  %address.local497 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local498 = or disjoint i64 %address.local423, 100, !dbg !13
  %address.local499 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local497, i64 %address.local498, !dbg !13
  store float %703, ptr addrspace(3) %address.local499, align 4, !dbg !13
  %address.local500 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local501 = or disjoint i64 %address.local423, 104, !dbg !13
  %address.local502 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local500, i64 %address.local501, !dbg !13
  store float %702, ptr addrspace(3) %address.local502, align 4, !dbg !13
  %address.local503 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local504 = or i64 %address.local422, 108, !dbg !13
  %address.local505 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local503, i64 %address.local504, !dbg !13
  store float %701, ptr addrspace(3) %address.local505, align 4, !dbg !13
  %address.local506 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local507 = or disjoint i64 %address.local423, 112, !dbg !13
  %address.local508 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local506, i64 %address.local507, !dbg !13
  store float %700, ptr addrspace(3) %address.local508, align 4, !dbg !13
  %address.local509 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local510 = or disjoint i64 %address.local423, 116, !dbg !13
  %address.local511 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local509, i64 %address.local510, !dbg !13
  store float %699, ptr addrspace(3) %address.local511, align 4, !dbg !13
  %address.local512 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local513 = or disjoint i64 %address.local423, 120, !dbg !13
  %address.local514 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local512, i64 %address.local513, !dbg !13
  store float %698, ptr addrspace(3) %address.local514, align 4, !dbg !13
  %address.local515 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local516 = or i64 %address.local422, 124, !dbg !13
  %address.local517 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local515, i64 %address.local516, !dbg !13
  store float %697, ptr addrspace(3) %address.local517, align 4, !dbg !13
  %address.local518 = shl nuw nsw i32 %address.thread413, 5, !dbg !13
  %address.local519 = and i32 %address.local518, 6144, !dbg !13
  %address.local520 = shl nuw nsw i32 %address.thread413, 7, !dbg !13
  %address.local521 = and i32 %address.local520, 1920, !dbg !13
  %address.local522 = or disjoint i32 %address.local519, %address.local521, !dbg !13
  %address.local523 = or disjoint i32 %address.local522, 8192, !dbg !13
  %address.local524 = zext nneg i32 %address.local523 to i64, !dbg !13
  %address.local525 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local526 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local423, !dbg !13
  store float %696, ptr addrspace(3) %address.local526, align 4, !dbg !13
  %address.local527 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local528 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local527, i64 %address.local426, !dbg !13
  store float %695, ptr addrspace(3) %address.local528, align 4, !dbg !13
  %address.local529 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local530 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local529, i64 %address.local429, !dbg !13
  store float %694, ptr addrspace(3) %address.local530, align 4, !dbg !13
  %address.local531 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local532 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local531, i64 %address.local432, !dbg !13
  store float %693, ptr addrspace(3) %address.local532, align 4, !dbg !13
  %address.local533 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local534 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local533, i64 %address.local435, !dbg !13
  store float %692, ptr addrspace(3) %address.local534, align 4, !dbg !13
  %address.local535 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local536 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local535, i64 %address.local438, !dbg !13
  store float %691, ptr addrspace(3) %address.local536, align 4, !dbg !13
  %address.local537 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local538 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local537, i64 %address.local441, !dbg !13
  store float %690, ptr addrspace(3) %address.local538, align 4, !dbg !13
  %address.local539 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local540 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local539, i64 %address.local444, !dbg !13
  store float %689, ptr addrspace(3) %address.local540, align 4, !dbg !13
  %address.local541 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local542 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local541, i64 %address.local447, !dbg !13
  store float %688, ptr addrspace(3) %address.local542, align 4, !dbg !13
  %address.local543 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local544 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local543, i64 %address.local450, !dbg !13
  store float %687, ptr addrspace(3) %address.local544, align 4, !dbg !13
  %address.local545 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local546 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local545, i64 %address.local453, !dbg !13
  store float %686, ptr addrspace(3) %address.local546, align 4, !dbg !13
  %address.local547 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local548 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local547, i64 %address.local456, !dbg !13
  store float %685, ptr addrspace(3) %address.local548, align 4, !dbg !13
  %address.local549 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local550 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local549, i64 %address.local459, !dbg !13
  store float %684, ptr addrspace(3) %address.local550, align 4, !dbg !13
  %address.local551 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local552 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local551, i64 %address.local462, !dbg !13
  store float %683, ptr addrspace(3) %address.local552, align 4, !dbg !13
  %address.local553 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local554 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local553, i64 %address.local465, !dbg !13
  store float %682, ptr addrspace(3) %address.local554, align 4, !dbg !13
  %address.local555 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local556 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local555, i64 %address.local468, !dbg !13
  store float %681, ptr addrspace(3) %address.local556, align 4, !dbg !13
  %address.local557 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local558 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local557, i64 %address.local471, !dbg !13
  store float %680, ptr addrspace(3) %address.local558, align 4, !dbg !13
  %address.local559 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local560 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local559, i64 %address.local474, !dbg !13
  store float %679, ptr addrspace(3) %address.local560, align 4, !dbg !13
  %address.local561 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local562 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local561, i64 %address.local477, !dbg !13
  store float %678, ptr addrspace(3) %address.local562, align 4, !dbg !13
  %address.local563 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local564 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local563, i64 %address.local480, !dbg !13
  store float %677, ptr addrspace(3) %address.local564, align 4, !dbg !13
  %address.local565 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local566 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local565, i64 %address.local483, !dbg !13
  store float %676, ptr addrspace(3) %address.local566, align 4, !dbg !13
  %address.local567 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local568 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local567, i64 %address.local486, !dbg !13
  store float %675, ptr addrspace(3) %address.local568, align 4, !dbg !13
  %address.local569 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local570 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local569, i64 %address.local489, !dbg !13
  store float %674, ptr addrspace(3) %address.local570, align 4, !dbg !13
  %address.local571 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local572 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local571, i64 %address.local492, !dbg !13
  store float %673, ptr addrspace(3) %address.local572, align 4, !dbg !13
  %address.local573 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local574 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local573, i64 %address.local495, !dbg !13
  store float %672, ptr addrspace(3) %address.local574, align 4, !dbg !13
  %address.local575 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local576 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local575, i64 %address.local498, !dbg !13
  store float %671, ptr addrspace(3) %address.local576, align 4, !dbg !13
  %address.local577 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local578 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local577, i64 %address.local501, !dbg !13
  store float %670, ptr addrspace(3) %address.local578, align 4, !dbg !13
  %address.local579 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local580 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local579, i64 %address.local504, !dbg !13
  store float %669, ptr addrspace(3) %address.local580, align 4, !dbg !13
  %address.local581 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local582 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local581, i64 %address.local507, !dbg !13
  store float %668, ptr addrspace(3) %address.local582, align 4, !dbg !13
  %address.local583 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local584 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local583, i64 %address.local510, !dbg !13
  store float %667, ptr addrspace(3) %address.local584, align 4, !dbg !13
  %address.local585 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local586 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local585, i64 %address.local513, !dbg !13
  store float %666, ptr addrspace(3) %address.local586, align 4, !dbg !13
  %address.local587 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local588 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local587, i64 %address.local516, !dbg !13
  store float %665, ptr addrspace(3) %address.local588, align 4, !dbg !13
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
  %730 = lshr i32 %address.thread589, 2, !dbg !12
  %731 = and i32 %730, 12, !dbg !12
  %address.local600 = zext nneg i32 %731 to i64, !dbg !12
  %address.local601 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local596, i64 %address.local600, !dbg !13
  %732 = load float, ptr addrspace(3) %address.local601, align 16, !dbg !13
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local601, i64 4, !dbg !13
  %733 = load float, ptr addrspace(3) %lds.const.gep1682, align 4, !dbg !13
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local601, i64 8, !dbg !13
  %734 = load float, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !13
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local601, i64 12, !dbg !13
  %735 = load float, ptr addrspace(3) %lds.const.gep1684, align 4, !dbg !13
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local601, i64 64, !dbg !13
  %736 = load float, ptr addrspace(3) %lds.const.gep1685, align 16, !dbg !13
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local601, i64 68, !dbg !13
  %737 = load float, ptr addrspace(3) %lds.const.gep1686, align 4, !dbg !13
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local601, i64 72, !dbg !13
  %738 = load float, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !13
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local601, i64 76, !dbg !13
  %739 = load float, ptr addrspace(3) %lds.const.gep1688, align 4, !dbg !13
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local601, i64 128, !dbg !13
  %740 = load float, ptr addrspace(3) %lds.const.gep1689, align 16, !dbg !13
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local601, i64 132, !dbg !13
  %741 = load float, ptr addrspace(3) %lds.const.gep1690, align 4, !dbg !13
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local601, i64 136, !dbg !13
  %742 = load float, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !13
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local601, i64 140, !dbg !13
  %743 = load float, ptr addrspace(3) %lds.const.gep1692, align 4, !dbg !13
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local601, i64 192, !dbg !13
  %744 = load float, ptr addrspace(3) %lds.const.gep1693, align 16, !dbg !13
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local601, i64 196, !dbg !13
  %745 = load float, ptr addrspace(3) %lds.const.gep1694, align 4, !dbg !13
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local601, i64 200, !dbg !13
  %746 = load float, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !13
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local601, i64 204, !dbg !13
  %747 = load float, ptr addrspace(3) %lds.const.gep1696, align 4, !dbg !13
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local601, i64 256, !dbg !13
  %748 = load float, ptr addrspace(3) %lds.const.gep1697, align 16, !dbg !13
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local601, i64 260, !dbg !13
  %749 = load float, ptr addrspace(3) %lds.const.gep1698, align 4, !dbg !13
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local601, i64 264, !dbg !13
  %750 = load float, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !13
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local601, i64 268, !dbg !13
  %751 = load float, ptr addrspace(3) %lds.const.gep1700, align 4, !dbg !13
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local601, i64 320, !dbg !13
  %752 = load float, ptr addrspace(3) %lds.const.gep1701, align 16, !dbg !13
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %address.local601, i64 324, !dbg !13
  %753 = load float, ptr addrspace(3) %lds.const.gep1702, align 4, !dbg !13
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %address.local601, i64 328, !dbg !13
  %754 = load float, ptr addrspace(3) %lds.const.gep1703, align 8, !dbg !13
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local601, i64 332, !dbg !13
  %755 = load float, ptr addrspace(3) %lds.const.gep1704, align 4, !dbg !13
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local601, i64 384, !dbg !13
  %756 = load float, ptr addrspace(3) %lds.const.gep1705, align 16, !dbg !13
  %lds.const.gep1706 = getelementptr i8, ptr addrspace(3) %address.local601, i64 388, !dbg !13
  %757 = load float, ptr addrspace(3) %lds.const.gep1706, align 4, !dbg !13
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %address.local601, i64 392, !dbg !13
  %758 = load float, ptr addrspace(3) %lds.const.gep1707, align 8, !dbg !13
  %lds.const.gep1708 = getelementptr i8, ptr addrspace(3) %address.local601, i64 396, !dbg !13
  %759 = load float, ptr addrspace(3) %lds.const.gep1708, align 4, !dbg !13
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local601, i64 448, !dbg !13
  %760 = load float, ptr addrspace(3) %lds.const.gep1709, align 16, !dbg !13
  %lds.const.gep1710 = getelementptr i8, ptr addrspace(3) %address.local601, i64 452, !dbg !13
  %761 = load float, ptr addrspace(3) %lds.const.gep1710, align 4, !dbg !13
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %address.local601, i64 456, !dbg !13
  %762 = load float, ptr addrspace(3) %lds.const.gep1711, align 8, !dbg !13
  %lds.const.gep1712 = getelementptr i8, ptr addrspace(3) %address.local601, i64 460, !dbg !13
  %763 = load float, ptr addrspace(3) %lds.const.gep1712, align 4, !dbg !13
  %lds.const.gep1713 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32768, !dbg !13
  %764 = load float, ptr addrspace(3) %lds.const.gep1713, align 16, !dbg !13
  %lds.const.gep1714 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32772, !dbg !13
  %765 = load float, ptr addrspace(3) %lds.const.gep1714, align 4, !dbg !13
  %lds.const.gep1715 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32776, !dbg !13
  %766 = load float, ptr addrspace(3) %lds.const.gep1715, align 8, !dbg !13
  %lds.const.gep1716 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32780, !dbg !13
  %767 = load float, ptr addrspace(3) %lds.const.gep1716, align 4, !dbg !13
  %lds.const.gep1717 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32832, !dbg !13
  %768 = load float, ptr addrspace(3) %lds.const.gep1717, align 16, !dbg !13
  %lds.const.gep1718 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32836, !dbg !13
  %769 = load float, ptr addrspace(3) %lds.const.gep1718, align 4, !dbg !13
  %lds.const.gep1719 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32840, !dbg !13
  %770 = load float, ptr addrspace(3) %lds.const.gep1719, align 8, !dbg !13
  %lds.const.gep1720 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32844, !dbg !13
  %771 = load float, ptr addrspace(3) %lds.const.gep1720, align 4, !dbg !13
  %lds.const.gep1721 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32896, !dbg !13
  %772 = load float, ptr addrspace(3) %lds.const.gep1721, align 16, !dbg !13
  %lds.const.gep1722 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32900, !dbg !13
  %773 = load float, ptr addrspace(3) %lds.const.gep1722, align 4, !dbg !13
  %lds.const.gep1723 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32904, !dbg !13
  %774 = load float, ptr addrspace(3) %lds.const.gep1723, align 8, !dbg !13
  %lds.const.gep1724 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32908, !dbg !13
  %775 = load float, ptr addrspace(3) %lds.const.gep1724, align 4, !dbg !13
  %lds.const.gep1725 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32960, !dbg !13
  %776 = load float, ptr addrspace(3) %lds.const.gep1725, align 16, !dbg !13
  %lds.const.gep1726 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32964, !dbg !13
  %777 = load float, ptr addrspace(3) %lds.const.gep1726, align 4, !dbg !13
  %lds.const.gep1727 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32968, !dbg !13
  %778 = load float, ptr addrspace(3) %lds.const.gep1727, align 8, !dbg !13
  %lds.const.gep1728 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32972, !dbg !13
  %779 = load float, ptr addrspace(3) %lds.const.gep1728, align 4, !dbg !13
  %lds.const.gep1729 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33024, !dbg !13
  %780 = load float, ptr addrspace(3) %lds.const.gep1729, align 16, !dbg !13
  %lds.const.gep1730 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33028, !dbg !13
  %781 = load float, ptr addrspace(3) %lds.const.gep1730, align 4, !dbg !13
  %lds.const.gep1731 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33032, !dbg !13
  %782 = load float, ptr addrspace(3) %lds.const.gep1731, align 8, !dbg !13
  %lds.const.gep1732 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33036, !dbg !13
  %783 = load float, ptr addrspace(3) %lds.const.gep1732, align 4, !dbg !13
  %lds.const.gep1733 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33088, !dbg !13
  %784 = load float, ptr addrspace(3) %lds.const.gep1733, align 16, !dbg !13
  %lds.const.gep1734 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33092, !dbg !13
  %785 = load float, ptr addrspace(3) %lds.const.gep1734, align 4, !dbg !13
  %lds.const.gep1735 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33096, !dbg !13
  %786 = load float, ptr addrspace(3) %lds.const.gep1735, align 8, !dbg !13
  %lds.const.gep1736 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33100, !dbg !13
  %787 = load float, ptr addrspace(3) %lds.const.gep1736, align 4, !dbg !13
  %lds.const.gep1737 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33152, !dbg !13
  %788 = load float, ptr addrspace(3) %lds.const.gep1737, align 16, !dbg !13
  %lds.const.gep1738 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33156, !dbg !13
  %789 = load float, ptr addrspace(3) %lds.const.gep1738, align 4, !dbg !13
  %lds.const.gep1739 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33160, !dbg !13
  %790 = load float, ptr addrspace(3) %lds.const.gep1739, align 8, !dbg !13
  %lds.const.gep1740 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33164, !dbg !13
  %791 = load float, ptr addrspace(3) %lds.const.gep1740, align 4, !dbg !13
  %lds.const.gep1741 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33216, !dbg !13
  %792 = load float, ptr addrspace(3) %lds.const.gep1741, align 16, !dbg !13
  %lds.const.gep1742 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33220, !dbg !13
  %793 = load float, ptr addrspace(3) %lds.const.gep1742, align 4, !dbg !13
  %lds.const.gep1743 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33224, !dbg !13
  %794 = load float, ptr addrspace(3) %lds.const.gep1743, align 8, !dbg !13
  %lds.const.gep1744 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33228, !dbg !13
  %795 = load float, ptr addrspace(3) %lds.const.gep1744, align 4, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %796 = or disjoint i64 %70, 127, !dbg !14
  %797 = icmp uge i64 %19, %796, !dbg !14
  %798 = and i1 %42, %797, !dbg !14
  br i1 %798, label %940, label %799, !dbg !14

799:                                              ; preds = %69
  %800 = add i64 %19, %35, !dbg !14
  %801 = add i64 %800, %59, !dbg !14
  %802 = sub i64 %801, %39, !dbg !14
  %803 = or disjoint i64 %70, %38, !dbg !14
  %804 = add i64 %802, 1, !dbg !25
  %805 = lshr i64 %804, 32, !dbg !26
  %compare.high = trunc nuw i64 %805 to i32, !dbg !26
  %806 = lshr i64 %70, 32, !dbg !26
  %compare.high1303 = trunc nuw i64 %806 to i32, !dbg !26
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1303, !dbg !26
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1303, !dbg !26
  %compare.low.base = trunc i64 %804 to i32, !dbg !26
  %compare.low.base1304 = trunc i64 %803 to i32, !dbg !26
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1304, !dbg !26
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !26
  %807 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 1) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !26
  %compare.low1305 = icmp ult i32 %compare.local, %compare.low.base, !dbg !26
  %compare.result1306 = select i1 %compare.high.equal, i1 %compare.low1305, i1 %compare.high.order, !dbg !26
  %808 = select i1 %compare.result1306, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1307 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 2) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1308 = or i32 %compare.local.base1307, 2, !dbg !26
  %compare.low1309 = icmp ult i32 %compare.local1308, %compare.low.base, !dbg !26
  %compare.result1310 = select i1 %compare.high.equal, i1 %compare.low1309, i1 %compare.high.order, !dbg !26
  %809 = select i1 %compare.result1310, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1311 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 3) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1312 = or i32 %compare.local.base1311, 3, !dbg !26
  %compare.low1313 = icmp ult i32 %compare.local1312, %compare.low.base, !dbg !26
  %compare.result1314 = select i1 %compare.high.equal, i1 %compare.low1313, i1 %compare.high.order, !dbg !26
  %810 = select i1 %compare.result1314, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1315 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 4) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1316 = or i32 %compare.local.base1315, 16, !dbg !26
  %compare.low1317 = icmp ult i32 %compare.local1316, %compare.low.base, !dbg !26
  %compare.result1318 = select i1 %compare.high.equal, i1 %compare.low1317, i1 %compare.high.order, !dbg !26
  %811 = select i1 %compare.result1318, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1319 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 5) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1320 = or i32 %compare.local.base1319, 17, !dbg !26
  %compare.low1321 = icmp ult i32 %compare.local1320, %compare.low.base, !dbg !26
  %compare.result1322 = select i1 %compare.high.equal, i1 %compare.low1321, i1 %compare.high.order, !dbg !26
  %812 = select i1 %compare.result1322, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1323 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 6) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1324 = or i32 %compare.local.base1323, 18, !dbg !26
  %compare.low1325 = icmp ult i32 %compare.local1324, %compare.low.base, !dbg !26
  %compare.result1326 = select i1 %compare.high.equal, i1 %compare.low1325, i1 %compare.high.order, !dbg !26
  %813 = select i1 %compare.result1326, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 7) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1328 = or i32 %compare.local.base1327, 19, !dbg !26
  %compare.low1329 = icmp ult i32 %compare.local1328, %compare.low.base, !dbg !26
  %compare.result1330 = select i1 %compare.high.equal, i1 %compare.low1329, i1 %compare.high.order, !dbg !26
  %814 = select i1 %compare.result1330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 8) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1332 = or i32 %compare.local.base1331, 32, !dbg !26
  %compare.low1333 = icmp ult i32 %compare.local1332, %compare.low.base, !dbg !26
  %compare.result1334 = select i1 %compare.high.equal, i1 %compare.low1333, i1 %compare.high.order, !dbg !26
  %815 = select i1 %compare.result1334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 9) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1336 = or i32 %compare.local.base1335, 33, !dbg !26
  %compare.low1337 = icmp ult i32 %compare.local1336, %compare.low.base, !dbg !26
  %compare.result1338 = select i1 %compare.high.equal, i1 %compare.low1337, i1 %compare.high.order, !dbg !26
  %816 = select i1 %compare.result1338, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 10) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1340 = or i32 %compare.local.base1339, 34, !dbg !26
  %compare.low1341 = icmp ult i32 %compare.local1340, %compare.low.base, !dbg !26
  %compare.result1342 = select i1 %compare.high.equal, i1 %compare.low1341, i1 %compare.high.order, !dbg !26
  %817 = select i1 %compare.result1342, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 11) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1344 = or i32 %compare.local.base1343, 35, !dbg !26
  %compare.low1345 = icmp ult i32 %compare.local1344, %compare.low.base, !dbg !26
  %compare.result1346 = select i1 %compare.high.equal, i1 %compare.low1345, i1 %compare.high.order, !dbg !26
  %818 = select i1 %compare.result1346, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 12) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1348 = or i32 %compare.local.base1347, 48, !dbg !26
  %compare.low1349 = icmp ult i32 %compare.local1348, %compare.low.base, !dbg !26
  %compare.result1350 = select i1 %compare.high.equal, i1 %compare.low1349, i1 %compare.high.order, !dbg !26
  %819 = select i1 %compare.result1350, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 13) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1352 = or i32 %compare.local.base1351, 49, !dbg !26
  %compare.low1353 = icmp ult i32 %compare.local1352, %compare.low.base, !dbg !26
  %compare.result1354 = select i1 %compare.high.equal, i1 %compare.low1353, i1 %compare.high.order, !dbg !26
  %820 = select i1 %compare.result1354, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 14) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1356 = or i32 %compare.local.base1355, 50, !dbg !26
  %compare.low1357 = icmp ult i32 %compare.local1356, %compare.low.base, !dbg !26
  %compare.result1358 = select i1 %compare.high.equal, i1 %compare.low1357, i1 %compare.high.order, !dbg !26
  %821 = select i1 %compare.result1358, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 15) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1360 = or i32 %compare.local.base1359, 51, !dbg !26
  %compare.low1361 = icmp ult i32 %compare.local1360, %compare.low.base, !dbg !26
  %compare.result1362 = select i1 %compare.high.equal, i1 %compare.low1361, i1 %compare.high.order, !dbg !26
  %822 = select i1 %compare.result1362, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 16) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1364 = or i32 %compare.local.base1363, 64, !dbg !26
  %compare.low1365 = icmp ult i32 %compare.local1364, %compare.low.base, !dbg !26
  %compare.result1366 = select i1 %compare.high.equal, i1 %compare.low1365, i1 %compare.high.order, !dbg !26
  %823 = select i1 %compare.result1366, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 17) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1368 = or i32 %compare.local.base1367, 65, !dbg !26
  %compare.low1369 = icmp ult i32 %compare.local1368, %compare.low.base, !dbg !26
  %compare.result1370 = select i1 %compare.high.equal, i1 %compare.low1369, i1 %compare.high.order, !dbg !26
  %824 = select i1 %compare.result1370, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 18) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1372 = or i32 %compare.local.base1371, 66, !dbg !26
  %compare.low1373 = icmp ult i32 %compare.local1372, %compare.low.base, !dbg !26
  %compare.result1374 = select i1 %compare.high.equal, i1 %compare.low1373, i1 %compare.high.order, !dbg !26
  %825 = select i1 %compare.result1374, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 19) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1376 = or i32 %compare.local.base1375, 67, !dbg !26
  %compare.low1377 = icmp ult i32 %compare.local1376, %compare.low.base, !dbg !26
  %compare.result1378 = select i1 %compare.high.equal, i1 %compare.low1377, i1 %compare.high.order, !dbg !26
  %826 = select i1 %compare.result1378, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 20) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1380 = or i32 %compare.local.base1379, 80, !dbg !26
  %compare.low1381 = icmp ult i32 %compare.local1380, %compare.low.base, !dbg !26
  %compare.result1382 = select i1 %compare.high.equal, i1 %compare.low1381, i1 %compare.high.order, !dbg !26
  %827 = select i1 %compare.result1382, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 21) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1384 = or i32 %compare.local.base1383, 81, !dbg !26
  %compare.low1385 = icmp ult i32 %compare.local1384, %compare.low.base, !dbg !26
  %compare.result1386 = select i1 %compare.high.equal, i1 %compare.low1385, i1 %compare.high.order, !dbg !26
  %828 = select i1 %compare.result1386, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 22) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1388 = or i32 %compare.local.base1387, 82, !dbg !26
  %compare.low1389 = icmp ult i32 %compare.local1388, %compare.low.base, !dbg !26
  %compare.result1390 = select i1 %compare.high.equal, i1 %compare.low1389, i1 %compare.high.order, !dbg !26
  %829 = select i1 %compare.result1390, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 23) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1392 = or i32 %compare.local.base1391, 83, !dbg !26
  %compare.low1393 = icmp ult i32 %compare.local1392, %compare.low.base, !dbg !26
  %compare.result1394 = select i1 %compare.high.equal, i1 %compare.low1393, i1 %compare.high.order, !dbg !26
  %830 = select i1 %compare.result1394, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 24) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1396 = or i32 %compare.local.base1395, 96, !dbg !26
  %compare.low1397 = icmp ult i32 %compare.local1396, %compare.low.base, !dbg !26
  %compare.result1398 = select i1 %compare.high.equal, i1 %compare.low1397, i1 %compare.high.order, !dbg !26
  %831 = select i1 %compare.result1398, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 25) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1400 = or i32 %compare.local.base1399, 97, !dbg !26
  %compare.low1401 = icmp ult i32 %compare.local1400, %compare.low.base, !dbg !26
  %compare.result1402 = select i1 %compare.high.equal, i1 %compare.low1401, i1 %compare.high.order, !dbg !26
  %832 = select i1 %compare.result1402, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 26) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1404 = or i32 %compare.local.base1403, 98, !dbg !26
  %compare.low1405 = icmp ult i32 %compare.local1404, %compare.low.base, !dbg !26
  %compare.result1406 = select i1 %compare.high.equal, i1 %compare.low1405, i1 %compare.high.order, !dbg !26
  %833 = select i1 %compare.result1406, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 27) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1408 = or i32 %compare.local.base1407, 99, !dbg !26
  %compare.low1409 = icmp ult i32 %compare.local1408, %compare.low.base, !dbg !26
  %compare.result1410 = select i1 %compare.high.equal, i1 %compare.low1409, i1 %compare.high.order, !dbg !26
  %834 = select i1 %compare.result1410, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 28) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1412 = or i32 %compare.local.base1411, 112, !dbg !26
  %compare.low1413 = icmp ult i32 %compare.local1412, %compare.low.base, !dbg !26
  %compare.result1414 = select i1 %compare.high.equal, i1 %compare.low1413, i1 %compare.high.order, !dbg !26
  %835 = select i1 %compare.result1414, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 29) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1416 = or i32 %compare.local.base1415, 113, !dbg !26
  %compare.low1417 = icmp ult i32 %compare.local1416, %compare.low.base, !dbg !26
  %compare.result1418 = select i1 %compare.high.equal, i1 %compare.low1417, i1 %compare.high.order, !dbg !26
  %836 = select i1 %compare.result1418, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 30) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1420 = or i32 %compare.local.base1419, 114, !dbg !26
  %compare.low1421 = icmp ult i32 %compare.local1420, %compare.low.base, !dbg !26
  %compare.result1422 = select i1 %compare.high.equal, i1 %compare.low1421, i1 %compare.high.order, !dbg !26
  %837 = select i1 %compare.result1422, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 31) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1424 = or i32 %compare.local.base1423, 115, !dbg !26
  %compare.low1425 = icmp ult i32 %compare.local1424, %compare.low.base, !dbg !26
  %compare.result1426 = select i1 %compare.high.equal, i1 %compare.low1425, i1 %compare.high.order, !dbg !26
  %838 = select i1 %compare.result1426, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %839 = add i64 %802, 65, !dbg !25
  %840 = lshr i64 %839, 32, !dbg !26
  %compare.high1427 = trunc nuw i64 %840 to i32, !dbg !26
  %841 = lshr i64 %70, 32, !dbg !26
  %compare.high1428 = trunc nuw i64 %841 to i32, !dbg !26
  %compare.high.equal1429 = icmp eq i32 %compare.high1427, %compare.high1428, !dbg !26
  %compare.high.order1430 = icmp ugt i32 %compare.high1427, %compare.high1428, !dbg !26
  %compare.low.base1431 = trunc i64 %839 to i32, !dbg !26
  %compare.low.base1432 = trunc i64 %803 to i32, !dbg !26
  %compare.low1433 = icmp ugt i32 %compare.low.base1431, %compare.low.base1432, !dbg !26
  %compare.result1434 = select i1 %compare.high.equal1429, i1 %compare.low1433, i1 %compare.high.order1430, !dbg !26
  %842 = select i1 %compare.result1434, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 32) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1436 = or i32 %compare.local.base1435, 1, !dbg !26
  %compare.low1437 = icmp ult i32 %compare.local1436, %compare.low.base1431, !dbg !26
  %compare.result1438 = select i1 %compare.high.equal1429, i1 %compare.low1437, i1 %compare.high.order1430, !dbg !26
  %843 = select i1 %compare.result1438, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 33) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1440 = or i32 %compare.local.base1439, 2, !dbg !26
  %compare.low1441 = icmp ult i32 %compare.local1440, %compare.low.base1431, !dbg !26
  %compare.result1442 = select i1 %compare.high.equal1429, i1 %compare.low1441, i1 %compare.high.order1430, !dbg !26
  %844 = select i1 %compare.result1442, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 34) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1444 = or i32 %compare.local.base1443, 3, !dbg !26
  %compare.low1445 = icmp ult i32 %compare.local1444, %compare.low.base1431, !dbg !26
  %compare.result1446 = select i1 %compare.high.equal1429, i1 %compare.low1445, i1 %compare.high.order1430, !dbg !26
  %845 = select i1 %compare.result1446, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1447 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 35) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1448 = or i32 %compare.local.base1447, 16, !dbg !26
  %compare.low1449 = icmp ult i32 %compare.local1448, %compare.low.base1431, !dbg !26
  %compare.result1450 = select i1 %compare.high.equal1429, i1 %compare.low1449, i1 %compare.high.order1430, !dbg !26
  %846 = select i1 %compare.result1450, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1451 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 36) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1452 = or i32 %compare.local.base1451, 17, !dbg !26
  %compare.low1453 = icmp ult i32 %compare.local1452, %compare.low.base1431, !dbg !26
  %compare.result1454 = select i1 %compare.high.equal1429, i1 %compare.low1453, i1 %compare.high.order1430, !dbg !26
  %847 = select i1 %compare.result1454, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1455 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 37) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1456 = or i32 %compare.local.base1455, 18, !dbg !26
  %compare.low1457 = icmp ult i32 %compare.local1456, %compare.low.base1431, !dbg !26
  %compare.result1458 = select i1 %compare.high.equal1429, i1 %compare.low1457, i1 %compare.high.order1430, !dbg !26
  %848 = select i1 %compare.result1458, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 38) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1460 = or i32 %compare.local.base1459, 19, !dbg !26
  %compare.low1461 = icmp ult i32 %compare.local1460, %compare.low.base1431, !dbg !26
  %compare.result1462 = select i1 %compare.high.equal1429, i1 %compare.low1461, i1 %compare.high.order1430, !dbg !26
  %849 = select i1 %compare.result1462, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 39) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1464 = or i32 %compare.local.base1463, 32, !dbg !26
  %compare.low1465 = icmp ult i32 %compare.local1464, %compare.low.base1431, !dbg !26
  %compare.result1466 = select i1 %compare.high.equal1429, i1 %compare.low1465, i1 %compare.high.order1430, !dbg !26
  %850 = select i1 %compare.result1466, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 40) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1468 = or i32 %compare.local.base1467, 33, !dbg !26
  %compare.low1469 = icmp ult i32 %compare.local1468, %compare.low.base1431, !dbg !26
  %compare.result1470 = select i1 %compare.high.equal1429, i1 %compare.low1469, i1 %compare.high.order1430, !dbg !26
  %851 = select i1 %compare.result1470, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 41) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1472 = or i32 %compare.local.base1471, 34, !dbg !26
  %compare.low1473 = icmp ult i32 %compare.local1472, %compare.low.base1431, !dbg !26
  %compare.result1474 = select i1 %compare.high.equal1429, i1 %compare.low1473, i1 %compare.high.order1430, !dbg !26
  %852 = select i1 %compare.result1474, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 42) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1476 = or i32 %compare.local.base1475, 35, !dbg !26
  %compare.low1477 = icmp ult i32 %compare.local1476, %compare.low.base1431, !dbg !26
  %compare.result1478 = select i1 %compare.high.equal1429, i1 %compare.low1477, i1 %compare.high.order1430, !dbg !26
  %853 = select i1 %compare.result1478, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 43) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1480 = or i32 %compare.local.base1479, 48, !dbg !26
  %compare.low1481 = icmp ult i32 %compare.local1480, %compare.low.base1431, !dbg !26
  %compare.result1482 = select i1 %compare.high.equal1429, i1 %compare.low1481, i1 %compare.high.order1430, !dbg !26
  %854 = select i1 %compare.result1482, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 44) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1484 = or i32 %compare.local.base1483, 49, !dbg !26
  %compare.low1485 = icmp ult i32 %compare.local1484, %compare.low.base1431, !dbg !26
  %compare.result1486 = select i1 %compare.high.equal1429, i1 %compare.low1485, i1 %compare.high.order1430, !dbg !26
  %855 = select i1 %compare.result1486, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1487 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 45) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1488 = or i32 %compare.local.base1487, 50, !dbg !26
  %compare.low1489 = icmp ult i32 %compare.local1488, %compare.low.base1431, !dbg !26
  %compare.result1490 = select i1 %compare.high.equal1429, i1 %compare.low1489, i1 %compare.high.order1430, !dbg !26
  %856 = select i1 %compare.result1490, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1491 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 46) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1492 = or i32 %compare.local.base1491, 51, !dbg !26
  %compare.low1493 = icmp ult i32 %compare.local1492, %compare.low.base1431, !dbg !26
  %compare.result1494 = select i1 %compare.high.equal1429, i1 %compare.low1493, i1 %compare.high.order1430, !dbg !26
  %857 = select i1 %compare.result1494, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1495 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 47) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1496 = or i32 %compare.local.base1495, 64, !dbg !26
  %compare.low1497 = icmp ult i32 %compare.local1496, %compare.low.base1431, !dbg !26
  %compare.result1498 = select i1 %compare.high.equal1429, i1 %compare.low1497, i1 %compare.high.order1430, !dbg !26
  %858 = select i1 %compare.result1498, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1499 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 48) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1500 = or i32 %compare.local.base1499, 65, !dbg !26
  %compare.low1501 = icmp ult i32 %compare.local1500, %compare.low.base1431, !dbg !26
  %compare.result1502 = select i1 %compare.high.equal1429, i1 %compare.low1501, i1 %compare.high.order1430, !dbg !26
  %859 = select i1 %compare.result1502, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1503 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 49) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1504 = or i32 %compare.local.base1503, 66, !dbg !26
  %compare.low1505 = icmp ult i32 %compare.local1504, %compare.low.base1431, !dbg !26
  %compare.result1506 = select i1 %compare.high.equal1429, i1 %compare.low1505, i1 %compare.high.order1430, !dbg !26
  %860 = select i1 %compare.result1506, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1507 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 50) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1508 = or i32 %compare.local.base1507, 67, !dbg !26
  %compare.low1509 = icmp ult i32 %compare.local1508, %compare.low.base1431, !dbg !26
  %compare.result1510 = select i1 %compare.high.equal1429, i1 %compare.low1509, i1 %compare.high.order1430, !dbg !26
  %861 = select i1 %compare.result1510, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1511 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 51) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1512 = or i32 %compare.local.base1511, 80, !dbg !26
  %compare.low1513 = icmp ult i32 %compare.local1512, %compare.low.base1431, !dbg !26
  %compare.result1514 = select i1 %compare.high.equal1429, i1 %compare.low1513, i1 %compare.high.order1430, !dbg !26
  %862 = select i1 %compare.result1514, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1515 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 52) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1516 = or i32 %compare.local.base1515, 81, !dbg !26
  %compare.low1517 = icmp ult i32 %compare.local1516, %compare.low.base1431, !dbg !26
  %compare.result1518 = select i1 %compare.high.equal1429, i1 %compare.low1517, i1 %compare.high.order1430, !dbg !26
  %863 = select i1 %compare.result1518, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1519 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 53) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1520 = or i32 %compare.local.base1519, 82, !dbg !26
  %compare.low1521 = icmp ult i32 %compare.local1520, %compare.low.base1431, !dbg !26
  %compare.result1522 = select i1 %compare.high.equal1429, i1 %compare.low1521, i1 %compare.high.order1430, !dbg !26
  %864 = select i1 %compare.result1522, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 54) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1524 = or i32 %compare.local.base1523, 83, !dbg !26
  %compare.low1525 = icmp ult i32 %compare.local1524, %compare.low.base1431, !dbg !26
  %compare.result1526 = select i1 %compare.high.equal1429, i1 %compare.low1525, i1 %compare.high.order1430, !dbg !26
  %865 = select i1 %compare.result1526, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 55) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1528 = or i32 %compare.local.base1527, 96, !dbg !26
  %compare.low1529 = icmp ult i32 %compare.local1528, %compare.low.base1431, !dbg !26
  %compare.result1530 = select i1 %compare.high.equal1429, i1 %compare.low1529, i1 %compare.high.order1430, !dbg !26
  %866 = select i1 %compare.result1530, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 56) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1532 = or i32 %compare.local.base1531, 97, !dbg !26
  %compare.low1533 = icmp ult i32 %compare.local1532, %compare.low.base1431, !dbg !26
  %compare.result1534 = select i1 %compare.high.equal1429, i1 %compare.low1533, i1 %compare.high.order1430, !dbg !26
  %867 = select i1 %compare.result1534, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 57) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1536 = or i32 %compare.local.base1535, 98, !dbg !26
  %compare.low1537 = icmp ult i32 %compare.local1536, %compare.low.base1431, !dbg !26
  %compare.result1538 = select i1 %compare.high.equal1429, i1 %compare.low1537, i1 %compare.high.order1430, !dbg !26
  %868 = select i1 %compare.result1538, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 58) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1540 = or i32 %compare.local.base1539, 99, !dbg !26
  %compare.low1541 = icmp ult i32 %compare.local1540, %compare.low.base1431, !dbg !26
  %compare.result1542 = select i1 %compare.high.equal1429, i1 %compare.low1541, i1 %compare.high.order1430, !dbg !26
  %869 = select i1 %compare.result1542, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 59) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1544 = or i32 %compare.local.base1543, 112, !dbg !26
  %compare.low1545 = icmp ult i32 %compare.local1544, %compare.low.base1431, !dbg !26
  %compare.result1546 = select i1 %compare.high.equal1429, i1 %compare.low1545, i1 %compare.high.order1430, !dbg !26
  %870 = select i1 %compare.result1546, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 60) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1548 = or i32 %compare.local.base1547, 113, !dbg !26
  %compare.low1549 = icmp ult i32 %compare.local1548, %compare.low.base1431, !dbg !26
  %compare.result1550 = select i1 %compare.high.equal1429, i1 %compare.low1549, i1 %compare.high.order1430, !dbg !26
  %871 = select i1 %compare.result1550, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 61) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1552 = or i32 %compare.local.base1551, 114, !dbg !26
  %compare.low1553 = icmp ult i32 %compare.local1552, %compare.low.base1431, !dbg !26
  %compare.result1554 = select i1 %compare.high.equal1429, i1 %compare.low1553, i1 %compare.high.order1430, !dbg !26
  %872 = select i1 %compare.result1554, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 62) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1556 = or i32 %compare.local.base1555, 115, !dbg !26
  %compare.low1557 = icmp ult i32 %compare.local1556, %compare.low.base1431, !dbg !26
  %compare.result1558 = select i1 %compare.high.equal1429, i1 %compare.low1557, i1 %compare.high.order1430, !dbg !26
  %873 = select i1 %compare.result1558, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %874 = insertelement <32 x float> poison, float %807, i64 0, !dbg !14
  %875 = insertelement <32 x float> %874, float %808, i64 1, !dbg !14
  %876 = insertelement <32 x float> %875, float %809, i64 2, !dbg !14
  %877 = insertelement <32 x float> %876, float %810, i64 3, !dbg !14
  %878 = insertelement <32 x float> %877, float %811, i64 4, !dbg !14
  %879 = insertelement <32 x float> %878, float %812, i64 5, !dbg !14
  %880 = insertelement <32 x float> %879, float %813, i64 6, !dbg !14
  %881 = insertelement <32 x float> %880, float %814, i64 7, !dbg !14
  %882 = insertelement <32 x float> %881, float %815, i64 8, !dbg !14
  %883 = insertelement <32 x float> %882, float %816, i64 9, !dbg !14
  %884 = insertelement <32 x float> %883, float %817, i64 10, !dbg !14
  %885 = insertelement <32 x float> %884, float %818, i64 11, !dbg !14
  %886 = insertelement <32 x float> %885, float %819, i64 12, !dbg !14
  %887 = insertelement <32 x float> %886, float %820, i64 13, !dbg !14
  %888 = insertelement <32 x float> %887, float %821, i64 14, !dbg !14
  %889 = insertelement <32 x float> %888, float %822, i64 15, !dbg !14
  %890 = insertelement <32 x float> %889, float %823, i64 16, !dbg !14
  %891 = insertelement <32 x float> %890, float %824, i64 17, !dbg !14
  %892 = insertelement <32 x float> %891, float %825, i64 18, !dbg !14
  %893 = insertelement <32 x float> %892, float %826, i64 19, !dbg !14
  %894 = insertelement <32 x float> %893, float %827, i64 20, !dbg !14
  %895 = insertelement <32 x float> %894, float %828, i64 21, !dbg !14
  %896 = insertelement <32 x float> %895, float %829, i64 22, !dbg !14
  %897 = insertelement <32 x float> %896, float %830, i64 23, !dbg !14
  %898 = insertelement <32 x float> %897, float %831, i64 24, !dbg !14
  %899 = insertelement <32 x float> %898, float %832, i64 25, !dbg !14
  %900 = insertelement <32 x float> %899, float %833, i64 26, !dbg !14
  %901 = insertelement <32 x float> %900, float %834, i64 27, !dbg !14
  %902 = insertelement <32 x float> %901, float %835, i64 28, !dbg !14
  %903 = insertelement <32 x float> %902, float %836, i64 29, !dbg !14
  %904 = insertelement <32 x float> %903, float %837, i64 30, !dbg !14
  %905 = insertelement <32 x float> %904, float %838, i64 31, !dbg !14
  %906 = insertvalue [2 x <32 x float>] poison, <32 x float> %905, 0, !dbg !14
  %907 = insertelement <32 x float> poison, float %842, i64 0, !dbg !14
  %908 = insertelement <32 x float> %907, float %843, i64 1, !dbg !14
  %909 = insertelement <32 x float> %908, float %844, i64 2, !dbg !14
  %910 = insertelement <32 x float> %909, float %845, i64 3, !dbg !14
  %911 = insertelement <32 x float> %910, float %846, i64 4, !dbg !14
  %912 = insertelement <32 x float> %911, float %847, i64 5, !dbg !14
  %913 = insertelement <32 x float> %912, float %848, i64 6, !dbg !14
  %914 = insertelement <32 x float> %913, float %849, i64 7, !dbg !14
  %915 = insertelement <32 x float> %914, float %850, i64 8, !dbg !14
  %916 = insertelement <32 x float> %915, float %851, i64 9, !dbg !14
  %917 = insertelement <32 x float> %916, float %852, i64 10, !dbg !14
  %918 = insertelement <32 x float> %917, float %853, i64 11, !dbg !14
  %919 = insertelement <32 x float> %918, float %854, i64 12, !dbg !14
  %920 = insertelement <32 x float> %919, float %855, i64 13, !dbg !14
  %921 = insertelement <32 x float> %920, float %856, i64 14, !dbg !14
  %922 = insertelement <32 x float> %921, float %857, i64 15, !dbg !14
  %923 = insertelement <32 x float> %922, float %858, i64 16, !dbg !14
  %924 = insertelement <32 x float> %923, float %859, i64 17, !dbg !14
  %925 = insertelement <32 x float> %924, float %860, i64 18, !dbg !14
  %926 = insertelement <32 x float> %925, float %861, i64 19, !dbg !14
  %927 = insertelement <32 x float> %926, float %862, i64 20, !dbg !14
  %928 = insertelement <32 x float> %927, float %863, i64 21, !dbg !14
  %929 = insertelement <32 x float> %928, float %864, i64 22, !dbg !14
  %930 = insertelement <32 x float> %929, float %865, i64 23, !dbg !14
  %931 = insertelement <32 x float> %930, float %866, i64 24, !dbg !14
  %932 = insertelement <32 x float> %931, float %867, i64 25, !dbg !14
  %933 = insertelement <32 x float> %932, float %868, i64 26, !dbg !14
  %934 = insertelement <32 x float> %933, float %869, i64 27, !dbg !14
  %935 = insertelement <32 x float> %934, float %870, i64 28, !dbg !14
  %936 = insertelement <32 x float> %935, float %871, i64 29, !dbg !14
  %937 = insertelement <32 x float> %936, float %872, i64 30, !dbg !14
  %938 = insertelement <32 x float> %937, float %873, i64 31, !dbg !14
  %939 = insertvalue [2 x <32 x float>] %906, <32 x float> %938, 1, !dbg !14
  br label %940, !dbg !14

940:                                              ; preds = %799, %69
  %941 = phi [2 x <32 x float>] [ %939, %799 ], [ zeroinitializer, %69 ], !dbg !14
  %942 = extractvalue [2 x <32 x float>] %941, 0, !dbg !28
  %943 = insertelement <4 x float> poison, float %732, i64 0, !dbg !28
  %944 = insertelement <4 x float> %943, float %733, i64 1, !dbg !28
  %945 = insertelement <4 x float> %944, float %734, i64 2, !dbg !28
  %946 = insertelement <4 x float> %945, float %735, i64 3, !dbg !28
  %947 = shufflevector <32 x float> %942, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %948 = fadd <4 x float> %946, %947, !dbg !28
  %949 = extractelement <4 x float> %948, i64 0, !dbg !28
  %950 = extractelement <4 x float> %948, i64 1, !dbg !28
  %951 = extractelement <4 x float> %948, i64 2, !dbg !28
  %952 = extractelement <4 x float> %948, i64 3, !dbg !28
  %953 = insertelement <4 x float> poison, float %736, i64 0, !dbg !28
  %954 = insertelement <4 x float> %953, float %737, i64 1, !dbg !28
  %955 = insertelement <4 x float> %954, float %738, i64 2, !dbg !28
  %956 = insertelement <4 x float> %955, float %739, i64 3, !dbg !28
  %957 = shufflevector <32 x float> %942, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %958 = fadd <4 x float> %956, %957, !dbg !28
  %959 = extractelement <4 x float> %958, i64 0, !dbg !28
  %960 = extractelement <4 x float> %958, i64 1, !dbg !28
  %961 = extractelement <4 x float> %958, i64 2, !dbg !28
  %962 = extractelement <4 x float> %958, i64 3, !dbg !28
  %963 = insertelement <4 x float> poison, float %740, i64 0, !dbg !28
  %964 = insertelement <4 x float> %963, float %741, i64 1, !dbg !28
  %965 = insertelement <4 x float> %964, float %742, i64 2, !dbg !28
  %966 = insertelement <4 x float> %965, float %743, i64 3, !dbg !28
  %967 = shufflevector <32 x float> %942, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !28
  %968 = fadd <4 x float> %966, %967, !dbg !28
  %969 = extractelement <4 x float> %968, i64 0, !dbg !28
  %970 = extractelement <4 x float> %968, i64 1, !dbg !28
  %971 = extractelement <4 x float> %968, i64 2, !dbg !28
  %972 = extractelement <4 x float> %968, i64 3, !dbg !28
  %973 = insertelement <4 x float> poison, float %744, i64 0, !dbg !28
  %974 = insertelement <4 x float> %973, float %745, i64 1, !dbg !28
  %975 = insertelement <4 x float> %974, float %746, i64 2, !dbg !28
  %976 = insertelement <4 x float> %975, float %747, i64 3, !dbg !28
  %977 = shufflevector <32 x float> %942, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !28
  %978 = fadd <4 x float> %976, %977, !dbg !28
  %979 = extractelement <4 x float> %978, i64 0, !dbg !28
  %980 = extractelement <4 x float> %978, i64 1, !dbg !28
  %981 = extractelement <4 x float> %978, i64 2, !dbg !28
  %982 = extractelement <4 x float> %978, i64 3, !dbg !28
  %983 = insertelement <4 x float> poison, float %748, i64 0, !dbg !28
  %984 = insertelement <4 x float> %983, float %749, i64 1, !dbg !28
  %985 = insertelement <4 x float> %984, float %750, i64 2, !dbg !28
  %986 = insertelement <4 x float> %985, float %751, i64 3, !dbg !28
  %987 = shufflevector <32 x float> %942, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !28
  %988 = fadd <4 x float> %986, %987, !dbg !28
  %989 = extractelement <4 x float> %988, i64 0, !dbg !28
  %990 = extractelement <4 x float> %988, i64 1, !dbg !28
  %991 = extractelement <4 x float> %988, i64 2, !dbg !28
  %992 = extractelement <4 x float> %988, i64 3, !dbg !28
  %993 = insertelement <4 x float> poison, float %752, i64 0, !dbg !28
  %994 = insertelement <4 x float> %993, float %753, i64 1, !dbg !28
  %995 = insertelement <4 x float> %994, float %754, i64 2, !dbg !28
  %996 = insertelement <4 x float> %995, float %755, i64 3, !dbg !28
  %997 = shufflevector <32 x float> %942, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !28
  %998 = fadd <4 x float> %996, %997, !dbg !28
  %999 = extractelement <4 x float> %998, i64 0, !dbg !28
  %1000 = extractelement <4 x float> %998, i64 1, !dbg !28
  %1001 = extractelement <4 x float> %998, i64 2, !dbg !28
  %1002 = extractelement <4 x float> %998, i64 3, !dbg !28
  %1003 = insertelement <4 x float> poison, float %756, i64 0, !dbg !28
  %1004 = insertelement <4 x float> %1003, float %757, i64 1, !dbg !28
  %1005 = insertelement <4 x float> %1004, float %758, i64 2, !dbg !28
  %1006 = insertelement <4 x float> %1005, float %759, i64 3, !dbg !28
  %1007 = shufflevector <32 x float> %942, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !28
  %1008 = fadd <4 x float> %1006, %1007, !dbg !28
  %1009 = extractelement <4 x float> %1008, i64 0, !dbg !28
  %1010 = extractelement <4 x float> %1008, i64 1, !dbg !28
  %1011 = extractelement <4 x float> %1008, i64 2, !dbg !28
  %1012 = extractelement <4 x float> %1008, i64 3, !dbg !28
  %1013 = insertelement <4 x float> poison, float %760, i64 0, !dbg !28
  %1014 = insertelement <4 x float> %1013, float %761, i64 1, !dbg !28
  %1015 = insertelement <4 x float> %1014, float %762, i64 2, !dbg !28
  %1016 = insertelement <4 x float> %1015, float %763, i64 3, !dbg !28
  %1017 = shufflevector <32 x float> %942, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !28
  %1018 = fadd <4 x float> %1016, %1017, !dbg !28
  %1019 = extractelement <4 x float> %1018, i64 0, !dbg !28
  %1020 = extractelement <4 x float> %1018, i64 1, !dbg !28
  %1021 = extractelement <4 x float> %1018, i64 2, !dbg !28
  %1022 = extractelement <4 x float> %1018, i64 3, !dbg !28
  %1023 = extractvalue [2 x <32 x float>] %941, 1, !dbg !28
  %1024 = insertelement <4 x float> poison, float %764, i64 0, !dbg !28
  %1025 = insertelement <4 x float> %1024, float %765, i64 1, !dbg !28
  %1026 = insertelement <4 x float> %1025, float %766, i64 2, !dbg !28
  %1027 = insertelement <4 x float> %1026, float %767, i64 3, !dbg !28
  %1028 = shufflevector <32 x float> %1023, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1029 = fadd <4 x float> %1027, %1028, !dbg !28
  %1030 = extractelement <4 x float> %1029, i64 0, !dbg !28
  %1031 = extractelement <4 x float> %1029, i64 1, !dbg !28
  %1032 = extractelement <4 x float> %1029, i64 2, !dbg !28
  %1033 = extractelement <4 x float> %1029, i64 3, !dbg !28
  %1034 = insertelement <4 x float> poison, float %768, i64 0, !dbg !28
  %1035 = insertelement <4 x float> %1034, float %769, i64 1, !dbg !28
  %1036 = insertelement <4 x float> %1035, float %770, i64 2, !dbg !28
  %1037 = insertelement <4 x float> %1036, float %771, i64 3, !dbg !28
  %1038 = shufflevector <32 x float> %1023, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1039 = fadd <4 x float> %1037, %1038, !dbg !28
  %1040 = extractelement <4 x float> %1039, i64 0, !dbg !28
  %1041 = extractelement <4 x float> %1039, i64 1, !dbg !28
  %1042 = extractelement <4 x float> %1039, i64 2, !dbg !28
  %1043 = extractelement <4 x float> %1039, i64 3, !dbg !28
  %1044 = insertelement <4 x float> poison, float %772, i64 0, !dbg !28
  %1045 = insertelement <4 x float> %1044, float %773, i64 1, !dbg !28
  %1046 = insertelement <4 x float> %1045, float %774, i64 2, !dbg !28
  %1047 = insertelement <4 x float> %1046, float %775, i64 3, !dbg !28
  %1048 = shufflevector <32 x float> %1023, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !28
  %1049 = fadd <4 x float> %1047, %1048, !dbg !28
  %1050 = extractelement <4 x float> %1049, i64 0, !dbg !28
  %1051 = extractelement <4 x float> %1049, i64 1, !dbg !28
  %1052 = extractelement <4 x float> %1049, i64 2, !dbg !28
  %1053 = extractelement <4 x float> %1049, i64 3, !dbg !28
  %1054 = insertelement <4 x float> poison, float %776, i64 0, !dbg !28
  %1055 = insertelement <4 x float> %1054, float %777, i64 1, !dbg !28
  %1056 = insertelement <4 x float> %1055, float %778, i64 2, !dbg !28
  %1057 = insertelement <4 x float> %1056, float %779, i64 3, !dbg !28
  %1058 = shufflevector <32 x float> %1023, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !28
  %1059 = fadd <4 x float> %1057, %1058, !dbg !28
  %1060 = extractelement <4 x float> %1059, i64 0, !dbg !28
  %1061 = extractelement <4 x float> %1059, i64 1, !dbg !28
  %1062 = extractelement <4 x float> %1059, i64 2, !dbg !28
  %1063 = extractelement <4 x float> %1059, i64 3, !dbg !28
  %1064 = insertelement <4 x float> poison, float %780, i64 0, !dbg !28
  %1065 = insertelement <4 x float> %1064, float %781, i64 1, !dbg !28
  %1066 = insertelement <4 x float> %1065, float %782, i64 2, !dbg !28
  %1067 = insertelement <4 x float> %1066, float %783, i64 3, !dbg !28
  %1068 = shufflevector <32 x float> %1023, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !28
  %1069 = fadd <4 x float> %1067, %1068, !dbg !28
  %1070 = extractelement <4 x float> %1069, i64 0, !dbg !28
  %1071 = extractelement <4 x float> %1069, i64 1, !dbg !28
  %1072 = extractelement <4 x float> %1069, i64 2, !dbg !28
  %1073 = extractelement <4 x float> %1069, i64 3, !dbg !28
  %1074 = insertelement <4 x float> poison, float %784, i64 0, !dbg !28
  %1075 = insertelement <4 x float> %1074, float %785, i64 1, !dbg !28
  %1076 = insertelement <4 x float> %1075, float %786, i64 2, !dbg !28
  %1077 = insertelement <4 x float> %1076, float %787, i64 3, !dbg !28
  %1078 = shufflevector <32 x float> %1023, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !28
  %1079 = fadd <4 x float> %1077, %1078, !dbg !28
  %1080 = extractelement <4 x float> %1079, i64 0, !dbg !28
  %1081 = extractelement <4 x float> %1079, i64 1, !dbg !28
  %1082 = extractelement <4 x float> %1079, i64 2, !dbg !28
  %1083 = extractelement <4 x float> %1079, i64 3, !dbg !28
  %1084 = insertelement <4 x float> poison, float %788, i64 0, !dbg !28
  %1085 = insertelement <4 x float> %1084, float %789, i64 1, !dbg !28
  %1086 = insertelement <4 x float> %1085, float %790, i64 2, !dbg !28
  %1087 = insertelement <4 x float> %1086, float %791, i64 3, !dbg !28
  %1088 = shufflevector <32 x float> %1023, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !28
  %1089 = fadd <4 x float> %1087, %1088, !dbg !28
  %1090 = extractelement <4 x float> %1089, i64 0, !dbg !28
  %1091 = extractelement <4 x float> %1089, i64 1, !dbg !28
  %1092 = extractelement <4 x float> %1089, i64 2, !dbg !28
  %1093 = extractelement <4 x float> %1089, i64 3, !dbg !28
  %1094 = insertelement <4 x float> poison, float %792, i64 0, !dbg !28
  %1095 = insertelement <4 x float> %1094, float %793, i64 1, !dbg !28
  %1096 = insertelement <4 x float> %1095, float %794, i64 2, !dbg !28
  %1097 = insertelement <4 x float> %1096, float %795, i64 3, !dbg !28
  %1098 = shufflevector <32 x float> %1023, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !28
  %1099 = fadd <4 x float> %1097, %1098, !dbg !28
  %1100 = extractelement <4 x float> %1099, i64 0, !dbg !28
  %1101 = extractelement <4 x float> %1099, i64 1, !dbg !28
  %1102 = extractelement <4 x float> %1099, i64 2, !dbg !28
  %1103 = extractelement <4 x float> %1099, i64 3, !dbg !28
  %1104 = call float @llvm.maxnum.f32(float %949, float 0xFFF0000000000000), !dbg !29, !frisk.local_reduction !1
  %local.reduce182 = call float @llvm.maxnum.f32(float %1104, float %959), !dbg !29, !frisk.local_reduction !1
  %local.reduce183 = call float @llvm.maxnum.f32(float %950, float %960), !dbg !29, !frisk.local_reduction !1
  %local.reduce184 = call float @llvm.maxnum.f32(float %951, float %961), !dbg !29, !frisk.local_reduction !1
  %local.reduce185 = call float @llvm.maxnum.f32(float %952, float %962), !dbg !29, !frisk.local_reduction !1
  %local.reduce186 = call float @llvm.maxnum.f32(float %local.reduce182, float %969), !dbg !29, !frisk.local_reduction !1
  %local.reduce187 = call float @llvm.maxnum.f32(float %local.reduce183, float %970), !dbg !29, !frisk.local_reduction !1
  %local.reduce188 = call float @llvm.maxnum.f32(float %local.reduce184, float %971), !dbg !29, !frisk.local_reduction !1
  %local.reduce189 = call float @llvm.maxnum.f32(float %local.reduce185, float %972), !dbg !29, !frisk.local_reduction !1
  %local.reduce190 = call float @llvm.maxnum.f32(float %local.reduce186, float %979), !dbg !29, !frisk.local_reduction !1
  %local.reduce191 = call float @llvm.maxnum.f32(float %local.reduce187, float %980), !dbg !29, !frisk.local_reduction !1
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %981), !dbg !29, !frisk.local_reduction !1
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %982), !dbg !29, !frisk.local_reduction !1
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %989), !dbg !29, !frisk.local_reduction !1
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %990), !dbg !29, !frisk.local_reduction !1
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %991), !dbg !29, !frisk.local_reduction !1
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %992), !dbg !29, !frisk.local_reduction !1
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %999), !dbg !29, !frisk.local_reduction !1
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %1000), !dbg !29, !frisk.local_reduction !1
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %1001), !dbg !29, !frisk.local_reduction !1
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %1002), !dbg !29, !frisk.local_reduction !1
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %1009), !dbg !29, !frisk.local_reduction !1
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %1010), !dbg !29, !frisk.local_reduction !1
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %1011), !dbg !29, !frisk.local_reduction !1
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %1012), !dbg !29, !frisk.local_reduction !1
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %1019), !dbg !29, !frisk.local_reduction !1
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %1020), !dbg !29, !frisk.local_reduction !1
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %1021), !dbg !29, !frisk.local_reduction !1
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %1022), !dbg !29, !frisk.local_reduction !1
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %local.reduce207), !dbg !29, !frisk.local_reduction !1
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce208, float %local.reduce209), !dbg !29, !frisk.local_reduction !1
  %1105 = call float @llvm.maxnum.f32(float %local.reduce210, float %local.reduce211), !dbg !29, !frisk.local_reduction !1
  %1106 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !29
  %1107 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1106), !dbg !29
  %1108 = add nuw nsw i32 %1107, 64, !dbg !29
  %1109 = xor i32 %1107, 16, !dbg !29
  %1110 = and i32 %1108, 192, !dbg !29
  %1111 = icmp samesign ult i32 %1109, %1110, !dbg !29
  %1112 = select i1 %1111, i32 %1109, i32 %1107, !dbg !29
  %1113 = shl nuw nsw i32 %1112, 2, !dbg !29
  %1114 = bitcast float %1105 to i32, !dbg !29
  %1115 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1114), !dbg !29
  %1116 = bitcast i32 %1115 to float, !dbg !29
  %1117 = call float @llvm.maxnum.f32(float %1105, float %1116), !dbg !29
  %1118 = xor i32 %1107, 32, !dbg !29
  %1119 = icmp samesign ult i32 %1118, %1110, !dbg !29
  %1120 = select i1 %1119, i32 %1118, i32 %1107, !dbg !29
  %1121 = shl nuw nsw i32 %1120, 2, !dbg !29
  %1122 = bitcast float %1117 to i32, !dbg !29
  %1123 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1122), !dbg !29
  %1124 = bitcast i32 %1123 to float, !dbg !29
  %1125 = call float @llvm.maxnum.f32(float %1117, float %1124), !dbg !29
  %1126 = extractvalue [2 x <1 x float>] %64, 0, !dbg !29
  %1127 = insertelement <1 x float> poison, float %1125, i64 0, !dbg !29
  %1128 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1127, <1 x float> %1126), !dbg !29
  %1129 = call float @llvm.maxnum.f32(float %1030, float 0xFFF0000000000000), !dbg !29, !frisk.local_reduction !1
  %local.reduce151 = call float @llvm.maxnum.f32(float %1129, float %1040), !dbg !29, !frisk.local_reduction !1
  %local.reduce152 = call float @llvm.maxnum.f32(float %1031, float %1041), !dbg !29, !frisk.local_reduction !1
  %local.reduce153 = call float @llvm.maxnum.f32(float %1032, float %1042), !dbg !29, !frisk.local_reduction !1
  %local.reduce154 = call float @llvm.maxnum.f32(float %1033, float %1043), !dbg !29, !frisk.local_reduction !1
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %1050), !dbg !29, !frisk.local_reduction !1
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %1051), !dbg !29, !frisk.local_reduction !1
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %1052), !dbg !29, !frisk.local_reduction !1
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %1053), !dbg !29, !frisk.local_reduction !1
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %1060), !dbg !29, !frisk.local_reduction !1
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %1061), !dbg !29, !frisk.local_reduction !1
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %1062), !dbg !29, !frisk.local_reduction !1
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %1063), !dbg !29, !frisk.local_reduction !1
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %1070), !dbg !29, !frisk.local_reduction !1
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %1071), !dbg !29, !frisk.local_reduction !1
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %1072), !dbg !29, !frisk.local_reduction !1
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %1073), !dbg !29, !frisk.local_reduction !1
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %1080), !dbg !29, !frisk.local_reduction !1
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %1081), !dbg !29, !frisk.local_reduction !1
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %1082), !dbg !29, !frisk.local_reduction !1
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %1083), !dbg !29, !frisk.local_reduction !1
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %1090), !dbg !29, !frisk.local_reduction !1
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %1091), !dbg !29, !frisk.local_reduction !1
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %1092), !dbg !29, !frisk.local_reduction !1
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %1093), !dbg !29, !frisk.local_reduction !1
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %1100), !dbg !29, !frisk.local_reduction !1
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %1101), !dbg !29, !frisk.local_reduction !1
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %1102), !dbg !29, !frisk.local_reduction !1
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %1103), !dbg !29, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !29, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !29, !frisk.local_reduction !1
  %1130 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !29, !frisk.local_reduction !1
  %1131 = bitcast float %1130 to i32, !dbg !29
  %1132 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1131), !dbg !29
  %1133 = bitcast i32 %1132 to float, !dbg !29
  %1134 = call float @llvm.maxnum.f32(float %1130, float %1133), !dbg !29
  %1135 = bitcast float %1134 to i32, !dbg !29
  %1136 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1135), !dbg !29
  %1137 = bitcast i32 %1136 to float, !dbg !29
  %1138 = call float @llvm.maxnum.f32(float %1134, float %1137), !dbg !29
  %1139 = extractvalue [2 x <1 x float>] %64, 1, !dbg !29
  %1140 = insertelement <1 x float> poison, float %1138, i64 0, !dbg !29
  %1141 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1140, <1 x float> %1139), !dbg !29
  %1142 = shufflevector <1 x float> %1128, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %1143 = fsub <4 x float> %948, %1142, !dbg !30
  %1144 = fsub <4 x float> %958, %1142, !dbg !30
  %1145 = fsub <4 x float> %968, %1142, !dbg !30
  %1146 = fsub <4 x float> %978, %1142, !dbg !30
  %1147 = fsub <4 x float> %988, %1142, !dbg !30
  %1148 = fsub <4 x float> %998, %1142, !dbg !30
  %1149 = fsub <4 x float> %1008, %1142, !dbg !30
  %1150 = fsub <4 x float> %1018, %1142, !dbg !30
  %1151 = shufflevector <1 x float> %1141, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %1152 = fsub <4 x float> %1029, %1151, !dbg !30
  %1153 = fsub <4 x float> %1039, %1151, !dbg !30
  %1154 = fsub <4 x float> %1049, %1151, !dbg !30
  %1155 = fsub <4 x float> %1059, %1151, !dbg !30
  %1156 = fsub <4 x float> %1069, %1151, !dbg !30
  %1157 = fsub <4 x float> %1079, %1151, !dbg !30
  %1158 = fsub <4 x float> %1089, %1151, !dbg !30
  %1159 = fsub <4 x float> %1099, %1151, !dbg !30
  br i1 %43, label %1160, label %1167, !dbg !15

1160:                                             ; preds = %940
  %1161 = fsub <1 x float> %1139, %1141, !dbg !31
  %1162 = extractelement <1 x float> %1161, i64 0, !dbg !31
  %1163 = fsub <1 x float> %1126, %1128, !dbg !31
  %1164 = extractelement <1 x float> %1163, i64 0, !dbg !31
  %address.thread766 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !40, !frisk.register_lifetimes !1
  %address.local767 = lshr i32 %address.thread766, 2, !dbg !13
  %address.local768 = and i32 %address.local767, 48, !dbg !13
  %address.local769 = and i32 %address.thread766, 15, !dbg !13
  %address.local770 = or disjoint i32 %address.local768, %address.local769, !dbg !13
  %address.local771 = zext nneg i32 %address.local770 to i64, !dbg !13
  %address.local772 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local771, !dbg !15
  store float %1164, ptr addrspace(3) %address.local772, align 4, !dbg !15
  %address.local773 = lshr i32 %address.thread766, 2, !dbg !13
  %address.local774 = and i32 %address.local773, 48, !dbg !13
  %address.local775 = and i32 %address.thread766, 15, !dbg !13
  %address.local776 = or disjoint i32 %address.local774, %address.local775, !dbg !13
  %1165 = zext nneg i32 %address.local776 to i64, !dbg !15
  %1166 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %1165, !dbg !15
  %address.local779 = getelementptr inbounds nuw i8, ptr addrspace(3) %1166, i64 256, !dbg !15
  store float %1162, ptr addrspace(3) %address.local779, align 4, !dbg !15
  br label %1167, !dbg !15

1167:                                             ; preds = %1160, %940
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
  %1168 = load float, ptr addrspace(3) %address.local786, align 4, !dbg !15
  %lds.const.gep1745 = getelementptr i8, ptr addrspace(3) %address.local786, i64 64, !dbg !15
  %1169 = load float, ptr addrspace(3) %lds.const.gep1745, align 4, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1170 = call float @llvm.amdgcn.exp2.f32(float %1168), !dbg !15
  %1171 = call float @llvm.amdgcn.exp2.f32(float %1169), !dbg !15
  br i1 %43, label %1172, label %1173, !dbg !32

1172:                                             ; preds = %1167
  %address.thread789 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !32, !range !40, !frisk.register_lifetimes !1
  %address.local790 = lshr i32 %address.thread789, 1, !dbg !15
  %address.local791 = and i32 %address.local790, 96, !dbg !15
  %address.local792 = and i32 %address.thread789, 15, !dbg !15
  %address.local793 = or disjoint i32 %address.local791, %address.local792, !dbg !15
  %address.local794 = zext nneg i32 %address.local793 to i64, !dbg !15
  %address.local795 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !15
  store float %1170, ptr addrspace(3) %address.local795, align 4, !dbg !32
  %address.local796 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !15
  %address.local797 = getelementptr i8, ptr addrspace(3) %address.local796, i64 64, !dbg !15
  store float %1171, ptr addrspace(3) %address.local797, align 4, !dbg !32
  br label %1173, !dbg !32

1173:                                             ; preds = %1172, %1167
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
  %1174 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %lds.const.gep1746 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !32
  %1175 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  fence syncscope("workgroup") release, !dbg !32
  call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %1176 = add nuw nsw i64 %101, %15, !dbg !20
  %1177 = shl i64 %1176, 19, !dbg !20
  %1178 = and i64 %1177, 16252928, !dbg !20
  %1179 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1178, !dbg !20
  %1180 = getelementptr [2 x i8], ptr addrspace(1) %1179, i64 %107, !dbg !20
  %address.local812 = zext nneg i32 %address.thread798 to i64, !dbg !12
  %address.local813 = shl nuw nsw i64 %address.local812, 3, !dbg !11
  %address.local814 = and i64 %address.local813, 1536, !dbg !11
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local814, !dbg !11
  %address.local816 = ptrtoint ptr addrspace(3) %address.local815 to i64, !dbg !11
  %address.local817 = trunc i64 %address.local816 to i32, !dbg !11
  %1181 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local817), !dbg !20
  %1182 = zext i32 %1181 to i64, !dbg !20
  %1183 = inttoptr i64 %1182 to ptr addrspace(3), !dbg !20
  %address.local818 = shl nuw nsw i64 %address.local812, 5, !dbg !20
  %address.local819 = and i64 %address.local818, 8064, !dbg !20
  %address.local820 = shl nuw nsw i64 %address.local812, 3, !dbg !20
  %address.local821 = and i64 %address.local820, 24, !dbg !20
  %1184 = getelementptr [2 x i8], ptr addrspace(1) %1180, i64 %address.local819, !dbg !20
  %address.local823 = getelementptr [2 x i8], ptr addrspace(1) %1184, i64 %address.local821, !dbg !20
  %1185 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1186 = ptrtoint ptr addrspace(1) %address.local823 to i64, !dbg !20
  %1187 = sub i64 %1186, %1185, !dbg !20
  %dma.buffer.offset1773 = trunc i64 %1187 to i32, !dbg !20
  %1188 = trunc i64 %1185 to i32, !dbg !20
  %1189 = insertelement <4 x i32> poison, i32 %1188, i64 0, !dbg !20
  %1190 = lshr i64 %1185, 32, !dbg !20
  %1191 = trunc i64 %1190 to i32, !dbg !20
  %1192 = insertelement <4 x i32> %1189, i32 %1191, i64 1, !dbg !20
  %1193 = insertelement <4 x i32> %1192, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1774 = insertelement <4 x i32> %1193, i32 131072, i64 3, !dbg !20
  %1194 = ptrtoint ptr addrspace(3) %1183 to i32, !dbg !20
  %dma.buffer.lds1775 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1194), !dbg !20
  %1195 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1775, i32 %dma.buffer.offset1773, <4 x i32> %dma.buffer.rsrc1774) #10, !dbg !20
  %address.local824 = or disjoint i64 %address.local812, 256, !dbg !11
  %address.local825 = shl nuw nsw i64 %address.local824, 3, !dbg !11
  %address.local826 = and i64 %address.local825, 3584, !dbg !11
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local826, !dbg !11
  %address.local828 = ptrtoint ptr addrspace(3) %address.local827 to i64, !dbg !11
  %address.local829 = trunc i64 %address.local828 to i32, !dbg !11
  %1196 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local829), !dbg !20
  %1197 = zext i32 %1196 to i64, !dbg !20
  %1198 = inttoptr i64 %1197 to ptr addrspace(3), !dbg !20
  %address.local830 = shl nuw nsw i64 %address.local824, 5, !dbg !20
  %address.local831 = and i64 %address.local830, 16256, !dbg !20
  %address.local832 = shl nuw nsw i64 %address.local812, 3, !dbg !20
  %address.local833 = and i64 %address.local832, 24, !dbg !20
  %1199 = getelementptr [2 x i8], ptr addrspace(1) %1180, i64 %address.local831, !dbg !20
  %address.local835 = getelementptr [2 x i8], ptr addrspace(1) %1199, i64 %address.local833, !dbg !20
  %1200 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1201 = ptrtoint ptr addrspace(1) %address.local835 to i64, !dbg !20
  %1202 = sub i64 %1201, %1200, !dbg !20
  %dma.buffer.offset1770 = trunc i64 %1202 to i32, !dbg !20
  %1203 = trunc i64 %1200 to i32, !dbg !20
  %1204 = insertelement <4 x i32> poison, i32 %1203, i64 0, !dbg !20
  %1205 = lshr i64 %1200, 32, !dbg !20
  %1206 = trunc i64 %1205 to i32, !dbg !20
  %1207 = insertelement <4 x i32> %1204, i32 %1206, i64 1, !dbg !20
  %1208 = insertelement <4 x i32> %1207, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1771 = insertelement <4 x i32> %1208, i32 131072, i64 3, !dbg !20
  %1209 = ptrtoint ptr addrspace(3) %1198 to i32, !dbg !20
  %dma.buffer.lds1772 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1209), !dbg !20
  %1210 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1772, i32 %dma.buffer.offset1770, <4 x i32> %dma.buffer.rsrc1771) #10, !dbg !20
  %1211 = extractvalue [2 x <1 x float>] %63, 0, !dbg !32
  %1212 = extractelement <1 x float> %1211, i64 0, !dbg !32
  %1213 = fmul float %1212, %1174, !dbg !32
  %1214 = extractvalue [2 x <1 x float>] %63, 1, !dbg !32
  %1215 = extractelement <1 x float> %1214, i64 0, !dbg !32
  %1216 = fmul float %1215, %1175, !dbg !32
  %1217 = extractelement <4 x float> %1143, i64 0, !dbg !33
  %1218 = call float @llvm.amdgcn.exp2.f32(float %1217), !dbg !33
  %1219 = insertelement <4 x float> poison, float %1218, i64 0, !dbg !35
  %1220 = extractelement <4 x float> %1143, i64 1, !dbg !33
  %1221 = call float @llvm.amdgcn.exp2.f32(float %1220), !dbg !33
  %1222 = insertelement <4 x float> %1219, float %1221, i64 1, !dbg !35
  %1223 = extractelement <4 x float> %1143, i64 2, !dbg !33
  %1224 = call float @llvm.amdgcn.exp2.f32(float %1223), !dbg !33
  %1225 = insertelement <4 x float> %1222, float %1224, i64 2, !dbg !35
  %1226 = extractelement <4 x float> %1143, i64 3, !dbg !33
  %1227 = call float @llvm.amdgcn.exp2.f32(float %1226), !dbg !33
  %1228 = insertelement <4 x float> %1225, float %1227, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1228) #9, !dbg !35, !frisk.register_lifetimes !1
  %1229 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1229) #9, !dbg !33, !frisk.register_lifetimes !1
  %1230 = extractelement <4 x float> %1144, i64 0, !dbg !33
  %1231 = call float @llvm.amdgcn.exp2.f32(float %1230), !dbg !33
  %1232 = insertelement <4 x float> poison, float %1231, i64 0, !dbg !35
  %local.reduce120 = fadd float %1218, %1231, !dbg !34, !frisk.local_reduction !1
  %1233 = extractelement <4 x float> %1144, i64 1, !dbg !33
  %1234 = call float @llvm.amdgcn.exp2.f32(float %1233), !dbg !33
  %1235 = insertelement <4 x float> %1232, float %1234, i64 1, !dbg !35
  %local.reduce121 = fadd float %1221, %1234, !dbg !34, !frisk.local_reduction !1
  %1236 = extractelement <4 x float> %1144, i64 2, !dbg !33
  %1237 = call float @llvm.amdgcn.exp2.f32(float %1236), !dbg !33
  %1238 = insertelement <4 x float> %1235, float %1237, i64 2, !dbg !35
  %local.reduce122 = fadd float %1224, %1237, !dbg !34, !frisk.local_reduction !1
  %1239 = extractelement <4 x float> %1144, i64 3, !dbg !33
  %1240 = call float @llvm.amdgcn.exp2.f32(float %1239), !dbg !33
  %1241 = insertelement <4 x float> %1238, float %1240, i64 3, !dbg !35
  %math.result228 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1241) #9, !dbg !35, !frisk.register_lifetimes !1
  %1242 = fptrunc <4 x float> %math.result228 to <4 x half>, !dbg !35
  %math.packed229 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1242) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce123 = fadd float %1227, %1240, !dbg !34, !frisk.local_reduction !1
  %1243 = extractelement <4 x float> %1145, i64 0, !dbg !33
  %1244 = call float @llvm.amdgcn.exp2.f32(float %1243), !dbg !33
  %1245 = insertelement <4 x float> poison, float %1244, i64 0, !dbg !35
  %local.reduce124 = fadd float %local.reduce120, %1244, !dbg !34, !frisk.local_reduction !1
  %1246 = extractelement <4 x float> %1145, i64 1, !dbg !33
  %1247 = call float @llvm.amdgcn.exp2.f32(float %1246), !dbg !33
  %1248 = insertelement <4 x float> %1245, float %1247, i64 1, !dbg !35
  %local.reduce125 = fadd float %local.reduce121, %1247, !dbg !34, !frisk.local_reduction !1
  %1249 = extractelement <4 x float> %1145, i64 2, !dbg !33
  %1250 = call float @llvm.amdgcn.exp2.f32(float %1249), !dbg !33
  %1251 = insertelement <4 x float> %1248, float %1250, i64 2, !dbg !35
  %local.reduce126 = fadd float %local.reduce122, %1250, !dbg !34, !frisk.local_reduction !1
  %1252 = extractelement <4 x float> %1145, i64 3, !dbg !33
  %1253 = call float @llvm.amdgcn.exp2.f32(float %1252), !dbg !33
  %1254 = insertelement <4 x float> %1251, float %1253, i64 3, !dbg !35
  %math.result230 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1254) #9, !dbg !35, !frisk.register_lifetimes !1
  %1255 = fptrunc <4 x float> %math.result230 to <4 x half>, !dbg !35
  %math.packed231 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1255) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce127 = fadd float %local.reduce123, %1253, !dbg !34, !frisk.local_reduction !1
  %1256 = extractelement <4 x float> %1146, i64 0, !dbg !33
  %1257 = call float @llvm.amdgcn.exp2.f32(float %1256), !dbg !33
  %1258 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !35
  %local.reduce128 = fadd float %local.reduce124, %1257, !dbg !34, !frisk.local_reduction !1
  %1259 = extractelement <4 x float> %1146, i64 1, !dbg !33
  %1260 = call float @llvm.amdgcn.exp2.f32(float %1259), !dbg !33
  %1261 = insertelement <4 x float> %1258, float %1260, i64 1, !dbg !35
  %local.reduce129 = fadd float %local.reduce125, %1260, !dbg !34, !frisk.local_reduction !1
  %1262 = extractelement <4 x float> %1146, i64 2, !dbg !33
  %1263 = call float @llvm.amdgcn.exp2.f32(float %1262), !dbg !33
  %1264 = insertelement <4 x float> %1261, float %1263, i64 2, !dbg !35
  %local.reduce130 = fadd float %local.reduce126, %1263, !dbg !34, !frisk.local_reduction !1
  %1265 = extractelement <4 x float> %1146, i64 3, !dbg !33
  %1266 = call float @llvm.amdgcn.exp2.f32(float %1265), !dbg !33
  %1267 = insertelement <4 x float> %1264, float %1266, i64 3, !dbg !35
  %math.result232 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1267) #9, !dbg !35, !frisk.register_lifetimes !1
  %1268 = fptrunc <4 x float> %math.result232 to <4 x half>, !dbg !35
  %math.packed233 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1268) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce131 = fadd float %local.reduce127, %1266, !dbg !34, !frisk.local_reduction !1
  %1269 = extractelement <4 x float> %1147, i64 0, !dbg !33
  %1270 = call float @llvm.amdgcn.exp2.f32(float %1269), !dbg !33
  %1271 = insertelement <4 x float> poison, float %1270, i64 0, !dbg !35
  %local.reduce132 = fadd float %local.reduce128, %1270, !dbg !34, !frisk.local_reduction !1
  %1272 = extractelement <4 x float> %1147, i64 1, !dbg !33
  %1273 = call float @llvm.amdgcn.exp2.f32(float %1272), !dbg !33
  %1274 = insertelement <4 x float> %1271, float %1273, i64 1, !dbg !35
  %local.reduce133 = fadd float %local.reduce129, %1273, !dbg !34, !frisk.local_reduction !1
  %1275 = extractelement <4 x float> %1147, i64 2, !dbg !33
  %1276 = call float @llvm.amdgcn.exp2.f32(float %1275), !dbg !33
  %1277 = insertelement <4 x float> %1274, float %1276, i64 2, !dbg !35
  %local.reduce134 = fadd float %local.reduce130, %1276, !dbg !34, !frisk.local_reduction !1
  %1278 = extractelement <4 x float> %1147, i64 3, !dbg !33
  %1279 = call float @llvm.amdgcn.exp2.f32(float %1278), !dbg !33
  %1280 = insertelement <4 x float> %1277, float %1279, i64 3, !dbg !35
  %math.result234 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1280) #9, !dbg !35, !frisk.register_lifetimes !1
  %1281 = fptrunc <4 x float> %math.result234 to <4 x half>, !dbg !35
  %math.packed235 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1281) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce135 = fadd float %local.reduce131, %1279, !dbg !34, !frisk.local_reduction !1
  %1282 = extractelement <4 x float> %1148, i64 0, !dbg !33
  %1283 = call float @llvm.amdgcn.exp2.f32(float %1282), !dbg !33
  %1284 = insertelement <4 x float> poison, float %1283, i64 0, !dbg !35
  %local.reduce136 = fadd float %local.reduce132, %1283, !dbg !34, !frisk.local_reduction !1
  %1285 = extractelement <4 x float> %1148, i64 1, !dbg !33
  %1286 = call float @llvm.amdgcn.exp2.f32(float %1285), !dbg !33
  %1287 = insertelement <4 x float> %1284, float %1286, i64 1, !dbg !35
  %local.reduce137 = fadd float %local.reduce133, %1286, !dbg !34, !frisk.local_reduction !1
  %1288 = extractelement <4 x float> %1148, i64 2, !dbg !33
  %1289 = call float @llvm.amdgcn.exp2.f32(float %1288), !dbg !33
  %1290 = insertelement <4 x float> %1287, float %1289, i64 2, !dbg !35
  %local.reduce138 = fadd float %local.reduce134, %1289, !dbg !34, !frisk.local_reduction !1
  %1291 = extractelement <4 x float> %1148, i64 3, !dbg !33
  %1292 = call float @llvm.amdgcn.exp2.f32(float %1291), !dbg !33
  %1293 = insertelement <4 x float> %1290, float %1292, i64 3, !dbg !35
  %math.result236 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1293) #9, !dbg !35, !frisk.register_lifetimes !1
  %1294 = fptrunc <4 x float> %math.result236 to <4 x half>, !dbg !35
  %math.packed237 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1294) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce139 = fadd float %local.reduce135, %1292, !dbg !34, !frisk.local_reduction !1
  %1295 = extractelement <4 x float> %1149, i64 0, !dbg !33
  %1296 = call float @llvm.amdgcn.exp2.f32(float %1295), !dbg !33
  %1297 = insertelement <4 x float> poison, float %1296, i64 0, !dbg !35
  %local.reduce140 = fadd float %local.reduce136, %1296, !dbg !34, !frisk.local_reduction !1
  %1298 = extractelement <4 x float> %1149, i64 1, !dbg !33
  %1299 = call float @llvm.amdgcn.exp2.f32(float %1298), !dbg !33
  %1300 = insertelement <4 x float> %1297, float %1299, i64 1, !dbg !35
  %local.reduce141 = fadd float %local.reduce137, %1299, !dbg !34, !frisk.local_reduction !1
  %1301 = extractelement <4 x float> %1149, i64 2, !dbg !33
  %1302 = call float @llvm.amdgcn.exp2.f32(float %1301), !dbg !33
  %1303 = insertelement <4 x float> %1300, float %1302, i64 2, !dbg !35
  %local.reduce142 = fadd float %local.reduce138, %1302, !dbg !34, !frisk.local_reduction !1
  %1304 = extractelement <4 x float> %1149, i64 3, !dbg !33
  %1305 = call float @llvm.amdgcn.exp2.f32(float %1304), !dbg !33
  %1306 = insertelement <4 x float> %1303, float %1305, i64 3, !dbg !35
  %math.result238 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1306) #9, !dbg !35, !frisk.register_lifetimes !1
  %1307 = fptrunc <4 x float> %math.result238 to <4 x half>, !dbg !35
  %math.packed239 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1307) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce143 = fadd float %local.reduce139, %1305, !dbg !34, !frisk.local_reduction !1
  %1308 = extractelement <4 x float> %1150, i64 0, !dbg !33
  %1309 = call float @llvm.amdgcn.exp2.f32(float %1308), !dbg !33
  %1310 = insertelement <4 x float> poison, float %1309, i64 0, !dbg !35
  %local.reduce144 = fadd float %local.reduce140, %1309, !dbg !34, !frisk.local_reduction !1
  %1311 = extractelement <4 x float> %1150, i64 1, !dbg !33
  %1312 = call float @llvm.amdgcn.exp2.f32(float %1311), !dbg !33
  %1313 = insertelement <4 x float> %1310, float %1312, i64 1, !dbg !35
  %local.reduce145 = fadd float %local.reduce141, %1312, !dbg !34, !frisk.local_reduction !1
  %local.reduce148 = fadd float %local.reduce144, %local.reduce145, !dbg !34, !frisk.local_reduction !1
  %1314 = extractelement <4 x float> %1150, i64 2, !dbg !33
  %1315 = call float @llvm.amdgcn.exp2.f32(float %1314), !dbg !33
  %1316 = insertelement <4 x float> %1313, float %1315, i64 2, !dbg !35
  %local.reduce146 = fadd float %local.reduce142, %1315, !dbg !34, !frisk.local_reduction !1
  %1317 = extractelement <4 x float> %1150, i64 3, !dbg !33
  %1318 = call float @llvm.amdgcn.exp2.f32(float %1317), !dbg !33
  %1319 = insertelement <4 x float> %1316, float %1318, i64 3, !dbg !35
  %math.result240 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1319) #9, !dbg !35, !frisk.register_lifetimes !1
  %1320 = fptrunc <4 x float> %math.result240 to <4 x half>, !dbg !35
  %math.packed241 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1320) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce147 = fadd float %local.reduce143, %1318, !dbg !34, !frisk.local_reduction !1
  %local.reduce149 = fadd float %local.reduce146, %local.reduce147, !dbg !34, !frisk.local_reduction !1
  %1321 = fadd float %local.reduce148, %local.reduce149, !dbg !34, !frisk.local_reduction !1
  %1322 = extractelement <4 x float> %1152, i64 0, !dbg !33
  %1323 = call float @llvm.amdgcn.exp2.f32(float %1322), !dbg !33
  %1324 = insertelement <4 x float> poison, float %1323, i64 0, !dbg !35
  %1325 = extractelement <4 x float> %1152, i64 1, !dbg !33
  %1326 = call float @llvm.amdgcn.exp2.f32(float %1325), !dbg !33
  %1327 = insertelement <4 x float> %1324, float %1326, i64 1, !dbg !35
  %1328 = extractelement <4 x float> %1152, i64 2, !dbg !33
  %1329 = call float @llvm.amdgcn.exp2.f32(float %1328), !dbg !33
  %1330 = insertelement <4 x float> %1327, float %1329, i64 2, !dbg !35
  %1331 = extractelement <4 x float> %1152, i64 3, !dbg !33
  %1332 = call float @llvm.amdgcn.exp2.f32(float %1331), !dbg !33
  %1333 = insertelement <4 x float> %1330, float %1332, i64 3, !dbg !35
  %math.result242 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1333) #9, !dbg !35, !frisk.register_lifetimes !1
  %1334 = fptrunc <4 x float> %math.result242 to <4 x half>, !dbg !35
  %math.packed243 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1334) #9, !dbg !33, !frisk.register_lifetimes !1
  %1335 = extractelement <4 x float> %1153, i64 0, !dbg !33
  %1336 = call float @llvm.amdgcn.exp2.f32(float %1335), !dbg !33
  %1337 = insertelement <4 x float> poison, float %1336, i64 0, !dbg !35
  %local.reduce = fadd float %1323, %1336, !dbg !34, !frisk.local_reduction !1
  %1338 = extractelement <4 x float> %1153, i64 1, !dbg !33
  %1339 = call float @llvm.amdgcn.exp2.f32(float %1338), !dbg !33
  %1340 = insertelement <4 x float> %1337, float %1339, i64 1, !dbg !35
  %local.reduce90 = fadd float %1326, %1339, !dbg !34, !frisk.local_reduction !1
  %1341 = extractelement <4 x float> %1153, i64 2, !dbg !33
  %1342 = call float @llvm.amdgcn.exp2.f32(float %1341), !dbg !33
  %1343 = insertelement <4 x float> %1340, float %1342, i64 2, !dbg !35
  %local.reduce91 = fadd float %1329, %1342, !dbg !34, !frisk.local_reduction !1
  %1344 = extractelement <4 x float> %1153, i64 3, !dbg !33
  %1345 = call float @llvm.amdgcn.exp2.f32(float %1344), !dbg !33
  %1346 = insertelement <4 x float> %1343, float %1345, i64 3, !dbg !35
  %math.result244 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1346) #9, !dbg !35, !frisk.register_lifetimes !1
  %1347 = fptrunc <4 x float> %math.result244 to <4 x half>, !dbg !35
  %math.packed245 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1347) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce92 = fadd float %1332, %1345, !dbg !34, !frisk.local_reduction !1
  %1348 = extractelement <4 x float> %1154, i64 0, !dbg !33
  %1349 = call float @llvm.amdgcn.exp2.f32(float %1348), !dbg !33
  %1350 = insertelement <4 x float> poison, float %1349, i64 0, !dbg !35
  %local.reduce93 = fadd float %local.reduce, %1349, !dbg !34, !frisk.local_reduction !1
  %1351 = extractelement <4 x float> %1154, i64 1, !dbg !33
  %1352 = call float @llvm.amdgcn.exp2.f32(float %1351), !dbg !33
  %1353 = insertelement <4 x float> %1350, float %1352, i64 1, !dbg !35
  %local.reduce94 = fadd float %local.reduce90, %1352, !dbg !34, !frisk.local_reduction !1
  %1354 = extractelement <4 x float> %1154, i64 2, !dbg !33
  %1355 = call float @llvm.amdgcn.exp2.f32(float %1354), !dbg !33
  %1356 = insertelement <4 x float> %1353, float %1355, i64 2, !dbg !35
  %local.reduce95 = fadd float %local.reduce91, %1355, !dbg !34, !frisk.local_reduction !1
  %1357 = extractelement <4 x float> %1154, i64 3, !dbg !33
  %1358 = call float @llvm.amdgcn.exp2.f32(float %1357), !dbg !33
  %1359 = insertelement <4 x float> %1356, float %1358, i64 3, !dbg !35
  %math.result246 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1359) #9, !dbg !35, !frisk.register_lifetimes !1
  %1360 = fptrunc <4 x float> %math.result246 to <4 x half>, !dbg !35
  %math.packed247 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1360) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce96 = fadd float %local.reduce92, %1358, !dbg !34, !frisk.local_reduction !1
  %1361 = extractelement <4 x float> %1155, i64 0, !dbg !33
  %1362 = call float @llvm.amdgcn.exp2.f32(float %1361), !dbg !33
  %1363 = insertelement <4 x float> poison, float %1362, i64 0, !dbg !35
  %local.reduce97 = fadd float %local.reduce93, %1362, !dbg !34, !frisk.local_reduction !1
  %1364 = extractelement <4 x float> %1155, i64 1, !dbg !33
  %1365 = call float @llvm.amdgcn.exp2.f32(float %1364), !dbg !33
  %1366 = insertelement <4 x float> %1363, float %1365, i64 1, !dbg !35
  %local.reduce98 = fadd float %local.reduce94, %1365, !dbg !34, !frisk.local_reduction !1
  %1367 = extractelement <4 x float> %1155, i64 2, !dbg !33
  %1368 = call float @llvm.amdgcn.exp2.f32(float %1367), !dbg !33
  %1369 = insertelement <4 x float> %1366, float %1368, i64 2, !dbg !35
  %local.reduce99 = fadd float %local.reduce95, %1368, !dbg !34, !frisk.local_reduction !1
  %1370 = extractelement <4 x float> %1155, i64 3, !dbg !33
  %1371 = call float @llvm.amdgcn.exp2.f32(float %1370), !dbg !33
  %1372 = insertelement <4 x float> %1369, float %1371, i64 3, !dbg !35
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1372) #9, !dbg !35, !frisk.register_lifetimes !1
  %1373 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !35
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1373) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce100 = fadd float %local.reduce96, %1371, !dbg !34, !frisk.local_reduction !1
  %1374 = extractelement <4 x float> %1156, i64 0, !dbg !33
  %1375 = call float @llvm.amdgcn.exp2.f32(float %1374), !dbg !33
  %1376 = insertelement <4 x float> poison, float %1375, i64 0, !dbg !35
  %local.reduce101 = fadd float %local.reduce97, %1375, !dbg !34, !frisk.local_reduction !1
  %1377 = extractelement <4 x float> %1156, i64 1, !dbg !33
  %1378 = call float @llvm.amdgcn.exp2.f32(float %1377), !dbg !33
  %1379 = insertelement <4 x float> %1376, float %1378, i64 1, !dbg !35
  %local.reduce102 = fadd float %local.reduce98, %1378, !dbg !34, !frisk.local_reduction !1
  %1380 = extractelement <4 x float> %1156, i64 2, !dbg !33
  %1381 = call float @llvm.amdgcn.exp2.f32(float %1380), !dbg !33
  %1382 = insertelement <4 x float> %1379, float %1381, i64 2, !dbg !35
  %local.reduce103 = fadd float %local.reduce99, %1381, !dbg !34, !frisk.local_reduction !1
  %1383 = extractelement <4 x float> %1156, i64 3, !dbg !33
  %1384 = call float @llvm.amdgcn.exp2.f32(float %1383), !dbg !33
  %1385 = insertelement <4 x float> %1382, float %1384, i64 3, !dbg !35
  %math.result250 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1385) #9, !dbg !35, !frisk.register_lifetimes !1
  %1386 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !35
  %math.packed251 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1386) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce104 = fadd float %local.reduce100, %1384, !dbg !34, !frisk.local_reduction !1
  %1387 = extractelement <4 x float> %1157, i64 0, !dbg !33
  %1388 = call float @llvm.amdgcn.exp2.f32(float %1387), !dbg !33
  %1389 = insertelement <4 x float> poison, float %1388, i64 0, !dbg !35
  %local.reduce105 = fadd float %local.reduce101, %1388, !dbg !34, !frisk.local_reduction !1
  %1390 = extractelement <4 x float> %1157, i64 1, !dbg !33
  %1391 = call float @llvm.amdgcn.exp2.f32(float %1390), !dbg !33
  %1392 = insertelement <4 x float> %1389, float %1391, i64 1, !dbg !35
  %local.reduce106 = fadd float %local.reduce102, %1391, !dbg !34, !frisk.local_reduction !1
  %1393 = extractelement <4 x float> %1157, i64 2, !dbg !33
  %1394 = call float @llvm.amdgcn.exp2.f32(float %1393), !dbg !33
  %1395 = insertelement <4 x float> %1392, float %1394, i64 2, !dbg !35
  %local.reduce107 = fadd float %local.reduce103, %1394, !dbg !34, !frisk.local_reduction !1
  %1396 = extractelement <4 x float> %1157, i64 3, !dbg !33
  %1397 = call float @llvm.amdgcn.exp2.f32(float %1396), !dbg !33
  %1398 = insertelement <4 x float> %1395, float %1397, i64 3, !dbg !35
  %math.result252 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1398) #9, !dbg !35, !frisk.register_lifetimes !1
  %1399 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !35
  %math.packed253 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1399) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce108 = fadd float %local.reduce104, %1397, !dbg !34, !frisk.local_reduction !1
  %1400 = extractelement <4 x float> %1158, i64 0, !dbg !33
  %1401 = call float @llvm.amdgcn.exp2.f32(float %1400), !dbg !33
  %1402 = insertelement <4 x float> poison, float %1401, i64 0, !dbg !35
  %local.reduce109 = fadd float %local.reduce105, %1401, !dbg !34, !frisk.local_reduction !1
  %1403 = extractelement <4 x float> %1158, i64 1, !dbg !33
  %1404 = call float @llvm.amdgcn.exp2.f32(float %1403), !dbg !33
  %1405 = insertelement <4 x float> %1402, float %1404, i64 1, !dbg !35
  %local.reduce110 = fadd float %local.reduce106, %1404, !dbg !34, !frisk.local_reduction !1
  %1406 = extractelement <4 x float> %1158, i64 2, !dbg !33
  %1407 = call float @llvm.amdgcn.exp2.f32(float %1406), !dbg !33
  %1408 = insertelement <4 x float> %1405, float %1407, i64 2, !dbg !35
  %local.reduce111 = fadd float %local.reduce107, %1407, !dbg !34, !frisk.local_reduction !1
  %1409 = extractelement <4 x float> %1158, i64 3, !dbg !33
  %1410 = call float @llvm.amdgcn.exp2.f32(float %1409), !dbg !33
  %1411 = insertelement <4 x float> %1408, float %1410, i64 3, !dbg !35
  %math.result254 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1411) #9, !dbg !35, !frisk.register_lifetimes !1
  %1412 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !35
  %math.packed255 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1412) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce112 = fadd float %local.reduce108, %1410, !dbg !34, !frisk.local_reduction !1
  %1413 = extractelement <4 x float> %1159, i64 0, !dbg !33
  %1414 = call float @llvm.amdgcn.exp2.f32(float %1413), !dbg !33
  %1415 = insertelement <4 x float> poison, float %1414, i64 0, !dbg !35
  %local.reduce113 = fadd float %local.reduce109, %1414, !dbg !34, !frisk.local_reduction !1
  %1416 = extractelement <4 x float> %1159, i64 1, !dbg !33
  %1417 = call float @llvm.amdgcn.exp2.f32(float %1416), !dbg !33
  %1418 = insertelement <4 x float> %1415, float %1417, i64 1, !dbg !35
  %local.reduce114 = fadd float %local.reduce110, %1417, !dbg !34, !frisk.local_reduction !1
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !34, !frisk.local_reduction !1
  %1419 = extractelement <4 x float> %1159, i64 2, !dbg !33
  %1420 = call float @llvm.amdgcn.exp2.f32(float %1419), !dbg !33
  %1421 = insertelement <4 x float> %1418, float %1420, i64 2, !dbg !35
  %local.reduce115 = fadd float %local.reduce111, %1420, !dbg !34, !frisk.local_reduction !1
  %1422 = extractelement <4 x float> %1159, i64 3, !dbg !33
  %1423 = call float @llvm.amdgcn.exp2.f32(float %1422), !dbg !33
  %1424 = insertelement <4 x float> %1421, float %1423, i64 3, !dbg !35
  %math.result256 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1424) #9, !dbg !35, !frisk.register_lifetimes !1
  %1425 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !35
  %math.packed257 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1425) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce116 = fadd float %local.reduce112, %1423, !dbg !34, !frisk.local_reduction !1
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !34, !frisk.local_reduction !1
  %1426 = fadd float %local.reduce117, %local.reduce118, !dbg !34, !frisk.local_reduction !1
  %1427 = bitcast float %1321 to i32, !dbg !34
  %1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1427), !dbg !34
  %1429 = bitcast i32 %1428 to float, !dbg !34
  %1430 = fadd float %1321, %1429, !dbg !34
  %1431 = bitcast float %1430 to i32, !dbg !34
  %1432 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1431), !dbg !34
  %1433 = bitcast i32 %1432 to float, !dbg !34
  %1434 = fadd float %1430, %1433, !dbg !34
  %1435 = fadd float %1434, %1213, !dbg !34
  %1436 = bitcast float %1426 to i32, !dbg !34
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1436), !dbg !34
  %1438 = bitcast i32 %1437 to float, !dbg !34
  %1439 = fadd float %1426, %1438, !dbg !34
  %1440 = bitcast float %1439 to i32, !dbg !34
  %1441 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1440), !dbg !34
  %1442 = bitcast i32 %1441 to float, !dbg !34
  %1443 = fadd float %1439, %1442, !dbg !34
  %1444 = fadd float %1443, %1216, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local836 = shl nuw nsw i32 %address.thread798, 3, !dbg !16
  %address.local837 = and i32 %address.local836, 504, !dbg !16
  %address.local838 = zext nneg i32 %address.local837 to i64, !dbg !16
  %address.local839 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %1445 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local839, i32 0) #0, !dbg !16
  %1446 = bitcast <4 x i32> %1445 to <8 x half>, !dbg !16
  %address.local840 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local840, i64 1024, !dbg !16
  %1447 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #0, !dbg !16
  %1448 = bitcast <4 x i32> %1447 to <8 x half>, !dbg !16
  %address.local842 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local843 = getelementptr i8, ptr addrspace(3) %address.local842, i64 2048, !dbg !16
  %1449 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local843, i32 0) #0, !dbg !16
  %1450 = bitcast <4 x i32> %1449 to <8 x half>, !dbg !16
  %address.local844 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local845 = getelementptr i8, ptr addrspace(3) %address.local844, i64 3072, !dbg !16
  %1451 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local845, i32 0) #0, !dbg !16
  %1452 = bitcast <4 x i32> %1451 to <8 x half>, !dbg !16
  %address.local846 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local847 = getelementptr i8, ptr addrspace(3) %address.local846, i64 4096, !dbg !16
  %1453 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local847, i32 0) #0, !dbg !16
  %1454 = bitcast <4 x i32> %1453 to <8 x half>, !dbg !16
  %address.local848 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local849 = getelementptr i8, ptr addrspace(3) %address.local848, i64 5120, !dbg !16
  %1455 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local849, i32 0) #0, !dbg !16
  %1456 = bitcast <4 x i32> %1455 to <8 x half>, !dbg !16
  %address.local850 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local851 = getelementptr i8, ptr addrspace(3) %address.local850, i64 6144, !dbg !16
  %1457 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #0, !dbg !16
  %1458 = bitcast <4 x i32> %1457 to <8 x half>, !dbg !16
  %address.local852 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local853 = getelementptr i8, ptr addrspace(3) %address.local852, i64 7168, !dbg !16
  %1459 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local853, i32 0) #0, !dbg !16
  %1460 = bitcast <4 x i32> %1459 to <8 x half>, !dbg !16
  %1461 = shufflevector <8 x half> %1446, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1461, <4 x float> zeroinitializer) #9, !dbg !16
  %1463 = shufflevector <8 x half> %1448, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1463, <4 x float> %1462) #9, !dbg !16
  %1465 = shufflevector <8 x half> %1450, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1465, <4 x float> %1464) #9, !dbg !16
  %1467 = shufflevector <8 x half> %1452, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1467, <4 x float> %1466) #9, !dbg !16
  %1469 = shufflevector <8 x half> %1454, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1469, <4 x float> %1468) #9, !dbg !16
  %1471 = shufflevector <8 x half> %1456, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1471, <4 x float> %1470) #9, !dbg !16
  %1473 = shufflevector <8 x half> %1458, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1473, <4 x float> %1472) #9, !dbg !16
  %1475 = shufflevector <8 x half> %1460, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1475, <4 x float> %1474) #9, !dbg !16
  %1477 = extractelement <4 x float> %1476, i64 0, !dbg !16
  %1478 = extractelement <4 x float> %1476, i64 1, !dbg !16
  %1479 = extractelement <4 x float> %1476, i64 2, !dbg !16
  %1480 = extractelement <4 x float> %1476, i64 3, !dbg !16
  %1481 = select i1 %.not, float %1478, float %1477, !dbg !16
  %1482 = bitcast float %1481 to i32, !dbg !16
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1482), !dbg !16
  %1484 = bitcast i32 %1483 to float, !dbg !16
  %1485 = select i1 %.not, float %1477, float %1484, !dbg !16
  %1486 = select i1 %.not, float %1484, float %1478, !dbg !16
  %1487 = select i1 %.not, float %1480, float %1479, !dbg !16
  %1488 = bitcast float %1487 to i32, !dbg !16
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1488), !dbg !16
  %1490 = bitcast i32 %1489 to float, !dbg !16
  %1491 = select i1 %.not, float %1479, float %1490, !dbg !16
  %1492 = select i1 %.not, float %1490, float %1480, !dbg !16
  %1493 = select i1 %.not1, float %1491, float %1485, !dbg !16
  %1494 = bitcast float %1493 to i32, !dbg !16
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1494), !dbg !16
  %1496 = bitcast i32 %1495 to float, !dbg !16
  %1497 = select i1 %.not1, float %1485, float %1496, !dbg !16
  %1498 = select i1 %.not1, float %1496, float %1491, !dbg !16
  %1499 = select i1 %.not1, float %1492, float %1486, !dbg !16
  %1500 = bitcast float %1499 to i32, !dbg !16
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1500), !dbg !16
  %1502 = bitcast i32 %1501 to float, !dbg !16
  %1503 = select i1 %.not1, float %1486, float %1502, !dbg !16
  %1504 = select i1 %.not1, float %1502, float %1492, !dbg !16
  %1505 = shufflevector <8 x half> %1446, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1505, <4 x float> zeroinitializer) #9, !dbg !16
  %1507 = shufflevector <8 x half> %1448, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1507, <4 x float> %1506) #9, !dbg !16
  %1509 = shufflevector <8 x half> %1450, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1509, <4 x float> %1508) #9, !dbg !16
  %1511 = shufflevector <8 x half> %1452, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1511, <4 x float> %1510) #9, !dbg !16
  %1513 = shufflevector <8 x half> %1454, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1513, <4 x float> %1512) #9, !dbg !16
  %1515 = shufflevector <8 x half> %1456, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1515, <4 x float> %1514) #9, !dbg !16
  %1517 = shufflevector <8 x half> %1458, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1517, <4 x float> %1516) #9, !dbg !16
  %1519 = shufflevector <8 x half> %1460, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1519, <4 x float> %1518) #9, !dbg !16
  %1521 = extractelement <4 x float> %1520, i64 0, !dbg !16
  %1522 = extractelement <4 x float> %1520, i64 1, !dbg !16
  %1523 = extractelement <4 x float> %1520, i64 2, !dbg !16
  %1524 = extractelement <4 x float> %1520, i64 3, !dbg !16
  %1525 = select i1 %.not, float %1522, float %1521, !dbg !16
  %1526 = bitcast float %1525 to i32, !dbg !16
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1526), !dbg !16
  %1528 = bitcast i32 %1527 to float, !dbg !16
  %1529 = select i1 %.not, float %1521, float %1528, !dbg !16
  %1530 = select i1 %.not, float %1528, float %1522, !dbg !16
  %1531 = select i1 %.not, float %1524, float %1523, !dbg !16
  %1532 = bitcast float %1531 to i32, !dbg !16
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1532), !dbg !16
  %1534 = bitcast i32 %1533 to float, !dbg !16
  %1535 = select i1 %.not, float %1523, float %1534, !dbg !16
  %1536 = select i1 %.not, float %1534, float %1524, !dbg !16
  %1537 = select i1 %.not1, float %1535, float %1529, !dbg !16
  %1538 = bitcast float %1537 to i32, !dbg !16
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1538), !dbg !16
  %1540 = bitcast i32 %1539 to float, !dbg !16
  %1541 = select i1 %.not1, float %1529, float %1540, !dbg !16
  %1542 = select i1 %.not1, float %1540, float %1535, !dbg !16
  %1543 = select i1 %.not1, float %1536, float %1530, !dbg !16
  %1544 = bitcast float %1543 to i32, !dbg !16
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1544), !dbg !16
  %1546 = bitcast i32 %1545 to float, !dbg !16
  %1547 = select i1 %.not1, float %1530, float %1546, !dbg !16
  %1548 = select i1 %.not1, float %1546, float %1536, !dbg !16
  %1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1461, <4 x float> zeroinitializer) #9, !dbg !16
  %1550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1463, <4 x float> %1549) #9, !dbg !16
  %1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1465, <4 x float> %1550) #9, !dbg !16
  %1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1467, <4 x float> %1551) #9, !dbg !16
  %1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1469, <4 x float> %1552) #9, !dbg !16
  %1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1471, <4 x float> %1553) #9, !dbg !16
  %1555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1473, <4 x float> %1554) #9, !dbg !16
  %1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1475, <4 x float> %1555) #9, !dbg !16
  %1557 = extractelement <4 x float> %1556, i64 0, !dbg !16
  %1558 = extractelement <4 x float> %1556, i64 1, !dbg !16
  %1559 = extractelement <4 x float> %1556, i64 2, !dbg !16
  %1560 = extractelement <4 x float> %1556, i64 3, !dbg !16
  %1561 = select i1 %.not, float %1558, float %1557, !dbg !16
  %1562 = bitcast float %1561 to i32, !dbg !16
  %1563 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1562), !dbg !16
  %1564 = bitcast i32 %1563 to float, !dbg !16
  %1565 = select i1 %.not, float %1557, float %1564, !dbg !16
  %1566 = select i1 %.not, float %1564, float %1558, !dbg !16
  %1567 = select i1 %.not, float %1560, float %1559, !dbg !16
  %1568 = bitcast float %1567 to i32, !dbg !16
  %1569 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1568), !dbg !16
  %1570 = bitcast i32 %1569 to float, !dbg !16
  %1571 = select i1 %.not, float %1559, float %1570, !dbg !16
  %1572 = select i1 %.not, float %1570, float %1560, !dbg !16
  %1573 = select i1 %.not1, float %1571, float %1565, !dbg !16
  %1574 = bitcast float %1573 to i32, !dbg !16
  %1575 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1574), !dbg !16
  %1576 = bitcast i32 %1575 to float, !dbg !16
  %1577 = select i1 %.not1, float %1565, float %1576, !dbg !16
  %1578 = select i1 %.not1, float %1576, float %1571, !dbg !16
  %1579 = select i1 %.not1, float %1572, float %1566, !dbg !16
  %1580 = bitcast float %1579 to i32, !dbg !16
  %1581 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1580), !dbg !16
  %1582 = bitcast i32 %1581 to float, !dbg !16
  %1583 = select i1 %.not1, float %1566, float %1582, !dbg !16
  %1584 = select i1 %.not1, float %1582, float %1572, !dbg !16
  %1585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1505, <4 x float> zeroinitializer) #9, !dbg !16
  %1586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1507, <4 x float> %1585) #9, !dbg !16
  %1587 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1509, <4 x float> %1586) #9, !dbg !16
  %1588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1511, <4 x float> %1587) #9, !dbg !16
  %1589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1513, <4 x float> %1588) #9, !dbg !16
  %1590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1515, <4 x float> %1589) #9, !dbg !16
  %1591 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1517, <4 x float> %1590) #9, !dbg !16
  %1592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1519, <4 x float> %1591) #9, !dbg !16
  %1593 = extractelement <4 x float> %1592, i64 0, !dbg !16
  %1594 = extractelement <4 x float> %1592, i64 1, !dbg !16
  %1595 = extractelement <4 x float> %1592, i64 2, !dbg !16
  %1596 = extractelement <4 x float> %1592, i64 3, !dbg !16
  %1597 = select i1 %.not, float %1594, float %1593, !dbg !16
  %1598 = bitcast float %1597 to i32, !dbg !16
  %1599 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1598), !dbg !16
  %1600 = bitcast i32 %1599 to float, !dbg !16
  %1601 = select i1 %.not, float %1593, float %1600, !dbg !16
  %1602 = select i1 %.not, float %1600, float %1594, !dbg !16
  %1603 = select i1 %.not, float %1596, float %1595, !dbg !16
  %1604 = bitcast float %1603 to i32, !dbg !16
  %1605 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1604), !dbg !16
  %1606 = bitcast i32 %1605 to float, !dbg !16
  %1607 = select i1 %.not, float %1595, float %1606, !dbg !16
  %1608 = select i1 %.not, float %1606, float %1596, !dbg !16
  %1609 = select i1 %.not1, float %1607, float %1601, !dbg !16
  %1610 = bitcast float %1609 to i32, !dbg !16
  %1611 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1610), !dbg !16
  %1612 = bitcast i32 %1611 to float, !dbg !16
  %1613 = select i1 %.not1, float %1601, float %1612, !dbg !16
  %1614 = select i1 %.not1, float %1612, float %1607, !dbg !16
  %1615 = select i1 %.not1, float %1608, float %1602, !dbg !16
  %1616 = bitcast float %1615 to i32, !dbg !16
  %1617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1616), !dbg !16
  %1618 = bitcast i32 %1617 to float, !dbg !16
  %1619 = select i1 %.not1, float %1602, float %1618, !dbg !16
  %1620 = select i1 %.not1, float %1618, float %1608, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %1621, label %1622, !dbg !36

1621:                                             ; preds = %1173
  %address.thread854 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local855 = lshr i32 %address.thread854, 1, !dbg !15
  %address.local856 = and i32 %address.local855, 96, !dbg !15
  %address.local857 = and i32 %address.thread854, 15, !dbg !15
  %address.local858 = or disjoint i32 %address.local856, %address.local857, !dbg !15
  %address.local859 = zext nneg i32 %address.local858 to i64, !dbg !15
  %address.local860 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !15
  store float %1170, ptr addrspace(3) %address.local860, align 4, !dbg !36
  %address.local861 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !15
  %address.local862 = getelementptr i8, ptr addrspace(3) %address.local861, i64 64, !dbg !15
  store float %1171, ptr addrspace(3) %address.local862, align 4, !dbg !36
  br label %1622, !dbg !36

1622:                                             ; preds = %1621, %1173
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread863 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %1623 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !36
  %lds.const.gep1748 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %1624 = load float, ptr addrspace(3) %lds.const.gep1748, align 4, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %1625 = add nuw nsw i64 %101, %15, !dbg !20
  %1626 = shl i64 %1625, 19, !dbg !20
  %1627 = and i64 %1626, 16252928, !dbg !20
  %1628 = add i64 %1627, %107, !dbg !20
  %1629 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1628, !dbg !20
  %1630 = getelementptr i8, ptr addrspace(1) %1629, i64 64, !dbg !20
  %address.local877 = zext nneg i32 %address.thread863 to i64, !dbg !12
  %address.local878 = shl nuw nsw i64 %address.local877, 3, !dbg !11
  %address.local879 = and i64 %address.local878, 1536, !dbg !11
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local879, !dbg !11
  %address.local881 = ptrtoint ptr addrspace(3) %address.local880 to i64, !dbg !11
  %address.local882 = trunc i64 %address.local881 to i32, !dbg !11
  %1631 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local882), !dbg !20
  %1632 = zext i32 %1631 to i64, !dbg !20
  %1633 = inttoptr i64 %1632 to ptr addrspace(3), !dbg !20
  %address.local883 = shl nuw nsw i64 %address.local877, 5, !dbg !20
  %address.local884 = and i64 %address.local883, 8064, !dbg !20
  %address.local885 = shl nuw nsw i64 %address.local877, 3, !dbg !20
  %address.local886 = and i64 %address.local885, 24, !dbg !20
  %1634 = getelementptr [2 x i8], ptr addrspace(1) %1630, i64 %address.local884, !dbg !20
  %address.local888 = getelementptr [2 x i8], ptr addrspace(1) %1634, i64 %address.local886, !dbg !20
  %1635 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1636 = ptrtoint ptr addrspace(1) %address.local888 to i64, !dbg !20
  %1637 = sub i64 %1636, %1635, !dbg !20
  %dma.buffer.offset1767 = trunc i64 %1637 to i32, !dbg !20
  %1638 = trunc i64 %1635 to i32, !dbg !20
  %1639 = insertelement <4 x i32> poison, i32 %1638, i64 0, !dbg !20
  %1640 = lshr i64 %1635, 32, !dbg !20
  %1641 = trunc i64 %1640 to i32, !dbg !20
  %1642 = insertelement <4 x i32> %1639, i32 %1641, i64 1, !dbg !20
  %1643 = insertelement <4 x i32> %1642, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1768 = insertelement <4 x i32> %1643, i32 131072, i64 3, !dbg !20
  %1644 = ptrtoint ptr addrspace(3) %1633 to i32, !dbg !20
  %dma.buffer.lds1769 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1644), !dbg !20
  %1645 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1769, i32 %dma.buffer.offset1767, <4 x i32> %dma.buffer.rsrc1768) #10, !dbg !20
  %address.local889 = or disjoint i64 %address.local877, 256, !dbg !11
  %address.local890 = shl nuw nsw i64 %address.local889, 3, !dbg !11
  %address.local891 = and i64 %address.local890, 3584, !dbg !11
  %address.local892 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local891, !dbg !11
  %address.local893 = ptrtoint ptr addrspace(3) %address.local892 to i64, !dbg !11
  %address.local894 = trunc i64 %address.local893 to i32, !dbg !11
  %1646 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !20
  %1647 = zext i32 %1646 to i64, !dbg !20
  %1648 = inttoptr i64 %1647 to ptr addrspace(3), !dbg !20
  %address.local895 = shl nuw nsw i64 %address.local889, 5, !dbg !20
  %address.local896 = and i64 %address.local895, 16256, !dbg !20
  %address.local897 = shl nuw nsw i64 %address.local877, 3, !dbg !20
  %address.local898 = and i64 %address.local897, 24, !dbg !20
  %1649 = getelementptr [2 x i8], ptr addrspace(1) %1630, i64 %address.local896, !dbg !20
  %address.local900 = getelementptr [2 x i8], ptr addrspace(1) %1649, i64 %address.local898, !dbg !20
  %1650 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1651 = ptrtoint ptr addrspace(1) %address.local900 to i64, !dbg !20
  %1652 = sub i64 %1651, %1650, !dbg !20
  %dma.buffer.offset1764 = trunc i64 %1652 to i32, !dbg !20
  %1653 = trunc i64 %1650 to i32, !dbg !20
  %1654 = insertelement <4 x i32> poison, i32 %1653, i64 0, !dbg !20
  %1655 = lshr i64 %1650, 32, !dbg !20
  %1656 = trunc i64 %1655 to i32, !dbg !20
  %1657 = insertelement <4 x i32> %1654, i32 %1656, i64 1, !dbg !20
  %1658 = insertelement <4 x i32> %1657, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1765 = insertelement <4 x i32> %1658, i32 131072, i64 3, !dbg !20
  %1659 = ptrtoint ptr addrspace(3) %1648 to i32, !dbg !20
  %dma.buffer.lds1766 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1659), !dbg !20
  %1660 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1766, i32 %dma.buffer.offset1764, <4 x i32> %dma.buffer.rsrc1765) #10, !dbg !20
  %1661 = extractvalue [2 x <8 x float>] %62, 0, !dbg !36
  %1662 = shufflevector <8 x float> %1661, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1663 = insertelement <4 x float> poison, float %1623, i64 0, !dbg !36
  %1664 = shufflevector <4 x float> %1663, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1665 = fmul <4 x float> %1662, %1664, !dbg !36
  %1666 = insertelement <4 x float> poison, float %1497, i64 0, !dbg !37
  %1667 = insertelement <4 x float> %1666, float %1503, i64 1, !dbg !37
  %1668 = insertelement <4 x float> %1667, float %1498, i64 2, !dbg !37
  %1669 = insertelement <4 x float> %1668, float %1504, i64 3, !dbg !37
  %1670 = fadd <4 x float> %1665, %1669, !dbg !37
  %1671 = shufflevector <8 x float> %1661, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1672 = fmul <4 x float> %1671, %1664, !dbg !36
  %1673 = insertelement <4 x float> poison, float %1541, i64 0, !dbg !37
  %1674 = insertelement <4 x float> %1673, float %1547, i64 1, !dbg !37
  %1675 = insertelement <4 x float> %1674, float %1542, i64 2, !dbg !37
  %1676 = insertelement <4 x float> %1675, float %1548, i64 3, !dbg !37
  %1677 = fadd <4 x float> %1672, %1676, !dbg !37
  %1678 = extractvalue [2 x <8 x float>] %62, 1, !dbg !36
  %1679 = shufflevector <8 x float> %1678, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1680 = insertelement <4 x float> poison, float %1624, i64 0, !dbg !36
  %1681 = shufflevector <4 x float> %1680, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1682 = fmul <4 x float> %1679, %1681, !dbg !36
  %1683 = insertelement <4 x float> poison, float %1577, i64 0, !dbg !37
  %1684 = insertelement <4 x float> %1683, float %1583, i64 1, !dbg !37
  %1685 = insertelement <4 x float> %1684, float %1578, i64 2, !dbg !37
  %1686 = insertelement <4 x float> %1685, float %1584, i64 3, !dbg !37
  %1687 = fadd <4 x float> %1682, %1686, !dbg !37
  %1688 = shufflevector <8 x float> %1678, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1689 = fmul <4 x float> %1688, %1681, !dbg !36
  %1690 = insertelement <4 x float> poison, float %1613, i64 0, !dbg !37
  %1691 = insertelement <4 x float> %1690, float %1619, i64 1, !dbg !37
  %1692 = insertelement <4 x float> %1691, float %1614, i64 2, !dbg !37
  %1693 = insertelement <4 x float> %1692, float %1620, i64 3, !dbg !37
  %1694 = fadd <4 x float> %1689, %1693, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local901 = shl nuw nsw i32 %address.thread863, 3, !dbg !16
  %address.local902 = and i32 %address.local901, 504, !dbg !16
  %address.local903 = zext nneg i32 %address.local902 to i64, !dbg !16
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %1695 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #0, !dbg !16
  %1696 = bitcast <4 x i32> %1695 to <8 x half>, !dbg !16
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 1024, !dbg !16
  %1697 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !16
  %1698 = bitcast <4 x i32> %1697 to <8 x half>, !dbg !16
  %address.local907 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local908 = getelementptr i8, ptr addrspace(3) %address.local907, i64 2048, !dbg !16
  %1699 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local908, i32 0) #0, !dbg !16
  %1700 = bitcast <4 x i32> %1699 to <8 x half>, !dbg !16
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local910 = getelementptr i8, ptr addrspace(3) %address.local909, i64 3072, !dbg !16
  %1701 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local910, i32 0) #0, !dbg !16
  %1702 = bitcast <4 x i32> %1701 to <8 x half>, !dbg !16
  %address.local911 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local912 = getelementptr i8, ptr addrspace(3) %address.local911, i64 4096, !dbg !16
  %1703 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local912, i32 0) #0, !dbg !16
  %1704 = bitcast <4 x i32> %1703 to <8 x half>, !dbg !16
  %address.local913 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local914 = getelementptr i8, ptr addrspace(3) %address.local913, i64 5120, !dbg !16
  %1705 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local914, i32 0) #0, !dbg !16
  %1706 = bitcast <4 x i32> %1705 to <8 x half>, !dbg !16
  %address.local915 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local916 = getelementptr i8, ptr addrspace(3) %address.local915, i64 6144, !dbg !16
  %1707 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local916, i32 0) #0, !dbg !16
  %1708 = bitcast <4 x i32> %1707 to <8 x half>, !dbg !16
  %address.local917 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local918 = getelementptr i8, ptr addrspace(3) %address.local917, i64 7168, !dbg !16
  %1709 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local918, i32 0) #0, !dbg !16
  %1710 = bitcast <4 x i32> %1709 to <8 x half>, !dbg !16
  %1711 = shufflevector <8 x half> %1696, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1712 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1711, <4 x float> zeroinitializer) #9, !dbg !16
  %1713 = shufflevector <8 x half> %1698, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1714 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1713, <4 x float> %1712) #9, !dbg !16
  %1715 = shufflevector <8 x half> %1700, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1715, <4 x float> %1714) #9, !dbg !16
  %1717 = shufflevector <8 x half> %1702, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1717, <4 x float> %1716) #9, !dbg !16
  %1719 = shufflevector <8 x half> %1704, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1720 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1719, <4 x float> %1718) #9, !dbg !16
  %1721 = shufflevector <8 x half> %1706, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1722 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1721, <4 x float> %1720) #9, !dbg !16
  %1723 = shufflevector <8 x half> %1708, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1724 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1723, <4 x float> %1722) #9, !dbg !16
  %1725 = shufflevector <8 x half> %1710, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1726 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1725, <4 x float> %1724) #9, !dbg !16
  %1727 = extractelement <4 x float> %1726, i64 0, !dbg !16
  %1728 = extractelement <4 x float> %1726, i64 1, !dbg !16
  %1729 = extractelement <4 x float> %1726, i64 2, !dbg !16
  %1730 = extractelement <4 x float> %1726, i64 3, !dbg !16
  %1731 = select i1 %.not, float %1728, float %1727, !dbg !16
  %1732 = bitcast float %1731 to i32, !dbg !16
  %1733 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1732), !dbg !16
  %1734 = bitcast i32 %1733 to float, !dbg !16
  %1735 = select i1 %.not, float %1727, float %1734, !dbg !16
  %1736 = select i1 %.not, float %1734, float %1728, !dbg !16
  %1737 = select i1 %.not, float %1730, float %1729, !dbg !16
  %1738 = bitcast float %1737 to i32, !dbg !16
  %1739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1738), !dbg !16
  %1740 = bitcast i32 %1739 to float, !dbg !16
  %1741 = select i1 %.not, float %1729, float %1740, !dbg !16
  %1742 = select i1 %.not, float %1740, float %1730, !dbg !16
  %1743 = select i1 %.not1, float %1741, float %1735, !dbg !16
  %1744 = bitcast float %1743 to i32, !dbg !16
  %1745 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1744), !dbg !16
  %1746 = bitcast i32 %1745 to float, !dbg !16
  %1747 = select i1 %.not1, float %1735, float %1746, !dbg !16
  %1748 = select i1 %.not1, float %1746, float %1741, !dbg !16
  %1749 = select i1 %.not1, float %1742, float %1736, !dbg !16
  %1750 = bitcast float %1749 to i32, !dbg !16
  %1751 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1750), !dbg !16
  %1752 = bitcast i32 %1751 to float, !dbg !16
  %1753 = select i1 %.not1, float %1736, float %1752, !dbg !16
  %1754 = select i1 %.not1, float %1752, float %1742, !dbg !16
  %1755 = shufflevector <8 x half> %1696, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1755, <4 x float> zeroinitializer) #9, !dbg !16
  %1757 = shufflevector <8 x half> %1698, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1757, <4 x float> %1756) #9, !dbg !16
  %1759 = shufflevector <8 x half> %1700, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1759, <4 x float> %1758) #9, !dbg !16
  %1761 = shufflevector <8 x half> %1702, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1762 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1761, <4 x float> %1760) #9, !dbg !16
  %1763 = shufflevector <8 x half> %1704, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1763, <4 x float> %1762) #9, !dbg !16
  %1765 = shufflevector <8 x half> %1706, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1765, <4 x float> %1764) #9, !dbg !16
  %1767 = shufflevector <8 x half> %1708, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1767, <4 x float> %1766) #9, !dbg !16
  %1769 = shufflevector <8 x half> %1710, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1769, <4 x float> %1768) #9, !dbg !16
  %1771 = extractelement <4 x float> %1770, i64 0, !dbg !16
  %1772 = extractelement <4 x float> %1770, i64 1, !dbg !16
  %1773 = extractelement <4 x float> %1770, i64 2, !dbg !16
  %1774 = extractelement <4 x float> %1770, i64 3, !dbg !16
  %1775 = select i1 %.not, float %1772, float %1771, !dbg !16
  %1776 = bitcast float %1775 to i32, !dbg !16
  %1777 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1776), !dbg !16
  %1778 = bitcast i32 %1777 to float, !dbg !16
  %1779 = select i1 %.not, float %1771, float %1778, !dbg !16
  %1780 = select i1 %.not, float %1778, float %1772, !dbg !16
  %1781 = select i1 %.not, float %1774, float %1773, !dbg !16
  %1782 = bitcast float %1781 to i32, !dbg !16
  %1783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1782), !dbg !16
  %1784 = bitcast i32 %1783 to float, !dbg !16
  %1785 = select i1 %.not, float %1773, float %1784, !dbg !16
  %1786 = select i1 %.not, float %1784, float %1774, !dbg !16
  %1787 = select i1 %.not1, float %1785, float %1779, !dbg !16
  %1788 = bitcast float %1787 to i32, !dbg !16
  %1789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1788), !dbg !16
  %1790 = bitcast i32 %1789 to float, !dbg !16
  %1791 = select i1 %.not1, float %1779, float %1790, !dbg !16
  %1792 = select i1 %.not1, float %1790, float %1785, !dbg !16
  %1793 = select i1 %.not1, float %1786, float %1780, !dbg !16
  %1794 = bitcast float %1793 to i32, !dbg !16
  %1795 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1794), !dbg !16
  %1796 = bitcast i32 %1795 to float, !dbg !16
  %1797 = select i1 %.not1, float %1780, float %1796, !dbg !16
  %1798 = select i1 %.not1, float %1796, float %1786, !dbg !16
  %1799 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1711, <4 x float> zeroinitializer) #9, !dbg !16
  %1800 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1713, <4 x float> %1799) #9, !dbg !16
  %1801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1715, <4 x float> %1800) #9, !dbg !16
  %1802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1717, <4 x float> %1801) #9, !dbg !16
  %1803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1719, <4 x float> %1802) #9, !dbg !16
  %1804 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1721, <4 x float> %1803) #9, !dbg !16
  %1805 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1723, <4 x float> %1804) #9, !dbg !16
  %1806 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1725, <4 x float> %1805) #9, !dbg !16
  %1807 = extractelement <4 x float> %1806, i64 0, !dbg !16
  %1808 = extractelement <4 x float> %1806, i64 1, !dbg !16
  %1809 = extractelement <4 x float> %1806, i64 2, !dbg !16
  %1810 = extractelement <4 x float> %1806, i64 3, !dbg !16
  %1811 = select i1 %.not, float %1808, float %1807, !dbg !16
  %1812 = bitcast float %1811 to i32, !dbg !16
  %1813 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1812), !dbg !16
  %1814 = bitcast i32 %1813 to float, !dbg !16
  %1815 = select i1 %.not, float %1807, float %1814, !dbg !16
  %1816 = select i1 %.not, float %1814, float %1808, !dbg !16
  %1817 = select i1 %.not, float %1810, float %1809, !dbg !16
  %1818 = bitcast float %1817 to i32, !dbg !16
  %1819 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1818), !dbg !16
  %1820 = bitcast i32 %1819 to float, !dbg !16
  %1821 = select i1 %.not, float %1809, float %1820, !dbg !16
  %1822 = select i1 %.not, float %1820, float %1810, !dbg !16
  %1823 = select i1 %.not1, float %1821, float %1815, !dbg !16
  %1824 = bitcast float %1823 to i32, !dbg !16
  %1825 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1824), !dbg !16
  %1826 = bitcast i32 %1825 to float, !dbg !16
  %1827 = select i1 %.not1, float %1815, float %1826, !dbg !16
  %1828 = select i1 %.not1, float %1826, float %1821, !dbg !16
  %1829 = select i1 %.not1, float %1822, float %1816, !dbg !16
  %1830 = bitcast float %1829 to i32, !dbg !16
  %1831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1830), !dbg !16
  %1832 = bitcast i32 %1831 to float, !dbg !16
  %1833 = select i1 %.not1, float %1816, float %1832, !dbg !16
  %1834 = select i1 %.not1, float %1832, float %1822, !dbg !16
  %1835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1755, <4 x float> zeroinitializer) #9, !dbg !16
  %1836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1757, <4 x float> %1835) #9, !dbg !16
  %1837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1759, <4 x float> %1836) #9, !dbg !16
  %1838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1761, <4 x float> %1837) #9, !dbg !16
  %1839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1763, <4 x float> %1838) #9, !dbg !16
  %1840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1765, <4 x float> %1839) #9, !dbg !16
  %1841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1767, <4 x float> %1840) #9, !dbg !16
  %1842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1769, <4 x float> %1841) #9, !dbg !16
  %1843 = extractelement <4 x float> %1842, i64 0, !dbg !16
  %1844 = extractelement <4 x float> %1842, i64 1, !dbg !16
  %1845 = extractelement <4 x float> %1842, i64 2, !dbg !16
  %1846 = extractelement <4 x float> %1842, i64 3, !dbg !16
  %1847 = select i1 %.not, float %1844, float %1843, !dbg !16
  %1848 = bitcast float %1847 to i32, !dbg !16
  %1849 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1848), !dbg !16
  %1850 = bitcast i32 %1849 to float, !dbg !16
  %1851 = select i1 %.not, float %1843, float %1850, !dbg !16
  %1852 = select i1 %.not, float %1850, float %1844, !dbg !16
  %1853 = select i1 %.not, float %1846, float %1845, !dbg !16
  %1854 = bitcast float %1853 to i32, !dbg !16
  %1855 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1854), !dbg !16
  %1856 = bitcast i32 %1855 to float, !dbg !16
  %1857 = select i1 %.not, float %1845, float %1856, !dbg !16
  %1858 = select i1 %.not, float %1856, float %1846, !dbg !16
  %1859 = select i1 %.not1, float %1857, float %1851, !dbg !16
  %1860 = bitcast float %1859 to i32, !dbg !16
  %1861 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1860), !dbg !16
  %1862 = bitcast i32 %1861 to float, !dbg !16
  %1863 = select i1 %.not1, float %1851, float %1862, !dbg !16
  %1864 = select i1 %.not1, float %1862, float %1857, !dbg !16
  %1865 = select i1 %.not1, float %1858, float %1852, !dbg !16
  %1866 = bitcast float %1865 to i32, !dbg !16
  %1867 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1866), !dbg !16
  %1868 = bitcast i32 %1867 to float, !dbg !16
  %1869 = select i1 %.not1, float %1852, float %1868, !dbg !16
  %1870 = select i1 %.not1, float %1868, float %1858, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %1871, label %1872, !dbg !36

1871:                                             ; preds = %1622
  %address.thread919 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local920 = lshr i32 %address.thread919, 1, !dbg !15
  %address.local921 = and i32 %address.local920, 96, !dbg !15
  %address.local922 = and i32 %address.thread919, 15, !dbg !15
  %address.local923 = or disjoint i32 %address.local921, %address.local922, !dbg !15
  %address.local924 = zext nneg i32 %address.local923 to i64, !dbg !15
  %address.local925 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !15
  store float %1170, ptr addrspace(3) %address.local925, align 4, !dbg !36
  %address.local926 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !15
  %address.local927 = getelementptr i8, ptr addrspace(3) %address.local926, i64 64, !dbg !15
  store float %1171, ptr addrspace(3) %address.local927, align 4, !dbg !36
  br label %1872, !dbg !36

1872:                                             ; preds = %1871, %1622
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread928 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %1873 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !36
  %lds.const.gep1750 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %1874 = load float, ptr addrspace(3) %lds.const.gep1750, align 4, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %1875 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1628, !dbg !20
  %1876 = getelementptr i8, ptr addrspace(1) %1875, i64 128, !dbg !20
  %address.local942 = zext nneg i32 %address.thread928 to i64, !dbg !12
  %address.local943 = shl nuw nsw i64 %address.local942, 3, !dbg !11
  %address.local944 = and i64 %address.local943, 1536, !dbg !11
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local944, !dbg !11
  %address.local946 = ptrtoint ptr addrspace(3) %address.local945 to i64, !dbg !11
  %address.local947 = trunc i64 %address.local946 to i32, !dbg !11
  %1877 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local947), !dbg !20
  %1878 = zext i32 %1877 to i64, !dbg !20
  %1879 = inttoptr i64 %1878 to ptr addrspace(3), !dbg !20
  %address.local948 = shl nuw nsw i64 %address.local942, 5, !dbg !20
  %address.local949 = and i64 %address.local948, 8064, !dbg !20
  %address.local950 = shl nuw nsw i64 %address.local942, 3, !dbg !20
  %address.local951 = and i64 %address.local950, 24, !dbg !20
  %1880 = getelementptr [2 x i8], ptr addrspace(1) %1876, i64 %address.local949, !dbg !20
  %address.local953 = getelementptr [2 x i8], ptr addrspace(1) %1880, i64 %address.local951, !dbg !20
  %1881 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1882 = ptrtoint ptr addrspace(1) %address.local953 to i64, !dbg !20
  %1883 = sub i64 %1882, %1881, !dbg !20
  %dma.buffer.offset1761 = trunc i64 %1883 to i32, !dbg !20
  %1884 = trunc i64 %1881 to i32, !dbg !20
  %1885 = insertelement <4 x i32> poison, i32 %1884, i64 0, !dbg !20
  %1886 = lshr i64 %1881, 32, !dbg !20
  %1887 = trunc i64 %1886 to i32, !dbg !20
  %1888 = insertelement <4 x i32> %1885, i32 %1887, i64 1, !dbg !20
  %1889 = insertelement <4 x i32> %1888, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1762 = insertelement <4 x i32> %1889, i32 131072, i64 3, !dbg !20
  %1890 = ptrtoint ptr addrspace(3) %1879 to i32, !dbg !20
  %dma.buffer.lds1763 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1890), !dbg !20
  %1891 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1763, i32 %dma.buffer.offset1761, <4 x i32> %dma.buffer.rsrc1762) #10, !dbg !20
  %address.local954 = or disjoint i64 %address.local942, 256, !dbg !11
  %address.local955 = shl nuw nsw i64 %address.local954, 3, !dbg !11
  %address.local956 = and i64 %address.local955, 3584, !dbg !11
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local956, !dbg !11
  %address.local958 = ptrtoint ptr addrspace(3) %address.local957 to i64, !dbg !11
  %address.local959 = trunc i64 %address.local958 to i32, !dbg !11
  %1892 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local959), !dbg !20
  %1893 = zext i32 %1892 to i64, !dbg !20
  %1894 = inttoptr i64 %1893 to ptr addrspace(3), !dbg !20
  %address.local960 = shl nuw nsw i64 %address.local954, 5, !dbg !20
  %address.local961 = and i64 %address.local960, 16256, !dbg !20
  %address.local962 = shl nuw nsw i64 %address.local942, 3, !dbg !20
  %address.local963 = and i64 %address.local962, 24, !dbg !20
  %1895 = getelementptr [2 x i8], ptr addrspace(1) %1876, i64 %address.local961, !dbg !20
  %address.local965 = getelementptr [2 x i8], ptr addrspace(1) %1895, i64 %address.local963, !dbg !20
  %1896 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1897 = ptrtoint ptr addrspace(1) %address.local965 to i64, !dbg !20
  %1898 = sub i64 %1897, %1896, !dbg !20
  %dma.buffer.offset1758 = trunc i64 %1898 to i32, !dbg !20
  %1899 = trunc i64 %1896 to i32, !dbg !20
  %1900 = insertelement <4 x i32> poison, i32 %1899, i64 0, !dbg !20
  %1901 = lshr i64 %1896, 32, !dbg !20
  %1902 = trunc i64 %1901 to i32, !dbg !20
  %1903 = insertelement <4 x i32> %1900, i32 %1902, i64 1, !dbg !20
  %1904 = insertelement <4 x i32> %1903, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1759 = insertelement <4 x i32> %1904, i32 131072, i64 3, !dbg !20
  %1905 = ptrtoint ptr addrspace(3) %1894 to i32, !dbg !20
  %dma.buffer.lds1760 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1905), !dbg !20
  %1906 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1760, i32 %dma.buffer.offset1758, <4 x i32> %dma.buffer.rsrc1759) #10, !dbg !20
  %1907 = extractvalue [2 x <8 x float>] %65, 0, !dbg !36
  %1908 = shufflevector <8 x float> %1907, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1909 = insertelement <4 x float> poison, float %1873, i64 0, !dbg !36
  %1910 = shufflevector <4 x float> %1909, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1911 = fmul <4 x float> %1908, %1910, !dbg !36
  %1912 = insertelement <4 x float> poison, float %1747, i64 0, !dbg !37
  %1913 = insertelement <4 x float> %1912, float %1753, i64 1, !dbg !37
  %1914 = insertelement <4 x float> %1913, float %1748, i64 2, !dbg !37
  %1915 = insertelement <4 x float> %1914, float %1754, i64 3, !dbg !37
  %1916 = fadd <4 x float> %1911, %1915, !dbg !37
  %1917 = shufflevector <8 x float> %1907, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1918 = fmul <4 x float> %1917, %1910, !dbg !36
  %1919 = insertelement <4 x float> poison, float %1791, i64 0, !dbg !37
  %1920 = insertelement <4 x float> %1919, float %1797, i64 1, !dbg !37
  %1921 = insertelement <4 x float> %1920, float %1792, i64 2, !dbg !37
  %1922 = insertelement <4 x float> %1921, float %1798, i64 3, !dbg !37
  %1923 = fadd <4 x float> %1918, %1922, !dbg !37
  %1924 = extractvalue [2 x <8 x float>] %65, 1, !dbg !36
  %1925 = shufflevector <8 x float> %1924, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1926 = insertelement <4 x float> poison, float %1874, i64 0, !dbg !36
  %1927 = shufflevector <4 x float> %1926, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1928 = fmul <4 x float> %1925, %1927, !dbg !36
  %1929 = insertelement <4 x float> poison, float %1827, i64 0, !dbg !37
  %1930 = insertelement <4 x float> %1929, float %1833, i64 1, !dbg !37
  %1931 = insertelement <4 x float> %1930, float %1828, i64 2, !dbg !37
  %1932 = insertelement <4 x float> %1931, float %1834, i64 3, !dbg !37
  %1933 = fadd <4 x float> %1928, %1932, !dbg !37
  %1934 = shufflevector <8 x float> %1924, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1935 = fmul <4 x float> %1934, %1927, !dbg !36
  %1936 = insertelement <4 x float> poison, float %1863, i64 0, !dbg !37
  %1937 = insertelement <4 x float> %1936, float %1869, i64 1, !dbg !37
  %1938 = insertelement <4 x float> %1937, float %1864, i64 2, !dbg !37
  %1939 = insertelement <4 x float> %1938, float %1870, i64 3, !dbg !37
  %1940 = fadd <4 x float> %1935, %1939, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local966 = shl nuw nsw i32 %address.thread928, 3, !dbg !16
  %address.local967 = and i32 %address.local966, 504, !dbg !16
  %address.local968 = zext nneg i32 %address.local967 to i64, !dbg !16
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %1941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #0, !dbg !16
  %1942 = bitcast <4 x i32> %1941 to <8 x half>, !dbg !16
  %address.local970 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local970, i64 1024, !dbg !16
  %1943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #0, !dbg !16
  %1944 = bitcast <4 x i32> %1943 to <8 x half>, !dbg !16
  %address.local972 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local972, i64 2048, !dbg !16
  %1945 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !16
  %1946 = bitcast <4 x i32> %1945 to <8 x half>, !dbg !16
  %address.local974 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local975 = getelementptr i8, ptr addrspace(3) %address.local974, i64 3072, !dbg !16
  %1947 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #0, !dbg !16
  %1948 = bitcast <4 x i32> %1947 to <8 x half>, !dbg !16
  %address.local976 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local976, i64 4096, !dbg !16
  %1949 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !16
  %1950 = bitcast <4 x i32> %1949 to <8 x half>, !dbg !16
  %address.local978 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local979 = getelementptr i8, ptr addrspace(3) %address.local978, i64 5120, !dbg !16
  %1951 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #0, !dbg !16
  %1952 = bitcast <4 x i32> %1951 to <8 x half>, !dbg !16
  %address.local980 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local981 = getelementptr i8, ptr addrspace(3) %address.local980, i64 6144, !dbg !16
  %1953 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #0, !dbg !16
  %1954 = bitcast <4 x i32> %1953 to <8 x half>, !dbg !16
  %address.local982 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local983 = getelementptr i8, ptr addrspace(3) %address.local982, i64 7168, !dbg !16
  %1955 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #0, !dbg !16
  %1956 = bitcast <4 x i32> %1955 to <8 x half>, !dbg !16
  %1957 = shufflevector <8 x half> %1942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1957, <4 x float> zeroinitializer) #9, !dbg !16
  %1959 = shufflevector <8 x half> %1944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1959, <4 x float> %1958) #9, !dbg !16
  %1961 = shufflevector <8 x half> %1946, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1961, <4 x float> %1960) #9, !dbg !16
  %1963 = shufflevector <8 x half> %1948, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1963, <4 x float> %1962) #9, !dbg !16
  %1965 = shufflevector <8 x half> %1950, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1965, <4 x float> %1964) #9, !dbg !16
  %1967 = shufflevector <8 x half> %1952, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1967, <4 x float> %1966) #9, !dbg !16
  %1969 = shufflevector <8 x half> %1954, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1969, <4 x float> %1968) #9, !dbg !16
  %1971 = shufflevector <8 x half> %1956, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1971, <4 x float> %1970) #9, !dbg !16
  %1973 = extractelement <4 x float> %1972, i64 0, !dbg !16
  %1974 = extractelement <4 x float> %1972, i64 1, !dbg !16
  %1975 = extractelement <4 x float> %1972, i64 2, !dbg !16
  %1976 = extractelement <4 x float> %1972, i64 3, !dbg !16
  %1977 = select i1 %.not, float %1974, float %1973, !dbg !16
  %1978 = bitcast float %1977 to i32, !dbg !16
  %1979 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1978), !dbg !16
  %1980 = bitcast i32 %1979 to float, !dbg !16
  %1981 = select i1 %.not, float %1973, float %1980, !dbg !16
  %1982 = select i1 %.not, float %1980, float %1974, !dbg !16
  %1983 = select i1 %.not, float %1976, float %1975, !dbg !16
  %1984 = bitcast float %1983 to i32, !dbg !16
  %1985 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %1984), !dbg !16
  %1986 = bitcast i32 %1985 to float, !dbg !16
  %1987 = select i1 %.not, float %1975, float %1986, !dbg !16
  %1988 = select i1 %.not, float %1986, float %1976, !dbg !16
  %1989 = select i1 %.not1, float %1987, float %1981, !dbg !16
  %1990 = bitcast float %1989 to i32, !dbg !16
  %1991 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1990), !dbg !16
  %1992 = bitcast i32 %1991 to float, !dbg !16
  %1993 = select i1 %.not1, float %1981, float %1992, !dbg !16
  %1994 = select i1 %.not1, float %1992, float %1987, !dbg !16
  %1995 = select i1 %.not1, float %1988, float %1982, !dbg !16
  %1996 = bitcast float %1995 to i32, !dbg !16
  %1997 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %1996), !dbg !16
  %1998 = bitcast i32 %1997 to float, !dbg !16
  %1999 = select i1 %.not1, float %1982, float %1998, !dbg !16
  %2000 = select i1 %.not1, float %1998, float %1988, !dbg !16
  %2001 = shufflevector <8 x half> %1942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2001, <4 x float> zeroinitializer) #9, !dbg !16
  %2003 = shufflevector <8 x half> %1944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2003, <4 x float> %2002) #9, !dbg !16
  %2005 = shufflevector <8 x half> %1946, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2005, <4 x float> %2004) #9, !dbg !16
  %2007 = shufflevector <8 x half> %1948, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2007, <4 x float> %2006) #9, !dbg !16
  %2009 = shufflevector <8 x half> %1950, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2009, <4 x float> %2008) #9, !dbg !16
  %2011 = shufflevector <8 x half> %1952, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2011, <4 x float> %2010) #9, !dbg !16
  %2013 = shufflevector <8 x half> %1954, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2013, <4 x float> %2012) #9, !dbg !16
  %2015 = shufflevector <8 x half> %1956, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2015, <4 x float> %2014) #9, !dbg !16
  %2017 = extractelement <4 x float> %2016, i64 0, !dbg !16
  %2018 = extractelement <4 x float> %2016, i64 1, !dbg !16
  %2019 = extractelement <4 x float> %2016, i64 2, !dbg !16
  %2020 = extractelement <4 x float> %2016, i64 3, !dbg !16
  %2021 = select i1 %.not, float %2018, float %2017, !dbg !16
  %2022 = bitcast float %2021 to i32, !dbg !16
  %2023 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2022), !dbg !16
  %2024 = bitcast i32 %2023 to float, !dbg !16
  %2025 = select i1 %.not, float %2017, float %2024, !dbg !16
  %2026 = select i1 %.not, float %2024, float %2018, !dbg !16
  %2027 = select i1 %.not, float %2020, float %2019, !dbg !16
  %2028 = bitcast float %2027 to i32, !dbg !16
  %2029 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2028), !dbg !16
  %2030 = bitcast i32 %2029 to float, !dbg !16
  %2031 = select i1 %.not, float %2019, float %2030, !dbg !16
  %2032 = select i1 %.not, float %2030, float %2020, !dbg !16
  %2033 = select i1 %.not1, float %2031, float %2025, !dbg !16
  %2034 = bitcast float %2033 to i32, !dbg !16
  %2035 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2034), !dbg !16
  %2036 = bitcast i32 %2035 to float, !dbg !16
  %2037 = select i1 %.not1, float %2025, float %2036, !dbg !16
  %2038 = select i1 %.not1, float %2036, float %2031, !dbg !16
  %2039 = select i1 %.not1, float %2032, float %2026, !dbg !16
  %2040 = bitcast float %2039 to i32, !dbg !16
  %2041 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2040), !dbg !16
  %2042 = bitcast i32 %2041 to float, !dbg !16
  %2043 = select i1 %.not1, float %2026, float %2042, !dbg !16
  %2044 = select i1 %.not1, float %2042, float %2032, !dbg !16
  %2045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1957, <4 x float> zeroinitializer) #9, !dbg !16
  %2046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1959, <4 x float> %2045) #9, !dbg !16
  %2047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1961, <4 x float> %2046) #9, !dbg !16
  %2048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1963, <4 x float> %2047) #9, !dbg !16
  %2049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1965, <4 x float> %2048) #9, !dbg !16
  %2050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1967, <4 x float> %2049) #9, !dbg !16
  %2051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1969, <4 x float> %2050) #9, !dbg !16
  %2052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1971, <4 x float> %2051) #9, !dbg !16
  %2053 = extractelement <4 x float> %2052, i64 0, !dbg !16
  %2054 = extractelement <4 x float> %2052, i64 1, !dbg !16
  %2055 = extractelement <4 x float> %2052, i64 2, !dbg !16
  %2056 = extractelement <4 x float> %2052, i64 3, !dbg !16
  %2057 = select i1 %.not, float %2054, float %2053, !dbg !16
  %2058 = bitcast float %2057 to i32, !dbg !16
  %2059 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2058), !dbg !16
  %2060 = bitcast i32 %2059 to float, !dbg !16
  %2061 = select i1 %.not, float %2053, float %2060, !dbg !16
  %2062 = select i1 %.not, float %2060, float %2054, !dbg !16
  %2063 = select i1 %.not, float %2056, float %2055, !dbg !16
  %2064 = bitcast float %2063 to i32, !dbg !16
  %2065 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2064), !dbg !16
  %2066 = bitcast i32 %2065 to float, !dbg !16
  %2067 = select i1 %.not, float %2055, float %2066, !dbg !16
  %2068 = select i1 %.not, float %2066, float %2056, !dbg !16
  %2069 = select i1 %.not1, float %2067, float %2061, !dbg !16
  %2070 = bitcast float %2069 to i32, !dbg !16
  %2071 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2070), !dbg !16
  %2072 = bitcast i32 %2071 to float, !dbg !16
  %2073 = select i1 %.not1, float %2061, float %2072, !dbg !16
  %2074 = select i1 %.not1, float %2072, float %2067, !dbg !16
  %2075 = select i1 %.not1, float %2068, float %2062, !dbg !16
  %2076 = bitcast float %2075 to i32, !dbg !16
  %2077 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2076), !dbg !16
  %2078 = bitcast i32 %2077 to float, !dbg !16
  %2079 = select i1 %.not1, float %2062, float %2078, !dbg !16
  %2080 = select i1 %.not1, float %2078, float %2068, !dbg !16
  %2081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2001, <4 x float> zeroinitializer) #9, !dbg !16
  %2082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2003, <4 x float> %2081) #9, !dbg !16
  %2083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2005, <4 x float> %2082) #9, !dbg !16
  %2084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2007, <4 x float> %2083) #9, !dbg !16
  %2085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2009, <4 x float> %2084) #9, !dbg !16
  %2086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2011, <4 x float> %2085) #9, !dbg !16
  %2087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2013, <4 x float> %2086) #9, !dbg !16
  %2088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2015, <4 x float> %2087) #9, !dbg !16
  %2089 = extractelement <4 x float> %2088, i64 0, !dbg !16
  %2090 = extractelement <4 x float> %2088, i64 1, !dbg !16
  %2091 = extractelement <4 x float> %2088, i64 2, !dbg !16
  %2092 = extractelement <4 x float> %2088, i64 3, !dbg !16
  %2093 = select i1 %.not, float %2090, float %2089, !dbg !16
  %2094 = bitcast float %2093 to i32, !dbg !16
  %2095 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2094), !dbg !16
  %2096 = bitcast i32 %2095 to float, !dbg !16
  %2097 = select i1 %.not, float %2089, float %2096, !dbg !16
  %2098 = select i1 %.not, float %2096, float %2090, !dbg !16
  %2099 = select i1 %.not, float %2092, float %2091, !dbg !16
  %2100 = bitcast float %2099 to i32, !dbg !16
  %2101 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2100), !dbg !16
  %2102 = bitcast i32 %2101 to float, !dbg !16
  %2103 = select i1 %.not, float %2091, float %2102, !dbg !16
  %2104 = select i1 %.not, float %2102, float %2092, !dbg !16
  %2105 = select i1 %.not1, float %2103, float %2097, !dbg !16
  %2106 = bitcast float %2105 to i32, !dbg !16
  %2107 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2106), !dbg !16
  %2108 = bitcast i32 %2107 to float, !dbg !16
  %2109 = select i1 %.not1, float %2097, float %2108, !dbg !16
  %2110 = select i1 %.not1, float %2108, float %2103, !dbg !16
  %2111 = select i1 %.not1, float %2104, float %2098, !dbg !16
  %2112 = bitcast float %2111 to i32, !dbg !16
  %2113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2112), !dbg !16
  %2114 = bitcast i32 %2113 to float, !dbg !16
  %2115 = select i1 %.not1, float %2098, float %2114, !dbg !16
  %2116 = select i1 %.not1, float %2114, float %2104, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %2117, label %2118, !dbg !36

2117:                                             ; preds = %1872
  %address.thread984 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local985 = lshr i32 %address.thread984, 1, !dbg !15
  %address.local986 = and i32 %address.local985, 96, !dbg !15
  %address.local987 = and i32 %address.thread984, 15, !dbg !15
  %address.local988 = or disjoint i32 %address.local986, %address.local987, !dbg !15
  %address.local989 = zext nneg i32 %address.local988 to i64, !dbg !15
  %address.local990 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !15
  store float %1170, ptr addrspace(3) %address.local990, align 4, !dbg !36
  %address.local991 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !15
  %address.local992 = getelementptr i8, ptr addrspace(3) %address.local991, i64 64, !dbg !15
  store float %1171, ptr addrspace(3) %address.local992, align 4, !dbg !36
  br label %2118, !dbg !36

2118:                                             ; preds = %2117, %1872
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread993 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %2119 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !36
  %lds.const.gep1752 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %2120 = load float, ptr addrspace(3) %lds.const.gep1752, align 4, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %2121 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1628, !dbg !20
  %2122 = getelementptr i8, ptr addrspace(1) %2121, i64 192, !dbg !20
  %address.local1007 = zext nneg i32 %address.thread993 to i64, !dbg !12
  %address.local1008 = shl nuw nsw i64 %address.local1007, 3, !dbg !11
  %address.local1009 = and i64 %address.local1008, 1536, !dbg !11
  %address.local1010 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1009, !dbg !11
  %address.local1011 = ptrtoint ptr addrspace(3) %address.local1010 to i64, !dbg !11
  %address.local1012 = trunc i64 %address.local1011 to i32, !dbg !11
  %2123 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1012), !dbg !20
  %2124 = zext i32 %2123 to i64, !dbg !20
  %2125 = inttoptr i64 %2124 to ptr addrspace(3), !dbg !20
  %address.local1013 = shl nuw nsw i64 %address.local1007, 5, !dbg !20
  %address.local1014 = and i64 %address.local1013, 8064, !dbg !20
  %address.local1015 = shl nuw nsw i64 %address.local1007, 3, !dbg !20
  %address.local1016 = and i64 %address.local1015, 24, !dbg !20
  %2126 = getelementptr [2 x i8], ptr addrspace(1) %2122, i64 %address.local1014, !dbg !20
  %address.local1018 = getelementptr [2 x i8], ptr addrspace(1) %2126, i64 %address.local1016, !dbg !20
  %2127 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %2128 = ptrtoint ptr addrspace(1) %address.local1018 to i64, !dbg !20
  %2129 = sub i64 %2128, %2127, !dbg !20
  %dma.buffer.offset1755 = trunc i64 %2129 to i32, !dbg !20
  %2130 = trunc i64 %2127 to i32, !dbg !20
  %2131 = insertelement <4 x i32> poison, i32 %2130, i64 0, !dbg !20
  %2132 = lshr i64 %2127, 32, !dbg !20
  %2133 = trunc i64 %2132 to i32, !dbg !20
  %2134 = insertelement <4 x i32> %2131, i32 %2133, i64 1, !dbg !20
  %2135 = insertelement <4 x i32> %2134, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1756 = insertelement <4 x i32> %2135, i32 131072, i64 3, !dbg !20
  %2136 = ptrtoint ptr addrspace(3) %2125 to i32, !dbg !20
  %dma.buffer.lds1757 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %2136), !dbg !20
  %2137 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1757, i32 %dma.buffer.offset1755, <4 x i32> %dma.buffer.rsrc1756) #10, !dbg !20
  %address.local1019 = or disjoint i64 %address.local1007, 256, !dbg !11
  %address.local1020 = shl nuw nsw i64 %address.local1019, 3, !dbg !11
  %address.local1021 = and i64 %address.local1020, 3584, !dbg !11
  %address.local1022 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1021, !dbg !11
  %address.local1023 = ptrtoint ptr addrspace(3) %address.local1022 to i64, !dbg !11
  %address.local1024 = trunc i64 %address.local1023 to i32, !dbg !11
  %2138 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1024), !dbg !20
  %2139 = zext i32 %2138 to i64, !dbg !20
  %2140 = inttoptr i64 %2139 to ptr addrspace(3), !dbg !20
  %address.local1025 = shl nuw nsw i64 %address.local1019, 5, !dbg !20
  %address.local1026 = and i64 %address.local1025, 16256, !dbg !20
  %address.local1027 = shl nuw nsw i64 %address.local1007, 3, !dbg !20
  %address.local1028 = and i64 %address.local1027, 24, !dbg !20
  %2141 = getelementptr [2 x i8], ptr addrspace(1) %2122, i64 %address.local1026, !dbg !20
  %address.local1030 = getelementptr [2 x i8], ptr addrspace(1) %2141, i64 %address.local1028, !dbg !20
  %2142 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %2143 = ptrtoint ptr addrspace(1) %address.local1030 to i64, !dbg !20
  %2144 = sub i64 %2143, %2142, !dbg !20
  %dma.buffer.offset = trunc i64 %2144 to i32, !dbg !20
  %2145 = trunc i64 %2142 to i32, !dbg !20
  %2146 = insertelement <4 x i32> poison, i32 %2145, i64 0, !dbg !20
  %2147 = lshr i64 %2142, 32, !dbg !20
  %2148 = trunc i64 %2147 to i32, !dbg !20
  %2149 = insertelement <4 x i32> %2146, i32 %2148, i64 1, !dbg !20
  %2150 = insertelement <4 x i32> %2149, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc = insertelement <4 x i32> %2150, i32 131072, i64 3, !dbg !20
  %2151 = ptrtoint ptr addrspace(3) %2140 to i32, !dbg !20
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %2151), !dbg !20
  %2152 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !20
  %2153 = extractvalue [2 x <8 x float>] %66, 0, !dbg !36
  %2154 = shufflevector <8 x float> %2153, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2155 = insertelement <4 x float> poison, float %2119, i64 0, !dbg !36
  %2156 = shufflevector <4 x float> %2155, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2157 = fmul <4 x float> %2154, %2156, !dbg !36
  %2158 = insertelement <4 x float> poison, float %1993, i64 0, !dbg !37
  %2159 = insertelement <4 x float> %2158, float %1999, i64 1, !dbg !37
  %2160 = insertelement <4 x float> %2159, float %1994, i64 2, !dbg !37
  %2161 = insertelement <4 x float> %2160, float %2000, i64 3, !dbg !37
  %2162 = fadd <4 x float> %2157, %2161, !dbg !37
  %2163 = shufflevector <8 x float> %2153, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2164 = fmul <4 x float> %2163, %2156, !dbg !36
  %2165 = insertelement <4 x float> poison, float %2037, i64 0, !dbg !37
  %2166 = insertelement <4 x float> %2165, float %2043, i64 1, !dbg !37
  %2167 = insertelement <4 x float> %2166, float %2038, i64 2, !dbg !37
  %2168 = insertelement <4 x float> %2167, float %2044, i64 3, !dbg !37
  %2169 = fadd <4 x float> %2164, %2168, !dbg !37
  %2170 = extractvalue [2 x <8 x float>] %66, 1, !dbg !36
  %2171 = shufflevector <8 x float> %2170, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2172 = insertelement <4 x float> poison, float %2120, i64 0, !dbg !36
  %2173 = shufflevector <4 x float> %2172, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2174 = fmul <4 x float> %2171, %2173, !dbg !36
  %2175 = insertelement <4 x float> poison, float %2073, i64 0, !dbg !37
  %2176 = insertelement <4 x float> %2175, float %2079, i64 1, !dbg !37
  %2177 = insertelement <4 x float> %2176, float %2074, i64 2, !dbg !37
  %2178 = insertelement <4 x float> %2177, float %2080, i64 3, !dbg !37
  %2179 = fadd <4 x float> %2174, %2178, !dbg !37
  %2180 = shufflevector <8 x float> %2170, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2181 = fmul <4 x float> %2180, %2173, !dbg !36
  %2182 = insertelement <4 x float> poison, float %2109, i64 0, !dbg !37
  %2183 = insertelement <4 x float> %2182, float %2115, i64 1, !dbg !37
  %2184 = insertelement <4 x float> %2183, float %2110, i64 2, !dbg !37
  %2185 = insertelement <4 x float> %2184, float %2116, i64 3, !dbg !37
  %2186 = fadd <4 x float> %2181, %2185, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local1031 = shl nuw nsw i32 %address.thread993, 3, !dbg !16
  %address.local1032 = and i32 %address.local1031, 504, !dbg !16
  %address.local1033 = zext nneg i32 %address.local1032 to i64, !dbg !16
  %address.local1034 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %2187 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !16
  %2188 = bitcast <4 x i32> %2187 to <8 x half>, !dbg !16
  %address.local1035 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1036 = getelementptr i8, ptr addrspace(3) %address.local1035, i64 1024, !dbg !16
  %2189 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1036, i32 0) #0, !dbg !16
  %2190 = bitcast <4 x i32> %2189 to <8 x half>, !dbg !16
  %address.local1037 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1038 = getelementptr i8, ptr addrspace(3) %address.local1037, i64 2048, !dbg !16
  %2191 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1038, i32 0) #0, !dbg !16
  %2192 = bitcast <4 x i32> %2191 to <8 x half>, !dbg !16
  %address.local1039 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 3072, !dbg !16
  %2193 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !16
  %2194 = bitcast <4 x i32> %2193 to <8 x half>, !dbg !16
  %address.local1041 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1042 = getelementptr i8, ptr addrspace(3) %address.local1041, i64 4096, !dbg !16
  %2195 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1042, i32 0) #0, !dbg !16
  %2196 = bitcast <4 x i32> %2195 to <8 x half>, !dbg !16
  %address.local1043 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1043, i64 5120, !dbg !16
  %2197 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !16
  %2198 = bitcast <4 x i32> %2197 to <8 x half>, !dbg !16
  %address.local1045 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1046 = getelementptr i8, ptr addrspace(3) %address.local1045, i64 6144, !dbg !16
  %2199 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1046, i32 0) #0, !dbg !16
  %2200 = bitcast <4 x i32> %2199 to <8 x half>, !dbg !16
  %address.local1047 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1047, i64 7168, !dbg !16
  %2201 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !16
  %2202 = bitcast <4 x i32> %2201 to <8 x half>, !dbg !16
  %2203 = shufflevector <8 x half> %2188, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2203, <4 x float> zeroinitializer) #9, !dbg !16
  %2205 = shufflevector <8 x half> %2190, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2205, <4 x float> %2204) #9, !dbg !16
  %2207 = shufflevector <8 x half> %2192, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2207, <4 x float> %2206) #9, !dbg !16
  %2209 = shufflevector <8 x half> %2194, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2209, <4 x float> %2208) #9, !dbg !16
  %2211 = shufflevector <8 x half> %2196, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2211, <4 x float> %2210) #9, !dbg !16
  %2213 = shufflevector <8 x half> %2198, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2213, <4 x float> %2212) #9, !dbg !16
  %2215 = shufflevector <8 x half> %2200, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2215, <4 x float> %2214) #9, !dbg !16
  %2217 = shufflevector <8 x half> %2202, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2217, <4 x float> %2216) #9, !dbg !16
  %2219 = extractelement <4 x float> %2218, i64 0, !dbg !16
  %2220 = extractelement <4 x float> %2218, i64 1, !dbg !16
  %2221 = extractelement <4 x float> %2218, i64 2, !dbg !16
  %2222 = extractelement <4 x float> %2218, i64 3, !dbg !16
  %2223 = select i1 %.not, float %2220, float %2219, !dbg !16
  %2224 = bitcast float %2223 to i32, !dbg !16
  %2225 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2224), !dbg !16
  %2226 = bitcast i32 %2225 to float, !dbg !16
  %2227 = select i1 %.not, float %2219, float %2226, !dbg !16
  %2228 = select i1 %.not, float %2226, float %2220, !dbg !16
  %2229 = select i1 %.not, float %2222, float %2221, !dbg !16
  %2230 = bitcast float %2229 to i32, !dbg !16
  %2231 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2230), !dbg !16
  %2232 = bitcast i32 %2231 to float, !dbg !16
  %2233 = select i1 %.not, float %2221, float %2232, !dbg !16
  %2234 = select i1 %.not, float %2232, float %2222, !dbg !16
  %2235 = select i1 %.not1, float %2233, float %2227, !dbg !16
  %2236 = bitcast float %2235 to i32, !dbg !16
  %2237 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2236), !dbg !16
  %2238 = bitcast i32 %2237 to float, !dbg !16
  %2239 = select i1 %.not1, float %2227, float %2238, !dbg !16
  %2240 = select i1 %.not1, float %2238, float %2233, !dbg !16
  %2241 = select i1 %.not1, float %2234, float %2228, !dbg !16
  %2242 = bitcast float %2241 to i32, !dbg !16
  %2243 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2242), !dbg !16
  %2244 = bitcast i32 %2243 to float, !dbg !16
  %2245 = select i1 %.not1, float %2228, float %2244, !dbg !16
  %2246 = select i1 %.not1, float %2244, float %2234, !dbg !16
  %2247 = shufflevector <8 x half> %2188, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2247, <4 x float> zeroinitializer) #9, !dbg !16
  %2249 = shufflevector <8 x half> %2190, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2249, <4 x float> %2248) #9, !dbg !16
  %2251 = shufflevector <8 x half> %2192, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2251, <4 x float> %2250) #9, !dbg !16
  %2253 = shufflevector <8 x half> %2194, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2253, <4 x float> %2252) #9, !dbg !16
  %2255 = shufflevector <8 x half> %2196, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2255, <4 x float> %2254) #9, !dbg !16
  %2257 = shufflevector <8 x half> %2198, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2257, <4 x float> %2256) #9, !dbg !16
  %2259 = shufflevector <8 x half> %2200, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2259, <4 x float> %2258) #9, !dbg !16
  %2261 = shufflevector <8 x half> %2202, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2261, <4 x float> %2260) #9, !dbg !16
  %2263 = extractelement <4 x float> %2262, i64 0, !dbg !16
  %2264 = extractelement <4 x float> %2262, i64 1, !dbg !16
  %2265 = extractelement <4 x float> %2262, i64 2, !dbg !16
  %2266 = extractelement <4 x float> %2262, i64 3, !dbg !16
  %2267 = select i1 %.not, float %2264, float %2263, !dbg !16
  %2268 = bitcast float %2267 to i32, !dbg !16
  %2269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2268), !dbg !16
  %2270 = bitcast i32 %2269 to float, !dbg !16
  %2271 = select i1 %.not, float %2263, float %2270, !dbg !16
  %2272 = select i1 %.not, float %2270, float %2264, !dbg !16
  %2273 = select i1 %.not, float %2266, float %2265, !dbg !16
  %2274 = bitcast float %2273 to i32, !dbg !16
  %2275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2274), !dbg !16
  %2276 = bitcast i32 %2275 to float, !dbg !16
  %2277 = select i1 %.not, float %2265, float %2276, !dbg !16
  %2278 = select i1 %.not, float %2276, float %2266, !dbg !16
  %2279 = select i1 %.not1, float %2277, float %2271, !dbg !16
  %2280 = bitcast float %2279 to i32, !dbg !16
  %2281 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2280), !dbg !16
  %2282 = bitcast i32 %2281 to float, !dbg !16
  %2283 = select i1 %.not1, float %2271, float %2282, !dbg !16
  %2284 = select i1 %.not1, float %2282, float %2277, !dbg !16
  %2285 = select i1 %.not1, float %2278, float %2272, !dbg !16
  %2286 = bitcast float %2285 to i32, !dbg !16
  %2287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2286), !dbg !16
  %2288 = bitcast i32 %2287 to float, !dbg !16
  %2289 = select i1 %.not1, float %2272, float %2288, !dbg !16
  %2290 = select i1 %.not1, float %2288, float %2278, !dbg !16
  %2291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2203, <4 x float> zeroinitializer) #9, !dbg !16
  %2292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2205, <4 x float> %2291) #9, !dbg !16
  %2293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2207, <4 x float> %2292) #9, !dbg !16
  %2294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2209, <4 x float> %2293) #9, !dbg !16
  %2295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2211, <4 x float> %2294) #9, !dbg !16
  %2296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2213, <4 x float> %2295) #9, !dbg !16
  %2297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2215, <4 x float> %2296) #9, !dbg !16
  %2298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2217, <4 x float> %2297) #9, !dbg !16
  %2299 = extractelement <4 x float> %2298, i64 0, !dbg !16
  %2300 = extractelement <4 x float> %2298, i64 1, !dbg !16
  %2301 = extractelement <4 x float> %2298, i64 2, !dbg !16
  %2302 = extractelement <4 x float> %2298, i64 3, !dbg !16
  %2303 = select i1 %.not, float %2300, float %2299, !dbg !16
  %2304 = bitcast float %2303 to i32, !dbg !16
  %2305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2304), !dbg !16
  %2306 = bitcast i32 %2305 to float, !dbg !16
  %2307 = select i1 %.not, float %2299, float %2306, !dbg !16
  %2308 = select i1 %.not, float %2306, float %2300, !dbg !16
  %2309 = select i1 %.not, float %2302, float %2301, !dbg !16
  %2310 = bitcast float %2309 to i32, !dbg !16
  %2311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2310), !dbg !16
  %2312 = bitcast i32 %2311 to float, !dbg !16
  %2313 = select i1 %.not, float %2301, float %2312, !dbg !16
  %2314 = select i1 %.not, float %2312, float %2302, !dbg !16
  %2315 = select i1 %.not1, float %2313, float %2307, !dbg !16
  %2316 = bitcast float %2315 to i32, !dbg !16
  %2317 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2316), !dbg !16
  %2318 = bitcast i32 %2317 to float, !dbg !16
  %2319 = select i1 %.not1, float %2307, float %2318, !dbg !16
  %2320 = select i1 %.not1, float %2318, float %2313, !dbg !16
  %2321 = select i1 %.not1, float %2314, float %2308, !dbg !16
  %2322 = bitcast float %2321 to i32, !dbg !16
  %2323 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2322), !dbg !16
  %2324 = bitcast i32 %2323 to float, !dbg !16
  %2325 = select i1 %.not1, float %2308, float %2324, !dbg !16
  %2326 = select i1 %.not1, float %2324, float %2314, !dbg !16
  %2327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2247, <4 x float> zeroinitializer) #9, !dbg !16
  %2328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2249, <4 x float> %2327) #9, !dbg !16
  %2329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2251, <4 x float> %2328) #9, !dbg !16
  %2330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2253, <4 x float> %2329) #9, !dbg !16
  %2331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2255, <4 x float> %2330) #9, !dbg !16
  %2332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2257, <4 x float> %2331) #9, !dbg !16
  %2333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2259, <4 x float> %2332) #9, !dbg !16
  %2334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2261, <4 x float> %2333) #9, !dbg !16
  %2335 = extractelement <4 x float> %2334, i64 0, !dbg !16
  %2336 = extractelement <4 x float> %2334, i64 1, !dbg !16
  %2337 = extractelement <4 x float> %2334, i64 2, !dbg !16
  %2338 = extractelement <4 x float> %2334, i64 3, !dbg !16
  %2339 = select i1 %.not, float %2336, float %2335, !dbg !16
  %2340 = bitcast float %2339 to i32, !dbg !16
  %2341 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2340), !dbg !16
  %2342 = bitcast i32 %2341 to float, !dbg !16
  %2343 = select i1 %.not, float %2335, float %2342, !dbg !16
  %2344 = select i1 %.not, float %2342, float %2336, !dbg !16
  %2345 = select i1 %.not, float %2338, float %2337, !dbg !16
  %2346 = bitcast float %2345 to i32, !dbg !16
  %2347 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1113, i32 %2346), !dbg !16
  %2348 = bitcast i32 %2347 to float, !dbg !16
  %2349 = select i1 %.not, float %2337, float %2348, !dbg !16
  %2350 = select i1 %.not, float %2348, float %2338, !dbg !16
  %2351 = select i1 %.not1, float %2349, float %2343, !dbg !16
  %2352 = bitcast float %2351 to i32, !dbg !16
  %2353 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2352), !dbg !16
  %2354 = bitcast i32 %2353 to float, !dbg !16
  %2355 = select i1 %.not1, float %2343, float %2354, !dbg !16
  %2356 = select i1 %.not1, float %2354, float %2349, !dbg !16
  %2357 = select i1 %.not1, float %2350, float %2344, !dbg !16
  %2358 = bitcast float %2357 to i32, !dbg !16
  %2359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1121, i32 %2358), !dbg !16
  %2360 = bitcast i32 %2359 to float, !dbg !16
  %2361 = select i1 %.not1, float %2344, float %2360, !dbg !16
  %2362 = select i1 %.not1, float %2360, float %2350, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %2363, label %2364, !dbg !36

2363:                                             ; preds = %2118
  %address.thread1049 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local1050 = lshr i32 %address.thread1049, 1, !dbg !15
  %address.local1051 = and i32 %address.local1050, 96, !dbg !15
  %address.local1052 = and i32 %address.thread1049, 15, !dbg !15
  %address.local1053 = or disjoint i32 %address.local1051, %address.local1052, !dbg !15
  %address.local1054 = zext nneg i32 %address.local1053 to i64, !dbg !15
  %address.local1055 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !15
  store float %1170, ptr addrspace(3) %address.local1055, align 4, !dbg !36
  %address.local1056 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !15
  %address.local1057 = getelementptr i8, ptr addrspace(3) %address.local1056, i64 64, !dbg !15
  store float %1171, ptr addrspace(3) %address.local1057, align 4, !dbg !36
  br label %2364, !dbg !36

2364:                                             ; preds = %2363, %2118
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %2365 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !36
  %lds.const.gep1754 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %2366 = load float, ptr addrspace(3) %lds.const.gep1754, align 4, !dbg !36
  %2367 = extractvalue [2 x <8 x float>] %67, 0, !dbg !36
  %2368 = shufflevector <8 x float> %2367, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2369 = insertelement <4 x float> poison, float %2365, i64 0, !dbg !36
  %2370 = shufflevector <4 x float> %2369, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2371 = fmul <4 x float> %2368, %2370, !dbg !36
  %2372 = insertelement <4 x float> poison, float %2239, i64 0, !dbg !37
  %2373 = insertelement <4 x float> %2372, float %2245, i64 1, !dbg !37
  %2374 = insertelement <4 x float> %2373, float %2240, i64 2, !dbg !37
  %2375 = insertelement <4 x float> %2374, float %2246, i64 3, !dbg !37
  %2376 = fadd <4 x float> %2371, %2375, !dbg !37
  %2377 = shufflevector <8 x float> %2367, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2378 = fmul <4 x float> %2377, %2370, !dbg !36
  %2379 = insertelement <4 x float> poison, float %2283, i64 0, !dbg !37
  %2380 = insertelement <4 x float> %2379, float %2289, i64 1, !dbg !37
  %2381 = insertelement <4 x float> %2380, float %2284, i64 2, !dbg !37
  %2382 = insertelement <4 x float> %2381, float %2290, i64 3, !dbg !37
  %2383 = fadd <4 x float> %2378, %2382, !dbg !37
  %2384 = extractvalue [2 x <8 x float>] %67, 1, !dbg !36
  %2385 = shufflevector <8 x float> %2384, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2386 = insertelement <4 x float> poison, float %2366, i64 0, !dbg !36
  %2387 = shufflevector <4 x float> %2386, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2388 = fmul <4 x float> %2385, %2387, !dbg !36
  %2389 = insertelement <4 x float> poison, float %2319, i64 0, !dbg !37
  %2390 = insertelement <4 x float> %2389, float %2325, i64 1, !dbg !37
  %2391 = insertelement <4 x float> %2390, float %2320, i64 2, !dbg !37
  %2392 = insertelement <4 x float> %2391, float %2326, i64 3, !dbg !37
  %2393 = fadd <4 x float> %2388, %2392, !dbg !37
  %2394 = shufflevector <8 x float> %2384, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2395 = fmul <4 x float> %2394, %2387, !dbg !36
  %2396 = insertelement <4 x float> poison, float %2355, i64 0, !dbg !37
  %2397 = insertelement <4 x float> %2396, float %2361, i64 1, !dbg !37
  %2398 = insertelement <4 x float> %2397, float %2356, i64 2, !dbg !37
  %2399 = insertelement <4 x float> %2398, float %2362, i64 3, !dbg !37
  %2400 = fadd <4 x float> %2395, %2399, !dbg !37
  %2401 = shufflevector <4 x float> %1670, <4 x float> %1677, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2402 = insertvalue [2 x <8 x float>] poison, <8 x float> %2401, 0, !dbg !7
  %2403 = shufflevector <4 x float> %1687, <4 x float> %1694, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2404 = insertvalue [2 x <8 x float>] %2402, <8 x float> %2403, 1, !dbg !7
  %2405 = insertelement <1 x float> poison, float %1435, i64 0, !dbg !7
  %2406 = insertvalue [2 x <1 x float>] poison, <1 x float> %2405, 0, !dbg !7
  %2407 = insertelement <1 x float> poison, float %1444, i64 0, !dbg !7
  %2408 = insertvalue [2 x <1 x float>] %2406, <1 x float> %2407, 1, !dbg !7
  %2409 = insertvalue [2 x <1 x float>] poison, <1 x float> %1128, 0, !dbg !7
  %2410 = insertvalue [2 x <1 x float>] %2409, <1 x float> %1141, 1, !dbg !7
  %2411 = shufflevector <4 x float> %1916, <4 x float> %1923, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2412 = insertvalue [2 x <8 x float>] poison, <8 x float> %2411, 0, !dbg !7
  %2413 = shufflevector <4 x float> %1933, <4 x float> %1940, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2414 = insertvalue [2 x <8 x float>] %2412, <8 x float> %2413, 1, !dbg !7
  %2415 = shufflevector <4 x float> %2162, <4 x float> %2169, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2416 = insertvalue [2 x <8 x float>] poison, <8 x float> %2415, 0, !dbg !7
  %2417 = shufflevector <4 x float> %2179, <4 x float> %2186, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2418 = insertvalue [2 x <8 x float>] %2416, <8 x float> %2417, 1, !dbg !7
  %2419 = shufflevector <4 x float> %2376, <4 x float> %2383, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2420 = insertvalue [2 x <8 x float>] poison, <8 x float> %2419, 0, !dbg !7
  %2421 = shufflevector <4 x float> %2393, <4 x float> %2400, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2422 = insertvalue [2 x <8 x float>] %2420, <8 x float> %2421, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %2423 = add nuw nsw i64 %61, 1, !dbg !7
  br label %60, !dbg !7

2424:                                             ; preds = %60
  %2425 = extractvalue [2 x <1 x float>] %63, 0, !dbg !21
  %2426 = shufflevector <1 x float> %2425, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %2427 = extractvalue [2 x <1 x float>] %63, 1, !dbg !21
  %2428 = shufflevector <1 x float> %2427, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %2429 = extractvalue [2 x <8 x float>] %62, 1, !dbg !21
  %2430 = shufflevector <8 x float> %2429, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2428, !dbg !21
  %normalized = fmul <4 x float> %2430, %row.reciprocal, !dbg !21
  %2431 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !22
  %2432 = extractelement <4 x half> %2431, i64 3, !dbg !22
  %2433 = extractelement <4 x half> %2431, i64 2, !dbg !22
  %2434 = extractelement <4 x half> %2431, i64 1, !dbg !22
  %2435 = extractelement <4 x half> %2431, i64 0, !dbg !22
  %2436 = shufflevector <8 x float> %2429, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized68 = fmul <4 x float> %2436, %row.reciprocal, !dbg !21
  %2437 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !22
  %2438 = extractelement <4 x half> %2437, i64 3, !dbg !22
  %2439 = extractelement <4 x half> %2437, i64 2, !dbg !22
  %2440 = extractelement <4 x half> %2437, i64 1, !dbg !22
  %2441 = extractelement <4 x half> %2437, i64 0, !dbg !22
  %2442 = extractvalue [2 x <8 x float>] %62, 0, !dbg !21
  %2443 = shufflevector <8 x float> %2442, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2426, !dbg !21
  %normalized70 = fmul <4 x float> %2443, %row.reciprocal69, !dbg !21
  %2444 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !22
  %2445 = extractelement <4 x half> %2444, i64 3, !dbg !22
  %2446 = extractelement <4 x half> %2444, i64 2, !dbg !22
  %2447 = extractelement <4 x half> %2444, i64 1, !dbg !22
  %2448 = extractelement <4 x half> %2444, i64 0, !dbg !22
  %2449 = shufflevector <8 x float> %2442, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized71 = fmul <4 x float> %2449, %row.reciprocal69, !dbg !21
  %2450 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !22
  %2451 = extractelement <4 x half> %2450, i64 3, !dbg !22
  %2452 = extractelement <4 x half> %2450, i64 2, !dbg !22
  %2453 = extractelement <4 x half> %2450, i64 1, !dbg !22
  %2454 = extractelement <4 x half> %2450, i64 0, !dbg !22
  %2455 = add nsw i64 %15, %25, !dbg !23
  %2456 = shl i64 %2455, 19, !dbg !23
  %2457 = and i64 %2456, 16252928, !dbg !23
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
  %address.local1083 = add nuw nsw i64 %2457, %address.local1082, !dbg !23
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %2458 = lshr i64 %address.local1073, 2, !dbg !12
  %address.local1087 = and i64 %2458, 12, !dbg !12
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1087, !dbg !23
  store half %2454, ptr addrspace(1) %address.local1088, align 2, !dbg !23
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1090 = or disjoint i64 %address.local1087, 1, !dbg !13
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1090, !dbg !23
  store half %2453, ptr addrspace(1) %address.local1091, align 2, !dbg !23
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1093 = or disjoint i64 %address.local1087, 2, !dbg !13
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1093, !dbg !23
  store half %2452, ptr addrspace(1) %address.local1094, align 2, !dbg !23
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1096 = or disjoint i64 %address.local1087, 3, !dbg !13
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1096, !dbg !23
  store half %2451, ptr addrspace(1) %address.local1097, align 2, !dbg !23
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1099 = or disjoint i64 %address.local1087, 16, !dbg !13
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1098, i64 %address.local1099, !dbg !23
  store half %2448, ptr addrspace(1) %address.local1100, align 2, !dbg !23
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1102 = or disjoint i64 %address.local1087, 17, !dbg !13
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !23
  store half %2447, ptr addrspace(1) %address.local1103, align 2, !dbg !23
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1105 = or disjoint i64 %address.local1087, 18, !dbg !13
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !23
  store half %2446, ptr addrspace(1) %address.local1106, align 2, !dbg !23
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1108 = or disjoint i64 %address.local1087, 19, !dbg !13
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !23
  store half %2445, ptr addrspace(1) %address.local1109, align 2, !dbg !23
  %address.local1110 = shl i64 %address.local1081, 7, !dbg !23
  %address.local1111 = add i64 %address.local1110, 8192, !dbg !23
  %address.local1112 = add nuw nsw i64 %2457, %address.local1111, !dbg !23
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1087, !dbg !23
  store half %2441, ptr addrspace(1) %address.local1114, align 2, !dbg !23
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1090, !dbg !23
  store half %2440, ptr addrspace(1) %address.local1116, align 2, !dbg !23
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1093, !dbg !23
  store half %2439, ptr addrspace(1) %address.local1118, align 2, !dbg !23
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1096, !dbg !23
  store half %2438, ptr addrspace(1) %address.local1120, align 2, !dbg !23
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1099, !dbg !23
  store half %2435, ptr addrspace(1) %address.local1122, align 2, !dbg !23
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1102, !dbg !23
  store half %2434, ptr addrspace(1) %address.local1124, align 2, !dbg !23
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1105, !dbg !23
  store half %2433, ptr addrspace(1) %address.local1126, align 2, !dbg !23
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1108, !dbg !23
  store half %2432, ptr addrspace(1) %address.local1128, align 2, !dbg !23
  %2459 = extractvalue [2 x <8 x float>] %65, 1, !dbg !21
  %2460 = shufflevector <8 x float> %2459, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2428, !dbg !21
  %normalized73 = fmul <4 x float> %2460, %row.reciprocal72, !dbg !21
  %2461 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !22
  %2462 = extractelement <4 x half> %2461, i64 3, !dbg !22
  %2463 = extractelement <4 x half> %2461, i64 2, !dbg !22
  %2464 = extractelement <4 x half> %2461, i64 1, !dbg !22
  %2465 = extractelement <4 x half> %2461, i64 0, !dbg !22
  %2466 = shufflevector <8 x float> %2459, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized74 = fmul <4 x float> %2466, %row.reciprocal72, !dbg !21
  %2467 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !22
  %2468 = extractelement <4 x half> %2467, i64 3, !dbg !22
  %2469 = extractelement <4 x half> %2467, i64 2, !dbg !22
  %2470 = extractelement <4 x half> %2467, i64 1, !dbg !22
  %2471 = extractelement <4 x half> %2467, i64 0, !dbg !22
  %2472 = extractvalue [2 x <8 x float>] %65, 0, !dbg !21
  %2473 = shufflevector <8 x float> %2472, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2426, !dbg !21
  %normalized76 = fmul <4 x float> %2473, %row.reciprocal75, !dbg !21
  %2474 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !22
  %2475 = extractelement <4 x half> %2474, i64 3, !dbg !22
  %2476 = extractelement <4 x half> %2474, i64 2, !dbg !22
  %2477 = extractelement <4 x half> %2474, i64 1, !dbg !22
  %2478 = extractelement <4 x half> %2474, i64 0, !dbg !22
  %2479 = shufflevector <8 x float> %2472, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized77 = fmul <4 x float> %2479, %row.reciprocal75, !dbg !21
  %2480 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !22
  %2481 = extractelement <4 x half> %2480, i64 3, !dbg !22
  %2482 = extractelement <4 x half> %2480, i64 2, !dbg !22
  %2483 = extractelement <4 x half> %2480, i64 1, !dbg !22
  %2484 = extractelement <4 x half> %2480, i64 0, !dbg !22
  %2485 = shl nuw nsw i64 %27, 19, !dbg !23
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
  %address.local1140 = add nuw nsw i64 %2485, %address.local1139, !dbg !23
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %2486 = lshr i64 %address.local1130, 2, !dbg !12
  %address.local1144 = and i64 %2486, 12, !dbg !12
  %address.local1145 = or disjoint i64 %address.local1144, 32, !dbg !13
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1145, !dbg !23
  store half %2484, ptr addrspace(1) %address.local1146, align 2, !dbg !23
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1148 = or disjoint i64 %address.local1144, 33, !dbg !13
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1148, !dbg !23
  store half %2483, ptr addrspace(1) %address.local1149, align 2, !dbg !23
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1151 = or disjoint i64 %address.local1144, 34, !dbg !13
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1150, i64 %address.local1151, !dbg !23
  store half %2482, ptr addrspace(1) %address.local1152, align 2, !dbg !23
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1154 = or disjoint i64 %address.local1144, 35, !dbg !13
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1154, !dbg !23
  store half %2481, ptr addrspace(1) %address.local1155, align 2, !dbg !23
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1157 = or disjoint i64 %address.local1144, 48, !dbg !13
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1156, i64 %address.local1157, !dbg !23
  store half %2478, ptr addrspace(1) %address.local1158, align 2, !dbg !23
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1160 = or disjoint i64 %address.local1144, 49, !dbg !13
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1160, !dbg !23
  store half %2477, ptr addrspace(1) %address.local1161, align 2, !dbg !23
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1163 = or disjoint i64 %address.local1144, 50, !dbg !13
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1162, i64 %address.local1163, !dbg !23
  store half %2476, ptr addrspace(1) %address.local1164, align 2, !dbg !23
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1166 = or i64 %2486, 51, !dbg !13
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1166, !dbg !23
  store half %2475, ptr addrspace(1) %address.local1167, align 2, !dbg !23
  %address.local1168 = shl i64 %address.local1138, 7, !dbg !23
  %address.local1169 = add i64 %address.local1168, 8192, !dbg !23
  %address.local1170 = add nuw nsw i64 %2485, %address.local1169, !dbg !23
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1145, !dbg !23
  store half %2471, ptr addrspace(1) %address.local1172, align 2, !dbg !23
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1148, !dbg !23
  store half %2470, ptr addrspace(1) %address.local1174, align 2, !dbg !23
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1151, !dbg !23
  store half %2469, ptr addrspace(1) %address.local1176, align 2, !dbg !23
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1154, !dbg !23
  store half %2468, ptr addrspace(1) %address.local1178, align 2, !dbg !23
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1157, !dbg !23
  store half %2465, ptr addrspace(1) %address.local1180, align 2, !dbg !23
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1160, !dbg !23
  store half %2464, ptr addrspace(1) %address.local1182, align 2, !dbg !23
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1163, !dbg !23
  store half %2463, ptr addrspace(1) %address.local1184, align 2, !dbg !23
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1166, !dbg !23
  store half %2462, ptr addrspace(1) %address.local1186, align 2, !dbg !23
  %2487 = extractvalue [2 x <8 x float>] %66, 1, !dbg !21
  %2488 = shufflevector <8 x float> %2487, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2428, !dbg !21
  %normalized79 = fmul <4 x float> %2488, %row.reciprocal78, !dbg !21
  %2489 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !22
  %2490 = extractelement <4 x half> %2489, i64 3, !dbg !22
  %2491 = extractelement <4 x half> %2489, i64 2, !dbg !22
  %2492 = extractelement <4 x half> %2489, i64 1, !dbg !22
  %2493 = extractelement <4 x half> %2489, i64 0, !dbg !22
  %2494 = shufflevector <8 x float> %2487, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized80 = fmul <4 x float> %2494, %row.reciprocal78, !dbg !21
  %2495 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !22
  %2496 = extractelement <4 x half> %2495, i64 3, !dbg !22
  %2497 = extractelement <4 x half> %2495, i64 2, !dbg !22
  %2498 = extractelement <4 x half> %2495, i64 1, !dbg !22
  %2499 = extractelement <4 x half> %2495, i64 0, !dbg !22
  %2500 = extractvalue [2 x <8 x float>] %66, 0, !dbg !21
  %2501 = shufflevector <8 x float> %2500, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2426, !dbg !21
  %normalized82 = fmul <4 x float> %2501, %row.reciprocal81, !dbg !21
  %2502 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !22
  %2503 = extractelement <4 x half> %2502, i64 3, !dbg !22
  %2504 = extractelement <4 x half> %2502, i64 2, !dbg !22
  %2505 = extractelement <4 x half> %2502, i64 1, !dbg !22
  %2506 = extractelement <4 x half> %2502, i64 0, !dbg !22
  %2507 = shufflevector <8 x float> %2500, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized83 = fmul <4 x float> %2507, %row.reciprocal81, !dbg !21
  %2508 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !22
  %2509 = extractelement <4 x half> %2508, i64 3, !dbg !22
  %2510 = extractelement <4 x half> %2508, i64 2, !dbg !22
  %2511 = extractelement <4 x half> %2508, i64 1, !dbg !22
  %2512 = extractelement <4 x half> %2508, i64 0, !dbg !22
  %2513 = shl nuw nsw i64 %27, 19, !dbg !23
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
  %address.local1198 = add nuw nsw i64 %2513, %address.local1197, !dbg !23
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %2514 = lshr i64 %address.local1188, 2, !dbg !12
  %address.local1202 = and i64 %2514, 12, !dbg !12
  %address.local1203 = or disjoint i64 %address.local1202, 64, !dbg !13
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1203, !dbg !23
  store half %2512, ptr addrspace(1) %address.local1204, align 2, !dbg !23
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1206 = or disjoint i64 %address.local1202, 65, !dbg !13
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1206, !dbg !23
  store half %2511, ptr addrspace(1) %address.local1207, align 2, !dbg !23
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1209 = or disjoint i64 %address.local1202, 66, !dbg !13
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1208, i64 %address.local1209, !dbg !23
  store half %2510, ptr addrspace(1) %address.local1210, align 2, !dbg !23
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1212 = or disjoint i64 %address.local1202, 67, !dbg !13
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1211, i64 %address.local1212, !dbg !23
  store half %2509, ptr addrspace(1) %address.local1213, align 2, !dbg !23
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1215 = or disjoint i64 %address.local1202, 80, !dbg !13
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1214, i64 %address.local1215, !dbg !23
  store half %2506, ptr addrspace(1) %address.local1216, align 2, !dbg !23
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1218 = or disjoint i64 %address.local1202, 81, !dbg !13
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1217, i64 %address.local1218, !dbg !23
  store half %2505, ptr addrspace(1) %address.local1219, align 2, !dbg !23
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1221 = or disjoint i64 %address.local1202, 82, !dbg !13
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1221, !dbg !23
  store half %2504, ptr addrspace(1) %address.local1222, align 2, !dbg !23
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1224 = or disjoint i64 %address.local1202, 83, !dbg !13
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1223, i64 %address.local1224, !dbg !23
  store half %2503, ptr addrspace(1) %address.local1225, align 2, !dbg !23
  %address.local1226 = shl i64 %address.local1196, 7, !dbg !23
  %address.local1227 = add i64 %address.local1226, 8192, !dbg !23
  %address.local1228 = add nuw nsw i64 %2513, %address.local1227, !dbg !23
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1229, i64 %address.local1203, !dbg !23
  store half %2499, ptr addrspace(1) %address.local1230, align 2, !dbg !23
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1231, i64 %address.local1206, !dbg !23
  store half %2498, ptr addrspace(1) %address.local1232, align 2, !dbg !23
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1233, i64 %address.local1209, !dbg !23
  store half %2497, ptr addrspace(1) %address.local1234, align 2, !dbg !23
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1235, i64 %address.local1212, !dbg !23
  store half %2496, ptr addrspace(1) %address.local1236, align 2, !dbg !23
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1237, i64 %address.local1215, !dbg !23
  store half %2493, ptr addrspace(1) %address.local1238, align 2, !dbg !23
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1239, i64 %address.local1218, !dbg !23
  store half %2492, ptr addrspace(1) %address.local1240, align 2, !dbg !23
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1241, i64 %address.local1221, !dbg !23
  store half %2491, ptr addrspace(1) %address.local1242, align 2, !dbg !23
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1243, i64 %address.local1224, !dbg !23
  store half %2490, ptr addrspace(1) %address.local1244, align 2, !dbg !23
  %2515 = extractvalue [2 x <8 x float>] %67, 1, !dbg !21
  %2516 = shufflevector <8 x float> %2515, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2428, !dbg !21
  %normalized85 = fmul <4 x float> %2516, %row.reciprocal84, !dbg !21
  %2517 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !22
  %2518 = extractelement <4 x half> %2517, i64 3, !dbg !22
  %2519 = extractelement <4 x half> %2517, i64 2, !dbg !22
  %2520 = extractelement <4 x half> %2517, i64 1, !dbg !22
  %2521 = extractelement <4 x half> %2517, i64 0, !dbg !22
  %2522 = shufflevector <8 x float> %2515, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized86 = fmul <4 x float> %2522, %row.reciprocal84, !dbg !21
  %2523 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !22
  %2524 = extractelement <4 x half> %2523, i64 3, !dbg !22
  %2525 = extractelement <4 x half> %2523, i64 2, !dbg !22
  %2526 = extractelement <4 x half> %2523, i64 1, !dbg !22
  %2527 = extractelement <4 x half> %2523, i64 0, !dbg !22
  %2528 = extractvalue [2 x <8 x float>] %67, 0, !dbg !21
  %2529 = shufflevector <8 x float> %2528, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2426, !dbg !21
  %normalized88 = fmul <4 x float> %2529, %row.reciprocal87, !dbg !21
  %2530 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !22
  %2531 = extractelement <4 x half> %2530, i64 3, !dbg !22
  %2532 = extractelement <4 x half> %2530, i64 2, !dbg !22
  %2533 = extractelement <4 x half> %2530, i64 1, !dbg !22
  %2534 = extractelement <4 x half> %2530, i64 0, !dbg !22
  %2535 = shufflevector <8 x float> %2528, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized89 = fmul <4 x float> %2535, %row.reciprocal87, !dbg !21
  %2536 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !22
  %2537 = extractelement <4 x half> %2536, i64 3, !dbg !22
  %2538 = extractelement <4 x half> %2536, i64 2, !dbg !22
  %2539 = extractelement <4 x half> %2536, i64 1, !dbg !22
  %2540 = extractelement <4 x half> %2536, i64 0, !dbg !22
  %2541 = shl nuw nsw i64 %27, 19, !dbg !23
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
  %address.local1256 = add nuw nsw i64 %2541, %address.local1255, !dbg !23
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %2542 = lshr i64 %address.local1246, 2, !dbg !12
  %address.local1260 = and i64 %2542, 12, !dbg !12
  %address.local1261 = or disjoint i64 %address.local1260, 96, !dbg !13
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1261, !dbg !23
  store half %2540, ptr addrspace(1) %address.local1262, align 2, !dbg !23
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1264 = or disjoint i64 %address.local1260, 97, !dbg !13
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1264, !dbg !23
  store half %2539, ptr addrspace(1) %address.local1265, align 2, !dbg !23
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1267 = or disjoint i64 %address.local1260, 98, !dbg !13
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1266, i64 %address.local1267, !dbg !23
  store half %2538, ptr addrspace(1) %address.local1268, align 2, !dbg !23
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1270 = or disjoint i64 %address.local1260, 99, !dbg !13
  %address.local1271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1269, i64 %address.local1270, !dbg !23
  store half %2537, ptr addrspace(1) %address.local1271, align 2, !dbg !23
  %address.local1272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1273 = or disjoint i64 %address.local1260, 112, !dbg !13
  %address.local1274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1272, i64 %address.local1273, !dbg !23
  store half %2534, ptr addrspace(1) %address.local1274, align 2, !dbg !23
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1276 = or disjoint i64 %address.local1260, 113, !dbg !13
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1275, i64 %address.local1276, !dbg !23
  store half %2533, ptr addrspace(1) %address.local1277, align 2, !dbg !23
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1279 = or disjoint i64 %address.local1260, 114, !dbg !13
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1278, i64 %address.local1279, !dbg !23
  store half %2532, ptr addrspace(1) %address.local1280, align 2, !dbg !23
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1282 = or i64 %2542, 115, !dbg !13
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1281, i64 %address.local1282, !dbg !23
  store half %2531, ptr addrspace(1) %address.local1283, align 2, !dbg !23
  %address.local1284 = shl i64 %address.local1254, 7, !dbg !23
  %address.local1285 = add i64 %address.local1284, 8192, !dbg !23
  %address.local1286 = add nuw nsw i64 %2541, %address.local1285, !dbg !23
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1287, i64 %address.local1261, !dbg !23
  store half %2527, ptr addrspace(1) %address.local1288, align 2, !dbg !23
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1264, !dbg !23
  store half %2526, ptr addrspace(1) %address.local1290, align 2, !dbg !23
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1291, i64 %address.local1267, !dbg !23
  store half %2525, ptr addrspace(1) %address.local1292, align 2, !dbg !23
  %address.local1293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1293, i64 %address.local1270, !dbg !23
  store half %2524, ptr addrspace(1) %address.local1294, align 2, !dbg !23
  %address.local1295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1295, i64 %address.local1273, !dbg !23
  store half %2521, ptr addrspace(1) %address.local1296, align 2, !dbg !23
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1297, i64 %address.local1276, !dbg !23
  store half %2520, ptr addrspace(1) %address.local1298, align 2, !dbg !23
  %address.local1299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1299, i64 %address.local1279, !dbg !23
  store half %2519, ptr addrspace(1) %address.local1300, align 2, !dbg !23
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1282, !dbg !23
  store half %2518, ptr addrspace(1) %address.local1302, align 2, !dbg !23
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-151454-z6uud_gi/grid-000001")
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
