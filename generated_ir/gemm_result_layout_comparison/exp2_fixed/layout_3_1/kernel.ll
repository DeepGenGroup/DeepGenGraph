; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000001/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000001/input.mlir"

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
  %39 = lshr i64 %35, 4, !dbg !12
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
  %60 = mul nsw i64 %39, -16, !dbg !13
  br label %61, !dbg !7

61:                                               ; preds = %2420, %4
  %62 = phi i64 [ %2479, %2420 ], [ 0, %4 ], !dbg !7
  %63 = phi [2 x <8 x float>] [ %2460, %2420 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi [2 x <1 x float>] [ %2464, %2420 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = phi [2 x <1 x float>] [ %2466, %2420 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %66 = phi [2 x <8 x float>] [ %2470, %2420 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = phi [2 x <8 x float>] [ %2474, %2420 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = phi [2 x <8 x float>] [ %2478, %2420 ], [ zeroinitializer, %4 ], !dbg !7
  %69 = icmp slt i64 %62, %46, !dbg !7
  br i1 %69, label %70, label %2480, !dbg !7

70:                                               ; preds = %61
  %71 = shl nsw i64 %62, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local258 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local259 = and i64 %address.local258, 1536, !dbg !11
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local259, !dbg !11
  %address.local261 = ptrtoint ptr addrspace(3) %address.local260 to i64, !dbg !11
  %address.local262 = trunc i64 %address.local261 to i32, !dbg !11
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %73 = zext i32 %72 to i64, !dbg !11
  %74 = inttoptr i64 %73 to ptr addrspace(3), !dbg !11
  %address.local263 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local264 = and i64 %address.local263, 16256, !dbg !11
  %address.local265 = lshr i64 %address.local, 4, !dbg !11
  %address.local266 = and i64 %address.local265, 8, !dbg !11
  %address.local267 = or disjoint i64 %address.local264, %address.local266, !dbg !11
  %address.local268 = getelementptr [2 x i8], ptr addrspace(1) %47, i64 %address.local267, !dbg !11
  %75 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %76 = ptrtoint ptr addrspace(1) %address.local268 to i64, !dbg !11
  %77 = sub i64 %76, %75, !dbg !11
  %dma.buffer.offset1821 = trunc i64 %77 to i32, !dbg !11
  %78 = trunc i64 %75 to i32, !dbg !11
  %79 = insertelement <4 x i32> poison, i32 %78, i64 0, !dbg !11
  %80 = lshr i64 %75, 32, !dbg !11
  %81 = trunc i64 %80 to i32, !dbg !11
  %82 = insertelement <4 x i32> %79, i32 %81, i64 1, !dbg !11
  %83 = insertelement <4 x i32> %82, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1822 = insertelement <4 x i32> %83, i32 131072, i64 3, !dbg !11
  %84 = ptrtoint ptr addrspace(3) %74 to i32, !dbg !11
  %dma.buffer.lds1823 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !11
  %85 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1821, <4 x i32> %dma.buffer.rsrc1822) #10, !dbg !11
  %address.local269 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local270 = shl nuw nsw i64 %address.local269, 3, !dbg !11
  %address.local271 = and i64 %address.local270, 3584, !dbg !11
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local271, !dbg !11
  %address.local273 = ptrtoint ptr addrspace(3) %address.local272 to i64, !dbg !11
  %address.local274 = trunc i64 %address.local273 to i32, !dbg !11
  %86 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %87 = zext i32 %86 to i64, !dbg !11
  %88 = inttoptr i64 %87 to ptr addrspace(3), !dbg !11
  %address.local275 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local276 = and i64 %address.local275, 16256, !dbg !11
  %address.local277 = lshr i64 %address.local269, 4, !dbg !11
  %address.local278 = and i64 %address.local277, 24, !dbg !11
  %address.local279 = or disjoint i64 %address.local276, %address.local278, !dbg !11
  %address.local280 = getelementptr [2 x i8], ptr addrspace(1) %47, i64 %address.local279, !dbg !11
  %89 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %90 = ptrtoint ptr addrspace(1) %address.local280 to i64, !dbg !11
  %91 = sub i64 %90, %89, !dbg !11
  %dma.buffer.offset1818 = trunc i64 %91 to i32, !dbg !11
  %92 = trunc i64 %89 to i32, !dbg !11
  %93 = insertelement <4 x i32> poison, i32 %92, i64 0, !dbg !11
  %94 = lshr i64 %89, 32, !dbg !11
  %95 = trunc i64 %94 to i32, !dbg !11
  %96 = insertelement <4 x i32> %93, i32 %95, i64 1, !dbg !11
  %97 = insertelement <4 x i32> %96, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1819 = insertelement <4 x i32> %97, i32 131072, i64 3, !dbg !11
  %98 = ptrtoint ptr addrspace(3) %88 to i32, !dbg !11
  %dma.buffer.lds1820 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %98), !dbg !11
  %99 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1818, <4 x i32> %dma.buffer.rsrc1819) #10, !dbg !11
  %100 = lshr i64 %62, 12, !dbg !18
  %101 = add nuw nsw i64 %100, %15, !dbg !18
  %102 = lshr i64 %62, 5, !dbg !18
  %103 = and i64 %102, 127, !dbg !18
  %104 = shl i64 %101, 19, !dbg !18
  %105 = and i64 %104, 16252928, !dbg !18
  %106 = shl i64 %102, 19, !dbg !18
  %107 = shl i64 %62, 14, !dbg !18
  %108 = sub i64 %107, %106, !dbg !18
  %109 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %105, !dbg !18
  %110 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %103, !dbg !18
  %111 = getelementptr [2 x i8], ptr addrspace(1) %110, i64 %108, !dbg !18
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local259, !dbg !18
  %address.local282 = ptrtoint ptr addrspace(3) %address.local281 to i64, !dbg !18
  %address.local283 = trunc i64 %address.local282 to i32, !dbg !18
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %113 = zext i32 %112 to i64, !dbg !18
  %114 = inttoptr i64 %113 to ptr addrspace(3), !dbg !18
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
  %address.local294 = getelementptr [2 x i8], ptr addrspace(1) %111, i64 %address.local293, !dbg !18
  %115 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %116 = ptrtoint ptr addrspace(1) %address.local294 to i64, !dbg !18
  %117 = sub i64 %116, %115, !dbg !18
  %dma.buffer.offset1815 = trunc i64 %117 to i32, !dbg !18
  %118 = trunc i64 %115 to i32, !dbg !18
  %119 = insertelement <4 x i32> poison, i32 %118, i64 0, !dbg !18
  %120 = lshr i64 %115, 32, !dbg !18
  %121 = trunc i64 %120 to i32, !dbg !18
  %122 = insertelement <4 x i32> %119, i32 %121, i64 1, !dbg !18
  %123 = insertelement <4 x i32> %122, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1816 = insertelement <4 x i32> %123, i32 131072, i64 3, !dbg !18
  %124 = ptrtoint ptr addrspace(3) %114 to i32, !dbg !18
  %dma.buffer.lds1817 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %124), !dbg !18
  %125 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1815, <4 x i32> %dma.buffer.rsrc1816) #10, !dbg !18
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local271, !dbg !18
  %address.local296 = ptrtoint ptr addrspace(3) %address.local295 to i64, !dbg !18
  %address.local297 = trunc i64 %address.local296 to i32, !dbg !18
  %126 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %127 = zext i32 %126 to i64, !dbg !18
  %128 = inttoptr i64 %127 to ptr addrspace(3), !dbg !18
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
  %address.local308 = getelementptr [2 x i8], ptr addrspace(1) %111, i64 %address.local307, !dbg !18
  %129 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %130 = ptrtoint ptr addrspace(1) %address.local308 to i64, !dbg !18
  %131 = sub i64 %130, %129, !dbg !18
  %dma.buffer.offset1812 = trunc i64 %131 to i32, !dbg !18
  %132 = trunc i64 %129 to i32, !dbg !18
  %133 = insertelement <4 x i32> poison, i32 %132, i64 0, !dbg !18
  %134 = lshr i64 %129, 32, !dbg !18
  %135 = trunc i64 %134 to i32, !dbg !18
  %136 = insertelement <4 x i32> %133, i32 %135, i64 1, !dbg !18
  %137 = insertelement <4 x i32> %136, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1813 = insertelement <4 x i32> %137, i32 131072, i64 3, !dbg !18
  %138 = ptrtoint ptr addrspace(3) %128 to i32, !dbg !18
  %dma.buffer.lds1814 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138), !dbg !18
  %139 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1812, <4 x i32> %dma.buffer.rsrc1813) #10, !dbg !18
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
  %140 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %141 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %142 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !12
  %143 = fmul <4 x half> %140, splat (half 0xH3015), !dbg !12
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !12
  %145 = fmul <4 x half> %141, splat (half 0xH3015), !dbg !12
  %146 = fmul <4 x half> %142, splat (half 0xH3015), !dbg !12
  %147 = fmul <4 x half> %144, splat (half 0xH3015), !dbg !12
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
  %148 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %148, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1561 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %150 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !19
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
  %151 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %150, <4 x float> %149) #9, !dbg !19
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1562 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4096, !dbg !19
  %154 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !19
  %lds.const.gep1563 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !19
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %154, <4 x float> %153) #9, !dbg !19
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %155, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1564 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %158 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !19
  %lds.const.gep1565 = getelementptr i8, ptr addrspace(3) %address.local362, i64 512, !dbg !19
  %159 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !19
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %158, <4 x float> %157) #9, !dbg !19
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %159, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1566 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4608, !dbg !19
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !19
  %lds.const.gep1567 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %163 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !19
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %162, <4 x float> %161) #9, !dbg !19
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %163, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1568 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %166 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !19
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1024, !dbg !19
  %167 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !19
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %166, <4 x float> %165) #9, !dbg !19
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %167, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5120, !dbg !19
  %170 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !19
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %171 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !19
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %170, <4 x float> %169) #9, !dbg !19
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %171, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !19
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1536, !dbg !19
  %175 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !19
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %174, <4 x float> %173) #9, !dbg !19
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %175, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5632, !dbg !19
  %178 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !19
  %179 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %178, <4 x float> %177) #9, !dbg !19
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %179, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1575 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1575, align 8, !dbg !19
  %183 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %182, <4 x float> %181) #9, !dbg !19
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %183, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1576 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4096, !dbg !19
  %186 = load <4 x half>, ptr addrspace(3) %lds.const.gep1576, align 8, !dbg !19
  %lds.const.gep1577 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %187 = load <4 x half>, ptr addrspace(3) %lds.const.gep1577, align 8, !dbg !19
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %186, <4 x float> %185) #9, !dbg !19
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %187, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1578 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %190 = load <4 x half>, ptr addrspace(3) %lds.const.gep1578, align 8, !dbg !19
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local362, i64 512, !dbg !19
  %191 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !19
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %190, <4 x float> %189) #9, !dbg !19
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4608, !dbg !19
  %194 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !19
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %195 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !19
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %194, <4 x float> %193) #9, !dbg !19
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %198 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !19
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1024, !dbg !19
  %199 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !19
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %198, <4 x float> %197) #9, !dbg !19
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5120, !dbg !19
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !19
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %203 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !19
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %202, <4 x float> %201) #9, !dbg !19
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !19
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1536, !dbg !19
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !19
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %206, <4 x float> %205) #9, !dbg !19
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %207, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5632, !dbg !19
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !19
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %210, <4 x float> %209) #9, !dbg !19
  %212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %213 = zext i32 %212 to i64, !dbg !11
  %214 = inttoptr i64 %213 to ptr addrspace(3), !dbg !11
  %address.local401 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local267, !dbg !11
  %215 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %216 = ptrtoint ptr addrspace(1) %address.local401 to i64, !dbg !11
  %217 = sub i64 %216, %215, !dbg !11
  %dma.buffer.offset1809 = trunc i64 %217 to i32, !dbg !11
  %218 = trunc i64 %215 to i32, !dbg !11
  %219 = insertelement <4 x i32> poison, i32 %218, i64 0, !dbg !11
  %220 = lshr i64 %215, 32, !dbg !11
  %221 = trunc i64 %220 to i32, !dbg !11
  %222 = insertelement <4 x i32> %219, i32 %221, i64 1, !dbg !11
  %223 = insertelement <4 x i32> %222, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1810 = insertelement <4 x i32> %223, i32 131072, i64 3, !dbg !11
  %224 = ptrtoint ptr addrspace(3) %214 to i32, !dbg !11
  %dma.buffer.lds1811 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %224), !dbg !11
  %225 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1811, i32 %dma.buffer.offset1809, <4 x i32> %dma.buffer.rsrc1810) #10, !dbg !11
  %226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %227 = zext i32 %226 to i64, !dbg !11
  %228 = inttoptr i64 %227 to ptr addrspace(3), !dbg !11
  %address.local402 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local279, !dbg !11
  %229 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %230 = ptrtoint ptr addrspace(1) %address.local402 to i64, !dbg !11
  %231 = sub i64 %230, %229, !dbg !11
  %dma.buffer.offset1806 = trunc i64 %231 to i32, !dbg !11
  %232 = trunc i64 %229 to i32, !dbg !11
  %233 = insertelement <4 x i32> poison, i32 %232, i64 0, !dbg !11
  %234 = lshr i64 %229, 32, !dbg !11
  %235 = trunc i64 %234 to i32, !dbg !11
  %236 = insertelement <4 x i32> %233, i32 %235, i64 1, !dbg !11
  %237 = insertelement <4 x i32> %236, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1807 = insertelement <4 x i32> %237, i32 131072, i64 3, !dbg !11
  %238 = ptrtoint ptr addrspace(3) %228 to i32, !dbg !11
  %dma.buffer.lds1808 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %238), !dbg !11
  %239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1806, <4 x i32> %dma.buffer.rsrc1807) #10, !dbg !11
  %240 = add nuw i64 %62, 1024, !dbg !18
  %241 = icmp slt i64 %240, 0, !dbg !18
  %242 = sub nuw i64 -1025, %62, !dbg !18
  %243 = select i1 %241, i64 %242, i64 %240, !dbg !18
  %244 = sdiv i64 %243, 4096, !dbg !18
  %.lobit = ashr i64 %240, 63, !dbg !18
  %245 = xor i64 %244, %.lobit, !dbg !18
  %246 = add nsw i64 %245, %15, !dbg !18
  %247 = add nsw i64 %102, %53, !dbg !18
  %248 = add nsw i64 %247, 32, !dbg !18
  %249 = icmp slt i64 %247, -32, !dbg !18
  %250 = sub nsw i64 -33, %247, !dbg !18
  %251 = select i1 %249, i64 %250, i64 %248, !dbg !18
  %252 = sdiv i64 %251, 128, !dbg !18
  %253 = sext i1 %249 to i64, !dbg !18
  %254 = xor i64 %252, %253, !dbg !18
  %255 = shl nsw i64 %254, 7, !dbg !18
  %256 = sub i64 %53, %255, !dbg !18
  %257 = shl i64 %246, 19, !dbg !18
  %258 = and i64 %257, 16252928, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %259 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %258, !dbg !18
  %260 = getelementptr [2 x i8], ptr addrspace(1) %259, i64 %102, !dbg !18
  %261 = getelementptr [2 x i8], ptr addrspace(1) %260, i64 %256, !dbg !18
  %262 = getelementptr i8, ptr addrspace(1) %261, i64 64, !dbg !18
  %263 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %108, !dbg !18
  %264 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %265 = zext i32 %264 to i64, !dbg !18
  %266 = inttoptr i64 %265 to ptr addrspace(3), !dbg !18
  %address.local403 = getelementptr [2 x i8], ptr addrspace(1) %263, i64 %address.local293, !dbg !18
  %267 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %268 = ptrtoint ptr addrspace(1) %address.local403 to i64, !dbg !18
  %269 = sub i64 %268, %267, !dbg !18
  %dma.buffer.offset1803 = trunc i64 %269 to i32, !dbg !18
  %270 = trunc i64 %267 to i32, !dbg !18
  %271 = insertelement <4 x i32> poison, i32 %270, i64 0, !dbg !18
  %272 = lshr i64 %267, 32, !dbg !18
  %273 = trunc i64 %272 to i32, !dbg !18
  %274 = insertelement <4 x i32> %271, i32 %273, i64 1, !dbg !18
  %275 = insertelement <4 x i32> %274, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1804 = insertelement <4 x i32> %275, i32 131072, i64 3, !dbg !18
  %276 = ptrtoint ptr addrspace(3) %266 to i32, !dbg !18
  %dma.buffer.lds1805 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %276), !dbg !18
  %277 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1803, <4 x i32> %dma.buffer.rsrc1804) #10, !dbg !18
  %278 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %279 = zext i32 %278 to i64, !dbg !18
  %280 = inttoptr i64 %279 to ptr addrspace(3), !dbg !18
  %address.local404 = getelementptr [2 x i8], ptr addrspace(1) %263, i64 %address.local307, !dbg !18
  %281 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %282 = ptrtoint ptr addrspace(1) %address.local404 to i64, !dbg !18
  %283 = sub i64 %282, %281, !dbg !18
  %dma.buffer.offset1800 = trunc i64 %283 to i32, !dbg !18
  %284 = trunc i64 %281 to i32, !dbg !18
  %285 = insertelement <4 x i32> poison, i32 %284, i64 0, !dbg !18
  %286 = lshr i64 %281, 32, !dbg !18
  %287 = trunc i64 %286 to i32, !dbg !18
  %288 = insertelement <4 x i32> %285, i32 %287, i64 1, !dbg !18
  %289 = insertelement <4 x i32> %288, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1801 = insertelement <4 x i32> %289, i32 131072, i64 3, !dbg !18
  %290 = ptrtoint ptr addrspace(3) %280 to i32, !dbg !18
  %dma.buffer.lds1802 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %290), !dbg !18
  %291 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1800, <4 x i32> %dma.buffer.rsrc1801) #10, !dbg !18
  %292 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !12
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !12
  %295 = fmul <4 x half> %292, splat (half 0xH3015), !dbg !12
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %296 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !12
  %297 = fmul <4 x half> %293, splat (half 0xH3015), !dbg !12
  %298 = fmul <4 x half> %294, splat (half 0xH3015), !dbg !12
  %299 = fmul <4 x half> %296, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %300 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %300, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !19
  %303 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %302, <4 x float> %301) #9, !dbg !19
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %303, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4096, !dbg !19
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !19
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !19
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %306, <4 x float> %305) #9, !dbg !19
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !19
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %address.local362, i64 512, !dbg !19
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !19
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %310, <4 x float> %309) #9, !dbg !19
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %311, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4608, !dbg !19
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !19
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !19
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %314, <4 x float> %313) #9, !dbg !19
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %315, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !19
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1024, !dbg !19
  %319 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !19
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %318, <4 x float> %317) #9, !dbg !19
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %319, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5120, !dbg !19
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !19
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !19
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %322, <4 x float> %321) #9, !dbg !19
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %323, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !19
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1536, !dbg !19
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !19
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %326, <4 x float> %325) #9, !dbg !19
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %327, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5632, !dbg !19
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !19
  %331 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %330, <4 x float> %329) #9, !dbg !19
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %331, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !19
  %335 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %334, <4 x float> %333) #9, !dbg !19
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %335, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4096, !dbg !19
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !19
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %339 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !19
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %338, <4 x float> %337) #9, !dbg !19
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %339, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !19
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local362, i64 512, !dbg !19
  %343 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !19
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %342, <4 x float> %341) #9, !dbg !19
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %343, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4608, !dbg !19
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !19
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !19
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %346, <4 x float> %345) #9, !dbg !19
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %347, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !19
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1024, !dbg !19
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !19
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %350, <4 x float> %349) #9, !dbg !19
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %351, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5120, !dbg !19
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !19
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !19
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %354, <4 x float> %353) #9, !dbg !19
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %355, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !19
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1536, !dbg !19
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !19
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %358, <4 x float> %357) #9, !dbg !19
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %359, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5632, !dbg !19
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !19
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %362, <4 x float> %361) #9, !dbg !19
  %364 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %365 = zext i32 %364 to i64, !dbg !11
  %366 = inttoptr i64 %365 to ptr addrspace(3), !dbg !11
  %address.local405 = getelementptr [2 x i8], ptr addrspace(1) %55, i64 %address.local267, !dbg !11
  %367 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %368 = ptrtoint ptr addrspace(1) %address.local405 to i64, !dbg !11
  %369 = sub i64 %368, %367, !dbg !11
  %dma.buffer.offset1797 = trunc i64 %369 to i32, !dbg !11
  %370 = trunc i64 %367 to i32, !dbg !11
  %371 = insertelement <4 x i32> poison, i32 %370, i64 0, !dbg !11
  %372 = lshr i64 %367, 32, !dbg !11
  %373 = trunc i64 %372 to i32, !dbg !11
  %374 = insertelement <4 x i32> %371, i32 %373, i64 1, !dbg !11
  %375 = insertelement <4 x i32> %374, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1798 = insertelement <4 x i32> %375, i32 131072, i64 3, !dbg !11
  %376 = ptrtoint ptr addrspace(3) %366 to i32, !dbg !11
  %dma.buffer.lds1799 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %376), !dbg !11
  %377 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1797, <4 x i32> %dma.buffer.rsrc1798) #10, !dbg !11
  %378 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %379 = zext i32 %378 to i64, !dbg !11
  %380 = inttoptr i64 %379 to ptr addrspace(3), !dbg !11
  %address.local406 = getelementptr [2 x i8], ptr addrspace(1) %55, i64 %address.local279, !dbg !11
  %381 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %382 = ptrtoint ptr addrspace(1) %address.local406 to i64, !dbg !11
  %383 = sub i64 %382, %381, !dbg !11
  %dma.buffer.offset1794 = trunc i64 %383 to i32, !dbg !11
  %384 = trunc i64 %381 to i32, !dbg !11
  %385 = insertelement <4 x i32> poison, i32 %384, i64 0, !dbg !11
  %386 = lshr i64 %381, 32, !dbg !11
  %387 = trunc i64 %386 to i32, !dbg !11
  %388 = insertelement <4 x i32> %385, i32 %387, i64 1, !dbg !11
  %389 = insertelement <4 x i32> %388, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1795 = insertelement <4 x i32> %389, i32 131072, i64 3, !dbg !11
  %390 = ptrtoint ptr addrspace(3) %380 to i32, !dbg !11
  %dma.buffer.lds1796 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %390), !dbg !11
  %391 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1796, i32 %dma.buffer.offset1794, <4 x i32> %dma.buffer.rsrc1795) #10, !dbg !11
  %392 = add nuw i64 %62, 2048, !dbg !18
  %393 = icmp slt i64 %392, 0, !dbg !18
  %394 = sub nuw i64 -2049, %62, !dbg !18
  %395 = select i1 %393, i64 %394, i64 %392, !dbg !18
  %396 = sdiv i64 %395, 4096, !dbg !18
  %.lobit2 = ashr i64 %392, 63, !dbg !18
  %397 = xor i64 %396, %.lobit2, !dbg !18
  %398 = add nsw i64 %397, %15, !dbg !18
  %399 = add nsw i64 %247, 64, !dbg !18
  %400 = icmp slt i64 %247, -64, !dbg !18
  %401 = sub nsw i64 -65, %247, !dbg !18
  %402 = select i1 %400, i64 %401, i64 %399, !dbg !18
  %403 = sdiv i64 %402, 128, !dbg !18
  %404 = sext i1 %400 to i64, !dbg !18
  %405 = xor i64 %403, %404, !dbg !18
  %406 = shl nsw i64 %405, 7, !dbg !18
  %407 = sub i64 %53, %406, !dbg !18
  %408 = shl i64 %398, 19, !dbg !18
  %409 = and i64 %408, 16252928, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %410 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %409, !dbg !18
  %411 = getelementptr [2 x i8], ptr addrspace(1) %410, i64 %102, !dbg !18
  %412 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %407, !dbg !18
  %413 = getelementptr i8, ptr addrspace(1) %412, i64 128, !dbg !18
  %414 = getelementptr [2 x i8], ptr addrspace(1) %413, i64 %108, !dbg !18
  %415 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %416 = zext i32 %415 to i64, !dbg !18
  %417 = inttoptr i64 %416 to ptr addrspace(3), !dbg !18
  %address.local407 = getelementptr [2 x i8], ptr addrspace(1) %414, i64 %address.local293, !dbg !18
  %418 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %419 = ptrtoint ptr addrspace(1) %address.local407 to i64, !dbg !18
  %420 = sub i64 %419, %418, !dbg !18
  %dma.buffer.offset1791 = trunc i64 %420 to i32, !dbg !18
  %421 = trunc i64 %418 to i32, !dbg !18
  %422 = insertelement <4 x i32> poison, i32 %421, i64 0, !dbg !18
  %423 = lshr i64 %418, 32, !dbg !18
  %424 = trunc i64 %423 to i32, !dbg !18
  %425 = insertelement <4 x i32> %422, i32 %424, i64 1, !dbg !18
  %426 = insertelement <4 x i32> %425, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1792 = insertelement <4 x i32> %426, i32 131072, i64 3, !dbg !18
  %427 = ptrtoint ptr addrspace(3) %417 to i32, !dbg !18
  %dma.buffer.lds1793 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %427), !dbg !18
  %428 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1793, i32 %dma.buffer.offset1791, <4 x i32> %dma.buffer.rsrc1792) #10, !dbg !18
  %429 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %430 = zext i32 %429 to i64, !dbg !18
  %431 = inttoptr i64 %430 to ptr addrspace(3), !dbg !18
  %address.local408 = getelementptr [2 x i8], ptr addrspace(1) %414, i64 %address.local307, !dbg !18
  %432 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %433 = ptrtoint ptr addrspace(1) %address.local408 to i64, !dbg !18
  %434 = sub i64 %433, %432, !dbg !18
  %dma.buffer.offset1788 = trunc i64 %434 to i32, !dbg !18
  %435 = trunc i64 %432 to i32, !dbg !18
  %436 = insertelement <4 x i32> poison, i32 %435, i64 0, !dbg !18
  %437 = lshr i64 %432, 32, !dbg !18
  %438 = trunc i64 %437 to i32, !dbg !18
  %439 = insertelement <4 x i32> %436, i32 %438, i64 1, !dbg !18
  %440 = insertelement <4 x i32> %439, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1789 = insertelement <4 x i32> %440, i32 131072, i64 3, !dbg !18
  %441 = ptrtoint ptr addrspace(3) %431 to i32, !dbg !18
  %dma.buffer.lds1790 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %441), !dbg !18
  %442 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1790, i32 %dma.buffer.offset1788, <4 x i32> %dma.buffer.rsrc1789) #10, !dbg !18
  %443 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !12
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !12
  %446 = fmul <4 x half> %443, splat (half 0xH3015), !dbg !12
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %447 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !12
  %448 = fmul <4 x half> %444, splat (half 0xH3015), !dbg !12
  %449 = fmul <4 x half> %445, splat (half 0xH3015), !dbg !12
  %450 = fmul <4 x half> %447, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %451 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %451, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !19
  %454 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %453, <4 x float> %452) #9, !dbg !19
  %456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %454, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4096, !dbg !19
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !19
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %458 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !19
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %457, <4 x float> %456) #9, !dbg !19
  %460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %458, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !19
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %address.local362, i64 512, !dbg !19
  %462 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !19
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %461, <4 x float> %460) #9, !dbg !19
  %464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %462, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4608, !dbg !19
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !19
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !19
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %465, <4 x float> %464) #9, !dbg !19
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %466, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !19
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1024, !dbg !19
  %470 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !19
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %469, <4 x float> %468) #9, !dbg !19
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %470, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5120, !dbg !19
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !19
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %474 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !19
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %473, <4 x float> %472) #9, !dbg !19
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %474, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !19
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1536, !dbg !19
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !19
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %477, <4 x float> %476) #9, !dbg !19
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %478, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5632, !dbg !19
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !19
  %482 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %481, <4 x float> %480) #9, !dbg !19
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %482, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %485 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !19
  %486 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %485, <4 x float> %484) #9, !dbg !19
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %486, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4096, !dbg !19
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !19
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %490 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !19
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %489, <4 x float> %488) #9, !dbg !19
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %490, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %493 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !19
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local362, i64 512, !dbg !19
  %494 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !19
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %493, <4 x float> %492) #9, !dbg !19
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %494, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4608, !dbg !19
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !19
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %498 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !19
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %497, <4 x float> %496) #9, !dbg !19
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %498, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !19
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1024, !dbg !19
  %502 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !19
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %501, <4 x float> %500) #9, !dbg !19
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %502, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5120, !dbg !19
  %505 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !19
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %506 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !19
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %505, <4 x float> %504) #9, !dbg !19
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %506, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %509 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !19
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1536, !dbg !19
  %510 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !19
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %509, <4 x float> %508) #9, !dbg !19
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %510, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5632, !dbg !19
  %513 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !19
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %513, <4 x float> %512) #9, !dbg !19
  %515 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !11
  %516 = zext i32 %515 to i64, !dbg !11
  %517 = inttoptr i64 %516 to ptr addrspace(3), !dbg !11
  %address.local409 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local267, !dbg !11
  %518 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %519 = ptrtoint ptr addrspace(1) %address.local409 to i64, !dbg !11
  %520 = sub i64 %519, %518, !dbg !11
  %dma.buffer.offset1785 = trunc i64 %520 to i32, !dbg !11
  %521 = trunc i64 %518 to i32, !dbg !11
  %522 = insertelement <4 x i32> poison, i32 %521, i64 0, !dbg !11
  %523 = lshr i64 %518, 32, !dbg !11
  %524 = trunc i64 %523 to i32, !dbg !11
  %525 = insertelement <4 x i32> %522, i32 %524, i64 1, !dbg !11
  %526 = insertelement <4 x i32> %525, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1786 = insertelement <4 x i32> %526, i32 131072, i64 3, !dbg !11
  %527 = ptrtoint ptr addrspace(3) %517 to i32, !dbg !11
  %dma.buffer.lds1787 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %527), !dbg !11
  %528 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1787, i32 %dma.buffer.offset1785, <4 x i32> %dma.buffer.rsrc1786) #10, !dbg !11
  %529 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !11
  %530 = zext i32 %529 to i64, !dbg !11
  %531 = inttoptr i64 %530 to ptr addrspace(3), !dbg !11
  %address.local410 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local279, !dbg !11
  %532 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %533 = ptrtoint ptr addrspace(1) %address.local410 to i64, !dbg !11
  %534 = sub i64 %533, %532, !dbg !11
  %dma.buffer.offset1782 = trunc i64 %534 to i32, !dbg !11
  %535 = trunc i64 %532 to i32, !dbg !11
  %536 = insertelement <4 x i32> poison, i32 %535, i64 0, !dbg !11
  %537 = lshr i64 %532, 32, !dbg !11
  %538 = trunc i64 %537 to i32, !dbg !11
  %539 = insertelement <4 x i32> %536, i32 %538, i64 1, !dbg !11
  %540 = insertelement <4 x i32> %539, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1783 = insertelement <4 x i32> %540, i32 131072, i64 3, !dbg !11
  %541 = ptrtoint ptr addrspace(3) %531 to i32, !dbg !11
  %dma.buffer.lds1784 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %541), !dbg !11
  %542 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1784, i32 %dma.buffer.offset1782, <4 x i32> %dma.buffer.rsrc1783) #10, !dbg !11
  %543 = add nuw i64 %62, 3072, !dbg !18
  %544 = icmp slt i64 %543, 0, !dbg !18
  %545 = sub nuw i64 -3073, %62, !dbg !18
  %546 = select i1 %544, i64 %545, i64 %543, !dbg !18
  %547 = sdiv i64 %546, 4096, !dbg !18
  %.lobit3 = ashr i64 %543, 63, !dbg !18
  %548 = xor i64 %547, %.lobit3, !dbg !18
  %549 = add nsw i64 %548, %15, !dbg !18
  %550 = add nsw i64 %247, 96, !dbg !18
  %551 = icmp slt i64 %247, -96, !dbg !18
  %552 = sub nsw i64 -97, %247, !dbg !18
  %553 = select i1 %551, i64 %552, i64 %550, !dbg !18
  %554 = sdiv i64 %553, 128, !dbg !18
  %555 = sext i1 %551 to i64, !dbg !18
  %556 = xor i64 %554, %555, !dbg !18
  %557 = shl nsw i64 %556, 7, !dbg !18
  %558 = sub i64 %53, %557, !dbg !18
  %559 = shl i64 %549, 19, !dbg !18
  %560 = and i64 %559, 16252928, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %561 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %560, !dbg !18
  %562 = getelementptr [2 x i8], ptr addrspace(1) %561, i64 %102, !dbg !18
  %563 = getelementptr [2 x i8], ptr addrspace(1) %562, i64 %558, !dbg !18
  %564 = getelementptr i8, ptr addrspace(1) %563, i64 192, !dbg !18
  %565 = getelementptr [2 x i8], ptr addrspace(1) %564, i64 %108, !dbg !18
  %566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !18
  %567 = zext i32 %566 to i64, !dbg !18
  %568 = inttoptr i64 %567 to ptr addrspace(3), !dbg !18
  %address.local411 = getelementptr [2 x i8], ptr addrspace(1) %565, i64 %address.local293, !dbg !18
  %569 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %570 = ptrtoint ptr addrspace(1) %address.local411 to i64, !dbg !18
  %571 = sub i64 %570, %569, !dbg !18
  %dma.buffer.offset1779 = trunc i64 %571 to i32, !dbg !18
  %572 = trunc i64 %569 to i32, !dbg !18
  %573 = insertelement <4 x i32> poison, i32 %572, i64 0, !dbg !18
  %574 = lshr i64 %569, 32, !dbg !18
  %575 = trunc i64 %574 to i32, !dbg !18
  %576 = insertelement <4 x i32> %573, i32 %575, i64 1, !dbg !18
  %577 = insertelement <4 x i32> %576, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1780 = insertelement <4 x i32> %577, i32 131072, i64 3, !dbg !18
  %578 = ptrtoint ptr addrspace(3) %568 to i32, !dbg !18
  %dma.buffer.lds1781 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %578), !dbg !18
  %579 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1781, i32 %dma.buffer.offset1779, <4 x i32> %dma.buffer.rsrc1780) #10, !dbg !18
  %580 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !18
  %581 = zext i32 %580 to i64, !dbg !18
  %582 = inttoptr i64 %581 to ptr addrspace(3), !dbg !18
  %address.local412 = getelementptr [2 x i8], ptr addrspace(1) %565, i64 %address.local307, !dbg !18
  %583 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %584 = ptrtoint ptr addrspace(1) %address.local412 to i64, !dbg !18
  %585 = sub i64 %584, %583, !dbg !18
  %dma.buffer.offset1776 = trunc i64 %585 to i32, !dbg !18
  %586 = trunc i64 %583 to i32, !dbg !18
  %587 = insertelement <4 x i32> poison, i32 %586, i64 0, !dbg !18
  %588 = lshr i64 %583, 32, !dbg !18
  %589 = trunc i64 %588 to i32, !dbg !18
  %590 = insertelement <4 x i32> %587, i32 %589, i64 1, !dbg !18
  %591 = insertelement <4 x i32> %590, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1777 = insertelement <4 x i32> %591, i32 131072, i64 3, !dbg !18
  %592 = ptrtoint ptr addrspace(3) %582 to i32, !dbg !18
  %dma.buffer.lds1778 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %592), !dbg !18
  %593 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1778, i32 %dma.buffer.offset1776, <4 x i32> %dma.buffer.rsrc1777) #10, !dbg !18
  %594 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !12
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !12
  %597 = fmul <4 x half> %594, splat (half 0xH3015), !dbg !12
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %598 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !12
  %599 = fmul <4 x half> %595, splat (half 0xH3015), !dbg !12
  %600 = fmul <4 x half> %596, splat (half 0xH3015), !dbg !12
  %601 = fmul <4 x half> %598, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %602 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %603 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %602, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %604 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !19
  %605 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %606 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %604, <4 x float> %603) #9, !dbg !19
  %607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %605, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4096, !dbg !19
  %608 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !19
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %609 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !19
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %608, <4 x float> %607) #9, !dbg !19
  %611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %609, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %612 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !19
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local362, i64 512, !dbg !19
  %613 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !19
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %612, <4 x float> %611) #9, !dbg !19
  %615 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %613, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4608, !dbg !19
  %616 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !19
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %617 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !19
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %616, <4 x float> %615) #9, !dbg !19
  %619 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %617, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %620 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !19
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1024, !dbg !19
  %621 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !19
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %620, <4 x float> %619) #9, !dbg !19
  %623 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %621, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5120, !dbg !19
  %624 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !19
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %625 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !19
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %624, <4 x float> %623) #9, !dbg !19
  %627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %625, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %628 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !19
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1536, !dbg !19
  %629 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !19
  %630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %628, <4 x float> %627) #9, !dbg !19
  %631 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %629, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5632, !dbg !19
  %632 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !19
  %633 = load <4 x half>, ptr addrspace(3) %address.local337, align 8, !dbg !19
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %632, <4 x float> %631) #9, !dbg !19
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %633, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4096, !dbg !19
  %636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !19
  %637 = load <4 x half>, ptr addrspace(3) %address.local362, align 8, !dbg !19
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %636, <4 x float> %635) #9, !dbg !19
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %637, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4096, !dbg !19
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !19
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %address.local337, i64 512, !dbg !19
  %641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !19
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %640, <4 x float> %639) #9, !dbg !19
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %641, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local337, i64 4608, !dbg !19
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !19
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local362, i64 512, !dbg !19
  %645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !19
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %644, <4 x float> %643) #9, !dbg !19
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %645, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4608, !dbg !19
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !19
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !19
  %649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !19
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %648, <4 x float> %647) #9, !dbg !19
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %649, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5120, !dbg !19
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !19
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1024, !dbg !19
  %653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !19
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %652, <4 x float> %651) #9, !dbg !19
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %653, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5120, !dbg !19
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !19
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1536, !dbg !19
  %657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !19
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %656, <4 x float> %655) #9, !dbg !19
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %657, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local337, i64 5632, !dbg !19
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !19
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local362, i64 1536, !dbg !19
  %661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !19
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %660, <4 x float> %659) #9, !dbg !19
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %661, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local362, i64 5632, !dbg !19
  %664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !19
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %664, <4 x float> %663) #9, !dbg !19
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  br i1 true, label %666, label %795, !dbg !13

666:                                              ; preds = %70
  %667 = fadd <4 x float> %211, zeroinitializer, !dbg !24
  %668 = fadd <4 x float> %667, %363, !dbg !24
  %669 = fadd <4 x float> %668, %514, !dbg !24
  %670 = fadd <4 x float> %669, %665, !dbg !24
  %671 = extractelement <4 x float> %670, i64 3, !dbg !24
  %672 = extractelement <4 x float> %670, i64 2, !dbg !24
  %673 = extractelement <4 x float> %670, i64 1, !dbg !24
  %674 = extractelement <4 x float> %670, i64 0, !dbg !24
  %675 = fadd <4 x float> %208, zeroinitializer, !dbg !24
  %676 = fadd <4 x float> %675, %360, !dbg !24
  %677 = fadd <4 x float> %676, %511, !dbg !24
  %678 = fadd <4 x float> %677, %662, !dbg !24
  %679 = extractelement <4 x float> %678, i64 3, !dbg !24
  %680 = extractelement <4 x float> %678, i64 2, !dbg !24
  %681 = extractelement <4 x float> %678, i64 1, !dbg !24
  %682 = extractelement <4 x float> %678, i64 0, !dbg !24
  %683 = fadd <4 x float> %204, zeroinitializer, !dbg !24
  %684 = fadd <4 x float> %683, %356, !dbg !24
  %685 = fadd <4 x float> %684, %507, !dbg !24
  %686 = fadd <4 x float> %685, %658, !dbg !24
  %687 = extractelement <4 x float> %686, i64 3, !dbg !24
  %688 = extractelement <4 x float> %686, i64 2, !dbg !24
  %689 = extractelement <4 x float> %686, i64 1, !dbg !24
  %690 = extractelement <4 x float> %686, i64 0, !dbg !24
  %691 = fadd <4 x float> %200, zeroinitializer, !dbg !24
  %692 = fadd <4 x float> %691, %352, !dbg !24
  %693 = fadd <4 x float> %692, %503, !dbg !24
  %694 = fadd <4 x float> %693, %654, !dbg !24
  %695 = extractelement <4 x float> %694, i64 3, !dbg !24
  %696 = extractelement <4 x float> %694, i64 2, !dbg !24
  %697 = extractelement <4 x float> %694, i64 1, !dbg !24
  %698 = extractelement <4 x float> %694, i64 0, !dbg !24
  %699 = fadd <4 x float> %196, zeroinitializer, !dbg !24
  %700 = fadd <4 x float> %699, %348, !dbg !24
  %701 = fadd <4 x float> %700, %499, !dbg !24
  %702 = fadd <4 x float> %701, %650, !dbg !24
  %703 = extractelement <4 x float> %702, i64 3, !dbg !24
  %704 = extractelement <4 x float> %702, i64 2, !dbg !24
  %705 = extractelement <4 x float> %702, i64 1, !dbg !24
  %706 = extractelement <4 x float> %702, i64 0, !dbg !24
  %707 = fadd <4 x float> %192, zeroinitializer, !dbg !24
  %708 = fadd <4 x float> %707, %344, !dbg !24
  %709 = fadd <4 x float> %708, %495, !dbg !24
  %710 = fadd <4 x float> %709, %646, !dbg !24
  %711 = extractelement <4 x float> %710, i64 3, !dbg !24
  %712 = extractelement <4 x float> %710, i64 2, !dbg !24
  %713 = extractelement <4 x float> %710, i64 1, !dbg !24
  %714 = extractelement <4 x float> %710, i64 0, !dbg !24
  %715 = fadd <4 x float> %188, zeroinitializer, !dbg !24
  %716 = fadd <4 x float> %715, %340, !dbg !24
  %717 = fadd <4 x float> %716, %491, !dbg !24
  %718 = fadd <4 x float> %717, %642, !dbg !24
  %719 = extractelement <4 x float> %718, i64 3, !dbg !24
  %720 = extractelement <4 x float> %718, i64 2, !dbg !24
  %721 = extractelement <4 x float> %718, i64 1, !dbg !24
  %722 = extractelement <4 x float> %718, i64 0, !dbg !24
  %723 = fadd <4 x float> %184, zeroinitializer, !dbg !24
  %724 = fadd <4 x float> %723, %336, !dbg !24
  %725 = fadd <4 x float> %724, %487, !dbg !24
  %726 = fadd <4 x float> %725, %638, !dbg !24
  %727 = extractelement <4 x float> %726, i64 3, !dbg !24
  %728 = extractelement <4 x float> %726, i64 2, !dbg !24
  %729 = extractelement <4 x float> %726, i64 1, !dbg !24
  %730 = extractelement <4 x float> %726, i64 0, !dbg !24
  %731 = fadd <4 x float> %180, zeroinitializer, !dbg !24
  %732 = fadd <4 x float> %731, %332, !dbg !24
  %733 = fadd <4 x float> %732, %483, !dbg !24
  %734 = fadd <4 x float> %733, %634, !dbg !24
  %735 = extractelement <4 x float> %734, i64 3, !dbg !24
  %736 = extractelement <4 x float> %734, i64 2, !dbg !24
  %737 = extractelement <4 x float> %734, i64 1, !dbg !24
  %738 = extractelement <4 x float> %734, i64 0, !dbg !24
  %739 = fadd <4 x float> %176, zeroinitializer, !dbg !24
  %740 = fadd <4 x float> %739, %328, !dbg !24
  %741 = fadd <4 x float> %740, %479, !dbg !24
  %742 = fadd <4 x float> %741, %630, !dbg !24
  %743 = extractelement <4 x float> %742, i64 3, !dbg !24
  %744 = extractelement <4 x float> %742, i64 2, !dbg !24
  %745 = extractelement <4 x float> %742, i64 1, !dbg !24
  %746 = extractelement <4 x float> %742, i64 0, !dbg !24
  %747 = fadd <4 x float> %172, zeroinitializer, !dbg !24
  %748 = fadd <4 x float> %747, %324, !dbg !24
  %749 = fadd <4 x float> %748, %475, !dbg !24
  %750 = fadd <4 x float> %749, %626, !dbg !24
  %751 = extractelement <4 x float> %750, i64 3, !dbg !24
  %752 = extractelement <4 x float> %750, i64 2, !dbg !24
  %753 = extractelement <4 x float> %750, i64 1, !dbg !24
  %754 = extractelement <4 x float> %750, i64 0, !dbg !24
  %755 = fadd <4 x float> %168, zeroinitializer, !dbg !24
  %756 = fadd <4 x float> %755, %320, !dbg !24
  %757 = fadd <4 x float> %756, %471, !dbg !24
  %758 = fadd <4 x float> %757, %622, !dbg !24
  %759 = extractelement <4 x float> %758, i64 3, !dbg !24
  %760 = extractelement <4 x float> %758, i64 2, !dbg !24
  %761 = extractelement <4 x float> %758, i64 1, !dbg !24
  %762 = extractelement <4 x float> %758, i64 0, !dbg !24
  %763 = fadd <4 x float> %164, zeroinitializer, !dbg !24
  %764 = fadd <4 x float> %763, %316, !dbg !24
  %765 = fadd <4 x float> %764, %467, !dbg !24
  %766 = fadd <4 x float> %765, %618, !dbg !24
  %767 = extractelement <4 x float> %766, i64 3, !dbg !24
  %768 = extractelement <4 x float> %766, i64 2, !dbg !24
  %769 = extractelement <4 x float> %766, i64 1, !dbg !24
  %770 = extractelement <4 x float> %766, i64 0, !dbg !24
  %771 = fadd <4 x float> %160, zeroinitializer, !dbg !24
  %772 = fadd <4 x float> %771, %312, !dbg !24
  %773 = fadd <4 x float> %772, %463, !dbg !24
  %774 = fadd <4 x float> %773, %614, !dbg !24
  %775 = extractelement <4 x float> %774, i64 3, !dbg !24
  %776 = extractelement <4 x float> %774, i64 2, !dbg !24
  %777 = extractelement <4 x float> %774, i64 1, !dbg !24
  %778 = extractelement <4 x float> %774, i64 0, !dbg !24
  %779 = fadd <4 x float> %156, zeroinitializer, !dbg !24
  %780 = fadd <4 x float> %779, %308, !dbg !24
  %781 = fadd <4 x float> %780, %459, !dbg !24
  %782 = fadd <4 x float> %781, %610, !dbg !24
  %783 = extractelement <4 x float> %782, i64 3, !dbg !24
  %784 = extractelement <4 x float> %782, i64 2, !dbg !24
  %785 = extractelement <4 x float> %782, i64 1, !dbg !24
  %786 = extractelement <4 x float> %782, i64 0, !dbg !24
  %787 = fadd <4 x float> %152, zeroinitializer, !dbg !24
  %788 = fadd <4 x float> %787, %304, !dbg !24
  %789 = fadd <4 x float> %788, %455, !dbg !24
  %790 = fadd <4 x float> %789, %606, !dbg !24
  %791 = extractelement <4 x float> %790, i64 3, !dbg !24
  %792 = extractelement <4 x float> %790, i64 2, !dbg !24
  %793 = extractelement <4 x float> %790, i64 1, !dbg !24
  %794 = extractelement <4 x float> %790, i64 0, !dbg !24
  %address.thread413 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local414 = shl nuw nsw i32 %address.thread413, 5, !dbg !13
  %address.local415 = and i32 %address.local414, 6144, !dbg !13
  %address.local416 = shl nuw nsw i32 %address.thread413, 7, !dbg !13
  %address.local417 = and i32 %address.local416, 1920, !dbg !13
  %address.local418 = or disjoint i32 %address.local415, %address.local417, !dbg !13
  %address.local419 = zext nneg i32 %address.local418 to i64, !dbg !13
  %address.local420 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local421 = zext nneg i32 %address.thread413 to i64, !dbg !12
  %address.local422 = lshr i64 %address.local421, 4, !dbg !12
  %address.local423 = and i64 %address.local422, 3, !dbg !12
  %address.local424 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local423, !dbg !13
  store float %794, ptr addrspace(3) %address.local424, align 4, !dbg !13
  %address.local425 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local426 = or disjoint i64 %address.local423, 4, !dbg !13
  %address.local427 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local425, i64 %address.local426, !dbg !13
  store float %793, ptr addrspace(3) %address.local427, align 4, !dbg !13
  %address.local428 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local429 = or disjoint i64 %address.local423, 8, !dbg !13
  %address.local430 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local428, i64 %address.local429, !dbg !13
  store float %792, ptr addrspace(3) %address.local430, align 4, !dbg !13
  %address.local431 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local432 = or i64 %address.local422, 12, !dbg !13
  %address.local433 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local431, i64 %address.local432, !dbg !13
  store float %791, ptr addrspace(3) %address.local433, align 4, !dbg !13
  %address.local434 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local435 = or disjoint i64 %address.local423, 16, !dbg !13
  %address.local436 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local434, i64 %address.local435, !dbg !13
  store float %786, ptr addrspace(3) %address.local436, align 4, !dbg !13
  %address.local437 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local438 = or disjoint i64 %address.local423, 20, !dbg !13
  %address.local439 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local437, i64 %address.local438, !dbg !13
  store float %785, ptr addrspace(3) %address.local439, align 4, !dbg !13
  %address.local440 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local441 = or disjoint i64 %address.local423, 24, !dbg !13
  %address.local442 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local440, i64 %address.local441, !dbg !13
  store float %784, ptr addrspace(3) %address.local442, align 4, !dbg !13
  %address.local443 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local444 = or i64 %address.local422, 28, !dbg !13
  %address.local445 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local443, i64 %address.local444, !dbg !13
  store float %783, ptr addrspace(3) %address.local445, align 4, !dbg !13
  %address.local446 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local447 = or disjoint i64 %address.local423, 32, !dbg !13
  %address.local448 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local446, i64 %address.local447, !dbg !13
  store float %778, ptr addrspace(3) %address.local448, align 4, !dbg !13
  %address.local449 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local450 = or disjoint i64 %address.local423, 36, !dbg !13
  %address.local451 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local449, i64 %address.local450, !dbg !13
  store float %777, ptr addrspace(3) %address.local451, align 4, !dbg !13
  %address.local452 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local453 = or disjoint i64 %address.local423, 40, !dbg !13
  %address.local454 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local452, i64 %address.local453, !dbg !13
  store float %776, ptr addrspace(3) %address.local454, align 4, !dbg !13
  %address.local455 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local456 = or i64 %address.local422, 44, !dbg !13
  %address.local457 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local455, i64 %address.local456, !dbg !13
  store float %775, ptr addrspace(3) %address.local457, align 4, !dbg !13
  %address.local458 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local459 = or disjoint i64 %address.local423, 48, !dbg !13
  %address.local460 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local458, i64 %address.local459, !dbg !13
  store float %770, ptr addrspace(3) %address.local460, align 4, !dbg !13
  %address.local461 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local462 = or disjoint i64 %address.local423, 52, !dbg !13
  %address.local463 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local461, i64 %address.local462, !dbg !13
  store float %769, ptr addrspace(3) %address.local463, align 4, !dbg !13
  %address.local464 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local465 = or disjoint i64 %address.local423, 56, !dbg !13
  %address.local466 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local464, i64 %address.local465, !dbg !13
  store float %768, ptr addrspace(3) %address.local466, align 4, !dbg !13
  %address.local467 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local468 = or i64 %address.local422, 60, !dbg !13
  %address.local469 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local467, i64 %address.local468, !dbg !13
  store float %767, ptr addrspace(3) %address.local469, align 4, !dbg !13
  %address.local470 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local471 = or disjoint i64 %address.local423, 64, !dbg !13
  %address.local472 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local470, i64 %address.local471, !dbg !13
  store float %762, ptr addrspace(3) %address.local472, align 4, !dbg !13
  %address.local473 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local474 = or disjoint i64 %address.local423, 68, !dbg !13
  %address.local475 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local473, i64 %address.local474, !dbg !13
  store float %761, ptr addrspace(3) %address.local475, align 4, !dbg !13
  %address.local476 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local477 = or disjoint i64 %address.local423, 72, !dbg !13
  %address.local478 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local476, i64 %address.local477, !dbg !13
  store float %760, ptr addrspace(3) %address.local478, align 4, !dbg !13
  %address.local479 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local480 = or i64 %address.local422, 76, !dbg !13
  %address.local481 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local479, i64 %address.local480, !dbg !13
  store float %759, ptr addrspace(3) %address.local481, align 4, !dbg !13
  %address.local482 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local483 = or disjoint i64 %address.local423, 80, !dbg !13
  %address.local484 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local482, i64 %address.local483, !dbg !13
  store float %754, ptr addrspace(3) %address.local484, align 4, !dbg !13
  %address.local485 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local486 = or disjoint i64 %address.local423, 84, !dbg !13
  %address.local487 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local485, i64 %address.local486, !dbg !13
  store float %753, ptr addrspace(3) %address.local487, align 4, !dbg !13
  %address.local488 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local489 = or disjoint i64 %address.local423, 88, !dbg !13
  %address.local490 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local488, i64 %address.local489, !dbg !13
  store float %752, ptr addrspace(3) %address.local490, align 4, !dbg !13
  %address.local491 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local492 = or i64 %address.local422, 92, !dbg !13
  %address.local493 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local491, i64 %address.local492, !dbg !13
  store float %751, ptr addrspace(3) %address.local493, align 4, !dbg !13
  %address.local494 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local495 = or disjoint i64 %address.local423, 96, !dbg !13
  %address.local496 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local494, i64 %address.local495, !dbg !13
  store float %746, ptr addrspace(3) %address.local496, align 4, !dbg !13
  %address.local497 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local498 = or disjoint i64 %address.local423, 100, !dbg !13
  %address.local499 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local497, i64 %address.local498, !dbg !13
  store float %745, ptr addrspace(3) %address.local499, align 4, !dbg !13
  %address.local500 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local501 = or disjoint i64 %address.local423, 104, !dbg !13
  %address.local502 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local500, i64 %address.local501, !dbg !13
  store float %744, ptr addrspace(3) %address.local502, align 4, !dbg !13
  %address.local503 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local504 = or i64 %address.local422, 108, !dbg !13
  %address.local505 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local503, i64 %address.local504, !dbg !13
  store float %743, ptr addrspace(3) %address.local505, align 4, !dbg !13
  %address.local506 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local507 = or disjoint i64 %address.local423, 112, !dbg !13
  %address.local508 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local506, i64 %address.local507, !dbg !13
  store float %738, ptr addrspace(3) %address.local508, align 4, !dbg !13
  %address.local509 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local510 = or disjoint i64 %address.local423, 116, !dbg !13
  %address.local511 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local509, i64 %address.local510, !dbg !13
  store float %737, ptr addrspace(3) %address.local511, align 4, !dbg !13
  %address.local512 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local513 = or disjoint i64 %address.local423, 120, !dbg !13
  %address.local514 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local512, i64 %address.local513, !dbg !13
  store float %736, ptr addrspace(3) %address.local514, align 4, !dbg !13
  %address.local515 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !13
  %address.local516 = or i64 %address.local422, 124, !dbg !13
  %address.local517 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local515, i64 %address.local516, !dbg !13
  store float %735, ptr addrspace(3) %address.local517, align 4, !dbg !13
  %address.local518 = shl nuw nsw i32 %address.thread413, 5, !dbg !13
  %address.local519 = and i32 %address.local518, 6144, !dbg !13
  %address.local520 = shl nuw nsw i32 %address.thread413, 7, !dbg !13
  %address.local521 = and i32 %address.local520, 1920, !dbg !13
  %address.local522 = or disjoint i32 %address.local519, %address.local521, !dbg !13
  %address.local523 = or disjoint i32 %address.local522, 8192, !dbg !13
  %address.local524 = zext nneg i32 %address.local523 to i64, !dbg !13
  %address.local525 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local526 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local423, !dbg !13
  store float %730, ptr addrspace(3) %address.local526, align 4, !dbg !13
  %address.local527 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local528 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local527, i64 %address.local426, !dbg !13
  store float %729, ptr addrspace(3) %address.local528, align 4, !dbg !13
  %address.local529 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local530 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local529, i64 %address.local429, !dbg !13
  store float %728, ptr addrspace(3) %address.local530, align 4, !dbg !13
  %address.local531 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local532 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local531, i64 %address.local432, !dbg !13
  store float %727, ptr addrspace(3) %address.local532, align 4, !dbg !13
  %address.local533 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local534 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local533, i64 %address.local435, !dbg !13
  store float %722, ptr addrspace(3) %address.local534, align 4, !dbg !13
  %address.local535 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local536 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local535, i64 %address.local438, !dbg !13
  store float %721, ptr addrspace(3) %address.local536, align 4, !dbg !13
  %address.local537 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local538 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local537, i64 %address.local441, !dbg !13
  store float %720, ptr addrspace(3) %address.local538, align 4, !dbg !13
  %address.local539 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local540 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local539, i64 %address.local444, !dbg !13
  store float %719, ptr addrspace(3) %address.local540, align 4, !dbg !13
  %address.local541 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local542 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local541, i64 %address.local447, !dbg !13
  store float %714, ptr addrspace(3) %address.local542, align 4, !dbg !13
  %address.local543 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local544 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local543, i64 %address.local450, !dbg !13
  store float %713, ptr addrspace(3) %address.local544, align 4, !dbg !13
  %address.local545 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local546 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local545, i64 %address.local453, !dbg !13
  store float %712, ptr addrspace(3) %address.local546, align 4, !dbg !13
  %address.local547 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local548 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local547, i64 %address.local456, !dbg !13
  store float %711, ptr addrspace(3) %address.local548, align 4, !dbg !13
  %address.local549 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local550 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local549, i64 %address.local459, !dbg !13
  store float %706, ptr addrspace(3) %address.local550, align 4, !dbg !13
  %address.local551 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local552 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local551, i64 %address.local462, !dbg !13
  store float %705, ptr addrspace(3) %address.local552, align 4, !dbg !13
  %address.local553 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local554 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local553, i64 %address.local465, !dbg !13
  store float %704, ptr addrspace(3) %address.local554, align 4, !dbg !13
  %address.local555 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local556 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local555, i64 %address.local468, !dbg !13
  store float %703, ptr addrspace(3) %address.local556, align 4, !dbg !13
  %address.local557 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local558 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local557, i64 %address.local471, !dbg !13
  store float %698, ptr addrspace(3) %address.local558, align 4, !dbg !13
  %address.local559 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local560 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local559, i64 %address.local474, !dbg !13
  store float %697, ptr addrspace(3) %address.local560, align 4, !dbg !13
  %address.local561 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local562 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local561, i64 %address.local477, !dbg !13
  store float %696, ptr addrspace(3) %address.local562, align 4, !dbg !13
  %address.local563 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local564 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local563, i64 %address.local480, !dbg !13
  store float %695, ptr addrspace(3) %address.local564, align 4, !dbg !13
  %address.local565 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local566 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local565, i64 %address.local483, !dbg !13
  store float %690, ptr addrspace(3) %address.local566, align 4, !dbg !13
  %address.local567 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local568 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local567, i64 %address.local486, !dbg !13
  store float %689, ptr addrspace(3) %address.local568, align 4, !dbg !13
  %address.local569 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local570 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local569, i64 %address.local489, !dbg !13
  store float %688, ptr addrspace(3) %address.local570, align 4, !dbg !13
  %address.local571 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local572 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local571, i64 %address.local492, !dbg !13
  store float %687, ptr addrspace(3) %address.local572, align 4, !dbg !13
  %address.local573 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local574 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local573, i64 %address.local495, !dbg !13
  store float %682, ptr addrspace(3) %address.local574, align 4, !dbg !13
  %address.local575 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local576 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local575, i64 %address.local498, !dbg !13
  store float %681, ptr addrspace(3) %address.local576, align 4, !dbg !13
  %address.local577 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local578 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local577, i64 %address.local501, !dbg !13
  store float %680, ptr addrspace(3) %address.local578, align 4, !dbg !13
  %address.local579 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local580 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local579, i64 %address.local504, !dbg !13
  store float %679, ptr addrspace(3) %address.local580, align 4, !dbg !13
  %address.local581 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local582 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local581, i64 %address.local507, !dbg !13
  store float %674, ptr addrspace(3) %address.local582, align 4, !dbg !13
  %address.local583 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local584 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local583, i64 %address.local510, !dbg !13
  store float %673, ptr addrspace(3) %address.local584, align 4, !dbg !13
  %address.local585 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local586 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local585, i64 %address.local513, !dbg !13
  store float %672, ptr addrspace(3) %address.local586, align 4, !dbg !13
  %address.local587 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %address.local588 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local587, i64 %address.local516, !dbg !13
  store float %671, ptr addrspace(3) %address.local588, align 4, !dbg !13
  br label %795, !dbg !13

795:                                              ; preds = %666, %70
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
  %address.local597 = zext nneg i32 %address.thread589 to i64, !dbg !12
  %address.local598 = lshr i64 %address.local597, 4, !dbg !12
  %address.local599 = and i64 %address.local598, 3, !dbg !12
  %address.local600 = shl nuw nsw i64 %address.local599, 2, !dbg !12
  %address.local601 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local596, i64 %address.local600, !dbg !13
  %796 = load float, ptr addrspace(3) %address.local601, align 16, !dbg !13
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local601, i64 4, !dbg !13
  %797 = load float, ptr addrspace(3) %lds.const.gep1682, align 4, !dbg !13
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local601, i64 8, !dbg !13
  %798 = load float, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !13
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local601, i64 12, !dbg !13
  %799 = load float, ptr addrspace(3) %lds.const.gep1684, align 4, !dbg !13
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local601, i64 64, !dbg !13
  %800 = load float, ptr addrspace(3) %lds.const.gep1685, align 16, !dbg !13
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local601, i64 68, !dbg !13
  %801 = load float, ptr addrspace(3) %lds.const.gep1686, align 4, !dbg !13
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local601, i64 72, !dbg !13
  %802 = load float, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !13
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local601, i64 76, !dbg !13
  %803 = load float, ptr addrspace(3) %lds.const.gep1688, align 4, !dbg !13
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local601, i64 128, !dbg !13
  %804 = load float, ptr addrspace(3) %lds.const.gep1689, align 16, !dbg !13
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local601, i64 132, !dbg !13
  %805 = load float, ptr addrspace(3) %lds.const.gep1690, align 4, !dbg !13
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local601, i64 136, !dbg !13
  %806 = load float, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !13
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local601, i64 140, !dbg !13
  %807 = load float, ptr addrspace(3) %lds.const.gep1692, align 4, !dbg !13
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local601, i64 192, !dbg !13
  %808 = load float, ptr addrspace(3) %lds.const.gep1693, align 16, !dbg !13
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local601, i64 196, !dbg !13
  %809 = load float, ptr addrspace(3) %lds.const.gep1694, align 4, !dbg !13
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local601, i64 200, !dbg !13
  %810 = load float, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !13
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local601, i64 204, !dbg !13
  %811 = load float, ptr addrspace(3) %lds.const.gep1696, align 4, !dbg !13
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local601, i64 256, !dbg !13
  %812 = load float, ptr addrspace(3) %lds.const.gep1697, align 16, !dbg !13
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local601, i64 260, !dbg !13
  %813 = load float, ptr addrspace(3) %lds.const.gep1698, align 4, !dbg !13
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local601, i64 264, !dbg !13
  %814 = load float, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !13
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local601, i64 268, !dbg !13
  %815 = load float, ptr addrspace(3) %lds.const.gep1700, align 4, !dbg !13
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local601, i64 320, !dbg !13
  %816 = load float, ptr addrspace(3) %lds.const.gep1701, align 16, !dbg !13
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %address.local601, i64 324, !dbg !13
  %817 = load float, ptr addrspace(3) %lds.const.gep1702, align 4, !dbg !13
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %address.local601, i64 328, !dbg !13
  %818 = load float, ptr addrspace(3) %lds.const.gep1703, align 8, !dbg !13
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local601, i64 332, !dbg !13
  %819 = load float, ptr addrspace(3) %lds.const.gep1704, align 4, !dbg !13
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local601, i64 384, !dbg !13
  %820 = load float, ptr addrspace(3) %lds.const.gep1705, align 16, !dbg !13
  %lds.const.gep1706 = getelementptr i8, ptr addrspace(3) %address.local601, i64 388, !dbg !13
  %821 = load float, ptr addrspace(3) %lds.const.gep1706, align 4, !dbg !13
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %address.local601, i64 392, !dbg !13
  %822 = load float, ptr addrspace(3) %lds.const.gep1707, align 8, !dbg !13
  %lds.const.gep1708 = getelementptr i8, ptr addrspace(3) %address.local601, i64 396, !dbg !13
  %823 = load float, ptr addrspace(3) %lds.const.gep1708, align 4, !dbg !13
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local601, i64 448, !dbg !13
  %824 = load float, ptr addrspace(3) %lds.const.gep1709, align 16, !dbg !13
  %lds.const.gep1710 = getelementptr i8, ptr addrspace(3) %address.local601, i64 452, !dbg !13
  %825 = load float, ptr addrspace(3) %lds.const.gep1710, align 4, !dbg !13
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %address.local601, i64 456, !dbg !13
  %826 = load float, ptr addrspace(3) %lds.const.gep1711, align 8, !dbg !13
  %lds.const.gep1712 = getelementptr i8, ptr addrspace(3) %address.local601, i64 460, !dbg !13
  %827 = load float, ptr addrspace(3) %lds.const.gep1712, align 4, !dbg !13
  %lds.const.gep1713 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32768, !dbg !13
  %828 = load float, ptr addrspace(3) %lds.const.gep1713, align 16, !dbg !13
  %lds.const.gep1714 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32772, !dbg !13
  %829 = load float, ptr addrspace(3) %lds.const.gep1714, align 4, !dbg !13
  %lds.const.gep1715 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32776, !dbg !13
  %830 = load float, ptr addrspace(3) %lds.const.gep1715, align 8, !dbg !13
  %lds.const.gep1716 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32780, !dbg !13
  %831 = load float, ptr addrspace(3) %lds.const.gep1716, align 4, !dbg !13
  %lds.const.gep1717 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32832, !dbg !13
  %832 = load float, ptr addrspace(3) %lds.const.gep1717, align 16, !dbg !13
  %lds.const.gep1718 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32836, !dbg !13
  %833 = load float, ptr addrspace(3) %lds.const.gep1718, align 4, !dbg !13
  %lds.const.gep1719 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32840, !dbg !13
  %834 = load float, ptr addrspace(3) %lds.const.gep1719, align 8, !dbg !13
  %lds.const.gep1720 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32844, !dbg !13
  %835 = load float, ptr addrspace(3) %lds.const.gep1720, align 4, !dbg !13
  %lds.const.gep1721 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32896, !dbg !13
  %836 = load float, ptr addrspace(3) %lds.const.gep1721, align 16, !dbg !13
  %lds.const.gep1722 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32900, !dbg !13
  %837 = load float, ptr addrspace(3) %lds.const.gep1722, align 4, !dbg !13
  %lds.const.gep1723 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32904, !dbg !13
  %838 = load float, ptr addrspace(3) %lds.const.gep1723, align 8, !dbg !13
  %lds.const.gep1724 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32908, !dbg !13
  %839 = load float, ptr addrspace(3) %lds.const.gep1724, align 4, !dbg !13
  %lds.const.gep1725 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32960, !dbg !13
  %840 = load float, ptr addrspace(3) %lds.const.gep1725, align 16, !dbg !13
  %lds.const.gep1726 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32964, !dbg !13
  %841 = load float, ptr addrspace(3) %lds.const.gep1726, align 4, !dbg !13
  %lds.const.gep1727 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32968, !dbg !13
  %842 = load float, ptr addrspace(3) %lds.const.gep1727, align 8, !dbg !13
  %lds.const.gep1728 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32972, !dbg !13
  %843 = load float, ptr addrspace(3) %lds.const.gep1728, align 4, !dbg !13
  %lds.const.gep1729 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33024, !dbg !13
  %844 = load float, ptr addrspace(3) %lds.const.gep1729, align 16, !dbg !13
  %lds.const.gep1730 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33028, !dbg !13
  %845 = load float, ptr addrspace(3) %lds.const.gep1730, align 4, !dbg !13
  %lds.const.gep1731 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33032, !dbg !13
  %846 = load float, ptr addrspace(3) %lds.const.gep1731, align 8, !dbg !13
  %lds.const.gep1732 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33036, !dbg !13
  %847 = load float, ptr addrspace(3) %lds.const.gep1732, align 4, !dbg !13
  %lds.const.gep1733 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33088, !dbg !13
  %848 = load float, ptr addrspace(3) %lds.const.gep1733, align 16, !dbg !13
  %lds.const.gep1734 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33092, !dbg !13
  %849 = load float, ptr addrspace(3) %lds.const.gep1734, align 4, !dbg !13
  %lds.const.gep1735 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33096, !dbg !13
  %850 = load float, ptr addrspace(3) %lds.const.gep1735, align 8, !dbg !13
  %lds.const.gep1736 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33100, !dbg !13
  %851 = load float, ptr addrspace(3) %lds.const.gep1736, align 4, !dbg !13
  %lds.const.gep1737 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33152, !dbg !13
  %852 = load float, ptr addrspace(3) %lds.const.gep1737, align 16, !dbg !13
  %lds.const.gep1738 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33156, !dbg !13
  %853 = load float, ptr addrspace(3) %lds.const.gep1738, align 4, !dbg !13
  %lds.const.gep1739 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33160, !dbg !13
  %854 = load float, ptr addrspace(3) %lds.const.gep1739, align 8, !dbg !13
  %lds.const.gep1740 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33164, !dbg !13
  %855 = load float, ptr addrspace(3) %lds.const.gep1740, align 4, !dbg !13
  %lds.const.gep1741 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33216, !dbg !13
  %856 = load float, ptr addrspace(3) %lds.const.gep1741, align 16, !dbg !13
  %lds.const.gep1742 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33220, !dbg !13
  %857 = load float, ptr addrspace(3) %lds.const.gep1742, align 4, !dbg !13
  %lds.const.gep1743 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33224, !dbg !13
  %858 = load float, ptr addrspace(3) %lds.const.gep1743, align 8, !dbg !13
  %lds.const.gep1744 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33228, !dbg !13
  %859 = load float, ptr addrspace(3) %lds.const.gep1744, align 4, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %860 = or disjoint i64 %71, 127, !dbg !14
  %861 = icmp uge i64 %19, %860, !dbg !14
  %862 = and i1 %42, %861, !dbg !14
  br i1 %862, label %863, label %864, !dbg !14

863:                                              ; preds = %795
  br label %1005, !dbg !14

864:                                              ; preds = %795
  %865 = add i64 %19, %35, !dbg !14
  %866 = add i64 %865, %59, !dbg !14
  %867 = add i64 %866, %60, !dbg !14
  %868 = or disjoint i64 %71, %38, !dbg !14
  %869 = add i64 %867, 1, !dbg !25
  %870 = lshr i64 %869, 32, !dbg !26
  %compare.high = trunc i64 %870 to i32, !dbg !26
  %871 = lshr i64 %868, 32, !dbg !26
  %compare.high1303 = trunc i64 %871 to i32, !dbg !26
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1303, !dbg !26
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1303, !dbg !26
  %compare.low.base = trunc i64 %869 to i32, !dbg !26
  %compare.low.base1304 = trunc i64 %868 to i32, !dbg !26
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1304, !dbg !26
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !26
  %872 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 1) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !26
  %compare.low1305 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !26
  %compare.result1306 = select i1 %compare.high.equal, i1 %compare.low1305, i1 %compare.high.order, !dbg !26
  %873 = select i1 %compare.result1306, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1307 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 2) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1308 = or i32 %compare.local.base1307, 2, !dbg !26
  %compare.low1309 = icmp ugt i32 %compare.low.base, %compare.local1308, !dbg !26
  %compare.result1310 = select i1 %compare.high.equal, i1 %compare.low1309, i1 %compare.high.order, !dbg !26
  %874 = select i1 %compare.result1310, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1311 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 3) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1312 = or i32 %compare.local.base1311, 3, !dbg !26
  %compare.low1313 = icmp ugt i32 %compare.low.base, %compare.local1312, !dbg !26
  %compare.result1314 = select i1 %compare.high.equal, i1 %compare.low1313, i1 %compare.high.order, !dbg !26
  %875 = select i1 %compare.result1314, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1315 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 4) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1316 = or i32 %compare.local.base1315, 16, !dbg !26
  %compare.low1317 = icmp ugt i32 %compare.low.base, %compare.local1316, !dbg !26
  %compare.result1318 = select i1 %compare.high.equal, i1 %compare.low1317, i1 %compare.high.order, !dbg !26
  %876 = select i1 %compare.result1318, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1319 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 5) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1320 = or i32 %compare.local.base1319, 17, !dbg !26
  %compare.low1321 = icmp ugt i32 %compare.low.base, %compare.local1320, !dbg !26
  %compare.result1322 = select i1 %compare.high.equal, i1 %compare.low1321, i1 %compare.high.order, !dbg !26
  %877 = select i1 %compare.result1322, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1323 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 6) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1324 = or i32 %compare.local.base1323, 18, !dbg !26
  %compare.low1325 = icmp ugt i32 %compare.low.base, %compare.local1324, !dbg !26
  %compare.result1326 = select i1 %compare.high.equal, i1 %compare.low1325, i1 %compare.high.order, !dbg !26
  %878 = select i1 %compare.result1326, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 7) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1328 = or i32 %compare.local.base1327, 19, !dbg !26
  %compare.low1329 = icmp ugt i32 %compare.low.base, %compare.local1328, !dbg !26
  %compare.result1330 = select i1 %compare.high.equal, i1 %compare.low1329, i1 %compare.high.order, !dbg !26
  %879 = select i1 %compare.result1330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 8) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1332 = or i32 %compare.local.base1331, 32, !dbg !26
  %compare.low1333 = icmp ugt i32 %compare.low.base, %compare.local1332, !dbg !26
  %compare.result1334 = select i1 %compare.high.equal, i1 %compare.low1333, i1 %compare.high.order, !dbg !26
  %880 = select i1 %compare.result1334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 9) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1336 = or i32 %compare.local.base1335, 33, !dbg !26
  %compare.low1337 = icmp ugt i32 %compare.low.base, %compare.local1336, !dbg !26
  %compare.result1338 = select i1 %compare.high.equal, i1 %compare.low1337, i1 %compare.high.order, !dbg !26
  %881 = select i1 %compare.result1338, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 10) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1340 = or i32 %compare.local.base1339, 34, !dbg !26
  %compare.low1341 = icmp ugt i32 %compare.low.base, %compare.local1340, !dbg !26
  %compare.result1342 = select i1 %compare.high.equal, i1 %compare.low1341, i1 %compare.high.order, !dbg !26
  %882 = select i1 %compare.result1342, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 11) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1344 = or i32 %compare.local.base1343, 35, !dbg !26
  %compare.low1345 = icmp ugt i32 %compare.low.base, %compare.local1344, !dbg !26
  %compare.result1346 = select i1 %compare.high.equal, i1 %compare.low1345, i1 %compare.high.order, !dbg !26
  %883 = select i1 %compare.result1346, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 12) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1348 = or i32 %compare.local.base1347, 48, !dbg !26
  %compare.low1349 = icmp ugt i32 %compare.low.base, %compare.local1348, !dbg !26
  %compare.result1350 = select i1 %compare.high.equal, i1 %compare.low1349, i1 %compare.high.order, !dbg !26
  %884 = select i1 %compare.result1350, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 13) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1352 = or i32 %compare.local.base1351, 49, !dbg !26
  %compare.low1353 = icmp ugt i32 %compare.low.base, %compare.local1352, !dbg !26
  %compare.result1354 = select i1 %compare.high.equal, i1 %compare.low1353, i1 %compare.high.order, !dbg !26
  %885 = select i1 %compare.result1354, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 14) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1356 = or i32 %compare.local.base1355, 50, !dbg !26
  %compare.low1357 = icmp ugt i32 %compare.low.base, %compare.local1356, !dbg !26
  %compare.result1358 = select i1 %compare.high.equal, i1 %compare.low1357, i1 %compare.high.order, !dbg !26
  %886 = select i1 %compare.result1358, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 15) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1360 = or i32 %compare.local.base1359, 51, !dbg !26
  %compare.low1361 = icmp ugt i32 %compare.low.base, %compare.local1360, !dbg !26
  %compare.result1362 = select i1 %compare.high.equal, i1 %compare.low1361, i1 %compare.high.order, !dbg !26
  %887 = select i1 %compare.result1362, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 16) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1364 = or i32 %compare.local.base1363, 64, !dbg !26
  %compare.low1365 = icmp ugt i32 %compare.low.base, %compare.local1364, !dbg !26
  %compare.result1366 = select i1 %compare.high.equal, i1 %compare.low1365, i1 %compare.high.order, !dbg !26
  %888 = select i1 %compare.result1366, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 17) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1368 = or i32 %compare.local.base1367, 65, !dbg !26
  %compare.low1369 = icmp ugt i32 %compare.low.base, %compare.local1368, !dbg !26
  %compare.result1370 = select i1 %compare.high.equal, i1 %compare.low1369, i1 %compare.high.order, !dbg !26
  %889 = select i1 %compare.result1370, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 18) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1372 = or i32 %compare.local.base1371, 66, !dbg !26
  %compare.low1373 = icmp ugt i32 %compare.low.base, %compare.local1372, !dbg !26
  %compare.result1374 = select i1 %compare.high.equal, i1 %compare.low1373, i1 %compare.high.order, !dbg !26
  %890 = select i1 %compare.result1374, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 19) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1376 = or i32 %compare.local.base1375, 67, !dbg !26
  %compare.low1377 = icmp ugt i32 %compare.low.base, %compare.local1376, !dbg !26
  %compare.result1378 = select i1 %compare.high.equal, i1 %compare.low1377, i1 %compare.high.order, !dbg !26
  %891 = select i1 %compare.result1378, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 20) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1380 = or i32 %compare.local.base1379, 80, !dbg !26
  %compare.low1381 = icmp ugt i32 %compare.low.base, %compare.local1380, !dbg !26
  %compare.result1382 = select i1 %compare.high.equal, i1 %compare.low1381, i1 %compare.high.order, !dbg !26
  %892 = select i1 %compare.result1382, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 21) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1384 = or i32 %compare.local.base1383, 81, !dbg !26
  %compare.low1385 = icmp ugt i32 %compare.low.base, %compare.local1384, !dbg !26
  %compare.result1386 = select i1 %compare.high.equal, i1 %compare.low1385, i1 %compare.high.order, !dbg !26
  %893 = select i1 %compare.result1386, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 22) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1388 = or i32 %compare.local.base1387, 82, !dbg !26
  %compare.low1389 = icmp ugt i32 %compare.low.base, %compare.local1388, !dbg !26
  %compare.result1390 = select i1 %compare.high.equal, i1 %compare.low1389, i1 %compare.high.order, !dbg !26
  %894 = select i1 %compare.result1390, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 23) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1392 = or i32 %compare.local.base1391, 83, !dbg !26
  %compare.low1393 = icmp ugt i32 %compare.low.base, %compare.local1392, !dbg !26
  %compare.result1394 = select i1 %compare.high.equal, i1 %compare.low1393, i1 %compare.high.order, !dbg !26
  %895 = select i1 %compare.result1394, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 24) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1396 = or i32 %compare.local.base1395, 96, !dbg !26
  %compare.low1397 = icmp ugt i32 %compare.low.base, %compare.local1396, !dbg !26
  %compare.result1398 = select i1 %compare.high.equal, i1 %compare.low1397, i1 %compare.high.order, !dbg !26
  %896 = select i1 %compare.result1398, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 25) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1400 = or i32 %compare.local.base1399, 97, !dbg !26
  %compare.low1401 = icmp ugt i32 %compare.low.base, %compare.local1400, !dbg !26
  %compare.result1402 = select i1 %compare.high.equal, i1 %compare.low1401, i1 %compare.high.order, !dbg !26
  %897 = select i1 %compare.result1402, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 26) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1404 = or i32 %compare.local.base1403, 98, !dbg !26
  %compare.low1405 = icmp ugt i32 %compare.low.base, %compare.local1404, !dbg !26
  %compare.result1406 = select i1 %compare.high.equal, i1 %compare.low1405, i1 %compare.high.order, !dbg !26
  %898 = select i1 %compare.result1406, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 27) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1408 = or i32 %compare.local.base1407, 99, !dbg !26
  %compare.low1409 = icmp ugt i32 %compare.low.base, %compare.local1408, !dbg !26
  %compare.result1410 = select i1 %compare.high.equal, i1 %compare.low1409, i1 %compare.high.order, !dbg !26
  %899 = select i1 %compare.result1410, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 28) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1412 = or i32 %compare.local.base1411, 112, !dbg !26
  %compare.low1413 = icmp ugt i32 %compare.low.base, %compare.local1412, !dbg !26
  %compare.result1414 = select i1 %compare.high.equal, i1 %compare.low1413, i1 %compare.high.order, !dbg !26
  %900 = select i1 %compare.result1414, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 29) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1416 = or i32 %compare.local.base1415, 113, !dbg !26
  %compare.low1417 = icmp ugt i32 %compare.low.base, %compare.local1416, !dbg !26
  %compare.result1418 = select i1 %compare.high.equal, i1 %compare.low1417, i1 %compare.high.order, !dbg !26
  %901 = select i1 %compare.result1418, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 30) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1420 = or i32 %compare.local.base1419, 114, !dbg !26
  %compare.low1421 = icmp ugt i32 %compare.low.base, %compare.local1420, !dbg !26
  %compare.result1422 = select i1 %compare.high.equal, i1 %compare.low1421, i1 %compare.high.order, !dbg !26
  %902 = select i1 %compare.result1422, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 31) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1424 = or i32 %compare.local.base1423, 115, !dbg !26
  %compare.low1425 = icmp ugt i32 %compare.low.base, %compare.local1424, !dbg !26
  %compare.result1426 = select i1 %compare.high.equal, i1 %compare.low1425, i1 %compare.high.order, !dbg !26
  %903 = select i1 %compare.result1426, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %904 = add i64 %867, 65, !dbg !25
  %905 = lshr i64 %904, 32, !dbg !26
  %compare.high1427 = trunc i64 %905 to i32, !dbg !26
  %906 = lshr i64 %868, 32, !dbg !26
  %compare.high1428 = trunc i64 %906 to i32, !dbg !26
  %compare.high.equal1429 = icmp eq i32 %compare.high1427, %compare.high1428, !dbg !26
  %compare.high.order1430 = icmp ugt i32 %compare.high1427, %compare.high1428, !dbg !26
  %compare.low.base1431 = trunc i64 %904 to i32, !dbg !26
  %compare.low.base1432 = trunc i64 %868 to i32, !dbg !26
  %compare.low1433 = icmp ugt i32 %compare.low.base1431, %compare.low.base1432, !dbg !26
  %compare.result1434 = select i1 %compare.high.equal1429, i1 %compare.low1433, i1 %compare.high.order1430, !dbg !26
  %907 = select i1 %compare.result1434, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 32) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1436 = or i32 %compare.local.base1435, 1, !dbg !26
  %compare.low1437 = icmp ugt i32 %compare.low.base1431, %compare.local1436, !dbg !26
  %compare.result1438 = select i1 %compare.high.equal1429, i1 %compare.low1437, i1 %compare.high.order1430, !dbg !26
  %908 = select i1 %compare.result1438, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 33) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1440 = or i32 %compare.local.base1439, 2, !dbg !26
  %compare.low1441 = icmp ugt i32 %compare.low.base1431, %compare.local1440, !dbg !26
  %compare.result1442 = select i1 %compare.high.equal1429, i1 %compare.low1441, i1 %compare.high.order1430, !dbg !26
  %909 = select i1 %compare.result1442, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 34) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1444 = or i32 %compare.local.base1443, 3, !dbg !26
  %compare.low1445 = icmp ugt i32 %compare.low.base1431, %compare.local1444, !dbg !26
  %compare.result1446 = select i1 %compare.high.equal1429, i1 %compare.low1445, i1 %compare.high.order1430, !dbg !26
  %910 = select i1 %compare.result1446, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1447 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 35) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1448 = or i32 %compare.local.base1447, 16, !dbg !26
  %compare.low1449 = icmp ugt i32 %compare.low.base1431, %compare.local1448, !dbg !26
  %compare.result1450 = select i1 %compare.high.equal1429, i1 %compare.low1449, i1 %compare.high.order1430, !dbg !26
  %911 = select i1 %compare.result1450, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1451 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 36) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1452 = or i32 %compare.local.base1451, 17, !dbg !26
  %compare.low1453 = icmp ugt i32 %compare.low.base1431, %compare.local1452, !dbg !26
  %compare.result1454 = select i1 %compare.high.equal1429, i1 %compare.low1453, i1 %compare.high.order1430, !dbg !26
  %912 = select i1 %compare.result1454, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1455 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 37) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1456 = or i32 %compare.local.base1455, 18, !dbg !26
  %compare.low1457 = icmp ugt i32 %compare.low.base1431, %compare.local1456, !dbg !26
  %compare.result1458 = select i1 %compare.high.equal1429, i1 %compare.low1457, i1 %compare.high.order1430, !dbg !26
  %913 = select i1 %compare.result1458, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 38) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1460 = or i32 %compare.local.base1459, 19, !dbg !26
  %compare.low1461 = icmp ugt i32 %compare.low.base1431, %compare.local1460, !dbg !26
  %compare.result1462 = select i1 %compare.high.equal1429, i1 %compare.low1461, i1 %compare.high.order1430, !dbg !26
  %914 = select i1 %compare.result1462, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 39) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1464 = or i32 %compare.local.base1463, 32, !dbg !26
  %compare.low1465 = icmp ugt i32 %compare.low.base1431, %compare.local1464, !dbg !26
  %compare.result1466 = select i1 %compare.high.equal1429, i1 %compare.low1465, i1 %compare.high.order1430, !dbg !26
  %915 = select i1 %compare.result1466, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 40) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1468 = or i32 %compare.local.base1467, 33, !dbg !26
  %compare.low1469 = icmp ugt i32 %compare.low.base1431, %compare.local1468, !dbg !26
  %compare.result1470 = select i1 %compare.high.equal1429, i1 %compare.low1469, i1 %compare.high.order1430, !dbg !26
  %916 = select i1 %compare.result1470, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 41) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1472 = or i32 %compare.local.base1471, 34, !dbg !26
  %compare.low1473 = icmp ugt i32 %compare.low.base1431, %compare.local1472, !dbg !26
  %compare.result1474 = select i1 %compare.high.equal1429, i1 %compare.low1473, i1 %compare.high.order1430, !dbg !26
  %917 = select i1 %compare.result1474, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 42) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1476 = or i32 %compare.local.base1475, 35, !dbg !26
  %compare.low1477 = icmp ugt i32 %compare.low.base1431, %compare.local1476, !dbg !26
  %compare.result1478 = select i1 %compare.high.equal1429, i1 %compare.low1477, i1 %compare.high.order1430, !dbg !26
  %918 = select i1 %compare.result1478, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 43) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1480 = or i32 %compare.local.base1479, 48, !dbg !26
  %compare.low1481 = icmp ugt i32 %compare.low.base1431, %compare.local1480, !dbg !26
  %compare.result1482 = select i1 %compare.high.equal1429, i1 %compare.low1481, i1 %compare.high.order1430, !dbg !26
  %919 = select i1 %compare.result1482, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 44) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1484 = or i32 %compare.local.base1483, 49, !dbg !26
  %compare.low1485 = icmp ugt i32 %compare.low.base1431, %compare.local1484, !dbg !26
  %compare.result1486 = select i1 %compare.high.equal1429, i1 %compare.low1485, i1 %compare.high.order1430, !dbg !26
  %920 = select i1 %compare.result1486, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1487 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 45) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1488 = or i32 %compare.local.base1487, 50, !dbg !26
  %compare.low1489 = icmp ugt i32 %compare.low.base1431, %compare.local1488, !dbg !26
  %compare.result1490 = select i1 %compare.high.equal1429, i1 %compare.low1489, i1 %compare.high.order1430, !dbg !26
  %921 = select i1 %compare.result1490, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1491 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 46) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1492 = or i32 %compare.local.base1491, 51, !dbg !26
  %compare.low1493 = icmp ugt i32 %compare.low.base1431, %compare.local1492, !dbg !26
  %compare.result1494 = select i1 %compare.high.equal1429, i1 %compare.low1493, i1 %compare.high.order1430, !dbg !26
  %922 = select i1 %compare.result1494, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1495 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 47) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1496 = or i32 %compare.local.base1495, 64, !dbg !26
  %compare.low1497 = icmp ugt i32 %compare.low.base1431, %compare.local1496, !dbg !26
  %compare.result1498 = select i1 %compare.high.equal1429, i1 %compare.low1497, i1 %compare.high.order1430, !dbg !26
  %923 = select i1 %compare.result1498, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1499 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 48) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1500 = or i32 %compare.local.base1499, 65, !dbg !26
  %compare.low1501 = icmp ugt i32 %compare.low.base1431, %compare.local1500, !dbg !26
  %compare.result1502 = select i1 %compare.high.equal1429, i1 %compare.low1501, i1 %compare.high.order1430, !dbg !26
  %924 = select i1 %compare.result1502, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1503 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 49) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1504 = or i32 %compare.local.base1503, 66, !dbg !26
  %compare.low1505 = icmp ugt i32 %compare.low.base1431, %compare.local1504, !dbg !26
  %compare.result1506 = select i1 %compare.high.equal1429, i1 %compare.low1505, i1 %compare.high.order1430, !dbg !26
  %925 = select i1 %compare.result1506, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1507 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 50) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1508 = or i32 %compare.local.base1507, 67, !dbg !26
  %compare.low1509 = icmp ugt i32 %compare.low.base1431, %compare.local1508, !dbg !26
  %compare.result1510 = select i1 %compare.high.equal1429, i1 %compare.low1509, i1 %compare.high.order1430, !dbg !26
  %926 = select i1 %compare.result1510, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1511 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 51) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1512 = or i32 %compare.local.base1511, 80, !dbg !26
  %compare.low1513 = icmp ugt i32 %compare.low.base1431, %compare.local1512, !dbg !26
  %compare.result1514 = select i1 %compare.high.equal1429, i1 %compare.low1513, i1 %compare.high.order1430, !dbg !26
  %927 = select i1 %compare.result1514, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1515 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 52) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1516 = or i32 %compare.local.base1515, 81, !dbg !26
  %compare.low1517 = icmp ugt i32 %compare.low.base1431, %compare.local1516, !dbg !26
  %compare.result1518 = select i1 %compare.high.equal1429, i1 %compare.low1517, i1 %compare.high.order1430, !dbg !26
  %928 = select i1 %compare.result1518, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1519 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 53) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1520 = or i32 %compare.local.base1519, 82, !dbg !26
  %compare.low1521 = icmp ugt i32 %compare.low.base1431, %compare.local1520, !dbg !26
  %compare.result1522 = select i1 %compare.high.equal1429, i1 %compare.low1521, i1 %compare.high.order1430, !dbg !26
  %929 = select i1 %compare.result1522, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 54) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1524 = or i32 %compare.local.base1523, 83, !dbg !26
  %compare.low1525 = icmp ugt i32 %compare.low.base1431, %compare.local1524, !dbg !26
  %compare.result1526 = select i1 %compare.high.equal1429, i1 %compare.low1525, i1 %compare.high.order1430, !dbg !26
  %930 = select i1 %compare.result1526, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 55) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1528 = or i32 %compare.local.base1527, 96, !dbg !26
  %compare.low1529 = icmp ugt i32 %compare.low.base1431, %compare.local1528, !dbg !26
  %compare.result1530 = select i1 %compare.high.equal1429, i1 %compare.low1529, i1 %compare.high.order1430, !dbg !26
  %931 = select i1 %compare.result1530, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 56) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1532 = or i32 %compare.local.base1531, 97, !dbg !26
  %compare.low1533 = icmp ugt i32 %compare.low.base1431, %compare.local1532, !dbg !26
  %compare.result1534 = select i1 %compare.high.equal1429, i1 %compare.low1533, i1 %compare.high.order1430, !dbg !26
  %932 = select i1 %compare.result1534, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 57) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1536 = or i32 %compare.local.base1535, 98, !dbg !26
  %compare.low1537 = icmp ugt i32 %compare.low.base1431, %compare.local1536, !dbg !26
  %compare.result1538 = select i1 %compare.high.equal1429, i1 %compare.low1537, i1 %compare.high.order1430, !dbg !26
  %933 = select i1 %compare.result1538, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 58) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1540 = or i32 %compare.local.base1539, 99, !dbg !26
  %compare.low1541 = icmp ugt i32 %compare.low.base1431, %compare.local1540, !dbg !26
  %compare.result1542 = select i1 %compare.high.equal1429, i1 %compare.low1541, i1 %compare.high.order1430, !dbg !26
  %934 = select i1 %compare.result1542, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 59) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1544 = or i32 %compare.local.base1543, 112, !dbg !26
  %compare.low1545 = icmp ugt i32 %compare.low.base1431, %compare.local1544, !dbg !26
  %compare.result1546 = select i1 %compare.high.equal1429, i1 %compare.low1545, i1 %compare.high.order1430, !dbg !26
  %935 = select i1 %compare.result1546, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 60) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1548 = or i32 %compare.local.base1547, 113, !dbg !26
  %compare.low1549 = icmp ugt i32 %compare.low.base1431, %compare.local1548, !dbg !26
  %compare.result1550 = select i1 %compare.high.equal1429, i1 %compare.low1549, i1 %compare.high.order1430, !dbg !26
  %936 = select i1 %compare.result1550, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 61) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1552 = or i32 %compare.local.base1551, 114, !dbg !26
  %compare.low1553 = icmp ugt i32 %compare.low.base1431, %compare.local1552, !dbg !26
  %compare.result1554 = select i1 %compare.high.equal1429, i1 %compare.low1553, i1 %compare.high.order1430, !dbg !26
  %937 = select i1 %compare.result1554, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 62) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1556 = or i32 %compare.local.base1555, 115, !dbg !26
  %compare.low1557 = icmp ugt i32 %compare.low.base1431, %compare.local1556, !dbg !26
  %compare.result1558 = select i1 %compare.high.equal1429, i1 %compare.low1557, i1 %compare.high.order1430, !dbg !26
  %938 = select i1 %compare.result1558, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %939 = insertelement <32 x float> poison, float %872, i64 0, !dbg !14
  %940 = insertelement <32 x float> %939, float %873, i64 1, !dbg !14
  %941 = insertelement <32 x float> %940, float %874, i64 2, !dbg !14
  %942 = insertelement <32 x float> %941, float %875, i64 3, !dbg !14
  %943 = insertelement <32 x float> %942, float %876, i64 4, !dbg !14
  %944 = insertelement <32 x float> %943, float %877, i64 5, !dbg !14
  %945 = insertelement <32 x float> %944, float %878, i64 6, !dbg !14
  %946 = insertelement <32 x float> %945, float %879, i64 7, !dbg !14
  %947 = insertelement <32 x float> %946, float %880, i64 8, !dbg !14
  %948 = insertelement <32 x float> %947, float %881, i64 9, !dbg !14
  %949 = insertelement <32 x float> %948, float %882, i64 10, !dbg !14
  %950 = insertelement <32 x float> %949, float %883, i64 11, !dbg !14
  %951 = insertelement <32 x float> %950, float %884, i64 12, !dbg !14
  %952 = insertelement <32 x float> %951, float %885, i64 13, !dbg !14
  %953 = insertelement <32 x float> %952, float %886, i64 14, !dbg !14
  %954 = insertelement <32 x float> %953, float %887, i64 15, !dbg !14
  %955 = insertelement <32 x float> %954, float %888, i64 16, !dbg !14
  %956 = insertelement <32 x float> %955, float %889, i64 17, !dbg !14
  %957 = insertelement <32 x float> %956, float %890, i64 18, !dbg !14
  %958 = insertelement <32 x float> %957, float %891, i64 19, !dbg !14
  %959 = insertelement <32 x float> %958, float %892, i64 20, !dbg !14
  %960 = insertelement <32 x float> %959, float %893, i64 21, !dbg !14
  %961 = insertelement <32 x float> %960, float %894, i64 22, !dbg !14
  %962 = insertelement <32 x float> %961, float %895, i64 23, !dbg !14
  %963 = insertelement <32 x float> %962, float %896, i64 24, !dbg !14
  %964 = insertelement <32 x float> %963, float %897, i64 25, !dbg !14
  %965 = insertelement <32 x float> %964, float %898, i64 26, !dbg !14
  %966 = insertelement <32 x float> %965, float %899, i64 27, !dbg !14
  %967 = insertelement <32 x float> %966, float %900, i64 28, !dbg !14
  %968 = insertelement <32 x float> %967, float %901, i64 29, !dbg !14
  %969 = insertelement <32 x float> %968, float %902, i64 30, !dbg !14
  %970 = insertelement <32 x float> %969, float %903, i64 31, !dbg !14
  %971 = insertvalue [2 x <32 x float>] poison, <32 x float> %970, 0, !dbg !14
  %972 = insertelement <32 x float> poison, float %907, i64 0, !dbg !14
  %973 = insertelement <32 x float> %972, float %908, i64 1, !dbg !14
  %974 = insertelement <32 x float> %973, float %909, i64 2, !dbg !14
  %975 = insertelement <32 x float> %974, float %910, i64 3, !dbg !14
  %976 = insertelement <32 x float> %975, float %911, i64 4, !dbg !14
  %977 = insertelement <32 x float> %976, float %912, i64 5, !dbg !14
  %978 = insertelement <32 x float> %977, float %913, i64 6, !dbg !14
  %979 = insertelement <32 x float> %978, float %914, i64 7, !dbg !14
  %980 = insertelement <32 x float> %979, float %915, i64 8, !dbg !14
  %981 = insertelement <32 x float> %980, float %916, i64 9, !dbg !14
  %982 = insertelement <32 x float> %981, float %917, i64 10, !dbg !14
  %983 = insertelement <32 x float> %982, float %918, i64 11, !dbg !14
  %984 = insertelement <32 x float> %983, float %919, i64 12, !dbg !14
  %985 = insertelement <32 x float> %984, float %920, i64 13, !dbg !14
  %986 = insertelement <32 x float> %985, float %921, i64 14, !dbg !14
  %987 = insertelement <32 x float> %986, float %922, i64 15, !dbg !14
  %988 = insertelement <32 x float> %987, float %923, i64 16, !dbg !14
  %989 = insertelement <32 x float> %988, float %924, i64 17, !dbg !14
  %990 = insertelement <32 x float> %989, float %925, i64 18, !dbg !14
  %991 = insertelement <32 x float> %990, float %926, i64 19, !dbg !14
  %992 = insertelement <32 x float> %991, float %927, i64 20, !dbg !14
  %993 = insertelement <32 x float> %992, float %928, i64 21, !dbg !14
  %994 = insertelement <32 x float> %993, float %929, i64 22, !dbg !14
  %995 = insertelement <32 x float> %994, float %930, i64 23, !dbg !14
  %996 = insertelement <32 x float> %995, float %931, i64 24, !dbg !14
  %997 = insertelement <32 x float> %996, float %932, i64 25, !dbg !14
  %998 = insertelement <32 x float> %997, float %933, i64 26, !dbg !14
  %999 = insertelement <32 x float> %998, float %934, i64 27, !dbg !14
  %1000 = insertelement <32 x float> %999, float %935, i64 28, !dbg !14
  %1001 = insertelement <32 x float> %1000, float %936, i64 29, !dbg !14
  %1002 = insertelement <32 x float> %1001, float %937, i64 30, !dbg !14
  %1003 = insertelement <32 x float> %1002, float %938, i64 31, !dbg !14
  %1004 = insertvalue [2 x <32 x float>] %971, <32 x float> %1003, 1, !dbg !14
  br label %1005, !dbg !14

1005:                                             ; preds = %863, %864
  %1006 = phi [2 x <32 x float>] [ %1004, %864 ], [ zeroinitializer, %863 ], !dbg !14
  br label %1007, !dbg !14

1007:                                             ; preds = %1005
  %1008 = extractvalue [2 x <32 x float>] %1006, 0, !dbg !28
  %1009 = insertelement <4 x float> poison, float %796, i64 0, !dbg !28
  %1010 = insertelement <4 x float> %1009, float %797, i64 1, !dbg !28
  %1011 = insertelement <4 x float> %1010, float %798, i64 2, !dbg !28
  %1012 = insertelement <4 x float> %1011, float %799, i64 3, !dbg !28
  %1013 = shufflevector <32 x float> %1008, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1014 = fadd <4 x float> %1012, %1013, !dbg !28
  %1015 = extractelement <4 x float> %1014, i64 0, !dbg !28
  %1016 = extractelement <4 x float> %1014, i64 1, !dbg !28
  %1017 = extractelement <4 x float> %1014, i64 2, !dbg !28
  %1018 = extractelement <4 x float> %1014, i64 3, !dbg !28
  %1019 = insertelement <4 x float> poison, float %800, i64 0, !dbg !28
  %1020 = insertelement <4 x float> %1019, float %801, i64 1, !dbg !28
  %1021 = insertelement <4 x float> %1020, float %802, i64 2, !dbg !28
  %1022 = insertelement <4 x float> %1021, float %803, i64 3, !dbg !28
  %1023 = shufflevector <32 x float> %1008, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1024 = fadd <4 x float> %1022, %1023, !dbg !28
  %1025 = extractelement <4 x float> %1024, i64 0, !dbg !28
  %1026 = extractelement <4 x float> %1024, i64 1, !dbg !28
  %1027 = extractelement <4 x float> %1024, i64 2, !dbg !28
  %1028 = extractelement <4 x float> %1024, i64 3, !dbg !28
  %1029 = insertelement <4 x float> poison, float %804, i64 0, !dbg !28
  %1030 = insertelement <4 x float> %1029, float %805, i64 1, !dbg !28
  %1031 = insertelement <4 x float> %1030, float %806, i64 2, !dbg !28
  %1032 = insertelement <4 x float> %1031, float %807, i64 3, !dbg !28
  %1033 = shufflevector <32 x float> %1008, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !28
  %1034 = fadd <4 x float> %1032, %1033, !dbg !28
  %1035 = extractelement <4 x float> %1034, i64 0, !dbg !28
  %1036 = extractelement <4 x float> %1034, i64 1, !dbg !28
  %1037 = extractelement <4 x float> %1034, i64 2, !dbg !28
  %1038 = extractelement <4 x float> %1034, i64 3, !dbg !28
  %1039 = insertelement <4 x float> poison, float %808, i64 0, !dbg !28
  %1040 = insertelement <4 x float> %1039, float %809, i64 1, !dbg !28
  %1041 = insertelement <4 x float> %1040, float %810, i64 2, !dbg !28
  %1042 = insertelement <4 x float> %1041, float %811, i64 3, !dbg !28
  %1043 = shufflevector <32 x float> %1008, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !28
  %1044 = fadd <4 x float> %1042, %1043, !dbg !28
  %1045 = extractelement <4 x float> %1044, i64 0, !dbg !28
  %1046 = extractelement <4 x float> %1044, i64 1, !dbg !28
  %1047 = extractelement <4 x float> %1044, i64 2, !dbg !28
  %1048 = extractelement <4 x float> %1044, i64 3, !dbg !28
  %1049 = insertelement <4 x float> poison, float %812, i64 0, !dbg !28
  %1050 = insertelement <4 x float> %1049, float %813, i64 1, !dbg !28
  %1051 = insertelement <4 x float> %1050, float %814, i64 2, !dbg !28
  %1052 = insertelement <4 x float> %1051, float %815, i64 3, !dbg !28
  %1053 = shufflevector <32 x float> %1008, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !28
  %1054 = fadd <4 x float> %1052, %1053, !dbg !28
  %1055 = extractelement <4 x float> %1054, i64 0, !dbg !28
  %1056 = extractelement <4 x float> %1054, i64 1, !dbg !28
  %1057 = extractelement <4 x float> %1054, i64 2, !dbg !28
  %1058 = extractelement <4 x float> %1054, i64 3, !dbg !28
  %1059 = insertelement <4 x float> poison, float %816, i64 0, !dbg !28
  %1060 = insertelement <4 x float> %1059, float %817, i64 1, !dbg !28
  %1061 = insertelement <4 x float> %1060, float %818, i64 2, !dbg !28
  %1062 = insertelement <4 x float> %1061, float %819, i64 3, !dbg !28
  %1063 = shufflevector <32 x float> %1008, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !28
  %1064 = fadd <4 x float> %1062, %1063, !dbg !28
  %1065 = extractelement <4 x float> %1064, i64 0, !dbg !28
  %1066 = extractelement <4 x float> %1064, i64 1, !dbg !28
  %1067 = extractelement <4 x float> %1064, i64 2, !dbg !28
  %1068 = extractelement <4 x float> %1064, i64 3, !dbg !28
  %1069 = insertelement <4 x float> poison, float %820, i64 0, !dbg !28
  %1070 = insertelement <4 x float> %1069, float %821, i64 1, !dbg !28
  %1071 = insertelement <4 x float> %1070, float %822, i64 2, !dbg !28
  %1072 = insertelement <4 x float> %1071, float %823, i64 3, !dbg !28
  %1073 = shufflevector <32 x float> %1008, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !28
  %1074 = fadd <4 x float> %1072, %1073, !dbg !28
  %1075 = extractelement <4 x float> %1074, i64 0, !dbg !28
  %1076 = extractelement <4 x float> %1074, i64 1, !dbg !28
  %1077 = extractelement <4 x float> %1074, i64 2, !dbg !28
  %1078 = extractelement <4 x float> %1074, i64 3, !dbg !28
  %1079 = insertelement <4 x float> poison, float %824, i64 0, !dbg !28
  %1080 = insertelement <4 x float> %1079, float %825, i64 1, !dbg !28
  %1081 = insertelement <4 x float> %1080, float %826, i64 2, !dbg !28
  %1082 = insertelement <4 x float> %1081, float %827, i64 3, !dbg !28
  %1083 = shufflevector <32 x float> %1008, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !28
  %1084 = fadd <4 x float> %1082, %1083, !dbg !28
  %1085 = extractelement <4 x float> %1084, i64 0, !dbg !28
  %1086 = extractelement <4 x float> %1084, i64 1, !dbg !28
  %1087 = extractelement <4 x float> %1084, i64 2, !dbg !28
  %1088 = extractelement <4 x float> %1084, i64 3, !dbg !28
  %1089 = extractvalue [2 x <32 x float>] %1006, 1, !dbg !28
  %1090 = insertelement <4 x float> poison, float %828, i64 0, !dbg !28
  %1091 = insertelement <4 x float> %1090, float %829, i64 1, !dbg !28
  %1092 = insertelement <4 x float> %1091, float %830, i64 2, !dbg !28
  %1093 = insertelement <4 x float> %1092, float %831, i64 3, !dbg !28
  %1094 = shufflevector <32 x float> %1089, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1095 = fadd <4 x float> %1093, %1094, !dbg !28
  %1096 = extractelement <4 x float> %1095, i64 0, !dbg !28
  %1097 = extractelement <4 x float> %1095, i64 1, !dbg !28
  %1098 = extractelement <4 x float> %1095, i64 2, !dbg !28
  %1099 = extractelement <4 x float> %1095, i64 3, !dbg !28
  %1100 = insertelement <4 x float> poison, float %832, i64 0, !dbg !28
  %1101 = insertelement <4 x float> %1100, float %833, i64 1, !dbg !28
  %1102 = insertelement <4 x float> %1101, float %834, i64 2, !dbg !28
  %1103 = insertelement <4 x float> %1102, float %835, i64 3, !dbg !28
  %1104 = shufflevector <32 x float> %1089, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1105 = fadd <4 x float> %1103, %1104, !dbg !28
  %1106 = extractelement <4 x float> %1105, i64 0, !dbg !28
  %1107 = extractelement <4 x float> %1105, i64 1, !dbg !28
  %1108 = extractelement <4 x float> %1105, i64 2, !dbg !28
  %1109 = extractelement <4 x float> %1105, i64 3, !dbg !28
  %1110 = insertelement <4 x float> poison, float %836, i64 0, !dbg !28
  %1111 = insertelement <4 x float> %1110, float %837, i64 1, !dbg !28
  %1112 = insertelement <4 x float> %1111, float %838, i64 2, !dbg !28
  %1113 = insertelement <4 x float> %1112, float %839, i64 3, !dbg !28
  %1114 = shufflevector <32 x float> %1089, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !28
  %1115 = fadd <4 x float> %1113, %1114, !dbg !28
  %1116 = extractelement <4 x float> %1115, i64 0, !dbg !28
  %1117 = extractelement <4 x float> %1115, i64 1, !dbg !28
  %1118 = extractelement <4 x float> %1115, i64 2, !dbg !28
  %1119 = extractelement <4 x float> %1115, i64 3, !dbg !28
  %1120 = insertelement <4 x float> poison, float %840, i64 0, !dbg !28
  %1121 = insertelement <4 x float> %1120, float %841, i64 1, !dbg !28
  %1122 = insertelement <4 x float> %1121, float %842, i64 2, !dbg !28
  %1123 = insertelement <4 x float> %1122, float %843, i64 3, !dbg !28
  %1124 = shufflevector <32 x float> %1089, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !28
  %1125 = fadd <4 x float> %1123, %1124, !dbg !28
  %1126 = extractelement <4 x float> %1125, i64 0, !dbg !28
  %1127 = extractelement <4 x float> %1125, i64 1, !dbg !28
  %1128 = extractelement <4 x float> %1125, i64 2, !dbg !28
  %1129 = extractelement <4 x float> %1125, i64 3, !dbg !28
  %1130 = insertelement <4 x float> poison, float %844, i64 0, !dbg !28
  %1131 = insertelement <4 x float> %1130, float %845, i64 1, !dbg !28
  %1132 = insertelement <4 x float> %1131, float %846, i64 2, !dbg !28
  %1133 = insertelement <4 x float> %1132, float %847, i64 3, !dbg !28
  %1134 = shufflevector <32 x float> %1089, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !28
  %1135 = fadd <4 x float> %1133, %1134, !dbg !28
  %1136 = extractelement <4 x float> %1135, i64 0, !dbg !28
  %1137 = extractelement <4 x float> %1135, i64 1, !dbg !28
  %1138 = extractelement <4 x float> %1135, i64 2, !dbg !28
  %1139 = extractelement <4 x float> %1135, i64 3, !dbg !28
  %1140 = insertelement <4 x float> poison, float %848, i64 0, !dbg !28
  %1141 = insertelement <4 x float> %1140, float %849, i64 1, !dbg !28
  %1142 = insertelement <4 x float> %1141, float %850, i64 2, !dbg !28
  %1143 = insertelement <4 x float> %1142, float %851, i64 3, !dbg !28
  %1144 = shufflevector <32 x float> %1089, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !28
  %1145 = fadd <4 x float> %1143, %1144, !dbg !28
  %1146 = extractelement <4 x float> %1145, i64 0, !dbg !28
  %1147 = extractelement <4 x float> %1145, i64 1, !dbg !28
  %1148 = extractelement <4 x float> %1145, i64 2, !dbg !28
  %1149 = extractelement <4 x float> %1145, i64 3, !dbg !28
  %1150 = insertelement <4 x float> poison, float %852, i64 0, !dbg !28
  %1151 = insertelement <4 x float> %1150, float %853, i64 1, !dbg !28
  %1152 = insertelement <4 x float> %1151, float %854, i64 2, !dbg !28
  %1153 = insertelement <4 x float> %1152, float %855, i64 3, !dbg !28
  %1154 = shufflevector <32 x float> %1089, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !28
  %1155 = fadd <4 x float> %1153, %1154, !dbg !28
  %1156 = extractelement <4 x float> %1155, i64 0, !dbg !28
  %1157 = extractelement <4 x float> %1155, i64 1, !dbg !28
  %1158 = extractelement <4 x float> %1155, i64 2, !dbg !28
  %1159 = extractelement <4 x float> %1155, i64 3, !dbg !28
  %1160 = insertelement <4 x float> poison, float %856, i64 0, !dbg !28
  %1161 = insertelement <4 x float> %1160, float %857, i64 1, !dbg !28
  %1162 = insertelement <4 x float> %1161, float %858, i64 2, !dbg !28
  %1163 = insertelement <4 x float> %1162, float %859, i64 3, !dbg !28
  %1164 = shufflevector <32 x float> %1089, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !28
  %1165 = fadd <4 x float> %1163, %1164, !dbg !28
  %1166 = extractelement <4 x float> %1165, i64 0, !dbg !28
  %1167 = extractelement <4 x float> %1165, i64 1, !dbg !28
  %1168 = extractelement <4 x float> %1165, i64 2, !dbg !28
  %1169 = extractelement <4 x float> %1165, i64 3, !dbg !28
  %1170 = call float @llvm.maxnum.f32(float %1015, float 0xFFF0000000000000), !dbg !29, !frisk.local_reduction !1
  %local.reduce182 = call float @llvm.maxnum.f32(float %1170, float %1025), !dbg !29, !frisk.local_reduction !1
  %local.reduce183 = call float @llvm.maxnum.f32(float %1016, float %1026), !dbg !29, !frisk.local_reduction !1
  %local.reduce184 = call float @llvm.maxnum.f32(float %1017, float %1027), !dbg !29, !frisk.local_reduction !1
  %local.reduce185 = call float @llvm.maxnum.f32(float %1018, float %1028), !dbg !29, !frisk.local_reduction !1
  %local.reduce186 = call float @llvm.maxnum.f32(float %local.reduce182, float %1035), !dbg !29, !frisk.local_reduction !1
  %local.reduce187 = call float @llvm.maxnum.f32(float %local.reduce183, float %1036), !dbg !29, !frisk.local_reduction !1
  %local.reduce188 = call float @llvm.maxnum.f32(float %local.reduce184, float %1037), !dbg !29, !frisk.local_reduction !1
  %local.reduce189 = call float @llvm.maxnum.f32(float %local.reduce185, float %1038), !dbg !29, !frisk.local_reduction !1
  %local.reduce190 = call float @llvm.maxnum.f32(float %local.reduce186, float %1045), !dbg !29, !frisk.local_reduction !1
  %local.reduce191 = call float @llvm.maxnum.f32(float %local.reduce187, float %1046), !dbg !29, !frisk.local_reduction !1
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %1047), !dbg !29, !frisk.local_reduction !1
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %1048), !dbg !29, !frisk.local_reduction !1
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %1055), !dbg !29, !frisk.local_reduction !1
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %1056), !dbg !29, !frisk.local_reduction !1
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %1057), !dbg !29, !frisk.local_reduction !1
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %1058), !dbg !29, !frisk.local_reduction !1
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %1065), !dbg !29, !frisk.local_reduction !1
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %1066), !dbg !29, !frisk.local_reduction !1
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %1067), !dbg !29, !frisk.local_reduction !1
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %1068), !dbg !29, !frisk.local_reduction !1
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %1075), !dbg !29, !frisk.local_reduction !1
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %1076), !dbg !29, !frisk.local_reduction !1
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %1077), !dbg !29, !frisk.local_reduction !1
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %1078), !dbg !29, !frisk.local_reduction !1
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %1085), !dbg !29, !frisk.local_reduction !1
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %1086), !dbg !29, !frisk.local_reduction !1
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %1087), !dbg !29, !frisk.local_reduction !1
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %1088), !dbg !29, !frisk.local_reduction !1
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %local.reduce207), !dbg !29, !frisk.local_reduction !1
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce208, float %local.reduce209), !dbg !29, !frisk.local_reduction !1
  %1171 = call float @llvm.maxnum.f32(float %local.reduce210, float %local.reduce211), !dbg !29, !frisk.local_reduction !1
  %1172 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !29
  %1173 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1172), !dbg !29
  %1174 = add nuw nsw i32 %1173, 64, !dbg !29
  %1175 = xor i32 %1173, 16, !dbg !29
  %1176 = and i32 %1174, 192, !dbg !29
  %1177 = icmp samesign ult i32 %1175, %1176, !dbg !29
  %1178 = select i1 %1177, i32 %1175, i32 %1173, !dbg !29
  %1179 = shl nuw nsw i32 %1178, 2, !dbg !29
  %1180 = bitcast float %1171 to i32, !dbg !29
  %1181 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1180), !dbg !29
  %1182 = bitcast i32 %1181 to float, !dbg !29
  %1183 = call float @llvm.maxnum.f32(float %1171, float %1182), !dbg !29
  %1184 = xor i32 %1173, 32, !dbg !29
  %1185 = icmp samesign ult i32 %1184, %1176, !dbg !29
  %1186 = select i1 %1185, i32 %1184, i32 %1173, !dbg !29
  %1187 = shl nuw nsw i32 %1186, 2, !dbg !29
  %1188 = bitcast float %1183 to i32, !dbg !29
  %1189 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1188), !dbg !29
  %1190 = bitcast i32 %1189 to float, !dbg !29
  %1191 = call float @llvm.maxnum.f32(float %1183, float %1190), !dbg !29
  %1192 = extractvalue [2 x <1 x float>] %65, 0, !dbg !29
  %1193 = insertelement <1 x float> poison, float %1191, i64 0, !dbg !29
  %1194 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1193, <1 x float> %1192), !dbg !29
  %1195 = call float @llvm.maxnum.f32(float %1096, float 0xFFF0000000000000), !dbg !29, !frisk.local_reduction !1
  %local.reduce151 = call float @llvm.maxnum.f32(float %1195, float %1106), !dbg !29, !frisk.local_reduction !1
  %local.reduce152 = call float @llvm.maxnum.f32(float %1097, float %1107), !dbg !29, !frisk.local_reduction !1
  %local.reduce153 = call float @llvm.maxnum.f32(float %1098, float %1108), !dbg !29, !frisk.local_reduction !1
  %local.reduce154 = call float @llvm.maxnum.f32(float %1099, float %1109), !dbg !29, !frisk.local_reduction !1
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %1116), !dbg !29, !frisk.local_reduction !1
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %1117), !dbg !29, !frisk.local_reduction !1
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %1118), !dbg !29, !frisk.local_reduction !1
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %1119), !dbg !29, !frisk.local_reduction !1
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %1126), !dbg !29, !frisk.local_reduction !1
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %1127), !dbg !29, !frisk.local_reduction !1
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %1128), !dbg !29, !frisk.local_reduction !1
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %1129), !dbg !29, !frisk.local_reduction !1
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %1136), !dbg !29, !frisk.local_reduction !1
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %1137), !dbg !29, !frisk.local_reduction !1
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %1138), !dbg !29, !frisk.local_reduction !1
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %1139), !dbg !29, !frisk.local_reduction !1
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %1146), !dbg !29, !frisk.local_reduction !1
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %1147), !dbg !29, !frisk.local_reduction !1
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %1148), !dbg !29, !frisk.local_reduction !1
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %1149), !dbg !29, !frisk.local_reduction !1
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %1156), !dbg !29, !frisk.local_reduction !1
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %1157), !dbg !29, !frisk.local_reduction !1
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %1158), !dbg !29, !frisk.local_reduction !1
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %1159), !dbg !29, !frisk.local_reduction !1
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %1166), !dbg !29, !frisk.local_reduction !1
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %1167), !dbg !29, !frisk.local_reduction !1
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %1168), !dbg !29, !frisk.local_reduction !1
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %1169), !dbg !29, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !29, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !29, !frisk.local_reduction !1
  %1196 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !29, !frisk.local_reduction !1
  %1197 = bitcast float %1196 to i32, !dbg !29
  %1198 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1197), !dbg !29
  %1199 = bitcast i32 %1198 to float, !dbg !29
  %1200 = call float @llvm.maxnum.f32(float %1196, float %1199), !dbg !29
  %1201 = bitcast float %1200 to i32, !dbg !29
  %1202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1201), !dbg !29
  %1203 = bitcast i32 %1202 to float, !dbg !29
  %1204 = call float @llvm.maxnum.f32(float %1200, float %1203), !dbg !29
  %1205 = extractvalue [2 x <1 x float>] %65, 1, !dbg !29
  %1206 = insertelement <1 x float> poison, float %1204, i64 0, !dbg !29
  %1207 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1206, <1 x float> %1205), !dbg !29
  %1208 = shufflevector <1 x float> %1194, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %1209 = fsub <4 x float> %1014, %1208, !dbg !30
  %1210 = fsub <4 x float> %1024, %1208, !dbg !30
  %1211 = fsub <4 x float> %1034, %1208, !dbg !30
  %1212 = fsub <4 x float> %1044, %1208, !dbg !30
  %1213 = fsub <4 x float> %1054, %1208, !dbg !30
  %1214 = fsub <4 x float> %1064, %1208, !dbg !30
  %1215 = fsub <4 x float> %1074, %1208, !dbg !30
  %1216 = fsub <4 x float> %1084, %1208, !dbg !30
  %1217 = shufflevector <1 x float> %1207, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %1218 = fsub <4 x float> %1095, %1217, !dbg !30
  %1219 = fsub <4 x float> %1105, %1217, !dbg !30
  %1220 = fsub <4 x float> %1115, %1217, !dbg !30
  %1221 = fsub <4 x float> %1125, %1217, !dbg !30
  %1222 = fsub <4 x float> %1135, %1217, !dbg !30
  %1223 = fsub <4 x float> %1145, %1217, !dbg !30
  %1224 = fsub <4 x float> %1155, %1217, !dbg !30
  %1225 = fsub <4 x float> %1165, %1217, !dbg !30
  br i1 %43, label %1226, label %1231, !dbg !15

1226:                                             ; preds = %1007
  %1227 = fsub <1 x float> %1205, %1207, !dbg !31
  %1228 = extractelement <1 x float> %1227, i64 0, !dbg !31
  %1229 = fsub <1 x float> %1192, %1194, !dbg !31
  %1230 = extractelement <1 x float> %1229, i64 0, !dbg !31
  %address.thread766 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !40, !frisk.register_lifetimes !1
  %address.local767 = lshr i32 %address.thread766, 2, !dbg !13
  %address.local768 = and i32 %address.local767, 48, !dbg !13
  %address.local769 = and i32 %address.thread766, 15, !dbg !13
  %address.local770 = or disjoint i32 %address.local768, %address.local769, !dbg !13
  %address.local771 = zext nneg i32 %address.local770 to i64, !dbg !13
  %address.local772 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local771, !dbg !15
  store float %1230, ptr addrspace(3) %address.local772, align 4, !dbg !15
  %address.local773 = lshr i32 %address.thread766, 2, !dbg !13
  %address.local774 = and i32 %address.local773, 48, !dbg !13
  %address.local775 = and i32 %address.thread766, 15, !dbg !13
  %address.local776 = or disjoint i32 %address.local774, %address.local775, !dbg !13
  %address.local777 = or disjoint i32 %address.local776, 64, !dbg !13
  %address.local778 = zext nneg i32 %address.local777 to i64, !dbg !13
  %address.local779 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local778, !dbg !15
  store float %1228, ptr addrspace(3) %address.local779, align 4, !dbg !15
  br label %1231, !dbg !15

1231:                                             ; preds = %1226, %1007
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
  %1232 = load float, ptr addrspace(3) %address.local786, align 4, !dbg !15
  %lds.const.gep1745 = getelementptr i8, ptr addrspace(3) %address.local786, i64 64, !dbg !15
  %1233 = load float, ptr addrspace(3) %lds.const.gep1745, align 4, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1234 = call float @llvm.amdgcn.exp2.f32(float %1232), !dbg !15
  %1235 = call float @llvm.amdgcn.exp2.f32(float %1233), !dbg !15
  br i1 %43, label %1236, label %1237, !dbg !32

1236:                                             ; preds = %1231
  %address.thread789 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !32, !range !40, !frisk.register_lifetimes !1
  %address.local790 = lshr i32 %address.thread789, 1, !dbg !15
  %address.local791 = and i32 %address.local790, 96, !dbg !15
  %address.local792 = and i32 %address.thread789, 15, !dbg !15
  %address.local793 = or disjoint i32 %address.local791, %address.local792, !dbg !15
  %address.local794 = zext nneg i32 %address.local793 to i64, !dbg !15
  %address.local795 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !15
  store float %1234, ptr addrspace(3) %address.local795, align 4, !dbg !32
  %address.local796 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !15
  %address.local797 = getelementptr i8, ptr addrspace(3) %address.local796, i64 64, !dbg !15
  store float %1235, ptr addrspace(3) %address.local797, align 4, !dbg !32
  br label %1237, !dbg !32

1237:                                             ; preds = %1236, %1231
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
  %1238 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !32
  %lds.const.gep1746 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !32
  %1239 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !32
  fence syncscope("workgroup") release, !dbg !32
  call void @llvm.amdgcn.s.barrier(), !dbg !32
  fence syncscope("workgroup") acquire, !dbg !32
  %1240 = add nuw nsw i64 %102, %15, !dbg !20
  %1241 = shl i64 %1240, 19, !dbg !20
  %1242 = and i64 %1241, 16252928, !dbg !20
  %1243 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1242, !dbg !20
  %1244 = getelementptr [2 x i8], ptr addrspace(1) %1243, i64 %108, !dbg !20
  %address.local812 = zext nneg i32 %address.thread798 to i64, !dbg !12
  %address.local813 = shl nuw nsw i64 %address.local812, 3, !dbg !11
  %address.local814 = and i64 %address.local813, 1536, !dbg !11
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local814, !dbg !11
  %address.local816 = ptrtoint ptr addrspace(3) %address.local815 to i64, !dbg !11
  %address.local817 = trunc i64 %address.local816 to i32, !dbg !11
  %1245 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local817), !dbg !20
  %1246 = zext i32 %1245 to i64, !dbg !20
  %1247 = inttoptr i64 %1246 to ptr addrspace(3), !dbg !20
  %address.local818 = shl nuw nsw i64 %address.local812, 5, !dbg !20
  %address.local819 = and i64 %address.local818, 8064, !dbg !20
  %address.local820 = shl nuw nsw i64 %address.local812, 3, !dbg !20
  %address.local821 = and i64 %address.local820, 24, !dbg !20
  %address.local822 = or disjoint i64 %address.local819, %address.local821, !dbg !20
  %address.local823 = getelementptr [2 x i8], ptr addrspace(1) %1244, i64 %address.local822, !dbg !20
  %1248 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1249 = ptrtoint ptr addrspace(1) %address.local823 to i64, !dbg !20
  %1250 = sub i64 %1249, %1248, !dbg !20
  %dma.buffer.offset1773 = trunc i64 %1250 to i32, !dbg !20
  %1251 = trunc i64 %1248 to i32, !dbg !20
  %1252 = insertelement <4 x i32> poison, i32 %1251, i64 0, !dbg !20
  %1253 = lshr i64 %1248, 32, !dbg !20
  %1254 = trunc i64 %1253 to i32, !dbg !20
  %1255 = insertelement <4 x i32> %1252, i32 %1254, i64 1, !dbg !20
  %1256 = insertelement <4 x i32> %1255, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1774 = insertelement <4 x i32> %1256, i32 131072, i64 3, !dbg !20
  %1257 = ptrtoint ptr addrspace(3) %1247 to i32, !dbg !20
  %dma.buffer.lds1775 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1257), !dbg !20
  %1258 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1775, i32 %dma.buffer.offset1773, <4 x i32> %dma.buffer.rsrc1774) #10, !dbg !20
  %address.local824 = or disjoint i64 %address.local812, 256, !dbg !11
  %address.local825 = shl nuw nsw i64 %address.local824, 3, !dbg !11
  %address.local826 = and i64 %address.local825, 3584, !dbg !11
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local826, !dbg !11
  %address.local828 = ptrtoint ptr addrspace(3) %address.local827 to i64, !dbg !11
  %address.local829 = trunc i64 %address.local828 to i32, !dbg !11
  %1259 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local829), !dbg !20
  %1260 = zext i32 %1259 to i64, !dbg !20
  %1261 = inttoptr i64 %1260 to ptr addrspace(3), !dbg !20
  %address.local830 = shl nuw nsw i64 %address.local824, 5, !dbg !20
  %address.local831 = and i64 %address.local830, 16256, !dbg !20
  %address.local832 = shl nuw nsw i64 %address.local812, 3, !dbg !20
  %address.local833 = and i64 %address.local832, 24, !dbg !20
  %address.local834 = or disjoint i64 %address.local831, %address.local833, !dbg !20
  %address.local835 = getelementptr [2 x i8], ptr addrspace(1) %1244, i64 %address.local834, !dbg !20
  %1262 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1263 = ptrtoint ptr addrspace(1) %address.local835 to i64, !dbg !20
  %1264 = sub i64 %1263, %1262, !dbg !20
  %dma.buffer.offset1770 = trunc i64 %1264 to i32, !dbg !20
  %1265 = trunc i64 %1262 to i32, !dbg !20
  %1266 = insertelement <4 x i32> poison, i32 %1265, i64 0, !dbg !20
  %1267 = lshr i64 %1262, 32, !dbg !20
  %1268 = trunc i64 %1267 to i32, !dbg !20
  %1269 = insertelement <4 x i32> %1266, i32 %1268, i64 1, !dbg !20
  %1270 = insertelement <4 x i32> %1269, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1771 = insertelement <4 x i32> %1270, i32 131072, i64 3, !dbg !20
  %1271 = ptrtoint ptr addrspace(3) %1261 to i32, !dbg !20
  %dma.buffer.lds1772 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1271), !dbg !20
  %1272 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1772, i32 %dma.buffer.offset1770, <4 x i32> %dma.buffer.rsrc1771) #10, !dbg !20
  %1273 = extractvalue [2 x <1 x float>] %64, 0, !dbg !32
  %1274 = extractelement <1 x float> %1273, i64 0, !dbg !32
  %1275 = fmul float %1274, %1238, !dbg !32
  %1276 = extractvalue [2 x <1 x float>] %64, 1, !dbg !32
  %1277 = extractelement <1 x float> %1276, i64 0, !dbg !32
  %1278 = fmul float %1277, %1239, !dbg !32
  %1279 = extractelement <4 x float> %1209, i64 0, !dbg !33
  %1280 = call float @llvm.amdgcn.exp2.f32(float %1279), !dbg !33
  %1281 = insertelement <4 x float> poison, float %1280, i64 0, !dbg !35
  %1282 = extractelement <4 x float> %1209, i64 1, !dbg !33
  %1283 = call float @llvm.amdgcn.exp2.f32(float %1282), !dbg !33
  %1284 = insertelement <4 x float> %1281, float %1283, i64 1, !dbg !35
  %1285 = extractelement <4 x float> %1209, i64 2, !dbg !33
  %1286 = call float @llvm.amdgcn.exp2.f32(float %1285), !dbg !33
  %1287 = insertelement <4 x float> %1284, float %1286, i64 2, !dbg !35
  %1288 = extractelement <4 x float> %1209, i64 3, !dbg !33
  %1289 = call float @llvm.amdgcn.exp2.f32(float %1288), !dbg !33
  %1290 = insertelement <4 x float> %1287, float %1289, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1290) #9, !dbg !35, !frisk.register_lifetimes !1
  %1291 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1291) #9, !dbg !33, !frisk.register_lifetimes !1
  %1292 = extractelement <4 x float> %1210, i64 0, !dbg !33
  %1293 = call float @llvm.amdgcn.exp2.f32(float %1292), !dbg !33
  %1294 = insertelement <4 x float> poison, float %1293, i64 0, !dbg !35
  %local.reduce120 = fadd float %1280, %1293, !dbg !34, !frisk.local_reduction !1
  %1295 = extractelement <4 x float> %1210, i64 1, !dbg !33
  %1296 = call float @llvm.amdgcn.exp2.f32(float %1295), !dbg !33
  %1297 = insertelement <4 x float> %1294, float %1296, i64 1, !dbg !35
  %local.reduce121 = fadd float %1283, %1296, !dbg !34, !frisk.local_reduction !1
  %1298 = extractelement <4 x float> %1210, i64 2, !dbg !33
  %1299 = call float @llvm.amdgcn.exp2.f32(float %1298), !dbg !33
  %1300 = insertelement <4 x float> %1297, float %1299, i64 2, !dbg !35
  %local.reduce122 = fadd float %1286, %1299, !dbg !34, !frisk.local_reduction !1
  %1301 = extractelement <4 x float> %1210, i64 3, !dbg !33
  %1302 = call float @llvm.amdgcn.exp2.f32(float %1301), !dbg !33
  %1303 = insertelement <4 x float> %1300, float %1302, i64 3, !dbg !35
  %math.result228 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1303) #9, !dbg !35, !frisk.register_lifetimes !1
  %1304 = fptrunc <4 x float> %math.result228 to <4 x half>, !dbg !35
  %math.packed229 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1304) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce123 = fadd float %1289, %1302, !dbg !34, !frisk.local_reduction !1
  %1305 = extractelement <4 x float> %1211, i64 0, !dbg !33
  %1306 = call float @llvm.amdgcn.exp2.f32(float %1305), !dbg !33
  %1307 = insertelement <4 x float> poison, float %1306, i64 0, !dbg !35
  %local.reduce124 = fadd float %local.reduce120, %1306, !dbg !34, !frisk.local_reduction !1
  %1308 = extractelement <4 x float> %1211, i64 1, !dbg !33
  %1309 = call float @llvm.amdgcn.exp2.f32(float %1308), !dbg !33
  %1310 = insertelement <4 x float> %1307, float %1309, i64 1, !dbg !35
  %local.reduce125 = fadd float %local.reduce121, %1309, !dbg !34, !frisk.local_reduction !1
  %1311 = extractelement <4 x float> %1211, i64 2, !dbg !33
  %1312 = call float @llvm.amdgcn.exp2.f32(float %1311), !dbg !33
  %1313 = insertelement <4 x float> %1310, float %1312, i64 2, !dbg !35
  %local.reduce126 = fadd float %local.reduce122, %1312, !dbg !34, !frisk.local_reduction !1
  %1314 = extractelement <4 x float> %1211, i64 3, !dbg !33
  %1315 = call float @llvm.amdgcn.exp2.f32(float %1314), !dbg !33
  %1316 = insertelement <4 x float> %1313, float %1315, i64 3, !dbg !35
  %math.result230 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1316) #9, !dbg !35, !frisk.register_lifetimes !1
  %1317 = fptrunc <4 x float> %math.result230 to <4 x half>, !dbg !35
  %math.packed231 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1317) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce127 = fadd float %local.reduce123, %1315, !dbg !34, !frisk.local_reduction !1
  %1318 = extractelement <4 x float> %1212, i64 0, !dbg !33
  %1319 = call float @llvm.amdgcn.exp2.f32(float %1318), !dbg !33
  %1320 = insertelement <4 x float> poison, float %1319, i64 0, !dbg !35
  %local.reduce128 = fadd float %local.reduce124, %1319, !dbg !34, !frisk.local_reduction !1
  %1321 = extractelement <4 x float> %1212, i64 1, !dbg !33
  %1322 = call float @llvm.amdgcn.exp2.f32(float %1321), !dbg !33
  %1323 = insertelement <4 x float> %1320, float %1322, i64 1, !dbg !35
  %local.reduce129 = fadd float %local.reduce125, %1322, !dbg !34, !frisk.local_reduction !1
  %1324 = extractelement <4 x float> %1212, i64 2, !dbg !33
  %1325 = call float @llvm.amdgcn.exp2.f32(float %1324), !dbg !33
  %1326 = insertelement <4 x float> %1323, float %1325, i64 2, !dbg !35
  %local.reduce130 = fadd float %local.reduce126, %1325, !dbg !34, !frisk.local_reduction !1
  %1327 = extractelement <4 x float> %1212, i64 3, !dbg !33
  %1328 = call float @llvm.amdgcn.exp2.f32(float %1327), !dbg !33
  %1329 = insertelement <4 x float> %1326, float %1328, i64 3, !dbg !35
  %math.result232 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1329) #9, !dbg !35, !frisk.register_lifetimes !1
  %1330 = fptrunc <4 x float> %math.result232 to <4 x half>, !dbg !35
  %math.packed233 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1330) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce131 = fadd float %local.reduce127, %1328, !dbg !34, !frisk.local_reduction !1
  %1331 = extractelement <4 x float> %1213, i64 0, !dbg !33
  %1332 = call float @llvm.amdgcn.exp2.f32(float %1331), !dbg !33
  %1333 = insertelement <4 x float> poison, float %1332, i64 0, !dbg !35
  %local.reduce132 = fadd float %local.reduce128, %1332, !dbg !34, !frisk.local_reduction !1
  %1334 = extractelement <4 x float> %1213, i64 1, !dbg !33
  %1335 = call float @llvm.amdgcn.exp2.f32(float %1334), !dbg !33
  %1336 = insertelement <4 x float> %1333, float %1335, i64 1, !dbg !35
  %local.reduce133 = fadd float %local.reduce129, %1335, !dbg !34, !frisk.local_reduction !1
  %1337 = extractelement <4 x float> %1213, i64 2, !dbg !33
  %1338 = call float @llvm.amdgcn.exp2.f32(float %1337), !dbg !33
  %1339 = insertelement <4 x float> %1336, float %1338, i64 2, !dbg !35
  %local.reduce134 = fadd float %local.reduce130, %1338, !dbg !34, !frisk.local_reduction !1
  %1340 = extractelement <4 x float> %1213, i64 3, !dbg !33
  %1341 = call float @llvm.amdgcn.exp2.f32(float %1340), !dbg !33
  %1342 = insertelement <4 x float> %1339, float %1341, i64 3, !dbg !35
  %math.result234 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1342) #9, !dbg !35, !frisk.register_lifetimes !1
  %1343 = fptrunc <4 x float> %math.result234 to <4 x half>, !dbg !35
  %math.packed235 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1343) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce135 = fadd float %local.reduce131, %1341, !dbg !34, !frisk.local_reduction !1
  %1344 = extractelement <4 x float> %1214, i64 0, !dbg !33
  %1345 = call float @llvm.amdgcn.exp2.f32(float %1344), !dbg !33
  %1346 = insertelement <4 x float> poison, float %1345, i64 0, !dbg !35
  %local.reduce136 = fadd float %local.reduce132, %1345, !dbg !34, !frisk.local_reduction !1
  %1347 = extractelement <4 x float> %1214, i64 1, !dbg !33
  %1348 = call float @llvm.amdgcn.exp2.f32(float %1347), !dbg !33
  %1349 = insertelement <4 x float> %1346, float %1348, i64 1, !dbg !35
  %local.reduce137 = fadd float %local.reduce133, %1348, !dbg !34, !frisk.local_reduction !1
  %1350 = extractelement <4 x float> %1214, i64 2, !dbg !33
  %1351 = call float @llvm.amdgcn.exp2.f32(float %1350), !dbg !33
  %1352 = insertelement <4 x float> %1349, float %1351, i64 2, !dbg !35
  %local.reduce138 = fadd float %local.reduce134, %1351, !dbg !34, !frisk.local_reduction !1
  %1353 = extractelement <4 x float> %1214, i64 3, !dbg !33
  %1354 = call float @llvm.amdgcn.exp2.f32(float %1353), !dbg !33
  %1355 = insertelement <4 x float> %1352, float %1354, i64 3, !dbg !35
  %math.result236 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1355) #9, !dbg !35, !frisk.register_lifetimes !1
  %1356 = fptrunc <4 x float> %math.result236 to <4 x half>, !dbg !35
  %math.packed237 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1356) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce139 = fadd float %local.reduce135, %1354, !dbg !34, !frisk.local_reduction !1
  %1357 = extractelement <4 x float> %1215, i64 0, !dbg !33
  %1358 = call float @llvm.amdgcn.exp2.f32(float %1357), !dbg !33
  %1359 = insertelement <4 x float> poison, float %1358, i64 0, !dbg !35
  %local.reduce140 = fadd float %local.reduce136, %1358, !dbg !34, !frisk.local_reduction !1
  %1360 = extractelement <4 x float> %1215, i64 1, !dbg !33
  %1361 = call float @llvm.amdgcn.exp2.f32(float %1360), !dbg !33
  %1362 = insertelement <4 x float> %1359, float %1361, i64 1, !dbg !35
  %local.reduce141 = fadd float %local.reduce137, %1361, !dbg !34, !frisk.local_reduction !1
  %1363 = extractelement <4 x float> %1215, i64 2, !dbg !33
  %1364 = call float @llvm.amdgcn.exp2.f32(float %1363), !dbg !33
  %1365 = insertelement <4 x float> %1362, float %1364, i64 2, !dbg !35
  %local.reduce142 = fadd float %local.reduce138, %1364, !dbg !34, !frisk.local_reduction !1
  %1366 = extractelement <4 x float> %1215, i64 3, !dbg !33
  %1367 = call float @llvm.amdgcn.exp2.f32(float %1366), !dbg !33
  %1368 = insertelement <4 x float> %1365, float %1367, i64 3, !dbg !35
  %math.result238 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1368) #9, !dbg !35, !frisk.register_lifetimes !1
  %1369 = fptrunc <4 x float> %math.result238 to <4 x half>, !dbg !35
  %math.packed239 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1369) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce143 = fadd float %local.reduce139, %1367, !dbg !34, !frisk.local_reduction !1
  %1370 = extractelement <4 x float> %1216, i64 0, !dbg !33
  %1371 = call float @llvm.amdgcn.exp2.f32(float %1370), !dbg !33
  %1372 = insertelement <4 x float> poison, float %1371, i64 0, !dbg !35
  %local.reduce144 = fadd float %local.reduce140, %1371, !dbg !34, !frisk.local_reduction !1
  %1373 = extractelement <4 x float> %1216, i64 1, !dbg !33
  %1374 = call float @llvm.amdgcn.exp2.f32(float %1373), !dbg !33
  %1375 = insertelement <4 x float> %1372, float %1374, i64 1, !dbg !35
  %local.reduce145 = fadd float %local.reduce141, %1374, !dbg !34, !frisk.local_reduction !1
  %local.reduce148 = fadd float %local.reduce144, %local.reduce145, !dbg !34, !frisk.local_reduction !1
  %1376 = extractelement <4 x float> %1216, i64 2, !dbg !33
  %1377 = call float @llvm.amdgcn.exp2.f32(float %1376), !dbg !33
  %1378 = insertelement <4 x float> %1375, float %1377, i64 2, !dbg !35
  %local.reduce146 = fadd float %local.reduce142, %1377, !dbg !34, !frisk.local_reduction !1
  %1379 = extractelement <4 x float> %1216, i64 3, !dbg !33
  %1380 = call float @llvm.amdgcn.exp2.f32(float %1379), !dbg !33
  %1381 = insertelement <4 x float> %1378, float %1380, i64 3, !dbg !35
  %math.result240 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1381) #9, !dbg !35, !frisk.register_lifetimes !1
  %1382 = fptrunc <4 x float> %math.result240 to <4 x half>, !dbg !35
  %math.packed241 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1382) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce147 = fadd float %local.reduce143, %1380, !dbg !34, !frisk.local_reduction !1
  %local.reduce149 = fadd float %local.reduce146, %local.reduce147, !dbg !34, !frisk.local_reduction !1
  %1383 = fadd float %local.reduce148, %local.reduce149, !dbg !34, !frisk.local_reduction !1
  %1384 = extractelement <4 x float> %1218, i64 0, !dbg !33
  %1385 = call float @llvm.amdgcn.exp2.f32(float %1384), !dbg !33
  %1386 = insertelement <4 x float> poison, float %1385, i64 0, !dbg !35
  %1387 = extractelement <4 x float> %1218, i64 1, !dbg !33
  %1388 = call float @llvm.amdgcn.exp2.f32(float %1387), !dbg !33
  %1389 = insertelement <4 x float> %1386, float %1388, i64 1, !dbg !35
  %1390 = extractelement <4 x float> %1218, i64 2, !dbg !33
  %1391 = call float @llvm.amdgcn.exp2.f32(float %1390), !dbg !33
  %1392 = insertelement <4 x float> %1389, float %1391, i64 2, !dbg !35
  %1393 = extractelement <4 x float> %1218, i64 3, !dbg !33
  %1394 = call float @llvm.amdgcn.exp2.f32(float %1393), !dbg !33
  %1395 = insertelement <4 x float> %1392, float %1394, i64 3, !dbg !35
  %math.result242 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1395) #9, !dbg !35, !frisk.register_lifetimes !1
  %1396 = fptrunc <4 x float> %math.result242 to <4 x half>, !dbg !35
  %math.packed243 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1396) #9, !dbg !33, !frisk.register_lifetimes !1
  %1397 = extractelement <4 x float> %1219, i64 0, !dbg !33
  %1398 = call float @llvm.amdgcn.exp2.f32(float %1397), !dbg !33
  %1399 = insertelement <4 x float> poison, float %1398, i64 0, !dbg !35
  %local.reduce = fadd float %1385, %1398, !dbg !34, !frisk.local_reduction !1
  %1400 = extractelement <4 x float> %1219, i64 1, !dbg !33
  %1401 = call float @llvm.amdgcn.exp2.f32(float %1400), !dbg !33
  %1402 = insertelement <4 x float> %1399, float %1401, i64 1, !dbg !35
  %local.reduce90 = fadd float %1388, %1401, !dbg !34, !frisk.local_reduction !1
  %1403 = extractelement <4 x float> %1219, i64 2, !dbg !33
  %1404 = call float @llvm.amdgcn.exp2.f32(float %1403), !dbg !33
  %1405 = insertelement <4 x float> %1402, float %1404, i64 2, !dbg !35
  %local.reduce91 = fadd float %1391, %1404, !dbg !34, !frisk.local_reduction !1
  %1406 = extractelement <4 x float> %1219, i64 3, !dbg !33
  %1407 = call float @llvm.amdgcn.exp2.f32(float %1406), !dbg !33
  %1408 = insertelement <4 x float> %1405, float %1407, i64 3, !dbg !35
  %math.result244 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1408) #9, !dbg !35, !frisk.register_lifetimes !1
  %1409 = fptrunc <4 x float> %math.result244 to <4 x half>, !dbg !35
  %math.packed245 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1409) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce92 = fadd float %1394, %1407, !dbg !34, !frisk.local_reduction !1
  %1410 = extractelement <4 x float> %1220, i64 0, !dbg !33
  %1411 = call float @llvm.amdgcn.exp2.f32(float %1410), !dbg !33
  %1412 = insertelement <4 x float> poison, float %1411, i64 0, !dbg !35
  %local.reduce93 = fadd float %local.reduce, %1411, !dbg !34, !frisk.local_reduction !1
  %1413 = extractelement <4 x float> %1220, i64 1, !dbg !33
  %1414 = call float @llvm.amdgcn.exp2.f32(float %1413), !dbg !33
  %1415 = insertelement <4 x float> %1412, float %1414, i64 1, !dbg !35
  %local.reduce94 = fadd float %local.reduce90, %1414, !dbg !34, !frisk.local_reduction !1
  %1416 = extractelement <4 x float> %1220, i64 2, !dbg !33
  %1417 = call float @llvm.amdgcn.exp2.f32(float %1416), !dbg !33
  %1418 = insertelement <4 x float> %1415, float %1417, i64 2, !dbg !35
  %local.reduce95 = fadd float %local.reduce91, %1417, !dbg !34, !frisk.local_reduction !1
  %1419 = extractelement <4 x float> %1220, i64 3, !dbg !33
  %1420 = call float @llvm.amdgcn.exp2.f32(float %1419), !dbg !33
  %1421 = insertelement <4 x float> %1418, float %1420, i64 3, !dbg !35
  %math.result246 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1421) #9, !dbg !35, !frisk.register_lifetimes !1
  %1422 = fptrunc <4 x float> %math.result246 to <4 x half>, !dbg !35
  %math.packed247 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1422) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce96 = fadd float %local.reduce92, %1420, !dbg !34, !frisk.local_reduction !1
  %1423 = extractelement <4 x float> %1221, i64 0, !dbg !33
  %1424 = call float @llvm.amdgcn.exp2.f32(float %1423), !dbg !33
  %1425 = insertelement <4 x float> poison, float %1424, i64 0, !dbg !35
  %local.reduce97 = fadd float %local.reduce93, %1424, !dbg !34, !frisk.local_reduction !1
  %1426 = extractelement <4 x float> %1221, i64 1, !dbg !33
  %1427 = call float @llvm.amdgcn.exp2.f32(float %1426), !dbg !33
  %1428 = insertelement <4 x float> %1425, float %1427, i64 1, !dbg !35
  %local.reduce98 = fadd float %local.reduce94, %1427, !dbg !34, !frisk.local_reduction !1
  %1429 = extractelement <4 x float> %1221, i64 2, !dbg !33
  %1430 = call float @llvm.amdgcn.exp2.f32(float %1429), !dbg !33
  %1431 = insertelement <4 x float> %1428, float %1430, i64 2, !dbg !35
  %local.reduce99 = fadd float %local.reduce95, %1430, !dbg !34, !frisk.local_reduction !1
  %1432 = extractelement <4 x float> %1221, i64 3, !dbg !33
  %1433 = call float @llvm.amdgcn.exp2.f32(float %1432), !dbg !33
  %1434 = insertelement <4 x float> %1431, float %1433, i64 3, !dbg !35
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1434) #9, !dbg !35, !frisk.register_lifetimes !1
  %1435 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !35
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1435) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce100 = fadd float %local.reduce96, %1433, !dbg !34, !frisk.local_reduction !1
  %1436 = extractelement <4 x float> %1222, i64 0, !dbg !33
  %1437 = call float @llvm.amdgcn.exp2.f32(float %1436), !dbg !33
  %1438 = insertelement <4 x float> poison, float %1437, i64 0, !dbg !35
  %local.reduce101 = fadd float %local.reduce97, %1437, !dbg !34, !frisk.local_reduction !1
  %1439 = extractelement <4 x float> %1222, i64 1, !dbg !33
  %1440 = call float @llvm.amdgcn.exp2.f32(float %1439), !dbg !33
  %1441 = insertelement <4 x float> %1438, float %1440, i64 1, !dbg !35
  %local.reduce102 = fadd float %local.reduce98, %1440, !dbg !34, !frisk.local_reduction !1
  %1442 = extractelement <4 x float> %1222, i64 2, !dbg !33
  %1443 = call float @llvm.amdgcn.exp2.f32(float %1442), !dbg !33
  %1444 = insertelement <4 x float> %1441, float %1443, i64 2, !dbg !35
  %local.reduce103 = fadd float %local.reduce99, %1443, !dbg !34, !frisk.local_reduction !1
  %1445 = extractelement <4 x float> %1222, i64 3, !dbg !33
  %1446 = call float @llvm.amdgcn.exp2.f32(float %1445), !dbg !33
  %1447 = insertelement <4 x float> %1444, float %1446, i64 3, !dbg !35
  %math.result250 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1447) #9, !dbg !35, !frisk.register_lifetimes !1
  %1448 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !35
  %math.packed251 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1448) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce104 = fadd float %local.reduce100, %1446, !dbg !34, !frisk.local_reduction !1
  %1449 = extractelement <4 x float> %1223, i64 0, !dbg !33
  %1450 = call float @llvm.amdgcn.exp2.f32(float %1449), !dbg !33
  %1451 = insertelement <4 x float> poison, float %1450, i64 0, !dbg !35
  %local.reduce105 = fadd float %local.reduce101, %1450, !dbg !34, !frisk.local_reduction !1
  %1452 = extractelement <4 x float> %1223, i64 1, !dbg !33
  %1453 = call float @llvm.amdgcn.exp2.f32(float %1452), !dbg !33
  %1454 = insertelement <4 x float> %1451, float %1453, i64 1, !dbg !35
  %local.reduce106 = fadd float %local.reduce102, %1453, !dbg !34, !frisk.local_reduction !1
  %1455 = extractelement <4 x float> %1223, i64 2, !dbg !33
  %1456 = call float @llvm.amdgcn.exp2.f32(float %1455), !dbg !33
  %1457 = insertelement <4 x float> %1454, float %1456, i64 2, !dbg !35
  %local.reduce107 = fadd float %local.reduce103, %1456, !dbg !34, !frisk.local_reduction !1
  %1458 = extractelement <4 x float> %1223, i64 3, !dbg !33
  %1459 = call float @llvm.amdgcn.exp2.f32(float %1458), !dbg !33
  %1460 = insertelement <4 x float> %1457, float %1459, i64 3, !dbg !35
  %math.result252 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1460) #9, !dbg !35, !frisk.register_lifetimes !1
  %1461 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !35
  %math.packed253 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1461) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce108 = fadd float %local.reduce104, %1459, !dbg !34, !frisk.local_reduction !1
  %1462 = extractelement <4 x float> %1224, i64 0, !dbg !33
  %1463 = call float @llvm.amdgcn.exp2.f32(float %1462), !dbg !33
  %1464 = insertelement <4 x float> poison, float %1463, i64 0, !dbg !35
  %local.reduce109 = fadd float %local.reduce105, %1463, !dbg !34, !frisk.local_reduction !1
  %1465 = extractelement <4 x float> %1224, i64 1, !dbg !33
  %1466 = call float @llvm.amdgcn.exp2.f32(float %1465), !dbg !33
  %1467 = insertelement <4 x float> %1464, float %1466, i64 1, !dbg !35
  %local.reduce110 = fadd float %local.reduce106, %1466, !dbg !34, !frisk.local_reduction !1
  %1468 = extractelement <4 x float> %1224, i64 2, !dbg !33
  %1469 = call float @llvm.amdgcn.exp2.f32(float %1468), !dbg !33
  %1470 = insertelement <4 x float> %1467, float %1469, i64 2, !dbg !35
  %local.reduce111 = fadd float %local.reduce107, %1469, !dbg !34, !frisk.local_reduction !1
  %1471 = extractelement <4 x float> %1224, i64 3, !dbg !33
  %1472 = call float @llvm.amdgcn.exp2.f32(float %1471), !dbg !33
  %1473 = insertelement <4 x float> %1470, float %1472, i64 3, !dbg !35
  %math.result254 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1473) #9, !dbg !35, !frisk.register_lifetimes !1
  %1474 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !35
  %math.packed255 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1474) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce112 = fadd float %local.reduce108, %1472, !dbg !34, !frisk.local_reduction !1
  %1475 = extractelement <4 x float> %1225, i64 0, !dbg !33
  %1476 = call float @llvm.amdgcn.exp2.f32(float %1475), !dbg !33
  %1477 = insertelement <4 x float> poison, float %1476, i64 0, !dbg !35
  %local.reduce113 = fadd float %local.reduce109, %1476, !dbg !34, !frisk.local_reduction !1
  %1478 = extractelement <4 x float> %1225, i64 1, !dbg !33
  %1479 = call float @llvm.amdgcn.exp2.f32(float %1478), !dbg !33
  %1480 = insertelement <4 x float> %1477, float %1479, i64 1, !dbg !35
  %local.reduce114 = fadd float %local.reduce110, %1479, !dbg !34, !frisk.local_reduction !1
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !34, !frisk.local_reduction !1
  %1481 = extractelement <4 x float> %1225, i64 2, !dbg !33
  %1482 = call float @llvm.amdgcn.exp2.f32(float %1481), !dbg !33
  %1483 = insertelement <4 x float> %1480, float %1482, i64 2, !dbg !35
  %local.reduce115 = fadd float %local.reduce111, %1482, !dbg !34, !frisk.local_reduction !1
  %1484 = extractelement <4 x float> %1225, i64 3, !dbg !33
  %1485 = call float @llvm.amdgcn.exp2.f32(float %1484), !dbg !33
  %1486 = insertelement <4 x float> %1483, float %1485, i64 3, !dbg !35
  %math.result256 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1486) #9, !dbg !35, !frisk.register_lifetimes !1
  %1487 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !35
  %math.packed257 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1487) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce116 = fadd float %local.reduce112, %1485, !dbg !34, !frisk.local_reduction !1
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !34, !frisk.local_reduction !1
  %1488 = fadd float %local.reduce117, %local.reduce118, !dbg !34, !frisk.local_reduction !1
  %1489 = bitcast float %1383 to i32, !dbg !34
  %1490 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1489), !dbg !34
  %1491 = bitcast i32 %1490 to float, !dbg !34
  %1492 = fadd float %1383, %1491, !dbg !34
  %1493 = bitcast float %1492 to i32, !dbg !34
  %1494 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1493), !dbg !34
  %1495 = bitcast i32 %1494 to float, !dbg !34
  %1496 = fadd float %1492, %1495, !dbg !34
  %1497 = fadd float %1496, %1275, !dbg !34
  %1498 = bitcast float %1488 to i32, !dbg !34
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1498), !dbg !34
  %1500 = bitcast i32 %1499 to float, !dbg !34
  %1501 = fadd float %1488, %1500, !dbg !34
  %1502 = bitcast float %1501 to i32, !dbg !34
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1502), !dbg !34
  %1504 = bitcast i32 %1503 to float, !dbg !34
  %1505 = fadd float %1501, %1504, !dbg !34
  %1506 = fadd float %1505, %1278, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local836 = shl nuw nsw i32 %address.thread798, 3, !dbg !16
  %address.local837 = and i32 %address.local836, 504, !dbg !16
  %address.local838 = zext nneg i32 %address.local837 to i64, !dbg !16
  %address.local839 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %1507 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local839, i32 0) #0, !dbg !16
  %1508 = bitcast <4 x i32> %1507 to <8 x half>, !dbg !16
  %address.local840 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local840, i64 1024, !dbg !16
  %1509 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #0, !dbg !16
  %1510 = bitcast <4 x i32> %1509 to <8 x half>, !dbg !16
  %address.local842 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local843 = getelementptr i8, ptr addrspace(3) %address.local842, i64 2048, !dbg !16
  %1511 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local843, i32 0) #0, !dbg !16
  %1512 = bitcast <4 x i32> %1511 to <8 x half>, !dbg !16
  %address.local844 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local845 = getelementptr i8, ptr addrspace(3) %address.local844, i64 3072, !dbg !16
  %1513 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local845, i32 0) #0, !dbg !16
  %1514 = bitcast <4 x i32> %1513 to <8 x half>, !dbg !16
  %address.local846 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local847 = getelementptr i8, ptr addrspace(3) %address.local846, i64 4096, !dbg !16
  %1515 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local847, i32 0) #0, !dbg !16
  %1516 = bitcast <4 x i32> %1515 to <8 x half>, !dbg !16
  %address.local848 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local849 = getelementptr i8, ptr addrspace(3) %address.local848, i64 5120, !dbg !16
  %1517 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local849, i32 0) #0, !dbg !16
  %1518 = bitcast <4 x i32> %1517 to <8 x half>, !dbg !16
  %address.local850 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local851 = getelementptr i8, ptr addrspace(3) %address.local850, i64 6144, !dbg !16
  %1519 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #0, !dbg !16
  %1520 = bitcast <4 x i32> %1519 to <8 x half>, !dbg !16
  %address.local852 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !16
  %address.local853 = getelementptr i8, ptr addrspace(3) %address.local852, i64 7168, !dbg !16
  %1521 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local853, i32 0) #0, !dbg !16
  %1522 = bitcast <4 x i32> %1521 to <8 x half>, !dbg !16
  %1523 = shufflevector <8 x half> %1508, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1523, <4 x float> zeroinitializer) #9, !dbg !16
  %1525 = shufflevector <8 x half> %1510, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1525, <4 x float> %1524) #9, !dbg !16
  %1527 = shufflevector <8 x half> %1512, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1527, <4 x float> %1526) #9, !dbg !16
  %1529 = shufflevector <8 x half> %1514, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1529, <4 x float> %1528) #9, !dbg !16
  %1531 = shufflevector <8 x half> %1516, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1531, <4 x float> %1530) #9, !dbg !16
  %1533 = shufflevector <8 x half> %1518, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1533, <4 x float> %1532) #9, !dbg !16
  %1535 = shufflevector <8 x half> %1520, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1535, <4 x float> %1534) #9, !dbg !16
  %1537 = shufflevector <8 x half> %1522, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1537, <4 x float> %1536) #9, !dbg !16
  %1539 = extractelement <4 x float> %1538, i64 0, !dbg !16
  %1540 = extractelement <4 x float> %1538, i64 1, !dbg !16
  %1541 = extractelement <4 x float> %1538, i64 2, !dbg !16
  %1542 = extractelement <4 x float> %1538, i64 3, !dbg !16
  %1543 = select i1 %.not, float %1540, float %1539, !dbg !16
  %1544 = bitcast float %1543 to i32, !dbg !16
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1544), !dbg !16
  %1546 = bitcast i32 %1545 to float, !dbg !16
  %1547 = select i1 %.not, float %1539, float %1546, !dbg !16
  %1548 = select i1 %.not, float %1546, float %1540, !dbg !16
  %1549 = select i1 %.not, float %1542, float %1541, !dbg !16
  %1550 = bitcast float %1549 to i32, !dbg !16
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1550), !dbg !16
  %1552 = bitcast i32 %1551 to float, !dbg !16
  %1553 = select i1 %.not, float %1541, float %1552, !dbg !16
  %1554 = select i1 %.not, float %1552, float %1542, !dbg !16
  %1555 = select i1 %.not1, float %1553, float %1547, !dbg !16
  %1556 = bitcast float %1555 to i32, !dbg !16
  %1557 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1556), !dbg !16
  %1558 = bitcast i32 %1557 to float, !dbg !16
  %1559 = select i1 %.not1, float %1547, float %1558, !dbg !16
  %1560 = select i1 %.not1, float %1558, float %1553, !dbg !16
  %1561 = select i1 %.not1, float %1554, float %1548, !dbg !16
  %1562 = bitcast float %1561 to i32, !dbg !16
  %1563 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1562), !dbg !16
  %1564 = bitcast i32 %1563 to float, !dbg !16
  %1565 = select i1 %.not1, float %1548, float %1564, !dbg !16
  %1566 = select i1 %.not1, float %1564, float %1554, !dbg !16
  %1567 = shufflevector <8 x half> %1508, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1567, <4 x float> zeroinitializer) #9, !dbg !16
  %1569 = shufflevector <8 x half> %1510, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1569, <4 x float> %1568) #9, !dbg !16
  %1571 = shufflevector <8 x half> %1512, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1571, <4 x float> %1570) #9, !dbg !16
  %1573 = shufflevector <8 x half> %1514, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1573, <4 x float> %1572) #9, !dbg !16
  %1575 = shufflevector <8 x half> %1516, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1575, <4 x float> %1574) #9, !dbg !16
  %1577 = shufflevector <8 x half> %1518, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1577, <4 x float> %1576) #9, !dbg !16
  %1579 = shufflevector <8 x half> %1520, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1579, <4 x float> %1578) #9, !dbg !16
  %1581 = shufflevector <8 x half> %1522, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1581, <4 x float> %1580) #9, !dbg !16
  %1583 = extractelement <4 x float> %1582, i64 0, !dbg !16
  %1584 = extractelement <4 x float> %1582, i64 1, !dbg !16
  %1585 = extractelement <4 x float> %1582, i64 2, !dbg !16
  %1586 = extractelement <4 x float> %1582, i64 3, !dbg !16
  %1587 = select i1 %.not, float %1584, float %1583, !dbg !16
  %1588 = bitcast float %1587 to i32, !dbg !16
  %1589 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1588), !dbg !16
  %1590 = bitcast i32 %1589 to float, !dbg !16
  %1591 = select i1 %.not, float %1583, float %1590, !dbg !16
  %1592 = select i1 %.not, float %1590, float %1584, !dbg !16
  %1593 = select i1 %.not, float %1586, float %1585, !dbg !16
  %1594 = bitcast float %1593 to i32, !dbg !16
  %1595 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1594), !dbg !16
  %1596 = bitcast i32 %1595 to float, !dbg !16
  %1597 = select i1 %.not, float %1585, float %1596, !dbg !16
  %1598 = select i1 %.not, float %1596, float %1586, !dbg !16
  %1599 = select i1 %.not1, float %1597, float %1591, !dbg !16
  %1600 = bitcast float %1599 to i32, !dbg !16
  %1601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1600), !dbg !16
  %1602 = bitcast i32 %1601 to float, !dbg !16
  %1603 = select i1 %.not1, float %1591, float %1602, !dbg !16
  %1604 = select i1 %.not1, float %1602, float %1597, !dbg !16
  %1605 = select i1 %.not1, float %1598, float %1592, !dbg !16
  %1606 = bitcast float %1605 to i32, !dbg !16
  %1607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1606), !dbg !16
  %1608 = bitcast i32 %1607 to float, !dbg !16
  %1609 = select i1 %.not1, float %1592, float %1608, !dbg !16
  %1610 = select i1 %.not1, float %1608, float %1598, !dbg !16
  %1611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1523, <4 x float> zeroinitializer) #9, !dbg !16
  %1612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1525, <4 x float> %1611) #9, !dbg !16
  %1613 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1527, <4 x float> %1612) #9, !dbg !16
  %1614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1529, <4 x float> %1613) #9, !dbg !16
  %1615 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1531, <4 x float> %1614) #9, !dbg !16
  %1616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1533, <4 x float> %1615) #9, !dbg !16
  %1617 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1535, <4 x float> %1616) #9, !dbg !16
  %1618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1537, <4 x float> %1617) #9, !dbg !16
  %1619 = extractelement <4 x float> %1618, i64 0, !dbg !16
  %1620 = extractelement <4 x float> %1618, i64 1, !dbg !16
  %1621 = extractelement <4 x float> %1618, i64 2, !dbg !16
  %1622 = extractelement <4 x float> %1618, i64 3, !dbg !16
  %1623 = select i1 %.not, float %1620, float %1619, !dbg !16
  %1624 = bitcast float %1623 to i32, !dbg !16
  %1625 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1624), !dbg !16
  %1626 = bitcast i32 %1625 to float, !dbg !16
  %1627 = select i1 %.not, float %1619, float %1626, !dbg !16
  %1628 = select i1 %.not, float %1626, float %1620, !dbg !16
  %1629 = select i1 %.not, float %1622, float %1621, !dbg !16
  %1630 = bitcast float %1629 to i32, !dbg !16
  %1631 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1630), !dbg !16
  %1632 = bitcast i32 %1631 to float, !dbg !16
  %1633 = select i1 %.not, float %1621, float %1632, !dbg !16
  %1634 = select i1 %.not, float %1632, float %1622, !dbg !16
  %1635 = select i1 %.not1, float %1633, float %1627, !dbg !16
  %1636 = bitcast float %1635 to i32, !dbg !16
  %1637 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1636), !dbg !16
  %1638 = bitcast i32 %1637 to float, !dbg !16
  %1639 = select i1 %.not1, float %1627, float %1638, !dbg !16
  %1640 = select i1 %.not1, float %1638, float %1633, !dbg !16
  %1641 = select i1 %.not1, float %1634, float %1628, !dbg !16
  %1642 = bitcast float %1641 to i32, !dbg !16
  %1643 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1642), !dbg !16
  %1644 = bitcast i32 %1643 to float, !dbg !16
  %1645 = select i1 %.not1, float %1628, float %1644, !dbg !16
  %1646 = select i1 %.not1, float %1644, float %1634, !dbg !16
  %1647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1567, <4 x float> zeroinitializer) #9, !dbg !16
  %1648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1569, <4 x float> %1647) #9, !dbg !16
  %1649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1571, <4 x float> %1648) #9, !dbg !16
  %1650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1573, <4 x float> %1649) #9, !dbg !16
  %1651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1575, <4 x float> %1650) #9, !dbg !16
  %1652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1577, <4 x float> %1651) #9, !dbg !16
  %1653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1579, <4 x float> %1652) #9, !dbg !16
  %1654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1581, <4 x float> %1653) #9, !dbg !16
  %1655 = extractelement <4 x float> %1654, i64 0, !dbg !16
  %1656 = extractelement <4 x float> %1654, i64 1, !dbg !16
  %1657 = extractelement <4 x float> %1654, i64 2, !dbg !16
  %1658 = extractelement <4 x float> %1654, i64 3, !dbg !16
  %1659 = select i1 %.not, float %1656, float %1655, !dbg !16
  %1660 = bitcast float %1659 to i32, !dbg !16
  %1661 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1660), !dbg !16
  %1662 = bitcast i32 %1661 to float, !dbg !16
  %1663 = select i1 %.not, float %1655, float %1662, !dbg !16
  %1664 = select i1 %.not, float %1662, float %1656, !dbg !16
  %1665 = select i1 %.not, float %1658, float %1657, !dbg !16
  %1666 = bitcast float %1665 to i32, !dbg !16
  %1667 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1666), !dbg !16
  %1668 = bitcast i32 %1667 to float, !dbg !16
  %1669 = select i1 %.not, float %1657, float %1668, !dbg !16
  %1670 = select i1 %.not, float %1668, float %1658, !dbg !16
  %1671 = select i1 %.not1, float %1669, float %1663, !dbg !16
  %1672 = bitcast float %1671 to i32, !dbg !16
  %1673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1672), !dbg !16
  %1674 = bitcast i32 %1673 to float, !dbg !16
  %1675 = select i1 %.not1, float %1663, float %1674, !dbg !16
  %1676 = select i1 %.not1, float %1674, float %1669, !dbg !16
  %1677 = select i1 %.not1, float %1670, float %1664, !dbg !16
  %1678 = bitcast float %1677 to i32, !dbg !16
  %1679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1678), !dbg !16
  %1680 = bitcast i32 %1679 to float, !dbg !16
  %1681 = select i1 %.not1, float %1664, float %1680, !dbg !16
  %1682 = select i1 %.not1, float %1680, float %1670, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %1683, label %1684, !dbg !36

1683:                                             ; preds = %1237
  %address.thread854 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local855 = lshr i32 %address.thread854, 1, !dbg !15
  %address.local856 = and i32 %address.local855, 96, !dbg !15
  %address.local857 = and i32 %address.thread854, 15, !dbg !15
  %address.local858 = or disjoint i32 %address.local856, %address.local857, !dbg !15
  %address.local859 = zext nneg i32 %address.local858 to i64, !dbg !15
  %address.local860 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !15
  store float %1234, ptr addrspace(3) %address.local860, align 4, !dbg !36
  %address.local861 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !15
  %address.local862 = getelementptr i8, ptr addrspace(3) %address.local861, i64 64, !dbg !15
  store float %1235, ptr addrspace(3) %address.local862, align 4, !dbg !36
  br label %1684, !dbg !36

1684:                                             ; preds = %1683, %1237
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread863 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %lds.const.gep1747 = getelementptr i8, ptr addrspace(3) %address.local804, i64 0, !dbg !36
  %1685 = load float, ptr addrspace(3) %lds.const.gep1747, align 4, !dbg !36
  %lds.const.gep1748 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %1686 = load float, ptr addrspace(3) %lds.const.gep1748, align 4, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %1687 = add nuw nsw i64 %102, %15, !dbg !20
  %1688 = shl i64 %1687, 19, !dbg !20
  %1689 = and i64 %1688, 16252928, !dbg !20
  %1690 = add i64 %1689, %108, !dbg !20
  %1691 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1690, !dbg !20
  %1692 = getelementptr i8, ptr addrspace(1) %1691, i64 64, !dbg !20
  %address.local877 = zext nneg i32 %address.thread863 to i64, !dbg !12
  %address.local878 = shl nuw nsw i64 %address.local877, 3, !dbg !11
  %address.local879 = and i64 %address.local878, 1536, !dbg !11
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local879, !dbg !11
  %address.local881 = ptrtoint ptr addrspace(3) %address.local880 to i64, !dbg !11
  %address.local882 = trunc i64 %address.local881 to i32, !dbg !11
  %1693 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local882), !dbg !20
  %1694 = zext i32 %1693 to i64, !dbg !20
  %1695 = inttoptr i64 %1694 to ptr addrspace(3), !dbg !20
  %address.local883 = shl nuw nsw i64 %address.local877, 5, !dbg !20
  %address.local884 = and i64 %address.local883, 8064, !dbg !20
  %address.local885 = shl nuw nsw i64 %address.local877, 3, !dbg !20
  %address.local886 = and i64 %address.local885, 24, !dbg !20
  %address.local887 = or disjoint i64 %address.local884, %address.local886, !dbg !20
  %address.local888 = getelementptr [2 x i8], ptr addrspace(1) %1692, i64 %address.local887, !dbg !20
  %1696 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1697 = ptrtoint ptr addrspace(1) %address.local888 to i64, !dbg !20
  %1698 = sub i64 %1697, %1696, !dbg !20
  %dma.buffer.offset1767 = trunc i64 %1698 to i32, !dbg !20
  %1699 = trunc i64 %1696 to i32, !dbg !20
  %1700 = insertelement <4 x i32> poison, i32 %1699, i64 0, !dbg !20
  %1701 = lshr i64 %1696, 32, !dbg !20
  %1702 = trunc i64 %1701 to i32, !dbg !20
  %1703 = insertelement <4 x i32> %1700, i32 %1702, i64 1, !dbg !20
  %1704 = insertelement <4 x i32> %1703, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1768 = insertelement <4 x i32> %1704, i32 131072, i64 3, !dbg !20
  %1705 = ptrtoint ptr addrspace(3) %1695 to i32, !dbg !20
  %dma.buffer.lds1769 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1705), !dbg !20
  %1706 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1769, i32 %dma.buffer.offset1767, <4 x i32> %dma.buffer.rsrc1768) #10, !dbg !20
  %address.local889 = or disjoint i64 %address.local877, 256, !dbg !11
  %address.local890 = shl nuw nsw i64 %address.local889, 3, !dbg !11
  %address.local891 = and i64 %address.local890, 3584, !dbg !11
  %address.local892 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local891, !dbg !11
  %address.local893 = ptrtoint ptr addrspace(3) %address.local892 to i64, !dbg !11
  %address.local894 = trunc i64 %address.local893 to i32, !dbg !11
  %1707 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !20
  %1708 = zext i32 %1707 to i64, !dbg !20
  %1709 = inttoptr i64 %1708 to ptr addrspace(3), !dbg !20
  %address.local895 = shl nuw nsw i64 %address.local889, 5, !dbg !20
  %address.local896 = and i64 %address.local895, 16256, !dbg !20
  %address.local897 = shl nuw nsw i64 %address.local877, 3, !dbg !20
  %address.local898 = and i64 %address.local897, 24, !dbg !20
  %address.local899 = or disjoint i64 %address.local896, %address.local898, !dbg !20
  %address.local900 = getelementptr [2 x i8], ptr addrspace(1) %1692, i64 %address.local899, !dbg !20
  %1710 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1711 = ptrtoint ptr addrspace(1) %address.local900 to i64, !dbg !20
  %1712 = sub i64 %1711, %1710, !dbg !20
  %dma.buffer.offset1764 = trunc i64 %1712 to i32, !dbg !20
  %1713 = trunc i64 %1710 to i32, !dbg !20
  %1714 = insertelement <4 x i32> poison, i32 %1713, i64 0, !dbg !20
  %1715 = lshr i64 %1710, 32, !dbg !20
  %1716 = trunc i64 %1715 to i32, !dbg !20
  %1717 = insertelement <4 x i32> %1714, i32 %1716, i64 1, !dbg !20
  %1718 = insertelement <4 x i32> %1717, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1765 = insertelement <4 x i32> %1718, i32 131072, i64 3, !dbg !20
  %1719 = ptrtoint ptr addrspace(3) %1709 to i32, !dbg !20
  %dma.buffer.lds1766 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1719), !dbg !20
  %1720 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1766, i32 %dma.buffer.offset1764, <4 x i32> %dma.buffer.rsrc1765) #10, !dbg !20
  %1721 = extractvalue [2 x <8 x float>] %63, 0, !dbg !36
  %1722 = shufflevector <8 x float> %1721, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1723 = insertelement <4 x float> poison, float %1685, i64 0, !dbg !36
  %1724 = shufflevector <4 x float> %1723, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1725 = fmul <4 x float> %1722, %1724, !dbg !36
  %1726 = insertelement <4 x float> poison, float %1559, i64 0, !dbg !37
  %1727 = insertelement <4 x float> %1726, float %1565, i64 1, !dbg !37
  %1728 = insertelement <4 x float> %1727, float %1560, i64 2, !dbg !37
  %1729 = insertelement <4 x float> %1728, float %1566, i64 3, !dbg !37
  %1730 = fadd <4 x float> %1725, %1729, !dbg !37
  %1731 = shufflevector <8 x float> %1721, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1732 = fmul <4 x float> %1731, %1724, !dbg !36
  %1733 = insertelement <4 x float> poison, float %1603, i64 0, !dbg !37
  %1734 = insertelement <4 x float> %1733, float %1609, i64 1, !dbg !37
  %1735 = insertelement <4 x float> %1734, float %1604, i64 2, !dbg !37
  %1736 = insertelement <4 x float> %1735, float %1610, i64 3, !dbg !37
  %1737 = fadd <4 x float> %1732, %1736, !dbg !37
  %1738 = extractvalue [2 x <8 x float>] %63, 1, !dbg !36
  %1739 = shufflevector <8 x float> %1738, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1740 = insertelement <4 x float> poison, float %1686, i64 0, !dbg !36
  %1741 = shufflevector <4 x float> %1740, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1742 = fmul <4 x float> %1739, %1741, !dbg !36
  %1743 = insertelement <4 x float> poison, float %1639, i64 0, !dbg !37
  %1744 = insertelement <4 x float> %1743, float %1645, i64 1, !dbg !37
  %1745 = insertelement <4 x float> %1744, float %1640, i64 2, !dbg !37
  %1746 = insertelement <4 x float> %1745, float %1646, i64 3, !dbg !37
  %1747 = fadd <4 x float> %1742, %1746, !dbg !37
  %1748 = shufflevector <8 x float> %1738, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1749 = fmul <4 x float> %1748, %1741, !dbg !36
  %1750 = insertelement <4 x float> poison, float %1675, i64 0, !dbg !37
  %1751 = insertelement <4 x float> %1750, float %1681, i64 1, !dbg !37
  %1752 = insertelement <4 x float> %1751, float %1676, i64 2, !dbg !37
  %1753 = insertelement <4 x float> %1752, float %1682, i64 3, !dbg !37
  %1754 = fadd <4 x float> %1749, %1753, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local901 = shl nuw nsw i32 %address.thread863, 3, !dbg !16
  %address.local902 = and i32 %address.local901, 504, !dbg !16
  %address.local903 = zext nneg i32 %address.local902 to i64, !dbg !16
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %1755 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #0, !dbg !16
  %1756 = bitcast <4 x i32> %1755 to <8 x half>, !dbg !16
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 1024, !dbg !16
  %1757 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !16
  %1758 = bitcast <4 x i32> %1757 to <8 x half>, !dbg !16
  %address.local907 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local908 = getelementptr i8, ptr addrspace(3) %address.local907, i64 2048, !dbg !16
  %1759 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local908, i32 0) #0, !dbg !16
  %1760 = bitcast <4 x i32> %1759 to <8 x half>, !dbg !16
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local910 = getelementptr i8, ptr addrspace(3) %address.local909, i64 3072, !dbg !16
  %1761 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local910, i32 0) #0, !dbg !16
  %1762 = bitcast <4 x i32> %1761 to <8 x half>, !dbg !16
  %address.local911 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local912 = getelementptr i8, ptr addrspace(3) %address.local911, i64 4096, !dbg !16
  %1763 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local912, i32 0) #0, !dbg !16
  %1764 = bitcast <4 x i32> %1763 to <8 x half>, !dbg !16
  %address.local913 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local914 = getelementptr i8, ptr addrspace(3) %address.local913, i64 5120, !dbg !16
  %1765 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local914, i32 0) #0, !dbg !16
  %1766 = bitcast <4 x i32> %1765 to <8 x half>, !dbg !16
  %address.local915 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local916 = getelementptr i8, ptr addrspace(3) %address.local915, i64 6144, !dbg !16
  %1767 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local916, i32 0) #0, !dbg !16
  %1768 = bitcast <4 x i32> %1767 to <8 x half>, !dbg !16
  %address.local917 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !16
  %address.local918 = getelementptr i8, ptr addrspace(3) %address.local917, i64 7168, !dbg !16
  %1769 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local918, i32 0) #0, !dbg !16
  %1770 = bitcast <4 x i32> %1769 to <8 x half>, !dbg !16
  %1771 = shufflevector <8 x half> %1756, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1772 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1771, <4 x float> zeroinitializer) #9, !dbg !16
  %1773 = shufflevector <8 x half> %1758, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1774 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1773, <4 x float> %1772) #9, !dbg !16
  %1775 = shufflevector <8 x half> %1760, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1775, <4 x float> %1774) #9, !dbg !16
  %1777 = shufflevector <8 x half> %1762, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1778 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1777, <4 x float> %1776) #9, !dbg !16
  %1779 = shufflevector <8 x half> %1764, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1780 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1779, <4 x float> %1778) #9, !dbg !16
  %1781 = shufflevector <8 x half> %1766, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1781, <4 x float> %1780) #9, !dbg !16
  %1783 = shufflevector <8 x half> %1768, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1784 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1783, <4 x float> %1782) #9, !dbg !16
  %1785 = shufflevector <8 x half> %1770, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1786 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1785, <4 x float> %1784) #9, !dbg !16
  %1787 = extractelement <4 x float> %1786, i64 0, !dbg !16
  %1788 = extractelement <4 x float> %1786, i64 1, !dbg !16
  %1789 = extractelement <4 x float> %1786, i64 2, !dbg !16
  %1790 = extractelement <4 x float> %1786, i64 3, !dbg !16
  %1791 = select i1 %.not, float %1788, float %1787, !dbg !16
  %1792 = bitcast float %1791 to i32, !dbg !16
  %1793 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1792), !dbg !16
  %1794 = bitcast i32 %1793 to float, !dbg !16
  %1795 = select i1 %.not, float %1787, float %1794, !dbg !16
  %1796 = select i1 %.not, float %1794, float %1788, !dbg !16
  %1797 = select i1 %.not, float %1790, float %1789, !dbg !16
  %1798 = bitcast float %1797 to i32, !dbg !16
  %1799 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1798), !dbg !16
  %1800 = bitcast i32 %1799 to float, !dbg !16
  %1801 = select i1 %.not, float %1789, float %1800, !dbg !16
  %1802 = select i1 %.not, float %1800, float %1790, !dbg !16
  %1803 = select i1 %.not1, float %1801, float %1795, !dbg !16
  %1804 = bitcast float %1803 to i32, !dbg !16
  %1805 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1804), !dbg !16
  %1806 = bitcast i32 %1805 to float, !dbg !16
  %1807 = select i1 %.not1, float %1795, float %1806, !dbg !16
  %1808 = select i1 %.not1, float %1806, float %1801, !dbg !16
  %1809 = select i1 %.not1, float %1802, float %1796, !dbg !16
  %1810 = bitcast float %1809 to i32, !dbg !16
  %1811 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1810), !dbg !16
  %1812 = bitcast i32 %1811 to float, !dbg !16
  %1813 = select i1 %.not1, float %1796, float %1812, !dbg !16
  %1814 = select i1 %.not1, float %1812, float %1802, !dbg !16
  %1815 = shufflevector <8 x half> %1756, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1815, <4 x float> zeroinitializer) #9, !dbg !16
  %1817 = shufflevector <8 x half> %1758, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1817, <4 x float> %1816) #9, !dbg !16
  %1819 = shufflevector <8 x half> %1760, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1819, <4 x float> %1818) #9, !dbg !16
  %1821 = shufflevector <8 x half> %1762, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1821, <4 x float> %1820) #9, !dbg !16
  %1823 = shufflevector <8 x half> %1764, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1823, <4 x float> %1822) #9, !dbg !16
  %1825 = shufflevector <8 x half> %1766, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1825, <4 x float> %1824) #9, !dbg !16
  %1827 = shufflevector <8 x half> %1768, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1827, <4 x float> %1826) #9, !dbg !16
  %1829 = shufflevector <8 x half> %1770, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1830 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1829, <4 x float> %1828) #9, !dbg !16
  %1831 = extractelement <4 x float> %1830, i64 0, !dbg !16
  %1832 = extractelement <4 x float> %1830, i64 1, !dbg !16
  %1833 = extractelement <4 x float> %1830, i64 2, !dbg !16
  %1834 = extractelement <4 x float> %1830, i64 3, !dbg !16
  %1835 = select i1 %.not, float %1832, float %1831, !dbg !16
  %1836 = bitcast float %1835 to i32, !dbg !16
  %1837 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1836), !dbg !16
  %1838 = bitcast i32 %1837 to float, !dbg !16
  %1839 = select i1 %.not, float %1831, float %1838, !dbg !16
  %1840 = select i1 %.not, float %1838, float %1832, !dbg !16
  %1841 = select i1 %.not, float %1834, float %1833, !dbg !16
  %1842 = bitcast float %1841 to i32, !dbg !16
  %1843 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1842), !dbg !16
  %1844 = bitcast i32 %1843 to float, !dbg !16
  %1845 = select i1 %.not, float %1833, float %1844, !dbg !16
  %1846 = select i1 %.not, float %1844, float %1834, !dbg !16
  %1847 = select i1 %.not1, float %1845, float %1839, !dbg !16
  %1848 = bitcast float %1847 to i32, !dbg !16
  %1849 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1848), !dbg !16
  %1850 = bitcast i32 %1849 to float, !dbg !16
  %1851 = select i1 %.not1, float %1839, float %1850, !dbg !16
  %1852 = select i1 %.not1, float %1850, float %1845, !dbg !16
  %1853 = select i1 %.not1, float %1846, float %1840, !dbg !16
  %1854 = bitcast float %1853 to i32, !dbg !16
  %1855 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1854), !dbg !16
  %1856 = bitcast i32 %1855 to float, !dbg !16
  %1857 = select i1 %.not1, float %1840, float %1856, !dbg !16
  %1858 = select i1 %.not1, float %1856, float %1846, !dbg !16
  %1859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1771, <4 x float> zeroinitializer) #9, !dbg !16
  %1860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1773, <4 x float> %1859) #9, !dbg !16
  %1861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1775, <4 x float> %1860) #9, !dbg !16
  %1862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1777, <4 x float> %1861) #9, !dbg !16
  %1863 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1779, <4 x float> %1862) #9, !dbg !16
  %1864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1781, <4 x float> %1863) #9, !dbg !16
  %1865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1783, <4 x float> %1864) #9, !dbg !16
  %1866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1785, <4 x float> %1865) #9, !dbg !16
  %1867 = extractelement <4 x float> %1866, i64 0, !dbg !16
  %1868 = extractelement <4 x float> %1866, i64 1, !dbg !16
  %1869 = extractelement <4 x float> %1866, i64 2, !dbg !16
  %1870 = extractelement <4 x float> %1866, i64 3, !dbg !16
  %1871 = select i1 %.not, float %1868, float %1867, !dbg !16
  %1872 = bitcast float %1871 to i32, !dbg !16
  %1873 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1872), !dbg !16
  %1874 = bitcast i32 %1873 to float, !dbg !16
  %1875 = select i1 %.not, float %1867, float %1874, !dbg !16
  %1876 = select i1 %.not, float %1874, float %1868, !dbg !16
  %1877 = select i1 %.not, float %1870, float %1869, !dbg !16
  %1878 = bitcast float %1877 to i32, !dbg !16
  %1879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1878), !dbg !16
  %1880 = bitcast i32 %1879 to float, !dbg !16
  %1881 = select i1 %.not, float %1869, float %1880, !dbg !16
  %1882 = select i1 %.not, float %1880, float %1870, !dbg !16
  %1883 = select i1 %.not1, float %1881, float %1875, !dbg !16
  %1884 = bitcast float %1883 to i32, !dbg !16
  %1885 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1884), !dbg !16
  %1886 = bitcast i32 %1885 to float, !dbg !16
  %1887 = select i1 %.not1, float %1875, float %1886, !dbg !16
  %1888 = select i1 %.not1, float %1886, float %1881, !dbg !16
  %1889 = select i1 %.not1, float %1882, float %1876, !dbg !16
  %1890 = bitcast float %1889 to i32, !dbg !16
  %1891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1890), !dbg !16
  %1892 = bitcast i32 %1891 to float, !dbg !16
  %1893 = select i1 %.not1, float %1876, float %1892, !dbg !16
  %1894 = select i1 %.not1, float %1892, float %1882, !dbg !16
  %1895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1815, <4 x float> zeroinitializer) #9, !dbg !16
  %1896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1817, <4 x float> %1895) #9, !dbg !16
  %1897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1819, <4 x float> %1896) #9, !dbg !16
  %1898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1821, <4 x float> %1897) #9, !dbg !16
  %1899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1823, <4 x float> %1898) #9, !dbg !16
  %1900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1825, <4 x float> %1899) #9, !dbg !16
  %1901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1827, <4 x float> %1900) #9, !dbg !16
  %1902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1829, <4 x float> %1901) #9, !dbg !16
  %1903 = extractelement <4 x float> %1902, i64 0, !dbg !16
  %1904 = extractelement <4 x float> %1902, i64 1, !dbg !16
  %1905 = extractelement <4 x float> %1902, i64 2, !dbg !16
  %1906 = extractelement <4 x float> %1902, i64 3, !dbg !16
  %1907 = select i1 %.not, float %1904, float %1903, !dbg !16
  %1908 = bitcast float %1907 to i32, !dbg !16
  %1909 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1908), !dbg !16
  %1910 = bitcast i32 %1909 to float, !dbg !16
  %1911 = select i1 %.not, float %1903, float %1910, !dbg !16
  %1912 = select i1 %.not, float %1910, float %1904, !dbg !16
  %1913 = select i1 %.not, float %1906, float %1905, !dbg !16
  %1914 = bitcast float %1913 to i32, !dbg !16
  %1915 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1914), !dbg !16
  %1916 = bitcast i32 %1915 to float, !dbg !16
  %1917 = select i1 %.not, float %1905, float %1916, !dbg !16
  %1918 = select i1 %.not, float %1916, float %1906, !dbg !16
  %1919 = select i1 %.not1, float %1917, float %1911, !dbg !16
  %1920 = bitcast float %1919 to i32, !dbg !16
  %1921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1920), !dbg !16
  %1922 = bitcast i32 %1921 to float, !dbg !16
  %1923 = select i1 %.not1, float %1911, float %1922, !dbg !16
  %1924 = select i1 %.not1, float %1922, float %1917, !dbg !16
  %1925 = select i1 %.not1, float %1918, float %1912, !dbg !16
  %1926 = bitcast float %1925 to i32, !dbg !16
  %1927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1926), !dbg !16
  %1928 = bitcast i32 %1927 to float, !dbg !16
  %1929 = select i1 %.not1, float %1912, float %1928, !dbg !16
  %1930 = select i1 %.not1, float %1928, float %1918, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %1931, label %1932, !dbg !36

1931:                                             ; preds = %1684
  %address.thread919 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local920 = lshr i32 %address.thread919, 1, !dbg !15
  %address.local921 = and i32 %address.local920, 96, !dbg !15
  %address.local922 = and i32 %address.thread919, 15, !dbg !15
  %address.local923 = or disjoint i32 %address.local921, %address.local922, !dbg !15
  %address.local924 = zext nneg i32 %address.local923 to i64, !dbg !15
  %address.local925 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !15
  store float %1234, ptr addrspace(3) %address.local925, align 4, !dbg !36
  %address.local926 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !15
  %address.local927 = getelementptr i8, ptr addrspace(3) %address.local926, i64 64, !dbg !15
  store float %1235, ptr addrspace(3) %address.local927, align 4, !dbg !36
  br label %1932, !dbg !36

1932:                                             ; preds = %1931, %1684
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread928 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %lds.const.gep1749 = getelementptr i8, ptr addrspace(3) %address.local804, i64 0, !dbg !36
  %1933 = load float, ptr addrspace(3) %lds.const.gep1749, align 4, !dbg !36
  %lds.const.gep1750 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %1934 = load float, ptr addrspace(3) %lds.const.gep1750, align 4, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %1935 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1690, !dbg !20
  %1936 = getelementptr i8, ptr addrspace(1) %1935, i64 128, !dbg !20
  %address.local942 = zext nneg i32 %address.thread928 to i64, !dbg !12
  %address.local943 = shl nuw nsw i64 %address.local942, 3, !dbg !11
  %address.local944 = and i64 %address.local943, 1536, !dbg !11
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local944, !dbg !11
  %address.local946 = ptrtoint ptr addrspace(3) %address.local945 to i64, !dbg !11
  %address.local947 = trunc i64 %address.local946 to i32, !dbg !11
  %1937 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local947), !dbg !20
  %1938 = zext i32 %1937 to i64, !dbg !20
  %1939 = inttoptr i64 %1938 to ptr addrspace(3), !dbg !20
  %address.local948 = shl nuw nsw i64 %address.local942, 5, !dbg !20
  %address.local949 = and i64 %address.local948, 8064, !dbg !20
  %address.local950 = shl nuw nsw i64 %address.local942, 3, !dbg !20
  %address.local951 = and i64 %address.local950, 24, !dbg !20
  %address.local952 = or disjoint i64 %address.local949, %address.local951, !dbg !20
  %address.local953 = getelementptr [2 x i8], ptr addrspace(1) %1936, i64 %address.local952, !dbg !20
  %1940 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1941 = ptrtoint ptr addrspace(1) %address.local953 to i64, !dbg !20
  %1942 = sub i64 %1941, %1940, !dbg !20
  %dma.buffer.offset1761 = trunc i64 %1942 to i32, !dbg !20
  %1943 = trunc i64 %1940 to i32, !dbg !20
  %1944 = insertelement <4 x i32> poison, i32 %1943, i64 0, !dbg !20
  %1945 = lshr i64 %1940, 32, !dbg !20
  %1946 = trunc i64 %1945 to i32, !dbg !20
  %1947 = insertelement <4 x i32> %1944, i32 %1946, i64 1, !dbg !20
  %1948 = insertelement <4 x i32> %1947, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1762 = insertelement <4 x i32> %1948, i32 131072, i64 3, !dbg !20
  %1949 = ptrtoint ptr addrspace(3) %1939 to i32, !dbg !20
  %dma.buffer.lds1763 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1949), !dbg !20
  %1950 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1763, i32 %dma.buffer.offset1761, <4 x i32> %dma.buffer.rsrc1762) #10, !dbg !20
  %address.local954 = or disjoint i64 %address.local942, 256, !dbg !11
  %address.local955 = shl nuw nsw i64 %address.local954, 3, !dbg !11
  %address.local956 = and i64 %address.local955, 3584, !dbg !11
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local956, !dbg !11
  %address.local958 = ptrtoint ptr addrspace(3) %address.local957 to i64, !dbg !11
  %address.local959 = trunc i64 %address.local958 to i32, !dbg !11
  %1951 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local959), !dbg !20
  %1952 = zext i32 %1951 to i64, !dbg !20
  %1953 = inttoptr i64 %1952 to ptr addrspace(3), !dbg !20
  %address.local960 = shl nuw nsw i64 %address.local954, 5, !dbg !20
  %address.local961 = and i64 %address.local960, 16256, !dbg !20
  %address.local962 = shl nuw nsw i64 %address.local942, 3, !dbg !20
  %address.local963 = and i64 %address.local962, 24, !dbg !20
  %address.local964 = or disjoint i64 %address.local961, %address.local963, !dbg !20
  %address.local965 = getelementptr [2 x i8], ptr addrspace(1) %1936, i64 %address.local964, !dbg !20
  %1954 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1955 = ptrtoint ptr addrspace(1) %address.local965 to i64, !dbg !20
  %1956 = sub i64 %1955, %1954, !dbg !20
  %dma.buffer.offset1758 = trunc i64 %1956 to i32, !dbg !20
  %1957 = trunc i64 %1954 to i32, !dbg !20
  %1958 = insertelement <4 x i32> poison, i32 %1957, i64 0, !dbg !20
  %1959 = lshr i64 %1954, 32, !dbg !20
  %1960 = trunc i64 %1959 to i32, !dbg !20
  %1961 = insertelement <4 x i32> %1958, i32 %1960, i64 1, !dbg !20
  %1962 = insertelement <4 x i32> %1961, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1759 = insertelement <4 x i32> %1962, i32 131072, i64 3, !dbg !20
  %1963 = ptrtoint ptr addrspace(3) %1953 to i32, !dbg !20
  %dma.buffer.lds1760 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1963), !dbg !20
  %1964 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1760, i32 %dma.buffer.offset1758, <4 x i32> %dma.buffer.rsrc1759) #10, !dbg !20
  %1965 = extractvalue [2 x <8 x float>] %66, 0, !dbg !36
  %1966 = shufflevector <8 x float> %1965, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1967 = insertelement <4 x float> poison, float %1933, i64 0, !dbg !36
  %1968 = shufflevector <4 x float> %1967, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1969 = fmul <4 x float> %1966, %1968, !dbg !36
  %1970 = insertelement <4 x float> poison, float %1807, i64 0, !dbg !37
  %1971 = insertelement <4 x float> %1970, float %1813, i64 1, !dbg !37
  %1972 = insertelement <4 x float> %1971, float %1808, i64 2, !dbg !37
  %1973 = insertelement <4 x float> %1972, float %1814, i64 3, !dbg !37
  %1974 = fadd <4 x float> %1969, %1973, !dbg !37
  %1975 = shufflevector <8 x float> %1965, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1976 = fmul <4 x float> %1975, %1968, !dbg !36
  %1977 = insertelement <4 x float> poison, float %1851, i64 0, !dbg !37
  %1978 = insertelement <4 x float> %1977, float %1857, i64 1, !dbg !37
  %1979 = insertelement <4 x float> %1978, float %1852, i64 2, !dbg !37
  %1980 = insertelement <4 x float> %1979, float %1858, i64 3, !dbg !37
  %1981 = fadd <4 x float> %1976, %1980, !dbg !37
  %1982 = extractvalue [2 x <8 x float>] %66, 1, !dbg !36
  %1983 = shufflevector <8 x float> %1982, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1984 = insertelement <4 x float> poison, float %1934, i64 0, !dbg !36
  %1985 = shufflevector <4 x float> %1984, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1986 = fmul <4 x float> %1983, %1985, !dbg !36
  %1987 = insertelement <4 x float> poison, float %1887, i64 0, !dbg !37
  %1988 = insertelement <4 x float> %1987, float %1893, i64 1, !dbg !37
  %1989 = insertelement <4 x float> %1988, float %1888, i64 2, !dbg !37
  %1990 = insertelement <4 x float> %1989, float %1894, i64 3, !dbg !37
  %1991 = fadd <4 x float> %1986, %1990, !dbg !37
  %1992 = shufflevector <8 x float> %1982, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1993 = fmul <4 x float> %1992, %1985, !dbg !36
  %1994 = insertelement <4 x float> poison, float %1923, i64 0, !dbg !37
  %1995 = insertelement <4 x float> %1994, float %1929, i64 1, !dbg !37
  %1996 = insertelement <4 x float> %1995, float %1924, i64 2, !dbg !37
  %1997 = insertelement <4 x float> %1996, float %1930, i64 3, !dbg !37
  %1998 = fadd <4 x float> %1993, %1997, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local966 = shl nuw nsw i32 %address.thread928, 3, !dbg !16
  %address.local967 = and i32 %address.local966, 504, !dbg !16
  %address.local968 = zext nneg i32 %address.local967 to i64, !dbg !16
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %1999 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #0, !dbg !16
  %2000 = bitcast <4 x i32> %1999 to <8 x half>, !dbg !16
  %address.local970 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local970, i64 1024, !dbg !16
  %2001 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #0, !dbg !16
  %2002 = bitcast <4 x i32> %2001 to <8 x half>, !dbg !16
  %address.local972 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local972, i64 2048, !dbg !16
  %2003 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !16
  %2004 = bitcast <4 x i32> %2003 to <8 x half>, !dbg !16
  %address.local974 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local975 = getelementptr i8, ptr addrspace(3) %address.local974, i64 3072, !dbg !16
  %2005 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #0, !dbg !16
  %2006 = bitcast <4 x i32> %2005 to <8 x half>, !dbg !16
  %address.local976 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local976, i64 4096, !dbg !16
  %2007 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !16
  %2008 = bitcast <4 x i32> %2007 to <8 x half>, !dbg !16
  %address.local978 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local979 = getelementptr i8, ptr addrspace(3) %address.local978, i64 5120, !dbg !16
  %2009 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #0, !dbg !16
  %2010 = bitcast <4 x i32> %2009 to <8 x half>, !dbg !16
  %address.local980 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local981 = getelementptr i8, ptr addrspace(3) %address.local980, i64 6144, !dbg !16
  %2011 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #0, !dbg !16
  %2012 = bitcast <4 x i32> %2011 to <8 x half>, !dbg !16
  %address.local982 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !16
  %address.local983 = getelementptr i8, ptr addrspace(3) %address.local982, i64 7168, !dbg !16
  %2013 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #0, !dbg !16
  %2014 = bitcast <4 x i32> %2013 to <8 x half>, !dbg !16
  %2015 = shufflevector <8 x half> %2000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2015, <4 x float> zeroinitializer) #9, !dbg !16
  %2017 = shufflevector <8 x half> %2002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2017, <4 x float> %2016) #9, !dbg !16
  %2019 = shufflevector <8 x half> %2004, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2019, <4 x float> %2018) #9, !dbg !16
  %2021 = shufflevector <8 x half> %2006, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2021, <4 x float> %2020) #9, !dbg !16
  %2023 = shufflevector <8 x half> %2008, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2023, <4 x float> %2022) #9, !dbg !16
  %2025 = shufflevector <8 x half> %2010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2025, <4 x float> %2024) #9, !dbg !16
  %2027 = shufflevector <8 x half> %2012, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2027, <4 x float> %2026) #9, !dbg !16
  %2029 = shufflevector <8 x half> %2014, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2029, <4 x float> %2028) #9, !dbg !16
  %2031 = extractelement <4 x float> %2030, i64 0, !dbg !16
  %2032 = extractelement <4 x float> %2030, i64 1, !dbg !16
  %2033 = extractelement <4 x float> %2030, i64 2, !dbg !16
  %2034 = extractelement <4 x float> %2030, i64 3, !dbg !16
  %2035 = select i1 %.not, float %2032, float %2031, !dbg !16
  %2036 = bitcast float %2035 to i32, !dbg !16
  %2037 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2036), !dbg !16
  %2038 = bitcast i32 %2037 to float, !dbg !16
  %2039 = select i1 %.not, float %2031, float %2038, !dbg !16
  %2040 = select i1 %.not, float %2038, float %2032, !dbg !16
  %2041 = select i1 %.not, float %2034, float %2033, !dbg !16
  %2042 = bitcast float %2041 to i32, !dbg !16
  %2043 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2042), !dbg !16
  %2044 = bitcast i32 %2043 to float, !dbg !16
  %2045 = select i1 %.not, float %2033, float %2044, !dbg !16
  %2046 = select i1 %.not, float %2044, float %2034, !dbg !16
  %2047 = select i1 %.not1, float %2045, float %2039, !dbg !16
  %2048 = bitcast float %2047 to i32, !dbg !16
  %2049 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2048), !dbg !16
  %2050 = bitcast i32 %2049 to float, !dbg !16
  %2051 = select i1 %.not1, float %2039, float %2050, !dbg !16
  %2052 = select i1 %.not1, float %2050, float %2045, !dbg !16
  %2053 = select i1 %.not1, float %2046, float %2040, !dbg !16
  %2054 = bitcast float %2053 to i32, !dbg !16
  %2055 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2054), !dbg !16
  %2056 = bitcast i32 %2055 to float, !dbg !16
  %2057 = select i1 %.not1, float %2040, float %2056, !dbg !16
  %2058 = select i1 %.not1, float %2056, float %2046, !dbg !16
  %2059 = shufflevector <8 x half> %2000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2059, <4 x float> zeroinitializer) #9, !dbg !16
  %2061 = shufflevector <8 x half> %2002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2061, <4 x float> %2060) #9, !dbg !16
  %2063 = shufflevector <8 x half> %2004, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2063, <4 x float> %2062) #9, !dbg !16
  %2065 = shufflevector <8 x half> %2006, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2065, <4 x float> %2064) #9, !dbg !16
  %2067 = shufflevector <8 x half> %2008, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2067, <4 x float> %2066) #9, !dbg !16
  %2069 = shufflevector <8 x half> %2010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2069, <4 x float> %2068) #9, !dbg !16
  %2071 = shufflevector <8 x half> %2012, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2071, <4 x float> %2070) #9, !dbg !16
  %2073 = shufflevector <8 x half> %2014, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2073, <4 x float> %2072) #9, !dbg !16
  %2075 = extractelement <4 x float> %2074, i64 0, !dbg !16
  %2076 = extractelement <4 x float> %2074, i64 1, !dbg !16
  %2077 = extractelement <4 x float> %2074, i64 2, !dbg !16
  %2078 = extractelement <4 x float> %2074, i64 3, !dbg !16
  %2079 = select i1 %.not, float %2076, float %2075, !dbg !16
  %2080 = bitcast float %2079 to i32, !dbg !16
  %2081 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2080), !dbg !16
  %2082 = bitcast i32 %2081 to float, !dbg !16
  %2083 = select i1 %.not, float %2075, float %2082, !dbg !16
  %2084 = select i1 %.not, float %2082, float %2076, !dbg !16
  %2085 = select i1 %.not, float %2078, float %2077, !dbg !16
  %2086 = bitcast float %2085 to i32, !dbg !16
  %2087 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2086), !dbg !16
  %2088 = bitcast i32 %2087 to float, !dbg !16
  %2089 = select i1 %.not, float %2077, float %2088, !dbg !16
  %2090 = select i1 %.not, float %2088, float %2078, !dbg !16
  %2091 = select i1 %.not1, float %2089, float %2083, !dbg !16
  %2092 = bitcast float %2091 to i32, !dbg !16
  %2093 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2092), !dbg !16
  %2094 = bitcast i32 %2093 to float, !dbg !16
  %2095 = select i1 %.not1, float %2083, float %2094, !dbg !16
  %2096 = select i1 %.not1, float %2094, float %2089, !dbg !16
  %2097 = select i1 %.not1, float %2090, float %2084, !dbg !16
  %2098 = bitcast float %2097 to i32, !dbg !16
  %2099 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2098), !dbg !16
  %2100 = bitcast i32 %2099 to float, !dbg !16
  %2101 = select i1 %.not1, float %2084, float %2100, !dbg !16
  %2102 = select i1 %.not1, float %2100, float %2090, !dbg !16
  %2103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2015, <4 x float> zeroinitializer) #9, !dbg !16
  %2104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2017, <4 x float> %2103) #9, !dbg !16
  %2105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2019, <4 x float> %2104) #9, !dbg !16
  %2106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2021, <4 x float> %2105) #9, !dbg !16
  %2107 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2023, <4 x float> %2106) #9, !dbg !16
  %2108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2025, <4 x float> %2107) #9, !dbg !16
  %2109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2027, <4 x float> %2108) #9, !dbg !16
  %2110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2029, <4 x float> %2109) #9, !dbg !16
  %2111 = extractelement <4 x float> %2110, i64 0, !dbg !16
  %2112 = extractelement <4 x float> %2110, i64 1, !dbg !16
  %2113 = extractelement <4 x float> %2110, i64 2, !dbg !16
  %2114 = extractelement <4 x float> %2110, i64 3, !dbg !16
  %2115 = select i1 %.not, float %2112, float %2111, !dbg !16
  %2116 = bitcast float %2115 to i32, !dbg !16
  %2117 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2116), !dbg !16
  %2118 = bitcast i32 %2117 to float, !dbg !16
  %2119 = select i1 %.not, float %2111, float %2118, !dbg !16
  %2120 = select i1 %.not, float %2118, float %2112, !dbg !16
  %2121 = select i1 %.not, float %2114, float %2113, !dbg !16
  %2122 = bitcast float %2121 to i32, !dbg !16
  %2123 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2122), !dbg !16
  %2124 = bitcast i32 %2123 to float, !dbg !16
  %2125 = select i1 %.not, float %2113, float %2124, !dbg !16
  %2126 = select i1 %.not, float %2124, float %2114, !dbg !16
  %2127 = select i1 %.not1, float %2125, float %2119, !dbg !16
  %2128 = bitcast float %2127 to i32, !dbg !16
  %2129 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2128), !dbg !16
  %2130 = bitcast i32 %2129 to float, !dbg !16
  %2131 = select i1 %.not1, float %2119, float %2130, !dbg !16
  %2132 = select i1 %.not1, float %2130, float %2125, !dbg !16
  %2133 = select i1 %.not1, float %2126, float %2120, !dbg !16
  %2134 = bitcast float %2133 to i32, !dbg !16
  %2135 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2134), !dbg !16
  %2136 = bitcast i32 %2135 to float, !dbg !16
  %2137 = select i1 %.not1, float %2120, float %2136, !dbg !16
  %2138 = select i1 %.not1, float %2136, float %2126, !dbg !16
  %2139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2059, <4 x float> zeroinitializer) #9, !dbg !16
  %2140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2061, <4 x float> %2139) #9, !dbg !16
  %2141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2063, <4 x float> %2140) #9, !dbg !16
  %2142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2065, <4 x float> %2141) #9, !dbg !16
  %2143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2067, <4 x float> %2142) #9, !dbg !16
  %2144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2069, <4 x float> %2143) #9, !dbg !16
  %2145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2071, <4 x float> %2144) #9, !dbg !16
  %2146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2073, <4 x float> %2145) #9, !dbg !16
  %2147 = extractelement <4 x float> %2146, i64 0, !dbg !16
  %2148 = extractelement <4 x float> %2146, i64 1, !dbg !16
  %2149 = extractelement <4 x float> %2146, i64 2, !dbg !16
  %2150 = extractelement <4 x float> %2146, i64 3, !dbg !16
  %2151 = select i1 %.not, float %2148, float %2147, !dbg !16
  %2152 = bitcast float %2151 to i32, !dbg !16
  %2153 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2152), !dbg !16
  %2154 = bitcast i32 %2153 to float, !dbg !16
  %2155 = select i1 %.not, float %2147, float %2154, !dbg !16
  %2156 = select i1 %.not, float %2154, float %2148, !dbg !16
  %2157 = select i1 %.not, float %2150, float %2149, !dbg !16
  %2158 = bitcast float %2157 to i32, !dbg !16
  %2159 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2158), !dbg !16
  %2160 = bitcast i32 %2159 to float, !dbg !16
  %2161 = select i1 %.not, float %2149, float %2160, !dbg !16
  %2162 = select i1 %.not, float %2160, float %2150, !dbg !16
  %2163 = select i1 %.not1, float %2161, float %2155, !dbg !16
  %2164 = bitcast float %2163 to i32, !dbg !16
  %2165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2164), !dbg !16
  %2166 = bitcast i32 %2165 to float, !dbg !16
  %2167 = select i1 %.not1, float %2155, float %2166, !dbg !16
  %2168 = select i1 %.not1, float %2166, float %2161, !dbg !16
  %2169 = select i1 %.not1, float %2162, float %2156, !dbg !16
  %2170 = bitcast float %2169 to i32, !dbg !16
  %2171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2170), !dbg !16
  %2172 = bitcast i32 %2171 to float, !dbg !16
  %2173 = select i1 %.not1, float %2156, float %2172, !dbg !16
  %2174 = select i1 %.not1, float %2172, float %2162, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %2175, label %2176, !dbg !36

2175:                                             ; preds = %1932
  %address.thread984 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local985 = lshr i32 %address.thread984, 1, !dbg !15
  %address.local986 = and i32 %address.local985, 96, !dbg !15
  %address.local987 = and i32 %address.thread984, 15, !dbg !15
  %address.local988 = or disjoint i32 %address.local986, %address.local987, !dbg !15
  %address.local989 = zext nneg i32 %address.local988 to i64, !dbg !15
  %address.local990 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !15
  store float %1234, ptr addrspace(3) %address.local990, align 4, !dbg !36
  %address.local991 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !15
  %address.local992 = getelementptr i8, ptr addrspace(3) %address.local991, i64 64, !dbg !15
  store float %1235, ptr addrspace(3) %address.local992, align 4, !dbg !36
  br label %2176, !dbg !36

2176:                                             ; preds = %2175, %1932
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread993 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %lds.const.gep1751 = getelementptr i8, ptr addrspace(3) %address.local804, i64 0, !dbg !36
  %2177 = load float, ptr addrspace(3) %lds.const.gep1751, align 4, !dbg !36
  %lds.const.gep1752 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %2178 = load float, ptr addrspace(3) %lds.const.gep1752, align 4, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %2179 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1690, !dbg !20
  %2180 = getelementptr i8, ptr addrspace(1) %2179, i64 192, !dbg !20
  %address.local1007 = zext nneg i32 %address.thread993 to i64, !dbg !12
  %address.local1008 = shl nuw nsw i64 %address.local1007, 3, !dbg !11
  %address.local1009 = and i64 %address.local1008, 1536, !dbg !11
  %address.local1010 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1009, !dbg !11
  %address.local1011 = ptrtoint ptr addrspace(3) %address.local1010 to i64, !dbg !11
  %address.local1012 = trunc i64 %address.local1011 to i32, !dbg !11
  %2181 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1012), !dbg !20
  %2182 = zext i32 %2181 to i64, !dbg !20
  %2183 = inttoptr i64 %2182 to ptr addrspace(3), !dbg !20
  %address.local1013 = shl nuw nsw i64 %address.local1007, 5, !dbg !20
  %address.local1014 = and i64 %address.local1013, 8064, !dbg !20
  %address.local1015 = shl nuw nsw i64 %address.local1007, 3, !dbg !20
  %address.local1016 = and i64 %address.local1015, 24, !dbg !20
  %address.local1017 = or disjoint i64 %address.local1014, %address.local1016, !dbg !20
  %address.local1018 = getelementptr [2 x i8], ptr addrspace(1) %2180, i64 %address.local1017, !dbg !20
  %2184 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %2185 = ptrtoint ptr addrspace(1) %address.local1018 to i64, !dbg !20
  %2186 = sub i64 %2185, %2184, !dbg !20
  %dma.buffer.offset1755 = trunc i64 %2186 to i32, !dbg !20
  %2187 = trunc i64 %2184 to i32, !dbg !20
  %2188 = insertelement <4 x i32> poison, i32 %2187, i64 0, !dbg !20
  %2189 = lshr i64 %2184, 32, !dbg !20
  %2190 = trunc i64 %2189 to i32, !dbg !20
  %2191 = insertelement <4 x i32> %2188, i32 %2190, i64 1, !dbg !20
  %2192 = insertelement <4 x i32> %2191, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1756 = insertelement <4 x i32> %2192, i32 131072, i64 3, !dbg !20
  %2193 = ptrtoint ptr addrspace(3) %2183 to i32, !dbg !20
  %dma.buffer.lds1757 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %2193), !dbg !20
  %2194 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1757, i32 %dma.buffer.offset1755, <4 x i32> %dma.buffer.rsrc1756) #10, !dbg !20
  %address.local1019 = or disjoint i64 %address.local1007, 256, !dbg !11
  %address.local1020 = shl nuw nsw i64 %address.local1019, 3, !dbg !11
  %address.local1021 = and i64 %address.local1020, 3584, !dbg !11
  %address.local1022 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1021, !dbg !11
  %address.local1023 = ptrtoint ptr addrspace(3) %address.local1022 to i64, !dbg !11
  %address.local1024 = trunc i64 %address.local1023 to i32, !dbg !11
  %2195 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1024), !dbg !20
  %2196 = zext i32 %2195 to i64, !dbg !20
  %2197 = inttoptr i64 %2196 to ptr addrspace(3), !dbg !20
  %address.local1025 = shl nuw nsw i64 %address.local1019, 5, !dbg !20
  %address.local1026 = and i64 %address.local1025, 16256, !dbg !20
  %address.local1027 = shl nuw nsw i64 %address.local1007, 3, !dbg !20
  %address.local1028 = and i64 %address.local1027, 24, !dbg !20
  %address.local1029 = or disjoint i64 %address.local1026, %address.local1028, !dbg !20
  %address.local1030 = getelementptr [2 x i8], ptr addrspace(1) %2180, i64 %address.local1029, !dbg !20
  %2198 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %2199 = ptrtoint ptr addrspace(1) %address.local1030 to i64, !dbg !20
  %2200 = sub i64 %2199, %2198, !dbg !20
  %dma.buffer.offset = trunc i64 %2200 to i32, !dbg !20
  %2201 = trunc i64 %2198 to i32, !dbg !20
  %2202 = insertelement <4 x i32> poison, i32 %2201, i64 0, !dbg !20
  %2203 = lshr i64 %2198, 32, !dbg !20
  %2204 = trunc i64 %2203 to i32, !dbg !20
  %2205 = insertelement <4 x i32> %2202, i32 %2204, i64 1, !dbg !20
  %2206 = insertelement <4 x i32> %2205, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc = insertelement <4 x i32> %2206, i32 131072, i64 3, !dbg !20
  %2207 = ptrtoint ptr addrspace(3) %2197 to i32, !dbg !20
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %2207), !dbg !20
  %2208 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !20
  %2209 = extractvalue [2 x <8 x float>] %67, 0, !dbg !36
  %2210 = shufflevector <8 x float> %2209, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2211 = insertelement <4 x float> poison, float %2177, i64 0, !dbg !36
  %2212 = shufflevector <4 x float> %2211, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2213 = fmul <4 x float> %2210, %2212, !dbg !36
  %2214 = insertelement <4 x float> poison, float %2051, i64 0, !dbg !37
  %2215 = insertelement <4 x float> %2214, float %2057, i64 1, !dbg !37
  %2216 = insertelement <4 x float> %2215, float %2052, i64 2, !dbg !37
  %2217 = insertelement <4 x float> %2216, float %2058, i64 3, !dbg !37
  %2218 = fadd <4 x float> %2213, %2217, !dbg !37
  %2219 = shufflevector <8 x float> %2209, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2220 = fmul <4 x float> %2219, %2212, !dbg !36
  %2221 = insertelement <4 x float> poison, float %2095, i64 0, !dbg !37
  %2222 = insertelement <4 x float> %2221, float %2101, i64 1, !dbg !37
  %2223 = insertelement <4 x float> %2222, float %2096, i64 2, !dbg !37
  %2224 = insertelement <4 x float> %2223, float %2102, i64 3, !dbg !37
  %2225 = fadd <4 x float> %2220, %2224, !dbg !37
  %2226 = extractvalue [2 x <8 x float>] %67, 1, !dbg !36
  %2227 = shufflevector <8 x float> %2226, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2228 = insertelement <4 x float> poison, float %2178, i64 0, !dbg !36
  %2229 = shufflevector <4 x float> %2228, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2230 = fmul <4 x float> %2227, %2229, !dbg !36
  %2231 = insertelement <4 x float> poison, float %2131, i64 0, !dbg !37
  %2232 = insertelement <4 x float> %2231, float %2137, i64 1, !dbg !37
  %2233 = insertelement <4 x float> %2232, float %2132, i64 2, !dbg !37
  %2234 = insertelement <4 x float> %2233, float %2138, i64 3, !dbg !37
  %2235 = fadd <4 x float> %2230, %2234, !dbg !37
  %2236 = shufflevector <8 x float> %2226, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2237 = fmul <4 x float> %2236, %2229, !dbg !36
  %2238 = insertelement <4 x float> poison, float %2167, i64 0, !dbg !37
  %2239 = insertelement <4 x float> %2238, float %2173, i64 1, !dbg !37
  %2240 = insertelement <4 x float> %2239, float %2168, i64 2, !dbg !37
  %2241 = insertelement <4 x float> %2240, float %2174, i64 3, !dbg !37
  %2242 = fadd <4 x float> %2237, %2241, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local1031 = shl nuw nsw i32 %address.thread993, 3, !dbg !16
  %address.local1032 = and i32 %address.local1031, 504, !dbg !16
  %address.local1033 = zext nneg i32 %address.local1032 to i64, !dbg !16
  %address.local1034 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %2243 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !16
  %2244 = bitcast <4 x i32> %2243 to <8 x half>, !dbg !16
  %address.local1035 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1036 = getelementptr i8, ptr addrspace(3) %address.local1035, i64 1024, !dbg !16
  %2245 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1036, i32 0) #0, !dbg !16
  %2246 = bitcast <4 x i32> %2245 to <8 x half>, !dbg !16
  %address.local1037 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1038 = getelementptr i8, ptr addrspace(3) %address.local1037, i64 2048, !dbg !16
  %2247 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1038, i32 0) #0, !dbg !16
  %2248 = bitcast <4 x i32> %2247 to <8 x half>, !dbg !16
  %address.local1039 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 3072, !dbg !16
  %2249 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !16
  %2250 = bitcast <4 x i32> %2249 to <8 x half>, !dbg !16
  %address.local1041 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1042 = getelementptr i8, ptr addrspace(3) %address.local1041, i64 4096, !dbg !16
  %2251 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1042, i32 0) #0, !dbg !16
  %2252 = bitcast <4 x i32> %2251 to <8 x half>, !dbg !16
  %address.local1043 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1043, i64 5120, !dbg !16
  %2253 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !16
  %2254 = bitcast <4 x i32> %2253 to <8 x half>, !dbg !16
  %address.local1045 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1046 = getelementptr i8, ptr addrspace(3) %address.local1045, i64 6144, !dbg !16
  %2255 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1046, i32 0) #0, !dbg !16
  %2256 = bitcast <4 x i32> %2255 to <8 x half>, !dbg !16
  %address.local1047 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !16
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1047, i64 7168, !dbg !16
  %2257 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !16
  %2258 = bitcast <4 x i32> %2257 to <8 x half>, !dbg !16
  %2259 = shufflevector <8 x half> %2244, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2259, <4 x float> zeroinitializer) #9, !dbg !16
  %2261 = shufflevector <8 x half> %2246, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2261, <4 x float> %2260) #9, !dbg !16
  %2263 = shufflevector <8 x half> %2248, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2263, <4 x float> %2262) #9, !dbg !16
  %2265 = shufflevector <8 x half> %2250, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2265, <4 x float> %2264) #9, !dbg !16
  %2267 = shufflevector <8 x half> %2252, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2267, <4 x float> %2266) #9, !dbg !16
  %2269 = shufflevector <8 x half> %2254, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2269, <4 x float> %2268) #9, !dbg !16
  %2271 = shufflevector <8 x half> %2256, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2271, <4 x float> %2270) #9, !dbg !16
  %2273 = shufflevector <8 x half> %2258, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2273, <4 x float> %2272) #9, !dbg !16
  %2275 = extractelement <4 x float> %2274, i64 0, !dbg !16
  %2276 = extractelement <4 x float> %2274, i64 1, !dbg !16
  %2277 = extractelement <4 x float> %2274, i64 2, !dbg !16
  %2278 = extractelement <4 x float> %2274, i64 3, !dbg !16
  %2279 = select i1 %.not, float %2276, float %2275, !dbg !16
  %2280 = bitcast float %2279 to i32, !dbg !16
  %2281 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2280), !dbg !16
  %2282 = bitcast i32 %2281 to float, !dbg !16
  %2283 = select i1 %.not, float %2275, float %2282, !dbg !16
  %2284 = select i1 %.not, float %2282, float %2276, !dbg !16
  %2285 = select i1 %.not, float %2278, float %2277, !dbg !16
  %2286 = bitcast float %2285 to i32, !dbg !16
  %2287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2286), !dbg !16
  %2288 = bitcast i32 %2287 to float, !dbg !16
  %2289 = select i1 %.not, float %2277, float %2288, !dbg !16
  %2290 = select i1 %.not, float %2288, float %2278, !dbg !16
  %2291 = select i1 %.not1, float %2289, float %2283, !dbg !16
  %2292 = bitcast float %2291 to i32, !dbg !16
  %2293 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2292), !dbg !16
  %2294 = bitcast i32 %2293 to float, !dbg !16
  %2295 = select i1 %.not1, float %2283, float %2294, !dbg !16
  %2296 = select i1 %.not1, float %2294, float %2289, !dbg !16
  %2297 = select i1 %.not1, float %2290, float %2284, !dbg !16
  %2298 = bitcast float %2297 to i32, !dbg !16
  %2299 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2298), !dbg !16
  %2300 = bitcast i32 %2299 to float, !dbg !16
  %2301 = select i1 %.not1, float %2284, float %2300, !dbg !16
  %2302 = select i1 %.not1, float %2300, float %2290, !dbg !16
  %2303 = shufflevector <8 x half> %2244, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2303, <4 x float> zeroinitializer) #9, !dbg !16
  %2305 = shufflevector <8 x half> %2246, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2305, <4 x float> %2304) #9, !dbg !16
  %2307 = shufflevector <8 x half> %2248, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2307, <4 x float> %2306) #9, !dbg !16
  %2309 = shufflevector <8 x half> %2250, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2309, <4 x float> %2308) #9, !dbg !16
  %2311 = shufflevector <8 x half> %2252, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2311, <4 x float> %2310) #9, !dbg !16
  %2313 = shufflevector <8 x half> %2254, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2313, <4 x float> %2312) #9, !dbg !16
  %2315 = shufflevector <8 x half> %2256, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2315, <4 x float> %2314) #9, !dbg !16
  %2317 = shufflevector <8 x half> %2258, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2317, <4 x float> %2316) #9, !dbg !16
  %2319 = extractelement <4 x float> %2318, i64 0, !dbg !16
  %2320 = extractelement <4 x float> %2318, i64 1, !dbg !16
  %2321 = extractelement <4 x float> %2318, i64 2, !dbg !16
  %2322 = extractelement <4 x float> %2318, i64 3, !dbg !16
  %2323 = select i1 %.not, float %2320, float %2319, !dbg !16
  %2324 = bitcast float %2323 to i32, !dbg !16
  %2325 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2324), !dbg !16
  %2326 = bitcast i32 %2325 to float, !dbg !16
  %2327 = select i1 %.not, float %2319, float %2326, !dbg !16
  %2328 = select i1 %.not, float %2326, float %2320, !dbg !16
  %2329 = select i1 %.not, float %2322, float %2321, !dbg !16
  %2330 = bitcast float %2329 to i32, !dbg !16
  %2331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2330), !dbg !16
  %2332 = bitcast i32 %2331 to float, !dbg !16
  %2333 = select i1 %.not, float %2321, float %2332, !dbg !16
  %2334 = select i1 %.not, float %2332, float %2322, !dbg !16
  %2335 = select i1 %.not1, float %2333, float %2327, !dbg !16
  %2336 = bitcast float %2335 to i32, !dbg !16
  %2337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2336), !dbg !16
  %2338 = bitcast i32 %2337 to float, !dbg !16
  %2339 = select i1 %.not1, float %2327, float %2338, !dbg !16
  %2340 = select i1 %.not1, float %2338, float %2333, !dbg !16
  %2341 = select i1 %.not1, float %2334, float %2328, !dbg !16
  %2342 = bitcast float %2341 to i32, !dbg !16
  %2343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2342), !dbg !16
  %2344 = bitcast i32 %2343 to float, !dbg !16
  %2345 = select i1 %.not1, float %2328, float %2344, !dbg !16
  %2346 = select i1 %.not1, float %2344, float %2334, !dbg !16
  %2347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2259, <4 x float> zeroinitializer) #9, !dbg !16
  %2348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2261, <4 x float> %2347) #9, !dbg !16
  %2349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2263, <4 x float> %2348) #9, !dbg !16
  %2350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2265, <4 x float> %2349) #9, !dbg !16
  %2351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2267, <4 x float> %2350) #9, !dbg !16
  %2352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2269, <4 x float> %2351) #9, !dbg !16
  %2353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2271, <4 x float> %2352) #9, !dbg !16
  %2354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2273, <4 x float> %2353) #9, !dbg !16
  %2355 = extractelement <4 x float> %2354, i64 0, !dbg !16
  %2356 = extractelement <4 x float> %2354, i64 1, !dbg !16
  %2357 = extractelement <4 x float> %2354, i64 2, !dbg !16
  %2358 = extractelement <4 x float> %2354, i64 3, !dbg !16
  %2359 = select i1 %.not, float %2356, float %2355, !dbg !16
  %2360 = bitcast float %2359 to i32, !dbg !16
  %2361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2360), !dbg !16
  %2362 = bitcast i32 %2361 to float, !dbg !16
  %2363 = select i1 %.not, float %2355, float %2362, !dbg !16
  %2364 = select i1 %.not, float %2362, float %2356, !dbg !16
  %2365 = select i1 %.not, float %2358, float %2357, !dbg !16
  %2366 = bitcast float %2365 to i32, !dbg !16
  %2367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2366), !dbg !16
  %2368 = bitcast i32 %2367 to float, !dbg !16
  %2369 = select i1 %.not, float %2357, float %2368, !dbg !16
  %2370 = select i1 %.not, float %2368, float %2358, !dbg !16
  %2371 = select i1 %.not1, float %2369, float %2363, !dbg !16
  %2372 = bitcast float %2371 to i32, !dbg !16
  %2373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2372), !dbg !16
  %2374 = bitcast i32 %2373 to float, !dbg !16
  %2375 = select i1 %.not1, float %2363, float %2374, !dbg !16
  %2376 = select i1 %.not1, float %2374, float %2369, !dbg !16
  %2377 = select i1 %.not1, float %2370, float %2364, !dbg !16
  %2378 = bitcast float %2377 to i32, !dbg !16
  %2379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2378), !dbg !16
  %2380 = bitcast i32 %2379 to float, !dbg !16
  %2381 = select i1 %.not1, float %2364, float %2380, !dbg !16
  %2382 = select i1 %.not1, float %2380, float %2370, !dbg !16
  %2383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2303, <4 x float> zeroinitializer) #9, !dbg !16
  %2384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2305, <4 x float> %2383) #9, !dbg !16
  %2385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2307, <4 x float> %2384) #9, !dbg !16
  %2386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2309, <4 x float> %2385) #9, !dbg !16
  %2387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2311, <4 x float> %2386) #9, !dbg !16
  %2388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2313, <4 x float> %2387) #9, !dbg !16
  %2389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2315, <4 x float> %2388) #9, !dbg !16
  %2390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2317, <4 x float> %2389) #9, !dbg !16
  %2391 = extractelement <4 x float> %2390, i64 0, !dbg !16
  %2392 = extractelement <4 x float> %2390, i64 1, !dbg !16
  %2393 = extractelement <4 x float> %2390, i64 2, !dbg !16
  %2394 = extractelement <4 x float> %2390, i64 3, !dbg !16
  %2395 = select i1 %.not, float %2392, float %2391, !dbg !16
  %2396 = bitcast float %2395 to i32, !dbg !16
  %2397 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2396), !dbg !16
  %2398 = bitcast i32 %2397 to float, !dbg !16
  %2399 = select i1 %.not, float %2391, float %2398, !dbg !16
  %2400 = select i1 %.not, float %2398, float %2392, !dbg !16
  %2401 = select i1 %.not, float %2394, float %2393, !dbg !16
  %2402 = bitcast float %2401 to i32, !dbg !16
  %2403 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2402), !dbg !16
  %2404 = bitcast i32 %2403 to float, !dbg !16
  %2405 = select i1 %.not, float %2393, float %2404, !dbg !16
  %2406 = select i1 %.not, float %2404, float %2394, !dbg !16
  %2407 = select i1 %.not1, float %2405, float %2399, !dbg !16
  %2408 = bitcast float %2407 to i32, !dbg !16
  %2409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2408), !dbg !16
  %2410 = bitcast i32 %2409 to float, !dbg !16
  %2411 = select i1 %.not1, float %2399, float %2410, !dbg !16
  %2412 = select i1 %.not1, float %2410, float %2405, !dbg !16
  %2413 = select i1 %.not1, float %2406, float %2400, !dbg !16
  %2414 = bitcast float %2413 to i32, !dbg !16
  %2415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2414), !dbg !16
  %2416 = bitcast i32 %2415 to float, !dbg !16
  %2417 = select i1 %.not1, float %2400, float %2416, !dbg !16
  %2418 = select i1 %.not1, float %2416, float %2406, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %43, label %2419, label %2420, !dbg !36

2419:                                             ; preds = %2176
  %address.thread1049 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local1050 = lshr i32 %address.thread1049, 1, !dbg !15
  %address.local1051 = and i32 %address.local1050, 96, !dbg !15
  %address.local1052 = and i32 %address.thread1049, 15, !dbg !15
  %address.local1053 = or disjoint i32 %address.local1051, %address.local1052, !dbg !15
  %address.local1054 = zext nneg i32 %address.local1053 to i64, !dbg !15
  %address.local1055 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !15
  store float %1234, ptr addrspace(3) %address.local1055, align 4, !dbg !36
  %address.local1056 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !15
  %address.local1057 = getelementptr i8, ptr addrspace(3) %address.local1056, i64 64, !dbg !15
  store float %1235, ptr addrspace(3) %address.local1057, align 4, !dbg !36
  br label %2420, !dbg !36

2420:                                             ; preds = %2419, %2176
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %lds.const.gep1753 = getelementptr i8, ptr addrspace(3) %address.local804, i64 0, !dbg !36
  %2421 = load float, ptr addrspace(3) %lds.const.gep1753, align 4, !dbg !36
  %lds.const.gep1754 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !36
  %2422 = load float, ptr addrspace(3) %lds.const.gep1754, align 4, !dbg !36
  %2423 = extractvalue [2 x <8 x float>] %68, 0, !dbg !36
  %2424 = shufflevector <8 x float> %2423, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2425 = insertelement <4 x float> poison, float %2421, i64 0, !dbg !36
  %2426 = shufflevector <4 x float> %2425, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2427 = fmul <4 x float> %2424, %2426, !dbg !36
  %2428 = insertelement <4 x float> poison, float %2295, i64 0, !dbg !37
  %2429 = insertelement <4 x float> %2428, float %2301, i64 1, !dbg !37
  %2430 = insertelement <4 x float> %2429, float %2296, i64 2, !dbg !37
  %2431 = insertelement <4 x float> %2430, float %2302, i64 3, !dbg !37
  %2432 = fadd <4 x float> %2427, %2431, !dbg !37
  %2433 = shufflevector <8 x float> %2423, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2434 = fmul <4 x float> %2433, %2426, !dbg !36
  %2435 = insertelement <4 x float> poison, float %2339, i64 0, !dbg !37
  %2436 = insertelement <4 x float> %2435, float %2345, i64 1, !dbg !37
  %2437 = insertelement <4 x float> %2436, float %2340, i64 2, !dbg !37
  %2438 = insertelement <4 x float> %2437, float %2346, i64 3, !dbg !37
  %2439 = fadd <4 x float> %2434, %2438, !dbg !37
  %2440 = extractvalue [2 x <8 x float>] %68, 1, !dbg !36
  %2441 = shufflevector <8 x float> %2440, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2442 = insertelement <4 x float> poison, float %2422, i64 0, !dbg !36
  %2443 = shufflevector <4 x float> %2442, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2444 = fmul <4 x float> %2441, %2443, !dbg !36
  %2445 = insertelement <4 x float> poison, float %2375, i64 0, !dbg !37
  %2446 = insertelement <4 x float> %2445, float %2381, i64 1, !dbg !37
  %2447 = insertelement <4 x float> %2446, float %2376, i64 2, !dbg !37
  %2448 = insertelement <4 x float> %2447, float %2382, i64 3, !dbg !37
  %2449 = fadd <4 x float> %2444, %2448, !dbg !37
  %2450 = shufflevector <8 x float> %2440, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2451 = fmul <4 x float> %2450, %2443, !dbg !36
  %2452 = insertelement <4 x float> poison, float %2411, i64 0, !dbg !37
  %2453 = insertelement <4 x float> %2452, float %2417, i64 1, !dbg !37
  %2454 = insertelement <4 x float> %2453, float %2412, i64 2, !dbg !37
  %2455 = insertelement <4 x float> %2454, float %2418, i64 3, !dbg !37
  %2456 = fadd <4 x float> %2451, %2455, !dbg !37
  %2457 = shufflevector <4 x float> %1730, <4 x float> %1737, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2458 = insertvalue [2 x <8 x float>] poison, <8 x float> %2457, 0, !dbg !7
  %2459 = shufflevector <4 x float> %1747, <4 x float> %1754, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2460 = insertvalue [2 x <8 x float>] %2458, <8 x float> %2459, 1, !dbg !7
  %2461 = insertelement <1 x float> poison, float %1497, i64 0, !dbg !7
  %2462 = insertvalue [2 x <1 x float>] poison, <1 x float> %2461, 0, !dbg !7
  %2463 = insertelement <1 x float> poison, float %1506, i64 0, !dbg !7
  %2464 = insertvalue [2 x <1 x float>] %2462, <1 x float> %2463, 1, !dbg !7
  %2465 = insertvalue [2 x <1 x float>] poison, <1 x float> %1194, 0, !dbg !7
  %2466 = insertvalue [2 x <1 x float>] %2465, <1 x float> %1207, 1, !dbg !7
  %2467 = shufflevector <4 x float> %1974, <4 x float> %1981, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2468 = insertvalue [2 x <8 x float>] poison, <8 x float> %2467, 0, !dbg !7
  %2469 = shufflevector <4 x float> %1991, <4 x float> %1998, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2470 = insertvalue [2 x <8 x float>] %2468, <8 x float> %2469, 1, !dbg !7
  %2471 = shufflevector <4 x float> %2218, <4 x float> %2225, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2472 = insertvalue [2 x <8 x float>] poison, <8 x float> %2471, 0, !dbg !7
  %2473 = shufflevector <4 x float> %2235, <4 x float> %2242, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2474 = insertvalue [2 x <8 x float>] %2472, <8 x float> %2473, 1, !dbg !7
  %2475 = shufflevector <4 x float> %2432, <4 x float> %2439, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2476 = insertvalue [2 x <8 x float>] poison, <8 x float> %2475, 0, !dbg !7
  %2477 = shufflevector <4 x float> %2449, <4 x float> %2456, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2478 = insertvalue [2 x <8 x float>] %2476, <8 x float> %2477, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %2479 = add nuw nsw i64 %62, 1, !dbg !7
  br label %61, !dbg !7

2480:                                             ; preds = %61
  %2481 = extractvalue [2 x <1 x float>] %64, 0, !dbg !21
  %2482 = shufflevector <1 x float> %2481, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %2483 = extractvalue [2 x <1 x float>] %64, 1, !dbg !21
  %2484 = shufflevector <1 x float> %2483, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  br i1 true, label %2485, label %2515, !dbg !23

2485:                                             ; preds = %2480
  %2486 = extractvalue [2 x <8 x float>] %63, 1, !dbg !21
  %2487 = shufflevector <8 x float> %2486, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2484, !dbg !21
  %normalized = fmul <4 x float> %2487, %row.reciprocal, !dbg !21
  %2488 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !22
  %2489 = extractelement <4 x half> %2488, i64 3, !dbg !22
  %2490 = extractelement <4 x half> %2488, i64 2, !dbg !22
  %2491 = extractelement <4 x half> %2488, i64 1, !dbg !22
  %2492 = extractelement <4 x half> %2488, i64 0, !dbg !22
  %2493 = shufflevector <8 x float> %2486, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized68 = fmul <4 x float> %2493, %row.reciprocal, !dbg !21
  %2494 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !22
  %2495 = extractelement <4 x half> %2494, i64 3, !dbg !22
  %2496 = extractelement <4 x half> %2494, i64 2, !dbg !22
  %2497 = extractelement <4 x half> %2494, i64 1, !dbg !22
  %2498 = extractelement <4 x half> %2494, i64 0, !dbg !22
  %2499 = extractvalue [2 x <8 x float>] %63, 0, !dbg !21
  %2500 = shufflevector <8 x float> %2499, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2482, !dbg !21
  %normalized70 = fmul <4 x float> %2500, %row.reciprocal69, !dbg !21
  %2501 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !22
  %2502 = extractelement <4 x half> %2501, i64 3, !dbg !22
  %2503 = extractelement <4 x half> %2501, i64 2, !dbg !22
  %2504 = extractelement <4 x half> %2501, i64 1, !dbg !22
  %2505 = extractelement <4 x half> %2501, i64 0, !dbg !22
  %2506 = shufflevector <8 x float> %2499, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized71 = fmul <4 x float> %2506, %row.reciprocal69, !dbg !21
  %2507 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !22
  %2508 = extractelement <4 x half> %2507, i64 3, !dbg !22
  %2509 = extractelement <4 x half> %2507, i64 2, !dbg !22
  %2510 = extractelement <4 x half> %2507, i64 1, !dbg !22
  %2511 = extractelement <4 x half> %2507, i64 0, !dbg !22
  %2512 = add nsw i64 %15, %25, !dbg !23
  %2513 = shl i64 %2512, 19, !dbg !23
  %2514 = and i64 %2513, 16252928, !dbg !23
  %address.thread1072 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1073 = zext nneg i32 %address.thread1072 to i64, !dbg !12
  %address.local1074 = add i64 %28, %address.local1073, !dbg !23
  %address.local1075 = lshr i64 %address.local1073, 2, !dbg !13
  %address.local1076 = and i64 %address.local1075, 48, !dbg !13
  %address.local1077 = add i64 %address.local1074, %address.local1076, !dbg !23
  %address.local1078 = lshr i64 %address.local1073, 4, !dbg !12
  %address.local1079 = mul nsw i64 %address.local1078, -16, !dbg !13
  %address.local1080 = add i64 %address.local1077, %address.local1079, !dbg !23
  %address.local1081 = add i64 %address.local1080, %29, !dbg !23
  %address.local1082 = shl nuw nsw i64 %address.local1081, 7, !dbg !23
  %address.local1083 = add nuw nsw i64 %2514, %address.local1082, !dbg !23
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1085 = lshr i64 %address.local1073, 4, !dbg !12
  %address.local1086 = and i64 %address.local1085, 3, !dbg !12
  %address.local1087 = shl nuw nsw i64 %address.local1086, 2, !dbg !12
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1087, !dbg !23
  store half %2511, ptr addrspace(1) %address.local1088, align 2, !dbg !23
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1090 = or disjoint i64 %address.local1087, 1, !dbg !13
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1090, !dbg !23
  store half %2510, ptr addrspace(1) %address.local1091, align 2, !dbg !23
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1093 = or disjoint i64 %address.local1087, 2, !dbg !13
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1093, !dbg !23
  store half %2509, ptr addrspace(1) %address.local1094, align 2, !dbg !23
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1096 = or disjoint i64 %address.local1087, 3, !dbg !13
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1096, !dbg !23
  store half %2508, ptr addrspace(1) %address.local1097, align 2, !dbg !23
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1099 = or disjoint i64 %address.local1087, 16, !dbg !13
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1098, i64 %address.local1099, !dbg !23
  store half %2505, ptr addrspace(1) %address.local1100, align 2, !dbg !23
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1102 = or disjoint i64 %address.local1087, 17, !dbg !13
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !23
  store half %2504, ptr addrspace(1) %address.local1103, align 2, !dbg !23
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1105 = or disjoint i64 %address.local1087, 18, !dbg !13
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !23
  store half %2503, ptr addrspace(1) %address.local1106, align 2, !dbg !23
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !23
  %address.local1108 = or disjoint i64 %address.local1087, 19, !dbg !13
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !23
  store half %2502, ptr addrspace(1) %address.local1109, align 2, !dbg !23
  %address.local1110 = shl i64 %address.local1081, 7, !dbg !23
  %address.local1111 = add i64 %address.local1110, 8192, !dbg !23
  %address.local1112 = add nuw nsw i64 %2514, %address.local1111, !dbg !23
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1087, !dbg !23
  store half %2498, ptr addrspace(1) %address.local1114, align 2, !dbg !23
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1090, !dbg !23
  store half %2497, ptr addrspace(1) %address.local1116, align 2, !dbg !23
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1093, !dbg !23
  store half %2496, ptr addrspace(1) %address.local1118, align 2, !dbg !23
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1096, !dbg !23
  store half %2495, ptr addrspace(1) %address.local1120, align 2, !dbg !23
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1099, !dbg !23
  store half %2492, ptr addrspace(1) %address.local1122, align 2, !dbg !23
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1102, !dbg !23
  store half %2491, ptr addrspace(1) %address.local1124, align 2, !dbg !23
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1105, !dbg !23
  store half %2490, ptr addrspace(1) %address.local1126, align 2, !dbg !23
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !23
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1108, !dbg !23
  store half %2489, ptr addrspace(1) %address.local1128, align 2, !dbg !23
  br label %2515, !dbg !23

2515:                                             ; preds = %2485, %2480
  br i1 true, label %2516, label %2544, !dbg !23

2516:                                             ; preds = %2515
  %2517 = extractvalue [2 x <8 x float>] %66, 1, !dbg !21
  %2518 = shufflevector <8 x float> %2517, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2484, !dbg !21
  %normalized73 = fmul <4 x float> %2518, %row.reciprocal72, !dbg !21
  %2519 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !22
  %2520 = extractelement <4 x half> %2519, i64 3, !dbg !22
  %2521 = extractelement <4 x half> %2519, i64 2, !dbg !22
  %2522 = extractelement <4 x half> %2519, i64 1, !dbg !22
  %2523 = extractelement <4 x half> %2519, i64 0, !dbg !22
  %2524 = shufflevector <8 x float> %2517, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized74 = fmul <4 x float> %2524, %row.reciprocal72, !dbg !21
  %2525 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !22
  %2526 = extractelement <4 x half> %2525, i64 3, !dbg !22
  %2527 = extractelement <4 x half> %2525, i64 2, !dbg !22
  %2528 = extractelement <4 x half> %2525, i64 1, !dbg !22
  %2529 = extractelement <4 x half> %2525, i64 0, !dbg !22
  %2530 = extractvalue [2 x <8 x float>] %66, 0, !dbg !21
  %2531 = shufflevector <8 x float> %2530, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2482, !dbg !21
  %normalized76 = fmul <4 x float> %2531, %row.reciprocal75, !dbg !21
  %2532 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !22
  %2533 = extractelement <4 x half> %2532, i64 3, !dbg !22
  %2534 = extractelement <4 x half> %2532, i64 2, !dbg !22
  %2535 = extractelement <4 x half> %2532, i64 1, !dbg !22
  %2536 = extractelement <4 x half> %2532, i64 0, !dbg !22
  %2537 = shufflevector <8 x float> %2530, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized77 = fmul <4 x float> %2537, %row.reciprocal75, !dbg !21
  %2538 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !22
  %2539 = extractelement <4 x half> %2538, i64 3, !dbg !22
  %2540 = extractelement <4 x half> %2538, i64 2, !dbg !22
  %2541 = extractelement <4 x half> %2538, i64 1, !dbg !22
  %2542 = extractelement <4 x half> %2538, i64 0, !dbg !22
  %2543 = shl nuw nsw i64 %27, 19, !dbg !23
  %address.thread1129 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1130 = zext nneg i32 %address.thread1129 to i64, !dbg !12
  %address.local1131 = add i64 %28, %address.local1130, !dbg !23
  %address.local1132 = lshr i64 %address.local1130, 2, !dbg !13
  %address.local1133 = and i64 %address.local1132, 48, !dbg !13
  %address.local1134 = add i64 %address.local1131, %address.local1133, !dbg !23
  %address.local1135 = lshr i64 %address.local1130, 4, !dbg !12
  %address.local1136 = mul nsw i64 %address.local1135, -16, !dbg !13
  %address.local1137 = add i64 %address.local1134, %address.local1136, !dbg !23
  %address.local1138 = add i64 %address.local1137, %29, !dbg !23
  %address.local1139 = shl nuw nsw i64 %address.local1138, 7, !dbg !23
  %address.local1140 = add nuw nsw i64 %2543, %address.local1139, !dbg !23
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1142 = lshr i64 %address.local1130, 4, !dbg !12
  %address.local1143 = and i64 %address.local1142, 3, !dbg !12
  %address.local1144 = shl nuw nsw i64 %address.local1143, 2, !dbg !12
  %address.local1145 = or disjoint i64 %address.local1144, 32, !dbg !13
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1145, !dbg !23
  store half %2542, ptr addrspace(1) %address.local1146, align 2, !dbg !23
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1148 = or disjoint i64 %address.local1144, 33, !dbg !13
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1148, !dbg !23
  store half %2541, ptr addrspace(1) %address.local1149, align 2, !dbg !23
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1151 = or disjoint i64 %address.local1144, 34, !dbg !13
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1150, i64 %address.local1151, !dbg !23
  store half %2540, ptr addrspace(1) %address.local1152, align 2, !dbg !23
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1154 = or disjoint i64 %address.local1144, 35, !dbg !13
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1154, !dbg !23
  store half %2539, ptr addrspace(1) %address.local1155, align 2, !dbg !23
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1157 = or disjoint i64 %address.local1144, 48, !dbg !13
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1156, i64 %address.local1157, !dbg !23
  store half %2536, ptr addrspace(1) %address.local1158, align 2, !dbg !23
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1160 = or disjoint i64 %address.local1144, 49, !dbg !13
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1160, !dbg !23
  store half %2535, ptr addrspace(1) %address.local1161, align 2, !dbg !23
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1163 = or disjoint i64 %address.local1144, 50, !dbg !13
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1162, i64 %address.local1163, !dbg !23
  store half %2534, ptr addrspace(1) %address.local1164, align 2, !dbg !23
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !23
  %address.local1166 = or disjoint i64 %address.local1144, 51, !dbg !13
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1166, !dbg !23
  store half %2533, ptr addrspace(1) %address.local1167, align 2, !dbg !23
  %address.local1168 = shl i64 %address.local1138, 7, !dbg !23
  %address.local1169 = add i64 %address.local1168, 8192, !dbg !23
  %address.local1170 = add nuw nsw i64 %2543, %address.local1169, !dbg !23
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1145, !dbg !23
  store half %2529, ptr addrspace(1) %address.local1172, align 2, !dbg !23
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1148, !dbg !23
  store half %2528, ptr addrspace(1) %address.local1174, align 2, !dbg !23
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1151, !dbg !23
  store half %2527, ptr addrspace(1) %address.local1176, align 2, !dbg !23
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1154, !dbg !23
  store half %2526, ptr addrspace(1) %address.local1178, align 2, !dbg !23
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1157, !dbg !23
  store half %2523, ptr addrspace(1) %address.local1180, align 2, !dbg !23
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1160, !dbg !23
  store half %2522, ptr addrspace(1) %address.local1182, align 2, !dbg !23
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1163, !dbg !23
  store half %2521, ptr addrspace(1) %address.local1184, align 2, !dbg !23
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !23
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1166, !dbg !23
  store half %2520, ptr addrspace(1) %address.local1186, align 2, !dbg !23
  br label %2544, !dbg !23

2544:                                             ; preds = %2516, %2515
  br i1 true, label %2545, label %2573, !dbg !23

2545:                                             ; preds = %2544
  %2546 = extractvalue [2 x <8 x float>] %67, 1, !dbg !21
  %2547 = shufflevector <8 x float> %2546, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2484, !dbg !21
  %normalized79 = fmul <4 x float> %2547, %row.reciprocal78, !dbg !21
  %2548 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !22
  %2549 = extractelement <4 x half> %2548, i64 3, !dbg !22
  %2550 = extractelement <4 x half> %2548, i64 2, !dbg !22
  %2551 = extractelement <4 x half> %2548, i64 1, !dbg !22
  %2552 = extractelement <4 x half> %2548, i64 0, !dbg !22
  %2553 = shufflevector <8 x float> %2546, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized80 = fmul <4 x float> %2553, %row.reciprocal78, !dbg !21
  %2554 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !22
  %2555 = extractelement <4 x half> %2554, i64 3, !dbg !22
  %2556 = extractelement <4 x half> %2554, i64 2, !dbg !22
  %2557 = extractelement <4 x half> %2554, i64 1, !dbg !22
  %2558 = extractelement <4 x half> %2554, i64 0, !dbg !22
  %2559 = extractvalue [2 x <8 x float>] %67, 0, !dbg !21
  %2560 = shufflevector <8 x float> %2559, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2482, !dbg !21
  %normalized82 = fmul <4 x float> %2560, %row.reciprocal81, !dbg !21
  %2561 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !22
  %2562 = extractelement <4 x half> %2561, i64 3, !dbg !22
  %2563 = extractelement <4 x half> %2561, i64 2, !dbg !22
  %2564 = extractelement <4 x half> %2561, i64 1, !dbg !22
  %2565 = extractelement <4 x half> %2561, i64 0, !dbg !22
  %2566 = shufflevector <8 x float> %2559, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized83 = fmul <4 x float> %2566, %row.reciprocal81, !dbg !21
  %2567 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !22
  %2568 = extractelement <4 x half> %2567, i64 3, !dbg !22
  %2569 = extractelement <4 x half> %2567, i64 2, !dbg !22
  %2570 = extractelement <4 x half> %2567, i64 1, !dbg !22
  %2571 = extractelement <4 x half> %2567, i64 0, !dbg !22
  %2572 = shl nuw nsw i64 %27, 19, !dbg !23
  %address.thread1187 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1188 = zext nneg i32 %address.thread1187 to i64, !dbg !12
  %address.local1189 = add i64 %28, %address.local1188, !dbg !23
  %address.local1190 = lshr i64 %address.local1188, 2, !dbg !13
  %address.local1191 = and i64 %address.local1190, 48, !dbg !13
  %address.local1192 = add i64 %address.local1189, %address.local1191, !dbg !23
  %address.local1193 = lshr i64 %address.local1188, 4, !dbg !12
  %address.local1194 = mul nsw i64 %address.local1193, -16, !dbg !13
  %address.local1195 = add i64 %address.local1192, %address.local1194, !dbg !23
  %address.local1196 = add i64 %address.local1195, %29, !dbg !23
  %address.local1197 = shl nuw nsw i64 %address.local1196, 7, !dbg !23
  %address.local1198 = add nuw nsw i64 %2572, %address.local1197, !dbg !23
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1200 = lshr i64 %address.local1188, 4, !dbg !12
  %address.local1201 = and i64 %address.local1200, 3, !dbg !12
  %address.local1202 = shl nuw nsw i64 %address.local1201, 2, !dbg !12
  %address.local1203 = or disjoint i64 %address.local1202, 64, !dbg !13
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1203, !dbg !23
  store half %2571, ptr addrspace(1) %address.local1204, align 2, !dbg !23
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1206 = or disjoint i64 %address.local1202, 65, !dbg !13
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1206, !dbg !23
  store half %2570, ptr addrspace(1) %address.local1207, align 2, !dbg !23
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1209 = or disjoint i64 %address.local1202, 66, !dbg !13
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1208, i64 %address.local1209, !dbg !23
  store half %2569, ptr addrspace(1) %address.local1210, align 2, !dbg !23
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1212 = or disjoint i64 %address.local1202, 67, !dbg !13
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1211, i64 %address.local1212, !dbg !23
  store half %2568, ptr addrspace(1) %address.local1213, align 2, !dbg !23
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1215 = or disjoint i64 %address.local1202, 80, !dbg !13
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1214, i64 %address.local1215, !dbg !23
  store half %2565, ptr addrspace(1) %address.local1216, align 2, !dbg !23
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1218 = or disjoint i64 %address.local1202, 81, !dbg !13
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1217, i64 %address.local1218, !dbg !23
  store half %2564, ptr addrspace(1) %address.local1219, align 2, !dbg !23
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1221 = or disjoint i64 %address.local1202, 82, !dbg !13
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1221, !dbg !23
  store half %2563, ptr addrspace(1) %address.local1222, align 2, !dbg !23
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !23
  %address.local1224 = or disjoint i64 %address.local1202, 83, !dbg !13
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1223, i64 %address.local1224, !dbg !23
  store half %2562, ptr addrspace(1) %address.local1225, align 2, !dbg !23
  %address.local1226 = shl i64 %address.local1196, 7, !dbg !23
  %address.local1227 = add i64 %address.local1226, 8192, !dbg !23
  %address.local1228 = add nuw nsw i64 %2572, %address.local1227, !dbg !23
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1229, i64 %address.local1203, !dbg !23
  store half %2558, ptr addrspace(1) %address.local1230, align 2, !dbg !23
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1231, i64 %address.local1206, !dbg !23
  store half %2557, ptr addrspace(1) %address.local1232, align 2, !dbg !23
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1233, i64 %address.local1209, !dbg !23
  store half %2556, ptr addrspace(1) %address.local1234, align 2, !dbg !23
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1235, i64 %address.local1212, !dbg !23
  store half %2555, ptr addrspace(1) %address.local1236, align 2, !dbg !23
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1237, i64 %address.local1215, !dbg !23
  store half %2552, ptr addrspace(1) %address.local1238, align 2, !dbg !23
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1239, i64 %address.local1218, !dbg !23
  store half %2551, ptr addrspace(1) %address.local1240, align 2, !dbg !23
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1241, i64 %address.local1221, !dbg !23
  store half %2550, ptr addrspace(1) %address.local1242, align 2, !dbg !23
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !23
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1243, i64 %address.local1224, !dbg !23
  store half %2549, ptr addrspace(1) %address.local1244, align 2, !dbg !23
  br label %2573, !dbg !23

2573:                                             ; preds = %2545, %2544
  br i1 true, label %2574, label %2602, !dbg !23

2574:                                             ; preds = %2573
  %2575 = extractvalue [2 x <8 x float>] %68, 1, !dbg !21
  %2576 = shufflevector <8 x float> %2575, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2484, !dbg !21
  %normalized85 = fmul <4 x float> %2576, %row.reciprocal84, !dbg !21
  %2577 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !22
  %2578 = extractelement <4 x half> %2577, i64 3, !dbg !22
  %2579 = extractelement <4 x half> %2577, i64 2, !dbg !22
  %2580 = extractelement <4 x half> %2577, i64 1, !dbg !22
  %2581 = extractelement <4 x half> %2577, i64 0, !dbg !22
  %2582 = shufflevector <8 x float> %2575, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized86 = fmul <4 x float> %2582, %row.reciprocal84, !dbg !21
  %2583 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !22
  %2584 = extractelement <4 x half> %2583, i64 3, !dbg !22
  %2585 = extractelement <4 x half> %2583, i64 2, !dbg !22
  %2586 = extractelement <4 x half> %2583, i64 1, !dbg !22
  %2587 = extractelement <4 x half> %2583, i64 0, !dbg !22
  %2588 = extractvalue [2 x <8 x float>] %68, 0, !dbg !21
  %2589 = shufflevector <8 x float> %2588, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2482, !dbg !21
  %normalized88 = fmul <4 x float> %2589, %row.reciprocal87, !dbg !21
  %2590 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !22
  %2591 = extractelement <4 x half> %2590, i64 3, !dbg !22
  %2592 = extractelement <4 x half> %2590, i64 2, !dbg !22
  %2593 = extractelement <4 x half> %2590, i64 1, !dbg !22
  %2594 = extractelement <4 x half> %2590, i64 0, !dbg !22
  %2595 = shufflevector <8 x float> %2588, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized89 = fmul <4 x float> %2595, %row.reciprocal87, !dbg !21
  %2596 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !22
  %2597 = extractelement <4 x half> %2596, i64 3, !dbg !22
  %2598 = extractelement <4 x half> %2596, i64 2, !dbg !22
  %2599 = extractelement <4 x half> %2596, i64 1, !dbg !22
  %2600 = extractelement <4 x half> %2596, i64 0, !dbg !22
  %2601 = shl nuw nsw i64 %27, 19, !dbg !23
  %address.thread1245 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1246 = zext nneg i32 %address.thread1245 to i64, !dbg !12
  %address.local1247 = add i64 %28, %address.local1246, !dbg !23
  %address.local1248 = lshr i64 %address.local1246, 2, !dbg !13
  %address.local1249 = and i64 %address.local1248, 48, !dbg !13
  %address.local1250 = add i64 %address.local1247, %address.local1249, !dbg !23
  %address.local1251 = lshr i64 %address.local1246, 4, !dbg !12
  %address.local1252 = mul nsw i64 %address.local1251, -16, !dbg !13
  %address.local1253 = add i64 %address.local1250, %address.local1252, !dbg !23
  %address.local1254 = add i64 %address.local1253, %29, !dbg !23
  %address.local1255 = shl nuw nsw i64 %address.local1254, 7, !dbg !23
  %address.local1256 = add nuw nsw i64 %2601, %address.local1255, !dbg !23
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1258 = lshr i64 %address.local1246, 4, !dbg !12
  %address.local1259 = and i64 %address.local1258, 3, !dbg !12
  %address.local1260 = shl nuw nsw i64 %address.local1259, 2, !dbg !12
  %address.local1261 = or disjoint i64 %address.local1260, 96, !dbg !13
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1261, !dbg !23
  store half %2600, ptr addrspace(1) %address.local1262, align 2, !dbg !23
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1264 = or disjoint i64 %address.local1260, 97, !dbg !13
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1264, !dbg !23
  store half %2599, ptr addrspace(1) %address.local1265, align 2, !dbg !23
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1267 = or disjoint i64 %address.local1260, 98, !dbg !13
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1266, i64 %address.local1267, !dbg !23
  store half %2598, ptr addrspace(1) %address.local1268, align 2, !dbg !23
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1270 = or disjoint i64 %address.local1260, 99, !dbg !13
  %address.local1271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1269, i64 %address.local1270, !dbg !23
  store half %2597, ptr addrspace(1) %address.local1271, align 2, !dbg !23
  %address.local1272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1273 = or disjoint i64 %address.local1260, 112, !dbg !13
  %address.local1274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1272, i64 %address.local1273, !dbg !23
  store half %2594, ptr addrspace(1) %address.local1274, align 2, !dbg !23
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1276 = or disjoint i64 %address.local1260, 113, !dbg !13
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1275, i64 %address.local1276, !dbg !23
  store half %2593, ptr addrspace(1) %address.local1277, align 2, !dbg !23
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1279 = or disjoint i64 %address.local1260, 114, !dbg !13
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1278, i64 %address.local1279, !dbg !23
  store half %2592, ptr addrspace(1) %address.local1280, align 2, !dbg !23
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !23
  %address.local1282 = or disjoint i64 %address.local1260, 115, !dbg !13
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1281, i64 %address.local1282, !dbg !23
  store half %2591, ptr addrspace(1) %address.local1283, align 2, !dbg !23
  %address.local1284 = shl i64 %address.local1254, 7, !dbg !23
  %address.local1285 = add i64 %address.local1284, 8192, !dbg !23
  %address.local1286 = add nuw nsw i64 %2601, %address.local1285, !dbg !23
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1287, i64 %address.local1261, !dbg !23
  store half %2587, ptr addrspace(1) %address.local1288, align 2, !dbg !23
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1264, !dbg !23
  store half %2586, ptr addrspace(1) %address.local1290, align 2, !dbg !23
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1291, i64 %address.local1267, !dbg !23
  store half %2585, ptr addrspace(1) %address.local1292, align 2, !dbg !23
  %address.local1293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1293, i64 %address.local1270, !dbg !23
  store half %2584, ptr addrspace(1) %address.local1294, align 2, !dbg !23
  %address.local1295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1295, i64 %address.local1273, !dbg !23
  store half %2581, ptr addrspace(1) %address.local1296, align 2, !dbg !23
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1297, i64 %address.local1276, !dbg !23
  store half %2580, ptr addrspace(1) %address.local1298, align 2, !dbg !23
  %address.local1299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1299, i64 %address.local1279, !dbg !23
  store half %2579, ptr addrspace(1) %address.local1300, align 2, !dbg !23
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !23
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1282, !dbg !23
  store half %2578, ptr addrspace(1) %address.local1302, align 2, !dbg !23
  br label %2602, !dbg !23

2602:                                             ; preds = %2574, %2573
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000001")
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
