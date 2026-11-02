; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/run-20261027-121856-x88fub13/grid-000001/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/run-20261027-121856-x88fub13/grid-000001/input.mlir"

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
  %36 = lshr i64 %35, 6, !dbg !12
  %37 = lshr i64 %35, 4, !dbg !12
  %38 = and i64 %37, 3, !dbg !12
  %39 = shl nuw nsw i64 %38, 2, !dbg !12
  %40 = lshr i64 %35, 4, !dbg !12
  %41 = or disjoint i64 %19, 127, !dbg !14
  %42 = add i64 %19, 128, !dbg !14
  %43 = icmp uge i64 %42, %41, !dbg !14
  %44 = icmp eq i64 %38, 0, !dbg !15
  %45 = and i32 %34, 16, !dbg !16
  %.not = icmp eq i32 %45, 0, !dbg !16
  %46 = and i32 %34, 32, !dbg !16
  %.not1 = icmp eq i32 %46, 0, !dbg !16
  %47 = sub nsw i64 32, %14, !dbg !17
  %48 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %49 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %50 = getelementptr i8, ptr addrspace(1) %49, i64 64, !dbg !11
  %51 = shl i64 %15, 7, !dbg !18
  %52 = and i64 %51, 3968, !dbg !18
  %53 = shl nsw i64 %18, 12, !dbg !18
  %54 = or disjoint i64 %52, %53, !dbg !18
  %55 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %56 = getelementptr i8, ptr addrspace(1) %55, i64 128, !dbg !11
  %57 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %58 = getelementptr i8, ptr addrspace(1) %57, i64 192, !dbg !11
  %59 = shl nuw nsw i64 %36, 4, !dbg !13
  %60 = mul nsw i64 %40, -16, !dbg !13
  br label %61, !dbg !7

61:                                               ; preds = %2416, %4
  %62 = phi i64 [ %2475, %2416 ], [ 0, %4 ], !dbg !7
  %63 = phi [2 x <8 x float>] [ %2456, %2416 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi [2 x <1 x float>] [ %2460, %2416 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = phi [2 x <1 x float>] [ %2462, %2416 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %66 = phi [2 x <8 x float>] [ %2466, %2416 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = phi [2 x <8 x float>] [ %2470, %2416 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = phi [2 x <8 x float>] [ %2474, %2416 ], [ zeroinitializer, %4 ], !dbg !7
  %69 = icmp slt i64 %62, %47, !dbg !7
  br i1 %69, label %70, label %2476, !dbg !7

70:                                               ; preds = %61
  %71 = shl nsw i64 %62, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local257 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local258 = and i64 %address.local257, 1536, !dbg !11
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local258, !dbg !11
  %address.local260 = ptrtoint ptr addrspace(3) %address.local259 to i64, !dbg !11
  %address.local261 = trunc i64 %address.local260 to i32, !dbg !11
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local261), !dbg !11
  %73 = zext i32 %72 to i64, !dbg !11
  %74 = inttoptr i64 %73 to ptr addrspace(3), !dbg !11
  %address.local262 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local263 = and i64 %address.local262, 16256, !dbg !11
  %address.local264 = lshr i64 %address.local, 4, !dbg !11
  %address.local265 = and i64 %address.local264, 8, !dbg !11
  %address.local266 = or disjoint i64 %address.local263, %address.local265, !dbg !11
  %address.local267 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local266, !dbg !11
  %75 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %76 = ptrtoint ptr addrspace(1) %address.local267 to i64, !dbg !11
  %77 = sub i64 %76, %75, !dbg !11
  %dma.buffer.offset1806 = trunc i64 %77 to i32, !dbg !11
  %78 = trunc i64 %75 to i32, !dbg !11
  %79 = insertelement <4 x i32> poison, i32 %78, i64 0, !dbg !11
  %80 = lshr i64 %75, 32, !dbg !11
  %81 = trunc i64 %80 to i32, !dbg !11
  %82 = insertelement <4 x i32> %79, i32 %81, i64 1, !dbg !11
  %83 = insertelement <4 x i32> %82, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1807 = insertelement <4 x i32> %83, i32 131072, i64 3, !dbg !11
  %84 = ptrtoint ptr addrspace(3) %74 to i32, !dbg !11
  %dma.buffer.lds1808 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !11
  %85 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1806, <4 x i32> %dma.buffer.rsrc1807) #10, !dbg !11
  %address.local268 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local269 = shl nuw nsw i64 %address.local268, 3, !dbg !11
  %address.local270 = and i64 %address.local269, 3584, !dbg !11
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local270, !dbg !11
  %address.local272 = ptrtoint ptr addrspace(3) %address.local271 to i64, !dbg !11
  %address.local273 = trunc i64 %address.local272 to i32, !dbg !11
  %86 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local273), !dbg !11
  %87 = zext i32 %86 to i64, !dbg !11
  %88 = inttoptr i64 %87 to ptr addrspace(3), !dbg !11
  %address.local274 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local275 = and i64 %address.local274, 16256, !dbg !11
  %address.local276 = lshr i64 %address.local268, 4, !dbg !11
  %address.local277 = and i64 %address.local276, 24, !dbg !11
  %address.local278 = or disjoint i64 %address.local275, %address.local277, !dbg !11
  %address.local279 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local278, !dbg !11
  %89 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %90 = ptrtoint ptr addrspace(1) %address.local279 to i64, !dbg !11
  %91 = sub i64 %90, %89, !dbg !11
  %dma.buffer.offset1803 = trunc i64 %91 to i32, !dbg !11
  %92 = trunc i64 %89 to i32, !dbg !11
  %93 = insertelement <4 x i32> poison, i32 %92, i64 0, !dbg !11
  %94 = lshr i64 %89, 32, !dbg !11
  %95 = trunc i64 %94 to i32, !dbg !11
  %96 = insertelement <4 x i32> %93, i32 %95, i64 1, !dbg !11
  %97 = insertelement <4 x i32> %96, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1804 = insertelement <4 x i32> %97, i32 131072, i64 3, !dbg !11
  %98 = ptrtoint ptr addrspace(3) %88 to i32, !dbg !11
  %dma.buffer.lds1805 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %98), !dbg !11
  %99 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1803, <4 x i32> %dma.buffer.rsrc1804) #10, !dbg !11
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
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local258, !dbg !18
  %address.local281 = ptrtoint ptr addrspace(3) %address.local280 to i64, !dbg !18
  %address.local282 = trunc i64 %address.local281 to i32, !dbg !18
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local282), !dbg !18
  %113 = zext i32 %112 to i64, !dbg !18
  %114 = inttoptr i64 %113 to ptr addrspace(3), !dbg !18
  %address.local283 = lshr i64 %address.local, 4, !dbg !18
  %address.local284 = and i64 %address.local283, 8, !dbg !18
  %address.local285 = and i64 %address.local, 123, !dbg !18
  %address.local286 = lshr i64 %address.local, 2, !dbg !18
  %address.local287 = and i64 %address.local286, 4, !dbg !18
  %address.local288 = add nuw nsw i64 %address.local287, %address.local, !dbg !18
  %address.local289 = and i64 %address.local288, 4, !dbg !18
  %address.local290 = or disjoint i64 %address.local285, %address.local289, !dbg !18
  %address.local291 = shl nuw nsw i64 %address.local290, 7, !dbg !18
  %address.local292 = or disjoint i64 %address.local284, %address.local291, !dbg !18
  %address.local293 = getelementptr [2 x i8], ptr addrspace(1) %111, i64 %address.local292, !dbg !18
  %115 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %116 = ptrtoint ptr addrspace(1) %address.local293 to i64, !dbg !18
  %117 = sub i64 %116, %115, !dbg !18
  %dma.buffer.offset1800 = trunc i64 %117 to i32, !dbg !18
  %118 = trunc i64 %115 to i32, !dbg !18
  %119 = insertelement <4 x i32> poison, i32 %118, i64 0, !dbg !18
  %120 = lshr i64 %115, 32, !dbg !18
  %121 = trunc i64 %120 to i32, !dbg !18
  %122 = insertelement <4 x i32> %119, i32 %121, i64 1, !dbg !18
  %123 = insertelement <4 x i32> %122, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1801 = insertelement <4 x i32> %123, i32 131072, i64 3, !dbg !18
  %124 = ptrtoint ptr addrspace(3) %114 to i32, !dbg !18
  %dma.buffer.lds1802 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %124), !dbg !18
  %125 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1800, <4 x i32> %dma.buffer.rsrc1801) #10, !dbg !18
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local270, !dbg !18
  %address.local295 = ptrtoint ptr addrspace(3) %address.local294 to i64, !dbg !18
  %address.local296 = trunc i64 %address.local295 to i32, !dbg !18
  %126 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local296), !dbg !18
  %127 = zext i32 %126 to i64, !dbg !18
  %128 = inttoptr i64 %127 to ptr addrspace(3), !dbg !18
  %address.local297 = lshr i64 %address.local268, 4, !dbg !18
  %address.local298 = and i64 %address.local297, 24, !dbg !18
  %address.local299 = and i64 %address.local, 123, !dbg !18
  %address.local300 = lshr i64 %address.local, 2, !dbg !18
  %address.local301 = and i64 %address.local300, 4, !dbg !18
  %address.local302 = add nuw nsw i64 %address.local301, %address.local, !dbg !18
  %address.local303 = and i64 %address.local302, 4, !dbg !18
  %address.local304 = or disjoint i64 %address.local299, %address.local303, !dbg !18
  %address.local305 = shl nuw nsw i64 %address.local304, 7, !dbg !18
  %address.local306 = or disjoint i64 %address.local298, %address.local305, !dbg !18
  %address.local307 = getelementptr [2 x i8], ptr addrspace(1) %111, i64 %address.local306, !dbg !18
  %129 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %130 = ptrtoint ptr addrspace(1) %address.local307 to i64, !dbg !18
  %131 = sub i64 %130, %129, !dbg !18
  %dma.buffer.offset1797 = trunc i64 %131 to i32, !dbg !18
  %132 = trunc i64 %129 to i32, !dbg !18
  %133 = insertelement <4 x i32> poison, i32 %132, i64 0, !dbg !18
  %134 = lshr i64 %129, 32, !dbg !18
  %135 = trunc i64 %134 to i32, !dbg !18
  %136 = insertelement <4 x i32> %133, i32 %135, i64 1, !dbg !18
  %137 = insertelement <4 x i32> %136, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1798 = insertelement <4 x i32> %137, i32 131072, i64 3, !dbg !18
  %138 = ptrtoint ptr addrspace(3) %128 to i32, !dbg !18
  %dma.buffer.lds1799 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138), !dbg !18
  %139 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1797, <4 x i32> %dma.buffer.rsrc1798) #10, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local308 = lshr i32 %address.thread, 2, !dbg !12
  %address.local309 = and i32 %address.local308, 4, !dbg !12
  %address.local310 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local311 = and i32 %address.local310, 384, !dbg !12
  %address.local312 = or disjoint i32 %address.local309, %address.local311, !dbg !12
  %address.local313 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local314 = and i32 %address.local313, 120, !dbg !12
  %address.local315 = or disjoint i32 %address.local312, %address.local314, !dbg !12
  %address.local316 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local317 = and i32 %address.local316, 1024, !dbg !12
  %address.local318 = or disjoint i32 %address.local315, %address.local317, !dbg !12
  %address.local319 = zext nneg i32 %address.local318 to i64, !dbg !12
  %address.local320 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local319, !dbg !12
  %140 = load <4 x half>, ptr addrspace(3) %address.local320, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local320, i64 4096, !dbg !12
  %141 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local320, i64 1024, !dbg !12
  %142 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !12
  %143 = fmul <4 x half> %140, splat (half 0xH3015), !dbg !12
  %lds.const.gep1546 = getelementptr i8, ptr addrspace(3) %address.local320, i64 5120, !dbg !12
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep1546, align 8, !dbg !12
  %145 = fmul <4 x half> %141, splat (half 0xH3015), !dbg !12
  %146 = fmul <4 x half> %142, splat (half 0xH3015), !dbg !12
  %147 = fmul <4 x half> %144, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.local327 = lshr i32 %address.thread, 2, !dbg !19
  %address.local328 = and i32 %address.local327, 4, !dbg !19
  %address.local329 = shl nuw nsw i32 %address.thread, 3, !dbg !19
  %address.local330 = and i32 %address.local329, 120, !dbg !19
  %address.local331 = or disjoint i32 %address.local328, %address.local330, !dbg !19
  %address.local332 = shl nuw nsw i32 %address.thread, 5, !dbg !19
  %address.local333 = and i32 %address.local332, 1024, !dbg !19
  %address.local334 = or disjoint i32 %address.local331, %address.local333, !dbg !19
  %address.local335 = zext nneg i32 %address.local334 to i64, !dbg !19
  %address.local336 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local335, !dbg !19
  %148 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !19
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %148, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1547 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4096, !dbg !19
  %150 = load <4 x half>, ptr addrspace(3) %lds.const.gep1547, align 8, !dbg !19
  %address.local348 = lshr i32 %address.thread, 2, !dbg !19
  %address.local349 = and i32 %address.local348, 4, !dbg !19
  %address.local350 = shl nuw nsw i32 %address.thread, 3, !dbg !19
  %address.local351 = and i32 %address.local350, 88, !dbg !19
  %address.local352 = or disjoint i32 %address.local349, %address.local351, !dbg !19
  %address.local353 = shl nuw nsw i32 %address.thread, 3, !dbg !19
  %address.local354 = and i32 %address.local353, 32, !dbg !19
  %address.local355 = sub nsw i32 %address.local352, %address.local354, !dbg !19
  %address.local356 = shl nuw nsw i32 %address.thread, 5, !dbg !19
  %address.local357 = and i32 %address.local356, 1024, !dbg !19
  %address.local358 = add nsw i32 %address.local355, %address.local357, !dbg !19
  %address.local359 = sext i32 %address.local358 to i64, !dbg !19
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local359, !dbg !19
  %address.local361 = getelementptr i8, ptr addrspace(3) %address.local360, i64 320, !dbg !19
  %151 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !19
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %150, <4 x float> %149) #9, !dbg !19
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1548 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4096, !dbg !19
  %154 = load <4 x half>, ptr addrspace(3) %lds.const.gep1548, align 8, !dbg !19
  %lds.const.gep1549 = getelementptr i8, ptr addrspace(3) %address.local336, i64 512, !dbg !19
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep1549, align 8, !dbg !19
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %154, <4 x float> %153) #9, !dbg !19
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %155, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1550 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4608, !dbg !19
  %158 = load <4 x half>, ptr addrspace(3) %lds.const.gep1550, align 8, !dbg !19
  %lds.const.gep1551 = getelementptr i8, ptr addrspace(3) %address.local361, i64 512, !dbg !19
  %159 = load <4 x half>, ptr addrspace(3) %lds.const.gep1551, align 8, !dbg !19
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %158, <4 x float> %157) #9, !dbg !19
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %159, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1552 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !19
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep1552, align 8, !dbg !19
  %lds.const.gep1553 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1024, !dbg !19
  %163 = load <4 x half>, ptr addrspace(3) %lds.const.gep1553, align 8, !dbg !19
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %162, <4 x float> %161) #9, !dbg !19
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %163, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1554 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5120, !dbg !19
  %166 = load <4 x half>, ptr addrspace(3) %lds.const.gep1554, align 8, !dbg !19
  %lds.const.gep1555 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1024, !dbg !19
  %167 = load <4 x half>, ptr addrspace(3) %lds.const.gep1555, align 8, !dbg !19
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %166, <4 x float> %165) #9, !dbg !19
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %167, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1556 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5120, !dbg !19
  %170 = load <4 x half>, ptr addrspace(3) %lds.const.gep1556, align 8, !dbg !19
  %lds.const.gep1557 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1536, !dbg !19
  %171 = load <4 x half>, ptr addrspace(3) %lds.const.gep1557, align 8, !dbg !19
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %170, <4 x float> %169) #9, !dbg !19
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %171, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1558 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5632, !dbg !19
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1558, align 8, !dbg !19
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1536, !dbg !19
  %175 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !19
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %174, <4 x float> %173) #9, !dbg !19
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %175, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5632, !dbg !19
  %178 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !19
  %179 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !19
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %178, <4 x float> %177) #9, !dbg !19
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %179, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1561 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4096, !dbg !19
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !19
  %183 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !19
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %182, <4 x float> %181) #9, !dbg !19
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %183, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1562 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4096, !dbg !19
  %186 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !19
  %lds.const.gep1563 = getelementptr i8, ptr addrspace(3) %address.local336, i64 512, !dbg !19
  %187 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !19
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %186, <4 x float> %185) #9, !dbg !19
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %187, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1564 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4608, !dbg !19
  %190 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !19
  %lds.const.gep1565 = getelementptr i8, ptr addrspace(3) %address.local361, i64 512, !dbg !19
  %191 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !19
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %190, <4 x float> %189) #9, !dbg !19
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1566 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !19
  %194 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !19
  %lds.const.gep1567 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1024, !dbg !19
  %195 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !19
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %194, <4 x float> %193) #9, !dbg !19
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1568 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5120, !dbg !19
  %198 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !19
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1024, !dbg !19
  %199 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !19
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %198, <4 x float> %197) #9, !dbg !19
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5120, !dbg !19
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !19
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1536, !dbg !19
  %203 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !19
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %202, <4 x float> %201) #9, !dbg !19
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5632, !dbg !19
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !19
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1536, !dbg !19
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !19
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %206, <4 x float> %205) #9, !dbg !19
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %207, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5632, !dbg !19
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !19
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %210, <4 x float> %209) #9, !dbg !19
  %212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local261), !dbg !11
  %213 = zext i32 %212 to i64, !dbg !11
  %214 = inttoptr i64 %213 to ptr addrspace(3), !dbg !11
  %address.local400 = getelementptr [2 x i8], ptr addrspace(1) %50, i64 %address.local266, !dbg !11
  %215 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %216 = ptrtoint ptr addrspace(1) %address.local400 to i64, !dbg !11
  %217 = sub i64 %216, %215, !dbg !11
  %dma.buffer.offset1794 = trunc i64 %217 to i32, !dbg !11
  %218 = trunc i64 %215 to i32, !dbg !11
  %219 = insertelement <4 x i32> poison, i32 %218, i64 0, !dbg !11
  %220 = lshr i64 %215, 32, !dbg !11
  %221 = trunc i64 %220 to i32, !dbg !11
  %222 = insertelement <4 x i32> %219, i32 %221, i64 1, !dbg !11
  %223 = insertelement <4 x i32> %222, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1795 = insertelement <4 x i32> %223, i32 131072, i64 3, !dbg !11
  %224 = ptrtoint ptr addrspace(3) %214 to i32, !dbg !11
  %dma.buffer.lds1796 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %224), !dbg !11
  %225 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1796, i32 %dma.buffer.offset1794, <4 x i32> %dma.buffer.rsrc1795) #10, !dbg !11
  %226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local273), !dbg !11
  %227 = zext i32 %226 to i64, !dbg !11
  %228 = inttoptr i64 %227 to ptr addrspace(3), !dbg !11
  %address.local401 = getelementptr [2 x i8], ptr addrspace(1) %50, i64 %address.local278, !dbg !11
  %229 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %230 = ptrtoint ptr addrspace(1) %address.local401 to i64, !dbg !11
  %231 = sub i64 %230, %229, !dbg !11
  %dma.buffer.offset1791 = trunc i64 %231 to i32, !dbg !11
  %232 = trunc i64 %229 to i32, !dbg !11
  %233 = insertelement <4 x i32> poison, i32 %232, i64 0, !dbg !11
  %234 = lshr i64 %229, 32, !dbg !11
  %235 = trunc i64 %234 to i32, !dbg !11
  %236 = insertelement <4 x i32> %233, i32 %235, i64 1, !dbg !11
  %237 = insertelement <4 x i32> %236, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1792 = insertelement <4 x i32> %237, i32 131072, i64 3, !dbg !11
  %238 = ptrtoint ptr addrspace(3) %228 to i32, !dbg !11
  %dma.buffer.lds1793 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %238), !dbg !11
  %239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1793, i32 %dma.buffer.offset1791, <4 x i32> %dma.buffer.rsrc1792) #10, !dbg !11
  %240 = add nuw i64 %62, 1024, !dbg !18
  %241 = icmp slt i64 %240, 0, !dbg !18
  %242 = sub nuw i64 -1025, %62, !dbg !18
  %243 = select i1 %241, i64 %242, i64 %240, !dbg !18
  %244 = sdiv i64 %243, 4096, !dbg !18
  %.lobit = ashr i64 %240, 63, !dbg !18
  %245 = xor i64 %244, %.lobit, !dbg !18
  %246 = add nsw i64 %245, %15, !dbg !18
  %247 = add nsw i64 %102, %54, !dbg !18
  %248 = add nsw i64 %247, 32, !dbg !18
  %249 = icmp slt i64 %247, -32, !dbg !18
  %250 = sub nsw i64 -33, %247, !dbg !18
  %251 = select i1 %249, i64 %250, i64 %248, !dbg !18
  %252 = sdiv i64 %251, 128, !dbg !18
  %253 = sext i1 %249 to i64, !dbg !18
  %254 = xor i64 %252, %253, !dbg !18
  %255 = shl nsw i64 %254, 7, !dbg !18
  %256 = sub i64 %54, %255, !dbg !18
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
  %264 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local282), !dbg !18
  %265 = zext i32 %264 to i64, !dbg !18
  %266 = inttoptr i64 %265 to ptr addrspace(3), !dbg !18
  %address.local402 = getelementptr [2 x i8], ptr addrspace(1) %263, i64 %address.local292, !dbg !18
  %267 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %268 = ptrtoint ptr addrspace(1) %address.local402 to i64, !dbg !18
  %269 = sub i64 %268, %267, !dbg !18
  %dma.buffer.offset1788 = trunc i64 %269 to i32, !dbg !18
  %270 = trunc i64 %267 to i32, !dbg !18
  %271 = insertelement <4 x i32> poison, i32 %270, i64 0, !dbg !18
  %272 = lshr i64 %267, 32, !dbg !18
  %273 = trunc i64 %272 to i32, !dbg !18
  %274 = insertelement <4 x i32> %271, i32 %273, i64 1, !dbg !18
  %275 = insertelement <4 x i32> %274, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1789 = insertelement <4 x i32> %275, i32 131072, i64 3, !dbg !18
  %276 = ptrtoint ptr addrspace(3) %266 to i32, !dbg !18
  %dma.buffer.lds1790 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %276), !dbg !18
  %277 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1790, i32 %dma.buffer.offset1788, <4 x i32> %dma.buffer.rsrc1789) #10, !dbg !18
  %278 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local296), !dbg !18
  %279 = zext i32 %278 to i64, !dbg !18
  %280 = inttoptr i64 %279 to ptr addrspace(3), !dbg !18
  %address.local403 = getelementptr [2 x i8], ptr addrspace(1) %263, i64 %address.local306, !dbg !18
  %281 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %282 = ptrtoint ptr addrspace(1) %address.local403 to i64, !dbg !18
  %283 = sub i64 %282, %281, !dbg !18
  %dma.buffer.offset1785 = trunc i64 %283 to i32, !dbg !18
  %284 = trunc i64 %281 to i32, !dbg !18
  %285 = insertelement <4 x i32> poison, i32 %284, i64 0, !dbg !18
  %286 = lshr i64 %281, 32, !dbg !18
  %287 = trunc i64 %286 to i32, !dbg !18
  %288 = insertelement <4 x i32> %285, i32 %287, i64 1, !dbg !18
  %289 = insertelement <4 x i32> %288, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1786 = insertelement <4 x i32> %289, i32 131072, i64 3, !dbg !18
  %290 = ptrtoint ptr addrspace(3) %280 to i32, !dbg !18
  %dma.buffer.lds1787 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %290), !dbg !18
  %291 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1787, i32 %dma.buffer.offset1785, <4 x i32> %dma.buffer.rsrc1786) #10, !dbg !18
  %292 = load <4 x half>, ptr addrspace(3) %address.local320, align 8, !dbg !12
  %lds.const.gep1575 = getelementptr i8, ptr addrspace(3) %address.local320, i64 4096, !dbg !12
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1575, align 8, !dbg !12
  %lds.const.gep1576 = getelementptr i8, ptr addrspace(3) %address.local320, i64 1024, !dbg !12
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1576, align 8, !dbg !12
  %295 = fmul <4 x half> %292, splat (half 0xH3015), !dbg !12
  %lds.const.gep1577 = getelementptr i8, ptr addrspace(3) %address.local320, i64 5120, !dbg !12
  %296 = load <4 x half>, ptr addrspace(3) %lds.const.gep1577, align 8, !dbg !12
  %297 = fmul <4 x half> %293, splat (half 0xH3015), !dbg !12
  %298 = fmul <4 x half> %294, splat (half 0xH3015), !dbg !12
  %299 = fmul <4 x half> %296, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %300 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !19
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %300, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1578 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4096, !dbg !19
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1578, align 8, !dbg !19
  %303 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !19
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %302, <4 x float> %301) #9, !dbg !19
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %303, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4096, !dbg !19
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !19
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local336, i64 512, !dbg !19
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !19
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %306, <4 x float> %305) #9, !dbg !19
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4608, !dbg !19
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !19
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local361, i64 512, !dbg !19
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !19
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %310, <4 x float> %309) #9, !dbg !19
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %311, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !19
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !19
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1024, !dbg !19
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !19
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %314, <4 x float> %313) #9, !dbg !19
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %315, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5120, !dbg !19
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !19
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1024, !dbg !19
  %319 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !19
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %318, <4 x float> %317) #9, !dbg !19
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %319, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5120, !dbg !19
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !19
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1536, !dbg !19
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !19
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %322, <4 x float> %321) #9, !dbg !19
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %323, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5632, !dbg !19
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !19
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1536, !dbg !19
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !19
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %326, <4 x float> %325) #9, !dbg !19
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %327, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5632, !dbg !19
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !19
  %331 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !19
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %330, <4 x float> %329) #9, !dbg !19
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %331, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4096, !dbg !19
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !19
  %335 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !19
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %334, <4 x float> %333) #9, !dbg !19
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %335, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4096, !dbg !19
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !19
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local336, i64 512, !dbg !19
  %339 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !19
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %338, <4 x float> %337) #9, !dbg !19
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %339, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4608, !dbg !19
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !19
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %address.local361, i64 512, !dbg !19
  %343 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !19
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %342, <4 x float> %341) #9, !dbg !19
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %343, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !19
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !19
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1024, !dbg !19
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !19
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %346, <4 x float> %345) #9, !dbg !19
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %347, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5120, !dbg !19
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !19
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1024, !dbg !19
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !19
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %350, <4 x float> %349) #9, !dbg !19
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %351, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5120, !dbg !19
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !19
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1536, !dbg !19
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !19
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %354, <4 x float> %353) #9, !dbg !19
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %355, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5632, !dbg !19
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !19
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1536, !dbg !19
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !19
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %358, <4 x float> %357) #9, !dbg !19
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %359, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5632, !dbg !19
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !19
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %362, <4 x float> %361) #9, !dbg !19
  %364 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local261), !dbg !11
  %365 = zext i32 %364 to i64, !dbg !11
  %366 = inttoptr i64 %365 to ptr addrspace(3), !dbg !11
  %address.local404 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local266, !dbg !11
  %367 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %368 = ptrtoint ptr addrspace(1) %address.local404 to i64, !dbg !11
  %369 = sub i64 %368, %367, !dbg !11
  %dma.buffer.offset1782 = trunc i64 %369 to i32, !dbg !11
  %370 = trunc i64 %367 to i32, !dbg !11
  %371 = insertelement <4 x i32> poison, i32 %370, i64 0, !dbg !11
  %372 = lshr i64 %367, 32, !dbg !11
  %373 = trunc i64 %372 to i32, !dbg !11
  %374 = insertelement <4 x i32> %371, i32 %373, i64 1, !dbg !11
  %375 = insertelement <4 x i32> %374, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1783 = insertelement <4 x i32> %375, i32 131072, i64 3, !dbg !11
  %376 = ptrtoint ptr addrspace(3) %366 to i32, !dbg !11
  %dma.buffer.lds1784 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %376), !dbg !11
  %377 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1784, i32 %dma.buffer.offset1782, <4 x i32> %dma.buffer.rsrc1783) #10, !dbg !11
  %378 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local273), !dbg !11
  %379 = zext i32 %378 to i64, !dbg !11
  %380 = inttoptr i64 %379 to ptr addrspace(3), !dbg !11
  %address.local405 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local278, !dbg !11
  %381 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %382 = ptrtoint ptr addrspace(1) %address.local405 to i64, !dbg !11
  %383 = sub i64 %382, %381, !dbg !11
  %dma.buffer.offset1779 = trunc i64 %383 to i32, !dbg !11
  %384 = trunc i64 %381 to i32, !dbg !11
  %385 = insertelement <4 x i32> poison, i32 %384, i64 0, !dbg !11
  %386 = lshr i64 %381, 32, !dbg !11
  %387 = trunc i64 %386 to i32, !dbg !11
  %388 = insertelement <4 x i32> %385, i32 %387, i64 1, !dbg !11
  %389 = insertelement <4 x i32> %388, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1780 = insertelement <4 x i32> %389, i32 131072, i64 3, !dbg !11
  %390 = ptrtoint ptr addrspace(3) %380 to i32, !dbg !11
  %dma.buffer.lds1781 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %390), !dbg !11
  %391 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1781, i32 %dma.buffer.offset1779, <4 x i32> %dma.buffer.rsrc1780) #10, !dbg !11
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
  %407 = sub i64 %54, %406, !dbg !18
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
  %415 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local282), !dbg !18
  %416 = zext i32 %415 to i64, !dbg !18
  %417 = inttoptr i64 %416 to ptr addrspace(3), !dbg !18
  %address.local406 = getelementptr [2 x i8], ptr addrspace(1) %414, i64 %address.local292, !dbg !18
  %418 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %419 = ptrtoint ptr addrspace(1) %address.local406 to i64, !dbg !18
  %420 = sub i64 %419, %418, !dbg !18
  %dma.buffer.offset1776 = trunc i64 %420 to i32, !dbg !18
  %421 = trunc i64 %418 to i32, !dbg !18
  %422 = insertelement <4 x i32> poison, i32 %421, i64 0, !dbg !18
  %423 = lshr i64 %418, 32, !dbg !18
  %424 = trunc i64 %423 to i32, !dbg !18
  %425 = insertelement <4 x i32> %422, i32 %424, i64 1, !dbg !18
  %426 = insertelement <4 x i32> %425, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1777 = insertelement <4 x i32> %426, i32 131072, i64 3, !dbg !18
  %427 = ptrtoint ptr addrspace(3) %417 to i32, !dbg !18
  %dma.buffer.lds1778 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %427), !dbg !18
  %428 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1778, i32 %dma.buffer.offset1776, <4 x i32> %dma.buffer.rsrc1777) #10, !dbg !18
  %429 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local296), !dbg !18
  %430 = zext i32 %429 to i64, !dbg !18
  %431 = inttoptr i64 %430 to ptr addrspace(3), !dbg !18
  %address.local407 = getelementptr [2 x i8], ptr addrspace(1) %414, i64 %address.local306, !dbg !18
  %432 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %433 = ptrtoint ptr addrspace(1) %address.local407 to i64, !dbg !18
  %434 = sub i64 %433, %432, !dbg !18
  %dma.buffer.offset1773 = trunc i64 %434 to i32, !dbg !18
  %435 = trunc i64 %432 to i32, !dbg !18
  %436 = insertelement <4 x i32> poison, i32 %435, i64 0, !dbg !18
  %437 = lshr i64 %432, 32, !dbg !18
  %438 = trunc i64 %437 to i32, !dbg !18
  %439 = insertelement <4 x i32> %436, i32 %438, i64 1, !dbg !18
  %440 = insertelement <4 x i32> %439, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1774 = insertelement <4 x i32> %440, i32 131072, i64 3, !dbg !18
  %441 = ptrtoint ptr addrspace(3) %431 to i32, !dbg !18
  %dma.buffer.lds1775 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %441), !dbg !18
  %442 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1775, i32 %dma.buffer.offset1773, <4 x i32> %dma.buffer.rsrc1774) #10, !dbg !18
  %443 = load <4 x half>, ptr addrspace(3) %address.local320, align 8, !dbg !12
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local320, i64 4096, !dbg !12
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !12
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %address.local320, i64 1024, !dbg !12
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !12
  %446 = fmul <4 x half> %443, splat (half 0xH3015), !dbg !12
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local320, i64 5120, !dbg !12
  %447 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !12
  %448 = fmul <4 x half> %444, splat (half 0xH3015), !dbg !12
  %449 = fmul <4 x half> %445, splat (half 0xH3015), !dbg !12
  %450 = fmul <4 x half> %447, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %451 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !19
  %452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %451, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4096, !dbg !19
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !19
  %454 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !19
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %453, <4 x float> %452) #9, !dbg !19
  %456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %454, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4096, !dbg !19
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !19
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local336, i64 512, !dbg !19
  %458 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !19
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %457, <4 x float> %456) #9, !dbg !19
  %460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %458, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4608, !dbg !19
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !19
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local361, i64 512, !dbg !19
  %462 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !19
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %461, <4 x float> %460) #9, !dbg !19
  %464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %462, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !19
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !19
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1024, !dbg !19
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !19
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %465, <4 x float> %464) #9, !dbg !19
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %466, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5120, !dbg !19
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !19
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1024, !dbg !19
  %470 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !19
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %469, <4 x float> %468) #9, !dbg !19
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %470, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5120, !dbg !19
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !19
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1536, !dbg !19
  %474 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !19
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %473, <4 x float> %472) #9, !dbg !19
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %474, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5632, !dbg !19
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !19
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1536, !dbg !19
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !19
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %477, <4 x float> %476) #9, !dbg !19
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %478, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5632, !dbg !19
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !19
  %482 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !19
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %481, <4 x float> %480) #9, !dbg !19
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %482, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4096, !dbg !19
  %485 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !19
  %486 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !19
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %485, <4 x float> %484) #9, !dbg !19
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %486, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4096, !dbg !19
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !19
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local336, i64 512, !dbg !19
  %490 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !19
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %489, <4 x float> %488) #9, !dbg !19
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %490, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4608, !dbg !19
  %493 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !19
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %address.local361, i64 512, !dbg !19
  %494 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !19
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %493, <4 x float> %492) #9, !dbg !19
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %494, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !19
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !19
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1024, !dbg !19
  %498 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !19
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %497, <4 x float> %496) #9, !dbg !19
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %498, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5120, !dbg !19
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !19
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1024, !dbg !19
  %502 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !19
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %501, <4 x float> %500) #9, !dbg !19
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %502, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5120, !dbg !19
  %505 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !19
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1536, !dbg !19
  %506 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !19
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %505, <4 x float> %504) #9, !dbg !19
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %506, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5632, !dbg !19
  %509 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !19
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1536, !dbg !19
  %510 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !19
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %509, <4 x float> %508) #9, !dbg !19
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %510, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5632, !dbg !19
  %513 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !19
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %513, <4 x float> %512) #9, !dbg !19
  %515 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local261), !dbg !11
  %516 = zext i32 %515 to i64, !dbg !11
  %517 = inttoptr i64 %516 to ptr addrspace(3), !dbg !11
  %address.local408 = getelementptr [2 x i8], ptr addrspace(1) %58, i64 %address.local266, !dbg !11
  %518 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %519 = ptrtoint ptr addrspace(1) %address.local408 to i64, !dbg !11
  %520 = sub i64 %519, %518, !dbg !11
  %dma.buffer.offset1770 = trunc i64 %520 to i32, !dbg !11
  %521 = trunc i64 %518 to i32, !dbg !11
  %522 = insertelement <4 x i32> poison, i32 %521, i64 0, !dbg !11
  %523 = lshr i64 %518, 32, !dbg !11
  %524 = trunc i64 %523 to i32, !dbg !11
  %525 = insertelement <4 x i32> %522, i32 %524, i64 1, !dbg !11
  %526 = insertelement <4 x i32> %525, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1771 = insertelement <4 x i32> %526, i32 131072, i64 3, !dbg !11
  %527 = ptrtoint ptr addrspace(3) %517 to i32, !dbg !11
  %dma.buffer.lds1772 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %527), !dbg !11
  %528 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1772, i32 %dma.buffer.offset1770, <4 x i32> %dma.buffer.rsrc1771) #10, !dbg !11
  %529 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local273), !dbg !11
  %530 = zext i32 %529 to i64, !dbg !11
  %531 = inttoptr i64 %530 to ptr addrspace(3), !dbg !11
  %address.local409 = getelementptr [2 x i8], ptr addrspace(1) %58, i64 %address.local278, !dbg !11
  %532 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %533 = ptrtoint ptr addrspace(1) %address.local409 to i64, !dbg !11
  %534 = sub i64 %533, %532, !dbg !11
  %dma.buffer.offset1767 = trunc i64 %534 to i32, !dbg !11
  %535 = trunc i64 %532 to i32, !dbg !11
  %536 = insertelement <4 x i32> poison, i32 %535, i64 0, !dbg !11
  %537 = lshr i64 %532, 32, !dbg !11
  %538 = trunc i64 %537 to i32, !dbg !11
  %539 = insertelement <4 x i32> %536, i32 %538, i64 1, !dbg !11
  %540 = insertelement <4 x i32> %539, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1768 = insertelement <4 x i32> %540, i32 131072, i64 3, !dbg !11
  %541 = ptrtoint ptr addrspace(3) %531 to i32, !dbg !11
  %dma.buffer.lds1769 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %541), !dbg !11
  %542 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1769, i32 %dma.buffer.offset1767, <4 x i32> %dma.buffer.rsrc1768) #10, !dbg !11
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
  %558 = sub i64 %54, %557, !dbg !18
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
  %566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local282), !dbg !18
  %567 = zext i32 %566 to i64, !dbg !18
  %568 = inttoptr i64 %567 to ptr addrspace(3), !dbg !18
  %address.local410 = getelementptr [2 x i8], ptr addrspace(1) %565, i64 %address.local292, !dbg !18
  %569 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %570 = ptrtoint ptr addrspace(1) %address.local410 to i64, !dbg !18
  %571 = sub i64 %570, %569, !dbg !18
  %dma.buffer.offset1764 = trunc i64 %571 to i32, !dbg !18
  %572 = trunc i64 %569 to i32, !dbg !18
  %573 = insertelement <4 x i32> poison, i32 %572, i64 0, !dbg !18
  %574 = lshr i64 %569, 32, !dbg !18
  %575 = trunc i64 %574 to i32, !dbg !18
  %576 = insertelement <4 x i32> %573, i32 %575, i64 1, !dbg !18
  %577 = insertelement <4 x i32> %576, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1765 = insertelement <4 x i32> %577, i32 131072, i64 3, !dbg !18
  %578 = ptrtoint ptr addrspace(3) %568 to i32, !dbg !18
  %dma.buffer.lds1766 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %578), !dbg !18
  %579 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1766, i32 %dma.buffer.offset1764, <4 x i32> %dma.buffer.rsrc1765) #10, !dbg !18
  %580 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local296), !dbg !18
  %581 = zext i32 %580 to i64, !dbg !18
  %582 = inttoptr i64 %581 to ptr addrspace(3), !dbg !18
  %address.local411 = getelementptr [2 x i8], ptr addrspace(1) %565, i64 %address.local306, !dbg !18
  %583 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %584 = ptrtoint ptr addrspace(1) %address.local411 to i64, !dbg !18
  %585 = sub i64 %584, %583, !dbg !18
  %dma.buffer.offset1761 = trunc i64 %585 to i32, !dbg !18
  %586 = trunc i64 %583 to i32, !dbg !18
  %587 = insertelement <4 x i32> poison, i32 %586, i64 0, !dbg !18
  %588 = lshr i64 %583, 32, !dbg !18
  %589 = trunc i64 %588 to i32, !dbg !18
  %590 = insertelement <4 x i32> %587, i32 %589, i64 1, !dbg !18
  %591 = insertelement <4 x i32> %590, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1762 = insertelement <4 x i32> %591, i32 131072, i64 3, !dbg !18
  %592 = ptrtoint ptr addrspace(3) %582 to i32, !dbg !18
  %dma.buffer.lds1763 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %592), !dbg !18
  %593 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1763, i32 %dma.buffer.offset1761, <4 x i32> %dma.buffer.rsrc1762) #10, !dbg !18
  %594 = load <4 x half>, ptr addrspace(3) %address.local320, align 8, !dbg !12
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local320, i64 4096, !dbg !12
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !12
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %address.local320, i64 1024, !dbg !12
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !12
  %597 = fmul <4 x half> %594, splat (half 0xH3015), !dbg !12
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %address.local320, i64 5120, !dbg !12
  %598 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !12
  %599 = fmul <4 x half> %595, splat (half 0xH3015), !dbg !12
  %600 = fmul <4 x half> %596, splat (half 0xH3015), !dbg !12
  %601 = fmul <4 x half> %598, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %602 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !19
  %603 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %602, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4096, !dbg !19
  %604 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !19
  %605 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !19
  %606 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %604, <4 x float> %603) #9, !dbg !19
  %607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %605, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4096, !dbg !19
  %608 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !19
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %address.local336, i64 512, !dbg !19
  %609 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !19
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %608, <4 x float> %607) #9, !dbg !19
  %611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %609, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4608, !dbg !19
  %612 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !19
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local361, i64 512, !dbg !19
  %613 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !19
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %612, <4 x float> %611) #9, !dbg !19
  %615 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %613, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !19
  %616 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !19
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1024, !dbg !19
  %617 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !19
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %616, <4 x float> %615) #9, !dbg !19
  %619 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %617, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5120, !dbg !19
  %620 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !19
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1024, !dbg !19
  %621 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !19
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %620, <4 x float> %619) #9, !dbg !19
  %623 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %621, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5120, !dbg !19
  %624 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !19
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1536, !dbg !19
  %625 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !19
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %624, <4 x float> %623) #9, !dbg !19
  %627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %625, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5632, !dbg !19
  %628 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !19
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1536, !dbg !19
  %629 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !19
  %630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %628, <4 x float> %627) #9, !dbg !19
  %631 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %629, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5632, !dbg !19
  %632 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !19
  %633 = load <4 x half>, ptr addrspace(3) %address.local336, align 8, !dbg !19
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %632, <4 x float> %631) #9, !dbg !19
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %633, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4096, !dbg !19
  %636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !19
  %637 = load <4 x half>, ptr addrspace(3) %address.local361, align 8, !dbg !19
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %636, <4 x float> %635) #9, !dbg !19
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %637, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4096, !dbg !19
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !19
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %address.local336, i64 512, !dbg !19
  %641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !19
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %640, <4 x float> %639) #9, !dbg !19
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %641, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %address.local336, i64 4608, !dbg !19
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !19
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local361, i64 512, !dbg !19
  %645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !19
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %644, <4 x float> %643) #9, !dbg !19
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %645, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !19
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !19
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1024, !dbg !19
  %649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !19
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %648, <4 x float> %647) #9, !dbg !19
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %649, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5120, !dbg !19
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !19
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1024, !dbg !19
  %653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !19
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %652, <4 x float> %651) #9, !dbg !19
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %653, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5120, !dbg !19
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !19
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %address.local336, i64 1536, !dbg !19
  %657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !19
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %656, <4 x float> %655) #9, !dbg !19
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %657, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %address.local336, i64 5632, !dbg !19
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !19
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local361, i64 1536, !dbg !19
  %661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !19
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %601, <4 x half> %660, <4 x float> %659) #9, !dbg !19
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %661, <4 x float> zeroinitializer) #9, !dbg !19
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %address.local361, i64 5632, !dbg !19
  %664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !19
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
  %address.thread412 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local413 = shl nuw nsw i32 %address.thread412, 5, !dbg !13
  %address.local414 = and i32 %address.local413, 6144, !dbg !13
  %address.local415 = shl nuw nsw i32 %address.thread412, 7, !dbg !13
  %address.local416 = and i32 %address.local415, 1920, !dbg !13
  %address.local417 = or disjoint i32 %address.local414, %address.local416, !dbg !13
  %address.local418 = zext nneg i32 %address.local417 to i64, !dbg !13
  %address.local419 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local420 = zext nneg i32 %address.thread412 to i64, !dbg !12
  %address.local421 = lshr i64 %address.local420, 4, !dbg !12
  %address.local422 = and i64 %address.local421, 3, !dbg !12
  %address.local423 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local419, i64 %address.local422, !dbg !13
  store float %794, ptr addrspace(3) %address.local423, align 4, !dbg !13
  %address.local424 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local425 = or disjoint i64 %address.local422, 4, !dbg !13
  %address.local426 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local424, i64 %address.local425, !dbg !13
  store float %793, ptr addrspace(3) %address.local426, align 4, !dbg !13
  %address.local427 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local428 = or disjoint i64 %address.local422, 8, !dbg !13
  %address.local429 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local427, i64 %address.local428, !dbg !13
  store float %792, ptr addrspace(3) %address.local429, align 4, !dbg !13
  %address.local430 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local431 = or i64 %address.local421, 12, !dbg !13
  %address.local432 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local430, i64 %address.local431, !dbg !13
  store float %791, ptr addrspace(3) %address.local432, align 4, !dbg !13
  %address.local433 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local434 = or disjoint i64 %address.local422, 16, !dbg !13
  %address.local435 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local433, i64 %address.local434, !dbg !13
  store float %786, ptr addrspace(3) %address.local435, align 4, !dbg !13
  %address.local436 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local437 = or disjoint i64 %address.local422, 20, !dbg !13
  %address.local438 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local436, i64 %address.local437, !dbg !13
  store float %785, ptr addrspace(3) %address.local438, align 4, !dbg !13
  %address.local439 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local440 = or disjoint i64 %address.local422, 24, !dbg !13
  %address.local441 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local439, i64 %address.local440, !dbg !13
  store float %784, ptr addrspace(3) %address.local441, align 4, !dbg !13
  %address.local442 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local443 = or i64 %address.local421, 28, !dbg !13
  %address.local444 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local442, i64 %address.local443, !dbg !13
  store float %783, ptr addrspace(3) %address.local444, align 4, !dbg !13
  %address.local445 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local446 = or disjoint i64 %address.local422, 32, !dbg !13
  %address.local447 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local445, i64 %address.local446, !dbg !13
  store float %778, ptr addrspace(3) %address.local447, align 4, !dbg !13
  %address.local448 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local449 = or disjoint i64 %address.local422, 36, !dbg !13
  %address.local450 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local448, i64 %address.local449, !dbg !13
  store float %777, ptr addrspace(3) %address.local450, align 4, !dbg !13
  %address.local451 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local452 = or disjoint i64 %address.local422, 40, !dbg !13
  %address.local453 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local451, i64 %address.local452, !dbg !13
  store float %776, ptr addrspace(3) %address.local453, align 4, !dbg !13
  %address.local454 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local455 = or i64 %address.local421, 44, !dbg !13
  %address.local456 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local454, i64 %address.local455, !dbg !13
  store float %775, ptr addrspace(3) %address.local456, align 4, !dbg !13
  %address.local457 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local458 = or disjoint i64 %address.local422, 48, !dbg !13
  %address.local459 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local457, i64 %address.local458, !dbg !13
  store float %770, ptr addrspace(3) %address.local459, align 4, !dbg !13
  %address.local460 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local461 = or disjoint i64 %address.local422, 52, !dbg !13
  %address.local462 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local460, i64 %address.local461, !dbg !13
  store float %769, ptr addrspace(3) %address.local462, align 4, !dbg !13
  %address.local463 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local464 = or disjoint i64 %address.local422, 56, !dbg !13
  %address.local465 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local463, i64 %address.local464, !dbg !13
  store float %768, ptr addrspace(3) %address.local465, align 4, !dbg !13
  %address.local466 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local467 = or i64 %address.local421, 60, !dbg !13
  %address.local468 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local466, i64 %address.local467, !dbg !13
  store float %767, ptr addrspace(3) %address.local468, align 4, !dbg !13
  %address.local469 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local470 = or disjoint i64 %address.local422, 64, !dbg !13
  %address.local471 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local469, i64 %address.local470, !dbg !13
  store float %762, ptr addrspace(3) %address.local471, align 4, !dbg !13
  %address.local472 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local473 = or disjoint i64 %address.local422, 68, !dbg !13
  %address.local474 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local472, i64 %address.local473, !dbg !13
  store float %761, ptr addrspace(3) %address.local474, align 4, !dbg !13
  %address.local475 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local476 = or disjoint i64 %address.local422, 72, !dbg !13
  %address.local477 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local475, i64 %address.local476, !dbg !13
  store float %760, ptr addrspace(3) %address.local477, align 4, !dbg !13
  %address.local478 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local479 = or i64 %address.local421, 76, !dbg !13
  %address.local480 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local478, i64 %address.local479, !dbg !13
  store float %759, ptr addrspace(3) %address.local480, align 4, !dbg !13
  %address.local481 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local482 = or disjoint i64 %address.local422, 80, !dbg !13
  %address.local483 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local481, i64 %address.local482, !dbg !13
  store float %754, ptr addrspace(3) %address.local483, align 4, !dbg !13
  %address.local484 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local485 = or disjoint i64 %address.local422, 84, !dbg !13
  %address.local486 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local484, i64 %address.local485, !dbg !13
  store float %753, ptr addrspace(3) %address.local486, align 4, !dbg !13
  %address.local487 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local488 = or disjoint i64 %address.local422, 88, !dbg !13
  %address.local489 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local487, i64 %address.local488, !dbg !13
  store float %752, ptr addrspace(3) %address.local489, align 4, !dbg !13
  %address.local490 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local491 = or i64 %address.local421, 92, !dbg !13
  %address.local492 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local490, i64 %address.local491, !dbg !13
  store float %751, ptr addrspace(3) %address.local492, align 4, !dbg !13
  %address.local493 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local494 = or disjoint i64 %address.local422, 96, !dbg !13
  %address.local495 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local493, i64 %address.local494, !dbg !13
  store float %746, ptr addrspace(3) %address.local495, align 4, !dbg !13
  %address.local496 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local497 = or disjoint i64 %address.local422, 100, !dbg !13
  %address.local498 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local496, i64 %address.local497, !dbg !13
  store float %745, ptr addrspace(3) %address.local498, align 4, !dbg !13
  %address.local499 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local500 = or disjoint i64 %address.local422, 104, !dbg !13
  %address.local501 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local499, i64 %address.local500, !dbg !13
  store float %744, ptr addrspace(3) %address.local501, align 4, !dbg !13
  %address.local502 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local503 = or i64 %address.local421, 108, !dbg !13
  %address.local504 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local502, i64 %address.local503, !dbg !13
  store float %743, ptr addrspace(3) %address.local504, align 4, !dbg !13
  %address.local505 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local506 = or disjoint i64 %address.local422, 112, !dbg !13
  %address.local507 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local505, i64 %address.local506, !dbg !13
  store float %738, ptr addrspace(3) %address.local507, align 4, !dbg !13
  %address.local508 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local509 = or disjoint i64 %address.local422, 116, !dbg !13
  %address.local510 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local508, i64 %address.local509, !dbg !13
  store float %737, ptr addrspace(3) %address.local510, align 4, !dbg !13
  %address.local511 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local512 = or disjoint i64 %address.local422, 120, !dbg !13
  %address.local513 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local511, i64 %address.local512, !dbg !13
  store float %736, ptr addrspace(3) %address.local513, align 4, !dbg !13
  %address.local514 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local515 = or i64 %address.local421, 124, !dbg !13
  %address.local516 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local514, i64 %address.local515, !dbg !13
  store float %735, ptr addrspace(3) %address.local516, align 4, !dbg !13
  %address.local517 = shl nuw nsw i32 %address.thread412, 5, !dbg !13
  %address.local518 = and i32 %address.local517, 6144, !dbg !13
  %address.local519 = shl nuw nsw i32 %address.thread412, 7, !dbg !13
  %address.local520 = and i32 %address.local519, 1920, !dbg !13
  %address.local521 = or disjoint i32 %address.local518, %address.local520, !dbg !13
  %address.local522 = or disjoint i32 %address.local521, 8192, !dbg !13
  %address.local523 = zext nneg i32 %address.local522 to i64, !dbg !13
  %address.local524 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local525 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local524, i64 %address.local422, !dbg !13
  store float %730, ptr addrspace(3) %address.local525, align 4, !dbg !13
  %address.local526 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local527 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local526, i64 %address.local425, !dbg !13
  store float %729, ptr addrspace(3) %address.local527, align 4, !dbg !13
  %address.local528 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local529 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local528, i64 %address.local428, !dbg !13
  store float %728, ptr addrspace(3) %address.local529, align 4, !dbg !13
  %address.local530 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local531 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local530, i64 %address.local431, !dbg !13
  store float %727, ptr addrspace(3) %address.local531, align 4, !dbg !13
  %address.local532 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local533 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local532, i64 %address.local434, !dbg !13
  store float %722, ptr addrspace(3) %address.local533, align 4, !dbg !13
  %address.local534 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local535 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local534, i64 %address.local437, !dbg !13
  store float %721, ptr addrspace(3) %address.local535, align 4, !dbg !13
  %address.local536 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local537 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local536, i64 %address.local440, !dbg !13
  store float %720, ptr addrspace(3) %address.local537, align 4, !dbg !13
  %address.local538 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local539 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local538, i64 %address.local443, !dbg !13
  store float %719, ptr addrspace(3) %address.local539, align 4, !dbg !13
  %address.local540 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local541 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local540, i64 %address.local446, !dbg !13
  store float %714, ptr addrspace(3) %address.local541, align 4, !dbg !13
  %address.local542 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local543 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local542, i64 %address.local449, !dbg !13
  store float %713, ptr addrspace(3) %address.local543, align 4, !dbg !13
  %address.local544 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local545 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local544, i64 %address.local452, !dbg !13
  store float %712, ptr addrspace(3) %address.local545, align 4, !dbg !13
  %address.local546 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local547 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local546, i64 %address.local455, !dbg !13
  store float %711, ptr addrspace(3) %address.local547, align 4, !dbg !13
  %address.local548 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local549 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local548, i64 %address.local458, !dbg !13
  store float %706, ptr addrspace(3) %address.local549, align 4, !dbg !13
  %address.local550 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local551 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local550, i64 %address.local461, !dbg !13
  store float %705, ptr addrspace(3) %address.local551, align 4, !dbg !13
  %address.local552 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local553 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local552, i64 %address.local464, !dbg !13
  store float %704, ptr addrspace(3) %address.local553, align 4, !dbg !13
  %address.local554 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local555 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local554, i64 %address.local467, !dbg !13
  store float %703, ptr addrspace(3) %address.local555, align 4, !dbg !13
  %address.local556 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local557 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local556, i64 %address.local470, !dbg !13
  store float %698, ptr addrspace(3) %address.local557, align 4, !dbg !13
  %address.local558 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local559 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local558, i64 %address.local473, !dbg !13
  store float %697, ptr addrspace(3) %address.local559, align 4, !dbg !13
  %address.local560 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local561 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local560, i64 %address.local476, !dbg !13
  store float %696, ptr addrspace(3) %address.local561, align 4, !dbg !13
  %address.local562 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local563 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local562, i64 %address.local479, !dbg !13
  store float %695, ptr addrspace(3) %address.local563, align 4, !dbg !13
  %address.local564 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local565 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local564, i64 %address.local482, !dbg !13
  store float %690, ptr addrspace(3) %address.local565, align 4, !dbg !13
  %address.local566 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local567 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local566, i64 %address.local485, !dbg !13
  store float %689, ptr addrspace(3) %address.local567, align 4, !dbg !13
  %address.local568 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local569 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local568, i64 %address.local488, !dbg !13
  store float %688, ptr addrspace(3) %address.local569, align 4, !dbg !13
  %address.local570 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local571 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local570, i64 %address.local491, !dbg !13
  store float %687, ptr addrspace(3) %address.local571, align 4, !dbg !13
  %address.local572 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local573 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local572, i64 %address.local494, !dbg !13
  store float %682, ptr addrspace(3) %address.local573, align 4, !dbg !13
  %address.local574 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local575 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local574, i64 %address.local497, !dbg !13
  store float %681, ptr addrspace(3) %address.local575, align 4, !dbg !13
  %address.local576 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local577 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local576, i64 %address.local500, !dbg !13
  store float %680, ptr addrspace(3) %address.local577, align 4, !dbg !13
  %address.local578 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local579 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local578, i64 %address.local503, !dbg !13
  store float %679, ptr addrspace(3) %address.local579, align 4, !dbg !13
  %address.local580 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local581 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local580, i64 %address.local506, !dbg !13
  store float %674, ptr addrspace(3) %address.local581, align 4, !dbg !13
  %address.local582 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local583 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local582, i64 %address.local509, !dbg !13
  store float %673, ptr addrspace(3) %address.local583, align 4, !dbg !13
  %address.local584 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local585 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local584, i64 %address.local512, !dbg !13
  store float %672, ptr addrspace(3) %address.local585, align 4, !dbg !13
  %address.local586 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !13
  %address.local587 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local586, i64 %address.local515, !dbg !13
  store float %671, ptr addrspace(3) %address.local587, align 4, !dbg !13
  br label %795, !dbg !13

795:                                              ; preds = %666, %70
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.thread588 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local589 = shl nuw nsw i32 %address.thread588, 5, !dbg !13
  %address.local590 = and i32 %address.local589, 6144, !dbg !13
  %address.local591 = shl nuw nsw i32 %address.thread588, 7, !dbg !13
  %address.local592 = and i32 %address.local591, 1920, !dbg !13
  %address.local593 = or disjoint i32 %address.local590, %address.local592, !dbg !13
  %address.local594 = zext nneg i32 %address.local593 to i64, !dbg !13
  %address.local595 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local594, !dbg !13
  %address.local596 = zext nneg i32 %address.thread588 to i64, !dbg !12
  %address.local597 = lshr i64 %address.local596, 4, !dbg !12
  %address.local598 = and i64 %address.local597, 3, !dbg !12
  %address.local599 = shl nuw nsw i64 %address.local598, 2, !dbg !12
  %address.local600 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local595, i64 %address.local599, !dbg !13
  %796 = load float, ptr addrspace(3) %address.local600, align 16, !dbg !13
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local600, i64 4, !dbg !13
  %797 = load float, ptr addrspace(3) %lds.const.gep1668, align 4, !dbg !13
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local600, i64 8, !dbg !13
  %798 = load float, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !13
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %address.local600, i64 12, !dbg !13
  %799 = load float, ptr addrspace(3) %lds.const.gep1670, align 4, !dbg !13
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local600, i64 64, !dbg !13
  %800 = load float, ptr addrspace(3) %lds.const.gep1671, align 16, !dbg !13
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local600, i64 68, !dbg !13
  %801 = load float, ptr addrspace(3) %lds.const.gep1672, align 4, !dbg !13
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local600, i64 72, !dbg !13
  %802 = load float, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !13
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %address.local600, i64 76, !dbg !13
  %803 = load float, ptr addrspace(3) %lds.const.gep1674, align 4, !dbg !13
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local600, i64 128, !dbg !13
  %804 = load float, ptr addrspace(3) %lds.const.gep1675, align 16, !dbg !13
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local600, i64 132, !dbg !13
  %805 = load float, ptr addrspace(3) %lds.const.gep1676, align 4, !dbg !13
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local600, i64 136, !dbg !13
  %806 = load float, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !13
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %address.local600, i64 140, !dbg !13
  %807 = load float, ptr addrspace(3) %lds.const.gep1678, align 4, !dbg !13
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local600, i64 192, !dbg !13
  %808 = load float, ptr addrspace(3) %lds.const.gep1679, align 16, !dbg !13
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local600, i64 196, !dbg !13
  %809 = load float, ptr addrspace(3) %lds.const.gep1680, align 4, !dbg !13
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local600, i64 200, !dbg !13
  %810 = load float, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !13
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local600, i64 204, !dbg !13
  %811 = load float, ptr addrspace(3) %lds.const.gep1682, align 4, !dbg !13
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local600, i64 256, !dbg !13
  %812 = load float, ptr addrspace(3) %lds.const.gep1683, align 16, !dbg !13
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local600, i64 260, !dbg !13
  %813 = load float, ptr addrspace(3) %lds.const.gep1684, align 4, !dbg !13
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local600, i64 264, !dbg !13
  %814 = load float, ptr addrspace(3) %lds.const.gep1685, align 8, !dbg !13
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local600, i64 268, !dbg !13
  %815 = load float, ptr addrspace(3) %lds.const.gep1686, align 4, !dbg !13
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local600, i64 320, !dbg !13
  %816 = load float, ptr addrspace(3) %lds.const.gep1687, align 16, !dbg !13
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local600, i64 324, !dbg !13
  %817 = load float, ptr addrspace(3) %lds.const.gep1688, align 4, !dbg !13
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local600, i64 328, !dbg !13
  %818 = load float, ptr addrspace(3) %lds.const.gep1689, align 8, !dbg !13
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local600, i64 332, !dbg !13
  %819 = load float, ptr addrspace(3) %lds.const.gep1690, align 4, !dbg !13
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local600, i64 384, !dbg !13
  %820 = load float, ptr addrspace(3) %lds.const.gep1691, align 16, !dbg !13
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local600, i64 388, !dbg !13
  %821 = load float, ptr addrspace(3) %lds.const.gep1692, align 4, !dbg !13
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local600, i64 392, !dbg !13
  %822 = load float, ptr addrspace(3) %lds.const.gep1693, align 8, !dbg !13
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local600, i64 396, !dbg !13
  %823 = load float, ptr addrspace(3) %lds.const.gep1694, align 4, !dbg !13
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local600, i64 448, !dbg !13
  %824 = load float, ptr addrspace(3) %lds.const.gep1695, align 16, !dbg !13
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local600, i64 452, !dbg !13
  %825 = load float, ptr addrspace(3) %lds.const.gep1696, align 4, !dbg !13
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local600, i64 456, !dbg !13
  %826 = load float, ptr addrspace(3) %lds.const.gep1697, align 8, !dbg !13
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local600, i64 460, !dbg !13
  %827 = load float, ptr addrspace(3) %lds.const.gep1698, align 4, !dbg !13
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32768, !dbg !13
  %828 = load float, ptr addrspace(3) %lds.const.gep1699, align 16, !dbg !13
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32772, !dbg !13
  %829 = load float, ptr addrspace(3) %lds.const.gep1700, align 4, !dbg !13
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32776, !dbg !13
  %830 = load float, ptr addrspace(3) %lds.const.gep1701, align 8, !dbg !13
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32780, !dbg !13
  %831 = load float, ptr addrspace(3) %lds.const.gep1702, align 4, !dbg !13
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32832, !dbg !13
  %832 = load float, ptr addrspace(3) %lds.const.gep1703, align 16, !dbg !13
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32836, !dbg !13
  %833 = load float, ptr addrspace(3) %lds.const.gep1704, align 4, !dbg !13
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32840, !dbg !13
  %834 = load float, ptr addrspace(3) %lds.const.gep1705, align 8, !dbg !13
  %lds.const.gep1706 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32844, !dbg !13
  %835 = load float, ptr addrspace(3) %lds.const.gep1706, align 4, !dbg !13
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32896, !dbg !13
  %836 = load float, ptr addrspace(3) %lds.const.gep1707, align 16, !dbg !13
  %lds.const.gep1708 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32900, !dbg !13
  %837 = load float, ptr addrspace(3) %lds.const.gep1708, align 4, !dbg !13
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32904, !dbg !13
  %838 = load float, ptr addrspace(3) %lds.const.gep1709, align 8, !dbg !13
  %lds.const.gep1710 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32908, !dbg !13
  %839 = load float, ptr addrspace(3) %lds.const.gep1710, align 4, !dbg !13
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32960, !dbg !13
  %840 = load float, ptr addrspace(3) %lds.const.gep1711, align 16, !dbg !13
  %lds.const.gep1712 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32964, !dbg !13
  %841 = load float, ptr addrspace(3) %lds.const.gep1712, align 4, !dbg !13
  %lds.const.gep1713 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32968, !dbg !13
  %842 = load float, ptr addrspace(3) %lds.const.gep1713, align 8, !dbg !13
  %lds.const.gep1714 = getelementptr i8, ptr addrspace(3) %address.local600, i64 32972, !dbg !13
  %843 = load float, ptr addrspace(3) %lds.const.gep1714, align 4, !dbg !13
  %lds.const.gep1715 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33024, !dbg !13
  %844 = load float, ptr addrspace(3) %lds.const.gep1715, align 16, !dbg !13
  %lds.const.gep1716 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33028, !dbg !13
  %845 = load float, ptr addrspace(3) %lds.const.gep1716, align 4, !dbg !13
  %lds.const.gep1717 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33032, !dbg !13
  %846 = load float, ptr addrspace(3) %lds.const.gep1717, align 8, !dbg !13
  %lds.const.gep1718 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33036, !dbg !13
  %847 = load float, ptr addrspace(3) %lds.const.gep1718, align 4, !dbg !13
  %lds.const.gep1719 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33088, !dbg !13
  %848 = load float, ptr addrspace(3) %lds.const.gep1719, align 16, !dbg !13
  %lds.const.gep1720 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33092, !dbg !13
  %849 = load float, ptr addrspace(3) %lds.const.gep1720, align 4, !dbg !13
  %lds.const.gep1721 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33096, !dbg !13
  %850 = load float, ptr addrspace(3) %lds.const.gep1721, align 8, !dbg !13
  %lds.const.gep1722 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33100, !dbg !13
  %851 = load float, ptr addrspace(3) %lds.const.gep1722, align 4, !dbg !13
  %lds.const.gep1723 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33152, !dbg !13
  %852 = load float, ptr addrspace(3) %lds.const.gep1723, align 16, !dbg !13
  %lds.const.gep1724 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33156, !dbg !13
  %853 = load float, ptr addrspace(3) %lds.const.gep1724, align 4, !dbg !13
  %lds.const.gep1725 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33160, !dbg !13
  %854 = load float, ptr addrspace(3) %lds.const.gep1725, align 8, !dbg !13
  %lds.const.gep1726 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33164, !dbg !13
  %855 = load float, ptr addrspace(3) %lds.const.gep1726, align 4, !dbg !13
  %lds.const.gep1727 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33216, !dbg !13
  %856 = load float, ptr addrspace(3) %lds.const.gep1727, align 16, !dbg !13
  %lds.const.gep1728 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33220, !dbg !13
  %857 = load float, ptr addrspace(3) %lds.const.gep1728, align 4, !dbg !13
  %lds.const.gep1729 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33224, !dbg !13
  %858 = load float, ptr addrspace(3) %lds.const.gep1729, align 8, !dbg !13
  %lds.const.gep1730 = getelementptr i8, ptr addrspace(3) %address.local600, i64 33228, !dbg !13
  %859 = load float, ptr addrspace(3) %lds.const.gep1730, align 4, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %860 = or disjoint i64 %71, 127, !dbg !14
  %861 = icmp uge i64 %19, %860, !dbg !14
  %862 = and i1 %43, %861, !dbg !14
  br i1 %862, label %863, label %864, !dbg !14

863:                                              ; preds = %795
  br label %1005, !dbg !14

864:                                              ; preds = %795
  %865 = add i64 %19, %35, !dbg !14
  %866 = add i64 %865, %59, !dbg !14
  %867 = add i64 %866, %60, !dbg !14
  %868 = or disjoint i64 %71, %39, !dbg !14
  %869 = add i64 %867, 1, !dbg !25
  %870 = lshr i64 %869, 32, !dbg !26
  %compare.high = trunc i64 %870 to i32, !dbg !26
  %871 = lshr i64 %868, 32, !dbg !26
  %compare.high1289 = trunc i64 %871 to i32, !dbg !26
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1289, !dbg !26
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1289, !dbg !26
  %compare.low.base = trunc i64 %869 to i32, !dbg !26
  %compare.low.base1290 = trunc i64 %868 to i32, !dbg !26
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1290, !dbg !26
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !26
  %872 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 1) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !26
  %compare.low1291 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !26
  %compare.result1292 = select i1 %compare.high.equal, i1 %compare.low1291, i1 %compare.high.order, !dbg !26
  %873 = select i1 %compare.result1292, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 2) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1294 = or i32 %compare.local.base1293, 2, !dbg !26
  %compare.low1295 = icmp ugt i32 %compare.low.base, %compare.local1294, !dbg !26
  %compare.result1296 = select i1 %compare.high.equal, i1 %compare.low1295, i1 %compare.high.order, !dbg !26
  %874 = select i1 %compare.result1296, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 3) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1298 = or i32 %compare.local.base1297, 3, !dbg !26
  %compare.low1299 = icmp ugt i32 %compare.low.base, %compare.local1298, !dbg !26
  %compare.result1300 = select i1 %compare.high.equal, i1 %compare.low1299, i1 %compare.high.order, !dbg !26
  %875 = select i1 %compare.result1300, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 4) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1302 = or i32 %compare.local.base1301, 16, !dbg !26
  %compare.low1303 = icmp ugt i32 %compare.low.base, %compare.local1302, !dbg !26
  %compare.result1304 = select i1 %compare.high.equal, i1 %compare.low1303, i1 %compare.high.order, !dbg !26
  %876 = select i1 %compare.result1304, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 5) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1306 = or i32 %compare.local.base1305, 17, !dbg !26
  %compare.low1307 = icmp ugt i32 %compare.low.base, %compare.local1306, !dbg !26
  %compare.result1308 = select i1 %compare.high.equal, i1 %compare.low1307, i1 %compare.high.order, !dbg !26
  %877 = select i1 %compare.result1308, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 6) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1310 = or i32 %compare.local.base1309, 18, !dbg !26
  %compare.low1311 = icmp ugt i32 %compare.low.base, %compare.local1310, !dbg !26
  %compare.result1312 = select i1 %compare.high.equal, i1 %compare.low1311, i1 %compare.high.order, !dbg !26
  %878 = select i1 %compare.result1312, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 7) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1314 = or i32 %compare.local.base1313, 19, !dbg !26
  %compare.low1315 = icmp ugt i32 %compare.low.base, %compare.local1314, !dbg !26
  %compare.result1316 = select i1 %compare.high.equal, i1 %compare.low1315, i1 %compare.high.order, !dbg !26
  %879 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 8) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 32, !dbg !26
  %compare.low1319 = icmp ugt i32 %compare.low.base, %compare.local1318, !dbg !26
  %compare.result1320 = select i1 %compare.high.equal, i1 %compare.low1319, i1 %compare.high.order, !dbg !26
  %880 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 9) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 33, !dbg !26
  %compare.low1323 = icmp ugt i32 %compare.low.base, %compare.local1322, !dbg !26
  %compare.result1324 = select i1 %compare.high.equal, i1 %compare.low1323, i1 %compare.high.order, !dbg !26
  %881 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 10) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 34, !dbg !26
  %compare.low1327 = icmp ugt i32 %compare.low.base, %compare.local1326, !dbg !26
  %compare.result1328 = select i1 %compare.high.equal, i1 %compare.low1327, i1 %compare.high.order, !dbg !26
  %882 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 11) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 35, !dbg !26
  %compare.low1331 = icmp ugt i32 %compare.low.base, %compare.local1330, !dbg !26
  %compare.result1332 = select i1 %compare.high.equal, i1 %compare.low1331, i1 %compare.high.order, !dbg !26
  %883 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 12) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 48, !dbg !26
  %compare.low1335 = icmp ugt i32 %compare.low.base, %compare.local1334, !dbg !26
  %compare.result1336 = select i1 %compare.high.equal, i1 %compare.low1335, i1 %compare.high.order, !dbg !26
  %884 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 13) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 49, !dbg !26
  %compare.low1339 = icmp ugt i32 %compare.low.base, %compare.local1338, !dbg !26
  %compare.result1340 = select i1 %compare.high.equal, i1 %compare.low1339, i1 %compare.high.order, !dbg !26
  %885 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 14) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 50, !dbg !26
  %compare.low1343 = icmp ugt i32 %compare.low.base, %compare.local1342, !dbg !26
  %compare.result1344 = select i1 %compare.high.equal, i1 %compare.low1343, i1 %compare.high.order, !dbg !26
  %886 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 15) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 51, !dbg !26
  %compare.low1347 = icmp ugt i32 %compare.low.base, %compare.local1346, !dbg !26
  %compare.result1348 = select i1 %compare.high.equal, i1 %compare.low1347, i1 %compare.high.order, !dbg !26
  %887 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 16) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 64, !dbg !26
  %compare.low1351 = icmp ugt i32 %compare.low.base, %compare.local1350, !dbg !26
  %compare.result1352 = select i1 %compare.high.equal, i1 %compare.low1351, i1 %compare.high.order, !dbg !26
  %888 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 17) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 65, !dbg !26
  %compare.low1355 = icmp ugt i32 %compare.low.base, %compare.local1354, !dbg !26
  %compare.result1356 = select i1 %compare.high.equal, i1 %compare.low1355, i1 %compare.high.order, !dbg !26
  %889 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 18) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 66, !dbg !26
  %compare.low1359 = icmp ugt i32 %compare.low.base, %compare.local1358, !dbg !26
  %compare.result1360 = select i1 %compare.high.equal, i1 %compare.low1359, i1 %compare.high.order, !dbg !26
  %890 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 19) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 67, !dbg !26
  %compare.low1363 = icmp ugt i32 %compare.low.base, %compare.local1362, !dbg !26
  %compare.result1364 = select i1 %compare.high.equal, i1 %compare.low1363, i1 %compare.high.order, !dbg !26
  %891 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 20) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 80, !dbg !26
  %compare.low1367 = icmp ugt i32 %compare.low.base, %compare.local1366, !dbg !26
  %compare.result1368 = select i1 %compare.high.equal, i1 %compare.low1367, i1 %compare.high.order, !dbg !26
  %892 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 21) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 81, !dbg !26
  %compare.low1371 = icmp ugt i32 %compare.low.base, %compare.local1370, !dbg !26
  %compare.result1372 = select i1 %compare.high.equal, i1 %compare.low1371, i1 %compare.high.order, !dbg !26
  %893 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 22) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 82, !dbg !26
  %compare.low1375 = icmp ugt i32 %compare.low.base, %compare.local1374, !dbg !26
  %compare.result1376 = select i1 %compare.high.equal, i1 %compare.low1375, i1 %compare.high.order, !dbg !26
  %894 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 23) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 83, !dbg !26
  %compare.low1379 = icmp ugt i32 %compare.low.base, %compare.local1378, !dbg !26
  %compare.result1380 = select i1 %compare.high.equal, i1 %compare.low1379, i1 %compare.high.order, !dbg !26
  %895 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 24) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 96, !dbg !26
  %compare.low1383 = icmp ugt i32 %compare.low.base, %compare.local1382, !dbg !26
  %compare.result1384 = select i1 %compare.high.equal, i1 %compare.low1383, i1 %compare.high.order, !dbg !26
  %896 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 25) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 97, !dbg !26
  %compare.low1387 = icmp ugt i32 %compare.low.base, %compare.local1386, !dbg !26
  %compare.result1388 = select i1 %compare.high.equal, i1 %compare.low1387, i1 %compare.high.order, !dbg !26
  %897 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 26) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 98, !dbg !26
  %compare.low1391 = icmp ugt i32 %compare.low.base, %compare.local1390, !dbg !26
  %compare.result1392 = select i1 %compare.high.equal, i1 %compare.low1391, i1 %compare.high.order, !dbg !26
  %898 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 27) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1394 = or i32 %compare.local.base1393, 99, !dbg !26
  %compare.low1395 = icmp ugt i32 %compare.low.base, %compare.local1394, !dbg !26
  %compare.result1396 = select i1 %compare.high.equal, i1 %compare.low1395, i1 %compare.high.order, !dbg !26
  %899 = select i1 %compare.result1396, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 28) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1398 = or i32 %compare.local.base1397, 112, !dbg !26
  %compare.low1399 = icmp ugt i32 %compare.low.base, %compare.local1398, !dbg !26
  %compare.result1400 = select i1 %compare.high.equal, i1 %compare.low1399, i1 %compare.high.order, !dbg !26
  %900 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 29) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 113, !dbg !26
  %compare.low1403 = icmp ugt i32 %compare.low.base, %compare.local1402, !dbg !26
  %compare.result1404 = select i1 %compare.high.equal, i1 %compare.low1403, i1 %compare.high.order, !dbg !26
  %901 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 30) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 114, !dbg !26
  %compare.low1407 = icmp ugt i32 %compare.low.base, %compare.local1406, !dbg !26
  %compare.result1408 = select i1 %compare.high.equal, i1 %compare.low1407, i1 %compare.high.order, !dbg !26
  %902 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1290, i32 31) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 115, !dbg !26
  %compare.low1411 = icmp ugt i32 %compare.low.base, %compare.local1410, !dbg !26
  %compare.result1412 = select i1 %compare.high.equal, i1 %compare.low1411, i1 %compare.high.order, !dbg !26
  %903 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %904 = add i64 %867, 65, !dbg !25
  %905 = lshr i64 %904, 32, !dbg !26
  %compare.high1413 = trunc i64 %905 to i32, !dbg !26
  %906 = lshr i64 %868, 32, !dbg !26
  %compare.high1414 = trunc i64 %906 to i32, !dbg !26
  %compare.high.equal1415 = icmp eq i32 %compare.high1413, %compare.high1414, !dbg !26
  %compare.high.order1416 = icmp ugt i32 %compare.high1413, %compare.high1414, !dbg !26
  %compare.low.base1417 = trunc i64 %904 to i32, !dbg !26
  %compare.low.base1418 = trunc i64 %868 to i32, !dbg !26
  %compare.low1419 = icmp ugt i32 %compare.low.base1417, %compare.low.base1418, !dbg !26
  %compare.result1420 = select i1 %compare.high.equal1415, i1 %compare.low1419, i1 %compare.high.order1416, !dbg !26
  %907 = select i1 %compare.result1420, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1421 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 32) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1422 = or i32 %compare.local.base1421, 1, !dbg !26
  %compare.low1423 = icmp ugt i32 %compare.low.base1417, %compare.local1422, !dbg !26
  %compare.result1424 = select i1 %compare.high.equal1415, i1 %compare.low1423, i1 %compare.high.order1416, !dbg !26
  %908 = select i1 %compare.result1424, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1425 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 33) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1426 = or i32 %compare.local.base1425, 2, !dbg !26
  %compare.low1427 = icmp ugt i32 %compare.low.base1417, %compare.local1426, !dbg !26
  %compare.result1428 = select i1 %compare.high.equal1415, i1 %compare.low1427, i1 %compare.high.order1416, !dbg !26
  %909 = select i1 %compare.result1428, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1429 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 34) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1430 = or i32 %compare.local.base1429, 3, !dbg !26
  %compare.low1431 = icmp ugt i32 %compare.low.base1417, %compare.local1430, !dbg !26
  %compare.result1432 = select i1 %compare.high.equal1415, i1 %compare.low1431, i1 %compare.high.order1416, !dbg !26
  %910 = select i1 %compare.result1432, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1433 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 35) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1434 = or i32 %compare.local.base1433, 16, !dbg !26
  %compare.low1435 = icmp ugt i32 %compare.low.base1417, %compare.local1434, !dbg !26
  %compare.result1436 = select i1 %compare.high.equal1415, i1 %compare.low1435, i1 %compare.high.order1416, !dbg !26
  %911 = select i1 %compare.result1436, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1437 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 36) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1438 = or i32 %compare.local.base1437, 17, !dbg !26
  %compare.low1439 = icmp ugt i32 %compare.low.base1417, %compare.local1438, !dbg !26
  %compare.result1440 = select i1 %compare.high.equal1415, i1 %compare.low1439, i1 %compare.high.order1416, !dbg !26
  %912 = select i1 %compare.result1440, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1441 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 37) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1442 = or i32 %compare.local.base1441, 18, !dbg !26
  %compare.low1443 = icmp ugt i32 %compare.low.base1417, %compare.local1442, !dbg !26
  %compare.result1444 = select i1 %compare.high.equal1415, i1 %compare.low1443, i1 %compare.high.order1416, !dbg !26
  %913 = select i1 %compare.result1444, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1445 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 38) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1446 = or i32 %compare.local.base1445, 19, !dbg !26
  %compare.low1447 = icmp ugt i32 %compare.low.base1417, %compare.local1446, !dbg !26
  %compare.result1448 = select i1 %compare.high.equal1415, i1 %compare.low1447, i1 %compare.high.order1416, !dbg !26
  %914 = select i1 %compare.result1448, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1449 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 39) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1450 = or i32 %compare.local.base1449, 32, !dbg !26
  %compare.low1451 = icmp ugt i32 %compare.low.base1417, %compare.local1450, !dbg !26
  %compare.result1452 = select i1 %compare.high.equal1415, i1 %compare.low1451, i1 %compare.high.order1416, !dbg !26
  %915 = select i1 %compare.result1452, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1453 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 40) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1454 = or i32 %compare.local.base1453, 33, !dbg !26
  %compare.low1455 = icmp ugt i32 %compare.low.base1417, %compare.local1454, !dbg !26
  %compare.result1456 = select i1 %compare.high.equal1415, i1 %compare.low1455, i1 %compare.high.order1416, !dbg !26
  %916 = select i1 %compare.result1456, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1457 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 41) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1458 = or i32 %compare.local.base1457, 34, !dbg !26
  %compare.low1459 = icmp ugt i32 %compare.low.base1417, %compare.local1458, !dbg !26
  %compare.result1460 = select i1 %compare.high.equal1415, i1 %compare.low1459, i1 %compare.high.order1416, !dbg !26
  %917 = select i1 %compare.result1460, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1461 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 42) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1462 = or i32 %compare.local.base1461, 35, !dbg !26
  %compare.low1463 = icmp ugt i32 %compare.low.base1417, %compare.local1462, !dbg !26
  %compare.result1464 = select i1 %compare.high.equal1415, i1 %compare.low1463, i1 %compare.high.order1416, !dbg !26
  %918 = select i1 %compare.result1464, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1465 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 43) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1466 = or i32 %compare.local.base1465, 48, !dbg !26
  %compare.low1467 = icmp ugt i32 %compare.low.base1417, %compare.local1466, !dbg !26
  %compare.result1468 = select i1 %compare.high.equal1415, i1 %compare.low1467, i1 %compare.high.order1416, !dbg !26
  %919 = select i1 %compare.result1468, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1469 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 44) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1470 = or i32 %compare.local.base1469, 49, !dbg !26
  %compare.low1471 = icmp ugt i32 %compare.low.base1417, %compare.local1470, !dbg !26
  %compare.result1472 = select i1 %compare.high.equal1415, i1 %compare.low1471, i1 %compare.high.order1416, !dbg !26
  %920 = select i1 %compare.result1472, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1473 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 45) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1474 = or i32 %compare.local.base1473, 50, !dbg !26
  %compare.low1475 = icmp ugt i32 %compare.low.base1417, %compare.local1474, !dbg !26
  %compare.result1476 = select i1 %compare.high.equal1415, i1 %compare.low1475, i1 %compare.high.order1416, !dbg !26
  %921 = select i1 %compare.result1476, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1477 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 46) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1478 = or i32 %compare.local.base1477, 51, !dbg !26
  %compare.low1479 = icmp ugt i32 %compare.low.base1417, %compare.local1478, !dbg !26
  %compare.result1480 = select i1 %compare.high.equal1415, i1 %compare.low1479, i1 %compare.high.order1416, !dbg !26
  %922 = select i1 %compare.result1480, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1481 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 47) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1482 = or i32 %compare.local.base1481, 64, !dbg !26
  %compare.low1483 = icmp ugt i32 %compare.low.base1417, %compare.local1482, !dbg !26
  %compare.result1484 = select i1 %compare.high.equal1415, i1 %compare.low1483, i1 %compare.high.order1416, !dbg !26
  %923 = select i1 %compare.result1484, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1485 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 48) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1486 = or i32 %compare.local.base1485, 65, !dbg !26
  %compare.low1487 = icmp ugt i32 %compare.low.base1417, %compare.local1486, !dbg !26
  %compare.result1488 = select i1 %compare.high.equal1415, i1 %compare.low1487, i1 %compare.high.order1416, !dbg !26
  %924 = select i1 %compare.result1488, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1489 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 49) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1490 = or i32 %compare.local.base1489, 66, !dbg !26
  %compare.low1491 = icmp ugt i32 %compare.low.base1417, %compare.local1490, !dbg !26
  %compare.result1492 = select i1 %compare.high.equal1415, i1 %compare.low1491, i1 %compare.high.order1416, !dbg !26
  %925 = select i1 %compare.result1492, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1493 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 50) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1494 = or i32 %compare.local.base1493, 67, !dbg !26
  %compare.low1495 = icmp ugt i32 %compare.low.base1417, %compare.local1494, !dbg !26
  %compare.result1496 = select i1 %compare.high.equal1415, i1 %compare.low1495, i1 %compare.high.order1416, !dbg !26
  %926 = select i1 %compare.result1496, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1497 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 51) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1498 = or i32 %compare.local.base1497, 80, !dbg !26
  %compare.low1499 = icmp ugt i32 %compare.low.base1417, %compare.local1498, !dbg !26
  %compare.result1500 = select i1 %compare.high.equal1415, i1 %compare.low1499, i1 %compare.high.order1416, !dbg !26
  %927 = select i1 %compare.result1500, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1501 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 52) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1502 = or i32 %compare.local.base1501, 81, !dbg !26
  %compare.low1503 = icmp ugt i32 %compare.low.base1417, %compare.local1502, !dbg !26
  %compare.result1504 = select i1 %compare.high.equal1415, i1 %compare.low1503, i1 %compare.high.order1416, !dbg !26
  %928 = select i1 %compare.result1504, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1505 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 53) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1506 = or i32 %compare.local.base1505, 82, !dbg !26
  %compare.low1507 = icmp ugt i32 %compare.low.base1417, %compare.local1506, !dbg !26
  %compare.result1508 = select i1 %compare.high.equal1415, i1 %compare.low1507, i1 %compare.high.order1416, !dbg !26
  %929 = select i1 %compare.result1508, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1509 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 54) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1510 = or i32 %compare.local.base1509, 83, !dbg !26
  %compare.low1511 = icmp ugt i32 %compare.low.base1417, %compare.local1510, !dbg !26
  %compare.result1512 = select i1 %compare.high.equal1415, i1 %compare.low1511, i1 %compare.high.order1416, !dbg !26
  %930 = select i1 %compare.result1512, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1513 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 55) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1514 = or i32 %compare.local.base1513, 96, !dbg !26
  %compare.low1515 = icmp ugt i32 %compare.low.base1417, %compare.local1514, !dbg !26
  %compare.result1516 = select i1 %compare.high.equal1415, i1 %compare.low1515, i1 %compare.high.order1416, !dbg !26
  %931 = select i1 %compare.result1516, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1517 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 56) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1518 = or i32 %compare.local.base1517, 97, !dbg !26
  %compare.low1519 = icmp ugt i32 %compare.low.base1417, %compare.local1518, !dbg !26
  %compare.result1520 = select i1 %compare.high.equal1415, i1 %compare.low1519, i1 %compare.high.order1416, !dbg !26
  %932 = select i1 %compare.result1520, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1521 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 57) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1522 = or i32 %compare.local.base1521, 98, !dbg !26
  %compare.low1523 = icmp ugt i32 %compare.low.base1417, %compare.local1522, !dbg !26
  %compare.result1524 = select i1 %compare.high.equal1415, i1 %compare.low1523, i1 %compare.high.order1416, !dbg !26
  %933 = select i1 %compare.result1524, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1525 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 58) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1526 = or i32 %compare.local.base1525, 99, !dbg !26
  %compare.low1527 = icmp ugt i32 %compare.low.base1417, %compare.local1526, !dbg !26
  %compare.result1528 = select i1 %compare.high.equal1415, i1 %compare.low1527, i1 %compare.high.order1416, !dbg !26
  %934 = select i1 %compare.result1528, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1529 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 59) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1530 = or i32 %compare.local.base1529, 112, !dbg !26
  %compare.low1531 = icmp ugt i32 %compare.low.base1417, %compare.local1530, !dbg !26
  %compare.result1532 = select i1 %compare.high.equal1415, i1 %compare.low1531, i1 %compare.high.order1416, !dbg !26
  %935 = select i1 %compare.result1532, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1533 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 60) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1534 = or i32 %compare.local.base1533, 113, !dbg !26
  %compare.low1535 = icmp ugt i32 %compare.low.base1417, %compare.local1534, !dbg !26
  %compare.result1536 = select i1 %compare.high.equal1415, i1 %compare.low1535, i1 %compare.high.order1416, !dbg !26
  %936 = select i1 %compare.result1536, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1537 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 61) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1538 = or i32 %compare.local.base1537, 114, !dbg !26
  %compare.low1539 = icmp ugt i32 %compare.low.base1417, %compare.local1538, !dbg !26
  %compare.result1540 = select i1 %compare.high.equal1415, i1 %compare.low1539, i1 %compare.high.order1416, !dbg !26
  %937 = select i1 %compare.result1540, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1541 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1418, i32 62) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1542 = or i32 %compare.local.base1541, 115, !dbg !26
  %compare.low1543 = icmp ugt i32 %compare.low.base1417, %compare.local1542, !dbg !26
  %compare.result1544 = select i1 %compare.high.equal1415, i1 %compare.low1543, i1 %compare.high.order1416, !dbg !26
  %938 = select i1 %compare.result1544, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
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
  %1226 = fsub <1 x float> %1192, %1194, !dbg !31
  %1227 = extractelement <1 x float> %1226, i64 0, !dbg !32
  %1228 = call float @llvm.amdgcn.exp2.f32(float %1227), !dbg !32
  %1229 = fsub <1 x float> %1205, %1207, !dbg !31
  %1230 = extractelement <1 x float> %1229, i64 0, !dbg !32
  %1231 = call float @llvm.amdgcn.exp2.f32(float %1230), !dbg !32
  br i1 %44, label %1232, label %1233, !dbg !15

1232:                                             ; preds = %1007
  %address.thread765 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !40, !frisk.register_lifetimes !1
  %address.local766 = zext nneg i32 %address.thread765 to i64, !dbg !12
  %address.local767 = lshr i64 %address.local766, 6, !dbg !12
  %address.local768 = shl nuw nsw i64 %address.local767, 5, !dbg !15
  %address.local769 = add nuw nsw i64 %address.local768, %address.local766, !dbg !15
  %address.local770 = lshr i64 %address.local766, 4, !dbg !12
  %address.local771 = mul nsw i64 %address.local770, -16, !dbg !13
  %address.local772 = add nsw i64 %address.local769, %address.local771, !dbg !15
  %address.local773 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local772, !dbg !15
  store float %1228, ptr addrspace(3) %address.local773, align 4, !dbg !15
  %address.local774 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local772, !dbg !15
  %address.local775 = getelementptr i8, ptr addrspace(3) %address.local774, i64 64, !dbg !15
  store float %1231, ptr addrspace(3) %address.local775, align 4, !dbg !15
  br label %1233, !dbg !15

1233:                                             ; preds = %1232, %1007
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.thread776 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !40, !frisk.register_lifetimes !1
  %address.local777 = lshr i32 %address.thread776, 2, !dbg !13
  %address.local778 = and i32 %address.local777, 48, !dbg !13
  %address.local779 = and i32 %address.thread776, 15, !dbg !13
  %address.local780 = or disjoint i32 %address.local778, %address.local779, !dbg !13
  %address.local781 = zext nneg i32 %address.local780 to i64, !dbg !13
  %address.local782 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local781, !dbg !15
  %lds.const.gep1731 = getelementptr i8, ptr addrspace(3) %address.local782, i64 256, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1236 = add nuw nsw i64 %102, %15, !dbg !20
  %1237 = shl i64 %1236, 19, !dbg !20
  %1238 = and i64 %1237, 16252928, !dbg !20
  %1239 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1238, !dbg !20
  %1240 = getelementptr [2 x i8], ptr addrspace(1) %1239, i64 %108, !dbg !20
  %address.local790 = zext nneg i32 %address.thread776 to i64, !dbg !12
  %address.local791 = shl nuw nsw i64 %address.local790, 3, !dbg !11
  %address.local792 = and i64 %address.local791, 1536, !dbg !11
  %address.local793 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local792, !dbg !11
  %address.local794 = ptrtoint ptr addrspace(3) %address.local793 to i64, !dbg !11
  %address.local795 = trunc i64 %address.local794 to i32, !dbg !11
  %1241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local795), !dbg !20
  %1242 = zext i32 %1241 to i64, !dbg !20
  %1243 = inttoptr i64 %1242 to ptr addrspace(3), !dbg !20
  %address.local796 = shl nuw nsw i64 %address.local790, 5, !dbg !20
  %address.local797 = and i64 %address.local796, 8064, !dbg !20
  %address.local798 = shl nuw nsw i64 %address.local790, 3, !dbg !20
  %address.local799 = and i64 %address.local798, 24, !dbg !20
  %address.local800 = or disjoint i64 %address.local797, %address.local799, !dbg !20
  %address.local801 = getelementptr [2 x i8], ptr addrspace(1) %1240, i64 %address.local800, !dbg !20
  %1244 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1245 = ptrtoint ptr addrspace(1) %address.local801 to i64, !dbg !20
  %1246 = sub i64 %1245, %1244, !dbg !20
  %dma.buffer.offset1758 = trunc i64 %1246 to i32, !dbg !20
  %1247 = trunc i64 %1244 to i32, !dbg !20
  %1248 = insertelement <4 x i32> poison, i32 %1247, i64 0, !dbg !20
  %1249 = lshr i64 %1244, 32, !dbg !20
  %1250 = trunc i64 %1249 to i32, !dbg !20
  %1251 = insertelement <4 x i32> %1248, i32 %1250, i64 1, !dbg !20
  %1252 = insertelement <4 x i32> %1251, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1759 = insertelement <4 x i32> %1252, i32 131072, i64 3, !dbg !20
  %1253 = ptrtoint ptr addrspace(3) %1243 to i32, !dbg !20
  %dma.buffer.lds1760 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1253), !dbg !20
  %1254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1760, i32 %dma.buffer.offset1758, <4 x i32> %dma.buffer.rsrc1759) #10, !dbg !20
  %address.local802 = or disjoint i64 %address.local790, 256, !dbg !11
  %address.local803 = shl nuw nsw i64 %address.local802, 3, !dbg !11
  %address.local804 = and i64 %address.local803, 3584, !dbg !11
  %address.local805 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local804, !dbg !11
  %address.local806 = ptrtoint ptr addrspace(3) %address.local805 to i64, !dbg !11
  %address.local807 = trunc i64 %address.local806 to i32, !dbg !11
  %1255 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local807), !dbg !20
  %1256 = zext i32 %1255 to i64, !dbg !20
  %1257 = inttoptr i64 %1256 to ptr addrspace(3), !dbg !20
  %address.local808 = shl nuw nsw i64 %address.local802, 5, !dbg !20
  %address.local809 = and i64 %address.local808, 16256, !dbg !20
  %address.local810 = shl nuw nsw i64 %address.local790, 3, !dbg !20
  %address.local811 = and i64 %address.local810, 24, !dbg !20
  %address.local812 = or disjoint i64 %address.local809, %address.local811, !dbg !20
  %address.local813 = getelementptr [2 x i8], ptr addrspace(1) %1240, i64 %address.local812, !dbg !20
  %1258 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1259 = ptrtoint ptr addrspace(1) %address.local813 to i64, !dbg !20
  %1260 = sub i64 %1259, %1258, !dbg !20
  %dma.buffer.offset1755 = trunc i64 %1260 to i32, !dbg !20
  %1261 = trunc i64 %1258 to i32, !dbg !20
  %1262 = insertelement <4 x i32> poison, i32 %1261, i64 0, !dbg !20
  %1263 = lshr i64 %1258, 32, !dbg !20
  %1264 = trunc i64 %1263 to i32, !dbg !20
  %1265 = insertelement <4 x i32> %1262, i32 %1264, i64 1, !dbg !20
  %1266 = insertelement <4 x i32> %1265, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1756 = insertelement <4 x i32> %1266, i32 131072, i64 3, !dbg !20
  %1267 = ptrtoint ptr addrspace(3) %1257 to i32, !dbg !20
  %dma.buffer.lds1757 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1267), !dbg !20
  %1268 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1757, i32 %dma.buffer.offset1755, <4 x i32> %dma.buffer.rsrc1756) #10, !dbg !20
  %1269 = extractvalue [2 x <1 x float>] %64, 0, !dbg !15
  %1270 = extractelement <1 x float> %1269, i64 0, !dbg !15
  %1271 = fmul float %1270, %1228, !dbg !15
  %1272 = extractvalue [2 x <1 x float>] %64, 1, !dbg !15
  %1273 = extractelement <1 x float> %1272, i64 0, !dbg !15
  %1274 = fmul float %1273, %1231, !dbg !15
  %1275 = extractelement <4 x float> %1209, i64 0, !dbg !33
  %1276 = call float @llvm.amdgcn.exp2.f32(float %1275), !dbg !33
  %1277 = insertelement <4 x float> poison, float %1276, i64 0, !dbg !35
  %1278 = extractelement <4 x float> %1209, i64 1, !dbg !33
  %1279 = call float @llvm.amdgcn.exp2.f32(float %1278), !dbg !33
  %1280 = insertelement <4 x float> %1277, float %1279, i64 1, !dbg !35
  %1281 = extractelement <4 x float> %1209, i64 2, !dbg !33
  %1282 = call float @llvm.amdgcn.exp2.f32(float %1281), !dbg !33
  %1283 = insertelement <4 x float> %1280, float %1282, i64 2, !dbg !35
  %1284 = extractelement <4 x float> %1209, i64 3, !dbg !33
  %1285 = call float @llvm.amdgcn.exp2.f32(float %1284), !dbg !33
  %1286 = insertelement <4 x float> %1283, float %1285, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1286) #9, !dbg !35, !frisk.register_lifetimes !1
  %1287 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1287) #9, !dbg !33, !frisk.register_lifetimes !1
  %1288 = extractelement <4 x float> %1210, i64 0, !dbg !33
  %1289 = call float @llvm.amdgcn.exp2.f32(float %1288), !dbg !33
  %1290 = insertelement <4 x float> poison, float %1289, i64 0, !dbg !35
  %local.reduce120 = fadd float %1276, %1289, !dbg !34, !frisk.local_reduction !1
  %1291 = extractelement <4 x float> %1210, i64 1, !dbg !33
  %1292 = call float @llvm.amdgcn.exp2.f32(float %1291), !dbg !33
  %1293 = insertelement <4 x float> %1290, float %1292, i64 1, !dbg !35
  %local.reduce121 = fadd float %1279, %1292, !dbg !34, !frisk.local_reduction !1
  %1294 = extractelement <4 x float> %1210, i64 2, !dbg !33
  %1295 = call float @llvm.amdgcn.exp2.f32(float %1294), !dbg !33
  %1296 = insertelement <4 x float> %1293, float %1295, i64 2, !dbg !35
  %local.reduce122 = fadd float %1282, %1295, !dbg !34, !frisk.local_reduction !1
  %1297 = extractelement <4 x float> %1210, i64 3, !dbg !33
  %1298 = call float @llvm.amdgcn.exp2.f32(float %1297), !dbg !33
  %1299 = insertelement <4 x float> %1296, float %1298, i64 3, !dbg !35
  %math.result227 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1299) #9, !dbg !35, !frisk.register_lifetimes !1
  %1300 = fptrunc <4 x float> %math.result227 to <4 x half>, !dbg !35
  %math.packed228 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1300) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce123 = fadd float %1285, %1298, !dbg !34, !frisk.local_reduction !1
  %1301 = extractelement <4 x float> %1211, i64 0, !dbg !33
  %1302 = call float @llvm.amdgcn.exp2.f32(float %1301), !dbg !33
  %1303 = insertelement <4 x float> poison, float %1302, i64 0, !dbg !35
  %local.reduce124 = fadd float %local.reduce120, %1302, !dbg !34, !frisk.local_reduction !1
  %1304 = extractelement <4 x float> %1211, i64 1, !dbg !33
  %1305 = call float @llvm.amdgcn.exp2.f32(float %1304), !dbg !33
  %1306 = insertelement <4 x float> %1303, float %1305, i64 1, !dbg !35
  %local.reduce125 = fadd float %local.reduce121, %1305, !dbg !34, !frisk.local_reduction !1
  %1307 = extractelement <4 x float> %1211, i64 2, !dbg !33
  %1308 = call float @llvm.amdgcn.exp2.f32(float %1307), !dbg !33
  %1309 = insertelement <4 x float> %1306, float %1308, i64 2, !dbg !35
  %local.reduce126 = fadd float %local.reduce122, %1308, !dbg !34, !frisk.local_reduction !1
  %1310 = extractelement <4 x float> %1211, i64 3, !dbg !33
  %1311 = call float @llvm.amdgcn.exp2.f32(float %1310), !dbg !33
  %1312 = insertelement <4 x float> %1309, float %1311, i64 3, !dbg !35
  %math.result229 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1312) #9, !dbg !35, !frisk.register_lifetimes !1
  %1313 = fptrunc <4 x float> %math.result229 to <4 x half>, !dbg !35
  %math.packed230 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1313) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce127 = fadd float %local.reduce123, %1311, !dbg !34, !frisk.local_reduction !1
  %1314 = extractelement <4 x float> %1212, i64 0, !dbg !33
  %1315 = call float @llvm.amdgcn.exp2.f32(float %1314), !dbg !33
  %1316 = insertelement <4 x float> poison, float %1315, i64 0, !dbg !35
  %local.reduce128 = fadd float %local.reduce124, %1315, !dbg !34, !frisk.local_reduction !1
  %1317 = extractelement <4 x float> %1212, i64 1, !dbg !33
  %1318 = call float @llvm.amdgcn.exp2.f32(float %1317), !dbg !33
  %1319 = insertelement <4 x float> %1316, float %1318, i64 1, !dbg !35
  %local.reduce129 = fadd float %local.reduce125, %1318, !dbg !34, !frisk.local_reduction !1
  %1320 = extractelement <4 x float> %1212, i64 2, !dbg !33
  %1321 = call float @llvm.amdgcn.exp2.f32(float %1320), !dbg !33
  %1322 = insertelement <4 x float> %1319, float %1321, i64 2, !dbg !35
  %local.reduce130 = fadd float %local.reduce126, %1321, !dbg !34, !frisk.local_reduction !1
  %1323 = extractelement <4 x float> %1212, i64 3, !dbg !33
  %1324 = call float @llvm.amdgcn.exp2.f32(float %1323), !dbg !33
  %1325 = insertelement <4 x float> %1322, float %1324, i64 3, !dbg !35
  %math.result231 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1325) #9, !dbg !35, !frisk.register_lifetimes !1
  %1326 = fptrunc <4 x float> %math.result231 to <4 x half>, !dbg !35
  %math.packed232 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1326) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce131 = fadd float %local.reduce127, %1324, !dbg !34, !frisk.local_reduction !1
  %1327 = extractelement <4 x float> %1213, i64 0, !dbg !33
  %1328 = call float @llvm.amdgcn.exp2.f32(float %1327), !dbg !33
  %1329 = insertelement <4 x float> poison, float %1328, i64 0, !dbg !35
  %local.reduce132 = fadd float %local.reduce128, %1328, !dbg !34, !frisk.local_reduction !1
  %1330 = extractelement <4 x float> %1213, i64 1, !dbg !33
  %1331 = call float @llvm.amdgcn.exp2.f32(float %1330), !dbg !33
  %1332 = insertelement <4 x float> %1329, float %1331, i64 1, !dbg !35
  %local.reduce133 = fadd float %local.reduce129, %1331, !dbg !34, !frisk.local_reduction !1
  %1333 = extractelement <4 x float> %1213, i64 2, !dbg !33
  %1334 = call float @llvm.amdgcn.exp2.f32(float %1333), !dbg !33
  %1335 = insertelement <4 x float> %1332, float %1334, i64 2, !dbg !35
  %local.reduce134 = fadd float %local.reduce130, %1334, !dbg !34, !frisk.local_reduction !1
  %1336 = extractelement <4 x float> %1213, i64 3, !dbg !33
  %1337 = call float @llvm.amdgcn.exp2.f32(float %1336), !dbg !33
  %1338 = insertelement <4 x float> %1335, float %1337, i64 3, !dbg !35
  %math.result233 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1338) #9, !dbg !35, !frisk.register_lifetimes !1
  %1339 = fptrunc <4 x float> %math.result233 to <4 x half>, !dbg !35
  %math.packed234 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1339) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce135 = fadd float %local.reduce131, %1337, !dbg !34, !frisk.local_reduction !1
  %1340 = extractelement <4 x float> %1214, i64 0, !dbg !33
  %1341 = call float @llvm.amdgcn.exp2.f32(float %1340), !dbg !33
  %1342 = insertelement <4 x float> poison, float %1341, i64 0, !dbg !35
  %local.reduce136 = fadd float %local.reduce132, %1341, !dbg !34, !frisk.local_reduction !1
  %1343 = extractelement <4 x float> %1214, i64 1, !dbg !33
  %1344 = call float @llvm.amdgcn.exp2.f32(float %1343), !dbg !33
  %1345 = insertelement <4 x float> %1342, float %1344, i64 1, !dbg !35
  %local.reduce137 = fadd float %local.reduce133, %1344, !dbg !34, !frisk.local_reduction !1
  %1346 = extractelement <4 x float> %1214, i64 2, !dbg !33
  %1347 = call float @llvm.amdgcn.exp2.f32(float %1346), !dbg !33
  %1348 = insertelement <4 x float> %1345, float %1347, i64 2, !dbg !35
  %local.reduce138 = fadd float %local.reduce134, %1347, !dbg !34, !frisk.local_reduction !1
  %1349 = extractelement <4 x float> %1214, i64 3, !dbg !33
  %1350 = call float @llvm.amdgcn.exp2.f32(float %1349), !dbg !33
  %1351 = insertelement <4 x float> %1348, float %1350, i64 3, !dbg !35
  %math.result235 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1351) #9, !dbg !35, !frisk.register_lifetimes !1
  %1352 = fptrunc <4 x float> %math.result235 to <4 x half>, !dbg !35
  %math.packed236 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1352) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce139 = fadd float %local.reduce135, %1350, !dbg !34, !frisk.local_reduction !1
  %1353 = extractelement <4 x float> %1215, i64 0, !dbg !33
  %1354 = call float @llvm.amdgcn.exp2.f32(float %1353), !dbg !33
  %1355 = insertelement <4 x float> poison, float %1354, i64 0, !dbg !35
  %local.reduce140 = fadd float %local.reduce136, %1354, !dbg !34, !frisk.local_reduction !1
  %1356 = extractelement <4 x float> %1215, i64 1, !dbg !33
  %1357 = call float @llvm.amdgcn.exp2.f32(float %1356), !dbg !33
  %1358 = insertelement <4 x float> %1355, float %1357, i64 1, !dbg !35
  %local.reduce141 = fadd float %local.reduce137, %1357, !dbg !34, !frisk.local_reduction !1
  %1359 = extractelement <4 x float> %1215, i64 2, !dbg !33
  %1360 = call float @llvm.amdgcn.exp2.f32(float %1359), !dbg !33
  %1361 = insertelement <4 x float> %1358, float %1360, i64 2, !dbg !35
  %local.reduce142 = fadd float %local.reduce138, %1360, !dbg !34, !frisk.local_reduction !1
  %1362 = extractelement <4 x float> %1215, i64 3, !dbg !33
  %1363 = call float @llvm.amdgcn.exp2.f32(float %1362), !dbg !33
  %1364 = insertelement <4 x float> %1361, float %1363, i64 3, !dbg !35
  %math.result237 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1364) #9, !dbg !35, !frisk.register_lifetimes !1
  %1365 = fptrunc <4 x float> %math.result237 to <4 x half>, !dbg !35
  %math.packed238 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1365) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce143 = fadd float %local.reduce139, %1363, !dbg !34, !frisk.local_reduction !1
  %1366 = extractelement <4 x float> %1216, i64 0, !dbg !33
  %1367 = call float @llvm.amdgcn.exp2.f32(float %1366), !dbg !33
  %1368 = insertelement <4 x float> poison, float %1367, i64 0, !dbg !35
  %local.reduce144 = fadd float %local.reduce140, %1367, !dbg !34, !frisk.local_reduction !1
  %1369 = extractelement <4 x float> %1216, i64 1, !dbg !33
  %1370 = call float @llvm.amdgcn.exp2.f32(float %1369), !dbg !33
  %1371 = insertelement <4 x float> %1368, float %1370, i64 1, !dbg !35
  %local.reduce145 = fadd float %local.reduce141, %1370, !dbg !34, !frisk.local_reduction !1
  %local.reduce148 = fadd float %local.reduce144, %local.reduce145, !dbg !34, !frisk.local_reduction !1
  %1372 = extractelement <4 x float> %1216, i64 2, !dbg !33
  %1373 = call float @llvm.amdgcn.exp2.f32(float %1372), !dbg !33
  %1374 = insertelement <4 x float> %1371, float %1373, i64 2, !dbg !35
  %local.reduce146 = fadd float %local.reduce142, %1373, !dbg !34, !frisk.local_reduction !1
  %1375 = extractelement <4 x float> %1216, i64 3, !dbg !33
  %1376 = call float @llvm.amdgcn.exp2.f32(float %1375), !dbg !33
  %1377 = insertelement <4 x float> %1374, float %1376, i64 3, !dbg !35
  %math.result239 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1377) #9, !dbg !35, !frisk.register_lifetimes !1
  %1378 = fptrunc <4 x float> %math.result239 to <4 x half>, !dbg !35
  %math.packed240 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1378) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce147 = fadd float %local.reduce143, %1376, !dbg !34, !frisk.local_reduction !1
  %local.reduce149 = fadd float %local.reduce146, %local.reduce147, !dbg !34, !frisk.local_reduction !1
  %1379 = fadd float %local.reduce148, %local.reduce149, !dbg !34, !frisk.local_reduction !1
  %1380 = extractelement <4 x float> %1218, i64 0, !dbg !33
  %1381 = call float @llvm.amdgcn.exp2.f32(float %1380), !dbg !33
  %1382 = insertelement <4 x float> poison, float %1381, i64 0, !dbg !35
  %1383 = extractelement <4 x float> %1218, i64 1, !dbg !33
  %1384 = call float @llvm.amdgcn.exp2.f32(float %1383), !dbg !33
  %1385 = insertelement <4 x float> %1382, float %1384, i64 1, !dbg !35
  %1386 = extractelement <4 x float> %1218, i64 2, !dbg !33
  %1387 = call float @llvm.amdgcn.exp2.f32(float %1386), !dbg !33
  %1388 = insertelement <4 x float> %1385, float %1387, i64 2, !dbg !35
  %1389 = extractelement <4 x float> %1218, i64 3, !dbg !33
  %1390 = call float @llvm.amdgcn.exp2.f32(float %1389), !dbg !33
  %1391 = insertelement <4 x float> %1388, float %1390, i64 3, !dbg !35
  %math.result241 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1391) #9, !dbg !35, !frisk.register_lifetimes !1
  %1392 = fptrunc <4 x float> %math.result241 to <4 x half>, !dbg !35
  %math.packed242 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1392) #9, !dbg !33, !frisk.register_lifetimes !1
  %1393 = extractelement <4 x float> %1219, i64 0, !dbg !33
  %1394 = call float @llvm.amdgcn.exp2.f32(float %1393), !dbg !33
  %1395 = insertelement <4 x float> poison, float %1394, i64 0, !dbg !35
  %local.reduce = fadd float %1381, %1394, !dbg !34, !frisk.local_reduction !1
  %1396 = extractelement <4 x float> %1219, i64 1, !dbg !33
  %1397 = call float @llvm.amdgcn.exp2.f32(float %1396), !dbg !33
  %1398 = insertelement <4 x float> %1395, float %1397, i64 1, !dbg !35
  %local.reduce90 = fadd float %1384, %1397, !dbg !34, !frisk.local_reduction !1
  %1399 = extractelement <4 x float> %1219, i64 2, !dbg !33
  %1400 = call float @llvm.amdgcn.exp2.f32(float %1399), !dbg !33
  %1401 = insertelement <4 x float> %1398, float %1400, i64 2, !dbg !35
  %local.reduce91 = fadd float %1387, %1400, !dbg !34, !frisk.local_reduction !1
  %1402 = extractelement <4 x float> %1219, i64 3, !dbg !33
  %1403 = call float @llvm.amdgcn.exp2.f32(float %1402), !dbg !33
  %1404 = insertelement <4 x float> %1401, float %1403, i64 3, !dbg !35
  %math.result243 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1404) #9, !dbg !35, !frisk.register_lifetimes !1
  %1405 = fptrunc <4 x float> %math.result243 to <4 x half>, !dbg !35
  %math.packed244 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1405) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce92 = fadd float %1390, %1403, !dbg !34, !frisk.local_reduction !1
  %1406 = extractelement <4 x float> %1220, i64 0, !dbg !33
  %1407 = call float @llvm.amdgcn.exp2.f32(float %1406), !dbg !33
  %1408 = insertelement <4 x float> poison, float %1407, i64 0, !dbg !35
  %local.reduce93 = fadd float %local.reduce, %1407, !dbg !34, !frisk.local_reduction !1
  %1409 = extractelement <4 x float> %1220, i64 1, !dbg !33
  %1410 = call float @llvm.amdgcn.exp2.f32(float %1409), !dbg !33
  %1411 = insertelement <4 x float> %1408, float %1410, i64 1, !dbg !35
  %local.reduce94 = fadd float %local.reduce90, %1410, !dbg !34, !frisk.local_reduction !1
  %1412 = extractelement <4 x float> %1220, i64 2, !dbg !33
  %1413 = call float @llvm.amdgcn.exp2.f32(float %1412), !dbg !33
  %1414 = insertelement <4 x float> %1411, float %1413, i64 2, !dbg !35
  %local.reduce95 = fadd float %local.reduce91, %1413, !dbg !34, !frisk.local_reduction !1
  %1415 = extractelement <4 x float> %1220, i64 3, !dbg !33
  %1416 = call float @llvm.amdgcn.exp2.f32(float %1415), !dbg !33
  %1417 = insertelement <4 x float> %1414, float %1416, i64 3, !dbg !35
  %math.result245 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1417) #9, !dbg !35, !frisk.register_lifetimes !1
  %1418 = fptrunc <4 x float> %math.result245 to <4 x half>, !dbg !35
  %math.packed246 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1418) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce96 = fadd float %local.reduce92, %1416, !dbg !34, !frisk.local_reduction !1
  %1419 = extractelement <4 x float> %1221, i64 0, !dbg !33
  %1420 = call float @llvm.amdgcn.exp2.f32(float %1419), !dbg !33
  %1421 = insertelement <4 x float> poison, float %1420, i64 0, !dbg !35
  %local.reduce97 = fadd float %local.reduce93, %1420, !dbg !34, !frisk.local_reduction !1
  %1422 = extractelement <4 x float> %1221, i64 1, !dbg !33
  %1423 = call float @llvm.amdgcn.exp2.f32(float %1422), !dbg !33
  %1424 = insertelement <4 x float> %1421, float %1423, i64 1, !dbg !35
  %local.reduce98 = fadd float %local.reduce94, %1423, !dbg !34, !frisk.local_reduction !1
  %1425 = extractelement <4 x float> %1221, i64 2, !dbg !33
  %1426 = call float @llvm.amdgcn.exp2.f32(float %1425), !dbg !33
  %1427 = insertelement <4 x float> %1424, float %1426, i64 2, !dbg !35
  %local.reduce99 = fadd float %local.reduce95, %1426, !dbg !34, !frisk.local_reduction !1
  %1428 = extractelement <4 x float> %1221, i64 3, !dbg !33
  %1429 = call float @llvm.amdgcn.exp2.f32(float %1428), !dbg !33
  %1430 = insertelement <4 x float> %1427, float %1429, i64 3, !dbg !35
  %math.result247 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1430) #9, !dbg !35, !frisk.register_lifetimes !1
  %1431 = fptrunc <4 x float> %math.result247 to <4 x half>, !dbg !35
  %math.packed248 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1431) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce100 = fadd float %local.reduce96, %1429, !dbg !34, !frisk.local_reduction !1
  %1432 = extractelement <4 x float> %1222, i64 0, !dbg !33
  %1433 = call float @llvm.amdgcn.exp2.f32(float %1432), !dbg !33
  %1434 = insertelement <4 x float> poison, float %1433, i64 0, !dbg !35
  %local.reduce101 = fadd float %local.reduce97, %1433, !dbg !34, !frisk.local_reduction !1
  %1435 = extractelement <4 x float> %1222, i64 1, !dbg !33
  %1436 = call float @llvm.amdgcn.exp2.f32(float %1435), !dbg !33
  %1437 = insertelement <4 x float> %1434, float %1436, i64 1, !dbg !35
  %local.reduce102 = fadd float %local.reduce98, %1436, !dbg !34, !frisk.local_reduction !1
  %1438 = extractelement <4 x float> %1222, i64 2, !dbg !33
  %1439 = call float @llvm.amdgcn.exp2.f32(float %1438), !dbg !33
  %1440 = insertelement <4 x float> %1437, float %1439, i64 2, !dbg !35
  %local.reduce103 = fadd float %local.reduce99, %1439, !dbg !34, !frisk.local_reduction !1
  %1441 = extractelement <4 x float> %1222, i64 3, !dbg !33
  %1442 = call float @llvm.amdgcn.exp2.f32(float %1441), !dbg !33
  %1443 = insertelement <4 x float> %1440, float %1442, i64 3, !dbg !35
  %math.result249 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1443) #9, !dbg !35, !frisk.register_lifetimes !1
  %1444 = fptrunc <4 x float> %math.result249 to <4 x half>, !dbg !35
  %math.packed250 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1444) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce104 = fadd float %local.reduce100, %1442, !dbg !34, !frisk.local_reduction !1
  %1445 = extractelement <4 x float> %1223, i64 0, !dbg !33
  %1446 = call float @llvm.amdgcn.exp2.f32(float %1445), !dbg !33
  %1447 = insertelement <4 x float> poison, float %1446, i64 0, !dbg !35
  %local.reduce105 = fadd float %local.reduce101, %1446, !dbg !34, !frisk.local_reduction !1
  %1448 = extractelement <4 x float> %1223, i64 1, !dbg !33
  %1449 = call float @llvm.amdgcn.exp2.f32(float %1448), !dbg !33
  %1450 = insertelement <4 x float> %1447, float %1449, i64 1, !dbg !35
  %local.reduce106 = fadd float %local.reduce102, %1449, !dbg !34, !frisk.local_reduction !1
  %1451 = extractelement <4 x float> %1223, i64 2, !dbg !33
  %1452 = call float @llvm.amdgcn.exp2.f32(float %1451), !dbg !33
  %1453 = insertelement <4 x float> %1450, float %1452, i64 2, !dbg !35
  %local.reduce107 = fadd float %local.reduce103, %1452, !dbg !34, !frisk.local_reduction !1
  %1454 = extractelement <4 x float> %1223, i64 3, !dbg !33
  %1455 = call float @llvm.amdgcn.exp2.f32(float %1454), !dbg !33
  %1456 = insertelement <4 x float> %1453, float %1455, i64 3, !dbg !35
  %math.result251 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1456) #9, !dbg !35, !frisk.register_lifetimes !1
  %1457 = fptrunc <4 x float> %math.result251 to <4 x half>, !dbg !35
  %math.packed252 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1457) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce108 = fadd float %local.reduce104, %1455, !dbg !34, !frisk.local_reduction !1
  %1458 = extractelement <4 x float> %1224, i64 0, !dbg !33
  %1459 = call float @llvm.amdgcn.exp2.f32(float %1458), !dbg !33
  %1460 = insertelement <4 x float> poison, float %1459, i64 0, !dbg !35
  %local.reduce109 = fadd float %local.reduce105, %1459, !dbg !34, !frisk.local_reduction !1
  %1461 = extractelement <4 x float> %1224, i64 1, !dbg !33
  %1462 = call float @llvm.amdgcn.exp2.f32(float %1461), !dbg !33
  %1463 = insertelement <4 x float> %1460, float %1462, i64 1, !dbg !35
  %local.reduce110 = fadd float %local.reduce106, %1462, !dbg !34, !frisk.local_reduction !1
  %1464 = extractelement <4 x float> %1224, i64 2, !dbg !33
  %1465 = call float @llvm.amdgcn.exp2.f32(float %1464), !dbg !33
  %1466 = insertelement <4 x float> %1463, float %1465, i64 2, !dbg !35
  %local.reduce111 = fadd float %local.reduce107, %1465, !dbg !34, !frisk.local_reduction !1
  %1467 = extractelement <4 x float> %1224, i64 3, !dbg !33
  %1468 = call float @llvm.amdgcn.exp2.f32(float %1467), !dbg !33
  %1469 = insertelement <4 x float> %1466, float %1468, i64 3, !dbg !35
  %math.result253 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1469) #9, !dbg !35, !frisk.register_lifetimes !1
  %1470 = fptrunc <4 x float> %math.result253 to <4 x half>, !dbg !35
  %math.packed254 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1470) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce112 = fadd float %local.reduce108, %1468, !dbg !34, !frisk.local_reduction !1
  %1471 = extractelement <4 x float> %1225, i64 0, !dbg !33
  %1472 = call float @llvm.amdgcn.exp2.f32(float %1471), !dbg !33
  %1473 = insertelement <4 x float> poison, float %1472, i64 0, !dbg !35
  %local.reduce113 = fadd float %local.reduce109, %1472, !dbg !34, !frisk.local_reduction !1
  %1474 = extractelement <4 x float> %1225, i64 1, !dbg !33
  %1475 = call float @llvm.amdgcn.exp2.f32(float %1474), !dbg !33
  %1476 = insertelement <4 x float> %1473, float %1475, i64 1, !dbg !35
  %local.reduce114 = fadd float %local.reduce110, %1475, !dbg !34, !frisk.local_reduction !1
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !34, !frisk.local_reduction !1
  %1477 = extractelement <4 x float> %1225, i64 2, !dbg !33
  %1478 = call float @llvm.amdgcn.exp2.f32(float %1477), !dbg !33
  %1479 = insertelement <4 x float> %1476, float %1478, i64 2, !dbg !35
  %local.reduce115 = fadd float %local.reduce111, %1478, !dbg !34, !frisk.local_reduction !1
  %1480 = extractelement <4 x float> %1225, i64 3, !dbg !33
  %1481 = call float @llvm.amdgcn.exp2.f32(float %1480), !dbg !33
  %1482 = insertelement <4 x float> %1479, float %1481, i64 3, !dbg !35
  %math.result255 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1482) #9, !dbg !35, !frisk.register_lifetimes !1
  %1483 = fptrunc <4 x float> %math.result255 to <4 x half>, !dbg !35
  %math.packed256 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1483) #9, !dbg !34, !frisk.register_lifetimes !1
  %local.reduce116 = fadd float %local.reduce112, %1481, !dbg !34, !frisk.local_reduction !1
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !34, !frisk.local_reduction !1
  %1484 = fadd float %local.reduce117, %local.reduce118, !dbg !34, !frisk.local_reduction !1
  %1485 = bitcast float %1379 to i32, !dbg !34
  %1486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1485), !dbg !34
  %1487 = bitcast i32 %1486 to float, !dbg !34
  %1488 = fadd float %1379, %1487, !dbg !34
  %1489 = bitcast float %1488 to i32, !dbg !34
  %1490 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1489), !dbg !34
  %1491 = bitcast i32 %1490 to float, !dbg !34
  %1492 = fadd float %1488, %1491, !dbg !34
  %1493 = fadd float %1492, %1271, !dbg !34
  %1494 = bitcast float %1484 to i32, !dbg !34
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1494), !dbg !34
  %1496 = bitcast i32 %1495 to float, !dbg !34
  %1497 = fadd float %1484, %1496, !dbg !34
  %1498 = bitcast float %1497 to i32, !dbg !34
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1498), !dbg !34
  %1500 = bitcast i32 %1499 to float, !dbg !34
  %1501 = fadd float %1497, %1500, !dbg !34
  %1502 = fadd float %1501, %1274, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local814 = shl nuw nsw i32 %address.thread776, 3, !dbg !16
  %address.local815 = and i32 %address.local814, 504, !dbg !16
  %address.local816 = zext nneg i32 %address.local815 to i64, !dbg !16
  %address.local817 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local816, !dbg !16
  %1503 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local817, i32 0) #0, !dbg !16
  %1504 = bitcast <4 x i32> %1503 to <8 x half>, !dbg !16
  %address.local818 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local816, !dbg !16
  %address.local819 = getelementptr i8, ptr addrspace(3) %address.local818, i64 1024, !dbg !16
  %1505 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #0, !dbg !16
  %1506 = bitcast <4 x i32> %1505 to <8 x half>, !dbg !16
  %address.local820 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local816, !dbg !16
  %address.local821 = getelementptr i8, ptr addrspace(3) %address.local820, i64 2048, !dbg !16
  %1507 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local821, i32 0) #0, !dbg !16
  %1508 = bitcast <4 x i32> %1507 to <8 x half>, !dbg !16
  %address.local822 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local816, !dbg !16
  %address.local823 = getelementptr i8, ptr addrspace(3) %address.local822, i64 3072, !dbg !16
  %1509 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #0, !dbg !16
  %1510 = bitcast <4 x i32> %1509 to <8 x half>, !dbg !16
  %address.local824 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local816, !dbg !16
  %address.local825 = getelementptr i8, ptr addrspace(3) %address.local824, i64 4096, !dbg !16
  %1511 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local825, i32 0) #0, !dbg !16
  %1512 = bitcast <4 x i32> %1511 to <8 x half>, !dbg !16
  %address.local826 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local816, !dbg !16
  %address.local827 = getelementptr i8, ptr addrspace(3) %address.local826, i64 5120, !dbg !16
  %1513 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local827, i32 0) #0, !dbg !16
  %1514 = bitcast <4 x i32> %1513 to <8 x half>, !dbg !16
  %address.local828 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local816, !dbg !16
  %address.local829 = getelementptr i8, ptr addrspace(3) %address.local828, i64 6144, !dbg !16
  %1515 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local829, i32 0) #0, !dbg !16
  %1516 = bitcast <4 x i32> %1515 to <8 x half>, !dbg !16
  %address.local830 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local816, !dbg !16
  %address.local831 = getelementptr i8, ptr addrspace(3) %address.local830, i64 7168, !dbg !16
  %1517 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local831, i32 0) #0, !dbg !16
  %1518 = bitcast <4 x i32> %1517 to <8 x half>, !dbg !16
  %1519 = shufflevector <8 x half> %1504, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1519, <4 x float> zeroinitializer) #9, !dbg !16
  %1521 = shufflevector <8 x half> %1506, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1521, <4 x float> %1520) #9, !dbg !16
  %1523 = shufflevector <8 x half> %1508, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1523, <4 x float> %1522) #9, !dbg !16
  %1525 = shufflevector <8 x half> %1510, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1525, <4 x float> %1524) #9, !dbg !16
  %1527 = shufflevector <8 x half> %1512, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1527, <4 x float> %1526) #9, !dbg !16
  %1529 = shufflevector <8 x half> %1514, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1529, <4 x float> %1528) #9, !dbg !16
  %1531 = shufflevector <8 x half> %1516, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1531, <4 x float> %1530) #9, !dbg !16
  %1533 = shufflevector <8 x half> %1518, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1533, <4 x float> %1532) #9, !dbg !16
  %1535 = extractelement <4 x float> %1534, i64 0, !dbg !16
  %1536 = extractelement <4 x float> %1534, i64 1, !dbg !16
  %1537 = extractelement <4 x float> %1534, i64 2, !dbg !16
  %1538 = extractelement <4 x float> %1534, i64 3, !dbg !16
  %1539 = select i1 %.not, float %1536, float %1535, !dbg !16
  %1540 = bitcast float %1539 to i32, !dbg !16
  %1541 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1540), !dbg !16
  %1542 = bitcast i32 %1541 to float, !dbg !16
  %1543 = select i1 %.not, float %1535, float %1542, !dbg !16
  %1544 = select i1 %.not, float %1542, float %1536, !dbg !16
  %1545 = select i1 %.not, float %1538, float %1537, !dbg !16
  %1546 = bitcast float %1545 to i32, !dbg !16
  %1547 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1546), !dbg !16
  %1548 = bitcast i32 %1547 to float, !dbg !16
  %1549 = select i1 %.not, float %1537, float %1548, !dbg !16
  %1550 = select i1 %.not, float %1548, float %1538, !dbg !16
  %1551 = select i1 %.not1, float %1549, float %1543, !dbg !16
  %1552 = bitcast float %1551 to i32, !dbg !16
  %1553 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1552), !dbg !16
  %1554 = bitcast i32 %1553 to float, !dbg !16
  %1555 = select i1 %.not1, float %1543, float %1554, !dbg !16
  %1556 = select i1 %.not1, float %1554, float %1549, !dbg !16
  %1557 = select i1 %.not1, float %1550, float %1544, !dbg !16
  %1558 = bitcast float %1557 to i32, !dbg !16
  %1559 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1558), !dbg !16
  %1560 = bitcast i32 %1559 to float, !dbg !16
  %1561 = select i1 %.not1, float %1544, float %1560, !dbg !16
  %1562 = select i1 %.not1, float %1560, float %1550, !dbg !16
  %1563 = shufflevector <8 x half> %1504, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1563, <4 x float> zeroinitializer) #9, !dbg !16
  %1565 = shufflevector <8 x half> %1506, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1565, <4 x float> %1564) #9, !dbg !16
  %1567 = shufflevector <8 x half> %1508, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1567, <4 x float> %1566) #9, !dbg !16
  %1569 = shufflevector <8 x half> %1510, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1569, <4 x float> %1568) #9, !dbg !16
  %1571 = shufflevector <8 x half> %1512, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1571, <4 x float> %1570) #9, !dbg !16
  %1573 = shufflevector <8 x half> %1514, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1573, <4 x float> %1572) #9, !dbg !16
  %1575 = shufflevector <8 x half> %1516, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1575, <4 x float> %1574) #9, !dbg !16
  %1577 = shufflevector <8 x half> %1518, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1577, <4 x float> %1576) #9, !dbg !16
  %1579 = extractelement <4 x float> %1578, i64 0, !dbg !16
  %1580 = extractelement <4 x float> %1578, i64 1, !dbg !16
  %1581 = extractelement <4 x float> %1578, i64 2, !dbg !16
  %1582 = extractelement <4 x float> %1578, i64 3, !dbg !16
  %1583 = select i1 %.not, float %1580, float %1579, !dbg !16
  %1584 = bitcast float %1583 to i32, !dbg !16
  %1585 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1584), !dbg !16
  %1586 = bitcast i32 %1585 to float, !dbg !16
  %1587 = select i1 %.not, float %1579, float %1586, !dbg !16
  %1588 = select i1 %.not, float %1586, float %1580, !dbg !16
  %1589 = select i1 %.not, float %1582, float %1581, !dbg !16
  %1590 = bitcast float %1589 to i32, !dbg !16
  %1591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1590), !dbg !16
  %1592 = bitcast i32 %1591 to float, !dbg !16
  %1593 = select i1 %.not, float %1581, float %1592, !dbg !16
  %1594 = select i1 %.not, float %1592, float %1582, !dbg !16
  %1595 = select i1 %.not1, float %1593, float %1587, !dbg !16
  %1596 = bitcast float %1595 to i32, !dbg !16
  %1597 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1596), !dbg !16
  %1598 = bitcast i32 %1597 to float, !dbg !16
  %1599 = select i1 %.not1, float %1587, float %1598, !dbg !16
  %1600 = select i1 %.not1, float %1598, float %1593, !dbg !16
  %1601 = select i1 %.not1, float %1594, float %1588, !dbg !16
  %1602 = bitcast float %1601 to i32, !dbg !16
  %1603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1602), !dbg !16
  %1604 = bitcast i32 %1603 to float, !dbg !16
  %1605 = select i1 %.not1, float %1588, float %1604, !dbg !16
  %1606 = select i1 %.not1, float %1604, float %1594, !dbg !16
  %1607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1519, <4 x float> zeroinitializer) #9, !dbg !16
  %1608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1521, <4 x float> %1607) #9, !dbg !16
  %1609 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1523, <4 x float> %1608) #9, !dbg !16
  %1610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1525, <4 x float> %1609) #9, !dbg !16
  %1611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1527, <4 x float> %1610) #9, !dbg !16
  %1612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1529, <4 x float> %1611) #9, !dbg !16
  %1613 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1531, <4 x float> %1612) #9, !dbg !16
  %1614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1533, <4 x float> %1613) #9, !dbg !16
  %1615 = extractelement <4 x float> %1614, i64 0, !dbg !16
  %1616 = extractelement <4 x float> %1614, i64 1, !dbg !16
  %1617 = extractelement <4 x float> %1614, i64 2, !dbg !16
  %1618 = extractelement <4 x float> %1614, i64 3, !dbg !16
  %1619 = select i1 %.not, float %1616, float %1615, !dbg !16
  %1620 = bitcast float %1619 to i32, !dbg !16
  %1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1620), !dbg !16
  %1622 = bitcast i32 %1621 to float, !dbg !16
  %1623 = select i1 %.not, float %1615, float %1622, !dbg !16
  %1624 = select i1 %.not, float %1622, float %1616, !dbg !16
  %1625 = select i1 %.not, float %1618, float %1617, !dbg !16
  %1626 = bitcast float %1625 to i32, !dbg !16
  %1627 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1626), !dbg !16
  %1628 = bitcast i32 %1627 to float, !dbg !16
  %1629 = select i1 %.not, float %1617, float %1628, !dbg !16
  %1630 = select i1 %.not, float %1628, float %1618, !dbg !16
  %1631 = select i1 %.not1, float %1629, float %1623, !dbg !16
  %1632 = bitcast float %1631 to i32, !dbg !16
  %1633 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1632), !dbg !16
  %1634 = bitcast i32 %1633 to float, !dbg !16
  %1635 = select i1 %.not1, float %1623, float %1634, !dbg !16
  %1636 = select i1 %.not1, float %1634, float %1629, !dbg !16
  %1637 = select i1 %.not1, float %1630, float %1624, !dbg !16
  %1638 = bitcast float %1637 to i32, !dbg !16
  %1639 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1638), !dbg !16
  %1640 = bitcast i32 %1639 to float, !dbg !16
  %1641 = select i1 %.not1, float %1624, float %1640, !dbg !16
  %1642 = select i1 %.not1, float %1640, float %1630, !dbg !16
  %1643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1563, <4 x float> zeroinitializer) #9, !dbg !16
  %1644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1565, <4 x float> %1643) #9, !dbg !16
  %1645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1567, <4 x float> %1644) #9, !dbg !16
  %1646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1569, <4 x float> %1645) #9, !dbg !16
  %1647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1571, <4 x float> %1646) #9, !dbg !16
  %1648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1573, <4 x float> %1647) #9, !dbg !16
  %1649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1575, <4 x float> %1648) #9, !dbg !16
  %1650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1577, <4 x float> %1649) #9, !dbg !16
  %1651 = extractelement <4 x float> %1650, i64 0, !dbg !16
  %1652 = extractelement <4 x float> %1650, i64 1, !dbg !16
  %1653 = extractelement <4 x float> %1650, i64 2, !dbg !16
  %1654 = extractelement <4 x float> %1650, i64 3, !dbg !16
  %1655 = select i1 %.not, float %1652, float %1651, !dbg !16
  %1656 = bitcast float %1655 to i32, !dbg !16
  %1657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1656), !dbg !16
  %1658 = bitcast i32 %1657 to float, !dbg !16
  %1659 = select i1 %.not, float %1651, float %1658, !dbg !16
  %1660 = select i1 %.not, float %1658, float %1652, !dbg !16
  %1661 = select i1 %.not, float %1654, float %1653, !dbg !16
  %1662 = bitcast float %1661 to i32, !dbg !16
  %1663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1662), !dbg !16
  %1664 = bitcast i32 %1663 to float, !dbg !16
  %1665 = select i1 %.not, float %1653, float %1664, !dbg !16
  %1666 = select i1 %.not, float %1664, float %1654, !dbg !16
  %1667 = select i1 %.not1, float %1665, float %1659, !dbg !16
  %1668 = bitcast float %1667 to i32, !dbg !16
  %1669 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1668), !dbg !16
  %1670 = bitcast i32 %1669 to float, !dbg !16
  %1671 = select i1 %.not1, float %1659, float %1670, !dbg !16
  %1672 = select i1 %.not1, float %1670, float %1665, !dbg !16
  %1673 = select i1 %.not1, float %1666, float %1660, !dbg !16
  %1674 = bitcast float %1673 to i32, !dbg !16
  %1675 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1674), !dbg !16
  %1676 = bitcast i32 %1675 to float, !dbg !16
  %1677 = select i1 %.not1, float %1660, float %1676, !dbg !16
  %1678 = select i1 %.not1, float %1676, float %1666, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %44, label %1679, label %1680, !dbg !36

1679:                                             ; preds = %1233
  %address.thread832 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local833 = zext nneg i32 %address.thread832 to i64, !dbg !12
  %address.local834 = lshr i64 %address.local833, 6, !dbg !12
  %address.local835 = shl nuw nsw i64 %address.local834, 5, !dbg !36
  %address.local836 = add nuw nsw i64 %address.local835, %address.local833, !dbg !36
  %address.local837 = lshr i64 %address.local833, 4, !dbg !12
  %address.local838 = mul nsw i64 %address.local837, -16, !dbg !13
  %address.local839 = add nsw i64 %address.local836, %address.local838, !dbg !36
  %address.local840 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local839, !dbg !36
  store float %1228, ptr addrspace(3) %address.local840, align 4, !dbg !36
  %address.local841 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local839, !dbg !36
  %address.local842 = getelementptr i8, ptr addrspace(3) %address.local841, i64 64, !dbg !36
  store float %1231, ptr addrspace(3) %address.local842, align 4, !dbg !36
  br label %1680, !dbg !36

1680:                                             ; preds = %1679, %1233
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread843 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %lds.const.gep1732 = getelementptr i8, ptr addrspace(3) %address.local782, i64 0, !dbg !36
  %lds.const.gep1733 = getelementptr i8, ptr addrspace(3) %address.local782, i64 256, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %1683 = add nuw nsw i64 %102, %15, !dbg !20
  %1684 = shl i64 %1683, 19, !dbg !20
  %1685 = and i64 %1684, 16252928, !dbg !20
  %1686 = add i64 %1685, %108, !dbg !20
  %1687 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1686, !dbg !20
  %1688 = getelementptr i8, ptr addrspace(1) %1687, i64 64, !dbg !20
  %address.local857 = zext nneg i32 %address.thread843 to i64, !dbg !12
  %address.local858 = shl nuw nsw i64 %address.local857, 3, !dbg !11
  %address.local859 = and i64 %address.local858, 1536, !dbg !11
  %address.local860 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !11
  %address.local861 = ptrtoint ptr addrspace(3) %address.local860 to i64, !dbg !11
  %address.local862 = trunc i64 %address.local861 to i32, !dbg !11
  %1689 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local862), !dbg !20
  %1690 = zext i32 %1689 to i64, !dbg !20
  %1691 = inttoptr i64 %1690 to ptr addrspace(3), !dbg !20
  %address.local863 = shl nuw nsw i64 %address.local857, 5, !dbg !20
  %address.local864 = and i64 %address.local863, 8064, !dbg !20
  %address.local865 = shl nuw nsw i64 %address.local857, 3, !dbg !20
  %address.local866 = and i64 %address.local865, 24, !dbg !20
  %address.local867 = or disjoint i64 %address.local864, %address.local866, !dbg !20
  %address.local868 = getelementptr [2 x i8], ptr addrspace(1) %1688, i64 %address.local867, !dbg !20
  %1692 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1693 = ptrtoint ptr addrspace(1) %address.local868 to i64, !dbg !20
  %1694 = sub i64 %1693, %1692, !dbg !20
  %dma.buffer.offset1752 = trunc i64 %1694 to i32, !dbg !20
  %1695 = trunc i64 %1692 to i32, !dbg !20
  %1696 = insertelement <4 x i32> poison, i32 %1695, i64 0, !dbg !20
  %1697 = lshr i64 %1692, 32, !dbg !20
  %1698 = trunc i64 %1697 to i32, !dbg !20
  %1699 = insertelement <4 x i32> %1696, i32 %1698, i64 1, !dbg !20
  %1700 = insertelement <4 x i32> %1699, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1753 = insertelement <4 x i32> %1700, i32 131072, i64 3, !dbg !20
  %1701 = ptrtoint ptr addrspace(3) %1691 to i32, !dbg !20
  %dma.buffer.lds1754 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1701), !dbg !20
  %1702 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1754, i32 %dma.buffer.offset1752, <4 x i32> %dma.buffer.rsrc1753) #10, !dbg !20
  %address.local869 = or disjoint i64 %address.local857, 256, !dbg !11
  %address.local870 = shl nuw nsw i64 %address.local869, 3, !dbg !11
  %address.local871 = and i64 %address.local870, 3584, !dbg !11
  %address.local872 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local871, !dbg !11
  %address.local873 = ptrtoint ptr addrspace(3) %address.local872 to i64, !dbg !11
  %address.local874 = trunc i64 %address.local873 to i32, !dbg !11
  %1703 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local874), !dbg !20
  %1704 = zext i32 %1703 to i64, !dbg !20
  %1705 = inttoptr i64 %1704 to ptr addrspace(3), !dbg !20
  %address.local875 = shl nuw nsw i64 %address.local869, 5, !dbg !20
  %address.local876 = and i64 %address.local875, 16256, !dbg !20
  %address.local877 = shl nuw nsw i64 %address.local857, 3, !dbg !20
  %address.local878 = and i64 %address.local877, 24, !dbg !20
  %address.local879 = or disjoint i64 %address.local876, %address.local878, !dbg !20
  %address.local880 = getelementptr [2 x i8], ptr addrspace(1) %1688, i64 %address.local879, !dbg !20
  %1706 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1707 = ptrtoint ptr addrspace(1) %address.local880 to i64, !dbg !20
  %1708 = sub i64 %1707, %1706, !dbg !20
  %dma.buffer.offset1749 = trunc i64 %1708 to i32, !dbg !20
  %1709 = trunc i64 %1706 to i32, !dbg !20
  %1710 = insertelement <4 x i32> poison, i32 %1709, i64 0, !dbg !20
  %1711 = lshr i64 %1706, 32, !dbg !20
  %1712 = trunc i64 %1711 to i32, !dbg !20
  %1713 = insertelement <4 x i32> %1710, i32 %1712, i64 1, !dbg !20
  %1714 = insertelement <4 x i32> %1713, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1750 = insertelement <4 x i32> %1714, i32 131072, i64 3, !dbg !20
  %1715 = ptrtoint ptr addrspace(3) %1705 to i32, !dbg !20
  %dma.buffer.lds1751 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1715), !dbg !20
  %1716 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1751, i32 %dma.buffer.offset1749, <4 x i32> %dma.buffer.rsrc1750) #10, !dbg !20
  %1717 = extractvalue [2 x <8 x float>] %63, 0, !dbg !36
  %1718 = shufflevector <8 x float> %1717, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1719 = insertelement <4 x float> poison, float %1228, i64 0, !dbg !36
  %1720 = shufflevector <4 x float> %1719, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1721 = fmul <4 x float> %1718, %1720, !dbg !36
  %1722 = insertelement <4 x float> poison, float %1555, i64 0, !dbg !37
  %1723 = insertelement <4 x float> %1722, float %1561, i64 1, !dbg !37
  %1724 = insertelement <4 x float> %1723, float %1556, i64 2, !dbg !37
  %1725 = insertelement <4 x float> %1724, float %1562, i64 3, !dbg !37
  %1726 = fadd <4 x float> %1721, %1725, !dbg !37
  %1727 = shufflevector <8 x float> %1717, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1728 = fmul <4 x float> %1727, %1720, !dbg !36
  %1729 = insertelement <4 x float> poison, float %1599, i64 0, !dbg !37
  %1730 = insertelement <4 x float> %1729, float %1605, i64 1, !dbg !37
  %1731 = insertelement <4 x float> %1730, float %1600, i64 2, !dbg !37
  %1732 = insertelement <4 x float> %1731, float %1606, i64 3, !dbg !37
  %1733 = fadd <4 x float> %1728, %1732, !dbg !37
  %1734 = extractvalue [2 x <8 x float>] %63, 1, !dbg !36
  %1735 = shufflevector <8 x float> %1734, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1736 = insertelement <4 x float> poison, float %1231, i64 0, !dbg !36
  %1737 = shufflevector <4 x float> %1736, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1738 = fmul <4 x float> %1735, %1737, !dbg !36
  %1739 = insertelement <4 x float> poison, float %1635, i64 0, !dbg !37
  %1740 = insertelement <4 x float> %1739, float %1641, i64 1, !dbg !37
  %1741 = insertelement <4 x float> %1740, float %1636, i64 2, !dbg !37
  %1742 = insertelement <4 x float> %1741, float %1642, i64 3, !dbg !37
  %1743 = fadd <4 x float> %1738, %1742, !dbg !37
  %1744 = shufflevector <8 x float> %1734, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1745 = fmul <4 x float> %1744, %1737, !dbg !36
  %1746 = insertelement <4 x float> poison, float %1671, i64 0, !dbg !37
  %1747 = insertelement <4 x float> %1746, float %1677, i64 1, !dbg !37
  %1748 = insertelement <4 x float> %1747, float %1672, i64 2, !dbg !37
  %1749 = insertelement <4 x float> %1748, float %1678, i64 3, !dbg !37
  %1750 = fadd <4 x float> %1745, %1749, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local881 = shl nuw nsw i32 %address.thread843, 3, !dbg !16
  %address.local882 = and i32 %address.local881, 504, !dbg !16
  %address.local883 = zext nneg i32 %address.local882 to i64, !dbg !16
  %address.local884 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !16
  %1751 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !16
  %1752 = bitcast <4 x i32> %1751 to <8 x half>, !dbg !16
  %address.local885 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !16
  %address.local886 = getelementptr i8, ptr addrspace(3) %address.local885, i64 1024, !dbg !16
  %1753 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local886, i32 0) #0, !dbg !16
  %1754 = bitcast <4 x i32> %1753 to <8 x half>, !dbg !16
  %address.local887 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !16
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local887, i64 2048, !dbg !16
  %1755 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !16
  %1756 = bitcast <4 x i32> %1755 to <8 x half>, !dbg !16
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !16
  %address.local890 = getelementptr i8, ptr addrspace(3) %address.local889, i64 3072, !dbg !16
  %1757 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local890, i32 0) #0, !dbg !16
  %1758 = bitcast <4 x i32> %1757 to <8 x half>, !dbg !16
  %address.local891 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !16
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local891, i64 4096, !dbg !16
  %1759 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !16
  %1760 = bitcast <4 x i32> %1759 to <8 x half>, !dbg !16
  %address.local893 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !16
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local893, i64 5120, !dbg !16
  %1761 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !16
  %1762 = bitcast <4 x i32> %1761 to <8 x half>, !dbg !16
  %address.local895 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !16
  %address.local896 = getelementptr i8, ptr addrspace(3) %address.local895, i64 6144, !dbg !16
  %1763 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local896, i32 0) #0, !dbg !16
  %1764 = bitcast <4 x i32> %1763 to <8 x half>, !dbg !16
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !16
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local897, i64 7168, !dbg !16
  %1765 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !16
  %1766 = bitcast <4 x i32> %1765 to <8 x half>, !dbg !16
  %1767 = shufflevector <8 x half> %1752, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1767, <4 x float> zeroinitializer) #9, !dbg !16
  %1769 = shufflevector <8 x half> %1754, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1769, <4 x float> %1768) #9, !dbg !16
  %1771 = shufflevector <8 x half> %1756, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1772 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1771, <4 x float> %1770) #9, !dbg !16
  %1773 = shufflevector <8 x half> %1758, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1774 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1773, <4 x float> %1772) #9, !dbg !16
  %1775 = shufflevector <8 x half> %1760, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1775, <4 x float> %1774) #9, !dbg !16
  %1777 = shufflevector <8 x half> %1762, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1778 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1777, <4 x float> %1776) #9, !dbg !16
  %1779 = shufflevector <8 x half> %1764, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1780 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1779, <4 x float> %1778) #9, !dbg !16
  %1781 = shufflevector <8 x half> %1766, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1781, <4 x float> %1780) #9, !dbg !16
  %1783 = extractelement <4 x float> %1782, i64 0, !dbg !16
  %1784 = extractelement <4 x float> %1782, i64 1, !dbg !16
  %1785 = extractelement <4 x float> %1782, i64 2, !dbg !16
  %1786 = extractelement <4 x float> %1782, i64 3, !dbg !16
  %1787 = select i1 %.not, float %1784, float %1783, !dbg !16
  %1788 = bitcast float %1787 to i32, !dbg !16
  %1789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1788), !dbg !16
  %1790 = bitcast i32 %1789 to float, !dbg !16
  %1791 = select i1 %.not, float %1783, float %1790, !dbg !16
  %1792 = select i1 %.not, float %1790, float %1784, !dbg !16
  %1793 = select i1 %.not, float %1786, float %1785, !dbg !16
  %1794 = bitcast float %1793 to i32, !dbg !16
  %1795 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1794), !dbg !16
  %1796 = bitcast i32 %1795 to float, !dbg !16
  %1797 = select i1 %.not, float %1785, float %1796, !dbg !16
  %1798 = select i1 %.not, float %1796, float %1786, !dbg !16
  %1799 = select i1 %.not1, float %1797, float %1791, !dbg !16
  %1800 = bitcast float %1799 to i32, !dbg !16
  %1801 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1800), !dbg !16
  %1802 = bitcast i32 %1801 to float, !dbg !16
  %1803 = select i1 %.not1, float %1791, float %1802, !dbg !16
  %1804 = select i1 %.not1, float %1802, float %1797, !dbg !16
  %1805 = select i1 %.not1, float %1798, float %1792, !dbg !16
  %1806 = bitcast float %1805 to i32, !dbg !16
  %1807 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1806), !dbg !16
  %1808 = bitcast i32 %1807 to float, !dbg !16
  %1809 = select i1 %.not1, float %1792, float %1808, !dbg !16
  %1810 = select i1 %.not1, float %1808, float %1798, !dbg !16
  %1811 = shufflevector <8 x half> %1752, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1811, <4 x float> zeroinitializer) #9, !dbg !16
  %1813 = shufflevector <8 x half> %1754, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1813, <4 x float> %1812) #9, !dbg !16
  %1815 = shufflevector <8 x half> %1756, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1815, <4 x float> %1814) #9, !dbg !16
  %1817 = shufflevector <8 x half> %1758, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1817, <4 x float> %1816) #9, !dbg !16
  %1819 = shufflevector <8 x half> %1760, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1819, <4 x float> %1818) #9, !dbg !16
  %1821 = shufflevector <8 x half> %1762, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1821, <4 x float> %1820) #9, !dbg !16
  %1823 = shufflevector <8 x half> %1764, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1823, <4 x float> %1822) #9, !dbg !16
  %1825 = shufflevector <8 x half> %1766, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1825, <4 x float> %1824) #9, !dbg !16
  %1827 = extractelement <4 x float> %1826, i64 0, !dbg !16
  %1828 = extractelement <4 x float> %1826, i64 1, !dbg !16
  %1829 = extractelement <4 x float> %1826, i64 2, !dbg !16
  %1830 = extractelement <4 x float> %1826, i64 3, !dbg !16
  %1831 = select i1 %.not, float %1828, float %1827, !dbg !16
  %1832 = bitcast float %1831 to i32, !dbg !16
  %1833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1832), !dbg !16
  %1834 = bitcast i32 %1833 to float, !dbg !16
  %1835 = select i1 %.not, float %1827, float %1834, !dbg !16
  %1836 = select i1 %.not, float %1834, float %1828, !dbg !16
  %1837 = select i1 %.not, float %1830, float %1829, !dbg !16
  %1838 = bitcast float %1837 to i32, !dbg !16
  %1839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1838), !dbg !16
  %1840 = bitcast i32 %1839 to float, !dbg !16
  %1841 = select i1 %.not, float %1829, float %1840, !dbg !16
  %1842 = select i1 %.not, float %1840, float %1830, !dbg !16
  %1843 = select i1 %.not1, float %1841, float %1835, !dbg !16
  %1844 = bitcast float %1843 to i32, !dbg !16
  %1845 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1844), !dbg !16
  %1846 = bitcast i32 %1845 to float, !dbg !16
  %1847 = select i1 %.not1, float %1835, float %1846, !dbg !16
  %1848 = select i1 %.not1, float %1846, float %1841, !dbg !16
  %1849 = select i1 %.not1, float %1842, float %1836, !dbg !16
  %1850 = bitcast float %1849 to i32, !dbg !16
  %1851 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1850), !dbg !16
  %1852 = bitcast i32 %1851 to float, !dbg !16
  %1853 = select i1 %.not1, float %1836, float %1852, !dbg !16
  %1854 = select i1 %.not1, float %1852, float %1842, !dbg !16
  %1855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1767, <4 x float> zeroinitializer) #9, !dbg !16
  %1856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1769, <4 x float> %1855) #9, !dbg !16
  %1857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1771, <4 x float> %1856) #9, !dbg !16
  %1858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1773, <4 x float> %1857) #9, !dbg !16
  %1859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1775, <4 x float> %1858) #9, !dbg !16
  %1860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1777, <4 x float> %1859) #9, !dbg !16
  %1861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1779, <4 x float> %1860) #9, !dbg !16
  %1862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1781, <4 x float> %1861) #9, !dbg !16
  %1863 = extractelement <4 x float> %1862, i64 0, !dbg !16
  %1864 = extractelement <4 x float> %1862, i64 1, !dbg !16
  %1865 = extractelement <4 x float> %1862, i64 2, !dbg !16
  %1866 = extractelement <4 x float> %1862, i64 3, !dbg !16
  %1867 = select i1 %.not, float %1864, float %1863, !dbg !16
  %1868 = bitcast float %1867 to i32, !dbg !16
  %1869 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1868), !dbg !16
  %1870 = bitcast i32 %1869 to float, !dbg !16
  %1871 = select i1 %.not, float %1863, float %1870, !dbg !16
  %1872 = select i1 %.not, float %1870, float %1864, !dbg !16
  %1873 = select i1 %.not, float %1866, float %1865, !dbg !16
  %1874 = bitcast float %1873 to i32, !dbg !16
  %1875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1874), !dbg !16
  %1876 = bitcast i32 %1875 to float, !dbg !16
  %1877 = select i1 %.not, float %1865, float %1876, !dbg !16
  %1878 = select i1 %.not, float %1876, float %1866, !dbg !16
  %1879 = select i1 %.not1, float %1877, float %1871, !dbg !16
  %1880 = bitcast float %1879 to i32, !dbg !16
  %1881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1880), !dbg !16
  %1882 = bitcast i32 %1881 to float, !dbg !16
  %1883 = select i1 %.not1, float %1871, float %1882, !dbg !16
  %1884 = select i1 %.not1, float %1882, float %1877, !dbg !16
  %1885 = select i1 %.not1, float %1878, float %1872, !dbg !16
  %1886 = bitcast float %1885 to i32, !dbg !16
  %1887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1886), !dbg !16
  %1888 = bitcast i32 %1887 to float, !dbg !16
  %1889 = select i1 %.not1, float %1872, float %1888, !dbg !16
  %1890 = select i1 %.not1, float %1888, float %1878, !dbg !16
  %1891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1811, <4 x float> zeroinitializer) #9, !dbg !16
  %1892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1813, <4 x float> %1891) #9, !dbg !16
  %1893 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1815, <4 x float> %1892) #9, !dbg !16
  %1894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1817, <4 x float> %1893) #9, !dbg !16
  %1895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1819, <4 x float> %1894) #9, !dbg !16
  %1896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1821, <4 x float> %1895) #9, !dbg !16
  %1897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1823, <4 x float> %1896) #9, !dbg !16
  %1898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1825, <4 x float> %1897) #9, !dbg !16
  %1899 = extractelement <4 x float> %1898, i64 0, !dbg !16
  %1900 = extractelement <4 x float> %1898, i64 1, !dbg !16
  %1901 = extractelement <4 x float> %1898, i64 2, !dbg !16
  %1902 = extractelement <4 x float> %1898, i64 3, !dbg !16
  %1903 = select i1 %.not, float %1900, float %1899, !dbg !16
  %1904 = bitcast float %1903 to i32, !dbg !16
  %1905 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1904), !dbg !16
  %1906 = bitcast i32 %1905 to float, !dbg !16
  %1907 = select i1 %.not, float %1899, float %1906, !dbg !16
  %1908 = select i1 %.not, float %1906, float %1900, !dbg !16
  %1909 = select i1 %.not, float %1902, float %1901, !dbg !16
  %1910 = bitcast float %1909 to i32, !dbg !16
  %1911 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1910), !dbg !16
  %1912 = bitcast i32 %1911 to float, !dbg !16
  %1913 = select i1 %.not, float %1901, float %1912, !dbg !16
  %1914 = select i1 %.not, float %1912, float %1902, !dbg !16
  %1915 = select i1 %.not1, float %1913, float %1907, !dbg !16
  %1916 = bitcast float %1915 to i32, !dbg !16
  %1917 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1916), !dbg !16
  %1918 = bitcast i32 %1917 to float, !dbg !16
  %1919 = select i1 %.not1, float %1907, float %1918, !dbg !16
  %1920 = select i1 %.not1, float %1918, float %1913, !dbg !16
  %1921 = select i1 %.not1, float %1914, float %1908, !dbg !16
  %1922 = bitcast float %1921 to i32, !dbg !16
  %1923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1922), !dbg !16
  %1924 = bitcast i32 %1923 to float, !dbg !16
  %1925 = select i1 %.not1, float %1908, float %1924, !dbg !16
  %1926 = select i1 %.not1, float %1924, float %1914, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %44, label %1927, label %1928, !dbg !36

1927:                                             ; preds = %1680
  %address.thread899 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local900 = zext nneg i32 %address.thread899 to i64, !dbg !12
  %address.local901 = lshr i64 %address.local900, 6, !dbg !12
  %address.local902 = shl nuw nsw i64 %address.local901, 5, !dbg !36
  %address.local903 = add nuw nsw i64 %address.local902, %address.local900, !dbg !36
  %address.local904 = lshr i64 %address.local900, 4, !dbg !12
  %address.local905 = mul nsw i64 %address.local904, -16, !dbg !13
  %address.local906 = add nsw i64 %address.local903, %address.local905, !dbg !36
  %address.local907 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local906, !dbg !36
  store float %1228, ptr addrspace(3) %address.local907, align 4, !dbg !36
  %address.local908 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local906, !dbg !36
  %address.local909 = getelementptr i8, ptr addrspace(3) %address.local908, i64 64, !dbg !36
  store float %1231, ptr addrspace(3) %address.local909, align 4, !dbg !36
  br label %1928, !dbg !36

1928:                                             ; preds = %1927, %1680
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread910 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %lds.const.gep1734 = getelementptr i8, ptr addrspace(3) %address.local782, i64 0, !dbg !36
  %lds.const.gep1735 = getelementptr i8, ptr addrspace(3) %address.local782, i64 256, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %1931 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1686, !dbg !20
  %1932 = getelementptr i8, ptr addrspace(1) %1931, i64 128, !dbg !20
  %address.local924 = zext nneg i32 %address.thread910 to i64, !dbg !12
  %address.local925 = shl nuw nsw i64 %address.local924, 3, !dbg !11
  %address.local926 = and i64 %address.local925, 1536, !dbg !11
  %address.local927 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local926, !dbg !11
  %address.local928 = ptrtoint ptr addrspace(3) %address.local927 to i64, !dbg !11
  %address.local929 = trunc i64 %address.local928 to i32, !dbg !11
  %1933 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local929), !dbg !20
  %1934 = zext i32 %1933 to i64, !dbg !20
  %1935 = inttoptr i64 %1934 to ptr addrspace(3), !dbg !20
  %address.local930 = shl nuw nsw i64 %address.local924, 5, !dbg !20
  %address.local931 = and i64 %address.local930, 8064, !dbg !20
  %address.local932 = shl nuw nsw i64 %address.local924, 3, !dbg !20
  %address.local933 = and i64 %address.local932, 24, !dbg !20
  %address.local934 = or disjoint i64 %address.local931, %address.local933, !dbg !20
  %address.local935 = getelementptr [2 x i8], ptr addrspace(1) %1932, i64 %address.local934, !dbg !20
  %1936 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1937 = ptrtoint ptr addrspace(1) %address.local935 to i64, !dbg !20
  %1938 = sub i64 %1937, %1936, !dbg !20
  %dma.buffer.offset1746 = trunc i64 %1938 to i32, !dbg !20
  %1939 = trunc i64 %1936 to i32, !dbg !20
  %1940 = insertelement <4 x i32> poison, i32 %1939, i64 0, !dbg !20
  %1941 = lshr i64 %1936, 32, !dbg !20
  %1942 = trunc i64 %1941 to i32, !dbg !20
  %1943 = insertelement <4 x i32> %1940, i32 %1942, i64 1, !dbg !20
  %1944 = insertelement <4 x i32> %1943, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1747 = insertelement <4 x i32> %1944, i32 131072, i64 3, !dbg !20
  %1945 = ptrtoint ptr addrspace(3) %1935 to i32, !dbg !20
  %dma.buffer.lds1748 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1945), !dbg !20
  %1946 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1748, i32 %dma.buffer.offset1746, <4 x i32> %dma.buffer.rsrc1747) #10, !dbg !20
  %address.local936 = or disjoint i64 %address.local924, 256, !dbg !11
  %address.local937 = shl nuw nsw i64 %address.local936, 3, !dbg !11
  %address.local938 = and i64 %address.local937, 3584, !dbg !11
  %address.local939 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !11
  %address.local940 = ptrtoint ptr addrspace(3) %address.local939 to i64, !dbg !11
  %address.local941 = trunc i64 %address.local940 to i32, !dbg !11
  %1947 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local941), !dbg !20
  %1948 = zext i32 %1947 to i64, !dbg !20
  %1949 = inttoptr i64 %1948 to ptr addrspace(3), !dbg !20
  %address.local942 = shl nuw nsw i64 %address.local936, 5, !dbg !20
  %address.local943 = and i64 %address.local942, 16256, !dbg !20
  %address.local944 = shl nuw nsw i64 %address.local924, 3, !dbg !20
  %address.local945 = and i64 %address.local944, 24, !dbg !20
  %address.local946 = or disjoint i64 %address.local943, %address.local945, !dbg !20
  %address.local947 = getelementptr [2 x i8], ptr addrspace(1) %1932, i64 %address.local946, !dbg !20
  %1950 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %1951 = ptrtoint ptr addrspace(1) %address.local947 to i64, !dbg !20
  %1952 = sub i64 %1951, %1950, !dbg !20
  %dma.buffer.offset1743 = trunc i64 %1952 to i32, !dbg !20
  %1953 = trunc i64 %1950 to i32, !dbg !20
  %1954 = insertelement <4 x i32> poison, i32 %1953, i64 0, !dbg !20
  %1955 = lshr i64 %1950, 32, !dbg !20
  %1956 = trunc i64 %1955 to i32, !dbg !20
  %1957 = insertelement <4 x i32> %1954, i32 %1956, i64 1, !dbg !20
  %1958 = insertelement <4 x i32> %1957, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1744 = insertelement <4 x i32> %1958, i32 131072, i64 3, !dbg !20
  %1959 = ptrtoint ptr addrspace(3) %1949 to i32, !dbg !20
  %dma.buffer.lds1745 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1959), !dbg !20
  %1960 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1745, i32 %dma.buffer.offset1743, <4 x i32> %dma.buffer.rsrc1744) #10, !dbg !20
  %1961 = extractvalue [2 x <8 x float>] %66, 0, !dbg !36
  %1962 = shufflevector <8 x float> %1961, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1963 = insertelement <4 x float> poison, float %1228, i64 0, !dbg !36
  %1964 = shufflevector <4 x float> %1963, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1965 = fmul <4 x float> %1962, %1964, !dbg !36
  %1966 = insertelement <4 x float> poison, float %1803, i64 0, !dbg !37
  %1967 = insertelement <4 x float> %1966, float %1809, i64 1, !dbg !37
  %1968 = insertelement <4 x float> %1967, float %1804, i64 2, !dbg !37
  %1969 = insertelement <4 x float> %1968, float %1810, i64 3, !dbg !37
  %1970 = fadd <4 x float> %1965, %1969, !dbg !37
  %1971 = shufflevector <8 x float> %1961, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1972 = fmul <4 x float> %1971, %1964, !dbg !36
  %1973 = insertelement <4 x float> poison, float %1847, i64 0, !dbg !37
  %1974 = insertelement <4 x float> %1973, float %1853, i64 1, !dbg !37
  %1975 = insertelement <4 x float> %1974, float %1848, i64 2, !dbg !37
  %1976 = insertelement <4 x float> %1975, float %1854, i64 3, !dbg !37
  %1977 = fadd <4 x float> %1972, %1976, !dbg !37
  %1978 = extractvalue [2 x <8 x float>] %66, 1, !dbg !36
  %1979 = shufflevector <8 x float> %1978, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1980 = insertelement <4 x float> poison, float %1231, i64 0, !dbg !36
  %1981 = shufflevector <4 x float> %1980, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1982 = fmul <4 x float> %1979, %1981, !dbg !36
  %1983 = insertelement <4 x float> poison, float %1883, i64 0, !dbg !37
  %1984 = insertelement <4 x float> %1983, float %1889, i64 1, !dbg !37
  %1985 = insertelement <4 x float> %1984, float %1884, i64 2, !dbg !37
  %1986 = insertelement <4 x float> %1985, float %1890, i64 3, !dbg !37
  %1987 = fadd <4 x float> %1982, %1986, !dbg !37
  %1988 = shufflevector <8 x float> %1978, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1989 = fmul <4 x float> %1988, %1981, !dbg !36
  %1990 = insertelement <4 x float> poison, float %1919, i64 0, !dbg !37
  %1991 = insertelement <4 x float> %1990, float %1925, i64 1, !dbg !37
  %1992 = insertelement <4 x float> %1991, float %1920, i64 2, !dbg !37
  %1993 = insertelement <4 x float> %1992, float %1926, i64 3, !dbg !37
  %1994 = fadd <4 x float> %1989, %1993, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local948 = shl nuw nsw i32 %address.thread910, 3, !dbg !16
  %address.local949 = and i32 %address.local948, 504, !dbg !16
  %address.local950 = zext nneg i32 %address.local949 to i64, !dbg !16
  %address.local951 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local950, !dbg !16
  %1995 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local951, i32 0) #0, !dbg !16
  %1996 = bitcast <4 x i32> %1995 to <8 x half>, !dbg !16
  %address.local952 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local950, !dbg !16
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local952, i64 1024, !dbg !16
  %1997 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !16
  %1998 = bitcast <4 x i32> %1997 to <8 x half>, !dbg !16
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local950, !dbg !16
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local954, i64 2048, !dbg !16
  %1999 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !16
  %2000 = bitcast <4 x i32> %1999 to <8 x half>, !dbg !16
  %address.local956 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local950, !dbg !16
  %address.local957 = getelementptr i8, ptr addrspace(3) %address.local956, i64 3072, !dbg !16
  %2001 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local957, i32 0) #0, !dbg !16
  %2002 = bitcast <4 x i32> %2001 to <8 x half>, !dbg !16
  %address.local958 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local950, !dbg !16
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local958, i64 4096, !dbg !16
  %2003 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !16
  %2004 = bitcast <4 x i32> %2003 to <8 x half>, !dbg !16
  %address.local960 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local950, !dbg !16
  %address.local961 = getelementptr i8, ptr addrspace(3) %address.local960, i64 5120, !dbg !16
  %2005 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local961, i32 0) #0, !dbg !16
  %2006 = bitcast <4 x i32> %2005 to <8 x half>, !dbg !16
  %address.local962 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local950, !dbg !16
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local962, i64 6144, !dbg !16
  %2007 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !16
  %2008 = bitcast <4 x i32> %2007 to <8 x half>, !dbg !16
  %address.local964 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local950, !dbg !16
  %address.local965 = getelementptr i8, ptr addrspace(3) %address.local964, i64 7168, !dbg !16
  %2009 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local965, i32 0) #0, !dbg !16
  %2010 = bitcast <4 x i32> %2009 to <8 x half>, !dbg !16
  %2011 = shufflevector <8 x half> %1996, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2011, <4 x float> zeroinitializer) #9, !dbg !16
  %2013 = shufflevector <8 x half> %1998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %2013, <4 x float> %2012) #9, !dbg !16
  %2015 = shufflevector <8 x half> %2000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %2015, <4 x float> %2014) #9, !dbg !16
  %2017 = shufflevector <8 x half> %2002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %2017, <4 x float> %2016) #9, !dbg !16
  %2019 = shufflevector <8 x half> %2004, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %2019, <4 x float> %2018) #9, !dbg !16
  %2021 = shufflevector <8 x half> %2006, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %2021, <4 x float> %2020) #9, !dbg !16
  %2023 = shufflevector <8 x half> %2008, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %2023, <4 x float> %2022) #9, !dbg !16
  %2025 = shufflevector <8 x half> %2010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %2025, <4 x float> %2024) #9, !dbg !16
  %2027 = extractelement <4 x float> %2026, i64 0, !dbg !16
  %2028 = extractelement <4 x float> %2026, i64 1, !dbg !16
  %2029 = extractelement <4 x float> %2026, i64 2, !dbg !16
  %2030 = extractelement <4 x float> %2026, i64 3, !dbg !16
  %2031 = select i1 %.not, float %2028, float %2027, !dbg !16
  %2032 = bitcast float %2031 to i32, !dbg !16
  %2033 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2032), !dbg !16
  %2034 = bitcast i32 %2033 to float, !dbg !16
  %2035 = select i1 %.not, float %2027, float %2034, !dbg !16
  %2036 = select i1 %.not, float %2034, float %2028, !dbg !16
  %2037 = select i1 %.not, float %2030, float %2029, !dbg !16
  %2038 = bitcast float %2037 to i32, !dbg !16
  %2039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2038), !dbg !16
  %2040 = bitcast i32 %2039 to float, !dbg !16
  %2041 = select i1 %.not, float %2029, float %2040, !dbg !16
  %2042 = select i1 %.not, float %2040, float %2030, !dbg !16
  %2043 = select i1 %.not1, float %2041, float %2035, !dbg !16
  %2044 = bitcast float %2043 to i32, !dbg !16
  %2045 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2044), !dbg !16
  %2046 = bitcast i32 %2045 to float, !dbg !16
  %2047 = select i1 %.not1, float %2035, float %2046, !dbg !16
  %2048 = select i1 %.not1, float %2046, float %2041, !dbg !16
  %2049 = select i1 %.not1, float %2042, float %2036, !dbg !16
  %2050 = bitcast float %2049 to i32, !dbg !16
  %2051 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2050), !dbg !16
  %2052 = bitcast i32 %2051 to float, !dbg !16
  %2053 = select i1 %.not1, float %2036, float %2052, !dbg !16
  %2054 = select i1 %.not1, float %2052, float %2042, !dbg !16
  %2055 = shufflevector <8 x half> %1996, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2055, <4 x float> zeroinitializer) #9, !dbg !16
  %2057 = shufflevector <8 x half> %1998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %2057, <4 x float> %2056) #9, !dbg !16
  %2059 = shufflevector <8 x half> %2000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %2059, <4 x float> %2058) #9, !dbg !16
  %2061 = shufflevector <8 x half> %2002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %2061, <4 x float> %2060) #9, !dbg !16
  %2063 = shufflevector <8 x half> %2004, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %2063, <4 x float> %2062) #9, !dbg !16
  %2065 = shufflevector <8 x half> %2006, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %2065, <4 x float> %2064) #9, !dbg !16
  %2067 = shufflevector <8 x half> %2008, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %2067, <4 x float> %2066) #9, !dbg !16
  %2069 = shufflevector <8 x half> %2010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %2069, <4 x float> %2068) #9, !dbg !16
  %2071 = extractelement <4 x float> %2070, i64 0, !dbg !16
  %2072 = extractelement <4 x float> %2070, i64 1, !dbg !16
  %2073 = extractelement <4 x float> %2070, i64 2, !dbg !16
  %2074 = extractelement <4 x float> %2070, i64 3, !dbg !16
  %2075 = select i1 %.not, float %2072, float %2071, !dbg !16
  %2076 = bitcast float %2075 to i32, !dbg !16
  %2077 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2076), !dbg !16
  %2078 = bitcast i32 %2077 to float, !dbg !16
  %2079 = select i1 %.not, float %2071, float %2078, !dbg !16
  %2080 = select i1 %.not, float %2078, float %2072, !dbg !16
  %2081 = select i1 %.not, float %2074, float %2073, !dbg !16
  %2082 = bitcast float %2081 to i32, !dbg !16
  %2083 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2082), !dbg !16
  %2084 = bitcast i32 %2083 to float, !dbg !16
  %2085 = select i1 %.not, float %2073, float %2084, !dbg !16
  %2086 = select i1 %.not, float %2084, float %2074, !dbg !16
  %2087 = select i1 %.not1, float %2085, float %2079, !dbg !16
  %2088 = bitcast float %2087 to i32, !dbg !16
  %2089 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2088), !dbg !16
  %2090 = bitcast i32 %2089 to float, !dbg !16
  %2091 = select i1 %.not1, float %2079, float %2090, !dbg !16
  %2092 = select i1 %.not1, float %2090, float %2085, !dbg !16
  %2093 = select i1 %.not1, float %2086, float %2080, !dbg !16
  %2094 = bitcast float %2093 to i32, !dbg !16
  %2095 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2094), !dbg !16
  %2096 = bitcast i32 %2095 to float, !dbg !16
  %2097 = select i1 %.not1, float %2080, float %2096, !dbg !16
  %2098 = select i1 %.not1, float %2096, float %2086, !dbg !16
  %2099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %2011, <4 x float> zeroinitializer) #9, !dbg !16
  %2100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %2013, <4 x float> %2099) #9, !dbg !16
  %2101 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %2015, <4 x float> %2100) #9, !dbg !16
  %2102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %2017, <4 x float> %2101) #9, !dbg !16
  %2103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %2019, <4 x float> %2102) #9, !dbg !16
  %2104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %2021, <4 x float> %2103) #9, !dbg !16
  %2105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %2023, <4 x float> %2104) #9, !dbg !16
  %2106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %2025, <4 x float> %2105) #9, !dbg !16
  %2107 = extractelement <4 x float> %2106, i64 0, !dbg !16
  %2108 = extractelement <4 x float> %2106, i64 1, !dbg !16
  %2109 = extractelement <4 x float> %2106, i64 2, !dbg !16
  %2110 = extractelement <4 x float> %2106, i64 3, !dbg !16
  %2111 = select i1 %.not, float %2108, float %2107, !dbg !16
  %2112 = bitcast float %2111 to i32, !dbg !16
  %2113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2112), !dbg !16
  %2114 = bitcast i32 %2113 to float, !dbg !16
  %2115 = select i1 %.not, float %2107, float %2114, !dbg !16
  %2116 = select i1 %.not, float %2114, float %2108, !dbg !16
  %2117 = select i1 %.not, float %2110, float %2109, !dbg !16
  %2118 = bitcast float %2117 to i32, !dbg !16
  %2119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2118), !dbg !16
  %2120 = bitcast i32 %2119 to float, !dbg !16
  %2121 = select i1 %.not, float %2109, float %2120, !dbg !16
  %2122 = select i1 %.not, float %2120, float %2110, !dbg !16
  %2123 = select i1 %.not1, float %2121, float %2115, !dbg !16
  %2124 = bitcast float %2123 to i32, !dbg !16
  %2125 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2124), !dbg !16
  %2126 = bitcast i32 %2125 to float, !dbg !16
  %2127 = select i1 %.not1, float %2115, float %2126, !dbg !16
  %2128 = select i1 %.not1, float %2126, float %2121, !dbg !16
  %2129 = select i1 %.not1, float %2122, float %2116, !dbg !16
  %2130 = bitcast float %2129 to i32, !dbg !16
  %2131 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2130), !dbg !16
  %2132 = bitcast i32 %2131 to float, !dbg !16
  %2133 = select i1 %.not1, float %2116, float %2132, !dbg !16
  %2134 = select i1 %.not1, float %2132, float %2122, !dbg !16
  %2135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %2055, <4 x float> zeroinitializer) #9, !dbg !16
  %2136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %2057, <4 x float> %2135) #9, !dbg !16
  %2137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %2059, <4 x float> %2136) #9, !dbg !16
  %2138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %2061, <4 x float> %2137) #9, !dbg !16
  %2139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %2063, <4 x float> %2138) #9, !dbg !16
  %2140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %2065, <4 x float> %2139) #9, !dbg !16
  %2141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %2067, <4 x float> %2140) #9, !dbg !16
  %2142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %2069, <4 x float> %2141) #9, !dbg !16
  %2143 = extractelement <4 x float> %2142, i64 0, !dbg !16
  %2144 = extractelement <4 x float> %2142, i64 1, !dbg !16
  %2145 = extractelement <4 x float> %2142, i64 2, !dbg !16
  %2146 = extractelement <4 x float> %2142, i64 3, !dbg !16
  %2147 = select i1 %.not, float %2144, float %2143, !dbg !16
  %2148 = bitcast float %2147 to i32, !dbg !16
  %2149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2148), !dbg !16
  %2150 = bitcast i32 %2149 to float, !dbg !16
  %2151 = select i1 %.not, float %2143, float %2150, !dbg !16
  %2152 = select i1 %.not, float %2150, float %2144, !dbg !16
  %2153 = select i1 %.not, float %2146, float %2145, !dbg !16
  %2154 = bitcast float %2153 to i32, !dbg !16
  %2155 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2154), !dbg !16
  %2156 = bitcast i32 %2155 to float, !dbg !16
  %2157 = select i1 %.not, float %2145, float %2156, !dbg !16
  %2158 = select i1 %.not, float %2156, float %2146, !dbg !16
  %2159 = select i1 %.not1, float %2157, float %2151, !dbg !16
  %2160 = bitcast float %2159 to i32, !dbg !16
  %2161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2160), !dbg !16
  %2162 = bitcast i32 %2161 to float, !dbg !16
  %2163 = select i1 %.not1, float %2151, float %2162, !dbg !16
  %2164 = select i1 %.not1, float %2162, float %2157, !dbg !16
  %2165 = select i1 %.not1, float %2158, float %2152, !dbg !16
  %2166 = bitcast float %2165 to i32, !dbg !16
  %2167 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2166), !dbg !16
  %2168 = bitcast i32 %2167 to float, !dbg !16
  %2169 = select i1 %.not1, float %2152, float %2168, !dbg !16
  %2170 = select i1 %.not1, float %2168, float %2158, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %44, label %2171, label %2172, !dbg !36

2171:                                             ; preds = %1928
  %address.thread966 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local967 = zext nneg i32 %address.thread966 to i64, !dbg !12
  %address.local968 = lshr i64 %address.local967, 6, !dbg !12
  %address.local969 = shl nuw nsw i64 %address.local968, 5, !dbg !36
  %address.local970 = add nuw nsw i64 %address.local969, %address.local967, !dbg !36
  %address.local971 = lshr i64 %address.local967, 4, !dbg !12
  %address.local972 = mul nsw i64 %address.local971, -16, !dbg !13
  %address.local973 = add nsw i64 %address.local970, %address.local972, !dbg !36
  %address.local974 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local973, !dbg !36
  store float %1228, ptr addrspace(3) %address.local974, align 4, !dbg !36
  %address.local975 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local973, !dbg !36
  %address.local976 = getelementptr i8, ptr addrspace(3) %address.local975, i64 64, !dbg !36
  store float %1231, ptr addrspace(3) %address.local976, align 4, !dbg !36
  br label %2172, !dbg !36

2172:                                             ; preds = %2171, %1928
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.thread977 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %lds.const.gep1736 = getelementptr i8, ptr addrspace(3) %address.local782, i64 0, !dbg !36
  %lds.const.gep1737 = getelementptr i8, ptr addrspace(3) %address.local782, i64 256, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %2175 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1686, !dbg !20
  %2176 = getelementptr i8, ptr addrspace(1) %2175, i64 192, !dbg !20
  %address.local991 = zext nneg i32 %address.thread977 to i64, !dbg !12
  %address.local992 = shl nuw nsw i64 %address.local991, 3, !dbg !11
  %address.local993 = and i64 %address.local992, 1536, !dbg !11
  %address.local994 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local993, !dbg !11
  %address.local995 = ptrtoint ptr addrspace(3) %address.local994 to i64, !dbg !11
  %address.local996 = trunc i64 %address.local995 to i32, !dbg !11
  %2177 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local996), !dbg !20
  %2178 = zext i32 %2177 to i64, !dbg !20
  %2179 = inttoptr i64 %2178 to ptr addrspace(3), !dbg !20
  %address.local997 = shl nuw nsw i64 %address.local991, 5, !dbg !20
  %address.local998 = and i64 %address.local997, 8064, !dbg !20
  %address.local999 = shl nuw nsw i64 %address.local991, 3, !dbg !20
  %address.local1000 = and i64 %address.local999, 24, !dbg !20
  %address.local1001 = or disjoint i64 %address.local998, %address.local1000, !dbg !20
  %address.local1002 = getelementptr [2 x i8], ptr addrspace(1) %2176, i64 %address.local1001, !dbg !20
  %2180 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %2181 = ptrtoint ptr addrspace(1) %address.local1002 to i64, !dbg !20
  %2182 = sub i64 %2181, %2180, !dbg !20
  %dma.buffer.offset1740 = trunc i64 %2182 to i32, !dbg !20
  %2183 = trunc i64 %2180 to i32, !dbg !20
  %2184 = insertelement <4 x i32> poison, i32 %2183, i64 0, !dbg !20
  %2185 = lshr i64 %2180, 32, !dbg !20
  %2186 = trunc i64 %2185 to i32, !dbg !20
  %2187 = insertelement <4 x i32> %2184, i32 %2186, i64 1, !dbg !20
  %2188 = insertelement <4 x i32> %2187, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc1741 = insertelement <4 x i32> %2188, i32 131072, i64 3, !dbg !20
  %2189 = ptrtoint ptr addrspace(3) %2179 to i32, !dbg !20
  %dma.buffer.lds1742 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %2189), !dbg !20
  %2190 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1742, i32 %dma.buffer.offset1740, <4 x i32> %dma.buffer.rsrc1741) #10, !dbg !20
  %address.local1003 = or disjoint i64 %address.local991, 256, !dbg !11
  %address.local1004 = shl nuw nsw i64 %address.local1003, 3, !dbg !11
  %address.local1005 = and i64 %address.local1004, 3584, !dbg !11
  %address.local1006 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1005, !dbg !11
  %address.local1007 = ptrtoint ptr addrspace(3) %address.local1006 to i64, !dbg !11
  %address.local1008 = trunc i64 %address.local1007 to i32, !dbg !11
  %2191 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1008), !dbg !20
  %2192 = zext i32 %2191 to i64, !dbg !20
  %2193 = inttoptr i64 %2192 to ptr addrspace(3), !dbg !20
  %address.local1009 = shl nuw nsw i64 %address.local1003, 5, !dbg !20
  %address.local1010 = and i64 %address.local1009, 16256, !dbg !20
  %address.local1011 = shl nuw nsw i64 %address.local991, 3, !dbg !20
  %address.local1012 = and i64 %address.local1011, 24, !dbg !20
  %address.local1013 = or disjoint i64 %address.local1010, %address.local1012, !dbg !20
  %address.local1014 = getelementptr [2 x i8], ptr addrspace(1) %2176, i64 %address.local1013, !dbg !20
  %2194 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !20
  %2195 = ptrtoint ptr addrspace(1) %address.local1014 to i64, !dbg !20
  %2196 = sub i64 %2195, %2194, !dbg !20
  %dma.buffer.offset = trunc i64 %2196 to i32, !dbg !20
  %2197 = trunc i64 %2194 to i32, !dbg !20
  %2198 = insertelement <4 x i32> poison, i32 %2197, i64 0, !dbg !20
  %2199 = lshr i64 %2194, 32, !dbg !20
  %2200 = trunc i64 %2199 to i32, !dbg !20
  %2201 = insertelement <4 x i32> %2198, i32 %2200, i64 1, !dbg !20
  %2202 = insertelement <4 x i32> %2201, i32 33554432, i64 2, !dbg !20
  %dma.buffer.rsrc = insertelement <4 x i32> %2202, i32 131072, i64 3, !dbg !20
  %2203 = ptrtoint ptr addrspace(3) %2193 to i32, !dbg !20
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %2203), !dbg !20
  %2204 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !20
  %2205 = extractvalue [2 x <8 x float>] %67, 0, !dbg !36
  %2206 = shufflevector <8 x float> %2205, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2207 = insertelement <4 x float> poison, float %1228, i64 0, !dbg !36
  %2208 = shufflevector <4 x float> %2207, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2209 = fmul <4 x float> %2206, %2208, !dbg !36
  %2210 = insertelement <4 x float> poison, float %2047, i64 0, !dbg !37
  %2211 = insertelement <4 x float> %2210, float %2053, i64 1, !dbg !37
  %2212 = insertelement <4 x float> %2211, float %2048, i64 2, !dbg !37
  %2213 = insertelement <4 x float> %2212, float %2054, i64 3, !dbg !37
  %2214 = fadd <4 x float> %2209, %2213, !dbg !37
  %2215 = shufflevector <8 x float> %2205, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2216 = fmul <4 x float> %2215, %2208, !dbg !36
  %2217 = insertelement <4 x float> poison, float %2091, i64 0, !dbg !37
  %2218 = insertelement <4 x float> %2217, float %2097, i64 1, !dbg !37
  %2219 = insertelement <4 x float> %2218, float %2092, i64 2, !dbg !37
  %2220 = insertelement <4 x float> %2219, float %2098, i64 3, !dbg !37
  %2221 = fadd <4 x float> %2216, %2220, !dbg !37
  %2222 = extractvalue [2 x <8 x float>] %67, 1, !dbg !36
  %2223 = shufflevector <8 x float> %2222, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2224 = insertelement <4 x float> poison, float %1231, i64 0, !dbg !36
  %2225 = shufflevector <4 x float> %2224, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2226 = fmul <4 x float> %2223, %2225, !dbg !36
  %2227 = insertelement <4 x float> poison, float %2127, i64 0, !dbg !37
  %2228 = insertelement <4 x float> %2227, float %2133, i64 1, !dbg !37
  %2229 = insertelement <4 x float> %2228, float %2128, i64 2, !dbg !37
  %2230 = insertelement <4 x float> %2229, float %2134, i64 3, !dbg !37
  %2231 = fadd <4 x float> %2226, %2230, !dbg !37
  %2232 = shufflevector <8 x float> %2222, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2233 = fmul <4 x float> %2232, %2225, !dbg !36
  %2234 = insertelement <4 x float> poison, float %2163, i64 0, !dbg !37
  %2235 = insertelement <4 x float> %2234, float %2169, i64 1, !dbg !37
  %2236 = insertelement <4 x float> %2235, float %2164, i64 2, !dbg !37
  %2237 = insertelement <4 x float> %2236, float %2170, i64 3, !dbg !37
  %2238 = fadd <4 x float> %2233, %2237, !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local1015 = shl nuw nsw i32 %address.thread977, 3, !dbg !16
  %address.local1016 = and i32 %address.local1015, 504, !dbg !16
  %address.local1017 = zext nneg i32 %address.local1016 to i64, !dbg !16
  %address.local1018 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1017, !dbg !16
  %2239 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1018, i32 0) #0, !dbg !16
  %2240 = bitcast <4 x i32> %2239 to <8 x half>, !dbg !16
  %address.local1019 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1017, !dbg !16
  %address.local1020 = getelementptr i8, ptr addrspace(3) %address.local1019, i64 1024, !dbg !16
  %2241 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #0, !dbg !16
  %2242 = bitcast <4 x i32> %2241 to <8 x half>, !dbg !16
  %address.local1021 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1017, !dbg !16
  %address.local1022 = getelementptr i8, ptr addrspace(3) %address.local1021, i64 2048, !dbg !16
  %2243 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #0, !dbg !16
  %2244 = bitcast <4 x i32> %2243 to <8 x half>, !dbg !16
  %address.local1023 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1017, !dbg !16
  %address.local1024 = getelementptr i8, ptr addrspace(3) %address.local1023, i64 3072, !dbg !16
  %2245 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1024, i32 0) #0, !dbg !16
  %2246 = bitcast <4 x i32> %2245 to <8 x half>, !dbg !16
  %address.local1025 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1017, !dbg !16
  %address.local1026 = getelementptr i8, ptr addrspace(3) %address.local1025, i64 4096, !dbg !16
  %2247 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #0, !dbg !16
  %2248 = bitcast <4 x i32> %2247 to <8 x half>, !dbg !16
  %address.local1027 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1017, !dbg !16
  %address.local1028 = getelementptr i8, ptr addrspace(3) %address.local1027, i64 5120, !dbg !16
  %2249 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1028, i32 0) #0, !dbg !16
  %2250 = bitcast <4 x i32> %2249 to <8 x half>, !dbg !16
  %address.local1029 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1017, !dbg !16
  %address.local1030 = getelementptr i8, ptr addrspace(3) %address.local1029, i64 6144, !dbg !16
  %2251 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !16
  %2252 = bitcast <4 x i32> %2251 to <8 x half>, !dbg !16
  %address.local1031 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1017, !dbg !16
  %address.local1032 = getelementptr i8, ptr addrspace(3) %address.local1031, i64 7168, !dbg !16
  %2253 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1032, i32 0) #0, !dbg !16
  %2254 = bitcast <4 x i32> %2253 to <8 x half>, !dbg !16
  %2255 = shufflevector <8 x half> %2240, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2255, <4 x float> zeroinitializer) #9, !dbg !16
  %2257 = shufflevector <8 x half> %2242, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %2257, <4 x float> %2256) #9, !dbg !16
  %2259 = shufflevector <8 x half> %2244, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %2259, <4 x float> %2258) #9, !dbg !16
  %2261 = shufflevector <8 x half> %2246, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %2261, <4 x float> %2260) #9, !dbg !16
  %2263 = shufflevector <8 x half> %2248, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %2263, <4 x float> %2262) #9, !dbg !16
  %2265 = shufflevector <8 x half> %2250, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %2265, <4 x float> %2264) #9, !dbg !16
  %2267 = shufflevector <8 x half> %2252, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %2267, <4 x float> %2266) #9, !dbg !16
  %2269 = shufflevector <8 x half> %2254, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %2270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %2269, <4 x float> %2268) #9, !dbg !16
  %2271 = extractelement <4 x float> %2270, i64 0, !dbg !16
  %2272 = extractelement <4 x float> %2270, i64 1, !dbg !16
  %2273 = extractelement <4 x float> %2270, i64 2, !dbg !16
  %2274 = extractelement <4 x float> %2270, i64 3, !dbg !16
  %2275 = select i1 %.not, float %2272, float %2271, !dbg !16
  %2276 = bitcast float %2275 to i32, !dbg !16
  %2277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2276), !dbg !16
  %2278 = bitcast i32 %2277 to float, !dbg !16
  %2279 = select i1 %.not, float %2271, float %2278, !dbg !16
  %2280 = select i1 %.not, float %2278, float %2272, !dbg !16
  %2281 = select i1 %.not, float %2274, float %2273, !dbg !16
  %2282 = bitcast float %2281 to i32, !dbg !16
  %2283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2282), !dbg !16
  %2284 = bitcast i32 %2283 to float, !dbg !16
  %2285 = select i1 %.not, float %2273, float %2284, !dbg !16
  %2286 = select i1 %.not, float %2284, float %2274, !dbg !16
  %2287 = select i1 %.not1, float %2285, float %2279, !dbg !16
  %2288 = bitcast float %2287 to i32, !dbg !16
  %2289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2288), !dbg !16
  %2290 = bitcast i32 %2289 to float, !dbg !16
  %2291 = select i1 %.not1, float %2279, float %2290, !dbg !16
  %2292 = select i1 %.not1, float %2290, float %2285, !dbg !16
  %2293 = select i1 %.not1, float %2286, float %2280, !dbg !16
  %2294 = bitcast float %2293 to i32, !dbg !16
  %2295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2294), !dbg !16
  %2296 = bitcast i32 %2295 to float, !dbg !16
  %2297 = select i1 %.not1, float %2280, float %2296, !dbg !16
  %2298 = select i1 %.not1, float %2296, float %2286, !dbg !16
  %2299 = shufflevector <8 x half> %2240, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2299, <4 x float> zeroinitializer) #9, !dbg !16
  %2301 = shufflevector <8 x half> %2242, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %2301, <4 x float> %2300) #9, !dbg !16
  %2303 = shufflevector <8 x half> %2244, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %2303, <4 x float> %2302) #9, !dbg !16
  %2305 = shufflevector <8 x half> %2246, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %2305, <4 x float> %2304) #9, !dbg !16
  %2307 = shufflevector <8 x half> %2248, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %2307, <4 x float> %2306) #9, !dbg !16
  %2309 = shufflevector <8 x half> %2250, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %2309, <4 x float> %2308) #9, !dbg !16
  %2311 = shufflevector <8 x half> %2252, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %2311, <4 x float> %2310) #9, !dbg !16
  %2313 = shufflevector <8 x half> %2254, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %2314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %2313, <4 x float> %2312) #9, !dbg !16
  %2315 = extractelement <4 x float> %2314, i64 0, !dbg !16
  %2316 = extractelement <4 x float> %2314, i64 1, !dbg !16
  %2317 = extractelement <4 x float> %2314, i64 2, !dbg !16
  %2318 = extractelement <4 x float> %2314, i64 3, !dbg !16
  %2319 = select i1 %.not, float %2316, float %2315, !dbg !16
  %2320 = bitcast float %2319 to i32, !dbg !16
  %2321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2320), !dbg !16
  %2322 = bitcast i32 %2321 to float, !dbg !16
  %2323 = select i1 %.not, float %2315, float %2322, !dbg !16
  %2324 = select i1 %.not, float %2322, float %2316, !dbg !16
  %2325 = select i1 %.not, float %2318, float %2317, !dbg !16
  %2326 = bitcast float %2325 to i32, !dbg !16
  %2327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2326), !dbg !16
  %2328 = bitcast i32 %2327 to float, !dbg !16
  %2329 = select i1 %.not, float %2317, float %2328, !dbg !16
  %2330 = select i1 %.not, float %2328, float %2318, !dbg !16
  %2331 = select i1 %.not1, float %2329, float %2323, !dbg !16
  %2332 = bitcast float %2331 to i32, !dbg !16
  %2333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2332), !dbg !16
  %2334 = bitcast i32 %2333 to float, !dbg !16
  %2335 = select i1 %.not1, float %2323, float %2334, !dbg !16
  %2336 = select i1 %.not1, float %2334, float %2329, !dbg !16
  %2337 = select i1 %.not1, float %2330, float %2324, !dbg !16
  %2338 = bitcast float %2337 to i32, !dbg !16
  %2339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2338), !dbg !16
  %2340 = bitcast i32 %2339 to float, !dbg !16
  %2341 = select i1 %.not1, float %2324, float %2340, !dbg !16
  %2342 = select i1 %.not1, float %2340, float %2330, !dbg !16
  %2343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %2255, <4 x float> zeroinitializer) #9, !dbg !16
  %2344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %2257, <4 x float> %2343) #9, !dbg !16
  %2345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %2259, <4 x float> %2344) #9, !dbg !16
  %2346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %2261, <4 x float> %2345) #9, !dbg !16
  %2347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %2263, <4 x float> %2346) #9, !dbg !16
  %2348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %2265, <4 x float> %2347) #9, !dbg !16
  %2349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %2267, <4 x float> %2348) #9, !dbg !16
  %2350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %2269, <4 x float> %2349) #9, !dbg !16
  %2351 = extractelement <4 x float> %2350, i64 0, !dbg !16
  %2352 = extractelement <4 x float> %2350, i64 1, !dbg !16
  %2353 = extractelement <4 x float> %2350, i64 2, !dbg !16
  %2354 = extractelement <4 x float> %2350, i64 3, !dbg !16
  %2355 = select i1 %.not, float %2352, float %2351, !dbg !16
  %2356 = bitcast float %2355 to i32, !dbg !16
  %2357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2356), !dbg !16
  %2358 = bitcast i32 %2357 to float, !dbg !16
  %2359 = select i1 %.not, float %2351, float %2358, !dbg !16
  %2360 = select i1 %.not, float %2358, float %2352, !dbg !16
  %2361 = select i1 %.not, float %2354, float %2353, !dbg !16
  %2362 = bitcast float %2361 to i32, !dbg !16
  %2363 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2362), !dbg !16
  %2364 = bitcast i32 %2363 to float, !dbg !16
  %2365 = select i1 %.not, float %2353, float %2364, !dbg !16
  %2366 = select i1 %.not, float %2364, float %2354, !dbg !16
  %2367 = select i1 %.not1, float %2365, float %2359, !dbg !16
  %2368 = bitcast float %2367 to i32, !dbg !16
  %2369 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2368), !dbg !16
  %2370 = bitcast i32 %2369 to float, !dbg !16
  %2371 = select i1 %.not1, float %2359, float %2370, !dbg !16
  %2372 = select i1 %.not1, float %2370, float %2365, !dbg !16
  %2373 = select i1 %.not1, float %2366, float %2360, !dbg !16
  %2374 = bitcast float %2373 to i32, !dbg !16
  %2375 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2374), !dbg !16
  %2376 = bitcast i32 %2375 to float, !dbg !16
  %2377 = select i1 %.not1, float %2360, float %2376, !dbg !16
  %2378 = select i1 %.not1, float %2376, float %2366, !dbg !16
  %2379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %2299, <4 x float> zeroinitializer) #9, !dbg !16
  %2380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %2301, <4 x float> %2379) #9, !dbg !16
  %2381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %2303, <4 x float> %2380) #9, !dbg !16
  %2382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %2305, <4 x float> %2381) #9, !dbg !16
  %2383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %2307, <4 x float> %2382) #9, !dbg !16
  %2384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %2309, <4 x float> %2383) #9, !dbg !16
  %2385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %2311, <4 x float> %2384) #9, !dbg !16
  %2386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %2313, <4 x float> %2385) #9, !dbg !16
  %2387 = extractelement <4 x float> %2386, i64 0, !dbg !16
  %2388 = extractelement <4 x float> %2386, i64 1, !dbg !16
  %2389 = extractelement <4 x float> %2386, i64 2, !dbg !16
  %2390 = extractelement <4 x float> %2386, i64 3, !dbg !16
  %2391 = select i1 %.not, float %2388, float %2387, !dbg !16
  %2392 = bitcast float %2391 to i32, !dbg !16
  %2393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2392), !dbg !16
  %2394 = bitcast i32 %2393 to float, !dbg !16
  %2395 = select i1 %.not, float %2387, float %2394, !dbg !16
  %2396 = select i1 %.not, float %2394, float %2388, !dbg !16
  %2397 = select i1 %.not, float %2390, float %2389, !dbg !16
  %2398 = bitcast float %2397 to i32, !dbg !16
  %2399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %2398), !dbg !16
  %2400 = bitcast i32 %2399 to float, !dbg !16
  %2401 = select i1 %.not, float %2389, float %2400, !dbg !16
  %2402 = select i1 %.not, float %2400, float %2390, !dbg !16
  %2403 = select i1 %.not1, float %2401, float %2395, !dbg !16
  %2404 = bitcast float %2403 to i32, !dbg !16
  %2405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2404), !dbg !16
  %2406 = bitcast i32 %2405 to float, !dbg !16
  %2407 = select i1 %.not1, float %2395, float %2406, !dbg !16
  %2408 = select i1 %.not1, float %2406, float %2401, !dbg !16
  %2409 = select i1 %.not1, float %2402, float %2396, !dbg !16
  %2410 = bitcast float %2409 to i32, !dbg !16
  %2411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %2410), !dbg !16
  %2412 = bitcast i32 %2411 to float, !dbg !16
  %2413 = select i1 %.not1, float %2396, float %2412, !dbg !16
  %2414 = select i1 %.not1, float %2412, float %2402, !dbg !16
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  br i1 %44, label %2415, label %2416, !dbg !36

2415:                                             ; preds = %2172
  %address.thread1033 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !36, !range !40, !frisk.register_lifetimes !1
  %address.local1034 = zext nneg i32 %address.thread1033 to i64, !dbg !12
  %address.local1035 = lshr i64 %address.local1034, 6, !dbg !12
  %address.local1036 = shl nuw nsw i64 %address.local1035, 5, !dbg !36
  %address.local1037 = add nuw nsw i64 %address.local1036, %address.local1034, !dbg !36
  %address.local1038 = lshr i64 %address.local1034, 4, !dbg !12
  %address.local1039 = mul nsw i64 %address.local1038, -16, !dbg !13
  %address.local1040 = add nsw i64 %address.local1037, %address.local1039, !dbg !36
  %address.local1041 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1040, !dbg !36
  store float %1228, ptr addrspace(3) %address.local1041, align 4, !dbg !36
  %address.local1042 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1040, !dbg !36
  %address.local1043 = getelementptr i8, ptr addrspace(3) %address.local1042, i64 64, !dbg !36
  store float %1231, ptr addrspace(3) %address.local1043, align 4, !dbg !36
  br label %2416, !dbg !36

2416:                                             ; preds = %2415, %2172
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %lds.const.gep1738 = getelementptr i8, ptr addrspace(3) %address.local782, i64 0, !dbg !36
  %lds.const.gep1739 = getelementptr i8, ptr addrspace(3) %address.local782, i64 256, !dbg !36
  %2419 = extractvalue [2 x <8 x float>] %68, 0, !dbg !36
  %2420 = shufflevector <8 x float> %2419, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2421 = insertelement <4 x float> poison, float %1228, i64 0, !dbg !36
  %2422 = shufflevector <4 x float> %2421, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2423 = fmul <4 x float> %2420, %2422, !dbg !36
  %2424 = insertelement <4 x float> poison, float %2291, i64 0, !dbg !37
  %2425 = insertelement <4 x float> %2424, float %2297, i64 1, !dbg !37
  %2426 = insertelement <4 x float> %2425, float %2292, i64 2, !dbg !37
  %2427 = insertelement <4 x float> %2426, float %2298, i64 3, !dbg !37
  %2428 = fadd <4 x float> %2423, %2427, !dbg !37
  %2429 = shufflevector <8 x float> %2419, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2430 = fmul <4 x float> %2429, %2422, !dbg !36
  %2431 = insertelement <4 x float> poison, float %2335, i64 0, !dbg !37
  %2432 = insertelement <4 x float> %2431, float %2341, i64 1, !dbg !37
  %2433 = insertelement <4 x float> %2432, float %2336, i64 2, !dbg !37
  %2434 = insertelement <4 x float> %2433, float %2342, i64 3, !dbg !37
  %2435 = fadd <4 x float> %2430, %2434, !dbg !37
  %2436 = extractvalue [2 x <8 x float>] %68, 1, !dbg !36
  %2437 = shufflevector <8 x float> %2436, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %2438 = insertelement <4 x float> poison, float %1231, i64 0, !dbg !36
  %2439 = shufflevector <4 x float> %2438, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %2440 = fmul <4 x float> %2437, %2439, !dbg !36
  %2441 = insertelement <4 x float> poison, float %2371, i64 0, !dbg !37
  %2442 = insertelement <4 x float> %2441, float %2377, i64 1, !dbg !37
  %2443 = insertelement <4 x float> %2442, float %2372, i64 2, !dbg !37
  %2444 = insertelement <4 x float> %2443, float %2378, i64 3, !dbg !37
  %2445 = fadd <4 x float> %2440, %2444, !dbg !37
  %2446 = shufflevector <8 x float> %2436, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %2447 = fmul <4 x float> %2446, %2439, !dbg !36
  %2448 = insertelement <4 x float> poison, float %2407, i64 0, !dbg !37
  %2449 = insertelement <4 x float> %2448, float %2413, i64 1, !dbg !37
  %2450 = insertelement <4 x float> %2449, float %2408, i64 2, !dbg !37
  %2451 = insertelement <4 x float> %2450, float %2414, i64 3, !dbg !37
  %2452 = fadd <4 x float> %2447, %2451, !dbg !37
  %2453 = shufflevector <4 x float> %1726, <4 x float> %1733, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2454 = insertvalue [2 x <8 x float>] poison, <8 x float> %2453, 0, !dbg !7
  %2455 = shufflevector <4 x float> %1743, <4 x float> %1750, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2456 = insertvalue [2 x <8 x float>] %2454, <8 x float> %2455, 1, !dbg !7
  %2457 = insertelement <1 x float> poison, float %1493, i64 0, !dbg !7
  %2458 = insertvalue [2 x <1 x float>] poison, <1 x float> %2457, 0, !dbg !7
  %2459 = insertelement <1 x float> poison, float %1502, i64 0, !dbg !7
  %2460 = insertvalue [2 x <1 x float>] %2458, <1 x float> %2459, 1, !dbg !7
  %2461 = insertvalue [2 x <1 x float>] poison, <1 x float> %1194, 0, !dbg !7
  %2462 = insertvalue [2 x <1 x float>] %2461, <1 x float> %1207, 1, !dbg !7
  %2463 = shufflevector <4 x float> %1970, <4 x float> %1977, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2464 = insertvalue [2 x <8 x float>] poison, <8 x float> %2463, 0, !dbg !7
  %2465 = shufflevector <4 x float> %1987, <4 x float> %1994, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2466 = insertvalue [2 x <8 x float>] %2464, <8 x float> %2465, 1, !dbg !7
  %2467 = shufflevector <4 x float> %2214, <4 x float> %2221, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2468 = insertvalue [2 x <8 x float>] poison, <8 x float> %2467, 0, !dbg !7
  %2469 = shufflevector <4 x float> %2231, <4 x float> %2238, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2470 = insertvalue [2 x <8 x float>] %2468, <8 x float> %2469, 1, !dbg !7
  %2471 = shufflevector <4 x float> %2428, <4 x float> %2435, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2472 = insertvalue [2 x <8 x float>] poison, <8 x float> %2471, 0, !dbg !7
  %2473 = shufflevector <4 x float> %2445, <4 x float> %2452, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2474 = insertvalue [2 x <8 x float>] %2472, <8 x float> %2473, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %2475 = add nuw nsw i64 %62, 1, !dbg !7
  br label %61, !dbg !7

2476:                                             ; preds = %61
  %2477 = extractvalue [2 x <1 x float>] %64, 0, !dbg !21
  %2478 = shufflevector <1 x float> %2477, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %2479 = extractvalue [2 x <1 x float>] %64, 1, !dbg !21
  %2480 = shufflevector <1 x float> %2479, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  br i1 true, label %2481, label %2511, !dbg !23

2481:                                             ; preds = %2476
  %2482 = extractvalue [2 x <8 x float>] %63, 1, !dbg !21
  %2483 = shufflevector <8 x float> %2482, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2480, !dbg !21
  %normalized = fmul <4 x float> %2483, %row.reciprocal, !dbg !21
  %2484 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !22
  %2485 = extractelement <4 x half> %2484, i64 3, !dbg !22
  %2486 = extractelement <4 x half> %2484, i64 2, !dbg !22
  %2487 = extractelement <4 x half> %2484, i64 1, !dbg !22
  %2488 = extractelement <4 x half> %2484, i64 0, !dbg !22
  %2489 = shufflevector <8 x float> %2482, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized68 = fmul <4 x float> %2489, %row.reciprocal, !dbg !21
  %2490 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !22
  %2491 = extractelement <4 x half> %2490, i64 3, !dbg !22
  %2492 = extractelement <4 x half> %2490, i64 2, !dbg !22
  %2493 = extractelement <4 x half> %2490, i64 1, !dbg !22
  %2494 = extractelement <4 x half> %2490, i64 0, !dbg !22
  %2495 = extractvalue [2 x <8 x float>] %63, 0, !dbg !21
  %2496 = shufflevector <8 x float> %2495, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2478, !dbg !21
  %normalized70 = fmul <4 x float> %2496, %row.reciprocal69, !dbg !21
  %2497 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !22
  %2498 = extractelement <4 x half> %2497, i64 3, !dbg !22
  %2499 = extractelement <4 x half> %2497, i64 2, !dbg !22
  %2500 = extractelement <4 x half> %2497, i64 1, !dbg !22
  %2501 = extractelement <4 x half> %2497, i64 0, !dbg !22
  %2502 = shufflevector <8 x float> %2495, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized71 = fmul <4 x float> %2502, %row.reciprocal69, !dbg !21
  %2503 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !22
  %2504 = extractelement <4 x half> %2503, i64 3, !dbg !22
  %2505 = extractelement <4 x half> %2503, i64 2, !dbg !22
  %2506 = extractelement <4 x half> %2503, i64 1, !dbg !22
  %2507 = extractelement <4 x half> %2503, i64 0, !dbg !22
  %2508 = add nsw i64 %15, %25, !dbg !23
  %2509 = shl i64 %2508, 19, !dbg !23
  %2510 = and i64 %2509, 16252928, !dbg !23
  %address.thread1058 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1059 = zext nneg i32 %address.thread1058 to i64, !dbg !12
  %address.local1060 = add i64 %28, %address.local1059, !dbg !23
  %address.local1061 = lshr i64 %address.local1059, 6, !dbg !12
  %address.local1062 = shl nuw nsw i64 %address.local1061, 4, !dbg !13
  %address.local1063 = add i64 %address.local1060, %address.local1062, !dbg !23
  %address.local1064 = lshr i64 %address.local1059, 4, !dbg !12
  %address.local1065 = mul nsw i64 %address.local1064, -16, !dbg !13
  %address.local1066 = add i64 %address.local1063, %address.local1065, !dbg !23
  %address.local1067 = add i64 %address.local1066, %29, !dbg !23
  %address.local1068 = shl nuw nsw i64 %address.local1067, 7, !dbg !23
  %address.local1069 = add nuw nsw i64 %2510, %address.local1068, !dbg !23
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1069, !dbg !23
  %address.local1071 = lshr i64 %address.local1059, 4, !dbg !12
  %address.local1072 = and i64 %address.local1071, 3, !dbg !12
  %address.local1073 = shl nuw nsw i64 %address.local1072, 2, !dbg !12
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1073, !dbg !23
  store half %2507, ptr addrspace(1) %address.local1074, align 2, !dbg !23
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1069, !dbg !23
  %address.local1076 = or disjoint i64 %address.local1073, 1, !dbg !13
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1076, !dbg !23
  store half %2506, ptr addrspace(1) %address.local1077, align 2, !dbg !23
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1069, !dbg !23
  %address.local1079 = or disjoint i64 %address.local1073, 2, !dbg !13
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1079, !dbg !23
  store half %2505, ptr addrspace(1) %address.local1080, align 2, !dbg !23
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1069, !dbg !23
  %address.local1082 = or disjoint i64 %address.local1073, 3, !dbg !13
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1082, !dbg !23
  store half %2504, ptr addrspace(1) %address.local1083, align 2, !dbg !23
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1069, !dbg !23
  %address.local1085 = or disjoint i64 %address.local1073, 16, !dbg !13
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1085, !dbg !23
  store half %2501, ptr addrspace(1) %address.local1086, align 2, !dbg !23
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1069, !dbg !23
  %address.local1088 = or disjoint i64 %address.local1073, 17, !dbg !13
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1087, i64 %address.local1088, !dbg !23
  store half %2500, ptr addrspace(1) %address.local1089, align 2, !dbg !23
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1069, !dbg !23
  %address.local1091 = or disjoint i64 %address.local1073, 18, !dbg !13
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1090, i64 %address.local1091, !dbg !23
  store half %2499, ptr addrspace(1) %address.local1092, align 2, !dbg !23
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1069, !dbg !23
  %address.local1094 = or disjoint i64 %address.local1073, 19, !dbg !13
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1094, !dbg !23
  store half %2498, ptr addrspace(1) %address.local1095, align 2, !dbg !23
  %address.local1096 = shl i64 %address.local1067, 7, !dbg !23
  %address.local1097 = add i64 %address.local1096, 8192, !dbg !23
  %address.local1098 = add nuw nsw i64 %2510, %address.local1097, !dbg !23
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1098, !dbg !23
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1073, !dbg !23
  store half %2494, ptr addrspace(1) %address.local1100, align 2, !dbg !23
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1098, !dbg !23
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1076, !dbg !23
  store half %2493, ptr addrspace(1) %address.local1102, align 2, !dbg !23
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1098, !dbg !23
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1103, i64 %address.local1079, !dbg !23
  store half %2492, ptr addrspace(1) %address.local1104, align 2, !dbg !23
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1098, !dbg !23
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1105, i64 %address.local1082, !dbg !23
  store half %2491, ptr addrspace(1) %address.local1106, align 2, !dbg !23
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1098, !dbg !23
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1085, !dbg !23
  store half %2488, ptr addrspace(1) %address.local1108, align 2, !dbg !23
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1098, !dbg !23
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1109, i64 %address.local1088, !dbg !23
  store half %2487, ptr addrspace(1) %address.local1110, align 2, !dbg !23
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1098, !dbg !23
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1091, !dbg !23
  store half %2486, ptr addrspace(1) %address.local1112, align 2, !dbg !23
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1098, !dbg !23
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1094, !dbg !23
  store half %2485, ptr addrspace(1) %address.local1114, align 2, !dbg !23
  br label %2511, !dbg !23

2511:                                             ; preds = %2481, %2476
  br i1 true, label %2512, label %2540, !dbg !23

2512:                                             ; preds = %2511
  %2513 = extractvalue [2 x <8 x float>] %66, 1, !dbg !21
  %2514 = shufflevector <8 x float> %2513, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2480, !dbg !21
  %normalized73 = fmul <4 x float> %2514, %row.reciprocal72, !dbg !21
  %2515 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !22
  %2516 = extractelement <4 x half> %2515, i64 3, !dbg !22
  %2517 = extractelement <4 x half> %2515, i64 2, !dbg !22
  %2518 = extractelement <4 x half> %2515, i64 1, !dbg !22
  %2519 = extractelement <4 x half> %2515, i64 0, !dbg !22
  %2520 = shufflevector <8 x float> %2513, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized74 = fmul <4 x float> %2520, %row.reciprocal72, !dbg !21
  %2521 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !22
  %2522 = extractelement <4 x half> %2521, i64 3, !dbg !22
  %2523 = extractelement <4 x half> %2521, i64 2, !dbg !22
  %2524 = extractelement <4 x half> %2521, i64 1, !dbg !22
  %2525 = extractelement <4 x half> %2521, i64 0, !dbg !22
  %2526 = extractvalue [2 x <8 x float>] %66, 0, !dbg !21
  %2527 = shufflevector <8 x float> %2526, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2478, !dbg !21
  %normalized76 = fmul <4 x float> %2527, %row.reciprocal75, !dbg !21
  %2528 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !22
  %2529 = extractelement <4 x half> %2528, i64 3, !dbg !22
  %2530 = extractelement <4 x half> %2528, i64 2, !dbg !22
  %2531 = extractelement <4 x half> %2528, i64 1, !dbg !22
  %2532 = extractelement <4 x half> %2528, i64 0, !dbg !22
  %2533 = shufflevector <8 x float> %2526, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized77 = fmul <4 x float> %2533, %row.reciprocal75, !dbg !21
  %2534 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !22
  %2535 = extractelement <4 x half> %2534, i64 3, !dbg !22
  %2536 = extractelement <4 x half> %2534, i64 2, !dbg !22
  %2537 = extractelement <4 x half> %2534, i64 1, !dbg !22
  %2538 = extractelement <4 x half> %2534, i64 0, !dbg !22
  %2539 = shl nuw nsw i64 %27, 19, !dbg !23
  %address.thread1115 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1116 = zext nneg i32 %address.thread1115 to i64, !dbg !12
  %address.local1117 = add i64 %28, %address.local1116, !dbg !23
  %address.local1118 = lshr i64 %address.local1116, 6, !dbg !12
  %address.local1119 = shl nuw nsw i64 %address.local1118, 4, !dbg !13
  %address.local1120 = add i64 %address.local1117, %address.local1119, !dbg !23
  %address.local1121 = lshr i64 %address.local1116, 4, !dbg !12
  %address.local1122 = mul nsw i64 %address.local1121, -16, !dbg !13
  %address.local1123 = add i64 %address.local1120, %address.local1122, !dbg !23
  %address.local1124 = add i64 %address.local1123, %29, !dbg !23
  %address.local1125 = shl nuw nsw i64 %address.local1124, 7, !dbg !23
  %address.local1126 = add nuw nsw i64 %2539, %address.local1125, !dbg !23
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1126, !dbg !23
  %address.local1128 = lshr i64 %address.local1116, 4, !dbg !12
  %address.local1129 = and i64 %address.local1128, 3, !dbg !12
  %address.local1130 = shl nuw nsw i64 %address.local1129, 2, !dbg !12
  %address.local1131 = or disjoint i64 %address.local1130, 32, !dbg !13
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1131, !dbg !23
  store half %2538, ptr addrspace(1) %address.local1132, align 2, !dbg !23
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1126, !dbg !23
  %address.local1134 = or disjoint i64 %address.local1130, 33, !dbg !13
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1134, !dbg !23
  store half %2537, ptr addrspace(1) %address.local1135, align 2, !dbg !23
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1126, !dbg !23
  %address.local1137 = or disjoint i64 %address.local1130, 34, !dbg !13
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1136, i64 %address.local1137, !dbg !23
  store half %2536, ptr addrspace(1) %address.local1138, align 2, !dbg !23
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1126, !dbg !23
  %address.local1140 = or disjoint i64 %address.local1130, 35, !dbg !13
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1140, !dbg !23
  store half %2535, ptr addrspace(1) %address.local1141, align 2, !dbg !23
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1126, !dbg !23
  %address.local1143 = or disjoint i64 %address.local1130, 48, !dbg !13
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1142, i64 %address.local1143, !dbg !23
  store half %2532, ptr addrspace(1) %address.local1144, align 2, !dbg !23
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1126, !dbg !23
  %address.local1146 = or disjoint i64 %address.local1130, 49, !dbg !13
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1146, !dbg !23
  store half %2531, ptr addrspace(1) %address.local1147, align 2, !dbg !23
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1126, !dbg !23
  %address.local1149 = or disjoint i64 %address.local1130, 50, !dbg !13
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1148, i64 %address.local1149, !dbg !23
  store half %2530, ptr addrspace(1) %address.local1150, align 2, !dbg !23
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1126, !dbg !23
  %address.local1152 = or disjoint i64 %address.local1130, 51, !dbg !13
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1152, !dbg !23
  store half %2529, ptr addrspace(1) %address.local1153, align 2, !dbg !23
  %address.local1154 = shl i64 %address.local1124, 7, !dbg !23
  %address.local1155 = add i64 %address.local1154, 8192, !dbg !23
  %address.local1156 = add nuw nsw i64 %2539, %address.local1155, !dbg !23
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1156, !dbg !23
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1157, i64 %address.local1131, !dbg !23
  store half %2525, ptr addrspace(1) %address.local1158, align 2, !dbg !23
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1156, !dbg !23
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1134, !dbg !23
  store half %2524, ptr addrspace(1) %address.local1160, align 2, !dbg !23
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1156, !dbg !23
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1161, i64 %address.local1137, !dbg !23
  store half %2523, ptr addrspace(1) %address.local1162, align 2, !dbg !23
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1156, !dbg !23
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1163, i64 %address.local1140, !dbg !23
  store half %2522, ptr addrspace(1) %address.local1164, align 2, !dbg !23
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1156, !dbg !23
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1143, !dbg !23
  store half %2519, ptr addrspace(1) %address.local1166, align 2, !dbg !23
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1156, !dbg !23
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1167, i64 %address.local1146, !dbg !23
  store half %2518, ptr addrspace(1) %address.local1168, align 2, !dbg !23
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1156, !dbg !23
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1169, i64 %address.local1149, !dbg !23
  store half %2517, ptr addrspace(1) %address.local1170, align 2, !dbg !23
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1156, !dbg !23
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1152, !dbg !23
  store half %2516, ptr addrspace(1) %address.local1172, align 2, !dbg !23
  br label %2540, !dbg !23

2540:                                             ; preds = %2512, %2511
  br i1 true, label %2541, label %2569, !dbg !23

2541:                                             ; preds = %2540
  %2542 = extractvalue [2 x <8 x float>] %67, 1, !dbg !21
  %2543 = shufflevector <8 x float> %2542, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2480, !dbg !21
  %normalized79 = fmul <4 x float> %2543, %row.reciprocal78, !dbg !21
  %2544 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !22
  %2545 = extractelement <4 x half> %2544, i64 3, !dbg !22
  %2546 = extractelement <4 x half> %2544, i64 2, !dbg !22
  %2547 = extractelement <4 x half> %2544, i64 1, !dbg !22
  %2548 = extractelement <4 x half> %2544, i64 0, !dbg !22
  %2549 = shufflevector <8 x float> %2542, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized80 = fmul <4 x float> %2549, %row.reciprocal78, !dbg !21
  %2550 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !22
  %2551 = extractelement <4 x half> %2550, i64 3, !dbg !22
  %2552 = extractelement <4 x half> %2550, i64 2, !dbg !22
  %2553 = extractelement <4 x half> %2550, i64 1, !dbg !22
  %2554 = extractelement <4 x half> %2550, i64 0, !dbg !22
  %2555 = extractvalue [2 x <8 x float>] %67, 0, !dbg !21
  %2556 = shufflevector <8 x float> %2555, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2478, !dbg !21
  %normalized82 = fmul <4 x float> %2556, %row.reciprocal81, !dbg !21
  %2557 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !22
  %2558 = extractelement <4 x half> %2557, i64 3, !dbg !22
  %2559 = extractelement <4 x half> %2557, i64 2, !dbg !22
  %2560 = extractelement <4 x half> %2557, i64 1, !dbg !22
  %2561 = extractelement <4 x half> %2557, i64 0, !dbg !22
  %2562 = shufflevector <8 x float> %2555, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized83 = fmul <4 x float> %2562, %row.reciprocal81, !dbg !21
  %2563 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !22
  %2564 = extractelement <4 x half> %2563, i64 3, !dbg !22
  %2565 = extractelement <4 x half> %2563, i64 2, !dbg !22
  %2566 = extractelement <4 x half> %2563, i64 1, !dbg !22
  %2567 = extractelement <4 x half> %2563, i64 0, !dbg !22
  %2568 = shl nuw nsw i64 %27, 19, !dbg !23
  %address.thread1173 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1174 = zext nneg i32 %address.thread1173 to i64, !dbg !12
  %address.local1175 = add i64 %28, %address.local1174, !dbg !23
  %address.local1176 = lshr i64 %address.local1174, 6, !dbg !12
  %address.local1177 = shl nuw nsw i64 %address.local1176, 4, !dbg !13
  %address.local1178 = add i64 %address.local1175, %address.local1177, !dbg !23
  %address.local1179 = lshr i64 %address.local1174, 4, !dbg !12
  %address.local1180 = mul nsw i64 %address.local1179, -16, !dbg !13
  %address.local1181 = add i64 %address.local1178, %address.local1180, !dbg !23
  %address.local1182 = add i64 %address.local1181, %29, !dbg !23
  %address.local1183 = shl nuw nsw i64 %address.local1182, 7, !dbg !23
  %address.local1184 = add nuw nsw i64 %2568, %address.local1183, !dbg !23
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !23
  %address.local1186 = lshr i64 %address.local1174, 4, !dbg !12
  %address.local1187 = and i64 %address.local1186, 3, !dbg !12
  %address.local1188 = shl nuw nsw i64 %address.local1187, 2, !dbg !12
  %address.local1189 = or disjoint i64 %address.local1188, 64, !dbg !13
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1189, !dbg !23
  store half %2567, ptr addrspace(1) %address.local1190, align 2, !dbg !23
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !23
  %address.local1192 = or disjoint i64 %address.local1188, 65, !dbg !13
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1192, !dbg !23
  store half %2566, ptr addrspace(1) %address.local1193, align 2, !dbg !23
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !23
  %address.local1195 = or disjoint i64 %address.local1188, 66, !dbg !13
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1194, i64 %address.local1195, !dbg !23
  store half %2565, ptr addrspace(1) %address.local1196, align 2, !dbg !23
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !23
  %address.local1198 = or disjoint i64 %address.local1188, 67, !dbg !13
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1197, i64 %address.local1198, !dbg !23
  store half %2564, ptr addrspace(1) %address.local1199, align 2, !dbg !23
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !23
  %address.local1201 = or disjoint i64 %address.local1188, 80, !dbg !13
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1200, i64 %address.local1201, !dbg !23
  store half %2561, ptr addrspace(1) %address.local1202, align 2, !dbg !23
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !23
  %address.local1204 = or disjoint i64 %address.local1188, 81, !dbg !13
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1203, i64 %address.local1204, !dbg !23
  store half %2560, ptr addrspace(1) %address.local1205, align 2, !dbg !23
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !23
  %address.local1207 = or disjoint i64 %address.local1188, 82, !dbg !13
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1206, i64 %address.local1207, !dbg !23
  store half %2559, ptr addrspace(1) %address.local1208, align 2, !dbg !23
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !23
  %address.local1210 = or disjoint i64 %address.local1188, 83, !dbg !13
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1209, i64 %address.local1210, !dbg !23
  store half %2558, ptr addrspace(1) %address.local1211, align 2, !dbg !23
  %address.local1212 = shl i64 %address.local1182, 7, !dbg !23
  %address.local1213 = add i64 %address.local1212, 8192, !dbg !23
  %address.local1214 = add nuw nsw i64 %2568, %address.local1213, !dbg !23
  %address.local1215 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1214, !dbg !23
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1215, i64 %address.local1189, !dbg !23
  store half %2554, ptr addrspace(1) %address.local1216, align 2, !dbg !23
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1214, !dbg !23
  %address.local1218 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1217, i64 %address.local1192, !dbg !23
  store half %2553, ptr addrspace(1) %address.local1218, align 2, !dbg !23
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1214, !dbg !23
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1219, i64 %address.local1195, !dbg !23
  store half %2552, ptr addrspace(1) %address.local1220, align 2, !dbg !23
  %address.local1221 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1214, !dbg !23
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1221, i64 %address.local1198, !dbg !23
  store half %2551, ptr addrspace(1) %address.local1222, align 2, !dbg !23
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1214, !dbg !23
  %address.local1224 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1223, i64 %address.local1201, !dbg !23
  store half %2548, ptr addrspace(1) %address.local1224, align 2, !dbg !23
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1214, !dbg !23
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1225, i64 %address.local1204, !dbg !23
  store half %2547, ptr addrspace(1) %address.local1226, align 2, !dbg !23
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1214, !dbg !23
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1227, i64 %address.local1207, !dbg !23
  store half %2546, ptr addrspace(1) %address.local1228, align 2, !dbg !23
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1214, !dbg !23
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1229, i64 %address.local1210, !dbg !23
  store half %2545, ptr addrspace(1) %address.local1230, align 2, !dbg !23
  br label %2569, !dbg !23

2569:                                             ; preds = %2541, %2540
  br i1 true, label %2570, label %2598, !dbg !23

2570:                                             ; preds = %2569
  %2571 = extractvalue [2 x <8 x float>] %68, 1, !dbg !21
  %2572 = shufflevector <8 x float> %2571, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2480, !dbg !21
  %normalized85 = fmul <4 x float> %2572, %row.reciprocal84, !dbg !21
  %2573 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !22
  %2574 = extractelement <4 x half> %2573, i64 3, !dbg !22
  %2575 = extractelement <4 x half> %2573, i64 2, !dbg !22
  %2576 = extractelement <4 x half> %2573, i64 1, !dbg !22
  %2577 = extractelement <4 x half> %2573, i64 0, !dbg !22
  %2578 = shufflevector <8 x float> %2571, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized86 = fmul <4 x float> %2578, %row.reciprocal84, !dbg !21
  %2579 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !22
  %2580 = extractelement <4 x half> %2579, i64 3, !dbg !22
  %2581 = extractelement <4 x half> %2579, i64 2, !dbg !22
  %2582 = extractelement <4 x half> %2579, i64 1, !dbg !22
  %2583 = extractelement <4 x half> %2579, i64 0, !dbg !22
  %2584 = extractvalue [2 x <8 x float>] %68, 0, !dbg !21
  %2585 = shufflevector <8 x float> %2584, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2478, !dbg !21
  %normalized88 = fmul <4 x float> %2585, %row.reciprocal87, !dbg !21
  %2586 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !22
  %2587 = extractelement <4 x half> %2586, i64 3, !dbg !22
  %2588 = extractelement <4 x half> %2586, i64 2, !dbg !22
  %2589 = extractelement <4 x half> %2586, i64 1, !dbg !22
  %2590 = extractelement <4 x half> %2586, i64 0, !dbg !22
  %2591 = shufflevector <8 x float> %2584, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %normalized89 = fmul <4 x float> %2591, %row.reciprocal87, !dbg !21
  %2592 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !22
  %2593 = extractelement <4 x half> %2592, i64 3, !dbg !22
  %2594 = extractelement <4 x half> %2592, i64 2, !dbg !22
  %2595 = extractelement <4 x half> %2592, i64 1, !dbg !22
  %2596 = extractelement <4 x half> %2592, i64 0, !dbg !22
  %2597 = shl nuw nsw i64 %27, 19, !dbg !23
  %address.thread1231 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !23, !range !40, !frisk.register_lifetimes !1
  %address.local1232 = zext nneg i32 %address.thread1231 to i64, !dbg !12
  %address.local1233 = add i64 %28, %address.local1232, !dbg !23
  %address.local1234 = lshr i64 %address.local1232, 6, !dbg !12
  %address.local1235 = shl nuw nsw i64 %address.local1234, 4, !dbg !13
  %address.local1236 = add i64 %address.local1233, %address.local1235, !dbg !23
  %address.local1237 = lshr i64 %address.local1232, 4, !dbg !12
  %address.local1238 = mul nsw i64 %address.local1237, -16, !dbg !13
  %address.local1239 = add i64 %address.local1236, %address.local1238, !dbg !23
  %address.local1240 = add i64 %address.local1239, %29, !dbg !23
  %address.local1241 = shl nuw nsw i64 %address.local1240, 7, !dbg !23
  %address.local1242 = add nuw nsw i64 %2597, %address.local1241, !dbg !23
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1242, !dbg !23
  %address.local1244 = lshr i64 %address.local1232, 4, !dbg !12
  %address.local1245 = and i64 %address.local1244, 3, !dbg !12
  %address.local1246 = shl nuw nsw i64 %address.local1245, 2, !dbg !12
  %address.local1247 = or disjoint i64 %address.local1246, 96, !dbg !13
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1243, i64 %address.local1247, !dbg !23
  store half %2596, ptr addrspace(1) %address.local1248, align 2, !dbg !23
  %address.local1249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1242, !dbg !23
  %address.local1250 = or disjoint i64 %address.local1246, 97, !dbg !13
  %address.local1251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1249, i64 %address.local1250, !dbg !23
  store half %2595, ptr addrspace(1) %address.local1251, align 2, !dbg !23
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1242, !dbg !23
  %address.local1253 = or disjoint i64 %address.local1246, 98, !dbg !13
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1252, i64 %address.local1253, !dbg !23
  store half %2594, ptr addrspace(1) %address.local1254, align 2, !dbg !23
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1242, !dbg !23
  %address.local1256 = or disjoint i64 %address.local1246, 99, !dbg !13
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1255, i64 %address.local1256, !dbg !23
  store half %2593, ptr addrspace(1) %address.local1257, align 2, !dbg !23
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1242, !dbg !23
  %address.local1259 = or disjoint i64 %address.local1246, 112, !dbg !13
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1258, i64 %address.local1259, !dbg !23
  store half %2590, ptr addrspace(1) %address.local1260, align 2, !dbg !23
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1242, !dbg !23
  %address.local1262 = or disjoint i64 %address.local1246, 113, !dbg !13
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1261, i64 %address.local1262, !dbg !23
  store half %2589, ptr addrspace(1) %address.local1263, align 2, !dbg !23
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1242, !dbg !23
  %address.local1265 = or disjoint i64 %address.local1246, 114, !dbg !13
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1264, i64 %address.local1265, !dbg !23
  store half %2588, ptr addrspace(1) %address.local1266, align 2, !dbg !23
  %address.local1267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1242, !dbg !23
  %address.local1268 = or disjoint i64 %address.local1246, 115, !dbg !13
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1267, i64 %address.local1268, !dbg !23
  store half %2587, ptr addrspace(1) %address.local1269, align 2, !dbg !23
  %address.local1270 = shl i64 %address.local1240, 7, !dbg !23
  %address.local1271 = add i64 %address.local1270, 8192, !dbg !23
  %address.local1272 = add nuw nsw i64 %2597, %address.local1271, !dbg !23
  %address.local1273 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1272, !dbg !23
  %address.local1274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1273, i64 %address.local1247, !dbg !23
  store half %2583, ptr addrspace(1) %address.local1274, align 2, !dbg !23
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1272, !dbg !23
  %address.local1276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1275, i64 %address.local1250, !dbg !23
  store half %2582, ptr addrspace(1) %address.local1276, align 2, !dbg !23
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1272, !dbg !23
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1277, i64 %address.local1253, !dbg !23
  store half %2581, ptr addrspace(1) %address.local1278, align 2, !dbg !23
  %address.local1279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1272, !dbg !23
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1279, i64 %address.local1256, !dbg !23
  store half %2580, ptr addrspace(1) %address.local1280, align 2, !dbg !23
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1272, !dbg !23
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1281, i64 %address.local1259, !dbg !23
  store half %2577, ptr addrspace(1) %address.local1282, align 2, !dbg !23
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1272, !dbg !23
  %address.local1284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1283, i64 %address.local1262, !dbg !23
  store half %2576, ptr addrspace(1) %address.local1284, align 2, !dbg !23
  %address.local1285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1272, !dbg !23
  %address.local1286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1285, i64 %address.local1265, !dbg !23
  store half %2575, ptr addrspace(1) %address.local1286, align 2, !dbg !23
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1272, !dbg !23
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1287, i64 %address.local1268, !dbg !23
  store half %2574, ptr addrspace(1) %address.local1288, align 2, !dbg !23
  br label %2598, !dbg !23

2598:                                             ; preds = %2570, %2569
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/run-20261027-121856-x88fub13/grid-000001")
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
!15 = !DILocation(line: 93, column: 22, scope: !4)
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
!32 = !DILocation(line: 92, column: 20, scope: !4)
!33 = !DILocation(line: 94, column: 22, scope: !4)
!34 = !DILocation(line: 95, column: 20, scope: !4)
!35 = !DILocation(line: 96, column: 16, scope: !4)
!36 = !DILocation(line: 103, column: 22, scope: !4)
!37 = !DILocation(line: 104, column: 20, scope: !4)
!38 = !{i32 256, i32 1, i32 1}
!39 = !{i32 2, !"Debug Info Version", i32 3}
!40 = !{i32 0, i32 256}
