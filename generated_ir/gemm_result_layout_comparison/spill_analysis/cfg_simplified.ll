; ModuleID = 'generated_ir/gemm_result_layout_comparison/exp2_fixed/layout_3_1/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000001/input.mlir"

@shm_0 = addrspace(3) global [65536 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = shl nsw i64 %8, 5, !dbg !7
  %10 = add nsw i64 %9, %6, !dbg !7
  %11 = lshr i64 %10, 5, !dbg !7
  %12 = and i64 %11, 576460752303423484, !dbg !7
  %13 = and i64 %6, 3, !dbg !7
  %14 = or disjoint i64 %12, %13, !dbg !7
  %15 = lshr i64 %10, 2, !dbg !7
  %16 = sub nsw i64 31, %14, !dbg !7
  %17 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %18 = sext i32 %17 to i64, !dbg !7
  %19 = shl i64 %16, 7, !dbg !8
  %20 = icmp ugt i64 %10, 1023, !dbg !9
  %21 = add nsw i64 %14, -32, !dbg !9
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !9
  %23 = sdiv i64 %22, 32, !dbg !9
  %24 = sext i1 %20 to i64, !dbg !9
  %25 = xor i64 %23, %24, !dbg !9
  %26 = add nsw i64 %15, %25, !dbg !9
  %27 = and i64 %26, 31, !dbg !9
  %28 = shl nsw i64 %16, 7, !dbg !9
  %29 = mul nsw i64 %25, -4096, !dbg !9
  %30 = add i64 %28, %29, !dbg !9
  %31 = shl nuw nsw i64 %27, 19, !dbg !9
  %32 = shl nsw i64 %30, 7, !dbg !9
  %33 = add i64 %31, %32, !dbg !9
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !10, !range !11
  %35 = zext nneg i32 %34 to i64, !dbg !10
  %36 = lshr i64 %35, 4, !dbg !10
  %37 = and i64 %36, 3, !dbg !10
  %38 = shl nuw nsw i64 %37, 2, !dbg !10
  %39 = lshr i64 %35, 4, !dbg !10
  %40 = or disjoint i64 %19, 127, !dbg !12
  %41 = add i64 %19, 128, !dbg !12
  %42 = icmp uge i64 %41, %40, !dbg !12
  %43 = icmp eq i64 %37, 0, !dbg !13
  %44 = and i32 %34, 16, !dbg !14
  %.not = icmp eq i32 %44, 0, !dbg !14
  %45 = and i32 %34, 32, !dbg !14
  %.not1 = icmp eq i32 %45, 0, !dbg !14
  %46 = sub nsw i64 32, %14, !dbg !15
  %47 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !9
  %48 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !9
  %49 = getelementptr i8, ptr addrspace(1) %48, i64 64, !dbg !9
  %50 = shl i64 %15, 7, !dbg !16
  %51 = and i64 %50, 3968, !dbg !16
  %52 = shl nsw i64 %18, 12, !dbg !16
  %53 = or disjoint i64 %51, %52, !dbg !16
  %54 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !9
  %55 = getelementptr i8, ptr addrspace(1) %54, i64 128, !dbg !9
  %56 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !9
  %57 = getelementptr i8, ptr addrspace(1) %56, i64 192, !dbg !9
  %58 = lshr i64 %35, 2, !dbg !17
  %59 = and i64 %58, 48, !dbg !17
  %60 = mul nsw i64 %39, -16, !dbg !17
  br label %61, !dbg !18

61:                                               ; preds = %2416, %4
  %62 = phi i64 [ %2475, %2416 ], [ 0, %4 ], !dbg !18
  %63 = phi [2 x <8 x float>] [ %2456, %2416 ], [ zeroinitializer, %4 ], !dbg !18
  %64 = phi [2 x <1 x float>] [ %2460, %2416 ], [ zeroinitializer, %4 ], !dbg !18
  %65 = phi [2 x <1 x float>] [ %2462, %2416 ], [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], !dbg !18
  %66 = phi [2 x <8 x float>] [ %2466, %2416 ], [ zeroinitializer, %4 ], !dbg !18
  %67 = phi [2 x <8 x float>] [ %2470, %2416 ], [ zeroinitializer, %4 ], !dbg !18
  %68 = phi [2 x <8 x float>] [ %2474, %2416 ], [ zeroinitializer, %4 ], !dbg !18
  %69 = icmp slt i64 %62, %46, !dbg !18
  br i1 %69, label %70, label %2476, !dbg !18

70:                                               ; preds = %61
  %71 = shl nsw i64 %62, 7, !dbg !18
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !9, !range !11, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !10
  %address.local258 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local259 = and i64 %address.local258, 1536, !dbg !9
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local259, !dbg !9
  %address.local261 = ptrtoint ptr addrspace(3) %address.local260 to i64, !dbg !9
  %address.local262 = trunc i64 %address.local261 to i32, !dbg !9
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !9
  %73 = zext i32 %72 to i64, !dbg !9
  %74 = inttoptr i64 %73 to ptr addrspace(3), !dbg !9
  %address.local263 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local264 = and i64 %address.local263, 16256, !dbg !9
  %address.local265 = lshr i64 %address.local, 4, !dbg !9
  %address.local266 = and i64 %address.local265, 8, !dbg !9
  %address.local267 = or disjoint i64 %address.local264, %address.local266, !dbg !9
  %address.local268 = getelementptr [2 x i8], ptr addrspace(1) %47, i64 %address.local267, !dbg !9
  %75 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %76 = ptrtoint ptr addrspace(1) %address.local268 to i64, !dbg !9
  %77 = sub i64 %76, %75, !dbg !9
  %dma.buffer.offset1821 = trunc i64 %77 to i32, !dbg !9
  %78 = trunc i64 %75 to i32, !dbg !9
  %79 = insertelement <4 x i32> poison, i32 %78, i64 0, !dbg !9
  %80 = lshr i64 %75, 32, !dbg !9
  %81 = trunc i64 %80 to i32, !dbg !9
  %82 = insertelement <4 x i32> %79, i32 %81, i64 1, !dbg !9
  %83 = insertelement <4 x i32> %82, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1822 = insertelement <4 x i32> %83, i32 131072, i64 3, !dbg !9
  %84 = ptrtoint ptr addrspace(3) %74 to i32, !dbg !9
  %dma.buffer.lds1823 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !9
  %85 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1823, i32 %dma.buffer.offset1821, <4 x i32> %dma.buffer.rsrc1822) #10, !dbg !9
  %address.local269 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local270 = shl nuw nsw i64 %address.local269, 3, !dbg !9
  %address.local271 = and i64 %address.local270, 3584, !dbg !9
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local271, !dbg !9
  %address.local273 = ptrtoint ptr addrspace(3) %address.local272 to i64, !dbg !9
  %address.local274 = trunc i64 %address.local273 to i32, !dbg !9
  %86 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !9
  %87 = zext i32 %86 to i64, !dbg !9
  %88 = inttoptr i64 %87 to ptr addrspace(3), !dbg !9
  %address.local275 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local276 = and i64 %address.local275, 16256, !dbg !9
  %address.local277 = lshr i64 %address.local269, 4, !dbg !9
  %address.local278 = and i64 %address.local277, 24, !dbg !9
  %address.local279 = or disjoint i64 %address.local276, %address.local278, !dbg !9
  %address.local280 = getelementptr [2 x i8], ptr addrspace(1) %47, i64 %address.local279, !dbg !9
  %89 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %90 = ptrtoint ptr addrspace(1) %address.local280 to i64, !dbg !9
  %91 = sub i64 %90, %89, !dbg !9
  %dma.buffer.offset1818 = trunc i64 %91 to i32, !dbg !9
  %92 = trunc i64 %89 to i32, !dbg !9
  %93 = insertelement <4 x i32> poison, i32 %92, i64 0, !dbg !9
  %94 = lshr i64 %89, 32, !dbg !9
  %95 = trunc i64 %94 to i32, !dbg !9
  %96 = insertelement <4 x i32> %93, i32 %95, i64 1, !dbg !9
  %97 = insertelement <4 x i32> %96, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1819 = insertelement <4 x i32> %97, i32 131072, i64 3, !dbg !9
  %98 = ptrtoint ptr addrspace(3) %88 to i32, !dbg !9
  %dma.buffer.lds1820 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %98), !dbg !9
  %99 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1820, i32 %dma.buffer.offset1818, <4 x i32> %dma.buffer.rsrc1819) #10, !dbg !9
  %100 = lshr i64 %62, 12, !dbg !16
  %101 = add nuw nsw i64 %100, %15, !dbg !16
  %102 = lshr i64 %62, 5, !dbg !16
  %103 = and i64 %102, 127, !dbg !16
  %104 = shl i64 %101, 19, !dbg !16
  %105 = and i64 %104, 16252928, !dbg !16
  %106 = shl i64 %102, 19, !dbg !16
  %107 = shl i64 %62, 14, !dbg !16
  %108 = sub i64 %107, %106, !dbg !16
  %109 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %105, !dbg !16
  %110 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %103, !dbg !16
  %111 = getelementptr [2 x i8], ptr addrspace(1) %110, i64 %108, !dbg !16
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local259, !dbg !16
  %address.local282 = ptrtoint ptr addrspace(3) %address.local281 to i64, !dbg !16
  %address.local283 = trunc i64 %address.local282 to i32, !dbg !16
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !16
  %113 = zext i32 %112 to i64, !dbg !16
  %114 = inttoptr i64 %113 to ptr addrspace(3), !dbg !16
  %address.local284 = lshr i64 %address.local, 4, !dbg !16
  %address.local285 = and i64 %address.local284, 8, !dbg !16
  %address.local286 = and i64 %address.local, 123, !dbg !16
  %address.local287 = lshr i64 %address.local, 2, !dbg !16
  %address.local288 = and i64 %address.local287, 4, !dbg !16
  %address.local289 = add nuw nsw i64 %address.local288, %address.local, !dbg !16
  %address.local290 = and i64 %address.local289, 4, !dbg !16
  %address.local291 = or disjoint i64 %address.local286, %address.local290, !dbg !16
  %address.local292 = shl nuw nsw i64 %address.local291, 7, !dbg !16
  %address.local293 = or disjoint i64 %address.local285, %address.local292, !dbg !16
  %address.local294 = getelementptr [2 x i8], ptr addrspace(1) %111, i64 %address.local293, !dbg !16
  %115 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %116 = ptrtoint ptr addrspace(1) %address.local294 to i64, !dbg !16
  %117 = sub i64 %116, %115, !dbg !16
  %dma.buffer.offset1815 = trunc i64 %117 to i32, !dbg !16
  %118 = trunc i64 %115 to i32, !dbg !16
  %119 = insertelement <4 x i32> poison, i32 %118, i64 0, !dbg !16
  %120 = lshr i64 %115, 32, !dbg !16
  %121 = trunc i64 %120 to i32, !dbg !16
  %122 = insertelement <4 x i32> %119, i32 %121, i64 1, !dbg !16
  %123 = insertelement <4 x i32> %122, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1816 = insertelement <4 x i32> %123, i32 131072, i64 3, !dbg !16
  %124 = ptrtoint ptr addrspace(3) %114 to i32, !dbg !16
  %dma.buffer.lds1817 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %124), !dbg !16
  %125 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1817, i32 %dma.buffer.offset1815, <4 x i32> %dma.buffer.rsrc1816) #10, !dbg !16
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local271, !dbg !16
  %address.local296 = ptrtoint ptr addrspace(3) %address.local295 to i64, !dbg !16
  %address.local297 = trunc i64 %address.local296 to i32, !dbg !16
  %126 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !16
  %127 = zext i32 %126 to i64, !dbg !16
  %128 = inttoptr i64 %127 to ptr addrspace(3), !dbg !16
  %address.local298 = lshr i64 %address.local269, 4, !dbg !16
  %address.local299 = and i64 %address.local298, 24, !dbg !16
  %address.local300 = and i64 %address.local, 123, !dbg !16
  %address.local301 = lshr i64 %address.local, 2, !dbg !16
  %address.local302 = and i64 %address.local301, 4, !dbg !16
  %address.local303 = add nuw nsw i64 %address.local302, %address.local, !dbg !16
  %address.local304 = and i64 %address.local303, 4, !dbg !16
  %address.local305 = or disjoint i64 %address.local300, %address.local304, !dbg !16
  %address.local306 = shl nuw nsw i64 %address.local305, 7, !dbg !16
  %address.local307 = or disjoint i64 %address.local299, %address.local306, !dbg !16
  %address.local308 = getelementptr [2 x i8], ptr addrspace(1) %111, i64 %address.local307, !dbg !16
  %129 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %130 = ptrtoint ptr addrspace(1) %address.local308 to i64, !dbg !16
  %131 = sub i64 %130, %129, !dbg !16
  %dma.buffer.offset1812 = trunc i64 %131 to i32, !dbg !16
  %132 = trunc i64 %129 to i32, !dbg !16
  %133 = insertelement <4 x i32> poison, i32 %132, i64 0, !dbg !16
  %134 = lshr i64 %129, 32, !dbg !16
  %135 = trunc i64 %134 to i32, !dbg !16
  %136 = insertelement <4 x i32> %133, i32 %135, i64 1, !dbg !16
  %137 = insertelement <4 x i32> %136, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1813 = insertelement <4 x i32> %137, i32 131072, i64 3, !dbg !16
  %138 = ptrtoint ptr addrspace(3) %128 to i32, !dbg !16
  %dma.buffer.lds1814 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138), !dbg !16
  %139 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1814, i32 %dma.buffer.offset1812, <4 x i32> %dma.buffer.rsrc1813) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local309 = lshr i32 %address.thread, 2, !dbg !10
  %address.local310 = and i32 %address.local309, 4, !dbg !10
  %address.local311 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local312 = and i32 %address.local311, 384, !dbg !10
  %address.local313 = or disjoint i32 %address.local310, %address.local312, !dbg !10
  %address.local314 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local315 = and i32 %address.local314, 120, !dbg !10
  %address.local316 = or disjoint i32 %address.local313, %address.local315, !dbg !10
  %address.local317 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local318 = and i32 %address.local317, 1024, !dbg !10
  %address.local319 = or disjoint i32 %address.local316, %address.local318, !dbg !10
  %address.local320 = zext nneg i32 %address.local319 to i64, !dbg !10
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local320, !dbg !10
  %140 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !10
  %141 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !10
  %142 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !10
  %143 = fmul <4 x half> %140, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !10
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !10
  %145 = fmul <4 x half> %141, splat (half 1.275630e-01), !dbg !10
  %146 = fmul <4 x half> %142, splat (half 1.275630e-01), !dbg !10
  %147 = fmul <4 x half> %144, splat (half 1.275630e-01), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
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
  %212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !9
  %213 = zext i32 %212 to i64, !dbg !9
  %214 = inttoptr i64 %213 to ptr addrspace(3), !dbg !9
  %address.local401 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local267, !dbg !9
  %215 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %216 = ptrtoint ptr addrspace(1) %address.local401 to i64, !dbg !9
  %217 = sub i64 %216, %215, !dbg !9
  %dma.buffer.offset1809 = trunc i64 %217 to i32, !dbg !9
  %218 = trunc i64 %215 to i32, !dbg !9
  %219 = insertelement <4 x i32> poison, i32 %218, i64 0, !dbg !9
  %220 = lshr i64 %215, 32, !dbg !9
  %221 = trunc i64 %220 to i32, !dbg !9
  %222 = insertelement <4 x i32> %219, i32 %221, i64 1, !dbg !9
  %223 = insertelement <4 x i32> %222, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1810 = insertelement <4 x i32> %223, i32 131072, i64 3, !dbg !9
  %224 = ptrtoint ptr addrspace(3) %214 to i32, !dbg !9
  %dma.buffer.lds1811 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %224), !dbg !9
  %225 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1811, i32 %dma.buffer.offset1809, <4 x i32> %dma.buffer.rsrc1810) #10, !dbg !9
  %226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !9
  %227 = zext i32 %226 to i64, !dbg !9
  %228 = inttoptr i64 %227 to ptr addrspace(3), !dbg !9
  %address.local402 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local279, !dbg !9
  %229 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %230 = ptrtoint ptr addrspace(1) %address.local402 to i64, !dbg !9
  %231 = sub i64 %230, %229, !dbg !9
  %dma.buffer.offset1806 = trunc i64 %231 to i32, !dbg !9
  %232 = trunc i64 %229 to i32, !dbg !9
  %233 = insertelement <4 x i32> poison, i32 %232, i64 0, !dbg !9
  %234 = lshr i64 %229, 32, !dbg !9
  %235 = trunc i64 %234 to i32, !dbg !9
  %236 = insertelement <4 x i32> %233, i32 %235, i64 1, !dbg !9
  %237 = insertelement <4 x i32> %236, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1807 = insertelement <4 x i32> %237, i32 131072, i64 3, !dbg !9
  %238 = ptrtoint ptr addrspace(3) %228 to i32, !dbg !9
  %dma.buffer.lds1808 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %238), !dbg !9
  %239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1808, i32 %dma.buffer.offset1806, <4 x i32> %dma.buffer.rsrc1807) #10, !dbg !9
  %240 = add nuw i64 %62, 1024, !dbg !16
  %241 = icmp slt i64 %240, 0, !dbg !16
  %242 = sub nuw i64 -1025, %62, !dbg !16
  %243 = select i1 %241, i64 %242, i64 %240, !dbg !16
  %244 = sdiv i64 %243, 4096, !dbg !16
  %.lobit = ashr i64 %240, 63, !dbg !16
  %245 = xor i64 %244, %.lobit, !dbg !16
  %246 = add nsw i64 %245, %15, !dbg !16
  %247 = add nsw i64 %102, %53, !dbg !16
  %248 = add nsw i64 %247, 32, !dbg !16
  %249 = icmp slt i64 %247, -32, !dbg !16
  %250 = sub nsw i64 -33, %247, !dbg !16
  %251 = select i1 %249, i64 %250, i64 %248, !dbg !16
  %252 = sdiv i64 %251, 128, !dbg !16
  %253 = sext i1 %249 to i64, !dbg !16
  %254 = xor i64 %252, %253, !dbg !16
  %255 = shl nsw i64 %254, 7, !dbg !16
  %256 = sub i64 %53, %255, !dbg !16
  %257 = shl i64 %246, 19, !dbg !16
  %258 = and i64 %257, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %259 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %258, !dbg !16
  %260 = getelementptr [2 x i8], ptr addrspace(1) %259, i64 %102, !dbg !16
  %261 = getelementptr [2 x i8], ptr addrspace(1) %260, i64 %256, !dbg !16
  %262 = getelementptr i8, ptr addrspace(1) %261, i64 64, !dbg !16
  %263 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %108, !dbg !16
  %264 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !16
  %265 = zext i32 %264 to i64, !dbg !16
  %266 = inttoptr i64 %265 to ptr addrspace(3), !dbg !16
  %address.local403 = getelementptr [2 x i8], ptr addrspace(1) %263, i64 %address.local293, !dbg !16
  %267 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %268 = ptrtoint ptr addrspace(1) %address.local403 to i64, !dbg !16
  %269 = sub i64 %268, %267, !dbg !16
  %dma.buffer.offset1803 = trunc i64 %269 to i32, !dbg !16
  %270 = trunc i64 %267 to i32, !dbg !16
  %271 = insertelement <4 x i32> poison, i32 %270, i64 0, !dbg !16
  %272 = lshr i64 %267, 32, !dbg !16
  %273 = trunc i64 %272 to i32, !dbg !16
  %274 = insertelement <4 x i32> %271, i32 %273, i64 1, !dbg !16
  %275 = insertelement <4 x i32> %274, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1804 = insertelement <4 x i32> %275, i32 131072, i64 3, !dbg !16
  %276 = ptrtoint ptr addrspace(3) %266 to i32, !dbg !16
  %dma.buffer.lds1805 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %276), !dbg !16
  %277 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1805, i32 %dma.buffer.offset1803, <4 x i32> %dma.buffer.rsrc1804) #10, !dbg !16
  %278 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !16
  %279 = zext i32 %278 to i64, !dbg !16
  %280 = inttoptr i64 %279 to ptr addrspace(3), !dbg !16
  %address.local404 = getelementptr [2 x i8], ptr addrspace(1) %263, i64 %address.local307, !dbg !16
  %281 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %282 = ptrtoint ptr addrspace(1) %address.local404 to i64, !dbg !16
  %283 = sub i64 %282, %281, !dbg !16
  %dma.buffer.offset1800 = trunc i64 %283 to i32, !dbg !16
  %284 = trunc i64 %281 to i32, !dbg !16
  %285 = insertelement <4 x i32> poison, i32 %284, i64 0, !dbg !16
  %286 = lshr i64 %281, 32, !dbg !16
  %287 = trunc i64 %286 to i32, !dbg !16
  %288 = insertelement <4 x i32> %285, i32 %287, i64 1, !dbg !16
  %289 = insertelement <4 x i32> %288, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1801 = insertelement <4 x i32> %289, i32 131072, i64 3, !dbg !16
  %290 = ptrtoint ptr addrspace(3) %280 to i32, !dbg !16
  %dma.buffer.lds1802 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %290), !dbg !16
  %291 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1802, i32 %dma.buffer.offset1800, <4 x i32> %dma.buffer.rsrc1801) #10, !dbg !16
  %292 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !10
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !10
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !10
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !10
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !10
  %295 = fmul <4 x half> %292, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !10
  %296 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !10
  %297 = fmul <4 x half> %293, splat (half 1.275630e-01), !dbg !10
  %298 = fmul <4 x half> %294, splat (half 1.275630e-01), !dbg !10
  %299 = fmul <4 x half> %296, splat (half 1.275630e-01), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
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
  %364 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !9
  %365 = zext i32 %364 to i64, !dbg !9
  %366 = inttoptr i64 %365 to ptr addrspace(3), !dbg !9
  %address.local405 = getelementptr [2 x i8], ptr addrspace(1) %55, i64 %address.local267, !dbg !9
  %367 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %368 = ptrtoint ptr addrspace(1) %address.local405 to i64, !dbg !9
  %369 = sub i64 %368, %367, !dbg !9
  %dma.buffer.offset1797 = trunc i64 %369 to i32, !dbg !9
  %370 = trunc i64 %367 to i32, !dbg !9
  %371 = insertelement <4 x i32> poison, i32 %370, i64 0, !dbg !9
  %372 = lshr i64 %367, 32, !dbg !9
  %373 = trunc i64 %372 to i32, !dbg !9
  %374 = insertelement <4 x i32> %371, i32 %373, i64 1, !dbg !9
  %375 = insertelement <4 x i32> %374, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1798 = insertelement <4 x i32> %375, i32 131072, i64 3, !dbg !9
  %376 = ptrtoint ptr addrspace(3) %366 to i32, !dbg !9
  %dma.buffer.lds1799 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %376), !dbg !9
  %377 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1799, i32 %dma.buffer.offset1797, <4 x i32> %dma.buffer.rsrc1798) #10, !dbg !9
  %378 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !9
  %379 = zext i32 %378 to i64, !dbg !9
  %380 = inttoptr i64 %379 to ptr addrspace(3), !dbg !9
  %address.local406 = getelementptr [2 x i8], ptr addrspace(1) %55, i64 %address.local279, !dbg !9
  %381 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %382 = ptrtoint ptr addrspace(1) %address.local406 to i64, !dbg !9
  %383 = sub i64 %382, %381, !dbg !9
  %dma.buffer.offset1794 = trunc i64 %383 to i32, !dbg !9
  %384 = trunc i64 %381 to i32, !dbg !9
  %385 = insertelement <4 x i32> poison, i32 %384, i64 0, !dbg !9
  %386 = lshr i64 %381, 32, !dbg !9
  %387 = trunc i64 %386 to i32, !dbg !9
  %388 = insertelement <4 x i32> %385, i32 %387, i64 1, !dbg !9
  %389 = insertelement <4 x i32> %388, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1795 = insertelement <4 x i32> %389, i32 131072, i64 3, !dbg !9
  %390 = ptrtoint ptr addrspace(3) %380 to i32, !dbg !9
  %dma.buffer.lds1796 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %390), !dbg !9
  %391 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1796, i32 %dma.buffer.offset1794, <4 x i32> %dma.buffer.rsrc1795) #10, !dbg !9
  %392 = add nuw i64 %62, 2048, !dbg !16
  %393 = icmp slt i64 %392, 0, !dbg !16
  %394 = sub nuw i64 -2049, %62, !dbg !16
  %395 = select i1 %393, i64 %394, i64 %392, !dbg !16
  %396 = sdiv i64 %395, 4096, !dbg !16
  %.lobit2 = ashr i64 %392, 63, !dbg !16
  %397 = xor i64 %396, %.lobit2, !dbg !16
  %398 = add nsw i64 %397, %15, !dbg !16
  %399 = add nsw i64 %247, 64, !dbg !16
  %400 = icmp slt i64 %247, -64, !dbg !16
  %401 = sub nsw i64 -65, %247, !dbg !16
  %402 = select i1 %400, i64 %401, i64 %399, !dbg !16
  %403 = sdiv i64 %402, 128, !dbg !16
  %404 = sext i1 %400 to i64, !dbg !16
  %405 = xor i64 %403, %404, !dbg !16
  %406 = shl nsw i64 %405, 7, !dbg !16
  %407 = sub i64 %53, %406, !dbg !16
  %408 = shl i64 %398, 19, !dbg !16
  %409 = and i64 %408, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %410 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %409, !dbg !16
  %411 = getelementptr [2 x i8], ptr addrspace(1) %410, i64 %102, !dbg !16
  %412 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %407, !dbg !16
  %413 = getelementptr i8, ptr addrspace(1) %412, i64 128, !dbg !16
  %414 = getelementptr [2 x i8], ptr addrspace(1) %413, i64 %108, !dbg !16
  %415 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !16
  %416 = zext i32 %415 to i64, !dbg !16
  %417 = inttoptr i64 %416 to ptr addrspace(3), !dbg !16
  %address.local407 = getelementptr [2 x i8], ptr addrspace(1) %414, i64 %address.local293, !dbg !16
  %418 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %419 = ptrtoint ptr addrspace(1) %address.local407 to i64, !dbg !16
  %420 = sub i64 %419, %418, !dbg !16
  %dma.buffer.offset1791 = trunc i64 %420 to i32, !dbg !16
  %421 = trunc i64 %418 to i32, !dbg !16
  %422 = insertelement <4 x i32> poison, i32 %421, i64 0, !dbg !16
  %423 = lshr i64 %418, 32, !dbg !16
  %424 = trunc i64 %423 to i32, !dbg !16
  %425 = insertelement <4 x i32> %422, i32 %424, i64 1, !dbg !16
  %426 = insertelement <4 x i32> %425, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1792 = insertelement <4 x i32> %426, i32 131072, i64 3, !dbg !16
  %427 = ptrtoint ptr addrspace(3) %417 to i32, !dbg !16
  %dma.buffer.lds1793 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %427), !dbg !16
  %428 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1793, i32 %dma.buffer.offset1791, <4 x i32> %dma.buffer.rsrc1792) #10, !dbg !16
  %429 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !16
  %430 = zext i32 %429 to i64, !dbg !16
  %431 = inttoptr i64 %430 to ptr addrspace(3), !dbg !16
  %address.local408 = getelementptr [2 x i8], ptr addrspace(1) %414, i64 %address.local307, !dbg !16
  %432 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %433 = ptrtoint ptr addrspace(1) %address.local408 to i64, !dbg !16
  %434 = sub i64 %433, %432, !dbg !16
  %dma.buffer.offset1788 = trunc i64 %434 to i32, !dbg !16
  %435 = trunc i64 %432 to i32, !dbg !16
  %436 = insertelement <4 x i32> poison, i32 %435, i64 0, !dbg !16
  %437 = lshr i64 %432, 32, !dbg !16
  %438 = trunc i64 %437 to i32, !dbg !16
  %439 = insertelement <4 x i32> %436, i32 %438, i64 1, !dbg !16
  %440 = insertelement <4 x i32> %439, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1789 = insertelement <4 x i32> %440, i32 131072, i64 3, !dbg !16
  %441 = ptrtoint ptr addrspace(3) %431 to i32, !dbg !16
  %dma.buffer.lds1790 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %441), !dbg !16
  %442 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1790, i32 %dma.buffer.offset1788, <4 x i32> %dma.buffer.rsrc1789) #10, !dbg !16
  %443 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !10
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !10
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !10
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !10
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !10
  %446 = fmul <4 x half> %443, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !10
  %447 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !10
  %448 = fmul <4 x half> %444, splat (half 1.275630e-01), !dbg !10
  %449 = fmul <4 x half> %445, splat (half 1.275630e-01), !dbg !10
  %450 = fmul <4 x half> %447, splat (half 1.275630e-01), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
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
  %515 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !9
  %516 = zext i32 %515 to i64, !dbg !9
  %517 = inttoptr i64 %516 to ptr addrspace(3), !dbg !9
  %address.local409 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local267, !dbg !9
  %518 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %519 = ptrtoint ptr addrspace(1) %address.local409 to i64, !dbg !9
  %520 = sub i64 %519, %518, !dbg !9
  %dma.buffer.offset1785 = trunc i64 %520 to i32, !dbg !9
  %521 = trunc i64 %518 to i32, !dbg !9
  %522 = insertelement <4 x i32> poison, i32 %521, i64 0, !dbg !9
  %523 = lshr i64 %518, 32, !dbg !9
  %524 = trunc i64 %523 to i32, !dbg !9
  %525 = insertelement <4 x i32> %522, i32 %524, i64 1, !dbg !9
  %526 = insertelement <4 x i32> %525, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1786 = insertelement <4 x i32> %526, i32 131072, i64 3, !dbg !9
  %527 = ptrtoint ptr addrspace(3) %517 to i32, !dbg !9
  %dma.buffer.lds1787 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %527), !dbg !9
  %528 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1787, i32 %dma.buffer.offset1785, <4 x i32> %dma.buffer.rsrc1786) #10, !dbg !9
  %529 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local274), !dbg !9
  %530 = zext i32 %529 to i64, !dbg !9
  %531 = inttoptr i64 %530 to ptr addrspace(3), !dbg !9
  %address.local410 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local279, !dbg !9
  %532 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %533 = ptrtoint ptr addrspace(1) %address.local410 to i64, !dbg !9
  %534 = sub i64 %533, %532, !dbg !9
  %dma.buffer.offset1782 = trunc i64 %534 to i32, !dbg !9
  %535 = trunc i64 %532 to i32, !dbg !9
  %536 = insertelement <4 x i32> poison, i32 %535, i64 0, !dbg !9
  %537 = lshr i64 %532, 32, !dbg !9
  %538 = trunc i64 %537 to i32, !dbg !9
  %539 = insertelement <4 x i32> %536, i32 %538, i64 1, !dbg !9
  %540 = insertelement <4 x i32> %539, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1783 = insertelement <4 x i32> %540, i32 131072, i64 3, !dbg !9
  %541 = ptrtoint ptr addrspace(3) %531 to i32, !dbg !9
  %dma.buffer.lds1784 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %541), !dbg !9
  %542 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1784, i32 %dma.buffer.offset1782, <4 x i32> %dma.buffer.rsrc1783) #10, !dbg !9
  %543 = add nuw i64 %62, 3072, !dbg !16
  %544 = icmp slt i64 %543, 0, !dbg !16
  %545 = sub nuw i64 -3073, %62, !dbg !16
  %546 = select i1 %544, i64 %545, i64 %543, !dbg !16
  %547 = sdiv i64 %546, 4096, !dbg !16
  %.lobit3 = ashr i64 %543, 63, !dbg !16
  %548 = xor i64 %547, %.lobit3, !dbg !16
  %549 = add nsw i64 %548, %15, !dbg !16
  %550 = add nsw i64 %247, 96, !dbg !16
  %551 = icmp slt i64 %247, -96, !dbg !16
  %552 = sub nsw i64 -97, %247, !dbg !16
  %553 = select i1 %551, i64 %552, i64 %550, !dbg !16
  %554 = sdiv i64 %553, 128, !dbg !16
  %555 = sext i1 %551 to i64, !dbg !16
  %556 = xor i64 %554, %555, !dbg !16
  %557 = shl nsw i64 %556, 7, !dbg !16
  %558 = sub i64 %53, %557, !dbg !16
  %559 = shl i64 %549, 19, !dbg !16
  %560 = and i64 %559, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %561 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %560, !dbg !16
  %562 = getelementptr [2 x i8], ptr addrspace(1) %561, i64 %102, !dbg !16
  %563 = getelementptr [2 x i8], ptr addrspace(1) %562, i64 %558, !dbg !16
  %564 = getelementptr i8, ptr addrspace(1) %563, i64 192, !dbg !16
  %565 = getelementptr [2 x i8], ptr addrspace(1) %564, i64 %108, !dbg !16
  %566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !16
  %567 = zext i32 %566 to i64, !dbg !16
  %568 = inttoptr i64 %567 to ptr addrspace(3), !dbg !16
  %address.local411 = getelementptr [2 x i8], ptr addrspace(1) %565, i64 %address.local293, !dbg !16
  %569 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %570 = ptrtoint ptr addrspace(1) %address.local411 to i64, !dbg !16
  %571 = sub i64 %570, %569, !dbg !16
  %dma.buffer.offset1779 = trunc i64 %571 to i32, !dbg !16
  %572 = trunc i64 %569 to i32, !dbg !16
  %573 = insertelement <4 x i32> poison, i32 %572, i64 0, !dbg !16
  %574 = lshr i64 %569, 32, !dbg !16
  %575 = trunc i64 %574 to i32, !dbg !16
  %576 = insertelement <4 x i32> %573, i32 %575, i64 1, !dbg !16
  %577 = insertelement <4 x i32> %576, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1780 = insertelement <4 x i32> %577, i32 131072, i64 3, !dbg !16
  %578 = ptrtoint ptr addrspace(3) %568 to i32, !dbg !16
  %dma.buffer.lds1781 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %578), !dbg !16
  %579 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1781, i32 %dma.buffer.offset1779, <4 x i32> %dma.buffer.rsrc1780) #10, !dbg !16
  %580 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local297), !dbg !16
  %581 = zext i32 %580 to i64, !dbg !16
  %582 = inttoptr i64 %581 to ptr addrspace(3), !dbg !16
  %address.local412 = getelementptr [2 x i8], ptr addrspace(1) %565, i64 %address.local307, !dbg !16
  %583 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %584 = ptrtoint ptr addrspace(1) %address.local412 to i64, !dbg !16
  %585 = sub i64 %584, %583, !dbg !16
  %dma.buffer.offset1776 = trunc i64 %585 to i32, !dbg !16
  %586 = trunc i64 %583 to i32, !dbg !16
  %587 = insertelement <4 x i32> poison, i32 %586, i64 0, !dbg !16
  %588 = lshr i64 %583, 32, !dbg !16
  %589 = trunc i64 %588 to i32, !dbg !16
  %590 = insertelement <4 x i32> %587, i32 %589, i64 1, !dbg !16
  %591 = insertelement <4 x i32> %590, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1777 = insertelement <4 x i32> %591, i32 131072, i64 3, !dbg !16
  %592 = ptrtoint ptr addrspace(3) %582 to i32, !dbg !16
  %dma.buffer.lds1778 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %592), !dbg !16
  %593 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1778, i32 %dma.buffer.offset1776, <4 x i32> %dma.buffer.rsrc1777) #10, !dbg !16
  %594 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !10
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !10
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !10
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !10
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !10
  %597 = fmul <4 x half> %594, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !10
  %598 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !10
  %599 = fmul <4 x half> %595, splat (half 1.275630e-01), !dbg !10
  %600 = fmul <4 x half> %596, splat (half 1.275630e-01), !dbg !10
  %601 = fmul <4 x half> %598, splat (half 1.275630e-01), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
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
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %666 = fadd <4 x float> %211, zeroinitializer, !dbg !20
  %667 = fadd <4 x float> %666, %363, !dbg !20
  %668 = fadd <4 x float> %667, %514, !dbg !20
  %669 = fadd <4 x float> %668, %665, !dbg !20
  %670 = extractelement <4 x float> %669, i64 3, !dbg !20
  %671 = extractelement <4 x float> %669, i64 2, !dbg !20
  %672 = extractelement <4 x float> %669, i64 1, !dbg !20
  %673 = extractelement <4 x float> %669, i64 0, !dbg !20
  %674 = fadd <4 x float> %208, zeroinitializer, !dbg !20
  %675 = fadd <4 x float> %674, %360, !dbg !20
  %676 = fadd <4 x float> %675, %511, !dbg !20
  %677 = fadd <4 x float> %676, %662, !dbg !20
  %678 = extractelement <4 x float> %677, i64 3, !dbg !20
  %679 = extractelement <4 x float> %677, i64 2, !dbg !20
  %680 = extractelement <4 x float> %677, i64 1, !dbg !20
  %681 = extractelement <4 x float> %677, i64 0, !dbg !20
  %682 = fadd <4 x float> %204, zeroinitializer, !dbg !20
  %683 = fadd <4 x float> %682, %356, !dbg !20
  %684 = fadd <4 x float> %683, %507, !dbg !20
  %685 = fadd <4 x float> %684, %658, !dbg !20
  %686 = extractelement <4 x float> %685, i64 3, !dbg !20
  %687 = extractelement <4 x float> %685, i64 2, !dbg !20
  %688 = extractelement <4 x float> %685, i64 1, !dbg !20
  %689 = extractelement <4 x float> %685, i64 0, !dbg !20
  %690 = fadd <4 x float> %200, zeroinitializer, !dbg !20
  %691 = fadd <4 x float> %690, %352, !dbg !20
  %692 = fadd <4 x float> %691, %503, !dbg !20
  %693 = fadd <4 x float> %692, %654, !dbg !20
  %694 = extractelement <4 x float> %693, i64 3, !dbg !20
  %695 = extractelement <4 x float> %693, i64 2, !dbg !20
  %696 = extractelement <4 x float> %693, i64 1, !dbg !20
  %697 = extractelement <4 x float> %693, i64 0, !dbg !20
  %698 = fadd <4 x float> %196, zeroinitializer, !dbg !20
  %699 = fadd <4 x float> %698, %348, !dbg !20
  %700 = fadd <4 x float> %699, %499, !dbg !20
  %701 = fadd <4 x float> %700, %650, !dbg !20
  %702 = extractelement <4 x float> %701, i64 3, !dbg !20
  %703 = extractelement <4 x float> %701, i64 2, !dbg !20
  %704 = extractelement <4 x float> %701, i64 1, !dbg !20
  %705 = extractelement <4 x float> %701, i64 0, !dbg !20
  %706 = fadd <4 x float> %192, zeroinitializer, !dbg !20
  %707 = fadd <4 x float> %706, %344, !dbg !20
  %708 = fadd <4 x float> %707, %495, !dbg !20
  %709 = fadd <4 x float> %708, %646, !dbg !20
  %710 = extractelement <4 x float> %709, i64 3, !dbg !20
  %711 = extractelement <4 x float> %709, i64 2, !dbg !20
  %712 = extractelement <4 x float> %709, i64 1, !dbg !20
  %713 = extractelement <4 x float> %709, i64 0, !dbg !20
  %714 = fadd <4 x float> %188, zeroinitializer, !dbg !20
  %715 = fadd <4 x float> %714, %340, !dbg !20
  %716 = fadd <4 x float> %715, %491, !dbg !20
  %717 = fadd <4 x float> %716, %642, !dbg !20
  %718 = extractelement <4 x float> %717, i64 3, !dbg !20
  %719 = extractelement <4 x float> %717, i64 2, !dbg !20
  %720 = extractelement <4 x float> %717, i64 1, !dbg !20
  %721 = extractelement <4 x float> %717, i64 0, !dbg !20
  %722 = fadd <4 x float> %184, zeroinitializer, !dbg !20
  %723 = fadd <4 x float> %722, %336, !dbg !20
  %724 = fadd <4 x float> %723, %487, !dbg !20
  %725 = fadd <4 x float> %724, %638, !dbg !20
  %726 = extractelement <4 x float> %725, i64 3, !dbg !20
  %727 = extractelement <4 x float> %725, i64 2, !dbg !20
  %728 = extractelement <4 x float> %725, i64 1, !dbg !20
  %729 = extractelement <4 x float> %725, i64 0, !dbg !20
  %730 = fadd <4 x float> %180, zeroinitializer, !dbg !20
  %731 = fadd <4 x float> %730, %332, !dbg !20
  %732 = fadd <4 x float> %731, %483, !dbg !20
  %733 = fadd <4 x float> %732, %634, !dbg !20
  %734 = extractelement <4 x float> %733, i64 3, !dbg !20
  %735 = extractelement <4 x float> %733, i64 2, !dbg !20
  %736 = extractelement <4 x float> %733, i64 1, !dbg !20
  %737 = extractelement <4 x float> %733, i64 0, !dbg !20
  %738 = fadd <4 x float> %176, zeroinitializer, !dbg !20
  %739 = fadd <4 x float> %738, %328, !dbg !20
  %740 = fadd <4 x float> %739, %479, !dbg !20
  %741 = fadd <4 x float> %740, %630, !dbg !20
  %742 = extractelement <4 x float> %741, i64 3, !dbg !20
  %743 = extractelement <4 x float> %741, i64 2, !dbg !20
  %744 = extractelement <4 x float> %741, i64 1, !dbg !20
  %745 = extractelement <4 x float> %741, i64 0, !dbg !20
  %746 = fadd <4 x float> %172, zeroinitializer, !dbg !20
  %747 = fadd <4 x float> %746, %324, !dbg !20
  %748 = fadd <4 x float> %747, %475, !dbg !20
  %749 = fadd <4 x float> %748, %626, !dbg !20
  %750 = extractelement <4 x float> %749, i64 3, !dbg !20
  %751 = extractelement <4 x float> %749, i64 2, !dbg !20
  %752 = extractelement <4 x float> %749, i64 1, !dbg !20
  %753 = extractelement <4 x float> %749, i64 0, !dbg !20
  %754 = fadd <4 x float> %168, zeroinitializer, !dbg !20
  %755 = fadd <4 x float> %754, %320, !dbg !20
  %756 = fadd <4 x float> %755, %471, !dbg !20
  %757 = fadd <4 x float> %756, %622, !dbg !20
  %758 = extractelement <4 x float> %757, i64 3, !dbg !20
  %759 = extractelement <4 x float> %757, i64 2, !dbg !20
  %760 = extractelement <4 x float> %757, i64 1, !dbg !20
  %761 = extractelement <4 x float> %757, i64 0, !dbg !20
  %762 = fadd <4 x float> %164, zeroinitializer, !dbg !20
  %763 = fadd <4 x float> %762, %316, !dbg !20
  %764 = fadd <4 x float> %763, %467, !dbg !20
  %765 = fadd <4 x float> %764, %618, !dbg !20
  %766 = extractelement <4 x float> %765, i64 3, !dbg !20
  %767 = extractelement <4 x float> %765, i64 2, !dbg !20
  %768 = extractelement <4 x float> %765, i64 1, !dbg !20
  %769 = extractelement <4 x float> %765, i64 0, !dbg !20
  %770 = fadd <4 x float> %160, zeroinitializer, !dbg !20
  %771 = fadd <4 x float> %770, %312, !dbg !20
  %772 = fadd <4 x float> %771, %463, !dbg !20
  %773 = fadd <4 x float> %772, %614, !dbg !20
  %774 = extractelement <4 x float> %773, i64 3, !dbg !20
  %775 = extractelement <4 x float> %773, i64 2, !dbg !20
  %776 = extractelement <4 x float> %773, i64 1, !dbg !20
  %777 = extractelement <4 x float> %773, i64 0, !dbg !20
  %778 = fadd <4 x float> %156, zeroinitializer, !dbg !20
  %779 = fadd <4 x float> %778, %308, !dbg !20
  %780 = fadd <4 x float> %779, %459, !dbg !20
  %781 = fadd <4 x float> %780, %610, !dbg !20
  %782 = extractelement <4 x float> %781, i64 3, !dbg !20
  %783 = extractelement <4 x float> %781, i64 2, !dbg !20
  %784 = extractelement <4 x float> %781, i64 1, !dbg !20
  %785 = extractelement <4 x float> %781, i64 0, !dbg !20
  %786 = fadd <4 x float> %152, zeroinitializer, !dbg !20
  %787 = fadd <4 x float> %786, %304, !dbg !20
  %788 = fadd <4 x float> %787, %455, !dbg !20
  %789 = fadd <4 x float> %788, %606, !dbg !20
  %790 = extractelement <4 x float> %789, i64 3, !dbg !20
  %791 = extractelement <4 x float> %789, i64 2, !dbg !20
  %792 = extractelement <4 x float> %789, i64 1, !dbg !20
  %793 = extractelement <4 x float> %789, i64 0, !dbg !20
  %address.thread413 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !17, !range !11, !frisk.register_lifetimes !5
  %address.local414 = shl nuw nsw i32 %address.thread413, 5, !dbg !17
  %address.local415 = and i32 %address.local414, 6144, !dbg !17
  %address.local416 = shl nuw nsw i32 %address.thread413, 7, !dbg !17
  %address.local417 = and i32 %address.local416, 1920, !dbg !17
  %address.local418 = or disjoint i32 %address.local415, %address.local417, !dbg !17
  %address.local419 = zext nneg i32 %address.local418 to i64, !dbg !17
  %address.local420 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local421 = zext nneg i32 %address.thread413 to i64, !dbg !10
  %address.local422 = lshr i64 %address.local421, 4, !dbg !10
  %address.local423 = and i64 %address.local422, 3, !dbg !10
  %address.local424 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local420, i64 %address.local423, !dbg !17
  store float %793, ptr addrspace(3) %address.local424, align 4, !dbg !17
  %address.local425 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local426 = or disjoint i64 %address.local423, 4, !dbg !17
  %address.local427 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local425, i64 %address.local426, !dbg !17
  store float %792, ptr addrspace(3) %address.local427, align 4, !dbg !17
  %address.local428 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local429 = or disjoint i64 %address.local423, 8, !dbg !17
  %address.local430 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local428, i64 %address.local429, !dbg !17
  store float %791, ptr addrspace(3) %address.local430, align 4, !dbg !17
  %address.local431 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local432 = or i64 %address.local422, 12, !dbg !17
  %address.local433 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local431, i64 %address.local432, !dbg !17
  store float %790, ptr addrspace(3) %address.local433, align 4, !dbg !17
  %address.local434 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local435 = or disjoint i64 %address.local423, 16, !dbg !17
  %address.local436 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local434, i64 %address.local435, !dbg !17
  store float %785, ptr addrspace(3) %address.local436, align 4, !dbg !17
  %address.local437 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local438 = or disjoint i64 %address.local423, 20, !dbg !17
  %address.local439 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local437, i64 %address.local438, !dbg !17
  store float %784, ptr addrspace(3) %address.local439, align 4, !dbg !17
  %address.local440 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local441 = or disjoint i64 %address.local423, 24, !dbg !17
  %address.local442 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local440, i64 %address.local441, !dbg !17
  store float %783, ptr addrspace(3) %address.local442, align 4, !dbg !17
  %address.local443 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local444 = or i64 %address.local422, 28, !dbg !17
  %address.local445 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local443, i64 %address.local444, !dbg !17
  store float %782, ptr addrspace(3) %address.local445, align 4, !dbg !17
  %address.local446 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local447 = or disjoint i64 %address.local423, 32, !dbg !17
  %address.local448 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local446, i64 %address.local447, !dbg !17
  store float %777, ptr addrspace(3) %address.local448, align 4, !dbg !17
  %address.local449 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local450 = or disjoint i64 %address.local423, 36, !dbg !17
  %address.local451 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local449, i64 %address.local450, !dbg !17
  store float %776, ptr addrspace(3) %address.local451, align 4, !dbg !17
  %address.local452 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local453 = or disjoint i64 %address.local423, 40, !dbg !17
  %address.local454 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local452, i64 %address.local453, !dbg !17
  store float %775, ptr addrspace(3) %address.local454, align 4, !dbg !17
  %address.local455 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local456 = or i64 %address.local422, 44, !dbg !17
  %address.local457 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local455, i64 %address.local456, !dbg !17
  store float %774, ptr addrspace(3) %address.local457, align 4, !dbg !17
  %address.local458 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local459 = or disjoint i64 %address.local423, 48, !dbg !17
  %address.local460 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local458, i64 %address.local459, !dbg !17
  store float %769, ptr addrspace(3) %address.local460, align 4, !dbg !17
  %address.local461 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local462 = or disjoint i64 %address.local423, 52, !dbg !17
  %address.local463 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local461, i64 %address.local462, !dbg !17
  store float %768, ptr addrspace(3) %address.local463, align 4, !dbg !17
  %address.local464 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local465 = or disjoint i64 %address.local423, 56, !dbg !17
  %address.local466 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local464, i64 %address.local465, !dbg !17
  store float %767, ptr addrspace(3) %address.local466, align 4, !dbg !17
  %address.local467 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local468 = or i64 %address.local422, 60, !dbg !17
  %address.local469 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local467, i64 %address.local468, !dbg !17
  store float %766, ptr addrspace(3) %address.local469, align 4, !dbg !17
  %address.local470 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local471 = or disjoint i64 %address.local423, 64, !dbg !17
  %address.local472 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local470, i64 %address.local471, !dbg !17
  store float %761, ptr addrspace(3) %address.local472, align 4, !dbg !17
  %address.local473 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local474 = or disjoint i64 %address.local423, 68, !dbg !17
  %address.local475 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local473, i64 %address.local474, !dbg !17
  store float %760, ptr addrspace(3) %address.local475, align 4, !dbg !17
  %address.local476 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local477 = or disjoint i64 %address.local423, 72, !dbg !17
  %address.local478 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local476, i64 %address.local477, !dbg !17
  store float %759, ptr addrspace(3) %address.local478, align 4, !dbg !17
  %address.local479 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local480 = or i64 %address.local422, 76, !dbg !17
  %address.local481 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local479, i64 %address.local480, !dbg !17
  store float %758, ptr addrspace(3) %address.local481, align 4, !dbg !17
  %address.local482 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local483 = or disjoint i64 %address.local423, 80, !dbg !17
  %address.local484 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local482, i64 %address.local483, !dbg !17
  store float %753, ptr addrspace(3) %address.local484, align 4, !dbg !17
  %address.local485 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local486 = or disjoint i64 %address.local423, 84, !dbg !17
  %address.local487 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local485, i64 %address.local486, !dbg !17
  store float %752, ptr addrspace(3) %address.local487, align 4, !dbg !17
  %address.local488 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local489 = or disjoint i64 %address.local423, 88, !dbg !17
  %address.local490 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local488, i64 %address.local489, !dbg !17
  store float %751, ptr addrspace(3) %address.local490, align 4, !dbg !17
  %address.local491 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local492 = or i64 %address.local422, 92, !dbg !17
  %address.local493 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local491, i64 %address.local492, !dbg !17
  store float %750, ptr addrspace(3) %address.local493, align 4, !dbg !17
  %address.local494 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local495 = or disjoint i64 %address.local423, 96, !dbg !17
  %address.local496 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local494, i64 %address.local495, !dbg !17
  store float %745, ptr addrspace(3) %address.local496, align 4, !dbg !17
  %address.local497 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local498 = or disjoint i64 %address.local423, 100, !dbg !17
  %address.local499 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local497, i64 %address.local498, !dbg !17
  store float %744, ptr addrspace(3) %address.local499, align 4, !dbg !17
  %address.local500 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local501 = or disjoint i64 %address.local423, 104, !dbg !17
  %address.local502 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local500, i64 %address.local501, !dbg !17
  store float %743, ptr addrspace(3) %address.local502, align 4, !dbg !17
  %address.local503 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local504 = or i64 %address.local422, 108, !dbg !17
  %address.local505 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local503, i64 %address.local504, !dbg !17
  store float %742, ptr addrspace(3) %address.local505, align 4, !dbg !17
  %address.local506 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local507 = or disjoint i64 %address.local423, 112, !dbg !17
  %address.local508 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local506, i64 %address.local507, !dbg !17
  store float %737, ptr addrspace(3) %address.local508, align 4, !dbg !17
  %address.local509 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local510 = or disjoint i64 %address.local423, 116, !dbg !17
  %address.local511 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local509, i64 %address.local510, !dbg !17
  store float %736, ptr addrspace(3) %address.local511, align 4, !dbg !17
  %address.local512 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local513 = or disjoint i64 %address.local423, 120, !dbg !17
  %address.local514 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local512, i64 %address.local513, !dbg !17
  store float %735, ptr addrspace(3) %address.local514, align 4, !dbg !17
  %address.local515 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local419, !dbg !17
  %address.local516 = or i64 %address.local422, 124, !dbg !17
  %address.local517 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local515, i64 %address.local516, !dbg !17
  store float %734, ptr addrspace(3) %address.local517, align 4, !dbg !17
  %address.local518 = shl nuw nsw i32 %address.thread413, 5, !dbg !17
  %address.local519 = and i32 %address.local518, 6144, !dbg !17
  %address.local520 = shl nuw nsw i32 %address.thread413, 7, !dbg !17
  %address.local521 = and i32 %address.local520, 1920, !dbg !17
  %address.local522 = or disjoint i32 %address.local519, %address.local521, !dbg !17
  %address.local523 = or disjoint i32 %address.local522, 8192, !dbg !17
  %address.local524 = zext nneg i32 %address.local523 to i64, !dbg !17
  %address.local525 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local526 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local525, i64 %address.local423, !dbg !17
  store float %729, ptr addrspace(3) %address.local526, align 4, !dbg !17
  %address.local527 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local528 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local527, i64 %address.local426, !dbg !17
  store float %728, ptr addrspace(3) %address.local528, align 4, !dbg !17
  %address.local529 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local530 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local529, i64 %address.local429, !dbg !17
  store float %727, ptr addrspace(3) %address.local530, align 4, !dbg !17
  %address.local531 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local532 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local531, i64 %address.local432, !dbg !17
  store float %726, ptr addrspace(3) %address.local532, align 4, !dbg !17
  %address.local533 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local534 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local533, i64 %address.local435, !dbg !17
  store float %721, ptr addrspace(3) %address.local534, align 4, !dbg !17
  %address.local535 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local536 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local535, i64 %address.local438, !dbg !17
  store float %720, ptr addrspace(3) %address.local536, align 4, !dbg !17
  %address.local537 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local538 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local537, i64 %address.local441, !dbg !17
  store float %719, ptr addrspace(3) %address.local538, align 4, !dbg !17
  %address.local539 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local540 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local539, i64 %address.local444, !dbg !17
  store float %718, ptr addrspace(3) %address.local540, align 4, !dbg !17
  %address.local541 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local542 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local541, i64 %address.local447, !dbg !17
  store float %713, ptr addrspace(3) %address.local542, align 4, !dbg !17
  %address.local543 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local544 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local543, i64 %address.local450, !dbg !17
  store float %712, ptr addrspace(3) %address.local544, align 4, !dbg !17
  %address.local545 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local546 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local545, i64 %address.local453, !dbg !17
  store float %711, ptr addrspace(3) %address.local546, align 4, !dbg !17
  %address.local547 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local548 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local547, i64 %address.local456, !dbg !17
  store float %710, ptr addrspace(3) %address.local548, align 4, !dbg !17
  %address.local549 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local550 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local549, i64 %address.local459, !dbg !17
  store float %705, ptr addrspace(3) %address.local550, align 4, !dbg !17
  %address.local551 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local552 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local551, i64 %address.local462, !dbg !17
  store float %704, ptr addrspace(3) %address.local552, align 4, !dbg !17
  %address.local553 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local554 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local553, i64 %address.local465, !dbg !17
  store float %703, ptr addrspace(3) %address.local554, align 4, !dbg !17
  %address.local555 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local556 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local555, i64 %address.local468, !dbg !17
  store float %702, ptr addrspace(3) %address.local556, align 4, !dbg !17
  %address.local557 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local558 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local557, i64 %address.local471, !dbg !17
  store float %697, ptr addrspace(3) %address.local558, align 4, !dbg !17
  %address.local559 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local560 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local559, i64 %address.local474, !dbg !17
  store float %696, ptr addrspace(3) %address.local560, align 4, !dbg !17
  %address.local561 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local562 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local561, i64 %address.local477, !dbg !17
  store float %695, ptr addrspace(3) %address.local562, align 4, !dbg !17
  %address.local563 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local564 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local563, i64 %address.local480, !dbg !17
  store float %694, ptr addrspace(3) %address.local564, align 4, !dbg !17
  %address.local565 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local566 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local565, i64 %address.local483, !dbg !17
  store float %689, ptr addrspace(3) %address.local566, align 4, !dbg !17
  %address.local567 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local568 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local567, i64 %address.local486, !dbg !17
  store float %688, ptr addrspace(3) %address.local568, align 4, !dbg !17
  %address.local569 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local570 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local569, i64 %address.local489, !dbg !17
  store float %687, ptr addrspace(3) %address.local570, align 4, !dbg !17
  %address.local571 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local572 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local571, i64 %address.local492, !dbg !17
  store float %686, ptr addrspace(3) %address.local572, align 4, !dbg !17
  %address.local573 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local574 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local573, i64 %address.local495, !dbg !17
  store float %681, ptr addrspace(3) %address.local574, align 4, !dbg !17
  %address.local575 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local576 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local575, i64 %address.local498, !dbg !17
  store float %680, ptr addrspace(3) %address.local576, align 4, !dbg !17
  %address.local577 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local578 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local577, i64 %address.local501, !dbg !17
  store float %679, ptr addrspace(3) %address.local578, align 4, !dbg !17
  %address.local579 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local580 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local579, i64 %address.local504, !dbg !17
  store float %678, ptr addrspace(3) %address.local580, align 4, !dbg !17
  %address.local581 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local582 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local581, i64 %address.local507, !dbg !17
  store float %673, ptr addrspace(3) %address.local582, align 4, !dbg !17
  %address.local583 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local584 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local583, i64 %address.local510, !dbg !17
  store float %672, ptr addrspace(3) %address.local584, align 4, !dbg !17
  %address.local585 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local586 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local585, i64 %address.local513, !dbg !17
  store float %671, ptr addrspace(3) %address.local586, align 4, !dbg !17
  %address.local587 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !17
  %address.local588 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local587, i64 %address.local516, !dbg !17
  store float %670, ptr addrspace(3) %address.local588, align 4, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread589 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !17, !range !11, !frisk.register_lifetimes !5
  %address.local590 = shl nuw nsw i32 %address.thread589, 5, !dbg !17
  %address.local591 = and i32 %address.local590, 6144, !dbg !17
  %address.local592 = shl nuw nsw i32 %address.thread589, 7, !dbg !17
  %address.local593 = and i32 %address.local592, 1920, !dbg !17
  %address.local594 = or disjoint i32 %address.local591, %address.local593, !dbg !17
  %address.local595 = zext nneg i32 %address.local594 to i64, !dbg !17
  %address.local596 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local595, !dbg !17
  %address.local597 = zext nneg i32 %address.thread589 to i64, !dbg !10
  %address.local598 = lshr i64 %address.local597, 4, !dbg !10
  %address.local599 = and i64 %address.local598, 3, !dbg !10
  %address.local600 = shl nuw nsw i64 %address.local599, 2, !dbg !10
  %address.local601 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %address.local596, i64 %address.local600, !dbg !17
  %794 = load float, ptr addrspace(3) %address.local601, align 16, !dbg !17
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local601, i64 4, !dbg !17
  %795 = load float, ptr addrspace(3) %lds.const.gep1682, align 4, !dbg !17
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local601, i64 8, !dbg !17
  %796 = load float, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !17
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local601, i64 12, !dbg !17
  %797 = load float, ptr addrspace(3) %lds.const.gep1684, align 4, !dbg !17
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local601, i64 64, !dbg !17
  %798 = load float, ptr addrspace(3) %lds.const.gep1685, align 16, !dbg !17
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local601, i64 68, !dbg !17
  %799 = load float, ptr addrspace(3) %lds.const.gep1686, align 4, !dbg !17
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local601, i64 72, !dbg !17
  %800 = load float, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !17
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local601, i64 76, !dbg !17
  %801 = load float, ptr addrspace(3) %lds.const.gep1688, align 4, !dbg !17
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local601, i64 128, !dbg !17
  %802 = load float, ptr addrspace(3) %lds.const.gep1689, align 16, !dbg !17
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local601, i64 132, !dbg !17
  %803 = load float, ptr addrspace(3) %lds.const.gep1690, align 4, !dbg !17
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local601, i64 136, !dbg !17
  %804 = load float, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !17
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local601, i64 140, !dbg !17
  %805 = load float, ptr addrspace(3) %lds.const.gep1692, align 4, !dbg !17
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local601, i64 192, !dbg !17
  %806 = load float, ptr addrspace(3) %lds.const.gep1693, align 16, !dbg !17
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local601, i64 196, !dbg !17
  %807 = load float, ptr addrspace(3) %lds.const.gep1694, align 4, !dbg !17
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local601, i64 200, !dbg !17
  %808 = load float, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !17
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local601, i64 204, !dbg !17
  %809 = load float, ptr addrspace(3) %lds.const.gep1696, align 4, !dbg !17
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local601, i64 256, !dbg !17
  %810 = load float, ptr addrspace(3) %lds.const.gep1697, align 16, !dbg !17
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local601, i64 260, !dbg !17
  %811 = load float, ptr addrspace(3) %lds.const.gep1698, align 4, !dbg !17
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local601, i64 264, !dbg !17
  %812 = load float, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !17
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local601, i64 268, !dbg !17
  %813 = load float, ptr addrspace(3) %lds.const.gep1700, align 4, !dbg !17
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local601, i64 320, !dbg !17
  %814 = load float, ptr addrspace(3) %lds.const.gep1701, align 16, !dbg !17
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %address.local601, i64 324, !dbg !17
  %815 = load float, ptr addrspace(3) %lds.const.gep1702, align 4, !dbg !17
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %address.local601, i64 328, !dbg !17
  %816 = load float, ptr addrspace(3) %lds.const.gep1703, align 8, !dbg !17
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local601, i64 332, !dbg !17
  %817 = load float, ptr addrspace(3) %lds.const.gep1704, align 4, !dbg !17
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local601, i64 384, !dbg !17
  %818 = load float, ptr addrspace(3) %lds.const.gep1705, align 16, !dbg !17
  %lds.const.gep1706 = getelementptr i8, ptr addrspace(3) %address.local601, i64 388, !dbg !17
  %819 = load float, ptr addrspace(3) %lds.const.gep1706, align 4, !dbg !17
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %address.local601, i64 392, !dbg !17
  %820 = load float, ptr addrspace(3) %lds.const.gep1707, align 8, !dbg !17
  %lds.const.gep1708 = getelementptr i8, ptr addrspace(3) %address.local601, i64 396, !dbg !17
  %821 = load float, ptr addrspace(3) %lds.const.gep1708, align 4, !dbg !17
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local601, i64 448, !dbg !17
  %822 = load float, ptr addrspace(3) %lds.const.gep1709, align 16, !dbg !17
  %lds.const.gep1710 = getelementptr i8, ptr addrspace(3) %address.local601, i64 452, !dbg !17
  %823 = load float, ptr addrspace(3) %lds.const.gep1710, align 4, !dbg !17
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %address.local601, i64 456, !dbg !17
  %824 = load float, ptr addrspace(3) %lds.const.gep1711, align 8, !dbg !17
  %lds.const.gep1712 = getelementptr i8, ptr addrspace(3) %address.local601, i64 460, !dbg !17
  %825 = load float, ptr addrspace(3) %lds.const.gep1712, align 4, !dbg !17
  %lds.const.gep1713 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32768, !dbg !17
  %826 = load float, ptr addrspace(3) %lds.const.gep1713, align 16, !dbg !17
  %lds.const.gep1714 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32772, !dbg !17
  %827 = load float, ptr addrspace(3) %lds.const.gep1714, align 4, !dbg !17
  %lds.const.gep1715 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32776, !dbg !17
  %828 = load float, ptr addrspace(3) %lds.const.gep1715, align 8, !dbg !17
  %lds.const.gep1716 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32780, !dbg !17
  %829 = load float, ptr addrspace(3) %lds.const.gep1716, align 4, !dbg !17
  %lds.const.gep1717 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32832, !dbg !17
  %830 = load float, ptr addrspace(3) %lds.const.gep1717, align 16, !dbg !17
  %lds.const.gep1718 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32836, !dbg !17
  %831 = load float, ptr addrspace(3) %lds.const.gep1718, align 4, !dbg !17
  %lds.const.gep1719 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32840, !dbg !17
  %832 = load float, ptr addrspace(3) %lds.const.gep1719, align 8, !dbg !17
  %lds.const.gep1720 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32844, !dbg !17
  %833 = load float, ptr addrspace(3) %lds.const.gep1720, align 4, !dbg !17
  %lds.const.gep1721 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32896, !dbg !17
  %834 = load float, ptr addrspace(3) %lds.const.gep1721, align 16, !dbg !17
  %lds.const.gep1722 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32900, !dbg !17
  %835 = load float, ptr addrspace(3) %lds.const.gep1722, align 4, !dbg !17
  %lds.const.gep1723 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32904, !dbg !17
  %836 = load float, ptr addrspace(3) %lds.const.gep1723, align 8, !dbg !17
  %lds.const.gep1724 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32908, !dbg !17
  %837 = load float, ptr addrspace(3) %lds.const.gep1724, align 4, !dbg !17
  %lds.const.gep1725 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32960, !dbg !17
  %838 = load float, ptr addrspace(3) %lds.const.gep1725, align 16, !dbg !17
  %lds.const.gep1726 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32964, !dbg !17
  %839 = load float, ptr addrspace(3) %lds.const.gep1726, align 4, !dbg !17
  %lds.const.gep1727 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32968, !dbg !17
  %840 = load float, ptr addrspace(3) %lds.const.gep1727, align 8, !dbg !17
  %lds.const.gep1728 = getelementptr i8, ptr addrspace(3) %address.local601, i64 32972, !dbg !17
  %841 = load float, ptr addrspace(3) %lds.const.gep1728, align 4, !dbg !17
  %lds.const.gep1729 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33024, !dbg !17
  %842 = load float, ptr addrspace(3) %lds.const.gep1729, align 16, !dbg !17
  %lds.const.gep1730 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33028, !dbg !17
  %843 = load float, ptr addrspace(3) %lds.const.gep1730, align 4, !dbg !17
  %lds.const.gep1731 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33032, !dbg !17
  %844 = load float, ptr addrspace(3) %lds.const.gep1731, align 8, !dbg !17
  %lds.const.gep1732 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33036, !dbg !17
  %845 = load float, ptr addrspace(3) %lds.const.gep1732, align 4, !dbg !17
  %lds.const.gep1733 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33088, !dbg !17
  %846 = load float, ptr addrspace(3) %lds.const.gep1733, align 16, !dbg !17
  %lds.const.gep1734 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33092, !dbg !17
  %847 = load float, ptr addrspace(3) %lds.const.gep1734, align 4, !dbg !17
  %lds.const.gep1735 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33096, !dbg !17
  %848 = load float, ptr addrspace(3) %lds.const.gep1735, align 8, !dbg !17
  %lds.const.gep1736 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33100, !dbg !17
  %849 = load float, ptr addrspace(3) %lds.const.gep1736, align 4, !dbg !17
  %lds.const.gep1737 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33152, !dbg !17
  %850 = load float, ptr addrspace(3) %lds.const.gep1737, align 16, !dbg !17
  %lds.const.gep1738 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33156, !dbg !17
  %851 = load float, ptr addrspace(3) %lds.const.gep1738, align 4, !dbg !17
  %lds.const.gep1739 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33160, !dbg !17
  %852 = load float, ptr addrspace(3) %lds.const.gep1739, align 8, !dbg !17
  %lds.const.gep1740 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33164, !dbg !17
  %853 = load float, ptr addrspace(3) %lds.const.gep1740, align 4, !dbg !17
  %lds.const.gep1741 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33216, !dbg !17
  %854 = load float, ptr addrspace(3) %lds.const.gep1741, align 16, !dbg !17
  %lds.const.gep1742 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33220, !dbg !17
  %855 = load float, ptr addrspace(3) %lds.const.gep1742, align 4, !dbg !17
  %lds.const.gep1743 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33224, !dbg !17
  %856 = load float, ptr addrspace(3) %lds.const.gep1743, align 8, !dbg !17
  %lds.const.gep1744 = getelementptr i8, ptr addrspace(3) %address.local601, i64 33228, !dbg !17
  %857 = load float, ptr addrspace(3) %lds.const.gep1744, align 4, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %858 = or disjoint i64 %71, 127, !dbg !12
  %859 = icmp uge i64 %19, %858, !dbg !12
  %860 = and i1 %42, %859, !dbg !12
  br i1 %860, label %1002, label %861, !dbg !12

861:                                              ; preds = %70
  %862 = add i64 %19, %35, !dbg !12
  %863 = add i64 %862, %59, !dbg !12
  %864 = add i64 %863, %60, !dbg !12
  %865 = or disjoint i64 %71, %38, !dbg !12
  %866 = add i64 %864, 1, !dbg !21
  %867 = lshr i64 %866, 32, !dbg !22
  %compare.high = trunc i64 %867 to i32, !dbg !22
  %868 = lshr i64 %865, 32, !dbg !22
  %compare.high1303 = trunc i64 %868 to i32, !dbg !22
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1303, !dbg !22
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1303, !dbg !22
  %compare.low.base = trunc i64 %866 to i32, !dbg !22
  %compare.low.base1304 = trunc i64 %865 to i32, !dbg !22
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1304, !dbg !22
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !22
  %869 = select i1 %compare.result, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 1) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local = or i32 %compare.local.base, 1, !dbg !22
  %compare.low1305 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !22
  %compare.result1306 = select i1 %compare.high.equal, i1 %compare.low1305, i1 %compare.high.order, !dbg !22
  %870 = select i1 %compare.result1306, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1307 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 2) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1308 = or i32 %compare.local.base1307, 2, !dbg !22
  %compare.low1309 = icmp ugt i32 %compare.low.base, %compare.local1308, !dbg !22
  %compare.result1310 = select i1 %compare.high.equal, i1 %compare.low1309, i1 %compare.high.order, !dbg !22
  %871 = select i1 %compare.result1310, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1311 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 3) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1312 = or i32 %compare.local.base1311, 3, !dbg !22
  %compare.low1313 = icmp ugt i32 %compare.low.base, %compare.local1312, !dbg !22
  %compare.result1314 = select i1 %compare.high.equal, i1 %compare.low1313, i1 %compare.high.order, !dbg !22
  %872 = select i1 %compare.result1314, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1315 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 4) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1316 = or i32 %compare.local.base1315, 16, !dbg !22
  %compare.low1317 = icmp ugt i32 %compare.low.base, %compare.local1316, !dbg !22
  %compare.result1318 = select i1 %compare.high.equal, i1 %compare.low1317, i1 %compare.high.order, !dbg !22
  %873 = select i1 %compare.result1318, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1319 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 5) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1320 = or i32 %compare.local.base1319, 17, !dbg !22
  %compare.low1321 = icmp ugt i32 %compare.low.base, %compare.local1320, !dbg !22
  %compare.result1322 = select i1 %compare.high.equal, i1 %compare.low1321, i1 %compare.high.order, !dbg !22
  %874 = select i1 %compare.result1322, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1323 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 6) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1324 = or i32 %compare.local.base1323, 18, !dbg !22
  %compare.low1325 = icmp ugt i32 %compare.low.base, %compare.local1324, !dbg !22
  %compare.result1326 = select i1 %compare.high.equal, i1 %compare.low1325, i1 %compare.high.order, !dbg !22
  %875 = select i1 %compare.result1326, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 7) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1328 = or i32 %compare.local.base1327, 19, !dbg !22
  %compare.low1329 = icmp ugt i32 %compare.low.base, %compare.local1328, !dbg !22
  %compare.result1330 = select i1 %compare.high.equal, i1 %compare.low1329, i1 %compare.high.order, !dbg !22
  %876 = select i1 %compare.result1330, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 8) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1332 = or i32 %compare.local.base1331, 32, !dbg !22
  %compare.low1333 = icmp ugt i32 %compare.low.base, %compare.local1332, !dbg !22
  %compare.result1334 = select i1 %compare.high.equal, i1 %compare.low1333, i1 %compare.high.order, !dbg !22
  %877 = select i1 %compare.result1334, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 9) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1336 = or i32 %compare.local.base1335, 33, !dbg !22
  %compare.low1337 = icmp ugt i32 %compare.low.base, %compare.local1336, !dbg !22
  %compare.result1338 = select i1 %compare.high.equal, i1 %compare.low1337, i1 %compare.high.order, !dbg !22
  %878 = select i1 %compare.result1338, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 10) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1340 = or i32 %compare.local.base1339, 34, !dbg !22
  %compare.low1341 = icmp ugt i32 %compare.low.base, %compare.local1340, !dbg !22
  %compare.result1342 = select i1 %compare.high.equal, i1 %compare.low1341, i1 %compare.high.order, !dbg !22
  %879 = select i1 %compare.result1342, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 11) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1344 = or i32 %compare.local.base1343, 35, !dbg !22
  %compare.low1345 = icmp ugt i32 %compare.low.base, %compare.local1344, !dbg !22
  %compare.result1346 = select i1 %compare.high.equal, i1 %compare.low1345, i1 %compare.high.order, !dbg !22
  %880 = select i1 %compare.result1346, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 12) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1348 = or i32 %compare.local.base1347, 48, !dbg !22
  %compare.low1349 = icmp ugt i32 %compare.low.base, %compare.local1348, !dbg !22
  %compare.result1350 = select i1 %compare.high.equal, i1 %compare.low1349, i1 %compare.high.order, !dbg !22
  %881 = select i1 %compare.result1350, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 13) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1352 = or i32 %compare.local.base1351, 49, !dbg !22
  %compare.low1353 = icmp ugt i32 %compare.low.base, %compare.local1352, !dbg !22
  %compare.result1354 = select i1 %compare.high.equal, i1 %compare.low1353, i1 %compare.high.order, !dbg !22
  %882 = select i1 %compare.result1354, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 14) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1356 = or i32 %compare.local.base1355, 50, !dbg !22
  %compare.low1357 = icmp ugt i32 %compare.low.base, %compare.local1356, !dbg !22
  %compare.result1358 = select i1 %compare.high.equal, i1 %compare.low1357, i1 %compare.high.order, !dbg !22
  %883 = select i1 %compare.result1358, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 15) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1360 = or i32 %compare.local.base1359, 51, !dbg !22
  %compare.low1361 = icmp ugt i32 %compare.low.base, %compare.local1360, !dbg !22
  %compare.result1362 = select i1 %compare.high.equal, i1 %compare.low1361, i1 %compare.high.order, !dbg !22
  %884 = select i1 %compare.result1362, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 16) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1364 = or i32 %compare.local.base1363, 64, !dbg !22
  %compare.low1365 = icmp ugt i32 %compare.low.base, %compare.local1364, !dbg !22
  %compare.result1366 = select i1 %compare.high.equal, i1 %compare.low1365, i1 %compare.high.order, !dbg !22
  %885 = select i1 %compare.result1366, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 17) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1368 = or i32 %compare.local.base1367, 65, !dbg !22
  %compare.low1369 = icmp ugt i32 %compare.low.base, %compare.local1368, !dbg !22
  %compare.result1370 = select i1 %compare.high.equal, i1 %compare.low1369, i1 %compare.high.order, !dbg !22
  %886 = select i1 %compare.result1370, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 18) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1372 = or i32 %compare.local.base1371, 66, !dbg !22
  %compare.low1373 = icmp ugt i32 %compare.low.base, %compare.local1372, !dbg !22
  %compare.result1374 = select i1 %compare.high.equal, i1 %compare.low1373, i1 %compare.high.order, !dbg !22
  %887 = select i1 %compare.result1374, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 19) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1376 = or i32 %compare.local.base1375, 67, !dbg !22
  %compare.low1377 = icmp ugt i32 %compare.low.base, %compare.local1376, !dbg !22
  %compare.result1378 = select i1 %compare.high.equal, i1 %compare.low1377, i1 %compare.high.order, !dbg !22
  %888 = select i1 %compare.result1378, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 20) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1380 = or i32 %compare.local.base1379, 80, !dbg !22
  %compare.low1381 = icmp ugt i32 %compare.low.base, %compare.local1380, !dbg !22
  %compare.result1382 = select i1 %compare.high.equal, i1 %compare.low1381, i1 %compare.high.order, !dbg !22
  %889 = select i1 %compare.result1382, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 21) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1384 = or i32 %compare.local.base1383, 81, !dbg !22
  %compare.low1385 = icmp ugt i32 %compare.low.base, %compare.local1384, !dbg !22
  %compare.result1386 = select i1 %compare.high.equal, i1 %compare.low1385, i1 %compare.high.order, !dbg !22
  %890 = select i1 %compare.result1386, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 22) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1388 = or i32 %compare.local.base1387, 82, !dbg !22
  %compare.low1389 = icmp ugt i32 %compare.low.base, %compare.local1388, !dbg !22
  %compare.result1390 = select i1 %compare.high.equal, i1 %compare.low1389, i1 %compare.high.order, !dbg !22
  %891 = select i1 %compare.result1390, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 23) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1392 = or i32 %compare.local.base1391, 83, !dbg !22
  %compare.low1393 = icmp ugt i32 %compare.low.base, %compare.local1392, !dbg !22
  %compare.result1394 = select i1 %compare.high.equal, i1 %compare.low1393, i1 %compare.high.order, !dbg !22
  %892 = select i1 %compare.result1394, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 24) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1396 = or i32 %compare.local.base1395, 96, !dbg !22
  %compare.low1397 = icmp ugt i32 %compare.low.base, %compare.local1396, !dbg !22
  %compare.result1398 = select i1 %compare.high.equal, i1 %compare.low1397, i1 %compare.high.order, !dbg !22
  %893 = select i1 %compare.result1398, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 25) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1400 = or i32 %compare.local.base1399, 97, !dbg !22
  %compare.low1401 = icmp ugt i32 %compare.low.base, %compare.local1400, !dbg !22
  %compare.result1402 = select i1 %compare.high.equal, i1 %compare.low1401, i1 %compare.high.order, !dbg !22
  %894 = select i1 %compare.result1402, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 26) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1404 = or i32 %compare.local.base1403, 98, !dbg !22
  %compare.low1405 = icmp ugt i32 %compare.low.base, %compare.local1404, !dbg !22
  %compare.result1406 = select i1 %compare.high.equal, i1 %compare.low1405, i1 %compare.high.order, !dbg !22
  %895 = select i1 %compare.result1406, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 27) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1408 = or i32 %compare.local.base1407, 99, !dbg !22
  %compare.low1409 = icmp ugt i32 %compare.low.base, %compare.local1408, !dbg !22
  %compare.result1410 = select i1 %compare.high.equal, i1 %compare.low1409, i1 %compare.high.order, !dbg !22
  %896 = select i1 %compare.result1410, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 28) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1412 = or i32 %compare.local.base1411, 112, !dbg !22
  %compare.low1413 = icmp ugt i32 %compare.low.base, %compare.local1412, !dbg !22
  %compare.result1414 = select i1 %compare.high.equal, i1 %compare.low1413, i1 %compare.high.order, !dbg !22
  %897 = select i1 %compare.result1414, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 29) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1416 = or i32 %compare.local.base1415, 113, !dbg !22
  %compare.low1417 = icmp ugt i32 %compare.low.base, %compare.local1416, !dbg !22
  %compare.result1418 = select i1 %compare.high.equal, i1 %compare.low1417, i1 %compare.high.order, !dbg !22
  %898 = select i1 %compare.result1418, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 30) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1420 = or i32 %compare.local.base1419, 114, !dbg !22
  %compare.low1421 = icmp ugt i32 %compare.low.base, %compare.local1420, !dbg !22
  %compare.result1422 = select i1 %compare.high.equal, i1 %compare.low1421, i1 %compare.high.order, !dbg !22
  %899 = select i1 %compare.result1422, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1304, i32 31) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1424 = or i32 %compare.local.base1423, 115, !dbg !22
  %compare.low1425 = icmp ugt i32 %compare.low.base, %compare.local1424, !dbg !22
  %compare.result1426 = select i1 %compare.high.equal, i1 %compare.low1425, i1 %compare.high.order, !dbg !22
  %900 = select i1 %compare.result1426, float 0.000000e+00, float -inf, !dbg !23
  %901 = add i64 %864, 65, !dbg !21
  %902 = lshr i64 %901, 32, !dbg !22
  %compare.high1427 = trunc i64 %902 to i32, !dbg !22
  %903 = lshr i64 %865, 32, !dbg !22
  %compare.high1428 = trunc i64 %903 to i32, !dbg !22
  %compare.high.equal1429 = icmp eq i32 %compare.high1427, %compare.high1428, !dbg !22
  %compare.high.order1430 = icmp ugt i32 %compare.high1427, %compare.high1428, !dbg !22
  %compare.low.base1431 = trunc i64 %901 to i32, !dbg !22
  %compare.low.base1432 = trunc i64 %865 to i32, !dbg !22
  %compare.low1433 = icmp ugt i32 %compare.low.base1431, %compare.low.base1432, !dbg !22
  %compare.result1434 = select i1 %compare.high.equal1429, i1 %compare.low1433, i1 %compare.high.order1430, !dbg !22
  %904 = select i1 %compare.result1434, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 32) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1436 = or i32 %compare.local.base1435, 1, !dbg !22
  %compare.low1437 = icmp ugt i32 %compare.low.base1431, %compare.local1436, !dbg !22
  %compare.result1438 = select i1 %compare.high.equal1429, i1 %compare.low1437, i1 %compare.high.order1430, !dbg !22
  %905 = select i1 %compare.result1438, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 33) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1440 = or i32 %compare.local.base1439, 2, !dbg !22
  %compare.low1441 = icmp ugt i32 %compare.low.base1431, %compare.local1440, !dbg !22
  %compare.result1442 = select i1 %compare.high.equal1429, i1 %compare.low1441, i1 %compare.high.order1430, !dbg !22
  %906 = select i1 %compare.result1442, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 34) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1444 = or i32 %compare.local.base1443, 3, !dbg !22
  %compare.low1445 = icmp ugt i32 %compare.low.base1431, %compare.local1444, !dbg !22
  %compare.result1446 = select i1 %compare.high.equal1429, i1 %compare.low1445, i1 %compare.high.order1430, !dbg !22
  %907 = select i1 %compare.result1446, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1447 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 35) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1448 = or i32 %compare.local.base1447, 16, !dbg !22
  %compare.low1449 = icmp ugt i32 %compare.low.base1431, %compare.local1448, !dbg !22
  %compare.result1450 = select i1 %compare.high.equal1429, i1 %compare.low1449, i1 %compare.high.order1430, !dbg !22
  %908 = select i1 %compare.result1450, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1451 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 36) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1452 = or i32 %compare.local.base1451, 17, !dbg !22
  %compare.low1453 = icmp ugt i32 %compare.low.base1431, %compare.local1452, !dbg !22
  %compare.result1454 = select i1 %compare.high.equal1429, i1 %compare.low1453, i1 %compare.high.order1430, !dbg !22
  %909 = select i1 %compare.result1454, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1455 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 37) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1456 = or i32 %compare.local.base1455, 18, !dbg !22
  %compare.low1457 = icmp ugt i32 %compare.low.base1431, %compare.local1456, !dbg !22
  %compare.result1458 = select i1 %compare.high.equal1429, i1 %compare.low1457, i1 %compare.high.order1430, !dbg !22
  %910 = select i1 %compare.result1458, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 38) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1460 = or i32 %compare.local.base1459, 19, !dbg !22
  %compare.low1461 = icmp ugt i32 %compare.low.base1431, %compare.local1460, !dbg !22
  %compare.result1462 = select i1 %compare.high.equal1429, i1 %compare.low1461, i1 %compare.high.order1430, !dbg !22
  %911 = select i1 %compare.result1462, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 39) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1464 = or i32 %compare.local.base1463, 32, !dbg !22
  %compare.low1465 = icmp ugt i32 %compare.low.base1431, %compare.local1464, !dbg !22
  %compare.result1466 = select i1 %compare.high.equal1429, i1 %compare.low1465, i1 %compare.high.order1430, !dbg !22
  %912 = select i1 %compare.result1466, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 40) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1468 = or i32 %compare.local.base1467, 33, !dbg !22
  %compare.low1469 = icmp ugt i32 %compare.low.base1431, %compare.local1468, !dbg !22
  %compare.result1470 = select i1 %compare.high.equal1429, i1 %compare.low1469, i1 %compare.high.order1430, !dbg !22
  %913 = select i1 %compare.result1470, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 41) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1472 = or i32 %compare.local.base1471, 34, !dbg !22
  %compare.low1473 = icmp ugt i32 %compare.low.base1431, %compare.local1472, !dbg !22
  %compare.result1474 = select i1 %compare.high.equal1429, i1 %compare.low1473, i1 %compare.high.order1430, !dbg !22
  %914 = select i1 %compare.result1474, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 42) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1476 = or i32 %compare.local.base1475, 35, !dbg !22
  %compare.low1477 = icmp ugt i32 %compare.low.base1431, %compare.local1476, !dbg !22
  %compare.result1478 = select i1 %compare.high.equal1429, i1 %compare.low1477, i1 %compare.high.order1430, !dbg !22
  %915 = select i1 %compare.result1478, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 43) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1480 = or i32 %compare.local.base1479, 48, !dbg !22
  %compare.low1481 = icmp ugt i32 %compare.low.base1431, %compare.local1480, !dbg !22
  %compare.result1482 = select i1 %compare.high.equal1429, i1 %compare.low1481, i1 %compare.high.order1430, !dbg !22
  %916 = select i1 %compare.result1482, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 44) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1484 = or i32 %compare.local.base1483, 49, !dbg !22
  %compare.low1485 = icmp ugt i32 %compare.low.base1431, %compare.local1484, !dbg !22
  %compare.result1486 = select i1 %compare.high.equal1429, i1 %compare.low1485, i1 %compare.high.order1430, !dbg !22
  %917 = select i1 %compare.result1486, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1487 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 45) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1488 = or i32 %compare.local.base1487, 50, !dbg !22
  %compare.low1489 = icmp ugt i32 %compare.low.base1431, %compare.local1488, !dbg !22
  %compare.result1490 = select i1 %compare.high.equal1429, i1 %compare.low1489, i1 %compare.high.order1430, !dbg !22
  %918 = select i1 %compare.result1490, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1491 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 46) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1492 = or i32 %compare.local.base1491, 51, !dbg !22
  %compare.low1493 = icmp ugt i32 %compare.low.base1431, %compare.local1492, !dbg !22
  %compare.result1494 = select i1 %compare.high.equal1429, i1 %compare.low1493, i1 %compare.high.order1430, !dbg !22
  %919 = select i1 %compare.result1494, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1495 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 47) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1496 = or i32 %compare.local.base1495, 64, !dbg !22
  %compare.low1497 = icmp ugt i32 %compare.low.base1431, %compare.local1496, !dbg !22
  %compare.result1498 = select i1 %compare.high.equal1429, i1 %compare.low1497, i1 %compare.high.order1430, !dbg !22
  %920 = select i1 %compare.result1498, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1499 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 48) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1500 = or i32 %compare.local.base1499, 65, !dbg !22
  %compare.low1501 = icmp ugt i32 %compare.low.base1431, %compare.local1500, !dbg !22
  %compare.result1502 = select i1 %compare.high.equal1429, i1 %compare.low1501, i1 %compare.high.order1430, !dbg !22
  %921 = select i1 %compare.result1502, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1503 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 49) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1504 = or i32 %compare.local.base1503, 66, !dbg !22
  %compare.low1505 = icmp ugt i32 %compare.low.base1431, %compare.local1504, !dbg !22
  %compare.result1506 = select i1 %compare.high.equal1429, i1 %compare.low1505, i1 %compare.high.order1430, !dbg !22
  %922 = select i1 %compare.result1506, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1507 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 50) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1508 = or i32 %compare.local.base1507, 67, !dbg !22
  %compare.low1509 = icmp ugt i32 %compare.low.base1431, %compare.local1508, !dbg !22
  %compare.result1510 = select i1 %compare.high.equal1429, i1 %compare.low1509, i1 %compare.high.order1430, !dbg !22
  %923 = select i1 %compare.result1510, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1511 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 51) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1512 = or i32 %compare.local.base1511, 80, !dbg !22
  %compare.low1513 = icmp ugt i32 %compare.low.base1431, %compare.local1512, !dbg !22
  %compare.result1514 = select i1 %compare.high.equal1429, i1 %compare.low1513, i1 %compare.high.order1430, !dbg !22
  %924 = select i1 %compare.result1514, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1515 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 52) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1516 = or i32 %compare.local.base1515, 81, !dbg !22
  %compare.low1517 = icmp ugt i32 %compare.low.base1431, %compare.local1516, !dbg !22
  %compare.result1518 = select i1 %compare.high.equal1429, i1 %compare.low1517, i1 %compare.high.order1430, !dbg !22
  %925 = select i1 %compare.result1518, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1519 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 53) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1520 = or i32 %compare.local.base1519, 82, !dbg !22
  %compare.low1521 = icmp ugt i32 %compare.low.base1431, %compare.local1520, !dbg !22
  %compare.result1522 = select i1 %compare.high.equal1429, i1 %compare.low1521, i1 %compare.high.order1430, !dbg !22
  %926 = select i1 %compare.result1522, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 54) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1524 = or i32 %compare.local.base1523, 83, !dbg !22
  %compare.low1525 = icmp ugt i32 %compare.low.base1431, %compare.local1524, !dbg !22
  %compare.result1526 = select i1 %compare.high.equal1429, i1 %compare.low1525, i1 %compare.high.order1430, !dbg !22
  %927 = select i1 %compare.result1526, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 55) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1528 = or i32 %compare.local.base1527, 96, !dbg !22
  %compare.low1529 = icmp ugt i32 %compare.low.base1431, %compare.local1528, !dbg !22
  %compare.result1530 = select i1 %compare.high.equal1429, i1 %compare.low1529, i1 %compare.high.order1430, !dbg !22
  %928 = select i1 %compare.result1530, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 56) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1532 = or i32 %compare.local.base1531, 97, !dbg !22
  %compare.low1533 = icmp ugt i32 %compare.low.base1431, %compare.local1532, !dbg !22
  %compare.result1534 = select i1 %compare.high.equal1429, i1 %compare.low1533, i1 %compare.high.order1430, !dbg !22
  %929 = select i1 %compare.result1534, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 57) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1536 = or i32 %compare.local.base1535, 98, !dbg !22
  %compare.low1537 = icmp ugt i32 %compare.low.base1431, %compare.local1536, !dbg !22
  %compare.result1538 = select i1 %compare.high.equal1429, i1 %compare.low1537, i1 %compare.high.order1430, !dbg !22
  %930 = select i1 %compare.result1538, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 58) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1540 = or i32 %compare.local.base1539, 99, !dbg !22
  %compare.low1541 = icmp ugt i32 %compare.low.base1431, %compare.local1540, !dbg !22
  %compare.result1542 = select i1 %compare.high.equal1429, i1 %compare.low1541, i1 %compare.high.order1430, !dbg !22
  %931 = select i1 %compare.result1542, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 59) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1544 = or i32 %compare.local.base1543, 112, !dbg !22
  %compare.low1545 = icmp ugt i32 %compare.low.base1431, %compare.local1544, !dbg !22
  %compare.result1546 = select i1 %compare.high.equal1429, i1 %compare.low1545, i1 %compare.high.order1430, !dbg !22
  %932 = select i1 %compare.result1546, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 60) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1548 = or i32 %compare.local.base1547, 113, !dbg !22
  %compare.low1549 = icmp ugt i32 %compare.low.base1431, %compare.local1548, !dbg !22
  %compare.result1550 = select i1 %compare.high.equal1429, i1 %compare.low1549, i1 %compare.high.order1430, !dbg !22
  %933 = select i1 %compare.result1550, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 61) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1552 = or i32 %compare.local.base1551, 114, !dbg !22
  %compare.low1553 = icmp ugt i32 %compare.low.base1431, %compare.local1552, !dbg !22
  %compare.result1554 = select i1 %compare.high.equal1429, i1 %compare.low1553, i1 %compare.high.order1430, !dbg !22
  %934 = select i1 %compare.result1554, float 0.000000e+00, float -inf, !dbg !23
  %compare.local.base1555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1432, i32 62) #9, !dbg !22, !frisk.register_lifetimes !5
  %compare.local1556 = or i32 %compare.local.base1555, 115, !dbg !22
  %compare.low1557 = icmp ugt i32 %compare.low.base1431, %compare.local1556, !dbg !22
  %compare.result1558 = select i1 %compare.high.equal1429, i1 %compare.low1557, i1 %compare.high.order1430, !dbg !22
  %935 = select i1 %compare.result1558, float 0.000000e+00, float -inf, !dbg !23
  %936 = insertelement <32 x float> poison, float %869, i64 0, !dbg !12
  %937 = insertelement <32 x float> %936, float %870, i64 1, !dbg !12
  %938 = insertelement <32 x float> %937, float %871, i64 2, !dbg !12
  %939 = insertelement <32 x float> %938, float %872, i64 3, !dbg !12
  %940 = insertelement <32 x float> %939, float %873, i64 4, !dbg !12
  %941 = insertelement <32 x float> %940, float %874, i64 5, !dbg !12
  %942 = insertelement <32 x float> %941, float %875, i64 6, !dbg !12
  %943 = insertelement <32 x float> %942, float %876, i64 7, !dbg !12
  %944 = insertelement <32 x float> %943, float %877, i64 8, !dbg !12
  %945 = insertelement <32 x float> %944, float %878, i64 9, !dbg !12
  %946 = insertelement <32 x float> %945, float %879, i64 10, !dbg !12
  %947 = insertelement <32 x float> %946, float %880, i64 11, !dbg !12
  %948 = insertelement <32 x float> %947, float %881, i64 12, !dbg !12
  %949 = insertelement <32 x float> %948, float %882, i64 13, !dbg !12
  %950 = insertelement <32 x float> %949, float %883, i64 14, !dbg !12
  %951 = insertelement <32 x float> %950, float %884, i64 15, !dbg !12
  %952 = insertelement <32 x float> %951, float %885, i64 16, !dbg !12
  %953 = insertelement <32 x float> %952, float %886, i64 17, !dbg !12
  %954 = insertelement <32 x float> %953, float %887, i64 18, !dbg !12
  %955 = insertelement <32 x float> %954, float %888, i64 19, !dbg !12
  %956 = insertelement <32 x float> %955, float %889, i64 20, !dbg !12
  %957 = insertelement <32 x float> %956, float %890, i64 21, !dbg !12
  %958 = insertelement <32 x float> %957, float %891, i64 22, !dbg !12
  %959 = insertelement <32 x float> %958, float %892, i64 23, !dbg !12
  %960 = insertelement <32 x float> %959, float %893, i64 24, !dbg !12
  %961 = insertelement <32 x float> %960, float %894, i64 25, !dbg !12
  %962 = insertelement <32 x float> %961, float %895, i64 26, !dbg !12
  %963 = insertelement <32 x float> %962, float %896, i64 27, !dbg !12
  %964 = insertelement <32 x float> %963, float %897, i64 28, !dbg !12
  %965 = insertelement <32 x float> %964, float %898, i64 29, !dbg !12
  %966 = insertelement <32 x float> %965, float %899, i64 30, !dbg !12
  %967 = insertelement <32 x float> %966, float %900, i64 31, !dbg !12
  %968 = insertvalue [2 x <32 x float>] poison, <32 x float> %967, 0, !dbg !12
  %969 = insertelement <32 x float> poison, float %904, i64 0, !dbg !12
  %970 = insertelement <32 x float> %969, float %905, i64 1, !dbg !12
  %971 = insertelement <32 x float> %970, float %906, i64 2, !dbg !12
  %972 = insertelement <32 x float> %971, float %907, i64 3, !dbg !12
  %973 = insertelement <32 x float> %972, float %908, i64 4, !dbg !12
  %974 = insertelement <32 x float> %973, float %909, i64 5, !dbg !12
  %975 = insertelement <32 x float> %974, float %910, i64 6, !dbg !12
  %976 = insertelement <32 x float> %975, float %911, i64 7, !dbg !12
  %977 = insertelement <32 x float> %976, float %912, i64 8, !dbg !12
  %978 = insertelement <32 x float> %977, float %913, i64 9, !dbg !12
  %979 = insertelement <32 x float> %978, float %914, i64 10, !dbg !12
  %980 = insertelement <32 x float> %979, float %915, i64 11, !dbg !12
  %981 = insertelement <32 x float> %980, float %916, i64 12, !dbg !12
  %982 = insertelement <32 x float> %981, float %917, i64 13, !dbg !12
  %983 = insertelement <32 x float> %982, float %918, i64 14, !dbg !12
  %984 = insertelement <32 x float> %983, float %919, i64 15, !dbg !12
  %985 = insertelement <32 x float> %984, float %920, i64 16, !dbg !12
  %986 = insertelement <32 x float> %985, float %921, i64 17, !dbg !12
  %987 = insertelement <32 x float> %986, float %922, i64 18, !dbg !12
  %988 = insertelement <32 x float> %987, float %923, i64 19, !dbg !12
  %989 = insertelement <32 x float> %988, float %924, i64 20, !dbg !12
  %990 = insertelement <32 x float> %989, float %925, i64 21, !dbg !12
  %991 = insertelement <32 x float> %990, float %926, i64 22, !dbg !12
  %992 = insertelement <32 x float> %991, float %927, i64 23, !dbg !12
  %993 = insertelement <32 x float> %992, float %928, i64 24, !dbg !12
  %994 = insertelement <32 x float> %993, float %929, i64 25, !dbg !12
  %995 = insertelement <32 x float> %994, float %930, i64 26, !dbg !12
  %996 = insertelement <32 x float> %995, float %931, i64 27, !dbg !12
  %997 = insertelement <32 x float> %996, float %932, i64 28, !dbg !12
  %998 = insertelement <32 x float> %997, float %933, i64 29, !dbg !12
  %999 = insertelement <32 x float> %998, float %934, i64 30, !dbg !12
  %1000 = insertelement <32 x float> %999, float %935, i64 31, !dbg !12
  %1001 = insertvalue [2 x <32 x float>] %968, <32 x float> %1000, 1, !dbg !12
  br label %1002, !dbg !12

1002:                                             ; preds = %861, %70
  %1003 = phi [2 x <32 x float>] [ %1001, %861 ], [ zeroinitializer, %70 ], !dbg !12
  %1004 = extractvalue [2 x <32 x float>] %1003, 0, !dbg !24
  %1005 = insertelement <4 x float> poison, float %794, i64 0, !dbg !24
  %1006 = insertelement <4 x float> %1005, float %795, i64 1, !dbg !24
  %1007 = insertelement <4 x float> %1006, float %796, i64 2, !dbg !24
  %1008 = insertelement <4 x float> %1007, float %797, i64 3, !dbg !24
  %1009 = shufflevector <32 x float> %1004, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1010 = fadd <4 x float> %1008, %1009, !dbg !24
  %1011 = extractelement <4 x float> %1010, i64 0, !dbg !24
  %1012 = extractelement <4 x float> %1010, i64 1, !dbg !24
  %1013 = extractelement <4 x float> %1010, i64 2, !dbg !24
  %1014 = extractelement <4 x float> %1010, i64 3, !dbg !24
  %1015 = insertelement <4 x float> poison, float %798, i64 0, !dbg !24
  %1016 = insertelement <4 x float> %1015, float %799, i64 1, !dbg !24
  %1017 = insertelement <4 x float> %1016, float %800, i64 2, !dbg !24
  %1018 = insertelement <4 x float> %1017, float %801, i64 3, !dbg !24
  %1019 = shufflevector <32 x float> %1004, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1020 = fadd <4 x float> %1018, %1019, !dbg !24
  %1021 = extractelement <4 x float> %1020, i64 0, !dbg !24
  %1022 = extractelement <4 x float> %1020, i64 1, !dbg !24
  %1023 = extractelement <4 x float> %1020, i64 2, !dbg !24
  %1024 = extractelement <4 x float> %1020, i64 3, !dbg !24
  %1025 = insertelement <4 x float> poison, float %802, i64 0, !dbg !24
  %1026 = insertelement <4 x float> %1025, float %803, i64 1, !dbg !24
  %1027 = insertelement <4 x float> %1026, float %804, i64 2, !dbg !24
  %1028 = insertelement <4 x float> %1027, float %805, i64 3, !dbg !24
  %1029 = shufflevector <32 x float> %1004, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !24
  %1030 = fadd <4 x float> %1028, %1029, !dbg !24
  %1031 = extractelement <4 x float> %1030, i64 0, !dbg !24
  %1032 = extractelement <4 x float> %1030, i64 1, !dbg !24
  %1033 = extractelement <4 x float> %1030, i64 2, !dbg !24
  %1034 = extractelement <4 x float> %1030, i64 3, !dbg !24
  %1035 = insertelement <4 x float> poison, float %806, i64 0, !dbg !24
  %1036 = insertelement <4 x float> %1035, float %807, i64 1, !dbg !24
  %1037 = insertelement <4 x float> %1036, float %808, i64 2, !dbg !24
  %1038 = insertelement <4 x float> %1037, float %809, i64 3, !dbg !24
  %1039 = shufflevector <32 x float> %1004, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !24
  %1040 = fadd <4 x float> %1038, %1039, !dbg !24
  %1041 = extractelement <4 x float> %1040, i64 0, !dbg !24
  %1042 = extractelement <4 x float> %1040, i64 1, !dbg !24
  %1043 = extractelement <4 x float> %1040, i64 2, !dbg !24
  %1044 = extractelement <4 x float> %1040, i64 3, !dbg !24
  %1045 = insertelement <4 x float> poison, float %810, i64 0, !dbg !24
  %1046 = insertelement <4 x float> %1045, float %811, i64 1, !dbg !24
  %1047 = insertelement <4 x float> %1046, float %812, i64 2, !dbg !24
  %1048 = insertelement <4 x float> %1047, float %813, i64 3, !dbg !24
  %1049 = shufflevector <32 x float> %1004, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !24
  %1050 = fadd <4 x float> %1048, %1049, !dbg !24
  %1051 = extractelement <4 x float> %1050, i64 0, !dbg !24
  %1052 = extractelement <4 x float> %1050, i64 1, !dbg !24
  %1053 = extractelement <4 x float> %1050, i64 2, !dbg !24
  %1054 = extractelement <4 x float> %1050, i64 3, !dbg !24
  %1055 = insertelement <4 x float> poison, float %814, i64 0, !dbg !24
  %1056 = insertelement <4 x float> %1055, float %815, i64 1, !dbg !24
  %1057 = insertelement <4 x float> %1056, float %816, i64 2, !dbg !24
  %1058 = insertelement <4 x float> %1057, float %817, i64 3, !dbg !24
  %1059 = shufflevector <32 x float> %1004, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !24
  %1060 = fadd <4 x float> %1058, %1059, !dbg !24
  %1061 = extractelement <4 x float> %1060, i64 0, !dbg !24
  %1062 = extractelement <4 x float> %1060, i64 1, !dbg !24
  %1063 = extractelement <4 x float> %1060, i64 2, !dbg !24
  %1064 = extractelement <4 x float> %1060, i64 3, !dbg !24
  %1065 = insertelement <4 x float> poison, float %818, i64 0, !dbg !24
  %1066 = insertelement <4 x float> %1065, float %819, i64 1, !dbg !24
  %1067 = insertelement <4 x float> %1066, float %820, i64 2, !dbg !24
  %1068 = insertelement <4 x float> %1067, float %821, i64 3, !dbg !24
  %1069 = shufflevector <32 x float> %1004, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !24
  %1070 = fadd <4 x float> %1068, %1069, !dbg !24
  %1071 = extractelement <4 x float> %1070, i64 0, !dbg !24
  %1072 = extractelement <4 x float> %1070, i64 1, !dbg !24
  %1073 = extractelement <4 x float> %1070, i64 2, !dbg !24
  %1074 = extractelement <4 x float> %1070, i64 3, !dbg !24
  %1075 = insertelement <4 x float> poison, float %822, i64 0, !dbg !24
  %1076 = insertelement <4 x float> %1075, float %823, i64 1, !dbg !24
  %1077 = insertelement <4 x float> %1076, float %824, i64 2, !dbg !24
  %1078 = insertelement <4 x float> %1077, float %825, i64 3, !dbg !24
  %1079 = shufflevector <32 x float> %1004, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !24
  %1080 = fadd <4 x float> %1078, %1079, !dbg !24
  %1081 = extractelement <4 x float> %1080, i64 0, !dbg !24
  %1082 = extractelement <4 x float> %1080, i64 1, !dbg !24
  %1083 = extractelement <4 x float> %1080, i64 2, !dbg !24
  %1084 = extractelement <4 x float> %1080, i64 3, !dbg !24
  %1085 = extractvalue [2 x <32 x float>] %1003, 1, !dbg !24
  %1086 = insertelement <4 x float> poison, float %826, i64 0, !dbg !24
  %1087 = insertelement <4 x float> %1086, float %827, i64 1, !dbg !24
  %1088 = insertelement <4 x float> %1087, float %828, i64 2, !dbg !24
  %1089 = insertelement <4 x float> %1088, float %829, i64 3, !dbg !24
  %1090 = shufflevector <32 x float> %1085, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1091 = fadd <4 x float> %1089, %1090, !dbg !24
  %1092 = extractelement <4 x float> %1091, i64 0, !dbg !24
  %1093 = extractelement <4 x float> %1091, i64 1, !dbg !24
  %1094 = extractelement <4 x float> %1091, i64 2, !dbg !24
  %1095 = extractelement <4 x float> %1091, i64 3, !dbg !24
  %1096 = insertelement <4 x float> poison, float %830, i64 0, !dbg !24
  %1097 = insertelement <4 x float> %1096, float %831, i64 1, !dbg !24
  %1098 = insertelement <4 x float> %1097, float %832, i64 2, !dbg !24
  %1099 = insertelement <4 x float> %1098, float %833, i64 3, !dbg !24
  %1100 = shufflevector <32 x float> %1085, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1101 = fadd <4 x float> %1099, %1100, !dbg !24
  %1102 = extractelement <4 x float> %1101, i64 0, !dbg !24
  %1103 = extractelement <4 x float> %1101, i64 1, !dbg !24
  %1104 = extractelement <4 x float> %1101, i64 2, !dbg !24
  %1105 = extractelement <4 x float> %1101, i64 3, !dbg !24
  %1106 = insertelement <4 x float> poison, float %834, i64 0, !dbg !24
  %1107 = insertelement <4 x float> %1106, float %835, i64 1, !dbg !24
  %1108 = insertelement <4 x float> %1107, float %836, i64 2, !dbg !24
  %1109 = insertelement <4 x float> %1108, float %837, i64 3, !dbg !24
  %1110 = shufflevector <32 x float> %1085, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !24
  %1111 = fadd <4 x float> %1109, %1110, !dbg !24
  %1112 = extractelement <4 x float> %1111, i64 0, !dbg !24
  %1113 = extractelement <4 x float> %1111, i64 1, !dbg !24
  %1114 = extractelement <4 x float> %1111, i64 2, !dbg !24
  %1115 = extractelement <4 x float> %1111, i64 3, !dbg !24
  %1116 = insertelement <4 x float> poison, float %838, i64 0, !dbg !24
  %1117 = insertelement <4 x float> %1116, float %839, i64 1, !dbg !24
  %1118 = insertelement <4 x float> %1117, float %840, i64 2, !dbg !24
  %1119 = insertelement <4 x float> %1118, float %841, i64 3, !dbg !24
  %1120 = shufflevector <32 x float> %1085, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !24
  %1121 = fadd <4 x float> %1119, %1120, !dbg !24
  %1122 = extractelement <4 x float> %1121, i64 0, !dbg !24
  %1123 = extractelement <4 x float> %1121, i64 1, !dbg !24
  %1124 = extractelement <4 x float> %1121, i64 2, !dbg !24
  %1125 = extractelement <4 x float> %1121, i64 3, !dbg !24
  %1126 = insertelement <4 x float> poison, float %842, i64 0, !dbg !24
  %1127 = insertelement <4 x float> %1126, float %843, i64 1, !dbg !24
  %1128 = insertelement <4 x float> %1127, float %844, i64 2, !dbg !24
  %1129 = insertelement <4 x float> %1128, float %845, i64 3, !dbg !24
  %1130 = shufflevector <32 x float> %1085, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !24
  %1131 = fadd <4 x float> %1129, %1130, !dbg !24
  %1132 = extractelement <4 x float> %1131, i64 0, !dbg !24
  %1133 = extractelement <4 x float> %1131, i64 1, !dbg !24
  %1134 = extractelement <4 x float> %1131, i64 2, !dbg !24
  %1135 = extractelement <4 x float> %1131, i64 3, !dbg !24
  %1136 = insertelement <4 x float> poison, float %846, i64 0, !dbg !24
  %1137 = insertelement <4 x float> %1136, float %847, i64 1, !dbg !24
  %1138 = insertelement <4 x float> %1137, float %848, i64 2, !dbg !24
  %1139 = insertelement <4 x float> %1138, float %849, i64 3, !dbg !24
  %1140 = shufflevector <32 x float> %1085, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !24
  %1141 = fadd <4 x float> %1139, %1140, !dbg !24
  %1142 = extractelement <4 x float> %1141, i64 0, !dbg !24
  %1143 = extractelement <4 x float> %1141, i64 1, !dbg !24
  %1144 = extractelement <4 x float> %1141, i64 2, !dbg !24
  %1145 = extractelement <4 x float> %1141, i64 3, !dbg !24
  %1146 = insertelement <4 x float> poison, float %850, i64 0, !dbg !24
  %1147 = insertelement <4 x float> %1146, float %851, i64 1, !dbg !24
  %1148 = insertelement <4 x float> %1147, float %852, i64 2, !dbg !24
  %1149 = insertelement <4 x float> %1148, float %853, i64 3, !dbg !24
  %1150 = shufflevector <32 x float> %1085, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !24
  %1151 = fadd <4 x float> %1149, %1150, !dbg !24
  %1152 = extractelement <4 x float> %1151, i64 0, !dbg !24
  %1153 = extractelement <4 x float> %1151, i64 1, !dbg !24
  %1154 = extractelement <4 x float> %1151, i64 2, !dbg !24
  %1155 = extractelement <4 x float> %1151, i64 3, !dbg !24
  %1156 = insertelement <4 x float> poison, float %854, i64 0, !dbg !24
  %1157 = insertelement <4 x float> %1156, float %855, i64 1, !dbg !24
  %1158 = insertelement <4 x float> %1157, float %856, i64 2, !dbg !24
  %1159 = insertelement <4 x float> %1158, float %857, i64 3, !dbg !24
  %1160 = shufflevector <32 x float> %1085, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !24
  %1161 = fadd <4 x float> %1159, %1160, !dbg !24
  %1162 = extractelement <4 x float> %1161, i64 0, !dbg !24
  %1163 = extractelement <4 x float> %1161, i64 1, !dbg !24
  %1164 = extractelement <4 x float> %1161, i64 2, !dbg !24
  %1165 = extractelement <4 x float> %1161, i64 3, !dbg !24
  %1166 = call float @llvm.maxnum.f32(float %1011, float -inf), !dbg !25, !frisk.local_reduction !5
  %local.reduce182 = call float @llvm.maxnum.f32(float %1166, float %1021), !dbg !25, !frisk.local_reduction !5
  %local.reduce183 = call float @llvm.maxnum.f32(float %1012, float %1022), !dbg !25, !frisk.local_reduction !5
  %local.reduce184 = call float @llvm.maxnum.f32(float %1013, float %1023), !dbg !25, !frisk.local_reduction !5
  %local.reduce185 = call float @llvm.maxnum.f32(float %1014, float %1024), !dbg !25, !frisk.local_reduction !5
  %local.reduce186 = call float @llvm.maxnum.f32(float %local.reduce182, float %1031), !dbg !25, !frisk.local_reduction !5
  %local.reduce187 = call float @llvm.maxnum.f32(float %local.reduce183, float %1032), !dbg !25, !frisk.local_reduction !5
  %local.reduce188 = call float @llvm.maxnum.f32(float %local.reduce184, float %1033), !dbg !25, !frisk.local_reduction !5
  %local.reduce189 = call float @llvm.maxnum.f32(float %local.reduce185, float %1034), !dbg !25, !frisk.local_reduction !5
  %local.reduce190 = call float @llvm.maxnum.f32(float %local.reduce186, float %1041), !dbg !25, !frisk.local_reduction !5
  %local.reduce191 = call float @llvm.maxnum.f32(float %local.reduce187, float %1042), !dbg !25, !frisk.local_reduction !5
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %1043), !dbg !25, !frisk.local_reduction !5
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %1044), !dbg !25, !frisk.local_reduction !5
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %1051), !dbg !25, !frisk.local_reduction !5
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %1052), !dbg !25, !frisk.local_reduction !5
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %1053), !dbg !25, !frisk.local_reduction !5
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %1054), !dbg !25, !frisk.local_reduction !5
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %1061), !dbg !25, !frisk.local_reduction !5
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %1062), !dbg !25, !frisk.local_reduction !5
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %1063), !dbg !25, !frisk.local_reduction !5
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %1064), !dbg !25, !frisk.local_reduction !5
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %1071), !dbg !25, !frisk.local_reduction !5
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %1072), !dbg !25, !frisk.local_reduction !5
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %1073), !dbg !25, !frisk.local_reduction !5
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %1074), !dbg !25, !frisk.local_reduction !5
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %1081), !dbg !25, !frisk.local_reduction !5
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %1082), !dbg !25, !frisk.local_reduction !5
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %1083), !dbg !25, !frisk.local_reduction !5
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %1084), !dbg !25, !frisk.local_reduction !5
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %local.reduce207), !dbg !25, !frisk.local_reduction !5
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce208, float %local.reduce209), !dbg !25, !frisk.local_reduction !5
  %1167 = call float @llvm.maxnum.f32(float %local.reduce210, float %local.reduce211), !dbg !25, !frisk.local_reduction !5
  %1168 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !25
  %1169 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1168), !dbg !25
  %1170 = add nuw nsw i32 %1169, 64, !dbg !25
  %1171 = xor i32 %1169, 16, !dbg !25
  %1172 = and i32 %1170, 192, !dbg !25
  %1173 = icmp samesign ult i32 %1171, %1172, !dbg !25
  %1174 = select i1 %1173, i32 %1171, i32 %1169, !dbg !25
  %1175 = shl nuw nsw i32 %1174, 2, !dbg !25
  %1176 = bitcast float %1167 to i32, !dbg !25
  %1177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1176), !dbg !25
  %1178 = bitcast i32 %1177 to float, !dbg !25
  %1179 = call float @llvm.maxnum.f32(float %1167, float %1178), !dbg !25
  %1180 = xor i32 %1169, 32, !dbg !25
  %1181 = icmp samesign ult i32 %1180, %1172, !dbg !25
  %1182 = select i1 %1181, i32 %1180, i32 %1169, !dbg !25
  %1183 = shl nuw nsw i32 %1182, 2, !dbg !25
  %1184 = bitcast float %1179 to i32, !dbg !25
  %1185 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1184), !dbg !25
  %1186 = bitcast i32 %1185 to float, !dbg !25
  %1187 = call float @llvm.maxnum.f32(float %1179, float %1186), !dbg !25
  %1188 = extractvalue [2 x <1 x float>] %65, 0, !dbg !25
  %1189 = insertelement <1 x float> poison, float %1187, i64 0, !dbg !25
  %1190 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1189, <1 x float> %1188), !dbg !25
  %1191 = call float @llvm.maxnum.f32(float %1092, float -inf), !dbg !25, !frisk.local_reduction !5
  %local.reduce151 = call float @llvm.maxnum.f32(float %1191, float %1102), !dbg !25, !frisk.local_reduction !5
  %local.reduce152 = call float @llvm.maxnum.f32(float %1093, float %1103), !dbg !25, !frisk.local_reduction !5
  %local.reduce153 = call float @llvm.maxnum.f32(float %1094, float %1104), !dbg !25, !frisk.local_reduction !5
  %local.reduce154 = call float @llvm.maxnum.f32(float %1095, float %1105), !dbg !25, !frisk.local_reduction !5
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %1112), !dbg !25, !frisk.local_reduction !5
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %1113), !dbg !25, !frisk.local_reduction !5
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %1114), !dbg !25, !frisk.local_reduction !5
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %1115), !dbg !25, !frisk.local_reduction !5
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %1122), !dbg !25, !frisk.local_reduction !5
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %1123), !dbg !25, !frisk.local_reduction !5
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %1124), !dbg !25, !frisk.local_reduction !5
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %1125), !dbg !25, !frisk.local_reduction !5
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %1132), !dbg !25, !frisk.local_reduction !5
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %1133), !dbg !25, !frisk.local_reduction !5
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %1134), !dbg !25, !frisk.local_reduction !5
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %1135), !dbg !25, !frisk.local_reduction !5
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %1142), !dbg !25, !frisk.local_reduction !5
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %1143), !dbg !25, !frisk.local_reduction !5
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %1144), !dbg !25, !frisk.local_reduction !5
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %1145), !dbg !25, !frisk.local_reduction !5
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %1152), !dbg !25, !frisk.local_reduction !5
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %1153), !dbg !25, !frisk.local_reduction !5
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %1154), !dbg !25, !frisk.local_reduction !5
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %1155), !dbg !25, !frisk.local_reduction !5
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %1162), !dbg !25, !frisk.local_reduction !5
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %1163), !dbg !25, !frisk.local_reduction !5
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %1164), !dbg !25, !frisk.local_reduction !5
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %1165), !dbg !25, !frisk.local_reduction !5
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !25, !frisk.local_reduction !5
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !25, !frisk.local_reduction !5
  %1192 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !25, !frisk.local_reduction !5
  %1193 = bitcast float %1192 to i32, !dbg !25
  %1194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1193), !dbg !25
  %1195 = bitcast i32 %1194 to float, !dbg !25
  %1196 = call float @llvm.maxnum.f32(float %1192, float %1195), !dbg !25
  %1197 = bitcast float %1196 to i32, !dbg !25
  %1198 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1197), !dbg !25
  %1199 = bitcast i32 %1198 to float, !dbg !25
  %1200 = call float @llvm.maxnum.f32(float %1196, float %1199), !dbg !25
  %1201 = extractvalue [2 x <1 x float>] %65, 1, !dbg !25
  %1202 = insertelement <1 x float> poison, float %1200, i64 0, !dbg !25
  %1203 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1202, <1 x float> %1201), !dbg !25
  %1204 = shufflevector <1 x float> %1190, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %1205 = fsub <4 x float> %1010, %1204, !dbg !26
  %1206 = fsub <4 x float> %1020, %1204, !dbg !26
  %1207 = fsub <4 x float> %1030, %1204, !dbg !26
  %1208 = fsub <4 x float> %1040, %1204, !dbg !26
  %1209 = fsub <4 x float> %1050, %1204, !dbg !26
  %1210 = fsub <4 x float> %1060, %1204, !dbg !26
  %1211 = fsub <4 x float> %1070, %1204, !dbg !26
  %1212 = fsub <4 x float> %1080, %1204, !dbg !26
  %1213 = shufflevector <1 x float> %1203, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %1214 = fsub <4 x float> %1091, %1213, !dbg !26
  %1215 = fsub <4 x float> %1101, %1213, !dbg !26
  %1216 = fsub <4 x float> %1111, %1213, !dbg !26
  %1217 = fsub <4 x float> %1121, %1213, !dbg !26
  %1218 = fsub <4 x float> %1131, %1213, !dbg !26
  %1219 = fsub <4 x float> %1141, %1213, !dbg !26
  %1220 = fsub <4 x float> %1151, %1213, !dbg !26
  %1221 = fsub <4 x float> %1161, %1213, !dbg !26
  br i1 %43, label %1222, label %1227, !dbg !13

1222:                                             ; preds = %1002
  %1223 = fsub <1 x float> %1201, %1203, !dbg !27
  %1224 = extractelement <1 x float> %1223, i64 0, !dbg !27
  %1225 = fsub <1 x float> %1188, %1190, !dbg !27
  %1226 = extractelement <1 x float> %1225, i64 0, !dbg !27
  %address.thread766 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !13, !range !11, !frisk.register_lifetimes !5
  %address.local767 = lshr i32 %address.thread766, 2, !dbg !17
  %address.local768 = and i32 %address.local767, 48, !dbg !17
  %address.local769 = and i32 %address.thread766, 15, !dbg !17
  %address.local770 = or disjoint i32 %address.local768, %address.local769, !dbg !17
  %address.local771 = zext nneg i32 %address.local770 to i64, !dbg !17
  %address.local772 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local771, !dbg !13
  store float %1226, ptr addrspace(3) %address.local772, align 4, !dbg !13
  %address.local773 = lshr i32 %address.thread766, 2, !dbg !17
  %address.local774 = and i32 %address.local773, 48, !dbg !17
  %address.local775 = and i32 %address.thread766, 15, !dbg !17
  %address.local776 = or disjoint i32 %address.local774, %address.local775, !dbg !17
  %address.local777 = or disjoint i32 %address.local776, 64, !dbg !17
  %address.local778 = zext nneg i32 %address.local777 to i64, !dbg !17
  %address.local779 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local778, !dbg !13
  store float %1224, ptr addrspace(3) %address.local779, align 4, !dbg !13
  br label %1227, !dbg !13

1227:                                             ; preds = %1222, %1002
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.thread780 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !13, !range !11, !frisk.register_lifetimes !5
  %address.local781 = lshr i32 %address.thread780, 1, !dbg !13
  %address.local782 = and i32 %address.local781, 96, !dbg !13
  %address.local783 = and i32 %address.thread780, 15, !dbg !13
  %address.local784 = or disjoint i32 %address.local782, %address.local783, !dbg !13
  %address.local785 = zext nneg i32 %address.local784 to i64, !dbg !13
  %address.local786 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local785, !dbg !13
  %1228 = load float, ptr addrspace(3) %address.local786, align 4, !dbg !13
  %lds.const.gep1745 = getelementptr i8, ptr addrspace(3) %address.local786, i64 64, !dbg !13
  %1229 = load float, ptr addrspace(3) %lds.const.gep1745, align 4, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %1230 = call float @llvm.amdgcn.exp2.f32(float %1228), !dbg !13
  %1231 = call float @llvm.amdgcn.exp2.f32(float %1229), !dbg !13
  br i1 %43, label %1232, label %1233, !dbg !28

1232:                                             ; preds = %1227
  %address.thread789 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !28, !range !11, !frisk.register_lifetimes !5
  %address.local790 = lshr i32 %address.thread789, 1, !dbg !13
  %address.local791 = and i32 %address.local790, 96, !dbg !13
  %address.local792 = and i32 %address.thread789, 15, !dbg !13
  %address.local793 = or disjoint i32 %address.local791, %address.local792, !dbg !13
  %address.local794 = zext nneg i32 %address.local793 to i64, !dbg !13
  %address.local795 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !13
  store float %1230, ptr addrspace(3) %address.local795, align 4, !dbg !28
  %address.local796 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local794, !dbg !13
  %address.local797 = getelementptr i8, ptr addrspace(3) %address.local796, i64 64, !dbg !13
  store float %1231, ptr addrspace(3) %address.local797, align 4, !dbg !28
  br label %1233, !dbg !28

1233:                                             ; preds = %1232, %1227
  fence syncscope("workgroup") release, !dbg !28
  call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %address.thread798 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !28, !range !11, !frisk.register_lifetimes !5
  %address.local799 = lshr i32 %address.thread798, 2, !dbg !17
  %address.local800 = and i32 %address.local799, 48, !dbg !17
  %address.local801 = and i32 %address.thread798, 15, !dbg !17
  %address.local802 = or disjoint i32 %address.local800, %address.local801, !dbg !17
  %address.local803 = zext nneg i32 %address.local802 to i64, !dbg !17
  %address.local804 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local803, !dbg !28
  %1234 = load float, ptr addrspace(3) %address.local804, align 4, !dbg !28
  %lds.const.gep1746 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !28
  %1235 = load float, ptr addrspace(3) %lds.const.gep1746, align 4, !dbg !28
  fence syncscope("workgroup") release, !dbg !28
  call void @llvm.amdgcn.s.barrier(), !dbg !28
  fence syncscope("workgroup") acquire, !dbg !28
  %1236 = add nuw nsw i64 %102, %15, !dbg !29
  %1237 = shl i64 %1236, 19, !dbg !29
  %1238 = and i64 %1237, 16252928, !dbg !29
  %1239 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1238, !dbg !29
  %1240 = getelementptr [2 x i8], ptr addrspace(1) %1239, i64 %108, !dbg !29
  %address.local812 = zext nneg i32 %address.thread798 to i64, !dbg !10
  %address.local813 = shl nuw nsw i64 %address.local812, 3, !dbg !9
  %address.local814 = and i64 %address.local813, 1536, !dbg !9
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local814, !dbg !9
  %address.local816 = ptrtoint ptr addrspace(3) %address.local815 to i64, !dbg !9
  %address.local817 = trunc i64 %address.local816 to i32, !dbg !9
  %1241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local817), !dbg !29
  %1242 = zext i32 %1241 to i64, !dbg !29
  %1243 = inttoptr i64 %1242 to ptr addrspace(3), !dbg !29
  %address.local818 = shl nuw nsw i64 %address.local812, 5, !dbg !29
  %address.local819 = and i64 %address.local818, 8064, !dbg !29
  %address.local820 = shl nuw nsw i64 %address.local812, 3, !dbg !29
  %address.local821 = and i64 %address.local820, 24, !dbg !29
  %address.local822 = or disjoint i64 %address.local819, %address.local821, !dbg !29
  %address.local823 = getelementptr [2 x i8], ptr addrspace(1) %1240, i64 %address.local822, !dbg !29
  %1244 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !29
  %1245 = ptrtoint ptr addrspace(1) %address.local823 to i64, !dbg !29
  %1246 = sub i64 %1245, %1244, !dbg !29
  %dma.buffer.offset1773 = trunc i64 %1246 to i32, !dbg !29
  %1247 = trunc i64 %1244 to i32, !dbg !29
  %1248 = insertelement <4 x i32> poison, i32 %1247, i64 0, !dbg !29
  %1249 = lshr i64 %1244, 32, !dbg !29
  %1250 = trunc i64 %1249 to i32, !dbg !29
  %1251 = insertelement <4 x i32> %1248, i32 %1250, i64 1, !dbg !29
  %1252 = insertelement <4 x i32> %1251, i32 33554432, i64 2, !dbg !29
  %dma.buffer.rsrc1774 = insertelement <4 x i32> %1252, i32 131072, i64 3, !dbg !29
  %1253 = ptrtoint ptr addrspace(3) %1243 to i32, !dbg !29
  %dma.buffer.lds1775 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1253), !dbg !29
  %1254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1775, i32 %dma.buffer.offset1773, <4 x i32> %dma.buffer.rsrc1774) #10, !dbg !29
  %address.local824 = or disjoint i64 %address.local812, 256, !dbg !9
  %address.local825 = shl nuw nsw i64 %address.local824, 3, !dbg !9
  %address.local826 = and i64 %address.local825, 3584, !dbg !9
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local826, !dbg !9
  %address.local828 = ptrtoint ptr addrspace(3) %address.local827 to i64, !dbg !9
  %address.local829 = trunc i64 %address.local828 to i32, !dbg !9
  %1255 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local829), !dbg !29
  %1256 = zext i32 %1255 to i64, !dbg !29
  %1257 = inttoptr i64 %1256 to ptr addrspace(3), !dbg !29
  %address.local830 = shl nuw nsw i64 %address.local824, 5, !dbg !29
  %address.local831 = and i64 %address.local830, 16256, !dbg !29
  %address.local832 = shl nuw nsw i64 %address.local812, 3, !dbg !29
  %address.local833 = and i64 %address.local832, 24, !dbg !29
  %address.local834 = or disjoint i64 %address.local831, %address.local833, !dbg !29
  %address.local835 = getelementptr [2 x i8], ptr addrspace(1) %1240, i64 %address.local834, !dbg !29
  %1258 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !29
  %1259 = ptrtoint ptr addrspace(1) %address.local835 to i64, !dbg !29
  %1260 = sub i64 %1259, %1258, !dbg !29
  %dma.buffer.offset1770 = trunc i64 %1260 to i32, !dbg !29
  %1261 = trunc i64 %1258 to i32, !dbg !29
  %1262 = insertelement <4 x i32> poison, i32 %1261, i64 0, !dbg !29
  %1263 = lshr i64 %1258, 32, !dbg !29
  %1264 = trunc i64 %1263 to i32, !dbg !29
  %1265 = insertelement <4 x i32> %1262, i32 %1264, i64 1, !dbg !29
  %1266 = insertelement <4 x i32> %1265, i32 33554432, i64 2, !dbg !29
  %dma.buffer.rsrc1771 = insertelement <4 x i32> %1266, i32 131072, i64 3, !dbg !29
  %1267 = ptrtoint ptr addrspace(3) %1257 to i32, !dbg !29
  %dma.buffer.lds1772 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1267), !dbg !29
  %1268 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1772, i32 %dma.buffer.offset1770, <4 x i32> %dma.buffer.rsrc1771) #10, !dbg !29
  %1269 = extractvalue [2 x <1 x float>] %64, 0, !dbg !28
  %1270 = extractelement <1 x float> %1269, i64 0, !dbg !28
  %1271 = fmul float %1270, %1234, !dbg !28
  %1272 = extractvalue [2 x <1 x float>] %64, 1, !dbg !28
  %1273 = extractelement <1 x float> %1272, i64 0, !dbg !28
  %1274 = fmul float %1273, %1235, !dbg !28
  %1275 = extractelement <4 x float> %1205, i64 0, !dbg !30
  %1276 = call float @llvm.amdgcn.exp2.f32(float %1275), !dbg !30
  %1277 = insertelement <4 x float> poison, float %1276, i64 0, !dbg !31
  %1278 = extractelement <4 x float> %1205, i64 1, !dbg !30
  %1279 = call float @llvm.amdgcn.exp2.f32(float %1278), !dbg !30
  %1280 = insertelement <4 x float> %1277, float %1279, i64 1, !dbg !31
  %1281 = extractelement <4 x float> %1205, i64 2, !dbg !30
  %1282 = call float @llvm.amdgcn.exp2.f32(float %1281), !dbg !30
  %1283 = insertelement <4 x float> %1280, float %1282, i64 2, !dbg !31
  %1284 = extractelement <4 x float> %1205, i64 3, !dbg !30
  %1285 = call float @llvm.amdgcn.exp2.f32(float %1284), !dbg !30
  %1286 = insertelement <4 x float> %1283, float %1285, i64 3, !dbg !31
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1286) #9, !dbg !31, !frisk.register_lifetimes !5
  %1287 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !31
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1287) #9, !dbg !30, !frisk.register_lifetimes !5
  %1288 = extractelement <4 x float> %1206, i64 0, !dbg !30
  %1289 = call float @llvm.amdgcn.exp2.f32(float %1288), !dbg !30
  %1290 = insertelement <4 x float> poison, float %1289, i64 0, !dbg !31
  %local.reduce120 = fadd float %1276, %1289, !dbg !32, !frisk.local_reduction !5
  %1291 = extractelement <4 x float> %1206, i64 1, !dbg !30
  %1292 = call float @llvm.amdgcn.exp2.f32(float %1291), !dbg !30
  %1293 = insertelement <4 x float> %1290, float %1292, i64 1, !dbg !31
  %local.reduce121 = fadd float %1279, %1292, !dbg !32, !frisk.local_reduction !5
  %1294 = extractelement <4 x float> %1206, i64 2, !dbg !30
  %1295 = call float @llvm.amdgcn.exp2.f32(float %1294), !dbg !30
  %1296 = insertelement <4 x float> %1293, float %1295, i64 2, !dbg !31
  %local.reduce122 = fadd float %1282, %1295, !dbg !32, !frisk.local_reduction !5
  %1297 = extractelement <4 x float> %1206, i64 3, !dbg !30
  %1298 = call float @llvm.amdgcn.exp2.f32(float %1297), !dbg !30
  %1299 = insertelement <4 x float> %1296, float %1298, i64 3, !dbg !31
  %math.result228 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1299) #9, !dbg !31, !frisk.register_lifetimes !5
  %1300 = fptrunc <4 x float> %math.result228 to <4 x half>, !dbg !31
  %math.packed229 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1300) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce123 = fadd float %1285, %1298, !dbg !32, !frisk.local_reduction !5
  %1301 = extractelement <4 x float> %1207, i64 0, !dbg !30
  %1302 = call float @llvm.amdgcn.exp2.f32(float %1301), !dbg !30
  %1303 = insertelement <4 x float> poison, float %1302, i64 0, !dbg !31
  %local.reduce124 = fadd float %local.reduce120, %1302, !dbg !32, !frisk.local_reduction !5
  %1304 = extractelement <4 x float> %1207, i64 1, !dbg !30
  %1305 = call float @llvm.amdgcn.exp2.f32(float %1304), !dbg !30
  %1306 = insertelement <4 x float> %1303, float %1305, i64 1, !dbg !31
  %local.reduce125 = fadd float %local.reduce121, %1305, !dbg !32, !frisk.local_reduction !5
  %1307 = extractelement <4 x float> %1207, i64 2, !dbg !30
  %1308 = call float @llvm.amdgcn.exp2.f32(float %1307), !dbg !30
  %1309 = insertelement <4 x float> %1306, float %1308, i64 2, !dbg !31
  %local.reduce126 = fadd float %local.reduce122, %1308, !dbg !32, !frisk.local_reduction !5
  %1310 = extractelement <4 x float> %1207, i64 3, !dbg !30
  %1311 = call float @llvm.amdgcn.exp2.f32(float %1310), !dbg !30
  %1312 = insertelement <4 x float> %1309, float %1311, i64 3, !dbg !31
  %math.result230 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1312) #9, !dbg !31, !frisk.register_lifetimes !5
  %1313 = fptrunc <4 x float> %math.result230 to <4 x half>, !dbg !31
  %math.packed231 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1313) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce127 = fadd float %local.reduce123, %1311, !dbg !32, !frisk.local_reduction !5
  %1314 = extractelement <4 x float> %1208, i64 0, !dbg !30
  %1315 = call float @llvm.amdgcn.exp2.f32(float %1314), !dbg !30
  %1316 = insertelement <4 x float> poison, float %1315, i64 0, !dbg !31
  %local.reduce128 = fadd float %local.reduce124, %1315, !dbg !32, !frisk.local_reduction !5
  %1317 = extractelement <4 x float> %1208, i64 1, !dbg !30
  %1318 = call float @llvm.amdgcn.exp2.f32(float %1317), !dbg !30
  %1319 = insertelement <4 x float> %1316, float %1318, i64 1, !dbg !31
  %local.reduce129 = fadd float %local.reduce125, %1318, !dbg !32, !frisk.local_reduction !5
  %1320 = extractelement <4 x float> %1208, i64 2, !dbg !30
  %1321 = call float @llvm.amdgcn.exp2.f32(float %1320), !dbg !30
  %1322 = insertelement <4 x float> %1319, float %1321, i64 2, !dbg !31
  %local.reduce130 = fadd float %local.reduce126, %1321, !dbg !32, !frisk.local_reduction !5
  %1323 = extractelement <4 x float> %1208, i64 3, !dbg !30
  %1324 = call float @llvm.amdgcn.exp2.f32(float %1323), !dbg !30
  %1325 = insertelement <4 x float> %1322, float %1324, i64 3, !dbg !31
  %math.result232 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1325) #9, !dbg !31, !frisk.register_lifetimes !5
  %1326 = fptrunc <4 x float> %math.result232 to <4 x half>, !dbg !31
  %math.packed233 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1326) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce131 = fadd float %local.reduce127, %1324, !dbg !32, !frisk.local_reduction !5
  %1327 = extractelement <4 x float> %1209, i64 0, !dbg !30
  %1328 = call float @llvm.amdgcn.exp2.f32(float %1327), !dbg !30
  %1329 = insertelement <4 x float> poison, float %1328, i64 0, !dbg !31
  %local.reduce132 = fadd float %local.reduce128, %1328, !dbg !32, !frisk.local_reduction !5
  %1330 = extractelement <4 x float> %1209, i64 1, !dbg !30
  %1331 = call float @llvm.amdgcn.exp2.f32(float %1330), !dbg !30
  %1332 = insertelement <4 x float> %1329, float %1331, i64 1, !dbg !31
  %local.reduce133 = fadd float %local.reduce129, %1331, !dbg !32, !frisk.local_reduction !5
  %1333 = extractelement <4 x float> %1209, i64 2, !dbg !30
  %1334 = call float @llvm.amdgcn.exp2.f32(float %1333), !dbg !30
  %1335 = insertelement <4 x float> %1332, float %1334, i64 2, !dbg !31
  %local.reduce134 = fadd float %local.reduce130, %1334, !dbg !32, !frisk.local_reduction !5
  %1336 = extractelement <4 x float> %1209, i64 3, !dbg !30
  %1337 = call float @llvm.amdgcn.exp2.f32(float %1336), !dbg !30
  %1338 = insertelement <4 x float> %1335, float %1337, i64 3, !dbg !31
  %math.result234 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1338) #9, !dbg !31, !frisk.register_lifetimes !5
  %1339 = fptrunc <4 x float> %math.result234 to <4 x half>, !dbg !31
  %math.packed235 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1339) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce135 = fadd float %local.reduce131, %1337, !dbg !32, !frisk.local_reduction !5
  %1340 = extractelement <4 x float> %1210, i64 0, !dbg !30
  %1341 = call float @llvm.amdgcn.exp2.f32(float %1340), !dbg !30
  %1342 = insertelement <4 x float> poison, float %1341, i64 0, !dbg !31
  %local.reduce136 = fadd float %local.reduce132, %1341, !dbg !32, !frisk.local_reduction !5
  %1343 = extractelement <4 x float> %1210, i64 1, !dbg !30
  %1344 = call float @llvm.amdgcn.exp2.f32(float %1343), !dbg !30
  %1345 = insertelement <4 x float> %1342, float %1344, i64 1, !dbg !31
  %local.reduce137 = fadd float %local.reduce133, %1344, !dbg !32, !frisk.local_reduction !5
  %1346 = extractelement <4 x float> %1210, i64 2, !dbg !30
  %1347 = call float @llvm.amdgcn.exp2.f32(float %1346), !dbg !30
  %1348 = insertelement <4 x float> %1345, float %1347, i64 2, !dbg !31
  %local.reduce138 = fadd float %local.reduce134, %1347, !dbg !32, !frisk.local_reduction !5
  %1349 = extractelement <4 x float> %1210, i64 3, !dbg !30
  %1350 = call float @llvm.amdgcn.exp2.f32(float %1349), !dbg !30
  %1351 = insertelement <4 x float> %1348, float %1350, i64 3, !dbg !31
  %math.result236 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1351) #9, !dbg !31, !frisk.register_lifetimes !5
  %1352 = fptrunc <4 x float> %math.result236 to <4 x half>, !dbg !31
  %math.packed237 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1352) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce139 = fadd float %local.reduce135, %1350, !dbg !32, !frisk.local_reduction !5
  %1353 = extractelement <4 x float> %1211, i64 0, !dbg !30
  %1354 = call float @llvm.amdgcn.exp2.f32(float %1353), !dbg !30
  %1355 = insertelement <4 x float> poison, float %1354, i64 0, !dbg !31
  %local.reduce140 = fadd float %local.reduce136, %1354, !dbg !32, !frisk.local_reduction !5
  %1356 = extractelement <4 x float> %1211, i64 1, !dbg !30
  %1357 = call float @llvm.amdgcn.exp2.f32(float %1356), !dbg !30
  %1358 = insertelement <4 x float> %1355, float %1357, i64 1, !dbg !31
  %local.reduce141 = fadd float %local.reduce137, %1357, !dbg !32, !frisk.local_reduction !5
  %1359 = extractelement <4 x float> %1211, i64 2, !dbg !30
  %1360 = call float @llvm.amdgcn.exp2.f32(float %1359), !dbg !30
  %1361 = insertelement <4 x float> %1358, float %1360, i64 2, !dbg !31
  %local.reduce142 = fadd float %local.reduce138, %1360, !dbg !32, !frisk.local_reduction !5
  %1362 = extractelement <4 x float> %1211, i64 3, !dbg !30
  %1363 = call float @llvm.amdgcn.exp2.f32(float %1362), !dbg !30
  %1364 = insertelement <4 x float> %1361, float %1363, i64 3, !dbg !31
  %math.result238 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1364) #9, !dbg !31, !frisk.register_lifetimes !5
  %1365 = fptrunc <4 x float> %math.result238 to <4 x half>, !dbg !31
  %math.packed239 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1365) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce143 = fadd float %local.reduce139, %1363, !dbg !32, !frisk.local_reduction !5
  %1366 = extractelement <4 x float> %1212, i64 0, !dbg !30
  %1367 = call float @llvm.amdgcn.exp2.f32(float %1366), !dbg !30
  %1368 = insertelement <4 x float> poison, float %1367, i64 0, !dbg !31
  %local.reduce144 = fadd float %local.reduce140, %1367, !dbg !32, !frisk.local_reduction !5
  %1369 = extractelement <4 x float> %1212, i64 1, !dbg !30
  %1370 = call float @llvm.amdgcn.exp2.f32(float %1369), !dbg !30
  %1371 = insertelement <4 x float> %1368, float %1370, i64 1, !dbg !31
  %local.reduce145 = fadd float %local.reduce141, %1370, !dbg !32, !frisk.local_reduction !5
  %local.reduce148 = fadd float %local.reduce144, %local.reduce145, !dbg !32, !frisk.local_reduction !5
  %1372 = extractelement <4 x float> %1212, i64 2, !dbg !30
  %1373 = call float @llvm.amdgcn.exp2.f32(float %1372), !dbg !30
  %1374 = insertelement <4 x float> %1371, float %1373, i64 2, !dbg !31
  %local.reduce146 = fadd float %local.reduce142, %1373, !dbg !32, !frisk.local_reduction !5
  %1375 = extractelement <4 x float> %1212, i64 3, !dbg !30
  %1376 = call float @llvm.amdgcn.exp2.f32(float %1375), !dbg !30
  %1377 = insertelement <4 x float> %1374, float %1376, i64 3, !dbg !31
  %math.result240 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1377) #9, !dbg !31, !frisk.register_lifetimes !5
  %1378 = fptrunc <4 x float> %math.result240 to <4 x half>, !dbg !31
  %math.packed241 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1378) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce147 = fadd float %local.reduce143, %1376, !dbg !32, !frisk.local_reduction !5
  %local.reduce149 = fadd float %local.reduce146, %local.reduce147, !dbg !32, !frisk.local_reduction !5
  %1379 = fadd float %local.reduce148, %local.reduce149, !dbg !32, !frisk.local_reduction !5
  %1380 = extractelement <4 x float> %1214, i64 0, !dbg !30
  %1381 = call float @llvm.amdgcn.exp2.f32(float %1380), !dbg !30
  %1382 = insertelement <4 x float> poison, float %1381, i64 0, !dbg !31
  %1383 = extractelement <4 x float> %1214, i64 1, !dbg !30
  %1384 = call float @llvm.amdgcn.exp2.f32(float %1383), !dbg !30
  %1385 = insertelement <4 x float> %1382, float %1384, i64 1, !dbg !31
  %1386 = extractelement <4 x float> %1214, i64 2, !dbg !30
  %1387 = call float @llvm.amdgcn.exp2.f32(float %1386), !dbg !30
  %1388 = insertelement <4 x float> %1385, float %1387, i64 2, !dbg !31
  %1389 = extractelement <4 x float> %1214, i64 3, !dbg !30
  %1390 = call float @llvm.amdgcn.exp2.f32(float %1389), !dbg !30
  %1391 = insertelement <4 x float> %1388, float %1390, i64 3, !dbg !31
  %math.result242 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1391) #9, !dbg !31, !frisk.register_lifetimes !5
  %1392 = fptrunc <4 x float> %math.result242 to <4 x half>, !dbg !31
  %math.packed243 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1392) #9, !dbg !30, !frisk.register_lifetimes !5
  %1393 = extractelement <4 x float> %1215, i64 0, !dbg !30
  %1394 = call float @llvm.amdgcn.exp2.f32(float %1393), !dbg !30
  %1395 = insertelement <4 x float> poison, float %1394, i64 0, !dbg !31
  %local.reduce = fadd float %1381, %1394, !dbg !32, !frisk.local_reduction !5
  %1396 = extractelement <4 x float> %1215, i64 1, !dbg !30
  %1397 = call float @llvm.amdgcn.exp2.f32(float %1396), !dbg !30
  %1398 = insertelement <4 x float> %1395, float %1397, i64 1, !dbg !31
  %local.reduce90 = fadd float %1384, %1397, !dbg !32, !frisk.local_reduction !5
  %1399 = extractelement <4 x float> %1215, i64 2, !dbg !30
  %1400 = call float @llvm.amdgcn.exp2.f32(float %1399), !dbg !30
  %1401 = insertelement <4 x float> %1398, float %1400, i64 2, !dbg !31
  %local.reduce91 = fadd float %1387, %1400, !dbg !32, !frisk.local_reduction !5
  %1402 = extractelement <4 x float> %1215, i64 3, !dbg !30
  %1403 = call float @llvm.amdgcn.exp2.f32(float %1402), !dbg !30
  %1404 = insertelement <4 x float> %1401, float %1403, i64 3, !dbg !31
  %math.result244 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1404) #9, !dbg !31, !frisk.register_lifetimes !5
  %1405 = fptrunc <4 x float> %math.result244 to <4 x half>, !dbg !31
  %math.packed245 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1405) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce92 = fadd float %1390, %1403, !dbg !32, !frisk.local_reduction !5
  %1406 = extractelement <4 x float> %1216, i64 0, !dbg !30
  %1407 = call float @llvm.amdgcn.exp2.f32(float %1406), !dbg !30
  %1408 = insertelement <4 x float> poison, float %1407, i64 0, !dbg !31
  %local.reduce93 = fadd float %local.reduce, %1407, !dbg !32, !frisk.local_reduction !5
  %1409 = extractelement <4 x float> %1216, i64 1, !dbg !30
  %1410 = call float @llvm.amdgcn.exp2.f32(float %1409), !dbg !30
  %1411 = insertelement <4 x float> %1408, float %1410, i64 1, !dbg !31
  %local.reduce94 = fadd float %local.reduce90, %1410, !dbg !32, !frisk.local_reduction !5
  %1412 = extractelement <4 x float> %1216, i64 2, !dbg !30
  %1413 = call float @llvm.amdgcn.exp2.f32(float %1412), !dbg !30
  %1414 = insertelement <4 x float> %1411, float %1413, i64 2, !dbg !31
  %local.reduce95 = fadd float %local.reduce91, %1413, !dbg !32, !frisk.local_reduction !5
  %1415 = extractelement <4 x float> %1216, i64 3, !dbg !30
  %1416 = call float @llvm.amdgcn.exp2.f32(float %1415), !dbg !30
  %1417 = insertelement <4 x float> %1414, float %1416, i64 3, !dbg !31
  %math.result246 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1417) #9, !dbg !31, !frisk.register_lifetimes !5
  %1418 = fptrunc <4 x float> %math.result246 to <4 x half>, !dbg !31
  %math.packed247 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1418) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce96 = fadd float %local.reduce92, %1416, !dbg !32, !frisk.local_reduction !5
  %1419 = extractelement <4 x float> %1217, i64 0, !dbg !30
  %1420 = call float @llvm.amdgcn.exp2.f32(float %1419), !dbg !30
  %1421 = insertelement <4 x float> poison, float %1420, i64 0, !dbg !31
  %local.reduce97 = fadd float %local.reduce93, %1420, !dbg !32, !frisk.local_reduction !5
  %1422 = extractelement <4 x float> %1217, i64 1, !dbg !30
  %1423 = call float @llvm.amdgcn.exp2.f32(float %1422), !dbg !30
  %1424 = insertelement <4 x float> %1421, float %1423, i64 1, !dbg !31
  %local.reduce98 = fadd float %local.reduce94, %1423, !dbg !32, !frisk.local_reduction !5
  %1425 = extractelement <4 x float> %1217, i64 2, !dbg !30
  %1426 = call float @llvm.amdgcn.exp2.f32(float %1425), !dbg !30
  %1427 = insertelement <4 x float> %1424, float %1426, i64 2, !dbg !31
  %local.reduce99 = fadd float %local.reduce95, %1426, !dbg !32, !frisk.local_reduction !5
  %1428 = extractelement <4 x float> %1217, i64 3, !dbg !30
  %1429 = call float @llvm.amdgcn.exp2.f32(float %1428), !dbg !30
  %1430 = insertelement <4 x float> %1427, float %1429, i64 3, !dbg !31
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1430) #9, !dbg !31, !frisk.register_lifetimes !5
  %1431 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !31
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1431) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce100 = fadd float %local.reduce96, %1429, !dbg !32, !frisk.local_reduction !5
  %1432 = extractelement <4 x float> %1218, i64 0, !dbg !30
  %1433 = call float @llvm.amdgcn.exp2.f32(float %1432), !dbg !30
  %1434 = insertelement <4 x float> poison, float %1433, i64 0, !dbg !31
  %local.reduce101 = fadd float %local.reduce97, %1433, !dbg !32, !frisk.local_reduction !5
  %1435 = extractelement <4 x float> %1218, i64 1, !dbg !30
  %1436 = call float @llvm.amdgcn.exp2.f32(float %1435), !dbg !30
  %1437 = insertelement <4 x float> %1434, float %1436, i64 1, !dbg !31
  %local.reduce102 = fadd float %local.reduce98, %1436, !dbg !32, !frisk.local_reduction !5
  %1438 = extractelement <4 x float> %1218, i64 2, !dbg !30
  %1439 = call float @llvm.amdgcn.exp2.f32(float %1438), !dbg !30
  %1440 = insertelement <4 x float> %1437, float %1439, i64 2, !dbg !31
  %local.reduce103 = fadd float %local.reduce99, %1439, !dbg !32, !frisk.local_reduction !5
  %1441 = extractelement <4 x float> %1218, i64 3, !dbg !30
  %1442 = call float @llvm.amdgcn.exp2.f32(float %1441), !dbg !30
  %1443 = insertelement <4 x float> %1440, float %1442, i64 3, !dbg !31
  %math.result250 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1443) #9, !dbg !31, !frisk.register_lifetimes !5
  %1444 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !31
  %math.packed251 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1444) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce104 = fadd float %local.reduce100, %1442, !dbg !32, !frisk.local_reduction !5
  %1445 = extractelement <4 x float> %1219, i64 0, !dbg !30
  %1446 = call float @llvm.amdgcn.exp2.f32(float %1445), !dbg !30
  %1447 = insertelement <4 x float> poison, float %1446, i64 0, !dbg !31
  %local.reduce105 = fadd float %local.reduce101, %1446, !dbg !32, !frisk.local_reduction !5
  %1448 = extractelement <4 x float> %1219, i64 1, !dbg !30
  %1449 = call float @llvm.amdgcn.exp2.f32(float %1448), !dbg !30
  %1450 = insertelement <4 x float> %1447, float %1449, i64 1, !dbg !31
  %local.reduce106 = fadd float %local.reduce102, %1449, !dbg !32, !frisk.local_reduction !5
  %1451 = extractelement <4 x float> %1219, i64 2, !dbg !30
  %1452 = call float @llvm.amdgcn.exp2.f32(float %1451), !dbg !30
  %1453 = insertelement <4 x float> %1450, float %1452, i64 2, !dbg !31
  %local.reduce107 = fadd float %local.reduce103, %1452, !dbg !32, !frisk.local_reduction !5
  %1454 = extractelement <4 x float> %1219, i64 3, !dbg !30
  %1455 = call float @llvm.amdgcn.exp2.f32(float %1454), !dbg !30
  %1456 = insertelement <4 x float> %1453, float %1455, i64 3, !dbg !31
  %math.result252 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1456) #9, !dbg !31, !frisk.register_lifetimes !5
  %1457 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !31
  %math.packed253 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1457) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce108 = fadd float %local.reduce104, %1455, !dbg !32, !frisk.local_reduction !5
  %1458 = extractelement <4 x float> %1220, i64 0, !dbg !30
  %1459 = call float @llvm.amdgcn.exp2.f32(float %1458), !dbg !30
  %1460 = insertelement <4 x float> poison, float %1459, i64 0, !dbg !31
  %local.reduce109 = fadd float %local.reduce105, %1459, !dbg !32, !frisk.local_reduction !5
  %1461 = extractelement <4 x float> %1220, i64 1, !dbg !30
  %1462 = call float @llvm.amdgcn.exp2.f32(float %1461), !dbg !30
  %1463 = insertelement <4 x float> %1460, float %1462, i64 1, !dbg !31
  %local.reduce110 = fadd float %local.reduce106, %1462, !dbg !32, !frisk.local_reduction !5
  %1464 = extractelement <4 x float> %1220, i64 2, !dbg !30
  %1465 = call float @llvm.amdgcn.exp2.f32(float %1464), !dbg !30
  %1466 = insertelement <4 x float> %1463, float %1465, i64 2, !dbg !31
  %local.reduce111 = fadd float %local.reduce107, %1465, !dbg !32, !frisk.local_reduction !5
  %1467 = extractelement <4 x float> %1220, i64 3, !dbg !30
  %1468 = call float @llvm.amdgcn.exp2.f32(float %1467), !dbg !30
  %1469 = insertelement <4 x float> %1466, float %1468, i64 3, !dbg !31
  %math.result254 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1469) #9, !dbg !31, !frisk.register_lifetimes !5
  %1470 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !31
  %math.packed255 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1470) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce112 = fadd float %local.reduce108, %1468, !dbg !32, !frisk.local_reduction !5
  %1471 = extractelement <4 x float> %1221, i64 0, !dbg !30
  %1472 = call float @llvm.amdgcn.exp2.f32(float %1471), !dbg !30
  %1473 = insertelement <4 x float> poison, float %1472, i64 0, !dbg !31
  %local.reduce113 = fadd float %local.reduce109, %1472, !dbg !32, !frisk.local_reduction !5
  %1474 = extractelement <4 x float> %1221, i64 1, !dbg !30
  %1475 = call float @llvm.amdgcn.exp2.f32(float %1474), !dbg !30
  %1476 = insertelement <4 x float> %1473, float %1475, i64 1, !dbg !31
  %local.reduce114 = fadd float %local.reduce110, %1475, !dbg !32, !frisk.local_reduction !5
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !32, !frisk.local_reduction !5
  %1477 = extractelement <4 x float> %1221, i64 2, !dbg !30
  %1478 = call float @llvm.amdgcn.exp2.f32(float %1477), !dbg !30
  %1479 = insertelement <4 x float> %1476, float %1478, i64 2, !dbg !31
  %local.reduce115 = fadd float %local.reduce111, %1478, !dbg !32, !frisk.local_reduction !5
  %1480 = extractelement <4 x float> %1221, i64 3, !dbg !30
  %1481 = call float @llvm.amdgcn.exp2.f32(float %1480), !dbg !30
  %1482 = insertelement <4 x float> %1479, float %1481, i64 3, !dbg !31
  %math.result256 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1482) #9, !dbg !31, !frisk.register_lifetimes !5
  %1483 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !31
  %math.packed257 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1483) #9, !dbg !32, !frisk.register_lifetimes !5
  %local.reduce116 = fadd float %local.reduce112, %1481, !dbg !32, !frisk.local_reduction !5
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !32, !frisk.local_reduction !5
  %1484 = fadd float %local.reduce117, %local.reduce118, !dbg !32, !frisk.local_reduction !5
  %1485 = bitcast float %1379 to i32, !dbg !32
  %1486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1485), !dbg !32
  %1487 = bitcast i32 %1486 to float, !dbg !32
  %1488 = fadd float %1379, %1487, !dbg !32
  %1489 = bitcast float %1488 to i32, !dbg !32
  %1490 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1489), !dbg !32
  %1491 = bitcast i32 %1490 to float, !dbg !32
  %1492 = fadd float %1488, %1491, !dbg !32
  %1493 = fadd float %1492, %1271, !dbg !32
  %1494 = bitcast float %1484 to i32, !dbg !32
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1494), !dbg !32
  %1496 = bitcast i32 %1495 to float, !dbg !32
  %1497 = fadd float %1484, %1496, !dbg !32
  %1498 = bitcast float %1497 to i32, !dbg !32
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1498), !dbg !32
  %1500 = bitcast i32 %1499 to float, !dbg !32
  %1501 = fadd float %1497, %1500, !dbg !32
  %1502 = fadd float %1501, %1274, !dbg !32
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !29
  fence syncscope("workgroup") release, !dbg !29
  call void @llvm.amdgcn.s.barrier(), !dbg !29
  fence syncscope("workgroup") acquire, !dbg !29
  %address.local836 = shl nuw nsw i32 %address.thread798, 3, !dbg !14
  %address.local837 = and i32 %address.local836, 504, !dbg !14
  %address.local838 = zext nneg i32 %address.local837 to i64, !dbg !14
  %address.local839 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !14
  %1503 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local839, i32 0) #0, !dbg !14
  %1504 = bitcast <4 x i32> %1503 to <8 x half>, !dbg !14
  %address.local840 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !14
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local840, i64 1024, !dbg !14
  %1505 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #0, !dbg !14
  %1506 = bitcast <4 x i32> %1505 to <8 x half>, !dbg !14
  %address.local842 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !14
  %address.local843 = getelementptr i8, ptr addrspace(3) %address.local842, i64 2048, !dbg !14
  %1507 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local843, i32 0) #0, !dbg !14
  %1508 = bitcast <4 x i32> %1507 to <8 x half>, !dbg !14
  %address.local844 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !14
  %address.local845 = getelementptr i8, ptr addrspace(3) %address.local844, i64 3072, !dbg !14
  %1509 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local845, i32 0) #0, !dbg !14
  %1510 = bitcast <4 x i32> %1509 to <8 x half>, !dbg !14
  %address.local846 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !14
  %address.local847 = getelementptr i8, ptr addrspace(3) %address.local846, i64 4096, !dbg !14
  %1511 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local847, i32 0) #0, !dbg !14
  %1512 = bitcast <4 x i32> %1511 to <8 x half>, !dbg !14
  %address.local848 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !14
  %address.local849 = getelementptr i8, ptr addrspace(3) %address.local848, i64 5120, !dbg !14
  %1513 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local849, i32 0) #0, !dbg !14
  %1514 = bitcast <4 x i32> %1513 to <8 x half>, !dbg !14
  %address.local850 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !14
  %address.local851 = getelementptr i8, ptr addrspace(3) %address.local850, i64 6144, !dbg !14
  %1515 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local851, i32 0) #0, !dbg !14
  %1516 = bitcast <4 x i32> %1515 to <8 x half>, !dbg !14
  %address.local852 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local838, !dbg !14
  %address.local853 = getelementptr i8, ptr addrspace(3) %address.local852, i64 7168, !dbg !14
  %1517 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local853, i32 0) #0, !dbg !14
  %1518 = bitcast <4 x i32> %1517 to <8 x half>, !dbg !14
  %1519 = shufflevector <8 x half> %1504, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1519, <4 x float> zeroinitializer) #9, !dbg !14
  %1521 = shufflevector <8 x half> %1506, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1521, <4 x float> %1520) #9, !dbg !14
  %1523 = shufflevector <8 x half> %1508, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1523, <4 x float> %1522) #9, !dbg !14
  %1525 = shufflevector <8 x half> %1510, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1525, <4 x float> %1524) #9, !dbg !14
  %1527 = shufflevector <8 x half> %1512, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1527, <4 x float> %1526) #9, !dbg !14
  %1529 = shufflevector <8 x half> %1514, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1529, <4 x float> %1528) #9, !dbg !14
  %1531 = shufflevector <8 x half> %1516, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1531, <4 x float> %1530) #9, !dbg !14
  %1533 = shufflevector <8 x half> %1518, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1533, <4 x float> %1532) #9, !dbg !14
  %1535 = extractelement <4 x float> %1534, i64 0, !dbg !14
  %1536 = extractelement <4 x float> %1534, i64 1, !dbg !14
  %1537 = extractelement <4 x float> %1534, i64 2, !dbg !14
  %1538 = extractelement <4 x float> %1534, i64 3, !dbg !14
  %1539 = select i1 %.not, float %1536, float %1535, !dbg !14
  %1540 = bitcast float %1539 to i32, !dbg !14
  %1541 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1540), !dbg !14
  %1542 = bitcast i32 %1541 to float, !dbg !14
  %1543 = select i1 %.not, float %1535, float %1542, !dbg !14
  %1544 = select i1 %.not, float %1542, float %1536, !dbg !14
  %1545 = select i1 %.not, float %1538, float %1537, !dbg !14
  %1546 = bitcast float %1545 to i32, !dbg !14
  %1547 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1546), !dbg !14
  %1548 = bitcast i32 %1547 to float, !dbg !14
  %1549 = select i1 %.not, float %1537, float %1548, !dbg !14
  %1550 = select i1 %.not, float %1548, float %1538, !dbg !14
  %1551 = select i1 %.not1, float %1549, float %1543, !dbg !14
  %1552 = bitcast float %1551 to i32, !dbg !14
  %1553 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1552), !dbg !14
  %1554 = bitcast i32 %1553 to float, !dbg !14
  %1555 = select i1 %.not1, float %1543, float %1554, !dbg !14
  %1556 = select i1 %.not1, float %1554, float %1549, !dbg !14
  %1557 = select i1 %.not1, float %1550, float %1544, !dbg !14
  %1558 = bitcast float %1557 to i32, !dbg !14
  %1559 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1558), !dbg !14
  %1560 = bitcast i32 %1559 to float, !dbg !14
  %1561 = select i1 %.not1, float %1544, float %1560, !dbg !14
  %1562 = select i1 %.not1, float %1560, float %1550, !dbg !14
  %1563 = shufflevector <8 x half> %1504, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1563, <4 x float> zeroinitializer) #9, !dbg !14
  %1565 = shufflevector <8 x half> %1506, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1565, <4 x float> %1564) #9, !dbg !14
  %1567 = shufflevector <8 x half> %1508, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1567, <4 x float> %1566) #9, !dbg !14
  %1569 = shufflevector <8 x half> %1510, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1569, <4 x float> %1568) #9, !dbg !14
  %1571 = shufflevector <8 x half> %1512, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1571, <4 x float> %1570) #9, !dbg !14
  %1573 = shufflevector <8 x half> %1514, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1573, <4 x float> %1572) #9, !dbg !14
  %1575 = shufflevector <8 x half> %1516, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1575, <4 x float> %1574) #9, !dbg !14
  %1577 = shufflevector <8 x half> %1518, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1577, <4 x float> %1576) #9, !dbg !14
  %1579 = extractelement <4 x float> %1578, i64 0, !dbg !14
  %1580 = extractelement <4 x float> %1578, i64 1, !dbg !14
  %1581 = extractelement <4 x float> %1578, i64 2, !dbg !14
  %1582 = extractelement <4 x float> %1578, i64 3, !dbg !14
  %1583 = select i1 %.not, float %1580, float %1579, !dbg !14
  %1584 = bitcast float %1583 to i32, !dbg !14
  %1585 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1584), !dbg !14
  %1586 = bitcast i32 %1585 to float, !dbg !14
  %1587 = select i1 %.not, float %1579, float %1586, !dbg !14
  %1588 = select i1 %.not, float %1586, float %1580, !dbg !14
  %1589 = select i1 %.not, float %1582, float %1581, !dbg !14
  %1590 = bitcast float %1589 to i32, !dbg !14
  %1591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1590), !dbg !14
  %1592 = bitcast i32 %1591 to float, !dbg !14
  %1593 = select i1 %.not, float %1581, float %1592, !dbg !14
  %1594 = select i1 %.not, float %1592, float %1582, !dbg !14
  %1595 = select i1 %.not1, float %1593, float %1587, !dbg !14
  %1596 = bitcast float %1595 to i32, !dbg !14
  %1597 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1596), !dbg !14
  %1598 = bitcast i32 %1597 to float, !dbg !14
  %1599 = select i1 %.not1, float %1587, float %1598, !dbg !14
  %1600 = select i1 %.not1, float %1598, float %1593, !dbg !14
  %1601 = select i1 %.not1, float %1594, float %1588, !dbg !14
  %1602 = bitcast float %1601 to i32, !dbg !14
  %1603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1602), !dbg !14
  %1604 = bitcast i32 %1603 to float, !dbg !14
  %1605 = select i1 %.not1, float %1588, float %1604, !dbg !14
  %1606 = select i1 %.not1, float %1604, float %1594, !dbg !14
  %1607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1519, <4 x float> zeroinitializer) #9, !dbg !14
  %1608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1521, <4 x float> %1607) #9, !dbg !14
  %1609 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1523, <4 x float> %1608) #9, !dbg !14
  %1610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1525, <4 x float> %1609) #9, !dbg !14
  %1611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1527, <4 x float> %1610) #9, !dbg !14
  %1612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1529, <4 x float> %1611) #9, !dbg !14
  %1613 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1531, <4 x float> %1612) #9, !dbg !14
  %1614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1533, <4 x float> %1613) #9, !dbg !14
  %1615 = extractelement <4 x float> %1614, i64 0, !dbg !14
  %1616 = extractelement <4 x float> %1614, i64 1, !dbg !14
  %1617 = extractelement <4 x float> %1614, i64 2, !dbg !14
  %1618 = extractelement <4 x float> %1614, i64 3, !dbg !14
  %1619 = select i1 %.not, float %1616, float %1615, !dbg !14
  %1620 = bitcast float %1619 to i32, !dbg !14
  %1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1620), !dbg !14
  %1622 = bitcast i32 %1621 to float, !dbg !14
  %1623 = select i1 %.not, float %1615, float %1622, !dbg !14
  %1624 = select i1 %.not, float %1622, float %1616, !dbg !14
  %1625 = select i1 %.not, float %1618, float %1617, !dbg !14
  %1626 = bitcast float %1625 to i32, !dbg !14
  %1627 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1626), !dbg !14
  %1628 = bitcast i32 %1627 to float, !dbg !14
  %1629 = select i1 %.not, float %1617, float %1628, !dbg !14
  %1630 = select i1 %.not, float %1628, float %1618, !dbg !14
  %1631 = select i1 %.not1, float %1629, float %1623, !dbg !14
  %1632 = bitcast float %1631 to i32, !dbg !14
  %1633 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1632), !dbg !14
  %1634 = bitcast i32 %1633 to float, !dbg !14
  %1635 = select i1 %.not1, float %1623, float %1634, !dbg !14
  %1636 = select i1 %.not1, float %1634, float %1629, !dbg !14
  %1637 = select i1 %.not1, float %1630, float %1624, !dbg !14
  %1638 = bitcast float %1637 to i32, !dbg !14
  %1639 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1638), !dbg !14
  %1640 = bitcast i32 %1639 to float, !dbg !14
  %1641 = select i1 %.not1, float %1624, float %1640, !dbg !14
  %1642 = select i1 %.not1, float %1640, float %1630, !dbg !14
  %1643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1563, <4 x float> zeroinitializer) #9, !dbg !14
  %1644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1565, <4 x float> %1643) #9, !dbg !14
  %1645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1567, <4 x float> %1644) #9, !dbg !14
  %1646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1569, <4 x float> %1645) #9, !dbg !14
  %1647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1571, <4 x float> %1646) #9, !dbg !14
  %1648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1573, <4 x float> %1647) #9, !dbg !14
  %1649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1575, <4 x float> %1648) #9, !dbg !14
  %1650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1577, <4 x float> %1649) #9, !dbg !14
  %1651 = extractelement <4 x float> %1650, i64 0, !dbg !14
  %1652 = extractelement <4 x float> %1650, i64 1, !dbg !14
  %1653 = extractelement <4 x float> %1650, i64 2, !dbg !14
  %1654 = extractelement <4 x float> %1650, i64 3, !dbg !14
  %1655 = select i1 %.not, float %1652, float %1651, !dbg !14
  %1656 = bitcast float %1655 to i32, !dbg !14
  %1657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1656), !dbg !14
  %1658 = bitcast i32 %1657 to float, !dbg !14
  %1659 = select i1 %.not, float %1651, float %1658, !dbg !14
  %1660 = select i1 %.not, float %1658, float %1652, !dbg !14
  %1661 = select i1 %.not, float %1654, float %1653, !dbg !14
  %1662 = bitcast float %1661 to i32, !dbg !14
  %1663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1662), !dbg !14
  %1664 = bitcast i32 %1663 to float, !dbg !14
  %1665 = select i1 %.not, float %1653, float %1664, !dbg !14
  %1666 = select i1 %.not, float %1664, float %1654, !dbg !14
  %1667 = select i1 %.not1, float %1665, float %1659, !dbg !14
  %1668 = bitcast float %1667 to i32, !dbg !14
  %1669 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1668), !dbg !14
  %1670 = bitcast i32 %1669 to float, !dbg !14
  %1671 = select i1 %.not1, float %1659, float %1670, !dbg !14
  %1672 = select i1 %.not1, float %1670, float %1665, !dbg !14
  %1673 = select i1 %.not1, float %1666, float %1660, !dbg !14
  %1674 = bitcast float %1673 to i32, !dbg !14
  %1675 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1674), !dbg !14
  %1676 = bitcast i32 %1675 to float, !dbg !14
  %1677 = select i1 %.not1, float %1660, float %1676, !dbg !14
  %1678 = select i1 %.not1, float %1676, float %1666, !dbg !14
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  br i1 %43, label %1679, label %1680, !dbg !33

1679:                                             ; preds = %1233
  %address.thread854 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !33, !range !11, !frisk.register_lifetimes !5
  %address.local855 = lshr i32 %address.thread854, 1, !dbg !13
  %address.local856 = and i32 %address.local855, 96, !dbg !13
  %address.local857 = and i32 %address.thread854, 15, !dbg !13
  %address.local858 = or disjoint i32 %address.local856, %address.local857, !dbg !13
  %address.local859 = zext nneg i32 %address.local858 to i64, !dbg !13
  %address.local860 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !13
  store float %1230, ptr addrspace(3) %address.local860, align 4, !dbg !33
  %address.local861 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !13
  %address.local862 = getelementptr i8, ptr addrspace(3) %address.local861, i64 64, !dbg !13
  store float %1231, ptr addrspace(3) %address.local862, align 4, !dbg !33
  br label %1680, !dbg !33

1680:                                             ; preds = %1679, %1233
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  %address.thread863 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !33, !range !11, !frisk.register_lifetimes !5
  %lds.const.gep1747 = getelementptr i8, ptr addrspace(3) %address.local804, i64 0, !dbg !33
  %1681 = load float, ptr addrspace(3) %lds.const.gep1747, align 4, !dbg !33
  %lds.const.gep1748 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !33
  %1682 = load float, ptr addrspace(3) %lds.const.gep1748, align 4, !dbg !33
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  %1683 = add nuw nsw i64 %102, %15, !dbg !29
  %1684 = shl i64 %1683, 19, !dbg !29
  %1685 = and i64 %1684, 16252928, !dbg !29
  %1686 = add i64 %1685, %108, !dbg !29
  %1687 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1686, !dbg !29
  %1688 = getelementptr i8, ptr addrspace(1) %1687, i64 64, !dbg !29
  %address.local877 = zext nneg i32 %address.thread863 to i64, !dbg !10
  %address.local878 = shl nuw nsw i64 %address.local877, 3, !dbg !9
  %address.local879 = and i64 %address.local878, 1536, !dbg !9
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local879, !dbg !9
  %address.local881 = ptrtoint ptr addrspace(3) %address.local880 to i64, !dbg !9
  %address.local882 = trunc i64 %address.local881 to i32, !dbg !9
  %1689 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local882), !dbg !29
  %1690 = zext i32 %1689 to i64, !dbg !29
  %1691 = inttoptr i64 %1690 to ptr addrspace(3), !dbg !29
  %address.local883 = shl nuw nsw i64 %address.local877, 5, !dbg !29
  %address.local884 = and i64 %address.local883, 8064, !dbg !29
  %address.local885 = shl nuw nsw i64 %address.local877, 3, !dbg !29
  %address.local886 = and i64 %address.local885, 24, !dbg !29
  %address.local887 = or disjoint i64 %address.local884, %address.local886, !dbg !29
  %address.local888 = getelementptr [2 x i8], ptr addrspace(1) %1688, i64 %address.local887, !dbg !29
  %1692 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !29
  %1693 = ptrtoint ptr addrspace(1) %address.local888 to i64, !dbg !29
  %1694 = sub i64 %1693, %1692, !dbg !29
  %dma.buffer.offset1767 = trunc i64 %1694 to i32, !dbg !29
  %1695 = trunc i64 %1692 to i32, !dbg !29
  %1696 = insertelement <4 x i32> poison, i32 %1695, i64 0, !dbg !29
  %1697 = lshr i64 %1692, 32, !dbg !29
  %1698 = trunc i64 %1697 to i32, !dbg !29
  %1699 = insertelement <4 x i32> %1696, i32 %1698, i64 1, !dbg !29
  %1700 = insertelement <4 x i32> %1699, i32 33554432, i64 2, !dbg !29
  %dma.buffer.rsrc1768 = insertelement <4 x i32> %1700, i32 131072, i64 3, !dbg !29
  %1701 = ptrtoint ptr addrspace(3) %1691 to i32, !dbg !29
  %dma.buffer.lds1769 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1701), !dbg !29
  %1702 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1769, i32 %dma.buffer.offset1767, <4 x i32> %dma.buffer.rsrc1768) #10, !dbg !29
  %address.local889 = or disjoint i64 %address.local877, 256, !dbg !9
  %address.local890 = shl nuw nsw i64 %address.local889, 3, !dbg !9
  %address.local891 = and i64 %address.local890, 3584, !dbg !9
  %address.local892 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local891, !dbg !9
  %address.local893 = ptrtoint ptr addrspace(3) %address.local892 to i64, !dbg !9
  %address.local894 = trunc i64 %address.local893 to i32, !dbg !9
  %1703 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !29
  %1704 = zext i32 %1703 to i64, !dbg !29
  %1705 = inttoptr i64 %1704 to ptr addrspace(3), !dbg !29
  %address.local895 = shl nuw nsw i64 %address.local889, 5, !dbg !29
  %address.local896 = and i64 %address.local895, 16256, !dbg !29
  %address.local897 = shl nuw nsw i64 %address.local877, 3, !dbg !29
  %address.local898 = and i64 %address.local897, 24, !dbg !29
  %address.local899 = or disjoint i64 %address.local896, %address.local898, !dbg !29
  %address.local900 = getelementptr [2 x i8], ptr addrspace(1) %1688, i64 %address.local899, !dbg !29
  %1706 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !29
  %1707 = ptrtoint ptr addrspace(1) %address.local900 to i64, !dbg !29
  %1708 = sub i64 %1707, %1706, !dbg !29
  %dma.buffer.offset1764 = trunc i64 %1708 to i32, !dbg !29
  %1709 = trunc i64 %1706 to i32, !dbg !29
  %1710 = insertelement <4 x i32> poison, i32 %1709, i64 0, !dbg !29
  %1711 = lshr i64 %1706, 32, !dbg !29
  %1712 = trunc i64 %1711 to i32, !dbg !29
  %1713 = insertelement <4 x i32> %1710, i32 %1712, i64 1, !dbg !29
  %1714 = insertelement <4 x i32> %1713, i32 33554432, i64 2, !dbg !29
  %dma.buffer.rsrc1765 = insertelement <4 x i32> %1714, i32 131072, i64 3, !dbg !29
  %1715 = ptrtoint ptr addrspace(3) %1705 to i32, !dbg !29
  %dma.buffer.lds1766 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1715), !dbg !29
  %1716 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1766, i32 %dma.buffer.offset1764, <4 x i32> %dma.buffer.rsrc1765) #10, !dbg !29
  %1717 = extractvalue [2 x <8 x float>] %63, 0, !dbg !33
  %1718 = shufflevector <8 x float> %1717, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %1719 = insertelement <4 x float> poison, float %1681, i64 0, !dbg !33
  %1720 = shufflevector <4 x float> %1719, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !33
  %1721 = fmul <4 x float> %1718, %1720, !dbg !33
  %1722 = insertelement <4 x float> poison, float %1555, i64 0, !dbg !34
  %1723 = insertelement <4 x float> %1722, float %1561, i64 1, !dbg !34
  %1724 = insertelement <4 x float> %1723, float %1556, i64 2, !dbg !34
  %1725 = insertelement <4 x float> %1724, float %1562, i64 3, !dbg !34
  %1726 = fadd <4 x float> %1721, %1725, !dbg !34
  %1727 = shufflevector <8 x float> %1717, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %1728 = fmul <4 x float> %1727, %1720, !dbg !33
  %1729 = insertelement <4 x float> poison, float %1599, i64 0, !dbg !34
  %1730 = insertelement <4 x float> %1729, float %1605, i64 1, !dbg !34
  %1731 = insertelement <4 x float> %1730, float %1600, i64 2, !dbg !34
  %1732 = insertelement <4 x float> %1731, float %1606, i64 3, !dbg !34
  %1733 = fadd <4 x float> %1728, %1732, !dbg !34
  %1734 = extractvalue [2 x <8 x float>] %63, 1, !dbg !33
  %1735 = shufflevector <8 x float> %1734, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %1736 = insertelement <4 x float> poison, float %1682, i64 0, !dbg !33
  %1737 = shufflevector <4 x float> %1736, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !33
  %1738 = fmul <4 x float> %1735, %1737, !dbg !33
  %1739 = insertelement <4 x float> poison, float %1635, i64 0, !dbg !34
  %1740 = insertelement <4 x float> %1739, float %1641, i64 1, !dbg !34
  %1741 = insertelement <4 x float> %1740, float %1636, i64 2, !dbg !34
  %1742 = insertelement <4 x float> %1741, float %1642, i64 3, !dbg !34
  %1743 = fadd <4 x float> %1738, %1742, !dbg !34
  %1744 = shufflevector <8 x float> %1734, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %1745 = fmul <4 x float> %1744, %1737, !dbg !33
  %1746 = insertelement <4 x float> poison, float %1671, i64 0, !dbg !34
  %1747 = insertelement <4 x float> %1746, float %1677, i64 1, !dbg !34
  %1748 = insertelement <4 x float> %1747, float %1672, i64 2, !dbg !34
  %1749 = insertelement <4 x float> %1748, float %1678, i64 3, !dbg !34
  %1750 = fadd <4 x float> %1745, %1749, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !29
  fence syncscope("workgroup") release, !dbg !29
  call void @llvm.amdgcn.s.barrier(), !dbg !29
  fence syncscope("workgroup") acquire, !dbg !29
  %address.local901 = shl nuw nsw i32 %address.thread863, 3, !dbg !14
  %address.local902 = and i32 %address.local901, 504, !dbg !14
  %address.local903 = zext nneg i32 %address.local902 to i64, !dbg !14
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %1751 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #0, !dbg !14
  %1752 = bitcast <4 x i32> %1751 to <8 x half>, !dbg !14
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 1024, !dbg !14
  %1753 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1754 = bitcast <4 x i32> %1753 to <8 x half>, !dbg !14
  %address.local907 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %address.local908 = getelementptr i8, ptr addrspace(3) %address.local907, i64 2048, !dbg !14
  %1755 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local908, i32 0) #0, !dbg !14
  %1756 = bitcast <4 x i32> %1755 to <8 x half>, !dbg !14
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %address.local910 = getelementptr i8, ptr addrspace(3) %address.local909, i64 3072, !dbg !14
  %1757 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local910, i32 0) #0, !dbg !14
  %1758 = bitcast <4 x i32> %1757 to <8 x half>, !dbg !14
  %address.local911 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %address.local912 = getelementptr i8, ptr addrspace(3) %address.local911, i64 4096, !dbg !14
  %1759 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local912, i32 0) #0, !dbg !14
  %1760 = bitcast <4 x i32> %1759 to <8 x half>, !dbg !14
  %address.local913 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %address.local914 = getelementptr i8, ptr addrspace(3) %address.local913, i64 5120, !dbg !14
  %1761 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local914, i32 0) #0, !dbg !14
  %1762 = bitcast <4 x i32> %1761 to <8 x half>, !dbg !14
  %address.local915 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %address.local916 = getelementptr i8, ptr addrspace(3) %address.local915, i64 6144, !dbg !14
  %1763 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local916, i32 0) #0, !dbg !14
  %1764 = bitcast <4 x i32> %1763 to <8 x half>, !dbg !14
  %address.local917 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %address.local918 = getelementptr i8, ptr addrspace(3) %address.local917, i64 7168, !dbg !14
  %1765 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local918, i32 0) #0, !dbg !14
  %1766 = bitcast <4 x i32> %1765 to <8 x half>, !dbg !14
  %1767 = shufflevector <8 x half> %1752, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1767, <4 x float> zeroinitializer) #9, !dbg !14
  %1769 = shufflevector <8 x half> %1754, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1769, <4 x float> %1768) #9, !dbg !14
  %1771 = shufflevector <8 x half> %1756, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1772 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1771, <4 x float> %1770) #9, !dbg !14
  %1773 = shufflevector <8 x half> %1758, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1774 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1773, <4 x float> %1772) #9, !dbg !14
  %1775 = shufflevector <8 x half> %1760, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1775, <4 x float> %1774) #9, !dbg !14
  %1777 = shufflevector <8 x half> %1762, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1778 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1777, <4 x float> %1776) #9, !dbg !14
  %1779 = shufflevector <8 x half> %1764, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1780 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1779, <4 x float> %1778) #9, !dbg !14
  %1781 = shufflevector <8 x half> %1766, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1781, <4 x float> %1780) #9, !dbg !14
  %1783 = extractelement <4 x float> %1782, i64 0, !dbg !14
  %1784 = extractelement <4 x float> %1782, i64 1, !dbg !14
  %1785 = extractelement <4 x float> %1782, i64 2, !dbg !14
  %1786 = extractelement <4 x float> %1782, i64 3, !dbg !14
  %1787 = select i1 %.not, float %1784, float %1783, !dbg !14
  %1788 = bitcast float %1787 to i32, !dbg !14
  %1789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1788), !dbg !14
  %1790 = bitcast i32 %1789 to float, !dbg !14
  %1791 = select i1 %.not, float %1783, float %1790, !dbg !14
  %1792 = select i1 %.not, float %1790, float %1784, !dbg !14
  %1793 = select i1 %.not, float %1786, float %1785, !dbg !14
  %1794 = bitcast float %1793 to i32, !dbg !14
  %1795 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1794), !dbg !14
  %1796 = bitcast i32 %1795 to float, !dbg !14
  %1797 = select i1 %.not, float %1785, float %1796, !dbg !14
  %1798 = select i1 %.not, float %1796, float %1786, !dbg !14
  %1799 = select i1 %.not1, float %1797, float %1791, !dbg !14
  %1800 = bitcast float %1799 to i32, !dbg !14
  %1801 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1800), !dbg !14
  %1802 = bitcast i32 %1801 to float, !dbg !14
  %1803 = select i1 %.not1, float %1791, float %1802, !dbg !14
  %1804 = select i1 %.not1, float %1802, float %1797, !dbg !14
  %1805 = select i1 %.not1, float %1798, float %1792, !dbg !14
  %1806 = bitcast float %1805 to i32, !dbg !14
  %1807 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1806), !dbg !14
  %1808 = bitcast i32 %1807 to float, !dbg !14
  %1809 = select i1 %.not1, float %1792, float %1808, !dbg !14
  %1810 = select i1 %.not1, float %1808, float %1798, !dbg !14
  %1811 = shufflevector <8 x half> %1752, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1811, <4 x float> zeroinitializer) #9, !dbg !14
  %1813 = shufflevector <8 x half> %1754, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1813, <4 x float> %1812) #9, !dbg !14
  %1815 = shufflevector <8 x half> %1756, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1815, <4 x float> %1814) #9, !dbg !14
  %1817 = shufflevector <8 x half> %1758, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1817, <4 x float> %1816) #9, !dbg !14
  %1819 = shufflevector <8 x half> %1760, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1819, <4 x float> %1818) #9, !dbg !14
  %1821 = shufflevector <8 x half> %1762, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1821, <4 x float> %1820) #9, !dbg !14
  %1823 = shufflevector <8 x half> %1764, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1823, <4 x float> %1822) #9, !dbg !14
  %1825 = shufflevector <8 x half> %1766, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1825, <4 x float> %1824) #9, !dbg !14
  %1827 = extractelement <4 x float> %1826, i64 0, !dbg !14
  %1828 = extractelement <4 x float> %1826, i64 1, !dbg !14
  %1829 = extractelement <4 x float> %1826, i64 2, !dbg !14
  %1830 = extractelement <4 x float> %1826, i64 3, !dbg !14
  %1831 = select i1 %.not, float %1828, float %1827, !dbg !14
  %1832 = bitcast float %1831 to i32, !dbg !14
  %1833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1832), !dbg !14
  %1834 = bitcast i32 %1833 to float, !dbg !14
  %1835 = select i1 %.not, float %1827, float %1834, !dbg !14
  %1836 = select i1 %.not, float %1834, float %1828, !dbg !14
  %1837 = select i1 %.not, float %1830, float %1829, !dbg !14
  %1838 = bitcast float %1837 to i32, !dbg !14
  %1839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1838), !dbg !14
  %1840 = bitcast i32 %1839 to float, !dbg !14
  %1841 = select i1 %.not, float %1829, float %1840, !dbg !14
  %1842 = select i1 %.not, float %1840, float %1830, !dbg !14
  %1843 = select i1 %.not1, float %1841, float %1835, !dbg !14
  %1844 = bitcast float %1843 to i32, !dbg !14
  %1845 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1844), !dbg !14
  %1846 = bitcast i32 %1845 to float, !dbg !14
  %1847 = select i1 %.not1, float %1835, float %1846, !dbg !14
  %1848 = select i1 %.not1, float %1846, float %1841, !dbg !14
  %1849 = select i1 %.not1, float %1842, float %1836, !dbg !14
  %1850 = bitcast float %1849 to i32, !dbg !14
  %1851 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1850), !dbg !14
  %1852 = bitcast i32 %1851 to float, !dbg !14
  %1853 = select i1 %.not1, float %1836, float %1852, !dbg !14
  %1854 = select i1 %.not1, float %1852, float %1842, !dbg !14
  %1855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1767, <4 x float> zeroinitializer) #9, !dbg !14
  %1856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1769, <4 x float> %1855) #9, !dbg !14
  %1857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1771, <4 x float> %1856) #9, !dbg !14
  %1858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1773, <4 x float> %1857) #9, !dbg !14
  %1859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1775, <4 x float> %1858) #9, !dbg !14
  %1860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1777, <4 x float> %1859) #9, !dbg !14
  %1861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1779, <4 x float> %1860) #9, !dbg !14
  %1862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1781, <4 x float> %1861) #9, !dbg !14
  %1863 = extractelement <4 x float> %1862, i64 0, !dbg !14
  %1864 = extractelement <4 x float> %1862, i64 1, !dbg !14
  %1865 = extractelement <4 x float> %1862, i64 2, !dbg !14
  %1866 = extractelement <4 x float> %1862, i64 3, !dbg !14
  %1867 = select i1 %.not, float %1864, float %1863, !dbg !14
  %1868 = bitcast float %1867 to i32, !dbg !14
  %1869 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1868), !dbg !14
  %1870 = bitcast i32 %1869 to float, !dbg !14
  %1871 = select i1 %.not, float %1863, float %1870, !dbg !14
  %1872 = select i1 %.not, float %1870, float %1864, !dbg !14
  %1873 = select i1 %.not, float %1866, float %1865, !dbg !14
  %1874 = bitcast float %1873 to i32, !dbg !14
  %1875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1874), !dbg !14
  %1876 = bitcast i32 %1875 to float, !dbg !14
  %1877 = select i1 %.not, float %1865, float %1876, !dbg !14
  %1878 = select i1 %.not, float %1876, float %1866, !dbg !14
  %1879 = select i1 %.not1, float %1877, float %1871, !dbg !14
  %1880 = bitcast float %1879 to i32, !dbg !14
  %1881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1880), !dbg !14
  %1882 = bitcast i32 %1881 to float, !dbg !14
  %1883 = select i1 %.not1, float %1871, float %1882, !dbg !14
  %1884 = select i1 %.not1, float %1882, float %1877, !dbg !14
  %1885 = select i1 %.not1, float %1878, float %1872, !dbg !14
  %1886 = bitcast float %1885 to i32, !dbg !14
  %1887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1886), !dbg !14
  %1888 = bitcast i32 %1887 to float, !dbg !14
  %1889 = select i1 %.not1, float %1872, float %1888, !dbg !14
  %1890 = select i1 %.not1, float %1888, float %1878, !dbg !14
  %1891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1811, <4 x float> zeroinitializer) #9, !dbg !14
  %1892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1813, <4 x float> %1891) #9, !dbg !14
  %1893 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1815, <4 x float> %1892) #9, !dbg !14
  %1894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1817, <4 x float> %1893) #9, !dbg !14
  %1895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1819, <4 x float> %1894) #9, !dbg !14
  %1896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1821, <4 x float> %1895) #9, !dbg !14
  %1897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1823, <4 x float> %1896) #9, !dbg !14
  %1898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1825, <4 x float> %1897) #9, !dbg !14
  %1899 = extractelement <4 x float> %1898, i64 0, !dbg !14
  %1900 = extractelement <4 x float> %1898, i64 1, !dbg !14
  %1901 = extractelement <4 x float> %1898, i64 2, !dbg !14
  %1902 = extractelement <4 x float> %1898, i64 3, !dbg !14
  %1903 = select i1 %.not, float %1900, float %1899, !dbg !14
  %1904 = bitcast float %1903 to i32, !dbg !14
  %1905 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1904), !dbg !14
  %1906 = bitcast i32 %1905 to float, !dbg !14
  %1907 = select i1 %.not, float %1899, float %1906, !dbg !14
  %1908 = select i1 %.not, float %1906, float %1900, !dbg !14
  %1909 = select i1 %.not, float %1902, float %1901, !dbg !14
  %1910 = bitcast float %1909 to i32, !dbg !14
  %1911 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1910), !dbg !14
  %1912 = bitcast i32 %1911 to float, !dbg !14
  %1913 = select i1 %.not, float %1901, float %1912, !dbg !14
  %1914 = select i1 %.not, float %1912, float %1902, !dbg !14
  %1915 = select i1 %.not1, float %1913, float %1907, !dbg !14
  %1916 = bitcast float %1915 to i32, !dbg !14
  %1917 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1916), !dbg !14
  %1918 = bitcast i32 %1917 to float, !dbg !14
  %1919 = select i1 %.not1, float %1907, float %1918, !dbg !14
  %1920 = select i1 %.not1, float %1918, float %1913, !dbg !14
  %1921 = select i1 %.not1, float %1914, float %1908, !dbg !14
  %1922 = bitcast float %1921 to i32, !dbg !14
  %1923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1922), !dbg !14
  %1924 = bitcast i32 %1923 to float, !dbg !14
  %1925 = select i1 %.not1, float %1908, float %1924, !dbg !14
  %1926 = select i1 %.not1, float %1924, float %1914, !dbg !14
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  br i1 %43, label %1927, label %1928, !dbg !33

1927:                                             ; preds = %1680
  %address.thread919 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !33, !range !11, !frisk.register_lifetimes !5
  %address.local920 = lshr i32 %address.thread919, 1, !dbg !13
  %address.local921 = and i32 %address.local920, 96, !dbg !13
  %address.local922 = and i32 %address.thread919, 15, !dbg !13
  %address.local923 = or disjoint i32 %address.local921, %address.local922, !dbg !13
  %address.local924 = zext nneg i32 %address.local923 to i64, !dbg !13
  %address.local925 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !13
  store float %1230, ptr addrspace(3) %address.local925, align 4, !dbg !33
  %address.local926 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local924, !dbg !13
  %address.local927 = getelementptr i8, ptr addrspace(3) %address.local926, i64 64, !dbg !13
  store float %1231, ptr addrspace(3) %address.local927, align 4, !dbg !33
  br label %1928, !dbg !33

1928:                                             ; preds = %1927, %1680
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  %address.thread928 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !33, !range !11, !frisk.register_lifetimes !5
  %lds.const.gep1749 = getelementptr i8, ptr addrspace(3) %address.local804, i64 0, !dbg !33
  %1929 = load float, ptr addrspace(3) %lds.const.gep1749, align 4, !dbg !33
  %lds.const.gep1750 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !33
  %1930 = load float, ptr addrspace(3) %lds.const.gep1750, align 4, !dbg !33
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  %1931 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1686, !dbg !29
  %1932 = getelementptr i8, ptr addrspace(1) %1931, i64 128, !dbg !29
  %address.local942 = zext nneg i32 %address.thread928 to i64, !dbg !10
  %address.local943 = shl nuw nsw i64 %address.local942, 3, !dbg !9
  %address.local944 = and i64 %address.local943, 1536, !dbg !9
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local944, !dbg !9
  %address.local946 = ptrtoint ptr addrspace(3) %address.local945 to i64, !dbg !9
  %address.local947 = trunc i64 %address.local946 to i32, !dbg !9
  %1933 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local947), !dbg !29
  %1934 = zext i32 %1933 to i64, !dbg !29
  %1935 = inttoptr i64 %1934 to ptr addrspace(3), !dbg !29
  %address.local948 = shl nuw nsw i64 %address.local942, 5, !dbg !29
  %address.local949 = and i64 %address.local948, 8064, !dbg !29
  %address.local950 = shl nuw nsw i64 %address.local942, 3, !dbg !29
  %address.local951 = and i64 %address.local950, 24, !dbg !29
  %address.local952 = or disjoint i64 %address.local949, %address.local951, !dbg !29
  %address.local953 = getelementptr [2 x i8], ptr addrspace(1) %1932, i64 %address.local952, !dbg !29
  %1936 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !29
  %1937 = ptrtoint ptr addrspace(1) %address.local953 to i64, !dbg !29
  %1938 = sub i64 %1937, %1936, !dbg !29
  %dma.buffer.offset1761 = trunc i64 %1938 to i32, !dbg !29
  %1939 = trunc i64 %1936 to i32, !dbg !29
  %1940 = insertelement <4 x i32> poison, i32 %1939, i64 0, !dbg !29
  %1941 = lshr i64 %1936, 32, !dbg !29
  %1942 = trunc i64 %1941 to i32, !dbg !29
  %1943 = insertelement <4 x i32> %1940, i32 %1942, i64 1, !dbg !29
  %1944 = insertelement <4 x i32> %1943, i32 33554432, i64 2, !dbg !29
  %dma.buffer.rsrc1762 = insertelement <4 x i32> %1944, i32 131072, i64 3, !dbg !29
  %1945 = ptrtoint ptr addrspace(3) %1935 to i32, !dbg !29
  %dma.buffer.lds1763 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1945), !dbg !29
  %1946 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1763, i32 %dma.buffer.offset1761, <4 x i32> %dma.buffer.rsrc1762) #10, !dbg !29
  %address.local954 = or disjoint i64 %address.local942, 256, !dbg !9
  %address.local955 = shl nuw nsw i64 %address.local954, 3, !dbg !9
  %address.local956 = and i64 %address.local955, 3584, !dbg !9
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local956, !dbg !9
  %address.local958 = ptrtoint ptr addrspace(3) %address.local957 to i64, !dbg !9
  %address.local959 = trunc i64 %address.local958 to i32, !dbg !9
  %1947 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local959), !dbg !29
  %1948 = zext i32 %1947 to i64, !dbg !29
  %1949 = inttoptr i64 %1948 to ptr addrspace(3), !dbg !29
  %address.local960 = shl nuw nsw i64 %address.local954, 5, !dbg !29
  %address.local961 = and i64 %address.local960, 16256, !dbg !29
  %address.local962 = shl nuw nsw i64 %address.local942, 3, !dbg !29
  %address.local963 = and i64 %address.local962, 24, !dbg !29
  %address.local964 = or disjoint i64 %address.local961, %address.local963, !dbg !29
  %address.local965 = getelementptr [2 x i8], ptr addrspace(1) %1932, i64 %address.local964, !dbg !29
  %1950 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !29
  %1951 = ptrtoint ptr addrspace(1) %address.local965 to i64, !dbg !29
  %1952 = sub i64 %1951, %1950, !dbg !29
  %dma.buffer.offset1758 = trunc i64 %1952 to i32, !dbg !29
  %1953 = trunc i64 %1950 to i32, !dbg !29
  %1954 = insertelement <4 x i32> poison, i32 %1953, i64 0, !dbg !29
  %1955 = lshr i64 %1950, 32, !dbg !29
  %1956 = trunc i64 %1955 to i32, !dbg !29
  %1957 = insertelement <4 x i32> %1954, i32 %1956, i64 1, !dbg !29
  %1958 = insertelement <4 x i32> %1957, i32 33554432, i64 2, !dbg !29
  %dma.buffer.rsrc1759 = insertelement <4 x i32> %1958, i32 131072, i64 3, !dbg !29
  %1959 = ptrtoint ptr addrspace(3) %1949 to i32, !dbg !29
  %dma.buffer.lds1760 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1959), !dbg !29
  %1960 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1760, i32 %dma.buffer.offset1758, <4 x i32> %dma.buffer.rsrc1759) #10, !dbg !29
  %1961 = extractvalue [2 x <8 x float>] %66, 0, !dbg !33
  %1962 = shufflevector <8 x float> %1961, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %1963 = insertelement <4 x float> poison, float %1929, i64 0, !dbg !33
  %1964 = shufflevector <4 x float> %1963, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !33
  %1965 = fmul <4 x float> %1962, %1964, !dbg !33
  %1966 = insertelement <4 x float> poison, float %1803, i64 0, !dbg !34
  %1967 = insertelement <4 x float> %1966, float %1809, i64 1, !dbg !34
  %1968 = insertelement <4 x float> %1967, float %1804, i64 2, !dbg !34
  %1969 = insertelement <4 x float> %1968, float %1810, i64 3, !dbg !34
  %1970 = fadd <4 x float> %1965, %1969, !dbg !34
  %1971 = shufflevector <8 x float> %1961, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %1972 = fmul <4 x float> %1971, %1964, !dbg !33
  %1973 = insertelement <4 x float> poison, float %1847, i64 0, !dbg !34
  %1974 = insertelement <4 x float> %1973, float %1853, i64 1, !dbg !34
  %1975 = insertelement <4 x float> %1974, float %1848, i64 2, !dbg !34
  %1976 = insertelement <4 x float> %1975, float %1854, i64 3, !dbg !34
  %1977 = fadd <4 x float> %1972, %1976, !dbg !34
  %1978 = extractvalue [2 x <8 x float>] %66, 1, !dbg !33
  %1979 = shufflevector <8 x float> %1978, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %1980 = insertelement <4 x float> poison, float %1930, i64 0, !dbg !33
  %1981 = shufflevector <4 x float> %1980, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !33
  %1982 = fmul <4 x float> %1979, %1981, !dbg !33
  %1983 = insertelement <4 x float> poison, float %1883, i64 0, !dbg !34
  %1984 = insertelement <4 x float> %1983, float %1889, i64 1, !dbg !34
  %1985 = insertelement <4 x float> %1984, float %1884, i64 2, !dbg !34
  %1986 = insertelement <4 x float> %1985, float %1890, i64 3, !dbg !34
  %1987 = fadd <4 x float> %1982, %1986, !dbg !34
  %1988 = shufflevector <8 x float> %1978, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %1989 = fmul <4 x float> %1988, %1981, !dbg !33
  %1990 = insertelement <4 x float> poison, float %1919, i64 0, !dbg !34
  %1991 = insertelement <4 x float> %1990, float %1925, i64 1, !dbg !34
  %1992 = insertelement <4 x float> %1991, float %1920, i64 2, !dbg !34
  %1993 = insertelement <4 x float> %1992, float %1926, i64 3, !dbg !34
  %1994 = fadd <4 x float> %1989, %1993, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !29
  fence syncscope("workgroup") release, !dbg !29
  call void @llvm.amdgcn.s.barrier(), !dbg !29
  fence syncscope("workgroup") acquire, !dbg !29
  %address.local966 = shl nuw nsw i32 %address.thread928, 3, !dbg !14
  %address.local967 = and i32 %address.local966, 504, !dbg !14
  %address.local968 = zext nneg i32 %address.local967 to i64, !dbg !14
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !14
  %1995 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #0, !dbg !14
  %1996 = bitcast <4 x i32> %1995 to <8 x half>, !dbg !14
  %address.local970 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !14
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local970, i64 1024, !dbg !14
  %1997 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #0, !dbg !14
  %1998 = bitcast <4 x i32> %1997 to <8 x half>, !dbg !14
  %address.local972 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !14
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local972, i64 2048, !dbg !14
  %1999 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !14
  %2000 = bitcast <4 x i32> %1999 to <8 x half>, !dbg !14
  %address.local974 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !14
  %address.local975 = getelementptr i8, ptr addrspace(3) %address.local974, i64 3072, !dbg !14
  %2001 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #0, !dbg !14
  %2002 = bitcast <4 x i32> %2001 to <8 x half>, !dbg !14
  %address.local976 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !14
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local976, i64 4096, !dbg !14
  %2003 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14
  %2004 = bitcast <4 x i32> %2003 to <8 x half>, !dbg !14
  %address.local978 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !14
  %address.local979 = getelementptr i8, ptr addrspace(3) %address.local978, i64 5120, !dbg !14
  %2005 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #0, !dbg !14
  %2006 = bitcast <4 x i32> %2005 to <8 x half>, !dbg !14
  %address.local980 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !14
  %address.local981 = getelementptr i8, ptr addrspace(3) %address.local980, i64 6144, !dbg !14
  %2007 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #0, !dbg !14
  %2008 = bitcast <4 x i32> %2007 to <8 x half>, !dbg !14
  %address.local982 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local968, !dbg !14
  %address.local983 = getelementptr i8, ptr addrspace(3) %address.local982, i64 7168, !dbg !14
  %2009 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #0, !dbg !14
  %2010 = bitcast <4 x i32> %2009 to <8 x half>, !dbg !14
  %2011 = shufflevector <8 x half> %1996, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2011, <4 x float> zeroinitializer) #9, !dbg !14
  %2013 = shufflevector <8 x half> %1998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2013, <4 x float> %2012) #9, !dbg !14
  %2015 = shufflevector <8 x half> %2000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2015, <4 x float> %2014) #9, !dbg !14
  %2017 = shufflevector <8 x half> %2002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2017, <4 x float> %2016) #9, !dbg !14
  %2019 = shufflevector <8 x half> %2004, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2019, <4 x float> %2018) #9, !dbg !14
  %2021 = shufflevector <8 x half> %2006, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2021, <4 x float> %2020) #9, !dbg !14
  %2023 = shufflevector <8 x half> %2008, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2023, <4 x float> %2022) #9, !dbg !14
  %2025 = shufflevector <8 x half> %2010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2025, <4 x float> %2024) #9, !dbg !14
  %2027 = extractelement <4 x float> %2026, i64 0, !dbg !14
  %2028 = extractelement <4 x float> %2026, i64 1, !dbg !14
  %2029 = extractelement <4 x float> %2026, i64 2, !dbg !14
  %2030 = extractelement <4 x float> %2026, i64 3, !dbg !14
  %2031 = select i1 %.not, float %2028, float %2027, !dbg !14
  %2032 = bitcast float %2031 to i32, !dbg !14
  %2033 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2032), !dbg !14
  %2034 = bitcast i32 %2033 to float, !dbg !14
  %2035 = select i1 %.not, float %2027, float %2034, !dbg !14
  %2036 = select i1 %.not, float %2034, float %2028, !dbg !14
  %2037 = select i1 %.not, float %2030, float %2029, !dbg !14
  %2038 = bitcast float %2037 to i32, !dbg !14
  %2039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2038), !dbg !14
  %2040 = bitcast i32 %2039 to float, !dbg !14
  %2041 = select i1 %.not, float %2029, float %2040, !dbg !14
  %2042 = select i1 %.not, float %2040, float %2030, !dbg !14
  %2043 = select i1 %.not1, float %2041, float %2035, !dbg !14
  %2044 = bitcast float %2043 to i32, !dbg !14
  %2045 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2044), !dbg !14
  %2046 = bitcast i32 %2045 to float, !dbg !14
  %2047 = select i1 %.not1, float %2035, float %2046, !dbg !14
  %2048 = select i1 %.not1, float %2046, float %2041, !dbg !14
  %2049 = select i1 %.not1, float %2042, float %2036, !dbg !14
  %2050 = bitcast float %2049 to i32, !dbg !14
  %2051 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2050), !dbg !14
  %2052 = bitcast i32 %2051 to float, !dbg !14
  %2053 = select i1 %.not1, float %2036, float %2052, !dbg !14
  %2054 = select i1 %.not1, float %2052, float %2042, !dbg !14
  %2055 = shufflevector <8 x half> %1996, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2055, <4 x float> zeroinitializer) #9, !dbg !14
  %2057 = shufflevector <8 x half> %1998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2057, <4 x float> %2056) #9, !dbg !14
  %2059 = shufflevector <8 x half> %2000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2059, <4 x float> %2058) #9, !dbg !14
  %2061 = shufflevector <8 x half> %2002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2061, <4 x float> %2060) #9, !dbg !14
  %2063 = shufflevector <8 x half> %2004, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2063, <4 x float> %2062) #9, !dbg !14
  %2065 = shufflevector <8 x half> %2006, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2065, <4 x float> %2064) #9, !dbg !14
  %2067 = shufflevector <8 x half> %2008, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2067, <4 x float> %2066) #9, !dbg !14
  %2069 = shufflevector <8 x half> %2010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2069, <4 x float> %2068) #9, !dbg !14
  %2071 = extractelement <4 x float> %2070, i64 0, !dbg !14
  %2072 = extractelement <4 x float> %2070, i64 1, !dbg !14
  %2073 = extractelement <4 x float> %2070, i64 2, !dbg !14
  %2074 = extractelement <4 x float> %2070, i64 3, !dbg !14
  %2075 = select i1 %.not, float %2072, float %2071, !dbg !14
  %2076 = bitcast float %2075 to i32, !dbg !14
  %2077 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2076), !dbg !14
  %2078 = bitcast i32 %2077 to float, !dbg !14
  %2079 = select i1 %.not, float %2071, float %2078, !dbg !14
  %2080 = select i1 %.not, float %2078, float %2072, !dbg !14
  %2081 = select i1 %.not, float %2074, float %2073, !dbg !14
  %2082 = bitcast float %2081 to i32, !dbg !14
  %2083 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2082), !dbg !14
  %2084 = bitcast i32 %2083 to float, !dbg !14
  %2085 = select i1 %.not, float %2073, float %2084, !dbg !14
  %2086 = select i1 %.not, float %2084, float %2074, !dbg !14
  %2087 = select i1 %.not1, float %2085, float %2079, !dbg !14
  %2088 = bitcast float %2087 to i32, !dbg !14
  %2089 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2088), !dbg !14
  %2090 = bitcast i32 %2089 to float, !dbg !14
  %2091 = select i1 %.not1, float %2079, float %2090, !dbg !14
  %2092 = select i1 %.not1, float %2090, float %2085, !dbg !14
  %2093 = select i1 %.not1, float %2086, float %2080, !dbg !14
  %2094 = bitcast float %2093 to i32, !dbg !14
  %2095 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2094), !dbg !14
  %2096 = bitcast i32 %2095 to float, !dbg !14
  %2097 = select i1 %.not1, float %2080, float %2096, !dbg !14
  %2098 = select i1 %.not1, float %2096, float %2086, !dbg !14
  %2099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2011, <4 x float> zeroinitializer) #9, !dbg !14
  %2100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2013, <4 x float> %2099) #9, !dbg !14
  %2101 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2015, <4 x float> %2100) #9, !dbg !14
  %2102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2017, <4 x float> %2101) #9, !dbg !14
  %2103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2019, <4 x float> %2102) #9, !dbg !14
  %2104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2021, <4 x float> %2103) #9, !dbg !14
  %2105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2023, <4 x float> %2104) #9, !dbg !14
  %2106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2025, <4 x float> %2105) #9, !dbg !14
  %2107 = extractelement <4 x float> %2106, i64 0, !dbg !14
  %2108 = extractelement <4 x float> %2106, i64 1, !dbg !14
  %2109 = extractelement <4 x float> %2106, i64 2, !dbg !14
  %2110 = extractelement <4 x float> %2106, i64 3, !dbg !14
  %2111 = select i1 %.not, float %2108, float %2107, !dbg !14
  %2112 = bitcast float %2111 to i32, !dbg !14
  %2113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2112), !dbg !14
  %2114 = bitcast i32 %2113 to float, !dbg !14
  %2115 = select i1 %.not, float %2107, float %2114, !dbg !14
  %2116 = select i1 %.not, float %2114, float %2108, !dbg !14
  %2117 = select i1 %.not, float %2110, float %2109, !dbg !14
  %2118 = bitcast float %2117 to i32, !dbg !14
  %2119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2118), !dbg !14
  %2120 = bitcast i32 %2119 to float, !dbg !14
  %2121 = select i1 %.not, float %2109, float %2120, !dbg !14
  %2122 = select i1 %.not, float %2120, float %2110, !dbg !14
  %2123 = select i1 %.not1, float %2121, float %2115, !dbg !14
  %2124 = bitcast float %2123 to i32, !dbg !14
  %2125 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2124), !dbg !14
  %2126 = bitcast i32 %2125 to float, !dbg !14
  %2127 = select i1 %.not1, float %2115, float %2126, !dbg !14
  %2128 = select i1 %.not1, float %2126, float %2121, !dbg !14
  %2129 = select i1 %.not1, float %2122, float %2116, !dbg !14
  %2130 = bitcast float %2129 to i32, !dbg !14
  %2131 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2130), !dbg !14
  %2132 = bitcast i32 %2131 to float, !dbg !14
  %2133 = select i1 %.not1, float %2116, float %2132, !dbg !14
  %2134 = select i1 %.not1, float %2132, float %2122, !dbg !14
  %2135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2055, <4 x float> zeroinitializer) #9, !dbg !14
  %2136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2057, <4 x float> %2135) #9, !dbg !14
  %2137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2059, <4 x float> %2136) #9, !dbg !14
  %2138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2061, <4 x float> %2137) #9, !dbg !14
  %2139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2063, <4 x float> %2138) #9, !dbg !14
  %2140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2065, <4 x float> %2139) #9, !dbg !14
  %2141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2067, <4 x float> %2140) #9, !dbg !14
  %2142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2069, <4 x float> %2141) #9, !dbg !14
  %2143 = extractelement <4 x float> %2142, i64 0, !dbg !14
  %2144 = extractelement <4 x float> %2142, i64 1, !dbg !14
  %2145 = extractelement <4 x float> %2142, i64 2, !dbg !14
  %2146 = extractelement <4 x float> %2142, i64 3, !dbg !14
  %2147 = select i1 %.not, float %2144, float %2143, !dbg !14
  %2148 = bitcast float %2147 to i32, !dbg !14
  %2149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2148), !dbg !14
  %2150 = bitcast i32 %2149 to float, !dbg !14
  %2151 = select i1 %.not, float %2143, float %2150, !dbg !14
  %2152 = select i1 %.not, float %2150, float %2144, !dbg !14
  %2153 = select i1 %.not, float %2146, float %2145, !dbg !14
  %2154 = bitcast float %2153 to i32, !dbg !14
  %2155 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2154), !dbg !14
  %2156 = bitcast i32 %2155 to float, !dbg !14
  %2157 = select i1 %.not, float %2145, float %2156, !dbg !14
  %2158 = select i1 %.not, float %2156, float %2146, !dbg !14
  %2159 = select i1 %.not1, float %2157, float %2151, !dbg !14
  %2160 = bitcast float %2159 to i32, !dbg !14
  %2161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2160), !dbg !14
  %2162 = bitcast i32 %2161 to float, !dbg !14
  %2163 = select i1 %.not1, float %2151, float %2162, !dbg !14
  %2164 = select i1 %.not1, float %2162, float %2157, !dbg !14
  %2165 = select i1 %.not1, float %2158, float %2152, !dbg !14
  %2166 = bitcast float %2165 to i32, !dbg !14
  %2167 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2166), !dbg !14
  %2168 = bitcast i32 %2167 to float, !dbg !14
  %2169 = select i1 %.not1, float %2152, float %2168, !dbg !14
  %2170 = select i1 %.not1, float %2168, float %2158, !dbg !14
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  br i1 %43, label %2171, label %2172, !dbg !33

2171:                                             ; preds = %1928
  %address.thread984 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !33, !range !11, !frisk.register_lifetimes !5
  %address.local985 = lshr i32 %address.thread984, 1, !dbg !13
  %address.local986 = and i32 %address.local985, 96, !dbg !13
  %address.local987 = and i32 %address.thread984, 15, !dbg !13
  %address.local988 = or disjoint i32 %address.local986, %address.local987, !dbg !13
  %address.local989 = zext nneg i32 %address.local988 to i64, !dbg !13
  %address.local990 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !13
  store float %1230, ptr addrspace(3) %address.local990, align 4, !dbg !33
  %address.local991 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local989, !dbg !13
  %address.local992 = getelementptr i8, ptr addrspace(3) %address.local991, i64 64, !dbg !13
  store float %1231, ptr addrspace(3) %address.local992, align 4, !dbg !33
  br label %2172, !dbg !33

2172:                                             ; preds = %2171, %1928
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  %address.thread993 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !33, !range !11, !frisk.register_lifetimes !5
  %lds.const.gep1751 = getelementptr i8, ptr addrspace(3) %address.local804, i64 0, !dbg !33
  %2173 = load float, ptr addrspace(3) %lds.const.gep1751, align 4, !dbg !33
  %lds.const.gep1752 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !33
  %2174 = load float, ptr addrspace(3) %lds.const.gep1752, align 4, !dbg !33
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  %2175 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1686, !dbg !29
  %2176 = getelementptr i8, ptr addrspace(1) %2175, i64 192, !dbg !29
  %address.local1007 = zext nneg i32 %address.thread993 to i64, !dbg !10
  %address.local1008 = shl nuw nsw i64 %address.local1007, 3, !dbg !9
  %address.local1009 = and i64 %address.local1008, 1536, !dbg !9
  %address.local1010 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1009, !dbg !9
  %address.local1011 = ptrtoint ptr addrspace(3) %address.local1010 to i64, !dbg !9
  %address.local1012 = trunc i64 %address.local1011 to i32, !dbg !9
  %2177 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1012), !dbg !29
  %2178 = zext i32 %2177 to i64, !dbg !29
  %2179 = inttoptr i64 %2178 to ptr addrspace(3), !dbg !29
  %address.local1013 = shl nuw nsw i64 %address.local1007, 5, !dbg !29
  %address.local1014 = and i64 %address.local1013, 8064, !dbg !29
  %address.local1015 = shl nuw nsw i64 %address.local1007, 3, !dbg !29
  %address.local1016 = and i64 %address.local1015, 24, !dbg !29
  %address.local1017 = or disjoint i64 %address.local1014, %address.local1016, !dbg !29
  %address.local1018 = getelementptr [2 x i8], ptr addrspace(1) %2176, i64 %address.local1017, !dbg !29
  %2180 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !29
  %2181 = ptrtoint ptr addrspace(1) %address.local1018 to i64, !dbg !29
  %2182 = sub i64 %2181, %2180, !dbg !29
  %dma.buffer.offset1755 = trunc i64 %2182 to i32, !dbg !29
  %2183 = trunc i64 %2180 to i32, !dbg !29
  %2184 = insertelement <4 x i32> poison, i32 %2183, i64 0, !dbg !29
  %2185 = lshr i64 %2180, 32, !dbg !29
  %2186 = trunc i64 %2185 to i32, !dbg !29
  %2187 = insertelement <4 x i32> %2184, i32 %2186, i64 1, !dbg !29
  %2188 = insertelement <4 x i32> %2187, i32 33554432, i64 2, !dbg !29
  %dma.buffer.rsrc1756 = insertelement <4 x i32> %2188, i32 131072, i64 3, !dbg !29
  %2189 = ptrtoint ptr addrspace(3) %2179 to i32, !dbg !29
  %dma.buffer.lds1757 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %2189), !dbg !29
  %2190 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1757, i32 %dma.buffer.offset1755, <4 x i32> %dma.buffer.rsrc1756) #10, !dbg !29
  %address.local1019 = or disjoint i64 %address.local1007, 256, !dbg !9
  %address.local1020 = shl nuw nsw i64 %address.local1019, 3, !dbg !9
  %address.local1021 = and i64 %address.local1020, 3584, !dbg !9
  %address.local1022 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1021, !dbg !9
  %address.local1023 = ptrtoint ptr addrspace(3) %address.local1022 to i64, !dbg !9
  %address.local1024 = trunc i64 %address.local1023 to i32, !dbg !9
  %2191 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1024), !dbg !29
  %2192 = zext i32 %2191 to i64, !dbg !29
  %2193 = inttoptr i64 %2192 to ptr addrspace(3), !dbg !29
  %address.local1025 = shl nuw nsw i64 %address.local1019, 5, !dbg !29
  %address.local1026 = and i64 %address.local1025, 16256, !dbg !29
  %address.local1027 = shl nuw nsw i64 %address.local1007, 3, !dbg !29
  %address.local1028 = and i64 %address.local1027, 24, !dbg !29
  %address.local1029 = or disjoint i64 %address.local1026, %address.local1028, !dbg !29
  %address.local1030 = getelementptr [2 x i8], ptr addrspace(1) %2176, i64 %address.local1029, !dbg !29
  %2194 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !29
  %2195 = ptrtoint ptr addrspace(1) %address.local1030 to i64, !dbg !29
  %2196 = sub i64 %2195, %2194, !dbg !29
  %dma.buffer.offset = trunc i64 %2196 to i32, !dbg !29
  %2197 = trunc i64 %2194 to i32, !dbg !29
  %2198 = insertelement <4 x i32> poison, i32 %2197, i64 0, !dbg !29
  %2199 = lshr i64 %2194, 32, !dbg !29
  %2200 = trunc i64 %2199 to i32, !dbg !29
  %2201 = insertelement <4 x i32> %2198, i32 %2200, i64 1, !dbg !29
  %2202 = insertelement <4 x i32> %2201, i32 33554432, i64 2, !dbg !29
  %dma.buffer.rsrc = insertelement <4 x i32> %2202, i32 131072, i64 3, !dbg !29
  %2203 = ptrtoint ptr addrspace(3) %2193 to i32, !dbg !29
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %2203), !dbg !29
  %2204 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !29
  %2205 = extractvalue [2 x <8 x float>] %67, 0, !dbg !33
  %2206 = shufflevector <8 x float> %2205, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %2207 = insertelement <4 x float> poison, float %2173, i64 0, !dbg !33
  %2208 = shufflevector <4 x float> %2207, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !33
  %2209 = fmul <4 x float> %2206, %2208, !dbg !33
  %2210 = insertelement <4 x float> poison, float %2047, i64 0, !dbg !34
  %2211 = insertelement <4 x float> %2210, float %2053, i64 1, !dbg !34
  %2212 = insertelement <4 x float> %2211, float %2048, i64 2, !dbg !34
  %2213 = insertelement <4 x float> %2212, float %2054, i64 3, !dbg !34
  %2214 = fadd <4 x float> %2209, %2213, !dbg !34
  %2215 = shufflevector <8 x float> %2205, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %2216 = fmul <4 x float> %2215, %2208, !dbg !33
  %2217 = insertelement <4 x float> poison, float %2091, i64 0, !dbg !34
  %2218 = insertelement <4 x float> %2217, float %2097, i64 1, !dbg !34
  %2219 = insertelement <4 x float> %2218, float %2092, i64 2, !dbg !34
  %2220 = insertelement <4 x float> %2219, float %2098, i64 3, !dbg !34
  %2221 = fadd <4 x float> %2216, %2220, !dbg !34
  %2222 = extractvalue [2 x <8 x float>] %67, 1, !dbg !33
  %2223 = shufflevector <8 x float> %2222, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %2224 = insertelement <4 x float> poison, float %2174, i64 0, !dbg !33
  %2225 = shufflevector <4 x float> %2224, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !33
  %2226 = fmul <4 x float> %2223, %2225, !dbg !33
  %2227 = insertelement <4 x float> poison, float %2127, i64 0, !dbg !34
  %2228 = insertelement <4 x float> %2227, float %2133, i64 1, !dbg !34
  %2229 = insertelement <4 x float> %2228, float %2128, i64 2, !dbg !34
  %2230 = insertelement <4 x float> %2229, float %2134, i64 3, !dbg !34
  %2231 = fadd <4 x float> %2226, %2230, !dbg !34
  %2232 = shufflevector <8 x float> %2222, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %2233 = fmul <4 x float> %2232, %2225, !dbg !33
  %2234 = insertelement <4 x float> poison, float %2163, i64 0, !dbg !34
  %2235 = insertelement <4 x float> %2234, float %2169, i64 1, !dbg !34
  %2236 = insertelement <4 x float> %2235, float %2164, i64 2, !dbg !34
  %2237 = insertelement <4 x float> %2236, float %2170, i64 3, !dbg !34
  %2238 = fadd <4 x float> %2233, %2237, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !29
  fence syncscope("workgroup") release, !dbg !29
  call void @llvm.amdgcn.s.barrier(), !dbg !29
  fence syncscope("workgroup") acquire, !dbg !29
  %address.local1031 = shl nuw nsw i32 %address.thread993, 3, !dbg !14
  %address.local1032 = and i32 %address.local1031, 504, !dbg !14
  %address.local1033 = zext nneg i32 %address.local1032 to i64, !dbg !14
  %address.local1034 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !14
  %2239 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !14
  %2240 = bitcast <4 x i32> %2239 to <8 x half>, !dbg !14
  %address.local1035 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !14
  %address.local1036 = getelementptr i8, ptr addrspace(3) %address.local1035, i64 1024, !dbg !14
  %2241 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1036, i32 0) #0, !dbg !14
  %2242 = bitcast <4 x i32> %2241 to <8 x half>, !dbg !14
  %address.local1037 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !14
  %address.local1038 = getelementptr i8, ptr addrspace(3) %address.local1037, i64 2048, !dbg !14
  %2243 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1038, i32 0) #0, !dbg !14
  %2244 = bitcast <4 x i32> %2243 to <8 x half>, !dbg !14
  %address.local1039 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !14
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 3072, !dbg !14
  %2245 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !14
  %2246 = bitcast <4 x i32> %2245 to <8 x half>, !dbg !14
  %address.local1041 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !14
  %address.local1042 = getelementptr i8, ptr addrspace(3) %address.local1041, i64 4096, !dbg !14
  %2247 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1042, i32 0) #0, !dbg !14
  %2248 = bitcast <4 x i32> %2247 to <8 x half>, !dbg !14
  %address.local1043 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !14
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1043, i64 5120, !dbg !14
  %2249 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !14
  %2250 = bitcast <4 x i32> %2249 to <8 x half>, !dbg !14
  %address.local1045 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !14
  %address.local1046 = getelementptr i8, ptr addrspace(3) %address.local1045, i64 6144, !dbg !14
  %2251 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1046, i32 0) #0, !dbg !14
  %2252 = bitcast <4 x i32> %2251 to <8 x half>, !dbg !14
  %address.local1047 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1033, !dbg !14
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1047, i64 7168, !dbg !14
  %2253 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !14
  %2254 = bitcast <4 x i32> %2253 to <8 x half>, !dbg !14
  %2255 = shufflevector <8 x half> %2240, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2255, <4 x float> zeroinitializer) #9, !dbg !14
  %2257 = shufflevector <8 x half> %2242, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2257, <4 x float> %2256) #9, !dbg !14
  %2259 = shufflevector <8 x half> %2244, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2259, <4 x float> %2258) #9, !dbg !14
  %2261 = shufflevector <8 x half> %2246, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2261, <4 x float> %2260) #9, !dbg !14
  %2263 = shufflevector <8 x half> %2248, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2263, <4 x float> %2262) #9, !dbg !14
  %2265 = shufflevector <8 x half> %2250, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2265, <4 x float> %2264) #9, !dbg !14
  %2267 = shufflevector <8 x half> %2252, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2267, <4 x float> %2266) #9, !dbg !14
  %2269 = shufflevector <8 x half> %2254, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2269, <4 x float> %2268) #9, !dbg !14
  %2271 = extractelement <4 x float> %2270, i64 0, !dbg !14
  %2272 = extractelement <4 x float> %2270, i64 1, !dbg !14
  %2273 = extractelement <4 x float> %2270, i64 2, !dbg !14
  %2274 = extractelement <4 x float> %2270, i64 3, !dbg !14
  %2275 = select i1 %.not, float %2272, float %2271, !dbg !14
  %2276 = bitcast float %2275 to i32, !dbg !14
  %2277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2276), !dbg !14
  %2278 = bitcast i32 %2277 to float, !dbg !14
  %2279 = select i1 %.not, float %2271, float %2278, !dbg !14
  %2280 = select i1 %.not, float %2278, float %2272, !dbg !14
  %2281 = select i1 %.not, float %2274, float %2273, !dbg !14
  %2282 = bitcast float %2281 to i32, !dbg !14
  %2283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2282), !dbg !14
  %2284 = bitcast i32 %2283 to float, !dbg !14
  %2285 = select i1 %.not, float %2273, float %2284, !dbg !14
  %2286 = select i1 %.not, float %2284, float %2274, !dbg !14
  %2287 = select i1 %.not1, float %2285, float %2279, !dbg !14
  %2288 = bitcast float %2287 to i32, !dbg !14
  %2289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2288), !dbg !14
  %2290 = bitcast i32 %2289 to float, !dbg !14
  %2291 = select i1 %.not1, float %2279, float %2290, !dbg !14
  %2292 = select i1 %.not1, float %2290, float %2285, !dbg !14
  %2293 = select i1 %.not1, float %2286, float %2280, !dbg !14
  %2294 = bitcast float %2293 to i32, !dbg !14
  %2295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2294), !dbg !14
  %2296 = bitcast i32 %2295 to float, !dbg !14
  %2297 = select i1 %.not1, float %2280, float %2296, !dbg !14
  %2298 = select i1 %.not1, float %2296, float %2286, !dbg !14
  %2299 = shufflevector <8 x half> %2240, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2299, <4 x float> zeroinitializer) #9, !dbg !14
  %2301 = shufflevector <8 x half> %2242, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %2301, <4 x float> %2300) #9, !dbg !14
  %2303 = shufflevector <8 x half> %2244, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %2303, <4 x float> %2302) #9, !dbg !14
  %2305 = shufflevector <8 x half> %2246, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %2305, <4 x float> %2304) #9, !dbg !14
  %2307 = shufflevector <8 x half> %2248, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %2307, <4 x float> %2306) #9, !dbg !14
  %2309 = shufflevector <8 x half> %2250, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %2309, <4 x float> %2308) #9, !dbg !14
  %2311 = shufflevector <8 x half> %2252, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %2311, <4 x float> %2310) #9, !dbg !14
  %2313 = shufflevector <8 x half> %2254, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %2313, <4 x float> %2312) #9, !dbg !14
  %2315 = extractelement <4 x float> %2314, i64 0, !dbg !14
  %2316 = extractelement <4 x float> %2314, i64 1, !dbg !14
  %2317 = extractelement <4 x float> %2314, i64 2, !dbg !14
  %2318 = extractelement <4 x float> %2314, i64 3, !dbg !14
  %2319 = select i1 %.not, float %2316, float %2315, !dbg !14
  %2320 = bitcast float %2319 to i32, !dbg !14
  %2321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2320), !dbg !14
  %2322 = bitcast i32 %2321 to float, !dbg !14
  %2323 = select i1 %.not, float %2315, float %2322, !dbg !14
  %2324 = select i1 %.not, float %2322, float %2316, !dbg !14
  %2325 = select i1 %.not, float %2318, float %2317, !dbg !14
  %2326 = bitcast float %2325 to i32, !dbg !14
  %2327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2326), !dbg !14
  %2328 = bitcast i32 %2327 to float, !dbg !14
  %2329 = select i1 %.not, float %2317, float %2328, !dbg !14
  %2330 = select i1 %.not, float %2328, float %2318, !dbg !14
  %2331 = select i1 %.not1, float %2329, float %2323, !dbg !14
  %2332 = bitcast float %2331 to i32, !dbg !14
  %2333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2332), !dbg !14
  %2334 = bitcast i32 %2333 to float, !dbg !14
  %2335 = select i1 %.not1, float %2323, float %2334, !dbg !14
  %2336 = select i1 %.not1, float %2334, float %2329, !dbg !14
  %2337 = select i1 %.not1, float %2330, float %2324, !dbg !14
  %2338 = bitcast float %2337 to i32, !dbg !14
  %2339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2338), !dbg !14
  %2340 = bitcast i32 %2339 to float, !dbg !14
  %2341 = select i1 %.not1, float %2324, float %2340, !dbg !14
  %2342 = select i1 %.not1, float %2340, float %2330, !dbg !14
  %2343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2255, <4 x float> zeroinitializer) #9, !dbg !14
  %2344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2257, <4 x float> %2343) #9, !dbg !14
  %2345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2259, <4 x float> %2344) #9, !dbg !14
  %2346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2261, <4 x float> %2345) #9, !dbg !14
  %2347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2263, <4 x float> %2346) #9, !dbg !14
  %2348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2265, <4 x float> %2347) #9, !dbg !14
  %2349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2267, <4 x float> %2348) #9, !dbg !14
  %2350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2269, <4 x float> %2349) #9, !dbg !14
  %2351 = extractelement <4 x float> %2350, i64 0, !dbg !14
  %2352 = extractelement <4 x float> %2350, i64 1, !dbg !14
  %2353 = extractelement <4 x float> %2350, i64 2, !dbg !14
  %2354 = extractelement <4 x float> %2350, i64 3, !dbg !14
  %2355 = select i1 %.not, float %2352, float %2351, !dbg !14
  %2356 = bitcast float %2355 to i32, !dbg !14
  %2357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2356), !dbg !14
  %2358 = bitcast i32 %2357 to float, !dbg !14
  %2359 = select i1 %.not, float %2351, float %2358, !dbg !14
  %2360 = select i1 %.not, float %2358, float %2352, !dbg !14
  %2361 = select i1 %.not, float %2354, float %2353, !dbg !14
  %2362 = bitcast float %2361 to i32, !dbg !14
  %2363 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2362), !dbg !14
  %2364 = bitcast i32 %2363 to float, !dbg !14
  %2365 = select i1 %.not, float %2353, float %2364, !dbg !14
  %2366 = select i1 %.not, float %2364, float %2354, !dbg !14
  %2367 = select i1 %.not1, float %2365, float %2359, !dbg !14
  %2368 = bitcast float %2367 to i32, !dbg !14
  %2369 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2368), !dbg !14
  %2370 = bitcast i32 %2369 to float, !dbg !14
  %2371 = select i1 %.not1, float %2359, float %2370, !dbg !14
  %2372 = select i1 %.not1, float %2370, float %2365, !dbg !14
  %2373 = select i1 %.not1, float %2366, float %2360, !dbg !14
  %2374 = bitcast float %2373 to i32, !dbg !14
  %2375 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2374), !dbg !14
  %2376 = bitcast i32 %2375 to float, !dbg !14
  %2377 = select i1 %.not1, float %2360, float %2376, !dbg !14
  %2378 = select i1 %.not1, float %2376, float %2366, !dbg !14
  %2379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %2299, <4 x float> zeroinitializer) #9, !dbg !14
  %2380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %2301, <4 x float> %2379) #9, !dbg !14
  %2381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %2303, <4 x float> %2380) #9, !dbg !14
  %2382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %2305, <4 x float> %2381) #9, !dbg !14
  %2383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %2307, <4 x float> %2382) #9, !dbg !14
  %2384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %2309, <4 x float> %2383) #9, !dbg !14
  %2385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %2311, <4 x float> %2384) #9, !dbg !14
  %2386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %2313, <4 x float> %2385) #9, !dbg !14
  %2387 = extractelement <4 x float> %2386, i64 0, !dbg !14
  %2388 = extractelement <4 x float> %2386, i64 1, !dbg !14
  %2389 = extractelement <4 x float> %2386, i64 2, !dbg !14
  %2390 = extractelement <4 x float> %2386, i64 3, !dbg !14
  %2391 = select i1 %.not, float %2388, float %2387, !dbg !14
  %2392 = bitcast float %2391 to i32, !dbg !14
  %2393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2392), !dbg !14
  %2394 = bitcast i32 %2393 to float, !dbg !14
  %2395 = select i1 %.not, float %2387, float %2394, !dbg !14
  %2396 = select i1 %.not, float %2394, float %2388, !dbg !14
  %2397 = select i1 %.not, float %2390, float %2389, !dbg !14
  %2398 = bitcast float %2397 to i32, !dbg !14
  %2399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %2398), !dbg !14
  %2400 = bitcast i32 %2399 to float, !dbg !14
  %2401 = select i1 %.not, float %2389, float %2400, !dbg !14
  %2402 = select i1 %.not, float %2400, float %2390, !dbg !14
  %2403 = select i1 %.not1, float %2401, float %2395, !dbg !14
  %2404 = bitcast float %2403 to i32, !dbg !14
  %2405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2404), !dbg !14
  %2406 = bitcast i32 %2405 to float, !dbg !14
  %2407 = select i1 %.not1, float %2395, float %2406, !dbg !14
  %2408 = select i1 %.not1, float %2406, float %2401, !dbg !14
  %2409 = select i1 %.not1, float %2402, float %2396, !dbg !14
  %2410 = bitcast float %2409 to i32, !dbg !14
  %2411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %2410), !dbg !14
  %2412 = bitcast i32 %2411 to float, !dbg !14
  %2413 = select i1 %.not1, float %2396, float %2412, !dbg !14
  %2414 = select i1 %.not1, float %2412, float %2402, !dbg !14
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  br i1 %43, label %2415, label %2416, !dbg !33

2415:                                             ; preds = %2172
  %address.thread1049 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !33, !range !11, !frisk.register_lifetimes !5
  %address.local1050 = lshr i32 %address.thread1049, 1, !dbg !13
  %address.local1051 = and i32 %address.local1050, 96, !dbg !13
  %address.local1052 = and i32 %address.thread1049, 15, !dbg !13
  %address.local1053 = or disjoint i32 %address.local1051, %address.local1052, !dbg !13
  %address.local1054 = zext nneg i32 %address.local1053 to i64, !dbg !13
  %address.local1055 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !13
  store float %1230, ptr addrspace(3) %address.local1055, align 4, !dbg !33
  %address.local1056 = getelementptr [4 x i8], ptr addrspace(3) @shm_0, i64 %address.local1054, !dbg !13
  %address.local1057 = getelementptr i8, ptr addrspace(3) %address.local1056, i64 64, !dbg !13
  store float %1231, ptr addrspace(3) %address.local1057, align 4, !dbg !33
  br label %2416, !dbg !33

2416:                                             ; preds = %2415, %2172
  fence syncscope("workgroup") release, !dbg !33
  call void @llvm.amdgcn.s.barrier(), !dbg !33
  fence syncscope("workgroup") acquire, !dbg !33
  %lds.const.gep1753 = getelementptr i8, ptr addrspace(3) %address.local804, i64 0, !dbg !33
  %2417 = load float, ptr addrspace(3) %lds.const.gep1753, align 4, !dbg !33
  %lds.const.gep1754 = getelementptr i8, ptr addrspace(3) %address.local804, i64 256, !dbg !33
  %2418 = load float, ptr addrspace(3) %lds.const.gep1754, align 4, !dbg !33
  %2419 = extractvalue [2 x <8 x float>] %68, 0, !dbg !33
  %2420 = shufflevector <8 x float> %2419, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %2421 = insertelement <4 x float> poison, float %2417, i64 0, !dbg !33
  %2422 = shufflevector <4 x float> %2421, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !33
  %2423 = fmul <4 x float> %2420, %2422, !dbg !33
  %2424 = insertelement <4 x float> poison, float %2291, i64 0, !dbg !34
  %2425 = insertelement <4 x float> %2424, float %2297, i64 1, !dbg !34
  %2426 = insertelement <4 x float> %2425, float %2292, i64 2, !dbg !34
  %2427 = insertelement <4 x float> %2426, float %2298, i64 3, !dbg !34
  %2428 = fadd <4 x float> %2423, %2427, !dbg !34
  %2429 = shufflevector <8 x float> %2419, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %2430 = fmul <4 x float> %2429, %2422, !dbg !33
  %2431 = insertelement <4 x float> poison, float %2335, i64 0, !dbg !34
  %2432 = insertelement <4 x float> %2431, float %2341, i64 1, !dbg !34
  %2433 = insertelement <4 x float> %2432, float %2336, i64 2, !dbg !34
  %2434 = insertelement <4 x float> %2433, float %2342, i64 3, !dbg !34
  %2435 = fadd <4 x float> %2430, %2434, !dbg !34
  %2436 = extractvalue [2 x <8 x float>] %68, 1, !dbg !33
  %2437 = shufflevector <8 x float> %2436, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %2438 = insertelement <4 x float> poison, float %2418, i64 0, !dbg !33
  %2439 = shufflevector <4 x float> %2438, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !33
  %2440 = fmul <4 x float> %2437, %2439, !dbg !33
  %2441 = insertelement <4 x float> poison, float %2371, i64 0, !dbg !34
  %2442 = insertelement <4 x float> %2441, float %2377, i64 1, !dbg !34
  %2443 = insertelement <4 x float> %2442, float %2372, i64 2, !dbg !34
  %2444 = insertelement <4 x float> %2443, float %2378, i64 3, !dbg !34
  %2445 = fadd <4 x float> %2440, %2444, !dbg !34
  %2446 = shufflevector <8 x float> %2436, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %2447 = fmul <4 x float> %2446, %2439, !dbg !33
  %2448 = insertelement <4 x float> poison, float %2407, i64 0, !dbg !34
  %2449 = insertelement <4 x float> %2448, float %2413, i64 1, !dbg !34
  %2450 = insertelement <4 x float> %2449, float %2408, i64 2, !dbg !34
  %2451 = insertelement <4 x float> %2450, float %2414, i64 3, !dbg !34
  %2452 = fadd <4 x float> %2447, %2451, !dbg !34
  %2453 = shufflevector <4 x float> %1726, <4 x float> %1733, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %2454 = insertvalue [2 x <8 x float>] poison, <8 x float> %2453, 0, !dbg !18
  %2455 = shufflevector <4 x float> %1743, <4 x float> %1750, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %2456 = insertvalue [2 x <8 x float>] %2454, <8 x float> %2455, 1, !dbg !18
  %2457 = insertelement <1 x float> poison, float %1493, i64 0, !dbg !18
  %2458 = insertvalue [2 x <1 x float>] poison, <1 x float> %2457, 0, !dbg !18
  %2459 = insertelement <1 x float> poison, float %1502, i64 0, !dbg !18
  %2460 = insertvalue [2 x <1 x float>] %2458, <1 x float> %2459, 1, !dbg !18
  %2461 = insertvalue [2 x <1 x float>] poison, <1 x float> %1190, 0, !dbg !18
  %2462 = insertvalue [2 x <1 x float>] %2461, <1 x float> %1203, 1, !dbg !18
  %2463 = shufflevector <4 x float> %1970, <4 x float> %1977, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %2464 = insertvalue [2 x <8 x float>] poison, <8 x float> %2463, 0, !dbg !18
  %2465 = shufflevector <4 x float> %1987, <4 x float> %1994, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %2466 = insertvalue [2 x <8 x float>] %2464, <8 x float> %2465, 1, !dbg !18
  %2467 = shufflevector <4 x float> %2214, <4 x float> %2221, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %2468 = insertvalue [2 x <8 x float>] poison, <8 x float> %2467, 0, !dbg !18
  %2469 = shufflevector <4 x float> %2231, <4 x float> %2238, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %2470 = insertvalue [2 x <8 x float>] %2468, <8 x float> %2469, 1, !dbg !18
  %2471 = shufflevector <4 x float> %2428, <4 x float> %2435, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %2472 = insertvalue [2 x <8 x float>] poison, <8 x float> %2471, 0, !dbg !18
  %2473 = shufflevector <4 x float> %2445, <4 x float> %2452, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %2474 = insertvalue [2 x <8 x float>] %2472, <8 x float> %2473, 1, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %2475 = add nuw nsw i64 %62, 1, !dbg !18
  br label %61, !dbg !18

2476:                                             ; preds = %61
  %2477 = extractvalue [2 x <1 x float>] %64, 0, !dbg !35
  %2478 = shufflevector <1 x float> %2477, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %2479 = extractvalue [2 x <1 x float>] %64, 1, !dbg !35
  %2480 = shufflevector <1 x float> %2479, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %2481 = extractvalue [2 x <8 x float>] %63, 1, !dbg !35
  %2482 = shufflevector <8 x float> %2481, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %2480, !dbg !35
  %normalized = fmul <4 x float> %2482, %row.reciprocal, !dbg !35
  %2483 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !36
  %2484 = extractelement <4 x half> %2483, i64 3, !dbg !36
  %2485 = extractelement <4 x half> %2483, i64 2, !dbg !36
  %2486 = extractelement <4 x half> %2483, i64 1, !dbg !36
  %2487 = extractelement <4 x half> %2483, i64 0, !dbg !36
  %2488 = shufflevector <8 x float> %2481, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %normalized68 = fmul <4 x float> %2488, %row.reciprocal, !dbg !35
  %2489 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !36
  %2490 = extractelement <4 x half> %2489, i64 3, !dbg !36
  %2491 = extractelement <4 x half> %2489, i64 2, !dbg !36
  %2492 = extractelement <4 x half> %2489, i64 1, !dbg !36
  %2493 = extractelement <4 x half> %2489, i64 0, !dbg !36
  %2494 = extractvalue [2 x <8 x float>] %63, 0, !dbg !35
  %2495 = shufflevector <8 x float> %2494, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2478, !dbg !35
  %normalized70 = fmul <4 x float> %2495, %row.reciprocal69, !dbg !35
  %2496 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !36
  %2497 = extractelement <4 x half> %2496, i64 3, !dbg !36
  %2498 = extractelement <4 x half> %2496, i64 2, !dbg !36
  %2499 = extractelement <4 x half> %2496, i64 1, !dbg !36
  %2500 = extractelement <4 x half> %2496, i64 0, !dbg !36
  %2501 = shufflevector <8 x float> %2494, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %normalized71 = fmul <4 x float> %2501, %row.reciprocal69, !dbg !35
  %2502 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !36
  %2503 = extractelement <4 x half> %2502, i64 3, !dbg !36
  %2504 = extractelement <4 x half> %2502, i64 2, !dbg !36
  %2505 = extractelement <4 x half> %2502, i64 1, !dbg !36
  %2506 = extractelement <4 x half> %2502, i64 0, !dbg !36
  %2507 = add nsw i64 %15, %25, !dbg !37
  %2508 = shl i64 %2507, 19, !dbg !37
  %2509 = and i64 %2508, 16252928, !dbg !37
  %address.thread1072 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !37, !range !11, !frisk.register_lifetimes !5
  %address.local1073 = zext nneg i32 %address.thread1072 to i64, !dbg !10
  %address.local1074 = add i64 %28, %address.local1073, !dbg !37
  %address.local1075 = lshr i64 %address.local1073, 2, !dbg !17
  %address.local1076 = and i64 %address.local1075, 48, !dbg !17
  %address.local1077 = add i64 %address.local1074, %address.local1076, !dbg !37
  %address.local1078 = lshr i64 %address.local1073, 4, !dbg !10
  %address.local1079 = mul nsw i64 %address.local1078, -16, !dbg !17
  %address.local1080 = add i64 %address.local1077, %address.local1079, !dbg !37
  %address.local1081 = add i64 %address.local1080, %29, !dbg !37
  %address.local1082 = shl nuw nsw i64 %address.local1081, 7, !dbg !37
  %address.local1083 = add nuw nsw i64 %2509, %address.local1082, !dbg !37
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !37
  %address.local1085 = lshr i64 %address.local1073, 4, !dbg !10
  %address.local1086 = and i64 %address.local1085, 3, !dbg !10
  %address.local1087 = shl nuw nsw i64 %address.local1086, 2, !dbg !10
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1084, i64 %address.local1087, !dbg !37
  store half %2506, ptr addrspace(1) %address.local1088, align 2, !dbg !37
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !37
  %address.local1090 = or disjoint i64 %address.local1087, 1, !dbg !17
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1090, !dbg !37
  store half %2505, ptr addrspace(1) %address.local1091, align 2, !dbg !37
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !37
  %address.local1093 = or disjoint i64 %address.local1087, 2, !dbg !17
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1093, !dbg !37
  store half %2504, ptr addrspace(1) %address.local1094, align 2, !dbg !37
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !37
  %address.local1096 = or disjoint i64 %address.local1087, 3, !dbg !17
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1096, !dbg !37
  store half %2503, ptr addrspace(1) %address.local1097, align 2, !dbg !37
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !37
  %address.local1099 = or disjoint i64 %address.local1087, 16, !dbg !17
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1098, i64 %address.local1099, !dbg !37
  store half %2500, ptr addrspace(1) %address.local1100, align 2, !dbg !37
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !37
  %address.local1102 = or disjoint i64 %address.local1087, 17, !dbg !17
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !37
  store half %2499, ptr addrspace(1) %address.local1103, align 2, !dbg !37
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !37
  %address.local1105 = or disjoint i64 %address.local1087, 18, !dbg !17
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !37
  store half %2498, ptr addrspace(1) %address.local1106, align 2, !dbg !37
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1083, !dbg !37
  %address.local1108 = or disjoint i64 %address.local1087, 19, !dbg !17
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !37
  store half %2497, ptr addrspace(1) %address.local1109, align 2, !dbg !37
  %address.local1110 = shl i64 %address.local1081, 7, !dbg !37
  %address.local1111 = add i64 %address.local1110, 8192, !dbg !37
  %address.local1112 = add nuw nsw i64 %2509, %address.local1111, !dbg !37
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !37
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1087, !dbg !37
  store half %2493, ptr addrspace(1) %address.local1114, align 2, !dbg !37
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !37
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1090, !dbg !37
  store half %2492, ptr addrspace(1) %address.local1116, align 2, !dbg !37
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !37
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1093, !dbg !37
  store half %2491, ptr addrspace(1) %address.local1118, align 2, !dbg !37
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !37
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1096, !dbg !37
  store half %2490, ptr addrspace(1) %address.local1120, align 2, !dbg !37
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !37
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1099, !dbg !37
  store half %2487, ptr addrspace(1) %address.local1122, align 2, !dbg !37
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !37
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1102, !dbg !37
  store half %2486, ptr addrspace(1) %address.local1124, align 2, !dbg !37
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !37
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1105, !dbg !37
  store half %2485, ptr addrspace(1) %address.local1126, align 2, !dbg !37
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1112, !dbg !37
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1108, !dbg !37
  store half %2484, ptr addrspace(1) %address.local1128, align 2, !dbg !37
  %2510 = extractvalue [2 x <8 x float>] %66, 1, !dbg !35
  %2511 = shufflevector <8 x float> %2510, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2480, !dbg !35
  %normalized73 = fmul <4 x float> %2511, %row.reciprocal72, !dbg !35
  %2512 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !36
  %2513 = extractelement <4 x half> %2512, i64 3, !dbg !36
  %2514 = extractelement <4 x half> %2512, i64 2, !dbg !36
  %2515 = extractelement <4 x half> %2512, i64 1, !dbg !36
  %2516 = extractelement <4 x half> %2512, i64 0, !dbg !36
  %2517 = shufflevector <8 x float> %2510, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %normalized74 = fmul <4 x float> %2517, %row.reciprocal72, !dbg !35
  %2518 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !36
  %2519 = extractelement <4 x half> %2518, i64 3, !dbg !36
  %2520 = extractelement <4 x half> %2518, i64 2, !dbg !36
  %2521 = extractelement <4 x half> %2518, i64 1, !dbg !36
  %2522 = extractelement <4 x half> %2518, i64 0, !dbg !36
  %2523 = extractvalue [2 x <8 x float>] %66, 0, !dbg !35
  %2524 = shufflevector <8 x float> %2523, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2478, !dbg !35
  %normalized76 = fmul <4 x float> %2524, %row.reciprocal75, !dbg !35
  %2525 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !36
  %2526 = extractelement <4 x half> %2525, i64 3, !dbg !36
  %2527 = extractelement <4 x half> %2525, i64 2, !dbg !36
  %2528 = extractelement <4 x half> %2525, i64 1, !dbg !36
  %2529 = extractelement <4 x half> %2525, i64 0, !dbg !36
  %2530 = shufflevector <8 x float> %2523, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %normalized77 = fmul <4 x float> %2530, %row.reciprocal75, !dbg !35
  %2531 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !36
  %2532 = extractelement <4 x half> %2531, i64 3, !dbg !36
  %2533 = extractelement <4 x half> %2531, i64 2, !dbg !36
  %2534 = extractelement <4 x half> %2531, i64 1, !dbg !36
  %2535 = extractelement <4 x half> %2531, i64 0, !dbg !36
  %2536 = shl nuw nsw i64 %27, 19, !dbg !37
  %address.thread1129 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !37, !range !11, !frisk.register_lifetimes !5
  %address.local1130 = zext nneg i32 %address.thread1129 to i64, !dbg !10
  %address.local1131 = add i64 %28, %address.local1130, !dbg !37
  %address.local1132 = lshr i64 %address.local1130, 2, !dbg !17
  %address.local1133 = and i64 %address.local1132, 48, !dbg !17
  %address.local1134 = add i64 %address.local1131, %address.local1133, !dbg !37
  %address.local1135 = lshr i64 %address.local1130, 4, !dbg !10
  %address.local1136 = mul nsw i64 %address.local1135, -16, !dbg !17
  %address.local1137 = add i64 %address.local1134, %address.local1136, !dbg !37
  %address.local1138 = add i64 %address.local1137, %29, !dbg !37
  %address.local1139 = shl nuw nsw i64 %address.local1138, 7, !dbg !37
  %address.local1140 = add nuw nsw i64 %2536, %address.local1139, !dbg !37
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !37
  %address.local1142 = lshr i64 %address.local1130, 4, !dbg !10
  %address.local1143 = and i64 %address.local1142, 3, !dbg !10
  %address.local1144 = shl nuw nsw i64 %address.local1143, 2, !dbg !10
  %address.local1145 = or disjoint i64 %address.local1144, 32, !dbg !17
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1145, !dbg !37
  store half %2535, ptr addrspace(1) %address.local1146, align 2, !dbg !37
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !37
  %address.local1148 = or disjoint i64 %address.local1144, 33, !dbg !17
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1148, !dbg !37
  store half %2534, ptr addrspace(1) %address.local1149, align 2, !dbg !37
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !37
  %address.local1151 = or disjoint i64 %address.local1144, 34, !dbg !17
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1150, i64 %address.local1151, !dbg !37
  store half %2533, ptr addrspace(1) %address.local1152, align 2, !dbg !37
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !37
  %address.local1154 = or disjoint i64 %address.local1144, 35, !dbg !17
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1154, !dbg !37
  store half %2532, ptr addrspace(1) %address.local1155, align 2, !dbg !37
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !37
  %address.local1157 = or disjoint i64 %address.local1144, 48, !dbg !17
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1156, i64 %address.local1157, !dbg !37
  store half %2529, ptr addrspace(1) %address.local1158, align 2, !dbg !37
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !37
  %address.local1160 = or disjoint i64 %address.local1144, 49, !dbg !17
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1160, !dbg !37
  store half %2528, ptr addrspace(1) %address.local1161, align 2, !dbg !37
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !37
  %address.local1163 = or disjoint i64 %address.local1144, 50, !dbg !17
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1162, i64 %address.local1163, !dbg !37
  store half %2527, ptr addrspace(1) %address.local1164, align 2, !dbg !37
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !37
  %address.local1166 = or disjoint i64 %address.local1144, 51, !dbg !17
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1166, !dbg !37
  store half %2526, ptr addrspace(1) %address.local1167, align 2, !dbg !37
  %address.local1168 = shl i64 %address.local1138, 7, !dbg !37
  %address.local1169 = add i64 %address.local1168, 8192, !dbg !37
  %address.local1170 = add nuw nsw i64 %2536, %address.local1169, !dbg !37
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !37
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1145, !dbg !37
  store half %2522, ptr addrspace(1) %address.local1172, align 2, !dbg !37
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !37
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1148, !dbg !37
  store half %2521, ptr addrspace(1) %address.local1174, align 2, !dbg !37
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !37
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1151, !dbg !37
  store half %2520, ptr addrspace(1) %address.local1176, align 2, !dbg !37
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !37
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1154, !dbg !37
  store half %2519, ptr addrspace(1) %address.local1178, align 2, !dbg !37
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !37
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1157, !dbg !37
  store half %2516, ptr addrspace(1) %address.local1180, align 2, !dbg !37
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !37
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1160, !dbg !37
  store half %2515, ptr addrspace(1) %address.local1182, align 2, !dbg !37
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !37
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1163, !dbg !37
  store half %2514, ptr addrspace(1) %address.local1184, align 2, !dbg !37
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !37
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1166, !dbg !37
  store half %2513, ptr addrspace(1) %address.local1186, align 2, !dbg !37
  %2537 = extractvalue [2 x <8 x float>] %67, 1, !dbg !35
  %2538 = shufflevector <8 x float> %2537, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2480, !dbg !35
  %normalized79 = fmul <4 x float> %2538, %row.reciprocal78, !dbg !35
  %2539 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !36
  %2540 = extractelement <4 x half> %2539, i64 3, !dbg !36
  %2541 = extractelement <4 x half> %2539, i64 2, !dbg !36
  %2542 = extractelement <4 x half> %2539, i64 1, !dbg !36
  %2543 = extractelement <4 x half> %2539, i64 0, !dbg !36
  %2544 = shufflevector <8 x float> %2537, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %normalized80 = fmul <4 x float> %2544, %row.reciprocal78, !dbg !35
  %2545 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !36
  %2546 = extractelement <4 x half> %2545, i64 3, !dbg !36
  %2547 = extractelement <4 x half> %2545, i64 2, !dbg !36
  %2548 = extractelement <4 x half> %2545, i64 1, !dbg !36
  %2549 = extractelement <4 x half> %2545, i64 0, !dbg !36
  %2550 = extractvalue [2 x <8 x float>] %67, 0, !dbg !35
  %2551 = shufflevector <8 x float> %2550, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2478, !dbg !35
  %normalized82 = fmul <4 x float> %2551, %row.reciprocal81, !dbg !35
  %2552 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !36
  %2553 = extractelement <4 x half> %2552, i64 3, !dbg !36
  %2554 = extractelement <4 x half> %2552, i64 2, !dbg !36
  %2555 = extractelement <4 x half> %2552, i64 1, !dbg !36
  %2556 = extractelement <4 x half> %2552, i64 0, !dbg !36
  %2557 = shufflevector <8 x float> %2550, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %normalized83 = fmul <4 x float> %2557, %row.reciprocal81, !dbg !35
  %2558 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !36
  %2559 = extractelement <4 x half> %2558, i64 3, !dbg !36
  %2560 = extractelement <4 x half> %2558, i64 2, !dbg !36
  %2561 = extractelement <4 x half> %2558, i64 1, !dbg !36
  %2562 = extractelement <4 x half> %2558, i64 0, !dbg !36
  %2563 = shl nuw nsw i64 %27, 19, !dbg !37
  %address.thread1187 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !37, !range !11, !frisk.register_lifetimes !5
  %address.local1188 = zext nneg i32 %address.thread1187 to i64, !dbg !10
  %address.local1189 = add i64 %28, %address.local1188, !dbg !37
  %address.local1190 = lshr i64 %address.local1188, 2, !dbg !17
  %address.local1191 = and i64 %address.local1190, 48, !dbg !17
  %address.local1192 = add i64 %address.local1189, %address.local1191, !dbg !37
  %address.local1193 = lshr i64 %address.local1188, 4, !dbg !10
  %address.local1194 = mul nsw i64 %address.local1193, -16, !dbg !17
  %address.local1195 = add i64 %address.local1192, %address.local1194, !dbg !37
  %address.local1196 = add i64 %address.local1195, %29, !dbg !37
  %address.local1197 = shl nuw nsw i64 %address.local1196, 7, !dbg !37
  %address.local1198 = add nuw nsw i64 %2563, %address.local1197, !dbg !37
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !37
  %address.local1200 = lshr i64 %address.local1188, 4, !dbg !10
  %address.local1201 = and i64 %address.local1200, 3, !dbg !10
  %address.local1202 = shl nuw nsw i64 %address.local1201, 2, !dbg !10
  %address.local1203 = or disjoint i64 %address.local1202, 64, !dbg !17
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1203, !dbg !37
  store half %2562, ptr addrspace(1) %address.local1204, align 2, !dbg !37
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !37
  %address.local1206 = or disjoint i64 %address.local1202, 65, !dbg !17
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1206, !dbg !37
  store half %2561, ptr addrspace(1) %address.local1207, align 2, !dbg !37
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !37
  %address.local1209 = or disjoint i64 %address.local1202, 66, !dbg !17
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1208, i64 %address.local1209, !dbg !37
  store half %2560, ptr addrspace(1) %address.local1210, align 2, !dbg !37
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !37
  %address.local1212 = or disjoint i64 %address.local1202, 67, !dbg !17
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1211, i64 %address.local1212, !dbg !37
  store half %2559, ptr addrspace(1) %address.local1213, align 2, !dbg !37
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !37
  %address.local1215 = or disjoint i64 %address.local1202, 80, !dbg !17
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1214, i64 %address.local1215, !dbg !37
  store half %2556, ptr addrspace(1) %address.local1216, align 2, !dbg !37
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !37
  %address.local1218 = or disjoint i64 %address.local1202, 81, !dbg !17
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1217, i64 %address.local1218, !dbg !37
  store half %2555, ptr addrspace(1) %address.local1219, align 2, !dbg !37
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !37
  %address.local1221 = or disjoint i64 %address.local1202, 82, !dbg !17
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1221, !dbg !37
  store half %2554, ptr addrspace(1) %address.local1222, align 2, !dbg !37
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !37
  %address.local1224 = or disjoint i64 %address.local1202, 83, !dbg !17
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1223, i64 %address.local1224, !dbg !37
  store half %2553, ptr addrspace(1) %address.local1225, align 2, !dbg !37
  %address.local1226 = shl i64 %address.local1196, 7, !dbg !37
  %address.local1227 = add i64 %address.local1226, 8192, !dbg !37
  %address.local1228 = add nuw nsw i64 %2563, %address.local1227, !dbg !37
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !37
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1229, i64 %address.local1203, !dbg !37
  store half %2549, ptr addrspace(1) %address.local1230, align 2, !dbg !37
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !37
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1231, i64 %address.local1206, !dbg !37
  store half %2548, ptr addrspace(1) %address.local1232, align 2, !dbg !37
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !37
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1233, i64 %address.local1209, !dbg !37
  store half %2547, ptr addrspace(1) %address.local1234, align 2, !dbg !37
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !37
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1235, i64 %address.local1212, !dbg !37
  store half %2546, ptr addrspace(1) %address.local1236, align 2, !dbg !37
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !37
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1237, i64 %address.local1215, !dbg !37
  store half %2543, ptr addrspace(1) %address.local1238, align 2, !dbg !37
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !37
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1239, i64 %address.local1218, !dbg !37
  store half %2542, ptr addrspace(1) %address.local1240, align 2, !dbg !37
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !37
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1241, i64 %address.local1221, !dbg !37
  store half %2541, ptr addrspace(1) %address.local1242, align 2, !dbg !37
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1228, !dbg !37
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1243, i64 %address.local1224, !dbg !37
  store half %2540, ptr addrspace(1) %address.local1244, align 2, !dbg !37
  %2564 = extractvalue [2 x <8 x float>] %68, 1, !dbg !35
  %2565 = shufflevector <8 x float> %2564, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2480, !dbg !35
  %normalized85 = fmul <4 x float> %2565, %row.reciprocal84, !dbg !35
  %2566 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !36
  %2567 = extractelement <4 x half> %2566, i64 3, !dbg !36
  %2568 = extractelement <4 x half> %2566, i64 2, !dbg !36
  %2569 = extractelement <4 x half> %2566, i64 1, !dbg !36
  %2570 = extractelement <4 x half> %2566, i64 0, !dbg !36
  %2571 = shufflevector <8 x float> %2564, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %normalized86 = fmul <4 x float> %2571, %row.reciprocal84, !dbg !35
  %2572 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !36
  %2573 = extractelement <4 x half> %2572, i64 3, !dbg !36
  %2574 = extractelement <4 x half> %2572, i64 2, !dbg !36
  %2575 = extractelement <4 x half> %2572, i64 1, !dbg !36
  %2576 = extractelement <4 x half> %2572, i64 0, !dbg !36
  %2577 = extractvalue [2 x <8 x float>] %68, 0, !dbg !35
  %2578 = shufflevector <8 x float> %2577, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2478, !dbg !35
  %normalized88 = fmul <4 x float> %2578, %row.reciprocal87, !dbg !35
  %2579 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !36
  %2580 = extractelement <4 x half> %2579, i64 3, !dbg !36
  %2581 = extractelement <4 x half> %2579, i64 2, !dbg !36
  %2582 = extractelement <4 x half> %2579, i64 1, !dbg !36
  %2583 = extractelement <4 x half> %2579, i64 0, !dbg !36
  %2584 = shufflevector <8 x float> %2577, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %normalized89 = fmul <4 x float> %2584, %row.reciprocal87, !dbg !35
  %2585 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !36
  %2586 = extractelement <4 x half> %2585, i64 3, !dbg !36
  %2587 = extractelement <4 x half> %2585, i64 2, !dbg !36
  %2588 = extractelement <4 x half> %2585, i64 1, !dbg !36
  %2589 = extractelement <4 x half> %2585, i64 0, !dbg !36
  %2590 = shl nuw nsw i64 %27, 19, !dbg !37
  %address.thread1245 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !37, !range !11, !frisk.register_lifetimes !5
  %address.local1246 = zext nneg i32 %address.thread1245 to i64, !dbg !10
  %address.local1247 = add i64 %28, %address.local1246, !dbg !37
  %address.local1248 = lshr i64 %address.local1246, 2, !dbg !17
  %address.local1249 = and i64 %address.local1248, 48, !dbg !17
  %address.local1250 = add i64 %address.local1247, %address.local1249, !dbg !37
  %address.local1251 = lshr i64 %address.local1246, 4, !dbg !10
  %address.local1252 = mul nsw i64 %address.local1251, -16, !dbg !17
  %address.local1253 = add i64 %address.local1250, %address.local1252, !dbg !37
  %address.local1254 = add i64 %address.local1253, %29, !dbg !37
  %address.local1255 = shl nuw nsw i64 %address.local1254, 7, !dbg !37
  %address.local1256 = add nuw nsw i64 %2590, %address.local1255, !dbg !37
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !37
  %address.local1258 = lshr i64 %address.local1246, 4, !dbg !10
  %address.local1259 = and i64 %address.local1258, 3, !dbg !10
  %address.local1260 = shl nuw nsw i64 %address.local1259, 2, !dbg !10
  %address.local1261 = or disjoint i64 %address.local1260, 96, !dbg !17
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1261, !dbg !37
  store half %2589, ptr addrspace(1) %address.local1262, align 2, !dbg !37
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !37
  %address.local1264 = or disjoint i64 %address.local1260, 97, !dbg !17
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1264, !dbg !37
  store half %2588, ptr addrspace(1) %address.local1265, align 2, !dbg !37
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !37
  %address.local1267 = or disjoint i64 %address.local1260, 98, !dbg !17
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1266, i64 %address.local1267, !dbg !37
  store half %2587, ptr addrspace(1) %address.local1268, align 2, !dbg !37
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !37
  %address.local1270 = or disjoint i64 %address.local1260, 99, !dbg !17
  %address.local1271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1269, i64 %address.local1270, !dbg !37
  store half %2586, ptr addrspace(1) %address.local1271, align 2, !dbg !37
  %address.local1272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !37
  %address.local1273 = or disjoint i64 %address.local1260, 112, !dbg !17
  %address.local1274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1272, i64 %address.local1273, !dbg !37
  store half %2583, ptr addrspace(1) %address.local1274, align 2, !dbg !37
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !37
  %address.local1276 = or disjoint i64 %address.local1260, 113, !dbg !17
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1275, i64 %address.local1276, !dbg !37
  store half %2582, ptr addrspace(1) %address.local1277, align 2, !dbg !37
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !37
  %address.local1279 = or disjoint i64 %address.local1260, 114, !dbg !17
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1278, i64 %address.local1279, !dbg !37
  store half %2581, ptr addrspace(1) %address.local1280, align 2, !dbg !37
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1256, !dbg !37
  %address.local1282 = or disjoint i64 %address.local1260, 115, !dbg !17
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1281, i64 %address.local1282, !dbg !37
  store half %2580, ptr addrspace(1) %address.local1283, align 2, !dbg !37
  %address.local1284 = shl i64 %address.local1254, 7, !dbg !37
  %address.local1285 = add i64 %address.local1284, 8192, !dbg !37
  %address.local1286 = add nuw nsw i64 %2590, %address.local1285, !dbg !37
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !37
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1287, i64 %address.local1261, !dbg !37
  store half %2576, ptr addrspace(1) %address.local1288, align 2, !dbg !37
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !37
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1264, !dbg !37
  store half %2575, ptr addrspace(1) %address.local1290, align 2, !dbg !37
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !37
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1291, i64 %address.local1267, !dbg !37
  store half %2574, ptr addrspace(1) %address.local1292, align 2, !dbg !37
  %address.local1293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !37
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1293, i64 %address.local1270, !dbg !37
  store half %2573, ptr addrspace(1) %address.local1294, align 2, !dbg !37
  %address.local1295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !37
  %address.local1296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1295, i64 %address.local1273, !dbg !37
  store half %2570, ptr addrspace(1) %address.local1296, align 2, !dbg !37
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !37
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1297, i64 %address.local1276, !dbg !37
  store half %2569, ptr addrspace(1) %address.local1298, align 2, !dbg !37
  %address.local1299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !37
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1299, i64 %address.local1279, !dbg !37
  store half %2568, ptr addrspace(1) %address.local1300, align 2, !dbg !37
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1286, !dbg !37
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1282, !dbg !37
  store half %2567, ptr addrspace(1) %address.local1302, align 2, !dbg !37
  ret void, !dbg !7
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
!17 = !DILocation(line: 63, column: 17, scope: !3)
!18 = !DILocation(line: 60, column: 24, scope: !3)
!19 = !DILocation(line: 73, column: 26, scope: !3)
!20 = !DILocation(line: 74, column: 23, scope: !3)
!21 = !DILocation(line: 79, column: 24, scope: !3)
!22 = !DILocation(line: 80, column: 23, scope: !3)
!23 = !DILocation(line: 81, column: 26, scope: !3)
!24 = !DILocation(line: 88, column: 21, scope: !3)
!25 = !DILocation(line: 89, column: 20, scope: !3)
!26 = !DILocation(line: 90, column: 22, scope: !3)
!27 = !DILocation(line: 91, column: 23, scope: !3)
!28 = !DILocation(line: 93, column: 22, scope: !3)
!29 = !DILocation(line: 101, column: 20, scope: !3)
!30 = !DILocation(line: 94, column: 22, scope: !3)
!31 = !DILocation(line: 96, column: 16, scope: !3)
!32 = !DILocation(line: 95, column: 20, scope: !3)
!33 = !DILocation(line: 103, column: 22, scope: !3)
!34 = !DILocation(line: 104, column: 20, scope: !3)
!35 = !DILocation(line: 107, column: 23, scope: !3)
!36 = !DILocation(line: 108, column: 19, scope: !3)
!37 = !DILocation(line: 109, column: 9, scope: !3)
