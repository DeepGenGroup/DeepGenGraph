; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/group8-reverse-k128-swizzle/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/group8-reverse-k128-swizzle/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !36 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 6, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423480, !dbg !5
  %13 = and i64 %6, 7, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = lshr i64 %10, 3, !dbg !5
  %16 = sub nsw i64 63, %14, !dbg !5
  %17 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %18 = sext i32 %17 to i64, !dbg !5
  %19 = shl i64 %16, 6, !dbg !9
  %20 = icmp ugt i64 %10, 2047, !dbg !11
  %21 = add nsw i64 %14, -64, !dbg !11
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !11
  %23 = sdiv i64 %22, 64, !dbg !11
  %24 = sext i1 %20 to i64, !dbg !11
  %25 = xor i64 %23, %24, !dbg !11
  %26 = add nsw i64 %15, %25, !dbg !11
  %27 = and i64 %26, 31, !dbg !11
  %28 = shl nsw i64 %16, 6, !dbg !11
  %29 = mul nsw i64 %25, -4096, !dbg !11
  %30 = add i64 %28, %29, !dbg !11
  %31 = shl nuw nsw i64 %27, 19, !dbg !11
  %32 = shl nsw i64 %30, 7, !dbg !11
  %33 = add i64 %31, %32, !dbg !11
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !38
  %35 = zext nneg i32 %34 to i64, !dbg !12
  %36 = lshr i64 %35, 6, !dbg !12
  %37 = or disjoint i64 %19, 63, !dbg !13
  %38 = add i64 %19, 64, !dbg !13
  %39 = icmp uge i64 %38, %37, !dbg !13
  %40 = icmp ugt i64 %10, 2047, !dbg !7
  %41 = select i1 %40, i64 %21, i64 %16, !dbg !7
  %42 = sdiv i64 %41, 2, !dbg !7
  %43 = sub nsw i64 0, %42, !dbg !7
  %44 = add nsw i64 %42, 1, !dbg !7
  %45 = select i1 %40, i64 %43, i64 %44, !dbg !7
  %46 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %47 = lshr i64 %35, 2, !dbg !16
  %48 = shl nuw nsw i64 %35, 2, !dbg !16
  %49 = mul nsw i64 %47, -16, !dbg !16
  %50 = add nsw i64 %48, %49, !dbg !16
  %51 = and i32 %34, 1, !dbg !16
  %index.compact248 = zext nneg i32 %51 to i64, !dbg !16
  %52 = or disjoint i64 %50, %index.compact248, !dbg !16
  %53 = add nuw nsw i64 %52, 1, !dbg !16
  %54 = sdiv i64 %53, 2, !dbg !16
  %55 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %56 = getelementptr i8, ptr addrspace(1) %55, i64 64, !dbg !11
  %57 = shl i64 %15, 7, !dbg !15
  %58 = and i64 %57, 3968, !dbg !15
  %59 = shl nsw i64 %18, 12, !dbg !15
  %60 = or disjoint i64 %58, %59, !dbg !15
  %61 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %62 = getelementptr i8, ptr addrspace(1) %61, i64 128, !dbg !11
  %63 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %64 = getelementptr i8, ptr addrspace(1) %63, i64 192, !dbg !11
  br label %65, !dbg !7

65:                                               ; preds = %592, %4
  %66 = phi i64 [ %1154, %592 ], [ 0, %4 ], !dbg !7
  %67 = phi <4 x float> [ %929, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = phi <4 x float> [ %945, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %69 = phi <8 x float> [ %1150, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %70 = phi <1 x float> [ %863, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %71 = phi <1 x float> [ %694, %592 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %72 = phi <4 x float> [ %1007, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %73 = phi <4 x float> [ %1023, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %74 = phi <8 x float> [ %1151, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %75 = phi <4 x float> [ %1085, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %76 = phi <4 x float> [ %1101, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %77 = phi <8 x float> [ %1152, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %78 = phi <4 x float> [ %1133, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %79 = phi <4 x float> [ %1149, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %80 = phi <8 x float> [ %1153, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %81 = icmp slt i64 %66, %45, !dbg !7
  br i1 %81, label %82, label %1155, !dbg !7

82:                                               ; preds = %65
  %83 = shl nsw i64 %66, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local290 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local291 = and i64 %address.local290, 1536, !dbg !11
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local291, !dbg !11
  %address.local293 = ptrtoint ptr addrspace(3) %address.local292 to i64, !dbg !11
  %address.local294 = trunc i64 %address.local293 to i32, !dbg !11
  %84 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local294), !dbg !11
  %85 = zext i32 %84 to i64, !dbg !11
  %86 = inttoptr i64 %85 to ptr addrspace(3), !dbg !11
  %address.local295 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local296 = and i64 %address.local295, 8064, !dbg !11
  %address.local297 = lshr i64 %address.local, 3, !dbg !11
  %address.local298 = and i64 %address.local297, 24, !dbg !11
  %address.local299 = or disjoint i64 %address.local296, %address.local298, !dbg !11
  %address.local300 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local299, !dbg !11
  %87 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %88 = ptrtoint ptr addrspace(1) %address.local300 to i64, !dbg !11
  %89 = sub i64 %88, %87, !dbg !11
  %dma.buffer.offset1215 = trunc i64 %89 to i32, !dbg !11
  %90 = trunc i64 %87 to i32, !dbg !11
  %91 = insertelement <4 x i32> poison, i32 %90, i64 0, !dbg !11
  %92 = lshr i64 %87, 32, !dbg !11
  %93 = trunc i64 %92 to i32, !dbg !11
  %94 = insertelement <4 x i32> %91, i32 %93, i64 1, !dbg !11
  %95 = insertelement <4 x i32> %94, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1216 = insertelement <4 x i32> %95, i32 131072, i64 3, !dbg !11
  %96 = ptrtoint ptr addrspace(3) %86 to i32, !dbg !11
  %dma.buffer.lds1217 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %96), !dbg !11
  %97 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1217, i32 %dma.buffer.offset1215, <4 x i32> %dma.buffer.rsrc1216) #10, !dbg !11
  %98 = lshr i64 %66, 12, !dbg !15
  %99 = add nuw nsw i64 %98, %15, !dbg !15
  %100 = lshr i64 %66, 5, !dbg !15
  %101 = and i64 %100, 127, !dbg !15
  %102 = shl i64 %99, 19, !dbg !15
  %103 = and i64 %102, 16252928, !dbg !15
  %104 = shl i64 %100, 19, !dbg !15
  %105 = shl i64 %66, 14, !dbg !15
  %106 = sub i64 %105, %104, !dbg !15
  %107 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %103, !dbg !15
  %108 = getelementptr [2 x i8], ptr addrspace(1) %107, i64 %101, !dbg !15
  %109 = getelementptr [2 x i8], ptr addrspace(1) %108, i64 %106, !dbg !15
  %address.local301 = and i64 %address.local290, 1536, !dbg !15
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local301, !dbg !15
  %address.local303 = ptrtoint ptr addrspace(3) %address.local302 to i64, !dbg !15
  %address.local304 = trunc i64 %address.local303 to i32, !dbg !15
  %110 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !15
  %111 = zext i32 %110 to i64, !dbg !15
  %112 = inttoptr i64 %111 to ptr addrspace(3), !dbg !15
  %address.local305 = lshr i64 %address.local, 4, !dbg !15
  %address.local306 = and i64 %address.local305, 8, !dbg !15
  %address.local307 = and i64 %address.local, 123, !dbg !15
  %address.local308 = lshr i64 %address.local, 2, !dbg !15
  %address.local309 = and i64 %address.local308, 4, !dbg !15
  %address.local310 = add nuw nsw i64 %address.local309, %address.local, !dbg !15
  %address.local311 = and i64 %address.local310, 4, !dbg !15
  %address.local312 = or disjoint i64 %address.local307, %address.local311, !dbg !15
  %address.local313 = shl nuw nsw i64 %address.local312, 7, !dbg !15
  %address.local314 = or disjoint i64 %address.local306, %address.local313, !dbg !15
  %address.local315 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %address.local314, !dbg !15
  %113 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %114 = ptrtoint ptr addrspace(1) %address.local315 to i64, !dbg !15
  %115 = sub i64 %114, %113, !dbg !15
  %dma.buffer.offset1212 = trunc i64 %115 to i32, !dbg !15
  %116 = trunc i64 %113 to i32, !dbg !15
  %117 = insertelement <4 x i32> poison, i32 %116, i64 0, !dbg !15
  %118 = lshr i64 %113, 32, !dbg !15
  %119 = trunc i64 %118 to i32, !dbg !15
  %120 = insertelement <4 x i32> %117, i32 %119, i64 1, !dbg !15
  %121 = insertelement <4 x i32> %120, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1213 = insertelement <4 x i32> %121, i32 131072, i64 3, !dbg !15
  %122 = ptrtoint ptr addrspace(3) %112 to i32, !dbg !15
  %dma.buffer.lds1214 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %122), !dbg !15
  %123 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1214, i32 %dma.buffer.offset1212, <4 x i32> %dma.buffer.rsrc1213) #10, !dbg !15
  %address.local316 = or disjoint i64 %address.local, 256, !dbg !15
  %address.local317 = shl nuw nsw i64 %address.local316, 3, !dbg !15
  %address.local318 = and i64 %address.local317, 3584, !dbg !15
  %address.local319 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local318, !dbg !15
  %address.local320 = ptrtoint ptr addrspace(3) %address.local319 to i64, !dbg !15
  %address.local321 = trunc i64 %address.local320 to i32, !dbg !15
  %124 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local321), !dbg !15
  %125 = zext i32 %124 to i64, !dbg !15
  %126 = inttoptr i64 %125 to ptr addrspace(3), !dbg !15
  %address.local322 = lshr i64 %address.local316, 4, !dbg !15
  %address.local323 = and i64 %address.local322, 24, !dbg !15
  %address.local324 = and i64 %address.local, 123, !dbg !15
  %address.local325 = lshr i64 %address.local, 2, !dbg !15
  %address.local326 = and i64 %address.local325, 4, !dbg !15
  %address.local327 = add nuw nsw i64 %address.local326, %address.local, !dbg !15
  %address.local328 = and i64 %address.local327, 4, !dbg !15
  %address.local329 = or disjoint i64 %address.local324, %address.local328, !dbg !15
  %address.local330 = shl nuw nsw i64 %address.local329, 7, !dbg !15
  %address.local331 = or disjoint i64 %address.local323, %address.local330, !dbg !15
  %address.local332 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %address.local331, !dbg !15
  %127 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %128 = ptrtoint ptr addrspace(1) %address.local332 to i64, !dbg !15
  %129 = sub i64 %128, %127, !dbg !15
  %dma.buffer.offset1209 = trunc i64 %129 to i32, !dbg !15
  %130 = trunc i64 %127 to i32, !dbg !15
  %131 = insertelement <4 x i32> poison, i32 %130, i64 0, !dbg !15
  %132 = lshr i64 %127, 32, !dbg !15
  %133 = trunc i64 %132 to i32, !dbg !15
  %134 = insertelement <4 x i32> %131, i32 %133, i64 1, !dbg !15
  %135 = insertelement <4 x i32> %134, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1210 = insertelement <4 x i32> %135, i32 131072, i64 3, !dbg !15
  %136 = ptrtoint ptr addrspace(3) %126 to i32, !dbg !15
  %dma.buffer.lds1211 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %136), !dbg !15
  %137 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1211, i32 %dma.buffer.offset1209, <4 x i32> %dma.buffer.rsrc1210) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local333 = lshr i32 %address.thread, 2, !dbg !12
  %address.local334 = and i32 %address.local333, 4, !dbg !12
  %address.local335 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local336 = and i32 %address.local335, 384, !dbg !12
  %address.local337 = or disjoint i32 %address.local334, %address.local336, !dbg !12
  %address.local338 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local339 = and i32 %address.local338, 120, !dbg !12
  %address.local340 = or disjoint i32 %address.local337, %address.local339, !dbg !12
  %address.local341 = shl nuw nsw i32 %address.thread, 4, !dbg !12
  %address.local342 = and i32 %address.local341, 512, !dbg !12
  %address.local343 = or disjoint i32 %address.local340, %address.local342, !dbg !12
  %address.local344 = zext nneg i32 %address.local343 to i64, !dbg !12
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local344, !dbg !12
  %138 = load <4 x half>, ptr addrspace(3) %address.local345, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local345, i64 2048, !dbg !12
  %139 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %140 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !12
  %141 = fmul <4 x half> %139, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local348 = mul nuw nsw i64 %address.local, 160, !dbg !16
  %address.local349 = lshr i64 %address.local, 2, !dbg !16
  %address.local350 = mul nsw i64 %address.local349, -640, !dbg !16
  %address.local351 = add nsw i64 %address.local348, %address.local350, !dbg !16
  %address.local352 = shl nuw nsw i64 %address.local, 1, !dbg !16
  %address.local353 = and i64 %address.local352, 24, !dbg !16
  %address.local354 = or disjoint i64 %address.local351, %address.local353, !dbg !16
  %address.local355 = shl i64 %address.local, 62, !dbg !16
  %address.local356 = ashr exact i64 %address.local355, 57, !dbg !16
  %address.local357 = and i64 %address.local356, -64, !dbg !16
  %address.local358 = add nsw i64 %address.local354, %address.local357, !dbg !16
  %address.local359 = lshr i64 %address.local, 2, !dbg !12
  %address.local360 = and i64 %address.local359, 12, !dbg !12
  %address.local361 = add nsw i64 %address.local358, %address.local360, !dbg !16
  %address.local362 = and i64 %address.local, 32, !dbg !16
  %address.local363 = icmp eq i64 %address.local362, 0, !dbg !16
  %address.local364 = select i1 %address.local363, i64 0, i64 1016, !dbg !16
  %address.local365 = add i64 %address.local361, %address.local364, !dbg !16
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local365, !dbg !16
  %142 = load <4 x half>, ptr addrspace(3) %address.local366, align 8, !dbg !16
  %lds.const.gep1102 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4096, !dbg !16
  %143 = load <4 x half>, ptr addrspace(3) %lds.const.gep1102, align 8, !dbg !16
  %144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %142, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local377 = shl nuw nsw i32 %address.thread, 1, !dbg !16
  %address.local378 = and i32 %address.local377, 24, !dbg !16
  %address.local379 = shl nuw nsw i32 %address.thread, 8, !dbg !16
  %address.local380 = and i32 %address.local379, 768, !dbg !16
  %address.local381 = or disjoint i32 %address.local378, %address.local380, !dbg !16
  %address.local382 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local383 = and i32 %address.local382, 96, !dbg !16
  %address.local384 = or disjoint i32 %address.local381, %address.local383, !dbg !16
  %address.local385 = zext nneg i32 %address.local384 to i64, !dbg !16
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %54) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local386 = lshr i32 %address.thread, 1, !dbg !16
  %address.local387 = and i32 %address.local386, 1, !dbg !16
  %address.local388 = zext nneg i32 %address.local387 to i64, !dbg !16
  %address.local389 = add nsw i64 %address.base, %address.local388, !dbg !16
  %address.local390 = shl i64 %address.local389, 6, !dbg !16
  %address.local391 = sub i64 %address.local385, %address.local390, !dbg !16
  %address.local392 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local391, !dbg !16
  %address.local393 = getelementptr [2 x i8], ptr addrspace(3) %address.local392, i64 %address.local360, !dbg !16
  %address.local394 = getelementptr [2 x i8], ptr addrspace(3) %address.local393, i64 %address.local364, !dbg !16
  %address.local395 = getelementptr i8, ptr addrspace(3) %address.local394, i64 64, !dbg !16
  %145 = load <4 x half>, ptr addrspace(3) %address.local395, align 8, !dbg !16
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %143, <4 x float> %144) #9, !dbg !16
  %lds.const.gep1103 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4096, !dbg !16
  %147 = load <4 x half>, ptr addrspace(3) %lds.const.gep1103, align 8, !dbg !16
  %lds.const.gep1104 = getelementptr i8, ptr addrspace(3) %address.local366, i64 128, !dbg !16
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1104, align 8, !dbg !16
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %145, <4 x float> zeroinitializer) #9, !dbg !16
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %147, <4 x float> %149) #9, !dbg !16
  %lds.const.gep1105 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4224, !dbg !16
  %151 = load <4 x half>, ptr addrspace(3) %lds.const.gep1105, align 8, !dbg !16
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %148, <4 x float> zeroinitializer) #9, !dbg !16
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %151, <4 x float> %152) #9, !dbg !16
  %lds.const.gep1106 = getelementptr i8, ptr addrspace(3) %address.local395, i64 128, !dbg !16
  %154 = load <4 x half>, ptr addrspace(3) %lds.const.gep1106, align 8, !dbg !16
  %lds.const.gep1107 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4224, !dbg !16
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep1107, align 8, !dbg !16
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep1108 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1024, !dbg !16
  %157 = load <4 x half>, ptr addrspace(3) %lds.const.gep1108, align 8, !dbg !16
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %155, <4 x float> %156) #9, !dbg !16
  %lds.const.gep1109 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5120, !dbg !16
  %159 = load <4 x half>, ptr addrspace(3) %lds.const.gep1109, align 8, !dbg !16
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !16
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %159, <4 x float> %160) #9, !dbg !16
  %lds.const.gep1110 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1024, !dbg !16
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep1110, align 8, !dbg !16
  %lds.const.gep1111 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5120, !dbg !16
  %163 = load <4 x half>, ptr addrspace(3) %lds.const.gep1111, align 8, !dbg !16
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep1112 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1152, !dbg !16
  %165 = load <4 x half>, ptr addrspace(3) %lds.const.gep1112, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %163, <4 x float> %164) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %lds.const.gep1113 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5248, !dbg !16
  %167 = load <4 x half>, ptr addrspace(3) %lds.const.gep1113, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %167, <4 x float> %168) #9, !dbg !16
  %lds.const.gep1114 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1152, !dbg !16
  %170 = load <4 x half>, ptr addrspace(3) %lds.const.gep1114, align 8, !dbg !16
  %lds.const.gep1115 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5248, !dbg !16
  %171 = load <4 x half>, ptr addrspace(3) %lds.const.gep1115, align 8, !dbg !16
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %170, <4 x float> zeroinitializer) #9, !dbg !16
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %171, <4 x float> %172) #9, !dbg !16
  %174 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local294), !dbg !11
  %175 = zext i32 %174 to i64, !dbg !11
  %176 = inttoptr i64 %175 to ptr addrspace(3), !dbg !11
  %address.local704 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local299, !dbg !11
  %177 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %178 = ptrtoint ptr addrspace(1) %address.local704 to i64, !dbg !11
  %179 = sub i64 %178, %177, !dbg !11
  %dma.buffer.offset1206 = trunc i64 %179 to i32, !dbg !11
  %180 = trunc i64 %177 to i32, !dbg !11
  %181 = insertelement <4 x i32> poison, i32 %180, i64 0, !dbg !11
  %182 = lshr i64 %177, 32, !dbg !11
  %183 = trunc i64 %182 to i32, !dbg !11
  %184 = insertelement <4 x i32> %181, i32 %183, i64 1, !dbg !11
  %185 = insertelement <4 x i32> %184, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1207 = insertelement <4 x i32> %185, i32 131072, i64 3, !dbg !11
  %186 = ptrtoint ptr addrspace(3) %176 to i32, !dbg !11
  %dma.buffer.lds1208 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %186), !dbg !11
  %187 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1208, i32 %dma.buffer.offset1206, <4 x i32> %dma.buffer.rsrc1207) #10, !dbg !11
  %188 = add nuw i64 %66, 1024, !dbg !15
  %189 = icmp slt i64 %188, 0, !dbg !15
  %190 = sub nuw i64 -1025, %66, !dbg !15
  %191 = select i1 %189, i64 %190, i64 %188, !dbg !15
  %192 = sdiv i64 %191, 4096, !dbg !15
  %.lobit37 = ashr i64 %188, 63, !dbg !15
  %193 = xor i64 %192, %.lobit37, !dbg !15
  %194 = add nsw i64 %193, %15, !dbg !15
  %195 = add nsw i64 %100, %60, !dbg !15
  %196 = add nsw i64 %195, 32, !dbg !15
  %197 = icmp slt i64 %195, -32, !dbg !15
  %198 = sub nsw i64 -33, %195, !dbg !15
  %199 = select i1 %197, i64 %198, i64 %196, !dbg !15
  %200 = sdiv i64 %199, 128, !dbg !15
  %201 = sext i1 %197 to i64, !dbg !15
  %202 = xor i64 %200, %201, !dbg !15
  %203 = shl nsw i64 %202, 7, !dbg !15
  %204 = sub i64 %60, %203, !dbg !15
  %205 = shl i64 %194, 19, !dbg !15
  %206 = and i64 %205, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %207 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %206, !dbg !15
  %208 = getelementptr [2 x i8], ptr addrspace(1) %207, i64 %100, !dbg !15
  %209 = getelementptr [2 x i8], ptr addrspace(1) %208, i64 %204, !dbg !15
  %210 = getelementptr i8, ptr addrspace(1) %209, i64 64, !dbg !15
  %211 = getelementptr [2 x i8], ptr addrspace(1) %210, i64 %106, !dbg !15
  %212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !15
  %213 = zext i32 %212 to i64, !dbg !15
  %214 = inttoptr i64 %213 to ptr addrspace(3), !dbg !15
  %address.local705 = getelementptr [2 x i8], ptr addrspace(1) %211, i64 %address.local314, !dbg !15
  %215 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %216 = ptrtoint ptr addrspace(1) %address.local705 to i64, !dbg !15
  %217 = sub i64 %216, %215, !dbg !15
  %dma.buffer.offset1203 = trunc i64 %217 to i32, !dbg !15
  %218 = trunc i64 %215 to i32, !dbg !15
  %219 = insertelement <4 x i32> poison, i32 %218, i64 0, !dbg !15
  %220 = lshr i64 %215, 32, !dbg !15
  %221 = trunc i64 %220 to i32, !dbg !15
  %222 = insertelement <4 x i32> %219, i32 %221, i64 1, !dbg !15
  %223 = insertelement <4 x i32> %222, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1204 = insertelement <4 x i32> %223, i32 131072, i64 3, !dbg !15
  %224 = ptrtoint ptr addrspace(3) %214 to i32, !dbg !15
  %dma.buffer.lds1205 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %224), !dbg !15
  %225 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1205, i32 %dma.buffer.offset1203, <4 x i32> %dma.buffer.rsrc1204) #10, !dbg !15
  %226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local321), !dbg !15
  %227 = zext i32 %226 to i64, !dbg !15
  %228 = inttoptr i64 %227 to ptr addrspace(3), !dbg !15
  %address.local706 = getelementptr [2 x i8], ptr addrspace(1) %211, i64 %address.local331, !dbg !15
  %229 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %230 = ptrtoint ptr addrspace(1) %address.local706 to i64, !dbg !15
  %231 = sub i64 %230, %229, !dbg !15
  %dma.buffer.offset1200 = trunc i64 %231 to i32, !dbg !15
  %232 = trunc i64 %229 to i32, !dbg !15
  %233 = insertelement <4 x i32> poison, i32 %232, i64 0, !dbg !15
  %234 = lshr i64 %229, 32, !dbg !15
  %235 = trunc i64 %234 to i32, !dbg !15
  %236 = insertelement <4 x i32> %233, i32 %235, i64 1, !dbg !15
  %237 = insertelement <4 x i32> %236, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1201 = insertelement <4 x i32> %237, i32 131072, i64 3, !dbg !15
  %238 = ptrtoint ptr addrspace(3) %228 to i32, !dbg !15
  %dma.buffer.lds1202 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %238), !dbg !15
  %239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1202, i32 %dma.buffer.offset1200, <4 x i32> %dma.buffer.rsrc1201) #10, !dbg !15
  %240 = load <4 x half>, ptr addrspace(3) %address.local345, align 8, !dbg !12
  %lds.const.gep1116 = getelementptr i8, ptr addrspace(3) %address.local345, i64 2048, !dbg !12
  %241 = load <4 x half>, ptr addrspace(3) %lds.const.gep1116, align 8, !dbg !12
  %242 = fmul <4 x half> %240, splat (half 0xH3015), !dbg !12
  %243 = fmul <4 x half> %241, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %244 = load <4 x half>, ptr addrspace(3) %address.local366, align 8, !dbg !16
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %244, <4 x float> %146) #9, !dbg !16
  %lds.const.gep1117 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4096, !dbg !16
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1117, align 8, !dbg !16
  %247 = load <4 x half>, ptr addrspace(3) %address.local395, align 8, !dbg !16
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %246, <4 x float> %245) #9, !dbg !16
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %247, <4 x float> %150) #9, !dbg !16
  %lds.const.gep1118 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4096, !dbg !16
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1118, align 8, !dbg !16
  %lds.const.gep1119 = getelementptr i8, ptr addrspace(3) %address.local366, i64 128, !dbg !16
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1119, align 8, !dbg !16
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %250, <4 x float> %249) #9, !dbg !16
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %251, <4 x float> %153) #9, !dbg !16
  %lds.const.gep1120 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4224, !dbg !16
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1120, align 8, !dbg !16
  %lds.const.gep1121 = getelementptr i8, ptr addrspace(3) %address.local395, i64 128, !dbg !16
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1121, align 8, !dbg !16
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %254, <4 x float> %253) #9, !dbg !16
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %255, <4 x float> %158) #9, !dbg !16
  %lds.const.gep1122 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4224, !dbg !16
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1122, align 8, !dbg !16
  %lds.const.gep1123 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1024, !dbg !16
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep1123, align 8, !dbg !16
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %258, <4 x float> %257) #9, !dbg !16
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %259, <4 x float> %161) #9, !dbg !16
  %lds.const.gep1124 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5120, !dbg !16
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1124, align 8, !dbg !16
  %lds.const.gep1125 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1024, !dbg !16
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep1125, align 8, !dbg !16
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %262, <4 x float> %261) #9, !dbg !16
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %263, <4 x float> %166) #9, !dbg !16
  %lds.const.gep1126 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5120, !dbg !16
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1126, align 8, !dbg !16
  %lds.const.gep1127 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1152, !dbg !16
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1127, align 8, !dbg !16
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %266, <4 x float> %265) #9, !dbg !16
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %267, <4 x float> %169) #9, !dbg !16
  %lds.const.gep1128 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5248, !dbg !16
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1128, align 8, !dbg !16
  %lds.const.gep1129 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1152, !dbg !16
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep1129, align 8, !dbg !16
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %270, <4 x float> %269) #9, !dbg !16
  %lds.const.gep1130 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5248, !dbg !16
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep1130, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %271, <4 x float> %173) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %273, <4 x float> %274) #9, !dbg !16
  %276 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local294), !dbg !11
  %277 = zext i32 %276 to i64, !dbg !11
  %278 = inttoptr i64 %277 to ptr addrspace(3), !dbg !11
  %address.local707 = getelementptr [2 x i8], ptr addrspace(1) %62, i64 %address.local299, !dbg !11
  %279 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %280 = ptrtoint ptr addrspace(1) %address.local707 to i64, !dbg !11
  %281 = sub i64 %280, %279, !dbg !11
  %dma.buffer.offset1197 = trunc i64 %281 to i32, !dbg !11
  %282 = trunc i64 %279 to i32, !dbg !11
  %283 = insertelement <4 x i32> poison, i32 %282, i64 0, !dbg !11
  %284 = lshr i64 %279, 32, !dbg !11
  %285 = trunc i64 %284 to i32, !dbg !11
  %286 = insertelement <4 x i32> %283, i32 %285, i64 1, !dbg !11
  %287 = insertelement <4 x i32> %286, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1198 = insertelement <4 x i32> %287, i32 131072, i64 3, !dbg !11
  %288 = ptrtoint ptr addrspace(3) %278 to i32, !dbg !11
  %dma.buffer.lds1199 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %288), !dbg !11
  %289 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1199, i32 %dma.buffer.offset1197, <4 x i32> %dma.buffer.rsrc1198) #10, !dbg !11
  %290 = add nuw i64 %66, 2048, !dbg !15
  %291 = icmp slt i64 %290, 0, !dbg !15
  %292 = sub nuw i64 -2049, %66, !dbg !15
  %293 = select i1 %291, i64 %292, i64 %290, !dbg !15
  %294 = sdiv i64 %293, 4096, !dbg !15
  %.lobit38 = ashr i64 %290, 63, !dbg !15
  %295 = xor i64 %294, %.lobit38, !dbg !15
  %296 = add nsw i64 %295, %15, !dbg !15
  %297 = add nsw i64 %195, 64, !dbg !15
  %298 = icmp slt i64 %195, -64, !dbg !15
  %299 = sub nsw i64 -65, %195, !dbg !15
  %300 = select i1 %298, i64 %299, i64 %297, !dbg !15
  %301 = sdiv i64 %300, 128, !dbg !15
  %302 = sext i1 %298 to i64, !dbg !15
  %303 = xor i64 %301, %302, !dbg !15
  %304 = shl nsw i64 %303, 7, !dbg !15
  %305 = sub i64 %60, %304, !dbg !15
  %306 = shl i64 %296, 19, !dbg !15
  %307 = and i64 %306, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %308 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %307, !dbg !15
  %309 = getelementptr [2 x i8], ptr addrspace(1) %308, i64 %100, !dbg !15
  %310 = getelementptr [2 x i8], ptr addrspace(1) %309, i64 %305, !dbg !15
  %311 = getelementptr i8, ptr addrspace(1) %310, i64 128, !dbg !15
  %312 = getelementptr [2 x i8], ptr addrspace(1) %311, i64 %106, !dbg !15
  %313 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !15
  %314 = zext i32 %313 to i64, !dbg !15
  %315 = inttoptr i64 %314 to ptr addrspace(3), !dbg !15
  %address.local708 = getelementptr [2 x i8], ptr addrspace(1) %312, i64 %address.local314, !dbg !15
  %316 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %317 = ptrtoint ptr addrspace(1) %address.local708 to i64, !dbg !15
  %318 = sub i64 %317, %316, !dbg !15
  %dma.buffer.offset1194 = trunc i64 %318 to i32, !dbg !15
  %319 = trunc i64 %316 to i32, !dbg !15
  %320 = insertelement <4 x i32> poison, i32 %319, i64 0, !dbg !15
  %321 = lshr i64 %316, 32, !dbg !15
  %322 = trunc i64 %321 to i32, !dbg !15
  %323 = insertelement <4 x i32> %320, i32 %322, i64 1, !dbg !15
  %324 = insertelement <4 x i32> %323, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1195 = insertelement <4 x i32> %324, i32 131072, i64 3, !dbg !15
  %325 = ptrtoint ptr addrspace(3) %315 to i32, !dbg !15
  %dma.buffer.lds1196 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %325), !dbg !15
  %326 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1196, i32 %dma.buffer.offset1194, <4 x i32> %dma.buffer.rsrc1195) #10, !dbg !15
  %327 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local321), !dbg !15
  %328 = zext i32 %327 to i64, !dbg !15
  %329 = inttoptr i64 %328 to ptr addrspace(3), !dbg !15
  %address.local709 = getelementptr [2 x i8], ptr addrspace(1) %312, i64 %address.local331, !dbg !15
  %330 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %331 = ptrtoint ptr addrspace(1) %address.local709 to i64, !dbg !15
  %332 = sub i64 %331, %330, !dbg !15
  %dma.buffer.offset1191 = trunc i64 %332 to i32, !dbg !15
  %333 = trunc i64 %330 to i32, !dbg !15
  %334 = insertelement <4 x i32> poison, i32 %333, i64 0, !dbg !15
  %335 = lshr i64 %330, 32, !dbg !15
  %336 = trunc i64 %335 to i32, !dbg !15
  %337 = insertelement <4 x i32> %334, i32 %336, i64 1, !dbg !15
  %338 = insertelement <4 x i32> %337, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1192 = insertelement <4 x i32> %338, i32 131072, i64 3, !dbg !15
  %339 = ptrtoint ptr addrspace(3) %329 to i32, !dbg !15
  %dma.buffer.lds1193 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %339), !dbg !15
  %340 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1193, i32 %dma.buffer.offset1191, <4 x i32> %dma.buffer.rsrc1192) #10, !dbg !15
  %341 = load <4 x half>, ptr addrspace(3) %address.local345, align 8, !dbg !12
  %lds.const.gep1131 = getelementptr i8, ptr addrspace(3) %address.local345, i64 2048, !dbg !12
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1131, align 8, !dbg !12
  %343 = fmul <4 x half> %341, splat (half 0xH3015), !dbg !12
  %344 = fmul <4 x half> %342, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %345 = load <4 x half>, ptr addrspace(3) %address.local366, align 8, !dbg !16
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %345, <4 x float> %248) #9, !dbg !16
  %lds.const.gep1132 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4096, !dbg !16
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1132, align 8, !dbg !16
  %348 = load <4 x half>, ptr addrspace(3) %address.local395, align 8, !dbg !16
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %347, <4 x float> %346) #9, !dbg !16
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %348, <4 x float> %252) #9, !dbg !16
  %lds.const.gep1133 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4096, !dbg !16
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1133, align 8, !dbg !16
  %lds.const.gep1134 = getelementptr i8, ptr addrspace(3) %address.local366, i64 128, !dbg !16
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1134, align 8, !dbg !16
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %351, <4 x float> %350) #9, !dbg !16
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %352, <4 x float> %256) #9, !dbg !16
  %lds.const.gep1135 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4224, !dbg !16
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1135, align 8, !dbg !16
  %lds.const.gep1136 = getelementptr i8, ptr addrspace(3) %address.local395, i64 128, !dbg !16
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1136, align 8, !dbg !16
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %355, <4 x float> %354) #9, !dbg !16
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %356, <4 x float> %260) #9, !dbg !16
  %lds.const.gep1137 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4224, !dbg !16
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1137, align 8, !dbg !16
  %lds.const.gep1138 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1024, !dbg !16
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1138, align 8, !dbg !16
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %359, <4 x float> %358) #9, !dbg !16
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %360, <4 x float> %264) #9, !dbg !16
  %lds.const.gep1139 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5120, !dbg !16
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1139, align 8, !dbg !16
  %lds.const.gep1140 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1024, !dbg !16
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1140, align 8, !dbg !16
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %363, <4 x float> %362) #9, !dbg !16
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %364, <4 x float> %268) #9, !dbg !16
  %lds.const.gep1141 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5120, !dbg !16
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1141, align 8, !dbg !16
  %lds.const.gep1142 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1152, !dbg !16
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1142, align 8, !dbg !16
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %367, <4 x float> %366) #9, !dbg !16
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %368, <4 x float> %272) #9, !dbg !16
  %lds.const.gep1143 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5248, !dbg !16
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1143, align 8, !dbg !16
  %lds.const.gep1144 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1152, !dbg !16
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep1144, align 8, !dbg !16
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %371, <4 x float> %370) #9, !dbg !16
  %lds.const.gep1145 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5248, !dbg !16
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep1145, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %372, <4 x float> %275) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %374, <4 x float> %375) #9, !dbg !16
  %377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local294), !dbg !11
  %378 = zext i32 %377 to i64, !dbg !11
  %379 = inttoptr i64 %378 to ptr addrspace(3), !dbg !11
  %address.local710 = getelementptr [2 x i8], ptr addrspace(1) %64, i64 %address.local299, !dbg !11
  %380 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %381 = ptrtoint ptr addrspace(1) %address.local710 to i64, !dbg !11
  %382 = sub i64 %381, %380, !dbg !11
  %dma.buffer.offset1188 = trunc i64 %382 to i32, !dbg !11
  %383 = trunc i64 %380 to i32, !dbg !11
  %384 = insertelement <4 x i32> poison, i32 %383, i64 0, !dbg !11
  %385 = lshr i64 %380, 32, !dbg !11
  %386 = trunc i64 %385 to i32, !dbg !11
  %387 = insertelement <4 x i32> %384, i32 %386, i64 1, !dbg !11
  %388 = insertelement <4 x i32> %387, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1189 = insertelement <4 x i32> %388, i32 131072, i64 3, !dbg !11
  %389 = ptrtoint ptr addrspace(3) %379 to i32, !dbg !11
  %dma.buffer.lds1190 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %389), !dbg !11
  %390 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1190, i32 %dma.buffer.offset1188, <4 x i32> %dma.buffer.rsrc1189) #10, !dbg !11
  %391 = add nuw i64 %66, 3072, !dbg !15
  %392 = icmp slt i64 %391, 0, !dbg !15
  %393 = sub nuw i64 -3073, %66, !dbg !15
  %394 = select i1 %392, i64 %393, i64 %391, !dbg !15
  %395 = sdiv i64 %394, 4096, !dbg !15
  %.lobit39 = ashr i64 %391, 63, !dbg !15
  %396 = xor i64 %395, %.lobit39, !dbg !15
  %397 = add nsw i64 %396, %15, !dbg !15
  %398 = add nsw i64 %195, 96, !dbg !15
  %399 = icmp slt i64 %195, -96, !dbg !15
  %400 = sub nsw i64 -97, %195, !dbg !15
  %401 = select i1 %399, i64 %400, i64 %398, !dbg !15
  %402 = sdiv i64 %401, 128, !dbg !15
  %403 = sext i1 %399 to i64, !dbg !15
  %404 = xor i64 %402, %403, !dbg !15
  %405 = shl nsw i64 %404, 7, !dbg !15
  %406 = sub i64 %60, %405, !dbg !15
  %407 = shl i64 %397, 19, !dbg !15
  %408 = and i64 %407, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %409 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %408, !dbg !15
  %410 = getelementptr [2 x i8], ptr addrspace(1) %409, i64 %100, !dbg !15
  %411 = getelementptr [2 x i8], ptr addrspace(1) %410, i64 %406, !dbg !15
  %412 = getelementptr i8, ptr addrspace(1) %411, i64 192, !dbg !15
  %413 = getelementptr [2 x i8], ptr addrspace(1) %412, i64 %106, !dbg !15
  %414 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !15
  %415 = zext i32 %414 to i64, !dbg !15
  %416 = inttoptr i64 %415 to ptr addrspace(3), !dbg !15
  %address.local711 = getelementptr [2 x i8], ptr addrspace(1) %413, i64 %address.local314, !dbg !15
  %417 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %418 = ptrtoint ptr addrspace(1) %address.local711 to i64, !dbg !15
  %419 = sub i64 %418, %417, !dbg !15
  %dma.buffer.offset1185 = trunc i64 %419 to i32, !dbg !15
  %420 = trunc i64 %417 to i32, !dbg !15
  %421 = insertelement <4 x i32> poison, i32 %420, i64 0, !dbg !15
  %422 = lshr i64 %417, 32, !dbg !15
  %423 = trunc i64 %422 to i32, !dbg !15
  %424 = insertelement <4 x i32> %421, i32 %423, i64 1, !dbg !15
  %425 = insertelement <4 x i32> %424, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1186 = insertelement <4 x i32> %425, i32 131072, i64 3, !dbg !15
  %426 = ptrtoint ptr addrspace(3) %416 to i32, !dbg !15
  %dma.buffer.lds1187 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %426), !dbg !15
  %427 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1187, i32 %dma.buffer.offset1185, <4 x i32> %dma.buffer.rsrc1186) #10, !dbg !15
  %428 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local321), !dbg !15
  %429 = zext i32 %428 to i64, !dbg !15
  %430 = inttoptr i64 %429 to ptr addrspace(3), !dbg !15
  %address.local712 = getelementptr [2 x i8], ptr addrspace(1) %413, i64 %address.local331, !dbg !15
  %431 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %432 = ptrtoint ptr addrspace(1) %address.local712 to i64, !dbg !15
  %433 = sub i64 %432, %431, !dbg !15
  %dma.buffer.offset1182 = trunc i64 %433 to i32, !dbg !15
  %434 = trunc i64 %431 to i32, !dbg !15
  %435 = insertelement <4 x i32> poison, i32 %434, i64 0, !dbg !15
  %436 = lshr i64 %431, 32, !dbg !15
  %437 = trunc i64 %436 to i32, !dbg !15
  %438 = insertelement <4 x i32> %435, i32 %437, i64 1, !dbg !15
  %439 = insertelement <4 x i32> %438, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1183 = insertelement <4 x i32> %439, i32 131072, i64 3, !dbg !15
  %440 = ptrtoint ptr addrspace(3) %430 to i32, !dbg !15
  %dma.buffer.lds1184 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %440), !dbg !15
  %441 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1184, i32 %dma.buffer.offset1182, <4 x i32> %dma.buffer.rsrc1183) #10, !dbg !15
  %442 = load <4 x half>, ptr addrspace(3) %address.local345, align 8, !dbg !12
  %lds.const.gep1146 = getelementptr i8, ptr addrspace(3) %address.local345, i64 2048, !dbg !12
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1146, align 8, !dbg !12
  %444 = fmul <4 x half> %442, splat (half 0xH3015), !dbg !12
  %445 = fmul <4 x half> %443, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %446 = add nuw nsw i64 %100, %15, !dbg !17
  %447 = shl i64 %446, 19, !dbg !17
  %448 = and i64 %447, 16252928, !dbg !17
  %449 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %448, !dbg !17
  %450 = getelementptr [2 x i8], ptr addrspace(1) %449, i64 %106, !dbg !17
  %address.local713 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local301, !dbg !17
  %address.local714 = ptrtoint ptr addrspace(3) %address.local713 to i64, !dbg !17
  %address.local715 = trunc i64 %address.local714 to i32, !dbg !17
  %451 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local715), !dbg !17
  %452 = zext i32 %451 to i64, !dbg !17
  %453 = inttoptr i64 %452 to ptr addrspace(3), !dbg !17
  %address.local716 = shl nuw nsw i64 %address.local, 5, !dbg !17
  %address.local717 = and i64 %address.local716, 8064, !dbg !17
  %address.local718 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local719 = and i64 %address.local718, 24, !dbg !17
  %address.local720 = or disjoint i64 %address.local717, %address.local719, !dbg !17
  %address.local721 = getelementptr [2 x i8], ptr addrspace(1) %450, i64 %address.local720, !dbg !17
  %454 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %455 = ptrtoint ptr addrspace(1) %address.local721 to i64, !dbg !17
  %456 = sub i64 %455, %454, !dbg !17
  %dma.buffer.offset1179 = trunc i64 %456 to i32, !dbg !17
  %457 = trunc i64 %454 to i32, !dbg !17
  %458 = insertelement <4 x i32> poison, i32 %457, i64 0, !dbg !17
  %459 = lshr i64 %454, 32, !dbg !17
  %460 = trunc i64 %459 to i32, !dbg !17
  %461 = insertelement <4 x i32> %458, i32 %460, i64 1, !dbg !17
  %462 = insertelement <4 x i32> %461, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1180 = insertelement <4 x i32> %462, i32 131072, i64 3, !dbg !17
  %463 = ptrtoint ptr addrspace(3) %453 to i32, !dbg !17
  %dma.buffer.lds1181 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %463), !dbg !17
  %464 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1181, i32 %dma.buffer.offset1179, <4 x i32> %dma.buffer.rsrc1180) #10, !dbg !17
  %address.local722 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local318, !dbg !17
  %address.local723 = ptrtoint ptr addrspace(3) %address.local722 to i64, !dbg !17
  %address.local724 = trunc i64 %address.local723 to i32, !dbg !17
  %465 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local724), !dbg !17
  %466 = zext i32 %465 to i64, !dbg !17
  %467 = inttoptr i64 %466 to ptr addrspace(3), !dbg !17
  %address.local725 = shl nuw nsw i64 %address.local316, 5, !dbg !17
  %address.local726 = and i64 %address.local725, 16256, !dbg !17
  %address.local727 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local728 = and i64 %address.local727, 24, !dbg !17
  %address.local729 = or disjoint i64 %address.local726, %address.local728, !dbg !17
  %address.local730 = getelementptr [2 x i8], ptr addrspace(1) %450, i64 %address.local729, !dbg !17
  %468 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %469 = ptrtoint ptr addrspace(1) %address.local730 to i64, !dbg !17
  %470 = sub i64 %469, %468, !dbg !17
  %dma.buffer.offset1176 = trunc i64 %470 to i32, !dbg !17
  %471 = trunc i64 %468 to i32, !dbg !17
  %472 = insertelement <4 x i32> poison, i32 %471, i64 0, !dbg !17
  %473 = lshr i64 %468, 32, !dbg !17
  %474 = trunc i64 %473 to i32, !dbg !17
  %475 = insertelement <4 x i32> %472, i32 %474, i64 1, !dbg !17
  %476 = insertelement <4 x i32> %475, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1177 = insertelement <4 x i32> %476, i32 131072, i64 3, !dbg !17
  %477 = ptrtoint ptr addrspace(3) %467 to i32, !dbg !17
  %dma.buffer.lds1178 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %477), !dbg !17
  %478 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1178, i32 %dma.buffer.offset1176, <4 x i32> %dma.buffer.rsrc1177) #10, !dbg !17
  %479 = load <4 x half>, ptr addrspace(3) %address.local366, align 8, !dbg !16
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %479, <4 x float> %349) #9, !dbg !16
  %lds.const.gep1147 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4096, !dbg !16
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1147, align 8, !dbg !16
  %482 = load <4 x half>, ptr addrspace(3) %address.local395, align 8, !dbg !16
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %481, <4 x float> %480) #9, !dbg !16
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %482, <4 x float> %353) #9, !dbg !16
  %lds.const.gep1148 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4096, !dbg !16
  %485 = load <4 x half>, ptr addrspace(3) %lds.const.gep1148, align 8, !dbg !16
  %lds.const.gep1149 = getelementptr i8, ptr addrspace(3) %address.local366, i64 128, !dbg !16
  %486 = load <4 x half>, ptr addrspace(3) %lds.const.gep1149, align 8, !dbg !16
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %485, <4 x float> %484) #9, !dbg !16
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %486, <4 x float> %357) #9, !dbg !16
  %lds.const.gep1150 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4224, !dbg !16
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1150, align 8, !dbg !16
  %lds.const.gep1151 = getelementptr i8, ptr addrspace(3) %address.local395, i64 128, !dbg !16
  %490 = load <4 x half>, ptr addrspace(3) %lds.const.gep1151, align 8, !dbg !16
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %489, <4 x float> %488) #9, !dbg !16
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %490, <4 x float> %361) #9, !dbg !16
  %lds.const.gep1152 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4224, !dbg !16
  %493 = load <4 x half>, ptr addrspace(3) %lds.const.gep1152, align 8, !dbg !16
  %lds.const.gep1153 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1024, !dbg !16
  %494 = load <4 x half>, ptr addrspace(3) %lds.const.gep1153, align 8, !dbg !16
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %493, <4 x float> %492) #9, !dbg !16
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %494, <4 x float> %365) #9, !dbg !16
  %lds.const.gep1154 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5120, !dbg !16
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep1154, align 8, !dbg !16
  %lds.const.gep1155 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1024, !dbg !16
  %498 = load <4 x half>, ptr addrspace(3) %lds.const.gep1155, align 8, !dbg !16
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %497, <4 x float> %496) #9, !dbg !16
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %498, <4 x float> %369) #9, !dbg !16
  %lds.const.gep1156 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5120, !dbg !16
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1156, align 8, !dbg !16
  %lds.const.gep1157 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1152, !dbg !16
  %502 = load <4 x half>, ptr addrspace(3) %lds.const.gep1157, align 8, !dbg !16
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %501, <4 x float> %500) #9, !dbg !16
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %502, <4 x float> %373) #9, !dbg !16
  %lds.const.gep1158 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5248, !dbg !16
  %505 = load <4 x half>, ptr addrspace(3) %lds.const.gep1158, align 8, !dbg !16
  %lds.const.gep1159 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1152, !dbg !16
  %506 = load <4 x half>, ptr addrspace(3) %lds.const.gep1159, align 8, !dbg !16
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %505, <4 x float> %504) #9, !dbg !16
  %lds.const.gep1160 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5248, !dbg !16
  %508 = load <4 x half>, ptr addrspace(3) %lds.const.gep1160, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %506, <4 x float> %376) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %510 = or disjoint i64 %83, 127, !dbg !13
  %511 = icmp uge i64 %19, %510, !dbg !13
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %508, <4 x float> %509) #9, !dbg !16
  %513 = and i1 %39, %511, !dbg !13
  br i1 %513, label %514, label %515, !dbg !13

514:                                              ; preds = %82
  br label %590, !dbg !13

515:                                              ; preds = %82
  %516 = add i64 %19, %35, !dbg !13
  %517 = shl nuw nsw i64 %36, 4, !dbg !13
  %518 = add i64 %516, %517, !dbg !13
  %519 = and i32 %34, 240, !dbg !13
  %index.compact = zext nneg i32 %519 to i64, !dbg !13
  %520 = sub i64 %518, %index.compact, !dbg !13
  %521 = add i64 %520, 1, !dbg !22
  %522 = and i64 %35, 48, !dbg !13
  %523 = or disjoint i64 %83, %522, !dbg !13
  %524 = lshr i64 %521, 32, !dbg !23
  %compare.high = trunc i64 %524 to i32, !dbg !23
  %525 = lshr i64 %523, 32, !dbg !23
  %compare.high978 = trunc i64 %525 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high978, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high978, !dbg !23
  %compare.low.base = trunc i64 %521 to i32, !dbg !23
  %compare.low.base979 = trunc i64 %523 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base979, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %526 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low980 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result981 = select i1 %compare.high.equal, i1 %compare.low980, i1 %compare.high.order, !dbg !23
  %527 = select i1 %compare.result981, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base982 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local983 = or i32 %compare.local.base982, 2, !dbg !23
  %compare.low984 = icmp ugt i32 %compare.low.base, %compare.local983, !dbg !23
  %compare.result985 = select i1 %compare.high.equal, i1 %compare.low984, i1 %compare.high.order, !dbg !23
  %528 = select i1 %compare.result985, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base986 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local987 = or i32 %compare.local.base986, 3, !dbg !23
  %compare.low988 = icmp ugt i32 %compare.low.base, %compare.local987, !dbg !23
  %compare.result989 = select i1 %compare.high.equal, i1 %compare.low988, i1 %compare.high.order, !dbg !23
  %529 = select i1 %compare.result989, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base990 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local991 = or i32 %compare.local.base990, 4, !dbg !23
  %compare.low992 = icmp ugt i32 %compare.low.base, %compare.local991, !dbg !23
  %compare.result993 = select i1 %compare.high.equal, i1 %compare.low992, i1 %compare.high.order, !dbg !23
  %530 = select i1 %compare.result993, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base994 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local995 = or i32 %compare.local.base994, 5, !dbg !23
  %compare.low996 = icmp ugt i32 %compare.low.base, %compare.local995, !dbg !23
  %compare.result997 = select i1 %compare.high.equal, i1 %compare.low996, i1 %compare.high.order, !dbg !23
  %531 = select i1 %compare.result997, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base998 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local999 = or i32 %compare.local.base998, 6, !dbg !23
  %compare.low1000 = icmp ugt i32 %compare.low.base, %compare.local999, !dbg !23
  %compare.result1001 = select i1 %compare.high.equal, i1 %compare.low1000, i1 %compare.high.order, !dbg !23
  %532 = select i1 %compare.result1001, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1002 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1003 = or i32 %compare.local.base1002, 7, !dbg !23
  %compare.low1004 = icmp ugt i32 %compare.low.base, %compare.local1003, !dbg !23
  %compare.result1005 = select i1 %compare.high.equal, i1 %compare.low1004, i1 %compare.high.order, !dbg !23
  %533 = select i1 %compare.result1005, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1006 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1007 = or i32 %compare.local.base1006, 8, !dbg !23
  %compare.low1008 = icmp ugt i32 %compare.low.base, %compare.local1007, !dbg !23
  %compare.result1009 = select i1 %compare.high.equal, i1 %compare.low1008, i1 %compare.high.order, !dbg !23
  %534 = select i1 %compare.result1009, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1010 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1011 = or i32 %compare.local.base1010, 9, !dbg !23
  %compare.low1012 = icmp ugt i32 %compare.low.base, %compare.local1011, !dbg !23
  %compare.result1013 = select i1 %compare.high.equal, i1 %compare.low1012, i1 %compare.high.order, !dbg !23
  %535 = select i1 %compare.result1013, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1014 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1015 = or i32 %compare.local.base1014, 10, !dbg !23
  %compare.low1016 = icmp ugt i32 %compare.low.base, %compare.local1015, !dbg !23
  %compare.result1017 = select i1 %compare.high.equal, i1 %compare.low1016, i1 %compare.high.order, !dbg !23
  %536 = select i1 %compare.result1017, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1018 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1019 = or i32 %compare.local.base1018, 11, !dbg !23
  %compare.low1020 = icmp ugt i32 %compare.low.base, %compare.local1019, !dbg !23
  %compare.result1021 = select i1 %compare.high.equal, i1 %compare.low1020, i1 %compare.high.order, !dbg !23
  %537 = select i1 %compare.result1021, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1022 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1023 = or i32 %compare.local.base1022, 12, !dbg !23
  %compare.low1024 = icmp ugt i32 %compare.low.base, %compare.local1023, !dbg !23
  %compare.result1025 = select i1 %compare.high.equal, i1 %compare.low1024, i1 %compare.high.order, !dbg !23
  %538 = select i1 %compare.result1025, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1026 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1027 = or i32 %compare.local.base1026, 13, !dbg !23
  %compare.low1028 = icmp ugt i32 %compare.low.base, %compare.local1027, !dbg !23
  %compare.result1029 = select i1 %compare.high.equal, i1 %compare.low1028, i1 %compare.high.order, !dbg !23
  %539 = select i1 %compare.result1029, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1030 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1031 = or i32 %compare.local.base1030, 14, !dbg !23
  %compare.low1032 = icmp ugt i32 %compare.low.base, %compare.local1031, !dbg !23
  %compare.result1033 = select i1 %compare.high.equal, i1 %compare.low1032, i1 %compare.high.order, !dbg !23
  %540 = select i1 %compare.result1033, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1034 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1035 = or i32 %compare.local.base1034, 15, !dbg !23
  %compare.low1036 = icmp ugt i32 %compare.low.base, %compare.local1035, !dbg !23
  %compare.result1037 = select i1 %compare.high.equal, i1 %compare.low1036, i1 %compare.high.order, !dbg !23
  %541 = select i1 %compare.result1037, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1038 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1039 = or i32 %compare.local.base1038, 64, !dbg !23
  %compare.low1040 = icmp ugt i32 %compare.low.base, %compare.local1039, !dbg !23
  %compare.result1041 = select i1 %compare.high.equal, i1 %compare.low1040, i1 %compare.high.order, !dbg !23
  %542 = select i1 %compare.result1041, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1042 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1043 = or i32 %compare.local.base1042, 65, !dbg !23
  %compare.low1044 = icmp ugt i32 %compare.low.base, %compare.local1043, !dbg !23
  %compare.result1045 = select i1 %compare.high.equal, i1 %compare.low1044, i1 %compare.high.order, !dbg !23
  %543 = select i1 %compare.result1045, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1046 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1047 = or i32 %compare.local.base1046, 66, !dbg !23
  %compare.low1048 = icmp ugt i32 %compare.low.base, %compare.local1047, !dbg !23
  %compare.result1049 = select i1 %compare.high.equal, i1 %compare.low1048, i1 %compare.high.order, !dbg !23
  %544 = select i1 %compare.result1049, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1050 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1051 = or i32 %compare.local.base1050, 67, !dbg !23
  %compare.low1052 = icmp ugt i32 %compare.low.base, %compare.local1051, !dbg !23
  %compare.result1053 = select i1 %compare.high.equal, i1 %compare.low1052, i1 %compare.high.order, !dbg !23
  %545 = select i1 %compare.result1053, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1054 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1055 = or i32 %compare.local.base1054, 68, !dbg !23
  %compare.low1056 = icmp ugt i32 %compare.low.base, %compare.local1055, !dbg !23
  %compare.result1057 = select i1 %compare.high.equal, i1 %compare.low1056, i1 %compare.high.order, !dbg !23
  %546 = select i1 %compare.result1057, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1058 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1059 = or i32 %compare.local.base1058, 69, !dbg !23
  %compare.low1060 = icmp ugt i32 %compare.low.base, %compare.local1059, !dbg !23
  %compare.result1061 = select i1 %compare.high.equal, i1 %compare.low1060, i1 %compare.high.order, !dbg !23
  %547 = select i1 %compare.result1061, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1062 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1063 = or i32 %compare.local.base1062, 70, !dbg !23
  %compare.low1064 = icmp ugt i32 %compare.low.base, %compare.local1063, !dbg !23
  %compare.result1065 = select i1 %compare.high.equal, i1 %compare.low1064, i1 %compare.high.order, !dbg !23
  %548 = select i1 %compare.result1065, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1066 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1067 = or i32 %compare.local.base1066, 71, !dbg !23
  %compare.low1068 = icmp ugt i32 %compare.low.base, %compare.local1067, !dbg !23
  %compare.result1069 = select i1 %compare.high.equal, i1 %compare.low1068, i1 %compare.high.order, !dbg !23
  %549 = select i1 %compare.result1069, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1070 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1071 = or i32 %compare.local.base1070, 72, !dbg !23
  %compare.low1072 = icmp ugt i32 %compare.low.base, %compare.local1071, !dbg !23
  %compare.result1073 = select i1 %compare.high.equal, i1 %compare.low1072, i1 %compare.high.order, !dbg !23
  %550 = select i1 %compare.result1073, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1074 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1075 = or i32 %compare.local.base1074, 73, !dbg !23
  %compare.low1076 = icmp ugt i32 %compare.low.base, %compare.local1075, !dbg !23
  %compare.result1077 = select i1 %compare.high.equal, i1 %compare.low1076, i1 %compare.high.order, !dbg !23
  %551 = select i1 %compare.result1077, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1078 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1079 = or i32 %compare.local.base1078, 74, !dbg !23
  %compare.low1080 = icmp ugt i32 %compare.low.base, %compare.local1079, !dbg !23
  %compare.result1081 = select i1 %compare.high.equal, i1 %compare.low1080, i1 %compare.high.order, !dbg !23
  %552 = select i1 %compare.result1081, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1082 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1083 = or i32 %compare.local.base1082, 75, !dbg !23
  %compare.low1084 = icmp ugt i32 %compare.low.base, %compare.local1083, !dbg !23
  %compare.result1085 = select i1 %compare.high.equal, i1 %compare.low1084, i1 %compare.high.order, !dbg !23
  %553 = select i1 %compare.result1085, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1086 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1087 = or i32 %compare.local.base1086, 76, !dbg !23
  %compare.low1088 = icmp ugt i32 %compare.low.base, %compare.local1087, !dbg !23
  %compare.result1089 = select i1 %compare.high.equal, i1 %compare.low1088, i1 %compare.high.order, !dbg !23
  %554 = select i1 %compare.result1089, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1090 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1091 = or i32 %compare.local.base1090, 77, !dbg !23
  %compare.low1092 = icmp ugt i32 %compare.low.base, %compare.local1091, !dbg !23
  %compare.result1093 = select i1 %compare.high.equal, i1 %compare.low1092, i1 %compare.high.order, !dbg !23
  %555 = select i1 %compare.result1093, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1094 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1095 = or i32 %compare.local.base1094, 78, !dbg !23
  %compare.low1096 = icmp ugt i32 %compare.low.base, %compare.local1095, !dbg !23
  %compare.result1097 = select i1 %compare.high.equal, i1 %compare.low1096, i1 %compare.high.order, !dbg !23
  %556 = select i1 %compare.result1097, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1098 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 31) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1099 = or i32 %compare.local.base1098, 79, !dbg !23
  %compare.low1100 = icmp ugt i32 %compare.low.base, %compare.local1099, !dbg !23
  %compare.result1101 = select i1 %compare.high.equal, i1 %compare.low1100, i1 %compare.high.order, !dbg !23
  %557 = select i1 %compare.result1101, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %558 = insertelement <32 x float> poison, float %526, i64 0, !dbg !13
  %559 = insertelement <32 x float> %558, float %527, i64 1, !dbg !13
  %560 = insertelement <32 x float> %559, float %528, i64 2, !dbg !13
  %561 = insertelement <32 x float> %560, float %529, i64 3, !dbg !13
  %562 = insertelement <32 x float> %561, float %530, i64 4, !dbg !13
  %563 = insertelement <32 x float> %562, float %531, i64 5, !dbg !13
  %564 = insertelement <32 x float> %563, float %532, i64 6, !dbg !13
  %565 = insertelement <32 x float> %564, float %533, i64 7, !dbg !13
  %566 = insertelement <32 x float> %565, float %534, i64 8, !dbg !13
  %567 = insertelement <32 x float> %566, float %535, i64 9, !dbg !13
  %568 = insertelement <32 x float> %567, float %536, i64 10, !dbg !13
  %569 = insertelement <32 x float> %568, float %537, i64 11, !dbg !13
  %570 = insertelement <32 x float> %569, float %538, i64 12, !dbg !13
  %571 = insertelement <32 x float> %570, float %539, i64 13, !dbg !13
  %572 = insertelement <32 x float> %571, float %540, i64 14, !dbg !13
  %573 = insertelement <32 x float> %572, float %541, i64 15, !dbg !13
  %574 = insertelement <32 x float> %573, float %542, i64 16, !dbg !13
  %575 = insertelement <32 x float> %574, float %543, i64 17, !dbg !13
  %576 = insertelement <32 x float> %575, float %544, i64 18, !dbg !13
  %577 = insertelement <32 x float> %576, float %545, i64 19, !dbg !13
  %578 = insertelement <32 x float> %577, float %546, i64 20, !dbg !13
  %579 = insertelement <32 x float> %578, float %547, i64 21, !dbg !13
  %580 = insertelement <32 x float> %579, float %548, i64 22, !dbg !13
  %581 = insertelement <32 x float> %580, float %549, i64 23, !dbg !13
  %582 = insertelement <32 x float> %581, float %550, i64 24, !dbg !13
  %583 = insertelement <32 x float> %582, float %551, i64 25, !dbg !13
  %584 = insertelement <32 x float> %583, float %552, i64 26, !dbg !13
  %585 = insertelement <32 x float> %584, float %553, i64 27, !dbg !13
  %586 = insertelement <32 x float> %585, float %554, i64 28, !dbg !13
  %587 = insertelement <32 x float> %586, float %555, i64 29, !dbg !13
  %588 = insertelement <32 x float> %587, float %556, i64 30, !dbg !13
  %589 = insertelement <32 x float> %588, float %557, i64 31, !dbg !13
  br label %590, !dbg !13

590:                                              ; preds = %514, %515
  %591 = phi <32 x float> [ %589, %515 ], [ zeroinitializer, %514 ], !dbg !13
  br label %592, !dbg !13

592:                                              ; preds = %590
  %593 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %594 = fadd <4 x float> %483, %593, !dbg !25
  %595 = extractelement <4 x float> %594, i64 0, !dbg !25
  %596 = extractelement <4 x float> %594, i64 1, !dbg !25
  %597 = extractelement <4 x float> %594, i64 2, !dbg !25
  %598 = extractelement <4 x float> %594, i64 3, !dbg !25
  %599 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %600 = fadd <4 x float> %487, %599, !dbg !25
  %601 = extractelement <4 x float> %600, i64 0, !dbg !25
  %602 = extractelement <4 x float> %600, i64 1, !dbg !25
  %603 = extractelement <4 x float> %600, i64 2, !dbg !25
  %604 = extractelement <4 x float> %600, i64 3, !dbg !25
  %605 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %606 = fadd <4 x float> %491, %605, !dbg !25
  %607 = extractelement <4 x float> %606, i64 0, !dbg !25
  %608 = extractelement <4 x float> %606, i64 1, !dbg !25
  %609 = extractelement <4 x float> %606, i64 2, !dbg !25
  %610 = extractelement <4 x float> %606, i64 3, !dbg !25
  %611 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %612 = fadd <4 x float> %495, %611, !dbg !25
  %613 = extractelement <4 x float> %612, i64 0, !dbg !25
  %614 = extractelement <4 x float> %612, i64 1, !dbg !25
  %615 = extractelement <4 x float> %612, i64 2, !dbg !25
  %616 = extractelement <4 x float> %612, i64 3, !dbg !25
  %617 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %618 = fadd <4 x float> %499, %617, !dbg !25
  %619 = extractelement <4 x float> %618, i64 0, !dbg !25
  %620 = extractelement <4 x float> %618, i64 1, !dbg !25
  %621 = extractelement <4 x float> %618, i64 2, !dbg !25
  %622 = extractelement <4 x float> %618, i64 3, !dbg !25
  %623 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %624 = fadd <4 x float> %503, %623, !dbg !25
  %625 = extractelement <4 x float> %624, i64 0, !dbg !25
  %626 = extractelement <4 x float> %624, i64 1, !dbg !25
  %627 = extractelement <4 x float> %624, i64 2, !dbg !25
  %628 = extractelement <4 x float> %624, i64 3, !dbg !25
  %629 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %630 = fadd <4 x float> %507, %629, !dbg !25
  %631 = extractelement <4 x float> %630, i64 0, !dbg !25
  %632 = extractelement <4 x float> %630, i64 1, !dbg !25
  %633 = extractelement <4 x float> %630, i64 2, !dbg !25
  %634 = extractelement <4 x float> %630, i64 3, !dbg !25
  %635 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %636 = fadd <4 x float> %512, %635, !dbg !25
  %637 = extractelement <4 x float> %636, i64 0, !dbg !25
  %638 = extractelement <4 x float> %636, i64 1, !dbg !25
  %639 = extractelement <4 x float> %636, i64 2, !dbg !25
  %640 = extractelement <4 x float> %636, i64 3, !dbg !25
  %641 = call float @llvm.maxnum.f32(float %595, float 0xFFF0000000000000), !dbg !26
  %642 = call float @llvm.maxnum.f32(float %641, float %596), !dbg !26
  %643 = call float @llvm.maxnum.f32(float %642, float %597), !dbg !26
  %644 = call float @llvm.maxnum.f32(float %643, float %598), !dbg !26
  %645 = call float @llvm.maxnum.f32(float %644, float %601), !dbg !26
  %646 = call float @llvm.maxnum.f32(float %645, float %602), !dbg !26
  %647 = call float @llvm.maxnum.f32(float %646, float %603), !dbg !26
  %648 = call float @llvm.maxnum.f32(float %647, float %604), !dbg !26
  %649 = call float @llvm.maxnum.f32(float %648, float %607), !dbg !26
  %650 = call float @llvm.maxnum.f32(float %649, float %608), !dbg !26
  %651 = call float @llvm.maxnum.f32(float %650, float %609), !dbg !26
  %652 = call float @llvm.maxnum.f32(float %651, float %610), !dbg !26
  %653 = call float @llvm.maxnum.f32(float %652, float %613), !dbg !26
  %654 = call float @llvm.maxnum.f32(float %653, float %614), !dbg !26
  %655 = call float @llvm.maxnum.f32(float %654, float %615), !dbg !26
  %656 = call float @llvm.maxnum.f32(float %655, float %616), !dbg !26
  %657 = call float @llvm.maxnum.f32(float %656, float %619), !dbg !26
  %658 = call float @llvm.maxnum.f32(float %657, float %620), !dbg !26
  %659 = call float @llvm.maxnum.f32(float %658, float %621), !dbg !26
  %660 = call float @llvm.maxnum.f32(float %659, float %622), !dbg !26
  %661 = call float @llvm.maxnum.f32(float %660, float %625), !dbg !26
  %662 = call float @llvm.maxnum.f32(float %661, float %626), !dbg !26
  %663 = call float @llvm.maxnum.f32(float %662, float %627), !dbg !26
  %664 = call float @llvm.maxnum.f32(float %663, float %628), !dbg !26
  %665 = call float @llvm.maxnum.f32(float %664, float %631), !dbg !26
  %666 = call float @llvm.maxnum.f32(float %665, float %632), !dbg !26
  %667 = call float @llvm.maxnum.f32(float %666, float %633), !dbg !26
  %668 = call float @llvm.maxnum.f32(float %667, float %634), !dbg !26
  %669 = call float @llvm.maxnum.f32(float %668, float %637), !dbg !26
  %670 = call float @llvm.maxnum.f32(float %669, float %638), !dbg !26
  %671 = call float @llvm.maxnum.f32(float %670, float %639), !dbg !26
  %672 = call float @llvm.maxnum.f32(float %671, float %640), !dbg !26
  %673 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %674 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %673), !dbg !26
  %675 = add nuw nsw i32 %674, 64, !dbg !26
  %676 = xor i32 %674, 16, !dbg !26
  %677 = and i32 %675, 192, !dbg !26
  %678 = icmp samesign ult i32 %676, %677, !dbg !26
  %679 = select i1 %678, i32 %676, i32 %674, !dbg !26
  %680 = shl nuw nsw i32 %679, 2, !dbg !26
  %681 = bitcast float %672 to i32, !dbg !26
  %682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %681), !dbg !26
  %683 = bitcast i32 %682 to float, !dbg !26
  %684 = call float @llvm.maxnum.f32(float %672, float %683), !dbg !26
  %685 = xor i32 %674, 32, !dbg !26
  %686 = icmp samesign ult i32 %685, %677, !dbg !26
  %687 = select i1 %686, i32 %685, i32 %674, !dbg !26
  %688 = shl nuw nsw i32 %687, 2, !dbg !26
  %689 = bitcast float %684 to i32, !dbg !26
  %690 = call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %689), !dbg !26
  %691 = bitcast i32 %690 to float, !dbg !26
  %692 = call float @llvm.maxnum.f32(float %684, float %691), !dbg !26
  %693 = insertelement <1 x float> poison, float %692, i64 0, !dbg !26
  %694 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %693, <1 x float> %71), !dbg !26
  %695 = shufflevector <1 x float> %694, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %696 = fsub <4 x float> %594, %695, !dbg !27
  %697 = fsub <4 x float> %600, %695, !dbg !27
  %698 = fsub <4 x float> %606, %695, !dbg !27
  %699 = fsub <4 x float> %612, %695, !dbg !27
  %700 = fsub <4 x float> %618, %695, !dbg !27
  %701 = fsub <4 x float> %624, %695, !dbg !27
  %702 = fsub <4 x float> %630, %695, !dbg !27
  %703 = fsub <4 x float> %636, %695, !dbg !27
  %704 = fsub <1 x float> %71, %694, !dbg !28
  %705 = extractelement <1 x float> %704, i64 0, !dbg !29
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !29
  %707 = insertelement <4 x float> poison, float %706, i64 0, !dbg !34
  %708 = shufflevector <4 x float> %707, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %709 = fmul <4 x float> %79, %708, !dbg !34
  %710 = fmul <4 x float> %78, %708, !dbg !34
  %711 = fmul <4 x float> %76, %708, !dbg !34
  %712 = fmul <4 x float> %75, %708, !dbg !34
  %713 = fmul <4 x float> %73, %708, !dbg !34
  %714 = fmul <4 x float> %72, %708, !dbg !34
  %715 = fmul <4 x float> %68, %708, !dbg !34
  %716 = fmul <4 x float> %67, %708, !dbg !34
  %717 = extractelement <1 x float> %70, i64 0, !dbg !30
  %718 = fmul float %717, %706, !dbg !30
  %719 = extractelement <4 x float> %696, i64 0, !dbg !31
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !31
  %721 = insertelement <4 x float> poison, float %720, i64 0, !dbg !33
  %722 = extractelement <4 x float> %696, i64 1, !dbg !31
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !31
  %724 = insertelement <4 x float> %721, float %723, i64 1, !dbg !33
  %725 = fadd float %720, %723, !dbg !32
  %726 = extractelement <4 x float> %696, i64 2, !dbg !31
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !31
  %728 = insertelement <4 x float> %724, float %727, i64 2, !dbg !33
  %729 = fadd float %725, %727, !dbg !32
  %730 = extractelement <4 x float> %696, i64 3, !dbg !31
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !31
  %732 = insertelement <4 x float> %728, float %731, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %732) #9, !dbg !33, !frisk.register_lifetimes !1
  %733 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %733) #9, !dbg !32, !frisk.register_lifetimes !1
  %734 = fadd float %729, %731, !dbg !32
  %735 = extractelement <4 x float> %697, i64 0, !dbg !31
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !31
  %737 = insertelement <4 x float> poison, float %736, i64 0, !dbg !33
  %738 = fadd float %734, %736, !dbg !32
  %739 = extractelement <4 x float> %697, i64 1, !dbg !31
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !31
  %741 = insertelement <4 x float> %737, float %740, i64 1, !dbg !33
  %742 = fadd float %738, %740, !dbg !32
  %743 = extractelement <4 x float> %697, i64 2, !dbg !31
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !31
  %745 = insertelement <4 x float> %741, float %744, i64 2, !dbg !33
  %746 = fadd float %742, %744, !dbg !32
  %747 = extractelement <4 x float> %697, i64 3, !dbg !31
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !31
  %749 = insertelement <4 x float> %745, float %748, i64 3, !dbg !33
  %math.result276 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %749) #9, !dbg !33, !frisk.register_lifetimes !1
  %750 = fptrunc <4 x float> %math.result276 to <4 x half>, !dbg !33
  %math.packed277 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %750) #9, !dbg !32, !frisk.register_lifetimes !1
  %751 = fadd float %746, %748, !dbg !32
  %752 = extractelement <4 x float> %698, i64 0, !dbg !31
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !31
  %754 = insertelement <4 x float> poison, float %753, i64 0, !dbg !33
  %755 = fadd float %751, %753, !dbg !32
  %756 = extractelement <4 x float> %698, i64 1, !dbg !31
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !31
  %758 = insertelement <4 x float> %754, float %757, i64 1, !dbg !33
  %759 = fadd float %755, %757, !dbg !32
  %760 = extractelement <4 x float> %698, i64 2, !dbg !31
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !31
  %762 = insertelement <4 x float> %758, float %761, i64 2, !dbg !33
  %763 = fadd float %759, %761, !dbg !32
  %764 = extractelement <4 x float> %698, i64 3, !dbg !31
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !31
  %766 = insertelement <4 x float> %762, float %765, i64 3, !dbg !33
  %math.result278 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %766) #9, !dbg !33, !frisk.register_lifetimes !1
  %767 = fptrunc <4 x float> %math.result278 to <4 x half>, !dbg !33
  %math.packed279 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %767) #9, !dbg !32, !frisk.register_lifetimes !1
  %768 = fadd float %763, %765, !dbg !32
  %769 = extractelement <4 x float> %699, i64 0, !dbg !31
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !31
  %771 = insertelement <4 x float> poison, float %770, i64 0, !dbg !33
  %772 = fadd float %768, %770, !dbg !32
  %773 = extractelement <4 x float> %699, i64 1, !dbg !31
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !31
  %775 = insertelement <4 x float> %771, float %774, i64 1, !dbg !33
  %776 = fadd float %772, %774, !dbg !32
  %777 = extractelement <4 x float> %699, i64 2, !dbg !31
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !31
  %779 = insertelement <4 x float> %775, float %778, i64 2, !dbg !33
  %780 = fadd float %776, %778, !dbg !32
  %781 = extractelement <4 x float> %699, i64 3, !dbg !31
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !31
  %783 = insertelement <4 x float> %779, float %782, i64 3, !dbg !33
  %math.result280 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %783) #9, !dbg !33, !frisk.register_lifetimes !1
  %784 = fptrunc <4 x float> %math.result280 to <4 x half>, !dbg !33
  %math.packed281 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %784) #9, !dbg !32, !frisk.register_lifetimes !1
  %785 = fadd float %780, %782, !dbg !32
  %786 = extractelement <4 x float> %700, i64 0, !dbg !31
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !31
  %788 = insertelement <4 x float> poison, float %787, i64 0, !dbg !33
  %789 = fadd float %785, %787, !dbg !32
  %790 = extractelement <4 x float> %700, i64 1, !dbg !31
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !31
  %792 = insertelement <4 x float> %788, float %791, i64 1, !dbg !33
  %793 = fadd float %789, %791, !dbg !32
  %794 = extractelement <4 x float> %700, i64 2, !dbg !31
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !31
  %796 = insertelement <4 x float> %792, float %795, i64 2, !dbg !33
  %797 = fadd float %793, %795, !dbg !32
  %798 = extractelement <4 x float> %700, i64 3, !dbg !31
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !31
  %800 = insertelement <4 x float> %796, float %799, i64 3, !dbg !33
  %math.result282 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %800) #9, !dbg !33, !frisk.register_lifetimes !1
  %801 = fptrunc <4 x float> %math.result282 to <4 x half>, !dbg !33
  %math.packed283 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %801) #9, !dbg !32, !frisk.register_lifetimes !1
  %802 = fadd float %797, %799, !dbg !32
  %803 = extractelement <4 x float> %701, i64 0, !dbg !31
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !31
  %805 = insertelement <4 x float> poison, float %804, i64 0, !dbg !33
  %806 = fadd float %802, %804, !dbg !32
  %807 = extractelement <4 x float> %701, i64 1, !dbg !31
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !31
  %809 = insertelement <4 x float> %805, float %808, i64 1, !dbg !33
  %810 = fadd float %806, %808, !dbg !32
  %811 = extractelement <4 x float> %701, i64 2, !dbg !31
  %812 = call float @llvm.amdgcn.exp2.f32(float %811), !dbg !31
  %813 = insertelement <4 x float> %809, float %812, i64 2, !dbg !33
  %814 = fadd float %810, %812, !dbg !32
  %815 = extractelement <4 x float> %701, i64 3, !dbg !31
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !31
  %817 = insertelement <4 x float> %813, float %816, i64 3, !dbg !33
  %math.result284 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %817) #9, !dbg !33, !frisk.register_lifetimes !1
  %818 = fptrunc <4 x float> %math.result284 to <4 x half>, !dbg !33
  %math.packed285 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %818) #9, !dbg !32, !frisk.register_lifetimes !1
  %819 = fadd float %814, %816, !dbg !32
  %820 = extractelement <4 x float> %702, i64 0, !dbg !31
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !31
  %822 = insertelement <4 x float> poison, float %821, i64 0, !dbg !33
  %823 = fadd float %819, %821, !dbg !32
  %824 = extractelement <4 x float> %702, i64 1, !dbg !31
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !31
  %826 = insertelement <4 x float> %822, float %825, i64 1, !dbg !33
  %827 = fadd float %823, %825, !dbg !32
  %828 = extractelement <4 x float> %702, i64 2, !dbg !31
  %829 = call float @llvm.amdgcn.exp2.f32(float %828), !dbg !31
  %830 = insertelement <4 x float> %826, float %829, i64 2, !dbg !33
  %831 = fadd float %827, %829, !dbg !32
  %832 = extractelement <4 x float> %702, i64 3, !dbg !31
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !31
  %834 = insertelement <4 x float> %830, float %833, i64 3, !dbg !33
  %math.result286 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %834) #9, !dbg !33, !frisk.register_lifetimes !1
  %835 = fptrunc <4 x float> %math.result286 to <4 x half>, !dbg !33
  %math.packed287 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %835) #9, !dbg !32, !frisk.register_lifetimes !1
  %836 = fadd float %831, %833, !dbg !32
  %837 = extractelement <4 x float> %703, i64 0, !dbg !31
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !31
  %839 = insertelement <4 x float> poison, float %838, i64 0, !dbg !33
  %840 = fadd float %836, %838, !dbg !32
  %841 = extractelement <4 x float> %703, i64 1, !dbg !31
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !31
  %843 = insertelement <4 x float> %839, float %842, i64 1, !dbg !33
  %844 = fadd float %840, %842, !dbg !32
  %845 = extractelement <4 x float> %703, i64 2, !dbg !31
  %846 = call float @llvm.amdgcn.exp2.f32(float %845), !dbg !31
  %847 = insertelement <4 x float> %843, float %846, i64 2, !dbg !33
  %848 = fadd float %844, %846, !dbg !32
  %849 = extractelement <4 x float> %703, i64 3, !dbg !31
  %850 = call float @llvm.amdgcn.exp2.f32(float %849), !dbg !31
  %851 = insertelement <4 x float> %847, float %850, i64 3, !dbg !33
  %math.result288 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %851) #9, !dbg !33, !frisk.register_lifetimes !1
  %852 = fptrunc <4 x float> %math.result288 to <4 x half>, !dbg !33
  %math.packed289 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %852) #9, !dbg !32, !frisk.register_lifetimes !1
  %853 = fadd float %848, %850, !dbg !32
  %854 = bitcast float %853 to i32, !dbg !32
  %855 = call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %854), !dbg !32
  %856 = bitcast i32 %855 to float, !dbg !32
  %857 = fadd float %853, %856, !dbg !32
  %858 = bitcast float %857 to i32, !dbg !32
  %859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %858), !dbg !32
  %860 = bitcast i32 %859 to float, !dbg !32
  %861 = fadd float %857, %860, !dbg !32
  %862 = fadd float %861, %718, !dbg !32
  %863 = insertelement <1 x float> poison, float %862, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread731 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local732 = shl nuw nsw i32 %address.thread731, 3, !dbg !14
  %address.local733 = and i32 %address.local732, 120, !dbg !14
  %address.local734 = shl nuw nsw i32 %address.thread731, 5, !dbg !14
  %address.local735 = and i32 %address.local734, 1536, !dbg !14
  %address.local736 = or disjoint i32 %address.local733, %address.local735, !dbg !14
  %address.local737 = zext nneg i32 %address.local736 to i64, !dbg !14
  %address.local738 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local737, !dbg !14
  %864 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local738, i32 0) #0, !dbg !14
  %865 = bitcast <4 x i32> %864 to <8 x half>, !dbg !14
  %address.local739 = shl nuw nsw i32 %address.thread731, 3, !dbg !14
  %address.local740 = and i32 %address.local739, 120, !dbg !14
  %address.local741 = shl nuw nsw i32 %address.thread731, 5, !dbg !14
  %address.local742 = and i32 %address.local741, 1536, !dbg !14
  %address.local743 = or disjoint i32 %address.local740, %address.local742, !dbg !14
  %address.local744 = or disjoint i32 %address.local743, 128, !dbg !14
  %address.local745 = zext nneg i32 %address.local744 to i64, !dbg !14
  %address.local746 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local745, !dbg !14
  %866 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local746, i32 0) #0, !dbg !14
  %867 = bitcast <4 x i32> %866 to <8 x half>, !dbg !14
  %address.local747 = shl nuw nsw i32 %address.thread731, 3, !dbg !14
  %address.local748 = and i32 %address.local747, 120, !dbg !14
  %address.local749 = shl nuw nsw i32 %address.thread731, 5, !dbg !14
  %address.local750 = and i32 %address.local749, 1536, !dbg !14
  %address.local751 = or disjoint i32 %address.local748, %address.local750, !dbg !14
  %address.local752 = or disjoint i32 %address.local751, 256, !dbg !14
  %address.local753 = zext nneg i32 %address.local752 to i64, !dbg !14
  %address.local754 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local753, !dbg !14
  %868 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local754, i32 0) #0, !dbg !14
  %869 = bitcast <4 x i32> %868 to <8 x half>, !dbg !14
  %address.local755 = shl nuw nsw i32 %address.thread731, 3, !dbg !14
  %address.local756 = and i32 %address.local755, 120, !dbg !14
  %address.local757 = shl nuw nsw i32 %address.thread731, 5, !dbg !14
  %address.local758 = and i32 %address.local757, 1536, !dbg !14
  %address.local759 = or disjoint i32 %address.local756, %address.local758, !dbg !14
  %address.local760 = or disjoint i32 %address.local759, 384, !dbg !14
  %address.local761 = zext nneg i32 %address.local760 to i64, !dbg !14
  %address.local762 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local761, !dbg !14
  %870 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local762, i32 0) #0, !dbg !14
  %871 = bitcast <4 x i32> %870 to <8 x half>, !dbg !14
  %address.local763 = shl nuw nsw i32 %address.thread731, 3, !dbg !14
  %address.local764 = and i32 %address.local763, 120, !dbg !14
  %address.local765 = shl nuw nsw i32 %address.thread731, 5, !dbg !14
  %address.local766 = and i32 %address.local765, 1536, !dbg !14
  %address.local767 = or disjoint i32 %address.local764, %address.local766, !dbg !14
  %address.local768 = or disjoint i32 %address.local767, 2048, !dbg !14
  %address.local769 = zext nneg i32 %address.local768 to i64, !dbg !14
  %address.local770 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local769, !dbg !14
  %872 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local770, i32 0) #0, !dbg !14
  %873 = bitcast <4 x i32> %872 to <8 x half>, !dbg !14
  %address.local771 = shl nuw nsw i32 %address.thread731, 3, !dbg !14
  %address.local772 = and i32 %address.local771, 120, !dbg !14
  %address.local773 = shl nuw nsw i32 %address.thread731, 5, !dbg !14
  %address.local774 = and i32 %address.local773, 1536, !dbg !14
  %address.local775 = or disjoint i32 %address.local772, %address.local774, !dbg !14
  %address.local776 = or disjoint i32 %address.local775, 2176, !dbg !14
  %address.local777 = zext nneg i32 %address.local776 to i64, !dbg !14
  %address.local778 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local777, !dbg !14
  %874 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #0, !dbg !14
  %875 = bitcast <4 x i32> %874 to <8 x half>, !dbg !14
  %address.local779 = shl nuw nsw i32 %address.thread731, 3, !dbg !14
  %address.local780 = and i32 %address.local779, 120, !dbg !14
  %address.local781 = shl nuw nsw i32 %address.thread731, 5, !dbg !14
  %address.local782 = and i32 %address.local781, 1536, !dbg !14
  %address.local783 = or disjoint i32 %address.local780, %address.local782, !dbg !14
  %address.local784 = or disjoint i32 %address.local783, 2304, !dbg !14
  %address.local785 = zext nneg i32 %address.local784 to i64, !dbg !14
  %address.local786 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local785, !dbg !14
  %876 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local786, i32 0) #0, !dbg !14
  %877 = bitcast <4 x i32> %876 to <8 x half>, !dbg !14
  %address.local787 = shl nuw nsw i32 %address.thread731, 3, !dbg !14
  %address.local788 = and i32 %address.local787, 120, !dbg !14
  %address.local789 = shl nuw nsw i32 %address.thread731, 5, !dbg !14
  %address.local790 = and i32 %address.local789, 1536, !dbg !14
  %address.local791 = or disjoint i32 %address.local788, %address.local790, !dbg !14
  %address.local792 = or disjoint i32 %address.local791, 2432, !dbg !14
  %address.local793 = zext nneg i32 %address.local792 to i64, !dbg !14
  %address.local794 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local793, !dbg !14
  %878 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local794, i32 0) #0, !dbg !14
  %879 = bitcast <4 x i32> %878 to <8 x half>, !dbg !14
  %880 = add nuw nsw i64 %100, %15, !dbg !17
  %881 = shl i64 %880, 19, !dbg !17
  %882 = and i64 %881, 16252928, !dbg !17
  %883 = add i64 %882, %106, !dbg !17
  %884 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %883, !dbg !17
  %885 = getelementptr i8, ptr addrspace(1) %884, i64 64, !dbg !17
  %address.local795 = zext nneg i32 %address.thread731 to i64, !dbg !12
  %address.local796 = shl nuw nsw i64 %address.local795, 3, !dbg !11
  %address.local797 = and i64 %address.local796, 1536, !dbg !15
  %address.local798 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !15
  %address.local799 = ptrtoint ptr addrspace(3) %address.local798 to i64, !dbg !15
  %address.local800 = trunc i64 %address.local799 to i32, !dbg !15
  %886 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local800), !dbg !17
  %887 = zext i32 %886 to i64, !dbg !17
  %888 = inttoptr i64 %887 to ptr addrspace(3), !dbg !17
  %address.local801 = shl nuw nsw i64 %address.local795, 5, !dbg !17
  %address.local802 = and i64 %address.local801, 8064, !dbg !17
  %address.local803 = shl nuw nsw i64 %address.local795, 3, !dbg !17
  %address.local804 = and i64 %address.local803, 24, !dbg !17
  %address.local805 = or disjoint i64 %address.local802, %address.local804, !dbg !17
  %address.local806 = getelementptr [2 x i8], ptr addrspace(1) %885, i64 %address.local805, !dbg !17
  %889 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %890 = ptrtoint ptr addrspace(1) %address.local806 to i64, !dbg !17
  %891 = sub i64 %890, %889, !dbg !17
  %dma.buffer.offset1173 = trunc i64 %891 to i32, !dbg !17
  %892 = trunc i64 %889 to i32, !dbg !17
  %893 = insertelement <4 x i32> poison, i32 %892, i64 0, !dbg !17
  %894 = lshr i64 %889, 32, !dbg !17
  %895 = trunc i64 %894 to i32, !dbg !17
  %896 = insertelement <4 x i32> %893, i32 %895, i64 1, !dbg !17
  %897 = insertelement <4 x i32> %896, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1174 = insertelement <4 x i32> %897, i32 131072, i64 3, !dbg !17
  %898 = ptrtoint ptr addrspace(3) %888 to i32, !dbg !17
  %dma.buffer.lds1175 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %898), !dbg !17
  %899 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1175, i32 %dma.buffer.offset1173, <4 x i32> %dma.buffer.rsrc1174) #10, !dbg !17
  %address.local807 = or disjoint i64 %address.local795, 256, !dbg !15
  %address.local808 = shl nuw nsw i64 %address.local807, 3, !dbg !15
  %address.local809 = and i64 %address.local808, 3584, !dbg !15
  %address.local810 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local809, !dbg !15
  %address.local811 = ptrtoint ptr addrspace(3) %address.local810 to i64, !dbg !15
  %address.local812 = trunc i64 %address.local811 to i32, !dbg !15
  %900 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local812), !dbg !17
  %901 = zext i32 %900 to i64, !dbg !17
  %902 = inttoptr i64 %901 to ptr addrspace(3), !dbg !17
  %address.local813 = shl nuw nsw i64 %address.local807, 5, !dbg !17
  %address.local814 = and i64 %address.local813, 16256, !dbg !17
  %address.local815 = shl nuw nsw i64 %address.local795, 3, !dbg !17
  %address.local816 = and i64 %address.local815, 24, !dbg !17
  %address.local817 = or disjoint i64 %address.local814, %address.local816, !dbg !17
  %address.local818 = getelementptr [2 x i8], ptr addrspace(1) %885, i64 %address.local817, !dbg !17
  %903 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %904 = ptrtoint ptr addrspace(1) %address.local818 to i64, !dbg !17
  %905 = sub i64 %904, %903, !dbg !17
  %dma.buffer.offset1170 = trunc i64 %905 to i32, !dbg !17
  %906 = trunc i64 %903 to i32, !dbg !17
  %907 = insertelement <4 x i32> poison, i32 %906, i64 0, !dbg !17
  %908 = lshr i64 %903, 32, !dbg !17
  %909 = trunc i64 %908 to i32, !dbg !17
  %910 = insertelement <4 x i32> %907, i32 %909, i64 1, !dbg !17
  %911 = insertelement <4 x i32> %910, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1171 = insertelement <4 x i32> %911, i32 131072, i64 3, !dbg !17
  %912 = ptrtoint ptr addrspace(3) %902 to i32, !dbg !17
  %dma.buffer.lds1172 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %912), !dbg !17
  %913 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1172, i32 %dma.buffer.offset1170, <4 x i32> %dma.buffer.rsrc1171) #10, !dbg !17
  %914 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %914, <4 x float> %716) #9, !dbg !14
  %916 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %916, <4 x float> %915) #9, !dbg !14
  %918 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %918, <4 x float> %917) #9, !dbg !14
  %920 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %920, <4 x float> %919) #9, !dbg !14
  %922 = shufflevector <8 x half> %873, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %922, <4 x float> %921) #9, !dbg !14
  %924 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %924, <4 x float> %923) #9, !dbg !14
  %926 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %926, <4 x float> %925) #9, !dbg !14
  %928 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %928, <4 x float> %927) #9, !dbg !14
  %930 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %930, <4 x float> %715) #9, !dbg !14
  %932 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %932, <4 x float> %931) #9, !dbg !14
  %934 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %934, <4 x float> %933) #9, !dbg !14
  %936 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %936, <4 x float> %935) #9, !dbg !14
  %938 = shufflevector <8 x half> %873, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %938, <4 x float> %937) #9, !dbg !14
  %940 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %940, <4 x float> %939) #9, !dbg !14
  %942 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %942, <4 x float> %941) #9, !dbg !14
  %944 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %944, <4 x float> %943) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local819 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local737, !dbg !14
  %946 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #0, !dbg !14
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !14
  %address.local820 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local745, !dbg !14
  %948 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !14
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !14
  %address.local821 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local753, !dbg !14
  %950 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local821, i32 0) #0, !dbg !14
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !14
  %address.local822 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local761, !dbg !14
  %952 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !14
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !14
  %address.local823 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local769, !dbg !14
  %954 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #0, !dbg !14
  %955 = bitcast <4 x i32> %954 to <8 x half>, !dbg !14
  %address.local824 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local777, !dbg !14
  %956 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local824, i32 0) #0, !dbg !14
  %957 = bitcast <4 x i32> %956 to <8 x half>, !dbg !14
  %address.local825 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local785, !dbg !14
  %958 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local825, i32 0) #0, !dbg !14
  %959 = bitcast <4 x i32> %958 to <8 x half>, !dbg !14
  %address.local826 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local793, !dbg !14
  %960 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !14
  %961 = bitcast <4 x i32> %960 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %962 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %883, !dbg !17
  %963 = getelementptr i8, ptr addrspace(1) %962, i64 128, !dbg !17
  %964 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local800), !dbg !17
  %965 = zext i32 %964 to i64, !dbg !17
  %966 = inttoptr i64 %965 to ptr addrspace(3), !dbg !17
  %address.local827 = getelementptr [2 x i8], ptr addrspace(1) %963, i64 %address.local805, !dbg !17
  %967 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %968 = ptrtoint ptr addrspace(1) %address.local827 to i64, !dbg !17
  %969 = sub i64 %968, %967, !dbg !17
  %dma.buffer.offset1167 = trunc i64 %969 to i32, !dbg !17
  %970 = trunc i64 %967 to i32, !dbg !17
  %971 = insertelement <4 x i32> poison, i32 %970, i64 0, !dbg !17
  %972 = lshr i64 %967, 32, !dbg !17
  %973 = trunc i64 %972 to i32, !dbg !17
  %974 = insertelement <4 x i32> %971, i32 %973, i64 1, !dbg !17
  %975 = insertelement <4 x i32> %974, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1168 = insertelement <4 x i32> %975, i32 131072, i64 3, !dbg !17
  %976 = ptrtoint ptr addrspace(3) %966 to i32, !dbg !17
  %dma.buffer.lds1169 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %976), !dbg !17
  %977 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1169, i32 %dma.buffer.offset1167, <4 x i32> %dma.buffer.rsrc1168) #10, !dbg !17
  %978 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local812), !dbg !17
  %979 = zext i32 %978 to i64, !dbg !17
  %980 = inttoptr i64 %979 to ptr addrspace(3), !dbg !17
  %address.local828 = getelementptr [2 x i8], ptr addrspace(1) %963, i64 %address.local817, !dbg !17
  %981 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %982 = ptrtoint ptr addrspace(1) %address.local828 to i64, !dbg !17
  %983 = sub i64 %982, %981, !dbg !17
  %dma.buffer.offset1164 = trunc i64 %983 to i32, !dbg !17
  %984 = trunc i64 %981 to i32, !dbg !17
  %985 = insertelement <4 x i32> poison, i32 %984, i64 0, !dbg !17
  %986 = lshr i64 %981, 32, !dbg !17
  %987 = trunc i64 %986 to i32, !dbg !17
  %988 = insertelement <4 x i32> %985, i32 %987, i64 1, !dbg !17
  %989 = insertelement <4 x i32> %988, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1165 = insertelement <4 x i32> %989, i32 131072, i64 3, !dbg !17
  %990 = ptrtoint ptr addrspace(3) %980 to i32, !dbg !17
  %dma.buffer.lds1166 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %990), !dbg !17
  %991 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1166, i32 %dma.buffer.offset1164, <4 x i32> %dma.buffer.rsrc1165) #10, !dbg !17
  %992 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %992, <4 x float> %714) #9, !dbg !14
  %994 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %994, <4 x float> %993) #9, !dbg !14
  %996 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %996, <4 x float> %995) #9, !dbg !14
  %998 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %998, <4 x float> %997) #9, !dbg !14
  %1000 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1000, <4 x float> %999) #9, !dbg !14
  %1002 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1002, <4 x float> %1001) #9, !dbg !14
  %1004 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1004, <4 x float> %1003) #9, !dbg !14
  %1006 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1007 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1006, <4 x float> %1005) #9, !dbg !14
  %1008 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1008, <4 x float> %713) #9, !dbg !14
  %1010 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1010, <4 x float> %1009) #9, !dbg !14
  %1012 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %1012, <4 x float> %1011) #9, !dbg !14
  %1014 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %1014, <4 x float> %1013) #9, !dbg !14
  %1016 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1017 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1016, <4 x float> %1015) #9, !dbg !14
  %1018 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1018, <4 x float> %1017) #9, !dbg !14
  %1020 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1020, <4 x float> %1019) #9, !dbg !14
  %1022 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1022, <4 x float> %1021) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1024 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #0, !dbg !14
  %1025 = bitcast <4 x i32> %1024 to <8 x half>, !dbg !14
  %1026 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !14
  %1027 = bitcast <4 x i32> %1026 to <8 x half>, !dbg !14
  %1028 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local821, i32 0) #0, !dbg !14
  %1029 = bitcast <4 x i32> %1028 to <8 x half>, !dbg !14
  %1030 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !14
  %1031 = bitcast <4 x i32> %1030 to <8 x half>, !dbg !14
  %1032 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #0, !dbg !14
  %1033 = bitcast <4 x i32> %1032 to <8 x half>, !dbg !14
  %1034 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local824, i32 0) #0, !dbg !14
  %1035 = bitcast <4 x i32> %1034 to <8 x half>, !dbg !14
  %1036 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local825, i32 0) #0, !dbg !14
  %1037 = bitcast <4 x i32> %1036 to <8 x half>, !dbg !14
  %1038 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !14
  %1039 = bitcast <4 x i32> %1038 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1040 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %883, !dbg !17
  %1041 = getelementptr i8, ptr addrspace(1) %1040, i64 192, !dbg !17
  %1042 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local800), !dbg !17
  %1043 = zext i32 %1042 to i64, !dbg !17
  %1044 = inttoptr i64 %1043 to ptr addrspace(3), !dbg !17
  %address.local829 = getelementptr [2 x i8], ptr addrspace(1) %1041, i64 %address.local805, !dbg !17
  %1045 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1046 = ptrtoint ptr addrspace(1) %address.local829 to i64, !dbg !17
  %1047 = sub i64 %1046, %1045, !dbg !17
  %dma.buffer.offset1161 = trunc i64 %1047 to i32, !dbg !17
  %1048 = trunc i64 %1045 to i32, !dbg !17
  %1049 = insertelement <4 x i32> poison, i32 %1048, i64 0, !dbg !17
  %1050 = lshr i64 %1045, 32, !dbg !17
  %1051 = trunc i64 %1050 to i32, !dbg !17
  %1052 = insertelement <4 x i32> %1049, i32 %1051, i64 1, !dbg !17
  %1053 = insertelement <4 x i32> %1052, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1162 = insertelement <4 x i32> %1053, i32 131072, i64 3, !dbg !17
  %1054 = ptrtoint ptr addrspace(3) %1044 to i32, !dbg !17
  %dma.buffer.lds1163 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1054), !dbg !17
  %1055 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1163, i32 %dma.buffer.offset1161, <4 x i32> %dma.buffer.rsrc1162) #10, !dbg !17
  %1056 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local812), !dbg !17
  %1057 = zext i32 %1056 to i64, !dbg !17
  %1058 = inttoptr i64 %1057 to ptr addrspace(3), !dbg !17
  %address.local830 = getelementptr [2 x i8], ptr addrspace(1) %1041, i64 %address.local817, !dbg !17
  %1059 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1060 = ptrtoint ptr addrspace(1) %address.local830 to i64, !dbg !17
  %1061 = sub i64 %1060, %1059, !dbg !17
  %dma.buffer.offset = trunc i64 %1061 to i32, !dbg !17
  %1062 = trunc i64 %1059 to i32, !dbg !17
  %1063 = insertelement <4 x i32> poison, i32 %1062, i64 0, !dbg !17
  %1064 = lshr i64 %1059, 32, !dbg !17
  %1065 = trunc i64 %1064 to i32, !dbg !17
  %1066 = insertelement <4 x i32> %1063, i32 %1065, i64 1, !dbg !17
  %1067 = insertelement <4 x i32> %1066, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %1067, i32 131072, i64 3, !dbg !17
  %1068 = ptrtoint ptr addrspace(3) %1058 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1068), !dbg !17
  %1069 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %1070 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1070, <4 x float> %712) #9, !dbg !14
  %1072 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1072, <4 x float> %1071) #9, !dbg !14
  %1074 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %1074, <4 x float> %1073) #9, !dbg !14
  %1076 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %1076, <4 x float> %1075) #9, !dbg !14
  %1078 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1078, <4 x float> %1077) #9, !dbg !14
  %1080 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1080, <4 x float> %1079) #9, !dbg !14
  %1082 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1082, <4 x float> %1081) #9, !dbg !14
  %1084 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1084, <4 x float> %1083) #9, !dbg !14
  %1086 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1086, <4 x float> %711) #9, !dbg !14
  %1088 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1088, <4 x float> %1087) #9, !dbg !14
  %1090 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %1090, <4 x float> %1089) #9, !dbg !14
  %1092 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %1092, <4 x float> %1091) #9, !dbg !14
  %1094 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1094, <4 x float> %1093) #9, !dbg !14
  %1096 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1096, <4 x float> %1095) #9, !dbg !14
  %1098 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1098, <4 x float> %1097) #9, !dbg !14
  %1100 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1101 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1100, <4 x float> %1099) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1102 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #0, !dbg !14
  %1103 = bitcast <4 x i32> %1102 to <8 x half>, !dbg !14
  %1104 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !14
  %1105 = bitcast <4 x i32> %1104 to <8 x half>, !dbg !14
  %1106 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local821, i32 0) #0, !dbg !14
  %1107 = bitcast <4 x i32> %1106 to <8 x half>, !dbg !14
  %1108 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !14
  %1109 = bitcast <4 x i32> %1108 to <8 x half>, !dbg !14
  %1110 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #0, !dbg !14
  %1111 = bitcast <4 x i32> %1110 to <8 x half>, !dbg !14
  %1112 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local824, i32 0) #0, !dbg !14
  %1113 = bitcast <4 x i32> %1112 to <8 x half>, !dbg !14
  %1114 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local825, i32 0) #0, !dbg !14
  %1115 = bitcast <4 x i32> %1114 to <8 x half>, !dbg !14
  %1116 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !14
  %1117 = bitcast <4 x i32> %1116 to <8 x half>, !dbg !14
  %1118 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1118, <4 x float> %710) #9, !dbg !14
  %1120 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1120, <4 x float> %1119) #9, !dbg !14
  %1122 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %1122, <4 x float> %1121) #9, !dbg !14
  %1124 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %1124, <4 x float> %1123) #9, !dbg !14
  %1126 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1126, <4 x float> %1125) #9, !dbg !14
  %1128 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1128, <4 x float> %1127) #9, !dbg !14
  %1130 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1130, <4 x float> %1129) #9, !dbg !14
  %1132 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1132, <4 x float> %1131) #9, !dbg !14
  %1134 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1134, <4 x float> %709) #9, !dbg !14
  %1136 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1136, <4 x float> %1135) #9, !dbg !14
  %1138 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %1138, <4 x float> %1137) #9, !dbg !14
  %1140 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %1140, <4 x float> %1139) #9, !dbg !14
  %1142 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1142, <4 x float> %1141) #9, !dbg !14
  %1144 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1144, <4 x float> %1143) #9, !dbg !14
  %1146 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1146, <4 x float> %1145) #9, !dbg !14
  %1148 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1148, <4 x float> %1147) #9, !dbg !14
  %1150 = shufflevector <4 x float> %929, <4 x float> %945, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1151 = shufflevector <4 x float> %1007, <4 x float> %1023, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1152 = shufflevector <4 x float> %1085, <4 x float> %1101, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1153 = shufflevector <4 x float> %1133, <4 x float> %1149, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1154 = add nuw nsw i64 %66, 1, !dbg !7
  br label %65, !dbg !7

1155:                                             ; preds = %65
  %1156 = and i32 %34, 16, !dbg !7
  %.not = icmp eq i32 %1156, 0, !dbg !7
  %1157 = and i32 %34, 32, !dbg !7
  %.not36 = icmp eq i32 %1157, 0, !dbg !7
  %1158 = extractelement <8 x float> %80, i64 0, !dbg !7
  %1159 = extractelement <8 x float> %80, i64 1, !dbg !7
  %1160 = extractelement <8 x float> %80, i64 2, !dbg !7
  %1161 = extractelement <8 x float> %80, i64 3, !dbg !7
  %1162 = select i1 %.not, float %1159, float %1158, !dbg !7
  %1163 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1164 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1163), !dbg !7
  %1165 = add nuw nsw i32 %1164, 64, !dbg !7
  %1166 = xor i32 %1164, 16, !dbg !7
  %1167 = and i32 %1165, 192, !dbg !7
  %1168 = icmp samesign ult i32 %1166, %1167, !dbg !7
  %1169 = select i1 %1168, i32 %1166, i32 %1164, !dbg !7
  %1170 = shl nuw nsw i32 %1169, 2, !dbg !7
  %1171 = bitcast float %1162 to i32, !dbg !7
  %1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1171), !dbg !7
  %1173 = bitcast i32 %1172 to float, !dbg !7
  %1174 = select i1 %.not, float %1158, float %1173, !dbg !7
  %1175 = select i1 %.not, float %1173, float %1159, !dbg !7
  %1176 = select i1 %.not, float %1161, float %1160, !dbg !7
  %1177 = bitcast float %1176 to i32, !dbg !7
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1177), !dbg !7
  %1179 = bitcast i32 %1178 to float, !dbg !7
  %1180 = select i1 %.not, float %1160, float %1179, !dbg !7
  %1181 = select i1 %.not, float %1179, float %1161, !dbg !7
  %1182 = select i1 %.not36, float %1180, float %1174, !dbg !7
  %1183 = xor i32 %1164, 32, !dbg !7
  %1184 = icmp samesign ult i32 %1183, %1167, !dbg !7
  %1185 = select i1 %1184, i32 %1183, i32 %1164, !dbg !7
  %1186 = shl nuw nsw i32 %1185, 2, !dbg !7
  %1187 = bitcast float %1182 to i32, !dbg !7
  %1188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1187), !dbg !7
  %1189 = bitcast i32 %1188 to float, !dbg !7
  %1190 = select i1 %.not36, float %1174, float %1189, !dbg !7
  %1191 = select i1 %.not36, float %1189, float %1180, !dbg !7
  %1192 = select i1 %.not36, float %1181, float %1175, !dbg !7
  %1193 = bitcast float %1192 to i32, !dbg !7
  %1194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1193), !dbg !7
  %1195 = bitcast i32 %1194 to float, !dbg !7
  %1196 = select i1 %.not36, float %1175, float %1195, !dbg !7
  %1197 = select i1 %.not36, float %1195, float %1181, !dbg !7
  %1198 = extractelement <8 x float> %77, i64 0, !dbg !7
  %1199 = extractelement <8 x float> %77, i64 1, !dbg !7
  %1200 = extractelement <8 x float> %77, i64 2, !dbg !7
  %1201 = extractelement <8 x float> %77, i64 3, !dbg !7
  %1202 = select i1 %.not, float %1199, float %1198, !dbg !7
  %1203 = bitcast float %1202 to i32, !dbg !7
  %1204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1203), !dbg !7
  %1205 = bitcast i32 %1204 to float, !dbg !7
  %1206 = select i1 %.not, float %1198, float %1205, !dbg !7
  %1207 = select i1 %.not, float %1205, float %1199, !dbg !7
  %1208 = select i1 %.not, float %1201, float %1200, !dbg !7
  %1209 = bitcast float %1208 to i32, !dbg !7
  %1210 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1209), !dbg !7
  %1211 = bitcast i32 %1210 to float, !dbg !7
  %1212 = select i1 %.not, float %1200, float %1211, !dbg !7
  %1213 = select i1 %.not, float %1211, float %1201, !dbg !7
  %1214 = select i1 %.not36, float %1212, float %1206, !dbg !7
  %1215 = bitcast float %1214 to i32, !dbg !7
  %1216 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1215), !dbg !7
  %1217 = bitcast i32 %1216 to float, !dbg !7
  %1218 = select i1 %.not36, float %1206, float %1217, !dbg !7
  %1219 = select i1 %.not36, float %1217, float %1212, !dbg !7
  %1220 = select i1 %.not36, float %1213, float %1207, !dbg !7
  %1221 = bitcast float %1220 to i32, !dbg !7
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1221), !dbg !7
  %1223 = bitcast i32 %1222 to float, !dbg !7
  %1224 = select i1 %.not36, float %1207, float %1223, !dbg !7
  %1225 = select i1 %.not36, float %1223, float %1213, !dbg !7
  %1226 = extractelement <8 x float> %74, i64 0, !dbg !7
  %1227 = extractelement <8 x float> %74, i64 1, !dbg !7
  %1228 = extractelement <8 x float> %74, i64 2, !dbg !7
  %1229 = extractelement <8 x float> %74, i64 3, !dbg !7
  %1230 = select i1 %.not, float %1227, float %1226, !dbg !7
  %1231 = bitcast float %1230 to i32, !dbg !7
  %1232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1231), !dbg !7
  %1233 = bitcast i32 %1232 to float, !dbg !7
  %1234 = select i1 %.not, float %1226, float %1233, !dbg !7
  %1235 = select i1 %.not, float %1233, float %1227, !dbg !7
  %1236 = select i1 %.not, float %1229, float %1228, !dbg !7
  %1237 = bitcast float %1236 to i32, !dbg !7
  %1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1237), !dbg !7
  %1239 = bitcast i32 %1238 to float, !dbg !7
  %1240 = select i1 %.not, float %1228, float %1239, !dbg !7
  %1241 = select i1 %.not, float %1239, float %1229, !dbg !7
  %1242 = select i1 %.not36, float %1240, float %1234, !dbg !7
  %1243 = bitcast float %1242 to i32, !dbg !7
  %1244 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1243), !dbg !7
  %1245 = bitcast i32 %1244 to float, !dbg !7
  %1246 = select i1 %.not36, float %1234, float %1245, !dbg !7
  %1247 = select i1 %.not36, float %1245, float %1240, !dbg !7
  %1248 = select i1 %.not36, float %1241, float %1235, !dbg !7
  %1249 = bitcast float %1248 to i32, !dbg !7
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1249), !dbg !7
  %1251 = bitcast i32 %1250 to float, !dbg !7
  %1252 = select i1 %.not36, float %1235, float %1251, !dbg !7
  %1253 = select i1 %.not36, float %1251, float %1241, !dbg !7
  %1254 = extractelement <8 x float> %69, i64 0, !dbg !7
  %1255 = extractelement <8 x float> %69, i64 1, !dbg !7
  %1256 = extractelement <8 x float> %69, i64 2, !dbg !7
  %1257 = extractelement <8 x float> %69, i64 3, !dbg !7
  %1258 = select i1 %.not, float %1255, float %1254, !dbg !7
  %1259 = bitcast float %1258 to i32, !dbg !7
  %1260 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1259), !dbg !7
  %1261 = bitcast i32 %1260 to float, !dbg !7
  %1262 = select i1 %.not, float %1254, float %1261, !dbg !7
  %1263 = select i1 %.not, float %1261, float %1255, !dbg !7
  %1264 = select i1 %.not, float %1257, float %1256, !dbg !7
  %1265 = bitcast float %1264 to i32, !dbg !7
  %1266 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1265), !dbg !7
  %1267 = bitcast i32 %1266 to float, !dbg !7
  %1268 = select i1 %.not, float %1256, float %1267, !dbg !7
  %1269 = select i1 %.not, float %1267, float %1257, !dbg !7
  %1270 = select i1 %.not36, float %1268, float %1262, !dbg !7
  %1271 = bitcast float %1270 to i32, !dbg !7
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1271), !dbg !7
  %1273 = select i1 %.not36, float %1269, float %1263, !dbg !7
  %1274 = bitcast float %1273 to i32, !dbg !7
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1274), !dbg !7
  %1276 = shufflevector <1 x float> %70, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1277 = extractelement <8 x float> %80, i64 4, !dbg !7
  %1278 = extractelement <8 x float> %80, i64 5, !dbg !7
  %1279 = extractelement <8 x float> %80, i64 6, !dbg !7
  %1280 = extractelement <8 x float> %80, i64 7, !dbg !7
  %1281 = select i1 %.not, float %1278, float %1277, !dbg !7
  %1282 = bitcast float %1281 to i32, !dbg !7
  %1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1282), !dbg !7
  %1284 = bitcast i32 %1283 to float, !dbg !7
  %1285 = select i1 %.not, float %1277, float %1284, !dbg !7
  %1286 = select i1 %.not, float %1284, float %1278, !dbg !7
  %1287 = select i1 %.not, float %1280, float %1279, !dbg !7
  %1288 = bitcast float %1287 to i32, !dbg !7
  %1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1288), !dbg !7
  %1290 = bitcast i32 %1289 to float, !dbg !7
  %1291 = select i1 %.not, float %1279, float %1290, !dbg !7
  %1292 = select i1 %.not, float %1290, float %1280, !dbg !7
  %1293 = select i1 %.not36, float %1291, float %1285, !dbg !7
  %1294 = bitcast float %1293 to i32, !dbg !7
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1294), !dbg !7
  %1296 = bitcast i32 %1295 to float, !dbg !7
  %1297 = select i1 %.not36, float %1285, float %1296, !dbg !7
  %1298 = select i1 %.not36, float %1296, float %1291, !dbg !7
  %1299 = select i1 %.not36, float %1292, float %1286, !dbg !7
  %1300 = bitcast float %1299 to i32, !dbg !7
  %1301 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1300), !dbg !7
  %1302 = bitcast i32 %1301 to float, !dbg !7
  %1303 = select i1 %.not36, float %1286, float %1302, !dbg !7
  %1304 = select i1 %.not36, float %1302, float %1292, !dbg !7
  %1305 = extractelement <8 x float> %77, i64 4, !dbg !7
  %1306 = extractelement <8 x float> %77, i64 5, !dbg !7
  %1307 = extractelement <8 x float> %77, i64 6, !dbg !7
  %1308 = extractelement <8 x float> %77, i64 7, !dbg !7
  %1309 = select i1 %.not, float %1306, float %1305, !dbg !7
  %1310 = bitcast float %1309 to i32, !dbg !7
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1310), !dbg !7
  %1312 = bitcast i32 %1311 to float, !dbg !7
  %1313 = select i1 %.not, float %1305, float %1312, !dbg !7
  %1314 = select i1 %.not, float %1312, float %1306, !dbg !7
  %1315 = select i1 %.not, float %1308, float %1307, !dbg !7
  %1316 = bitcast float %1315 to i32, !dbg !7
  %1317 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1316), !dbg !7
  %1318 = bitcast i32 %1317 to float, !dbg !7
  %1319 = select i1 %.not, float %1307, float %1318, !dbg !7
  %1320 = select i1 %.not, float %1318, float %1308, !dbg !7
  %1321 = select i1 %.not36, float %1319, float %1313, !dbg !7
  %1322 = bitcast float %1321 to i32, !dbg !7
  %1323 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1322), !dbg !7
  %1324 = bitcast i32 %1323 to float, !dbg !7
  %1325 = select i1 %.not36, float %1313, float %1324, !dbg !7
  %1326 = select i1 %.not36, float %1324, float %1319, !dbg !7
  %1327 = select i1 %.not36, float %1320, float %1314, !dbg !7
  %1328 = bitcast float %1327 to i32, !dbg !7
  %1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1328), !dbg !7
  %1330 = bitcast i32 %1329 to float, !dbg !7
  %1331 = select i1 %.not36, float %1314, float %1330, !dbg !7
  %1332 = select i1 %.not36, float %1330, float %1320, !dbg !7
  %1333 = extractelement <8 x float> %74, i64 4, !dbg !7
  %1334 = extractelement <8 x float> %74, i64 5, !dbg !7
  %1335 = extractelement <8 x float> %74, i64 6, !dbg !7
  %1336 = extractelement <8 x float> %74, i64 7, !dbg !7
  %1337 = select i1 %.not, float %1334, float %1333, !dbg !7
  %1338 = bitcast float %1337 to i32, !dbg !7
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1338), !dbg !7
  %1340 = bitcast i32 %1339 to float, !dbg !7
  %1341 = select i1 %.not, float %1333, float %1340, !dbg !7
  %1342 = select i1 %.not, float %1340, float %1334, !dbg !7
  %1343 = select i1 %.not, float %1336, float %1335, !dbg !7
  %1344 = bitcast float %1343 to i32, !dbg !7
  %1345 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1344), !dbg !7
  %1346 = bitcast i32 %1345 to float, !dbg !7
  %1347 = select i1 %.not, float %1335, float %1346, !dbg !7
  %1348 = select i1 %.not, float %1346, float %1336, !dbg !7
  %1349 = select i1 %.not36, float %1347, float %1341, !dbg !7
  %1350 = bitcast float %1349 to i32, !dbg !7
  %1351 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1350), !dbg !7
  %1352 = bitcast i32 %1351 to float, !dbg !7
  %1353 = select i1 %.not36, float %1341, float %1352, !dbg !7
  %1354 = select i1 %.not36, float %1352, float %1347, !dbg !7
  %1355 = select i1 %.not36, float %1348, float %1342, !dbg !7
  %1356 = bitcast float %1355 to i32, !dbg !7
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1356), !dbg !7
  %1358 = bitcast i32 %1357 to float, !dbg !7
  %1359 = select i1 %.not36, float %1342, float %1358, !dbg !7
  %1360 = select i1 %.not36, float %1358, float %1348, !dbg !7
  %1361 = extractelement <8 x float> %69, i64 4, !dbg !7
  %1362 = extractelement <8 x float> %69, i64 5, !dbg !7
  %1363 = extractelement <8 x float> %69, i64 6, !dbg !7
  %1364 = extractelement <8 x float> %69, i64 7, !dbg !7
  %1365 = select i1 %.not, float %1362, float %1361, !dbg !7
  %1366 = bitcast float %1365 to i32, !dbg !7
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1366), !dbg !7
  %1368 = bitcast i32 %1367 to float, !dbg !7
  %1369 = select i1 %.not, float %1361, float %1368, !dbg !7
  %1370 = select i1 %.not, float %1368, float %1362, !dbg !7
  %1371 = select i1 %.not, float %1364, float %1363, !dbg !7
  %1372 = bitcast float %1371 to i32, !dbg !7
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1372), !dbg !7
  %1374 = bitcast i32 %1373 to float, !dbg !7
  %1375 = select i1 %.not, float %1363, float %1374, !dbg !7
  %1376 = select i1 %.not, float %1374, float %1364, !dbg !7
  %1377 = select i1 %.not36, float %1375, float %1369, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1378), !dbg !7
  %1380 = select i1 %.not36, float %1376, float %1370, !dbg !7
  %1381 = bitcast float %1380 to i32, !dbg !7
  %1382 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1381), !dbg !7
  br i1 true, label %1383, label %1417, !dbg !20

1383:                                             ; preds = %1155
  %1384 = bitcast i32 %1379 to float, !dbg !7
  %1385 = select i1 %.not36, float %1369, float %1384, !dbg !7
  %1386 = insertelement <4 x float> poison, float %1385, i64 0, !dbg !18
  %1387 = bitcast i32 %1382 to float, !dbg !7
  %1388 = select i1 %.not36, float %1370, float %1387, !dbg !7
  %1389 = insertelement <4 x float> %1386, float %1388, i64 1, !dbg !18
  %1390 = select i1 %.not36, float %1384, float %1375, !dbg !7
  %1391 = insertelement <4 x float> %1389, float %1390, i64 2, !dbg !18
  %1392 = select i1 %.not36, float %1387, float %1376, !dbg !7
  %1393 = insertelement <4 x float> %1391, float %1392, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !18
  %normalized = fmul <4 x float> %1393, %row.reciprocal, !dbg !18
  %1394 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %1395 = extractelement <4 x half> %1394, i64 3, !dbg !19
  %1396 = extractelement <4 x half> %1394, i64 2, !dbg !19
  %1397 = extractelement <4 x half> %1394, i64 1, !dbg !19
  %1398 = extractelement <4 x half> %1394, i64 0, !dbg !19
  %1399 = bitcast i32 %1272 to float, !dbg !7
  %1400 = select i1 %.not36, float %1262, float %1399, !dbg !7
  %1401 = insertelement <4 x float> poison, float %1400, i64 0, !dbg !18
  %1402 = bitcast i32 %1275 to float, !dbg !7
  %1403 = select i1 %.not36, float %1263, float %1402, !dbg !7
  %1404 = insertelement <4 x float> %1401, float %1403, i64 1, !dbg !18
  %1405 = select i1 %.not36, float %1399, float %1268, !dbg !7
  %1406 = insertelement <4 x float> %1404, float %1405, i64 2, !dbg !18
  %1407 = select i1 %.not36, float %1402, float %1269, !dbg !7
  %1408 = insertelement <4 x float> %1406, float %1407, i64 3, !dbg !18
  %normalized72 = fmul <4 x float> %1408, %row.reciprocal, !dbg !18
  %1409 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !19
  %1410 = extractelement <4 x half> %1409, i64 3, !dbg !19
  %1411 = extractelement <4 x half> %1409, i64 2, !dbg !19
  %1412 = extractelement <4 x half> %1409, i64 1, !dbg !19
  %1413 = extractelement <4 x half> %1409, i64 0, !dbg !19
  %1414 = add nsw i64 %15, %25, !dbg !20
  %1415 = shl i64 %1414, 19, !dbg !20
  %1416 = and i64 %1415, 16252928, !dbg !20
  %address.thread831 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local832 = zext nneg i32 %address.thread831 to i64, !dbg !12
  %address.local833 = add i64 %28, %address.local832, !dbg !20
  %address.local834 = lshr i64 %address.local832, 6, !dbg !12
  %address.local835 = shl nuw nsw i64 %address.local834, 4, !dbg !20
  %address.local836 = add i64 %address.local833, %address.local835, !dbg !20
  %address.local837 = and i64 %address.local832, 240, !dbg !20
  %address.local838 = sub i64 %address.local836, %address.local837, !dbg !20
  %address.local839 = add i64 %address.local838, %29, !dbg !20
  %address.local840 = shl nuw nsw i64 %address.local839, 7, !dbg !20
  %address.local841 = add nuw nsw i64 %1416, %address.local840, !dbg !20
  %address.local842 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !20
  %address.local843 = lshr i64 %address.local832, 2, !dbg !12
  %address.local844 = and i64 %address.local843, 12, !dbg !12
  %address.local845 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local842, i64 %address.local844, !dbg !20
  store half %1413, ptr addrspace(1) %address.local845, align 2, !dbg !20
  %address.local846 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !20
  %address.local847 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local846, i64 %address.local844, !dbg !20
  %address.local848 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local847, i64 2, !dbg !20
  store half %1412, ptr addrspace(1) %address.local848, align 2, !dbg !20
  %address.local849 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !20
  %address.local850 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local849, i64 %address.local844, !dbg !20
  %address.local851 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local850, i64 4, !dbg !20
  store half %1411, ptr addrspace(1) %address.local851, align 2, !dbg !20
  %address.local852 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !20
  %address.local853 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local844, !dbg !20
  %address.local854 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local853, i64 6, !dbg !20
  store half %1410, ptr addrspace(1) %address.local854, align 2, !dbg !20
  %address.local855 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !20
  %address.local856 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local855, i64 %address.local844, !dbg !20
  %address.local857 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local856, i64 32, !dbg !20
  store half %1398, ptr addrspace(1) %address.local857, align 2, !dbg !20
  %address.local858 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !20
  %address.local859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local858, i64 %address.local844, !dbg !20
  %address.local860 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local859, i64 34, !dbg !20
  store half %1397, ptr addrspace(1) %address.local860, align 2, !dbg !20
  %address.local861 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !20
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local861, i64 %address.local844, !dbg !20
  %address.local863 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local862, i64 36, !dbg !20
  store half %1396, ptr addrspace(1) %address.local863, align 2, !dbg !20
  %address.local864 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !20
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local864, i64 %address.local844, !dbg !20
  %address.local866 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local865, i64 38, !dbg !20
  store half %1395, ptr addrspace(1) %address.local866, align 2, !dbg !20
  br label %1417, !dbg !20

1417:                                             ; preds = %1383, %1155
  br i1 true, label %1418, label %1438, !dbg !20

1418:                                             ; preds = %1417
  %1419 = insertelement <4 x float> poison, float %1353, i64 0, !dbg !18
  %1420 = insertelement <4 x float> %1419, float %1359, i64 1, !dbg !18
  %1421 = insertelement <4 x float> %1420, float %1354, i64 2, !dbg !18
  %1422 = insertelement <4 x float> %1421, float %1360, i64 3, !dbg !18
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !18
  %normalized74 = fmul <4 x float> %1422, %row.reciprocal73, !dbg !18
  %1423 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !19
  %1424 = extractelement <4 x half> %1423, i64 3, !dbg !19
  %1425 = extractelement <4 x half> %1423, i64 2, !dbg !19
  %1426 = extractelement <4 x half> %1423, i64 1, !dbg !19
  %1427 = extractelement <4 x half> %1423, i64 0, !dbg !19
  %1428 = insertelement <4 x float> poison, float %1246, i64 0, !dbg !18
  %1429 = insertelement <4 x float> %1428, float %1252, i64 1, !dbg !18
  %1430 = insertelement <4 x float> %1429, float %1247, i64 2, !dbg !18
  %1431 = insertelement <4 x float> %1430, float %1253, i64 3, !dbg !18
  %normalized75 = fmul <4 x float> %1431, %row.reciprocal73, !dbg !18
  %1432 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !19
  %1433 = extractelement <4 x half> %1432, i64 3, !dbg !19
  %1434 = extractelement <4 x half> %1432, i64 2, !dbg !19
  %1435 = extractelement <4 x half> %1432, i64 1, !dbg !19
  %1436 = extractelement <4 x half> %1432, i64 0, !dbg !19
  %1437 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread867 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local868 = zext nneg i32 %address.thread867 to i64, !dbg !12
  %address.local869 = add i64 %28, %address.local868, !dbg !20
  %address.local870 = lshr i64 %address.local868, 6, !dbg !12
  %address.local871 = shl nuw nsw i64 %address.local870, 4, !dbg !20
  %address.local872 = add i64 %address.local869, %address.local871, !dbg !20
  %address.local873 = and i64 %address.local868, 240, !dbg !20
  %address.local874 = sub i64 %address.local872, %address.local873, !dbg !20
  %address.local875 = add i64 %address.local874, %29, !dbg !20
  %address.local876 = shl nuw nsw i64 %address.local875, 7, !dbg !20
  %address.local877 = add nuw nsw i64 %1437, %address.local876, !dbg !20
  %address.local878 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !20
  %address.local879 = lshr i64 %address.local868, 2, !dbg !12
  %address.local880 = and i64 %address.local879, 12, !dbg !12
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local878, i64 %address.local880, !dbg !20
  %address.local882 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local881, i64 64, !dbg !20
  store half %1436, ptr addrspace(1) %address.local882, align 2, !dbg !20
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !20
  %address.local884 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local883, i64 %address.local880, !dbg !20
  %address.local885 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local884, i64 66, !dbg !20
  store half %1435, ptr addrspace(1) %address.local885, align 2, !dbg !20
  %address.local886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !20
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local886, i64 %address.local880, !dbg !20
  %address.local888 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local887, i64 68, !dbg !20
  store half %1434, ptr addrspace(1) %address.local888, align 2, !dbg !20
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !20
  %address.local890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local889, i64 %address.local880, !dbg !20
  %address.local891 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local890, i64 70, !dbg !20
  store half %1433, ptr addrspace(1) %address.local891, align 2, !dbg !20
  %address.local892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !20
  %address.local893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local892, i64 %address.local880, !dbg !20
  %address.local894 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local893, i64 96, !dbg !20
  store half %1427, ptr addrspace(1) %address.local894, align 2, !dbg !20
  %address.local895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !20
  %address.local896 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local895, i64 %address.local880, !dbg !20
  %address.local897 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local896, i64 98, !dbg !20
  store half %1426, ptr addrspace(1) %address.local897, align 2, !dbg !20
  %address.local898 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !20
  %address.local899 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local898, i64 %address.local880, !dbg !20
  %address.local900 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local899, i64 100, !dbg !20
  store half %1425, ptr addrspace(1) %address.local900, align 2, !dbg !20
  %address.local901 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !20
  %address.local902 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local901, i64 %address.local880, !dbg !20
  %address.local903 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local902, i64 102, !dbg !20
  store half %1424, ptr addrspace(1) %address.local903, align 2, !dbg !20
  br label %1438, !dbg !20

1438:                                             ; preds = %1418, %1417
  br i1 true, label %1439, label %1459, !dbg !20

1439:                                             ; preds = %1438
  %1440 = insertelement <4 x float> poison, float %1325, i64 0, !dbg !18
  %1441 = insertelement <4 x float> %1440, float %1331, i64 1, !dbg !18
  %1442 = insertelement <4 x float> %1441, float %1326, i64 2, !dbg !18
  %1443 = insertelement <4 x float> %1442, float %1332, i64 3, !dbg !18
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !18
  %normalized77 = fmul <4 x float> %1443, %row.reciprocal76, !dbg !18
  %1444 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !19
  %1445 = extractelement <4 x half> %1444, i64 3, !dbg !19
  %1446 = extractelement <4 x half> %1444, i64 2, !dbg !19
  %1447 = extractelement <4 x half> %1444, i64 1, !dbg !19
  %1448 = extractelement <4 x half> %1444, i64 0, !dbg !19
  %1449 = insertelement <4 x float> poison, float %1218, i64 0, !dbg !18
  %1450 = insertelement <4 x float> %1449, float %1224, i64 1, !dbg !18
  %1451 = insertelement <4 x float> %1450, float %1219, i64 2, !dbg !18
  %1452 = insertelement <4 x float> %1451, float %1225, i64 3, !dbg !18
  %normalized78 = fmul <4 x float> %1452, %row.reciprocal76, !dbg !18
  %1453 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !19
  %1454 = extractelement <4 x half> %1453, i64 3, !dbg !19
  %1455 = extractelement <4 x half> %1453, i64 2, !dbg !19
  %1456 = extractelement <4 x half> %1453, i64 1, !dbg !19
  %1457 = extractelement <4 x half> %1453, i64 0, !dbg !19
  %1458 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread904 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local905 = zext nneg i32 %address.thread904 to i64, !dbg !12
  %address.local906 = add i64 %28, %address.local905, !dbg !20
  %address.local907 = lshr i64 %address.local905, 6, !dbg !12
  %address.local908 = shl nuw nsw i64 %address.local907, 4, !dbg !20
  %address.local909 = add i64 %address.local906, %address.local908, !dbg !20
  %address.local910 = and i64 %address.local905, 240, !dbg !20
  %address.local911 = sub i64 %address.local909, %address.local910, !dbg !20
  %address.local912 = add i64 %address.local911, %29, !dbg !20
  %address.local913 = shl nuw nsw i64 %address.local912, 7, !dbg !20
  %address.local914 = add nuw nsw i64 %1458, %address.local913, !dbg !20
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !20
  %address.local916 = lshr i64 %address.local905, 2, !dbg !12
  %address.local917 = and i64 %address.local916, 12, !dbg !12
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local915, i64 %address.local917, !dbg !20
  %address.local919 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local918, i64 128, !dbg !20
  store half %1457, ptr addrspace(1) %address.local919, align 2, !dbg !20
  %address.local920 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !20
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local920, i64 %address.local917, !dbg !20
  %address.local922 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local921, i64 130, !dbg !20
  store half %1456, ptr addrspace(1) %address.local922, align 2, !dbg !20
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !20
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local923, i64 %address.local917, !dbg !20
  %address.local925 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local924, i64 132, !dbg !20
  store half %1455, ptr addrspace(1) %address.local925, align 2, !dbg !20
  %address.local926 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !20
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local926, i64 %address.local917, !dbg !20
  %address.local928 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local927, i64 134, !dbg !20
  store half %1454, ptr addrspace(1) %address.local928, align 2, !dbg !20
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !20
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local929, i64 %address.local917, !dbg !20
  %address.local931 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local930, i64 160, !dbg !20
  store half %1448, ptr addrspace(1) %address.local931, align 2, !dbg !20
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !20
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local932, i64 %address.local917, !dbg !20
  %address.local934 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local933, i64 162, !dbg !20
  store half %1447, ptr addrspace(1) %address.local934, align 2, !dbg !20
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !20
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local935, i64 %address.local917, !dbg !20
  %address.local937 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local936, i64 164, !dbg !20
  store half %1446, ptr addrspace(1) %address.local937, align 2, !dbg !20
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !20
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local938, i64 %address.local917, !dbg !20
  %address.local940 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local939, i64 166, !dbg !20
  store half %1445, ptr addrspace(1) %address.local940, align 2, !dbg !20
  br label %1459, !dbg !20

1459:                                             ; preds = %1439, %1438
  br i1 true, label %1460, label %1480, !dbg !20

1460:                                             ; preds = %1459
  %1461 = insertelement <4 x float> poison, float %1297, i64 0, !dbg !18
  %1462 = insertelement <4 x float> %1461, float %1303, i64 1, !dbg !18
  %1463 = insertelement <4 x float> %1462, float %1298, i64 2, !dbg !18
  %1464 = insertelement <4 x float> %1463, float %1304, i64 3, !dbg !18
  %row.reciprocal79 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !18
  %normalized80 = fmul <4 x float> %1464, %row.reciprocal79, !dbg !18
  %1465 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !19
  %1466 = extractelement <4 x half> %1465, i64 3, !dbg !19
  %1467 = extractelement <4 x half> %1465, i64 2, !dbg !19
  %1468 = extractelement <4 x half> %1465, i64 1, !dbg !19
  %1469 = extractelement <4 x half> %1465, i64 0, !dbg !19
  %1470 = insertelement <4 x float> poison, float %1190, i64 0, !dbg !18
  %1471 = insertelement <4 x float> %1470, float %1196, i64 1, !dbg !18
  %1472 = insertelement <4 x float> %1471, float %1191, i64 2, !dbg !18
  %1473 = insertelement <4 x float> %1472, float %1197, i64 3, !dbg !18
  %normalized81 = fmul <4 x float> %1473, %row.reciprocal79, !dbg !18
  %1474 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !19
  %1475 = extractelement <4 x half> %1474, i64 3, !dbg !19
  %1476 = extractelement <4 x half> %1474, i64 2, !dbg !19
  %1477 = extractelement <4 x half> %1474, i64 1, !dbg !19
  %1478 = extractelement <4 x half> %1474, i64 0, !dbg !19
  %1479 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread941 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local942 = zext nneg i32 %address.thread941 to i64, !dbg !12
  %address.local943 = add i64 %28, %address.local942, !dbg !20
  %address.local944 = lshr i64 %address.local942, 6, !dbg !12
  %address.local945 = shl nuw nsw i64 %address.local944, 4, !dbg !20
  %address.local946 = add i64 %address.local943, %address.local945, !dbg !20
  %address.local947 = and i64 %address.local942, 240, !dbg !20
  %address.local948 = sub i64 %address.local946, %address.local947, !dbg !20
  %address.local949 = add i64 %address.local948, %29, !dbg !20
  %address.local950 = shl nuw nsw i64 %address.local949, 7, !dbg !20
  %address.local951 = add nuw nsw i64 %1479, %address.local950, !dbg !20
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !20
  %address.local953 = lshr i64 %address.local942, 2, !dbg !12
  %address.local954 = and i64 %address.local953, 12, !dbg !12
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local952, i64 %address.local954, !dbg !20
  %address.local956 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local955, i64 192, !dbg !20
  store half %1478, ptr addrspace(1) %address.local956, align 2, !dbg !20
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !20
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local957, i64 %address.local954, !dbg !20
  %address.local959 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local958, i64 194, !dbg !20
  store half %1477, ptr addrspace(1) %address.local959, align 2, !dbg !20
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !20
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local960, i64 %address.local954, !dbg !20
  %address.local962 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local961, i64 196, !dbg !20
  store half %1476, ptr addrspace(1) %address.local962, align 2, !dbg !20
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !20
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local963, i64 %address.local954, !dbg !20
  %address.local965 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local964, i64 198, !dbg !20
  store half %1475, ptr addrspace(1) %address.local965, align 2, !dbg !20
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !20
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local966, i64 %address.local954, !dbg !20
  %address.local968 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local967, i64 224, !dbg !20
  store half %1469, ptr addrspace(1) %address.local968, align 2, !dbg !20
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !20
  %address.local970 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local954, !dbg !20
  %address.local971 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local970, i64 226, !dbg !20
  store half %1468, ptr addrspace(1) %address.local971, align 2, !dbg !20
  %address.local972 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !20
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local972, i64 %address.local954, !dbg !20
  %address.local974 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local973, i64 228, !dbg !20
  store half %1467, ptr addrspace(1) %address.local974, align 2, !dbg !20
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !20
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local975, i64 %address.local954, !dbg !20
  %address.local977 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local976, i64 230, !dbg !20
  store half %1466, ptr addrspace(1) %address.local977, align 2, !dbg !20
  br label %1480, !dbg !20

1480:                                             ; preds = %1460, %1459
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #7

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!37}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/group8-reverse-k128-swizzle")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!9 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 70, column: 22, scope: !4)
!12 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 77, column: 19, scope: !4)
!14 = !DILocation(line: 102, column: 23, scope: !4)
!15 = !DILocation(line: 71, column: 22, scope: !4)
!16 = !DILocation(line: 73, column: 26, scope: !4)
!17 = !DILocation(line: 101, column: 20, scope: !4)
!18 = !DILocation(line: 107, column: 23, scope: !4)
!19 = !DILocation(line: 108, column: 19, scope: !4)
!20 = !DILocation(line: 109, column: 9, scope: !4)
!22 = !DILocation(line: 79, column: 24, scope: !4)
!23 = !DILocation(line: 80, column: 23, scope: !4)
!24 = !DILocation(line: 81, column: 26, scope: !4)
!25 = !DILocation(line: 88, column: 21, scope: !4)
!26 = !DILocation(line: 89, column: 20, scope: !4)
!27 = !DILocation(line: 90, column: 22, scope: !4)
!28 = !DILocation(line: 91, column: 23, scope: !4)
!29 = !DILocation(line: 92, column: 20, scope: !4)
!30 = !DILocation(line: 93, column: 22, scope: !4)
!31 = !DILocation(line: 94, column: 22, scope: !4)
!32 = !DILocation(line: 95, column: 20, scope: !4)
!33 = !DILocation(line: 96, column: 16, scope: !4)
!34 = !DILocation(line: 103, column: 22, scope: !4)
!36 = !{i32 256, i32 1, i32 1}
!37 = !{i32 2, !"Debug Info Version", i32 3}
!38 = !{i32 0, i32 256}
