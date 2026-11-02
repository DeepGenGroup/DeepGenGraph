; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/online/session-20261026-101601-tehle7yk/tuning/run-20261026-101601-inpooz4_/input-0002-test_input_1x32x4096x128-128-128-32-online/grid-000003/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/online/session-20261026-101601-tehle7yk/tuning/run-20261026-101601-inpooz4_/input-0002-test_input_1x32x4096x128-128-128-32-online/grid-000003/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
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
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !39
  %35 = zext nneg i32 %34 to i64, !dbg !12
  %36 = lshr i64 %35, 6, !dbg !12
  %37 = or disjoint i64 %19, 127, !dbg !13
  %38 = add i64 %19, 128, !dbg !13
  %39 = icmp uge i64 %38, %37, !dbg !13
  %40 = sub nsw i64 32, %14, !dbg !15
  %41 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %42 = lshr i64 %35, 2, !dbg !17
  %43 = shl nuw nsw i64 %35, 2, !dbg !17
  %44 = mul nsw i64 %42, -16, !dbg !17
  %45 = add nsw i64 %43, %44, !dbg !17
  %46 = and i32 %34, 1, !dbg !17
  %index.compact279 = zext nneg i32 %46 to i64, !dbg !17
  %47 = or disjoint i64 %45, %index.compact279, !dbg !17
  %48 = add nuw nsw i64 %47, 1, !dbg !17
  %49 = sdiv i64 %48, 2, !dbg !17
  %50 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %51 = getelementptr i8, ptr addrspace(1) %50, i64 64, !dbg !11
  %52 = shl i64 %15, 7, !dbg !16
  %53 = and i64 %52, 3968, !dbg !16
  %54 = shl nsw i64 %18, 12, !dbg !16
  %55 = or disjoint i64 %53, %54, !dbg !16
  %56 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %57 = getelementptr i8, ptr addrspace(1) %56, i64 128, !dbg !11
  %58 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %59 = getelementptr i8, ptr addrspace(1) %58, i64 192, !dbg !11
  br label %60, !dbg !7

60:                                               ; preds = %848, %4
  %61 = phi i64 [ %1779, %848 ], [ 0, %4 ], !dbg !7
  %62 = phi [2 x <8 x float>] [ %1764, %848 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi [2 x <1 x float>] [ %1370, %848 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi [2 x <1 x float>] [ %1766, %848 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %65 = phi [2 x <8 x float>] [ %1770, %848 ], [ zeroinitializer, %4 ], !dbg !7
  %66 = phi [2 x <8 x float>] [ %1774, %848 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = phi [2 x <8 x float>] [ %1778, %848 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = icmp slt i64 %61, %40, !dbg !7
  br i1 %68, label %69, label %1780, !dbg !7

69:                                               ; preds = %60
  %70 = shl nsw i64 %61, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local336 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local337 = and i64 %address.local336, 1536, !dbg !11
  %address.local338 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local337, !dbg !11
  %address.local339 = ptrtoint ptr addrspace(3) %address.local338 to i64, !dbg !11
  %address.local340 = trunc i64 %address.local339 to i32, !dbg !11
  %71 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !11
  %72 = zext i32 %71 to i64, !dbg !11
  %73 = inttoptr i64 %72 to ptr addrspace(3), !dbg !11
  %address.local341 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local342 = and i64 %address.local341, 16256, !dbg !11
  %address.local343 = lshr i64 %address.local, 4, !dbg !11
  %address.local344 = and i64 %address.local343, 8, !dbg !11
  %address.local345 = or disjoint i64 %address.local342, %address.local344, !dbg !11
  %address.local346 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local345, !dbg !11
  %74 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %75 = ptrtoint ptr addrspace(1) %address.local346 to i64, !dbg !11
  %76 = sub i64 %75, %74, !dbg !11
  %dma.buffer.offset1524 = trunc i64 %76 to i32, !dbg !11
  %77 = trunc i64 %74 to i32, !dbg !11
  %78 = insertelement <4 x i32> poison, i32 %77, i64 0, !dbg !11
  %79 = lshr i64 %74, 32, !dbg !11
  %80 = trunc i64 %79 to i32, !dbg !11
  %81 = insertelement <4 x i32> %78, i32 %80, i64 1, !dbg !11
  %82 = insertelement <4 x i32> %81, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1525 = insertelement <4 x i32> %82, i32 131072, i64 3, !dbg !11
  %83 = ptrtoint ptr addrspace(3) %73 to i32, !dbg !11
  %dma.buffer.lds1526 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !11
  %84 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1526, i32 %dma.buffer.offset1524, <4 x i32> %dma.buffer.rsrc1525) #10, !dbg !11
  %address.local347 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local348 = shl nuw nsw i64 %address.local347, 3, !dbg !11
  %address.local349 = and i64 %address.local348, 3584, !dbg !11
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local349, !dbg !11
  %address.local351 = ptrtoint ptr addrspace(3) %address.local350 to i64, !dbg !11
  %address.local352 = trunc i64 %address.local351 to i32, !dbg !11
  %85 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !11
  %86 = zext i32 %85 to i64, !dbg !11
  %87 = inttoptr i64 %86 to ptr addrspace(3), !dbg !11
  %address.local353 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local354 = and i64 %address.local353, 16256, !dbg !11
  %address.local355 = lshr i64 %address.local347, 4, !dbg !11
  %address.local356 = and i64 %address.local355, 24, !dbg !11
  %address.local357 = or disjoint i64 %address.local354, %address.local356, !dbg !11
  %address.local358 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local357, !dbg !11
  %88 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %89 = ptrtoint ptr addrspace(1) %address.local358 to i64, !dbg !11
  %90 = sub i64 %89, %88, !dbg !11
  %dma.buffer.offset1521 = trunc i64 %90 to i32, !dbg !11
  %91 = trunc i64 %88 to i32, !dbg !11
  %92 = insertelement <4 x i32> poison, i32 %91, i64 0, !dbg !11
  %93 = lshr i64 %88, 32, !dbg !11
  %94 = trunc i64 %93 to i32, !dbg !11
  %95 = insertelement <4 x i32> %92, i32 %94, i64 1, !dbg !11
  %96 = insertelement <4 x i32> %95, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1522 = insertelement <4 x i32> %96, i32 131072, i64 3, !dbg !11
  %97 = ptrtoint ptr addrspace(3) %87 to i32, !dbg !11
  %dma.buffer.lds1523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %97), !dbg !11
  %98 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1523, i32 %dma.buffer.offset1521, <4 x i32> %dma.buffer.rsrc1522) #10, !dbg !11
  %99 = lshr i64 %61, 12, !dbg !16
  %100 = add nuw nsw i64 %99, %15, !dbg !16
  %101 = lshr i64 %61, 5, !dbg !16
  %102 = and i64 %101, 127, !dbg !16
  %103 = shl i64 %100, 19, !dbg !16
  %104 = and i64 %103, 16252928, !dbg !16
  %105 = shl i64 %101, 19, !dbg !16
  %106 = shl i64 %61, 14, !dbg !16
  %107 = sub i64 %106, %105, !dbg !16
  %108 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %104, !dbg !16
  %109 = getelementptr [2 x i8], ptr addrspace(1) %108, i64 %102, !dbg !16
  %110 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %107, !dbg !16
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local337, !dbg !16
  %address.local360 = ptrtoint ptr addrspace(3) %address.local359 to i64, !dbg !16
  %address.local361 = trunc i64 %address.local360 to i32, !dbg !16
  %111 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local361), !dbg !16
  %112 = zext i32 %111 to i64, !dbg !16
  %113 = inttoptr i64 %112 to ptr addrspace(3), !dbg !16
  %address.local362 = lshr i64 %address.local, 4, !dbg !16
  %address.local363 = and i64 %address.local362, 8, !dbg !16
  %address.local364 = and i64 %address.local, 123, !dbg !16
  %address.local365 = lshr i64 %address.local, 2, !dbg !16
  %address.local366 = and i64 %address.local365, 4, !dbg !16
  %address.local367 = add nuw nsw i64 %address.local366, %address.local, !dbg !16
  %address.local368 = and i64 %address.local367, 4, !dbg !16
  %address.local369 = or disjoint i64 %address.local364, %address.local368, !dbg !16
  %address.local370 = shl nuw nsw i64 %address.local369, 7, !dbg !16
  %address.local371 = or disjoint i64 %address.local363, %address.local370, !dbg !16
  %address.local372 = getelementptr [2 x i8], ptr addrspace(1) %110, i64 %address.local371, !dbg !16
  %114 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %115 = ptrtoint ptr addrspace(1) %address.local372 to i64, !dbg !16
  %116 = sub i64 %115, %114, !dbg !16
  %dma.buffer.offset1518 = trunc i64 %116 to i32, !dbg !16
  %117 = trunc i64 %114 to i32, !dbg !16
  %118 = insertelement <4 x i32> poison, i32 %117, i64 0, !dbg !16
  %119 = lshr i64 %114, 32, !dbg !16
  %120 = trunc i64 %119 to i32, !dbg !16
  %121 = insertelement <4 x i32> %118, i32 %120, i64 1, !dbg !16
  %122 = insertelement <4 x i32> %121, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1519 = insertelement <4 x i32> %122, i32 131072, i64 3, !dbg !16
  %123 = ptrtoint ptr addrspace(3) %113 to i32, !dbg !16
  %dma.buffer.lds1520 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %123), !dbg !16
  %124 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1520, i32 %dma.buffer.offset1518, <4 x i32> %dma.buffer.rsrc1519) #10, !dbg !16
  %address.local373 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local349, !dbg !16
  %address.local374 = ptrtoint ptr addrspace(3) %address.local373 to i64, !dbg !16
  %address.local375 = trunc i64 %address.local374 to i32, !dbg !16
  %125 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local375), !dbg !16
  %126 = zext i32 %125 to i64, !dbg !16
  %127 = inttoptr i64 %126 to ptr addrspace(3), !dbg !16
  %address.local376 = lshr i64 %address.local347, 4, !dbg !16
  %address.local377 = and i64 %address.local376, 24, !dbg !16
  %address.local378 = and i64 %address.local, 123, !dbg !16
  %address.local379 = lshr i64 %address.local, 2, !dbg !16
  %address.local380 = and i64 %address.local379, 4, !dbg !16
  %address.local381 = add nuw nsw i64 %address.local380, %address.local, !dbg !16
  %address.local382 = and i64 %address.local381, 4, !dbg !16
  %address.local383 = or disjoint i64 %address.local378, %address.local382, !dbg !16
  %address.local384 = shl nuw nsw i64 %address.local383, 7, !dbg !16
  %address.local385 = or disjoint i64 %address.local377, %address.local384, !dbg !16
  %address.local386 = getelementptr [2 x i8], ptr addrspace(1) %110, i64 %address.local385, !dbg !16
  %128 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %129 = ptrtoint ptr addrspace(1) %address.local386 to i64, !dbg !16
  %130 = sub i64 %129, %128, !dbg !16
  %dma.buffer.offset1515 = trunc i64 %130 to i32, !dbg !16
  %131 = trunc i64 %128 to i32, !dbg !16
  %132 = insertelement <4 x i32> poison, i32 %131, i64 0, !dbg !16
  %133 = lshr i64 %128, 32, !dbg !16
  %134 = trunc i64 %133 to i32, !dbg !16
  %135 = insertelement <4 x i32> %132, i32 %134, i64 1, !dbg !16
  %136 = insertelement <4 x i32> %135, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1516 = insertelement <4 x i32> %136, i32 131072, i64 3, !dbg !16
  %137 = ptrtoint ptr addrspace(3) %127 to i32, !dbg !16
  %dma.buffer.lds1517 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %137), !dbg !16
  %138 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1517, i32 %dma.buffer.offset1515, <4 x i32> %dma.buffer.rsrc1516) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local387 = lshr i32 %address.thread, 2, !dbg !12
  %address.local388 = and i32 %address.local387, 4, !dbg !12
  %address.local389 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local390 = and i32 %address.local389, 384, !dbg !12
  %address.local391 = or disjoint i32 %address.local388, %address.local390, !dbg !12
  %address.local392 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local393 = and i32 %address.local392, 120, !dbg !12
  %address.local394 = or disjoint i32 %address.local391, %address.local393, !dbg !12
  %address.local395 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local396 = and i32 %address.local395, 1024, !dbg !12
  %address.local397 = or disjoint i32 %address.local394, %address.local396, !dbg !12
  %address.local398 = zext nneg i32 %address.local397 to i64, !dbg !12
  %address.local399 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local398, !dbg !12
  %139 = load <4 x half>, ptr addrspace(3) %address.local399, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local399, i64 4096, !dbg !12
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %lds.const.gep1335 = getelementptr i8, ptr addrspace(3) %address.local399, i64 1024, !dbg !12
  %141 = load <4 x half>, ptr addrspace(3) %lds.const.gep1335, align 8, !dbg !12
  %142 = fmul <4 x half> %139, splat (half 0xH3015), !dbg !12
  %lds.const.gep1336 = getelementptr i8, ptr addrspace(3) %address.local399, i64 5120, !dbg !12
  %143 = load <4 x half>, ptr addrspace(3) %lds.const.gep1336, align 8, !dbg !12
  %144 = fmul <4 x half> %140, splat (half 0xH3015), !dbg !12
  %145 = fmul <4 x half> %141, splat (half 0xH3015), !dbg !12
  %146 = fmul <4 x half> %143, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local406 = mul nuw nsw i64 %address.local, 160, !dbg !17
  %address.local407 = lshr i64 %address.local, 2, !dbg !17
  %address.local408 = mul nsw i64 %address.local407, -640, !dbg !17
  %address.local409 = add nsw i64 %address.local406, %address.local408, !dbg !17
  %address.local410 = shl nuw nsw i64 %address.local, 1, !dbg !17
  %address.local411 = and i64 %address.local410, 24, !dbg !17
  %address.local412 = or disjoint i64 %address.local409, %address.local411, !dbg !17
  %address.local413 = shl i64 %address.local, 62, !dbg !17
  %address.local414 = ashr exact i64 %address.local413, 57, !dbg !17
  %address.local415 = and i64 %address.local414, -64, !dbg !17
  %address.local416 = add nsw i64 %address.local412, %address.local415, !dbg !17
  %address.local417 = lshr i64 %address.local, 2, !dbg !12
  %address.local418 = and i64 %address.local417, 12, !dbg !12
  %address.local419 = add nsw i64 %address.local416, %address.local418, !dbg !17
  %address.local420 = and i64 %address.local, 63, !dbg !12
  %address.local421 = lshr i64 %address.local420, 5, !dbg !12
  %address.local422 = mul nuw nsw i64 %address.local421, 1016, !dbg !12
  %address.local423 = add i64 %address.local419, %address.local422, !dbg !17
  %address.local424 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local423, !dbg !17
  %147 = load <4 x half>, ptr addrspace(3) %address.local424, align 8, !dbg !17
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1337 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4096, !dbg !17
  %149 = load <4 x half>, ptr addrspace(3) %lds.const.gep1337, align 8, !dbg !17
  %address.local435 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local436 = and i32 %address.local435, 24, !dbg !17
  %address.local437 = shl nuw nsw i32 %address.thread, 8, !dbg !17
  %address.local438 = and i32 %address.local437, 768, !dbg !17
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !17
  %address.local440 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local441 = and i32 %address.local440, 96, !dbg !17
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !17
  %address.local443 = zext nneg i32 %address.local442 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %49) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local444 = lshr i32 %address.thread, 1, !dbg !17
  %address.local445 = and i32 %address.local444, 1, !dbg !17
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !17
  %address.local447 = add nsw i64 %address.base, %address.local446, !dbg !17
  %address.local448 = shl i64 %address.local447, 6, !dbg !17
  %address.local449 = sub i64 %address.local443, %address.local448, !dbg !17
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local449, !dbg !17
  %address.local451 = getelementptr [2 x i8], ptr addrspace(3) %address.local450, i64 %address.local418, !dbg !17
  %address.local452 = getelementptr [2 x i8], ptr addrspace(3) %address.local451, i64 %address.local422, !dbg !17
  %address.local453 = getelementptr i8, ptr addrspace(3) %address.local452, i64 64, !dbg !17
  %150 = load <4 x half>, ptr addrspace(3) %address.local453, align 8, !dbg !17
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %149, <4 x float> %148) #9, !dbg !17
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1338 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4096, !dbg !17
  %153 = load <4 x half>, ptr addrspace(3) %lds.const.gep1338, align 8, !dbg !17
  %lds.const.gep1339 = getelementptr i8, ptr addrspace(3) %address.local424, i64 128, !dbg !17
  %154 = load <4 x half>, ptr addrspace(3) %lds.const.gep1339, align 8, !dbg !17
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %153, <4 x float> %152) #9, !dbg !17
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1340 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4224, !dbg !17
  %157 = load <4 x half>, ptr addrspace(3) %lds.const.gep1340, align 8, !dbg !17
  %lds.const.gep1341 = getelementptr i8, ptr addrspace(3) %address.local453, i64 128, !dbg !17
  %158 = load <4 x half>, ptr addrspace(3) %lds.const.gep1341, align 8, !dbg !17
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %157, <4 x float> %156) #9, !dbg !17
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1342 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4224, !dbg !17
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1342, align 8, !dbg !17
  %lds.const.gep1343 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1024, !dbg !17
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep1343, align 8, !dbg !17
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %161, <4 x float> %160) #9, !dbg !17
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1344 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5120, !dbg !17
  %165 = load <4 x half>, ptr addrspace(3) %lds.const.gep1344, align 8, !dbg !17
  %lds.const.gep1345 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1024, !dbg !17
  %166 = load <4 x half>, ptr addrspace(3) %lds.const.gep1345, align 8, !dbg !17
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %165, <4 x float> %164) #9, !dbg !17
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %166, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1346 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5120, !dbg !17
  %169 = load <4 x half>, ptr addrspace(3) %lds.const.gep1346, align 8, !dbg !17
  %lds.const.gep1347 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1152, !dbg !17
  %170 = load <4 x half>, ptr addrspace(3) %lds.const.gep1347, align 8, !dbg !17
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %169, <4 x float> %168) #9, !dbg !17
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %170, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1348 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5248, !dbg !17
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1348, align 8, !dbg !17
  %lds.const.gep1349 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1152, !dbg !17
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1349, align 8, !dbg !17
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %173, <4 x float> %172) #9, !dbg !17
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %174, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1350 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5248, !dbg !17
  %177 = load <4 x half>, ptr addrspace(3) %lds.const.gep1350, align 8, !dbg !17
  %178 = load <4 x half>, ptr addrspace(3) %address.local424, align 8, !dbg !17
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %177, <4 x float> %176) #9, !dbg !17
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %178, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1351 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4096, !dbg !17
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1351, align 8, !dbg !17
  %182 = load <4 x half>, ptr addrspace(3) %address.local453, align 8, !dbg !17
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %181, <4 x float> %180) #9, !dbg !17
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %182, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1352 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4096, !dbg !17
  %185 = load <4 x half>, ptr addrspace(3) %lds.const.gep1352, align 8, !dbg !17
  %lds.const.gep1353 = getelementptr i8, ptr addrspace(3) %address.local424, i64 128, !dbg !17
  %186 = load <4 x half>, ptr addrspace(3) %lds.const.gep1353, align 8, !dbg !17
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %185, <4 x float> %184) #9, !dbg !17
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %186, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1354 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4224, !dbg !17
  %189 = load <4 x half>, ptr addrspace(3) %lds.const.gep1354, align 8, !dbg !17
  %lds.const.gep1355 = getelementptr i8, ptr addrspace(3) %address.local453, i64 128, !dbg !17
  %190 = load <4 x half>, ptr addrspace(3) %lds.const.gep1355, align 8, !dbg !17
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %189, <4 x float> %188) #9, !dbg !17
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %190, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1356 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4224, !dbg !17
  %193 = load <4 x half>, ptr addrspace(3) %lds.const.gep1356, align 8, !dbg !17
  %lds.const.gep1357 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1024, !dbg !17
  %194 = load <4 x half>, ptr addrspace(3) %lds.const.gep1357, align 8, !dbg !17
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %193, <4 x float> %192) #9, !dbg !17
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %194, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1358 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5120, !dbg !17
  %197 = load <4 x half>, ptr addrspace(3) %lds.const.gep1358, align 8, !dbg !17
  %lds.const.gep1359 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1024, !dbg !17
  %198 = load <4 x half>, ptr addrspace(3) %lds.const.gep1359, align 8, !dbg !17
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %197, <4 x float> %196) #9, !dbg !17
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %198, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1360 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5120, !dbg !17
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep1360, align 8, !dbg !17
  %lds.const.gep1361 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1152, !dbg !17
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep1361, align 8, !dbg !17
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %201, <4 x float> %200) #9, !dbg !17
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %202, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1362 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5248, !dbg !17
  %205 = load <4 x half>, ptr addrspace(3) %lds.const.gep1362, align 8, !dbg !17
  %lds.const.gep1363 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1152, !dbg !17
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep1363, align 8, !dbg !17
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %205, <4 x float> %204) #9, !dbg !17
  %lds.const.gep1364 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5248, !dbg !17
  %208 = load <4 x half>, ptr addrspace(3) %lds.const.gep1364, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %206, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %208, <4 x float> %209) #9, !dbg !17
  %211 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !11
  %212 = zext i32 %211 to i64, !dbg !11
  %213 = inttoptr i64 %212 to ptr addrspace(3), !dbg !11
  %address.local762 = getelementptr [2 x i8], ptr addrspace(1) %51, i64 %address.local345, !dbg !11
  %214 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %215 = ptrtoint ptr addrspace(1) %address.local762 to i64, !dbg !11
  %216 = sub i64 %215, %214, !dbg !11
  %dma.buffer.offset1512 = trunc i64 %216 to i32, !dbg !11
  %217 = trunc i64 %214 to i32, !dbg !11
  %218 = insertelement <4 x i32> poison, i32 %217, i64 0, !dbg !11
  %219 = lshr i64 %214, 32, !dbg !11
  %220 = trunc i64 %219 to i32, !dbg !11
  %221 = insertelement <4 x i32> %218, i32 %220, i64 1, !dbg !11
  %222 = insertelement <4 x i32> %221, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1513 = insertelement <4 x i32> %222, i32 131072, i64 3, !dbg !11
  %223 = ptrtoint ptr addrspace(3) %213 to i32, !dbg !11
  %dma.buffer.lds1514 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %223), !dbg !11
  %224 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1514, i32 %dma.buffer.offset1512, <4 x i32> %dma.buffer.rsrc1513) #10, !dbg !11
  %225 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !11
  %226 = zext i32 %225 to i64, !dbg !11
  %227 = inttoptr i64 %226 to ptr addrspace(3), !dbg !11
  %address.local763 = getelementptr [2 x i8], ptr addrspace(1) %51, i64 %address.local357, !dbg !11
  %228 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %229 = ptrtoint ptr addrspace(1) %address.local763 to i64, !dbg !11
  %230 = sub i64 %229, %228, !dbg !11
  %dma.buffer.offset1509 = trunc i64 %230 to i32, !dbg !11
  %231 = trunc i64 %228 to i32, !dbg !11
  %232 = insertelement <4 x i32> poison, i32 %231, i64 0, !dbg !11
  %233 = lshr i64 %228, 32, !dbg !11
  %234 = trunc i64 %233 to i32, !dbg !11
  %235 = insertelement <4 x i32> %232, i32 %234, i64 1, !dbg !11
  %236 = insertelement <4 x i32> %235, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1510 = insertelement <4 x i32> %236, i32 131072, i64 3, !dbg !11
  %237 = ptrtoint ptr addrspace(3) %227 to i32, !dbg !11
  %dma.buffer.lds1511 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %237), !dbg !11
  %238 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1511, i32 %dma.buffer.offset1509, <4 x i32> %dma.buffer.rsrc1510) #10, !dbg !11
  %239 = add nuw i64 %61, 1024, !dbg !16
  %240 = icmp slt i64 %239, 0, !dbg !16
  %241 = sub nuw i64 -1025, %61, !dbg !16
  %242 = select i1 %240, i64 %241, i64 %239, !dbg !16
  %243 = sdiv i64 %242, 4096, !dbg !16
  %.lobit37 = ashr i64 %239, 63, !dbg !16
  %244 = xor i64 %243, %.lobit37, !dbg !16
  %245 = add nsw i64 %244, %15, !dbg !16
  %246 = add nsw i64 %101, %55, !dbg !16
  %247 = add nsw i64 %246, 32, !dbg !16
  %248 = icmp slt i64 %246, -32, !dbg !16
  %249 = sub nsw i64 -33, %246, !dbg !16
  %250 = select i1 %248, i64 %249, i64 %247, !dbg !16
  %251 = sdiv i64 %250, 128, !dbg !16
  %252 = sext i1 %248 to i64, !dbg !16
  %253 = xor i64 %251, %252, !dbg !16
  %254 = shl nsw i64 %253, 7, !dbg !16
  %255 = sub i64 %55, %254, !dbg !16
  %256 = shl i64 %245, 19, !dbg !16
  %257 = and i64 %256, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %258 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %257, !dbg !16
  %259 = getelementptr [2 x i8], ptr addrspace(1) %258, i64 %101, !dbg !16
  %260 = getelementptr [2 x i8], ptr addrspace(1) %259, i64 %255, !dbg !16
  %261 = getelementptr i8, ptr addrspace(1) %260, i64 64, !dbg !16
  %262 = getelementptr [2 x i8], ptr addrspace(1) %261, i64 %107, !dbg !16
  %263 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local361), !dbg !16
  %264 = zext i32 %263 to i64, !dbg !16
  %265 = inttoptr i64 %264 to ptr addrspace(3), !dbg !16
  %address.local764 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %address.local371, !dbg !16
  %266 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %267 = ptrtoint ptr addrspace(1) %address.local764 to i64, !dbg !16
  %268 = sub i64 %267, %266, !dbg !16
  %dma.buffer.offset1506 = trunc i64 %268 to i32, !dbg !16
  %269 = trunc i64 %266 to i32, !dbg !16
  %270 = insertelement <4 x i32> poison, i32 %269, i64 0, !dbg !16
  %271 = lshr i64 %266, 32, !dbg !16
  %272 = trunc i64 %271 to i32, !dbg !16
  %273 = insertelement <4 x i32> %270, i32 %272, i64 1, !dbg !16
  %274 = insertelement <4 x i32> %273, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1507 = insertelement <4 x i32> %274, i32 131072, i64 3, !dbg !16
  %275 = ptrtoint ptr addrspace(3) %265 to i32, !dbg !16
  %dma.buffer.lds1508 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %275), !dbg !16
  %276 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1508, i32 %dma.buffer.offset1506, <4 x i32> %dma.buffer.rsrc1507) #10, !dbg !16
  %277 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local375), !dbg !16
  %278 = zext i32 %277 to i64, !dbg !16
  %279 = inttoptr i64 %278 to ptr addrspace(3), !dbg !16
  %address.local765 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %address.local385, !dbg !16
  %280 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %281 = ptrtoint ptr addrspace(1) %address.local765 to i64, !dbg !16
  %282 = sub i64 %281, %280, !dbg !16
  %dma.buffer.offset1503 = trunc i64 %282 to i32, !dbg !16
  %283 = trunc i64 %280 to i32, !dbg !16
  %284 = insertelement <4 x i32> poison, i32 %283, i64 0, !dbg !16
  %285 = lshr i64 %280, 32, !dbg !16
  %286 = trunc i64 %285 to i32, !dbg !16
  %287 = insertelement <4 x i32> %284, i32 %286, i64 1, !dbg !16
  %288 = insertelement <4 x i32> %287, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1504 = insertelement <4 x i32> %288, i32 131072, i64 3, !dbg !16
  %289 = ptrtoint ptr addrspace(3) %279 to i32, !dbg !16
  %dma.buffer.lds1505 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %289), !dbg !16
  %290 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1505, i32 %dma.buffer.offset1503, <4 x i32> %dma.buffer.rsrc1504) #10, !dbg !16
  %291 = load <4 x half>, ptr addrspace(3) %address.local399, align 8, !dbg !12
  %lds.const.gep1365 = getelementptr i8, ptr addrspace(3) %address.local399, i64 4096, !dbg !12
  %292 = load <4 x half>, ptr addrspace(3) %lds.const.gep1365, align 8, !dbg !12
  %lds.const.gep1366 = getelementptr i8, ptr addrspace(3) %address.local399, i64 1024, !dbg !12
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1366, align 8, !dbg !12
  %294 = fmul <4 x half> %291, splat (half 0xH3015), !dbg !12
  %lds.const.gep1367 = getelementptr i8, ptr addrspace(3) %address.local399, i64 5120, !dbg !12
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1367, align 8, !dbg !12
  %296 = fmul <4 x half> %292, splat (half 0xH3015), !dbg !12
  %297 = fmul <4 x half> %293, splat (half 0xH3015), !dbg !12
  %298 = fmul <4 x half> %295, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %299 = load <4 x half>, ptr addrspace(3) %address.local424, align 8, !dbg !17
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %299, <4 x float> %151) #9, !dbg !17
  %lds.const.gep1368 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4096, !dbg !17
  %301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1368, align 8, !dbg !17
  %302 = load <4 x half>, ptr addrspace(3) %address.local453, align 8, !dbg !17
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %301, <4 x float> %300) #9, !dbg !17
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %302, <4 x float> %155) #9, !dbg !17
  %lds.const.gep1369 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4096, !dbg !17
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1369, align 8, !dbg !17
  %lds.const.gep1370 = getelementptr i8, ptr addrspace(3) %address.local424, i64 128, !dbg !17
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1370, align 8, !dbg !17
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %305, <4 x float> %304) #9, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %306, <4 x float> %159) #9, !dbg !17
  %lds.const.gep1371 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4224, !dbg !17
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1371, align 8, !dbg !17
  %lds.const.gep1372 = getelementptr i8, ptr addrspace(3) %address.local453, i64 128, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep1372, align 8, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %309, <4 x float> %308) #9, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %310, <4 x float> %163) #9, !dbg !17
  %lds.const.gep1373 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4224, !dbg !17
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1373, align 8, !dbg !17
  %lds.const.gep1374 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1024, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1374, align 8, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %313, <4 x float> %312) #9, !dbg !17
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %314, <4 x float> %167) #9, !dbg !17
  %lds.const.gep1375 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5120, !dbg !17
  %317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1375, align 8, !dbg !17
  %lds.const.gep1376 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1024, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1376, align 8, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %317, <4 x float> %316) #9, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %318, <4 x float> %171) #9, !dbg !17
  %lds.const.gep1377 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5120, !dbg !17
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1377, align 8, !dbg !17
  %lds.const.gep1378 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1152, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1378, align 8, !dbg !17
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %321, <4 x float> %320) #9, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %322, <4 x float> %175) #9, !dbg !17
  %lds.const.gep1379 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5248, !dbg !17
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1379, align 8, !dbg !17
  %lds.const.gep1380 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1152, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1380, align 8, !dbg !17
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %325, <4 x float> %324) #9, !dbg !17
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %326, <4 x float> %179) #9, !dbg !17
  %lds.const.gep1381 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5248, !dbg !17
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1381, align 8, !dbg !17
  %330 = load <4 x half>, ptr addrspace(3) %address.local424, align 8, !dbg !17
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %329, <4 x float> %328) #9, !dbg !17
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %330, <4 x float> %183) #9, !dbg !17
  %lds.const.gep1382 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4096, !dbg !17
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1382, align 8, !dbg !17
  %334 = load <4 x half>, ptr addrspace(3) %address.local453, align 8, !dbg !17
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %333, <4 x float> %332) #9, !dbg !17
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %334, <4 x float> %187) #9, !dbg !17
  %lds.const.gep1383 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4096, !dbg !17
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1383, align 8, !dbg !17
  %lds.const.gep1384 = getelementptr i8, ptr addrspace(3) %address.local424, i64 128, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1384, align 8, !dbg !17
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %337, <4 x float> %336) #9, !dbg !17
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %338, <4 x float> %191) #9, !dbg !17
  %lds.const.gep1385 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4224, !dbg !17
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1385, align 8, !dbg !17
  %lds.const.gep1386 = getelementptr i8, ptr addrspace(3) %address.local453, i64 128, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1386, align 8, !dbg !17
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %341, <4 x float> %340) #9, !dbg !17
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %342, <4 x float> %195) #9, !dbg !17
  %lds.const.gep1387 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4224, !dbg !17
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1387, align 8, !dbg !17
  %lds.const.gep1388 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1024, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1388, align 8, !dbg !17
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %345, <4 x float> %344) #9, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %346, <4 x float> %199) #9, !dbg !17
  %lds.const.gep1389 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5120, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1389, align 8, !dbg !17
  %lds.const.gep1390 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1024, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1390, align 8, !dbg !17
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %349, <4 x float> %348) #9, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %350, <4 x float> %203) #9, !dbg !17
  %lds.const.gep1391 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5120, !dbg !17
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1391, align 8, !dbg !17
  %lds.const.gep1392 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1152, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1392, align 8, !dbg !17
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %353, <4 x float> %352) #9, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %354, <4 x float> %207) #9, !dbg !17
  %lds.const.gep1393 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5248, !dbg !17
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1393, align 8, !dbg !17
  %lds.const.gep1394 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1152, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1394, align 8, !dbg !17
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %357, <4 x float> %356) #9, !dbg !17
  %lds.const.gep1395 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5248, !dbg !17
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1395, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %358, <4 x float> %210) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %360, <4 x float> %361) #9, !dbg !17
  %363 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !11
  %364 = zext i32 %363 to i64, !dbg !11
  %365 = inttoptr i64 %364 to ptr addrspace(3), !dbg !11
  %address.local766 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local345, !dbg !11
  %366 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %367 = ptrtoint ptr addrspace(1) %address.local766 to i64, !dbg !11
  %368 = sub i64 %367, %366, !dbg !11
  %dma.buffer.offset1500 = trunc i64 %368 to i32, !dbg !11
  %369 = trunc i64 %366 to i32, !dbg !11
  %370 = insertelement <4 x i32> poison, i32 %369, i64 0, !dbg !11
  %371 = lshr i64 %366, 32, !dbg !11
  %372 = trunc i64 %371 to i32, !dbg !11
  %373 = insertelement <4 x i32> %370, i32 %372, i64 1, !dbg !11
  %374 = insertelement <4 x i32> %373, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1501 = insertelement <4 x i32> %374, i32 131072, i64 3, !dbg !11
  %375 = ptrtoint ptr addrspace(3) %365 to i32, !dbg !11
  %dma.buffer.lds1502 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %375), !dbg !11
  %376 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1502, i32 %dma.buffer.offset1500, <4 x i32> %dma.buffer.rsrc1501) #10, !dbg !11
  %377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !11
  %378 = zext i32 %377 to i64, !dbg !11
  %379 = inttoptr i64 %378 to ptr addrspace(3), !dbg !11
  %address.local767 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local357, !dbg !11
  %380 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %381 = ptrtoint ptr addrspace(1) %address.local767 to i64, !dbg !11
  %382 = sub i64 %381, %380, !dbg !11
  %dma.buffer.offset1497 = trunc i64 %382 to i32, !dbg !11
  %383 = trunc i64 %380 to i32, !dbg !11
  %384 = insertelement <4 x i32> poison, i32 %383, i64 0, !dbg !11
  %385 = lshr i64 %380, 32, !dbg !11
  %386 = trunc i64 %385 to i32, !dbg !11
  %387 = insertelement <4 x i32> %384, i32 %386, i64 1, !dbg !11
  %388 = insertelement <4 x i32> %387, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1498 = insertelement <4 x i32> %388, i32 131072, i64 3, !dbg !11
  %389 = ptrtoint ptr addrspace(3) %379 to i32, !dbg !11
  %dma.buffer.lds1499 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %389), !dbg !11
  %390 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1499, i32 %dma.buffer.offset1497, <4 x i32> %dma.buffer.rsrc1498) #10, !dbg !11
  %391 = add nuw i64 %61, 2048, !dbg !16
  %392 = icmp slt i64 %391, 0, !dbg !16
  %393 = sub nuw i64 -2049, %61, !dbg !16
  %394 = select i1 %392, i64 %393, i64 %391, !dbg !16
  %395 = sdiv i64 %394, 4096, !dbg !16
  %.lobit38 = ashr i64 %391, 63, !dbg !16
  %396 = xor i64 %395, %.lobit38, !dbg !16
  %397 = add nsw i64 %396, %15, !dbg !16
  %398 = add nsw i64 %246, 64, !dbg !16
  %399 = icmp slt i64 %246, -64, !dbg !16
  %400 = sub nsw i64 -65, %246, !dbg !16
  %401 = select i1 %399, i64 %400, i64 %398, !dbg !16
  %402 = sdiv i64 %401, 128, !dbg !16
  %403 = sext i1 %399 to i64, !dbg !16
  %404 = xor i64 %402, %403, !dbg !16
  %405 = shl nsw i64 %404, 7, !dbg !16
  %406 = sub i64 %55, %405, !dbg !16
  %407 = shl i64 %397, 19, !dbg !16
  %408 = and i64 %407, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %409 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %408, !dbg !16
  %410 = getelementptr [2 x i8], ptr addrspace(1) %409, i64 %101, !dbg !16
  %411 = getelementptr [2 x i8], ptr addrspace(1) %410, i64 %406, !dbg !16
  %412 = getelementptr i8, ptr addrspace(1) %411, i64 128, !dbg !16
  %413 = getelementptr [2 x i8], ptr addrspace(1) %412, i64 %107, !dbg !16
  %414 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local361), !dbg !16
  %415 = zext i32 %414 to i64, !dbg !16
  %416 = inttoptr i64 %415 to ptr addrspace(3), !dbg !16
  %address.local768 = getelementptr [2 x i8], ptr addrspace(1) %413, i64 %address.local371, !dbg !16
  %417 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %418 = ptrtoint ptr addrspace(1) %address.local768 to i64, !dbg !16
  %419 = sub i64 %418, %417, !dbg !16
  %dma.buffer.offset1494 = trunc i64 %419 to i32, !dbg !16
  %420 = trunc i64 %417 to i32, !dbg !16
  %421 = insertelement <4 x i32> poison, i32 %420, i64 0, !dbg !16
  %422 = lshr i64 %417, 32, !dbg !16
  %423 = trunc i64 %422 to i32, !dbg !16
  %424 = insertelement <4 x i32> %421, i32 %423, i64 1, !dbg !16
  %425 = insertelement <4 x i32> %424, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1495 = insertelement <4 x i32> %425, i32 131072, i64 3, !dbg !16
  %426 = ptrtoint ptr addrspace(3) %416 to i32, !dbg !16
  %dma.buffer.lds1496 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %426), !dbg !16
  %427 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1496, i32 %dma.buffer.offset1494, <4 x i32> %dma.buffer.rsrc1495) #10, !dbg !16
  %428 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local375), !dbg !16
  %429 = zext i32 %428 to i64, !dbg !16
  %430 = inttoptr i64 %429 to ptr addrspace(3), !dbg !16
  %address.local769 = getelementptr [2 x i8], ptr addrspace(1) %413, i64 %address.local385, !dbg !16
  %431 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %432 = ptrtoint ptr addrspace(1) %address.local769 to i64, !dbg !16
  %433 = sub i64 %432, %431, !dbg !16
  %dma.buffer.offset1491 = trunc i64 %433 to i32, !dbg !16
  %434 = trunc i64 %431 to i32, !dbg !16
  %435 = insertelement <4 x i32> poison, i32 %434, i64 0, !dbg !16
  %436 = lshr i64 %431, 32, !dbg !16
  %437 = trunc i64 %436 to i32, !dbg !16
  %438 = insertelement <4 x i32> %435, i32 %437, i64 1, !dbg !16
  %439 = insertelement <4 x i32> %438, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1492 = insertelement <4 x i32> %439, i32 131072, i64 3, !dbg !16
  %440 = ptrtoint ptr addrspace(3) %430 to i32, !dbg !16
  %dma.buffer.lds1493 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %440), !dbg !16
  %441 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1493, i32 %dma.buffer.offset1491, <4 x i32> %dma.buffer.rsrc1492) #10, !dbg !16
  %442 = load <4 x half>, ptr addrspace(3) %address.local399, align 8, !dbg !12
  %lds.const.gep1396 = getelementptr i8, ptr addrspace(3) %address.local399, i64 4096, !dbg !12
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1396, align 8, !dbg !12
  %lds.const.gep1397 = getelementptr i8, ptr addrspace(3) %address.local399, i64 1024, !dbg !12
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1397, align 8, !dbg !12
  %445 = fmul <4 x half> %442, splat (half 0xH3015), !dbg !12
  %lds.const.gep1398 = getelementptr i8, ptr addrspace(3) %address.local399, i64 5120, !dbg !12
  %446 = load <4 x half>, ptr addrspace(3) %lds.const.gep1398, align 8, !dbg !12
  %447 = fmul <4 x half> %443, splat (half 0xH3015), !dbg !12
  %448 = fmul <4 x half> %444, splat (half 0xH3015), !dbg !12
  %449 = fmul <4 x half> %446, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %450 = load <4 x half>, ptr addrspace(3) %address.local424, align 8, !dbg !17
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %450, <4 x float> %303) #9, !dbg !17
  %lds.const.gep1399 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4096, !dbg !17
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1399, align 8, !dbg !17
  %453 = load <4 x half>, ptr addrspace(3) %address.local453, align 8, !dbg !17
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %452, <4 x float> %451) #9, !dbg !17
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %453, <4 x float> %307) #9, !dbg !17
  %lds.const.gep1400 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4096, !dbg !17
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1400, align 8, !dbg !17
  %lds.const.gep1401 = getelementptr i8, ptr addrspace(3) %address.local424, i64 128, !dbg !17
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1401, align 8, !dbg !17
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %456, <4 x float> %455) #9, !dbg !17
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %457, <4 x float> %311) #9, !dbg !17
  %lds.const.gep1402 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4224, !dbg !17
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1402, align 8, !dbg !17
  %lds.const.gep1403 = getelementptr i8, ptr addrspace(3) %address.local453, i64 128, !dbg !17
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1403, align 8, !dbg !17
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %460, <4 x float> %459) #9, !dbg !17
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %461, <4 x float> %315) #9, !dbg !17
  %lds.const.gep1404 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4224, !dbg !17
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1404, align 8, !dbg !17
  %lds.const.gep1405 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1024, !dbg !17
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1405, align 8, !dbg !17
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %464, <4 x float> %463) #9, !dbg !17
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %465, <4 x float> %319) #9, !dbg !17
  %lds.const.gep1406 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5120, !dbg !17
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1406, align 8, !dbg !17
  %lds.const.gep1407 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1024, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1407, align 8, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %468, <4 x float> %467) #9, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %469, <4 x float> %323) #9, !dbg !17
  %lds.const.gep1408 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5120, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1408, align 8, !dbg !17
  %lds.const.gep1409 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1152, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1409, align 8, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %472, <4 x float> %471) #9, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %473, <4 x float> %327) #9, !dbg !17
  %lds.const.gep1410 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5248, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1410, align 8, !dbg !17
  %lds.const.gep1411 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1152, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1411, align 8, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %476, <4 x float> %475) #9, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %477, <4 x float> %331) #9, !dbg !17
  %lds.const.gep1412 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5248, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1412, align 8, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %address.local424, align 8, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %480, <4 x float> %479) #9, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %481, <4 x float> %335) #9, !dbg !17
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4096, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !17
  %485 = load <4 x half>, ptr addrspace(3) %address.local453, align 8, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %484, <4 x float> %483) #9, !dbg !17
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %485, <4 x float> %339) #9, !dbg !17
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4096, !dbg !17
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !17
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local424, i64 128, !dbg !17
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !17
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %488, <4 x float> %487) #9, !dbg !17
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %489, <4 x float> %343) #9, !dbg !17
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4224, !dbg !17
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !17
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local453, i64 128, !dbg !17
  %493 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !17
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %492, <4 x float> %491) #9, !dbg !17
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %493, <4 x float> %347) #9, !dbg !17
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4224, !dbg !17
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !17
  %lds.const.gep1419 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1024, !dbg !17
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep1419, align 8, !dbg !17
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %496, <4 x float> %495) #9, !dbg !17
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %497, <4 x float> %351) #9, !dbg !17
  %lds.const.gep1420 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5120, !dbg !17
  %500 = load <4 x half>, ptr addrspace(3) %lds.const.gep1420, align 8, !dbg !17
  %lds.const.gep1421 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1024, !dbg !17
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1421, align 8, !dbg !17
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %500, <4 x float> %499) #9, !dbg !17
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %501, <4 x float> %355) #9, !dbg !17
  %lds.const.gep1422 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5120, !dbg !17
  %504 = load <4 x half>, ptr addrspace(3) %lds.const.gep1422, align 8, !dbg !17
  %lds.const.gep1423 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1152, !dbg !17
  %505 = load <4 x half>, ptr addrspace(3) %lds.const.gep1423, align 8, !dbg !17
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %504, <4 x float> %503) #9, !dbg !17
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %505, <4 x float> %359) #9, !dbg !17
  %lds.const.gep1424 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5248, !dbg !17
  %508 = load <4 x half>, ptr addrspace(3) %lds.const.gep1424, align 8, !dbg !17
  %lds.const.gep1425 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1152, !dbg !17
  %509 = load <4 x half>, ptr addrspace(3) %lds.const.gep1425, align 8, !dbg !17
  %510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %508, <4 x float> %507) #9, !dbg !17
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5248, !dbg !17
  %511 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %509, <4 x float> %362) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %511, <4 x float> %512) #9, !dbg !17
  %514 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !11
  %515 = zext i32 %514 to i64, !dbg !11
  %516 = inttoptr i64 %515 to ptr addrspace(3), !dbg !11
  %address.local770 = getelementptr [2 x i8], ptr addrspace(1) %59, i64 %address.local345, !dbg !11
  %517 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %518 = ptrtoint ptr addrspace(1) %address.local770 to i64, !dbg !11
  %519 = sub i64 %518, %517, !dbg !11
  %dma.buffer.offset1488 = trunc i64 %519 to i32, !dbg !11
  %520 = trunc i64 %517 to i32, !dbg !11
  %521 = insertelement <4 x i32> poison, i32 %520, i64 0, !dbg !11
  %522 = lshr i64 %517, 32, !dbg !11
  %523 = trunc i64 %522 to i32, !dbg !11
  %524 = insertelement <4 x i32> %521, i32 %523, i64 1, !dbg !11
  %525 = insertelement <4 x i32> %524, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1489 = insertelement <4 x i32> %525, i32 131072, i64 3, !dbg !11
  %526 = ptrtoint ptr addrspace(3) %516 to i32, !dbg !11
  %dma.buffer.lds1490 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %526), !dbg !11
  %527 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1490, i32 %dma.buffer.offset1488, <4 x i32> %dma.buffer.rsrc1489) #10, !dbg !11
  %528 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !11
  %529 = zext i32 %528 to i64, !dbg !11
  %530 = inttoptr i64 %529 to ptr addrspace(3), !dbg !11
  %address.local771 = getelementptr [2 x i8], ptr addrspace(1) %59, i64 %address.local357, !dbg !11
  %531 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %532 = ptrtoint ptr addrspace(1) %address.local771 to i64, !dbg !11
  %533 = sub i64 %532, %531, !dbg !11
  %dma.buffer.offset1485 = trunc i64 %533 to i32, !dbg !11
  %534 = trunc i64 %531 to i32, !dbg !11
  %535 = insertelement <4 x i32> poison, i32 %534, i64 0, !dbg !11
  %536 = lshr i64 %531, 32, !dbg !11
  %537 = trunc i64 %536 to i32, !dbg !11
  %538 = insertelement <4 x i32> %535, i32 %537, i64 1, !dbg !11
  %539 = insertelement <4 x i32> %538, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1486 = insertelement <4 x i32> %539, i32 131072, i64 3, !dbg !11
  %540 = ptrtoint ptr addrspace(3) %530 to i32, !dbg !11
  %dma.buffer.lds1487 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %540), !dbg !11
  %541 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1487, i32 %dma.buffer.offset1485, <4 x i32> %dma.buffer.rsrc1486) #10, !dbg !11
  %542 = add nuw i64 %61, 3072, !dbg !16
  %543 = icmp slt i64 %542, 0, !dbg !16
  %544 = sub nuw i64 -3073, %61, !dbg !16
  %545 = select i1 %543, i64 %544, i64 %542, !dbg !16
  %546 = sdiv i64 %545, 4096, !dbg !16
  %.lobit39 = ashr i64 %542, 63, !dbg !16
  %547 = xor i64 %546, %.lobit39, !dbg !16
  %548 = add nsw i64 %547, %15, !dbg !16
  %549 = add nsw i64 %246, 96, !dbg !16
  %550 = icmp slt i64 %246, -96, !dbg !16
  %551 = sub nsw i64 -97, %246, !dbg !16
  %552 = select i1 %550, i64 %551, i64 %549, !dbg !16
  %553 = sdiv i64 %552, 128, !dbg !16
  %554 = sext i1 %550 to i64, !dbg !16
  %555 = xor i64 %553, %554, !dbg !16
  %556 = shl nsw i64 %555, 7, !dbg !16
  %557 = sub i64 %55, %556, !dbg !16
  %558 = shl i64 %548, 19, !dbg !16
  %559 = and i64 %558, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %560 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %559, !dbg !16
  %561 = getelementptr [2 x i8], ptr addrspace(1) %560, i64 %101, !dbg !16
  %562 = getelementptr [2 x i8], ptr addrspace(1) %561, i64 %557, !dbg !16
  %563 = getelementptr i8, ptr addrspace(1) %562, i64 192, !dbg !16
  %564 = getelementptr [2 x i8], ptr addrspace(1) %563, i64 %107, !dbg !16
  %565 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local361), !dbg !16
  %566 = zext i32 %565 to i64, !dbg !16
  %567 = inttoptr i64 %566 to ptr addrspace(3), !dbg !16
  %address.local772 = getelementptr [2 x i8], ptr addrspace(1) %564, i64 %address.local371, !dbg !16
  %568 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %569 = ptrtoint ptr addrspace(1) %address.local772 to i64, !dbg !16
  %570 = sub i64 %569, %568, !dbg !16
  %dma.buffer.offset1482 = trunc i64 %570 to i32, !dbg !16
  %571 = trunc i64 %568 to i32, !dbg !16
  %572 = insertelement <4 x i32> poison, i32 %571, i64 0, !dbg !16
  %573 = lshr i64 %568, 32, !dbg !16
  %574 = trunc i64 %573 to i32, !dbg !16
  %575 = insertelement <4 x i32> %572, i32 %574, i64 1, !dbg !16
  %576 = insertelement <4 x i32> %575, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1483 = insertelement <4 x i32> %576, i32 131072, i64 3, !dbg !16
  %577 = ptrtoint ptr addrspace(3) %567 to i32, !dbg !16
  %dma.buffer.lds1484 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %577), !dbg !16
  %578 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1484, i32 %dma.buffer.offset1482, <4 x i32> %dma.buffer.rsrc1483) #10, !dbg !16
  %579 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local375), !dbg !16
  %580 = zext i32 %579 to i64, !dbg !16
  %581 = inttoptr i64 %580 to ptr addrspace(3), !dbg !16
  %address.local773 = getelementptr [2 x i8], ptr addrspace(1) %564, i64 %address.local385, !dbg !16
  %582 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %583 = ptrtoint ptr addrspace(1) %address.local773 to i64, !dbg !16
  %584 = sub i64 %583, %582, !dbg !16
  %dma.buffer.offset1479 = trunc i64 %584 to i32, !dbg !16
  %585 = trunc i64 %582 to i32, !dbg !16
  %586 = insertelement <4 x i32> poison, i32 %585, i64 0, !dbg !16
  %587 = lshr i64 %582, 32, !dbg !16
  %588 = trunc i64 %587 to i32, !dbg !16
  %589 = insertelement <4 x i32> %586, i32 %588, i64 1, !dbg !16
  %590 = insertelement <4 x i32> %589, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1480 = insertelement <4 x i32> %590, i32 131072, i64 3, !dbg !16
  %591 = ptrtoint ptr addrspace(3) %581 to i32, !dbg !16
  %dma.buffer.lds1481 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %591), !dbg !16
  %592 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1481, i32 %dma.buffer.offset1479, <4 x i32> %dma.buffer.rsrc1480) #10, !dbg !16
  %593 = load <4 x half>, ptr addrspace(3) %address.local399, align 8, !dbg !12
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local399, i64 4096, !dbg !12
  %594 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !12
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local399, i64 1024, !dbg !12
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !12
  %596 = fmul <4 x half> %593, splat (half 0xH3015), !dbg !12
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local399, i64 5120, !dbg !12
  %597 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !12
  %598 = fmul <4 x half> %594, splat (half 0xH3015), !dbg !12
  %599 = fmul <4 x half> %595, splat (half 0xH3015), !dbg !12
  %600 = fmul <4 x half> %597, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %601 = add nuw nsw i64 %101, %15, !dbg !18
  %602 = shl i64 %601, 19, !dbg !18
  %603 = and i64 %602, 16252928, !dbg !18
  %604 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %603, !dbg !18
  %605 = getelementptr [2 x i8], ptr addrspace(1) %604, i64 %107, !dbg !18
  %606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !18
  %607 = zext i32 %606 to i64, !dbg !18
  %608 = inttoptr i64 %607 to ptr addrspace(3), !dbg !18
  %address.local774 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local775 = and i64 %address.local774, 8064, !dbg !18
  %address.local776 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local777 = and i64 %address.local776, 24, !dbg !18
  %address.local778 = or disjoint i64 %address.local775, %address.local777, !dbg !18
  %address.local779 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local778, !dbg !18
  %609 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %610 = ptrtoint ptr addrspace(1) %address.local779 to i64, !dbg !18
  %611 = sub i64 %610, %609, !dbg !18
  %dma.buffer.offset1476 = trunc i64 %611 to i32, !dbg !18
  %612 = trunc i64 %609 to i32, !dbg !18
  %613 = insertelement <4 x i32> poison, i32 %612, i64 0, !dbg !18
  %614 = lshr i64 %609, 32, !dbg !18
  %615 = trunc i64 %614 to i32, !dbg !18
  %616 = insertelement <4 x i32> %613, i32 %615, i64 1, !dbg !18
  %617 = insertelement <4 x i32> %616, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1477 = insertelement <4 x i32> %617, i32 131072, i64 3, !dbg !18
  %618 = ptrtoint ptr addrspace(3) %608 to i32, !dbg !18
  %dma.buffer.lds1478 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %618), !dbg !18
  %619 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1478, i32 %dma.buffer.offset1476, <4 x i32> %dma.buffer.rsrc1477) #10, !dbg !18
  %620 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !18
  %621 = zext i32 %620 to i64, !dbg !18
  %622 = inttoptr i64 %621 to ptr addrspace(3), !dbg !18
  %address.local780 = shl nuw nsw i64 %address.local347, 5, !dbg !18
  %address.local781 = and i64 %address.local780, 16256, !dbg !18
  %address.local782 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local783 = and i64 %address.local782, 24, !dbg !18
  %address.local784 = or disjoint i64 %address.local781, %address.local783, !dbg !18
  %address.local785 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local784, !dbg !18
  %623 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %624 = ptrtoint ptr addrspace(1) %address.local785 to i64, !dbg !18
  %625 = sub i64 %624, %623, !dbg !18
  %dma.buffer.offset1473 = trunc i64 %625 to i32, !dbg !18
  %626 = trunc i64 %623 to i32, !dbg !18
  %627 = insertelement <4 x i32> poison, i32 %626, i64 0, !dbg !18
  %628 = lshr i64 %623, 32, !dbg !18
  %629 = trunc i64 %628 to i32, !dbg !18
  %630 = insertelement <4 x i32> %627, i32 %629, i64 1, !dbg !18
  %631 = insertelement <4 x i32> %630, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1474 = insertelement <4 x i32> %631, i32 131072, i64 3, !dbg !18
  %632 = ptrtoint ptr addrspace(3) %622 to i32, !dbg !18
  %dma.buffer.lds1475 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %632), !dbg !18
  %633 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1475, i32 %dma.buffer.offset1473, <4 x i32> %dma.buffer.rsrc1474) #10, !dbg !18
  %634 = load <4 x half>, ptr addrspace(3) %address.local424, align 8, !dbg !17
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %634, <4 x float> %454) #9, !dbg !17
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4096, !dbg !17
  %636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !17
  %637 = load <4 x half>, ptr addrspace(3) %address.local453, align 8, !dbg !17
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %636, <4 x float> %635) #9, !dbg !17
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %637, <4 x float> %458) #9, !dbg !17
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4096, !dbg !17
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !17
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local424, i64 128, !dbg !17
  %641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !17
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %640, <4 x float> %639) #9, !dbg !17
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %641, <4 x float> %462) #9, !dbg !17
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4224, !dbg !17
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !17
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %address.local453, i64 128, !dbg !17
  %645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !17
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %644, <4 x float> %643) #9, !dbg !17
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %645, <4 x float> %466) #9, !dbg !17
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4224, !dbg !17
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !17
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1024, !dbg !17
  %649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !17
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %648, <4 x float> %647) #9, !dbg !17
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %649, <4 x float> %470) #9, !dbg !17
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5120, !dbg !17
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !17
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1024, !dbg !17
  %653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !17
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %652, <4 x float> %651) #9, !dbg !17
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %653, <4 x float> %474) #9, !dbg !17
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5120, !dbg !17
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !17
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1152, !dbg !17
  %657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !17
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %656, <4 x float> %655) #9, !dbg !17
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %657, <4 x float> %478) #9, !dbg !17
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5248, !dbg !17
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !17
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1152, !dbg !17
  %661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !17
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %660, <4 x float> %659) #9, !dbg !17
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %596, <4 x half> %661, <4 x float> %482) #9, !dbg !17
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5248, !dbg !17
  %664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !17
  %665 = load <4 x half>, ptr addrspace(3) %address.local424, align 8, !dbg !17
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %664, <4 x float> %663) #9, !dbg !17
  %667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %665, <4 x float> %486) #9, !dbg !17
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4096, !dbg !17
  %668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !17
  %669 = load <4 x half>, ptr addrspace(3) %address.local453, align 8, !dbg !17
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %668, <4 x float> %667) #9, !dbg !17
  %671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %669, <4 x float> %490) #9, !dbg !17
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4096, !dbg !17
  %672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !17
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local424, i64 128, !dbg !17
  %673 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !17
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %672, <4 x float> %671) #9, !dbg !17
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %673, <4 x float> %494) #9, !dbg !17
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4224, !dbg !17
  %676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !17
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local453, i64 128, !dbg !17
  %677 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !17
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %676, <4 x float> %675) #9, !dbg !17
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %677, <4 x float> %498) #9, !dbg !17
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4224, !dbg !17
  %680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !17
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1024, !dbg !17
  %681 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !17
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %680, <4 x float> %679) #9, !dbg !17
  %683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %681, <4 x float> %502) #9, !dbg !17
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5120, !dbg !17
  %684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !17
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1024, !dbg !17
  %685 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !17
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %684, <4 x float> %683) #9, !dbg !17
  %687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %685, <4 x float> %506) #9, !dbg !17
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5120, !dbg !17
  %688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !17
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1152, !dbg !17
  %689 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !17
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %688, <4 x float> %687) #9, !dbg !17
  %691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %689, <4 x float> %510) #9, !dbg !17
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5248, !dbg !17
  %692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !17
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1152, !dbg !17
  %693 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !17
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %692, <4 x float> %691) #9, !dbg !17
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5248, !dbg !17
  %695 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %693, <4 x float> %513) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %697 = or disjoint i64 %70, 127, !dbg !13
  %698 = icmp uge i64 %19, %697, !dbg !13
  %699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %600, <4 x half> %695, <4 x float> %696) #9, !dbg !17
  %700 = and i1 %39, %698, !dbg !13
  br i1 %700, label %701, label %702, !dbg !13

701:                                              ; preds = %69
  br label %846, !dbg !13

702:                                              ; preds = %69
  %703 = add i64 %19, %35, !dbg !13
  %704 = shl nuw nsw i64 %36, 4, !dbg !13
  %705 = add i64 %703, %704, !dbg !13
  %706 = and i32 %34, 240, !dbg !13
  %index.compact = zext nneg i32 %706 to i64, !dbg !13
  %707 = sub i64 %705, %index.compact, !dbg !13
  %708 = and i64 %35, 48, !dbg !13
  %709 = or disjoint i64 %70, %708, !dbg !13
  %710 = add i64 %707, 1, !dbg !23
  %711 = lshr i64 %710, 32, !dbg !24
  %compare.high = trunc i64 %711 to i32, !dbg !24
  %712 = lshr i64 %709, 32, !dbg !24
  %compare.high1079 = trunc i64 %712 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1079, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1079, !dbg !24
  %compare.low.base = trunc i64 %710 to i32, !dbg !24
  %compare.low.base1080 = trunc i64 %709 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1080, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %713 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1081 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result1082 = select i1 %compare.high.equal, i1 %compare.low1081, i1 %compare.high.order, !dbg !24
  %714 = select i1 %compare.result1082, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1083 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1084 = or i32 %compare.local.base1083, 2, !dbg !24
  %compare.low1085 = icmp ugt i32 %compare.low.base, %compare.local1084, !dbg !24
  %compare.result1086 = select i1 %compare.high.equal, i1 %compare.low1085, i1 %compare.high.order, !dbg !24
  %715 = select i1 %compare.result1086, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1087 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1088 = or i32 %compare.local.base1087, 3, !dbg !24
  %compare.low1089 = icmp ugt i32 %compare.low.base, %compare.local1088, !dbg !24
  %compare.result1090 = select i1 %compare.high.equal, i1 %compare.low1089, i1 %compare.high.order, !dbg !24
  %716 = select i1 %compare.result1090, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1091 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1092 = or i32 %compare.local.base1091, 4, !dbg !24
  %compare.low1093 = icmp ugt i32 %compare.low.base, %compare.local1092, !dbg !24
  %compare.result1094 = select i1 %compare.high.equal, i1 %compare.low1093, i1 %compare.high.order, !dbg !24
  %717 = select i1 %compare.result1094, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1095 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1096 = or i32 %compare.local.base1095, 5, !dbg !24
  %compare.low1097 = icmp ugt i32 %compare.low.base, %compare.local1096, !dbg !24
  %compare.result1098 = select i1 %compare.high.equal, i1 %compare.low1097, i1 %compare.high.order, !dbg !24
  %718 = select i1 %compare.result1098, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1099 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1100 = or i32 %compare.local.base1099, 6, !dbg !24
  %compare.low1101 = icmp ugt i32 %compare.low.base, %compare.local1100, !dbg !24
  %compare.result1102 = select i1 %compare.high.equal, i1 %compare.low1101, i1 %compare.high.order, !dbg !24
  %719 = select i1 %compare.result1102, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1103 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1104 = or i32 %compare.local.base1103, 7, !dbg !24
  %compare.low1105 = icmp ugt i32 %compare.low.base, %compare.local1104, !dbg !24
  %compare.result1106 = select i1 %compare.high.equal, i1 %compare.low1105, i1 %compare.high.order, !dbg !24
  %720 = select i1 %compare.result1106, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1107 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1108 = or i32 %compare.local.base1107, 8, !dbg !24
  %compare.low1109 = icmp ugt i32 %compare.low.base, %compare.local1108, !dbg !24
  %compare.result1110 = select i1 %compare.high.equal, i1 %compare.low1109, i1 %compare.high.order, !dbg !24
  %721 = select i1 %compare.result1110, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1111 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1112 = or i32 %compare.local.base1111, 9, !dbg !24
  %compare.low1113 = icmp ugt i32 %compare.low.base, %compare.local1112, !dbg !24
  %compare.result1114 = select i1 %compare.high.equal, i1 %compare.low1113, i1 %compare.high.order, !dbg !24
  %722 = select i1 %compare.result1114, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1115 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1116 = or i32 %compare.local.base1115, 10, !dbg !24
  %compare.low1117 = icmp ugt i32 %compare.low.base, %compare.local1116, !dbg !24
  %compare.result1118 = select i1 %compare.high.equal, i1 %compare.low1117, i1 %compare.high.order, !dbg !24
  %723 = select i1 %compare.result1118, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1119 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1120 = or i32 %compare.local.base1119, 11, !dbg !24
  %compare.low1121 = icmp ugt i32 %compare.low.base, %compare.local1120, !dbg !24
  %compare.result1122 = select i1 %compare.high.equal, i1 %compare.low1121, i1 %compare.high.order, !dbg !24
  %724 = select i1 %compare.result1122, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1123 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1124 = or i32 %compare.local.base1123, 12, !dbg !24
  %compare.low1125 = icmp ugt i32 %compare.low.base, %compare.local1124, !dbg !24
  %compare.result1126 = select i1 %compare.high.equal, i1 %compare.low1125, i1 %compare.high.order, !dbg !24
  %725 = select i1 %compare.result1126, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1127 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1128 = or i32 %compare.local.base1127, 13, !dbg !24
  %compare.low1129 = icmp ugt i32 %compare.low.base, %compare.local1128, !dbg !24
  %compare.result1130 = select i1 %compare.high.equal, i1 %compare.low1129, i1 %compare.high.order, !dbg !24
  %726 = select i1 %compare.result1130, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1131 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1132 = or i32 %compare.local.base1131, 14, !dbg !24
  %compare.low1133 = icmp ugt i32 %compare.low.base, %compare.local1132, !dbg !24
  %compare.result1134 = select i1 %compare.high.equal, i1 %compare.low1133, i1 %compare.high.order, !dbg !24
  %727 = select i1 %compare.result1134, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1135 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1136 = or i32 %compare.local.base1135, 15, !dbg !24
  %compare.low1137 = icmp ugt i32 %compare.low.base, %compare.local1136, !dbg !24
  %compare.result1138 = select i1 %compare.high.equal, i1 %compare.low1137, i1 %compare.high.order, !dbg !24
  %728 = select i1 %compare.result1138, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1139 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1140 = or i32 %compare.local.base1139, 64, !dbg !24
  %compare.low1141 = icmp ugt i32 %compare.low.base, %compare.local1140, !dbg !24
  %compare.result1142 = select i1 %compare.high.equal, i1 %compare.low1141, i1 %compare.high.order, !dbg !24
  %729 = select i1 %compare.result1142, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1143 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1144 = or i32 %compare.local.base1143, 65, !dbg !24
  %compare.low1145 = icmp ugt i32 %compare.low.base, %compare.local1144, !dbg !24
  %compare.result1146 = select i1 %compare.high.equal, i1 %compare.low1145, i1 %compare.high.order, !dbg !24
  %730 = select i1 %compare.result1146, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1147 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1148 = or i32 %compare.local.base1147, 66, !dbg !24
  %compare.low1149 = icmp ugt i32 %compare.low.base, %compare.local1148, !dbg !24
  %compare.result1150 = select i1 %compare.high.equal, i1 %compare.low1149, i1 %compare.high.order, !dbg !24
  %731 = select i1 %compare.result1150, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1151 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1152 = or i32 %compare.local.base1151, 67, !dbg !24
  %compare.low1153 = icmp ugt i32 %compare.low.base, %compare.local1152, !dbg !24
  %compare.result1154 = select i1 %compare.high.equal, i1 %compare.low1153, i1 %compare.high.order, !dbg !24
  %732 = select i1 %compare.result1154, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1155 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1156 = or i32 %compare.local.base1155, 68, !dbg !24
  %compare.low1157 = icmp ugt i32 %compare.low.base, %compare.local1156, !dbg !24
  %compare.result1158 = select i1 %compare.high.equal, i1 %compare.low1157, i1 %compare.high.order, !dbg !24
  %733 = select i1 %compare.result1158, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1159 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1160 = or i32 %compare.local.base1159, 69, !dbg !24
  %compare.low1161 = icmp ugt i32 %compare.low.base, %compare.local1160, !dbg !24
  %compare.result1162 = select i1 %compare.high.equal, i1 %compare.low1161, i1 %compare.high.order, !dbg !24
  %734 = select i1 %compare.result1162, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1163 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1164 = or i32 %compare.local.base1163, 70, !dbg !24
  %compare.low1165 = icmp ugt i32 %compare.low.base, %compare.local1164, !dbg !24
  %compare.result1166 = select i1 %compare.high.equal, i1 %compare.low1165, i1 %compare.high.order, !dbg !24
  %735 = select i1 %compare.result1166, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1167 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1168 = or i32 %compare.local.base1167, 71, !dbg !24
  %compare.low1169 = icmp ugt i32 %compare.low.base, %compare.local1168, !dbg !24
  %compare.result1170 = select i1 %compare.high.equal, i1 %compare.low1169, i1 %compare.high.order, !dbg !24
  %736 = select i1 %compare.result1170, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1171 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1172 = or i32 %compare.local.base1171, 72, !dbg !24
  %compare.low1173 = icmp ugt i32 %compare.low.base, %compare.local1172, !dbg !24
  %compare.result1174 = select i1 %compare.high.equal, i1 %compare.low1173, i1 %compare.high.order, !dbg !24
  %737 = select i1 %compare.result1174, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1175 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1176 = or i32 %compare.local.base1175, 73, !dbg !24
  %compare.low1177 = icmp ugt i32 %compare.low.base, %compare.local1176, !dbg !24
  %compare.result1178 = select i1 %compare.high.equal, i1 %compare.low1177, i1 %compare.high.order, !dbg !24
  %738 = select i1 %compare.result1178, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1179 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1180 = or i32 %compare.local.base1179, 74, !dbg !24
  %compare.low1181 = icmp ugt i32 %compare.low.base, %compare.local1180, !dbg !24
  %compare.result1182 = select i1 %compare.high.equal, i1 %compare.low1181, i1 %compare.high.order, !dbg !24
  %739 = select i1 %compare.result1182, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1183 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1184 = or i32 %compare.local.base1183, 75, !dbg !24
  %compare.low1185 = icmp ugt i32 %compare.low.base, %compare.local1184, !dbg !24
  %compare.result1186 = select i1 %compare.high.equal, i1 %compare.low1185, i1 %compare.high.order, !dbg !24
  %740 = select i1 %compare.result1186, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1187 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1188 = or i32 %compare.local.base1187, 76, !dbg !24
  %compare.low1189 = icmp ugt i32 %compare.low.base, %compare.local1188, !dbg !24
  %compare.result1190 = select i1 %compare.high.equal, i1 %compare.low1189, i1 %compare.high.order, !dbg !24
  %741 = select i1 %compare.result1190, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1191 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1192 = or i32 %compare.local.base1191, 77, !dbg !24
  %compare.low1193 = icmp ugt i32 %compare.low.base, %compare.local1192, !dbg !24
  %compare.result1194 = select i1 %compare.high.equal, i1 %compare.low1193, i1 %compare.high.order, !dbg !24
  %742 = select i1 %compare.result1194, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1195 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1196 = or i32 %compare.local.base1195, 78, !dbg !24
  %compare.low1197 = icmp ugt i32 %compare.low.base, %compare.local1196, !dbg !24
  %compare.result1198 = select i1 %compare.high.equal, i1 %compare.low1197, i1 %compare.high.order, !dbg !24
  %743 = select i1 %compare.result1198, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1199 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1080, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1200 = or i32 %compare.local.base1199, 79, !dbg !24
  %compare.low1201 = icmp ugt i32 %compare.low.base, %compare.local1200, !dbg !24
  %compare.result1202 = select i1 %compare.high.equal, i1 %compare.low1201, i1 %compare.high.order, !dbg !24
  %744 = select i1 %compare.result1202, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %745 = add i64 %707, 65, !dbg !23
  %746 = lshr i64 %745, 32, !dbg !24
  %compare.high1203 = trunc i64 %746 to i32, !dbg !24
  %747 = lshr i64 %709, 32, !dbg !24
  %compare.high1204 = trunc i64 %747 to i32, !dbg !24
  %compare.high.equal1205 = icmp eq i32 %compare.high1203, %compare.high1204, !dbg !24
  %compare.high.order1206 = icmp ugt i32 %compare.high1203, %compare.high1204, !dbg !24
  %compare.low.base1207 = trunc i64 %745 to i32, !dbg !24
  %compare.low.base1208 = trunc i64 %709 to i32, !dbg !24
  %compare.low1209 = icmp ugt i32 %compare.low.base1207, %compare.low.base1208, !dbg !24
  %compare.result1210 = select i1 %compare.high.equal1205, i1 %compare.low1209, i1 %compare.high.order1206, !dbg !24
  %748 = select i1 %compare.result1210, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1211 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1212 = or i32 %compare.local.base1211, 1, !dbg !24
  %compare.low1213 = icmp ugt i32 %compare.low.base1207, %compare.local1212, !dbg !24
  %compare.result1214 = select i1 %compare.high.equal1205, i1 %compare.low1213, i1 %compare.high.order1206, !dbg !24
  %749 = select i1 %compare.result1214, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1215 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1216 = or i32 %compare.local.base1215, 2, !dbg !24
  %compare.low1217 = icmp ugt i32 %compare.low.base1207, %compare.local1216, !dbg !24
  %compare.result1218 = select i1 %compare.high.equal1205, i1 %compare.low1217, i1 %compare.high.order1206, !dbg !24
  %750 = select i1 %compare.result1218, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1219 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1220 = or i32 %compare.local.base1219, 3, !dbg !24
  %compare.low1221 = icmp ugt i32 %compare.low.base1207, %compare.local1220, !dbg !24
  %compare.result1222 = select i1 %compare.high.equal1205, i1 %compare.low1221, i1 %compare.high.order1206, !dbg !24
  %751 = select i1 %compare.result1222, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1223 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1224 = or i32 %compare.local.base1223, 4, !dbg !24
  %compare.low1225 = icmp ugt i32 %compare.low.base1207, %compare.local1224, !dbg !24
  %compare.result1226 = select i1 %compare.high.equal1205, i1 %compare.low1225, i1 %compare.high.order1206, !dbg !24
  %752 = select i1 %compare.result1226, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1227 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1228 = or i32 %compare.local.base1227, 5, !dbg !24
  %compare.low1229 = icmp ugt i32 %compare.low.base1207, %compare.local1228, !dbg !24
  %compare.result1230 = select i1 %compare.high.equal1205, i1 %compare.low1229, i1 %compare.high.order1206, !dbg !24
  %753 = select i1 %compare.result1230, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1231 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1232 = or i32 %compare.local.base1231, 6, !dbg !24
  %compare.low1233 = icmp ugt i32 %compare.low.base1207, %compare.local1232, !dbg !24
  %compare.result1234 = select i1 %compare.high.equal1205, i1 %compare.low1233, i1 %compare.high.order1206, !dbg !24
  %754 = select i1 %compare.result1234, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1235 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1236 = or i32 %compare.local.base1235, 7, !dbg !24
  %compare.low1237 = icmp ugt i32 %compare.low.base1207, %compare.local1236, !dbg !24
  %compare.result1238 = select i1 %compare.high.equal1205, i1 %compare.low1237, i1 %compare.high.order1206, !dbg !24
  %755 = select i1 %compare.result1238, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1239 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1240 = or i32 %compare.local.base1239, 8, !dbg !24
  %compare.low1241 = icmp ugt i32 %compare.low.base1207, %compare.local1240, !dbg !24
  %compare.result1242 = select i1 %compare.high.equal1205, i1 %compare.low1241, i1 %compare.high.order1206, !dbg !24
  %756 = select i1 %compare.result1242, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1243 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1244 = or i32 %compare.local.base1243, 9, !dbg !24
  %compare.low1245 = icmp ugt i32 %compare.low.base1207, %compare.local1244, !dbg !24
  %compare.result1246 = select i1 %compare.high.equal1205, i1 %compare.low1245, i1 %compare.high.order1206, !dbg !24
  %757 = select i1 %compare.result1246, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1247 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1248 = or i32 %compare.local.base1247, 10, !dbg !24
  %compare.low1249 = icmp ugt i32 %compare.low.base1207, %compare.local1248, !dbg !24
  %compare.result1250 = select i1 %compare.high.equal1205, i1 %compare.low1249, i1 %compare.high.order1206, !dbg !24
  %758 = select i1 %compare.result1250, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1251 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1252 = or i32 %compare.local.base1251, 11, !dbg !24
  %compare.low1253 = icmp ugt i32 %compare.low.base1207, %compare.local1252, !dbg !24
  %compare.result1254 = select i1 %compare.high.equal1205, i1 %compare.low1253, i1 %compare.high.order1206, !dbg !24
  %759 = select i1 %compare.result1254, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1255 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1256 = or i32 %compare.local.base1255, 12, !dbg !24
  %compare.low1257 = icmp ugt i32 %compare.low.base1207, %compare.local1256, !dbg !24
  %compare.result1258 = select i1 %compare.high.equal1205, i1 %compare.low1257, i1 %compare.high.order1206, !dbg !24
  %760 = select i1 %compare.result1258, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1259 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1260 = or i32 %compare.local.base1259, 13, !dbg !24
  %compare.low1261 = icmp ugt i32 %compare.low.base1207, %compare.local1260, !dbg !24
  %compare.result1262 = select i1 %compare.high.equal1205, i1 %compare.low1261, i1 %compare.high.order1206, !dbg !24
  %761 = select i1 %compare.result1262, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1263 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1264 = or i32 %compare.local.base1263, 14, !dbg !24
  %compare.low1265 = icmp ugt i32 %compare.low.base1207, %compare.local1264, !dbg !24
  %compare.result1266 = select i1 %compare.high.equal1205, i1 %compare.low1265, i1 %compare.high.order1206, !dbg !24
  %762 = select i1 %compare.result1266, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1267 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1268 = or i32 %compare.local.base1267, 15, !dbg !24
  %compare.low1269 = icmp ugt i32 %compare.low.base1207, %compare.local1268, !dbg !24
  %compare.result1270 = select i1 %compare.high.equal1205, i1 %compare.low1269, i1 %compare.high.order1206, !dbg !24
  %763 = select i1 %compare.result1270, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1271 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1272 = or i32 %compare.local.base1271, 64, !dbg !24
  %compare.low1273 = icmp ugt i32 %compare.low.base1207, %compare.local1272, !dbg !24
  %compare.result1274 = select i1 %compare.high.equal1205, i1 %compare.low1273, i1 %compare.high.order1206, !dbg !24
  %764 = select i1 %compare.result1274, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1275 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1276 = or i32 %compare.local.base1275, 65, !dbg !24
  %compare.low1277 = icmp ugt i32 %compare.low.base1207, %compare.local1276, !dbg !24
  %compare.result1278 = select i1 %compare.high.equal1205, i1 %compare.low1277, i1 %compare.high.order1206, !dbg !24
  %765 = select i1 %compare.result1278, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1279 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1280 = or i32 %compare.local.base1279, 66, !dbg !24
  %compare.low1281 = icmp ugt i32 %compare.low.base1207, %compare.local1280, !dbg !24
  %compare.result1282 = select i1 %compare.high.equal1205, i1 %compare.low1281, i1 %compare.high.order1206, !dbg !24
  %766 = select i1 %compare.result1282, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1283 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1284 = or i32 %compare.local.base1283, 67, !dbg !24
  %compare.low1285 = icmp ugt i32 %compare.low.base1207, %compare.local1284, !dbg !24
  %compare.result1286 = select i1 %compare.high.equal1205, i1 %compare.low1285, i1 %compare.high.order1206, !dbg !24
  %767 = select i1 %compare.result1286, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1287 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1288 = or i32 %compare.local.base1287, 68, !dbg !24
  %compare.low1289 = icmp ugt i32 %compare.low.base1207, %compare.local1288, !dbg !24
  %compare.result1290 = select i1 %compare.high.equal1205, i1 %compare.low1289, i1 %compare.high.order1206, !dbg !24
  %768 = select i1 %compare.result1290, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1291 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1292 = or i32 %compare.local.base1291, 69, !dbg !24
  %compare.low1293 = icmp ugt i32 %compare.low.base1207, %compare.local1292, !dbg !24
  %compare.result1294 = select i1 %compare.high.equal1205, i1 %compare.low1293, i1 %compare.high.order1206, !dbg !24
  %769 = select i1 %compare.result1294, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1295 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1296 = or i32 %compare.local.base1295, 70, !dbg !24
  %compare.low1297 = icmp ugt i32 %compare.low.base1207, %compare.local1296, !dbg !24
  %compare.result1298 = select i1 %compare.high.equal1205, i1 %compare.low1297, i1 %compare.high.order1206, !dbg !24
  %770 = select i1 %compare.result1298, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1299 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1300 = or i32 %compare.local.base1299, 71, !dbg !24
  %compare.low1301 = icmp ugt i32 %compare.low.base1207, %compare.local1300, !dbg !24
  %compare.result1302 = select i1 %compare.high.equal1205, i1 %compare.low1301, i1 %compare.high.order1206, !dbg !24
  %771 = select i1 %compare.result1302, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1303 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1304 = or i32 %compare.local.base1303, 72, !dbg !24
  %compare.low1305 = icmp ugt i32 %compare.low.base1207, %compare.local1304, !dbg !24
  %compare.result1306 = select i1 %compare.high.equal1205, i1 %compare.low1305, i1 %compare.high.order1206, !dbg !24
  %772 = select i1 %compare.result1306, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1307 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1308 = or i32 %compare.local.base1307, 73, !dbg !24
  %compare.low1309 = icmp ugt i32 %compare.low.base1207, %compare.local1308, !dbg !24
  %compare.result1310 = select i1 %compare.high.equal1205, i1 %compare.low1309, i1 %compare.high.order1206, !dbg !24
  %773 = select i1 %compare.result1310, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1311 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1312 = or i32 %compare.local.base1311, 74, !dbg !24
  %compare.low1313 = icmp ugt i32 %compare.low.base1207, %compare.local1312, !dbg !24
  %compare.result1314 = select i1 %compare.high.equal1205, i1 %compare.low1313, i1 %compare.high.order1206, !dbg !24
  %774 = select i1 %compare.result1314, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1315 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1316 = or i32 %compare.local.base1315, 75, !dbg !24
  %compare.low1317 = icmp ugt i32 %compare.low.base1207, %compare.local1316, !dbg !24
  %compare.result1318 = select i1 %compare.high.equal1205, i1 %compare.low1317, i1 %compare.high.order1206, !dbg !24
  %775 = select i1 %compare.result1318, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1319 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1320 = or i32 %compare.local.base1319, 76, !dbg !24
  %compare.low1321 = icmp ugt i32 %compare.low.base1207, %compare.local1320, !dbg !24
  %compare.result1322 = select i1 %compare.high.equal1205, i1 %compare.low1321, i1 %compare.high.order1206, !dbg !24
  %776 = select i1 %compare.result1322, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1323 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1324 = or i32 %compare.local.base1323, 77, !dbg !24
  %compare.low1325 = icmp ugt i32 %compare.low.base1207, %compare.local1324, !dbg !24
  %compare.result1326 = select i1 %compare.high.equal1205, i1 %compare.low1325, i1 %compare.high.order1206, !dbg !24
  %777 = select i1 %compare.result1326, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1328 = or i32 %compare.local.base1327, 78, !dbg !24
  %compare.low1329 = icmp ugt i32 %compare.low.base1207, %compare.local1328, !dbg !24
  %compare.result1330 = select i1 %compare.high.equal1205, i1 %compare.low1329, i1 %compare.high.order1206, !dbg !24
  %778 = select i1 %compare.result1330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1208, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1332 = or i32 %compare.local.base1331, 79, !dbg !24
  %compare.low1333 = icmp ugt i32 %compare.low.base1207, %compare.local1332, !dbg !24
  %compare.result1334 = select i1 %compare.high.equal1205, i1 %compare.low1333, i1 %compare.high.order1206, !dbg !24
  %779 = select i1 %compare.result1334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %780 = insertelement <32 x float> poison, float %713, i64 0, !dbg !13
  %781 = insertelement <32 x float> %780, float %714, i64 1, !dbg !13
  %782 = insertelement <32 x float> %781, float %715, i64 2, !dbg !13
  %783 = insertelement <32 x float> %782, float %716, i64 3, !dbg !13
  %784 = insertelement <32 x float> %783, float %717, i64 4, !dbg !13
  %785 = insertelement <32 x float> %784, float %718, i64 5, !dbg !13
  %786 = insertelement <32 x float> %785, float %719, i64 6, !dbg !13
  %787 = insertelement <32 x float> %786, float %720, i64 7, !dbg !13
  %788 = insertelement <32 x float> %787, float %721, i64 8, !dbg !13
  %789 = insertelement <32 x float> %788, float %722, i64 9, !dbg !13
  %790 = insertelement <32 x float> %789, float %723, i64 10, !dbg !13
  %791 = insertelement <32 x float> %790, float %724, i64 11, !dbg !13
  %792 = insertelement <32 x float> %791, float %725, i64 12, !dbg !13
  %793 = insertelement <32 x float> %792, float %726, i64 13, !dbg !13
  %794 = insertelement <32 x float> %793, float %727, i64 14, !dbg !13
  %795 = insertelement <32 x float> %794, float %728, i64 15, !dbg !13
  %796 = insertelement <32 x float> %795, float %729, i64 16, !dbg !13
  %797 = insertelement <32 x float> %796, float %730, i64 17, !dbg !13
  %798 = insertelement <32 x float> %797, float %731, i64 18, !dbg !13
  %799 = insertelement <32 x float> %798, float %732, i64 19, !dbg !13
  %800 = insertelement <32 x float> %799, float %733, i64 20, !dbg !13
  %801 = insertelement <32 x float> %800, float %734, i64 21, !dbg !13
  %802 = insertelement <32 x float> %801, float %735, i64 22, !dbg !13
  %803 = insertelement <32 x float> %802, float %736, i64 23, !dbg !13
  %804 = insertelement <32 x float> %803, float %737, i64 24, !dbg !13
  %805 = insertelement <32 x float> %804, float %738, i64 25, !dbg !13
  %806 = insertelement <32 x float> %805, float %739, i64 26, !dbg !13
  %807 = insertelement <32 x float> %806, float %740, i64 27, !dbg !13
  %808 = insertelement <32 x float> %807, float %741, i64 28, !dbg !13
  %809 = insertelement <32 x float> %808, float %742, i64 29, !dbg !13
  %810 = insertelement <32 x float> %809, float %743, i64 30, !dbg !13
  %811 = insertelement <32 x float> %810, float %744, i64 31, !dbg !13
  %812 = insertvalue [2 x <32 x float>] poison, <32 x float> %811, 0, !dbg !13
  %813 = insertelement <32 x float> poison, float %748, i64 0, !dbg !13
  %814 = insertelement <32 x float> %813, float %749, i64 1, !dbg !13
  %815 = insertelement <32 x float> %814, float %750, i64 2, !dbg !13
  %816 = insertelement <32 x float> %815, float %751, i64 3, !dbg !13
  %817 = insertelement <32 x float> %816, float %752, i64 4, !dbg !13
  %818 = insertelement <32 x float> %817, float %753, i64 5, !dbg !13
  %819 = insertelement <32 x float> %818, float %754, i64 6, !dbg !13
  %820 = insertelement <32 x float> %819, float %755, i64 7, !dbg !13
  %821 = insertelement <32 x float> %820, float %756, i64 8, !dbg !13
  %822 = insertelement <32 x float> %821, float %757, i64 9, !dbg !13
  %823 = insertelement <32 x float> %822, float %758, i64 10, !dbg !13
  %824 = insertelement <32 x float> %823, float %759, i64 11, !dbg !13
  %825 = insertelement <32 x float> %824, float %760, i64 12, !dbg !13
  %826 = insertelement <32 x float> %825, float %761, i64 13, !dbg !13
  %827 = insertelement <32 x float> %826, float %762, i64 14, !dbg !13
  %828 = insertelement <32 x float> %827, float %763, i64 15, !dbg !13
  %829 = insertelement <32 x float> %828, float %764, i64 16, !dbg !13
  %830 = insertelement <32 x float> %829, float %765, i64 17, !dbg !13
  %831 = insertelement <32 x float> %830, float %766, i64 18, !dbg !13
  %832 = insertelement <32 x float> %831, float %767, i64 19, !dbg !13
  %833 = insertelement <32 x float> %832, float %768, i64 20, !dbg !13
  %834 = insertelement <32 x float> %833, float %769, i64 21, !dbg !13
  %835 = insertelement <32 x float> %834, float %770, i64 22, !dbg !13
  %836 = insertelement <32 x float> %835, float %771, i64 23, !dbg !13
  %837 = insertelement <32 x float> %836, float %772, i64 24, !dbg !13
  %838 = insertelement <32 x float> %837, float %773, i64 25, !dbg !13
  %839 = insertelement <32 x float> %838, float %774, i64 26, !dbg !13
  %840 = insertelement <32 x float> %839, float %775, i64 27, !dbg !13
  %841 = insertelement <32 x float> %840, float %776, i64 28, !dbg !13
  %842 = insertelement <32 x float> %841, float %777, i64 29, !dbg !13
  %843 = insertelement <32 x float> %842, float %778, i64 30, !dbg !13
  %844 = insertelement <32 x float> %843, float %779, i64 31, !dbg !13
  %845 = insertvalue [2 x <32 x float>] %812, <32 x float> %844, 1, !dbg !13
  br label %846, !dbg !13

846:                                              ; preds = %701, %702
  %847 = phi [2 x <32 x float>] [ %845, %702 ], [ zeroinitializer, %701 ], !dbg !13
  br label %848, !dbg !13

848:                                              ; preds = %846
  %849 = extractvalue [2 x <32 x float>] %847, 0, !dbg !26
  %850 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %851 = fadd <4 x float> %638, %850, !dbg !26
  %852 = extractelement <4 x float> %851, i64 0, !dbg !26
  %853 = extractelement <4 x float> %851, i64 1, !dbg !26
  %854 = extractelement <4 x float> %851, i64 2, !dbg !26
  %855 = extractelement <4 x float> %851, i64 3, !dbg !26
  %856 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %857 = fadd <4 x float> %642, %856, !dbg !26
  %858 = extractelement <4 x float> %857, i64 0, !dbg !26
  %859 = extractelement <4 x float> %857, i64 1, !dbg !26
  %860 = extractelement <4 x float> %857, i64 2, !dbg !26
  %861 = extractelement <4 x float> %857, i64 3, !dbg !26
  %862 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %863 = fadd <4 x float> %646, %862, !dbg !26
  %864 = extractelement <4 x float> %863, i64 0, !dbg !26
  %865 = extractelement <4 x float> %863, i64 1, !dbg !26
  %866 = extractelement <4 x float> %863, i64 2, !dbg !26
  %867 = extractelement <4 x float> %863, i64 3, !dbg !26
  %868 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %869 = fadd <4 x float> %650, %868, !dbg !26
  %870 = extractelement <4 x float> %869, i64 0, !dbg !26
  %871 = extractelement <4 x float> %869, i64 1, !dbg !26
  %872 = extractelement <4 x float> %869, i64 2, !dbg !26
  %873 = extractelement <4 x float> %869, i64 3, !dbg !26
  %874 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %875 = fadd <4 x float> %654, %874, !dbg !26
  %876 = extractelement <4 x float> %875, i64 0, !dbg !26
  %877 = extractelement <4 x float> %875, i64 1, !dbg !26
  %878 = extractelement <4 x float> %875, i64 2, !dbg !26
  %879 = extractelement <4 x float> %875, i64 3, !dbg !26
  %880 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %881 = fadd <4 x float> %658, %880, !dbg !26
  %882 = extractelement <4 x float> %881, i64 0, !dbg !26
  %883 = extractelement <4 x float> %881, i64 1, !dbg !26
  %884 = extractelement <4 x float> %881, i64 2, !dbg !26
  %885 = extractelement <4 x float> %881, i64 3, !dbg !26
  %886 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %887 = fadd <4 x float> %662, %886, !dbg !26
  %888 = extractelement <4 x float> %887, i64 0, !dbg !26
  %889 = extractelement <4 x float> %887, i64 1, !dbg !26
  %890 = extractelement <4 x float> %887, i64 2, !dbg !26
  %891 = extractelement <4 x float> %887, i64 3, !dbg !26
  %892 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %893 = fadd <4 x float> %666, %892, !dbg !26
  %894 = extractelement <4 x float> %893, i64 0, !dbg !26
  %895 = extractelement <4 x float> %893, i64 1, !dbg !26
  %896 = extractelement <4 x float> %893, i64 2, !dbg !26
  %897 = extractelement <4 x float> %893, i64 3, !dbg !26
  %898 = extractvalue [2 x <32 x float>] %847, 1, !dbg !26
  %899 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %900 = fadd <4 x float> %670, %899, !dbg !26
  %901 = extractelement <4 x float> %900, i64 0, !dbg !26
  %902 = extractelement <4 x float> %900, i64 1, !dbg !26
  %903 = extractelement <4 x float> %900, i64 2, !dbg !26
  %904 = extractelement <4 x float> %900, i64 3, !dbg !26
  %905 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %906 = fadd <4 x float> %674, %905, !dbg !26
  %907 = extractelement <4 x float> %906, i64 0, !dbg !26
  %908 = extractelement <4 x float> %906, i64 1, !dbg !26
  %909 = extractelement <4 x float> %906, i64 2, !dbg !26
  %910 = extractelement <4 x float> %906, i64 3, !dbg !26
  %911 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %912 = fadd <4 x float> %678, %911, !dbg !26
  %913 = extractelement <4 x float> %912, i64 0, !dbg !26
  %914 = extractelement <4 x float> %912, i64 1, !dbg !26
  %915 = extractelement <4 x float> %912, i64 2, !dbg !26
  %916 = extractelement <4 x float> %912, i64 3, !dbg !26
  %917 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %918 = fadd <4 x float> %682, %917, !dbg !26
  %919 = extractelement <4 x float> %918, i64 0, !dbg !26
  %920 = extractelement <4 x float> %918, i64 1, !dbg !26
  %921 = extractelement <4 x float> %918, i64 2, !dbg !26
  %922 = extractelement <4 x float> %918, i64 3, !dbg !26
  %923 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %924 = fadd <4 x float> %686, %923, !dbg !26
  %925 = extractelement <4 x float> %924, i64 0, !dbg !26
  %926 = extractelement <4 x float> %924, i64 1, !dbg !26
  %927 = extractelement <4 x float> %924, i64 2, !dbg !26
  %928 = extractelement <4 x float> %924, i64 3, !dbg !26
  %929 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %930 = fadd <4 x float> %690, %929, !dbg !26
  %931 = extractelement <4 x float> %930, i64 0, !dbg !26
  %932 = extractelement <4 x float> %930, i64 1, !dbg !26
  %933 = extractelement <4 x float> %930, i64 2, !dbg !26
  %934 = extractelement <4 x float> %930, i64 3, !dbg !26
  %935 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %936 = fadd <4 x float> %694, %935, !dbg !26
  %937 = extractelement <4 x float> %936, i64 0, !dbg !26
  %938 = extractelement <4 x float> %936, i64 1, !dbg !26
  %939 = extractelement <4 x float> %936, i64 2, !dbg !26
  %940 = extractelement <4 x float> %936, i64 3, !dbg !26
  %941 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %942 = fadd <4 x float> %699, %941, !dbg !26
  %943 = extractelement <4 x float> %942, i64 0, !dbg !26
  %944 = extractelement <4 x float> %942, i64 1, !dbg !26
  %945 = extractelement <4 x float> %942, i64 2, !dbg !26
  %946 = extractelement <4 x float> %942, i64 3, !dbg !26
  %947 = call float @llvm.maxnum.f32(float %852, float 0xFFF0000000000000), !dbg !27
  %948 = call float @llvm.maxnum.f32(float %947, float %853), !dbg !27
  %949 = call float @llvm.maxnum.f32(float %948, float %854), !dbg !27
  %950 = call float @llvm.maxnum.f32(float %949, float %855), !dbg !27
  %951 = call float @llvm.maxnum.f32(float %950, float %858), !dbg !27
  %952 = call float @llvm.maxnum.f32(float %951, float %859), !dbg !27
  %953 = call float @llvm.maxnum.f32(float %952, float %860), !dbg !27
  %954 = call float @llvm.maxnum.f32(float %953, float %861), !dbg !27
  %955 = call float @llvm.maxnum.f32(float %954, float %864), !dbg !27
  %956 = call float @llvm.maxnum.f32(float %955, float %865), !dbg !27
  %957 = call float @llvm.maxnum.f32(float %956, float %866), !dbg !27
  %958 = call float @llvm.maxnum.f32(float %957, float %867), !dbg !27
  %959 = call float @llvm.maxnum.f32(float %958, float %870), !dbg !27
  %960 = call float @llvm.maxnum.f32(float %959, float %871), !dbg !27
  %961 = call float @llvm.maxnum.f32(float %960, float %872), !dbg !27
  %962 = call float @llvm.maxnum.f32(float %961, float %873), !dbg !27
  %963 = call float @llvm.maxnum.f32(float %962, float %876), !dbg !27
  %964 = call float @llvm.maxnum.f32(float %963, float %877), !dbg !27
  %965 = call float @llvm.maxnum.f32(float %964, float %878), !dbg !27
  %966 = call float @llvm.maxnum.f32(float %965, float %879), !dbg !27
  %967 = call float @llvm.maxnum.f32(float %966, float %882), !dbg !27
  %968 = call float @llvm.maxnum.f32(float %967, float %883), !dbg !27
  %969 = call float @llvm.maxnum.f32(float %968, float %884), !dbg !27
  %970 = call float @llvm.maxnum.f32(float %969, float %885), !dbg !27
  %971 = call float @llvm.maxnum.f32(float %970, float %888), !dbg !27
  %972 = call float @llvm.maxnum.f32(float %971, float %889), !dbg !27
  %973 = call float @llvm.maxnum.f32(float %972, float %890), !dbg !27
  %974 = call float @llvm.maxnum.f32(float %973, float %891), !dbg !27
  %975 = call float @llvm.maxnum.f32(float %974, float %894), !dbg !27
  %976 = call float @llvm.maxnum.f32(float %975, float %895), !dbg !27
  %977 = call float @llvm.maxnum.f32(float %976, float %896), !dbg !27
  %978 = call float @llvm.maxnum.f32(float %977, float %897), !dbg !27
  %979 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %980 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %979), !dbg !27
  %981 = add nuw nsw i32 %980, 64, !dbg !27
  %982 = xor i32 %980, 16, !dbg !27
  %983 = and i32 %981, 192, !dbg !27
  %984 = icmp samesign ult i32 %982, %983, !dbg !27
  %985 = select i1 %984, i32 %982, i32 %980, !dbg !27
  %986 = shl nuw nsw i32 %985, 2, !dbg !27
  %987 = bitcast float %978 to i32, !dbg !27
  %988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %987), !dbg !27
  %989 = bitcast i32 %988 to float, !dbg !27
  %990 = call float @llvm.maxnum.f32(float %978, float %989), !dbg !27
  %991 = xor i32 %980, 32, !dbg !27
  %992 = icmp samesign ult i32 %991, %983, !dbg !27
  %993 = select i1 %992, i32 %991, i32 %980, !dbg !27
  %994 = shl nuw nsw i32 %993, 2, !dbg !27
  %995 = bitcast float %990 to i32, !dbg !27
  %996 = call i32 @llvm.amdgcn.ds.bpermute(i32 %994, i32 %995), !dbg !27
  %997 = bitcast i32 %996 to float, !dbg !27
  %998 = call float @llvm.maxnum.f32(float %990, float %997), !dbg !27
  %999 = extractvalue [2 x <1 x float>] %64, 0, !dbg !27
  %1000 = insertelement <1 x float> poison, float %998, i64 0, !dbg !27
  %1001 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1000, <1 x float> %999), !dbg !27
  %1002 = call float @llvm.maxnum.f32(float %901, float 0xFFF0000000000000), !dbg !27
  %1003 = call float @llvm.maxnum.f32(float %1002, float %902), !dbg !27
  %1004 = call float @llvm.maxnum.f32(float %1003, float %903), !dbg !27
  %1005 = call float @llvm.maxnum.f32(float %1004, float %904), !dbg !27
  %1006 = call float @llvm.maxnum.f32(float %1005, float %907), !dbg !27
  %1007 = call float @llvm.maxnum.f32(float %1006, float %908), !dbg !27
  %1008 = call float @llvm.maxnum.f32(float %1007, float %909), !dbg !27
  %1009 = call float @llvm.maxnum.f32(float %1008, float %910), !dbg !27
  %1010 = call float @llvm.maxnum.f32(float %1009, float %913), !dbg !27
  %1011 = call float @llvm.maxnum.f32(float %1010, float %914), !dbg !27
  %1012 = call float @llvm.maxnum.f32(float %1011, float %915), !dbg !27
  %1013 = call float @llvm.maxnum.f32(float %1012, float %916), !dbg !27
  %1014 = call float @llvm.maxnum.f32(float %1013, float %919), !dbg !27
  %1015 = call float @llvm.maxnum.f32(float %1014, float %920), !dbg !27
  %1016 = call float @llvm.maxnum.f32(float %1015, float %921), !dbg !27
  %1017 = call float @llvm.maxnum.f32(float %1016, float %922), !dbg !27
  %1018 = call float @llvm.maxnum.f32(float %1017, float %925), !dbg !27
  %1019 = call float @llvm.maxnum.f32(float %1018, float %926), !dbg !27
  %1020 = call float @llvm.maxnum.f32(float %1019, float %927), !dbg !27
  %1021 = call float @llvm.maxnum.f32(float %1020, float %928), !dbg !27
  %1022 = call float @llvm.maxnum.f32(float %1021, float %931), !dbg !27
  %1023 = call float @llvm.maxnum.f32(float %1022, float %932), !dbg !27
  %1024 = call float @llvm.maxnum.f32(float %1023, float %933), !dbg !27
  %1025 = call float @llvm.maxnum.f32(float %1024, float %934), !dbg !27
  %1026 = call float @llvm.maxnum.f32(float %1025, float %937), !dbg !27
  %1027 = call float @llvm.maxnum.f32(float %1026, float %938), !dbg !27
  %1028 = call float @llvm.maxnum.f32(float %1027, float %939), !dbg !27
  %1029 = call float @llvm.maxnum.f32(float %1028, float %940), !dbg !27
  %1030 = call float @llvm.maxnum.f32(float %1029, float %943), !dbg !27
  %1031 = call float @llvm.maxnum.f32(float %1030, float %944), !dbg !27
  %1032 = call float @llvm.maxnum.f32(float %1031, float %945), !dbg !27
  %1033 = call float @llvm.maxnum.f32(float %1032, float %946), !dbg !27
  %1034 = bitcast float %1033 to i32, !dbg !27
  %1035 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %1034), !dbg !27
  %1036 = bitcast i32 %1035 to float, !dbg !27
  %1037 = call float @llvm.maxnum.f32(float %1033, float %1036), !dbg !27
  %1038 = bitcast float %1037 to i32, !dbg !27
  %1039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %994, i32 %1038), !dbg !27
  %1040 = bitcast i32 %1039 to float, !dbg !27
  %1041 = call float @llvm.maxnum.f32(float %1037, float %1040), !dbg !27
  %1042 = extractvalue [2 x <1 x float>] %64, 1, !dbg !27
  %1043 = insertelement <1 x float> poison, float %1041, i64 0, !dbg !27
  %1044 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1043, <1 x float> %1042), !dbg !27
  %1045 = shufflevector <1 x float> %1001, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1046 = fsub <4 x float> %851, %1045, !dbg !28
  %1047 = fsub <4 x float> %857, %1045, !dbg !28
  %1048 = fsub <4 x float> %863, %1045, !dbg !28
  %1049 = fsub <4 x float> %869, %1045, !dbg !28
  %1050 = fsub <4 x float> %875, %1045, !dbg !28
  %1051 = fsub <4 x float> %881, %1045, !dbg !28
  %1052 = fsub <4 x float> %887, %1045, !dbg !28
  %1053 = fsub <4 x float> %893, %1045, !dbg !28
  %1054 = shufflevector <1 x float> %1044, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1055 = fsub <4 x float> %900, %1054, !dbg !28
  %1056 = fsub <4 x float> %906, %1054, !dbg !28
  %1057 = fsub <4 x float> %912, %1054, !dbg !28
  %1058 = fsub <4 x float> %918, %1054, !dbg !28
  %1059 = fsub <4 x float> %924, %1054, !dbg !28
  %1060 = fsub <4 x float> %930, %1054, !dbg !28
  %1061 = fsub <4 x float> %936, %1054, !dbg !28
  %1062 = fsub <4 x float> %942, %1054, !dbg !28
  %1063 = fsub <1 x float> %999, %1001, !dbg !29
  %1064 = extractelement <1 x float> %1063, i64 0, !dbg !30
  %1065 = call float @llvm.amdgcn.exp2.f32(float %1064), !dbg !30
  %1066 = insertelement <4 x float> poison, float %1065, i64 0, !dbg !35
  %1067 = shufflevector <4 x float> %1066, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1068 = extractvalue [2 x <1 x float>] %63, 0, !dbg !31
  %1069 = extractelement <1 x float> %1068, i64 0, !dbg !31
  %1070 = fmul float %1069, %1065, !dbg !31
  %1071 = fsub <1 x float> %1042, %1044, !dbg !29
  %1072 = extractelement <1 x float> %1071, i64 0, !dbg !30
  %1073 = call float @llvm.amdgcn.exp2.f32(float %1072), !dbg !30
  %1074 = insertelement <4 x float> poison, float %1073, i64 0, !dbg !35
  %1075 = shufflevector <4 x float> %1074, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1076 = extractvalue [2 x <1 x float>] %63, 1, !dbg !31
  %1077 = extractelement <1 x float> %1076, i64 0, !dbg !31
  %1078 = fmul float %1077, %1073, !dbg !31
  %1079 = extractelement <4 x float> %1046, i64 0, !dbg !32
  %1080 = call float @llvm.amdgcn.exp2.f32(float %1079), !dbg !32
  %1081 = insertelement <4 x float> poison, float %1080, i64 0, !dbg !34
  %1082 = extractelement <4 x float> %1046, i64 1, !dbg !32
  %1083 = call float @llvm.amdgcn.exp2.f32(float %1082), !dbg !32
  %1084 = insertelement <4 x float> %1081, float %1083, i64 1, !dbg !34
  %1085 = fadd float %1080, %1083, !dbg !33
  %1086 = extractelement <4 x float> %1046, i64 2, !dbg !32
  %1087 = call float @llvm.amdgcn.exp2.f32(float %1086), !dbg !32
  %1088 = insertelement <4 x float> %1084, float %1087, i64 2, !dbg !34
  %1089 = fadd float %1085, %1087, !dbg !33
  %1090 = extractelement <4 x float> %1046, i64 3, !dbg !32
  %1091 = call float @llvm.amdgcn.exp2.f32(float %1090), !dbg !32
  %1092 = insertelement <4 x float> %1088, float %1091, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1092) #9, !dbg !34, !frisk.register_lifetimes !1
  %1093 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1093) #9, !dbg !33, !frisk.register_lifetimes !1
  %1094 = fadd float %1089, %1091, !dbg !33
  %1095 = extractelement <4 x float> %1047, i64 0, !dbg !32
  %1096 = call float @llvm.amdgcn.exp2.f32(float %1095), !dbg !32
  %1097 = insertelement <4 x float> poison, float %1096, i64 0, !dbg !34
  %1098 = fadd float %1094, %1096, !dbg !33
  %1099 = extractelement <4 x float> %1047, i64 1, !dbg !32
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !32
  %1101 = insertelement <4 x float> %1097, float %1100, i64 1, !dbg !34
  %1102 = fadd float %1098, %1100, !dbg !33
  %1103 = extractelement <4 x float> %1047, i64 2, !dbg !32
  %1104 = call float @llvm.amdgcn.exp2.f32(float %1103), !dbg !32
  %1105 = insertelement <4 x float> %1101, float %1104, i64 2, !dbg !34
  %1106 = fadd float %1102, %1104, !dbg !33
  %1107 = extractelement <4 x float> %1047, i64 3, !dbg !32
  %1108 = call float @llvm.amdgcn.exp2.f32(float %1107), !dbg !32
  %1109 = insertelement <4 x float> %1105, float %1108, i64 3, !dbg !34
  %math.result306 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1109) #9, !dbg !34, !frisk.register_lifetimes !1
  %1110 = fptrunc <4 x float> %math.result306 to <4 x half>, !dbg !34
  %math.packed307 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1110) #9, !dbg !33, !frisk.register_lifetimes !1
  %1111 = fadd float %1106, %1108, !dbg !33
  %1112 = extractelement <4 x float> %1048, i64 0, !dbg !32
  %1113 = call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !32
  %1114 = insertelement <4 x float> poison, float %1113, i64 0, !dbg !34
  %1115 = fadd float %1111, %1113, !dbg !33
  %1116 = extractelement <4 x float> %1048, i64 1, !dbg !32
  %1117 = call float @llvm.amdgcn.exp2.f32(float %1116), !dbg !32
  %1118 = insertelement <4 x float> %1114, float %1117, i64 1, !dbg !34
  %1119 = fadd float %1115, %1117, !dbg !33
  %1120 = extractelement <4 x float> %1048, i64 2, !dbg !32
  %1121 = call float @llvm.amdgcn.exp2.f32(float %1120), !dbg !32
  %1122 = insertelement <4 x float> %1118, float %1121, i64 2, !dbg !34
  %1123 = fadd float %1119, %1121, !dbg !33
  %1124 = extractelement <4 x float> %1048, i64 3, !dbg !32
  %1125 = call float @llvm.amdgcn.exp2.f32(float %1124), !dbg !32
  %1126 = insertelement <4 x float> %1122, float %1125, i64 3, !dbg !34
  %math.result308 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1126) #9, !dbg !34, !frisk.register_lifetimes !1
  %1127 = fptrunc <4 x float> %math.result308 to <4 x half>, !dbg !34
  %math.packed309 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1127) #9, !dbg !33, !frisk.register_lifetimes !1
  %1128 = fadd float %1123, %1125, !dbg !33
  %1129 = extractelement <4 x float> %1049, i64 0, !dbg !32
  %1130 = call float @llvm.amdgcn.exp2.f32(float %1129), !dbg !32
  %1131 = insertelement <4 x float> poison, float %1130, i64 0, !dbg !34
  %1132 = fadd float %1128, %1130, !dbg !33
  %1133 = extractelement <4 x float> %1049, i64 1, !dbg !32
  %1134 = call float @llvm.amdgcn.exp2.f32(float %1133), !dbg !32
  %1135 = insertelement <4 x float> %1131, float %1134, i64 1, !dbg !34
  %1136 = fadd float %1132, %1134, !dbg !33
  %1137 = extractelement <4 x float> %1049, i64 2, !dbg !32
  %1138 = call float @llvm.amdgcn.exp2.f32(float %1137), !dbg !32
  %1139 = insertelement <4 x float> %1135, float %1138, i64 2, !dbg !34
  %1140 = fadd float %1136, %1138, !dbg !33
  %1141 = extractelement <4 x float> %1049, i64 3, !dbg !32
  %1142 = call float @llvm.amdgcn.exp2.f32(float %1141), !dbg !32
  %1143 = insertelement <4 x float> %1139, float %1142, i64 3, !dbg !34
  %math.result310 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1143) #9, !dbg !34, !frisk.register_lifetimes !1
  %1144 = fptrunc <4 x float> %math.result310 to <4 x half>, !dbg !34
  %math.packed311 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1144) #9, !dbg !33, !frisk.register_lifetimes !1
  %1145 = fadd float %1140, %1142, !dbg !33
  %1146 = extractelement <4 x float> %1050, i64 0, !dbg !32
  %1147 = call float @llvm.amdgcn.exp2.f32(float %1146), !dbg !32
  %1148 = insertelement <4 x float> poison, float %1147, i64 0, !dbg !34
  %1149 = fadd float %1145, %1147, !dbg !33
  %1150 = extractelement <4 x float> %1050, i64 1, !dbg !32
  %1151 = call float @llvm.amdgcn.exp2.f32(float %1150), !dbg !32
  %1152 = insertelement <4 x float> %1148, float %1151, i64 1, !dbg !34
  %1153 = fadd float %1149, %1151, !dbg !33
  %1154 = extractelement <4 x float> %1050, i64 2, !dbg !32
  %1155 = call float @llvm.amdgcn.exp2.f32(float %1154), !dbg !32
  %1156 = insertelement <4 x float> %1152, float %1155, i64 2, !dbg !34
  %1157 = fadd float %1153, %1155, !dbg !33
  %1158 = extractelement <4 x float> %1050, i64 3, !dbg !32
  %1159 = call float @llvm.amdgcn.exp2.f32(float %1158), !dbg !32
  %1160 = insertelement <4 x float> %1156, float %1159, i64 3, !dbg !34
  %math.result312 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1160) #9, !dbg !34, !frisk.register_lifetimes !1
  %1161 = fptrunc <4 x float> %math.result312 to <4 x half>, !dbg !34
  %math.packed313 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1161) #9, !dbg !33, !frisk.register_lifetimes !1
  %1162 = fadd float %1157, %1159, !dbg !33
  %1163 = extractelement <4 x float> %1051, i64 0, !dbg !32
  %1164 = call float @llvm.amdgcn.exp2.f32(float %1163), !dbg !32
  %1165 = insertelement <4 x float> poison, float %1164, i64 0, !dbg !34
  %1166 = fadd float %1162, %1164, !dbg !33
  %1167 = extractelement <4 x float> %1051, i64 1, !dbg !32
  %1168 = call float @llvm.amdgcn.exp2.f32(float %1167), !dbg !32
  %1169 = insertelement <4 x float> %1165, float %1168, i64 1, !dbg !34
  %1170 = fadd float %1166, %1168, !dbg !33
  %1171 = extractelement <4 x float> %1051, i64 2, !dbg !32
  %1172 = call float @llvm.amdgcn.exp2.f32(float %1171), !dbg !32
  %1173 = insertelement <4 x float> %1169, float %1172, i64 2, !dbg !34
  %1174 = fadd float %1170, %1172, !dbg !33
  %1175 = extractelement <4 x float> %1051, i64 3, !dbg !32
  %1176 = call float @llvm.amdgcn.exp2.f32(float %1175), !dbg !32
  %1177 = insertelement <4 x float> %1173, float %1176, i64 3, !dbg !34
  %math.result314 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1177) #9, !dbg !34, !frisk.register_lifetimes !1
  %1178 = fptrunc <4 x float> %math.result314 to <4 x half>, !dbg !34
  %math.packed315 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1178) #9, !dbg !33, !frisk.register_lifetimes !1
  %1179 = fadd float %1174, %1176, !dbg !33
  %1180 = extractelement <4 x float> %1052, i64 0, !dbg !32
  %1181 = call float @llvm.amdgcn.exp2.f32(float %1180), !dbg !32
  %1182 = insertelement <4 x float> poison, float %1181, i64 0, !dbg !34
  %1183 = fadd float %1179, %1181, !dbg !33
  %1184 = extractelement <4 x float> %1052, i64 1, !dbg !32
  %1185 = call float @llvm.amdgcn.exp2.f32(float %1184), !dbg !32
  %1186 = insertelement <4 x float> %1182, float %1185, i64 1, !dbg !34
  %1187 = fadd float %1183, %1185, !dbg !33
  %1188 = extractelement <4 x float> %1052, i64 2, !dbg !32
  %1189 = call float @llvm.amdgcn.exp2.f32(float %1188), !dbg !32
  %1190 = insertelement <4 x float> %1186, float %1189, i64 2, !dbg !34
  %1191 = fadd float %1187, %1189, !dbg !33
  %1192 = extractelement <4 x float> %1052, i64 3, !dbg !32
  %1193 = call float @llvm.amdgcn.exp2.f32(float %1192), !dbg !32
  %1194 = insertelement <4 x float> %1190, float %1193, i64 3, !dbg !34
  %math.result316 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1194) #9, !dbg !34, !frisk.register_lifetimes !1
  %1195 = fptrunc <4 x float> %math.result316 to <4 x half>, !dbg !34
  %math.packed317 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1195) #9, !dbg !33, !frisk.register_lifetimes !1
  %1196 = fadd float %1191, %1193, !dbg !33
  %1197 = extractelement <4 x float> %1053, i64 0, !dbg !32
  %1198 = call float @llvm.amdgcn.exp2.f32(float %1197), !dbg !32
  %1199 = insertelement <4 x float> poison, float %1198, i64 0, !dbg !34
  %1200 = fadd float %1196, %1198, !dbg !33
  %1201 = extractelement <4 x float> %1053, i64 1, !dbg !32
  %1202 = call float @llvm.amdgcn.exp2.f32(float %1201), !dbg !32
  %1203 = insertelement <4 x float> %1199, float %1202, i64 1, !dbg !34
  %1204 = fadd float %1200, %1202, !dbg !33
  %1205 = extractelement <4 x float> %1053, i64 2, !dbg !32
  %1206 = call float @llvm.amdgcn.exp2.f32(float %1205), !dbg !32
  %1207 = insertelement <4 x float> %1203, float %1206, i64 2, !dbg !34
  %1208 = fadd float %1204, %1206, !dbg !33
  %1209 = extractelement <4 x float> %1053, i64 3, !dbg !32
  %1210 = call float @llvm.amdgcn.exp2.f32(float %1209), !dbg !32
  %1211 = insertelement <4 x float> %1207, float %1210, i64 3, !dbg !34
  %math.result318 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1211) #9, !dbg !34, !frisk.register_lifetimes !1
  %1212 = fptrunc <4 x float> %math.result318 to <4 x half>, !dbg !34
  %math.packed319 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1212) #9, !dbg !33, !frisk.register_lifetimes !1
  %1213 = fadd float %1208, %1210, !dbg !33
  %1214 = extractelement <4 x float> %1055, i64 0, !dbg !32
  %1215 = call float @llvm.amdgcn.exp2.f32(float %1214), !dbg !32
  %1216 = insertelement <4 x float> poison, float %1215, i64 0, !dbg !34
  %1217 = extractelement <4 x float> %1055, i64 1, !dbg !32
  %1218 = call float @llvm.amdgcn.exp2.f32(float %1217), !dbg !32
  %1219 = insertelement <4 x float> %1216, float %1218, i64 1, !dbg !34
  %1220 = fadd float %1215, %1218, !dbg !33
  %1221 = extractelement <4 x float> %1055, i64 2, !dbg !32
  %1222 = call float @llvm.amdgcn.exp2.f32(float %1221), !dbg !32
  %1223 = insertelement <4 x float> %1219, float %1222, i64 2, !dbg !34
  %1224 = fadd float %1220, %1222, !dbg !33
  %1225 = extractelement <4 x float> %1055, i64 3, !dbg !32
  %1226 = call float @llvm.amdgcn.exp2.f32(float %1225), !dbg !32
  %1227 = insertelement <4 x float> %1223, float %1226, i64 3, !dbg !34
  %math.result320 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1227) #9, !dbg !34, !frisk.register_lifetimes !1
  %1228 = fptrunc <4 x float> %math.result320 to <4 x half>, !dbg !34
  %math.packed321 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1228) #9, !dbg !33, !frisk.register_lifetimes !1
  %1229 = fadd float %1224, %1226, !dbg !33
  %1230 = extractelement <4 x float> %1056, i64 0, !dbg !32
  %1231 = call float @llvm.amdgcn.exp2.f32(float %1230), !dbg !32
  %1232 = insertelement <4 x float> poison, float %1231, i64 0, !dbg !34
  %1233 = fadd float %1229, %1231, !dbg !33
  %1234 = extractelement <4 x float> %1056, i64 1, !dbg !32
  %1235 = call float @llvm.amdgcn.exp2.f32(float %1234), !dbg !32
  %1236 = insertelement <4 x float> %1232, float %1235, i64 1, !dbg !34
  %1237 = fadd float %1233, %1235, !dbg !33
  %1238 = extractelement <4 x float> %1056, i64 2, !dbg !32
  %1239 = call float @llvm.amdgcn.exp2.f32(float %1238), !dbg !32
  %1240 = insertelement <4 x float> %1236, float %1239, i64 2, !dbg !34
  %1241 = fadd float %1237, %1239, !dbg !33
  %1242 = extractelement <4 x float> %1056, i64 3, !dbg !32
  %1243 = call float @llvm.amdgcn.exp2.f32(float %1242), !dbg !32
  %1244 = insertelement <4 x float> %1240, float %1243, i64 3, !dbg !34
  %math.result322 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1244) #9, !dbg !34, !frisk.register_lifetimes !1
  %1245 = fptrunc <4 x float> %math.result322 to <4 x half>, !dbg !34
  %math.packed323 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1245) #9, !dbg !33, !frisk.register_lifetimes !1
  %1246 = fadd float %1241, %1243, !dbg !33
  %1247 = extractelement <4 x float> %1057, i64 0, !dbg !32
  %1248 = call float @llvm.amdgcn.exp2.f32(float %1247), !dbg !32
  %1249 = insertelement <4 x float> poison, float %1248, i64 0, !dbg !34
  %1250 = fadd float %1246, %1248, !dbg !33
  %1251 = extractelement <4 x float> %1057, i64 1, !dbg !32
  %1252 = call float @llvm.amdgcn.exp2.f32(float %1251), !dbg !32
  %1253 = insertelement <4 x float> %1249, float %1252, i64 1, !dbg !34
  %1254 = fadd float %1250, %1252, !dbg !33
  %1255 = extractelement <4 x float> %1057, i64 2, !dbg !32
  %1256 = call float @llvm.amdgcn.exp2.f32(float %1255), !dbg !32
  %1257 = insertelement <4 x float> %1253, float %1256, i64 2, !dbg !34
  %1258 = fadd float %1254, %1256, !dbg !33
  %1259 = extractelement <4 x float> %1057, i64 3, !dbg !32
  %1260 = call float @llvm.amdgcn.exp2.f32(float %1259), !dbg !32
  %1261 = insertelement <4 x float> %1257, float %1260, i64 3, !dbg !34
  %math.result324 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1261) #9, !dbg !34, !frisk.register_lifetimes !1
  %1262 = fptrunc <4 x float> %math.result324 to <4 x half>, !dbg !34
  %math.packed325 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1262) #9, !dbg !33, !frisk.register_lifetimes !1
  %1263 = fadd float %1258, %1260, !dbg !33
  %1264 = extractelement <4 x float> %1058, i64 0, !dbg !32
  %1265 = call float @llvm.amdgcn.exp2.f32(float %1264), !dbg !32
  %1266 = insertelement <4 x float> poison, float %1265, i64 0, !dbg !34
  %1267 = fadd float %1263, %1265, !dbg !33
  %1268 = extractelement <4 x float> %1058, i64 1, !dbg !32
  %1269 = call float @llvm.amdgcn.exp2.f32(float %1268), !dbg !32
  %1270 = insertelement <4 x float> %1266, float %1269, i64 1, !dbg !34
  %1271 = fadd float %1267, %1269, !dbg !33
  %1272 = extractelement <4 x float> %1058, i64 2, !dbg !32
  %1273 = call float @llvm.amdgcn.exp2.f32(float %1272), !dbg !32
  %1274 = insertelement <4 x float> %1270, float %1273, i64 2, !dbg !34
  %1275 = fadd float %1271, %1273, !dbg !33
  %1276 = extractelement <4 x float> %1058, i64 3, !dbg !32
  %1277 = call float @llvm.amdgcn.exp2.f32(float %1276), !dbg !32
  %1278 = insertelement <4 x float> %1274, float %1277, i64 3, !dbg !34
  %math.result326 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1278) #9, !dbg !34, !frisk.register_lifetimes !1
  %1279 = fptrunc <4 x float> %math.result326 to <4 x half>, !dbg !34
  %math.packed327 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1279) #9, !dbg !33, !frisk.register_lifetimes !1
  %1280 = fadd float %1275, %1277, !dbg !33
  %1281 = extractelement <4 x float> %1059, i64 0, !dbg !32
  %1282 = call float @llvm.amdgcn.exp2.f32(float %1281), !dbg !32
  %1283 = insertelement <4 x float> poison, float %1282, i64 0, !dbg !34
  %1284 = fadd float %1280, %1282, !dbg !33
  %1285 = extractelement <4 x float> %1059, i64 1, !dbg !32
  %1286 = call float @llvm.amdgcn.exp2.f32(float %1285), !dbg !32
  %1287 = insertelement <4 x float> %1283, float %1286, i64 1, !dbg !34
  %1288 = fadd float %1284, %1286, !dbg !33
  %1289 = extractelement <4 x float> %1059, i64 2, !dbg !32
  %1290 = call float @llvm.amdgcn.exp2.f32(float %1289), !dbg !32
  %1291 = insertelement <4 x float> %1287, float %1290, i64 2, !dbg !34
  %1292 = fadd float %1288, %1290, !dbg !33
  %1293 = extractelement <4 x float> %1059, i64 3, !dbg !32
  %1294 = call float @llvm.amdgcn.exp2.f32(float %1293), !dbg !32
  %1295 = insertelement <4 x float> %1291, float %1294, i64 3, !dbg !34
  %math.result328 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1295) #9, !dbg !34, !frisk.register_lifetimes !1
  %1296 = fptrunc <4 x float> %math.result328 to <4 x half>, !dbg !34
  %math.packed329 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1296) #9, !dbg !33, !frisk.register_lifetimes !1
  %1297 = fadd float %1292, %1294, !dbg !33
  %1298 = extractelement <4 x float> %1060, i64 0, !dbg !32
  %1299 = call float @llvm.amdgcn.exp2.f32(float %1298), !dbg !32
  %1300 = insertelement <4 x float> poison, float %1299, i64 0, !dbg !34
  %1301 = fadd float %1297, %1299, !dbg !33
  %1302 = extractelement <4 x float> %1060, i64 1, !dbg !32
  %1303 = call float @llvm.amdgcn.exp2.f32(float %1302), !dbg !32
  %1304 = insertelement <4 x float> %1300, float %1303, i64 1, !dbg !34
  %1305 = fadd float %1301, %1303, !dbg !33
  %1306 = extractelement <4 x float> %1060, i64 2, !dbg !32
  %1307 = call float @llvm.amdgcn.exp2.f32(float %1306), !dbg !32
  %1308 = insertelement <4 x float> %1304, float %1307, i64 2, !dbg !34
  %1309 = fadd float %1305, %1307, !dbg !33
  %1310 = extractelement <4 x float> %1060, i64 3, !dbg !32
  %1311 = call float @llvm.amdgcn.exp2.f32(float %1310), !dbg !32
  %1312 = insertelement <4 x float> %1308, float %1311, i64 3, !dbg !34
  %math.result330 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1312) #9, !dbg !34, !frisk.register_lifetimes !1
  %1313 = fptrunc <4 x float> %math.result330 to <4 x half>, !dbg !34
  %math.packed331 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1313) #9, !dbg !33, !frisk.register_lifetimes !1
  %1314 = fadd float %1309, %1311, !dbg !33
  %1315 = extractelement <4 x float> %1061, i64 0, !dbg !32
  %1316 = call float @llvm.amdgcn.exp2.f32(float %1315), !dbg !32
  %1317 = insertelement <4 x float> poison, float %1316, i64 0, !dbg !34
  %1318 = fadd float %1314, %1316, !dbg !33
  %1319 = extractelement <4 x float> %1061, i64 1, !dbg !32
  %1320 = call float @llvm.amdgcn.exp2.f32(float %1319), !dbg !32
  %1321 = insertelement <4 x float> %1317, float %1320, i64 1, !dbg !34
  %1322 = fadd float %1318, %1320, !dbg !33
  %1323 = extractelement <4 x float> %1061, i64 2, !dbg !32
  %1324 = call float @llvm.amdgcn.exp2.f32(float %1323), !dbg !32
  %1325 = insertelement <4 x float> %1321, float %1324, i64 2, !dbg !34
  %1326 = fadd float %1322, %1324, !dbg !33
  %1327 = extractelement <4 x float> %1061, i64 3, !dbg !32
  %1328 = call float @llvm.amdgcn.exp2.f32(float %1327), !dbg !32
  %1329 = insertelement <4 x float> %1325, float %1328, i64 3, !dbg !34
  %math.result332 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1329) #9, !dbg !34, !frisk.register_lifetimes !1
  %1330 = fptrunc <4 x float> %math.result332 to <4 x half>, !dbg !34
  %math.packed333 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1330) #9, !dbg !33, !frisk.register_lifetimes !1
  %1331 = fadd float %1326, %1328, !dbg !33
  %1332 = extractelement <4 x float> %1062, i64 0, !dbg !32
  %1333 = call float @llvm.amdgcn.exp2.f32(float %1332), !dbg !32
  %1334 = insertelement <4 x float> poison, float %1333, i64 0, !dbg !34
  %1335 = fadd float %1331, %1333, !dbg !33
  %1336 = extractelement <4 x float> %1062, i64 1, !dbg !32
  %1337 = call float @llvm.amdgcn.exp2.f32(float %1336), !dbg !32
  %1338 = insertelement <4 x float> %1334, float %1337, i64 1, !dbg !34
  %1339 = fadd float %1335, %1337, !dbg !33
  %1340 = extractelement <4 x float> %1062, i64 2, !dbg !32
  %1341 = call float @llvm.amdgcn.exp2.f32(float %1340), !dbg !32
  %1342 = insertelement <4 x float> %1338, float %1341, i64 2, !dbg !34
  %1343 = fadd float %1339, %1341, !dbg !33
  %1344 = extractelement <4 x float> %1062, i64 3, !dbg !32
  %1345 = call float @llvm.amdgcn.exp2.f32(float %1344), !dbg !32
  %1346 = insertelement <4 x float> %1342, float %1345, i64 3, !dbg !34
  %math.result334 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1346) #9, !dbg !34, !frisk.register_lifetimes !1
  %1347 = fptrunc <4 x float> %math.result334 to <4 x half>, !dbg !34
  %math.packed335 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1347) #9, !dbg !33, !frisk.register_lifetimes !1
  %1348 = fadd float %1343, %1345, !dbg !33
  %1349 = bitcast float %1213 to i32, !dbg !33
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %1349), !dbg !33
  %1351 = bitcast i32 %1350 to float, !dbg !33
  %1352 = fadd float %1213, %1351, !dbg !33
  %1353 = bitcast float %1352 to i32, !dbg !33
  %1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %994, i32 %1353), !dbg !33
  %1355 = bitcast i32 %1354 to float, !dbg !33
  %1356 = fadd float %1352, %1355, !dbg !33
  %1357 = fadd float %1356, %1070, !dbg !33
  %1358 = insertelement <1 x float> poison, float %1357, i64 0, !dbg !7
  %1359 = insertvalue [2 x <1 x float>] poison, <1 x float> %1358, 0, !dbg !7
  %1360 = bitcast float %1348 to i32, !dbg !33
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %1360), !dbg !33
  %1362 = bitcast i32 %1361 to float, !dbg !33
  %1363 = fadd float %1348, %1362, !dbg !33
  %1364 = bitcast float %1363 to i32, !dbg !33
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %994, i32 %1364), !dbg !33
  %1366 = bitcast i32 %1365 to float, !dbg !33
  %1367 = fadd float %1363, %1366, !dbg !33
  %1368 = fadd float %1367, %1078, !dbg !33
  %1369 = insertelement <1 x float> poison, float %1368, i64 0, !dbg !7
  %1370 = insertvalue [2 x <1 x float>] %1359, <1 x float> %1369, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread786 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !14, !range !39, !frisk.register_lifetimes !1
  %address.local787 = shl nuw nsw i32 %address.thread786, 3, !dbg !14
  %address.local788 = and i32 %address.local787, 120, !dbg !14
  %address.local789 = shl nuw nsw i32 %address.thread786, 5, !dbg !14
  %address.local790 = and i32 %address.local789, 1536, !dbg !14
  %address.local791 = or disjoint i32 %address.local788, %address.local790, !dbg !14
  %address.local792 = zext nneg i32 %address.local791 to i64, !dbg !14
  %address.local793 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local792, !dbg !14
  %1371 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local793, i32 0) #0, !dbg !14
  %1372 = bitcast <4 x i32> %1371 to <8 x half>, !dbg !14
  %address.local794 = zext nneg i32 %address.thread786 to i64, !dbg !12
  %address.local795 = and i64 %address.local794, 60, !dbg !14
  %address.local796 = add nuw nsw i64 %address.local795, %address.local794, !dbg !14
  %address.local797 = shl nuw nsw i64 %address.local796, 3, !dbg !14
  %address.local798 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !14
  %address.local799 = and i64 %address.local794, 63, !dbg !12
  %address.local800 = lshr i64 %address.local799, 4, !dbg !14
  %address.local801 = mul nuw nsw i64 %address.local800, 384, !dbg !14
  %address.local802 = getelementptr [2 x i8], ptr addrspace(3) %address.local798, i64 %address.local801, !dbg !14
  %address.local803 = lshr i64 %address.local794, 2, !dbg !17
  %address.local804 = mul nsw i64 %address.local803, -32, !dbg !14
  %address.local805 = getelementptr [2 x i8], ptr addrspace(3) %address.local802, i64 %address.local804, !dbg !14
  %address.local806 = getelementptr i8, ptr addrspace(3) %address.local805, i64 256, !dbg !14
  %1373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local806, i32 0) #0, !dbg !14
  %1374 = bitcast <4 x i32> %1373 to <8 x half>, !dbg !14
  %address.local807 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !14
  %address.local808 = getelementptr [2 x i8], ptr addrspace(3) %address.local807, i64 %address.local801, !dbg !14
  %address.local809 = getelementptr [2 x i8], ptr addrspace(3) %address.local808, i64 %address.local804, !dbg !14
  %address.local810 = getelementptr i8, ptr addrspace(3) %address.local809, i64 512, !dbg !14
  %1375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local810, i32 0) #0, !dbg !14
  %1376 = bitcast <4 x i32> %1375 to <8 x half>, !dbg !14
  %address.local811 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !14
  %address.local812 = getelementptr [2 x i8], ptr addrspace(3) %address.local811, i64 %address.local801, !dbg !14
  %address.local813 = getelementptr [2 x i8], ptr addrspace(3) %address.local812, i64 %address.local804, !dbg !14
  %address.local814 = getelementptr i8, ptr addrspace(3) %address.local813, i64 768, !dbg !14
  %1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local814, i32 0) #0, !dbg !14
  %1378 = bitcast <4 x i32> %1377 to <8 x half>, !dbg !14
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local792, !dbg !14
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local815, i64 4096, !dbg !14
  %1379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !14
  %1380 = bitcast <4 x i32> %1379 to <8 x half>, !dbg !14
  %address.local817 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !14
  %address.local818 = getelementptr [2 x i8], ptr addrspace(3) %address.local817, i64 %address.local801, !dbg !14
  %address.local819 = getelementptr [2 x i8], ptr addrspace(3) %address.local818, i64 %address.local804, !dbg !14
  %address.local820 = getelementptr i8, ptr addrspace(3) %address.local819, i64 4352, !dbg !14
  %1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !14
  %1382 = bitcast <4 x i32> %1381 to <8 x half>, !dbg !14
  %address.local821 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !14
  %address.local822 = getelementptr [2 x i8], ptr addrspace(3) %address.local821, i64 %address.local801, !dbg !14
  %address.local823 = getelementptr [2 x i8], ptr addrspace(3) %address.local822, i64 %address.local804, !dbg !14
  %address.local824 = getelementptr i8, ptr addrspace(3) %address.local823, i64 4608, !dbg !14
  %1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local824, i32 0) #0, !dbg !14
  %1384 = bitcast <4 x i32> %1383 to <8 x half>, !dbg !14
  %address.local825 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !14
  %address.local826 = getelementptr [2 x i8], ptr addrspace(3) %address.local825, i64 %address.local801, !dbg !14
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) %address.local826, i64 %address.local804, !dbg !14
  %address.local828 = getelementptr i8, ptr addrspace(3) %address.local827, i64 4864, !dbg !14
  %1385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local828, i32 0) #0, !dbg !14
  %1386 = bitcast <4 x i32> %1385 to <8 x half>, !dbg !14
  %1387 = add nuw nsw i64 %101, %15, !dbg !18
  %1388 = shl i64 %1387, 19, !dbg !18
  %1389 = and i64 %1388, 16252928, !dbg !18
  %1390 = add i64 %1389, %107, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1391 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1390, !dbg !18
  %1392 = getelementptr i8, ptr addrspace(1) %1391, i64 64, !dbg !18
  %address.local829 = shl nuw nsw i64 %address.local794, 3, !dbg !11
  %address.local830 = and i64 %address.local829, 1536, !dbg !11
  %address.local831 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local830, !dbg !11
  %address.local832 = ptrtoint ptr addrspace(3) %address.local831 to i64, !dbg !11
  %address.local833 = trunc i64 %address.local832 to i32, !dbg !11
  %1393 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local833), !dbg !18
  %1394 = zext i32 %1393 to i64, !dbg !18
  %1395 = inttoptr i64 %1394 to ptr addrspace(3), !dbg !18
  %address.local834 = shl nuw nsw i64 %address.local794, 5, !dbg !18
  %address.local835 = and i64 %address.local834, 8064, !dbg !18
  %address.local836 = shl nuw nsw i64 %address.local794, 3, !dbg !18
  %address.local837 = and i64 %address.local836, 24, !dbg !18
  %address.local838 = or disjoint i64 %address.local835, %address.local837, !dbg !18
  %address.local839 = getelementptr [2 x i8], ptr addrspace(1) %1392, i64 %address.local838, !dbg !18
  %1396 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1397 = ptrtoint ptr addrspace(1) %address.local839 to i64, !dbg !18
  %1398 = sub i64 %1397, %1396, !dbg !18
  %dma.buffer.offset1470 = trunc i64 %1398 to i32, !dbg !18
  %1399 = trunc i64 %1396 to i32, !dbg !18
  %1400 = insertelement <4 x i32> poison, i32 %1399, i64 0, !dbg !18
  %1401 = lshr i64 %1396, 32, !dbg !18
  %1402 = trunc i64 %1401 to i32, !dbg !18
  %1403 = insertelement <4 x i32> %1400, i32 %1402, i64 1, !dbg !18
  %1404 = insertelement <4 x i32> %1403, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1471 = insertelement <4 x i32> %1404, i32 131072, i64 3, !dbg !18
  %1405 = ptrtoint ptr addrspace(3) %1395 to i32, !dbg !18
  %dma.buffer.lds1472 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1405), !dbg !18
  %1406 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1472, i32 %dma.buffer.offset1470, <4 x i32> %dma.buffer.rsrc1471) #10, !dbg !18
  %address.local840 = or disjoint i64 %address.local794, 256, !dbg !11
  %address.local841 = shl nuw nsw i64 %address.local840, 3, !dbg !11
  %address.local842 = and i64 %address.local841, 3584, !dbg !11
  %address.local843 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local842, !dbg !11
  %address.local844 = ptrtoint ptr addrspace(3) %address.local843 to i64, !dbg !11
  %address.local845 = trunc i64 %address.local844 to i32, !dbg !11
  %1407 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local845), !dbg !18
  %1408 = zext i32 %1407 to i64, !dbg !18
  %1409 = inttoptr i64 %1408 to ptr addrspace(3), !dbg !18
  %address.local846 = shl nuw nsw i64 %address.local840, 5, !dbg !18
  %address.local847 = and i64 %address.local846, 16256, !dbg !18
  %address.local848 = shl nuw nsw i64 %address.local794, 3, !dbg !18
  %address.local849 = and i64 %address.local848, 24, !dbg !18
  %address.local850 = or disjoint i64 %address.local847, %address.local849, !dbg !18
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %1392, i64 %address.local850, !dbg !18
  %1410 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1411 = ptrtoint ptr addrspace(1) %address.local851 to i64, !dbg !18
  %1412 = sub i64 %1411, %1410, !dbg !18
  %dma.buffer.offset1467 = trunc i64 %1412 to i32, !dbg !18
  %1413 = trunc i64 %1410 to i32, !dbg !18
  %1414 = insertelement <4 x i32> poison, i32 %1413, i64 0, !dbg !18
  %1415 = lshr i64 %1410, 32, !dbg !18
  %1416 = trunc i64 %1415 to i32, !dbg !18
  %1417 = insertelement <4 x i32> %1414, i32 %1416, i64 1, !dbg !18
  %1418 = insertelement <4 x i32> %1417, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1468 = insertelement <4 x i32> %1418, i32 131072, i64 3, !dbg !18
  %1419 = ptrtoint ptr addrspace(3) %1409 to i32, !dbg !18
  %dma.buffer.lds1469 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1419), !dbg !18
  %1420 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1469, i32 %dma.buffer.offset1467, <4 x i32> %dma.buffer.rsrc1468) #10, !dbg !18
  %1421 = shufflevector <8 x half> %1372, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1422 = extractvalue [2 x <8 x float>] %62, 0, !dbg !35
  %1423 = shufflevector <8 x float> %1422, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1424 = fmul <4 x float> %1423, %1067, !dbg !35
  %1425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1421, <4 x float> %1424) #9, !dbg !14
  %1426 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1426, <4 x float> %1425) #9, !dbg !14
  %1428 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1428, <4 x float> %1427) #9, !dbg !14
  %1430 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1430, <4 x float> %1429) #9, !dbg !14
  %1432 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1432, <4 x float> %1431) #9, !dbg !14
  %1434 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1434, <4 x float> %1433) #9, !dbg !14
  %1436 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1436, <4 x float> %1435) #9, !dbg !14
  %1438 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1438, <4 x float> %1437) #9, !dbg !14
  %1440 = shufflevector <8 x half> %1372, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1441 = shufflevector <8 x float> %1422, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1442 = fmul <4 x float> %1441, %1067, !dbg !35
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1440, <4 x float> %1442) #9, !dbg !14
  %1444 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1444, <4 x float> %1443) #9, !dbg !14
  %1446 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1446, <4 x float> %1445) #9, !dbg !14
  %1448 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1448, <4 x float> %1447) #9, !dbg !14
  %1450 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1450, <4 x float> %1449) #9, !dbg !14
  %1452 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1452, <4 x float> %1451) #9, !dbg !14
  %1454 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1454, <4 x float> %1453) #9, !dbg !14
  %1456 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1456, <4 x float> %1455) #9, !dbg !14
  %1458 = extractvalue [2 x <8 x float>] %62, 1, !dbg !35
  %1459 = shufflevector <8 x float> %1458, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1460 = fmul <4 x float> %1459, %1075, !dbg !35
  %1461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1421, <4 x float> %1460) #9, !dbg !14
  %1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1426, <4 x float> %1461) #9, !dbg !14
  %1463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1428, <4 x float> %1462) #9, !dbg !14
  %1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1430, <4 x float> %1463) #9, !dbg !14
  %1465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1432, <4 x float> %1464) #9, !dbg !14
  %1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1434, <4 x float> %1465) #9, !dbg !14
  %1467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1436, <4 x float> %1466) #9, !dbg !14
  %1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1438, <4 x float> %1467) #9, !dbg !14
  %1469 = shufflevector <8 x float> %1458, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1470 = fmul <4 x float> %1469, %1075, !dbg !35
  %1471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1440, <4 x float> %1470) #9, !dbg !14
  %1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1444, <4 x float> %1471) #9, !dbg !14
  %1473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1446, <4 x float> %1472) #9, !dbg !14
  %1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1448, <4 x float> %1473) #9, !dbg !14
  %1475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1450, <4 x float> %1474) #9, !dbg !14
  %1476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1452, <4 x float> %1475) #9, !dbg !14
  %1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1454, <4 x float> %1476) #9, !dbg !14
  %1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1456, <4 x float> %1477) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1479 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local793, i32 0) #0, !dbg !14
  %1480 = bitcast <4 x i32> %1479 to <8 x half>, !dbg !14
  %1481 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local806, i32 0) #0, !dbg !14
  %1482 = bitcast <4 x i32> %1481 to <8 x half>, !dbg !14
  %1483 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local810, i32 0) #0, !dbg !14
  %1484 = bitcast <4 x i32> %1483 to <8 x half>, !dbg !14
  %1485 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local814, i32 0) #0, !dbg !14
  %1486 = bitcast <4 x i32> %1485 to <8 x half>, !dbg !14
  %1487 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !14
  %1488 = bitcast <4 x i32> %1487 to <8 x half>, !dbg !14
  %1489 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !14
  %1490 = bitcast <4 x i32> %1489 to <8 x half>, !dbg !14
  %1491 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local824, i32 0) #0, !dbg !14
  %1492 = bitcast <4 x i32> %1491 to <8 x half>, !dbg !14
  %1493 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local828, i32 0) #0, !dbg !14
  %1494 = bitcast <4 x i32> %1493 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1495 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1390, !dbg !18
  %1496 = getelementptr i8, ptr addrspace(1) %1495, i64 128, !dbg !18
  %1497 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local833), !dbg !18
  %1498 = zext i32 %1497 to i64, !dbg !18
  %1499 = inttoptr i64 %1498 to ptr addrspace(3), !dbg !18
  %address.local852 = getelementptr [2 x i8], ptr addrspace(1) %1496, i64 %address.local838, !dbg !18
  %1500 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1501 = ptrtoint ptr addrspace(1) %address.local852 to i64, !dbg !18
  %1502 = sub i64 %1501, %1500, !dbg !18
  %dma.buffer.offset1464 = trunc i64 %1502 to i32, !dbg !18
  %1503 = trunc i64 %1500 to i32, !dbg !18
  %1504 = insertelement <4 x i32> poison, i32 %1503, i64 0, !dbg !18
  %1505 = lshr i64 %1500, 32, !dbg !18
  %1506 = trunc i64 %1505 to i32, !dbg !18
  %1507 = insertelement <4 x i32> %1504, i32 %1506, i64 1, !dbg !18
  %1508 = insertelement <4 x i32> %1507, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1465 = insertelement <4 x i32> %1508, i32 131072, i64 3, !dbg !18
  %1509 = ptrtoint ptr addrspace(3) %1499 to i32, !dbg !18
  %dma.buffer.lds1466 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1509), !dbg !18
  %1510 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1466, i32 %dma.buffer.offset1464, <4 x i32> %dma.buffer.rsrc1465) #10, !dbg !18
  %1511 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local845), !dbg !18
  %1512 = zext i32 %1511 to i64, !dbg !18
  %1513 = inttoptr i64 %1512 to ptr addrspace(3), !dbg !18
  %address.local853 = getelementptr [2 x i8], ptr addrspace(1) %1496, i64 %address.local850, !dbg !18
  %1514 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1515 = ptrtoint ptr addrspace(1) %address.local853 to i64, !dbg !18
  %1516 = sub i64 %1515, %1514, !dbg !18
  %dma.buffer.offset1461 = trunc i64 %1516 to i32, !dbg !18
  %1517 = trunc i64 %1514 to i32, !dbg !18
  %1518 = insertelement <4 x i32> poison, i32 %1517, i64 0, !dbg !18
  %1519 = lshr i64 %1514, 32, !dbg !18
  %1520 = trunc i64 %1519 to i32, !dbg !18
  %1521 = insertelement <4 x i32> %1518, i32 %1520, i64 1, !dbg !18
  %1522 = insertelement <4 x i32> %1521, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1462 = insertelement <4 x i32> %1522, i32 131072, i64 3, !dbg !18
  %1523 = ptrtoint ptr addrspace(3) %1513 to i32, !dbg !18
  %dma.buffer.lds1463 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1523), !dbg !18
  %1524 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1463, i32 %dma.buffer.offset1461, <4 x i32> %dma.buffer.rsrc1462) #10, !dbg !18
  %1525 = shufflevector <8 x half> %1480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1526 = extractvalue [2 x <8 x float>] %65, 0, !dbg !35
  %1527 = shufflevector <8 x float> %1526, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1528 = fmul <4 x float> %1527, %1067, !dbg !35
  %1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1525, <4 x float> %1528) #9, !dbg !14
  %1530 = shufflevector <8 x half> %1482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1530, <4 x float> %1529) #9, !dbg !14
  %1532 = shufflevector <8 x half> %1484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1532, <4 x float> %1531) #9, !dbg !14
  %1534 = shufflevector <8 x half> %1486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1534, <4 x float> %1533) #9, !dbg !14
  %1536 = shufflevector <8 x half> %1488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1536, <4 x float> %1535) #9, !dbg !14
  %1538 = shufflevector <8 x half> %1490, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1538, <4 x float> %1537) #9, !dbg !14
  %1540 = shufflevector <8 x half> %1492, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1540, <4 x float> %1539) #9, !dbg !14
  %1542 = shufflevector <8 x half> %1494, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1542, <4 x float> %1541) #9, !dbg !14
  %1544 = shufflevector <8 x half> %1480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1545 = shufflevector <8 x float> %1526, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1546 = fmul <4 x float> %1545, %1067, !dbg !35
  %1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1544, <4 x float> %1546) #9, !dbg !14
  %1548 = shufflevector <8 x half> %1482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1548, <4 x float> %1547) #9, !dbg !14
  %1550 = shufflevector <8 x half> %1484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1550, <4 x float> %1549) #9, !dbg !14
  %1552 = shufflevector <8 x half> %1486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1552, <4 x float> %1551) #9, !dbg !14
  %1554 = shufflevector <8 x half> %1488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1554, <4 x float> %1553) #9, !dbg !14
  %1556 = shufflevector <8 x half> %1490, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1556, <4 x float> %1555) #9, !dbg !14
  %1558 = shufflevector <8 x half> %1492, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1558, <4 x float> %1557) #9, !dbg !14
  %1560 = shufflevector <8 x half> %1494, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1560, <4 x float> %1559) #9, !dbg !14
  %1562 = extractvalue [2 x <8 x float>] %65, 1, !dbg !35
  %1563 = shufflevector <8 x float> %1562, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1564 = fmul <4 x float> %1563, %1075, !dbg !35
  %1565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1525, <4 x float> %1564) #9, !dbg !14
  %1566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1530, <4 x float> %1565) #9, !dbg !14
  %1567 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1532, <4 x float> %1566) #9, !dbg !14
  %1568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1534, <4 x float> %1567) #9, !dbg !14
  %1569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1536, <4 x float> %1568) #9, !dbg !14
  %1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1538, <4 x float> %1569) #9, !dbg !14
  %1571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1540, <4 x float> %1570) #9, !dbg !14
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1542, <4 x float> %1571) #9, !dbg !14
  %1573 = shufflevector <8 x float> %1562, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1574 = fmul <4 x float> %1573, %1075, !dbg !35
  %1575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1544, <4 x float> %1574) #9, !dbg !14
  %1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1548, <4 x float> %1575) #9, !dbg !14
  %1577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1550, <4 x float> %1576) #9, !dbg !14
  %1578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1552, <4 x float> %1577) #9, !dbg !14
  %1579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1554, <4 x float> %1578) #9, !dbg !14
  %1580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1556, <4 x float> %1579) #9, !dbg !14
  %1581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1558, <4 x float> %1580) #9, !dbg !14
  %1582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1560, <4 x float> %1581) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1583 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local793, i32 0) #0, !dbg !14
  %1584 = bitcast <4 x i32> %1583 to <8 x half>, !dbg !14
  %1585 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local806, i32 0) #0, !dbg !14
  %1586 = bitcast <4 x i32> %1585 to <8 x half>, !dbg !14
  %1587 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local810, i32 0) #0, !dbg !14
  %1588 = bitcast <4 x i32> %1587 to <8 x half>, !dbg !14
  %1589 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local814, i32 0) #0, !dbg !14
  %1590 = bitcast <4 x i32> %1589 to <8 x half>, !dbg !14
  %1591 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !14
  %1592 = bitcast <4 x i32> %1591 to <8 x half>, !dbg !14
  %1593 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !14
  %1594 = bitcast <4 x i32> %1593 to <8 x half>, !dbg !14
  %1595 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local824, i32 0) #0, !dbg !14
  %1596 = bitcast <4 x i32> %1595 to <8 x half>, !dbg !14
  %1597 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local828, i32 0) #0, !dbg !14
  %1598 = bitcast <4 x i32> %1597 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1599 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1390, !dbg !18
  %1600 = getelementptr i8, ptr addrspace(1) %1599, i64 192, !dbg !18
  %1601 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local833), !dbg !18
  %1602 = zext i32 %1601 to i64, !dbg !18
  %1603 = inttoptr i64 %1602 to ptr addrspace(3), !dbg !18
  %address.local854 = getelementptr [2 x i8], ptr addrspace(1) %1600, i64 %address.local838, !dbg !18
  %1604 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1605 = ptrtoint ptr addrspace(1) %address.local854 to i64, !dbg !18
  %1606 = sub i64 %1605, %1604, !dbg !18
  %dma.buffer.offset1458 = trunc i64 %1606 to i32, !dbg !18
  %1607 = trunc i64 %1604 to i32, !dbg !18
  %1608 = insertelement <4 x i32> poison, i32 %1607, i64 0, !dbg !18
  %1609 = lshr i64 %1604, 32, !dbg !18
  %1610 = trunc i64 %1609 to i32, !dbg !18
  %1611 = insertelement <4 x i32> %1608, i32 %1610, i64 1, !dbg !18
  %1612 = insertelement <4 x i32> %1611, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1459 = insertelement <4 x i32> %1612, i32 131072, i64 3, !dbg !18
  %1613 = ptrtoint ptr addrspace(3) %1603 to i32, !dbg !18
  %dma.buffer.lds1460 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1613), !dbg !18
  %1614 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1460, i32 %dma.buffer.offset1458, <4 x i32> %dma.buffer.rsrc1459) #10, !dbg !18
  %1615 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local845), !dbg !18
  %1616 = zext i32 %1615 to i64, !dbg !18
  %1617 = inttoptr i64 %1616 to ptr addrspace(3), !dbg !18
  %address.local855 = getelementptr [2 x i8], ptr addrspace(1) %1600, i64 %address.local850, !dbg !18
  %1618 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1619 = ptrtoint ptr addrspace(1) %address.local855 to i64, !dbg !18
  %1620 = sub i64 %1619, %1618, !dbg !18
  %dma.buffer.offset = trunc i64 %1620 to i32, !dbg !18
  %1621 = trunc i64 %1618 to i32, !dbg !18
  %1622 = insertelement <4 x i32> poison, i32 %1621, i64 0, !dbg !18
  %1623 = lshr i64 %1618, 32, !dbg !18
  %1624 = trunc i64 %1623 to i32, !dbg !18
  %1625 = insertelement <4 x i32> %1622, i32 %1624, i64 1, !dbg !18
  %1626 = insertelement <4 x i32> %1625, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1626, i32 131072, i64 3, !dbg !18
  %1627 = ptrtoint ptr addrspace(3) %1617 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1627), !dbg !18
  %1628 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1629 = shufflevector <8 x half> %1584, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1630 = extractvalue [2 x <8 x float>] %66, 0, !dbg !35
  %1631 = shufflevector <8 x float> %1630, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1632 = fmul <4 x float> %1631, %1067, !dbg !35
  %1633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1629, <4 x float> %1632) #9, !dbg !14
  %1634 = shufflevector <8 x half> %1586, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1634, <4 x float> %1633) #9, !dbg !14
  %1636 = shufflevector <8 x half> %1588, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1636, <4 x float> %1635) #9, !dbg !14
  %1638 = shufflevector <8 x half> %1590, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1638, <4 x float> %1637) #9, !dbg !14
  %1640 = shufflevector <8 x half> %1592, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1640, <4 x float> %1639) #9, !dbg !14
  %1642 = shufflevector <8 x half> %1594, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1642, <4 x float> %1641) #9, !dbg !14
  %1644 = shufflevector <8 x half> %1596, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1644, <4 x float> %1643) #9, !dbg !14
  %1646 = shufflevector <8 x half> %1598, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1646, <4 x float> %1645) #9, !dbg !14
  %1648 = shufflevector <8 x half> %1584, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1649 = shufflevector <8 x float> %1630, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1650 = fmul <4 x float> %1649, %1067, !dbg !35
  %1651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1648, <4 x float> %1650) #9, !dbg !14
  %1652 = shufflevector <8 x half> %1586, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1652, <4 x float> %1651) #9, !dbg !14
  %1654 = shufflevector <8 x half> %1588, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1654, <4 x float> %1653) #9, !dbg !14
  %1656 = shufflevector <8 x half> %1590, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1656, <4 x float> %1655) #9, !dbg !14
  %1658 = shufflevector <8 x half> %1592, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1658, <4 x float> %1657) #9, !dbg !14
  %1660 = shufflevector <8 x half> %1594, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1660, <4 x float> %1659) #9, !dbg !14
  %1662 = shufflevector <8 x half> %1596, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1662, <4 x float> %1661) #9, !dbg !14
  %1664 = shufflevector <8 x half> %1598, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1664, <4 x float> %1663) #9, !dbg !14
  %1666 = extractvalue [2 x <8 x float>] %66, 1, !dbg !35
  %1667 = shufflevector <8 x float> %1666, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1668 = fmul <4 x float> %1667, %1075, !dbg !35
  %1669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1629, <4 x float> %1668) #9, !dbg !14
  %1670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1634, <4 x float> %1669) #9, !dbg !14
  %1671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1636, <4 x float> %1670) #9, !dbg !14
  %1672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1638, <4 x float> %1671) #9, !dbg !14
  %1673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1640, <4 x float> %1672) #9, !dbg !14
  %1674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1642, <4 x float> %1673) #9, !dbg !14
  %1675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1644, <4 x float> %1674) #9, !dbg !14
  %1676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1646, <4 x float> %1675) #9, !dbg !14
  %1677 = shufflevector <8 x float> %1666, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1678 = fmul <4 x float> %1677, %1075, !dbg !35
  %1679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1648, <4 x float> %1678) #9, !dbg !14
  %1680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1652, <4 x float> %1679) #9, !dbg !14
  %1681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1654, <4 x float> %1680) #9, !dbg !14
  %1682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1656, <4 x float> %1681) #9, !dbg !14
  %1683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1658, <4 x float> %1682) #9, !dbg !14
  %1684 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1660, <4 x float> %1683) #9, !dbg !14
  %1685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1662, <4 x float> %1684) #9, !dbg !14
  %1686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1664, <4 x float> %1685) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1687 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local793, i32 0) #0, !dbg !14
  %1688 = bitcast <4 x i32> %1687 to <8 x half>, !dbg !14
  %1689 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local806, i32 0) #0, !dbg !14
  %1690 = bitcast <4 x i32> %1689 to <8 x half>, !dbg !14
  %1691 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local810, i32 0) #0, !dbg !14
  %1692 = bitcast <4 x i32> %1691 to <8 x half>, !dbg !14
  %1693 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local814, i32 0) #0, !dbg !14
  %1694 = bitcast <4 x i32> %1693 to <8 x half>, !dbg !14
  %1695 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !14
  %1696 = bitcast <4 x i32> %1695 to <8 x half>, !dbg !14
  %1697 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !14
  %1698 = bitcast <4 x i32> %1697 to <8 x half>, !dbg !14
  %1699 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local824, i32 0) #0, !dbg !14
  %1700 = bitcast <4 x i32> %1699 to <8 x half>, !dbg !14
  %1701 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local828, i32 0) #0, !dbg !14
  %1702 = bitcast <4 x i32> %1701 to <8 x half>, !dbg !14
  %1703 = shufflevector <8 x half> %1688, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1704 = extractvalue [2 x <8 x float>] %67, 0, !dbg !35
  %1705 = shufflevector <8 x float> %1704, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1706 = fmul <4 x float> %1705, %1067, !dbg !35
  %1707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1703, <4 x float> %1706) #9, !dbg !14
  %1708 = shufflevector <8 x half> %1690, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1708, <4 x float> %1707) #9, !dbg !14
  %1710 = shufflevector <8 x half> %1692, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1710, <4 x float> %1709) #9, !dbg !14
  %1712 = shufflevector <8 x half> %1694, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1712, <4 x float> %1711) #9, !dbg !14
  %1714 = shufflevector <8 x half> %1696, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1715 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1714, <4 x float> %1713) #9, !dbg !14
  %1716 = shufflevector <8 x half> %1698, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1716, <4 x float> %1715) #9, !dbg !14
  %1718 = shufflevector <8 x half> %1700, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1718, <4 x float> %1717) #9, !dbg !14
  %1720 = shufflevector <8 x half> %1702, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1720, <4 x float> %1719) #9, !dbg !14
  %1722 = shufflevector <8 x half> %1688, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1723 = shufflevector <8 x float> %1704, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1724 = fmul <4 x float> %1723, %1067, !dbg !35
  %1725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1722, <4 x float> %1724) #9, !dbg !14
  %1726 = shufflevector <8 x half> %1690, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1726, <4 x float> %1725) #9, !dbg !14
  %1728 = shufflevector <8 x half> %1692, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1728, <4 x float> %1727) #9, !dbg !14
  %1730 = shufflevector <8 x half> %1694, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1730, <4 x float> %1729) #9, !dbg !14
  %1732 = shufflevector <8 x half> %1696, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1733 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %1732, <4 x float> %1731) #9, !dbg !14
  %1734 = shufflevector <8 x half> %1698, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1735 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %1734, <4 x float> %1733) #9, !dbg !14
  %1736 = shufflevector <8 x half> %1700, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed317, <4 x half> %1736, <4 x float> %1735) #9, !dbg !14
  %1738 = shufflevector <8 x half> %1702, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed319, <4 x half> %1738, <4 x float> %1737) #9, !dbg !14
  %1740 = extractvalue [2 x <8 x float>] %67, 1, !dbg !35
  %1741 = shufflevector <8 x float> %1740, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1742 = fmul <4 x float> %1741, %1075, !dbg !35
  %1743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1703, <4 x float> %1742) #9, !dbg !14
  %1744 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1708, <4 x float> %1743) #9, !dbg !14
  %1745 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1710, <4 x float> %1744) #9, !dbg !14
  %1746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1712, <4 x float> %1745) #9, !dbg !14
  %1747 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1714, <4 x float> %1746) #9, !dbg !14
  %1748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1716, <4 x float> %1747) #9, !dbg !14
  %1749 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1718, <4 x float> %1748) #9, !dbg !14
  %1750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1720, <4 x float> %1749) #9, !dbg !14
  %1751 = shufflevector <8 x float> %1740, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1752 = fmul <4 x float> %1751, %1075, !dbg !35
  %1753 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed321, <4 x half> %1722, <4 x float> %1752) #9, !dbg !14
  %1754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed323, <4 x half> %1726, <4 x float> %1753) #9, !dbg !14
  %1755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed325, <4 x half> %1728, <4 x float> %1754) #9, !dbg !14
  %1756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed327, <4 x half> %1730, <4 x float> %1755) #9, !dbg !14
  %1757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed329, <4 x half> %1732, <4 x float> %1756) #9, !dbg !14
  %1758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed331, <4 x half> %1734, <4 x float> %1757) #9, !dbg !14
  %1759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed333, <4 x half> %1736, <4 x float> %1758) #9, !dbg !14
  %1760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed335, <4 x half> %1738, <4 x float> %1759) #9, !dbg !14
  %1761 = shufflevector <4 x float> %1439, <4 x float> %1457, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1762 = insertvalue [2 x <8 x float>] poison, <8 x float> %1761, 0, !dbg !7
  %1763 = shufflevector <4 x float> %1468, <4 x float> %1478, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1764 = insertvalue [2 x <8 x float>] %1762, <8 x float> %1763, 1, !dbg !7
  %1765 = insertvalue [2 x <1 x float>] poison, <1 x float> %1001, 0, !dbg !7
  %1766 = insertvalue [2 x <1 x float>] %1765, <1 x float> %1044, 1, !dbg !7
  %1767 = shufflevector <4 x float> %1543, <4 x float> %1561, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1768 = insertvalue [2 x <8 x float>] poison, <8 x float> %1767, 0, !dbg !7
  %1769 = shufflevector <4 x float> %1572, <4 x float> %1582, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1770 = insertvalue [2 x <8 x float>] %1768, <8 x float> %1769, 1, !dbg !7
  %1771 = shufflevector <4 x float> %1647, <4 x float> %1665, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1772 = insertvalue [2 x <8 x float>] poison, <8 x float> %1771, 0, !dbg !7
  %1773 = shufflevector <4 x float> %1676, <4 x float> %1686, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1774 = insertvalue [2 x <8 x float>] %1772, <8 x float> %1773, 1, !dbg !7
  %1775 = shufflevector <4 x float> %1721, <4 x float> %1739, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1776 = insertvalue [2 x <8 x float>] poison, <8 x float> %1775, 0, !dbg !7
  %1777 = shufflevector <4 x float> %1750, <4 x float> %1760, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1778 = insertvalue [2 x <8 x float>] %1776, <8 x float> %1777, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1779 = add nuw nsw i64 %61, 1, !dbg !7
  br label %60, !dbg !7

1780:                                             ; preds = %60
  %1781 = and i32 %34, 16, !dbg !7
  %.not = icmp eq i32 %1781, 0, !dbg !7
  %1782 = and i32 %34, 32, !dbg !7
  %.not36 = icmp eq i32 %1782, 0, !dbg !7
  %1783 = extractvalue [2 x <8 x float>] %67, 0, !dbg !7
  %1784 = extractelement <8 x float> %1783, i64 0, !dbg !7
  %1785 = extractelement <8 x float> %1783, i64 1, !dbg !7
  %1786 = extractelement <8 x float> %1783, i64 2, !dbg !7
  %1787 = extractelement <8 x float> %1783, i64 3, !dbg !7
  %1788 = select i1 %.not, float %1785, float %1784, !dbg !7
  %1789 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1790 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1789), !dbg !7
  %1791 = add nuw nsw i32 %1790, 64, !dbg !7
  %1792 = xor i32 %1790, 16, !dbg !7
  %1793 = and i32 %1791, 192, !dbg !7
  %1794 = icmp samesign ult i32 %1792, %1793, !dbg !7
  %1795 = select i1 %1794, i32 %1792, i32 %1790, !dbg !7
  %1796 = shl nuw nsw i32 %1795, 2, !dbg !7
  %1797 = bitcast float %1788 to i32, !dbg !7
  %1798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1797), !dbg !7
  %1799 = bitcast i32 %1798 to float, !dbg !7
  %1800 = select i1 %.not, float %1784, float %1799, !dbg !7
  %1801 = select i1 %.not, float %1799, float %1785, !dbg !7
  %1802 = select i1 %.not, float %1787, float %1786, !dbg !7
  %1803 = bitcast float %1802 to i32, !dbg !7
  %1804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1803), !dbg !7
  %1805 = bitcast i32 %1804 to float, !dbg !7
  %1806 = select i1 %.not, float %1786, float %1805, !dbg !7
  %1807 = select i1 %.not, float %1805, float %1787, !dbg !7
  %1808 = select i1 %.not36, float %1806, float %1800, !dbg !7
  %1809 = xor i32 %1790, 32, !dbg !7
  %1810 = icmp samesign ult i32 %1809, %1793, !dbg !7
  %1811 = select i1 %1810, i32 %1809, i32 %1790, !dbg !7
  %1812 = shl nuw nsw i32 %1811, 2, !dbg !7
  %1813 = bitcast float %1808 to i32, !dbg !7
  %1814 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1813), !dbg !7
  %1815 = bitcast i32 %1814 to float, !dbg !7
  %1816 = select i1 %.not36, float %1800, float %1815, !dbg !7
  %1817 = select i1 %.not36, float %1815, float %1806, !dbg !7
  %1818 = select i1 %.not36, float %1807, float %1801, !dbg !7
  %1819 = bitcast float %1818 to i32, !dbg !7
  %1820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1819), !dbg !7
  %1821 = bitcast i32 %1820 to float, !dbg !7
  %1822 = select i1 %.not36, float %1801, float %1821, !dbg !7
  %1823 = select i1 %.not36, float %1821, float %1807, !dbg !7
  %1824 = extractvalue [2 x <8 x float>] %66, 0, !dbg !7
  %1825 = extractelement <8 x float> %1824, i64 0, !dbg !7
  %1826 = extractelement <8 x float> %1824, i64 1, !dbg !7
  %1827 = extractelement <8 x float> %1824, i64 2, !dbg !7
  %1828 = extractelement <8 x float> %1824, i64 3, !dbg !7
  %1829 = select i1 %.not, float %1826, float %1825, !dbg !7
  %1830 = bitcast float %1829 to i32, !dbg !7
  %1831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1830), !dbg !7
  %1832 = bitcast i32 %1831 to float, !dbg !7
  %1833 = select i1 %.not, float %1825, float %1832, !dbg !7
  %1834 = select i1 %.not, float %1832, float %1826, !dbg !7
  %1835 = select i1 %.not, float %1828, float %1827, !dbg !7
  %1836 = bitcast float %1835 to i32, !dbg !7
  %1837 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1836), !dbg !7
  %1838 = bitcast i32 %1837 to float, !dbg !7
  %1839 = select i1 %.not, float %1827, float %1838, !dbg !7
  %1840 = select i1 %.not, float %1838, float %1828, !dbg !7
  %1841 = select i1 %.not36, float %1839, float %1833, !dbg !7
  %1842 = bitcast float %1841 to i32, !dbg !7
  %1843 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1842), !dbg !7
  %1844 = bitcast i32 %1843 to float, !dbg !7
  %1845 = select i1 %.not36, float %1833, float %1844, !dbg !7
  %1846 = select i1 %.not36, float %1844, float %1839, !dbg !7
  %1847 = select i1 %.not36, float %1840, float %1834, !dbg !7
  %1848 = bitcast float %1847 to i32, !dbg !7
  %1849 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1848), !dbg !7
  %1850 = bitcast i32 %1849 to float, !dbg !7
  %1851 = select i1 %.not36, float %1834, float %1850, !dbg !7
  %1852 = select i1 %.not36, float %1850, float %1840, !dbg !7
  %1853 = extractvalue [2 x <8 x float>] %65, 0, !dbg !7
  %1854 = extractelement <8 x float> %1853, i64 0, !dbg !7
  %1855 = extractelement <8 x float> %1853, i64 1, !dbg !7
  %1856 = extractelement <8 x float> %1853, i64 2, !dbg !7
  %1857 = extractelement <8 x float> %1853, i64 3, !dbg !7
  %1858 = select i1 %.not, float %1855, float %1854, !dbg !7
  %1859 = bitcast float %1858 to i32, !dbg !7
  %1860 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1859), !dbg !7
  %1861 = bitcast i32 %1860 to float, !dbg !7
  %1862 = select i1 %.not, float %1854, float %1861, !dbg !7
  %1863 = select i1 %.not, float %1861, float %1855, !dbg !7
  %1864 = select i1 %.not, float %1857, float %1856, !dbg !7
  %1865 = bitcast float %1864 to i32, !dbg !7
  %1866 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1865), !dbg !7
  %1867 = bitcast i32 %1866 to float, !dbg !7
  %1868 = select i1 %.not, float %1856, float %1867, !dbg !7
  %1869 = select i1 %.not, float %1867, float %1857, !dbg !7
  %1870 = select i1 %.not36, float %1868, float %1862, !dbg !7
  %1871 = bitcast float %1870 to i32, !dbg !7
  %1872 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1871), !dbg !7
  %1873 = bitcast i32 %1872 to float, !dbg !7
  %1874 = select i1 %.not36, float %1862, float %1873, !dbg !7
  %1875 = select i1 %.not36, float %1873, float %1868, !dbg !7
  %1876 = select i1 %.not36, float %1869, float %1863, !dbg !7
  %1877 = bitcast float %1876 to i32, !dbg !7
  %1878 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1877), !dbg !7
  %1879 = bitcast i32 %1878 to float, !dbg !7
  %1880 = select i1 %.not36, float %1863, float %1879, !dbg !7
  %1881 = select i1 %.not36, float %1879, float %1869, !dbg !7
  %1882 = extractvalue [2 x <8 x float>] %62, 0, !dbg !7
  %1883 = extractelement <8 x float> %1882, i64 0, !dbg !7
  %1884 = extractelement <8 x float> %1882, i64 1, !dbg !7
  %1885 = extractelement <8 x float> %1882, i64 2, !dbg !7
  %1886 = extractelement <8 x float> %1882, i64 3, !dbg !7
  %1887 = select i1 %.not, float %1884, float %1883, !dbg !7
  %1888 = bitcast float %1887 to i32, !dbg !7
  %1889 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1888), !dbg !7
  %1890 = bitcast i32 %1889 to float, !dbg !7
  %1891 = select i1 %.not, float %1883, float %1890, !dbg !7
  %1892 = select i1 %.not, float %1890, float %1884, !dbg !7
  %1893 = select i1 %.not, float %1886, float %1885, !dbg !7
  %1894 = bitcast float %1893 to i32, !dbg !7
  %1895 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1894), !dbg !7
  %1896 = bitcast i32 %1895 to float, !dbg !7
  %1897 = select i1 %.not, float %1885, float %1896, !dbg !7
  %1898 = select i1 %.not, float %1896, float %1886, !dbg !7
  %1899 = select i1 %.not36, float %1897, float %1891, !dbg !7
  %1900 = bitcast float %1899 to i32, !dbg !7
  %1901 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1900), !dbg !7
  %1902 = select i1 %.not36, float %1898, float %1892, !dbg !7
  %1903 = bitcast float %1902 to i32, !dbg !7
  %1904 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1903), !dbg !7
  %1905 = extractvalue [2 x <1 x float>] %63, 0, !dbg !19
  %1906 = shufflevector <1 x float> %1905, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1907 = extractelement <8 x float> %1783, i64 4, !dbg !7
  %1908 = extractelement <8 x float> %1783, i64 5, !dbg !7
  %1909 = extractelement <8 x float> %1783, i64 6, !dbg !7
  %1910 = extractelement <8 x float> %1783, i64 7, !dbg !7
  %1911 = select i1 %.not, float %1908, float %1907, !dbg !7
  %1912 = bitcast float %1911 to i32, !dbg !7
  %1913 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1912), !dbg !7
  %1914 = bitcast i32 %1913 to float, !dbg !7
  %1915 = select i1 %.not, float %1907, float %1914, !dbg !7
  %1916 = select i1 %.not, float %1914, float %1908, !dbg !7
  %1917 = select i1 %.not, float %1910, float %1909, !dbg !7
  %1918 = bitcast float %1917 to i32, !dbg !7
  %1919 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1918), !dbg !7
  %1920 = bitcast i32 %1919 to float, !dbg !7
  %1921 = select i1 %.not, float %1909, float %1920, !dbg !7
  %1922 = select i1 %.not, float %1920, float %1910, !dbg !7
  %1923 = select i1 %.not36, float %1921, float %1915, !dbg !7
  %1924 = bitcast float %1923 to i32, !dbg !7
  %1925 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1924), !dbg !7
  %1926 = bitcast i32 %1925 to float, !dbg !7
  %1927 = select i1 %.not36, float %1915, float %1926, !dbg !7
  %1928 = select i1 %.not36, float %1926, float %1921, !dbg !7
  %1929 = select i1 %.not36, float %1922, float %1916, !dbg !7
  %1930 = bitcast float %1929 to i32, !dbg !7
  %1931 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1930), !dbg !7
  %1932 = bitcast i32 %1931 to float, !dbg !7
  %1933 = select i1 %.not36, float %1916, float %1932, !dbg !7
  %1934 = select i1 %.not36, float %1932, float %1922, !dbg !7
  %1935 = extractelement <8 x float> %1824, i64 4, !dbg !7
  %1936 = extractelement <8 x float> %1824, i64 5, !dbg !7
  %1937 = extractelement <8 x float> %1824, i64 6, !dbg !7
  %1938 = extractelement <8 x float> %1824, i64 7, !dbg !7
  %1939 = select i1 %.not, float %1936, float %1935, !dbg !7
  %1940 = bitcast float %1939 to i32, !dbg !7
  %1941 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1940), !dbg !7
  %1942 = bitcast i32 %1941 to float, !dbg !7
  %1943 = select i1 %.not, float %1935, float %1942, !dbg !7
  %1944 = select i1 %.not, float %1942, float %1936, !dbg !7
  %1945 = select i1 %.not, float %1938, float %1937, !dbg !7
  %1946 = bitcast float %1945 to i32, !dbg !7
  %1947 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1946), !dbg !7
  %1948 = bitcast i32 %1947 to float, !dbg !7
  %1949 = select i1 %.not, float %1937, float %1948, !dbg !7
  %1950 = select i1 %.not, float %1948, float %1938, !dbg !7
  %1951 = select i1 %.not36, float %1949, float %1943, !dbg !7
  %1952 = bitcast float %1951 to i32, !dbg !7
  %1953 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1952), !dbg !7
  %1954 = bitcast i32 %1953 to float, !dbg !7
  %1955 = select i1 %.not36, float %1943, float %1954, !dbg !7
  %1956 = select i1 %.not36, float %1954, float %1949, !dbg !7
  %1957 = select i1 %.not36, float %1950, float %1944, !dbg !7
  %1958 = bitcast float %1957 to i32, !dbg !7
  %1959 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1958), !dbg !7
  %1960 = bitcast i32 %1959 to float, !dbg !7
  %1961 = select i1 %.not36, float %1944, float %1960, !dbg !7
  %1962 = select i1 %.not36, float %1960, float %1950, !dbg !7
  %1963 = extractelement <8 x float> %1853, i64 4, !dbg !7
  %1964 = extractelement <8 x float> %1853, i64 5, !dbg !7
  %1965 = extractelement <8 x float> %1853, i64 6, !dbg !7
  %1966 = extractelement <8 x float> %1853, i64 7, !dbg !7
  %1967 = select i1 %.not, float %1964, float %1963, !dbg !7
  %1968 = bitcast float %1967 to i32, !dbg !7
  %1969 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1968), !dbg !7
  %1970 = bitcast i32 %1969 to float, !dbg !7
  %1971 = select i1 %.not, float %1963, float %1970, !dbg !7
  %1972 = select i1 %.not, float %1970, float %1964, !dbg !7
  %1973 = select i1 %.not, float %1966, float %1965, !dbg !7
  %1974 = bitcast float %1973 to i32, !dbg !7
  %1975 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1974), !dbg !7
  %1976 = bitcast i32 %1975 to float, !dbg !7
  %1977 = select i1 %.not, float %1965, float %1976, !dbg !7
  %1978 = select i1 %.not, float %1976, float %1966, !dbg !7
  %1979 = select i1 %.not36, float %1977, float %1971, !dbg !7
  %1980 = bitcast float %1979 to i32, !dbg !7
  %1981 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1980), !dbg !7
  %1982 = bitcast i32 %1981 to float, !dbg !7
  %1983 = select i1 %.not36, float %1971, float %1982, !dbg !7
  %1984 = select i1 %.not36, float %1982, float %1977, !dbg !7
  %1985 = select i1 %.not36, float %1978, float %1972, !dbg !7
  %1986 = bitcast float %1985 to i32, !dbg !7
  %1987 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %1986), !dbg !7
  %1988 = bitcast i32 %1987 to float, !dbg !7
  %1989 = select i1 %.not36, float %1972, float %1988, !dbg !7
  %1990 = select i1 %.not36, float %1988, float %1978, !dbg !7
  %1991 = extractelement <8 x float> %1882, i64 4, !dbg !7
  %1992 = extractelement <8 x float> %1882, i64 5, !dbg !7
  %1993 = extractelement <8 x float> %1882, i64 6, !dbg !7
  %1994 = extractelement <8 x float> %1882, i64 7, !dbg !7
  %1995 = select i1 %.not, float %1992, float %1991, !dbg !7
  %1996 = bitcast float %1995 to i32, !dbg !7
  %1997 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %1996), !dbg !7
  %1998 = bitcast i32 %1997 to float, !dbg !7
  %1999 = select i1 %.not, float %1991, float %1998, !dbg !7
  %2000 = select i1 %.not, float %1998, float %1992, !dbg !7
  %2001 = select i1 %.not, float %1994, float %1993, !dbg !7
  %2002 = bitcast float %2001 to i32, !dbg !7
  %2003 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2002), !dbg !7
  %2004 = bitcast i32 %2003 to float, !dbg !7
  %2005 = select i1 %.not, float %1993, float %2004, !dbg !7
  %2006 = select i1 %.not, float %2004, float %1994, !dbg !7
  %2007 = select i1 %.not36, float %2005, float %1999, !dbg !7
  %2008 = bitcast float %2007 to i32, !dbg !7
  %2009 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2008), !dbg !7
  %2010 = select i1 %.not36, float %2006, float %2000, !dbg !7
  %2011 = bitcast float %2010 to i32, !dbg !7
  %2012 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2011), !dbg !7
  %2013 = extractvalue [2 x <8 x float>] %67, 1, !dbg !7
  %2014 = extractelement <8 x float> %2013, i64 0, !dbg !7
  %2015 = extractelement <8 x float> %2013, i64 1, !dbg !7
  %2016 = extractelement <8 x float> %2013, i64 2, !dbg !7
  %2017 = extractelement <8 x float> %2013, i64 3, !dbg !7
  %2018 = select i1 %.not, float %2015, float %2014, !dbg !7
  %2019 = bitcast float %2018 to i32, !dbg !7
  %2020 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2019), !dbg !7
  %2021 = bitcast i32 %2020 to float, !dbg !7
  %2022 = select i1 %.not, float %2014, float %2021, !dbg !7
  %2023 = select i1 %.not, float %2021, float %2015, !dbg !7
  %2024 = select i1 %.not, float %2017, float %2016, !dbg !7
  %2025 = bitcast float %2024 to i32, !dbg !7
  %2026 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2025), !dbg !7
  %2027 = bitcast i32 %2026 to float, !dbg !7
  %2028 = select i1 %.not, float %2016, float %2027, !dbg !7
  %2029 = select i1 %.not, float %2027, float %2017, !dbg !7
  %2030 = select i1 %.not36, float %2028, float %2022, !dbg !7
  %2031 = bitcast float %2030 to i32, !dbg !7
  %2032 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2031), !dbg !7
  %2033 = bitcast i32 %2032 to float, !dbg !7
  %2034 = select i1 %.not36, float %2022, float %2033, !dbg !7
  %2035 = select i1 %.not36, float %2033, float %2028, !dbg !7
  %2036 = select i1 %.not36, float %2029, float %2023, !dbg !7
  %2037 = bitcast float %2036 to i32, !dbg !7
  %2038 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2037), !dbg !7
  %2039 = bitcast i32 %2038 to float, !dbg !7
  %2040 = select i1 %.not36, float %2023, float %2039, !dbg !7
  %2041 = select i1 %.not36, float %2039, float %2029, !dbg !7
  %2042 = extractvalue [2 x <8 x float>] %66, 1, !dbg !7
  %2043 = extractelement <8 x float> %2042, i64 0, !dbg !7
  %2044 = extractelement <8 x float> %2042, i64 1, !dbg !7
  %2045 = extractelement <8 x float> %2042, i64 2, !dbg !7
  %2046 = extractelement <8 x float> %2042, i64 3, !dbg !7
  %2047 = select i1 %.not, float %2044, float %2043, !dbg !7
  %2048 = bitcast float %2047 to i32, !dbg !7
  %2049 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2048), !dbg !7
  %2050 = bitcast i32 %2049 to float, !dbg !7
  %2051 = select i1 %.not, float %2043, float %2050, !dbg !7
  %2052 = select i1 %.not, float %2050, float %2044, !dbg !7
  %2053 = select i1 %.not, float %2046, float %2045, !dbg !7
  %2054 = bitcast float %2053 to i32, !dbg !7
  %2055 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2054), !dbg !7
  %2056 = bitcast i32 %2055 to float, !dbg !7
  %2057 = select i1 %.not, float %2045, float %2056, !dbg !7
  %2058 = select i1 %.not, float %2056, float %2046, !dbg !7
  %2059 = select i1 %.not36, float %2057, float %2051, !dbg !7
  %2060 = bitcast float %2059 to i32, !dbg !7
  %2061 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2060), !dbg !7
  %2062 = bitcast i32 %2061 to float, !dbg !7
  %2063 = select i1 %.not36, float %2051, float %2062, !dbg !7
  %2064 = select i1 %.not36, float %2062, float %2057, !dbg !7
  %2065 = select i1 %.not36, float %2058, float %2052, !dbg !7
  %2066 = bitcast float %2065 to i32, !dbg !7
  %2067 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2066), !dbg !7
  %2068 = bitcast i32 %2067 to float, !dbg !7
  %2069 = select i1 %.not36, float %2052, float %2068, !dbg !7
  %2070 = select i1 %.not36, float %2068, float %2058, !dbg !7
  %2071 = extractvalue [2 x <8 x float>] %65, 1, !dbg !7
  %2072 = extractelement <8 x float> %2071, i64 0, !dbg !7
  %2073 = extractelement <8 x float> %2071, i64 1, !dbg !7
  %2074 = extractelement <8 x float> %2071, i64 2, !dbg !7
  %2075 = extractelement <8 x float> %2071, i64 3, !dbg !7
  %2076 = select i1 %.not, float %2073, float %2072, !dbg !7
  %2077 = bitcast float %2076 to i32, !dbg !7
  %2078 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2077), !dbg !7
  %2079 = bitcast i32 %2078 to float, !dbg !7
  %2080 = select i1 %.not, float %2072, float %2079, !dbg !7
  %2081 = select i1 %.not, float %2079, float %2073, !dbg !7
  %2082 = select i1 %.not, float %2075, float %2074, !dbg !7
  %2083 = bitcast float %2082 to i32, !dbg !7
  %2084 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2083), !dbg !7
  %2085 = bitcast i32 %2084 to float, !dbg !7
  %2086 = select i1 %.not, float %2074, float %2085, !dbg !7
  %2087 = select i1 %.not, float %2085, float %2075, !dbg !7
  %2088 = select i1 %.not36, float %2086, float %2080, !dbg !7
  %2089 = bitcast float %2088 to i32, !dbg !7
  %2090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2089), !dbg !7
  %2091 = bitcast i32 %2090 to float, !dbg !7
  %2092 = select i1 %.not36, float %2080, float %2091, !dbg !7
  %2093 = select i1 %.not36, float %2091, float %2086, !dbg !7
  %2094 = select i1 %.not36, float %2087, float %2081, !dbg !7
  %2095 = bitcast float %2094 to i32, !dbg !7
  %2096 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2095), !dbg !7
  %2097 = bitcast i32 %2096 to float, !dbg !7
  %2098 = select i1 %.not36, float %2081, float %2097, !dbg !7
  %2099 = select i1 %.not36, float %2097, float %2087, !dbg !7
  %2100 = extractvalue [2 x <8 x float>] %62, 1, !dbg !7
  %2101 = extractelement <8 x float> %2100, i64 0, !dbg !7
  %2102 = extractelement <8 x float> %2100, i64 1, !dbg !7
  %2103 = extractelement <8 x float> %2100, i64 2, !dbg !7
  %2104 = extractelement <8 x float> %2100, i64 3, !dbg !7
  %2105 = select i1 %.not, float %2102, float %2101, !dbg !7
  %2106 = bitcast float %2105 to i32, !dbg !7
  %2107 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2106), !dbg !7
  %2108 = bitcast i32 %2107 to float, !dbg !7
  %2109 = select i1 %.not, float %2101, float %2108, !dbg !7
  %2110 = select i1 %.not, float %2108, float %2102, !dbg !7
  %2111 = select i1 %.not, float %2104, float %2103, !dbg !7
  %2112 = bitcast float %2111 to i32, !dbg !7
  %2113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2112), !dbg !7
  %2114 = bitcast i32 %2113 to float, !dbg !7
  %2115 = select i1 %.not, float %2103, float %2114, !dbg !7
  %2116 = select i1 %.not, float %2114, float %2104, !dbg !7
  %2117 = select i1 %.not36, float %2115, float %2109, !dbg !7
  %2118 = bitcast float %2117 to i32, !dbg !7
  %2119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2118), !dbg !7
  %2120 = select i1 %.not36, float %2116, float %2110, !dbg !7
  %2121 = bitcast float %2120 to i32, !dbg !7
  %2122 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2121), !dbg !7
  %2123 = extractvalue [2 x <1 x float>] %63, 1, !dbg !19
  %2124 = shufflevector <1 x float> %2123, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %2125 = extractelement <8 x float> %2013, i64 4, !dbg !7
  %2126 = extractelement <8 x float> %2013, i64 5, !dbg !7
  %2127 = extractelement <8 x float> %2013, i64 6, !dbg !7
  %2128 = extractelement <8 x float> %2013, i64 7, !dbg !7
  %2129 = select i1 %.not, float %2126, float %2125, !dbg !7
  %2130 = bitcast float %2129 to i32, !dbg !7
  %2131 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2130), !dbg !7
  %2132 = bitcast i32 %2131 to float, !dbg !7
  %2133 = select i1 %.not, float %2125, float %2132, !dbg !7
  %2134 = select i1 %.not, float %2132, float %2126, !dbg !7
  %2135 = select i1 %.not, float %2128, float %2127, !dbg !7
  %2136 = bitcast float %2135 to i32, !dbg !7
  %2137 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2136), !dbg !7
  %2138 = bitcast i32 %2137 to float, !dbg !7
  %2139 = select i1 %.not, float %2127, float %2138, !dbg !7
  %2140 = select i1 %.not, float %2138, float %2128, !dbg !7
  %2141 = select i1 %.not36, float %2139, float %2133, !dbg !7
  %2142 = bitcast float %2141 to i32, !dbg !7
  %2143 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2142), !dbg !7
  %2144 = bitcast i32 %2143 to float, !dbg !7
  %2145 = select i1 %.not36, float %2133, float %2144, !dbg !7
  %2146 = select i1 %.not36, float %2144, float %2139, !dbg !7
  %2147 = select i1 %.not36, float %2140, float %2134, !dbg !7
  %2148 = bitcast float %2147 to i32, !dbg !7
  %2149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2148), !dbg !7
  %2150 = bitcast i32 %2149 to float, !dbg !7
  %2151 = select i1 %.not36, float %2134, float %2150, !dbg !7
  %2152 = select i1 %.not36, float %2150, float %2140, !dbg !7
  %2153 = extractelement <8 x float> %2042, i64 4, !dbg !7
  %2154 = extractelement <8 x float> %2042, i64 5, !dbg !7
  %2155 = extractelement <8 x float> %2042, i64 6, !dbg !7
  %2156 = extractelement <8 x float> %2042, i64 7, !dbg !7
  %2157 = select i1 %.not, float %2154, float %2153, !dbg !7
  %2158 = bitcast float %2157 to i32, !dbg !7
  %2159 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2158), !dbg !7
  %2160 = bitcast i32 %2159 to float, !dbg !7
  %2161 = select i1 %.not, float %2153, float %2160, !dbg !7
  %2162 = select i1 %.not, float %2160, float %2154, !dbg !7
  %2163 = select i1 %.not, float %2156, float %2155, !dbg !7
  %2164 = bitcast float %2163 to i32, !dbg !7
  %2165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2164), !dbg !7
  %2166 = bitcast i32 %2165 to float, !dbg !7
  %2167 = select i1 %.not, float %2155, float %2166, !dbg !7
  %2168 = select i1 %.not, float %2166, float %2156, !dbg !7
  %2169 = select i1 %.not36, float %2167, float %2161, !dbg !7
  %2170 = bitcast float %2169 to i32, !dbg !7
  %2171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2170), !dbg !7
  %2172 = bitcast i32 %2171 to float, !dbg !7
  %2173 = select i1 %.not36, float %2161, float %2172, !dbg !7
  %2174 = select i1 %.not36, float %2172, float %2167, !dbg !7
  %2175 = select i1 %.not36, float %2168, float %2162, !dbg !7
  %2176 = bitcast float %2175 to i32, !dbg !7
  %2177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2176), !dbg !7
  %2178 = bitcast i32 %2177 to float, !dbg !7
  %2179 = select i1 %.not36, float %2162, float %2178, !dbg !7
  %2180 = select i1 %.not36, float %2178, float %2168, !dbg !7
  %2181 = extractelement <8 x float> %2071, i64 4, !dbg !7
  %2182 = extractelement <8 x float> %2071, i64 5, !dbg !7
  %2183 = extractelement <8 x float> %2071, i64 6, !dbg !7
  %2184 = extractelement <8 x float> %2071, i64 7, !dbg !7
  %2185 = select i1 %.not, float %2182, float %2181, !dbg !7
  %2186 = bitcast float %2185 to i32, !dbg !7
  %2187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2186), !dbg !7
  %2188 = bitcast i32 %2187 to float, !dbg !7
  %2189 = select i1 %.not, float %2181, float %2188, !dbg !7
  %2190 = select i1 %.not, float %2188, float %2182, !dbg !7
  %2191 = select i1 %.not, float %2184, float %2183, !dbg !7
  %2192 = bitcast float %2191 to i32, !dbg !7
  %2193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2192), !dbg !7
  %2194 = bitcast i32 %2193 to float, !dbg !7
  %2195 = select i1 %.not, float %2183, float %2194, !dbg !7
  %2196 = select i1 %.not, float %2194, float %2184, !dbg !7
  %2197 = select i1 %.not36, float %2195, float %2189, !dbg !7
  %2198 = bitcast float %2197 to i32, !dbg !7
  %2199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2198), !dbg !7
  %2200 = bitcast i32 %2199 to float, !dbg !7
  %2201 = select i1 %.not36, float %2189, float %2200, !dbg !7
  %2202 = select i1 %.not36, float %2200, float %2195, !dbg !7
  %2203 = select i1 %.not36, float %2196, float %2190, !dbg !7
  %2204 = bitcast float %2203 to i32, !dbg !7
  %2205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2204), !dbg !7
  %2206 = bitcast i32 %2205 to float, !dbg !7
  %2207 = select i1 %.not36, float %2190, float %2206, !dbg !7
  %2208 = select i1 %.not36, float %2206, float %2196, !dbg !7
  %2209 = extractelement <8 x float> %2100, i64 4, !dbg !7
  %2210 = extractelement <8 x float> %2100, i64 5, !dbg !7
  %2211 = extractelement <8 x float> %2100, i64 6, !dbg !7
  %2212 = extractelement <8 x float> %2100, i64 7, !dbg !7
  %2213 = select i1 %.not, float %2210, float %2209, !dbg !7
  %2214 = bitcast float %2213 to i32, !dbg !7
  %2215 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2214), !dbg !7
  %2216 = bitcast i32 %2215 to float, !dbg !7
  %2217 = select i1 %.not, float %2209, float %2216, !dbg !7
  %2218 = select i1 %.not, float %2216, float %2210, !dbg !7
  %2219 = select i1 %.not, float %2212, float %2211, !dbg !7
  %2220 = bitcast float %2219 to i32, !dbg !7
  %2221 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1796, i32 %2220), !dbg !7
  %2222 = bitcast i32 %2221 to float, !dbg !7
  %2223 = select i1 %.not, float %2211, float %2222, !dbg !7
  %2224 = select i1 %.not, float %2222, float %2212, !dbg !7
  %2225 = select i1 %.not36, float %2223, float %2217, !dbg !7
  %2226 = bitcast float %2225 to i32, !dbg !7
  %2227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2226), !dbg !7
  %2228 = select i1 %.not36, float %2224, float %2218, !dbg !7
  %2229 = bitcast float %2228 to i32, !dbg !7
  %2230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1812, i32 %2229), !dbg !7
  br i1 true, label %2231, label %2295, !dbg !21

2231:                                             ; preds = %1780
  %2232 = bitcast i32 %2227 to float, !dbg !7
  %2233 = select i1 %.not36, float %2217, float %2232, !dbg !7
  %2234 = insertelement <4 x float> poison, float %2233, i64 0, !dbg !19
  %2235 = bitcast i32 %2230 to float, !dbg !7
  %2236 = select i1 %.not36, float %2218, float %2235, !dbg !7
  %2237 = insertelement <4 x float> %2234, float %2236, i64 1, !dbg !19
  %2238 = select i1 %.not36, float %2232, float %2223, !dbg !7
  %2239 = insertelement <4 x float> %2237, float %2238, i64 2, !dbg !19
  %2240 = select i1 %.not36, float %2235, float %2224, !dbg !7
  %2241 = insertelement <4 x float> %2239, float %2240, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2124, !dbg !19
  %normalized = fmul <4 x float> %2241, %row.reciprocal, !dbg !19
  %2242 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2243 = extractelement <4 x half> %2242, i64 3, !dbg !20
  %2244 = extractelement <4 x half> %2242, i64 2, !dbg !20
  %2245 = extractelement <4 x half> %2242, i64 1, !dbg !20
  %2246 = extractelement <4 x half> %2242, i64 0, !dbg !20
  %2247 = bitcast i32 %2119 to float, !dbg !7
  %2248 = select i1 %.not36, float %2109, float %2247, !dbg !7
  %2249 = insertelement <4 x float> poison, float %2248, i64 0, !dbg !19
  %2250 = bitcast i32 %2122 to float, !dbg !7
  %2251 = select i1 %.not36, float %2110, float %2250, !dbg !7
  %2252 = insertelement <4 x float> %2249, float %2251, i64 1, !dbg !19
  %2253 = select i1 %.not36, float %2247, float %2115, !dbg !7
  %2254 = insertelement <4 x float> %2252, float %2253, i64 2, !dbg !19
  %2255 = select i1 %.not36, float %2250, float %2116, !dbg !7
  %2256 = insertelement <4 x float> %2254, float %2255, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %2256, %row.reciprocal, !dbg !19
  %2257 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %2258 = extractelement <4 x half> %2257, i64 3, !dbg !20
  %2259 = extractelement <4 x half> %2257, i64 2, !dbg !20
  %2260 = extractelement <4 x half> %2257, i64 1, !dbg !20
  %2261 = extractelement <4 x half> %2257, i64 0, !dbg !20
  %2262 = bitcast i32 %2009 to float, !dbg !7
  %2263 = select i1 %.not36, float %1999, float %2262, !dbg !7
  %2264 = insertelement <4 x float> poison, float %2263, i64 0, !dbg !19
  %2265 = bitcast i32 %2012 to float, !dbg !7
  %2266 = select i1 %.not36, float %2000, float %2265, !dbg !7
  %2267 = insertelement <4 x float> %2264, float %2266, i64 1, !dbg !19
  %2268 = select i1 %.not36, float %2262, float %2005, !dbg !7
  %2269 = insertelement <4 x float> %2267, float %2268, i64 2, !dbg !19
  %2270 = select i1 %.not36, float %2265, float %2006, !dbg !7
  %2271 = insertelement <4 x float> %2269, float %2270, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1906, !dbg !19
  %normalized106 = fmul <4 x float> %2271, %row.reciprocal105, !dbg !19
  %2272 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %2273 = extractelement <4 x half> %2272, i64 3, !dbg !20
  %2274 = extractelement <4 x half> %2272, i64 2, !dbg !20
  %2275 = extractelement <4 x half> %2272, i64 1, !dbg !20
  %2276 = extractelement <4 x half> %2272, i64 0, !dbg !20
  %2277 = bitcast i32 %1901 to float, !dbg !7
  %2278 = select i1 %.not36, float %1891, float %2277, !dbg !7
  %2279 = insertelement <4 x float> poison, float %2278, i64 0, !dbg !19
  %2280 = bitcast i32 %1904 to float, !dbg !7
  %2281 = select i1 %.not36, float %1892, float %2280, !dbg !7
  %2282 = insertelement <4 x float> %2279, float %2281, i64 1, !dbg !19
  %2283 = select i1 %.not36, float %2277, float %1897, !dbg !7
  %2284 = insertelement <4 x float> %2282, float %2283, i64 2, !dbg !19
  %2285 = select i1 %.not36, float %2280, float %1898, !dbg !7
  %2286 = insertelement <4 x float> %2284, float %2285, i64 3, !dbg !19
  %normalized107 = fmul <4 x float> %2286, %row.reciprocal105, !dbg !19
  %2287 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %2288 = extractelement <4 x half> %2287, i64 3, !dbg !20
  %2289 = extractelement <4 x half> %2287, i64 2, !dbg !20
  %2290 = extractelement <4 x half> %2287, i64 1, !dbg !20
  %2291 = extractelement <4 x half> %2287, i64 0, !dbg !20
  %2292 = add nsw i64 %15, %25, !dbg !21
  %2293 = shl i64 %2292, 19, !dbg !21
  %2294 = and i64 %2293, 16252928, !dbg !21
  %address.thread856 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local857 = zext nneg i32 %address.thread856 to i64, !dbg !12
  %address.local858 = add i64 %28, %address.local857, !dbg !21
  %address.local859 = lshr i64 %address.local857, 6, !dbg !12
  %address.local860 = shl nuw nsw i64 %address.local859, 4, !dbg !21
  %address.local861 = add i64 %address.local858, %address.local860, !dbg !21
  %address.local862 = and i64 %address.local857, 240, !dbg !21
  %address.local863 = sub i64 %address.local861, %address.local862, !dbg !21
  %address.local864 = add i64 %address.local863, %29, !dbg !21
  %address.local865 = shl nuw nsw i64 %address.local864, 7, !dbg !21
  %address.local866 = add nuw nsw i64 %2294, %address.local865, !dbg !21
  %address.local867 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local866, !dbg !21
  %address.local868 = lshr i64 %address.local857, 2, !dbg !12
  %address.local869 = and i64 %address.local868, 12, !dbg !12
  %address.local870 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local867, i64 %address.local869, !dbg !21
  store half %2291, ptr addrspace(1) %address.local870, align 2, !dbg !21
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local866, !dbg !21
  %address.local872 = or disjoint i64 %address.local869, 1, !dbg !21
  %address.local873 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local871, i64 %address.local872, !dbg !21
  store half %2290, ptr addrspace(1) %address.local873, align 2, !dbg !21
  %address.local874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local866, !dbg !21
  %address.local875 = or disjoint i64 %address.local869, 2, !dbg !21
  %address.local876 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local874, i64 %address.local875, !dbg !21
  store half %2289, ptr addrspace(1) %address.local876, align 2, !dbg !21
  %address.local877 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local866, !dbg !21
  %address.local878 = or disjoint i64 %address.local869, 3, !dbg !21
  %address.local879 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local877, i64 %address.local878, !dbg !21
  store half %2288, ptr addrspace(1) %address.local879, align 2, !dbg !21
  %address.local880 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local866, !dbg !21
  %address.local881 = or disjoint i64 %address.local869, 16, !dbg !21
  %address.local882 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local881, !dbg !21
  store half %2276, ptr addrspace(1) %address.local882, align 2, !dbg !21
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local866, !dbg !21
  %address.local884 = or disjoint i64 %address.local869, 17, !dbg !21
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local883, i64 %address.local884, !dbg !21
  store half %2275, ptr addrspace(1) %address.local885, align 2, !dbg !21
  %address.local886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local866, !dbg !21
  %address.local887 = or disjoint i64 %address.local869, 18, !dbg !21
  %address.local888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local886, i64 %address.local887, !dbg !21
  store half %2274, ptr addrspace(1) %address.local888, align 2, !dbg !21
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local866, !dbg !21
  %address.local890 = or disjoint i64 %address.local869, 19, !dbg !21
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local889, i64 %address.local890, !dbg !21
  store half %2273, ptr addrspace(1) %address.local891, align 2, !dbg !21
  %address.local892 = shl i64 %address.local864, 7, !dbg !21
  %address.local893 = add i64 %address.local892, 8192, !dbg !21
  %address.local894 = add nuw nsw i64 %2294, %address.local893, !dbg !21
  %address.local895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local894, !dbg !21
  %address.local896 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local895, i64 %address.local869, !dbg !21
  store half %2261, ptr addrspace(1) %address.local896, align 2, !dbg !21
  %address.local897 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local894, !dbg !21
  %address.local898 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local897, i64 %address.local872, !dbg !21
  store half %2260, ptr addrspace(1) %address.local898, align 2, !dbg !21
  %address.local899 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local894, !dbg !21
  %address.local900 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local899, i64 %address.local875, !dbg !21
  store half %2259, ptr addrspace(1) %address.local900, align 2, !dbg !21
  %address.local901 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local894, !dbg !21
  %address.local902 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local901, i64 %address.local878, !dbg !21
  store half %2258, ptr addrspace(1) %address.local902, align 2, !dbg !21
  %address.local903 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local894, !dbg !21
  %address.local904 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local903, i64 %address.local881, !dbg !21
  store half %2246, ptr addrspace(1) %address.local904, align 2, !dbg !21
  %address.local905 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local894, !dbg !21
  %address.local906 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local905, i64 %address.local884, !dbg !21
  store half %2245, ptr addrspace(1) %address.local906, align 2, !dbg !21
  %address.local907 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local894, !dbg !21
  %address.local908 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local907, i64 %address.local887, !dbg !21
  store half %2244, ptr addrspace(1) %address.local908, align 2, !dbg !21
  %address.local909 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local894, !dbg !21
  %address.local910 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local909, i64 %address.local890, !dbg !21
  store half %2243, ptr addrspace(1) %address.local910, align 2, !dbg !21
  br label %2295, !dbg !21

2295:                                             ; preds = %2231, %1780
  br i1 true, label %2296, label %2334, !dbg !21

2296:                                             ; preds = %2295
  %2297 = insertelement <4 x float> poison, float %2201, i64 0, !dbg !19
  %2298 = insertelement <4 x float> %2297, float %2207, i64 1, !dbg !19
  %2299 = insertelement <4 x float> %2298, float %2202, i64 2, !dbg !19
  %2300 = insertelement <4 x float> %2299, float %2208, i64 3, !dbg !19
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2124, !dbg !19
  %normalized109 = fmul <4 x float> %2300, %row.reciprocal108, !dbg !19
  %2301 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %2302 = extractelement <4 x half> %2301, i64 3, !dbg !20
  %2303 = extractelement <4 x half> %2301, i64 2, !dbg !20
  %2304 = extractelement <4 x half> %2301, i64 1, !dbg !20
  %2305 = extractelement <4 x half> %2301, i64 0, !dbg !20
  %2306 = insertelement <4 x float> poison, float %2092, i64 0, !dbg !19
  %2307 = insertelement <4 x float> %2306, float %2098, i64 1, !dbg !19
  %2308 = insertelement <4 x float> %2307, float %2093, i64 2, !dbg !19
  %2309 = insertelement <4 x float> %2308, float %2099, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %2309, %row.reciprocal108, !dbg !19
  %2310 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %2311 = extractelement <4 x half> %2310, i64 3, !dbg !20
  %2312 = extractelement <4 x half> %2310, i64 2, !dbg !20
  %2313 = extractelement <4 x half> %2310, i64 1, !dbg !20
  %2314 = extractelement <4 x half> %2310, i64 0, !dbg !20
  %2315 = insertelement <4 x float> poison, float %1983, i64 0, !dbg !19
  %2316 = insertelement <4 x float> %2315, float %1989, i64 1, !dbg !19
  %2317 = insertelement <4 x float> %2316, float %1984, i64 2, !dbg !19
  %2318 = insertelement <4 x float> %2317, float %1990, i64 3, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1906, !dbg !19
  %normalized112 = fmul <4 x float> %2318, %row.reciprocal111, !dbg !19
  %2319 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %2320 = extractelement <4 x half> %2319, i64 3, !dbg !20
  %2321 = extractelement <4 x half> %2319, i64 2, !dbg !20
  %2322 = extractelement <4 x half> %2319, i64 1, !dbg !20
  %2323 = extractelement <4 x half> %2319, i64 0, !dbg !20
  %2324 = insertelement <4 x float> poison, float %1874, i64 0, !dbg !19
  %2325 = insertelement <4 x float> %2324, float %1880, i64 1, !dbg !19
  %2326 = insertelement <4 x float> %2325, float %1875, i64 2, !dbg !19
  %2327 = insertelement <4 x float> %2326, float %1881, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %2327, %row.reciprocal111, !dbg !19
  %2328 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %2329 = extractelement <4 x half> %2328, i64 3, !dbg !20
  %2330 = extractelement <4 x half> %2328, i64 2, !dbg !20
  %2331 = extractelement <4 x half> %2328, i64 1, !dbg !20
  %2332 = extractelement <4 x half> %2328, i64 0, !dbg !20
  %2333 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread911 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local912 = zext nneg i32 %address.thread911 to i64, !dbg !12
  %address.local913 = add i64 %28, %address.local912, !dbg !21
  %address.local914 = lshr i64 %address.local912, 6, !dbg !12
  %address.local915 = shl nuw nsw i64 %address.local914, 4, !dbg !21
  %address.local916 = add i64 %address.local913, %address.local915, !dbg !21
  %address.local917 = and i64 %address.local912, 240, !dbg !21
  %address.local918 = sub i64 %address.local916, %address.local917, !dbg !21
  %address.local919 = add i64 %address.local918, %29, !dbg !21
  %address.local920 = shl nuw nsw i64 %address.local919, 7, !dbg !21
  %address.local921 = add nuw nsw i64 %2333, %address.local920, !dbg !21
  %address.local922 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local921, !dbg !21
  %address.local923 = lshr i64 %address.local912, 2, !dbg !12
  %address.local924 = and i64 %address.local923, 12, !dbg !12
  %address.local925 = or disjoint i64 %address.local924, 32, !dbg !21
  %address.local926 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local925, !dbg !21
  store half %2332, ptr addrspace(1) %address.local926, align 2, !dbg !21
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local921, !dbg !21
  %address.local928 = or disjoint i64 %address.local924, 33, !dbg !21
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local927, i64 %address.local928, !dbg !21
  store half %2331, ptr addrspace(1) %address.local929, align 2, !dbg !21
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local921, !dbg !21
  %address.local931 = or disjoint i64 %address.local924, 34, !dbg !21
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local930, i64 %address.local931, !dbg !21
  store half %2330, ptr addrspace(1) %address.local932, align 2, !dbg !21
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local921, !dbg !21
  %address.local934 = or disjoint i64 %address.local924, 35, !dbg !21
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local933, i64 %address.local934, !dbg !21
  store half %2329, ptr addrspace(1) %address.local935, align 2, !dbg !21
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local921, !dbg !21
  %address.local937 = or disjoint i64 %address.local924, 48, !dbg !21
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local936, i64 %address.local937, !dbg !21
  store half %2323, ptr addrspace(1) %address.local938, align 2, !dbg !21
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local921, !dbg !21
  %address.local940 = or disjoint i64 %address.local924, 49, !dbg !21
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local939, i64 %address.local940, !dbg !21
  store half %2322, ptr addrspace(1) %address.local941, align 2, !dbg !21
  %address.local942 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local921, !dbg !21
  %address.local943 = or disjoint i64 %address.local924, 50, !dbg !21
  %address.local944 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local942, i64 %address.local943, !dbg !21
  store half %2321, ptr addrspace(1) %address.local944, align 2, !dbg !21
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local921, !dbg !21
  %address.local946 = or i64 %address.local923, 51, !dbg !21
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local946, !dbg !21
  store half %2320, ptr addrspace(1) %address.local947, align 2, !dbg !21
  %address.local948 = shl i64 %address.local919, 7, !dbg !21
  %address.local949 = add i64 %address.local948, 8192, !dbg !21
  %address.local950 = add nuw nsw i64 %2333, %address.local949, !dbg !21
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local950, !dbg !21
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local951, i64 %address.local925, !dbg !21
  store half %2314, ptr addrspace(1) %address.local952, align 2, !dbg !21
  %address.local953 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local950, !dbg !21
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local928, !dbg !21
  store half %2313, ptr addrspace(1) %address.local954, align 2, !dbg !21
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local950, !dbg !21
  %address.local956 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local955, i64 %address.local931, !dbg !21
  store half %2312, ptr addrspace(1) %address.local956, align 2, !dbg !21
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local950, !dbg !21
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local957, i64 %address.local934, !dbg !21
  store half %2311, ptr addrspace(1) %address.local958, align 2, !dbg !21
  %address.local959 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local950, !dbg !21
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local959, i64 %address.local937, !dbg !21
  store half %2305, ptr addrspace(1) %address.local960, align 2, !dbg !21
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local950, !dbg !21
  %address.local962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local961, i64 %address.local940, !dbg !21
  store half %2304, ptr addrspace(1) %address.local962, align 2, !dbg !21
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local950, !dbg !21
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local963, i64 %address.local943, !dbg !21
  store half %2303, ptr addrspace(1) %address.local964, align 2, !dbg !21
  %address.local965 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local950, !dbg !21
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local965, i64 %address.local946, !dbg !21
  store half %2302, ptr addrspace(1) %address.local966, align 2, !dbg !21
  br label %2334, !dbg !21

2334:                                             ; preds = %2296, %2295
  br i1 true, label %2335, label %2373, !dbg !21

2335:                                             ; preds = %2334
  %2336 = insertelement <4 x float> poison, float %2173, i64 0, !dbg !19
  %2337 = insertelement <4 x float> %2336, float %2179, i64 1, !dbg !19
  %2338 = insertelement <4 x float> %2337, float %2174, i64 2, !dbg !19
  %2339 = insertelement <4 x float> %2338, float %2180, i64 3, !dbg !19
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2124, !dbg !19
  %normalized115 = fmul <4 x float> %2339, %row.reciprocal114, !dbg !19
  %2340 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %2341 = extractelement <4 x half> %2340, i64 3, !dbg !20
  %2342 = extractelement <4 x half> %2340, i64 2, !dbg !20
  %2343 = extractelement <4 x half> %2340, i64 1, !dbg !20
  %2344 = extractelement <4 x half> %2340, i64 0, !dbg !20
  %2345 = insertelement <4 x float> poison, float %2063, i64 0, !dbg !19
  %2346 = insertelement <4 x float> %2345, float %2069, i64 1, !dbg !19
  %2347 = insertelement <4 x float> %2346, float %2064, i64 2, !dbg !19
  %2348 = insertelement <4 x float> %2347, float %2070, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %2348, %row.reciprocal114, !dbg !19
  %2349 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %2350 = extractelement <4 x half> %2349, i64 3, !dbg !20
  %2351 = extractelement <4 x half> %2349, i64 2, !dbg !20
  %2352 = extractelement <4 x half> %2349, i64 1, !dbg !20
  %2353 = extractelement <4 x half> %2349, i64 0, !dbg !20
  %2354 = insertelement <4 x float> poison, float %1955, i64 0, !dbg !19
  %2355 = insertelement <4 x float> %2354, float %1961, i64 1, !dbg !19
  %2356 = insertelement <4 x float> %2355, float %1956, i64 2, !dbg !19
  %2357 = insertelement <4 x float> %2356, float %1962, i64 3, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1906, !dbg !19
  %normalized118 = fmul <4 x float> %2357, %row.reciprocal117, !dbg !19
  %2358 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %2359 = extractelement <4 x half> %2358, i64 3, !dbg !20
  %2360 = extractelement <4 x half> %2358, i64 2, !dbg !20
  %2361 = extractelement <4 x half> %2358, i64 1, !dbg !20
  %2362 = extractelement <4 x half> %2358, i64 0, !dbg !20
  %2363 = insertelement <4 x float> poison, float %1845, i64 0, !dbg !19
  %2364 = insertelement <4 x float> %2363, float %1851, i64 1, !dbg !19
  %2365 = insertelement <4 x float> %2364, float %1846, i64 2, !dbg !19
  %2366 = insertelement <4 x float> %2365, float %1852, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %2366, %row.reciprocal117, !dbg !19
  %2367 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %2368 = extractelement <4 x half> %2367, i64 3, !dbg !20
  %2369 = extractelement <4 x half> %2367, i64 2, !dbg !20
  %2370 = extractelement <4 x half> %2367, i64 1, !dbg !20
  %2371 = extractelement <4 x half> %2367, i64 0, !dbg !20
  %2372 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread967 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local968 = zext nneg i32 %address.thread967 to i64, !dbg !12
  %address.local969 = add i64 %28, %address.local968, !dbg !21
  %address.local970 = lshr i64 %address.local968, 6, !dbg !12
  %address.local971 = shl nuw nsw i64 %address.local970, 4, !dbg !21
  %address.local972 = add i64 %address.local969, %address.local971, !dbg !21
  %address.local973 = and i64 %address.local968, 240, !dbg !21
  %address.local974 = sub i64 %address.local972, %address.local973, !dbg !21
  %address.local975 = add i64 %address.local974, %29, !dbg !21
  %address.local976 = shl nuw nsw i64 %address.local975, 7, !dbg !21
  %address.local977 = add nuw nsw i64 %2372, %address.local976, !dbg !21
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local979 = lshr i64 %address.local968, 2, !dbg !12
  %address.local980 = and i64 %address.local979, 12, !dbg !12
  %address.local981 = or disjoint i64 %address.local980, 64, !dbg !21
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local978, i64 %address.local981, !dbg !21
  store half %2371, ptr addrspace(1) %address.local982, align 2, !dbg !21
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local984 = or disjoint i64 %address.local980, 65, !dbg !21
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local984, !dbg !21
  store half %2370, ptr addrspace(1) %address.local985, align 2, !dbg !21
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local987 = or disjoint i64 %address.local980, 66, !dbg !21
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local986, i64 %address.local987, !dbg !21
  store half %2369, ptr addrspace(1) %address.local988, align 2, !dbg !21
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local990 = or disjoint i64 %address.local980, 67, !dbg !21
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local989, i64 %address.local990, !dbg !21
  store half %2368, ptr addrspace(1) %address.local991, align 2, !dbg !21
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local993 = or disjoint i64 %address.local980, 80, !dbg !21
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local993, !dbg !21
  store half %2362, ptr addrspace(1) %address.local994, align 2, !dbg !21
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local996 = or disjoint i64 %address.local980, 81, !dbg !21
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local995, i64 %address.local996, !dbg !21
  store half %2361, ptr addrspace(1) %address.local997, align 2, !dbg !21
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local999 = or disjoint i64 %address.local980, 82, !dbg !21
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local999, !dbg !21
  store half %2360, ptr addrspace(1) %address.local1000, align 2, !dbg !21
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local1002 = or disjoint i64 %address.local980, 83, !dbg !21
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1001, i64 %address.local1002, !dbg !21
  store half %2359, ptr addrspace(1) %address.local1003, align 2, !dbg !21
  %address.local1004 = shl i64 %address.local975, 7, !dbg !21
  %address.local1005 = add i64 %address.local1004, 8192, !dbg !21
  %address.local1006 = add nuw nsw i64 %2372, %address.local1005, !dbg !21
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1007, i64 %address.local981, !dbg !21
  store half %2353, ptr addrspace(1) %address.local1008, align 2, !dbg !21
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1009, i64 %address.local984, !dbg !21
  store half %2352, ptr addrspace(1) %address.local1010, align 2, !dbg !21
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local987, !dbg !21
  store half %2351, ptr addrspace(1) %address.local1012, align 2, !dbg !21
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !21
  store half %2350, ptr addrspace(1) %address.local1014, align 2, !dbg !21
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local993, !dbg !21
  store half %2344, ptr addrspace(1) %address.local1016, align 2, !dbg !21
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local996, !dbg !21
  store half %2343, ptr addrspace(1) %address.local1018, align 2, !dbg !21
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local999, !dbg !21
  store half %2342, ptr addrspace(1) %address.local1020, align 2, !dbg !21
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local1002, !dbg !21
  store half %2341, ptr addrspace(1) %address.local1022, align 2, !dbg !21
  br label %2373, !dbg !21

2373:                                             ; preds = %2335, %2334
  br i1 true, label %2374, label %2412, !dbg !21

2374:                                             ; preds = %2373
  %2375 = insertelement <4 x float> poison, float %2145, i64 0, !dbg !19
  %2376 = insertelement <4 x float> %2375, float %2151, i64 1, !dbg !19
  %2377 = insertelement <4 x float> %2376, float %2146, i64 2, !dbg !19
  %2378 = insertelement <4 x float> %2377, float %2152, i64 3, !dbg !19
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2124, !dbg !19
  %normalized121 = fmul <4 x float> %2378, %row.reciprocal120, !dbg !19
  %2379 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %2380 = extractelement <4 x half> %2379, i64 3, !dbg !20
  %2381 = extractelement <4 x half> %2379, i64 2, !dbg !20
  %2382 = extractelement <4 x half> %2379, i64 1, !dbg !20
  %2383 = extractelement <4 x half> %2379, i64 0, !dbg !20
  %2384 = insertelement <4 x float> poison, float %2034, i64 0, !dbg !19
  %2385 = insertelement <4 x float> %2384, float %2040, i64 1, !dbg !19
  %2386 = insertelement <4 x float> %2385, float %2035, i64 2, !dbg !19
  %2387 = insertelement <4 x float> %2386, float %2041, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %2387, %row.reciprocal120, !dbg !19
  %2388 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %2389 = extractelement <4 x half> %2388, i64 3, !dbg !20
  %2390 = extractelement <4 x half> %2388, i64 2, !dbg !20
  %2391 = extractelement <4 x half> %2388, i64 1, !dbg !20
  %2392 = extractelement <4 x half> %2388, i64 0, !dbg !20
  %2393 = insertelement <4 x float> poison, float %1927, i64 0, !dbg !19
  %2394 = insertelement <4 x float> %2393, float %1933, i64 1, !dbg !19
  %2395 = insertelement <4 x float> %2394, float %1928, i64 2, !dbg !19
  %2396 = insertelement <4 x float> %2395, float %1934, i64 3, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1906, !dbg !19
  %normalized124 = fmul <4 x float> %2396, %row.reciprocal123, !dbg !19
  %2397 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %2398 = extractelement <4 x half> %2397, i64 3, !dbg !20
  %2399 = extractelement <4 x half> %2397, i64 2, !dbg !20
  %2400 = extractelement <4 x half> %2397, i64 1, !dbg !20
  %2401 = extractelement <4 x half> %2397, i64 0, !dbg !20
  %2402 = insertelement <4 x float> poison, float %1816, i64 0, !dbg !19
  %2403 = insertelement <4 x float> %2402, float %1822, i64 1, !dbg !19
  %2404 = insertelement <4 x float> %2403, float %1817, i64 2, !dbg !19
  %2405 = insertelement <4 x float> %2404, float %1823, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %2405, %row.reciprocal123, !dbg !19
  %2406 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %2407 = extractelement <4 x half> %2406, i64 3, !dbg !20
  %2408 = extractelement <4 x half> %2406, i64 2, !dbg !20
  %2409 = extractelement <4 x half> %2406, i64 1, !dbg !20
  %2410 = extractelement <4 x half> %2406, i64 0, !dbg !20
  %2411 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread1023 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1024 = zext nneg i32 %address.thread1023 to i64, !dbg !12
  %address.local1025 = add i64 %28, %address.local1024, !dbg !21
  %address.local1026 = lshr i64 %address.local1024, 6, !dbg !12
  %address.local1027 = shl nuw nsw i64 %address.local1026, 4, !dbg !21
  %address.local1028 = add i64 %address.local1025, %address.local1027, !dbg !21
  %address.local1029 = and i64 %address.local1024, 240, !dbg !21
  %address.local1030 = sub i64 %address.local1028, %address.local1029, !dbg !21
  %address.local1031 = add i64 %address.local1030, %29, !dbg !21
  %address.local1032 = shl nuw nsw i64 %address.local1031, 7, !dbg !21
  %address.local1033 = add nuw nsw i64 %2411, %address.local1032, !dbg !21
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1033, !dbg !21
  %address.local1035 = lshr i64 %address.local1024, 2, !dbg !12
  %address.local1036 = and i64 %address.local1035, 12, !dbg !12
  %address.local1037 = or disjoint i64 %address.local1036, 96, !dbg !21
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1034, i64 %address.local1037, !dbg !21
  store half %2410, ptr addrspace(1) %address.local1038, align 2, !dbg !21
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1033, !dbg !21
  %address.local1040 = or disjoint i64 %address.local1036, 97, !dbg !21
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1040, !dbg !21
  store half %2409, ptr addrspace(1) %address.local1041, align 2, !dbg !21
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1033, !dbg !21
  %address.local1043 = or disjoint i64 %address.local1036, 98, !dbg !21
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1042, i64 %address.local1043, !dbg !21
  store half %2408, ptr addrspace(1) %address.local1044, align 2, !dbg !21
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1033, !dbg !21
  %address.local1046 = or disjoint i64 %address.local1036, 99, !dbg !21
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1045, i64 %address.local1046, !dbg !21
  store half %2407, ptr addrspace(1) %address.local1047, align 2, !dbg !21
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1033, !dbg !21
  %address.local1049 = or disjoint i64 %address.local1036, 112, !dbg !21
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1049, !dbg !21
  store half %2401, ptr addrspace(1) %address.local1050, align 2, !dbg !21
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1033, !dbg !21
  %address.local1052 = or disjoint i64 %address.local1036, 113, !dbg !21
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1051, i64 %address.local1052, !dbg !21
  store half %2400, ptr addrspace(1) %address.local1053, align 2, !dbg !21
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1033, !dbg !21
  %address.local1055 = or disjoint i64 %address.local1036, 114, !dbg !21
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1055, !dbg !21
  store half %2399, ptr addrspace(1) %address.local1056, align 2, !dbg !21
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1033, !dbg !21
  %address.local1058 = or i64 %address.local1035, 115, !dbg !21
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1057, i64 %address.local1058, !dbg !21
  store half %2398, ptr addrspace(1) %address.local1059, align 2, !dbg !21
  %address.local1060 = shl i64 %address.local1031, 7, !dbg !21
  %address.local1061 = add i64 %address.local1060, 8192, !dbg !21
  %address.local1062 = add nuw nsw i64 %2411, %address.local1061, !dbg !21
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1062, !dbg !21
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1063, i64 %address.local1037, !dbg !21
  store half %2392, ptr addrspace(1) %address.local1064, align 2, !dbg !21
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1062, !dbg !21
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1065, i64 %address.local1040, !dbg !21
  store half %2391, ptr addrspace(1) %address.local1066, align 2, !dbg !21
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1062, !dbg !21
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1043, !dbg !21
  store half %2390, ptr addrspace(1) %address.local1068, align 2, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1062, !dbg !21
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1046, !dbg !21
  store half %2389, ptr addrspace(1) %address.local1070, align 2, !dbg !21
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1062, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1049, !dbg !21
  store half %2383, ptr addrspace(1) %address.local1072, align 2, !dbg !21
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1062, !dbg !21
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1052, !dbg !21
  store half %2382, ptr addrspace(1) %address.local1074, align 2, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1062, !dbg !21
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1055, !dbg !21
  store half %2381, ptr addrspace(1) %address.local1076, align 2, !dbg !21
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1062, !dbg !21
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1058, !dbg !21
  store half %2380, ptr addrspace(1) %address.local1078, align 2, !dbg !21
  br label %2412, !dbg !21

2412:                                             ; preds = %2374, %2373
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
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/online/session-20261026-101601-tehle7yk/tuning/run-20261026-101601-inpooz4_/input-0002-test_input_1x32x4096x128-128-128-32-online/grid-000003")
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
!15 = !DILocation(line: 0, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
!17 = !DILocation(line: 73, column: 26, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!23 = !DILocation(line: 79, column: 24, scope: !4)
!24 = !DILocation(line: 80, column: 23, scope: !4)
!25 = !DILocation(line: 81, column: 26, scope: !4)
!26 = !DILocation(line: 88, column: 21, scope: !4)
!27 = !DILocation(line: 89, column: 20, scope: !4)
!28 = !DILocation(line: 90, column: 22, scope: !4)
!29 = !DILocation(line: 91, column: 23, scope: !4)
!30 = !DILocation(line: 92, column: 20, scope: !4)
!31 = !DILocation(line: 93, column: 22, scope: !4)
!32 = !DILocation(line: 94, column: 22, scope: !4)
!33 = !DILocation(line: 95, column: 20, scope: !4)
!34 = !DILocation(line: 96, column: 16, scope: !4)
!35 = !DILocation(line: 103, column: 22, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
