; ModuleID = 'generated_ir/noncausal_optimization/threads512/run-20261027-163952-a49ris5l/online-baseline/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/threads512/run-20261027-163952-a49ris5l/online-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

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
  %15 = sub nsw i64 31, %14, !dbg !7
  %16 = lshr i64 %10, 2, !dbg !7
  %17 = icmp ugt i64 %10, 1023, !dbg !8
  %18 = add nsw i64 %14, -32, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 32, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %22, %16, !dbg !8
  %24 = shl nsw i64 %15, 7, !dbg !8
  %25 = shl i64 %22, 12, !dbg !8
  %26 = sub i64 %24, %25, !dbg !8
  %27 = shl i64 %23, 19, !dbg !8
  %28 = and i64 %27, 16252928, !dbg !8
  %29 = shl nsw i64 %26, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local304 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local305 = and i64 %address.local304, 3584, !dbg !8
  %address.local306 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local305, !dbg !8
  %address.local307 = ptrtoint ptr addrspace(3) %address.local306 to i64, !dbg !8
  %address.local308 = trunc i64 %address.local307 to i32, !dbg !8
  %32 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local308), !dbg !8
  %address.local309 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local310 = and i64 %address.local309, 16256, !dbg !8
  %address.local311 = lshr i64 %address.local, 4, !dbg !8
  %address.local312 = and i64 %address.local311, 24, !dbg !8
  %address.local313 = or disjoint i64 %address.local310, %address.local312, !dbg !8
  %33 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1039 = or disjoint i64 %address.local313, %30, !dbg !8
  %.idx1039.tr = trunc i64 %.idx1039 to i32, !dbg !8
  %dma.buffer.offset1036 = shl i32 %.idx1039.tr, 1, !dbg !8
  %34 = insertelement <2 x i64> poison, i64 %33, i64 0, !dbg !8
  %35 = bitcast <2 x i64> %34 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1037 = shufflevector <4 x i32> %35, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1038 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %32), !dbg !8
  %36 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1038, i32 %dma.buffer.offset1036, <4 x i32> %dma.buffer.rsrc1037) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local305, !dbg !8
  %address.local316 = ptrtoint ptr addrspace(3) %address.local315 to i64, !dbg !8
  %address.local317 = trunc i64 %address.local316 to i32, !dbg !8
  %37 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local317), !dbg !8
  %dma.buffer.offset1033 = or disjoint i32 %dma.buffer.offset1036, 64, !dbg !8
  %dma.buffer.lds1035 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %37), !dbg !8
  %38 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1035, i32 %dma.buffer.offset1033, <4 x i32> %dma.buffer.rsrc1037) #9, !dbg !8
  %address.local319 = lshr i32 %address.thread, 2, !dbg !10
  %address.local320 = and i32 %address.local319, 4, !dbg !10
  %address.local321 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local322 = and i32 %address.local321, 896, !dbg !10
  %address.local323 = or disjoint i32 %address.local320, %address.local322, !dbg !10
  %address.local324 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local325 = and i32 %address.local324, 120, !dbg !10
  %address.local326 = or disjoint i32 %address.local323, %address.local325, !dbg !10
  %address.local327 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local328 = and i32 %address.local327, 1024, !dbg !10
  %address.local329 = or disjoint i32 %address.local326, %address.local328, !dbg !10
  %address.local330 = zext nneg i32 %address.local329 to i64, !dbg !10
  %address.local331 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !10
  %39 = load <4 x half>, ptr addrspace(3) %address.local331, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local331, i64 4096, !dbg !10
  %40 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %41 = fmul <4 x half> %39, splat (half 1.275630e-01), !dbg !10
  %42 = fmul <4 x half> %40, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1030 = or disjoint i32 %dma.buffer.offset1036, 128, !dbg !8
  %43 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1038, i32 %dma.buffer.offset1030, <4 x i32> %dma.buffer.rsrc1037) #9, !dbg !8
  %lds.const.gep941 = getelementptr i8, ptr addrspace(3) %address.local331, i64 8192, !dbg !10
  %44 = load <4 x half>, ptr addrspace(3) %lds.const.gep941, align 8, !dbg !10
  %lds.const.gep942 = getelementptr i8, ptr addrspace(3) %address.local331, i64 12288, !dbg !10
  %45 = load <4 x half>, ptr addrspace(3) %lds.const.gep942, align 8, !dbg !10
  %46 = fmul <4 x half> %44, splat (half 1.275630e-01), !dbg !10
  %47 = fmul <4 x half> %45, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1027 = or disjoint i32 %dma.buffer.offset1036, 192, !dbg !8
  %48 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1035, i32 %dma.buffer.offset1027, <4 x i32> %dma.buffer.rsrc1037) #9, !dbg !8
  %49 = load <4 x half>, ptr addrspace(3) %address.local331, align 8, !dbg !10
  %50 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %51 = fmul <4 x half> %49, splat (half 1.275630e-01), !dbg !10
  %52 = fmul <4 x half> %50, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %53 = load <4 x half>, ptr addrspace(3) %lds.const.gep941, align 8, !dbg !10
  %54 = load <4 x half>, ptr addrspace(3) %lds.const.gep942, align 8, !dbg !10
  %55 = fmul <4 x half> %53, splat (half 1.275630e-01), !dbg !10
  %56 = fmul <4 x half> %54, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %57 = shl nuw nsw i32 %31, 2, !dbg !12
  %58 = and i32 %57, 12, !dbg !12
  %59 = and i32 %31, 1, !dbg !12
  %60 = add nuw nsw i32 %59, 1, !dbg !12
  %narrow = or disjoint i32 %60, %58, !dbg !12
  %61 = lshr i32 %narrow, 1, !dbg !12
  %62 = zext nneg i32 %61 to i64, !dbg !12
  %63 = shl i64 %16, 19
  %64 = and i64 %63, 16252928
  %65 = ptrtoint ptr addrspace(1) %1 to i64
  %66 = insertelement <2 x i64> poison, i64 %65, i64 0
  %67 = bitcast <2 x i64> %66 to <4 x i32>
  %dma.buffer.rsrc1025 = shufflevector <4 x i32> %67, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %68 = ptrtoint ptr addrspace(1) %2 to i64
  %69 = insertelement <2 x i64> poison, i64 %68, i64 0
  %70 = bitcast <2 x i64> %69 to <4 x i32>
  %dma.buffer.rsrc1013 = shufflevector <4 x i32> %70, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %71, !dbg !11

71:                                               ; preds = %4, %71
  %72 = phi <4 x float> [ zeroinitializer, %4 ], [ %612, %71 ]
  %73 = phi <4 x float> [ zeroinitializer, %4 ], [ %596, %71 ]
  %74 = phi <4 x float> [ zeroinitializer, %4 ], [ %564, %71 ]
  %75 = phi <4 x float> [ zeroinitializer, %4 ], [ %548, %71 ]
  %76 = phi <4 x float> [ zeroinitializer, %4 ], [ %515, %71 ]
  %77 = phi <4 x float> [ zeroinitializer, %4 ], [ %499, %71 ]
  %78 = phi <1 x float> [ splat (float -inf), %4 ], [ %286, %71 ]
  %79 = phi <1 x float> [ zeroinitializer, %4 ], [ %417, %71 ]
  %80 = phi <4 x float> [ zeroinitializer, %4 ], [ %466, %71 ]
  %81 = phi <4 x float> [ zeroinitializer, %4 ], [ %450, %71 ]
  %82 = phi i64 [ 0, %4 ], [ %613, %71 ]
  %83 = shl nuw nsw i64 %82, 14, !dbg !13
  %84 = add nuw nsw i64 %83, %64, !dbg !13
  %address.thread350 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local351 = zext nneg i32 %address.thread350 to i64, !dbg !8
  %address.local352 = shl nuw nsw i64 %address.local351, 3, !dbg !8
  %address.local353 = and i64 %address.local352, 3584, !dbg !8
  %address.local354 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local353, !dbg !8
  %address.local355 = ptrtoint ptr addrspace(3) %address.local354 to i64, !dbg !8
  %address.local356 = trunc i64 %address.local355 to i32, !dbg !8
  %85 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local356), !dbg !13
  %address.local357 = lshr i64 %address.local351, 4, !dbg !13
  %address.local358 = and i64 %address.local357, 24, !dbg !13
  %address.local359 = and i64 %address.local351, 123, !dbg !13
  %address.local360 = lshr i64 %address.local351, 2, !dbg !13
  %address.local361 = and i64 %address.local360, 4, !dbg !13
  %address.local362 = add nuw nsw i64 %address.local361, %address.local351, !dbg !13
  %address.local363 = and i64 %address.local362, 4, !dbg !13
  %address.local364 = or disjoint i64 %address.local363, %address.local359, !dbg !13
  %address.local365 = shl nuw nsw i64 %address.local364, 7, !dbg !13
  %address.local366 = or disjoint i64 %address.local365, %address.local358, !dbg !13
  %.idx1042 = or disjoint i64 %address.local366, %84, !dbg !13
  %.idx1042.tr = trunc nuw nsw i64 %.idx1042 to i32, !dbg !13
  %dma.buffer.offset1024 = shl nuw nsw i32 %.idx1042.tr, 1, !dbg !13
  %dma.buffer.lds1026 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %85), !dbg !13
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1026, i32 %dma.buffer.offset1024, <4 x i32> %dma.buffer.rsrc1025) #9, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %87 = or disjoint i64 %84, 32, !dbg !13
  %address.local368 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local353, !dbg !8
  %address.local369 = ptrtoint ptr addrspace(3) %address.local368 to i64, !dbg !8
  %address.local370 = trunc i64 %address.local369 to i32, !dbg !8
  %88 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local370), !dbg !13
  %.idx1043 = or disjoint i64 %address.local366, %87, !dbg !13
  %.idx1043.tr = trunc nuw nsw i64 %.idx1043 to i32, !dbg !13
  %dma.buffer.offset1021 = shl nuw nsw i32 %.idx1043.tr, 1, !dbg !13
  %dma.buffer.lds1023 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %88), !dbg !13
  %89 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1023, i32 %dma.buffer.offset1021, <4 x i32> %dma.buffer.rsrc1025) #9, !dbg !13
  %address.local376 = shl nuw nsw i64 %address.local351, 1, !dbg !12
  %address.local377 = and i64 %address.local376, 24, !dbg !12
  %address.local379 = shl i64 %address.local351, 62, !dbg !12
  %address.local380 = ashr exact i64 %address.local379, 57, !dbg !12
  %address.local381 = and i64 %address.local380, -64, !dbg !12
  %address.local384 = and i64 %address.local360, 12, !dbg !10
  %address.local386 = and i64 %address.local351, 63, !dbg !10
  %address.local387 = lshr i64 %address.local386, 5, !dbg !10
  %address.local388 = mul nuw nsw i64 %address.local387, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local351, 320, !dbg !12
  %90 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !12
  %.idx1044 = mul nsw i64 %address.local360, -1280, !dbg !12
  %91 = getelementptr i8, ptr addrspace(3) %90, i64 %.idx1044, !dbg !12
  %92 = getelementptr [2 x i8], ptr addrspace(3) %91, i64 %address.local377, !dbg !12
  %93 = getelementptr [2 x i8], ptr addrspace(3) %92, i64 %address.local381, !dbg !12
  %94 = getelementptr [2 x i8], ptr addrspace(3) %93, i64 %address.local384, !dbg !12
  %address.local390 = getelementptr [2 x i8], ptr addrspace(3) %94, i64 %address.local388, !dbg !12
  %95 = load <4 x half>, ptr addrspace(3) %address.local390, align 8, !dbg !12
  %lds.const.gep946 = getelementptr i8, ptr addrspace(3) %address.local390, i64 4096, !dbg !12
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep946, align 8, !dbg !12
  %97 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %41, <4 x half> %95, <4 x float> zeroinitializer) #8, !dbg !12
  %address.local401 = shl nuw nsw i32 %address.thread350, 1, !dbg !12
  %address.local402 = and i32 %address.local401, 24, !dbg !12
  %address.local403 = shl nuw nsw i32 %address.thread350, 8, !dbg !12
  %address.local404 = and i32 %address.local403, 768, !dbg !12
  %address.local405 = or disjoint i32 %address.local402, %address.local404, !dbg !12
  %address.local406 = shl nuw nsw i32 %address.thread350, 5, !dbg !12
  %address.local407 = and i32 %address.local406, 96, !dbg !12
  %address.local408 = or disjoint i32 %address.local405, %address.local407, !dbg !12
  %address.local409 = zext nneg i32 %address.local408 to i64, !dbg !12
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %62) #8, !dbg !12, !frisk.register_lifetimes !5
  %address.local410 = lshr i32 %address.thread350, 1, !dbg !12
  %address.local411 = and i32 %address.local410, 1, !dbg !12
  %address.local412 = zext nneg i32 %address.local411 to i64, !dbg !12
  %address.local413 = add nsw i64 %address.base, %address.local412, !dbg !12
  %address.local414 = shl i64 %address.local413, 6, !dbg !12
  %address.local415 = sub i64 %address.local409, %address.local414, !dbg !12
  %98 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local415, !dbg !12
  %99 = getelementptr [2 x i8], ptr addrspace(3) %98, i64 %address.local384, !dbg !12
  %100 = getelementptr [2 x i8], ptr addrspace(3) %99, i64 %address.local388, !dbg !12
  %address.local419 = getelementptr i8, ptr addrspace(3) %100, i64 64, !dbg !12
  %101 = load <4 x half>, ptr addrspace(3) %address.local419, align 8, !dbg !12
  %102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %96, <4 x float> %97) #8, !dbg !12
  %lds.const.gep947 = getelementptr i8, ptr addrspace(3) %100, i64 4160, !dbg !12
  %103 = load <4 x half>, ptr addrspace(3) %lds.const.gep947, align 8, !dbg !12
  %lds.const.gep948 = getelementptr i8, ptr addrspace(3) %address.local390, i64 128, !dbg !12
  %104 = load <4 x half>, ptr addrspace(3) %lds.const.gep948, align 8, !dbg !12
  %105 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %41, <4 x half> %101, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep949 = getelementptr i8, ptr addrspace(3) %address.local390, i64 4224, !dbg !12
  %106 = load <4 x half>, ptr addrspace(3) %lds.const.gep949, align 8, !dbg !12
  %107 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %103, <4 x float> %105) #8, !dbg !12
  %lds.const.gep950 = getelementptr i8, ptr addrspace(3) %100, i64 192, !dbg !12
  %108 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !12
  %lds.const.gep951 = getelementptr i8, ptr addrspace(3) %100, i64 4288, !dbg !12
  %109 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !12
  %110 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %41, <4 x half> %104, <4 x float> zeroinitializer) #8, !dbg !12
  %111 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %106, <4 x float> %110) #8, !dbg !12
  %112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %41, <4 x half> %108, <4 x float> zeroinitializer) #8, !dbg !12
  %113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %109, <4 x float> %112) #8, !dbg !12
  %lds.const.gep952 = getelementptr i8, ptr addrspace(3) %address.local390, i64 1024, !dbg !12
  %114 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !12
  %lds.const.gep953 = getelementptr i8, ptr addrspace(3) %address.local390, i64 5120, !dbg !12
  %115 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !12
  %116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %41, <4 x half> %114, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep954 = getelementptr i8, ptr addrspace(3) %100, i64 1088, !dbg !12
  %117 = load <4 x half>, ptr addrspace(3) %lds.const.gep954, align 8, !dbg !12
  %118 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %115, <4 x float> %116) #8, !dbg !12
  %lds.const.gep955 = getelementptr i8, ptr addrspace(3) %100, i64 5184, !dbg !12
  %119 = load <4 x half>, ptr addrspace(3) %lds.const.gep955, align 8, !dbg !12
  %120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %41, <4 x half> %117, <4 x float> zeroinitializer) #8, !dbg !12
  %121 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %119, <4 x float> %120) #8, !dbg !12
  %lds.const.gep956 = getelementptr i8, ptr addrspace(3) %address.local390, i64 1152, !dbg !12
  %122 = load <4 x half>, ptr addrspace(3) %lds.const.gep956, align 8, !dbg !12
  %lds.const.gep957 = getelementptr i8, ptr addrspace(3) %address.local390, i64 5248, !dbg !12
  %123 = load <4 x half>, ptr addrspace(3) %lds.const.gep957, align 8, !dbg !12
  %124 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %41, <4 x half> %122, <4 x float> zeroinitializer) #8, !dbg !12
  %125 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %123, <4 x float> %124) #8, !dbg !12
  %lds.const.gep958 = getelementptr i8, ptr addrspace(3) %100, i64 1216, !dbg !12
  %126 = load <4 x half>, ptr addrspace(3) %lds.const.gep958, align 8, !dbg !12
  %127 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %41, <4 x half> %126, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep959 = getelementptr i8, ptr addrspace(3) %100, i64 5312, !dbg !12
  %128 = load <4 x half>, ptr addrspace(3) %lds.const.gep959, align 8, !dbg !12
  %129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %128, <4 x float> %127) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %130 = or disjoint i64 %84, 64, !dbg !13
  %.idx10451046 = or disjoint i64 %address.local366, %130, !dbg !13
  %.idx10451046.tr = trunc nuw nsw i64 %.idx10451046 to i32, !dbg !13
  %dma.buffer.offset1018 = shl nuw nsw i32 %.idx10451046.tr, 1, !dbg !13
  %131 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1026, i32 %dma.buffer.offset1018, <4 x i32> %dma.buffer.rsrc1025) #9, !dbg !13
  %lds.const.gep960 = getelementptr i8, ptr addrspace(3) %address.local390, i64 8192, !dbg !12
  %132 = load <4 x half>, ptr addrspace(3) %lds.const.gep960, align 8, !dbg !12
  %133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %132, <4 x float> %102) #8, !dbg !12
  %lds.const.gep961 = getelementptr i8, ptr addrspace(3) %address.local390, i64 12288, !dbg !12
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep961, align 8, !dbg !12
  %lds.const.gep962 = getelementptr i8, ptr addrspace(3) %100, i64 8256, !dbg !12
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep962, align 8, !dbg !12
  %136 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %134, <4 x float> %133) #8, !dbg !12
  %137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %135, <4 x float> %107) #8, !dbg !12
  %lds.const.gep963 = getelementptr i8, ptr addrspace(3) %100, i64 12352, !dbg !12
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep963, align 8, !dbg !12
  %lds.const.gep964 = getelementptr i8, ptr addrspace(3) %address.local390, i64 8320, !dbg !12
  %139 = load <4 x half>, ptr addrspace(3) %lds.const.gep964, align 8, !dbg !12
  %140 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %138, <4 x float> %137) #8, !dbg !12
  %141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %139, <4 x float> %111) #8, !dbg !12
  %lds.const.gep965 = getelementptr i8, ptr addrspace(3) %address.local390, i64 12416, !dbg !12
  %142 = load <4 x half>, ptr addrspace(3) %lds.const.gep965, align 8, !dbg !12
  %lds.const.gep966 = getelementptr i8, ptr addrspace(3) %100, i64 8384, !dbg !12
  %143 = load <4 x half>, ptr addrspace(3) %lds.const.gep966, align 8, !dbg !12
  %144 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %142, <4 x float> %141) #8, !dbg !12
  %145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %143, <4 x float> %113) #8, !dbg !12
  %lds.const.gep967 = getelementptr i8, ptr addrspace(3) %100, i64 12480, !dbg !12
  %146 = load <4 x half>, ptr addrspace(3) %lds.const.gep967, align 8, !dbg !12
  %lds.const.gep968 = getelementptr i8, ptr addrspace(3) %address.local390, i64 9216, !dbg !12
  %147 = load <4 x half>, ptr addrspace(3) %lds.const.gep968, align 8, !dbg !12
  %148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %146, <4 x float> %145) #8, !dbg !12
  %149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %147, <4 x float> %118) #8, !dbg !12
  %lds.const.gep969 = getelementptr i8, ptr addrspace(3) %address.local390, i64 13312, !dbg !12
  %150 = load <4 x half>, ptr addrspace(3) %lds.const.gep969, align 8, !dbg !12
  %lds.const.gep970 = getelementptr i8, ptr addrspace(3) %100, i64 9280, !dbg !12
  %151 = load <4 x half>, ptr addrspace(3) %lds.const.gep970, align 8, !dbg !12
  %152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %150, <4 x float> %149) #8, !dbg !12
  %153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %151, <4 x float> %121) #8, !dbg !12
  %lds.const.gep971 = getelementptr i8, ptr addrspace(3) %100, i64 13376, !dbg !12
  %154 = load <4 x half>, ptr addrspace(3) %lds.const.gep971, align 8, !dbg !12
  %lds.const.gep972 = getelementptr i8, ptr addrspace(3) %address.local390, i64 9344, !dbg !12
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep972, align 8, !dbg !12
  %156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %154, <4 x float> %153) #8, !dbg !12
  %157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %155, <4 x float> %125) #8, !dbg !12
  %lds.const.gep973 = getelementptr i8, ptr addrspace(3) %address.local390, i64 13440, !dbg !12
  %158 = load <4 x half>, ptr addrspace(3) %lds.const.gep973, align 8, !dbg !12
  %lds.const.gep974 = getelementptr i8, ptr addrspace(3) %100, i64 9408, !dbg !12
  %159 = load <4 x half>, ptr addrspace(3) %lds.const.gep974, align 8, !dbg !12
  %160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %158, <4 x float> %157) #8, !dbg !12
  %161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %159, <4 x float> %129) #8, !dbg !12
  %lds.const.gep975 = getelementptr i8, ptr addrspace(3) %100, i64 13504, !dbg !12
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep975, align 8, !dbg !12
  %163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %162, <4 x float> %161) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %164 = or disjoint i64 %84, 96, !dbg !13
  %.idx10471048 = or disjoint i64 %address.local366, %164, !dbg !13
  %.idx10471048.tr = trunc nuw nsw i64 %.idx10471048 to i32, !dbg !13
  %dma.buffer.offset1015 = shl nuw nsw i32 %.idx10471048.tr, 1, !dbg !13
  %165 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1023, i32 %dma.buffer.offset1015, <4 x i32> %dma.buffer.rsrc1025) #9, !dbg !13
  %166 = load <4 x half>, ptr addrspace(3) %address.local390, align 8, !dbg !12
  %167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %166, <4 x float> %136) #8, !dbg !12
  %168 = load <4 x half>, ptr addrspace(3) %lds.const.gep946, align 8, !dbg !12
  %169 = load <4 x half>, ptr addrspace(3) %address.local419, align 8, !dbg !12
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %168, <4 x float> %167) #8, !dbg !12
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %169, <4 x float> %140) #8, !dbg !12
  %172 = load <4 x half>, ptr addrspace(3) %lds.const.gep947, align 8, !dbg !12
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep948, align 8, !dbg !12
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %172, <4 x float> %171) #8, !dbg !12
  %175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %173, <4 x float> %144) #8, !dbg !12
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep949, align 8, !dbg !12
  %177 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !12
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %176, <4 x float> %175) #8, !dbg !12
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %177, <4 x float> %148) #8, !dbg !12
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !12
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !12
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %180, <4 x float> %179) #8, !dbg !12
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %181, <4 x float> %152) #8, !dbg !12
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !12
  %185 = load <4 x half>, ptr addrspace(3) %lds.const.gep954, align 8, !dbg !12
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %184, <4 x float> %183) #8, !dbg !12
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %185, <4 x float> %156) #8, !dbg !12
  %188 = load <4 x half>, ptr addrspace(3) %lds.const.gep955, align 8, !dbg !12
  %189 = load <4 x half>, ptr addrspace(3) %lds.const.gep956, align 8, !dbg !12
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %188, <4 x float> %187) #8, !dbg !12
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %189, <4 x float> %160) #8, !dbg !12
  %192 = load <4 x half>, ptr addrspace(3) %lds.const.gep957, align 8, !dbg !12
  %193 = load <4 x half>, ptr addrspace(3) %lds.const.gep958, align 8, !dbg !12
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %192, <4 x float> %191) #8, !dbg !12
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %193, <4 x float> %163) #8, !dbg !12
  %196 = load <4 x half>, ptr addrspace(3) %lds.const.gep959, align 8, !dbg !12
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %196, <4 x float> %195) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local746 = shl nuw nsw i64 %address.local351, 5, !dbg !14
  %address.local747 = and i64 %address.local746, 16256, !dbg !14
  %address.local749 = and i64 %address.local352, 24, !dbg !14
  %address.local750 = or disjoint i64 %address.local747, %address.local749, !dbg !14
  %.idx10491050 = or disjoint i64 %address.local750, %84, !dbg !14
  %.idx10491050.tr = trunc nuw nsw i64 %.idx10491050 to i32, !dbg !14
  %dma.buffer.offset1012 = shl nuw nsw i32 %.idx10491050.tr, 1, !dbg !14
  %198 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1026, i32 %dma.buffer.offset1012, <4 x i32> %dma.buffer.rsrc1013) #9, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %lds.const.gep960, align 8, !dbg !12
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %199, <4 x float> %170) #8, !dbg !12
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep961, align 8, !dbg !12
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep962, align 8, !dbg !12
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %201, <4 x float> %200) #8, !dbg !12
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %202, <4 x float> %174) #8, !dbg !12
  %205 = extractelement <4 x float> %203, i64 0, !dbg !15
  %206 = tail call float @llvm.maxnum.f32(float %205, float -inf), !dbg !16, !frisk.local_reduction !5
  %207 = extractelement <4 x float> %203, i64 1, !dbg !15
  %208 = extractelement <4 x float> %203, i64 2, !dbg !15
  %209 = extractelement <4 x float> %203, i64 3, !dbg !15
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep963, align 8, !dbg !12
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep964, align 8, !dbg !12
  %212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %210, <4 x float> %204) #8, !dbg !12
  %213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %211, <4 x float> %178) #8, !dbg !12
  %214 = extractelement <4 x float> %212, i64 0, !dbg !15
  %local.reduce77 = tail call float @llvm.maxnum.f32(float %206, float %214), !dbg !16, !frisk.local_reduction !5
  %215 = extractelement <4 x float> %212, i64 1, !dbg !15
  %local.reduce78 = tail call float @llvm.maxnum.f32(float %207, float %215), !dbg !16, !frisk.local_reduction !5
  %216 = extractelement <4 x float> %212, i64 2, !dbg !15
  %local.reduce79 = tail call float @llvm.maxnum.f32(float %208, float %216), !dbg !16, !frisk.local_reduction !5
  %217 = extractelement <4 x float> %212, i64 3, !dbg !15
  %local.reduce80 = tail call float @llvm.maxnum.f32(float %209, float %217), !dbg !16, !frisk.local_reduction !5
  %218 = load <4 x half>, ptr addrspace(3) %lds.const.gep965, align 8, !dbg !12
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep966, align 8, !dbg !12
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %218, <4 x float> %213) #8, !dbg !12
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %219, <4 x float> %182) #8, !dbg !12
  %222 = extractelement <4 x float> %220, i64 0, !dbg !15
  %local.reduce81 = tail call float @llvm.maxnum.f32(float %local.reduce77, float %222), !dbg !16, !frisk.local_reduction !5
  %223 = extractelement <4 x float> %220, i64 1, !dbg !15
  %local.reduce82 = tail call float @llvm.maxnum.f32(float %local.reduce78, float %223), !dbg !16, !frisk.local_reduction !5
  %224 = extractelement <4 x float> %220, i64 2, !dbg !15
  %local.reduce83 = tail call float @llvm.maxnum.f32(float %local.reduce79, float %224), !dbg !16, !frisk.local_reduction !5
  %225 = extractelement <4 x float> %220, i64 3, !dbg !15
  %local.reduce84 = tail call float @llvm.maxnum.f32(float %local.reduce80, float %225), !dbg !16, !frisk.local_reduction !5
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep967, align 8, !dbg !12
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep968, align 8, !dbg !12
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %226, <4 x float> %221) #8, !dbg !12
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %227, <4 x float> %186) #8, !dbg !12
  %230 = extractelement <4 x float> %228, i64 0, !dbg !15
  %local.reduce85 = tail call float @llvm.maxnum.f32(float %local.reduce81, float %230), !dbg !16, !frisk.local_reduction !5
  %231 = extractelement <4 x float> %228, i64 1, !dbg !15
  %local.reduce86 = tail call float @llvm.maxnum.f32(float %local.reduce82, float %231), !dbg !16, !frisk.local_reduction !5
  %232 = extractelement <4 x float> %228, i64 2, !dbg !15
  %local.reduce87 = tail call float @llvm.maxnum.f32(float %local.reduce83, float %232), !dbg !16, !frisk.local_reduction !5
  %233 = extractelement <4 x float> %228, i64 3, !dbg !15
  %local.reduce88 = tail call float @llvm.maxnum.f32(float %local.reduce84, float %233), !dbg !16, !frisk.local_reduction !5
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep969, align 8, !dbg !12
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep970, align 8, !dbg !12
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %234, <4 x float> %229) #8, !dbg !12
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %235, <4 x float> %190) #8, !dbg !12
  %238 = extractelement <4 x float> %236, i64 0, !dbg !15
  %local.reduce89 = tail call float @llvm.maxnum.f32(float %local.reduce85, float %238), !dbg !16, !frisk.local_reduction !5
  %239 = extractelement <4 x float> %236, i64 1, !dbg !15
  %local.reduce90 = tail call float @llvm.maxnum.f32(float %local.reduce86, float %239), !dbg !16, !frisk.local_reduction !5
  %240 = extractelement <4 x float> %236, i64 2, !dbg !15
  %local.reduce91 = tail call float @llvm.maxnum.f32(float %local.reduce87, float %240), !dbg !16, !frisk.local_reduction !5
  %241 = extractelement <4 x float> %236, i64 3, !dbg !15
  %local.reduce92 = tail call float @llvm.maxnum.f32(float %local.reduce88, float %241), !dbg !16, !frisk.local_reduction !5
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep971, align 8, !dbg !12
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep972, align 8, !dbg !12
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %242, <4 x float> %237) #8, !dbg !12
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %243, <4 x float> %194) #8, !dbg !12
  %246 = extractelement <4 x float> %244, i64 0, !dbg !15
  %local.reduce93 = tail call float @llvm.maxnum.f32(float %local.reduce89, float %246), !dbg !16, !frisk.local_reduction !5
  %247 = extractelement <4 x float> %244, i64 1, !dbg !15
  %local.reduce94 = tail call float @llvm.maxnum.f32(float %local.reduce90, float %247), !dbg !16, !frisk.local_reduction !5
  %248 = extractelement <4 x float> %244, i64 2, !dbg !15
  %local.reduce95 = tail call float @llvm.maxnum.f32(float %local.reduce91, float %248), !dbg !16, !frisk.local_reduction !5
  %249 = extractelement <4 x float> %244, i64 3, !dbg !15
  %local.reduce96 = tail call float @llvm.maxnum.f32(float %local.reduce92, float %249), !dbg !16, !frisk.local_reduction !5
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep973, align 8, !dbg !12
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep974, align 8, !dbg !12
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %250, <4 x float> %245) #8, !dbg !12
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %251, <4 x float> %197) #8, !dbg !12
  %254 = extractelement <4 x float> %252, i64 0, !dbg !15
  %local.reduce97 = tail call float @llvm.maxnum.f32(float %local.reduce93, float %254), !dbg !16, !frisk.local_reduction !5
  %255 = extractelement <4 x float> %252, i64 1, !dbg !15
  %local.reduce98 = tail call float @llvm.maxnum.f32(float %local.reduce94, float %255), !dbg !16, !frisk.local_reduction !5
  %256 = extractelement <4 x float> %252, i64 2, !dbg !15
  %local.reduce99 = tail call float @llvm.maxnum.f32(float %local.reduce95, float %256), !dbg !16, !frisk.local_reduction !5
  %257 = extractelement <4 x float> %252, i64 3, !dbg !15
  %local.reduce100 = tail call float @llvm.maxnum.f32(float %local.reduce96, float %257), !dbg !16, !frisk.local_reduction !5
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep975, align 8, !dbg !12
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %258, <4 x float> %253) #8, !dbg !12
  %260 = extractelement <4 x float> %259, i64 0, !dbg !15
  %local.reduce101 = tail call float @llvm.maxnum.f32(float %local.reduce97, float %260), !dbg !16, !frisk.local_reduction !5
  %261 = extractelement <4 x float> %259, i64 1, !dbg !15
  %local.reduce102 = tail call float @llvm.maxnum.f32(float %local.reduce98, float %261), !dbg !16, !frisk.local_reduction !5
  %262 = extractelement <4 x float> %259, i64 2, !dbg !15
  %local.reduce103 = tail call float @llvm.maxnum.f32(float %local.reduce99, float %262), !dbg !16, !frisk.local_reduction !5
  %263 = extractelement <4 x float> %259, i64 3, !dbg !15
  %local.reduce104 = tail call float @llvm.maxnum.f32(float %local.reduce100, float %263), !dbg !16, !frisk.local_reduction !5
  %local.reduce105 = tail call float @llvm.maxnum.f32(float %local.reduce101, float %local.reduce102), !dbg !16, !frisk.local_reduction !5
  %local.reduce106 = tail call float @llvm.maxnum.f32(float %local.reduce103, float %local.reduce104), !dbg !16, !frisk.local_reduction !5
  %264 = tail call float @llvm.maxnum.f32(float %local.reduce105, float %local.reduce106), !dbg !16, !frisk.local_reduction !5
  %265 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %266 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %265), !dbg !16
  %267 = add nuw nsw i32 %266, 64, !dbg !16
  %268 = xor i32 %266, 16, !dbg !16
  %269 = and i32 %267, 192, !dbg !16
  %270 = icmp samesign ult i32 %268, %269, !dbg !16
  %271 = select i1 %270, i32 %268, i32 %266, !dbg !16
  %272 = shl nuw nsw i32 %271, 2, !dbg !16
  %273 = bitcast float %264 to i32, !dbg !16
  %274 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %273), !dbg !16
  %275 = bitcast i32 %274 to float, !dbg !16
  %276 = tail call float @llvm.maxnum.f32(float %264, float %275), !dbg !16
  %277 = xor i32 %266, 32, !dbg !16
  %278 = icmp samesign ult i32 %277, %269, !dbg !16
  %279 = select i1 %278, i32 %277, i32 %266, !dbg !16
  %280 = shl nuw nsw i32 %279, 2, !dbg !16
  %281 = bitcast float %276 to i32, !dbg !16
  %282 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %281), !dbg !16
  %283 = bitcast i32 %282 to float, !dbg !16
  %284 = tail call float @llvm.maxnum.f32(float %276, float %283), !dbg !16
  %285 = insertelement <1 x float> poison, float %284, i64 0, !dbg !16
  %286 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %285, <1 x float> %78), !dbg !16
  %287 = shufflevector <1 x float> %286, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %288 = fsub <4 x float> %203, %287, !dbg !17
  %289 = fsub <4 x float> %212, %287, !dbg !17
  %290 = fsub <4 x float> %220, %287, !dbg !17
  %291 = fsub <4 x float> %228, %287, !dbg !17
  %292 = fsub <4 x float> %236, %287, !dbg !17
  %293 = fsub <4 x float> %244, %287, !dbg !17
  %294 = fsub <4 x float> %252, %287, !dbg !17
  %295 = fsub <4 x float> %259, %287, !dbg !17
  %296 = fsub <1 x float> %78, %286, !dbg !18
  %297 = extractelement <1 x float> %296, i64 0, !dbg !19
  %298 = tail call float @llvm.amdgcn.exp2.f32(float %297), !dbg !19
  %299 = insertelement <4 x float> poison, float %298, i64 0, !dbg !20
  %300 = shufflevector <4 x float> %299, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %301 = fmul <4 x float> %72, %300, !dbg !20
  %302 = fmul <4 x float> %73, %300, !dbg !20
  %303 = fmul <4 x float> %74, %300, !dbg !20
  %304 = fmul <4 x float> %75, %300, !dbg !20
  %305 = fmul <4 x float> %76, %300, !dbg !20
  %306 = fmul <4 x float> %77, %300, !dbg !20
  %307 = fmul <4 x float> %80, %300, !dbg !20
  %308 = fmul <4 x float> %81, %300, !dbg !20
  %309 = extractelement <1 x float> %79, i64 0, !dbg !21
  %310 = fmul float %309, %298, !dbg !21
  %311 = extractelement <4 x float> %288, i64 0, !dbg !22
  %312 = tail call float @llvm.amdgcn.exp2.f32(float %311), !dbg !22
  %313 = insertelement <4 x float> poison, float %312, i64 0, !dbg !23
  %314 = extractelement <4 x float> %288, i64 1, !dbg !22
  %315 = tail call float @llvm.amdgcn.exp2.f32(float %314), !dbg !22
  %316 = insertelement <4 x float> %313, float %315, i64 1, !dbg !23
  %317 = extractelement <4 x float> %288, i64 2, !dbg !22
  %318 = tail call float @llvm.amdgcn.exp2.f32(float %317), !dbg !22
  %319 = insertelement <4 x float> %316, float %318, i64 2, !dbg !23
  %320 = extractelement <4 x float> %288, i64 3, !dbg !22
  %321 = tail call float @llvm.amdgcn.exp2.f32(float %320), !dbg !22
  %322 = insertelement <4 x float> %319, float %321, i64 3, !dbg !23
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %322) #8, !dbg !23, !frisk.register_lifetimes !5
  %323 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !23
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %323) #8, !dbg !22, !frisk.register_lifetimes !5
  %324 = extractelement <4 x float> %289, i64 0, !dbg !22
  %325 = tail call float @llvm.amdgcn.exp2.f32(float %324), !dbg !22
  %326 = insertelement <4 x float> poison, float %325, i64 0, !dbg !23
  %local.reduce = fadd float %312, %325, !dbg !24, !frisk.local_reduction !5
  %327 = extractelement <4 x float> %289, i64 1, !dbg !22
  %328 = tail call float @llvm.amdgcn.exp2.f32(float %327), !dbg !22
  %329 = insertelement <4 x float> %326, float %328, i64 1, !dbg !23
  %local.reduce47 = fadd float %315, %328, !dbg !24, !frisk.local_reduction !5
  %330 = extractelement <4 x float> %289, i64 2, !dbg !22
  %331 = tail call float @llvm.amdgcn.exp2.f32(float %330), !dbg !22
  %332 = insertelement <4 x float> %329, float %331, i64 2, !dbg !23
  %local.reduce48 = fadd float %318, %331, !dbg !24, !frisk.local_reduction !5
  %333 = extractelement <4 x float> %289, i64 3, !dbg !22
  %334 = tail call float @llvm.amdgcn.exp2.f32(float %333), !dbg !22
  %335 = insertelement <4 x float> %332, float %334, i64 3, !dbg !23
  %math.result290 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %335) #8, !dbg !23, !frisk.register_lifetimes !5
  %336 = fptrunc <4 x float> %math.result290 to <4 x half>, !dbg !23
  %math.packed291 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %336) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce49 = fadd float %321, %334, !dbg !24, !frisk.local_reduction !5
  %337 = extractelement <4 x float> %290, i64 0, !dbg !22
  %338 = tail call float @llvm.amdgcn.exp2.f32(float %337), !dbg !22
  %339 = insertelement <4 x float> poison, float %338, i64 0, !dbg !23
  %local.reduce50 = fadd float %338, %local.reduce, !dbg !24, !frisk.local_reduction !5
  %340 = extractelement <4 x float> %290, i64 1, !dbg !22
  %341 = tail call float @llvm.amdgcn.exp2.f32(float %340), !dbg !22
  %342 = insertelement <4 x float> %339, float %341, i64 1, !dbg !23
  %local.reduce51 = fadd float %341, %local.reduce47, !dbg !24, !frisk.local_reduction !5
  %343 = extractelement <4 x float> %290, i64 2, !dbg !22
  %344 = tail call float @llvm.amdgcn.exp2.f32(float %343), !dbg !22
  %345 = insertelement <4 x float> %342, float %344, i64 2, !dbg !23
  %local.reduce52 = fadd float %344, %local.reduce48, !dbg !24, !frisk.local_reduction !5
  %346 = extractelement <4 x float> %290, i64 3, !dbg !22
  %347 = tail call float @llvm.amdgcn.exp2.f32(float %346), !dbg !22
  %348 = insertelement <4 x float> %345, float %347, i64 3, !dbg !23
  %math.result292 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %348) #8, !dbg !23, !frisk.register_lifetimes !5
  %349 = fptrunc <4 x float> %math.result292 to <4 x half>, !dbg !23
  %math.packed293 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %349) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce53 = fadd float %347, %local.reduce49, !dbg !24, !frisk.local_reduction !5
  %350 = extractelement <4 x float> %291, i64 0, !dbg !22
  %351 = tail call float @llvm.amdgcn.exp2.f32(float %350), !dbg !22
  %352 = insertelement <4 x float> poison, float %351, i64 0, !dbg !23
  %local.reduce54 = fadd float %351, %local.reduce50, !dbg !24, !frisk.local_reduction !5
  %353 = extractelement <4 x float> %291, i64 1, !dbg !22
  %354 = tail call float @llvm.amdgcn.exp2.f32(float %353), !dbg !22
  %355 = insertelement <4 x float> %352, float %354, i64 1, !dbg !23
  %local.reduce55 = fadd float %354, %local.reduce51, !dbg !24, !frisk.local_reduction !5
  %356 = extractelement <4 x float> %291, i64 2, !dbg !22
  %357 = tail call float @llvm.amdgcn.exp2.f32(float %356), !dbg !22
  %358 = insertelement <4 x float> %355, float %357, i64 2, !dbg !23
  %local.reduce56 = fadd float %357, %local.reduce52, !dbg !24, !frisk.local_reduction !5
  %359 = extractelement <4 x float> %291, i64 3, !dbg !22
  %360 = tail call float @llvm.amdgcn.exp2.f32(float %359), !dbg !22
  %361 = insertelement <4 x float> %358, float %360, i64 3, !dbg !23
  %math.result294 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %361) #8, !dbg !23, !frisk.register_lifetimes !5
  %362 = fptrunc <4 x float> %math.result294 to <4 x half>, !dbg !23
  %math.packed295 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %362) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce57 = fadd float %360, %local.reduce53, !dbg !24, !frisk.local_reduction !5
  %363 = extractelement <4 x float> %292, i64 0, !dbg !22
  %364 = tail call float @llvm.amdgcn.exp2.f32(float %363), !dbg !22
  %365 = insertelement <4 x float> poison, float %364, i64 0, !dbg !23
  %local.reduce58 = fadd float %364, %local.reduce54, !dbg !24, !frisk.local_reduction !5
  %366 = extractelement <4 x float> %292, i64 1, !dbg !22
  %367 = tail call float @llvm.amdgcn.exp2.f32(float %366), !dbg !22
  %368 = insertelement <4 x float> %365, float %367, i64 1, !dbg !23
  %local.reduce59 = fadd float %367, %local.reduce55, !dbg !24, !frisk.local_reduction !5
  %369 = extractelement <4 x float> %292, i64 2, !dbg !22
  %370 = tail call float @llvm.amdgcn.exp2.f32(float %369), !dbg !22
  %371 = insertelement <4 x float> %368, float %370, i64 2, !dbg !23
  %local.reduce60 = fadd float %370, %local.reduce56, !dbg !24, !frisk.local_reduction !5
  %372 = extractelement <4 x float> %292, i64 3, !dbg !22
  %373 = tail call float @llvm.amdgcn.exp2.f32(float %372), !dbg !22
  %374 = insertelement <4 x float> %371, float %373, i64 3, !dbg !23
  %math.result296 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %374) #8, !dbg !23, !frisk.register_lifetimes !5
  %375 = fptrunc <4 x float> %math.result296 to <4 x half>, !dbg !23
  %math.packed297 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %375) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce61 = fadd float %373, %local.reduce57, !dbg !24, !frisk.local_reduction !5
  %376 = extractelement <4 x float> %293, i64 0, !dbg !22
  %377 = tail call float @llvm.amdgcn.exp2.f32(float %376), !dbg !22
  %378 = insertelement <4 x float> poison, float %377, i64 0, !dbg !23
  %local.reduce62 = fadd float %377, %local.reduce58, !dbg !24, !frisk.local_reduction !5
  %379 = extractelement <4 x float> %293, i64 1, !dbg !22
  %380 = tail call float @llvm.amdgcn.exp2.f32(float %379), !dbg !22
  %381 = insertelement <4 x float> %378, float %380, i64 1, !dbg !23
  %local.reduce63 = fadd float %380, %local.reduce59, !dbg !24, !frisk.local_reduction !5
  %382 = extractelement <4 x float> %293, i64 2, !dbg !22
  %383 = tail call float @llvm.amdgcn.exp2.f32(float %382), !dbg !22
  %384 = insertelement <4 x float> %381, float %383, i64 2, !dbg !23
  %local.reduce64 = fadd float %383, %local.reduce60, !dbg !24, !frisk.local_reduction !5
  %385 = extractelement <4 x float> %293, i64 3, !dbg !22
  %386 = tail call float @llvm.amdgcn.exp2.f32(float %385), !dbg !22
  %387 = insertelement <4 x float> %384, float %386, i64 3, !dbg !23
  %math.result298 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %387) #8, !dbg !23, !frisk.register_lifetimes !5
  %388 = fptrunc <4 x float> %math.result298 to <4 x half>, !dbg !23
  %math.packed299 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %388) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce65 = fadd float %386, %local.reduce61, !dbg !24, !frisk.local_reduction !5
  %389 = extractelement <4 x float> %294, i64 0, !dbg !22
  %390 = tail call float @llvm.amdgcn.exp2.f32(float %389), !dbg !22
  %391 = insertelement <4 x float> poison, float %390, i64 0, !dbg !23
  %local.reduce66 = fadd float %390, %local.reduce62, !dbg !24, !frisk.local_reduction !5
  %392 = extractelement <4 x float> %294, i64 1, !dbg !22
  %393 = tail call float @llvm.amdgcn.exp2.f32(float %392), !dbg !22
  %394 = insertelement <4 x float> %391, float %393, i64 1, !dbg !23
  %local.reduce67 = fadd float %393, %local.reduce63, !dbg !24, !frisk.local_reduction !5
  %395 = extractelement <4 x float> %294, i64 2, !dbg !22
  %396 = tail call float @llvm.amdgcn.exp2.f32(float %395), !dbg !22
  %397 = insertelement <4 x float> %394, float %396, i64 2, !dbg !23
  %local.reduce68 = fadd float %396, %local.reduce64, !dbg !24, !frisk.local_reduction !5
  %398 = extractelement <4 x float> %294, i64 3, !dbg !22
  %399 = tail call float @llvm.amdgcn.exp2.f32(float %398), !dbg !22
  %400 = insertelement <4 x float> %397, float %399, i64 3, !dbg !23
  %math.result300 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %400) #8, !dbg !23, !frisk.register_lifetimes !5
  %401 = fptrunc <4 x float> %math.result300 to <4 x half>, !dbg !23
  %math.packed301 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %401) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce69 = fadd float %399, %local.reduce65, !dbg !24, !frisk.local_reduction !5
  %402 = extractelement <4 x float> %295, i64 0, !dbg !22
  %403 = tail call float @llvm.amdgcn.exp2.f32(float %402), !dbg !22
  %404 = insertelement <4 x float> poison, float %403, i64 0, !dbg !23
  %local.reduce70 = fadd float %403, %local.reduce66, !dbg !24, !frisk.local_reduction !5
  %405 = extractelement <4 x float> %295, i64 1, !dbg !22
  %406 = tail call float @llvm.amdgcn.exp2.f32(float %405), !dbg !22
  %407 = insertelement <4 x float> %404, float %406, i64 1, !dbg !23
  %local.reduce71 = fadd float %406, %local.reduce67, !dbg !24, !frisk.local_reduction !5
  %local.reduce74 = fadd float %local.reduce70, %local.reduce71, !dbg !24, !frisk.local_reduction !5
  %408 = extractelement <4 x float> %295, i64 2, !dbg !22
  %409 = tail call float @llvm.amdgcn.exp2.f32(float %408), !dbg !22
  %410 = insertelement <4 x float> %407, float %409, i64 2, !dbg !23
  %local.reduce72 = fadd float %409, %local.reduce68, !dbg !24, !frisk.local_reduction !5
  %411 = extractelement <4 x float> %295, i64 3, !dbg !22
  %412 = tail call float @llvm.amdgcn.exp2.f32(float %411), !dbg !22
  %413 = insertelement <4 x float> %410, float %412, i64 3, !dbg !23
  %math.result302 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %413) #8, !dbg !23, !frisk.register_lifetimes !5
  %414 = fptrunc <4 x float> %math.result302 to <4 x half>, !dbg !23
  %math.packed303 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %414) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce73 = fadd float %412, %local.reduce69, !dbg !24, !frisk.local_reduction !5
  %local.reduce75 = fadd float %local.reduce72, %local.reduce73, !dbg !24, !frisk.local_reduction !5
  %415 = fadd float %local.reduce74, %local.reduce75, !dbg !24, !frisk.local_reduction !5
  %416 = fadd float %310, %415, !dbg !24
  %417 = insertelement <1 x float> poison, float %416, i64 0, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local752 = shl nuw nsw i32 %address.thread350, 3, !dbg !25
  %address.local753 = and i32 %address.local752, 120, !dbg !25
  %address.local755 = and i32 %address.local406, 1536, !dbg !25
  %address.local756 = or disjoint i32 %address.local753, %address.local755, !dbg !25
  %address.local757 = zext nneg i32 %address.local756 to i64, !dbg !25
  %address.local758 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local757, !dbg !25
  %418 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local758, i32 0) #10, !dbg !25
  %419 = bitcast <4 x i32> %418 to <8 x half>, !dbg !25
  %address.local759 = and i64 %address.local351, 60, !dbg !25
  %address.local760 = add nuw nsw i64 %address.local759, %address.local351, !dbg !25
  %address.local762.idx = shl nuw nsw i64 %address.local760, 4, !dbg !25
  %address.local762 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local762.idx, !dbg !25
  %address.local763 = lshr i64 %address.local386, 4, !dbg !25
  %address.local765.idx = mul nuw nsw i64 %address.local763, 768, !dbg !25
  %address.local765 = getelementptr i8, ptr addrspace(3) %address.local762, i64 %address.local765.idx, !dbg !25
  %address.local767.idx = mul nsw i64 %address.local360, -64, !dbg !25
  %address.local767 = getelementptr i8, ptr addrspace(3) %address.local765, i64 %address.local767.idx, !dbg !25
  %address.local768 = getelementptr i8, ptr addrspace(3) %address.local767, i64 256, !dbg !25
  %420 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local768, i32 0) #10, !dbg !25
  %421 = bitcast <4 x i32> %420 to <8 x half>, !dbg !25
  %address.local772 = getelementptr i8, ptr addrspace(3) %address.local767, i64 512, !dbg !25
  %422 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local772, i32 0) #10, !dbg !25
  %423 = bitcast <4 x i32> %422 to <8 x half>, !dbg !25
  %address.local776 = getelementptr i8, ptr addrspace(3) %address.local767, i64 768, !dbg !25
  %424 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local776, i32 0) #10, !dbg !25
  %425 = bitcast <4 x i32> %424 to <8 x half>, !dbg !25
  %address.local778 = getelementptr i8, ptr addrspace(3) %address.local758, i64 4096, !dbg !25
  %426 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #10, !dbg !25
  %427 = bitcast <4 x i32> %426 to <8 x half>, !dbg !25
  %address.local782 = getelementptr i8, ptr addrspace(3) %address.local767, i64 4352, !dbg !25
  %428 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #10, !dbg !25
  %429 = bitcast <4 x i32> %428 to <8 x half>, !dbg !25
  %address.local786 = getelementptr i8, ptr addrspace(3) %address.local767, i64 4608, !dbg !25
  %430 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local786, i32 0) #10, !dbg !25
  %431 = bitcast <4 x i32> %430 to <8 x half>, !dbg !25
  %address.local790 = getelementptr i8, ptr addrspace(3) %address.local767, i64 4864, !dbg !25
  %432 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local790, i32 0) #10, !dbg !25
  %433 = bitcast <4 x i32> %432 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx10511052 = or disjoint i64 %address.local750, %87, !dbg !14
  %.idx10511052.tr = trunc nuw nsw i64 %.idx10511052 to i32, !dbg !14
  %dma.buffer.offset1009 = shl nuw nsw i32 %.idx10511052.tr, 1, !dbg !14
  %434 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1026, i32 %dma.buffer.offset1009, <4 x i32> %dma.buffer.rsrc1013) #9, !dbg !14
  %435 = shufflevector <8 x half> %419, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %436 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %435, <4 x float> %308) #8, !dbg !25
  %437 = shufflevector <8 x half> %421, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %437, <4 x float> %436) #8, !dbg !25
  %439 = shufflevector <8 x half> %423, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %439, <4 x float> %438) #8, !dbg !25
  %441 = shufflevector <8 x half> %425, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %441, <4 x float> %440) #8, !dbg !25
  %443 = shufflevector <8 x half> %427, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %444 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %443, <4 x float> %442) #8, !dbg !25
  %445 = shufflevector <8 x half> %429, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %445, <4 x float> %444) #8, !dbg !25
  %447 = shufflevector <8 x half> %431, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %447, <4 x float> %446) #8, !dbg !25
  %449 = shufflevector <8 x half> %433, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %449, <4 x float> %448) #8, !dbg !25
  %451 = shufflevector <8 x half> %419, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %452 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %451, <4 x float> %307) #8, !dbg !25
  %453 = shufflevector <8 x half> %421, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %453, <4 x float> %452) #8, !dbg !25
  %455 = shufflevector <8 x half> %423, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %456 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %455, <4 x float> %454) #8, !dbg !25
  %457 = shufflevector <8 x half> %425, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %458 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %457, <4 x float> %456) #8, !dbg !25
  %459 = shufflevector <8 x half> %427, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %460 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %459, <4 x float> %458) #8, !dbg !25
  %461 = shufflevector <8 x half> %429, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %461, <4 x float> %460) #8, !dbg !25
  %463 = shufflevector <8 x half> %431, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %463, <4 x float> %462) #8, !dbg !25
  %465 = shufflevector <8 x half> %433, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %466 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %465, <4 x float> %464) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %467 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local758, i32 0) #10, !dbg !25
  %468 = bitcast <4 x i32> %467 to <8 x half>, !dbg !25
  %469 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local768, i32 0) #10, !dbg !25
  %470 = bitcast <4 x i32> %469 to <8 x half>, !dbg !25
  %471 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local772, i32 0) #10, !dbg !25
  %472 = bitcast <4 x i32> %471 to <8 x half>, !dbg !25
  %473 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local776, i32 0) #10, !dbg !25
  %474 = bitcast <4 x i32> %473 to <8 x half>, !dbg !25
  %475 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #10, !dbg !25
  %476 = bitcast <4 x i32> %475 to <8 x half>, !dbg !25
  %477 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #10, !dbg !25
  %478 = bitcast <4 x i32> %477 to <8 x half>, !dbg !25
  %479 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local786, i32 0) #10, !dbg !25
  %480 = bitcast <4 x i32> %479 to <8 x half>, !dbg !25
  %481 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local790, i32 0) #10, !dbg !25
  %482 = bitcast <4 x i32> %481 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx10531054 = or disjoint i64 %address.local750, %130, !dbg !14
  %.idx10531054.tr = trunc nuw nsw i64 %.idx10531054 to i32, !dbg !14
  %dma.buffer.offset1006 = shl nuw nsw i32 %.idx10531054.tr, 1, !dbg !14
  %483 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1026, i32 %dma.buffer.offset1006, <4 x i32> %dma.buffer.rsrc1013) #9, !dbg !14
  %484 = shufflevector <8 x half> %468, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %485 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %484, <4 x float> %306) #8, !dbg !25
  %486 = shufflevector <8 x half> %470, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %487 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %486, <4 x float> %485) #8, !dbg !25
  %488 = shufflevector <8 x half> %472, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %489 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %488, <4 x float> %487) #8, !dbg !25
  %490 = shufflevector <8 x half> %474, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %491 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %490, <4 x float> %489) #8, !dbg !25
  %492 = shufflevector <8 x half> %476, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %493 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %492, <4 x float> %491) #8, !dbg !25
  %494 = shufflevector <8 x half> %478, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %495 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %494, <4 x float> %493) #8, !dbg !25
  %496 = shufflevector <8 x half> %480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %497 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %496, <4 x float> %495) #8, !dbg !25
  %498 = shufflevector <8 x half> %482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %499 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %498, <4 x float> %497) #8, !dbg !25
  %500 = shufflevector <8 x half> %468, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %501 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %500, <4 x float> %305) #8, !dbg !25
  %502 = shufflevector <8 x half> %470, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %503 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %502, <4 x float> %501) #8, !dbg !25
  %504 = shufflevector <8 x half> %472, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %505 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %504, <4 x float> %503) #8, !dbg !25
  %506 = shufflevector <8 x half> %474, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %507 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %506, <4 x float> %505) #8, !dbg !25
  %508 = shufflevector <8 x half> %476, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %509 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %508, <4 x float> %507) #8, !dbg !25
  %510 = shufflevector <8 x half> %478, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %511 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %510, <4 x float> %509) #8, !dbg !25
  %512 = shufflevector <8 x half> %480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %513 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %512, <4 x float> %511) #8, !dbg !25
  %514 = shufflevector <8 x half> %482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %515 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %514, <4 x float> %513) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %516 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local758, i32 0) #10, !dbg !25
  %517 = bitcast <4 x i32> %516 to <8 x half>, !dbg !25
  %518 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local768, i32 0) #10, !dbg !25
  %519 = bitcast <4 x i32> %518 to <8 x half>, !dbg !25
  %520 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local772, i32 0) #10, !dbg !25
  %521 = bitcast <4 x i32> %520 to <8 x half>, !dbg !25
  %522 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local776, i32 0) #10, !dbg !25
  %523 = bitcast <4 x i32> %522 to <8 x half>, !dbg !25
  %524 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #10, !dbg !25
  %525 = bitcast <4 x i32> %524 to <8 x half>, !dbg !25
  %526 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #10, !dbg !25
  %527 = bitcast <4 x i32> %526 to <8 x half>, !dbg !25
  %528 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local786, i32 0) #10, !dbg !25
  %529 = bitcast <4 x i32> %528 to <8 x half>, !dbg !25
  %530 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local790, i32 0) #10, !dbg !25
  %531 = bitcast <4 x i32> %530 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx10551056 = or disjoint i64 %address.local750, %164, !dbg !14
  %.idx10551056.tr = trunc nuw nsw i64 %.idx10551056 to i32, !dbg !14
  %dma.buffer.offset = shl nuw nsw i32 %.idx10551056.tr, 1, !dbg !14
  %532 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1026, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1013) #9, !dbg !14
  %533 = shufflevector <8 x half> %517, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %534 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %533, <4 x float> %304) #8, !dbg !25
  %535 = shufflevector <8 x half> %519, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %536 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %535, <4 x float> %534) #8, !dbg !25
  %537 = shufflevector <8 x half> %521, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %538 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %537, <4 x float> %536) #8, !dbg !25
  %539 = shufflevector <8 x half> %523, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %540 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %539, <4 x float> %538) #8, !dbg !25
  %541 = shufflevector <8 x half> %525, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %541, <4 x float> %540) #8, !dbg !25
  %543 = shufflevector <8 x half> %527, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %544 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %543, <4 x float> %542) #8, !dbg !25
  %545 = shufflevector <8 x half> %529, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %546 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %545, <4 x float> %544) #8, !dbg !25
  %547 = shufflevector <8 x half> %531, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %548 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %547, <4 x float> %546) #8, !dbg !25
  %549 = shufflevector <8 x half> %517, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %550 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %549, <4 x float> %303) #8, !dbg !25
  %551 = shufflevector <8 x half> %519, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %552 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %551, <4 x float> %550) #8, !dbg !25
  %553 = shufflevector <8 x half> %521, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %554 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %553, <4 x float> %552) #8, !dbg !25
  %555 = shufflevector <8 x half> %523, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %556 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %555, <4 x float> %554) #8, !dbg !25
  %557 = shufflevector <8 x half> %525, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %558 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %557, <4 x float> %556) #8, !dbg !25
  %559 = shufflevector <8 x half> %527, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %560 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %559, <4 x float> %558) #8, !dbg !25
  %561 = shufflevector <8 x half> %529, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %562 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %561, <4 x float> %560) #8, !dbg !25
  %563 = shufflevector <8 x half> %531, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %564 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %563, <4 x float> %562) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %565 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local758, i32 0) #10, !dbg !25
  %566 = bitcast <4 x i32> %565 to <8 x half>, !dbg !25
  %567 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local768, i32 0) #10, !dbg !25
  %568 = bitcast <4 x i32> %567 to <8 x half>, !dbg !25
  %569 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local772, i32 0) #10, !dbg !25
  %570 = bitcast <4 x i32> %569 to <8 x half>, !dbg !25
  %571 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local776, i32 0) #10, !dbg !25
  %572 = bitcast <4 x i32> %571 to <8 x half>, !dbg !25
  %573 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #10, !dbg !25
  %574 = bitcast <4 x i32> %573 to <8 x half>, !dbg !25
  %575 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #10, !dbg !25
  %576 = bitcast <4 x i32> %575 to <8 x half>, !dbg !25
  %577 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local786, i32 0) #10, !dbg !25
  %578 = bitcast <4 x i32> %577 to <8 x half>, !dbg !25
  %579 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local790, i32 0) #10, !dbg !25
  %580 = bitcast <4 x i32> %579 to <8 x half>, !dbg !25
  %581 = shufflevector <8 x half> %566, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %582 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %581, <4 x float> %302) #8, !dbg !25
  %583 = shufflevector <8 x half> %568, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %584 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %583, <4 x float> %582) #8, !dbg !25
  %585 = shufflevector <8 x half> %570, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %586 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %585, <4 x float> %584) #8, !dbg !25
  %587 = shufflevector <8 x half> %572, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %587, <4 x float> %586) #8, !dbg !25
  %589 = shufflevector <8 x half> %574, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %589, <4 x float> %588) #8, !dbg !25
  %591 = shufflevector <8 x half> %576, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %591, <4 x float> %590) #8, !dbg !25
  %593 = shufflevector <8 x half> %578, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %594 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %593, <4 x float> %592) #8, !dbg !25
  %595 = shufflevector <8 x half> %580, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %596 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %595, <4 x float> %594) #8, !dbg !25
  %597 = shufflevector <8 x half> %566, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %598 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %597, <4 x float> %301) #8, !dbg !25
  %599 = shufflevector <8 x half> %568, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %600 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %599, <4 x float> %598) #8, !dbg !25
  %601 = shufflevector <8 x half> %570, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %602 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %601, <4 x float> %600) #8, !dbg !25
  %603 = shufflevector <8 x half> %572, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %604 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %603, <4 x float> %602) #8, !dbg !25
  %605 = shufflevector <8 x half> %574, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %606 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %605, <4 x float> %604) #8, !dbg !25
  %607 = shufflevector <8 x half> %576, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %608 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %607, <4 x float> %606) #8, !dbg !25
  %609 = shufflevector <8 x half> %578, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %610 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %609, <4 x float> %608) #8, !dbg !25
  %611 = shufflevector <8 x half> %580, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %612 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %611, <4 x float> %610) #8, !dbg !25
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %613 = add nuw nsw i64 %82, 1, !dbg !11
  %614 = icmp samesign ult i64 %82, 31, !dbg !11
  br i1 %614, label %71, label %615, !dbg !11

615:                                              ; preds = %71
  %616 = shl nuw nsw i32 %266, 2, !dbg !11
  %617 = xor i32 %616, 64, !dbg !11
  %618 = bitcast float %416 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %617, i32 %618), !dbg !11
  %619 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %416, %619, !dbg !11
  %620 = xor i32 %616, 128, !dbg !11
  %621 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle108 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %620, i32 %621), !dbg !11
  %622 = bitcast i32 %lane.sum.shuffle108 to float, !dbg !11
  %lane.sum.final109 = fadd float %lane.sum.final, %622, !dbg !11
  %623 = insertelement <1 x float> poison, float %lane.sum.final109, i64 0, !dbg !11
  %624 = and i32 %31, 16, !dbg !11
  %.not = icmp eq i32 %624, 0, !dbg !11
  %625 = and i32 %31, 32, !dbg !11
  %.not36 = icmp eq i32 %625, 0, !dbg !11
  %626 = extractelement <4 x float> %596, i64 0, !dbg !11
  %627 = extractelement <4 x float> %596, i64 1, !dbg !11
  %628 = extractelement <4 x float> %596, i64 2, !dbg !11
  %629 = extractelement <4 x float> %596, i64 3, !dbg !11
  %630 = select i1 %.not, float %627, float %626, !dbg !11
  %631 = bitcast float %630 to i32, !dbg !11
  %632 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %631), !dbg !11
  %633 = bitcast i32 %632 to float, !dbg !11
  %634 = select i1 %.not, float %626, float %633, !dbg !11
  %635 = select i1 %.not, float %633, float %627, !dbg !11
  %636 = select i1 %.not, float %629, float %628, !dbg !11
  %637 = bitcast float %636 to i32, !dbg !11
  %638 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %637), !dbg !11
  %639 = bitcast i32 %638 to float, !dbg !11
  %640 = select i1 %.not, float %628, float %639, !dbg !11
  %641 = select i1 %.not, float %639, float %629, !dbg !11
  %642 = select i1 %.not36, float %640, float %634, !dbg !11
  %643 = bitcast float %642 to i32, !dbg !11
  %644 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %643), !dbg !11
  %645 = bitcast i32 %644 to float, !dbg !11
  %646 = select i1 %.not36, float %641, float %635, !dbg !11
  %647 = bitcast float %646 to i32, !dbg !11
  %648 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %647), !dbg !11
  %649 = bitcast i32 %648 to float, !dbg !11
  %650 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %651 = shufflevector <4 x i1> %650, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %652 = insertelement <4 x float> poison, float %634, i64 0, !dbg !11
  %653 = insertelement <4 x float> %652, float %635, i64 1, !dbg !11
  %654 = insertelement <4 x float> %653, float %645, i64 2, !dbg !11
  %655 = insertelement <4 x float> %654, float %649, i64 3, !dbg !11
  %656 = insertelement <4 x float> poison, float %645, i64 0, !dbg !11
  %657 = insertelement <4 x float> %656, float %649, i64 1, !dbg !11
  %658 = insertelement <4 x float> %657, float %640, i64 2, !dbg !11
  %659 = insertelement <4 x float> %658, float %641, i64 3, !dbg !11
  %660 = select <4 x i1> %651, <4 x float> %655, <4 x float> %659, !dbg !11
  %661 = extractelement <4 x float> %548, i64 0, !dbg !11
  %662 = extractelement <4 x float> %548, i64 1, !dbg !11
  %663 = extractelement <4 x float> %548, i64 2, !dbg !11
  %664 = extractelement <4 x float> %548, i64 3, !dbg !11
  %665 = select i1 %.not, float %662, float %661, !dbg !11
  %666 = bitcast float %665 to i32, !dbg !11
  %667 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %666), !dbg !11
  %668 = bitcast i32 %667 to float, !dbg !11
  %669 = select i1 %.not, float %661, float %668, !dbg !11
  %670 = select i1 %.not, float %668, float %662, !dbg !11
  %671 = select i1 %.not, float %664, float %663, !dbg !11
  %672 = bitcast float %671 to i32, !dbg !11
  %673 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %672), !dbg !11
  %674 = bitcast i32 %673 to float, !dbg !11
  %675 = select i1 %.not, float %663, float %674, !dbg !11
  %676 = select i1 %.not, float %674, float %664, !dbg !11
  %677 = select i1 %.not36, float %675, float %669, !dbg !11
  %678 = bitcast float %677 to i32, !dbg !11
  %679 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %678), !dbg !11
  %680 = bitcast i32 %679 to float, !dbg !11
  %681 = select i1 %.not36, float %676, float %670, !dbg !11
  %682 = bitcast float %681 to i32, !dbg !11
  %683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %682), !dbg !11
  %684 = bitcast i32 %683 to float, !dbg !11
  %685 = insertelement <4 x float> poison, float %669, i64 0, !dbg !11
  %686 = insertelement <4 x float> %685, float %670, i64 1, !dbg !11
  %687 = insertelement <4 x float> %686, float %680, i64 2, !dbg !11
  %688 = insertelement <4 x float> %687, float %684, i64 3, !dbg !11
  %689 = insertelement <4 x float> poison, float %680, i64 0, !dbg !11
  %690 = insertelement <4 x float> %689, float %684, i64 1, !dbg !11
  %691 = insertelement <4 x float> %690, float %675, i64 2, !dbg !11
  %692 = insertelement <4 x float> %691, float %676, i64 3, !dbg !11
  %693 = select <4 x i1> %651, <4 x float> %688, <4 x float> %692, !dbg !11
  %694 = extractelement <4 x float> %499, i64 0, !dbg !11
  %695 = extractelement <4 x float> %499, i64 1, !dbg !11
  %696 = extractelement <4 x float> %499, i64 2, !dbg !11
  %697 = extractelement <4 x float> %499, i64 3, !dbg !11
  %698 = select i1 %.not, float %695, float %694, !dbg !11
  %699 = bitcast float %698 to i32, !dbg !11
  %700 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %699), !dbg !11
  %701 = bitcast i32 %700 to float, !dbg !11
  %702 = select i1 %.not, float %694, float %701, !dbg !11
  %703 = select i1 %.not, float %701, float %695, !dbg !11
  %704 = select i1 %.not, float %697, float %696, !dbg !11
  %705 = bitcast float %704 to i32, !dbg !11
  %706 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %705), !dbg !11
  %707 = bitcast i32 %706 to float, !dbg !11
  %708 = select i1 %.not, float %696, float %707, !dbg !11
  %709 = select i1 %.not, float %707, float %697, !dbg !11
  %710 = select i1 %.not36, float %708, float %702, !dbg !11
  %711 = bitcast float %710 to i32, !dbg !11
  %712 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %711), !dbg !11
  %713 = bitcast i32 %712 to float, !dbg !11
  %714 = select i1 %.not36, float %709, float %703, !dbg !11
  %715 = bitcast float %714 to i32, !dbg !11
  %716 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %715), !dbg !11
  %717 = bitcast i32 %716 to float, !dbg !11
  %718 = insertelement <4 x float> poison, float %702, i64 0, !dbg !11
  %719 = insertelement <4 x float> %718, float %703, i64 1, !dbg !11
  %720 = insertelement <4 x float> %719, float %713, i64 2, !dbg !11
  %721 = insertelement <4 x float> %720, float %717, i64 3, !dbg !11
  %722 = insertelement <4 x float> poison, float %713, i64 0, !dbg !11
  %723 = insertelement <4 x float> %722, float %717, i64 1, !dbg !11
  %724 = insertelement <4 x float> %723, float %708, i64 2, !dbg !11
  %725 = insertelement <4 x float> %724, float %709, i64 3, !dbg !11
  %726 = select <4 x i1> %651, <4 x float> %721, <4 x float> %725, !dbg !11
  %727 = extractelement <4 x float> %450, i64 0, !dbg !11
  %728 = extractelement <4 x float> %450, i64 1, !dbg !11
  %729 = extractelement <4 x float> %450, i64 2, !dbg !11
  %730 = extractelement <4 x float> %450, i64 3, !dbg !11
  %731 = select i1 %.not, float %728, float %727, !dbg !11
  %732 = bitcast float %731 to i32, !dbg !11
  %733 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %732), !dbg !11
  %734 = bitcast i32 %733 to float, !dbg !11
  %735 = select i1 %.not, float %727, float %734, !dbg !11
  %736 = select i1 %.not, float %734, float %728, !dbg !11
  %737 = select i1 %.not, float %730, float %729, !dbg !11
  %738 = bitcast float %737 to i32, !dbg !11
  %739 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %738), !dbg !11
  %740 = bitcast i32 %739 to float, !dbg !11
  %741 = select i1 %.not, float %729, float %740, !dbg !11
  %742 = select i1 %.not, float %740, float %730, !dbg !11
  %743 = select i1 %.not36, float %741, float %735, !dbg !11
  %744 = bitcast float %743 to i32, !dbg !11
  %745 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %744), !dbg !11
  %746 = select i1 %.not36, float %742, float %736, !dbg !11
  %747 = bitcast float %746 to i32, !dbg !11
  %748 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %747), !dbg !11
  %749 = shufflevector <1 x float> %623, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %750 = extractelement <4 x float> %612, i64 0, !dbg !11
  %751 = extractelement <4 x float> %612, i64 1, !dbg !11
  %752 = extractelement <4 x float> %612, i64 2, !dbg !11
  %753 = extractelement <4 x float> %612, i64 3, !dbg !11
  %754 = select i1 %.not, float %751, float %750, !dbg !11
  %755 = bitcast float %754 to i32, !dbg !11
  %756 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %755), !dbg !11
  %757 = bitcast i32 %756 to float, !dbg !11
  %758 = select i1 %.not, float %750, float %757, !dbg !11
  %759 = select i1 %.not, float %757, float %751, !dbg !11
  %760 = select i1 %.not, float %753, float %752, !dbg !11
  %761 = bitcast float %760 to i32, !dbg !11
  %762 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %761), !dbg !11
  %763 = bitcast i32 %762 to float, !dbg !11
  %764 = select i1 %.not, float %752, float %763, !dbg !11
  %765 = select i1 %.not, float %763, float %753, !dbg !11
  %766 = select i1 %.not36, float %764, float %758, !dbg !11
  %767 = bitcast float %766 to i32, !dbg !11
  %768 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %767), !dbg !11
  %769 = bitcast i32 %768 to float, !dbg !11
  %770 = select i1 %.not36, float %765, float %759, !dbg !11
  %771 = bitcast float %770 to i32, !dbg !11
  %772 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %771), !dbg !11
  %773 = bitcast i32 %772 to float, !dbg !11
  %774 = insertelement <4 x float> poison, float %758, i64 0, !dbg !11
  %775 = insertelement <4 x float> %774, float %759, i64 1, !dbg !11
  %776 = insertelement <4 x float> %775, float %769, i64 2, !dbg !11
  %777 = insertelement <4 x float> %776, float %773, i64 3, !dbg !11
  %778 = insertelement <4 x float> poison, float %769, i64 0, !dbg !11
  %779 = insertelement <4 x float> %778, float %773, i64 1, !dbg !11
  %780 = insertelement <4 x float> %779, float %764, i64 2, !dbg !11
  %781 = insertelement <4 x float> %780, float %765, i64 3, !dbg !11
  %782 = select <4 x i1> %651, <4 x float> %777, <4 x float> %781, !dbg !11
  %783 = extractelement <4 x float> %564, i64 0, !dbg !11
  %784 = extractelement <4 x float> %564, i64 1, !dbg !11
  %785 = extractelement <4 x float> %564, i64 2, !dbg !11
  %786 = extractelement <4 x float> %564, i64 3, !dbg !11
  %787 = select i1 %.not, float %784, float %783, !dbg !11
  %788 = bitcast float %787 to i32, !dbg !11
  %789 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %788), !dbg !11
  %790 = bitcast i32 %789 to float, !dbg !11
  %791 = select i1 %.not, float %783, float %790, !dbg !11
  %792 = select i1 %.not, float %790, float %784, !dbg !11
  %793 = select i1 %.not, float %786, float %785, !dbg !11
  %794 = bitcast float %793 to i32, !dbg !11
  %795 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %794), !dbg !11
  %796 = bitcast i32 %795 to float, !dbg !11
  %797 = select i1 %.not, float %785, float %796, !dbg !11
  %798 = select i1 %.not, float %796, float %786, !dbg !11
  %799 = select i1 %.not36, float %797, float %791, !dbg !11
  %800 = bitcast float %799 to i32, !dbg !11
  %801 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %800), !dbg !11
  %802 = bitcast i32 %801 to float, !dbg !11
  %803 = select i1 %.not36, float %798, float %792, !dbg !11
  %804 = bitcast float %803 to i32, !dbg !11
  %805 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %804), !dbg !11
  %806 = bitcast i32 %805 to float, !dbg !11
  %807 = insertelement <4 x float> poison, float %791, i64 0, !dbg !11
  %808 = insertelement <4 x float> %807, float %792, i64 1, !dbg !11
  %809 = insertelement <4 x float> %808, float %802, i64 2, !dbg !11
  %810 = insertelement <4 x float> %809, float %806, i64 3, !dbg !11
  %811 = insertelement <4 x float> poison, float %802, i64 0, !dbg !11
  %812 = insertelement <4 x float> %811, float %806, i64 1, !dbg !11
  %813 = insertelement <4 x float> %812, float %797, i64 2, !dbg !11
  %814 = insertelement <4 x float> %813, float %798, i64 3, !dbg !11
  %815 = select <4 x i1> %651, <4 x float> %810, <4 x float> %814, !dbg !11
  %816 = extractelement <4 x float> %515, i64 0, !dbg !11
  %817 = extractelement <4 x float> %515, i64 1, !dbg !11
  %818 = extractelement <4 x float> %515, i64 2, !dbg !11
  %819 = extractelement <4 x float> %515, i64 3, !dbg !11
  %820 = select i1 %.not, float %817, float %816, !dbg !11
  %821 = bitcast float %820 to i32, !dbg !11
  %822 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %821), !dbg !11
  %823 = bitcast i32 %822 to float, !dbg !11
  %824 = select i1 %.not, float %816, float %823, !dbg !11
  %825 = select i1 %.not, float %823, float %817, !dbg !11
  %826 = select i1 %.not, float %819, float %818, !dbg !11
  %827 = bitcast float %826 to i32, !dbg !11
  %828 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %827), !dbg !11
  %829 = bitcast i32 %828 to float, !dbg !11
  %830 = select i1 %.not, float %818, float %829, !dbg !11
  %831 = select i1 %.not, float %829, float %819, !dbg !11
  %832 = select i1 %.not36, float %830, float %824, !dbg !11
  %833 = bitcast float %832 to i32, !dbg !11
  %834 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %833), !dbg !11
  %835 = bitcast i32 %834 to float, !dbg !11
  %836 = select i1 %.not36, float %831, float %825, !dbg !11
  %837 = bitcast float %836 to i32, !dbg !11
  %838 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %837), !dbg !11
  %839 = bitcast i32 %838 to float, !dbg !11
  %840 = insertelement <4 x float> poison, float %824, i64 0, !dbg !11
  %841 = insertelement <4 x float> %840, float %825, i64 1, !dbg !11
  %842 = insertelement <4 x float> %841, float %835, i64 2, !dbg !11
  %843 = insertelement <4 x float> %842, float %839, i64 3, !dbg !11
  %844 = insertelement <4 x float> poison, float %835, i64 0, !dbg !11
  %845 = insertelement <4 x float> %844, float %839, i64 1, !dbg !11
  %846 = insertelement <4 x float> %845, float %830, i64 2, !dbg !11
  %847 = insertelement <4 x float> %846, float %831, i64 3, !dbg !11
  %848 = select <4 x i1> %651, <4 x float> %843, <4 x float> %847, !dbg !11
  %849 = extractelement <4 x float> %466, i64 0, !dbg !11
  %850 = extractelement <4 x float> %466, i64 1, !dbg !11
  %851 = extractelement <4 x float> %466, i64 2, !dbg !11
  %852 = extractelement <4 x float> %466, i64 3, !dbg !11
  %853 = select i1 %.not, float %850, float %849, !dbg !11
  %854 = bitcast float %853 to i32, !dbg !11
  %855 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %854), !dbg !11
  %856 = bitcast i32 %855 to float, !dbg !11
  %857 = select i1 %.not, float %849, float %856, !dbg !11
  %858 = select i1 %.not, float %856, float %850, !dbg !11
  %859 = select i1 %.not, float %852, float %851, !dbg !11
  %860 = bitcast float %859 to i32, !dbg !11
  %861 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %860), !dbg !11
  %862 = bitcast i32 %861 to float, !dbg !11
  %863 = select i1 %.not, float %851, float %862, !dbg !11
  %864 = select i1 %.not, float %862, float %852, !dbg !11
  %865 = select i1 %.not36, float %863, float %857, !dbg !11
  %866 = bitcast float %865 to i32, !dbg !11
  %867 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %866), !dbg !11
  %868 = select i1 %.not36, float %864, float %858, !dbg !11
  %869 = bitcast float %868 to i32, !dbg !11
  %870 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %280, i32 %869), !dbg !11
  %871 = bitcast i32 %867 to float, !dbg !11
  %872 = bitcast i32 %870 to float, !dbg !11
  %873 = insertelement <4 x float> poison, float %857, i64 0, !dbg !11
  %874 = insertelement <4 x float> %873, float %858, i64 1, !dbg !11
  %875 = insertelement <4 x float> %874, float %871, i64 2, !dbg !11
  %876 = insertelement <4 x float> %875, float %872, i64 3, !dbg !11
  %877 = insertelement <4 x float> poison, float %871, i64 0, !dbg !11
  %878 = insertelement <4 x float> %877, float %872, i64 1, !dbg !11
  %879 = insertelement <4 x float> %878, float %863, i64 2, !dbg !11
  %880 = insertelement <4 x float> %879, float %864, i64 3, !dbg !11
  %881 = select <4 x i1> %651, <4 x float> %876, <4 x float> %880, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %749, !dbg !26
  %normalized = fmul <4 x float> %row.reciprocal, %881, !dbg !26
  %882 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !27
  %883 = extractelement <4 x half> %882, i64 3, !dbg !27
  %884 = extractelement <4 x half> %882, i64 2, !dbg !27
  %885 = extractelement <4 x half> %882, i64 1, !dbg !27
  %886 = extractelement <4 x half> %882, i64 0, !dbg !27
  %887 = bitcast i32 %745 to float, !dbg !11
  %888 = bitcast i32 %748 to float, !dbg !11
  %889 = insertelement <4 x float> poison, float %735, i64 0, !dbg !11
  %890 = insertelement <4 x float> %889, float %736, i64 1, !dbg !11
  %891 = insertelement <4 x float> %890, float %887, i64 2, !dbg !11
  %892 = insertelement <4 x float> %891, float %888, i64 3, !dbg !11
  %893 = insertelement <4 x float> poison, float %887, i64 0, !dbg !11
  %894 = insertelement <4 x float> %893, float %888, i64 1, !dbg !11
  %895 = insertelement <4 x float> %894, float %741, i64 2, !dbg !11
  %896 = insertelement <4 x float> %895, float %742, i64 3, !dbg !11
  %897 = select <4 x i1> %651, <4 x float> %892, <4 x float> %896, !dbg !11
  %normalized37 = fmul <4 x float> %row.reciprocal, %897, !dbg !26
  %898 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !27
  %899 = extractelement <4 x half> %898, i64 3, !dbg !27
  %900 = extractelement <4 x half> %898, i64 2, !dbg !27
  %901 = extractelement <4 x half> %898, i64 1, !dbg !27
  %902 = extractelement <4 x half> %898, i64 0, !dbg !27
  %address.thread794 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !28, !range !9, !frisk.register_lifetimes !5
  %address.local795 = zext nneg i32 %address.thread794 to i64, !dbg !8
  %903 = lshr i64 %address.local795, 2, !dbg !28
  %address.local798 = and i64 %903, 112, !dbg !28
  %address.local800 = and i64 %address.local795, 496, !dbg !28
  %address.local799 = add i64 %26, %address.local795, !dbg !28
  %address.local801 = sub i64 %address.local799, %address.local800, !dbg !28
  %address.local802 = add i64 %address.local801, %address.local798, !dbg !28
  %904 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !28
  %address.local805.idx = shl nuw nsw i64 %address.local802, 8, !dbg !28
  %address.local805 = getelementptr inbounds nuw i8, ptr addrspace(1) %904, i64 %address.local805.idx, !dbg !28
  %address.local807 = and i64 %903, 12, !dbg !10
  %address.local808 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local805, i64 %address.local807, !dbg !28
  store half %902, ptr addrspace(1) %address.local808, align 2, !dbg !28
  %address.local811 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 2, !dbg !28
  store half %901, ptr addrspace(1) %address.local811, align 2, !dbg !28
  %address.local814 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 4, !dbg !28
  store half %900, ptr addrspace(1) %address.local814, align 2, !dbg !28
  %address.local817 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 6, !dbg !28
  store half %899, ptr addrspace(1) %address.local817, align 2, !dbg !28
  %address.local820 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 32, !dbg !28
  store half %886, ptr addrspace(1) %address.local820, align 2, !dbg !28
  %address.local823 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 34, !dbg !28
  store half %885, ptr addrspace(1) %address.local823, align 2, !dbg !28
  %address.local826 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 36, !dbg !28
  store half %884, ptr addrspace(1) %address.local826, align 2, !dbg !28
  %address.local829 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 38, !dbg !28
  store half %883, ptr addrspace(1) %address.local829, align 2, !dbg !28
  %normalized39 = fmul <4 x float> %row.reciprocal, %848, !dbg !26
  %905 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !27
  %906 = extractelement <4 x half> %905, i64 3, !dbg !27
  %907 = extractelement <4 x half> %905, i64 2, !dbg !27
  %908 = extractelement <4 x half> %905, i64 1, !dbg !27
  %909 = extractelement <4 x half> %905, i64 0, !dbg !27
  %normalized40 = fmul <4 x float> %row.reciprocal, %726, !dbg !26
  %910 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !27
  %911 = extractelement <4 x half> %910, i64 3, !dbg !27
  %912 = extractelement <4 x half> %910, i64 2, !dbg !27
  %913 = extractelement <4 x half> %910, i64 1, !dbg !27
  %914 = extractelement <4 x half> %910, i64 0, !dbg !27
  %address.local845 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 64, !dbg !28
  store half %914, ptr addrspace(1) %address.local845, align 2, !dbg !28
  %address.local848 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 66, !dbg !28
  store half %913, ptr addrspace(1) %address.local848, align 2, !dbg !28
  %address.local851 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 68, !dbg !28
  store half %912, ptr addrspace(1) %address.local851, align 2, !dbg !28
  %address.local854 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 70, !dbg !28
  store half %911, ptr addrspace(1) %address.local854, align 2, !dbg !28
  %address.local857 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 96, !dbg !28
  store half %909, ptr addrspace(1) %address.local857, align 2, !dbg !28
  %address.local860 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 98, !dbg !28
  store half %908, ptr addrspace(1) %address.local860, align 2, !dbg !28
  %address.local863 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 100, !dbg !28
  store half %907, ptr addrspace(1) %address.local863, align 2, !dbg !28
  %address.local866 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 102, !dbg !28
  store half %906, ptr addrspace(1) %address.local866, align 2, !dbg !28
  %normalized42 = fmul <4 x float> %row.reciprocal, %815, !dbg !26
  %915 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !27
  %916 = extractelement <4 x half> %915, i64 3, !dbg !27
  %917 = extractelement <4 x half> %915, i64 2, !dbg !27
  %918 = extractelement <4 x half> %915, i64 1, !dbg !27
  %919 = extractelement <4 x half> %915, i64 0, !dbg !27
  %normalized43 = fmul <4 x float> %row.reciprocal, %693, !dbg !26
  %920 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !27
  %921 = extractelement <4 x half> %920, i64 3, !dbg !27
  %922 = extractelement <4 x half> %920, i64 2, !dbg !27
  %923 = extractelement <4 x half> %920, i64 1, !dbg !27
  %924 = extractelement <4 x half> %920, i64 0, !dbg !27
  %address.local882 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 128, !dbg !28
  store half %924, ptr addrspace(1) %address.local882, align 2, !dbg !28
  %address.local885 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 130, !dbg !28
  store half %923, ptr addrspace(1) %address.local885, align 2, !dbg !28
  %address.local888 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 132, !dbg !28
  store half %922, ptr addrspace(1) %address.local888, align 2, !dbg !28
  %address.local891 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 134, !dbg !28
  store half %921, ptr addrspace(1) %address.local891, align 2, !dbg !28
  %address.local894 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 160, !dbg !28
  store half %919, ptr addrspace(1) %address.local894, align 2, !dbg !28
  %address.local897 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 162, !dbg !28
  store half %918, ptr addrspace(1) %address.local897, align 2, !dbg !28
  %address.local900 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 164, !dbg !28
  store half %917, ptr addrspace(1) %address.local900, align 2, !dbg !28
  %address.local903 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 166, !dbg !28
  store half %916, ptr addrspace(1) %address.local903, align 2, !dbg !28
  %normalized45 = fmul <4 x float> %row.reciprocal, %782, !dbg !26
  %925 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !27
  %926 = extractelement <4 x half> %925, i64 3, !dbg !27
  %927 = extractelement <4 x half> %925, i64 2, !dbg !27
  %928 = extractelement <4 x half> %925, i64 1, !dbg !27
  %929 = extractelement <4 x half> %925, i64 0, !dbg !27
  %normalized46 = fmul <4 x float> %row.reciprocal, %660, !dbg !26
  %930 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !27
  %931 = extractelement <4 x half> %930, i64 3, !dbg !27
  %932 = extractelement <4 x half> %930, i64 2, !dbg !27
  %933 = extractelement <4 x half> %930, i64 1, !dbg !27
  %934 = extractelement <4 x half> %930, i64 0, !dbg !27
  %address.local919 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 192, !dbg !28
  store half %934, ptr addrspace(1) %address.local919, align 2, !dbg !28
  %address.local922 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 194, !dbg !28
  store half %933, ptr addrspace(1) %address.local922, align 2, !dbg !28
  %address.local925 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 196, !dbg !28
  store half %932, ptr addrspace(1) %address.local925, align 2, !dbg !28
  %address.local928 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 198, !dbg !28
  store half %931, ptr addrspace(1) %address.local928, align 2, !dbg !28
  %address.local931 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 224, !dbg !28
  store half %929, ptr addrspace(1) %address.local931, align 2, !dbg !28
  %address.local934 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 226, !dbg !28
  store half %928, ptr addrspace(1) %address.local934, align 2, !dbg !28
  %address.local937 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local808, i64 228, !dbg !28
  store half %927, ptr addrspace(1) %address.local937, align 2, !dbg !28
  %address.local939 = or i64 %903, 115, !dbg !28
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local805, i64 %address.local939, !dbg !28
  store half %926, ptr addrspace(1) %address.local940, align 2, !dbg !28
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="512,512" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/threads512/run-20261027-163952-a49ris5l/online-baseline")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 512, i32 1, i32 1}
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 67, column: 22, scope: !3)
!9 = !{i32 0, i32 512}
!10 = !DILocation(line: 69, column: 24, scope: !3)
!11 = !DILocation(line: 57, column: 24, scope: !3)
!12 = !DILocation(line: 70, column: 26, scope: !3)
!13 = !DILocation(line: 68, column: 22, scope: !3)
!14 = !DILocation(line: 86, column: 20, scope: !3)
!15 = !DILocation(line: 71, column: 23, scope: !3)
!16 = !DILocation(line: 74, column: 20, scope: !3)
!17 = !DILocation(line: 75, column: 22, scope: !3)
!18 = !DILocation(line: 76, column: 23, scope: !3)
!19 = !DILocation(line: 77, column: 20, scope: !3)
!20 = !DILocation(line: 88, column: 22, scope: !3)
!21 = !DILocation(line: 78, column: 22, scope: !3)
!22 = !DILocation(line: 79, column: 22, scope: !3)
!23 = !DILocation(line: 81, column: 16, scope: !3)
!24 = !DILocation(line: 80, column: 20, scope: !3)
!25 = !DILocation(line: 87, column: 23, scope: !3)
!26 = !DILocation(line: 92, column: 23, scope: !3)
!27 = !DILocation(line: 93, column: 19, scope: !3)
!28 = !DILocation(line: 94, column: 9, scope: !3)
