; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/threads512/run-20261027-163952-a49ris5l/online-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/threads512/run-20261027-163952-a49ris5l/online-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
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
  %15 = sub nsw i64 31, %14, !dbg !5
  %16 = lshr i64 %10, 2, !dbg !5
  %17 = icmp ugt i64 %10, 1023, !dbg !8
  %18 = add nsw i64 %14, -32, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 32, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %16, %22, !dbg !8
  %24 = and i64 %23, 31, !dbg !8
  %25 = shl nsw i64 %15, 7, !dbg !8
  %26 = mul nsw i64 %22, -4096, !dbg !8
  %27 = add i64 %25, %26, !dbg !8
  %28 = shl nuw nsw i64 %24, 19, !dbg !8
  %29 = shl nsw i64 %27, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !32
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local304 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local305 = and i64 %address.local304, 3584, !dbg !8
  %address.local306 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local305, !dbg !8
  %address.local307 = ptrtoint ptr addrspace(3) %address.local306 to i64, !dbg !8
  %address.local308 = trunc i64 %address.local307 to i32, !dbg !8
  %33 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local308), !dbg !8
  %address.local309 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local310 = and i64 %address.local309, 16256, !dbg !8
  %address.local311 = lshr i64 %address.local, 4, !dbg !8
  %address.local312 = and i64 %address.local311, 24, !dbg !8
  %address.local313 = or disjoint i64 %address.local310, %address.local312, !dbg !8
  %34 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1039 = or disjoint i64 %30, %address.local313, !dbg !8
  %.idx1039.tr = trunc i64 %.idx1039 to i32, !dbg !8
  %dma.buffer.offset1036 = shl i32 %.idx1039.tr, 1, !dbg !8
  %35 = insertelement <2 x i64> poison, i64 %34, i64 0, !dbg !8
  %36 = bitcast <2 x i64> %35 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1037 = shufflevector <4 x i32> %36, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1038 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %33), !dbg !8
  %37 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1038, i32 %dma.buffer.offset1036, <4 x i32> %dma.buffer.rsrc1037) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local305, !dbg !8
  %address.local316 = ptrtoint ptr addrspace(3) %address.local315 to i64, !dbg !8
  %address.local317 = trunc i64 %address.local316 to i32, !dbg !8
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local317), !dbg !8
  %39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %40 = or disjoint i64 %30, 32, !dbg !8
  %41 = or disjoint i64 %address.local313, %40, !dbg !8
  %.tr = trunc i64 %41 to i32, !dbg !8
  %dma.buffer.offset1033 = shl i32 %.tr, 1, !dbg !8
  %42 = insertelement <2 x i64> poison, i64 %39, i64 0, !dbg !8
  %43 = bitcast <2 x i64> %42 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1034 = shufflevector <4 x i32> %43, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1035 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38), !dbg !8
  %44 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1035, i32 %dma.buffer.offset1033, <4 x i32> %dma.buffer.rsrc1034) #10, !dbg !8
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
  %45 = load <4 x half>, ptr addrspace(3) %address.local331, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local331, i64 4096, !dbg !10
  %46 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %47 = fmul <4 x half> %45, splat (half 0xH3015), !dbg !10
  %48 = fmul <4 x half> %46, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %49 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local308), !dbg !8
  %50 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %51 = or disjoint i64 %30, 64, !dbg !8
  %52 = or disjoint i64 %address.local313, %51, !dbg !8
  %.tr1040 = trunc i64 %52 to i32, !dbg !8
  %dma.buffer.offset1030 = shl i32 %.tr1040, 1, !dbg !8
  %53 = insertelement <2 x i64> poison, i64 %50, i64 0, !dbg !8
  %54 = bitcast <2 x i64> %53 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1031 = shufflevector <4 x i32> %54, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1032 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %49), !dbg !8
  %55 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1032, i32 %dma.buffer.offset1030, <4 x i32> %dma.buffer.rsrc1031) #10, !dbg !8
  %lds.const.gep941 = getelementptr i8, ptr addrspace(3) %address.local331, i64 8192, !dbg !10
  %56 = load <4 x half>, ptr addrspace(3) %lds.const.gep941, align 8, !dbg !10
  %lds.const.gep942 = getelementptr i8, ptr addrspace(3) %address.local331, i64 12288, !dbg !10
  %57 = load <4 x half>, ptr addrspace(3) %lds.const.gep942, align 8, !dbg !10
  %58 = fmul <4 x half> %56, splat (half 0xH3015), !dbg !10
  %59 = fmul <4 x half> %57, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %60 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local317), !dbg !8
  %61 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %62 = or disjoint i64 %30, 96, !dbg !8
  %63 = or disjoint i64 %address.local313, %62, !dbg !8
  %.tr1041 = trunc i64 %63 to i32, !dbg !8
  %dma.buffer.offset1027 = shl i32 %.tr1041, 1, !dbg !8
  %64 = insertelement <2 x i64> poison, i64 %61, i64 0, !dbg !8
  %65 = bitcast <2 x i64> %64 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1028 = shufflevector <4 x i32> %65, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1029 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %66 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1029, i32 %dma.buffer.offset1027, <4 x i32> %dma.buffer.rsrc1028) #10, !dbg !8
  %67 = load <4 x half>, ptr addrspace(3) %address.local331, align 8, !dbg !10
  %lds.const.gep943 = getelementptr i8, ptr addrspace(3) %address.local331, i64 4096, !dbg !10
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep943, align 8, !dbg !10
  %69 = fmul <4 x half> %67, splat (half 0xH3015), !dbg !10
  %70 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep944 = getelementptr i8, ptr addrspace(3) %address.local331, i64 8192, !dbg !10
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep944, align 8, !dbg !10
  %lds.const.gep945 = getelementptr i8, ptr addrspace(3) %address.local331, i64 12288, !dbg !10
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep945, align 8, !dbg !10
  %73 = fmul <4 x half> %71, splat (half 0xH3015), !dbg !10
  %74 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !10
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %75 = shl nuw nsw i64 %32, 2, !dbg !14
  %76 = shl nuw nsw i64 %32, 2, !dbg !14
  %77 = and i64 %76, 2032, !dbg !14
  %78 = sub nsw i64 %75, %77, !dbg !14
  %79 = and i32 %31, 1, !dbg !14
  %index.compact262 = zext nneg i32 %79 to i64, !dbg !14
  %80 = or disjoint i64 %78, %index.compact262, !dbg !14
  %81 = add nuw nsw i64 %80, 1, !dbg !14
  %82 = sdiv i64 %81, 2, !dbg !14
  br label %83, !dbg !7

83:                                               ; preds = %100, %4
  %84 = phi i64 [ %667, %100 ], [ 0, %4 ], !dbg !7
  %85 = phi <4 x float> [ %492, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %86 = phi <4 x float> [ %508, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %87 = phi <8 x float> [ %663, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %88 = phi <1 x float> [ %455, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %89 = phi <1 x float> [ %324, %100 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %90 = phi <4 x float> [ %545, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %91 = phi <4 x float> [ %561, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %92 = phi <8 x float> [ %664, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %93 = phi <4 x float> [ %598, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %94 = phi <4 x float> [ %614, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %95 = phi <8 x float> [ %665, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %96 = phi <4 x float> [ %646, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %97 = phi <4 x float> [ %662, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %98 = phi <8 x float> [ %666, %100 ], [ zeroinitializer, %4 ], !dbg !7
  %99 = icmp samesign ult i64 %84, 32, !dbg !7
  br i1 %99, label %100, label %668, !dbg !7

100:                                              ; preds = %83
  %101 = shl i64 %16, 19, !dbg !12
  %102 = and i64 %101, 16252928, !dbg !12
  %103 = shl nuw nsw i64 %84, 14, !dbg !12
  %104 = or disjoint i64 %102, %103, !dbg !12
  %address.thread350 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %address.local351 = zext nneg i32 %address.thread350 to i64, !dbg !8
  %address.local352 = shl nuw nsw i64 %address.local351, 3, !dbg !8
  %address.local353 = and i64 %address.local352, 3584, !dbg !8
  %address.local354 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local353, !dbg !8
  %address.local355 = ptrtoint ptr addrspace(3) %address.local354 to i64, !dbg !8
  %address.local356 = trunc i64 %address.local355 to i32, !dbg !8
  %105 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local356), !dbg !12
  %address.local357 = lshr i64 %address.local351, 4, !dbg !12
  %address.local358 = and i64 %address.local357, 24, !dbg !12
  %address.local359 = and i64 %address.local351, 123, !dbg !12
  %address.local360 = lshr i64 %address.local351, 2, !dbg !12
  %address.local361 = and i64 %address.local360, 4, !dbg !12
  %address.local362 = add nuw nsw i64 %address.local361, %address.local351, !dbg !12
  %address.local363 = and i64 %address.local362, 4, !dbg !12
  %address.local364 = or disjoint i64 %address.local359, %address.local363, !dbg !12
  %address.local365 = shl nuw nsw i64 %address.local364, 7, !dbg !12
  %address.local366 = or disjoint i64 %address.local358, %address.local365, !dbg !12
  %106 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1042 = or disjoint i64 %104, %address.local366, !dbg !12
  %.idx1042.tr = trunc nuw i64 %.idx1042 to i32, !dbg !12
  %dma.buffer.offset1024 = shl nuw nsw i32 %.idx1042.tr, 1, !dbg !12
  %107 = insertelement <2 x i64> poison, i64 %106, i64 0, !dbg !12
  %108 = bitcast <2 x i64> %107 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1025 = shufflevector <4 x i32> %108, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1026 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %105), !dbg !12
  %109 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1026, i32 %dma.buffer.offset1024, <4 x i32> %dma.buffer.rsrc1025) #10, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %110 = or disjoint i64 %104, 32, !dbg !12
  %address.local368 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local353, !dbg !8
  %address.local369 = ptrtoint ptr addrspace(3) %address.local368 to i64, !dbg !8
  %address.local370 = trunc i64 %address.local369 to i32, !dbg !8
  %111 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local370), !dbg !12
  %112 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1043 = or disjoint i64 %110, %address.local366, !dbg !12
  %.idx1043.tr = trunc nuw i64 %.idx1043 to i32, !dbg !12
  %dma.buffer.offset1021 = shl nuw nsw i32 %.idx1043.tr, 1, !dbg !12
  %113 = insertelement <2 x i64> poison, i64 %112, i64 0, !dbg !12
  %114 = bitcast <2 x i64> %113 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1022 = shufflevector <4 x i32> %114, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1023 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %111), !dbg !12
  %115 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1023, i32 %dma.buffer.offset1021, <4 x i32> %dma.buffer.rsrc1022) #10, !dbg !12
  %address.local373 = lshr i64 %address.local351, 2, !dbg !14
  %address.local376 = shl nuw nsw i64 %address.local351, 1, !dbg !14
  %address.local377 = and i64 %address.local376, 24, !dbg !14
  %address.local379 = shl i64 %address.local351, 62, !dbg !14
  %address.local380 = ashr exact i64 %address.local379, 57, !dbg !14
  %address.local381 = and i64 %address.local380, -64, !dbg !14
  %address.local383 = lshr i64 %address.local351, 2, !dbg !10
  %address.local384 = and i64 %address.local383, 12, !dbg !10
  %address.local386 = and i64 %address.local351, 63, !dbg !10
  %address.local387 = lshr i64 %address.local386, 5, !dbg !10
  %address.local388 = mul nuw nsw i64 %address.local387, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local351, 320, !dbg !14
  %116 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !14
  %.idx1044 = mul nsw i64 %address.local373, -1280, !dbg !14
  %117 = getelementptr i8, ptr addrspace(3) %116, i64 %.idx1044, !dbg !14
  %118 = getelementptr [2 x i8], ptr addrspace(3) %117, i64 %address.local377, !dbg !14
  %119 = getelementptr [2 x i8], ptr addrspace(3) %118, i64 %address.local381, !dbg !14
  %120 = getelementptr [2 x i8], ptr addrspace(3) %119, i64 %address.local384, !dbg !14
  %address.local390 = getelementptr [2 x i8], ptr addrspace(3) %120, i64 %address.local388, !dbg !14
  %121 = load <4 x half>, ptr addrspace(3) %address.local390, align 8, !dbg !14
  %lds.const.gep946 = getelementptr i8, ptr addrspace(3) %address.local390, i64 4096, !dbg !14
  %122 = load <4 x half>, ptr addrspace(3) %lds.const.gep946, align 8, !dbg !14
  %123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %121, <4 x float> zeroinitializer) #9, !dbg !14
  %address.local401 = shl nuw nsw i32 %address.thread350, 1, !dbg !14
  %address.local402 = and i32 %address.local401, 24, !dbg !14
  %address.local403 = shl nuw nsw i32 %address.thread350, 8, !dbg !14
  %address.local404 = and i32 %address.local403, 768, !dbg !14
  %address.local405 = or disjoint i32 %address.local402, %address.local404, !dbg !14
  %address.local406 = shl nuw nsw i32 %address.thread350, 5, !dbg !14
  %address.local407 = and i32 %address.local406, 96, !dbg !14
  %address.local408 = or disjoint i32 %address.local405, %address.local407, !dbg !14
  %address.local409 = zext nneg i32 %address.local408 to i64, !dbg !14
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %82) #9, !dbg !14, !frisk.register_lifetimes !1
  %address.local410 = lshr i32 %address.thread350, 1, !dbg !14
  %address.local411 = and i32 %address.local410, 1, !dbg !14
  %address.local412 = zext nneg i32 %address.local411 to i64, !dbg !14
  %address.local413 = add nsw i64 %address.base, %address.local412, !dbg !14
  %address.local414 = shl i64 %address.local413, 6, !dbg !14
  %address.local415 = sub i64 %address.local409, %address.local414, !dbg !14
  %124 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local415, !dbg !14
  %125 = getelementptr [2 x i8], ptr addrspace(3) %124, i64 %address.local384, !dbg !14
  %126 = getelementptr [2 x i8], ptr addrspace(3) %125, i64 %address.local388, !dbg !14
  %address.local419 = getelementptr i8, ptr addrspace(3) %126, i64 64, !dbg !14
  %127 = load <4 x half>, ptr addrspace(3) %address.local419, align 8, !dbg !14
  %128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %122, <4 x float> %123) #9, !dbg !14
  %lds.const.gep947 = getelementptr i8, ptr addrspace(3) %126, i64 4160, !dbg !14
  %129 = load <4 x half>, ptr addrspace(3) %lds.const.gep947, align 8, !dbg !14
  %lds.const.gep948 = getelementptr i8, ptr addrspace(3) %address.local390, i64 128, !dbg !14
  %130 = load <4 x half>, ptr addrspace(3) %lds.const.gep948, align 8, !dbg !14
  %131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %127, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep949 = getelementptr i8, ptr addrspace(3) %address.local390, i64 4224, !dbg !14
  %132 = load <4 x half>, ptr addrspace(3) %lds.const.gep949, align 8, !dbg !14
  %133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %129, <4 x float> %131) #9, !dbg !14
  %lds.const.gep950 = getelementptr i8, ptr addrspace(3) %126, i64 192, !dbg !14
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !14
  %lds.const.gep951 = getelementptr i8, ptr addrspace(3) %126, i64 4288, !dbg !14
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !14
  %136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %130, <4 x float> zeroinitializer) #9, !dbg !14
  %137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %132, <4 x float> %136) #9, !dbg !14
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %134, <4 x float> zeroinitializer) #9, !dbg !14
  %139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %135, <4 x float> %138) #9, !dbg !14
  %lds.const.gep952 = getelementptr i8, ptr addrspace(3) %address.local390, i64 1024, !dbg !14
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !14
  %lds.const.gep953 = getelementptr i8, ptr addrspace(3) %address.local390, i64 5120, !dbg !14
  %141 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !14
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %140, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep954 = getelementptr i8, ptr addrspace(3) %126, i64 1088, !dbg !14
  %143 = load <4 x half>, ptr addrspace(3) %lds.const.gep954, align 8, !dbg !14
  %144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %141, <4 x float> %142) #9, !dbg !14
  %lds.const.gep955 = getelementptr i8, ptr addrspace(3) %126, i64 5184, !dbg !14
  %145 = load <4 x half>, ptr addrspace(3) %lds.const.gep955, align 8, !dbg !14
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %143, <4 x float> zeroinitializer) #9, !dbg !14
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %145, <4 x float> %146) #9, !dbg !14
  %lds.const.gep956 = getelementptr i8, ptr addrspace(3) %address.local390, i64 1152, !dbg !14
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep956, align 8, !dbg !14
  %lds.const.gep957 = getelementptr i8, ptr addrspace(3) %address.local390, i64 5248, !dbg !14
  %149 = load <4 x half>, ptr addrspace(3) %lds.const.gep957, align 8, !dbg !14
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %148, <4 x float> zeroinitializer) #9, !dbg !14
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %149, <4 x float> %150) #9, !dbg !14
  %lds.const.gep958 = getelementptr i8, ptr addrspace(3) %126, i64 1216, !dbg !14
  %152 = load <4 x half>, ptr addrspace(3) %lds.const.gep958, align 8, !dbg !14
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %152, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep959 = getelementptr i8, ptr addrspace(3) %126, i64 5312, !dbg !14
  %154 = load <4 x half>, ptr addrspace(3) %lds.const.gep959, align 8, !dbg !14
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %154, <4 x float> %153) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %156 = or disjoint i64 %104, 64, !dbg !12
  %157 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local356), !dbg !12
  %158 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx10451046 = or disjoint i64 %156, %address.local366, !dbg !12
  %.idx10451046.tr = trunc nuw i64 %.idx10451046 to i32, !dbg !12
  %dma.buffer.offset1018 = shl nuw nsw i32 %.idx10451046.tr, 1, !dbg !12
  %159 = insertelement <2 x i64> poison, i64 %158, i64 0, !dbg !12
  %160 = bitcast <2 x i64> %159 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1019 = shufflevector <4 x i32> %160, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1020 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %157), !dbg !12
  %161 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1020, i32 %dma.buffer.offset1018, <4 x i32> %dma.buffer.rsrc1019) #10, !dbg !12
  %lds.const.gep960 = getelementptr i8, ptr addrspace(3) %address.local390, i64 8192, !dbg !14
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep960, align 8, !dbg !14
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %162, <4 x float> %128) #9, !dbg !14
  %lds.const.gep961 = getelementptr i8, ptr addrspace(3) %address.local390, i64 12288, !dbg !14
  %164 = load <4 x half>, ptr addrspace(3) %lds.const.gep961, align 8, !dbg !14
  %lds.const.gep962 = getelementptr i8, ptr addrspace(3) %126, i64 8256, !dbg !14
  %165 = load <4 x half>, ptr addrspace(3) %lds.const.gep962, align 8, !dbg !14
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %164, <4 x float> %163) #9, !dbg !14
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %165, <4 x float> %133) #9, !dbg !14
  %lds.const.gep963 = getelementptr i8, ptr addrspace(3) %126, i64 12352, !dbg !14
  %168 = load <4 x half>, ptr addrspace(3) %lds.const.gep963, align 8, !dbg !14
  %lds.const.gep964 = getelementptr i8, ptr addrspace(3) %address.local390, i64 8320, !dbg !14
  %169 = load <4 x half>, ptr addrspace(3) %lds.const.gep964, align 8, !dbg !14
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %168, <4 x float> %167) #9, !dbg !14
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %169, <4 x float> %137) #9, !dbg !14
  %lds.const.gep965 = getelementptr i8, ptr addrspace(3) %address.local390, i64 12416, !dbg !14
  %172 = load <4 x half>, ptr addrspace(3) %lds.const.gep965, align 8, !dbg !14
  %lds.const.gep966 = getelementptr i8, ptr addrspace(3) %126, i64 8384, !dbg !14
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep966, align 8, !dbg !14
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %172, <4 x float> %171) #9, !dbg !14
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %173, <4 x float> %139) #9, !dbg !14
  %lds.const.gep967 = getelementptr i8, ptr addrspace(3) %126, i64 12480, !dbg !14
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep967, align 8, !dbg !14
  %lds.const.gep968 = getelementptr i8, ptr addrspace(3) %address.local390, i64 9216, !dbg !14
  %177 = load <4 x half>, ptr addrspace(3) %lds.const.gep968, align 8, !dbg !14
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %176, <4 x float> %175) #9, !dbg !14
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %177, <4 x float> %144) #9, !dbg !14
  %lds.const.gep969 = getelementptr i8, ptr addrspace(3) %address.local390, i64 13312, !dbg !14
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep969, align 8, !dbg !14
  %lds.const.gep970 = getelementptr i8, ptr addrspace(3) %126, i64 9280, !dbg !14
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep970, align 8, !dbg !14
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %180, <4 x float> %179) #9, !dbg !14
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %181, <4 x float> %147) #9, !dbg !14
  %lds.const.gep971 = getelementptr i8, ptr addrspace(3) %126, i64 13376, !dbg !14
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep971, align 8, !dbg !14
  %lds.const.gep972 = getelementptr i8, ptr addrspace(3) %address.local390, i64 9344, !dbg !14
  %185 = load <4 x half>, ptr addrspace(3) %lds.const.gep972, align 8, !dbg !14
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %184, <4 x float> %183) #9, !dbg !14
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %185, <4 x float> %151) #9, !dbg !14
  %lds.const.gep973 = getelementptr i8, ptr addrspace(3) %address.local390, i64 13440, !dbg !14
  %188 = load <4 x half>, ptr addrspace(3) %lds.const.gep973, align 8, !dbg !14
  %lds.const.gep974 = getelementptr i8, ptr addrspace(3) %126, i64 9408, !dbg !14
  %189 = load <4 x half>, ptr addrspace(3) %lds.const.gep974, align 8, !dbg !14
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %188, <4 x float> %187) #9, !dbg !14
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %189, <4 x float> %155) #9, !dbg !14
  %lds.const.gep975 = getelementptr i8, ptr addrspace(3) %126, i64 13504, !dbg !14
  %192 = load <4 x half>, ptr addrspace(3) %lds.const.gep975, align 8, !dbg !14
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %192, <4 x float> %191) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %194 = or disjoint i64 %104, 96, !dbg !12
  %195 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local370), !dbg !12
  %196 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx10471048 = or disjoint i64 %194, %address.local366, !dbg !12
  %.idx10471048.tr = trunc nuw i64 %.idx10471048 to i32, !dbg !12
  %dma.buffer.offset1015 = shl nuw nsw i32 %.idx10471048.tr, 1, !dbg !12
  %197 = insertelement <2 x i64> poison, i64 %196, i64 0, !dbg !12
  %198 = bitcast <2 x i64> %197 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1016 = shufflevector <4 x i32> %198, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1017 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %195), !dbg !12
  %199 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1017, i32 %dma.buffer.offset1015, <4 x i32> %dma.buffer.rsrc1016) #10, !dbg !12
  %200 = load <4 x half>, ptr addrspace(3) %address.local390, align 8, !dbg !14
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %200, <4 x float> %166) #9, !dbg !14
  %lds.const.gep976 = getelementptr i8, ptr addrspace(3) %address.local390, i64 4096, !dbg !14
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep976, align 8, !dbg !14
  %203 = load <4 x half>, ptr addrspace(3) %address.local419, align 8, !dbg !14
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %202, <4 x float> %201) #9, !dbg !14
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %203, <4 x float> %170) #9, !dbg !14
  %lds.const.gep977 = getelementptr i8, ptr addrspace(3) %126, i64 4160, !dbg !14
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep977, align 8, !dbg !14
  %lds.const.gep978 = getelementptr i8, ptr addrspace(3) %address.local390, i64 128, !dbg !14
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep978, align 8, !dbg !14
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %206, <4 x float> %205) #9, !dbg !14
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %207, <4 x float> %174) #9, !dbg !14
  %lds.const.gep979 = getelementptr i8, ptr addrspace(3) %address.local390, i64 4224, !dbg !14
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep979, align 8, !dbg !14
  %lds.const.gep980 = getelementptr i8, ptr addrspace(3) %126, i64 192, !dbg !14
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep980, align 8, !dbg !14
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %210, <4 x float> %209) #9, !dbg !14
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %211, <4 x float> %178) #9, !dbg !14
  %lds.const.gep981 = getelementptr i8, ptr addrspace(3) %126, i64 4288, !dbg !14
  %214 = load <4 x half>, ptr addrspace(3) %lds.const.gep981, align 8, !dbg !14
  %lds.const.gep982 = getelementptr i8, ptr addrspace(3) %address.local390, i64 1024, !dbg !14
  %215 = load <4 x half>, ptr addrspace(3) %lds.const.gep982, align 8, !dbg !14
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %214, <4 x float> %213) #9, !dbg !14
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %215, <4 x float> %182) #9, !dbg !14
  %lds.const.gep983 = getelementptr i8, ptr addrspace(3) %address.local390, i64 5120, !dbg !14
  %218 = load <4 x half>, ptr addrspace(3) %lds.const.gep983, align 8, !dbg !14
  %lds.const.gep984 = getelementptr i8, ptr addrspace(3) %126, i64 1088, !dbg !14
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep984, align 8, !dbg !14
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %218, <4 x float> %217) #9, !dbg !14
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %219, <4 x float> %186) #9, !dbg !14
  %lds.const.gep985 = getelementptr i8, ptr addrspace(3) %126, i64 5184, !dbg !14
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep985, align 8, !dbg !14
  %lds.const.gep986 = getelementptr i8, ptr addrspace(3) %address.local390, i64 1152, !dbg !14
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep986, align 8, !dbg !14
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %222, <4 x float> %221) #9, !dbg !14
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %223, <4 x float> %190) #9, !dbg !14
  %lds.const.gep987 = getelementptr i8, ptr addrspace(3) %address.local390, i64 5248, !dbg !14
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep987, align 8, !dbg !14
  %lds.const.gep988 = getelementptr i8, ptr addrspace(3) %126, i64 1216, !dbg !14
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep988, align 8, !dbg !14
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %226, <4 x float> %225) #9, !dbg !14
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %227, <4 x float> %193) #9, !dbg !14
  %lds.const.gep989 = getelementptr i8, ptr addrspace(3) %126, i64 5312, !dbg !14
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep989, align 8, !dbg !14
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %230, <4 x float> %229) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %232 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local356), !dbg !15
  %address.local746 = shl nuw nsw i64 %address.local351, 5, !dbg !15
  %address.local747 = and i64 %address.local746, 16256, !dbg !15
  %address.local748 = shl nuw nsw i64 %address.local351, 3, !dbg !15
  %address.local749 = and i64 %address.local748, 24, !dbg !15
  %address.local750 = or disjoint i64 %address.local747, %address.local749, !dbg !15
  %233 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx10491050 = or disjoint i64 %104, %address.local750, !dbg !15
  %.idx10491050.tr = trunc nuw i64 %.idx10491050 to i32, !dbg !15
  %dma.buffer.offset1012 = shl nuw nsw i32 %.idx10491050.tr, 1, !dbg !15
  %234 = insertelement <2 x i64> poison, i64 %233, i64 0, !dbg !15
  %235 = bitcast <2 x i64> %234 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1013 = shufflevector <4 x i32> %235, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1014 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %232), !dbg !15
  %236 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1014, i32 %dma.buffer.offset1012, <4 x i32> %dma.buffer.rsrc1013) #10, !dbg !15
  %lds.const.gep990 = getelementptr i8, ptr addrspace(3) %address.local390, i64 8192, !dbg !14
  %237 = load <4 x half>, ptr addrspace(3) %lds.const.gep990, align 8, !dbg !14
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %237, <4 x float> %204) #9, !dbg !14
  %lds.const.gep991 = getelementptr i8, ptr addrspace(3) %address.local390, i64 12288, !dbg !14
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep991, align 8, !dbg !14
  %lds.const.gep992 = getelementptr i8, ptr addrspace(3) %126, i64 8256, !dbg !14
  %240 = load <4 x half>, ptr addrspace(3) %lds.const.gep992, align 8, !dbg !14
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %239, <4 x float> %238) #9, !dbg !14
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %240, <4 x float> %208) #9, !dbg !14
  %243 = extractelement <4 x float> %241, i64 0, !dbg !19
  %244 = call float @llvm.maxnum.f32(float %243, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %245 = extractelement <4 x float> %241, i64 1, !dbg !19
  %246 = extractelement <4 x float> %241, i64 2, !dbg !19
  %247 = extractelement <4 x float> %241, i64 3, !dbg !19
  %lds.const.gep993 = getelementptr i8, ptr addrspace(3) %126, i64 12352, !dbg !14
  %248 = load <4 x half>, ptr addrspace(3) %lds.const.gep993, align 8, !dbg !14
  %lds.const.gep994 = getelementptr i8, ptr addrspace(3) %address.local390, i64 8320, !dbg !14
  %249 = load <4 x half>, ptr addrspace(3) %lds.const.gep994, align 8, !dbg !14
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %248, <4 x float> %242) #9, !dbg !14
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %249, <4 x float> %212) #9, !dbg !14
  %252 = extractelement <4 x float> %250, i64 0, !dbg !19
  %local.reduce77 = call float @llvm.maxnum.f32(float %244, float %252), !dbg !20, !frisk.local_reduction !1
  %253 = extractelement <4 x float> %250, i64 1, !dbg !19
  %local.reduce78 = call float @llvm.maxnum.f32(float %245, float %253), !dbg !20, !frisk.local_reduction !1
  %254 = extractelement <4 x float> %250, i64 2, !dbg !19
  %local.reduce79 = call float @llvm.maxnum.f32(float %246, float %254), !dbg !20, !frisk.local_reduction !1
  %255 = extractelement <4 x float> %250, i64 3, !dbg !19
  %local.reduce80 = call float @llvm.maxnum.f32(float %247, float %255), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep995 = getelementptr i8, ptr addrspace(3) %address.local390, i64 12416, !dbg !14
  %256 = load <4 x half>, ptr addrspace(3) %lds.const.gep995, align 8, !dbg !14
  %lds.const.gep996 = getelementptr i8, ptr addrspace(3) %126, i64 8384, !dbg !14
  %257 = load <4 x half>, ptr addrspace(3) %lds.const.gep996, align 8, !dbg !14
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %256, <4 x float> %251) #9, !dbg !14
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %257, <4 x float> %216) #9, !dbg !14
  %260 = extractelement <4 x float> %258, i64 0, !dbg !19
  %local.reduce81 = call float @llvm.maxnum.f32(float %local.reduce77, float %260), !dbg !20, !frisk.local_reduction !1
  %261 = extractelement <4 x float> %258, i64 1, !dbg !19
  %local.reduce82 = call float @llvm.maxnum.f32(float %local.reduce78, float %261), !dbg !20, !frisk.local_reduction !1
  %262 = extractelement <4 x float> %258, i64 2, !dbg !19
  %local.reduce83 = call float @llvm.maxnum.f32(float %local.reduce79, float %262), !dbg !20, !frisk.local_reduction !1
  %263 = extractelement <4 x float> %258, i64 3, !dbg !19
  %local.reduce84 = call float @llvm.maxnum.f32(float %local.reduce80, float %263), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep997 = getelementptr i8, ptr addrspace(3) %126, i64 12480, !dbg !14
  %264 = load <4 x half>, ptr addrspace(3) %lds.const.gep997, align 8, !dbg !14
  %lds.const.gep998 = getelementptr i8, ptr addrspace(3) %address.local390, i64 9216, !dbg !14
  %265 = load <4 x half>, ptr addrspace(3) %lds.const.gep998, align 8, !dbg !14
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %264, <4 x float> %259) #9, !dbg !14
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %265, <4 x float> %220) #9, !dbg !14
  %268 = extractelement <4 x float> %266, i64 0, !dbg !19
  %local.reduce85 = call float @llvm.maxnum.f32(float %local.reduce81, float %268), !dbg !20, !frisk.local_reduction !1
  %269 = extractelement <4 x float> %266, i64 1, !dbg !19
  %local.reduce86 = call float @llvm.maxnum.f32(float %local.reduce82, float %269), !dbg !20, !frisk.local_reduction !1
  %270 = extractelement <4 x float> %266, i64 2, !dbg !19
  %local.reduce87 = call float @llvm.maxnum.f32(float %local.reduce83, float %270), !dbg !20, !frisk.local_reduction !1
  %271 = extractelement <4 x float> %266, i64 3, !dbg !19
  %local.reduce88 = call float @llvm.maxnum.f32(float %local.reduce84, float %271), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep999 = getelementptr i8, ptr addrspace(3) %address.local390, i64 13312, !dbg !14
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep999, align 8, !dbg !14
  %lds.const.gep1000 = getelementptr i8, ptr addrspace(3) %126, i64 9280, !dbg !14
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep1000, align 8, !dbg !14
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %272, <4 x float> %267) #9, !dbg !14
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %273, <4 x float> %224) #9, !dbg !14
  %276 = extractelement <4 x float> %274, i64 0, !dbg !19
  %local.reduce89 = call float @llvm.maxnum.f32(float %local.reduce85, float %276), !dbg !20, !frisk.local_reduction !1
  %277 = extractelement <4 x float> %274, i64 1, !dbg !19
  %local.reduce90 = call float @llvm.maxnum.f32(float %local.reduce86, float %277), !dbg !20, !frisk.local_reduction !1
  %278 = extractelement <4 x float> %274, i64 2, !dbg !19
  %local.reduce91 = call float @llvm.maxnum.f32(float %local.reduce87, float %278), !dbg !20, !frisk.local_reduction !1
  %279 = extractelement <4 x float> %274, i64 3, !dbg !19
  %local.reduce92 = call float @llvm.maxnum.f32(float %local.reduce88, float %279), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1001 = getelementptr i8, ptr addrspace(3) %126, i64 13376, !dbg !14
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep1001, align 8, !dbg !14
  %lds.const.gep1002 = getelementptr i8, ptr addrspace(3) %address.local390, i64 9344, !dbg !14
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep1002, align 8, !dbg !14
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %280, <4 x float> %275) #9, !dbg !14
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %281, <4 x float> %228) #9, !dbg !14
  %284 = extractelement <4 x float> %282, i64 0, !dbg !19
  %local.reduce93 = call float @llvm.maxnum.f32(float %local.reduce89, float %284), !dbg !20, !frisk.local_reduction !1
  %285 = extractelement <4 x float> %282, i64 1, !dbg !19
  %local.reduce94 = call float @llvm.maxnum.f32(float %local.reduce90, float %285), !dbg !20, !frisk.local_reduction !1
  %286 = extractelement <4 x float> %282, i64 2, !dbg !19
  %local.reduce95 = call float @llvm.maxnum.f32(float %local.reduce91, float %286), !dbg !20, !frisk.local_reduction !1
  %287 = extractelement <4 x float> %282, i64 3, !dbg !19
  %local.reduce96 = call float @llvm.maxnum.f32(float %local.reduce92, float %287), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1003 = getelementptr i8, ptr addrspace(3) %address.local390, i64 13440, !dbg !14
  %288 = load <4 x half>, ptr addrspace(3) %lds.const.gep1003, align 8, !dbg !14
  %lds.const.gep1004 = getelementptr i8, ptr addrspace(3) %126, i64 9408, !dbg !14
  %289 = load <4 x half>, ptr addrspace(3) %lds.const.gep1004, align 8, !dbg !14
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %288, <4 x float> %283) #9, !dbg !14
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %289, <4 x float> %231) #9, !dbg !14
  %292 = extractelement <4 x float> %290, i64 0, !dbg !19
  %local.reduce97 = call float @llvm.maxnum.f32(float %local.reduce93, float %292), !dbg !20, !frisk.local_reduction !1
  %293 = extractelement <4 x float> %290, i64 1, !dbg !19
  %local.reduce98 = call float @llvm.maxnum.f32(float %local.reduce94, float %293), !dbg !20, !frisk.local_reduction !1
  %294 = extractelement <4 x float> %290, i64 2, !dbg !19
  %local.reduce99 = call float @llvm.maxnum.f32(float %local.reduce95, float %294), !dbg !20, !frisk.local_reduction !1
  %295 = extractelement <4 x float> %290, i64 3, !dbg !19
  %local.reduce100 = call float @llvm.maxnum.f32(float %local.reduce96, float %295), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1005 = getelementptr i8, ptr addrspace(3) %126, i64 13504, !dbg !14
  %296 = load <4 x half>, ptr addrspace(3) %lds.const.gep1005, align 8, !dbg !14
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %296, <4 x float> %291) #9, !dbg !14
  %298 = extractelement <4 x float> %297, i64 0, !dbg !19
  %local.reduce101 = call float @llvm.maxnum.f32(float %local.reduce97, float %298), !dbg !20, !frisk.local_reduction !1
  %299 = extractelement <4 x float> %297, i64 1, !dbg !19
  %local.reduce102 = call float @llvm.maxnum.f32(float %local.reduce98, float %299), !dbg !20, !frisk.local_reduction !1
  %300 = extractelement <4 x float> %297, i64 2, !dbg !19
  %local.reduce103 = call float @llvm.maxnum.f32(float %local.reduce99, float %300), !dbg !20, !frisk.local_reduction !1
  %301 = extractelement <4 x float> %297, i64 3, !dbg !19
  %local.reduce104 = call float @llvm.maxnum.f32(float %local.reduce100, float %301), !dbg !20, !frisk.local_reduction !1
  %local.reduce105 = call float @llvm.maxnum.f32(float %local.reduce101, float %local.reduce102), !dbg !20, !frisk.local_reduction !1
  %local.reduce106 = call float @llvm.maxnum.f32(float %local.reduce103, float %local.reduce104), !dbg !20, !frisk.local_reduction !1
  %302 = call float @llvm.maxnum.f32(float %local.reduce105, float %local.reduce106), !dbg !20, !frisk.local_reduction !1
  %303 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %304 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %303), !dbg !20
  %305 = add nuw nsw i32 %304, 64, !dbg !20
  %306 = xor i32 %304, 16, !dbg !20
  %307 = and i32 %305, 192, !dbg !20
  %308 = icmp samesign ult i32 %306, %307, !dbg !20
  %309 = select i1 %308, i32 %306, i32 %304, !dbg !20
  %310 = shl nuw nsw i32 %309, 2, !dbg !20
  %311 = bitcast float %302 to i32, !dbg !20
  %312 = call i32 @llvm.amdgcn.ds.bpermute(i32 %310, i32 %311), !dbg !20
  %313 = bitcast i32 %312 to float, !dbg !20
  %314 = call float @llvm.maxnum.f32(float %302, float %313), !dbg !20
  %315 = xor i32 %304, 32, !dbg !20
  %316 = icmp samesign ult i32 %315, %307, !dbg !20
  %317 = select i1 %316, i32 %315, i32 %304, !dbg !20
  %318 = shl nuw nsw i32 %317, 2, !dbg !20
  %319 = bitcast float %314 to i32, !dbg !20
  %320 = call i32 @llvm.amdgcn.ds.bpermute(i32 %318, i32 %319), !dbg !20
  %321 = bitcast i32 %320 to float, !dbg !20
  %322 = call float @llvm.maxnum.f32(float %314, float %321), !dbg !20
  %323 = insertelement <1 x float> poison, float %322, i64 0, !dbg !20
  %324 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %323, <1 x float> %89), !dbg !20
  %325 = shufflevector <1 x float> %324, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %326 = fsub <4 x float> %241, %325, !dbg !21
  %327 = fsub <4 x float> %250, %325, !dbg !21
  %328 = fsub <4 x float> %258, %325, !dbg !21
  %329 = fsub <4 x float> %266, %325, !dbg !21
  %330 = fsub <4 x float> %274, %325, !dbg !21
  %331 = fsub <4 x float> %282, %325, !dbg !21
  %332 = fsub <4 x float> %290, %325, !dbg !21
  %333 = fsub <4 x float> %297, %325, !dbg !21
  %334 = fsub <1 x float> %89, %324, !dbg !22
  %335 = extractelement <1 x float> %334, i64 0, !dbg !23
  %336 = call float @llvm.amdgcn.exp2.f32(float %335), !dbg !23
  %337 = insertelement <4 x float> poison, float %336, i64 0, !dbg !28
  %338 = shufflevector <4 x float> %337, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %339 = fmul <4 x float> %97, %338, !dbg !28
  %340 = fmul <4 x float> %96, %338, !dbg !28
  %341 = fmul <4 x float> %94, %338, !dbg !28
  %342 = fmul <4 x float> %93, %338, !dbg !28
  %343 = fmul <4 x float> %91, %338, !dbg !28
  %344 = fmul <4 x float> %90, %338, !dbg !28
  %345 = fmul <4 x float> %86, %338, !dbg !28
  %346 = fmul <4 x float> %85, %338, !dbg !28
  %347 = extractelement <1 x float> %88, i64 0, !dbg !24
  %348 = fmul float %347, %336, !dbg !24
  %349 = extractelement <4 x float> %326, i64 0, !dbg !25
  %350 = call float @llvm.amdgcn.exp2.f32(float %349), !dbg !25
  %351 = insertelement <4 x float> poison, float %350, i64 0, !dbg !27
  %352 = extractelement <4 x float> %326, i64 1, !dbg !25
  %353 = call float @llvm.amdgcn.exp2.f32(float %352), !dbg !25
  %354 = insertelement <4 x float> %351, float %353, i64 1, !dbg !27
  %355 = extractelement <4 x float> %326, i64 2, !dbg !25
  %356 = call float @llvm.amdgcn.exp2.f32(float %355), !dbg !25
  %357 = insertelement <4 x float> %354, float %356, i64 2, !dbg !27
  %358 = extractelement <4 x float> %326, i64 3, !dbg !25
  %359 = call float @llvm.amdgcn.exp2.f32(float %358), !dbg !25
  %360 = insertelement <4 x float> %357, float %359, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %360) #9, !dbg !27, !frisk.register_lifetimes !1
  %361 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %361) #9, !dbg !25, !frisk.register_lifetimes !1
  %362 = extractelement <4 x float> %327, i64 0, !dbg !25
  %363 = call float @llvm.amdgcn.exp2.f32(float %362), !dbg !25
  %364 = insertelement <4 x float> poison, float %363, i64 0, !dbg !27
  %local.reduce = fadd float %350, %363, !dbg !26, !frisk.local_reduction !1
  %365 = extractelement <4 x float> %327, i64 1, !dbg !25
  %366 = call float @llvm.amdgcn.exp2.f32(float %365), !dbg !25
  %367 = insertelement <4 x float> %364, float %366, i64 1, !dbg !27
  %local.reduce47 = fadd float %353, %366, !dbg !26, !frisk.local_reduction !1
  %368 = extractelement <4 x float> %327, i64 2, !dbg !25
  %369 = call float @llvm.amdgcn.exp2.f32(float %368), !dbg !25
  %370 = insertelement <4 x float> %367, float %369, i64 2, !dbg !27
  %local.reduce48 = fadd float %356, %369, !dbg !26, !frisk.local_reduction !1
  %371 = extractelement <4 x float> %327, i64 3, !dbg !25
  %372 = call float @llvm.amdgcn.exp2.f32(float %371), !dbg !25
  %373 = insertelement <4 x float> %370, float %372, i64 3, !dbg !27
  %math.result290 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %373) #9, !dbg !27, !frisk.register_lifetimes !1
  %374 = fptrunc <4 x float> %math.result290 to <4 x half>, !dbg !27
  %math.packed291 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %374) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce49 = fadd float %359, %372, !dbg !26, !frisk.local_reduction !1
  %375 = extractelement <4 x float> %328, i64 0, !dbg !25
  %376 = call float @llvm.amdgcn.exp2.f32(float %375), !dbg !25
  %377 = insertelement <4 x float> poison, float %376, i64 0, !dbg !27
  %local.reduce50 = fadd float %local.reduce, %376, !dbg !26, !frisk.local_reduction !1
  %378 = extractelement <4 x float> %328, i64 1, !dbg !25
  %379 = call float @llvm.amdgcn.exp2.f32(float %378), !dbg !25
  %380 = insertelement <4 x float> %377, float %379, i64 1, !dbg !27
  %local.reduce51 = fadd float %local.reduce47, %379, !dbg !26, !frisk.local_reduction !1
  %381 = extractelement <4 x float> %328, i64 2, !dbg !25
  %382 = call float @llvm.amdgcn.exp2.f32(float %381), !dbg !25
  %383 = insertelement <4 x float> %380, float %382, i64 2, !dbg !27
  %local.reduce52 = fadd float %local.reduce48, %382, !dbg !26, !frisk.local_reduction !1
  %384 = extractelement <4 x float> %328, i64 3, !dbg !25
  %385 = call float @llvm.amdgcn.exp2.f32(float %384), !dbg !25
  %386 = insertelement <4 x float> %383, float %385, i64 3, !dbg !27
  %math.result292 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %386) #9, !dbg !27, !frisk.register_lifetimes !1
  %387 = fptrunc <4 x float> %math.result292 to <4 x half>, !dbg !27
  %math.packed293 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %387) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce53 = fadd float %local.reduce49, %385, !dbg !26, !frisk.local_reduction !1
  %388 = extractelement <4 x float> %329, i64 0, !dbg !25
  %389 = call float @llvm.amdgcn.exp2.f32(float %388), !dbg !25
  %390 = insertelement <4 x float> poison, float %389, i64 0, !dbg !27
  %local.reduce54 = fadd float %local.reduce50, %389, !dbg !26, !frisk.local_reduction !1
  %391 = extractelement <4 x float> %329, i64 1, !dbg !25
  %392 = call float @llvm.amdgcn.exp2.f32(float %391), !dbg !25
  %393 = insertelement <4 x float> %390, float %392, i64 1, !dbg !27
  %local.reduce55 = fadd float %local.reduce51, %392, !dbg !26, !frisk.local_reduction !1
  %394 = extractelement <4 x float> %329, i64 2, !dbg !25
  %395 = call float @llvm.amdgcn.exp2.f32(float %394), !dbg !25
  %396 = insertelement <4 x float> %393, float %395, i64 2, !dbg !27
  %local.reduce56 = fadd float %local.reduce52, %395, !dbg !26, !frisk.local_reduction !1
  %397 = extractelement <4 x float> %329, i64 3, !dbg !25
  %398 = call float @llvm.amdgcn.exp2.f32(float %397), !dbg !25
  %399 = insertelement <4 x float> %396, float %398, i64 3, !dbg !27
  %math.result294 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %399) #9, !dbg !27, !frisk.register_lifetimes !1
  %400 = fptrunc <4 x float> %math.result294 to <4 x half>, !dbg !27
  %math.packed295 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %400) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce57 = fadd float %local.reduce53, %398, !dbg !26, !frisk.local_reduction !1
  %401 = extractelement <4 x float> %330, i64 0, !dbg !25
  %402 = call float @llvm.amdgcn.exp2.f32(float %401), !dbg !25
  %403 = insertelement <4 x float> poison, float %402, i64 0, !dbg !27
  %local.reduce58 = fadd float %local.reduce54, %402, !dbg !26, !frisk.local_reduction !1
  %404 = extractelement <4 x float> %330, i64 1, !dbg !25
  %405 = call float @llvm.amdgcn.exp2.f32(float %404), !dbg !25
  %406 = insertelement <4 x float> %403, float %405, i64 1, !dbg !27
  %local.reduce59 = fadd float %local.reduce55, %405, !dbg !26, !frisk.local_reduction !1
  %407 = extractelement <4 x float> %330, i64 2, !dbg !25
  %408 = call float @llvm.amdgcn.exp2.f32(float %407), !dbg !25
  %409 = insertelement <4 x float> %406, float %408, i64 2, !dbg !27
  %local.reduce60 = fadd float %local.reduce56, %408, !dbg !26, !frisk.local_reduction !1
  %410 = extractelement <4 x float> %330, i64 3, !dbg !25
  %411 = call float @llvm.amdgcn.exp2.f32(float %410), !dbg !25
  %412 = insertelement <4 x float> %409, float %411, i64 3, !dbg !27
  %math.result296 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %412) #9, !dbg !27, !frisk.register_lifetimes !1
  %413 = fptrunc <4 x float> %math.result296 to <4 x half>, !dbg !27
  %math.packed297 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %413) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce61 = fadd float %local.reduce57, %411, !dbg !26, !frisk.local_reduction !1
  %414 = extractelement <4 x float> %331, i64 0, !dbg !25
  %415 = call float @llvm.amdgcn.exp2.f32(float %414), !dbg !25
  %416 = insertelement <4 x float> poison, float %415, i64 0, !dbg !27
  %local.reduce62 = fadd float %local.reduce58, %415, !dbg !26, !frisk.local_reduction !1
  %417 = extractelement <4 x float> %331, i64 1, !dbg !25
  %418 = call float @llvm.amdgcn.exp2.f32(float %417), !dbg !25
  %419 = insertelement <4 x float> %416, float %418, i64 1, !dbg !27
  %local.reduce63 = fadd float %local.reduce59, %418, !dbg !26, !frisk.local_reduction !1
  %420 = extractelement <4 x float> %331, i64 2, !dbg !25
  %421 = call float @llvm.amdgcn.exp2.f32(float %420), !dbg !25
  %422 = insertelement <4 x float> %419, float %421, i64 2, !dbg !27
  %local.reduce64 = fadd float %local.reduce60, %421, !dbg !26, !frisk.local_reduction !1
  %423 = extractelement <4 x float> %331, i64 3, !dbg !25
  %424 = call float @llvm.amdgcn.exp2.f32(float %423), !dbg !25
  %425 = insertelement <4 x float> %422, float %424, i64 3, !dbg !27
  %math.result298 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %425) #9, !dbg !27, !frisk.register_lifetimes !1
  %426 = fptrunc <4 x float> %math.result298 to <4 x half>, !dbg !27
  %math.packed299 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %426) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce65 = fadd float %local.reduce61, %424, !dbg !26, !frisk.local_reduction !1
  %427 = extractelement <4 x float> %332, i64 0, !dbg !25
  %428 = call float @llvm.amdgcn.exp2.f32(float %427), !dbg !25
  %429 = insertelement <4 x float> poison, float %428, i64 0, !dbg !27
  %local.reduce66 = fadd float %local.reduce62, %428, !dbg !26, !frisk.local_reduction !1
  %430 = extractelement <4 x float> %332, i64 1, !dbg !25
  %431 = call float @llvm.amdgcn.exp2.f32(float %430), !dbg !25
  %432 = insertelement <4 x float> %429, float %431, i64 1, !dbg !27
  %local.reduce67 = fadd float %local.reduce63, %431, !dbg !26, !frisk.local_reduction !1
  %433 = extractelement <4 x float> %332, i64 2, !dbg !25
  %434 = call float @llvm.amdgcn.exp2.f32(float %433), !dbg !25
  %435 = insertelement <4 x float> %432, float %434, i64 2, !dbg !27
  %local.reduce68 = fadd float %local.reduce64, %434, !dbg !26, !frisk.local_reduction !1
  %436 = extractelement <4 x float> %332, i64 3, !dbg !25
  %437 = call float @llvm.amdgcn.exp2.f32(float %436), !dbg !25
  %438 = insertelement <4 x float> %435, float %437, i64 3, !dbg !27
  %math.result300 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %438) #9, !dbg !27, !frisk.register_lifetimes !1
  %439 = fptrunc <4 x float> %math.result300 to <4 x half>, !dbg !27
  %math.packed301 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %439) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce69 = fadd float %local.reduce65, %437, !dbg !26, !frisk.local_reduction !1
  %440 = extractelement <4 x float> %333, i64 0, !dbg !25
  %441 = call float @llvm.amdgcn.exp2.f32(float %440), !dbg !25
  %442 = insertelement <4 x float> poison, float %441, i64 0, !dbg !27
  %local.reduce70 = fadd float %local.reduce66, %441, !dbg !26, !frisk.local_reduction !1
  %443 = extractelement <4 x float> %333, i64 1, !dbg !25
  %444 = call float @llvm.amdgcn.exp2.f32(float %443), !dbg !25
  %445 = insertelement <4 x float> %442, float %444, i64 1, !dbg !27
  %local.reduce71 = fadd float %local.reduce67, %444, !dbg !26, !frisk.local_reduction !1
  %local.reduce74 = fadd float %local.reduce70, %local.reduce71, !dbg !26, !frisk.local_reduction !1
  %446 = extractelement <4 x float> %333, i64 2, !dbg !25
  %447 = call float @llvm.amdgcn.exp2.f32(float %446), !dbg !25
  %448 = insertelement <4 x float> %445, float %447, i64 2, !dbg !27
  %local.reduce72 = fadd float %local.reduce68, %447, !dbg !26, !frisk.local_reduction !1
  %449 = extractelement <4 x float> %333, i64 3, !dbg !25
  %450 = call float @llvm.amdgcn.exp2.f32(float %449), !dbg !25
  %451 = insertelement <4 x float> %448, float %450, i64 3, !dbg !27
  %math.result302 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %451) #9, !dbg !27, !frisk.register_lifetimes !1
  %452 = fptrunc <4 x float> %math.result302 to <4 x half>, !dbg !27
  %math.packed303 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %452) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce73 = fadd float %local.reduce69, %450, !dbg !26, !frisk.local_reduction !1
  %local.reduce75 = fadd float %local.reduce72, %local.reduce73, !dbg !26, !frisk.local_reduction !1
  %453 = fadd float %local.reduce74, %local.reduce75, !dbg !26, !frisk.local_reduction !1
  %454 = fadd float %453, %348, !dbg !26
  %455 = insertelement <1 x float> poison, float %454, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local752 = shl nuw nsw i32 %address.thread350, 3, !dbg !13
  %address.local753 = and i32 %address.local752, 120, !dbg !13
  %address.local754 = shl nuw nsw i32 %address.thread350, 5, !dbg !13
  %address.local755 = and i32 %address.local754, 1536, !dbg !13
  %address.local756 = or disjoint i32 %address.local753, %address.local755, !dbg !13
  %address.local757 = zext nneg i32 %address.local756 to i64, !dbg !13
  %address.local758 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local757, !dbg !13
  %456 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local758, i32 0) #0, !dbg !13
  %457 = bitcast <4 x i32> %456 to <8 x half>, !dbg !13
  %address.local759 = and i64 %address.local351, 60, !dbg !13
  %address.local760 = add nuw nsw i64 %address.local759, %address.local351, !dbg !13
  %address.local761 = shl nuw nsw i64 %address.local760, 3, !dbg !13
  %address.local762 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local761, !dbg !13
  %address.local763 = lshr i64 %address.local386, 4, !dbg !13
  %address.local764 = mul nuw nsw i64 %address.local763, 384, !dbg !13
  %address.local765 = getelementptr [2 x i8], ptr addrspace(3) %address.local762, i64 %address.local764, !dbg !13
  %address.local766 = mul nsw i64 %address.local373, -32, !dbg !13
  %address.local767 = getelementptr [2 x i8], ptr addrspace(3) %address.local765, i64 %address.local766, !dbg !13
  %address.local768 = getelementptr i8, ptr addrspace(3) %address.local767, i64 256, !dbg !13
  %458 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local768, i32 0) #0, !dbg !13
  %459 = bitcast <4 x i32> %458 to <8 x half>, !dbg !13
  %address.local769 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local761, !dbg !13
  %address.local770 = getelementptr [2 x i8], ptr addrspace(3) %address.local769, i64 %address.local764, !dbg !13
  %address.local771 = getelementptr [2 x i8], ptr addrspace(3) %address.local770, i64 %address.local766, !dbg !13
  %address.local772 = getelementptr i8, ptr addrspace(3) %address.local771, i64 512, !dbg !13
  %460 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local772, i32 0) #0, !dbg !13
  %461 = bitcast <4 x i32> %460 to <8 x half>, !dbg !13
  %address.local773 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local761, !dbg !13
  %address.local774 = getelementptr [2 x i8], ptr addrspace(3) %address.local773, i64 %address.local764, !dbg !13
  %address.local775 = getelementptr [2 x i8], ptr addrspace(3) %address.local774, i64 %address.local766, !dbg !13
  %address.local776 = getelementptr i8, ptr addrspace(3) %address.local775, i64 768, !dbg !13
  %462 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local776, i32 0) #0, !dbg !13
  %463 = bitcast <4 x i32> %462 to <8 x half>, !dbg !13
  %address.local777 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local757, !dbg !13
  %address.local778 = getelementptr i8, ptr addrspace(3) %address.local777, i64 4096, !dbg !13
  %464 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #0, !dbg !13
  %465 = bitcast <4 x i32> %464 to <8 x half>, !dbg !13
  %address.local779 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local761, !dbg !13
  %address.local780 = getelementptr [2 x i8], ptr addrspace(3) %address.local779, i64 %address.local764, !dbg !13
  %address.local781 = getelementptr [2 x i8], ptr addrspace(3) %address.local780, i64 %address.local766, !dbg !13
  %address.local782 = getelementptr i8, ptr addrspace(3) %address.local781, i64 4352, !dbg !13
  %466 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #0, !dbg !13
  %467 = bitcast <4 x i32> %466 to <8 x half>, !dbg !13
  %address.local783 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local761, !dbg !13
  %address.local784 = getelementptr [2 x i8], ptr addrspace(3) %address.local783, i64 %address.local764, !dbg !13
  %address.local785 = getelementptr [2 x i8], ptr addrspace(3) %address.local784, i64 %address.local766, !dbg !13
  %address.local786 = getelementptr i8, ptr addrspace(3) %address.local785, i64 4608, !dbg !13
  %468 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local786, i32 0) #0, !dbg !13
  %469 = bitcast <4 x i32> %468 to <8 x half>, !dbg !13
  %address.local787 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local761, !dbg !13
  %address.local788 = getelementptr [2 x i8], ptr addrspace(3) %address.local787, i64 %address.local764, !dbg !13
  %address.local789 = getelementptr [2 x i8], ptr addrspace(3) %address.local788, i64 %address.local766, !dbg !13
  %address.local790 = getelementptr i8, ptr addrspace(3) %address.local789, i64 4864, !dbg !13
  %470 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local790, i32 0) #0, !dbg !13
  %471 = bitcast <4 x i32> %470 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %472 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local356), !dbg !15
  %473 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx10511052 = or disjoint i64 %110, %address.local750, !dbg !15
  %.idx10511052.tr = trunc nuw i64 %.idx10511052 to i32, !dbg !15
  %dma.buffer.offset1009 = shl nuw nsw i32 %.idx10511052.tr, 1, !dbg !15
  %474 = insertelement <2 x i64> poison, i64 %473, i64 0, !dbg !15
  %475 = bitcast <2 x i64> %474 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1010 = shufflevector <4 x i32> %475, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1011 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %472), !dbg !15
  %476 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1011, i32 %dma.buffer.offset1009, <4 x i32> %dma.buffer.rsrc1010) #10, !dbg !15
  %477 = shufflevector <8 x half> %457, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %477, <4 x float> %346) #9, !dbg !13
  %479 = shufflevector <8 x half> %459, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %479, <4 x float> %478) #9, !dbg !13
  %481 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %481, <4 x float> %480) #9, !dbg !13
  %483 = shufflevector <8 x half> %463, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %483, <4 x float> %482) #9, !dbg !13
  %485 = shufflevector <8 x half> %465, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %485, <4 x float> %484) #9, !dbg !13
  %487 = shufflevector <8 x half> %467, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %487, <4 x float> %486) #9, !dbg !13
  %489 = shufflevector <8 x half> %469, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %489, <4 x float> %488) #9, !dbg !13
  %491 = shufflevector <8 x half> %471, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %491, <4 x float> %490) #9, !dbg !13
  %493 = shufflevector <8 x half> %457, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %493, <4 x float> %345) #9, !dbg !13
  %495 = shufflevector <8 x half> %459, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %495, <4 x float> %494) #9, !dbg !13
  %497 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %497, <4 x float> %496) #9, !dbg !13
  %499 = shufflevector <8 x half> %463, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %499, <4 x float> %498) #9, !dbg !13
  %501 = shufflevector <8 x half> %465, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %501, <4 x float> %500) #9, !dbg !13
  %503 = shufflevector <8 x half> %467, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %503, <4 x float> %502) #9, !dbg !13
  %505 = shufflevector <8 x half> %469, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %505, <4 x float> %504) #9, !dbg !13
  %507 = shufflevector <8 x half> %471, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %507, <4 x float> %506) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %509 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local758, i32 0) #0, !dbg !13
  %510 = bitcast <4 x i32> %509 to <8 x half>, !dbg !13
  %511 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local768, i32 0) #0, !dbg !13
  %512 = bitcast <4 x i32> %511 to <8 x half>, !dbg !13
  %513 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local772, i32 0) #0, !dbg !13
  %514 = bitcast <4 x i32> %513 to <8 x half>, !dbg !13
  %515 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local776, i32 0) #0, !dbg !13
  %516 = bitcast <4 x i32> %515 to <8 x half>, !dbg !13
  %517 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #0, !dbg !13
  %518 = bitcast <4 x i32> %517 to <8 x half>, !dbg !13
  %519 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #0, !dbg !13
  %520 = bitcast <4 x i32> %519 to <8 x half>, !dbg !13
  %521 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local786, i32 0) #0, !dbg !13
  %522 = bitcast <4 x i32> %521 to <8 x half>, !dbg !13
  %523 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local790, i32 0) #0, !dbg !13
  %524 = bitcast <4 x i32> %523 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %525 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local356), !dbg !15
  %526 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx10531054 = or disjoint i64 %156, %address.local750, !dbg !15
  %.idx10531054.tr = trunc nuw i64 %.idx10531054 to i32, !dbg !15
  %dma.buffer.offset1006 = shl nuw nsw i32 %.idx10531054.tr, 1, !dbg !15
  %527 = insertelement <2 x i64> poison, i64 %526, i64 0, !dbg !15
  %528 = bitcast <2 x i64> %527 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1007 = shufflevector <4 x i32> %528, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1008 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %525), !dbg !15
  %529 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1008, i32 %dma.buffer.offset1006, <4 x i32> %dma.buffer.rsrc1007) #10, !dbg !15
  %530 = shufflevector <8 x half> %510, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %530, <4 x float> %344) #9, !dbg !13
  %532 = shufflevector <8 x half> %512, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %532, <4 x float> %531) #9, !dbg !13
  %534 = shufflevector <8 x half> %514, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %534, <4 x float> %533) #9, !dbg !13
  %536 = shufflevector <8 x half> %516, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %536, <4 x float> %535) #9, !dbg !13
  %538 = shufflevector <8 x half> %518, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %538, <4 x float> %537) #9, !dbg !13
  %540 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %540, <4 x float> %539) #9, !dbg !13
  %542 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %542, <4 x float> %541) #9, !dbg !13
  %544 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %544, <4 x float> %543) #9, !dbg !13
  %546 = shufflevector <8 x half> %510, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %546, <4 x float> %343) #9, !dbg !13
  %548 = shufflevector <8 x half> %512, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %548, <4 x float> %547) #9, !dbg !13
  %550 = shufflevector <8 x half> %514, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %550, <4 x float> %549) #9, !dbg !13
  %552 = shufflevector <8 x half> %516, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %552, <4 x float> %551) #9, !dbg !13
  %554 = shufflevector <8 x half> %518, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %554, <4 x float> %553) #9, !dbg !13
  %556 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %556, <4 x float> %555) #9, !dbg !13
  %558 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %558, <4 x float> %557) #9, !dbg !13
  %560 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %560, <4 x float> %559) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %562 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local758, i32 0) #0, !dbg !13
  %563 = bitcast <4 x i32> %562 to <8 x half>, !dbg !13
  %564 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local768, i32 0) #0, !dbg !13
  %565 = bitcast <4 x i32> %564 to <8 x half>, !dbg !13
  %566 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local772, i32 0) #0, !dbg !13
  %567 = bitcast <4 x i32> %566 to <8 x half>, !dbg !13
  %568 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local776, i32 0) #0, !dbg !13
  %569 = bitcast <4 x i32> %568 to <8 x half>, !dbg !13
  %570 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #0, !dbg !13
  %571 = bitcast <4 x i32> %570 to <8 x half>, !dbg !13
  %572 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #0, !dbg !13
  %573 = bitcast <4 x i32> %572 to <8 x half>, !dbg !13
  %574 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local786, i32 0) #0, !dbg !13
  %575 = bitcast <4 x i32> %574 to <8 x half>, !dbg !13
  %576 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local790, i32 0) #0, !dbg !13
  %577 = bitcast <4 x i32> %576 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local356), !dbg !15
  %579 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx10551056 = or disjoint i64 %194, %address.local750, !dbg !15
  %.idx10551056.tr = trunc nuw i64 %.idx10551056 to i32, !dbg !15
  %dma.buffer.offset = shl nuw nsw i32 %.idx10551056.tr, 1, !dbg !15
  %580 = insertelement <2 x i64> poison, i64 %579, i64 0, !dbg !15
  %581 = bitcast <2 x i64> %580 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc = shufflevector <4 x i32> %581, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %578), !dbg !15
  %582 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !15
  %583 = shufflevector <8 x half> %563, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %584 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %583, <4 x float> %342) #9, !dbg !13
  %585 = shufflevector <8 x half> %565, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %585, <4 x float> %584) #9, !dbg !13
  %587 = shufflevector <8 x half> %567, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %587, <4 x float> %586) #9, !dbg !13
  %589 = shufflevector <8 x half> %569, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %589, <4 x float> %588) #9, !dbg !13
  %591 = shufflevector <8 x half> %571, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %591, <4 x float> %590) #9, !dbg !13
  %593 = shufflevector <8 x half> %573, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %593, <4 x float> %592) #9, !dbg !13
  %595 = shufflevector <8 x half> %575, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %595, <4 x float> %594) #9, !dbg !13
  %597 = shufflevector <8 x half> %577, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %597, <4 x float> %596) #9, !dbg !13
  %599 = shufflevector <8 x half> %563, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %599, <4 x float> %341) #9, !dbg !13
  %601 = shufflevector <8 x half> %565, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %601, <4 x float> %600) #9, !dbg !13
  %603 = shufflevector <8 x half> %567, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %604 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %603, <4 x float> %602) #9, !dbg !13
  %605 = shufflevector <8 x half> %569, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %606 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %605, <4 x float> %604) #9, !dbg !13
  %607 = shufflevector <8 x half> %571, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %607, <4 x float> %606) #9, !dbg !13
  %609 = shufflevector <8 x half> %573, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %609, <4 x float> %608) #9, !dbg !13
  %611 = shufflevector <8 x half> %575, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %611, <4 x float> %610) #9, !dbg !13
  %613 = shufflevector <8 x half> %577, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %613, <4 x float> %612) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %615 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local758, i32 0) #0, !dbg !13
  %616 = bitcast <4 x i32> %615 to <8 x half>, !dbg !13
  %617 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local768, i32 0) #0, !dbg !13
  %618 = bitcast <4 x i32> %617 to <8 x half>, !dbg !13
  %619 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local772, i32 0) #0, !dbg !13
  %620 = bitcast <4 x i32> %619 to <8 x half>, !dbg !13
  %621 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local776, i32 0) #0, !dbg !13
  %622 = bitcast <4 x i32> %621 to <8 x half>, !dbg !13
  %623 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #0, !dbg !13
  %624 = bitcast <4 x i32> %623 to <8 x half>, !dbg !13
  %625 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #0, !dbg !13
  %626 = bitcast <4 x i32> %625 to <8 x half>, !dbg !13
  %627 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local786, i32 0) #0, !dbg !13
  %628 = bitcast <4 x i32> %627 to <8 x half>, !dbg !13
  %629 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local790, i32 0) #0, !dbg !13
  %630 = bitcast <4 x i32> %629 to <8 x half>, !dbg !13
  %631 = shufflevector <8 x half> %616, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %631, <4 x float> %340) #9, !dbg !13
  %633 = shufflevector <8 x half> %618, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %633, <4 x float> %632) #9, !dbg !13
  %635 = shufflevector <8 x half> %620, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %635, <4 x float> %634) #9, !dbg !13
  %637 = shufflevector <8 x half> %622, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %637, <4 x float> %636) #9, !dbg !13
  %639 = shufflevector <8 x half> %624, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %639, <4 x float> %638) #9, !dbg !13
  %641 = shufflevector <8 x half> %626, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %641, <4 x float> %640) #9, !dbg !13
  %643 = shufflevector <8 x half> %628, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %643, <4 x float> %642) #9, !dbg !13
  %645 = shufflevector <8 x half> %630, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %645, <4 x float> %644) #9, !dbg !13
  %647 = shufflevector <8 x half> %616, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %647, <4 x float> %339) #9, !dbg !13
  %649 = shufflevector <8 x half> %618, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %649, <4 x float> %648) #9, !dbg !13
  %651 = shufflevector <8 x half> %620, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %651, <4 x float> %650) #9, !dbg !13
  %653 = shufflevector <8 x half> %622, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %653, <4 x float> %652) #9, !dbg !13
  %655 = shufflevector <8 x half> %624, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %655, <4 x float> %654) #9, !dbg !13
  %657 = shufflevector <8 x half> %626, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %657, <4 x float> %656) #9, !dbg !13
  %659 = shufflevector <8 x half> %628, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %659, <4 x float> %658) #9, !dbg !13
  %661 = shufflevector <8 x half> %630, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %661, <4 x float> %660) #9, !dbg !13
  %663 = shufflevector <4 x float> %492, <4 x float> %508, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %664 = shufflevector <4 x float> %545, <4 x float> %561, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %665 = shufflevector <4 x float> %598, <4 x float> %614, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %666 = shufflevector <4 x float> %646, <4 x float> %662, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %667 = add nuw nsw i64 %84, 1, !dbg !7
  br label %83, !dbg !7

668:                                              ; preds = %83
  %lane.sum.local = extractelement <1 x float> %88, i64 0, !dbg !7
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !7
  %669 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %670 = xor i32 %669, 64, !dbg !7
  %671 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %671), !dbg !7
  %672 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %672, !dbg !7
  %673 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %674 = xor i32 %673, 128, !dbg !7
  %675 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle108 = call i32 @llvm.amdgcn.ds.bpermute(i32 %674, i32 %675), !dbg !7
  %676 = bitcast i32 %lane.sum.shuffle108 to float, !dbg !7
  %lane.sum.final109 = fadd float %lane.sum.final, %676, !dbg !7
  %677 = insertelement <1 x float> poison, float %lane.sum.final109, i64 0, !dbg !7
  %678 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %678, 0, !dbg !7
  %679 = and i32 %31, 32, !dbg !7
  %.not36 = icmp eq i32 %679, 0, !dbg !7
  %680 = extractelement <8 x float> %98, i64 0, !dbg !7
  %681 = extractelement <8 x float> %98, i64 1, !dbg !7
  %682 = extractelement <8 x float> %98, i64 2, !dbg !7
  %683 = extractelement <8 x float> %98, i64 3, !dbg !7
  %684 = select i1 %.not, float %681, float %680, !dbg !7
  %685 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %686 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %685), !dbg !7
  %687 = add nuw nsw i32 %686, 64, !dbg !7
  %688 = xor i32 %686, 16, !dbg !7
  %689 = and i32 %687, 192, !dbg !7
  %690 = icmp samesign ult i32 %688, %689, !dbg !7
  %691 = select i1 %690, i32 %688, i32 %686, !dbg !7
  %692 = shl nuw nsw i32 %691, 2, !dbg !7
  %693 = bitcast float %684 to i32, !dbg !7
  %694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %693), !dbg !7
  %695 = bitcast i32 %694 to float, !dbg !7
  %696 = select i1 %.not, float %680, float %695, !dbg !7
  %697 = select i1 %.not, float %695, float %681, !dbg !7
  %698 = select i1 %.not, float %683, float %682, !dbg !7
  %699 = bitcast float %698 to i32, !dbg !7
  %700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %699), !dbg !7
  %701 = bitcast i32 %700 to float, !dbg !7
  %702 = select i1 %.not, float %682, float %701, !dbg !7
  %703 = select i1 %.not, float %701, float %683, !dbg !7
  %704 = select i1 %.not36, float %702, float %696, !dbg !7
  %705 = xor i32 %686, 32, !dbg !7
  %706 = icmp samesign ult i32 %705, %689, !dbg !7
  %707 = select i1 %706, i32 %705, i32 %686, !dbg !7
  %708 = shl nuw nsw i32 %707, 2, !dbg !7
  %709 = bitcast float %704 to i32, !dbg !7
  %710 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %709), !dbg !7
  %711 = bitcast i32 %710 to float, !dbg !7
  %712 = select i1 %.not36, float %696, float %711, !dbg !7
  %713 = select i1 %.not36, float %711, float %702, !dbg !7
  %714 = select i1 %.not36, float %703, float %697, !dbg !7
  %715 = bitcast float %714 to i32, !dbg !7
  %716 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %715), !dbg !7
  %717 = bitcast i32 %716 to float, !dbg !7
  %718 = select i1 %.not36, float %697, float %717, !dbg !7
  %719 = select i1 %.not36, float %717, float %703, !dbg !7
  %720 = extractelement <8 x float> %95, i64 0, !dbg !7
  %721 = extractelement <8 x float> %95, i64 1, !dbg !7
  %722 = extractelement <8 x float> %95, i64 2, !dbg !7
  %723 = extractelement <8 x float> %95, i64 3, !dbg !7
  %724 = select i1 %.not, float %721, float %720, !dbg !7
  %725 = bitcast float %724 to i32, !dbg !7
  %726 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %725), !dbg !7
  %727 = bitcast i32 %726 to float, !dbg !7
  %728 = select i1 %.not, float %720, float %727, !dbg !7
  %729 = select i1 %.not, float %727, float %721, !dbg !7
  %730 = select i1 %.not, float %723, float %722, !dbg !7
  %731 = bitcast float %730 to i32, !dbg !7
  %732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %731), !dbg !7
  %733 = bitcast i32 %732 to float, !dbg !7
  %734 = select i1 %.not, float %722, float %733, !dbg !7
  %735 = select i1 %.not, float %733, float %723, !dbg !7
  %736 = select i1 %.not36, float %734, float %728, !dbg !7
  %737 = bitcast float %736 to i32, !dbg !7
  %738 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %737), !dbg !7
  %739 = bitcast i32 %738 to float, !dbg !7
  %740 = select i1 %.not36, float %728, float %739, !dbg !7
  %741 = select i1 %.not36, float %739, float %734, !dbg !7
  %742 = select i1 %.not36, float %735, float %729, !dbg !7
  %743 = bitcast float %742 to i32, !dbg !7
  %744 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %743), !dbg !7
  %745 = bitcast i32 %744 to float, !dbg !7
  %746 = select i1 %.not36, float %729, float %745, !dbg !7
  %747 = select i1 %.not36, float %745, float %735, !dbg !7
  %748 = extractelement <8 x float> %92, i64 0, !dbg !7
  %749 = extractelement <8 x float> %92, i64 1, !dbg !7
  %750 = extractelement <8 x float> %92, i64 2, !dbg !7
  %751 = extractelement <8 x float> %92, i64 3, !dbg !7
  %752 = select i1 %.not, float %749, float %748, !dbg !7
  %753 = bitcast float %752 to i32, !dbg !7
  %754 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %753), !dbg !7
  %755 = bitcast i32 %754 to float, !dbg !7
  %756 = select i1 %.not, float %748, float %755, !dbg !7
  %757 = select i1 %.not, float %755, float %749, !dbg !7
  %758 = select i1 %.not, float %751, float %750, !dbg !7
  %759 = bitcast float %758 to i32, !dbg !7
  %760 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %759), !dbg !7
  %761 = bitcast i32 %760 to float, !dbg !7
  %762 = select i1 %.not, float %750, float %761, !dbg !7
  %763 = select i1 %.not, float %761, float %751, !dbg !7
  %764 = select i1 %.not36, float %762, float %756, !dbg !7
  %765 = bitcast float %764 to i32, !dbg !7
  %766 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %765), !dbg !7
  %767 = bitcast i32 %766 to float, !dbg !7
  %768 = select i1 %.not36, float %756, float %767, !dbg !7
  %769 = select i1 %.not36, float %767, float %762, !dbg !7
  %770 = select i1 %.not36, float %763, float %757, !dbg !7
  %771 = bitcast float %770 to i32, !dbg !7
  %772 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %771), !dbg !7
  %773 = bitcast i32 %772 to float, !dbg !7
  %774 = select i1 %.not36, float %757, float %773, !dbg !7
  %775 = select i1 %.not36, float %773, float %763, !dbg !7
  %776 = extractelement <8 x float> %87, i64 0, !dbg !7
  %777 = extractelement <8 x float> %87, i64 1, !dbg !7
  %778 = extractelement <8 x float> %87, i64 2, !dbg !7
  %779 = extractelement <8 x float> %87, i64 3, !dbg !7
  %780 = select i1 %.not, float %777, float %776, !dbg !7
  %781 = bitcast float %780 to i32, !dbg !7
  %782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %781), !dbg !7
  %783 = bitcast i32 %782 to float, !dbg !7
  %784 = select i1 %.not, float %776, float %783, !dbg !7
  %785 = select i1 %.not, float %783, float %777, !dbg !7
  %786 = select i1 %.not, float %779, float %778, !dbg !7
  %787 = bitcast float %786 to i32, !dbg !7
  %788 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %787), !dbg !7
  %789 = bitcast i32 %788 to float, !dbg !7
  %790 = select i1 %.not, float %778, float %789, !dbg !7
  %791 = select i1 %.not, float %789, float %779, !dbg !7
  %792 = select i1 %.not36, float %790, float %784, !dbg !7
  %793 = bitcast float %792 to i32, !dbg !7
  %794 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %793), !dbg !7
  %795 = select i1 %.not36, float %791, float %785, !dbg !7
  %796 = bitcast float %795 to i32, !dbg !7
  %797 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %796), !dbg !7
  %798 = shufflevector <1 x float> %677, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %799 = extractelement <8 x float> %98, i64 4, !dbg !7
  %800 = extractelement <8 x float> %98, i64 5, !dbg !7
  %801 = extractelement <8 x float> %98, i64 6, !dbg !7
  %802 = extractelement <8 x float> %98, i64 7, !dbg !7
  %803 = select i1 %.not, float %800, float %799, !dbg !7
  %804 = bitcast float %803 to i32, !dbg !7
  %805 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %804), !dbg !7
  %806 = bitcast i32 %805 to float, !dbg !7
  %807 = select i1 %.not, float %799, float %806, !dbg !7
  %808 = select i1 %.not, float %806, float %800, !dbg !7
  %809 = select i1 %.not, float %802, float %801, !dbg !7
  %810 = bitcast float %809 to i32, !dbg !7
  %811 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %810), !dbg !7
  %812 = bitcast i32 %811 to float, !dbg !7
  %813 = select i1 %.not, float %801, float %812, !dbg !7
  %814 = select i1 %.not, float %812, float %802, !dbg !7
  %815 = select i1 %.not36, float %813, float %807, !dbg !7
  %816 = bitcast float %815 to i32, !dbg !7
  %817 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %816), !dbg !7
  %818 = bitcast i32 %817 to float, !dbg !7
  %819 = select i1 %.not36, float %807, float %818, !dbg !7
  %820 = select i1 %.not36, float %818, float %813, !dbg !7
  %821 = select i1 %.not36, float %814, float %808, !dbg !7
  %822 = bitcast float %821 to i32, !dbg !7
  %823 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %822), !dbg !7
  %824 = bitcast i32 %823 to float, !dbg !7
  %825 = select i1 %.not36, float %808, float %824, !dbg !7
  %826 = select i1 %.not36, float %824, float %814, !dbg !7
  %827 = extractelement <8 x float> %95, i64 4, !dbg !7
  %828 = extractelement <8 x float> %95, i64 5, !dbg !7
  %829 = extractelement <8 x float> %95, i64 6, !dbg !7
  %830 = extractelement <8 x float> %95, i64 7, !dbg !7
  %831 = select i1 %.not, float %828, float %827, !dbg !7
  %832 = bitcast float %831 to i32, !dbg !7
  %833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %832), !dbg !7
  %834 = bitcast i32 %833 to float, !dbg !7
  %835 = select i1 %.not, float %827, float %834, !dbg !7
  %836 = select i1 %.not, float %834, float %828, !dbg !7
  %837 = select i1 %.not, float %830, float %829, !dbg !7
  %838 = bitcast float %837 to i32, !dbg !7
  %839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %838), !dbg !7
  %840 = bitcast i32 %839 to float, !dbg !7
  %841 = select i1 %.not, float %829, float %840, !dbg !7
  %842 = select i1 %.not, float %840, float %830, !dbg !7
  %843 = select i1 %.not36, float %841, float %835, !dbg !7
  %844 = bitcast float %843 to i32, !dbg !7
  %845 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %844), !dbg !7
  %846 = bitcast i32 %845 to float, !dbg !7
  %847 = select i1 %.not36, float %835, float %846, !dbg !7
  %848 = select i1 %.not36, float %846, float %841, !dbg !7
  %849 = select i1 %.not36, float %842, float %836, !dbg !7
  %850 = bitcast float %849 to i32, !dbg !7
  %851 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %850), !dbg !7
  %852 = bitcast i32 %851 to float, !dbg !7
  %853 = select i1 %.not36, float %836, float %852, !dbg !7
  %854 = select i1 %.not36, float %852, float %842, !dbg !7
  %855 = extractelement <8 x float> %92, i64 4, !dbg !7
  %856 = extractelement <8 x float> %92, i64 5, !dbg !7
  %857 = extractelement <8 x float> %92, i64 6, !dbg !7
  %858 = extractelement <8 x float> %92, i64 7, !dbg !7
  %859 = select i1 %.not, float %856, float %855, !dbg !7
  %860 = bitcast float %859 to i32, !dbg !7
  %861 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %860), !dbg !7
  %862 = bitcast i32 %861 to float, !dbg !7
  %863 = select i1 %.not, float %855, float %862, !dbg !7
  %864 = select i1 %.not, float %862, float %856, !dbg !7
  %865 = select i1 %.not, float %858, float %857, !dbg !7
  %866 = bitcast float %865 to i32, !dbg !7
  %867 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %866), !dbg !7
  %868 = bitcast i32 %867 to float, !dbg !7
  %869 = select i1 %.not, float %857, float %868, !dbg !7
  %870 = select i1 %.not, float %868, float %858, !dbg !7
  %871 = select i1 %.not36, float %869, float %863, !dbg !7
  %872 = bitcast float %871 to i32, !dbg !7
  %873 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %872), !dbg !7
  %874 = bitcast i32 %873 to float, !dbg !7
  %875 = select i1 %.not36, float %863, float %874, !dbg !7
  %876 = select i1 %.not36, float %874, float %869, !dbg !7
  %877 = select i1 %.not36, float %870, float %864, !dbg !7
  %878 = bitcast float %877 to i32, !dbg !7
  %879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %878), !dbg !7
  %880 = bitcast i32 %879 to float, !dbg !7
  %881 = select i1 %.not36, float %864, float %880, !dbg !7
  %882 = select i1 %.not36, float %880, float %870, !dbg !7
  %883 = extractelement <8 x float> %87, i64 4, !dbg !7
  %884 = extractelement <8 x float> %87, i64 5, !dbg !7
  %885 = extractelement <8 x float> %87, i64 6, !dbg !7
  %886 = extractelement <8 x float> %87, i64 7, !dbg !7
  %887 = select i1 %.not, float %884, float %883, !dbg !7
  %888 = bitcast float %887 to i32, !dbg !7
  %889 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %888), !dbg !7
  %890 = bitcast i32 %889 to float, !dbg !7
  %891 = select i1 %.not, float %883, float %890, !dbg !7
  %892 = select i1 %.not, float %890, float %884, !dbg !7
  %893 = select i1 %.not, float %886, float %885, !dbg !7
  %894 = bitcast float %893 to i32, !dbg !7
  %895 = call i32 @llvm.amdgcn.ds.bpermute(i32 %692, i32 %894), !dbg !7
  %896 = bitcast i32 %895 to float, !dbg !7
  %897 = select i1 %.not, float %885, float %896, !dbg !7
  %898 = select i1 %.not, float %896, float %886, !dbg !7
  %899 = select i1 %.not36, float %897, float %891, !dbg !7
  %900 = bitcast float %899 to i32, !dbg !7
  %901 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %900), !dbg !7
  %902 = select i1 %.not36, float %898, float %892, !dbg !7
  %903 = bitcast float %902 to i32, !dbg !7
  %904 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %903), !dbg !7
  %905 = bitcast i32 %901 to float, !dbg !7
  %906 = select i1 %.not36, float %891, float %905, !dbg !7
  %907 = insertelement <4 x float> poison, float %906, i64 0, !dbg !16
  %908 = bitcast i32 %904 to float, !dbg !7
  %909 = select i1 %.not36, float %892, float %908, !dbg !7
  %910 = insertelement <4 x float> %907, float %909, i64 1, !dbg !16
  %911 = select i1 %.not36, float %905, float %897, !dbg !7
  %912 = insertelement <4 x float> %910, float %911, i64 2, !dbg !16
  %913 = select i1 %.not36, float %908, float %898, !dbg !7
  %914 = insertelement <4 x float> %912, float %913, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %798, !dbg !16
  %normalized = fmul <4 x float> %914, %row.reciprocal, !dbg !16
  %915 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %916 = extractelement <4 x half> %915, i64 3, !dbg !17
  %917 = extractelement <4 x half> %915, i64 2, !dbg !17
  %918 = extractelement <4 x half> %915, i64 1, !dbg !17
  %919 = extractelement <4 x half> %915, i64 0, !dbg !17
  %920 = bitcast i32 %794 to float, !dbg !7
  %921 = select i1 %.not36, float %784, float %920, !dbg !7
  %922 = insertelement <4 x float> poison, float %921, i64 0, !dbg !16
  %923 = bitcast i32 %797 to float, !dbg !7
  %924 = select i1 %.not36, float %785, float %923, !dbg !7
  %925 = insertelement <4 x float> %922, float %924, i64 1, !dbg !16
  %926 = select i1 %.not36, float %920, float %790, !dbg !7
  %927 = insertelement <4 x float> %925, float %926, i64 2, !dbg !16
  %928 = select i1 %.not36, float %923, float %791, !dbg !7
  %929 = insertelement <4 x float> %927, float %928, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %929, %row.reciprocal, !dbg !16
  %930 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %931 = extractelement <4 x half> %930, i64 3, !dbg !17
  %932 = extractelement <4 x half> %930, i64 2, !dbg !17
  %933 = extractelement <4 x half> %930, i64 1, !dbg !17
  %934 = extractelement <4 x half> %930, i64 0, !dbg !17
  %935 = add nsw i64 %16, %22, !dbg !18
  %936 = shl i64 %935, 19, !dbg !18
  %937 = and i64 %936, 16252928, !dbg !18
  %address.thread794 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local795 = zext nneg i32 %address.thread794 to i64, !dbg !8
  %address.local796 = add i64 %25, %address.local795, !dbg !18
  %938 = lshr i64 %address.local795, 2, !dbg !18
  %address.local798 = and i64 %938, 112, !dbg !18
  %address.local799 = add i64 %address.local796, %address.local798, !dbg !18
  %address.local800 = and i64 %address.local795, 496, !dbg !18
  %address.local801 = sub i64 %address.local799, %address.local800, !dbg !18
  %address.local802 = add i64 %address.local801, %26, !dbg !18
  %address.local803 = shl nuw nsw i64 %address.local802, 7, !dbg !18
  %address.local804 = add nuw nsw i64 %937, %address.local803, !dbg !18
  %address.local805 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local804, !dbg !18
  %address.local806 = lshr i64 %address.local795, 2, !dbg !10
  %address.local807 = and i64 %address.local806, 12, !dbg !10
  %address.local808 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local805, i64 %address.local807, !dbg !18
  store half %934, ptr addrspace(1) %address.local808, align 2, !dbg !18
  %address.local809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local804, !dbg !18
  %address.local810 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local809, i64 %address.local807, !dbg !18
  %address.local811 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local810, i64 2, !dbg !18
  store half %933, ptr addrspace(1) %address.local811, align 2, !dbg !18
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local804, !dbg !18
  %address.local813 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local812, i64 %address.local807, !dbg !18
  %address.local814 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local813, i64 4, !dbg !18
  store half %932, ptr addrspace(1) %address.local814, align 2, !dbg !18
  %address.local815 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local804, !dbg !18
  %address.local816 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local815, i64 %address.local807, !dbg !18
  %address.local817 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local816, i64 6, !dbg !18
  store half %931, ptr addrspace(1) %address.local817, align 2, !dbg !18
  %address.local818 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local804, !dbg !18
  %address.local819 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local818, i64 %address.local807, !dbg !18
  %address.local820 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local819, i64 32, !dbg !18
  store half %919, ptr addrspace(1) %address.local820, align 2, !dbg !18
  %address.local821 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local804, !dbg !18
  %address.local822 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local821, i64 %address.local807, !dbg !18
  %address.local823 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local822, i64 34, !dbg !18
  store half %918, ptr addrspace(1) %address.local823, align 2, !dbg !18
  %address.local824 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local804, !dbg !18
  %address.local825 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local824, i64 %address.local807, !dbg !18
  %address.local826 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local825, i64 36, !dbg !18
  store half %917, ptr addrspace(1) %address.local826, align 2, !dbg !18
  %address.local827 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local804, !dbg !18
  %address.local828 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local827, i64 %address.local807, !dbg !18
  %address.local829 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local828, i64 38, !dbg !18
  store half %916, ptr addrspace(1) %address.local829, align 2, !dbg !18
  %939 = insertelement <4 x float> poison, float %875, i64 0, !dbg !16
  %940 = insertelement <4 x float> %939, float %881, i64 1, !dbg !16
  %941 = insertelement <4 x float> %940, float %876, i64 2, !dbg !16
  %942 = insertelement <4 x float> %941, float %882, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %798, !dbg !16
  %normalized39 = fmul <4 x float> %942, %row.reciprocal38, !dbg !16
  %943 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %944 = extractelement <4 x half> %943, i64 3, !dbg !17
  %945 = extractelement <4 x half> %943, i64 2, !dbg !17
  %946 = extractelement <4 x half> %943, i64 1, !dbg !17
  %947 = extractelement <4 x half> %943, i64 0, !dbg !17
  %948 = insertelement <4 x float> poison, float %768, i64 0, !dbg !16
  %949 = insertelement <4 x float> %948, float %774, i64 1, !dbg !16
  %950 = insertelement <4 x float> %949, float %769, i64 2, !dbg !16
  %951 = insertelement <4 x float> %950, float %775, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %951, %row.reciprocal38, !dbg !16
  %952 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %953 = extractelement <4 x half> %952, i64 3, !dbg !17
  %954 = extractelement <4 x half> %952, i64 2, !dbg !17
  %955 = extractelement <4 x half> %952, i64 1, !dbg !17
  %956 = extractelement <4 x half> %952, i64 0, !dbg !17
  %957 = shl nuw nsw i64 %24, 19, !dbg !18
  %address.thread830 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local831 = zext nneg i32 %address.thread830 to i64, !dbg !8
  %address.local832 = add i64 %25, %address.local831, !dbg !18
  %958 = lshr i64 %address.local831, 2, !dbg !18
  %address.local834 = and i64 %958, 112, !dbg !18
  %address.local835 = add i64 %address.local832, %address.local834, !dbg !18
  %address.local836 = and i64 %address.local831, 496, !dbg !18
  %address.local837 = sub i64 %address.local835, %address.local836, !dbg !18
  %address.local838 = add i64 %address.local837, %26, !dbg !18
  %address.local839 = shl nuw nsw i64 %address.local838, 7, !dbg !18
  %address.local840 = add nuw nsw i64 %957, %address.local839, !dbg !18
  %address.local841 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local840, !dbg !18
  %address.local842 = lshr i64 %address.local831, 2, !dbg !10
  %address.local843 = and i64 %address.local842, 12, !dbg !10
  %address.local844 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local841, i64 %address.local843, !dbg !18
  %address.local845 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local844, i64 64, !dbg !18
  store half %956, ptr addrspace(1) %address.local845, align 2, !dbg !18
  %address.local846 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local840, !dbg !18
  %address.local847 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local846, i64 %address.local843, !dbg !18
  %address.local848 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local847, i64 66, !dbg !18
  store half %955, ptr addrspace(1) %address.local848, align 2, !dbg !18
  %address.local849 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local840, !dbg !18
  %address.local850 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local849, i64 %address.local843, !dbg !18
  %address.local851 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local850, i64 68, !dbg !18
  store half %954, ptr addrspace(1) %address.local851, align 2, !dbg !18
  %address.local852 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local840, !dbg !18
  %address.local853 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local843, !dbg !18
  %address.local854 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local853, i64 70, !dbg !18
  store half %953, ptr addrspace(1) %address.local854, align 2, !dbg !18
  %address.local855 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local840, !dbg !18
  %address.local856 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local855, i64 %address.local843, !dbg !18
  %address.local857 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local856, i64 96, !dbg !18
  store half %947, ptr addrspace(1) %address.local857, align 2, !dbg !18
  %address.local858 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local840, !dbg !18
  %address.local859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local858, i64 %address.local843, !dbg !18
  %address.local860 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local859, i64 98, !dbg !18
  store half %946, ptr addrspace(1) %address.local860, align 2, !dbg !18
  %address.local861 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local840, !dbg !18
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local861, i64 %address.local843, !dbg !18
  %address.local863 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local862, i64 100, !dbg !18
  store half %945, ptr addrspace(1) %address.local863, align 2, !dbg !18
  %address.local864 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local840, !dbg !18
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local864, i64 %address.local843, !dbg !18
  %address.local866 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local865, i64 102, !dbg !18
  store half %944, ptr addrspace(1) %address.local866, align 2, !dbg !18
  %959 = insertelement <4 x float> poison, float %847, i64 0, !dbg !16
  %960 = insertelement <4 x float> %959, float %853, i64 1, !dbg !16
  %961 = insertelement <4 x float> %960, float %848, i64 2, !dbg !16
  %962 = insertelement <4 x float> %961, float %854, i64 3, !dbg !16
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %798, !dbg !16
  %normalized42 = fmul <4 x float> %962, %row.reciprocal41, !dbg !16
  %963 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %964 = extractelement <4 x half> %963, i64 3, !dbg !17
  %965 = extractelement <4 x half> %963, i64 2, !dbg !17
  %966 = extractelement <4 x half> %963, i64 1, !dbg !17
  %967 = extractelement <4 x half> %963, i64 0, !dbg !17
  %968 = insertelement <4 x float> poison, float %740, i64 0, !dbg !16
  %969 = insertelement <4 x float> %968, float %746, i64 1, !dbg !16
  %970 = insertelement <4 x float> %969, float %741, i64 2, !dbg !16
  %971 = insertelement <4 x float> %970, float %747, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %971, %row.reciprocal41, !dbg !16
  %972 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %973 = extractelement <4 x half> %972, i64 3, !dbg !17
  %974 = extractelement <4 x half> %972, i64 2, !dbg !17
  %975 = extractelement <4 x half> %972, i64 1, !dbg !17
  %976 = extractelement <4 x half> %972, i64 0, !dbg !17
  %977 = shl nuw nsw i64 %24, 19, !dbg !18
  %address.thread867 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local868 = zext nneg i32 %address.thread867 to i64, !dbg !8
  %address.local869 = add i64 %25, %address.local868, !dbg !18
  %978 = lshr i64 %address.local868, 2, !dbg !18
  %address.local871 = and i64 %978, 112, !dbg !18
  %address.local872 = add i64 %address.local869, %address.local871, !dbg !18
  %address.local873 = and i64 %address.local868, 496, !dbg !18
  %address.local874 = sub i64 %address.local872, %address.local873, !dbg !18
  %address.local875 = add i64 %address.local874, %26, !dbg !18
  %address.local876 = shl nuw nsw i64 %address.local875, 7, !dbg !18
  %address.local877 = add nuw nsw i64 %977, %address.local876, !dbg !18
  %address.local878 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !18
  %address.local879 = lshr i64 %address.local868, 2, !dbg !10
  %address.local880 = and i64 %address.local879, 12, !dbg !10
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local878, i64 %address.local880, !dbg !18
  %address.local882 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local881, i64 128, !dbg !18
  store half %976, ptr addrspace(1) %address.local882, align 2, !dbg !18
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !18
  %address.local884 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local883, i64 %address.local880, !dbg !18
  %address.local885 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local884, i64 130, !dbg !18
  store half %975, ptr addrspace(1) %address.local885, align 2, !dbg !18
  %address.local886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !18
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local886, i64 %address.local880, !dbg !18
  %address.local888 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local887, i64 132, !dbg !18
  store half %974, ptr addrspace(1) %address.local888, align 2, !dbg !18
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !18
  %address.local890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local889, i64 %address.local880, !dbg !18
  %address.local891 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local890, i64 134, !dbg !18
  store half %973, ptr addrspace(1) %address.local891, align 2, !dbg !18
  %address.local892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !18
  %address.local893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local892, i64 %address.local880, !dbg !18
  %address.local894 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local893, i64 160, !dbg !18
  store half %967, ptr addrspace(1) %address.local894, align 2, !dbg !18
  %address.local895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !18
  %address.local896 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local895, i64 %address.local880, !dbg !18
  %address.local897 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local896, i64 162, !dbg !18
  store half %966, ptr addrspace(1) %address.local897, align 2, !dbg !18
  %address.local898 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !18
  %address.local899 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local898, i64 %address.local880, !dbg !18
  %address.local900 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local899, i64 164, !dbg !18
  store half %965, ptr addrspace(1) %address.local900, align 2, !dbg !18
  %address.local901 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !18
  %address.local902 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local901, i64 %address.local880, !dbg !18
  %address.local903 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local902, i64 166, !dbg !18
  store half %964, ptr addrspace(1) %address.local903, align 2, !dbg !18
  %979 = insertelement <4 x float> poison, float %819, i64 0, !dbg !16
  %980 = insertelement <4 x float> %979, float %825, i64 1, !dbg !16
  %981 = insertelement <4 x float> %980, float %820, i64 2, !dbg !16
  %982 = insertelement <4 x float> %981, float %826, i64 3, !dbg !16
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %798, !dbg !16
  %normalized45 = fmul <4 x float> %982, %row.reciprocal44, !dbg !16
  %983 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %984 = extractelement <4 x half> %983, i64 3, !dbg !17
  %985 = extractelement <4 x half> %983, i64 2, !dbg !17
  %986 = extractelement <4 x half> %983, i64 1, !dbg !17
  %987 = extractelement <4 x half> %983, i64 0, !dbg !17
  %988 = insertelement <4 x float> poison, float %712, i64 0, !dbg !16
  %989 = insertelement <4 x float> %988, float %718, i64 1, !dbg !16
  %990 = insertelement <4 x float> %989, float %713, i64 2, !dbg !16
  %991 = insertelement <4 x float> %990, float %719, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %991, %row.reciprocal44, !dbg !16
  %992 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %993 = extractelement <4 x half> %992, i64 3, !dbg !17
  %994 = extractelement <4 x half> %992, i64 2, !dbg !17
  %995 = extractelement <4 x half> %992, i64 1, !dbg !17
  %996 = extractelement <4 x half> %992, i64 0, !dbg !17
  %997 = shl nuw nsw i64 %24, 19, !dbg !18
  %address.thread904 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local905 = zext nneg i32 %address.thread904 to i64, !dbg !8
  %address.local906 = add i64 %25, %address.local905, !dbg !18
  %998 = lshr i64 %address.local905, 2, !dbg !18
  %address.local908 = and i64 %998, 112, !dbg !18
  %address.local909 = add i64 %address.local906, %address.local908, !dbg !18
  %address.local910 = and i64 %address.local905, 496, !dbg !18
  %address.local911 = sub i64 %address.local909, %address.local910, !dbg !18
  %address.local912 = add i64 %address.local911, %26, !dbg !18
  %address.local913 = shl nuw nsw i64 %address.local912, 7, !dbg !18
  %address.local914 = add nuw nsw i64 %997, %address.local913, !dbg !18
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !18
  %address.local916 = lshr i64 %address.local905, 2, !dbg !10
  %address.local917 = and i64 %address.local916, 12, !dbg !10
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local915, i64 %address.local917, !dbg !18
  %address.local919 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local918, i64 192, !dbg !18
  store half %996, ptr addrspace(1) %address.local919, align 2, !dbg !18
  %address.local920 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !18
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local920, i64 %address.local917, !dbg !18
  %address.local922 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local921, i64 194, !dbg !18
  store half %995, ptr addrspace(1) %address.local922, align 2, !dbg !18
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !18
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local923, i64 %address.local917, !dbg !18
  %address.local925 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local924, i64 196, !dbg !18
  store half %994, ptr addrspace(1) %address.local925, align 2, !dbg !18
  %address.local926 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !18
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local926, i64 %address.local917, !dbg !18
  %address.local928 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local927, i64 198, !dbg !18
  store half %993, ptr addrspace(1) %address.local928, align 2, !dbg !18
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !18
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local929, i64 %address.local917, !dbg !18
  %address.local931 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local930, i64 224, !dbg !18
  store half %987, ptr addrspace(1) %address.local931, align 2, !dbg !18
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !18
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local932, i64 %address.local917, !dbg !18
  %address.local934 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local933, i64 226, !dbg !18
  store half %986, ptr addrspace(1) %address.local934, align 2, !dbg !18
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !18
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local935, i64 %address.local917, !dbg !18
  %address.local937 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local936, i64 228, !dbg !18
  store half %985, ptr addrspace(1) %address.local937, align 2, !dbg !18
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !18
  %address.local939 = or i64 %address.local916, 115, !dbg !18
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local938, i64 %address.local939, !dbg !18
  store half %984, ptr addrspace(1) %address.local940, align 2, !dbg !18
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="512,512" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!31}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/threads512/run-20261027-163952-a49ris5l/online-baseline")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 57, column: 24, scope: !4)
!8 = !DILocation(line: 67, column: 22, scope: !4)
!10 = !DILocation(line: 69, column: 24, scope: !4)
!12 = !DILocation(line: 68, column: 22, scope: !4)
!13 = !DILocation(line: 87, column: 23, scope: !4)
!14 = !DILocation(line: 70, column: 26, scope: !4)
!15 = !DILocation(line: 86, column: 20, scope: !4)
!16 = !DILocation(line: 92, column: 23, scope: !4)
!17 = !DILocation(line: 93, column: 19, scope: !4)
!18 = !DILocation(line: 94, column: 9, scope: !4)
!19 = !DILocation(line: 71, column: 23, scope: !4)
!20 = !DILocation(line: 74, column: 20, scope: !4)
!21 = !DILocation(line: 75, column: 22, scope: !4)
!22 = !DILocation(line: 76, column: 23, scope: !4)
!23 = !DILocation(line: 77, column: 20, scope: !4)
!24 = !DILocation(line: 78, column: 22, scope: !4)
!25 = !DILocation(line: 79, column: 22, scope: !4)
!26 = !DILocation(line: 80, column: 20, scope: !4)
!27 = !DILocation(line: 81, column: 16, scope: !4)
!28 = !DILocation(line: 88, column: 22, scope: !4)
!30 = !{i32 512, i32 1, i32 1}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{i32 0, i32 512}
