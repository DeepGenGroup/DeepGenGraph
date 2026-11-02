; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/nohint-rolled/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/nohint-rolled/input.mlir"

@shm_0 = addrspace(3) global [10240 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 7, !dbg !10
  %.lobit = ashr i32 %7, 31, !dbg !12
  %10 = xor i32 %7, %.lobit, !dbg !12
  %11 = sdiv i32 %10, 32, !dbg !12
  %.lobit1 = ashr i32 %7, 31, !dbg !12
  %12 = xor i32 %11, %.lobit1, !dbg !12
  %13 = sext i32 %12 to i64, !dbg !12
  %14 = add nsw i64 %6, %13, !dbg !12
  %15 = and i64 %14, 31, !dbg !12
  %16 = shl nsw i64 %8, 7, !dbg !12
  %17 = mul nsw i64 %13, -4096, !dbg !12
  %18 = add nsw i64 %16, %17, !dbg !12
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !39
  %20 = zext nneg i32 %19 to i64, !dbg !13
  %21 = lshr i64 %20, 6, !dbg !13
  %22 = lshr i64 %20, 2, !dbg !13
  %23 = and i64 %22, 12, !dbg !13
  %24 = and i32 %19, 16, !dbg !14
  %.not = icmp eq i32 %24, 0, !dbg !14
  %25 = and i32 %19, 32, !dbg !14
  %.not2 = icmp eq i32 %25, 0, !dbg !14
  %26 = or disjoint i64 %9, 127, !dbg !15
  %27 = add nsw i64 %9, 128, !dbg !15
  %28 = icmp uge i64 %27, %26, !dbg !15
  %29 = shl nsw i64 %8, 2, !dbg !8
  %30 = add nsw i64 %29, 4, !dbg !8
  br label %31, !dbg !8

31:                                               ; preds = %723, %4
  %32 = phi i64 [ %794, %723 ], [ 0, %4 ], !dbg !8
  %33 = phi [2 x <8 x float>] [ %775, %723 ], [ zeroinitializer, %4 ], !dbg !8
  %34 = phi [2 x <1 x float>] [ %779, %723 ], [ zeroinitializer, %4 ], !dbg !8
  %35 = phi [2 x <1 x float>] [ %781, %723 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !8
  %36 = phi [2 x <8 x float>] [ %785, %723 ], [ zeroinitializer, %4 ], !dbg !8
  %37 = phi [2 x <8 x float>] [ %789, %723 ], [ zeroinitializer, %4 ], !dbg !8
  %38 = phi [2 x <8 x float>] [ %793, %723 ], [ zeroinitializer, %4 ], !dbg !8
  %39 = icmp slt i64 %32, %30, !dbg !8
  br i1 %39, label %40, label %795, !dbg !8

40:                                               ; preds = %31
  %41 = shl nsw i64 %32, 5, !dbg !8
  %42 = lshr i64 %32, 7, !dbg !17
  %43 = shl i64 %42, 12, !dbg !17
  %44 = sub i64 %41, %43, !dbg !17
  br label %45, !dbg !14

45:                                               ; preds = %125, %40
  %46 = phi i64 [ %160, %125 ], [ 0, %40 ], !dbg !14
  %47 = phi [2 x <8 x float>] [ %159, %125 ], [ zeroinitializer, %40 ], !dbg !14
  %48 = icmp samesign ult i64 %46, 4, !dbg !14
  br i1 %48, label %49, label %161, !dbg !14

49:                                               ; preds = %45
  %50 = shl nuw nsw i64 %46, 5, !dbg !14
  %.idx = shl nuw nsw i64 %15, 20, !dbg !12
  %51 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx19 = shl nsw i64 %18, 8, !dbg !12
  %52 = getelementptr i8, ptr addrspace(1) %51, i64 %.idx19, !dbg !12
  %53 = getelementptr [2 x i8], ptr addrspace(1) %52, i64 %50, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local55 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local56 = and i64 %address.local55, 1536, !dbg !12
  %address.local57 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local56, !dbg !12
  %address.local58 = ptrtoint ptr addrspace(3) %address.local57 to i64, !dbg !12
  %address.local59 = trunc i64 %address.local58 to i32, !dbg !12
  %54 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local59), !dbg !12
  %55 = zext i32 %54 to i64, !dbg !12
  %56 = inttoptr i64 %55 to ptr addrspace(3), !dbg !12
  %address.local60 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local61 = and i64 %address.local60, 16256, !dbg !12
  %address.local62 = getelementptr [2 x i8], ptr addrspace(1) %53, i64 %address.local61, !dbg !12
  %address.local63 = lshr i64 %address.local, 4, !dbg !12
  %address.local64 = and i64 %address.local63, 8, !dbg !12
  %address.local65 = getelementptr [2 x i8], ptr addrspace(1) %address.local62, i64 %address.local64, !dbg !12
  %57 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %58 = ptrtoint ptr addrspace(1) %address.local65 to i64, !dbg !12
  %59 = sub i64 %58, %57, !dbg !12
  %dma.buffer.offset435 = trunc i64 %59 to i32, !dbg !12
  %60 = trunc i64 %57 to i32, !dbg !12
  %61 = insertelement <4 x i32> poison, i32 %60, i64 0, !dbg !12
  %62 = lshr i64 %57, 32, !dbg !12
  %63 = trunc i64 %62 to i32, !dbg !12
  %64 = insertelement <4 x i32> %61, i32 %63, i64 1, !dbg !12
  %65 = insertelement <4 x i32> %64, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc436 = insertelement <4 x i32> %65, i32 131072, i64 3, !dbg !12
  %66 = ptrtoint ptr addrspace(3) %56 to i32, !dbg !12
  %dma.buffer.lds437 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %66), !dbg !12
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds437, i32 %dma.buffer.offset435, <4 x i32> %dma.buffer.rsrc436) #10, !dbg !12
  %address.local66 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local67 = shl nuw nsw i64 %address.local66, 3, !dbg !12
  %address.local68 = and i64 %address.local67, 3584, !dbg !12
  %address.local69 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local68, !dbg !12
  %address.local70 = ptrtoint ptr addrspace(3) %address.local69 to i64, !dbg !12
  %address.local71 = trunc i64 %address.local70 to i32, !dbg !12
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local71), !dbg !12
  %69 = zext i32 %68 to i64, !dbg !12
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !12
  %address.local72 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local73 = and i64 %address.local72, 16256, !dbg !12
  %address.local74 = getelementptr [2 x i8], ptr addrspace(1) %53, i64 %address.local73, !dbg !12
  %address.local75 = lshr i64 %address.local66, 4, !dbg !12
  %address.local76 = and i64 %address.local75, 24, !dbg !12
  %address.local77 = getelementptr [2 x i8], ptr addrspace(1) %address.local74, i64 %address.local76, !dbg !12
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %72 = ptrtoint ptr addrspace(1) %address.local77 to i64, !dbg !12
  %73 = sub i64 %72, %71, !dbg !12
  %dma.buffer.offset432 = trunc i64 %73 to i32, !dbg !12
  %74 = trunc i64 %71 to i32, !dbg !12
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !12
  %76 = lshr i64 %71, 32, !dbg !12
  %77 = trunc i64 %76 to i32, !dbg !12
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !12
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc433 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !12
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !12
  %dma.buffer.lds434 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !12
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds434, i32 %dma.buffer.offset432, <4 x i32> %dma.buffer.rsrc433) #10, !dbg !12
  %82 = shl nuw nsw i64 %46, 12, !dbg !17
  %83 = add nuw i64 %32, %82, !dbg !17
  %.lobit20 = ashr i64 %83, 63, !dbg !17
  %84 = xor i64 %83, %.lobit20, !dbg !17
  %85 = sdiv i64 %84, 16384, !dbg !17
  %.lobit21 = ashr i64 %83, 63, !dbg !17
  %86 = xor i64 %85, %.lobit21, !dbg !17
  %87 = add nsw i64 %86, %6, !dbg !17
  %88 = add nuw nsw i64 %50, %42, !dbg !17
  %89 = and i64 %88, 127, !dbg !17
  %90 = shl i64 %87, 19, !dbg !17
  %91 = and i64 %90, 16252928, !dbg !17
  %92 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %91, !dbg !17
  %93 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %89, !dbg !17
  %.idx22 = shl i64 %44, 8, !dbg !17
  %94 = getelementptr i8, ptr addrspace(1) %93, i64 %.idx22, !dbg !17
  br label %95, !dbg !17

95:                                               ; preds = %98, %49
  %96 = phi i64 [ %124, %98 ], [ %20, %49 ], !dbg !17
  %97 = icmp samesign ult i64 %96, 128, !dbg !17
  br i1 %97, label %98, label %125, !dbg !17

98:                                               ; preds = %95
  %99 = lshr i64 %96, 2, !dbg !17
  %100 = and i64 %99, 24, !dbg !17
  %101 = shl nuw nsw i64 %96, 3, !dbg !17
  %102 = and i64 %101, 512, !dbg !17
  %103 = shl nuw nsw i64 %96, 7, !dbg !17
  %104 = and i64 %103, 3968, !dbg !17
  %105 = getelementptr [2 x i8], ptr addrspace(1) %94, i64 %100, !dbg !17
  %106 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %104, !dbg !17
  %107 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %102, !dbg !17
  %108 = ptrtoint ptr addrspace(3) %107 to i64, !dbg !17
  %109 = trunc i64 %108 to i32, !dbg !17
  %110 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %109), !dbg !17
  %111 = zext i32 %110 to i64, !dbg !17
  %112 = inttoptr i64 %111 to ptr addrspace(3), !dbg !17
  %113 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %114 = ptrtoint ptr addrspace(1) %106 to i64, !dbg !17
  %115 = sub i64 %114, %113, !dbg !17
  %dma.buffer.offset = trunc i64 %115 to i32, !dbg !17
  %116 = trunc i64 %113 to i32, !dbg !17
  %117 = insertelement <4 x i32> poison, i32 %116, i64 0, !dbg !17
  %118 = lshr i64 %113, 32, !dbg !17
  %119 = trunc i64 %118 to i32, !dbg !17
  %120 = insertelement <4 x i32> %117, i32 %119, i64 1, !dbg !17
  %121 = insertelement <4 x i32> %120, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %121, i32 131072, i64 3, !dbg !17
  %122 = ptrtoint ptr addrspace(3) %112 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %122), !dbg !17
  %123 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %124 = or disjoint i64 %20, 256, !dbg !17
  br label %95, !dbg !17

125:                                              ; preds = %95
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.thread78 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !13, !range !39, !frisk.register_lifetimes !1
  %address.local79 = lshr i32 %address.thread78, 2, !dbg !13
  %address.local80 = and i32 %address.local79, 4, !dbg !13
  %address.local81 = shl nuw nsw i32 %address.thread78, 1, !dbg !13
  %address.local82 = and i32 %address.local81, 384, !dbg !13
  %address.local83 = or disjoint i32 %address.local80, %address.local82, !dbg !13
  %address.local84 = shl nuw nsw i32 %address.thread78, 3, !dbg !13
  %address.local85 = and i32 %address.local84, 120, !dbg !13
  %address.local86 = or disjoint i32 %address.local83, %address.local85, !dbg !13
  %address.local87 = shl nuw nsw i32 %address.thread78, 5, !dbg !13
  %address.local88 = and i32 %address.local87, 1024, !dbg !13
  %address.local89 = or disjoint i32 %address.local86, %address.local88, !dbg !13
  %address.local90 = zext nneg i32 %address.local89 to i64, !dbg !13
  %address.local91 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local90, !dbg !13
  %126 = load <4 x half>, ptr addrspace(3) %address.local91, align 2, !dbg !13
  %address.local92 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local90, !dbg !13
  %address.local93 = getelementptr i8, ptr addrspace(3) %address.local92, i64 4096, !dbg !13
  %127 = load <4 x half>, ptr addrspace(3) %address.local93, align 2, !dbg !13
  %address.local94 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local90, !dbg !13
  %address.local95 = getelementptr i8, ptr addrspace(3) %address.local94, i64 1024, !dbg !13
  %128 = load <4 x half>, ptr addrspace(3) %address.local95, align 2, !dbg !13
  %129 = fmul <4 x half> %126, splat (half 0xH3015), !dbg !13
  %address.local96 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local90, !dbg !13
  %address.local97 = getelementptr i8, ptr addrspace(3) %address.local96, i64 5120, !dbg !13
  %130 = load <4 x half>, ptr addrspace(3) %address.local97, align 2, !dbg !13
  %131 = fmul <4 x half> %127, splat (half 0xH3015), !dbg !13
  %132 = fmul <4 x half> %128, splat (half 0xH3015), !dbg !13
  %133 = fmul <4 x half> %130, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local98 = lshr i32 %address.thread78, 2, !dbg !7
  %address.local99 = and i32 %address.local98, 4, !dbg !7
  %address.local100 = shl nuw nsw i32 %address.thread78, 3, !dbg !7
  %address.local101 = and i32 %address.local100, 376, !dbg !7
  %address.local102 = or disjoint i32 %address.local99, %address.local101, !dbg !7
  %address.local103 = zext nneg i32 %address.local102 to i64, !dbg !7
  %address.local104 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local103, !dbg !7
  %134 = load <4 x half>, ptr addrspace(3) %address.local104, align 2, !dbg !7
  %135 = extractvalue [2 x <8 x float>] %47, 0, !dbg !36
  %136 = shufflevector <8 x float> %135, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %address.local105 = lshr i32 %address.thread78, 2, !dbg !7
  %address.local106 = and i32 %address.local105, 4, !dbg !7
  %address.local107 = shl nuw nsw i32 %address.thread78, 3, !dbg !7
  %address.local108 = and i32 %address.local107, 376, !dbg !7
  %address.local109 = or disjoint i32 %address.local106, %address.local108, !dbg !7
  %address.local110 = zext nneg i32 %address.local109 to i64, !dbg !7
  %address.local111 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local110, !dbg !7
  %address.local112 = getelementptr i8, ptr addrspace(3) %address.local111, i64 1024, !dbg !7
  %137 = load <4 x half>, ptr addrspace(3) %address.local112, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %129, <4 x half> %134, <4 x float> %136) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %139 = shufflevector <8 x float> %135, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %137, <4 x float> %138) #9, !dbg !7
  %141 = extractvalue [2 x <8 x float>] %47, 1, !dbg !36
  %142 = shufflevector <8 x float> %141, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %143 = shufflevector <8 x float> %141, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %address.local113 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local103, !dbg !7
  %address.local114 = getelementptr i8, ptr addrspace(3) %address.local113, i64 256, !dbg !7
  %144 = load <4 x half>, ptr addrspace(3) %address.local114, align 2, !dbg !7
  %address.local115 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local110, !dbg !7
  %address.local116 = getelementptr i8, ptr addrspace(3) %address.local115, i64 1280, !dbg !7
  %145 = load <4 x half>, ptr addrspace(3) %address.local116, align 2, !dbg !7
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %129, <4 x half> %144, <4 x float> %139) #9, !dbg !7
  %147 = load <4 x half>, ptr addrspace(3) %address.local104, align 2, !dbg !7
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %145, <4 x float> %146) #9, !dbg !7
  %149 = load <4 x half>, ptr addrspace(3) %address.local112, align 2, !dbg !7
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %147, <4 x float> %142) #9, !dbg !7
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %149, <4 x float> %150) #9, !dbg !7
  %152 = load <4 x half>, ptr addrspace(3) %address.local114, align 2, !dbg !7
  %153 = load <4 x half>, ptr addrspace(3) %address.local116, align 2, !dbg !7
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %152, <4 x float> %143) #9, !dbg !7
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %153, <4 x float> %154) #9, !dbg !7
  %156 = shufflevector <4 x float> %140, <4 x float> %148, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %157 = insertvalue [2 x <8 x float>] poison, <8 x float> %156, 0, !dbg !14
  %158 = shufflevector <4 x float> %151, <4 x float> %155, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %159 = insertvalue [2 x <8 x float>] %157, <8 x float> %158, 1, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %160 = add nuw nsw i64 %46, 1, !dbg !14
  br label %45, !dbg !14

161:                                              ; preds = %45
  %162 = add nsw i64 %42, %6, !dbg !18
  %163 = shl i64 %162, 19, !dbg !18
  %164 = and i64 %163, 16252928, !dbg !18
  %165 = shl nsw i64 %44, 7, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %166 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %164, !dbg !18
  %167 = getelementptr [2 x i8], ptr addrspace(1) %166, i64 %165, !dbg !18
  br label %168, !dbg !18

168:                                              ; preds = %171, %161
  %169 = phi i64 [ %197, %171 ], [ %20, %161 ], !dbg !18
  %170 = icmp samesign ult i64 %169, 128, !dbg !18
  br i1 %170, label %171, label %198, !dbg !18

171:                                              ; preds = %168
  %172 = shl nuw nsw i64 %169, 3, !dbg !18
  %173 = and i64 %172, 24, !dbg !18
  %174 = shl nuw nsw i64 %169, 3, !dbg !18
  %175 = and i64 %174, 512, !dbg !18
  %176 = shl nuw nsw i64 %169, 5, !dbg !18
  %177 = and i64 %176, 3968, !dbg !18
  %178 = getelementptr [2 x i8], ptr addrspace(1) %167, i64 %177, !dbg !18
  %179 = getelementptr [2 x i8], ptr addrspace(1) %178, i64 %173, !dbg !18
  %180 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %175, !dbg !18
  %181 = ptrtoint ptr addrspace(3) %180 to i64, !dbg !18
  %182 = trunc i64 %181 to i32, !dbg !18
  %183 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %182), !dbg !18
  %184 = zext i32 %183 to i64, !dbg !18
  %185 = inttoptr i64 %184 to ptr addrspace(3), !dbg !18
  %186 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %187 = ptrtoint ptr addrspace(1) %179 to i64, !dbg !18
  %188 = sub i64 %187, %186, !dbg !18
  %dma.buffer.offset438 = trunc i64 %188 to i32, !dbg !18
  %189 = trunc i64 %186 to i32, !dbg !18
  %190 = insertelement <4 x i32> poison, i32 %189, i64 0, !dbg !18
  %191 = lshr i64 %186, 32, !dbg !18
  %192 = trunc i64 %191 to i32, !dbg !18
  %193 = insertelement <4 x i32> %190, i32 %192, i64 1, !dbg !18
  %194 = insertelement <4 x i32> %193, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc439 = insertelement <4 x i32> %194, i32 131072, i64 3, !dbg !18
  %195 = ptrtoint ptr addrspace(3) %185 to i32, !dbg !18
  %dma.buffer.lds440 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %195), !dbg !18
  %196 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds440, i32 %dma.buffer.offset438, <4 x i32> %dma.buffer.rsrc439) #10, !dbg !18
  %197 = or disjoint i64 %20, 256, !dbg !18
  br label %168, !dbg !18

198:                                              ; preds = %168
  %199 = extractvalue [2 x <8 x float>] %47, 0, !dbg !14
  %200 = extractelement <8 x float> %199, i64 0, !dbg !14
  %201 = extractelement <8 x float> %199, i64 1, !dbg !14
  %202 = extractelement <8 x float> %199, i64 2, !dbg !14
  %203 = extractelement <8 x float> %199, i64 3, !dbg !14
  %204 = select i1 %.not, float %201, float %200, !dbg !14
  %205 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !14
  %206 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %205), !dbg !14
  %207 = add nuw nsw i32 %206, 64, !dbg !14
  %208 = xor i32 %206, 16, !dbg !14
  %209 = and i32 %207, 192, !dbg !14
  %210 = icmp samesign ult i32 %208, %209, !dbg !14
  %211 = select i1 %210, i32 %208, i32 %206, !dbg !14
  %212 = shl nuw nsw i32 %211, 2, !dbg !14
  %213 = bitcast float %204 to i32, !dbg !14
  %214 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %213), !dbg !14
  %215 = bitcast i32 %214 to float, !dbg !14
  %216 = select i1 %.not, float %200, float %215, !dbg !14
  %217 = select i1 %.not, float %215, float %201, !dbg !14
  %218 = select i1 %.not, float %203, float %202, !dbg !14
  %219 = bitcast float %218 to i32, !dbg !14
  %220 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %219), !dbg !14
  %221 = bitcast i32 %220 to float, !dbg !14
  %222 = select i1 %.not, float %202, float %221, !dbg !14
  %223 = select i1 %.not, float %221, float %203, !dbg !14
  %224 = select i1 %.not2, float %222, float %216, !dbg !14
  %225 = xor i32 %206, 32, !dbg !14
  %226 = icmp samesign ult i32 %225, %209, !dbg !14
  %227 = select i1 %226, i32 %225, i32 %206, !dbg !14
  %228 = shl nuw nsw i32 %227, 2, !dbg !14
  %229 = bitcast float %224 to i32, !dbg !14
  %230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %229), !dbg !14
  %231 = bitcast i32 %230 to float, !dbg !14
  %232 = select i1 %.not2, float %216, float %231, !dbg !14
  %233 = select i1 %.not2, float %231, float %222, !dbg !14
  %234 = select i1 %.not2, float %223, float %217, !dbg !14
  %235 = bitcast float %234 to i32, !dbg !14
  %236 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %235), !dbg !14
  %237 = bitcast i32 %236 to float, !dbg !14
  %238 = select i1 %.not2, float %217, float %237, !dbg !14
  %239 = select i1 %.not2, float %237, float %223, !dbg !14
  %240 = extractelement <8 x float> %199, i64 4, !dbg !14
  %241 = extractelement <8 x float> %199, i64 5, !dbg !14
  %242 = extractelement <8 x float> %199, i64 6, !dbg !14
  %243 = extractelement <8 x float> %199, i64 7, !dbg !14
  %244 = select i1 %.not, float %241, float %240, !dbg !14
  %245 = bitcast float %244 to i32, !dbg !14
  %246 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %245), !dbg !14
  %247 = bitcast i32 %246 to float, !dbg !14
  %248 = select i1 %.not, float %240, float %247, !dbg !14
  %249 = select i1 %.not, float %247, float %241, !dbg !14
  %250 = select i1 %.not, float %243, float %242, !dbg !14
  %251 = bitcast float %250 to i32, !dbg !14
  %252 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %251), !dbg !14
  %253 = bitcast i32 %252 to float, !dbg !14
  %254 = select i1 %.not, float %242, float %253, !dbg !14
  %255 = select i1 %.not, float %253, float %243, !dbg !14
  %256 = select i1 %.not2, float %254, float %248, !dbg !14
  %257 = bitcast float %256 to i32, !dbg !14
  %258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %257), !dbg !14
  %259 = bitcast i32 %258 to float, !dbg !14
  %260 = select i1 %.not2, float %248, float %259, !dbg !14
  %261 = select i1 %.not2, float %259, float %254, !dbg !14
  %262 = select i1 %.not2, float %255, float %249, !dbg !14
  %263 = bitcast float %262 to i32, !dbg !14
  %264 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %263), !dbg !14
  %265 = bitcast i32 %264 to float, !dbg !14
  %266 = select i1 %.not2, float %249, float %265, !dbg !14
  %267 = select i1 %.not2, float %265, float %255, !dbg !14
  %268 = extractvalue [2 x <8 x float>] %47, 1, !dbg !14
  %269 = extractelement <8 x float> %268, i64 0, !dbg !14
  %270 = extractelement <8 x float> %268, i64 1, !dbg !14
  %271 = extractelement <8 x float> %268, i64 2, !dbg !14
  %272 = extractelement <8 x float> %268, i64 3, !dbg !14
  %273 = select i1 %.not, float %270, float %269, !dbg !14
  %274 = bitcast float %273 to i32, !dbg !14
  %275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %274), !dbg !14
  %276 = bitcast i32 %275 to float, !dbg !14
  %277 = select i1 %.not, float %269, float %276, !dbg !14
  %278 = select i1 %.not, float %276, float %270, !dbg !14
  %279 = select i1 %.not, float %272, float %271, !dbg !14
  %280 = bitcast float %279 to i32, !dbg !14
  %281 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %280), !dbg !14
  %282 = bitcast i32 %281 to float, !dbg !14
  %283 = select i1 %.not, float %271, float %282, !dbg !14
  %284 = select i1 %.not, float %282, float %272, !dbg !14
  %285 = select i1 %.not2, float %283, float %277, !dbg !14
  %286 = bitcast float %285 to i32, !dbg !14
  %287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %286), !dbg !14
  %288 = bitcast i32 %287 to float, !dbg !14
  %289 = select i1 %.not2, float %277, float %288, !dbg !14
  %290 = select i1 %.not2, float %288, float %283, !dbg !14
  %291 = select i1 %.not2, float %284, float %278, !dbg !14
  %292 = bitcast float %291 to i32, !dbg !14
  %293 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %292), !dbg !14
  %294 = bitcast i32 %293 to float, !dbg !14
  %295 = select i1 %.not2, float %278, float %294, !dbg !14
  %296 = select i1 %.not2, float %294, float %284, !dbg !14
  %297 = extractelement <8 x float> %268, i64 4, !dbg !14
  %298 = extractelement <8 x float> %268, i64 5, !dbg !14
  %299 = extractelement <8 x float> %268, i64 6, !dbg !14
  %300 = extractelement <8 x float> %268, i64 7, !dbg !14
  %301 = select i1 %.not, float %298, float %297, !dbg !14
  %302 = bitcast float %301 to i32, !dbg !14
  %303 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %302), !dbg !14
  %304 = bitcast i32 %303 to float, !dbg !14
  %305 = select i1 %.not, float %297, float %304, !dbg !14
  %306 = select i1 %.not, float %304, float %298, !dbg !14
  %307 = select i1 %.not, float %300, float %299, !dbg !14
  %308 = bitcast float %307 to i32, !dbg !14
  %309 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %308), !dbg !14
  %310 = bitcast i32 %309 to float, !dbg !14
  %311 = select i1 %.not, float %299, float %310, !dbg !14
  %312 = select i1 %.not, float %310, float %300, !dbg !14
  %313 = select i1 %.not2, float %311, float %305, !dbg !14
  %314 = bitcast float %313 to i32, !dbg !14
  %315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %314), !dbg !14
  %316 = bitcast i32 %315 to float, !dbg !14
  %317 = select i1 %.not2, float %305, float %316, !dbg !14
  %318 = select i1 %.not2, float %316, float %311, !dbg !14
  %319 = select i1 %.not2, float %312, float %306, !dbg !14
  %320 = bitcast float %319 to i32, !dbg !14
  %321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %320), !dbg !14
  %322 = bitcast i32 %321 to float, !dbg !14
  %323 = select i1 %.not2, float %306, float %322, !dbg !14
  %324 = select i1 %.not2, float %322, float %312, !dbg !14
  %325 = or disjoint i64 %41, 31, !dbg !15
  %326 = icmp uge i64 %9, %325, !dbg !15
  %327 = and i1 %28, %326, !dbg !15
  br i1 %327, label %328, label %329, !dbg !15

328:                                              ; preds = %198
  br label %376, !dbg !15

329:                                              ; preds = %198
  %330 = add nsw i64 %9, %20, !dbg !15
  %331 = shl nuw nsw i64 %21, 4, !dbg !15
  %332 = add nsw i64 %330, %331, !dbg !15
  %333 = and i32 %19, 240, !dbg !15
  %index.compact = zext nneg i32 %333 to i64, !dbg !15
  %334 = sub nsw i64 %332, %index.compact, !dbg !15
  %335 = or disjoint i64 %41, %23, !dbg !15
  %336 = add nsw i64 %334, 1, !dbg !22
  %337 = lshr i64 %336, 32, !dbg !23
  %compare.high = trunc i64 %337 to i32, !dbg !23
  %338 = lshr i64 %335, 32, !dbg !23
  %compare.high368 = trunc i64 %338 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high368, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high368, !dbg !23
  %compare.low.base = trunc i64 %336 to i32, !dbg !23
  %compare.low.base369 = trunc i64 %335 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base369, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %339 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base369, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low370 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result371 = select i1 %compare.high.equal, i1 %compare.low370, i1 %compare.high.order, !dbg !23
  %340 = select i1 %compare.result371, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base372 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base369, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local373 = or i32 %compare.local.base372, 2, !dbg !23
  %compare.low374 = icmp ugt i32 %compare.low.base, %compare.local373, !dbg !23
  %compare.result375 = select i1 %compare.high.equal, i1 %compare.low374, i1 %compare.high.order, !dbg !23
  %341 = select i1 %compare.result375, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base376 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base369, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local377 = or i32 %compare.local.base376, 3, !dbg !23
  %compare.low378 = icmp ugt i32 %compare.low.base, %compare.local377, !dbg !23
  %compare.result379 = select i1 %compare.high.equal, i1 %compare.low378, i1 %compare.high.order, !dbg !23
  %342 = select i1 %compare.result379, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base380 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base369, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local381 = or i32 %compare.local.base380, 16, !dbg !23
  %compare.low382 = icmp ugt i32 %compare.low.base, %compare.local381, !dbg !23
  %compare.result383 = select i1 %compare.high.equal, i1 %compare.low382, i1 %compare.high.order, !dbg !23
  %343 = select i1 %compare.result383, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base384 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base369, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local385 = or i32 %compare.local.base384, 17, !dbg !23
  %compare.low386 = icmp ugt i32 %compare.low.base, %compare.local385, !dbg !23
  %compare.result387 = select i1 %compare.high.equal, i1 %compare.low386, i1 %compare.high.order, !dbg !23
  %344 = select i1 %compare.result387, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base388 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base369, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local389 = or i32 %compare.local.base388, 18, !dbg !23
  %compare.low390 = icmp ugt i32 %compare.low.base, %compare.local389, !dbg !23
  %compare.result391 = select i1 %compare.high.equal, i1 %compare.low390, i1 %compare.high.order, !dbg !23
  %345 = select i1 %compare.result391, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base392 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base369, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local393 = or i32 %compare.local.base392, 19, !dbg !23
  %compare.low394 = icmp ugt i32 %compare.low.base, %compare.local393, !dbg !23
  %compare.result395 = select i1 %compare.high.equal, i1 %compare.low394, i1 %compare.high.order, !dbg !23
  %346 = select i1 %compare.result395, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %347 = add nsw i64 %334, 65, !dbg !22
  %348 = lshr i64 %347, 32, !dbg !23
  %compare.high396 = trunc i64 %348 to i32, !dbg !23
  %349 = lshr i64 %335, 32, !dbg !23
  %compare.high397 = trunc i64 %349 to i32, !dbg !23
  %compare.high.equal398 = icmp eq i32 %compare.high396, %compare.high397, !dbg !23
  %compare.high.order399 = icmp ugt i32 %compare.high396, %compare.high397, !dbg !23
  %compare.low.base400 = trunc i64 %347 to i32, !dbg !23
  %compare.low.base401 = trunc i64 %335 to i32, !dbg !23
  %compare.low402 = icmp ugt i32 %compare.low.base400, %compare.low.base401, !dbg !23
  %compare.result403 = select i1 %compare.high.equal398, i1 %compare.low402, i1 %compare.high.order399, !dbg !23
  %350 = select i1 %compare.result403, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base404 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base401, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local405 = or i32 %compare.local.base404, 1, !dbg !23
  %compare.low406 = icmp ugt i32 %compare.low.base400, %compare.local405, !dbg !23
  %compare.result407 = select i1 %compare.high.equal398, i1 %compare.low406, i1 %compare.high.order399, !dbg !23
  %351 = select i1 %compare.result407, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base408 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base401, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local409 = or i32 %compare.local.base408, 2, !dbg !23
  %compare.low410 = icmp ugt i32 %compare.low.base400, %compare.local409, !dbg !23
  %compare.result411 = select i1 %compare.high.equal398, i1 %compare.low410, i1 %compare.high.order399, !dbg !23
  %352 = select i1 %compare.result411, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base412 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base401, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local413 = or i32 %compare.local.base412, 3, !dbg !23
  %compare.low414 = icmp ugt i32 %compare.low.base400, %compare.local413, !dbg !23
  %compare.result415 = select i1 %compare.high.equal398, i1 %compare.low414, i1 %compare.high.order399, !dbg !23
  %353 = select i1 %compare.result415, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base416 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base401, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local417 = or i32 %compare.local.base416, 16, !dbg !23
  %compare.low418 = icmp ugt i32 %compare.low.base400, %compare.local417, !dbg !23
  %compare.result419 = select i1 %compare.high.equal398, i1 %compare.low418, i1 %compare.high.order399, !dbg !23
  %354 = select i1 %compare.result419, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base420 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base401, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local421 = or i32 %compare.local.base420, 17, !dbg !23
  %compare.low422 = icmp ugt i32 %compare.low.base400, %compare.local421, !dbg !23
  %compare.result423 = select i1 %compare.high.equal398, i1 %compare.low422, i1 %compare.high.order399, !dbg !23
  %355 = select i1 %compare.result423, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base424 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base401, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local425 = or i32 %compare.local.base424, 18, !dbg !23
  %compare.low426 = icmp ugt i32 %compare.low.base400, %compare.local425, !dbg !23
  %compare.result427 = select i1 %compare.high.equal398, i1 %compare.low426, i1 %compare.high.order399, !dbg !23
  %356 = select i1 %compare.result427, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base428 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base401, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local429 = or i32 %compare.local.base428, 19, !dbg !23
  %compare.low430 = icmp ugt i32 %compare.low.base400, %compare.local429, !dbg !23
  %compare.result431 = select i1 %compare.high.equal398, i1 %compare.low430, i1 %compare.high.order399, !dbg !23
  %357 = select i1 %compare.result431, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %358 = insertelement <8 x float> poison, float %339, i64 0, !dbg !15
  %359 = insertelement <8 x float> %358, float %340, i64 1, !dbg !15
  %360 = insertelement <8 x float> %359, float %341, i64 2, !dbg !15
  %361 = insertelement <8 x float> %360, float %342, i64 3, !dbg !15
  %362 = insertelement <8 x float> %361, float %343, i64 4, !dbg !15
  %363 = insertelement <8 x float> %362, float %344, i64 5, !dbg !15
  %364 = insertelement <8 x float> %363, float %345, i64 6, !dbg !15
  %365 = insertelement <8 x float> %364, float %346, i64 7, !dbg !15
  %366 = insertvalue [2 x <8 x float>] poison, <8 x float> %365, 0, !dbg !15
  %367 = insertelement <8 x float> poison, float %350, i64 0, !dbg !15
  %368 = insertelement <8 x float> %367, float %351, i64 1, !dbg !15
  %369 = insertelement <8 x float> %368, float %352, i64 2, !dbg !15
  %370 = insertelement <8 x float> %369, float %353, i64 3, !dbg !15
  %371 = insertelement <8 x float> %370, float %354, i64 4, !dbg !15
  %372 = insertelement <8 x float> %371, float %355, i64 5, !dbg !15
  %373 = insertelement <8 x float> %372, float %356, i64 6, !dbg !15
  %374 = insertelement <8 x float> %373, float %357, i64 7, !dbg !15
  %375 = insertvalue [2 x <8 x float>] %366, <8 x float> %374, 1, !dbg !15
  br label %376, !dbg !15

376:                                              ; preds = %328, %329
  %377 = phi [2 x <8 x float>] [ %375, %329 ], [ zeroinitializer, %328 ], !dbg !15
  br label %378, !dbg !15

378:                                              ; preds = %376
  %379 = extractvalue [2 x <8 x float>] %377, 0, !dbg !25
  %380 = insertelement <4 x float> poison, float %232, i64 0, !dbg !25
  %381 = insertelement <4 x float> %380, float %238, i64 1, !dbg !25
  %382 = insertelement <4 x float> %381, float %233, i64 2, !dbg !25
  %383 = insertelement <4 x float> %382, float %239, i64 3, !dbg !25
  %384 = shufflevector <8 x float> %379, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %385 = fadd <4 x float> %383, %384, !dbg !25
  %386 = extractelement <4 x float> %385, i64 0, !dbg !25
  %387 = extractelement <4 x float> %385, i64 1, !dbg !25
  %388 = extractelement <4 x float> %385, i64 2, !dbg !25
  %389 = extractelement <4 x float> %385, i64 3, !dbg !25
  %390 = insertelement <4 x float> poison, float %260, i64 0, !dbg !25
  %391 = insertelement <4 x float> %390, float %266, i64 1, !dbg !25
  %392 = insertelement <4 x float> %391, float %261, i64 2, !dbg !25
  %393 = insertelement <4 x float> %392, float %267, i64 3, !dbg !25
  %394 = shufflevector <8 x float> %379, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %395 = fadd <4 x float> %393, %394, !dbg !25
  %396 = extractelement <4 x float> %395, i64 0, !dbg !25
  %397 = extractelement <4 x float> %395, i64 1, !dbg !25
  %398 = extractelement <4 x float> %395, i64 2, !dbg !25
  %399 = extractelement <4 x float> %395, i64 3, !dbg !25
  %400 = extractvalue [2 x <8 x float>] %377, 1, !dbg !25
  %401 = insertelement <4 x float> poison, float %289, i64 0, !dbg !25
  %402 = insertelement <4 x float> %401, float %295, i64 1, !dbg !25
  %403 = insertelement <4 x float> %402, float %290, i64 2, !dbg !25
  %404 = insertelement <4 x float> %403, float %296, i64 3, !dbg !25
  %405 = shufflevector <8 x float> %400, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %406 = fadd <4 x float> %404, %405, !dbg !25
  %407 = extractelement <4 x float> %406, i64 0, !dbg !25
  %408 = extractelement <4 x float> %406, i64 1, !dbg !25
  %409 = extractelement <4 x float> %406, i64 2, !dbg !25
  %410 = extractelement <4 x float> %406, i64 3, !dbg !25
  %411 = insertelement <4 x float> poison, float %317, i64 0, !dbg !25
  %412 = insertelement <4 x float> %411, float %323, i64 1, !dbg !25
  %413 = insertelement <4 x float> %412, float %318, i64 2, !dbg !25
  %414 = insertelement <4 x float> %413, float %324, i64 3, !dbg !25
  %415 = shufflevector <8 x float> %400, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %416 = fadd <4 x float> %414, %415, !dbg !25
  %417 = extractelement <4 x float> %416, i64 0, !dbg !25
  %418 = extractelement <4 x float> %416, i64 1, !dbg !25
  %419 = extractelement <4 x float> %416, i64 2, !dbg !25
  %420 = extractelement <4 x float> %416, i64 3, !dbg !25
  %421 = call float @llvm.maxnum.f32(float %386, float 0xFFF0000000000000), !dbg !26
  %422 = call float @llvm.maxnum.f32(float %421, float %387), !dbg !26
  %423 = call float @llvm.maxnum.f32(float %422, float %388), !dbg !26
  %424 = call float @llvm.maxnum.f32(float %423, float %389), !dbg !26
  %425 = call float @llvm.maxnum.f32(float %424, float %396), !dbg !26
  %426 = call float @llvm.maxnum.f32(float %425, float %397), !dbg !26
  %427 = call float @llvm.maxnum.f32(float %426, float %398), !dbg !26
  %428 = call float @llvm.maxnum.f32(float %427, float %399), !dbg !26
  %429 = bitcast float %428 to i32, !dbg !26
  %430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %429), !dbg !26
  %431 = bitcast i32 %430 to float, !dbg !26
  %432 = call float @llvm.maxnum.f32(float %428, float %431), !dbg !26
  %433 = bitcast float %432 to i32, !dbg !26
  %434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %433), !dbg !26
  %435 = bitcast i32 %434 to float, !dbg !26
  %436 = call float @llvm.maxnum.f32(float %432, float %435), !dbg !26
  %437 = extractvalue [2 x <1 x float>] %35, 0, !dbg !26
  %438 = insertelement <1 x float> poison, float %436, i64 0, !dbg !26
  %439 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %438, <1 x float> %437), !dbg !26
  %440 = call float @llvm.maxnum.f32(float %407, float 0xFFF0000000000000), !dbg !26
  %441 = call float @llvm.maxnum.f32(float %440, float %408), !dbg !26
  %442 = call float @llvm.maxnum.f32(float %441, float %409), !dbg !26
  %443 = call float @llvm.maxnum.f32(float %442, float %410), !dbg !26
  %444 = call float @llvm.maxnum.f32(float %443, float %417), !dbg !26
  %445 = call float @llvm.maxnum.f32(float %444, float %418), !dbg !26
  %446 = call float @llvm.maxnum.f32(float %445, float %419), !dbg !26
  %447 = call float @llvm.maxnum.f32(float %446, float %420), !dbg !26
  %448 = bitcast float %447 to i32, !dbg !26
  %449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %448), !dbg !26
  %450 = bitcast i32 %449 to float, !dbg !26
  %451 = call float @llvm.maxnum.f32(float %447, float %450), !dbg !26
  %452 = bitcast float %451 to i32, !dbg !26
  %453 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %452), !dbg !26
  %454 = bitcast i32 %453 to float, !dbg !26
  %455 = call float @llvm.maxnum.f32(float %451, float %454), !dbg !26
  %456 = extractvalue [2 x <1 x float>] %35, 1, !dbg !26
  %457 = insertelement <1 x float> poison, float %455, i64 0, !dbg !26
  %458 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %457, <1 x float> %456), !dbg !26
  %459 = shufflevector <1 x float> %439, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %460 = fsub <4 x float> %385, %459, !dbg !27
  %461 = fsub <4 x float> %395, %459, !dbg !27
  %462 = shufflevector <1 x float> %458, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %463 = fsub <4 x float> %406, %462, !dbg !27
  %464 = fsub <4 x float> %416, %462, !dbg !27
  %465 = fsub <1 x float> %437, %439, !dbg !28
  %466 = extractelement <1 x float> %465, i64 0, !dbg !29
  %467 = call float @llvm.amdgcn.exp2.f32(float %466), !dbg !29
  %468 = extractvalue [2 x <1 x float>] %34, 0, !dbg !30
  %469 = extractelement <1 x float> %468, i64 0, !dbg !30
  %470 = fmul float %469, %467, !dbg !30
  %471 = fsub <1 x float> %456, %458, !dbg !28
  %472 = extractelement <1 x float> %471, i64 0, !dbg !29
  %473 = call float @llvm.amdgcn.exp2.f32(float %472), !dbg !29
  %474 = extractvalue [2 x <1 x float>] %34, 1, !dbg !30
  %475 = extractelement <1 x float> %474, i64 0, !dbg !30
  %476 = fmul float %475, %473, !dbg !30
  %477 = extractelement <4 x float> %460, i64 0, !dbg !31
  %478 = call float @llvm.amdgcn.exp2.f32(float %477), !dbg !31
  %479 = insertelement <4 x float> poison, float %478, i64 0, !dbg !33
  %480 = extractelement <4 x float> %460, i64 1, !dbg !31
  %481 = call float @llvm.amdgcn.exp2.f32(float %480), !dbg !31
  %482 = insertelement <4 x float> %479, float %481, i64 1, !dbg !33
  %483 = fadd float %478, %481, !dbg !32
  %484 = extractelement <4 x float> %460, i64 2, !dbg !31
  %485 = call float @llvm.amdgcn.exp2.f32(float %484), !dbg !31
  %486 = insertelement <4 x float> %482, float %485, i64 2, !dbg !33
  %487 = fadd float %483, %485, !dbg !32
  %488 = extractelement <4 x float> %460, i64 3, !dbg !31
  %489 = call float @llvm.amdgcn.exp2.f32(float %488), !dbg !31
  %490 = insertelement <4 x float> %486, float %489, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %490) #9, !dbg !33, !frisk.register_lifetimes !1
  %491 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %491) #9, !dbg !32, !frisk.register_lifetimes !1
  %492 = fadd float %487, %489, !dbg !32
  %493 = extractelement <4 x float> %461, i64 0, !dbg !31
  %494 = call float @llvm.amdgcn.exp2.f32(float %493), !dbg !31
  %495 = insertelement <4 x float> poison, float %494, i64 0, !dbg !33
  %496 = fadd float %492, %494, !dbg !32
  %497 = extractelement <4 x float> %461, i64 1, !dbg !31
  %498 = call float @llvm.amdgcn.exp2.f32(float %497), !dbg !31
  %499 = insertelement <4 x float> %495, float %498, i64 1, !dbg !33
  %500 = fadd float %496, %498, !dbg !32
  %501 = extractelement <4 x float> %461, i64 2, !dbg !31
  %502 = call float @llvm.amdgcn.exp2.f32(float %501), !dbg !31
  %503 = insertelement <4 x float> %499, float %502, i64 2, !dbg !33
  %504 = fadd float %500, %502, !dbg !32
  %505 = extractelement <4 x float> %461, i64 3, !dbg !31
  %506 = call float @llvm.amdgcn.exp2.f32(float %505), !dbg !31
  %507 = insertelement <4 x float> %503, float %506, i64 3, !dbg !33
  %math.result49 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %507) #9, !dbg !33, !frisk.register_lifetimes !1
  %508 = fptrunc <4 x float> %math.result49 to <4 x half>, !dbg !33
  %math.packed50 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %508) #9, !dbg !32, !frisk.register_lifetimes !1
  %509 = fadd float %504, %506, !dbg !32
  %510 = extractelement <4 x float> %463, i64 0, !dbg !31
  %511 = call float @llvm.amdgcn.exp2.f32(float %510), !dbg !31
  %512 = insertelement <4 x float> poison, float %511, i64 0, !dbg !33
  %513 = extractelement <4 x float> %463, i64 1, !dbg !31
  %514 = call float @llvm.amdgcn.exp2.f32(float %513), !dbg !31
  %515 = insertelement <4 x float> %512, float %514, i64 1, !dbg !33
  %516 = fadd float %511, %514, !dbg !32
  %517 = extractelement <4 x float> %463, i64 2, !dbg !31
  %518 = call float @llvm.amdgcn.exp2.f32(float %517), !dbg !31
  %519 = insertelement <4 x float> %515, float %518, i64 2, !dbg !33
  %520 = fadd float %516, %518, !dbg !32
  %521 = extractelement <4 x float> %463, i64 3, !dbg !31
  %522 = call float @llvm.amdgcn.exp2.f32(float %521), !dbg !31
  %523 = insertelement <4 x float> %519, float %522, i64 3, !dbg !33
  %math.result51 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %523) #9, !dbg !33, !frisk.register_lifetimes !1
  %524 = fptrunc <4 x float> %math.result51 to <4 x half>, !dbg !33
  %math.packed52 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %524) #9, !dbg !32, !frisk.register_lifetimes !1
  %525 = fadd float %520, %522, !dbg !32
  %526 = extractelement <4 x float> %464, i64 0, !dbg !31
  %527 = call float @llvm.amdgcn.exp2.f32(float %526), !dbg !31
  %528 = insertelement <4 x float> poison, float %527, i64 0, !dbg !33
  %529 = fadd float %525, %527, !dbg !32
  %530 = extractelement <4 x float> %464, i64 1, !dbg !31
  %531 = call float @llvm.amdgcn.exp2.f32(float %530), !dbg !31
  %532 = insertelement <4 x float> %528, float %531, i64 1, !dbg !33
  %533 = fadd float %529, %531, !dbg !32
  %534 = extractelement <4 x float> %464, i64 2, !dbg !31
  %535 = call float @llvm.amdgcn.exp2.f32(float %534), !dbg !31
  %536 = insertelement <4 x float> %532, float %535, i64 2, !dbg !33
  %537 = fadd float %533, %535, !dbg !32
  %538 = extractelement <4 x float> %464, i64 3, !dbg !31
  %539 = call float @llvm.amdgcn.exp2.f32(float %538), !dbg !31
  %540 = insertelement <4 x float> %536, float %539, i64 3, !dbg !33
  %math.result53 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %540) #9, !dbg !33, !frisk.register_lifetimes !1
  %541 = fptrunc <4 x float> %math.result53 to <4 x half>, !dbg !33
  %math.packed54 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %541) #9, !dbg !32, !frisk.register_lifetimes !1
  %542 = fadd float %537, %539, !dbg !32
  %543 = bitcast float %509 to i32, !dbg !32
  %544 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %543), !dbg !32
  %545 = bitcast i32 %544 to float, !dbg !32
  %546 = fadd float %509, %545, !dbg !32
  %547 = bitcast float %546 to i32, !dbg !32
  %548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %547), !dbg !32
  %549 = bitcast i32 %548 to float, !dbg !32
  %550 = fadd float %546, %549, !dbg !32
  %551 = fadd float %550, %470, !dbg !32
  %552 = bitcast float %542 to i32, !dbg !32
  %553 = call i32 @llvm.amdgcn.ds.bpermute(i32 %212, i32 %552), !dbg !32
  %554 = bitcast i32 %553 to float, !dbg !32
  %555 = fadd float %542, %554, !dbg !32
  %556 = bitcast float %555 to i32, !dbg !32
  %557 = call i32 @llvm.amdgcn.ds.bpermute(i32 %228, i32 %556), !dbg !32
  %558 = bitcast i32 %557 to float, !dbg !32
  %559 = fadd float %555, %558, !dbg !32
  %560 = fadd float %559, %476, !dbg !32
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread117 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local118 = shl nuw nsw i32 %address.thread117, 3, !dbg !16
  %address.local119 = and i32 %address.local118, 504, !dbg !16
  %address.local120 = zext nneg i32 %address.local119 to i64, !dbg !16
  %address.local121 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local120, !dbg !16
  %561 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local121, i32 0) #0, !dbg !16
  %562 = bitcast <4 x i32> %561 to <8 x half>, !dbg !16
  %address.local122 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local120, !dbg !16
  %address.local123 = getelementptr i8, ptr addrspace(3) %address.local122, i64 1024, !dbg !16
  %563 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local123, i32 0) #0, !dbg !16
  %564 = bitcast <4 x i32> %563 to <8 x half>, !dbg !16
  %565 = add nsw i64 %42, %6, !dbg !18
  %566 = shl i64 %565, 19, !dbg !18
  %567 = and i64 %566, 16252928, !dbg !18
  %568 = add i64 %567, %165, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %569 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %568, !dbg !18
  %570 = getelementptr i8, ptr addrspace(1) %569, i64 64, !dbg !18
  br label %571, !dbg !18

571:                                              ; preds = %574, %378
  %572 = phi i64 [ %600, %574 ], [ %20, %378 ], !dbg !18
  %573 = icmp samesign ult i64 %572, 128, !dbg !18
  br i1 %573, label %574, label %601, !dbg !18

574:                                              ; preds = %571
  %575 = shl nuw nsw i64 %572, 3, !dbg !18
  %576 = and i64 %575, 24, !dbg !18
  %577 = shl nuw nsw i64 %572, 3, !dbg !18
  %578 = and i64 %577, 512, !dbg !18
  %579 = shl nuw nsw i64 %572, 5, !dbg !18
  %580 = and i64 %579, 3968, !dbg !18
  %581 = getelementptr [2 x i8], ptr addrspace(1) %570, i64 %580, !dbg !18
  %582 = getelementptr [2 x i8], ptr addrspace(1) %581, i64 %576, !dbg !18
  %583 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %578, !dbg !18
  %584 = ptrtoint ptr addrspace(3) %583 to i64, !dbg !18
  %585 = trunc i64 %584 to i32, !dbg !18
  %586 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %585), !dbg !18
  %587 = zext i32 %586 to i64, !dbg !18
  %588 = inttoptr i64 %587 to ptr addrspace(3), !dbg !18
  %589 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %590 = ptrtoint ptr addrspace(1) %582 to i64, !dbg !18
  %591 = sub i64 %590, %589, !dbg !18
  %dma.buffer.offset441 = trunc i64 %591 to i32, !dbg !18
  %592 = trunc i64 %589 to i32, !dbg !18
  %593 = insertelement <4 x i32> poison, i32 %592, i64 0, !dbg !18
  %594 = lshr i64 %589, 32, !dbg !18
  %595 = trunc i64 %594 to i32, !dbg !18
  %596 = insertelement <4 x i32> %593, i32 %595, i64 1, !dbg !18
  %597 = insertelement <4 x i32> %596, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc442 = insertelement <4 x i32> %597, i32 131072, i64 3, !dbg !18
  %598 = ptrtoint ptr addrspace(3) %588 to i32, !dbg !18
  %dma.buffer.lds443 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %598), !dbg !18
  %599 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds443, i32 %dma.buffer.offset441, <4 x i32> %dma.buffer.rsrc442) #10, !dbg !18
  %600 = or disjoint i64 %20, 256, !dbg !18
  br label %571, !dbg !18

601:                                              ; preds = %571
  %602 = shufflevector <8 x half> %562, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %603 = extractvalue [2 x <8 x float>] %33, 0, !dbg !34
  %604 = shufflevector <8 x float> %603, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %605 = insertelement <4 x float> poison, float %467, i64 0, !dbg !34
  %606 = shufflevector <4 x float> %605, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %607 = fmul <4 x float> %604, %606, !dbg !34
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %602, <4 x float> %607) #9, !dbg !16
  %609 = shufflevector <8 x half> %564, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %609, <4 x float> %608) #9, !dbg !16
  %611 = shufflevector <8 x half> %562, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %612 = shufflevector <8 x float> %603, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %613 = fmul <4 x float> %612, %606, !dbg !34
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %611, <4 x float> %613) #9, !dbg !16
  %615 = shufflevector <8 x half> %564, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %615, <4 x float> %614) #9, !dbg !16
  %617 = extractvalue [2 x <8 x float>] %33, 1, !dbg !34
  %618 = shufflevector <8 x float> %617, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %619 = insertelement <4 x float> poison, float %473, i64 0, !dbg !34
  %620 = shufflevector <4 x float> %619, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %621 = fmul <4 x float> %618, %620, !dbg !34
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %602, <4 x float> %621) #9, !dbg !16
  %623 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %609, <4 x float> %622) #9, !dbg !16
  %624 = shufflevector <8 x float> %617, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %625 = fmul <4 x float> %624, %620, !dbg !34
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %611, <4 x float> %625) #9, !dbg !16
  %627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %615, <4 x float> %626) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread124 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local125 = shl nuw nsw i32 %address.thread124, 3, !dbg !16
  %address.local126 = and i32 %address.local125, 504, !dbg !16
  %address.local127 = zext nneg i32 %address.local126 to i64, !dbg !16
  %address.local128 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local127, !dbg !16
  %628 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local128, i32 0) #0, !dbg !16
  %629 = bitcast <4 x i32> %628 to <8 x half>, !dbg !16
  %address.local129 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local127, !dbg !16
  %address.local130 = getelementptr i8, ptr addrspace(3) %address.local129, i64 1024, !dbg !16
  %630 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local130, i32 0) #0, !dbg !16
  %631 = bitcast <4 x i32> %630 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %632 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %568, !dbg !18
  %633 = getelementptr i8, ptr addrspace(1) %632, i64 128, !dbg !18
  br label %634, !dbg !18

634:                                              ; preds = %637, %601
  %635 = phi i64 [ %663, %637 ], [ %20, %601 ], !dbg !18
  %636 = icmp samesign ult i64 %635, 128, !dbg !18
  br i1 %636, label %637, label %664, !dbg !18

637:                                              ; preds = %634
  %638 = shl nuw nsw i64 %635, 3, !dbg !18
  %639 = and i64 %638, 24, !dbg !18
  %640 = shl nuw nsw i64 %635, 3, !dbg !18
  %641 = and i64 %640, 512, !dbg !18
  %642 = shl nuw nsw i64 %635, 5, !dbg !18
  %643 = and i64 %642, 3968, !dbg !18
  %644 = getelementptr [2 x i8], ptr addrspace(1) %633, i64 %643, !dbg !18
  %645 = getelementptr [2 x i8], ptr addrspace(1) %644, i64 %639, !dbg !18
  %646 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %641, !dbg !18
  %647 = ptrtoint ptr addrspace(3) %646 to i64, !dbg !18
  %648 = trunc i64 %647 to i32, !dbg !18
  %649 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %648), !dbg !18
  %650 = zext i32 %649 to i64, !dbg !18
  %651 = inttoptr i64 %650 to ptr addrspace(3), !dbg !18
  %652 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %653 = ptrtoint ptr addrspace(1) %645 to i64, !dbg !18
  %654 = sub i64 %653, %652, !dbg !18
  %dma.buffer.offset444 = trunc i64 %654 to i32, !dbg !18
  %655 = trunc i64 %652 to i32, !dbg !18
  %656 = insertelement <4 x i32> poison, i32 %655, i64 0, !dbg !18
  %657 = lshr i64 %652, 32, !dbg !18
  %658 = trunc i64 %657 to i32, !dbg !18
  %659 = insertelement <4 x i32> %656, i32 %658, i64 1, !dbg !18
  %660 = insertelement <4 x i32> %659, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc445 = insertelement <4 x i32> %660, i32 131072, i64 3, !dbg !18
  %661 = ptrtoint ptr addrspace(3) %651 to i32, !dbg !18
  %dma.buffer.lds446 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %661), !dbg !18
  %662 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds446, i32 %dma.buffer.offset444, <4 x i32> %dma.buffer.rsrc445) #10, !dbg !18
  %663 = or disjoint i64 %20, 256, !dbg !18
  br label %634, !dbg !18

664:                                              ; preds = %634
  %665 = shufflevector <8 x half> %629, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %666 = extractvalue [2 x <8 x float>] %36, 0, !dbg !34
  %667 = shufflevector <8 x float> %666, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %668 = fmul <4 x float> %667, %606, !dbg !34
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %665, <4 x float> %668) #9, !dbg !16
  %670 = shufflevector <8 x half> %631, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %670, <4 x float> %669) #9, !dbg !16
  %672 = shufflevector <8 x half> %629, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %673 = shufflevector <8 x float> %666, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %674 = fmul <4 x float> %673, %606, !dbg !34
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %672, <4 x float> %674) #9, !dbg !16
  %676 = shufflevector <8 x half> %631, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %676, <4 x float> %675) #9, !dbg !16
  %678 = extractvalue [2 x <8 x float>] %36, 1, !dbg !34
  %679 = shufflevector <8 x float> %678, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %680 = fmul <4 x float> %679, %620, !dbg !34
  %681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %665, <4 x float> %680) #9, !dbg !16
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %670, <4 x float> %681) #9, !dbg !16
  %683 = shufflevector <8 x float> %678, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %684 = fmul <4 x float> %683, %620, !dbg !34
  %685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %672, <4 x float> %684) #9, !dbg !16
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %676, <4 x float> %685) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread131 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local132 = shl nuw nsw i32 %address.thread131, 3, !dbg !16
  %address.local133 = and i32 %address.local132, 504, !dbg !16
  %address.local134 = zext nneg i32 %address.local133 to i64, !dbg !16
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local134, !dbg !16
  %687 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local135, i32 0) #0, !dbg !16
  %688 = bitcast <4 x i32> %687 to <8 x half>, !dbg !16
  %address.local136 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local134, !dbg !16
  %address.local137 = getelementptr i8, ptr addrspace(3) %address.local136, i64 1024, !dbg !16
  %689 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local137, i32 0) #0, !dbg !16
  %690 = bitcast <4 x i32> %689 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %691 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %568, !dbg !18
  %692 = getelementptr i8, ptr addrspace(1) %691, i64 192, !dbg !18
  br label %693, !dbg !18

693:                                              ; preds = %696, %664
  %694 = phi i64 [ %722, %696 ], [ %20, %664 ], !dbg !18
  %695 = icmp samesign ult i64 %694, 128, !dbg !18
  br i1 %695, label %696, label %723, !dbg !18

696:                                              ; preds = %693
  %697 = shl nuw nsw i64 %694, 3, !dbg !18
  %698 = and i64 %697, 24, !dbg !18
  %699 = shl nuw nsw i64 %694, 3, !dbg !18
  %700 = and i64 %699, 512, !dbg !18
  %701 = shl nuw nsw i64 %694, 5, !dbg !18
  %702 = and i64 %701, 3968, !dbg !18
  %703 = getelementptr [2 x i8], ptr addrspace(1) %692, i64 %702, !dbg !18
  %704 = getelementptr [2 x i8], ptr addrspace(1) %703, i64 %698, !dbg !18
  %705 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %700, !dbg !18
  %706 = ptrtoint ptr addrspace(3) %705 to i64, !dbg !18
  %707 = trunc i64 %706 to i32, !dbg !18
  %708 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %707), !dbg !18
  %709 = zext i32 %708 to i64, !dbg !18
  %710 = inttoptr i64 %709 to ptr addrspace(3), !dbg !18
  %711 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %712 = ptrtoint ptr addrspace(1) %704 to i64, !dbg !18
  %713 = sub i64 %712, %711, !dbg !18
  %dma.buffer.offset447 = trunc i64 %713 to i32, !dbg !18
  %714 = trunc i64 %711 to i32, !dbg !18
  %715 = insertelement <4 x i32> poison, i32 %714, i64 0, !dbg !18
  %716 = lshr i64 %711, 32, !dbg !18
  %717 = trunc i64 %716 to i32, !dbg !18
  %718 = insertelement <4 x i32> %715, i32 %717, i64 1, !dbg !18
  %719 = insertelement <4 x i32> %718, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc448 = insertelement <4 x i32> %719, i32 131072, i64 3, !dbg !18
  %720 = ptrtoint ptr addrspace(3) %710 to i32, !dbg !18
  %dma.buffer.lds449 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %720), !dbg !18
  %721 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds449, i32 %dma.buffer.offset447, <4 x i32> %dma.buffer.rsrc448) #10, !dbg !18
  %722 = or disjoint i64 %20, 256, !dbg !18
  br label %693, !dbg !18

723:                                              ; preds = %693
  %724 = shufflevector <8 x half> %688, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %725 = extractvalue [2 x <8 x float>] %37, 0, !dbg !34
  %726 = shufflevector <8 x float> %725, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %727 = fmul <4 x float> %726, %606, !dbg !34
  %728 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %724, <4 x float> %727) #9, !dbg !16
  %729 = shufflevector <8 x half> %690, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %730 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %729, <4 x float> %728) #9, !dbg !16
  %731 = shufflevector <8 x half> %688, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %732 = shufflevector <8 x float> %725, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %733 = fmul <4 x float> %732, %606, !dbg !34
  %734 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %731, <4 x float> %733) #9, !dbg !16
  %735 = shufflevector <8 x half> %690, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %735, <4 x float> %734) #9, !dbg !16
  %737 = extractvalue [2 x <8 x float>] %37, 1, !dbg !34
  %738 = shufflevector <8 x float> %737, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %739 = fmul <4 x float> %738, %620, !dbg !34
  %740 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %724, <4 x float> %739) #9, !dbg !16
  %741 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %729, <4 x float> %740) #9, !dbg !16
  %742 = shufflevector <8 x float> %737, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %743 = fmul <4 x float> %742, %620, !dbg !34
  %744 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %731, <4 x float> %743) #9, !dbg !16
  %745 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %735, <4 x float> %744) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread138 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local139 = shl nuw nsw i32 %address.thread138, 3, !dbg !16
  %address.local140 = and i32 %address.local139, 504, !dbg !16
  %address.local141 = zext nneg i32 %address.local140 to i64, !dbg !16
  %address.local142 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local141, !dbg !16
  %746 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local142, i32 0) #0, !dbg !16
  %747 = bitcast <4 x i32> %746 to <8 x half>, !dbg !16
  %address.local143 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local141, !dbg !16
  %address.local144 = getelementptr i8, ptr addrspace(3) %address.local143, i64 1024, !dbg !16
  %748 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local144, i32 0) #0, !dbg !16
  %749 = bitcast <4 x i32> %748 to <8 x half>, !dbg !16
  %750 = shufflevector <8 x half> %747, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %751 = extractvalue [2 x <8 x float>] %38, 0, !dbg !34
  %752 = shufflevector <8 x float> %751, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %753 = fmul <4 x float> %752, %606, !dbg !34
  %754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %750, <4 x float> %753) #9, !dbg !16
  %755 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %755, <4 x float> %754) #9, !dbg !16
  %757 = shufflevector <8 x half> %747, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %758 = shufflevector <8 x float> %751, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %759 = fmul <4 x float> %758, %606, !dbg !34
  %760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %757, <4 x float> %759) #9, !dbg !16
  %761 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %762 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %761, <4 x float> %760) #9, !dbg !16
  %763 = extractvalue [2 x <8 x float>] %38, 1, !dbg !34
  %764 = shufflevector <8 x float> %763, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %765 = fmul <4 x float> %764, %620, !dbg !34
  %766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %750, <4 x float> %765) #9, !dbg !16
  %767 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %755, <4 x float> %766) #9, !dbg !16
  %768 = shufflevector <8 x float> %763, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %769 = fmul <4 x float> %768, %620, !dbg !34
  %770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %757, <4 x float> %769) #9, !dbg !16
  %771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %761, <4 x float> %770) #9, !dbg !16
  %772 = shufflevector <4 x float> %610, <4 x float> %616, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %773 = insertvalue [2 x <8 x float>] poison, <8 x float> %772, 0, !dbg !8
  %774 = shufflevector <4 x float> %623, <4 x float> %627, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %775 = insertvalue [2 x <8 x float>] %773, <8 x float> %774, 1, !dbg !8
  %776 = insertelement <1 x float> poison, float %551, i64 0, !dbg !8
  %777 = insertvalue [2 x <1 x float>] poison, <1 x float> %776, 0, !dbg !8
  %778 = insertelement <1 x float> poison, float %560, i64 0, !dbg !8
  %779 = insertvalue [2 x <1 x float>] %777, <1 x float> %778, 1, !dbg !8
  %780 = insertvalue [2 x <1 x float>] poison, <1 x float> %439, 0, !dbg !8
  %781 = insertvalue [2 x <1 x float>] %780, <1 x float> %458, 1, !dbg !8
  %782 = shufflevector <4 x float> %671, <4 x float> %677, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %783 = insertvalue [2 x <8 x float>] poison, <8 x float> %782, 0, !dbg !8
  %784 = shufflevector <4 x float> %682, <4 x float> %686, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %785 = insertvalue [2 x <8 x float>] %783, <8 x float> %784, 1, !dbg !8
  %786 = shufflevector <4 x float> %730, <4 x float> %736, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %787 = insertvalue [2 x <8 x float>] poison, <8 x float> %786, 0, !dbg !8
  %788 = shufflevector <4 x float> %741, <4 x float> %745, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %789 = insertvalue [2 x <8 x float>] %787, <8 x float> %788, 1, !dbg !8
  %790 = shufflevector <4 x float> %756, <4 x float> %762, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %791 = insertvalue [2 x <8 x float>] poison, <8 x float> %790, 0, !dbg !8
  %792 = shufflevector <4 x float> %767, <4 x float> %771, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %793 = insertvalue [2 x <8 x float>] %791, <8 x float> %792, 1, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %794 = add nuw nsw i64 %32, 1, !dbg !8
  br label %31, !dbg !8

795:                                              ; preds = %31
  %796 = extractvalue [2 x <8 x float>] %38, 0, !dbg !8
  %797 = extractelement <8 x float> %796, i64 0, !dbg !8
  %798 = extractelement <8 x float> %796, i64 1, !dbg !8
  %799 = extractelement <8 x float> %796, i64 2, !dbg !8
  %800 = extractelement <8 x float> %796, i64 3, !dbg !8
  %801 = select i1 %.not, float %798, float %797, !dbg !8
  %802 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %803 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %802), !dbg !8
  %804 = add nuw nsw i32 %803, 64, !dbg !8
  %805 = xor i32 %803, 16, !dbg !8
  %806 = and i32 %804, 192, !dbg !8
  %807 = icmp samesign ult i32 %805, %806, !dbg !8
  %808 = select i1 %807, i32 %805, i32 %803, !dbg !8
  %809 = shl nuw nsw i32 %808, 2, !dbg !8
  %810 = bitcast float %801 to i32, !dbg !8
  %811 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %810), !dbg !8
  %812 = bitcast i32 %811 to float, !dbg !8
  %813 = select i1 %.not, float %797, float %812, !dbg !8
  %814 = select i1 %.not, float %812, float %798, !dbg !8
  %815 = select i1 %.not, float %800, float %799, !dbg !8
  %816 = bitcast float %815 to i32, !dbg !8
  %817 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %816), !dbg !8
  %818 = bitcast i32 %817 to float, !dbg !8
  %819 = select i1 %.not, float %799, float %818, !dbg !8
  %820 = select i1 %.not, float %818, float %800, !dbg !8
  %821 = select i1 %.not2, float %819, float %813, !dbg !8
  %822 = xor i32 %803, 32, !dbg !8
  %823 = icmp samesign ult i32 %822, %806, !dbg !8
  %824 = select i1 %823, i32 %822, i32 %803, !dbg !8
  %825 = shl nuw nsw i32 %824, 2, !dbg !8
  %826 = bitcast float %821 to i32, !dbg !8
  %827 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %826), !dbg !8
  %828 = bitcast i32 %827 to float, !dbg !8
  %829 = select i1 %.not2, float %813, float %828, !dbg !8
  %830 = select i1 %.not2, float %828, float %819, !dbg !8
  %831 = select i1 %.not2, float %820, float %814, !dbg !8
  %832 = bitcast float %831 to i32, !dbg !8
  %833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %832), !dbg !8
  %834 = bitcast i32 %833 to float, !dbg !8
  %835 = select i1 %.not2, float %814, float %834, !dbg !8
  %836 = select i1 %.not2, float %834, float %820, !dbg !8
  %837 = extractvalue [2 x <8 x float>] %37, 0, !dbg !8
  %838 = extractelement <8 x float> %837, i64 0, !dbg !8
  %839 = extractelement <8 x float> %837, i64 1, !dbg !8
  %840 = extractelement <8 x float> %837, i64 2, !dbg !8
  %841 = extractelement <8 x float> %837, i64 3, !dbg !8
  %842 = select i1 %.not, float %839, float %838, !dbg !8
  %843 = bitcast float %842 to i32, !dbg !8
  %844 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %843), !dbg !8
  %845 = bitcast i32 %844 to float, !dbg !8
  %846 = select i1 %.not, float %838, float %845, !dbg !8
  %847 = select i1 %.not, float %845, float %839, !dbg !8
  %848 = select i1 %.not, float %841, float %840, !dbg !8
  %849 = bitcast float %848 to i32, !dbg !8
  %850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %849), !dbg !8
  %851 = bitcast i32 %850 to float, !dbg !8
  %852 = select i1 %.not, float %840, float %851, !dbg !8
  %853 = select i1 %.not, float %851, float %841, !dbg !8
  %854 = select i1 %.not2, float %852, float %846, !dbg !8
  %855 = bitcast float %854 to i32, !dbg !8
  %856 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %855), !dbg !8
  %857 = bitcast i32 %856 to float, !dbg !8
  %858 = select i1 %.not2, float %846, float %857, !dbg !8
  %859 = select i1 %.not2, float %857, float %852, !dbg !8
  %860 = select i1 %.not2, float %853, float %847, !dbg !8
  %861 = bitcast float %860 to i32, !dbg !8
  %862 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %861), !dbg !8
  %863 = bitcast i32 %862 to float, !dbg !8
  %864 = select i1 %.not2, float %847, float %863, !dbg !8
  %865 = select i1 %.not2, float %863, float %853, !dbg !8
  %866 = extractvalue [2 x <8 x float>] %36, 0, !dbg !8
  %867 = extractelement <8 x float> %866, i64 0, !dbg !8
  %868 = extractelement <8 x float> %866, i64 1, !dbg !8
  %869 = extractelement <8 x float> %866, i64 2, !dbg !8
  %870 = extractelement <8 x float> %866, i64 3, !dbg !8
  %871 = select i1 %.not, float %868, float %867, !dbg !8
  %872 = bitcast float %871 to i32, !dbg !8
  %873 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %872), !dbg !8
  %874 = bitcast i32 %873 to float, !dbg !8
  %875 = select i1 %.not, float %867, float %874, !dbg !8
  %876 = select i1 %.not, float %874, float %868, !dbg !8
  %877 = select i1 %.not, float %870, float %869, !dbg !8
  %878 = bitcast float %877 to i32, !dbg !8
  %879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %878), !dbg !8
  %880 = bitcast i32 %879 to float, !dbg !8
  %881 = select i1 %.not, float %869, float %880, !dbg !8
  %882 = select i1 %.not, float %880, float %870, !dbg !8
  %883 = select i1 %.not2, float %881, float %875, !dbg !8
  %884 = bitcast float %883 to i32, !dbg !8
  %885 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %884), !dbg !8
  %886 = bitcast i32 %885 to float, !dbg !8
  %887 = select i1 %.not2, float %875, float %886, !dbg !8
  %888 = select i1 %.not2, float %886, float %881, !dbg !8
  %889 = select i1 %.not2, float %882, float %876, !dbg !8
  %890 = bitcast float %889 to i32, !dbg !8
  %891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %890), !dbg !8
  %892 = bitcast i32 %891 to float, !dbg !8
  %893 = select i1 %.not2, float %876, float %892, !dbg !8
  %894 = select i1 %.not2, float %892, float %882, !dbg !8
  %895 = extractvalue [2 x <8 x float>] %33, 0, !dbg !8
  %896 = extractelement <8 x float> %895, i64 0, !dbg !8
  %897 = extractelement <8 x float> %895, i64 1, !dbg !8
  %898 = extractelement <8 x float> %895, i64 2, !dbg !8
  %899 = extractelement <8 x float> %895, i64 3, !dbg !8
  %900 = select i1 %.not, float %897, float %896, !dbg !8
  %901 = bitcast float %900 to i32, !dbg !8
  %902 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %901), !dbg !8
  %903 = bitcast i32 %902 to float, !dbg !8
  %904 = select i1 %.not, float %896, float %903, !dbg !8
  %905 = select i1 %.not, float %903, float %897, !dbg !8
  %906 = select i1 %.not, float %899, float %898, !dbg !8
  %907 = bitcast float %906 to i32, !dbg !8
  %908 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %907), !dbg !8
  %909 = bitcast i32 %908 to float, !dbg !8
  %910 = select i1 %.not, float %898, float %909, !dbg !8
  %911 = select i1 %.not, float %909, float %899, !dbg !8
  %912 = select i1 %.not2, float %910, float %904, !dbg !8
  %913 = bitcast float %912 to i32, !dbg !8
  %914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %913), !dbg !8
  %915 = select i1 %.not2, float %911, float %905, !dbg !8
  %916 = bitcast float %915 to i32, !dbg !8
  %917 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %916), !dbg !8
  %918 = extractvalue [2 x <1 x float>] %34, 0, !dbg !19
  %919 = shufflevector <1 x float> %918, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %920 = extractelement <8 x float> %796, i64 4, !dbg !8
  %921 = extractelement <8 x float> %796, i64 5, !dbg !8
  %922 = extractelement <8 x float> %796, i64 6, !dbg !8
  %923 = extractelement <8 x float> %796, i64 7, !dbg !8
  %924 = select i1 %.not, float %921, float %920, !dbg !8
  %925 = bitcast float %924 to i32, !dbg !8
  %926 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %925), !dbg !8
  %927 = bitcast i32 %926 to float, !dbg !8
  %928 = select i1 %.not, float %920, float %927, !dbg !8
  %929 = select i1 %.not, float %927, float %921, !dbg !8
  %930 = select i1 %.not, float %923, float %922, !dbg !8
  %931 = bitcast float %930 to i32, !dbg !8
  %932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %931), !dbg !8
  %933 = bitcast i32 %932 to float, !dbg !8
  %934 = select i1 %.not, float %922, float %933, !dbg !8
  %935 = select i1 %.not, float %933, float %923, !dbg !8
  %936 = select i1 %.not2, float %934, float %928, !dbg !8
  %937 = bitcast float %936 to i32, !dbg !8
  %938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %937), !dbg !8
  %939 = bitcast i32 %938 to float, !dbg !8
  %940 = select i1 %.not2, float %928, float %939, !dbg !8
  %941 = select i1 %.not2, float %939, float %934, !dbg !8
  %942 = select i1 %.not2, float %935, float %929, !dbg !8
  %943 = bitcast float %942 to i32, !dbg !8
  %944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %943), !dbg !8
  %945 = bitcast i32 %944 to float, !dbg !8
  %946 = select i1 %.not2, float %929, float %945, !dbg !8
  %947 = select i1 %.not2, float %945, float %935, !dbg !8
  %948 = extractelement <8 x float> %837, i64 4, !dbg !8
  %949 = extractelement <8 x float> %837, i64 5, !dbg !8
  %950 = extractelement <8 x float> %837, i64 6, !dbg !8
  %951 = extractelement <8 x float> %837, i64 7, !dbg !8
  %952 = select i1 %.not, float %949, float %948, !dbg !8
  %953 = bitcast float %952 to i32, !dbg !8
  %954 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %953), !dbg !8
  %955 = bitcast i32 %954 to float, !dbg !8
  %956 = select i1 %.not, float %948, float %955, !dbg !8
  %957 = select i1 %.not, float %955, float %949, !dbg !8
  %958 = select i1 %.not, float %951, float %950, !dbg !8
  %959 = bitcast float %958 to i32, !dbg !8
  %960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %959), !dbg !8
  %961 = bitcast i32 %960 to float, !dbg !8
  %962 = select i1 %.not, float %950, float %961, !dbg !8
  %963 = select i1 %.not, float %961, float %951, !dbg !8
  %964 = select i1 %.not2, float %962, float %956, !dbg !8
  %965 = bitcast float %964 to i32, !dbg !8
  %966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %965), !dbg !8
  %967 = bitcast i32 %966 to float, !dbg !8
  %968 = select i1 %.not2, float %956, float %967, !dbg !8
  %969 = select i1 %.not2, float %967, float %962, !dbg !8
  %970 = select i1 %.not2, float %963, float %957, !dbg !8
  %971 = bitcast float %970 to i32, !dbg !8
  %972 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %971), !dbg !8
  %973 = bitcast i32 %972 to float, !dbg !8
  %974 = select i1 %.not2, float %957, float %973, !dbg !8
  %975 = select i1 %.not2, float %973, float %963, !dbg !8
  %976 = extractelement <8 x float> %866, i64 4, !dbg !8
  %977 = extractelement <8 x float> %866, i64 5, !dbg !8
  %978 = extractelement <8 x float> %866, i64 6, !dbg !8
  %979 = extractelement <8 x float> %866, i64 7, !dbg !8
  %980 = select i1 %.not, float %977, float %976, !dbg !8
  %981 = bitcast float %980 to i32, !dbg !8
  %982 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %981), !dbg !8
  %983 = bitcast i32 %982 to float, !dbg !8
  %984 = select i1 %.not, float %976, float %983, !dbg !8
  %985 = select i1 %.not, float %983, float %977, !dbg !8
  %986 = select i1 %.not, float %979, float %978, !dbg !8
  %987 = bitcast float %986 to i32, !dbg !8
  %988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %987), !dbg !8
  %989 = bitcast i32 %988 to float, !dbg !8
  %990 = select i1 %.not, float %978, float %989, !dbg !8
  %991 = select i1 %.not, float %989, float %979, !dbg !8
  %992 = select i1 %.not2, float %990, float %984, !dbg !8
  %993 = bitcast float %992 to i32, !dbg !8
  %994 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %993), !dbg !8
  %995 = bitcast i32 %994 to float, !dbg !8
  %996 = select i1 %.not2, float %984, float %995, !dbg !8
  %997 = select i1 %.not2, float %995, float %990, !dbg !8
  %998 = select i1 %.not2, float %991, float %985, !dbg !8
  %999 = bitcast float %998 to i32, !dbg !8
  %1000 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %999), !dbg !8
  %1001 = bitcast i32 %1000 to float, !dbg !8
  %1002 = select i1 %.not2, float %985, float %1001, !dbg !8
  %1003 = select i1 %.not2, float %1001, float %991, !dbg !8
  %1004 = extractelement <8 x float> %895, i64 4, !dbg !8
  %1005 = extractelement <8 x float> %895, i64 5, !dbg !8
  %1006 = extractelement <8 x float> %895, i64 6, !dbg !8
  %1007 = extractelement <8 x float> %895, i64 7, !dbg !8
  %1008 = select i1 %.not, float %1005, float %1004, !dbg !8
  %1009 = bitcast float %1008 to i32, !dbg !8
  %1010 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1009), !dbg !8
  %1011 = bitcast i32 %1010 to float, !dbg !8
  %1012 = select i1 %.not, float %1004, float %1011, !dbg !8
  %1013 = select i1 %.not, float %1011, float %1005, !dbg !8
  %1014 = select i1 %.not, float %1007, float %1006, !dbg !8
  %1015 = bitcast float %1014 to i32, !dbg !8
  %1016 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1015), !dbg !8
  %1017 = bitcast i32 %1016 to float, !dbg !8
  %1018 = select i1 %.not, float %1006, float %1017, !dbg !8
  %1019 = select i1 %.not, float %1017, float %1007, !dbg !8
  %1020 = select i1 %.not2, float %1018, float %1012, !dbg !8
  %1021 = bitcast float %1020 to i32, !dbg !8
  %1022 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1021), !dbg !8
  %1023 = select i1 %.not2, float %1019, float %1013, !dbg !8
  %1024 = bitcast float %1023 to i32, !dbg !8
  %1025 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1024), !dbg !8
  %1026 = extractvalue [2 x <8 x float>] %38, 1, !dbg !8
  %1027 = extractelement <8 x float> %1026, i64 0, !dbg !8
  %1028 = extractelement <8 x float> %1026, i64 1, !dbg !8
  %1029 = extractelement <8 x float> %1026, i64 2, !dbg !8
  %1030 = extractelement <8 x float> %1026, i64 3, !dbg !8
  %1031 = select i1 %.not, float %1028, float %1027, !dbg !8
  %1032 = bitcast float %1031 to i32, !dbg !8
  %1033 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1032), !dbg !8
  %1034 = bitcast i32 %1033 to float, !dbg !8
  %1035 = select i1 %.not, float %1027, float %1034, !dbg !8
  %1036 = select i1 %.not, float %1034, float %1028, !dbg !8
  %1037 = select i1 %.not, float %1030, float %1029, !dbg !8
  %1038 = bitcast float %1037 to i32, !dbg !8
  %1039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1038), !dbg !8
  %1040 = bitcast i32 %1039 to float, !dbg !8
  %1041 = select i1 %.not, float %1029, float %1040, !dbg !8
  %1042 = select i1 %.not, float %1040, float %1030, !dbg !8
  %1043 = select i1 %.not2, float %1041, float %1035, !dbg !8
  %1044 = bitcast float %1043 to i32, !dbg !8
  %1045 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1044), !dbg !8
  %1046 = bitcast i32 %1045 to float, !dbg !8
  %1047 = select i1 %.not2, float %1035, float %1046, !dbg !8
  %1048 = select i1 %.not2, float %1046, float %1041, !dbg !8
  %1049 = select i1 %.not2, float %1042, float %1036, !dbg !8
  %1050 = bitcast float %1049 to i32, !dbg !8
  %1051 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1050), !dbg !8
  %1052 = bitcast i32 %1051 to float, !dbg !8
  %1053 = select i1 %.not2, float %1036, float %1052, !dbg !8
  %1054 = select i1 %.not2, float %1052, float %1042, !dbg !8
  %1055 = extractvalue [2 x <8 x float>] %37, 1, !dbg !8
  %1056 = extractelement <8 x float> %1055, i64 0, !dbg !8
  %1057 = extractelement <8 x float> %1055, i64 1, !dbg !8
  %1058 = extractelement <8 x float> %1055, i64 2, !dbg !8
  %1059 = extractelement <8 x float> %1055, i64 3, !dbg !8
  %1060 = select i1 %.not, float %1057, float %1056, !dbg !8
  %1061 = bitcast float %1060 to i32, !dbg !8
  %1062 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1061), !dbg !8
  %1063 = bitcast i32 %1062 to float, !dbg !8
  %1064 = select i1 %.not, float %1056, float %1063, !dbg !8
  %1065 = select i1 %.not, float %1063, float %1057, !dbg !8
  %1066 = select i1 %.not, float %1059, float %1058, !dbg !8
  %1067 = bitcast float %1066 to i32, !dbg !8
  %1068 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1067), !dbg !8
  %1069 = bitcast i32 %1068 to float, !dbg !8
  %1070 = select i1 %.not, float %1058, float %1069, !dbg !8
  %1071 = select i1 %.not, float %1069, float %1059, !dbg !8
  %1072 = select i1 %.not2, float %1070, float %1064, !dbg !8
  %1073 = bitcast float %1072 to i32, !dbg !8
  %1074 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1073), !dbg !8
  %1075 = bitcast i32 %1074 to float, !dbg !8
  %1076 = select i1 %.not2, float %1064, float %1075, !dbg !8
  %1077 = select i1 %.not2, float %1075, float %1070, !dbg !8
  %1078 = select i1 %.not2, float %1071, float %1065, !dbg !8
  %1079 = bitcast float %1078 to i32, !dbg !8
  %1080 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1079), !dbg !8
  %1081 = bitcast i32 %1080 to float, !dbg !8
  %1082 = select i1 %.not2, float %1065, float %1081, !dbg !8
  %1083 = select i1 %.not2, float %1081, float %1071, !dbg !8
  %1084 = extractvalue [2 x <8 x float>] %36, 1, !dbg !8
  %1085 = extractelement <8 x float> %1084, i64 0, !dbg !8
  %1086 = extractelement <8 x float> %1084, i64 1, !dbg !8
  %1087 = extractelement <8 x float> %1084, i64 2, !dbg !8
  %1088 = extractelement <8 x float> %1084, i64 3, !dbg !8
  %1089 = select i1 %.not, float %1086, float %1085, !dbg !8
  %1090 = bitcast float %1089 to i32, !dbg !8
  %1091 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1090), !dbg !8
  %1092 = bitcast i32 %1091 to float, !dbg !8
  %1093 = select i1 %.not, float %1085, float %1092, !dbg !8
  %1094 = select i1 %.not, float %1092, float %1086, !dbg !8
  %1095 = select i1 %.not, float %1088, float %1087, !dbg !8
  %1096 = bitcast float %1095 to i32, !dbg !8
  %1097 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1096), !dbg !8
  %1098 = bitcast i32 %1097 to float, !dbg !8
  %1099 = select i1 %.not, float %1087, float %1098, !dbg !8
  %1100 = select i1 %.not, float %1098, float %1088, !dbg !8
  %1101 = select i1 %.not2, float %1099, float %1093, !dbg !8
  %1102 = bitcast float %1101 to i32, !dbg !8
  %1103 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1102), !dbg !8
  %1104 = bitcast i32 %1103 to float, !dbg !8
  %1105 = select i1 %.not2, float %1093, float %1104, !dbg !8
  %1106 = select i1 %.not2, float %1104, float %1099, !dbg !8
  %1107 = select i1 %.not2, float %1100, float %1094, !dbg !8
  %1108 = bitcast float %1107 to i32, !dbg !8
  %1109 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1108), !dbg !8
  %1110 = bitcast i32 %1109 to float, !dbg !8
  %1111 = select i1 %.not2, float %1094, float %1110, !dbg !8
  %1112 = select i1 %.not2, float %1110, float %1100, !dbg !8
  %1113 = extractvalue [2 x <8 x float>] %33, 1, !dbg !8
  %1114 = extractelement <8 x float> %1113, i64 0, !dbg !8
  %1115 = extractelement <8 x float> %1113, i64 1, !dbg !8
  %1116 = extractelement <8 x float> %1113, i64 2, !dbg !8
  %1117 = extractelement <8 x float> %1113, i64 3, !dbg !8
  %1118 = select i1 %.not, float %1115, float %1114, !dbg !8
  %1119 = bitcast float %1118 to i32, !dbg !8
  %1120 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1119), !dbg !8
  %1121 = bitcast i32 %1120 to float, !dbg !8
  %1122 = select i1 %.not, float %1114, float %1121, !dbg !8
  %1123 = select i1 %.not, float %1121, float %1115, !dbg !8
  %1124 = select i1 %.not, float %1117, float %1116, !dbg !8
  %1125 = bitcast float %1124 to i32, !dbg !8
  %1126 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1125), !dbg !8
  %1127 = bitcast i32 %1126 to float, !dbg !8
  %1128 = select i1 %.not, float %1116, float %1127, !dbg !8
  %1129 = select i1 %.not, float %1127, float %1117, !dbg !8
  %1130 = select i1 %.not2, float %1128, float %1122, !dbg !8
  %1131 = bitcast float %1130 to i32, !dbg !8
  %1132 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1131), !dbg !8
  %1133 = select i1 %.not2, float %1129, float %1123, !dbg !8
  %1134 = bitcast float %1133 to i32, !dbg !8
  %1135 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1134), !dbg !8
  %1136 = extractvalue [2 x <1 x float>] %34, 1, !dbg !19
  %1137 = shufflevector <1 x float> %1136, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1138 = extractelement <8 x float> %1026, i64 4, !dbg !8
  %1139 = extractelement <8 x float> %1026, i64 5, !dbg !8
  %1140 = extractelement <8 x float> %1026, i64 6, !dbg !8
  %1141 = extractelement <8 x float> %1026, i64 7, !dbg !8
  %1142 = select i1 %.not, float %1139, float %1138, !dbg !8
  %1143 = bitcast float %1142 to i32, !dbg !8
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1143), !dbg !8
  %1145 = bitcast i32 %1144 to float, !dbg !8
  %1146 = select i1 %.not, float %1138, float %1145, !dbg !8
  %1147 = select i1 %.not, float %1145, float %1139, !dbg !8
  %1148 = select i1 %.not, float %1141, float %1140, !dbg !8
  %1149 = bitcast float %1148 to i32, !dbg !8
  %1150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1149), !dbg !8
  %1151 = bitcast i32 %1150 to float, !dbg !8
  %1152 = select i1 %.not, float %1140, float %1151, !dbg !8
  %1153 = select i1 %.not, float %1151, float %1141, !dbg !8
  %1154 = select i1 %.not2, float %1152, float %1146, !dbg !8
  %1155 = bitcast float %1154 to i32, !dbg !8
  %1156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1155), !dbg !8
  %1157 = bitcast i32 %1156 to float, !dbg !8
  %1158 = select i1 %.not2, float %1146, float %1157, !dbg !8
  %1159 = select i1 %.not2, float %1157, float %1152, !dbg !8
  %1160 = select i1 %.not2, float %1153, float %1147, !dbg !8
  %1161 = bitcast float %1160 to i32, !dbg !8
  %1162 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1161), !dbg !8
  %1163 = bitcast i32 %1162 to float, !dbg !8
  %1164 = select i1 %.not2, float %1147, float %1163, !dbg !8
  %1165 = select i1 %.not2, float %1163, float %1153, !dbg !8
  %1166 = extractelement <8 x float> %1055, i64 4, !dbg !8
  %1167 = extractelement <8 x float> %1055, i64 5, !dbg !8
  %1168 = extractelement <8 x float> %1055, i64 6, !dbg !8
  %1169 = extractelement <8 x float> %1055, i64 7, !dbg !8
  %1170 = select i1 %.not, float %1167, float %1166, !dbg !8
  %1171 = bitcast float %1170 to i32, !dbg !8
  %1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1171), !dbg !8
  %1173 = bitcast i32 %1172 to float, !dbg !8
  %1174 = select i1 %.not, float %1166, float %1173, !dbg !8
  %1175 = select i1 %.not, float %1173, float %1167, !dbg !8
  %1176 = select i1 %.not, float %1169, float %1168, !dbg !8
  %1177 = bitcast float %1176 to i32, !dbg !8
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1177), !dbg !8
  %1179 = bitcast i32 %1178 to float, !dbg !8
  %1180 = select i1 %.not, float %1168, float %1179, !dbg !8
  %1181 = select i1 %.not, float %1179, float %1169, !dbg !8
  %1182 = select i1 %.not2, float %1180, float %1174, !dbg !8
  %1183 = bitcast float %1182 to i32, !dbg !8
  %1184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1183), !dbg !8
  %1185 = bitcast i32 %1184 to float, !dbg !8
  %1186 = select i1 %.not2, float %1174, float %1185, !dbg !8
  %1187 = select i1 %.not2, float %1185, float %1180, !dbg !8
  %1188 = select i1 %.not2, float %1181, float %1175, !dbg !8
  %1189 = bitcast float %1188 to i32, !dbg !8
  %1190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1189), !dbg !8
  %1191 = bitcast i32 %1190 to float, !dbg !8
  %1192 = select i1 %.not2, float %1175, float %1191, !dbg !8
  %1193 = select i1 %.not2, float %1191, float %1181, !dbg !8
  %1194 = extractelement <8 x float> %1084, i64 4, !dbg !8
  %1195 = extractelement <8 x float> %1084, i64 5, !dbg !8
  %1196 = extractelement <8 x float> %1084, i64 6, !dbg !8
  %1197 = extractelement <8 x float> %1084, i64 7, !dbg !8
  %1198 = select i1 %.not, float %1195, float %1194, !dbg !8
  %1199 = bitcast float %1198 to i32, !dbg !8
  %1200 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1199), !dbg !8
  %1201 = bitcast i32 %1200 to float, !dbg !8
  %1202 = select i1 %.not, float %1194, float %1201, !dbg !8
  %1203 = select i1 %.not, float %1201, float %1195, !dbg !8
  %1204 = select i1 %.not, float %1197, float %1196, !dbg !8
  %1205 = bitcast float %1204 to i32, !dbg !8
  %1206 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1205), !dbg !8
  %1207 = bitcast i32 %1206 to float, !dbg !8
  %1208 = select i1 %.not, float %1196, float %1207, !dbg !8
  %1209 = select i1 %.not, float %1207, float %1197, !dbg !8
  %1210 = select i1 %.not2, float %1208, float %1202, !dbg !8
  %1211 = bitcast float %1210 to i32, !dbg !8
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1211), !dbg !8
  %1213 = bitcast i32 %1212 to float, !dbg !8
  %1214 = select i1 %.not2, float %1202, float %1213, !dbg !8
  %1215 = select i1 %.not2, float %1213, float %1208, !dbg !8
  %1216 = select i1 %.not2, float %1209, float %1203, !dbg !8
  %1217 = bitcast float %1216 to i32, !dbg !8
  %1218 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1217), !dbg !8
  %1219 = bitcast i32 %1218 to float, !dbg !8
  %1220 = select i1 %.not2, float %1203, float %1219, !dbg !8
  %1221 = select i1 %.not2, float %1219, float %1209, !dbg !8
  %1222 = extractelement <8 x float> %1113, i64 4, !dbg !8
  %1223 = extractelement <8 x float> %1113, i64 5, !dbg !8
  %1224 = extractelement <8 x float> %1113, i64 6, !dbg !8
  %1225 = extractelement <8 x float> %1113, i64 7, !dbg !8
  %1226 = select i1 %.not, float %1223, float %1222, !dbg !8
  %1227 = bitcast float %1226 to i32, !dbg !8
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1227), !dbg !8
  %1229 = bitcast i32 %1228 to float, !dbg !8
  %1230 = select i1 %.not, float %1222, float %1229, !dbg !8
  %1231 = select i1 %.not, float %1229, float %1223, !dbg !8
  %1232 = select i1 %.not, float %1225, float %1224, !dbg !8
  %1233 = bitcast float %1232 to i32, !dbg !8
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %809, i32 %1233), !dbg !8
  %1235 = bitcast i32 %1234 to float, !dbg !8
  %1236 = select i1 %.not, float %1224, float %1235, !dbg !8
  %1237 = select i1 %.not, float %1235, float %1225, !dbg !8
  %1238 = select i1 %.not2, float %1236, float %1230, !dbg !8
  %1239 = bitcast float %1238 to i32, !dbg !8
  %1240 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1239), !dbg !8
  %1241 = select i1 %.not2, float %1237, float %1231, !dbg !8
  %1242 = bitcast float %1241 to i32, !dbg !8
  %1243 = call i32 @llvm.amdgcn.ds.bpermute(i32 %825, i32 %1242), !dbg !8
  br i1 true, label %1244, label %1308, !dbg !21

1244:                                             ; preds = %795
  %1245 = bitcast i32 %1240 to float, !dbg !8
  %1246 = select i1 %.not2, float %1230, float %1245, !dbg !8
  %1247 = insertelement <4 x float> poison, float %1246, i64 0, !dbg !19
  %1248 = bitcast i32 %1243 to float, !dbg !8
  %1249 = select i1 %.not2, float %1231, float %1248, !dbg !8
  %1250 = insertelement <4 x float> %1247, float %1249, i64 1, !dbg !19
  %1251 = select i1 %.not2, float %1245, float %1236, !dbg !8
  %1252 = insertelement <4 x float> %1250, float %1251, i64 2, !dbg !19
  %1253 = select i1 %.not2, float %1248, float %1237, !dbg !8
  %1254 = insertelement <4 x float> %1252, float %1253, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1137, !dbg !19
  %normalized = fmul <4 x float> %1254, %row.reciprocal, !dbg !19
  %1255 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1256 = extractelement <4 x half> %1255, i64 3, !dbg !20
  %1257 = extractelement <4 x half> %1255, i64 2, !dbg !20
  %1258 = extractelement <4 x half> %1255, i64 1, !dbg !20
  %1259 = extractelement <4 x half> %1255, i64 0, !dbg !20
  %1260 = bitcast i32 %1132 to float, !dbg !8
  %1261 = select i1 %.not2, float %1122, float %1260, !dbg !8
  %1262 = insertelement <4 x float> poison, float %1261, i64 0, !dbg !19
  %1263 = bitcast i32 %1135 to float, !dbg !8
  %1264 = select i1 %.not2, float %1123, float %1263, !dbg !8
  %1265 = insertelement <4 x float> %1262, float %1264, i64 1, !dbg !19
  %1266 = select i1 %.not2, float %1260, float %1128, !dbg !8
  %1267 = insertelement <4 x float> %1265, float %1266, i64 2, !dbg !19
  %1268 = select i1 %.not2, float %1263, float %1129, !dbg !8
  %1269 = insertelement <4 x float> %1267, float %1268, i64 3, !dbg !19
  %normalized23 = fmul <4 x float> %1269, %row.reciprocal, !dbg !19
  %1270 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !20
  %1271 = extractelement <4 x half> %1270, i64 3, !dbg !20
  %1272 = extractelement <4 x half> %1270, i64 2, !dbg !20
  %1273 = extractelement <4 x half> %1270, i64 1, !dbg !20
  %1274 = extractelement <4 x half> %1270, i64 0, !dbg !20
  %1275 = bitcast i32 %1022 to float, !dbg !8
  %1276 = select i1 %.not2, float %1012, float %1275, !dbg !8
  %1277 = insertelement <4 x float> poison, float %1276, i64 0, !dbg !19
  %1278 = bitcast i32 %1025 to float, !dbg !8
  %1279 = select i1 %.not2, float %1013, float %1278, !dbg !8
  %1280 = insertelement <4 x float> %1277, float %1279, i64 1, !dbg !19
  %1281 = select i1 %.not2, float %1275, float %1018, !dbg !8
  %1282 = insertelement <4 x float> %1280, float %1281, i64 2, !dbg !19
  %1283 = select i1 %.not2, float %1278, float %1019, !dbg !8
  %1284 = insertelement <4 x float> %1282, float %1283, i64 3, !dbg !19
  %row.reciprocal24 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %919, !dbg !19
  %normalized25 = fmul <4 x float> %1284, %row.reciprocal24, !dbg !19
  %1285 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !20
  %1286 = extractelement <4 x half> %1285, i64 3, !dbg !20
  %1287 = extractelement <4 x half> %1285, i64 2, !dbg !20
  %1288 = extractelement <4 x half> %1285, i64 1, !dbg !20
  %1289 = extractelement <4 x half> %1285, i64 0, !dbg !20
  %1290 = bitcast i32 %914 to float, !dbg !8
  %1291 = select i1 %.not2, float %904, float %1290, !dbg !8
  %1292 = insertelement <4 x float> poison, float %1291, i64 0, !dbg !19
  %1293 = bitcast i32 %917 to float, !dbg !8
  %1294 = select i1 %.not2, float %905, float %1293, !dbg !8
  %1295 = insertelement <4 x float> %1292, float %1294, i64 1, !dbg !19
  %1296 = select i1 %.not2, float %1290, float %910, !dbg !8
  %1297 = insertelement <4 x float> %1295, float %1296, i64 2, !dbg !19
  %1298 = select i1 %.not2, float %1293, float %911, !dbg !8
  %1299 = insertelement <4 x float> %1297, float %1298, i64 3, !dbg !19
  %normalized26 = fmul <4 x float> %1299, %row.reciprocal24, !dbg !19
  %1300 = fptrunc <4 x float> %normalized26 to <4 x half>, !dbg !20
  %1301 = extractelement <4 x half> %1300, i64 3, !dbg !20
  %1302 = extractelement <4 x half> %1300, i64 2, !dbg !20
  %1303 = extractelement <4 x half> %1300, i64 1, !dbg !20
  %1304 = extractelement <4 x half> %1300, i64 0, !dbg !20
  %1305 = add nsw i64 %6, %13, !dbg !21
  %1306 = shl nsw i64 %1305, 19, !dbg !21
  %1307 = and i64 %1306, 16252928, !dbg !21
  %address.thread145 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local146 = zext nneg i32 %address.thread145 to i64, !dbg !13
  %address.local147 = add nsw i64 %16, %address.local146, !dbg !21
  %address.local148 = lshr i64 %address.local146, 6, !dbg !13
  %address.local149 = shl nuw nsw i64 %address.local148, 4, !dbg !21
  %address.local150 = add nsw i64 %address.local147, %address.local149, !dbg !21
  %address.local151 = and i64 %address.local146, 240, !dbg !21
  %address.local152 = sub nsw i64 %address.local150, %address.local151, !dbg !21
  %address.local153 = add nsw i64 %address.local152, %17, !dbg !21
  %address.local154 = shl nuw nsw i64 %address.local153, 7, !dbg !21
  %address.local155 = add nuw nsw i64 %1307, %address.local154, !dbg !21
  %address.local156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local155, !dbg !21
  %address.local157 = lshr i64 %address.local146, 2, !dbg !13
  %address.local158 = and i64 %address.local157, 12, !dbg !13
  %address.local159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local156, i64 %address.local158, !dbg !21
  store half %1304, ptr addrspace(1) %address.local159, align 2, !dbg !21
  %address.local160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local155, !dbg !21
  %address.local161 = or disjoint i64 %address.local158, 1, !dbg !21
  %address.local162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local160, i64 %address.local161, !dbg !21
  store half %1303, ptr addrspace(1) %address.local162, align 2, !dbg !21
  %address.local163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local155, !dbg !21
  %address.local164 = or disjoint i64 %address.local158, 2, !dbg !21
  %address.local165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local163, i64 %address.local164, !dbg !21
  store half %1302, ptr addrspace(1) %address.local165, align 2, !dbg !21
  %address.local166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local155, !dbg !21
  %address.local167 = or disjoint i64 %address.local158, 3, !dbg !21
  %address.local168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local166, i64 %address.local167, !dbg !21
  store half %1301, ptr addrspace(1) %address.local168, align 2, !dbg !21
  %address.local169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local155, !dbg !21
  %address.local170 = or disjoint i64 %address.local158, 16, !dbg !21
  %address.local171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local169, i64 %address.local170, !dbg !21
  store half %1289, ptr addrspace(1) %address.local171, align 2, !dbg !21
  %address.local172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local155, !dbg !21
  %address.local173 = or disjoint i64 %address.local158, 17, !dbg !21
  %address.local174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local172, i64 %address.local173, !dbg !21
  store half %1288, ptr addrspace(1) %address.local174, align 2, !dbg !21
  %address.local175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local155, !dbg !21
  %address.local176 = or disjoint i64 %address.local158, 18, !dbg !21
  %address.local177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local175, i64 %address.local176, !dbg !21
  store half %1287, ptr addrspace(1) %address.local177, align 2, !dbg !21
  %address.local178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local155, !dbg !21
  %address.local179 = or disjoint i64 %address.local158, 19, !dbg !21
  %address.local180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local178, i64 %address.local179, !dbg !21
  store half %1286, ptr addrspace(1) %address.local180, align 2, !dbg !21
  %address.local181 = shl nsw i64 %address.local153, 7, !dbg !21
  %address.local182 = add i64 %address.local181, 8192, !dbg !21
  %address.local183 = add nuw nsw i64 %1307, %address.local182, !dbg !21
  %address.local184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local183, !dbg !21
  %address.local185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local184, i64 %address.local158, !dbg !21
  store half %1274, ptr addrspace(1) %address.local185, align 2, !dbg !21
  %address.local186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local183, !dbg !21
  %address.local187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local186, i64 %address.local161, !dbg !21
  store half %1273, ptr addrspace(1) %address.local187, align 2, !dbg !21
  %address.local188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local183, !dbg !21
  %address.local189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local188, i64 %address.local164, !dbg !21
  store half %1272, ptr addrspace(1) %address.local189, align 2, !dbg !21
  %address.local190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local183, !dbg !21
  %address.local191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local190, i64 %address.local167, !dbg !21
  store half %1271, ptr addrspace(1) %address.local191, align 2, !dbg !21
  %address.local192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local183, !dbg !21
  %address.local193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local192, i64 %address.local170, !dbg !21
  store half %1259, ptr addrspace(1) %address.local193, align 2, !dbg !21
  %address.local194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local183, !dbg !21
  %address.local195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local194, i64 %address.local173, !dbg !21
  store half %1258, ptr addrspace(1) %address.local195, align 2, !dbg !21
  %address.local196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local183, !dbg !21
  %address.local197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local196, i64 %address.local176, !dbg !21
  store half %1257, ptr addrspace(1) %address.local197, align 2, !dbg !21
  %address.local198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local183, !dbg !21
  %address.local199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local198, i64 %address.local179, !dbg !21
  store half %1256, ptr addrspace(1) %address.local199, align 2, !dbg !21
  br label %1308, !dbg !21

1308:                                             ; preds = %1244, %795
  br i1 true, label %1309, label %1347, !dbg !21

1309:                                             ; preds = %1308
  %1310 = insertelement <4 x float> poison, float %1214, i64 0, !dbg !19
  %1311 = insertelement <4 x float> %1310, float %1220, i64 1, !dbg !19
  %1312 = insertelement <4 x float> %1311, float %1215, i64 2, !dbg !19
  %1313 = insertelement <4 x float> %1312, float %1221, i64 3, !dbg !19
  %row.reciprocal27 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1137, !dbg !19
  %normalized28 = fmul <4 x float> %1313, %row.reciprocal27, !dbg !19
  %1314 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !20
  %1315 = extractelement <4 x half> %1314, i64 3, !dbg !20
  %1316 = extractelement <4 x half> %1314, i64 2, !dbg !20
  %1317 = extractelement <4 x half> %1314, i64 1, !dbg !20
  %1318 = extractelement <4 x half> %1314, i64 0, !dbg !20
  %1319 = insertelement <4 x float> poison, float %1105, i64 0, !dbg !19
  %1320 = insertelement <4 x float> %1319, float %1111, i64 1, !dbg !19
  %1321 = insertelement <4 x float> %1320, float %1106, i64 2, !dbg !19
  %1322 = insertelement <4 x float> %1321, float %1112, i64 3, !dbg !19
  %normalized29 = fmul <4 x float> %1322, %row.reciprocal27, !dbg !19
  %1323 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !20
  %1324 = extractelement <4 x half> %1323, i64 3, !dbg !20
  %1325 = extractelement <4 x half> %1323, i64 2, !dbg !20
  %1326 = extractelement <4 x half> %1323, i64 1, !dbg !20
  %1327 = extractelement <4 x half> %1323, i64 0, !dbg !20
  %1328 = insertelement <4 x float> poison, float %996, i64 0, !dbg !19
  %1329 = insertelement <4 x float> %1328, float %1002, i64 1, !dbg !19
  %1330 = insertelement <4 x float> %1329, float %997, i64 2, !dbg !19
  %1331 = insertelement <4 x float> %1330, float %1003, i64 3, !dbg !19
  %row.reciprocal30 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %919, !dbg !19
  %normalized31 = fmul <4 x float> %1331, %row.reciprocal30, !dbg !19
  %1332 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !20
  %1333 = extractelement <4 x half> %1332, i64 3, !dbg !20
  %1334 = extractelement <4 x half> %1332, i64 2, !dbg !20
  %1335 = extractelement <4 x half> %1332, i64 1, !dbg !20
  %1336 = extractelement <4 x half> %1332, i64 0, !dbg !20
  %1337 = insertelement <4 x float> poison, float %887, i64 0, !dbg !19
  %1338 = insertelement <4 x float> %1337, float %893, i64 1, !dbg !19
  %1339 = insertelement <4 x float> %1338, float %888, i64 2, !dbg !19
  %1340 = insertelement <4 x float> %1339, float %894, i64 3, !dbg !19
  %normalized32 = fmul <4 x float> %1340, %row.reciprocal30, !dbg !19
  %1341 = fptrunc <4 x float> %normalized32 to <4 x half>, !dbg !20
  %1342 = extractelement <4 x half> %1341, i64 3, !dbg !20
  %1343 = extractelement <4 x half> %1341, i64 2, !dbg !20
  %1344 = extractelement <4 x half> %1341, i64 1, !dbg !20
  %1345 = extractelement <4 x half> %1341, i64 0, !dbg !20
  %1346 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread200 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local201 = zext nneg i32 %address.thread200 to i64, !dbg !13
  %address.local202 = add nsw i64 %16, %address.local201, !dbg !21
  %address.local203 = lshr i64 %address.local201, 6, !dbg !13
  %address.local204 = shl nuw nsw i64 %address.local203, 4, !dbg !21
  %address.local205 = add nsw i64 %address.local202, %address.local204, !dbg !21
  %address.local206 = and i64 %address.local201, 240, !dbg !21
  %address.local207 = sub nsw i64 %address.local205, %address.local206, !dbg !21
  %address.local208 = add nsw i64 %address.local207, %17, !dbg !21
  %address.local209 = shl nuw nsw i64 %address.local208, 7, !dbg !21
  %address.local210 = add nuw nsw i64 %1346, %address.local209, !dbg !21
  %address.local211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local210, !dbg !21
  %address.local212 = lshr i64 %address.local201, 2, !dbg !13
  %address.local213 = and i64 %address.local212, 12, !dbg !13
  %address.local214 = or disjoint i64 %address.local213, 32, !dbg !21
  %address.local215 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local211, i64 %address.local214, !dbg !21
  store half %1345, ptr addrspace(1) %address.local215, align 2, !dbg !21
  %address.local216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local210, !dbg !21
  %address.local217 = or disjoint i64 %address.local213, 33, !dbg !21
  %address.local218 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local216, i64 %address.local217, !dbg !21
  store half %1344, ptr addrspace(1) %address.local218, align 2, !dbg !21
  %address.local219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local210, !dbg !21
  %address.local220 = or disjoint i64 %address.local213, 34, !dbg !21
  %address.local221 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local219, i64 %address.local220, !dbg !21
  store half %1343, ptr addrspace(1) %address.local221, align 2, !dbg !21
  %address.local222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local210, !dbg !21
  %address.local223 = or disjoint i64 %address.local213, 35, !dbg !21
  %address.local224 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local222, i64 %address.local223, !dbg !21
  store half %1342, ptr addrspace(1) %address.local224, align 2, !dbg !21
  %address.local225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local210, !dbg !21
  %address.local226 = or disjoint i64 %address.local213, 48, !dbg !21
  %address.local227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local225, i64 %address.local226, !dbg !21
  store half %1336, ptr addrspace(1) %address.local227, align 2, !dbg !21
  %address.local228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local210, !dbg !21
  %address.local229 = or disjoint i64 %address.local213, 49, !dbg !21
  %address.local230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local228, i64 %address.local229, !dbg !21
  store half %1335, ptr addrspace(1) %address.local230, align 2, !dbg !21
  %address.local231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local210, !dbg !21
  %address.local232 = or disjoint i64 %address.local213, 50, !dbg !21
  %address.local233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local231, i64 %address.local232, !dbg !21
  store half %1334, ptr addrspace(1) %address.local233, align 2, !dbg !21
  %address.local234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local210, !dbg !21
  %address.local235 = or i64 %address.local212, 51, !dbg !21
  %address.local236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local234, i64 %address.local235, !dbg !21
  store half %1333, ptr addrspace(1) %address.local236, align 2, !dbg !21
  %address.local237 = shl nsw i64 %address.local208, 7, !dbg !21
  %address.local238 = add i64 %address.local237, 8192, !dbg !21
  %address.local239 = add nuw nsw i64 %1346, %address.local238, !dbg !21
  %address.local240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local239, !dbg !21
  %address.local241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local240, i64 %address.local214, !dbg !21
  store half %1327, ptr addrspace(1) %address.local241, align 2, !dbg !21
  %address.local242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local239, !dbg !21
  %address.local243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local242, i64 %address.local217, !dbg !21
  store half %1326, ptr addrspace(1) %address.local243, align 2, !dbg !21
  %address.local244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local239, !dbg !21
  %address.local245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local244, i64 %address.local220, !dbg !21
  store half %1325, ptr addrspace(1) %address.local245, align 2, !dbg !21
  %address.local246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local239, !dbg !21
  %address.local247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local246, i64 %address.local223, !dbg !21
  store half %1324, ptr addrspace(1) %address.local247, align 2, !dbg !21
  %address.local248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local239, !dbg !21
  %address.local249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local248, i64 %address.local226, !dbg !21
  store half %1318, ptr addrspace(1) %address.local249, align 2, !dbg !21
  %address.local250 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local239, !dbg !21
  %address.local251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local250, i64 %address.local229, !dbg !21
  store half %1317, ptr addrspace(1) %address.local251, align 2, !dbg !21
  %address.local252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local239, !dbg !21
  %address.local253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local252, i64 %address.local232, !dbg !21
  store half %1316, ptr addrspace(1) %address.local253, align 2, !dbg !21
  %address.local254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local239, !dbg !21
  %address.local255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local254, i64 %address.local235, !dbg !21
  store half %1315, ptr addrspace(1) %address.local255, align 2, !dbg !21
  br label %1347, !dbg !21

1347:                                             ; preds = %1309, %1308
  br i1 true, label %1348, label %1386, !dbg !21

1348:                                             ; preds = %1347
  %1349 = insertelement <4 x float> poison, float %1186, i64 0, !dbg !19
  %1350 = insertelement <4 x float> %1349, float %1192, i64 1, !dbg !19
  %1351 = insertelement <4 x float> %1350, float %1187, i64 2, !dbg !19
  %1352 = insertelement <4 x float> %1351, float %1193, i64 3, !dbg !19
  %row.reciprocal33 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1137, !dbg !19
  %normalized34 = fmul <4 x float> %1352, %row.reciprocal33, !dbg !19
  %1353 = fptrunc <4 x float> %normalized34 to <4 x half>, !dbg !20
  %1354 = extractelement <4 x half> %1353, i64 3, !dbg !20
  %1355 = extractelement <4 x half> %1353, i64 2, !dbg !20
  %1356 = extractelement <4 x half> %1353, i64 1, !dbg !20
  %1357 = extractelement <4 x half> %1353, i64 0, !dbg !20
  %1358 = insertelement <4 x float> poison, float %1076, i64 0, !dbg !19
  %1359 = insertelement <4 x float> %1358, float %1082, i64 1, !dbg !19
  %1360 = insertelement <4 x float> %1359, float %1077, i64 2, !dbg !19
  %1361 = insertelement <4 x float> %1360, float %1083, i64 3, !dbg !19
  %normalized35 = fmul <4 x float> %1361, %row.reciprocal33, !dbg !19
  %1362 = fptrunc <4 x float> %normalized35 to <4 x half>, !dbg !20
  %1363 = extractelement <4 x half> %1362, i64 3, !dbg !20
  %1364 = extractelement <4 x half> %1362, i64 2, !dbg !20
  %1365 = extractelement <4 x half> %1362, i64 1, !dbg !20
  %1366 = extractelement <4 x half> %1362, i64 0, !dbg !20
  %1367 = insertelement <4 x float> poison, float %968, i64 0, !dbg !19
  %1368 = insertelement <4 x float> %1367, float %974, i64 1, !dbg !19
  %1369 = insertelement <4 x float> %1368, float %969, i64 2, !dbg !19
  %1370 = insertelement <4 x float> %1369, float %975, i64 3, !dbg !19
  %row.reciprocal36 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %919, !dbg !19
  %normalized37 = fmul <4 x float> %1370, %row.reciprocal36, !dbg !19
  %1371 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !20
  %1372 = extractelement <4 x half> %1371, i64 3, !dbg !20
  %1373 = extractelement <4 x half> %1371, i64 2, !dbg !20
  %1374 = extractelement <4 x half> %1371, i64 1, !dbg !20
  %1375 = extractelement <4 x half> %1371, i64 0, !dbg !20
  %1376 = insertelement <4 x float> poison, float %858, i64 0, !dbg !19
  %1377 = insertelement <4 x float> %1376, float %864, i64 1, !dbg !19
  %1378 = insertelement <4 x float> %1377, float %859, i64 2, !dbg !19
  %1379 = insertelement <4 x float> %1378, float %865, i64 3, !dbg !19
  %normalized38 = fmul <4 x float> %1379, %row.reciprocal36, !dbg !19
  %1380 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !20
  %1381 = extractelement <4 x half> %1380, i64 3, !dbg !20
  %1382 = extractelement <4 x half> %1380, i64 2, !dbg !20
  %1383 = extractelement <4 x half> %1380, i64 1, !dbg !20
  %1384 = extractelement <4 x half> %1380, i64 0, !dbg !20
  %1385 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread256 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local257 = zext nneg i32 %address.thread256 to i64, !dbg !13
  %address.local258 = add nsw i64 %16, %address.local257, !dbg !21
  %address.local259 = lshr i64 %address.local257, 6, !dbg !13
  %address.local260 = shl nuw nsw i64 %address.local259, 4, !dbg !21
  %address.local261 = add nsw i64 %address.local258, %address.local260, !dbg !21
  %address.local262 = and i64 %address.local257, 240, !dbg !21
  %address.local263 = sub nsw i64 %address.local261, %address.local262, !dbg !21
  %address.local264 = add nsw i64 %address.local263, %17, !dbg !21
  %address.local265 = shl nuw nsw i64 %address.local264, 7, !dbg !21
  %address.local266 = add nuw nsw i64 %1385, %address.local265, !dbg !21
  %address.local267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local268 = lshr i64 %address.local257, 2, !dbg !13
  %address.local269 = and i64 %address.local268, 12, !dbg !13
  %address.local270 = or disjoint i64 %address.local269, 64, !dbg !21
  %address.local271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local267, i64 %address.local270, !dbg !21
  store half %1384, ptr addrspace(1) %address.local271, align 2, !dbg !21
  %address.local272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local273 = or disjoint i64 %address.local269, 65, !dbg !21
  %address.local274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local272, i64 %address.local273, !dbg !21
  store half %1383, ptr addrspace(1) %address.local274, align 2, !dbg !21
  %address.local275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local276 = or disjoint i64 %address.local269, 66, !dbg !21
  %address.local277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local275, i64 %address.local276, !dbg !21
  store half %1382, ptr addrspace(1) %address.local277, align 2, !dbg !21
  %address.local278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local279 = or disjoint i64 %address.local269, 67, !dbg !21
  %address.local280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local278, i64 %address.local279, !dbg !21
  store half %1381, ptr addrspace(1) %address.local280, align 2, !dbg !21
  %address.local281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local282 = or disjoint i64 %address.local269, 80, !dbg !21
  %address.local283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local281, i64 %address.local282, !dbg !21
  store half %1375, ptr addrspace(1) %address.local283, align 2, !dbg !21
  %address.local284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local285 = or disjoint i64 %address.local269, 81, !dbg !21
  %address.local286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local284, i64 %address.local285, !dbg !21
  store half %1374, ptr addrspace(1) %address.local286, align 2, !dbg !21
  %address.local287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local288 = or disjoint i64 %address.local269, 82, !dbg !21
  %address.local289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local287, i64 %address.local288, !dbg !21
  store half %1373, ptr addrspace(1) %address.local289, align 2, !dbg !21
  %address.local290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local291 = or disjoint i64 %address.local269, 83, !dbg !21
  %address.local292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local290, i64 %address.local291, !dbg !21
  store half %1372, ptr addrspace(1) %address.local292, align 2, !dbg !21
  %address.local293 = shl nsw i64 %address.local264, 7, !dbg !21
  %address.local294 = add i64 %address.local293, 8192, !dbg !21
  %address.local295 = add nuw nsw i64 %1385, %address.local294, !dbg !21
  %address.local296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !21
  %address.local297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local296, i64 %address.local270, !dbg !21
  store half %1366, ptr addrspace(1) %address.local297, align 2, !dbg !21
  %address.local298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !21
  %address.local299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local298, i64 %address.local273, !dbg !21
  store half %1365, ptr addrspace(1) %address.local299, align 2, !dbg !21
  %address.local300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !21
  %address.local301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local300, i64 %address.local276, !dbg !21
  store half %1364, ptr addrspace(1) %address.local301, align 2, !dbg !21
  %address.local302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !21
  %address.local303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local302, i64 %address.local279, !dbg !21
  store half %1363, ptr addrspace(1) %address.local303, align 2, !dbg !21
  %address.local304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !21
  %address.local305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local304, i64 %address.local282, !dbg !21
  store half %1357, ptr addrspace(1) %address.local305, align 2, !dbg !21
  %address.local306 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !21
  %address.local307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local306, i64 %address.local285, !dbg !21
  store half %1356, ptr addrspace(1) %address.local307, align 2, !dbg !21
  %address.local308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !21
  %address.local309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local308, i64 %address.local288, !dbg !21
  store half %1355, ptr addrspace(1) %address.local309, align 2, !dbg !21
  %address.local310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !21
  %address.local311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local310, i64 %address.local291, !dbg !21
  store half %1354, ptr addrspace(1) %address.local311, align 2, !dbg !21
  br label %1386, !dbg !21

1386:                                             ; preds = %1348, %1347
  br i1 true, label %1387, label %1425, !dbg !21

1387:                                             ; preds = %1386
  %1388 = insertelement <4 x float> poison, float %1158, i64 0, !dbg !19
  %1389 = insertelement <4 x float> %1388, float %1164, i64 1, !dbg !19
  %1390 = insertelement <4 x float> %1389, float %1159, i64 2, !dbg !19
  %1391 = insertelement <4 x float> %1390, float %1165, i64 3, !dbg !19
  %row.reciprocal39 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1137, !dbg !19
  %normalized40 = fmul <4 x float> %1391, %row.reciprocal39, !dbg !19
  %1392 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !20
  %1393 = extractelement <4 x half> %1392, i64 3, !dbg !20
  %1394 = extractelement <4 x half> %1392, i64 2, !dbg !20
  %1395 = extractelement <4 x half> %1392, i64 1, !dbg !20
  %1396 = extractelement <4 x half> %1392, i64 0, !dbg !20
  %1397 = insertelement <4 x float> poison, float %1047, i64 0, !dbg !19
  %1398 = insertelement <4 x float> %1397, float %1053, i64 1, !dbg !19
  %1399 = insertelement <4 x float> %1398, float %1048, i64 2, !dbg !19
  %1400 = insertelement <4 x float> %1399, float %1054, i64 3, !dbg !19
  %normalized41 = fmul <4 x float> %1400, %row.reciprocal39, !dbg !19
  %1401 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !20
  %1402 = extractelement <4 x half> %1401, i64 3, !dbg !20
  %1403 = extractelement <4 x half> %1401, i64 2, !dbg !20
  %1404 = extractelement <4 x half> %1401, i64 1, !dbg !20
  %1405 = extractelement <4 x half> %1401, i64 0, !dbg !20
  %1406 = insertelement <4 x float> poison, float %940, i64 0, !dbg !19
  %1407 = insertelement <4 x float> %1406, float %946, i64 1, !dbg !19
  %1408 = insertelement <4 x float> %1407, float %941, i64 2, !dbg !19
  %1409 = insertelement <4 x float> %1408, float %947, i64 3, !dbg !19
  %row.reciprocal42 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %919, !dbg !19
  %normalized43 = fmul <4 x float> %1409, %row.reciprocal42, !dbg !19
  %1410 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !20
  %1411 = extractelement <4 x half> %1410, i64 3, !dbg !20
  %1412 = extractelement <4 x half> %1410, i64 2, !dbg !20
  %1413 = extractelement <4 x half> %1410, i64 1, !dbg !20
  %1414 = extractelement <4 x half> %1410, i64 0, !dbg !20
  %1415 = insertelement <4 x float> poison, float %829, i64 0, !dbg !19
  %1416 = insertelement <4 x float> %1415, float %835, i64 1, !dbg !19
  %1417 = insertelement <4 x float> %1416, float %830, i64 2, !dbg !19
  %1418 = insertelement <4 x float> %1417, float %836, i64 3, !dbg !19
  %normalized44 = fmul <4 x float> %1418, %row.reciprocal42, !dbg !19
  %1419 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !20
  %1420 = extractelement <4 x half> %1419, i64 3, !dbg !20
  %1421 = extractelement <4 x half> %1419, i64 2, !dbg !20
  %1422 = extractelement <4 x half> %1419, i64 1, !dbg !20
  %1423 = extractelement <4 x half> %1419, i64 0, !dbg !20
  %1424 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread312 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local313 = zext nneg i32 %address.thread312 to i64, !dbg !13
  %address.local314 = add nsw i64 %16, %address.local313, !dbg !21
  %address.local315 = lshr i64 %address.local313, 6, !dbg !13
  %address.local316 = shl nuw nsw i64 %address.local315, 4, !dbg !21
  %address.local317 = add nsw i64 %address.local314, %address.local316, !dbg !21
  %address.local318 = and i64 %address.local313, 240, !dbg !21
  %address.local319 = sub nsw i64 %address.local317, %address.local318, !dbg !21
  %address.local320 = add nsw i64 %address.local319, %17, !dbg !21
  %address.local321 = shl nuw nsw i64 %address.local320, 7, !dbg !21
  %address.local322 = add nuw nsw i64 %1424, %address.local321, !dbg !21
  %address.local323 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local322, !dbg !21
  %address.local324 = lshr i64 %address.local313, 2, !dbg !13
  %address.local325 = and i64 %address.local324, 12, !dbg !13
  %address.local326 = or disjoint i64 %address.local325, 96, !dbg !21
  %address.local327 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local323, i64 %address.local326, !dbg !21
  store half %1423, ptr addrspace(1) %address.local327, align 2, !dbg !21
  %address.local328 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local322, !dbg !21
  %address.local329 = or disjoint i64 %address.local325, 97, !dbg !21
  %address.local330 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local328, i64 %address.local329, !dbg !21
  store half %1422, ptr addrspace(1) %address.local330, align 2, !dbg !21
  %address.local331 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local322, !dbg !21
  %address.local332 = or disjoint i64 %address.local325, 98, !dbg !21
  %address.local333 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local331, i64 %address.local332, !dbg !21
  store half %1421, ptr addrspace(1) %address.local333, align 2, !dbg !21
  %address.local334 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local322, !dbg !21
  %address.local335 = or disjoint i64 %address.local325, 99, !dbg !21
  %address.local336 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local334, i64 %address.local335, !dbg !21
  store half %1420, ptr addrspace(1) %address.local336, align 2, !dbg !21
  %address.local337 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local322, !dbg !21
  %address.local338 = or disjoint i64 %address.local325, 112, !dbg !21
  %address.local339 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local337, i64 %address.local338, !dbg !21
  store half %1414, ptr addrspace(1) %address.local339, align 2, !dbg !21
  %address.local340 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local322, !dbg !21
  %address.local341 = or disjoint i64 %address.local325, 113, !dbg !21
  %address.local342 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local340, i64 %address.local341, !dbg !21
  store half %1413, ptr addrspace(1) %address.local342, align 2, !dbg !21
  %address.local343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local322, !dbg !21
  %address.local344 = or disjoint i64 %address.local325, 114, !dbg !21
  %address.local345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local343, i64 %address.local344, !dbg !21
  store half %1412, ptr addrspace(1) %address.local345, align 2, !dbg !21
  %address.local346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local322, !dbg !21
  %address.local347 = or i64 %address.local324, 115, !dbg !21
  %address.local348 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local346, i64 %address.local347, !dbg !21
  store half %1411, ptr addrspace(1) %address.local348, align 2, !dbg !21
  %address.local349 = shl nsw i64 %address.local320, 7, !dbg !21
  %address.local350 = add i64 %address.local349, 8192, !dbg !21
  %address.local351 = add nuw nsw i64 %1424, %address.local350, !dbg !21
  %address.local352 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local351, !dbg !21
  %address.local353 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local352, i64 %address.local326, !dbg !21
  store half %1405, ptr addrspace(1) %address.local353, align 2, !dbg !21
  %address.local354 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local351, !dbg !21
  %address.local355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local354, i64 %address.local329, !dbg !21
  store half %1404, ptr addrspace(1) %address.local355, align 2, !dbg !21
  %address.local356 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local351, !dbg !21
  %address.local357 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local356, i64 %address.local332, !dbg !21
  store half %1403, ptr addrspace(1) %address.local357, align 2, !dbg !21
  %address.local358 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local351, !dbg !21
  %address.local359 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local358, i64 %address.local335, !dbg !21
  store half %1402, ptr addrspace(1) %address.local359, align 2, !dbg !21
  %address.local360 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local351, !dbg !21
  %address.local361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local360, i64 %address.local338, !dbg !21
  store half %1396, ptr addrspace(1) %address.local361, align 2, !dbg !21
  %address.local362 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local351, !dbg !21
  %address.local363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local362, i64 %address.local341, !dbg !21
  store half %1395, ptr addrspace(1) %address.local363, align 2, !dbg !21
  %address.local364 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local351, !dbg !21
  %address.local365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local364, i64 %address.local344, !dbg !21
  store half %1394, ptr addrspace(1) %address.local365, align 2, !dbg !21
  %address.local366 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local351, !dbg !21
  %address.local367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local366, i64 %address.local347, !dbg !21
  store half %1393, ptr addrspace(1) %address.local367, align 2, !dbg !21
  br label %1425, !dbg !21

1425:                                             ; preds = %1387, %1386
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #6

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nofree nounwind willreturn }
attributes #7 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/nohint-rolled")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 73, column: 26, scope: !4)
!8 = !DILocation(line: 60, column: 24, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!12 = !DILocation(line: 70, column: 22, scope: !4)
!13 = !DILocation(line: 72, column: 24, scope: !4)
!14 = !DILocation(line: 63, column: 17, scope: !4)
!15 = !DILocation(line: 77, column: 19, scope: !4)
!16 = !DILocation(line: 102, column: 23, scope: !4)
!17 = !DILocation(line: 71, column: 22, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
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
!36 = !DILocation(line: 74, column: 23, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
