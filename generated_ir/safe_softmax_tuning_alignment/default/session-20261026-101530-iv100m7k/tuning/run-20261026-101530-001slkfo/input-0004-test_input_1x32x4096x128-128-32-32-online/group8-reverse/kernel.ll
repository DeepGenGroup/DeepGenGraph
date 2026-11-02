; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/group8-reverse/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/group8-reverse/input.mlir"

@shm_0 = addrspace(3) global [10240 x i8] undef, align 16

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
  %12 = and i64 %11, 576460752303423480, !dbg !5
  %13 = and i64 %6, 7, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = lshr i64 %10, 3, !dbg !5
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
  %37 = lshr i64 %35, 2, !dbg !12
  %38 = and i64 %37, 12, !dbg !12
  %39 = and i32 %34, 16, !dbg !13
  %.not = icmp eq i32 %39, 0, !dbg !13
  %40 = and i32 %34, 32, !dbg !13
  %.not1 = icmp eq i32 %40, 0, !dbg !13
  %41 = or disjoint i64 %19, 127, !dbg !14
  %42 = add i64 %19, 128, !dbg !14
  %43 = icmp uge i64 %42, %41, !dbg !14
  %44 = shl nsw i64 %16, 2, !dbg !7
  %45 = add nsw i64 %44, 4, !dbg !7
  %46 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %47 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %48 = getelementptr i8, ptr addrspace(1) %47, i64 64, !dbg !11
  %49 = shl i64 %15, 7, !dbg !17
  %50 = and i64 %49, 3968, !dbg !17
  %51 = shl nsw i64 %18, 12, !dbg !17
  %52 = or disjoint i64 %50, %51, !dbg !17
  %53 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %54 = getelementptr i8, ptr addrspace(1) %53, i64 128, !dbg !11
  %55 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %56 = getelementptr i8, ptr addrspace(1) %55, i64 192, !dbg !11
  br label %57, !dbg !7

57:                                               ; preds = %1040, %4
  %58 = phi i64 [ %1111, %1040 ], [ 0, %4 ], !dbg !7
  %59 = phi [2 x <8 x float>] [ %1092, %1040 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi [2 x <1 x float>] [ %1096, %1040 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = phi [2 x <1 x float>] [ %1098, %1040 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %62 = phi [2 x <8 x float>] [ %1102, %1040 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi [2 x <8 x float>] [ %1106, %1040 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi [2 x <8 x float>] [ %1110, %1040 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = icmp slt i64 %58, %45, !dbg !7
  br i1 %65, label %66, label %1112, !dbg !7

66:                                               ; preds = %57
  %67 = shl nsw i64 %58, 5, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local52 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local53 = and i64 %address.local52, 1536, !dbg !11
  %address.local54 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local53, !dbg !11
  %address.local55 = ptrtoint ptr addrspace(3) %address.local54 to i64, !dbg !11
  %address.local56 = trunc i64 %address.local55 to i32, !dbg !11
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local56), !dbg !11
  %69 = zext i32 %68 to i64, !dbg !11
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !11
  %address.local57 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local58 = and i64 %address.local57, 16256, !dbg !11
  %address.local59 = lshr i64 %address.local, 4, !dbg !11
  %address.local60 = and i64 %address.local59, 8, !dbg !11
  %address.local61 = or disjoint i64 %address.local58, %address.local60, !dbg !11
  %address.local62 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local61, !dbg !11
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %72 = ptrtoint ptr addrspace(1) %address.local62 to i64, !dbg !11
  %73 = sub i64 %72, %71, !dbg !11
  %dma.buffer.offset661 = trunc i64 %73 to i32, !dbg !11
  %74 = trunc i64 %71 to i32, !dbg !11
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !11
  %76 = lshr i64 %71, 32, !dbg !11
  %77 = trunc i64 %76 to i32, !dbg !11
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !11
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc662 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !11
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !11
  %dma.buffer.lds663 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !11
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds663, i32 %dma.buffer.offset661, <4 x i32> %dma.buffer.rsrc662) #10, !dbg !11
  %address.local63 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local64 = shl nuw nsw i64 %address.local63, 3, !dbg !11
  %address.local65 = and i64 %address.local64, 3584, !dbg !11
  %address.local66 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local65, !dbg !11
  %address.local67 = ptrtoint ptr addrspace(3) %address.local66 to i64, !dbg !11
  %address.local68 = trunc i64 %address.local67 to i32, !dbg !11
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local68), !dbg !11
  %83 = zext i32 %82 to i64, !dbg !11
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !11
  %address.local69 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local70 = and i64 %address.local69, 16256, !dbg !11
  %address.local71 = lshr i64 %address.local63, 4, !dbg !11
  %address.local72 = and i64 %address.local71, 24, !dbg !11
  %address.local73 = or disjoint i64 %address.local70, %address.local72, !dbg !11
  %address.local74 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local73, !dbg !11
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %86 = ptrtoint ptr addrspace(1) %address.local74 to i64, !dbg !11
  %87 = sub i64 %86, %85, !dbg !11
  %dma.buffer.offset658 = trunc i64 %87 to i32, !dbg !11
  %88 = trunc i64 %85 to i32, !dbg !11
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !11
  %90 = lshr i64 %85, 32, !dbg !11
  %91 = trunc i64 %90 to i32, !dbg !11
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !11
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc659 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !11
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !11
  %dma.buffer.lds660 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !11
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds660, i32 %dma.buffer.offset658, <4 x i32> %dma.buffer.rsrc659) #10, !dbg !11
  %96 = lshr i64 %58, 14, !dbg !17
  %97 = add nuw nsw i64 %96, %15, !dbg !17
  %98 = lshr i64 %58, 7, !dbg !17
  %99 = and i64 %98, 127, !dbg !17
  %100 = shl i64 %97, 19, !dbg !17
  %101 = and i64 %100, 16252928, !dbg !17
  %102 = shl i64 %98, 19, !dbg !17
  %103 = shl i64 %58, 12, !dbg !17
  %104 = sub i64 %103, %102, !dbg !17
  %105 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %101, !dbg !17
  %106 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %99, !dbg !17
  %107 = getelementptr [2 x i8], ptr addrspace(1) %106, i64 %104, !dbg !17
  br label %108, !dbg !17

108:                                              ; preds = %111, %66
  %109 = phi i64 [ %137, %111 ], [ %35, %66 ], !dbg !17
  %110 = icmp samesign ult i64 %109, 128, !dbg !17
  br i1 %110, label %111, label %138, !dbg !17

111:                                              ; preds = %108
  %112 = lshr i64 %109, 2, !dbg !17
  %113 = and i64 %112, 24, !dbg !17
  %114 = shl nuw nsw i64 %109, 3, !dbg !17
  %115 = and i64 %114, 512, !dbg !17
  %116 = shl nuw nsw i64 %109, 7, !dbg !17
  %117 = and i64 %116, 3968, !dbg !17
  %118 = getelementptr [2 x i8], ptr addrspace(1) %107, i64 %113, !dbg !17
  %119 = getelementptr [2 x i8], ptr addrspace(1) %118, i64 %117, !dbg !17
  %120 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %115, !dbg !17
  %121 = ptrtoint ptr addrspace(3) %120 to i64, !dbg !17
  %122 = trunc i64 %121 to i32, !dbg !17
  %123 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %122), !dbg !17
  %124 = zext i32 %123 to i64, !dbg !17
  %125 = inttoptr i64 %124 to ptr addrspace(3), !dbg !17
  %126 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %127 = ptrtoint ptr addrspace(1) %119 to i64, !dbg !17
  %128 = sub i64 %127, %126, !dbg !17
  %dma.buffer.offset = trunc i64 %128 to i32, !dbg !17
  %129 = trunc i64 %126 to i32, !dbg !17
  %130 = insertelement <4 x i32> poison, i32 %129, i64 0, !dbg !17
  %131 = lshr i64 %126, 32, !dbg !17
  %132 = trunc i64 %131 to i32, !dbg !17
  %133 = insertelement <4 x i32> %130, i32 %132, i64 1, !dbg !17
  %134 = insertelement <4 x i32> %133, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %134, i32 131072, i64 3, !dbg !17
  %135 = ptrtoint ptr addrspace(3) %125 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %135), !dbg !17
  %136 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %137 = or disjoint i64 %35, 256, !dbg !17
  br label %108, !dbg !17

138:                                              ; preds = %108
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.thread75 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local76 = lshr i32 %address.thread75, 2, !dbg !12
  %address.local77 = and i32 %address.local76, 4, !dbg !12
  %address.local78 = shl nuw nsw i32 %address.thread75, 1, !dbg !12
  %address.local79 = and i32 %address.local78, 384, !dbg !12
  %address.local80 = or disjoint i32 %address.local77, %address.local79, !dbg !12
  %address.local81 = shl nuw nsw i32 %address.thread75, 3, !dbg !12
  %address.local82 = and i32 %address.local81, 120, !dbg !12
  %address.local83 = or disjoint i32 %address.local80, %address.local82, !dbg !12
  %address.local84 = shl nuw nsw i32 %address.thread75, 5, !dbg !12
  %address.local85 = and i32 %address.local84, 1024, !dbg !12
  %address.local86 = or disjoint i32 %address.local83, %address.local85, !dbg !12
  %address.local87 = zext nneg i32 %address.local86 to i64, !dbg !12
  %address.local88 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local87, !dbg !12
  %139 = load <4 x half>, ptr addrspace(3) %address.local88, align 2, !dbg !12
  %address.local89 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local87, !dbg !12
  %address.local90 = getelementptr i8, ptr addrspace(3) %address.local89, i64 4096, !dbg !12
  %140 = load <4 x half>, ptr addrspace(3) %address.local90, align 2, !dbg !12
  %address.local91 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local87, !dbg !12
  %address.local92 = getelementptr i8, ptr addrspace(3) %address.local91, i64 1024, !dbg !12
  %141 = load <4 x half>, ptr addrspace(3) %address.local92, align 2, !dbg !12
  %142 = fmul <4 x half> %139, splat (half 0xH3015), !dbg !12
  %address.local93 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local87, !dbg !12
  %address.local94 = getelementptr i8, ptr addrspace(3) %address.local93, i64 5120, !dbg !12
  %143 = load <4 x half>, ptr addrspace(3) %address.local94, align 2, !dbg !12
  %144 = fmul <4 x half> %140, splat (half 0xH3015), !dbg !12
  %145 = fmul <4 x half> %141, splat (half 0xH3015), !dbg !12
  %146 = fmul <4 x half> %143, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local95 = lshr i32 %address.thread75, 2, !dbg !16
  %address.local96 = and i32 %address.local95, 4, !dbg !16
  %address.local97 = shl nuw nsw i32 %address.thread75, 3, !dbg !16
  %address.local98 = and i32 %address.local97, 376, !dbg !16
  %address.local99 = or disjoint i32 %address.local96, %address.local98, !dbg !16
  %address.local100 = zext nneg i32 %address.local99 to i64, !dbg !16
  %address.local101 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local100, !dbg !16
  %147 = load <4 x half>, ptr addrspace(3) %address.local101, align 2, !dbg !16
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local102 = lshr i32 %address.thread75, 2, !dbg !16
  %address.local103 = and i32 %address.local102, 4, !dbg !16
  %address.local104 = shl nuw nsw i32 %address.thread75, 3, !dbg !16
  %address.local105 = and i32 %address.local104, 376, !dbg !16
  %address.local106 = or disjoint i32 %address.local103, %address.local105, !dbg !16
  %address.local107 = zext nneg i32 %address.local106 to i64, !dbg !16
  %address.local108 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local107, !dbg !16
  %address.local109 = getelementptr i8, ptr addrspace(3) %address.local108, i64 1024, !dbg !16
  %149 = load <4 x half>, ptr addrspace(3) %address.local109, align 2, !dbg !16
  %address.local110 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local100, !dbg !16
  %address.local111 = getelementptr i8, ptr addrspace(3) %address.local110, i64 256, !dbg !16
  %150 = load <4 x half>, ptr addrspace(3) %address.local111, align 2, !dbg !16
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %149, <4 x float> %148) #9, !dbg !16
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local112 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local107, !dbg !16
  %address.local113 = getelementptr i8, ptr addrspace(3) %address.local112, i64 1280, !dbg !16
  %153 = load <4 x half>, ptr addrspace(3) %address.local113, align 2, !dbg !16
  %154 = load <4 x half>, ptr addrspace(3) %address.local101, align 2, !dbg !16
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %153, <4 x float> %152) #9, !dbg !16
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !16
  %157 = load <4 x half>, ptr addrspace(3) %address.local109, align 2, !dbg !16
  %158 = load <4 x half>, ptr addrspace(3) %address.local111, align 2, !dbg !16
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %157, <4 x float> %156) #9, !dbg !16
  %160 = load <4 x half>, ptr addrspace(3) %address.local113, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %160, <4 x float> %161) #9, !dbg !16
  %address.local114 = zext nneg i32 %address.thread75 to i64, !dbg !12
  %address.local115 = shl nuw nsw i64 %address.local114, 3, !dbg !11
  %address.local116 = and i64 %address.local115, 1536, !dbg !11
  %address.local117 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local116, !dbg !11
  %address.local118 = ptrtoint ptr addrspace(3) %address.local117 to i64, !dbg !11
  %address.local119 = trunc i64 %address.local118 to i32, !dbg !11
  %163 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local119), !dbg !11
  %164 = zext i32 %163 to i64, !dbg !11
  %165 = inttoptr i64 %164 to ptr addrspace(3), !dbg !11
  %address.local120 = shl nuw nsw i64 %address.local114, 7, !dbg !11
  %address.local121 = and i64 %address.local120, 16256, !dbg !11
  %address.local122 = lshr i64 %address.local114, 4, !dbg !11
  %address.local123 = and i64 %address.local122, 8, !dbg !11
  %address.local124 = or disjoint i64 %address.local121, %address.local123, !dbg !11
  %address.local125 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local124, !dbg !11
  %166 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %167 = ptrtoint ptr addrspace(1) %address.local125 to i64, !dbg !11
  %168 = sub i64 %167, %166, !dbg !11
  %dma.buffer.offset655 = trunc i64 %168 to i32, !dbg !11
  %169 = trunc i64 %166 to i32, !dbg !11
  %170 = insertelement <4 x i32> poison, i32 %169, i64 0, !dbg !11
  %171 = lshr i64 %166, 32, !dbg !11
  %172 = trunc i64 %171 to i32, !dbg !11
  %173 = insertelement <4 x i32> %170, i32 %172, i64 1, !dbg !11
  %174 = insertelement <4 x i32> %173, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc656 = insertelement <4 x i32> %174, i32 131072, i64 3, !dbg !11
  %175 = ptrtoint ptr addrspace(3) %165 to i32, !dbg !11
  %dma.buffer.lds657 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %175), !dbg !11
  %176 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds657, i32 %dma.buffer.offset655, <4 x i32> %dma.buffer.rsrc656) #10, !dbg !11
  %address.local126 = or disjoint i64 %address.local114, 256, !dbg !11
  %address.local127 = shl nuw nsw i64 %address.local126, 3, !dbg !11
  %address.local128 = and i64 %address.local127, 3584, !dbg !11
  %address.local129 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local128, !dbg !11
  %address.local130 = ptrtoint ptr addrspace(3) %address.local129 to i64, !dbg !11
  %address.local131 = trunc i64 %address.local130 to i32, !dbg !11
  %177 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local131), !dbg !11
  %178 = zext i32 %177 to i64, !dbg !11
  %179 = inttoptr i64 %178 to ptr addrspace(3), !dbg !11
  %address.local132 = shl nuw nsw i64 %address.local114, 7, !dbg !11
  %address.local133 = and i64 %address.local132, 16256, !dbg !11
  %address.local134 = lshr i64 %address.local126, 4, !dbg !11
  %address.local135 = and i64 %address.local134, 24, !dbg !11
  %address.local136 = or disjoint i64 %address.local133, %address.local135, !dbg !11
  %address.local137 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local136, !dbg !11
  %180 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %181 = ptrtoint ptr addrspace(1) %address.local137 to i64, !dbg !11
  %182 = sub i64 %181, %180, !dbg !11
  %dma.buffer.offset652 = trunc i64 %182 to i32, !dbg !11
  %183 = trunc i64 %180 to i32, !dbg !11
  %184 = insertelement <4 x i32> poison, i32 %183, i64 0, !dbg !11
  %185 = lshr i64 %180, 32, !dbg !11
  %186 = trunc i64 %185 to i32, !dbg !11
  %187 = insertelement <4 x i32> %184, i32 %186, i64 1, !dbg !11
  %188 = insertelement <4 x i32> %187, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc653 = insertelement <4 x i32> %188, i32 131072, i64 3, !dbg !11
  %189 = ptrtoint ptr addrspace(3) %179 to i32, !dbg !11
  %dma.buffer.lds654 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %189), !dbg !11
  %190 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds654, i32 %dma.buffer.offset652, <4 x i32> %dma.buffer.rsrc653) #10, !dbg !11
  %191 = add nuw i64 %58, 4096, !dbg !17
  %192 = icmp slt i64 %191, 0, !dbg !17
  %193 = sub nuw i64 -4097, %58, !dbg !17
  %194 = select i1 %192, i64 %193, i64 %191, !dbg !17
  %195 = sdiv i64 %194, 16384, !dbg !17
  %.lobit = ashr i64 %191, 63, !dbg !17
  %196 = xor i64 %195, %.lobit, !dbg !17
  %197 = add nsw i64 %196, %15, !dbg !17
  %198 = add nsw i64 %98, %52, !dbg !17
  %199 = add nsw i64 %198, 32, !dbg !17
  %200 = icmp slt i64 %198, -32, !dbg !17
  %201 = sub nsw i64 -33, %198, !dbg !17
  %202 = select i1 %200, i64 %201, i64 %199, !dbg !17
  %203 = sdiv i64 %202, 128, !dbg !17
  %204 = sext i1 %200 to i64, !dbg !17
  %205 = xor i64 %203, %204, !dbg !17
  %206 = shl nsw i64 %205, 7, !dbg !17
  %207 = sub i64 %52, %206, !dbg !17
  %208 = shl i64 %197, 19, !dbg !17
  %209 = and i64 %208, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %210 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %209, !dbg !17
  %211 = getelementptr [2 x i8], ptr addrspace(1) %210, i64 %98, !dbg !17
  %212 = getelementptr [2 x i8], ptr addrspace(1) %211, i64 %207, !dbg !17
  %213 = getelementptr i8, ptr addrspace(1) %212, i64 64, !dbg !17
  %214 = getelementptr [2 x i8], ptr addrspace(1) %213, i64 %104, !dbg !17
  br label %215, !dbg !17

215:                                              ; preds = %218, %138
  %216 = phi i64 [ %244, %218 ], [ %35, %138 ], !dbg !17
  %217 = icmp samesign ult i64 %216, 128, !dbg !17
  br i1 %217, label %218, label %245, !dbg !17

218:                                              ; preds = %215
  %219 = lshr i64 %216, 2, !dbg !17
  %220 = and i64 %219, 24, !dbg !17
  %221 = shl nuw nsw i64 %216, 3, !dbg !17
  %222 = and i64 %221, 512, !dbg !17
  %223 = shl nuw nsw i64 %216, 7, !dbg !17
  %224 = and i64 %223, 3968, !dbg !17
  %225 = getelementptr [2 x i8], ptr addrspace(1) %214, i64 %220, !dbg !17
  %226 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %224, !dbg !17
  %227 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %222, !dbg !17
  %228 = ptrtoint ptr addrspace(3) %227 to i64, !dbg !17
  %229 = trunc i64 %228 to i32, !dbg !17
  %230 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %229), !dbg !17
  %231 = zext i32 %230 to i64, !dbg !17
  %232 = inttoptr i64 %231 to ptr addrspace(3), !dbg !17
  %233 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %234 = ptrtoint ptr addrspace(1) %226 to i64, !dbg !17
  %235 = sub i64 %234, %233, !dbg !17
  %dma.buffer.offset619 = trunc i64 %235 to i32, !dbg !17
  %236 = trunc i64 %233 to i32, !dbg !17
  %237 = insertelement <4 x i32> poison, i32 %236, i64 0, !dbg !17
  %238 = lshr i64 %233, 32, !dbg !17
  %239 = trunc i64 %238 to i32, !dbg !17
  %240 = insertelement <4 x i32> %237, i32 %239, i64 1, !dbg !17
  %241 = insertelement <4 x i32> %240, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc620 = insertelement <4 x i32> %241, i32 131072, i64 3, !dbg !17
  %242 = ptrtoint ptr addrspace(3) %232 to i32, !dbg !17
  %dma.buffer.lds621 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %242), !dbg !17
  %243 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds621, i32 %dma.buffer.offset619, <4 x i32> %dma.buffer.rsrc620) #10, !dbg !17
  %244 = or disjoint i64 %35, 256, !dbg !17
  br label %215, !dbg !17

245:                                              ; preds = %215
  %address.thread138 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local139 = lshr i32 %address.thread138, 2, !dbg !12
  %address.local140 = and i32 %address.local139, 4, !dbg !12
  %address.local141 = shl nuw nsw i32 %address.thread138, 1, !dbg !12
  %address.local142 = and i32 %address.local141, 384, !dbg !12
  %address.local143 = or disjoint i32 %address.local140, %address.local142, !dbg !12
  %address.local144 = shl nuw nsw i32 %address.thread138, 3, !dbg !12
  %address.local145 = and i32 %address.local144, 120, !dbg !12
  %address.local146 = or disjoint i32 %address.local143, %address.local145, !dbg !12
  %address.local147 = shl nuw nsw i32 %address.thread138, 5, !dbg !12
  %address.local148 = and i32 %address.local147, 1024, !dbg !12
  %address.local149 = or disjoint i32 %address.local146, %address.local148, !dbg !12
  %address.local150 = zext nneg i32 %address.local149 to i64, !dbg !12
  %address.local151 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local150, !dbg !12
  %246 = load <4 x half>, ptr addrspace(3) %address.local151, align 2, !dbg !12
  %address.local152 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local150, !dbg !12
  %address.local153 = getelementptr i8, ptr addrspace(3) %address.local152, i64 4096, !dbg !12
  %247 = load <4 x half>, ptr addrspace(3) %address.local153, align 2, !dbg !12
  %address.local154 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local150, !dbg !12
  %address.local155 = getelementptr i8, ptr addrspace(3) %address.local154, i64 1024, !dbg !12
  %248 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !12
  %249 = fmul <4 x half> %246, splat (half 0xH3015), !dbg !12
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local150, !dbg !12
  %address.local157 = getelementptr i8, ptr addrspace(3) %address.local156, i64 5120, !dbg !12
  %250 = load <4 x half>, ptr addrspace(3) %address.local157, align 2, !dbg !12
  %251 = fmul <4 x half> %247, splat (half 0xH3015), !dbg !12
  %252 = fmul <4 x half> %248, splat (half 0xH3015), !dbg !12
  %253 = fmul <4 x half> %250, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local158 = lshr i32 %address.thread138, 2, !dbg !16
  %address.local159 = and i32 %address.local158, 4, !dbg !16
  %address.local160 = shl nuw nsw i32 %address.thread138, 3, !dbg !16
  %address.local161 = and i32 %address.local160, 376, !dbg !16
  %address.local162 = or disjoint i32 %address.local159, %address.local161, !dbg !16
  %address.local163 = zext nneg i32 %address.local162 to i64, !dbg !16
  %address.local164 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local163, !dbg !16
  %254 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !16
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %249, <4 x half> %254, <4 x float> %151) #9, !dbg !16
  %address.local165 = lshr i32 %address.thread138, 2, !dbg !16
  %address.local166 = and i32 %address.local165, 4, !dbg !16
  %address.local167 = shl nuw nsw i32 %address.thread138, 3, !dbg !16
  %address.local168 = and i32 %address.local167, 376, !dbg !16
  %address.local169 = or disjoint i32 %address.local166, %address.local168, !dbg !16
  %address.local170 = zext nneg i32 %address.local169 to i64, !dbg !16
  %address.local171 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local170, !dbg !16
  %address.local172 = getelementptr i8, ptr addrspace(3) %address.local171, i64 1024, !dbg !16
  %256 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !16
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local163, !dbg !16
  %address.local174 = getelementptr i8, ptr addrspace(3) %address.local173, i64 256, !dbg !16
  %257 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !16
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %251, <4 x half> %256, <4 x float> %255) #9, !dbg !16
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %249, <4 x half> %257, <4 x float> %155) #9, !dbg !16
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local170, !dbg !16
  %address.local176 = getelementptr i8, ptr addrspace(3) %address.local175, i64 1280, !dbg !16
  %260 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %261 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !16
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %251, <4 x half> %260, <4 x float> %259) #9, !dbg !16
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %252, <4 x half> %261, <4 x float> %159) #9, !dbg !16
  %264 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !16
  %265 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !16
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %253, <4 x half> %264, <4 x float> %263) #9, !dbg !16
  %267 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %252, <4 x half> %265, <4 x float> %162) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %253, <4 x half> %267, <4 x float> %268) #9, !dbg !16
  %address.local177 = zext nneg i32 %address.thread138 to i64, !dbg !12
  %address.local178 = shl nuw nsw i64 %address.local177, 3, !dbg !11
  %address.local179 = and i64 %address.local178, 1536, !dbg !11
  %address.local180 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local179, !dbg !11
  %address.local181 = ptrtoint ptr addrspace(3) %address.local180 to i64, !dbg !11
  %address.local182 = trunc i64 %address.local181 to i32, !dbg !11
  %270 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local182), !dbg !11
  %271 = zext i32 %270 to i64, !dbg !11
  %272 = inttoptr i64 %271 to ptr addrspace(3), !dbg !11
  %address.local183 = shl nuw nsw i64 %address.local177, 7, !dbg !11
  %address.local184 = and i64 %address.local183, 16256, !dbg !11
  %address.local185 = lshr i64 %address.local177, 4, !dbg !11
  %address.local186 = and i64 %address.local185, 8, !dbg !11
  %address.local187 = or disjoint i64 %address.local184, %address.local186, !dbg !11
  %address.local188 = getelementptr [2 x i8], ptr addrspace(1) %54, i64 %address.local187, !dbg !11
  %273 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %274 = ptrtoint ptr addrspace(1) %address.local188 to i64, !dbg !11
  %275 = sub i64 %274, %273, !dbg !11
  %dma.buffer.offset649 = trunc i64 %275 to i32, !dbg !11
  %276 = trunc i64 %273 to i32, !dbg !11
  %277 = insertelement <4 x i32> poison, i32 %276, i64 0, !dbg !11
  %278 = lshr i64 %273, 32, !dbg !11
  %279 = trunc i64 %278 to i32, !dbg !11
  %280 = insertelement <4 x i32> %277, i32 %279, i64 1, !dbg !11
  %281 = insertelement <4 x i32> %280, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc650 = insertelement <4 x i32> %281, i32 131072, i64 3, !dbg !11
  %282 = ptrtoint ptr addrspace(3) %272 to i32, !dbg !11
  %dma.buffer.lds651 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %282), !dbg !11
  %283 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds651, i32 %dma.buffer.offset649, <4 x i32> %dma.buffer.rsrc650) #10, !dbg !11
  %address.local189 = or disjoint i64 %address.local177, 256, !dbg !11
  %address.local190 = shl nuw nsw i64 %address.local189, 3, !dbg !11
  %address.local191 = and i64 %address.local190, 3584, !dbg !11
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local191, !dbg !11
  %address.local193 = ptrtoint ptr addrspace(3) %address.local192 to i64, !dbg !11
  %address.local194 = trunc i64 %address.local193 to i32, !dbg !11
  %284 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local194), !dbg !11
  %285 = zext i32 %284 to i64, !dbg !11
  %286 = inttoptr i64 %285 to ptr addrspace(3), !dbg !11
  %address.local195 = shl nuw nsw i64 %address.local177, 7, !dbg !11
  %address.local196 = and i64 %address.local195, 16256, !dbg !11
  %address.local197 = lshr i64 %address.local189, 4, !dbg !11
  %address.local198 = and i64 %address.local197, 24, !dbg !11
  %address.local199 = or disjoint i64 %address.local196, %address.local198, !dbg !11
  %address.local200 = getelementptr [2 x i8], ptr addrspace(1) %54, i64 %address.local199, !dbg !11
  %287 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %288 = ptrtoint ptr addrspace(1) %address.local200 to i64, !dbg !11
  %289 = sub i64 %288, %287, !dbg !11
  %dma.buffer.offset646 = trunc i64 %289 to i32, !dbg !11
  %290 = trunc i64 %287 to i32, !dbg !11
  %291 = insertelement <4 x i32> poison, i32 %290, i64 0, !dbg !11
  %292 = lshr i64 %287, 32, !dbg !11
  %293 = trunc i64 %292 to i32, !dbg !11
  %294 = insertelement <4 x i32> %291, i32 %293, i64 1, !dbg !11
  %295 = insertelement <4 x i32> %294, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc647 = insertelement <4 x i32> %295, i32 131072, i64 3, !dbg !11
  %296 = ptrtoint ptr addrspace(3) %286 to i32, !dbg !11
  %dma.buffer.lds648 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %296), !dbg !11
  %297 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds648, i32 %dma.buffer.offset646, <4 x i32> %dma.buffer.rsrc647) #10, !dbg !11
  %298 = add nuw i64 %58, 8192, !dbg !17
  %299 = icmp slt i64 %298, 0, !dbg !17
  %300 = sub nuw i64 -8193, %58, !dbg !17
  %301 = select i1 %299, i64 %300, i64 %298, !dbg !17
  %302 = sdiv i64 %301, 16384, !dbg !17
  %.lobit2 = ashr i64 %298, 63, !dbg !17
  %303 = xor i64 %302, %.lobit2, !dbg !17
  %304 = add nsw i64 %303, %15, !dbg !17
  %305 = add nsw i64 %198, 64, !dbg !17
  %306 = icmp slt i64 %198, -64, !dbg !17
  %307 = sub nsw i64 -65, %198, !dbg !17
  %308 = select i1 %306, i64 %307, i64 %305, !dbg !17
  %309 = sdiv i64 %308, 128, !dbg !17
  %310 = sext i1 %306 to i64, !dbg !17
  %311 = xor i64 %309, %310, !dbg !17
  %312 = shl nsw i64 %311, 7, !dbg !17
  %313 = sub i64 %52, %312, !dbg !17
  %314 = shl i64 %304, 19, !dbg !17
  %315 = and i64 %314, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %316 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %315, !dbg !17
  %317 = getelementptr [2 x i8], ptr addrspace(1) %316, i64 %98, !dbg !17
  %318 = getelementptr [2 x i8], ptr addrspace(1) %317, i64 %313, !dbg !17
  %319 = getelementptr i8, ptr addrspace(1) %318, i64 128, !dbg !17
  %320 = getelementptr [2 x i8], ptr addrspace(1) %319, i64 %104, !dbg !17
  br label %321, !dbg !17

321:                                              ; preds = %324, %245
  %322 = phi i64 [ %350, %324 ], [ %35, %245 ], !dbg !17
  %323 = icmp samesign ult i64 %322, 128, !dbg !17
  br i1 %323, label %324, label %351, !dbg !17

324:                                              ; preds = %321
  %325 = lshr i64 %322, 2, !dbg !17
  %326 = and i64 %325, 24, !dbg !17
  %327 = shl nuw nsw i64 %322, 3, !dbg !17
  %328 = and i64 %327, 512, !dbg !17
  %329 = shl nuw nsw i64 %322, 7, !dbg !17
  %330 = and i64 %329, 3968, !dbg !17
  %331 = getelementptr [2 x i8], ptr addrspace(1) %320, i64 %326, !dbg !17
  %332 = getelementptr [2 x i8], ptr addrspace(1) %331, i64 %330, !dbg !17
  %333 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %328, !dbg !17
  %334 = ptrtoint ptr addrspace(3) %333 to i64, !dbg !17
  %335 = trunc i64 %334 to i32, !dbg !17
  %336 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %335), !dbg !17
  %337 = zext i32 %336 to i64, !dbg !17
  %338 = inttoptr i64 %337 to ptr addrspace(3), !dbg !17
  %339 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %340 = ptrtoint ptr addrspace(1) %332 to i64, !dbg !17
  %341 = sub i64 %340, %339, !dbg !17
  %dma.buffer.offset622 = trunc i64 %341 to i32, !dbg !17
  %342 = trunc i64 %339 to i32, !dbg !17
  %343 = insertelement <4 x i32> poison, i32 %342, i64 0, !dbg !17
  %344 = lshr i64 %339, 32, !dbg !17
  %345 = trunc i64 %344 to i32, !dbg !17
  %346 = insertelement <4 x i32> %343, i32 %345, i64 1, !dbg !17
  %347 = insertelement <4 x i32> %346, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc623 = insertelement <4 x i32> %347, i32 131072, i64 3, !dbg !17
  %348 = ptrtoint ptr addrspace(3) %338 to i32, !dbg !17
  %dma.buffer.lds624 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %348), !dbg !17
  %349 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds624, i32 %dma.buffer.offset622, <4 x i32> %dma.buffer.rsrc623) #10, !dbg !17
  %350 = or disjoint i64 %35, 256, !dbg !17
  br label %321, !dbg !17

351:                                              ; preds = %321
  %address.thread201 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local202 = lshr i32 %address.thread201, 2, !dbg !12
  %address.local203 = and i32 %address.local202, 4, !dbg !12
  %address.local204 = shl nuw nsw i32 %address.thread201, 1, !dbg !12
  %address.local205 = and i32 %address.local204, 384, !dbg !12
  %address.local206 = or disjoint i32 %address.local203, %address.local205, !dbg !12
  %address.local207 = shl nuw nsw i32 %address.thread201, 3, !dbg !12
  %address.local208 = and i32 %address.local207, 120, !dbg !12
  %address.local209 = or disjoint i32 %address.local206, %address.local208, !dbg !12
  %address.local210 = shl nuw nsw i32 %address.thread201, 5, !dbg !12
  %address.local211 = and i32 %address.local210, 1024, !dbg !12
  %address.local212 = or disjoint i32 %address.local209, %address.local211, !dbg !12
  %address.local213 = zext nneg i32 %address.local212 to i64, !dbg !12
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local213, !dbg !12
  %352 = load <4 x half>, ptr addrspace(3) %address.local214, align 2, !dbg !12
  %address.local215 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local213, !dbg !12
  %address.local216 = getelementptr i8, ptr addrspace(3) %address.local215, i64 4096, !dbg !12
  %353 = load <4 x half>, ptr addrspace(3) %address.local216, align 2, !dbg !12
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local213, !dbg !12
  %address.local218 = getelementptr i8, ptr addrspace(3) %address.local217, i64 1024, !dbg !12
  %354 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !12
  %355 = fmul <4 x half> %352, splat (half 0xH3015), !dbg !12
  %address.local219 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local213, !dbg !12
  %address.local220 = getelementptr i8, ptr addrspace(3) %address.local219, i64 5120, !dbg !12
  %356 = load <4 x half>, ptr addrspace(3) %address.local220, align 2, !dbg !12
  %357 = fmul <4 x half> %353, splat (half 0xH3015), !dbg !12
  %358 = fmul <4 x half> %354, splat (half 0xH3015), !dbg !12
  %359 = fmul <4 x half> %356, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local221 = lshr i32 %address.thread201, 2, !dbg !16
  %address.local222 = and i32 %address.local221, 4, !dbg !16
  %address.local223 = shl nuw nsw i32 %address.thread201, 3, !dbg !16
  %address.local224 = and i32 %address.local223, 376, !dbg !16
  %address.local225 = or disjoint i32 %address.local222, %address.local224, !dbg !16
  %address.local226 = zext nneg i32 %address.local225 to i64, !dbg !16
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local226, !dbg !16
  %360 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !16
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %355, <4 x half> %360, <4 x float> %258) #9, !dbg !16
  %address.local228 = lshr i32 %address.thread201, 2, !dbg !16
  %address.local229 = and i32 %address.local228, 4, !dbg !16
  %address.local230 = shl nuw nsw i32 %address.thread201, 3, !dbg !16
  %address.local231 = and i32 %address.local230, 376, !dbg !16
  %address.local232 = or disjoint i32 %address.local229, %address.local231, !dbg !16
  %address.local233 = zext nneg i32 %address.local232 to i64, !dbg !16
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local233, !dbg !16
  %address.local235 = getelementptr i8, ptr addrspace(3) %address.local234, i64 1024, !dbg !16
  %362 = load <4 x half>, ptr addrspace(3) %address.local235, align 2, !dbg !16
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local226, !dbg !16
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local236, i64 256, !dbg !16
  %363 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !16
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %362, <4 x float> %361) #9, !dbg !16
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %355, <4 x half> %363, <4 x float> %262) #9, !dbg !16
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local233, !dbg !16
  %address.local239 = getelementptr i8, ptr addrspace(3) %address.local238, i64 1280, !dbg !16
  %366 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %367 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !16
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %366, <4 x float> %365) #9, !dbg !16
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %367, <4 x float> %266) #9, !dbg !16
  %370 = load <4 x half>, ptr addrspace(3) %address.local235, align 2, !dbg !16
  %371 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !16
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %370, <4 x float> %369) #9, !dbg !16
  %373 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %371, <4 x float> %269) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %373, <4 x float> %374) #9, !dbg !16
  %address.local240 = zext nneg i32 %address.thread201 to i64, !dbg !12
  %address.local241 = shl nuw nsw i64 %address.local240, 3, !dbg !11
  %address.local242 = and i64 %address.local241, 1536, !dbg !11
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local242, !dbg !11
  %address.local244 = ptrtoint ptr addrspace(3) %address.local243 to i64, !dbg !11
  %address.local245 = trunc i64 %address.local244 to i32, !dbg !11
  %376 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local245), !dbg !11
  %377 = zext i32 %376 to i64, !dbg !11
  %378 = inttoptr i64 %377 to ptr addrspace(3), !dbg !11
  %address.local246 = shl nuw nsw i64 %address.local240, 7, !dbg !11
  %address.local247 = and i64 %address.local246, 16256, !dbg !11
  %address.local248 = lshr i64 %address.local240, 4, !dbg !11
  %address.local249 = and i64 %address.local248, 8, !dbg !11
  %address.local250 = or disjoint i64 %address.local247, %address.local249, !dbg !11
  %address.local251 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local250, !dbg !11
  %379 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %380 = ptrtoint ptr addrspace(1) %address.local251 to i64, !dbg !11
  %381 = sub i64 %380, %379, !dbg !11
  %dma.buffer.offset643 = trunc i64 %381 to i32, !dbg !11
  %382 = trunc i64 %379 to i32, !dbg !11
  %383 = insertelement <4 x i32> poison, i32 %382, i64 0, !dbg !11
  %384 = lshr i64 %379, 32, !dbg !11
  %385 = trunc i64 %384 to i32, !dbg !11
  %386 = insertelement <4 x i32> %383, i32 %385, i64 1, !dbg !11
  %387 = insertelement <4 x i32> %386, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc644 = insertelement <4 x i32> %387, i32 131072, i64 3, !dbg !11
  %388 = ptrtoint ptr addrspace(3) %378 to i32, !dbg !11
  %dma.buffer.lds645 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %388), !dbg !11
  %389 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds645, i32 %dma.buffer.offset643, <4 x i32> %dma.buffer.rsrc644) #10, !dbg !11
  %address.local252 = or disjoint i64 %address.local240, 256, !dbg !11
  %address.local253 = shl nuw nsw i64 %address.local252, 3, !dbg !11
  %address.local254 = and i64 %address.local253, 3584, !dbg !11
  %address.local255 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local254, !dbg !11
  %address.local256 = ptrtoint ptr addrspace(3) %address.local255 to i64, !dbg !11
  %address.local257 = trunc i64 %address.local256 to i32, !dbg !11
  %390 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local257), !dbg !11
  %391 = zext i32 %390 to i64, !dbg !11
  %392 = inttoptr i64 %391 to ptr addrspace(3), !dbg !11
  %address.local258 = shl nuw nsw i64 %address.local240, 7, !dbg !11
  %address.local259 = and i64 %address.local258, 16256, !dbg !11
  %address.local260 = lshr i64 %address.local252, 4, !dbg !11
  %address.local261 = and i64 %address.local260, 24, !dbg !11
  %address.local262 = or disjoint i64 %address.local259, %address.local261, !dbg !11
  %address.local263 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local262, !dbg !11
  %393 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %394 = ptrtoint ptr addrspace(1) %address.local263 to i64, !dbg !11
  %395 = sub i64 %394, %393, !dbg !11
  %dma.buffer.offset640 = trunc i64 %395 to i32, !dbg !11
  %396 = trunc i64 %393 to i32, !dbg !11
  %397 = insertelement <4 x i32> poison, i32 %396, i64 0, !dbg !11
  %398 = lshr i64 %393, 32, !dbg !11
  %399 = trunc i64 %398 to i32, !dbg !11
  %400 = insertelement <4 x i32> %397, i32 %399, i64 1, !dbg !11
  %401 = insertelement <4 x i32> %400, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc641 = insertelement <4 x i32> %401, i32 131072, i64 3, !dbg !11
  %402 = ptrtoint ptr addrspace(3) %392 to i32, !dbg !11
  %dma.buffer.lds642 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %402), !dbg !11
  %403 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds642, i32 %dma.buffer.offset640, <4 x i32> %dma.buffer.rsrc641) #10, !dbg !11
  %404 = add nuw i64 %58, 12288, !dbg !17
  %405 = icmp slt i64 %404, 0, !dbg !17
  %406 = sub nuw i64 -12289, %58, !dbg !17
  %407 = select i1 %405, i64 %406, i64 %404, !dbg !17
  %408 = sdiv i64 %407, 16384, !dbg !17
  %.lobit3 = ashr i64 %404, 63, !dbg !17
  %409 = xor i64 %408, %.lobit3, !dbg !17
  %410 = add nsw i64 %409, %15, !dbg !17
  %411 = add nsw i64 %198, 96, !dbg !17
  %412 = icmp slt i64 %198, -96, !dbg !17
  %413 = sub nsw i64 -97, %198, !dbg !17
  %414 = select i1 %412, i64 %413, i64 %411, !dbg !17
  %415 = sdiv i64 %414, 128, !dbg !17
  %416 = sext i1 %412 to i64, !dbg !17
  %417 = xor i64 %415, %416, !dbg !17
  %418 = shl nsw i64 %417, 7, !dbg !17
  %419 = sub i64 %52, %418, !dbg !17
  %420 = shl i64 %410, 19, !dbg !17
  %421 = and i64 %420, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %422 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %421, !dbg !17
  %423 = getelementptr [2 x i8], ptr addrspace(1) %422, i64 %98, !dbg !17
  %424 = getelementptr [2 x i8], ptr addrspace(1) %423, i64 %419, !dbg !17
  %425 = getelementptr i8, ptr addrspace(1) %424, i64 192, !dbg !17
  %426 = getelementptr [2 x i8], ptr addrspace(1) %425, i64 %104, !dbg !17
  br label %427, !dbg !17

427:                                              ; preds = %430, %351
  %428 = phi i64 [ %456, %430 ], [ %35, %351 ], !dbg !17
  %429 = icmp samesign ult i64 %428, 128, !dbg !17
  br i1 %429, label %430, label %457, !dbg !17

430:                                              ; preds = %427
  %431 = lshr i64 %428, 2, !dbg !17
  %432 = and i64 %431, 24, !dbg !17
  %433 = shl nuw nsw i64 %428, 3, !dbg !17
  %434 = and i64 %433, 512, !dbg !17
  %435 = shl nuw nsw i64 %428, 7, !dbg !17
  %436 = and i64 %435, 3968, !dbg !17
  %437 = getelementptr [2 x i8], ptr addrspace(1) %426, i64 %432, !dbg !17
  %438 = getelementptr [2 x i8], ptr addrspace(1) %437, i64 %436, !dbg !17
  %439 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %434, !dbg !17
  %440 = ptrtoint ptr addrspace(3) %439 to i64, !dbg !17
  %441 = trunc i64 %440 to i32, !dbg !17
  %442 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %441), !dbg !17
  %443 = zext i32 %442 to i64, !dbg !17
  %444 = inttoptr i64 %443 to ptr addrspace(3), !dbg !17
  %445 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %446 = ptrtoint ptr addrspace(1) %438 to i64, !dbg !17
  %447 = sub i64 %446, %445, !dbg !17
  %dma.buffer.offset625 = trunc i64 %447 to i32, !dbg !17
  %448 = trunc i64 %445 to i32, !dbg !17
  %449 = insertelement <4 x i32> poison, i32 %448, i64 0, !dbg !17
  %450 = lshr i64 %445, 32, !dbg !17
  %451 = trunc i64 %450 to i32, !dbg !17
  %452 = insertelement <4 x i32> %449, i32 %451, i64 1, !dbg !17
  %453 = insertelement <4 x i32> %452, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc626 = insertelement <4 x i32> %453, i32 131072, i64 3, !dbg !17
  %454 = ptrtoint ptr addrspace(3) %444 to i32, !dbg !17
  %dma.buffer.lds627 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %454), !dbg !17
  %455 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds627, i32 %dma.buffer.offset625, <4 x i32> %dma.buffer.rsrc626) #10, !dbg !17
  %456 = or disjoint i64 %35, 256, !dbg !17
  br label %427, !dbg !17

457:                                              ; preds = %427
  %address.thread264 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local265 = lshr i32 %address.thread264, 2, !dbg !12
  %address.local266 = and i32 %address.local265, 4, !dbg !12
  %address.local267 = shl nuw nsw i32 %address.thread264, 1, !dbg !12
  %address.local268 = and i32 %address.local267, 384, !dbg !12
  %address.local269 = or disjoint i32 %address.local266, %address.local268, !dbg !12
  %address.local270 = shl nuw nsw i32 %address.thread264, 3, !dbg !12
  %address.local271 = and i32 %address.local270, 120, !dbg !12
  %address.local272 = or disjoint i32 %address.local269, %address.local271, !dbg !12
  %address.local273 = shl nuw nsw i32 %address.thread264, 5, !dbg !12
  %address.local274 = and i32 %address.local273, 1024, !dbg !12
  %address.local275 = or disjoint i32 %address.local272, %address.local274, !dbg !12
  %address.local276 = zext nneg i32 %address.local275 to i64, !dbg !12
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local276, !dbg !12
  %458 = load <4 x half>, ptr addrspace(3) %address.local277, align 2, !dbg !12
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local276, !dbg !12
  %address.local279 = getelementptr i8, ptr addrspace(3) %address.local278, i64 4096, !dbg !12
  %459 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !12
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local276, !dbg !12
  %address.local281 = getelementptr i8, ptr addrspace(3) %address.local280, i64 1024, !dbg !12
  %460 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !12
  %461 = fmul <4 x half> %458, splat (half 0xH3015), !dbg !12
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local276, !dbg !12
  %address.local283 = getelementptr i8, ptr addrspace(3) %address.local282, i64 5120, !dbg !12
  %462 = load <4 x half>, ptr addrspace(3) %address.local283, align 2, !dbg !12
  %463 = fmul <4 x half> %459, splat (half 0xH3015), !dbg !12
  %464 = fmul <4 x half> %460, splat (half 0xH3015), !dbg !12
  %465 = fmul <4 x half> %462, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %466 = add nuw nsw i64 %98, %15, !dbg !18
  %467 = shl i64 %466, 19, !dbg !18
  %468 = and i64 %467, 16252928, !dbg !18
  %469 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %468, !dbg !18
  %470 = getelementptr [2 x i8], ptr addrspace(1) %469, i64 %104, !dbg !18
  br label %471, !dbg !18

471:                                              ; preds = %474, %457
  %472 = phi i64 [ %500, %474 ], [ %35, %457 ], !dbg !18
  %473 = icmp samesign ult i64 %472, 128, !dbg !18
  br i1 %473, label %474, label %501, !dbg !18

474:                                              ; preds = %471
  %475 = shl nuw nsw i64 %472, 3, !dbg !18
  %476 = and i64 %475, 24, !dbg !18
  %477 = shl nuw nsw i64 %472, 3, !dbg !18
  %478 = and i64 %477, 512, !dbg !18
  %479 = shl nuw nsw i64 %472, 5, !dbg !18
  %480 = and i64 %479, 3968, !dbg !18
  %481 = getelementptr [2 x i8], ptr addrspace(1) %470, i64 %480, !dbg !18
  %482 = getelementptr [2 x i8], ptr addrspace(1) %481, i64 %476, !dbg !18
  %483 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %478, !dbg !18
  %484 = ptrtoint ptr addrspace(3) %483 to i64, !dbg !18
  %485 = trunc i64 %484 to i32, !dbg !18
  %486 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %485), !dbg !18
  %487 = zext i32 %486 to i64, !dbg !18
  %488 = inttoptr i64 %487 to ptr addrspace(3), !dbg !18
  %489 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %490 = ptrtoint ptr addrspace(1) %482 to i64, !dbg !18
  %491 = sub i64 %490, %489, !dbg !18
  %dma.buffer.offset628 = trunc i64 %491 to i32, !dbg !18
  %492 = trunc i64 %489 to i32, !dbg !18
  %493 = insertelement <4 x i32> poison, i32 %492, i64 0, !dbg !18
  %494 = lshr i64 %489, 32, !dbg !18
  %495 = trunc i64 %494 to i32, !dbg !18
  %496 = insertelement <4 x i32> %493, i32 %495, i64 1, !dbg !18
  %497 = insertelement <4 x i32> %496, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc629 = insertelement <4 x i32> %497, i32 131072, i64 3, !dbg !18
  %498 = ptrtoint ptr addrspace(3) %488 to i32, !dbg !18
  %dma.buffer.lds630 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %498), !dbg !18
  %499 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds630, i32 %dma.buffer.offset628, <4 x i32> %dma.buffer.rsrc629) #10, !dbg !18
  %500 = or disjoint i64 %35, 256, !dbg !18
  br label %471, !dbg !18

501:                                              ; preds = %471
  %address.thread284 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local285 = lshr i32 %address.thread284, 2, !dbg !16
  %address.local286 = and i32 %address.local285, 4, !dbg !16
  %address.local287 = shl nuw nsw i32 %address.thread284, 3, !dbg !16
  %address.local288 = and i32 %address.local287, 376, !dbg !16
  %address.local289 = or disjoint i32 %address.local286, %address.local288, !dbg !16
  %address.local290 = zext nneg i32 %address.local289 to i64, !dbg !16
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local290, !dbg !16
  %502 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !16
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %461, <4 x half> %502, <4 x float> %364) #9, !dbg !16
  %address.local292 = lshr i32 %address.thread284, 2, !dbg !16
  %address.local293 = and i32 %address.local292, 4, !dbg !16
  %address.local294 = shl nuw nsw i32 %address.thread284, 3, !dbg !16
  %address.local295 = and i32 %address.local294, 376, !dbg !16
  %address.local296 = or disjoint i32 %address.local293, %address.local295, !dbg !16
  %address.local297 = zext nneg i32 %address.local296 to i64, !dbg !16
  %address.local298 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local297, !dbg !16
  %address.local299 = getelementptr i8, ptr addrspace(3) %address.local298, i64 1024, !dbg !16
  %504 = load <4 x half>, ptr addrspace(3) %address.local299, align 2, !dbg !16
  %address.local300 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local290, !dbg !16
  %address.local301 = getelementptr i8, ptr addrspace(3) %address.local300, i64 256, !dbg !16
  %505 = load <4 x half>, ptr addrspace(3) %address.local301, align 2, !dbg !16
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %463, <4 x half> %504, <4 x float> %503) #9, !dbg !16
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %461, <4 x half> %505, <4 x float> %368) #9, !dbg !16
  %508 = extractelement <4 x float> %506, i64 0, !dbg !22
  %509 = extractelement <4 x float> %506, i64 1, !dbg !22
  %510 = select i1 %.not, float %509, float %508, !dbg !13
  %511 = extractelement <4 x float> %506, i64 2, !dbg !22
  %512 = extractelement <4 x float> %506, i64 3, !dbg !22
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local297, !dbg !16
  %address.local303 = getelementptr i8, ptr addrspace(3) %address.local302, i64 1280, !dbg !16
  %513 = load <4 x half>, ptr addrspace(3) %address.local303, align 2, !dbg !16
  %514 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !16
  %515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %463, <4 x half> %513, <4 x float> %507) #9, !dbg !16
  %516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %464, <4 x half> %514, <4 x float> %372) #9, !dbg !16
  %517 = extractelement <4 x float> %515, i64 0, !dbg !22
  %518 = extractelement <4 x float> %515, i64 1, !dbg !22
  %519 = extractelement <4 x float> %515, i64 2, !dbg !22
  %520 = extractelement <4 x float> %515, i64 3, !dbg !22
  %521 = load <4 x half>, ptr addrspace(3) %address.local299, align 2, !dbg !16
  %522 = load <4 x half>, ptr addrspace(3) %address.local301, align 2, !dbg !16
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %465, <4 x half> %521, <4 x float> %516) #9, !dbg !16
  %524 = load <4 x half>, ptr addrspace(3) %address.local303, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %464, <4 x half> %522, <4 x float> %375) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !22
  %526 = extractelement <4 x float> %523, i64 0, !dbg !22
  %527 = extractelement <4 x float> %523, i64 1, !dbg !22
  %528 = extractelement <4 x float> %523, i64 2, !dbg !22
  %529 = extractelement <4 x float> %523, i64 3, !dbg !22
  %530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %465, <4 x half> %524, <4 x float> %525) #9, !dbg !16
  %531 = extractelement <4 x float> %530, i64 0, !dbg !22
  %532 = extractelement <4 x float> %530, i64 1, !dbg !22
  %533 = extractelement <4 x float> %530, i64 2, !dbg !22
  %534 = extractelement <4 x float> %530, i64 3, !dbg !22
  %535 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %536 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %535), !dbg !13
  %537 = add nuw nsw i32 %536, 64, !dbg !13
  %538 = xor i32 %536, 16, !dbg !13
  %539 = and i32 %537, 192, !dbg !13
  %540 = icmp samesign ult i32 %538, %539, !dbg !13
  %541 = select i1 %540, i32 %538, i32 %536, !dbg !13
  %542 = shl nuw nsw i32 %541, 2, !dbg !13
  %543 = bitcast float %510 to i32, !dbg !13
  %544 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %543), !dbg !13
  %545 = bitcast i32 %544 to float, !dbg !13
  %546 = select i1 %.not, float %508, float %545, !dbg !13
  %547 = select i1 %.not, float %545, float %509, !dbg !13
  %548 = select i1 %.not, float %512, float %511, !dbg !13
  %549 = bitcast float %548 to i32, !dbg !13
  %550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %549), !dbg !13
  %551 = bitcast i32 %550 to float, !dbg !13
  %552 = select i1 %.not, float %511, float %551, !dbg !13
  %553 = select i1 %.not, float %551, float %512, !dbg !13
  %554 = select i1 %.not1, float %552, float %546, !dbg !13
  %555 = xor i32 %536, 32, !dbg !13
  %556 = icmp samesign ult i32 %555, %539, !dbg !13
  %557 = select i1 %556, i32 %555, i32 %536, !dbg !13
  %558 = shl nuw nsw i32 %557, 2, !dbg !13
  %559 = bitcast float %554 to i32, !dbg !13
  %560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %559), !dbg !13
  %561 = bitcast i32 %560 to float, !dbg !13
  %562 = select i1 %.not1, float %546, float %561, !dbg !13
  %563 = select i1 %.not1, float %561, float %552, !dbg !13
  %564 = select i1 %.not1, float %553, float %547, !dbg !13
  %565 = bitcast float %564 to i32, !dbg !13
  %566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %565), !dbg !13
  %567 = bitcast i32 %566 to float, !dbg !13
  %568 = select i1 %.not1, float %547, float %567, !dbg !13
  %569 = select i1 %.not1, float %567, float %553, !dbg !13
  %570 = select i1 %.not, float %518, float %517, !dbg !13
  %571 = bitcast float %570 to i32, !dbg !13
  %572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %571), !dbg !13
  %573 = bitcast i32 %572 to float, !dbg !13
  %574 = select i1 %.not, float %517, float %573, !dbg !13
  %575 = select i1 %.not, float %573, float %518, !dbg !13
  %576 = select i1 %.not, float %520, float %519, !dbg !13
  %577 = bitcast float %576 to i32, !dbg !13
  %578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %577), !dbg !13
  %579 = bitcast i32 %578 to float, !dbg !13
  %580 = select i1 %.not, float %519, float %579, !dbg !13
  %581 = select i1 %.not, float %579, float %520, !dbg !13
  %582 = select i1 %.not1, float %580, float %574, !dbg !13
  %583 = bitcast float %582 to i32, !dbg !13
  %584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %583), !dbg !13
  %585 = bitcast i32 %584 to float, !dbg !13
  %586 = select i1 %.not1, float %574, float %585, !dbg !13
  %587 = select i1 %.not1, float %585, float %580, !dbg !13
  %588 = select i1 %.not1, float %581, float %575, !dbg !13
  %589 = bitcast float %588 to i32, !dbg !13
  %590 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %589), !dbg !13
  %591 = bitcast i32 %590 to float, !dbg !13
  %592 = select i1 %.not1, float %575, float %591, !dbg !13
  %593 = select i1 %.not1, float %591, float %581, !dbg !13
  %594 = select i1 %.not, float %527, float %526, !dbg !13
  %595 = bitcast float %594 to i32, !dbg !13
  %596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %595), !dbg !13
  %597 = bitcast i32 %596 to float, !dbg !13
  %598 = select i1 %.not, float %526, float %597, !dbg !13
  %599 = select i1 %.not, float %597, float %527, !dbg !13
  %600 = select i1 %.not, float %529, float %528, !dbg !13
  %601 = bitcast float %600 to i32, !dbg !13
  %602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %601), !dbg !13
  %603 = bitcast i32 %602 to float, !dbg !13
  %604 = select i1 %.not, float %528, float %603, !dbg !13
  %605 = select i1 %.not, float %603, float %529, !dbg !13
  %606 = select i1 %.not1, float %604, float %598, !dbg !13
  %607 = bitcast float %606 to i32, !dbg !13
  %608 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %607), !dbg !13
  %609 = bitcast i32 %608 to float, !dbg !13
  %610 = select i1 %.not1, float %598, float %609, !dbg !13
  %611 = select i1 %.not1, float %609, float %604, !dbg !13
  %612 = select i1 %.not1, float %605, float %599, !dbg !13
  %613 = bitcast float %612 to i32, !dbg !13
  %614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %613), !dbg !13
  %615 = bitcast i32 %614 to float, !dbg !13
  %616 = select i1 %.not1, float %599, float %615, !dbg !13
  %617 = select i1 %.not1, float %615, float %605, !dbg !13
  %618 = select i1 %.not, float %532, float %531, !dbg !13
  %619 = bitcast float %618 to i32, !dbg !13
  %620 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %619), !dbg !13
  %621 = bitcast i32 %620 to float, !dbg !13
  %622 = select i1 %.not, float %531, float %621, !dbg !13
  %623 = select i1 %.not, float %621, float %532, !dbg !13
  %624 = select i1 %.not, float %534, float %533, !dbg !13
  %625 = bitcast float %624 to i32, !dbg !13
  %626 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %625), !dbg !13
  %627 = bitcast i32 %626 to float, !dbg !13
  %628 = select i1 %.not, float %533, float %627, !dbg !13
  %629 = select i1 %.not, float %627, float %534, !dbg !13
  %630 = select i1 %.not1, float %628, float %622, !dbg !13
  %631 = bitcast float %630 to i32, !dbg !13
  %632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %631), !dbg !13
  %633 = bitcast i32 %632 to float, !dbg !13
  %634 = select i1 %.not1, float %622, float %633, !dbg !13
  %635 = select i1 %.not1, float %633, float %628, !dbg !13
  %636 = select i1 %.not1, float %629, float %623, !dbg !13
  %637 = bitcast float %636 to i32, !dbg !13
  %638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %637), !dbg !13
  %639 = bitcast i32 %638 to float, !dbg !13
  %640 = select i1 %.not1, float %623, float %639, !dbg !13
  %641 = select i1 %.not1, float %639, float %629, !dbg !13
  %642 = or disjoint i64 %67, 31, !dbg !14
  %643 = icmp uge i64 %19, %642, !dbg !14
  %644 = and i1 %43, %643, !dbg !14
  br i1 %644, label %645, label %646, !dbg !14

645:                                              ; preds = %501
  br label %693, !dbg !14

646:                                              ; preds = %501
  %647 = add i64 %19, %35, !dbg !14
  %648 = shl nuw nsw i64 %36, 4, !dbg !14
  %649 = add i64 %647, %648, !dbg !14
  %650 = and i32 %34, 240, !dbg !14
  %index.compact = zext nneg i32 %650 to i64, !dbg !14
  %651 = sub i64 %649, %index.compact, !dbg !14
  %652 = or disjoint i64 %67, %38, !dbg !14
  %653 = add i64 %651, 1, !dbg !23
  %654 = lshr i64 %653, 32, !dbg !24
  %compare.high = trunc i64 %654 to i32, !dbg !24
  %655 = lshr i64 %652, 32, !dbg !24
  %compare.high555 = trunc i64 %655 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high555, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high555, !dbg !24
  %compare.low.base = trunc i64 %653 to i32, !dbg !24
  %compare.low.base556 = trunc i64 %652 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base556, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %656 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base556, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low557 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result558 = select i1 %compare.high.equal, i1 %compare.low557, i1 %compare.high.order, !dbg !24
  %657 = select i1 %compare.result558, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base559 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base556, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local560 = or i32 %compare.local.base559, 2, !dbg !24
  %compare.low561 = icmp ugt i32 %compare.low.base, %compare.local560, !dbg !24
  %compare.result562 = select i1 %compare.high.equal, i1 %compare.low561, i1 %compare.high.order, !dbg !24
  %658 = select i1 %compare.result562, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base563 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base556, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local564 = or i32 %compare.local.base563, 3, !dbg !24
  %compare.low565 = icmp ugt i32 %compare.low.base, %compare.local564, !dbg !24
  %compare.result566 = select i1 %compare.high.equal, i1 %compare.low565, i1 %compare.high.order, !dbg !24
  %659 = select i1 %compare.result566, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base567 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base556, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local568 = or i32 %compare.local.base567, 16, !dbg !24
  %compare.low569 = icmp ugt i32 %compare.low.base, %compare.local568, !dbg !24
  %compare.result570 = select i1 %compare.high.equal, i1 %compare.low569, i1 %compare.high.order, !dbg !24
  %660 = select i1 %compare.result570, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base571 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base556, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local572 = or i32 %compare.local.base571, 17, !dbg !24
  %compare.low573 = icmp ugt i32 %compare.low.base, %compare.local572, !dbg !24
  %compare.result574 = select i1 %compare.high.equal, i1 %compare.low573, i1 %compare.high.order, !dbg !24
  %661 = select i1 %compare.result574, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base575 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base556, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local576 = or i32 %compare.local.base575, 18, !dbg !24
  %compare.low577 = icmp ugt i32 %compare.low.base, %compare.local576, !dbg !24
  %compare.result578 = select i1 %compare.high.equal, i1 %compare.low577, i1 %compare.high.order, !dbg !24
  %662 = select i1 %compare.result578, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base579 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base556, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local580 = or i32 %compare.local.base579, 19, !dbg !24
  %compare.low581 = icmp ugt i32 %compare.low.base, %compare.local580, !dbg !24
  %compare.result582 = select i1 %compare.high.equal, i1 %compare.low581, i1 %compare.high.order, !dbg !24
  %663 = select i1 %compare.result582, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %664 = add i64 %651, 65, !dbg !23
  %665 = lshr i64 %664, 32, !dbg !24
  %compare.high583 = trunc i64 %665 to i32, !dbg !24
  %666 = lshr i64 %652, 32, !dbg !24
  %compare.high584 = trunc i64 %666 to i32, !dbg !24
  %compare.high.equal585 = icmp eq i32 %compare.high583, %compare.high584, !dbg !24
  %compare.high.order586 = icmp ugt i32 %compare.high583, %compare.high584, !dbg !24
  %compare.low.base587 = trunc i64 %664 to i32, !dbg !24
  %compare.low.base588 = trunc i64 %652 to i32, !dbg !24
  %compare.low589 = icmp ugt i32 %compare.low.base587, %compare.low.base588, !dbg !24
  %compare.result590 = select i1 %compare.high.equal585, i1 %compare.low589, i1 %compare.high.order586, !dbg !24
  %667 = select i1 %compare.result590, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base591 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base588, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local592 = or i32 %compare.local.base591, 1, !dbg !24
  %compare.low593 = icmp ugt i32 %compare.low.base587, %compare.local592, !dbg !24
  %compare.result594 = select i1 %compare.high.equal585, i1 %compare.low593, i1 %compare.high.order586, !dbg !24
  %668 = select i1 %compare.result594, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base595 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base588, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local596 = or i32 %compare.local.base595, 2, !dbg !24
  %compare.low597 = icmp ugt i32 %compare.low.base587, %compare.local596, !dbg !24
  %compare.result598 = select i1 %compare.high.equal585, i1 %compare.low597, i1 %compare.high.order586, !dbg !24
  %669 = select i1 %compare.result598, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base599 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base588, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local600 = or i32 %compare.local.base599, 3, !dbg !24
  %compare.low601 = icmp ugt i32 %compare.low.base587, %compare.local600, !dbg !24
  %compare.result602 = select i1 %compare.high.equal585, i1 %compare.low601, i1 %compare.high.order586, !dbg !24
  %670 = select i1 %compare.result602, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base603 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base588, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local604 = or i32 %compare.local.base603, 16, !dbg !24
  %compare.low605 = icmp ugt i32 %compare.low.base587, %compare.local604, !dbg !24
  %compare.result606 = select i1 %compare.high.equal585, i1 %compare.low605, i1 %compare.high.order586, !dbg !24
  %671 = select i1 %compare.result606, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base607 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base588, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local608 = or i32 %compare.local.base607, 17, !dbg !24
  %compare.low609 = icmp ugt i32 %compare.low.base587, %compare.local608, !dbg !24
  %compare.result610 = select i1 %compare.high.equal585, i1 %compare.low609, i1 %compare.high.order586, !dbg !24
  %672 = select i1 %compare.result610, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base611 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base588, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local612 = or i32 %compare.local.base611, 18, !dbg !24
  %compare.low613 = icmp ugt i32 %compare.low.base587, %compare.local612, !dbg !24
  %compare.result614 = select i1 %compare.high.equal585, i1 %compare.low613, i1 %compare.high.order586, !dbg !24
  %673 = select i1 %compare.result614, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base615 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base588, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local616 = or i32 %compare.local.base615, 19, !dbg !24
  %compare.low617 = icmp ugt i32 %compare.low.base587, %compare.local616, !dbg !24
  %compare.result618 = select i1 %compare.high.equal585, i1 %compare.low617, i1 %compare.high.order586, !dbg !24
  %674 = select i1 %compare.result618, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %675 = insertelement <8 x float> poison, float %656, i64 0, !dbg !14
  %676 = insertelement <8 x float> %675, float %657, i64 1, !dbg !14
  %677 = insertelement <8 x float> %676, float %658, i64 2, !dbg !14
  %678 = insertelement <8 x float> %677, float %659, i64 3, !dbg !14
  %679 = insertelement <8 x float> %678, float %660, i64 4, !dbg !14
  %680 = insertelement <8 x float> %679, float %661, i64 5, !dbg !14
  %681 = insertelement <8 x float> %680, float %662, i64 6, !dbg !14
  %682 = insertelement <8 x float> %681, float %663, i64 7, !dbg !14
  %683 = insertvalue [2 x <8 x float>] poison, <8 x float> %682, 0, !dbg !14
  %684 = insertelement <8 x float> poison, float %667, i64 0, !dbg !14
  %685 = insertelement <8 x float> %684, float %668, i64 1, !dbg !14
  %686 = insertelement <8 x float> %685, float %669, i64 2, !dbg !14
  %687 = insertelement <8 x float> %686, float %670, i64 3, !dbg !14
  %688 = insertelement <8 x float> %687, float %671, i64 4, !dbg !14
  %689 = insertelement <8 x float> %688, float %672, i64 5, !dbg !14
  %690 = insertelement <8 x float> %689, float %673, i64 6, !dbg !14
  %691 = insertelement <8 x float> %690, float %674, i64 7, !dbg !14
  %692 = insertvalue [2 x <8 x float>] %683, <8 x float> %691, 1, !dbg !14
  br label %693, !dbg !14

693:                                              ; preds = %645, %646
  %694 = phi [2 x <8 x float>] [ %692, %646 ], [ zeroinitializer, %645 ], !dbg !14
  br label %695, !dbg !14

695:                                              ; preds = %693
  %696 = extractvalue [2 x <8 x float>] %694, 0, !dbg !26
  %697 = insertelement <4 x float> poison, float %562, i64 0, !dbg !26
  %698 = insertelement <4 x float> %697, float %568, i64 1, !dbg !26
  %699 = insertelement <4 x float> %698, float %563, i64 2, !dbg !26
  %700 = insertelement <4 x float> %699, float %569, i64 3, !dbg !26
  %701 = shufflevector <8 x float> %696, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %702 = fadd <4 x float> %700, %701, !dbg !26
  %703 = extractelement <4 x float> %702, i64 0, !dbg !26
  %704 = extractelement <4 x float> %702, i64 1, !dbg !26
  %705 = extractelement <4 x float> %702, i64 2, !dbg !26
  %706 = extractelement <4 x float> %702, i64 3, !dbg !26
  %707 = insertelement <4 x float> poison, float %586, i64 0, !dbg !26
  %708 = insertelement <4 x float> %707, float %592, i64 1, !dbg !26
  %709 = insertelement <4 x float> %708, float %587, i64 2, !dbg !26
  %710 = insertelement <4 x float> %709, float %593, i64 3, !dbg !26
  %711 = shufflevector <8 x float> %696, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %712 = fadd <4 x float> %710, %711, !dbg !26
  %713 = extractelement <4 x float> %712, i64 0, !dbg !26
  %714 = extractelement <4 x float> %712, i64 1, !dbg !26
  %715 = extractelement <4 x float> %712, i64 2, !dbg !26
  %716 = extractelement <4 x float> %712, i64 3, !dbg !26
  %717 = extractvalue [2 x <8 x float>] %694, 1, !dbg !26
  %718 = insertelement <4 x float> poison, float %610, i64 0, !dbg !26
  %719 = insertelement <4 x float> %718, float %616, i64 1, !dbg !26
  %720 = insertelement <4 x float> %719, float %611, i64 2, !dbg !26
  %721 = insertelement <4 x float> %720, float %617, i64 3, !dbg !26
  %722 = shufflevector <8 x float> %717, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %723 = fadd <4 x float> %721, %722, !dbg !26
  %724 = extractelement <4 x float> %723, i64 0, !dbg !26
  %725 = extractelement <4 x float> %723, i64 1, !dbg !26
  %726 = extractelement <4 x float> %723, i64 2, !dbg !26
  %727 = extractelement <4 x float> %723, i64 3, !dbg !26
  %728 = insertelement <4 x float> poison, float %634, i64 0, !dbg !26
  %729 = insertelement <4 x float> %728, float %640, i64 1, !dbg !26
  %730 = insertelement <4 x float> %729, float %635, i64 2, !dbg !26
  %731 = insertelement <4 x float> %730, float %641, i64 3, !dbg !26
  %732 = shufflevector <8 x float> %717, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %733 = fadd <4 x float> %731, %732, !dbg !26
  %734 = extractelement <4 x float> %733, i64 0, !dbg !26
  %735 = extractelement <4 x float> %733, i64 1, !dbg !26
  %736 = extractelement <4 x float> %733, i64 2, !dbg !26
  %737 = extractelement <4 x float> %733, i64 3, !dbg !26
  %738 = call float @llvm.maxnum.f32(float %703, float 0xFFF0000000000000), !dbg !27
  %739 = call float @llvm.maxnum.f32(float %738, float %704), !dbg !27
  %740 = call float @llvm.maxnum.f32(float %739, float %705), !dbg !27
  %741 = call float @llvm.maxnum.f32(float %740, float %706), !dbg !27
  %742 = call float @llvm.maxnum.f32(float %741, float %713), !dbg !27
  %743 = call float @llvm.maxnum.f32(float %742, float %714), !dbg !27
  %744 = call float @llvm.maxnum.f32(float %743, float %715), !dbg !27
  %745 = call float @llvm.maxnum.f32(float %744, float %716), !dbg !27
  %746 = bitcast float %745 to i32, !dbg !27
  %747 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %746), !dbg !27
  %748 = bitcast i32 %747 to float, !dbg !27
  %749 = call float @llvm.maxnum.f32(float %745, float %748), !dbg !27
  %750 = bitcast float %749 to i32, !dbg !27
  %751 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %750), !dbg !27
  %752 = bitcast i32 %751 to float, !dbg !27
  %753 = call float @llvm.maxnum.f32(float %749, float %752), !dbg !27
  %754 = extractvalue [2 x <1 x float>] %61, 0, !dbg !27
  %755 = insertelement <1 x float> poison, float %753, i64 0, !dbg !27
  %756 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %755, <1 x float> %754), !dbg !27
  %757 = call float @llvm.maxnum.f32(float %724, float 0xFFF0000000000000), !dbg !27
  %758 = call float @llvm.maxnum.f32(float %757, float %725), !dbg !27
  %759 = call float @llvm.maxnum.f32(float %758, float %726), !dbg !27
  %760 = call float @llvm.maxnum.f32(float %759, float %727), !dbg !27
  %761 = call float @llvm.maxnum.f32(float %760, float %734), !dbg !27
  %762 = call float @llvm.maxnum.f32(float %761, float %735), !dbg !27
  %763 = call float @llvm.maxnum.f32(float %762, float %736), !dbg !27
  %764 = call float @llvm.maxnum.f32(float %763, float %737), !dbg !27
  %765 = bitcast float %764 to i32, !dbg !27
  %766 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %765), !dbg !27
  %767 = bitcast i32 %766 to float, !dbg !27
  %768 = call float @llvm.maxnum.f32(float %764, float %767), !dbg !27
  %769 = bitcast float %768 to i32, !dbg !27
  %770 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %769), !dbg !27
  %771 = bitcast i32 %770 to float, !dbg !27
  %772 = call float @llvm.maxnum.f32(float %768, float %771), !dbg !27
  %773 = extractvalue [2 x <1 x float>] %61, 1, !dbg !27
  %774 = insertelement <1 x float> poison, float %772, i64 0, !dbg !27
  %775 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %774, <1 x float> %773), !dbg !27
  %776 = shufflevector <1 x float> %756, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %777 = fsub <4 x float> %702, %776, !dbg !28
  %778 = fsub <4 x float> %712, %776, !dbg !28
  %779 = shufflevector <1 x float> %775, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %780 = fsub <4 x float> %723, %779, !dbg !28
  %781 = fsub <4 x float> %733, %779, !dbg !28
  %782 = fsub <1 x float> %754, %756, !dbg !29
  %783 = extractelement <1 x float> %782, i64 0, !dbg !30
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !30
  %785 = extractvalue [2 x <1 x float>] %60, 0, !dbg !31
  %786 = extractelement <1 x float> %785, i64 0, !dbg !31
  %787 = fmul float %786, %784, !dbg !31
  %788 = fsub <1 x float> %773, %775, !dbg !29
  %789 = extractelement <1 x float> %788, i64 0, !dbg !30
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !30
  %791 = extractvalue [2 x <1 x float>] %60, 1, !dbg !31
  %792 = extractelement <1 x float> %791, i64 0, !dbg !31
  %793 = fmul float %792, %790, !dbg !31
  %794 = extractelement <4 x float> %777, i64 0, !dbg !32
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !32
  %796 = insertelement <4 x float> poison, float %795, i64 0, !dbg !34
  %797 = extractelement <4 x float> %777, i64 1, !dbg !32
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !32
  %799 = insertelement <4 x float> %796, float %798, i64 1, !dbg !34
  %800 = fadd float %795, %798, !dbg !33
  %801 = extractelement <4 x float> %777, i64 2, !dbg !32
  %802 = call float @llvm.amdgcn.exp2.f32(float %801), !dbg !32
  %803 = insertelement <4 x float> %799, float %802, i64 2, !dbg !34
  %804 = fadd float %800, %802, !dbg !33
  %805 = extractelement <4 x float> %777, i64 3, !dbg !32
  %806 = call float @llvm.amdgcn.exp2.f32(float %805), !dbg !32
  %807 = insertelement <4 x float> %803, float %806, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %807) #9, !dbg !34, !frisk.register_lifetimes !1
  %808 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %808) #9, !dbg !33, !frisk.register_lifetimes !1
  %809 = fadd float %804, %806, !dbg !33
  %810 = extractelement <4 x float> %778, i64 0, !dbg !32
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !32
  %812 = insertelement <4 x float> poison, float %811, i64 0, !dbg !34
  %813 = fadd float %809, %811, !dbg !33
  %814 = extractelement <4 x float> %778, i64 1, !dbg !32
  %815 = call float @llvm.amdgcn.exp2.f32(float %814), !dbg !32
  %816 = insertelement <4 x float> %812, float %815, i64 1, !dbg !34
  %817 = fadd float %813, %815, !dbg !33
  %818 = extractelement <4 x float> %778, i64 2, !dbg !32
  %819 = call float @llvm.amdgcn.exp2.f32(float %818), !dbg !32
  %820 = insertelement <4 x float> %816, float %819, i64 2, !dbg !34
  %821 = fadd float %817, %819, !dbg !33
  %822 = extractelement <4 x float> %778, i64 3, !dbg !32
  %823 = call float @llvm.amdgcn.exp2.f32(float %822), !dbg !32
  %824 = insertelement <4 x float> %820, float %823, i64 3, !dbg !34
  %math.result46 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %824) #9, !dbg !34, !frisk.register_lifetimes !1
  %825 = fptrunc <4 x float> %math.result46 to <4 x half>, !dbg !34
  %math.packed47 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %825) #9, !dbg !33, !frisk.register_lifetimes !1
  %826 = fadd float %821, %823, !dbg !33
  %827 = extractelement <4 x float> %780, i64 0, !dbg !32
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !32
  %829 = insertelement <4 x float> poison, float %828, i64 0, !dbg !34
  %830 = extractelement <4 x float> %780, i64 1, !dbg !32
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !32
  %832 = insertelement <4 x float> %829, float %831, i64 1, !dbg !34
  %833 = fadd float %828, %831, !dbg !33
  %834 = extractelement <4 x float> %780, i64 2, !dbg !32
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !32
  %836 = insertelement <4 x float> %832, float %835, i64 2, !dbg !34
  %837 = fadd float %833, %835, !dbg !33
  %838 = extractelement <4 x float> %780, i64 3, !dbg !32
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !32
  %840 = insertelement <4 x float> %836, float %839, i64 3, !dbg !34
  %math.result48 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %840) #9, !dbg !34, !frisk.register_lifetimes !1
  %841 = fptrunc <4 x float> %math.result48 to <4 x half>, !dbg !34
  %math.packed49 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %841) #9, !dbg !33, !frisk.register_lifetimes !1
  %842 = fadd float %837, %839, !dbg !33
  %843 = extractelement <4 x float> %781, i64 0, !dbg !32
  %844 = call float @llvm.amdgcn.exp2.f32(float %843), !dbg !32
  %845 = insertelement <4 x float> poison, float %844, i64 0, !dbg !34
  %846 = fadd float %842, %844, !dbg !33
  %847 = extractelement <4 x float> %781, i64 1, !dbg !32
  %848 = call float @llvm.amdgcn.exp2.f32(float %847), !dbg !32
  %849 = insertelement <4 x float> %845, float %848, i64 1, !dbg !34
  %850 = fadd float %846, %848, !dbg !33
  %851 = extractelement <4 x float> %781, i64 2, !dbg !32
  %852 = call float @llvm.amdgcn.exp2.f32(float %851), !dbg !32
  %853 = insertelement <4 x float> %849, float %852, i64 2, !dbg !34
  %854 = fadd float %850, %852, !dbg !33
  %855 = extractelement <4 x float> %781, i64 3, !dbg !32
  %856 = call float @llvm.amdgcn.exp2.f32(float %855), !dbg !32
  %857 = insertelement <4 x float> %853, float %856, i64 3, !dbg !34
  %math.result50 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %857) #9, !dbg !34, !frisk.register_lifetimes !1
  %858 = fptrunc <4 x float> %math.result50 to <4 x half>, !dbg !34
  %math.packed51 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %858) #9, !dbg !33, !frisk.register_lifetimes !1
  %859 = fadd float %854, %856, !dbg !33
  %860 = bitcast float %826 to i32, !dbg !33
  %861 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %860), !dbg !33
  %862 = bitcast i32 %861 to float, !dbg !33
  %863 = fadd float %826, %862, !dbg !33
  %864 = bitcast float %863 to i32, !dbg !33
  %865 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %864), !dbg !33
  %866 = bitcast i32 %865 to float, !dbg !33
  %867 = fadd float %863, %866, !dbg !33
  %868 = fadd float %867, %787, !dbg !33
  %869 = bitcast float %859 to i32, !dbg !33
  %870 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %869), !dbg !33
  %871 = bitcast i32 %870 to float, !dbg !33
  %872 = fadd float %859, %871, !dbg !33
  %873 = bitcast float %872 to i32, !dbg !33
  %874 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %873), !dbg !33
  %875 = bitcast i32 %874 to float, !dbg !33
  %876 = fadd float %872, %875, !dbg !33
  %877 = fadd float %876, %793, !dbg !33
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread304 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local305 = shl nuw nsw i32 %address.thread304, 3, !dbg !15
  %address.local306 = and i32 %address.local305, 504, !dbg !15
  %address.local307 = zext nneg i32 %address.local306 to i64, !dbg !15
  %address.local308 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local307, !dbg !15
  %878 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local308, i32 0) #0, !dbg !15
  %879 = bitcast <4 x i32> %878 to <8 x half>, !dbg !15
  %address.local309 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local307, !dbg !15
  %address.local310 = getelementptr i8, ptr addrspace(3) %address.local309, i64 1024, !dbg !15
  %880 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local310, i32 0) #0, !dbg !15
  %881 = bitcast <4 x i32> %880 to <8 x half>, !dbg !15
  %882 = add nuw nsw i64 %98, %15, !dbg !18
  %883 = shl i64 %882, 19, !dbg !18
  %884 = and i64 %883, 16252928, !dbg !18
  %885 = add i64 %884, %104, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %886 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %885, !dbg !18
  %887 = getelementptr i8, ptr addrspace(1) %886, i64 64, !dbg !18
  br label %888, !dbg !18

888:                                              ; preds = %891, %695
  %889 = phi i64 [ %917, %891 ], [ %35, %695 ], !dbg !18
  %890 = icmp samesign ult i64 %889, 128, !dbg !18
  br i1 %890, label %891, label %918, !dbg !18

891:                                              ; preds = %888
  %892 = shl nuw nsw i64 %889, 3, !dbg !18
  %893 = and i64 %892, 24, !dbg !18
  %894 = shl nuw nsw i64 %889, 3, !dbg !18
  %895 = and i64 %894, 512, !dbg !18
  %896 = shl nuw nsw i64 %889, 5, !dbg !18
  %897 = and i64 %896, 3968, !dbg !18
  %898 = getelementptr [2 x i8], ptr addrspace(1) %887, i64 %897, !dbg !18
  %899 = getelementptr [2 x i8], ptr addrspace(1) %898, i64 %893, !dbg !18
  %900 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %895, !dbg !18
  %901 = ptrtoint ptr addrspace(3) %900 to i64, !dbg !18
  %902 = trunc i64 %901 to i32, !dbg !18
  %903 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %902), !dbg !18
  %904 = zext i32 %903 to i64, !dbg !18
  %905 = inttoptr i64 %904 to ptr addrspace(3), !dbg !18
  %906 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %907 = ptrtoint ptr addrspace(1) %899 to i64, !dbg !18
  %908 = sub i64 %907, %906, !dbg !18
  %dma.buffer.offset631 = trunc i64 %908 to i32, !dbg !18
  %909 = trunc i64 %906 to i32, !dbg !18
  %910 = insertelement <4 x i32> poison, i32 %909, i64 0, !dbg !18
  %911 = lshr i64 %906, 32, !dbg !18
  %912 = trunc i64 %911 to i32, !dbg !18
  %913 = insertelement <4 x i32> %910, i32 %912, i64 1, !dbg !18
  %914 = insertelement <4 x i32> %913, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc632 = insertelement <4 x i32> %914, i32 131072, i64 3, !dbg !18
  %915 = ptrtoint ptr addrspace(3) %905 to i32, !dbg !18
  %dma.buffer.lds633 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %915), !dbg !18
  %916 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds633, i32 %dma.buffer.offset631, <4 x i32> %dma.buffer.rsrc632) #10, !dbg !18
  %917 = or disjoint i64 %35, 256, !dbg !18
  br label %888, !dbg !18

918:                                              ; preds = %888
  %919 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %920 = extractvalue [2 x <8 x float>] %59, 0, !dbg !35
  %921 = shufflevector <8 x float> %920, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %922 = insertelement <4 x float> poison, float %784, i64 0, !dbg !35
  %923 = shufflevector <4 x float> %922, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %924 = fmul <4 x float> %921, %923, !dbg !35
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %919, <4 x float> %924) #9, !dbg !15
  %926 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %926, <4 x float> %925) #9, !dbg !15
  %928 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %929 = shufflevector <8 x float> %920, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %930 = fmul <4 x float> %929, %923, !dbg !35
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %928, <4 x float> %930) #9, !dbg !15
  %932 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %932, <4 x float> %931) #9, !dbg !15
  %934 = extractvalue [2 x <8 x float>] %59, 1, !dbg !35
  %935 = shufflevector <8 x float> %934, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %936 = insertelement <4 x float> poison, float %790, i64 0, !dbg !35
  %937 = shufflevector <4 x float> %936, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %938 = fmul <4 x float> %935, %937, !dbg !35
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %919, <4 x float> %938) #9, !dbg !15
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %926, <4 x float> %939) #9, !dbg !15
  %941 = shufflevector <8 x float> %934, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %942 = fmul <4 x float> %941, %937, !dbg !35
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %928, <4 x float> %942) #9, !dbg !15
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %932, <4 x float> %943) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread311 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local312 = shl nuw nsw i32 %address.thread311, 3, !dbg !15
  %address.local313 = and i32 %address.local312, 504, !dbg !15
  %address.local314 = zext nneg i32 %address.local313 to i64, !dbg !15
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local314, !dbg !15
  %945 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local315, i32 0) #0, !dbg !15
  %946 = bitcast <4 x i32> %945 to <8 x half>, !dbg !15
  %address.local316 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local314, !dbg !15
  %address.local317 = getelementptr i8, ptr addrspace(3) %address.local316, i64 1024, !dbg !15
  %947 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local317, i32 0) #0, !dbg !15
  %948 = bitcast <4 x i32> %947 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %949 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %885, !dbg !18
  %950 = getelementptr i8, ptr addrspace(1) %949, i64 128, !dbg !18
  br label %951, !dbg !18

951:                                              ; preds = %954, %918
  %952 = phi i64 [ %980, %954 ], [ %35, %918 ], !dbg !18
  %953 = icmp samesign ult i64 %952, 128, !dbg !18
  br i1 %953, label %954, label %981, !dbg !18

954:                                              ; preds = %951
  %955 = shl nuw nsw i64 %952, 3, !dbg !18
  %956 = and i64 %955, 24, !dbg !18
  %957 = shl nuw nsw i64 %952, 3, !dbg !18
  %958 = and i64 %957, 512, !dbg !18
  %959 = shl nuw nsw i64 %952, 5, !dbg !18
  %960 = and i64 %959, 3968, !dbg !18
  %961 = getelementptr [2 x i8], ptr addrspace(1) %950, i64 %960, !dbg !18
  %962 = getelementptr [2 x i8], ptr addrspace(1) %961, i64 %956, !dbg !18
  %963 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %958, !dbg !18
  %964 = ptrtoint ptr addrspace(3) %963 to i64, !dbg !18
  %965 = trunc i64 %964 to i32, !dbg !18
  %966 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %965), !dbg !18
  %967 = zext i32 %966 to i64, !dbg !18
  %968 = inttoptr i64 %967 to ptr addrspace(3), !dbg !18
  %969 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %970 = ptrtoint ptr addrspace(1) %962 to i64, !dbg !18
  %971 = sub i64 %970, %969, !dbg !18
  %dma.buffer.offset634 = trunc i64 %971 to i32, !dbg !18
  %972 = trunc i64 %969 to i32, !dbg !18
  %973 = insertelement <4 x i32> poison, i32 %972, i64 0, !dbg !18
  %974 = lshr i64 %969, 32, !dbg !18
  %975 = trunc i64 %974 to i32, !dbg !18
  %976 = insertelement <4 x i32> %973, i32 %975, i64 1, !dbg !18
  %977 = insertelement <4 x i32> %976, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc635 = insertelement <4 x i32> %977, i32 131072, i64 3, !dbg !18
  %978 = ptrtoint ptr addrspace(3) %968 to i32, !dbg !18
  %dma.buffer.lds636 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %978), !dbg !18
  %979 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds636, i32 %dma.buffer.offset634, <4 x i32> %dma.buffer.rsrc635) #10, !dbg !18
  %980 = or disjoint i64 %35, 256, !dbg !18
  br label %951, !dbg !18

981:                                              ; preds = %951
  %982 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %983 = extractvalue [2 x <8 x float>] %62, 0, !dbg !35
  %984 = shufflevector <8 x float> %983, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %985 = fmul <4 x float> %984, %923, !dbg !35
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %982, <4 x float> %985) #9, !dbg !15
  %987 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %987, <4 x float> %986) #9, !dbg !15
  %989 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %990 = shufflevector <8 x float> %983, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %991 = fmul <4 x float> %990, %923, !dbg !35
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %989, <4 x float> %991) #9, !dbg !15
  %993 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %993, <4 x float> %992) #9, !dbg !15
  %995 = extractvalue [2 x <8 x float>] %62, 1, !dbg !35
  %996 = shufflevector <8 x float> %995, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %997 = fmul <4 x float> %996, %937, !dbg !35
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %982, <4 x float> %997) #9, !dbg !15
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %987, <4 x float> %998) #9, !dbg !15
  %1000 = shufflevector <8 x float> %995, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1001 = fmul <4 x float> %1000, %937, !dbg !35
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %989, <4 x float> %1001) #9, !dbg !15
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %993, <4 x float> %1002) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread318 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local319 = shl nuw nsw i32 %address.thread318, 3, !dbg !15
  %address.local320 = and i32 %address.local319, 504, !dbg !15
  %address.local321 = zext nneg i32 %address.local320 to i64, !dbg !15
  %address.local322 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local321, !dbg !15
  %1004 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local322, i32 0) #0, !dbg !15
  %1005 = bitcast <4 x i32> %1004 to <8 x half>, !dbg !15
  %address.local323 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local321, !dbg !15
  %address.local324 = getelementptr i8, ptr addrspace(3) %address.local323, i64 1024, !dbg !15
  %1006 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local324, i32 0) #0, !dbg !15
  %1007 = bitcast <4 x i32> %1006 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1008 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %885, !dbg !18
  %1009 = getelementptr i8, ptr addrspace(1) %1008, i64 192, !dbg !18
  br label %1010, !dbg !18

1010:                                             ; preds = %1013, %981
  %1011 = phi i64 [ %1039, %1013 ], [ %35, %981 ], !dbg !18
  %1012 = icmp samesign ult i64 %1011, 128, !dbg !18
  br i1 %1012, label %1013, label %1040, !dbg !18

1013:                                             ; preds = %1010
  %1014 = shl nuw nsw i64 %1011, 3, !dbg !18
  %1015 = and i64 %1014, 24, !dbg !18
  %1016 = shl nuw nsw i64 %1011, 3, !dbg !18
  %1017 = and i64 %1016, 512, !dbg !18
  %1018 = shl nuw nsw i64 %1011, 5, !dbg !18
  %1019 = and i64 %1018, 3968, !dbg !18
  %1020 = getelementptr [2 x i8], ptr addrspace(1) %1009, i64 %1019, !dbg !18
  %1021 = getelementptr [2 x i8], ptr addrspace(1) %1020, i64 %1015, !dbg !18
  %1022 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1017, !dbg !18
  %1023 = ptrtoint ptr addrspace(3) %1022 to i64, !dbg !18
  %1024 = trunc i64 %1023 to i32, !dbg !18
  %1025 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1024), !dbg !18
  %1026 = zext i32 %1025 to i64, !dbg !18
  %1027 = inttoptr i64 %1026 to ptr addrspace(3), !dbg !18
  %1028 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1029 = ptrtoint ptr addrspace(1) %1021 to i64, !dbg !18
  %1030 = sub i64 %1029, %1028, !dbg !18
  %dma.buffer.offset637 = trunc i64 %1030 to i32, !dbg !18
  %1031 = trunc i64 %1028 to i32, !dbg !18
  %1032 = insertelement <4 x i32> poison, i32 %1031, i64 0, !dbg !18
  %1033 = lshr i64 %1028, 32, !dbg !18
  %1034 = trunc i64 %1033 to i32, !dbg !18
  %1035 = insertelement <4 x i32> %1032, i32 %1034, i64 1, !dbg !18
  %1036 = insertelement <4 x i32> %1035, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc638 = insertelement <4 x i32> %1036, i32 131072, i64 3, !dbg !18
  %1037 = ptrtoint ptr addrspace(3) %1027 to i32, !dbg !18
  %dma.buffer.lds639 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1037), !dbg !18
  %1038 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds639, i32 %dma.buffer.offset637, <4 x i32> %dma.buffer.rsrc638) #10, !dbg !18
  %1039 = or disjoint i64 %35, 256, !dbg !18
  br label %1010, !dbg !18

1040:                                             ; preds = %1010
  %1041 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1042 = extractvalue [2 x <8 x float>] %63, 0, !dbg !35
  %1043 = shufflevector <8 x float> %1042, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1044 = fmul <4 x float> %1043, %923, !dbg !35
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1041, <4 x float> %1044) #9, !dbg !15
  %1046 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %1046, <4 x float> %1045) #9, !dbg !15
  %1048 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1049 = shufflevector <8 x float> %1042, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1050 = fmul <4 x float> %1049, %923, !dbg !35
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1048, <4 x float> %1050) #9, !dbg !15
  %1052 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %1052, <4 x float> %1051) #9, !dbg !15
  %1054 = extractvalue [2 x <8 x float>] %63, 1, !dbg !35
  %1055 = shufflevector <8 x float> %1054, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1056 = fmul <4 x float> %1055, %937, !dbg !35
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %1041, <4 x float> %1056) #9, !dbg !15
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %1046, <4 x float> %1057) #9, !dbg !15
  %1059 = shufflevector <8 x float> %1054, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1060 = fmul <4 x float> %1059, %937, !dbg !35
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %1048, <4 x float> %1060) #9, !dbg !15
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %1052, <4 x float> %1061) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread325 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local326 = shl nuw nsw i32 %address.thread325, 3, !dbg !15
  %address.local327 = and i32 %address.local326, 504, !dbg !15
  %address.local328 = zext nneg i32 %address.local327 to i64, !dbg !15
  %address.local329 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local328, !dbg !15
  %1063 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local329, i32 0) #0, !dbg !15
  %1064 = bitcast <4 x i32> %1063 to <8 x half>, !dbg !15
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local328, !dbg !15
  %address.local331 = getelementptr i8, ptr addrspace(3) %address.local330, i64 1024, !dbg !15
  %1065 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !15
  %1066 = bitcast <4 x i32> %1065 to <8 x half>, !dbg !15
  %1067 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1068 = extractvalue [2 x <8 x float>] %64, 0, !dbg !35
  %1069 = shufflevector <8 x float> %1068, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1070 = fmul <4 x float> %1069, %923, !dbg !35
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1067, <4 x float> %1070) #9, !dbg !15
  %1072 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %1072, <4 x float> %1071) #9, !dbg !15
  %1074 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1075 = shufflevector <8 x float> %1068, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1076 = fmul <4 x float> %1075, %923, !dbg !35
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1074, <4 x float> %1076) #9, !dbg !15
  %1078 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %1078, <4 x float> %1077) #9, !dbg !15
  %1080 = extractvalue [2 x <8 x float>] %64, 1, !dbg !35
  %1081 = shufflevector <8 x float> %1080, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1082 = fmul <4 x float> %1081, %937, !dbg !35
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %1067, <4 x float> %1082) #9, !dbg !15
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %1072, <4 x float> %1083) #9, !dbg !15
  %1085 = shufflevector <8 x float> %1080, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1086 = fmul <4 x float> %1085, %937, !dbg !35
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %1074, <4 x float> %1086) #9, !dbg !15
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %1078, <4 x float> %1087) #9, !dbg !15
  %1089 = shufflevector <4 x float> %927, <4 x float> %933, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1090 = insertvalue [2 x <8 x float>] poison, <8 x float> %1089, 0, !dbg !7
  %1091 = shufflevector <4 x float> %940, <4 x float> %944, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1092 = insertvalue [2 x <8 x float>] %1090, <8 x float> %1091, 1, !dbg !7
  %1093 = insertelement <1 x float> poison, float %868, i64 0, !dbg !7
  %1094 = insertvalue [2 x <1 x float>] poison, <1 x float> %1093, 0, !dbg !7
  %1095 = insertelement <1 x float> poison, float %877, i64 0, !dbg !7
  %1096 = insertvalue [2 x <1 x float>] %1094, <1 x float> %1095, 1, !dbg !7
  %1097 = insertvalue [2 x <1 x float>] poison, <1 x float> %756, 0, !dbg !7
  %1098 = insertvalue [2 x <1 x float>] %1097, <1 x float> %775, 1, !dbg !7
  %1099 = shufflevector <4 x float> %988, <4 x float> %994, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1100 = insertvalue [2 x <8 x float>] poison, <8 x float> %1099, 0, !dbg !7
  %1101 = shufflevector <4 x float> %999, <4 x float> %1003, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1102 = insertvalue [2 x <8 x float>] %1100, <8 x float> %1101, 1, !dbg !7
  %1103 = shufflevector <4 x float> %1047, <4 x float> %1053, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1104 = insertvalue [2 x <8 x float>] poison, <8 x float> %1103, 0, !dbg !7
  %1105 = shufflevector <4 x float> %1058, <4 x float> %1062, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1106 = insertvalue [2 x <8 x float>] %1104, <8 x float> %1105, 1, !dbg !7
  %1107 = shufflevector <4 x float> %1073, <4 x float> %1079, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1108 = insertvalue [2 x <8 x float>] poison, <8 x float> %1107, 0, !dbg !7
  %1109 = shufflevector <4 x float> %1084, <4 x float> %1088, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1110 = insertvalue [2 x <8 x float>] %1108, <8 x float> %1109, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1111 = add nuw nsw i64 %58, 1, !dbg !7
  br label %57, !dbg !7

1112:                                             ; preds = %57
  %1113 = extractvalue [2 x <8 x float>] %64, 0, !dbg !7
  %1114 = extractelement <8 x float> %1113, i64 0, !dbg !7
  %1115 = extractelement <8 x float> %1113, i64 1, !dbg !7
  %1116 = extractelement <8 x float> %1113, i64 2, !dbg !7
  %1117 = extractelement <8 x float> %1113, i64 3, !dbg !7
  %1118 = select i1 %.not, float %1115, float %1114, !dbg !7
  %1119 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1120 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1119), !dbg !7
  %1121 = add nuw nsw i32 %1120, 64, !dbg !7
  %1122 = xor i32 %1120, 16, !dbg !7
  %1123 = and i32 %1121, 192, !dbg !7
  %1124 = icmp samesign ult i32 %1122, %1123, !dbg !7
  %1125 = select i1 %1124, i32 %1122, i32 %1120, !dbg !7
  %1126 = shl nuw nsw i32 %1125, 2, !dbg !7
  %1127 = bitcast float %1118 to i32, !dbg !7
  %1128 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1127), !dbg !7
  %1129 = bitcast i32 %1128 to float, !dbg !7
  %1130 = select i1 %.not, float %1114, float %1129, !dbg !7
  %1131 = select i1 %.not, float %1129, float %1115, !dbg !7
  %1132 = select i1 %.not, float %1117, float %1116, !dbg !7
  %1133 = bitcast float %1132 to i32, !dbg !7
  %1134 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1133), !dbg !7
  %1135 = bitcast i32 %1134 to float, !dbg !7
  %1136 = select i1 %.not, float %1116, float %1135, !dbg !7
  %1137 = select i1 %.not, float %1135, float %1117, !dbg !7
  %1138 = select i1 %.not1, float %1136, float %1130, !dbg !7
  %1139 = xor i32 %1120, 32, !dbg !7
  %1140 = icmp samesign ult i32 %1139, %1123, !dbg !7
  %1141 = select i1 %1140, i32 %1139, i32 %1120, !dbg !7
  %1142 = shl nuw nsw i32 %1141, 2, !dbg !7
  %1143 = bitcast float %1138 to i32, !dbg !7
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1143), !dbg !7
  %1145 = bitcast i32 %1144 to float, !dbg !7
  %1146 = select i1 %.not1, float %1130, float %1145, !dbg !7
  %1147 = select i1 %.not1, float %1145, float %1136, !dbg !7
  %1148 = select i1 %.not1, float %1137, float %1131, !dbg !7
  %1149 = bitcast float %1148 to i32, !dbg !7
  %1150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1149), !dbg !7
  %1151 = bitcast i32 %1150 to float, !dbg !7
  %1152 = select i1 %.not1, float %1131, float %1151, !dbg !7
  %1153 = select i1 %.not1, float %1151, float %1137, !dbg !7
  %1154 = extractvalue [2 x <8 x float>] %63, 0, !dbg !7
  %1155 = extractelement <8 x float> %1154, i64 0, !dbg !7
  %1156 = extractelement <8 x float> %1154, i64 1, !dbg !7
  %1157 = extractelement <8 x float> %1154, i64 2, !dbg !7
  %1158 = extractelement <8 x float> %1154, i64 3, !dbg !7
  %1159 = select i1 %.not, float %1156, float %1155, !dbg !7
  %1160 = bitcast float %1159 to i32, !dbg !7
  %1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1160), !dbg !7
  %1162 = bitcast i32 %1161 to float, !dbg !7
  %1163 = select i1 %.not, float %1155, float %1162, !dbg !7
  %1164 = select i1 %.not, float %1162, float %1156, !dbg !7
  %1165 = select i1 %.not, float %1158, float %1157, !dbg !7
  %1166 = bitcast float %1165 to i32, !dbg !7
  %1167 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1166), !dbg !7
  %1168 = bitcast i32 %1167 to float, !dbg !7
  %1169 = select i1 %.not, float %1157, float %1168, !dbg !7
  %1170 = select i1 %.not, float %1168, float %1158, !dbg !7
  %1171 = select i1 %.not1, float %1169, float %1163, !dbg !7
  %1172 = bitcast float %1171 to i32, !dbg !7
  %1173 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1172), !dbg !7
  %1174 = bitcast i32 %1173 to float, !dbg !7
  %1175 = select i1 %.not1, float %1163, float %1174, !dbg !7
  %1176 = select i1 %.not1, float %1174, float %1169, !dbg !7
  %1177 = select i1 %.not1, float %1170, float %1164, !dbg !7
  %1178 = bitcast float %1177 to i32, !dbg !7
  %1179 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1178), !dbg !7
  %1180 = bitcast i32 %1179 to float, !dbg !7
  %1181 = select i1 %.not1, float %1164, float %1180, !dbg !7
  %1182 = select i1 %.not1, float %1180, float %1170, !dbg !7
  %1183 = extractvalue [2 x <8 x float>] %62, 0, !dbg !7
  %1184 = extractelement <8 x float> %1183, i64 0, !dbg !7
  %1185 = extractelement <8 x float> %1183, i64 1, !dbg !7
  %1186 = extractelement <8 x float> %1183, i64 2, !dbg !7
  %1187 = extractelement <8 x float> %1183, i64 3, !dbg !7
  %1188 = select i1 %.not, float %1185, float %1184, !dbg !7
  %1189 = bitcast float %1188 to i32, !dbg !7
  %1190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1189), !dbg !7
  %1191 = bitcast i32 %1190 to float, !dbg !7
  %1192 = select i1 %.not, float %1184, float %1191, !dbg !7
  %1193 = select i1 %.not, float %1191, float %1185, !dbg !7
  %1194 = select i1 %.not, float %1187, float %1186, !dbg !7
  %1195 = bitcast float %1194 to i32, !dbg !7
  %1196 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1195), !dbg !7
  %1197 = bitcast i32 %1196 to float, !dbg !7
  %1198 = select i1 %.not, float %1186, float %1197, !dbg !7
  %1199 = select i1 %.not, float %1197, float %1187, !dbg !7
  %1200 = select i1 %.not1, float %1198, float %1192, !dbg !7
  %1201 = bitcast float %1200 to i32, !dbg !7
  %1202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1201), !dbg !7
  %1203 = bitcast i32 %1202 to float, !dbg !7
  %1204 = select i1 %.not1, float %1192, float %1203, !dbg !7
  %1205 = select i1 %.not1, float %1203, float %1198, !dbg !7
  %1206 = select i1 %.not1, float %1199, float %1193, !dbg !7
  %1207 = bitcast float %1206 to i32, !dbg !7
  %1208 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1207), !dbg !7
  %1209 = bitcast i32 %1208 to float, !dbg !7
  %1210 = select i1 %.not1, float %1193, float %1209, !dbg !7
  %1211 = select i1 %.not1, float %1209, float %1199, !dbg !7
  %1212 = extractvalue [2 x <8 x float>] %59, 0, !dbg !7
  %1213 = extractelement <8 x float> %1212, i64 0, !dbg !7
  %1214 = extractelement <8 x float> %1212, i64 1, !dbg !7
  %1215 = extractelement <8 x float> %1212, i64 2, !dbg !7
  %1216 = extractelement <8 x float> %1212, i64 3, !dbg !7
  %1217 = select i1 %.not, float %1214, float %1213, !dbg !7
  %1218 = bitcast float %1217 to i32, !dbg !7
  %1219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1218), !dbg !7
  %1220 = bitcast i32 %1219 to float, !dbg !7
  %1221 = select i1 %.not, float %1213, float %1220, !dbg !7
  %1222 = select i1 %.not, float %1220, float %1214, !dbg !7
  %1223 = select i1 %.not, float %1216, float %1215, !dbg !7
  %1224 = bitcast float %1223 to i32, !dbg !7
  %1225 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1224), !dbg !7
  %1226 = bitcast i32 %1225 to float, !dbg !7
  %1227 = select i1 %.not, float %1215, float %1226, !dbg !7
  %1228 = select i1 %.not, float %1226, float %1216, !dbg !7
  %1229 = select i1 %.not1, float %1227, float %1221, !dbg !7
  %1230 = bitcast float %1229 to i32, !dbg !7
  %1231 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1230), !dbg !7
  %1232 = select i1 %.not1, float %1228, float %1222, !dbg !7
  %1233 = bitcast float %1232 to i32, !dbg !7
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1233), !dbg !7
  %1235 = extractvalue [2 x <1 x float>] %60, 0, !dbg !19
  %1236 = shufflevector <1 x float> %1235, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1237 = extractelement <8 x float> %1113, i64 4, !dbg !7
  %1238 = extractelement <8 x float> %1113, i64 5, !dbg !7
  %1239 = extractelement <8 x float> %1113, i64 6, !dbg !7
  %1240 = extractelement <8 x float> %1113, i64 7, !dbg !7
  %1241 = select i1 %.not, float %1238, float %1237, !dbg !7
  %1242 = bitcast float %1241 to i32, !dbg !7
  %1243 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1242), !dbg !7
  %1244 = bitcast i32 %1243 to float, !dbg !7
  %1245 = select i1 %.not, float %1237, float %1244, !dbg !7
  %1246 = select i1 %.not, float %1244, float %1238, !dbg !7
  %1247 = select i1 %.not, float %1240, float %1239, !dbg !7
  %1248 = bitcast float %1247 to i32, !dbg !7
  %1249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1248), !dbg !7
  %1250 = bitcast i32 %1249 to float, !dbg !7
  %1251 = select i1 %.not, float %1239, float %1250, !dbg !7
  %1252 = select i1 %.not, float %1250, float %1240, !dbg !7
  %1253 = select i1 %.not1, float %1251, float %1245, !dbg !7
  %1254 = bitcast float %1253 to i32, !dbg !7
  %1255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1254), !dbg !7
  %1256 = bitcast i32 %1255 to float, !dbg !7
  %1257 = select i1 %.not1, float %1245, float %1256, !dbg !7
  %1258 = select i1 %.not1, float %1256, float %1251, !dbg !7
  %1259 = select i1 %.not1, float %1252, float %1246, !dbg !7
  %1260 = bitcast float %1259 to i32, !dbg !7
  %1261 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1260), !dbg !7
  %1262 = bitcast i32 %1261 to float, !dbg !7
  %1263 = select i1 %.not1, float %1246, float %1262, !dbg !7
  %1264 = select i1 %.not1, float %1262, float %1252, !dbg !7
  %1265 = extractelement <8 x float> %1154, i64 4, !dbg !7
  %1266 = extractelement <8 x float> %1154, i64 5, !dbg !7
  %1267 = extractelement <8 x float> %1154, i64 6, !dbg !7
  %1268 = extractelement <8 x float> %1154, i64 7, !dbg !7
  %1269 = select i1 %.not, float %1266, float %1265, !dbg !7
  %1270 = bitcast float %1269 to i32, !dbg !7
  %1271 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1270), !dbg !7
  %1272 = bitcast i32 %1271 to float, !dbg !7
  %1273 = select i1 %.not, float %1265, float %1272, !dbg !7
  %1274 = select i1 %.not, float %1272, float %1266, !dbg !7
  %1275 = select i1 %.not, float %1268, float %1267, !dbg !7
  %1276 = bitcast float %1275 to i32, !dbg !7
  %1277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1276), !dbg !7
  %1278 = bitcast i32 %1277 to float, !dbg !7
  %1279 = select i1 %.not, float %1267, float %1278, !dbg !7
  %1280 = select i1 %.not, float %1278, float %1268, !dbg !7
  %1281 = select i1 %.not1, float %1279, float %1273, !dbg !7
  %1282 = bitcast float %1281 to i32, !dbg !7
  %1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1282), !dbg !7
  %1284 = bitcast i32 %1283 to float, !dbg !7
  %1285 = select i1 %.not1, float %1273, float %1284, !dbg !7
  %1286 = select i1 %.not1, float %1284, float %1279, !dbg !7
  %1287 = select i1 %.not1, float %1280, float %1274, !dbg !7
  %1288 = bitcast float %1287 to i32, !dbg !7
  %1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1288), !dbg !7
  %1290 = bitcast i32 %1289 to float, !dbg !7
  %1291 = select i1 %.not1, float %1274, float %1290, !dbg !7
  %1292 = select i1 %.not1, float %1290, float %1280, !dbg !7
  %1293 = extractelement <8 x float> %1183, i64 4, !dbg !7
  %1294 = extractelement <8 x float> %1183, i64 5, !dbg !7
  %1295 = extractelement <8 x float> %1183, i64 6, !dbg !7
  %1296 = extractelement <8 x float> %1183, i64 7, !dbg !7
  %1297 = select i1 %.not, float %1294, float %1293, !dbg !7
  %1298 = bitcast float %1297 to i32, !dbg !7
  %1299 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1298), !dbg !7
  %1300 = bitcast i32 %1299 to float, !dbg !7
  %1301 = select i1 %.not, float %1293, float %1300, !dbg !7
  %1302 = select i1 %.not, float %1300, float %1294, !dbg !7
  %1303 = select i1 %.not, float %1296, float %1295, !dbg !7
  %1304 = bitcast float %1303 to i32, !dbg !7
  %1305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1304), !dbg !7
  %1306 = bitcast i32 %1305 to float, !dbg !7
  %1307 = select i1 %.not, float %1295, float %1306, !dbg !7
  %1308 = select i1 %.not, float %1306, float %1296, !dbg !7
  %1309 = select i1 %.not1, float %1307, float %1301, !dbg !7
  %1310 = bitcast float %1309 to i32, !dbg !7
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1310), !dbg !7
  %1312 = bitcast i32 %1311 to float, !dbg !7
  %1313 = select i1 %.not1, float %1301, float %1312, !dbg !7
  %1314 = select i1 %.not1, float %1312, float %1307, !dbg !7
  %1315 = select i1 %.not1, float %1308, float %1302, !dbg !7
  %1316 = bitcast float %1315 to i32, !dbg !7
  %1317 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1316), !dbg !7
  %1318 = bitcast i32 %1317 to float, !dbg !7
  %1319 = select i1 %.not1, float %1302, float %1318, !dbg !7
  %1320 = select i1 %.not1, float %1318, float %1308, !dbg !7
  %1321 = extractelement <8 x float> %1212, i64 4, !dbg !7
  %1322 = extractelement <8 x float> %1212, i64 5, !dbg !7
  %1323 = extractelement <8 x float> %1212, i64 6, !dbg !7
  %1324 = extractelement <8 x float> %1212, i64 7, !dbg !7
  %1325 = select i1 %.not, float %1322, float %1321, !dbg !7
  %1326 = bitcast float %1325 to i32, !dbg !7
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1326), !dbg !7
  %1328 = bitcast i32 %1327 to float, !dbg !7
  %1329 = select i1 %.not, float %1321, float %1328, !dbg !7
  %1330 = select i1 %.not, float %1328, float %1322, !dbg !7
  %1331 = select i1 %.not, float %1324, float %1323, !dbg !7
  %1332 = bitcast float %1331 to i32, !dbg !7
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1332), !dbg !7
  %1334 = bitcast i32 %1333 to float, !dbg !7
  %1335 = select i1 %.not, float %1323, float %1334, !dbg !7
  %1336 = select i1 %.not, float %1334, float %1324, !dbg !7
  %1337 = select i1 %.not1, float %1335, float %1329, !dbg !7
  %1338 = bitcast float %1337 to i32, !dbg !7
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1338), !dbg !7
  %1340 = select i1 %.not1, float %1336, float %1330, !dbg !7
  %1341 = bitcast float %1340 to i32, !dbg !7
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1341), !dbg !7
  %1343 = extractvalue [2 x <8 x float>] %64, 1, !dbg !7
  %1344 = extractelement <8 x float> %1343, i64 0, !dbg !7
  %1345 = extractelement <8 x float> %1343, i64 1, !dbg !7
  %1346 = extractelement <8 x float> %1343, i64 2, !dbg !7
  %1347 = extractelement <8 x float> %1343, i64 3, !dbg !7
  %1348 = select i1 %.not, float %1345, float %1344, !dbg !7
  %1349 = bitcast float %1348 to i32, !dbg !7
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1349), !dbg !7
  %1351 = bitcast i32 %1350 to float, !dbg !7
  %1352 = select i1 %.not, float %1344, float %1351, !dbg !7
  %1353 = select i1 %.not, float %1351, float %1345, !dbg !7
  %1354 = select i1 %.not, float %1347, float %1346, !dbg !7
  %1355 = bitcast float %1354 to i32, !dbg !7
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1355), !dbg !7
  %1357 = bitcast i32 %1356 to float, !dbg !7
  %1358 = select i1 %.not, float %1346, float %1357, !dbg !7
  %1359 = select i1 %.not, float %1357, float %1347, !dbg !7
  %1360 = select i1 %.not1, float %1358, float %1352, !dbg !7
  %1361 = bitcast float %1360 to i32, !dbg !7
  %1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1361), !dbg !7
  %1363 = bitcast i32 %1362 to float, !dbg !7
  %1364 = select i1 %.not1, float %1352, float %1363, !dbg !7
  %1365 = select i1 %.not1, float %1363, float %1358, !dbg !7
  %1366 = select i1 %.not1, float %1359, float %1353, !dbg !7
  %1367 = bitcast float %1366 to i32, !dbg !7
  %1368 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1367), !dbg !7
  %1369 = bitcast i32 %1368 to float, !dbg !7
  %1370 = select i1 %.not1, float %1353, float %1369, !dbg !7
  %1371 = select i1 %.not1, float %1369, float %1359, !dbg !7
  %1372 = extractvalue [2 x <8 x float>] %63, 1, !dbg !7
  %1373 = extractelement <8 x float> %1372, i64 0, !dbg !7
  %1374 = extractelement <8 x float> %1372, i64 1, !dbg !7
  %1375 = extractelement <8 x float> %1372, i64 2, !dbg !7
  %1376 = extractelement <8 x float> %1372, i64 3, !dbg !7
  %1377 = select i1 %.not, float %1374, float %1373, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1378), !dbg !7
  %1380 = bitcast i32 %1379 to float, !dbg !7
  %1381 = select i1 %.not, float %1373, float %1380, !dbg !7
  %1382 = select i1 %.not, float %1380, float %1374, !dbg !7
  %1383 = select i1 %.not, float %1376, float %1375, !dbg !7
  %1384 = bitcast float %1383 to i32, !dbg !7
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1384), !dbg !7
  %1386 = bitcast i32 %1385 to float, !dbg !7
  %1387 = select i1 %.not, float %1375, float %1386, !dbg !7
  %1388 = select i1 %.not, float %1386, float %1376, !dbg !7
  %1389 = select i1 %.not1, float %1387, float %1381, !dbg !7
  %1390 = bitcast float %1389 to i32, !dbg !7
  %1391 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1390), !dbg !7
  %1392 = bitcast i32 %1391 to float, !dbg !7
  %1393 = select i1 %.not1, float %1381, float %1392, !dbg !7
  %1394 = select i1 %.not1, float %1392, float %1387, !dbg !7
  %1395 = select i1 %.not1, float %1388, float %1382, !dbg !7
  %1396 = bitcast float %1395 to i32, !dbg !7
  %1397 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1396), !dbg !7
  %1398 = bitcast i32 %1397 to float, !dbg !7
  %1399 = select i1 %.not1, float %1382, float %1398, !dbg !7
  %1400 = select i1 %.not1, float %1398, float %1388, !dbg !7
  %1401 = extractvalue [2 x <8 x float>] %62, 1, !dbg !7
  %1402 = extractelement <8 x float> %1401, i64 0, !dbg !7
  %1403 = extractelement <8 x float> %1401, i64 1, !dbg !7
  %1404 = extractelement <8 x float> %1401, i64 2, !dbg !7
  %1405 = extractelement <8 x float> %1401, i64 3, !dbg !7
  %1406 = select i1 %.not, float %1403, float %1402, !dbg !7
  %1407 = bitcast float %1406 to i32, !dbg !7
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1407), !dbg !7
  %1409 = bitcast i32 %1408 to float, !dbg !7
  %1410 = select i1 %.not, float %1402, float %1409, !dbg !7
  %1411 = select i1 %.not, float %1409, float %1403, !dbg !7
  %1412 = select i1 %.not, float %1405, float %1404, !dbg !7
  %1413 = bitcast float %1412 to i32, !dbg !7
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1413), !dbg !7
  %1415 = bitcast i32 %1414 to float, !dbg !7
  %1416 = select i1 %.not, float %1404, float %1415, !dbg !7
  %1417 = select i1 %.not, float %1415, float %1405, !dbg !7
  %1418 = select i1 %.not1, float %1416, float %1410, !dbg !7
  %1419 = bitcast float %1418 to i32, !dbg !7
  %1420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1419), !dbg !7
  %1421 = bitcast i32 %1420 to float, !dbg !7
  %1422 = select i1 %.not1, float %1410, float %1421, !dbg !7
  %1423 = select i1 %.not1, float %1421, float %1416, !dbg !7
  %1424 = select i1 %.not1, float %1417, float %1411, !dbg !7
  %1425 = bitcast float %1424 to i32, !dbg !7
  %1426 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1425), !dbg !7
  %1427 = bitcast i32 %1426 to float, !dbg !7
  %1428 = select i1 %.not1, float %1411, float %1427, !dbg !7
  %1429 = select i1 %.not1, float %1427, float %1417, !dbg !7
  %1430 = extractvalue [2 x <8 x float>] %59, 1, !dbg !7
  %1431 = extractelement <8 x float> %1430, i64 0, !dbg !7
  %1432 = extractelement <8 x float> %1430, i64 1, !dbg !7
  %1433 = extractelement <8 x float> %1430, i64 2, !dbg !7
  %1434 = extractelement <8 x float> %1430, i64 3, !dbg !7
  %1435 = select i1 %.not, float %1432, float %1431, !dbg !7
  %1436 = bitcast float %1435 to i32, !dbg !7
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1436), !dbg !7
  %1438 = bitcast i32 %1437 to float, !dbg !7
  %1439 = select i1 %.not, float %1431, float %1438, !dbg !7
  %1440 = select i1 %.not, float %1438, float %1432, !dbg !7
  %1441 = select i1 %.not, float %1434, float %1433, !dbg !7
  %1442 = bitcast float %1441 to i32, !dbg !7
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1442), !dbg !7
  %1444 = bitcast i32 %1443 to float, !dbg !7
  %1445 = select i1 %.not, float %1433, float %1444, !dbg !7
  %1446 = select i1 %.not, float %1444, float %1434, !dbg !7
  %1447 = select i1 %.not1, float %1445, float %1439, !dbg !7
  %1448 = bitcast float %1447 to i32, !dbg !7
  %1449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1448), !dbg !7
  %1450 = select i1 %.not1, float %1446, float %1440, !dbg !7
  %1451 = bitcast float %1450 to i32, !dbg !7
  %1452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1451), !dbg !7
  %1453 = extractvalue [2 x <1 x float>] %60, 1, !dbg !19
  %1454 = shufflevector <1 x float> %1453, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1455 = extractelement <8 x float> %1343, i64 4, !dbg !7
  %1456 = extractelement <8 x float> %1343, i64 5, !dbg !7
  %1457 = extractelement <8 x float> %1343, i64 6, !dbg !7
  %1458 = extractelement <8 x float> %1343, i64 7, !dbg !7
  %1459 = select i1 %.not, float %1456, float %1455, !dbg !7
  %1460 = bitcast float %1459 to i32, !dbg !7
  %1461 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1460), !dbg !7
  %1462 = bitcast i32 %1461 to float, !dbg !7
  %1463 = select i1 %.not, float %1455, float %1462, !dbg !7
  %1464 = select i1 %.not, float %1462, float %1456, !dbg !7
  %1465 = select i1 %.not, float %1458, float %1457, !dbg !7
  %1466 = bitcast float %1465 to i32, !dbg !7
  %1467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1466), !dbg !7
  %1468 = bitcast i32 %1467 to float, !dbg !7
  %1469 = select i1 %.not, float %1457, float %1468, !dbg !7
  %1470 = select i1 %.not, float %1468, float %1458, !dbg !7
  %1471 = select i1 %.not1, float %1469, float %1463, !dbg !7
  %1472 = bitcast float %1471 to i32, !dbg !7
  %1473 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1472), !dbg !7
  %1474 = bitcast i32 %1473 to float, !dbg !7
  %1475 = select i1 %.not1, float %1463, float %1474, !dbg !7
  %1476 = select i1 %.not1, float %1474, float %1469, !dbg !7
  %1477 = select i1 %.not1, float %1470, float %1464, !dbg !7
  %1478 = bitcast float %1477 to i32, !dbg !7
  %1479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1478), !dbg !7
  %1480 = bitcast i32 %1479 to float, !dbg !7
  %1481 = select i1 %.not1, float %1464, float %1480, !dbg !7
  %1482 = select i1 %.not1, float %1480, float %1470, !dbg !7
  %1483 = extractelement <8 x float> %1372, i64 4, !dbg !7
  %1484 = extractelement <8 x float> %1372, i64 5, !dbg !7
  %1485 = extractelement <8 x float> %1372, i64 6, !dbg !7
  %1486 = extractelement <8 x float> %1372, i64 7, !dbg !7
  %1487 = select i1 %.not, float %1484, float %1483, !dbg !7
  %1488 = bitcast float %1487 to i32, !dbg !7
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1488), !dbg !7
  %1490 = bitcast i32 %1489 to float, !dbg !7
  %1491 = select i1 %.not, float %1483, float %1490, !dbg !7
  %1492 = select i1 %.not, float %1490, float %1484, !dbg !7
  %1493 = select i1 %.not, float %1486, float %1485, !dbg !7
  %1494 = bitcast float %1493 to i32, !dbg !7
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1494), !dbg !7
  %1496 = bitcast i32 %1495 to float, !dbg !7
  %1497 = select i1 %.not, float %1485, float %1496, !dbg !7
  %1498 = select i1 %.not, float %1496, float %1486, !dbg !7
  %1499 = select i1 %.not1, float %1497, float %1491, !dbg !7
  %1500 = bitcast float %1499 to i32, !dbg !7
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1500), !dbg !7
  %1502 = bitcast i32 %1501 to float, !dbg !7
  %1503 = select i1 %.not1, float %1491, float %1502, !dbg !7
  %1504 = select i1 %.not1, float %1502, float %1497, !dbg !7
  %1505 = select i1 %.not1, float %1498, float %1492, !dbg !7
  %1506 = bitcast float %1505 to i32, !dbg !7
  %1507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1506), !dbg !7
  %1508 = bitcast i32 %1507 to float, !dbg !7
  %1509 = select i1 %.not1, float %1492, float %1508, !dbg !7
  %1510 = select i1 %.not1, float %1508, float %1498, !dbg !7
  %1511 = extractelement <8 x float> %1401, i64 4, !dbg !7
  %1512 = extractelement <8 x float> %1401, i64 5, !dbg !7
  %1513 = extractelement <8 x float> %1401, i64 6, !dbg !7
  %1514 = extractelement <8 x float> %1401, i64 7, !dbg !7
  %1515 = select i1 %.not, float %1512, float %1511, !dbg !7
  %1516 = bitcast float %1515 to i32, !dbg !7
  %1517 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1516), !dbg !7
  %1518 = bitcast i32 %1517 to float, !dbg !7
  %1519 = select i1 %.not, float %1511, float %1518, !dbg !7
  %1520 = select i1 %.not, float %1518, float %1512, !dbg !7
  %1521 = select i1 %.not, float %1514, float %1513, !dbg !7
  %1522 = bitcast float %1521 to i32, !dbg !7
  %1523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1522), !dbg !7
  %1524 = bitcast i32 %1523 to float, !dbg !7
  %1525 = select i1 %.not, float %1513, float %1524, !dbg !7
  %1526 = select i1 %.not, float %1524, float %1514, !dbg !7
  %1527 = select i1 %.not1, float %1525, float %1519, !dbg !7
  %1528 = bitcast float %1527 to i32, !dbg !7
  %1529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1528), !dbg !7
  %1530 = bitcast i32 %1529 to float, !dbg !7
  %1531 = select i1 %.not1, float %1519, float %1530, !dbg !7
  %1532 = select i1 %.not1, float %1530, float %1525, !dbg !7
  %1533 = select i1 %.not1, float %1526, float %1520, !dbg !7
  %1534 = bitcast float %1533 to i32, !dbg !7
  %1535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1534), !dbg !7
  %1536 = bitcast i32 %1535 to float, !dbg !7
  %1537 = select i1 %.not1, float %1520, float %1536, !dbg !7
  %1538 = select i1 %.not1, float %1536, float %1526, !dbg !7
  %1539 = extractelement <8 x float> %1430, i64 4, !dbg !7
  %1540 = extractelement <8 x float> %1430, i64 5, !dbg !7
  %1541 = extractelement <8 x float> %1430, i64 6, !dbg !7
  %1542 = extractelement <8 x float> %1430, i64 7, !dbg !7
  %1543 = select i1 %.not, float %1540, float %1539, !dbg !7
  %1544 = bitcast float %1543 to i32, !dbg !7
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1544), !dbg !7
  %1546 = bitcast i32 %1545 to float, !dbg !7
  %1547 = select i1 %.not, float %1539, float %1546, !dbg !7
  %1548 = select i1 %.not, float %1546, float %1540, !dbg !7
  %1549 = select i1 %.not, float %1542, float %1541, !dbg !7
  %1550 = bitcast float %1549 to i32, !dbg !7
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1550), !dbg !7
  %1552 = bitcast i32 %1551 to float, !dbg !7
  %1553 = select i1 %.not, float %1541, float %1552, !dbg !7
  %1554 = select i1 %.not, float %1552, float %1542, !dbg !7
  %1555 = select i1 %.not1, float %1553, float %1547, !dbg !7
  %1556 = bitcast float %1555 to i32, !dbg !7
  %1557 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1556), !dbg !7
  %1558 = select i1 %.not1, float %1554, float %1548, !dbg !7
  %1559 = bitcast float %1558 to i32, !dbg !7
  %1560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1559), !dbg !7
  br i1 true, label %1561, label %1625, !dbg !21

1561:                                             ; preds = %1112
  %1562 = bitcast i32 %1557 to float, !dbg !7
  %1563 = select i1 %.not1, float %1547, float %1562, !dbg !7
  %1564 = insertelement <4 x float> poison, float %1563, i64 0, !dbg !19
  %1565 = bitcast i32 %1560 to float, !dbg !7
  %1566 = select i1 %.not1, float %1548, float %1565, !dbg !7
  %1567 = insertelement <4 x float> %1564, float %1566, i64 1, !dbg !19
  %1568 = select i1 %.not1, float %1562, float %1553, !dbg !7
  %1569 = insertelement <4 x float> %1567, float %1568, i64 2, !dbg !19
  %1570 = select i1 %.not1, float %1565, float %1554, !dbg !7
  %1571 = insertelement <4 x float> %1569, float %1570, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1454, !dbg !19
  %normalized = fmul <4 x float> %1571, %row.reciprocal, !dbg !19
  %1572 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1573 = extractelement <4 x half> %1572, i64 3, !dbg !20
  %1574 = extractelement <4 x half> %1572, i64 2, !dbg !20
  %1575 = extractelement <4 x half> %1572, i64 1, !dbg !20
  %1576 = extractelement <4 x half> %1572, i64 0, !dbg !20
  %1577 = bitcast i32 %1449 to float, !dbg !7
  %1578 = select i1 %.not1, float %1439, float %1577, !dbg !7
  %1579 = insertelement <4 x float> poison, float %1578, i64 0, !dbg !19
  %1580 = bitcast i32 %1452 to float, !dbg !7
  %1581 = select i1 %.not1, float %1440, float %1580, !dbg !7
  %1582 = insertelement <4 x float> %1579, float %1581, i64 1, !dbg !19
  %1583 = select i1 %.not1, float %1577, float %1445, !dbg !7
  %1584 = insertelement <4 x float> %1582, float %1583, i64 2, !dbg !19
  %1585 = select i1 %.not1, float %1580, float %1446, !dbg !7
  %1586 = insertelement <4 x float> %1584, float %1585, i64 3, !dbg !19
  %normalized20 = fmul <4 x float> %1586, %row.reciprocal, !dbg !19
  %1587 = fptrunc <4 x float> %normalized20 to <4 x half>, !dbg !20
  %1588 = extractelement <4 x half> %1587, i64 3, !dbg !20
  %1589 = extractelement <4 x half> %1587, i64 2, !dbg !20
  %1590 = extractelement <4 x half> %1587, i64 1, !dbg !20
  %1591 = extractelement <4 x half> %1587, i64 0, !dbg !20
  %1592 = bitcast i32 %1339 to float, !dbg !7
  %1593 = select i1 %.not1, float %1329, float %1592, !dbg !7
  %1594 = insertelement <4 x float> poison, float %1593, i64 0, !dbg !19
  %1595 = bitcast i32 %1342 to float, !dbg !7
  %1596 = select i1 %.not1, float %1330, float %1595, !dbg !7
  %1597 = insertelement <4 x float> %1594, float %1596, i64 1, !dbg !19
  %1598 = select i1 %.not1, float %1592, float %1335, !dbg !7
  %1599 = insertelement <4 x float> %1597, float %1598, i64 2, !dbg !19
  %1600 = select i1 %.not1, float %1595, float %1336, !dbg !7
  %1601 = insertelement <4 x float> %1599, float %1600, i64 3, !dbg !19
  %row.reciprocal21 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1236, !dbg !19
  %normalized22 = fmul <4 x float> %1601, %row.reciprocal21, !dbg !19
  %1602 = fptrunc <4 x float> %normalized22 to <4 x half>, !dbg !20
  %1603 = extractelement <4 x half> %1602, i64 3, !dbg !20
  %1604 = extractelement <4 x half> %1602, i64 2, !dbg !20
  %1605 = extractelement <4 x half> %1602, i64 1, !dbg !20
  %1606 = extractelement <4 x half> %1602, i64 0, !dbg !20
  %1607 = bitcast i32 %1231 to float, !dbg !7
  %1608 = select i1 %.not1, float %1221, float %1607, !dbg !7
  %1609 = insertelement <4 x float> poison, float %1608, i64 0, !dbg !19
  %1610 = bitcast i32 %1234 to float, !dbg !7
  %1611 = select i1 %.not1, float %1222, float %1610, !dbg !7
  %1612 = insertelement <4 x float> %1609, float %1611, i64 1, !dbg !19
  %1613 = select i1 %.not1, float %1607, float %1227, !dbg !7
  %1614 = insertelement <4 x float> %1612, float %1613, i64 2, !dbg !19
  %1615 = select i1 %.not1, float %1610, float %1228, !dbg !7
  %1616 = insertelement <4 x float> %1614, float %1615, i64 3, !dbg !19
  %normalized23 = fmul <4 x float> %1616, %row.reciprocal21, !dbg !19
  %1617 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !20
  %1618 = extractelement <4 x half> %1617, i64 3, !dbg !20
  %1619 = extractelement <4 x half> %1617, i64 2, !dbg !20
  %1620 = extractelement <4 x half> %1617, i64 1, !dbg !20
  %1621 = extractelement <4 x half> %1617, i64 0, !dbg !20
  %1622 = add nsw i64 %15, %25, !dbg !21
  %1623 = shl i64 %1622, 19, !dbg !21
  %1624 = and i64 %1623, 16252928, !dbg !21
  %address.thread332 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local333 = zext nneg i32 %address.thread332 to i64, !dbg !12
  %address.local334 = add i64 %28, %address.local333, !dbg !21
  %address.local335 = lshr i64 %address.local333, 6, !dbg !12
  %address.local336 = shl nuw nsw i64 %address.local335, 4, !dbg !21
  %address.local337 = add i64 %address.local334, %address.local336, !dbg !21
  %address.local338 = and i64 %address.local333, 240, !dbg !21
  %address.local339 = sub i64 %address.local337, %address.local338, !dbg !21
  %address.local340 = add i64 %address.local339, %29, !dbg !21
  %address.local341 = shl nuw nsw i64 %address.local340, 7, !dbg !21
  %address.local342 = add nuw nsw i64 %1624, %address.local341, !dbg !21
  %address.local343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local342, !dbg !21
  %address.local344 = lshr i64 %address.local333, 2, !dbg !12
  %address.local345 = and i64 %address.local344, 12, !dbg !12
  %address.local346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local343, i64 %address.local345, !dbg !21
  store half %1621, ptr addrspace(1) %address.local346, align 2, !dbg !21
  %address.local347 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local342, !dbg !21
  %address.local348 = or disjoint i64 %address.local345, 1, !dbg !21
  %address.local349 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local347, i64 %address.local348, !dbg !21
  store half %1620, ptr addrspace(1) %address.local349, align 2, !dbg !21
  %address.local350 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local342, !dbg !21
  %address.local351 = or disjoint i64 %address.local345, 2, !dbg !21
  %address.local352 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local350, i64 %address.local351, !dbg !21
  store half %1619, ptr addrspace(1) %address.local352, align 2, !dbg !21
  %address.local353 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local342, !dbg !21
  %address.local354 = or disjoint i64 %address.local345, 3, !dbg !21
  %address.local355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local353, i64 %address.local354, !dbg !21
  store half %1618, ptr addrspace(1) %address.local355, align 2, !dbg !21
  %address.local356 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local342, !dbg !21
  %address.local357 = or disjoint i64 %address.local345, 16, !dbg !21
  %address.local358 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local356, i64 %address.local357, !dbg !21
  store half %1606, ptr addrspace(1) %address.local358, align 2, !dbg !21
  %address.local359 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local342, !dbg !21
  %address.local360 = or disjoint i64 %address.local345, 17, !dbg !21
  %address.local361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local359, i64 %address.local360, !dbg !21
  store half %1605, ptr addrspace(1) %address.local361, align 2, !dbg !21
  %address.local362 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local342, !dbg !21
  %address.local363 = or disjoint i64 %address.local345, 18, !dbg !21
  %address.local364 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local362, i64 %address.local363, !dbg !21
  store half %1604, ptr addrspace(1) %address.local364, align 2, !dbg !21
  %address.local365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local342, !dbg !21
  %address.local366 = or disjoint i64 %address.local345, 19, !dbg !21
  %address.local367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local365, i64 %address.local366, !dbg !21
  store half %1603, ptr addrspace(1) %address.local367, align 2, !dbg !21
  %address.local368 = shl i64 %address.local340, 7, !dbg !21
  %address.local369 = add i64 %address.local368, 8192, !dbg !21
  %address.local370 = add nuw nsw i64 %1624, %address.local369, !dbg !21
  %address.local371 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local370, !dbg !21
  %address.local372 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local371, i64 %address.local345, !dbg !21
  store half %1591, ptr addrspace(1) %address.local372, align 2, !dbg !21
  %address.local373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local370, !dbg !21
  %address.local374 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local373, i64 %address.local348, !dbg !21
  store half %1590, ptr addrspace(1) %address.local374, align 2, !dbg !21
  %address.local375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local370, !dbg !21
  %address.local376 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local375, i64 %address.local351, !dbg !21
  store half %1589, ptr addrspace(1) %address.local376, align 2, !dbg !21
  %address.local377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local370, !dbg !21
  %address.local378 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local377, i64 %address.local354, !dbg !21
  store half %1588, ptr addrspace(1) %address.local378, align 2, !dbg !21
  %address.local379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local370, !dbg !21
  %address.local380 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local379, i64 %address.local357, !dbg !21
  store half %1576, ptr addrspace(1) %address.local380, align 2, !dbg !21
  %address.local381 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local370, !dbg !21
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local381, i64 %address.local360, !dbg !21
  store half %1575, ptr addrspace(1) %address.local382, align 2, !dbg !21
  %address.local383 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local370, !dbg !21
  %address.local384 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local383, i64 %address.local363, !dbg !21
  store half %1574, ptr addrspace(1) %address.local384, align 2, !dbg !21
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local370, !dbg !21
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local385, i64 %address.local366, !dbg !21
  store half %1573, ptr addrspace(1) %address.local386, align 2, !dbg !21
  br label %1625, !dbg !21

1625:                                             ; preds = %1561, %1112
  br i1 true, label %1626, label %1664, !dbg !21

1626:                                             ; preds = %1625
  %1627 = insertelement <4 x float> poison, float %1531, i64 0, !dbg !19
  %1628 = insertelement <4 x float> %1627, float %1537, i64 1, !dbg !19
  %1629 = insertelement <4 x float> %1628, float %1532, i64 2, !dbg !19
  %1630 = insertelement <4 x float> %1629, float %1538, i64 3, !dbg !19
  %row.reciprocal24 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1454, !dbg !19
  %normalized25 = fmul <4 x float> %1630, %row.reciprocal24, !dbg !19
  %1631 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !20
  %1632 = extractelement <4 x half> %1631, i64 3, !dbg !20
  %1633 = extractelement <4 x half> %1631, i64 2, !dbg !20
  %1634 = extractelement <4 x half> %1631, i64 1, !dbg !20
  %1635 = extractelement <4 x half> %1631, i64 0, !dbg !20
  %1636 = insertelement <4 x float> poison, float %1422, i64 0, !dbg !19
  %1637 = insertelement <4 x float> %1636, float %1428, i64 1, !dbg !19
  %1638 = insertelement <4 x float> %1637, float %1423, i64 2, !dbg !19
  %1639 = insertelement <4 x float> %1638, float %1429, i64 3, !dbg !19
  %normalized26 = fmul <4 x float> %1639, %row.reciprocal24, !dbg !19
  %1640 = fptrunc <4 x float> %normalized26 to <4 x half>, !dbg !20
  %1641 = extractelement <4 x half> %1640, i64 3, !dbg !20
  %1642 = extractelement <4 x half> %1640, i64 2, !dbg !20
  %1643 = extractelement <4 x half> %1640, i64 1, !dbg !20
  %1644 = extractelement <4 x half> %1640, i64 0, !dbg !20
  %1645 = insertelement <4 x float> poison, float %1313, i64 0, !dbg !19
  %1646 = insertelement <4 x float> %1645, float %1319, i64 1, !dbg !19
  %1647 = insertelement <4 x float> %1646, float %1314, i64 2, !dbg !19
  %1648 = insertelement <4 x float> %1647, float %1320, i64 3, !dbg !19
  %row.reciprocal27 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1236, !dbg !19
  %normalized28 = fmul <4 x float> %1648, %row.reciprocal27, !dbg !19
  %1649 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !20
  %1650 = extractelement <4 x half> %1649, i64 3, !dbg !20
  %1651 = extractelement <4 x half> %1649, i64 2, !dbg !20
  %1652 = extractelement <4 x half> %1649, i64 1, !dbg !20
  %1653 = extractelement <4 x half> %1649, i64 0, !dbg !20
  %1654 = insertelement <4 x float> poison, float %1204, i64 0, !dbg !19
  %1655 = insertelement <4 x float> %1654, float %1210, i64 1, !dbg !19
  %1656 = insertelement <4 x float> %1655, float %1205, i64 2, !dbg !19
  %1657 = insertelement <4 x float> %1656, float %1211, i64 3, !dbg !19
  %normalized29 = fmul <4 x float> %1657, %row.reciprocal27, !dbg !19
  %1658 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !20
  %1659 = extractelement <4 x half> %1658, i64 3, !dbg !20
  %1660 = extractelement <4 x half> %1658, i64 2, !dbg !20
  %1661 = extractelement <4 x half> %1658, i64 1, !dbg !20
  %1662 = extractelement <4 x half> %1658, i64 0, !dbg !20
  %1663 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread387 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local388 = zext nneg i32 %address.thread387 to i64, !dbg !12
  %address.local389 = add i64 %28, %address.local388, !dbg !21
  %address.local390 = lshr i64 %address.local388, 6, !dbg !12
  %address.local391 = shl nuw nsw i64 %address.local390, 4, !dbg !21
  %address.local392 = add i64 %address.local389, %address.local391, !dbg !21
  %address.local393 = and i64 %address.local388, 240, !dbg !21
  %address.local394 = sub i64 %address.local392, %address.local393, !dbg !21
  %address.local395 = add i64 %address.local394, %29, !dbg !21
  %address.local396 = shl nuw nsw i64 %address.local395, 7, !dbg !21
  %address.local397 = add nuw nsw i64 %1663, %address.local396, !dbg !21
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local397, !dbg !21
  %address.local399 = lshr i64 %address.local388, 2, !dbg !12
  %address.local400 = and i64 %address.local399, 12, !dbg !12
  %address.local401 = or disjoint i64 %address.local400, 32, !dbg !21
  %address.local402 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local398, i64 %address.local401, !dbg !21
  store half %1662, ptr addrspace(1) %address.local402, align 2, !dbg !21
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local397, !dbg !21
  %address.local404 = or disjoint i64 %address.local400, 33, !dbg !21
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local403, i64 %address.local404, !dbg !21
  store half %1661, ptr addrspace(1) %address.local405, align 2, !dbg !21
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local397, !dbg !21
  %address.local407 = or disjoint i64 %address.local400, 34, !dbg !21
  %address.local408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local406, i64 %address.local407, !dbg !21
  store half %1660, ptr addrspace(1) %address.local408, align 2, !dbg !21
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local397, !dbg !21
  %address.local410 = or disjoint i64 %address.local400, 35, !dbg !21
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local409, i64 %address.local410, !dbg !21
  store half %1659, ptr addrspace(1) %address.local411, align 2, !dbg !21
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local397, !dbg !21
  %address.local413 = or disjoint i64 %address.local400, 48, !dbg !21
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local412, i64 %address.local413, !dbg !21
  store half %1653, ptr addrspace(1) %address.local414, align 2, !dbg !21
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local397, !dbg !21
  %address.local416 = or disjoint i64 %address.local400, 49, !dbg !21
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local415, i64 %address.local416, !dbg !21
  store half %1652, ptr addrspace(1) %address.local417, align 2, !dbg !21
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local397, !dbg !21
  %address.local419 = or disjoint i64 %address.local400, 50, !dbg !21
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local418, i64 %address.local419, !dbg !21
  store half %1651, ptr addrspace(1) %address.local420, align 2, !dbg !21
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local397, !dbg !21
  %address.local422 = or i64 %address.local399, 51, !dbg !21
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local421, i64 %address.local422, !dbg !21
  store half %1650, ptr addrspace(1) %address.local423, align 2, !dbg !21
  %address.local424 = shl i64 %address.local395, 7, !dbg !21
  %address.local425 = add i64 %address.local424, 8192, !dbg !21
  %address.local426 = add nuw nsw i64 %1663, %address.local425, !dbg !21
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local426, !dbg !21
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local427, i64 %address.local401, !dbg !21
  store half %1644, ptr addrspace(1) %address.local428, align 2, !dbg !21
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local426, !dbg !21
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local429, i64 %address.local404, !dbg !21
  store half %1643, ptr addrspace(1) %address.local430, align 2, !dbg !21
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local426, !dbg !21
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local431, i64 %address.local407, !dbg !21
  store half %1642, ptr addrspace(1) %address.local432, align 2, !dbg !21
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local426, !dbg !21
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local410, !dbg !21
  store half %1641, ptr addrspace(1) %address.local434, align 2, !dbg !21
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local426, !dbg !21
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local413, !dbg !21
  store half %1635, ptr addrspace(1) %address.local436, align 2, !dbg !21
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local426, !dbg !21
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local437, i64 %address.local416, !dbg !21
  store half %1634, ptr addrspace(1) %address.local438, align 2, !dbg !21
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local426, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local419, !dbg !21
  store half %1633, ptr addrspace(1) %address.local440, align 2, !dbg !21
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local426, !dbg !21
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local441, i64 %address.local422, !dbg !21
  store half %1632, ptr addrspace(1) %address.local442, align 2, !dbg !21
  br label %1664, !dbg !21

1664:                                             ; preds = %1626, %1625
  br i1 true, label %1665, label %1703, !dbg !21

1665:                                             ; preds = %1664
  %1666 = insertelement <4 x float> poison, float %1503, i64 0, !dbg !19
  %1667 = insertelement <4 x float> %1666, float %1509, i64 1, !dbg !19
  %1668 = insertelement <4 x float> %1667, float %1504, i64 2, !dbg !19
  %1669 = insertelement <4 x float> %1668, float %1510, i64 3, !dbg !19
  %row.reciprocal30 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1454, !dbg !19
  %normalized31 = fmul <4 x float> %1669, %row.reciprocal30, !dbg !19
  %1670 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !20
  %1671 = extractelement <4 x half> %1670, i64 3, !dbg !20
  %1672 = extractelement <4 x half> %1670, i64 2, !dbg !20
  %1673 = extractelement <4 x half> %1670, i64 1, !dbg !20
  %1674 = extractelement <4 x half> %1670, i64 0, !dbg !20
  %1675 = insertelement <4 x float> poison, float %1393, i64 0, !dbg !19
  %1676 = insertelement <4 x float> %1675, float %1399, i64 1, !dbg !19
  %1677 = insertelement <4 x float> %1676, float %1394, i64 2, !dbg !19
  %1678 = insertelement <4 x float> %1677, float %1400, i64 3, !dbg !19
  %normalized32 = fmul <4 x float> %1678, %row.reciprocal30, !dbg !19
  %1679 = fptrunc <4 x float> %normalized32 to <4 x half>, !dbg !20
  %1680 = extractelement <4 x half> %1679, i64 3, !dbg !20
  %1681 = extractelement <4 x half> %1679, i64 2, !dbg !20
  %1682 = extractelement <4 x half> %1679, i64 1, !dbg !20
  %1683 = extractelement <4 x half> %1679, i64 0, !dbg !20
  %1684 = insertelement <4 x float> poison, float %1285, i64 0, !dbg !19
  %1685 = insertelement <4 x float> %1684, float %1291, i64 1, !dbg !19
  %1686 = insertelement <4 x float> %1685, float %1286, i64 2, !dbg !19
  %1687 = insertelement <4 x float> %1686, float %1292, i64 3, !dbg !19
  %row.reciprocal33 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1236, !dbg !19
  %normalized34 = fmul <4 x float> %1687, %row.reciprocal33, !dbg !19
  %1688 = fptrunc <4 x float> %normalized34 to <4 x half>, !dbg !20
  %1689 = extractelement <4 x half> %1688, i64 3, !dbg !20
  %1690 = extractelement <4 x half> %1688, i64 2, !dbg !20
  %1691 = extractelement <4 x half> %1688, i64 1, !dbg !20
  %1692 = extractelement <4 x half> %1688, i64 0, !dbg !20
  %1693 = insertelement <4 x float> poison, float %1175, i64 0, !dbg !19
  %1694 = insertelement <4 x float> %1693, float %1181, i64 1, !dbg !19
  %1695 = insertelement <4 x float> %1694, float %1176, i64 2, !dbg !19
  %1696 = insertelement <4 x float> %1695, float %1182, i64 3, !dbg !19
  %normalized35 = fmul <4 x float> %1696, %row.reciprocal33, !dbg !19
  %1697 = fptrunc <4 x float> %normalized35 to <4 x half>, !dbg !20
  %1698 = extractelement <4 x half> %1697, i64 3, !dbg !20
  %1699 = extractelement <4 x half> %1697, i64 2, !dbg !20
  %1700 = extractelement <4 x half> %1697, i64 1, !dbg !20
  %1701 = extractelement <4 x half> %1697, i64 0, !dbg !20
  %1702 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread443 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local444 = zext nneg i32 %address.thread443 to i64, !dbg !12
  %address.local445 = add i64 %28, %address.local444, !dbg !21
  %address.local446 = lshr i64 %address.local444, 6, !dbg !12
  %address.local447 = shl nuw nsw i64 %address.local446, 4, !dbg !21
  %address.local448 = add i64 %address.local445, %address.local447, !dbg !21
  %address.local449 = and i64 %address.local444, 240, !dbg !21
  %address.local450 = sub i64 %address.local448, %address.local449, !dbg !21
  %address.local451 = add i64 %address.local450, %29, !dbg !21
  %address.local452 = shl nuw nsw i64 %address.local451, 7, !dbg !21
  %address.local453 = add nuw nsw i64 %1702, %address.local452, !dbg !21
  %address.local454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local453, !dbg !21
  %address.local455 = lshr i64 %address.local444, 2, !dbg !12
  %address.local456 = and i64 %address.local455, 12, !dbg !12
  %address.local457 = or disjoint i64 %address.local456, 64, !dbg !21
  %address.local458 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local454, i64 %address.local457, !dbg !21
  store half %1701, ptr addrspace(1) %address.local458, align 2, !dbg !21
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local453, !dbg !21
  %address.local460 = or disjoint i64 %address.local456, 65, !dbg !21
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local459, i64 %address.local460, !dbg !21
  store half %1700, ptr addrspace(1) %address.local461, align 2, !dbg !21
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local453, !dbg !21
  %address.local463 = or disjoint i64 %address.local456, 66, !dbg !21
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local462, i64 %address.local463, !dbg !21
  store half %1699, ptr addrspace(1) %address.local464, align 2, !dbg !21
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local453, !dbg !21
  %address.local466 = or disjoint i64 %address.local456, 67, !dbg !21
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local465, i64 %address.local466, !dbg !21
  store half %1698, ptr addrspace(1) %address.local467, align 2, !dbg !21
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local453, !dbg !21
  %address.local469 = or disjoint i64 %address.local456, 80, !dbg !21
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local468, i64 %address.local469, !dbg !21
  store half %1692, ptr addrspace(1) %address.local470, align 2, !dbg !21
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local453, !dbg !21
  %address.local472 = or disjoint i64 %address.local456, 81, !dbg !21
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local471, i64 %address.local472, !dbg !21
  store half %1691, ptr addrspace(1) %address.local473, align 2, !dbg !21
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local453, !dbg !21
  %address.local475 = or disjoint i64 %address.local456, 82, !dbg !21
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local474, i64 %address.local475, !dbg !21
  store half %1690, ptr addrspace(1) %address.local476, align 2, !dbg !21
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local453, !dbg !21
  %address.local478 = or disjoint i64 %address.local456, 83, !dbg !21
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local477, i64 %address.local478, !dbg !21
  store half %1689, ptr addrspace(1) %address.local479, align 2, !dbg !21
  %address.local480 = shl i64 %address.local451, 7, !dbg !21
  %address.local481 = add i64 %address.local480, 8192, !dbg !21
  %address.local482 = add nuw nsw i64 %1702, %address.local481, !dbg !21
  %address.local483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local482, !dbg !21
  %address.local484 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local483, i64 %address.local457, !dbg !21
  store half %1683, ptr addrspace(1) %address.local484, align 2, !dbg !21
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local482, !dbg !21
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local485, i64 %address.local460, !dbg !21
  store half %1682, ptr addrspace(1) %address.local486, align 2, !dbg !21
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local482, !dbg !21
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local487, i64 %address.local463, !dbg !21
  store half %1681, ptr addrspace(1) %address.local488, align 2, !dbg !21
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local482, !dbg !21
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local466, !dbg !21
  store half %1680, ptr addrspace(1) %address.local490, align 2, !dbg !21
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local482, !dbg !21
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local491, i64 %address.local469, !dbg !21
  store half %1674, ptr addrspace(1) %address.local492, align 2, !dbg !21
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local482, !dbg !21
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local493, i64 %address.local472, !dbg !21
  store half %1673, ptr addrspace(1) %address.local494, align 2, !dbg !21
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local482, !dbg !21
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local475, !dbg !21
  store half %1672, ptr addrspace(1) %address.local496, align 2, !dbg !21
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local482, !dbg !21
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local497, i64 %address.local478, !dbg !21
  store half %1671, ptr addrspace(1) %address.local498, align 2, !dbg !21
  br label %1703, !dbg !21

1703:                                             ; preds = %1665, %1664
  br i1 true, label %1704, label %1742, !dbg !21

1704:                                             ; preds = %1703
  %1705 = insertelement <4 x float> poison, float %1475, i64 0, !dbg !19
  %1706 = insertelement <4 x float> %1705, float %1481, i64 1, !dbg !19
  %1707 = insertelement <4 x float> %1706, float %1476, i64 2, !dbg !19
  %1708 = insertelement <4 x float> %1707, float %1482, i64 3, !dbg !19
  %row.reciprocal36 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1454, !dbg !19
  %normalized37 = fmul <4 x float> %1708, %row.reciprocal36, !dbg !19
  %1709 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !20
  %1710 = extractelement <4 x half> %1709, i64 3, !dbg !20
  %1711 = extractelement <4 x half> %1709, i64 2, !dbg !20
  %1712 = extractelement <4 x half> %1709, i64 1, !dbg !20
  %1713 = extractelement <4 x half> %1709, i64 0, !dbg !20
  %1714 = insertelement <4 x float> poison, float %1364, i64 0, !dbg !19
  %1715 = insertelement <4 x float> %1714, float %1370, i64 1, !dbg !19
  %1716 = insertelement <4 x float> %1715, float %1365, i64 2, !dbg !19
  %1717 = insertelement <4 x float> %1716, float %1371, i64 3, !dbg !19
  %normalized38 = fmul <4 x float> %1717, %row.reciprocal36, !dbg !19
  %1718 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !20
  %1719 = extractelement <4 x half> %1718, i64 3, !dbg !20
  %1720 = extractelement <4 x half> %1718, i64 2, !dbg !20
  %1721 = extractelement <4 x half> %1718, i64 1, !dbg !20
  %1722 = extractelement <4 x half> %1718, i64 0, !dbg !20
  %1723 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !19
  %1724 = insertelement <4 x float> %1723, float %1263, i64 1, !dbg !19
  %1725 = insertelement <4 x float> %1724, float %1258, i64 2, !dbg !19
  %1726 = insertelement <4 x float> %1725, float %1264, i64 3, !dbg !19
  %row.reciprocal39 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1236, !dbg !19
  %normalized40 = fmul <4 x float> %1726, %row.reciprocal39, !dbg !19
  %1727 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !20
  %1728 = extractelement <4 x half> %1727, i64 3, !dbg !20
  %1729 = extractelement <4 x half> %1727, i64 2, !dbg !20
  %1730 = extractelement <4 x half> %1727, i64 1, !dbg !20
  %1731 = extractelement <4 x half> %1727, i64 0, !dbg !20
  %1732 = insertelement <4 x float> poison, float %1146, i64 0, !dbg !19
  %1733 = insertelement <4 x float> %1732, float %1152, i64 1, !dbg !19
  %1734 = insertelement <4 x float> %1733, float %1147, i64 2, !dbg !19
  %1735 = insertelement <4 x float> %1734, float %1153, i64 3, !dbg !19
  %normalized41 = fmul <4 x float> %1735, %row.reciprocal39, !dbg !19
  %1736 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !20
  %1737 = extractelement <4 x half> %1736, i64 3, !dbg !20
  %1738 = extractelement <4 x half> %1736, i64 2, !dbg !20
  %1739 = extractelement <4 x half> %1736, i64 1, !dbg !20
  %1740 = extractelement <4 x half> %1736, i64 0, !dbg !20
  %1741 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread499 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local500 = zext nneg i32 %address.thread499 to i64, !dbg !12
  %address.local501 = add i64 %28, %address.local500, !dbg !21
  %address.local502 = lshr i64 %address.local500, 6, !dbg !12
  %address.local503 = shl nuw nsw i64 %address.local502, 4, !dbg !21
  %address.local504 = add i64 %address.local501, %address.local503, !dbg !21
  %address.local505 = and i64 %address.local500, 240, !dbg !21
  %address.local506 = sub i64 %address.local504, %address.local505, !dbg !21
  %address.local507 = add i64 %address.local506, %29, !dbg !21
  %address.local508 = shl nuw nsw i64 %address.local507, 7, !dbg !21
  %address.local509 = add nuw nsw i64 %1741, %address.local508, !dbg !21
  %address.local510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local509, !dbg !21
  %address.local511 = lshr i64 %address.local500, 2, !dbg !12
  %address.local512 = and i64 %address.local511, 12, !dbg !12
  %address.local513 = or disjoint i64 %address.local512, 96, !dbg !21
  %address.local514 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local510, i64 %address.local513, !dbg !21
  store half %1740, ptr addrspace(1) %address.local514, align 2, !dbg !21
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local509, !dbg !21
  %address.local516 = or disjoint i64 %address.local512, 97, !dbg !21
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local515, i64 %address.local516, !dbg !21
  store half %1739, ptr addrspace(1) %address.local517, align 2, !dbg !21
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local509, !dbg !21
  %address.local519 = or disjoint i64 %address.local512, 98, !dbg !21
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local518, i64 %address.local519, !dbg !21
  store half %1738, ptr addrspace(1) %address.local520, align 2, !dbg !21
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local509, !dbg !21
  %address.local522 = or disjoint i64 %address.local512, 99, !dbg !21
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local521, i64 %address.local522, !dbg !21
  store half %1737, ptr addrspace(1) %address.local523, align 2, !dbg !21
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local509, !dbg !21
  %address.local525 = or disjoint i64 %address.local512, 112, !dbg !21
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local524, i64 %address.local525, !dbg !21
  store half %1731, ptr addrspace(1) %address.local526, align 2, !dbg !21
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local509, !dbg !21
  %address.local528 = or disjoint i64 %address.local512, 113, !dbg !21
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local527, i64 %address.local528, !dbg !21
  store half %1730, ptr addrspace(1) %address.local529, align 2, !dbg !21
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local509, !dbg !21
  %address.local531 = or disjoint i64 %address.local512, 114, !dbg !21
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local530, i64 %address.local531, !dbg !21
  store half %1729, ptr addrspace(1) %address.local532, align 2, !dbg !21
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local509, !dbg !21
  %address.local534 = or i64 %address.local511, 115, !dbg !21
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local533, i64 %address.local534, !dbg !21
  store half %1728, ptr addrspace(1) %address.local535, align 2, !dbg !21
  %address.local536 = shl i64 %address.local507, 7, !dbg !21
  %address.local537 = add i64 %address.local536, 8192, !dbg !21
  %address.local538 = add nuw nsw i64 %1741, %address.local537, !dbg !21
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local538, !dbg !21
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local539, i64 %address.local513, !dbg !21
  store half %1722, ptr addrspace(1) %address.local540, align 2, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local538, !dbg !21
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local541, i64 %address.local516, !dbg !21
  store half %1721, ptr addrspace(1) %address.local542, align 2, !dbg !21
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local538, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local543, i64 %address.local519, !dbg !21
  store half %1720, ptr addrspace(1) %address.local544, align 2, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local538, !dbg !21
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local522, !dbg !21
  store half %1719, ptr addrspace(1) %address.local546, align 2, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local538, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local525, !dbg !21
  store half %1713, ptr addrspace(1) %address.local548, align 2, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local538, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local528, !dbg !21
  store half %1712, ptr addrspace(1) %address.local550, align 2, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local538, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local551, i64 %address.local531, !dbg !21
  store half %1711, ptr addrspace(1) %address.local552, align 2, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local538, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local553, i64 %address.local534, !dbg !21
  store half %1710, ptr addrspace(1) %address.local554, align 2, !dbg !21
  br label %1742, !dbg !21

1742:                                             ; preds = %1704, %1703
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/group8-reverse")
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
!15 = !DILocation(line: 102, column: 23, scope: !4)
!16 = !DILocation(line: 73, column: 26, scope: !4)
!17 = !DILocation(line: 71, column: 22, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!22 = !DILocation(line: 74, column: 23, scope: !4)
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
