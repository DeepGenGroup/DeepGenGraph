; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/group8-reverse-k128-swizzle/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/group8-reverse-k128-swizzle/input.mlir"

@shm_0 = addrspace(3) global [6144 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
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
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !39
  %35 = zext nneg i32 %34 to i64, !dbg !12
  %36 = lshr i64 %35, 6, !dbg !12
  %37 = lshr i64 %35, 2, !dbg !12
  %38 = and i64 %37, 12, !dbg !12
  %39 = and i32 %34, 16, !dbg !13
  %.not = icmp eq i32 %39, 0, !dbg !13
  %40 = and i32 %34, 32, !dbg !13
  %.not1 = icmp eq i32 %40, 0, !dbg !13
  %41 = or disjoint i64 %19, 63, !dbg !14
  %42 = add i64 %19, 64, !dbg !14
  %43 = icmp uge i64 %42, %41, !dbg !14
  %44 = shl nsw i64 %16, 1, !dbg !7
  %45 = add nsw i64 %44, 2, !dbg !7
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

57:                                               ; preds = %763, %4
  %58 = phi i64 [ %793, %763 ], [ 0, %4 ], !dbg !7
  %59 = phi <4 x float> [ %674, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi <4 x float> [ %679, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = phi <8 x float> [ %788, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %62 = phi <1 x float> [ %789, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi <1 x float> [ %576, %763 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %64 = phi <4 x float> [ %721, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = phi <4 x float> [ %726, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %66 = phi <8 x float> [ %790, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = phi <4 x float> [ %768, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = phi <4 x float> [ %773, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %69 = phi <8 x float> [ %791, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %70 = phi <4 x float> [ %782, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %71 = phi <4 x float> [ %787, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %72 = phi <8 x float> [ %792, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %73 = icmp slt i64 %58, %45, !dbg !7
  br i1 %73, label %74, label %794, !dbg !7

74:                                               ; preds = %57
  %75 = shl nsw i64 %58, 5, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local31 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local32 = and i64 %address.local31, 1536, !dbg !11
  %address.local33 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local32, !dbg !11
  %address.local34 = ptrtoint ptr addrspace(3) %address.local33 to i64, !dbg !11
  %address.local35 = trunc i64 %address.local34 to i32, !dbg !11
  %76 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local35), !dbg !11
  %77 = zext i32 %76 to i64, !dbg !11
  %78 = inttoptr i64 %77 to ptr addrspace(3), !dbg !11
  %address.local36 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local37 = and i64 %address.local36, 8064, !dbg !11
  %address.local38 = lshr i64 %address.local, 3, !dbg !11
  %address.local39 = and i64 %address.local38, 24, !dbg !11
  %address.local40 = or disjoint i64 %address.local37, %address.local39, !dbg !11
  %address.local41 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local40, !dbg !11
  %79 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %80 = ptrtoint ptr addrspace(1) %address.local41 to i64, !dbg !11
  %81 = sub i64 %80, %79, !dbg !11
  %dma.buffer.offset533 = trunc i64 %81 to i32, !dbg !11
  %82 = trunc i64 %79 to i32, !dbg !11
  %83 = insertelement <4 x i32> poison, i32 %82, i64 0, !dbg !11
  %84 = lshr i64 %79, 32, !dbg !11
  %85 = trunc i64 %84 to i32, !dbg !11
  %86 = insertelement <4 x i32> %83, i32 %85, i64 1, !dbg !11
  %87 = insertelement <4 x i32> %86, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc534 = insertelement <4 x i32> %87, i32 131072, i64 3, !dbg !11
  %88 = ptrtoint ptr addrspace(3) %78 to i32, !dbg !11
  %dma.buffer.lds535 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %88), !dbg !11
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds535, i32 %dma.buffer.offset533, <4 x i32> %dma.buffer.rsrc534) #10, !dbg !11
  %90 = lshr i64 %58, 14, !dbg !17
  %91 = add nuw nsw i64 %90, %15, !dbg !17
  %92 = lshr i64 %58, 7, !dbg !17
  %93 = and i64 %92, 127, !dbg !17
  %94 = shl i64 %91, 19, !dbg !17
  %95 = and i64 %94, 16252928, !dbg !17
  %96 = shl i64 %92, 19, !dbg !17
  %97 = shl i64 %58, 12, !dbg !17
  %98 = sub i64 %97, %96, !dbg !17
  %99 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %95, !dbg !17
  %100 = getelementptr [2 x i8], ptr addrspace(1) %99, i64 %93, !dbg !17
  %101 = getelementptr [2 x i8], ptr addrspace(1) %100, i64 %98, !dbg !17
  br label %102, !dbg !17

102:                                              ; preds = %105, %74
  %103 = phi i64 [ %135, %105 ], [ %35, %74 ], !dbg !17
  %104 = icmp samesign ult i64 %103, 128, !dbg !17
  br i1 %104, label %105, label %136, !dbg !17

105:                                              ; preds = %102
  %106 = and i64 %103, 27, !dbg !17
  %107 = lshr i64 %103, 2, !dbg !17
  %108 = and i64 %107, 4, !dbg !17
  %109 = add nuw nsw i64 %103, %108, !dbg !17
  %110 = and i64 %109, 4, !dbg !17
  %111 = or disjoint i64 %106, %110, !dbg !17
  %112 = lshr i64 %103, 2, !dbg !17
  %113 = and i64 %112, 24, !dbg !17
  %114 = shl nuw nsw i64 %103, 3, !dbg !17
  %115 = and i64 %114, 512, !dbg !17
  %116 = getelementptr [2 x i8], ptr addrspace(1) %101, i64 %113, !dbg !17
  %.idx14 = shl nuw nsw i64 %111, 8, !dbg !17
  %117 = getelementptr i8, ptr addrspace(1) %116, i64 %.idx14, !dbg !17
  %118 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %115, !dbg !17
  %119 = ptrtoint ptr addrspace(3) %118 to i64, !dbg !17
  %120 = trunc i64 %119 to i32, !dbg !17
  %121 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %120), !dbg !17
  %122 = zext i32 %121 to i64, !dbg !17
  %123 = inttoptr i64 %122 to ptr addrspace(3), !dbg !17
  %124 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %125 = ptrtoint ptr addrspace(1) %117 to i64, !dbg !17
  %126 = sub i64 %125, %124, !dbg !17
  %dma.buffer.offset = trunc i64 %126 to i32, !dbg !17
  %127 = trunc i64 %124 to i32, !dbg !17
  %128 = insertelement <4 x i32> poison, i32 %127, i64 0, !dbg !17
  %129 = lshr i64 %124, 32, !dbg !17
  %130 = trunc i64 %129 to i32, !dbg !17
  %131 = insertelement <4 x i32> %128, i32 %130, i64 1, !dbg !17
  %132 = insertelement <4 x i32> %131, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %132, i32 131072, i64 3, !dbg !17
  %133 = ptrtoint ptr addrspace(3) %123 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %133), !dbg !17
  %134 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %135 = or disjoint i64 %35, 256, !dbg !17
  br label %102, !dbg !17

136:                                              ; preds = %102
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.thread42 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local43 = lshr i32 %address.thread42, 2, !dbg !12
  %address.local44 = and i32 %address.local43, 4, !dbg !12
  %address.local45 = shl nuw nsw i32 %address.thread42, 1, !dbg !12
  %address.local46 = and i32 %address.local45, 384, !dbg !12
  %address.local47 = or disjoint i32 %address.local44, %address.local46, !dbg !12
  %address.local48 = shl nuw nsw i32 %address.thread42, 3, !dbg !12
  %address.local49 = and i32 %address.local48, 120, !dbg !12
  %address.local50 = or disjoint i32 %address.local47, %address.local49, !dbg !12
  %address.local51 = shl nuw nsw i32 %address.thread42, 4, !dbg !12
  %address.local52 = and i32 %address.local51, 512, !dbg !12
  %address.local53 = or disjoint i32 %address.local50, %address.local52, !dbg !12
  %address.local54 = zext nneg i32 %address.local53 to i64, !dbg !12
  %address.local55 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local54, !dbg !12
  %137 = load <4 x half>, ptr addrspace(3) %address.local55, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local55, i64 2048, !dbg !12
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %139 = fmul <4 x half> %137, splat (half 0xH3015), !dbg !12
  %140 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local58 = zext nneg i32 %address.thread42 to i64, !dbg !12
  %address.local59 = shl nuw nsw i64 %address.local58, 3, !dbg !12
  %address.local60 = lshr i64 %address.local58, 4, !dbg !12
  %address.local61 = mul nsw i64 %address.local60, -128, !dbg !12
  %address.local62 = add nsw i64 %address.local59, %address.local61, !dbg !16
  %address.local63 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local62, !dbg !16
  %address.local64 = lshr i64 %address.local58, 2, !dbg !12
  %address.local65 = and i64 %address.local64, 12, !dbg !12
  %address.local66 = getelementptr [2 x i8], ptr addrspace(3) %address.local63, i64 %address.local65, !dbg !16
  %address.local67 = and i64 %address.local58, 32, !dbg !16
  %address.local68 = icmp eq i64 %address.local67, 0, !dbg !16
  %address.local69 = select i1 %address.local68, i64 0, i64 248, !dbg !16
  %address.local70 = getelementptr [2 x i8], ptr addrspace(3) %address.local66, i64 %address.local69, !dbg !16
  %141 = load <4 x half>, ptr addrspace(3) %address.local70, align 8, !dbg !16
  %lds.const.gep483 = getelementptr i8, ptr addrspace(3) %address.local70, i64 1024, !dbg !16
  %142 = load <4 x half>, ptr addrspace(3) %lds.const.gep483, align 8, !dbg !16
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %141, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local76 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local62, !dbg !16
  %address.local77 = shl i64 %address.local58, 61, !dbg !16
  %address.local78 = ashr exact i64 %address.local77, 57, !dbg !16
  %address.local79 = and i64 %address.local78, -64, !dbg !16
  %address.local80 = getelementptr [2 x i8], ptr addrspace(3) %address.local76, i64 %address.local79, !dbg !16
  %address.local81 = getelementptr [2 x i8], ptr addrspace(3) %address.local80, i64 %address.local65, !dbg !16
  %address.local82 = getelementptr [2 x i8], ptr addrspace(3) %address.local81, i64 %address.local69, !dbg !16
  %address.local83 = getelementptr i8, ptr addrspace(3) %address.local82, i64 320, !dbg !16
  %144 = load <4 x half>, ptr addrspace(3) %address.local83, align 8, !dbg !16
  %145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %142, <4 x float> %143) #9, !dbg !16
  %lds.const.gep484 = getelementptr i8, ptr addrspace(3) %address.local83, i64 1024, !dbg !16
  %146 = load <4 x half>, ptr addrspace(3) %lds.const.gep484, align 8, !dbg !16
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %144, <4 x float> zeroinitializer) #9, !dbg !16
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %146, <4 x float> %147) #9, !dbg !16
  %address.local93 = shl nuw nsw i64 %address.local58, 3, !dbg !11
  %address.local94 = and i64 %address.local93, 1536, !dbg !11
  %address.local95 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local94, !dbg !11
  %address.local96 = ptrtoint ptr addrspace(3) %address.local95 to i64, !dbg !11
  %address.local97 = trunc i64 %address.local96 to i32, !dbg !11
  %149 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local97), !dbg !11
  %150 = zext i32 %149 to i64, !dbg !11
  %151 = inttoptr i64 %150 to ptr addrspace(3), !dbg !11
  %address.local98 = shl nuw nsw i64 %address.local58, 7, !dbg !11
  %address.local99 = and i64 %address.local98, 8064, !dbg !11
  %address.local100 = lshr i64 %address.local58, 3, !dbg !11
  %address.local101 = and i64 %address.local100, 24, !dbg !11
  %address.local102 = or disjoint i64 %address.local99, %address.local101, !dbg !11
  %address.local103 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local102, !dbg !11
  %152 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %153 = ptrtoint ptr addrspace(1) %address.local103 to i64, !dbg !11
  %154 = sub i64 %153, %152, !dbg !11
  %dma.buffer.offset530 = trunc i64 %154 to i32, !dbg !11
  %155 = trunc i64 %152 to i32, !dbg !11
  %156 = insertelement <4 x i32> poison, i32 %155, i64 0, !dbg !11
  %157 = lshr i64 %152, 32, !dbg !11
  %158 = trunc i64 %157 to i32, !dbg !11
  %159 = insertelement <4 x i32> %156, i32 %158, i64 1, !dbg !11
  %160 = insertelement <4 x i32> %159, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc531 = insertelement <4 x i32> %160, i32 131072, i64 3, !dbg !11
  %161 = ptrtoint ptr addrspace(3) %151 to i32, !dbg !11
  %dma.buffer.lds532 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %161), !dbg !11
  %162 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds532, i32 %dma.buffer.offset530, <4 x i32> %dma.buffer.rsrc531) #10, !dbg !11
  %163 = add nuw i64 %58, 4096, !dbg !17
  %164 = icmp slt i64 %163, 0, !dbg !17
  %165 = sub nuw i64 -4097, %58, !dbg !17
  %166 = select i1 %164, i64 %165, i64 %163, !dbg !17
  %167 = sdiv i64 %166, 16384, !dbg !17
  %.lobit = ashr i64 %163, 63, !dbg !17
  %168 = xor i64 %167, %.lobit, !dbg !17
  %169 = add nsw i64 %168, %15, !dbg !17
  %170 = add nsw i64 %92, %52, !dbg !17
  %171 = add nsw i64 %170, 32, !dbg !17
  %172 = icmp slt i64 %170, -32, !dbg !17
  %173 = sub nsw i64 -33, %170, !dbg !17
  %174 = select i1 %172, i64 %173, i64 %171, !dbg !17
  %175 = sdiv i64 %174, 128, !dbg !17
  %176 = sext i1 %172 to i64, !dbg !17
  %177 = xor i64 %175, %176, !dbg !17
  %178 = shl nsw i64 %177, 7, !dbg !17
  %179 = sub i64 %52, %178, !dbg !17
  %180 = shl i64 %169, 19, !dbg !17
  %181 = and i64 %180, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %182 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %181, !dbg !17
  %183 = getelementptr [2 x i8], ptr addrspace(1) %182, i64 %92, !dbg !17
  %184 = getelementptr [2 x i8], ptr addrspace(1) %183, i64 %179, !dbg !17
  %185 = getelementptr i8, ptr addrspace(1) %184, i64 64, !dbg !17
  %186 = getelementptr [2 x i8], ptr addrspace(1) %185, i64 %98, !dbg !17
  br label %187, !dbg !17

187:                                              ; preds = %190, %136
  %188 = phi i64 [ %220, %190 ], [ %35, %136 ], !dbg !17
  %189 = icmp samesign ult i64 %188, 128, !dbg !17
  br i1 %189, label %190, label %221, !dbg !17

190:                                              ; preds = %187
  %191 = and i64 %188, 27, !dbg !17
  %192 = lshr i64 %188, 2, !dbg !17
  %193 = and i64 %192, 4, !dbg !17
  %194 = add nuw nsw i64 %188, %193, !dbg !17
  %195 = and i64 %194, 4, !dbg !17
  %196 = or disjoint i64 %191, %195, !dbg !17
  %197 = lshr i64 %188, 2, !dbg !17
  %198 = and i64 %197, 24, !dbg !17
  %199 = shl nuw nsw i64 %188, 3, !dbg !17
  %200 = and i64 %199, 512, !dbg !17
  %201 = getelementptr [2 x i8], ptr addrspace(1) %186, i64 %198, !dbg !17
  %.idx13 = shl nuw nsw i64 %196, 8, !dbg !17
  %202 = getelementptr i8, ptr addrspace(1) %201, i64 %.idx13, !dbg !17
  %203 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %200, !dbg !17
  %204 = ptrtoint ptr addrspace(3) %203 to i64, !dbg !17
  %205 = trunc i64 %204 to i32, !dbg !17
  %206 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %205), !dbg !17
  %207 = zext i32 %206 to i64, !dbg !17
  %208 = inttoptr i64 %207 to ptr addrspace(3), !dbg !17
  %209 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %210 = ptrtoint ptr addrspace(1) %202 to i64, !dbg !17
  %211 = sub i64 %210, %209, !dbg !17
  %dma.buffer.offset503 = trunc i64 %211 to i32, !dbg !17
  %212 = trunc i64 %209 to i32, !dbg !17
  %213 = insertelement <4 x i32> poison, i32 %212, i64 0, !dbg !17
  %214 = lshr i64 %209, 32, !dbg !17
  %215 = trunc i64 %214 to i32, !dbg !17
  %216 = insertelement <4 x i32> %213, i32 %215, i64 1, !dbg !17
  %217 = insertelement <4 x i32> %216, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc504 = insertelement <4 x i32> %217, i32 131072, i64 3, !dbg !17
  %218 = ptrtoint ptr addrspace(3) %208 to i32, !dbg !17
  %dma.buffer.lds505 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %218), !dbg !17
  %219 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds505, i32 %dma.buffer.offset503, <4 x i32> %dma.buffer.rsrc504) #10, !dbg !17
  %220 = or disjoint i64 %35, 256, !dbg !17
  br label %187, !dbg !17

221:                                              ; preds = %187
  %address.thread104 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %lds.const.gep485 = getelementptr i8, ptr addrspace(3) %address.local55, i64 0, !dbg !12
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep485, align 8, !dbg !12
  %lds.const.gep486 = getelementptr i8, ptr addrspace(3) %address.local55, i64 2048, !dbg !12
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep486, align 8, !dbg !12
  %224 = fmul <4 x half> %222, splat (half 0xH3015), !dbg !12
  %225 = fmul <4 x half> %223, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local120 = zext nneg i32 %address.thread104 to i64, !dbg !12
  %lds.const.gep487 = getelementptr i8, ptr addrspace(3) %address.local70, i64 0, !dbg !16
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep487, align 8, !dbg !16
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %226, <4 x float> %145) #9, !dbg !16
  %lds.const.gep488 = getelementptr i8, ptr addrspace(3) %address.local70, i64 1024, !dbg !16
  %228 = load <4 x half>, ptr addrspace(3) %lds.const.gep488, align 8, !dbg !16
  %lds.const.gep489 = getelementptr i8, ptr addrspace(3) %address.local83, i64 0, !dbg !16
  %229 = load <4 x half>, ptr addrspace(3) %lds.const.gep489, align 8, !dbg !16
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %228, <4 x float> %227) #9, !dbg !16
  %lds.const.gep490 = getelementptr i8, ptr addrspace(3) %address.local83, i64 1024, !dbg !16
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep490, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %229, <4 x float> %148) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %231, <4 x float> %232) #9, !dbg !16
  %address.local155 = shl nuw nsw i64 %address.local120, 3, !dbg !11
  %address.local156 = and i64 %address.local155, 1536, !dbg !11
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local156, !dbg !11
  %address.local158 = ptrtoint ptr addrspace(3) %address.local157 to i64, !dbg !11
  %address.local159 = trunc i64 %address.local158 to i32, !dbg !11
  %234 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local159), !dbg !11
  %235 = zext i32 %234 to i64, !dbg !11
  %236 = inttoptr i64 %235 to ptr addrspace(3), !dbg !11
  %address.local160 = shl nuw nsw i64 %address.local120, 7, !dbg !11
  %address.local161 = and i64 %address.local160, 8064, !dbg !11
  %address.local162 = lshr i64 %address.local120, 3, !dbg !11
  %address.local163 = and i64 %address.local162, 24, !dbg !11
  %address.local164 = or disjoint i64 %address.local161, %address.local163, !dbg !11
  %address.local165 = getelementptr [2 x i8], ptr addrspace(1) %54, i64 %address.local164, !dbg !11
  %237 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %238 = ptrtoint ptr addrspace(1) %address.local165 to i64, !dbg !11
  %239 = sub i64 %238, %237, !dbg !11
  %dma.buffer.offset527 = trunc i64 %239 to i32, !dbg !11
  %240 = trunc i64 %237 to i32, !dbg !11
  %241 = insertelement <4 x i32> poison, i32 %240, i64 0, !dbg !11
  %242 = lshr i64 %237, 32, !dbg !11
  %243 = trunc i64 %242 to i32, !dbg !11
  %244 = insertelement <4 x i32> %241, i32 %243, i64 1, !dbg !11
  %245 = insertelement <4 x i32> %244, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc528 = insertelement <4 x i32> %245, i32 131072, i64 3, !dbg !11
  %246 = ptrtoint ptr addrspace(3) %236 to i32, !dbg !11
  %dma.buffer.lds529 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %246), !dbg !11
  %247 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds529, i32 %dma.buffer.offset527, <4 x i32> %dma.buffer.rsrc528) #10, !dbg !11
  %248 = add nuw i64 %58, 8192, !dbg !17
  %249 = icmp slt i64 %248, 0, !dbg !17
  %250 = sub nuw i64 -8193, %58, !dbg !17
  %251 = select i1 %249, i64 %250, i64 %248, !dbg !17
  %252 = sdiv i64 %251, 16384, !dbg !17
  %.lobit2 = ashr i64 %248, 63, !dbg !17
  %253 = xor i64 %252, %.lobit2, !dbg !17
  %254 = add nsw i64 %253, %15, !dbg !17
  %255 = add nsw i64 %170, 64, !dbg !17
  %256 = icmp slt i64 %170, -64, !dbg !17
  %257 = sub nsw i64 -65, %170, !dbg !17
  %258 = select i1 %256, i64 %257, i64 %255, !dbg !17
  %259 = sdiv i64 %258, 128, !dbg !17
  %260 = sext i1 %256 to i64, !dbg !17
  %261 = xor i64 %259, %260, !dbg !17
  %262 = shl nsw i64 %261, 7, !dbg !17
  %263 = sub i64 %52, %262, !dbg !17
  %264 = shl i64 %254, 19, !dbg !17
  %265 = and i64 %264, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %266 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %265, !dbg !17
  %267 = getelementptr [2 x i8], ptr addrspace(1) %266, i64 %92, !dbg !17
  %268 = getelementptr [2 x i8], ptr addrspace(1) %267, i64 %263, !dbg !17
  %269 = getelementptr i8, ptr addrspace(1) %268, i64 128, !dbg !17
  %270 = getelementptr [2 x i8], ptr addrspace(1) %269, i64 %98, !dbg !17
  br label %271, !dbg !17

271:                                              ; preds = %274, %221
  %272 = phi i64 [ %304, %274 ], [ %35, %221 ], !dbg !17
  %273 = icmp samesign ult i64 %272, 128, !dbg !17
  br i1 %273, label %274, label %305, !dbg !17

274:                                              ; preds = %271
  %275 = and i64 %272, 27, !dbg !17
  %276 = lshr i64 %272, 2, !dbg !17
  %277 = and i64 %276, 4, !dbg !17
  %278 = add nuw nsw i64 %272, %277, !dbg !17
  %279 = and i64 %278, 4, !dbg !17
  %280 = or disjoint i64 %275, %279, !dbg !17
  %281 = lshr i64 %272, 2, !dbg !17
  %282 = and i64 %281, 24, !dbg !17
  %283 = shl nuw nsw i64 %272, 3, !dbg !17
  %284 = and i64 %283, 512, !dbg !17
  %285 = getelementptr [2 x i8], ptr addrspace(1) %270, i64 %282, !dbg !17
  %.idx12 = shl nuw nsw i64 %280, 8, !dbg !17
  %286 = getelementptr i8, ptr addrspace(1) %285, i64 %.idx12, !dbg !17
  %287 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %284, !dbg !17
  %288 = ptrtoint ptr addrspace(3) %287 to i64, !dbg !17
  %289 = trunc i64 %288 to i32, !dbg !17
  %290 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %289), !dbg !17
  %291 = zext i32 %290 to i64, !dbg !17
  %292 = inttoptr i64 %291 to ptr addrspace(3), !dbg !17
  %293 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %294 = ptrtoint ptr addrspace(1) %286 to i64, !dbg !17
  %295 = sub i64 %294, %293, !dbg !17
  %dma.buffer.offset506 = trunc i64 %295 to i32, !dbg !17
  %296 = trunc i64 %293 to i32, !dbg !17
  %297 = insertelement <4 x i32> poison, i32 %296, i64 0, !dbg !17
  %298 = lshr i64 %293, 32, !dbg !17
  %299 = trunc i64 %298 to i32, !dbg !17
  %300 = insertelement <4 x i32> %297, i32 %299, i64 1, !dbg !17
  %301 = insertelement <4 x i32> %300, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc507 = insertelement <4 x i32> %301, i32 131072, i64 3, !dbg !17
  %302 = ptrtoint ptr addrspace(3) %292 to i32, !dbg !17
  %dma.buffer.lds508 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %302), !dbg !17
  %303 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds508, i32 %dma.buffer.offset506, <4 x i32> %dma.buffer.rsrc507) #10, !dbg !17
  %304 = or disjoint i64 %35, 256, !dbg !17
  br label %271, !dbg !17

305:                                              ; preds = %271
  %address.thread166 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %lds.const.gep491 = getelementptr i8, ptr addrspace(3) %address.local55, i64 0, !dbg !12
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep491, align 8, !dbg !12
  %lds.const.gep492 = getelementptr i8, ptr addrspace(3) %address.local55, i64 2048, !dbg !12
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep492, align 8, !dbg !12
  %308 = fmul <4 x half> %306, splat (half 0xH3015), !dbg !12
  %309 = fmul <4 x half> %307, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local182 = zext nneg i32 %address.thread166 to i64, !dbg !12
  %lds.const.gep493 = getelementptr i8, ptr addrspace(3) %address.local70, i64 0, !dbg !16
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep493, align 8, !dbg !16
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %308, <4 x half> %310, <4 x float> %230) #9, !dbg !16
  %lds.const.gep494 = getelementptr i8, ptr addrspace(3) %address.local70, i64 1024, !dbg !16
  %312 = load <4 x half>, ptr addrspace(3) %lds.const.gep494, align 8, !dbg !16
  %lds.const.gep495 = getelementptr i8, ptr addrspace(3) %address.local83, i64 0, !dbg !16
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep495, align 8, !dbg !16
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %309, <4 x half> %312, <4 x float> %311) #9, !dbg !16
  %lds.const.gep496 = getelementptr i8, ptr addrspace(3) %address.local83, i64 1024, !dbg !16
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep496, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %308, <4 x half> %313, <4 x float> %233) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %309, <4 x half> %315, <4 x float> %316) #9, !dbg !16
  %address.local217 = shl nuw nsw i64 %address.local182, 3, !dbg !11
  %address.local218 = and i64 %address.local217, 1536, !dbg !11
  %address.local219 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local218, !dbg !11
  %address.local220 = ptrtoint ptr addrspace(3) %address.local219 to i64, !dbg !11
  %address.local221 = trunc i64 %address.local220 to i32, !dbg !11
  %318 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local221), !dbg !11
  %319 = zext i32 %318 to i64, !dbg !11
  %320 = inttoptr i64 %319 to ptr addrspace(3), !dbg !11
  %address.local222 = shl nuw nsw i64 %address.local182, 7, !dbg !11
  %address.local223 = and i64 %address.local222, 8064, !dbg !11
  %address.local224 = lshr i64 %address.local182, 3, !dbg !11
  %address.local225 = and i64 %address.local224, 24, !dbg !11
  %address.local226 = or disjoint i64 %address.local223, %address.local225, !dbg !11
  %address.local227 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local226, !dbg !11
  %321 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %322 = ptrtoint ptr addrspace(1) %address.local227 to i64, !dbg !11
  %323 = sub i64 %322, %321, !dbg !11
  %dma.buffer.offset524 = trunc i64 %323 to i32, !dbg !11
  %324 = trunc i64 %321 to i32, !dbg !11
  %325 = insertelement <4 x i32> poison, i32 %324, i64 0, !dbg !11
  %326 = lshr i64 %321, 32, !dbg !11
  %327 = trunc i64 %326 to i32, !dbg !11
  %328 = insertelement <4 x i32> %325, i32 %327, i64 1, !dbg !11
  %329 = insertelement <4 x i32> %328, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc525 = insertelement <4 x i32> %329, i32 131072, i64 3, !dbg !11
  %330 = ptrtoint ptr addrspace(3) %320 to i32, !dbg !11
  %dma.buffer.lds526 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %330), !dbg !11
  %331 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds526, i32 %dma.buffer.offset524, <4 x i32> %dma.buffer.rsrc525) #10, !dbg !11
  %332 = add nuw i64 %58, 12288, !dbg !17
  %333 = icmp slt i64 %332, 0, !dbg !17
  %334 = sub nuw i64 -12289, %58, !dbg !17
  %335 = select i1 %333, i64 %334, i64 %332, !dbg !17
  %336 = sdiv i64 %335, 16384, !dbg !17
  %.lobit3 = ashr i64 %332, 63, !dbg !17
  %337 = xor i64 %336, %.lobit3, !dbg !17
  %338 = add nsw i64 %337, %15, !dbg !17
  %339 = add nsw i64 %170, 96, !dbg !17
  %340 = icmp slt i64 %170, -96, !dbg !17
  %341 = sub nsw i64 -97, %170, !dbg !17
  %342 = select i1 %340, i64 %341, i64 %339, !dbg !17
  %343 = sdiv i64 %342, 128, !dbg !17
  %344 = sext i1 %340 to i64, !dbg !17
  %345 = xor i64 %343, %344, !dbg !17
  %346 = shl nsw i64 %345, 7, !dbg !17
  %347 = sub i64 %52, %346, !dbg !17
  %348 = shl i64 %338, 19, !dbg !17
  %349 = and i64 %348, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %350 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %349, !dbg !17
  %351 = getelementptr [2 x i8], ptr addrspace(1) %350, i64 %92, !dbg !17
  %352 = getelementptr [2 x i8], ptr addrspace(1) %351, i64 %347, !dbg !17
  %353 = getelementptr i8, ptr addrspace(1) %352, i64 192, !dbg !17
  %354 = getelementptr [2 x i8], ptr addrspace(1) %353, i64 %98, !dbg !17
  br label %355, !dbg !17

355:                                              ; preds = %358, %305
  %356 = phi i64 [ %388, %358 ], [ %35, %305 ], !dbg !17
  %357 = icmp samesign ult i64 %356, 128, !dbg !17
  br i1 %357, label %358, label %389, !dbg !17

358:                                              ; preds = %355
  %359 = and i64 %356, 27, !dbg !17
  %360 = lshr i64 %356, 2, !dbg !17
  %361 = and i64 %360, 4, !dbg !17
  %362 = add nuw nsw i64 %356, %361, !dbg !17
  %363 = and i64 %362, 4, !dbg !17
  %364 = or disjoint i64 %359, %363, !dbg !17
  %365 = lshr i64 %356, 2, !dbg !17
  %366 = and i64 %365, 24, !dbg !17
  %367 = shl nuw nsw i64 %356, 3, !dbg !17
  %368 = and i64 %367, 512, !dbg !17
  %369 = getelementptr [2 x i8], ptr addrspace(1) %354, i64 %366, !dbg !17
  %.idx = shl nuw nsw i64 %364, 8, !dbg !17
  %370 = getelementptr i8, ptr addrspace(1) %369, i64 %.idx, !dbg !17
  %371 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %368, !dbg !17
  %372 = ptrtoint ptr addrspace(3) %371 to i64, !dbg !17
  %373 = trunc i64 %372 to i32, !dbg !17
  %374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %373), !dbg !17
  %375 = zext i32 %374 to i64, !dbg !17
  %376 = inttoptr i64 %375 to ptr addrspace(3), !dbg !17
  %377 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %378 = ptrtoint ptr addrspace(1) %370 to i64, !dbg !17
  %379 = sub i64 %378, %377, !dbg !17
  %dma.buffer.offset509 = trunc i64 %379 to i32, !dbg !17
  %380 = trunc i64 %377 to i32, !dbg !17
  %381 = insertelement <4 x i32> poison, i32 %380, i64 0, !dbg !17
  %382 = lshr i64 %377, 32, !dbg !17
  %383 = trunc i64 %382 to i32, !dbg !17
  %384 = insertelement <4 x i32> %381, i32 %383, i64 1, !dbg !17
  %385 = insertelement <4 x i32> %384, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc510 = insertelement <4 x i32> %385, i32 131072, i64 3, !dbg !17
  %386 = ptrtoint ptr addrspace(3) %376 to i32, !dbg !17
  %dma.buffer.lds511 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %386), !dbg !17
  %387 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds511, i32 %dma.buffer.offset509, <4 x i32> %dma.buffer.rsrc510) #10, !dbg !17
  %388 = or disjoint i64 %35, 256, !dbg !17
  br label %355, !dbg !17

389:                                              ; preds = %355
  %lds.const.gep497 = getelementptr i8, ptr addrspace(3) %address.local55, i64 0, !dbg !12
  %390 = load <4 x half>, ptr addrspace(3) %lds.const.gep497, align 8, !dbg !12
  %lds.const.gep498 = getelementptr i8, ptr addrspace(3) %address.local55, i64 2048, !dbg !12
  %391 = load <4 x half>, ptr addrspace(3) %lds.const.gep498, align 8, !dbg !12
  %392 = fmul <4 x half> %390, splat (half 0xH3015), !dbg !12
  %393 = fmul <4 x half> %391, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %394 = add nuw nsw i64 %92, %15, !dbg !18
  %395 = shl i64 %394, 19, !dbg !18
  %396 = and i64 %395, 16252928, !dbg !18
  %397 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %396, !dbg !18
  %398 = getelementptr [2 x i8], ptr addrspace(1) %397, i64 %98, !dbg !18
  br label %399, !dbg !18

399:                                              ; preds = %402, %389
  %400 = phi i64 [ %428, %402 ], [ %35, %389 ], !dbg !18
  %401 = icmp samesign ult i64 %400, 128, !dbg !18
  br i1 %401, label %402, label %429, !dbg !18

402:                                              ; preds = %399
  %403 = shl nuw nsw i64 %400, 3, !dbg !18
  %404 = and i64 %403, 24, !dbg !18
  %405 = shl nuw nsw i64 %400, 3, !dbg !18
  %406 = and i64 %405, 512, !dbg !18
  %407 = shl nuw nsw i64 %400, 5, !dbg !18
  %408 = and i64 %407, 3968, !dbg !18
  %409 = getelementptr [2 x i8], ptr addrspace(1) %398, i64 %408, !dbg !18
  %410 = getelementptr [2 x i8], ptr addrspace(1) %409, i64 %404, !dbg !18
  %411 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %406, !dbg !18
  %412 = ptrtoint ptr addrspace(3) %411 to i64, !dbg !18
  %413 = trunc i64 %412 to i32, !dbg !18
  %414 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %413), !dbg !18
  %415 = zext i32 %414 to i64, !dbg !18
  %416 = inttoptr i64 %415 to ptr addrspace(3), !dbg !18
  %417 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %418 = ptrtoint ptr addrspace(1) %410 to i64, !dbg !18
  %419 = sub i64 %418, %417, !dbg !18
  %dma.buffer.offset512 = trunc i64 %419 to i32, !dbg !18
  %420 = trunc i64 %417 to i32, !dbg !18
  %421 = insertelement <4 x i32> poison, i32 %420, i64 0, !dbg !18
  %422 = lshr i64 %417, 32, !dbg !18
  %423 = trunc i64 %422 to i32, !dbg !18
  %424 = insertelement <4 x i32> %421, i32 %423, i64 1, !dbg !18
  %425 = insertelement <4 x i32> %424, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc513 = insertelement <4 x i32> %425, i32 131072, i64 3, !dbg !18
  %426 = ptrtoint ptr addrspace(3) %416 to i32, !dbg !18
  %dma.buffer.lds514 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %426), !dbg !18
  %427 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds514, i32 %dma.buffer.offset512, <4 x i32> %dma.buffer.rsrc513) #10, !dbg !18
  %428 = or disjoint i64 %35, 256, !dbg !18
  br label %399, !dbg !18

429:                                              ; preds = %399
  %lds.const.gep499 = getelementptr i8, ptr addrspace(3) %address.local70, i64 0, !dbg !16
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep499, align 8, !dbg !16
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %392, <4 x half> %430, <4 x float> %314) #9, !dbg !16
  %lds.const.gep500 = getelementptr i8, ptr addrspace(3) %address.local70, i64 1024, !dbg !16
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep500, align 8, !dbg !16
  %lds.const.gep501 = getelementptr i8, ptr addrspace(3) %address.local83, i64 0, !dbg !16
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep501, align 8, !dbg !16
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %393, <4 x half> %432, <4 x float> %431) #9, !dbg !16
  %lds.const.gep502 = getelementptr i8, ptr addrspace(3) %address.local83, i64 1024, !dbg !16
  %435 = load <4 x half>, ptr addrspace(3) %lds.const.gep502, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %392, <4 x half> %433, <4 x float> %317) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !22
  %437 = extractelement <4 x float> %434, i64 0, !dbg !22
  %438 = extractelement <4 x float> %434, i64 1, !dbg !22
  %439 = select i1 %.not, float %438, float %437, !dbg !13
  %440 = extractelement <4 x float> %434, i64 2, !dbg !22
  %441 = extractelement <4 x float> %434, i64 3, !dbg !22
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %393, <4 x half> %435, <4 x float> %436) #9, !dbg !16
  %443 = extractelement <4 x float> %442, i64 0, !dbg !22
  %444 = extractelement <4 x float> %442, i64 1, !dbg !22
  %445 = extractelement <4 x float> %442, i64 2, !dbg !22
  %446 = extractelement <4 x float> %442, i64 3, !dbg !22
  %447 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %448 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %447), !dbg !13
  %449 = add nuw nsw i32 %448, 64, !dbg !13
  %450 = xor i32 %448, 16, !dbg !13
  %451 = and i32 %449, 192, !dbg !13
  %452 = icmp samesign ult i32 %450, %451, !dbg !13
  %453 = select i1 %452, i32 %450, i32 %448, !dbg !13
  %454 = shl nuw nsw i32 %453, 2, !dbg !13
  %455 = bitcast float %439 to i32, !dbg !13
  %456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %455), !dbg !13
  %457 = bitcast i32 %456 to float, !dbg !13
  %458 = select i1 %.not, float %437, float %457, !dbg !13
  %459 = select i1 %.not, float %457, float %438, !dbg !13
  %460 = select i1 %.not, float %441, float %440, !dbg !13
  %461 = bitcast float %460 to i32, !dbg !13
  %462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %461), !dbg !13
  %463 = bitcast i32 %462 to float, !dbg !13
  %464 = select i1 %.not, float %440, float %463, !dbg !13
  %465 = select i1 %.not, float %463, float %441, !dbg !13
  %466 = select i1 %.not1, float %464, float %458, !dbg !13
  %467 = xor i32 %448, 32, !dbg !13
  %468 = icmp samesign ult i32 %467, %451, !dbg !13
  %469 = select i1 %468, i32 %467, i32 %448, !dbg !13
  %470 = shl nuw nsw i32 %469, 2, !dbg !13
  %471 = bitcast float %466 to i32, !dbg !13
  %472 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %471), !dbg !13
  %473 = bitcast i32 %472 to float, !dbg !13
  %474 = select i1 %.not1, float %458, float %473, !dbg !13
  %475 = select i1 %.not1, float %473, float %464, !dbg !13
  %476 = select i1 %.not1, float %465, float %459, !dbg !13
  %477 = bitcast float %476 to i32, !dbg !13
  %478 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %477), !dbg !13
  %479 = bitcast i32 %478 to float, !dbg !13
  %480 = select i1 %.not1, float %459, float %479, !dbg !13
  %481 = select i1 %.not1, float %479, float %465, !dbg !13
  %482 = select i1 %.not, float %444, float %443, !dbg !13
  %483 = bitcast float %482 to i32, !dbg !13
  %484 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %483), !dbg !13
  %485 = bitcast i32 %484 to float, !dbg !13
  %486 = select i1 %.not, float %443, float %485, !dbg !13
  %487 = select i1 %.not, float %485, float %444, !dbg !13
  %488 = select i1 %.not, float %446, float %445, !dbg !13
  %489 = bitcast float %488 to i32, !dbg !13
  %490 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %489), !dbg !13
  %491 = bitcast i32 %490 to float, !dbg !13
  %492 = select i1 %.not, float %445, float %491, !dbg !13
  %493 = select i1 %.not, float %491, float %446, !dbg !13
  %494 = select i1 %.not1, float %492, float %486, !dbg !13
  %495 = bitcast float %494 to i32, !dbg !13
  %496 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %495), !dbg !13
  %497 = bitcast i32 %496 to float, !dbg !13
  %498 = select i1 %.not1, float %486, float %497, !dbg !13
  %499 = select i1 %.not1, float %497, float %492, !dbg !13
  %500 = select i1 %.not1, float %493, float %487, !dbg !13
  %501 = bitcast float %500 to i32, !dbg !13
  %502 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %501), !dbg !13
  %503 = bitcast i32 %502 to float, !dbg !13
  %504 = select i1 %.not1, float %487, float %503, !dbg !13
  %505 = select i1 %.not1, float %503, float %493, !dbg !13
  %506 = or disjoint i64 %75, 31, !dbg !14
  %507 = icmp uge i64 %19, %506, !dbg !14
  %508 = and i1 %43, %507, !dbg !14
  br i1 %508, label %509, label %510, !dbg !14

509:                                              ; preds = %429
  br label %536, !dbg !14

510:                                              ; preds = %429
  %511 = add i64 %19, %35, !dbg !14
  %512 = shl nuw nsw i64 %36, 4, !dbg !14
  %513 = add i64 %511, %512, !dbg !14
  %514 = and i32 %34, 240, !dbg !14
  %index.compact = zext nneg i32 %514 to i64, !dbg !14
  %515 = sub i64 %513, %index.compact, !dbg !14
  %516 = add i64 %515, 1, !dbg !23
  %517 = or disjoint i64 %75, %38, !dbg !14
  %518 = lshr i64 %516, 32, !dbg !24
  %compare.high = trunc i64 %518 to i32, !dbg !24
  %519 = lshr i64 %517, 32, !dbg !24
  %compare.high455 = trunc i64 %519 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high455, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high455, !dbg !24
  %compare.low.base = trunc i64 %516 to i32, !dbg !24
  %compare.low.base456 = trunc i64 %517 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base456, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %520 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base456, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low457 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result458 = select i1 %compare.high.equal, i1 %compare.low457, i1 %compare.high.order, !dbg !24
  %521 = select i1 %compare.result458, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base456, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local460 = or i32 %compare.local.base459, 2, !dbg !24
  %compare.low461 = icmp ugt i32 %compare.low.base, %compare.local460, !dbg !24
  %compare.result462 = select i1 %compare.high.equal, i1 %compare.low461, i1 %compare.high.order, !dbg !24
  %522 = select i1 %compare.result462, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base456, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local464 = or i32 %compare.local.base463, 3, !dbg !24
  %compare.low465 = icmp ugt i32 %compare.low.base, %compare.local464, !dbg !24
  %compare.result466 = select i1 %compare.high.equal, i1 %compare.low465, i1 %compare.high.order, !dbg !24
  %523 = select i1 %compare.result466, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base456, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local468 = or i32 %compare.local.base467, 16, !dbg !24
  %compare.low469 = icmp ugt i32 %compare.low.base, %compare.local468, !dbg !24
  %compare.result470 = select i1 %compare.high.equal, i1 %compare.low469, i1 %compare.high.order, !dbg !24
  %524 = select i1 %compare.result470, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base456, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local472 = or i32 %compare.local.base471, 17, !dbg !24
  %compare.low473 = icmp ugt i32 %compare.low.base, %compare.local472, !dbg !24
  %compare.result474 = select i1 %compare.high.equal, i1 %compare.low473, i1 %compare.high.order, !dbg !24
  %525 = select i1 %compare.result474, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base456, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local476 = or i32 %compare.local.base475, 18, !dbg !24
  %compare.low477 = icmp ugt i32 %compare.low.base, %compare.local476, !dbg !24
  %compare.result478 = select i1 %compare.high.equal, i1 %compare.low477, i1 %compare.high.order, !dbg !24
  %526 = select i1 %compare.result478, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base456, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local480 = or i32 %compare.local.base479, 19, !dbg !24
  %compare.low481 = icmp ugt i32 %compare.low.base, %compare.local480, !dbg !24
  %compare.result482 = select i1 %compare.high.equal, i1 %compare.low481, i1 %compare.high.order, !dbg !24
  %527 = select i1 %compare.result482, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %528 = insertelement <8 x float> poison, float %520, i64 0, !dbg !14
  %529 = insertelement <8 x float> %528, float %521, i64 1, !dbg !14
  %530 = insertelement <8 x float> %529, float %522, i64 2, !dbg !14
  %531 = insertelement <8 x float> %530, float %523, i64 3, !dbg !14
  %532 = insertelement <8 x float> %531, float %524, i64 4, !dbg !14
  %533 = insertelement <8 x float> %532, float %525, i64 5, !dbg !14
  %534 = insertelement <8 x float> %533, float %526, i64 6, !dbg !14
  %535 = insertelement <8 x float> %534, float %527, i64 7, !dbg !14
  br label %536, !dbg !14

536:                                              ; preds = %509, %510
  %537 = phi <8 x float> [ %535, %510 ], [ zeroinitializer, %509 ], !dbg !14
  br label %538, !dbg !14

538:                                              ; preds = %536
  %539 = insertelement <4 x float> poison, float %474, i64 0, !dbg !26
  %540 = insertelement <4 x float> %539, float %480, i64 1, !dbg !26
  %541 = insertelement <4 x float> %540, float %475, i64 2, !dbg !26
  %542 = insertelement <4 x float> %541, float %481, i64 3, !dbg !26
  %543 = shufflevector <8 x float> %537, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %544 = fadd <4 x float> %542, %543, !dbg !26
  %545 = extractelement <4 x float> %544, i64 0, !dbg !26
  %546 = extractelement <4 x float> %544, i64 1, !dbg !26
  %547 = extractelement <4 x float> %544, i64 2, !dbg !26
  %548 = extractelement <4 x float> %544, i64 3, !dbg !26
  %549 = insertelement <4 x float> poison, float %498, i64 0, !dbg !26
  %550 = insertelement <4 x float> %549, float %504, i64 1, !dbg !26
  %551 = insertelement <4 x float> %550, float %499, i64 2, !dbg !26
  %552 = insertelement <4 x float> %551, float %505, i64 3, !dbg !26
  %553 = shufflevector <8 x float> %537, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %554 = fadd <4 x float> %552, %553, !dbg !26
  %555 = extractelement <4 x float> %554, i64 0, !dbg !26
  %556 = extractelement <4 x float> %554, i64 1, !dbg !26
  %557 = extractelement <4 x float> %554, i64 2, !dbg !26
  %558 = extractelement <4 x float> %554, i64 3, !dbg !26
  %559 = call float @llvm.maxnum.f32(float %545, float 0xFFF0000000000000), !dbg !27
  %560 = call float @llvm.maxnum.f32(float %559, float %546), !dbg !27
  %561 = call float @llvm.maxnum.f32(float %560, float %547), !dbg !27
  %562 = call float @llvm.maxnum.f32(float %561, float %548), !dbg !27
  %563 = call float @llvm.maxnum.f32(float %562, float %555), !dbg !27
  %564 = call float @llvm.maxnum.f32(float %563, float %556), !dbg !27
  %565 = call float @llvm.maxnum.f32(float %564, float %557), !dbg !27
  %566 = call float @llvm.maxnum.f32(float %565, float %558), !dbg !27
  %567 = bitcast float %566 to i32, !dbg !27
  %568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %567), !dbg !27
  %569 = bitcast i32 %568 to float, !dbg !27
  %570 = call float @llvm.maxnum.f32(float %566, float %569), !dbg !27
  %571 = bitcast float %570 to i32, !dbg !27
  %572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %571), !dbg !27
  %573 = bitcast i32 %572 to float, !dbg !27
  %574 = call float @llvm.maxnum.f32(float %570, float %573), !dbg !27
  %575 = insertelement <1 x float> poison, float %574, i64 0, !dbg !27
  %576 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %575, <1 x float> %63), !dbg !27
  %577 = shufflevector <1 x float> %576, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %578 = fsub <4 x float> %544, %577, !dbg !28
  %579 = fsub <4 x float> %554, %577, !dbg !28
  %580 = fsub <1 x float> %63, %576, !dbg !29
  %581 = extractelement <1 x float> %580, i64 0, !dbg !30
  %582 = call float @llvm.amdgcn.exp2.f32(float %581), !dbg !30
  %583 = extractelement <1 x float> %62, i64 0, !dbg !31
  %584 = fmul float %583, %582, !dbg !31
  %585 = extractelement <4 x float> %578, i64 0, !dbg !32
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !32
  %587 = insertelement <4 x float> poison, float %586, i64 0, !dbg !34
  %588 = extractelement <4 x float> %578, i64 1, !dbg !32
  %589 = call float @llvm.amdgcn.exp2.f32(float %588), !dbg !32
  %590 = insertelement <4 x float> %587, float %589, i64 1, !dbg !34
  %591 = fadd float %586, %589, !dbg !33
  %592 = extractelement <4 x float> %578, i64 2, !dbg !32
  %593 = call float @llvm.amdgcn.exp2.f32(float %592), !dbg !32
  %594 = insertelement <4 x float> %590, float %593, i64 2, !dbg !34
  %595 = fadd float %591, %593, !dbg !33
  %596 = extractelement <4 x float> %578, i64 3, !dbg !32
  %597 = call float @llvm.amdgcn.exp2.f32(float %596), !dbg !32
  %598 = insertelement <4 x float> %594, float %597, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %598) #9, !dbg !34, !frisk.register_lifetimes !1
  %599 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %599) #9, !dbg !33, !frisk.register_lifetimes !1
  %600 = fadd float %595, %597, !dbg !33
  %601 = extractelement <4 x float> %579, i64 0, !dbg !32
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !32
  %603 = insertelement <4 x float> poison, float %602, i64 0, !dbg !34
  %604 = fadd float %600, %602, !dbg !33
  %605 = extractelement <4 x float> %579, i64 1, !dbg !32
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !32
  %607 = insertelement <4 x float> %603, float %606, i64 1, !dbg !34
  %608 = fadd float %604, %606, !dbg !33
  %609 = extractelement <4 x float> %579, i64 2, !dbg !32
  %610 = call float @llvm.amdgcn.exp2.f32(float %609), !dbg !32
  %611 = insertelement <4 x float> %607, float %610, i64 2, !dbg !34
  %612 = fadd float %608, %610, !dbg !33
  %613 = extractelement <4 x float> %579, i64 3, !dbg !32
  %614 = call float @llvm.amdgcn.exp2.f32(float %613), !dbg !32
  %615 = insertelement <4 x float> %611, float %614, i64 3, !dbg !34
  %math.result29 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %615) #9, !dbg !34, !frisk.register_lifetimes !1
  %616 = fptrunc <4 x float> %math.result29 to <4 x half>, !dbg !34
  %math.packed30 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %616) #9, !dbg !33, !frisk.register_lifetimes !1
  %617 = fadd float %612, %614, !dbg !33
  %618 = bitcast float %617 to i32, !dbg !33
  %619 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %618), !dbg !33
  %620 = bitcast i32 %619 to float, !dbg !33
  %621 = fadd float %617, %620, !dbg !33
  %622 = bitcast float %621 to i32, !dbg !33
  %623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %622), !dbg !33
  %624 = bitcast i32 %623 to float, !dbg !33
  %625 = fadd float %621, %624, !dbg !33
  %626 = fadd float %625, %584, !dbg !33
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread280 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local281 = shl nuw nsw i32 %address.thread280, 3, !dbg !15
  %address.local282 = and i32 %address.local281, 504, !dbg !15
  %address.local283 = zext nneg i32 %address.local282 to i64, !dbg !15
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local283, !dbg !15
  %627 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local284, i32 0) #0, !dbg !15
  %628 = bitcast <4 x i32> %627 to <8 x half>, !dbg !15
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local283, !dbg !15
  %address.local286 = getelementptr i8, ptr addrspace(3) %address.local285, i64 1024, !dbg !15
  %629 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local286, i32 0) #0, !dbg !15
  %630 = bitcast <4 x i32> %629 to <8 x half>, !dbg !15
  %631 = add nuw nsw i64 %92, %15, !dbg !18
  %632 = shl i64 %631, 19, !dbg !18
  %633 = and i64 %632, 16252928, !dbg !18
  %634 = add i64 %633, %98, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %635 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %634, !dbg !18
  %636 = getelementptr i8, ptr addrspace(1) %635, i64 64, !dbg !18
  br label %637, !dbg !18

637:                                              ; preds = %640, %538
  %638 = phi i64 [ %666, %640 ], [ %35, %538 ], !dbg !18
  %639 = icmp samesign ult i64 %638, 128, !dbg !18
  br i1 %639, label %640, label %667, !dbg !18

640:                                              ; preds = %637
  %641 = shl nuw nsw i64 %638, 3, !dbg !18
  %642 = and i64 %641, 24, !dbg !18
  %643 = shl nuw nsw i64 %638, 3, !dbg !18
  %644 = and i64 %643, 512, !dbg !18
  %645 = shl nuw nsw i64 %638, 5, !dbg !18
  %646 = and i64 %645, 3968, !dbg !18
  %647 = getelementptr [2 x i8], ptr addrspace(1) %636, i64 %646, !dbg !18
  %648 = getelementptr [2 x i8], ptr addrspace(1) %647, i64 %642, !dbg !18
  %649 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %644, !dbg !18
  %650 = ptrtoint ptr addrspace(3) %649 to i64, !dbg !18
  %651 = trunc i64 %650 to i32, !dbg !18
  %652 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %651), !dbg !18
  %653 = zext i32 %652 to i64, !dbg !18
  %654 = inttoptr i64 %653 to ptr addrspace(3), !dbg !18
  %655 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %656 = ptrtoint ptr addrspace(1) %648 to i64, !dbg !18
  %657 = sub i64 %656, %655, !dbg !18
  %dma.buffer.offset515 = trunc i64 %657 to i32, !dbg !18
  %658 = trunc i64 %655 to i32, !dbg !18
  %659 = insertelement <4 x i32> poison, i32 %658, i64 0, !dbg !18
  %660 = lshr i64 %655, 32, !dbg !18
  %661 = trunc i64 %660 to i32, !dbg !18
  %662 = insertelement <4 x i32> %659, i32 %661, i64 1, !dbg !18
  %663 = insertelement <4 x i32> %662, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc516 = insertelement <4 x i32> %663, i32 131072, i64 3, !dbg !18
  %664 = ptrtoint ptr addrspace(3) %654 to i32, !dbg !18
  %dma.buffer.lds517 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %664), !dbg !18
  %665 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds517, i32 %dma.buffer.offset515, <4 x i32> %dma.buffer.rsrc516) #10, !dbg !18
  %666 = or disjoint i64 %35, 256, !dbg !18
  br label %637, !dbg !18

667:                                              ; preds = %637
  %668 = shufflevector <8 x half> %628, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %669 = insertelement <4 x float> poison, float %582, i64 0, !dbg !35
  %670 = shufflevector <4 x float> %669, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %671 = fmul <4 x float> %59, %670, !dbg !35
  %672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %668, <4 x float> %671) #9, !dbg !15
  %673 = shufflevector <8 x half> %630, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %673, <4 x float> %672) #9, !dbg !15
  %675 = shufflevector <8 x half> %628, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %676 = fmul <4 x float> %60, %670, !dbg !35
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %675, <4 x float> %676) #9, !dbg !15
  %678 = shufflevector <8 x half> %630, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %678, <4 x float> %677) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread287 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local288 = shl nuw nsw i32 %address.thread287, 3, !dbg !15
  %address.local289 = and i32 %address.local288, 504, !dbg !15
  %address.local290 = zext nneg i32 %address.local289 to i64, !dbg !15
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local290, !dbg !15
  %680 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local291, i32 0) #0, !dbg !15
  %681 = bitcast <4 x i32> %680 to <8 x half>, !dbg !15
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local290, !dbg !15
  %address.local293 = getelementptr i8, ptr addrspace(3) %address.local292, i64 1024, !dbg !15
  %682 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local293, i32 0) #0, !dbg !15
  %683 = bitcast <4 x i32> %682 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %684 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %634, !dbg !18
  %685 = getelementptr i8, ptr addrspace(1) %684, i64 128, !dbg !18
  br label %686, !dbg !18

686:                                              ; preds = %689, %667
  %687 = phi i64 [ %715, %689 ], [ %35, %667 ], !dbg !18
  %688 = icmp samesign ult i64 %687, 128, !dbg !18
  br i1 %688, label %689, label %716, !dbg !18

689:                                              ; preds = %686
  %690 = shl nuw nsw i64 %687, 3, !dbg !18
  %691 = and i64 %690, 24, !dbg !18
  %692 = shl nuw nsw i64 %687, 3, !dbg !18
  %693 = and i64 %692, 512, !dbg !18
  %694 = shl nuw nsw i64 %687, 5, !dbg !18
  %695 = and i64 %694, 3968, !dbg !18
  %696 = getelementptr [2 x i8], ptr addrspace(1) %685, i64 %695, !dbg !18
  %697 = getelementptr [2 x i8], ptr addrspace(1) %696, i64 %691, !dbg !18
  %698 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %693, !dbg !18
  %699 = ptrtoint ptr addrspace(3) %698 to i64, !dbg !18
  %700 = trunc i64 %699 to i32, !dbg !18
  %701 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %700), !dbg !18
  %702 = zext i32 %701 to i64, !dbg !18
  %703 = inttoptr i64 %702 to ptr addrspace(3), !dbg !18
  %704 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %705 = ptrtoint ptr addrspace(1) %697 to i64, !dbg !18
  %706 = sub i64 %705, %704, !dbg !18
  %dma.buffer.offset518 = trunc i64 %706 to i32, !dbg !18
  %707 = trunc i64 %704 to i32, !dbg !18
  %708 = insertelement <4 x i32> poison, i32 %707, i64 0, !dbg !18
  %709 = lshr i64 %704, 32, !dbg !18
  %710 = trunc i64 %709 to i32, !dbg !18
  %711 = insertelement <4 x i32> %708, i32 %710, i64 1, !dbg !18
  %712 = insertelement <4 x i32> %711, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc519 = insertelement <4 x i32> %712, i32 131072, i64 3, !dbg !18
  %713 = ptrtoint ptr addrspace(3) %703 to i32, !dbg !18
  %dma.buffer.lds520 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %713), !dbg !18
  %714 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds520, i32 %dma.buffer.offset518, <4 x i32> %dma.buffer.rsrc519) #10, !dbg !18
  %715 = or disjoint i64 %35, 256, !dbg !18
  br label %686, !dbg !18

716:                                              ; preds = %686
  %717 = shufflevector <8 x half> %681, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %718 = fmul <4 x float> %64, %670, !dbg !35
  %719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %717, <4 x float> %718) #9, !dbg !15
  %720 = shufflevector <8 x half> %683, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %720, <4 x float> %719) #9, !dbg !15
  %722 = shufflevector <8 x half> %681, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %723 = fmul <4 x float> %65, %670, !dbg !35
  %724 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %722, <4 x float> %723) #9, !dbg !15
  %725 = shufflevector <8 x half> %683, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %726 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %725, <4 x float> %724) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread294 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local295 = shl nuw nsw i32 %address.thread294, 3, !dbg !15
  %address.local296 = and i32 %address.local295, 504, !dbg !15
  %address.local297 = zext nneg i32 %address.local296 to i64, !dbg !15
  %address.local298 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local297, !dbg !15
  %727 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local298, i32 0) #0, !dbg !15
  %728 = bitcast <4 x i32> %727 to <8 x half>, !dbg !15
  %address.local299 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local297, !dbg !15
  %address.local300 = getelementptr i8, ptr addrspace(3) %address.local299, i64 1024, !dbg !15
  %729 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local300, i32 0) #0, !dbg !15
  %730 = bitcast <4 x i32> %729 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %731 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %634, !dbg !18
  %732 = getelementptr i8, ptr addrspace(1) %731, i64 192, !dbg !18
  br label %733, !dbg !18

733:                                              ; preds = %736, %716
  %734 = phi i64 [ %762, %736 ], [ %35, %716 ], !dbg !18
  %735 = icmp samesign ult i64 %734, 128, !dbg !18
  br i1 %735, label %736, label %763, !dbg !18

736:                                              ; preds = %733
  %737 = shl nuw nsw i64 %734, 3, !dbg !18
  %738 = and i64 %737, 24, !dbg !18
  %739 = shl nuw nsw i64 %734, 3, !dbg !18
  %740 = and i64 %739, 512, !dbg !18
  %741 = shl nuw nsw i64 %734, 5, !dbg !18
  %742 = and i64 %741, 3968, !dbg !18
  %743 = getelementptr [2 x i8], ptr addrspace(1) %732, i64 %742, !dbg !18
  %744 = getelementptr [2 x i8], ptr addrspace(1) %743, i64 %738, !dbg !18
  %745 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %740, !dbg !18
  %746 = ptrtoint ptr addrspace(3) %745 to i64, !dbg !18
  %747 = trunc i64 %746 to i32, !dbg !18
  %748 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %747), !dbg !18
  %749 = zext i32 %748 to i64, !dbg !18
  %750 = inttoptr i64 %749 to ptr addrspace(3), !dbg !18
  %751 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %752 = ptrtoint ptr addrspace(1) %744 to i64, !dbg !18
  %753 = sub i64 %752, %751, !dbg !18
  %dma.buffer.offset521 = trunc i64 %753 to i32, !dbg !18
  %754 = trunc i64 %751 to i32, !dbg !18
  %755 = insertelement <4 x i32> poison, i32 %754, i64 0, !dbg !18
  %756 = lshr i64 %751, 32, !dbg !18
  %757 = trunc i64 %756 to i32, !dbg !18
  %758 = insertelement <4 x i32> %755, i32 %757, i64 1, !dbg !18
  %759 = insertelement <4 x i32> %758, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc522 = insertelement <4 x i32> %759, i32 131072, i64 3, !dbg !18
  %760 = ptrtoint ptr addrspace(3) %750 to i32, !dbg !18
  %dma.buffer.lds523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %760), !dbg !18
  %761 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds523, i32 %dma.buffer.offset521, <4 x i32> %dma.buffer.rsrc522) #10, !dbg !18
  %762 = or disjoint i64 %35, 256, !dbg !18
  br label %733, !dbg !18

763:                                              ; preds = %733
  %764 = shufflevector <8 x half> %728, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %765 = fmul <4 x float> %67, %670, !dbg !35
  %766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %764, <4 x float> %765) #9, !dbg !15
  %767 = shufflevector <8 x half> %730, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %767, <4 x float> %766) #9, !dbg !15
  %769 = shufflevector <8 x half> %728, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %770 = fmul <4 x float> %68, %670, !dbg !35
  %771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %769, <4 x float> %770) #9, !dbg !15
  %772 = shufflevector <8 x half> %730, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %772, <4 x float> %771) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread301 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local302 = shl nuw nsw i32 %address.thread301, 3, !dbg !15
  %address.local303 = and i32 %address.local302, 504, !dbg !15
  %address.local304 = zext nneg i32 %address.local303 to i64, !dbg !15
  %address.local305 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local304, !dbg !15
  %774 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local305, i32 0) #0, !dbg !15
  %775 = bitcast <4 x i32> %774 to <8 x half>, !dbg !15
  %address.local306 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local304, !dbg !15
  %address.local307 = getelementptr i8, ptr addrspace(3) %address.local306, i64 1024, !dbg !15
  %776 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local307, i32 0) #0, !dbg !15
  %777 = bitcast <4 x i32> %776 to <8 x half>, !dbg !15
  %778 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %779 = fmul <4 x float> %70, %670, !dbg !35
  %780 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %778, <4 x float> %779) #9, !dbg !15
  %781 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %781, <4 x float> %780) #9, !dbg !15
  %783 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %784 = fmul <4 x float> %71, %670, !dbg !35
  %785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %783, <4 x float> %784) #9, !dbg !15
  %786 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %787 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %786, <4 x float> %785) #9, !dbg !15
  %788 = shufflevector <4 x float> %674, <4 x float> %679, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %789 = insertelement <1 x float> poison, float %626, i64 0, !dbg !7
  %790 = shufflevector <4 x float> %721, <4 x float> %726, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %791 = shufflevector <4 x float> %768, <4 x float> %773, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %792 = shufflevector <4 x float> %782, <4 x float> %787, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %793 = add nuw nsw i64 %58, 1, !dbg !7
  br label %57, !dbg !7

794:                                              ; preds = %57
  %795 = extractelement <8 x float> %72, i64 0, !dbg !7
  %796 = extractelement <8 x float> %72, i64 1, !dbg !7
  %797 = extractelement <8 x float> %72, i64 2, !dbg !7
  %798 = extractelement <8 x float> %72, i64 3, !dbg !7
  %799 = select i1 %.not, float %796, float %795, !dbg !7
  %800 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %801 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %800), !dbg !7
  %802 = add nuw nsw i32 %801, 64, !dbg !7
  %803 = xor i32 %801, 16, !dbg !7
  %804 = and i32 %802, 192, !dbg !7
  %805 = icmp samesign ult i32 %803, %804, !dbg !7
  %806 = select i1 %805, i32 %803, i32 %801, !dbg !7
  %807 = shl nuw nsw i32 %806, 2, !dbg !7
  %808 = bitcast float %799 to i32, !dbg !7
  %809 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %808), !dbg !7
  %810 = bitcast i32 %809 to float, !dbg !7
  %811 = select i1 %.not, float %795, float %810, !dbg !7
  %812 = select i1 %.not, float %810, float %796, !dbg !7
  %813 = select i1 %.not, float %798, float %797, !dbg !7
  %814 = bitcast float %813 to i32, !dbg !7
  %815 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %814), !dbg !7
  %816 = bitcast i32 %815 to float, !dbg !7
  %817 = select i1 %.not, float %797, float %816, !dbg !7
  %818 = select i1 %.not, float %816, float %798, !dbg !7
  %819 = select i1 %.not1, float %817, float %811, !dbg !7
  %820 = xor i32 %801, 32, !dbg !7
  %821 = icmp samesign ult i32 %820, %804, !dbg !7
  %822 = select i1 %821, i32 %820, i32 %801, !dbg !7
  %823 = shl nuw nsw i32 %822, 2, !dbg !7
  %824 = bitcast float %819 to i32, !dbg !7
  %825 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %824), !dbg !7
  %826 = bitcast i32 %825 to float, !dbg !7
  %827 = select i1 %.not1, float %811, float %826, !dbg !7
  %828 = select i1 %.not1, float %826, float %817, !dbg !7
  %829 = select i1 %.not1, float %818, float %812, !dbg !7
  %830 = bitcast float %829 to i32, !dbg !7
  %831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %830), !dbg !7
  %832 = bitcast i32 %831 to float, !dbg !7
  %833 = select i1 %.not1, float %812, float %832, !dbg !7
  %834 = select i1 %.not1, float %832, float %818, !dbg !7
  %835 = extractelement <8 x float> %69, i64 0, !dbg !7
  %836 = extractelement <8 x float> %69, i64 1, !dbg !7
  %837 = extractelement <8 x float> %69, i64 2, !dbg !7
  %838 = extractelement <8 x float> %69, i64 3, !dbg !7
  %839 = select i1 %.not, float %836, float %835, !dbg !7
  %840 = bitcast float %839 to i32, !dbg !7
  %841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %840), !dbg !7
  %842 = bitcast i32 %841 to float, !dbg !7
  %843 = select i1 %.not, float %835, float %842, !dbg !7
  %844 = select i1 %.not, float %842, float %836, !dbg !7
  %845 = select i1 %.not, float %838, float %837, !dbg !7
  %846 = bitcast float %845 to i32, !dbg !7
  %847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %846), !dbg !7
  %848 = bitcast i32 %847 to float, !dbg !7
  %849 = select i1 %.not, float %837, float %848, !dbg !7
  %850 = select i1 %.not, float %848, float %838, !dbg !7
  %851 = select i1 %.not1, float %849, float %843, !dbg !7
  %852 = bitcast float %851 to i32, !dbg !7
  %853 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %852), !dbg !7
  %854 = bitcast i32 %853 to float, !dbg !7
  %855 = select i1 %.not1, float %843, float %854, !dbg !7
  %856 = select i1 %.not1, float %854, float %849, !dbg !7
  %857 = select i1 %.not1, float %850, float %844, !dbg !7
  %858 = bitcast float %857 to i32, !dbg !7
  %859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %858), !dbg !7
  %860 = bitcast i32 %859 to float, !dbg !7
  %861 = select i1 %.not1, float %844, float %860, !dbg !7
  %862 = select i1 %.not1, float %860, float %850, !dbg !7
  %863 = extractelement <8 x float> %66, i64 0, !dbg !7
  %864 = extractelement <8 x float> %66, i64 1, !dbg !7
  %865 = extractelement <8 x float> %66, i64 2, !dbg !7
  %866 = extractelement <8 x float> %66, i64 3, !dbg !7
  %867 = select i1 %.not, float %864, float %863, !dbg !7
  %868 = bitcast float %867 to i32, !dbg !7
  %869 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %868), !dbg !7
  %870 = bitcast i32 %869 to float, !dbg !7
  %871 = select i1 %.not, float %863, float %870, !dbg !7
  %872 = select i1 %.not, float %870, float %864, !dbg !7
  %873 = select i1 %.not, float %866, float %865, !dbg !7
  %874 = bitcast float %873 to i32, !dbg !7
  %875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %874), !dbg !7
  %876 = bitcast i32 %875 to float, !dbg !7
  %877 = select i1 %.not, float %865, float %876, !dbg !7
  %878 = select i1 %.not, float %876, float %866, !dbg !7
  %879 = select i1 %.not1, float %877, float %871, !dbg !7
  %880 = bitcast float %879 to i32, !dbg !7
  %881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %880), !dbg !7
  %882 = bitcast i32 %881 to float, !dbg !7
  %883 = select i1 %.not1, float %871, float %882, !dbg !7
  %884 = select i1 %.not1, float %882, float %877, !dbg !7
  %885 = select i1 %.not1, float %878, float %872, !dbg !7
  %886 = bitcast float %885 to i32, !dbg !7
  %887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %886), !dbg !7
  %888 = bitcast i32 %887 to float, !dbg !7
  %889 = select i1 %.not1, float %872, float %888, !dbg !7
  %890 = select i1 %.not1, float %888, float %878, !dbg !7
  %891 = extractelement <8 x float> %61, i64 0, !dbg !7
  %892 = extractelement <8 x float> %61, i64 1, !dbg !7
  %893 = extractelement <8 x float> %61, i64 2, !dbg !7
  %894 = extractelement <8 x float> %61, i64 3, !dbg !7
  %895 = select i1 %.not, float %892, float %891, !dbg !7
  %896 = bitcast float %895 to i32, !dbg !7
  %897 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %896), !dbg !7
  %898 = bitcast i32 %897 to float, !dbg !7
  %899 = select i1 %.not, float %891, float %898, !dbg !7
  %900 = select i1 %.not, float %898, float %892, !dbg !7
  %901 = select i1 %.not, float %894, float %893, !dbg !7
  %902 = bitcast float %901 to i32, !dbg !7
  %903 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %902), !dbg !7
  %904 = bitcast i32 %903 to float, !dbg !7
  %905 = select i1 %.not, float %893, float %904, !dbg !7
  %906 = select i1 %.not, float %904, float %894, !dbg !7
  %907 = select i1 %.not1, float %905, float %899, !dbg !7
  %908 = bitcast float %907 to i32, !dbg !7
  %909 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %908), !dbg !7
  %910 = select i1 %.not1, float %906, float %900, !dbg !7
  %911 = bitcast float %910 to i32, !dbg !7
  %912 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %911), !dbg !7
  %913 = shufflevector <1 x float> %62, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %914 = extractelement <8 x float> %72, i64 4, !dbg !7
  %915 = extractelement <8 x float> %72, i64 5, !dbg !7
  %916 = extractelement <8 x float> %72, i64 6, !dbg !7
  %917 = extractelement <8 x float> %72, i64 7, !dbg !7
  %918 = select i1 %.not, float %915, float %914, !dbg !7
  %919 = bitcast float %918 to i32, !dbg !7
  %920 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %919), !dbg !7
  %921 = bitcast i32 %920 to float, !dbg !7
  %922 = select i1 %.not, float %914, float %921, !dbg !7
  %923 = select i1 %.not, float %921, float %915, !dbg !7
  %924 = select i1 %.not, float %917, float %916, !dbg !7
  %925 = bitcast float %924 to i32, !dbg !7
  %926 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %925), !dbg !7
  %927 = bitcast i32 %926 to float, !dbg !7
  %928 = select i1 %.not, float %916, float %927, !dbg !7
  %929 = select i1 %.not, float %927, float %917, !dbg !7
  %930 = select i1 %.not1, float %928, float %922, !dbg !7
  %931 = bitcast float %930 to i32, !dbg !7
  %932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %931), !dbg !7
  %933 = bitcast i32 %932 to float, !dbg !7
  %934 = select i1 %.not1, float %922, float %933, !dbg !7
  %935 = select i1 %.not1, float %933, float %928, !dbg !7
  %936 = select i1 %.not1, float %929, float %923, !dbg !7
  %937 = bitcast float %936 to i32, !dbg !7
  %938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %937), !dbg !7
  %939 = bitcast i32 %938 to float, !dbg !7
  %940 = select i1 %.not1, float %923, float %939, !dbg !7
  %941 = select i1 %.not1, float %939, float %929, !dbg !7
  %942 = extractelement <8 x float> %69, i64 4, !dbg !7
  %943 = extractelement <8 x float> %69, i64 5, !dbg !7
  %944 = extractelement <8 x float> %69, i64 6, !dbg !7
  %945 = extractelement <8 x float> %69, i64 7, !dbg !7
  %946 = select i1 %.not, float %943, float %942, !dbg !7
  %947 = bitcast float %946 to i32, !dbg !7
  %948 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %947), !dbg !7
  %949 = bitcast i32 %948 to float, !dbg !7
  %950 = select i1 %.not, float %942, float %949, !dbg !7
  %951 = select i1 %.not, float %949, float %943, !dbg !7
  %952 = select i1 %.not, float %945, float %944, !dbg !7
  %953 = bitcast float %952 to i32, !dbg !7
  %954 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %953), !dbg !7
  %955 = bitcast i32 %954 to float, !dbg !7
  %956 = select i1 %.not, float %944, float %955, !dbg !7
  %957 = select i1 %.not, float %955, float %945, !dbg !7
  %958 = select i1 %.not1, float %956, float %950, !dbg !7
  %959 = bitcast float %958 to i32, !dbg !7
  %960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %959), !dbg !7
  %961 = bitcast i32 %960 to float, !dbg !7
  %962 = select i1 %.not1, float %950, float %961, !dbg !7
  %963 = select i1 %.not1, float %961, float %956, !dbg !7
  %964 = select i1 %.not1, float %957, float %951, !dbg !7
  %965 = bitcast float %964 to i32, !dbg !7
  %966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %965), !dbg !7
  %967 = bitcast i32 %966 to float, !dbg !7
  %968 = select i1 %.not1, float %951, float %967, !dbg !7
  %969 = select i1 %.not1, float %967, float %957, !dbg !7
  %970 = extractelement <8 x float> %66, i64 4, !dbg !7
  %971 = extractelement <8 x float> %66, i64 5, !dbg !7
  %972 = extractelement <8 x float> %66, i64 6, !dbg !7
  %973 = extractelement <8 x float> %66, i64 7, !dbg !7
  %974 = select i1 %.not, float %971, float %970, !dbg !7
  %975 = bitcast float %974 to i32, !dbg !7
  %976 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %975), !dbg !7
  %977 = bitcast i32 %976 to float, !dbg !7
  %978 = select i1 %.not, float %970, float %977, !dbg !7
  %979 = select i1 %.not, float %977, float %971, !dbg !7
  %980 = select i1 %.not, float %973, float %972, !dbg !7
  %981 = bitcast float %980 to i32, !dbg !7
  %982 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %981), !dbg !7
  %983 = bitcast i32 %982 to float, !dbg !7
  %984 = select i1 %.not, float %972, float %983, !dbg !7
  %985 = select i1 %.not, float %983, float %973, !dbg !7
  %986 = select i1 %.not1, float %984, float %978, !dbg !7
  %987 = bitcast float %986 to i32, !dbg !7
  %988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %987), !dbg !7
  %989 = bitcast i32 %988 to float, !dbg !7
  %990 = select i1 %.not1, float %978, float %989, !dbg !7
  %991 = select i1 %.not1, float %989, float %984, !dbg !7
  %992 = select i1 %.not1, float %985, float %979, !dbg !7
  %993 = bitcast float %992 to i32, !dbg !7
  %994 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %993), !dbg !7
  %995 = bitcast i32 %994 to float, !dbg !7
  %996 = select i1 %.not1, float %979, float %995, !dbg !7
  %997 = select i1 %.not1, float %995, float %985, !dbg !7
  %998 = extractelement <8 x float> %61, i64 4, !dbg !7
  %999 = extractelement <8 x float> %61, i64 5, !dbg !7
  %1000 = extractelement <8 x float> %61, i64 6, !dbg !7
  %1001 = extractelement <8 x float> %61, i64 7, !dbg !7
  %1002 = select i1 %.not, float %999, float %998, !dbg !7
  %1003 = bitcast float %1002 to i32, !dbg !7
  %1004 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %1003), !dbg !7
  %1005 = bitcast i32 %1004 to float, !dbg !7
  %1006 = select i1 %.not, float %998, float %1005, !dbg !7
  %1007 = select i1 %.not, float %1005, float %999, !dbg !7
  %1008 = select i1 %.not, float %1001, float %1000, !dbg !7
  %1009 = bitcast float %1008 to i32, !dbg !7
  %1010 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %1009), !dbg !7
  %1011 = bitcast i32 %1010 to float, !dbg !7
  %1012 = select i1 %.not, float %1000, float %1011, !dbg !7
  %1013 = select i1 %.not, float %1011, float %1001, !dbg !7
  %1014 = select i1 %.not1, float %1012, float %1006, !dbg !7
  %1015 = bitcast float %1014 to i32, !dbg !7
  %1016 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %1015), !dbg !7
  %1017 = select i1 %.not1, float %1013, float %1007, !dbg !7
  %1018 = bitcast float %1017 to i32, !dbg !7
  %1019 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %1018), !dbg !7
  br i1 true, label %1020, label %1054, !dbg !21

1020:                                             ; preds = %794
  %1021 = bitcast i32 %1016 to float, !dbg !7
  %1022 = select i1 %.not1, float %1006, float %1021, !dbg !7
  %1023 = insertelement <4 x float> poison, float %1022, i64 0, !dbg !19
  %1024 = bitcast i32 %1019 to float, !dbg !7
  %1025 = select i1 %.not1, float %1007, float %1024, !dbg !7
  %1026 = insertelement <4 x float> %1023, float %1025, i64 1, !dbg !19
  %1027 = select i1 %.not1, float %1021, float %1012, !dbg !7
  %1028 = insertelement <4 x float> %1026, float %1027, i64 2, !dbg !19
  %1029 = select i1 %.not1, float %1024, float %1013, !dbg !7
  %1030 = insertelement <4 x float> %1028, float %1029, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %913, !dbg !19
  %normalized = fmul <4 x float> %1030, %row.reciprocal, !dbg !19
  %1031 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1032 = extractelement <4 x half> %1031, i64 3, !dbg !20
  %1033 = extractelement <4 x half> %1031, i64 2, !dbg !20
  %1034 = extractelement <4 x half> %1031, i64 1, !dbg !20
  %1035 = extractelement <4 x half> %1031, i64 0, !dbg !20
  %1036 = bitcast i32 %909 to float, !dbg !7
  %1037 = select i1 %.not1, float %899, float %1036, !dbg !7
  %1038 = insertelement <4 x float> poison, float %1037, i64 0, !dbg !19
  %1039 = bitcast i32 %912 to float, !dbg !7
  %1040 = select i1 %.not1, float %900, float %1039, !dbg !7
  %1041 = insertelement <4 x float> %1038, float %1040, i64 1, !dbg !19
  %1042 = select i1 %.not1, float %1036, float %905, !dbg !7
  %1043 = insertelement <4 x float> %1041, float %1042, i64 2, !dbg !19
  %1044 = select i1 %.not1, float %1039, float %906, !dbg !7
  %1045 = insertelement <4 x float> %1043, float %1044, i64 3, !dbg !19
  %normalized15 = fmul <4 x float> %1045, %row.reciprocal, !dbg !19
  %1046 = fptrunc <4 x float> %normalized15 to <4 x half>, !dbg !20
  %1047 = extractelement <4 x half> %1046, i64 3, !dbg !20
  %1048 = extractelement <4 x half> %1046, i64 2, !dbg !20
  %1049 = extractelement <4 x half> %1046, i64 1, !dbg !20
  %1050 = extractelement <4 x half> %1046, i64 0, !dbg !20
  %1051 = add nsw i64 %15, %25, !dbg !21
  %1052 = shl i64 %1051, 19, !dbg !21
  %1053 = and i64 %1052, 16252928, !dbg !21
  %address.thread308 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local309 = zext nneg i32 %address.thread308 to i64, !dbg !12
  %address.local310 = add i64 %28, %address.local309, !dbg !21
  %address.local311 = lshr i64 %address.local309, 6, !dbg !12
  %address.local312 = shl nuw nsw i64 %address.local311, 4, !dbg !21
  %address.local313 = add i64 %address.local310, %address.local312, !dbg !21
  %address.local314 = and i64 %address.local309, 240, !dbg !21
  %address.local315 = sub i64 %address.local313, %address.local314, !dbg !21
  %address.local316 = add i64 %address.local315, %29, !dbg !21
  %address.local317 = shl nuw nsw i64 %address.local316, 7, !dbg !21
  %address.local318 = add nuw nsw i64 %1053, %address.local317, !dbg !21
  %address.local319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local318, !dbg !21
  %address.local320 = lshr i64 %address.local309, 2, !dbg !12
  %address.local321 = and i64 %address.local320, 12, !dbg !12
  %address.local322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local319, i64 %address.local321, !dbg !21
  store half %1050, ptr addrspace(1) %address.local322, align 2, !dbg !21
  %address.local323 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local318, !dbg !21
  %address.local324 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local323, i64 %address.local321, !dbg !21
  %address.local325 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local324, i64 2, !dbg !21
  store half %1049, ptr addrspace(1) %address.local325, align 2, !dbg !21
  %address.local326 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local318, !dbg !21
  %address.local327 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local326, i64 %address.local321, !dbg !21
  %address.local328 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local327, i64 4, !dbg !21
  store half %1048, ptr addrspace(1) %address.local328, align 2, !dbg !21
  %address.local329 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local318, !dbg !21
  %address.local330 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local329, i64 %address.local321, !dbg !21
  %address.local331 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local330, i64 6, !dbg !21
  store half %1047, ptr addrspace(1) %address.local331, align 2, !dbg !21
  %address.local332 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local318, !dbg !21
  %address.local333 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local332, i64 %address.local321, !dbg !21
  %address.local334 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local333, i64 32, !dbg !21
  store half %1035, ptr addrspace(1) %address.local334, align 2, !dbg !21
  %address.local335 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local318, !dbg !21
  %address.local336 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local335, i64 %address.local321, !dbg !21
  %address.local337 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local336, i64 34, !dbg !21
  store half %1034, ptr addrspace(1) %address.local337, align 2, !dbg !21
  %address.local338 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local318, !dbg !21
  %address.local339 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local338, i64 %address.local321, !dbg !21
  %address.local340 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local339, i64 36, !dbg !21
  store half %1033, ptr addrspace(1) %address.local340, align 2, !dbg !21
  %address.local341 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local318, !dbg !21
  %address.local342 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local341, i64 %address.local321, !dbg !21
  %address.local343 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local342, i64 38, !dbg !21
  store half %1032, ptr addrspace(1) %address.local343, align 2, !dbg !21
  br label %1054, !dbg !21

1054:                                             ; preds = %1020, %794
  br i1 true, label %1055, label %1075, !dbg !21

1055:                                             ; preds = %1054
  %1056 = insertelement <4 x float> poison, float %990, i64 0, !dbg !19
  %1057 = insertelement <4 x float> %1056, float %996, i64 1, !dbg !19
  %1058 = insertelement <4 x float> %1057, float %991, i64 2, !dbg !19
  %1059 = insertelement <4 x float> %1058, float %997, i64 3, !dbg !19
  %row.reciprocal16 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %913, !dbg !19
  %normalized17 = fmul <4 x float> %1059, %row.reciprocal16, !dbg !19
  %1060 = fptrunc <4 x float> %normalized17 to <4 x half>, !dbg !20
  %1061 = extractelement <4 x half> %1060, i64 3, !dbg !20
  %1062 = extractelement <4 x half> %1060, i64 2, !dbg !20
  %1063 = extractelement <4 x half> %1060, i64 1, !dbg !20
  %1064 = extractelement <4 x half> %1060, i64 0, !dbg !20
  %1065 = insertelement <4 x float> poison, float %883, i64 0, !dbg !19
  %1066 = insertelement <4 x float> %1065, float %889, i64 1, !dbg !19
  %1067 = insertelement <4 x float> %1066, float %884, i64 2, !dbg !19
  %1068 = insertelement <4 x float> %1067, float %890, i64 3, !dbg !19
  %normalized18 = fmul <4 x float> %1068, %row.reciprocal16, !dbg !19
  %1069 = fptrunc <4 x float> %normalized18 to <4 x half>, !dbg !20
  %1070 = extractelement <4 x half> %1069, i64 3, !dbg !20
  %1071 = extractelement <4 x half> %1069, i64 2, !dbg !20
  %1072 = extractelement <4 x half> %1069, i64 1, !dbg !20
  %1073 = extractelement <4 x half> %1069, i64 0, !dbg !20
  %1074 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread344 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local345 = zext nneg i32 %address.thread344 to i64, !dbg !12
  %address.local346 = add i64 %28, %address.local345, !dbg !21
  %address.local347 = lshr i64 %address.local345, 6, !dbg !12
  %address.local348 = shl nuw nsw i64 %address.local347, 4, !dbg !21
  %address.local349 = add i64 %address.local346, %address.local348, !dbg !21
  %address.local350 = and i64 %address.local345, 240, !dbg !21
  %address.local351 = sub i64 %address.local349, %address.local350, !dbg !21
  %address.local352 = add i64 %address.local351, %29, !dbg !21
  %address.local353 = shl nuw nsw i64 %address.local352, 7, !dbg !21
  %address.local354 = add nuw nsw i64 %1074, %address.local353, !dbg !21
  %address.local355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local354, !dbg !21
  %address.local356 = lshr i64 %address.local345, 2, !dbg !12
  %address.local357 = and i64 %address.local356, 12, !dbg !12
  %address.local358 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local355, i64 %address.local357, !dbg !21
  %address.local359 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local358, i64 64, !dbg !21
  store half %1073, ptr addrspace(1) %address.local359, align 2, !dbg !21
  %address.local360 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local354, !dbg !21
  %address.local361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local360, i64 %address.local357, !dbg !21
  %address.local362 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local361, i64 66, !dbg !21
  store half %1072, ptr addrspace(1) %address.local362, align 2, !dbg !21
  %address.local363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local354, !dbg !21
  %address.local364 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local363, i64 %address.local357, !dbg !21
  %address.local365 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local364, i64 68, !dbg !21
  store half %1071, ptr addrspace(1) %address.local365, align 2, !dbg !21
  %address.local366 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local354, !dbg !21
  %address.local367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local366, i64 %address.local357, !dbg !21
  %address.local368 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local367, i64 70, !dbg !21
  store half %1070, ptr addrspace(1) %address.local368, align 2, !dbg !21
  %address.local369 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local354, !dbg !21
  %address.local370 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local369, i64 %address.local357, !dbg !21
  %address.local371 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local370, i64 96, !dbg !21
  store half %1064, ptr addrspace(1) %address.local371, align 2, !dbg !21
  %address.local372 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local354, !dbg !21
  %address.local373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local372, i64 %address.local357, !dbg !21
  %address.local374 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local373, i64 98, !dbg !21
  store half %1063, ptr addrspace(1) %address.local374, align 2, !dbg !21
  %address.local375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local354, !dbg !21
  %address.local376 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local375, i64 %address.local357, !dbg !21
  %address.local377 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local376, i64 100, !dbg !21
  store half %1062, ptr addrspace(1) %address.local377, align 2, !dbg !21
  %address.local378 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local354, !dbg !21
  %address.local379 = or i64 %address.local356, 51, !dbg !21
  %address.local380 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local378, i64 %address.local379, !dbg !21
  store half %1061, ptr addrspace(1) %address.local380, align 2, !dbg !21
  br label %1075, !dbg !21

1075:                                             ; preds = %1055, %1054
  br i1 true, label %1076, label %1096, !dbg !21

1076:                                             ; preds = %1075
  %1077 = insertelement <4 x float> poison, float %962, i64 0, !dbg !19
  %1078 = insertelement <4 x float> %1077, float %968, i64 1, !dbg !19
  %1079 = insertelement <4 x float> %1078, float %963, i64 2, !dbg !19
  %1080 = insertelement <4 x float> %1079, float %969, i64 3, !dbg !19
  %row.reciprocal19 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %913, !dbg !19
  %normalized20 = fmul <4 x float> %1080, %row.reciprocal19, !dbg !19
  %1081 = fptrunc <4 x float> %normalized20 to <4 x half>, !dbg !20
  %1082 = extractelement <4 x half> %1081, i64 3, !dbg !20
  %1083 = extractelement <4 x half> %1081, i64 2, !dbg !20
  %1084 = extractelement <4 x half> %1081, i64 1, !dbg !20
  %1085 = extractelement <4 x half> %1081, i64 0, !dbg !20
  %1086 = insertelement <4 x float> poison, float %855, i64 0, !dbg !19
  %1087 = insertelement <4 x float> %1086, float %861, i64 1, !dbg !19
  %1088 = insertelement <4 x float> %1087, float %856, i64 2, !dbg !19
  %1089 = insertelement <4 x float> %1088, float %862, i64 3, !dbg !19
  %normalized21 = fmul <4 x float> %1089, %row.reciprocal19, !dbg !19
  %1090 = fptrunc <4 x float> %normalized21 to <4 x half>, !dbg !20
  %1091 = extractelement <4 x half> %1090, i64 3, !dbg !20
  %1092 = extractelement <4 x half> %1090, i64 2, !dbg !20
  %1093 = extractelement <4 x half> %1090, i64 1, !dbg !20
  %1094 = extractelement <4 x half> %1090, i64 0, !dbg !20
  %1095 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread381 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local382 = zext nneg i32 %address.thread381 to i64, !dbg !12
  %address.local383 = add i64 %28, %address.local382, !dbg !21
  %address.local384 = lshr i64 %address.local382, 6, !dbg !12
  %address.local385 = shl nuw nsw i64 %address.local384, 4, !dbg !21
  %address.local386 = add i64 %address.local383, %address.local385, !dbg !21
  %address.local387 = and i64 %address.local382, 240, !dbg !21
  %address.local388 = sub i64 %address.local386, %address.local387, !dbg !21
  %address.local389 = add i64 %address.local388, %29, !dbg !21
  %address.local390 = shl nuw nsw i64 %address.local389, 7, !dbg !21
  %address.local391 = add nuw nsw i64 %1095, %address.local390, !dbg !21
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !21
  %address.local393 = lshr i64 %address.local382, 2, !dbg !12
  %address.local394 = and i64 %address.local393, 12, !dbg !12
  %address.local395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local392, i64 %address.local394, !dbg !21
  %address.local396 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local395, i64 128, !dbg !21
  store half %1094, ptr addrspace(1) %address.local396, align 2, !dbg !21
  %address.local397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !21
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local397, i64 %address.local394, !dbg !21
  %address.local399 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local398, i64 130, !dbg !21
  store half %1093, ptr addrspace(1) %address.local399, align 2, !dbg !21
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !21
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local394, !dbg !21
  %address.local402 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local401, i64 132, !dbg !21
  store half %1092, ptr addrspace(1) %address.local402, align 2, !dbg !21
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !21
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local403, i64 %address.local394, !dbg !21
  %address.local405 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local404, i64 134, !dbg !21
  store half %1091, ptr addrspace(1) %address.local405, align 2, !dbg !21
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !21
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local406, i64 %address.local394, !dbg !21
  %address.local408 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local407, i64 160, !dbg !21
  store half %1085, ptr addrspace(1) %address.local408, align 2, !dbg !21
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !21
  %address.local410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local409, i64 %address.local394, !dbg !21
  %address.local411 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local410, i64 162, !dbg !21
  store half %1084, ptr addrspace(1) %address.local411, align 2, !dbg !21
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !21
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local412, i64 %address.local394, !dbg !21
  %address.local414 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local413, i64 164, !dbg !21
  store half %1083, ptr addrspace(1) %address.local414, align 2, !dbg !21
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !21
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local415, i64 %address.local394, !dbg !21
  %address.local417 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local416, i64 166, !dbg !21
  store half %1082, ptr addrspace(1) %address.local417, align 2, !dbg !21
  br label %1096, !dbg !21

1096:                                             ; preds = %1076, %1075
  br i1 true, label %1097, label %1117, !dbg !21

1097:                                             ; preds = %1096
  %1098 = insertelement <4 x float> poison, float %934, i64 0, !dbg !19
  %1099 = insertelement <4 x float> %1098, float %940, i64 1, !dbg !19
  %1100 = insertelement <4 x float> %1099, float %935, i64 2, !dbg !19
  %1101 = insertelement <4 x float> %1100, float %941, i64 3, !dbg !19
  %row.reciprocal22 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %913, !dbg !19
  %normalized23 = fmul <4 x float> %1101, %row.reciprocal22, !dbg !19
  %1102 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !20
  %1103 = extractelement <4 x half> %1102, i64 3, !dbg !20
  %1104 = extractelement <4 x half> %1102, i64 2, !dbg !20
  %1105 = extractelement <4 x half> %1102, i64 1, !dbg !20
  %1106 = extractelement <4 x half> %1102, i64 0, !dbg !20
  %1107 = insertelement <4 x float> poison, float %827, i64 0, !dbg !19
  %1108 = insertelement <4 x float> %1107, float %833, i64 1, !dbg !19
  %1109 = insertelement <4 x float> %1108, float %828, i64 2, !dbg !19
  %1110 = insertelement <4 x float> %1109, float %834, i64 3, !dbg !19
  %normalized24 = fmul <4 x float> %1110, %row.reciprocal22, !dbg !19
  %1111 = fptrunc <4 x float> %normalized24 to <4 x half>, !dbg !20
  %1112 = extractelement <4 x half> %1111, i64 3, !dbg !20
  %1113 = extractelement <4 x half> %1111, i64 2, !dbg !20
  %1114 = extractelement <4 x half> %1111, i64 1, !dbg !20
  %1115 = extractelement <4 x half> %1111, i64 0, !dbg !20
  %1116 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread418 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local419 = zext nneg i32 %address.thread418 to i64, !dbg !12
  %address.local420 = add i64 %28, %address.local419, !dbg !21
  %address.local421 = lshr i64 %address.local419, 6, !dbg !12
  %address.local422 = shl nuw nsw i64 %address.local421, 4, !dbg !21
  %address.local423 = add i64 %address.local420, %address.local422, !dbg !21
  %address.local424 = and i64 %address.local419, 240, !dbg !21
  %address.local425 = sub i64 %address.local423, %address.local424, !dbg !21
  %address.local426 = add i64 %address.local425, %29, !dbg !21
  %address.local427 = shl nuw nsw i64 %address.local426, 7, !dbg !21
  %address.local428 = add nuw nsw i64 %1116, %address.local427, !dbg !21
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local430 = lshr i64 %address.local419, 2, !dbg !12
  %address.local431 = and i64 %address.local430, 12, !dbg !12
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local429, i64 %address.local431, !dbg !21
  %address.local433 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local432, i64 192, !dbg !21
  store half %1115, ptr addrspace(1) %address.local433, align 2, !dbg !21
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local434, i64 %address.local431, !dbg !21
  %address.local436 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local435, i64 194, !dbg !21
  store half %1114, ptr addrspace(1) %address.local436, align 2, !dbg !21
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local437, i64 %address.local431, !dbg !21
  %address.local439 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local438, i64 196, !dbg !21
  store half %1113, ptr addrspace(1) %address.local439, align 2, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local440, i64 %address.local431, !dbg !21
  %address.local442 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local441, i64 198, !dbg !21
  store half %1112, ptr addrspace(1) %address.local442, align 2, !dbg !21
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local443, i64 %address.local431, !dbg !21
  %address.local445 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local444, i64 224, !dbg !21
  store half %1106, ptr addrspace(1) %address.local445, align 2, !dbg !21
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local447 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local446, i64 %address.local431, !dbg !21
  %address.local448 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local447, i64 226, !dbg !21
  store half %1105, ptr addrspace(1) %address.local448, align 2, !dbg !21
  %address.local449 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local450 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local449, i64 %address.local431, !dbg !21
  %address.local451 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local450, i64 228, !dbg !21
  store half %1104, ptr addrspace(1) %address.local451, align 2, !dbg !21
  %address.local452 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local453 = or i64 %address.local430, 115, !dbg !21
  %address.local454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local452, i64 %address.local453, !dbg !21
  store half %1103, ptr addrspace(1) %address.local454, align 2, !dbg !21
  br label %1117, !dbg !21

1117:                                             ; preds = %1097, %1096
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/group8-reverse-k128-swizzle")
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
