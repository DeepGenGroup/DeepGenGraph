; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/group8-reverse/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/group8-reverse/input.mlir"

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

57:                                               ; preds = %747, %4
  %58 = phi i64 [ %777, %747 ], [ 0, %4 ], !dbg !7
  %59 = phi <4 x float> [ %658, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi <4 x float> [ %663, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = phi <8 x float> [ %772, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %62 = phi <1 x float> [ %773, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi <1 x float> [ %560, %747 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %64 = phi <4 x float> [ %705, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = phi <4 x float> [ %710, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %66 = phi <8 x float> [ %774, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = phi <4 x float> [ %752, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = phi <4 x float> [ %757, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %69 = phi <8 x float> [ %775, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %70 = phi <4 x float> [ %766, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %71 = phi <4 x float> [ %771, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %72 = phi <8 x float> [ %776, %747 ], [ zeroinitializer, %4 ], !dbg !7
  %73 = icmp slt i64 %58, %45, !dbg !7
  br i1 %73, label %74, label %778, !dbg !7

74:                                               ; preds = %57
  %75 = shl nsw i64 %58, 5, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local28 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local29 = and i64 %address.local28, 1536, !dbg !11
  %address.local30 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local29, !dbg !11
  %address.local31 = ptrtoint ptr addrspace(3) %address.local30 to i64, !dbg !11
  %address.local32 = trunc i64 %address.local31 to i32, !dbg !11
  %76 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local32), !dbg !11
  %77 = zext i32 %76 to i64, !dbg !11
  %78 = inttoptr i64 %77 to ptr addrspace(3), !dbg !11
  %address.local33 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local34 = and i64 %address.local33, 8064, !dbg !11
  %address.local35 = lshr i64 %address.local, 3, !dbg !11
  %address.local36 = and i64 %address.local35, 24, !dbg !11
  %address.local37 = or disjoint i64 %address.local34, %address.local36, !dbg !11
  %address.local38 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local37, !dbg !11
  %79 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %80 = ptrtoint ptr addrspace(1) %address.local38 to i64, !dbg !11
  %81 = sub i64 %80, %79, !dbg !11
  %dma.buffer.offset449 = trunc i64 %81 to i32, !dbg !11
  %82 = trunc i64 %79 to i32, !dbg !11
  %83 = insertelement <4 x i32> poison, i32 %82, i64 0, !dbg !11
  %84 = lshr i64 %79, 32, !dbg !11
  %85 = trunc i64 %84 to i32, !dbg !11
  %86 = insertelement <4 x i32> %83, i32 %85, i64 1, !dbg !11
  %87 = insertelement <4 x i32> %86, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc450 = insertelement <4 x i32> %87, i32 131072, i64 3, !dbg !11
  %88 = ptrtoint ptr addrspace(3) %78 to i32, !dbg !11
  %dma.buffer.lds451 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %88), !dbg !11
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds451, i32 %dma.buffer.offset449, <4 x i32> %dma.buffer.rsrc450) #10, !dbg !11
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
  %103 = phi i64 [ %131, %105 ], [ %35, %74 ], !dbg !17
  %104 = icmp samesign ult i64 %103, 128, !dbg !17
  br i1 %104, label %105, label %132, !dbg !17

105:                                              ; preds = %102
  %106 = lshr i64 %103, 2, !dbg !17
  %107 = and i64 %106, 24, !dbg !17
  %108 = shl nuw nsw i64 %103, 3, !dbg !17
  %109 = and i64 %108, 512, !dbg !17
  %110 = shl nuw nsw i64 %103, 7, !dbg !17
  %111 = and i64 %110, 3968, !dbg !17
  %112 = getelementptr [2 x i8], ptr addrspace(1) %101, i64 %107, !dbg !17
  %113 = getelementptr [2 x i8], ptr addrspace(1) %112, i64 %111, !dbg !17
  %114 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %109, !dbg !17
  %115 = ptrtoint ptr addrspace(3) %114 to i64, !dbg !17
  %116 = trunc i64 %115 to i32, !dbg !17
  %117 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %116), !dbg !17
  %118 = zext i32 %117 to i64, !dbg !17
  %119 = inttoptr i64 %118 to ptr addrspace(3), !dbg !17
  %120 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %121 = ptrtoint ptr addrspace(1) %113 to i64, !dbg !17
  %122 = sub i64 %121, %120, !dbg !17
  %dma.buffer.offset = trunc i64 %122 to i32, !dbg !17
  %123 = trunc i64 %120 to i32, !dbg !17
  %124 = insertelement <4 x i32> poison, i32 %123, i64 0, !dbg !17
  %125 = lshr i64 %120, 32, !dbg !17
  %126 = trunc i64 %125 to i32, !dbg !17
  %127 = insertelement <4 x i32> %124, i32 %126, i64 1, !dbg !17
  %128 = insertelement <4 x i32> %127, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %128, i32 131072, i64 3, !dbg !17
  %129 = ptrtoint ptr addrspace(3) %119 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %129), !dbg !17
  %130 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %131 = or disjoint i64 %35, 256, !dbg !17
  br label %102, !dbg !17

132:                                              ; preds = %102
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.thread39 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local40 = lshr i32 %address.thread39, 2, !dbg !12
  %address.local41 = and i32 %address.local40, 4, !dbg !12
  %address.local42 = shl nuw nsw i32 %address.thread39, 1, !dbg !12
  %address.local43 = and i32 %address.local42, 384, !dbg !12
  %address.local44 = or disjoint i32 %address.local41, %address.local43, !dbg !12
  %address.local45 = shl nuw nsw i32 %address.thread39, 3, !dbg !12
  %address.local46 = and i32 %address.local45, 120, !dbg !12
  %address.local47 = or disjoint i32 %address.local44, %address.local46, !dbg !12
  %address.local48 = shl nuw nsw i32 %address.thread39, 4, !dbg !12
  %address.local49 = and i32 %address.local48, 512, !dbg !12
  %address.local50 = or disjoint i32 %address.local47, %address.local49, !dbg !12
  %address.local51 = zext nneg i32 %address.local50 to i64, !dbg !12
  %address.local52 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local51, !dbg !12
  %133 = load <4 x half>, ptr addrspace(3) %address.local52, align 2, !dbg !12
  %address.local53 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local51, !dbg !12
  %address.local54 = getelementptr i8, ptr addrspace(3) %address.local53, i64 2048, !dbg !12
  %134 = load <4 x half>, ptr addrspace(3) %address.local54, align 2, !dbg !12
  %135 = fmul <4 x half> %133, splat (half 0xH3015), !dbg !12
  %136 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local55 = lshr i32 %address.thread39, 2, !dbg !16
  %address.local56 = and i32 %address.local55, 4, !dbg !16
  %address.local57 = shl nuw nsw i32 %address.thread39, 3, !dbg !16
  %address.local58 = and i32 %address.local57, 376, !dbg !16
  %address.local59 = or disjoint i32 %address.local56, %address.local58, !dbg !16
  %address.local60 = zext nneg i32 %address.local59 to i64, !dbg !16
  %address.local61 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local60, !dbg !16
  %137 = load <4 x half>, ptr addrspace(3) %address.local61, align 2, !dbg !16
  %address.local62 = lshr i32 %address.thread39, 2, !dbg !16
  %address.local63 = and i32 %address.local62, 4, !dbg !16
  %address.local64 = shl nuw nsw i32 %address.thread39, 3, !dbg !16
  %address.local65 = and i32 %address.local64, 376, !dbg !16
  %address.local66 = or disjoint i32 %address.local63, %address.local65, !dbg !16
  %address.local67 = zext nneg i32 %address.local66 to i64, !dbg !16
  %address.local68 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local67, !dbg !16
  %address.local69 = getelementptr i8, ptr addrspace(3) %address.local68, i64 1024, !dbg !16
  %138 = load <4 x half>, ptr addrspace(3) %address.local69, align 2, !dbg !16
  %139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %137, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local70 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local60, !dbg !16
  %address.local71 = getelementptr i8, ptr addrspace(3) %address.local70, i64 256, !dbg !16
  %140 = load <4 x half>, ptr addrspace(3) %address.local71, align 2, !dbg !16
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %138, <4 x float> %139) #9, !dbg !16
  %address.local72 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local67, !dbg !16
  %address.local73 = getelementptr i8, ptr addrspace(3) %address.local72, i64 1280, !dbg !16
  %142 = load <4 x half>, ptr addrspace(3) %address.local73, align 2, !dbg !16
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %140, <4 x float> zeroinitializer) #9, !dbg !16
  %144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %142, <4 x float> %143) #9, !dbg !16
  %address.local74 = zext nneg i32 %address.thread39 to i64, !dbg !12
  %address.local75 = shl nuw nsw i64 %address.local74, 3, !dbg !11
  %address.local76 = and i64 %address.local75, 1536, !dbg !11
  %address.local77 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local76, !dbg !11
  %address.local78 = ptrtoint ptr addrspace(3) %address.local77 to i64, !dbg !11
  %address.local79 = trunc i64 %address.local78 to i32, !dbg !11
  %145 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local79), !dbg !11
  %146 = zext i32 %145 to i64, !dbg !11
  %147 = inttoptr i64 %146 to ptr addrspace(3), !dbg !11
  %address.local80 = shl nuw nsw i64 %address.local74, 7, !dbg !11
  %address.local81 = and i64 %address.local80, 8064, !dbg !11
  %address.local82 = lshr i64 %address.local74, 3, !dbg !11
  %address.local83 = and i64 %address.local82, 24, !dbg !11
  %address.local84 = or disjoint i64 %address.local81, %address.local83, !dbg !11
  %address.local85 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local84, !dbg !11
  %148 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %149 = ptrtoint ptr addrspace(1) %address.local85 to i64, !dbg !11
  %150 = sub i64 %149, %148, !dbg !11
  %dma.buffer.offset446 = trunc i64 %150 to i32, !dbg !11
  %151 = trunc i64 %148 to i32, !dbg !11
  %152 = insertelement <4 x i32> poison, i32 %151, i64 0, !dbg !11
  %153 = lshr i64 %148, 32, !dbg !11
  %154 = trunc i64 %153 to i32, !dbg !11
  %155 = insertelement <4 x i32> %152, i32 %154, i64 1, !dbg !11
  %156 = insertelement <4 x i32> %155, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc447 = insertelement <4 x i32> %156, i32 131072, i64 3, !dbg !11
  %157 = ptrtoint ptr addrspace(3) %147 to i32, !dbg !11
  %dma.buffer.lds448 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %157), !dbg !11
  %158 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds448, i32 %dma.buffer.offset446, <4 x i32> %dma.buffer.rsrc447) #10, !dbg !11
  %159 = add nuw i64 %58, 4096, !dbg !17
  %160 = icmp slt i64 %159, 0, !dbg !17
  %161 = sub nuw i64 -4097, %58, !dbg !17
  %162 = select i1 %160, i64 %161, i64 %159, !dbg !17
  %163 = sdiv i64 %162, 16384, !dbg !17
  %.lobit = ashr i64 %159, 63, !dbg !17
  %164 = xor i64 %163, %.lobit, !dbg !17
  %165 = add nsw i64 %164, %15, !dbg !17
  %166 = add nsw i64 %92, %52, !dbg !17
  %167 = add nsw i64 %166, 32, !dbg !17
  %168 = icmp slt i64 %166, -32, !dbg !17
  %169 = sub nsw i64 -33, %166, !dbg !17
  %170 = select i1 %168, i64 %169, i64 %167, !dbg !17
  %171 = sdiv i64 %170, 128, !dbg !17
  %172 = sext i1 %168 to i64, !dbg !17
  %173 = xor i64 %171, %172, !dbg !17
  %174 = shl nsw i64 %173, 7, !dbg !17
  %175 = sub i64 %52, %174, !dbg !17
  %176 = shl i64 %165, 19, !dbg !17
  %177 = and i64 %176, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %178 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %177, !dbg !17
  %179 = getelementptr [2 x i8], ptr addrspace(1) %178, i64 %92, !dbg !17
  %180 = getelementptr [2 x i8], ptr addrspace(1) %179, i64 %175, !dbg !17
  %181 = getelementptr i8, ptr addrspace(1) %180, i64 64, !dbg !17
  %182 = getelementptr [2 x i8], ptr addrspace(1) %181, i64 %98, !dbg !17
  br label %183, !dbg !17

183:                                              ; preds = %186, %132
  %184 = phi i64 [ %212, %186 ], [ %35, %132 ], !dbg !17
  %185 = icmp samesign ult i64 %184, 128, !dbg !17
  br i1 %185, label %186, label %213, !dbg !17

186:                                              ; preds = %183
  %187 = lshr i64 %184, 2, !dbg !17
  %188 = and i64 %187, 24, !dbg !17
  %189 = shl nuw nsw i64 %184, 3, !dbg !17
  %190 = and i64 %189, 512, !dbg !17
  %191 = shl nuw nsw i64 %184, 7, !dbg !17
  %192 = and i64 %191, 3968, !dbg !17
  %193 = getelementptr [2 x i8], ptr addrspace(1) %182, i64 %188, !dbg !17
  %194 = getelementptr [2 x i8], ptr addrspace(1) %193, i64 %192, !dbg !17
  %195 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %190, !dbg !17
  %196 = ptrtoint ptr addrspace(3) %195 to i64, !dbg !17
  %197 = trunc i64 %196 to i32, !dbg !17
  %198 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %197), !dbg !17
  %199 = zext i32 %198 to i64, !dbg !17
  %200 = inttoptr i64 %199 to ptr addrspace(3), !dbg !17
  %201 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %202 = ptrtoint ptr addrspace(1) %194 to i64, !dbg !17
  %203 = sub i64 %202, %201, !dbg !17
  %dma.buffer.offset419 = trunc i64 %203 to i32, !dbg !17
  %204 = trunc i64 %201 to i32, !dbg !17
  %205 = insertelement <4 x i32> poison, i32 %204, i64 0, !dbg !17
  %206 = lshr i64 %201, 32, !dbg !17
  %207 = trunc i64 %206 to i32, !dbg !17
  %208 = insertelement <4 x i32> %205, i32 %207, i64 1, !dbg !17
  %209 = insertelement <4 x i32> %208, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc420 = insertelement <4 x i32> %209, i32 131072, i64 3, !dbg !17
  %210 = ptrtoint ptr addrspace(3) %200 to i32, !dbg !17
  %dma.buffer.lds421 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %210), !dbg !17
  %211 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds421, i32 %dma.buffer.offset419, <4 x i32> %dma.buffer.rsrc420) #10, !dbg !17
  %212 = or disjoint i64 %35, 256, !dbg !17
  br label %183, !dbg !17

213:                                              ; preds = %183
  %address.thread86 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local87 = lshr i32 %address.thread86, 2, !dbg !12
  %address.local88 = and i32 %address.local87, 4, !dbg !12
  %address.local89 = shl nuw nsw i32 %address.thread86, 1, !dbg !12
  %address.local90 = and i32 %address.local89, 384, !dbg !12
  %address.local91 = or disjoint i32 %address.local88, %address.local90, !dbg !12
  %address.local92 = shl nuw nsw i32 %address.thread86, 3, !dbg !12
  %address.local93 = and i32 %address.local92, 120, !dbg !12
  %address.local94 = or disjoint i32 %address.local91, %address.local93, !dbg !12
  %address.local95 = shl nuw nsw i32 %address.thread86, 4, !dbg !12
  %address.local96 = and i32 %address.local95, 512, !dbg !12
  %address.local97 = or disjoint i32 %address.local94, %address.local96, !dbg !12
  %address.local98 = zext nneg i32 %address.local97 to i64, !dbg !12
  %address.local99 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local98, !dbg !12
  %214 = load <4 x half>, ptr addrspace(3) %address.local99, align 2, !dbg !12
  %address.local100 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local98, !dbg !12
  %address.local101 = getelementptr i8, ptr addrspace(3) %address.local100, i64 2048, !dbg !12
  %215 = load <4 x half>, ptr addrspace(3) %address.local101, align 2, !dbg !12
  %216 = fmul <4 x half> %214, splat (half 0xH3015), !dbg !12
  %217 = fmul <4 x half> %215, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local102 = lshr i32 %address.thread86, 2, !dbg !16
  %address.local103 = and i32 %address.local102, 4, !dbg !16
  %address.local104 = shl nuw nsw i32 %address.thread86, 3, !dbg !16
  %address.local105 = and i32 %address.local104, 376, !dbg !16
  %address.local106 = or disjoint i32 %address.local103, %address.local105, !dbg !16
  %address.local107 = zext nneg i32 %address.local106 to i64, !dbg !16
  %address.local108 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local107, !dbg !16
  %218 = load <4 x half>, ptr addrspace(3) %address.local108, align 2, !dbg !16
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %216, <4 x half> %218, <4 x float> %141) #9, !dbg !16
  %address.local109 = lshr i32 %address.thread86, 2, !dbg !16
  %address.local110 = and i32 %address.local109, 4, !dbg !16
  %address.local111 = shl nuw nsw i32 %address.thread86, 3, !dbg !16
  %address.local112 = and i32 %address.local111, 376, !dbg !16
  %address.local113 = or disjoint i32 %address.local110, %address.local112, !dbg !16
  %address.local114 = zext nneg i32 %address.local113 to i64, !dbg !16
  %address.local115 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local114, !dbg !16
  %address.local116 = getelementptr i8, ptr addrspace(3) %address.local115, i64 1024, !dbg !16
  %220 = load <4 x half>, ptr addrspace(3) %address.local116, align 2, !dbg !16
  %address.local117 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local107, !dbg !16
  %address.local118 = getelementptr i8, ptr addrspace(3) %address.local117, i64 256, !dbg !16
  %221 = load <4 x half>, ptr addrspace(3) %address.local118, align 2, !dbg !16
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %217, <4 x half> %220, <4 x float> %219) #9, !dbg !16
  %address.local119 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local114, !dbg !16
  %address.local120 = getelementptr i8, ptr addrspace(3) %address.local119, i64 1280, !dbg !16
  %223 = load <4 x half>, ptr addrspace(3) %address.local120, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %216, <4 x half> %221, <4 x float> %144) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %217, <4 x half> %223, <4 x float> %224) #9, !dbg !16
  %address.local121 = zext nneg i32 %address.thread86 to i64, !dbg !12
  %address.local122 = shl nuw nsw i64 %address.local121, 3, !dbg !11
  %address.local123 = and i64 %address.local122, 1536, !dbg !11
  %address.local124 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local123, !dbg !11
  %address.local125 = ptrtoint ptr addrspace(3) %address.local124 to i64, !dbg !11
  %address.local126 = trunc i64 %address.local125 to i32, !dbg !11
  %226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local126), !dbg !11
  %227 = zext i32 %226 to i64, !dbg !11
  %228 = inttoptr i64 %227 to ptr addrspace(3), !dbg !11
  %address.local127 = shl nuw nsw i64 %address.local121, 7, !dbg !11
  %address.local128 = and i64 %address.local127, 8064, !dbg !11
  %address.local129 = lshr i64 %address.local121, 3, !dbg !11
  %address.local130 = and i64 %address.local129, 24, !dbg !11
  %address.local131 = or disjoint i64 %address.local128, %address.local130, !dbg !11
  %address.local132 = getelementptr [2 x i8], ptr addrspace(1) %54, i64 %address.local131, !dbg !11
  %229 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %230 = ptrtoint ptr addrspace(1) %address.local132 to i64, !dbg !11
  %231 = sub i64 %230, %229, !dbg !11
  %dma.buffer.offset443 = trunc i64 %231 to i32, !dbg !11
  %232 = trunc i64 %229 to i32, !dbg !11
  %233 = insertelement <4 x i32> poison, i32 %232, i64 0, !dbg !11
  %234 = lshr i64 %229, 32, !dbg !11
  %235 = trunc i64 %234 to i32, !dbg !11
  %236 = insertelement <4 x i32> %233, i32 %235, i64 1, !dbg !11
  %237 = insertelement <4 x i32> %236, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc444 = insertelement <4 x i32> %237, i32 131072, i64 3, !dbg !11
  %238 = ptrtoint ptr addrspace(3) %228 to i32, !dbg !11
  %dma.buffer.lds445 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %238), !dbg !11
  %239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds445, i32 %dma.buffer.offset443, <4 x i32> %dma.buffer.rsrc444) #10, !dbg !11
  %240 = add nuw i64 %58, 8192, !dbg !17
  %241 = icmp slt i64 %240, 0, !dbg !17
  %242 = sub nuw i64 -8193, %58, !dbg !17
  %243 = select i1 %241, i64 %242, i64 %240, !dbg !17
  %244 = sdiv i64 %243, 16384, !dbg !17
  %.lobit2 = ashr i64 %240, 63, !dbg !17
  %245 = xor i64 %244, %.lobit2, !dbg !17
  %246 = add nsw i64 %245, %15, !dbg !17
  %247 = add nsw i64 %166, 64, !dbg !17
  %248 = icmp slt i64 %166, -64, !dbg !17
  %249 = sub nsw i64 -65, %166, !dbg !17
  %250 = select i1 %248, i64 %249, i64 %247, !dbg !17
  %251 = sdiv i64 %250, 128, !dbg !17
  %252 = sext i1 %248 to i64, !dbg !17
  %253 = xor i64 %251, %252, !dbg !17
  %254 = shl nsw i64 %253, 7, !dbg !17
  %255 = sub i64 %52, %254, !dbg !17
  %256 = shl i64 %246, 19, !dbg !17
  %257 = and i64 %256, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %258 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %257, !dbg !17
  %259 = getelementptr [2 x i8], ptr addrspace(1) %258, i64 %92, !dbg !17
  %260 = getelementptr [2 x i8], ptr addrspace(1) %259, i64 %255, !dbg !17
  %261 = getelementptr i8, ptr addrspace(1) %260, i64 128, !dbg !17
  %262 = getelementptr [2 x i8], ptr addrspace(1) %261, i64 %98, !dbg !17
  br label %263, !dbg !17

263:                                              ; preds = %266, %213
  %264 = phi i64 [ %292, %266 ], [ %35, %213 ], !dbg !17
  %265 = icmp samesign ult i64 %264, 128, !dbg !17
  br i1 %265, label %266, label %293, !dbg !17

266:                                              ; preds = %263
  %267 = lshr i64 %264, 2, !dbg !17
  %268 = and i64 %267, 24, !dbg !17
  %269 = shl nuw nsw i64 %264, 3, !dbg !17
  %270 = and i64 %269, 512, !dbg !17
  %271 = shl nuw nsw i64 %264, 7, !dbg !17
  %272 = and i64 %271, 3968, !dbg !17
  %273 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %268, !dbg !17
  %274 = getelementptr [2 x i8], ptr addrspace(1) %273, i64 %272, !dbg !17
  %275 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %270, !dbg !17
  %276 = ptrtoint ptr addrspace(3) %275 to i64, !dbg !17
  %277 = trunc i64 %276 to i32, !dbg !17
  %278 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %277), !dbg !17
  %279 = zext i32 %278 to i64, !dbg !17
  %280 = inttoptr i64 %279 to ptr addrspace(3), !dbg !17
  %281 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %282 = ptrtoint ptr addrspace(1) %274 to i64, !dbg !17
  %283 = sub i64 %282, %281, !dbg !17
  %dma.buffer.offset422 = trunc i64 %283 to i32, !dbg !17
  %284 = trunc i64 %281 to i32, !dbg !17
  %285 = insertelement <4 x i32> poison, i32 %284, i64 0, !dbg !17
  %286 = lshr i64 %281, 32, !dbg !17
  %287 = trunc i64 %286 to i32, !dbg !17
  %288 = insertelement <4 x i32> %285, i32 %287, i64 1, !dbg !17
  %289 = insertelement <4 x i32> %288, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc423 = insertelement <4 x i32> %289, i32 131072, i64 3, !dbg !17
  %290 = ptrtoint ptr addrspace(3) %280 to i32, !dbg !17
  %dma.buffer.lds424 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %290), !dbg !17
  %291 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds424, i32 %dma.buffer.offset422, <4 x i32> %dma.buffer.rsrc423) #10, !dbg !17
  %292 = or disjoint i64 %35, 256, !dbg !17
  br label %263, !dbg !17

293:                                              ; preds = %263
  %address.thread133 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local134 = lshr i32 %address.thread133, 2, !dbg !12
  %address.local135 = and i32 %address.local134, 4, !dbg !12
  %address.local136 = shl nuw nsw i32 %address.thread133, 1, !dbg !12
  %address.local137 = and i32 %address.local136, 384, !dbg !12
  %address.local138 = or disjoint i32 %address.local135, %address.local137, !dbg !12
  %address.local139 = shl nuw nsw i32 %address.thread133, 3, !dbg !12
  %address.local140 = and i32 %address.local139, 120, !dbg !12
  %address.local141 = or disjoint i32 %address.local138, %address.local140, !dbg !12
  %address.local142 = shl nuw nsw i32 %address.thread133, 4, !dbg !12
  %address.local143 = and i32 %address.local142, 512, !dbg !12
  %address.local144 = or disjoint i32 %address.local141, %address.local143, !dbg !12
  %address.local145 = zext nneg i32 %address.local144 to i64, !dbg !12
  %address.local146 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local145, !dbg !12
  %294 = load <4 x half>, ptr addrspace(3) %address.local146, align 2, !dbg !12
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local145, !dbg !12
  %address.local148 = getelementptr i8, ptr addrspace(3) %address.local147, i64 2048, !dbg !12
  %295 = load <4 x half>, ptr addrspace(3) %address.local148, align 2, !dbg !12
  %296 = fmul <4 x half> %294, splat (half 0xH3015), !dbg !12
  %297 = fmul <4 x half> %295, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local149 = lshr i32 %address.thread133, 2, !dbg !16
  %address.local150 = and i32 %address.local149, 4, !dbg !16
  %address.local151 = shl nuw nsw i32 %address.thread133, 3, !dbg !16
  %address.local152 = and i32 %address.local151, 376, !dbg !16
  %address.local153 = or disjoint i32 %address.local150, %address.local152, !dbg !16
  %address.local154 = zext nneg i32 %address.local153 to i64, !dbg !16
  %address.local155 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local154, !dbg !16
  %298 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !16
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %298, <4 x float> %222) #9, !dbg !16
  %address.local156 = lshr i32 %address.thread133, 2, !dbg !16
  %address.local157 = and i32 %address.local156, 4, !dbg !16
  %address.local158 = shl nuw nsw i32 %address.thread133, 3, !dbg !16
  %address.local159 = and i32 %address.local158, 376, !dbg !16
  %address.local160 = or disjoint i32 %address.local157, %address.local159, !dbg !16
  %address.local161 = zext nneg i32 %address.local160 to i64, !dbg !16
  %address.local162 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local161, !dbg !16
  %address.local163 = getelementptr i8, ptr addrspace(3) %address.local162, i64 1024, !dbg !16
  %300 = load <4 x half>, ptr addrspace(3) %address.local163, align 2, !dbg !16
  %address.local164 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local154, !dbg !16
  %address.local165 = getelementptr i8, ptr addrspace(3) %address.local164, i64 256, !dbg !16
  %301 = load <4 x half>, ptr addrspace(3) %address.local165, align 2, !dbg !16
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %300, <4 x float> %299) #9, !dbg !16
  %address.local166 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local161, !dbg !16
  %address.local167 = getelementptr i8, ptr addrspace(3) %address.local166, i64 1280, !dbg !16
  %303 = load <4 x half>, ptr addrspace(3) %address.local167, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %301, <4 x float> %225) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %303, <4 x float> %304) #9, !dbg !16
  %address.local168 = zext nneg i32 %address.thread133 to i64, !dbg !12
  %address.local169 = shl nuw nsw i64 %address.local168, 3, !dbg !11
  %address.local170 = and i64 %address.local169, 1536, !dbg !11
  %address.local171 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local170, !dbg !11
  %address.local172 = ptrtoint ptr addrspace(3) %address.local171 to i64, !dbg !11
  %address.local173 = trunc i64 %address.local172 to i32, !dbg !11
  %306 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local173), !dbg !11
  %307 = zext i32 %306 to i64, !dbg !11
  %308 = inttoptr i64 %307 to ptr addrspace(3), !dbg !11
  %address.local174 = shl nuw nsw i64 %address.local168, 7, !dbg !11
  %address.local175 = and i64 %address.local174, 8064, !dbg !11
  %address.local176 = lshr i64 %address.local168, 3, !dbg !11
  %address.local177 = and i64 %address.local176, 24, !dbg !11
  %address.local178 = or disjoint i64 %address.local175, %address.local177, !dbg !11
  %address.local179 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local178, !dbg !11
  %309 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %310 = ptrtoint ptr addrspace(1) %address.local179 to i64, !dbg !11
  %311 = sub i64 %310, %309, !dbg !11
  %dma.buffer.offset440 = trunc i64 %311 to i32, !dbg !11
  %312 = trunc i64 %309 to i32, !dbg !11
  %313 = insertelement <4 x i32> poison, i32 %312, i64 0, !dbg !11
  %314 = lshr i64 %309, 32, !dbg !11
  %315 = trunc i64 %314 to i32, !dbg !11
  %316 = insertelement <4 x i32> %313, i32 %315, i64 1, !dbg !11
  %317 = insertelement <4 x i32> %316, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc441 = insertelement <4 x i32> %317, i32 131072, i64 3, !dbg !11
  %318 = ptrtoint ptr addrspace(3) %308 to i32, !dbg !11
  %dma.buffer.lds442 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %318), !dbg !11
  %319 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds442, i32 %dma.buffer.offset440, <4 x i32> %dma.buffer.rsrc441) #10, !dbg !11
  %320 = add nuw i64 %58, 12288, !dbg !17
  %321 = icmp slt i64 %320, 0, !dbg !17
  %322 = sub nuw i64 -12289, %58, !dbg !17
  %323 = select i1 %321, i64 %322, i64 %320, !dbg !17
  %324 = sdiv i64 %323, 16384, !dbg !17
  %.lobit3 = ashr i64 %320, 63, !dbg !17
  %325 = xor i64 %324, %.lobit3, !dbg !17
  %326 = add nsw i64 %325, %15, !dbg !17
  %327 = add nsw i64 %166, 96, !dbg !17
  %328 = icmp slt i64 %166, -96, !dbg !17
  %329 = sub nsw i64 -97, %166, !dbg !17
  %330 = select i1 %328, i64 %329, i64 %327, !dbg !17
  %331 = sdiv i64 %330, 128, !dbg !17
  %332 = sext i1 %328 to i64, !dbg !17
  %333 = xor i64 %331, %332, !dbg !17
  %334 = shl nsw i64 %333, 7, !dbg !17
  %335 = sub i64 %52, %334, !dbg !17
  %336 = shl i64 %326, 19, !dbg !17
  %337 = and i64 %336, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %338 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %337, !dbg !17
  %339 = getelementptr [2 x i8], ptr addrspace(1) %338, i64 %92, !dbg !17
  %340 = getelementptr [2 x i8], ptr addrspace(1) %339, i64 %335, !dbg !17
  %341 = getelementptr i8, ptr addrspace(1) %340, i64 192, !dbg !17
  %342 = getelementptr [2 x i8], ptr addrspace(1) %341, i64 %98, !dbg !17
  br label %343, !dbg !17

343:                                              ; preds = %346, %293
  %344 = phi i64 [ %372, %346 ], [ %35, %293 ], !dbg !17
  %345 = icmp samesign ult i64 %344, 128, !dbg !17
  br i1 %345, label %346, label %373, !dbg !17

346:                                              ; preds = %343
  %347 = lshr i64 %344, 2, !dbg !17
  %348 = and i64 %347, 24, !dbg !17
  %349 = shl nuw nsw i64 %344, 3, !dbg !17
  %350 = and i64 %349, 512, !dbg !17
  %351 = shl nuw nsw i64 %344, 7, !dbg !17
  %352 = and i64 %351, 3968, !dbg !17
  %353 = getelementptr [2 x i8], ptr addrspace(1) %342, i64 %348, !dbg !17
  %354 = getelementptr [2 x i8], ptr addrspace(1) %353, i64 %352, !dbg !17
  %355 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %350, !dbg !17
  %356 = ptrtoint ptr addrspace(3) %355 to i64, !dbg !17
  %357 = trunc i64 %356 to i32, !dbg !17
  %358 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %357), !dbg !17
  %359 = zext i32 %358 to i64, !dbg !17
  %360 = inttoptr i64 %359 to ptr addrspace(3), !dbg !17
  %361 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %362 = ptrtoint ptr addrspace(1) %354 to i64, !dbg !17
  %363 = sub i64 %362, %361, !dbg !17
  %dma.buffer.offset425 = trunc i64 %363 to i32, !dbg !17
  %364 = trunc i64 %361 to i32, !dbg !17
  %365 = insertelement <4 x i32> poison, i32 %364, i64 0, !dbg !17
  %366 = lshr i64 %361, 32, !dbg !17
  %367 = trunc i64 %366 to i32, !dbg !17
  %368 = insertelement <4 x i32> %365, i32 %367, i64 1, !dbg !17
  %369 = insertelement <4 x i32> %368, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc426 = insertelement <4 x i32> %369, i32 131072, i64 3, !dbg !17
  %370 = ptrtoint ptr addrspace(3) %360 to i32, !dbg !17
  %dma.buffer.lds427 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %370), !dbg !17
  %371 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds427, i32 %dma.buffer.offset425, <4 x i32> %dma.buffer.rsrc426) #10, !dbg !17
  %372 = or disjoint i64 %35, 256, !dbg !17
  br label %343, !dbg !17

373:                                              ; preds = %343
  %address.thread180 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local181 = lshr i32 %address.thread180, 2, !dbg !12
  %address.local182 = and i32 %address.local181, 4, !dbg !12
  %address.local183 = shl nuw nsw i32 %address.thread180, 1, !dbg !12
  %address.local184 = and i32 %address.local183, 384, !dbg !12
  %address.local185 = or disjoint i32 %address.local182, %address.local184, !dbg !12
  %address.local186 = shl nuw nsw i32 %address.thread180, 3, !dbg !12
  %address.local187 = and i32 %address.local186, 120, !dbg !12
  %address.local188 = or disjoint i32 %address.local185, %address.local187, !dbg !12
  %address.local189 = shl nuw nsw i32 %address.thread180, 4, !dbg !12
  %address.local190 = and i32 %address.local189, 512, !dbg !12
  %address.local191 = or disjoint i32 %address.local188, %address.local190, !dbg !12
  %address.local192 = zext nneg i32 %address.local191 to i64, !dbg !12
  %address.local193 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local192, !dbg !12
  %374 = load <4 x half>, ptr addrspace(3) %address.local193, align 2, !dbg !12
  %address.local194 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local192, !dbg !12
  %address.local195 = getelementptr i8, ptr addrspace(3) %address.local194, i64 2048, !dbg !12
  %375 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !12
  %376 = fmul <4 x half> %374, splat (half 0xH3015), !dbg !12
  %377 = fmul <4 x half> %375, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %378 = add nuw nsw i64 %92, %15, !dbg !18
  %379 = shl i64 %378, 19, !dbg !18
  %380 = and i64 %379, 16252928, !dbg !18
  %381 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %380, !dbg !18
  %382 = getelementptr [2 x i8], ptr addrspace(1) %381, i64 %98, !dbg !18
  br label %383, !dbg !18

383:                                              ; preds = %386, %373
  %384 = phi i64 [ %412, %386 ], [ %35, %373 ], !dbg !18
  %385 = icmp samesign ult i64 %384, 128, !dbg !18
  br i1 %385, label %386, label %413, !dbg !18

386:                                              ; preds = %383
  %387 = shl nuw nsw i64 %384, 3, !dbg !18
  %388 = and i64 %387, 24, !dbg !18
  %389 = shl nuw nsw i64 %384, 3, !dbg !18
  %390 = and i64 %389, 512, !dbg !18
  %391 = shl nuw nsw i64 %384, 5, !dbg !18
  %392 = and i64 %391, 3968, !dbg !18
  %393 = getelementptr [2 x i8], ptr addrspace(1) %382, i64 %392, !dbg !18
  %394 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %388, !dbg !18
  %395 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %390, !dbg !18
  %396 = ptrtoint ptr addrspace(3) %395 to i64, !dbg !18
  %397 = trunc i64 %396 to i32, !dbg !18
  %398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %397), !dbg !18
  %399 = zext i32 %398 to i64, !dbg !18
  %400 = inttoptr i64 %399 to ptr addrspace(3), !dbg !18
  %401 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %402 = ptrtoint ptr addrspace(1) %394 to i64, !dbg !18
  %403 = sub i64 %402, %401, !dbg !18
  %dma.buffer.offset428 = trunc i64 %403 to i32, !dbg !18
  %404 = trunc i64 %401 to i32, !dbg !18
  %405 = insertelement <4 x i32> poison, i32 %404, i64 0, !dbg !18
  %406 = lshr i64 %401, 32, !dbg !18
  %407 = trunc i64 %406 to i32, !dbg !18
  %408 = insertelement <4 x i32> %405, i32 %407, i64 1, !dbg !18
  %409 = insertelement <4 x i32> %408, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc429 = insertelement <4 x i32> %409, i32 131072, i64 3, !dbg !18
  %410 = ptrtoint ptr addrspace(3) %400 to i32, !dbg !18
  %dma.buffer.lds430 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %410), !dbg !18
  %411 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds430, i32 %dma.buffer.offset428, <4 x i32> %dma.buffer.rsrc429) #10, !dbg !18
  %412 = or disjoint i64 %35, 256, !dbg !18
  br label %383, !dbg !18

413:                                              ; preds = %383
  %address.thread196 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local197 = lshr i32 %address.thread196, 2, !dbg !16
  %address.local198 = and i32 %address.local197, 4, !dbg !16
  %address.local199 = shl nuw nsw i32 %address.thread196, 3, !dbg !16
  %address.local200 = and i32 %address.local199, 376, !dbg !16
  %address.local201 = or disjoint i32 %address.local198, %address.local200, !dbg !16
  %address.local202 = zext nneg i32 %address.local201 to i64, !dbg !16
  %address.local203 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local202, !dbg !16
  %414 = load <4 x half>, ptr addrspace(3) %address.local203, align 2, !dbg !16
  %415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %376, <4 x half> %414, <4 x float> %302) #9, !dbg !16
  %address.local204 = lshr i32 %address.thread196, 2, !dbg !16
  %address.local205 = and i32 %address.local204, 4, !dbg !16
  %address.local206 = shl nuw nsw i32 %address.thread196, 3, !dbg !16
  %address.local207 = and i32 %address.local206, 376, !dbg !16
  %address.local208 = or disjoint i32 %address.local205, %address.local207, !dbg !16
  %address.local209 = zext nneg i32 %address.local208 to i64, !dbg !16
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local209, !dbg !16
  %address.local211 = getelementptr i8, ptr addrspace(3) %address.local210, i64 1024, !dbg !16
  %416 = load <4 x half>, ptr addrspace(3) %address.local211, align 2, !dbg !16
  %address.local212 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local202, !dbg !16
  %address.local213 = getelementptr i8, ptr addrspace(3) %address.local212, i64 256, !dbg !16
  %417 = load <4 x half>, ptr addrspace(3) %address.local213, align 2, !dbg !16
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %416, <4 x float> %415) #9, !dbg !16
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local209, !dbg !16
  %address.local215 = getelementptr i8, ptr addrspace(3) %address.local214, i64 1280, !dbg !16
  %419 = load <4 x half>, ptr addrspace(3) %address.local215, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %376, <4 x half> %417, <4 x float> %305) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !22
  %421 = extractelement <4 x float> %418, i64 0, !dbg !22
  %422 = extractelement <4 x float> %418, i64 1, !dbg !22
  %423 = select i1 %.not, float %422, float %421, !dbg !13
  %424 = extractelement <4 x float> %418, i64 2, !dbg !22
  %425 = extractelement <4 x float> %418, i64 3, !dbg !22
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %419, <4 x float> %420) #9, !dbg !16
  %427 = extractelement <4 x float> %426, i64 0, !dbg !22
  %428 = extractelement <4 x float> %426, i64 1, !dbg !22
  %429 = extractelement <4 x float> %426, i64 2, !dbg !22
  %430 = extractelement <4 x float> %426, i64 3, !dbg !22
  %431 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %432 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %431), !dbg !13
  %433 = add nuw nsw i32 %432, 64, !dbg !13
  %434 = xor i32 %432, 16, !dbg !13
  %435 = and i32 %433, 192, !dbg !13
  %436 = icmp samesign ult i32 %434, %435, !dbg !13
  %437 = select i1 %436, i32 %434, i32 %432, !dbg !13
  %438 = shl nuw nsw i32 %437, 2, !dbg !13
  %439 = bitcast float %423 to i32, !dbg !13
  %440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %438, i32 %439), !dbg !13
  %441 = bitcast i32 %440 to float, !dbg !13
  %442 = select i1 %.not, float %421, float %441, !dbg !13
  %443 = select i1 %.not, float %441, float %422, !dbg !13
  %444 = select i1 %.not, float %425, float %424, !dbg !13
  %445 = bitcast float %444 to i32, !dbg !13
  %446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %438, i32 %445), !dbg !13
  %447 = bitcast i32 %446 to float, !dbg !13
  %448 = select i1 %.not, float %424, float %447, !dbg !13
  %449 = select i1 %.not, float %447, float %425, !dbg !13
  %450 = select i1 %.not1, float %448, float %442, !dbg !13
  %451 = xor i32 %432, 32, !dbg !13
  %452 = icmp samesign ult i32 %451, %435, !dbg !13
  %453 = select i1 %452, i32 %451, i32 %432, !dbg !13
  %454 = shl nuw nsw i32 %453, 2, !dbg !13
  %455 = bitcast float %450 to i32, !dbg !13
  %456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %455), !dbg !13
  %457 = bitcast i32 %456 to float, !dbg !13
  %458 = select i1 %.not1, float %442, float %457, !dbg !13
  %459 = select i1 %.not1, float %457, float %448, !dbg !13
  %460 = select i1 %.not1, float %449, float %443, !dbg !13
  %461 = bitcast float %460 to i32, !dbg !13
  %462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %461), !dbg !13
  %463 = bitcast i32 %462 to float, !dbg !13
  %464 = select i1 %.not1, float %443, float %463, !dbg !13
  %465 = select i1 %.not1, float %463, float %449, !dbg !13
  %466 = select i1 %.not, float %428, float %427, !dbg !13
  %467 = bitcast float %466 to i32, !dbg !13
  %468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %438, i32 %467), !dbg !13
  %469 = bitcast i32 %468 to float, !dbg !13
  %470 = select i1 %.not, float %427, float %469, !dbg !13
  %471 = select i1 %.not, float %469, float %428, !dbg !13
  %472 = select i1 %.not, float %430, float %429, !dbg !13
  %473 = bitcast float %472 to i32, !dbg !13
  %474 = call i32 @llvm.amdgcn.ds.bpermute(i32 %438, i32 %473), !dbg !13
  %475 = bitcast i32 %474 to float, !dbg !13
  %476 = select i1 %.not, float %429, float %475, !dbg !13
  %477 = select i1 %.not, float %475, float %430, !dbg !13
  %478 = select i1 %.not1, float %476, float %470, !dbg !13
  %479 = bitcast float %478 to i32, !dbg !13
  %480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %479), !dbg !13
  %481 = bitcast i32 %480 to float, !dbg !13
  %482 = select i1 %.not1, float %470, float %481, !dbg !13
  %483 = select i1 %.not1, float %481, float %476, !dbg !13
  %484 = select i1 %.not1, float %477, float %471, !dbg !13
  %485 = bitcast float %484 to i32, !dbg !13
  %486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %485), !dbg !13
  %487 = bitcast i32 %486 to float, !dbg !13
  %488 = select i1 %.not1, float %471, float %487, !dbg !13
  %489 = select i1 %.not1, float %487, float %477, !dbg !13
  %490 = or disjoint i64 %75, 31, !dbg !14
  %491 = icmp uge i64 %19, %490, !dbg !14
  %492 = and i1 %43, %491, !dbg !14
  br i1 %492, label %493, label %494, !dbg !14

493:                                              ; preds = %413
  br label %520, !dbg !14

494:                                              ; preds = %413
  %495 = add i64 %19, %35, !dbg !14
  %496 = shl nuw nsw i64 %36, 4, !dbg !14
  %497 = add i64 %495, %496, !dbg !14
  %498 = and i32 %34, 240, !dbg !14
  %index.compact = zext nneg i32 %498 to i64, !dbg !14
  %499 = sub i64 %497, %index.compact, !dbg !14
  %500 = add i64 %499, 1, !dbg !23
  %501 = or disjoint i64 %75, %38, !dbg !14
  %502 = lshr i64 %500, 32, !dbg !24
  %compare.high = trunc i64 %502 to i32, !dbg !24
  %503 = lshr i64 %501, 32, !dbg !24
  %compare.high391 = trunc i64 %503 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high391, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high391, !dbg !24
  %compare.low.base = trunc i64 %500 to i32, !dbg !24
  %compare.low.base392 = trunc i64 %501 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base392, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %504 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base392, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low393 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result394 = select i1 %compare.high.equal, i1 %compare.low393, i1 %compare.high.order, !dbg !24
  %505 = select i1 %compare.result394, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base392, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local396 = or i32 %compare.local.base395, 2, !dbg !24
  %compare.low397 = icmp ugt i32 %compare.low.base, %compare.local396, !dbg !24
  %compare.result398 = select i1 %compare.high.equal, i1 %compare.low397, i1 %compare.high.order, !dbg !24
  %506 = select i1 %compare.result398, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base392, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local400 = or i32 %compare.local.base399, 3, !dbg !24
  %compare.low401 = icmp ugt i32 %compare.low.base, %compare.local400, !dbg !24
  %compare.result402 = select i1 %compare.high.equal, i1 %compare.low401, i1 %compare.high.order, !dbg !24
  %507 = select i1 %compare.result402, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base392, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local404 = or i32 %compare.local.base403, 16, !dbg !24
  %compare.low405 = icmp ugt i32 %compare.low.base, %compare.local404, !dbg !24
  %compare.result406 = select i1 %compare.high.equal, i1 %compare.low405, i1 %compare.high.order, !dbg !24
  %508 = select i1 %compare.result406, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base392, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local408 = or i32 %compare.local.base407, 17, !dbg !24
  %compare.low409 = icmp ugt i32 %compare.low.base, %compare.local408, !dbg !24
  %compare.result410 = select i1 %compare.high.equal, i1 %compare.low409, i1 %compare.high.order, !dbg !24
  %509 = select i1 %compare.result410, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base392, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local412 = or i32 %compare.local.base411, 18, !dbg !24
  %compare.low413 = icmp ugt i32 %compare.low.base, %compare.local412, !dbg !24
  %compare.result414 = select i1 %compare.high.equal, i1 %compare.low413, i1 %compare.high.order, !dbg !24
  %510 = select i1 %compare.result414, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base392, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local416 = or i32 %compare.local.base415, 19, !dbg !24
  %compare.low417 = icmp ugt i32 %compare.low.base, %compare.local416, !dbg !24
  %compare.result418 = select i1 %compare.high.equal, i1 %compare.low417, i1 %compare.high.order, !dbg !24
  %511 = select i1 %compare.result418, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %512 = insertelement <8 x float> poison, float %504, i64 0, !dbg !14
  %513 = insertelement <8 x float> %512, float %505, i64 1, !dbg !14
  %514 = insertelement <8 x float> %513, float %506, i64 2, !dbg !14
  %515 = insertelement <8 x float> %514, float %507, i64 3, !dbg !14
  %516 = insertelement <8 x float> %515, float %508, i64 4, !dbg !14
  %517 = insertelement <8 x float> %516, float %509, i64 5, !dbg !14
  %518 = insertelement <8 x float> %517, float %510, i64 6, !dbg !14
  %519 = insertelement <8 x float> %518, float %511, i64 7, !dbg !14
  br label %520, !dbg !14

520:                                              ; preds = %493, %494
  %521 = phi <8 x float> [ %519, %494 ], [ zeroinitializer, %493 ], !dbg !14
  br label %522, !dbg !14

522:                                              ; preds = %520
  %523 = insertelement <4 x float> poison, float %458, i64 0, !dbg !26
  %524 = insertelement <4 x float> %523, float %464, i64 1, !dbg !26
  %525 = insertelement <4 x float> %524, float %459, i64 2, !dbg !26
  %526 = insertelement <4 x float> %525, float %465, i64 3, !dbg !26
  %527 = shufflevector <8 x float> %521, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %528 = fadd <4 x float> %526, %527, !dbg !26
  %529 = extractelement <4 x float> %528, i64 0, !dbg !26
  %530 = extractelement <4 x float> %528, i64 1, !dbg !26
  %531 = extractelement <4 x float> %528, i64 2, !dbg !26
  %532 = extractelement <4 x float> %528, i64 3, !dbg !26
  %533 = insertelement <4 x float> poison, float %482, i64 0, !dbg !26
  %534 = insertelement <4 x float> %533, float %488, i64 1, !dbg !26
  %535 = insertelement <4 x float> %534, float %483, i64 2, !dbg !26
  %536 = insertelement <4 x float> %535, float %489, i64 3, !dbg !26
  %537 = shufflevector <8 x float> %521, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %538 = fadd <4 x float> %536, %537, !dbg !26
  %539 = extractelement <4 x float> %538, i64 0, !dbg !26
  %540 = extractelement <4 x float> %538, i64 1, !dbg !26
  %541 = extractelement <4 x float> %538, i64 2, !dbg !26
  %542 = extractelement <4 x float> %538, i64 3, !dbg !26
  %543 = call float @llvm.maxnum.f32(float %529, float 0xFFF0000000000000), !dbg !27
  %544 = call float @llvm.maxnum.f32(float %543, float %530), !dbg !27
  %545 = call float @llvm.maxnum.f32(float %544, float %531), !dbg !27
  %546 = call float @llvm.maxnum.f32(float %545, float %532), !dbg !27
  %547 = call float @llvm.maxnum.f32(float %546, float %539), !dbg !27
  %548 = call float @llvm.maxnum.f32(float %547, float %540), !dbg !27
  %549 = call float @llvm.maxnum.f32(float %548, float %541), !dbg !27
  %550 = call float @llvm.maxnum.f32(float %549, float %542), !dbg !27
  %551 = bitcast float %550 to i32, !dbg !27
  %552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %438, i32 %551), !dbg !27
  %553 = bitcast i32 %552 to float, !dbg !27
  %554 = call float @llvm.maxnum.f32(float %550, float %553), !dbg !27
  %555 = bitcast float %554 to i32, !dbg !27
  %556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %555), !dbg !27
  %557 = bitcast i32 %556 to float, !dbg !27
  %558 = call float @llvm.maxnum.f32(float %554, float %557), !dbg !27
  %559 = insertelement <1 x float> poison, float %558, i64 0, !dbg !27
  %560 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %559, <1 x float> %63), !dbg !27
  %561 = shufflevector <1 x float> %560, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %562 = fsub <4 x float> %528, %561, !dbg !28
  %563 = fsub <4 x float> %538, %561, !dbg !28
  %564 = fsub <1 x float> %63, %560, !dbg !29
  %565 = extractelement <1 x float> %564, i64 0, !dbg !30
  %566 = call float @llvm.amdgcn.exp2.f32(float %565), !dbg !30
  %567 = extractelement <1 x float> %62, i64 0, !dbg !31
  %568 = fmul float %567, %566, !dbg !31
  %569 = extractelement <4 x float> %562, i64 0, !dbg !32
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !32
  %571 = insertelement <4 x float> poison, float %570, i64 0, !dbg !34
  %572 = extractelement <4 x float> %562, i64 1, !dbg !32
  %573 = call float @llvm.amdgcn.exp2.f32(float %572), !dbg !32
  %574 = insertelement <4 x float> %571, float %573, i64 1, !dbg !34
  %575 = fadd float %570, %573, !dbg !33
  %576 = extractelement <4 x float> %562, i64 2, !dbg !32
  %577 = call float @llvm.amdgcn.exp2.f32(float %576), !dbg !32
  %578 = insertelement <4 x float> %574, float %577, i64 2, !dbg !34
  %579 = fadd float %575, %577, !dbg !33
  %580 = extractelement <4 x float> %562, i64 3, !dbg !32
  %581 = call float @llvm.amdgcn.exp2.f32(float %580), !dbg !32
  %582 = insertelement <4 x float> %578, float %581, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %582) #9, !dbg !34, !frisk.register_lifetimes !1
  %583 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %583) #9, !dbg !33, !frisk.register_lifetimes !1
  %584 = fadd float %579, %581, !dbg !33
  %585 = extractelement <4 x float> %563, i64 0, !dbg !32
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !32
  %587 = insertelement <4 x float> poison, float %586, i64 0, !dbg !34
  %588 = fadd float %584, %586, !dbg !33
  %589 = extractelement <4 x float> %563, i64 1, !dbg !32
  %590 = call float @llvm.amdgcn.exp2.f32(float %589), !dbg !32
  %591 = insertelement <4 x float> %587, float %590, i64 1, !dbg !34
  %592 = fadd float %588, %590, !dbg !33
  %593 = extractelement <4 x float> %563, i64 2, !dbg !32
  %594 = call float @llvm.amdgcn.exp2.f32(float %593), !dbg !32
  %595 = insertelement <4 x float> %591, float %594, i64 2, !dbg !34
  %596 = fadd float %592, %594, !dbg !33
  %597 = extractelement <4 x float> %563, i64 3, !dbg !32
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !32
  %599 = insertelement <4 x float> %595, float %598, i64 3, !dbg !34
  %math.result26 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %599) #9, !dbg !34, !frisk.register_lifetimes !1
  %600 = fptrunc <4 x float> %math.result26 to <4 x half>, !dbg !34
  %math.packed27 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %600) #9, !dbg !33, !frisk.register_lifetimes !1
  %601 = fadd float %596, %598, !dbg !33
  %602 = bitcast float %601 to i32, !dbg !33
  %603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %438, i32 %602), !dbg !33
  %604 = bitcast i32 %603 to float, !dbg !33
  %605 = fadd float %601, %604, !dbg !33
  %606 = bitcast float %605 to i32, !dbg !33
  %607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %606), !dbg !33
  %608 = bitcast i32 %607 to float, !dbg !33
  %609 = fadd float %605, %608, !dbg !33
  %610 = fadd float %609, %568, !dbg !33
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread216 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local217 = shl nuw nsw i32 %address.thread216, 3, !dbg !15
  %address.local218 = and i32 %address.local217, 504, !dbg !15
  %address.local219 = zext nneg i32 %address.local218 to i64, !dbg !15
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !15
  %611 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local220, i32 0) #0, !dbg !15
  %612 = bitcast <4 x i32> %611 to <8 x half>, !dbg !15
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !15
  %address.local222 = getelementptr i8, ptr addrspace(3) %address.local221, i64 1024, !dbg !15
  %613 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local222, i32 0) #0, !dbg !15
  %614 = bitcast <4 x i32> %613 to <8 x half>, !dbg !15
  %615 = add nuw nsw i64 %92, %15, !dbg !18
  %616 = shl i64 %615, 19, !dbg !18
  %617 = and i64 %616, 16252928, !dbg !18
  %618 = add i64 %617, %98, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %619 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %618, !dbg !18
  %620 = getelementptr i8, ptr addrspace(1) %619, i64 64, !dbg !18
  br label %621, !dbg !18

621:                                              ; preds = %624, %522
  %622 = phi i64 [ %650, %624 ], [ %35, %522 ], !dbg !18
  %623 = icmp samesign ult i64 %622, 128, !dbg !18
  br i1 %623, label %624, label %651, !dbg !18

624:                                              ; preds = %621
  %625 = shl nuw nsw i64 %622, 3, !dbg !18
  %626 = and i64 %625, 24, !dbg !18
  %627 = shl nuw nsw i64 %622, 3, !dbg !18
  %628 = and i64 %627, 512, !dbg !18
  %629 = shl nuw nsw i64 %622, 5, !dbg !18
  %630 = and i64 %629, 3968, !dbg !18
  %631 = getelementptr [2 x i8], ptr addrspace(1) %620, i64 %630, !dbg !18
  %632 = getelementptr [2 x i8], ptr addrspace(1) %631, i64 %626, !dbg !18
  %633 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %628, !dbg !18
  %634 = ptrtoint ptr addrspace(3) %633 to i64, !dbg !18
  %635 = trunc i64 %634 to i32, !dbg !18
  %636 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %635), !dbg !18
  %637 = zext i32 %636 to i64, !dbg !18
  %638 = inttoptr i64 %637 to ptr addrspace(3), !dbg !18
  %639 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %640 = ptrtoint ptr addrspace(1) %632 to i64, !dbg !18
  %641 = sub i64 %640, %639, !dbg !18
  %dma.buffer.offset431 = trunc i64 %641 to i32, !dbg !18
  %642 = trunc i64 %639 to i32, !dbg !18
  %643 = insertelement <4 x i32> poison, i32 %642, i64 0, !dbg !18
  %644 = lshr i64 %639, 32, !dbg !18
  %645 = trunc i64 %644 to i32, !dbg !18
  %646 = insertelement <4 x i32> %643, i32 %645, i64 1, !dbg !18
  %647 = insertelement <4 x i32> %646, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc432 = insertelement <4 x i32> %647, i32 131072, i64 3, !dbg !18
  %648 = ptrtoint ptr addrspace(3) %638 to i32, !dbg !18
  %dma.buffer.lds433 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %648), !dbg !18
  %649 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds433, i32 %dma.buffer.offset431, <4 x i32> %dma.buffer.rsrc432) #10, !dbg !18
  %650 = or disjoint i64 %35, 256, !dbg !18
  br label %621, !dbg !18

651:                                              ; preds = %621
  %652 = shufflevector <8 x half> %612, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %653 = insertelement <4 x float> poison, float %566, i64 0, !dbg !35
  %654 = shufflevector <4 x float> %653, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %655 = fmul <4 x float> %59, %654, !dbg !35
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %652, <4 x float> %655) #9, !dbg !15
  %657 = shufflevector <8 x half> %614, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed27, <4 x half> %657, <4 x float> %656) #9, !dbg !15
  %659 = shufflevector <8 x half> %612, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %660 = fmul <4 x float> %60, %654, !dbg !35
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %659, <4 x float> %660) #9, !dbg !15
  %662 = shufflevector <8 x half> %614, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed27, <4 x half> %662, <4 x float> %661) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread223 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local224 = shl nuw nsw i32 %address.thread223, 3, !dbg !15
  %address.local225 = and i32 %address.local224, 504, !dbg !15
  %address.local226 = zext nneg i32 %address.local225 to i64, !dbg !15
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local226, !dbg !15
  %664 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local227, i32 0) #0, !dbg !15
  %665 = bitcast <4 x i32> %664 to <8 x half>, !dbg !15
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local226, !dbg !15
  %address.local229 = getelementptr i8, ptr addrspace(3) %address.local228, i64 1024, !dbg !15
  %666 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local229, i32 0) #0, !dbg !15
  %667 = bitcast <4 x i32> %666 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %668 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %618, !dbg !18
  %669 = getelementptr i8, ptr addrspace(1) %668, i64 128, !dbg !18
  br label %670, !dbg !18

670:                                              ; preds = %673, %651
  %671 = phi i64 [ %699, %673 ], [ %35, %651 ], !dbg !18
  %672 = icmp samesign ult i64 %671, 128, !dbg !18
  br i1 %672, label %673, label %700, !dbg !18

673:                                              ; preds = %670
  %674 = shl nuw nsw i64 %671, 3, !dbg !18
  %675 = and i64 %674, 24, !dbg !18
  %676 = shl nuw nsw i64 %671, 3, !dbg !18
  %677 = and i64 %676, 512, !dbg !18
  %678 = shl nuw nsw i64 %671, 5, !dbg !18
  %679 = and i64 %678, 3968, !dbg !18
  %680 = getelementptr [2 x i8], ptr addrspace(1) %669, i64 %679, !dbg !18
  %681 = getelementptr [2 x i8], ptr addrspace(1) %680, i64 %675, !dbg !18
  %682 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %677, !dbg !18
  %683 = ptrtoint ptr addrspace(3) %682 to i64, !dbg !18
  %684 = trunc i64 %683 to i32, !dbg !18
  %685 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %684), !dbg !18
  %686 = zext i32 %685 to i64, !dbg !18
  %687 = inttoptr i64 %686 to ptr addrspace(3), !dbg !18
  %688 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %689 = ptrtoint ptr addrspace(1) %681 to i64, !dbg !18
  %690 = sub i64 %689, %688, !dbg !18
  %dma.buffer.offset434 = trunc i64 %690 to i32, !dbg !18
  %691 = trunc i64 %688 to i32, !dbg !18
  %692 = insertelement <4 x i32> poison, i32 %691, i64 0, !dbg !18
  %693 = lshr i64 %688, 32, !dbg !18
  %694 = trunc i64 %693 to i32, !dbg !18
  %695 = insertelement <4 x i32> %692, i32 %694, i64 1, !dbg !18
  %696 = insertelement <4 x i32> %695, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc435 = insertelement <4 x i32> %696, i32 131072, i64 3, !dbg !18
  %697 = ptrtoint ptr addrspace(3) %687 to i32, !dbg !18
  %dma.buffer.lds436 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %697), !dbg !18
  %698 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds436, i32 %dma.buffer.offset434, <4 x i32> %dma.buffer.rsrc435) #10, !dbg !18
  %699 = or disjoint i64 %35, 256, !dbg !18
  br label %670, !dbg !18

700:                                              ; preds = %670
  %701 = shufflevector <8 x half> %665, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %702 = fmul <4 x float> %64, %654, !dbg !35
  %703 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %701, <4 x float> %702) #9, !dbg !15
  %704 = shufflevector <8 x half> %667, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %705 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed27, <4 x half> %704, <4 x float> %703) #9, !dbg !15
  %706 = shufflevector <8 x half> %665, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %707 = fmul <4 x float> %65, %654, !dbg !35
  %708 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %706, <4 x float> %707) #9, !dbg !15
  %709 = shufflevector <8 x half> %667, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %710 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed27, <4 x half> %709, <4 x float> %708) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread230 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local231 = shl nuw nsw i32 %address.thread230, 3, !dbg !15
  %address.local232 = and i32 %address.local231, 504, !dbg !15
  %address.local233 = zext nneg i32 %address.local232 to i64, !dbg !15
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local233, !dbg !15
  %711 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local234, i32 0) #0, !dbg !15
  %712 = bitcast <4 x i32> %711 to <8 x half>, !dbg !15
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local233, !dbg !15
  %address.local236 = getelementptr i8, ptr addrspace(3) %address.local235, i64 1024, !dbg !15
  %713 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local236, i32 0) #0, !dbg !15
  %714 = bitcast <4 x i32> %713 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %715 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %618, !dbg !18
  %716 = getelementptr i8, ptr addrspace(1) %715, i64 192, !dbg !18
  br label %717, !dbg !18

717:                                              ; preds = %720, %700
  %718 = phi i64 [ %746, %720 ], [ %35, %700 ], !dbg !18
  %719 = icmp samesign ult i64 %718, 128, !dbg !18
  br i1 %719, label %720, label %747, !dbg !18

720:                                              ; preds = %717
  %721 = shl nuw nsw i64 %718, 3, !dbg !18
  %722 = and i64 %721, 24, !dbg !18
  %723 = shl nuw nsw i64 %718, 3, !dbg !18
  %724 = and i64 %723, 512, !dbg !18
  %725 = shl nuw nsw i64 %718, 5, !dbg !18
  %726 = and i64 %725, 3968, !dbg !18
  %727 = getelementptr [2 x i8], ptr addrspace(1) %716, i64 %726, !dbg !18
  %728 = getelementptr [2 x i8], ptr addrspace(1) %727, i64 %722, !dbg !18
  %729 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %724, !dbg !18
  %730 = ptrtoint ptr addrspace(3) %729 to i64, !dbg !18
  %731 = trunc i64 %730 to i32, !dbg !18
  %732 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %731), !dbg !18
  %733 = zext i32 %732 to i64, !dbg !18
  %734 = inttoptr i64 %733 to ptr addrspace(3), !dbg !18
  %735 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %736 = ptrtoint ptr addrspace(1) %728 to i64, !dbg !18
  %737 = sub i64 %736, %735, !dbg !18
  %dma.buffer.offset437 = trunc i64 %737 to i32, !dbg !18
  %738 = trunc i64 %735 to i32, !dbg !18
  %739 = insertelement <4 x i32> poison, i32 %738, i64 0, !dbg !18
  %740 = lshr i64 %735, 32, !dbg !18
  %741 = trunc i64 %740 to i32, !dbg !18
  %742 = insertelement <4 x i32> %739, i32 %741, i64 1, !dbg !18
  %743 = insertelement <4 x i32> %742, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc438 = insertelement <4 x i32> %743, i32 131072, i64 3, !dbg !18
  %744 = ptrtoint ptr addrspace(3) %734 to i32, !dbg !18
  %dma.buffer.lds439 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %744), !dbg !18
  %745 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds439, i32 %dma.buffer.offset437, <4 x i32> %dma.buffer.rsrc438) #10, !dbg !18
  %746 = or disjoint i64 %35, 256, !dbg !18
  br label %717, !dbg !18

747:                                              ; preds = %717
  %748 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %749 = fmul <4 x float> %67, %654, !dbg !35
  %750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %748, <4 x float> %749) #9, !dbg !15
  %751 = shufflevector <8 x half> %714, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %752 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed27, <4 x half> %751, <4 x float> %750) #9, !dbg !15
  %753 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %754 = fmul <4 x float> %68, %654, !dbg !35
  %755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %753, <4 x float> %754) #9, !dbg !15
  %756 = shufflevector <8 x half> %714, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed27, <4 x half> %756, <4 x float> %755) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread237 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local238 = shl nuw nsw i32 %address.thread237, 3, !dbg !15
  %address.local239 = and i32 %address.local238, 504, !dbg !15
  %address.local240 = zext nneg i32 %address.local239 to i64, !dbg !15
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local240, !dbg !15
  %758 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local241, i32 0) #0, !dbg !15
  %759 = bitcast <4 x i32> %758 to <8 x half>, !dbg !15
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local240, !dbg !15
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local242, i64 1024, !dbg !15
  %760 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local243, i32 0) #0, !dbg !15
  %761 = bitcast <4 x i32> %760 to <8 x half>, !dbg !15
  %762 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %763 = fmul <4 x float> %70, %654, !dbg !35
  %764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %762, <4 x float> %763) #9, !dbg !15
  %765 = shufflevector <8 x half> %761, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed27, <4 x half> %765, <4 x float> %764) #9, !dbg !15
  %767 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %768 = fmul <4 x float> %71, %654, !dbg !35
  %769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %767, <4 x float> %768) #9, !dbg !15
  %770 = shufflevector <8 x half> %761, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed27, <4 x half> %770, <4 x float> %769) #9, !dbg !15
  %772 = shufflevector <4 x float> %658, <4 x float> %663, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %773 = insertelement <1 x float> poison, float %610, i64 0, !dbg !7
  %774 = shufflevector <4 x float> %705, <4 x float> %710, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %775 = shufflevector <4 x float> %752, <4 x float> %757, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %776 = shufflevector <4 x float> %766, <4 x float> %771, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %777 = add nuw nsw i64 %58, 1, !dbg !7
  br label %57, !dbg !7

778:                                              ; preds = %57
  %779 = extractelement <8 x float> %72, i64 0, !dbg !7
  %780 = extractelement <8 x float> %72, i64 1, !dbg !7
  %781 = extractelement <8 x float> %72, i64 2, !dbg !7
  %782 = extractelement <8 x float> %72, i64 3, !dbg !7
  %783 = select i1 %.not, float %780, float %779, !dbg !7
  %784 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %785 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %784), !dbg !7
  %786 = add nuw nsw i32 %785, 64, !dbg !7
  %787 = xor i32 %785, 16, !dbg !7
  %788 = and i32 %786, 192, !dbg !7
  %789 = icmp samesign ult i32 %787, %788, !dbg !7
  %790 = select i1 %789, i32 %787, i32 %785, !dbg !7
  %791 = shl nuw nsw i32 %790, 2, !dbg !7
  %792 = bitcast float %783 to i32, !dbg !7
  %793 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %792), !dbg !7
  %794 = bitcast i32 %793 to float, !dbg !7
  %795 = select i1 %.not, float %779, float %794, !dbg !7
  %796 = select i1 %.not, float %794, float %780, !dbg !7
  %797 = select i1 %.not, float %782, float %781, !dbg !7
  %798 = bitcast float %797 to i32, !dbg !7
  %799 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %798), !dbg !7
  %800 = bitcast i32 %799 to float, !dbg !7
  %801 = select i1 %.not, float %781, float %800, !dbg !7
  %802 = select i1 %.not, float %800, float %782, !dbg !7
  %803 = select i1 %.not1, float %801, float %795, !dbg !7
  %804 = xor i32 %785, 32, !dbg !7
  %805 = icmp samesign ult i32 %804, %788, !dbg !7
  %806 = select i1 %805, i32 %804, i32 %785, !dbg !7
  %807 = shl nuw nsw i32 %806, 2, !dbg !7
  %808 = bitcast float %803 to i32, !dbg !7
  %809 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %808), !dbg !7
  %810 = bitcast i32 %809 to float, !dbg !7
  %811 = select i1 %.not1, float %795, float %810, !dbg !7
  %812 = select i1 %.not1, float %810, float %801, !dbg !7
  %813 = select i1 %.not1, float %802, float %796, !dbg !7
  %814 = bitcast float %813 to i32, !dbg !7
  %815 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %814), !dbg !7
  %816 = bitcast i32 %815 to float, !dbg !7
  %817 = select i1 %.not1, float %796, float %816, !dbg !7
  %818 = select i1 %.not1, float %816, float %802, !dbg !7
  %819 = extractelement <8 x float> %69, i64 0, !dbg !7
  %820 = extractelement <8 x float> %69, i64 1, !dbg !7
  %821 = extractelement <8 x float> %69, i64 2, !dbg !7
  %822 = extractelement <8 x float> %69, i64 3, !dbg !7
  %823 = select i1 %.not, float %820, float %819, !dbg !7
  %824 = bitcast float %823 to i32, !dbg !7
  %825 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %824), !dbg !7
  %826 = bitcast i32 %825 to float, !dbg !7
  %827 = select i1 %.not, float %819, float %826, !dbg !7
  %828 = select i1 %.not, float %826, float %820, !dbg !7
  %829 = select i1 %.not, float %822, float %821, !dbg !7
  %830 = bitcast float %829 to i32, !dbg !7
  %831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %830), !dbg !7
  %832 = bitcast i32 %831 to float, !dbg !7
  %833 = select i1 %.not, float %821, float %832, !dbg !7
  %834 = select i1 %.not, float %832, float %822, !dbg !7
  %835 = select i1 %.not1, float %833, float %827, !dbg !7
  %836 = bitcast float %835 to i32, !dbg !7
  %837 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %836), !dbg !7
  %838 = bitcast i32 %837 to float, !dbg !7
  %839 = select i1 %.not1, float %827, float %838, !dbg !7
  %840 = select i1 %.not1, float %838, float %833, !dbg !7
  %841 = select i1 %.not1, float %834, float %828, !dbg !7
  %842 = bitcast float %841 to i32, !dbg !7
  %843 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %842), !dbg !7
  %844 = bitcast i32 %843 to float, !dbg !7
  %845 = select i1 %.not1, float %828, float %844, !dbg !7
  %846 = select i1 %.not1, float %844, float %834, !dbg !7
  %847 = extractelement <8 x float> %66, i64 0, !dbg !7
  %848 = extractelement <8 x float> %66, i64 1, !dbg !7
  %849 = extractelement <8 x float> %66, i64 2, !dbg !7
  %850 = extractelement <8 x float> %66, i64 3, !dbg !7
  %851 = select i1 %.not, float %848, float %847, !dbg !7
  %852 = bitcast float %851 to i32, !dbg !7
  %853 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %852), !dbg !7
  %854 = bitcast i32 %853 to float, !dbg !7
  %855 = select i1 %.not, float %847, float %854, !dbg !7
  %856 = select i1 %.not, float %854, float %848, !dbg !7
  %857 = select i1 %.not, float %850, float %849, !dbg !7
  %858 = bitcast float %857 to i32, !dbg !7
  %859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %858), !dbg !7
  %860 = bitcast i32 %859 to float, !dbg !7
  %861 = select i1 %.not, float %849, float %860, !dbg !7
  %862 = select i1 %.not, float %860, float %850, !dbg !7
  %863 = select i1 %.not1, float %861, float %855, !dbg !7
  %864 = bitcast float %863 to i32, !dbg !7
  %865 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %864), !dbg !7
  %866 = bitcast i32 %865 to float, !dbg !7
  %867 = select i1 %.not1, float %855, float %866, !dbg !7
  %868 = select i1 %.not1, float %866, float %861, !dbg !7
  %869 = select i1 %.not1, float %862, float %856, !dbg !7
  %870 = bitcast float %869 to i32, !dbg !7
  %871 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %870), !dbg !7
  %872 = bitcast i32 %871 to float, !dbg !7
  %873 = select i1 %.not1, float %856, float %872, !dbg !7
  %874 = select i1 %.not1, float %872, float %862, !dbg !7
  %875 = extractelement <8 x float> %61, i64 0, !dbg !7
  %876 = extractelement <8 x float> %61, i64 1, !dbg !7
  %877 = extractelement <8 x float> %61, i64 2, !dbg !7
  %878 = extractelement <8 x float> %61, i64 3, !dbg !7
  %879 = select i1 %.not, float %876, float %875, !dbg !7
  %880 = bitcast float %879 to i32, !dbg !7
  %881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %880), !dbg !7
  %882 = bitcast i32 %881 to float, !dbg !7
  %883 = select i1 %.not, float %875, float %882, !dbg !7
  %884 = select i1 %.not, float %882, float %876, !dbg !7
  %885 = select i1 %.not, float %878, float %877, !dbg !7
  %886 = bitcast float %885 to i32, !dbg !7
  %887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %886), !dbg !7
  %888 = bitcast i32 %887 to float, !dbg !7
  %889 = select i1 %.not, float %877, float %888, !dbg !7
  %890 = select i1 %.not, float %888, float %878, !dbg !7
  %891 = select i1 %.not1, float %889, float %883, !dbg !7
  %892 = bitcast float %891 to i32, !dbg !7
  %893 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %892), !dbg !7
  %894 = select i1 %.not1, float %890, float %884, !dbg !7
  %895 = bitcast float %894 to i32, !dbg !7
  %896 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %895), !dbg !7
  %897 = shufflevector <1 x float> %62, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %898 = extractelement <8 x float> %72, i64 4, !dbg !7
  %899 = extractelement <8 x float> %72, i64 5, !dbg !7
  %900 = extractelement <8 x float> %72, i64 6, !dbg !7
  %901 = extractelement <8 x float> %72, i64 7, !dbg !7
  %902 = select i1 %.not, float %899, float %898, !dbg !7
  %903 = bitcast float %902 to i32, !dbg !7
  %904 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %903), !dbg !7
  %905 = bitcast i32 %904 to float, !dbg !7
  %906 = select i1 %.not, float %898, float %905, !dbg !7
  %907 = select i1 %.not, float %905, float %899, !dbg !7
  %908 = select i1 %.not, float %901, float %900, !dbg !7
  %909 = bitcast float %908 to i32, !dbg !7
  %910 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %909), !dbg !7
  %911 = bitcast i32 %910 to float, !dbg !7
  %912 = select i1 %.not, float %900, float %911, !dbg !7
  %913 = select i1 %.not, float %911, float %901, !dbg !7
  %914 = select i1 %.not1, float %912, float %906, !dbg !7
  %915 = bitcast float %914 to i32, !dbg !7
  %916 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %915), !dbg !7
  %917 = bitcast i32 %916 to float, !dbg !7
  %918 = select i1 %.not1, float %906, float %917, !dbg !7
  %919 = select i1 %.not1, float %917, float %912, !dbg !7
  %920 = select i1 %.not1, float %913, float %907, !dbg !7
  %921 = bitcast float %920 to i32, !dbg !7
  %922 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %921), !dbg !7
  %923 = bitcast i32 %922 to float, !dbg !7
  %924 = select i1 %.not1, float %907, float %923, !dbg !7
  %925 = select i1 %.not1, float %923, float %913, !dbg !7
  %926 = extractelement <8 x float> %69, i64 4, !dbg !7
  %927 = extractelement <8 x float> %69, i64 5, !dbg !7
  %928 = extractelement <8 x float> %69, i64 6, !dbg !7
  %929 = extractelement <8 x float> %69, i64 7, !dbg !7
  %930 = select i1 %.not, float %927, float %926, !dbg !7
  %931 = bitcast float %930 to i32, !dbg !7
  %932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %931), !dbg !7
  %933 = bitcast i32 %932 to float, !dbg !7
  %934 = select i1 %.not, float %926, float %933, !dbg !7
  %935 = select i1 %.not, float %933, float %927, !dbg !7
  %936 = select i1 %.not, float %929, float %928, !dbg !7
  %937 = bitcast float %936 to i32, !dbg !7
  %938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %937), !dbg !7
  %939 = bitcast i32 %938 to float, !dbg !7
  %940 = select i1 %.not, float %928, float %939, !dbg !7
  %941 = select i1 %.not, float %939, float %929, !dbg !7
  %942 = select i1 %.not1, float %940, float %934, !dbg !7
  %943 = bitcast float %942 to i32, !dbg !7
  %944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %943), !dbg !7
  %945 = bitcast i32 %944 to float, !dbg !7
  %946 = select i1 %.not1, float %934, float %945, !dbg !7
  %947 = select i1 %.not1, float %945, float %940, !dbg !7
  %948 = select i1 %.not1, float %941, float %935, !dbg !7
  %949 = bitcast float %948 to i32, !dbg !7
  %950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %949), !dbg !7
  %951 = bitcast i32 %950 to float, !dbg !7
  %952 = select i1 %.not1, float %935, float %951, !dbg !7
  %953 = select i1 %.not1, float %951, float %941, !dbg !7
  %954 = extractelement <8 x float> %66, i64 4, !dbg !7
  %955 = extractelement <8 x float> %66, i64 5, !dbg !7
  %956 = extractelement <8 x float> %66, i64 6, !dbg !7
  %957 = extractelement <8 x float> %66, i64 7, !dbg !7
  %958 = select i1 %.not, float %955, float %954, !dbg !7
  %959 = bitcast float %958 to i32, !dbg !7
  %960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %959), !dbg !7
  %961 = bitcast i32 %960 to float, !dbg !7
  %962 = select i1 %.not, float %954, float %961, !dbg !7
  %963 = select i1 %.not, float %961, float %955, !dbg !7
  %964 = select i1 %.not, float %957, float %956, !dbg !7
  %965 = bitcast float %964 to i32, !dbg !7
  %966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %965), !dbg !7
  %967 = bitcast i32 %966 to float, !dbg !7
  %968 = select i1 %.not, float %956, float %967, !dbg !7
  %969 = select i1 %.not, float %967, float %957, !dbg !7
  %970 = select i1 %.not1, float %968, float %962, !dbg !7
  %971 = bitcast float %970 to i32, !dbg !7
  %972 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %971), !dbg !7
  %973 = bitcast i32 %972 to float, !dbg !7
  %974 = select i1 %.not1, float %962, float %973, !dbg !7
  %975 = select i1 %.not1, float %973, float %968, !dbg !7
  %976 = select i1 %.not1, float %969, float %963, !dbg !7
  %977 = bitcast float %976 to i32, !dbg !7
  %978 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %977), !dbg !7
  %979 = bitcast i32 %978 to float, !dbg !7
  %980 = select i1 %.not1, float %963, float %979, !dbg !7
  %981 = select i1 %.not1, float %979, float %969, !dbg !7
  %982 = extractelement <8 x float> %61, i64 4, !dbg !7
  %983 = extractelement <8 x float> %61, i64 5, !dbg !7
  %984 = extractelement <8 x float> %61, i64 6, !dbg !7
  %985 = extractelement <8 x float> %61, i64 7, !dbg !7
  %986 = select i1 %.not, float %983, float %982, !dbg !7
  %987 = bitcast float %986 to i32, !dbg !7
  %988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %987), !dbg !7
  %989 = bitcast i32 %988 to float, !dbg !7
  %990 = select i1 %.not, float %982, float %989, !dbg !7
  %991 = select i1 %.not, float %989, float %983, !dbg !7
  %992 = select i1 %.not, float %985, float %984, !dbg !7
  %993 = bitcast float %992 to i32, !dbg !7
  %994 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %993), !dbg !7
  %995 = bitcast i32 %994 to float, !dbg !7
  %996 = select i1 %.not, float %984, float %995, !dbg !7
  %997 = select i1 %.not, float %995, float %985, !dbg !7
  %998 = select i1 %.not1, float %996, float %990, !dbg !7
  %999 = bitcast float %998 to i32, !dbg !7
  %1000 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %999), !dbg !7
  %1001 = select i1 %.not1, float %997, float %991, !dbg !7
  %1002 = bitcast float %1001 to i32, !dbg !7
  %1003 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %1002), !dbg !7
  br i1 true, label %1004, label %1038, !dbg !21

1004:                                             ; preds = %778
  %1005 = bitcast i32 %1000 to float, !dbg !7
  %1006 = select i1 %.not1, float %990, float %1005, !dbg !7
  %1007 = insertelement <4 x float> poison, float %1006, i64 0, !dbg !19
  %1008 = bitcast i32 %1003 to float, !dbg !7
  %1009 = select i1 %.not1, float %991, float %1008, !dbg !7
  %1010 = insertelement <4 x float> %1007, float %1009, i64 1, !dbg !19
  %1011 = select i1 %.not1, float %1005, float %996, !dbg !7
  %1012 = insertelement <4 x float> %1010, float %1011, i64 2, !dbg !19
  %1013 = select i1 %.not1, float %1008, float %997, !dbg !7
  %1014 = insertelement <4 x float> %1012, float %1013, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %897, !dbg !19
  %normalized = fmul <4 x float> %1014, %row.reciprocal, !dbg !19
  %1015 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1016 = extractelement <4 x half> %1015, i64 3, !dbg !20
  %1017 = extractelement <4 x half> %1015, i64 2, !dbg !20
  %1018 = extractelement <4 x half> %1015, i64 1, !dbg !20
  %1019 = extractelement <4 x half> %1015, i64 0, !dbg !20
  %1020 = bitcast i32 %893 to float, !dbg !7
  %1021 = select i1 %.not1, float %883, float %1020, !dbg !7
  %1022 = insertelement <4 x float> poison, float %1021, i64 0, !dbg !19
  %1023 = bitcast i32 %896 to float, !dbg !7
  %1024 = select i1 %.not1, float %884, float %1023, !dbg !7
  %1025 = insertelement <4 x float> %1022, float %1024, i64 1, !dbg !19
  %1026 = select i1 %.not1, float %1020, float %889, !dbg !7
  %1027 = insertelement <4 x float> %1025, float %1026, i64 2, !dbg !19
  %1028 = select i1 %.not1, float %1023, float %890, !dbg !7
  %1029 = insertelement <4 x float> %1027, float %1028, i64 3, !dbg !19
  %normalized12 = fmul <4 x float> %1029, %row.reciprocal, !dbg !19
  %1030 = fptrunc <4 x float> %normalized12 to <4 x half>, !dbg !20
  %1031 = extractelement <4 x half> %1030, i64 3, !dbg !20
  %1032 = extractelement <4 x half> %1030, i64 2, !dbg !20
  %1033 = extractelement <4 x half> %1030, i64 1, !dbg !20
  %1034 = extractelement <4 x half> %1030, i64 0, !dbg !20
  %1035 = add nsw i64 %15, %25, !dbg !21
  %1036 = shl i64 %1035, 19, !dbg !21
  %1037 = and i64 %1036, 16252928, !dbg !21
  %address.thread244 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local245 = zext nneg i32 %address.thread244 to i64, !dbg !12
  %address.local246 = add i64 %28, %address.local245, !dbg !21
  %address.local247 = lshr i64 %address.local245, 6, !dbg !12
  %address.local248 = shl nuw nsw i64 %address.local247, 4, !dbg !21
  %address.local249 = add i64 %address.local246, %address.local248, !dbg !21
  %address.local250 = and i64 %address.local245, 240, !dbg !21
  %address.local251 = sub i64 %address.local249, %address.local250, !dbg !21
  %address.local252 = add i64 %address.local251, %29, !dbg !21
  %address.local253 = shl nuw nsw i64 %address.local252, 7, !dbg !21
  %address.local254 = add nuw nsw i64 %1037, %address.local253, !dbg !21
  %address.local255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local254, !dbg !21
  %address.local256 = lshr i64 %address.local245, 2, !dbg !12
  %address.local257 = and i64 %address.local256, 12, !dbg !12
  %address.local258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local255, i64 %address.local257, !dbg !21
  store half %1034, ptr addrspace(1) %address.local258, align 2, !dbg !21
  %address.local259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local254, !dbg !21
  %address.local260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local259, i64 %address.local257, !dbg !21
  %address.local261 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local260, i64 2, !dbg !21
  store half %1033, ptr addrspace(1) %address.local261, align 2, !dbg !21
  %address.local262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local254, !dbg !21
  %address.local263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local262, i64 %address.local257, !dbg !21
  %address.local264 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local263, i64 4, !dbg !21
  store half %1032, ptr addrspace(1) %address.local264, align 2, !dbg !21
  %address.local265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local254, !dbg !21
  %address.local266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local265, i64 %address.local257, !dbg !21
  %address.local267 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local266, i64 6, !dbg !21
  store half %1031, ptr addrspace(1) %address.local267, align 2, !dbg !21
  %address.local268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local254, !dbg !21
  %address.local269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local268, i64 %address.local257, !dbg !21
  %address.local270 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local269, i64 32, !dbg !21
  store half %1019, ptr addrspace(1) %address.local270, align 2, !dbg !21
  %address.local271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local254, !dbg !21
  %address.local272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local271, i64 %address.local257, !dbg !21
  %address.local273 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local272, i64 34, !dbg !21
  store half %1018, ptr addrspace(1) %address.local273, align 2, !dbg !21
  %address.local274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local254, !dbg !21
  %address.local275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local274, i64 %address.local257, !dbg !21
  %address.local276 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local275, i64 36, !dbg !21
  store half %1017, ptr addrspace(1) %address.local276, align 2, !dbg !21
  %address.local277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local254, !dbg !21
  %address.local278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local277, i64 %address.local257, !dbg !21
  %address.local279 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local278, i64 38, !dbg !21
  store half %1016, ptr addrspace(1) %address.local279, align 2, !dbg !21
  br label %1038, !dbg !21

1038:                                             ; preds = %1004, %778
  br i1 true, label %1039, label %1059, !dbg !21

1039:                                             ; preds = %1038
  %1040 = insertelement <4 x float> poison, float %974, i64 0, !dbg !19
  %1041 = insertelement <4 x float> %1040, float %980, i64 1, !dbg !19
  %1042 = insertelement <4 x float> %1041, float %975, i64 2, !dbg !19
  %1043 = insertelement <4 x float> %1042, float %981, i64 3, !dbg !19
  %row.reciprocal13 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %897, !dbg !19
  %normalized14 = fmul <4 x float> %1043, %row.reciprocal13, !dbg !19
  %1044 = fptrunc <4 x float> %normalized14 to <4 x half>, !dbg !20
  %1045 = extractelement <4 x half> %1044, i64 3, !dbg !20
  %1046 = extractelement <4 x half> %1044, i64 2, !dbg !20
  %1047 = extractelement <4 x half> %1044, i64 1, !dbg !20
  %1048 = extractelement <4 x half> %1044, i64 0, !dbg !20
  %1049 = insertelement <4 x float> poison, float %867, i64 0, !dbg !19
  %1050 = insertelement <4 x float> %1049, float %873, i64 1, !dbg !19
  %1051 = insertelement <4 x float> %1050, float %868, i64 2, !dbg !19
  %1052 = insertelement <4 x float> %1051, float %874, i64 3, !dbg !19
  %normalized15 = fmul <4 x float> %1052, %row.reciprocal13, !dbg !19
  %1053 = fptrunc <4 x float> %normalized15 to <4 x half>, !dbg !20
  %1054 = extractelement <4 x half> %1053, i64 3, !dbg !20
  %1055 = extractelement <4 x half> %1053, i64 2, !dbg !20
  %1056 = extractelement <4 x half> %1053, i64 1, !dbg !20
  %1057 = extractelement <4 x half> %1053, i64 0, !dbg !20
  %1058 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread280 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local281 = zext nneg i32 %address.thread280 to i64, !dbg !12
  %address.local282 = add i64 %28, %address.local281, !dbg !21
  %address.local283 = lshr i64 %address.local281, 6, !dbg !12
  %address.local284 = shl nuw nsw i64 %address.local283, 4, !dbg !21
  %address.local285 = add i64 %address.local282, %address.local284, !dbg !21
  %address.local286 = and i64 %address.local281, 240, !dbg !21
  %address.local287 = sub i64 %address.local285, %address.local286, !dbg !21
  %address.local288 = add i64 %address.local287, %29, !dbg !21
  %address.local289 = shl nuw nsw i64 %address.local288, 7, !dbg !21
  %address.local290 = add nuw nsw i64 %1058, %address.local289, !dbg !21
  %address.local291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local292 = lshr i64 %address.local281, 2, !dbg !12
  %address.local293 = and i64 %address.local292, 12, !dbg !12
  %address.local294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local291, i64 %address.local293, !dbg !21
  %address.local295 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local294, i64 64, !dbg !21
  store half %1057, ptr addrspace(1) %address.local295, align 2, !dbg !21
  %address.local296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local296, i64 %address.local293, !dbg !21
  %address.local298 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local297, i64 66, !dbg !21
  store half %1056, ptr addrspace(1) %address.local298, align 2, !dbg !21
  %address.local299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local299, i64 %address.local293, !dbg !21
  %address.local301 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local300, i64 68, !dbg !21
  store half %1055, ptr addrspace(1) %address.local301, align 2, !dbg !21
  %address.local302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local302, i64 %address.local293, !dbg !21
  %address.local304 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local303, i64 70, !dbg !21
  store half %1054, ptr addrspace(1) %address.local304, align 2, !dbg !21
  %address.local305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local306 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local305, i64 %address.local293, !dbg !21
  %address.local307 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local306, i64 96, !dbg !21
  store half %1048, ptr addrspace(1) %address.local307, align 2, !dbg !21
  %address.local308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local308, i64 %address.local293, !dbg !21
  %address.local310 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local309, i64 98, !dbg !21
  store half %1047, ptr addrspace(1) %address.local310, align 2, !dbg !21
  %address.local311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local311, i64 %address.local293, !dbg !21
  %address.local313 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local312, i64 100, !dbg !21
  store half %1046, ptr addrspace(1) %address.local313, align 2, !dbg !21
  %address.local314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local315 = or i64 %address.local292, 51, !dbg !21
  %address.local316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local314, i64 %address.local315, !dbg !21
  store half %1045, ptr addrspace(1) %address.local316, align 2, !dbg !21
  br label %1059, !dbg !21

1059:                                             ; preds = %1039, %1038
  br i1 true, label %1060, label %1080, !dbg !21

1060:                                             ; preds = %1059
  %1061 = insertelement <4 x float> poison, float %946, i64 0, !dbg !19
  %1062 = insertelement <4 x float> %1061, float %952, i64 1, !dbg !19
  %1063 = insertelement <4 x float> %1062, float %947, i64 2, !dbg !19
  %1064 = insertelement <4 x float> %1063, float %953, i64 3, !dbg !19
  %row.reciprocal16 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %897, !dbg !19
  %normalized17 = fmul <4 x float> %1064, %row.reciprocal16, !dbg !19
  %1065 = fptrunc <4 x float> %normalized17 to <4 x half>, !dbg !20
  %1066 = extractelement <4 x half> %1065, i64 3, !dbg !20
  %1067 = extractelement <4 x half> %1065, i64 2, !dbg !20
  %1068 = extractelement <4 x half> %1065, i64 1, !dbg !20
  %1069 = extractelement <4 x half> %1065, i64 0, !dbg !20
  %1070 = insertelement <4 x float> poison, float %839, i64 0, !dbg !19
  %1071 = insertelement <4 x float> %1070, float %845, i64 1, !dbg !19
  %1072 = insertelement <4 x float> %1071, float %840, i64 2, !dbg !19
  %1073 = insertelement <4 x float> %1072, float %846, i64 3, !dbg !19
  %normalized18 = fmul <4 x float> %1073, %row.reciprocal16, !dbg !19
  %1074 = fptrunc <4 x float> %normalized18 to <4 x half>, !dbg !20
  %1075 = extractelement <4 x half> %1074, i64 3, !dbg !20
  %1076 = extractelement <4 x half> %1074, i64 2, !dbg !20
  %1077 = extractelement <4 x half> %1074, i64 1, !dbg !20
  %1078 = extractelement <4 x half> %1074, i64 0, !dbg !20
  %1079 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread317 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local318 = zext nneg i32 %address.thread317 to i64, !dbg !12
  %address.local319 = add i64 %28, %address.local318, !dbg !21
  %address.local320 = lshr i64 %address.local318, 6, !dbg !12
  %address.local321 = shl nuw nsw i64 %address.local320, 4, !dbg !21
  %address.local322 = add i64 %address.local319, %address.local321, !dbg !21
  %address.local323 = and i64 %address.local318, 240, !dbg !21
  %address.local324 = sub i64 %address.local322, %address.local323, !dbg !21
  %address.local325 = add i64 %address.local324, %29, !dbg !21
  %address.local326 = shl nuw nsw i64 %address.local325, 7, !dbg !21
  %address.local327 = add nuw nsw i64 %1079, %address.local326, !dbg !21
  %address.local328 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local327, !dbg !21
  %address.local329 = lshr i64 %address.local318, 2, !dbg !12
  %address.local330 = and i64 %address.local329, 12, !dbg !12
  %address.local331 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local328, i64 %address.local330, !dbg !21
  %address.local332 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local331, i64 128, !dbg !21
  store half %1078, ptr addrspace(1) %address.local332, align 2, !dbg !21
  %address.local333 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local327, !dbg !21
  %address.local334 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local333, i64 %address.local330, !dbg !21
  %address.local335 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local334, i64 130, !dbg !21
  store half %1077, ptr addrspace(1) %address.local335, align 2, !dbg !21
  %address.local336 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local327, !dbg !21
  %address.local337 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local336, i64 %address.local330, !dbg !21
  %address.local338 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local337, i64 132, !dbg !21
  store half %1076, ptr addrspace(1) %address.local338, align 2, !dbg !21
  %address.local339 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local327, !dbg !21
  %address.local340 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local339, i64 %address.local330, !dbg !21
  %address.local341 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local340, i64 134, !dbg !21
  store half %1075, ptr addrspace(1) %address.local341, align 2, !dbg !21
  %address.local342 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local327, !dbg !21
  %address.local343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local342, i64 %address.local330, !dbg !21
  %address.local344 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local343, i64 160, !dbg !21
  store half %1069, ptr addrspace(1) %address.local344, align 2, !dbg !21
  %address.local345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local327, !dbg !21
  %address.local346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local345, i64 %address.local330, !dbg !21
  %address.local347 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local346, i64 162, !dbg !21
  store half %1068, ptr addrspace(1) %address.local347, align 2, !dbg !21
  %address.local348 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local327, !dbg !21
  %address.local349 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local348, i64 %address.local330, !dbg !21
  %address.local350 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local349, i64 164, !dbg !21
  store half %1067, ptr addrspace(1) %address.local350, align 2, !dbg !21
  %address.local351 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local327, !dbg !21
  %address.local352 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local351, i64 %address.local330, !dbg !21
  %address.local353 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local352, i64 166, !dbg !21
  store half %1066, ptr addrspace(1) %address.local353, align 2, !dbg !21
  br label %1080, !dbg !21

1080:                                             ; preds = %1060, %1059
  br i1 true, label %1081, label %1101, !dbg !21

1081:                                             ; preds = %1080
  %1082 = insertelement <4 x float> poison, float %918, i64 0, !dbg !19
  %1083 = insertelement <4 x float> %1082, float %924, i64 1, !dbg !19
  %1084 = insertelement <4 x float> %1083, float %919, i64 2, !dbg !19
  %1085 = insertelement <4 x float> %1084, float %925, i64 3, !dbg !19
  %row.reciprocal19 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %897, !dbg !19
  %normalized20 = fmul <4 x float> %1085, %row.reciprocal19, !dbg !19
  %1086 = fptrunc <4 x float> %normalized20 to <4 x half>, !dbg !20
  %1087 = extractelement <4 x half> %1086, i64 3, !dbg !20
  %1088 = extractelement <4 x half> %1086, i64 2, !dbg !20
  %1089 = extractelement <4 x half> %1086, i64 1, !dbg !20
  %1090 = extractelement <4 x half> %1086, i64 0, !dbg !20
  %1091 = insertelement <4 x float> poison, float %811, i64 0, !dbg !19
  %1092 = insertelement <4 x float> %1091, float %817, i64 1, !dbg !19
  %1093 = insertelement <4 x float> %1092, float %812, i64 2, !dbg !19
  %1094 = insertelement <4 x float> %1093, float %818, i64 3, !dbg !19
  %normalized21 = fmul <4 x float> %1094, %row.reciprocal19, !dbg !19
  %1095 = fptrunc <4 x float> %normalized21 to <4 x half>, !dbg !20
  %1096 = extractelement <4 x half> %1095, i64 3, !dbg !20
  %1097 = extractelement <4 x half> %1095, i64 2, !dbg !20
  %1098 = extractelement <4 x half> %1095, i64 1, !dbg !20
  %1099 = extractelement <4 x half> %1095, i64 0, !dbg !20
  %1100 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread354 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local355 = zext nneg i32 %address.thread354 to i64, !dbg !12
  %address.local356 = add i64 %28, %address.local355, !dbg !21
  %address.local357 = lshr i64 %address.local355, 6, !dbg !12
  %address.local358 = shl nuw nsw i64 %address.local357, 4, !dbg !21
  %address.local359 = add i64 %address.local356, %address.local358, !dbg !21
  %address.local360 = and i64 %address.local355, 240, !dbg !21
  %address.local361 = sub i64 %address.local359, %address.local360, !dbg !21
  %address.local362 = add i64 %address.local361, %29, !dbg !21
  %address.local363 = shl nuw nsw i64 %address.local362, 7, !dbg !21
  %address.local364 = add nuw nsw i64 %1100, %address.local363, !dbg !21
  %address.local365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local364, !dbg !21
  %address.local366 = lshr i64 %address.local355, 2, !dbg !12
  %address.local367 = and i64 %address.local366, 12, !dbg !12
  %address.local368 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local365, i64 %address.local367, !dbg !21
  %address.local369 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local368, i64 192, !dbg !21
  store half %1099, ptr addrspace(1) %address.local369, align 2, !dbg !21
  %address.local370 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local364, !dbg !21
  %address.local371 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local370, i64 %address.local367, !dbg !21
  %address.local372 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local371, i64 194, !dbg !21
  store half %1098, ptr addrspace(1) %address.local372, align 2, !dbg !21
  %address.local373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local364, !dbg !21
  %address.local374 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local373, i64 %address.local367, !dbg !21
  %address.local375 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local374, i64 196, !dbg !21
  store half %1097, ptr addrspace(1) %address.local375, align 2, !dbg !21
  %address.local376 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local364, !dbg !21
  %address.local377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local376, i64 %address.local367, !dbg !21
  %address.local378 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local377, i64 198, !dbg !21
  store half %1096, ptr addrspace(1) %address.local378, align 2, !dbg !21
  %address.local379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local364, !dbg !21
  %address.local380 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local379, i64 %address.local367, !dbg !21
  %address.local381 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local380, i64 224, !dbg !21
  store half %1090, ptr addrspace(1) %address.local381, align 2, !dbg !21
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local364, !dbg !21
  %address.local383 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local382, i64 %address.local367, !dbg !21
  %address.local384 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local383, i64 226, !dbg !21
  store half %1089, ptr addrspace(1) %address.local384, align 2, !dbg !21
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local364, !dbg !21
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local385, i64 %address.local367, !dbg !21
  %address.local387 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local386, i64 228, !dbg !21
  store half %1088, ptr addrspace(1) %address.local387, align 2, !dbg !21
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local364, !dbg !21
  %address.local389 = or i64 %address.local366, 115, !dbg !21
  %address.local390 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local388, i64 %address.local389, !dbg !21
  store half %1087, ptr addrspace(1) %address.local390, align 2, !dbg !21
  br label %1101, !dbg !21

1101:                                             ; preds = %1081, %1080
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/group8-reverse")
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
