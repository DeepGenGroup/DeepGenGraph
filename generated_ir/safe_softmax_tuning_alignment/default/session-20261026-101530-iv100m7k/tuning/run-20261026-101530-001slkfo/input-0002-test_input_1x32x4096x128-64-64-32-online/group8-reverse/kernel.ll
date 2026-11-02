; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/group8-reverse/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/group8-reverse/input.mlir"

@shm_0 = addrspace(3) global [8192 x i8] undef, align 16

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
  %40 = sub nsw i64 64, %14, !dbg !7
  %41 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %42 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %43 = getelementptr i8, ptr addrspace(1) %42, i64 64, !dbg !11
  %44 = shl i64 %15, 7, !dbg !15
  %45 = and i64 %44, 3968, !dbg !15
  %46 = shl nsw i64 %18, 12, !dbg !15
  %47 = or disjoint i64 %45, %46, !dbg !15
  %48 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %49 = getelementptr i8, ptr addrspace(1) %48, i64 128, !dbg !11
  %50 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %51 = getelementptr i8, ptr addrspace(1) %50, i64 192, !dbg !11
  br label %52, !dbg !7

52:                                               ; preds = %413, %4
  %53 = phi i64 [ %725, %413 ], [ 0, %4 ], !dbg !7
  %54 = phi <4 x float> [ %608, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %55 = phi <4 x float> [ %616, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %56 = phi <8 x float> [ %721, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %57 = phi <1 x float> [ %572, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %58 = phi <1 x float> [ %475, %413 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %59 = phi <4 x float> [ %648, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi <4 x float> [ %656, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = phi <8 x float> [ %722, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %62 = phi <4 x float> [ %688, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi <4 x float> [ %696, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi <8 x float> [ %723, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = phi <4 x float> [ %712, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %66 = phi <4 x float> [ %720, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = phi <8 x float> [ %724, %413 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = icmp slt i64 %53, %40, !dbg !7
  br i1 %68, label %69, label %726, !dbg !7

69:                                               ; preds = %52
  %70 = shl nsw i64 %53, 6, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local38 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local39 = and i64 %address.local38, 1536, !dbg !11
  %address.local40 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local39, !dbg !11
  %address.local41 = ptrtoint ptr addrspace(3) %address.local40 to i64, !dbg !11
  %address.local42 = trunc i64 %address.local41 to i32, !dbg !11
  %71 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local42), !dbg !11
  %72 = zext i32 %71 to i64, !dbg !11
  %73 = inttoptr i64 %72 to ptr addrspace(3), !dbg !11
  %address.local43 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local44 = and i64 %address.local43, 8064, !dbg !11
  %address.local45 = lshr i64 %address.local, 3, !dbg !11
  %address.local46 = and i64 %address.local45, 24, !dbg !11
  %address.local47 = or disjoint i64 %address.local44, %address.local46, !dbg !11
  %address.local48 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local47, !dbg !11
  %74 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %75 = ptrtoint ptr addrspace(1) %address.local48 to i64, !dbg !11
  %76 = sub i64 %75, %74, !dbg !11
  %dma.buffer.offset412 = trunc i64 %76 to i32, !dbg !11
  %77 = trunc i64 %74 to i32, !dbg !11
  %78 = insertelement <4 x i32> poison, i32 %77, i64 0, !dbg !11
  %79 = lshr i64 %74, 32, !dbg !11
  %80 = trunc i64 %79 to i32, !dbg !11
  %81 = insertelement <4 x i32> %78, i32 %80, i64 1, !dbg !11
  %82 = insertelement <4 x i32> %81, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc413 = insertelement <4 x i32> %82, i32 131072, i64 3, !dbg !11
  %83 = ptrtoint ptr addrspace(3) %73 to i32, !dbg !11
  %dma.buffer.lds414 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !11
  %84 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds414, i32 %dma.buffer.offset412, <4 x i32> %dma.buffer.rsrc413) #10, !dbg !11
  %85 = lshr i64 %53, 13, !dbg !15
  %86 = add nuw nsw i64 %85, %15, !dbg !15
  %87 = lshr i64 %53, 6, !dbg !15
  %88 = and i64 %87, 127, !dbg !15
  %89 = shl i64 %86, 19, !dbg !15
  %90 = and i64 %89, 16252928, !dbg !15
  %91 = shl i64 %87, 19, !dbg !15
  %92 = shl i64 %53, 13, !dbg !15
  %93 = sub i64 %92, %91, !dbg !15
  %94 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %90, !dbg !15
  %95 = getelementptr [2 x i8], ptr addrspace(1) %94, i64 %88, !dbg !15
  %96 = getelementptr [2 x i8], ptr addrspace(1) %95, i64 %93, !dbg !15
  %address.local49 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local39, !dbg !15
  %address.local50 = ptrtoint ptr addrspace(3) %address.local49 to i64, !dbg !15
  %address.local51 = trunc i64 %address.local50 to i32, !dbg !15
  %97 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local51), !dbg !15
  %98 = zext i32 %97 to i64, !dbg !15
  %99 = inttoptr i64 %98 to ptr addrspace(3), !dbg !15
  %address.local52 = getelementptr [2 x i8], ptr addrspace(1) %96, i64 %address.local47, !dbg !15
  %100 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %101 = ptrtoint ptr addrspace(1) %address.local52 to i64, !dbg !15
  %102 = sub i64 %101, %100, !dbg !15
  %dma.buffer.offset409 = trunc i64 %102 to i32, !dbg !15
  %103 = trunc i64 %100 to i32, !dbg !15
  %104 = insertelement <4 x i32> poison, i32 %103, i64 0, !dbg !15
  %105 = lshr i64 %100, 32, !dbg !15
  %106 = trunc i64 %105 to i32, !dbg !15
  %107 = insertelement <4 x i32> %104, i32 %106, i64 1, !dbg !15
  %108 = insertelement <4 x i32> %107, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc410 = insertelement <4 x i32> %108, i32 131072, i64 3, !dbg !15
  %109 = ptrtoint ptr addrspace(3) %99 to i32, !dbg !15
  %dma.buffer.lds411 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %109), !dbg !15
  %110 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds411, i32 %dma.buffer.offset409, <4 x i32> %dma.buffer.rsrc410) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local53 = lshr i32 %address.thread, 2, !dbg !12
  %address.local54 = and i32 %address.local53, 4, !dbg !12
  %address.local55 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local56 = and i32 %address.local55, 384, !dbg !12
  %address.local57 = or disjoint i32 %address.local54, %address.local56, !dbg !12
  %address.local58 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local59 = and i32 %address.local58, 120, !dbg !12
  %address.local60 = or disjoint i32 %address.local57, %address.local59, !dbg !12
  %address.local61 = shl nuw nsw i32 %address.thread, 4, !dbg !12
  %address.local62 = and i32 %address.local61, 512, !dbg !12
  %address.local63 = or disjoint i32 %address.local60, %address.local62, !dbg !12
  %address.local64 = zext nneg i32 %address.local63 to i64, !dbg !12
  %address.local65 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local64, !dbg !12
  %111 = load <4 x half>, ptr addrspace(3) %address.local65, align 2, !dbg !12
  %address.local66 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local64, !dbg !12
  %address.local67 = getelementptr i8, ptr addrspace(3) %address.local66, i64 2048, !dbg !12
  %112 = load <4 x half>, ptr addrspace(3) %address.local67, align 2, !dbg !12
  %113 = fmul <4 x half> %111, splat (half 0xH3015), !dbg !12
  %114 = fmul <4 x half> %112, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local68 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local69 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local68, !dbg !16
  %address.local70 = lshr i64 %address.local, 2, !dbg !16
  %address.local71 = mul nsw i64 %address.local70, -512, !dbg !16
  %address.local72 = getelementptr [2 x i8], ptr addrspace(3) %address.local69, i64 %address.local71, !dbg !16
  %address.local73 = shl nuw nsw i64 %address.local, 1, !dbg !16
  %address.local74 = and i64 %address.local73, 24, !dbg !16
  %address.local75 = getelementptr [2 x i8], ptr addrspace(3) %address.local72, i64 %address.local74, !dbg !16
  %address.local76 = lshr i64 %address.local, 2, !dbg !12
  %address.local77 = and i64 %address.local76, 12, !dbg !12
  %address.local78 = getelementptr [2 x i8], ptr addrspace(3) %address.local75, i64 %address.local77, !dbg !16
  %address.local79 = and i64 %address.local, 63, !dbg !12
  %address.local80 = lshr i64 %address.local79, 5, !dbg !12
  %address.local81 = mul nuw nsw i64 %address.local80, 504, !dbg !12
  %address.local82 = getelementptr [2 x i8], ptr addrspace(3) %address.local78, i64 %address.local81, !dbg !16
  %115 = load <4 x half>, ptr addrspace(3) %address.local82, align 2, !dbg !16
  %address.local83 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local68, !dbg !16
  %address.local84 = getelementptr [2 x i8], ptr addrspace(3) %address.local83, i64 %address.local77, !dbg !16
  %address.local85 = getelementptr [2 x i8], ptr addrspace(3) %address.local84, i64 %address.local81, !dbg !16
  %address.local86 = getelementptr [2 x i8], ptr addrspace(3) %address.local85, i64 %address.local71, !dbg !16
  %address.local87 = getelementptr [2 x i8], ptr addrspace(3) %address.local86, i64 %address.local74, !dbg !16
  %address.local88 = getelementptr i8, ptr addrspace(3) %address.local87, i64 2048, !dbg !16
  %116 = load <4 x half>, ptr addrspace(3) %address.local88, align 2, !dbg !16
  %117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %115, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local89 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local68, !dbg !16
  %address.local90 = getelementptr [2 x i8], ptr addrspace(3) %address.local89, i64 %address.local71, !dbg !16
  %address.local91 = getelementptr [2 x i8], ptr addrspace(3) %address.local90, i64 %address.local74, !dbg !16
  %address.local92 = getelementptr [2 x i8], ptr addrspace(3) %address.local91, i64 %address.local77, !dbg !16
  %address.local93 = getelementptr [2 x i8], ptr addrspace(3) %address.local92, i64 %address.local81, !dbg !16
  %address.local94 = getelementptr i8, ptr addrspace(3) %address.local93, i64 64, !dbg !16
  %118 = load <4 x half>, ptr addrspace(3) %address.local94, align 2, !dbg !16
  %119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %116, <4 x float> %117) #9, !dbg !16
  %address.local95 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local68, !dbg !16
  %address.local96 = getelementptr [2 x i8], ptr addrspace(3) %address.local95, i64 %address.local77, !dbg !16
  %address.local97 = getelementptr [2 x i8], ptr addrspace(3) %address.local96, i64 %address.local81, !dbg !16
  %address.local98 = getelementptr [2 x i8], ptr addrspace(3) %address.local97, i64 %address.local71, !dbg !16
  %address.local99 = getelementptr [2 x i8], ptr addrspace(3) %address.local98, i64 %address.local74, !dbg !16
  %address.local100 = getelementptr i8, ptr addrspace(3) %address.local99, i64 2112, !dbg !16
  %120 = load <4 x half>, ptr addrspace(3) %address.local100, align 2, !dbg !16
  %address.local101 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local68, !dbg !16
  %address.local102 = getelementptr [2 x i8], ptr addrspace(3) %address.local101, i64 %address.local71, !dbg !16
  %address.local103 = getelementptr [2 x i8], ptr addrspace(3) %address.local102, i64 %address.local74, !dbg !16
  %address.local104 = getelementptr [2 x i8], ptr addrspace(3) %address.local103, i64 %address.local77, !dbg !16
  %address.local105 = getelementptr [2 x i8], ptr addrspace(3) %address.local104, i64 %address.local81, !dbg !16
  %address.local106 = getelementptr i8, ptr addrspace(3) %address.local105, i64 128, !dbg !16
  %121 = load <4 x half>, ptr addrspace(3) %address.local106, align 2, !dbg !16
  %122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %118, <4 x float> zeroinitializer) #9, !dbg !16
  %123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %120, <4 x float> %122) #9, !dbg !16
  %address.local107 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local68, !dbg !16
  %address.local108 = getelementptr [2 x i8], ptr addrspace(3) %address.local107, i64 %address.local77, !dbg !16
  %address.local109 = getelementptr [2 x i8], ptr addrspace(3) %address.local108, i64 %address.local81, !dbg !16
  %address.local110 = getelementptr [2 x i8], ptr addrspace(3) %address.local109, i64 %address.local71, !dbg !16
  %address.local111 = getelementptr [2 x i8], ptr addrspace(3) %address.local110, i64 %address.local74, !dbg !16
  %address.local112 = getelementptr i8, ptr addrspace(3) %address.local111, i64 2176, !dbg !16
  %124 = load <4 x half>, ptr addrspace(3) %address.local112, align 2, !dbg !16
  %125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %121, <4 x float> zeroinitializer) #9, !dbg !16
  %126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %124, <4 x float> %125) #9, !dbg !16
  %address.local113 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local68, !dbg !16
  %address.local114 = getelementptr [2 x i8], ptr addrspace(3) %address.local113, i64 %address.local71, !dbg !16
  %address.local115 = getelementptr [2 x i8], ptr addrspace(3) %address.local114, i64 %address.local74, !dbg !16
  %address.local116 = getelementptr [2 x i8], ptr addrspace(3) %address.local115, i64 %address.local77, !dbg !16
  %address.local117 = getelementptr [2 x i8], ptr addrspace(3) %address.local116, i64 %address.local81, !dbg !16
  %address.local118 = getelementptr i8, ptr addrspace(3) %address.local117, i64 192, !dbg !16
  %127 = load <4 x half>, ptr addrspace(3) %address.local118, align 2, !dbg !16
  %address.local119 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local68, !dbg !16
  %address.local120 = getelementptr [2 x i8], ptr addrspace(3) %address.local119, i64 %address.local77, !dbg !16
  %address.local121 = getelementptr [2 x i8], ptr addrspace(3) %address.local120, i64 %address.local81, !dbg !16
  %address.local122 = getelementptr [2 x i8], ptr addrspace(3) %address.local121, i64 %address.local71, !dbg !16
  %address.local123 = getelementptr [2 x i8], ptr addrspace(3) %address.local122, i64 %address.local74, !dbg !16
  %address.local124 = getelementptr i8, ptr addrspace(3) %address.local123, i64 2240, !dbg !16
  %128 = load <4 x half>, ptr addrspace(3) %address.local124, align 2, !dbg !16
  %129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %127, <4 x float> zeroinitializer) #9, !dbg !16
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %128, <4 x float> %129) #9, !dbg !16
  %131 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local42), !dbg !11
  %132 = zext i32 %131 to i64, !dbg !11
  %133 = inttoptr i64 %132 to ptr addrspace(3), !dbg !11
  %address.local125 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local47, !dbg !11
  %134 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %135 = ptrtoint ptr addrspace(1) %address.local125 to i64, !dbg !11
  %136 = sub i64 %135, %134, !dbg !11
  %dma.buffer.offset406 = trunc i64 %136 to i32, !dbg !11
  %137 = trunc i64 %134 to i32, !dbg !11
  %138 = insertelement <4 x i32> poison, i32 %137, i64 0, !dbg !11
  %139 = lshr i64 %134, 32, !dbg !11
  %140 = trunc i64 %139 to i32, !dbg !11
  %141 = insertelement <4 x i32> %138, i32 %140, i64 1, !dbg !11
  %142 = insertelement <4 x i32> %141, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc407 = insertelement <4 x i32> %142, i32 131072, i64 3, !dbg !11
  %143 = ptrtoint ptr addrspace(3) %133 to i32, !dbg !11
  %dma.buffer.lds408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %143), !dbg !11
  %144 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds408, i32 %dma.buffer.offset406, <4 x i32> %dma.buffer.rsrc407) #10, !dbg !11
  %145 = add nuw i64 %53, 2048, !dbg !15
  %146 = icmp slt i64 %145, 0, !dbg !15
  %147 = sub nuw i64 -2049, %53, !dbg !15
  %148 = select i1 %146, i64 %147, i64 %145, !dbg !15
  %149 = sdiv i64 %148, 8192, !dbg !15
  %.lobit = ashr i64 %145, 63, !dbg !15
  %150 = xor i64 %149, %.lobit, !dbg !15
  %151 = add nsw i64 %150, %15, !dbg !15
  %152 = add nsw i64 %87, %47, !dbg !15
  %153 = add nsw i64 %152, 32, !dbg !15
  %154 = icmp slt i64 %152, -32, !dbg !15
  %155 = sub nsw i64 -33, %152, !dbg !15
  %156 = select i1 %154, i64 %155, i64 %153, !dbg !15
  %157 = sdiv i64 %156, 128, !dbg !15
  %158 = sext i1 %154 to i64, !dbg !15
  %159 = xor i64 %157, %158, !dbg !15
  %160 = shl nsw i64 %159, 7, !dbg !15
  %161 = sub i64 %47, %160, !dbg !15
  %162 = shl i64 %151, 19, !dbg !15
  %163 = and i64 %162, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %164 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %163, !dbg !15
  %165 = getelementptr [2 x i8], ptr addrspace(1) %164, i64 %87, !dbg !15
  %166 = getelementptr [2 x i8], ptr addrspace(1) %165, i64 %161, !dbg !15
  %167 = getelementptr i8, ptr addrspace(1) %166, i64 64, !dbg !15
  %168 = getelementptr [2 x i8], ptr addrspace(1) %167, i64 %93, !dbg !15
  %169 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local51), !dbg !15
  %170 = zext i32 %169 to i64, !dbg !15
  %171 = inttoptr i64 %170 to ptr addrspace(3), !dbg !15
  %address.local126 = getelementptr [2 x i8], ptr addrspace(1) %168, i64 %address.local47, !dbg !15
  %172 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %173 = ptrtoint ptr addrspace(1) %address.local126 to i64, !dbg !15
  %174 = sub i64 %173, %172, !dbg !15
  %dma.buffer.offset403 = trunc i64 %174 to i32, !dbg !15
  %175 = trunc i64 %172 to i32, !dbg !15
  %176 = insertelement <4 x i32> poison, i32 %175, i64 0, !dbg !15
  %177 = lshr i64 %172, 32, !dbg !15
  %178 = trunc i64 %177 to i32, !dbg !15
  %179 = insertelement <4 x i32> %176, i32 %178, i64 1, !dbg !15
  %180 = insertelement <4 x i32> %179, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc404 = insertelement <4 x i32> %180, i32 131072, i64 3, !dbg !15
  %181 = ptrtoint ptr addrspace(3) %171 to i32, !dbg !15
  %dma.buffer.lds405 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %181), !dbg !15
  %182 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds405, i32 %dma.buffer.offset403, <4 x i32> %dma.buffer.rsrc404) #10, !dbg !15
  %183 = load <4 x half>, ptr addrspace(3) %address.local65, align 2, !dbg !12
  %184 = load <4 x half>, ptr addrspace(3) %address.local67, align 2, !dbg !12
  %185 = fmul <4 x half> %183, splat (half 0xH3015), !dbg !12
  %186 = fmul <4 x half> %184, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %187 = load <4 x half>, ptr addrspace(3) %address.local82, align 2, !dbg !16
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %187, <4 x float> %119) #9, !dbg !16
  %189 = load <4 x half>, ptr addrspace(3) %address.local88, align 2, !dbg !16
  %190 = load <4 x half>, ptr addrspace(3) %address.local94, align 2, !dbg !16
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %189, <4 x float> %188) #9, !dbg !16
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %190, <4 x float> %123) #9, !dbg !16
  %193 = load <4 x half>, ptr addrspace(3) %address.local100, align 2, !dbg !16
  %194 = load <4 x half>, ptr addrspace(3) %address.local106, align 2, !dbg !16
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %193, <4 x float> %192) #9, !dbg !16
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %194, <4 x float> %126) #9, !dbg !16
  %197 = load <4 x half>, ptr addrspace(3) %address.local112, align 2, !dbg !16
  %198 = load <4 x half>, ptr addrspace(3) %address.local118, align 2, !dbg !16
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %197, <4 x float> %196) #9, !dbg !16
  %200 = load <4 x half>, ptr addrspace(3) %address.local124, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %198, <4 x float> %130) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %200, <4 x float> %201) #9, !dbg !16
  %203 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local42), !dbg !11
  %204 = zext i32 %203 to i64, !dbg !11
  %205 = inttoptr i64 %204 to ptr addrspace(3), !dbg !11
  %address.local127 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local47, !dbg !11
  %206 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %207 = ptrtoint ptr addrspace(1) %address.local127 to i64, !dbg !11
  %208 = sub i64 %207, %206, !dbg !11
  %dma.buffer.offset400 = trunc i64 %208 to i32, !dbg !11
  %209 = trunc i64 %206 to i32, !dbg !11
  %210 = insertelement <4 x i32> poison, i32 %209, i64 0, !dbg !11
  %211 = lshr i64 %206, 32, !dbg !11
  %212 = trunc i64 %211 to i32, !dbg !11
  %213 = insertelement <4 x i32> %210, i32 %212, i64 1, !dbg !11
  %214 = insertelement <4 x i32> %213, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc401 = insertelement <4 x i32> %214, i32 131072, i64 3, !dbg !11
  %215 = ptrtoint ptr addrspace(3) %205 to i32, !dbg !11
  %dma.buffer.lds402 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %215), !dbg !11
  %216 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds402, i32 %dma.buffer.offset400, <4 x i32> %dma.buffer.rsrc401) #10, !dbg !11
  %217 = add nuw i64 %53, 4096, !dbg !15
  %218 = icmp slt i64 %217, 0, !dbg !15
  %219 = sub nuw i64 -4097, %53, !dbg !15
  %220 = select i1 %218, i64 %219, i64 %217, !dbg !15
  %221 = sdiv i64 %220, 8192, !dbg !15
  %.lobit2 = ashr i64 %217, 63, !dbg !15
  %222 = xor i64 %221, %.lobit2, !dbg !15
  %223 = add nsw i64 %222, %15, !dbg !15
  %224 = add nsw i64 %152, 64, !dbg !15
  %225 = icmp slt i64 %152, -64, !dbg !15
  %226 = sub nsw i64 -65, %152, !dbg !15
  %227 = select i1 %225, i64 %226, i64 %224, !dbg !15
  %228 = sdiv i64 %227, 128, !dbg !15
  %229 = sext i1 %225 to i64, !dbg !15
  %230 = xor i64 %228, %229, !dbg !15
  %231 = shl nsw i64 %230, 7, !dbg !15
  %232 = sub i64 %47, %231, !dbg !15
  %233 = shl i64 %223, 19, !dbg !15
  %234 = and i64 %233, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %235 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %234, !dbg !15
  %236 = getelementptr [2 x i8], ptr addrspace(1) %235, i64 %87, !dbg !15
  %237 = getelementptr [2 x i8], ptr addrspace(1) %236, i64 %232, !dbg !15
  %238 = getelementptr i8, ptr addrspace(1) %237, i64 128, !dbg !15
  %239 = getelementptr [2 x i8], ptr addrspace(1) %238, i64 %93, !dbg !15
  %240 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local51), !dbg !15
  %241 = zext i32 %240 to i64, !dbg !15
  %242 = inttoptr i64 %241 to ptr addrspace(3), !dbg !15
  %address.local128 = getelementptr [2 x i8], ptr addrspace(1) %239, i64 %address.local47, !dbg !15
  %243 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %244 = ptrtoint ptr addrspace(1) %address.local128 to i64, !dbg !15
  %245 = sub i64 %244, %243, !dbg !15
  %dma.buffer.offset397 = trunc i64 %245 to i32, !dbg !15
  %246 = trunc i64 %243 to i32, !dbg !15
  %247 = insertelement <4 x i32> poison, i32 %246, i64 0, !dbg !15
  %248 = lshr i64 %243, 32, !dbg !15
  %249 = trunc i64 %248 to i32, !dbg !15
  %250 = insertelement <4 x i32> %247, i32 %249, i64 1, !dbg !15
  %251 = insertelement <4 x i32> %250, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc398 = insertelement <4 x i32> %251, i32 131072, i64 3, !dbg !15
  %252 = ptrtoint ptr addrspace(3) %242 to i32, !dbg !15
  %dma.buffer.lds399 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %252), !dbg !15
  %253 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds399, i32 %dma.buffer.offset397, <4 x i32> %dma.buffer.rsrc398) #10, !dbg !15
  %254 = load <4 x half>, ptr addrspace(3) %address.local65, align 2, !dbg !12
  %255 = load <4 x half>, ptr addrspace(3) %address.local67, align 2, !dbg !12
  %256 = fmul <4 x half> %254, splat (half 0xH3015), !dbg !12
  %257 = fmul <4 x half> %255, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %258 = load <4 x half>, ptr addrspace(3) %address.local82, align 2, !dbg !16
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %256, <4 x half> %258, <4 x float> %191) #9, !dbg !16
  %260 = load <4 x half>, ptr addrspace(3) %address.local88, align 2, !dbg !16
  %261 = load <4 x half>, ptr addrspace(3) %address.local94, align 2, !dbg !16
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %257, <4 x half> %260, <4 x float> %259) #9, !dbg !16
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %256, <4 x half> %261, <4 x float> %195) #9, !dbg !16
  %264 = load <4 x half>, ptr addrspace(3) %address.local100, align 2, !dbg !16
  %265 = load <4 x half>, ptr addrspace(3) %address.local106, align 2, !dbg !16
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %257, <4 x half> %264, <4 x float> %263) #9, !dbg !16
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %256, <4 x half> %265, <4 x float> %199) #9, !dbg !16
  %268 = load <4 x half>, ptr addrspace(3) %address.local112, align 2, !dbg !16
  %269 = load <4 x half>, ptr addrspace(3) %address.local118, align 2, !dbg !16
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %257, <4 x half> %268, <4 x float> %267) #9, !dbg !16
  %271 = load <4 x half>, ptr addrspace(3) %address.local124, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %256, <4 x half> %269, <4 x float> %202) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %257, <4 x half> %271, <4 x float> %272) #9, !dbg !16
  %274 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local42), !dbg !11
  %275 = zext i32 %274 to i64, !dbg !11
  %276 = inttoptr i64 %275 to ptr addrspace(3), !dbg !11
  %address.local129 = getelementptr [2 x i8], ptr addrspace(1) %51, i64 %address.local47, !dbg !11
  %277 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %278 = ptrtoint ptr addrspace(1) %address.local129 to i64, !dbg !11
  %279 = sub i64 %278, %277, !dbg !11
  %dma.buffer.offset394 = trunc i64 %279 to i32, !dbg !11
  %280 = trunc i64 %277 to i32, !dbg !11
  %281 = insertelement <4 x i32> poison, i32 %280, i64 0, !dbg !11
  %282 = lshr i64 %277, 32, !dbg !11
  %283 = trunc i64 %282 to i32, !dbg !11
  %284 = insertelement <4 x i32> %281, i32 %283, i64 1, !dbg !11
  %285 = insertelement <4 x i32> %284, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc395 = insertelement <4 x i32> %285, i32 131072, i64 3, !dbg !11
  %286 = ptrtoint ptr addrspace(3) %276 to i32, !dbg !11
  %dma.buffer.lds396 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %286), !dbg !11
  %287 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds396, i32 %dma.buffer.offset394, <4 x i32> %dma.buffer.rsrc395) #10, !dbg !11
  %288 = add nuw i64 %53, 6144, !dbg !15
  %289 = icmp slt i64 %288, 0, !dbg !15
  %290 = sub nuw i64 -6145, %53, !dbg !15
  %291 = select i1 %289, i64 %290, i64 %288, !dbg !15
  %292 = sdiv i64 %291, 8192, !dbg !15
  %.lobit3 = ashr i64 %288, 63, !dbg !15
  %293 = xor i64 %292, %.lobit3, !dbg !15
  %294 = add nsw i64 %293, %15, !dbg !15
  %295 = add nsw i64 %152, 96, !dbg !15
  %296 = icmp slt i64 %152, -96, !dbg !15
  %297 = sub nsw i64 -97, %152, !dbg !15
  %298 = select i1 %296, i64 %297, i64 %295, !dbg !15
  %299 = sdiv i64 %298, 128, !dbg !15
  %300 = sext i1 %296 to i64, !dbg !15
  %301 = xor i64 %299, %300, !dbg !15
  %302 = shl nsw i64 %301, 7, !dbg !15
  %303 = sub i64 %47, %302, !dbg !15
  %304 = shl i64 %294, 19, !dbg !15
  %305 = and i64 %304, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %306 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %305, !dbg !15
  %307 = getelementptr [2 x i8], ptr addrspace(1) %306, i64 %87, !dbg !15
  %308 = getelementptr [2 x i8], ptr addrspace(1) %307, i64 %303, !dbg !15
  %309 = getelementptr i8, ptr addrspace(1) %308, i64 192, !dbg !15
  %310 = getelementptr [2 x i8], ptr addrspace(1) %309, i64 %93, !dbg !15
  %311 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local51), !dbg !15
  %312 = zext i32 %311 to i64, !dbg !15
  %313 = inttoptr i64 %312 to ptr addrspace(3), !dbg !15
  %address.local130 = getelementptr [2 x i8], ptr addrspace(1) %310, i64 %address.local47, !dbg !15
  %314 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %315 = ptrtoint ptr addrspace(1) %address.local130 to i64, !dbg !15
  %316 = sub i64 %315, %314, !dbg !15
  %dma.buffer.offset391 = trunc i64 %316 to i32, !dbg !15
  %317 = trunc i64 %314 to i32, !dbg !15
  %318 = insertelement <4 x i32> poison, i32 %317, i64 0, !dbg !15
  %319 = lshr i64 %314, 32, !dbg !15
  %320 = trunc i64 %319 to i32, !dbg !15
  %321 = insertelement <4 x i32> %318, i32 %320, i64 1, !dbg !15
  %322 = insertelement <4 x i32> %321, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc392 = insertelement <4 x i32> %322, i32 131072, i64 3, !dbg !15
  %323 = ptrtoint ptr addrspace(3) %313 to i32, !dbg !15
  %dma.buffer.lds393 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %323), !dbg !15
  %324 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds393, i32 %dma.buffer.offset391, <4 x i32> %dma.buffer.rsrc392) #10, !dbg !15
  %325 = load <4 x half>, ptr addrspace(3) %address.local65, align 2, !dbg !12
  %326 = load <4 x half>, ptr addrspace(3) %address.local67, align 2, !dbg !12
  %327 = fmul <4 x half> %325, splat (half 0xH3015), !dbg !12
  %328 = fmul <4 x half> %326, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %329 = add nuw nsw i64 %87, %15, !dbg !17
  %330 = shl i64 %329, 19, !dbg !17
  %331 = and i64 %330, 16252928, !dbg !17
  %332 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %331, !dbg !17
  %333 = getelementptr [2 x i8], ptr addrspace(1) %332, i64 %93, !dbg !17
  %334 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local42), !dbg !17
  %335 = zext i32 %334 to i64, !dbg !17
  %336 = inttoptr i64 %335 to ptr addrspace(3), !dbg !17
  %address.local131 = shl nuw nsw i64 %address.local, 5, !dbg !17
  %address.local132 = and i64 %address.local131, 8064, !dbg !17
  %address.local133 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local134 = and i64 %address.local133, 24, !dbg !17
  %address.local135 = or disjoint i64 %address.local132, %address.local134, !dbg !17
  %address.local136 = getelementptr [2 x i8], ptr addrspace(1) %333, i64 %address.local135, !dbg !17
  %337 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %338 = ptrtoint ptr addrspace(1) %address.local136 to i64, !dbg !17
  %339 = sub i64 %338, %337, !dbg !17
  %dma.buffer.offset388 = trunc i64 %339 to i32, !dbg !17
  %340 = trunc i64 %337 to i32, !dbg !17
  %341 = insertelement <4 x i32> poison, i32 %340, i64 0, !dbg !17
  %342 = lshr i64 %337, 32, !dbg !17
  %343 = trunc i64 %342 to i32, !dbg !17
  %344 = insertelement <4 x i32> %341, i32 %343, i64 1, !dbg !17
  %345 = insertelement <4 x i32> %344, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc389 = insertelement <4 x i32> %345, i32 131072, i64 3, !dbg !17
  %346 = ptrtoint ptr addrspace(3) %336 to i32, !dbg !17
  %dma.buffer.lds390 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %346), !dbg !17
  %347 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds390, i32 %dma.buffer.offset388, <4 x i32> %dma.buffer.rsrc389) #10, !dbg !17
  %348 = load <4 x half>, ptr addrspace(3) %address.local82, align 2, !dbg !16
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %348, <4 x float> %262) #9, !dbg !16
  %350 = load <4 x half>, ptr addrspace(3) %address.local88, align 2, !dbg !16
  %351 = load <4 x half>, ptr addrspace(3) %address.local94, align 2, !dbg !16
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %350, <4 x float> %349) #9, !dbg !16
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %351, <4 x float> %266) #9, !dbg !16
  %354 = load <4 x half>, ptr addrspace(3) %address.local100, align 2, !dbg !16
  %355 = load <4 x half>, ptr addrspace(3) %address.local106, align 2, !dbg !16
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %354, <4 x float> %353) #9, !dbg !16
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %355, <4 x float> %270) #9, !dbg !16
  %358 = load <4 x half>, ptr addrspace(3) %address.local112, align 2, !dbg !16
  %359 = load <4 x half>, ptr addrspace(3) %address.local118, align 2, !dbg !16
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %358, <4 x float> %357) #9, !dbg !16
  %361 = load <4 x half>, ptr addrspace(3) %address.local124, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %359, <4 x float> %273) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %363 = or disjoint i64 %70, 63, !dbg !13
  %364 = icmp uge i64 %19, %363, !dbg !13
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %361, <4 x float> %362) #9, !dbg !16
  %366 = and i1 %39, %364, !dbg !13
  br i1 %366, label %367, label %368, !dbg !13

367:                                              ; preds = %69
  br label %411, !dbg !13

368:                                              ; preds = %69
  %369 = add i64 %19, %35, !dbg !13
  %370 = shl nuw nsw i64 %36, 4, !dbg !13
  %371 = add i64 %369, %370, !dbg !13
  %372 = and i32 %34, 240, !dbg !13
  %index.compact = zext nneg i32 %372 to i64, !dbg !13
  %373 = sub i64 %371, %index.compact, !dbg !13
  %374 = add i64 %373, 1, !dbg !22
  %375 = and i64 %35, 48, !dbg !13
  %376 = or disjoint i64 %70, %375, !dbg !13
  %377 = lshr i64 %374, 32, !dbg !23
  %compare.high = trunc i64 %377 to i32, !dbg !23
  %378 = lshr i64 %376, 32, !dbg !23
  %compare.high322 = trunc i64 %378 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high322, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high322, !dbg !23
  %compare.low.base = trunc i64 %374 to i32, !dbg !23
  %compare.low.base323 = trunc i64 %376 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base323, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %379 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low324 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result325 = select i1 %compare.high.equal, i1 %compare.low324, i1 %compare.high.order, !dbg !23
  %380 = select i1 %compare.result325, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base326 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local327 = or i32 %compare.local.base326, 2, !dbg !23
  %compare.low328 = icmp ugt i32 %compare.low.base, %compare.local327, !dbg !23
  %compare.result329 = select i1 %compare.high.equal, i1 %compare.low328, i1 %compare.high.order, !dbg !23
  %381 = select i1 %compare.result329, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base330 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local331 = or i32 %compare.local.base330, 3, !dbg !23
  %compare.low332 = icmp ugt i32 %compare.low.base, %compare.local331, !dbg !23
  %compare.result333 = select i1 %compare.high.equal, i1 %compare.low332, i1 %compare.high.order, !dbg !23
  %382 = select i1 %compare.result333, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base334 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local335 = or i32 %compare.local.base334, 4, !dbg !23
  %compare.low336 = icmp ugt i32 %compare.low.base, %compare.local335, !dbg !23
  %compare.result337 = select i1 %compare.high.equal, i1 %compare.low336, i1 %compare.high.order, !dbg !23
  %383 = select i1 %compare.result337, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base338 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local339 = or i32 %compare.local.base338, 5, !dbg !23
  %compare.low340 = icmp ugt i32 %compare.low.base, %compare.local339, !dbg !23
  %compare.result341 = select i1 %compare.high.equal, i1 %compare.low340, i1 %compare.high.order, !dbg !23
  %384 = select i1 %compare.result341, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base342 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local343 = or i32 %compare.local.base342, 6, !dbg !23
  %compare.low344 = icmp ugt i32 %compare.low.base, %compare.local343, !dbg !23
  %compare.result345 = select i1 %compare.high.equal, i1 %compare.low344, i1 %compare.high.order, !dbg !23
  %385 = select i1 %compare.result345, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base346 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local347 = or i32 %compare.local.base346, 7, !dbg !23
  %compare.low348 = icmp ugt i32 %compare.low.base, %compare.local347, !dbg !23
  %compare.result349 = select i1 %compare.high.equal, i1 %compare.low348, i1 %compare.high.order, !dbg !23
  %386 = select i1 %compare.result349, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base350 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local351 = or i32 %compare.local.base350, 8, !dbg !23
  %compare.low352 = icmp ugt i32 %compare.low.base, %compare.local351, !dbg !23
  %compare.result353 = select i1 %compare.high.equal, i1 %compare.low352, i1 %compare.high.order, !dbg !23
  %387 = select i1 %compare.result353, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base354 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local355 = or i32 %compare.local.base354, 9, !dbg !23
  %compare.low356 = icmp ugt i32 %compare.low.base, %compare.local355, !dbg !23
  %compare.result357 = select i1 %compare.high.equal, i1 %compare.low356, i1 %compare.high.order, !dbg !23
  %388 = select i1 %compare.result357, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base358 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local359 = or i32 %compare.local.base358, 10, !dbg !23
  %compare.low360 = icmp ugt i32 %compare.low.base, %compare.local359, !dbg !23
  %compare.result361 = select i1 %compare.high.equal, i1 %compare.low360, i1 %compare.high.order, !dbg !23
  %389 = select i1 %compare.result361, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base362 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local363 = or i32 %compare.local.base362, 11, !dbg !23
  %compare.low364 = icmp ugt i32 %compare.low.base, %compare.local363, !dbg !23
  %compare.result365 = select i1 %compare.high.equal, i1 %compare.low364, i1 %compare.high.order, !dbg !23
  %390 = select i1 %compare.result365, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base366 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local367 = or i32 %compare.local.base366, 12, !dbg !23
  %compare.low368 = icmp ugt i32 %compare.low.base, %compare.local367, !dbg !23
  %compare.result369 = select i1 %compare.high.equal, i1 %compare.low368, i1 %compare.high.order, !dbg !23
  %391 = select i1 %compare.result369, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base370 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local371 = or i32 %compare.local.base370, 13, !dbg !23
  %compare.low372 = icmp ugt i32 %compare.low.base, %compare.local371, !dbg !23
  %compare.result373 = select i1 %compare.high.equal, i1 %compare.low372, i1 %compare.high.order, !dbg !23
  %392 = select i1 %compare.result373, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base374 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local375 = or i32 %compare.local.base374, 14, !dbg !23
  %compare.low376 = icmp ugt i32 %compare.low.base, %compare.local375, !dbg !23
  %compare.result377 = select i1 %compare.high.equal, i1 %compare.low376, i1 %compare.high.order, !dbg !23
  %393 = select i1 %compare.result377, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base378 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base323, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local379 = or i32 %compare.local.base378, 15, !dbg !23
  %compare.low380 = icmp ugt i32 %compare.low.base, %compare.local379, !dbg !23
  %compare.result381 = select i1 %compare.high.equal, i1 %compare.low380, i1 %compare.high.order, !dbg !23
  %394 = select i1 %compare.result381, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %395 = insertelement <16 x float> poison, float %379, i64 0, !dbg !13
  %396 = insertelement <16 x float> %395, float %380, i64 1, !dbg !13
  %397 = insertelement <16 x float> %396, float %381, i64 2, !dbg !13
  %398 = insertelement <16 x float> %397, float %382, i64 3, !dbg !13
  %399 = insertelement <16 x float> %398, float %383, i64 4, !dbg !13
  %400 = insertelement <16 x float> %399, float %384, i64 5, !dbg !13
  %401 = insertelement <16 x float> %400, float %385, i64 6, !dbg !13
  %402 = insertelement <16 x float> %401, float %386, i64 7, !dbg !13
  %403 = insertelement <16 x float> %402, float %387, i64 8, !dbg !13
  %404 = insertelement <16 x float> %403, float %388, i64 9, !dbg !13
  %405 = insertelement <16 x float> %404, float %389, i64 10, !dbg !13
  %406 = insertelement <16 x float> %405, float %390, i64 11, !dbg !13
  %407 = insertelement <16 x float> %406, float %391, i64 12, !dbg !13
  %408 = insertelement <16 x float> %407, float %392, i64 13, !dbg !13
  %409 = insertelement <16 x float> %408, float %393, i64 14, !dbg !13
  %410 = insertelement <16 x float> %409, float %394, i64 15, !dbg !13
  br label %411, !dbg !13

411:                                              ; preds = %367, %368
  %412 = phi <16 x float> [ %410, %368 ], [ zeroinitializer, %367 ], !dbg !13
  br label %413, !dbg !13

413:                                              ; preds = %411
  %414 = shufflevector <16 x float> %412, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %415 = fadd <4 x float> %352, %414, !dbg !25
  %416 = extractelement <4 x float> %415, i64 0, !dbg !25
  %417 = extractelement <4 x float> %415, i64 1, !dbg !25
  %418 = extractelement <4 x float> %415, i64 2, !dbg !25
  %419 = extractelement <4 x float> %415, i64 3, !dbg !25
  %420 = shufflevector <16 x float> %412, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %421 = fadd <4 x float> %356, %420, !dbg !25
  %422 = extractelement <4 x float> %421, i64 0, !dbg !25
  %423 = extractelement <4 x float> %421, i64 1, !dbg !25
  %424 = extractelement <4 x float> %421, i64 2, !dbg !25
  %425 = extractelement <4 x float> %421, i64 3, !dbg !25
  %426 = shufflevector <16 x float> %412, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %427 = fadd <4 x float> %360, %426, !dbg !25
  %428 = extractelement <4 x float> %427, i64 0, !dbg !25
  %429 = extractelement <4 x float> %427, i64 1, !dbg !25
  %430 = extractelement <4 x float> %427, i64 2, !dbg !25
  %431 = extractelement <4 x float> %427, i64 3, !dbg !25
  %432 = shufflevector <16 x float> %412, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %433 = fadd <4 x float> %365, %432, !dbg !25
  %434 = extractelement <4 x float> %433, i64 0, !dbg !25
  %435 = extractelement <4 x float> %433, i64 1, !dbg !25
  %436 = extractelement <4 x float> %433, i64 2, !dbg !25
  %437 = extractelement <4 x float> %433, i64 3, !dbg !25
  %438 = call float @llvm.maxnum.f32(float %416, float 0xFFF0000000000000), !dbg !26
  %439 = call float @llvm.maxnum.f32(float %438, float %417), !dbg !26
  %440 = call float @llvm.maxnum.f32(float %439, float %418), !dbg !26
  %441 = call float @llvm.maxnum.f32(float %440, float %419), !dbg !26
  %442 = call float @llvm.maxnum.f32(float %441, float %422), !dbg !26
  %443 = call float @llvm.maxnum.f32(float %442, float %423), !dbg !26
  %444 = call float @llvm.maxnum.f32(float %443, float %424), !dbg !26
  %445 = call float @llvm.maxnum.f32(float %444, float %425), !dbg !26
  %446 = call float @llvm.maxnum.f32(float %445, float %428), !dbg !26
  %447 = call float @llvm.maxnum.f32(float %446, float %429), !dbg !26
  %448 = call float @llvm.maxnum.f32(float %447, float %430), !dbg !26
  %449 = call float @llvm.maxnum.f32(float %448, float %431), !dbg !26
  %450 = call float @llvm.maxnum.f32(float %449, float %434), !dbg !26
  %451 = call float @llvm.maxnum.f32(float %450, float %435), !dbg !26
  %452 = call float @llvm.maxnum.f32(float %451, float %436), !dbg !26
  %453 = call float @llvm.maxnum.f32(float %452, float %437), !dbg !26
  %454 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %455 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %454), !dbg !26
  %456 = add nuw nsw i32 %455, 64, !dbg !26
  %457 = xor i32 %455, 16, !dbg !26
  %458 = and i32 %456, 192, !dbg !26
  %459 = icmp samesign ult i32 %457, %458, !dbg !26
  %460 = select i1 %459, i32 %457, i32 %455, !dbg !26
  %461 = shl nuw nsw i32 %460, 2, !dbg !26
  %462 = bitcast float %453 to i32, !dbg !26
  %463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %461, i32 %462), !dbg !26
  %464 = bitcast i32 %463 to float, !dbg !26
  %465 = call float @llvm.maxnum.f32(float %453, float %464), !dbg !26
  %466 = xor i32 %455, 32, !dbg !26
  %467 = icmp samesign ult i32 %466, %458, !dbg !26
  %468 = select i1 %467, i32 %466, i32 %455, !dbg !26
  %469 = shl nuw nsw i32 %468, 2, !dbg !26
  %470 = bitcast float %465 to i32, !dbg !26
  %471 = call i32 @llvm.amdgcn.ds.bpermute(i32 %469, i32 %470), !dbg !26
  %472 = bitcast i32 %471 to float, !dbg !26
  %473 = call float @llvm.maxnum.f32(float %465, float %472), !dbg !26
  %474 = insertelement <1 x float> poison, float %473, i64 0, !dbg !26
  %475 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %474, <1 x float> %58), !dbg !26
  %476 = shufflevector <1 x float> %475, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %477 = fsub <4 x float> %415, %476, !dbg !27
  %478 = fsub <4 x float> %421, %476, !dbg !27
  %479 = fsub <4 x float> %427, %476, !dbg !27
  %480 = fsub <4 x float> %433, %476, !dbg !27
  %481 = fsub <1 x float> %58, %475, !dbg !28
  %482 = extractelement <1 x float> %481, i64 0, !dbg !29
  %483 = call float @llvm.amdgcn.exp2.f32(float %482), !dbg !29
  %484 = insertelement <4 x float> poison, float %483, i64 0, !dbg !34
  %485 = shufflevector <4 x float> %484, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %486 = fmul <4 x float> %66, %485, !dbg !34
  %487 = fmul <4 x float> %65, %485, !dbg !34
  %488 = fmul <4 x float> %63, %485, !dbg !34
  %489 = fmul <4 x float> %62, %485, !dbg !34
  %490 = fmul <4 x float> %60, %485, !dbg !34
  %491 = fmul <4 x float> %59, %485, !dbg !34
  %492 = fmul <4 x float> %55, %485, !dbg !34
  %493 = fmul <4 x float> %54, %485, !dbg !34
  %494 = extractelement <1 x float> %57, i64 0, !dbg !30
  %495 = fmul float %494, %483, !dbg !30
  %496 = extractelement <4 x float> %477, i64 0, !dbg !31
  %497 = call float @llvm.amdgcn.exp2.f32(float %496), !dbg !31
  %498 = insertelement <4 x float> poison, float %497, i64 0, !dbg !33
  %499 = extractelement <4 x float> %477, i64 1, !dbg !31
  %500 = call float @llvm.amdgcn.exp2.f32(float %499), !dbg !31
  %501 = insertelement <4 x float> %498, float %500, i64 1, !dbg !33
  %502 = fadd float %497, %500, !dbg !32
  %503 = extractelement <4 x float> %477, i64 2, !dbg !31
  %504 = call float @llvm.amdgcn.exp2.f32(float %503), !dbg !31
  %505 = insertelement <4 x float> %501, float %504, i64 2, !dbg !33
  %506 = fadd float %502, %504, !dbg !32
  %507 = extractelement <4 x float> %477, i64 3, !dbg !31
  %508 = call float @llvm.amdgcn.exp2.f32(float %507), !dbg !31
  %509 = insertelement <4 x float> %505, float %508, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %509) #9, !dbg !33, !frisk.register_lifetimes !1
  %510 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %510) #9, !dbg !32, !frisk.register_lifetimes !1
  %511 = fadd float %506, %508, !dbg !32
  %512 = extractelement <4 x float> %478, i64 0, !dbg !31
  %513 = call float @llvm.amdgcn.exp2.f32(float %512), !dbg !31
  %514 = insertelement <4 x float> poison, float %513, i64 0, !dbg !33
  %515 = fadd float %511, %513, !dbg !32
  %516 = extractelement <4 x float> %478, i64 1, !dbg !31
  %517 = call float @llvm.amdgcn.exp2.f32(float %516), !dbg !31
  %518 = insertelement <4 x float> %514, float %517, i64 1, !dbg !33
  %519 = fadd float %515, %517, !dbg !32
  %520 = extractelement <4 x float> %478, i64 2, !dbg !31
  %521 = call float @llvm.amdgcn.exp2.f32(float %520), !dbg !31
  %522 = insertelement <4 x float> %518, float %521, i64 2, !dbg !33
  %523 = fadd float %519, %521, !dbg !32
  %524 = extractelement <4 x float> %478, i64 3, !dbg !31
  %525 = call float @llvm.amdgcn.exp2.f32(float %524), !dbg !31
  %526 = insertelement <4 x float> %522, float %525, i64 3, !dbg !33
  %math.result32 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %526) #9, !dbg !33, !frisk.register_lifetimes !1
  %527 = fptrunc <4 x float> %math.result32 to <4 x half>, !dbg !33
  %math.packed33 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %527) #9, !dbg !32, !frisk.register_lifetimes !1
  %528 = fadd float %523, %525, !dbg !32
  %529 = extractelement <4 x float> %479, i64 0, !dbg !31
  %530 = call float @llvm.amdgcn.exp2.f32(float %529), !dbg !31
  %531 = insertelement <4 x float> poison, float %530, i64 0, !dbg !33
  %532 = fadd float %528, %530, !dbg !32
  %533 = extractelement <4 x float> %479, i64 1, !dbg !31
  %534 = call float @llvm.amdgcn.exp2.f32(float %533), !dbg !31
  %535 = insertelement <4 x float> %531, float %534, i64 1, !dbg !33
  %536 = fadd float %532, %534, !dbg !32
  %537 = extractelement <4 x float> %479, i64 2, !dbg !31
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !31
  %539 = insertelement <4 x float> %535, float %538, i64 2, !dbg !33
  %540 = fadd float %536, %538, !dbg !32
  %541 = extractelement <4 x float> %479, i64 3, !dbg !31
  %542 = call float @llvm.amdgcn.exp2.f32(float %541), !dbg !31
  %543 = insertelement <4 x float> %539, float %542, i64 3, !dbg !33
  %math.result34 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %543) #9, !dbg !33, !frisk.register_lifetimes !1
  %544 = fptrunc <4 x float> %math.result34 to <4 x half>, !dbg !33
  %math.packed35 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %544) #9, !dbg !32, !frisk.register_lifetimes !1
  %545 = fadd float %540, %542, !dbg !32
  %546 = extractelement <4 x float> %480, i64 0, !dbg !31
  %547 = call float @llvm.amdgcn.exp2.f32(float %546), !dbg !31
  %548 = insertelement <4 x float> poison, float %547, i64 0, !dbg !33
  %549 = fadd float %545, %547, !dbg !32
  %550 = extractelement <4 x float> %480, i64 1, !dbg !31
  %551 = call float @llvm.amdgcn.exp2.f32(float %550), !dbg !31
  %552 = insertelement <4 x float> %548, float %551, i64 1, !dbg !33
  %553 = fadd float %549, %551, !dbg !32
  %554 = extractelement <4 x float> %480, i64 2, !dbg !31
  %555 = call float @llvm.amdgcn.exp2.f32(float %554), !dbg !31
  %556 = insertelement <4 x float> %552, float %555, i64 2, !dbg !33
  %557 = fadd float %553, %555, !dbg !32
  %558 = extractelement <4 x float> %480, i64 3, !dbg !31
  %559 = call float @llvm.amdgcn.exp2.f32(float %558), !dbg !31
  %560 = insertelement <4 x float> %556, float %559, i64 3, !dbg !33
  %math.result36 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %560) #9, !dbg !33, !frisk.register_lifetimes !1
  %561 = fptrunc <4 x float> %math.result36 to <4 x half>, !dbg !33
  %math.packed37 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %561) #9, !dbg !32, !frisk.register_lifetimes !1
  %562 = fadd float %557, %559, !dbg !32
  %563 = bitcast float %562 to i32, !dbg !32
  %564 = call i32 @llvm.amdgcn.ds.bpermute(i32 %461, i32 %563), !dbg !32
  %565 = bitcast i32 %564 to float, !dbg !32
  %566 = fadd float %562, %565, !dbg !32
  %567 = bitcast float %566 to i32, !dbg !32
  %568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %469, i32 %567), !dbg !32
  %569 = bitcast i32 %568 to float, !dbg !32
  %570 = fadd float %566, %569, !dbg !32
  %571 = fadd float %570, %495, !dbg !32
  %572 = insertelement <1 x float> poison, float %571, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread137 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local138 = zext nneg i32 %address.thread137 to i64, !dbg !12
  %address.local139 = and i64 %address.local138, 60, !dbg !14
  %address.local140 = add nuw nsw i64 %address.local139, %address.local138, !dbg !14
  %address.local141 = shl nuw nsw i64 %address.local140, 3, !dbg !14
  %address.local142 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local141, !dbg !14
  %address.local143 = and i64 %address.local138, 63, !dbg !12
  %address.local144 = lshr i64 %address.local143, 4, !dbg !14
  %address.local145 = mul nuw nsw i64 %address.local144, 384, !dbg !14
  %address.local146 = getelementptr [2 x i8], ptr addrspace(3) %address.local142, i64 %address.local145, !dbg !14
  %address.local147 = lshr i64 %address.local138, 2, !dbg !16
  %address.local148 = mul nsw i64 %address.local147, -32, !dbg !14
  %address.local149 = getelementptr [2 x i8], ptr addrspace(3) %address.local146, i64 %address.local148, !dbg !14
  %573 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local149, i32 0) #0, !dbg !14
  %574 = bitcast <4 x i32> %573 to <8 x half>, !dbg !14
  %address.local150 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local141, !dbg !14
  %address.local151 = getelementptr [2 x i8], ptr addrspace(3) %address.local150, i64 %address.local145, !dbg !14
  %address.local152 = getelementptr [2 x i8], ptr addrspace(3) %address.local151, i64 %address.local148, !dbg !14
  %address.local153 = getelementptr i8, ptr addrspace(3) %address.local152, i64 256, !dbg !14
  %575 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local153, i32 0) #0, !dbg !14
  %576 = bitcast <4 x i32> %575 to <8 x half>, !dbg !14
  %address.local154 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local141, !dbg !14
  %address.local155 = getelementptr [2 x i8], ptr addrspace(3) %address.local154, i64 %address.local145, !dbg !14
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) %address.local155, i64 %address.local148, !dbg !14
  %address.local157 = getelementptr i8, ptr addrspace(3) %address.local156, i64 512, !dbg !14
  %577 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local157, i32 0) #0, !dbg !14
  %578 = bitcast <4 x i32> %577 to <8 x half>, !dbg !14
  %address.local158 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local141, !dbg !14
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) %address.local158, i64 %address.local145, !dbg !14
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) %address.local159, i64 %address.local148, !dbg !14
  %address.local161 = getelementptr i8, ptr addrspace(3) %address.local160, i64 768, !dbg !14
  %579 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local161, i32 0) #0, !dbg !14
  %580 = bitcast <4 x i32> %579 to <8 x half>, !dbg !14
  %581 = add nuw nsw i64 %87, %15, !dbg !17
  %582 = shl i64 %581, 19, !dbg !17
  %583 = and i64 %582, 16252928, !dbg !17
  %584 = add i64 %583, %93, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %585 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %584, !dbg !17
  %586 = getelementptr i8, ptr addrspace(1) %585, i64 64, !dbg !17
  %address.local162 = shl nuw nsw i64 %address.local138, 3, !dbg !11
  %address.local163 = and i64 %address.local162, 1536, !dbg !11
  %address.local164 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local163, !dbg !11
  %address.local165 = ptrtoint ptr addrspace(3) %address.local164 to i64, !dbg !11
  %address.local166 = trunc i64 %address.local165 to i32, !dbg !11
  %587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local166), !dbg !17
  %588 = zext i32 %587 to i64, !dbg !17
  %589 = inttoptr i64 %588 to ptr addrspace(3), !dbg !17
  %address.local167 = shl nuw nsw i64 %address.local138, 5, !dbg !17
  %address.local168 = and i64 %address.local167, 8064, !dbg !17
  %address.local169 = shl nuw nsw i64 %address.local138, 3, !dbg !17
  %address.local170 = and i64 %address.local169, 24, !dbg !17
  %address.local171 = or disjoint i64 %address.local168, %address.local170, !dbg !17
  %address.local172 = getelementptr [2 x i8], ptr addrspace(1) %586, i64 %address.local171, !dbg !17
  %590 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %591 = ptrtoint ptr addrspace(1) %address.local172 to i64, !dbg !17
  %592 = sub i64 %591, %590, !dbg !17
  %dma.buffer.offset385 = trunc i64 %592 to i32, !dbg !17
  %593 = trunc i64 %590 to i32, !dbg !17
  %594 = insertelement <4 x i32> poison, i32 %593, i64 0, !dbg !17
  %595 = lshr i64 %590, 32, !dbg !17
  %596 = trunc i64 %595 to i32, !dbg !17
  %597 = insertelement <4 x i32> %594, i32 %596, i64 1, !dbg !17
  %598 = insertelement <4 x i32> %597, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc386 = insertelement <4 x i32> %598, i32 131072, i64 3, !dbg !17
  %599 = ptrtoint ptr addrspace(3) %589 to i32, !dbg !17
  %dma.buffer.lds387 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %599), !dbg !17
  %600 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds387, i32 %dma.buffer.offset385, <4 x i32> %dma.buffer.rsrc386) #10, !dbg !17
  %601 = shufflevector <8 x half> %574, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %601, <4 x float> %493) #9, !dbg !14
  %603 = shufflevector <8 x half> %576, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %604 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %603, <4 x float> %602) #9, !dbg !14
  %605 = shufflevector <8 x half> %578, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %606 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed35, <4 x half> %605, <4 x float> %604) #9, !dbg !14
  %607 = shufflevector <8 x half> %580, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %607, <4 x float> %606) #9, !dbg !14
  %609 = shufflevector <8 x half> %574, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %609, <4 x float> %492) #9, !dbg !14
  %611 = shufflevector <8 x half> %576, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %611, <4 x float> %610) #9, !dbg !14
  %613 = shufflevector <8 x half> %578, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed35, <4 x half> %613, <4 x float> %612) #9, !dbg !14
  %615 = shufflevector <8 x half> %580, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %615, <4 x float> %614) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %617 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local149, i32 0) #0, !dbg !14
  %618 = bitcast <4 x i32> %617 to <8 x half>, !dbg !14
  %619 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local153, i32 0) #0, !dbg !14
  %620 = bitcast <4 x i32> %619 to <8 x half>, !dbg !14
  %621 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local157, i32 0) #0, !dbg !14
  %622 = bitcast <4 x i32> %621 to <8 x half>, !dbg !14
  %623 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local161, i32 0) #0, !dbg !14
  %624 = bitcast <4 x i32> %623 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %625 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %584, !dbg !17
  %626 = getelementptr i8, ptr addrspace(1) %625, i64 128, !dbg !17
  %627 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local166), !dbg !17
  %628 = zext i32 %627 to i64, !dbg !17
  %629 = inttoptr i64 %628 to ptr addrspace(3), !dbg !17
  %address.local173 = getelementptr [2 x i8], ptr addrspace(1) %626, i64 %address.local171, !dbg !17
  %630 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %631 = ptrtoint ptr addrspace(1) %address.local173 to i64, !dbg !17
  %632 = sub i64 %631, %630, !dbg !17
  %dma.buffer.offset382 = trunc i64 %632 to i32, !dbg !17
  %633 = trunc i64 %630 to i32, !dbg !17
  %634 = insertelement <4 x i32> poison, i32 %633, i64 0, !dbg !17
  %635 = lshr i64 %630, 32, !dbg !17
  %636 = trunc i64 %635 to i32, !dbg !17
  %637 = insertelement <4 x i32> %634, i32 %636, i64 1, !dbg !17
  %638 = insertelement <4 x i32> %637, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc383 = insertelement <4 x i32> %638, i32 131072, i64 3, !dbg !17
  %639 = ptrtoint ptr addrspace(3) %629 to i32, !dbg !17
  %dma.buffer.lds384 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %639), !dbg !17
  %640 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds384, i32 %dma.buffer.offset382, <4 x i32> %dma.buffer.rsrc383) #10, !dbg !17
  %641 = shufflevector <8 x half> %618, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %641, <4 x float> %491) #9, !dbg !14
  %643 = shufflevector <8 x half> %620, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %643, <4 x float> %642) #9, !dbg !14
  %645 = shufflevector <8 x half> %622, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed35, <4 x half> %645, <4 x float> %644) #9, !dbg !14
  %647 = shufflevector <8 x half> %624, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %647, <4 x float> %646) #9, !dbg !14
  %649 = shufflevector <8 x half> %618, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %649, <4 x float> %490) #9, !dbg !14
  %651 = shufflevector <8 x half> %620, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %651, <4 x float> %650) #9, !dbg !14
  %653 = shufflevector <8 x half> %622, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed35, <4 x half> %653, <4 x float> %652) #9, !dbg !14
  %655 = shufflevector <8 x half> %624, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %655, <4 x float> %654) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %657 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local149, i32 0) #0, !dbg !14
  %658 = bitcast <4 x i32> %657 to <8 x half>, !dbg !14
  %659 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local153, i32 0) #0, !dbg !14
  %660 = bitcast <4 x i32> %659 to <8 x half>, !dbg !14
  %661 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local157, i32 0) #0, !dbg !14
  %662 = bitcast <4 x i32> %661 to <8 x half>, !dbg !14
  %663 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local161, i32 0) #0, !dbg !14
  %664 = bitcast <4 x i32> %663 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %665 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %584, !dbg !17
  %666 = getelementptr i8, ptr addrspace(1) %665, i64 192, !dbg !17
  %667 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local166), !dbg !17
  %668 = zext i32 %667 to i64, !dbg !17
  %669 = inttoptr i64 %668 to ptr addrspace(3), !dbg !17
  %address.local174 = getelementptr [2 x i8], ptr addrspace(1) %666, i64 %address.local171, !dbg !17
  %670 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %671 = ptrtoint ptr addrspace(1) %address.local174 to i64, !dbg !17
  %672 = sub i64 %671, %670, !dbg !17
  %dma.buffer.offset = trunc i64 %672 to i32, !dbg !17
  %673 = trunc i64 %670 to i32, !dbg !17
  %674 = insertelement <4 x i32> poison, i32 %673, i64 0, !dbg !17
  %675 = lshr i64 %670, 32, !dbg !17
  %676 = trunc i64 %675 to i32, !dbg !17
  %677 = insertelement <4 x i32> %674, i32 %676, i64 1, !dbg !17
  %678 = insertelement <4 x i32> %677, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %678, i32 131072, i64 3, !dbg !17
  %679 = ptrtoint ptr addrspace(3) %669 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %679), !dbg !17
  %680 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %681 = shufflevector <8 x half> %658, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %681, <4 x float> %489) #9, !dbg !14
  %683 = shufflevector <8 x half> %660, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %684 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %683, <4 x float> %682) #9, !dbg !14
  %685 = shufflevector <8 x half> %662, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed35, <4 x half> %685, <4 x float> %684) #9, !dbg !14
  %687 = shufflevector <8 x half> %664, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %687, <4 x float> %686) #9, !dbg !14
  %689 = shufflevector <8 x half> %658, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %689, <4 x float> %488) #9, !dbg !14
  %691 = shufflevector <8 x half> %660, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %691, <4 x float> %690) #9, !dbg !14
  %693 = shufflevector <8 x half> %662, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed35, <4 x half> %693, <4 x float> %692) #9, !dbg !14
  %695 = shufflevector <8 x half> %664, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %695, <4 x float> %694) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %697 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local149, i32 0) #0, !dbg !14
  %698 = bitcast <4 x i32> %697 to <8 x half>, !dbg !14
  %699 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local153, i32 0) #0, !dbg !14
  %700 = bitcast <4 x i32> %699 to <8 x half>, !dbg !14
  %701 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local157, i32 0) #0, !dbg !14
  %702 = bitcast <4 x i32> %701 to <8 x half>, !dbg !14
  %703 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local161, i32 0) #0, !dbg !14
  %704 = bitcast <4 x i32> %703 to <8 x half>, !dbg !14
  %705 = shufflevector <8 x half> %698, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %706 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %705, <4 x float> %487) #9, !dbg !14
  %707 = shufflevector <8 x half> %700, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %708 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %707, <4 x float> %706) #9, !dbg !14
  %709 = shufflevector <8 x half> %702, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %710 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed35, <4 x half> %709, <4 x float> %708) #9, !dbg !14
  %711 = shufflevector <8 x half> %704, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %712 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %711, <4 x float> %710) #9, !dbg !14
  %713 = shufflevector <8 x half> %698, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %714 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %713, <4 x float> %486) #9, !dbg !14
  %715 = shufflevector <8 x half> %700, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %715, <4 x float> %714) #9, !dbg !14
  %717 = shufflevector <8 x half> %702, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed35, <4 x half> %717, <4 x float> %716) #9, !dbg !14
  %719 = shufflevector <8 x half> %704, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %720 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %719, <4 x float> %718) #9, !dbg !14
  %721 = shufflevector <4 x float> %608, <4 x float> %616, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %722 = shufflevector <4 x float> %648, <4 x float> %656, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %723 = shufflevector <4 x float> %688, <4 x float> %696, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %724 = shufflevector <4 x float> %712, <4 x float> %720, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %725 = add nuw nsw i64 %53, 1, !dbg !7
  br label %52, !dbg !7

726:                                              ; preds = %52
  %727 = and i32 %34, 16, !dbg !7
  %.not = icmp eq i32 %727, 0, !dbg !7
  %728 = and i32 %34, 32, !dbg !7
  %.not1 = icmp eq i32 %728, 0, !dbg !7
  %729 = extractelement <8 x float> %67, i64 0, !dbg !7
  %730 = extractelement <8 x float> %67, i64 1, !dbg !7
  %731 = extractelement <8 x float> %67, i64 2, !dbg !7
  %732 = extractelement <8 x float> %67, i64 3, !dbg !7
  %733 = select i1 %.not, float %730, float %729, !dbg !7
  %734 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %735 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %734), !dbg !7
  %736 = add nuw nsw i32 %735, 64, !dbg !7
  %737 = xor i32 %735, 16, !dbg !7
  %738 = and i32 %736, 192, !dbg !7
  %739 = icmp samesign ult i32 %737, %738, !dbg !7
  %740 = select i1 %739, i32 %737, i32 %735, !dbg !7
  %741 = shl nuw nsw i32 %740, 2, !dbg !7
  %742 = bitcast float %733 to i32, !dbg !7
  %743 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %742), !dbg !7
  %744 = bitcast i32 %743 to float, !dbg !7
  %745 = select i1 %.not, float %729, float %744, !dbg !7
  %746 = select i1 %.not, float %744, float %730, !dbg !7
  %747 = select i1 %.not, float %732, float %731, !dbg !7
  %748 = bitcast float %747 to i32, !dbg !7
  %749 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %748), !dbg !7
  %750 = bitcast i32 %749 to float, !dbg !7
  %751 = select i1 %.not, float %731, float %750, !dbg !7
  %752 = select i1 %.not, float %750, float %732, !dbg !7
  %753 = select i1 %.not1, float %751, float %745, !dbg !7
  %754 = xor i32 %735, 32, !dbg !7
  %755 = icmp samesign ult i32 %754, %738, !dbg !7
  %756 = select i1 %755, i32 %754, i32 %735, !dbg !7
  %757 = shl nuw nsw i32 %756, 2, !dbg !7
  %758 = bitcast float %753 to i32, !dbg !7
  %759 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %758), !dbg !7
  %760 = bitcast i32 %759 to float, !dbg !7
  %761 = select i1 %.not1, float %745, float %760, !dbg !7
  %762 = select i1 %.not1, float %760, float %751, !dbg !7
  %763 = select i1 %.not1, float %752, float %746, !dbg !7
  %764 = bitcast float %763 to i32, !dbg !7
  %765 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %764), !dbg !7
  %766 = bitcast i32 %765 to float, !dbg !7
  %767 = select i1 %.not1, float %746, float %766, !dbg !7
  %768 = select i1 %.not1, float %766, float %752, !dbg !7
  %769 = extractelement <8 x float> %64, i64 0, !dbg !7
  %770 = extractelement <8 x float> %64, i64 1, !dbg !7
  %771 = extractelement <8 x float> %64, i64 2, !dbg !7
  %772 = extractelement <8 x float> %64, i64 3, !dbg !7
  %773 = select i1 %.not, float %770, float %769, !dbg !7
  %774 = bitcast float %773 to i32, !dbg !7
  %775 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %774), !dbg !7
  %776 = bitcast i32 %775 to float, !dbg !7
  %777 = select i1 %.not, float %769, float %776, !dbg !7
  %778 = select i1 %.not, float %776, float %770, !dbg !7
  %779 = select i1 %.not, float %772, float %771, !dbg !7
  %780 = bitcast float %779 to i32, !dbg !7
  %781 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %780), !dbg !7
  %782 = bitcast i32 %781 to float, !dbg !7
  %783 = select i1 %.not, float %771, float %782, !dbg !7
  %784 = select i1 %.not, float %782, float %772, !dbg !7
  %785 = select i1 %.not1, float %783, float %777, !dbg !7
  %786 = bitcast float %785 to i32, !dbg !7
  %787 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %786), !dbg !7
  %788 = bitcast i32 %787 to float, !dbg !7
  %789 = select i1 %.not1, float %777, float %788, !dbg !7
  %790 = select i1 %.not1, float %788, float %783, !dbg !7
  %791 = select i1 %.not1, float %784, float %778, !dbg !7
  %792 = bitcast float %791 to i32, !dbg !7
  %793 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %792), !dbg !7
  %794 = bitcast i32 %793 to float, !dbg !7
  %795 = select i1 %.not1, float %778, float %794, !dbg !7
  %796 = select i1 %.not1, float %794, float %784, !dbg !7
  %797 = extractelement <8 x float> %61, i64 0, !dbg !7
  %798 = extractelement <8 x float> %61, i64 1, !dbg !7
  %799 = extractelement <8 x float> %61, i64 2, !dbg !7
  %800 = extractelement <8 x float> %61, i64 3, !dbg !7
  %801 = select i1 %.not, float %798, float %797, !dbg !7
  %802 = bitcast float %801 to i32, !dbg !7
  %803 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %802), !dbg !7
  %804 = bitcast i32 %803 to float, !dbg !7
  %805 = select i1 %.not, float %797, float %804, !dbg !7
  %806 = select i1 %.not, float %804, float %798, !dbg !7
  %807 = select i1 %.not, float %800, float %799, !dbg !7
  %808 = bitcast float %807 to i32, !dbg !7
  %809 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %808), !dbg !7
  %810 = bitcast i32 %809 to float, !dbg !7
  %811 = select i1 %.not, float %799, float %810, !dbg !7
  %812 = select i1 %.not, float %810, float %800, !dbg !7
  %813 = select i1 %.not1, float %811, float %805, !dbg !7
  %814 = bitcast float %813 to i32, !dbg !7
  %815 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %814), !dbg !7
  %816 = bitcast i32 %815 to float, !dbg !7
  %817 = select i1 %.not1, float %805, float %816, !dbg !7
  %818 = select i1 %.not1, float %816, float %811, !dbg !7
  %819 = select i1 %.not1, float %812, float %806, !dbg !7
  %820 = bitcast float %819 to i32, !dbg !7
  %821 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %820), !dbg !7
  %822 = bitcast i32 %821 to float, !dbg !7
  %823 = select i1 %.not1, float %806, float %822, !dbg !7
  %824 = select i1 %.not1, float %822, float %812, !dbg !7
  %825 = extractelement <8 x float> %56, i64 0, !dbg !7
  %826 = extractelement <8 x float> %56, i64 1, !dbg !7
  %827 = extractelement <8 x float> %56, i64 2, !dbg !7
  %828 = extractelement <8 x float> %56, i64 3, !dbg !7
  %829 = select i1 %.not, float %826, float %825, !dbg !7
  %830 = bitcast float %829 to i32, !dbg !7
  %831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %830), !dbg !7
  %832 = bitcast i32 %831 to float, !dbg !7
  %833 = select i1 %.not, float %825, float %832, !dbg !7
  %834 = select i1 %.not, float %832, float %826, !dbg !7
  %835 = select i1 %.not, float %828, float %827, !dbg !7
  %836 = bitcast float %835 to i32, !dbg !7
  %837 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %836), !dbg !7
  %838 = bitcast i32 %837 to float, !dbg !7
  %839 = select i1 %.not, float %827, float %838, !dbg !7
  %840 = select i1 %.not, float %838, float %828, !dbg !7
  %841 = select i1 %.not1, float %839, float %833, !dbg !7
  %842 = bitcast float %841 to i32, !dbg !7
  %843 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %842), !dbg !7
  %844 = select i1 %.not1, float %840, float %834, !dbg !7
  %845 = bitcast float %844 to i32, !dbg !7
  %846 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %845), !dbg !7
  %847 = shufflevector <1 x float> %57, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %848 = extractelement <8 x float> %67, i64 4, !dbg !7
  %849 = extractelement <8 x float> %67, i64 5, !dbg !7
  %850 = extractelement <8 x float> %67, i64 6, !dbg !7
  %851 = extractelement <8 x float> %67, i64 7, !dbg !7
  %852 = select i1 %.not, float %849, float %848, !dbg !7
  %853 = bitcast float %852 to i32, !dbg !7
  %854 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %853), !dbg !7
  %855 = bitcast i32 %854 to float, !dbg !7
  %856 = select i1 %.not, float %848, float %855, !dbg !7
  %857 = select i1 %.not, float %855, float %849, !dbg !7
  %858 = select i1 %.not, float %851, float %850, !dbg !7
  %859 = bitcast float %858 to i32, !dbg !7
  %860 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %859), !dbg !7
  %861 = bitcast i32 %860 to float, !dbg !7
  %862 = select i1 %.not, float %850, float %861, !dbg !7
  %863 = select i1 %.not, float %861, float %851, !dbg !7
  %864 = select i1 %.not1, float %862, float %856, !dbg !7
  %865 = bitcast float %864 to i32, !dbg !7
  %866 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %865), !dbg !7
  %867 = bitcast i32 %866 to float, !dbg !7
  %868 = select i1 %.not1, float %856, float %867, !dbg !7
  %869 = select i1 %.not1, float %867, float %862, !dbg !7
  %870 = select i1 %.not1, float %863, float %857, !dbg !7
  %871 = bitcast float %870 to i32, !dbg !7
  %872 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %871), !dbg !7
  %873 = bitcast i32 %872 to float, !dbg !7
  %874 = select i1 %.not1, float %857, float %873, !dbg !7
  %875 = select i1 %.not1, float %873, float %863, !dbg !7
  %876 = extractelement <8 x float> %64, i64 4, !dbg !7
  %877 = extractelement <8 x float> %64, i64 5, !dbg !7
  %878 = extractelement <8 x float> %64, i64 6, !dbg !7
  %879 = extractelement <8 x float> %64, i64 7, !dbg !7
  %880 = select i1 %.not, float %877, float %876, !dbg !7
  %881 = bitcast float %880 to i32, !dbg !7
  %882 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %881), !dbg !7
  %883 = bitcast i32 %882 to float, !dbg !7
  %884 = select i1 %.not, float %876, float %883, !dbg !7
  %885 = select i1 %.not, float %883, float %877, !dbg !7
  %886 = select i1 %.not, float %879, float %878, !dbg !7
  %887 = bitcast float %886 to i32, !dbg !7
  %888 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %887), !dbg !7
  %889 = bitcast i32 %888 to float, !dbg !7
  %890 = select i1 %.not, float %878, float %889, !dbg !7
  %891 = select i1 %.not, float %889, float %879, !dbg !7
  %892 = select i1 %.not1, float %890, float %884, !dbg !7
  %893 = bitcast float %892 to i32, !dbg !7
  %894 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %893), !dbg !7
  %895 = bitcast i32 %894 to float, !dbg !7
  %896 = select i1 %.not1, float %884, float %895, !dbg !7
  %897 = select i1 %.not1, float %895, float %890, !dbg !7
  %898 = select i1 %.not1, float %891, float %885, !dbg !7
  %899 = bitcast float %898 to i32, !dbg !7
  %900 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %899), !dbg !7
  %901 = bitcast i32 %900 to float, !dbg !7
  %902 = select i1 %.not1, float %885, float %901, !dbg !7
  %903 = select i1 %.not1, float %901, float %891, !dbg !7
  %904 = extractelement <8 x float> %61, i64 4, !dbg !7
  %905 = extractelement <8 x float> %61, i64 5, !dbg !7
  %906 = extractelement <8 x float> %61, i64 6, !dbg !7
  %907 = extractelement <8 x float> %61, i64 7, !dbg !7
  %908 = select i1 %.not, float %905, float %904, !dbg !7
  %909 = bitcast float %908 to i32, !dbg !7
  %910 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %909), !dbg !7
  %911 = bitcast i32 %910 to float, !dbg !7
  %912 = select i1 %.not, float %904, float %911, !dbg !7
  %913 = select i1 %.not, float %911, float %905, !dbg !7
  %914 = select i1 %.not, float %907, float %906, !dbg !7
  %915 = bitcast float %914 to i32, !dbg !7
  %916 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %915), !dbg !7
  %917 = bitcast i32 %916 to float, !dbg !7
  %918 = select i1 %.not, float %906, float %917, !dbg !7
  %919 = select i1 %.not, float %917, float %907, !dbg !7
  %920 = select i1 %.not1, float %918, float %912, !dbg !7
  %921 = bitcast float %920 to i32, !dbg !7
  %922 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %921), !dbg !7
  %923 = bitcast i32 %922 to float, !dbg !7
  %924 = select i1 %.not1, float %912, float %923, !dbg !7
  %925 = select i1 %.not1, float %923, float %918, !dbg !7
  %926 = select i1 %.not1, float %919, float %913, !dbg !7
  %927 = bitcast float %926 to i32, !dbg !7
  %928 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %927), !dbg !7
  %929 = bitcast i32 %928 to float, !dbg !7
  %930 = select i1 %.not1, float %913, float %929, !dbg !7
  %931 = select i1 %.not1, float %929, float %919, !dbg !7
  %932 = extractelement <8 x float> %56, i64 4, !dbg !7
  %933 = extractelement <8 x float> %56, i64 5, !dbg !7
  %934 = extractelement <8 x float> %56, i64 6, !dbg !7
  %935 = extractelement <8 x float> %56, i64 7, !dbg !7
  %936 = select i1 %.not, float %933, float %932, !dbg !7
  %937 = bitcast float %936 to i32, !dbg !7
  %938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %937), !dbg !7
  %939 = bitcast i32 %938 to float, !dbg !7
  %940 = select i1 %.not, float %932, float %939, !dbg !7
  %941 = select i1 %.not, float %939, float %933, !dbg !7
  %942 = select i1 %.not, float %935, float %934, !dbg !7
  %943 = bitcast float %942 to i32, !dbg !7
  %944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %943), !dbg !7
  %945 = bitcast i32 %944 to float, !dbg !7
  %946 = select i1 %.not, float %934, float %945, !dbg !7
  %947 = select i1 %.not, float %945, float %935, !dbg !7
  %948 = select i1 %.not1, float %946, float %940, !dbg !7
  %949 = bitcast float %948 to i32, !dbg !7
  %950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %949), !dbg !7
  %951 = select i1 %.not1, float %947, float %941, !dbg !7
  %952 = bitcast float %951 to i32, !dbg !7
  %953 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %952), !dbg !7
  br i1 true, label %954, label %988, !dbg !20

954:                                              ; preds = %726
  %955 = bitcast i32 %950 to float, !dbg !7
  %956 = select i1 %.not1, float %940, float %955, !dbg !7
  %957 = insertelement <4 x float> poison, float %956, i64 0, !dbg !18
  %958 = bitcast i32 %953 to float, !dbg !7
  %959 = select i1 %.not1, float %941, float %958, !dbg !7
  %960 = insertelement <4 x float> %957, float %959, i64 1, !dbg !18
  %961 = select i1 %.not1, float %955, float %946, !dbg !7
  %962 = insertelement <4 x float> %960, float %961, i64 2, !dbg !18
  %963 = select i1 %.not1, float %958, float %947, !dbg !7
  %964 = insertelement <4 x float> %962, float %963, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %847, !dbg !18
  %normalized = fmul <4 x float> %964, %row.reciprocal, !dbg !18
  %965 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %966 = extractelement <4 x half> %965, i64 3, !dbg !19
  %967 = extractelement <4 x half> %965, i64 2, !dbg !19
  %968 = extractelement <4 x half> %965, i64 1, !dbg !19
  %969 = extractelement <4 x half> %965, i64 0, !dbg !19
  %970 = bitcast i32 %843 to float, !dbg !7
  %971 = select i1 %.not1, float %833, float %970, !dbg !7
  %972 = insertelement <4 x float> poison, float %971, i64 0, !dbg !18
  %973 = bitcast i32 %846 to float, !dbg !7
  %974 = select i1 %.not1, float %834, float %973, !dbg !7
  %975 = insertelement <4 x float> %972, float %974, i64 1, !dbg !18
  %976 = select i1 %.not1, float %970, float %839, !dbg !7
  %977 = insertelement <4 x float> %975, float %976, i64 2, !dbg !18
  %978 = select i1 %.not1, float %973, float %840, !dbg !7
  %979 = insertelement <4 x float> %977, float %978, i64 3, !dbg !18
  %normalized20 = fmul <4 x float> %979, %row.reciprocal, !dbg !18
  %980 = fptrunc <4 x float> %normalized20 to <4 x half>, !dbg !19
  %981 = extractelement <4 x half> %980, i64 3, !dbg !19
  %982 = extractelement <4 x half> %980, i64 2, !dbg !19
  %983 = extractelement <4 x half> %980, i64 1, !dbg !19
  %984 = extractelement <4 x half> %980, i64 0, !dbg !19
  %985 = add nsw i64 %15, %25, !dbg !20
  %986 = shl i64 %985, 19, !dbg !20
  %987 = and i64 %986, 16252928, !dbg !20
  %address.thread175 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local176 = zext nneg i32 %address.thread175 to i64, !dbg !12
  %address.local177 = add i64 %28, %address.local176, !dbg !20
  %address.local178 = lshr i64 %address.local176, 6, !dbg !12
  %address.local179 = shl nuw nsw i64 %address.local178, 4, !dbg !20
  %address.local180 = add i64 %address.local177, %address.local179, !dbg !20
  %address.local181 = and i64 %address.local176, 240, !dbg !20
  %address.local182 = sub i64 %address.local180, %address.local181, !dbg !20
  %address.local183 = add i64 %address.local182, %29, !dbg !20
  %address.local184 = shl nuw nsw i64 %address.local183, 7, !dbg !20
  %address.local185 = add nuw nsw i64 %987, %address.local184, !dbg !20
  %address.local186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local185, !dbg !20
  %address.local187 = lshr i64 %address.local176, 2, !dbg !12
  %address.local188 = and i64 %address.local187, 12, !dbg !12
  %address.local189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local186, i64 %address.local188, !dbg !20
  store half %984, ptr addrspace(1) %address.local189, align 2, !dbg !20
  %address.local190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local185, !dbg !20
  %address.local191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local190, i64 %address.local188, !dbg !20
  %address.local192 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local191, i64 2, !dbg !20
  store half %983, ptr addrspace(1) %address.local192, align 2, !dbg !20
  %address.local193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local185, !dbg !20
  %address.local194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local193, i64 %address.local188, !dbg !20
  %address.local195 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local194, i64 4, !dbg !20
  store half %982, ptr addrspace(1) %address.local195, align 2, !dbg !20
  %address.local196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local185, !dbg !20
  %address.local197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local196, i64 %address.local188, !dbg !20
  %address.local198 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local197, i64 6, !dbg !20
  store half %981, ptr addrspace(1) %address.local198, align 2, !dbg !20
  %address.local199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local185, !dbg !20
  %address.local200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local199, i64 %address.local188, !dbg !20
  %address.local201 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local200, i64 32, !dbg !20
  store half %969, ptr addrspace(1) %address.local201, align 2, !dbg !20
  %address.local202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local185, !dbg !20
  %address.local203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local202, i64 %address.local188, !dbg !20
  %address.local204 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local203, i64 34, !dbg !20
  store half %968, ptr addrspace(1) %address.local204, align 2, !dbg !20
  %address.local205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local185, !dbg !20
  %address.local206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local205, i64 %address.local188, !dbg !20
  %address.local207 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local206, i64 36, !dbg !20
  store half %967, ptr addrspace(1) %address.local207, align 2, !dbg !20
  %address.local208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local185, !dbg !20
  %address.local209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local208, i64 %address.local188, !dbg !20
  %address.local210 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local209, i64 38, !dbg !20
  store half %966, ptr addrspace(1) %address.local210, align 2, !dbg !20
  br label %988, !dbg !20

988:                                              ; preds = %954, %726
  br i1 true, label %989, label %1009, !dbg !20

989:                                              ; preds = %988
  %990 = insertelement <4 x float> poison, float %924, i64 0, !dbg !18
  %991 = insertelement <4 x float> %990, float %930, i64 1, !dbg !18
  %992 = insertelement <4 x float> %991, float %925, i64 2, !dbg !18
  %993 = insertelement <4 x float> %992, float %931, i64 3, !dbg !18
  %row.reciprocal21 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %847, !dbg !18
  %normalized22 = fmul <4 x float> %993, %row.reciprocal21, !dbg !18
  %994 = fptrunc <4 x float> %normalized22 to <4 x half>, !dbg !19
  %995 = extractelement <4 x half> %994, i64 3, !dbg !19
  %996 = extractelement <4 x half> %994, i64 2, !dbg !19
  %997 = extractelement <4 x half> %994, i64 1, !dbg !19
  %998 = extractelement <4 x half> %994, i64 0, !dbg !19
  %999 = insertelement <4 x float> poison, float %817, i64 0, !dbg !18
  %1000 = insertelement <4 x float> %999, float %823, i64 1, !dbg !18
  %1001 = insertelement <4 x float> %1000, float %818, i64 2, !dbg !18
  %1002 = insertelement <4 x float> %1001, float %824, i64 3, !dbg !18
  %normalized23 = fmul <4 x float> %1002, %row.reciprocal21, !dbg !18
  %1003 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !19
  %1004 = extractelement <4 x half> %1003, i64 3, !dbg !19
  %1005 = extractelement <4 x half> %1003, i64 2, !dbg !19
  %1006 = extractelement <4 x half> %1003, i64 1, !dbg !19
  %1007 = extractelement <4 x half> %1003, i64 0, !dbg !19
  %1008 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread211 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local212 = zext nneg i32 %address.thread211 to i64, !dbg !12
  %address.local213 = add i64 %28, %address.local212, !dbg !20
  %address.local214 = lshr i64 %address.local212, 6, !dbg !12
  %address.local215 = shl nuw nsw i64 %address.local214, 4, !dbg !20
  %address.local216 = add i64 %address.local213, %address.local215, !dbg !20
  %address.local217 = and i64 %address.local212, 240, !dbg !20
  %address.local218 = sub i64 %address.local216, %address.local217, !dbg !20
  %address.local219 = add i64 %address.local218, %29, !dbg !20
  %address.local220 = shl nuw nsw i64 %address.local219, 7, !dbg !20
  %address.local221 = add nuw nsw i64 %1008, %address.local220, !dbg !20
  %address.local222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local221, !dbg !20
  %address.local223 = lshr i64 %address.local212, 2, !dbg !12
  %address.local224 = and i64 %address.local223, 12, !dbg !12
  %address.local225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local222, i64 %address.local224, !dbg !20
  %address.local226 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local225, i64 64, !dbg !20
  store half %1007, ptr addrspace(1) %address.local226, align 2, !dbg !20
  %address.local227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local221, !dbg !20
  %address.local228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local227, i64 %address.local224, !dbg !20
  %address.local229 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local228, i64 66, !dbg !20
  store half %1006, ptr addrspace(1) %address.local229, align 2, !dbg !20
  %address.local230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local221, !dbg !20
  %address.local231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local230, i64 %address.local224, !dbg !20
  %address.local232 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local231, i64 68, !dbg !20
  store half %1005, ptr addrspace(1) %address.local232, align 2, !dbg !20
  %address.local233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local221, !dbg !20
  %address.local234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local233, i64 %address.local224, !dbg !20
  %address.local235 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local234, i64 70, !dbg !20
  store half %1004, ptr addrspace(1) %address.local235, align 2, !dbg !20
  %address.local236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local221, !dbg !20
  %address.local237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local236, i64 %address.local224, !dbg !20
  %address.local238 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local237, i64 96, !dbg !20
  store half %998, ptr addrspace(1) %address.local238, align 2, !dbg !20
  %address.local239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local221, !dbg !20
  %address.local240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local239, i64 %address.local224, !dbg !20
  %address.local241 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local240, i64 98, !dbg !20
  store half %997, ptr addrspace(1) %address.local241, align 2, !dbg !20
  %address.local242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local221, !dbg !20
  %address.local243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local242, i64 %address.local224, !dbg !20
  %address.local244 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local243, i64 100, !dbg !20
  store half %996, ptr addrspace(1) %address.local244, align 2, !dbg !20
  %address.local245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local221, !dbg !20
  %address.local246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local245, i64 %address.local224, !dbg !20
  %address.local247 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local246, i64 102, !dbg !20
  store half %995, ptr addrspace(1) %address.local247, align 2, !dbg !20
  br label %1009, !dbg !20

1009:                                             ; preds = %989, %988
  br i1 true, label %1010, label %1030, !dbg !20

1010:                                             ; preds = %1009
  %1011 = insertelement <4 x float> poison, float %896, i64 0, !dbg !18
  %1012 = insertelement <4 x float> %1011, float %902, i64 1, !dbg !18
  %1013 = insertelement <4 x float> %1012, float %897, i64 2, !dbg !18
  %1014 = insertelement <4 x float> %1013, float %903, i64 3, !dbg !18
  %row.reciprocal24 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %847, !dbg !18
  %normalized25 = fmul <4 x float> %1014, %row.reciprocal24, !dbg !18
  %1015 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !19
  %1016 = extractelement <4 x half> %1015, i64 3, !dbg !19
  %1017 = extractelement <4 x half> %1015, i64 2, !dbg !19
  %1018 = extractelement <4 x half> %1015, i64 1, !dbg !19
  %1019 = extractelement <4 x half> %1015, i64 0, !dbg !19
  %1020 = insertelement <4 x float> poison, float %789, i64 0, !dbg !18
  %1021 = insertelement <4 x float> %1020, float %795, i64 1, !dbg !18
  %1022 = insertelement <4 x float> %1021, float %790, i64 2, !dbg !18
  %1023 = insertelement <4 x float> %1022, float %796, i64 3, !dbg !18
  %normalized26 = fmul <4 x float> %1023, %row.reciprocal24, !dbg !18
  %1024 = fptrunc <4 x float> %normalized26 to <4 x half>, !dbg !19
  %1025 = extractelement <4 x half> %1024, i64 3, !dbg !19
  %1026 = extractelement <4 x half> %1024, i64 2, !dbg !19
  %1027 = extractelement <4 x half> %1024, i64 1, !dbg !19
  %1028 = extractelement <4 x half> %1024, i64 0, !dbg !19
  %1029 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread248 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local249 = zext nneg i32 %address.thread248 to i64, !dbg !12
  %address.local250 = add i64 %28, %address.local249, !dbg !20
  %address.local251 = lshr i64 %address.local249, 6, !dbg !12
  %address.local252 = shl nuw nsw i64 %address.local251, 4, !dbg !20
  %address.local253 = add i64 %address.local250, %address.local252, !dbg !20
  %address.local254 = and i64 %address.local249, 240, !dbg !20
  %address.local255 = sub i64 %address.local253, %address.local254, !dbg !20
  %address.local256 = add i64 %address.local255, %29, !dbg !20
  %address.local257 = shl nuw nsw i64 %address.local256, 7, !dbg !20
  %address.local258 = add nuw nsw i64 %1029, %address.local257, !dbg !20
  %address.local259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local258, !dbg !20
  %address.local260 = lshr i64 %address.local249, 2, !dbg !12
  %address.local261 = and i64 %address.local260, 12, !dbg !12
  %address.local262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local259, i64 %address.local261, !dbg !20
  %address.local263 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local262, i64 128, !dbg !20
  store half %1028, ptr addrspace(1) %address.local263, align 2, !dbg !20
  %address.local264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local258, !dbg !20
  %address.local265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local264, i64 %address.local261, !dbg !20
  %address.local266 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local265, i64 130, !dbg !20
  store half %1027, ptr addrspace(1) %address.local266, align 2, !dbg !20
  %address.local267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local258, !dbg !20
  %address.local268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local267, i64 %address.local261, !dbg !20
  %address.local269 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local268, i64 132, !dbg !20
  store half %1026, ptr addrspace(1) %address.local269, align 2, !dbg !20
  %address.local270 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local258, !dbg !20
  %address.local271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local270, i64 %address.local261, !dbg !20
  %address.local272 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local271, i64 134, !dbg !20
  store half %1025, ptr addrspace(1) %address.local272, align 2, !dbg !20
  %address.local273 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local258, !dbg !20
  %address.local274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local273, i64 %address.local261, !dbg !20
  %address.local275 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local274, i64 160, !dbg !20
  store half %1019, ptr addrspace(1) %address.local275, align 2, !dbg !20
  %address.local276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local258, !dbg !20
  %address.local277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local276, i64 %address.local261, !dbg !20
  %address.local278 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local277, i64 162, !dbg !20
  store half %1018, ptr addrspace(1) %address.local278, align 2, !dbg !20
  %address.local279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local258, !dbg !20
  %address.local280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local279, i64 %address.local261, !dbg !20
  %address.local281 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local280, i64 164, !dbg !20
  store half %1017, ptr addrspace(1) %address.local281, align 2, !dbg !20
  %address.local282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local258, !dbg !20
  %address.local283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local282, i64 %address.local261, !dbg !20
  %address.local284 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local283, i64 166, !dbg !20
  store half %1016, ptr addrspace(1) %address.local284, align 2, !dbg !20
  br label %1030, !dbg !20

1030:                                             ; preds = %1010, %1009
  br i1 true, label %1031, label %1051, !dbg !20

1031:                                             ; preds = %1030
  %1032 = insertelement <4 x float> poison, float %868, i64 0, !dbg !18
  %1033 = insertelement <4 x float> %1032, float %874, i64 1, !dbg !18
  %1034 = insertelement <4 x float> %1033, float %869, i64 2, !dbg !18
  %1035 = insertelement <4 x float> %1034, float %875, i64 3, !dbg !18
  %row.reciprocal27 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %847, !dbg !18
  %normalized28 = fmul <4 x float> %1035, %row.reciprocal27, !dbg !18
  %1036 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !19
  %1037 = extractelement <4 x half> %1036, i64 3, !dbg !19
  %1038 = extractelement <4 x half> %1036, i64 2, !dbg !19
  %1039 = extractelement <4 x half> %1036, i64 1, !dbg !19
  %1040 = extractelement <4 x half> %1036, i64 0, !dbg !19
  %1041 = insertelement <4 x float> poison, float %761, i64 0, !dbg !18
  %1042 = insertelement <4 x float> %1041, float %767, i64 1, !dbg !18
  %1043 = insertelement <4 x float> %1042, float %762, i64 2, !dbg !18
  %1044 = insertelement <4 x float> %1043, float %768, i64 3, !dbg !18
  %normalized29 = fmul <4 x float> %1044, %row.reciprocal27, !dbg !18
  %1045 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !19
  %1046 = extractelement <4 x half> %1045, i64 3, !dbg !19
  %1047 = extractelement <4 x half> %1045, i64 2, !dbg !19
  %1048 = extractelement <4 x half> %1045, i64 1, !dbg !19
  %1049 = extractelement <4 x half> %1045, i64 0, !dbg !19
  %1050 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread285 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local286 = zext nneg i32 %address.thread285 to i64, !dbg !12
  %address.local287 = add i64 %28, %address.local286, !dbg !20
  %address.local288 = lshr i64 %address.local286, 6, !dbg !12
  %address.local289 = shl nuw nsw i64 %address.local288, 4, !dbg !20
  %address.local290 = add i64 %address.local287, %address.local289, !dbg !20
  %address.local291 = and i64 %address.local286, 240, !dbg !20
  %address.local292 = sub i64 %address.local290, %address.local291, !dbg !20
  %address.local293 = add i64 %address.local292, %29, !dbg !20
  %address.local294 = shl nuw nsw i64 %address.local293, 7, !dbg !20
  %address.local295 = add nuw nsw i64 %1050, %address.local294, !dbg !20
  %address.local296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !20
  %address.local297 = lshr i64 %address.local286, 2, !dbg !12
  %address.local298 = and i64 %address.local297, 12, !dbg !12
  %address.local299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local296, i64 %address.local298, !dbg !20
  %address.local300 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local299, i64 192, !dbg !20
  store half %1049, ptr addrspace(1) %address.local300, align 2, !dbg !20
  %address.local301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !20
  %address.local302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local301, i64 %address.local298, !dbg !20
  %address.local303 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local302, i64 194, !dbg !20
  store half %1048, ptr addrspace(1) %address.local303, align 2, !dbg !20
  %address.local304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !20
  %address.local305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local304, i64 %address.local298, !dbg !20
  %address.local306 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local305, i64 196, !dbg !20
  store half %1047, ptr addrspace(1) %address.local306, align 2, !dbg !20
  %address.local307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !20
  %address.local308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local307, i64 %address.local298, !dbg !20
  %address.local309 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local308, i64 198, !dbg !20
  store half %1046, ptr addrspace(1) %address.local309, align 2, !dbg !20
  %address.local310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !20
  %address.local311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local310, i64 %address.local298, !dbg !20
  %address.local312 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local311, i64 224, !dbg !20
  store half %1040, ptr addrspace(1) %address.local312, align 2, !dbg !20
  %address.local313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !20
  %address.local314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local313, i64 %address.local298, !dbg !20
  %address.local315 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local314, i64 226, !dbg !20
  store half %1039, ptr addrspace(1) %address.local315, align 2, !dbg !20
  %address.local316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !20
  %address.local317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local316, i64 %address.local298, !dbg !20
  %address.local318 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local317, i64 228, !dbg !20
  store half %1038, ptr addrspace(1) %address.local318, align 2, !dbg !20
  %address.local319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local295, !dbg !20
  %address.local320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local319, i64 %address.local298, !dbg !20
  %address.local321 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local320, i64 230, !dbg !20
  store half %1037, ptr addrspace(1) %address.local321, align 2, !dbg !20
  br label %1051, !dbg !20

1051:                                             ; preds = %1031, %1030
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/group8-reverse")
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
