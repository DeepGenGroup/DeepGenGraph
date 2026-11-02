; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/baseline/input.mlir"

@shm_0 = addrspace(3) global [8192 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !36 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %10, 6, !dbg !9
  %.lobit = ashr i32 %9, 31, !dbg !11
  %12 = xor i32 %9, %.lobit, !dbg !11
  %13 = sdiv i32 %12, 64, !dbg !11
  %.lobit1 = ashr i32 %9, 31, !dbg !11
  %14 = xor i32 %13, %.lobit1, !dbg !11
  %15 = sext i32 %14 to i64, !dbg !11
  %16 = add nsw i64 %8, %15, !dbg !11
  %17 = and i64 %16, 31, !dbg !11
  %18 = shl nsw i64 %10, 6, !dbg !11
  %19 = mul nsw i64 %15, -4096, !dbg !11
  %20 = add nsw i64 %18, %19, !dbg !11
  %21 = shl nuw nsw i64 %17, 19, !dbg !11
  %22 = shl nsw i64 %20, 7, !dbg !11
  %23 = add nsw i64 %21, %22, !dbg !11
  %24 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !38
  %25 = zext nneg i32 %24 to i64, !dbg !12
  %26 = lshr i64 %25, 6, !dbg !12
  %27 = or disjoint i64 %11, 63, !dbg !13
  %28 = add nsw i64 %11, 64, !dbg !13
  %29 = icmp uge i64 %28, %27, !dbg !13
  %30 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %31 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %32 = getelementptr i8, ptr addrspace(1) %31, i64 64, !dbg !11
  %33 = shl nsw i64 %8, 7, !dbg !15
  %34 = shl nsw i64 %6, 12, !dbg !15
  %35 = add nsw i64 %33, %34, !dbg !15
  %36 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %37 = getelementptr i8, ptr addrspace(1) %36, i64 128, !dbg !11
  %38 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %39 = getelementptr i8, ptr addrspace(1) %38, i64 192, !dbg !11
  br label %40, !dbg !7

40:                                               ; preds = %400, %4
  %41 = phi i64 [ %712, %400 ], [ 0, %4 ], !dbg !7
  %42 = phi <4 x float> [ %595, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %43 = phi <4 x float> [ %603, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %44 = phi <8 x float> [ %708, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %45 = phi <1 x float> [ %559, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %46 = phi <1 x float> [ %462, %400 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %47 = phi <4 x float> [ %635, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %48 = phi <4 x float> [ %643, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %49 = phi <8 x float> [ %709, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %50 = phi <4 x float> [ %675, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %51 = phi <4 x float> [ %683, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %52 = phi <8 x float> [ %710, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %53 = phi <4 x float> [ %699, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %54 = phi <4 x float> [ %707, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %55 = phi <8 x float> [ %711, %400 ], [ zeroinitializer, %4 ], !dbg !7
  %.not = icmp sgt i64 %41, %10, !dbg !7
  br i1 %.not, label %713, label %56, !dbg !7

56:                                               ; preds = %40
  %57 = shl nsw i64 %41, 6, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !11, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local41 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local42 = and i64 %address.local41, 1536, !dbg !11
  %address.local43 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local42, !dbg !11
  %address.local44 = ptrtoint ptr addrspace(3) %address.local43 to i64, !dbg !11
  %address.local45 = trunc i64 %address.local44 to i32, !dbg !11
  %58 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local45), !dbg !11
  %59 = zext i32 %58 to i64, !dbg !11
  %60 = inttoptr i64 %59 to ptr addrspace(3), !dbg !11
  %address.local46 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local47 = and i64 %address.local46, 8064, !dbg !11
  %address.local48 = lshr i64 %address.local, 3, !dbg !11
  %address.local49 = and i64 %address.local48, 24, !dbg !11
  %address.local50 = or disjoint i64 %address.local47, %address.local49, !dbg !11
  %address.local51 = getelementptr [2 x i8], ptr addrspace(1) %30, i64 %address.local50, !dbg !11
  %61 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %62 = ptrtoint ptr addrspace(1) %address.local51 to i64, !dbg !11
  %63 = sub i64 %62, %61, !dbg !11
  %dma.buffer.offset415 = trunc i64 %63 to i32, !dbg !11
  %64 = trunc i64 %61 to i32, !dbg !11
  %65 = insertelement <4 x i32> poison, i32 %64, i64 0, !dbg !11
  %66 = lshr i64 %61, 32, !dbg !11
  %67 = trunc i64 %66 to i32, !dbg !11
  %68 = insertelement <4 x i32> %65, i32 %67, i64 1, !dbg !11
  %69 = insertelement <4 x i32> %68, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc416 = insertelement <4 x i32> %69, i32 131072, i64 3, !dbg !11
  %70 = ptrtoint ptr addrspace(3) %60 to i32, !dbg !11
  %dma.buffer.lds417 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %70), !dbg !11
  %71 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds417, i32 %dma.buffer.offset415, <4 x i32> %dma.buffer.rsrc416) #10, !dbg !11
  %72 = lshr i64 %41, 13, !dbg !15
  %73 = add nsw i64 %72, %8, !dbg !15
  %74 = lshr i64 %41, 6, !dbg !15
  %75 = and i64 %74, 127, !dbg !15
  %76 = shl i64 %73, 19, !dbg !15
  %77 = and i64 %76, 16252928, !dbg !15
  %78 = shl i64 %74, 19, !dbg !15
  %79 = shl i64 %41, 13, !dbg !15
  %80 = sub i64 %79, %78, !dbg !15
  %81 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %77, !dbg !15
  %82 = getelementptr [2 x i8], ptr addrspace(1) %81, i64 %75, !dbg !15
  %83 = getelementptr [2 x i8], ptr addrspace(1) %82, i64 %80, !dbg !15
  %address.local52 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local42, !dbg !15
  %address.local53 = ptrtoint ptr addrspace(3) %address.local52 to i64, !dbg !15
  %address.local54 = trunc i64 %address.local53 to i32, !dbg !15
  %84 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local54), !dbg !15
  %85 = zext i32 %84 to i64, !dbg !15
  %86 = inttoptr i64 %85 to ptr addrspace(3), !dbg !15
  %address.local55 = getelementptr [2 x i8], ptr addrspace(1) %83, i64 %address.local50, !dbg !15
  %87 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %88 = ptrtoint ptr addrspace(1) %address.local55 to i64, !dbg !15
  %89 = sub i64 %88, %87, !dbg !15
  %dma.buffer.offset412 = trunc i64 %89 to i32, !dbg !15
  %90 = trunc i64 %87 to i32, !dbg !15
  %91 = insertelement <4 x i32> poison, i32 %90, i64 0, !dbg !15
  %92 = lshr i64 %87, 32, !dbg !15
  %93 = trunc i64 %92 to i32, !dbg !15
  %94 = insertelement <4 x i32> %91, i32 %93, i64 1, !dbg !15
  %95 = insertelement <4 x i32> %94, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc413 = insertelement <4 x i32> %95, i32 131072, i64 3, !dbg !15
  %96 = ptrtoint ptr addrspace(3) %86 to i32, !dbg !15
  %dma.buffer.lds414 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %96), !dbg !15
  %97 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds414, i32 %dma.buffer.offset412, <4 x i32> %dma.buffer.rsrc413) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local56 = lshr i32 %address.thread, 2, !dbg !12
  %address.local57 = and i32 %address.local56, 4, !dbg !12
  %address.local58 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local59 = and i32 %address.local58, 384, !dbg !12
  %address.local60 = or disjoint i32 %address.local57, %address.local59, !dbg !12
  %address.local61 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local62 = and i32 %address.local61, 120, !dbg !12
  %address.local63 = or disjoint i32 %address.local60, %address.local62, !dbg !12
  %address.local64 = shl nuw nsw i32 %address.thread, 4, !dbg !12
  %address.local65 = and i32 %address.local64, 512, !dbg !12
  %address.local66 = or disjoint i32 %address.local63, %address.local65, !dbg !12
  %address.local67 = zext nneg i32 %address.local66 to i64, !dbg !12
  %address.local68 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local67, !dbg !12
  %98 = load <4 x half>, ptr addrspace(3) %address.local68, align 2, !dbg !12
  %address.local69 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local67, !dbg !12
  %address.local70 = getelementptr i8, ptr addrspace(3) %address.local69, i64 2048, !dbg !12
  %99 = load <4 x half>, ptr addrspace(3) %address.local70, align 2, !dbg !12
  %100 = fmul <4 x half> %98, splat (half 0xH3015), !dbg !12
  %101 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local71 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local72 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local71, !dbg !16
  %address.local73 = lshr i64 %address.local, 2, !dbg !16
  %address.local74 = mul nsw i64 %address.local73, -512, !dbg !16
  %address.local75 = getelementptr [2 x i8], ptr addrspace(3) %address.local72, i64 %address.local74, !dbg !16
  %address.local76 = shl nuw nsw i64 %address.local, 1, !dbg !16
  %address.local77 = and i64 %address.local76, 24, !dbg !16
  %address.local78 = getelementptr [2 x i8], ptr addrspace(3) %address.local75, i64 %address.local77, !dbg !16
  %address.local79 = lshr i64 %address.local, 2, !dbg !12
  %address.local80 = and i64 %address.local79, 12, !dbg !12
  %address.local81 = getelementptr [2 x i8], ptr addrspace(3) %address.local78, i64 %address.local80, !dbg !16
  %address.local82 = and i64 %address.local, 63, !dbg !12
  %address.local83 = lshr i64 %address.local82, 5, !dbg !12
  %address.local84 = mul nuw nsw i64 %address.local83, 504, !dbg !12
  %address.local85 = getelementptr [2 x i8], ptr addrspace(3) %address.local81, i64 %address.local84, !dbg !16
  %102 = load <4 x half>, ptr addrspace(3) %address.local85, align 2, !dbg !16
  %address.local86 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local71, !dbg !16
  %address.local87 = getelementptr [2 x i8], ptr addrspace(3) %address.local86, i64 %address.local80, !dbg !16
  %address.local88 = getelementptr [2 x i8], ptr addrspace(3) %address.local87, i64 %address.local84, !dbg !16
  %address.local89 = getelementptr [2 x i8], ptr addrspace(3) %address.local88, i64 %address.local74, !dbg !16
  %address.local90 = getelementptr [2 x i8], ptr addrspace(3) %address.local89, i64 %address.local77, !dbg !16
  %address.local91 = getelementptr i8, ptr addrspace(3) %address.local90, i64 2048, !dbg !16
  %103 = load <4 x half>, ptr addrspace(3) %address.local91, align 2, !dbg !16
  %104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %102, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local92 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local71, !dbg !16
  %address.local93 = getelementptr [2 x i8], ptr addrspace(3) %address.local92, i64 %address.local74, !dbg !16
  %address.local94 = getelementptr [2 x i8], ptr addrspace(3) %address.local93, i64 %address.local77, !dbg !16
  %address.local95 = getelementptr [2 x i8], ptr addrspace(3) %address.local94, i64 %address.local80, !dbg !16
  %address.local96 = getelementptr [2 x i8], ptr addrspace(3) %address.local95, i64 %address.local84, !dbg !16
  %address.local97 = getelementptr i8, ptr addrspace(3) %address.local96, i64 64, !dbg !16
  %105 = load <4 x half>, ptr addrspace(3) %address.local97, align 2, !dbg !16
  %106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %103, <4 x float> %104) #9, !dbg !16
  %address.local98 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local71, !dbg !16
  %address.local99 = getelementptr [2 x i8], ptr addrspace(3) %address.local98, i64 %address.local80, !dbg !16
  %address.local100 = getelementptr [2 x i8], ptr addrspace(3) %address.local99, i64 %address.local84, !dbg !16
  %address.local101 = getelementptr [2 x i8], ptr addrspace(3) %address.local100, i64 %address.local74, !dbg !16
  %address.local102 = getelementptr [2 x i8], ptr addrspace(3) %address.local101, i64 %address.local77, !dbg !16
  %address.local103 = getelementptr i8, ptr addrspace(3) %address.local102, i64 2112, !dbg !16
  %107 = load <4 x half>, ptr addrspace(3) %address.local103, align 2, !dbg !16
  %address.local104 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local71, !dbg !16
  %address.local105 = getelementptr [2 x i8], ptr addrspace(3) %address.local104, i64 %address.local74, !dbg !16
  %address.local106 = getelementptr [2 x i8], ptr addrspace(3) %address.local105, i64 %address.local77, !dbg !16
  %address.local107 = getelementptr [2 x i8], ptr addrspace(3) %address.local106, i64 %address.local80, !dbg !16
  %address.local108 = getelementptr [2 x i8], ptr addrspace(3) %address.local107, i64 %address.local84, !dbg !16
  %address.local109 = getelementptr i8, ptr addrspace(3) %address.local108, i64 128, !dbg !16
  %108 = load <4 x half>, ptr addrspace(3) %address.local109, align 2, !dbg !16
  %109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %105, <4 x float> zeroinitializer) #9, !dbg !16
  %110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %107, <4 x float> %109) #9, !dbg !16
  %address.local110 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local71, !dbg !16
  %address.local111 = getelementptr [2 x i8], ptr addrspace(3) %address.local110, i64 %address.local80, !dbg !16
  %address.local112 = getelementptr [2 x i8], ptr addrspace(3) %address.local111, i64 %address.local84, !dbg !16
  %address.local113 = getelementptr [2 x i8], ptr addrspace(3) %address.local112, i64 %address.local74, !dbg !16
  %address.local114 = getelementptr [2 x i8], ptr addrspace(3) %address.local113, i64 %address.local77, !dbg !16
  %address.local115 = getelementptr i8, ptr addrspace(3) %address.local114, i64 2176, !dbg !16
  %111 = load <4 x half>, ptr addrspace(3) %address.local115, align 2, !dbg !16
  %112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %108, <4 x float> zeroinitializer) #9, !dbg !16
  %113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %111, <4 x float> %112) #9, !dbg !16
  %address.local116 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local71, !dbg !16
  %address.local117 = getelementptr [2 x i8], ptr addrspace(3) %address.local116, i64 %address.local74, !dbg !16
  %address.local118 = getelementptr [2 x i8], ptr addrspace(3) %address.local117, i64 %address.local77, !dbg !16
  %address.local119 = getelementptr [2 x i8], ptr addrspace(3) %address.local118, i64 %address.local80, !dbg !16
  %address.local120 = getelementptr [2 x i8], ptr addrspace(3) %address.local119, i64 %address.local84, !dbg !16
  %address.local121 = getelementptr i8, ptr addrspace(3) %address.local120, i64 192, !dbg !16
  %114 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !16
  %address.local122 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local71, !dbg !16
  %address.local123 = getelementptr [2 x i8], ptr addrspace(3) %address.local122, i64 %address.local80, !dbg !16
  %address.local124 = getelementptr [2 x i8], ptr addrspace(3) %address.local123, i64 %address.local84, !dbg !16
  %address.local125 = getelementptr [2 x i8], ptr addrspace(3) %address.local124, i64 %address.local74, !dbg !16
  %address.local126 = getelementptr [2 x i8], ptr addrspace(3) %address.local125, i64 %address.local77, !dbg !16
  %address.local127 = getelementptr i8, ptr addrspace(3) %address.local126, i64 2240, !dbg !16
  %115 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !16
  %116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %114, <4 x float> zeroinitializer) #9, !dbg !16
  %117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %115, <4 x float> %116) #9, !dbg !16
  %118 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local45), !dbg !11
  %119 = zext i32 %118 to i64, !dbg !11
  %120 = inttoptr i64 %119 to ptr addrspace(3), !dbg !11
  %address.local128 = getelementptr [2 x i8], ptr addrspace(1) %32, i64 %address.local50, !dbg !11
  %121 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %122 = ptrtoint ptr addrspace(1) %address.local128 to i64, !dbg !11
  %123 = sub i64 %122, %121, !dbg !11
  %dma.buffer.offset409 = trunc i64 %123 to i32, !dbg !11
  %124 = trunc i64 %121 to i32, !dbg !11
  %125 = insertelement <4 x i32> poison, i32 %124, i64 0, !dbg !11
  %126 = lshr i64 %121, 32, !dbg !11
  %127 = trunc i64 %126 to i32, !dbg !11
  %128 = insertelement <4 x i32> %125, i32 %127, i64 1, !dbg !11
  %129 = insertelement <4 x i32> %128, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc410 = insertelement <4 x i32> %129, i32 131072, i64 3, !dbg !11
  %130 = ptrtoint ptr addrspace(3) %120 to i32, !dbg !11
  %dma.buffer.lds411 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %130), !dbg !11
  %131 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds411, i32 %dma.buffer.offset409, <4 x i32> %dma.buffer.rsrc410) #10, !dbg !11
  %132 = add nuw i64 %41, 2048, !dbg !15
  %133 = icmp slt i64 %132, 0, !dbg !15
  %134 = sub nuw i64 -2049, %41, !dbg !15
  %135 = select i1 %133, i64 %134, i64 %132, !dbg !15
  %136 = sdiv i64 %135, 8192, !dbg !15
  %.lobit4 = ashr i64 %132, 63, !dbg !15
  %137 = xor i64 %136, %.lobit4, !dbg !15
  %138 = add nsw i64 %137, %8, !dbg !15
  %139 = add nsw i64 %74, %35, !dbg !15
  %140 = add nsw i64 %139, 32, !dbg !15
  %141 = icmp slt i64 %139, -32, !dbg !15
  %142 = sub nsw i64 -33, %139, !dbg !15
  %143 = select i1 %141, i64 %142, i64 %140, !dbg !15
  %144 = sdiv i64 %143, 128, !dbg !15
  %145 = sext i1 %141 to i64, !dbg !15
  %146 = xor i64 %144, %145, !dbg !15
  %147 = shl nsw i64 %146, 7, !dbg !15
  %148 = sub i64 %35, %147, !dbg !15
  %149 = shl i64 %138, 19, !dbg !15
  %150 = and i64 %149, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %151 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %150, !dbg !15
  %152 = getelementptr [2 x i8], ptr addrspace(1) %151, i64 %74, !dbg !15
  %153 = getelementptr [2 x i8], ptr addrspace(1) %152, i64 %148, !dbg !15
  %154 = getelementptr i8, ptr addrspace(1) %153, i64 64, !dbg !15
  %155 = getelementptr [2 x i8], ptr addrspace(1) %154, i64 %80, !dbg !15
  %156 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local54), !dbg !15
  %157 = zext i32 %156 to i64, !dbg !15
  %158 = inttoptr i64 %157 to ptr addrspace(3), !dbg !15
  %address.local129 = getelementptr [2 x i8], ptr addrspace(1) %155, i64 %address.local50, !dbg !15
  %159 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %160 = ptrtoint ptr addrspace(1) %address.local129 to i64, !dbg !15
  %161 = sub i64 %160, %159, !dbg !15
  %dma.buffer.offset406 = trunc i64 %161 to i32, !dbg !15
  %162 = trunc i64 %159 to i32, !dbg !15
  %163 = insertelement <4 x i32> poison, i32 %162, i64 0, !dbg !15
  %164 = lshr i64 %159, 32, !dbg !15
  %165 = trunc i64 %164 to i32, !dbg !15
  %166 = insertelement <4 x i32> %163, i32 %165, i64 1, !dbg !15
  %167 = insertelement <4 x i32> %166, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc407 = insertelement <4 x i32> %167, i32 131072, i64 3, !dbg !15
  %168 = ptrtoint ptr addrspace(3) %158 to i32, !dbg !15
  %dma.buffer.lds408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %168), !dbg !15
  %169 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds408, i32 %dma.buffer.offset406, <4 x i32> %dma.buffer.rsrc407) #10, !dbg !15
  %170 = load <4 x half>, ptr addrspace(3) %address.local68, align 2, !dbg !12
  %171 = load <4 x half>, ptr addrspace(3) %address.local70, align 2, !dbg !12
  %172 = fmul <4 x half> %170, splat (half 0xH3015), !dbg !12
  %173 = fmul <4 x half> %171, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %174 = load <4 x half>, ptr addrspace(3) %address.local85, align 2, !dbg !16
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %174, <4 x float> %106) #9, !dbg !16
  %176 = load <4 x half>, ptr addrspace(3) %address.local91, align 2, !dbg !16
  %177 = load <4 x half>, ptr addrspace(3) %address.local97, align 2, !dbg !16
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %176, <4 x float> %175) #9, !dbg !16
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %177, <4 x float> %110) #9, !dbg !16
  %180 = load <4 x half>, ptr addrspace(3) %address.local103, align 2, !dbg !16
  %181 = load <4 x half>, ptr addrspace(3) %address.local109, align 2, !dbg !16
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %180, <4 x float> %179) #9, !dbg !16
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %181, <4 x float> %113) #9, !dbg !16
  %184 = load <4 x half>, ptr addrspace(3) %address.local115, align 2, !dbg !16
  %185 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !16
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %184, <4 x float> %183) #9, !dbg !16
  %187 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %185, <4 x float> %117) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %187, <4 x float> %188) #9, !dbg !16
  %190 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local45), !dbg !11
  %191 = zext i32 %190 to i64, !dbg !11
  %192 = inttoptr i64 %191 to ptr addrspace(3), !dbg !11
  %address.local130 = getelementptr [2 x i8], ptr addrspace(1) %37, i64 %address.local50, !dbg !11
  %193 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %194 = ptrtoint ptr addrspace(1) %address.local130 to i64, !dbg !11
  %195 = sub i64 %194, %193, !dbg !11
  %dma.buffer.offset403 = trunc i64 %195 to i32, !dbg !11
  %196 = trunc i64 %193 to i32, !dbg !11
  %197 = insertelement <4 x i32> poison, i32 %196, i64 0, !dbg !11
  %198 = lshr i64 %193, 32, !dbg !11
  %199 = trunc i64 %198 to i32, !dbg !11
  %200 = insertelement <4 x i32> %197, i32 %199, i64 1, !dbg !11
  %201 = insertelement <4 x i32> %200, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc404 = insertelement <4 x i32> %201, i32 131072, i64 3, !dbg !11
  %202 = ptrtoint ptr addrspace(3) %192 to i32, !dbg !11
  %dma.buffer.lds405 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %202), !dbg !11
  %203 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds405, i32 %dma.buffer.offset403, <4 x i32> %dma.buffer.rsrc404) #10, !dbg !11
  %204 = add nuw i64 %41, 4096, !dbg !15
  %205 = icmp slt i64 %204, 0, !dbg !15
  %206 = sub nuw i64 -4097, %41, !dbg !15
  %207 = select i1 %205, i64 %206, i64 %204, !dbg !15
  %208 = sdiv i64 %207, 8192, !dbg !15
  %.lobit5 = ashr i64 %204, 63, !dbg !15
  %209 = xor i64 %208, %.lobit5, !dbg !15
  %210 = add nsw i64 %209, %8, !dbg !15
  %211 = add nsw i64 %139, 64, !dbg !15
  %212 = icmp slt i64 %139, -64, !dbg !15
  %213 = sub nsw i64 -65, %139, !dbg !15
  %214 = select i1 %212, i64 %213, i64 %211, !dbg !15
  %215 = sdiv i64 %214, 128, !dbg !15
  %216 = sext i1 %212 to i64, !dbg !15
  %217 = xor i64 %215, %216, !dbg !15
  %218 = shl nsw i64 %217, 7, !dbg !15
  %219 = sub i64 %35, %218, !dbg !15
  %220 = shl i64 %210, 19, !dbg !15
  %221 = and i64 %220, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %222 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %221, !dbg !15
  %223 = getelementptr [2 x i8], ptr addrspace(1) %222, i64 %74, !dbg !15
  %224 = getelementptr [2 x i8], ptr addrspace(1) %223, i64 %219, !dbg !15
  %225 = getelementptr i8, ptr addrspace(1) %224, i64 128, !dbg !15
  %226 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %80, !dbg !15
  %227 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local54), !dbg !15
  %228 = zext i32 %227 to i64, !dbg !15
  %229 = inttoptr i64 %228 to ptr addrspace(3), !dbg !15
  %address.local131 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local50, !dbg !15
  %230 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %231 = ptrtoint ptr addrspace(1) %address.local131 to i64, !dbg !15
  %232 = sub i64 %231, %230, !dbg !15
  %dma.buffer.offset400 = trunc i64 %232 to i32, !dbg !15
  %233 = trunc i64 %230 to i32, !dbg !15
  %234 = insertelement <4 x i32> poison, i32 %233, i64 0, !dbg !15
  %235 = lshr i64 %230, 32, !dbg !15
  %236 = trunc i64 %235 to i32, !dbg !15
  %237 = insertelement <4 x i32> %234, i32 %236, i64 1, !dbg !15
  %238 = insertelement <4 x i32> %237, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc401 = insertelement <4 x i32> %238, i32 131072, i64 3, !dbg !15
  %239 = ptrtoint ptr addrspace(3) %229 to i32, !dbg !15
  %dma.buffer.lds402 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %239), !dbg !15
  %240 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds402, i32 %dma.buffer.offset400, <4 x i32> %dma.buffer.rsrc401) #10, !dbg !15
  %241 = load <4 x half>, ptr addrspace(3) %address.local68, align 2, !dbg !12
  %242 = load <4 x half>, ptr addrspace(3) %address.local70, align 2, !dbg !12
  %243 = fmul <4 x half> %241, splat (half 0xH3015), !dbg !12
  %244 = fmul <4 x half> %242, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %245 = load <4 x half>, ptr addrspace(3) %address.local85, align 2, !dbg !16
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %245, <4 x float> %178) #9, !dbg !16
  %247 = load <4 x half>, ptr addrspace(3) %address.local91, align 2, !dbg !16
  %248 = load <4 x half>, ptr addrspace(3) %address.local97, align 2, !dbg !16
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %244, <4 x half> %247, <4 x float> %246) #9, !dbg !16
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %248, <4 x float> %182) #9, !dbg !16
  %251 = load <4 x half>, ptr addrspace(3) %address.local103, align 2, !dbg !16
  %252 = load <4 x half>, ptr addrspace(3) %address.local109, align 2, !dbg !16
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %244, <4 x half> %251, <4 x float> %250) #9, !dbg !16
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %252, <4 x float> %186) #9, !dbg !16
  %255 = load <4 x half>, ptr addrspace(3) %address.local115, align 2, !dbg !16
  %256 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !16
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %244, <4 x half> %255, <4 x float> %254) #9, !dbg !16
  %258 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %243, <4 x half> %256, <4 x float> %189) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %244, <4 x half> %258, <4 x float> %259) #9, !dbg !16
  %261 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local45), !dbg !11
  %262 = zext i32 %261 to i64, !dbg !11
  %263 = inttoptr i64 %262 to ptr addrspace(3), !dbg !11
  %address.local132 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %address.local50, !dbg !11
  %264 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %265 = ptrtoint ptr addrspace(1) %address.local132 to i64, !dbg !11
  %266 = sub i64 %265, %264, !dbg !11
  %dma.buffer.offset397 = trunc i64 %266 to i32, !dbg !11
  %267 = trunc i64 %264 to i32, !dbg !11
  %268 = insertelement <4 x i32> poison, i32 %267, i64 0, !dbg !11
  %269 = lshr i64 %264, 32, !dbg !11
  %270 = trunc i64 %269 to i32, !dbg !11
  %271 = insertelement <4 x i32> %268, i32 %270, i64 1, !dbg !11
  %272 = insertelement <4 x i32> %271, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc398 = insertelement <4 x i32> %272, i32 131072, i64 3, !dbg !11
  %273 = ptrtoint ptr addrspace(3) %263 to i32, !dbg !11
  %dma.buffer.lds399 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %273), !dbg !11
  %274 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds399, i32 %dma.buffer.offset397, <4 x i32> %dma.buffer.rsrc398) #10, !dbg !11
  %275 = add nuw i64 %41, 6144, !dbg !15
  %276 = icmp slt i64 %275, 0, !dbg !15
  %277 = sub nuw i64 -6145, %41, !dbg !15
  %278 = select i1 %276, i64 %277, i64 %275, !dbg !15
  %279 = sdiv i64 %278, 8192, !dbg !15
  %.lobit6 = ashr i64 %275, 63, !dbg !15
  %280 = xor i64 %279, %.lobit6, !dbg !15
  %281 = add nsw i64 %280, %8, !dbg !15
  %282 = add nsw i64 %139, 96, !dbg !15
  %283 = icmp slt i64 %139, -96, !dbg !15
  %284 = sub nsw i64 -97, %139, !dbg !15
  %285 = select i1 %283, i64 %284, i64 %282, !dbg !15
  %286 = sdiv i64 %285, 128, !dbg !15
  %287 = sext i1 %283 to i64, !dbg !15
  %288 = xor i64 %286, %287, !dbg !15
  %289 = shl nsw i64 %288, 7, !dbg !15
  %290 = sub i64 %35, %289, !dbg !15
  %291 = shl i64 %281, 19, !dbg !15
  %292 = and i64 %291, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %293 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %292, !dbg !15
  %294 = getelementptr [2 x i8], ptr addrspace(1) %293, i64 %74, !dbg !15
  %295 = getelementptr [2 x i8], ptr addrspace(1) %294, i64 %290, !dbg !15
  %296 = getelementptr i8, ptr addrspace(1) %295, i64 192, !dbg !15
  %297 = getelementptr [2 x i8], ptr addrspace(1) %296, i64 %80, !dbg !15
  %298 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local54), !dbg !15
  %299 = zext i32 %298 to i64, !dbg !15
  %300 = inttoptr i64 %299 to ptr addrspace(3), !dbg !15
  %address.local133 = getelementptr [2 x i8], ptr addrspace(1) %297, i64 %address.local50, !dbg !15
  %301 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %302 = ptrtoint ptr addrspace(1) %address.local133 to i64, !dbg !15
  %303 = sub i64 %302, %301, !dbg !15
  %dma.buffer.offset394 = trunc i64 %303 to i32, !dbg !15
  %304 = trunc i64 %301 to i32, !dbg !15
  %305 = insertelement <4 x i32> poison, i32 %304, i64 0, !dbg !15
  %306 = lshr i64 %301, 32, !dbg !15
  %307 = trunc i64 %306 to i32, !dbg !15
  %308 = insertelement <4 x i32> %305, i32 %307, i64 1, !dbg !15
  %309 = insertelement <4 x i32> %308, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc395 = insertelement <4 x i32> %309, i32 131072, i64 3, !dbg !15
  %310 = ptrtoint ptr addrspace(3) %300 to i32, !dbg !15
  %dma.buffer.lds396 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %310), !dbg !15
  %311 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds396, i32 %dma.buffer.offset394, <4 x i32> %dma.buffer.rsrc395) #10, !dbg !15
  %312 = load <4 x half>, ptr addrspace(3) %address.local68, align 2, !dbg !12
  %313 = load <4 x half>, ptr addrspace(3) %address.local70, align 2, !dbg !12
  %314 = fmul <4 x half> %312, splat (half 0xH3015), !dbg !12
  %315 = fmul <4 x half> %313, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %316 = add nsw i64 %74, %8, !dbg !17
  %317 = shl i64 %316, 19, !dbg !17
  %318 = and i64 %317, 16252928, !dbg !17
  %319 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %318, !dbg !17
  %320 = getelementptr [2 x i8], ptr addrspace(1) %319, i64 %80, !dbg !17
  %321 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local45), !dbg !17
  %322 = zext i32 %321 to i64, !dbg !17
  %323 = inttoptr i64 %322 to ptr addrspace(3), !dbg !17
  %address.local134 = shl nuw nsw i64 %address.local, 5, !dbg !17
  %address.local135 = and i64 %address.local134, 8064, !dbg !17
  %address.local136 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local137 = and i64 %address.local136, 24, !dbg !17
  %address.local138 = or disjoint i64 %address.local135, %address.local137, !dbg !17
  %address.local139 = getelementptr [2 x i8], ptr addrspace(1) %320, i64 %address.local138, !dbg !17
  %324 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %325 = ptrtoint ptr addrspace(1) %address.local139 to i64, !dbg !17
  %326 = sub i64 %325, %324, !dbg !17
  %dma.buffer.offset391 = trunc i64 %326 to i32, !dbg !17
  %327 = trunc i64 %324 to i32, !dbg !17
  %328 = insertelement <4 x i32> poison, i32 %327, i64 0, !dbg !17
  %329 = lshr i64 %324, 32, !dbg !17
  %330 = trunc i64 %329 to i32, !dbg !17
  %331 = insertelement <4 x i32> %328, i32 %330, i64 1, !dbg !17
  %332 = insertelement <4 x i32> %331, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc392 = insertelement <4 x i32> %332, i32 131072, i64 3, !dbg !17
  %333 = ptrtoint ptr addrspace(3) %323 to i32, !dbg !17
  %dma.buffer.lds393 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %333), !dbg !17
  %334 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds393, i32 %dma.buffer.offset391, <4 x i32> %dma.buffer.rsrc392) #10, !dbg !17
  %335 = load <4 x half>, ptr addrspace(3) %address.local85, align 2, !dbg !16
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %335, <4 x float> %249) #9, !dbg !16
  %337 = load <4 x half>, ptr addrspace(3) %address.local91, align 2, !dbg !16
  %338 = load <4 x half>, ptr addrspace(3) %address.local97, align 2, !dbg !16
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %337, <4 x float> %336) #9, !dbg !16
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %338, <4 x float> %253) #9, !dbg !16
  %341 = load <4 x half>, ptr addrspace(3) %address.local103, align 2, !dbg !16
  %342 = load <4 x half>, ptr addrspace(3) %address.local109, align 2, !dbg !16
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %341, <4 x float> %340) #9, !dbg !16
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %342, <4 x float> %257) #9, !dbg !16
  %345 = load <4 x half>, ptr addrspace(3) %address.local115, align 2, !dbg !16
  %346 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !16
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %345, <4 x float> %344) #9, !dbg !16
  %348 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %314, <4 x half> %346, <4 x float> %260) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %350 = or disjoint i64 %57, 63, !dbg !13
  %351 = icmp uge i64 %11, %350, !dbg !13
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %315, <4 x half> %348, <4 x float> %349) #9, !dbg !16
  %353 = and i1 %29, %351, !dbg !13
  br i1 %353, label %354, label %355, !dbg !13

354:                                              ; preds = %56
  br label %398, !dbg !13

355:                                              ; preds = %56
  %356 = add nsw i64 %11, %25, !dbg !13
  %357 = shl nuw nsw i64 %26, 4, !dbg !13
  %358 = add nsw i64 %356, %357, !dbg !13
  %359 = and i32 %24, 240, !dbg !13
  %index.compact = zext nneg i32 %359 to i64, !dbg !13
  %360 = sub nsw i64 %358, %index.compact, !dbg !13
  %361 = add nsw i64 %360, 1, !dbg !22
  %362 = and i64 %25, 48, !dbg !13
  %363 = or disjoint i64 %57, %362, !dbg !13
  %364 = lshr i64 %361, 32, !dbg !23
  %compare.high = trunc i64 %364 to i32, !dbg !23
  %365 = lshr i64 %363, 32, !dbg !23
  %compare.high325 = trunc i64 %365 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high325, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high325, !dbg !23
  %compare.low.base = trunc i64 %361 to i32, !dbg !23
  %compare.low.base326 = trunc i64 %363 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base326, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %366 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low327 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result328 = select i1 %compare.high.equal, i1 %compare.low327, i1 %compare.high.order, !dbg !23
  %367 = select i1 %compare.result328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local330 = or i32 %compare.local.base329, 2, !dbg !23
  %compare.low331 = icmp ugt i32 %compare.low.base, %compare.local330, !dbg !23
  %compare.result332 = select i1 %compare.high.equal, i1 %compare.low331, i1 %compare.high.order, !dbg !23
  %368 = select i1 %compare.result332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local334 = or i32 %compare.local.base333, 3, !dbg !23
  %compare.low335 = icmp ugt i32 %compare.low.base, %compare.local334, !dbg !23
  %compare.result336 = select i1 %compare.high.equal, i1 %compare.low335, i1 %compare.high.order, !dbg !23
  %369 = select i1 %compare.result336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local338 = or i32 %compare.local.base337, 4, !dbg !23
  %compare.low339 = icmp ugt i32 %compare.low.base, %compare.local338, !dbg !23
  %compare.result340 = select i1 %compare.high.equal, i1 %compare.low339, i1 %compare.high.order, !dbg !23
  %370 = select i1 %compare.result340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local342 = or i32 %compare.local.base341, 5, !dbg !23
  %compare.low343 = icmp ugt i32 %compare.low.base, %compare.local342, !dbg !23
  %compare.result344 = select i1 %compare.high.equal, i1 %compare.low343, i1 %compare.high.order, !dbg !23
  %371 = select i1 %compare.result344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local346 = or i32 %compare.local.base345, 6, !dbg !23
  %compare.low347 = icmp ugt i32 %compare.low.base, %compare.local346, !dbg !23
  %compare.result348 = select i1 %compare.high.equal, i1 %compare.low347, i1 %compare.high.order, !dbg !23
  %372 = select i1 %compare.result348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local350 = or i32 %compare.local.base349, 7, !dbg !23
  %compare.low351 = icmp ugt i32 %compare.low.base, %compare.local350, !dbg !23
  %compare.result352 = select i1 %compare.high.equal, i1 %compare.low351, i1 %compare.high.order, !dbg !23
  %373 = select i1 %compare.result352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local354 = or i32 %compare.local.base353, 8, !dbg !23
  %compare.low355 = icmp ugt i32 %compare.low.base, %compare.local354, !dbg !23
  %compare.result356 = select i1 %compare.high.equal, i1 %compare.low355, i1 %compare.high.order, !dbg !23
  %374 = select i1 %compare.result356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local358 = or i32 %compare.local.base357, 9, !dbg !23
  %compare.low359 = icmp ugt i32 %compare.low.base, %compare.local358, !dbg !23
  %compare.result360 = select i1 %compare.high.equal, i1 %compare.low359, i1 %compare.high.order, !dbg !23
  %375 = select i1 %compare.result360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local362 = or i32 %compare.local.base361, 10, !dbg !23
  %compare.low363 = icmp ugt i32 %compare.low.base, %compare.local362, !dbg !23
  %compare.result364 = select i1 %compare.high.equal, i1 %compare.low363, i1 %compare.high.order, !dbg !23
  %376 = select i1 %compare.result364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local366 = or i32 %compare.local.base365, 11, !dbg !23
  %compare.low367 = icmp ugt i32 %compare.low.base, %compare.local366, !dbg !23
  %compare.result368 = select i1 %compare.high.equal, i1 %compare.low367, i1 %compare.high.order, !dbg !23
  %377 = select i1 %compare.result368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local370 = or i32 %compare.local.base369, 12, !dbg !23
  %compare.low371 = icmp ugt i32 %compare.low.base, %compare.local370, !dbg !23
  %compare.result372 = select i1 %compare.high.equal, i1 %compare.low371, i1 %compare.high.order, !dbg !23
  %378 = select i1 %compare.result372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local374 = or i32 %compare.local.base373, 13, !dbg !23
  %compare.low375 = icmp ugt i32 %compare.low.base, %compare.local374, !dbg !23
  %compare.result376 = select i1 %compare.high.equal, i1 %compare.low375, i1 %compare.high.order, !dbg !23
  %379 = select i1 %compare.result376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local378 = or i32 %compare.local.base377, 14, !dbg !23
  %compare.low379 = icmp ugt i32 %compare.low.base, %compare.local378, !dbg !23
  %compare.result380 = select i1 %compare.high.equal, i1 %compare.low379, i1 %compare.high.order, !dbg !23
  %380 = select i1 %compare.result380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base326, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local382 = or i32 %compare.local.base381, 15, !dbg !23
  %compare.low383 = icmp ugt i32 %compare.low.base, %compare.local382, !dbg !23
  %compare.result384 = select i1 %compare.high.equal, i1 %compare.low383, i1 %compare.high.order, !dbg !23
  %381 = select i1 %compare.result384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %382 = insertelement <16 x float> poison, float %366, i64 0, !dbg !13
  %383 = insertelement <16 x float> %382, float %367, i64 1, !dbg !13
  %384 = insertelement <16 x float> %383, float %368, i64 2, !dbg !13
  %385 = insertelement <16 x float> %384, float %369, i64 3, !dbg !13
  %386 = insertelement <16 x float> %385, float %370, i64 4, !dbg !13
  %387 = insertelement <16 x float> %386, float %371, i64 5, !dbg !13
  %388 = insertelement <16 x float> %387, float %372, i64 6, !dbg !13
  %389 = insertelement <16 x float> %388, float %373, i64 7, !dbg !13
  %390 = insertelement <16 x float> %389, float %374, i64 8, !dbg !13
  %391 = insertelement <16 x float> %390, float %375, i64 9, !dbg !13
  %392 = insertelement <16 x float> %391, float %376, i64 10, !dbg !13
  %393 = insertelement <16 x float> %392, float %377, i64 11, !dbg !13
  %394 = insertelement <16 x float> %393, float %378, i64 12, !dbg !13
  %395 = insertelement <16 x float> %394, float %379, i64 13, !dbg !13
  %396 = insertelement <16 x float> %395, float %380, i64 14, !dbg !13
  %397 = insertelement <16 x float> %396, float %381, i64 15, !dbg !13
  br label %398, !dbg !13

398:                                              ; preds = %354, %355
  %399 = phi <16 x float> [ %397, %355 ], [ zeroinitializer, %354 ], !dbg !13
  br label %400, !dbg !13

400:                                              ; preds = %398
  %401 = shufflevector <16 x float> %399, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %402 = fadd <4 x float> %339, %401, !dbg !25
  %403 = extractelement <4 x float> %402, i64 0, !dbg !25
  %404 = extractelement <4 x float> %402, i64 1, !dbg !25
  %405 = extractelement <4 x float> %402, i64 2, !dbg !25
  %406 = extractelement <4 x float> %402, i64 3, !dbg !25
  %407 = shufflevector <16 x float> %399, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %408 = fadd <4 x float> %343, %407, !dbg !25
  %409 = extractelement <4 x float> %408, i64 0, !dbg !25
  %410 = extractelement <4 x float> %408, i64 1, !dbg !25
  %411 = extractelement <4 x float> %408, i64 2, !dbg !25
  %412 = extractelement <4 x float> %408, i64 3, !dbg !25
  %413 = shufflevector <16 x float> %399, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %414 = fadd <4 x float> %347, %413, !dbg !25
  %415 = extractelement <4 x float> %414, i64 0, !dbg !25
  %416 = extractelement <4 x float> %414, i64 1, !dbg !25
  %417 = extractelement <4 x float> %414, i64 2, !dbg !25
  %418 = extractelement <4 x float> %414, i64 3, !dbg !25
  %419 = shufflevector <16 x float> %399, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %420 = fadd <4 x float> %352, %419, !dbg !25
  %421 = extractelement <4 x float> %420, i64 0, !dbg !25
  %422 = extractelement <4 x float> %420, i64 1, !dbg !25
  %423 = extractelement <4 x float> %420, i64 2, !dbg !25
  %424 = extractelement <4 x float> %420, i64 3, !dbg !25
  %425 = call float @llvm.maxnum.f32(float %403, float 0xFFF0000000000000), !dbg !26
  %426 = call float @llvm.maxnum.f32(float %425, float %404), !dbg !26
  %427 = call float @llvm.maxnum.f32(float %426, float %405), !dbg !26
  %428 = call float @llvm.maxnum.f32(float %427, float %406), !dbg !26
  %429 = call float @llvm.maxnum.f32(float %428, float %409), !dbg !26
  %430 = call float @llvm.maxnum.f32(float %429, float %410), !dbg !26
  %431 = call float @llvm.maxnum.f32(float %430, float %411), !dbg !26
  %432 = call float @llvm.maxnum.f32(float %431, float %412), !dbg !26
  %433 = call float @llvm.maxnum.f32(float %432, float %415), !dbg !26
  %434 = call float @llvm.maxnum.f32(float %433, float %416), !dbg !26
  %435 = call float @llvm.maxnum.f32(float %434, float %417), !dbg !26
  %436 = call float @llvm.maxnum.f32(float %435, float %418), !dbg !26
  %437 = call float @llvm.maxnum.f32(float %436, float %421), !dbg !26
  %438 = call float @llvm.maxnum.f32(float %437, float %422), !dbg !26
  %439 = call float @llvm.maxnum.f32(float %438, float %423), !dbg !26
  %440 = call float @llvm.maxnum.f32(float %439, float %424), !dbg !26
  %441 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %442 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %441), !dbg !26
  %443 = add nuw nsw i32 %442, 64, !dbg !26
  %444 = xor i32 %442, 16, !dbg !26
  %445 = and i32 %443, 192, !dbg !26
  %446 = icmp samesign ult i32 %444, %445, !dbg !26
  %447 = select i1 %446, i32 %444, i32 %442, !dbg !26
  %448 = shl nuw nsw i32 %447, 2, !dbg !26
  %449 = bitcast float %440 to i32, !dbg !26
  %450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %448, i32 %449), !dbg !26
  %451 = bitcast i32 %450 to float, !dbg !26
  %452 = call float @llvm.maxnum.f32(float %440, float %451), !dbg !26
  %453 = xor i32 %442, 32, !dbg !26
  %454 = icmp samesign ult i32 %453, %445, !dbg !26
  %455 = select i1 %454, i32 %453, i32 %442, !dbg !26
  %456 = shl nuw nsw i32 %455, 2, !dbg !26
  %457 = bitcast float %452 to i32, !dbg !26
  %458 = call i32 @llvm.amdgcn.ds.bpermute(i32 %456, i32 %457), !dbg !26
  %459 = bitcast i32 %458 to float, !dbg !26
  %460 = call float @llvm.maxnum.f32(float %452, float %459), !dbg !26
  %461 = insertelement <1 x float> poison, float %460, i64 0, !dbg !26
  %462 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %461, <1 x float> %46), !dbg !26
  %463 = shufflevector <1 x float> %462, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %464 = fsub <4 x float> %402, %463, !dbg !27
  %465 = fsub <4 x float> %408, %463, !dbg !27
  %466 = fsub <4 x float> %414, %463, !dbg !27
  %467 = fsub <4 x float> %420, %463, !dbg !27
  %468 = fsub <1 x float> %46, %462, !dbg !28
  %469 = extractelement <1 x float> %468, i64 0, !dbg !29
  %470 = call float @llvm.amdgcn.exp2.f32(float %469), !dbg !29
  %471 = insertelement <4 x float> poison, float %470, i64 0, !dbg !34
  %472 = shufflevector <4 x float> %471, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %473 = fmul <4 x float> %54, %472, !dbg !34
  %474 = fmul <4 x float> %53, %472, !dbg !34
  %475 = fmul <4 x float> %51, %472, !dbg !34
  %476 = fmul <4 x float> %50, %472, !dbg !34
  %477 = fmul <4 x float> %48, %472, !dbg !34
  %478 = fmul <4 x float> %47, %472, !dbg !34
  %479 = fmul <4 x float> %43, %472, !dbg !34
  %480 = fmul <4 x float> %42, %472, !dbg !34
  %481 = extractelement <1 x float> %45, i64 0, !dbg !30
  %482 = fmul float %481, %470, !dbg !30
  %483 = extractelement <4 x float> %464, i64 0, !dbg !31
  %484 = call float @llvm.amdgcn.exp2.f32(float %483), !dbg !31
  %485 = insertelement <4 x float> poison, float %484, i64 0, !dbg !33
  %486 = extractelement <4 x float> %464, i64 1, !dbg !31
  %487 = call float @llvm.amdgcn.exp2.f32(float %486), !dbg !31
  %488 = insertelement <4 x float> %485, float %487, i64 1, !dbg !33
  %489 = fadd float %484, %487, !dbg !32
  %490 = extractelement <4 x float> %464, i64 2, !dbg !31
  %491 = call float @llvm.amdgcn.exp2.f32(float %490), !dbg !31
  %492 = insertelement <4 x float> %488, float %491, i64 2, !dbg !33
  %493 = fadd float %489, %491, !dbg !32
  %494 = extractelement <4 x float> %464, i64 3, !dbg !31
  %495 = call float @llvm.amdgcn.exp2.f32(float %494), !dbg !31
  %496 = insertelement <4 x float> %492, float %495, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %496) #9, !dbg !33, !frisk.register_lifetimes !1
  %497 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %497) #9, !dbg !32, !frisk.register_lifetimes !1
  %498 = fadd float %493, %495, !dbg !32
  %499 = extractelement <4 x float> %465, i64 0, !dbg !31
  %500 = call float @llvm.amdgcn.exp2.f32(float %499), !dbg !31
  %501 = insertelement <4 x float> poison, float %500, i64 0, !dbg !33
  %502 = fadd float %498, %500, !dbg !32
  %503 = extractelement <4 x float> %465, i64 1, !dbg !31
  %504 = call float @llvm.amdgcn.exp2.f32(float %503), !dbg !31
  %505 = insertelement <4 x float> %501, float %504, i64 1, !dbg !33
  %506 = fadd float %502, %504, !dbg !32
  %507 = extractelement <4 x float> %465, i64 2, !dbg !31
  %508 = call float @llvm.amdgcn.exp2.f32(float %507), !dbg !31
  %509 = insertelement <4 x float> %505, float %508, i64 2, !dbg !33
  %510 = fadd float %506, %508, !dbg !32
  %511 = extractelement <4 x float> %465, i64 3, !dbg !31
  %512 = call float @llvm.amdgcn.exp2.f32(float %511), !dbg !31
  %513 = insertelement <4 x float> %509, float %512, i64 3, !dbg !33
  %math.result35 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %513) #9, !dbg !33, !frisk.register_lifetimes !1
  %514 = fptrunc <4 x float> %math.result35 to <4 x half>, !dbg !33
  %math.packed36 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %514) #9, !dbg !32, !frisk.register_lifetimes !1
  %515 = fadd float %510, %512, !dbg !32
  %516 = extractelement <4 x float> %466, i64 0, !dbg !31
  %517 = call float @llvm.amdgcn.exp2.f32(float %516), !dbg !31
  %518 = insertelement <4 x float> poison, float %517, i64 0, !dbg !33
  %519 = fadd float %515, %517, !dbg !32
  %520 = extractelement <4 x float> %466, i64 1, !dbg !31
  %521 = call float @llvm.amdgcn.exp2.f32(float %520), !dbg !31
  %522 = insertelement <4 x float> %518, float %521, i64 1, !dbg !33
  %523 = fadd float %519, %521, !dbg !32
  %524 = extractelement <4 x float> %466, i64 2, !dbg !31
  %525 = call float @llvm.amdgcn.exp2.f32(float %524), !dbg !31
  %526 = insertelement <4 x float> %522, float %525, i64 2, !dbg !33
  %527 = fadd float %523, %525, !dbg !32
  %528 = extractelement <4 x float> %466, i64 3, !dbg !31
  %529 = call float @llvm.amdgcn.exp2.f32(float %528), !dbg !31
  %530 = insertelement <4 x float> %526, float %529, i64 3, !dbg !33
  %math.result37 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %530) #9, !dbg !33, !frisk.register_lifetimes !1
  %531 = fptrunc <4 x float> %math.result37 to <4 x half>, !dbg !33
  %math.packed38 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %531) #9, !dbg !32, !frisk.register_lifetimes !1
  %532 = fadd float %527, %529, !dbg !32
  %533 = extractelement <4 x float> %467, i64 0, !dbg !31
  %534 = call float @llvm.amdgcn.exp2.f32(float %533), !dbg !31
  %535 = insertelement <4 x float> poison, float %534, i64 0, !dbg !33
  %536 = fadd float %532, %534, !dbg !32
  %537 = extractelement <4 x float> %467, i64 1, !dbg !31
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !31
  %539 = insertelement <4 x float> %535, float %538, i64 1, !dbg !33
  %540 = fadd float %536, %538, !dbg !32
  %541 = extractelement <4 x float> %467, i64 2, !dbg !31
  %542 = call float @llvm.amdgcn.exp2.f32(float %541), !dbg !31
  %543 = insertelement <4 x float> %539, float %542, i64 2, !dbg !33
  %544 = fadd float %540, %542, !dbg !32
  %545 = extractelement <4 x float> %467, i64 3, !dbg !31
  %546 = call float @llvm.amdgcn.exp2.f32(float %545), !dbg !31
  %547 = insertelement <4 x float> %543, float %546, i64 3, !dbg !33
  %math.result39 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %547) #9, !dbg !33, !frisk.register_lifetimes !1
  %548 = fptrunc <4 x float> %math.result39 to <4 x half>, !dbg !33
  %math.packed40 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %548) #9, !dbg !32, !frisk.register_lifetimes !1
  %549 = fadd float %544, %546, !dbg !32
  %550 = bitcast float %549 to i32, !dbg !32
  %551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %448, i32 %550), !dbg !32
  %552 = bitcast i32 %551 to float, !dbg !32
  %553 = fadd float %549, %552, !dbg !32
  %554 = bitcast float %553 to i32, !dbg !32
  %555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %456, i32 %554), !dbg !32
  %556 = bitcast i32 %555 to float, !dbg !32
  %557 = fadd float %553, %556, !dbg !32
  %558 = fadd float %557, %482, !dbg !32
  %559 = insertelement <1 x float> poison, float %558, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread140 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local141 = zext nneg i32 %address.thread140 to i64, !dbg !12
  %address.local142 = and i64 %address.local141, 60, !dbg !14
  %address.local143 = add nuw nsw i64 %address.local142, %address.local141, !dbg !14
  %address.local144 = shl nuw nsw i64 %address.local143, 3, !dbg !14
  %address.local145 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local144, !dbg !14
  %address.local146 = and i64 %address.local141, 63, !dbg !12
  %address.local147 = lshr i64 %address.local146, 4, !dbg !14
  %address.local148 = mul nuw nsw i64 %address.local147, 384, !dbg !14
  %address.local149 = getelementptr [2 x i8], ptr addrspace(3) %address.local145, i64 %address.local148, !dbg !14
  %address.local150 = lshr i64 %address.local141, 2, !dbg !16
  %address.local151 = mul nsw i64 %address.local150, -32, !dbg !14
  %address.local152 = getelementptr [2 x i8], ptr addrspace(3) %address.local149, i64 %address.local151, !dbg !14
  %560 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local152, i32 0) #0, !dbg !14
  %561 = bitcast <4 x i32> %560 to <8 x half>, !dbg !14
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local144, !dbg !14
  %address.local154 = getelementptr [2 x i8], ptr addrspace(3) %address.local153, i64 %address.local148, !dbg !14
  %address.local155 = getelementptr [2 x i8], ptr addrspace(3) %address.local154, i64 %address.local151, !dbg !14
  %address.local156 = getelementptr i8, ptr addrspace(3) %address.local155, i64 256, !dbg !14
  %562 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local156, i32 0) #0, !dbg !14
  %563 = bitcast <4 x i32> %562 to <8 x half>, !dbg !14
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local144, !dbg !14
  %address.local158 = getelementptr [2 x i8], ptr addrspace(3) %address.local157, i64 %address.local148, !dbg !14
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) %address.local158, i64 %address.local151, !dbg !14
  %address.local160 = getelementptr i8, ptr addrspace(3) %address.local159, i64 512, !dbg !14
  %564 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local160, i32 0) #0, !dbg !14
  %565 = bitcast <4 x i32> %564 to <8 x half>, !dbg !14
  %address.local161 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local144, !dbg !14
  %address.local162 = getelementptr [2 x i8], ptr addrspace(3) %address.local161, i64 %address.local148, !dbg !14
  %address.local163 = getelementptr [2 x i8], ptr addrspace(3) %address.local162, i64 %address.local151, !dbg !14
  %address.local164 = getelementptr i8, ptr addrspace(3) %address.local163, i64 768, !dbg !14
  %566 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local164, i32 0) #0, !dbg !14
  %567 = bitcast <4 x i32> %566 to <8 x half>, !dbg !14
  %568 = add nsw i64 %74, %8, !dbg !17
  %569 = shl i64 %568, 19, !dbg !17
  %570 = and i64 %569, 16252928, !dbg !17
  %571 = add i64 %570, %80, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %572 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %571, !dbg !17
  %573 = getelementptr i8, ptr addrspace(1) %572, i64 64, !dbg !17
  %address.local165 = shl nuw nsw i64 %address.local141, 3, !dbg !11
  %address.local166 = and i64 %address.local165, 1536, !dbg !11
  %address.local167 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local166, !dbg !11
  %address.local168 = ptrtoint ptr addrspace(3) %address.local167 to i64, !dbg !11
  %address.local169 = trunc i64 %address.local168 to i32, !dbg !11
  %574 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local169), !dbg !17
  %575 = zext i32 %574 to i64, !dbg !17
  %576 = inttoptr i64 %575 to ptr addrspace(3), !dbg !17
  %address.local170 = shl nuw nsw i64 %address.local141, 5, !dbg !17
  %address.local171 = and i64 %address.local170, 8064, !dbg !17
  %address.local172 = shl nuw nsw i64 %address.local141, 3, !dbg !17
  %address.local173 = and i64 %address.local172, 24, !dbg !17
  %address.local174 = or disjoint i64 %address.local171, %address.local173, !dbg !17
  %address.local175 = getelementptr [2 x i8], ptr addrspace(1) %573, i64 %address.local174, !dbg !17
  %577 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %578 = ptrtoint ptr addrspace(1) %address.local175 to i64, !dbg !17
  %579 = sub i64 %578, %577, !dbg !17
  %dma.buffer.offset388 = trunc i64 %579 to i32, !dbg !17
  %580 = trunc i64 %577 to i32, !dbg !17
  %581 = insertelement <4 x i32> poison, i32 %580, i64 0, !dbg !17
  %582 = lshr i64 %577, 32, !dbg !17
  %583 = trunc i64 %582 to i32, !dbg !17
  %584 = insertelement <4 x i32> %581, i32 %583, i64 1, !dbg !17
  %585 = insertelement <4 x i32> %584, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc389 = insertelement <4 x i32> %585, i32 131072, i64 3, !dbg !17
  %586 = ptrtoint ptr addrspace(3) %576 to i32, !dbg !17
  %dma.buffer.lds390 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %586), !dbg !17
  %587 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds390, i32 %dma.buffer.offset388, <4 x i32> %dma.buffer.rsrc389) #10, !dbg !17
  %588 = shufflevector <8 x half> %561, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %588, <4 x float> %480) #9, !dbg !14
  %590 = shufflevector <8 x half> %563, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %591 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed36, <4 x half> %590, <4 x float> %589) #9, !dbg !14
  %592 = shufflevector <8 x half> %565, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %592, <4 x float> %591) #9, !dbg !14
  %594 = shufflevector <8 x half> %567, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %595 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed40, <4 x half> %594, <4 x float> %593) #9, !dbg !14
  %596 = shufflevector <8 x half> %561, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %596, <4 x float> %479) #9, !dbg !14
  %598 = shufflevector <8 x half> %563, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed36, <4 x half> %598, <4 x float> %597) #9, !dbg !14
  %600 = shufflevector <8 x half> %565, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %600, <4 x float> %599) #9, !dbg !14
  %602 = shufflevector <8 x half> %567, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %603 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed40, <4 x half> %602, <4 x float> %601) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %604 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local152, i32 0) #0, !dbg !14
  %605 = bitcast <4 x i32> %604 to <8 x half>, !dbg !14
  %606 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local156, i32 0) #0, !dbg !14
  %607 = bitcast <4 x i32> %606 to <8 x half>, !dbg !14
  %608 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local160, i32 0) #0, !dbg !14
  %609 = bitcast <4 x i32> %608 to <8 x half>, !dbg !14
  %610 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local164, i32 0) #0, !dbg !14
  %611 = bitcast <4 x i32> %610 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %612 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %571, !dbg !17
  %613 = getelementptr i8, ptr addrspace(1) %612, i64 128, !dbg !17
  %614 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local169), !dbg !17
  %615 = zext i32 %614 to i64, !dbg !17
  %616 = inttoptr i64 %615 to ptr addrspace(3), !dbg !17
  %address.local176 = getelementptr [2 x i8], ptr addrspace(1) %613, i64 %address.local174, !dbg !17
  %617 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %618 = ptrtoint ptr addrspace(1) %address.local176 to i64, !dbg !17
  %619 = sub i64 %618, %617, !dbg !17
  %dma.buffer.offset385 = trunc i64 %619 to i32, !dbg !17
  %620 = trunc i64 %617 to i32, !dbg !17
  %621 = insertelement <4 x i32> poison, i32 %620, i64 0, !dbg !17
  %622 = lshr i64 %617, 32, !dbg !17
  %623 = trunc i64 %622 to i32, !dbg !17
  %624 = insertelement <4 x i32> %621, i32 %623, i64 1, !dbg !17
  %625 = insertelement <4 x i32> %624, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc386 = insertelement <4 x i32> %625, i32 131072, i64 3, !dbg !17
  %626 = ptrtoint ptr addrspace(3) %616 to i32, !dbg !17
  %dma.buffer.lds387 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %626), !dbg !17
  %627 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds387, i32 %dma.buffer.offset385, <4 x i32> %dma.buffer.rsrc386) #10, !dbg !17
  %628 = shufflevector <8 x half> %605, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %628, <4 x float> %478) #9, !dbg !14
  %630 = shufflevector <8 x half> %607, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %631 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed36, <4 x half> %630, <4 x float> %629) #9, !dbg !14
  %632 = shufflevector <8 x half> %609, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %632, <4 x float> %631) #9, !dbg !14
  %634 = shufflevector <8 x half> %611, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed40, <4 x half> %634, <4 x float> %633) #9, !dbg !14
  %636 = shufflevector <8 x half> %605, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %636, <4 x float> %477) #9, !dbg !14
  %638 = shufflevector <8 x half> %607, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed36, <4 x half> %638, <4 x float> %637) #9, !dbg !14
  %640 = shufflevector <8 x half> %609, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %640, <4 x float> %639) #9, !dbg !14
  %642 = shufflevector <8 x half> %611, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed40, <4 x half> %642, <4 x float> %641) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %644 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local152, i32 0) #0, !dbg !14
  %645 = bitcast <4 x i32> %644 to <8 x half>, !dbg !14
  %646 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local156, i32 0) #0, !dbg !14
  %647 = bitcast <4 x i32> %646 to <8 x half>, !dbg !14
  %648 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local160, i32 0) #0, !dbg !14
  %649 = bitcast <4 x i32> %648 to <8 x half>, !dbg !14
  %650 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local164, i32 0) #0, !dbg !14
  %651 = bitcast <4 x i32> %650 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %652 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %571, !dbg !17
  %653 = getelementptr i8, ptr addrspace(1) %652, i64 192, !dbg !17
  %654 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local169), !dbg !17
  %655 = zext i32 %654 to i64, !dbg !17
  %656 = inttoptr i64 %655 to ptr addrspace(3), !dbg !17
  %address.local177 = getelementptr [2 x i8], ptr addrspace(1) %653, i64 %address.local174, !dbg !17
  %657 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %658 = ptrtoint ptr addrspace(1) %address.local177 to i64, !dbg !17
  %659 = sub i64 %658, %657, !dbg !17
  %dma.buffer.offset = trunc i64 %659 to i32, !dbg !17
  %660 = trunc i64 %657 to i32, !dbg !17
  %661 = insertelement <4 x i32> poison, i32 %660, i64 0, !dbg !17
  %662 = lshr i64 %657, 32, !dbg !17
  %663 = trunc i64 %662 to i32, !dbg !17
  %664 = insertelement <4 x i32> %661, i32 %663, i64 1, !dbg !17
  %665 = insertelement <4 x i32> %664, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %665, i32 131072, i64 3, !dbg !17
  %666 = ptrtoint ptr addrspace(3) %656 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %666), !dbg !17
  %667 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %668 = shufflevector <8 x half> %645, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %668, <4 x float> %476) #9, !dbg !14
  %670 = shufflevector <8 x half> %647, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed36, <4 x half> %670, <4 x float> %669) #9, !dbg !14
  %672 = shufflevector <8 x half> %649, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %672, <4 x float> %671) #9, !dbg !14
  %674 = shufflevector <8 x half> %651, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed40, <4 x half> %674, <4 x float> %673) #9, !dbg !14
  %676 = shufflevector <8 x half> %645, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %676, <4 x float> %475) #9, !dbg !14
  %678 = shufflevector <8 x half> %647, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed36, <4 x half> %678, <4 x float> %677) #9, !dbg !14
  %680 = shufflevector <8 x half> %649, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %680, <4 x float> %679) #9, !dbg !14
  %682 = shufflevector <8 x half> %651, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed40, <4 x half> %682, <4 x float> %681) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %684 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local152, i32 0) #0, !dbg !14
  %685 = bitcast <4 x i32> %684 to <8 x half>, !dbg !14
  %686 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local156, i32 0) #0, !dbg !14
  %687 = bitcast <4 x i32> %686 to <8 x half>, !dbg !14
  %688 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local160, i32 0) #0, !dbg !14
  %689 = bitcast <4 x i32> %688 to <8 x half>, !dbg !14
  %690 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local164, i32 0) #0, !dbg !14
  %691 = bitcast <4 x i32> %690 to <8 x half>, !dbg !14
  %692 = shufflevector <8 x half> %685, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %693 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %692, <4 x float> %474) #9, !dbg !14
  %694 = shufflevector <8 x half> %687, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed36, <4 x half> %694, <4 x float> %693) #9, !dbg !14
  %696 = shufflevector <8 x half> %689, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %697 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %696, <4 x float> %695) #9, !dbg !14
  %698 = shufflevector <8 x half> %691, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed40, <4 x half> %698, <4 x float> %697) #9, !dbg !14
  %700 = shufflevector <8 x half> %685, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %701 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %700, <4 x float> %473) #9, !dbg !14
  %702 = shufflevector <8 x half> %687, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %703 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed36, <4 x half> %702, <4 x float> %701) #9, !dbg !14
  %704 = shufflevector <8 x half> %689, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %705 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %704, <4 x float> %703) #9, !dbg !14
  %706 = shufflevector <8 x half> %691, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed40, <4 x half> %706, <4 x float> %705) #9, !dbg !14
  %708 = shufflevector <4 x float> %595, <4 x float> %603, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %709 = shufflevector <4 x float> %635, <4 x float> %643, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %710 = shufflevector <4 x float> %675, <4 x float> %683, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %711 = shufflevector <4 x float> %699, <4 x float> %707, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %712 = add nuw nsw i64 %41, 1, !dbg !7
  br label %40, !dbg !7

713:                                              ; preds = %40
  %714 = and i32 %24, 16, !dbg !7
  %.not2 = icmp eq i32 %714, 0, !dbg !7
  %715 = and i32 %24, 32, !dbg !7
  %.not3 = icmp eq i32 %715, 0, !dbg !7
  %716 = extractelement <8 x float> %55, i64 0, !dbg !7
  %717 = extractelement <8 x float> %55, i64 1, !dbg !7
  %718 = extractelement <8 x float> %55, i64 2, !dbg !7
  %719 = extractelement <8 x float> %55, i64 3, !dbg !7
  %720 = select i1 %.not2, float %717, float %716, !dbg !7
  %721 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %722 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %721), !dbg !7
  %723 = add nuw nsw i32 %722, 64, !dbg !7
  %724 = xor i32 %722, 16, !dbg !7
  %725 = and i32 %723, 192, !dbg !7
  %726 = icmp samesign ult i32 %724, %725, !dbg !7
  %727 = select i1 %726, i32 %724, i32 %722, !dbg !7
  %728 = shl nuw nsw i32 %727, 2, !dbg !7
  %729 = bitcast float %720 to i32, !dbg !7
  %730 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %729), !dbg !7
  %731 = bitcast i32 %730 to float, !dbg !7
  %732 = select i1 %.not2, float %716, float %731, !dbg !7
  %733 = select i1 %.not2, float %731, float %717, !dbg !7
  %734 = select i1 %.not2, float %719, float %718, !dbg !7
  %735 = bitcast float %734 to i32, !dbg !7
  %736 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %735), !dbg !7
  %737 = bitcast i32 %736 to float, !dbg !7
  %738 = select i1 %.not2, float %718, float %737, !dbg !7
  %739 = select i1 %.not2, float %737, float %719, !dbg !7
  %740 = select i1 %.not3, float %738, float %732, !dbg !7
  %741 = xor i32 %722, 32, !dbg !7
  %742 = icmp samesign ult i32 %741, %725, !dbg !7
  %743 = select i1 %742, i32 %741, i32 %722, !dbg !7
  %744 = shl nuw nsw i32 %743, 2, !dbg !7
  %745 = bitcast float %740 to i32, !dbg !7
  %746 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %745), !dbg !7
  %747 = bitcast i32 %746 to float, !dbg !7
  %748 = select i1 %.not3, float %732, float %747, !dbg !7
  %749 = select i1 %.not3, float %747, float %738, !dbg !7
  %750 = select i1 %.not3, float %739, float %733, !dbg !7
  %751 = bitcast float %750 to i32, !dbg !7
  %752 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %751), !dbg !7
  %753 = bitcast i32 %752 to float, !dbg !7
  %754 = select i1 %.not3, float %733, float %753, !dbg !7
  %755 = select i1 %.not3, float %753, float %739, !dbg !7
  %756 = extractelement <8 x float> %52, i64 0, !dbg !7
  %757 = extractelement <8 x float> %52, i64 1, !dbg !7
  %758 = extractelement <8 x float> %52, i64 2, !dbg !7
  %759 = extractelement <8 x float> %52, i64 3, !dbg !7
  %760 = select i1 %.not2, float %757, float %756, !dbg !7
  %761 = bitcast float %760 to i32, !dbg !7
  %762 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %761), !dbg !7
  %763 = bitcast i32 %762 to float, !dbg !7
  %764 = select i1 %.not2, float %756, float %763, !dbg !7
  %765 = select i1 %.not2, float %763, float %757, !dbg !7
  %766 = select i1 %.not2, float %759, float %758, !dbg !7
  %767 = bitcast float %766 to i32, !dbg !7
  %768 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %767), !dbg !7
  %769 = bitcast i32 %768 to float, !dbg !7
  %770 = select i1 %.not2, float %758, float %769, !dbg !7
  %771 = select i1 %.not2, float %769, float %759, !dbg !7
  %772 = select i1 %.not3, float %770, float %764, !dbg !7
  %773 = bitcast float %772 to i32, !dbg !7
  %774 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %773), !dbg !7
  %775 = bitcast i32 %774 to float, !dbg !7
  %776 = select i1 %.not3, float %764, float %775, !dbg !7
  %777 = select i1 %.not3, float %775, float %770, !dbg !7
  %778 = select i1 %.not3, float %771, float %765, !dbg !7
  %779 = bitcast float %778 to i32, !dbg !7
  %780 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %779), !dbg !7
  %781 = bitcast i32 %780 to float, !dbg !7
  %782 = select i1 %.not3, float %765, float %781, !dbg !7
  %783 = select i1 %.not3, float %781, float %771, !dbg !7
  %784 = extractelement <8 x float> %49, i64 0, !dbg !7
  %785 = extractelement <8 x float> %49, i64 1, !dbg !7
  %786 = extractelement <8 x float> %49, i64 2, !dbg !7
  %787 = extractelement <8 x float> %49, i64 3, !dbg !7
  %788 = select i1 %.not2, float %785, float %784, !dbg !7
  %789 = bitcast float %788 to i32, !dbg !7
  %790 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %789), !dbg !7
  %791 = bitcast i32 %790 to float, !dbg !7
  %792 = select i1 %.not2, float %784, float %791, !dbg !7
  %793 = select i1 %.not2, float %791, float %785, !dbg !7
  %794 = select i1 %.not2, float %787, float %786, !dbg !7
  %795 = bitcast float %794 to i32, !dbg !7
  %796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %795), !dbg !7
  %797 = bitcast i32 %796 to float, !dbg !7
  %798 = select i1 %.not2, float %786, float %797, !dbg !7
  %799 = select i1 %.not2, float %797, float %787, !dbg !7
  %800 = select i1 %.not3, float %798, float %792, !dbg !7
  %801 = bitcast float %800 to i32, !dbg !7
  %802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %801), !dbg !7
  %803 = bitcast i32 %802 to float, !dbg !7
  %804 = select i1 %.not3, float %792, float %803, !dbg !7
  %805 = select i1 %.not3, float %803, float %798, !dbg !7
  %806 = select i1 %.not3, float %799, float %793, !dbg !7
  %807 = bitcast float %806 to i32, !dbg !7
  %808 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %807), !dbg !7
  %809 = bitcast i32 %808 to float, !dbg !7
  %810 = select i1 %.not3, float %793, float %809, !dbg !7
  %811 = select i1 %.not3, float %809, float %799, !dbg !7
  %812 = extractelement <8 x float> %44, i64 0, !dbg !7
  %813 = extractelement <8 x float> %44, i64 1, !dbg !7
  %814 = extractelement <8 x float> %44, i64 2, !dbg !7
  %815 = extractelement <8 x float> %44, i64 3, !dbg !7
  %816 = select i1 %.not2, float %813, float %812, !dbg !7
  %817 = bitcast float %816 to i32, !dbg !7
  %818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %817), !dbg !7
  %819 = bitcast i32 %818 to float, !dbg !7
  %820 = select i1 %.not2, float %812, float %819, !dbg !7
  %821 = select i1 %.not2, float %819, float %813, !dbg !7
  %822 = select i1 %.not2, float %815, float %814, !dbg !7
  %823 = bitcast float %822 to i32, !dbg !7
  %824 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %823), !dbg !7
  %825 = bitcast i32 %824 to float, !dbg !7
  %826 = select i1 %.not2, float %814, float %825, !dbg !7
  %827 = select i1 %.not2, float %825, float %815, !dbg !7
  %828 = select i1 %.not3, float %826, float %820, !dbg !7
  %829 = bitcast float %828 to i32, !dbg !7
  %830 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %829), !dbg !7
  %831 = select i1 %.not3, float %827, float %821, !dbg !7
  %832 = bitcast float %831 to i32, !dbg !7
  %833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %832), !dbg !7
  %834 = shufflevector <1 x float> %45, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %835 = extractelement <8 x float> %55, i64 4, !dbg !7
  %836 = extractelement <8 x float> %55, i64 5, !dbg !7
  %837 = extractelement <8 x float> %55, i64 6, !dbg !7
  %838 = extractelement <8 x float> %55, i64 7, !dbg !7
  %839 = select i1 %.not2, float %836, float %835, !dbg !7
  %840 = bitcast float %839 to i32, !dbg !7
  %841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %840), !dbg !7
  %842 = bitcast i32 %841 to float, !dbg !7
  %843 = select i1 %.not2, float %835, float %842, !dbg !7
  %844 = select i1 %.not2, float %842, float %836, !dbg !7
  %845 = select i1 %.not2, float %838, float %837, !dbg !7
  %846 = bitcast float %845 to i32, !dbg !7
  %847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %846), !dbg !7
  %848 = bitcast i32 %847 to float, !dbg !7
  %849 = select i1 %.not2, float %837, float %848, !dbg !7
  %850 = select i1 %.not2, float %848, float %838, !dbg !7
  %851 = select i1 %.not3, float %849, float %843, !dbg !7
  %852 = bitcast float %851 to i32, !dbg !7
  %853 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %852), !dbg !7
  %854 = bitcast i32 %853 to float, !dbg !7
  %855 = select i1 %.not3, float %843, float %854, !dbg !7
  %856 = select i1 %.not3, float %854, float %849, !dbg !7
  %857 = select i1 %.not3, float %850, float %844, !dbg !7
  %858 = bitcast float %857 to i32, !dbg !7
  %859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %858), !dbg !7
  %860 = bitcast i32 %859 to float, !dbg !7
  %861 = select i1 %.not3, float %844, float %860, !dbg !7
  %862 = select i1 %.not3, float %860, float %850, !dbg !7
  %863 = extractelement <8 x float> %52, i64 4, !dbg !7
  %864 = extractelement <8 x float> %52, i64 5, !dbg !7
  %865 = extractelement <8 x float> %52, i64 6, !dbg !7
  %866 = extractelement <8 x float> %52, i64 7, !dbg !7
  %867 = select i1 %.not2, float %864, float %863, !dbg !7
  %868 = bitcast float %867 to i32, !dbg !7
  %869 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %868), !dbg !7
  %870 = bitcast i32 %869 to float, !dbg !7
  %871 = select i1 %.not2, float %863, float %870, !dbg !7
  %872 = select i1 %.not2, float %870, float %864, !dbg !7
  %873 = select i1 %.not2, float %866, float %865, !dbg !7
  %874 = bitcast float %873 to i32, !dbg !7
  %875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %874), !dbg !7
  %876 = bitcast i32 %875 to float, !dbg !7
  %877 = select i1 %.not2, float %865, float %876, !dbg !7
  %878 = select i1 %.not2, float %876, float %866, !dbg !7
  %879 = select i1 %.not3, float %877, float %871, !dbg !7
  %880 = bitcast float %879 to i32, !dbg !7
  %881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %880), !dbg !7
  %882 = bitcast i32 %881 to float, !dbg !7
  %883 = select i1 %.not3, float %871, float %882, !dbg !7
  %884 = select i1 %.not3, float %882, float %877, !dbg !7
  %885 = select i1 %.not3, float %878, float %872, !dbg !7
  %886 = bitcast float %885 to i32, !dbg !7
  %887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %886), !dbg !7
  %888 = bitcast i32 %887 to float, !dbg !7
  %889 = select i1 %.not3, float %872, float %888, !dbg !7
  %890 = select i1 %.not3, float %888, float %878, !dbg !7
  %891 = extractelement <8 x float> %49, i64 4, !dbg !7
  %892 = extractelement <8 x float> %49, i64 5, !dbg !7
  %893 = extractelement <8 x float> %49, i64 6, !dbg !7
  %894 = extractelement <8 x float> %49, i64 7, !dbg !7
  %895 = select i1 %.not2, float %892, float %891, !dbg !7
  %896 = bitcast float %895 to i32, !dbg !7
  %897 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %896), !dbg !7
  %898 = bitcast i32 %897 to float, !dbg !7
  %899 = select i1 %.not2, float %891, float %898, !dbg !7
  %900 = select i1 %.not2, float %898, float %892, !dbg !7
  %901 = select i1 %.not2, float %894, float %893, !dbg !7
  %902 = bitcast float %901 to i32, !dbg !7
  %903 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %902), !dbg !7
  %904 = bitcast i32 %903 to float, !dbg !7
  %905 = select i1 %.not2, float %893, float %904, !dbg !7
  %906 = select i1 %.not2, float %904, float %894, !dbg !7
  %907 = select i1 %.not3, float %905, float %899, !dbg !7
  %908 = bitcast float %907 to i32, !dbg !7
  %909 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %908), !dbg !7
  %910 = bitcast i32 %909 to float, !dbg !7
  %911 = select i1 %.not3, float %899, float %910, !dbg !7
  %912 = select i1 %.not3, float %910, float %905, !dbg !7
  %913 = select i1 %.not3, float %906, float %900, !dbg !7
  %914 = bitcast float %913 to i32, !dbg !7
  %915 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %914), !dbg !7
  %916 = bitcast i32 %915 to float, !dbg !7
  %917 = select i1 %.not3, float %900, float %916, !dbg !7
  %918 = select i1 %.not3, float %916, float %906, !dbg !7
  %919 = extractelement <8 x float> %44, i64 4, !dbg !7
  %920 = extractelement <8 x float> %44, i64 5, !dbg !7
  %921 = extractelement <8 x float> %44, i64 6, !dbg !7
  %922 = extractelement <8 x float> %44, i64 7, !dbg !7
  %923 = select i1 %.not2, float %920, float %919, !dbg !7
  %924 = bitcast float %923 to i32, !dbg !7
  %925 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %924), !dbg !7
  %926 = bitcast i32 %925 to float, !dbg !7
  %927 = select i1 %.not2, float %919, float %926, !dbg !7
  %928 = select i1 %.not2, float %926, float %920, !dbg !7
  %929 = select i1 %.not2, float %922, float %921, !dbg !7
  %930 = bitcast float %929 to i32, !dbg !7
  %931 = call i32 @llvm.amdgcn.ds.bpermute(i32 %728, i32 %930), !dbg !7
  %932 = bitcast i32 %931 to float, !dbg !7
  %933 = select i1 %.not2, float %921, float %932, !dbg !7
  %934 = select i1 %.not2, float %932, float %922, !dbg !7
  %935 = select i1 %.not3, float %933, float %927, !dbg !7
  %936 = bitcast float %935 to i32, !dbg !7
  %937 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %936), !dbg !7
  %938 = select i1 %.not3, float %934, float %928, !dbg !7
  %939 = bitcast float %938 to i32, !dbg !7
  %940 = call i32 @llvm.amdgcn.ds.bpermute(i32 %744, i32 %939), !dbg !7
  br i1 true, label %941, label %975, !dbg !20

941:                                              ; preds = %713
  %942 = bitcast i32 %937 to float, !dbg !7
  %943 = select i1 %.not3, float %927, float %942, !dbg !7
  %944 = insertelement <4 x float> poison, float %943, i64 0, !dbg !18
  %945 = bitcast i32 %940 to float, !dbg !7
  %946 = select i1 %.not3, float %928, float %945, !dbg !7
  %947 = insertelement <4 x float> %944, float %946, i64 1, !dbg !18
  %948 = select i1 %.not3, float %942, float %933, !dbg !7
  %949 = insertelement <4 x float> %947, float %948, i64 2, !dbg !18
  %950 = select i1 %.not3, float %945, float %934, !dbg !7
  %951 = insertelement <4 x float> %949, float %950, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %834, !dbg !18
  %normalized = fmul <4 x float> %951, %row.reciprocal, !dbg !18
  %952 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %953 = extractelement <4 x half> %952, i64 3, !dbg !19
  %954 = extractelement <4 x half> %952, i64 2, !dbg !19
  %955 = extractelement <4 x half> %952, i64 1, !dbg !19
  %956 = extractelement <4 x half> %952, i64 0, !dbg !19
  %957 = bitcast i32 %830 to float, !dbg !7
  %958 = select i1 %.not3, float %820, float %957, !dbg !7
  %959 = insertelement <4 x float> poison, float %958, i64 0, !dbg !18
  %960 = bitcast i32 %833 to float, !dbg !7
  %961 = select i1 %.not3, float %821, float %960, !dbg !7
  %962 = insertelement <4 x float> %959, float %961, i64 1, !dbg !18
  %963 = select i1 %.not3, float %957, float %826, !dbg !7
  %964 = insertelement <4 x float> %962, float %963, i64 2, !dbg !18
  %965 = select i1 %.not3, float %960, float %827, !dbg !7
  %966 = insertelement <4 x float> %964, float %965, i64 3, !dbg !18
  %normalized23 = fmul <4 x float> %966, %row.reciprocal, !dbg !18
  %967 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !19
  %968 = extractelement <4 x half> %967, i64 3, !dbg !19
  %969 = extractelement <4 x half> %967, i64 2, !dbg !19
  %970 = extractelement <4 x half> %967, i64 1, !dbg !19
  %971 = extractelement <4 x half> %967, i64 0, !dbg !19
  %972 = add nsw i64 %8, %15, !dbg !20
  %973 = shl nsw i64 %972, 19, !dbg !20
  %974 = and i64 %973, 16252928, !dbg !20
  %address.thread178 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local179 = zext nneg i32 %address.thread178 to i64, !dbg !12
  %address.local180 = add nsw i64 %18, %address.local179, !dbg !20
  %address.local181 = lshr i64 %address.local179, 6, !dbg !12
  %address.local182 = shl nuw nsw i64 %address.local181, 4, !dbg !20
  %address.local183 = add nsw i64 %address.local180, %address.local182, !dbg !20
  %address.local184 = and i64 %address.local179, 240, !dbg !20
  %address.local185 = sub nsw i64 %address.local183, %address.local184, !dbg !20
  %address.local186 = add nsw i64 %address.local185, %19, !dbg !20
  %address.local187 = shl nuw nsw i64 %address.local186, 7, !dbg !20
  %address.local188 = add nuw nsw i64 %974, %address.local187, !dbg !20
  %address.local189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !20
  %address.local190 = lshr i64 %address.local179, 2, !dbg !12
  %address.local191 = and i64 %address.local190, 12, !dbg !12
  %address.local192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local189, i64 %address.local191, !dbg !20
  store half %971, ptr addrspace(1) %address.local192, align 2, !dbg !20
  %address.local193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !20
  %address.local194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local193, i64 %address.local191, !dbg !20
  %address.local195 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local194, i64 2, !dbg !20
  store half %970, ptr addrspace(1) %address.local195, align 2, !dbg !20
  %address.local196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !20
  %address.local197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local196, i64 %address.local191, !dbg !20
  %address.local198 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local197, i64 4, !dbg !20
  store half %969, ptr addrspace(1) %address.local198, align 2, !dbg !20
  %address.local199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !20
  %address.local200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local199, i64 %address.local191, !dbg !20
  %address.local201 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local200, i64 6, !dbg !20
  store half %968, ptr addrspace(1) %address.local201, align 2, !dbg !20
  %address.local202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !20
  %address.local203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local202, i64 %address.local191, !dbg !20
  %address.local204 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local203, i64 32, !dbg !20
  store half %956, ptr addrspace(1) %address.local204, align 2, !dbg !20
  %address.local205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !20
  %address.local206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local205, i64 %address.local191, !dbg !20
  %address.local207 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local206, i64 34, !dbg !20
  store half %955, ptr addrspace(1) %address.local207, align 2, !dbg !20
  %address.local208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !20
  %address.local209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local208, i64 %address.local191, !dbg !20
  %address.local210 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local209, i64 36, !dbg !20
  store half %954, ptr addrspace(1) %address.local210, align 2, !dbg !20
  %address.local211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !20
  %address.local212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local211, i64 %address.local191, !dbg !20
  %address.local213 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local212, i64 38, !dbg !20
  store half %953, ptr addrspace(1) %address.local213, align 2, !dbg !20
  br label %975, !dbg !20

975:                                              ; preds = %941, %713
  br i1 true, label %976, label %996, !dbg !20

976:                                              ; preds = %975
  %977 = insertelement <4 x float> poison, float %911, i64 0, !dbg !18
  %978 = insertelement <4 x float> %977, float %917, i64 1, !dbg !18
  %979 = insertelement <4 x float> %978, float %912, i64 2, !dbg !18
  %980 = insertelement <4 x float> %979, float %918, i64 3, !dbg !18
  %row.reciprocal24 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %834, !dbg !18
  %normalized25 = fmul <4 x float> %980, %row.reciprocal24, !dbg !18
  %981 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !19
  %982 = extractelement <4 x half> %981, i64 3, !dbg !19
  %983 = extractelement <4 x half> %981, i64 2, !dbg !19
  %984 = extractelement <4 x half> %981, i64 1, !dbg !19
  %985 = extractelement <4 x half> %981, i64 0, !dbg !19
  %986 = insertelement <4 x float> poison, float %804, i64 0, !dbg !18
  %987 = insertelement <4 x float> %986, float %810, i64 1, !dbg !18
  %988 = insertelement <4 x float> %987, float %805, i64 2, !dbg !18
  %989 = insertelement <4 x float> %988, float %811, i64 3, !dbg !18
  %normalized26 = fmul <4 x float> %989, %row.reciprocal24, !dbg !18
  %990 = fptrunc <4 x float> %normalized26 to <4 x half>, !dbg !19
  %991 = extractelement <4 x half> %990, i64 3, !dbg !19
  %992 = extractelement <4 x half> %990, i64 2, !dbg !19
  %993 = extractelement <4 x half> %990, i64 1, !dbg !19
  %994 = extractelement <4 x half> %990, i64 0, !dbg !19
  %995 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread214 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local215 = zext nneg i32 %address.thread214 to i64, !dbg !12
  %address.local216 = add nsw i64 %18, %address.local215, !dbg !20
  %address.local217 = lshr i64 %address.local215, 6, !dbg !12
  %address.local218 = shl nuw nsw i64 %address.local217, 4, !dbg !20
  %address.local219 = add nsw i64 %address.local216, %address.local218, !dbg !20
  %address.local220 = and i64 %address.local215, 240, !dbg !20
  %address.local221 = sub nsw i64 %address.local219, %address.local220, !dbg !20
  %address.local222 = add nsw i64 %address.local221, %19, !dbg !20
  %address.local223 = shl nuw nsw i64 %address.local222, 7, !dbg !20
  %address.local224 = add nuw nsw i64 %995, %address.local223, !dbg !20
  %address.local225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local224, !dbg !20
  %address.local226 = lshr i64 %address.local215, 2, !dbg !12
  %address.local227 = and i64 %address.local226, 12, !dbg !12
  %address.local228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local225, i64 %address.local227, !dbg !20
  %address.local229 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local228, i64 64, !dbg !20
  store half %994, ptr addrspace(1) %address.local229, align 2, !dbg !20
  %address.local230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local224, !dbg !20
  %address.local231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local230, i64 %address.local227, !dbg !20
  %address.local232 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local231, i64 66, !dbg !20
  store half %993, ptr addrspace(1) %address.local232, align 2, !dbg !20
  %address.local233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local224, !dbg !20
  %address.local234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local233, i64 %address.local227, !dbg !20
  %address.local235 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local234, i64 68, !dbg !20
  store half %992, ptr addrspace(1) %address.local235, align 2, !dbg !20
  %address.local236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local224, !dbg !20
  %address.local237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local236, i64 %address.local227, !dbg !20
  %address.local238 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local237, i64 70, !dbg !20
  store half %991, ptr addrspace(1) %address.local238, align 2, !dbg !20
  %address.local239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local224, !dbg !20
  %address.local240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local239, i64 %address.local227, !dbg !20
  %address.local241 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local240, i64 96, !dbg !20
  store half %985, ptr addrspace(1) %address.local241, align 2, !dbg !20
  %address.local242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local224, !dbg !20
  %address.local243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local242, i64 %address.local227, !dbg !20
  %address.local244 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local243, i64 98, !dbg !20
  store half %984, ptr addrspace(1) %address.local244, align 2, !dbg !20
  %address.local245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local224, !dbg !20
  %address.local246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local245, i64 %address.local227, !dbg !20
  %address.local247 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local246, i64 100, !dbg !20
  store half %983, ptr addrspace(1) %address.local247, align 2, !dbg !20
  %address.local248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local224, !dbg !20
  %address.local249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local248, i64 %address.local227, !dbg !20
  %address.local250 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local249, i64 102, !dbg !20
  store half %982, ptr addrspace(1) %address.local250, align 2, !dbg !20
  br label %996, !dbg !20

996:                                              ; preds = %976, %975
  br i1 true, label %997, label %1017, !dbg !20

997:                                              ; preds = %996
  %998 = insertelement <4 x float> poison, float %883, i64 0, !dbg !18
  %999 = insertelement <4 x float> %998, float %889, i64 1, !dbg !18
  %1000 = insertelement <4 x float> %999, float %884, i64 2, !dbg !18
  %1001 = insertelement <4 x float> %1000, float %890, i64 3, !dbg !18
  %row.reciprocal27 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %834, !dbg !18
  %normalized28 = fmul <4 x float> %1001, %row.reciprocal27, !dbg !18
  %1002 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !19
  %1003 = extractelement <4 x half> %1002, i64 3, !dbg !19
  %1004 = extractelement <4 x half> %1002, i64 2, !dbg !19
  %1005 = extractelement <4 x half> %1002, i64 1, !dbg !19
  %1006 = extractelement <4 x half> %1002, i64 0, !dbg !19
  %1007 = insertelement <4 x float> poison, float %776, i64 0, !dbg !18
  %1008 = insertelement <4 x float> %1007, float %782, i64 1, !dbg !18
  %1009 = insertelement <4 x float> %1008, float %777, i64 2, !dbg !18
  %1010 = insertelement <4 x float> %1009, float %783, i64 3, !dbg !18
  %normalized29 = fmul <4 x float> %1010, %row.reciprocal27, !dbg !18
  %1011 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !19
  %1012 = extractelement <4 x half> %1011, i64 3, !dbg !19
  %1013 = extractelement <4 x half> %1011, i64 2, !dbg !19
  %1014 = extractelement <4 x half> %1011, i64 1, !dbg !19
  %1015 = extractelement <4 x half> %1011, i64 0, !dbg !19
  %1016 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread251 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local252 = zext nneg i32 %address.thread251 to i64, !dbg !12
  %address.local253 = add nsw i64 %18, %address.local252, !dbg !20
  %address.local254 = lshr i64 %address.local252, 6, !dbg !12
  %address.local255 = shl nuw nsw i64 %address.local254, 4, !dbg !20
  %address.local256 = add nsw i64 %address.local253, %address.local255, !dbg !20
  %address.local257 = and i64 %address.local252, 240, !dbg !20
  %address.local258 = sub nsw i64 %address.local256, %address.local257, !dbg !20
  %address.local259 = add nsw i64 %address.local258, %19, !dbg !20
  %address.local260 = shl nuw nsw i64 %address.local259, 7, !dbg !20
  %address.local261 = add nuw nsw i64 %1016, %address.local260, !dbg !20
  %address.local262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local261, !dbg !20
  %address.local263 = lshr i64 %address.local252, 2, !dbg !12
  %address.local264 = and i64 %address.local263, 12, !dbg !12
  %address.local265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local262, i64 %address.local264, !dbg !20
  %address.local266 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local265, i64 128, !dbg !20
  store half %1015, ptr addrspace(1) %address.local266, align 2, !dbg !20
  %address.local267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local261, !dbg !20
  %address.local268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local267, i64 %address.local264, !dbg !20
  %address.local269 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local268, i64 130, !dbg !20
  store half %1014, ptr addrspace(1) %address.local269, align 2, !dbg !20
  %address.local270 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local261, !dbg !20
  %address.local271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local270, i64 %address.local264, !dbg !20
  %address.local272 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local271, i64 132, !dbg !20
  store half %1013, ptr addrspace(1) %address.local272, align 2, !dbg !20
  %address.local273 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local261, !dbg !20
  %address.local274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local273, i64 %address.local264, !dbg !20
  %address.local275 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local274, i64 134, !dbg !20
  store half %1012, ptr addrspace(1) %address.local275, align 2, !dbg !20
  %address.local276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local261, !dbg !20
  %address.local277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local276, i64 %address.local264, !dbg !20
  %address.local278 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local277, i64 160, !dbg !20
  store half %1006, ptr addrspace(1) %address.local278, align 2, !dbg !20
  %address.local279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local261, !dbg !20
  %address.local280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local279, i64 %address.local264, !dbg !20
  %address.local281 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local280, i64 162, !dbg !20
  store half %1005, ptr addrspace(1) %address.local281, align 2, !dbg !20
  %address.local282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local261, !dbg !20
  %address.local283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local282, i64 %address.local264, !dbg !20
  %address.local284 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local283, i64 164, !dbg !20
  store half %1004, ptr addrspace(1) %address.local284, align 2, !dbg !20
  %address.local285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local261, !dbg !20
  %address.local286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local285, i64 %address.local264, !dbg !20
  %address.local287 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local286, i64 166, !dbg !20
  store half %1003, ptr addrspace(1) %address.local287, align 2, !dbg !20
  br label %1017, !dbg !20

1017:                                             ; preds = %997, %996
  br i1 true, label %1018, label %1038, !dbg !20

1018:                                             ; preds = %1017
  %1019 = insertelement <4 x float> poison, float %855, i64 0, !dbg !18
  %1020 = insertelement <4 x float> %1019, float %861, i64 1, !dbg !18
  %1021 = insertelement <4 x float> %1020, float %856, i64 2, !dbg !18
  %1022 = insertelement <4 x float> %1021, float %862, i64 3, !dbg !18
  %row.reciprocal30 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %834, !dbg !18
  %normalized31 = fmul <4 x float> %1022, %row.reciprocal30, !dbg !18
  %1023 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !19
  %1024 = extractelement <4 x half> %1023, i64 3, !dbg !19
  %1025 = extractelement <4 x half> %1023, i64 2, !dbg !19
  %1026 = extractelement <4 x half> %1023, i64 1, !dbg !19
  %1027 = extractelement <4 x half> %1023, i64 0, !dbg !19
  %1028 = insertelement <4 x float> poison, float %748, i64 0, !dbg !18
  %1029 = insertelement <4 x float> %1028, float %754, i64 1, !dbg !18
  %1030 = insertelement <4 x float> %1029, float %749, i64 2, !dbg !18
  %1031 = insertelement <4 x float> %1030, float %755, i64 3, !dbg !18
  %normalized32 = fmul <4 x float> %1031, %row.reciprocal30, !dbg !18
  %1032 = fptrunc <4 x float> %normalized32 to <4 x half>, !dbg !19
  %1033 = extractelement <4 x half> %1032, i64 3, !dbg !19
  %1034 = extractelement <4 x half> %1032, i64 2, !dbg !19
  %1035 = extractelement <4 x half> %1032, i64 1, !dbg !19
  %1036 = extractelement <4 x half> %1032, i64 0, !dbg !19
  %1037 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread288 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local289 = zext nneg i32 %address.thread288 to i64, !dbg !12
  %address.local290 = add nsw i64 %18, %address.local289, !dbg !20
  %address.local291 = lshr i64 %address.local289, 6, !dbg !12
  %address.local292 = shl nuw nsw i64 %address.local291, 4, !dbg !20
  %address.local293 = add nsw i64 %address.local290, %address.local292, !dbg !20
  %address.local294 = and i64 %address.local289, 240, !dbg !20
  %address.local295 = sub nsw i64 %address.local293, %address.local294, !dbg !20
  %address.local296 = add nsw i64 %address.local295, %19, !dbg !20
  %address.local297 = shl nuw nsw i64 %address.local296, 7, !dbg !20
  %address.local298 = add nuw nsw i64 %1037, %address.local297, !dbg !20
  %address.local299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local298, !dbg !20
  %address.local300 = lshr i64 %address.local289, 2, !dbg !12
  %address.local301 = and i64 %address.local300, 12, !dbg !12
  %address.local302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local299, i64 %address.local301, !dbg !20
  %address.local303 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local302, i64 192, !dbg !20
  store half %1036, ptr addrspace(1) %address.local303, align 2, !dbg !20
  %address.local304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local298, !dbg !20
  %address.local305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local304, i64 %address.local301, !dbg !20
  %address.local306 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local305, i64 194, !dbg !20
  store half %1035, ptr addrspace(1) %address.local306, align 2, !dbg !20
  %address.local307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local298, !dbg !20
  %address.local308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local307, i64 %address.local301, !dbg !20
  %address.local309 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local308, i64 196, !dbg !20
  store half %1034, ptr addrspace(1) %address.local309, align 2, !dbg !20
  %address.local310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local298, !dbg !20
  %address.local311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local310, i64 %address.local301, !dbg !20
  %address.local312 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local311, i64 198, !dbg !20
  store half %1033, ptr addrspace(1) %address.local312, align 2, !dbg !20
  %address.local313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local298, !dbg !20
  %address.local314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local313, i64 %address.local301, !dbg !20
  %address.local315 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local314, i64 224, !dbg !20
  store half %1027, ptr addrspace(1) %address.local315, align 2, !dbg !20
  %address.local316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local298, !dbg !20
  %address.local317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local316, i64 %address.local301, !dbg !20
  %address.local318 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local317, i64 226, !dbg !20
  store half %1026, ptr addrspace(1) %address.local318, align 2, !dbg !20
  %address.local319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local298, !dbg !20
  %address.local320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local319, i64 %address.local301, !dbg !20
  %address.local321 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local320, i64 228, !dbg !20
  store half %1025, ptr addrspace(1) %address.local321, align 2, !dbg !20
  %address.local322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local298, !dbg !20
  %address.local323 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local322, i64 %address.local301, !dbg !20
  %address.local324 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local323, i64 230, !dbg !20
  store half %1024, ptr addrspace(1) %address.local324, align 2, !dbg !20
  br label %1038, !dbg !20

1038:                                             ; preds = %1018, %1017
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/baseline")
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
