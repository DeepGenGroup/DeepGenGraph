; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/nohint-rolled/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/nohint-rolled/input.mlir"

@shm_0 = addrspace(3) global [8192 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 6, !dbg !10
  %.lobit = ashr i32 %7, 31, !dbg !12
  %10 = xor i32 %7, %.lobit, !dbg !12
  %11 = sdiv i32 %10, 64, !dbg !12
  %.lobit1 = ashr i32 %7, 31, !dbg !12
  %12 = xor i32 %11, %.lobit1, !dbg !12
  %13 = sext i32 %12 to i64, !dbg !12
  %14 = add nsw i64 %6, %13, !dbg !12
  %15 = and i64 %14, 31, !dbg !12
  %16 = shl nsw i64 %8, 6, !dbg !12
  %17 = mul nsw i64 %13, -4096, !dbg !12
  %18 = add nsw i64 %16, %17, !dbg !12
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !39
  %20 = zext nneg i32 %19 to i64, !dbg !13
  %21 = lshr i64 %20, 6, !dbg !13
  %22 = or disjoint i64 %9, 63, !dbg !14
  %23 = add nsw i64 %9, 64, !dbg !14
  %24 = icmp uge i64 %23, %22, !dbg !14
  br label %25, !dbg !7

25:                                               ; preds = %196, %4
  %26 = phi i64 [ %511, %196 ], [ 0, %4 ], !dbg !7
  %27 = phi <4 x float> [ %394, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %28 = phi <4 x float> [ %402, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %29 = phi <8 x float> [ %507, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %30 = phi <1 x float> [ %358, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %31 = phi <1 x float> [ %261, %196 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %32 = phi <4 x float> [ %434, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %33 = phi <4 x float> [ %442, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %34 = phi <8 x float> [ %508, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %35 = phi <4 x float> [ %474, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %36 = phi <4 x float> [ %482, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %37 = phi <8 x float> [ %509, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %38 = phi <4 x float> [ %498, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %39 = phi <4 x float> [ %506, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %40 = phi <8 x float> [ %510, %196 ], [ zeroinitializer, %4 ], !dbg !7
  %.not = icmp sgt i64 %26, %8, !dbg !7
  br i1 %.not, label %512, label %41, !dbg !7

41:                                               ; preds = %25
  %42 = shl nsw i64 %26, 6, !dbg !7
  %43 = lshr i64 %26, 6, !dbg !16
  %44 = shl i64 %43, 12, !dbg !16
  %45 = sub i64 %42, %44, !dbg !16
  br label %46, !dbg !21

46:                                               ; preds = %51, %41
  %47 = phi i64 [ %125, %51 ], [ 0, %41 ], !dbg !21
  %48 = phi <4 x float> [ %120, %51 ], [ zeroinitializer, %41 ], !dbg !21
  %49 = phi <16 x float> [ %124, %51 ], [ zeroinitializer, %41 ], !dbg !21
  %50 = icmp samesign ult i64 %47, 4, !dbg !21
  br i1 %50, label %51, label %126, !dbg !21

51:                                               ; preds = %46
  %52 = shl nuw nsw i64 %47, 5, !dbg !21
  %.idx = shl nuw nsw i64 %15, 20, !dbg !12
  %53 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx20 = shl nsw i64 %18, 8, !dbg !12
  %54 = getelementptr i8, ptr addrspace(1) %53, i64 %.idx20, !dbg !12
  %55 = getelementptr [2 x i8], ptr addrspace(1) %54, i64 %52, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local42 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local43 = and i64 %address.local42, 1536, !dbg !12
  %address.local44 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local43, !dbg !12
  %address.local45 = ptrtoint ptr addrspace(3) %address.local44 to i64, !dbg !12
  %address.local46 = trunc i64 %address.local45 to i32, !dbg !12
  %56 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local46), !dbg !12
  %57 = zext i32 %56 to i64, !dbg !12
  %58 = inttoptr i64 %57 to ptr addrspace(3), !dbg !12
  %address.local47 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local48 = and i64 %address.local47, 8064, !dbg !12
  %address.local49 = lshr i64 %address.local, 3, !dbg !12
  %address.local50 = and i64 %address.local49, 24, !dbg !12
  %address.local51 = or disjoint i64 %address.local48, %address.local50, !dbg !12
  %address.local52 = getelementptr [2 x i8], ptr addrspace(1) %55, i64 %address.local51, !dbg !12
  %59 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %60 = ptrtoint ptr addrspace(1) %address.local52 to i64, !dbg !12
  %61 = sub i64 %60, %59, !dbg !12
  %dma.buffer.offset387 = trunc i64 %61 to i32, !dbg !12
  %62 = trunc i64 %59 to i32, !dbg !12
  %63 = insertelement <4 x i32> poison, i32 %62, i64 0, !dbg !12
  %64 = lshr i64 %59, 32, !dbg !12
  %65 = trunc i64 %64 to i32, !dbg !12
  %66 = insertelement <4 x i32> %63, i32 %65, i64 1, !dbg !12
  %67 = insertelement <4 x i32> %66, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc388 = insertelement <4 x i32> %67, i32 131072, i64 3, !dbg !12
  %68 = ptrtoint ptr addrspace(3) %58 to i32, !dbg !12
  %dma.buffer.lds389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %68), !dbg !12
  %69 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds389, i32 %dma.buffer.offset387, <4 x i32> %dma.buffer.rsrc388) #10, !dbg !12
  %70 = shl nuw nsw i64 %47, 11, !dbg !16
  %71 = add nuw i64 %26, %70, !dbg !16
  %.lobit21 = ashr i64 %71, 63, !dbg !16
  %72 = xor i64 %71, %.lobit21, !dbg !16
  %73 = sdiv i64 %72, 8192, !dbg !16
  %.lobit22 = ashr i64 %71, 63, !dbg !16
  %74 = xor i64 %73, %.lobit22, !dbg !16
  %75 = add nsw i64 %74, %6, !dbg !16
  %76 = add nuw nsw i64 %52, %43, !dbg !16
  %77 = and i64 %76, 127, !dbg !16
  %78 = shl i64 %75, 19, !dbg !16
  %79 = and i64 %78, 16252928, !dbg !16
  %80 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %79, !dbg !16
  %81 = getelementptr [2 x i8], ptr addrspace(1) %80, i64 %77, !dbg !16
  %.idx23 = shl i64 %45, 8, !dbg !16
  %82 = getelementptr i8, ptr addrspace(1) %81, i64 %.idx23, !dbg !16
  %address.local53 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local43, !dbg !16
  %address.local54 = ptrtoint ptr addrspace(3) %address.local53 to i64, !dbg !16
  %address.local55 = trunc i64 %address.local54 to i32, !dbg !16
  %83 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local55), !dbg !16
  %84 = zext i32 %83 to i64, !dbg !16
  %85 = inttoptr i64 %84 to ptr addrspace(3), !dbg !16
  %address.local56 = getelementptr [2 x i8], ptr addrspace(1) %82, i64 %address.local51, !dbg !16
  %86 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %87 = ptrtoint ptr addrspace(1) %address.local56 to i64, !dbg !16
  %88 = sub i64 %87, %86, !dbg !16
  %dma.buffer.offset = trunc i64 %88 to i32, !dbg !16
  %89 = trunc i64 %86 to i32, !dbg !16
  %90 = insertelement <4 x i32> poison, i32 %89, i64 0, !dbg !16
  %91 = lshr i64 %86, 32, !dbg !16
  %92 = trunc i64 %91 to i32, !dbg !16
  %93 = insertelement <4 x i32> %90, i32 %92, i64 1, !dbg !16
  %94 = insertelement <4 x i32> %93, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc = insertelement <4 x i32> %94, i32 131072, i64 3, !dbg !16
  %95 = ptrtoint ptr addrspace(3) %85 to i32, !dbg !16
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %95), !dbg !16
  %96 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local57 = lshr i32 %address.thread, 2, !dbg !13
  %address.local58 = and i32 %address.local57, 4, !dbg !13
  %address.local59 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local60 = and i32 %address.local59, 384, !dbg !13
  %address.local61 = or disjoint i32 %address.local58, %address.local60, !dbg !13
  %address.local62 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local63 = and i32 %address.local62, 120, !dbg !13
  %address.local64 = or disjoint i32 %address.local61, %address.local63, !dbg !13
  %address.local65 = shl nuw nsw i32 %address.thread, 4, !dbg !13
  %address.local66 = and i32 %address.local65, 512, !dbg !13
  %address.local67 = or disjoint i32 %address.local64, %address.local66, !dbg !13
  %address.local68 = zext nneg i32 %address.local67 to i64, !dbg !13
  %address.local69 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local68, !dbg !13
  %97 = load <4 x half>, ptr addrspace(3) %address.local69, align 2, !dbg !13
  %address.local70 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local68, !dbg !13
  %address.local71 = getelementptr i8, ptr addrspace(3) %address.local70, i64 2048, !dbg !13
  %98 = load <4 x half>, ptr addrspace(3) %address.local71, align 2, !dbg !13
  %99 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !13
  %100 = fmul <4 x half> %98, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local72 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local73 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local72, !dbg !8
  %address.local74 = lshr i64 %address.local, 2, !dbg !8
  %address.local75 = mul nsw i64 %address.local74, -512, !dbg !8
  %address.local76 = getelementptr [2 x i8], ptr addrspace(3) %address.local73, i64 %address.local75, !dbg !8
  %address.local77 = shl nuw nsw i64 %address.local, 1, !dbg !8
  %address.local78 = and i64 %address.local77, 24, !dbg !8
  %address.local79 = getelementptr [2 x i8], ptr addrspace(3) %address.local76, i64 %address.local78, !dbg !8
  %address.local80 = lshr i64 %address.local, 2, !dbg !13
  %address.local81 = and i64 %address.local80, 12, !dbg !13
  %address.local82 = getelementptr [2 x i8], ptr addrspace(3) %address.local79, i64 %address.local81, !dbg !8
  %address.local83 = and i64 %address.local, 63, !dbg !13
  %address.local84 = lshr i64 %address.local83, 5, !dbg !13
  %address.local85 = mul nuw nsw i64 %address.local84, 504, !dbg !13
  %address.local86 = getelementptr [2 x i8], ptr addrspace(3) %address.local82, i64 %address.local85, !dbg !8
  %101 = load <4 x half>, ptr addrspace(3) %address.local86, align 2, !dbg !8
  %102 = shufflevector <16 x float> %49, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %address.local87 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local72, !dbg !8
  %address.local88 = getelementptr [2 x i8], ptr addrspace(3) %address.local87, i64 %address.local81, !dbg !8
  %address.local89 = getelementptr [2 x i8], ptr addrspace(3) %address.local88, i64 %address.local85, !dbg !8
  %address.local90 = getelementptr [2 x i8], ptr addrspace(3) %address.local89, i64 %address.local75, !dbg !8
  %address.local91 = getelementptr [2 x i8], ptr addrspace(3) %address.local90, i64 %address.local78, !dbg !8
  %address.local92 = getelementptr i8, ptr addrspace(3) %address.local91, i64 2048, !dbg !8
  %103 = load <4 x half>, ptr addrspace(3) %address.local92, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %101, <4 x float> %102) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %105 = shufflevector <16 x float> %49, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %103, <4 x float> %104) #9, !dbg !8
  %107 = shufflevector <16 x float> %49, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !36
  %address.local93 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local72, !dbg !8
  %address.local94 = getelementptr [2 x i8], ptr addrspace(3) %address.local93, i64 %address.local75, !dbg !8
  %address.local95 = getelementptr [2 x i8], ptr addrspace(3) %address.local94, i64 %address.local78, !dbg !8
  %address.local96 = getelementptr [2 x i8], ptr addrspace(3) %address.local95, i64 %address.local81, !dbg !8
  %address.local97 = getelementptr [2 x i8], ptr addrspace(3) %address.local96, i64 %address.local85, !dbg !8
  %address.local98 = getelementptr i8, ptr addrspace(3) %address.local97, i64 64, !dbg !8
  %108 = load <4 x half>, ptr addrspace(3) %address.local98, align 2, !dbg !8
  %address.local99 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local72, !dbg !8
  %address.local100 = getelementptr [2 x i8], ptr addrspace(3) %address.local99, i64 %address.local81, !dbg !8
  %address.local101 = getelementptr [2 x i8], ptr addrspace(3) %address.local100, i64 %address.local85, !dbg !8
  %address.local102 = getelementptr [2 x i8], ptr addrspace(3) %address.local101, i64 %address.local75, !dbg !8
  %address.local103 = getelementptr [2 x i8], ptr addrspace(3) %address.local102, i64 %address.local78, !dbg !8
  %address.local104 = getelementptr i8, ptr addrspace(3) %address.local103, i64 2112, !dbg !8
  %109 = load <4 x half>, ptr addrspace(3) %address.local104, align 2, !dbg !8
  %110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %108, <4 x float> %105) #9, !dbg !8
  %address.local105 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local72, !dbg !8
  %address.local106 = getelementptr [2 x i8], ptr addrspace(3) %address.local105, i64 %address.local75, !dbg !8
  %address.local107 = getelementptr [2 x i8], ptr addrspace(3) %address.local106, i64 %address.local78, !dbg !8
  %address.local108 = getelementptr [2 x i8], ptr addrspace(3) %address.local107, i64 %address.local81, !dbg !8
  %address.local109 = getelementptr [2 x i8], ptr addrspace(3) %address.local108, i64 %address.local85, !dbg !8
  %address.local110 = getelementptr i8, ptr addrspace(3) %address.local109, i64 128, !dbg !8
  %111 = load <4 x half>, ptr addrspace(3) %address.local110, align 2, !dbg !8
  %112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %109, <4 x float> %110) #9, !dbg !8
  %address.local111 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local72, !dbg !8
  %address.local112 = getelementptr [2 x i8], ptr addrspace(3) %address.local111, i64 %address.local81, !dbg !8
  %address.local113 = getelementptr [2 x i8], ptr addrspace(3) %address.local112, i64 %address.local85, !dbg !8
  %address.local114 = getelementptr [2 x i8], ptr addrspace(3) %address.local113, i64 %address.local75, !dbg !8
  %address.local115 = getelementptr [2 x i8], ptr addrspace(3) %address.local114, i64 %address.local78, !dbg !8
  %address.local116 = getelementptr i8, ptr addrspace(3) %address.local115, i64 2176, !dbg !8
  %113 = load <4 x half>, ptr addrspace(3) %address.local116, align 2, !dbg !8
  %114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %111, <4 x float> %107) #9, !dbg !8
  %115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %113, <4 x float> %114) #9, !dbg !8
  %address.local117 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local72, !dbg !8
  %address.local118 = getelementptr [2 x i8], ptr addrspace(3) %address.local117, i64 %address.local75, !dbg !8
  %address.local119 = getelementptr [2 x i8], ptr addrspace(3) %address.local118, i64 %address.local78, !dbg !8
  %address.local120 = getelementptr [2 x i8], ptr addrspace(3) %address.local119, i64 %address.local81, !dbg !8
  %address.local121 = getelementptr [2 x i8], ptr addrspace(3) %address.local120, i64 %address.local85, !dbg !8
  %address.local122 = getelementptr i8, ptr addrspace(3) %address.local121, i64 192, !dbg !8
  %116 = load <4 x half>, ptr addrspace(3) %address.local122, align 2, !dbg !8
  %address.local123 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local72, !dbg !8
  %address.local124 = getelementptr [2 x i8], ptr addrspace(3) %address.local123, i64 %address.local81, !dbg !8
  %address.local125 = getelementptr [2 x i8], ptr addrspace(3) %address.local124, i64 %address.local85, !dbg !8
  %address.local126 = getelementptr [2 x i8], ptr addrspace(3) %address.local125, i64 %address.local75, !dbg !8
  %address.local127 = getelementptr [2 x i8], ptr addrspace(3) %address.local126, i64 %address.local78, !dbg !8
  %address.local128 = getelementptr i8, ptr addrspace(3) %address.local127, i64 2240, !dbg !8
  %117 = load <4 x half>, ptr addrspace(3) %address.local128, align 2, !dbg !8
  %118 = shufflevector <4 x float> %115, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %116, <4 x float> %48) #9, !dbg !8
  %120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %117, <4 x float> %119) #9, !dbg !8
  %121 = shufflevector <4 x float> %120, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %122 = shufflevector <4 x float> %106, <4 x float> %112, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %123 = shufflevector <16 x float> %122, <16 x float> %118, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %124 = shufflevector <16 x float> %123, <16 x float> %121, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !21
  fence syncscope("workgroup") release, !dbg !21
  call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  %125 = add nuw nsw i64 %47, 1, !dbg !21
  br label %46, !dbg !21

126:                                              ; preds = %46
  %127 = add nsw i64 %43, %6, !dbg !17
  %128 = shl i64 %127, 19, !dbg !17
  %129 = and i64 %128, 16252928, !dbg !17
  %130 = shl nsw i64 %45, 7, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %131 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %129, !dbg !17
  %132 = getelementptr [2 x i8], ptr addrspace(1) %131, i64 %130, !dbg !17
  %address.thread129 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !17, !range !39, !frisk.register_lifetimes !1
  %address.local130 = zext nneg i32 %address.thread129 to i64, !dbg !13
  %address.local131 = shl nuw nsw i64 %address.local130, 3, !dbg !12
  %address.local132 = and i64 %address.local131, 1536, !dbg !12
  %address.local133 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local132, !dbg !12
  %address.local134 = ptrtoint ptr addrspace(3) %address.local133 to i64, !dbg !12
  %address.local135 = trunc i64 %address.local134 to i32, !dbg !12
  %133 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local135), !dbg !17
  %134 = zext i32 %133 to i64, !dbg !17
  %135 = inttoptr i64 %134 to ptr addrspace(3), !dbg !17
  %address.local136 = shl nuw nsw i64 %address.local130, 5, !dbg !17
  %address.local137 = and i64 %address.local136, 8064, !dbg !17
  %address.local138 = shl nuw nsw i64 %address.local130, 3, !dbg !17
  %address.local139 = and i64 %address.local138, 24, !dbg !17
  %address.local140 = or disjoint i64 %address.local137, %address.local139, !dbg !17
  %address.local141 = getelementptr [2 x i8], ptr addrspace(1) %132, i64 %address.local140, !dbg !17
  %136 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %137 = ptrtoint ptr addrspace(1) %address.local141 to i64, !dbg !17
  %138 = sub i64 %137, %136, !dbg !17
  %dma.buffer.offset399 = trunc i64 %138 to i32, !dbg !17
  %139 = trunc i64 %136 to i32, !dbg !17
  %140 = insertelement <4 x i32> poison, i32 %139, i64 0, !dbg !17
  %141 = lshr i64 %136, 32, !dbg !17
  %142 = trunc i64 %141 to i32, !dbg !17
  %143 = insertelement <4 x i32> %140, i32 %142, i64 1, !dbg !17
  %144 = insertelement <4 x i32> %143, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc400 = insertelement <4 x i32> %144, i32 131072, i64 3, !dbg !17
  %145 = ptrtoint ptr addrspace(3) %135 to i32, !dbg !17
  %dma.buffer.lds401 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %145), !dbg !17
  %146 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds401, i32 %dma.buffer.offset399, <4 x i32> %dma.buffer.rsrc400) #10, !dbg !17
  %147 = or disjoint i64 %42, 63, !dbg !14
  %148 = icmp uge i64 %9, %147, !dbg !14
  %149 = and i1 %24, %148, !dbg !14
  br i1 %149, label %150, label %151, !dbg !14

150:                                              ; preds = %126
  br label %194, !dbg !14

151:                                              ; preds = %126
  %152 = add nsw i64 %9, %20, !dbg !14
  %153 = shl nuw nsw i64 %21, 4, !dbg !14
  %154 = add nsw i64 %152, %153, !dbg !14
  %155 = and i32 %19, 240, !dbg !14
  %index.compact = zext nneg i32 %155 to i64, !dbg !14
  %156 = sub nsw i64 %154, %index.compact, !dbg !14
  %157 = add nsw i64 %156, 1, !dbg !22
  %158 = and i64 %20, 48, !dbg !14
  %159 = or disjoint i64 %42, %158, !dbg !14
  %160 = lshr i64 %157, 32, !dbg !23
  %compare.high = trunc i64 %160 to i32, !dbg !23
  %161 = lshr i64 %159, 32, !dbg !23
  %compare.high327 = trunc i64 %161 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high327, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high327, !dbg !23
  %compare.low.base = trunc i64 %157 to i32, !dbg !23
  %compare.low.base328 = trunc i64 %159 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base328, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %162 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low329 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result330 = select i1 %compare.high.equal, i1 %compare.low329, i1 %compare.high.order, !dbg !23
  %163 = select i1 %compare.result330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local332 = or i32 %compare.local.base331, 2, !dbg !23
  %compare.low333 = icmp ugt i32 %compare.low.base, %compare.local332, !dbg !23
  %compare.result334 = select i1 %compare.high.equal, i1 %compare.low333, i1 %compare.high.order, !dbg !23
  %164 = select i1 %compare.result334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local336 = or i32 %compare.local.base335, 3, !dbg !23
  %compare.low337 = icmp ugt i32 %compare.low.base, %compare.local336, !dbg !23
  %compare.result338 = select i1 %compare.high.equal, i1 %compare.low337, i1 %compare.high.order, !dbg !23
  %165 = select i1 %compare.result338, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local340 = or i32 %compare.local.base339, 4, !dbg !23
  %compare.low341 = icmp ugt i32 %compare.low.base, %compare.local340, !dbg !23
  %compare.result342 = select i1 %compare.high.equal, i1 %compare.low341, i1 %compare.high.order, !dbg !23
  %166 = select i1 %compare.result342, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local344 = or i32 %compare.local.base343, 5, !dbg !23
  %compare.low345 = icmp ugt i32 %compare.low.base, %compare.local344, !dbg !23
  %compare.result346 = select i1 %compare.high.equal, i1 %compare.low345, i1 %compare.high.order, !dbg !23
  %167 = select i1 %compare.result346, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local348 = or i32 %compare.local.base347, 6, !dbg !23
  %compare.low349 = icmp ugt i32 %compare.low.base, %compare.local348, !dbg !23
  %compare.result350 = select i1 %compare.high.equal, i1 %compare.low349, i1 %compare.high.order, !dbg !23
  %168 = select i1 %compare.result350, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local352 = or i32 %compare.local.base351, 7, !dbg !23
  %compare.low353 = icmp ugt i32 %compare.low.base, %compare.local352, !dbg !23
  %compare.result354 = select i1 %compare.high.equal, i1 %compare.low353, i1 %compare.high.order, !dbg !23
  %169 = select i1 %compare.result354, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local356 = or i32 %compare.local.base355, 8, !dbg !23
  %compare.low357 = icmp ugt i32 %compare.low.base, %compare.local356, !dbg !23
  %compare.result358 = select i1 %compare.high.equal, i1 %compare.low357, i1 %compare.high.order, !dbg !23
  %170 = select i1 %compare.result358, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local360 = or i32 %compare.local.base359, 9, !dbg !23
  %compare.low361 = icmp ugt i32 %compare.low.base, %compare.local360, !dbg !23
  %compare.result362 = select i1 %compare.high.equal, i1 %compare.low361, i1 %compare.high.order, !dbg !23
  %171 = select i1 %compare.result362, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local364 = or i32 %compare.local.base363, 10, !dbg !23
  %compare.low365 = icmp ugt i32 %compare.low.base, %compare.local364, !dbg !23
  %compare.result366 = select i1 %compare.high.equal, i1 %compare.low365, i1 %compare.high.order, !dbg !23
  %172 = select i1 %compare.result366, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local368 = or i32 %compare.local.base367, 11, !dbg !23
  %compare.low369 = icmp ugt i32 %compare.low.base, %compare.local368, !dbg !23
  %compare.result370 = select i1 %compare.high.equal, i1 %compare.low369, i1 %compare.high.order, !dbg !23
  %173 = select i1 %compare.result370, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local372 = or i32 %compare.local.base371, 12, !dbg !23
  %compare.low373 = icmp ugt i32 %compare.low.base, %compare.local372, !dbg !23
  %compare.result374 = select i1 %compare.high.equal, i1 %compare.low373, i1 %compare.high.order, !dbg !23
  %174 = select i1 %compare.result374, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local376 = or i32 %compare.local.base375, 13, !dbg !23
  %compare.low377 = icmp ugt i32 %compare.low.base, %compare.local376, !dbg !23
  %compare.result378 = select i1 %compare.high.equal, i1 %compare.low377, i1 %compare.high.order, !dbg !23
  %175 = select i1 %compare.result378, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local380 = or i32 %compare.local.base379, 14, !dbg !23
  %compare.low381 = icmp ugt i32 %compare.low.base, %compare.local380, !dbg !23
  %compare.result382 = select i1 %compare.high.equal, i1 %compare.low381, i1 %compare.high.order, !dbg !23
  %176 = select i1 %compare.result382, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base328, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local384 = or i32 %compare.local.base383, 15, !dbg !23
  %compare.low385 = icmp ugt i32 %compare.low.base, %compare.local384, !dbg !23
  %compare.result386 = select i1 %compare.high.equal, i1 %compare.low385, i1 %compare.high.order, !dbg !23
  %177 = select i1 %compare.result386, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %178 = insertelement <16 x float> poison, float %162, i64 0, !dbg !14
  %179 = insertelement <16 x float> %178, float %163, i64 1, !dbg !14
  %180 = insertelement <16 x float> %179, float %164, i64 2, !dbg !14
  %181 = insertelement <16 x float> %180, float %165, i64 3, !dbg !14
  %182 = insertelement <16 x float> %181, float %166, i64 4, !dbg !14
  %183 = insertelement <16 x float> %182, float %167, i64 5, !dbg !14
  %184 = insertelement <16 x float> %183, float %168, i64 6, !dbg !14
  %185 = insertelement <16 x float> %184, float %169, i64 7, !dbg !14
  %186 = insertelement <16 x float> %185, float %170, i64 8, !dbg !14
  %187 = insertelement <16 x float> %186, float %171, i64 9, !dbg !14
  %188 = insertelement <16 x float> %187, float %172, i64 10, !dbg !14
  %189 = insertelement <16 x float> %188, float %173, i64 11, !dbg !14
  %190 = insertelement <16 x float> %189, float %174, i64 12, !dbg !14
  %191 = insertelement <16 x float> %190, float %175, i64 13, !dbg !14
  %192 = insertelement <16 x float> %191, float %176, i64 14, !dbg !14
  %193 = insertelement <16 x float> %192, float %177, i64 15, !dbg !14
  br label %194, !dbg !14

194:                                              ; preds = %150, %151
  %195 = phi <16 x float> [ %193, %151 ], [ zeroinitializer, %150 ], !dbg !14
  br label %196, !dbg !14

196:                                              ; preds = %194
  %197 = shufflevector <16 x float> %49, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %198 = shufflevector <16 x float> %195, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %199 = fadd <4 x float> %197, %198, !dbg !25
  %200 = extractelement <4 x float> %199, i64 0, !dbg !25
  %201 = extractelement <4 x float> %199, i64 1, !dbg !25
  %202 = extractelement <4 x float> %199, i64 2, !dbg !25
  %203 = extractelement <4 x float> %199, i64 3, !dbg !25
  %204 = shufflevector <16 x float> %49, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %205 = shufflevector <16 x float> %195, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %206 = fadd <4 x float> %204, %205, !dbg !25
  %207 = extractelement <4 x float> %206, i64 0, !dbg !25
  %208 = extractelement <4 x float> %206, i64 1, !dbg !25
  %209 = extractelement <4 x float> %206, i64 2, !dbg !25
  %210 = extractelement <4 x float> %206, i64 3, !dbg !25
  %211 = shufflevector <16 x float> %49, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %212 = shufflevector <16 x float> %195, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %213 = fadd <4 x float> %211, %212, !dbg !25
  %214 = extractelement <4 x float> %213, i64 0, !dbg !25
  %215 = extractelement <4 x float> %213, i64 1, !dbg !25
  %216 = extractelement <4 x float> %213, i64 2, !dbg !25
  %217 = extractelement <4 x float> %213, i64 3, !dbg !25
  %218 = shufflevector <16 x float> %195, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %219 = fadd <4 x float> %48, %218, !dbg !25
  %220 = extractelement <4 x float> %219, i64 0, !dbg !25
  %221 = extractelement <4 x float> %219, i64 1, !dbg !25
  %222 = extractelement <4 x float> %219, i64 2, !dbg !25
  %223 = extractelement <4 x float> %219, i64 3, !dbg !25
  %224 = call float @llvm.maxnum.f32(float %200, float 0xFFF0000000000000), !dbg !26
  %225 = call float @llvm.maxnum.f32(float %224, float %201), !dbg !26
  %226 = call float @llvm.maxnum.f32(float %225, float %202), !dbg !26
  %227 = call float @llvm.maxnum.f32(float %226, float %203), !dbg !26
  %228 = call float @llvm.maxnum.f32(float %227, float %207), !dbg !26
  %229 = call float @llvm.maxnum.f32(float %228, float %208), !dbg !26
  %230 = call float @llvm.maxnum.f32(float %229, float %209), !dbg !26
  %231 = call float @llvm.maxnum.f32(float %230, float %210), !dbg !26
  %232 = call float @llvm.maxnum.f32(float %231, float %214), !dbg !26
  %233 = call float @llvm.maxnum.f32(float %232, float %215), !dbg !26
  %234 = call float @llvm.maxnum.f32(float %233, float %216), !dbg !26
  %235 = call float @llvm.maxnum.f32(float %234, float %217), !dbg !26
  %236 = call float @llvm.maxnum.f32(float %235, float %220), !dbg !26
  %237 = call float @llvm.maxnum.f32(float %236, float %221), !dbg !26
  %238 = call float @llvm.maxnum.f32(float %237, float %222), !dbg !26
  %239 = call float @llvm.maxnum.f32(float %238, float %223), !dbg !26
  %240 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %241 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %240), !dbg !26
  %242 = add nuw nsw i32 %241, 64, !dbg !26
  %243 = xor i32 %241, 16, !dbg !26
  %244 = and i32 %242, 192, !dbg !26
  %245 = icmp samesign ult i32 %243, %244, !dbg !26
  %246 = select i1 %245, i32 %243, i32 %241, !dbg !26
  %247 = shl nuw nsw i32 %246, 2, !dbg !26
  %248 = bitcast float %239 to i32, !dbg !26
  %249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %247, i32 %248), !dbg !26
  %250 = bitcast i32 %249 to float, !dbg !26
  %251 = call float @llvm.maxnum.f32(float %239, float %250), !dbg !26
  %252 = xor i32 %241, 32, !dbg !26
  %253 = icmp samesign ult i32 %252, %244, !dbg !26
  %254 = select i1 %253, i32 %252, i32 %241, !dbg !26
  %255 = shl nuw nsw i32 %254, 2, !dbg !26
  %256 = bitcast float %251 to i32, !dbg !26
  %257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %255, i32 %256), !dbg !26
  %258 = bitcast i32 %257 to float, !dbg !26
  %259 = call float @llvm.maxnum.f32(float %251, float %258), !dbg !26
  %260 = insertelement <1 x float> poison, float %259, i64 0, !dbg !26
  %261 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %260, <1 x float> %31), !dbg !26
  %262 = shufflevector <1 x float> %261, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %263 = fsub <4 x float> %199, %262, !dbg !27
  %264 = fsub <4 x float> %206, %262, !dbg !27
  %265 = fsub <4 x float> %213, %262, !dbg !27
  %266 = fsub <4 x float> %219, %262, !dbg !27
  %267 = fsub <1 x float> %31, %261, !dbg !28
  %268 = extractelement <1 x float> %267, i64 0, !dbg !29
  %269 = call float @llvm.amdgcn.exp2.f32(float %268), !dbg !29
  %270 = insertelement <4 x float> poison, float %269, i64 0, !dbg !34
  %271 = shufflevector <4 x float> %270, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %272 = fmul <4 x float> %39, %271, !dbg !34
  %273 = fmul <4 x float> %38, %271, !dbg !34
  %274 = fmul <4 x float> %36, %271, !dbg !34
  %275 = fmul <4 x float> %35, %271, !dbg !34
  %276 = fmul <4 x float> %33, %271, !dbg !34
  %277 = fmul <4 x float> %32, %271, !dbg !34
  %278 = fmul <4 x float> %28, %271, !dbg !34
  %279 = fmul <4 x float> %27, %271, !dbg !34
  %280 = extractelement <1 x float> %30, i64 0, !dbg !30
  %281 = fmul float %280, %269, !dbg !30
  %282 = extractelement <4 x float> %263, i64 0, !dbg !31
  %283 = call float @llvm.amdgcn.exp2.f32(float %282), !dbg !31
  %284 = insertelement <4 x float> poison, float %283, i64 0, !dbg !33
  %285 = extractelement <4 x float> %263, i64 1, !dbg !31
  %286 = call float @llvm.amdgcn.exp2.f32(float %285), !dbg !31
  %287 = insertelement <4 x float> %284, float %286, i64 1, !dbg !33
  %288 = fadd float %283, %286, !dbg !32
  %289 = extractelement <4 x float> %263, i64 2, !dbg !31
  %290 = call float @llvm.amdgcn.exp2.f32(float %289), !dbg !31
  %291 = insertelement <4 x float> %287, float %290, i64 2, !dbg !33
  %292 = fadd float %288, %290, !dbg !32
  %293 = extractelement <4 x float> %263, i64 3, !dbg !31
  %294 = call float @llvm.amdgcn.exp2.f32(float %293), !dbg !31
  %295 = insertelement <4 x float> %291, float %294, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %295) #9, !dbg !33, !frisk.register_lifetimes !1
  %296 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %296) #9, !dbg !32, !frisk.register_lifetimes !1
  %297 = fadd float %292, %294, !dbg !32
  %298 = extractelement <4 x float> %264, i64 0, !dbg !31
  %299 = call float @llvm.amdgcn.exp2.f32(float %298), !dbg !31
  %300 = insertelement <4 x float> poison, float %299, i64 0, !dbg !33
  %301 = fadd float %297, %299, !dbg !32
  %302 = extractelement <4 x float> %264, i64 1, !dbg !31
  %303 = call float @llvm.amdgcn.exp2.f32(float %302), !dbg !31
  %304 = insertelement <4 x float> %300, float %303, i64 1, !dbg !33
  %305 = fadd float %301, %303, !dbg !32
  %306 = extractelement <4 x float> %264, i64 2, !dbg !31
  %307 = call float @llvm.amdgcn.exp2.f32(float %306), !dbg !31
  %308 = insertelement <4 x float> %304, float %307, i64 2, !dbg !33
  %309 = fadd float %305, %307, !dbg !32
  %310 = extractelement <4 x float> %264, i64 3, !dbg !31
  %311 = call float @llvm.amdgcn.exp2.f32(float %310), !dbg !31
  %312 = insertelement <4 x float> %308, float %311, i64 3, !dbg !33
  %math.result36 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %312) #9, !dbg !33, !frisk.register_lifetimes !1
  %313 = fptrunc <4 x float> %math.result36 to <4 x half>, !dbg !33
  %math.packed37 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %313) #9, !dbg !32, !frisk.register_lifetimes !1
  %314 = fadd float %309, %311, !dbg !32
  %315 = extractelement <4 x float> %265, i64 0, !dbg !31
  %316 = call float @llvm.amdgcn.exp2.f32(float %315), !dbg !31
  %317 = insertelement <4 x float> poison, float %316, i64 0, !dbg !33
  %318 = fadd float %314, %316, !dbg !32
  %319 = extractelement <4 x float> %265, i64 1, !dbg !31
  %320 = call float @llvm.amdgcn.exp2.f32(float %319), !dbg !31
  %321 = insertelement <4 x float> %317, float %320, i64 1, !dbg !33
  %322 = fadd float %318, %320, !dbg !32
  %323 = extractelement <4 x float> %265, i64 2, !dbg !31
  %324 = call float @llvm.amdgcn.exp2.f32(float %323), !dbg !31
  %325 = insertelement <4 x float> %321, float %324, i64 2, !dbg !33
  %326 = fadd float %322, %324, !dbg !32
  %327 = extractelement <4 x float> %265, i64 3, !dbg !31
  %328 = call float @llvm.amdgcn.exp2.f32(float %327), !dbg !31
  %329 = insertelement <4 x float> %325, float %328, i64 3, !dbg !33
  %math.result38 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %329) #9, !dbg !33, !frisk.register_lifetimes !1
  %330 = fptrunc <4 x float> %math.result38 to <4 x half>, !dbg !33
  %math.packed39 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %330) #9, !dbg !32, !frisk.register_lifetimes !1
  %331 = fadd float %326, %328, !dbg !32
  %332 = extractelement <4 x float> %266, i64 0, !dbg !31
  %333 = call float @llvm.amdgcn.exp2.f32(float %332), !dbg !31
  %334 = insertelement <4 x float> poison, float %333, i64 0, !dbg !33
  %335 = fadd float %331, %333, !dbg !32
  %336 = extractelement <4 x float> %266, i64 1, !dbg !31
  %337 = call float @llvm.amdgcn.exp2.f32(float %336), !dbg !31
  %338 = insertelement <4 x float> %334, float %337, i64 1, !dbg !33
  %339 = fadd float %335, %337, !dbg !32
  %340 = extractelement <4 x float> %266, i64 2, !dbg !31
  %341 = call float @llvm.amdgcn.exp2.f32(float %340), !dbg !31
  %342 = insertelement <4 x float> %338, float %341, i64 2, !dbg !33
  %343 = fadd float %339, %341, !dbg !32
  %344 = extractelement <4 x float> %266, i64 3, !dbg !31
  %345 = call float @llvm.amdgcn.exp2.f32(float %344), !dbg !31
  %346 = insertelement <4 x float> %342, float %345, i64 3, !dbg !33
  %math.result40 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %346) #9, !dbg !33, !frisk.register_lifetimes !1
  %347 = fptrunc <4 x float> %math.result40 to <4 x half>, !dbg !33
  %math.packed41 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %347) #9, !dbg !32, !frisk.register_lifetimes !1
  %348 = fadd float %343, %345, !dbg !32
  %349 = bitcast float %348 to i32, !dbg !32
  %350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %247, i32 %349), !dbg !32
  %351 = bitcast i32 %350 to float, !dbg !32
  %352 = fadd float %348, %351, !dbg !32
  %353 = bitcast float %352 to i32, !dbg !32
  %354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %255, i32 %353), !dbg !32
  %355 = bitcast i32 %354 to float, !dbg !32
  %356 = fadd float %352, %355, !dbg !32
  %357 = fadd float %356, %281, !dbg !32
  %358 = insertelement <1 x float> poison, float %357, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread142 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local143 = zext nneg i32 %address.thread142 to i64, !dbg !13
  %address.local144 = and i64 %address.local143, 60, !dbg !15
  %address.local145 = add nuw nsw i64 %address.local144, %address.local143, !dbg !15
  %address.local146 = shl nuw nsw i64 %address.local145, 3, !dbg !15
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !15
  %address.local148 = and i64 %address.local143, 63, !dbg !13
  %address.local149 = lshr i64 %address.local148, 4, !dbg !15
  %address.local150 = mul nuw nsw i64 %address.local149, 384, !dbg !15
  %address.local151 = getelementptr [2 x i8], ptr addrspace(3) %address.local147, i64 %address.local150, !dbg !15
  %address.local152 = lshr i64 %address.local143, 2, !dbg !8
  %address.local153 = mul nsw i64 %address.local152, -32, !dbg !15
  %address.local154 = getelementptr [2 x i8], ptr addrspace(3) %address.local151, i64 %address.local153, !dbg !15
  %359 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local154, i32 0) #0, !dbg !15
  %360 = bitcast <4 x i32> %359 to <8 x half>, !dbg !15
  %address.local155 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !15
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) %address.local155, i64 %address.local150, !dbg !15
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) %address.local156, i64 %address.local153, !dbg !15
  %address.local158 = getelementptr i8, ptr addrspace(3) %address.local157, i64 256, !dbg !15
  %361 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local158, i32 0) #0, !dbg !15
  %362 = bitcast <4 x i32> %361 to <8 x half>, !dbg !15
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !15
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) %address.local159, i64 %address.local150, !dbg !15
  %address.local161 = getelementptr [2 x i8], ptr addrspace(3) %address.local160, i64 %address.local153, !dbg !15
  %address.local162 = getelementptr i8, ptr addrspace(3) %address.local161, i64 512, !dbg !15
  %363 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local162, i32 0) #0, !dbg !15
  %364 = bitcast <4 x i32> %363 to <8 x half>, !dbg !15
  %address.local163 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !15
  %address.local164 = getelementptr [2 x i8], ptr addrspace(3) %address.local163, i64 %address.local150, !dbg !15
  %address.local165 = getelementptr [2 x i8], ptr addrspace(3) %address.local164, i64 %address.local153, !dbg !15
  %address.local166 = getelementptr i8, ptr addrspace(3) %address.local165, i64 768, !dbg !15
  %365 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local166, i32 0) #0, !dbg !15
  %366 = bitcast <4 x i32> %365 to <8 x half>, !dbg !15
  %367 = add nsw i64 %43, %6, !dbg !17
  %368 = shl i64 %367, 19, !dbg !17
  %369 = and i64 %368, 16252928, !dbg !17
  %370 = add i64 %369, %130, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %371 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %370, !dbg !17
  %372 = getelementptr i8, ptr addrspace(1) %371, i64 64, !dbg !17
  %address.local167 = shl nuw nsw i64 %address.local143, 3, !dbg !12
  %address.local168 = and i64 %address.local167, 1536, !dbg !12
  %address.local169 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local168, !dbg !12
  %address.local170 = ptrtoint ptr addrspace(3) %address.local169 to i64, !dbg !12
  %address.local171 = trunc i64 %address.local170 to i32, !dbg !12
  %373 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local171), !dbg !17
  %374 = zext i32 %373 to i64, !dbg !17
  %375 = inttoptr i64 %374 to ptr addrspace(3), !dbg !17
  %address.local172 = shl nuw nsw i64 %address.local143, 5, !dbg !17
  %address.local173 = and i64 %address.local172, 8064, !dbg !17
  %address.local174 = shl nuw nsw i64 %address.local143, 3, !dbg !17
  %address.local175 = and i64 %address.local174, 24, !dbg !17
  %address.local176 = or disjoint i64 %address.local173, %address.local175, !dbg !17
  %address.local177 = getelementptr [2 x i8], ptr addrspace(1) %372, i64 %address.local176, !dbg !17
  %376 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %377 = ptrtoint ptr addrspace(1) %address.local177 to i64, !dbg !17
  %378 = sub i64 %377, %376, !dbg !17
  %dma.buffer.offset396 = trunc i64 %378 to i32, !dbg !17
  %379 = trunc i64 %376 to i32, !dbg !17
  %380 = insertelement <4 x i32> poison, i32 %379, i64 0, !dbg !17
  %381 = lshr i64 %376, 32, !dbg !17
  %382 = trunc i64 %381 to i32, !dbg !17
  %383 = insertelement <4 x i32> %380, i32 %382, i64 1, !dbg !17
  %384 = insertelement <4 x i32> %383, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc397 = insertelement <4 x i32> %384, i32 131072, i64 3, !dbg !17
  %385 = ptrtoint ptr addrspace(3) %375 to i32, !dbg !17
  %dma.buffer.lds398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %385), !dbg !17
  %386 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds398, i32 %dma.buffer.offset396, <4 x i32> %dma.buffer.rsrc397) #10, !dbg !17
  %387 = shufflevector <8 x half> %360, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %387, <4 x float> %279) #9, !dbg !15
  %389 = shufflevector <8 x half> %362, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %389, <4 x float> %388) #9, !dbg !15
  %391 = shufflevector <8 x half> %364, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed39, <4 x half> %391, <4 x float> %390) #9, !dbg !15
  %393 = shufflevector <8 x half> %366, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed41, <4 x half> %393, <4 x float> %392) #9, !dbg !15
  %395 = shufflevector <8 x half> %360, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %395, <4 x float> %278) #9, !dbg !15
  %397 = shufflevector <8 x half> %362, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %397, <4 x float> %396) #9, !dbg !15
  %399 = shufflevector <8 x half> %364, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed39, <4 x half> %399, <4 x float> %398) #9, !dbg !15
  %401 = shufflevector <8 x half> %366, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed41, <4 x half> %401, <4 x float> %400) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %403 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local154, i32 0) #0, !dbg !15
  %404 = bitcast <4 x i32> %403 to <8 x half>, !dbg !15
  %405 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local158, i32 0) #0, !dbg !15
  %406 = bitcast <4 x i32> %405 to <8 x half>, !dbg !15
  %407 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local162, i32 0) #0, !dbg !15
  %408 = bitcast <4 x i32> %407 to <8 x half>, !dbg !15
  %409 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local166, i32 0) #0, !dbg !15
  %410 = bitcast <4 x i32> %409 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %411 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %370, !dbg !17
  %412 = getelementptr i8, ptr addrspace(1) %411, i64 128, !dbg !17
  %413 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local171), !dbg !17
  %414 = zext i32 %413 to i64, !dbg !17
  %415 = inttoptr i64 %414 to ptr addrspace(3), !dbg !17
  %address.local178 = getelementptr [2 x i8], ptr addrspace(1) %412, i64 %address.local176, !dbg !17
  %416 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %417 = ptrtoint ptr addrspace(1) %address.local178 to i64, !dbg !17
  %418 = sub i64 %417, %416, !dbg !17
  %dma.buffer.offset393 = trunc i64 %418 to i32, !dbg !17
  %419 = trunc i64 %416 to i32, !dbg !17
  %420 = insertelement <4 x i32> poison, i32 %419, i64 0, !dbg !17
  %421 = lshr i64 %416, 32, !dbg !17
  %422 = trunc i64 %421 to i32, !dbg !17
  %423 = insertelement <4 x i32> %420, i32 %422, i64 1, !dbg !17
  %424 = insertelement <4 x i32> %423, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc394 = insertelement <4 x i32> %424, i32 131072, i64 3, !dbg !17
  %425 = ptrtoint ptr addrspace(3) %415 to i32, !dbg !17
  %dma.buffer.lds395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %425), !dbg !17
  %426 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds395, i32 %dma.buffer.offset393, <4 x i32> %dma.buffer.rsrc394) #10, !dbg !17
  %427 = shufflevector <8 x half> %404, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %427, <4 x float> %277) #9, !dbg !15
  %429 = shufflevector <8 x half> %406, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %429, <4 x float> %428) #9, !dbg !15
  %431 = shufflevector <8 x half> %408, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed39, <4 x half> %431, <4 x float> %430) #9, !dbg !15
  %433 = shufflevector <8 x half> %410, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed41, <4 x half> %433, <4 x float> %432) #9, !dbg !15
  %435 = shufflevector <8 x half> %404, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %435, <4 x float> %276) #9, !dbg !15
  %437 = shufflevector <8 x half> %406, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %437, <4 x float> %436) #9, !dbg !15
  %439 = shufflevector <8 x half> %408, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed39, <4 x half> %439, <4 x float> %438) #9, !dbg !15
  %441 = shufflevector <8 x half> %410, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed41, <4 x half> %441, <4 x float> %440) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %443 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local154, i32 0) #0, !dbg !15
  %444 = bitcast <4 x i32> %443 to <8 x half>, !dbg !15
  %445 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local158, i32 0) #0, !dbg !15
  %446 = bitcast <4 x i32> %445 to <8 x half>, !dbg !15
  %447 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local162, i32 0) #0, !dbg !15
  %448 = bitcast <4 x i32> %447 to <8 x half>, !dbg !15
  %449 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local166, i32 0) #0, !dbg !15
  %450 = bitcast <4 x i32> %449 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %451 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %370, !dbg !17
  %452 = getelementptr i8, ptr addrspace(1) %451, i64 192, !dbg !17
  %453 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local171), !dbg !17
  %454 = zext i32 %453 to i64, !dbg !17
  %455 = inttoptr i64 %454 to ptr addrspace(3), !dbg !17
  %address.local179 = getelementptr [2 x i8], ptr addrspace(1) %452, i64 %address.local176, !dbg !17
  %456 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %457 = ptrtoint ptr addrspace(1) %address.local179 to i64, !dbg !17
  %458 = sub i64 %457, %456, !dbg !17
  %dma.buffer.offset390 = trunc i64 %458 to i32, !dbg !17
  %459 = trunc i64 %456 to i32, !dbg !17
  %460 = insertelement <4 x i32> poison, i32 %459, i64 0, !dbg !17
  %461 = lshr i64 %456, 32, !dbg !17
  %462 = trunc i64 %461 to i32, !dbg !17
  %463 = insertelement <4 x i32> %460, i32 %462, i64 1, !dbg !17
  %464 = insertelement <4 x i32> %463, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc391 = insertelement <4 x i32> %464, i32 131072, i64 3, !dbg !17
  %465 = ptrtoint ptr addrspace(3) %455 to i32, !dbg !17
  %dma.buffer.lds392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %465), !dbg !17
  %466 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds392, i32 %dma.buffer.offset390, <4 x i32> %dma.buffer.rsrc391) #10, !dbg !17
  %467 = shufflevector <8 x half> %444, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %467, <4 x float> %275) #9, !dbg !15
  %469 = shufflevector <8 x half> %446, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %469, <4 x float> %468) #9, !dbg !15
  %471 = shufflevector <8 x half> %448, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed39, <4 x half> %471, <4 x float> %470) #9, !dbg !15
  %473 = shufflevector <8 x half> %450, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed41, <4 x half> %473, <4 x float> %472) #9, !dbg !15
  %475 = shufflevector <8 x half> %444, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %475, <4 x float> %274) #9, !dbg !15
  %477 = shufflevector <8 x half> %446, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %477, <4 x float> %476) #9, !dbg !15
  %479 = shufflevector <8 x half> %448, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed39, <4 x half> %479, <4 x float> %478) #9, !dbg !15
  %481 = shufflevector <8 x half> %450, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed41, <4 x half> %481, <4 x float> %480) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %483 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local154, i32 0) #0, !dbg !15
  %484 = bitcast <4 x i32> %483 to <8 x half>, !dbg !15
  %485 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local158, i32 0) #0, !dbg !15
  %486 = bitcast <4 x i32> %485 to <8 x half>, !dbg !15
  %487 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local162, i32 0) #0, !dbg !15
  %488 = bitcast <4 x i32> %487 to <8 x half>, !dbg !15
  %489 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local166, i32 0) #0, !dbg !15
  %490 = bitcast <4 x i32> %489 to <8 x half>, !dbg !15
  %491 = shufflevector <8 x half> %484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %491, <4 x float> %273) #9, !dbg !15
  %493 = shufflevector <8 x half> %486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %493, <4 x float> %492) #9, !dbg !15
  %495 = shufflevector <8 x half> %488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed39, <4 x half> %495, <4 x float> %494) #9, !dbg !15
  %497 = shufflevector <8 x half> %490, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed41, <4 x half> %497, <4 x float> %496) #9, !dbg !15
  %499 = shufflevector <8 x half> %484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %499, <4 x float> %272) #9, !dbg !15
  %501 = shufflevector <8 x half> %486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed37, <4 x half> %501, <4 x float> %500) #9, !dbg !15
  %503 = shufflevector <8 x half> %488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed39, <4 x half> %503, <4 x float> %502) #9, !dbg !15
  %505 = shufflevector <8 x half> %490, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed41, <4 x half> %505, <4 x float> %504) #9, !dbg !15
  %507 = shufflevector <4 x float> %394, <4 x float> %402, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %508 = shufflevector <4 x float> %434, <4 x float> %442, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %509 = shufflevector <4 x float> %474, <4 x float> %482, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %510 = shufflevector <4 x float> %498, <4 x float> %506, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %511 = add nuw nsw i64 %26, 1, !dbg !7
  br label %25, !dbg !7

512:                                              ; preds = %25
  %513 = and i32 %19, 16, !dbg !7
  %.not2 = icmp eq i32 %513, 0, !dbg !7
  %514 = and i32 %19, 32, !dbg !7
  %.not3 = icmp eq i32 %514, 0, !dbg !7
  %515 = extractelement <8 x float> %40, i64 0, !dbg !7
  %516 = extractelement <8 x float> %40, i64 1, !dbg !7
  %517 = extractelement <8 x float> %40, i64 2, !dbg !7
  %518 = extractelement <8 x float> %40, i64 3, !dbg !7
  %519 = select i1 %.not2, float %516, float %515, !dbg !7
  %520 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %521 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %520), !dbg !7
  %522 = add nuw nsw i32 %521, 64, !dbg !7
  %523 = xor i32 %521, 16, !dbg !7
  %524 = and i32 %522, 192, !dbg !7
  %525 = icmp samesign ult i32 %523, %524, !dbg !7
  %526 = select i1 %525, i32 %523, i32 %521, !dbg !7
  %527 = shl nuw nsw i32 %526, 2, !dbg !7
  %528 = bitcast float %519 to i32, !dbg !7
  %529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %528), !dbg !7
  %530 = bitcast i32 %529 to float, !dbg !7
  %531 = select i1 %.not2, float %515, float %530, !dbg !7
  %532 = select i1 %.not2, float %530, float %516, !dbg !7
  %533 = select i1 %.not2, float %518, float %517, !dbg !7
  %534 = bitcast float %533 to i32, !dbg !7
  %535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %534), !dbg !7
  %536 = bitcast i32 %535 to float, !dbg !7
  %537 = select i1 %.not2, float %517, float %536, !dbg !7
  %538 = select i1 %.not2, float %536, float %518, !dbg !7
  %539 = select i1 %.not3, float %537, float %531, !dbg !7
  %540 = xor i32 %521, 32, !dbg !7
  %541 = icmp samesign ult i32 %540, %524, !dbg !7
  %542 = select i1 %541, i32 %540, i32 %521, !dbg !7
  %543 = shl nuw nsw i32 %542, 2, !dbg !7
  %544 = bitcast float %539 to i32, !dbg !7
  %545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %544), !dbg !7
  %546 = bitcast i32 %545 to float, !dbg !7
  %547 = select i1 %.not3, float %531, float %546, !dbg !7
  %548 = select i1 %.not3, float %546, float %537, !dbg !7
  %549 = select i1 %.not3, float %538, float %532, !dbg !7
  %550 = bitcast float %549 to i32, !dbg !7
  %551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %550), !dbg !7
  %552 = bitcast i32 %551 to float, !dbg !7
  %553 = select i1 %.not3, float %532, float %552, !dbg !7
  %554 = select i1 %.not3, float %552, float %538, !dbg !7
  %555 = extractelement <8 x float> %37, i64 0, !dbg !7
  %556 = extractelement <8 x float> %37, i64 1, !dbg !7
  %557 = extractelement <8 x float> %37, i64 2, !dbg !7
  %558 = extractelement <8 x float> %37, i64 3, !dbg !7
  %559 = select i1 %.not2, float %556, float %555, !dbg !7
  %560 = bitcast float %559 to i32, !dbg !7
  %561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %560), !dbg !7
  %562 = bitcast i32 %561 to float, !dbg !7
  %563 = select i1 %.not2, float %555, float %562, !dbg !7
  %564 = select i1 %.not2, float %562, float %556, !dbg !7
  %565 = select i1 %.not2, float %558, float %557, !dbg !7
  %566 = bitcast float %565 to i32, !dbg !7
  %567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %566), !dbg !7
  %568 = bitcast i32 %567 to float, !dbg !7
  %569 = select i1 %.not2, float %557, float %568, !dbg !7
  %570 = select i1 %.not2, float %568, float %558, !dbg !7
  %571 = select i1 %.not3, float %569, float %563, !dbg !7
  %572 = bitcast float %571 to i32, !dbg !7
  %573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %572), !dbg !7
  %574 = bitcast i32 %573 to float, !dbg !7
  %575 = select i1 %.not3, float %563, float %574, !dbg !7
  %576 = select i1 %.not3, float %574, float %569, !dbg !7
  %577 = select i1 %.not3, float %570, float %564, !dbg !7
  %578 = bitcast float %577 to i32, !dbg !7
  %579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %578), !dbg !7
  %580 = bitcast i32 %579 to float, !dbg !7
  %581 = select i1 %.not3, float %564, float %580, !dbg !7
  %582 = select i1 %.not3, float %580, float %570, !dbg !7
  %583 = extractelement <8 x float> %34, i64 0, !dbg !7
  %584 = extractelement <8 x float> %34, i64 1, !dbg !7
  %585 = extractelement <8 x float> %34, i64 2, !dbg !7
  %586 = extractelement <8 x float> %34, i64 3, !dbg !7
  %587 = select i1 %.not2, float %584, float %583, !dbg !7
  %588 = bitcast float %587 to i32, !dbg !7
  %589 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %588), !dbg !7
  %590 = bitcast i32 %589 to float, !dbg !7
  %591 = select i1 %.not2, float %583, float %590, !dbg !7
  %592 = select i1 %.not2, float %590, float %584, !dbg !7
  %593 = select i1 %.not2, float %586, float %585, !dbg !7
  %594 = bitcast float %593 to i32, !dbg !7
  %595 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %594), !dbg !7
  %596 = bitcast i32 %595 to float, !dbg !7
  %597 = select i1 %.not2, float %585, float %596, !dbg !7
  %598 = select i1 %.not2, float %596, float %586, !dbg !7
  %599 = select i1 %.not3, float %597, float %591, !dbg !7
  %600 = bitcast float %599 to i32, !dbg !7
  %601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %600), !dbg !7
  %602 = bitcast i32 %601 to float, !dbg !7
  %603 = select i1 %.not3, float %591, float %602, !dbg !7
  %604 = select i1 %.not3, float %602, float %597, !dbg !7
  %605 = select i1 %.not3, float %598, float %592, !dbg !7
  %606 = bitcast float %605 to i32, !dbg !7
  %607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %606), !dbg !7
  %608 = bitcast i32 %607 to float, !dbg !7
  %609 = select i1 %.not3, float %592, float %608, !dbg !7
  %610 = select i1 %.not3, float %608, float %598, !dbg !7
  %611 = extractelement <8 x float> %29, i64 0, !dbg !7
  %612 = extractelement <8 x float> %29, i64 1, !dbg !7
  %613 = extractelement <8 x float> %29, i64 2, !dbg !7
  %614 = extractelement <8 x float> %29, i64 3, !dbg !7
  %615 = select i1 %.not2, float %612, float %611, !dbg !7
  %616 = bitcast float %615 to i32, !dbg !7
  %617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %616), !dbg !7
  %618 = bitcast i32 %617 to float, !dbg !7
  %619 = select i1 %.not2, float %611, float %618, !dbg !7
  %620 = select i1 %.not2, float %618, float %612, !dbg !7
  %621 = select i1 %.not2, float %614, float %613, !dbg !7
  %622 = bitcast float %621 to i32, !dbg !7
  %623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %622), !dbg !7
  %624 = bitcast i32 %623 to float, !dbg !7
  %625 = select i1 %.not2, float %613, float %624, !dbg !7
  %626 = select i1 %.not2, float %624, float %614, !dbg !7
  %627 = select i1 %.not3, float %625, float %619, !dbg !7
  %628 = bitcast float %627 to i32, !dbg !7
  %629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %628), !dbg !7
  %630 = select i1 %.not3, float %626, float %620, !dbg !7
  %631 = bitcast float %630 to i32, !dbg !7
  %632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %631), !dbg !7
  %633 = shufflevector <1 x float> %30, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %634 = extractelement <8 x float> %40, i64 4, !dbg !7
  %635 = extractelement <8 x float> %40, i64 5, !dbg !7
  %636 = extractelement <8 x float> %40, i64 6, !dbg !7
  %637 = extractelement <8 x float> %40, i64 7, !dbg !7
  %638 = select i1 %.not2, float %635, float %634, !dbg !7
  %639 = bitcast float %638 to i32, !dbg !7
  %640 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %639), !dbg !7
  %641 = bitcast i32 %640 to float, !dbg !7
  %642 = select i1 %.not2, float %634, float %641, !dbg !7
  %643 = select i1 %.not2, float %641, float %635, !dbg !7
  %644 = select i1 %.not2, float %637, float %636, !dbg !7
  %645 = bitcast float %644 to i32, !dbg !7
  %646 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %645), !dbg !7
  %647 = bitcast i32 %646 to float, !dbg !7
  %648 = select i1 %.not2, float %636, float %647, !dbg !7
  %649 = select i1 %.not2, float %647, float %637, !dbg !7
  %650 = select i1 %.not3, float %648, float %642, !dbg !7
  %651 = bitcast float %650 to i32, !dbg !7
  %652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %651), !dbg !7
  %653 = bitcast i32 %652 to float, !dbg !7
  %654 = select i1 %.not3, float %642, float %653, !dbg !7
  %655 = select i1 %.not3, float %653, float %648, !dbg !7
  %656 = select i1 %.not3, float %649, float %643, !dbg !7
  %657 = bitcast float %656 to i32, !dbg !7
  %658 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %657), !dbg !7
  %659 = bitcast i32 %658 to float, !dbg !7
  %660 = select i1 %.not3, float %643, float %659, !dbg !7
  %661 = select i1 %.not3, float %659, float %649, !dbg !7
  %662 = extractelement <8 x float> %37, i64 4, !dbg !7
  %663 = extractelement <8 x float> %37, i64 5, !dbg !7
  %664 = extractelement <8 x float> %37, i64 6, !dbg !7
  %665 = extractelement <8 x float> %37, i64 7, !dbg !7
  %666 = select i1 %.not2, float %663, float %662, !dbg !7
  %667 = bitcast float %666 to i32, !dbg !7
  %668 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %667), !dbg !7
  %669 = bitcast i32 %668 to float, !dbg !7
  %670 = select i1 %.not2, float %662, float %669, !dbg !7
  %671 = select i1 %.not2, float %669, float %663, !dbg !7
  %672 = select i1 %.not2, float %665, float %664, !dbg !7
  %673 = bitcast float %672 to i32, !dbg !7
  %674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %673), !dbg !7
  %675 = bitcast i32 %674 to float, !dbg !7
  %676 = select i1 %.not2, float %664, float %675, !dbg !7
  %677 = select i1 %.not2, float %675, float %665, !dbg !7
  %678 = select i1 %.not3, float %676, float %670, !dbg !7
  %679 = bitcast float %678 to i32, !dbg !7
  %680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %679), !dbg !7
  %681 = bitcast i32 %680 to float, !dbg !7
  %682 = select i1 %.not3, float %670, float %681, !dbg !7
  %683 = select i1 %.not3, float %681, float %676, !dbg !7
  %684 = select i1 %.not3, float %677, float %671, !dbg !7
  %685 = bitcast float %684 to i32, !dbg !7
  %686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %685), !dbg !7
  %687 = bitcast i32 %686 to float, !dbg !7
  %688 = select i1 %.not3, float %671, float %687, !dbg !7
  %689 = select i1 %.not3, float %687, float %677, !dbg !7
  %690 = extractelement <8 x float> %34, i64 4, !dbg !7
  %691 = extractelement <8 x float> %34, i64 5, !dbg !7
  %692 = extractelement <8 x float> %34, i64 6, !dbg !7
  %693 = extractelement <8 x float> %34, i64 7, !dbg !7
  %694 = select i1 %.not2, float %691, float %690, !dbg !7
  %695 = bitcast float %694 to i32, !dbg !7
  %696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %695), !dbg !7
  %697 = bitcast i32 %696 to float, !dbg !7
  %698 = select i1 %.not2, float %690, float %697, !dbg !7
  %699 = select i1 %.not2, float %697, float %691, !dbg !7
  %700 = select i1 %.not2, float %693, float %692, !dbg !7
  %701 = bitcast float %700 to i32, !dbg !7
  %702 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %701), !dbg !7
  %703 = bitcast i32 %702 to float, !dbg !7
  %704 = select i1 %.not2, float %692, float %703, !dbg !7
  %705 = select i1 %.not2, float %703, float %693, !dbg !7
  %706 = select i1 %.not3, float %704, float %698, !dbg !7
  %707 = bitcast float %706 to i32, !dbg !7
  %708 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %707), !dbg !7
  %709 = bitcast i32 %708 to float, !dbg !7
  %710 = select i1 %.not3, float %698, float %709, !dbg !7
  %711 = select i1 %.not3, float %709, float %704, !dbg !7
  %712 = select i1 %.not3, float %705, float %699, !dbg !7
  %713 = bitcast float %712 to i32, !dbg !7
  %714 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %713), !dbg !7
  %715 = bitcast i32 %714 to float, !dbg !7
  %716 = select i1 %.not3, float %699, float %715, !dbg !7
  %717 = select i1 %.not3, float %715, float %705, !dbg !7
  %718 = extractelement <8 x float> %29, i64 4, !dbg !7
  %719 = extractelement <8 x float> %29, i64 5, !dbg !7
  %720 = extractelement <8 x float> %29, i64 6, !dbg !7
  %721 = extractelement <8 x float> %29, i64 7, !dbg !7
  %722 = select i1 %.not2, float %719, float %718, !dbg !7
  %723 = bitcast float %722 to i32, !dbg !7
  %724 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %723), !dbg !7
  %725 = bitcast i32 %724 to float, !dbg !7
  %726 = select i1 %.not2, float %718, float %725, !dbg !7
  %727 = select i1 %.not2, float %725, float %719, !dbg !7
  %728 = select i1 %.not2, float %721, float %720, !dbg !7
  %729 = bitcast float %728 to i32, !dbg !7
  %730 = call i32 @llvm.amdgcn.ds.bpermute(i32 %527, i32 %729), !dbg !7
  %731 = bitcast i32 %730 to float, !dbg !7
  %732 = select i1 %.not2, float %720, float %731, !dbg !7
  %733 = select i1 %.not2, float %731, float %721, !dbg !7
  %734 = select i1 %.not3, float %732, float %726, !dbg !7
  %735 = bitcast float %734 to i32, !dbg !7
  %736 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %735), !dbg !7
  %737 = select i1 %.not3, float %733, float %727, !dbg !7
  %738 = bitcast float %737 to i32, !dbg !7
  %739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %738), !dbg !7
  br i1 true, label %740, label %774, !dbg !20

740:                                              ; preds = %512
  %741 = bitcast i32 %736 to float, !dbg !7
  %742 = select i1 %.not3, float %726, float %741, !dbg !7
  %743 = insertelement <4 x float> poison, float %742, i64 0, !dbg !18
  %744 = bitcast i32 %739 to float, !dbg !7
  %745 = select i1 %.not3, float %727, float %744, !dbg !7
  %746 = insertelement <4 x float> %743, float %745, i64 1, !dbg !18
  %747 = select i1 %.not3, float %741, float %732, !dbg !7
  %748 = insertelement <4 x float> %746, float %747, i64 2, !dbg !18
  %749 = select i1 %.not3, float %744, float %733, !dbg !7
  %750 = insertelement <4 x float> %748, float %749, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %633, !dbg !18
  %normalized = fmul <4 x float> %750, %row.reciprocal, !dbg !18
  %751 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %752 = extractelement <4 x half> %751, i64 3, !dbg !19
  %753 = extractelement <4 x half> %751, i64 2, !dbg !19
  %754 = extractelement <4 x half> %751, i64 1, !dbg !19
  %755 = extractelement <4 x half> %751, i64 0, !dbg !19
  %756 = bitcast i32 %629 to float, !dbg !7
  %757 = select i1 %.not3, float %619, float %756, !dbg !7
  %758 = insertelement <4 x float> poison, float %757, i64 0, !dbg !18
  %759 = bitcast i32 %632 to float, !dbg !7
  %760 = select i1 %.not3, float %620, float %759, !dbg !7
  %761 = insertelement <4 x float> %758, float %760, i64 1, !dbg !18
  %762 = select i1 %.not3, float %756, float %625, !dbg !7
  %763 = insertelement <4 x float> %761, float %762, i64 2, !dbg !18
  %764 = select i1 %.not3, float %759, float %626, !dbg !7
  %765 = insertelement <4 x float> %763, float %764, i64 3, !dbg !18
  %normalized24 = fmul <4 x float> %765, %row.reciprocal, !dbg !18
  %766 = fptrunc <4 x float> %normalized24 to <4 x half>, !dbg !19
  %767 = extractelement <4 x half> %766, i64 3, !dbg !19
  %768 = extractelement <4 x half> %766, i64 2, !dbg !19
  %769 = extractelement <4 x half> %766, i64 1, !dbg !19
  %770 = extractelement <4 x half> %766, i64 0, !dbg !19
  %771 = add nsw i64 %6, %13, !dbg !20
  %772 = shl nsw i64 %771, 19, !dbg !20
  %773 = and i64 %772, 16252928, !dbg !20
  %address.thread180 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local181 = zext nneg i32 %address.thread180 to i64, !dbg !13
  %address.local182 = add nsw i64 %16, %address.local181, !dbg !20
  %address.local183 = lshr i64 %address.local181, 6, !dbg !13
  %address.local184 = shl nuw nsw i64 %address.local183, 4, !dbg !20
  %address.local185 = add nsw i64 %address.local182, %address.local184, !dbg !20
  %address.local186 = and i64 %address.local181, 240, !dbg !20
  %address.local187 = sub nsw i64 %address.local185, %address.local186, !dbg !20
  %address.local188 = add nsw i64 %address.local187, %17, !dbg !20
  %address.local189 = shl nuw nsw i64 %address.local188, 7, !dbg !20
  %address.local190 = add nuw nsw i64 %773, %address.local189, !dbg !20
  %address.local191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local190, !dbg !20
  %address.local192 = lshr i64 %address.local181, 2, !dbg !13
  %address.local193 = and i64 %address.local192, 12, !dbg !13
  %address.local194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local191, i64 %address.local193, !dbg !20
  store half %770, ptr addrspace(1) %address.local194, align 2, !dbg !20
  %address.local195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local190, !dbg !20
  %address.local196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local195, i64 %address.local193, !dbg !20
  %address.local197 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local196, i64 2, !dbg !20
  store half %769, ptr addrspace(1) %address.local197, align 2, !dbg !20
  %address.local198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local190, !dbg !20
  %address.local199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local198, i64 %address.local193, !dbg !20
  %address.local200 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local199, i64 4, !dbg !20
  store half %768, ptr addrspace(1) %address.local200, align 2, !dbg !20
  %address.local201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local190, !dbg !20
  %address.local202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local201, i64 %address.local193, !dbg !20
  %address.local203 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local202, i64 6, !dbg !20
  store half %767, ptr addrspace(1) %address.local203, align 2, !dbg !20
  %address.local204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local190, !dbg !20
  %address.local205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local204, i64 %address.local193, !dbg !20
  %address.local206 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local205, i64 32, !dbg !20
  store half %755, ptr addrspace(1) %address.local206, align 2, !dbg !20
  %address.local207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local190, !dbg !20
  %address.local208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local207, i64 %address.local193, !dbg !20
  %address.local209 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local208, i64 34, !dbg !20
  store half %754, ptr addrspace(1) %address.local209, align 2, !dbg !20
  %address.local210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local190, !dbg !20
  %address.local211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local210, i64 %address.local193, !dbg !20
  %address.local212 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local211, i64 36, !dbg !20
  store half %753, ptr addrspace(1) %address.local212, align 2, !dbg !20
  %address.local213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local190, !dbg !20
  %address.local214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local213, i64 %address.local193, !dbg !20
  %address.local215 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local214, i64 38, !dbg !20
  store half %752, ptr addrspace(1) %address.local215, align 2, !dbg !20
  br label %774, !dbg !20

774:                                              ; preds = %740, %512
  br i1 true, label %775, label %795, !dbg !20

775:                                              ; preds = %774
  %776 = insertelement <4 x float> poison, float %710, i64 0, !dbg !18
  %777 = insertelement <4 x float> %776, float %716, i64 1, !dbg !18
  %778 = insertelement <4 x float> %777, float %711, i64 2, !dbg !18
  %779 = insertelement <4 x float> %778, float %717, i64 3, !dbg !18
  %row.reciprocal25 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %633, !dbg !18
  %normalized26 = fmul <4 x float> %779, %row.reciprocal25, !dbg !18
  %780 = fptrunc <4 x float> %normalized26 to <4 x half>, !dbg !19
  %781 = extractelement <4 x half> %780, i64 3, !dbg !19
  %782 = extractelement <4 x half> %780, i64 2, !dbg !19
  %783 = extractelement <4 x half> %780, i64 1, !dbg !19
  %784 = extractelement <4 x half> %780, i64 0, !dbg !19
  %785 = insertelement <4 x float> poison, float %603, i64 0, !dbg !18
  %786 = insertelement <4 x float> %785, float %609, i64 1, !dbg !18
  %787 = insertelement <4 x float> %786, float %604, i64 2, !dbg !18
  %788 = insertelement <4 x float> %787, float %610, i64 3, !dbg !18
  %normalized27 = fmul <4 x float> %788, %row.reciprocal25, !dbg !18
  %789 = fptrunc <4 x float> %normalized27 to <4 x half>, !dbg !19
  %790 = extractelement <4 x half> %789, i64 3, !dbg !19
  %791 = extractelement <4 x half> %789, i64 2, !dbg !19
  %792 = extractelement <4 x half> %789, i64 1, !dbg !19
  %793 = extractelement <4 x half> %789, i64 0, !dbg !19
  %794 = shl nuw nsw i64 %15, 19, !dbg !20
  %address.thread216 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local217 = zext nneg i32 %address.thread216 to i64, !dbg !13
  %address.local218 = add nsw i64 %16, %address.local217, !dbg !20
  %address.local219 = lshr i64 %address.local217, 6, !dbg !13
  %address.local220 = shl nuw nsw i64 %address.local219, 4, !dbg !20
  %address.local221 = add nsw i64 %address.local218, %address.local220, !dbg !20
  %address.local222 = and i64 %address.local217, 240, !dbg !20
  %address.local223 = sub nsw i64 %address.local221, %address.local222, !dbg !20
  %address.local224 = add nsw i64 %address.local223, %17, !dbg !20
  %address.local225 = shl nuw nsw i64 %address.local224, 7, !dbg !20
  %address.local226 = add nuw nsw i64 %794, %address.local225, !dbg !20
  %address.local227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local226, !dbg !20
  %address.local228 = lshr i64 %address.local217, 2, !dbg !13
  %address.local229 = and i64 %address.local228, 12, !dbg !13
  %address.local230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local227, i64 %address.local229, !dbg !20
  %address.local231 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local230, i64 64, !dbg !20
  store half %793, ptr addrspace(1) %address.local231, align 2, !dbg !20
  %address.local232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local226, !dbg !20
  %address.local233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local232, i64 %address.local229, !dbg !20
  %address.local234 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local233, i64 66, !dbg !20
  store half %792, ptr addrspace(1) %address.local234, align 2, !dbg !20
  %address.local235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local226, !dbg !20
  %address.local236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local235, i64 %address.local229, !dbg !20
  %address.local237 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local236, i64 68, !dbg !20
  store half %791, ptr addrspace(1) %address.local237, align 2, !dbg !20
  %address.local238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local226, !dbg !20
  %address.local239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local238, i64 %address.local229, !dbg !20
  %address.local240 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local239, i64 70, !dbg !20
  store half %790, ptr addrspace(1) %address.local240, align 2, !dbg !20
  %address.local241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local226, !dbg !20
  %address.local242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local241, i64 %address.local229, !dbg !20
  %address.local243 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local242, i64 96, !dbg !20
  store half %784, ptr addrspace(1) %address.local243, align 2, !dbg !20
  %address.local244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local226, !dbg !20
  %address.local245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local244, i64 %address.local229, !dbg !20
  %address.local246 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local245, i64 98, !dbg !20
  store half %783, ptr addrspace(1) %address.local246, align 2, !dbg !20
  %address.local247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local226, !dbg !20
  %address.local248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local247, i64 %address.local229, !dbg !20
  %address.local249 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local248, i64 100, !dbg !20
  store half %782, ptr addrspace(1) %address.local249, align 2, !dbg !20
  %address.local250 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local226, !dbg !20
  %address.local251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local250, i64 %address.local229, !dbg !20
  %address.local252 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local251, i64 102, !dbg !20
  store half %781, ptr addrspace(1) %address.local252, align 2, !dbg !20
  br label %795, !dbg !20

795:                                              ; preds = %775, %774
  br i1 true, label %796, label %816, !dbg !20

796:                                              ; preds = %795
  %797 = insertelement <4 x float> poison, float %682, i64 0, !dbg !18
  %798 = insertelement <4 x float> %797, float %688, i64 1, !dbg !18
  %799 = insertelement <4 x float> %798, float %683, i64 2, !dbg !18
  %800 = insertelement <4 x float> %799, float %689, i64 3, !dbg !18
  %row.reciprocal28 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %633, !dbg !18
  %normalized29 = fmul <4 x float> %800, %row.reciprocal28, !dbg !18
  %801 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !19
  %802 = extractelement <4 x half> %801, i64 3, !dbg !19
  %803 = extractelement <4 x half> %801, i64 2, !dbg !19
  %804 = extractelement <4 x half> %801, i64 1, !dbg !19
  %805 = extractelement <4 x half> %801, i64 0, !dbg !19
  %806 = insertelement <4 x float> poison, float %575, i64 0, !dbg !18
  %807 = insertelement <4 x float> %806, float %581, i64 1, !dbg !18
  %808 = insertelement <4 x float> %807, float %576, i64 2, !dbg !18
  %809 = insertelement <4 x float> %808, float %582, i64 3, !dbg !18
  %normalized30 = fmul <4 x float> %809, %row.reciprocal28, !dbg !18
  %810 = fptrunc <4 x float> %normalized30 to <4 x half>, !dbg !19
  %811 = extractelement <4 x half> %810, i64 3, !dbg !19
  %812 = extractelement <4 x half> %810, i64 2, !dbg !19
  %813 = extractelement <4 x half> %810, i64 1, !dbg !19
  %814 = extractelement <4 x half> %810, i64 0, !dbg !19
  %815 = shl nuw nsw i64 %15, 19, !dbg !20
  %address.thread253 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local254 = zext nneg i32 %address.thread253 to i64, !dbg !13
  %address.local255 = add nsw i64 %16, %address.local254, !dbg !20
  %address.local256 = lshr i64 %address.local254, 6, !dbg !13
  %address.local257 = shl nuw nsw i64 %address.local256, 4, !dbg !20
  %address.local258 = add nsw i64 %address.local255, %address.local257, !dbg !20
  %address.local259 = and i64 %address.local254, 240, !dbg !20
  %address.local260 = sub nsw i64 %address.local258, %address.local259, !dbg !20
  %address.local261 = add nsw i64 %address.local260, %17, !dbg !20
  %address.local262 = shl nuw nsw i64 %address.local261, 7, !dbg !20
  %address.local263 = add nuw nsw i64 %815, %address.local262, !dbg !20
  %address.local264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local263, !dbg !20
  %address.local265 = lshr i64 %address.local254, 2, !dbg !13
  %address.local266 = and i64 %address.local265, 12, !dbg !13
  %address.local267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local264, i64 %address.local266, !dbg !20
  %address.local268 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local267, i64 128, !dbg !20
  store half %814, ptr addrspace(1) %address.local268, align 2, !dbg !20
  %address.local269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local263, !dbg !20
  %address.local270 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local269, i64 %address.local266, !dbg !20
  %address.local271 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local270, i64 130, !dbg !20
  store half %813, ptr addrspace(1) %address.local271, align 2, !dbg !20
  %address.local272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local263, !dbg !20
  %address.local273 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local272, i64 %address.local266, !dbg !20
  %address.local274 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local273, i64 132, !dbg !20
  store half %812, ptr addrspace(1) %address.local274, align 2, !dbg !20
  %address.local275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local263, !dbg !20
  %address.local276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local275, i64 %address.local266, !dbg !20
  %address.local277 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local276, i64 134, !dbg !20
  store half %811, ptr addrspace(1) %address.local277, align 2, !dbg !20
  %address.local278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local263, !dbg !20
  %address.local279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local278, i64 %address.local266, !dbg !20
  %address.local280 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local279, i64 160, !dbg !20
  store half %805, ptr addrspace(1) %address.local280, align 2, !dbg !20
  %address.local281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local263, !dbg !20
  %address.local282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local281, i64 %address.local266, !dbg !20
  %address.local283 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local282, i64 162, !dbg !20
  store half %804, ptr addrspace(1) %address.local283, align 2, !dbg !20
  %address.local284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local263, !dbg !20
  %address.local285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local284, i64 %address.local266, !dbg !20
  %address.local286 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local285, i64 164, !dbg !20
  store half %803, ptr addrspace(1) %address.local286, align 2, !dbg !20
  %address.local287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local263, !dbg !20
  %address.local288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local287, i64 %address.local266, !dbg !20
  %address.local289 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local288, i64 166, !dbg !20
  store half %802, ptr addrspace(1) %address.local289, align 2, !dbg !20
  br label %816, !dbg !20

816:                                              ; preds = %796, %795
  br i1 true, label %817, label %837, !dbg !20

817:                                              ; preds = %816
  %818 = insertelement <4 x float> poison, float %654, i64 0, !dbg !18
  %819 = insertelement <4 x float> %818, float %660, i64 1, !dbg !18
  %820 = insertelement <4 x float> %819, float %655, i64 2, !dbg !18
  %821 = insertelement <4 x float> %820, float %661, i64 3, !dbg !18
  %row.reciprocal31 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %633, !dbg !18
  %normalized32 = fmul <4 x float> %821, %row.reciprocal31, !dbg !18
  %822 = fptrunc <4 x float> %normalized32 to <4 x half>, !dbg !19
  %823 = extractelement <4 x half> %822, i64 3, !dbg !19
  %824 = extractelement <4 x half> %822, i64 2, !dbg !19
  %825 = extractelement <4 x half> %822, i64 1, !dbg !19
  %826 = extractelement <4 x half> %822, i64 0, !dbg !19
  %827 = insertelement <4 x float> poison, float %547, i64 0, !dbg !18
  %828 = insertelement <4 x float> %827, float %553, i64 1, !dbg !18
  %829 = insertelement <4 x float> %828, float %548, i64 2, !dbg !18
  %830 = insertelement <4 x float> %829, float %554, i64 3, !dbg !18
  %normalized33 = fmul <4 x float> %830, %row.reciprocal31, !dbg !18
  %831 = fptrunc <4 x float> %normalized33 to <4 x half>, !dbg !19
  %832 = extractelement <4 x half> %831, i64 3, !dbg !19
  %833 = extractelement <4 x half> %831, i64 2, !dbg !19
  %834 = extractelement <4 x half> %831, i64 1, !dbg !19
  %835 = extractelement <4 x half> %831, i64 0, !dbg !19
  %836 = shl nuw nsw i64 %15, 19, !dbg !20
  %address.thread290 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local291 = zext nneg i32 %address.thread290 to i64, !dbg !13
  %address.local292 = add nsw i64 %16, %address.local291, !dbg !20
  %address.local293 = lshr i64 %address.local291, 6, !dbg !13
  %address.local294 = shl nuw nsw i64 %address.local293, 4, !dbg !20
  %address.local295 = add nsw i64 %address.local292, %address.local294, !dbg !20
  %address.local296 = and i64 %address.local291, 240, !dbg !20
  %address.local297 = sub nsw i64 %address.local295, %address.local296, !dbg !20
  %address.local298 = add nsw i64 %address.local297, %17, !dbg !20
  %address.local299 = shl nuw nsw i64 %address.local298, 7, !dbg !20
  %address.local300 = add nuw nsw i64 %836, %address.local299, !dbg !20
  %address.local301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local300, !dbg !20
  %address.local302 = lshr i64 %address.local291, 2, !dbg !13
  %address.local303 = and i64 %address.local302, 12, !dbg !13
  %address.local304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local301, i64 %address.local303, !dbg !20
  %address.local305 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local304, i64 192, !dbg !20
  store half %835, ptr addrspace(1) %address.local305, align 2, !dbg !20
  %address.local306 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local300, !dbg !20
  %address.local307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local306, i64 %address.local303, !dbg !20
  %address.local308 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local307, i64 194, !dbg !20
  store half %834, ptr addrspace(1) %address.local308, align 2, !dbg !20
  %address.local309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local300, !dbg !20
  %address.local310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local309, i64 %address.local303, !dbg !20
  %address.local311 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local310, i64 196, !dbg !20
  store half %833, ptr addrspace(1) %address.local311, align 2, !dbg !20
  %address.local312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local300, !dbg !20
  %address.local313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local312, i64 %address.local303, !dbg !20
  %address.local314 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local313, i64 198, !dbg !20
  store half %832, ptr addrspace(1) %address.local314, align 2, !dbg !20
  %address.local315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local300, !dbg !20
  %address.local316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local315, i64 %address.local303, !dbg !20
  %address.local317 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local316, i64 224, !dbg !20
  store half %826, ptr addrspace(1) %address.local317, align 2, !dbg !20
  %address.local318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local300, !dbg !20
  %address.local319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local318, i64 %address.local303, !dbg !20
  %address.local320 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local319, i64 226, !dbg !20
  store half %825, ptr addrspace(1) %address.local320, align 2, !dbg !20
  %address.local321 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local300, !dbg !20
  %address.local322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local321, i64 %address.local303, !dbg !20
  %address.local323 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local322, i64 228, !dbg !20
  store half %824, ptr addrspace(1) %address.local323, align 2, !dbg !20
  %address.local324 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local300, !dbg !20
  %address.local325 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local324, i64 %address.local303, !dbg !20
  %address.local326 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local325, i64 230, !dbg !20
  store half %823, ptr addrspace(1) %address.local326, align 2, !dbg !20
  br label %837, !dbg !20

837:                                              ; preds = %817, %816
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

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/nohint-rolled")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 73, column: 26, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!12 = !DILocation(line: 70, column: 22, scope: !4)
!13 = !DILocation(line: 72, column: 24, scope: !4)
!14 = !DILocation(line: 77, column: 19, scope: !4)
!15 = !DILocation(line: 102, column: 23, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
!17 = !DILocation(line: 101, column: 20, scope: !4)
!18 = !DILocation(line: 107, column: 23, scope: !4)
!19 = !DILocation(line: 108, column: 19, scope: !4)
!20 = !DILocation(line: 109, column: 9, scope: !4)
!21 = !DILocation(line: 63, column: 17, scope: !4)
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
