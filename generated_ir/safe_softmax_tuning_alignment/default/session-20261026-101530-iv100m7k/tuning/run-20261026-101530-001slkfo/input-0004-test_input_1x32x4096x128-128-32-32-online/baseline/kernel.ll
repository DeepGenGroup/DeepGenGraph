; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/baseline/input.mlir"

@shm_0 = addrspace(3) global [10240 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %10, 7, !dbg !9
  %.lobit = ashr i32 %9, 31, !dbg !11
  %12 = xor i32 %9, %.lobit, !dbg !11
  %13 = sdiv i32 %12, 32, !dbg !11
  %.lobit1 = ashr i32 %9, 31, !dbg !11
  %14 = xor i32 %13, %.lobit1, !dbg !11
  %15 = sext i32 %14 to i64, !dbg !11
  %16 = add nsw i64 %8, %15, !dbg !11
  %17 = and i64 %16, 31, !dbg !11
  %18 = shl nsw i64 %10, 7, !dbg !11
  %19 = mul nsw i64 %15, -4096, !dbg !11
  %20 = add nsw i64 %18, %19, !dbg !11
  %21 = shl nuw nsw i64 %17, 19, !dbg !11
  %22 = shl nsw i64 %20, 7, !dbg !11
  %23 = add nsw i64 %21, %22, !dbg !11
  %24 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !39
  %25 = zext nneg i32 %24 to i64, !dbg !12
  %26 = lshr i64 %25, 6, !dbg !12
  %27 = lshr i64 %25, 2, !dbg !12
  %28 = and i64 %27, 12, !dbg !12
  %29 = and i32 %24, 16, !dbg !13
  %.not = icmp eq i32 %29, 0, !dbg !13
  %30 = and i32 %24, 32, !dbg !13
  %.not2 = icmp eq i32 %30, 0, !dbg !13
  %31 = or disjoint i64 %11, 127, !dbg !14
  %32 = add nsw i64 %11, 128, !dbg !14
  %33 = icmp uge i64 %32, %31, !dbg !14
  %34 = shl nsw i64 %10, 2, !dbg !7
  %35 = add nsw i64 %34, 4, !dbg !7
  %36 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %37 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %38 = getelementptr i8, ptr addrspace(1) %37, i64 64, !dbg !11
  %39 = shl nsw i64 %8, 7, !dbg !17
  %40 = shl nsw i64 %6, 12, !dbg !17
  %41 = add nsw i64 %39, %40, !dbg !17
  %42 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %43 = getelementptr i8, ptr addrspace(1) %42, i64 128, !dbg !11
  %44 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %45 = getelementptr i8, ptr addrspace(1) %44, i64 192, !dbg !11
  br label %46, !dbg !7

46:                                               ; preds = %1029, %4
  %47 = phi i64 [ %1100, %1029 ], [ 0, %4 ], !dbg !7
  %48 = phi [2 x <8 x float>] [ %1081, %1029 ], [ zeroinitializer, %4 ], !dbg !7
  %49 = phi [2 x <1 x float>] [ %1085, %1029 ], [ zeroinitializer, %4 ], !dbg !7
  %50 = phi [2 x <1 x float>] [ %1087, %1029 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %51 = phi [2 x <8 x float>] [ %1091, %1029 ], [ zeroinitializer, %4 ], !dbg !7
  %52 = phi [2 x <8 x float>] [ %1095, %1029 ], [ zeroinitializer, %4 ], !dbg !7
  %53 = phi [2 x <8 x float>] [ %1099, %1029 ], [ zeroinitializer, %4 ], !dbg !7
  %54 = icmp slt i64 %47, %35, !dbg !7
  br i1 %54, label %55, label %1101, !dbg !7

55:                                               ; preds = %46
  %56 = shl nsw i64 %47, 5, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !11, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local54 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local55 = and i64 %address.local54, 1536, !dbg !11
  %address.local56 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local55, !dbg !11
  %address.local57 = ptrtoint ptr addrspace(3) %address.local56 to i64, !dbg !11
  %address.local58 = trunc i64 %address.local57 to i32, !dbg !11
  %57 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local58), !dbg !11
  %58 = zext i32 %57 to i64, !dbg !11
  %59 = inttoptr i64 %58 to ptr addrspace(3), !dbg !11
  %address.local59 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local60 = and i64 %address.local59, 16256, !dbg !11
  %address.local61 = lshr i64 %address.local, 4, !dbg !11
  %address.local62 = and i64 %address.local61, 8, !dbg !11
  %address.local63 = or disjoint i64 %address.local60, %address.local62, !dbg !11
  %address.local64 = getelementptr [2 x i8], ptr addrspace(1) %36, i64 %address.local63, !dbg !11
  %60 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %61 = ptrtoint ptr addrspace(1) %address.local64 to i64, !dbg !11
  %62 = sub i64 %61, %60, !dbg !11
  %dma.buffer.offset663 = trunc i64 %62 to i32, !dbg !11
  %63 = trunc i64 %60 to i32, !dbg !11
  %64 = insertelement <4 x i32> poison, i32 %63, i64 0, !dbg !11
  %65 = lshr i64 %60, 32, !dbg !11
  %66 = trunc i64 %65 to i32, !dbg !11
  %67 = insertelement <4 x i32> %64, i32 %66, i64 1, !dbg !11
  %68 = insertelement <4 x i32> %67, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc664 = insertelement <4 x i32> %68, i32 131072, i64 3, !dbg !11
  %69 = ptrtoint ptr addrspace(3) %59 to i32, !dbg !11
  %dma.buffer.lds665 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %69), !dbg !11
  %70 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds665, i32 %dma.buffer.offset663, <4 x i32> %dma.buffer.rsrc664) #10, !dbg !11
  %address.local65 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local66 = shl nuw nsw i64 %address.local65, 3, !dbg !11
  %address.local67 = and i64 %address.local66, 3584, !dbg !11
  %address.local68 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local67, !dbg !11
  %address.local69 = ptrtoint ptr addrspace(3) %address.local68 to i64, !dbg !11
  %address.local70 = trunc i64 %address.local69 to i32, !dbg !11
  %71 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local70), !dbg !11
  %72 = zext i32 %71 to i64, !dbg !11
  %73 = inttoptr i64 %72 to ptr addrspace(3), !dbg !11
  %address.local71 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local72 = and i64 %address.local71, 16256, !dbg !11
  %address.local73 = lshr i64 %address.local65, 4, !dbg !11
  %address.local74 = and i64 %address.local73, 24, !dbg !11
  %address.local75 = or disjoint i64 %address.local72, %address.local74, !dbg !11
  %address.local76 = getelementptr [2 x i8], ptr addrspace(1) %36, i64 %address.local75, !dbg !11
  %74 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %75 = ptrtoint ptr addrspace(1) %address.local76 to i64, !dbg !11
  %76 = sub i64 %75, %74, !dbg !11
  %dma.buffer.offset660 = trunc i64 %76 to i32, !dbg !11
  %77 = trunc i64 %74 to i32, !dbg !11
  %78 = insertelement <4 x i32> poison, i32 %77, i64 0, !dbg !11
  %79 = lshr i64 %74, 32, !dbg !11
  %80 = trunc i64 %79 to i32, !dbg !11
  %81 = insertelement <4 x i32> %78, i32 %80, i64 1, !dbg !11
  %82 = insertelement <4 x i32> %81, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc661 = insertelement <4 x i32> %82, i32 131072, i64 3, !dbg !11
  %83 = ptrtoint ptr addrspace(3) %73 to i32, !dbg !11
  %dma.buffer.lds662 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !11
  %84 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds662, i32 %dma.buffer.offset660, <4 x i32> %dma.buffer.rsrc661) #10, !dbg !11
  %85 = lshr i64 %47, 14, !dbg !17
  %86 = add nsw i64 %85, %8, !dbg !17
  %87 = lshr i64 %47, 7, !dbg !17
  %88 = and i64 %87, 127, !dbg !17
  %89 = shl i64 %86, 19, !dbg !17
  %90 = and i64 %89, 16252928, !dbg !17
  %91 = shl i64 %87, 19, !dbg !17
  %92 = shl i64 %47, 12, !dbg !17
  %93 = sub i64 %92, %91, !dbg !17
  %94 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %90, !dbg !17
  %95 = getelementptr [2 x i8], ptr addrspace(1) %94, i64 %88, !dbg !17
  %96 = getelementptr [2 x i8], ptr addrspace(1) %95, i64 %93, !dbg !17
  br label %97, !dbg !17

97:                                               ; preds = %100, %55
  %98 = phi i64 [ %126, %100 ], [ %25, %55 ], !dbg !17
  %99 = icmp samesign ult i64 %98, 128, !dbg !17
  br i1 %99, label %100, label %127, !dbg !17

100:                                              ; preds = %97
  %101 = lshr i64 %98, 2, !dbg !17
  %102 = and i64 %101, 24, !dbg !17
  %103 = shl nuw nsw i64 %98, 3, !dbg !17
  %104 = and i64 %103, 512, !dbg !17
  %105 = shl nuw nsw i64 %98, 7, !dbg !17
  %106 = and i64 %105, 3968, !dbg !17
  %107 = getelementptr [2 x i8], ptr addrspace(1) %96, i64 %102, !dbg !17
  %108 = getelementptr [2 x i8], ptr addrspace(1) %107, i64 %106, !dbg !17
  %109 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %104, !dbg !17
  %110 = ptrtoint ptr addrspace(3) %109 to i64, !dbg !17
  %111 = trunc i64 %110 to i32, !dbg !17
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %111), !dbg !17
  %113 = zext i32 %112 to i64, !dbg !17
  %114 = inttoptr i64 %113 to ptr addrspace(3), !dbg !17
  %115 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %116 = ptrtoint ptr addrspace(1) %108 to i64, !dbg !17
  %117 = sub i64 %116, %115, !dbg !17
  %dma.buffer.offset = trunc i64 %117 to i32, !dbg !17
  %118 = trunc i64 %115 to i32, !dbg !17
  %119 = insertelement <4 x i32> poison, i32 %118, i64 0, !dbg !17
  %120 = lshr i64 %115, 32, !dbg !17
  %121 = trunc i64 %120 to i32, !dbg !17
  %122 = insertelement <4 x i32> %119, i32 %121, i64 1, !dbg !17
  %123 = insertelement <4 x i32> %122, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %123, i32 131072, i64 3, !dbg !17
  %124 = ptrtoint ptr addrspace(3) %114 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %124), !dbg !17
  %125 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %126 = or disjoint i64 %25, 256, !dbg !17
  br label %97, !dbg !17

127:                                              ; preds = %97
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.thread77 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local78 = lshr i32 %address.thread77, 2, !dbg !12
  %address.local79 = and i32 %address.local78, 4, !dbg !12
  %address.local80 = shl nuw nsw i32 %address.thread77, 1, !dbg !12
  %address.local81 = and i32 %address.local80, 384, !dbg !12
  %address.local82 = or disjoint i32 %address.local79, %address.local81, !dbg !12
  %address.local83 = shl nuw nsw i32 %address.thread77, 3, !dbg !12
  %address.local84 = and i32 %address.local83, 120, !dbg !12
  %address.local85 = or disjoint i32 %address.local82, %address.local84, !dbg !12
  %address.local86 = shl nuw nsw i32 %address.thread77, 5, !dbg !12
  %address.local87 = and i32 %address.local86, 1024, !dbg !12
  %address.local88 = or disjoint i32 %address.local85, %address.local87, !dbg !12
  %address.local89 = zext nneg i32 %address.local88 to i64, !dbg !12
  %address.local90 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local89, !dbg !12
  %128 = load <4 x half>, ptr addrspace(3) %address.local90, align 2, !dbg !12
  %address.local91 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local89, !dbg !12
  %address.local92 = getelementptr i8, ptr addrspace(3) %address.local91, i64 4096, !dbg !12
  %129 = load <4 x half>, ptr addrspace(3) %address.local92, align 2, !dbg !12
  %address.local93 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local89, !dbg !12
  %address.local94 = getelementptr i8, ptr addrspace(3) %address.local93, i64 1024, !dbg !12
  %130 = load <4 x half>, ptr addrspace(3) %address.local94, align 2, !dbg !12
  %131 = fmul <4 x half> %128, splat (half 0xH3015), !dbg !12
  %address.local95 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local89, !dbg !12
  %address.local96 = getelementptr i8, ptr addrspace(3) %address.local95, i64 5120, !dbg !12
  %132 = load <4 x half>, ptr addrspace(3) %address.local96, align 2, !dbg !12
  %133 = fmul <4 x half> %129, splat (half 0xH3015), !dbg !12
  %134 = fmul <4 x half> %130, splat (half 0xH3015), !dbg !12
  %135 = fmul <4 x half> %132, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local97 = lshr i32 %address.thread77, 2, !dbg !16
  %address.local98 = and i32 %address.local97, 4, !dbg !16
  %address.local99 = shl nuw nsw i32 %address.thread77, 3, !dbg !16
  %address.local100 = and i32 %address.local99, 376, !dbg !16
  %address.local101 = or disjoint i32 %address.local98, %address.local100, !dbg !16
  %address.local102 = zext nneg i32 %address.local101 to i64, !dbg !16
  %address.local103 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local102, !dbg !16
  %136 = load <4 x half>, ptr addrspace(3) %address.local103, align 2, !dbg !16
  %137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %136, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local104 = lshr i32 %address.thread77, 2, !dbg !16
  %address.local105 = and i32 %address.local104, 4, !dbg !16
  %address.local106 = shl nuw nsw i32 %address.thread77, 3, !dbg !16
  %address.local107 = and i32 %address.local106, 376, !dbg !16
  %address.local108 = or disjoint i32 %address.local105, %address.local107, !dbg !16
  %address.local109 = zext nneg i32 %address.local108 to i64, !dbg !16
  %address.local110 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local109, !dbg !16
  %address.local111 = getelementptr i8, ptr addrspace(3) %address.local110, i64 1024, !dbg !16
  %138 = load <4 x half>, ptr addrspace(3) %address.local111, align 2, !dbg !16
  %address.local112 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local102, !dbg !16
  %address.local113 = getelementptr i8, ptr addrspace(3) %address.local112, i64 256, !dbg !16
  %139 = load <4 x half>, ptr addrspace(3) %address.local113, align 2, !dbg !16
  %140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %138, <4 x float> %137) #9, !dbg !16
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %139, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local114 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local109, !dbg !16
  %address.local115 = getelementptr i8, ptr addrspace(3) %address.local114, i64 1280, !dbg !16
  %142 = load <4 x half>, ptr addrspace(3) %address.local115, align 2, !dbg !16
  %143 = load <4 x half>, ptr addrspace(3) %address.local103, align 2, !dbg !16
  %144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %142, <4 x float> %141) #9, !dbg !16
  %145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %143, <4 x float> zeroinitializer) #9, !dbg !16
  %146 = load <4 x half>, ptr addrspace(3) %address.local111, align 2, !dbg !16
  %147 = load <4 x half>, ptr addrspace(3) %address.local113, align 2, !dbg !16
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %146, <4 x float> %145) #9, !dbg !16
  %149 = load <4 x half>, ptr addrspace(3) %address.local115, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %149, <4 x float> %150) #9, !dbg !16
  %address.local116 = zext nneg i32 %address.thread77 to i64, !dbg !12
  %address.local117 = shl nuw nsw i64 %address.local116, 3, !dbg !11
  %address.local118 = and i64 %address.local117, 1536, !dbg !11
  %address.local119 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local118, !dbg !11
  %address.local120 = ptrtoint ptr addrspace(3) %address.local119 to i64, !dbg !11
  %address.local121 = trunc i64 %address.local120 to i32, !dbg !11
  %152 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local121), !dbg !11
  %153 = zext i32 %152 to i64, !dbg !11
  %154 = inttoptr i64 %153 to ptr addrspace(3), !dbg !11
  %address.local122 = shl nuw nsw i64 %address.local116, 7, !dbg !11
  %address.local123 = and i64 %address.local122, 16256, !dbg !11
  %address.local124 = lshr i64 %address.local116, 4, !dbg !11
  %address.local125 = and i64 %address.local124, 8, !dbg !11
  %address.local126 = or disjoint i64 %address.local123, %address.local125, !dbg !11
  %address.local127 = getelementptr [2 x i8], ptr addrspace(1) %38, i64 %address.local126, !dbg !11
  %155 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %156 = ptrtoint ptr addrspace(1) %address.local127 to i64, !dbg !11
  %157 = sub i64 %156, %155, !dbg !11
  %dma.buffer.offset657 = trunc i64 %157 to i32, !dbg !11
  %158 = trunc i64 %155 to i32, !dbg !11
  %159 = insertelement <4 x i32> poison, i32 %158, i64 0, !dbg !11
  %160 = lshr i64 %155, 32, !dbg !11
  %161 = trunc i64 %160 to i32, !dbg !11
  %162 = insertelement <4 x i32> %159, i32 %161, i64 1, !dbg !11
  %163 = insertelement <4 x i32> %162, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc658 = insertelement <4 x i32> %163, i32 131072, i64 3, !dbg !11
  %164 = ptrtoint ptr addrspace(3) %154 to i32, !dbg !11
  %dma.buffer.lds659 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %164), !dbg !11
  %165 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds659, i32 %dma.buffer.offset657, <4 x i32> %dma.buffer.rsrc658) #10, !dbg !11
  %address.local128 = or disjoint i64 %address.local116, 256, !dbg !11
  %address.local129 = shl nuw nsw i64 %address.local128, 3, !dbg !11
  %address.local130 = and i64 %address.local129, 3584, !dbg !11
  %address.local131 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local130, !dbg !11
  %address.local132 = ptrtoint ptr addrspace(3) %address.local131 to i64, !dbg !11
  %address.local133 = trunc i64 %address.local132 to i32, !dbg !11
  %166 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %167 = zext i32 %166 to i64, !dbg !11
  %168 = inttoptr i64 %167 to ptr addrspace(3), !dbg !11
  %address.local134 = shl nuw nsw i64 %address.local116, 7, !dbg !11
  %address.local135 = and i64 %address.local134, 16256, !dbg !11
  %address.local136 = lshr i64 %address.local128, 4, !dbg !11
  %address.local137 = and i64 %address.local136, 24, !dbg !11
  %address.local138 = or disjoint i64 %address.local135, %address.local137, !dbg !11
  %address.local139 = getelementptr [2 x i8], ptr addrspace(1) %38, i64 %address.local138, !dbg !11
  %169 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %170 = ptrtoint ptr addrspace(1) %address.local139 to i64, !dbg !11
  %171 = sub i64 %170, %169, !dbg !11
  %dma.buffer.offset654 = trunc i64 %171 to i32, !dbg !11
  %172 = trunc i64 %169 to i32, !dbg !11
  %173 = insertelement <4 x i32> poison, i32 %172, i64 0, !dbg !11
  %174 = lshr i64 %169, 32, !dbg !11
  %175 = trunc i64 %174 to i32, !dbg !11
  %176 = insertelement <4 x i32> %173, i32 %175, i64 1, !dbg !11
  %177 = insertelement <4 x i32> %176, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc655 = insertelement <4 x i32> %177, i32 131072, i64 3, !dbg !11
  %178 = ptrtoint ptr addrspace(3) %168 to i32, !dbg !11
  %dma.buffer.lds656 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %178), !dbg !11
  %179 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds656, i32 %dma.buffer.offset654, <4 x i32> %dma.buffer.rsrc655) #10, !dbg !11
  %180 = add nuw i64 %47, 4096, !dbg !17
  %181 = icmp slt i64 %180, 0, !dbg !17
  %182 = sub nuw i64 -4097, %47, !dbg !17
  %183 = select i1 %181, i64 %182, i64 %180, !dbg !17
  %184 = sdiv i64 %183, 16384, !dbg !17
  %.lobit3 = ashr i64 %180, 63, !dbg !17
  %185 = xor i64 %184, %.lobit3, !dbg !17
  %186 = add nsw i64 %185, %8, !dbg !17
  %187 = add nsw i64 %87, %41, !dbg !17
  %188 = add nsw i64 %187, 32, !dbg !17
  %189 = icmp slt i64 %187, -32, !dbg !17
  %190 = sub nsw i64 -33, %187, !dbg !17
  %191 = select i1 %189, i64 %190, i64 %188, !dbg !17
  %192 = sdiv i64 %191, 128, !dbg !17
  %193 = sext i1 %189 to i64, !dbg !17
  %194 = xor i64 %192, %193, !dbg !17
  %195 = shl nsw i64 %194, 7, !dbg !17
  %196 = sub i64 %41, %195, !dbg !17
  %197 = shl i64 %186, 19, !dbg !17
  %198 = and i64 %197, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %199 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %198, !dbg !17
  %200 = getelementptr [2 x i8], ptr addrspace(1) %199, i64 %87, !dbg !17
  %201 = getelementptr [2 x i8], ptr addrspace(1) %200, i64 %196, !dbg !17
  %202 = getelementptr i8, ptr addrspace(1) %201, i64 64, !dbg !17
  %203 = getelementptr [2 x i8], ptr addrspace(1) %202, i64 %93, !dbg !17
  br label %204, !dbg !17

204:                                              ; preds = %207, %127
  %205 = phi i64 [ %233, %207 ], [ %25, %127 ], !dbg !17
  %206 = icmp samesign ult i64 %205, 128, !dbg !17
  br i1 %206, label %207, label %234, !dbg !17

207:                                              ; preds = %204
  %208 = lshr i64 %205, 2, !dbg !17
  %209 = and i64 %208, 24, !dbg !17
  %210 = shl nuw nsw i64 %205, 3, !dbg !17
  %211 = and i64 %210, 512, !dbg !17
  %212 = shl nuw nsw i64 %205, 7, !dbg !17
  %213 = and i64 %212, 3968, !dbg !17
  %214 = getelementptr [2 x i8], ptr addrspace(1) %203, i64 %209, !dbg !17
  %215 = getelementptr [2 x i8], ptr addrspace(1) %214, i64 %213, !dbg !17
  %216 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %211, !dbg !17
  %217 = ptrtoint ptr addrspace(3) %216 to i64, !dbg !17
  %218 = trunc i64 %217 to i32, !dbg !17
  %219 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %218), !dbg !17
  %220 = zext i32 %219 to i64, !dbg !17
  %221 = inttoptr i64 %220 to ptr addrspace(3), !dbg !17
  %222 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %223 = ptrtoint ptr addrspace(1) %215 to i64, !dbg !17
  %224 = sub i64 %223, %222, !dbg !17
  %dma.buffer.offset621 = trunc i64 %224 to i32, !dbg !17
  %225 = trunc i64 %222 to i32, !dbg !17
  %226 = insertelement <4 x i32> poison, i32 %225, i64 0, !dbg !17
  %227 = lshr i64 %222, 32, !dbg !17
  %228 = trunc i64 %227 to i32, !dbg !17
  %229 = insertelement <4 x i32> %226, i32 %228, i64 1, !dbg !17
  %230 = insertelement <4 x i32> %229, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc622 = insertelement <4 x i32> %230, i32 131072, i64 3, !dbg !17
  %231 = ptrtoint ptr addrspace(3) %221 to i32, !dbg !17
  %dma.buffer.lds623 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %231), !dbg !17
  %232 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds623, i32 %dma.buffer.offset621, <4 x i32> %dma.buffer.rsrc622) #10, !dbg !17
  %233 = or disjoint i64 %25, 256, !dbg !17
  br label %204, !dbg !17

234:                                              ; preds = %204
  %address.thread140 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local141 = lshr i32 %address.thread140, 2, !dbg !12
  %address.local142 = and i32 %address.local141, 4, !dbg !12
  %address.local143 = shl nuw nsw i32 %address.thread140, 1, !dbg !12
  %address.local144 = and i32 %address.local143, 384, !dbg !12
  %address.local145 = or disjoint i32 %address.local142, %address.local144, !dbg !12
  %address.local146 = shl nuw nsw i32 %address.thread140, 3, !dbg !12
  %address.local147 = and i32 %address.local146, 120, !dbg !12
  %address.local148 = or disjoint i32 %address.local145, %address.local147, !dbg !12
  %address.local149 = shl nuw nsw i32 %address.thread140, 5, !dbg !12
  %address.local150 = and i32 %address.local149, 1024, !dbg !12
  %address.local151 = or disjoint i32 %address.local148, %address.local150, !dbg !12
  %address.local152 = zext nneg i32 %address.local151 to i64, !dbg !12
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local152, !dbg !12
  %235 = load <4 x half>, ptr addrspace(3) %address.local153, align 2, !dbg !12
  %address.local154 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local152, !dbg !12
  %address.local155 = getelementptr i8, ptr addrspace(3) %address.local154, i64 4096, !dbg !12
  %236 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !12
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local152, !dbg !12
  %address.local157 = getelementptr i8, ptr addrspace(3) %address.local156, i64 1024, !dbg !12
  %237 = load <4 x half>, ptr addrspace(3) %address.local157, align 2, !dbg !12
  %238 = fmul <4 x half> %235, splat (half 0xH3015), !dbg !12
  %address.local158 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local152, !dbg !12
  %address.local159 = getelementptr i8, ptr addrspace(3) %address.local158, i64 5120, !dbg !12
  %239 = load <4 x half>, ptr addrspace(3) %address.local159, align 2, !dbg !12
  %240 = fmul <4 x half> %236, splat (half 0xH3015), !dbg !12
  %241 = fmul <4 x half> %237, splat (half 0xH3015), !dbg !12
  %242 = fmul <4 x half> %239, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local160 = lshr i32 %address.thread140, 2, !dbg !16
  %address.local161 = and i32 %address.local160, 4, !dbg !16
  %address.local162 = shl nuw nsw i32 %address.thread140, 3, !dbg !16
  %address.local163 = and i32 %address.local162, 376, !dbg !16
  %address.local164 = or disjoint i32 %address.local161, %address.local163, !dbg !16
  %address.local165 = zext nneg i32 %address.local164 to i64, !dbg !16
  %address.local166 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local165, !dbg !16
  %243 = load <4 x half>, ptr addrspace(3) %address.local166, align 2, !dbg !16
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %243, <4 x float> %140) #9, !dbg !16
  %address.local167 = lshr i32 %address.thread140, 2, !dbg !16
  %address.local168 = and i32 %address.local167, 4, !dbg !16
  %address.local169 = shl nuw nsw i32 %address.thread140, 3, !dbg !16
  %address.local170 = and i32 %address.local169, 376, !dbg !16
  %address.local171 = or disjoint i32 %address.local168, %address.local170, !dbg !16
  %address.local172 = zext nneg i32 %address.local171 to i64, !dbg !16
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local172, !dbg !16
  %address.local174 = getelementptr i8, ptr addrspace(3) %address.local173, i64 1024, !dbg !16
  %245 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !16
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local165, !dbg !16
  %address.local176 = getelementptr i8, ptr addrspace(3) %address.local175, i64 256, !dbg !16
  %246 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %240, <4 x half> %245, <4 x float> %244) #9, !dbg !16
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %246, <4 x float> %144) #9, !dbg !16
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local172, !dbg !16
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local177, i64 1280, !dbg !16
  %249 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !16
  %250 = load <4 x half>, ptr addrspace(3) %address.local166, align 2, !dbg !16
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %240, <4 x half> %249, <4 x float> %248) #9, !dbg !16
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %241, <4 x half> %250, <4 x float> %148) #9, !dbg !16
  %253 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !16
  %254 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %253, <4 x float> %252) #9, !dbg !16
  %256 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %241, <4 x half> %254, <4 x float> %151) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %256, <4 x float> %257) #9, !dbg !16
  %address.local179 = zext nneg i32 %address.thread140 to i64, !dbg !12
  %address.local180 = shl nuw nsw i64 %address.local179, 3, !dbg !11
  %address.local181 = and i64 %address.local180, 1536, !dbg !11
  %address.local182 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local181, !dbg !11
  %address.local183 = ptrtoint ptr addrspace(3) %address.local182 to i64, !dbg !11
  %address.local184 = trunc i64 %address.local183 to i32, !dbg !11
  %259 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local184), !dbg !11
  %260 = zext i32 %259 to i64, !dbg !11
  %261 = inttoptr i64 %260 to ptr addrspace(3), !dbg !11
  %address.local185 = shl nuw nsw i64 %address.local179, 7, !dbg !11
  %address.local186 = and i64 %address.local185, 16256, !dbg !11
  %address.local187 = lshr i64 %address.local179, 4, !dbg !11
  %address.local188 = and i64 %address.local187, 8, !dbg !11
  %address.local189 = or disjoint i64 %address.local186, %address.local188, !dbg !11
  %address.local190 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local189, !dbg !11
  %262 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %263 = ptrtoint ptr addrspace(1) %address.local190 to i64, !dbg !11
  %264 = sub i64 %263, %262, !dbg !11
  %dma.buffer.offset651 = trunc i64 %264 to i32, !dbg !11
  %265 = trunc i64 %262 to i32, !dbg !11
  %266 = insertelement <4 x i32> poison, i32 %265, i64 0, !dbg !11
  %267 = lshr i64 %262, 32, !dbg !11
  %268 = trunc i64 %267 to i32, !dbg !11
  %269 = insertelement <4 x i32> %266, i32 %268, i64 1, !dbg !11
  %270 = insertelement <4 x i32> %269, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc652 = insertelement <4 x i32> %270, i32 131072, i64 3, !dbg !11
  %271 = ptrtoint ptr addrspace(3) %261 to i32, !dbg !11
  %dma.buffer.lds653 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %271), !dbg !11
  %272 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds653, i32 %dma.buffer.offset651, <4 x i32> %dma.buffer.rsrc652) #10, !dbg !11
  %address.local191 = or disjoint i64 %address.local179, 256, !dbg !11
  %address.local192 = shl nuw nsw i64 %address.local191, 3, !dbg !11
  %address.local193 = and i64 %address.local192, 3584, !dbg !11
  %address.local194 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local193, !dbg !11
  %address.local195 = ptrtoint ptr addrspace(3) %address.local194 to i64, !dbg !11
  %address.local196 = trunc i64 %address.local195 to i32, !dbg !11
  %273 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local196), !dbg !11
  %274 = zext i32 %273 to i64, !dbg !11
  %275 = inttoptr i64 %274 to ptr addrspace(3), !dbg !11
  %address.local197 = shl nuw nsw i64 %address.local179, 7, !dbg !11
  %address.local198 = and i64 %address.local197, 16256, !dbg !11
  %address.local199 = lshr i64 %address.local191, 4, !dbg !11
  %address.local200 = and i64 %address.local199, 24, !dbg !11
  %address.local201 = or disjoint i64 %address.local198, %address.local200, !dbg !11
  %address.local202 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local201, !dbg !11
  %276 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %277 = ptrtoint ptr addrspace(1) %address.local202 to i64, !dbg !11
  %278 = sub i64 %277, %276, !dbg !11
  %dma.buffer.offset648 = trunc i64 %278 to i32, !dbg !11
  %279 = trunc i64 %276 to i32, !dbg !11
  %280 = insertelement <4 x i32> poison, i32 %279, i64 0, !dbg !11
  %281 = lshr i64 %276, 32, !dbg !11
  %282 = trunc i64 %281 to i32, !dbg !11
  %283 = insertelement <4 x i32> %280, i32 %282, i64 1, !dbg !11
  %284 = insertelement <4 x i32> %283, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc649 = insertelement <4 x i32> %284, i32 131072, i64 3, !dbg !11
  %285 = ptrtoint ptr addrspace(3) %275 to i32, !dbg !11
  %dma.buffer.lds650 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %285), !dbg !11
  %286 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds650, i32 %dma.buffer.offset648, <4 x i32> %dma.buffer.rsrc649) #10, !dbg !11
  %287 = add nuw i64 %47, 8192, !dbg !17
  %288 = icmp slt i64 %287, 0, !dbg !17
  %289 = sub nuw i64 -8193, %47, !dbg !17
  %290 = select i1 %288, i64 %289, i64 %287, !dbg !17
  %291 = sdiv i64 %290, 16384, !dbg !17
  %.lobit4 = ashr i64 %287, 63, !dbg !17
  %292 = xor i64 %291, %.lobit4, !dbg !17
  %293 = add nsw i64 %292, %8, !dbg !17
  %294 = add nsw i64 %187, 64, !dbg !17
  %295 = icmp slt i64 %187, -64, !dbg !17
  %296 = sub nsw i64 -65, %187, !dbg !17
  %297 = select i1 %295, i64 %296, i64 %294, !dbg !17
  %298 = sdiv i64 %297, 128, !dbg !17
  %299 = sext i1 %295 to i64, !dbg !17
  %300 = xor i64 %298, %299, !dbg !17
  %301 = shl nsw i64 %300, 7, !dbg !17
  %302 = sub i64 %41, %301, !dbg !17
  %303 = shl i64 %293, 19, !dbg !17
  %304 = and i64 %303, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %305 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %304, !dbg !17
  %306 = getelementptr [2 x i8], ptr addrspace(1) %305, i64 %87, !dbg !17
  %307 = getelementptr [2 x i8], ptr addrspace(1) %306, i64 %302, !dbg !17
  %308 = getelementptr i8, ptr addrspace(1) %307, i64 128, !dbg !17
  %309 = getelementptr [2 x i8], ptr addrspace(1) %308, i64 %93, !dbg !17
  br label %310, !dbg !17

310:                                              ; preds = %313, %234
  %311 = phi i64 [ %339, %313 ], [ %25, %234 ], !dbg !17
  %312 = icmp samesign ult i64 %311, 128, !dbg !17
  br i1 %312, label %313, label %340, !dbg !17

313:                                              ; preds = %310
  %314 = lshr i64 %311, 2, !dbg !17
  %315 = and i64 %314, 24, !dbg !17
  %316 = shl nuw nsw i64 %311, 3, !dbg !17
  %317 = and i64 %316, 512, !dbg !17
  %318 = shl nuw nsw i64 %311, 7, !dbg !17
  %319 = and i64 %318, 3968, !dbg !17
  %320 = getelementptr [2 x i8], ptr addrspace(1) %309, i64 %315, !dbg !17
  %321 = getelementptr [2 x i8], ptr addrspace(1) %320, i64 %319, !dbg !17
  %322 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %317, !dbg !17
  %323 = ptrtoint ptr addrspace(3) %322 to i64, !dbg !17
  %324 = trunc i64 %323 to i32, !dbg !17
  %325 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %324), !dbg !17
  %326 = zext i32 %325 to i64, !dbg !17
  %327 = inttoptr i64 %326 to ptr addrspace(3), !dbg !17
  %328 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %329 = ptrtoint ptr addrspace(1) %321 to i64, !dbg !17
  %330 = sub i64 %329, %328, !dbg !17
  %dma.buffer.offset624 = trunc i64 %330 to i32, !dbg !17
  %331 = trunc i64 %328 to i32, !dbg !17
  %332 = insertelement <4 x i32> poison, i32 %331, i64 0, !dbg !17
  %333 = lshr i64 %328, 32, !dbg !17
  %334 = trunc i64 %333 to i32, !dbg !17
  %335 = insertelement <4 x i32> %332, i32 %334, i64 1, !dbg !17
  %336 = insertelement <4 x i32> %335, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc625 = insertelement <4 x i32> %336, i32 131072, i64 3, !dbg !17
  %337 = ptrtoint ptr addrspace(3) %327 to i32, !dbg !17
  %dma.buffer.lds626 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %337), !dbg !17
  %338 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds626, i32 %dma.buffer.offset624, <4 x i32> %dma.buffer.rsrc625) #10, !dbg !17
  %339 = or disjoint i64 %25, 256, !dbg !17
  br label %310, !dbg !17

340:                                              ; preds = %310
  %address.thread203 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local204 = lshr i32 %address.thread203, 2, !dbg !12
  %address.local205 = and i32 %address.local204, 4, !dbg !12
  %address.local206 = shl nuw nsw i32 %address.thread203, 1, !dbg !12
  %address.local207 = and i32 %address.local206, 384, !dbg !12
  %address.local208 = or disjoint i32 %address.local205, %address.local207, !dbg !12
  %address.local209 = shl nuw nsw i32 %address.thread203, 3, !dbg !12
  %address.local210 = and i32 %address.local209, 120, !dbg !12
  %address.local211 = or disjoint i32 %address.local208, %address.local210, !dbg !12
  %address.local212 = shl nuw nsw i32 %address.thread203, 5, !dbg !12
  %address.local213 = and i32 %address.local212, 1024, !dbg !12
  %address.local214 = or disjoint i32 %address.local211, %address.local213, !dbg !12
  %address.local215 = zext nneg i32 %address.local214 to i64, !dbg !12
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local215, !dbg !12
  %341 = load <4 x half>, ptr addrspace(3) %address.local216, align 2, !dbg !12
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local215, !dbg !12
  %address.local218 = getelementptr i8, ptr addrspace(3) %address.local217, i64 4096, !dbg !12
  %342 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !12
  %address.local219 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local215, !dbg !12
  %address.local220 = getelementptr i8, ptr addrspace(3) %address.local219, i64 1024, !dbg !12
  %343 = load <4 x half>, ptr addrspace(3) %address.local220, align 2, !dbg !12
  %344 = fmul <4 x half> %341, splat (half 0xH3015), !dbg !12
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local215, !dbg !12
  %address.local222 = getelementptr i8, ptr addrspace(3) %address.local221, i64 5120, !dbg !12
  %345 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !12
  %346 = fmul <4 x half> %342, splat (half 0xH3015), !dbg !12
  %347 = fmul <4 x half> %343, splat (half 0xH3015), !dbg !12
  %348 = fmul <4 x half> %345, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local223 = lshr i32 %address.thread203, 2, !dbg !16
  %address.local224 = and i32 %address.local223, 4, !dbg !16
  %address.local225 = shl nuw nsw i32 %address.thread203, 3, !dbg !16
  %address.local226 = and i32 %address.local225, 376, !dbg !16
  %address.local227 = or disjoint i32 %address.local224, %address.local226, !dbg !16
  %address.local228 = zext nneg i32 %address.local227 to i64, !dbg !16
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local228, !dbg !16
  %349 = load <4 x half>, ptr addrspace(3) %address.local229, align 2, !dbg !16
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %349, <4 x float> %247) #9, !dbg !16
  %address.local230 = lshr i32 %address.thread203, 2, !dbg !16
  %address.local231 = and i32 %address.local230, 4, !dbg !16
  %address.local232 = shl nuw nsw i32 %address.thread203, 3, !dbg !16
  %address.local233 = and i32 %address.local232, 376, !dbg !16
  %address.local234 = or disjoint i32 %address.local231, %address.local233, !dbg !16
  %address.local235 = zext nneg i32 %address.local234 to i64, !dbg !16
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local235, !dbg !16
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local236, i64 1024, !dbg !16
  %351 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !16
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local228, !dbg !16
  %address.local239 = getelementptr i8, ptr addrspace(3) %address.local238, i64 256, !dbg !16
  %352 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %351, <4 x float> %350) #9, !dbg !16
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %352, <4 x float> %251) #9, !dbg !16
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local235, !dbg !16
  %address.local241 = getelementptr i8, ptr addrspace(3) %address.local240, i64 1280, !dbg !16
  %355 = load <4 x half>, ptr addrspace(3) %address.local241, align 2, !dbg !16
  %356 = load <4 x half>, ptr addrspace(3) %address.local229, align 2, !dbg !16
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %346, <4 x half> %355, <4 x float> %354) #9, !dbg !16
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %356, <4 x float> %255) #9, !dbg !16
  %359 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !16
  %360 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %359, <4 x float> %358) #9, !dbg !16
  %362 = load <4 x half>, ptr addrspace(3) %address.local241, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %360, <4 x float> %258) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %348, <4 x half> %362, <4 x float> %363) #9, !dbg !16
  %address.local242 = zext nneg i32 %address.thread203 to i64, !dbg !12
  %address.local243 = shl nuw nsw i64 %address.local242, 3, !dbg !11
  %address.local244 = and i64 %address.local243, 1536, !dbg !11
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local244, !dbg !11
  %address.local246 = ptrtoint ptr addrspace(3) %address.local245 to i64, !dbg !11
  %address.local247 = trunc i64 %address.local246 to i32, !dbg !11
  %365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local247), !dbg !11
  %366 = zext i32 %365 to i64, !dbg !11
  %367 = inttoptr i64 %366 to ptr addrspace(3), !dbg !11
  %address.local248 = shl nuw nsw i64 %address.local242, 7, !dbg !11
  %address.local249 = and i64 %address.local248, 16256, !dbg !11
  %address.local250 = lshr i64 %address.local242, 4, !dbg !11
  %address.local251 = and i64 %address.local250, 8, !dbg !11
  %address.local252 = or disjoint i64 %address.local249, %address.local251, !dbg !11
  %address.local253 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local252, !dbg !11
  %368 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %369 = ptrtoint ptr addrspace(1) %address.local253 to i64, !dbg !11
  %370 = sub i64 %369, %368, !dbg !11
  %dma.buffer.offset645 = trunc i64 %370 to i32, !dbg !11
  %371 = trunc i64 %368 to i32, !dbg !11
  %372 = insertelement <4 x i32> poison, i32 %371, i64 0, !dbg !11
  %373 = lshr i64 %368, 32, !dbg !11
  %374 = trunc i64 %373 to i32, !dbg !11
  %375 = insertelement <4 x i32> %372, i32 %374, i64 1, !dbg !11
  %376 = insertelement <4 x i32> %375, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc646 = insertelement <4 x i32> %376, i32 131072, i64 3, !dbg !11
  %377 = ptrtoint ptr addrspace(3) %367 to i32, !dbg !11
  %dma.buffer.lds647 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %377), !dbg !11
  %378 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds647, i32 %dma.buffer.offset645, <4 x i32> %dma.buffer.rsrc646) #10, !dbg !11
  %address.local254 = or disjoint i64 %address.local242, 256, !dbg !11
  %address.local255 = shl nuw nsw i64 %address.local254, 3, !dbg !11
  %address.local256 = and i64 %address.local255, 3584, !dbg !11
  %address.local257 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local256, !dbg !11
  %address.local258 = ptrtoint ptr addrspace(3) %address.local257 to i64, !dbg !11
  %address.local259 = trunc i64 %address.local258 to i32, !dbg !11
  %379 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local259), !dbg !11
  %380 = zext i32 %379 to i64, !dbg !11
  %381 = inttoptr i64 %380 to ptr addrspace(3), !dbg !11
  %address.local260 = shl nuw nsw i64 %address.local242, 7, !dbg !11
  %address.local261 = and i64 %address.local260, 16256, !dbg !11
  %address.local262 = lshr i64 %address.local254, 4, !dbg !11
  %address.local263 = and i64 %address.local262, 24, !dbg !11
  %address.local264 = or disjoint i64 %address.local261, %address.local263, !dbg !11
  %address.local265 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local264, !dbg !11
  %382 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %383 = ptrtoint ptr addrspace(1) %address.local265 to i64, !dbg !11
  %384 = sub i64 %383, %382, !dbg !11
  %dma.buffer.offset642 = trunc i64 %384 to i32, !dbg !11
  %385 = trunc i64 %382 to i32, !dbg !11
  %386 = insertelement <4 x i32> poison, i32 %385, i64 0, !dbg !11
  %387 = lshr i64 %382, 32, !dbg !11
  %388 = trunc i64 %387 to i32, !dbg !11
  %389 = insertelement <4 x i32> %386, i32 %388, i64 1, !dbg !11
  %390 = insertelement <4 x i32> %389, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc643 = insertelement <4 x i32> %390, i32 131072, i64 3, !dbg !11
  %391 = ptrtoint ptr addrspace(3) %381 to i32, !dbg !11
  %dma.buffer.lds644 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %391), !dbg !11
  %392 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds644, i32 %dma.buffer.offset642, <4 x i32> %dma.buffer.rsrc643) #10, !dbg !11
  %393 = add nuw i64 %47, 12288, !dbg !17
  %394 = icmp slt i64 %393, 0, !dbg !17
  %395 = sub nuw i64 -12289, %47, !dbg !17
  %396 = select i1 %394, i64 %395, i64 %393, !dbg !17
  %397 = sdiv i64 %396, 16384, !dbg !17
  %.lobit5 = ashr i64 %393, 63, !dbg !17
  %398 = xor i64 %397, %.lobit5, !dbg !17
  %399 = add nsw i64 %398, %8, !dbg !17
  %400 = add nsw i64 %187, 96, !dbg !17
  %401 = icmp slt i64 %187, -96, !dbg !17
  %402 = sub nsw i64 -97, %187, !dbg !17
  %403 = select i1 %401, i64 %402, i64 %400, !dbg !17
  %404 = sdiv i64 %403, 128, !dbg !17
  %405 = sext i1 %401 to i64, !dbg !17
  %406 = xor i64 %404, %405, !dbg !17
  %407 = shl nsw i64 %406, 7, !dbg !17
  %408 = sub i64 %41, %407, !dbg !17
  %409 = shl i64 %399, 19, !dbg !17
  %410 = and i64 %409, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %411 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %410, !dbg !17
  %412 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %87, !dbg !17
  %413 = getelementptr [2 x i8], ptr addrspace(1) %412, i64 %408, !dbg !17
  %414 = getelementptr i8, ptr addrspace(1) %413, i64 192, !dbg !17
  %415 = getelementptr [2 x i8], ptr addrspace(1) %414, i64 %93, !dbg !17
  br label %416, !dbg !17

416:                                              ; preds = %419, %340
  %417 = phi i64 [ %445, %419 ], [ %25, %340 ], !dbg !17
  %418 = icmp samesign ult i64 %417, 128, !dbg !17
  br i1 %418, label %419, label %446, !dbg !17

419:                                              ; preds = %416
  %420 = lshr i64 %417, 2, !dbg !17
  %421 = and i64 %420, 24, !dbg !17
  %422 = shl nuw nsw i64 %417, 3, !dbg !17
  %423 = and i64 %422, 512, !dbg !17
  %424 = shl nuw nsw i64 %417, 7, !dbg !17
  %425 = and i64 %424, 3968, !dbg !17
  %426 = getelementptr [2 x i8], ptr addrspace(1) %415, i64 %421, !dbg !17
  %427 = getelementptr [2 x i8], ptr addrspace(1) %426, i64 %425, !dbg !17
  %428 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %423, !dbg !17
  %429 = ptrtoint ptr addrspace(3) %428 to i64, !dbg !17
  %430 = trunc i64 %429 to i32, !dbg !17
  %431 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %430), !dbg !17
  %432 = zext i32 %431 to i64, !dbg !17
  %433 = inttoptr i64 %432 to ptr addrspace(3), !dbg !17
  %434 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %435 = ptrtoint ptr addrspace(1) %427 to i64, !dbg !17
  %436 = sub i64 %435, %434, !dbg !17
  %dma.buffer.offset627 = trunc i64 %436 to i32, !dbg !17
  %437 = trunc i64 %434 to i32, !dbg !17
  %438 = insertelement <4 x i32> poison, i32 %437, i64 0, !dbg !17
  %439 = lshr i64 %434, 32, !dbg !17
  %440 = trunc i64 %439 to i32, !dbg !17
  %441 = insertelement <4 x i32> %438, i32 %440, i64 1, !dbg !17
  %442 = insertelement <4 x i32> %441, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc628 = insertelement <4 x i32> %442, i32 131072, i64 3, !dbg !17
  %443 = ptrtoint ptr addrspace(3) %433 to i32, !dbg !17
  %dma.buffer.lds629 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %443), !dbg !17
  %444 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds629, i32 %dma.buffer.offset627, <4 x i32> %dma.buffer.rsrc628) #10, !dbg !17
  %445 = or disjoint i64 %25, 256, !dbg !17
  br label %416, !dbg !17

446:                                              ; preds = %416
  %address.thread266 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local267 = lshr i32 %address.thread266, 2, !dbg !12
  %address.local268 = and i32 %address.local267, 4, !dbg !12
  %address.local269 = shl nuw nsw i32 %address.thread266, 1, !dbg !12
  %address.local270 = and i32 %address.local269, 384, !dbg !12
  %address.local271 = or disjoint i32 %address.local268, %address.local270, !dbg !12
  %address.local272 = shl nuw nsw i32 %address.thread266, 3, !dbg !12
  %address.local273 = and i32 %address.local272, 120, !dbg !12
  %address.local274 = or disjoint i32 %address.local271, %address.local273, !dbg !12
  %address.local275 = shl nuw nsw i32 %address.thread266, 5, !dbg !12
  %address.local276 = and i32 %address.local275, 1024, !dbg !12
  %address.local277 = or disjoint i32 %address.local274, %address.local276, !dbg !12
  %address.local278 = zext nneg i32 %address.local277 to i64, !dbg !12
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local278, !dbg !12
  %447 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !12
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local278, !dbg !12
  %address.local281 = getelementptr i8, ptr addrspace(3) %address.local280, i64 4096, !dbg !12
  %448 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !12
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local278, !dbg !12
  %address.local283 = getelementptr i8, ptr addrspace(3) %address.local282, i64 1024, !dbg !12
  %449 = load <4 x half>, ptr addrspace(3) %address.local283, align 2, !dbg !12
  %450 = fmul <4 x half> %447, splat (half 0xH3015), !dbg !12
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local278, !dbg !12
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local284, i64 5120, !dbg !12
  %451 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !12
  %452 = fmul <4 x half> %448, splat (half 0xH3015), !dbg !12
  %453 = fmul <4 x half> %449, splat (half 0xH3015), !dbg !12
  %454 = fmul <4 x half> %451, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %455 = add nsw i64 %87, %8, !dbg !18
  %456 = shl i64 %455, 19, !dbg !18
  %457 = and i64 %456, 16252928, !dbg !18
  %458 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %457, !dbg !18
  %459 = getelementptr [2 x i8], ptr addrspace(1) %458, i64 %93, !dbg !18
  br label %460, !dbg !18

460:                                              ; preds = %463, %446
  %461 = phi i64 [ %489, %463 ], [ %25, %446 ], !dbg !18
  %462 = icmp samesign ult i64 %461, 128, !dbg !18
  br i1 %462, label %463, label %490, !dbg !18

463:                                              ; preds = %460
  %464 = shl nuw nsw i64 %461, 3, !dbg !18
  %465 = and i64 %464, 24, !dbg !18
  %466 = shl nuw nsw i64 %461, 3, !dbg !18
  %467 = and i64 %466, 512, !dbg !18
  %468 = shl nuw nsw i64 %461, 5, !dbg !18
  %469 = and i64 %468, 3968, !dbg !18
  %470 = getelementptr [2 x i8], ptr addrspace(1) %459, i64 %469, !dbg !18
  %471 = getelementptr [2 x i8], ptr addrspace(1) %470, i64 %465, !dbg !18
  %472 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %467, !dbg !18
  %473 = ptrtoint ptr addrspace(3) %472 to i64, !dbg !18
  %474 = trunc i64 %473 to i32, !dbg !18
  %475 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %474), !dbg !18
  %476 = zext i32 %475 to i64, !dbg !18
  %477 = inttoptr i64 %476 to ptr addrspace(3), !dbg !18
  %478 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %479 = ptrtoint ptr addrspace(1) %471 to i64, !dbg !18
  %480 = sub i64 %479, %478, !dbg !18
  %dma.buffer.offset630 = trunc i64 %480 to i32, !dbg !18
  %481 = trunc i64 %478 to i32, !dbg !18
  %482 = insertelement <4 x i32> poison, i32 %481, i64 0, !dbg !18
  %483 = lshr i64 %478, 32, !dbg !18
  %484 = trunc i64 %483 to i32, !dbg !18
  %485 = insertelement <4 x i32> %482, i32 %484, i64 1, !dbg !18
  %486 = insertelement <4 x i32> %485, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc631 = insertelement <4 x i32> %486, i32 131072, i64 3, !dbg !18
  %487 = ptrtoint ptr addrspace(3) %477 to i32, !dbg !18
  %dma.buffer.lds632 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %487), !dbg !18
  %488 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds632, i32 %dma.buffer.offset630, <4 x i32> %dma.buffer.rsrc631) #10, !dbg !18
  %489 = or disjoint i64 %25, 256, !dbg !18
  br label %460, !dbg !18

490:                                              ; preds = %460
  %address.thread286 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local287 = lshr i32 %address.thread286, 2, !dbg !16
  %address.local288 = and i32 %address.local287, 4, !dbg !16
  %address.local289 = shl nuw nsw i32 %address.thread286, 3, !dbg !16
  %address.local290 = and i32 %address.local289, 376, !dbg !16
  %address.local291 = or disjoint i32 %address.local288, %address.local290, !dbg !16
  %address.local292 = zext nneg i32 %address.local291 to i64, !dbg !16
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local292, !dbg !16
  %491 = load <4 x half>, ptr addrspace(3) %address.local293, align 2, !dbg !16
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %491, <4 x float> %353) #9, !dbg !16
  %address.local294 = lshr i32 %address.thread286, 2, !dbg !16
  %address.local295 = and i32 %address.local294, 4, !dbg !16
  %address.local296 = shl nuw nsw i32 %address.thread286, 3, !dbg !16
  %address.local297 = and i32 %address.local296, 376, !dbg !16
  %address.local298 = or disjoint i32 %address.local295, %address.local297, !dbg !16
  %address.local299 = zext nneg i32 %address.local298 to i64, !dbg !16
  %address.local300 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local299, !dbg !16
  %address.local301 = getelementptr i8, ptr addrspace(3) %address.local300, i64 1024, !dbg !16
  %493 = load <4 x half>, ptr addrspace(3) %address.local301, align 2, !dbg !16
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local292, !dbg !16
  %address.local303 = getelementptr i8, ptr addrspace(3) %address.local302, i64 256, !dbg !16
  %494 = load <4 x half>, ptr addrspace(3) %address.local303, align 2, !dbg !16
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %452, <4 x half> %493, <4 x float> %492) #9, !dbg !16
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %494, <4 x float> %357) #9, !dbg !16
  %497 = extractelement <4 x float> %495, i64 0, !dbg !22
  %498 = extractelement <4 x float> %495, i64 1, !dbg !22
  %499 = select i1 %.not, float %498, float %497, !dbg !13
  %500 = extractelement <4 x float> %495, i64 2, !dbg !22
  %501 = extractelement <4 x float> %495, i64 3, !dbg !22
  %address.local304 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local299, !dbg !16
  %address.local305 = getelementptr i8, ptr addrspace(3) %address.local304, i64 1280, !dbg !16
  %502 = load <4 x half>, ptr addrspace(3) %address.local305, align 2, !dbg !16
  %503 = load <4 x half>, ptr addrspace(3) %address.local293, align 2, !dbg !16
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %452, <4 x half> %502, <4 x float> %496) #9, !dbg !16
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %453, <4 x half> %503, <4 x float> %361) #9, !dbg !16
  %506 = extractelement <4 x float> %504, i64 0, !dbg !22
  %507 = extractelement <4 x float> %504, i64 1, !dbg !22
  %508 = extractelement <4 x float> %504, i64 2, !dbg !22
  %509 = extractelement <4 x float> %504, i64 3, !dbg !22
  %510 = load <4 x half>, ptr addrspace(3) %address.local301, align 2, !dbg !16
  %511 = load <4 x half>, ptr addrspace(3) %address.local303, align 2, !dbg !16
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %454, <4 x half> %510, <4 x float> %505) #9, !dbg !16
  %513 = load <4 x half>, ptr addrspace(3) %address.local305, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %453, <4 x half> %511, <4 x float> %364) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !22
  %515 = extractelement <4 x float> %512, i64 0, !dbg !22
  %516 = extractelement <4 x float> %512, i64 1, !dbg !22
  %517 = extractelement <4 x float> %512, i64 2, !dbg !22
  %518 = extractelement <4 x float> %512, i64 3, !dbg !22
  %519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %454, <4 x half> %513, <4 x float> %514) #9, !dbg !16
  %520 = extractelement <4 x float> %519, i64 0, !dbg !22
  %521 = extractelement <4 x float> %519, i64 1, !dbg !22
  %522 = extractelement <4 x float> %519, i64 2, !dbg !22
  %523 = extractelement <4 x float> %519, i64 3, !dbg !22
  %524 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %525 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %524), !dbg !13
  %526 = add nuw nsw i32 %525, 64, !dbg !13
  %527 = xor i32 %525, 16, !dbg !13
  %528 = and i32 %526, 192, !dbg !13
  %529 = icmp samesign ult i32 %527, %528, !dbg !13
  %530 = select i1 %529, i32 %527, i32 %525, !dbg !13
  %531 = shl nuw nsw i32 %530, 2, !dbg !13
  %532 = bitcast float %499 to i32, !dbg !13
  %533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %532), !dbg !13
  %534 = bitcast i32 %533 to float, !dbg !13
  %535 = select i1 %.not, float %497, float %534, !dbg !13
  %536 = select i1 %.not, float %534, float %498, !dbg !13
  %537 = select i1 %.not, float %501, float %500, !dbg !13
  %538 = bitcast float %537 to i32, !dbg !13
  %539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %538), !dbg !13
  %540 = bitcast i32 %539 to float, !dbg !13
  %541 = select i1 %.not, float %500, float %540, !dbg !13
  %542 = select i1 %.not, float %540, float %501, !dbg !13
  %543 = select i1 %.not2, float %541, float %535, !dbg !13
  %544 = xor i32 %525, 32, !dbg !13
  %545 = icmp samesign ult i32 %544, %528, !dbg !13
  %546 = select i1 %545, i32 %544, i32 %525, !dbg !13
  %547 = shl nuw nsw i32 %546, 2, !dbg !13
  %548 = bitcast float %543 to i32, !dbg !13
  %549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %548), !dbg !13
  %550 = bitcast i32 %549 to float, !dbg !13
  %551 = select i1 %.not2, float %535, float %550, !dbg !13
  %552 = select i1 %.not2, float %550, float %541, !dbg !13
  %553 = select i1 %.not2, float %542, float %536, !dbg !13
  %554 = bitcast float %553 to i32, !dbg !13
  %555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %554), !dbg !13
  %556 = bitcast i32 %555 to float, !dbg !13
  %557 = select i1 %.not2, float %536, float %556, !dbg !13
  %558 = select i1 %.not2, float %556, float %542, !dbg !13
  %559 = select i1 %.not, float %507, float %506, !dbg !13
  %560 = bitcast float %559 to i32, !dbg !13
  %561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %560), !dbg !13
  %562 = bitcast i32 %561 to float, !dbg !13
  %563 = select i1 %.not, float %506, float %562, !dbg !13
  %564 = select i1 %.not, float %562, float %507, !dbg !13
  %565 = select i1 %.not, float %509, float %508, !dbg !13
  %566 = bitcast float %565 to i32, !dbg !13
  %567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %566), !dbg !13
  %568 = bitcast i32 %567 to float, !dbg !13
  %569 = select i1 %.not, float %508, float %568, !dbg !13
  %570 = select i1 %.not, float %568, float %509, !dbg !13
  %571 = select i1 %.not2, float %569, float %563, !dbg !13
  %572 = bitcast float %571 to i32, !dbg !13
  %573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %572), !dbg !13
  %574 = bitcast i32 %573 to float, !dbg !13
  %575 = select i1 %.not2, float %563, float %574, !dbg !13
  %576 = select i1 %.not2, float %574, float %569, !dbg !13
  %577 = select i1 %.not2, float %570, float %564, !dbg !13
  %578 = bitcast float %577 to i32, !dbg !13
  %579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %578), !dbg !13
  %580 = bitcast i32 %579 to float, !dbg !13
  %581 = select i1 %.not2, float %564, float %580, !dbg !13
  %582 = select i1 %.not2, float %580, float %570, !dbg !13
  %583 = select i1 %.not, float %516, float %515, !dbg !13
  %584 = bitcast float %583 to i32, !dbg !13
  %585 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %584), !dbg !13
  %586 = bitcast i32 %585 to float, !dbg !13
  %587 = select i1 %.not, float %515, float %586, !dbg !13
  %588 = select i1 %.not, float %586, float %516, !dbg !13
  %589 = select i1 %.not, float %518, float %517, !dbg !13
  %590 = bitcast float %589 to i32, !dbg !13
  %591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %590), !dbg !13
  %592 = bitcast i32 %591 to float, !dbg !13
  %593 = select i1 %.not, float %517, float %592, !dbg !13
  %594 = select i1 %.not, float %592, float %518, !dbg !13
  %595 = select i1 %.not2, float %593, float %587, !dbg !13
  %596 = bitcast float %595 to i32, !dbg !13
  %597 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %596), !dbg !13
  %598 = bitcast i32 %597 to float, !dbg !13
  %599 = select i1 %.not2, float %587, float %598, !dbg !13
  %600 = select i1 %.not2, float %598, float %593, !dbg !13
  %601 = select i1 %.not2, float %594, float %588, !dbg !13
  %602 = bitcast float %601 to i32, !dbg !13
  %603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %602), !dbg !13
  %604 = bitcast i32 %603 to float, !dbg !13
  %605 = select i1 %.not2, float %588, float %604, !dbg !13
  %606 = select i1 %.not2, float %604, float %594, !dbg !13
  %607 = select i1 %.not, float %521, float %520, !dbg !13
  %608 = bitcast float %607 to i32, !dbg !13
  %609 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %608), !dbg !13
  %610 = bitcast i32 %609 to float, !dbg !13
  %611 = select i1 %.not, float %520, float %610, !dbg !13
  %612 = select i1 %.not, float %610, float %521, !dbg !13
  %613 = select i1 %.not, float %523, float %522, !dbg !13
  %614 = bitcast float %613 to i32, !dbg !13
  %615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %614), !dbg !13
  %616 = bitcast i32 %615 to float, !dbg !13
  %617 = select i1 %.not, float %522, float %616, !dbg !13
  %618 = select i1 %.not, float %616, float %523, !dbg !13
  %619 = select i1 %.not2, float %617, float %611, !dbg !13
  %620 = bitcast float %619 to i32, !dbg !13
  %621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %620), !dbg !13
  %622 = bitcast i32 %621 to float, !dbg !13
  %623 = select i1 %.not2, float %611, float %622, !dbg !13
  %624 = select i1 %.not2, float %622, float %617, !dbg !13
  %625 = select i1 %.not2, float %618, float %612, !dbg !13
  %626 = bitcast float %625 to i32, !dbg !13
  %627 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %626), !dbg !13
  %628 = bitcast i32 %627 to float, !dbg !13
  %629 = select i1 %.not2, float %612, float %628, !dbg !13
  %630 = select i1 %.not2, float %628, float %618, !dbg !13
  %631 = or disjoint i64 %56, 31, !dbg !14
  %632 = icmp uge i64 %11, %631, !dbg !14
  %633 = and i1 %33, %632, !dbg !14
  br i1 %633, label %634, label %635, !dbg !14

634:                                              ; preds = %490
  br label %682, !dbg !14

635:                                              ; preds = %490
  %636 = add nsw i64 %11, %25, !dbg !14
  %637 = shl nuw nsw i64 %26, 4, !dbg !14
  %638 = add nsw i64 %636, %637, !dbg !14
  %639 = and i32 %24, 240, !dbg !14
  %index.compact = zext nneg i32 %639 to i64, !dbg !14
  %640 = sub nsw i64 %638, %index.compact, !dbg !14
  %641 = or disjoint i64 %56, %28, !dbg !14
  %642 = add nsw i64 %640, 1, !dbg !23
  %643 = lshr i64 %642, 32, !dbg !24
  %compare.high = trunc i64 %643 to i32, !dbg !24
  %644 = lshr i64 %641, 32, !dbg !24
  %compare.high557 = trunc i64 %644 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high557, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high557, !dbg !24
  %compare.low.base = trunc i64 %642 to i32, !dbg !24
  %compare.low.base558 = trunc i64 %641 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base558, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %645 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base558, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low559 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result560 = select i1 %compare.high.equal, i1 %compare.low559, i1 %compare.high.order, !dbg !24
  %646 = select i1 %compare.result560, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base561 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base558, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local562 = or i32 %compare.local.base561, 2, !dbg !24
  %compare.low563 = icmp ugt i32 %compare.low.base, %compare.local562, !dbg !24
  %compare.result564 = select i1 %compare.high.equal, i1 %compare.low563, i1 %compare.high.order, !dbg !24
  %647 = select i1 %compare.result564, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base565 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base558, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local566 = or i32 %compare.local.base565, 3, !dbg !24
  %compare.low567 = icmp ugt i32 %compare.low.base, %compare.local566, !dbg !24
  %compare.result568 = select i1 %compare.high.equal, i1 %compare.low567, i1 %compare.high.order, !dbg !24
  %648 = select i1 %compare.result568, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base569 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base558, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local570 = or i32 %compare.local.base569, 16, !dbg !24
  %compare.low571 = icmp ugt i32 %compare.low.base, %compare.local570, !dbg !24
  %compare.result572 = select i1 %compare.high.equal, i1 %compare.low571, i1 %compare.high.order, !dbg !24
  %649 = select i1 %compare.result572, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base573 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base558, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local574 = or i32 %compare.local.base573, 17, !dbg !24
  %compare.low575 = icmp ugt i32 %compare.low.base, %compare.local574, !dbg !24
  %compare.result576 = select i1 %compare.high.equal, i1 %compare.low575, i1 %compare.high.order, !dbg !24
  %650 = select i1 %compare.result576, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base577 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base558, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local578 = or i32 %compare.local.base577, 18, !dbg !24
  %compare.low579 = icmp ugt i32 %compare.low.base, %compare.local578, !dbg !24
  %compare.result580 = select i1 %compare.high.equal, i1 %compare.low579, i1 %compare.high.order, !dbg !24
  %651 = select i1 %compare.result580, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base581 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base558, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local582 = or i32 %compare.local.base581, 19, !dbg !24
  %compare.low583 = icmp ugt i32 %compare.low.base, %compare.local582, !dbg !24
  %compare.result584 = select i1 %compare.high.equal, i1 %compare.low583, i1 %compare.high.order, !dbg !24
  %652 = select i1 %compare.result584, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %653 = add nsw i64 %640, 65, !dbg !23
  %654 = lshr i64 %653, 32, !dbg !24
  %compare.high585 = trunc i64 %654 to i32, !dbg !24
  %655 = lshr i64 %641, 32, !dbg !24
  %compare.high586 = trunc i64 %655 to i32, !dbg !24
  %compare.high.equal587 = icmp eq i32 %compare.high585, %compare.high586, !dbg !24
  %compare.high.order588 = icmp ugt i32 %compare.high585, %compare.high586, !dbg !24
  %compare.low.base589 = trunc i64 %653 to i32, !dbg !24
  %compare.low.base590 = trunc i64 %641 to i32, !dbg !24
  %compare.low591 = icmp ugt i32 %compare.low.base589, %compare.low.base590, !dbg !24
  %compare.result592 = select i1 %compare.high.equal587, i1 %compare.low591, i1 %compare.high.order588, !dbg !24
  %656 = select i1 %compare.result592, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base593 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base590, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local594 = or i32 %compare.local.base593, 1, !dbg !24
  %compare.low595 = icmp ugt i32 %compare.low.base589, %compare.local594, !dbg !24
  %compare.result596 = select i1 %compare.high.equal587, i1 %compare.low595, i1 %compare.high.order588, !dbg !24
  %657 = select i1 %compare.result596, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base597 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base590, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local598 = or i32 %compare.local.base597, 2, !dbg !24
  %compare.low599 = icmp ugt i32 %compare.low.base589, %compare.local598, !dbg !24
  %compare.result600 = select i1 %compare.high.equal587, i1 %compare.low599, i1 %compare.high.order588, !dbg !24
  %658 = select i1 %compare.result600, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base601 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base590, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local602 = or i32 %compare.local.base601, 3, !dbg !24
  %compare.low603 = icmp ugt i32 %compare.low.base589, %compare.local602, !dbg !24
  %compare.result604 = select i1 %compare.high.equal587, i1 %compare.low603, i1 %compare.high.order588, !dbg !24
  %659 = select i1 %compare.result604, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base605 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base590, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local606 = or i32 %compare.local.base605, 16, !dbg !24
  %compare.low607 = icmp ugt i32 %compare.low.base589, %compare.local606, !dbg !24
  %compare.result608 = select i1 %compare.high.equal587, i1 %compare.low607, i1 %compare.high.order588, !dbg !24
  %660 = select i1 %compare.result608, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base609 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base590, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local610 = or i32 %compare.local.base609, 17, !dbg !24
  %compare.low611 = icmp ugt i32 %compare.low.base589, %compare.local610, !dbg !24
  %compare.result612 = select i1 %compare.high.equal587, i1 %compare.low611, i1 %compare.high.order588, !dbg !24
  %661 = select i1 %compare.result612, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base613 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base590, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local614 = or i32 %compare.local.base613, 18, !dbg !24
  %compare.low615 = icmp ugt i32 %compare.low.base589, %compare.local614, !dbg !24
  %compare.result616 = select i1 %compare.high.equal587, i1 %compare.low615, i1 %compare.high.order588, !dbg !24
  %662 = select i1 %compare.result616, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base617 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base590, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local618 = or i32 %compare.local.base617, 19, !dbg !24
  %compare.low619 = icmp ugt i32 %compare.low.base589, %compare.local618, !dbg !24
  %compare.result620 = select i1 %compare.high.equal587, i1 %compare.low619, i1 %compare.high.order588, !dbg !24
  %663 = select i1 %compare.result620, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %664 = insertelement <8 x float> poison, float %645, i64 0, !dbg !14
  %665 = insertelement <8 x float> %664, float %646, i64 1, !dbg !14
  %666 = insertelement <8 x float> %665, float %647, i64 2, !dbg !14
  %667 = insertelement <8 x float> %666, float %648, i64 3, !dbg !14
  %668 = insertelement <8 x float> %667, float %649, i64 4, !dbg !14
  %669 = insertelement <8 x float> %668, float %650, i64 5, !dbg !14
  %670 = insertelement <8 x float> %669, float %651, i64 6, !dbg !14
  %671 = insertelement <8 x float> %670, float %652, i64 7, !dbg !14
  %672 = insertvalue [2 x <8 x float>] poison, <8 x float> %671, 0, !dbg !14
  %673 = insertelement <8 x float> poison, float %656, i64 0, !dbg !14
  %674 = insertelement <8 x float> %673, float %657, i64 1, !dbg !14
  %675 = insertelement <8 x float> %674, float %658, i64 2, !dbg !14
  %676 = insertelement <8 x float> %675, float %659, i64 3, !dbg !14
  %677 = insertelement <8 x float> %676, float %660, i64 4, !dbg !14
  %678 = insertelement <8 x float> %677, float %661, i64 5, !dbg !14
  %679 = insertelement <8 x float> %678, float %662, i64 6, !dbg !14
  %680 = insertelement <8 x float> %679, float %663, i64 7, !dbg !14
  %681 = insertvalue [2 x <8 x float>] %672, <8 x float> %680, 1, !dbg !14
  br label %682, !dbg !14

682:                                              ; preds = %634, %635
  %683 = phi [2 x <8 x float>] [ %681, %635 ], [ zeroinitializer, %634 ], !dbg !14
  br label %684, !dbg !14

684:                                              ; preds = %682
  %685 = extractvalue [2 x <8 x float>] %683, 0, !dbg !26
  %686 = insertelement <4 x float> poison, float %551, i64 0, !dbg !26
  %687 = insertelement <4 x float> %686, float %557, i64 1, !dbg !26
  %688 = insertelement <4 x float> %687, float %552, i64 2, !dbg !26
  %689 = insertelement <4 x float> %688, float %558, i64 3, !dbg !26
  %690 = shufflevector <8 x float> %685, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %691 = fadd <4 x float> %689, %690, !dbg !26
  %692 = extractelement <4 x float> %691, i64 0, !dbg !26
  %693 = extractelement <4 x float> %691, i64 1, !dbg !26
  %694 = extractelement <4 x float> %691, i64 2, !dbg !26
  %695 = extractelement <4 x float> %691, i64 3, !dbg !26
  %696 = insertelement <4 x float> poison, float %575, i64 0, !dbg !26
  %697 = insertelement <4 x float> %696, float %581, i64 1, !dbg !26
  %698 = insertelement <4 x float> %697, float %576, i64 2, !dbg !26
  %699 = insertelement <4 x float> %698, float %582, i64 3, !dbg !26
  %700 = shufflevector <8 x float> %685, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %701 = fadd <4 x float> %699, %700, !dbg !26
  %702 = extractelement <4 x float> %701, i64 0, !dbg !26
  %703 = extractelement <4 x float> %701, i64 1, !dbg !26
  %704 = extractelement <4 x float> %701, i64 2, !dbg !26
  %705 = extractelement <4 x float> %701, i64 3, !dbg !26
  %706 = extractvalue [2 x <8 x float>] %683, 1, !dbg !26
  %707 = insertelement <4 x float> poison, float %599, i64 0, !dbg !26
  %708 = insertelement <4 x float> %707, float %605, i64 1, !dbg !26
  %709 = insertelement <4 x float> %708, float %600, i64 2, !dbg !26
  %710 = insertelement <4 x float> %709, float %606, i64 3, !dbg !26
  %711 = shufflevector <8 x float> %706, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %712 = fadd <4 x float> %710, %711, !dbg !26
  %713 = extractelement <4 x float> %712, i64 0, !dbg !26
  %714 = extractelement <4 x float> %712, i64 1, !dbg !26
  %715 = extractelement <4 x float> %712, i64 2, !dbg !26
  %716 = extractelement <4 x float> %712, i64 3, !dbg !26
  %717 = insertelement <4 x float> poison, float %623, i64 0, !dbg !26
  %718 = insertelement <4 x float> %717, float %629, i64 1, !dbg !26
  %719 = insertelement <4 x float> %718, float %624, i64 2, !dbg !26
  %720 = insertelement <4 x float> %719, float %630, i64 3, !dbg !26
  %721 = shufflevector <8 x float> %706, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %722 = fadd <4 x float> %720, %721, !dbg !26
  %723 = extractelement <4 x float> %722, i64 0, !dbg !26
  %724 = extractelement <4 x float> %722, i64 1, !dbg !26
  %725 = extractelement <4 x float> %722, i64 2, !dbg !26
  %726 = extractelement <4 x float> %722, i64 3, !dbg !26
  %727 = call float @llvm.maxnum.f32(float %692, float 0xFFF0000000000000), !dbg !27
  %728 = call float @llvm.maxnum.f32(float %727, float %693), !dbg !27
  %729 = call float @llvm.maxnum.f32(float %728, float %694), !dbg !27
  %730 = call float @llvm.maxnum.f32(float %729, float %695), !dbg !27
  %731 = call float @llvm.maxnum.f32(float %730, float %702), !dbg !27
  %732 = call float @llvm.maxnum.f32(float %731, float %703), !dbg !27
  %733 = call float @llvm.maxnum.f32(float %732, float %704), !dbg !27
  %734 = call float @llvm.maxnum.f32(float %733, float %705), !dbg !27
  %735 = bitcast float %734 to i32, !dbg !27
  %736 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %735), !dbg !27
  %737 = bitcast i32 %736 to float, !dbg !27
  %738 = call float @llvm.maxnum.f32(float %734, float %737), !dbg !27
  %739 = bitcast float %738 to i32, !dbg !27
  %740 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %739), !dbg !27
  %741 = bitcast i32 %740 to float, !dbg !27
  %742 = call float @llvm.maxnum.f32(float %738, float %741), !dbg !27
  %743 = extractvalue [2 x <1 x float>] %50, 0, !dbg !27
  %744 = insertelement <1 x float> poison, float %742, i64 0, !dbg !27
  %745 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %744, <1 x float> %743), !dbg !27
  %746 = call float @llvm.maxnum.f32(float %713, float 0xFFF0000000000000), !dbg !27
  %747 = call float @llvm.maxnum.f32(float %746, float %714), !dbg !27
  %748 = call float @llvm.maxnum.f32(float %747, float %715), !dbg !27
  %749 = call float @llvm.maxnum.f32(float %748, float %716), !dbg !27
  %750 = call float @llvm.maxnum.f32(float %749, float %723), !dbg !27
  %751 = call float @llvm.maxnum.f32(float %750, float %724), !dbg !27
  %752 = call float @llvm.maxnum.f32(float %751, float %725), !dbg !27
  %753 = call float @llvm.maxnum.f32(float %752, float %726), !dbg !27
  %754 = bitcast float %753 to i32, !dbg !27
  %755 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %754), !dbg !27
  %756 = bitcast i32 %755 to float, !dbg !27
  %757 = call float @llvm.maxnum.f32(float %753, float %756), !dbg !27
  %758 = bitcast float %757 to i32, !dbg !27
  %759 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %758), !dbg !27
  %760 = bitcast i32 %759 to float, !dbg !27
  %761 = call float @llvm.maxnum.f32(float %757, float %760), !dbg !27
  %762 = extractvalue [2 x <1 x float>] %50, 1, !dbg !27
  %763 = insertelement <1 x float> poison, float %761, i64 0, !dbg !27
  %764 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %763, <1 x float> %762), !dbg !27
  %765 = shufflevector <1 x float> %745, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %766 = fsub <4 x float> %691, %765, !dbg !28
  %767 = fsub <4 x float> %701, %765, !dbg !28
  %768 = shufflevector <1 x float> %764, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %769 = fsub <4 x float> %712, %768, !dbg !28
  %770 = fsub <4 x float> %722, %768, !dbg !28
  %771 = fsub <1 x float> %743, %745, !dbg !29
  %772 = extractelement <1 x float> %771, i64 0, !dbg !30
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !30
  %774 = extractvalue [2 x <1 x float>] %49, 0, !dbg !31
  %775 = extractelement <1 x float> %774, i64 0, !dbg !31
  %776 = fmul float %775, %773, !dbg !31
  %777 = fsub <1 x float> %762, %764, !dbg !29
  %778 = extractelement <1 x float> %777, i64 0, !dbg !30
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !30
  %780 = extractvalue [2 x <1 x float>] %49, 1, !dbg !31
  %781 = extractelement <1 x float> %780, i64 0, !dbg !31
  %782 = fmul float %781, %779, !dbg !31
  %783 = extractelement <4 x float> %766, i64 0, !dbg !32
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !32
  %785 = insertelement <4 x float> poison, float %784, i64 0, !dbg !34
  %786 = extractelement <4 x float> %766, i64 1, !dbg !32
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !32
  %788 = insertelement <4 x float> %785, float %787, i64 1, !dbg !34
  %789 = fadd float %784, %787, !dbg !33
  %790 = extractelement <4 x float> %766, i64 2, !dbg !32
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !32
  %792 = insertelement <4 x float> %788, float %791, i64 2, !dbg !34
  %793 = fadd float %789, %791, !dbg !33
  %794 = extractelement <4 x float> %766, i64 3, !dbg !32
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !32
  %796 = insertelement <4 x float> %792, float %795, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %796) #9, !dbg !34, !frisk.register_lifetimes !1
  %797 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %797) #9, !dbg !33, !frisk.register_lifetimes !1
  %798 = fadd float %793, %795, !dbg !33
  %799 = extractelement <4 x float> %767, i64 0, !dbg !32
  %800 = call float @llvm.amdgcn.exp2.f32(float %799), !dbg !32
  %801 = insertelement <4 x float> poison, float %800, i64 0, !dbg !34
  %802 = fadd float %798, %800, !dbg !33
  %803 = extractelement <4 x float> %767, i64 1, !dbg !32
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !32
  %805 = insertelement <4 x float> %801, float %804, i64 1, !dbg !34
  %806 = fadd float %802, %804, !dbg !33
  %807 = extractelement <4 x float> %767, i64 2, !dbg !32
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !32
  %809 = insertelement <4 x float> %805, float %808, i64 2, !dbg !34
  %810 = fadd float %806, %808, !dbg !33
  %811 = extractelement <4 x float> %767, i64 3, !dbg !32
  %812 = call float @llvm.amdgcn.exp2.f32(float %811), !dbg !32
  %813 = insertelement <4 x float> %809, float %812, i64 3, !dbg !34
  %math.result48 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %813) #9, !dbg !34, !frisk.register_lifetimes !1
  %814 = fptrunc <4 x float> %math.result48 to <4 x half>, !dbg !34
  %math.packed49 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %814) #9, !dbg !33, !frisk.register_lifetimes !1
  %815 = fadd float %810, %812, !dbg !33
  %816 = extractelement <4 x float> %769, i64 0, !dbg !32
  %817 = call float @llvm.amdgcn.exp2.f32(float %816), !dbg !32
  %818 = insertelement <4 x float> poison, float %817, i64 0, !dbg !34
  %819 = extractelement <4 x float> %769, i64 1, !dbg !32
  %820 = call float @llvm.amdgcn.exp2.f32(float %819), !dbg !32
  %821 = insertelement <4 x float> %818, float %820, i64 1, !dbg !34
  %822 = fadd float %817, %820, !dbg !33
  %823 = extractelement <4 x float> %769, i64 2, !dbg !32
  %824 = call float @llvm.amdgcn.exp2.f32(float %823), !dbg !32
  %825 = insertelement <4 x float> %821, float %824, i64 2, !dbg !34
  %826 = fadd float %822, %824, !dbg !33
  %827 = extractelement <4 x float> %769, i64 3, !dbg !32
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !32
  %829 = insertelement <4 x float> %825, float %828, i64 3, !dbg !34
  %math.result50 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %829) #9, !dbg !34, !frisk.register_lifetimes !1
  %830 = fptrunc <4 x float> %math.result50 to <4 x half>, !dbg !34
  %math.packed51 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %830) #9, !dbg !33, !frisk.register_lifetimes !1
  %831 = fadd float %826, %828, !dbg !33
  %832 = extractelement <4 x float> %770, i64 0, !dbg !32
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !32
  %834 = insertelement <4 x float> poison, float %833, i64 0, !dbg !34
  %835 = fadd float %831, %833, !dbg !33
  %836 = extractelement <4 x float> %770, i64 1, !dbg !32
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !32
  %838 = insertelement <4 x float> %834, float %837, i64 1, !dbg !34
  %839 = fadd float %835, %837, !dbg !33
  %840 = extractelement <4 x float> %770, i64 2, !dbg !32
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !32
  %842 = insertelement <4 x float> %838, float %841, i64 2, !dbg !34
  %843 = fadd float %839, %841, !dbg !33
  %844 = extractelement <4 x float> %770, i64 3, !dbg !32
  %845 = call float @llvm.amdgcn.exp2.f32(float %844), !dbg !32
  %846 = insertelement <4 x float> %842, float %845, i64 3, !dbg !34
  %math.result52 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %846) #9, !dbg !34, !frisk.register_lifetimes !1
  %847 = fptrunc <4 x float> %math.result52 to <4 x half>, !dbg !34
  %math.packed53 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %847) #9, !dbg !33, !frisk.register_lifetimes !1
  %848 = fadd float %843, %845, !dbg !33
  %849 = bitcast float %815 to i32, !dbg !33
  %850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %849), !dbg !33
  %851 = bitcast i32 %850 to float, !dbg !33
  %852 = fadd float %815, %851, !dbg !33
  %853 = bitcast float %852 to i32, !dbg !33
  %854 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %853), !dbg !33
  %855 = bitcast i32 %854 to float, !dbg !33
  %856 = fadd float %852, %855, !dbg !33
  %857 = fadd float %856, %776, !dbg !33
  %858 = bitcast float %848 to i32, !dbg !33
  %859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %858), !dbg !33
  %860 = bitcast i32 %859 to float, !dbg !33
  %861 = fadd float %848, %860, !dbg !33
  %862 = bitcast float %861 to i32, !dbg !33
  %863 = call i32 @llvm.amdgcn.ds.bpermute(i32 %547, i32 %862), !dbg !33
  %864 = bitcast i32 %863 to float, !dbg !33
  %865 = fadd float %861, %864, !dbg !33
  %866 = fadd float %865, %782, !dbg !33
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread306 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local307 = shl nuw nsw i32 %address.thread306, 3, !dbg !15
  %address.local308 = and i32 %address.local307, 504, !dbg !15
  %address.local309 = zext nneg i32 %address.local308 to i64, !dbg !15
  %address.local310 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local309, !dbg !15
  %867 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local310, i32 0) #0, !dbg !15
  %868 = bitcast <4 x i32> %867 to <8 x half>, !dbg !15
  %address.local311 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local309, !dbg !15
  %address.local312 = getelementptr i8, ptr addrspace(3) %address.local311, i64 1024, !dbg !15
  %869 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local312, i32 0) #0, !dbg !15
  %870 = bitcast <4 x i32> %869 to <8 x half>, !dbg !15
  %871 = add nsw i64 %87, %8, !dbg !18
  %872 = shl i64 %871, 19, !dbg !18
  %873 = and i64 %872, 16252928, !dbg !18
  %874 = add i64 %873, %93, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %875 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %874, !dbg !18
  %876 = getelementptr i8, ptr addrspace(1) %875, i64 64, !dbg !18
  br label %877, !dbg !18

877:                                              ; preds = %880, %684
  %878 = phi i64 [ %906, %880 ], [ %25, %684 ], !dbg !18
  %879 = icmp samesign ult i64 %878, 128, !dbg !18
  br i1 %879, label %880, label %907, !dbg !18

880:                                              ; preds = %877
  %881 = shl nuw nsw i64 %878, 3, !dbg !18
  %882 = and i64 %881, 24, !dbg !18
  %883 = shl nuw nsw i64 %878, 3, !dbg !18
  %884 = and i64 %883, 512, !dbg !18
  %885 = shl nuw nsw i64 %878, 5, !dbg !18
  %886 = and i64 %885, 3968, !dbg !18
  %887 = getelementptr [2 x i8], ptr addrspace(1) %876, i64 %886, !dbg !18
  %888 = getelementptr [2 x i8], ptr addrspace(1) %887, i64 %882, !dbg !18
  %889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %884, !dbg !18
  %890 = ptrtoint ptr addrspace(3) %889 to i64, !dbg !18
  %891 = trunc i64 %890 to i32, !dbg !18
  %892 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %891), !dbg !18
  %893 = zext i32 %892 to i64, !dbg !18
  %894 = inttoptr i64 %893 to ptr addrspace(3), !dbg !18
  %895 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %896 = ptrtoint ptr addrspace(1) %888 to i64, !dbg !18
  %897 = sub i64 %896, %895, !dbg !18
  %dma.buffer.offset633 = trunc i64 %897 to i32, !dbg !18
  %898 = trunc i64 %895 to i32, !dbg !18
  %899 = insertelement <4 x i32> poison, i32 %898, i64 0, !dbg !18
  %900 = lshr i64 %895, 32, !dbg !18
  %901 = trunc i64 %900 to i32, !dbg !18
  %902 = insertelement <4 x i32> %899, i32 %901, i64 1, !dbg !18
  %903 = insertelement <4 x i32> %902, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc634 = insertelement <4 x i32> %903, i32 131072, i64 3, !dbg !18
  %904 = ptrtoint ptr addrspace(3) %894 to i32, !dbg !18
  %dma.buffer.lds635 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %904), !dbg !18
  %905 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds635, i32 %dma.buffer.offset633, <4 x i32> %dma.buffer.rsrc634) #10, !dbg !18
  %906 = or disjoint i64 %25, 256, !dbg !18
  br label %877, !dbg !18

907:                                              ; preds = %877
  %908 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %909 = extractvalue [2 x <8 x float>] %48, 0, !dbg !35
  %910 = shufflevector <8 x float> %909, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %911 = insertelement <4 x float> poison, float %773, i64 0, !dbg !35
  %912 = shufflevector <4 x float> %911, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %913 = fmul <4 x float> %910, %912, !dbg !35
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %908, <4 x float> %913) #9, !dbg !15
  %915 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %915, <4 x float> %914) #9, !dbg !15
  %917 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %918 = shufflevector <8 x float> %909, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %919 = fmul <4 x float> %918, %912, !dbg !35
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %917, <4 x float> %919) #9, !dbg !15
  %921 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %921, <4 x float> %920) #9, !dbg !15
  %923 = extractvalue [2 x <8 x float>] %48, 1, !dbg !35
  %924 = shufflevector <8 x float> %923, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %925 = insertelement <4 x float> poison, float %779, i64 0, !dbg !35
  %926 = shufflevector <4 x float> %925, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %927 = fmul <4 x float> %924, %926, !dbg !35
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %908, <4 x float> %927) #9, !dbg !15
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed53, <4 x half> %915, <4 x float> %928) #9, !dbg !15
  %930 = shufflevector <8 x float> %923, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %931 = fmul <4 x float> %930, %926, !dbg !35
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %917, <4 x float> %931) #9, !dbg !15
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed53, <4 x half> %921, <4 x float> %932) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread313 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local314 = shl nuw nsw i32 %address.thread313, 3, !dbg !15
  %address.local315 = and i32 %address.local314, 504, !dbg !15
  %address.local316 = zext nneg i32 %address.local315 to i64, !dbg !15
  %address.local317 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local316, !dbg !15
  %934 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local317, i32 0) #0, !dbg !15
  %935 = bitcast <4 x i32> %934 to <8 x half>, !dbg !15
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local316, !dbg !15
  %address.local319 = getelementptr i8, ptr addrspace(3) %address.local318, i64 1024, !dbg !15
  %936 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local319, i32 0) #0, !dbg !15
  %937 = bitcast <4 x i32> %936 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %938 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %874, !dbg !18
  %939 = getelementptr i8, ptr addrspace(1) %938, i64 128, !dbg !18
  br label %940, !dbg !18

940:                                              ; preds = %943, %907
  %941 = phi i64 [ %969, %943 ], [ %25, %907 ], !dbg !18
  %942 = icmp samesign ult i64 %941, 128, !dbg !18
  br i1 %942, label %943, label %970, !dbg !18

943:                                              ; preds = %940
  %944 = shl nuw nsw i64 %941, 3, !dbg !18
  %945 = and i64 %944, 24, !dbg !18
  %946 = shl nuw nsw i64 %941, 3, !dbg !18
  %947 = and i64 %946, 512, !dbg !18
  %948 = shl nuw nsw i64 %941, 5, !dbg !18
  %949 = and i64 %948, 3968, !dbg !18
  %950 = getelementptr [2 x i8], ptr addrspace(1) %939, i64 %949, !dbg !18
  %951 = getelementptr [2 x i8], ptr addrspace(1) %950, i64 %945, !dbg !18
  %952 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %947, !dbg !18
  %953 = ptrtoint ptr addrspace(3) %952 to i64, !dbg !18
  %954 = trunc i64 %953 to i32, !dbg !18
  %955 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %954), !dbg !18
  %956 = zext i32 %955 to i64, !dbg !18
  %957 = inttoptr i64 %956 to ptr addrspace(3), !dbg !18
  %958 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %959 = ptrtoint ptr addrspace(1) %951 to i64, !dbg !18
  %960 = sub i64 %959, %958, !dbg !18
  %dma.buffer.offset636 = trunc i64 %960 to i32, !dbg !18
  %961 = trunc i64 %958 to i32, !dbg !18
  %962 = insertelement <4 x i32> poison, i32 %961, i64 0, !dbg !18
  %963 = lshr i64 %958, 32, !dbg !18
  %964 = trunc i64 %963 to i32, !dbg !18
  %965 = insertelement <4 x i32> %962, i32 %964, i64 1, !dbg !18
  %966 = insertelement <4 x i32> %965, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc637 = insertelement <4 x i32> %966, i32 131072, i64 3, !dbg !18
  %967 = ptrtoint ptr addrspace(3) %957 to i32, !dbg !18
  %dma.buffer.lds638 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %967), !dbg !18
  %968 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds638, i32 %dma.buffer.offset636, <4 x i32> %dma.buffer.rsrc637) #10, !dbg !18
  %969 = or disjoint i64 %25, 256, !dbg !18
  br label %940, !dbg !18

970:                                              ; preds = %940
  %971 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %972 = extractvalue [2 x <8 x float>] %51, 0, !dbg !35
  %973 = shufflevector <8 x float> %972, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %974 = fmul <4 x float> %973, %912, !dbg !35
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %971, <4 x float> %974) #9, !dbg !15
  %976 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %976, <4 x float> %975) #9, !dbg !15
  %978 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %979 = shufflevector <8 x float> %972, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %980 = fmul <4 x float> %979, %912, !dbg !35
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %978, <4 x float> %980) #9, !dbg !15
  %982 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %982, <4 x float> %981) #9, !dbg !15
  %984 = extractvalue [2 x <8 x float>] %51, 1, !dbg !35
  %985 = shufflevector <8 x float> %984, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %986 = fmul <4 x float> %985, %926, !dbg !35
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %971, <4 x float> %986) #9, !dbg !15
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed53, <4 x half> %976, <4 x float> %987) #9, !dbg !15
  %989 = shufflevector <8 x float> %984, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %990 = fmul <4 x float> %989, %926, !dbg !35
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %978, <4 x float> %990) #9, !dbg !15
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed53, <4 x half> %982, <4 x float> %991) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread320 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local321 = shl nuw nsw i32 %address.thread320, 3, !dbg !15
  %address.local322 = and i32 %address.local321, 504, !dbg !15
  %address.local323 = zext nneg i32 %address.local322 to i64, !dbg !15
  %address.local324 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local323, !dbg !15
  %993 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local324, i32 0) #0, !dbg !15
  %994 = bitcast <4 x i32> %993 to <8 x half>, !dbg !15
  %address.local325 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local323, !dbg !15
  %address.local326 = getelementptr i8, ptr addrspace(3) %address.local325, i64 1024, !dbg !15
  %995 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !15
  %996 = bitcast <4 x i32> %995 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %997 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %874, !dbg !18
  %998 = getelementptr i8, ptr addrspace(1) %997, i64 192, !dbg !18
  br label %999, !dbg !18

999:                                              ; preds = %1002, %970
  %1000 = phi i64 [ %1028, %1002 ], [ %25, %970 ], !dbg !18
  %1001 = icmp samesign ult i64 %1000, 128, !dbg !18
  br i1 %1001, label %1002, label %1029, !dbg !18

1002:                                             ; preds = %999
  %1003 = shl nuw nsw i64 %1000, 3, !dbg !18
  %1004 = and i64 %1003, 24, !dbg !18
  %1005 = shl nuw nsw i64 %1000, 3, !dbg !18
  %1006 = and i64 %1005, 512, !dbg !18
  %1007 = shl nuw nsw i64 %1000, 5, !dbg !18
  %1008 = and i64 %1007, 3968, !dbg !18
  %1009 = getelementptr [2 x i8], ptr addrspace(1) %998, i64 %1008, !dbg !18
  %1010 = getelementptr [2 x i8], ptr addrspace(1) %1009, i64 %1004, !dbg !18
  %1011 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1006, !dbg !18
  %1012 = ptrtoint ptr addrspace(3) %1011 to i64, !dbg !18
  %1013 = trunc i64 %1012 to i32, !dbg !18
  %1014 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1013), !dbg !18
  %1015 = zext i32 %1014 to i64, !dbg !18
  %1016 = inttoptr i64 %1015 to ptr addrspace(3), !dbg !18
  %1017 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1018 = ptrtoint ptr addrspace(1) %1010 to i64, !dbg !18
  %1019 = sub i64 %1018, %1017, !dbg !18
  %dma.buffer.offset639 = trunc i64 %1019 to i32, !dbg !18
  %1020 = trunc i64 %1017 to i32, !dbg !18
  %1021 = insertelement <4 x i32> poison, i32 %1020, i64 0, !dbg !18
  %1022 = lshr i64 %1017, 32, !dbg !18
  %1023 = trunc i64 %1022 to i32, !dbg !18
  %1024 = insertelement <4 x i32> %1021, i32 %1023, i64 1, !dbg !18
  %1025 = insertelement <4 x i32> %1024, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc640 = insertelement <4 x i32> %1025, i32 131072, i64 3, !dbg !18
  %1026 = ptrtoint ptr addrspace(3) %1016 to i32, !dbg !18
  %dma.buffer.lds641 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1026), !dbg !18
  %1027 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds641, i32 %dma.buffer.offset639, <4 x i32> %dma.buffer.rsrc640) #10, !dbg !18
  %1028 = or disjoint i64 %25, 256, !dbg !18
  br label %999, !dbg !18

1029:                                             ; preds = %999
  %1030 = shufflevector <8 x half> %994, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1031 = extractvalue [2 x <8 x float>] %52, 0, !dbg !35
  %1032 = shufflevector <8 x float> %1031, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1033 = fmul <4 x float> %1032, %912, !dbg !35
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1030, <4 x float> %1033) #9, !dbg !15
  %1035 = shufflevector <8 x half> %996, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %1035, <4 x float> %1034) #9, !dbg !15
  %1037 = shufflevector <8 x half> %994, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1038 = shufflevector <8 x float> %1031, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1039 = fmul <4 x float> %1038, %912, !dbg !35
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1037, <4 x float> %1039) #9, !dbg !15
  %1041 = shufflevector <8 x half> %996, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %1041, <4 x float> %1040) #9, !dbg !15
  %1043 = extractvalue [2 x <8 x float>] %52, 1, !dbg !35
  %1044 = shufflevector <8 x float> %1043, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1045 = fmul <4 x float> %1044, %926, !dbg !35
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %1030, <4 x float> %1045) #9, !dbg !15
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed53, <4 x half> %1035, <4 x float> %1046) #9, !dbg !15
  %1048 = shufflevector <8 x float> %1043, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1049 = fmul <4 x float> %1048, %926, !dbg !35
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %1037, <4 x float> %1049) #9, !dbg !15
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed53, <4 x half> %1041, <4 x float> %1050) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread327 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local328 = shl nuw nsw i32 %address.thread327, 3, !dbg !15
  %address.local329 = and i32 %address.local328, 504, !dbg !15
  %address.local330 = zext nneg i32 %address.local329 to i64, !dbg !15
  %address.local331 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !15
  %1052 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !15
  %1053 = bitcast <4 x i32> %1052 to <8 x half>, !dbg !15
  %address.local332 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !15
  %address.local333 = getelementptr i8, ptr addrspace(3) %address.local332, i64 1024, !dbg !15
  %1054 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local333, i32 0) #0, !dbg !15
  %1055 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !15
  %1056 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1057 = extractvalue [2 x <8 x float>] %53, 0, !dbg !35
  %1058 = shufflevector <8 x float> %1057, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1059 = fmul <4 x float> %1058, %912, !dbg !35
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1056, <4 x float> %1059) #9, !dbg !15
  %1061 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %1061, <4 x float> %1060) #9, !dbg !15
  %1063 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1064 = shufflevector <8 x float> %1057, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1065 = fmul <4 x float> %1064, %912, !dbg !35
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1063, <4 x float> %1065) #9, !dbg !15
  %1067 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %1067, <4 x float> %1066) #9, !dbg !15
  %1069 = extractvalue [2 x <8 x float>] %53, 1, !dbg !35
  %1070 = shufflevector <8 x float> %1069, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1071 = fmul <4 x float> %1070, %926, !dbg !35
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %1056, <4 x float> %1071) #9, !dbg !15
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed53, <4 x half> %1061, <4 x float> %1072) #9, !dbg !15
  %1074 = shufflevector <8 x float> %1069, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1075 = fmul <4 x float> %1074, %926, !dbg !35
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed51, <4 x half> %1063, <4 x float> %1075) #9, !dbg !15
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed53, <4 x half> %1067, <4 x float> %1076) #9, !dbg !15
  %1078 = shufflevector <4 x float> %916, <4 x float> %922, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1079 = insertvalue [2 x <8 x float>] poison, <8 x float> %1078, 0, !dbg !7
  %1080 = shufflevector <4 x float> %929, <4 x float> %933, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1081 = insertvalue [2 x <8 x float>] %1079, <8 x float> %1080, 1, !dbg !7
  %1082 = insertelement <1 x float> poison, float %857, i64 0, !dbg !7
  %1083 = insertvalue [2 x <1 x float>] poison, <1 x float> %1082, 0, !dbg !7
  %1084 = insertelement <1 x float> poison, float %866, i64 0, !dbg !7
  %1085 = insertvalue [2 x <1 x float>] %1083, <1 x float> %1084, 1, !dbg !7
  %1086 = insertvalue [2 x <1 x float>] poison, <1 x float> %745, 0, !dbg !7
  %1087 = insertvalue [2 x <1 x float>] %1086, <1 x float> %764, 1, !dbg !7
  %1088 = shufflevector <4 x float> %977, <4 x float> %983, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1089 = insertvalue [2 x <8 x float>] poison, <8 x float> %1088, 0, !dbg !7
  %1090 = shufflevector <4 x float> %988, <4 x float> %992, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1091 = insertvalue [2 x <8 x float>] %1089, <8 x float> %1090, 1, !dbg !7
  %1092 = shufflevector <4 x float> %1036, <4 x float> %1042, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1093 = insertvalue [2 x <8 x float>] poison, <8 x float> %1092, 0, !dbg !7
  %1094 = shufflevector <4 x float> %1047, <4 x float> %1051, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1095 = insertvalue [2 x <8 x float>] %1093, <8 x float> %1094, 1, !dbg !7
  %1096 = shufflevector <4 x float> %1062, <4 x float> %1068, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1097 = insertvalue [2 x <8 x float>] poison, <8 x float> %1096, 0, !dbg !7
  %1098 = shufflevector <4 x float> %1073, <4 x float> %1077, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1099 = insertvalue [2 x <8 x float>] %1097, <8 x float> %1098, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1100 = add nuw nsw i64 %47, 1, !dbg !7
  br label %46, !dbg !7

1101:                                             ; preds = %46
  %1102 = extractvalue [2 x <8 x float>] %53, 0, !dbg !7
  %1103 = extractelement <8 x float> %1102, i64 0, !dbg !7
  %1104 = extractelement <8 x float> %1102, i64 1, !dbg !7
  %1105 = extractelement <8 x float> %1102, i64 2, !dbg !7
  %1106 = extractelement <8 x float> %1102, i64 3, !dbg !7
  %1107 = select i1 %.not, float %1104, float %1103, !dbg !7
  %1108 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1109 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1108), !dbg !7
  %1110 = add nuw nsw i32 %1109, 64, !dbg !7
  %1111 = xor i32 %1109, 16, !dbg !7
  %1112 = and i32 %1110, 192, !dbg !7
  %1113 = icmp samesign ult i32 %1111, %1112, !dbg !7
  %1114 = select i1 %1113, i32 %1111, i32 %1109, !dbg !7
  %1115 = shl nuw nsw i32 %1114, 2, !dbg !7
  %1116 = bitcast float %1107 to i32, !dbg !7
  %1117 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1116), !dbg !7
  %1118 = bitcast i32 %1117 to float, !dbg !7
  %1119 = select i1 %.not, float %1103, float %1118, !dbg !7
  %1120 = select i1 %.not, float %1118, float %1104, !dbg !7
  %1121 = select i1 %.not, float %1106, float %1105, !dbg !7
  %1122 = bitcast float %1121 to i32, !dbg !7
  %1123 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1122), !dbg !7
  %1124 = bitcast i32 %1123 to float, !dbg !7
  %1125 = select i1 %.not, float %1105, float %1124, !dbg !7
  %1126 = select i1 %.not, float %1124, float %1106, !dbg !7
  %1127 = select i1 %.not2, float %1125, float %1119, !dbg !7
  %1128 = xor i32 %1109, 32, !dbg !7
  %1129 = icmp samesign ult i32 %1128, %1112, !dbg !7
  %1130 = select i1 %1129, i32 %1128, i32 %1109, !dbg !7
  %1131 = shl nuw nsw i32 %1130, 2, !dbg !7
  %1132 = bitcast float %1127 to i32, !dbg !7
  %1133 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1132), !dbg !7
  %1134 = bitcast i32 %1133 to float, !dbg !7
  %1135 = select i1 %.not2, float %1119, float %1134, !dbg !7
  %1136 = select i1 %.not2, float %1134, float %1125, !dbg !7
  %1137 = select i1 %.not2, float %1126, float %1120, !dbg !7
  %1138 = bitcast float %1137 to i32, !dbg !7
  %1139 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1138), !dbg !7
  %1140 = bitcast i32 %1139 to float, !dbg !7
  %1141 = select i1 %.not2, float %1120, float %1140, !dbg !7
  %1142 = select i1 %.not2, float %1140, float %1126, !dbg !7
  %1143 = extractvalue [2 x <8 x float>] %52, 0, !dbg !7
  %1144 = extractelement <8 x float> %1143, i64 0, !dbg !7
  %1145 = extractelement <8 x float> %1143, i64 1, !dbg !7
  %1146 = extractelement <8 x float> %1143, i64 2, !dbg !7
  %1147 = extractelement <8 x float> %1143, i64 3, !dbg !7
  %1148 = select i1 %.not, float %1145, float %1144, !dbg !7
  %1149 = bitcast float %1148 to i32, !dbg !7
  %1150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1149), !dbg !7
  %1151 = bitcast i32 %1150 to float, !dbg !7
  %1152 = select i1 %.not, float %1144, float %1151, !dbg !7
  %1153 = select i1 %.not, float %1151, float %1145, !dbg !7
  %1154 = select i1 %.not, float %1147, float %1146, !dbg !7
  %1155 = bitcast float %1154 to i32, !dbg !7
  %1156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1155), !dbg !7
  %1157 = bitcast i32 %1156 to float, !dbg !7
  %1158 = select i1 %.not, float %1146, float %1157, !dbg !7
  %1159 = select i1 %.not, float %1157, float %1147, !dbg !7
  %1160 = select i1 %.not2, float %1158, float %1152, !dbg !7
  %1161 = bitcast float %1160 to i32, !dbg !7
  %1162 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1161), !dbg !7
  %1163 = bitcast i32 %1162 to float, !dbg !7
  %1164 = select i1 %.not2, float %1152, float %1163, !dbg !7
  %1165 = select i1 %.not2, float %1163, float %1158, !dbg !7
  %1166 = select i1 %.not2, float %1159, float %1153, !dbg !7
  %1167 = bitcast float %1166 to i32, !dbg !7
  %1168 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1167), !dbg !7
  %1169 = bitcast i32 %1168 to float, !dbg !7
  %1170 = select i1 %.not2, float %1153, float %1169, !dbg !7
  %1171 = select i1 %.not2, float %1169, float %1159, !dbg !7
  %1172 = extractvalue [2 x <8 x float>] %51, 0, !dbg !7
  %1173 = extractelement <8 x float> %1172, i64 0, !dbg !7
  %1174 = extractelement <8 x float> %1172, i64 1, !dbg !7
  %1175 = extractelement <8 x float> %1172, i64 2, !dbg !7
  %1176 = extractelement <8 x float> %1172, i64 3, !dbg !7
  %1177 = select i1 %.not, float %1174, float %1173, !dbg !7
  %1178 = bitcast float %1177 to i32, !dbg !7
  %1179 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1178), !dbg !7
  %1180 = bitcast i32 %1179 to float, !dbg !7
  %1181 = select i1 %.not, float %1173, float %1180, !dbg !7
  %1182 = select i1 %.not, float %1180, float %1174, !dbg !7
  %1183 = select i1 %.not, float %1176, float %1175, !dbg !7
  %1184 = bitcast float %1183 to i32, !dbg !7
  %1185 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1184), !dbg !7
  %1186 = bitcast i32 %1185 to float, !dbg !7
  %1187 = select i1 %.not, float %1175, float %1186, !dbg !7
  %1188 = select i1 %.not, float %1186, float %1176, !dbg !7
  %1189 = select i1 %.not2, float %1187, float %1181, !dbg !7
  %1190 = bitcast float %1189 to i32, !dbg !7
  %1191 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1190), !dbg !7
  %1192 = bitcast i32 %1191 to float, !dbg !7
  %1193 = select i1 %.not2, float %1181, float %1192, !dbg !7
  %1194 = select i1 %.not2, float %1192, float %1187, !dbg !7
  %1195 = select i1 %.not2, float %1188, float %1182, !dbg !7
  %1196 = bitcast float %1195 to i32, !dbg !7
  %1197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1196), !dbg !7
  %1198 = bitcast i32 %1197 to float, !dbg !7
  %1199 = select i1 %.not2, float %1182, float %1198, !dbg !7
  %1200 = select i1 %.not2, float %1198, float %1188, !dbg !7
  %1201 = extractvalue [2 x <8 x float>] %48, 0, !dbg !7
  %1202 = extractelement <8 x float> %1201, i64 0, !dbg !7
  %1203 = extractelement <8 x float> %1201, i64 1, !dbg !7
  %1204 = extractelement <8 x float> %1201, i64 2, !dbg !7
  %1205 = extractelement <8 x float> %1201, i64 3, !dbg !7
  %1206 = select i1 %.not, float %1203, float %1202, !dbg !7
  %1207 = bitcast float %1206 to i32, !dbg !7
  %1208 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1207), !dbg !7
  %1209 = bitcast i32 %1208 to float, !dbg !7
  %1210 = select i1 %.not, float %1202, float %1209, !dbg !7
  %1211 = select i1 %.not, float %1209, float %1203, !dbg !7
  %1212 = select i1 %.not, float %1205, float %1204, !dbg !7
  %1213 = bitcast float %1212 to i32, !dbg !7
  %1214 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1213), !dbg !7
  %1215 = bitcast i32 %1214 to float, !dbg !7
  %1216 = select i1 %.not, float %1204, float %1215, !dbg !7
  %1217 = select i1 %.not, float %1215, float %1205, !dbg !7
  %1218 = select i1 %.not2, float %1216, float %1210, !dbg !7
  %1219 = bitcast float %1218 to i32, !dbg !7
  %1220 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1219), !dbg !7
  %1221 = select i1 %.not2, float %1217, float %1211, !dbg !7
  %1222 = bitcast float %1221 to i32, !dbg !7
  %1223 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1222), !dbg !7
  %1224 = extractvalue [2 x <1 x float>] %49, 0, !dbg !19
  %1225 = shufflevector <1 x float> %1224, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1226 = extractelement <8 x float> %1102, i64 4, !dbg !7
  %1227 = extractelement <8 x float> %1102, i64 5, !dbg !7
  %1228 = extractelement <8 x float> %1102, i64 6, !dbg !7
  %1229 = extractelement <8 x float> %1102, i64 7, !dbg !7
  %1230 = select i1 %.not, float %1227, float %1226, !dbg !7
  %1231 = bitcast float %1230 to i32, !dbg !7
  %1232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1231), !dbg !7
  %1233 = bitcast i32 %1232 to float, !dbg !7
  %1234 = select i1 %.not, float %1226, float %1233, !dbg !7
  %1235 = select i1 %.not, float %1233, float %1227, !dbg !7
  %1236 = select i1 %.not, float %1229, float %1228, !dbg !7
  %1237 = bitcast float %1236 to i32, !dbg !7
  %1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1237), !dbg !7
  %1239 = bitcast i32 %1238 to float, !dbg !7
  %1240 = select i1 %.not, float %1228, float %1239, !dbg !7
  %1241 = select i1 %.not, float %1239, float %1229, !dbg !7
  %1242 = select i1 %.not2, float %1240, float %1234, !dbg !7
  %1243 = bitcast float %1242 to i32, !dbg !7
  %1244 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1243), !dbg !7
  %1245 = bitcast i32 %1244 to float, !dbg !7
  %1246 = select i1 %.not2, float %1234, float %1245, !dbg !7
  %1247 = select i1 %.not2, float %1245, float %1240, !dbg !7
  %1248 = select i1 %.not2, float %1241, float %1235, !dbg !7
  %1249 = bitcast float %1248 to i32, !dbg !7
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1249), !dbg !7
  %1251 = bitcast i32 %1250 to float, !dbg !7
  %1252 = select i1 %.not2, float %1235, float %1251, !dbg !7
  %1253 = select i1 %.not2, float %1251, float %1241, !dbg !7
  %1254 = extractelement <8 x float> %1143, i64 4, !dbg !7
  %1255 = extractelement <8 x float> %1143, i64 5, !dbg !7
  %1256 = extractelement <8 x float> %1143, i64 6, !dbg !7
  %1257 = extractelement <8 x float> %1143, i64 7, !dbg !7
  %1258 = select i1 %.not, float %1255, float %1254, !dbg !7
  %1259 = bitcast float %1258 to i32, !dbg !7
  %1260 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1259), !dbg !7
  %1261 = bitcast i32 %1260 to float, !dbg !7
  %1262 = select i1 %.not, float %1254, float %1261, !dbg !7
  %1263 = select i1 %.not, float %1261, float %1255, !dbg !7
  %1264 = select i1 %.not, float %1257, float %1256, !dbg !7
  %1265 = bitcast float %1264 to i32, !dbg !7
  %1266 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1265), !dbg !7
  %1267 = bitcast i32 %1266 to float, !dbg !7
  %1268 = select i1 %.not, float %1256, float %1267, !dbg !7
  %1269 = select i1 %.not, float %1267, float %1257, !dbg !7
  %1270 = select i1 %.not2, float %1268, float %1262, !dbg !7
  %1271 = bitcast float %1270 to i32, !dbg !7
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1271), !dbg !7
  %1273 = bitcast i32 %1272 to float, !dbg !7
  %1274 = select i1 %.not2, float %1262, float %1273, !dbg !7
  %1275 = select i1 %.not2, float %1273, float %1268, !dbg !7
  %1276 = select i1 %.not2, float %1269, float %1263, !dbg !7
  %1277 = bitcast float %1276 to i32, !dbg !7
  %1278 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1277), !dbg !7
  %1279 = bitcast i32 %1278 to float, !dbg !7
  %1280 = select i1 %.not2, float %1263, float %1279, !dbg !7
  %1281 = select i1 %.not2, float %1279, float %1269, !dbg !7
  %1282 = extractelement <8 x float> %1172, i64 4, !dbg !7
  %1283 = extractelement <8 x float> %1172, i64 5, !dbg !7
  %1284 = extractelement <8 x float> %1172, i64 6, !dbg !7
  %1285 = extractelement <8 x float> %1172, i64 7, !dbg !7
  %1286 = select i1 %.not, float %1283, float %1282, !dbg !7
  %1287 = bitcast float %1286 to i32, !dbg !7
  %1288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1287), !dbg !7
  %1289 = bitcast i32 %1288 to float, !dbg !7
  %1290 = select i1 %.not, float %1282, float %1289, !dbg !7
  %1291 = select i1 %.not, float %1289, float %1283, !dbg !7
  %1292 = select i1 %.not, float %1285, float %1284, !dbg !7
  %1293 = bitcast float %1292 to i32, !dbg !7
  %1294 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1293), !dbg !7
  %1295 = bitcast i32 %1294 to float, !dbg !7
  %1296 = select i1 %.not, float %1284, float %1295, !dbg !7
  %1297 = select i1 %.not, float %1295, float %1285, !dbg !7
  %1298 = select i1 %.not2, float %1296, float %1290, !dbg !7
  %1299 = bitcast float %1298 to i32, !dbg !7
  %1300 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1299), !dbg !7
  %1301 = bitcast i32 %1300 to float, !dbg !7
  %1302 = select i1 %.not2, float %1290, float %1301, !dbg !7
  %1303 = select i1 %.not2, float %1301, float %1296, !dbg !7
  %1304 = select i1 %.not2, float %1297, float %1291, !dbg !7
  %1305 = bitcast float %1304 to i32, !dbg !7
  %1306 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1305), !dbg !7
  %1307 = bitcast i32 %1306 to float, !dbg !7
  %1308 = select i1 %.not2, float %1291, float %1307, !dbg !7
  %1309 = select i1 %.not2, float %1307, float %1297, !dbg !7
  %1310 = extractelement <8 x float> %1201, i64 4, !dbg !7
  %1311 = extractelement <8 x float> %1201, i64 5, !dbg !7
  %1312 = extractelement <8 x float> %1201, i64 6, !dbg !7
  %1313 = extractelement <8 x float> %1201, i64 7, !dbg !7
  %1314 = select i1 %.not, float %1311, float %1310, !dbg !7
  %1315 = bitcast float %1314 to i32, !dbg !7
  %1316 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1315), !dbg !7
  %1317 = bitcast i32 %1316 to float, !dbg !7
  %1318 = select i1 %.not, float %1310, float %1317, !dbg !7
  %1319 = select i1 %.not, float %1317, float %1311, !dbg !7
  %1320 = select i1 %.not, float %1313, float %1312, !dbg !7
  %1321 = bitcast float %1320 to i32, !dbg !7
  %1322 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1321), !dbg !7
  %1323 = bitcast i32 %1322 to float, !dbg !7
  %1324 = select i1 %.not, float %1312, float %1323, !dbg !7
  %1325 = select i1 %.not, float %1323, float %1313, !dbg !7
  %1326 = select i1 %.not2, float %1324, float %1318, !dbg !7
  %1327 = bitcast float %1326 to i32, !dbg !7
  %1328 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1327), !dbg !7
  %1329 = select i1 %.not2, float %1325, float %1319, !dbg !7
  %1330 = bitcast float %1329 to i32, !dbg !7
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1330), !dbg !7
  %1332 = extractvalue [2 x <8 x float>] %53, 1, !dbg !7
  %1333 = extractelement <8 x float> %1332, i64 0, !dbg !7
  %1334 = extractelement <8 x float> %1332, i64 1, !dbg !7
  %1335 = extractelement <8 x float> %1332, i64 2, !dbg !7
  %1336 = extractelement <8 x float> %1332, i64 3, !dbg !7
  %1337 = select i1 %.not, float %1334, float %1333, !dbg !7
  %1338 = bitcast float %1337 to i32, !dbg !7
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1338), !dbg !7
  %1340 = bitcast i32 %1339 to float, !dbg !7
  %1341 = select i1 %.not, float %1333, float %1340, !dbg !7
  %1342 = select i1 %.not, float %1340, float %1334, !dbg !7
  %1343 = select i1 %.not, float %1336, float %1335, !dbg !7
  %1344 = bitcast float %1343 to i32, !dbg !7
  %1345 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1344), !dbg !7
  %1346 = bitcast i32 %1345 to float, !dbg !7
  %1347 = select i1 %.not, float %1335, float %1346, !dbg !7
  %1348 = select i1 %.not, float %1346, float %1336, !dbg !7
  %1349 = select i1 %.not2, float %1347, float %1341, !dbg !7
  %1350 = bitcast float %1349 to i32, !dbg !7
  %1351 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1350), !dbg !7
  %1352 = bitcast i32 %1351 to float, !dbg !7
  %1353 = select i1 %.not2, float %1341, float %1352, !dbg !7
  %1354 = select i1 %.not2, float %1352, float %1347, !dbg !7
  %1355 = select i1 %.not2, float %1348, float %1342, !dbg !7
  %1356 = bitcast float %1355 to i32, !dbg !7
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1356), !dbg !7
  %1358 = bitcast i32 %1357 to float, !dbg !7
  %1359 = select i1 %.not2, float %1342, float %1358, !dbg !7
  %1360 = select i1 %.not2, float %1358, float %1348, !dbg !7
  %1361 = extractvalue [2 x <8 x float>] %52, 1, !dbg !7
  %1362 = extractelement <8 x float> %1361, i64 0, !dbg !7
  %1363 = extractelement <8 x float> %1361, i64 1, !dbg !7
  %1364 = extractelement <8 x float> %1361, i64 2, !dbg !7
  %1365 = extractelement <8 x float> %1361, i64 3, !dbg !7
  %1366 = select i1 %.not, float %1363, float %1362, !dbg !7
  %1367 = bitcast float %1366 to i32, !dbg !7
  %1368 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1367), !dbg !7
  %1369 = bitcast i32 %1368 to float, !dbg !7
  %1370 = select i1 %.not, float %1362, float %1369, !dbg !7
  %1371 = select i1 %.not, float %1369, float %1363, !dbg !7
  %1372 = select i1 %.not, float %1365, float %1364, !dbg !7
  %1373 = bitcast float %1372 to i32, !dbg !7
  %1374 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1373), !dbg !7
  %1375 = bitcast i32 %1374 to float, !dbg !7
  %1376 = select i1 %.not, float %1364, float %1375, !dbg !7
  %1377 = select i1 %.not, float %1375, float %1365, !dbg !7
  %1378 = select i1 %.not2, float %1376, float %1370, !dbg !7
  %1379 = bitcast float %1378 to i32, !dbg !7
  %1380 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1379), !dbg !7
  %1381 = bitcast i32 %1380 to float, !dbg !7
  %1382 = select i1 %.not2, float %1370, float %1381, !dbg !7
  %1383 = select i1 %.not2, float %1381, float %1376, !dbg !7
  %1384 = select i1 %.not2, float %1377, float %1371, !dbg !7
  %1385 = bitcast float %1384 to i32, !dbg !7
  %1386 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1385), !dbg !7
  %1387 = bitcast i32 %1386 to float, !dbg !7
  %1388 = select i1 %.not2, float %1371, float %1387, !dbg !7
  %1389 = select i1 %.not2, float %1387, float %1377, !dbg !7
  %1390 = extractvalue [2 x <8 x float>] %51, 1, !dbg !7
  %1391 = extractelement <8 x float> %1390, i64 0, !dbg !7
  %1392 = extractelement <8 x float> %1390, i64 1, !dbg !7
  %1393 = extractelement <8 x float> %1390, i64 2, !dbg !7
  %1394 = extractelement <8 x float> %1390, i64 3, !dbg !7
  %1395 = select i1 %.not, float %1392, float %1391, !dbg !7
  %1396 = bitcast float %1395 to i32, !dbg !7
  %1397 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1396), !dbg !7
  %1398 = bitcast i32 %1397 to float, !dbg !7
  %1399 = select i1 %.not, float %1391, float %1398, !dbg !7
  %1400 = select i1 %.not, float %1398, float %1392, !dbg !7
  %1401 = select i1 %.not, float %1394, float %1393, !dbg !7
  %1402 = bitcast float %1401 to i32, !dbg !7
  %1403 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1402), !dbg !7
  %1404 = bitcast i32 %1403 to float, !dbg !7
  %1405 = select i1 %.not, float %1393, float %1404, !dbg !7
  %1406 = select i1 %.not, float %1404, float %1394, !dbg !7
  %1407 = select i1 %.not2, float %1405, float %1399, !dbg !7
  %1408 = bitcast float %1407 to i32, !dbg !7
  %1409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1408), !dbg !7
  %1410 = bitcast i32 %1409 to float, !dbg !7
  %1411 = select i1 %.not2, float %1399, float %1410, !dbg !7
  %1412 = select i1 %.not2, float %1410, float %1405, !dbg !7
  %1413 = select i1 %.not2, float %1406, float %1400, !dbg !7
  %1414 = bitcast float %1413 to i32, !dbg !7
  %1415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1414), !dbg !7
  %1416 = bitcast i32 %1415 to float, !dbg !7
  %1417 = select i1 %.not2, float %1400, float %1416, !dbg !7
  %1418 = select i1 %.not2, float %1416, float %1406, !dbg !7
  %1419 = extractvalue [2 x <8 x float>] %48, 1, !dbg !7
  %1420 = extractelement <8 x float> %1419, i64 0, !dbg !7
  %1421 = extractelement <8 x float> %1419, i64 1, !dbg !7
  %1422 = extractelement <8 x float> %1419, i64 2, !dbg !7
  %1423 = extractelement <8 x float> %1419, i64 3, !dbg !7
  %1424 = select i1 %.not, float %1421, float %1420, !dbg !7
  %1425 = bitcast float %1424 to i32, !dbg !7
  %1426 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1425), !dbg !7
  %1427 = bitcast i32 %1426 to float, !dbg !7
  %1428 = select i1 %.not, float %1420, float %1427, !dbg !7
  %1429 = select i1 %.not, float %1427, float %1421, !dbg !7
  %1430 = select i1 %.not, float %1423, float %1422, !dbg !7
  %1431 = bitcast float %1430 to i32, !dbg !7
  %1432 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1431), !dbg !7
  %1433 = bitcast i32 %1432 to float, !dbg !7
  %1434 = select i1 %.not, float %1422, float %1433, !dbg !7
  %1435 = select i1 %.not, float %1433, float %1423, !dbg !7
  %1436 = select i1 %.not2, float %1434, float %1428, !dbg !7
  %1437 = bitcast float %1436 to i32, !dbg !7
  %1438 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1437), !dbg !7
  %1439 = select i1 %.not2, float %1435, float %1429, !dbg !7
  %1440 = bitcast float %1439 to i32, !dbg !7
  %1441 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1440), !dbg !7
  %1442 = extractvalue [2 x <1 x float>] %49, 1, !dbg !19
  %1443 = shufflevector <1 x float> %1442, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1444 = extractelement <8 x float> %1332, i64 4, !dbg !7
  %1445 = extractelement <8 x float> %1332, i64 5, !dbg !7
  %1446 = extractelement <8 x float> %1332, i64 6, !dbg !7
  %1447 = extractelement <8 x float> %1332, i64 7, !dbg !7
  %1448 = select i1 %.not, float %1445, float %1444, !dbg !7
  %1449 = bitcast float %1448 to i32, !dbg !7
  %1450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1449), !dbg !7
  %1451 = bitcast i32 %1450 to float, !dbg !7
  %1452 = select i1 %.not, float %1444, float %1451, !dbg !7
  %1453 = select i1 %.not, float %1451, float %1445, !dbg !7
  %1454 = select i1 %.not, float %1447, float %1446, !dbg !7
  %1455 = bitcast float %1454 to i32, !dbg !7
  %1456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1455), !dbg !7
  %1457 = bitcast i32 %1456 to float, !dbg !7
  %1458 = select i1 %.not, float %1446, float %1457, !dbg !7
  %1459 = select i1 %.not, float %1457, float %1447, !dbg !7
  %1460 = select i1 %.not2, float %1458, float %1452, !dbg !7
  %1461 = bitcast float %1460 to i32, !dbg !7
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1461), !dbg !7
  %1463 = bitcast i32 %1462 to float, !dbg !7
  %1464 = select i1 %.not2, float %1452, float %1463, !dbg !7
  %1465 = select i1 %.not2, float %1463, float %1458, !dbg !7
  %1466 = select i1 %.not2, float %1459, float %1453, !dbg !7
  %1467 = bitcast float %1466 to i32, !dbg !7
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1467), !dbg !7
  %1469 = bitcast i32 %1468 to float, !dbg !7
  %1470 = select i1 %.not2, float %1453, float %1469, !dbg !7
  %1471 = select i1 %.not2, float %1469, float %1459, !dbg !7
  %1472 = extractelement <8 x float> %1361, i64 4, !dbg !7
  %1473 = extractelement <8 x float> %1361, i64 5, !dbg !7
  %1474 = extractelement <8 x float> %1361, i64 6, !dbg !7
  %1475 = extractelement <8 x float> %1361, i64 7, !dbg !7
  %1476 = select i1 %.not, float %1473, float %1472, !dbg !7
  %1477 = bitcast float %1476 to i32, !dbg !7
  %1478 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1477), !dbg !7
  %1479 = bitcast i32 %1478 to float, !dbg !7
  %1480 = select i1 %.not, float %1472, float %1479, !dbg !7
  %1481 = select i1 %.not, float %1479, float %1473, !dbg !7
  %1482 = select i1 %.not, float %1475, float %1474, !dbg !7
  %1483 = bitcast float %1482 to i32, !dbg !7
  %1484 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1483), !dbg !7
  %1485 = bitcast i32 %1484 to float, !dbg !7
  %1486 = select i1 %.not, float %1474, float %1485, !dbg !7
  %1487 = select i1 %.not, float %1485, float %1475, !dbg !7
  %1488 = select i1 %.not2, float %1486, float %1480, !dbg !7
  %1489 = bitcast float %1488 to i32, !dbg !7
  %1490 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1489), !dbg !7
  %1491 = bitcast i32 %1490 to float, !dbg !7
  %1492 = select i1 %.not2, float %1480, float %1491, !dbg !7
  %1493 = select i1 %.not2, float %1491, float %1486, !dbg !7
  %1494 = select i1 %.not2, float %1487, float %1481, !dbg !7
  %1495 = bitcast float %1494 to i32, !dbg !7
  %1496 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1495), !dbg !7
  %1497 = bitcast i32 %1496 to float, !dbg !7
  %1498 = select i1 %.not2, float %1481, float %1497, !dbg !7
  %1499 = select i1 %.not2, float %1497, float %1487, !dbg !7
  %1500 = extractelement <8 x float> %1390, i64 4, !dbg !7
  %1501 = extractelement <8 x float> %1390, i64 5, !dbg !7
  %1502 = extractelement <8 x float> %1390, i64 6, !dbg !7
  %1503 = extractelement <8 x float> %1390, i64 7, !dbg !7
  %1504 = select i1 %.not, float %1501, float %1500, !dbg !7
  %1505 = bitcast float %1504 to i32, !dbg !7
  %1506 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1505), !dbg !7
  %1507 = bitcast i32 %1506 to float, !dbg !7
  %1508 = select i1 %.not, float %1500, float %1507, !dbg !7
  %1509 = select i1 %.not, float %1507, float %1501, !dbg !7
  %1510 = select i1 %.not, float %1503, float %1502, !dbg !7
  %1511 = bitcast float %1510 to i32, !dbg !7
  %1512 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1511), !dbg !7
  %1513 = bitcast i32 %1512 to float, !dbg !7
  %1514 = select i1 %.not, float %1502, float %1513, !dbg !7
  %1515 = select i1 %.not, float %1513, float %1503, !dbg !7
  %1516 = select i1 %.not2, float %1514, float %1508, !dbg !7
  %1517 = bitcast float %1516 to i32, !dbg !7
  %1518 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1517), !dbg !7
  %1519 = bitcast i32 %1518 to float, !dbg !7
  %1520 = select i1 %.not2, float %1508, float %1519, !dbg !7
  %1521 = select i1 %.not2, float %1519, float %1514, !dbg !7
  %1522 = select i1 %.not2, float %1515, float %1509, !dbg !7
  %1523 = bitcast float %1522 to i32, !dbg !7
  %1524 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1523), !dbg !7
  %1525 = bitcast i32 %1524 to float, !dbg !7
  %1526 = select i1 %.not2, float %1509, float %1525, !dbg !7
  %1527 = select i1 %.not2, float %1525, float %1515, !dbg !7
  %1528 = extractelement <8 x float> %1419, i64 4, !dbg !7
  %1529 = extractelement <8 x float> %1419, i64 5, !dbg !7
  %1530 = extractelement <8 x float> %1419, i64 6, !dbg !7
  %1531 = extractelement <8 x float> %1419, i64 7, !dbg !7
  %1532 = select i1 %.not, float %1529, float %1528, !dbg !7
  %1533 = bitcast float %1532 to i32, !dbg !7
  %1534 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1533), !dbg !7
  %1535 = bitcast i32 %1534 to float, !dbg !7
  %1536 = select i1 %.not, float %1528, float %1535, !dbg !7
  %1537 = select i1 %.not, float %1535, float %1529, !dbg !7
  %1538 = select i1 %.not, float %1531, float %1530, !dbg !7
  %1539 = bitcast float %1538 to i32, !dbg !7
  %1540 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1115, i32 %1539), !dbg !7
  %1541 = bitcast i32 %1540 to float, !dbg !7
  %1542 = select i1 %.not, float %1530, float %1541, !dbg !7
  %1543 = select i1 %.not, float %1541, float %1531, !dbg !7
  %1544 = select i1 %.not2, float %1542, float %1536, !dbg !7
  %1545 = bitcast float %1544 to i32, !dbg !7
  %1546 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1545), !dbg !7
  %1547 = select i1 %.not2, float %1543, float %1537, !dbg !7
  %1548 = bitcast float %1547 to i32, !dbg !7
  %1549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1548), !dbg !7
  br i1 true, label %1550, label %1614, !dbg !21

1550:                                             ; preds = %1101
  %1551 = bitcast i32 %1546 to float, !dbg !7
  %1552 = select i1 %.not2, float %1536, float %1551, !dbg !7
  %1553 = insertelement <4 x float> poison, float %1552, i64 0, !dbg !19
  %1554 = bitcast i32 %1549 to float, !dbg !7
  %1555 = select i1 %.not2, float %1537, float %1554, !dbg !7
  %1556 = insertelement <4 x float> %1553, float %1555, i64 1, !dbg !19
  %1557 = select i1 %.not2, float %1551, float %1542, !dbg !7
  %1558 = insertelement <4 x float> %1556, float %1557, i64 2, !dbg !19
  %1559 = select i1 %.not2, float %1554, float %1543, !dbg !7
  %1560 = insertelement <4 x float> %1558, float %1559, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1443, !dbg !19
  %normalized = fmul <4 x float> %1560, %row.reciprocal, !dbg !19
  %1561 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1562 = extractelement <4 x half> %1561, i64 3, !dbg !20
  %1563 = extractelement <4 x half> %1561, i64 2, !dbg !20
  %1564 = extractelement <4 x half> %1561, i64 1, !dbg !20
  %1565 = extractelement <4 x half> %1561, i64 0, !dbg !20
  %1566 = bitcast i32 %1438 to float, !dbg !7
  %1567 = select i1 %.not2, float %1428, float %1566, !dbg !7
  %1568 = insertelement <4 x float> poison, float %1567, i64 0, !dbg !19
  %1569 = bitcast i32 %1441 to float, !dbg !7
  %1570 = select i1 %.not2, float %1429, float %1569, !dbg !7
  %1571 = insertelement <4 x float> %1568, float %1570, i64 1, !dbg !19
  %1572 = select i1 %.not2, float %1566, float %1434, !dbg !7
  %1573 = insertelement <4 x float> %1571, float %1572, i64 2, !dbg !19
  %1574 = select i1 %.not2, float %1569, float %1435, !dbg !7
  %1575 = insertelement <4 x float> %1573, float %1574, i64 3, !dbg !19
  %normalized22 = fmul <4 x float> %1575, %row.reciprocal, !dbg !19
  %1576 = fptrunc <4 x float> %normalized22 to <4 x half>, !dbg !20
  %1577 = extractelement <4 x half> %1576, i64 3, !dbg !20
  %1578 = extractelement <4 x half> %1576, i64 2, !dbg !20
  %1579 = extractelement <4 x half> %1576, i64 1, !dbg !20
  %1580 = extractelement <4 x half> %1576, i64 0, !dbg !20
  %1581 = bitcast i32 %1328 to float, !dbg !7
  %1582 = select i1 %.not2, float %1318, float %1581, !dbg !7
  %1583 = insertelement <4 x float> poison, float %1582, i64 0, !dbg !19
  %1584 = bitcast i32 %1331 to float, !dbg !7
  %1585 = select i1 %.not2, float %1319, float %1584, !dbg !7
  %1586 = insertelement <4 x float> %1583, float %1585, i64 1, !dbg !19
  %1587 = select i1 %.not2, float %1581, float %1324, !dbg !7
  %1588 = insertelement <4 x float> %1586, float %1587, i64 2, !dbg !19
  %1589 = select i1 %.not2, float %1584, float %1325, !dbg !7
  %1590 = insertelement <4 x float> %1588, float %1589, i64 3, !dbg !19
  %row.reciprocal23 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1225, !dbg !19
  %normalized24 = fmul <4 x float> %1590, %row.reciprocal23, !dbg !19
  %1591 = fptrunc <4 x float> %normalized24 to <4 x half>, !dbg !20
  %1592 = extractelement <4 x half> %1591, i64 3, !dbg !20
  %1593 = extractelement <4 x half> %1591, i64 2, !dbg !20
  %1594 = extractelement <4 x half> %1591, i64 1, !dbg !20
  %1595 = extractelement <4 x half> %1591, i64 0, !dbg !20
  %1596 = bitcast i32 %1220 to float, !dbg !7
  %1597 = select i1 %.not2, float %1210, float %1596, !dbg !7
  %1598 = insertelement <4 x float> poison, float %1597, i64 0, !dbg !19
  %1599 = bitcast i32 %1223 to float, !dbg !7
  %1600 = select i1 %.not2, float %1211, float %1599, !dbg !7
  %1601 = insertelement <4 x float> %1598, float %1600, i64 1, !dbg !19
  %1602 = select i1 %.not2, float %1596, float %1216, !dbg !7
  %1603 = insertelement <4 x float> %1601, float %1602, i64 2, !dbg !19
  %1604 = select i1 %.not2, float %1599, float %1217, !dbg !7
  %1605 = insertelement <4 x float> %1603, float %1604, i64 3, !dbg !19
  %normalized25 = fmul <4 x float> %1605, %row.reciprocal23, !dbg !19
  %1606 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !20
  %1607 = extractelement <4 x half> %1606, i64 3, !dbg !20
  %1608 = extractelement <4 x half> %1606, i64 2, !dbg !20
  %1609 = extractelement <4 x half> %1606, i64 1, !dbg !20
  %1610 = extractelement <4 x half> %1606, i64 0, !dbg !20
  %1611 = add nsw i64 %8, %15, !dbg !21
  %1612 = shl nsw i64 %1611, 19, !dbg !21
  %1613 = and i64 %1612, 16252928, !dbg !21
  %address.thread334 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local335 = zext nneg i32 %address.thread334 to i64, !dbg !12
  %address.local336 = add nsw i64 %18, %address.local335, !dbg !21
  %address.local337 = lshr i64 %address.local335, 6, !dbg !12
  %address.local338 = shl nuw nsw i64 %address.local337, 4, !dbg !21
  %address.local339 = add nsw i64 %address.local336, %address.local338, !dbg !21
  %address.local340 = and i64 %address.local335, 240, !dbg !21
  %address.local341 = sub nsw i64 %address.local339, %address.local340, !dbg !21
  %address.local342 = add nsw i64 %address.local341, %19, !dbg !21
  %address.local343 = shl nuw nsw i64 %address.local342, 7, !dbg !21
  %address.local344 = add nuw nsw i64 %1613, %address.local343, !dbg !21
  %address.local345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local344, !dbg !21
  %address.local346 = lshr i64 %address.local335, 2, !dbg !12
  %address.local347 = and i64 %address.local346, 12, !dbg !12
  %address.local348 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local345, i64 %address.local347, !dbg !21
  store half %1610, ptr addrspace(1) %address.local348, align 2, !dbg !21
  %address.local349 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local344, !dbg !21
  %address.local350 = or disjoint i64 %address.local347, 1, !dbg !21
  %address.local351 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local349, i64 %address.local350, !dbg !21
  store half %1609, ptr addrspace(1) %address.local351, align 2, !dbg !21
  %address.local352 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local344, !dbg !21
  %address.local353 = or disjoint i64 %address.local347, 2, !dbg !21
  %address.local354 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local352, i64 %address.local353, !dbg !21
  store half %1608, ptr addrspace(1) %address.local354, align 2, !dbg !21
  %address.local355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local344, !dbg !21
  %address.local356 = or disjoint i64 %address.local347, 3, !dbg !21
  %address.local357 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local355, i64 %address.local356, !dbg !21
  store half %1607, ptr addrspace(1) %address.local357, align 2, !dbg !21
  %address.local358 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local344, !dbg !21
  %address.local359 = or disjoint i64 %address.local347, 16, !dbg !21
  %address.local360 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local358, i64 %address.local359, !dbg !21
  store half %1595, ptr addrspace(1) %address.local360, align 2, !dbg !21
  %address.local361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local344, !dbg !21
  %address.local362 = or disjoint i64 %address.local347, 17, !dbg !21
  %address.local363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local361, i64 %address.local362, !dbg !21
  store half %1594, ptr addrspace(1) %address.local363, align 2, !dbg !21
  %address.local364 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local344, !dbg !21
  %address.local365 = or disjoint i64 %address.local347, 18, !dbg !21
  %address.local366 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local364, i64 %address.local365, !dbg !21
  store half %1593, ptr addrspace(1) %address.local366, align 2, !dbg !21
  %address.local367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local344, !dbg !21
  %address.local368 = or disjoint i64 %address.local347, 19, !dbg !21
  %address.local369 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local367, i64 %address.local368, !dbg !21
  store half %1592, ptr addrspace(1) %address.local369, align 2, !dbg !21
  %address.local370 = shl nsw i64 %address.local342, 7, !dbg !21
  %address.local371 = add i64 %address.local370, 8192, !dbg !21
  %address.local372 = add nuw nsw i64 %1613, %address.local371, !dbg !21
  %address.local373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local372, !dbg !21
  %address.local374 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local373, i64 %address.local347, !dbg !21
  store half %1580, ptr addrspace(1) %address.local374, align 2, !dbg !21
  %address.local375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local372, !dbg !21
  %address.local376 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local375, i64 %address.local350, !dbg !21
  store half %1579, ptr addrspace(1) %address.local376, align 2, !dbg !21
  %address.local377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local372, !dbg !21
  %address.local378 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local377, i64 %address.local353, !dbg !21
  store half %1578, ptr addrspace(1) %address.local378, align 2, !dbg !21
  %address.local379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local372, !dbg !21
  %address.local380 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local379, i64 %address.local356, !dbg !21
  store half %1577, ptr addrspace(1) %address.local380, align 2, !dbg !21
  %address.local381 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local372, !dbg !21
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local381, i64 %address.local359, !dbg !21
  store half %1565, ptr addrspace(1) %address.local382, align 2, !dbg !21
  %address.local383 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local372, !dbg !21
  %address.local384 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local383, i64 %address.local362, !dbg !21
  store half %1564, ptr addrspace(1) %address.local384, align 2, !dbg !21
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local372, !dbg !21
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local385, i64 %address.local365, !dbg !21
  store half %1563, ptr addrspace(1) %address.local386, align 2, !dbg !21
  %address.local387 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local372, !dbg !21
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local387, i64 %address.local368, !dbg !21
  store half %1562, ptr addrspace(1) %address.local388, align 2, !dbg !21
  br label %1614, !dbg !21

1614:                                             ; preds = %1550, %1101
  br i1 true, label %1615, label %1653, !dbg !21

1615:                                             ; preds = %1614
  %1616 = insertelement <4 x float> poison, float %1520, i64 0, !dbg !19
  %1617 = insertelement <4 x float> %1616, float %1526, i64 1, !dbg !19
  %1618 = insertelement <4 x float> %1617, float %1521, i64 2, !dbg !19
  %1619 = insertelement <4 x float> %1618, float %1527, i64 3, !dbg !19
  %row.reciprocal26 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1443, !dbg !19
  %normalized27 = fmul <4 x float> %1619, %row.reciprocal26, !dbg !19
  %1620 = fptrunc <4 x float> %normalized27 to <4 x half>, !dbg !20
  %1621 = extractelement <4 x half> %1620, i64 3, !dbg !20
  %1622 = extractelement <4 x half> %1620, i64 2, !dbg !20
  %1623 = extractelement <4 x half> %1620, i64 1, !dbg !20
  %1624 = extractelement <4 x half> %1620, i64 0, !dbg !20
  %1625 = insertelement <4 x float> poison, float %1411, i64 0, !dbg !19
  %1626 = insertelement <4 x float> %1625, float %1417, i64 1, !dbg !19
  %1627 = insertelement <4 x float> %1626, float %1412, i64 2, !dbg !19
  %1628 = insertelement <4 x float> %1627, float %1418, i64 3, !dbg !19
  %normalized28 = fmul <4 x float> %1628, %row.reciprocal26, !dbg !19
  %1629 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !20
  %1630 = extractelement <4 x half> %1629, i64 3, !dbg !20
  %1631 = extractelement <4 x half> %1629, i64 2, !dbg !20
  %1632 = extractelement <4 x half> %1629, i64 1, !dbg !20
  %1633 = extractelement <4 x half> %1629, i64 0, !dbg !20
  %1634 = insertelement <4 x float> poison, float %1302, i64 0, !dbg !19
  %1635 = insertelement <4 x float> %1634, float %1308, i64 1, !dbg !19
  %1636 = insertelement <4 x float> %1635, float %1303, i64 2, !dbg !19
  %1637 = insertelement <4 x float> %1636, float %1309, i64 3, !dbg !19
  %row.reciprocal29 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1225, !dbg !19
  %normalized30 = fmul <4 x float> %1637, %row.reciprocal29, !dbg !19
  %1638 = fptrunc <4 x float> %normalized30 to <4 x half>, !dbg !20
  %1639 = extractelement <4 x half> %1638, i64 3, !dbg !20
  %1640 = extractelement <4 x half> %1638, i64 2, !dbg !20
  %1641 = extractelement <4 x half> %1638, i64 1, !dbg !20
  %1642 = extractelement <4 x half> %1638, i64 0, !dbg !20
  %1643 = insertelement <4 x float> poison, float %1193, i64 0, !dbg !19
  %1644 = insertelement <4 x float> %1643, float %1199, i64 1, !dbg !19
  %1645 = insertelement <4 x float> %1644, float %1194, i64 2, !dbg !19
  %1646 = insertelement <4 x float> %1645, float %1200, i64 3, !dbg !19
  %normalized31 = fmul <4 x float> %1646, %row.reciprocal29, !dbg !19
  %1647 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !20
  %1648 = extractelement <4 x half> %1647, i64 3, !dbg !20
  %1649 = extractelement <4 x half> %1647, i64 2, !dbg !20
  %1650 = extractelement <4 x half> %1647, i64 1, !dbg !20
  %1651 = extractelement <4 x half> %1647, i64 0, !dbg !20
  %1652 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread389 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local390 = zext nneg i32 %address.thread389 to i64, !dbg !12
  %address.local391 = add nsw i64 %18, %address.local390, !dbg !21
  %address.local392 = lshr i64 %address.local390, 6, !dbg !12
  %address.local393 = shl nuw nsw i64 %address.local392, 4, !dbg !21
  %address.local394 = add nsw i64 %address.local391, %address.local393, !dbg !21
  %address.local395 = and i64 %address.local390, 240, !dbg !21
  %address.local396 = sub nsw i64 %address.local394, %address.local395, !dbg !21
  %address.local397 = add nsw i64 %address.local396, %19, !dbg !21
  %address.local398 = shl nuw nsw i64 %address.local397, 7, !dbg !21
  %address.local399 = add nuw nsw i64 %1652, %address.local398, !dbg !21
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local401 = lshr i64 %address.local390, 2, !dbg !12
  %address.local402 = and i64 %address.local401, 12, !dbg !12
  %address.local403 = or disjoint i64 %address.local402, 32, !dbg !21
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local403, !dbg !21
  store half %1651, ptr addrspace(1) %address.local404, align 2, !dbg !21
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local406 = or disjoint i64 %address.local402, 33, !dbg !21
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local406, !dbg !21
  store half %1650, ptr addrspace(1) %address.local407, align 2, !dbg !21
  %address.local408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local409 = or disjoint i64 %address.local402, 34, !dbg !21
  %address.local410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local408, i64 %address.local409, !dbg !21
  store half %1649, ptr addrspace(1) %address.local410, align 2, !dbg !21
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local412 = or disjoint i64 %address.local402, 35, !dbg !21
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local411, i64 %address.local412, !dbg !21
  store half %1648, ptr addrspace(1) %address.local413, align 2, !dbg !21
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local415 = or disjoint i64 %address.local402, 48, !dbg !21
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local414, i64 %address.local415, !dbg !21
  store half %1642, ptr addrspace(1) %address.local416, align 2, !dbg !21
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local418 = or disjoint i64 %address.local402, 49, !dbg !21
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local417, i64 %address.local418, !dbg !21
  store half %1641, ptr addrspace(1) %address.local419, align 2, !dbg !21
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local421 = or disjoint i64 %address.local402, 50, !dbg !21
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local420, i64 %address.local421, !dbg !21
  store half %1640, ptr addrspace(1) %address.local422, align 2, !dbg !21
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local424 = or i64 %address.local401, 51, !dbg !21
  %address.local425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local423, i64 %address.local424, !dbg !21
  store half %1639, ptr addrspace(1) %address.local425, align 2, !dbg !21
  %address.local426 = shl nsw i64 %address.local397, 7, !dbg !21
  %address.local427 = add i64 %address.local426, 8192, !dbg !21
  %address.local428 = add nuw nsw i64 %1652, %address.local427, !dbg !21
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local429, i64 %address.local403, !dbg !21
  store half %1633, ptr addrspace(1) %address.local430, align 2, !dbg !21
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local431, i64 %address.local406, !dbg !21
  store half %1632, ptr addrspace(1) %address.local432, align 2, !dbg !21
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local409, !dbg !21
  store half %1631, ptr addrspace(1) %address.local434, align 2, !dbg !21
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local412, !dbg !21
  store half %1630, ptr addrspace(1) %address.local436, align 2, !dbg !21
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local437, i64 %address.local415, !dbg !21
  store half %1624, ptr addrspace(1) %address.local438, align 2, !dbg !21
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local418, !dbg !21
  store half %1623, ptr addrspace(1) %address.local440, align 2, !dbg !21
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local441, i64 %address.local421, !dbg !21
  store half %1622, ptr addrspace(1) %address.local442, align 2, !dbg !21
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local428, !dbg !21
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local443, i64 %address.local424, !dbg !21
  store half %1621, ptr addrspace(1) %address.local444, align 2, !dbg !21
  br label %1653, !dbg !21

1653:                                             ; preds = %1615, %1614
  br i1 true, label %1654, label %1692, !dbg !21

1654:                                             ; preds = %1653
  %1655 = insertelement <4 x float> poison, float %1492, i64 0, !dbg !19
  %1656 = insertelement <4 x float> %1655, float %1498, i64 1, !dbg !19
  %1657 = insertelement <4 x float> %1656, float %1493, i64 2, !dbg !19
  %1658 = insertelement <4 x float> %1657, float %1499, i64 3, !dbg !19
  %row.reciprocal32 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1443, !dbg !19
  %normalized33 = fmul <4 x float> %1658, %row.reciprocal32, !dbg !19
  %1659 = fptrunc <4 x float> %normalized33 to <4 x half>, !dbg !20
  %1660 = extractelement <4 x half> %1659, i64 3, !dbg !20
  %1661 = extractelement <4 x half> %1659, i64 2, !dbg !20
  %1662 = extractelement <4 x half> %1659, i64 1, !dbg !20
  %1663 = extractelement <4 x half> %1659, i64 0, !dbg !20
  %1664 = insertelement <4 x float> poison, float %1382, i64 0, !dbg !19
  %1665 = insertelement <4 x float> %1664, float %1388, i64 1, !dbg !19
  %1666 = insertelement <4 x float> %1665, float %1383, i64 2, !dbg !19
  %1667 = insertelement <4 x float> %1666, float %1389, i64 3, !dbg !19
  %normalized34 = fmul <4 x float> %1667, %row.reciprocal32, !dbg !19
  %1668 = fptrunc <4 x float> %normalized34 to <4 x half>, !dbg !20
  %1669 = extractelement <4 x half> %1668, i64 3, !dbg !20
  %1670 = extractelement <4 x half> %1668, i64 2, !dbg !20
  %1671 = extractelement <4 x half> %1668, i64 1, !dbg !20
  %1672 = extractelement <4 x half> %1668, i64 0, !dbg !20
  %1673 = insertelement <4 x float> poison, float %1274, i64 0, !dbg !19
  %1674 = insertelement <4 x float> %1673, float %1280, i64 1, !dbg !19
  %1675 = insertelement <4 x float> %1674, float %1275, i64 2, !dbg !19
  %1676 = insertelement <4 x float> %1675, float %1281, i64 3, !dbg !19
  %row.reciprocal35 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1225, !dbg !19
  %normalized36 = fmul <4 x float> %1676, %row.reciprocal35, !dbg !19
  %1677 = fptrunc <4 x float> %normalized36 to <4 x half>, !dbg !20
  %1678 = extractelement <4 x half> %1677, i64 3, !dbg !20
  %1679 = extractelement <4 x half> %1677, i64 2, !dbg !20
  %1680 = extractelement <4 x half> %1677, i64 1, !dbg !20
  %1681 = extractelement <4 x half> %1677, i64 0, !dbg !20
  %1682 = insertelement <4 x float> poison, float %1164, i64 0, !dbg !19
  %1683 = insertelement <4 x float> %1682, float %1170, i64 1, !dbg !19
  %1684 = insertelement <4 x float> %1683, float %1165, i64 2, !dbg !19
  %1685 = insertelement <4 x float> %1684, float %1171, i64 3, !dbg !19
  %normalized37 = fmul <4 x float> %1685, %row.reciprocal35, !dbg !19
  %1686 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !20
  %1687 = extractelement <4 x half> %1686, i64 3, !dbg !20
  %1688 = extractelement <4 x half> %1686, i64 2, !dbg !20
  %1689 = extractelement <4 x half> %1686, i64 1, !dbg !20
  %1690 = extractelement <4 x half> %1686, i64 0, !dbg !20
  %1691 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread445 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local446 = zext nneg i32 %address.thread445 to i64, !dbg !12
  %address.local447 = add nsw i64 %18, %address.local446, !dbg !21
  %address.local448 = lshr i64 %address.local446, 6, !dbg !12
  %address.local449 = shl nuw nsw i64 %address.local448, 4, !dbg !21
  %address.local450 = add nsw i64 %address.local447, %address.local449, !dbg !21
  %address.local451 = and i64 %address.local446, 240, !dbg !21
  %address.local452 = sub nsw i64 %address.local450, %address.local451, !dbg !21
  %address.local453 = add nsw i64 %address.local452, %19, !dbg !21
  %address.local454 = shl nuw nsw i64 %address.local453, 7, !dbg !21
  %address.local455 = add nuw nsw i64 %1691, %address.local454, !dbg !21
  %address.local456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !21
  %address.local457 = lshr i64 %address.local446, 2, !dbg !12
  %address.local458 = and i64 %address.local457, 12, !dbg !12
  %address.local459 = or disjoint i64 %address.local458, 64, !dbg !21
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local456, i64 %address.local459, !dbg !21
  store half %1690, ptr addrspace(1) %address.local460, align 2, !dbg !21
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !21
  %address.local462 = or disjoint i64 %address.local458, 65, !dbg !21
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local461, i64 %address.local462, !dbg !21
  store half %1689, ptr addrspace(1) %address.local463, align 2, !dbg !21
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !21
  %address.local465 = or disjoint i64 %address.local458, 66, !dbg !21
  %address.local466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local464, i64 %address.local465, !dbg !21
  store half %1688, ptr addrspace(1) %address.local466, align 2, !dbg !21
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !21
  %address.local468 = or disjoint i64 %address.local458, 67, !dbg !21
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local467, i64 %address.local468, !dbg !21
  store half %1687, ptr addrspace(1) %address.local469, align 2, !dbg !21
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !21
  %address.local471 = or disjoint i64 %address.local458, 80, !dbg !21
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local470, i64 %address.local471, !dbg !21
  store half %1681, ptr addrspace(1) %address.local472, align 2, !dbg !21
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !21
  %address.local474 = or disjoint i64 %address.local458, 81, !dbg !21
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local473, i64 %address.local474, !dbg !21
  store half %1680, ptr addrspace(1) %address.local475, align 2, !dbg !21
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !21
  %address.local477 = or disjoint i64 %address.local458, 82, !dbg !21
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local476, i64 %address.local477, !dbg !21
  store half %1679, ptr addrspace(1) %address.local478, align 2, !dbg !21
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !21
  %address.local480 = or disjoint i64 %address.local458, 83, !dbg !21
  %address.local481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local479, i64 %address.local480, !dbg !21
  store half %1678, ptr addrspace(1) %address.local481, align 2, !dbg !21
  %address.local482 = shl nsw i64 %address.local453, 7, !dbg !21
  %address.local483 = add i64 %address.local482, 8192, !dbg !21
  %address.local484 = add nuw nsw i64 %1691, %address.local483, !dbg !21
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local485, i64 %address.local459, !dbg !21
  store half %1672, ptr addrspace(1) %address.local486, align 2, !dbg !21
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local487, i64 %address.local462, !dbg !21
  store half %1671, ptr addrspace(1) %address.local488, align 2, !dbg !21
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local465, !dbg !21
  store half %1670, ptr addrspace(1) %address.local490, align 2, !dbg !21
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local491, i64 %address.local468, !dbg !21
  store half %1669, ptr addrspace(1) %address.local492, align 2, !dbg !21
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local493, i64 %address.local471, !dbg !21
  store half %1663, ptr addrspace(1) %address.local494, align 2, !dbg !21
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local474, !dbg !21
  store half %1662, ptr addrspace(1) %address.local496, align 2, !dbg !21
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local497, i64 %address.local477, !dbg !21
  store half %1661, ptr addrspace(1) %address.local498, align 2, !dbg !21
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local499, i64 %address.local480, !dbg !21
  store half %1660, ptr addrspace(1) %address.local500, align 2, !dbg !21
  br label %1692, !dbg !21

1692:                                             ; preds = %1654, %1653
  br i1 true, label %1693, label %1731, !dbg !21

1693:                                             ; preds = %1692
  %1694 = insertelement <4 x float> poison, float %1464, i64 0, !dbg !19
  %1695 = insertelement <4 x float> %1694, float %1470, i64 1, !dbg !19
  %1696 = insertelement <4 x float> %1695, float %1465, i64 2, !dbg !19
  %1697 = insertelement <4 x float> %1696, float %1471, i64 3, !dbg !19
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1443, !dbg !19
  %normalized39 = fmul <4 x float> %1697, %row.reciprocal38, !dbg !19
  %1698 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !20
  %1699 = extractelement <4 x half> %1698, i64 3, !dbg !20
  %1700 = extractelement <4 x half> %1698, i64 2, !dbg !20
  %1701 = extractelement <4 x half> %1698, i64 1, !dbg !20
  %1702 = extractelement <4 x half> %1698, i64 0, !dbg !20
  %1703 = insertelement <4 x float> poison, float %1353, i64 0, !dbg !19
  %1704 = insertelement <4 x float> %1703, float %1359, i64 1, !dbg !19
  %1705 = insertelement <4 x float> %1704, float %1354, i64 2, !dbg !19
  %1706 = insertelement <4 x float> %1705, float %1360, i64 3, !dbg !19
  %normalized40 = fmul <4 x float> %1706, %row.reciprocal38, !dbg !19
  %1707 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !20
  %1708 = extractelement <4 x half> %1707, i64 3, !dbg !20
  %1709 = extractelement <4 x half> %1707, i64 2, !dbg !20
  %1710 = extractelement <4 x half> %1707, i64 1, !dbg !20
  %1711 = extractelement <4 x half> %1707, i64 0, !dbg !20
  %1712 = insertelement <4 x float> poison, float %1246, i64 0, !dbg !19
  %1713 = insertelement <4 x float> %1712, float %1252, i64 1, !dbg !19
  %1714 = insertelement <4 x float> %1713, float %1247, i64 2, !dbg !19
  %1715 = insertelement <4 x float> %1714, float %1253, i64 3, !dbg !19
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1225, !dbg !19
  %normalized42 = fmul <4 x float> %1715, %row.reciprocal41, !dbg !19
  %1716 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !20
  %1717 = extractelement <4 x half> %1716, i64 3, !dbg !20
  %1718 = extractelement <4 x half> %1716, i64 2, !dbg !20
  %1719 = extractelement <4 x half> %1716, i64 1, !dbg !20
  %1720 = extractelement <4 x half> %1716, i64 0, !dbg !20
  %1721 = insertelement <4 x float> poison, float %1135, i64 0, !dbg !19
  %1722 = insertelement <4 x float> %1721, float %1141, i64 1, !dbg !19
  %1723 = insertelement <4 x float> %1722, float %1136, i64 2, !dbg !19
  %1724 = insertelement <4 x float> %1723, float %1142, i64 3, !dbg !19
  %normalized43 = fmul <4 x float> %1724, %row.reciprocal41, !dbg !19
  %1725 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !20
  %1726 = extractelement <4 x half> %1725, i64 3, !dbg !20
  %1727 = extractelement <4 x half> %1725, i64 2, !dbg !20
  %1728 = extractelement <4 x half> %1725, i64 1, !dbg !20
  %1729 = extractelement <4 x half> %1725, i64 0, !dbg !20
  %1730 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread501 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local502 = zext nneg i32 %address.thread501 to i64, !dbg !12
  %address.local503 = add nsw i64 %18, %address.local502, !dbg !21
  %address.local504 = lshr i64 %address.local502, 6, !dbg !12
  %address.local505 = shl nuw nsw i64 %address.local504, 4, !dbg !21
  %address.local506 = add nsw i64 %address.local503, %address.local505, !dbg !21
  %address.local507 = and i64 %address.local502, 240, !dbg !21
  %address.local508 = sub nsw i64 %address.local506, %address.local507, !dbg !21
  %address.local509 = add nsw i64 %address.local508, %19, !dbg !21
  %address.local510 = shl nuw nsw i64 %address.local509, 7, !dbg !21
  %address.local511 = add nuw nsw i64 %1730, %address.local510, !dbg !21
  %address.local512 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local511, !dbg !21
  %address.local513 = lshr i64 %address.local502, 2, !dbg !12
  %address.local514 = and i64 %address.local513, 12, !dbg !12
  %address.local515 = or disjoint i64 %address.local514, 96, !dbg !21
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local512, i64 %address.local515, !dbg !21
  store half %1729, ptr addrspace(1) %address.local516, align 2, !dbg !21
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local511, !dbg !21
  %address.local518 = or disjoint i64 %address.local514, 97, !dbg !21
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local517, i64 %address.local518, !dbg !21
  store half %1728, ptr addrspace(1) %address.local519, align 2, !dbg !21
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local511, !dbg !21
  %address.local521 = or disjoint i64 %address.local514, 98, !dbg !21
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local520, i64 %address.local521, !dbg !21
  store half %1727, ptr addrspace(1) %address.local522, align 2, !dbg !21
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local511, !dbg !21
  %address.local524 = or disjoint i64 %address.local514, 99, !dbg !21
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local523, i64 %address.local524, !dbg !21
  store half %1726, ptr addrspace(1) %address.local525, align 2, !dbg !21
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local511, !dbg !21
  %address.local527 = or disjoint i64 %address.local514, 112, !dbg !21
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local526, i64 %address.local527, !dbg !21
  store half %1720, ptr addrspace(1) %address.local528, align 2, !dbg !21
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local511, !dbg !21
  %address.local530 = or disjoint i64 %address.local514, 113, !dbg !21
  %address.local531 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local529, i64 %address.local530, !dbg !21
  store half %1719, ptr addrspace(1) %address.local531, align 2, !dbg !21
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local511, !dbg !21
  %address.local533 = or disjoint i64 %address.local514, 114, !dbg !21
  %address.local534 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local532, i64 %address.local533, !dbg !21
  store half %1718, ptr addrspace(1) %address.local534, align 2, !dbg !21
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local511, !dbg !21
  %address.local536 = or i64 %address.local513, 115, !dbg !21
  %address.local537 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local535, i64 %address.local536, !dbg !21
  store half %1717, ptr addrspace(1) %address.local537, align 2, !dbg !21
  %address.local538 = shl nsw i64 %address.local509, 7, !dbg !21
  %address.local539 = add i64 %address.local538, 8192, !dbg !21
  %address.local540 = add nuw nsw i64 %1730, %address.local539, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local540, !dbg !21
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local541, i64 %address.local515, !dbg !21
  store half %1711, ptr addrspace(1) %address.local542, align 2, !dbg !21
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local540, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local543, i64 %address.local518, !dbg !21
  store half %1710, ptr addrspace(1) %address.local544, align 2, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local540, !dbg !21
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local521, !dbg !21
  store half %1709, ptr addrspace(1) %address.local546, align 2, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local540, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local524, !dbg !21
  store half %1708, ptr addrspace(1) %address.local548, align 2, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local540, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local527, !dbg !21
  store half %1702, ptr addrspace(1) %address.local550, align 2, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local540, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local551, i64 %address.local530, !dbg !21
  store half %1701, ptr addrspace(1) %address.local552, align 2, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local540, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local553, i64 %address.local533, !dbg !21
  store half %1700, ptr addrspace(1) %address.local554, align 2, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local540, !dbg !21
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local536, !dbg !21
  store half %1699, ptr addrspace(1) %address.local556, align 2, !dbg !21
  br label %1731, !dbg !21

1731:                                             ; preds = %1693, %1692
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
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/baseline")
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
