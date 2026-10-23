; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles/bm64_bn64.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles/bm64_bn64.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !32 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %6, 6, !dbg !11
  %.lobit = ashr i32 %5, 31, !dbg !13
  %10 = xor i32 %5, %.lobit, !dbg !13
  %11 = sdiv i32 %10, 64, !dbg !13
  %.lobit1 = ashr i32 %5, 31, !dbg !13
  %12 = xor i32 %11, %.lobit1, !dbg !13
  %13 = sext i32 %12 to i64, !dbg !13
  %14 = add nsw i64 %8, %13, !dbg !13
  %15 = and i64 %14, 31, !dbg !13
  %16 = shl nsw i64 %6, 6, !dbg !13
  %17 = mul nsw i64 %13, -4096, !dbg !13
  %18 = add nsw i64 %16, %17, !dbg !13
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !14, !range !34
  %20 = zext nneg i32 %19 to i64, !dbg !14
  %21 = lshr i64 %20, 6, !dbg !14
  %22 = or disjoint i64 %9, 63, !dbg !15
  %23 = add nsw i64 %9, 64, !dbg !15
  %24 = icmp uge i64 %23, %22, !dbg !15
  br label %25, !dbg !8

25:                                               ; preds = %256, %4
  %26 = phi i64 [ %510, %256 ], [ 0, %4 ], !dbg !8
  %27 = phi <4 x float> [ %393, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %28 = phi <4 x float> [ %401, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %29 = phi <8 x float> [ %506, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %30 = phi <1 x float> [ %357, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %31 = phi <4 x float> [ %433, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %32 = phi <4 x float> [ %441, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %33 = phi <8 x float> [ %507, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %34 = phi <4 x float> [ %473, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %35 = phi <4 x float> [ %481, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %36 = phi <8 x float> [ %508, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %37 = phi <4 x float> [ %497, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %38 = phi <4 x float> [ %505, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %39 = phi <8 x float> [ %509, %256 ], [ zeroinitializer, %4 ], !dbg !8
  %.not = icmp sgt i64 %26, %6, !dbg !8
  br i1 %.not, label %511, label %40, !dbg !8

40:                                               ; preds = %25
  %41 = shl nsw i64 %26, 6, !dbg !8
  %42 = lshr i64 %26, 6, !dbg !17
  %43 = shl i64 %42, 12, !dbg !17
  %44 = sub i64 %41, %43, !dbg !17
  br label %45, !dbg !22

45:                                               ; preds = %50, %40
  %46 = phi i64 [ %172, %50 ], [ 0, %40 ], !dbg !22
  %47 = phi <4 x float> [ %167, %50 ], [ zeroinitializer, %40 ], !dbg !22
  %48 = phi <16 x float> [ %171, %50 ], [ zeroinitializer, %40 ], !dbg !22
  %49 = icmp samesign ult i64 %46, 2, !dbg !22
  br i1 %49, label %50, label %173, !dbg !22

50:                                               ; preds = %45
  %51 = shl nuw nsw i64 %46, 6, !dbg !22
  %.idx = shl nuw nsw i64 %15, 20, !dbg !13
  %52 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !13
  %.idx20 = shl nsw i64 %18, 8, !dbg !13
  %53 = getelementptr i8, ptr addrspace(1) %52, i64 %.idx20, !dbg !13
  %54 = getelementptr [2 x i8], ptr addrspace(1) %53, i64 %51, !dbg !13
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !13, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !14
  %address.local50 = shl nuw nsw i64 %address.local, 3, !dbg !13
  %address.local51 = and i64 %address.local50, 1536, !dbg !13
  %address.local52 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local51, !dbg !13
  %address.local53 = ptrtoint ptr addrspace(3) %address.local52 to i64, !dbg !13
  %address.local54 = trunc i64 %address.local53 to i32, !dbg !13
  %55 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local54), !dbg !13
  %56 = zext i32 %55 to i64, !dbg !13
  %57 = inttoptr i64 %56 to ptr addrspace(3), !dbg !13
  %address.local55 = shl nuw nsw i64 %address.local, 7, !dbg !13
  %address.local56 = and i64 %address.local55, 8064, !dbg !13
  %address.local57 = lshr i64 %address.local, 3, !dbg !13
  %address.local58 = and i64 %address.local57, 24, !dbg !13
  %address.local59 = or disjoint i64 %address.local56, %address.local58, !dbg !13
  %address.local60 = getelementptr [2 x i8], ptr addrspace(1) %54, i64 %address.local59, !dbg !13
  %58 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %59 = ptrtoint ptr addrspace(1) %address.local60 to i64, !dbg !13
  %60 = sub i64 %59, %58, !dbg !13
  %dma.buffer.offset409 = trunc i64 %60 to i32, !dbg !13
  %61 = trunc i64 %58 to i32, !dbg !13
  %62 = insertelement <4 x i32> poison, i32 %61, i64 0, !dbg !13
  %63 = lshr i64 %58, 32, !dbg !13
  %64 = trunc i64 %63 to i32, !dbg !13
  %65 = insertelement <4 x i32> %62, i32 %64, i64 1, !dbg !13
  %66 = insertelement <4 x i32> %65, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc410 = insertelement <4 x i32> %66, i32 131072, i64 3, !dbg !13
  %67 = ptrtoint ptr addrspace(3) %57 to i32, !dbg !13
  %dma.buffer.lds411 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %67), !dbg !13
  %68 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds411, i32 %dma.buffer.offset409, <4 x i32> %dma.buffer.rsrc410) #10, !dbg !13
  %address.local61 = or disjoint i64 %address.local, 256, !dbg !13
  %address.local62 = shl nuw nsw i64 %address.local61, 3, !dbg !13
  %address.local63 = and i64 %address.local62, 3584, !dbg !13
  %address.local64 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local63, !dbg !13
  %address.local65 = ptrtoint ptr addrspace(3) %address.local64 to i64, !dbg !13
  %address.local66 = trunc i64 %address.local65 to i32, !dbg !13
  %69 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local66), !dbg !13
  %70 = zext i32 %69 to i64, !dbg !13
  %71 = inttoptr i64 %70 to ptr addrspace(3), !dbg !13
  %address.local67 = shl nuw nsw i64 %address.local, 7, !dbg !13
  %address.local68 = and i64 %address.local67, 8064, !dbg !13
  %address.local69 = lshr i64 %address.local61, 3, !dbg !13
  %address.local70 = and i64 %address.local69, 56, !dbg !13
  %address.local71 = or disjoint i64 %address.local68, %address.local70, !dbg !13
  %address.local72 = getelementptr [2 x i8], ptr addrspace(1) %54, i64 %address.local71, !dbg !13
  %72 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %73 = ptrtoint ptr addrspace(1) %address.local72 to i64, !dbg !13
  %74 = sub i64 %73, %72, !dbg !13
  %dma.buffer.offset406 = trunc i64 %74 to i32, !dbg !13
  %75 = trunc i64 %72 to i32, !dbg !13
  %76 = insertelement <4 x i32> poison, i32 %75, i64 0, !dbg !13
  %77 = lshr i64 %72, 32, !dbg !13
  %78 = trunc i64 %77 to i32, !dbg !13
  %79 = insertelement <4 x i32> %76, i32 %78, i64 1, !dbg !13
  %80 = insertelement <4 x i32> %79, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc407 = insertelement <4 x i32> %80, i32 131072, i64 3, !dbg !13
  %81 = ptrtoint ptr addrspace(3) %71 to i32, !dbg !13
  %dma.buffer.lds408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %81), !dbg !13
  %82 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds408, i32 %dma.buffer.offset406, <4 x i32> %dma.buffer.rsrc407) #10, !dbg !13
  %83 = shl nuw nsw i64 %46, 12, !dbg !17
  %84 = add nuw i64 %26, %83, !dbg !17
  %.lobit21 = ashr i64 %84, 63, !dbg !17
  %85 = xor i64 %84, %.lobit21, !dbg !17
  %86 = sdiv i64 %85, 8192, !dbg !17
  %.lobit22 = ashr i64 %84, 63, !dbg !17
  %87 = xor i64 %86, %.lobit22, !dbg !17
  %88 = add nsw i64 %87, %8, !dbg !17
  %89 = add nuw nsw i64 %51, %42, !dbg !17
  %90 = and i64 %89, 127, !dbg !17
  %91 = shl i64 %88, 19, !dbg !17
  %92 = and i64 %91, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %93 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %92, !dbg !17
  %94 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %90, !dbg !17
  %.idx23 = shl i64 %44, 8, !dbg !17
  %95 = getelementptr i8, ptr addrspace(1) %94, i64 %.idx23, !dbg !17
  %address.local73 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local51, !dbg !17
  %address.local74 = ptrtoint ptr addrspace(3) %address.local73 to i64, !dbg !17
  %address.local75 = trunc i64 %address.local74 to i32, !dbg !17
  %96 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local75), !dbg !17
  %97 = zext i32 %96 to i64, !dbg !17
  %98 = inttoptr i64 %97 to ptr addrspace(3), !dbg !17
  %address.local76 = getelementptr [2 x i8], ptr addrspace(1) %95, i64 %address.local59, !dbg !17
  %99 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %100 = ptrtoint ptr addrspace(1) %address.local76 to i64, !dbg !17
  %101 = sub i64 %100, %99, !dbg !17
  %dma.buffer.offset403 = trunc i64 %101 to i32, !dbg !17
  %102 = trunc i64 %99 to i32, !dbg !17
  %103 = insertelement <4 x i32> poison, i32 %102, i64 0, !dbg !17
  %104 = lshr i64 %99, 32, !dbg !17
  %105 = trunc i64 %104 to i32, !dbg !17
  %106 = insertelement <4 x i32> %103, i32 %105, i64 1, !dbg !17
  %107 = insertelement <4 x i32> %106, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc404 = insertelement <4 x i32> %107, i32 131072, i64 3, !dbg !17
  %108 = ptrtoint ptr addrspace(3) %98 to i32, !dbg !17
  %dma.buffer.lds405 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %108), !dbg !17
  %109 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds405, i32 %dma.buffer.offset403, <4 x i32> %dma.buffer.rsrc404) #10, !dbg !17
  %address.local77 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local63, !dbg !17
  %address.local78 = ptrtoint ptr addrspace(3) %address.local77 to i64, !dbg !17
  %address.local79 = trunc i64 %address.local78 to i32, !dbg !17
  %110 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local79), !dbg !17
  %111 = zext i32 %110 to i64, !dbg !17
  %112 = inttoptr i64 %111 to ptr addrspace(3), !dbg !17
  %address.local80 = getelementptr [2 x i8], ptr addrspace(1) %95, i64 %address.local71, !dbg !17
  %113 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %114 = ptrtoint ptr addrspace(1) %address.local80 to i64, !dbg !17
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
  %address.local81 = lshr i32 %address.thread, 2, !dbg !14
  %address.local82 = and i32 %address.local81, 4, !dbg !14
  %address.local83 = shl nuw nsw i32 %address.thread, 1, !dbg !14
  %address.local84 = and i32 %address.local83, 384, !dbg !14
  %address.local85 = or disjoint i32 %address.local82, %address.local84, !dbg !14
  %address.local86 = shl nuw nsw i32 %address.thread, 3, !dbg !14
  %address.local87 = and i32 %address.local86, 120, !dbg !14
  %address.local88 = or disjoint i32 %address.local85, %address.local87, !dbg !14
  %address.local89 = shl nuw nsw i32 %address.thread, 4, !dbg !14
  %address.local90 = and i32 %address.local89, 512, !dbg !14
  %address.local91 = or disjoint i32 %address.local88, %address.local90, !dbg !14
  %address.local92 = zext nneg i32 %address.local91 to i64, !dbg !14
  %address.local93 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local92, !dbg !14
  %124 = load <4 x half>, ptr addrspace(3) %address.local93, align 2, !dbg !14
  %address.local94 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local92, !dbg !14
  %address.local95 = getelementptr i8, ptr addrspace(3) %address.local94, i64 2048, !dbg !14
  %125 = load <4 x half>, ptr addrspace(3) %address.local95, align 2, !dbg !14
  %address.local96 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local92, !dbg !14
  %address.local97 = getelementptr i8, ptr addrspace(3) %address.local96, i64 4096, !dbg !14
  %126 = load <4 x half>, ptr addrspace(3) %address.local97, align 2, !dbg !14
  %127 = fmul <4 x half> %124, splat (half 0xH3015), !dbg !14
  %address.local98 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local92, !dbg !14
  %address.local99 = getelementptr i8, ptr addrspace(3) %address.local98, i64 6144, !dbg !14
  %128 = load <4 x half>, ptr addrspace(3) %address.local99, align 2, !dbg !14
  %129 = fmul <4 x half> %125, splat (half 0xH3015), !dbg !14
  %130 = fmul <4 x half> %126, splat (half 0xH3015), !dbg !14
  %131 = fmul <4 x half> %128, splat (half 0xH3015), !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local100 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local101 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local100, !dbg !9
  %address.local102 = lshr i64 %address.local, 2, !dbg !9
  %address.local103 = mul nsw i64 %address.local102, -512, !dbg !9
  %address.local104 = getelementptr [2 x i8], ptr addrspace(3) %address.local101, i64 %address.local103, !dbg !9
  %address.local105 = shl nuw nsw i64 %address.local, 1, !dbg !9
  %address.local106 = and i64 %address.local105, 24, !dbg !9
  %address.local107 = getelementptr [2 x i8], ptr addrspace(3) %address.local104, i64 %address.local106, !dbg !9
  %address.local108 = lshr i64 %address.local, 2, !dbg !14
  %address.local109 = and i64 %address.local108, 12, !dbg !14
  %address.local110 = getelementptr [2 x i8], ptr addrspace(3) %address.local107, i64 %address.local109, !dbg !9
  %address.local111 = and i64 %address.local, 63, !dbg !14
  %address.local112 = lshr i64 %address.local111, 5, !dbg !14
  %address.local113 = mul nuw nsw i64 %address.local112, 504, !dbg !14
  %address.local114 = getelementptr [2 x i8], ptr addrspace(3) %address.local110, i64 %address.local113, !dbg !9
  %132 = load <4 x half>, ptr addrspace(3) %address.local114, align 2, !dbg !9
  %133 = shufflevector <16 x float> %48, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %134 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %132, <4 x float> %133) #11, !dbg !9
  %address.local115 = lshr i32 %address.thread, 2, !dbg !9
  %address.local116 = and i32 %address.local115, 4, !dbg !9
  %address.local117 = shl nuw nsw i32 %address.thread, 1, !dbg !9
  %address.local118 = and i32 %address.local117, 24, !dbg !9
  %address.local119 = or disjoint i32 %address.local116, %address.local118, !dbg !9
  %address.local120 = shl nuw nsw i32 %address.thread, 4, !dbg !9
  %address.local121 = and i32 %address.local120, 512, !dbg !9
  %address.local122 = or disjoint i32 %address.local119, %address.local121, !dbg !9
  %address.local123 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local124 = and i32 %address.local123, 384, !dbg !9
  %address.local125 = or disjoint i32 %address.local122, %address.local124, !dbg !9
  %address.local126 = zext nneg i32 %address.local125 to i64, !dbg !9
  %address.local127 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !9
  %address.local128 = getelementptr i8, ptr addrspace(3) %address.local127, i64 2048, !dbg !9
  %135 = load <4 x half>, ptr addrspace(3) %address.local128, align 2, !dbg !9
  %136 = shufflevector <16 x float> %48, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %137 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %129, <4 x half> %135, <4 x float> %134) #11, !dbg !9
  %138 = shufflevector <16 x float> %48, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %address.local129 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !9
  %address.local130 = getelementptr i8, ptr addrspace(3) %address.local129, i64 4096, !dbg !9
  %139 = load <4 x half>, ptr addrspace(3) %address.local130, align 2, !dbg !9
  %address.local131 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !9
  %address.local132 = getelementptr i8, ptr addrspace(3) %address.local131, i64 6144, !dbg !9
  %140 = load <4 x half>, ptr addrspace(3) %address.local132, align 2, !dbg !9
  %141 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %139, <4 x float> %137) #11, !dbg !9
  %address.local133 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local100, !dbg !9
  %address.local134 = getelementptr [2 x i8], ptr addrspace(3) %address.local133, i64 %address.local103, !dbg !9
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) %address.local134, i64 %address.local106, !dbg !9
  %address.local136 = getelementptr [2 x i8], ptr addrspace(3) %address.local135, i64 %address.local109, !dbg !9
  %address.local137 = getelementptr [2 x i8], ptr addrspace(3) %address.local136, i64 %address.local113, !dbg !9
  %address.local138 = getelementptr i8, ptr addrspace(3) %address.local137, i64 64, !dbg !9
  %142 = load <4 x half>, ptr addrspace(3) %address.local138, align 2, !dbg !9
  %143 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %140, <4 x float> %141) #11, !dbg !9
  %address.local139 = lshr i32 %address.thread, 2, !dbg !9
  %address.local140 = and i32 %address.local139, 4, !dbg !9
  %address.local141 = shl nuw nsw i32 %address.thread, 1, !dbg !9
  %address.local142 = and i32 %address.local141, 24, !dbg !9
  %address.local143 = or disjoint i32 %address.local140, %address.local142, !dbg !9
  %address.local144 = shl nuw nsw i32 %address.thread, 4, !dbg !9
  %address.local145 = and i32 %address.local144, 512, !dbg !9
  %address.local146 = or disjoint i32 %address.local143, %address.local145, !dbg !9
  %address.local147 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local148 = and i32 %address.local147, 384, !dbg !9
  %address.local149 = or disjoint i32 %address.local146, %address.local148, !dbg !9
  %address.local150 = zext nneg i32 %address.local149 to i64, !dbg !9
  %address.local151 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local150, !dbg !9
  %address.local152 = getelementptr i8, ptr addrspace(3) %address.local151, i64 2112, !dbg !9
  %144 = load <4 x half>, ptr addrspace(3) %address.local152, align 2, !dbg !9
  %145 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %142, <4 x float> %136) #11, !dbg !9
  %146 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %129, <4 x half> %144, <4 x float> %145) #11, !dbg !9
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local150, !dbg !9
  %address.local154 = getelementptr i8, ptr addrspace(3) %address.local153, i64 4160, !dbg !9
  %147 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !9
  %address.local155 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local150, !dbg !9
  %address.local156 = getelementptr i8, ptr addrspace(3) %address.local155, i64 6208, !dbg !9
  %148 = load <4 x half>, ptr addrspace(3) %address.local156, align 2, !dbg !9
  %149 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %147, <4 x float> %146) #11, !dbg !9
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local100, !dbg !9
  %address.local158 = getelementptr [2 x i8], ptr addrspace(3) %address.local157, i64 %address.local103, !dbg !9
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) %address.local158, i64 %address.local106, !dbg !9
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) %address.local159, i64 %address.local109, !dbg !9
  %address.local161 = getelementptr [2 x i8], ptr addrspace(3) %address.local160, i64 %address.local113, !dbg !9
  %address.local162 = getelementptr i8, ptr addrspace(3) %address.local161, i64 128, !dbg !9
  %150 = load <4 x half>, ptr addrspace(3) %address.local162, align 2, !dbg !9
  %151 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %148, <4 x float> %149) #11, !dbg !9
  %address.local163 = lshr i32 %address.thread, 2, !dbg !9
  %address.local164 = and i32 %address.local163, 4, !dbg !9
  %address.local165 = shl nuw nsw i32 %address.thread, 1, !dbg !9
  %address.local166 = and i32 %address.local165, 24, !dbg !9
  %address.local167 = or disjoint i32 %address.local164, %address.local166, !dbg !9
  %address.local168 = shl nuw nsw i32 %address.thread, 4, !dbg !9
  %address.local169 = and i32 %address.local168, 512, !dbg !9
  %address.local170 = or disjoint i32 %address.local167, %address.local169, !dbg !9
  %address.local171 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local172 = and i32 %address.local171, 384, !dbg !9
  %address.local173 = or disjoint i32 %address.local170, %address.local172, !dbg !9
  %address.local174 = zext nneg i32 %address.local173 to i64, !dbg !9
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local174, !dbg !9
  %address.local176 = getelementptr i8, ptr addrspace(3) %address.local175, i64 2176, !dbg !9
  %152 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !9
  %153 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %150, <4 x float> %138) #11, !dbg !9
  %154 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %129, <4 x half> %152, <4 x float> %153) #11, !dbg !9
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local174, !dbg !9
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local177, i64 4224, !dbg !9
  %155 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !9
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local174, !dbg !9
  %address.local180 = getelementptr i8, ptr addrspace(3) %address.local179, i64 6272, !dbg !9
  %156 = load <4 x half>, ptr addrspace(3) %address.local180, align 2, !dbg !9
  %157 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %155, <4 x float> %154) #11, !dbg !9
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local100, !dbg !9
  %address.local182 = getelementptr [2 x i8], ptr addrspace(3) %address.local181, i64 %address.local103, !dbg !9
  %address.local183 = getelementptr [2 x i8], ptr addrspace(3) %address.local182, i64 %address.local106, !dbg !9
  %address.local184 = getelementptr [2 x i8], ptr addrspace(3) %address.local183, i64 %address.local109, !dbg !9
  %address.local185 = getelementptr [2 x i8], ptr addrspace(3) %address.local184, i64 %address.local113, !dbg !9
  %address.local186 = getelementptr i8, ptr addrspace(3) %address.local185, i64 192, !dbg !9
  %158 = load <4 x half>, ptr addrspace(3) %address.local186, align 2, !dbg !9
  %159 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %156, <4 x float> %157) #11, !dbg !9
  %address.local187 = lshr i32 %address.thread, 2, !dbg !9
  %address.local188 = and i32 %address.local187, 4, !dbg !9
  %address.local189 = shl nuw nsw i32 %address.thread, 1, !dbg !9
  %address.local190 = and i32 %address.local189, 24, !dbg !9
  %address.local191 = or disjoint i32 %address.local188, %address.local190, !dbg !9
  %address.local192 = shl nuw nsw i32 %address.thread, 4, !dbg !9
  %address.local193 = and i32 %address.local192, 512, !dbg !9
  %address.local194 = or disjoint i32 %address.local191, %address.local193, !dbg !9
  %address.local195 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local196 = and i32 %address.local195, 384, !dbg !9
  %address.local197 = or disjoint i32 %address.local194, %address.local196, !dbg !9
  %address.local198 = zext nneg i32 %address.local197 to i64, !dbg !9
  %address.local199 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local198, !dbg !9
  %address.local200 = getelementptr i8, ptr addrspace(3) %address.local199, i64 2240, !dbg !9
  %160 = load <4 x half>, ptr addrspace(3) %address.local200, align 2, !dbg !9
  %161 = shufflevector <4 x float> %159, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %162 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %158, <4 x float> %47) #11, !dbg !9
  %163 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %129, <4 x half> %160, <4 x float> %162) #11, !dbg !9
  %address.local201 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local198, !dbg !9
  %address.local202 = getelementptr i8, ptr addrspace(3) %address.local201, i64 4288, !dbg !9
  %164 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !9
  %address.local203 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local198, !dbg !9
  %address.local204 = getelementptr i8, ptr addrspace(3) %address.local203, i64 6336, !dbg !9
  %165 = load <4 x half>, ptr addrspace(3) %address.local204, align 2, !dbg !9
  %166 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %164, <4 x float> %163) #11, !dbg !9
  %167 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %165, <4 x float> %166) #11, !dbg !9
  %168 = shufflevector <4 x float> %167, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %169 = shufflevector <4 x float> %143, <4 x float> %151, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %170 = shufflevector <16 x float> %169, <16 x float> %161, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %171 = shufflevector <16 x float> %170, <16 x float> %168, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %172 = add nuw nsw i64 %46, 1, !dbg !22
  br label %45, !dbg !22

173:                                              ; preds = %45
  %174 = add nsw i64 %42, %8, !dbg !18
  %175 = shl i64 %174, 19, !dbg !18
  %176 = and i64 %175, 16252928, !dbg !18
  %177 = shl nsw i64 %44, 7, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %178 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %176, !dbg !18
  %179 = getelementptr [2 x i8], ptr addrspace(1) %178, i64 %177, !dbg !18
  %address.thread205 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !18, !range !34, !frisk.register_lifetimes !1
  %address.local206 = zext nneg i32 %address.thread205 to i64, !dbg !14
  %address.local207 = shl nuw nsw i64 %address.local206, 3, !dbg !13
  %address.local208 = and i64 %address.local207, 1536, !dbg !18
  %address.local209 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local208, !dbg !18
  %address.local210 = ptrtoint ptr addrspace(3) %address.local209 to i64, !dbg !18
  %address.local211 = trunc i64 %address.local210 to i32, !dbg !18
  %180 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local211), !dbg !18
  %181 = zext i32 %180 to i64, !dbg !18
  %182 = inttoptr i64 %181 to ptr addrspace(3), !dbg !18
  %address.local212 = shl nuw nsw i64 %address.local206, 5, !dbg !18
  %address.local213 = and i64 %address.local212, 8064, !dbg !18
  %address.local214 = shl nuw nsw i64 %address.local206, 3, !dbg !18
  %address.local215 = and i64 %address.local214, 24, !dbg !18
  %address.local216 = or disjoint i64 %address.local213, %address.local215, !dbg !18
  %address.local217 = getelementptr [2 x i8], ptr addrspace(1) %179, i64 %address.local216, !dbg !18
  %183 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %184 = ptrtoint ptr addrspace(1) %address.local217 to i64, !dbg !18
  %185 = sub i64 %184, %183, !dbg !18
  %dma.buffer.offset421 = trunc i64 %185 to i32, !dbg !18
  %186 = trunc i64 %183 to i32, !dbg !18
  %187 = insertelement <4 x i32> poison, i32 %186, i64 0, !dbg !18
  %188 = lshr i64 %183, 32, !dbg !18
  %189 = trunc i64 %188 to i32, !dbg !18
  %190 = insertelement <4 x i32> %187, i32 %189, i64 1, !dbg !18
  %191 = insertelement <4 x i32> %190, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc422 = insertelement <4 x i32> %191, i32 131072, i64 3, !dbg !18
  %192 = ptrtoint ptr addrspace(3) %182 to i32, !dbg !18
  %dma.buffer.lds423 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %192), !dbg !18
  %193 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds423, i32 %dma.buffer.offset421, <4 x i32> %dma.buffer.rsrc422) #10, !dbg !18
  %194 = or disjoint i64 %41, 63, !dbg !15
  %195 = icmp uge i64 %9, %194, !dbg !15
  %196 = and i1 %24, %195, !dbg !15
  br i1 %196, label %197, label %198, !dbg !15

197:                                              ; preds = %173
  br label %254, !dbg !15

198:                                              ; preds = %173
  %199 = add nsw i64 %9, %20, !dbg !15
  %200 = shl nuw nsw i64 %21, 4, !dbg !15
  %201 = add nsw i64 %199, %200, !dbg !15
  %202 = and i32 %19, 240, !dbg !15
  %index.compact = zext nneg i32 %202 to i64, !dbg !15
  %203 = sub nsw i64 %201, %index.compact, !dbg !15
  %204 = add nsw i64 %203, 1, !dbg !23
  %205 = and i64 %20, 48, !dbg !15
  %206 = or disjoint i64 %41, %205, !dbg !15
  %.not4 = icmp ugt i64 %204, %206, !dbg !24
  %207 = select i1 %.not4, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %208 = or disjoint i64 %206, 1, !dbg !15
  %.not5 = icmp ugt i64 %204, %208, !dbg !24
  %209 = select i1 %.not5, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %210 = or disjoint i64 %206, 2, !dbg !15
  %.not6 = icmp ugt i64 %204, %210, !dbg !24
  %211 = select i1 %.not6, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %212 = or disjoint i64 %206, 3, !dbg !15
  %.not7 = icmp ugt i64 %204, %212, !dbg !24
  %213 = select i1 %.not7, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %214 = or disjoint i64 %206, 4, !dbg !15
  %.not8 = icmp ugt i64 %204, %214, !dbg !24
  %215 = select i1 %.not8, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %216 = or disjoint i64 %206, 5, !dbg !15
  %.not9 = icmp ugt i64 %204, %216, !dbg !24
  %217 = select i1 %.not9, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %218 = or disjoint i64 %206, 6, !dbg !15
  %.not10 = icmp ugt i64 %204, %218, !dbg !24
  %219 = select i1 %.not10, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %220 = or disjoint i64 %206, 7, !dbg !15
  %.not11 = icmp ugt i64 %204, %220, !dbg !24
  %221 = select i1 %.not11, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %222 = or disjoint i64 %206, 8, !dbg !15
  %.not12 = icmp ugt i64 %204, %222, !dbg !24
  %223 = select i1 %.not12, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %224 = or disjoint i64 %206, 9, !dbg !15
  %.not13 = icmp ugt i64 %204, %224, !dbg !24
  %225 = select i1 %.not13, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %226 = or disjoint i64 %206, 10, !dbg !15
  %.not14 = icmp ugt i64 %204, %226, !dbg !24
  %227 = select i1 %.not14, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %228 = or disjoint i64 %206, 11, !dbg !15
  %.not15 = icmp ugt i64 %204, %228, !dbg !24
  %229 = select i1 %.not15, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %230 = or disjoint i64 %206, 12, !dbg !15
  %.not16 = icmp ugt i64 %204, %230, !dbg !24
  %231 = select i1 %.not16, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %232 = or disjoint i64 %206, 13, !dbg !15
  %.not17 = icmp ugt i64 %204, %232, !dbg !24
  %233 = select i1 %.not17, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %234 = or disjoint i64 %206, 14, !dbg !15
  %.not18 = icmp ugt i64 %204, %234, !dbg !24
  %235 = select i1 %.not18, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %236 = or disjoint i64 %206, 15, !dbg !15
  %.not19 = icmp ugt i64 %204, %236, !dbg !24
  %237 = select i1 %.not19, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %238 = insertelement <16 x float> poison, float %207, i64 0, !dbg !15
  %239 = insertelement <16 x float> %238, float %209, i64 1, !dbg !15
  %240 = insertelement <16 x float> %239, float %211, i64 2, !dbg !15
  %241 = insertelement <16 x float> %240, float %213, i64 3, !dbg !15
  %242 = insertelement <16 x float> %241, float %215, i64 4, !dbg !15
  %243 = insertelement <16 x float> %242, float %217, i64 5, !dbg !15
  %244 = insertelement <16 x float> %243, float %219, i64 6, !dbg !15
  %245 = insertelement <16 x float> %244, float %221, i64 7, !dbg !15
  %246 = insertelement <16 x float> %245, float %223, i64 8, !dbg !15
  %247 = insertelement <16 x float> %246, float %225, i64 9, !dbg !15
  %248 = insertelement <16 x float> %247, float %227, i64 10, !dbg !15
  %249 = insertelement <16 x float> %248, float %229, i64 11, !dbg !15
  %250 = insertelement <16 x float> %249, float %231, i64 12, !dbg !15
  %251 = insertelement <16 x float> %250, float %233, i64 13, !dbg !15
  %252 = insertelement <16 x float> %251, float %235, i64 14, !dbg !15
  %253 = insertelement <16 x float> %252, float %237, i64 15, !dbg !15
  br label %254, !dbg !15

254:                                              ; preds = %197, %198
  %255 = phi <16 x float> [ %253, %198 ], [ zeroinitializer, %197 ], !dbg !15
  br label %256, !dbg !15

256:                                              ; preds = %254
  %257 = shufflevector <16 x float> %48, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %258 = shufflevector <16 x float> %255, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %259 = fadd <4 x float> %257, %258, !dbg !26
  %260 = extractelement <4 x float> %259, i64 0, !dbg !27
  %261 = call float @llvm.amdgcn.exp2.f32(float %260), !dbg !27
  %262 = insertelement <4 x float> poison, float %261, i64 0, !dbg !27
  %263 = extractelement <4 x float> %259, i64 1, !dbg !27
  %264 = call float @llvm.amdgcn.exp2.f32(float %263), !dbg !27
  %265 = fadd float %261, %264, !dbg !28
  %266 = insertelement <4 x float> %262, float %264, i64 1, !dbg !27
  %267 = extractelement <4 x float> %259, i64 2, !dbg !27
  %268 = call float @llvm.amdgcn.exp2.f32(float %267), !dbg !27
  %269 = fadd float %265, %268, !dbg !28
  %270 = insertelement <4 x float> %266, float %268, i64 2, !dbg !27
  %271 = extractelement <4 x float> %259, i64 3, !dbg !27
  %272 = call float @llvm.amdgcn.exp2.f32(float %271), !dbg !27
  %273 = fadd float %269, %272, !dbg !28
  %274 = insertelement <4 x float> %270, float %272, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %274) #9, !dbg !29, !frisk.register_lifetimes !1
  %275 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %275) #9, !dbg !26, !frisk.register_lifetimes !1
  %276 = shufflevector <16 x float> %48, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %277 = shufflevector <16 x float> %255, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %278 = fadd <4 x float> %276, %277, !dbg !26
  %279 = extractelement <4 x float> %278, i64 0, !dbg !27
  %280 = call float @llvm.amdgcn.exp2.f32(float %279), !dbg !27
  %281 = fadd float %273, %280, !dbg !28
  %282 = insertelement <4 x float> poison, float %280, i64 0, !dbg !27
  %283 = extractelement <4 x float> %278, i64 1, !dbg !27
  %284 = call float @llvm.amdgcn.exp2.f32(float %283), !dbg !27
  %285 = fadd float %281, %284, !dbg !28
  %286 = insertelement <4 x float> %282, float %284, i64 1, !dbg !27
  %287 = extractelement <4 x float> %278, i64 2, !dbg !27
  %288 = call float @llvm.amdgcn.exp2.f32(float %287), !dbg !27
  %289 = fadd float %285, %288, !dbg !28
  %290 = insertelement <4 x float> %286, float %288, i64 2, !dbg !27
  %291 = extractelement <4 x float> %278, i64 3, !dbg !27
  %292 = call float @llvm.amdgcn.exp2.f32(float %291), !dbg !27
  %293 = fadd float %289, %292, !dbg !28
  %294 = insertelement <4 x float> %290, float %292, i64 3, !dbg !27
  %math.result44 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %294) #9, !dbg !29, !frisk.register_lifetimes !1
  %295 = fptrunc <4 x float> %math.result44 to <4 x half>, !dbg !29
  %math.packed45 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %295) #9, !dbg !26, !frisk.register_lifetimes !1
  %296 = shufflevector <16 x float> %48, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %297 = shufflevector <16 x float> %255, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %298 = fadd <4 x float> %296, %297, !dbg !26
  %299 = extractelement <4 x float> %298, i64 0, !dbg !27
  %300 = call float @llvm.amdgcn.exp2.f32(float %299), !dbg !27
  %301 = fadd float %293, %300, !dbg !28
  %302 = insertelement <4 x float> poison, float %300, i64 0, !dbg !27
  %303 = extractelement <4 x float> %298, i64 1, !dbg !27
  %304 = call float @llvm.amdgcn.exp2.f32(float %303), !dbg !27
  %305 = fadd float %301, %304, !dbg !28
  %306 = insertelement <4 x float> %302, float %304, i64 1, !dbg !27
  %307 = extractelement <4 x float> %298, i64 2, !dbg !27
  %308 = call float @llvm.amdgcn.exp2.f32(float %307), !dbg !27
  %309 = fadd float %305, %308, !dbg !28
  %310 = insertelement <4 x float> %306, float %308, i64 2, !dbg !27
  %311 = extractelement <4 x float> %298, i64 3, !dbg !27
  %312 = call float @llvm.amdgcn.exp2.f32(float %311), !dbg !27
  %313 = fadd float %309, %312, !dbg !28
  %314 = insertelement <4 x float> %310, float %312, i64 3, !dbg !27
  %math.result46 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %314) #9, !dbg !29, !frisk.register_lifetimes !1
  %315 = fptrunc <4 x float> %math.result46 to <4 x half>, !dbg !29
  %math.packed47 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %315) #9, !dbg !26, !frisk.register_lifetimes !1
  %316 = shufflevector <16 x float> %255, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %317 = fadd <4 x float> %47, %316, !dbg !26
  %318 = extractelement <4 x float> %317, i64 0, !dbg !27
  %319 = call float @llvm.amdgcn.exp2.f32(float %318), !dbg !27
  %320 = fadd float %313, %319, !dbg !28
  %321 = insertelement <4 x float> poison, float %319, i64 0, !dbg !27
  %322 = extractelement <4 x float> %317, i64 1, !dbg !27
  %323 = call float @llvm.amdgcn.exp2.f32(float %322), !dbg !27
  %324 = fadd float %320, %323, !dbg !28
  %325 = insertelement <4 x float> %321, float %323, i64 1, !dbg !27
  %326 = extractelement <4 x float> %317, i64 2, !dbg !27
  %327 = call float @llvm.amdgcn.exp2.f32(float %326), !dbg !27
  %328 = fadd float %324, %327, !dbg !28
  %329 = insertelement <4 x float> %325, float %327, i64 2, !dbg !27
  %330 = extractelement <4 x float> %317, i64 3, !dbg !27
  %331 = call float @llvm.amdgcn.exp2.f32(float %330), !dbg !27
  %332 = fadd float %328, %331, !dbg !28
  %333 = insertelement <4 x float> %329, float %331, i64 3, !dbg !27
  %math.result48 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %333) #9, !dbg !29, !frisk.register_lifetimes !1
  %334 = fptrunc <4 x float> %math.result48 to <4 x half>, !dbg !29
  %math.packed49 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %334) #9, !dbg !28, !frisk.register_lifetimes !1
  %335 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %336 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %335), !dbg !28
  %337 = add nuw nsw i32 %336, 64, !dbg !28
  %338 = xor i32 %336, 16, !dbg !28
  %339 = and i32 %337, 192, !dbg !28
  %340 = icmp samesign ult i32 %338, %339, !dbg !28
  %341 = select i1 %340, i32 %338, i32 %336, !dbg !28
  %342 = shl nuw nsw i32 %341, 2, !dbg !28
  %343 = bitcast float %332 to i32, !dbg !28
  %344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %342, i32 %343), !dbg !28
  %345 = bitcast i32 %344 to float, !dbg !28
  %346 = fadd float %332, %345, !dbg !28
  %347 = xor i32 %336, 32, !dbg !28
  %348 = icmp samesign ult i32 %347, %339, !dbg !28
  %349 = select i1 %348, i32 %347, i32 %336, !dbg !28
  %350 = shl nuw nsw i32 %349, 2, !dbg !28
  %351 = bitcast float %346 to i32, !dbg !28
  %352 = call i32 @llvm.amdgcn.ds.bpermute(i32 %350, i32 %351), !dbg !28
  %353 = bitcast i32 %352 to float, !dbg !28
  %354 = fadd float %346, %353, !dbg !28
  %355 = extractelement <1 x float> %30, i64 0, !dbg !28
  %356 = fadd float %354, %355, !dbg !28
  %357 = insertelement <1 x float> poison, float %356, i64 0, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread218 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !16, !range !34, !frisk.register_lifetimes !1
  %address.local219 = zext nneg i32 %address.thread218 to i64, !dbg !14
  %address.local220 = and i64 %address.local219, 60, !dbg !16
  %address.local221 = add nuw nsw i64 %address.local220, %address.local219, !dbg !16
  %address.local222 = shl nuw nsw i64 %address.local221, 3, !dbg !16
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local222, !dbg !16
  %address.local224 = and i64 %address.local219, 63, !dbg !14
  %address.local225 = lshr i64 %address.local224, 4, !dbg !16
  %address.local226 = mul nuw nsw i64 %address.local225, 384, !dbg !16
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local226, !dbg !16
  %address.local228 = lshr i64 %address.local219, 2, !dbg !9
  %address.local229 = mul nsw i64 %address.local228, -32, !dbg !16
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local229, !dbg !16
  %358 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local230, i32 0) #0, !dbg !16
  %359 = bitcast <4 x i32> %358 to <8 x half>, !dbg !16
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local222, !dbg !16
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %address.local231, i64 %address.local226, !dbg !16
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local229, !dbg !16
  %address.local234 = getelementptr i8, ptr addrspace(3) %address.local233, i64 256, !dbg !16
  %360 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local234, i32 0) #0, !dbg !16
  %361 = bitcast <4 x i32> %360 to <8 x half>, !dbg !16
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local222, !dbg !16
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local226, !dbg !16
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local236, i64 %address.local229, !dbg !16
  %address.local238 = getelementptr i8, ptr addrspace(3) %address.local237, i64 512, !dbg !16
  %362 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local238, i32 0) #0, !dbg !16
  %363 = bitcast <4 x i32> %362 to <8 x half>, !dbg !16
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local222, !dbg !16
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) %address.local239, i64 %address.local226, !dbg !16
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local229, !dbg !16
  %address.local242 = getelementptr i8, ptr addrspace(3) %address.local241, i64 768, !dbg !16
  %364 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local242, i32 0) #0, !dbg !16
  %365 = bitcast <4 x i32> %364 to <8 x half>, !dbg !16
  %366 = add nsw i64 %42, %8, !dbg !18
  %367 = shl i64 %366, 19, !dbg !18
  %368 = and i64 %367, 16252928, !dbg !18
  %369 = add i64 %368, %177, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %370 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %369, !dbg !18
  %371 = getelementptr i8, ptr addrspace(1) %370, i64 64, !dbg !18
  %address.local243 = shl nuw nsw i64 %address.local219, 3, !dbg !13
  %address.local244 = and i64 %address.local243, 1536, !dbg !18
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local244, !dbg !18
  %address.local246 = ptrtoint ptr addrspace(3) %address.local245 to i64, !dbg !18
  %address.local247 = trunc i64 %address.local246 to i32, !dbg !18
  %372 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local247), !dbg !18
  %373 = zext i32 %372 to i64, !dbg !18
  %374 = inttoptr i64 %373 to ptr addrspace(3), !dbg !18
  %address.local248 = shl nuw nsw i64 %address.local219, 5, !dbg !18
  %address.local249 = and i64 %address.local248, 8064, !dbg !18
  %address.local250 = shl nuw nsw i64 %address.local219, 3, !dbg !18
  %address.local251 = and i64 %address.local250, 24, !dbg !18
  %address.local252 = or disjoint i64 %address.local249, %address.local251, !dbg !18
  %address.local253 = getelementptr [2 x i8], ptr addrspace(1) %371, i64 %address.local252, !dbg !18
  %375 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %376 = ptrtoint ptr addrspace(1) %address.local253 to i64, !dbg !18
  %377 = sub i64 %376, %375, !dbg !18
  %dma.buffer.offset418 = trunc i64 %377 to i32, !dbg !18
  %378 = trunc i64 %375 to i32, !dbg !18
  %379 = insertelement <4 x i32> poison, i32 %378, i64 0, !dbg !18
  %380 = lshr i64 %375, 32, !dbg !18
  %381 = trunc i64 %380 to i32, !dbg !18
  %382 = insertelement <4 x i32> %379, i32 %381, i64 1, !dbg !18
  %383 = insertelement <4 x i32> %382, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc419 = insertelement <4 x i32> %383, i32 131072, i64 3, !dbg !18
  %384 = ptrtoint ptr addrspace(3) %374 to i32, !dbg !18
  %dma.buffer.lds420 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %384), !dbg !18
  %385 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds420, i32 %dma.buffer.offset418, <4 x i32> %dma.buffer.rsrc419) #10, !dbg !18
  %386 = shufflevector <8 x half> %359, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %387 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %386, <4 x float> %27) #11, !dbg !16
  %388 = shufflevector <8 x half> %361, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %389 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed45, <4 x half> %388, <4 x float> %387) #11, !dbg !16
  %390 = shufflevector <8 x half> %363, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %391 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %390, <4 x float> %389) #11, !dbg !16
  %392 = shufflevector <8 x half> %365, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %393 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %392, <4 x float> %391) #11, !dbg !16
  %394 = shufflevector <8 x half> %359, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %395 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %394, <4 x float> %28) #11, !dbg !16
  %396 = shufflevector <8 x half> %361, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %397 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed45, <4 x half> %396, <4 x float> %395) #11, !dbg !16
  %398 = shufflevector <8 x half> %363, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %399 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %398, <4 x float> %397) #11, !dbg !16
  %400 = shufflevector <8 x half> %365, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %401 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %400, <4 x float> %399) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %402 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local230, i32 0) #0, !dbg !16
  %403 = bitcast <4 x i32> %402 to <8 x half>, !dbg !16
  %404 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local234, i32 0) #0, !dbg !16
  %405 = bitcast <4 x i32> %404 to <8 x half>, !dbg !16
  %406 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local238, i32 0) #0, !dbg !16
  %407 = bitcast <4 x i32> %406 to <8 x half>, !dbg !16
  %408 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local242, i32 0) #0, !dbg !16
  %409 = bitcast <4 x i32> %408 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %410 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %369, !dbg !18
  %411 = getelementptr i8, ptr addrspace(1) %410, i64 128, !dbg !18
  %412 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local247), !dbg !18
  %413 = zext i32 %412 to i64, !dbg !18
  %414 = inttoptr i64 %413 to ptr addrspace(3), !dbg !18
  %address.local254 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local252, !dbg !18
  %415 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %416 = ptrtoint ptr addrspace(1) %address.local254 to i64, !dbg !18
  %417 = sub i64 %416, %415, !dbg !18
  %dma.buffer.offset415 = trunc i64 %417 to i32, !dbg !18
  %418 = trunc i64 %415 to i32, !dbg !18
  %419 = insertelement <4 x i32> poison, i32 %418, i64 0, !dbg !18
  %420 = lshr i64 %415, 32, !dbg !18
  %421 = trunc i64 %420 to i32, !dbg !18
  %422 = insertelement <4 x i32> %419, i32 %421, i64 1, !dbg !18
  %423 = insertelement <4 x i32> %422, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc416 = insertelement <4 x i32> %423, i32 131072, i64 3, !dbg !18
  %424 = ptrtoint ptr addrspace(3) %414 to i32, !dbg !18
  %dma.buffer.lds417 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %424), !dbg !18
  %425 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds417, i32 %dma.buffer.offset415, <4 x i32> %dma.buffer.rsrc416) #10, !dbg !18
  %426 = shufflevector <8 x half> %403, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %427 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %426, <4 x float> %31) #11, !dbg !16
  %428 = shufflevector <8 x half> %405, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %429 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed45, <4 x half> %428, <4 x float> %427) #11, !dbg !16
  %430 = shufflevector <8 x half> %407, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %431 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %430, <4 x float> %429) #11, !dbg !16
  %432 = shufflevector <8 x half> %409, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %433 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %432, <4 x float> %431) #11, !dbg !16
  %434 = shufflevector <8 x half> %403, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %435 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %434, <4 x float> %32) #11, !dbg !16
  %436 = shufflevector <8 x half> %405, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %437 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed45, <4 x half> %436, <4 x float> %435) #11, !dbg !16
  %438 = shufflevector <8 x half> %407, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %439 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %438, <4 x float> %437) #11, !dbg !16
  %440 = shufflevector <8 x half> %409, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %441 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %440, <4 x float> %439) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %442 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local230, i32 0) #0, !dbg !16
  %443 = bitcast <4 x i32> %442 to <8 x half>, !dbg !16
  %444 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local234, i32 0) #0, !dbg !16
  %445 = bitcast <4 x i32> %444 to <8 x half>, !dbg !16
  %446 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local238, i32 0) #0, !dbg !16
  %447 = bitcast <4 x i32> %446 to <8 x half>, !dbg !16
  %448 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local242, i32 0) #0, !dbg !16
  %449 = bitcast <4 x i32> %448 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %450 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %369, !dbg !18
  %451 = getelementptr i8, ptr addrspace(1) %450, i64 192, !dbg !18
  %452 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local247), !dbg !18
  %453 = zext i32 %452 to i64, !dbg !18
  %454 = inttoptr i64 %453 to ptr addrspace(3), !dbg !18
  %address.local255 = getelementptr [2 x i8], ptr addrspace(1) %451, i64 %address.local252, !dbg !18
  %455 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %456 = ptrtoint ptr addrspace(1) %address.local255 to i64, !dbg !18
  %457 = sub i64 %456, %455, !dbg !18
  %dma.buffer.offset412 = trunc i64 %457 to i32, !dbg !18
  %458 = trunc i64 %455 to i32, !dbg !18
  %459 = insertelement <4 x i32> poison, i32 %458, i64 0, !dbg !18
  %460 = lshr i64 %455, 32, !dbg !18
  %461 = trunc i64 %460 to i32, !dbg !18
  %462 = insertelement <4 x i32> %459, i32 %461, i64 1, !dbg !18
  %463 = insertelement <4 x i32> %462, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc413 = insertelement <4 x i32> %463, i32 131072, i64 3, !dbg !18
  %464 = ptrtoint ptr addrspace(3) %454 to i32, !dbg !18
  %dma.buffer.lds414 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %464), !dbg !18
  %465 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds414, i32 %dma.buffer.offset412, <4 x i32> %dma.buffer.rsrc413) #10, !dbg !18
  %466 = shufflevector <8 x half> %443, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %467 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %466, <4 x float> %34) #11, !dbg !16
  %468 = shufflevector <8 x half> %445, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %469 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed45, <4 x half> %468, <4 x float> %467) #11, !dbg !16
  %470 = shufflevector <8 x half> %447, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %471 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %470, <4 x float> %469) #11, !dbg !16
  %472 = shufflevector <8 x half> %449, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %473 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %472, <4 x float> %471) #11, !dbg !16
  %474 = shufflevector <8 x half> %443, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %475 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %474, <4 x float> %35) #11, !dbg !16
  %476 = shufflevector <8 x half> %445, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %477 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed45, <4 x half> %476, <4 x float> %475) #11, !dbg !16
  %478 = shufflevector <8 x half> %447, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %479 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %478, <4 x float> %477) #11, !dbg !16
  %480 = shufflevector <8 x half> %449, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %481 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %480, <4 x float> %479) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %482 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local230, i32 0) #0, !dbg !16
  %483 = bitcast <4 x i32> %482 to <8 x half>, !dbg !16
  %484 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local234, i32 0) #0, !dbg !16
  %485 = bitcast <4 x i32> %484 to <8 x half>, !dbg !16
  %486 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local238, i32 0) #0, !dbg !16
  %487 = bitcast <4 x i32> %486 to <8 x half>, !dbg !16
  %488 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local242, i32 0) #0, !dbg !16
  %489 = bitcast <4 x i32> %488 to <8 x half>, !dbg !16
  %490 = shufflevector <8 x half> %483, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %491 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %490, <4 x float> %37) #11, !dbg !16
  %492 = shufflevector <8 x half> %485, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %493 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed45, <4 x half> %492, <4 x float> %491) #11, !dbg !16
  %494 = shufflevector <8 x half> %487, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %495 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %494, <4 x float> %493) #11, !dbg !16
  %496 = shufflevector <8 x half> %489, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %497 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %496, <4 x float> %495) #11, !dbg !16
  %498 = shufflevector <8 x half> %483, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %499 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %498, <4 x float> %38) #11, !dbg !16
  %500 = shufflevector <8 x half> %485, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %501 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed45, <4 x half> %500, <4 x float> %499) #11, !dbg !16
  %502 = shufflevector <8 x half> %487, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %503 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed47, <4 x half> %502, <4 x float> %501) #11, !dbg !16
  %504 = shufflevector <8 x half> %489, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %505 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed49, <4 x half> %504, <4 x float> %503) #11, !dbg !16
  %506 = shufflevector <4 x float> %393, <4 x float> %401, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %507 = shufflevector <4 x float> %433, <4 x float> %441, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %508 = shufflevector <4 x float> %473, <4 x float> %481, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %509 = shufflevector <4 x float> %497, <4 x float> %505, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %510 = add nuw nsw i64 %26, 1, !dbg !8
  br label %25, !dbg !8

511:                                              ; preds = %25
  %512 = and i32 %19, 16, !dbg !8
  %.not2 = icmp eq i32 %512, 0, !dbg !8
  %513 = and i32 %19, 32, !dbg !8
  %.not3 = icmp eq i32 %513, 0, !dbg !8
  %514 = extractelement <8 x float> %39, i64 0, !dbg !8
  %515 = extractelement <8 x float> %39, i64 1, !dbg !8
  %516 = extractelement <8 x float> %39, i64 2, !dbg !8
  %517 = extractelement <8 x float> %39, i64 3, !dbg !8
  %518 = select i1 %.not2, float %515, float %514, !dbg !8
  %519 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %520 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %519), !dbg !8
  %521 = add nuw nsw i32 %520, 64, !dbg !8
  %522 = xor i32 %520, 16, !dbg !8
  %523 = and i32 %521, 192, !dbg !8
  %524 = icmp samesign ult i32 %522, %523, !dbg !8
  %525 = select i1 %524, i32 %522, i32 %520, !dbg !8
  %526 = shl nuw nsw i32 %525, 2, !dbg !8
  %527 = bitcast float %518 to i32, !dbg !8
  %528 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %527), !dbg !8
  %529 = bitcast i32 %528 to float, !dbg !8
  %530 = select i1 %.not2, float %514, float %529, !dbg !8
  %531 = select i1 %.not2, float %529, float %515, !dbg !8
  %532 = select i1 %.not2, float %517, float %516, !dbg !8
  %533 = bitcast float %532 to i32, !dbg !8
  %534 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %533), !dbg !8
  %535 = bitcast i32 %534 to float, !dbg !8
  %536 = select i1 %.not2, float %516, float %535, !dbg !8
  %537 = select i1 %.not2, float %535, float %517, !dbg !8
  %538 = select i1 %.not3, float %536, float %530, !dbg !8
  %539 = xor i32 %520, 32, !dbg !8
  %540 = icmp samesign ult i32 %539, %523, !dbg !8
  %541 = select i1 %540, i32 %539, i32 %520, !dbg !8
  %542 = shl nuw nsw i32 %541, 2, !dbg !8
  %543 = bitcast float %538 to i32, !dbg !8
  %544 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %543), !dbg !8
  %545 = bitcast i32 %544 to float, !dbg !8
  %546 = select i1 %.not3, float %530, float %545, !dbg !8
  %547 = select i1 %.not3, float %545, float %536, !dbg !8
  %548 = select i1 %.not3, float %537, float %531, !dbg !8
  %549 = bitcast float %548 to i32, !dbg !8
  %550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %549), !dbg !8
  %551 = bitcast i32 %550 to float, !dbg !8
  %552 = select i1 %.not3, float %531, float %551, !dbg !8
  %553 = select i1 %.not3, float %551, float %537, !dbg !8
  %554 = extractelement <8 x float> %36, i64 0, !dbg !8
  %555 = extractelement <8 x float> %36, i64 1, !dbg !8
  %556 = extractelement <8 x float> %36, i64 2, !dbg !8
  %557 = extractelement <8 x float> %36, i64 3, !dbg !8
  %558 = select i1 %.not2, float %555, float %554, !dbg !8
  %559 = bitcast float %558 to i32, !dbg !8
  %560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %559), !dbg !8
  %561 = bitcast i32 %560 to float, !dbg !8
  %562 = select i1 %.not2, float %554, float %561, !dbg !8
  %563 = select i1 %.not2, float %561, float %555, !dbg !8
  %564 = select i1 %.not2, float %557, float %556, !dbg !8
  %565 = bitcast float %564 to i32, !dbg !8
  %566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %565), !dbg !8
  %567 = bitcast i32 %566 to float, !dbg !8
  %568 = select i1 %.not2, float %556, float %567, !dbg !8
  %569 = select i1 %.not2, float %567, float %557, !dbg !8
  %570 = select i1 %.not3, float %568, float %562, !dbg !8
  %571 = bitcast float %570 to i32, !dbg !8
  %572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %571), !dbg !8
  %573 = bitcast i32 %572 to float, !dbg !8
  %574 = select i1 %.not3, float %562, float %573, !dbg !8
  %575 = select i1 %.not3, float %573, float %568, !dbg !8
  %576 = select i1 %.not3, float %569, float %563, !dbg !8
  %577 = bitcast float %576 to i32, !dbg !8
  %578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %577), !dbg !8
  %579 = bitcast i32 %578 to float, !dbg !8
  %580 = select i1 %.not3, float %563, float %579, !dbg !8
  %581 = select i1 %.not3, float %579, float %569, !dbg !8
  %582 = extractelement <8 x float> %33, i64 0, !dbg !8
  %583 = extractelement <8 x float> %33, i64 1, !dbg !8
  %584 = extractelement <8 x float> %33, i64 2, !dbg !8
  %585 = extractelement <8 x float> %33, i64 3, !dbg !8
  %586 = select i1 %.not2, float %583, float %582, !dbg !8
  %587 = bitcast float %586 to i32, !dbg !8
  %588 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %587), !dbg !8
  %589 = bitcast i32 %588 to float, !dbg !8
  %590 = select i1 %.not2, float %582, float %589, !dbg !8
  %591 = select i1 %.not2, float %589, float %583, !dbg !8
  %592 = select i1 %.not2, float %585, float %584, !dbg !8
  %593 = bitcast float %592 to i32, !dbg !8
  %594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %593), !dbg !8
  %595 = bitcast i32 %594 to float, !dbg !8
  %596 = select i1 %.not2, float %584, float %595, !dbg !8
  %597 = select i1 %.not2, float %595, float %585, !dbg !8
  %598 = select i1 %.not3, float %596, float %590, !dbg !8
  %599 = bitcast float %598 to i32, !dbg !8
  %600 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %599), !dbg !8
  %601 = bitcast i32 %600 to float, !dbg !8
  %602 = select i1 %.not3, float %590, float %601, !dbg !8
  %603 = select i1 %.not3, float %601, float %596, !dbg !8
  %604 = select i1 %.not3, float %597, float %591, !dbg !8
  %605 = bitcast float %604 to i32, !dbg !8
  %606 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %605), !dbg !8
  %607 = bitcast i32 %606 to float, !dbg !8
  %608 = select i1 %.not3, float %591, float %607, !dbg !8
  %609 = select i1 %.not3, float %607, float %597, !dbg !8
  %610 = extractelement <8 x float> %29, i64 0, !dbg !8
  %611 = extractelement <8 x float> %29, i64 1, !dbg !8
  %612 = extractelement <8 x float> %29, i64 2, !dbg !8
  %613 = extractelement <8 x float> %29, i64 3, !dbg !8
  %614 = select i1 %.not2, float %611, float %610, !dbg !8
  %615 = bitcast float %614 to i32, !dbg !8
  %616 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %615), !dbg !8
  %617 = bitcast i32 %616 to float, !dbg !8
  %618 = select i1 %.not2, float %610, float %617, !dbg !8
  %619 = select i1 %.not2, float %617, float %611, !dbg !8
  %620 = select i1 %.not2, float %613, float %612, !dbg !8
  %621 = bitcast float %620 to i32, !dbg !8
  %622 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %621), !dbg !8
  %623 = bitcast i32 %622 to float, !dbg !8
  %624 = select i1 %.not2, float %612, float %623, !dbg !8
  %625 = select i1 %.not2, float %623, float %613, !dbg !8
  %626 = select i1 %.not3, float %624, float %618, !dbg !8
  %627 = bitcast float %626 to i32, !dbg !8
  %628 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %627), !dbg !8
  %629 = select i1 %.not3, float %625, float %619, !dbg !8
  %630 = bitcast float %629 to i32, !dbg !8
  %631 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %630), !dbg !8
  %632 = shufflevector <1 x float> %30, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %633 = extractelement <8 x float> %39, i64 4, !dbg !8
  %634 = extractelement <8 x float> %39, i64 5, !dbg !8
  %635 = extractelement <8 x float> %39, i64 6, !dbg !8
  %636 = extractelement <8 x float> %39, i64 7, !dbg !8
  %637 = select i1 %.not2, float %634, float %633, !dbg !8
  %638 = bitcast float %637 to i32, !dbg !8
  %639 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %638), !dbg !8
  %640 = bitcast i32 %639 to float, !dbg !8
  %641 = select i1 %.not2, float %633, float %640, !dbg !8
  %642 = select i1 %.not2, float %640, float %634, !dbg !8
  %643 = select i1 %.not2, float %636, float %635, !dbg !8
  %644 = bitcast float %643 to i32, !dbg !8
  %645 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %644), !dbg !8
  %646 = bitcast i32 %645 to float, !dbg !8
  %647 = select i1 %.not2, float %635, float %646, !dbg !8
  %648 = select i1 %.not2, float %646, float %636, !dbg !8
  %649 = select i1 %.not3, float %647, float %641, !dbg !8
  %650 = bitcast float %649 to i32, !dbg !8
  %651 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %650), !dbg !8
  %652 = bitcast i32 %651 to float, !dbg !8
  %653 = select i1 %.not3, float %641, float %652, !dbg !8
  %654 = select i1 %.not3, float %652, float %647, !dbg !8
  %655 = select i1 %.not3, float %648, float %642, !dbg !8
  %656 = bitcast float %655 to i32, !dbg !8
  %657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %656), !dbg !8
  %658 = bitcast i32 %657 to float, !dbg !8
  %659 = select i1 %.not3, float %642, float %658, !dbg !8
  %660 = select i1 %.not3, float %658, float %648, !dbg !8
  %661 = extractelement <8 x float> %36, i64 4, !dbg !8
  %662 = extractelement <8 x float> %36, i64 5, !dbg !8
  %663 = extractelement <8 x float> %36, i64 6, !dbg !8
  %664 = extractelement <8 x float> %36, i64 7, !dbg !8
  %665 = select i1 %.not2, float %662, float %661, !dbg !8
  %666 = bitcast float %665 to i32, !dbg !8
  %667 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %666), !dbg !8
  %668 = bitcast i32 %667 to float, !dbg !8
  %669 = select i1 %.not2, float %661, float %668, !dbg !8
  %670 = select i1 %.not2, float %668, float %662, !dbg !8
  %671 = select i1 %.not2, float %664, float %663, !dbg !8
  %672 = bitcast float %671 to i32, !dbg !8
  %673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %672), !dbg !8
  %674 = bitcast i32 %673 to float, !dbg !8
  %675 = select i1 %.not2, float %663, float %674, !dbg !8
  %676 = select i1 %.not2, float %674, float %664, !dbg !8
  %677 = select i1 %.not3, float %675, float %669, !dbg !8
  %678 = bitcast float %677 to i32, !dbg !8
  %679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %678), !dbg !8
  %680 = bitcast i32 %679 to float, !dbg !8
  %681 = select i1 %.not3, float %669, float %680, !dbg !8
  %682 = select i1 %.not3, float %680, float %675, !dbg !8
  %683 = select i1 %.not3, float %676, float %670, !dbg !8
  %684 = bitcast float %683 to i32, !dbg !8
  %685 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %684), !dbg !8
  %686 = bitcast i32 %685 to float, !dbg !8
  %687 = select i1 %.not3, float %670, float %686, !dbg !8
  %688 = select i1 %.not3, float %686, float %676, !dbg !8
  %689 = extractelement <8 x float> %33, i64 4, !dbg !8
  %690 = extractelement <8 x float> %33, i64 5, !dbg !8
  %691 = extractelement <8 x float> %33, i64 6, !dbg !8
  %692 = extractelement <8 x float> %33, i64 7, !dbg !8
  %693 = select i1 %.not2, float %690, float %689, !dbg !8
  %694 = bitcast float %693 to i32, !dbg !8
  %695 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %694), !dbg !8
  %696 = bitcast i32 %695 to float, !dbg !8
  %697 = select i1 %.not2, float %689, float %696, !dbg !8
  %698 = select i1 %.not2, float %696, float %690, !dbg !8
  %699 = select i1 %.not2, float %692, float %691, !dbg !8
  %700 = bitcast float %699 to i32, !dbg !8
  %701 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %700), !dbg !8
  %702 = bitcast i32 %701 to float, !dbg !8
  %703 = select i1 %.not2, float %691, float %702, !dbg !8
  %704 = select i1 %.not2, float %702, float %692, !dbg !8
  %705 = select i1 %.not3, float %703, float %697, !dbg !8
  %706 = bitcast float %705 to i32, !dbg !8
  %707 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %706), !dbg !8
  %708 = bitcast i32 %707 to float, !dbg !8
  %709 = select i1 %.not3, float %697, float %708, !dbg !8
  %710 = select i1 %.not3, float %708, float %703, !dbg !8
  %711 = select i1 %.not3, float %704, float %698, !dbg !8
  %712 = bitcast float %711 to i32, !dbg !8
  %713 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %712), !dbg !8
  %714 = bitcast i32 %713 to float, !dbg !8
  %715 = select i1 %.not3, float %698, float %714, !dbg !8
  %716 = select i1 %.not3, float %714, float %704, !dbg !8
  %717 = extractelement <8 x float> %29, i64 4, !dbg !8
  %718 = extractelement <8 x float> %29, i64 5, !dbg !8
  %719 = extractelement <8 x float> %29, i64 6, !dbg !8
  %720 = extractelement <8 x float> %29, i64 7, !dbg !8
  %721 = select i1 %.not2, float %718, float %717, !dbg !8
  %722 = bitcast float %721 to i32, !dbg !8
  %723 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %722), !dbg !8
  %724 = bitcast i32 %723 to float, !dbg !8
  %725 = select i1 %.not2, float %717, float %724, !dbg !8
  %726 = select i1 %.not2, float %724, float %718, !dbg !8
  %727 = select i1 %.not2, float %720, float %719, !dbg !8
  %728 = bitcast float %727 to i32, !dbg !8
  %729 = call i32 @llvm.amdgcn.ds.bpermute(i32 %526, i32 %728), !dbg !8
  %730 = bitcast i32 %729 to float, !dbg !8
  %731 = select i1 %.not2, float %719, float %730, !dbg !8
  %732 = select i1 %.not2, float %730, float %720, !dbg !8
  %733 = select i1 %.not3, float %731, float %725, !dbg !8
  %734 = bitcast float %733 to i32, !dbg !8
  %735 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %734), !dbg !8
  %736 = select i1 %.not3, float %732, float %726, !dbg !8
  %737 = bitcast float %736 to i32, !dbg !8
  %738 = call i32 @llvm.amdgcn.ds.bpermute(i32 %542, i32 %737), !dbg !8
  br i1 true, label %739, label %773, !dbg !21

739:                                              ; preds = %511
  %740 = bitcast i32 %735 to float, !dbg !8
  %741 = select i1 %.not3, float %725, float %740, !dbg !8
  %742 = insertelement <4 x float> poison, float %741, i64 0, !dbg !19
  %743 = bitcast i32 %738 to float, !dbg !8
  %744 = select i1 %.not3, float %726, float %743, !dbg !8
  %745 = insertelement <4 x float> %742, float %744, i64 1, !dbg !19
  %746 = select i1 %.not3, float %740, float %731, !dbg !8
  %747 = insertelement <4 x float> %745, float %746, i64 2, !dbg !19
  %748 = select i1 %.not3, float %743, float %732, !dbg !8
  %749 = insertelement <4 x float> %747, float %748, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %632, !dbg !19
  %normalized = fmul <4 x float> %749, %row.reciprocal, !dbg !19
  %750 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %751 = extractelement <4 x half> %750, i64 3, !dbg !20
  %752 = extractelement <4 x half> %750, i64 2, !dbg !20
  %753 = extractelement <4 x half> %750, i64 1, !dbg !20
  %754 = extractelement <4 x half> %750, i64 0, !dbg !20
  %755 = bitcast i32 %628 to float, !dbg !8
  %756 = select i1 %.not3, float %618, float %755, !dbg !8
  %757 = insertelement <4 x float> poison, float %756, i64 0, !dbg !19
  %758 = bitcast i32 %631 to float, !dbg !8
  %759 = select i1 %.not3, float %619, float %758, !dbg !8
  %760 = insertelement <4 x float> %757, float %759, i64 1, !dbg !19
  %761 = select i1 %.not3, float %755, float %624, !dbg !8
  %762 = insertelement <4 x float> %760, float %761, i64 2, !dbg !19
  %763 = select i1 %.not3, float %758, float %625, !dbg !8
  %764 = insertelement <4 x float> %762, float %763, i64 3, !dbg !19
  %normalized24 = fmul <4 x float> %764, %row.reciprocal, !dbg !19
  %765 = fptrunc <4 x float> %normalized24 to <4 x half>, !dbg !20
  %766 = extractelement <4 x half> %765, i64 3, !dbg !20
  %767 = extractelement <4 x half> %765, i64 2, !dbg !20
  %768 = extractelement <4 x half> %765, i64 1, !dbg !20
  %769 = extractelement <4 x half> %765, i64 0, !dbg !20
  %770 = add nsw i64 %8, %13, !dbg !21
  %771 = shl nsw i64 %770, 19, !dbg !21
  %772 = and i64 %771, 16252928, !dbg !21
  %address.thread256 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local257 = zext nneg i32 %address.thread256 to i64, !dbg !14
  %address.local258 = add nsw i64 %16, %address.local257, !dbg !21
  %address.local259 = lshr i64 %address.local257, 6, !dbg !14
  %address.local260 = shl nuw nsw i64 %address.local259, 4, !dbg !21
  %address.local261 = add nsw i64 %address.local258, %address.local260, !dbg !21
  %address.local262 = and i64 %address.local257, 240, !dbg !21
  %address.local263 = sub nsw i64 %address.local261, %address.local262, !dbg !21
  %address.local264 = add nsw i64 %address.local263, %17, !dbg !21
  %address.local265 = shl nuw nsw i64 %address.local264, 7, !dbg !21
  %address.local266 = add nuw nsw i64 %772, %address.local265, !dbg !21
  %address.local267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local268 = lshr i64 %address.local257, 2, !dbg !14
  %address.local269 = and i64 %address.local268, 12, !dbg !14
  %address.local270 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local267, i64 %address.local269, !dbg !21
  store half %769, ptr addrspace(1) %address.local270, align 2, !dbg !21
  %address.local271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local271, i64 %address.local269, !dbg !21
  %address.local273 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local272, i64 2, !dbg !21
  store half %768, ptr addrspace(1) %address.local273, align 2, !dbg !21
  %address.local274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local274, i64 %address.local269, !dbg !21
  %address.local276 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local275, i64 4, !dbg !21
  store half %767, ptr addrspace(1) %address.local276, align 2, !dbg !21
  %address.local277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local277, i64 %address.local269, !dbg !21
  %address.local279 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local278, i64 6, !dbg !21
  store half %766, ptr addrspace(1) %address.local279, align 2, !dbg !21
  %address.local280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local280, i64 %address.local269, !dbg !21
  %address.local282 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local281, i64 32, !dbg !21
  store half %754, ptr addrspace(1) %address.local282, align 2, !dbg !21
  %address.local283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local283, i64 %address.local269, !dbg !21
  %address.local285 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local284, i64 34, !dbg !21
  store half %753, ptr addrspace(1) %address.local285, align 2, !dbg !21
  %address.local286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local286, i64 %address.local269, !dbg !21
  %address.local288 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local287, i64 36, !dbg !21
  store half %752, ptr addrspace(1) %address.local288, align 2, !dbg !21
  %address.local289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local266, !dbg !21
  %address.local290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local289, i64 %address.local269, !dbg !21
  %address.local291 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local290, i64 38, !dbg !21
  store half %751, ptr addrspace(1) %address.local291, align 2, !dbg !21
  br label %773, !dbg !21

773:                                              ; preds = %739, %511
  br i1 true, label %774, label %794, !dbg !21

774:                                              ; preds = %773
  %775 = insertelement <4 x float> poison, float %709, i64 0, !dbg !19
  %776 = insertelement <4 x float> %775, float %715, i64 1, !dbg !19
  %777 = insertelement <4 x float> %776, float %710, i64 2, !dbg !19
  %778 = insertelement <4 x float> %777, float %716, i64 3, !dbg !19
  %row.reciprocal25 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %632, !dbg !19
  %normalized26 = fmul <4 x float> %778, %row.reciprocal25, !dbg !19
  %779 = fptrunc <4 x float> %normalized26 to <4 x half>, !dbg !20
  %780 = extractelement <4 x half> %779, i64 3, !dbg !20
  %781 = extractelement <4 x half> %779, i64 2, !dbg !20
  %782 = extractelement <4 x half> %779, i64 1, !dbg !20
  %783 = extractelement <4 x half> %779, i64 0, !dbg !20
  %784 = insertelement <4 x float> poison, float %602, i64 0, !dbg !19
  %785 = insertelement <4 x float> %784, float %608, i64 1, !dbg !19
  %786 = insertelement <4 x float> %785, float %603, i64 2, !dbg !19
  %787 = insertelement <4 x float> %786, float %609, i64 3, !dbg !19
  %normalized27 = fmul <4 x float> %787, %row.reciprocal25, !dbg !19
  %788 = fptrunc <4 x float> %normalized27 to <4 x half>, !dbg !20
  %789 = extractelement <4 x half> %788, i64 3, !dbg !20
  %790 = extractelement <4 x half> %788, i64 2, !dbg !20
  %791 = extractelement <4 x half> %788, i64 1, !dbg !20
  %792 = extractelement <4 x half> %788, i64 0, !dbg !20
  %793 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread292 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local293 = zext nneg i32 %address.thread292 to i64, !dbg !14
  %address.local294 = add nsw i64 %16, %address.local293, !dbg !21
  %address.local295 = lshr i64 %address.local293, 6, !dbg !14
  %address.local296 = shl nuw nsw i64 %address.local295, 4, !dbg !21
  %address.local297 = add nsw i64 %address.local294, %address.local296, !dbg !21
  %address.local298 = and i64 %address.local293, 240, !dbg !21
  %address.local299 = sub nsw i64 %address.local297, %address.local298, !dbg !21
  %address.local300 = add nsw i64 %address.local299, %17, !dbg !21
  %address.local301 = shl nuw nsw i64 %address.local300, 7, !dbg !21
  %address.local302 = add nuw nsw i64 %793, %address.local301, !dbg !21
  %address.local303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !21
  %address.local304 = lshr i64 %address.local293, 2, !dbg !14
  %address.local305 = and i64 %address.local304, 12, !dbg !14
  %address.local306 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local303, i64 %address.local305, !dbg !21
  %address.local307 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local306, i64 64, !dbg !21
  store half %792, ptr addrspace(1) %address.local307, align 2, !dbg !21
  %address.local308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !21
  %address.local309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local308, i64 %address.local305, !dbg !21
  %address.local310 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local309, i64 66, !dbg !21
  store half %791, ptr addrspace(1) %address.local310, align 2, !dbg !21
  %address.local311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !21
  %address.local312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local311, i64 %address.local305, !dbg !21
  %address.local313 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local312, i64 68, !dbg !21
  store half %790, ptr addrspace(1) %address.local313, align 2, !dbg !21
  %address.local314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !21
  %address.local315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local314, i64 %address.local305, !dbg !21
  %address.local316 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local315, i64 70, !dbg !21
  store half %789, ptr addrspace(1) %address.local316, align 2, !dbg !21
  %address.local317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !21
  %address.local318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local317, i64 %address.local305, !dbg !21
  %address.local319 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local318, i64 96, !dbg !21
  store half %783, ptr addrspace(1) %address.local319, align 2, !dbg !21
  %address.local320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !21
  %address.local321 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local320, i64 %address.local305, !dbg !21
  %address.local322 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local321, i64 98, !dbg !21
  store half %782, ptr addrspace(1) %address.local322, align 2, !dbg !21
  %address.local323 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !21
  %address.local324 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local323, i64 %address.local305, !dbg !21
  %address.local325 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local324, i64 100, !dbg !21
  store half %781, ptr addrspace(1) %address.local325, align 2, !dbg !21
  %address.local326 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !21
  %address.local327 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local326, i64 %address.local305, !dbg !21
  %address.local328 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local327, i64 102, !dbg !21
  store half %780, ptr addrspace(1) %address.local328, align 2, !dbg !21
  br label %794, !dbg !21

794:                                              ; preds = %774, %773
  br i1 true, label %795, label %815, !dbg !21

795:                                              ; preds = %794
  %796 = insertelement <4 x float> poison, float %681, i64 0, !dbg !19
  %797 = insertelement <4 x float> %796, float %687, i64 1, !dbg !19
  %798 = insertelement <4 x float> %797, float %682, i64 2, !dbg !19
  %799 = insertelement <4 x float> %798, float %688, i64 3, !dbg !19
  %row.reciprocal28 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %632, !dbg !19
  %normalized29 = fmul <4 x float> %799, %row.reciprocal28, !dbg !19
  %800 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !20
  %801 = extractelement <4 x half> %800, i64 3, !dbg !20
  %802 = extractelement <4 x half> %800, i64 2, !dbg !20
  %803 = extractelement <4 x half> %800, i64 1, !dbg !20
  %804 = extractelement <4 x half> %800, i64 0, !dbg !20
  %805 = insertelement <4 x float> poison, float %574, i64 0, !dbg !19
  %806 = insertelement <4 x float> %805, float %580, i64 1, !dbg !19
  %807 = insertelement <4 x float> %806, float %575, i64 2, !dbg !19
  %808 = insertelement <4 x float> %807, float %581, i64 3, !dbg !19
  %normalized30 = fmul <4 x float> %808, %row.reciprocal28, !dbg !19
  %809 = fptrunc <4 x float> %normalized30 to <4 x half>, !dbg !20
  %810 = extractelement <4 x half> %809, i64 3, !dbg !20
  %811 = extractelement <4 x half> %809, i64 2, !dbg !20
  %812 = extractelement <4 x half> %809, i64 1, !dbg !20
  %813 = extractelement <4 x half> %809, i64 0, !dbg !20
  %814 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread329 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local330 = zext nneg i32 %address.thread329 to i64, !dbg !14
  %address.local331 = add nsw i64 %16, %address.local330, !dbg !21
  %address.local332 = lshr i64 %address.local330, 6, !dbg !14
  %address.local333 = shl nuw nsw i64 %address.local332, 4, !dbg !21
  %address.local334 = add nsw i64 %address.local331, %address.local333, !dbg !21
  %address.local335 = and i64 %address.local330, 240, !dbg !21
  %address.local336 = sub nsw i64 %address.local334, %address.local335, !dbg !21
  %address.local337 = add nsw i64 %address.local336, %17, !dbg !21
  %address.local338 = shl nuw nsw i64 %address.local337, 7, !dbg !21
  %address.local339 = add nuw nsw i64 %814, %address.local338, !dbg !21
  %address.local340 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local339, !dbg !21
  %address.local341 = lshr i64 %address.local330, 2, !dbg !14
  %address.local342 = and i64 %address.local341, 12, !dbg !14
  %address.local343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local340, i64 %address.local342, !dbg !21
  %address.local344 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local343, i64 128, !dbg !21
  store half %813, ptr addrspace(1) %address.local344, align 2, !dbg !21
  %address.local345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local339, !dbg !21
  %address.local346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local345, i64 %address.local342, !dbg !21
  %address.local347 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local346, i64 130, !dbg !21
  store half %812, ptr addrspace(1) %address.local347, align 2, !dbg !21
  %address.local348 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local339, !dbg !21
  %address.local349 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local348, i64 %address.local342, !dbg !21
  %address.local350 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local349, i64 132, !dbg !21
  store half %811, ptr addrspace(1) %address.local350, align 2, !dbg !21
  %address.local351 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local339, !dbg !21
  %address.local352 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local351, i64 %address.local342, !dbg !21
  %address.local353 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local352, i64 134, !dbg !21
  store half %810, ptr addrspace(1) %address.local353, align 2, !dbg !21
  %address.local354 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local339, !dbg !21
  %address.local355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local354, i64 %address.local342, !dbg !21
  %address.local356 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local355, i64 160, !dbg !21
  store half %804, ptr addrspace(1) %address.local356, align 2, !dbg !21
  %address.local357 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local339, !dbg !21
  %address.local358 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local357, i64 %address.local342, !dbg !21
  %address.local359 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local358, i64 162, !dbg !21
  store half %803, ptr addrspace(1) %address.local359, align 2, !dbg !21
  %address.local360 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local339, !dbg !21
  %address.local361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local360, i64 %address.local342, !dbg !21
  %address.local362 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local361, i64 164, !dbg !21
  store half %802, ptr addrspace(1) %address.local362, align 2, !dbg !21
  %address.local363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local339, !dbg !21
  %address.local364 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local363, i64 %address.local342, !dbg !21
  %address.local365 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local364, i64 166, !dbg !21
  store half %801, ptr addrspace(1) %address.local365, align 2, !dbg !21
  br label %815, !dbg !21

815:                                              ; preds = %795, %794
  br i1 true, label %816, label %836, !dbg !21

816:                                              ; preds = %815
  %817 = insertelement <4 x float> poison, float %653, i64 0, !dbg !19
  %818 = insertelement <4 x float> %817, float %659, i64 1, !dbg !19
  %819 = insertelement <4 x float> %818, float %654, i64 2, !dbg !19
  %820 = insertelement <4 x float> %819, float %660, i64 3, !dbg !19
  %row.reciprocal31 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %632, !dbg !19
  %normalized32 = fmul <4 x float> %820, %row.reciprocal31, !dbg !19
  %821 = fptrunc <4 x float> %normalized32 to <4 x half>, !dbg !20
  %822 = extractelement <4 x half> %821, i64 3, !dbg !20
  %823 = extractelement <4 x half> %821, i64 2, !dbg !20
  %824 = extractelement <4 x half> %821, i64 1, !dbg !20
  %825 = extractelement <4 x half> %821, i64 0, !dbg !20
  %826 = insertelement <4 x float> poison, float %546, i64 0, !dbg !19
  %827 = insertelement <4 x float> %826, float %552, i64 1, !dbg !19
  %828 = insertelement <4 x float> %827, float %547, i64 2, !dbg !19
  %829 = insertelement <4 x float> %828, float %553, i64 3, !dbg !19
  %normalized33 = fmul <4 x float> %829, %row.reciprocal31, !dbg !19
  %830 = fptrunc <4 x float> %normalized33 to <4 x half>, !dbg !20
  %831 = extractelement <4 x half> %830, i64 3, !dbg !20
  %832 = extractelement <4 x half> %830, i64 2, !dbg !20
  %833 = extractelement <4 x half> %830, i64 1, !dbg !20
  %834 = extractelement <4 x half> %830, i64 0, !dbg !20
  %835 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread366 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local367 = zext nneg i32 %address.thread366 to i64, !dbg !14
  %address.local368 = add nsw i64 %16, %address.local367, !dbg !21
  %address.local369 = lshr i64 %address.local367, 6, !dbg !14
  %address.local370 = shl nuw nsw i64 %address.local369, 4, !dbg !21
  %address.local371 = add nsw i64 %address.local368, %address.local370, !dbg !21
  %address.local372 = and i64 %address.local367, 240, !dbg !21
  %address.local373 = sub nsw i64 %address.local371, %address.local372, !dbg !21
  %address.local374 = add nsw i64 %address.local373, %17, !dbg !21
  %address.local375 = shl nuw nsw i64 %address.local374, 7, !dbg !21
  %address.local376 = add nuw nsw i64 %835, %address.local375, !dbg !21
  %address.local377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local376, !dbg !21
  %address.local378 = lshr i64 %address.local367, 2, !dbg !14
  %address.local379 = and i64 %address.local378, 12, !dbg !14
  %address.local380 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local377, i64 %address.local379, !dbg !21
  %address.local381 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local380, i64 192, !dbg !21
  store half %834, ptr addrspace(1) %address.local381, align 2, !dbg !21
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local376, !dbg !21
  %address.local383 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local382, i64 %address.local379, !dbg !21
  %address.local384 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local383, i64 194, !dbg !21
  store half %833, ptr addrspace(1) %address.local384, align 2, !dbg !21
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local376, !dbg !21
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local385, i64 %address.local379, !dbg !21
  %address.local387 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local386, i64 196, !dbg !21
  store half %832, ptr addrspace(1) %address.local387, align 2, !dbg !21
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local376, !dbg !21
  %address.local389 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local388, i64 %address.local379, !dbg !21
  %address.local390 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local389, i64 198, !dbg !21
  store half %831, ptr addrspace(1) %address.local390, align 2, !dbg !21
  %address.local391 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local376, !dbg !21
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local391, i64 %address.local379, !dbg !21
  %address.local393 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local392, i64 224, !dbg !21
  store half %825, ptr addrspace(1) %address.local393, align 2, !dbg !21
  %address.local394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local376, !dbg !21
  %address.local395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local394, i64 %address.local379, !dbg !21
  %address.local396 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local395, i64 226, !dbg !21
  store half %824, ptr addrspace(1) %address.local396, align 2, !dbg !21
  %address.local397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local376, !dbg !21
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local397, i64 %address.local379, !dbg !21
  %address.local399 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local398, i64 228, !dbg !21
  store half %823, ptr addrspace(1) %address.local399, align 2, !dbg !21
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local376, !dbg !21
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local379, !dbg !21
  %address.local402 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local401, i64 230, !dbg !21
  store half %822, ptr addrspace(1) %address.local402, align 2, !dbg !21
  br label %836, !dbg !21

836:                                              ; preds = %816, %815
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

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { convergent nounwind memory(none) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!33}

!0 = !DIFile(filename: "bm64_bn64.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!8 = !DILocation(line: 56, column: 24, scope: !4)
!9 = !DILocation(line: 69, column: 26, scope: !4)
!11 = !DILocation(line: 46, column: 17, scope: !4)
!13 = !DILocation(line: 66, column: 22, scope: !4)
!14 = !DILocation(line: 68, column: 24, scope: !4)
!15 = !DILocation(line: 73, column: 19, scope: !4)
!16 = !DILocation(line: 93, column: 23, scope: !4)
!17 = !DILocation(line: 67, column: 22, scope: !4)
!18 = !DILocation(line: 92, column: 20, scope: !4)
!19 = !DILocation(line: 97, column: 23, scope: !4)
!20 = !DILocation(line: 98, column: 19, scope: !4)
!21 = !DILocation(line: 99, column: 9, scope: !4)
!22 = !DILocation(line: 59, column: 17, scope: !4)
!23 = !DILocation(line: 75, column: 24, scope: !4)
!24 = !DILocation(line: 76, column: 23, scope: !4)
!25 = !DILocation(line: 77, column: 26, scope: !4)
!26 = !DILocation(line: 84, column: 21, scope: !4)
!27 = !DILocation(line: 85, column: 22, scope: !4)
!28 = !DILocation(line: 86, column: 20, scope: !4)
!29 = !DILocation(line: 87, column: 16, scope: !4)
!31 = !DILocation(line: 70, column: 23, scope: !4)
!32 = !{i32 256, i32 1, i32 1}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !{i32 0, i32 256}
