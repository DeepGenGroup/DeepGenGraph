; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/baseline/input.mlir"

@shm_0 = addrspace(3) global [6144 x i8] undef, align 16

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
  %24 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !39
  %25 = zext nneg i32 %24 to i64, !dbg !12
  %26 = lshr i64 %25, 6, !dbg !12
  %27 = lshr i64 %25, 2, !dbg !12
  %28 = and i64 %27, 12, !dbg !12
  %29 = and i32 %24, 16, !dbg !13
  %.not = icmp eq i32 %29, 0, !dbg !13
  %30 = and i32 %24, 32, !dbg !13
  %.not2 = icmp eq i32 %30, 0, !dbg !13
  %31 = or disjoint i64 %11, 63, !dbg !14
  %32 = add nsw i64 %11, 64, !dbg !14
  %33 = icmp uge i64 %32, %31, !dbg !14
  %34 = shl nsw i64 %10, 1, !dbg !7
  %35 = add nsw i64 %34, 2, !dbg !7
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

46:                                               ; preds = %736, %4
  %47 = phi i64 [ %766, %736 ], [ 0, %4 ], !dbg !7
  %48 = phi <4 x float> [ %647, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %49 = phi <4 x float> [ %652, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %50 = phi <8 x float> [ %761, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %51 = phi <1 x float> [ %762, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %52 = phi <1 x float> [ %549, %736 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %53 = phi <4 x float> [ %694, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %54 = phi <4 x float> [ %699, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %55 = phi <8 x float> [ %763, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %56 = phi <4 x float> [ %741, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %57 = phi <4 x float> [ %746, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %58 = phi <8 x float> [ %764, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %59 = phi <4 x float> [ %755, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi <4 x float> [ %760, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = phi <8 x float> [ %765, %736 ], [ zeroinitializer, %4 ], !dbg !7
  %62 = icmp slt i64 %47, %35, !dbg !7
  br i1 %62, label %63, label %767, !dbg !7

63:                                               ; preds = %46
  %64 = shl nsw i64 %47, 5, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !11, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local30 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local31 = and i64 %address.local30, 1536, !dbg !11
  %address.local32 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local31, !dbg !11
  %address.local33 = ptrtoint ptr addrspace(3) %address.local32 to i64, !dbg !11
  %address.local34 = trunc i64 %address.local33 to i32, !dbg !11
  %65 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local34), !dbg !11
  %66 = zext i32 %65 to i64, !dbg !11
  %67 = inttoptr i64 %66 to ptr addrspace(3), !dbg !11
  %address.local35 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local36 = and i64 %address.local35, 8064, !dbg !11
  %address.local37 = lshr i64 %address.local, 3, !dbg !11
  %address.local38 = and i64 %address.local37, 24, !dbg !11
  %address.local39 = or disjoint i64 %address.local36, %address.local38, !dbg !11
  %address.local40 = getelementptr [2 x i8], ptr addrspace(1) %36, i64 %address.local39, !dbg !11
  %68 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %69 = ptrtoint ptr addrspace(1) %address.local40 to i64, !dbg !11
  %70 = sub i64 %69, %68, !dbg !11
  %dma.buffer.offset451 = trunc i64 %70 to i32, !dbg !11
  %71 = trunc i64 %68 to i32, !dbg !11
  %72 = insertelement <4 x i32> poison, i32 %71, i64 0, !dbg !11
  %73 = lshr i64 %68, 32, !dbg !11
  %74 = trunc i64 %73 to i32, !dbg !11
  %75 = insertelement <4 x i32> %72, i32 %74, i64 1, !dbg !11
  %76 = insertelement <4 x i32> %75, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc452 = insertelement <4 x i32> %76, i32 131072, i64 3, !dbg !11
  %77 = ptrtoint ptr addrspace(3) %67 to i32, !dbg !11
  %dma.buffer.lds453 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %77), !dbg !11
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds453, i32 %dma.buffer.offset451, <4 x i32> %dma.buffer.rsrc452) #10, !dbg !11
  %79 = lshr i64 %47, 14, !dbg !17
  %80 = add nsw i64 %79, %8, !dbg !17
  %81 = lshr i64 %47, 7, !dbg !17
  %82 = and i64 %81, 127, !dbg !17
  %83 = shl i64 %80, 19, !dbg !17
  %84 = and i64 %83, 16252928, !dbg !17
  %85 = shl i64 %81, 19, !dbg !17
  %86 = shl i64 %47, 12, !dbg !17
  %87 = sub i64 %86, %85, !dbg !17
  %88 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %84, !dbg !17
  %89 = getelementptr [2 x i8], ptr addrspace(1) %88, i64 %82, !dbg !17
  %90 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %87, !dbg !17
  br label %91, !dbg !17

91:                                               ; preds = %94, %63
  %92 = phi i64 [ %120, %94 ], [ %25, %63 ], !dbg !17
  %93 = icmp samesign ult i64 %92, 128, !dbg !17
  br i1 %93, label %94, label %121, !dbg !17

94:                                               ; preds = %91
  %95 = lshr i64 %92, 2, !dbg !17
  %96 = and i64 %95, 24, !dbg !17
  %97 = shl nuw nsw i64 %92, 3, !dbg !17
  %98 = and i64 %97, 512, !dbg !17
  %99 = shl nuw nsw i64 %92, 7, !dbg !17
  %100 = and i64 %99, 3968, !dbg !17
  %101 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %96, !dbg !17
  %102 = getelementptr [2 x i8], ptr addrspace(1) %101, i64 %100, !dbg !17
  %103 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %98, !dbg !17
  %104 = ptrtoint ptr addrspace(3) %103 to i64, !dbg !17
  %105 = trunc i64 %104 to i32, !dbg !17
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %105), !dbg !17
  %107 = zext i32 %106 to i64, !dbg !17
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !17
  %109 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %110 = ptrtoint ptr addrspace(1) %102 to i64, !dbg !17
  %111 = sub i64 %110, %109, !dbg !17
  %dma.buffer.offset = trunc i64 %111 to i32, !dbg !17
  %112 = trunc i64 %109 to i32, !dbg !17
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !17
  %114 = lshr i64 %109, 32, !dbg !17
  %115 = trunc i64 %114 to i32, !dbg !17
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !17
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !17
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !17
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %120 = or disjoint i64 %25, 256, !dbg !17
  br label %91, !dbg !17

121:                                              ; preds = %91
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.thread41 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local42 = lshr i32 %address.thread41, 2, !dbg !12
  %address.local43 = and i32 %address.local42, 4, !dbg !12
  %address.local44 = shl nuw nsw i32 %address.thread41, 1, !dbg !12
  %address.local45 = and i32 %address.local44, 384, !dbg !12
  %address.local46 = or disjoint i32 %address.local43, %address.local45, !dbg !12
  %address.local47 = shl nuw nsw i32 %address.thread41, 3, !dbg !12
  %address.local48 = and i32 %address.local47, 120, !dbg !12
  %address.local49 = or disjoint i32 %address.local46, %address.local48, !dbg !12
  %address.local50 = shl nuw nsw i32 %address.thread41, 4, !dbg !12
  %address.local51 = and i32 %address.local50, 512, !dbg !12
  %address.local52 = or disjoint i32 %address.local49, %address.local51, !dbg !12
  %address.local53 = zext nneg i32 %address.local52 to i64, !dbg !12
  %address.local54 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local53, !dbg !12
  %122 = load <4 x half>, ptr addrspace(3) %address.local54, align 2, !dbg !12
  %address.local55 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local53, !dbg !12
  %address.local56 = getelementptr i8, ptr addrspace(3) %address.local55, i64 2048, !dbg !12
  %123 = load <4 x half>, ptr addrspace(3) %address.local56, align 2, !dbg !12
  %124 = fmul <4 x half> %122, splat (half 0xH3015), !dbg !12
  %125 = fmul <4 x half> %123, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local57 = lshr i32 %address.thread41, 2, !dbg !16
  %address.local58 = and i32 %address.local57, 4, !dbg !16
  %address.local59 = shl nuw nsw i32 %address.thread41, 3, !dbg !16
  %address.local60 = and i32 %address.local59, 376, !dbg !16
  %address.local61 = or disjoint i32 %address.local58, %address.local60, !dbg !16
  %address.local62 = zext nneg i32 %address.local61 to i64, !dbg !16
  %address.local63 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local62, !dbg !16
  %126 = load <4 x half>, ptr addrspace(3) %address.local63, align 2, !dbg !16
  %address.local64 = lshr i32 %address.thread41, 2, !dbg !16
  %address.local65 = and i32 %address.local64, 4, !dbg !16
  %address.local66 = shl nuw nsw i32 %address.thread41, 3, !dbg !16
  %address.local67 = and i32 %address.local66, 376, !dbg !16
  %address.local68 = or disjoint i32 %address.local65, %address.local67, !dbg !16
  %address.local69 = zext nneg i32 %address.local68 to i64, !dbg !16
  %address.local70 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local69, !dbg !16
  %address.local71 = getelementptr i8, ptr addrspace(3) %address.local70, i64 1024, !dbg !16
  %127 = load <4 x half>, ptr addrspace(3) %address.local71, align 2, !dbg !16
  %128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %126, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local72 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local62, !dbg !16
  %address.local73 = getelementptr i8, ptr addrspace(3) %address.local72, i64 256, !dbg !16
  %129 = load <4 x half>, ptr addrspace(3) %address.local73, align 2, !dbg !16
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %127, <4 x float> %128) #9, !dbg !16
  %address.local74 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local69, !dbg !16
  %address.local75 = getelementptr i8, ptr addrspace(3) %address.local74, i64 1280, !dbg !16
  %131 = load <4 x half>, ptr addrspace(3) %address.local75, align 2, !dbg !16
  %132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %129, <4 x float> zeroinitializer) #9, !dbg !16
  %133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %131, <4 x float> %132) #9, !dbg !16
  %address.local76 = zext nneg i32 %address.thread41 to i64, !dbg !12
  %address.local77 = shl nuw nsw i64 %address.local76, 3, !dbg !11
  %address.local78 = and i64 %address.local77, 1536, !dbg !11
  %address.local79 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local78, !dbg !11
  %address.local80 = ptrtoint ptr addrspace(3) %address.local79 to i64, !dbg !11
  %address.local81 = trunc i64 %address.local80 to i32, !dbg !11
  %134 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local81), !dbg !11
  %135 = zext i32 %134 to i64, !dbg !11
  %136 = inttoptr i64 %135 to ptr addrspace(3), !dbg !11
  %address.local82 = shl nuw nsw i64 %address.local76, 7, !dbg !11
  %address.local83 = and i64 %address.local82, 8064, !dbg !11
  %address.local84 = lshr i64 %address.local76, 3, !dbg !11
  %address.local85 = and i64 %address.local84, 24, !dbg !11
  %address.local86 = or disjoint i64 %address.local83, %address.local85, !dbg !11
  %address.local87 = getelementptr [2 x i8], ptr addrspace(1) %38, i64 %address.local86, !dbg !11
  %137 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %138 = ptrtoint ptr addrspace(1) %address.local87 to i64, !dbg !11
  %139 = sub i64 %138, %137, !dbg !11
  %dma.buffer.offset448 = trunc i64 %139 to i32, !dbg !11
  %140 = trunc i64 %137 to i32, !dbg !11
  %141 = insertelement <4 x i32> poison, i32 %140, i64 0, !dbg !11
  %142 = lshr i64 %137, 32, !dbg !11
  %143 = trunc i64 %142 to i32, !dbg !11
  %144 = insertelement <4 x i32> %141, i32 %143, i64 1, !dbg !11
  %145 = insertelement <4 x i32> %144, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc449 = insertelement <4 x i32> %145, i32 131072, i64 3, !dbg !11
  %146 = ptrtoint ptr addrspace(3) %136 to i32, !dbg !11
  %dma.buffer.lds450 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %146), !dbg !11
  %147 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds450, i32 %dma.buffer.offset448, <4 x i32> %dma.buffer.rsrc449) #10, !dbg !11
  %148 = add nuw i64 %47, 4096, !dbg !17
  %149 = icmp slt i64 %148, 0, !dbg !17
  %150 = sub nuw i64 -4097, %47, !dbg !17
  %151 = select i1 %149, i64 %150, i64 %148, !dbg !17
  %152 = sdiv i64 %151, 16384, !dbg !17
  %.lobit3 = ashr i64 %148, 63, !dbg !17
  %153 = xor i64 %152, %.lobit3, !dbg !17
  %154 = add nsw i64 %153, %8, !dbg !17
  %155 = add nsw i64 %81, %41, !dbg !17
  %156 = add nsw i64 %155, 32, !dbg !17
  %157 = icmp slt i64 %155, -32, !dbg !17
  %158 = sub nsw i64 -33, %155, !dbg !17
  %159 = select i1 %157, i64 %158, i64 %156, !dbg !17
  %160 = sdiv i64 %159, 128, !dbg !17
  %161 = sext i1 %157 to i64, !dbg !17
  %162 = xor i64 %160, %161, !dbg !17
  %163 = shl nsw i64 %162, 7, !dbg !17
  %164 = sub i64 %41, %163, !dbg !17
  %165 = shl i64 %154, 19, !dbg !17
  %166 = and i64 %165, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %167 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %166, !dbg !17
  %168 = getelementptr [2 x i8], ptr addrspace(1) %167, i64 %81, !dbg !17
  %169 = getelementptr [2 x i8], ptr addrspace(1) %168, i64 %164, !dbg !17
  %170 = getelementptr i8, ptr addrspace(1) %169, i64 64, !dbg !17
  %171 = getelementptr [2 x i8], ptr addrspace(1) %170, i64 %87, !dbg !17
  br label %172, !dbg !17

172:                                              ; preds = %175, %121
  %173 = phi i64 [ %201, %175 ], [ %25, %121 ], !dbg !17
  %174 = icmp samesign ult i64 %173, 128, !dbg !17
  br i1 %174, label %175, label %202, !dbg !17

175:                                              ; preds = %172
  %176 = lshr i64 %173, 2, !dbg !17
  %177 = and i64 %176, 24, !dbg !17
  %178 = shl nuw nsw i64 %173, 3, !dbg !17
  %179 = and i64 %178, 512, !dbg !17
  %180 = shl nuw nsw i64 %173, 7, !dbg !17
  %181 = and i64 %180, 3968, !dbg !17
  %182 = getelementptr [2 x i8], ptr addrspace(1) %171, i64 %177, !dbg !17
  %183 = getelementptr [2 x i8], ptr addrspace(1) %182, i64 %181, !dbg !17
  %184 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %179, !dbg !17
  %185 = ptrtoint ptr addrspace(3) %184 to i64, !dbg !17
  %186 = trunc i64 %185 to i32, !dbg !17
  %187 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %186), !dbg !17
  %188 = zext i32 %187 to i64, !dbg !17
  %189 = inttoptr i64 %188 to ptr addrspace(3), !dbg !17
  %190 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %191 = ptrtoint ptr addrspace(1) %183 to i64, !dbg !17
  %192 = sub i64 %191, %190, !dbg !17
  %dma.buffer.offset421 = trunc i64 %192 to i32, !dbg !17
  %193 = trunc i64 %190 to i32, !dbg !17
  %194 = insertelement <4 x i32> poison, i32 %193, i64 0, !dbg !17
  %195 = lshr i64 %190, 32, !dbg !17
  %196 = trunc i64 %195 to i32, !dbg !17
  %197 = insertelement <4 x i32> %194, i32 %196, i64 1, !dbg !17
  %198 = insertelement <4 x i32> %197, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc422 = insertelement <4 x i32> %198, i32 131072, i64 3, !dbg !17
  %199 = ptrtoint ptr addrspace(3) %189 to i32, !dbg !17
  %dma.buffer.lds423 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %199), !dbg !17
  %200 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds423, i32 %dma.buffer.offset421, <4 x i32> %dma.buffer.rsrc422) #10, !dbg !17
  %201 = or disjoint i64 %25, 256, !dbg !17
  br label %172, !dbg !17

202:                                              ; preds = %172
  %address.thread88 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local89 = lshr i32 %address.thread88, 2, !dbg !12
  %address.local90 = and i32 %address.local89, 4, !dbg !12
  %address.local91 = shl nuw nsw i32 %address.thread88, 1, !dbg !12
  %address.local92 = and i32 %address.local91, 384, !dbg !12
  %address.local93 = or disjoint i32 %address.local90, %address.local92, !dbg !12
  %address.local94 = shl nuw nsw i32 %address.thread88, 3, !dbg !12
  %address.local95 = and i32 %address.local94, 120, !dbg !12
  %address.local96 = or disjoint i32 %address.local93, %address.local95, !dbg !12
  %address.local97 = shl nuw nsw i32 %address.thread88, 4, !dbg !12
  %address.local98 = and i32 %address.local97, 512, !dbg !12
  %address.local99 = or disjoint i32 %address.local96, %address.local98, !dbg !12
  %address.local100 = zext nneg i32 %address.local99 to i64, !dbg !12
  %address.local101 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local100, !dbg !12
  %203 = load <4 x half>, ptr addrspace(3) %address.local101, align 2, !dbg !12
  %address.local102 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local100, !dbg !12
  %address.local103 = getelementptr i8, ptr addrspace(3) %address.local102, i64 2048, !dbg !12
  %204 = load <4 x half>, ptr addrspace(3) %address.local103, align 2, !dbg !12
  %205 = fmul <4 x half> %203, splat (half 0xH3015), !dbg !12
  %206 = fmul <4 x half> %204, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local104 = lshr i32 %address.thread88, 2, !dbg !16
  %address.local105 = and i32 %address.local104, 4, !dbg !16
  %address.local106 = shl nuw nsw i32 %address.thread88, 3, !dbg !16
  %address.local107 = and i32 %address.local106, 376, !dbg !16
  %address.local108 = or disjoint i32 %address.local105, %address.local107, !dbg !16
  %address.local109 = zext nneg i32 %address.local108 to i64, !dbg !16
  %address.local110 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local109, !dbg !16
  %207 = load <4 x half>, ptr addrspace(3) %address.local110, align 2, !dbg !16
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %205, <4 x half> %207, <4 x float> %130) #9, !dbg !16
  %address.local111 = lshr i32 %address.thread88, 2, !dbg !16
  %address.local112 = and i32 %address.local111, 4, !dbg !16
  %address.local113 = shl nuw nsw i32 %address.thread88, 3, !dbg !16
  %address.local114 = and i32 %address.local113, 376, !dbg !16
  %address.local115 = or disjoint i32 %address.local112, %address.local114, !dbg !16
  %address.local116 = zext nneg i32 %address.local115 to i64, !dbg !16
  %address.local117 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local116, !dbg !16
  %address.local118 = getelementptr i8, ptr addrspace(3) %address.local117, i64 1024, !dbg !16
  %209 = load <4 x half>, ptr addrspace(3) %address.local118, align 2, !dbg !16
  %address.local119 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local109, !dbg !16
  %address.local120 = getelementptr i8, ptr addrspace(3) %address.local119, i64 256, !dbg !16
  %210 = load <4 x half>, ptr addrspace(3) %address.local120, align 2, !dbg !16
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %206, <4 x half> %209, <4 x float> %208) #9, !dbg !16
  %address.local121 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local116, !dbg !16
  %address.local122 = getelementptr i8, ptr addrspace(3) %address.local121, i64 1280, !dbg !16
  %212 = load <4 x half>, ptr addrspace(3) %address.local122, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %205, <4 x half> %210, <4 x float> %133) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %206, <4 x half> %212, <4 x float> %213) #9, !dbg !16
  %address.local123 = zext nneg i32 %address.thread88 to i64, !dbg !12
  %address.local124 = shl nuw nsw i64 %address.local123, 3, !dbg !11
  %address.local125 = and i64 %address.local124, 1536, !dbg !11
  %address.local126 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local125, !dbg !11
  %address.local127 = ptrtoint ptr addrspace(3) %address.local126 to i64, !dbg !11
  %address.local128 = trunc i64 %address.local127 to i32, !dbg !11
  %215 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local128), !dbg !11
  %216 = zext i32 %215 to i64, !dbg !11
  %217 = inttoptr i64 %216 to ptr addrspace(3), !dbg !11
  %address.local129 = shl nuw nsw i64 %address.local123, 7, !dbg !11
  %address.local130 = and i64 %address.local129, 8064, !dbg !11
  %address.local131 = lshr i64 %address.local123, 3, !dbg !11
  %address.local132 = and i64 %address.local131, 24, !dbg !11
  %address.local133 = or disjoint i64 %address.local130, %address.local132, !dbg !11
  %address.local134 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local133, !dbg !11
  %218 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %219 = ptrtoint ptr addrspace(1) %address.local134 to i64, !dbg !11
  %220 = sub i64 %219, %218, !dbg !11
  %dma.buffer.offset445 = trunc i64 %220 to i32, !dbg !11
  %221 = trunc i64 %218 to i32, !dbg !11
  %222 = insertelement <4 x i32> poison, i32 %221, i64 0, !dbg !11
  %223 = lshr i64 %218, 32, !dbg !11
  %224 = trunc i64 %223 to i32, !dbg !11
  %225 = insertelement <4 x i32> %222, i32 %224, i64 1, !dbg !11
  %226 = insertelement <4 x i32> %225, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc446 = insertelement <4 x i32> %226, i32 131072, i64 3, !dbg !11
  %227 = ptrtoint ptr addrspace(3) %217 to i32, !dbg !11
  %dma.buffer.lds447 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %227), !dbg !11
  %228 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds447, i32 %dma.buffer.offset445, <4 x i32> %dma.buffer.rsrc446) #10, !dbg !11
  %229 = add nuw i64 %47, 8192, !dbg !17
  %230 = icmp slt i64 %229, 0, !dbg !17
  %231 = sub nuw i64 -8193, %47, !dbg !17
  %232 = select i1 %230, i64 %231, i64 %229, !dbg !17
  %233 = sdiv i64 %232, 16384, !dbg !17
  %.lobit4 = ashr i64 %229, 63, !dbg !17
  %234 = xor i64 %233, %.lobit4, !dbg !17
  %235 = add nsw i64 %234, %8, !dbg !17
  %236 = add nsw i64 %155, 64, !dbg !17
  %237 = icmp slt i64 %155, -64, !dbg !17
  %238 = sub nsw i64 -65, %155, !dbg !17
  %239 = select i1 %237, i64 %238, i64 %236, !dbg !17
  %240 = sdiv i64 %239, 128, !dbg !17
  %241 = sext i1 %237 to i64, !dbg !17
  %242 = xor i64 %240, %241, !dbg !17
  %243 = shl nsw i64 %242, 7, !dbg !17
  %244 = sub i64 %41, %243, !dbg !17
  %245 = shl i64 %235, 19, !dbg !17
  %246 = and i64 %245, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %247 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %246, !dbg !17
  %248 = getelementptr [2 x i8], ptr addrspace(1) %247, i64 %81, !dbg !17
  %249 = getelementptr [2 x i8], ptr addrspace(1) %248, i64 %244, !dbg !17
  %250 = getelementptr i8, ptr addrspace(1) %249, i64 128, !dbg !17
  %251 = getelementptr [2 x i8], ptr addrspace(1) %250, i64 %87, !dbg !17
  br label %252, !dbg !17

252:                                              ; preds = %255, %202
  %253 = phi i64 [ %281, %255 ], [ %25, %202 ], !dbg !17
  %254 = icmp samesign ult i64 %253, 128, !dbg !17
  br i1 %254, label %255, label %282, !dbg !17

255:                                              ; preds = %252
  %256 = lshr i64 %253, 2, !dbg !17
  %257 = and i64 %256, 24, !dbg !17
  %258 = shl nuw nsw i64 %253, 3, !dbg !17
  %259 = and i64 %258, 512, !dbg !17
  %260 = shl nuw nsw i64 %253, 7, !dbg !17
  %261 = and i64 %260, 3968, !dbg !17
  %262 = getelementptr [2 x i8], ptr addrspace(1) %251, i64 %257, !dbg !17
  %263 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %261, !dbg !17
  %264 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %259, !dbg !17
  %265 = ptrtoint ptr addrspace(3) %264 to i64, !dbg !17
  %266 = trunc i64 %265 to i32, !dbg !17
  %267 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %266), !dbg !17
  %268 = zext i32 %267 to i64, !dbg !17
  %269 = inttoptr i64 %268 to ptr addrspace(3), !dbg !17
  %270 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %271 = ptrtoint ptr addrspace(1) %263 to i64, !dbg !17
  %272 = sub i64 %271, %270, !dbg !17
  %dma.buffer.offset424 = trunc i64 %272 to i32, !dbg !17
  %273 = trunc i64 %270 to i32, !dbg !17
  %274 = insertelement <4 x i32> poison, i32 %273, i64 0, !dbg !17
  %275 = lshr i64 %270, 32, !dbg !17
  %276 = trunc i64 %275 to i32, !dbg !17
  %277 = insertelement <4 x i32> %274, i32 %276, i64 1, !dbg !17
  %278 = insertelement <4 x i32> %277, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc425 = insertelement <4 x i32> %278, i32 131072, i64 3, !dbg !17
  %279 = ptrtoint ptr addrspace(3) %269 to i32, !dbg !17
  %dma.buffer.lds426 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %279), !dbg !17
  %280 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds426, i32 %dma.buffer.offset424, <4 x i32> %dma.buffer.rsrc425) #10, !dbg !17
  %281 = or disjoint i64 %25, 256, !dbg !17
  br label %252, !dbg !17

282:                                              ; preds = %252
  %address.thread135 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local136 = lshr i32 %address.thread135, 2, !dbg !12
  %address.local137 = and i32 %address.local136, 4, !dbg !12
  %address.local138 = shl nuw nsw i32 %address.thread135, 1, !dbg !12
  %address.local139 = and i32 %address.local138, 384, !dbg !12
  %address.local140 = or disjoint i32 %address.local137, %address.local139, !dbg !12
  %address.local141 = shl nuw nsw i32 %address.thread135, 3, !dbg !12
  %address.local142 = and i32 %address.local141, 120, !dbg !12
  %address.local143 = or disjoint i32 %address.local140, %address.local142, !dbg !12
  %address.local144 = shl nuw nsw i32 %address.thread135, 4, !dbg !12
  %address.local145 = and i32 %address.local144, 512, !dbg !12
  %address.local146 = or disjoint i32 %address.local143, %address.local145, !dbg !12
  %address.local147 = zext nneg i32 %address.local146 to i64, !dbg !12
  %address.local148 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local147, !dbg !12
  %283 = load <4 x half>, ptr addrspace(3) %address.local148, align 2, !dbg !12
  %address.local149 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local147, !dbg !12
  %address.local150 = getelementptr i8, ptr addrspace(3) %address.local149, i64 2048, !dbg !12
  %284 = load <4 x half>, ptr addrspace(3) %address.local150, align 2, !dbg !12
  %285 = fmul <4 x half> %283, splat (half 0xH3015), !dbg !12
  %286 = fmul <4 x half> %284, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local151 = lshr i32 %address.thread135, 2, !dbg !16
  %address.local152 = and i32 %address.local151, 4, !dbg !16
  %address.local153 = shl nuw nsw i32 %address.thread135, 3, !dbg !16
  %address.local154 = and i32 %address.local153, 376, !dbg !16
  %address.local155 = or disjoint i32 %address.local152, %address.local154, !dbg !16
  %address.local156 = zext nneg i32 %address.local155 to i64, !dbg !16
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local156, !dbg !16
  %287 = load <4 x half>, ptr addrspace(3) %address.local157, align 2, !dbg !16
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %287, <4 x float> %211) #9, !dbg !16
  %address.local158 = lshr i32 %address.thread135, 2, !dbg !16
  %address.local159 = and i32 %address.local158, 4, !dbg !16
  %address.local160 = shl nuw nsw i32 %address.thread135, 3, !dbg !16
  %address.local161 = and i32 %address.local160, 376, !dbg !16
  %address.local162 = or disjoint i32 %address.local159, %address.local161, !dbg !16
  %address.local163 = zext nneg i32 %address.local162 to i64, !dbg !16
  %address.local164 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local163, !dbg !16
  %address.local165 = getelementptr i8, ptr addrspace(3) %address.local164, i64 1024, !dbg !16
  %289 = load <4 x half>, ptr addrspace(3) %address.local165, align 2, !dbg !16
  %address.local166 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local156, !dbg !16
  %address.local167 = getelementptr i8, ptr addrspace(3) %address.local166, i64 256, !dbg !16
  %290 = load <4 x half>, ptr addrspace(3) %address.local167, align 2, !dbg !16
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %289, <4 x float> %288) #9, !dbg !16
  %address.local168 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local163, !dbg !16
  %address.local169 = getelementptr i8, ptr addrspace(3) %address.local168, i64 1280, !dbg !16
  %292 = load <4 x half>, ptr addrspace(3) %address.local169, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %285, <4 x half> %290, <4 x float> %214) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %292, <4 x float> %293) #9, !dbg !16
  %address.local170 = zext nneg i32 %address.thread135 to i64, !dbg !12
  %address.local171 = shl nuw nsw i64 %address.local170, 3, !dbg !11
  %address.local172 = and i64 %address.local171, 1536, !dbg !11
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !11
  %address.local174 = ptrtoint ptr addrspace(3) %address.local173 to i64, !dbg !11
  %address.local175 = trunc i64 %address.local174 to i32, !dbg !11
  %295 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local175), !dbg !11
  %296 = zext i32 %295 to i64, !dbg !11
  %297 = inttoptr i64 %296 to ptr addrspace(3), !dbg !11
  %address.local176 = shl nuw nsw i64 %address.local170, 7, !dbg !11
  %address.local177 = and i64 %address.local176, 8064, !dbg !11
  %address.local178 = lshr i64 %address.local170, 3, !dbg !11
  %address.local179 = and i64 %address.local178, 24, !dbg !11
  %address.local180 = or disjoint i64 %address.local177, %address.local179, !dbg !11
  %address.local181 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local180, !dbg !11
  %298 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %299 = ptrtoint ptr addrspace(1) %address.local181 to i64, !dbg !11
  %300 = sub i64 %299, %298, !dbg !11
  %dma.buffer.offset442 = trunc i64 %300 to i32, !dbg !11
  %301 = trunc i64 %298 to i32, !dbg !11
  %302 = insertelement <4 x i32> poison, i32 %301, i64 0, !dbg !11
  %303 = lshr i64 %298, 32, !dbg !11
  %304 = trunc i64 %303 to i32, !dbg !11
  %305 = insertelement <4 x i32> %302, i32 %304, i64 1, !dbg !11
  %306 = insertelement <4 x i32> %305, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc443 = insertelement <4 x i32> %306, i32 131072, i64 3, !dbg !11
  %307 = ptrtoint ptr addrspace(3) %297 to i32, !dbg !11
  %dma.buffer.lds444 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %307), !dbg !11
  %308 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds444, i32 %dma.buffer.offset442, <4 x i32> %dma.buffer.rsrc443) #10, !dbg !11
  %309 = add nuw i64 %47, 12288, !dbg !17
  %310 = icmp slt i64 %309, 0, !dbg !17
  %311 = sub nuw i64 -12289, %47, !dbg !17
  %312 = select i1 %310, i64 %311, i64 %309, !dbg !17
  %313 = sdiv i64 %312, 16384, !dbg !17
  %.lobit5 = ashr i64 %309, 63, !dbg !17
  %314 = xor i64 %313, %.lobit5, !dbg !17
  %315 = add nsw i64 %314, %8, !dbg !17
  %316 = add nsw i64 %155, 96, !dbg !17
  %317 = icmp slt i64 %155, -96, !dbg !17
  %318 = sub nsw i64 -97, %155, !dbg !17
  %319 = select i1 %317, i64 %318, i64 %316, !dbg !17
  %320 = sdiv i64 %319, 128, !dbg !17
  %321 = sext i1 %317 to i64, !dbg !17
  %322 = xor i64 %320, %321, !dbg !17
  %323 = shl nsw i64 %322, 7, !dbg !17
  %324 = sub i64 %41, %323, !dbg !17
  %325 = shl i64 %315, 19, !dbg !17
  %326 = and i64 %325, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %327 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %326, !dbg !17
  %328 = getelementptr [2 x i8], ptr addrspace(1) %327, i64 %81, !dbg !17
  %329 = getelementptr [2 x i8], ptr addrspace(1) %328, i64 %324, !dbg !17
  %330 = getelementptr i8, ptr addrspace(1) %329, i64 192, !dbg !17
  %331 = getelementptr [2 x i8], ptr addrspace(1) %330, i64 %87, !dbg !17
  br label %332, !dbg !17

332:                                              ; preds = %335, %282
  %333 = phi i64 [ %361, %335 ], [ %25, %282 ], !dbg !17
  %334 = icmp samesign ult i64 %333, 128, !dbg !17
  br i1 %334, label %335, label %362, !dbg !17

335:                                              ; preds = %332
  %336 = lshr i64 %333, 2, !dbg !17
  %337 = and i64 %336, 24, !dbg !17
  %338 = shl nuw nsw i64 %333, 3, !dbg !17
  %339 = and i64 %338, 512, !dbg !17
  %340 = shl nuw nsw i64 %333, 7, !dbg !17
  %341 = and i64 %340, 3968, !dbg !17
  %342 = getelementptr [2 x i8], ptr addrspace(1) %331, i64 %337, !dbg !17
  %343 = getelementptr [2 x i8], ptr addrspace(1) %342, i64 %341, !dbg !17
  %344 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %339, !dbg !17
  %345 = ptrtoint ptr addrspace(3) %344 to i64, !dbg !17
  %346 = trunc i64 %345 to i32, !dbg !17
  %347 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %346), !dbg !17
  %348 = zext i32 %347 to i64, !dbg !17
  %349 = inttoptr i64 %348 to ptr addrspace(3), !dbg !17
  %350 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %351 = ptrtoint ptr addrspace(1) %343 to i64, !dbg !17
  %352 = sub i64 %351, %350, !dbg !17
  %dma.buffer.offset427 = trunc i64 %352 to i32, !dbg !17
  %353 = trunc i64 %350 to i32, !dbg !17
  %354 = insertelement <4 x i32> poison, i32 %353, i64 0, !dbg !17
  %355 = lshr i64 %350, 32, !dbg !17
  %356 = trunc i64 %355 to i32, !dbg !17
  %357 = insertelement <4 x i32> %354, i32 %356, i64 1, !dbg !17
  %358 = insertelement <4 x i32> %357, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc428 = insertelement <4 x i32> %358, i32 131072, i64 3, !dbg !17
  %359 = ptrtoint ptr addrspace(3) %349 to i32, !dbg !17
  %dma.buffer.lds429 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %359), !dbg !17
  %360 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds429, i32 %dma.buffer.offset427, <4 x i32> %dma.buffer.rsrc428) #10, !dbg !17
  %361 = or disjoint i64 %25, 256, !dbg !17
  br label %332, !dbg !17

362:                                              ; preds = %332
  %address.thread182 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local183 = lshr i32 %address.thread182, 2, !dbg !12
  %address.local184 = and i32 %address.local183, 4, !dbg !12
  %address.local185 = shl nuw nsw i32 %address.thread182, 1, !dbg !12
  %address.local186 = and i32 %address.local185, 384, !dbg !12
  %address.local187 = or disjoint i32 %address.local184, %address.local186, !dbg !12
  %address.local188 = shl nuw nsw i32 %address.thread182, 3, !dbg !12
  %address.local189 = and i32 %address.local188, 120, !dbg !12
  %address.local190 = or disjoint i32 %address.local187, %address.local189, !dbg !12
  %address.local191 = shl nuw nsw i32 %address.thread182, 4, !dbg !12
  %address.local192 = and i32 %address.local191, 512, !dbg !12
  %address.local193 = or disjoint i32 %address.local190, %address.local192, !dbg !12
  %address.local194 = zext nneg i32 %address.local193 to i64, !dbg !12
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local194, !dbg !12
  %363 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !12
  %address.local196 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local194, !dbg !12
  %address.local197 = getelementptr i8, ptr addrspace(3) %address.local196, i64 2048, !dbg !12
  %364 = load <4 x half>, ptr addrspace(3) %address.local197, align 2, !dbg !12
  %365 = fmul <4 x half> %363, splat (half 0xH3015), !dbg !12
  %366 = fmul <4 x half> %364, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %367 = add nsw i64 %81, %8, !dbg !18
  %368 = shl i64 %367, 19, !dbg !18
  %369 = and i64 %368, 16252928, !dbg !18
  %370 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %369, !dbg !18
  %371 = getelementptr [2 x i8], ptr addrspace(1) %370, i64 %87, !dbg !18
  br label %372, !dbg !18

372:                                              ; preds = %375, %362
  %373 = phi i64 [ %401, %375 ], [ %25, %362 ], !dbg !18
  %374 = icmp samesign ult i64 %373, 128, !dbg !18
  br i1 %374, label %375, label %402, !dbg !18

375:                                              ; preds = %372
  %376 = shl nuw nsw i64 %373, 3, !dbg !18
  %377 = and i64 %376, 24, !dbg !18
  %378 = shl nuw nsw i64 %373, 3, !dbg !18
  %379 = and i64 %378, 512, !dbg !18
  %380 = shl nuw nsw i64 %373, 5, !dbg !18
  %381 = and i64 %380, 3968, !dbg !18
  %382 = getelementptr [2 x i8], ptr addrspace(1) %371, i64 %381, !dbg !18
  %383 = getelementptr [2 x i8], ptr addrspace(1) %382, i64 %377, !dbg !18
  %384 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %379, !dbg !18
  %385 = ptrtoint ptr addrspace(3) %384 to i64, !dbg !18
  %386 = trunc i64 %385 to i32, !dbg !18
  %387 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %386), !dbg !18
  %388 = zext i32 %387 to i64, !dbg !18
  %389 = inttoptr i64 %388 to ptr addrspace(3), !dbg !18
  %390 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %391 = ptrtoint ptr addrspace(1) %383 to i64, !dbg !18
  %392 = sub i64 %391, %390, !dbg !18
  %dma.buffer.offset430 = trunc i64 %392 to i32, !dbg !18
  %393 = trunc i64 %390 to i32, !dbg !18
  %394 = insertelement <4 x i32> poison, i32 %393, i64 0, !dbg !18
  %395 = lshr i64 %390, 32, !dbg !18
  %396 = trunc i64 %395 to i32, !dbg !18
  %397 = insertelement <4 x i32> %394, i32 %396, i64 1, !dbg !18
  %398 = insertelement <4 x i32> %397, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc431 = insertelement <4 x i32> %398, i32 131072, i64 3, !dbg !18
  %399 = ptrtoint ptr addrspace(3) %389 to i32, !dbg !18
  %dma.buffer.lds432 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %399), !dbg !18
  %400 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds432, i32 %dma.buffer.offset430, <4 x i32> %dma.buffer.rsrc431) #10, !dbg !18
  %401 = or disjoint i64 %25, 256, !dbg !18
  br label %372, !dbg !18

402:                                              ; preds = %372
  %address.thread198 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local199 = lshr i32 %address.thread198, 2, !dbg !16
  %address.local200 = and i32 %address.local199, 4, !dbg !16
  %address.local201 = shl nuw nsw i32 %address.thread198, 3, !dbg !16
  %address.local202 = and i32 %address.local201, 376, !dbg !16
  %address.local203 = or disjoint i32 %address.local200, %address.local202, !dbg !16
  %address.local204 = zext nneg i32 %address.local203 to i64, !dbg !16
  %address.local205 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local204, !dbg !16
  %403 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %365, <4 x half> %403, <4 x float> %291) #9, !dbg !16
  %address.local206 = lshr i32 %address.thread198, 2, !dbg !16
  %address.local207 = and i32 %address.local206, 4, !dbg !16
  %address.local208 = shl nuw nsw i32 %address.thread198, 3, !dbg !16
  %address.local209 = and i32 %address.local208, 376, !dbg !16
  %address.local210 = or disjoint i32 %address.local207, %address.local209, !dbg !16
  %address.local211 = zext nneg i32 %address.local210 to i64, !dbg !16
  %address.local212 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local211, !dbg !16
  %address.local213 = getelementptr i8, ptr addrspace(3) %address.local212, i64 1024, !dbg !16
  %405 = load <4 x half>, ptr addrspace(3) %address.local213, align 2, !dbg !16
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local204, !dbg !16
  %address.local215 = getelementptr i8, ptr addrspace(3) %address.local214, i64 256, !dbg !16
  %406 = load <4 x half>, ptr addrspace(3) %address.local215, align 2, !dbg !16
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %405, <4 x float> %404) #9, !dbg !16
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local211, !dbg !16
  %address.local217 = getelementptr i8, ptr addrspace(3) %address.local216, i64 1280, !dbg !16
  %408 = load <4 x half>, ptr addrspace(3) %address.local217, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %365, <4 x half> %406, <4 x float> %294) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !22
  %410 = extractelement <4 x float> %407, i64 0, !dbg !22
  %411 = extractelement <4 x float> %407, i64 1, !dbg !22
  %412 = select i1 %.not, float %411, float %410, !dbg !13
  %413 = extractelement <4 x float> %407, i64 2, !dbg !22
  %414 = extractelement <4 x float> %407, i64 3, !dbg !22
  %415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %408, <4 x float> %409) #9, !dbg !16
  %416 = extractelement <4 x float> %415, i64 0, !dbg !22
  %417 = extractelement <4 x float> %415, i64 1, !dbg !22
  %418 = extractelement <4 x float> %415, i64 2, !dbg !22
  %419 = extractelement <4 x float> %415, i64 3, !dbg !22
  %420 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %421 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %420), !dbg !13
  %422 = add nuw nsw i32 %421, 64, !dbg !13
  %423 = xor i32 %421, 16, !dbg !13
  %424 = and i32 %422, 192, !dbg !13
  %425 = icmp samesign ult i32 %423, %424, !dbg !13
  %426 = select i1 %425, i32 %423, i32 %421, !dbg !13
  %427 = shl nuw nsw i32 %426, 2, !dbg !13
  %428 = bitcast float %412 to i32, !dbg !13
  %429 = call i32 @llvm.amdgcn.ds.bpermute(i32 %427, i32 %428), !dbg !13
  %430 = bitcast i32 %429 to float, !dbg !13
  %431 = select i1 %.not, float %410, float %430, !dbg !13
  %432 = select i1 %.not, float %430, float %411, !dbg !13
  %433 = select i1 %.not, float %414, float %413, !dbg !13
  %434 = bitcast float %433 to i32, !dbg !13
  %435 = call i32 @llvm.amdgcn.ds.bpermute(i32 %427, i32 %434), !dbg !13
  %436 = bitcast i32 %435 to float, !dbg !13
  %437 = select i1 %.not, float %413, float %436, !dbg !13
  %438 = select i1 %.not, float %436, float %414, !dbg !13
  %439 = select i1 %.not2, float %437, float %431, !dbg !13
  %440 = xor i32 %421, 32, !dbg !13
  %441 = icmp samesign ult i32 %440, %424, !dbg !13
  %442 = select i1 %441, i32 %440, i32 %421, !dbg !13
  %443 = shl nuw nsw i32 %442, 2, !dbg !13
  %444 = bitcast float %439 to i32, !dbg !13
  %445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %443, i32 %444), !dbg !13
  %446 = bitcast i32 %445 to float, !dbg !13
  %447 = select i1 %.not2, float %431, float %446, !dbg !13
  %448 = select i1 %.not2, float %446, float %437, !dbg !13
  %449 = select i1 %.not2, float %438, float %432, !dbg !13
  %450 = bitcast float %449 to i32, !dbg !13
  %451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %443, i32 %450), !dbg !13
  %452 = bitcast i32 %451 to float, !dbg !13
  %453 = select i1 %.not2, float %432, float %452, !dbg !13
  %454 = select i1 %.not2, float %452, float %438, !dbg !13
  %455 = select i1 %.not, float %417, float %416, !dbg !13
  %456 = bitcast float %455 to i32, !dbg !13
  %457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %427, i32 %456), !dbg !13
  %458 = bitcast i32 %457 to float, !dbg !13
  %459 = select i1 %.not, float %416, float %458, !dbg !13
  %460 = select i1 %.not, float %458, float %417, !dbg !13
  %461 = select i1 %.not, float %419, float %418, !dbg !13
  %462 = bitcast float %461 to i32, !dbg !13
  %463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %427, i32 %462), !dbg !13
  %464 = bitcast i32 %463 to float, !dbg !13
  %465 = select i1 %.not, float %418, float %464, !dbg !13
  %466 = select i1 %.not, float %464, float %419, !dbg !13
  %467 = select i1 %.not2, float %465, float %459, !dbg !13
  %468 = bitcast float %467 to i32, !dbg !13
  %469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %443, i32 %468), !dbg !13
  %470 = bitcast i32 %469 to float, !dbg !13
  %471 = select i1 %.not2, float %459, float %470, !dbg !13
  %472 = select i1 %.not2, float %470, float %465, !dbg !13
  %473 = select i1 %.not2, float %466, float %460, !dbg !13
  %474 = bitcast float %473 to i32, !dbg !13
  %475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %443, i32 %474), !dbg !13
  %476 = bitcast i32 %475 to float, !dbg !13
  %477 = select i1 %.not2, float %460, float %476, !dbg !13
  %478 = select i1 %.not2, float %476, float %466, !dbg !13
  %479 = or disjoint i64 %64, 31, !dbg !14
  %480 = icmp uge i64 %11, %479, !dbg !14
  %481 = and i1 %33, %480, !dbg !14
  br i1 %481, label %482, label %483, !dbg !14

482:                                              ; preds = %402
  br label %509, !dbg !14

483:                                              ; preds = %402
  %484 = add nsw i64 %11, %25, !dbg !14
  %485 = shl nuw nsw i64 %26, 4, !dbg !14
  %486 = add nsw i64 %484, %485, !dbg !14
  %487 = and i32 %24, 240, !dbg !14
  %index.compact = zext nneg i32 %487 to i64, !dbg !14
  %488 = sub nsw i64 %486, %index.compact, !dbg !14
  %489 = add nsw i64 %488, 1, !dbg !23
  %490 = or disjoint i64 %64, %28, !dbg !14
  %491 = lshr i64 %489, 32, !dbg !24
  %compare.high = trunc i64 %491 to i32, !dbg !24
  %492 = lshr i64 %490, 32, !dbg !24
  %compare.high393 = trunc i64 %492 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high393, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high393, !dbg !24
  %compare.low.base = trunc i64 %489 to i32, !dbg !24
  %compare.low.base394 = trunc i64 %490 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base394, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %493 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base394, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low395 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result396 = select i1 %compare.high.equal, i1 %compare.low395, i1 %compare.high.order, !dbg !24
  %494 = select i1 %compare.result396, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base394, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local398 = or i32 %compare.local.base397, 2, !dbg !24
  %compare.low399 = icmp ugt i32 %compare.low.base, %compare.local398, !dbg !24
  %compare.result400 = select i1 %compare.high.equal, i1 %compare.low399, i1 %compare.high.order, !dbg !24
  %495 = select i1 %compare.result400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base394, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local402 = or i32 %compare.local.base401, 3, !dbg !24
  %compare.low403 = icmp ugt i32 %compare.low.base, %compare.local402, !dbg !24
  %compare.result404 = select i1 %compare.high.equal, i1 %compare.low403, i1 %compare.high.order, !dbg !24
  %496 = select i1 %compare.result404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base394, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local406 = or i32 %compare.local.base405, 16, !dbg !24
  %compare.low407 = icmp ugt i32 %compare.low.base, %compare.local406, !dbg !24
  %compare.result408 = select i1 %compare.high.equal, i1 %compare.low407, i1 %compare.high.order, !dbg !24
  %497 = select i1 %compare.result408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base394, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local410 = or i32 %compare.local.base409, 17, !dbg !24
  %compare.low411 = icmp ugt i32 %compare.low.base, %compare.local410, !dbg !24
  %compare.result412 = select i1 %compare.high.equal, i1 %compare.low411, i1 %compare.high.order, !dbg !24
  %498 = select i1 %compare.result412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base413 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base394, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local414 = or i32 %compare.local.base413, 18, !dbg !24
  %compare.low415 = icmp ugt i32 %compare.low.base, %compare.local414, !dbg !24
  %compare.result416 = select i1 %compare.high.equal, i1 %compare.low415, i1 %compare.high.order, !dbg !24
  %499 = select i1 %compare.result416, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base417 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base394, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local418 = or i32 %compare.local.base417, 19, !dbg !24
  %compare.low419 = icmp ugt i32 %compare.low.base, %compare.local418, !dbg !24
  %compare.result420 = select i1 %compare.high.equal, i1 %compare.low419, i1 %compare.high.order, !dbg !24
  %500 = select i1 %compare.result420, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %501 = insertelement <8 x float> poison, float %493, i64 0, !dbg !14
  %502 = insertelement <8 x float> %501, float %494, i64 1, !dbg !14
  %503 = insertelement <8 x float> %502, float %495, i64 2, !dbg !14
  %504 = insertelement <8 x float> %503, float %496, i64 3, !dbg !14
  %505 = insertelement <8 x float> %504, float %497, i64 4, !dbg !14
  %506 = insertelement <8 x float> %505, float %498, i64 5, !dbg !14
  %507 = insertelement <8 x float> %506, float %499, i64 6, !dbg !14
  %508 = insertelement <8 x float> %507, float %500, i64 7, !dbg !14
  br label %509, !dbg !14

509:                                              ; preds = %482, %483
  %510 = phi <8 x float> [ %508, %483 ], [ zeroinitializer, %482 ], !dbg !14
  br label %511, !dbg !14

511:                                              ; preds = %509
  %512 = insertelement <4 x float> poison, float %447, i64 0, !dbg !26
  %513 = insertelement <4 x float> %512, float %453, i64 1, !dbg !26
  %514 = insertelement <4 x float> %513, float %448, i64 2, !dbg !26
  %515 = insertelement <4 x float> %514, float %454, i64 3, !dbg !26
  %516 = shufflevector <8 x float> %510, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %517 = fadd <4 x float> %515, %516, !dbg !26
  %518 = extractelement <4 x float> %517, i64 0, !dbg !26
  %519 = extractelement <4 x float> %517, i64 1, !dbg !26
  %520 = extractelement <4 x float> %517, i64 2, !dbg !26
  %521 = extractelement <4 x float> %517, i64 3, !dbg !26
  %522 = insertelement <4 x float> poison, float %471, i64 0, !dbg !26
  %523 = insertelement <4 x float> %522, float %477, i64 1, !dbg !26
  %524 = insertelement <4 x float> %523, float %472, i64 2, !dbg !26
  %525 = insertelement <4 x float> %524, float %478, i64 3, !dbg !26
  %526 = shufflevector <8 x float> %510, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %527 = fadd <4 x float> %525, %526, !dbg !26
  %528 = extractelement <4 x float> %527, i64 0, !dbg !26
  %529 = extractelement <4 x float> %527, i64 1, !dbg !26
  %530 = extractelement <4 x float> %527, i64 2, !dbg !26
  %531 = extractelement <4 x float> %527, i64 3, !dbg !26
  %532 = call float @llvm.maxnum.f32(float %518, float 0xFFF0000000000000), !dbg !27
  %533 = call float @llvm.maxnum.f32(float %532, float %519), !dbg !27
  %534 = call float @llvm.maxnum.f32(float %533, float %520), !dbg !27
  %535 = call float @llvm.maxnum.f32(float %534, float %521), !dbg !27
  %536 = call float @llvm.maxnum.f32(float %535, float %528), !dbg !27
  %537 = call float @llvm.maxnum.f32(float %536, float %529), !dbg !27
  %538 = call float @llvm.maxnum.f32(float %537, float %530), !dbg !27
  %539 = call float @llvm.maxnum.f32(float %538, float %531), !dbg !27
  %540 = bitcast float %539 to i32, !dbg !27
  %541 = call i32 @llvm.amdgcn.ds.bpermute(i32 %427, i32 %540), !dbg !27
  %542 = bitcast i32 %541 to float, !dbg !27
  %543 = call float @llvm.maxnum.f32(float %539, float %542), !dbg !27
  %544 = bitcast float %543 to i32, !dbg !27
  %545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %443, i32 %544), !dbg !27
  %546 = bitcast i32 %545 to float, !dbg !27
  %547 = call float @llvm.maxnum.f32(float %543, float %546), !dbg !27
  %548 = insertelement <1 x float> poison, float %547, i64 0, !dbg !27
  %549 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %548, <1 x float> %52), !dbg !27
  %550 = shufflevector <1 x float> %549, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %551 = fsub <4 x float> %517, %550, !dbg !28
  %552 = fsub <4 x float> %527, %550, !dbg !28
  %553 = fsub <1 x float> %52, %549, !dbg !29
  %554 = extractelement <1 x float> %553, i64 0, !dbg !30
  %555 = call float @llvm.amdgcn.exp2.f32(float %554), !dbg !30
  %556 = extractelement <1 x float> %51, i64 0, !dbg !31
  %557 = fmul float %556, %555, !dbg !31
  %558 = extractelement <4 x float> %551, i64 0, !dbg !32
  %559 = call float @llvm.amdgcn.exp2.f32(float %558), !dbg !32
  %560 = insertelement <4 x float> poison, float %559, i64 0, !dbg !34
  %561 = extractelement <4 x float> %551, i64 1, !dbg !32
  %562 = call float @llvm.amdgcn.exp2.f32(float %561), !dbg !32
  %563 = insertelement <4 x float> %560, float %562, i64 1, !dbg !34
  %564 = fadd float %559, %562, !dbg !33
  %565 = extractelement <4 x float> %551, i64 2, !dbg !32
  %566 = call float @llvm.amdgcn.exp2.f32(float %565), !dbg !32
  %567 = insertelement <4 x float> %563, float %566, i64 2, !dbg !34
  %568 = fadd float %564, %566, !dbg !33
  %569 = extractelement <4 x float> %551, i64 3, !dbg !32
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !32
  %571 = insertelement <4 x float> %567, float %570, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %571) #9, !dbg !34, !frisk.register_lifetimes !1
  %572 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %572) #9, !dbg !33, !frisk.register_lifetimes !1
  %573 = fadd float %568, %570, !dbg !33
  %574 = extractelement <4 x float> %552, i64 0, !dbg !32
  %575 = call float @llvm.amdgcn.exp2.f32(float %574), !dbg !32
  %576 = insertelement <4 x float> poison, float %575, i64 0, !dbg !34
  %577 = fadd float %573, %575, !dbg !33
  %578 = extractelement <4 x float> %552, i64 1, !dbg !32
  %579 = call float @llvm.amdgcn.exp2.f32(float %578), !dbg !32
  %580 = insertelement <4 x float> %576, float %579, i64 1, !dbg !34
  %581 = fadd float %577, %579, !dbg !33
  %582 = extractelement <4 x float> %552, i64 2, !dbg !32
  %583 = call float @llvm.amdgcn.exp2.f32(float %582), !dbg !32
  %584 = insertelement <4 x float> %580, float %583, i64 2, !dbg !34
  %585 = fadd float %581, %583, !dbg !33
  %586 = extractelement <4 x float> %552, i64 3, !dbg !32
  %587 = call float @llvm.amdgcn.exp2.f32(float %586), !dbg !32
  %588 = insertelement <4 x float> %584, float %587, i64 3, !dbg !34
  %math.result28 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %588) #9, !dbg !34, !frisk.register_lifetimes !1
  %589 = fptrunc <4 x float> %math.result28 to <4 x half>, !dbg !34
  %math.packed29 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %589) #9, !dbg !33, !frisk.register_lifetimes !1
  %590 = fadd float %585, %587, !dbg !33
  %591 = bitcast float %590 to i32, !dbg !33
  %592 = call i32 @llvm.amdgcn.ds.bpermute(i32 %427, i32 %591), !dbg !33
  %593 = bitcast i32 %592 to float, !dbg !33
  %594 = fadd float %590, %593, !dbg !33
  %595 = bitcast float %594 to i32, !dbg !33
  %596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %443, i32 %595), !dbg !33
  %597 = bitcast i32 %596 to float, !dbg !33
  %598 = fadd float %594, %597, !dbg !33
  %599 = fadd float %598, %557, !dbg !33
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread218 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local219 = shl nuw nsw i32 %address.thread218, 3, !dbg !15
  %address.local220 = and i32 %address.local219, 504, !dbg !15
  %address.local221 = zext nneg i32 %address.local220 to i64, !dbg !15
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local221, !dbg !15
  %600 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local222, i32 0) #0, !dbg !15
  %601 = bitcast <4 x i32> %600 to <8 x half>, !dbg !15
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local221, !dbg !15
  %address.local224 = getelementptr i8, ptr addrspace(3) %address.local223, i64 1024, !dbg !15
  %602 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local224, i32 0) #0, !dbg !15
  %603 = bitcast <4 x i32> %602 to <8 x half>, !dbg !15
  %604 = add nsw i64 %81, %8, !dbg !18
  %605 = shl i64 %604, 19, !dbg !18
  %606 = and i64 %605, 16252928, !dbg !18
  %607 = add i64 %606, %87, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %608 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %607, !dbg !18
  %609 = getelementptr i8, ptr addrspace(1) %608, i64 64, !dbg !18
  br label %610, !dbg !18

610:                                              ; preds = %613, %511
  %611 = phi i64 [ %639, %613 ], [ %25, %511 ], !dbg !18
  %612 = icmp samesign ult i64 %611, 128, !dbg !18
  br i1 %612, label %613, label %640, !dbg !18

613:                                              ; preds = %610
  %614 = shl nuw nsw i64 %611, 3, !dbg !18
  %615 = and i64 %614, 24, !dbg !18
  %616 = shl nuw nsw i64 %611, 3, !dbg !18
  %617 = and i64 %616, 512, !dbg !18
  %618 = shl nuw nsw i64 %611, 5, !dbg !18
  %619 = and i64 %618, 3968, !dbg !18
  %620 = getelementptr [2 x i8], ptr addrspace(1) %609, i64 %619, !dbg !18
  %621 = getelementptr [2 x i8], ptr addrspace(1) %620, i64 %615, !dbg !18
  %622 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %617, !dbg !18
  %623 = ptrtoint ptr addrspace(3) %622 to i64, !dbg !18
  %624 = trunc i64 %623 to i32, !dbg !18
  %625 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %624), !dbg !18
  %626 = zext i32 %625 to i64, !dbg !18
  %627 = inttoptr i64 %626 to ptr addrspace(3), !dbg !18
  %628 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %629 = ptrtoint ptr addrspace(1) %621 to i64, !dbg !18
  %630 = sub i64 %629, %628, !dbg !18
  %dma.buffer.offset433 = trunc i64 %630 to i32, !dbg !18
  %631 = trunc i64 %628 to i32, !dbg !18
  %632 = insertelement <4 x i32> poison, i32 %631, i64 0, !dbg !18
  %633 = lshr i64 %628, 32, !dbg !18
  %634 = trunc i64 %633 to i32, !dbg !18
  %635 = insertelement <4 x i32> %632, i32 %634, i64 1, !dbg !18
  %636 = insertelement <4 x i32> %635, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc434 = insertelement <4 x i32> %636, i32 131072, i64 3, !dbg !18
  %637 = ptrtoint ptr addrspace(3) %627 to i32, !dbg !18
  %dma.buffer.lds435 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %637), !dbg !18
  %638 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds435, i32 %dma.buffer.offset433, <4 x i32> %dma.buffer.rsrc434) #10, !dbg !18
  %639 = or disjoint i64 %25, 256, !dbg !18
  br label %610, !dbg !18

640:                                              ; preds = %610
  %641 = shufflevector <8 x half> %601, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %642 = insertelement <4 x float> poison, float %555, i64 0, !dbg !35
  %643 = shufflevector <4 x float> %642, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %644 = fmul <4 x float> %48, %643, !dbg !35
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %641, <4 x float> %644) #9, !dbg !15
  %646 = shufflevector <8 x half> %603, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed29, <4 x half> %646, <4 x float> %645) #9, !dbg !15
  %648 = shufflevector <8 x half> %601, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %649 = fmul <4 x float> %49, %643, !dbg !35
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %648, <4 x float> %649) #9, !dbg !15
  %651 = shufflevector <8 x half> %603, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed29, <4 x half> %651, <4 x float> %650) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread225 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local226 = shl nuw nsw i32 %address.thread225, 3, !dbg !15
  %address.local227 = and i32 %address.local226, 504, !dbg !15
  %address.local228 = zext nneg i32 %address.local227 to i64, !dbg !15
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local228, !dbg !15
  %653 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local229, i32 0) #0, !dbg !15
  %654 = bitcast <4 x i32> %653 to <8 x half>, !dbg !15
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local228, !dbg !15
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 1024, !dbg !15
  %655 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local231, i32 0) #0, !dbg !15
  %656 = bitcast <4 x i32> %655 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %657 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %607, !dbg !18
  %658 = getelementptr i8, ptr addrspace(1) %657, i64 128, !dbg !18
  br label %659, !dbg !18

659:                                              ; preds = %662, %640
  %660 = phi i64 [ %688, %662 ], [ %25, %640 ], !dbg !18
  %661 = icmp samesign ult i64 %660, 128, !dbg !18
  br i1 %661, label %662, label %689, !dbg !18

662:                                              ; preds = %659
  %663 = shl nuw nsw i64 %660, 3, !dbg !18
  %664 = and i64 %663, 24, !dbg !18
  %665 = shl nuw nsw i64 %660, 3, !dbg !18
  %666 = and i64 %665, 512, !dbg !18
  %667 = shl nuw nsw i64 %660, 5, !dbg !18
  %668 = and i64 %667, 3968, !dbg !18
  %669 = getelementptr [2 x i8], ptr addrspace(1) %658, i64 %668, !dbg !18
  %670 = getelementptr [2 x i8], ptr addrspace(1) %669, i64 %664, !dbg !18
  %671 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %666, !dbg !18
  %672 = ptrtoint ptr addrspace(3) %671 to i64, !dbg !18
  %673 = trunc i64 %672 to i32, !dbg !18
  %674 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %673), !dbg !18
  %675 = zext i32 %674 to i64, !dbg !18
  %676 = inttoptr i64 %675 to ptr addrspace(3), !dbg !18
  %677 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %678 = ptrtoint ptr addrspace(1) %670 to i64, !dbg !18
  %679 = sub i64 %678, %677, !dbg !18
  %dma.buffer.offset436 = trunc i64 %679 to i32, !dbg !18
  %680 = trunc i64 %677 to i32, !dbg !18
  %681 = insertelement <4 x i32> poison, i32 %680, i64 0, !dbg !18
  %682 = lshr i64 %677, 32, !dbg !18
  %683 = trunc i64 %682 to i32, !dbg !18
  %684 = insertelement <4 x i32> %681, i32 %683, i64 1, !dbg !18
  %685 = insertelement <4 x i32> %684, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc437 = insertelement <4 x i32> %685, i32 131072, i64 3, !dbg !18
  %686 = ptrtoint ptr addrspace(3) %676 to i32, !dbg !18
  %dma.buffer.lds438 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %686), !dbg !18
  %687 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds438, i32 %dma.buffer.offset436, <4 x i32> %dma.buffer.rsrc437) #10, !dbg !18
  %688 = or disjoint i64 %25, 256, !dbg !18
  br label %659, !dbg !18

689:                                              ; preds = %659
  %690 = shufflevector <8 x half> %654, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %691 = fmul <4 x float> %53, %643, !dbg !35
  %692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %690, <4 x float> %691) #9, !dbg !15
  %693 = shufflevector <8 x half> %656, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed29, <4 x half> %693, <4 x float> %692) #9, !dbg !15
  %695 = shufflevector <8 x half> %654, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %696 = fmul <4 x float> %54, %643, !dbg !35
  %697 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %695, <4 x float> %696) #9, !dbg !15
  %698 = shufflevector <8 x half> %656, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed29, <4 x half> %698, <4 x float> %697) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread232 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local233 = shl nuw nsw i32 %address.thread232, 3, !dbg !15
  %address.local234 = and i32 %address.local233, 504, !dbg !15
  %address.local235 = zext nneg i32 %address.local234 to i64, !dbg !15
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local235, !dbg !15
  %700 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local236, i32 0) #0, !dbg !15
  %701 = bitcast <4 x i32> %700 to <8 x half>, !dbg !15
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local235, !dbg !15
  %address.local238 = getelementptr i8, ptr addrspace(3) %address.local237, i64 1024, !dbg !15
  %702 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local238, i32 0) #0, !dbg !15
  %703 = bitcast <4 x i32> %702 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %704 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %607, !dbg !18
  %705 = getelementptr i8, ptr addrspace(1) %704, i64 192, !dbg !18
  br label %706, !dbg !18

706:                                              ; preds = %709, %689
  %707 = phi i64 [ %735, %709 ], [ %25, %689 ], !dbg !18
  %708 = icmp samesign ult i64 %707, 128, !dbg !18
  br i1 %708, label %709, label %736, !dbg !18

709:                                              ; preds = %706
  %710 = shl nuw nsw i64 %707, 3, !dbg !18
  %711 = and i64 %710, 24, !dbg !18
  %712 = shl nuw nsw i64 %707, 3, !dbg !18
  %713 = and i64 %712, 512, !dbg !18
  %714 = shl nuw nsw i64 %707, 5, !dbg !18
  %715 = and i64 %714, 3968, !dbg !18
  %716 = getelementptr [2 x i8], ptr addrspace(1) %705, i64 %715, !dbg !18
  %717 = getelementptr [2 x i8], ptr addrspace(1) %716, i64 %711, !dbg !18
  %718 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %713, !dbg !18
  %719 = ptrtoint ptr addrspace(3) %718 to i64, !dbg !18
  %720 = trunc i64 %719 to i32, !dbg !18
  %721 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %720), !dbg !18
  %722 = zext i32 %721 to i64, !dbg !18
  %723 = inttoptr i64 %722 to ptr addrspace(3), !dbg !18
  %724 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %725 = ptrtoint ptr addrspace(1) %717 to i64, !dbg !18
  %726 = sub i64 %725, %724, !dbg !18
  %dma.buffer.offset439 = trunc i64 %726 to i32, !dbg !18
  %727 = trunc i64 %724 to i32, !dbg !18
  %728 = insertelement <4 x i32> poison, i32 %727, i64 0, !dbg !18
  %729 = lshr i64 %724, 32, !dbg !18
  %730 = trunc i64 %729 to i32, !dbg !18
  %731 = insertelement <4 x i32> %728, i32 %730, i64 1, !dbg !18
  %732 = insertelement <4 x i32> %731, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc440 = insertelement <4 x i32> %732, i32 131072, i64 3, !dbg !18
  %733 = ptrtoint ptr addrspace(3) %723 to i32, !dbg !18
  %dma.buffer.lds441 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %733), !dbg !18
  %734 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds441, i32 %dma.buffer.offset439, <4 x i32> %dma.buffer.rsrc440) #10, !dbg !18
  %735 = or disjoint i64 %25, 256, !dbg !18
  br label %706, !dbg !18

736:                                              ; preds = %706
  %737 = shufflevector <8 x half> %701, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %738 = fmul <4 x float> %56, %643, !dbg !35
  %739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %737, <4 x float> %738) #9, !dbg !15
  %740 = shufflevector <8 x half> %703, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %741 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed29, <4 x half> %740, <4 x float> %739) #9, !dbg !15
  %742 = shufflevector <8 x half> %701, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %743 = fmul <4 x float> %57, %643, !dbg !35
  %744 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %742, <4 x float> %743) #9, !dbg !15
  %745 = shufflevector <8 x half> %703, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed29, <4 x half> %745, <4 x float> %744) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread239 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local240 = shl nuw nsw i32 %address.thread239, 3, !dbg !15
  %address.local241 = and i32 %address.local240, 504, !dbg !15
  %address.local242 = zext nneg i32 %address.local241 to i64, !dbg !15
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local242, !dbg !15
  %747 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local243, i32 0) #0, !dbg !15
  %748 = bitcast <4 x i32> %747 to <8 x half>, !dbg !15
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local242, !dbg !15
  %address.local245 = getelementptr i8, ptr addrspace(3) %address.local244, i64 1024, !dbg !15
  %749 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local245, i32 0) #0, !dbg !15
  %750 = bitcast <4 x i32> %749 to <8 x half>, !dbg !15
  %751 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %752 = fmul <4 x float> %59, %643, !dbg !35
  %753 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %751, <4 x float> %752) #9, !dbg !15
  %754 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed29, <4 x half> %754, <4 x float> %753) #9, !dbg !15
  %756 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %757 = fmul <4 x float> %60, %643, !dbg !35
  %758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %756, <4 x float> %757) #9, !dbg !15
  %759 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed29, <4 x half> %759, <4 x float> %758) #9, !dbg !15
  %761 = shufflevector <4 x float> %647, <4 x float> %652, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %762 = insertelement <1 x float> poison, float %599, i64 0, !dbg !7
  %763 = shufflevector <4 x float> %694, <4 x float> %699, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %764 = shufflevector <4 x float> %741, <4 x float> %746, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %765 = shufflevector <4 x float> %755, <4 x float> %760, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %766 = add nuw nsw i64 %47, 1, !dbg !7
  br label %46, !dbg !7

767:                                              ; preds = %46
  %768 = extractelement <8 x float> %61, i64 0, !dbg !7
  %769 = extractelement <8 x float> %61, i64 1, !dbg !7
  %770 = extractelement <8 x float> %61, i64 2, !dbg !7
  %771 = extractelement <8 x float> %61, i64 3, !dbg !7
  %772 = select i1 %.not, float %769, float %768, !dbg !7
  %773 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %774 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %773), !dbg !7
  %775 = add nuw nsw i32 %774, 64, !dbg !7
  %776 = xor i32 %774, 16, !dbg !7
  %777 = and i32 %775, 192, !dbg !7
  %778 = icmp samesign ult i32 %776, %777, !dbg !7
  %779 = select i1 %778, i32 %776, i32 %774, !dbg !7
  %780 = shl nuw nsw i32 %779, 2, !dbg !7
  %781 = bitcast float %772 to i32, !dbg !7
  %782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %781), !dbg !7
  %783 = bitcast i32 %782 to float, !dbg !7
  %784 = select i1 %.not, float %768, float %783, !dbg !7
  %785 = select i1 %.not, float %783, float %769, !dbg !7
  %786 = select i1 %.not, float %771, float %770, !dbg !7
  %787 = bitcast float %786 to i32, !dbg !7
  %788 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %787), !dbg !7
  %789 = bitcast i32 %788 to float, !dbg !7
  %790 = select i1 %.not, float %770, float %789, !dbg !7
  %791 = select i1 %.not, float %789, float %771, !dbg !7
  %792 = select i1 %.not2, float %790, float %784, !dbg !7
  %793 = xor i32 %774, 32, !dbg !7
  %794 = icmp samesign ult i32 %793, %777, !dbg !7
  %795 = select i1 %794, i32 %793, i32 %774, !dbg !7
  %796 = shl nuw nsw i32 %795, 2, !dbg !7
  %797 = bitcast float %792 to i32, !dbg !7
  %798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %797), !dbg !7
  %799 = bitcast i32 %798 to float, !dbg !7
  %800 = select i1 %.not2, float %784, float %799, !dbg !7
  %801 = select i1 %.not2, float %799, float %790, !dbg !7
  %802 = select i1 %.not2, float %791, float %785, !dbg !7
  %803 = bitcast float %802 to i32, !dbg !7
  %804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %803), !dbg !7
  %805 = bitcast i32 %804 to float, !dbg !7
  %806 = select i1 %.not2, float %785, float %805, !dbg !7
  %807 = select i1 %.not2, float %805, float %791, !dbg !7
  %808 = extractelement <8 x float> %58, i64 0, !dbg !7
  %809 = extractelement <8 x float> %58, i64 1, !dbg !7
  %810 = extractelement <8 x float> %58, i64 2, !dbg !7
  %811 = extractelement <8 x float> %58, i64 3, !dbg !7
  %812 = select i1 %.not, float %809, float %808, !dbg !7
  %813 = bitcast float %812 to i32, !dbg !7
  %814 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %813), !dbg !7
  %815 = bitcast i32 %814 to float, !dbg !7
  %816 = select i1 %.not, float %808, float %815, !dbg !7
  %817 = select i1 %.not, float %815, float %809, !dbg !7
  %818 = select i1 %.not, float %811, float %810, !dbg !7
  %819 = bitcast float %818 to i32, !dbg !7
  %820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %819), !dbg !7
  %821 = bitcast i32 %820 to float, !dbg !7
  %822 = select i1 %.not, float %810, float %821, !dbg !7
  %823 = select i1 %.not, float %821, float %811, !dbg !7
  %824 = select i1 %.not2, float %822, float %816, !dbg !7
  %825 = bitcast float %824 to i32, !dbg !7
  %826 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %825), !dbg !7
  %827 = bitcast i32 %826 to float, !dbg !7
  %828 = select i1 %.not2, float %816, float %827, !dbg !7
  %829 = select i1 %.not2, float %827, float %822, !dbg !7
  %830 = select i1 %.not2, float %823, float %817, !dbg !7
  %831 = bitcast float %830 to i32, !dbg !7
  %832 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %831), !dbg !7
  %833 = bitcast i32 %832 to float, !dbg !7
  %834 = select i1 %.not2, float %817, float %833, !dbg !7
  %835 = select i1 %.not2, float %833, float %823, !dbg !7
  %836 = extractelement <8 x float> %55, i64 0, !dbg !7
  %837 = extractelement <8 x float> %55, i64 1, !dbg !7
  %838 = extractelement <8 x float> %55, i64 2, !dbg !7
  %839 = extractelement <8 x float> %55, i64 3, !dbg !7
  %840 = select i1 %.not, float %837, float %836, !dbg !7
  %841 = bitcast float %840 to i32, !dbg !7
  %842 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %841), !dbg !7
  %843 = bitcast i32 %842 to float, !dbg !7
  %844 = select i1 %.not, float %836, float %843, !dbg !7
  %845 = select i1 %.not, float %843, float %837, !dbg !7
  %846 = select i1 %.not, float %839, float %838, !dbg !7
  %847 = bitcast float %846 to i32, !dbg !7
  %848 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %847), !dbg !7
  %849 = bitcast i32 %848 to float, !dbg !7
  %850 = select i1 %.not, float %838, float %849, !dbg !7
  %851 = select i1 %.not, float %849, float %839, !dbg !7
  %852 = select i1 %.not2, float %850, float %844, !dbg !7
  %853 = bitcast float %852 to i32, !dbg !7
  %854 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %853), !dbg !7
  %855 = bitcast i32 %854 to float, !dbg !7
  %856 = select i1 %.not2, float %844, float %855, !dbg !7
  %857 = select i1 %.not2, float %855, float %850, !dbg !7
  %858 = select i1 %.not2, float %851, float %845, !dbg !7
  %859 = bitcast float %858 to i32, !dbg !7
  %860 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %859), !dbg !7
  %861 = bitcast i32 %860 to float, !dbg !7
  %862 = select i1 %.not2, float %845, float %861, !dbg !7
  %863 = select i1 %.not2, float %861, float %851, !dbg !7
  %864 = extractelement <8 x float> %50, i64 0, !dbg !7
  %865 = extractelement <8 x float> %50, i64 1, !dbg !7
  %866 = extractelement <8 x float> %50, i64 2, !dbg !7
  %867 = extractelement <8 x float> %50, i64 3, !dbg !7
  %868 = select i1 %.not, float %865, float %864, !dbg !7
  %869 = bitcast float %868 to i32, !dbg !7
  %870 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %869), !dbg !7
  %871 = bitcast i32 %870 to float, !dbg !7
  %872 = select i1 %.not, float %864, float %871, !dbg !7
  %873 = select i1 %.not, float %871, float %865, !dbg !7
  %874 = select i1 %.not, float %867, float %866, !dbg !7
  %875 = bitcast float %874 to i32, !dbg !7
  %876 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %875), !dbg !7
  %877 = bitcast i32 %876 to float, !dbg !7
  %878 = select i1 %.not, float %866, float %877, !dbg !7
  %879 = select i1 %.not, float %877, float %867, !dbg !7
  %880 = select i1 %.not2, float %878, float %872, !dbg !7
  %881 = bitcast float %880 to i32, !dbg !7
  %882 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %881), !dbg !7
  %883 = select i1 %.not2, float %879, float %873, !dbg !7
  %884 = bitcast float %883 to i32, !dbg !7
  %885 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %884), !dbg !7
  %886 = shufflevector <1 x float> %51, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %887 = extractelement <8 x float> %61, i64 4, !dbg !7
  %888 = extractelement <8 x float> %61, i64 5, !dbg !7
  %889 = extractelement <8 x float> %61, i64 6, !dbg !7
  %890 = extractelement <8 x float> %61, i64 7, !dbg !7
  %891 = select i1 %.not, float %888, float %887, !dbg !7
  %892 = bitcast float %891 to i32, !dbg !7
  %893 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %892), !dbg !7
  %894 = bitcast i32 %893 to float, !dbg !7
  %895 = select i1 %.not, float %887, float %894, !dbg !7
  %896 = select i1 %.not, float %894, float %888, !dbg !7
  %897 = select i1 %.not, float %890, float %889, !dbg !7
  %898 = bitcast float %897 to i32, !dbg !7
  %899 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %898), !dbg !7
  %900 = bitcast i32 %899 to float, !dbg !7
  %901 = select i1 %.not, float %889, float %900, !dbg !7
  %902 = select i1 %.not, float %900, float %890, !dbg !7
  %903 = select i1 %.not2, float %901, float %895, !dbg !7
  %904 = bitcast float %903 to i32, !dbg !7
  %905 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %904), !dbg !7
  %906 = bitcast i32 %905 to float, !dbg !7
  %907 = select i1 %.not2, float %895, float %906, !dbg !7
  %908 = select i1 %.not2, float %906, float %901, !dbg !7
  %909 = select i1 %.not2, float %902, float %896, !dbg !7
  %910 = bitcast float %909 to i32, !dbg !7
  %911 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %910), !dbg !7
  %912 = bitcast i32 %911 to float, !dbg !7
  %913 = select i1 %.not2, float %896, float %912, !dbg !7
  %914 = select i1 %.not2, float %912, float %902, !dbg !7
  %915 = extractelement <8 x float> %58, i64 4, !dbg !7
  %916 = extractelement <8 x float> %58, i64 5, !dbg !7
  %917 = extractelement <8 x float> %58, i64 6, !dbg !7
  %918 = extractelement <8 x float> %58, i64 7, !dbg !7
  %919 = select i1 %.not, float %916, float %915, !dbg !7
  %920 = bitcast float %919 to i32, !dbg !7
  %921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %920), !dbg !7
  %922 = bitcast i32 %921 to float, !dbg !7
  %923 = select i1 %.not, float %915, float %922, !dbg !7
  %924 = select i1 %.not, float %922, float %916, !dbg !7
  %925 = select i1 %.not, float %918, float %917, !dbg !7
  %926 = bitcast float %925 to i32, !dbg !7
  %927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %926), !dbg !7
  %928 = bitcast i32 %927 to float, !dbg !7
  %929 = select i1 %.not, float %917, float %928, !dbg !7
  %930 = select i1 %.not, float %928, float %918, !dbg !7
  %931 = select i1 %.not2, float %929, float %923, !dbg !7
  %932 = bitcast float %931 to i32, !dbg !7
  %933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %932), !dbg !7
  %934 = bitcast i32 %933 to float, !dbg !7
  %935 = select i1 %.not2, float %923, float %934, !dbg !7
  %936 = select i1 %.not2, float %934, float %929, !dbg !7
  %937 = select i1 %.not2, float %930, float %924, !dbg !7
  %938 = bitcast float %937 to i32, !dbg !7
  %939 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %938), !dbg !7
  %940 = bitcast i32 %939 to float, !dbg !7
  %941 = select i1 %.not2, float %924, float %940, !dbg !7
  %942 = select i1 %.not2, float %940, float %930, !dbg !7
  %943 = extractelement <8 x float> %55, i64 4, !dbg !7
  %944 = extractelement <8 x float> %55, i64 5, !dbg !7
  %945 = extractelement <8 x float> %55, i64 6, !dbg !7
  %946 = extractelement <8 x float> %55, i64 7, !dbg !7
  %947 = select i1 %.not, float %944, float %943, !dbg !7
  %948 = bitcast float %947 to i32, !dbg !7
  %949 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %948), !dbg !7
  %950 = bitcast i32 %949 to float, !dbg !7
  %951 = select i1 %.not, float %943, float %950, !dbg !7
  %952 = select i1 %.not, float %950, float %944, !dbg !7
  %953 = select i1 %.not, float %946, float %945, !dbg !7
  %954 = bitcast float %953 to i32, !dbg !7
  %955 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %954), !dbg !7
  %956 = bitcast i32 %955 to float, !dbg !7
  %957 = select i1 %.not, float %945, float %956, !dbg !7
  %958 = select i1 %.not, float %956, float %946, !dbg !7
  %959 = select i1 %.not2, float %957, float %951, !dbg !7
  %960 = bitcast float %959 to i32, !dbg !7
  %961 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %960), !dbg !7
  %962 = bitcast i32 %961 to float, !dbg !7
  %963 = select i1 %.not2, float %951, float %962, !dbg !7
  %964 = select i1 %.not2, float %962, float %957, !dbg !7
  %965 = select i1 %.not2, float %958, float %952, !dbg !7
  %966 = bitcast float %965 to i32, !dbg !7
  %967 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %966), !dbg !7
  %968 = bitcast i32 %967 to float, !dbg !7
  %969 = select i1 %.not2, float %952, float %968, !dbg !7
  %970 = select i1 %.not2, float %968, float %958, !dbg !7
  %971 = extractelement <8 x float> %50, i64 4, !dbg !7
  %972 = extractelement <8 x float> %50, i64 5, !dbg !7
  %973 = extractelement <8 x float> %50, i64 6, !dbg !7
  %974 = extractelement <8 x float> %50, i64 7, !dbg !7
  %975 = select i1 %.not, float %972, float %971, !dbg !7
  %976 = bitcast float %975 to i32, !dbg !7
  %977 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %976), !dbg !7
  %978 = bitcast i32 %977 to float, !dbg !7
  %979 = select i1 %.not, float %971, float %978, !dbg !7
  %980 = select i1 %.not, float %978, float %972, !dbg !7
  %981 = select i1 %.not, float %974, float %973, !dbg !7
  %982 = bitcast float %981 to i32, !dbg !7
  %983 = call i32 @llvm.amdgcn.ds.bpermute(i32 %780, i32 %982), !dbg !7
  %984 = bitcast i32 %983 to float, !dbg !7
  %985 = select i1 %.not, float %973, float %984, !dbg !7
  %986 = select i1 %.not, float %984, float %974, !dbg !7
  %987 = select i1 %.not2, float %985, float %979, !dbg !7
  %988 = bitcast float %987 to i32, !dbg !7
  %989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %988), !dbg !7
  %990 = select i1 %.not2, float %986, float %980, !dbg !7
  %991 = bitcast float %990 to i32, !dbg !7
  %992 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %991), !dbg !7
  br i1 true, label %993, label %1027, !dbg !21

993:                                              ; preds = %767
  %994 = bitcast i32 %989 to float, !dbg !7
  %995 = select i1 %.not2, float %979, float %994, !dbg !7
  %996 = insertelement <4 x float> poison, float %995, i64 0, !dbg !19
  %997 = bitcast i32 %992 to float, !dbg !7
  %998 = select i1 %.not2, float %980, float %997, !dbg !7
  %999 = insertelement <4 x float> %996, float %998, i64 1, !dbg !19
  %1000 = select i1 %.not2, float %994, float %985, !dbg !7
  %1001 = insertelement <4 x float> %999, float %1000, i64 2, !dbg !19
  %1002 = select i1 %.not2, float %997, float %986, !dbg !7
  %1003 = insertelement <4 x float> %1001, float %1002, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %886, !dbg !19
  %normalized = fmul <4 x float> %1003, %row.reciprocal, !dbg !19
  %1004 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1005 = extractelement <4 x half> %1004, i64 3, !dbg !20
  %1006 = extractelement <4 x half> %1004, i64 2, !dbg !20
  %1007 = extractelement <4 x half> %1004, i64 1, !dbg !20
  %1008 = extractelement <4 x half> %1004, i64 0, !dbg !20
  %1009 = bitcast i32 %882 to float, !dbg !7
  %1010 = select i1 %.not2, float %872, float %1009, !dbg !7
  %1011 = insertelement <4 x float> poison, float %1010, i64 0, !dbg !19
  %1012 = bitcast i32 %885 to float, !dbg !7
  %1013 = select i1 %.not2, float %873, float %1012, !dbg !7
  %1014 = insertelement <4 x float> %1011, float %1013, i64 1, !dbg !19
  %1015 = select i1 %.not2, float %1009, float %878, !dbg !7
  %1016 = insertelement <4 x float> %1014, float %1015, i64 2, !dbg !19
  %1017 = select i1 %.not2, float %1012, float %879, !dbg !7
  %1018 = insertelement <4 x float> %1016, float %1017, i64 3, !dbg !19
  %normalized14 = fmul <4 x float> %1018, %row.reciprocal, !dbg !19
  %1019 = fptrunc <4 x float> %normalized14 to <4 x half>, !dbg !20
  %1020 = extractelement <4 x half> %1019, i64 3, !dbg !20
  %1021 = extractelement <4 x half> %1019, i64 2, !dbg !20
  %1022 = extractelement <4 x half> %1019, i64 1, !dbg !20
  %1023 = extractelement <4 x half> %1019, i64 0, !dbg !20
  %1024 = add nsw i64 %8, %15, !dbg !21
  %1025 = shl nsw i64 %1024, 19, !dbg !21
  %1026 = and i64 %1025, 16252928, !dbg !21
  %address.thread246 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local247 = zext nneg i32 %address.thread246 to i64, !dbg !12
  %address.local248 = add nsw i64 %18, %address.local247, !dbg !21
  %address.local249 = lshr i64 %address.local247, 6, !dbg !12
  %address.local250 = shl nuw nsw i64 %address.local249, 4, !dbg !21
  %address.local251 = add nsw i64 %address.local248, %address.local250, !dbg !21
  %address.local252 = and i64 %address.local247, 240, !dbg !21
  %address.local253 = sub nsw i64 %address.local251, %address.local252, !dbg !21
  %address.local254 = add nsw i64 %address.local253, %19, !dbg !21
  %address.local255 = shl nuw nsw i64 %address.local254, 7, !dbg !21
  %address.local256 = add nuw nsw i64 %1026, %address.local255, !dbg !21
  %address.local257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local256, !dbg !21
  %address.local258 = lshr i64 %address.local247, 2, !dbg !12
  %address.local259 = and i64 %address.local258, 12, !dbg !12
  %address.local260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local257, i64 %address.local259, !dbg !21
  store half %1023, ptr addrspace(1) %address.local260, align 2, !dbg !21
  %address.local261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local256, !dbg !21
  %address.local262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local261, i64 %address.local259, !dbg !21
  %address.local263 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local262, i64 2, !dbg !21
  store half %1022, ptr addrspace(1) %address.local263, align 2, !dbg !21
  %address.local264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local256, !dbg !21
  %address.local265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local264, i64 %address.local259, !dbg !21
  %address.local266 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local265, i64 4, !dbg !21
  store half %1021, ptr addrspace(1) %address.local266, align 2, !dbg !21
  %address.local267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local256, !dbg !21
  %address.local268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local267, i64 %address.local259, !dbg !21
  %address.local269 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local268, i64 6, !dbg !21
  store half %1020, ptr addrspace(1) %address.local269, align 2, !dbg !21
  %address.local270 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local256, !dbg !21
  %address.local271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local270, i64 %address.local259, !dbg !21
  %address.local272 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local271, i64 32, !dbg !21
  store half %1008, ptr addrspace(1) %address.local272, align 2, !dbg !21
  %address.local273 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local256, !dbg !21
  %address.local274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local273, i64 %address.local259, !dbg !21
  %address.local275 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local274, i64 34, !dbg !21
  store half %1007, ptr addrspace(1) %address.local275, align 2, !dbg !21
  %address.local276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local256, !dbg !21
  %address.local277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local276, i64 %address.local259, !dbg !21
  %address.local278 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local277, i64 36, !dbg !21
  store half %1006, ptr addrspace(1) %address.local278, align 2, !dbg !21
  %address.local279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local256, !dbg !21
  %address.local280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local279, i64 %address.local259, !dbg !21
  %address.local281 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local280, i64 38, !dbg !21
  store half %1005, ptr addrspace(1) %address.local281, align 2, !dbg !21
  br label %1027, !dbg !21

1027:                                             ; preds = %993, %767
  br i1 true, label %1028, label %1048, !dbg !21

1028:                                             ; preds = %1027
  %1029 = insertelement <4 x float> poison, float %963, i64 0, !dbg !19
  %1030 = insertelement <4 x float> %1029, float %969, i64 1, !dbg !19
  %1031 = insertelement <4 x float> %1030, float %964, i64 2, !dbg !19
  %1032 = insertelement <4 x float> %1031, float %970, i64 3, !dbg !19
  %row.reciprocal15 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %886, !dbg !19
  %normalized16 = fmul <4 x float> %1032, %row.reciprocal15, !dbg !19
  %1033 = fptrunc <4 x float> %normalized16 to <4 x half>, !dbg !20
  %1034 = extractelement <4 x half> %1033, i64 3, !dbg !20
  %1035 = extractelement <4 x half> %1033, i64 2, !dbg !20
  %1036 = extractelement <4 x half> %1033, i64 1, !dbg !20
  %1037 = extractelement <4 x half> %1033, i64 0, !dbg !20
  %1038 = insertelement <4 x float> poison, float %856, i64 0, !dbg !19
  %1039 = insertelement <4 x float> %1038, float %862, i64 1, !dbg !19
  %1040 = insertelement <4 x float> %1039, float %857, i64 2, !dbg !19
  %1041 = insertelement <4 x float> %1040, float %863, i64 3, !dbg !19
  %normalized17 = fmul <4 x float> %1041, %row.reciprocal15, !dbg !19
  %1042 = fptrunc <4 x float> %normalized17 to <4 x half>, !dbg !20
  %1043 = extractelement <4 x half> %1042, i64 3, !dbg !20
  %1044 = extractelement <4 x half> %1042, i64 2, !dbg !20
  %1045 = extractelement <4 x half> %1042, i64 1, !dbg !20
  %1046 = extractelement <4 x half> %1042, i64 0, !dbg !20
  %1047 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread282 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local283 = zext nneg i32 %address.thread282 to i64, !dbg !12
  %address.local284 = add nsw i64 %18, %address.local283, !dbg !21
  %address.local285 = lshr i64 %address.local283, 6, !dbg !12
  %address.local286 = shl nuw nsw i64 %address.local285, 4, !dbg !21
  %address.local287 = add nsw i64 %address.local284, %address.local286, !dbg !21
  %address.local288 = and i64 %address.local283, 240, !dbg !21
  %address.local289 = sub nsw i64 %address.local287, %address.local288, !dbg !21
  %address.local290 = add nsw i64 %address.local289, %19, !dbg !21
  %address.local291 = shl nuw nsw i64 %address.local290, 7, !dbg !21
  %address.local292 = add nuw nsw i64 %1047, %address.local291, !dbg !21
  %address.local293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local292, !dbg !21
  %address.local294 = lshr i64 %address.local283, 2, !dbg !12
  %address.local295 = and i64 %address.local294, 12, !dbg !12
  %address.local296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local293, i64 %address.local295, !dbg !21
  %address.local297 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local296, i64 64, !dbg !21
  store half %1046, ptr addrspace(1) %address.local297, align 2, !dbg !21
  %address.local298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local292, !dbg !21
  %address.local299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local298, i64 %address.local295, !dbg !21
  %address.local300 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local299, i64 66, !dbg !21
  store half %1045, ptr addrspace(1) %address.local300, align 2, !dbg !21
  %address.local301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local292, !dbg !21
  %address.local302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local301, i64 %address.local295, !dbg !21
  %address.local303 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local302, i64 68, !dbg !21
  store half %1044, ptr addrspace(1) %address.local303, align 2, !dbg !21
  %address.local304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local292, !dbg !21
  %address.local305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local304, i64 %address.local295, !dbg !21
  %address.local306 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local305, i64 70, !dbg !21
  store half %1043, ptr addrspace(1) %address.local306, align 2, !dbg !21
  %address.local307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local292, !dbg !21
  %address.local308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local307, i64 %address.local295, !dbg !21
  %address.local309 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local308, i64 96, !dbg !21
  store half %1037, ptr addrspace(1) %address.local309, align 2, !dbg !21
  %address.local310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local292, !dbg !21
  %address.local311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local310, i64 %address.local295, !dbg !21
  %address.local312 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local311, i64 98, !dbg !21
  store half %1036, ptr addrspace(1) %address.local312, align 2, !dbg !21
  %address.local313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local292, !dbg !21
  %address.local314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local313, i64 %address.local295, !dbg !21
  %address.local315 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local314, i64 100, !dbg !21
  store half %1035, ptr addrspace(1) %address.local315, align 2, !dbg !21
  %address.local316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local292, !dbg !21
  %address.local317 = or i64 %address.local294, 51, !dbg !21
  %address.local318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local316, i64 %address.local317, !dbg !21
  store half %1034, ptr addrspace(1) %address.local318, align 2, !dbg !21
  br label %1048, !dbg !21

1048:                                             ; preds = %1028, %1027
  br i1 true, label %1049, label %1069, !dbg !21

1049:                                             ; preds = %1048
  %1050 = insertelement <4 x float> poison, float %935, i64 0, !dbg !19
  %1051 = insertelement <4 x float> %1050, float %941, i64 1, !dbg !19
  %1052 = insertelement <4 x float> %1051, float %936, i64 2, !dbg !19
  %1053 = insertelement <4 x float> %1052, float %942, i64 3, !dbg !19
  %row.reciprocal18 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %886, !dbg !19
  %normalized19 = fmul <4 x float> %1053, %row.reciprocal18, !dbg !19
  %1054 = fptrunc <4 x float> %normalized19 to <4 x half>, !dbg !20
  %1055 = extractelement <4 x half> %1054, i64 3, !dbg !20
  %1056 = extractelement <4 x half> %1054, i64 2, !dbg !20
  %1057 = extractelement <4 x half> %1054, i64 1, !dbg !20
  %1058 = extractelement <4 x half> %1054, i64 0, !dbg !20
  %1059 = insertelement <4 x float> poison, float %828, i64 0, !dbg !19
  %1060 = insertelement <4 x float> %1059, float %834, i64 1, !dbg !19
  %1061 = insertelement <4 x float> %1060, float %829, i64 2, !dbg !19
  %1062 = insertelement <4 x float> %1061, float %835, i64 3, !dbg !19
  %normalized20 = fmul <4 x float> %1062, %row.reciprocal18, !dbg !19
  %1063 = fptrunc <4 x float> %normalized20 to <4 x half>, !dbg !20
  %1064 = extractelement <4 x half> %1063, i64 3, !dbg !20
  %1065 = extractelement <4 x half> %1063, i64 2, !dbg !20
  %1066 = extractelement <4 x half> %1063, i64 1, !dbg !20
  %1067 = extractelement <4 x half> %1063, i64 0, !dbg !20
  %1068 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread319 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local320 = zext nneg i32 %address.thread319 to i64, !dbg !12
  %address.local321 = add nsw i64 %18, %address.local320, !dbg !21
  %address.local322 = lshr i64 %address.local320, 6, !dbg !12
  %address.local323 = shl nuw nsw i64 %address.local322, 4, !dbg !21
  %address.local324 = add nsw i64 %address.local321, %address.local323, !dbg !21
  %address.local325 = and i64 %address.local320, 240, !dbg !21
  %address.local326 = sub nsw i64 %address.local324, %address.local325, !dbg !21
  %address.local327 = add nsw i64 %address.local326, %19, !dbg !21
  %address.local328 = shl nuw nsw i64 %address.local327, 7, !dbg !21
  %address.local329 = add nuw nsw i64 %1068, %address.local328, !dbg !21
  %address.local330 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local329, !dbg !21
  %address.local331 = lshr i64 %address.local320, 2, !dbg !12
  %address.local332 = and i64 %address.local331, 12, !dbg !12
  %address.local333 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local330, i64 %address.local332, !dbg !21
  %address.local334 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local333, i64 128, !dbg !21
  store half %1067, ptr addrspace(1) %address.local334, align 2, !dbg !21
  %address.local335 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local329, !dbg !21
  %address.local336 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local335, i64 %address.local332, !dbg !21
  %address.local337 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local336, i64 130, !dbg !21
  store half %1066, ptr addrspace(1) %address.local337, align 2, !dbg !21
  %address.local338 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local329, !dbg !21
  %address.local339 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local338, i64 %address.local332, !dbg !21
  %address.local340 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local339, i64 132, !dbg !21
  store half %1065, ptr addrspace(1) %address.local340, align 2, !dbg !21
  %address.local341 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local329, !dbg !21
  %address.local342 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local341, i64 %address.local332, !dbg !21
  %address.local343 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local342, i64 134, !dbg !21
  store half %1064, ptr addrspace(1) %address.local343, align 2, !dbg !21
  %address.local344 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local329, !dbg !21
  %address.local345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local344, i64 %address.local332, !dbg !21
  %address.local346 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local345, i64 160, !dbg !21
  store half %1058, ptr addrspace(1) %address.local346, align 2, !dbg !21
  %address.local347 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local329, !dbg !21
  %address.local348 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local347, i64 %address.local332, !dbg !21
  %address.local349 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local348, i64 162, !dbg !21
  store half %1057, ptr addrspace(1) %address.local349, align 2, !dbg !21
  %address.local350 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local329, !dbg !21
  %address.local351 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local350, i64 %address.local332, !dbg !21
  %address.local352 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local351, i64 164, !dbg !21
  store half %1056, ptr addrspace(1) %address.local352, align 2, !dbg !21
  %address.local353 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local329, !dbg !21
  %address.local354 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local353, i64 %address.local332, !dbg !21
  %address.local355 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local354, i64 166, !dbg !21
  store half %1055, ptr addrspace(1) %address.local355, align 2, !dbg !21
  br label %1069, !dbg !21

1069:                                             ; preds = %1049, %1048
  br i1 true, label %1070, label %1090, !dbg !21

1070:                                             ; preds = %1069
  %1071 = insertelement <4 x float> poison, float %907, i64 0, !dbg !19
  %1072 = insertelement <4 x float> %1071, float %913, i64 1, !dbg !19
  %1073 = insertelement <4 x float> %1072, float %908, i64 2, !dbg !19
  %1074 = insertelement <4 x float> %1073, float %914, i64 3, !dbg !19
  %row.reciprocal21 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %886, !dbg !19
  %normalized22 = fmul <4 x float> %1074, %row.reciprocal21, !dbg !19
  %1075 = fptrunc <4 x float> %normalized22 to <4 x half>, !dbg !20
  %1076 = extractelement <4 x half> %1075, i64 3, !dbg !20
  %1077 = extractelement <4 x half> %1075, i64 2, !dbg !20
  %1078 = extractelement <4 x half> %1075, i64 1, !dbg !20
  %1079 = extractelement <4 x half> %1075, i64 0, !dbg !20
  %1080 = insertelement <4 x float> poison, float %800, i64 0, !dbg !19
  %1081 = insertelement <4 x float> %1080, float %806, i64 1, !dbg !19
  %1082 = insertelement <4 x float> %1081, float %801, i64 2, !dbg !19
  %1083 = insertelement <4 x float> %1082, float %807, i64 3, !dbg !19
  %normalized23 = fmul <4 x float> %1083, %row.reciprocal21, !dbg !19
  %1084 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !20
  %1085 = extractelement <4 x half> %1084, i64 3, !dbg !20
  %1086 = extractelement <4 x half> %1084, i64 2, !dbg !20
  %1087 = extractelement <4 x half> %1084, i64 1, !dbg !20
  %1088 = extractelement <4 x half> %1084, i64 0, !dbg !20
  %1089 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread356 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local357 = zext nneg i32 %address.thread356 to i64, !dbg !12
  %address.local358 = add nsw i64 %18, %address.local357, !dbg !21
  %address.local359 = lshr i64 %address.local357, 6, !dbg !12
  %address.local360 = shl nuw nsw i64 %address.local359, 4, !dbg !21
  %address.local361 = add nsw i64 %address.local358, %address.local360, !dbg !21
  %address.local362 = and i64 %address.local357, 240, !dbg !21
  %address.local363 = sub nsw i64 %address.local361, %address.local362, !dbg !21
  %address.local364 = add nsw i64 %address.local363, %19, !dbg !21
  %address.local365 = shl nuw nsw i64 %address.local364, 7, !dbg !21
  %address.local366 = add nuw nsw i64 %1089, %address.local365, !dbg !21
  %address.local367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local366, !dbg !21
  %address.local368 = lshr i64 %address.local357, 2, !dbg !12
  %address.local369 = and i64 %address.local368, 12, !dbg !12
  %address.local370 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local367, i64 %address.local369, !dbg !21
  %address.local371 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local370, i64 192, !dbg !21
  store half %1088, ptr addrspace(1) %address.local371, align 2, !dbg !21
  %address.local372 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local366, !dbg !21
  %address.local373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local372, i64 %address.local369, !dbg !21
  %address.local374 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local373, i64 194, !dbg !21
  store half %1087, ptr addrspace(1) %address.local374, align 2, !dbg !21
  %address.local375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local366, !dbg !21
  %address.local376 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local375, i64 %address.local369, !dbg !21
  %address.local377 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local376, i64 196, !dbg !21
  store half %1086, ptr addrspace(1) %address.local377, align 2, !dbg !21
  %address.local378 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local366, !dbg !21
  %address.local379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local378, i64 %address.local369, !dbg !21
  %address.local380 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local379, i64 198, !dbg !21
  store half %1085, ptr addrspace(1) %address.local380, align 2, !dbg !21
  %address.local381 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local366, !dbg !21
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local381, i64 %address.local369, !dbg !21
  %address.local383 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local382, i64 224, !dbg !21
  store half %1079, ptr addrspace(1) %address.local383, align 2, !dbg !21
  %address.local384 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local366, !dbg !21
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local384, i64 %address.local369, !dbg !21
  %address.local386 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local385, i64 226, !dbg !21
  store half %1078, ptr addrspace(1) %address.local386, align 2, !dbg !21
  %address.local387 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local366, !dbg !21
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local387, i64 %address.local369, !dbg !21
  %address.local389 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local388, i64 228, !dbg !21
  store half %1077, ptr addrspace(1) %address.local389, align 2, !dbg !21
  %address.local390 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local366, !dbg !21
  %address.local391 = or i64 %address.local368, 115, !dbg !21
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local390, i64 %address.local391, !dbg !21
  store half %1076, ptr addrspace(1) %address.local392, align 2, !dbg !21
  br label %1090, !dbg !21

1090:                                             ; preds = %1070, %1069
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/baseline")
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
