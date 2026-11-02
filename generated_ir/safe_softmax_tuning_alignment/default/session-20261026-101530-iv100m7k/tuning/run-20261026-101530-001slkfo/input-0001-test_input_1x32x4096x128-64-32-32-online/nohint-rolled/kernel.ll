; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/nohint-rolled/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/nohint-rolled/input.mlir"

@shm_0 = addrspace(3) global [6144 x i8] undef, align 16

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
  %22 = lshr i64 %20, 2, !dbg !13
  %23 = and i64 %22, 12, !dbg !13
  %24 = and i32 %19, 16, !dbg !14
  %.not = icmp eq i32 %24, 0, !dbg !14
  %25 = and i32 %19, 32, !dbg !14
  %.not2 = icmp eq i32 %25, 0, !dbg !14
  %26 = or disjoint i64 %9, 63, !dbg !15
  %27 = add nsw i64 %9, 64, !dbg !15
  %28 = icmp uge i64 %27, %26, !dbg !15
  %29 = shl nsw i64 %8, 1, !dbg !7
  %30 = add nsw i64 %29, 2, !dbg !7
  br label %31, !dbg !7

31:                                               ; preds = %499, %4
  %32 = phi i64 [ %529, %499 ], [ 0, %4 ], !dbg !7
  %33 = phi <4 x float> [ %410, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %34 = phi <4 x float> [ %415, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %35 = phi <8 x float> [ %524, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %36 = phi <1 x float> [ %525, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %37 = phi <1 x float> [ %312, %499 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %38 = phi <4 x float> [ %457, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %39 = phi <4 x float> [ %462, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %40 = phi <8 x float> [ %526, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %41 = phi <4 x float> [ %504, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %42 = phi <4 x float> [ %509, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %43 = phi <8 x float> [ %527, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %44 = phi <4 x float> [ %518, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %45 = phi <4 x float> [ %523, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %46 = phi <8 x float> [ %528, %499 ], [ zeroinitializer, %4 ], !dbg !7
  %47 = icmp slt i64 %32, %30, !dbg !7
  br i1 %47, label %48, label %530, !dbg !7

48:                                               ; preds = %31
  %49 = shl nsw i64 %32, 5, !dbg !7
  %50 = lshr i64 %32, 7, !dbg !17
  %51 = shl i64 %50, 12, !dbg !17
  %52 = sub i64 %49, %51, !dbg !17
  br label %53, !dbg !14

53:                                               ; preds = %121, %48
  %54 = phi i64 [ %135, %121 ], [ 0, %48 ], !dbg !14
  %55 = phi <4 x float> [ %129, %121 ], [ zeroinitializer, %48 ], !dbg !14
  %56 = phi <4 x float> [ %133, %121 ], [ zeroinitializer, %48 ], !dbg !14
  %57 = phi <8 x float> [ %134, %121 ], [ zeroinitializer, %48 ], !dbg !14
  %58 = icmp samesign ult i64 %54, 4, !dbg !14
  br i1 %58, label %59, label %136, !dbg !14

59:                                               ; preds = %53
  %60 = shl nuw nsw i64 %54, 5, !dbg !14
  %.idx = shl nuw nsw i64 %15, 20, !dbg !12
  %61 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx11 = shl nsw i64 %18, 8, !dbg !12
  %62 = getelementptr i8, ptr addrspace(1) %61, i64 %.idx11, !dbg !12
  %63 = getelementptr [2 x i8], ptr addrspace(1) %62, i64 %60, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local31 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local32 = and i64 %address.local31, 1536, !dbg !12
  %address.local33 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local32, !dbg !12
  %address.local34 = ptrtoint ptr addrspace(3) %address.local33 to i64, !dbg !12
  %address.local35 = trunc i64 %address.local34 to i32, !dbg !12
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local35), !dbg !12
  %65 = zext i32 %64 to i64, !dbg !12
  %66 = inttoptr i64 %65 to ptr addrspace(3), !dbg !12
  %address.local36 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local37 = and i64 %address.local36, 8064, !dbg !12
  %address.local38 = getelementptr [2 x i8], ptr addrspace(1) %63, i64 %address.local37, !dbg !12
  %address.local39 = lshr i64 %address.local, 3, !dbg !12
  %address.local40 = and i64 %address.local39, 24, !dbg !12
  %address.local41 = getelementptr [2 x i8], ptr addrspace(1) %address.local38, i64 %address.local40, !dbg !12
  %67 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %68 = ptrtoint ptr addrspace(1) %address.local41 to i64, !dbg !12
  %69 = sub i64 %68, %67, !dbg !12
  %dma.buffer.offset280 = trunc i64 %69 to i32, !dbg !12
  %70 = trunc i64 %67 to i32, !dbg !12
  %71 = insertelement <4 x i32> poison, i32 %70, i64 0, !dbg !12
  %72 = lshr i64 %67, 32, !dbg !12
  %73 = trunc i64 %72 to i32, !dbg !12
  %74 = insertelement <4 x i32> %71, i32 %73, i64 1, !dbg !12
  %75 = insertelement <4 x i32> %74, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc281 = insertelement <4 x i32> %75, i32 131072, i64 3, !dbg !12
  %76 = ptrtoint ptr addrspace(3) %66 to i32, !dbg !12
  %dma.buffer.lds282 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %76), !dbg !12
  %77 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds282, i32 %dma.buffer.offset280, <4 x i32> %dma.buffer.rsrc281) #10, !dbg !12
  %78 = shl nuw nsw i64 %54, 12, !dbg !17
  %79 = add nuw i64 %32, %78, !dbg !17
  %.lobit12 = ashr i64 %79, 63, !dbg !17
  %80 = xor i64 %79, %.lobit12, !dbg !17
  %81 = sdiv i64 %80, 16384, !dbg !17
  %.lobit13 = ashr i64 %79, 63, !dbg !17
  %82 = xor i64 %81, %.lobit13, !dbg !17
  %83 = add nsw i64 %82, %6, !dbg !17
  %84 = add nuw nsw i64 %60, %50, !dbg !17
  %85 = and i64 %84, 127, !dbg !17
  %86 = shl i64 %83, 19, !dbg !17
  %87 = and i64 %86, 16252928, !dbg !17
  %88 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %87, !dbg !17
  %89 = getelementptr [2 x i8], ptr addrspace(1) %88, i64 %85, !dbg !17
  %.idx14 = shl i64 %52, 8, !dbg !17
  %90 = getelementptr i8, ptr addrspace(1) %89, i64 %.idx14, !dbg !17
  br label %91, !dbg !17

91:                                               ; preds = %94, %59
  %92 = phi i64 [ %120, %94 ], [ %20, %59 ], !dbg !17
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
  %120 = or disjoint i64 %20, 256, !dbg !17
  br label %91, !dbg !17

121:                                              ; preds = %91
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.thread42 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !13, !range !39, !frisk.register_lifetimes !1
  %address.local43 = lshr i32 %address.thread42, 2, !dbg !13
  %address.local44 = and i32 %address.local43, 4, !dbg !13
  %address.local45 = shl nuw nsw i32 %address.thread42, 1, !dbg !13
  %address.local46 = and i32 %address.local45, 384, !dbg !13
  %address.local47 = or disjoint i32 %address.local44, %address.local46, !dbg !13
  %address.local48 = shl nuw nsw i32 %address.thread42, 3, !dbg !13
  %address.local49 = and i32 %address.local48, 120, !dbg !13
  %address.local50 = or disjoint i32 %address.local47, %address.local49, !dbg !13
  %address.local51 = shl nuw nsw i32 %address.thread42, 4, !dbg !13
  %address.local52 = and i32 %address.local51, 512, !dbg !13
  %address.local53 = or disjoint i32 %address.local50, %address.local52, !dbg !13
  %address.local54 = zext nneg i32 %address.local53 to i64, !dbg !13
  %address.local55 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local54, !dbg !13
  %122 = load <4 x half>, ptr addrspace(3) %address.local55, align 2, !dbg !13
  %address.local56 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local54, !dbg !13
  %address.local57 = getelementptr i8, ptr addrspace(3) %address.local56, i64 2048, !dbg !13
  %123 = load <4 x half>, ptr addrspace(3) %address.local57, align 2, !dbg !13
  %124 = fmul <4 x half> %122, splat (half 0xH3015), !dbg !13
  %125 = fmul <4 x half> %123, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local58 = lshr i32 %address.thread42, 2, !dbg !8
  %address.local59 = and i32 %address.local58, 4, !dbg !8
  %address.local60 = shl nuw nsw i32 %address.thread42, 3, !dbg !8
  %address.local61 = and i32 %address.local60, 376, !dbg !8
  %address.local62 = or disjoint i32 %address.local59, %address.local61, !dbg !8
  %address.local63 = zext nneg i32 %address.local62 to i64, !dbg !8
  %address.local64 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local63, !dbg !8
  %126 = load <4 x half>, ptr addrspace(3) %address.local64, align 2, !dbg !8
  %address.local65 = lshr i32 %address.thread42, 2, !dbg !8
  %address.local66 = and i32 %address.local65, 4, !dbg !8
  %address.local67 = shl nuw nsw i32 %address.thread42, 3, !dbg !8
  %address.local68 = and i32 %address.local67, 376, !dbg !8
  %address.local69 = or disjoint i32 %address.local66, %address.local68, !dbg !8
  %address.local70 = zext nneg i32 %address.local69 to i64, !dbg !8
  %address.local71 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local70, !dbg !8
  %address.local72 = getelementptr i8, ptr addrspace(3) %address.local71, i64 1024, !dbg !8
  %127 = load <4 x half>, ptr addrspace(3) %address.local72, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %126, <4 x float> %55) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %127, <4 x float> %128) #9, !dbg !8
  %address.local73 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local63, !dbg !8
  %address.local74 = getelementptr i8, ptr addrspace(3) %address.local73, i64 256, !dbg !8
  %130 = load <4 x half>, ptr addrspace(3) %address.local74, align 2, !dbg !8
  %address.local75 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local70, !dbg !8
  %address.local76 = getelementptr i8, ptr addrspace(3) %address.local75, i64 1280, !dbg !8
  %131 = load <4 x half>, ptr addrspace(3) %address.local76, align 2, !dbg !8
  %132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %130, <4 x float> %56) #9, !dbg !8
  %133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %131, <4 x float> %132) #9, !dbg !8
  %134 = shufflevector <4 x float> %129, <4 x float> %133, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %135 = add nuw nsw i64 %54, 1, !dbg !14
  br label %53, !dbg !14

136:                                              ; preds = %53
  %137 = add nsw i64 %50, %6, !dbg !18
  %138 = shl i64 %137, 19, !dbg !18
  %139 = and i64 %138, 16252928, !dbg !18
  %140 = shl nsw i64 %52, 7, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %141 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %139, !dbg !18
  %142 = getelementptr [2 x i8], ptr addrspace(1) %141, i64 %140, !dbg !18
  br label %143, !dbg !18

143:                                              ; preds = %146, %136
  %144 = phi i64 [ %172, %146 ], [ %20, %136 ], !dbg !18
  %145 = icmp samesign ult i64 %144, 128, !dbg !18
  br i1 %145, label %146, label %173, !dbg !18

146:                                              ; preds = %143
  %147 = shl nuw nsw i64 %144, 3, !dbg !18
  %148 = and i64 %147, 24, !dbg !18
  %149 = shl nuw nsw i64 %144, 3, !dbg !18
  %150 = and i64 %149, 512, !dbg !18
  %151 = shl nuw nsw i64 %144, 5, !dbg !18
  %152 = and i64 %151, 3968, !dbg !18
  %153 = getelementptr [2 x i8], ptr addrspace(1) %142, i64 %152, !dbg !18
  %154 = getelementptr [2 x i8], ptr addrspace(1) %153, i64 %148, !dbg !18
  %155 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %150, !dbg !18
  %156 = ptrtoint ptr addrspace(3) %155 to i64, !dbg !18
  %157 = trunc i64 %156 to i32, !dbg !18
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %157), !dbg !18
  %159 = zext i32 %158 to i64, !dbg !18
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !18
  %161 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %162 = ptrtoint ptr addrspace(1) %154 to i64, !dbg !18
  %163 = sub i64 %162, %161, !dbg !18
  %dma.buffer.offset283 = trunc i64 %163 to i32, !dbg !18
  %164 = trunc i64 %161 to i32, !dbg !18
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !18
  %166 = lshr i64 %161, 32, !dbg !18
  %167 = trunc i64 %166 to i32, !dbg !18
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !18
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc284 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !18
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !18
  %dma.buffer.lds285 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !18
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds285, i32 %dma.buffer.offset283, <4 x i32> %dma.buffer.rsrc284) #10, !dbg !18
  %172 = or disjoint i64 %20, 256, !dbg !18
  br label %143, !dbg !18

173:                                              ; preds = %143
  %174 = extractelement <8 x float> %57, i64 0, !dbg !14
  %175 = extractelement <8 x float> %57, i64 1, !dbg !14
  %176 = extractelement <8 x float> %57, i64 2, !dbg !14
  %177 = extractelement <8 x float> %57, i64 3, !dbg !14
  %178 = select i1 %.not, float %175, float %174, !dbg !14
  %179 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !14
  %180 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %179), !dbg !14
  %181 = add nuw nsw i32 %180, 64, !dbg !14
  %182 = xor i32 %180, 16, !dbg !14
  %183 = and i32 %181, 192, !dbg !14
  %184 = icmp samesign ult i32 %182, %183, !dbg !14
  %185 = select i1 %184, i32 %182, i32 %180, !dbg !14
  %186 = shl nuw nsw i32 %185, 2, !dbg !14
  %187 = bitcast float %178 to i32, !dbg !14
  %188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %186, i32 %187), !dbg !14
  %189 = bitcast i32 %188 to float, !dbg !14
  %190 = select i1 %.not, float %174, float %189, !dbg !14
  %191 = select i1 %.not, float %189, float %175, !dbg !14
  %192 = select i1 %.not, float %177, float %176, !dbg !14
  %193 = bitcast float %192 to i32, !dbg !14
  %194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %186, i32 %193), !dbg !14
  %195 = bitcast i32 %194 to float, !dbg !14
  %196 = select i1 %.not, float %176, float %195, !dbg !14
  %197 = select i1 %.not, float %195, float %177, !dbg !14
  %198 = select i1 %.not2, float %196, float %190, !dbg !14
  %199 = xor i32 %180, 32, !dbg !14
  %200 = icmp samesign ult i32 %199, %183, !dbg !14
  %201 = select i1 %200, i32 %199, i32 %180, !dbg !14
  %202 = shl nuw nsw i32 %201, 2, !dbg !14
  %203 = bitcast float %198 to i32, !dbg !14
  %204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %202, i32 %203), !dbg !14
  %205 = bitcast i32 %204 to float, !dbg !14
  %206 = select i1 %.not2, float %190, float %205, !dbg !14
  %207 = select i1 %.not2, float %205, float %196, !dbg !14
  %208 = select i1 %.not2, float %197, float %191, !dbg !14
  %209 = bitcast float %208 to i32, !dbg !14
  %210 = call i32 @llvm.amdgcn.ds.bpermute(i32 %202, i32 %209), !dbg !14
  %211 = bitcast i32 %210 to float, !dbg !14
  %212 = select i1 %.not2, float %191, float %211, !dbg !14
  %213 = select i1 %.not2, float %211, float %197, !dbg !14
  %214 = extractelement <8 x float> %57, i64 4, !dbg !14
  %215 = extractelement <8 x float> %57, i64 5, !dbg !14
  %216 = extractelement <8 x float> %57, i64 6, !dbg !14
  %217 = extractelement <8 x float> %57, i64 7, !dbg !14
  %218 = select i1 %.not, float %215, float %214, !dbg !14
  %219 = bitcast float %218 to i32, !dbg !14
  %220 = call i32 @llvm.amdgcn.ds.bpermute(i32 %186, i32 %219), !dbg !14
  %221 = bitcast i32 %220 to float, !dbg !14
  %222 = select i1 %.not, float %214, float %221, !dbg !14
  %223 = select i1 %.not, float %221, float %215, !dbg !14
  %224 = select i1 %.not, float %217, float %216, !dbg !14
  %225 = bitcast float %224 to i32, !dbg !14
  %226 = call i32 @llvm.amdgcn.ds.bpermute(i32 %186, i32 %225), !dbg !14
  %227 = bitcast i32 %226 to float, !dbg !14
  %228 = select i1 %.not, float %216, float %227, !dbg !14
  %229 = select i1 %.not, float %227, float %217, !dbg !14
  %230 = select i1 %.not2, float %228, float %222, !dbg !14
  %231 = bitcast float %230 to i32, !dbg !14
  %232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %202, i32 %231), !dbg !14
  %233 = bitcast i32 %232 to float, !dbg !14
  %234 = select i1 %.not2, float %222, float %233, !dbg !14
  %235 = select i1 %.not2, float %233, float %228, !dbg !14
  %236 = select i1 %.not2, float %229, float %223, !dbg !14
  %237 = bitcast float %236 to i32, !dbg !14
  %238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %202, i32 %237), !dbg !14
  %239 = bitcast i32 %238 to float, !dbg !14
  %240 = select i1 %.not2, float %223, float %239, !dbg !14
  %241 = select i1 %.not2, float %239, float %229, !dbg !14
  %242 = or disjoint i64 %49, 31, !dbg !15
  %243 = icmp uge i64 %9, %242, !dbg !15
  %244 = and i1 %28, %243, !dbg !15
  br i1 %244, label %245, label %246, !dbg !15

245:                                              ; preds = %173
  br label %272, !dbg !15

246:                                              ; preds = %173
  %247 = add nsw i64 %9, %20, !dbg !15
  %248 = shl nuw nsw i64 %21, 4, !dbg !15
  %249 = add nsw i64 %247, %248, !dbg !15
  %250 = and i32 %19, 240, !dbg !15
  %index.compact = zext nneg i32 %250 to i64, !dbg !15
  %251 = sub nsw i64 %249, %index.compact, !dbg !15
  %252 = add nsw i64 %251, 1, !dbg !22
  %253 = or disjoint i64 %49, %23, !dbg !15
  %254 = lshr i64 %252, 32, !dbg !23
  %compare.high = trunc i64 %254 to i32, !dbg !23
  %255 = lshr i64 %253, 32, !dbg !23
  %compare.high252 = trunc i64 %255 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high252, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high252, !dbg !23
  %compare.low.base = trunc i64 %252 to i32, !dbg !23
  %compare.low.base253 = trunc i64 %253 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base253, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %256 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base253, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low254 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result255 = select i1 %compare.high.equal, i1 %compare.low254, i1 %compare.high.order, !dbg !23
  %257 = select i1 %compare.result255, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base256 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base253, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local257 = or i32 %compare.local.base256, 2, !dbg !23
  %compare.low258 = icmp ugt i32 %compare.low.base, %compare.local257, !dbg !23
  %compare.result259 = select i1 %compare.high.equal, i1 %compare.low258, i1 %compare.high.order, !dbg !23
  %258 = select i1 %compare.result259, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base260 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base253, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local261 = or i32 %compare.local.base260, 3, !dbg !23
  %compare.low262 = icmp ugt i32 %compare.low.base, %compare.local261, !dbg !23
  %compare.result263 = select i1 %compare.high.equal, i1 %compare.low262, i1 %compare.high.order, !dbg !23
  %259 = select i1 %compare.result263, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base264 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base253, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local265 = or i32 %compare.local.base264, 16, !dbg !23
  %compare.low266 = icmp ugt i32 %compare.low.base, %compare.local265, !dbg !23
  %compare.result267 = select i1 %compare.high.equal, i1 %compare.low266, i1 %compare.high.order, !dbg !23
  %260 = select i1 %compare.result267, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base268 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base253, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local269 = or i32 %compare.local.base268, 17, !dbg !23
  %compare.low270 = icmp ugt i32 %compare.low.base, %compare.local269, !dbg !23
  %compare.result271 = select i1 %compare.high.equal, i1 %compare.low270, i1 %compare.high.order, !dbg !23
  %261 = select i1 %compare.result271, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base272 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base253, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local273 = or i32 %compare.local.base272, 18, !dbg !23
  %compare.low274 = icmp ugt i32 %compare.low.base, %compare.local273, !dbg !23
  %compare.result275 = select i1 %compare.high.equal, i1 %compare.low274, i1 %compare.high.order, !dbg !23
  %262 = select i1 %compare.result275, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base276 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base253, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local277 = or i32 %compare.local.base276, 19, !dbg !23
  %compare.low278 = icmp ugt i32 %compare.low.base, %compare.local277, !dbg !23
  %compare.result279 = select i1 %compare.high.equal, i1 %compare.low278, i1 %compare.high.order, !dbg !23
  %263 = select i1 %compare.result279, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %264 = insertelement <8 x float> poison, float %256, i64 0, !dbg !15
  %265 = insertelement <8 x float> %264, float %257, i64 1, !dbg !15
  %266 = insertelement <8 x float> %265, float %258, i64 2, !dbg !15
  %267 = insertelement <8 x float> %266, float %259, i64 3, !dbg !15
  %268 = insertelement <8 x float> %267, float %260, i64 4, !dbg !15
  %269 = insertelement <8 x float> %268, float %261, i64 5, !dbg !15
  %270 = insertelement <8 x float> %269, float %262, i64 6, !dbg !15
  %271 = insertelement <8 x float> %270, float %263, i64 7, !dbg !15
  br label %272, !dbg !15

272:                                              ; preds = %245, %246
  %273 = phi <8 x float> [ %271, %246 ], [ zeroinitializer, %245 ], !dbg !15
  br label %274, !dbg !15

274:                                              ; preds = %272
  %275 = insertelement <4 x float> poison, float %206, i64 0, !dbg !25
  %276 = insertelement <4 x float> %275, float %212, i64 1, !dbg !25
  %277 = insertelement <4 x float> %276, float %207, i64 2, !dbg !25
  %278 = insertelement <4 x float> %277, float %213, i64 3, !dbg !25
  %279 = shufflevector <8 x float> %273, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %280 = fadd <4 x float> %278, %279, !dbg !25
  %281 = extractelement <4 x float> %280, i64 0, !dbg !25
  %282 = extractelement <4 x float> %280, i64 1, !dbg !25
  %283 = extractelement <4 x float> %280, i64 2, !dbg !25
  %284 = extractelement <4 x float> %280, i64 3, !dbg !25
  %285 = insertelement <4 x float> poison, float %234, i64 0, !dbg !25
  %286 = insertelement <4 x float> %285, float %240, i64 1, !dbg !25
  %287 = insertelement <4 x float> %286, float %235, i64 2, !dbg !25
  %288 = insertelement <4 x float> %287, float %241, i64 3, !dbg !25
  %289 = shufflevector <8 x float> %273, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %290 = fadd <4 x float> %288, %289, !dbg !25
  %291 = extractelement <4 x float> %290, i64 0, !dbg !25
  %292 = extractelement <4 x float> %290, i64 1, !dbg !25
  %293 = extractelement <4 x float> %290, i64 2, !dbg !25
  %294 = extractelement <4 x float> %290, i64 3, !dbg !25
  %295 = call float @llvm.maxnum.f32(float %281, float 0xFFF0000000000000), !dbg !26
  %296 = call float @llvm.maxnum.f32(float %295, float %282), !dbg !26
  %297 = call float @llvm.maxnum.f32(float %296, float %283), !dbg !26
  %298 = call float @llvm.maxnum.f32(float %297, float %284), !dbg !26
  %299 = call float @llvm.maxnum.f32(float %298, float %291), !dbg !26
  %300 = call float @llvm.maxnum.f32(float %299, float %292), !dbg !26
  %301 = call float @llvm.maxnum.f32(float %300, float %293), !dbg !26
  %302 = call float @llvm.maxnum.f32(float %301, float %294), !dbg !26
  %303 = bitcast float %302 to i32, !dbg !26
  %304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %186, i32 %303), !dbg !26
  %305 = bitcast i32 %304 to float, !dbg !26
  %306 = call float @llvm.maxnum.f32(float %302, float %305), !dbg !26
  %307 = bitcast float %306 to i32, !dbg !26
  %308 = call i32 @llvm.amdgcn.ds.bpermute(i32 %202, i32 %307), !dbg !26
  %309 = bitcast i32 %308 to float, !dbg !26
  %310 = call float @llvm.maxnum.f32(float %306, float %309), !dbg !26
  %311 = insertelement <1 x float> poison, float %310, i64 0, !dbg !26
  %312 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %311, <1 x float> %37), !dbg !26
  %313 = shufflevector <1 x float> %312, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %314 = fsub <4 x float> %280, %313, !dbg !27
  %315 = fsub <4 x float> %290, %313, !dbg !27
  %316 = fsub <1 x float> %37, %312, !dbg !28
  %317 = extractelement <1 x float> %316, i64 0, !dbg !29
  %318 = call float @llvm.amdgcn.exp2.f32(float %317), !dbg !29
  %319 = extractelement <1 x float> %36, i64 0, !dbg !30
  %320 = fmul float %319, %318, !dbg !30
  %321 = extractelement <4 x float> %314, i64 0, !dbg !31
  %322 = call float @llvm.amdgcn.exp2.f32(float %321), !dbg !31
  %323 = insertelement <4 x float> poison, float %322, i64 0, !dbg !33
  %324 = extractelement <4 x float> %314, i64 1, !dbg !31
  %325 = call float @llvm.amdgcn.exp2.f32(float %324), !dbg !31
  %326 = insertelement <4 x float> %323, float %325, i64 1, !dbg !33
  %327 = fadd float %322, %325, !dbg !32
  %328 = extractelement <4 x float> %314, i64 2, !dbg !31
  %329 = call float @llvm.amdgcn.exp2.f32(float %328), !dbg !31
  %330 = insertelement <4 x float> %326, float %329, i64 2, !dbg !33
  %331 = fadd float %327, %329, !dbg !32
  %332 = extractelement <4 x float> %314, i64 3, !dbg !31
  %333 = call float @llvm.amdgcn.exp2.f32(float %332), !dbg !31
  %334 = insertelement <4 x float> %330, float %333, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %334) #9, !dbg !33, !frisk.register_lifetimes !1
  %335 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %335) #9, !dbg !32, !frisk.register_lifetimes !1
  %336 = fadd float %331, %333, !dbg !32
  %337 = extractelement <4 x float> %315, i64 0, !dbg !31
  %338 = call float @llvm.amdgcn.exp2.f32(float %337), !dbg !31
  %339 = insertelement <4 x float> poison, float %338, i64 0, !dbg !33
  %340 = fadd float %336, %338, !dbg !32
  %341 = extractelement <4 x float> %315, i64 1, !dbg !31
  %342 = call float @llvm.amdgcn.exp2.f32(float %341), !dbg !31
  %343 = insertelement <4 x float> %339, float %342, i64 1, !dbg !33
  %344 = fadd float %340, %342, !dbg !32
  %345 = extractelement <4 x float> %315, i64 2, !dbg !31
  %346 = call float @llvm.amdgcn.exp2.f32(float %345), !dbg !31
  %347 = insertelement <4 x float> %343, float %346, i64 2, !dbg !33
  %348 = fadd float %344, %346, !dbg !32
  %349 = extractelement <4 x float> %315, i64 3, !dbg !31
  %350 = call float @llvm.amdgcn.exp2.f32(float %349), !dbg !31
  %351 = insertelement <4 x float> %347, float %350, i64 3, !dbg !33
  %math.result29 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %351) #9, !dbg !33, !frisk.register_lifetimes !1
  %352 = fptrunc <4 x float> %math.result29 to <4 x half>, !dbg !33
  %math.packed30 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %352) #9, !dbg !32, !frisk.register_lifetimes !1
  %353 = fadd float %348, %350, !dbg !32
  %354 = bitcast float %353 to i32, !dbg !32
  %355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %186, i32 %354), !dbg !32
  %356 = bitcast i32 %355 to float, !dbg !32
  %357 = fadd float %353, %356, !dbg !32
  %358 = bitcast float %357 to i32, !dbg !32
  %359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %202, i32 %358), !dbg !32
  %360 = bitcast i32 %359 to float, !dbg !32
  %361 = fadd float %357, %360, !dbg !32
  %362 = fadd float %361, %320, !dbg !32
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread77 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local78 = shl nuw nsw i32 %address.thread77, 3, !dbg !16
  %address.local79 = and i32 %address.local78, 504, !dbg !16
  %address.local80 = zext nneg i32 %address.local79 to i64, !dbg !16
  %address.local81 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local80, !dbg !16
  %363 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local81, i32 0) #0, !dbg !16
  %364 = bitcast <4 x i32> %363 to <8 x half>, !dbg !16
  %address.local82 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local80, !dbg !16
  %address.local83 = getelementptr i8, ptr addrspace(3) %address.local82, i64 1024, !dbg !16
  %365 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local83, i32 0) #0, !dbg !16
  %366 = bitcast <4 x i32> %365 to <8 x half>, !dbg !16
  %367 = add nsw i64 %50, %6, !dbg !18
  %368 = shl i64 %367, 19, !dbg !18
  %369 = and i64 %368, 16252928, !dbg !18
  %370 = add i64 %369, %140, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %371 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %370, !dbg !18
  %372 = getelementptr i8, ptr addrspace(1) %371, i64 64, !dbg !18
  br label %373, !dbg !18

373:                                              ; preds = %376, %274
  %374 = phi i64 [ %402, %376 ], [ %20, %274 ], !dbg !18
  %375 = icmp samesign ult i64 %374, 128, !dbg !18
  br i1 %375, label %376, label %403, !dbg !18

376:                                              ; preds = %373
  %377 = shl nuw nsw i64 %374, 3, !dbg !18
  %378 = and i64 %377, 24, !dbg !18
  %379 = shl nuw nsw i64 %374, 3, !dbg !18
  %380 = and i64 %379, 512, !dbg !18
  %381 = shl nuw nsw i64 %374, 5, !dbg !18
  %382 = and i64 %381, 3968, !dbg !18
  %383 = getelementptr [2 x i8], ptr addrspace(1) %372, i64 %382, !dbg !18
  %384 = getelementptr [2 x i8], ptr addrspace(1) %383, i64 %378, !dbg !18
  %385 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %380, !dbg !18
  %386 = ptrtoint ptr addrspace(3) %385 to i64, !dbg !18
  %387 = trunc i64 %386 to i32, !dbg !18
  %388 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %387), !dbg !18
  %389 = zext i32 %388 to i64, !dbg !18
  %390 = inttoptr i64 %389 to ptr addrspace(3), !dbg !18
  %391 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %392 = ptrtoint ptr addrspace(1) %384 to i64, !dbg !18
  %393 = sub i64 %392, %391, !dbg !18
  %dma.buffer.offset286 = trunc i64 %393 to i32, !dbg !18
  %394 = trunc i64 %391 to i32, !dbg !18
  %395 = insertelement <4 x i32> poison, i32 %394, i64 0, !dbg !18
  %396 = lshr i64 %391, 32, !dbg !18
  %397 = trunc i64 %396 to i32, !dbg !18
  %398 = insertelement <4 x i32> %395, i32 %397, i64 1, !dbg !18
  %399 = insertelement <4 x i32> %398, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc287 = insertelement <4 x i32> %399, i32 131072, i64 3, !dbg !18
  %400 = ptrtoint ptr addrspace(3) %390 to i32, !dbg !18
  %dma.buffer.lds288 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %400), !dbg !18
  %401 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds288, i32 %dma.buffer.offset286, <4 x i32> %dma.buffer.rsrc287) #10, !dbg !18
  %402 = or disjoint i64 %20, 256, !dbg !18
  br label %373, !dbg !18

403:                                              ; preds = %373
  %404 = shufflevector <8 x half> %364, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %405 = insertelement <4 x float> poison, float %318, i64 0, !dbg !34
  %406 = shufflevector <4 x float> %405, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %407 = fmul <4 x float> %33, %406, !dbg !34
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %404, <4 x float> %407) #9, !dbg !16
  %409 = shufflevector <8 x half> %366, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %409, <4 x float> %408) #9, !dbg !16
  %411 = shufflevector <8 x half> %364, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %412 = fmul <4 x float> %34, %406, !dbg !34
  %413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %411, <4 x float> %412) #9, !dbg !16
  %414 = shufflevector <8 x half> %366, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %414, <4 x float> %413) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread84 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local85 = shl nuw nsw i32 %address.thread84, 3, !dbg !16
  %address.local86 = and i32 %address.local85, 504, !dbg !16
  %address.local87 = zext nneg i32 %address.local86 to i64, !dbg !16
  %address.local88 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local87, !dbg !16
  %416 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local88, i32 0) #0, !dbg !16
  %417 = bitcast <4 x i32> %416 to <8 x half>, !dbg !16
  %address.local89 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local87, !dbg !16
  %address.local90 = getelementptr i8, ptr addrspace(3) %address.local89, i64 1024, !dbg !16
  %418 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local90, i32 0) #0, !dbg !16
  %419 = bitcast <4 x i32> %418 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %420 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %370, !dbg !18
  %421 = getelementptr i8, ptr addrspace(1) %420, i64 128, !dbg !18
  br label %422, !dbg !18

422:                                              ; preds = %425, %403
  %423 = phi i64 [ %451, %425 ], [ %20, %403 ], !dbg !18
  %424 = icmp samesign ult i64 %423, 128, !dbg !18
  br i1 %424, label %425, label %452, !dbg !18

425:                                              ; preds = %422
  %426 = shl nuw nsw i64 %423, 3, !dbg !18
  %427 = and i64 %426, 24, !dbg !18
  %428 = shl nuw nsw i64 %423, 3, !dbg !18
  %429 = and i64 %428, 512, !dbg !18
  %430 = shl nuw nsw i64 %423, 5, !dbg !18
  %431 = and i64 %430, 3968, !dbg !18
  %432 = getelementptr [2 x i8], ptr addrspace(1) %421, i64 %431, !dbg !18
  %433 = getelementptr [2 x i8], ptr addrspace(1) %432, i64 %427, !dbg !18
  %434 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %429, !dbg !18
  %435 = ptrtoint ptr addrspace(3) %434 to i64, !dbg !18
  %436 = trunc i64 %435 to i32, !dbg !18
  %437 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %436), !dbg !18
  %438 = zext i32 %437 to i64, !dbg !18
  %439 = inttoptr i64 %438 to ptr addrspace(3), !dbg !18
  %440 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %441 = ptrtoint ptr addrspace(1) %433 to i64, !dbg !18
  %442 = sub i64 %441, %440, !dbg !18
  %dma.buffer.offset289 = trunc i64 %442 to i32, !dbg !18
  %443 = trunc i64 %440 to i32, !dbg !18
  %444 = insertelement <4 x i32> poison, i32 %443, i64 0, !dbg !18
  %445 = lshr i64 %440, 32, !dbg !18
  %446 = trunc i64 %445 to i32, !dbg !18
  %447 = insertelement <4 x i32> %444, i32 %446, i64 1, !dbg !18
  %448 = insertelement <4 x i32> %447, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc290 = insertelement <4 x i32> %448, i32 131072, i64 3, !dbg !18
  %449 = ptrtoint ptr addrspace(3) %439 to i32, !dbg !18
  %dma.buffer.lds291 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %449), !dbg !18
  %450 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds291, i32 %dma.buffer.offset289, <4 x i32> %dma.buffer.rsrc290) #10, !dbg !18
  %451 = or disjoint i64 %20, 256, !dbg !18
  br label %422, !dbg !18

452:                                              ; preds = %422
  %453 = shufflevector <8 x half> %417, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %454 = fmul <4 x float> %38, %406, !dbg !34
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %453, <4 x float> %454) #9, !dbg !16
  %456 = shufflevector <8 x half> %419, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %456, <4 x float> %455) #9, !dbg !16
  %458 = shufflevector <8 x half> %417, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %459 = fmul <4 x float> %39, %406, !dbg !34
  %460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %458, <4 x float> %459) #9, !dbg !16
  %461 = shufflevector <8 x half> %419, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %461, <4 x float> %460) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread91 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local92 = shl nuw nsw i32 %address.thread91, 3, !dbg !16
  %address.local93 = and i32 %address.local92, 504, !dbg !16
  %address.local94 = zext nneg i32 %address.local93 to i64, !dbg !16
  %address.local95 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local94, !dbg !16
  %463 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local95, i32 0) #0, !dbg !16
  %464 = bitcast <4 x i32> %463 to <8 x half>, !dbg !16
  %address.local96 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local94, !dbg !16
  %address.local97 = getelementptr i8, ptr addrspace(3) %address.local96, i64 1024, !dbg !16
  %465 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local97, i32 0) #0, !dbg !16
  %466 = bitcast <4 x i32> %465 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %467 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %370, !dbg !18
  %468 = getelementptr i8, ptr addrspace(1) %467, i64 192, !dbg !18
  br label %469, !dbg !18

469:                                              ; preds = %472, %452
  %470 = phi i64 [ %498, %472 ], [ %20, %452 ], !dbg !18
  %471 = icmp samesign ult i64 %470, 128, !dbg !18
  br i1 %471, label %472, label %499, !dbg !18

472:                                              ; preds = %469
  %473 = shl nuw nsw i64 %470, 3, !dbg !18
  %474 = and i64 %473, 24, !dbg !18
  %475 = shl nuw nsw i64 %470, 3, !dbg !18
  %476 = and i64 %475, 512, !dbg !18
  %477 = shl nuw nsw i64 %470, 5, !dbg !18
  %478 = and i64 %477, 3968, !dbg !18
  %479 = getelementptr [2 x i8], ptr addrspace(1) %468, i64 %478, !dbg !18
  %480 = getelementptr [2 x i8], ptr addrspace(1) %479, i64 %474, !dbg !18
  %481 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %476, !dbg !18
  %482 = ptrtoint ptr addrspace(3) %481 to i64, !dbg !18
  %483 = trunc i64 %482 to i32, !dbg !18
  %484 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %483), !dbg !18
  %485 = zext i32 %484 to i64, !dbg !18
  %486 = inttoptr i64 %485 to ptr addrspace(3), !dbg !18
  %487 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %488 = ptrtoint ptr addrspace(1) %480 to i64, !dbg !18
  %489 = sub i64 %488, %487, !dbg !18
  %dma.buffer.offset292 = trunc i64 %489 to i32, !dbg !18
  %490 = trunc i64 %487 to i32, !dbg !18
  %491 = insertelement <4 x i32> poison, i32 %490, i64 0, !dbg !18
  %492 = lshr i64 %487, 32, !dbg !18
  %493 = trunc i64 %492 to i32, !dbg !18
  %494 = insertelement <4 x i32> %491, i32 %493, i64 1, !dbg !18
  %495 = insertelement <4 x i32> %494, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc293 = insertelement <4 x i32> %495, i32 131072, i64 3, !dbg !18
  %496 = ptrtoint ptr addrspace(3) %486 to i32, !dbg !18
  %dma.buffer.lds294 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %496), !dbg !18
  %497 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds294, i32 %dma.buffer.offset292, <4 x i32> %dma.buffer.rsrc293) #10, !dbg !18
  %498 = or disjoint i64 %20, 256, !dbg !18
  br label %469, !dbg !18

499:                                              ; preds = %469
  %500 = shufflevector <8 x half> %464, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %501 = fmul <4 x float> %41, %406, !dbg !34
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %500, <4 x float> %501) #9, !dbg !16
  %503 = shufflevector <8 x half> %466, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %503, <4 x float> %502) #9, !dbg !16
  %505 = shufflevector <8 x half> %464, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %506 = fmul <4 x float> %42, %406, !dbg !34
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %505, <4 x float> %506) #9, !dbg !16
  %508 = shufflevector <8 x half> %466, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %508, <4 x float> %507) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread98 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local99 = shl nuw nsw i32 %address.thread98, 3, !dbg !16
  %address.local100 = and i32 %address.local99, 504, !dbg !16
  %address.local101 = zext nneg i32 %address.local100 to i64, !dbg !16
  %address.local102 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local101, !dbg !16
  %510 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local102, i32 0) #0, !dbg !16
  %511 = bitcast <4 x i32> %510 to <8 x half>, !dbg !16
  %address.local103 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local101, !dbg !16
  %address.local104 = getelementptr i8, ptr addrspace(3) %address.local103, i64 1024, !dbg !16
  %512 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local104, i32 0) #0, !dbg !16
  %513 = bitcast <4 x i32> %512 to <8 x half>, !dbg !16
  %514 = shufflevector <8 x half> %511, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %515 = fmul <4 x float> %44, %406, !dbg !34
  %516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %514, <4 x float> %515) #9, !dbg !16
  %517 = shufflevector <8 x half> %513, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %517, <4 x float> %516) #9, !dbg !16
  %519 = shufflevector <8 x half> %511, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %520 = fmul <4 x float> %45, %406, !dbg !34
  %521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %519, <4 x float> %520) #9, !dbg !16
  %522 = shufflevector <8 x half> %513, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed30, <4 x half> %522, <4 x float> %521) #9, !dbg !16
  %524 = shufflevector <4 x float> %410, <4 x float> %415, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %525 = insertelement <1 x float> poison, float %362, i64 0, !dbg !7
  %526 = shufflevector <4 x float> %457, <4 x float> %462, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %527 = shufflevector <4 x float> %504, <4 x float> %509, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %528 = shufflevector <4 x float> %518, <4 x float> %523, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %529 = add nuw nsw i64 %32, 1, !dbg !7
  br label %31, !dbg !7

530:                                              ; preds = %31
  %531 = extractelement <8 x float> %46, i64 0, !dbg !7
  %532 = extractelement <8 x float> %46, i64 1, !dbg !7
  %533 = extractelement <8 x float> %46, i64 2, !dbg !7
  %534 = extractelement <8 x float> %46, i64 3, !dbg !7
  %535 = select i1 %.not, float %532, float %531, !dbg !7
  %536 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %537 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %536), !dbg !7
  %538 = add nuw nsw i32 %537, 64, !dbg !7
  %539 = xor i32 %537, 16, !dbg !7
  %540 = and i32 %538, 192, !dbg !7
  %541 = icmp samesign ult i32 %539, %540, !dbg !7
  %542 = select i1 %541, i32 %539, i32 %537, !dbg !7
  %543 = shl nuw nsw i32 %542, 2, !dbg !7
  %544 = bitcast float %535 to i32, !dbg !7
  %545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %544), !dbg !7
  %546 = bitcast i32 %545 to float, !dbg !7
  %547 = select i1 %.not, float %531, float %546, !dbg !7
  %548 = select i1 %.not, float %546, float %532, !dbg !7
  %549 = select i1 %.not, float %534, float %533, !dbg !7
  %550 = bitcast float %549 to i32, !dbg !7
  %551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %550), !dbg !7
  %552 = bitcast i32 %551 to float, !dbg !7
  %553 = select i1 %.not, float %533, float %552, !dbg !7
  %554 = select i1 %.not, float %552, float %534, !dbg !7
  %555 = select i1 %.not2, float %553, float %547, !dbg !7
  %556 = xor i32 %537, 32, !dbg !7
  %557 = icmp samesign ult i32 %556, %540, !dbg !7
  %558 = select i1 %557, i32 %556, i32 %537, !dbg !7
  %559 = shl nuw nsw i32 %558, 2, !dbg !7
  %560 = bitcast float %555 to i32, !dbg !7
  %561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %560), !dbg !7
  %562 = bitcast i32 %561 to float, !dbg !7
  %563 = select i1 %.not2, float %547, float %562, !dbg !7
  %564 = select i1 %.not2, float %562, float %553, !dbg !7
  %565 = select i1 %.not2, float %554, float %548, !dbg !7
  %566 = bitcast float %565 to i32, !dbg !7
  %567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %566), !dbg !7
  %568 = bitcast i32 %567 to float, !dbg !7
  %569 = select i1 %.not2, float %548, float %568, !dbg !7
  %570 = select i1 %.not2, float %568, float %554, !dbg !7
  %571 = extractelement <8 x float> %43, i64 0, !dbg !7
  %572 = extractelement <8 x float> %43, i64 1, !dbg !7
  %573 = extractelement <8 x float> %43, i64 2, !dbg !7
  %574 = extractelement <8 x float> %43, i64 3, !dbg !7
  %575 = select i1 %.not, float %572, float %571, !dbg !7
  %576 = bitcast float %575 to i32, !dbg !7
  %577 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %576), !dbg !7
  %578 = bitcast i32 %577 to float, !dbg !7
  %579 = select i1 %.not, float %571, float %578, !dbg !7
  %580 = select i1 %.not, float %578, float %572, !dbg !7
  %581 = select i1 %.not, float %574, float %573, !dbg !7
  %582 = bitcast float %581 to i32, !dbg !7
  %583 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %582), !dbg !7
  %584 = bitcast i32 %583 to float, !dbg !7
  %585 = select i1 %.not, float %573, float %584, !dbg !7
  %586 = select i1 %.not, float %584, float %574, !dbg !7
  %587 = select i1 %.not2, float %585, float %579, !dbg !7
  %588 = bitcast float %587 to i32, !dbg !7
  %589 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %588), !dbg !7
  %590 = bitcast i32 %589 to float, !dbg !7
  %591 = select i1 %.not2, float %579, float %590, !dbg !7
  %592 = select i1 %.not2, float %590, float %585, !dbg !7
  %593 = select i1 %.not2, float %586, float %580, !dbg !7
  %594 = bitcast float %593 to i32, !dbg !7
  %595 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %594), !dbg !7
  %596 = bitcast i32 %595 to float, !dbg !7
  %597 = select i1 %.not2, float %580, float %596, !dbg !7
  %598 = select i1 %.not2, float %596, float %586, !dbg !7
  %599 = extractelement <8 x float> %40, i64 0, !dbg !7
  %600 = extractelement <8 x float> %40, i64 1, !dbg !7
  %601 = extractelement <8 x float> %40, i64 2, !dbg !7
  %602 = extractelement <8 x float> %40, i64 3, !dbg !7
  %603 = select i1 %.not, float %600, float %599, !dbg !7
  %604 = bitcast float %603 to i32, !dbg !7
  %605 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %604), !dbg !7
  %606 = bitcast i32 %605 to float, !dbg !7
  %607 = select i1 %.not, float %599, float %606, !dbg !7
  %608 = select i1 %.not, float %606, float %600, !dbg !7
  %609 = select i1 %.not, float %602, float %601, !dbg !7
  %610 = bitcast float %609 to i32, !dbg !7
  %611 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %610), !dbg !7
  %612 = bitcast i32 %611 to float, !dbg !7
  %613 = select i1 %.not, float %601, float %612, !dbg !7
  %614 = select i1 %.not, float %612, float %602, !dbg !7
  %615 = select i1 %.not2, float %613, float %607, !dbg !7
  %616 = bitcast float %615 to i32, !dbg !7
  %617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %616), !dbg !7
  %618 = bitcast i32 %617 to float, !dbg !7
  %619 = select i1 %.not2, float %607, float %618, !dbg !7
  %620 = select i1 %.not2, float %618, float %613, !dbg !7
  %621 = select i1 %.not2, float %614, float %608, !dbg !7
  %622 = bitcast float %621 to i32, !dbg !7
  %623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %622), !dbg !7
  %624 = bitcast i32 %623 to float, !dbg !7
  %625 = select i1 %.not2, float %608, float %624, !dbg !7
  %626 = select i1 %.not2, float %624, float %614, !dbg !7
  %627 = extractelement <8 x float> %35, i64 0, !dbg !7
  %628 = extractelement <8 x float> %35, i64 1, !dbg !7
  %629 = extractelement <8 x float> %35, i64 2, !dbg !7
  %630 = extractelement <8 x float> %35, i64 3, !dbg !7
  %631 = select i1 %.not, float %628, float %627, !dbg !7
  %632 = bitcast float %631 to i32, !dbg !7
  %633 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %632), !dbg !7
  %634 = bitcast i32 %633 to float, !dbg !7
  %635 = select i1 %.not, float %627, float %634, !dbg !7
  %636 = select i1 %.not, float %634, float %628, !dbg !7
  %637 = select i1 %.not, float %630, float %629, !dbg !7
  %638 = bitcast float %637 to i32, !dbg !7
  %639 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %638), !dbg !7
  %640 = bitcast i32 %639 to float, !dbg !7
  %641 = select i1 %.not, float %629, float %640, !dbg !7
  %642 = select i1 %.not, float %640, float %630, !dbg !7
  %643 = select i1 %.not2, float %641, float %635, !dbg !7
  %644 = bitcast float %643 to i32, !dbg !7
  %645 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %644), !dbg !7
  %646 = select i1 %.not2, float %642, float %636, !dbg !7
  %647 = bitcast float %646 to i32, !dbg !7
  %648 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %647), !dbg !7
  %649 = shufflevector <1 x float> %36, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %650 = extractelement <8 x float> %46, i64 4, !dbg !7
  %651 = extractelement <8 x float> %46, i64 5, !dbg !7
  %652 = extractelement <8 x float> %46, i64 6, !dbg !7
  %653 = extractelement <8 x float> %46, i64 7, !dbg !7
  %654 = select i1 %.not, float %651, float %650, !dbg !7
  %655 = bitcast float %654 to i32, !dbg !7
  %656 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %655), !dbg !7
  %657 = bitcast i32 %656 to float, !dbg !7
  %658 = select i1 %.not, float %650, float %657, !dbg !7
  %659 = select i1 %.not, float %657, float %651, !dbg !7
  %660 = select i1 %.not, float %653, float %652, !dbg !7
  %661 = bitcast float %660 to i32, !dbg !7
  %662 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %661), !dbg !7
  %663 = bitcast i32 %662 to float, !dbg !7
  %664 = select i1 %.not, float %652, float %663, !dbg !7
  %665 = select i1 %.not, float %663, float %653, !dbg !7
  %666 = select i1 %.not2, float %664, float %658, !dbg !7
  %667 = bitcast float %666 to i32, !dbg !7
  %668 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %667), !dbg !7
  %669 = bitcast i32 %668 to float, !dbg !7
  %670 = select i1 %.not2, float %658, float %669, !dbg !7
  %671 = select i1 %.not2, float %669, float %664, !dbg !7
  %672 = select i1 %.not2, float %665, float %659, !dbg !7
  %673 = bitcast float %672 to i32, !dbg !7
  %674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %673), !dbg !7
  %675 = bitcast i32 %674 to float, !dbg !7
  %676 = select i1 %.not2, float %659, float %675, !dbg !7
  %677 = select i1 %.not2, float %675, float %665, !dbg !7
  %678 = extractelement <8 x float> %43, i64 4, !dbg !7
  %679 = extractelement <8 x float> %43, i64 5, !dbg !7
  %680 = extractelement <8 x float> %43, i64 6, !dbg !7
  %681 = extractelement <8 x float> %43, i64 7, !dbg !7
  %682 = select i1 %.not, float %679, float %678, !dbg !7
  %683 = bitcast float %682 to i32, !dbg !7
  %684 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %683), !dbg !7
  %685 = bitcast i32 %684 to float, !dbg !7
  %686 = select i1 %.not, float %678, float %685, !dbg !7
  %687 = select i1 %.not, float %685, float %679, !dbg !7
  %688 = select i1 %.not, float %681, float %680, !dbg !7
  %689 = bitcast float %688 to i32, !dbg !7
  %690 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %689), !dbg !7
  %691 = bitcast i32 %690 to float, !dbg !7
  %692 = select i1 %.not, float %680, float %691, !dbg !7
  %693 = select i1 %.not, float %691, float %681, !dbg !7
  %694 = select i1 %.not2, float %692, float %686, !dbg !7
  %695 = bitcast float %694 to i32, !dbg !7
  %696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %695), !dbg !7
  %697 = bitcast i32 %696 to float, !dbg !7
  %698 = select i1 %.not2, float %686, float %697, !dbg !7
  %699 = select i1 %.not2, float %697, float %692, !dbg !7
  %700 = select i1 %.not2, float %693, float %687, !dbg !7
  %701 = bitcast float %700 to i32, !dbg !7
  %702 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %701), !dbg !7
  %703 = bitcast i32 %702 to float, !dbg !7
  %704 = select i1 %.not2, float %687, float %703, !dbg !7
  %705 = select i1 %.not2, float %703, float %693, !dbg !7
  %706 = extractelement <8 x float> %40, i64 4, !dbg !7
  %707 = extractelement <8 x float> %40, i64 5, !dbg !7
  %708 = extractelement <8 x float> %40, i64 6, !dbg !7
  %709 = extractelement <8 x float> %40, i64 7, !dbg !7
  %710 = select i1 %.not, float %707, float %706, !dbg !7
  %711 = bitcast float %710 to i32, !dbg !7
  %712 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %711), !dbg !7
  %713 = bitcast i32 %712 to float, !dbg !7
  %714 = select i1 %.not, float %706, float %713, !dbg !7
  %715 = select i1 %.not, float %713, float %707, !dbg !7
  %716 = select i1 %.not, float %709, float %708, !dbg !7
  %717 = bitcast float %716 to i32, !dbg !7
  %718 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %717), !dbg !7
  %719 = bitcast i32 %718 to float, !dbg !7
  %720 = select i1 %.not, float %708, float %719, !dbg !7
  %721 = select i1 %.not, float %719, float %709, !dbg !7
  %722 = select i1 %.not2, float %720, float %714, !dbg !7
  %723 = bitcast float %722 to i32, !dbg !7
  %724 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %723), !dbg !7
  %725 = bitcast i32 %724 to float, !dbg !7
  %726 = select i1 %.not2, float %714, float %725, !dbg !7
  %727 = select i1 %.not2, float %725, float %720, !dbg !7
  %728 = select i1 %.not2, float %721, float %715, !dbg !7
  %729 = bitcast float %728 to i32, !dbg !7
  %730 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %729), !dbg !7
  %731 = bitcast i32 %730 to float, !dbg !7
  %732 = select i1 %.not2, float %715, float %731, !dbg !7
  %733 = select i1 %.not2, float %731, float %721, !dbg !7
  %734 = extractelement <8 x float> %35, i64 4, !dbg !7
  %735 = extractelement <8 x float> %35, i64 5, !dbg !7
  %736 = extractelement <8 x float> %35, i64 6, !dbg !7
  %737 = extractelement <8 x float> %35, i64 7, !dbg !7
  %738 = select i1 %.not, float %735, float %734, !dbg !7
  %739 = bitcast float %738 to i32, !dbg !7
  %740 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %739), !dbg !7
  %741 = bitcast i32 %740 to float, !dbg !7
  %742 = select i1 %.not, float %734, float %741, !dbg !7
  %743 = select i1 %.not, float %741, float %735, !dbg !7
  %744 = select i1 %.not, float %737, float %736, !dbg !7
  %745 = bitcast float %744 to i32, !dbg !7
  %746 = call i32 @llvm.amdgcn.ds.bpermute(i32 %543, i32 %745), !dbg !7
  %747 = bitcast i32 %746 to float, !dbg !7
  %748 = select i1 %.not, float %736, float %747, !dbg !7
  %749 = select i1 %.not, float %747, float %737, !dbg !7
  %750 = select i1 %.not2, float %748, float %742, !dbg !7
  %751 = bitcast float %750 to i32, !dbg !7
  %752 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %751), !dbg !7
  %753 = select i1 %.not2, float %749, float %743, !dbg !7
  %754 = bitcast float %753 to i32, !dbg !7
  %755 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %754), !dbg !7
  br i1 true, label %756, label %790, !dbg !21

756:                                              ; preds = %530
  %757 = bitcast i32 %752 to float, !dbg !7
  %758 = select i1 %.not2, float %742, float %757, !dbg !7
  %759 = insertelement <4 x float> poison, float %758, i64 0, !dbg !19
  %760 = bitcast i32 %755 to float, !dbg !7
  %761 = select i1 %.not2, float %743, float %760, !dbg !7
  %762 = insertelement <4 x float> %759, float %761, i64 1, !dbg !19
  %763 = select i1 %.not2, float %757, float %748, !dbg !7
  %764 = insertelement <4 x float> %762, float %763, i64 2, !dbg !19
  %765 = select i1 %.not2, float %760, float %749, !dbg !7
  %766 = insertelement <4 x float> %764, float %765, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %649, !dbg !19
  %normalized = fmul <4 x float> %766, %row.reciprocal, !dbg !19
  %767 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %768 = extractelement <4 x half> %767, i64 3, !dbg !20
  %769 = extractelement <4 x half> %767, i64 2, !dbg !20
  %770 = extractelement <4 x half> %767, i64 1, !dbg !20
  %771 = extractelement <4 x half> %767, i64 0, !dbg !20
  %772 = bitcast i32 %645 to float, !dbg !7
  %773 = select i1 %.not2, float %635, float %772, !dbg !7
  %774 = insertelement <4 x float> poison, float %773, i64 0, !dbg !19
  %775 = bitcast i32 %648 to float, !dbg !7
  %776 = select i1 %.not2, float %636, float %775, !dbg !7
  %777 = insertelement <4 x float> %774, float %776, i64 1, !dbg !19
  %778 = select i1 %.not2, float %772, float %641, !dbg !7
  %779 = insertelement <4 x float> %777, float %778, i64 2, !dbg !19
  %780 = select i1 %.not2, float %775, float %642, !dbg !7
  %781 = insertelement <4 x float> %779, float %780, i64 3, !dbg !19
  %normalized15 = fmul <4 x float> %781, %row.reciprocal, !dbg !19
  %782 = fptrunc <4 x float> %normalized15 to <4 x half>, !dbg !20
  %783 = extractelement <4 x half> %782, i64 3, !dbg !20
  %784 = extractelement <4 x half> %782, i64 2, !dbg !20
  %785 = extractelement <4 x half> %782, i64 1, !dbg !20
  %786 = extractelement <4 x half> %782, i64 0, !dbg !20
  %787 = add nsw i64 %6, %13, !dbg !21
  %788 = shl nsw i64 %787, 19, !dbg !21
  %789 = and i64 %788, 16252928, !dbg !21
  %address.thread105 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local106 = zext nneg i32 %address.thread105 to i64, !dbg !13
  %address.local107 = add nsw i64 %16, %address.local106, !dbg !21
  %address.local108 = lshr i64 %address.local106, 6, !dbg !13
  %address.local109 = shl nuw nsw i64 %address.local108, 4, !dbg !21
  %address.local110 = add nsw i64 %address.local107, %address.local109, !dbg !21
  %address.local111 = and i64 %address.local106, 240, !dbg !21
  %address.local112 = sub nsw i64 %address.local110, %address.local111, !dbg !21
  %address.local113 = add nsw i64 %address.local112, %17, !dbg !21
  %address.local114 = shl nuw nsw i64 %address.local113, 7, !dbg !21
  %address.local115 = add nuw nsw i64 %789, %address.local114, !dbg !21
  %address.local116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local115, !dbg !21
  %address.local117 = lshr i64 %address.local106, 2, !dbg !13
  %address.local118 = and i64 %address.local117, 12, !dbg !13
  %address.local119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local116, i64 %address.local118, !dbg !21
  store half %786, ptr addrspace(1) %address.local119, align 2, !dbg !21
  %address.local120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local115, !dbg !21
  %address.local121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local120, i64 %address.local118, !dbg !21
  %address.local122 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local121, i64 2, !dbg !21
  store half %785, ptr addrspace(1) %address.local122, align 2, !dbg !21
  %address.local123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local115, !dbg !21
  %address.local124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local123, i64 %address.local118, !dbg !21
  %address.local125 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local124, i64 4, !dbg !21
  store half %784, ptr addrspace(1) %address.local125, align 2, !dbg !21
  %address.local126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local115, !dbg !21
  %address.local127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local126, i64 %address.local118, !dbg !21
  %address.local128 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local127, i64 6, !dbg !21
  store half %783, ptr addrspace(1) %address.local128, align 2, !dbg !21
  %address.local129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local115, !dbg !21
  %address.local130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local129, i64 %address.local118, !dbg !21
  %address.local131 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local130, i64 32, !dbg !21
  store half %771, ptr addrspace(1) %address.local131, align 2, !dbg !21
  %address.local132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local115, !dbg !21
  %address.local133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local132, i64 %address.local118, !dbg !21
  %address.local134 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local133, i64 34, !dbg !21
  store half %770, ptr addrspace(1) %address.local134, align 2, !dbg !21
  %address.local135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local115, !dbg !21
  %address.local136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local135, i64 %address.local118, !dbg !21
  %address.local137 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local136, i64 36, !dbg !21
  store half %769, ptr addrspace(1) %address.local137, align 2, !dbg !21
  %address.local138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local115, !dbg !21
  %address.local139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local138, i64 %address.local118, !dbg !21
  %address.local140 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local139, i64 38, !dbg !21
  store half %768, ptr addrspace(1) %address.local140, align 2, !dbg !21
  br label %790, !dbg !21

790:                                              ; preds = %756, %530
  br i1 true, label %791, label %811, !dbg !21

791:                                              ; preds = %790
  %792 = insertelement <4 x float> poison, float %726, i64 0, !dbg !19
  %793 = insertelement <4 x float> %792, float %732, i64 1, !dbg !19
  %794 = insertelement <4 x float> %793, float %727, i64 2, !dbg !19
  %795 = insertelement <4 x float> %794, float %733, i64 3, !dbg !19
  %row.reciprocal16 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %649, !dbg !19
  %normalized17 = fmul <4 x float> %795, %row.reciprocal16, !dbg !19
  %796 = fptrunc <4 x float> %normalized17 to <4 x half>, !dbg !20
  %797 = extractelement <4 x half> %796, i64 3, !dbg !20
  %798 = extractelement <4 x half> %796, i64 2, !dbg !20
  %799 = extractelement <4 x half> %796, i64 1, !dbg !20
  %800 = extractelement <4 x half> %796, i64 0, !dbg !20
  %801 = insertelement <4 x float> poison, float %619, i64 0, !dbg !19
  %802 = insertelement <4 x float> %801, float %625, i64 1, !dbg !19
  %803 = insertelement <4 x float> %802, float %620, i64 2, !dbg !19
  %804 = insertelement <4 x float> %803, float %626, i64 3, !dbg !19
  %normalized18 = fmul <4 x float> %804, %row.reciprocal16, !dbg !19
  %805 = fptrunc <4 x float> %normalized18 to <4 x half>, !dbg !20
  %806 = extractelement <4 x half> %805, i64 3, !dbg !20
  %807 = extractelement <4 x half> %805, i64 2, !dbg !20
  %808 = extractelement <4 x half> %805, i64 1, !dbg !20
  %809 = extractelement <4 x half> %805, i64 0, !dbg !20
  %810 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread141 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local142 = zext nneg i32 %address.thread141 to i64, !dbg !13
  %address.local143 = add nsw i64 %16, %address.local142, !dbg !21
  %address.local144 = lshr i64 %address.local142, 6, !dbg !13
  %address.local145 = shl nuw nsw i64 %address.local144, 4, !dbg !21
  %address.local146 = add nsw i64 %address.local143, %address.local145, !dbg !21
  %address.local147 = and i64 %address.local142, 240, !dbg !21
  %address.local148 = sub nsw i64 %address.local146, %address.local147, !dbg !21
  %address.local149 = add nsw i64 %address.local148, %17, !dbg !21
  %address.local150 = shl nuw nsw i64 %address.local149, 7, !dbg !21
  %address.local151 = add nuw nsw i64 %810, %address.local150, !dbg !21
  %address.local152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local151, !dbg !21
  %address.local153 = lshr i64 %address.local142, 2, !dbg !13
  %address.local154 = and i64 %address.local153, 12, !dbg !13
  %address.local155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local152, i64 %address.local154, !dbg !21
  %address.local156 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local155, i64 64, !dbg !21
  store half %809, ptr addrspace(1) %address.local156, align 2, !dbg !21
  %address.local157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local151, !dbg !21
  %address.local158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local157, i64 %address.local154, !dbg !21
  %address.local159 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local158, i64 66, !dbg !21
  store half %808, ptr addrspace(1) %address.local159, align 2, !dbg !21
  %address.local160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local151, !dbg !21
  %address.local161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local160, i64 %address.local154, !dbg !21
  %address.local162 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local161, i64 68, !dbg !21
  store half %807, ptr addrspace(1) %address.local162, align 2, !dbg !21
  %address.local163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local151, !dbg !21
  %address.local164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local163, i64 %address.local154, !dbg !21
  %address.local165 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local164, i64 70, !dbg !21
  store half %806, ptr addrspace(1) %address.local165, align 2, !dbg !21
  %address.local166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local151, !dbg !21
  %address.local167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local166, i64 %address.local154, !dbg !21
  %address.local168 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local167, i64 96, !dbg !21
  store half %800, ptr addrspace(1) %address.local168, align 2, !dbg !21
  %address.local169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local151, !dbg !21
  %address.local170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local169, i64 %address.local154, !dbg !21
  %address.local171 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local170, i64 98, !dbg !21
  store half %799, ptr addrspace(1) %address.local171, align 2, !dbg !21
  %address.local172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local151, !dbg !21
  %address.local173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local172, i64 %address.local154, !dbg !21
  %address.local174 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local173, i64 100, !dbg !21
  store half %798, ptr addrspace(1) %address.local174, align 2, !dbg !21
  %address.local175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local151, !dbg !21
  %address.local176 = or i64 %address.local153, 51, !dbg !21
  %address.local177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local175, i64 %address.local176, !dbg !21
  store half %797, ptr addrspace(1) %address.local177, align 2, !dbg !21
  br label %811, !dbg !21

811:                                              ; preds = %791, %790
  br i1 true, label %812, label %832, !dbg !21

812:                                              ; preds = %811
  %813 = insertelement <4 x float> poison, float %698, i64 0, !dbg !19
  %814 = insertelement <4 x float> %813, float %704, i64 1, !dbg !19
  %815 = insertelement <4 x float> %814, float %699, i64 2, !dbg !19
  %816 = insertelement <4 x float> %815, float %705, i64 3, !dbg !19
  %row.reciprocal19 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %649, !dbg !19
  %normalized20 = fmul <4 x float> %816, %row.reciprocal19, !dbg !19
  %817 = fptrunc <4 x float> %normalized20 to <4 x half>, !dbg !20
  %818 = extractelement <4 x half> %817, i64 3, !dbg !20
  %819 = extractelement <4 x half> %817, i64 2, !dbg !20
  %820 = extractelement <4 x half> %817, i64 1, !dbg !20
  %821 = extractelement <4 x half> %817, i64 0, !dbg !20
  %822 = insertelement <4 x float> poison, float %591, i64 0, !dbg !19
  %823 = insertelement <4 x float> %822, float %597, i64 1, !dbg !19
  %824 = insertelement <4 x float> %823, float %592, i64 2, !dbg !19
  %825 = insertelement <4 x float> %824, float %598, i64 3, !dbg !19
  %normalized21 = fmul <4 x float> %825, %row.reciprocal19, !dbg !19
  %826 = fptrunc <4 x float> %normalized21 to <4 x half>, !dbg !20
  %827 = extractelement <4 x half> %826, i64 3, !dbg !20
  %828 = extractelement <4 x half> %826, i64 2, !dbg !20
  %829 = extractelement <4 x half> %826, i64 1, !dbg !20
  %830 = extractelement <4 x half> %826, i64 0, !dbg !20
  %831 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread178 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local179 = zext nneg i32 %address.thread178 to i64, !dbg !13
  %address.local180 = add nsw i64 %16, %address.local179, !dbg !21
  %address.local181 = lshr i64 %address.local179, 6, !dbg !13
  %address.local182 = shl nuw nsw i64 %address.local181, 4, !dbg !21
  %address.local183 = add nsw i64 %address.local180, %address.local182, !dbg !21
  %address.local184 = and i64 %address.local179, 240, !dbg !21
  %address.local185 = sub nsw i64 %address.local183, %address.local184, !dbg !21
  %address.local186 = add nsw i64 %address.local185, %17, !dbg !21
  %address.local187 = shl nuw nsw i64 %address.local186, 7, !dbg !21
  %address.local188 = add nuw nsw i64 %831, %address.local187, !dbg !21
  %address.local189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !21
  %address.local190 = lshr i64 %address.local179, 2, !dbg !13
  %address.local191 = and i64 %address.local190, 12, !dbg !13
  %address.local192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local189, i64 %address.local191, !dbg !21
  %address.local193 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local192, i64 128, !dbg !21
  store half %830, ptr addrspace(1) %address.local193, align 2, !dbg !21
  %address.local194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !21
  %address.local195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local194, i64 %address.local191, !dbg !21
  %address.local196 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local195, i64 130, !dbg !21
  store half %829, ptr addrspace(1) %address.local196, align 2, !dbg !21
  %address.local197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !21
  %address.local198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local197, i64 %address.local191, !dbg !21
  %address.local199 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local198, i64 132, !dbg !21
  store half %828, ptr addrspace(1) %address.local199, align 2, !dbg !21
  %address.local200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !21
  %address.local201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local200, i64 %address.local191, !dbg !21
  %address.local202 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local201, i64 134, !dbg !21
  store half %827, ptr addrspace(1) %address.local202, align 2, !dbg !21
  %address.local203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !21
  %address.local204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local203, i64 %address.local191, !dbg !21
  %address.local205 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local204, i64 160, !dbg !21
  store half %821, ptr addrspace(1) %address.local205, align 2, !dbg !21
  %address.local206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !21
  %address.local207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local206, i64 %address.local191, !dbg !21
  %address.local208 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local207, i64 162, !dbg !21
  store half %820, ptr addrspace(1) %address.local208, align 2, !dbg !21
  %address.local209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !21
  %address.local210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local209, i64 %address.local191, !dbg !21
  %address.local211 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local210, i64 164, !dbg !21
  store half %819, ptr addrspace(1) %address.local211, align 2, !dbg !21
  %address.local212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local188, !dbg !21
  %address.local213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local212, i64 %address.local191, !dbg !21
  %address.local214 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local213, i64 166, !dbg !21
  store half %818, ptr addrspace(1) %address.local214, align 2, !dbg !21
  br label %832, !dbg !21

832:                                              ; preds = %812, %811
  br i1 true, label %833, label %853, !dbg !21

833:                                              ; preds = %832
  %834 = insertelement <4 x float> poison, float %670, i64 0, !dbg !19
  %835 = insertelement <4 x float> %834, float %676, i64 1, !dbg !19
  %836 = insertelement <4 x float> %835, float %671, i64 2, !dbg !19
  %837 = insertelement <4 x float> %836, float %677, i64 3, !dbg !19
  %row.reciprocal22 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %649, !dbg !19
  %normalized23 = fmul <4 x float> %837, %row.reciprocal22, !dbg !19
  %838 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !20
  %839 = extractelement <4 x half> %838, i64 3, !dbg !20
  %840 = extractelement <4 x half> %838, i64 2, !dbg !20
  %841 = extractelement <4 x half> %838, i64 1, !dbg !20
  %842 = extractelement <4 x half> %838, i64 0, !dbg !20
  %843 = insertelement <4 x float> poison, float %563, i64 0, !dbg !19
  %844 = insertelement <4 x float> %843, float %569, i64 1, !dbg !19
  %845 = insertelement <4 x float> %844, float %564, i64 2, !dbg !19
  %846 = insertelement <4 x float> %845, float %570, i64 3, !dbg !19
  %normalized24 = fmul <4 x float> %846, %row.reciprocal22, !dbg !19
  %847 = fptrunc <4 x float> %normalized24 to <4 x half>, !dbg !20
  %848 = extractelement <4 x half> %847, i64 3, !dbg !20
  %849 = extractelement <4 x half> %847, i64 2, !dbg !20
  %850 = extractelement <4 x half> %847, i64 1, !dbg !20
  %851 = extractelement <4 x half> %847, i64 0, !dbg !20
  %852 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread215 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local216 = zext nneg i32 %address.thread215 to i64, !dbg !13
  %address.local217 = add nsw i64 %16, %address.local216, !dbg !21
  %address.local218 = lshr i64 %address.local216, 6, !dbg !13
  %address.local219 = shl nuw nsw i64 %address.local218, 4, !dbg !21
  %address.local220 = add nsw i64 %address.local217, %address.local219, !dbg !21
  %address.local221 = and i64 %address.local216, 240, !dbg !21
  %address.local222 = sub nsw i64 %address.local220, %address.local221, !dbg !21
  %address.local223 = add nsw i64 %address.local222, %17, !dbg !21
  %address.local224 = shl nuw nsw i64 %address.local223, 7, !dbg !21
  %address.local225 = add nuw nsw i64 %852, %address.local224, !dbg !21
  %address.local226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local225, !dbg !21
  %address.local227 = lshr i64 %address.local216, 2, !dbg !13
  %address.local228 = and i64 %address.local227, 12, !dbg !13
  %address.local229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local226, i64 %address.local228, !dbg !21
  %address.local230 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local229, i64 192, !dbg !21
  store half %851, ptr addrspace(1) %address.local230, align 2, !dbg !21
  %address.local231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local225, !dbg !21
  %address.local232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local231, i64 %address.local228, !dbg !21
  %address.local233 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local232, i64 194, !dbg !21
  store half %850, ptr addrspace(1) %address.local233, align 2, !dbg !21
  %address.local234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local225, !dbg !21
  %address.local235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local234, i64 %address.local228, !dbg !21
  %address.local236 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local235, i64 196, !dbg !21
  store half %849, ptr addrspace(1) %address.local236, align 2, !dbg !21
  %address.local237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local225, !dbg !21
  %address.local238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local237, i64 %address.local228, !dbg !21
  %address.local239 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local238, i64 198, !dbg !21
  store half %848, ptr addrspace(1) %address.local239, align 2, !dbg !21
  %address.local240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local225, !dbg !21
  %address.local241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local240, i64 %address.local228, !dbg !21
  %address.local242 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local241, i64 224, !dbg !21
  store half %842, ptr addrspace(1) %address.local242, align 2, !dbg !21
  %address.local243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local225, !dbg !21
  %address.local244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local243, i64 %address.local228, !dbg !21
  %address.local245 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local244, i64 226, !dbg !21
  store half %841, ptr addrspace(1) %address.local245, align 2, !dbg !21
  %address.local246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local225, !dbg !21
  %address.local247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local246, i64 %address.local228, !dbg !21
  %address.local248 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local247, i64 228, !dbg !21
  store half %840, ptr addrspace(1) %address.local248, align 2, !dbg !21
  %address.local249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local225, !dbg !21
  %address.local250 = or i64 %address.local227, 115, !dbg !21
  %address.local251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local249, i64 %address.local250, !dbg !21
  store half %839, ptr addrspace(1) %address.local251, align 2, !dbg !21
  br label %853, !dbg !21

853:                                              ; preds = %833, %832
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/nohint-rolled")
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
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
