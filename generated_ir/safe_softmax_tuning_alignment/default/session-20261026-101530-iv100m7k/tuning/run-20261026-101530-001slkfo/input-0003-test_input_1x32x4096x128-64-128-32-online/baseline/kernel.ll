; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

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
  %12 = icmp slt i32 %9, 0, !dbg !11
  %13 = xor i64 %10, -1, !dbg !11
  %14 = select i1 %12, i64 %13, i64 %10, !dbg !11
  %15 = sdiv i64 %14, 64, !dbg !11
  %.lobit = ashr i32 %9, 31, !dbg !11
  %16 = sext i32 %.lobit to i64, !dbg !11
  %17 = xor i64 %15, %16, !dbg !11
  %18 = add nsw i64 %17, %8, !dbg !11
  %19 = and i64 %18, 31, !dbg !11
  %20 = shl nsw i64 %10, 6, !dbg !11
  %21 = mul nsw i64 %17, -4096, !dbg !11
  %22 = add nsw i64 %20, %21, !dbg !11
  %23 = shl nuw nsw i64 %19, 19, !dbg !11
  %24 = shl nsw i64 %22, 7, !dbg !11
  %25 = add i64 %23, %24, !dbg !11
  %26 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !38
  %27 = zext nneg i32 %26 to i64, !dbg !12
  %28 = lshr i64 %27, 6, !dbg !12
  %29 = or disjoint i64 %11, 63, !dbg !13
  %30 = add nsw i64 %11, 64, !dbg !13
  %31 = icmp uge i64 %30, %29, !dbg !13
  %32 = icmp slt i32 %9, 0, !dbg !7
  %33 = select i1 %32, i64 %13, i64 %10, !dbg !7
  %34 = sdiv i64 %33, 2, !dbg !7
  %35 = sub nsw i64 0, %34, !dbg !7
  %36 = add nsw i64 %34, 1, !dbg !7
  %37 = select i1 %32, i64 %35, i64 %36, !dbg !7
  %38 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !11
  %39 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !11
  %40 = getelementptr i8, ptr addrspace(1) %39, i64 64, !dbg !11
  %41 = shl nsw i64 %8, 7, !dbg !15
  %42 = shl nsw i64 %6, 12, !dbg !15
  %43 = add nsw i64 %41, %42, !dbg !15
  %44 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !11
  %45 = getelementptr i8, ptr addrspace(1) %44, i64 128, !dbg !11
  %46 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !11
  %47 = getelementptr i8, ptr addrspace(1) %46, i64 192, !dbg !11
  br label %48, !dbg !7

48:                                               ; preds = %575, %4
  %49 = phi i64 [ %1137, %575 ], [ 0, %4 ], !dbg !7
  %50 = phi <4 x float> [ %912, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %51 = phi <4 x float> [ %928, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %52 = phi <8 x float> [ %1133, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %53 = phi <1 x float> [ %846, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %54 = phi <1 x float> [ %677, %575 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %55 = phi <4 x float> [ %990, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %56 = phi <4 x float> [ %1006, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %57 = phi <8 x float> [ %1134, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %58 = phi <4 x float> [ %1068, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %59 = phi <4 x float> [ %1084, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi <8 x float> [ %1135, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = phi <4 x float> [ %1116, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %62 = phi <4 x float> [ %1132, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi <8 x float> [ %1136, %575 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = icmp slt i64 %49, %37, !dbg !7
  br i1 %64, label %65, label %1138, !dbg !7

65:                                               ; preds = %48
  %66 = shl nsw i64 %49, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !11, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local81 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local82 = and i64 %address.local81, 1536, !dbg !11
  %address.local83 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local82, !dbg !11
  %address.local84 = ptrtoint ptr addrspace(3) %address.local83 to i64, !dbg !11
  %address.local85 = trunc i64 %address.local84 to i32, !dbg !11
  %67 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local85), !dbg !11
  %68 = zext i32 %67 to i64, !dbg !11
  %69 = inttoptr i64 %68 to ptr addrspace(3), !dbg !11
  %address.local86 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local87 = and i64 %address.local86, 8064, !dbg !11
  %address.local88 = lshr i64 %address.local, 3, !dbg !11
  %address.local89 = and i64 %address.local88, 24, !dbg !11
  %address.local90 = or disjoint i64 %address.local87, %address.local89, !dbg !11
  %address.local91 = getelementptr [2 x i8], ptr addrspace(1) %38, i64 %address.local90, !dbg !11
  %70 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %71 = ptrtoint ptr addrspace(1) %address.local91 to i64, !dbg !11
  %72 = sub i64 %71, %70, !dbg !11
  %dma.buffer.offset697 = trunc i64 %72 to i32, !dbg !11
  %73 = trunc i64 %70 to i32, !dbg !11
  %74 = insertelement <4 x i32> poison, i32 %73, i64 0, !dbg !11
  %75 = lshr i64 %70, 32, !dbg !11
  %76 = trunc i64 %75 to i32, !dbg !11
  %77 = insertelement <4 x i32> %74, i32 %76, i64 1, !dbg !11
  %78 = insertelement <4 x i32> %77, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc698 = insertelement <4 x i32> %78, i32 131072, i64 3, !dbg !11
  %79 = ptrtoint ptr addrspace(3) %69 to i32, !dbg !11
  %dma.buffer.lds699 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %79), !dbg !11
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds699, i32 %dma.buffer.offset697, <4 x i32> %dma.buffer.rsrc698) #10, !dbg !11
  %81 = lshr i64 %49, 12, !dbg !15
  %82 = add nsw i64 %81, %8, !dbg !15
  %83 = lshr i64 %49, 5, !dbg !15
  %84 = and i64 %83, 127, !dbg !15
  %85 = shl i64 %82, 19, !dbg !15
  %86 = and i64 %85, 16252928, !dbg !15
  %87 = shl i64 %83, 19, !dbg !15
  %88 = shl i64 %49, 14, !dbg !15
  %89 = sub i64 %88, %87, !dbg !15
  %90 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %86, !dbg !15
  %91 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %84, !dbg !15
  %92 = getelementptr [2 x i8], ptr addrspace(1) %91, i64 %89, !dbg !15
  %address.local92 = and i64 %address.local81, 1536, !dbg !15
  %address.local93 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local92, !dbg !15
  %address.local94 = ptrtoint ptr addrspace(3) %address.local93 to i64, !dbg !15
  %address.local95 = trunc i64 %address.local94 to i32, !dbg !15
  %93 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local95), !dbg !15
  %94 = zext i32 %93 to i64, !dbg !15
  %95 = inttoptr i64 %94 to ptr addrspace(3), !dbg !15
  %address.local96 = lshr i64 %address.local, 4, !dbg !15
  %address.local97 = and i64 %address.local96, 8, !dbg !15
  %address.local98 = shl nuw nsw i64 %address.local, 7, !dbg !15
  %address.local99 = and i64 %address.local98, 16256, !dbg !15
  %address.local100 = or disjoint i64 %address.local97, %address.local99, !dbg !15
  %address.local101 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local100, !dbg !15
  %96 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %97 = ptrtoint ptr addrspace(1) %address.local101 to i64, !dbg !15
  %98 = sub i64 %97, %96, !dbg !15
  %dma.buffer.offset694 = trunc i64 %98 to i32, !dbg !15
  %99 = trunc i64 %96 to i32, !dbg !15
  %100 = insertelement <4 x i32> poison, i32 %99, i64 0, !dbg !15
  %101 = lshr i64 %96, 32, !dbg !15
  %102 = trunc i64 %101 to i32, !dbg !15
  %103 = insertelement <4 x i32> %100, i32 %102, i64 1, !dbg !15
  %104 = insertelement <4 x i32> %103, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc695 = insertelement <4 x i32> %104, i32 131072, i64 3, !dbg !15
  %105 = ptrtoint ptr addrspace(3) %95 to i32, !dbg !15
  %dma.buffer.lds696 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %105), !dbg !15
  %106 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds696, i32 %dma.buffer.offset694, <4 x i32> %dma.buffer.rsrc695) #10, !dbg !15
  %address.local102 = or disjoint i64 %address.local, 256, !dbg !15
  %address.local103 = shl nuw nsw i64 %address.local102, 3, !dbg !15
  %address.local104 = and i64 %address.local103, 3584, !dbg !15
  %address.local105 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local104, !dbg !15
  %address.local106 = ptrtoint ptr addrspace(3) %address.local105 to i64, !dbg !15
  %address.local107 = trunc i64 %address.local106 to i32, !dbg !15
  %107 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local107), !dbg !15
  %108 = zext i32 %107 to i64, !dbg !15
  %109 = inttoptr i64 %108 to ptr addrspace(3), !dbg !15
  %address.local108 = lshr i64 %address.local102, 4, !dbg !15
  %address.local109 = and i64 %address.local108, 24, !dbg !15
  %address.local110 = shl nuw nsw i64 %address.local, 7, !dbg !15
  %address.local111 = and i64 %address.local110, 16256, !dbg !15
  %address.local112 = or disjoint i64 %address.local109, %address.local111, !dbg !15
  %address.local113 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local112, !dbg !15
  %110 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %111 = ptrtoint ptr addrspace(1) %address.local113 to i64, !dbg !15
  %112 = sub i64 %111, %110, !dbg !15
  %dma.buffer.offset691 = trunc i64 %112 to i32, !dbg !15
  %113 = trunc i64 %110 to i32, !dbg !15
  %114 = insertelement <4 x i32> poison, i32 %113, i64 0, !dbg !15
  %115 = lshr i64 %110, 32, !dbg !15
  %116 = trunc i64 %115 to i32, !dbg !15
  %117 = insertelement <4 x i32> %114, i32 %116, i64 1, !dbg !15
  %118 = insertelement <4 x i32> %117, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc692 = insertelement <4 x i32> %118, i32 131072, i64 3, !dbg !15
  %119 = ptrtoint ptr addrspace(3) %109 to i32, !dbg !15
  %dma.buffer.lds693 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %119), !dbg !15
  %120 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds693, i32 %dma.buffer.offset691, <4 x i32> %dma.buffer.rsrc692) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local114 = lshr i32 %address.thread, 2, !dbg !12
  %address.local115 = and i32 %address.local114, 4, !dbg !12
  %address.local116 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local117 = and i32 %address.local116, 384, !dbg !12
  %address.local118 = or disjoint i32 %address.local115, %address.local117, !dbg !12
  %address.local119 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local120 = and i32 %address.local119, 120, !dbg !12
  %address.local121 = or disjoint i32 %address.local118, %address.local120, !dbg !12
  %address.local122 = shl nuw nsw i32 %address.thread, 4, !dbg !12
  %address.local123 = and i32 %address.local122, 512, !dbg !12
  %address.local124 = or disjoint i32 %address.local121, %address.local123, !dbg !12
  %address.local125 = zext nneg i32 %address.local124 to i64, !dbg !12
  %address.local126 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local125, !dbg !12
  %121 = load <4 x half>, ptr addrspace(3) %address.local126, align 2, !dbg !12
  %address.local127 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local125, !dbg !12
  %address.local128 = getelementptr i8, ptr addrspace(3) %address.local127, i64 2048, !dbg !12
  %122 = load <4 x half>, ptr addrspace(3) %address.local128, align 2, !dbg !12
  %123 = fmul <4 x half> %121, splat (half 0xH3015), !dbg !12
  %124 = fmul <4 x half> %122, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local129 = lshr i32 %address.thread, 2, !dbg !16
  %address.local130 = and i32 %address.local129, 4, !dbg !16
  %address.local131 = shl nuw nsw i32 %address.thread, 1, !dbg !16
  %address.local132 = and i32 %address.local131, 24, !dbg !16
  %address.local133 = or disjoint i32 %address.local130, %address.local132, !dbg !16
  %address.local134 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local135 = and i32 %address.local134, 1024, !dbg !16
  %address.local136 = or disjoint i32 %address.local133, %address.local135, !dbg !16
  %address.local137 = shl nuw nsw i32 %address.thread, 7, !dbg !16
  %address.local138 = and i32 %address.local137, 384, !dbg !16
  %address.local139 = or disjoint i32 %address.local136, %address.local138, !dbg !16
  %address.local140 = zext nneg i32 %address.local139 to i64, !dbg !16
  %address.local141 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local140, !dbg !16
  %125 = load <4 x half>, ptr addrspace(3) %address.local141, align 2, !dbg !16
  %address.local142 = lshr i32 %address.thread, 2, !dbg !16
  %address.local143 = and i32 %address.local142, 4, !dbg !16
  %address.local144 = shl nuw nsw i32 %address.thread, 1, !dbg !16
  %address.local145 = and i32 %address.local144, 24, !dbg !16
  %address.local146 = or disjoint i32 %address.local143, %address.local145, !dbg !16
  %address.local147 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local148 = and i32 %address.local147, 1024, !dbg !16
  %address.local149 = or disjoint i32 %address.local146, %address.local148, !dbg !16
  %address.local150 = shl nuw nsw i32 %address.thread, 7, !dbg !16
  %address.local151 = and i32 %address.local150, 384, !dbg !16
  %address.local152 = or disjoint i32 %address.local149, %address.local151, !dbg !16
  %address.local153 = zext nneg i32 %address.local152 to i64, !dbg !16
  %address.local154 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local153, !dbg !16
  %address.local155 = getelementptr i8, ptr addrspace(3) %address.local154, i64 4096, !dbg !16
  %126 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !16
  %127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %125, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local156 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local156, !dbg !16
  %address.local158 = lshr i64 %address.local, 2, !dbg !16
  %address.local159 = mul nsw i64 %address.local158, -512, !dbg !16
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) %address.local157, i64 %address.local159, !dbg !16
  %address.local161 = shl nuw nsw i64 %address.local, 1, !dbg !16
  %address.local162 = and i64 %address.local161, 24, !dbg !16
  %address.local163 = getelementptr [2 x i8], ptr addrspace(3) %address.local160, i64 %address.local162, !dbg !16
  %address.local164 = lshr i64 %address.local, 2, !dbg !12
  %address.local165 = and i64 %address.local164, 12, !dbg !12
  %address.local166 = getelementptr [2 x i8], ptr addrspace(3) %address.local163, i64 %address.local165, !dbg !16
  %address.local167 = and i64 %address.local, 32, !dbg !16
  %address.local168 = icmp eq i64 %address.local167, 0, !dbg !16
  %address.local169 = select i1 %address.local168, i64 0, i64 1016, !dbg !16
  %address.local170 = getelementptr [2 x i8], ptr addrspace(3) %address.local166, i64 %address.local169, !dbg !16
  %address.local171 = getelementptr i8, ptr addrspace(3) %address.local170, i64 64, !dbg !16
  %128 = load <4 x half>, ptr addrspace(3) %address.local171, align 2, !dbg !16
  %129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %126, <4 x float> %127) #9, !dbg !16
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local156, !dbg !16
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) %address.local172, i64 %address.local165, !dbg !16
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) %address.local173, i64 %address.local169, !dbg !16
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) %address.local174, i64 %address.local159, !dbg !16
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) %address.local175, i64 %address.local162, !dbg !16
  %address.local177 = getelementptr i8, ptr addrspace(3) %address.local176, i64 4160, !dbg !16
  %130 = load <4 x half>, ptr addrspace(3) %address.local177, align 2, !dbg !16
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local156, !dbg !16
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) %address.local178, i64 %address.local159, !dbg !16
  %address.local180 = getelementptr [2 x i8], ptr addrspace(3) %address.local179, i64 %address.local162, !dbg !16
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) %address.local180, i64 %address.local165, !dbg !16
  %address.local182 = getelementptr [2 x i8], ptr addrspace(3) %address.local181, i64 %address.local169, !dbg !16
  %address.local183 = getelementptr i8, ptr addrspace(3) %address.local182, i64 128, !dbg !16
  %131 = load <4 x half>, ptr addrspace(3) %address.local183, align 2, !dbg !16
  %132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %128, <4 x float> zeroinitializer) #9, !dbg !16
  %133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %130, <4 x float> %132) #9, !dbg !16
  %address.local184 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local156, !dbg !16
  %address.local185 = getelementptr [2 x i8], ptr addrspace(3) %address.local184, i64 %address.local165, !dbg !16
  %address.local186 = getelementptr [2 x i8], ptr addrspace(3) %address.local185, i64 %address.local169, !dbg !16
  %address.local187 = getelementptr [2 x i8], ptr addrspace(3) %address.local186, i64 %address.local159, !dbg !16
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) %address.local187, i64 %address.local162, !dbg !16
  %address.local189 = getelementptr i8, ptr addrspace(3) %address.local188, i64 4224, !dbg !16
  %134 = load <4 x half>, ptr addrspace(3) %address.local189, align 2, !dbg !16
  %135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %131, <4 x float> zeroinitializer) #9, !dbg !16
  %136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %134, <4 x float> %135) #9, !dbg !16
  %address.local190 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local156, !dbg !16
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) %address.local190, i64 %address.local159, !dbg !16
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) %address.local191, i64 %address.local162, !dbg !16
  %address.local193 = getelementptr [2 x i8], ptr addrspace(3) %address.local192, i64 %address.local165, !dbg !16
  %address.local194 = getelementptr [2 x i8], ptr addrspace(3) %address.local193, i64 %address.local169, !dbg !16
  %address.local195 = getelementptr i8, ptr addrspace(3) %address.local194, i64 192, !dbg !16
  %137 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !16
  %address.local196 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local156, !dbg !16
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) %address.local196, i64 %address.local165, !dbg !16
  %address.local198 = getelementptr [2 x i8], ptr addrspace(3) %address.local197, i64 %address.local169, !dbg !16
  %address.local199 = getelementptr [2 x i8], ptr addrspace(3) %address.local198, i64 %address.local159, !dbg !16
  %address.local200 = getelementptr [2 x i8], ptr addrspace(3) %address.local199, i64 %address.local162, !dbg !16
  %address.local201 = getelementptr i8, ptr addrspace(3) %address.local200, i64 4288, !dbg !16
  %138 = load <4 x half>, ptr addrspace(3) %address.local201, align 2, !dbg !16
  %139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %137, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local202 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local140, !dbg !16
  %address.local203 = getelementptr i8, ptr addrspace(3) %address.local202, i64 1024, !dbg !16
  %140 = load <4 x half>, ptr addrspace(3) %address.local203, align 2, !dbg !16
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %138, <4 x float> %139) #9, !dbg !16
  %address.local204 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local153, !dbg !16
  %address.local205 = getelementptr i8, ptr addrspace(3) %address.local204, i64 5120, !dbg !16
  %142 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %140, <4 x float> zeroinitializer) #9, !dbg !16
  %144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %142, <4 x float> %143) #9, !dbg !16
  %address.local206 = add nuw nsw i64 %address.local156, 512, !dbg !16
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local206, !dbg !16
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i64 %address.local159, !dbg !16
  %address.local209 = getelementptr [2 x i8], ptr addrspace(3) %address.local208, i64 %address.local162, !dbg !16
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local209, i64 %address.local165, !dbg !16
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) %address.local210, i64 %address.local169, !dbg !16
  %address.local212 = getelementptr i8, ptr addrspace(3) %address.local211, i64 64, !dbg !16
  %145 = load <4 x half>, ptr addrspace(3) %address.local212, align 2, !dbg !16
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local206, !dbg !16
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i64 %address.local165, !dbg !16
  %address.local215 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local169, !dbg !16
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) %address.local215, i64 %address.local159, !dbg !16
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local216, i64 %address.local162, !dbg !16
  %address.local218 = getelementptr i8, ptr addrspace(3) %address.local217, i64 4160, !dbg !16
  %146 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !16
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %145, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local219 = add nuw nsw i64 %address.local156, 512, !dbg !16
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !16
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local159, !dbg !16
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) %address.local221, i64 %address.local162, !dbg !16
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local165, !dbg !16
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local169, !dbg !16
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !16
  %148 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %146, <4 x float> %147) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !16
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local165, !dbg !16
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local169, !dbg !16
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local159, !dbg !16
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local162, !dbg !16
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4224, !dbg !16
  %150 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %148, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %150, <4 x float> %151) #9, !dbg !16
  %address.local232 = add nuw nsw i64 %address.local156, 512, !dbg !16
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local232, !dbg !16
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) %address.local233, i64 %address.local159, !dbg !16
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local162, !dbg !16
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local165, !dbg !16
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local236, i64 %address.local169, !dbg !16
  %address.local238 = getelementptr i8, ptr addrspace(3) %address.local237, i64 192, !dbg !16
  %153 = load <4 x half>, ptr addrspace(3) %address.local238, align 2, !dbg !16
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local232, !dbg !16
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) %address.local239, i64 %address.local165, !dbg !16
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local169, !dbg !16
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local159, !dbg !16
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) %address.local242, i64 %address.local162, !dbg !16
  %address.local244 = getelementptr i8, ptr addrspace(3) %address.local243, i64 4288, !dbg !16
  %154 = load <4 x half>, ptr addrspace(3) %address.local244, align 2, !dbg !16
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !16
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %154, <4 x float> %155) #9, !dbg !16
  %157 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local85), !dbg !11
  %158 = zext i32 %157 to i64, !dbg !11
  %159 = inttoptr i64 %158 to ptr addrspace(3), !dbg !11
  %address.local245 = getelementptr [2 x i8], ptr addrspace(1) %40, i64 %address.local90, !dbg !11
  %160 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %161 = ptrtoint ptr addrspace(1) %address.local245 to i64, !dbg !11
  %162 = sub i64 %161, %160, !dbg !11
  %dma.buffer.offset688 = trunc i64 %162 to i32, !dbg !11
  %163 = trunc i64 %160 to i32, !dbg !11
  %164 = insertelement <4 x i32> poison, i32 %163, i64 0, !dbg !11
  %165 = lshr i64 %160, 32, !dbg !11
  %166 = trunc i64 %165 to i32, !dbg !11
  %167 = insertelement <4 x i32> %164, i32 %166, i64 1, !dbg !11
  %168 = insertelement <4 x i32> %167, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc689 = insertelement <4 x i32> %168, i32 131072, i64 3, !dbg !11
  %169 = ptrtoint ptr addrspace(3) %159 to i32, !dbg !11
  %dma.buffer.lds690 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %169), !dbg !11
  %170 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds690, i32 %dma.buffer.offset688, <4 x i32> %dma.buffer.rsrc689) #10, !dbg !11
  %171 = add nuw i64 %49, 1024, !dbg !15
  %172 = icmp slt i64 %171, 0, !dbg !15
  %173 = sub nuw i64 -1025, %49, !dbg !15
  %174 = select i1 %172, i64 %173, i64 %171, !dbg !15
  %175 = sdiv i64 %174, 4096, !dbg !15
  %.lobit2 = ashr i64 %171, 63, !dbg !15
  %176 = xor i64 %175, %.lobit2, !dbg !15
  %177 = add nsw i64 %176, %8, !dbg !15
  %178 = add nsw i64 %83, %43, !dbg !15
  %179 = add nsw i64 %178, 32, !dbg !15
  %180 = icmp slt i64 %178, -32, !dbg !15
  %181 = sub nsw i64 -33, %178, !dbg !15
  %182 = select i1 %180, i64 %181, i64 %179, !dbg !15
  %183 = sdiv i64 %182, 128, !dbg !15
  %184 = sext i1 %180 to i64, !dbg !15
  %185 = xor i64 %183, %184, !dbg !15
  %186 = shl nsw i64 %185, 7, !dbg !15
  %187 = sub i64 %43, %186, !dbg !15
  %188 = shl i64 %177, 19, !dbg !15
  %189 = and i64 %188, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %190 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %189, !dbg !15
  %191 = getelementptr [2 x i8], ptr addrspace(1) %190, i64 %83, !dbg !15
  %192 = getelementptr [2 x i8], ptr addrspace(1) %191, i64 %187, !dbg !15
  %193 = getelementptr i8, ptr addrspace(1) %192, i64 64, !dbg !15
  %194 = getelementptr [2 x i8], ptr addrspace(1) %193, i64 %89, !dbg !15
  %195 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local95), !dbg !15
  %196 = zext i32 %195 to i64, !dbg !15
  %197 = inttoptr i64 %196 to ptr addrspace(3), !dbg !15
  %address.local246 = getelementptr [2 x i8], ptr addrspace(1) %194, i64 %address.local100, !dbg !15
  %198 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %199 = ptrtoint ptr addrspace(1) %address.local246 to i64, !dbg !15
  %200 = sub i64 %199, %198, !dbg !15
  %dma.buffer.offset685 = trunc i64 %200 to i32, !dbg !15
  %201 = trunc i64 %198 to i32, !dbg !15
  %202 = insertelement <4 x i32> poison, i32 %201, i64 0, !dbg !15
  %203 = lshr i64 %198, 32, !dbg !15
  %204 = trunc i64 %203 to i32, !dbg !15
  %205 = insertelement <4 x i32> %202, i32 %204, i64 1, !dbg !15
  %206 = insertelement <4 x i32> %205, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc686 = insertelement <4 x i32> %206, i32 131072, i64 3, !dbg !15
  %207 = ptrtoint ptr addrspace(3) %197 to i32, !dbg !15
  %dma.buffer.lds687 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %207), !dbg !15
  %208 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds687, i32 %dma.buffer.offset685, <4 x i32> %dma.buffer.rsrc686) #10, !dbg !15
  %209 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local107), !dbg !15
  %210 = zext i32 %209 to i64, !dbg !15
  %211 = inttoptr i64 %210 to ptr addrspace(3), !dbg !15
  %address.local247 = getelementptr [2 x i8], ptr addrspace(1) %194, i64 %address.local112, !dbg !15
  %212 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %213 = ptrtoint ptr addrspace(1) %address.local247 to i64, !dbg !15
  %214 = sub i64 %213, %212, !dbg !15
  %dma.buffer.offset682 = trunc i64 %214 to i32, !dbg !15
  %215 = trunc i64 %212 to i32, !dbg !15
  %216 = insertelement <4 x i32> poison, i32 %215, i64 0, !dbg !15
  %217 = lshr i64 %212, 32, !dbg !15
  %218 = trunc i64 %217 to i32, !dbg !15
  %219 = insertelement <4 x i32> %216, i32 %218, i64 1, !dbg !15
  %220 = insertelement <4 x i32> %219, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc683 = insertelement <4 x i32> %220, i32 131072, i64 3, !dbg !15
  %221 = ptrtoint ptr addrspace(3) %211 to i32, !dbg !15
  %dma.buffer.lds684 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %221), !dbg !15
  %222 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds684, i32 %dma.buffer.offset682, <4 x i32> %dma.buffer.rsrc683) #10, !dbg !15
  %223 = load <4 x half>, ptr addrspace(3) %address.local126, align 2, !dbg !12
  %224 = load <4 x half>, ptr addrspace(3) %address.local128, align 2, !dbg !12
  %225 = fmul <4 x half> %223, splat (half 0xH3015), !dbg !12
  %226 = fmul <4 x half> %224, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %227 = load <4 x half>, ptr addrspace(3) %address.local141, align 2, !dbg !16
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %227, <4 x float> %129) #9, !dbg !16
  %229 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !16
  %230 = load <4 x half>, ptr addrspace(3) %address.local171, align 2, !dbg !16
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %229, <4 x float> %228) #9, !dbg !16
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %230, <4 x float> %133) #9, !dbg !16
  %233 = load <4 x half>, ptr addrspace(3) %address.local177, align 2, !dbg !16
  %234 = load <4 x half>, ptr addrspace(3) %address.local183, align 2, !dbg !16
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %233, <4 x float> %232) #9, !dbg !16
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %234, <4 x float> %136) #9, !dbg !16
  %237 = load <4 x half>, ptr addrspace(3) %address.local189, align 2, !dbg !16
  %238 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !16
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %237, <4 x float> %236) #9, !dbg !16
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %238, <4 x float> %141) #9, !dbg !16
  %241 = load <4 x half>, ptr addrspace(3) %address.local201, align 2, !dbg !16
  %242 = load <4 x half>, ptr addrspace(3) %address.local203, align 2, !dbg !16
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %241, <4 x float> %240) #9, !dbg !16
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %242, <4 x float> %144) #9, !dbg !16
  %245 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %246 = load <4 x half>, ptr addrspace(3) %address.local212, align 2, !dbg !16
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %245, <4 x float> %244) #9, !dbg !16
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %246, <4 x float> %149) #9, !dbg !16
  %249 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !16
  %250 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !16
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %249, <4 x float> %248) #9, !dbg !16
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %250, <4 x float> %152) #9, !dbg !16
  %253 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !16
  %254 = load <4 x half>, ptr addrspace(3) %address.local238, align 2, !dbg !16
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %253, <4 x float> %252) #9, !dbg !16
  %256 = load <4 x half>, ptr addrspace(3) %address.local244, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %225, <4 x half> %254, <4 x float> %156) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %256, <4 x float> %257) #9, !dbg !16
  %259 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local85), !dbg !11
  %260 = zext i32 %259 to i64, !dbg !11
  %261 = inttoptr i64 %260 to ptr addrspace(3), !dbg !11
  %address.local248 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local90, !dbg !11
  %262 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %263 = ptrtoint ptr addrspace(1) %address.local248 to i64, !dbg !11
  %264 = sub i64 %263, %262, !dbg !11
  %dma.buffer.offset679 = trunc i64 %264 to i32, !dbg !11
  %265 = trunc i64 %262 to i32, !dbg !11
  %266 = insertelement <4 x i32> poison, i32 %265, i64 0, !dbg !11
  %267 = lshr i64 %262, 32, !dbg !11
  %268 = trunc i64 %267 to i32, !dbg !11
  %269 = insertelement <4 x i32> %266, i32 %268, i64 1, !dbg !11
  %270 = insertelement <4 x i32> %269, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc680 = insertelement <4 x i32> %270, i32 131072, i64 3, !dbg !11
  %271 = ptrtoint ptr addrspace(3) %261 to i32, !dbg !11
  %dma.buffer.lds681 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %271), !dbg !11
  %272 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds681, i32 %dma.buffer.offset679, <4 x i32> %dma.buffer.rsrc680) #10, !dbg !11
  %273 = add nuw i64 %49, 2048, !dbg !15
  %274 = icmp slt i64 %273, 0, !dbg !15
  %275 = sub nuw i64 -2049, %49, !dbg !15
  %276 = select i1 %274, i64 %275, i64 %273, !dbg !15
  %277 = sdiv i64 %276, 4096, !dbg !15
  %.lobit3 = ashr i64 %273, 63, !dbg !15
  %278 = xor i64 %277, %.lobit3, !dbg !15
  %279 = add nsw i64 %278, %8, !dbg !15
  %280 = add nsw i64 %178, 64, !dbg !15
  %281 = icmp slt i64 %178, -64, !dbg !15
  %282 = sub nsw i64 -65, %178, !dbg !15
  %283 = select i1 %281, i64 %282, i64 %280, !dbg !15
  %284 = sdiv i64 %283, 128, !dbg !15
  %285 = sext i1 %281 to i64, !dbg !15
  %286 = xor i64 %284, %285, !dbg !15
  %287 = shl nsw i64 %286, 7, !dbg !15
  %288 = sub i64 %43, %287, !dbg !15
  %289 = shl i64 %279, 19, !dbg !15
  %290 = and i64 %289, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %291 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %290, !dbg !15
  %292 = getelementptr [2 x i8], ptr addrspace(1) %291, i64 %83, !dbg !15
  %293 = getelementptr [2 x i8], ptr addrspace(1) %292, i64 %288, !dbg !15
  %294 = getelementptr i8, ptr addrspace(1) %293, i64 128, !dbg !15
  %295 = getelementptr [2 x i8], ptr addrspace(1) %294, i64 %89, !dbg !15
  %296 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local95), !dbg !15
  %297 = zext i32 %296 to i64, !dbg !15
  %298 = inttoptr i64 %297 to ptr addrspace(3), !dbg !15
  %address.local249 = getelementptr [2 x i8], ptr addrspace(1) %295, i64 %address.local100, !dbg !15
  %299 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %300 = ptrtoint ptr addrspace(1) %address.local249 to i64, !dbg !15
  %301 = sub i64 %300, %299, !dbg !15
  %dma.buffer.offset676 = trunc i64 %301 to i32, !dbg !15
  %302 = trunc i64 %299 to i32, !dbg !15
  %303 = insertelement <4 x i32> poison, i32 %302, i64 0, !dbg !15
  %304 = lshr i64 %299, 32, !dbg !15
  %305 = trunc i64 %304 to i32, !dbg !15
  %306 = insertelement <4 x i32> %303, i32 %305, i64 1, !dbg !15
  %307 = insertelement <4 x i32> %306, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc677 = insertelement <4 x i32> %307, i32 131072, i64 3, !dbg !15
  %308 = ptrtoint ptr addrspace(3) %298 to i32, !dbg !15
  %dma.buffer.lds678 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %308), !dbg !15
  %309 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds678, i32 %dma.buffer.offset676, <4 x i32> %dma.buffer.rsrc677) #10, !dbg !15
  %310 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local107), !dbg !15
  %311 = zext i32 %310 to i64, !dbg !15
  %312 = inttoptr i64 %311 to ptr addrspace(3), !dbg !15
  %address.local250 = getelementptr [2 x i8], ptr addrspace(1) %295, i64 %address.local112, !dbg !15
  %313 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %314 = ptrtoint ptr addrspace(1) %address.local250 to i64, !dbg !15
  %315 = sub i64 %314, %313, !dbg !15
  %dma.buffer.offset673 = trunc i64 %315 to i32, !dbg !15
  %316 = trunc i64 %313 to i32, !dbg !15
  %317 = insertelement <4 x i32> poison, i32 %316, i64 0, !dbg !15
  %318 = lshr i64 %313, 32, !dbg !15
  %319 = trunc i64 %318 to i32, !dbg !15
  %320 = insertelement <4 x i32> %317, i32 %319, i64 1, !dbg !15
  %321 = insertelement <4 x i32> %320, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc674 = insertelement <4 x i32> %321, i32 131072, i64 3, !dbg !15
  %322 = ptrtoint ptr addrspace(3) %312 to i32, !dbg !15
  %dma.buffer.lds675 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %322), !dbg !15
  %323 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds675, i32 %dma.buffer.offset673, <4 x i32> %dma.buffer.rsrc674) #10, !dbg !15
  %324 = load <4 x half>, ptr addrspace(3) %address.local126, align 2, !dbg !12
  %325 = load <4 x half>, ptr addrspace(3) %address.local128, align 2, !dbg !12
  %326 = fmul <4 x half> %324, splat (half 0xH3015), !dbg !12
  %327 = fmul <4 x half> %325, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %328 = load <4 x half>, ptr addrspace(3) %address.local141, align 2, !dbg !16
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %326, <4 x half> %328, <4 x float> %231) #9, !dbg !16
  %330 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !16
  %331 = load <4 x half>, ptr addrspace(3) %address.local171, align 2, !dbg !16
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %330, <4 x float> %329) #9, !dbg !16
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %326, <4 x half> %331, <4 x float> %235) #9, !dbg !16
  %334 = load <4 x half>, ptr addrspace(3) %address.local177, align 2, !dbg !16
  %335 = load <4 x half>, ptr addrspace(3) %address.local183, align 2, !dbg !16
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %334, <4 x float> %333) #9, !dbg !16
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %326, <4 x half> %335, <4 x float> %239) #9, !dbg !16
  %338 = load <4 x half>, ptr addrspace(3) %address.local189, align 2, !dbg !16
  %339 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !16
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %338, <4 x float> %337) #9, !dbg !16
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %326, <4 x half> %339, <4 x float> %243) #9, !dbg !16
  %342 = load <4 x half>, ptr addrspace(3) %address.local201, align 2, !dbg !16
  %343 = load <4 x half>, ptr addrspace(3) %address.local203, align 2, !dbg !16
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %342, <4 x float> %341) #9, !dbg !16
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %326, <4 x half> %343, <4 x float> %247) #9, !dbg !16
  %346 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %347 = load <4 x half>, ptr addrspace(3) %address.local212, align 2, !dbg !16
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %346, <4 x float> %345) #9, !dbg !16
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %326, <4 x half> %347, <4 x float> %251) #9, !dbg !16
  %350 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !16
  %351 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !16
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %350, <4 x float> %349) #9, !dbg !16
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %326, <4 x half> %351, <4 x float> %255) #9, !dbg !16
  %354 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !16
  %355 = load <4 x half>, ptr addrspace(3) %address.local238, align 2, !dbg !16
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %354, <4 x float> %353) #9, !dbg !16
  %357 = load <4 x half>, ptr addrspace(3) %address.local244, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %326, <4 x half> %355, <4 x float> %258) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %357, <4 x float> %358) #9, !dbg !16
  %360 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local85), !dbg !11
  %361 = zext i32 %360 to i64, !dbg !11
  %362 = inttoptr i64 %361 to ptr addrspace(3), !dbg !11
  %address.local251 = getelementptr [2 x i8], ptr addrspace(1) %47, i64 %address.local90, !dbg !11
  %363 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %364 = ptrtoint ptr addrspace(1) %address.local251 to i64, !dbg !11
  %365 = sub i64 %364, %363, !dbg !11
  %dma.buffer.offset670 = trunc i64 %365 to i32, !dbg !11
  %366 = trunc i64 %363 to i32, !dbg !11
  %367 = insertelement <4 x i32> poison, i32 %366, i64 0, !dbg !11
  %368 = lshr i64 %363, 32, !dbg !11
  %369 = trunc i64 %368 to i32, !dbg !11
  %370 = insertelement <4 x i32> %367, i32 %369, i64 1, !dbg !11
  %371 = insertelement <4 x i32> %370, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc671 = insertelement <4 x i32> %371, i32 131072, i64 3, !dbg !11
  %372 = ptrtoint ptr addrspace(3) %362 to i32, !dbg !11
  %dma.buffer.lds672 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %372), !dbg !11
  %373 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds672, i32 %dma.buffer.offset670, <4 x i32> %dma.buffer.rsrc671) #10, !dbg !11
  %374 = add nuw i64 %49, 3072, !dbg !15
  %375 = icmp slt i64 %374, 0, !dbg !15
  %376 = sub nuw i64 -3073, %49, !dbg !15
  %377 = select i1 %375, i64 %376, i64 %374, !dbg !15
  %378 = sdiv i64 %377, 4096, !dbg !15
  %.lobit4 = ashr i64 %374, 63, !dbg !15
  %379 = xor i64 %378, %.lobit4, !dbg !15
  %380 = add nsw i64 %379, %8, !dbg !15
  %381 = add nsw i64 %178, 96, !dbg !15
  %382 = icmp slt i64 %178, -96, !dbg !15
  %383 = sub nsw i64 -97, %178, !dbg !15
  %384 = select i1 %382, i64 %383, i64 %381, !dbg !15
  %385 = sdiv i64 %384, 128, !dbg !15
  %386 = sext i1 %382 to i64, !dbg !15
  %387 = xor i64 %385, %386, !dbg !15
  %388 = shl nsw i64 %387, 7, !dbg !15
  %389 = sub i64 %43, %388, !dbg !15
  %390 = shl i64 %380, 19, !dbg !15
  %391 = and i64 %390, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %392 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %391, !dbg !15
  %393 = getelementptr [2 x i8], ptr addrspace(1) %392, i64 %83, !dbg !15
  %394 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %389, !dbg !15
  %395 = getelementptr i8, ptr addrspace(1) %394, i64 192, !dbg !15
  %396 = getelementptr [2 x i8], ptr addrspace(1) %395, i64 %89, !dbg !15
  %397 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local95), !dbg !15
  %398 = zext i32 %397 to i64, !dbg !15
  %399 = inttoptr i64 %398 to ptr addrspace(3), !dbg !15
  %address.local252 = getelementptr [2 x i8], ptr addrspace(1) %396, i64 %address.local100, !dbg !15
  %400 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %401 = ptrtoint ptr addrspace(1) %address.local252 to i64, !dbg !15
  %402 = sub i64 %401, %400, !dbg !15
  %dma.buffer.offset667 = trunc i64 %402 to i32, !dbg !15
  %403 = trunc i64 %400 to i32, !dbg !15
  %404 = insertelement <4 x i32> poison, i32 %403, i64 0, !dbg !15
  %405 = lshr i64 %400, 32, !dbg !15
  %406 = trunc i64 %405 to i32, !dbg !15
  %407 = insertelement <4 x i32> %404, i32 %406, i64 1, !dbg !15
  %408 = insertelement <4 x i32> %407, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc668 = insertelement <4 x i32> %408, i32 131072, i64 3, !dbg !15
  %409 = ptrtoint ptr addrspace(3) %399 to i32, !dbg !15
  %dma.buffer.lds669 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %409), !dbg !15
  %410 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds669, i32 %dma.buffer.offset667, <4 x i32> %dma.buffer.rsrc668) #10, !dbg !15
  %411 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local107), !dbg !15
  %412 = zext i32 %411 to i64, !dbg !15
  %413 = inttoptr i64 %412 to ptr addrspace(3), !dbg !15
  %address.local253 = getelementptr [2 x i8], ptr addrspace(1) %396, i64 %address.local112, !dbg !15
  %414 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %415 = ptrtoint ptr addrspace(1) %address.local253 to i64, !dbg !15
  %416 = sub i64 %415, %414, !dbg !15
  %dma.buffer.offset664 = trunc i64 %416 to i32, !dbg !15
  %417 = trunc i64 %414 to i32, !dbg !15
  %418 = insertelement <4 x i32> poison, i32 %417, i64 0, !dbg !15
  %419 = lshr i64 %414, 32, !dbg !15
  %420 = trunc i64 %419 to i32, !dbg !15
  %421 = insertelement <4 x i32> %418, i32 %420, i64 1, !dbg !15
  %422 = insertelement <4 x i32> %421, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc665 = insertelement <4 x i32> %422, i32 131072, i64 3, !dbg !15
  %423 = ptrtoint ptr addrspace(3) %413 to i32, !dbg !15
  %dma.buffer.lds666 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %423), !dbg !15
  %424 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds666, i32 %dma.buffer.offset664, <4 x i32> %dma.buffer.rsrc665) #10, !dbg !15
  %425 = load <4 x half>, ptr addrspace(3) %address.local126, align 2, !dbg !12
  %426 = load <4 x half>, ptr addrspace(3) %address.local128, align 2, !dbg !12
  %427 = fmul <4 x half> %425, splat (half 0xH3015), !dbg !12
  %428 = fmul <4 x half> %426, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %429 = add nsw i64 %83, %8, !dbg !17
  %430 = shl i64 %429, 19, !dbg !17
  %431 = and i64 %430, 16252928, !dbg !17
  %432 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %431, !dbg !17
  %433 = getelementptr [2 x i8], ptr addrspace(1) %432, i64 %89, !dbg !17
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local92, !dbg !17
  %address.local255 = ptrtoint ptr addrspace(3) %address.local254 to i64, !dbg !17
  %address.local256 = trunc i64 %address.local255 to i32, !dbg !17
  %434 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local256), !dbg !17
  %435 = zext i32 %434 to i64, !dbg !17
  %436 = inttoptr i64 %435 to ptr addrspace(3), !dbg !17
  %address.local257 = shl nuw nsw i64 %address.local, 5, !dbg !17
  %address.local258 = and i64 %address.local257, 8064, !dbg !17
  %address.local259 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local260 = and i64 %address.local259, 24, !dbg !17
  %address.local261 = or disjoint i64 %address.local258, %address.local260, !dbg !17
  %address.local262 = getelementptr [2 x i8], ptr addrspace(1) %433, i64 %address.local261, !dbg !17
  %437 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %438 = ptrtoint ptr addrspace(1) %address.local262 to i64, !dbg !17
  %439 = sub i64 %438, %437, !dbg !17
  %dma.buffer.offset661 = trunc i64 %439 to i32, !dbg !17
  %440 = trunc i64 %437 to i32, !dbg !17
  %441 = insertelement <4 x i32> poison, i32 %440, i64 0, !dbg !17
  %442 = lshr i64 %437, 32, !dbg !17
  %443 = trunc i64 %442 to i32, !dbg !17
  %444 = insertelement <4 x i32> %441, i32 %443, i64 1, !dbg !17
  %445 = insertelement <4 x i32> %444, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc662 = insertelement <4 x i32> %445, i32 131072, i64 3, !dbg !17
  %446 = ptrtoint ptr addrspace(3) %436 to i32, !dbg !17
  %dma.buffer.lds663 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %446), !dbg !17
  %447 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds663, i32 %dma.buffer.offset661, <4 x i32> %dma.buffer.rsrc662) #10, !dbg !17
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local104, !dbg !17
  %address.local264 = ptrtoint ptr addrspace(3) %address.local263 to i64, !dbg !17
  %address.local265 = trunc i64 %address.local264 to i32, !dbg !17
  %448 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local265), !dbg !17
  %449 = zext i32 %448 to i64, !dbg !17
  %450 = inttoptr i64 %449 to ptr addrspace(3), !dbg !17
  %address.local266 = shl nuw nsw i64 %address.local102, 5, !dbg !17
  %address.local267 = and i64 %address.local266, 16256, !dbg !17
  %address.local268 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local269 = and i64 %address.local268, 24, !dbg !17
  %address.local270 = or disjoint i64 %address.local267, %address.local269, !dbg !17
  %address.local271 = getelementptr [2 x i8], ptr addrspace(1) %433, i64 %address.local270, !dbg !17
  %451 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %452 = ptrtoint ptr addrspace(1) %address.local271 to i64, !dbg !17
  %453 = sub i64 %452, %451, !dbg !17
  %dma.buffer.offset658 = trunc i64 %453 to i32, !dbg !17
  %454 = trunc i64 %451 to i32, !dbg !17
  %455 = insertelement <4 x i32> poison, i32 %454, i64 0, !dbg !17
  %456 = lshr i64 %451, 32, !dbg !17
  %457 = trunc i64 %456 to i32, !dbg !17
  %458 = insertelement <4 x i32> %455, i32 %457, i64 1, !dbg !17
  %459 = insertelement <4 x i32> %458, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc659 = insertelement <4 x i32> %459, i32 131072, i64 3, !dbg !17
  %460 = ptrtoint ptr addrspace(3) %450 to i32, !dbg !17
  %dma.buffer.lds660 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %460), !dbg !17
  %461 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds660, i32 %dma.buffer.offset658, <4 x i32> %dma.buffer.rsrc659) #10, !dbg !17
  %462 = load <4 x half>, ptr addrspace(3) %address.local141, align 2, !dbg !16
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %462, <4 x float> %332) #9, !dbg !16
  %464 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !16
  %465 = load <4 x half>, ptr addrspace(3) %address.local171, align 2, !dbg !16
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %464, <4 x float> %463) #9, !dbg !16
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %465, <4 x float> %336) #9, !dbg !16
  %468 = load <4 x half>, ptr addrspace(3) %address.local177, align 2, !dbg !16
  %469 = load <4 x half>, ptr addrspace(3) %address.local183, align 2, !dbg !16
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %468, <4 x float> %467) #9, !dbg !16
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %469, <4 x float> %340) #9, !dbg !16
  %472 = load <4 x half>, ptr addrspace(3) %address.local189, align 2, !dbg !16
  %473 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !16
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %472, <4 x float> %471) #9, !dbg !16
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %473, <4 x float> %344) #9, !dbg !16
  %476 = load <4 x half>, ptr addrspace(3) %address.local201, align 2, !dbg !16
  %477 = load <4 x half>, ptr addrspace(3) %address.local203, align 2, !dbg !16
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %476, <4 x float> %475) #9, !dbg !16
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %477, <4 x float> %348) #9, !dbg !16
  %480 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %481 = load <4 x half>, ptr addrspace(3) %address.local212, align 2, !dbg !16
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %480, <4 x float> %479) #9, !dbg !16
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %481, <4 x float> %352) #9, !dbg !16
  %484 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !16
  %485 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !16
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %484, <4 x float> %483) #9, !dbg !16
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %485, <4 x float> %356) #9, !dbg !16
  %488 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !16
  %489 = load <4 x half>, ptr addrspace(3) %address.local238, align 2, !dbg !16
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %488, <4 x float> %487) #9, !dbg !16
  %491 = load <4 x half>, ptr addrspace(3) %address.local244, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %489, <4 x float> %359) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %493 = or disjoint i64 %66, 127, !dbg !13
  %494 = icmp uge i64 %11, %493, !dbg !13
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %491, <4 x float> %492) #9, !dbg !16
  %496 = and i1 %31, %494, !dbg !13
  br i1 %496, label %497, label %498, !dbg !13

497:                                              ; preds = %65
  br label %573, !dbg !13

498:                                              ; preds = %65
  %499 = add nsw i64 %11, %27, !dbg !13
  %500 = shl nuw nsw i64 %28, 4, !dbg !13
  %501 = add nsw i64 %499, %500, !dbg !13
  %502 = and i32 %26, 240, !dbg !13
  %index.compact = zext nneg i32 %502 to i64, !dbg !13
  %503 = sub nsw i64 %501, %index.compact, !dbg !13
  %504 = add nsw i64 %503, 1, !dbg !22
  %505 = and i64 %27, 48, !dbg !13
  %506 = or disjoint i64 %66, %505, !dbg !13
  %507 = lshr i64 %504, 32, !dbg !23
  %compare.high = trunc i64 %507 to i32, !dbg !23
  %508 = lshr i64 %506, 32, !dbg !23
  %compare.high519 = trunc i64 %508 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high519, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high519, !dbg !23
  %compare.low.base = trunc i64 %504 to i32, !dbg !23
  %compare.low.base520 = trunc i64 %506 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base520, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %509 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low521 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result522 = select i1 %compare.high.equal, i1 %compare.low521, i1 %compare.high.order, !dbg !23
  %510 = select i1 %compare.result522, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local524 = or i32 %compare.local.base523, 2, !dbg !23
  %compare.low525 = icmp ugt i32 %compare.low.base, %compare.local524, !dbg !23
  %compare.result526 = select i1 %compare.high.equal, i1 %compare.low525, i1 %compare.high.order, !dbg !23
  %511 = select i1 %compare.result526, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local528 = or i32 %compare.local.base527, 3, !dbg !23
  %compare.low529 = icmp ugt i32 %compare.low.base, %compare.local528, !dbg !23
  %compare.result530 = select i1 %compare.high.equal, i1 %compare.low529, i1 %compare.high.order, !dbg !23
  %512 = select i1 %compare.result530, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local532 = or i32 %compare.local.base531, 4, !dbg !23
  %compare.low533 = icmp ugt i32 %compare.low.base, %compare.local532, !dbg !23
  %compare.result534 = select i1 %compare.high.equal, i1 %compare.low533, i1 %compare.high.order, !dbg !23
  %513 = select i1 %compare.result534, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local536 = or i32 %compare.local.base535, 5, !dbg !23
  %compare.low537 = icmp ugt i32 %compare.low.base, %compare.local536, !dbg !23
  %compare.result538 = select i1 %compare.high.equal, i1 %compare.low537, i1 %compare.high.order, !dbg !23
  %514 = select i1 %compare.result538, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local540 = or i32 %compare.local.base539, 6, !dbg !23
  %compare.low541 = icmp ugt i32 %compare.low.base, %compare.local540, !dbg !23
  %compare.result542 = select i1 %compare.high.equal, i1 %compare.low541, i1 %compare.high.order, !dbg !23
  %515 = select i1 %compare.result542, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local544 = or i32 %compare.local.base543, 7, !dbg !23
  %compare.low545 = icmp ugt i32 %compare.low.base, %compare.local544, !dbg !23
  %compare.result546 = select i1 %compare.high.equal, i1 %compare.low545, i1 %compare.high.order, !dbg !23
  %516 = select i1 %compare.result546, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local548 = or i32 %compare.local.base547, 8, !dbg !23
  %compare.low549 = icmp ugt i32 %compare.low.base, %compare.local548, !dbg !23
  %compare.result550 = select i1 %compare.high.equal, i1 %compare.low549, i1 %compare.high.order, !dbg !23
  %517 = select i1 %compare.result550, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local552 = or i32 %compare.local.base551, 9, !dbg !23
  %compare.low553 = icmp ugt i32 %compare.low.base, %compare.local552, !dbg !23
  %compare.result554 = select i1 %compare.high.equal, i1 %compare.low553, i1 %compare.high.order, !dbg !23
  %518 = select i1 %compare.result554, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local556 = or i32 %compare.local.base555, 10, !dbg !23
  %compare.low557 = icmp ugt i32 %compare.low.base, %compare.local556, !dbg !23
  %compare.result558 = select i1 %compare.high.equal, i1 %compare.low557, i1 %compare.high.order, !dbg !23
  %519 = select i1 %compare.result558, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base559 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local560 = or i32 %compare.local.base559, 11, !dbg !23
  %compare.low561 = icmp ugt i32 %compare.low.base, %compare.local560, !dbg !23
  %compare.result562 = select i1 %compare.high.equal, i1 %compare.low561, i1 %compare.high.order, !dbg !23
  %520 = select i1 %compare.result562, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base563 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local564 = or i32 %compare.local.base563, 12, !dbg !23
  %compare.low565 = icmp ugt i32 %compare.low.base, %compare.local564, !dbg !23
  %compare.result566 = select i1 %compare.high.equal, i1 %compare.low565, i1 %compare.high.order, !dbg !23
  %521 = select i1 %compare.result566, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base567 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local568 = or i32 %compare.local.base567, 13, !dbg !23
  %compare.low569 = icmp ugt i32 %compare.low.base, %compare.local568, !dbg !23
  %compare.result570 = select i1 %compare.high.equal, i1 %compare.low569, i1 %compare.high.order, !dbg !23
  %522 = select i1 %compare.result570, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base571 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local572 = or i32 %compare.local.base571, 14, !dbg !23
  %compare.low573 = icmp ugt i32 %compare.low.base, %compare.local572, !dbg !23
  %compare.result574 = select i1 %compare.high.equal, i1 %compare.low573, i1 %compare.high.order, !dbg !23
  %523 = select i1 %compare.result574, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base575 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local576 = or i32 %compare.local.base575, 15, !dbg !23
  %compare.low577 = icmp ugt i32 %compare.low.base, %compare.local576, !dbg !23
  %compare.result578 = select i1 %compare.high.equal, i1 %compare.low577, i1 %compare.high.order, !dbg !23
  %524 = select i1 %compare.result578, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base579 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local580 = or i32 %compare.local.base579, 64, !dbg !23
  %compare.low581 = icmp ugt i32 %compare.low.base, %compare.local580, !dbg !23
  %compare.result582 = select i1 %compare.high.equal, i1 %compare.low581, i1 %compare.high.order, !dbg !23
  %525 = select i1 %compare.result582, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base583 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local584 = or i32 %compare.local.base583, 65, !dbg !23
  %compare.low585 = icmp ugt i32 %compare.low.base, %compare.local584, !dbg !23
  %compare.result586 = select i1 %compare.high.equal, i1 %compare.low585, i1 %compare.high.order, !dbg !23
  %526 = select i1 %compare.result586, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base587 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local588 = or i32 %compare.local.base587, 66, !dbg !23
  %compare.low589 = icmp ugt i32 %compare.low.base, %compare.local588, !dbg !23
  %compare.result590 = select i1 %compare.high.equal, i1 %compare.low589, i1 %compare.high.order, !dbg !23
  %527 = select i1 %compare.result590, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base591 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local592 = or i32 %compare.local.base591, 67, !dbg !23
  %compare.low593 = icmp ugt i32 %compare.low.base, %compare.local592, !dbg !23
  %compare.result594 = select i1 %compare.high.equal, i1 %compare.low593, i1 %compare.high.order, !dbg !23
  %528 = select i1 %compare.result594, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base595 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local596 = or i32 %compare.local.base595, 68, !dbg !23
  %compare.low597 = icmp ugt i32 %compare.low.base, %compare.local596, !dbg !23
  %compare.result598 = select i1 %compare.high.equal, i1 %compare.low597, i1 %compare.high.order, !dbg !23
  %529 = select i1 %compare.result598, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base599 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local600 = or i32 %compare.local.base599, 69, !dbg !23
  %compare.low601 = icmp ugt i32 %compare.low.base, %compare.local600, !dbg !23
  %compare.result602 = select i1 %compare.high.equal, i1 %compare.low601, i1 %compare.high.order, !dbg !23
  %530 = select i1 %compare.result602, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base603 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local604 = or i32 %compare.local.base603, 70, !dbg !23
  %compare.low605 = icmp ugt i32 %compare.low.base, %compare.local604, !dbg !23
  %compare.result606 = select i1 %compare.high.equal, i1 %compare.low605, i1 %compare.high.order, !dbg !23
  %531 = select i1 %compare.result606, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base607 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local608 = or i32 %compare.local.base607, 71, !dbg !23
  %compare.low609 = icmp ugt i32 %compare.low.base, %compare.local608, !dbg !23
  %compare.result610 = select i1 %compare.high.equal, i1 %compare.low609, i1 %compare.high.order, !dbg !23
  %532 = select i1 %compare.result610, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base611 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local612 = or i32 %compare.local.base611, 72, !dbg !23
  %compare.low613 = icmp ugt i32 %compare.low.base, %compare.local612, !dbg !23
  %compare.result614 = select i1 %compare.high.equal, i1 %compare.low613, i1 %compare.high.order, !dbg !23
  %533 = select i1 %compare.result614, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base615 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local616 = or i32 %compare.local.base615, 73, !dbg !23
  %compare.low617 = icmp ugt i32 %compare.low.base, %compare.local616, !dbg !23
  %compare.result618 = select i1 %compare.high.equal, i1 %compare.low617, i1 %compare.high.order, !dbg !23
  %534 = select i1 %compare.result618, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base619 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local620 = or i32 %compare.local.base619, 74, !dbg !23
  %compare.low621 = icmp ugt i32 %compare.low.base, %compare.local620, !dbg !23
  %compare.result622 = select i1 %compare.high.equal, i1 %compare.low621, i1 %compare.high.order, !dbg !23
  %535 = select i1 %compare.result622, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base623 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local624 = or i32 %compare.local.base623, 75, !dbg !23
  %compare.low625 = icmp ugt i32 %compare.low.base, %compare.local624, !dbg !23
  %compare.result626 = select i1 %compare.high.equal, i1 %compare.low625, i1 %compare.high.order, !dbg !23
  %536 = select i1 %compare.result626, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base627 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local628 = or i32 %compare.local.base627, 76, !dbg !23
  %compare.low629 = icmp ugt i32 %compare.low.base, %compare.local628, !dbg !23
  %compare.result630 = select i1 %compare.high.equal, i1 %compare.low629, i1 %compare.high.order, !dbg !23
  %537 = select i1 %compare.result630, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base631 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local632 = or i32 %compare.local.base631, 77, !dbg !23
  %compare.low633 = icmp ugt i32 %compare.low.base, %compare.local632, !dbg !23
  %compare.result634 = select i1 %compare.high.equal, i1 %compare.low633, i1 %compare.high.order, !dbg !23
  %538 = select i1 %compare.result634, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base635 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local636 = or i32 %compare.local.base635, 78, !dbg !23
  %compare.low637 = icmp ugt i32 %compare.low.base, %compare.local636, !dbg !23
  %compare.result638 = select i1 %compare.high.equal, i1 %compare.low637, i1 %compare.high.order, !dbg !23
  %539 = select i1 %compare.result638, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base639 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base520, i32 31) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local640 = or i32 %compare.local.base639, 79, !dbg !23
  %compare.low641 = icmp ugt i32 %compare.low.base, %compare.local640, !dbg !23
  %compare.result642 = select i1 %compare.high.equal, i1 %compare.low641, i1 %compare.high.order, !dbg !23
  %540 = select i1 %compare.result642, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %541 = insertelement <32 x float> poison, float %509, i64 0, !dbg !13
  %542 = insertelement <32 x float> %541, float %510, i64 1, !dbg !13
  %543 = insertelement <32 x float> %542, float %511, i64 2, !dbg !13
  %544 = insertelement <32 x float> %543, float %512, i64 3, !dbg !13
  %545 = insertelement <32 x float> %544, float %513, i64 4, !dbg !13
  %546 = insertelement <32 x float> %545, float %514, i64 5, !dbg !13
  %547 = insertelement <32 x float> %546, float %515, i64 6, !dbg !13
  %548 = insertelement <32 x float> %547, float %516, i64 7, !dbg !13
  %549 = insertelement <32 x float> %548, float %517, i64 8, !dbg !13
  %550 = insertelement <32 x float> %549, float %518, i64 9, !dbg !13
  %551 = insertelement <32 x float> %550, float %519, i64 10, !dbg !13
  %552 = insertelement <32 x float> %551, float %520, i64 11, !dbg !13
  %553 = insertelement <32 x float> %552, float %521, i64 12, !dbg !13
  %554 = insertelement <32 x float> %553, float %522, i64 13, !dbg !13
  %555 = insertelement <32 x float> %554, float %523, i64 14, !dbg !13
  %556 = insertelement <32 x float> %555, float %524, i64 15, !dbg !13
  %557 = insertelement <32 x float> %556, float %525, i64 16, !dbg !13
  %558 = insertelement <32 x float> %557, float %526, i64 17, !dbg !13
  %559 = insertelement <32 x float> %558, float %527, i64 18, !dbg !13
  %560 = insertelement <32 x float> %559, float %528, i64 19, !dbg !13
  %561 = insertelement <32 x float> %560, float %529, i64 20, !dbg !13
  %562 = insertelement <32 x float> %561, float %530, i64 21, !dbg !13
  %563 = insertelement <32 x float> %562, float %531, i64 22, !dbg !13
  %564 = insertelement <32 x float> %563, float %532, i64 23, !dbg !13
  %565 = insertelement <32 x float> %564, float %533, i64 24, !dbg !13
  %566 = insertelement <32 x float> %565, float %534, i64 25, !dbg !13
  %567 = insertelement <32 x float> %566, float %535, i64 26, !dbg !13
  %568 = insertelement <32 x float> %567, float %536, i64 27, !dbg !13
  %569 = insertelement <32 x float> %568, float %537, i64 28, !dbg !13
  %570 = insertelement <32 x float> %569, float %538, i64 29, !dbg !13
  %571 = insertelement <32 x float> %570, float %539, i64 30, !dbg !13
  %572 = insertelement <32 x float> %571, float %540, i64 31, !dbg !13
  br label %573, !dbg !13

573:                                              ; preds = %497, %498
  %574 = phi <32 x float> [ %572, %498 ], [ zeroinitializer, %497 ], !dbg !13
  br label %575, !dbg !13

575:                                              ; preds = %573
  %576 = shufflevector <32 x float> %574, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %577 = fadd <4 x float> %466, %576, !dbg !25
  %578 = extractelement <4 x float> %577, i64 0, !dbg !25
  %579 = extractelement <4 x float> %577, i64 1, !dbg !25
  %580 = extractelement <4 x float> %577, i64 2, !dbg !25
  %581 = extractelement <4 x float> %577, i64 3, !dbg !25
  %582 = shufflevector <32 x float> %574, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %583 = fadd <4 x float> %470, %582, !dbg !25
  %584 = extractelement <4 x float> %583, i64 0, !dbg !25
  %585 = extractelement <4 x float> %583, i64 1, !dbg !25
  %586 = extractelement <4 x float> %583, i64 2, !dbg !25
  %587 = extractelement <4 x float> %583, i64 3, !dbg !25
  %588 = shufflevector <32 x float> %574, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %589 = fadd <4 x float> %474, %588, !dbg !25
  %590 = extractelement <4 x float> %589, i64 0, !dbg !25
  %591 = extractelement <4 x float> %589, i64 1, !dbg !25
  %592 = extractelement <4 x float> %589, i64 2, !dbg !25
  %593 = extractelement <4 x float> %589, i64 3, !dbg !25
  %594 = shufflevector <32 x float> %574, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %595 = fadd <4 x float> %478, %594, !dbg !25
  %596 = extractelement <4 x float> %595, i64 0, !dbg !25
  %597 = extractelement <4 x float> %595, i64 1, !dbg !25
  %598 = extractelement <4 x float> %595, i64 2, !dbg !25
  %599 = extractelement <4 x float> %595, i64 3, !dbg !25
  %600 = shufflevector <32 x float> %574, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %601 = fadd <4 x float> %482, %600, !dbg !25
  %602 = extractelement <4 x float> %601, i64 0, !dbg !25
  %603 = extractelement <4 x float> %601, i64 1, !dbg !25
  %604 = extractelement <4 x float> %601, i64 2, !dbg !25
  %605 = extractelement <4 x float> %601, i64 3, !dbg !25
  %606 = shufflevector <32 x float> %574, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %607 = fadd <4 x float> %486, %606, !dbg !25
  %608 = extractelement <4 x float> %607, i64 0, !dbg !25
  %609 = extractelement <4 x float> %607, i64 1, !dbg !25
  %610 = extractelement <4 x float> %607, i64 2, !dbg !25
  %611 = extractelement <4 x float> %607, i64 3, !dbg !25
  %612 = shufflevector <32 x float> %574, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %613 = fadd <4 x float> %490, %612, !dbg !25
  %614 = extractelement <4 x float> %613, i64 0, !dbg !25
  %615 = extractelement <4 x float> %613, i64 1, !dbg !25
  %616 = extractelement <4 x float> %613, i64 2, !dbg !25
  %617 = extractelement <4 x float> %613, i64 3, !dbg !25
  %618 = shufflevector <32 x float> %574, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %619 = fadd <4 x float> %495, %618, !dbg !25
  %620 = extractelement <4 x float> %619, i64 0, !dbg !25
  %621 = extractelement <4 x float> %619, i64 1, !dbg !25
  %622 = extractelement <4 x float> %619, i64 2, !dbg !25
  %623 = extractelement <4 x float> %619, i64 3, !dbg !25
  %624 = call float @llvm.maxnum.f32(float %578, float 0xFFF0000000000000), !dbg !26
  %625 = call float @llvm.maxnum.f32(float %624, float %579), !dbg !26
  %626 = call float @llvm.maxnum.f32(float %625, float %580), !dbg !26
  %627 = call float @llvm.maxnum.f32(float %626, float %581), !dbg !26
  %628 = call float @llvm.maxnum.f32(float %627, float %584), !dbg !26
  %629 = call float @llvm.maxnum.f32(float %628, float %585), !dbg !26
  %630 = call float @llvm.maxnum.f32(float %629, float %586), !dbg !26
  %631 = call float @llvm.maxnum.f32(float %630, float %587), !dbg !26
  %632 = call float @llvm.maxnum.f32(float %631, float %590), !dbg !26
  %633 = call float @llvm.maxnum.f32(float %632, float %591), !dbg !26
  %634 = call float @llvm.maxnum.f32(float %633, float %592), !dbg !26
  %635 = call float @llvm.maxnum.f32(float %634, float %593), !dbg !26
  %636 = call float @llvm.maxnum.f32(float %635, float %596), !dbg !26
  %637 = call float @llvm.maxnum.f32(float %636, float %597), !dbg !26
  %638 = call float @llvm.maxnum.f32(float %637, float %598), !dbg !26
  %639 = call float @llvm.maxnum.f32(float %638, float %599), !dbg !26
  %640 = call float @llvm.maxnum.f32(float %639, float %602), !dbg !26
  %641 = call float @llvm.maxnum.f32(float %640, float %603), !dbg !26
  %642 = call float @llvm.maxnum.f32(float %641, float %604), !dbg !26
  %643 = call float @llvm.maxnum.f32(float %642, float %605), !dbg !26
  %644 = call float @llvm.maxnum.f32(float %643, float %608), !dbg !26
  %645 = call float @llvm.maxnum.f32(float %644, float %609), !dbg !26
  %646 = call float @llvm.maxnum.f32(float %645, float %610), !dbg !26
  %647 = call float @llvm.maxnum.f32(float %646, float %611), !dbg !26
  %648 = call float @llvm.maxnum.f32(float %647, float %614), !dbg !26
  %649 = call float @llvm.maxnum.f32(float %648, float %615), !dbg !26
  %650 = call float @llvm.maxnum.f32(float %649, float %616), !dbg !26
  %651 = call float @llvm.maxnum.f32(float %650, float %617), !dbg !26
  %652 = call float @llvm.maxnum.f32(float %651, float %620), !dbg !26
  %653 = call float @llvm.maxnum.f32(float %652, float %621), !dbg !26
  %654 = call float @llvm.maxnum.f32(float %653, float %622), !dbg !26
  %655 = call float @llvm.maxnum.f32(float %654, float %623), !dbg !26
  %656 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %657 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %656), !dbg !26
  %658 = add nuw nsw i32 %657, 64, !dbg !26
  %659 = xor i32 %657, 16, !dbg !26
  %660 = and i32 %658, 192, !dbg !26
  %661 = icmp samesign ult i32 %659, %660, !dbg !26
  %662 = select i1 %661, i32 %659, i32 %657, !dbg !26
  %663 = shl nuw nsw i32 %662, 2, !dbg !26
  %664 = bitcast float %655 to i32, !dbg !26
  %665 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %664), !dbg !26
  %666 = bitcast i32 %665 to float, !dbg !26
  %667 = call float @llvm.maxnum.f32(float %655, float %666), !dbg !26
  %668 = xor i32 %657, 32, !dbg !26
  %669 = icmp samesign ult i32 %668, %660, !dbg !26
  %670 = select i1 %669, i32 %668, i32 %657, !dbg !26
  %671 = shl nuw nsw i32 %670, 2, !dbg !26
  %672 = bitcast float %667 to i32, !dbg !26
  %673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %672), !dbg !26
  %674 = bitcast i32 %673 to float, !dbg !26
  %675 = call float @llvm.maxnum.f32(float %667, float %674), !dbg !26
  %676 = insertelement <1 x float> poison, float %675, i64 0, !dbg !26
  %677 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %676, <1 x float> %54), !dbg !26
  %678 = shufflevector <1 x float> %677, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %679 = fsub <4 x float> %577, %678, !dbg !27
  %680 = fsub <4 x float> %583, %678, !dbg !27
  %681 = fsub <4 x float> %589, %678, !dbg !27
  %682 = fsub <4 x float> %595, %678, !dbg !27
  %683 = fsub <4 x float> %601, %678, !dbg !27
  %684 = fsub <4 x float> %607, %678, !dbg !27
  %685 = fsub <4 x float> %613, %678, !dbg !27
  %686 = fsub <4 x float> %619, %678, !dbg !27
  %687 = fsub <1 x float> %54, %677, !dbg !28
  %688 = extractelement <1 x float> %687, i64 0, !dbg !29
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !29
  %690 = insertelement <4 x float> poison, float %689, i64 0, !dbg !34
  %691 = shufflevector <4 x float> %690, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %692 = fmul <4 x float> %62, %691, !dbg !34
  %693 = fmul <4 x float> %61, %691, !dbg !34
  %694 = fmul <4 x float> %59, %691, !dbg !34
  %695 = fmul <4 x float> %58, %691, !dbg !34
  %696 = fmul <4 x float> %56, %691, !dbg !34
  %697 = fmul <4 x float> %55, %691, !dbg !34
  %698 = fmul <4 x float> %51, %691, !dbg !34
  %699 = fmul <4 x float> %50, %691, !dbg !34
  %700 = extractelement <1 x float> %53, i64 0, !dbg !30
  %701 = fmul float %700, %689, !dbg !30
  %702 = extractelement <4 x float> %679, i64 0, !dbg !31
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !31
  %704 = insertelement <4 x float> poison, float %703, i64 0, !dbg !33
  %705 = extractelement <4 x float> %679, i64 1, !dbg !31
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !31
  %707 = insertelement <4 x float> %704, float %706, i64 1, !dbg !33
  %708 = fadd float %703, %706, !dbg !32
  %709 = extractelement <4 x float> %679, i64 2, !dbg !31
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !31
  %711 = insertelement <4 x float> %707, float %710, i64 2, !dbg !33
  %712 = fadd float %708, %710, !dbg !32
  %713 = extractelement <4 x float> %679, i64 3, !dbg !31
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !31
  %715 = insertelement <4 x float> %711, float %714, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %715) #9, !dbg !33, !frisk.register_lifetimes !1
  %716 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %716) #9, !dbg !32, !frisk.register_lifetimes !1
  %717 = fadd float %712, %714, !dbg !32
  %718 = extractelement <4 x float> %680, i64 0, !dbg !31
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !31
  %720 = insertelement <4 x float> poison, float %719, i64 0, !dbg !33
  %721 = fadd float %717, %719, !dbg !32
  %722 = extractelement <4 x float> %680, i64 1, !dbg !31
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !31
  %724 = insertelement <4 x float> %720, float %723, i64 1, !dbg !33
  %725 = fadd float %721, %723, !dbg !32
  %726 = extractelement <4 x float> %680, i64 2, !dbg !31
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !31
  %728 = insertelement <4 x float> %724, float %727, i64 2, !dbg !33
  %729 = fadd float %725, %727, !dbg !32
  %730 = extractelement <4 x float> %680, i64 3, !dbg !31
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !31
  %732 = insertelement <4 x float> %728, float %731, i64 3, !dbg !33
  %math.result67 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %732) #9, !dbg !33, !frisk.register_lifetimes !1
  %733 = fptrunc <4 x float> %math.result67 to <4 x half>, !dbg !33
  %math.packed68 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %733) #9, !dbg !32, !frisk.register_lifetimes !1
  %734 = fadd float %729, %731, !dbg !32
  %735 = extractelement <4 x float> %681, i64 0, !dbg !31
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !31
  %737 = insertelement <4 x float> poison, float %736, i64 0, !dbg !33
  %738 = fadd float %734, %736, !dbg !32
  %739 = extractelement <4 x float> %681, i64 1, !dbg !31
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !31
  %741 = insertelement <4 x float> %737, float %740, i64 1, !dbg !33
  %742 = fadd float %738, %740, !dbg !32
  %743 = extractelement <4 x float> %681, i64 2, !dbg !31
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !31
  %745 = insertelement <4 x float> %741, float %744, i64 2, !dbg !33
  %746 = fadd float %742, %744, !dbg !32
  %747 = extractelement <4 x float> %681, i64 3, !dbg !31
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !31
  %749 = insertelement <4 x float> %745, float %748, i64 3, !dbg !33
  %math.result69 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %749) #9, !dbg !33, !frisk.register_lifetimes !1
  %750 = fptrunc <4 x float> %math.result69 to <4 x half>, !dbg !33
  %math.packed70 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %750) #9, !dbg !32, !frisk.register_lifetimes !1
  %751 = fadd float %746, %748, !dbg !32
  %752 = extractelement <4 x float> %682, i64 0, !dbg !31
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !31
  %754 = insertelement <4 x float> poison, float %753, i64 0, !dbg !33
  %755 = fadd float %751, %753, !dbg !32
  %756 = extractelement <4 x float> %682, i64 1, !dbg !31
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !31
  %758 = insertelement <4 x float> %754, float %757, i64 1, !dbg !33
  %759 = fadd float %755, %757, !dbg !32
  %760 = extractelement <4 x float> %682, i64 2, !dbg !31
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !31
  %762 = insertelement <4 x float> %758, float %761, i64 2, !dbg !33
  %763 = fadd float %759, %761, !dbg !32
  %764 = extractelement <4 x float> %682, i64 3, !dbg !31
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !31
  %766 = insertelement <4 x float> %762, float %765, i64 3, !dbg !33
  %math.result71 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %766) #9, !dbg !33, !frisk.register_lifetimes !1
  %767 = fptrunc <4 x float> %math.result71 to <4 x half>, !dbg !33
  %math.packed72 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %767) #9, !dbg !32, !frisk.register_lifetimes !1
  %768 = fadd float %763, %765, !dbg !32
  %769 = extractelement <4 x float> %683, i64 0, !dbg !31
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !31
  %771 = insertelement <4 x float> poison, float %770, i64 0, !dbg !33
  %772 = fadd float %768, %770, !dbg !32
  %773 = extractelement <4 x float> %683, i64 1, !dbg !31
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !31
  %775 = insertelement <4 x float> %771, float %774, i64 1, !dbg !33
  %776 = fadd float %772, %774, !dbg !32
  %777 = extractelement <4 x float> %683, i64 2, !dbg !31
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !31
  %779 = insertelement <4 x float> %775, float %778, i64 2, !dbg !33
  %780 = fadd float %776, %778, !dbg !32
  %781 = extractelement <4 x float> %683, i64 3, !dbg !31
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !31
  %783 = insertelement <4 x float> %779, float %782, i64 3, !dbg !33
  %math.result73 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %783) #9, !dbg !33, !frisk.register_lifetimes !1
  %784 = fptrunc <4 x float> %math.result73 to <4 x half>, !dbg !33
  %math.packed74 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %784) #9, !dbg !32, !frisk.register_lifetimes !1
  %785 = fadd float %780, %782, !dbg !32
  %786 = extractelement <4 x float> %684, i64 0, !dbg !31
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !31
  %788 = insertelement <4 x float> poison, float %787, i64 0, !dbg !33
  %789 = fadd float %785, %787, !dbg !32
  %790 = extractelement <4 x float> %684, i64 1, !dbg !31
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !31
  %792 = insertelement <4 x float> %788, float %791, i64 1, !dbg !33
  %793 = fadd float %789, %791, !dbg !32
  %794 = extractelement <4 x float> %684, i64 2, !dbg !31
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !31
  %796 = insertelement <4 x float> %792, float %795, i64 2, !dbg !33
  %797 = fadd float %793, %795, !dbg !32
  %798 = extractelement <4 x float> %684, i64 3, !dbg !31
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !31
  %800 = insertelement <4 x float> %796, float %799, i64 3, !dbg !33
  %math.result75 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %800) #9, !dbg !33, !frisk.register_lifetimes !1
  %801 = fptrunc <4 x float> %math.result75 to <4 x half>, !dbg !33
  %math.packed76 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %801) #9, !dbg !32, !frisk.register_lifetimes !1
  %802 = fadd float %797, %799, !dbg !32
  %803 = extractelement <4 x float> %685, i64 0, !dbg !31
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !31
  %805 = insertelement <4 x float> poison, float %804, i64 0, !dbg !33
  %806 = fadd float %802, %804, !dbg !32
  %807 = extractelement <4 x float> %685, i64 1, !dbg !31
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !31
  %809 = insertelement <4 x float> %805, float %808, i64 1, !dbg !33
  %810 = fadd float %806, %808, !dbg !32
  %811 = extractelement <4 x float> %685, i64 2, !dbg !31
  %812 = call float @llvm.amdgcn.exp2.f32(float %811), !dbg !31
  %813 = insertelement <4 x float> %809, float %812, i64 2, !dbg !33
  %814 = fadd float %810, %812, !dbg !32
  %815 = extractelement <4 x float> %685, i64 3, !dbg !31
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !31
  %817 = insertelement <4 x float> %813, float %816, i64 3, !dbg !33
  %math.result77 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %817) #9, !dbg !33, !frisk.register_lifetimes !1
  %818 = fptrunc <4 x float> %math.result77 to <4 x half>, !dbg !33
  %math.packed78 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %818) #9, !dbg !32, !frisk.register_lifetimes !1
  %819 = fadd float %814, %816, !dbg !32
  %820 = extractelement <4 x float> %686, i64 0, !dbg !31
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !31
  %822 = insertelement <4 x float> poison, float %821, i64 0, !dbg !33
  %823 = fadd float %819, %821, !dbg !32
  %824 = extractelement <4 x float> %686, i64 1, !dbg !31
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !31
  %826 = insertelement <4 x float> %822, float %825, i64 1, !dbg !33
  %827 = fadd float %823, %825, !dbg !32
  %828 = extractelement <4 x float> %686, i64 2, !dbg !31
  %829 = call float @llvm.amdgcn.exp2.f32(float %828), !dbg !31
  %830 = insertelement <4 x float> %826, float %829, i64 2, !dbg !33
  %831 = fadd float %827, %829, !dbg !32
  %832 = extractelement <4 x float> %686, i64 3, !dbg !31
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !31
  %834 = insertelement <4 x float> %830, float %833, i64 3, !dbg !33
  %math.result79 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %834) #9, !dbg !33, !frisk.register_lifetimes !1
  %835 = fptrunc <4 x float> %math.result79 to <4 x half>, !dbg !33
  %math.packed80 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %835) #9, !dbg !32, !frisk.register_lifetimes !1
  %836 = fadd float %831, %833, !dbg !32
  %837 = bitcast float %836 to i32, !dbg !32
  %838 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %837), !dbg !32
  %839 = bitcast i32 %838 to float, !dbg !32
  %840 = fadd float %836, %839, !dbg !32
  %841 = bitcast float %840 to i32, !dbg !32
  %842 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %841), !dbg !32
  %843 = bitcast i32 %842 to float, !dbg !32
  %844 = fadd float %840, %843, !dbg !32
  %845 = fadd float %844, %701, !dbg !32
  %846 = insertelement <1 x float> poison, float %845, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread272 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local273 = shl nuw nsw i32 %address.thread272, 3, !dbg !14
  %address.local274 = and i32 %address.local273, 120, !dbg !14
  %address.local275 = shl nuw nsw i32 %address.thread272, 5, !dbg !14
  %address.local276 = and i32 %address.local275, 1536, !dbg !14
  %address.local277 = or disjoint i32 %address.local274, %address.local276, !dbg !14
  %address.local278 = zext nneg i32 %address.local277 to i64, !dbg !14
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local278, !dbg !14
  %847 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local279, i32 0) #0, !dbg !14
  %848 = bitcast <4 x i32> %847 to <8 x half>, !dbg !14
  %address.local280 = shl nuw nsw i32 %address.thread272, 3, !dbg !14
  %address.local281 = and i32 %address.local280, 120, !dbg !14
  %address.local282 = shl nuw nsw i32 %address.thread272, 5, !dbg !14
  %address.local283 = and i32 %address.local282, 1536, !dbg !14
  %address.local284 = or disjoint i32 %address.local281, %address.local283, !dbg !14
  %address.local285 = or disjoint i32 %address.local284, 128, !dbg !14
  %address.local286 = zext nneg i32 %address.local285 to i64, !dbg !14
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local286, !dbg !14
  %849 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local287, i32 0) #0, !dbg !14
  %850 = bitcast <4 x i32> %849 to <8 x half>, !dbg !14
  %address.local288 = shl nuw nsw i32 %address.thread272, 3, !dbg !14
  %address.local289 = and i32 %address.local288, 120, !dbg !14
  %address.local290 = shl nuw nsw i32 %address.thread272, 5, !dbg !14
  %address.local291 = and i32 %address.local290, 1536, !dbg !14
  %address.local292 = or disjoint i32 %address.local289, %address.local291, !dbg !14
  %address.local293 = or disjoint i32 %address.local292, 256, !dbg !14
  %address.local294 = zext nneg i32 %address.local293 to i64, !dbg !14
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local294, !dbg !14
  %851 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local295, i32 0) #0, !dbg !14
  %852 = bitcast <4 x i32> %851 to <8 x half>, !dbg !14
  %address.local296 = shl nuw nsw i32 %address.thread272, 3, !dbg !14
  %address.local297 = and i32 %address.local296, 120, !dbg !14
  %address.local298 = shl nuw nsw i32 %address.thread272, 5, !dbg !14
  %address.local299 = and i32 %address.local298, 1536, !dbg !14
  %address.local300 = or disjoint i32 %address.local297, %address.local299, !dbg !14
  %address.local301 = or disjoint i32 %address.local300, 384, !dbg !14
  %address.local302 = zext nneg i32 %address.local301 to i64, !dbg !14
  %address.local303 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local302, !dbg !14
  %853 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local303, i32 0) #0, !dbg !14
  %854 = bitcast <4 x i32> %853 to <8 x half>, !dbg !14
  %address.local304 = shl nuw nsw i32 %address.thread272, 3, !dbg !14
  %address.local305 = and i32 %address.local304, 120, !dbg !14
  %address.local306 = shl nuw nsw i32 %address.thread272, 5, !dbg !14
  %address.local307 = and i32 %address.local306, 1536, !dbg !14
  %address.local308 = or disjoint i32 %address.local305, %address.local307, !dbg !14
  %address.local309 = or disjoint i32 %address.local308, 2048, !dbg !14
  %address.local310 = zext nneg i32 %address.local309 to i64, !dbg !14
  %address.local311 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local310, !dbg !14
  %855 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local311, i32 0) #0, !dbg !14
  %856 = bitcast <4 x i32> %855 to <8 x half>, !dbg !14
  %address.local312 = shl nuw nsw i32 %address.thread272, 3, !dbg !14
  %address.local313 = and i32 %address.local312, 120, !dbg !14
  %address.local314 = shl nuw nsw i32 %address.thread272, 5, !dbg !14
  %address.local315 = and i32 %address.local314, 1536, !dbg !14
  %address.local316 = or disjoint i32 %address.local313, %address.local315, !dbg !14
  %address.local317 = or disjoint i32 %address.local316, 2176, !dbg !14
  %address.local318 = zext nneg i32 %address.local317 to i64, !dbg !14
  %address.local319 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local318, !dbg !14
  %857 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local319, i32 0) #0, !dbg !14
  %858 = bitcast <4 x i32> %857 to <8 x half>, !dbg !14
  %address.local320 = shl nuw nsw i32 %address.thread272, 3, !dbg !14
  %address.local321 = and i32 %address.local320, 120, !dbg !14
  %address.local322 = shl nuw nsw i32 %address.thread272, 5, !dbg !14
  %address.local323 = and i32 %address.local322, 1536, !dbg !14
  %address.local324 = or disjoint i32 %address.local321, %address.local323, !dbg !14
  %address.local325 = or disjoint i32 %address.local324, 2304, !dbg !14
  %address.local326 = zext nneg i32 %address.local325 to i64, !dbg !14
  %address.local327 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local326, !dbg !14
  %859 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local327, i32 0) #0, !dbg !14
  %860 = bitcast <4 x i32> %859 to <8 x half>, !dbg !14
  %address.local328 = shl nuw nsw i32 %address.thread272, 3, !dbg !14
  %address.local329 = and i32 %address.local328, 120, !dbg !14
  %address.local330 = shl nuw nsw i32 %address.thread272, 5, !dbg !14
  %address.local331 = and i32 %address.local330, 1536, !dbg !14
  %address.local332 = or disjoint i32 %address.local329, %address.local331, !dbg !14
  %address.local333 = or disjoint i32 %address.local332, 2432, !dbg !14
  %address.local334 = zext nneg i32 %address.local333 to i64, !dbg !14
  %address.local335 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local334, !dbg !14
  %861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local335, i32 0) #0, !dbg !14
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !14
  %863 = add nsw i64 %83, %8, !dbg !17
  %864 = shl i64 %863, 19, !dbg !17
  %865 = and i64 %864, 16252928, !dbg !17
  %866 = add i64 %865, %89, !dbg !17
  %867 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %866, !dbg !17
  %868 = getelementptr i8, ptr addrspace(1) %867, i64 64, !dbg !17
  %address.local336 = zext nneg i32 %address.thread272 to i64, !dbg !12
  %address.local337 = shl nuw nsw i64 %address.local336, 3, !dbg !11
  %address.local338 = and i64 %address.local337, 1536, !dbg !15
  %address.local339 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local338, !dbg !15
  %address.local340 = ptrtoint ptr addrspace(3) %address.local339 to i64, !dbg !15
  %address.local341 = trunc i64 %address.local340 to i32, !dbg !15
  %869 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local341), !dbg !17
  %870 = zext i32 %869 to i64, !dbg !17
  %871 = inttoptr i64 %870 to ptr addrspace(3), !dbg !17
  %address.local342 = shl nuw nsw i64 %address.local336, 5, !dbg !17
  %address.local343 = and i64 %address.local342, 8064, !dbg !17
  %address.local344 = shl nuw nsw i64 %address.local336, 3, !dbg !17
  %address.local345 = and i64 %address.local344, 24, !dbg !17
  %address.local346 = or disjoint i64 %address.local343, %address.local345, !dbg !17
  %address.local347 = getelementptr [2 x i8], ptr addrspace(1) %868, i64 %address.local346, !dbg !17
  %872 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %873 = ptrtoint ptr addrspace(1) %address.local347 to i64, !dbg !17
  %874 = sub i64 %873, %872, !dbg !17
  %dma.buffer.offset655 = trunc i64 %874 to i32, !dbg !17
  %875 = trunc i64 %872 to i32, !dbg !17
  %876 = insertelement <4 x i32> poison, i32 %875, i64 0, !dbg !17
  %877 = lshr i64 %872, 32, !dbg !17
  %878 = trunc i64 %877 to i32, !dbg !17
  %879 = insertelement <4 x i32> %876, i32 %878, i64 1, !dbg !17
  %880 = insertelement <4 x i32> %879, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc656 = insertelement <4 x i32> %880, i32 131072, i64 3, !dbg !17
  %881 = ptrtoint ptr addrspace(3) %871 to i32, !dbg !17
  %dma.buffer.lds657 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %881), !dbg !17
  %882 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds657, i32 %dma.buffer.offset655, <4 x i32> %dma.buffer.rsrc656) #10, !dbg !17
  %address.local348 = or disjoint i64 %address.local336, 256, !dbg !15
  %address.local349 = shl nuw nsw i64 %address.local348, 3, !dbg !15
  %address.local350 = and i64 %address.local349, 3584, !dbg !15
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local350, !dbg !15
  %address.local352 = ptrtoint ptr addrspace(3) %address.local351 to i64, !dbg !15
  %address.local353 = trunc i64 %address.local352 to i32, !dbg !15
  %883 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local353), !dbg !17
  %884 = zext i32 %883 to i64, !dbg !17
  %885 = inttoptr i64 %884 to ptr addrspace(3), !dbg !17
  %address.local354 = shl nuw nsw i64 %address.local348, 5, !dbg !17
  %address.local355 = and i64 %address.local354, 16256, !dbg !17
  %address.local356 = shl nuw nsw i64 %address.local336, 3, !dbg !17
  %address.local357 = and i64 %address.local356, 24, !dbg !17
  %address.local358 = or disjoint i64 %address.local355, %address.local357, !dbg !17
  %address.local359 = getelementptr [2 x i8], ptr addrspace(1) %868, i64 %address.local358, !dbg !17
  %886 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %887 = ptrtoint ptr addrspace(1) %address.local359 to i64, !dbg !17
  %888 = sub i64 %887, %886, !dbg !17
  %dma.buffer.offset652 = trunc i64 %888 to i32, !dbg !17
  %889 = trunc i64 %886 to i32, !dbg !17
  %890 = insertelement <4 x i32> poison, i32 %889, i64 0, !dbg !17
  %891 = lshr i64 %886, 32, !dbg !17
  %892 = trunc i64 %891 to i32, !dbg !17
  %893 = insertelement <4 x i32> %890, i32 %892, i64 1, !dbg !17
  %894 = insertelement <4 x i32> %893, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc653 = insertelement <4 x i32> %894, i32 131072, i64 3, !dbg !17
  %895 = ptrtoint ptr addrspace(3) %885 to i32, !dbg !17
  %dma.buffer.lds654 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %895), !dbg !17
  %896 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds654, i32 %dma.buffer.offset652, <4 x i32> %dma.buffer.rsrc653) #10, !dbg !17
  %897 = shufflevector <8 x half> %848, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %897, <4 x float> %699) #9, !dbg !14
  %899 = shufflevector <8 x half> %850, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %899, <4 x float> %898) #9, !dbg !14
  %901 = shufflevector <8 x half> %852, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %901, <4 x float> %900) #9, !dbg !14
  %903 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %903, <4 x float> %902) #9, !dbg !14
  %905 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %905, <4 x float> %904) #9, !dbg !14
  %907 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %907, <4 x float> %906) #9, !dbg !14
  %909 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed78, <4 x half> %909, <4 x float> %908) #9, !dbg !14
  %911 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed80, <4 x half> %911, <4 x float> %910) #9, !dbg !14
  %913 = shufflevector <8 x half> %848, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %913, <4 x float> %698) #9, !dbg !14
  %915 = shufflevector <8 x half> %850, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %915, <4 x float> %914) #9, !dbg !14
  %917 = shufflevector <8 x half> %852, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %917, <4 x float> %916) #9, !dbg !14
  %919 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %919, <4 x float> %918) #9, !dbg !14
  %921 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %921, <4 x float> %920) #9, !dbg !14
  %923 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %923, <4 x float> %922) #9, !dbg !14
  %925 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed78, <4 x half> %925, <4 x float> %924) #9, !dbg !14
  %927 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed80, <4 x half> %927, <4 x float> %926) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local278, !dbg !14
  %929 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local360, i32 0) #0, !dbg !14
  %930 = bitcast <4 x i32> %929 to <8 x half>, !dbg !14
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local286, !dbg !14
  %931 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %932 = bitcast <4 x i32> %931 to <8 x half>, !dbg !14
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local294, !dbg !14
  %933 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !14
  %934 = bitcast <4 x i32> %933 to <8 x half>, !dbg !14
  %address.local363 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local302, !dbg !14
  %935 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !14
  %936 = bitcast <4 x i32> %935 to <8 x half>, !dbg !14
  %address.local364 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local310, !dbg !14
  %937 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local364, i32 0) #0, !dbg !14
  %938 = bitcast <4 x i32> %937 to <8 x half>, !dbg !14
  %address.local365 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local318, !dbg !14
  %939 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local365, i32 0) #0, !dbg !14
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !14
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local326, !dbg !14
  %941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !14
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !14
  %address.local367 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local334, !dbg !14
  %943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local367, i32 0) #0, !dbg !14
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %945 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %866, !dbg !17
  %946 = getelementptr i8, ptr addrspace(1) %945, i64 128, !dbg !17
  %947 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local341), !dbg !17
  %948 = zext i32 %947 to i64, !dbg !17
  %949 = inttoptr i64 %948 to ptr addrspace(3), !dbg !17
  %address.local368 = getelementptr [2 x i8], ptr addrspace(1) %946, i64 %address.local346, !dbg !17
  %950 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %951 = ptrtoint ptr addrspace(1) %address.local368 to i64, !dbg !17
  %952 = sub i64 %951, %950, !dbg !17
  %dma.buffer.offset649 = trunc i64 %952 to i32, !dbg !17
  %953 = trunc i64 %950 to i32, !dbg !17
  %954 = insertelement <4 x i32> poison, i32 %953, i64 0, !dbg !17
  %955 = lshr i64 %950, 32, !dbg !17
  %956 = trunc i64 %955 to i32, !dbg !17
  %957 = insertelement <4 x i32> %954, i32 %956, i64 1, !dbg !17
  %958 = insertelement <4 x i32> %957, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc650 = insertelement <4 x i32> %958, i32 131072, i64 3, !dbg !17
  %959 = ptrtoint ptr addrspace(3) %949 to i32, !dbg !17
  %dma.buffer.lds651 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %959), !dbg !17
  %960 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds651, i32 %dma.buffer.offset649, <4 x i32> %dma.buffer.rsrc650) #10, !dbg !17
  %961 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local353), !dbg !17
  %962 = zext i32 %961 to i64, !dbg !17
  %963 = inttoptr i64 %962 to ptr addrspace(3), !dbg !17
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %946, i64 %address.local358, !dbg !17
  %964 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %965 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !17
  %966 = sub i64 %965, %964, !dbg !17
  %dma.buffer.offset646 = trunc i64 %966 to i32, !dbg !17
  %967 = trunc i64 %964 to i32, !dbg !17
  %968 = insertelement <4 x i32> poison, i32 %967, i64 0, !dbg !17
  %969 = lshr i64 %964, 32, !dbg !17
  %970 = trunc i64 %969 to i32, !dbg !17
  %971 = insertelement <4 x i32> %968, i32 %970, i64 1, !dbg !17
  %972 = insertelement <4 x i32> %971, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc647 = insertelement <4 x i32> %972, i32 131072, i64 3, !dbg !17
  %973 = ptrtoint ptr addrspace(3) %963 to i32, !dbg !17
  %dma.buffer.lds648 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %973), !dbg !17
  %974 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds648, i32 %dma.buffer.offset646, <4 x i32> %dma.buffer.rsrc647) #10, !dbg !17
  %975 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %975, <4 x float> %697) #9, !dbg !14
  %977 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %977, <4 x float> %976) #9, !dbg !14
  %979 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %979, <4 x float> %978) #9, !dbg !14
  %981 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %981, <4 x float> %980) #9, !dbg !14
  %983 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %983, <4 x float> %982) #9, !dbg !14
  %985 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %985, <4 x float> %984) #9, !dbg !14
  %987 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed78, <4 x half> %987, <4 x float> %986) #9, !dbg !14
  %989 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed80, <4 x half> %989, <4 x float> %988) #9, !dbg !14
  %991 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %991, <4 x float> %696) #9, !dbg !14
  %993 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %993, <4 x float> %992) #9, !dbg !14
  %995 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %995, <4 x float> %994) #9, !dbg !14
  %997 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %997, <4 x float> %996) #9, !dbg !14
  %999 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %999, <4 x float> %998) #9, !dbg !14
  %1001 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %1001, <4 x float> %1000) #9, !dbg !14
  %1003 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed78, <4 x half> %1003, <4 x float> %1002) #9, !dbg !14
  %1005 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed80, <4 x half> %1005, <4 x float> %1004) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1007 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local360, i32 0) #0, !dbg !14
  %1008 = bitcast <4 x i32> %1007 to <8 x half>, !dbg !14
  %1009 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1010 = bitcast <4 x i32> %1009 to <8 x half>, !dbg !14
  %1011 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !14
  %1012 = bitcast <4 x i32> %1011 to <8 x half>, !dbg !14
  %1013 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !14
  %1014 = bitcast <4 x i32> %1013 to <8 x half>, !dbg !14
  %1015 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local364, i32 0) #0, !dbg !14
  %1016 = bitcast <4 x i32> %1015 to <8 x half>, !dbg !14
  %1017 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local365, i32 0) #0, !dbg !14
  %1018 = bitcast <4 x i32> %1017 to <8 x half>, !dbg !14
  %1019 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !14
  %1020 = bitcast <4 x i32> %1019 to <8 x half>, !dbg !14
  %1021 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local367, i32 0) #0, !dbg !14
  %1022 = bitcast <4 x i32> %1021 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1023 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %866, !dbg !17
  %1024 = getelementptr i8, ptr addrspace(1) %1023, i64 192, !dbg !17
  %1025 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local341), !dbg !17
  %1026 = zext i32 %1025 to i64, !dbg !17
  %1027 = inttoptr i64 %1026 to ptr addrspace(3), !dbg !17
  %address.local370 = getelementptr [2 x i8], ptr addrspace(1) %1024, i64 %address.local346, !dbg !17
  %1028 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1029 = ptrtoint ptr addrspace(1) %address.local370 to i64, !dbg !17
  %1030 = sub i64 %1029, %1028, !dbg !17
  %dma.buffer.offset643 = trunc i64 %1030 to i32, !dbg !17
  %1031 = trunc i64 %1028 to i32, !dbg !17
  %1032 = insertelement <4 x i32> poison, i32 %1031, i64 0, !dbg !17
  %1033 = lshr i64 %1028, 32, !dbg !17
  %1034 = trunc i64 %1033 to i32, !dbg !17
  %1035 = insertelement <4 x i32> %1032, i32 %1034, i64 1, !dbg !17
  %1036 = insertelement <4 x i32> %1035, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc644 = insertelement <4 x i32> %1036, i32 131072, i64 3, !dbg !17
  %1037 = ptrtoint ptr addrspace(3) %1027 to i32, !dbg !17
  %dma.buffer.lds645 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1037), !dbg !17
  %1038 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds645, i32 %dma.buffer.offset643, <4 x i32> %dma.buffer.rsrc644) #10, !dbg !17
  %1039 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local353), !dbg !17
  %1040 = zext i32 %1039 to i64, !dbg !17
  %1041 = inttoptr i64 %1040 to ptr addrspace(3), !dbg !17
  %address.local371 = getelementptr [2 x i8], ptr addrspace(1) %1024, i64 %address.local358, !dbg !17
  %1042 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1043 = ptrtoint ptr addrspace(1) %address.local371 to i64, !dbg !17
  %1044 = sub i64 %1043, %1042, !dbg !17
  %dma.buffer.offset = trunc i64 %1044 to i32, !dbg !17
  %1045 = trunc i64 %1042 to i32, !dbg !17
  %1046 = insertelement <4 x i32> poison, i32 %1045, i64 0, !dbg !17
  %1047 = lshr i64 %1042, 32, !dbg !17
  %1048 = trunc i64 %1047 to i32, !dbg !17
  %1049 = insertelement <4 x i32> %1046, i32 %1048, i64 1, !dbg !17
  %1050 = insertelement <4 x i32> %1049, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %1050, i32 131072, i64 3, !dbg !17
  %1051 = ptrtoint ptr addrspace(3) %1041 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1051), !dbg !17
  %1052 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %1053 = shufflevector <8 x half> %1008, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1053, <4 x float> %695) #9, !dbg !14
  %1055 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %1055, <4 x float> %1054) #9, !dbg !14
  %1057 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %1057, <4 x float> %1056) #9, !dbg !14
  %1059 = shufflevector <8 x half> %1014, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %1059, <4 x float> %1058) #9, !dbg !14
  %1061 = shufflevector <8 x half> %1016, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %1061, <4 x float> %1060) #9, !dbg !14
  %1063 = shufflevector <8 x half> %1018, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %1063, <4 x float> %1062) #9, !dbg !14
  %1065 = shufflevector <8 x half> %1020, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed78, <4 x half> %1065, <4 x float> %1064) #9, !dbg !14
  %1067 = shufflevector <8 x half> %1022, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed80, <4 x half> %1067, <4 x float> %1066) #9, !dbg !14
  %1069 = shufflevector <8 x half> %1008, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1069, <4 x float> %694) #9, !dbg !14
  %1071 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %1071, <4 x float> %1070) #9, !dbg !14
  %1073 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %1073, <4 x float> %1072) #9, !dbg !14
  %1075 = shufflevector <8 x half> %1014, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %1075, <4 x float> %1074) #9, !dbg !14
  %1077 = shufflevector <8 x half> %1016, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %1077, <4 x float> %1076) #9, !dbg !14
  %1079 = shufflevector <8 x half> %1018, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %1079, <4 x float> %1078) #9, !dbg !14
  %1081 = shufflevector <8 x half> %1020, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed78, <4 x half> %1081, <4 x float> %1080) #9, !dbg !14
  %1083 = shufflevector <8 x half> %1022, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed80, <4 x half> %1083, <4 x float> %1082) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1085 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local360, i32 0) #0, !dbg !14
  %1086 = bitcast <4 x i32> %1085 to <8 x half>, !dbg !14
  %1087 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1088 = bitcast <4 x i32> %1087 to <8 x half>, !dbg !14
  %1089 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !14
  %1090 = bitcast <4 x i32> %1089 to <8 x half>, !dbg !14
  %1091 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !14
  %1092 = bitcast <4 x i32> %1091 to <8 x half>, !dbg !14
  %1093 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local364, i32 0) #0, !dbg !14
  %1094 = bitcast <4 x i32> %1093 to <8 x half>, !dbg !14
  %1095 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local365, i32 0) #0, !dbg !14
  %1096 = bitcast <4 x i32> %1095 to <8 x half>, !dbg !14
  %1097 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !14
  %1098 = bitcast <4 x i32> %1097 to <8 x half>, !dbg !14
  %1099 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local367, i32 0) #0, !dbg !14
  %1100 = bitcast <4 x i32> %1099 to <8 x half>, !dbg !14
  %1101 = shufflevector <8 x half> %1086, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1101, <4 x float> %693) #9, !dbg !14
  %1103 = shufflevector <8 x half> %1088, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %1103, <4 x float> %1102) #9, !dbg !14
  %1105 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %1105, <4 x float> %1104) #9, !dbg !14
  %1107 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %1107, <4 x float> %1106) #9, !dbg !14
  %1109 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %1109, <4 x float> %1108) #9, !dbg !14
  %1111 = shufflevector <8 x half> %1096, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %1111, <4 x float> %1110) #9, !dbg !14
  %1113 = shufflevector <8 x half> %1098, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed78, <4 x half> %1113, <4 x float> %1112) #9, !dbg !14
  %1115 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed80, <4 x half> %1115, <4 x float> %1114) #9, !dbg !14
  %1117 = shufflevector <8 x half> %1086, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1117, <4 x float> %692) #9, !dbg !14
  %1119 = shufflevector <8 x half> %1088, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %1119, <4 x float> %1118) #9, !dbg !14
  %1121 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %1121, <4 x float> %1120) #9, !dbg !14
  %1123 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %1123, <4 x float> %1122) #9, !dbg !14
  %1125 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %1125, <4 x float> %1124) #9, !dbg !14
  %1127 = shufflevector <8 x half> %1096, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %1127, <4 x float> %1126) #9, !dbg !14
  %1129 = shufflevector <8 x half> %1098, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed78, <4 x half> %1129, <4 x float> %1128) #9, !dbg !14
  %1131 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed80, <4 x half> %1131, <4 x float> %1130) #9, !dbg !14
  %1133 = shufflevector <4 x float> %912, <4 x float> %928, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1134 = shufflevector <4 x float> %990, <4 x float> %1006, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1135 = shufflevector <4 x float> %1068, <4 x float> %1084, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1136 = shufflevector <4 x float> %1116, <4 x float> %1132, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1137 = add nuw nsw i64 %49, 1, !dbg !7
  br label %48, !dbg !7

1138:                                             ; preds = %48
  %1139 = and i32 %26, 16, !dbg !7
  %.not = icmp eq i32 %1139, 0, !dbg !7
  %1140 = and i32 %26, 32, !dbg !7
  %.not1 = icmp eq i32 %1140, 0, !dbg !7
  %1141 = extractelement <8 x float> %63, i64 0, !dbg !7
  %1142 = extractelement <8 x float> %63, i64 1, !dbg !7
  %1143 = extractelement <8 x float> %63, i64 2, !dbg !7
  %1144 = extractelement <8 x float> %63, i64 3, !dbg !7
  %1145 = select i1 %.not, float %1142, float %1141, !dbg !7
  %1146 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1147 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1146), !dbg !7
  %1148 = add nuw nsw i32 %1147, 64, !dbg !7
  %1149 = xor i32 %1147, 16, !dbg !7
  %1150 = and i32 %1148, 192, !dbg !7
  %1151 = icmp samesign ult i32 %1149, %1150, !dbg !7
  %1152 = select i1 %1151, i32 %1149, i32 %1147, !dbg !7
  %1153 = shl nuw nsw i32 %1152, 2, !dbg !7
  %1154 = bitcast float %1145 to i32, !dbg !7
  %1155 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1154), !dbg !7
  %1156 = bitcast i32 %1155 to float, !dbg !7
  %1157 = select i1 %.not, float %1141, float %1156, !dbg !7
  %1158 = select i1 %.not, float %1156, float %1142, !dbg !7
  %1159 = select i1 %.not, float %1144, float %1143, !dbg !7
  %1160 = bitcast float %1159 to i32, !dbg !7
  %1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1160), !dbg !7
  %1162 = bitcast i32 %1161 to float, !dbg !7
  %1163 = select i1 %.not, float %1143, float %1162, !dbg !7
  %1164 = select i1 %.not, float %1162, float %1144, !dbg !7
  %1165 = select i1 %.not1, float %1163, float %1157, !dbg !7
  %1166 = xor i32 %1147, 32, !dbg !7
  %1167 = icmp samesign ult i32 %1166, %1150, !dbg !7
  %1168 = select i1 %1167, i32 %1166, i32 %1147, !dbg !7
  %1169 = shl nuw nsw i32 %1168, 2, !dbg !7
  %1170 = bitcast float %1165 to i32, !dbg !7
  %1171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1170), !dbg !7
  %1172 = bitcast i32 %1171 to float, !dbg !7
  %1173 = select i1 %.not1, float %1157, float %1172, !dbg !7
  %1174 = select i1 %.not1, float %1172, float %1163, !dbg !7
  %1175 = select i1 %.not1, float %1164, float %1158, !dbg !7
  %1176 = bitcast float %1175 to i32, !dbg !7
  %1177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1176), !dbg !7
  %1178 = bitcast i32 %1177 to float, !dbg !7
  %1179 = select i1 %.not1, float %1158, float %1178, !dbg !7
  %1180 = select i1 %.not1, float %1178, float %1164, !dbg !7
  %1181 = extractelement <8 x float> %60, i64 0, !dbg !7
  %1182 = extractelement <8 x float> %60, i64 1, !dbg !7
  %1183 = extractelement <8 x float> %60, i64 2, !dbg !7
  %1184 = extractelement <8 x float> %60, i64 3, !dbg !7
  %1185 = select i1 %.not, float %1182, float %1181, !dbg !7
  %1186 = bitcast float %1185 to i32, !dbg !7
  %1187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1186), !dbg !7
  %1188 = bitcast i32 %1187 to float, !dbg !7
  %1189 = select i1 %.not, float %1181, float %1188, !dbg !7
  %1190 = select i1 %.not, float %1188, float %1182, !dbg !7
  %1191 = select i1 %.not, float %1184, float %1183, !dbg !7
  %1192 = bitcast float %1191 to i32, !dbg !7
  %1193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1192), !dbg !7
  %1194 = bitcast i32 %1193 to float, !dbg !7
  %1195 = select i1 %.not, float %1183, float %1194, !dbg !7
  %1196 = select i1 %.not, float %1194, float %1184, !dbg !7
  %1197 = select i1 %.not1, float %1195, float %1189, !dbg !7
  %1198 = bitcast float %1197 to i32, !dbg !7
  %1199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1198), !dbg !7
  %1200 = bitcast i32 %1199 to float, !dbg !7
  %1201 = select i1 %.not1, float %1189, float %1200, !dbg !7
  %1202 = select i1 %.not1, float %1200, float %1195, !dbg !7
  %1203 = select i1 %.not1, float %1196, float %1190, !dbg !7
  %1204 = bitcast float %1203 to i32, !dbg !7
  %1205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1204), !dbg !7
  %1206 = bitcast i32 %1205 to float, !dbg !7
  %1207 = select i1 %.not1, float %1190, float %1206, !dbg !7
  %1208 = select i1 %.not1, float %1206, float %1196, !dbg !7
  %1209 = extractelement <8 x float> %57, i64 0, !dbg !7
  %1210 = extractelement <8 x float> %57, i64 1, !dbg !7
  %1211 = extractelement <8 x float> %57, i64 2, !dbg !7
  %1212 = extractelement <8 x float> %57, i64 3, !dbg !7
  %1213 = select i1 %.not, float %1210, float %1209, !dbg !7
  %1214 = bitcast float %1213 to i32, !dbg !7
  %1215 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1214), !dbg !7
  %1216 = bitcast i32 %1215 to float, !dbg !7
  %1217 = select i1 %.not, float %1209, float %1216, !dbg !7
  %1218 = select i1 %.not, float %1216, float %1210, !dbg !7
  %1219 = select i1 %.not, float %1212, float %1211, !dbg !7
  %1220 = bitcast float %1219 to i32, !dbg !7
  %1221 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1220), !dbg !7
  %1222 = bitcast i32 %1221 to float, !dbg !7
  %1223 = select i1 %.not, float %1211, float %1222, !dbg !7
  %1224 = select i1 %.not, float %1222, float %1212, !dbg !7
  %1225 = select i1 %.not1, float %1223, float %1217, !dbg !7
  %1226 = bitcast float %1225 to i32, !dbg !7
  %1227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1226), !dbg !7
  %1228 = bitcast i32 %1227 to float, !dbg !7
  %1229 = select i1 %.not1, float %1217, float %1228, !dbg !7
  %1230 = select i1 %.not1, float %1228, float %1223, !dbg !7
  %1231 = select i1 %.not1, float %1224, float %1218, !dbg !7
  %1232 = bitcast float %1231 to i32, !dbg !7
  %1233 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1232), !dbg !7
  %1234 = bitcast i32 %1233 to float, !dbg !7
  %1235 = select i1 %.not1, float %1218, float %1234, !dbg !7
  %1236 = select i1 %.not1, float %1234, float %1224, !dbg !7
  %1237 = extractelement <8 x float> %52, i64 0, !dbg !7
  %1238 = extractelement <8 x float> %52, i64 1, !dbg !7
  %1239 = extractelement <8 x float> %52, i64 2, !dbg !7
  %1240 = extractelement <8 x float> %52, i64 3, !dbg !7
  %1241 = select i1 %.not, float %1238, float %1237, !dbg !7
  %1242 = bitcast float %1241 to i32, !dbg !7
  %1243 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1242), !dbg !7
  %1244 = bitcast i32 %1243 to float, !dbg !7
  %1245 = select i1 %.not, float %1237, float %1244, !dbg !7
  %1246 = select i1 %.not, float %1244, float %1238, !dbg !7
  %1247 = select i1 %.not, float %1240, float %1239, !dbg !7
  %1248 = bitcast float %1247 to i32, !dbg !7
  %1249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1248), !dbg !7
  %1250 = bitcast i32 %1249 to float, !dbg !7
  %1251 = select i1 %.not, float %1239, float %1250, !dbg !7
  %1252 = select i1 %.not, float %1250, float %1240, !dbg !7
  %1253 = select i1 %.not1, float %1251, float %1245, !dbg !7
  %1254 = bitcast float %1253 to i32, !dbg !7
  %1255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1254), !dbg !7
  %1256 = select i1 %.not1, float %1252, float %1246, !dbg !7
  %1257 = bitcast float %1256 to i32, !dbg !7
  %1258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1257), !dbg !7
  %1259 = shufflevector <1 x float> %53, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1260 = extractelement <8 x float> %63, i64 4, !dbg !7
  %1261 = extractelement <8 x float> %63, i64 5, !dbg !7
  %1262 = extractelement <8 x float> %63, i64 6, !dbg !7
  %1263 = extractelement <8 x float> %63, i64 7, !dbg !7
  %1264 = select i1 %.not, float %1261, float %1260, !dbg !7
  %1265 = bitcast float %1264 to i32, !dbg !7
  %1266 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1265), !dbg !7
  %1267 = bitcast i32 %1266 to float, !dbg !7
  %1268 = select i1 %.not, float %1260, float %1267, !dbg !7
  %1269 = select i1 %.not, float %1267, float %1261, !dbg !7
  %1270 = select i1 %.not, float %1263, float %1262, !dbg !7
  %1271 = bitcast float %1270 to i32, !dbg !7
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1271), !dbg !7
  %1273 = bitcast i32 %1272 to float, !dbg !7
  %1274 = select i1 %.not, float %1262, float %1273, !dbg !7
  %1275 = select i1 %.not, float %1273, float %1263, !dbg !7
  %1276 = select i1 %.not1, float %1274, float %1268, !dbg !7
  %1277 = bitcast float %1276 to i32, !dbg !7
  %1278 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1277), !dbg !7
  %1279 = bitcast i32 %1278 to float, !dbg !7
  %1280 = select i1 %.not1, float %1268, float %1279, !dbg !7
  %1281 = select i1 %.not1, float %1279, float %1274, !dbg !7
  %1282 = select i1 %.not1, float %1275, float %1269, !dbg !7
  %1283 = bitcast float %1282 to i32, !dbg !7
  %1284 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1283), !dbg !7
  %1285 = bitcast i32 %1284 to float, !dbg !7
  %1286 = select i1 %.not1, float %1269, float %1285, !dbg !7
  %1287 = select i1 %.not1, float %1285, float %1275, !dbg !7
  %1288 = extractelement <8 x float> %60, i64 4, !dbg !7
  %1289 = extractelement <8 x float> %60, i64 5, !dbg !7
  %1290 = extractelement <8 x float> %60, i64 6, !dbg !7
  %1291 = extractelement <8 x float> %60, i64 7, !dbg !7
  %1292 = select i1 %.not, float %1289, float %1288, !dbg !7
  %1293 = bitcast float %1292 to i32, !dbg !7
  %1294 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1293), !dbg !7
  %1295 = bitcast i32 %1294 to float, !dbg !7
  %1296 = select i1 %.not, float %1288, float %1295, !dbg !7
  %1297 = select i1 %.not, float %1295, float %1289, !dbg !7
  %1298 = select i1 %.not, float %1291, float %1290, !dbg !7
  %1299 = bitcast float %1298 to i32, !dbg !7
  %1300 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1299), !dbg !7
  %1301 = bitcast i32 %1300 to float, !dbg !7
  %1302 = select i1 %.not, float %1290, float %1301, !dbg !7
  %1303 = select i1 %.not, float %1301, float %1291, !dbg !7
  %1304 = select i1 %.not1, float %1302, float %1296, !dbg !7
  %1305 = bitcast float %1304 to i32, !dbg !7
  %1306 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1305), !dbg !7
  %1307 = bitcast i32 %1306 to float, !dbg !7
  %1308 = select i1 %.not1, float %1296, float %1307, !dbg !7
  %1309 = select i1 %.not1, float %1307, float %1302, !dbg !7
  %1310 = select i1 %.not1, float %1303, float %1297, !dbg !7
  %1311 = bitcast float %1310 to i32, !dbg !7
  %1312 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1311), !dbg !7
  %1313 = bitcast i32 %1312 to float, !dbg !7
  %1314 = select i1 %.not1, float %1297, float %1313, !dbg !7
  %1315 = select i1 %.not1, float %1313, float %1303, !dbg !7
  %1316 = extractelement <8 x float> %57, i64 4, !dbg !7
  %1317 = extractelement <8 x float> %57, i64 5, !dbg !7
  %1318 = extractelement <8 x float> %57, i64 6, !dbg !7
  %1319 = extractelement <8 x float> %57, i64 7, !dbg !7
  %1320 = select i1 %.not, float %1317, float %1316, !dbg !7
  %1321 = bitcast float %1320 to i32, !dbg !7
  %1322 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1321), !dbg !7
  %1323 = bitcast i32 %1322 to float, !dbg !7
  %1324 = select i1 %.not, float %1316, float %1323, !dbg !7
  %1325 = select i1 %.not, float %1323, float %1317, !dbg !7
  %1326 = select i1 %.not, float %1319, float %1318, !dbg !7
  %1327 = bitcast float %1326 to i32, !dbg !7
  %1328 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1327), !dbg !7
  %1329 = bitcast i32 %1328 to float, !dbg !7
  %1330 = select i1 %.not, float %1318, float %1329, !dbg !7
  %1331 = select i1 %.not, float %1329, float %1319, !dbg !7
  %1332 = select i1 %.not1, float %1330, float %1324, !dbg !7
  %1333 = bitcast float %1332 to i32, !dbg !7
  %1334 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1333), !dbg !7
  %1335 = bitcast i32 %1334 to float, !dbg !7
  %1336 = select i1 %.not1, float %1324, float %1335, !dbg !7
  %1337 = select i1 %.not1, float %1335, float %1330, !dbg !7
  %1338 = select i1 %.not1, float %1331, float %1325, !dbg !7
  %1339 = bitcast float %1338 to i32, !dbg !7
  %1340 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1339), !dbg !7
  %1341 = bitcast i32 %1340 to float, !dbg !7
  %1342 = select i1 %.not1, float %1325, float %1341, !dbg !7
  %1343 = select i1 %.not1, float %1341, float %1331, !dbg !7
  %1344 = extractelement <8 x float> %52, i64 4, !dbg !7
  %1345 = extractelement <8 x float> %52, i64 5, !dbg !7
  %1346 = extractelement <8 x float> %52, i64 6, !dbg !7
  %1347 = extractelement <8 x float> %52, i64 7, !dbg !7
  %1348 = select i1 %.not, float %1345, float %1344, !dbg !7
  %1349 = bitcast float %1348 to i32, !dbg !7
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1349), !dbg !7
  %1351 = bitcast i32 %1350 to float, !dbg !7
  %1352 = select i1 %.not, float %1344, float %1351, !dbg !7
  %1353 = select i1 %.not, float %1351, float %1345, !dbg !7
  %1354 = select i1 %.not, float %1347, float %1346, !dbg !7
  %1355 = bitcast float %1354 to i32, !dbg !7
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1355), !dbg !7
  %1357 = bitcast i32 %1356 to float, !dbg !7
  %1358 = select i1 %.not, float %1346, float %1357, !dbg !7
  %1359 = select i1 %.not, float %1357, float %1347, !dbg !7
  %1360 = select i1 %.not1, float %1358, float %1352, !dbg !7
  %1361 = bitcast float %1360 to i32, !dbg !7
  %1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1361), !dbg !7
  %1363 = select i1 %.not1, float %1359, float %1353, !dbg !7
  %1364 = bitcast float %1363 to i32, !dbg !7
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1364), !dbg !7
  br i1 true, label %1366, label %1400, !dbg !20

1366:                                             ; preds = %1138
  %1367 = bitcast i32 %1362 to float, !dbg !7
  %1368 = select i1 %.not1, float %1352, float %1367, !dbg !7
  %1369 = insertelement <4 x float> poison, float %1368, i64 0, !dbg !18
  %1370 = bitcast i32 %1365 to float, !dbg !7
  %1371 = select i1 %.not1, float %1353, float %1370, !dbg !7
  %1372 = insertelement <4 x float> %1369, float %1371, i64 1, !dbg !18
  %1373 = select i1 %.not1, float %1367, float %1358, !dbg !7
  %1374 = insertelement <4 x float> %1372, float %1373, i64 2, !dbg !18
  %1375 = select i1 %.not1, float %1370, float %1359, !dbg !7
  %1376 = insertelement <4 x float> %1374, float %1375, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1259, !dbg !18
  %normalized = fmul <4 x float> %1376, %row.reciprocal, !dbg !18
  %1377 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %1378 = extractelement <4 x half> %1377, i64 3, !dbg !19
  %1379 = extractelement <4 x half> %1377, i64 2, !dbg !19
  %1380 = extractelement <4 x half> %1377, i64 1, !dbg !19
  %1381 = extractelement <4 x half> %1377, i64 0, !dbg !19
  %1382 = bitcast i32 %1255 to float, !dbg !7
  %1383 = select i1 %.not1, float %1245, float %1382, !dbg !7
  %1384 = insertelement <4 x float> poison, float %1383, i64 0, !dbg !18
  %1385 = bitcast i32 %1258 to float, !dbg !7
  %1386 = select i1 %.not1, float %1246, float %1385, !dbg !7
  %1387 = insertelement <4 x float> %1384, float %1386, i64 1, !dbg !18
  %1388 = select i1 %.not1, float %1382, float %1251, !dbg !7
  %1389 = insertelement <4 x float> %1387, float %1388, i64 2, !dbg !18
  %1390 = select i1 %.not1, float %1385, float %1252, !dbg !7
  %1391 = insertelement <4 x float> %1389, float %1390, i64 3, !dbg !18
  %normalized37 = fmul <4 x float> %1391, %row.reciprocal, !dbg !18
  %1392 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !19
  %1393 = extractelement <4 x half> %1392, i64 3, !dbg !19
  %1394 = extractelement <4 x half> %1392, i64 2, !dbg !19
  %1395 = extractelement <4 x half> %1392, i64 1, !dbg !19
  %1396 = extractelement <4 x half> %1392, i64 0, !dbg !19
  %1397 = add nsw i64 %17, %8, !dbg !20
  %1398 = shl nsw i64 %1397, 19, !dbg !20
  %1399 = and i64 %1398, 16252928, !dbg !20
  %address.thread372 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local373 = zext nneg i32 %address.thread372 to i64, !dbg !12
  %address.local374 = add nsw i64 %20, %address.local373, !dbg !20
  %address.local375 = lshr i64 %address.local373, 6, !dbg !12
  %address.local376 = shl nuw nsw i64 %address.local375, 4, !dbg !20
  %address.local377 = add nsw i64 %address.local374, %address.local376, !dbg !20
  %address.local378 = and i64 %address.local373, 240, !dbg !20
  %address.local379 = sub nsw i64 %address.local377, %address.local378, !dbg !20
  %address.local380 = add nsw i64 %address.local379, %21, !dbg !20
  %address.local381 = shl nuw nsw i64 %address.local380, 7, !dbg !20
  %address.local382 = add nuw nsw i64 %1399, %address.local381, !dbg !20
  %address.local383 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local382, !dbg !20
  %address.local384 = lshr i64 %address.local373, 2, !dbg !12
  %address.local385 = and i64 %address.local384, 12, !dbg !12
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local383, i64 %address.local385, !dbg !20
  store half %1396, ptr addrspace(1) %address.local386, align 2, !dbg !20
  %address.local387 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local382, !dbg !20
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local387, i64 %address.local385, !dbg !20
  %address.local389 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local388, i64 2, !dbg !20
  store half %1395, ptr addrspace(1) %address.local389, align 2, !dbg !20
  %address.local390 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local382, !dbg !20
  %address.local391 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local390, i64 %address.local385, !dbg !20
  %address.local392 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local391, i64 4, !dbg !20
  store half %1394, ptr addrspace(1) %address.local392, align 2, !dbg !20
  %address.local393 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local382, !dbg !20
  %address.local394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local393, i64 %address.local385, !dbg !20
  %address.local395 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local394, i64 6, !dbg !20
  store half %1393, ptr addrspace(1) %address.local395, align 2, !dbg !20
  %address.local396 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local382, !dbg !20
  %address.local397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local396, i64 %address.local385, !dbg !20
  %address.local398 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local397, i64 32, !dbg !20
  store half %1381, ptr addrspace(1) %address.local398, align 2, !dbg !20
  %address.local399 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local382, !dbg !20
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local399, i64 %address.local385, !dbg !20
  %address.local401 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local400, i64 34, !dbg !20
  store half %1380, ptr addrspace(1) %address.local401, align 2, !dbg !20
  %address.local402 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local382, !dbg !20
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local402, i64 %address.local385, !dbg !20
  %address.local404 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local403, i64 36, !dbg !20
  store half %1379, ptr addrspace(1) %address.local404, align 2, !dbg !20
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local382, !dbg !20
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local385, !dbg !20
  %address.local407 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local406, i64 38, !dbg !20
  store half %1378, ptr addrspace(1) %address.local407, align 2, !dbg !20
  br label %1400, !dbg !20

1400:                                             ; preds = %1366, %1138
  br i1 true, label %1401, label %1421, !dbg !20

1401:                                             ; preds = %1400
  %1402 = insertelement <4 x float> poison, float %1336, i64 0, !dbg !18
  %1403 = insertelement <4 x float> %1402, float %1342, i64 1, !dbg !18
  %1404 = insertelement <4 x float> %1403, float %1337, i64 2, !dbg !18
  %1405 = insertelement <4 x float> %1404, float %1343, i64 3, !dbg !18
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1259, !dbg !18
  %normalized39 = fmul <4 x float> %1405, %row.reciprocal38, !dbg !18
  %1406 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !19
  %1407 = extractelement <4 x half> %1406, i64 3, !dbg !19
  %1408 = extractelement <4 x half> %1406, i64 2, !dbg !19
  %1409 = extractelement <4 x half> %1406, i64 1, !dbg !19
  %1410 = extractelement <4 x half> %1406, i64 0, !dbg !19
  %1411 = insertelement <4 x float> poison, float %1229, i64 0, !dbg !18
  %1412 = insertelement <4 x float> %1411, float %1235, i64 1, !dbg !18
  %1413 = insertelement <4 x float> %1412, float %1230, i64 2, !dbg !18
  %1414 = insertelement <4 x float> %1413, float %1236, i64 3, !dbg !18
  %normalized40 = fmul <4 x float> %1414, %row.reciprocal38, !dbg !18
  %1415 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !19
  %1416 = extractelement <4 x half> %1415, i64 3, !dbg !19
  %1417 = extractelement <4 x half> %1415, i64 2, !dbg !19
  %1418 = extractelement <4 x half> %1415, i64 1, !dbg !19
  %1419 = extractelement <4 x half> %1415, i64 0, !dbg !19
  %1420 = shl nuw nsw i64 %19, 19, !dbg !20
  %address.thread408 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local409 = zext nneg i32 %address.thread408 to i64, !dbg !12
  %address.local410 = add nsw i64 %20, %address.local409, !dbg !20
  %address.local411 = lshr i64 %address.local409, 6, !dbg !12
  %address.local412 = shl nuw nsw i64 %address.local411, 4, !dbg !20
  %address.local413 = add nsw i64 %address.local410, %address.local412, !dbg !20
  %address.local414 = and i64 %address.local409, 240, !dbg !20
  %address.local415 = sub nsw i64 %address.local413, %address.local414, !dbg !20
  %address.local416 = add nsw i64 %address.local415, %21, !dbg !20
  %address.local417 = shl nuw nsw i64 %address.local416, 7, !dbg !20
  %address.local418 = add nuw nsw i64 %1420, %address.local417, !dbg !20
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local420 = lshr i64 %address.local409, 2, !dbg !12
  %address.local421 = and i64 %address.local420, 12, !dbg !12
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local419, i64 %address.local421, !dbg !20
  %address.local423 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local422, i64 64, !dbg !20
  store half %1419, ptr addrspace(1) %address.local423, align 2, !dbg !20
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local424, i64 %address.local421, !dbg !20
  %address.local426 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local425, i64 66, !dbg !20
  store half %1418, ptr addrspace(1) %address.local426, align 2, !dbg !20
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local427, i64 %address.local421, !dbg !20
  %address.local429 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local428, i64 68, !dbg !20
  store half %1417, ptr addrspace(1) %address.local429, align 2, !dbg !20
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local430, i64 %address.local421, !dbg !20
  %address.local432 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local431, i64 70, !dbg !20
  store half %1416, ptr addrspace(1) %address.local432, align 2, !dbg !20
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local421, !dbg !20
  %address.local435 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local434, i64 96, !dbg !20
  store half %1410, ptr addrspace(1) %address.local435, align 2, !dbg !20
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local436, i64 %address.local421, !dbg !20
  %address.local438 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local437, i64 98, !dbg !20
  store half %1409, ptr addrspace(1) %address.local438, align 2, !dbg !20
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local421, !dbg !20
  %address.local441 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local440, i64 100, !dbg !20
  store half %1408, ptr addrspace(1) %address.local441, align 2, !dbg !20
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local442, i64 %address.local421, !dbg !20
  %address.local444 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local443, i64 102, !dbg !20
  store half %1407, ptr addrspace(1) %address.local444, align 2, !dbg !20
  br label %1421, !dbg !20

1421:                                             ; preds = %1401, %1400
  br i1 true, label %1422, label %1442, !dbg !20

1422:                                             ; preds = %1421
  %1423 = insertelement <4 x float> poison, float %1308, i64 0, !dbg !18
  %1424 = insertelement <4 x float> %1423, float %1314, i64 1, !dbg !18
  %1425 = insertelement <4 x float> %1424, float %1309, i64 2, !dbg !18
  %1426 = insertelement <4 x float> %1425, float %1315, i64 3, !dbg !18
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1259, !dbg !18
  %normalized42 = fmul <4 x float> %1426, %row.reciprocal41, !dbg !18
  %1427 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !19
  %1428 = extractelement <4 x half> %1427, i64 3, !dbg !19
  %1429 = extractelement <4 x half> %1427, i64 2, !dbg !19
  %1430 = extractelement <4 x half> %1427, i64 1, !dbg !19
  %1431 = extractelement <4 x half> %1427, i64 0, !dbg !19
  %1432 = insertelement <4 x float> poison, float %1201, i64 0, !dbg !18
  %1433 = insertelement <4 x float> %1432, float %1207, i64 1, !dbg !18
  %1434 = insertelement <4 x float> %1433, float %1202, i64 2, !dbg !18
  %1435 = insertelement <4 x float> %1434, float %1208, i64 3, !dbg !18
  %normalized43 = fmul <4 x float> %1435, %row.reciprocal41, !dbg !18
  %1436 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !19
  %1437 = extractelement <4 x half> %1436, i64 3, !dbg !19
  %1438 = extractelement <4 x half> %1436, i64 2, !dbg !19
  %1439 = extractelement <4 x half> %1436, i64 1, !dbg !19
  %1440 = extractelement <4 x half> %1436, i64 0, !dbg !19
  %1441 = shl nuw nsw i64 %19, 19, !dbg !20
  %address.thread445 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local446 = zext nneg i32 %address.thread445 to i64, !dbg !12
  %address.local447 = add nsw i64 %20, %address.local446, !dbg !20
  %address.local448 = lshr i64 %address.local446, 6, !dbg !12
  %address.local449 = shl nuw nsw i64 %address.local448, 4, !dbg !20
  %address.local450 = add nsw i64 %address.local447, %address.local449, !dbg !20
  %address.local451 = and i64 %address.local446, 240, !dbg !20
  %address.local452 = sub nsw i64 %address.local450, %address.local451, !dbg !20
  %address.local453 = add nsw i64 %address.local452, %21, !dbg !20
  %address.local454 = shl nuw nsw i64 %address.local453, 7, !dbg !20
  %address.local455 = add nuw nsw i64 %1441, %address.local454, !dbg !20
  %address.local456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !20
  %address.local457 = lshr i64 %address.local446, 2, !dbg !12
  %address.local458 = and i64 %address.local457, 12, !dbg !12
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local456, i64 %address.local458, !dbg !20
  %address.local460 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local459, i64 128, !dbg !20
  store half %1440, ptr addrspace(1) %address.local460, align 2, !dbg !20
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !20
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local461, i64 %address.local458, !dbg !20
  %address.local463 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local462, i64 130, !dbg !20
  store half %1439, ptr addrspace(1) %address.local463, align 2, !dbg !20
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !20
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local464, i64 %address.local458, !dbg !20
  %address.local466 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local465, i64 132, !dbg !20
  store half %1438, ptr addrspace(1) %address.local466, align 2, !dbg !20
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !20
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local467, i64 %address.local458, !dbg !20
  %address.local469 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local468, i64 134, !dbg !20
  store half %1437, ptr addrspace(1) %address.local469, align 2, !dbg !20
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !20
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local470, i64 %address.local458, !dbg !20
  %address.local472 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local471, i64 160, !dbg !20
  store half %1431, ptr addrspace(1) %address.local472, align 2, !dbg !20
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !20
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local473, i64 %address.local458, !dbg !20
  %address.local475 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local474, i64 162, !dbg !20
  store half %1430, ptr addrspace(1) %address.local475, align 2, !dbg !20
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !20
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local476, i64 %address.local458, !dbg !20
  %address.local478 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local477, i64 164, !dbg !20
  store half %1429, ptr addrspace(1) %address.local478, align 2, !dbg !20
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local455, !dbg !20
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local479, i64 %address.local458, !dbg !20
  %address.local481 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local480, i64 166, !dbg !20
  store half %1428, ptr addrspace(1) %address.local481, align 2, !dbg !20
  br label %1442, !dbg !20

1442:                                             ; preds = %1422, %1421
  br i1 true, label %1443, label %1463, !dbg !20

1443:                                             ; preds = %1442
  %1444 = insertelement <4 x float> poison, float %1280, i64 0, !dbg !18
  %1445 = insertelement <4 x float> %1444, float %1286, i64 1, !dbg !18
  %1446 = insertelement <4 x float> %1445, float %1281, i64 2, !dbg !18
  %1447 = insertelement <4 x float> %1446, float %1287, i64 3, !dbg !18
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1259, !dbg !18
  %normalized45 = fmul <4 x float> %1447, %row.reciprocal44, !dbg !18
  %1448 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !19
  %1449 = extractelement <4 x half> %1448, i64 3, !dbg !19
  %1450 = extractelement <4 x half> %1448, i64 2, !dbg !19
  %1451 = extractelement <4 x half> %1448, i64 1, !dbg !19
  %1452 = extractelement <4 x half> %1448, i64 0, !dbg !19
  %1453 = insertelement <4 x float> poison, float %1173, i64 0, !dbg !18
  %1454 = insertelement <4 x float> %1453, float %1179, i64 1, !dbg !18
  %1455 = insertelement <4 x float> %1454, float %1174, i64 2, !dbg !18
  %1456 = insertelement <4 x float> %1455, float %1180, i64 3, !dbg !18
  %normalized46 = fmul <4 x float> %1456, %row.reciprocal44, !dbg !18
  %1457 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !19
  %1458 = extractelement <4 x half> %1457, i64 3, !dbg !19
  %1459 = extractelement <4 x half> %1457, i64 2, !dbg !19
  %1460 = extractelement <4 x half> %1457, i64 1, !dbg !19
  %1461 = extractelement <4 x half> %1457, i64 0, !dbg !19
  %1462 = shl nuw nsw i64 %19, 19, !dbg !20
  %address.thread482 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local483 = zext nneg i32 %address.thread482 to i64, !dbg !12
  %address.local484 = add nsw i64 %20, %address.local483, !dbg !20
  %address.local485 = lshr i64 %address.local483, 6, !dbg !12
  %address.local486 = shl nuw nsw i64 %address.local485, 4, !dbg !20
  %address.local487 = add nsw i64 %address.local484, %address.local486, !dbg !20
  %address.local488 = and i64 %address.local483, 240, !dbg !20
  %address.local489 = sub nsw i64 %address.local487, %address.local488, !dbg !20
  %address.local490 = add nsw i64 %address.local489, %21, !dbg !20
  %address.local491 = shl nuw nsw i64 %address.local490, 7, !dbg !20
  %address.local492 = add nuw nsw i64 %1462, %address.local491, !dbg !20
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local492, !dbg !20
  %address.local494 = lshr i64 %address.local483, 2, !dbg !12
  %address.local495 = and i64 %address.local494, 12, !dbg !12
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local493, i64 %address.local495, !dbg !20
  %address.local497 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local496, i64 192, !dbg !20
  store half %1461, ptr addrspace(1) %address.local497, align 2, !dbg !20
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local492, !dbg !20
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local498, i64 %address.local495, !dbg !20
  %address.local500 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local499, i64 194, !dbg !20
  store half %1460, ptr addrspace(1) %address.local500, align 2, !dbg !20
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local492, !dbg !20
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local501, i64 %address.local495, !dbg !20
  %address.local503 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local502, i64 196, !dbg !20
  store half %1459, ptr addrspace(1) %address.local503, align 2, !dbg !20
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local492, !dbg !20
  %address.local505 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local504, i64 %address.local495, !dbg !20
  %address.local506 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local505, i64 198, !dbg !20
  store half %1458, ptr addrspace(1) %address.local506, align 2, !dbg !20
  %address.local507 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local492, !dbg !20
  %address.local508 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local507, i64 %address.local495, !dbg !20
  %address.local509 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local508, i64 224, !dbg !20
  store half %1452, ptr addrspace(1) %address.local509, align 2, !dbg !20
  %address.local510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local492, !dbg !20
  %address.local511 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local510, i64 %address.local495, !dbg !20
  %address.local512 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local511, i64 226, !dbg !20
  store half %1451, ptr addrspace(1) %address.local512, align 2, !dbg !20
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local492, !dbg !20
  %address.local514 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local513, i64 %address.local495, !dbg !20
  %address.local515 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local514, i64 228, !dbg !20
  store half %1450, ptr addrspace(1) %address.local515, align 2, !dbg !20
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local492, !dbg !20
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local495, !dbg !20
  %address.local518 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local517, i64 230, !dbg !20
  store half %1449, ptr addrspace(1) %address.local518, align 2, !dbg !20
  br label %1463, !dbg !20

1463:                                             ; preds = %1443, %1442
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/baseline")
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
