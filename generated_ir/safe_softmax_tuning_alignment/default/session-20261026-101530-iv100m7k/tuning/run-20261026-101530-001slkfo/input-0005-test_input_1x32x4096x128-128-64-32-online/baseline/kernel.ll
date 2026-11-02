; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/baseline/input.mlir"

@shm_0 = addrspace(3) global [12288 x i8] undef, align 16

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
  %24 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !38
  %25 = zext nneg i32 %24 to i64, !dbg !12
  %26 = lshr i64 %25, 6, !dbg !12
  %27 = or disjoint i64 %11, 127, !dbg !13
  %28 = add nsw i64 %11, 128, !dbg !13
  %29 = icmp uge i64 %28, %27, !dbg !13
  %30 = shl nsw i64 %10, 1, !dbg !7
  %31 = add nsw i64 %30, 2, !dbg !7
  %32 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %34 = getelementptr i8, ptr addrspace(1) %33, i64 64, !dbg !11
  %35 = shl nsw i64 %8, 7, !dbg !15
  %36 = shl nsw i64 %6, 12, !dbg !15
  %37 = add nsw i64 %35, %36, !dbg !15
  %38 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %39 = getelementptr i8, ptr addrspace(1) %38, i64 128, !dbg !11
  %40 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %41 = getelementptr i8, ptr addrspace(1) %40, i64 192, !dbg !11
  br label %42, !dbg !7

42:                                               ; preds = %568, %4
  %43 = phi i64 [ %1105, %568 ], [ 0, %4 ], !dbg !7
  %44 = phi [2 x <8 x float>] [ %1090, %568 ], [ zeroinitializer, %4 ], !dbg !7
  %45 = phi [2 x <1 x float>] [ %866, %568 ], [ zeroinitializer, %4 ], !dbg !7
  %46 = phi [2 x <1 x float>] [ %1092, %568 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %47 = phi [2 x <8 x float>] [ %1096, %568 ], [ zeroinitializer, %4 ], !dbg !7
  %48 = phi [2 x <8 x float>] [ %1100, %568 ], [ zeroinitializer, %4 ], !dbg !7
  %49 = phi [2 x <8 x float>] [ %1104, %568 ], [ zeroinitializer, %4 ], !dbg !7
  %50 = icmp slt i64 %43, %31, !dbg !7
  br i1 %50, label %51, label %1106, !dbg !7

51:                                               ; preds = %42
  %52 = shl nsw i64 %43, 6, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !11, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local76 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local77 = and i64 %address.local76, 1536, !dbg !11
  %address.local78 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local77, !dbg !11
  %address.local79 = ptrtoint ptr addrspace(3) %address.local78 to i64, !dbg !11
  %address.local80 = trunc i64 %address.local79 to i32, !dbg !11
  %53 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local80), !dbg !11
  %54 = zext i32 %53 to i64, !dbg !11
  %55 = inttoptr i64 %54 to ptr addrspace(3), !dbg !11
  %address.local81 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local82 = and i64 %address.local81, 16256, !dbg !11
  %address.local83 = lshr i64 %address.local, 4, !dbg !11
  %address.local84 = and i64 %address.local83, 8, !dbg !11
  %address.local85 = or disjoint i64 %address.local82, %address.local84, !dbg !11
  %address.local86 = getelementptr [2 x i8], ptr addrspace(1) %32, i64 %address.local85, !dbg !11
  %56 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %57 = ptrtoint ptr addrspace(1) %address.local86 to i64, !dbg !11
  %58 = sub i64 %57, %56, !dbg !11
  %dma.buffer.offset634 = trunc i64 %58 to i32, !dbg !11
  %59 = trunc i64 %56 to i32, !dbg !11
  %60 = insertelement <4 x i32> poison, i32 %59, i64 0, !dbg !11
  %61 = lshr i64 %56, 32, !dbg !11
  %62 = trunc i64 %61 to i32, !dbg !11
  %63 = insertelement <4 x i32> %60, i32 %62, i64 1, !dbg !11
  %64 = insertelement <4 x i32> %63, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc635 = insertelement <4 x i32> %64, i32 131072, i64 3, !dbg !11
  %65 = ptrtoint ptr addrspace(3) %55 to i32, !dbg !11
  %dma.buffer.lds636 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %65), !dbg !11
  %66 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds636, i32 %dma.buffer.offset634, <4 x i32> %dma.buffer.rsrc635) #10, !dbg !11
  %address.local87 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local88 = shl nuw nsw i64 %address.local87, 3, !dbg !11
  %address.local89 = and i64 %address.local88, 3584, !dbg !11
  %address.local90 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local89, !dbg !11
  %address.local91 = ptrtoint ptr addrspace(3) %address.local90 to i64, !dbg !11
  %address.local92 = trunc i64 %address.local91 to i32, !dbg !11
  %67 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local92), !dbg !11
  %68 = zext i32 %67 to i64, !dbg !11
  %69 = inttoptr i64 %68 to ptr addrspace(3), !dbg !11
  %address.local93 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local94 = and i64 %address.local93, 16256, !dbg !11
  %address.local95 = lshr i64 %address.local87, 4, !dbg !11
  %address.local96 = and i64 %address.local95, 24, !dbg !11
  %address.local97 = or disjoint i64 %address.local94, %address.local96, !dbg !11
  %address.local98 = getelementptr [2 x i8], ptr addrspace(1) %32, i64 %address.local97, !dbg !11
  %70 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %71 = ptrtoint ptr addrspace(1) %address.local98 to i64, !dbg !11
  %72 = sub i64 %71, %70, !dbg !11
  %dma.buffer.offset631 = trunc i64 %72 to i32, !dbg !11
  %73 = trunc i64 %70 to i32, !dbg !11
  %74 = insertelement <4 x i32> poison, i32 %73, i64 0, !dbg !11
  %75 = lshr i64 %70, 32, !dbg !11
  %76 = trunc i64 %75 to i32, !dbg !11
  %77 = insertelement <4 x i32> %74, i32 %76, i64 1, !dbg !11
  %78 = insertelement <4 x i32> %77, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc632 = insertelement <4 x i32> %78, i32 131072, i64 3, !dbg !11
  %79 = ptrtoint ptr addrspace(3) %69 to i32, !dbg !11
  %dma.buffer.lds633 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %79), !dbg !11
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds633, i32 %dma.buffer.offset631, <4 x i32> %dma.buffer.rsrc632) #10, !dbg !11
  %81 = lshr i64 %43, 13, !dbg !15
  %82 = add nsw i64 %81, %8, !dbg !15
  %83 = lshr i64 %43, 6, !dbg !15
  %84 = and i64 %83, 127, !dbg !15
  %85 = shl i64 %82, 19, !dbg !15
  %86 = and i64 %85, 16252928, !dbg !15
  %87 = shl i64 %83, 19, !dbg !15
  %88 = shl i64 %43, 13, !dbg !15
  %89 = sub i64 %88, %87, !dbg !15
  %90 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %86, !dbg !15
  %91 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %84, !dbg !15
  %92 = getelementptr [2 x i8], ptr addrspace(1) %91, i64 %89, !dbg !15
  %address.local99 = and i64 %address.local76, 1536, !dbg !15
  %address.local100 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local99, !dbg !15
  %address.local101 = ptrtoint ptr addrspace(3) %address.local100 to i64, !dbg !15
  %address.local102 = trunc i64 %address.local101 to i32, !dbg !15
  %93 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local102), !dbg !15
  %94 = zext i32 %93 to i64, !dbg !15
  %95 = inttoptr i64 %94 to ptr addrspace(3), !dbg !15
  %address.local103 = lshr i64 %address.local, 3, !dbg !15
  %address.local104 = and i64 %address.local103, 24, !dbg !15
  %address.local105 = shl nuw nsw i64 %address.local, 7, !dbg !15
  %address.local106 = and i64 %address.local105, 8064, !dbg !15
  %address.local107 = or disjoint i64 %address.local104, %address.local106, !dbg !15
  %address.local108 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local107, !dbg !15
  %96 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %97 = ptrtoint ptr addrspace(1) %address.local108 to i64, !dbg !15
  %98 = sub i64 %97, %96, !dbg !15
  %dma.buffer.offset628 = trunc i64 %98 to i32, !dbg !15
  %99 = trunc i64 %96 to i32, !dbg !15
  %100 = insertelement <4 x i32> poison, i32 %99, i64 0, !dbg !15
  %101 = lshr i64 %96, 32, !dbg !15
  %102 = trunc i64 %101 to i32, !dbg !15
  %103 = insertelement <4 x i32> %100, i32 %102, i64 1, !dbg !15
  %104 = insertelement <4 x i32> %103, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc629 = insertelement <4 x i32> %104, i32 131072, i64 3, !dbg !15
  %105 = ptrtoint ptr addrspace(3) %95 to i32, !dbg !15
  %dma.buffer.lds630 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %105), !dbg !15
  %106 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds630, i32 %dma.buffer.offset628, <4 x i32> %dma.buffer.rsrc629) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local109 = lshr i32 %address.thread, 2, !dbg !12
  %address.local110 = and i32 %address.local109, 4, !dbg !12
  %address.local111 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local112 = and i32 %address.local111, 384, !dbg !12
  %address.local113 = or disjoint i32 %address.local110, %address.local112, !dbg !12
  %address.local114 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local115 = and i32 %address.local114, 120, !dbg !12
  %address.local116 = or disjoint i32 %address.local113, %address.local115, !dbg !12
  %address.local117 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local118 = and i32 %address.local117, 1024, !dbg !12
  %address.local119 = or disjoint i32 %address.local116, %address.local118, !dbg !12
  %address.local120 = zext nneg i32 %address.local119 to i64, !dbg !12
  %address.local121 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local120, !dbg !12
  %107 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !12
  %address.local122 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local120, !dbg !12
  %address.local123 = getelementptr i8, ptr addrspace(3) %address.local122, i64 4096, !dbg !12
  %108 = load <4 x half>, ptr addrspace(3) %address.local123, align 2, !dbg !12
  %address.local124 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local120, !dbg !12
  %address.local125 = getelementptr i8, ptr addrspace(3) %address.local124, i64 1024, !dbg !12
  %109 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %110 = fmul <4 x half> %107, splat (half 0xH3015), !dbg !12
  %address.local126 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local120, !dbg !12
  %address.local127 = getelementptr i8, ptr addrspace(3) %address.local126, i64 5120, !dbg !12
  %111 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !12
  %112 = fmul <4 x half> %108, splat (half 0xH3015), !dbg !12
  %113 = fmul <4 x half> %109, splat (half 0xH3015), !dbg !12
  %114 = fmul <4 x half> %111, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local128 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local129 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local128, !dbg !16
  %address.local130 = lshr i64 %address.local, 2, !dbg !16
  %address.local131 = mul nsw i64 %address.local130, -512, !dbg !16
  %address.local132 = getelementptr [2 x i8], ptr addrspace(3) %address.local129, i64 %address.local131, !dbg !16
  %address.local133 = shl nuw nsw i64 %address.local, 1, !dbg !16
  %address.local134 = and i64 %address.local133, 24, !dbg !16
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) %address.local132, i64 %address.local134, !dbg !16
  %address.local136 = lshr i64 %address.local, 2, !dbg !12
  %address.local137 = and i64 %address.local136, 12, !dbg !12
  %address.local138 = getelementptr [2 x i8], ptr addrspace(3) %address.local135, i64 %address.local137, !dbg !16
  %address.local139 = and i64 %address.local, 63, !dbg !12
  %address.local140 = lshr i64 %address.local139, 5, !dbg !12
  %address.local141 = mul nuw nsw i64 %address.local140, 504, !dbg !16
  %address.local142 = getelementptr [2 x i8], ptr addrspace(3) %address.local138, i64 %address.local141, !dbg !16
  %115 = load <4 x half>, ptr addrspace(3) %address.local142, align 2, !dbg !16
  %116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %115, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local143 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local128, !dbg !16
  %address.local144 = getelementptr [2 x i8], ptr addrspace(3) %address.local143, i64 %address.local137, !dbg !16
  %address.local145 = getelementptr [2 x i8], ptr addrspace(3) %address.local144, i64 %address.local141, !dbg !16
  %address.local146 = getelementptr [2 x i8], ptr addrspace(3) %address.local145, i64 %address.local131, !dbg !16
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) %address.local146, i64 %address.local134, !dbg !16
  %address.local148 = getelementptr i8, ptr addrspace(3) %address.local147, i64 2048, !dbg !16
  %117 = load <4 x half>, ptr addrspace(3) %address.local148, align 2, !dbg !16
  %address.local149 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local128, !dbg !16
  %address.local150 = getelementptr [2 x i8], ptr addrspace(3) %address.local149, i64 %address.local131, !dbg !16
  %address.local151 = getelementptr [2 x i8], ptr addrspace(3) %address.local150, i64 %address.local134, !dbg !16
  %address.local152 = getelementptr [2 x i8], ptr addrspace(3) %address.local151, i64 %address.local137, !dbg !16
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) %address.local152, i64 %address.local141, !dbg !16
  %address.local154 = getelementptr i8, ptr addrspace(3) %address.local153, i64 64, !dbg !16
  %118 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %117, <4 x float> %116) #9, !dbg !16
  %120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %118, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local155 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local128, !dbg !16
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) %address.local155, i64 %address.local137, !dbg !16
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) %address.local156, i64 %address.local141, !dbg !16
  %address.local158 = getelementptr [2 x i8], ptr addrspace(3) %address.local157, i64 %address.local131, !dbg !16
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) %address.local158, i64 %address.local134, !dbg !16
  %address.local160 = getelementptr i8, ptr addrspace(3) %address.local159, i64 2112, !dbg !16
  %121 = load <4 x half>, ptr addrspace(3) %address.local160, align 2, !dbg !16
  %address.local161 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local128, !dbg !16
  %address.local162 = getelementptr [2 x i8], ptr addrspace(3) %address.local161, i64 %address.local131, !dbg !16
  %address.local163 = getelementptr [2 x i8], ptr addrspace(3) %address.local162, i64 %address.local134, !dbg !16
  %address.local164 = getelementptr [2 x i8], ptr addrspace(3) %address.local163, i64 %address.local137, !dbg !16
  %address.local165 = getelementptr [2 x i8], ptr addrspace(3) %address.local164, i64 %address.local141, !dbg !16
  %address.local166 = getelementptr i8, ptr addrspace(3) %address.local165, i64 128, !dbg !16
  %122 = load <4 x half>, ptr addrspace(3) %address.local166, align 2, !dbg !16
  %123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %121, <4 x float> %120) #9, !dbg !16
  %124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %122, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local167 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local128, !dbg !16
  %address.local168 = getelementptr [2 x i8], ptr addrspace(3) %address.local167, i64 %address.local137, !dbg !16
  %address.local169 = getelementptr [2 x i8], ptr addrspace(3) %address.local168, i64 %address.local141, !dbg !16
  %address.local170 = getelementptr [2 x i8], ptr addrspace(3) %address.local169, i64 %address.local131, !dbg !16
  %address.local171 = getelementptr [2 x i8], ptr addrspace(3) %address.local170, i64 %address.local134, !dbg !16
  %address.local172 = getelementptr i8, ptr addrspace(3) %address.local171, i64 2176, !dbg !16
  %125 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !16
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local128, !dbg !16
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) %address.local173, i64 %address.local131, !dbg !16
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) %address.local174, i64 %address.local134, !dbg !16
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) %address.local175, i64 %address.local137, !dbg !16
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) %address.local176, i64 %address.local141, !dbg !16
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local177, i64 192, !dbg !16
  %126 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !16
  %127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %125, <4 x float> %124) #9, !dbg !16
  %128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %126, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local128, !dbg !16
  %address.local180 = getelementptr [2 x i8], ptr addrspace(3) %address.local179, i64 %address.local137, !dbg !16
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) %address.local180, i64 %address.local141, !dbg !16
  %address.local182 = getelementptr [2 x i8], ptr addrspace(3) %address.local181, i64 %address.local131, !dbg !16
  %address.local183 = getelementptr [2 x i8], ptr addrspace(3) %address.local182, i64 %address.local134, !dbg !16
  %address.local184 = getelementptr i8, ptr addrspace(3) %address.local183, i64 2240, !dbg !16
  %129 = load <4 x half>, ptr addrspace(3) %address.local184, align 2, !dbg !16
  %130 = load <4 x half>, ptr addrspace(3) %address.local142, align 2, !dbg !16
  %131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %129, <4 x float> %128) #9, !dbg !16
  %132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %130, <4 x float> zeroinitializer) #9, !dbg !16
  %133 = load <4 x half>, ptr addrspace(3) %address.local148, align 2, !dbg !16
  %134 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %133, <4 x float> %132) #9, !dbg !16
  %136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %134, <4 x float> zeroinitializer) #9, !dbg !16
  %137 = load <4 x half>, ptr addrspace(3) %address.local160, align 2, !dbg !16
  %138 = load <4 x half>, ptr addrspace(3) %address.local166, align 2, !dbg !16
  %139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %137, <4 x float> %136) #9, !dbg !16
  %140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %138, <4 x float> zeroinitializer) #9, !dbg !16
  %141 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !16
  %142 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !16
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %141, <4 x float> %140) #9, !dbg !16
  %144 = load <4 x half>, ptr addrspace(3) %address.local184, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %142, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %144, <4 x float> %145) #9, !dbg !16
  %147 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local80), !dbg !11
  %148 = zext i32 %147 to i64, !dbg !11
  %149 = inttoptr i64 %148 to ptr addrspace(3), !dbg !11
  %address.local185 = getelementptr [2 x i8], ptr addrspace(1) %34, i64 %address.local85, !dbg !11
  %150 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %151 = ptrtoint ptr addrspace(1) %address.local185 to i64, !dbg !11
  %152 = sub i64 %151, %150, !dbg !11
  %dma.buffer.offset625 = trunc i64 %152 to i32, !dbg !11
  %153 = trunc i64 %150 to i32, !dbg !11
  %154 = insertelement <4 x i32> poison, i32 %153, i64 0, !dbg !11
  %155 = lshr i64 %150, 32, !dbg !11
  %156 = trunc i64 %155 to i32, !dbg !11
  %157 = insertelement <4 x i32> %154, i32 %156, i64 1, !dbg !11
  %158 = insertelement <4 x i32> %157, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc626 = insertelement <4 x i32> %158, i32 131072, i64 3, !dbg !11
  %159 = ptrtoint ptr addrspace(3) %149 to i32, !dbg !11
  %dma.buffer.lds627 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %159), !dbg !11
  %160 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds627, i32 %dma.buffer.offset625, <4 x i32> %dma.buffer.rsrc626) #10, !dbg !11
  %161 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local92), !dbg !11
  %162 = zext i32 %161 to i64, !dbg !11
  %163 = inttoptr i64 %162 to ptr addrspace(3), !dbg !11
  %address.local186 = getelementptr [2 x i8], ptr addrspace(1) %34, i64 %address.local97, !dbg !11
  %164 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %165 = ptrtoint ptr addrspace(1) %address.local186 to i64, !dbg !11
  %166 = sub i64 %165, %164, !dbg !11
  %dma.buffer.offset622 = trunc i64 %166 to i32, !dbg !11
  %167 = trunc i64 %164 to i32, !dbg !11
  %168 = insertelement <4 x i32> poison, i32 %167, i64 0, !dbg !11
  %169 = lshr i64 %164, 32, !dbg !11
  %170 = trunc i64 %169 to i32, !dbg !11
  %171 = insertelement <4 x i32> %168, i32 %170, i64 1, !dbg !11
  %172 = insertelement <4 x i32> %171, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc623 = insertelement <4 x i32> %172, i32 131072, i64 3, !dbg !11
  %173 = ptrtoint ptr addrspace(3) %163 to i32, !dbg !11
  %dma.buffer.lds624 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %173), !dbg !11
  %174 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds624, i32 %dma.buffer.offset622, <4 x i32> %dma.buffer.rsrc623) #10, !dbg !11
  %175 = add nuw i64 %43, 2048, !dbg !15
  %176 = icmp slt i64 %175, 0, !dbg !15
  %177 = sub nuw i64 -2049, %43, !dbg !15
  %178 = select i1 %176, i64 %177, i64 %175, !dbg !15
  %179 = sdiv i64 %178, 8192, !dbg !15
  %.lobit3 = ashr i64 %175, 63, !dbg !15
  %180 = xor i64 %179, %.lobit3, !dbg !15
  %181 = add nsw i64 %180, %8, !dbg !15
  %182 = add nsw i64 %83, %37, !dbg !15
  %183 = add nsw i64 %182, 32, !dbg !15
  %184 = icmp slt i64 %182, -32, !dbg !15
  %185 = sub nsw i64 -33, %182, !dbg !15
  %186 = select i1 %184, i64 %185, i64 %183, !dbg !15
  %187 = sdiv i64 %186, 128, !dbg !15
  %188 = sext i1 %184 to i64, !dbg !15
  %189 = xor i64 %187, %188, !dbg !15
  %190 = shl nsw i64 %189, 7, !dbg !15
  %191 = sub i64 %37, %190, !dbg !15
  %192 = shl i64 %181, 19, !dbg !15
  %193 = and i64 %192, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %194 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %193, !dbg !15
  %195 = getelementptr [2 x i8], ptr addrspace(1) %194, i64 %83, !dbg !15
  %196 = getelementptr [2 x i8], ptr addrspace(1) %195, i64 %191, !dbg !15
  %197 = getelementptr i8, ptr addrspace(1) %196, i64 64, !dbg !15
  %198 = getelementptr [2 x i8], ptr addrspace(1) %197, i64 %89, !dbg !15
  %199 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local102), !dbg !15
  %200 = zext i32 %199 to i64, !dbg !15
  %201 = inttoptr i64 %200 to ptr addrspace(3), !dbg !15
  %address.local187 = getelementptr [2 x i8], ptr addrspace(1) %198, i64 %address.local107, !dbg !15
  %202 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %203 = ptrtoint ptr addrspace(1) %address.local187 to i64, !dbg !15
  %204 = sub i64 %203, %202, !dbg !15
  %dma.buffer.offset619 = trunc i64 %204 to i32, !dbg !15
  %205 = trunc i64 %202 to i32, !dbg !15
  %206 = insertelement <4 x i32> poison, i32 %205, i64 0, !dbg !15
  %207 = lshr i64 %202, 32, !dbg !15
  %208 = trunc i64 %207 to i32, !dbg !15
  %209 = insertelement <4 x i32> %206, i32 %208, i64 1, !dbg !15
  %210 = insertelement <4 x i32> %209, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc620 = insertelement <4 x i32> %210, i32 131072, i64 3, !dbg !15
  %211 = ptrtoint ptr addrspace(3) %201 to i32, !dbg !15
  %dma.buffer.lds621 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %211), !dbg !15
  %212 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds621, i32 %dma.buffer.offset619, <4 x i32> %dma.buffer.rsrc620) #10, !dbg !15
  %213 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !12
  %214 = load <4 x half>, ptr addrspace(3) %address.local123, align 2, !dbg !12
  %215 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %216 = fmul <4 x half> %213, splat (half 0xH3015), !dbg !12
  %217 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !12
  %218 = fmul <4 x half> %214, splat (half 0xH3015), !dbg !12
  %219 = fmul <4 x half> %215, splat (half 0xH3015), !dbg !12
  %220 = fmul <4 x half> %217, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %221 = load <4 x half>, ptr addrspace(3) %address.local142, align 2, !dbg !16
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %216, <4 x half> %221, <4 x float> %119) #9, !dbg !16
  %223 = load <4 x half>, ptr addrspace(3) %address.local148, align 2, !dbg !16
  %224 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %218, <4 x half> %223, <4 x float> %222) #9, !dbg !16
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %216, <4 x half> %224, <4 x float> %123) #9, !dbg !16
  %227 = load <4 x half>, ptr addrspace(3) %address.local160, align 2, !dbg !16
  %228 = load <4 x half>, ptr addrspace(3) %address.local166, align 2, !dbg !16
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %218, <4 x half> %227, <4 x float> %226) #9, !dbg !16
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %216, <4 x half> %228, <4 x float> %127) #9, !dbg !16
  %231 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !16
  %232 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !16
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %218, <4 x half> %231, <4 x float> %230) #9, !dbg !16
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %216, <4 x half> %232, <4 x float> %131) #9, !dbg !16
  %235 = load <4 x half>, ptr addrspace(3) %address.local184, align 2, !dbg !16
  %236 = load <4 x half>, ptr addrspace(3) %address.local142, align 2, !dbg !16
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %218, <4 x half> %235, <4 x float> %234) #9, !dbg !16
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %219, <4 x half> %236, <4 x float> %135) #9, !dbg !16
  %239 = load <4 x half>, ptr addrspace(3) %address.local148, align 2, !dbg !16
  %240 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %220, <4 x half> %239, <4 x float> %238) #9, !dbg !16
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %219, <4 x half> %240, <4 x float> %139) #9, !dbg !16
  %243 = load <4 x half>, ptr addrspace(3) %address.local160, align 2, !dbg !16
  %244 = load <4 x half>, ptr addrspace(3) %address.local166, align 2, !dbg !16
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %220, <4 x half> %243, <4 x float> %242) #9, !dbg !16
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %219, <4 x half> %244, <4 x float> %143) #9, !dbg !16
  %247 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !16
  %248 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !16
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %220, <4 x half> %247, <4 x float> %246) #9, !dbg !16
  %250 = load <4 x half>, ptr addrspace(3) %address.local184, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %219, <4 x half> %248, <4 x float> %146) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %220, <4 x half> %250, <4 x float> %251) #9, !dbg !16
  %253 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local80), !dbg !11
  %254 = zext i32 %253 to i64, !dbg !11
  %255 = inttoptr i64 %254 to ptr addrspace(3), !dbg !11
  %address.local188 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %address.local85, !dbg !11
  %256 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %257 = ptrtoint ptr addrspace(1) %address.local188 to i64, !dbg !11
  %258 = sub i64 %257, %256, !dbg !11
  %dma.buffer.offset616 = trunc i64 %258 to i32, !dbg !11
  %259 = trunc i64 %256 to i32, !dbg !11
  %260 = insertelement <4 x i32> poison, i32 %259, i64 0, !dbg !11
  %261 = lshr i64 %256, 32, !dbg !11
  %262 = trunc i64 %261 to i32, !dbg !11
  %263 = insertelement <4 x i32> %260, i32 %262, i64 1, !dbg !11
  %264 = insertelement <4 x i32> %263, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc617 = insertelement <4 x i32> %264, i32 131072, i64 3, !dbg !11
  %265 = ptrtoint ptr addrspace(3) %255 to i32, !dbg !11
  %dma.buffer.lds618 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %265), !dbg !11
  %266 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds618, i32 %dma.buffer.offset616, <4 x i32> %dma.buffer.rsrc617) #10, !dbg !11
  %267 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local92), !dbg !11
  %268 = zext i32 %267 to i64, !dbg !11
  %269 = inttoptr i64 %268 to ptr addrspace(3), !dbg !11
  %address.local189 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %address.local97, !dbg !11
  %270 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %271 = ptrtoint ptr addrspace(1) %address.local189 to i64, !dbg !11
  %272 = sub i64 %271, %270, !dbg !11
  %dma.buffer.offset613 = trunc i64 %272 to i32, !dbg !11
  %273 = trunc i64 %270 to i32, !dbg !11
  %274 = insertelement <4 x i32> poison, i32 %273, i64 0, !dbg !11
  %275 = lshr i64 %270, 32, !dbg !11
  %276 = trunc i64 %275 to i32, !dbg !11
  %277 = insertelement <4 x i32> %274, i32 %276, i64 1, !dbg !11
  %278 = insertelement <4 x i32> %277, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc614 = insertelement <4 x i32> %278, i32 131072, i64 3, !dbg !11
  %279 = ptrtoint ptr addrspace(3) %269 to i32, !dbg !11
  %dma.buffer.lds615 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %279), !dbg !11
  %280 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds615, i32 %dma.buffer.offset613, <4 x i32> %dma.buffer.rsrc614) #10, !dbg !11
  %281 = add nuw i64 %43, 4096, !dbg !15
  %282 = icmp slt i64 %281, 0, !dbg !15
  %283 = sub nuw i64 -4097, %43, !dbg !15
  %284 = select i1 %282, i64 %283, i64 %281, !dbg !15
  %285 = sdiv i64 %284, 8192, !dbg !15
  %.lobit4 = ashr i64 %281, 63, !dbg !15
  %286 = xor i64 %285, %.lobit4, !dbg !15
  %287 = add nsw i64 %286, %8, !dbg !15
  %288 = add nsw i64 %182, 64, !dbg !15
  %289 = icmp slt i64 %182, -64, !dbg !15
  %290 = sub nsw i64 -65, %182, !dbg !15
  %291 = select i1 %289, i64 %290, i64 %288, !dbg !15
  %292 = sdiv i64 %291, 128, !dbg !15
  %293 = sext i1 %289 to i64, !dbg !15
  %294 = xor i64 %292, %293, !dbg !15
  %295 = shl nsw i64 %294, 7, !dbg !15
  %296 = sub i64 %37, %295, !dbg !15
  %297 = shl i64 %287, 19, !dbg !15
  %298 = and i64 %297, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %299 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %298, !dbg !15
  %300 = getelementptr [2 x i8], ptr addrspace(1) %299, i64 %83, !dbg !15
  %301 = getelementptr [2 x i8], ptr addrspace(1) %300, i64 %296, !dbg !15
  %302 = getelementptr i8, ptr addrspace(1) %301, i64 128, !dbg !15
  %303 = getelementptr [2 x i8], ptr addrspace(1) %302, i64 %89, !dbg !15
  %304 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local102), !dbg !15
  %305 = zext i32 %304 to i64, !dbg !15
  %306 = inttoptr i64 %305 to ptr addrspace(3), !dbg !15
  %address.local190 = getelementptr [2 x i8], ptr addrspace(1) %303, i64 %address.local107, !dbg !15
  %307 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %308 = ptrtoint ptr addrspace(1) %address.local190 to i64, !dbg !15
  %309 = sub i64 %308, %307, !dbg !15
  %dma.buffer.offset610 = trunc i64 %309 to i32, !dbg !15
  %310 = trunc i64 %307 to i32, !dbg !15
  %311 = insertelement <4 x i32> poison, i32 %310, i64 0, !dbg !15
  %312 = lshr i64 %307, 32, !dbg !15
  %313 = trunc i64 %312 to i32, !dbg !15
  %314 = insertelement <4 x i32> %311, i32 %313, i64 1, !dbg !15
  %315 = insertelement <4 x i32> %314, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc611 = insertelement <4 x i32> %315, i32 131072, i64 3, !dbg !15
  %316 = ptrtoint ptr addrspace(3) %306 to i32, !dbg !15
  %dma.buffer.lds612 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %316), !dbg !15
  %317 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds612, i32 %dma.buffer.offset610, <4 x i32> %dma.buffer.rsrc611) #10, !dbg !15
  %318 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !12
  %319 = load <4 x half>, ptr addrspace(3) %address.local123, align 2, !dbg !12
  %320 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %321 = fmul <4 x half> %318, splat (half 0xH3015), !dbg !12
  %322 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !12
  %323 = fmul <4 x half> %319, splat (half 0xH3015), !dbg !12
  %324 = fmul <4 x half> %320, splat (half 0xH3015), !dbg !12
  %325 = fmul <4 x half> %322, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %326 = load <4 x half>, ptr addrspace(3) %address.local142, align 2, !dbg !16
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %321, <4 x half> %326, <4 x float> %225) #9, !dbg !16
  %328 = load <4 x half>, ptr addrspace(3) %address.local148, align 2, !dbg !16
  %329 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %323, <4 x half> %328, <4 x float> %327) #9, !dbg !16
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %321, <4 x half> %329, <4 x float> %229) #9, !dbg !16
  %332 = load <4 x half>, ptr addrspace(3) %address.local160, align 2, !dbg !16
  %333 = load <4 x half>, ptr addrspace(3) %address.local166, align 2, !dbg !16
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %323, <4 x half> %332, <4 x float> %331) #9, !dbg !16
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %321, <4 x half> %333, <4 x float> %233) #9, !dbg !16
  %336 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !16
  %337 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !16
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %323, <4 x half> %336, <4 x float> %335) #9, !dbg !16
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %321, <4 x half> %337, <4 x float> %237) #9, !dbg !16
  %340 = load <4 x half>, ptr addrspace(3) %address.local184, align 2, !dbg !16
  %341 = load <4 x half>, ptr addrspace(3) %address.local142, align 2, !dbg !16
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %323, <4 x half> %340, <4 x float> %339) #9, !dbg !16
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %324, <4 x half> %341, <4 x float> %241) #9, !dbg !16
  %344 = load <4 x half>, ptr addrspace(3) %address.local148, align 2, !dbg !16
  %345 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %344, <4 x float> %343) #9, !dbg !16
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %324, <4 x half> %345, <4 x float> %245) #9, !dbg !16
  %348 = load <4 x half>, ptr addrspace(3) %address.local160, align 2, !dbg !16
  %349 = load <4 x half>, ptr addrspace(3) %address.local166, align 2, !dbg !16
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %348, <4 x float> %347) #9, !dbg !16
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %324, <4 x half> %349, <4 x float> %249) #9, !dbg !16
  %352 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !16
  %353 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !16
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %352, <4 x float> %351) #9, !dbg !16
  %355 = load <4 x half>, ptr addrspace(3) %address.local184, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %324, <4 x half> %353, <4 x float> %252) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %355, <4 x float> %356) #9, !dbg !16
  %358 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local80), !dbg !11
  %359 = zext i32 %358 to i64, !dbg !11
  %360 = inttoptr i64 %359 to ptr addrspace(3), !dbg !11
  %address.local191 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local85, !dbg !11
  %361 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %362 = ptrtoint ptr addrspace(1) %address.local191 to i64, !dbg !11
  %363 = sub i64 %362, %361, !dbg !11
  %dma.buffer.offset607 = trunc i64 %363 to i32, !dbg !11
  %364 = trunc i64 %361 to i32, !dbg !11
  %365 = insertelement <4 x i32> poison, i32 %364, i64 0, !dbg !11
  %366 = lshr i64 %361, 32, !dbg !11
  %367 = trunc i64 %366 to i32, !dbg !11
  %368 = insertelement <4 x i32> %365, i32 %367, i64 1, !dbg !11
  %369 = insertelement <4 x i32> %368, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc608 = insertelement <4 x i32> %369, i32 131072, i64 3, !dbg !11
  %370 = ptrtoint ptr addrspace(3) %360 to i32, !dbg !11
  %dma.buffer.lds609 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %370), !dbg !11
  %371 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds609, i32 %dma.buffer.offset607, <4 x i32> %dma.buffer.rsrc608) #10, !dbg !11
  %372 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local92), !dbg !11
  %373 = zext i32 %372 to i64, !dbg !11
  %374 = inttoptr i64 %373 to ptr addrspace(3), !dbg !11
  %address.local192 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local97, !dbg !11
  %375 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %376 = ptrtoint ptr addrspace(1) %address.local192 to i64, !dbg !11
  %377 = sub i64 %376, %375, !dbg !11
  %dma.buffer.offset604 = trunc i64 %377 to i32, !dbg !11
  %378 = trunc i64 %375 to i32, !dbg !11
  %379 = insertelement <4 x i32> poison, i32 %378, i64 0, !dbg !11
  %380 = lshr i64 %375, 32, !dbg !11
  %381 = trunc i64 %380 to i32, !dbg !11
  %382 = insertelement <4 x i32> %379, i32 %381, i64 1, !dbg !11
  %383 = insertelement <4 x i32> %382, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc605 = insertelement <4 x i32> %383, i32 131072, i64 3, !dbg !11
  %384 = ptrtoint ptr addrspace(3) %374 to i32, !dbg !11
  %dma.buffer.lds606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %384), !dbg !11
  %385 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds606, i32 %dma.buffer.offset604, <4 x i32> %dma.buffer.rsrc605) #10, !dbg !11
  %386 = add nuw i64 %43, 6144, !dbg !15
  %387 = icmp slt i64 %386, 0, !dbg !15
  %388 = sub nuw i64 -6145, %43, !dbg !15
  %389 = select i1 %387, i64 %388, i64 %386, !dbg !15
  %390 = sdiv i64 %389, 8192, !dbg !15
  %.lobit5 = ashr i64 %386, 63, !dbg !15
  %391 = xor i64 %390, %.lobit5, !dbg !15
  %392 = add nsw i64 %391, %8, !dbg !15
  %393 = add nsw i64 %182, 96, !dbg !15
  %394 = icmp slt i64 %182, -96, !dbg !15
  %395 = sub nsw i64 -97, %182, !dbg !15
  %396 = select i1 %394, i64 %395, i64 %393, !dbg !15
  %397 = sdiv i64 %396, 128, !dbg !15
  %398 = sext i1 %394 to i64, !dbg !15
  %399 = xor i64 %397, %398, !dbg !15
  %400 = shl nsw i64 %399, 7, !dbg !15
  %401 = sub i64 %37, %400, !dbg !15
  %402 = shl i64 %392, 19, !dbg !15
  %403 = and i64 %402, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %404 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %403, !dbg !15
  %405 = getelementptr [2 x i8], ptr addrspace(1) %404, i64 %83, !dbg !15
  %406 = getelementptr [2 x i8], ptr addrspace(1) %405, i64 %401, !dbg !15
  %407 = getelementptr i8, ptr addrspace(1) %406, i64 192, !dbg !15
  %408 = getelementptr [2 x i8], ptr addrspace(1) %407, i64 %89, !dbg !15
  %409 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local102), !dbg !15
  %410 = zext i32 %409 to i64, !dbg !15
  %411 = inttoptr i64 %410 to ptr addrspace(3), !dbg !15
  %address.local193 = getelementptr [2 x i8], ptr addrspace(1) %408, i64 %address.local107, !dbg !15
  %412 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %413 = ptrtoint ptr addrspace(1) %address.local193 to i64, !dbg !15
  %414 = sub i64 %413, %412, !dbg !15
  %dma.buffer.offset601 = trunc i64 %414 to i32, !dbg !15
  %415 = trunc i64 %412 to i32, !dbg !15
  %416 = insertelement <4 x i32> poison, i32 %415, i64 0, !dbg !15
  %417 = lshr i64 %412, 32, !dbg !15
  %418 = trunc i64 %417 to i32, !dbg !15
  %419 = insertelement <4 x i32> %416, i32 %418, i64 1, !dbg !15
  %420 = insertelement <4 x i32> %419, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc602 = insertelement <4 x i32> %420, i32 131072, i64 3, !dbg !15
  %421 = ptrtoint ptr addrspace(3) %411 to i32, !dbg !15
  %dma.buffer.lds603 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %421), !dbg !15
  %422 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds603, i32 %dma.buffer.offset601, <4 x i32> %dma.buffer.rsrc602) #10, !dbg !15
  %423 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !12
  %424 = load <4 x half>, ptr addrspace(3) %address.local123, align 2, !dbg !12
  %425 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %426 = fmul <4 x half> %423, splat (half 0xH3015), !dbg !12
  %427 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !12
  %428 = fmul <4 x half> %424, splat (half 0xH3015), !dbg !12
  %429 = fmul <4 x half> %425, splat (half 0xH3015), !dbg !12
  %430 = fmul <4 x half> %427, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %431 = add nsw i64 %83, %8, !dbg !17
  %432 = shl i64 %431, 19, !dbg !17
  %433 = and i64 %432, 16252928, !dbg !17
  %434 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %433, !dbg !17
  %435 = getelementptr [2 x i8], ptr addrspace(1) %434, i64 %89, !dbg !17
  %address.local194 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local99, !dbg !17
  %address.local195 = ptrtoint ptr addrspace(3) %address.local194 to i64, !dbg !17
  %address.local196 = trunc i64 %address.local195 to i32, !dbg !17
  %436 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local196), !dbg !17
  %437 = zext i32 %436 to i64, !dbg !17
  %438 = inttoptr i64 %437 to ptr addrspace(3), !dbg !17
  %address.local197 = shl nuw nsw i64 %address.local, 5, !dbg !17
  %address.local198 = and i64 %address.local197, 8064, !dbg !17
  %address.local199 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local200 = and i64 %address.local199, 24, !dbg !17
  %address.local201 = or disjoint i64 %address.local198, %address.local200, !dbg !17
  %address.local202 = getelementptr [2 x i8], ptr addrspace(1) %435, i64 %address.local201, !dbg !17
  %439 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %440 = ptrtoint ptr addrspace(1) %address.local202 to i64, !dbg !17
  %441 = sub i64 %440, %439, !dbg !17
  %dma.buffer.offset598 = trunc i64 %441 to i32, !dbg !17
  %442 = trunc i64 %439 to i32, !dbg !17
  %443 = insertelement <4 x i32> poison, i32 %442, i64 0, !dbg !17
  %444 = lshr i64 %439, 32, !dbg !17
  %445 = trunc i64 %444 to i32, !dbg !17
  %446 = insertelement <4 x i32> %443, i32 %445, i64 1, !dbg !17
  %447 = insertelement <4 x i32> %446, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc599 = insertelement <4 x i32> %447, i32 131072, i64 3, !dbg !17
  %448 = ptrtoint ptr addrspace(3) %438 to i32, !dbg !17
  %dma.buffer.lds600 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %448), !dbg !17
  %449 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds600, i32 %dma.buffer.offset598, <4 x i32> %dma.buffer.rsrc599) #10, !dbg !17
  %450 = load <4 x half>, ptr addrspace(3) %address.local142, align 2, !dbg !16
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %450, <4 x float> %330) #9, !dbg !16
  %452 = load <4 x half>, ptr addrspace(3) %address.local148, align 2, !dbg !16
  %453 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %452, <4 x float> %451) #9, !dbg !16
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %453, <4 x float> %334) #9, !dbg !16
  %456 = load <4 x half>, ptr addrspace(3) %address.local160, align 2, !dbg !16
  %457 = load <4 x half>, ptr addrspace(3) %address.local166, align 2, !dbg !16
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %456, <4 x float> %455) #9, !dbg !16
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %457, <4 x float> %338) #9, !dbg !16
  %460 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !16
  %461 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !16
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %460, <4 x float> %459) #9, !dbg !16
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %461, <4 x float> %342) #9, !dbg !16
  %464 = load <4 x half>, ptr addrspace(3) %address.local184, align 2, !dbg !16
  %465 = load <4 x half>, ptr addrspace(3) %address.local142, align 2, !dbg !16
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %464, <4 x float> %463) #9, !dbg !16
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %429, <4 x half> %465, <4 x float> %346) #9, !dbg !16
  %468 = load <4 x half>, ptr addrspace(3) %address.local148, align 2, !dbg !16
  %469 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %430, <4 x half> %468, <4 x float> %467) #9, !dbg !16
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %429, <4 x half> %469, <4 x float> %350) #9, !dbg !16
  %472 = load <4 x half>, ptr addrspace(3) %address.local160, align 2, !dbg !16
  %473 = load <4 x half>, ptr addrspace(3) %address.local166, align 2, !dbg !16
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %430, <4 x half> %472, <4 x float> %471) #9, !dbg !16
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %429, <4 x half> %473, <4 x float> %354) #9, !dbg !16
  %476 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !16
  %477 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !16
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %430, <4 x half> %476, <4 x float> %475) #9, !dbg !16
  %479 = load <4 x half>, ptr addrspace(3) %address.local184, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %429, <4 x half> %477, <4 x float> %357) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %481 = or disjoint i64 %52, 63, !dbg !13
  %482 = icmp uge i64 %11, %481, !dbg !13
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %430, <4 x half> %479, <4 x float> %480) #9, !dbg !16
  %484 = and i1 %29, %482, !dbg !13
  br i1 %484, label %485, label %486, !dbg !13

485:                                              ; preds = %51
  br label %566, !dbg !13

486:                                              ; preds = %51
  %487 = add nsw i64 %11, %25, !dbg !13
  %488 = shl nuw nsw i64 %26, 4, !dbg !13
  %489 = add nsw i64 %487, %488, !dbg !13
  %490 = and i32 %24, 240, !dbg !13
  %index.compact = zext nneg i32 %490 to i64, !dbg !13
  %491 = sub nsw i64 %489, %index.compact, !dbg !13
  %492 = and i64 %25, 48, !dbg !13
  %493 = or disjoint i64 %52, %492, !dbg !13
  %494 = add nsw i64 %491, 1, !dbg !22
  %495 = lshr i64 %494, 32, !dbg !23
  %compare.high = trunc i64 %495 to i32, !dbg !23
  %496 = lshr i64 %493, 32, !dbg !23
  %compare.high464 = trunc i64 %496 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high464, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high464, !dbg !23
  %compare.low.base = trunc i64 %494 to i32, !dbg !23
  %compare.low.base465 = trunc i64 %493 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base465, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %497 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low466 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result467 = select i1 %compare.high.equal, i1 %compare.low466, i1 %compare.high.order, !dbg !23
  %498 = select i1 %compare.result467, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base468 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local469 = or i32 %compare.local.base468, 2, !dbg !23
  %compare.low470 = icmp ugt i32 %compare.low.base, %compare.local469, !dbg !23
  %compare.result471 = select i1 %compare.high.equal, i1 %compare.low470, i1 %compare.high.order, !dbg !23
  %499 = select i1 %compare.result471, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base472 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local473 = or i32 %compare.local.base472, 3, !dbg !23
  %compare.low474 = icmp ugt i32 %compare.low.base, %compare.local473, !dbg !23
  %compare.result475 = select i1 %compare.high.equal, i1 %compare.low474, i1 %compare.high.order, !dbg !23
  %500 = select i1 %compare.result475, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base476 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local477 = or i32 %compare.local.base476, 4, !dbg !23
  %compare.low478 = icmp ugt i32 %compare.low.base, %compare.local477, !dbg !23
  %compare.result479 = select i1 %compare.high.equal, i1 %compare.low478, i1 %compare.high.order, !dbg !23
  %501 = select i1 %compare.result479, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base480 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local481 = or i32 %compare.local.base480, 5, !dbg !23
  %compare.low482 = icmp ugt i32 %compare.low.base, %compare.local481, !dbg !23
  %compare.result483 = select i1 %compare.high.equal, i1 %compare.low482, i1 %compare.high.order, !dbg !23
  %502 = select i1 %compare.result483, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base484 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local485 = or i32 %compare.local.base484, 6, !dbg !23
  %compare.low486 = icmp ugt i32 %compare.low.base, %compare.local485, !dbg !23
  %compare.result487 = select i1 %compare.high.equal, i1 %compare.low486, i1 %compare.high.order, !dbg !23
  %503 = select i1 %compare.result487, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base488 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local489 = or i32 %compare.local.base488, 7, !dbg !23
  %compare.low490 = icmp ugt i32 %compare.low.base, %compare.local489, !dbg !23
  %compare.result491 = select i1 %compare.high.equal, i1 %compare.low490, i1 %compare.high.order, !dbg !23
  %504 = select i1 %compare.result491, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base492 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local493 = or i32 %compare.local.base492, 8, !dbg !23
  %compare.low494 = icmp ugt i32 %compare.low.base, %compare.local493, !dbg !23
  %compare.result495 = select i1 %compare.high.equal, i1 %compare.low494, i1 %compare.high.order, !dbg !23
  %505 = select i1 %compare.result495, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base496 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local497 = or i32 %compare.local.base496, 9, !dbg !23
  %compare.low498 = icmp ugt i32 %compare.low.base, %compare.local497, !dbg !23
  %compare.result499 = select i1 %compare.high.equal, i1 %compare.low498, i1 %compare.high.order, !dbg !23
  %506 = select i1 %compare.result499, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base500 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local501 = or i32 %compare.local.base500, 10, !dbg !23
  %compare.low502 = icmp ugt i32 %compare.low.base, %compare.local501, !dbg !23
  %compare.result503 = select i1 %compare.high.equal, i1 %compare.low502, i1 %compare.high.order, !dbg !23
  %507 = select i1 %compare.result503, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base504 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local505 = or i32 %compare.local.base504, 11, !dbg !23
  %compare.low506 = icmp ugt i32 %compare.low.base, %compare.local505, !dbg !23
  %compare.result507 = select i1 %compare.high.equal, i1 %compare.low506, i1 %compare.high.order, !dbg !23
  %508 = select i1 %compare.result507, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base508 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local509 = or i32 %compare.local.base508, 12, !dbg !23
  %compare.low510 = icmp ugt i32 %compare.low.base, %compare.local509, !dbg !23
  %compare.result511 = select i1 %compare.high.equal, i1 %compare.low510, i1 %compare.high.order, !dbg !23
  %509 = select i1 %compare.result511, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base512 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local513 = or i32 %compare.local.base512, 13, !dbg !23
  %compare.low514 = icmp ugt i32 %compare.low.base, %compare.local513, !dbg !23
  %compare.result515 = select i1 %compare.high.equal, i1 %compare.low514, i1 %compare.high.order, !dbg !23
  %510 = select i1 %compare.result515, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base516 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local517 = or i32 %compare.local.base516, 14, !dbg !23
  %compare.low518 = icmp ugt i32 %compare.low.base, %compare.local517, !dbg !23
  %compare.result519 = select i1 %compare.high.equal, i1 %compare.low518, i1 %compare.high.order, !dbg !23
  %511 = select i1 %compare.result519, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base520 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base465, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local521 = or i32 %compare.local.base520, 15, !dbg !23
  %compare.low522 = icmp ugt i32 %compare.low.base, %compare.local521, !dbg !23
  %compare.result523 = select i1 %compare.high.equal, i1 %compare.low522, i1 %compare.high.order, !dbg !23
  %512 = select i1 %compare.result523, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %513 = add nsw i64 %491, 65, !dbg !22
  %514 = lshr i64 %513, 32, !dbg !23
  %compare.high524 = trunc i64 %514 to i32, !dbg !23
  %515 = lshr i64 %493, 32, !dbg !23
  %compare.high525 = trunc i64 %515 to i32, !dbg !23
  %compare.high.equal526 = icmp eq i32 %compare.high524, %compare.high525, !dbg !23
  %compare.high.order527 = icmp ugt i32 %compare.high524, %compare.high525, !dbg !23
  %compare.low.base528 = trunc i64 %513 to i32, !dbg !23
  %compare.low.base529 = trunc i64 %493 to i32, !dbg !23
  %compare.low530 = icmp ugt i32 %compare.low.base528, %compare.low.base529, !dbg !23
  %compare.result531 = select i1 %compare.high.equal526, i1 %compare.low530, i1 %compare.high.order527, !dbg !23
  %516 = select i1 %compare.result531, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base532 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local533 = or i32 %compare.local.base532, 1, !dbg !23
  %compare.low534 = icmp ugt i32 %compare.low.base528, %compare.local533, !dbg !23
  %compare.result535 = select i1 %compare.high.equal526, i1 %compare.low534, i1 %compare.high.order527, !dbg !23
  %517 = select i1 %compare.result535, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base536 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local537 = or i32 %compare.local.base536, 2, !dbg !23
  %compare.low538 = icmp ugt i32 %compare.low.base528, %compare.local537, !dbg !23
  %compare.result539 = select i1 %compare.high.equal526, i1 %compare.low538, i1 %compare.high.order527, !dbg !23
  %518 = select i1 %compare.result539, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base540 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local541 = or i32 %compare.local.base540, 3, !dbg !23
  %compare.low542 = icmp ugt i32 %compare.low.base528, %compare.local541, !dbg !23
  %compare.result543 = select i1 %compare.high.equal526, i1 %compare.low542, i1 %compare.high.order527, !dbg !23
  %519 = select i1 %compare.result543, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base544 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local545 = or i32 %compare.local.base544, 4, !dbg !23
  %compare.low546 = icmp ugt i32 %compare.low.base528, %compare.local545, !dbg !23
  %compare.result547 = select i1 %compare.high.equal526, i1 %compare.low546, i1 %compare.high.order527, !dbg !23
  %520 = select i1 %compare.result547, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base548 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local549 = or i32 %compare.local.base548, 5, !dbg !23
  %compare.low550 = icmp ugt i32 %compare.low.base528, %compare.local549, !dbg !23
  %compare.result551 = select i1 %compare.high.equal526, i1 %compare.low550, i1 %compare.high.order527, !dbg !23
  %521 = select i1 %compare.result551, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base552 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local553 = or i32 %compare.local.base552, 6, !dbg !23
  %compare.low554 = icmp ugt i32 %compare.low.base528, %compare.local553, !dbg !23
  %compare.result555 = select i1 %compare.high.equal526, i1 %compare.low554, i1 %compare.high.order527, !dbg !23
  %522 = select i1 %compare.result555, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base556 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local557 = or i32 %compare.local.base556, 7, !dbg !23
  %compare.low558 = icmp ugt i32 %compare.low.base528, %compare.local557, !dbg !23
  %compare.result559 = select i1 %compare.high.equal526, i1 %compare.low558, i1 %compare.high.order527, !dbg !23
  %523 = select i1 %compare.result559, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base560 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local561 = or i32 %compare.local.base560, 8, !dbg !23
  %compare.low562 = icmp ugt i32 %compare.low.base528, %compare.local561, !dbg !23
  %compare.result563 = select i1 %compare.high.equal526, i1 %compare.low562, i1 %compare.high.order527, !dbg !23
  %524 = select i1 %compare.result563, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base564 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local565 = or i32 %compare.local.base564, 9, !dbg !23
  %compare.low566 = icmp ugt i32 %compare.low.base528, %compare.local565, !dbg !23
  %compare.result567 = select i1 %compare.high.equal526, i1 %compare.low566, i1 %compare.high.order527, !dbg !23
  %525 = select i1 %compare.result567, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base568 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local569 = or i32 %compare.local.base568, 10, !dbg !23
  %compare.low570 = icmp ugt i32 %compare.low.base528, %compare.local569, !dbg !23
  %compare.result571 = select i1 %compare.high.equal526, i1 %compare.low570, i1 %compare.high.order527, !dbg !23
  %526 = select i1 %compare.result571, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base572 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local573 = or i32 %compare.local.base572, 11, !dbg !23
  %compare.low574 = icmp ugt i32 %compare.low.base528, %compare.local573, !dbg !23
  %compare.result575 = select i1 %compare.high.equal526, i1 %compare.low574, i1 %compare.high.order527, !dbg !23
  %527 = select i1 %compare.result575, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base576 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local577 = or i32 %compare.local.base576, 12, !dbg !23
  %compare.low578 = icmp ugt i32 %compare.low.base528, %compare.local577, !dbg !23
  %compare.result579 = select i1 %compare.high.equal526, i1 %compare.low578, i1 %compare.high.order527, !dbg !23
  %528 = select i1 %compare.result579, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base580 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local581 = or i32 %compare.local.base580, 13, !dbg !23
  %compare.low582 = icmp ugt i32 %compare.low.base528, %compare.local581, !dbg !23
  %compare.result583 = select i1 %compare.high.equal526, i1 %compare.low582, i1 %compare.high.order527, !dbg !23
  %529 = select i1 %compare.result583, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base584 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local585 = or i32 %compare.local.base584, 14, !dbg !23
  %compare.low586 = icmp ugt i32 %compare.low.base528, %compare.local585, !dbg !23
  %compare.result587 = select i1 %compare.high.equal526, i1 %compare.low586, i1 %compare.high.order527, !dbg !23
  %530 = select i1 %compare.result587, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base588 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base529, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local589 = or i32 %compare.local.base588, 15, !dbg !23
  %compare.low590 = icmp ugt i32 %compare.low.base528, %compare.local589, !dbg !23
  %compare.result591 = select i1 %compare.high.equal526, i1 %compare.low590, i1 %compare.high.order527, !dbg !23
  %531 = select i1 %compare.result591, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %532 = insertelement <16 x float> poison, float %497, i64 0, !dbg !13
  %533 = insertelement <16 x float> %532, float %498, i64 1, !dbg !13
  %534 = insertelement <16 x float> %533, float %499, i64 2, !dbg !13
  %535 = insertelement <16 x float> %534, float %500, i64 3, !dbg !13
  %536 = insertelement <16 x float> %535, float %501, i64 4, !dbg !13
  %537 = insertelement <16 x float> %536, float %502, i64 5, !dbg !13
  %538 = insertelement <16 x float> %537, float %503, i64 6, !dbg !13
  %539 = insertelement <16 x float> %538, float %504, i64 7, !dbg !13
  %540 = insertelement <16 x float> %539, float %505, i64 8, !dbg !13
  %541 = insertelement <16 x float> %540, float %506, i64 9, !dbg !13
  %542 = insertelement <16 x float> %541, float %507, i64 10, !dbg !13
  %543 = insertelement <16 x float> %542, float %508, i64 11, !dbg !13
  %544 = insertelement <16 x float> %543, float %509, i64 12, !dbg !13
  %545 = insertelement <16 x float> %544, float %510, i64 13, !dbg !13
  %546 = insertelement <16 x float> %545, float %511, i64 14, !dbg !13
  %547 = insertelement <16 x float> %546, float %512, i64 15, !dbg !13
  %548 = insertvalue [2 x <16 x float>] poison, <16 x float> %547, 0, !dbg !13
  %549 = insertelement <16 x float> poison, float %516, i64 0, !dbg !13
  %550 = insertelement <16 x float> %549, float %517, i64 1, !dbg !13
  %551 = insertelement <16 x float> %550, float %518, i64 2, !dbg !13
  %552 = insertelement <16 x float> %551, float %519, i64 3, !dbg !13
  %553 = insertelement <16 x float> %552, float %520, i64 4, !dbg !13
  %554 = insertelement <16 x float> %553, float %521, i64 5, !dbg !13
  %555 = insertelement <16 x float> %554, float %522, i64 6, !dbg !13
  %556 = insertelement <16 x float> %555, float %523, i64 7, !dbg !13
  %557 = insertelement <16 x float> %556, float %524, i64 8, !dbg !13
  %558 = insertelement <16 x float> %557, float %525, i64 9, !dbg !13
  %559 = insertelement <16 x float> %558, float %526, i64 10, !dbg !13
  %560 = insertelement <16 x float> %559, float %527, i64 11, !dbg !13
  %561 = insertelement <16 x float> %560, float %528, i64 12, !dbg !13
  %562 = insertelement <16 x float> %561, float %529, i64 13, !dbg !13
  %563 = insertelement <16 x float> %562, float %530, i64 14, !dbg !13
  %564 = insertelement <16 x float> %563, float %531, i64 15, !dbg !13
  %565 = insertvalue [2 x <16 x float>] %548, <16 x float> %564, 1, !dbg !13
  br label %566, !dbg !13

566:                                              ; preds = %485, %486
  %567 = phi [2 x <16 x float>] [ %565, %486 ], [ zeroinitializer, %485 ], !dbg !13
  br label %568, !dbg !13

568:                                              ; preds = %566
  %569 = extractvalue [2 x <16 x float>] %567, 0, !dbg !25
  %570 = shufflevector <16 x float> %569, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %571 = fadd <4 x float> %454, %570, !dbg !25
  %572 = extractelement <4 x float> %571, i64 0, !dbg !25
  %573 = extractelement <4 x float> %571, i64 1, !dbg !25
  %574 = extractelement <4 x float> %571, i64 2, !dbg !25
  %575 = extractelement <4 x float> %571, i64 3, !dbg !25
  %576 = shufflevector <16 x float> %569, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %577 = fadd <4 x float> %458, %576, !dbg !25
  %578 = extractelement <4 x float> %577, i64 0, !dbg !25
  %579 = extractelement <4 x float> %577, i64 1, !dbg !25
  %580 = extractelement <4 x float> %577, i64 2, !dbg !25
  %581 = extractelement <4 x float> %577, i64 3, !dbg !25
  %582 = shufflevector <16 x float> %569, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %583 = fadd <4 x float> %462, %582, !dbg !25
  %584 = extractelement <4 x float> %583, i64 0, !dbg !25
  %585 = extractelement <4 x float> %583, i64 1, !dbg !25
  %586 = extractelement <4 x float> %583, i64 2, !dbg !25
  %587 = extractelement <4 x float> %583, i64 3, !dbg !25
  %588 = shufflevector <16 x float> %569, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %589 = fadd <4 x float> %466, %588, !dbg !25
  %590 = extractelement <4 x float> %589, i64 0, !dbg !25
  %591 = extractelement <4 x float> %589, i64 1, !dbg !25
  %592 = extractelement <4 x float> %589, i64 2, !dbg !25
  %593 = extractelement <4 x float> %589, i64 3, !dbg !25
  %594 = extractvalue [2 x <16 x float>] %567, 1, !dbg !25
  %595 = shufflevector <16 x float> %594, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %596 = fadd <4 x float> %470, %595, !dbg !25
  %597 = extractelement <4 x float> %596, i64 0, !dbg !25
  %598 = extractelement <4 x float> %596, i64 1, !dbg !25
  %599 = extractelement <4 x float> %596, i64 2, !dbg !25
  %600 = extractelement <4 x float> %596, i64 3, !dbg !25
  %601 = shufflevector <16 x float> %594, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %602 = fadd <4 x float> %474, %601, !dbg !25
  %603 = extractelement <4 x float> %602, i64 0, !dbg !25
  %604 = extractelement <4 x float> %602, i64 1, !dbg !25
  %605 = extractelement <4 x float> %602, i64 2, !dbg !25
  %606 = extractelement <4 x float> %602, i64 3, !dbg !25
  %607 = shufflevector <16 x float> %594, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %608 = fadd <4 x float> %478, %607, !dbg !25
  %609 = extractelement <4 x float> %608, i64 0, !dbg !25
  %610 = extractelement <4 x float> %608, i64 1, !dbg !25
  %611 = extractelement <4 x float> %608, i64 2, !dbg !25
  %612 = extractelement <4 x float> %608, i64 3, !dbg !25
  %613 = shufflevector <16 x float> %594, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %614 = fadd <4 x float> %483, %613, !dbg !25
  %615 = extractelement <4 x float> %614, i64 0, !dbg !25
  %616 = extractelement <4 x float> %614, i64 1, !dbg !25
  %617 = extractelement <4 x float> %614, i64 2, !dbg !25
  %618 = extractelement <4 x float> %614, i64 3, !dbg !25
  %619 = call float @llvm.maxnum.f32(float %572, float 0xFFF0000000000000), !dbg !26
  %620 = call float @llvm.maxnum.f32(float %619, float %573), !dbg !26
  %621 = call float @llvm.maxnum.f32(float %620, float %574), !dbg !26
  %622 = call float @llvm.maxnum.f32(float %621, float %575), !dbg !26
  %623 = call float @llvm.maxnum.f32(float %622, float %578), !dbg !26
  %624 = call float @llvm.maxnum.f32(float %623, float %579), !dbg !26
  %625 = call float @llvm.maxnum.f32(float %624, float %580), !dbg !26
  %626 = call float @llvm.maxnum.f32(float %625, float %581), !dbg !26
  %627 = call float @llvm.maxnum.f32(float %626, float %584), !dbg !26
  %628 = call float @llvm.maxnum.f32(float %627, float %585), !dbg !26
  %629 = call float @llvm.maxnum.f32(float %628, float %586), !dbg !26
  %630 = call float @llvm.maxnum.f32(float %629, float %587), !dbg !26
  %631 = call float @llvm.maxnum.f32(float %630, float %590), !dbg !26
  %632 = call float @llvm.maxnum.f32(float %631, float %591), !dbg !26
  %633 = call float @llvm.maxnum.f32(float %632, float %592), !dbg !26
  %634 = call float @llvm.maxnum.f32(float %633, float %593), !dbg !26
  %635 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %636 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %635), !dbg !26
  %637 = add nuw nsw i32 %636, 64, !dbg !26
  %638 = xor i32 %636, 16, !dbg !26
  %639 = and i32 %637, 192, !dbg !26
  %640 = icmp samesign ult i32 %638, %639, !dbg !26
  %641 = select i1 %640, i32 %638, i32 %636, !dbg !26
  %642 = shl nuw nsw i32 %641, 2, !dbg !26
  %643 = bitcast float %634 to i32, !dbg !26
  %644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %642, i32 %643), !dbg !26
  %645 = bitcast i32 %644 to float, !dbg !26
  %646 = call float @llvm.maxnum.f32(float %634, float %645), !dbg !26
  %647 = xor i32 %636, 32, !dbg !26
  %648 = icmp samesign ult i32 %647, %639, !dbg !26
  %649 = select i1 %648, i32 %647, i32 %636, !dbg !26
  %650 = shl nuw nsw i32 %649, 2, !dbg !26
  %651 = bitcast float %646 to i32, !dbg !26
  %652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %650, i32 %651), !dbg !26
  %653 = bitcast i32 %652 to float, !dbg !26
  %654 = call float @llvm.maxnum.f32(float %646, float %653), !dbg !26
  %655 = extractvalue [2 x <1 x float>] %46, 0, !dbg !26
  %656 = insertelement <1 x float> poison, float %654, i64 0, !dbg !26
  %657 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %656, <1 x float> %655), !dbg !26
  %658 = call float @llvm.maxnum.f32(float %597, float 0xFFF0000000000000), !dbg !26
  %659 = call float @llvm.maxnum.f32(float %658, float %598), !dbg !26
  %660 = call float @llvm.maxnum.f32(float %659, float %599), !dbg !26
  %661 = call float @llvm.maxnum.f32(float %660, float %600), !dbg !26
  %662 = call float @llvm.maxnum.f32(float %661, float %603), !dbg !26
  %663 = call float @llvm.maxnum.f32(float %662, float %604), !dbg !26
  %664 = call float @llvm.maxnum.f32(float %663, float %605), !dbg !26
  %665 = call float @llvm.maxnum.f32(float %664, float %606), !dbg !26
  %666 = call float @llvm.maxnum.f32(float %665, float %609), !dbg !26
  %667 = call float @llvm.maxnum.f32(float %666, float %610), !dbg !26
  %668 = call float @llvm.maxnum.f32(float %667, float %611), !dbg !26
  %669 = call float @llvm.maxnum.f32(float %668, float %612), !dbg !26
  %670 = call float @llvm.maxnum.f32(float %669, float %615), !dbg !26
  %671 = call float @llvm.maxnum.f32(float %670, float %616), !dbg !26
  %672 = call float @llvm.maxnum.f32(float %671, float %617), !dbg !26
  %673 = call float @llvm.maxnum.f32(float %672, float %618), !dbg !26
  %674 = bitcast float %673 to i32, !dbg !26
  %675 = call i32 @llvm.amdgcn.ds.bpermute(i32 %642, i32 %674), !dbg !26
  %676 = bitcast i32 %675 to float, !dbg !26
  %677 = call float @llvm.maxnum.f32(float %673, float %676), !dbg !26
  %678 = bitcast float %677 to i32, !dbg !26
  %679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %650, i32 %678), !dbg !26
  %680 = bitcast i32 %679 to float, !dbg !26
  %681 = call float @llvm.maxnum.f32(float %677, float %680), !dbg !26
  %682 = extractvalue [2 x <1 x float>] %46, 1, !dbg !26
  %683 = insertelement <1 x float> poison, float %681, i64 0, !dbg !26
  %684 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %683, <1 x float> %682), !dbg !26
  %685 = shufflevector <1 x float> %657, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %686 = fsub <4 x float> %571, %685, !dbg !27
  %687 = fsub <4 x float> %577, %685, !dbg !27
  %688 = fsub <4 x float> %583, %685, !dbg !27
  %689 = fsub <4 x float> %589, %685, !dbg !27
  %690 = shufflevector <1 x float> %684, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %691 = fsub <4 x float> %596, %690, !dbg !27
  %692 = fsub <4 x float> %602, %690, !dbg !27
  %693 = fsub <4 x float> %608, %690, !dbg !27
  %694 = fsub <4 x float> %614, %690, !dbg !27
  %695 = fsub <1 x float> %655, %657, !dbg !28
  %696 = extractelement <1 x float> %695, i64 0, !dbg !29
  %697 = call float @llvm.amdgcn.exp2.f32(float %696), !dbg !29
  %698 = insertelement <4 x float> poison, float %697, i64 0, !dbg !34
  %699 = shufflevector <4 x float> %698, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %700 = extractvalue [2 x <1 x float>] %45, 0, !dbg !30
  %701 = extractelement <1 x float> %700, i64 0, !dbg !30
  %702 = fmul float %701, %697, !dbg !30
  %703 = fsub <1 x float> %682, %684, !dbg !28
  %704 = extractelement <1 x float> %703, i64 0, !dbg !29
  %705 = call float @llvm.amdgcn.exp2.f32(float %704), !dbg !29
  %706 = insertelement <4 x float> poison, float %705, i64 0, !dbg !34
  %707 = shufflevector <4 x float> %706, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %708 = extractvalue [2 x <1 x float>] %45, 1, !dbg !30
  %709 = extractelement <1 x float> %708, i64 0, !dbg !30
  %710 = fmul float %709, %705, !dbg !30
  %711 = extractelement <4 x float> %686, i64 0, !dbg !31
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !31
  %713 = insertelement <4 x float> poison, float %712, i64 0, !dbg !33
  %714 = extractelement <4 x float> %686, i64 1, !dbg !31
  %715 = call float @llvm.amdgcn.exp2.f32(float %714), !dbg !31
  %716 = insertelement <4 x float> %713, float %715, i64 1, !dbg !33
  %717 = fadd float %712, %715, !dbg !32
  %718 = extractelement <4 x float> %686, i64 2, !dbg !31
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !31
  %720 = insertelement <4 x float> %716, float %719, i64 2, !dbg !33
  %721 = fadd float %717, %719, !dbg !32
  %722 = extractelement <4 x float> %686, i64 3, !dbg !31
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !31
  %724 = insertelement <4 x float> %720, float %723, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %724) #9, !dbg !33, !frisk.register_lifetimes !1
  %725 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %725) #9, !dbg !32, !frisk.register_lifetimes !1
  %726 = fadd float %721, %723, !dbg !32
  %727 = extractelement <4 x float> %687, i64 0, !dbg !31
  %728 = call float @llvm.amdgcn.exp2.f32(float %727), !dbg !31
  %729 = insertelement <4 x float> poison, float %728, i64 0, !dbg !33
  %730 = fadd float %726, %728, !dbg !32
  %731 = extractelement <4 x float> %687, i64 1, !dbg !31
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !31
  %733 = insertelement <4 x float> %729, float %732, i64 1, !dbg !33
  %734 = fadd float %730, %732, !dbg !32
  %735 = extractelement <4 x float> %687, i64 2, !dbg !31
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !31
  %737 = insertelement <4 x float> %733, float %736, i64 2, !dbg !33
  %738 = fadd float %734, %736, !dbg !32
  %739 = extractelement <4 x float> %687, i64 3, !dbg !31
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !31
  %741 = insertelement <4 x float> %737, float %740, i64 3, !dbg !33
  %math.result62 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %741) #9, !dbg !33, !frisk.register_lifetimes !1
  %742 = fptrunc <4 x float> %math.result62 to <4 x half>, !dbg !33
  %math.packed63 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %742) #9, !dbg !32, !frisk.register_lifetimes !1
  %743 = fadd float %738, %740, !dbg !32
  %744 = extractelement <4 x float> %688, i64 0, !dbg !31
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !31
  %746 = insertelement <4 x float> poison, float %745, i64 0, !dbg !33
  %747 = fadd float %743, %745, !dbg !32
  %748 = extractelement <4 x float> %688, i64 1, !dbg !31
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !31
  %750 = insertelement <4 x float> %746, float %749, i64 1, !dbg !33
  %751 = fadd float %747, %749, !dbg !32
  %752 = extractelement <4 x float> %688, i64 2, !dbg !31
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !31
  %754 = insertelement <4 x float> %750, float %753, i64 2, !dbg !33
  %755 = fadd float %751, %753, !dbg !32
  %756 = extractelement <4 x float> %688, i64 3, !dbg !31
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !31
  %758 = insertelement <4 x float> %754, float %757, i64 3, !dbg !33
  %math.result64 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %758) #9, !dbg !33, !frisk.register_lifetimes !1
  %759 = fptrunc <4 x float> %math.result64 to <4 x half>, !dbg !33
  %math.packed65 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %759) #9, !dbg !32, !frisk.register_lifetimes !1
  %760 = fadd float %755, %757, !dbg !32
  %761 = extractelement <4 x float> %689, i64 0, !dbg !31
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !31
  %763 = insertelement <4 x float> poison, float %762, i64 0, !dbg !33
  %764 = fadd float %760, %762, !dbg !32
  %765 = extractelement <4 x float> %689, i64 1, !dbg !31
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !31
  %767 = insertelement <4 x float> %763, float %766, i64 1, !dbg !33
  %768 = fadd float %764, %766, !dbg !32
  %769 = extractelement <4 x float> %689, i64 2, !dbg !31
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !31
  %771 = insertelement <4 x float> %767, float %770, i64 2, !dbg !33
  %772 = fadd float %768, %770, !dbg !32
  %773 = extractelement <4 x float> %689, i64 3, !dbg !31
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !31
  %775 = insertelement <4 x float> %771, float %774, i64 3, !dbg !33
  %math.result66 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #9, !dbg !33, !frisk.register_lifetimes !1
  %776 = fptrunc <4 x float> %math.result66 to <4 x half>, !dbg !33
  %math.packed67 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #9, !dbg !32, !frisk.register_lifetimes !1
  %777 = fadd float %772, %774, !dbg !32
  %778 = extractelement <4 x float> %691, i64 0, !dbg !31
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !31
  %780 = insertelement <4 x float> poison, float %779, i64 0, !dbg !33
  %781 = extractelement <4 x float> %691, i64 1, !dbg !31
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !31
  %783 = insertelement <4 x float> %780, float %782, i64 1, !dbg !33
  %784 = fadd float %779, %782, !dbg !32
  %785 = extractelement <4 x float> %691, i64 2, !dbg !31
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !31
  %787 = insertelement <4 x float> %783, float %786, i64 2, !dbg !33
  %788 = fadd float %784, %786, !dbg !32
  %789 = extractelement <4 x float> %691, i64 3, !dbg !31
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !31
  %791 = insertelement <4 x float> %787, float %790, i64 3, !dbg !33
  %math.result68 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %791) #9, !dbg !33, !frisk.register_lifetimes !1
  %792 = fptrunc <4 x float> %math.result68 to <4 x half>, !dbg !33
  %math.packed69 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %792) #9, !dbg !32, !frisk.register_lifetimes !1
  %793 = fadd float %788, %790, !dbg !32
  %794 = extractelement <4 x float> %692, i64 0, !dbg !31
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !31
  %796 = insertelement <4 x float> poison, float %795, i64 0, !dbg !33
  %797 = fadd float %793, %795, !dbg !32
  %798 = extractelement <4 x float> %692, i64 1, !dbg !31
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !31
  %800 = insertelement <4 x float> %796, float %799, i64 1, !dbg !33
  %801 = fadd float %797, %799, !dbg !32
  %802 = extractelement <4 x float> %692, i64 2, !dbg !31
  %803 = call float @llvm.amdgcn.exp2.f32(float %802), !dbg !31
  %804 = insertelement <4 x float> %800, float %803, i64 2, !dbg !33
  %805 = fadd float %801, %803, !dbg !32
  %806 = extractelement <4 x float> %692, i64 3, !dbg !31
  %807 = call float @llvm.amdgcn.exp2.f32(float %806), !dbg !31
  %808 = insertelement <4 x float> %804, float %807, i64 3, !dbg !33
  %math.result70 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %808) #9, !dbg !33, !frisk.register_lifetimes !1
  %809 = fptrunc <4 x float> %math.result70 to <4 x half>, !dbg !33
  %math.packed71 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %809) #9, !dbg !32, !frisk.register_lifetimes !1
  %810 = fadd float %805, %807, !dbg !32
  %811 = extractelement <4 x float> %693, i64 0, !dbg !31
  %812 = call float @llvm.amdgcn.exp2.f32(float %811), !dbg !31
  %813 = insertelement <4 x float> poison, float %812, i64 0, !dbg !33
  %814 = fadd float %810, %812, !dbg !32
  %815 = extractelement <4 x float> %693, i64 1, !dbg !31
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !31
  %817 = insertelement <4 x float> %813, float %816, i64 1, !dbg !33
  %818 = fadd float %814, %816, !dbg !32
  %819 = extractelement <4 x float> %693, i64 2, !dbg !31
  %820 = call float @llvm.amdgcn.exp2.f32(float %819), !dbg !31
  %821 = insertelement <4 x float> %817, float %820, i64 2, !dbg !33
  %822 = fadd float %818, %820, !dbg !32
  %823 = extractelement <4 x float> %693, i64 3, !dbg !31
  %824 = call float @llvm.amdgcn.exp2.f32(float %823), !dbg !31
  %825 = insertelement <4 x float> %821, float %824, i64 3, !dbg !33
  %math.result72 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %825) #9, !dbg !33, !frisk.register_lifetimes !1
  %826 = fptrunc <4 x float> %math.result72 to <4 x half>, !dbg !33
  %math.packed73 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %826) #9, !dbg !32, !frisk.register_lifetimes !1
  %827 = fadd float %822, %824, !dbg !32
  %828 = extractelement <4 x float> %694, i64 0, !dbg !31
  %829 = call float @llvm.amdgcn.exp2.f32(float %828), !dbg !31
  %830 = insertelement <4 x float> poison, float %829, i64 0, !dbg !33
  %831 = fadd float %827, %829, !dbg !32
  %832 = extractelement <4 x float> %694, i64 1, !dbg !31
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !31
  %834 = insertelement <4 x float> %830, float %833, i64 1, !dbg !33
  %835 = fadd float %831, %833, !dbg !32
  %836 = extractelement <4 x float> %694, i64 2, !dbg !31
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !31
  %838 = insertelement <4 x float> %834, float %837, i64 2, !dbg !33
  %839 = fadd float %835, %837, !dbg !32
  %840 = extractelement <4 x float> %694, i64 3, !dbg !31
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !31
  %842 = insertelement <4 x float> %838, float %841, i64 3, !dbg !33
  %math.result74 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %842) #9, !dbg !33, !frisk.register_lifetimes !1
  %843 = fptrunc <4 x float> %math.result74 to <4 x half>, !dbg !33
  %math.packed75 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %843) #9, !dbg !32, !frisk.register_lifetimes !1
  %844 = fadd float %839, %841, !dbg !32
  %845 = bitcast float %777 to i32, !dbg !32
  %846 = call i32 @llvm.amdgcn.ds.bpermute(i32 %642, i32 %845), !dbg !32
  %847 = bitcast i32 %846 to float, !dbg !32
  %848 = fadd float %777, %847, !dbg !32
  %849 = bitcast float %848 to i32, !dbg !32
  %850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %650, i32 %849), !dbg !32
  %851 = bitcast i32 %850 to float, !dbg !32
  %852 = fadd float %848, %851, !dbg !32
  %853 = fadd float %852, %702, !dbg !32
  %854 = insertelement <1 x float> poison, float %853, i64 0, !dbg !7
  %855 = insertvalue [2 x <1 x float>] poison, <1 x float> %854, 0, !dbg !7
  %856 = bitcast float %844 to i32, !dbg !32
  %857 = call i32 @llvm.amdgcn.ds.bpermute(i32 %642, i32 %856), !dbg !32
  %858 = bitcast i32 %857 to float, !dbg !32
  %859 = fadd float %844, %858, !dbg !32
  %860 = bitcast float %859 to i32, !dbg !32
  %861 = call i32 @llvm.amdgcn.ds.bpermute(i32 %650, i32 %860), !dbg !32
  %862 = bitcast i32 %861 to float, !dbg !32
  %863 = fadd float %859, %862, !dbg !32
  %864 = fadd float %863, %710, !dbg !32
  %865 = insertelement <1 x float> poison, float %864, i64 0, !dbg !7
  %866 = insertvalue [2 x <1 x float>] %855, <1 x float> %865, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread203 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local204 = zext nneg i32 %address.thread203 to i64, !dbg !12
  %address.local205 = and i64 %address.local204, 60, !dbg !14
  %address.local206 = add nuw nsw i64 %address.local205, %address.local204, !dbg !14
  %address.local207 = shl nuw nsw i64 %address.local206, 3, !dbg !14
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local207, !dbg !14
  %address.local209 = and i64 %address.local204, 63, !dbg !12
  %address.local210 = lshr i64 %address.local209, 4, !dbg !14
  %address.local211 = mul nuw nsw i64 %address.local210, 384, !dbg !14
  %address.local212 = getelementptr [2 x i8], ptr addrspace(3) %address.local208, i64 %address.local211, !dbg !14
  %address.local213 = lshr i64 %address.local204, 2, !dbg !16
  %address.local214 = mul nsw i64 %address.local213, -32, !dbg !14
  %address.local215 = getelementptr [2 x i8], ptr addrspace(3) %address.local212, i64 %address.local214, !dbg !14
  %867 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local215, i32 0) #0, !dbg !14
  %868 = bitcast <4 x i32> %867 to <8 x half>, !dbg !14
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local207, !dbg !14
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local216, i64 %address.local211, !dbg !14
  %address.local218 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local214, !dbg !14
  %address.local219 = getelementptr i8, ptr addrspace(3) %address.local218, i64 256, !dbg !14
  %869 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local219, i32 0) #0, !dbg !14
  %870 = bitcast <4 x i32> %869 to <8 x half>, !dbg !14
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local207, !dbg !14
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local211, !dbg !14
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) %address.local221, i64 %address.local214, !dbg !14
  %address.local223 = getelementptr i8, ptr addrspace(3) %address.local222, i64 512, !dbg !14
  %871 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local223, i32 0) #0, !dbg !14
  %872 = bitcast <4 x i32> %871 to <8 x half>, !dbg !14
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local207, !dbg !14
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i64 %address.local211, !dbg !14
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local214, !dbg !14
  %address.local227 = getelementptr i8, ptr addrspace(3) %address.local226, i64 768, !dbg !14
  %873 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local227, i32 0) #0, !dbg !14
  %874 = bitcast <4 x i32> %873 to <8 x half>, !dbg !14
  %875 = add nsw i64 %83, %8, !dbg !17
  %876 = shl i64 %875, 19, !dbg !17
  %877 = and i64 %876, 16252928, !dbg !17
  %878 = add i64 %877, %89, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %879 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %878, !dbg !17
  %880 = getelementptr i8, ptr addrspace(1) %879, i64 64, !dbg !17
  %address.local228 = shl nuw nsw i64 %address.local204, 3, !dbg !11
  %address.local229 = and i64 %address.local228, 1536, !dbg !15
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local229, !dbg !17
  %address.local231 = ptrtoint ptr addrspace(3) %address.local230 to i64, !dbg !17
  %address.local232 = trunc i64 %address.local231 to i32, !dbg !17
  %881 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local232), !dbg !17
  %882 = zext i32 %881 to i64, !dbg !17
  %883 = inttoptr i64 %882 to ptr addrspace(3), !dbg !17
  %address.local233 = shl nuw nsw i64 %address.local204, 5, !dbg !17
  %address.local234 = and i64 %address.local233, 8064, !dbg !17
  %address.local235 = shl nuw nsw i64 %address.local204, 3, !dbg !17
  %address.local236 = and i64 %address.local235, 24, !dbg !17
  %address.local237 = or disjoint i64 %address.local234, %address.local236, !dbg !17
  %address.local238 = getelementptr [2 x i8], ptr addrspace(1) %880, i64 %address.local237, !dbg !17
  %884 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %885 = ptrtoint ptr addrspace(1) %address.local238 to i64, !dbg !17
  %886 = sub i64 %885, %884, !dbg !17
  %dma.buffer.offset595 = trunc i64 %886 to i32, !dbg !17
  %887 = trunc i64 %884 to i32, !dbg !17
  %888 = insertelement <4 x i32> poison, i32 %887, i64 0, !dbg !17
  %889 = lshr i64 %884, 32, !dbg !17
  %890 = trunc i64 %889 to i32, !dbg !17
  %891 = insertelement <4 x i32> %888, i32 %890, i64 1, !dbg !17
  %892 = insertelement <4 x i32> %891, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc596 = insertelement <4 x i32> %892, i32 131072, i64 3, !dbg !17
  %893 = ptrtoint ptr addrspace(3) %883 to i32, !dbg !17
  %dma.buffer.lds597 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %893), !dbg !17
  %894 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds597, i32 %dma.buffer.offset595, <4 x i32> %dma.buffer.rsrc596) #10, !dbg !17
  %895 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %896 = extractvalue [2 x <8 x float>] %44, 0, !dbg !34
  %897 = shufflevector <8 x float> %896, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %898 = fmul <4 x float> %897, %699, !dbg !34
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %895, <4 x float> %898) #9, !dbg !14
  %900 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %900, <4 x float> %899) #9, !dbg !14
  %902 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %902, <4 x float> %901) #9, !dbg !14
  %904 = shufflevector <8 x half> %874, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %904, <4 x float> %903) #9, !dbg !14
  %906 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %907 = shufflevector <8 x float> %896, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %908 = fmul <4 x float> %907, %699, !dbg !34
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %906, <4 x float> %908) #9, !dbg !14
  %910 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %910, <4 x float> %909) #9, !dbg !14
  %912 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %912, <4 x float> %911) #9, !dbg !14
  %914 = shufflevector <8 x half> %874, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %914, <4 x float> %913) #9, !dbg !14
  %916 = extractvalue [2 x <8 x float>] %44, 1, !dbg !34
  %917 = shufflevector <8 x float> %916, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %918 = fmul <4 x float> %917, %707, !dbg !34
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %895, <4 x float> %918) #9, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %900, <4 x float> %919) #9, !dbg !14
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %902, <4 x float> %920) #9, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %904, <4 x float> %921) #9, !dbg !14
  %923 = shufflevector <8 x float> %916, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %924 = fmul <4 x float> %923, %707, !dbg !34
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %906, <4 x float> %924) #9, !dbg !14
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %910, <4 x float> %925) #9, !dbg !14
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %912, <4 x float> %926) #9, !dbg !14
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %914, <4 x float> %927) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %929 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local215, i32 0) #0, !dbg !14
  %930 = bitcast <4 x i32> %929 to <8 x half>, !dbg !14
  %931 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local219, i32 0) #0, !dbg !14
  %932 = bitcast <4 x i32> %931 to <8 x half>, !dbg !14
  %933 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local223, i32 0) #0, !dbg !14
  %934 = bitcast <4 x i32> %933 to <8 x half>, !dbg !14
  %935 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local227, i32 0) #0, !dbg !14
  %936 = bitcast <4 x i32> %935 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %937 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %878, !dbg !17
  %938 = getelementptr i8, ptr addrspace(1) %937, i64 128, !dbg !17
  %939 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local232), !dbg !17
  %940 = zext i32 %939 to i64, !dbg !17
  %941 = inttoptr i64 %940 to ptr addrspace(3), !dbg !17
  %address.local239 = getelementptr [2 x i8], ptr addrspace(1) %938, i64 %address.local237, !dbg !17
  %942 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %943 = ptrtoint ptr addrspace(1) %address.local239 to i64, !dbg !17
  %944 = sub i64 %943, %942, !dbg !17
  %dma.buffer.offset592 = trunc i64 %944 to i32, !dbg !17
  %945 = trunc i64 %942 to i32, !dbg !17
  %946 = insertelement <4 x i32> poison, i32 %945, i64 0, !dbg !17
  %947 = lshr i64 %942, 32, !dbg !17
  %948 = trunc i64 %947 to i32, !dbg !17
  %949 = insertelement <4 x i32> %946, i32 %948, i64 1, !dbg !17
  %950 = insertelement <4 x i32> %949, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc593 = insertelement <4 x i32> %950, i32 131072, i64 3, !dbg !17
  %951 = ptrtoint ptr addrspace(3) %941 to i32, !dbg !17
  %dma.buffer.lds594 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %951), !dbg !17
  %952 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds594, i32 %dma.buffer.offset592, <4 x i32> %dma.buffer.rsrc593) #10, !dbg !17
  %953 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %954 = extractvalue [2 x <8 x float>] %47, 0, !dbg !34
  %955 = shufflevector <8 x float> %954, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %956 = fmul <4 x float> %955, %699, !dbg !34
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %953, <4 x float> %956) #9, !dbg !14
  %958 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %958, <4 x float> %957) #9, !dbg !14
  %960 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %960, <4 x float> %959) #9, !dbg !14
  %962 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %962, <4 x float> %961) #9, !dbg !14
  %964 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %965 = shufflevector <8 x float> %954, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %966 = fmul <4 x float> %965, %699, !dbg !34
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %964, <4 x float> %966) #9, !dbg !14
  %968 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %968, <4 x float> %967) #9, !dbg !14
  %970 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %970, <4 x float> %969) #9, !dbg !14
  %972 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %972, <4 x float> %971) #9, !dbg !14
  %974 = extractvalue [2 x <8 x float>] %47, 1, !dbg !34
  %975 = shufflevector <8 x float> %974, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %976 = fmul <4 x float> %975, %707, !dbg !34
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %953, <4 x float> %976) #9, !dbg !14
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %958, <4 x float> %977) #9, !dbg !14
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %960, <4 x float> %978) #9, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %962, <4 x float> %979) #9, !dbg !14
  %981 = shufflevector <8 x float> %974, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %982 = fmul <4 x float> %981, %707, !dbg !34
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %964, <4 x float> %982) #9, !dbg !14
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %968, <4 x float> %983) #9, !dbg !14
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %970, <4 x float> %984) #9, !dbg !14
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %972, <4 x float> %985) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %987 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local215, i32 0) #0, !dbg !14
  %988 = bitcast <4 x i32> %987 to <8 x half>, !dbg !14
  %989 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local219, i32 0) #0, !dbg !14
  %990 = bitcast <4 x i32> %989 to <8 x half>, !dbg !14
  %991 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local223, i32 0) #0, !dbg !14
  %992 = bitcast <4 x i32> %991 to <8 x half>, !dbg !14
  %993 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local227, i32 0) #0, !dbg !14
  %994 = bitcast <4 x i32> %993 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %995 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %878, !dbg !17
  %996 = getelementptr i8, ptr addrspace(1) %995, i64 192, !dbg !17
  %997 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local232), !dbg !17
  %998 = zext i32 %997 to i64, !dbg !17
  %999 = inttoptr i64 %998 to ptr addrspace(3), !dbg !17
  %address.local240 = getelementptr [2 x i8], ptr addrspace(1) %996, i64 %address.local237, !dbg !17
  %1000 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1001 = ptrtoint ptr addrspace(1) %address.local240 to i64, !dbg !17
  %1002 = sub i64 %1001, %1000, !dbg !17
  %dma.buffer.offset = trunc i64 %1002 to i32, !dbg !17
  %1003 = trunc i64 %1000 to i32, !dbg !17
  %1004 = insertelement <4 x i32> poison, i32 %1003, i64 0, !dbg !17
  %1005 = lshr i64 %1000, 32, !dbg !17
  %1006 = trunc i64 %1005 to i32, !dbg !17
  %1007 = insertelement <4 x i32> %1004, i32 %1006, i64 1, !dbg !17
  %1008 = insertelement <4 x i32> %1007, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %1008, i32 131072, i64 3, !dbg !17
  %1009 = ptrtoint ptr addrspace(3) %999 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1009), !dbg !17
  %1010 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %1011 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1012 = extractvalue [2 x <8 x float>] %48, 0, !dbg !34
  %1013 = shufflevector <8 x float> %1012, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1014 = fmul <4 x float> %1013, %699, !dbg !34
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1011, <4 x float> %1014) #9, !dbg !14
  %1016 = shufflevector <8 x half> %990, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1017 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %1016, <4 x float> %1015) #9, !dbg !14
  %1018 = shufflevector <8 x half> %992, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %1018, <4 x float> %1017) #9, !dbg !14
  %1020 = shufflevector <8 x half> %994, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1020, <4 x float> %1019) #9, !dbg !14
  %1022 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1023 = shufflevector <8 x float> %1012, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1024 = fmul <4 x float> %1023, %699, !dbg !34
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1022, <4 x float> %1024) #9, !dbg !14
  %1026 = shufflevector <8 x half> %990, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %1026, <4 x float> %1025) #9, !dbg !14
  %1028 = shufflevector <8 x half> %992, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %1028, <4 x float> %1027) #9, !dbg !14
  %1030 = shufflevector <8 x half> %994, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1030, <4 x float> %1029) #9, !dbg !14
  %1032 = extractvalue [2 x <8 x float>] %48, 1, !dbg !34
  %1033 = shufflevector <8 x float> %1032, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1034 = fmul <4 x float> %1033, %707, !dbg !34
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1011, <4 x float> %1034) #9, !dbg !14
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1016, <4 x float> %1035) #9, !dbg !14
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1018, <4 x float> %1036) #9, !dbg !14
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %1020, <4 x float> %1037) #9, !dbg !14
  %1039 = shufflevector <8 x float> %1032, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1040 = fmul <4 x float> %1039, %707, !dbg !34
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1022, <4 x float> %1040) #9, !dbg !14
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1026, <4 x float> %1041) #9, !dbg !14
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1028, <4 x float> %1042) #9, !dbg !14
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %1030, <4 x float> %1043) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1045 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local215, i32 0) #0, !dbg !14
  %1046 = bitcast <4 x i32> %1045 to <8 x half>, !dbg !14
  %1047 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local219, i32 0) #0, !dbg !14
  %1048 = bitcast <4 x i32> %1047 to <8 x half>, !dbg !14
  %1049 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local223, i32 0) #0, !dbg !14
  %1050 = bitcast <4 x i32> %1049 to <8 x half>, !dbg !14
  %1051 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local227, i32 0) #0, !dbg !14
  %1052 = bitcast <4 x i32> %1051 to <8 x half>, !dbg !14
  %1053 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1054 = extractvalue [2 x <8 x float>] %49, 0, !dbg !34
  %1055 = shufflevector <8 x float> %1054, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1056 = fmul <4 x float> %1055, %699, !dbg !34
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1053, <4 x float> %1056) #9, !dbg !14
  %1058 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %1058, <4 x float> %1057) #9, !dbg !14
  %1060 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %1060, <4 x float> %1059) #9, !dbg !14
  %1062 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1062, <4 x float> %1061) #9, !dbg !14
  %1064 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1065 = shufflevector <8 x float> %1054, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1066 = fmul <4 x float> %1065, %699, !dbg !34
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1064, <4 x float> %1066) #9, !dbg !14
  %1068 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %1068, <4 x float> %1067) #9, !dbg !14
  %1070 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %1070, <4 x float> %1069) #9, !dbg !14
  %1072 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1072, <4 x float> %1071) #9, !dbg !14
  %1074 = extractvalue [2 x <8 x float>] %49, 1, !dbg !34
  %1075 = shufflevector <8 x float> %1074, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1076 = fmul <4 x float> %1075, %707, !dbg !34
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1053, <4 x float> %1076) #9, !dbg !14
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1058, <4 x float> %1077) #9, !dbg !14
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1060, <4 x float> %1078) #9, !dbg !14
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %1062, <4 x float> %1079) #9, !dbg !14
  %1081 = shufflevector <8 x float> %1074, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1082 = fmul <4 x float> %1081, %707, !dbg !34
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1064, <4 x float> %1082) #9, !dbg !14
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1068, <4 x float> %1083) #9, !dbg !14
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1070, <4 x float> %1084) #9, !dbg !14
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %1072, <4 x float> %1085) #9, !dbg !14
  %1087 = shufflevector <4 x float> %905, <4 x float> %915, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1088 = insertvalue [2 x <8 x float>] poison, <8 x float> %1087, 0, !dbg !7
  %1089 = shufflevector <4 x float> %922, <4 x float> %928, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1090 = insertvalue [2 x <8 x float>] %1088, <8 x float> %1089, 1, !dbg !7
  %1091 = insertvalue [2 x <1 x float>] poison, <1 x float> %657, 0, !dbg !7
  %1092 = insertvalue [2 x <1 x float>] %1091, <1 x float> %684, 1, !dbg !7
  %1093 = shufflevector <4 x float> %963, <4 x float> %973, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1094 = insertvalue [2 x <8 x float>] poison, <8 x float> %1093, 0, !dbg !7
  %1095 = shufflevector <4 x float> %980, <4 x float> %986, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1096 = insertvalue [2 x <8 x float>] %1094, <8 x float> %1095, 1, !dbg !7
  %1097 = shufflevector <4 x float> %1021, <4 x float> %1031, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1098 = insertvalue [2 x <8 x float>] poison, <8 x float> %1097, 0, !dbg !7
  %1099 = shufflevector <4 x float> %1038, <4 x float> %1044, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1100 = insertvalue [2 x <8 x float>] %1098, <8 x float> %1099, 1, !dbg !7
  %1101 = shufflevector <4 x float> %1063, <4 x float> %1073, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1102 = insertvalue [2 x <8 x float>] poison, <8 x float> %1101, 0, !dbg !7
  %1103 = shufflevector <4 x float> %1080, <4 x float> %1086, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1104 = insertvalue [2 x <8 x float>] %1102, <8 x float> %1103, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1105 = add nuw nsw i64 %43, 1, !dbg !7
  br label %42, !dbg !7

1106:                                             ; preds = %42
  %1107 = and i32 %24, 16, !dbg !7
  %.not = icmp eq i32 %1107, 0, !dbg !7
  %1108 = and i32 %24, 32, !dbg !7
  %.not2 = icmp eq i32 %1108, 0, !dbg !7
  %1109 = extractvalue [2 x <8 x float>] %49, 0, !dbg !7
  %1110 = extractelement <8 x float> %1109, i64 0, !dbg !7
  %1111 = extractelement <8 x float> %1109, i64 1, !dbg !7
  %1112 = extractelement <8 x float> %1109, i64 2, !dbg !7
  %1113 = extractelement <8 x float> %1109, i64 3, !dbg !7
  %1114 = select i1 %.not, float %1111, float %1110, !dbg !7
  %1115 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1116 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1115), !dbg !7
  %1117 = add nuw nsw i32 %1116, 64, !dbg !7
  %1118 = xor i32 %1116, 16, !dbg !7
  %1119 = and i32 %1117, 192, !dbg !7
  %1120 = icmp samesign ult i32 %1118, %1119, !dbg !7
  %1121 = select i1 %1120, i32 %1118, i32 %1116, !dbg !7
  %1122 = shl nuw nsw i32 %1121, 2, !dbg !7
  %1123 = bitcast float %1114 to i32, !dbg !7
  %1124 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1123), !dbg !7
  %1125 = bitcast i32 %1124 to float, !dbg !7
  %1126 = select i1 %.not, float %1110, float %1125, !dbg !7
  %1127 = select i1 %.not, float %1125, float %1111, !dbg !7
  %1128 = select i1 %.not, float %1113, float %1112, !dbg !7
  %1129 = bitcast float %1128 to i32, !dbg !7
  %1130 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1129), !dbg !7
  %1131 = bitcast i32 %1130 to float, !dbg !7
  %1132 = select i1 %.not, float %1112, float %1131, !dbg !7
  %1133 = select i1 %.not, float %1131, float %1113, !dbg !7
  %1134 = select i1 %.not2, float %1132, float %1126, !dbg !7
  %1135 = xor i32 %1116, 32, !dbg !7
  %1136 = icmp samesign ult i32 %1135, %1119, !dbg !7
  %1137 = select i1 %1136, i32 %1135, i32 %1116, !dbg !7
  %1138 = shl nuw nsw i32 %1137, 2, !dbg !7
  %1139 = bitcast float %1134 to i32, !dbg !7
  %1140 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1139), !dbg !7
  %1141 = bitcast i32 %1140 to float, !dbg !7
  %1142 = select i1 %.not2, float %1126, float %1141, !dbg !7
  %1143 = select i1 %.not2, float %1141, float %1132, !dbg !7
  %1144 = select i1 %.not2, float %1133, float %1127, !dbg !7
  %1145 = bitcast float %1144 to i32, !dbg !7
  %1146 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1145), !dbg !7
  %1147 = bitcast i32 %1146 to float, !dbg !7
  %1148 = select i1 %.not2, float %1127, float %1147, !dbg !7
  %1149 = select i1 %.not2, float %1147, float %1133, !dbg !7
  %1150 = extractvalue [2 x <8 x float>] %48, 0, !dbg !7
  %1151 = extractelement <8 x float> %1150, i64 0, !dbg !7
  %1152 = extractelement <8 x float> %1150, i64 1, !dbg !7
  %1153 = extractelement <8 x float> %1150, i64 2, !dbg !7
  %1154 = extractelement <8 x float> %1150, i64 3, !dbg !7
  %1155 = select i1 %.not, float %1152, float %1151, !dbg !7
  %1156 = bitcast float %1155 to i32, !dbg !7
  %1157 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1156), !dbg !7
  %1158 = bitcast i32 %1157 to float, !dbg !7
  %1159 = select i1 %.not, float %1151, float %1158, !dbg !7
  %1160 = select i1 %.not, float %1158, float %1152, !dbg !7
  %1161 = select i1 %.not, float %1154, float %1153, !dbg !7
  %1162 = bitcast float %1161 to i32, !dbg !7
  %1163 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1162), !dbg !7
  %1164 = bitcast i32 %1163 to float, !dbg !7
  %1165 = select i1 %.not, float %1153, float %1164, !dbg !7
  %1166 = select i1 %.not, float %1164, float %1154, !dbg !7
  %1167 = select i1 %.not2, float %1165, float %1159, !dbg !7
  %1168 = bitcast float %1167 to i32, !dbg !7
  %1169 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1168), !dbg !7
  %1170 = bitcast i32 %1169 to float, !dbg !7
  %1171 = select i1 %.not2, float %1159, float %1170, !dbg !7
  %1172 = select i1 %.not2, float %1170, float %1165, !dbg !7
  %1173 = select i1 %.not2, float %1166, float %1160, !dbg !7
  %1174 = bitcast float %1173 to i32, !dbg !7
  %1175 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1174), !dbg !7
  %1176 = bitcast i32 %1175 to float, !dbg !7
  %1177 = select i1 %.not2, float %1160, float %1176, !dbg !7
  %1178 = select i1 %.not2, float %1176, float %1166, !dbg !7
  %1179 = extractvalue [2 x <8 x float>] %47, 0, !dbg !7
  %1180 = extractelement <8 x float> %1179, i64 0, !dbg !7
  %1181 = extractelement <8 x float> %1179, i64 1, !dbg !7
  %1182 = extractelement <8 x float> %1179, i64 2, !dbg !7
  %1183 = extractelement <8 x float> %1179, i64 3, !dbg !7
  %1184 = select i1 %.not, float %1181, float %1180, !dbg !7
  %1185 = bitcast float %1184 to i32, !dbg !7
  %1186 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1185), !dbg !7
  %1187 = bitcast i32 %1186 to float, !dbg !7
  %1188 = select i1 %.not, float %1180, float %1187, !dbg !7
  %1189 = select i1 %.not, float %1187, float %1181, !dbg !7
  %1190 = select i1 %.not, float %1183, float %1182, !dbg !7
  %1191 = bitcast float %1190 to i32, !dbg !7
  %1192 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1191), !dbg !7
  %1193 = bitcast i32 %1192 to float, !dbg !7
  %1194 = select i1 %.not, float %1182, float %1193, !dbg !7
  %1195 = select i1 %.not, float %1193, float %1183, !dbg !7
  %1196 = select i1 %.not2, float %1194, float %1188, !dbg !7
  %1197 = bitcast float %1196 to i32, !dbg !7
  %1198 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1197), !dbg !7
  %1199 = bitcast i32 %1198 to float, !dbg !7
  %1200 = select i1 %.not2, float %1188, float %1199, !dbg !7
  %1201 = select i1 %.not2, float %1199, float %1194, !dbg !7
  %1202 = select i1 %.not2, float %1195, float %1189, !dbg !7
  %1203 = bitcast float %1202 to i32, !dbg !7
  %1204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1203), !dbg !7
  %1205 = bitcast i32 %1204 to float, !dbg !7
  %1206 = select i1 %.not2, float %1189, float %1205, !dbg !7
  %1207 = select i1 %.not2, float %1205, float %1195, !dbg !7
  %1208 = extractvalue [2 x <8 x float>] %44, 0, !dbg !7
  %1209 = extractelement <8 x float> %1208, i64 0, !dbg !7
  %1210 = extractelement <8 x float> %1208, i64 1, !dbg !7
  %1211 = extractelement <8 x float> %1208, i64 2, !dbg !7
  %1212 = extractelement <8 x float> %1208, i64 3, !dbg !7
  %1213 = select i1 %.not, float %1210, float %1209, !dbg !7
  %1214 = bitcast float %1213 to i32, !dbg !7
  %1215 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1214), !dbg !7
  %1216 = bitcast i32 %1215 to float, !dbg !7
  %1217 = select i1 %.not, float %1209, float %1216, !dbg !7
  %1218 = select i1 %.not, float %1216, float %1210, !dbg !7
  %1219 = select i1 %.not, float %1212, float %1211, !dbg !7
  %1220 = bitcast float %1219 to i32, !dbg !7
  %1221 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1220), !dbg !7
  %1222 = bitcast i32 %1221 to float, !dbg !7
  %1223 = select i1 %.not, float %1211, float %1222, !dbg !7
  %1224 = select i1 %.not, float %1222, float %1212, !dbg !7
  %1225 = select i1 %.not2, float %1223, float %1217, !dbg !7
  %1226 = bitcast float %1225 to i32, !dbg !7
  %1227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1226), !dbg !7
  %1228 = select i1 %.not2, float %1224, float %1218, !dbg !7
  %1229 = bitcast float %1228 to i32, !dbg !7
  %1230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1229), !dbg !7
  %1231 = extractvalue [2 x <1 x float>] %45, 0, !dbg !18
  %1232 = shufflevector <1 x float> %1231, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1233 = extractelement <8 x float> %1109, i64 4, !dbg !7
  %1234 = extractelement <8 x float> %1109, i64 5, !dbg !7
  %1235 = extractelement <8 x float> %1109, i64 6, !dbg !7
  %1236 = extractelement <8 x float> %1109, i64 7, !dbg !7
  %1237 = select i1 %.not, float %1234, float %1233, !dbg !7
  %1238 = bitcast float %1237 to i32, !dbg !7
  %1239 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1238), !dbg !7
  %1240 = bitcast i32 %1239 to float, !dbg !7
  %1241 = select i1 %.not, float %1233, float %1240, !dbg !7
  %1242 = select i1 %.not, float %1240, float %1234, !dbg !7
  %1243 = select i1 %.not, float %1236, float %1235, !dbg !7
  %1244 = bitcast float %1243 to i32, !dbg !7
  %1245 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1244), !dbg !7
  %1246 = bitcast i32 %1245 to float, !dbg !7
  %1247 = select i1 %.not, float %1235, float %1246, !dbg !7
  %1248 = select i1 %.not, float %1246, float %1236, !dbg !7
  %1249 = select i1 %.not2, float %1247, float %1241, !dbg !7
  %1250 = bitcast float %1249 to i32, !dbg !7
  %1251 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1250), !dbg !7
  %1252 = bitcast i32 %1251 to float, !dbg !7
  %1253 = select i1 %.not2, float %1241, float %1252, !dbg !7
  %1254 = select i1 %.not2, float %1252, float %1247, !dbg !7
  %1255 = select i1 %.not2, float %1248, float %1242, !dbg !7
  %1256 = bitcast float %1255 to i32, !dbg !7
  %1257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1256), !dbg !7
  %1258 = bitcast i32 %1257 to float, !dbg !7
  %1259 = select i1 %.not2, float %1242, float %1258, !dbg !7
  %1260 = select i1 %.not2, float %1258, float %1248, !dbg !7
  %1261 = extractelement <8 x float> %1150, i64 4, !dbg !7
  %1262 = extractelement <8 x float> %1150, i64 5, !dbg !7
  %1263 = extractelement <8 x float> %1150, i64 6, !dbg !7
  %1264 = extractelement <8 x float> %1150, i64 7, !dbg !7
  %1265 = select i1 %.not, float %1262, float %1261, !dbg !7
  %1266 = bitcast float %1265 to i32, !dbg !7
  %1267 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1266), !dbg !7
  %1268 = bitcast i32 %1267 to float, !dbg !7
  %1269 = select i1 %.not, float %1261, float %1268, !dbg !7
  %1270 = select i1 %.not, float %1268, float %1262, !dbg !7
  %1271 = select i1 %.not, float %1264, float %1263, !dbg !7
  %1272 = bitcast float %1271 to i32, !dbg !7
  %1273 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1272), !dbg !7
  %1274 = bitcast i32 %1273 to float, !dbg !7
  %1275 = select i1 %.not, float %1263, float %1274, !dbg !7
  %1276 = select i1 %.not, float %1274, float %1264, !dbg !7
  %1277 = select i1 %.not2, float %1275, float %1269, !dbg !7
  %1278 = bitcast float %1277 to i32, !dbg !7
  %1279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1278), !dbg !7
  %1280 = bitcast i32 %1279 to float, !dbg !7
  %1281 = select i1 %.not2, float %1269, float %1280, !dbg !7
  %1282 = select i1 %.not2, float %1280, float %1275, !dbg !7
  %1283 = select i1 %.not2, float %1276, float %1270, !dbg !7
  %1284 = bitcast float %1283 to i32, !dbg !7
  %1285 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1284), !dbg !7
  %1286 = bitcast i32 %1285 to float, !dbg !7
  %1287 = select i1 %.not2, float %1270, float %1286, !dbg !7
  %1288 = select i1 %.not2, float %1286, float %1276, !dbg !7
  %1289 = extractelement <8 x float> %1179, i64 4, !dbg !7
  %1290 = extractelement <8 x float> %1179, i64 5, !dbg !7
  %1291 = extractelement <8 x float> %1179, i64 6, !dbg !7
  %1292 = extractelement <8 x float> %1179, i64 7, !dbg !7
  %1293 = select i1 %.not, float %1290, float %1289, !dbg !7
  %1294 = bitcast float %1293 to i32, !dbg !7
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1294), !dbg !7
  %1296 = bitcast i32 %1295 to float, !dbg !7
  %1297 = select i1 %.not, float %1289, float %1296, !dbg !7
  %1298 = select i1 %.not, float %1296, float %1290, !dbg !7
  %1299 = select i1 %.not, float %1292, float %1291, !dbg !7
  %1300 = bitcast float %1299 to i32, !dbg !7
  %1301 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1300), !dbg !7
  %1302 = bitcast i32 %1301 to float, !dbg !7
  %1303 = select i1 %.not, float %1291, float %1302, !dbg !7
  %1304 = select i1 %.not, float %1302, float %1292, !dbg !7
  %1305 = select i1 %.not2, float %1303, float %1297, !dbg !7
  %1306 = bitcast float %1305 to i32, !dbg !7
  %1307 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1306), !dbg !7
  %1308 = bitcast i32 %1307 to float, !dbg !7
  %1309 = select i1 %.not2, float %1297, float %1308, !dbg !7
  %1310 = select i1 %.not2, float %1308, float %1303, !dbg !7
  %1311 = select i1 %.not2, float %1304, float %1298, !dbg !7
  %1312 = bitcast float %1311 to i32, !dbg !7
  %1313 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1312), !dbg !7
  %1314 = bitcast i32 %1313 to float, !dbg !7
  %1315 = select i1 %.not2, float %1298, float %1314, !dbg !7
  %1316 = select i1 %.not2, float %1314, float %1304, !dbg !7
  %1317 = extractelement <8 x float> %1208, i64 4, !dbg !7
  %1318 = extractelement <8 x float> %1208, i64 5, !dbg !7
  %1319 = extractelement <8 x float> %1208, i64 6, !dbg !7
  %1320 = extractelement <8 x float> %1208, i64 7, !dbg !7
  %1321 = select i1 %.not, float %1318, float %1317, !dbg !7
  %1322 = bitcast float %1321 to i32, !dbg !7
  %1323 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1322), !dbg !7
  %1324 = bitcast i32 %1323 to float, !dbg !7
  %1325 = select i1 %.not, float %1317, float %1324, !dbg !7
  %1326 = select i1 %.not, float %1324, float %1318, !dbg !7
  %1327 = select i1 %.not, float %1320, float %1319, !dbg !7
  %1328 = bitcast float %1327 to i32, !dbg !7
  %1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1328), !dbg !7
  %1330 = bitcast i32 %1329 to float, !dbg !7
  %1331 = select i1 %.not, float %1319, float %1330, !dbg !7
  %1332 = select i1 %.not, float %1330, float %1320, !dbg !7
  %1333 = select i1 %.not2, float %1331, float %1325, !dbg !7
  %1334 = bitcast float %1333 to i32, !dbg !7
  %1335 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1334), !dbg !7
  %1336 = select i1 %.not2, float %1332, float %1326, !dbg !7
  %1337 = bitcast float %1336 to i32, !dbg !7
  %1338 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1337), !dbg !7
  %1339 = extractvalue [2 x <8 x float>] %49, 1, !dbg !7
  %1340 = extractelement <8 x float> %1339, i64 0, !dbg !7
  %1341 = extractelement <8 x float> %1339, i64 1, !dbg !7
  %1342 = extractelement <8 x float> %1339, i64 2, !dbg !7
  %1343 = extractelement <8 x float> %1339, i64 3, !dbg !7
  %1344 = select i1 %.not, float %1341, float %1340, !dbg !7
  %1345 = bitcast float %1344 to i32, !dbg !7
  %1346 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1345), !dbg !7
  %1347 = bitcast i32 %1346 to float, !dbg !7
  %1348 = select i1 %.not, float %1340, float %1347, !dbg !7
  %1349 = select i1 %.not, float %1347, float %1341, !dbg !7
  %1350 = select i1 %.not, float %1343, float %1342, !dbg !7
  %1351 = bitcast float %1350 to i32, !dbg !7
  %1352 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1351), !dbg !7
  %1353 = bitcast i32 %1352 to float, !dbg !7
  %1354 = select i1 %.not, float %1342, float %1353, !dbg !7
  %1355 = select i1 %.not, float %1353, float %1343, !dbg !7
  %1356 = select i1 %.not2, float %1354, float %1348, !dbg !7
  %1357 = bitcast float %1356 to i32, !dbg !7
  %1358 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1357), !dbg !7
  %1359 = bitcast i32 %1358 to float, !dbg !7
  %1360 = select i1 %.not2, float %1348, float %1359, !dbg !7
  %1361 = select i1 %.not2, float %1359, float %1354, !dbg !7
  %1362 = select i1 %.not2, float %1355, float %1349, !dbg !7
  %1363 = bitcast float %1362 to i32, !dbg !7
  %1364 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1363), !dbg !7
  %1365 = bitcast i32 %1364 to float, !dbg !7
  %1366 = select i1 %.not2, float %1349, float %1365, !dbg !7
  %1367 = select i1 %.not2, float %1365, float %1355, !dbg !7
  %1368 = extractvalue [2 x <8 x float>] %48, 1, !dbg !7
  %1369 = extractelement <8 x float> %1368, i64 0, !dbg !7
  %1370 = extractelement <8 x float> %1368, i64 1, !dbg !7
  %1371 = extractelement <8 x float> %1368, i64 2, !dbg !7
  %1372 = extractelement <8 x float> %1368, i64 3, !dbg !7
  %1373 = select i1 %.not, float %1370, float %1369, !dbg !7
  %1374 = bitcast float %1373 to i32, !dbg !7
  %1375 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1374), !dbg !7
  %1376 = bitcast i32 %1375 to float, !dbg !7
  %1377 = select i1 %.not, float %1369, float %1376, !dbg !7
  %1378 = select i1 %.not, float %1376, float %1370, !dbg !7
  %1379 = select i1 %.not, float %1372, float %1371, !dbg !7
  %1380 = bitcast float %1379 to i32, !dbg !7
  %1381 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1380), !dbg !7
  %1382 = bitcast i32 %1381 to float, !dbg !7
  %1383 = select i1 %.not, float %1371, float %1382, !dbg !7
  %1384 = select i1 %.not, float %1382, float %1372, !dbg !7
  %1385 = select i1 %.not2, float %1383, float %1377, !dbg !7
  %1386 = bitcast float %1385 to i32, !dbg !7
  %1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1386), !dbg !7
  %1388 = bitcast i32 %1387 to float, !dbg !7
  %1389 = select i1 %.not2, float %1377, float %1388, !dbg !7
  %1390 = select i1 %.not2, float %1388, float %1383, !dbg !7
  %1391 = select i1 %.not2, float %1384, float %1378, !dbg !7
  %1392 = bitcast float %1391 to i32, !dbg !7
  %1393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1392), !dbg !7
  %1394 = bitcast i32 %1393 to float, !dbg !7
  %1395 = select i1 %.not2, float %1378, float %1394, !dbg !7
  %1396 = select i1 %.not2, float %1394, float %1384, !dbg !7
  %1397 = extractvalue [2 x <8 x float>] %47, 1, !dbg !7
  %1398 = extractelement <8 x float> %1397, i64 0, !dbg !7
  %1399 = extractelement <8 x float> %1397, i64 1, !dbg !7
  %1400 = extractelement <8 x float> %1397, i64 2, !dbg !7
  %1401 = extractelement <8 x float> %1397, i64 3, !dbg !7
  %1402 = select i1 %.not, float %1399, float %1398, !dbg !7
  %1403 = bitcast float %1402 to i32, !dbg !7
  %1404 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1403), !dbg !7
  %1405 = bitcast i32 %1404 to float, !dbg !7
  %1406 = select i1 %.not, float %1398, float %1405, !dbg !7
  %1407 = select i1 %.not, float %1405, float %1399, !dbg !7
  %1408 = select i1 %.not, float %1401, float %1400, !dbg !7
  %1409 = bitcast float %1408 to i32, !dbg !7
  %1410 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1409), !dbg !7
  %1411 = bitcast i32 %1410 to float, !dbg !7
  %1412 = select i1 %.not, float %1400, float %1411, !dbg !7
  %1413 = select i1 %.not, float %1411, float %1401, !dbg !7
  %1414 = select i1 %.not2, float %1412, float %1406, !dbg !7
  %1415 = bitcast float %1414 to i32, !dbg !7
  %1416 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1415), !dbg !7
  %1417 = bitcast i32 %1416 to float, !dbg !7
  %1418 = select i1 %.not2, float %1406, float %1417, !dbg !7
  %1419 = select i1 %.not2, float %1417, float %1412, !dbg !7
  %1420 = select i1 %.not2, float %1413, float %1407, !dbg !7
  %1421 = bitcast float %1420 to i32, !dbg !7
  %1422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1421), !dbg !7
  %1423 = bitcast i32 %1422 to float, !dbg !7
  %1424 = select i1 %.not2, float %1407, float %1423, !dbg !7
  %1425 = select i1 %.not2, float %1423, float %1413, !dbg !7
  %1426 = extractvalue [2 x <8 x float>] %44, 1, !dbg !7
  %1427 = extractelement <8 x float> %1426, i64 0, !dbg !7
  %1428 = extractelement <8 x float> %1426, i64 1, !dbg !7
  %1429 = extractelement <8 x float> %1426, i64 2, !dbg !7
  %1430 = extractelement <8 x float> %1426, i64 3, !dbg !7
  %1431 = select i1 %.not, float %1428, float %1427, !dbg !7
  %1432 = bitcast float %1431 to i32, !dbg !7
  %1433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1432), !dbg !7
  %1434 = bitcast i32 %1433 to float, !dbg !7
  %1435 = select i1 %.not, float %1427, float %1434, !dbg !7
  %1436 = select i1 %.not, float %1434, float %1428, !dbg !7
  %1437 = select i1 %.not, float %1430, float %1429, !dbg !7
  %1438 = bitcast float %1437 to i32, !dbg !7
  %1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1438), !dbg !7
  %1440 = bitcast i32 %1439 to float, !dbg !7
  %1441 = select i1 %.not, float %1429, float %1440, !dbg !7
  %1442 = select i1 %.not, float %1440, float %1430, !dbg !7
  %1443 = select i1 %.not2, float %1441, float %1435, !dbg !7
  %1444 = bitcast float %1443 to i32, !dbg !7
  %1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1444), !dbg !7
  %1446 = select i1 %.not2, float %1442, float %1436, !dbg !7
  %1447 = bitcast float %1446 to i32, !dbg !7
  %1448 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1447), !dbg !7
  %1449 = extractvalue [2 x <1 x float>] %45, 1, !dbg !18
  %1450 = shufflevector <1 x float> %1449, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1451 = extractelement <8 x float> %1339, i64 4, !dbg !7
  %1452 = extractelement <8 x float> %1339, i64 5, !dbg !7
  %1453 = extractelement <8 x float> %1339, i64 6, !dbg !7
  %1454 = extractelement <8 x float> %1339, i64 7, !dbg !7
  %1455 = select i1 %.not, float %1452, float %1451, !dbg !7
  %1456 = bitcast float %1455 to i32, !dbg !7
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1456), !dbg !7
  %1458 = bitcast i32 %1457 to float, !dbg !7
  %1459 = select i1 %.not, float %1451, float %1458, !dbg !7
  %1460 = select i1 %.not, float %1458, float %1452, !dbg !7
  %1461 = select i1 %.not, float %1454, float %1453, !dbg !7
  %1462 = bitcast float %1461 to i32, !dbg !7
  %1463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1462), !dbg !7
  %1464 = bitcast i32 %1463 to float, !dbg !7
  %1465 = select i1 %.not, float %1453, float %1464, !dbg !7
  %1466 = select i1 %.not, float %1464, float %1454, !dbg !7
  %1467 = select i1 %.not2, float %1465, float %1459, !dbg !7
  %1468 = bitcast float %1467 to i32, !dbg !7
  %1469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1468), !dbg !7
  %1470 = bitcast i32 %1469 to float, !dbg !7
  %1471 = select i1 %.not2, float %1459, float %1470, !dbg !7
  %1472 = select i1 %.not2, float %1470, float %1465, !dbg !7
  %1473 = select i1 %.not2, float %1466, float %1460, !dbg !7
  %1474 = bitcast float %1473 to i32, !dbg !7
  %1475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1474), !dbg !7
  %1476 = bitcast i32 %1475 to float, !dbg !7
  %1477 = select i1 %.not2, float %1460, float %1476, !dbg !7
  %1478 = select i1 %.not2, float %1476, float %1466, !dbg !7
  %1479 = extractelement <8 x float> %1368, i64 4, !dbg !7
  %1480 = extractelement <8 x float> %1368, i64 5, !dbg !7
  %1481 = extractelement <8 x float> %1368, i64 6, !dbg !7
  %1482 = extractelement <8 x float> %1368, i64 7, !dbg !7
  %1483 = select i1 %.not, float %1480, float %1479, !dbg !7
  %1484 = bitcast float %1483 to i32, !dbg !7
  %1485 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1484), !dbg !7
  %1486 = bitcast i32 %1485 to float, !dbg !7
  %1487 = select i1 %.not, float %1479, float %1486, !dbg !7
  %1488 = select i1 %.not, float %1486, float %1480, !dbg !7
  %1489 = select i1 %.not, float %1482, float %1481, !dbg !7
  %1490 = bitcast float %1489 to i32, !dbg !7
  %1491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1490), !dbg !7
  %1492 = bitcast i32 %1491 to float, !dbg !7
  %1493 = select i1 %.not, float %1481, float %1492, !dbg !7
  %1494 = select i1 %.not, float %1492, float %1482, !dbg !7
  %1495 = select i1 %.not2, float %1493, float %1487, !dbg !7
  %1496 = bitcast float %1495 to i32, !dbg !7
  %1497 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1496), !dbg !7
  %1498 = bitcast i32 %1497 to float, !dbg !7
  %1499 = select i1 %.not2, float %1487, float %1498, !dbg !7
  %1500 = select i1 %.not2, float %1498, float %1493, !dbg !7
  %1501 = select i1 %.not2, float %1494, float %1488, !dbg !7
  %1502 = bitcast float %1501 to i32, !dbg !7
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1502), !dbg !7
  %1504 = bitcast i32 %1503 to float, !dbg !7
  %1505 = select i1 %.not2, float %1488, float %1504, !dbg !7
  %1506 = select i1 %.not2, float %1504, float %1494, !dbg !7
  %1507 = extractelement <8 x float> %1397, i64 4, !dbg !7
  %1508 = extractelement <8 x float> %1397, i64 5, !dbg !7
  %1509 = extractelement <8 x float> %1397, i64 6, !dbg !7
  %1510 = extractelement <8 x float> %1397, i64 7, !dbg !7
  %1511 = select i1 %.not, float %1508, float %1507, !dbg !7
  %1512 = bitcast float %1511 to i32, !dbg !7
  %1513 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1512), !dbg !7
  %1514 = bitcast i32 %1513 to float, !dbg !7
  %1515 = select i1 %.not, float %1507, float %1514, !dbg !7
  %1516 = select i1 %.not, float %1514, float %1508, !dbg !7
  %1517 = select i1 %.not, float %1510, float %1509, !dbg !7
  %1518 = bitcast float %1517 to i32, !dbg !7
  %1519 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1518), !dbg !7
  %1520 = bitcast i32 %1519 to float, !dbg !7
  %1521 = select i1 %.not, float %1509, float %1520, !dbg !7
  %1522 = select i1 %.not, float %1520, float %1510, !dbg !7
  %1523 = select i1 %.not2, float %1521, float %1515, !dbg !7
  %1524 = bitcast float %1523 to i32, !dbg !7
  %1525 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1524), !dbg !7
  %1526 = bitcast i32 %1525 to float, !dbg !7
  %1527 = select i1 %.not2, float %1515, float %1526, !dbg !7
  %1528 = select i1 %.not2, float %1526, float %1521, !dbg !7
  %1529 = select i1 %.not2, float %1522, float %1516, !dbg !7
  %1530 = bitcast float %1529 to i32, !dbg !7
  %1531 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1530), !dbg !7
  %1532 = bitcast i32 %1531 to float, !dbg !7
  %1533 = select i1 %.not2, float %1516, float %1532, !dbg !7
  %1534 = select i1 %.not2, float %1532, float %1522, !dbg !7
  %1535 = extractelement <8 x float> %1426, i64 4, !dbg !7
  %1536 = extractelement <8 x float> %1426, i64 5, !dbg !7
  %1537 = extractelement <8 x float> %1426, i64 6, !dbg !7
  %1538 = extractelement <8 x float> %1426, i64 7, !dbg !7
  %1539 = select i1 %.not, float %1536, float %1535, !dbg !7
  %1540 = bitcast float %1539 to i32, !dbg !7
  %1541 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1540), !dbg !7
  %1542 = bitcast i32 %1541 to float, !dbg !7
  %1543 = select i1 %.not, float %1535, float %1542, !dbg !7
  %1544 = select i1 %.not, float %1542, float %1536, !dbg !7
  %1545 = select i1 %.not, float %1538, float %1537, !dbg !7
  %1546 = bitcast float %1545 to i32, !dbg !7
  %1547 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1546), !dbg !7
  %1548 = bitcast i32 %1547 to float, !dbg !7
  %1549 = select i1 %.not, float %1537, float %1548, !dbg !7
  %1550 = select i1 %.not, float %1548, float %1538, !dbg !7
  %1551 = select i1 %.not2, float %1549, float %1543, !dbg !7
  %1552 = bitcast float %1551 to i32, !dbg !7
  %1553 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1552), !dbg !7
  %1554 = select i1 %.not2, float %1550, float %1544, !dbg !7
  %1555 = bitcast float %1554 to i32, !dbg !7
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1555), !dbg !7
  br i1 true, label %1557, label %1621, !dbg !20

1557:                                             ; preds = %1106
  %1558 = bitcast i32 %1553 to float, !dbg !7
  %1559 = select i1 %.not2, float %1543, float %1558, !dbg !7
  %1560 = insertelement <4 x float> poison, float %1559, i64 0, !dbg !18
  %1561 = bitcast i32 %1556 to float, !dbg !7
  %1562 = select i1 %.not2, float %1544, float %1561, !dbg !7
  %1563 = insertelement <4 x float> %1560, float %1562, i64 1, !dbg !18
  %1564 = select i1 %.not2, float %1558, float %1549, !dbg !7
  %1565 = insertelement <4 x float> %1563, float %1564, i64 2, !dbg !18
  %1566 = select i1 %.not2, float %1561, float %1550, !dbg !7
  %1567 = insertelement <4 x float> %1565, float %1566, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1450, !dbg !18
  %normalized = fmul <4 x float> %1567, %row.reciprocal, !dbg !18
  %1568 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %1569 = extractelement <4 x half> %1568, i64 3, !dbg !19
  %1570 = extractelement <4 x half> %1568, i64 2, !dbg !19
  %1571 = extractelement <4 x half> %1568, i64 1, !dbg !19
  %1572 = extractelement <4 x half> %1568, i64 0, !dbg !19
  %1573 = bitcast i32 %1445 to float, !dbg !7
  %1574 = select i1 %.not2, float %1435, float %1573, !dbg !7
  %1575 = insertelement <4 x float> poison, float %1574, i64 0, !dbg !18
  %1576 = bitcast i32 %1448 to float, !dbg !7
  %1577 = select i1 %.not2, float %1436, float %1576, !dbg !7
  %1578 = insertelement <4 x float> %1575, float %1577, i64 1, !dbg !18
  %1579 = select i1 %.not2, float %1573, float %1441, !dbg !7
  %1580 = insertelement <4 x float> %1578, float %1579, i64 2, !dbg !18
  %1581 = select i1 %.not2, float %1576, float %1442, !dbg !7
  %1582 = insertelement <4 x float> %1580, float %1581, i64 3, !dbg !18
  %normalized38 = fmul <4 x float> %1582, %row.reciprocal, !dbg !18
  %1583 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !19
  %1584 = extractelement <4 x half> %1583, i64 3, !dbg !19
  %1585 = extractelement <4 x half> %1583, i64 2, !dbg !19
  %1586 = extractelement <4 x half> %1583, i64 1, !dbg !19
  %1587 = extractelement <4 x half> %1583, i64 0, !dbg !19
  %1588 = bitcast i32 %1335 to float, !dbg !7
  %1589 = select i1 %.not2, float %1325, float %1588, !dbg !7
  %1590 = insertelement <4 x float> poison, float %1589, i64 0, !dbg !18
  %1591 = bitcast i32 %1338 to float, !dbg !7
  %1592 = select i1 %.not2, float %1326, float %1591, !dbg !7
  %1593 = insertelement <4 x float> %1590, float %1592, i64 1, !dbg !18
  %1594 = select i1 %.not2, float %1588, float %1331, !dbg !7
  %1595 = insertelement <4 x float> %1593, float %1594, i64 2, !dbg !18
  %1596 = select i1 %.not2, float %1591, float %1332, !dbg !7
  %1597 = insertelement <4 x float> %1595, float %1596, i64 3, !dbg !18
  %row.reciprocal39 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1232, !dbg !18
  %normalized40 = fmul <4 x float> %1597, %row.reciprocal39, !dbg !18
  %1598 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !19
  %1599 = extractelement <4 x half> %1598, i64 3, !dbg !19
  %1600 = extractelement <4 x half> %1598, i64 2, !dbg !19
  %1601 = extractelement <4 x half> %1598, i64 1, !dbg !19
  %1602 = extractelement <4 x half> %1598, i64 0, !dbg !19
  %1603 = bitcast i32 %1227 to float, !dbg !7
  %1604 = select i1 %.not2, float %1217, float %1603, !dbg !7
  %1605 = insertelement <4 x float> poison, float %1604, i64 0, !dbg !18
  %1606 = bitcast i32 %1230 to float, !dbg !7
  %1607 = select i1 %.not2, float %1218, float %1606, !dbg !7
  %1608 = insertelement <4 x float> %1605, float %1607, i64 1, !dbg !18
  %1609 = select i1 %.not2, float %1603, float %1223, !dbg !7
  %1610 = insertelement <4 x float> %1608, float %1609, i64 2, !dbg !18
  %1611 = select i1 %.not2, float %1606, float %1224, !dbg !7
  %1612 = insertelement <4 x float> %1610, float %1611, i64 3, !dbg !18
  %normalized41 = fmul <4 x float> %1612, %row.reciprocal39, !dbg !18
  %1613 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !19
  %1614 = extractelement <4 x half> %1613, i64 3, !dbg !19
  %1615 = extractelement <4 x half> %1613, i64 2, !dbg !19
  %1616 = extractelement <4 x half> %1613, i64 1, !dbg !19
  %1617 = extractelement <4 x half> %1613, i64 0, !dbg !19
  %1618 = add nsw i64 %8, %15, !dbg !20
  %1619 = shl nsw i64 %1618, 19, !dbg !20
  %1620 = and i64 %1619, 16252928, !dbg !20
  %address.thread241 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local242 = zext nneg i32 %address.thread241 to i64, !dbg !12
  %address.local243 = add nsw i64 %18, %address.local242, !dbg !20
  %address.local244 = lshr i64 %address.local242, 6, !dbg !12
  %address.local245 = shl nuw nsw i64 %address.local244, 4, !dbg !20
  %address.local246 = add nsw i64 %address.local243, %address.local245, !dbg !20
  %address.local247 = and i64 %address.local242, 240, !dbg !20
  %address.local248 = sub nsw i64 %address.local246, %address.local247, !dbg !20
  %address.local249 = add nsw i64 %address.local248, %19, !dbg !20
  %address.local250 = shl nuw nsw i64 %address.local249, 7, !dbg !20
  %address.local251 = add nuw nsw i64 %1620, %address.local250, !dbg !20
  %address.local252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local251, !dbg !20
  %address.local253 = lshr i64 %address.local242, 2, !dbg !12
  %address.local254 = and i64 %address.local253, 12, !dbg !12
  %address.local255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local252, i64 %address.local254, !dbg !20
  store half %1617, ptr addrspace(1) %address.local255, align 2, !dbg !20
  %address.local256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local251, !dbg !20
  %address.local257 = or disjoint i64 %address.local254, 1, !dbg !20
  %address.local258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local256, i64 %address.local257, !dbg !20
  store half %1616, ptr addrspace(1) %address.local258, align 2, !dbg !20
  %address.local259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local251, !dbg !20
  %address.local260 = or disjoint i64 %address.local254, 2, !dbg !20
  %address.local261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local259, i64 %address.local260, !dbg !20
  store half %1615, ptr addrspace(1) %address.local261, align 2, !dbg !20
  %address.local262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local251, !dbg !20
  %address.local263 = or disjoint i64 %address.local254, 3, !dbg !20
  %address.local264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local262, i64 %address.local263, !dbg !20
  store half %1614, ptr addrspace(1) %address.local264, align 2, !dbg !20
  %address.local265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local251, !dbg !20
  %address.local266 = or disjoint i64 %address.local254, 16, !dbg !20
  %address.local267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local265, i64 %address.local266, !dbg !20
  store half %1602, ptr addrspace(1) %address.local267, align 2, !dbg !20
  %address.local268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local251, !dbg !20
  %address.local269 = or disjoint i64 %address.local254, 17, !dbg !20
  %address.local270 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local268, i64 %address.local269, !dbg !20
  store half %1601, ptr addrspace(1) %address.local270, align 2, !dbg !20
  %address.local271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local251, !dbg !20
  %address.local272 = or disjoint i64 %address.local254, 18, !dbg !20
  %address.local273 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local271, i64 %address.local272, !dbg !20
  store half %1600, ptr addrspace(1) %address.local273, align 2, !dbg !20
  %address.local274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local251, !dbg !20
  %address.local275 = or disjoint i64 %address.local254, 19, !dbg !20
  %address.local276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local274, i64 %address.local275, !dbg !20
  store half %1599, ptr addrspace(1) %address.local276, align 2, !dbg !20
  %address.local277 = shl nsw i64 %address.local249, 7, !dbg !20
  %address.local278 = add i64 %address.local277, 8192, !dbg !20
  %address.local279 = add nuw nsw i64 %1620, %address.local278, !dbg !20
  %address.local280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local279, !dbg !20
  %address.local281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local280, i64 %address.local254, !dbg !20
  store half %1587, ptr addrspace(1) %address.local281, align 2, !dbg !20
  %address.local282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local279, !dbg !20
  %address.local283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local282, i64 %address.local257, !dbg !20
  store half %1586, ptr addrspace(1) %address.local283, align 2, !dbg !20
  %address.local284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local279, !dbg !20
  %address.local285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local284, i64 %address.local260, !dbg !20
  store half %1585, ptr addrspace(1) %address.local285, align 2, !dbg !20
  %address.local286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local279, !dbg !20
  %address.local287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local286, i64 %address.local263, !dbg !20
  store half %1584, ptr addrspace(1) %address.local287, align 2, !dbg !20
  %address.local288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local279, !dbg !20
  %address.local289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local288, i64 %address.local266, !dbg !20
  store half %1572, ptr addrspace(1) %address.local289, align 2, !dbg !20
  %address.local290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local279, !dbg !20
  %address.local291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local290, i64 %address.local269, !dbg !20
  store half %1571, ptr addrspace(1) %address.local291, align 2, !dbg !20
  %address.local292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local279, !dbg !20
  %address.local293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local292, i64 %address.local272, !dbg !20
  store half %1570, ptr addrspace(1) %address.local293, align 2, !dbg !20
  %address.local294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local279, !dbg !20
  %address.local295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local294, i64 %address.local275, !dbg !20
  store half %1569, ptr addrspace(1) %address.local295, align 2, !dbg !20
  br label %1621, !dbg !20

1621:                                             ; preds = %1557, %1106
  br i1 true, label %1622, label %1660, !dbg !20

1622:                                             ; preds = %1621
  %1623 = insertelement <4 x float> poison, float %1527, i64 0, !dbg !18
  %1624 = insertelement <4 x float> %1623, float %1533, i64 1, !dbg !18
  %1625 = insertelement <4 x float> %1624, float %1528, i64 2, !dbg !18
  %1626 = insertelement <4 x float> %1625, float %1534, i64 3, !dbg !18
  %row.reciprocal42 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1450, !dbg !18
  %normalized43 = fmul <4 x float> %1626, %row.reciprocal42, !dbg !18
  %1627 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !19
  %1628 = extractelement <4 x half> %1627, i64 3, !dbg !19
  %1629 = extractelement <4 x half> %1627, i64 2, !dbg !19
  %1630 = extractelement <4 x half> %1627, i64 1, !dbg !19
  %1631 = extractelement <4 x half> %1627, i64 0, !dbg !19
  %1632 = insertelement <4 x float> poison, float %1418, i64 0, !dbg !18
  %1633 = insertelement <4 x float> %1632, float %1424, i64 1, !dbg !18
  %1634 = insertelement <4 x float> %1633, float %1419, i64 2, !dbg !18
  %1635 = insertelement <4 x float> %1634, float %1425, i64 3, !dbg !18
  %normalized44 = fmul <4 x float> %1635, %row.reciprocal42, !dbg !18
  %1636 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !19
  %1637 = extractelement <4 x half> %1636, i64 3, !dbg !19
  %1638 = extractelement <4 x half> %1636, i64 2, !dbg !19
  %1639 = extractelement <4 x half> %1636, i64 1, !dbg !19
  %1640 = extractelement <4 x half> %1636, i64 0, !dbg !19
  %1641 = insertelement <4 x float> poison, float %1309, i64 0, !dbg !18
  %1642 = insertelement <4 x float> %1641, float %1315, i64 1, !dbg !18
  %1643 = insertelement <4 x float> %1642, float %1310, i64 2, !dbg !18
  %1644 = insertelement <4 x float> %1643, float %1316, i64 3, !dbg !18
  %row.reciprocal45 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1232, !dbg !18
  %normalized46 = fmul <4 x float> %1644, %row.reciprocal45, !dbg !18
  %1645 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !19
  %1646 = extractelement <4 x half> %1645, i64 3, !dbg !19
  %1647 = extractelement <4 x half> %1645, i64 2, !dbg !19
  %1648 = extractelement <4 x half> %1645, i64 1, !dbg !19
  %1649 = extractelement <4 x half> %1645, i64 0, !dbg !19
  %1650 = insertelement <4 x float> poison, float %1200, i64 0, !dbg !18
  %1651 = insertelement <4 x float> %1650, float %1206, i64 1, !dbg !18
  %1652 = insertelement <4 x float> %1651, float %1201, i64 2, !dbg !18
  %1653 = insertelement <4 x float> %1652, float %1207, i64 3, !dbg !18
  %normalized47 = fmul <4 x float> %1653, %row.reciprocal45, !dbg !18
  %1654 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !19
  %1655 = extractelement <4 x half> %1654, i64 3, !dbg !19
  %1656 = extractelement <4 x half> %1654, i64 2, !dbg !19
  %1657 = extractelement <4 x half> %1654, i64 1, !dbg !19
  %1658 = extractelement <4 x half> %1654, i64 0, !dbg !19
  %1659 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread296 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local297 = zext nneg i32 %address.thread296 to i64, !dbg !12
  %address.local298 = add nsw i64 %18, %address.local297, !dbg !20
  %address.local299 = lshr i64 %address.local297, 6, !dbg !12
  %address.local300 = shl nuw nsw i64 %address.local299, 4, !dbg !20
  %address.local301 = add nsw i64 %address.local298, %address.local300, !dbg !20
  %address.local302 = and i64 %address.local297, 240, !dbg !20
  %address.local303 = sub nsw i64 %address.local301, %address.local302, !dbg !20
  %address.local304 = add nsw i64 %address.local303, %19, !dbg !20
  %address.local305 = shl nuw nsw i64 %address.local304, 7, !dbg !20
  %address.local306 = add nuw nsw i64 %1659, %address.local305, !dbg !20
  %address.local307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local306, !dbg !20
  %address.local308 = lshr i64 %address.local297, 2, !dbg !12
  %address.local309 = and i64 %address.local308, 12, !dbg !12
  %address.local310 = or disjoint i64 %address.local309, 32, !dbg !20
  %address.local311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local307, i64 %address.local310, !dbg !20
  store half %1658, ptr addrspace(1) %address.local311, align 2, !dbg !20
  %address.local312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local306, !dbg !20
  %address.local313 = or disjoint i64 %address.local309, 33, !dbg !20
  %address.local314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local312, i64 %address.local313, !dbg !20
  store half %1657, ptr addrspace(1) %address.local314, align 2, !dbg !20
  %address.local315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local306, !dbg !20
  %address.local316 = or disjoint i64 %address.local309, 34, !dbg !20
  %address.local317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local315, i64 %address.local316, !dbg !20
  store half %1656, ptr addrspace(1) %address.local317, align 2, !dbg !20
  %address.local318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local306, !dbg !20
  %address.local319 = or disjoint i64 %address.local309, 35, !dbg !20
  %address.local320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local318, i64 %address.local319, !dbg !20
  store half %1655, ptr addrspace(1) %address.local320, align 2, !dbg !20
  %address.local321 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local306, !dbg !20
  %address.local322 = or disjoint i64 %address.local309, 48, !dbg !20
  %address.local323 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local321, i64 %address.local322, !dbg !20
  store half %1649, ptr addrspace(1) %address.local323, align 2, !dbg !20
  %address.local324 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local306, !dbg !20
  %address.local325 = or disjoint i64 %address.local309, 49, !dbg !20
  %address.local326 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local324, i64 %address.local325, !dbg !20
  store half %1648, ptr addrspace(1) %address.local326, align 2, !dbg !20
  %address.local327 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local306, !dbg !20
  %address.local328 = or disjoint i64 %address.local309, 50, !dbg !20
  %address.local329 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local327, i64 %address.local328, !dbg !20
  store half %1647, ptr addrspace(1) %address.local329, align 2, !dbg !20
  %address.local330 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local306, !dbg !20
  %address.local331 = or i64 %address.local308, 51, !dbg !20
  %address.local332 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local330, i64 %address.local331, !dbg !20
  store half %1646, ptr addrspace(1) %address.local332, align 2, !dbg !20
  %address.local333 = shl nsw i64 %address.local304, 7, !dbg !20
  %address.local334 = add i64 %address.local333, 8192, !dbg !20
  %address.local335 = add nuw nsw i64 %1659, %address.local334, !dbg !20
  %address.local336 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local335, !dbg !20
  %address.local337 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local336, i64 %address.local310, !dbg !20
  store half %1640, ptr addrspace(1) %address.local337, align 2, !dbg !20
  %address.local338 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local335, !dbg !20
  %address.local339 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local338, i64 %address.local313, !dbg !20
  store half %1639, ptr addrspace(1) %address.local339, align 2, !dbg !20
  %address.local340 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local335, !dbg !20
  %address.local341 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local340, i64 %address.local316, !dbg !20
  store half %1638, ptr addrspace(1) %address.local341, align 2, !dbg !20
  %address.local342 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local335, !dbg !20
  %address.local343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local342, i64 %address.local319, !dbg !20
  store half %1637, ptr addrspace(1) %address.local343, align 2, !dbg !20
  %address.local344 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local335, !dbg !20
  %address.local345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local344, i64 %address.local322, !dbg !20
  store half %1631, ptr addrspace(1) %address.local345, align 2, !dbg !20
  %address.local346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local335, !dbg !20
  %address.local347 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local346, i64 %address.local325, !dbg !20
  store half %1630, ptr addrspace(1) %address.local347, align 2, !dbg !20
  %address.local348 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local335, !dbg !20
  %address.local349 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local348, i64 %address.local328, !dbg !20
  store half %1629, ptr addrspace(1) %address.local349, align 2, !dbg !20
  %address.local350 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local335, !dbg !20
  %address.local351 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local350, i64 %address.local331, !dbg !20
  store half %1628, ptr addrspace(1) %address.local351, align 2, !dbg !20
  br label %1660, !dbg !20

1660:                                             ; preds = %1622, %1621
  br i1 true, label %1661, label %1699, !dbg !20

1661:                                             ; preds = %1660
  %1662 = insertelement <4 x float> poison, float %1499, i64 0, !dbg !18
  %1663 = insertelement <4 x float> %1662, float %1505, i64 1, !dbg !18
  %1664 = insertelement <4 x float> %1663, float %1500, i64 2, !dbg !18
  %1665 = insertelement <4 x float> %1664, float %1506, i64 3, !dbg !18
  %row.reciprocal48 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1450, !dbg !18
  %normalized49 = fmul <4 x float> %1665, %row.reciprocal48, !dbg !18
  %1666 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !19
  %1667 = extractelement <4 x half> %1666, i64 3, !dbg !19
  %1668 = extractelement <4 x half> %1666, i64 2, !dbg !19
  %1669 = extractelement <4 x half> %1666, i64 1, !dbg !19
  %1670 = extractelement <4 x half> %1666, i64 0, !dbg !19
  %1671 = insertelement <4 x float> poison, float %1389, i64 0, !dbg !18
  %1672 = insertelement <4 x float> %1671, float %1395, i64 1, !dbg !18
  %1673 = insertelement <4 x float> %1672, float %1390, i64 2, !dbg !18
  %1674 = insertelement <4 x float> %1673, float %1396, i64 3, !dbg !18
  %normalized50 = fmul <4 x float> %1674, %row.reciprocal48, !dbg !18
  %1675 = fptrunc <4 x float> %normalized50 to <4 x half>, !dbg !19
  %1676 = extractelement <4 x half> %1675, i64 3, !dbg !19
  %1677 = extractelement <4 x half> %1675, i64 2, !dbg !19
  %1678 = extractelement <4 x half> %1675, i64 1, !dbg !19
  %1679 = extractelement <4 x half> %1675, i64 0, !dbg !19
  %1680 = insertelement <4 x float> poison, float %1281, i64 0, !dbg !18
  %1681 = insertelement <4 x float> %1680, float %1287, i64 1, !dbg !18
  %1682 = insertelement <4 x float> %1681, float %1282, i64 2, !dbg !18
  %1683 = insertelement <4 x float> %1682, float %1288, i64 3, !dbg !18
  %row.reciprocal51 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1232, !dbg !18
  %normalized52 = fmul <4 x float> %1683, %row.reciprocal51, !dbg !18
  %1684 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !19
  %1685 = extractelement <4 x half> %1684, i64 3, !dbg !19
  %1686 = extractelement <4 x half> %1684, i64 2, !dbg !19
  %1687 = extractelement <4 x half> %1684, i64 1, !dbg !19
  %1688 = extractelement <4 x half> %1684, i64 0, !dbg !19
  %1689 = insertelement <4 x float> poison, float %1171, i64 0, !dbg !18
  %1690 = insertelement <4 x float> %1689, float %1177, i64 1, !dbg !18
  %1691 = insertelement <4 x float> %1690, float %1172, i64 2, !dbg !18
  %1692 = insertelement <4 x float> %1691, float %1178, i64 3, !dbg !18
  %normalized53 = fmul <4 x float> %1692, %row.reciprocal51, !dbg !18
  %1693 = fptrunc <4 x float> %normalized53 to <4 x half>, !dbg !19
  %1694 = extractelement <4 x half> %1693, i64 3, !dbg !19
  %1695 = extractelement <4 x half> %1693, i64 2, !dbg !19
  %1696 = extractelement <4 x half> %1693, i64 1, !dbg !19
  %1697 = extractelement <4 x half> %1693, i64 0, !dbg !19
  %1698 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread352 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local353 = zext nneg i32 %address.thread352 to i64, !dbg !12
  %address.local354 = add nsw i64 %18, %address.local353, !dbg !20
  %address.local355 = lshr i64 %address.local353, 6, !dbg !12
  %address.local356 = shl nuw nsw i64 %address.local355, 4, !dbg !20
  %address.local357 = add nsw i64 %address.local354, %address.local356, !dbg !20
  %address.local358 = and i64 %address.local353, 240, !dbg !20
  %address.local359 = sub nsw i64 %address.local357, %address.local358, !dbg !20
  %address.local360 = add nsw i64 %address.local359, %19, !dbg !20
  %address.local361 = shl nuw nsw i64 %address.local360, 7, !dbg !20
  %address.local362 = add nuw nsw i64 %1698, %address.local361, !dbg !20
  %address.local363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local362, !dbg !20
  %address.local364 = lshr i64 %address.local353, 2, !dbg !12
  %address.local365 = and i64 %address.local364, 12, !dbg !12
  %address.local366 = or disjoint i64 %address.local365, 64, !dbg !20
  %address.local367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local363, i64 %address.local366, !dbg !20
  store half %1697, ptr addrspace(1) %address.local367, align 2, !dbg !20
  %address.local368 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local362, !dbg !20
  %address.local369 = or disjoint i64 %address.local365, 65, !dbg !20
  %address.local370 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local368, i64 %address.local369, !dbg !20
  store half %1696, ptr addrspace(1) %address.local370, align 2, !dbg !20
  %address.local371 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local362, !dbg !20
  %address.local372 = or disjoint i64 %address.local365, 66, !dbg !20
  %address.local373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local371, i64 %address.local372, !dbg !20
  store half %1695, ptr addrspace(1) %address.local373, align 2, !dbg !20
  %address.local374 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local362, !dbg !20
  %address.local375 = or disjoint i64 %address.local365, 67, !dbg !20
  %address.local376 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local374, i64 %address.local375, !dbg !20
  store half %1694, ptr addrspace(1) %address.local376, align 2, !dbg !20
  %address.local377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local362, !dbg !20
  %address.local378 = or disjoint i64 %address.local365, 80, !dbg !20
  %address.local379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local377, i64 %address.local378, !dbg !20
  store half %1688, ptr addrspace(1) %address.local379, align 2, !dbg !20
  %address.local380 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local362, !dbg !20
  %address.local381 = or disjoint i64 %address.local365, 81, !dbg !20
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local380, i64 %address.local381, !dbg !20
  store half %1687, ptr addrspace(1) %address.local382, align 2, !dbg !20
  %address.local383 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local362, !dbg !20
  %address.local384 = or disjoint i64 %address.local365, 82, !dbg !20
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local383, i64 %address.local384, !dbg !20
  store half %1686, ptr addrspace(1) %address.local385, align 2, !dbg !20
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local362, !dbg !20
  %address.local387 = or disjoint i64 %address.local365, 83, !dbg !20
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local386, i64 %address.local387, !dbg !20
  store half %1685, ptr addrspace(1) %address.local388, align 2, !dbg !20
  %address.local389 = shl nsw i64 %address.local360, 7, !dbg !20
  %address.local390 = add i64 %address.local389, 8192, !dbg !20
  %address.local391 = add nuw nsw i64 %1698, %address.local390, !dbg !20
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !20
  %address.local393 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local392, i64 %address.local366, !dbg !20
  store half %1679, ptr addrspace(1) %address.local393, align 2, !dbg !20
  %address.local394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !20
  %address.local395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local394, i64 %address.local369, !dbg !20
  store half %1678, ptr addrspace(1) %address.local395, align 2, !dbg !20
  %address.local396 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !20
  %address.local397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local396, i64 %address.local372, !dbg !20
  store half %1677, ptr addrspace(1) %address.local397, align 2, !dbg !20
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !20
  %address.local399 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local398, i64 %address.local375, !dbg !20
  store half %1676, ptr addrspace(1) %address.local399, align 2, !dbg !20
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !20
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local378, !dbg !20
  store half %1670, ptr addrspace(1) %address.local401, align 2, !dbg !20
  %address.local402 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !20
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local402, i64 %address.local381, !dbg !20
  store half %1669, ptr addrspace(1) %address.local403, align 2, !dbg !20
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !20
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local404, i64 %address.local384, !dbg !20
  store half %1668, ptr addrspace(1) %address.local405, align 2, !dbg !20
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local391, !dbg !20
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local406, i64 %address.local387, !dbg !20
  store half %1667, ptr addrspace(1) %address.local407, align 2, !dbg !20
  br label %1699, !dbg !20

1699:                                             ; preds = %1661, %1660
  br i1 true, label %1700, label %1738, !dbg !20

1700:                                             ; preds = %1699
  %1701 = insertelement <4 x float> poison, float %1471, i64 0, !dbg !18
  %1702 = insertelement <4 x float> %1701, float %1477, i64 1, !dbg !18
  %1703 = insertelement <4 x float> %1702, float %1472, i64 2, !dbg !18
  %1704 = insertelement <4 x float> %1703, float %1478, i64 3, !dbg !18
  %row.reciprocal54 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1450, !dbg !18
  %normalized55 = fmul <4 x float> %1704, %row.reciprocal54, !dbg !18
  %1705 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !19
  %1706 = extractelement <4 x half> %1705, i64 3, !dbg !19
  %1707 = extractelement <4 x half> %1705, i64 2, !dbg !19
  %1708 = extractelement <4 x half> %1705, i64 1, !dbg !19
  %1709 = extractelement <4 x half> %1705, i64 0, !dbg !19
  %1710 = insertelement <4 x float> poison, float %1360, i64 0, !dbg !18
  %1711 = insertelement <4 x float> %1710, float %1366, i64 1, !dbg !18
  %1712 = insertelement <4 x float> %1711, float %1361, i64 2, !dbg !18
  %1713 = insertelement <4 x float> %1712, float %1367, i64 3, !dbg !18
  %normalized56 = fmul <4 x float> %1713, %row.reciprocal54, !dbg !18
  %1714 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !19
  %1715 = extractelement <4 x half> %1714, i64 3, !dbg !19
  %1716 = extractelement <4 x half> %1714, i64 2, !dbg !19
  %1717 = extractelement <4 x half> %1714, i64 1, !dbg !19
  %1718 = extractelement <4 x half> %1714, i64 0, !dbg !19
  %1719 = insertelement <4 x float> poison, float %1253, i64 0, !dbg !18
  %1720 = insertelement <4 x float> %1719, float %1259, i64 1, !dbg !18
  %1721 = insertelement <4 x float> %1720, float %1254, i64 2, !dbg !18
  %1722 = insertelement <4 x float> %1721, float %1260, i64 3, !dbg !18
  %row.reciprocal57 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1232, !dbg !18
  %normalized58 = fmul <4 x float> %1722, %row.reciprocal57, !dbg !18
  %1723 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !19
  %1724 = extractelement <4 x half> %1723, i64 3, !dbg !19
  %1725 = extractelement <4 x half> %1723, i64 2, !dbg !19
  %1726 = extractelement <4 x half> %1723, i64 1, !dbg !19
  %1727 = extractelement <4 x half> %1723, i64 0, !dbg !19
  %1728 = insertelement <4 x float> poison, float %1142, i64 0, !dbg !18
  %1729 = insertelement <4 x float> %1728, float %1148, i64 1, !dbg !18
  %1730 = insertelement <4 x float> %1729, float %1143, i64 2, !dbg !18
  %1731 = insertelement <4 x float> %1730, float %1149, i64 3, !dbg !18
  %normalized59 = fmul <4 x float> %1731, %row.reciprocal57, !dbg !18
  %1732 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !19
  %1733 = extractelement <4 x half> %1732, i64 3, !dbg !19
  %1734 = extractelement <4 x half> %1732, i64 2, !dbg !19
  %1735 = extractelement <4 x half> %1732, i64 1, !dbg !19
  %1736 = extractelement <4 x half> %1732, i64 0, !dbg !19
  %1737 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread408 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local409 = zext nneg i32 %address.thread408 to i64, !dbg !12
  %address.local410 = add nsw i64 %18, %address.local409, !dbg !20
  %address.local411 = lshr i64 %address.local409, 6, !dbg !12
  %address.local412 = shl nuw nsw i64 %address.local411, 4, !dbg !20
  %address.local413 = add nsw i64 %address.local410, %address.local412, !dbg !20
  %address.local414 = and i64 %address.local409, 240, !dbg !20
  %address.local415 = sub nsw i64 %address.local413, %address.local414, !dbg !20
  %address.local416 = add nsw i64 %address.local415, %19, !dbg !20
  %address.local417 = shl nuw nsw i64 %address.local416, 7, !dbg !20
  %address.local418 = add nuw nsw i64 %1737, %address.local417, !dbg !20
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local420 = lshr i64 %address.local409, 2, !dbg !12
  %address.local421 = and i64 %address.local420, 12, !dbg !12
  %address.local422 = or disjoint i64 %address.local421, 96, !dbg !20
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local419, i64 %address.local422, !dbg !20
  store half %1736, ptr addrspace(1) %address.local423, align 2, !dbg !20
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local425 = or disjoint i64 %address.local421, 97, !dbg !20
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local424, i64 %address.local425, !dbg !20
  store half %1735, ptr addrspace(1) %address.local426, align 2, !dbg !20
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local428 = or disjoint i64 %address.local421, 98, !dbg !20
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local427, i64 %address.local428, !dbg !20
  store half %1734, ptr addrspace(1) %address.local429, align 2, !dbg !20
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local431 = or disjoint i64 %address.local421, 99, !dbg !20
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local430, i64 %address.local431, !dbg !20
  store half %1733, ptr addrspace(1) %address.local432, align 2, !dbg !20
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local434 = or disjoint i64 %address.local421, 112, !dbg !20
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local434, !dbg !20
  store half %1727, ptr addrspace(1) %address.local435, align 2, !dbg !20
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local437 = or disjoint i64 %address.local421, 113, !dbg !20
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local436, i64 %address.local437, !dbg !20
  store half %1726, ptr addrspace(1) %address.local438, align 2, !dbg !20
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local440 = or disjoint i64 %address.local421, 114, !dbg !20
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local440, !dbg !20
  store half %1725, ptr addrspace(1) %address.local441, align 2, !dbg !20
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local418, !dbg !20
  %address.local443 = or i64 %address.local420, 115, !dbg !20
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local442, i64 %address.local443, !dbg !20
  store half %1724, ptr addrspace(1) %address.local444, align 2, !dbg !20
  %address.local445 = shl nsw i64 %address.local416, 7, !dbg !20
  %address.local446 = add i64 %address.local445, 8192, !dbg !20
  %address.local447 = add nuw nsw i64 %1737, %address.local446, !dbg !20
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local449 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local448, i64 %address.local422, !dbg !20
  store half %1718, ptr addrspace(1) %address.local449, align 2, !dbg !20
  %address.local450 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local451 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local450, i64 %address.local425, !dbg !20
  store half %1717, ptr addrspace(1) %address.local451, align 2, !dbg !20
  %address.local452 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local453 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local452, i64 %address.local428, !dbg !20
  store half %1716, ptr addrspace(1) %address.local453, align 2, !dbg !20
  %address.local454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local455 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local454, i64 %address.local431, !dbg !20
  store half %1715, ptr addrspace(1) %address.local455, align 2, !dbg !20
  %address.local456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local457 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local456, i64 %address.local434, !dbg !20
  store half %1709, ptr addrspace(1) %address.local457, align 2, !dbg !20
  %address.local458 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local458, i64 %address.local437, !dbg !20
  store half %1708, ptr addrspace(1) %address.local459, align 2, !dbg !20
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local440, !dbg !20
  store half %1707, ptr addrspace(1) %address.local461, align 2, !dbg !20
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local462, i64 %address.local443, !dbg !20
  store half %1706, ptr addrspace(1) %address.local463, align 2, !dbg !20
  br label %1738, !dbg !20

1738:                                             ; preds = %1700, %1699
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/baseline")
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
