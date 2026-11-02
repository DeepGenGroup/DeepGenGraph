; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/baseline/input.mlir"

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

40:                                               ; preds = %827, %4
  %41 = phi i64 [ %1758, %827 ], [ 0, %4 ], !dbg !7
  %42 = phi [2 x <8 x float>] [ %1743, %827 ], [ zeroinitializer, %4 ], !dbg !7
  %43 = phi [2 x <1 x float>] [ %1349, %827 ], [ zeroinitializer, %4 ], !dbg !7
  %44 = phi [2 x <1 x float>] [ %1745, %827 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %45 = phi [2 x <8 x float>] [ %1749, %827 ], [ zeroinitializer, %4 ], !dbg !7
  %46 = phi [2 x <8 x float>] [ %1753, %827 ], [ zeroinitializer, %4 ], !dbg !7
  %47 = phi [2 x <8 x float>] [ %1757, %827 ], [ zeroinitializer, %4 ], !dbg !7
  %.not = icmp sgt i64 %41, %10, !dbg !7
  br i1 %.not, label %1759, label %48, !dbg !7

48:                                               ; preds = %40
  %49 = shl nsw i64 %41, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !11, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local129 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local130 = and i64 %address.local129, 1536, !dbg !11
  %address.local131 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local130, !dbg !11
  %address.local132 = ptrtoint ptr addrspace(3) %address.local131 to i64, !dbg !11
  %address.local133 = trunc i64 %address.local132 to i32, !dbg !11
  %50 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %51 = zext i32 %50 to i64, !dbg !11
  %52 = inttoptr i64 %51 to ptr addrspace(3), !dbg !11
  %address.local134 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local135 = and i64 %address.local134, 16256, !dbg !11
  %address.local136 = lshr i64 %address.local, 4, !dbg !11
  %address.local137 = and i64 %address.local136, 8, !dbg !11
  %address.local138 = or disjoint i64 %address.local135, %address.local137, !dbg !11
  %address.local139 = getelementptr [2 x i8], ptr addrspace(1) %30, i64 %address.local138, !dbg !11
  %53 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %54 = ptrtoint ptr addrspace(1) %address.local139 to i64, !dbg !11
  %55 = sub i64 %54, %53, !dbg !11
  %dma.buffer.offset934 = trunc i64 %55 to i32, !dbg !11
  %56 = trunc i64 %53 to i32, !dbg !11
  %57 = insertelement <4 x i32> poison, i32 %56, i64 0, !dbg !11
  %58 = lshr i64 %53, 32, !dbg !11
  %59 = trunc i64 %58 to i32, !dbg !11
  %60 = insertelement <4 x i32> %57, i32 %59, i64 1, !dbg !11
  %61 = insertelement <4 x i32> %60, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc935 = insertelement <4 x i32> %61, i32 131072, i64 3, !dbg !11
  %62 = ptrtoint ptr addrspace(3) %52 to i32, !dbg !11
  %dma.buffer.lds936 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %62), !dbg !11
  %63 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds936, i32 %dma.buffer.offset934, <4 x i32> %dma.buffer.rsrc935) #10, !dbg !11
  %address.local140 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local141 = shl nuw nsw i64 %address.local140, 3, !dbg !11
  %address.local142 = and i64 %address.local141, 3584, !dbg !11
  %address.local143 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local142, !dbg !11
  %address.local144 = ptrtoint ptr addrspace(3) %address.local143 to i64, !dbg !11
  %address.local145 = trunc i64 %address.local144 to i32, !dbg !11
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %65 = zext i32 %64 to i64, !dbg !11
  %66 = inttoptr i64 %65 to ptr addrspace(3), !dbg !11
  %address.local146 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local147 = and i64 %address.local146, 16256, !dbg !11
  %address.local148 = lshr i64 %address.local140, 4, !dbg !11
  %address.local149 = and i64 %address.local148, 24, !dbg !11
  %address.local150 = or disjoint i64 %address.local147, %address.local149, !dbg !11
  %address.local151 = getelementptr [2 x i8], ptr addrspace(1) %30, i64 %address.local150, !dbg !11
  %67 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %68 = ptrtoint ptr addrspace(1) %address.local151 to i64, !dbg !11
  %69 = sub i64 %68, %67, !dbg !11
  %dma.buffer.offset931 = trunc i64 %69 to i32, !dbg !11
  %70 = trunc i64 %67 to i32, !dbg !11
  %71 = insertelement <4 x i32> poison, i32 %70, i64 0, !dbg !11
  %72 = lshr i64 %67, 32, !dbg !11
  %73 = trunc i64 %72 to i32, !dbg !11
  %74 = insertelement <4 x i32> %71, i32 %73, i64 1, !dbg !11
  %75 = insertelement <4 x i32> %74, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc932 = insertelement <4 x i32> %75, i32 131072, i64 3, !dbg !11
  %76 = ptrtoint ptr addrspace(3) %66 to i32, !dbg !11
  %dma.buffer.lds933 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %76), !dbg !11
  %77 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds933, i32 %dma.buffer.offset931, <4 x i32> %dma.buffer.rsrc932) #10, !dbg !11
  %78 = lshr i64 %41, 12, !dbg !15
  %79 = add nsw i64 %78, %8, !dbg !15
  %80 = lshr i64 %41, 5, !dbg !15
  %81 = and i64 %80, 127, !dbg !15
  %82 = shl i64 %79, 19, !dbg !15
  %83 = and i64 %82, 16252928, !dbg !15
  %84 = shl i64 %80, 19, !dbg !15
  %85 = shl i64 %41, 14, !dbg !15
  %86 = sub i64 %85, %84, !dbg !15
  %87 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %83, !dbg !15
  %88 = getelementptr [2 x i8], ptr addrspace(1) %87, i64 %81, !dbg !15
  %89 = getelementptr [2 x i8], ptr addrspace(1) %88, i64 %86, !dbg !15
  %address.local152 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local130, !dbg !15
  %address.local153 = ptrtoint ptr addrspace(3) %address.local152 to i64, !dbg !15
  %address.local154 = trunc i64 %address.local153 to i32, !dbg !15
  %90 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !15
  %91 = zext i32 %90 to i64, !dbg !15
  %92 = inttoptr i64 %91 to ptr addrspace(3), !dbg !15
  %address.local155 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local138, !dbg !15
  %93 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %94 = ptrtoint ptr addrspace(1) %address.local155 to i64, !dbg !15
  %95 = sub i64 %94, %93, !dbg !15
  %dma.buffer.offset928 = trunc i64 %95 to i32, !dbg !15
  %96 = trunc i64 %93 to i32, !dbg !15
  %97 = insertelement <4 x i32> poison, i32 %96, i64 0, !dbg !15
  %98 = lshr i64 %93, 32, !dbg !15
  %99 = trunc i64 %98 to i32, !dbg !15
  %100 = insertelement <4 x i32> %97, i32 %99, i64 1, !dbg !15
  %101 = insertelement <4 x i32> %100, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc929 = insertelement <4 x i32> %101, i32 131072, i64 3, !dbg !15
  %102 = ptrtoint ptr addrspace(3) %92 to i32, !dbg !15
  %dma.buffer.lds930 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %102), !dbg !15
  %103 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds930, i32 %dma.buffer.offset928, <4 x i32> %dma.buffer.rsrc929) #10, !dbg !15
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local142, !dbg !15
  %address.local157 = ptrtoint ptr addrspace(3) %address.local156 to i64, !dbg !15
  %address.local158 = trunc i64 %address.local157 to i32, !dbg !15
  %104 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !15
  %105 = zext i32 %104 to i64, !dbg !15
  %106 = inttoptr i64 %105 to ptr addrspace(3), !dbg !15
  %address.local159 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local150, !dbg !15
  %107 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %108 = ptrtoint ptr addrspace(1) %address.local159 to i64, !dbg !15
  %109 = sub i64 %108, %107, !dbg !15
  %dma.buffer.offset925 = trunc i64 %109 to i32, !dbg !15
  %110 = trunc i64 %107 to i32, !dbg !15
  %111 = insertelement <4 x i32> poison, i32 %110, i64 0, !dbg !15
  %112 = lshr i64 %107, 32, !dbg !15
  %113 = trunc i64 %112 to i32, !dbg !15
  %114 = insertelement <4 x i32> %111, i32 %113, i64 1, !dbg !15
  %115 = insertelement <4 x i32> %114, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc926 = insertelement <4 x i32> %115, i32 131072, i64 3, !dbg !15
  %116 = ptrtoint ptr addrspace(3) %106 to i32, !dbg !15
  %dma.buffer.lds927 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %116), !dbg !15
  %117 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds927, i32 %dma.buffer.offset925, <4 x i32> %dma.buffer.rsrc926) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local160 = lshr i32 %address.thread, 2, !dbg !12
  %address.local161 = and i32 %address.local160, 4, !dbg !12
  %address.local162 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local163 = and i32 %address.local162, 384, !dbg !12
  %address.local164 = or disjoint i32 %address.local161, %address.local163, !dbg !12
  %address.local165 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local166 = and i32 %address.local165, 120, !dbg !12
  %address.local167 = or disjoint i32 %address.local164, %address.local166, !dbg !12
  %address.local168 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local169 = and i32 %address.local168, 1024, !dbg !12
  %address.local170 = or disjoint i32 %address.local167, %address.local169, !dbg !12
  %address.local171 = zext nneg i32 %address.local170 to i64, !dbg !12
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %118 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %address.local174 = getelementptr i8, ptr addrspace(3) %address.local173, i64 4096, !dbg !12
  %119 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %address.local176 = getelementptr i8, ptr addrspace(3) %address.local175, i64 1024, !dbg !12
  %120 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %121 = fmul <4 x half> %118, splat (half 0xH3015), !dbg !12
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local177, i64 5120, !dbg !12
  %122 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %123 = fmul <4 x half> %119, splat (half 0xH3015), !dbg !12
  %124 = fmul <4 x half> %120, splat (half 0xH3015), !dbg !12
  %125 = fmul <4 x half> %122, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local179 = lshr i32 %address.thread, 2, !dbg !16
  %address.local180 = and i32 %address.local179, 4, !dbg !16
  %address.local181 = shl nuw nsw i32 %address.thread, 1, !dbg !16
  %address.local182 = and i32 %address.local181, 24, !dbg !16
  %address.local183 = or disjoint i32 %address.local180, %address.local182, !dbg !16
  %address.local184 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local185 = and i32 %address.local184, 1024, !dbg !16
  %address.local186 = or disjoint i32 %address.local183, %address.local185, !dbg !16
  %address.local187 = shl nuw nsw i32 %address.thread, 7, !dbg !16
  %address.local188 = and i32 %address.local187, 384, !dbg !16
  %address.local189 = or disjoint i32 %address.local186, %address.local188, !dbg !16
  %address.local190 = zext nneg i32 %address.local189 to i64, !dbg !16
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local190, !dbg !16
  %126 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !16
  %127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %126, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local192 = lshr i32 %address.thread, 2, !dbg !16
  %address.local193 = and i32 %address.local192, 4, !dbg !16
  %address.local194 = shl nuw nsw i32 %address.thread, 1, !dbg !16
  %address.local195 = and i32 %address.local194, 24, !dbg !16
  %address.local196 = or disjoint i32 %address.local193, %address.local195, !dbg !16
  %address.local197 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local198 = and i32 %address.local197, 1024, !dbg !16
  %address.local199 = or disjoint i32 %address.local196, %address.local198, !dbg !16
  %address.local200 = shl nuw nsw i32 %address.thread, 7, !dbg !16
  %address.local201 = and i32 %address.local200, 384, !dbg !16
  %address.local202 = or disjoint i32 %address.local199, %address.local201, !dbg !16
  %address.local203 = zext nneg i32 %address.local202 to i64, !dbg !16
  %address.local204 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !16
  %address.local205 = getelementptr i8, ptr addrspace(3) %address.local204, i64 4096, !dbg !16
  %128 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %address.local206 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !16
  %address.local208 = lshr i64 %address.local, 2, !dbg !16
  %address.local209 = mul nsw i64 %address.local208, -512, !dbg !16
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i64 %address.local209, !dbg !16
  %address.local211 = shl nuw nsw i64 %address.local, 1, !dbg !16
  %address.local212 = and i64 %address.local211, 24, !dbg !16
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) %address.local210, i64 %address.local212, !dbg !16
  %address.local214 = lshr i64 %address.local, 2, !dbg !12
  %address.local215 = and i64 %address.local214, 12, !dbg !12
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i64 %address.local215, !dbg !16
  %address.local217 = and i64 %address.local, 63, !dbg !12
  %address.local218 = lshr i64 %address.local217, 5, !dbg !12
  %address.local219 = mul nuw nsw i64 %address.local218, 1016, !dbg !12
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local216, i64 %address.local219, !dbg !16
  %address.local221 = getelementptr i8, ptr addrspace(3) %address.local220, i64 64, !dbg !16
  %129 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !16
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %128, <4 x float> %127) #9, !dbg !16
  %131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %129, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !16
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local215, !dbg !16
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local219, !dbg !16
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i64 %address.local209, !dbg !16
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local212, !dbg !16
  %address.local227 = getelementptr i8, ptr addrspace(3) %address.local226, i64 4160, !dbg !16
  %132 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !16
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !16
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local209, !dbg !16
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local212, !dbg !16
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) %address.local230, i64 %address.local215, !dbg !16
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %address.local231, i64 %address.local219, !dbg !16
  %address.local233 = getelementptr i8, ptr addrspace(3) %address.local232, i64 128, !dbg !16
  %133 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !16
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %132, <4 x float> %131) #9, !dbg !16
  %135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %133, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !16
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local215, !dbg !16
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local219, !dbg !16
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local236, i64 %address.local209, !dbg !16
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %address.local237, i64 %address.local212, !dbg !16
  %address.local239 = getelementptr i8, ptr addrspace(3) %address.local238, i64 4224, !dbg !16
  %136 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !16
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local209, !dbg !16
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local212, !dbg !16
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) %address.local242, i64 %address.local215, !dbg !16
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local243, i64 %address.local219, !dbg !16
  %address.local245 = getelementptr i8, ptr addrspace(3) %address.local244, i64 192, !dbg !16
  %137 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !16
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %136, <4 x float> %135) #9, !dbg !16
  %139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %137, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !16
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) %address.local246, i64 %address.local215, !dbg !16
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local219, !dbg !16
  %address.local249 = getelementptr [2 x i8], ptr addrspace(3) %address.local248, i64 %address.local209, !dbg !16
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) %address.local249, i64 %address.local212, !dbg !16
  %address.local251 = getelementptr i8, ptr addrspace(3) %address.local250, i64 4288, !dbg !16
  %140 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !16
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local190, !dbg !16
  %address.local253 = getelementptr i8, ptr addrspace(3) %address.local252, i64 1024, !dbg !16
  %141 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !16
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %140, <4 x float> %139) #9, !dbg !16
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %141, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !16
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local254, i64 5120, !dbg !16
  %144 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !16
  %address.local256 = add nuw nsw i64 %address.local206, 512, !dbg !16
  %address.local257 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local256, !dbg !16
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %address.local257, i64 %address.local209, !dbg !16
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i64 %address.local212, !dbg !16
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i64 %address.local215, !dbg !16
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local219, !dbg !16
  %address.local262 = getelementptr i8, ptr addrspace(3) %address.local261, i64 64, !dbg !16
  %145 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !16
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %144, <4 x float> %143) #9, !dbg !16
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %145, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local256, !dbg !16
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local215, !dbg !16
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local219, !dbg !16
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %address.local265, i64 %address.local209, !dbg !16
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i64 %address.local212, !dbg !16
  %address.local268 = getelementptr i8, ptr addrspace(3) %address.local267, i64 4160, !dbg !16
  %148 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !16
  %address.local269 = add nuw nsw i64 %address.local206, 512, !dbg !16
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local269, !dbg !16
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local209, !dbg !16
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) %address.local271, i64 %address.local212, !dbg !16
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local272, i64 %address.local215, !dbg !16
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local219, !dbg !16
  %address.local275 = getelementptr i8, ptr addrspace(3) %address.local274, i64 128, !dbg !16
  %149 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !16
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %148, <4 x float> %147) #9, !dbg !16
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %149, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local269, !dbg !16
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local215, !dbg !16
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local219, !dbg !16
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) %address.local278, i64 %address.local209, !dbg !16
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %address.local279, i64 %address.local212, !dbg !16
  %address.local281 = getelementptr i8, ptr addrspace(3) %address.local280, i64 4224, !dbg !16
  %152 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !16
  %address.local282 = add nuw nsw i64 %address.local206, 512, !dbg !16
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local282, !dbg !16
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local209, !dbg !16
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) %address.local284, i64 %address.local212, !dbg !16
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) %address.local285, i64 %address.local215, !dbg !16
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %address.local286, i64 %address.local219, !dbg !16
  %address.local288 = getelementptr i8, ptr addrspace(3) %address.local287, i64 192, !dbg !16
  %153 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !16
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %152, <4 x float> %151) #9, !dbg !16
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local282, !dbg !16
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local215, !dbg !16
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local219, !dbg !16
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) %address.local291, i64 %address.local209, !dbg !16
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %address.local292, i64 %address.local212, !dbg !16
  %address.local294 = getelementptr i8, ptr addrspace(3) %address.local293, i64 4288, !dbg !16
  %156 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !16
  %157 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !16
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %156, <4 x float> %155) #9, !dbg !16
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !16
  %160 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %161 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !16
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %160, <4 x float> %159) #9, !dbg !16
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %161, <4 x float> zeroinitializer) #9, !dbg !16
  %164 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !16
  %165 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !16
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %164, <4 x float> %163) #9, !dbg !16
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !16
  %168 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %169 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !16
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %168, <4 x float> %167) #9, !dbg !16
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %169, <4 x float> zeroinitializer) #9, !dbg !16
  %172 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !16
  %173 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !16
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %172, <4 x float> %171) #9, !dbg !16
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %173, <4 x float> zeroinitializer) #9, !dbg !16
  %176 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !16
  %177 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !16
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %176, <4 x float> %175) #9, !dbg !16
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %177, <4 x float> zeroinitializer) #9, !dbg !16
  %180 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !16
  %181 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !16
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %180, <4 x float> %179) #9, !dbg !16
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %181, <4 x float> zeroinitializer) #9, !dbg !16
  %184 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !16
  %185 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !16
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %184, <4 x float> %183) #9, !dbg !16
  %187 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %185, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %187, <4 x float> %188) #9, !dbg !16
  %190 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %191 = zext i32 %190 to i64, !dbg !11
  %192 = inttoptr i64 %191 to ptr addrspace(3), !dbg !11
  %address.local295 = getelementptr [2 x i8], ptr addrspace(1) %32, i64 %address.local138, !dbg !11
  %193 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %194 = ptrtoint ptr addrspace(1) %address.local295 to i64, !dbg !11
  %195 = sub i64 %194, %193, !dbg !11
  %dma.buffer.offset922 = trunc i64 %195 to i32, !dbg !11
  %196 = trunc i64 %193 to i32, !dbg !11
  %197 = insertelement <4 x i32> poison, i32 %196, i64 0, !dbg !11
  %198 = lshr i64 %193, 32, !dbg !11
  %199 = trunc i64 %198 to i32, !dbg !11
  %200 = insertelement <4 x i32> %197, i32 %199, i64 1, !dbg !11
  %201 = insertelement <4 x i32> %200, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc923 = insertelement <4 x i32> %201, i32 131072, i64 3, !dbg !11
  %202 = ptrtoint ptr addrspace(3) %192 to i32, !dbg !11
  %dma.buffer.lds924 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %202), !dbg !11
  %203 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds924, i32 %dma.buffer.offset922, <4 x i32> %dma.buffer.rsrc923) #10, !dbg !11
  %204 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %205 = zext i32 %204 to i64, !dbg !11
  %206 = inttoptr i64 %205 to ptr addrspace(3), !dbg !11
  %address.local296 = getelementptr [2 x i8], ptr addrspace(1) %32, i64 %address.local150, !dbg !11
  %207 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %208 = ptrtoint ptr addrspace(1) %address.local296 to i64, !dbg !11
  %209 = sub i64 %208, %207, !dbg !11
  %dma.buffer.offset919 = trunc i64 %209 to i32, !dbg !11
  %210 = trunc i64 %207 to i32, !dbg !11
  %211 = insertelement <4 x i32> poison, i32 %210, i64 0, !dbg !11
  %212 = lshr i64 %207, 32, !dbg !11
  %213 = trunc i64 %212 to i32, !dbg !11
  %214 = insertelement <4 x i32> %211, i32 %213, i64 1, !dbg !11
  %215 = insertelement <4 x i32> %214, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc920 = insertelement <4 x i32> %215, i32 131072, i64 3, !dbg !11
  %216 = ptrtoint ptr addrspace(3) %206 to i32, !dbg !11
  %dma.buffer.lds921 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %216), !dbg !11
  %217 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds921, i32 %dma.buffer.offset919, <4 x i32> %dma.buffer.rsrc920) #10, !dbg !11
  %218 = add nuw i64 %41, 1024, !dbg !15
  %219 = icmp slt i64 %218, 0, !dbg !15
  %220 = sub nuw i64 -1025, %41, !dbg !15
  %221 = select i1 %219, i64 %220, i64 %218, !dbg !15
  %222 = sdiv i64 %221, 4096, !dbg !15
  %.lobit4 = ashr i64 %218, 63, !dbg !15
  %223 = xor i64 %222, %.lobit4, !dbg !15
  %224 = add nsw i64 %223, %8, !dbg !15
  %225 = add nsw i64 %80, %35, !dbg !15
  %226 = add nsw i64 %225, 32, !dbg !15
  %227 = icmp slt i64 %225, -32, !dbg !15
  %228 = sub nsw i64 -33, %225, !dbg !15
  %229 = select i1 %227, i64 %228, i64 %226, !dbg !15
  %230 = sdiv i64 %229, 128, !dbg !15
  %231 = sext i1 %227 to i64, !dbg !15
  %232 = xor i64 %230, %231, !dbg !15
  %233 = shl nsw i64 %232, 7, !dbg !15
  %234 = sub i64 %35, %233, !dbg !15
  %235 = shl i64 %224, 19, !dbg !15
  %236 = and i64 %235, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %237 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %236, !dbg !15
  %238 = getelementptr [2 x i8], ptr addrspace(1) %237, i64 %80, !dbg !15
  %239 = getelementptr [2 x i8], ptr addrspace(1) %238, i64 %234, !dbg !15
  %240 = getelementptr i8, ptr addrspace(1) %239, i64 64, !dbg !15
  %241 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %86, !dbg !15
  %242 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !15
  %243 = zext i32 %242 to i64, !dbg !15
  %244 = inttoptr i64 %243 to ptr addrspace(3), !dbg !15
  %address.local297 = getelementptr [2 x i8], ptr addrspace(1) %241, i64 %address.local138, !dbg !15
  %245 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %246 = ptrtoint ptr addrspace(1) %address.local297 to i64, !dbg !15
  %247 = sub i64 %246, %245, !dbg !15
  %dma.buffer.offset916 = trunc i64 %247 to i32, !dbg !15
  %248 = trunc i64 %245 to i32, !dbg !15
  %249 = insertelement <4 x i32> poison, i32 %248, i64 0, !dbg !15
  %250 = lshr i64 %245, 32, !dbg !15
  %251 = trunc i64 %250 to i32, !dbg !15
  %252 = insertelement <4 x i32> %249, i32 %251, i64 1, !dbg !15
  %253 = insertelement <4 x i32> %252, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc917 = insertelement <4 x i32> %253, i32 131072, i64 3, !dbg !15
  %254 = ptrtoint ptr addrspace(3) %244 to i32, !dbg !15
  %dma.buffer.lds918 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %254), !dbg !15
  %255 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds918, i32 %dma.buffer.offset916, <4 x i32> %dma.buffer.rsrc917) #10, !dbg !15
  %256 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !15
  %257 = zext i32 %256 to i64, !dbg !15
  %258 = inttoptr i64 %257 to ptr addrspace(3), !dbg !15
  %address.local298 = getelementptr [2 x i8], ptr addrspace(1) %241, i64 %address.local150, !dbg !15
  %259 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %260 = ptrtoint ptr addrspace(1) %address.local298 to i64, !dbg !15
  %261 = sub i64 %260, %259, !dbg !15
  %dma.buffer.offset913 = trunc i64 %261 to i32, !dbg !15
  %262 = trunc i64 %259 to i32, !dbg !15
  %263 = insertelement <4 x i32> poison, i32 %262, i64 0, !dbg !15
  %264 = lshr i64 %259, 32, !dbg !15
  %265 = trunc i64 %264 to i32, !dbg !15
  %266 = insertelement <4 x i32> %263, i32 %265, i64 1, !dbg !15
  %267 = insertelement <4 x i32> %266, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc914 = insertelement <4 x i32> %267, i32 131072, i64 3, !dbg !15
  %268 = ptrtoint ptr addrspace(3) %258 to i32, !dbg !15
  %dma.buffer.lds915 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %268), !dbg !15
  %269 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds915, i32 %dma.buffer.offset913, <4 x i32> %dma.buffer.rsrc914) #10, !dbg !15
  %270 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %271 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %272 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %273 = fmul <4 x half> %270, splat (half 0xH3015), !dbg !12
  %274 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %275 = fmul <4 x half> %271, splat (half 0xH3015), !dbg !12
  %276 = fmul <4 x half> %272, splat (half 0xH3015), !dbg !12
  %277 = fmul <4 x half> %274, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %278 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !16
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %273, <4 x half> %278, <4 x float> %130) #9, !dbg !16
  %280 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %281 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !16
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %280, <4 x float> %279) #9, !dbg !16
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %273, <4 x half> %281, <4 x float> %134) #9, !dbg !16
  %284 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !16
  %285 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !16
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %284, <4 x float> %283) #9, !dbg !16
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %273, <4 x half> %285, <4 x float> %138) #9, !dbg !16
  %288 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %289 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !16
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %288, <4 x float> %287) #9, !dbg !16
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %273, <4 x half> %289, <4 x float> %142) #9, !dbg !16
  %292 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !16
  %293 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !16
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %292, <4 x float> %291) #9, !dbg !16
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %273, <4 x half> %293, <4 x float> %146) #9, !dbg !16
  %296 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !16
  %297 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !16
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %296, <4 x float> %295) #9, !dbg !16
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %273, <4 x half> %297, <4 x float> %150) #9, !dbg !16
  %300 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !16
  %301 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !16
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %300, <4 x float> %299) #9, !dbg !16
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %273, <4 x half> %301, <4 x float> %154) #9, !dbg !16
  %304 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !16
  %305 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !16
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %304, <4 x float> %303) #9, !dbg !16
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %273, <4 x half> %305, <4 x float> %158) #9, !dbg !16
  %308 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !16
  %309 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !16
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %308, <4 x float> %307) #9, !dbg !16
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %309, <4 x float> %162) #9, !dbg !16
  %312 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %313 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !16
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %277, <4 x half> %312, <4 x float> %311) #9, !dbg !16
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %313, <4 x float> %166) #9, !dbg !16
  %316 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !16
  %317 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !16
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %277, <4 x half> %316, <4 x float> %315) #9, !dbg !16
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %317, <4 x float> %170) #9, !dbg !16
  %320 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %321 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !16
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %277, <4 x half> %320, <4 x float> %319) #9, !dbg !16
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %321, <4 x float> %174) #9, !dbg !16
  %324 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !16
  %325 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !16
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %277, <4 x half> %324, <4 x float> %323) #9, !dbg !16
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %325, <4 x float> %178) #9, !dbg !16
  %328 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !16
  %329 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !16
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %277, <4 x half> %328, <4 x float> %327) #9, !dbg !16
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %329, <4 x float> %182) #9, !dbg !16
  %332 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !16
  %333 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !16
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %277, <4 x half> %332, <4 x float> %331) #9, !dbg !16
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %333, <4 x float> %186) #9, !dbg !16
  %336 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !16
  %337 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !16
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %277, <4 x half> %336, <4 x float> %335) #9, !dbg !16
  %339 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %337, <4 x float> %189) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %277, <4 x half> %339, <4 x float> %340) #9, !dbg !16
  %342 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %343 = zext i32 %342 to i64, !dbg !11
  %344 = inttoptr i64 %343 to ptr addrspace(3), !dbg !11
  %address.local299 = getelementptr [2 x i8], ptr addrspace(1) %37, i64 %address.local138, !dbg !11
  %345 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %346 = ptrtoint ptr addrspace(1) %address.local299 to i64, !dbg !11
  %347 = sub i64 %346, %345, !dbg !11
  %dma.buffer.offset910 = trunc i64 %347 to i32, !dbg !11
  %348 = trunc i64 %345 to i32, !dbg !11
  %349 = insertelement <4 x i32> poison, i32 %348, i64 0, !dbg !11
  %350 = lshr i64 %345, 32, !dbg !11
  %351 = trunc i64 %350 to i32, !dbg !11
  %352 = insertelement <4 x i32> %349, i32 %351, i64 1, !dbg !11
  %353 = insertelement <4 x i32> %352, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc911 = insertelement <4 x i32> %353, i32 131072, i64 3, !dbg !11
  %354 = ptrtoint ptr addrspace(3) %344 to i32, !dbg !11
  %dma.buffer.lds912 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %354), !dbg !11
  %355 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds912, i32 %dma.buffer.offset910, <4 x i32> %dma.buffer.rsrc911) #10, !dbg !11
  %356 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %357 = zext i32 %356 to i64, !dbg !11
  %358 = inttoptr i64 %357 to ptr addrspace(3), !dbg !11
  %address.local300 = getelementptr [2 x i8], ptr addrspace(1) %37, i64 %address.local150, !dbg !11
  %359 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %360 = ptrtoint ptr addrspace(1) %address.local300 to i64, !dbg !11
  %361 = sub i64 %360, %359, !dbg !11
  %dma.buffer.offset907 = trunc i64 %361 to i32, !dbg !11
  %362 = trunc i64 %359 to i32, !dbg !11
  %363 = insertelement <4 x i32> poison, i32 %362, i64 0, !dbg !11
  %364 = lshr i64 %359, 32, !dbg !11
  %365 = trunc i64 %364 to i32, !dbg !11
  %366 = insertelement <4 x i32> %363, i32 %365, i64 1, !dbg !11
  %367 = insertelement <4 x i32> %366, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc908 = insertelement <4 x i32> %367, i32 131072, i64 3, !dbg !11
  %368 = ptrtoint ptr addrspace(3) %358 to i32, !dbg !11
  %dma.buffer.lds909 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %368), !dbg !11
  %369 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds909, i32 %dma.buffer.offset907, <4 x i32> %dma.buffer.rsrc908) #10, !dbg !11
  %370 = add nuw i64 %41, 2048, !dbg !15
  %371 = icmp slt i64 %370, 0, !dbg !15
  %372 = sub nuw i64 -2049, %41, !dbg !15
  %373 = select i1 %371, i64 %372, i64 %370, !dbg !15
  %374 = sdiv i64 %373, 4096, !dbg !15
  %.lobit5 = ashr i64 %370, 63, !dbg !15
  %375 = xor i64 %374, %.lobit5, !dbg !15
  %376 = add nsw i64 %375, %8, !dbg !15
  %377 = add nsw i64 %225, 64, !dbg !15
  %378 = icmp slt i64 %225, -64, !dbg !15
  %379 = sub nsw i64 -65, %225, !dbg !15
  %380 = select i1 %378, i64 %379, i64 %377, !dbg !15
  %381 = sdiv i64 %380, 128, !dbg !15
  %382 = sext i1 %378 to i64, !dbg !15
  %383 = xor i64 %381, %382, !dbg !15
  %384 = shl nsw i64 %383, 7, !dbg !15
  %385 = sub i64 %35, %384, !dbg !15
  %386 = shl i64 %376, 19, !dbg !15
  %387 = and i64 %386, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %388 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %387, !dbg !15
  %389 = getelementptr [2 x i8], ptr addrspace(1) %388, i64 %80, !dbg !15
  %390 = getelementptr [2 x i8], ptr addrspace(1) %389, i64 %385, !dbg !15
  %391 = getelementptr i8, ptr addrspace(1) %390, i64 128, !dbg !15
  %392 = getelementptr [2 x i8], ptr addrspace(1) %391, i64 %86, !dbg !15
  %393 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !15
  %394 = zext i32 %393 to i64, !dbg !15
  %395 = inttoptr i64 %394 to ptr addrspace(3), !dbg !15
  %address.local301 = getelementptr [2 x i8], ptr addrspace(1) %392, i64 %address.local138, !dbg !15
  %396 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %397 = ptrtoint ptr addrspace(1) %address.local301 to i64, !dbg !15
  %398 = sub i64 %397, %396, !dbg !15
  %dma.buffer.offset904 = trunc i64 %398 to i32, !dbg !15
  %399 = trunc i64 %396 to i32, !dbg !15
  %400 = insertelement <4 x i32> poison, i32 %399, i64 0, !dbg !15
  %401 = lshr i64 %396, 32, !dbg !15
  %402 = trunc i64 %401 to i32, !dbg !15
  %403 = insertelement <4 x i32> %400, i32 %402, i64 1, !dbg !15
  %404 = insertelement <4 x i32> %403, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc905 = insertelement <4 x i32> %404, i32 131072, i64 3, !dbg !15
  %405 = ptrtoint ptr addrspace(3) %395 to i32, !dbg !15
  %dma.buffer.lds906 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %405), !dbg !15
  %406 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds906, i32 %dma.buffer.offset904, <4 x i32> %dma.buffer.rsrc905) #10, !dbg !15
  %407 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !15
  %408 = zext i32 %407 to i64, !dbg !15
  %409 = inttoptr i64 %408 to ptr addrspace(3), !dbg !15
  %address.local302 = getelementptr [2 x i8], ptr addrspace(1) %392, i64 %address.local150, !dbg !15
  %410 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %411 = ptrtoint ptr addrspace(1) %address.local302 to i64, !dbg !15
  %412 = sub i64 %411, %410, !dbg !15
  %dma.buffer.offset901 = trunc i64 %412 to i32, !dbg !15
  %413 = trunc i64 %410 to i32, !dbg !15
  %414 = insertelement <4 x i32> poison, i32 %413, i64 0, !dbg !15
  %415 = lshr i64 %410, 32, !dbg !15
  %416 = trunc i64 %415 to i32, !dbg !15
  %417 = insertelement <4 x i32> %414, i32 %416, i64 1, !dbg !15
  %418 = insertelement <4 x i32> %417, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc902 = insertelement <4 x i32> %418, i32 131072, i64 3, !dbg !15
  %419 = ptrtoint ptr addrspace(3) %409 to i32, !dbg !15
  %dma.buffer.lds903 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %419), !dbg !15
  %420 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds903, i32 %dma.buffer.offset901, <4 x i32> %dma.buffer.rsrc902) #10, !dbg !15
  %421 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %422 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %423 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %424 = fmul <4 x half> %421, splat (half 0xH3015), !dbg !12
  %425 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %426 = fmul <4 x half> %422, splat (half 0xH3015), !dbg !12
  %427 = fmul <4 x half> %423, splat (half 0xH3015), !dbg !12
  %428 = fmul <4 x half> %425, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %429 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !16
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %424, <4 x half> %429, <4 x float> %282) #9, !dbg !16
  %431 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %432 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !16
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %431, <4 x float> %430) #9, !dbg !16
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %424, <4 x half> %432, <4 x float> %286) #9, !dbg !16
  %435 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !16
  %436 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !16
  %437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %435, <4 x float> %434) #9, !dbg !16
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %424, <4 x half> %436, <4 x float> %290) #9, !dbg !16
  %439 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %440 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !16
  %441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %439, <4 x float> %438) #9, !dbg !16
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %424, <4 x half> %440, <4 x float> %294) #9, !dbg !16
  %443 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !16
  %444 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !16
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %443, <4 x float> %442) #9, !dbg !16
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %424, <4 x half> %444, <4 x float> %298) #9, !dbg !16
  %447 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !16
  %448 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !16
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %447, <4 x float> %446) #9, !dbg !16
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %424, <4 x half> %448, <4 x float> %302) #9, !dbg !16
  %451 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !16
  %452 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !16
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %451, <4 x float> %450) #9, !dbg !16
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %424, <4 x half> %452, <4 x float> %306) #9, !dbg !16
  %455 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !16
  %456 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !16
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %455, <4 x float> %454) #9, !dbg !16
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %424, <4 x half> %456, <4 x float> %310) #9, !dbg !16
  %459 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !16
  %460 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !16
  %461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %459, <4 x float> %458) #9, !dbg !16
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %460, <4 x float> %314) #9, !dbg !16
  %463 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %464 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !16
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %463, <4 x float> %462) #9, !dbg !16
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %464, <4 x float> %318) #9, !dbg !16
  %467 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !16
  %468 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !16
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %467, <4 x float> %466) #9, !dbg !16
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %468, <4 x float> %322) #9, !dbg !16
  %471 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %472 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !16
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %471, <4 x float> %470) #9, !dbg !16
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %472, <4 x float> %326) #9, !dbg !16
  %475 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !16
  %476 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !16
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %475, <4 x float> %474) #9, !dbg !16
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %476, <4 x float> %330) #9, !dbg !16
  %479 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !16
  %480 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !16
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %479, <4 x float> %478) #9, !dbg !16
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %480, <4 x float> %334) #9, !dbg !16
  %483 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !16
  %484 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !16
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %483, <4 x float> %482) #9, !dbg !16
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %484, <4 x float> %338) #9, !dbg !16
  %487 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !16
  %488 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !16
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %487, <4 x float> %486) #9, !dbg !16
  %490 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %488, <4 x float> %341) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %428, <4 x half> %490, <4 x float> %491) #9, !dbg !16
  %493 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %494 = zext i32 %493 to i64, !dbg !11
  %495 = inttoptr i64 %494 to ptr addrspace(3), !dbg !11
  %address.local303 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %address.local138, !dbg !11
  %496 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %497 = ptrtoint ptr addrspace(1) %address.local303 to i64, !dbg !11
  %498 = sub i64 %497, %496, !dbg !11
  %dma.buffer.offset898 = trunc i64 %498 to i32, !dbg !11
  %499 = trunc i64 %496 to i32, !dbg !11
  %500 = insertelement <4 x i32> poison, i32 %499, i64 0, !dbg !11
  %501 = lshr i64 %496, 32, !dbg !11
  %502 = trunc i64 %501 to i32, !dbg !11
  %503 = insertelement <4 x i32> %500, i32 %502, i64 1, !dbg !11
  %504 = insertelement <4 x i32> %503, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc899 = insertelement <4 x i32> %504, i32 131072, i64 3, !dbg !11
  %505 = ptrtoint ptr addrspace(3) %495 to i32, !dbg !11
  %dma.buffer.lds900 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %505), !dbg !11
  %506 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds900, i32 %dma.buffer.offset898, <4 x i32> %dma.buffer.rsrc899) #10, !dbg !11
  %507 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %508 = zext i32 %507 to i64, !dbg !11
  %509 = inttoptr i64 %508 to ptr addrspace(3), !dbg !11
  %address.local304 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %address.local150, !dbg !11
  %510 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %511 = ptrtoint ptr addrspace(1) %address.local304 to i64, !dbg !11
  %512 = sub i64 %511, %510, !dbg !11
  %dma.buffer.offset895 = trunc i64 %512 to i32, !dbg !11
  %513 = trunc i64 %510 to i32, !dbg !11
  %514 = insertelement <4 x i32> poison, i32 %513, i64 0, !dbg !11
  %515 = lshr i64 %510, 32, !dbg !11
  %516 = trunc i64 %515 to i32, !dbg !11
  %517 = insertelement <4 x i32> %514, i32 %516, i64 1, !dbg !11
  %518 = insertelement <4 x i32> %517, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc896 = insertelement <4 x i32> %518, i32 131072, i64 3, !dbg !11
  %519 = ptrtoint ptr addrspace(3) %509 to i32, !dbg !11
  %dma.buffer.lds897 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %519), !dbg !11
  %520 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds897, i32 %dma.buffer.offset895, <4 x i32> %dma.buffer.rsrc896) #10, !dbg !11
  %521 = add nuw i64 %41, 3072, !dbg !15
  %522 = icmp slt i64 %521, 0, !dbg !15
  %523 = sub nuw i64 -3073, %41, !dbg !15
  %524 = select i1 %522, i64 %523, i64 %521, !dbg !15
  %525 = sdiv i64 %524, 4096, !dbg !15
  %.lobit6 = ashr i64 %521, 63, !dbg !15
  %526 = xor i64 %525, %.lobit6, !dbg !15
  %527 = add nsw i64 %526, %8, !dbg !15
  %528 = add nsw i64 %225, 96, !dbg !15
  %529 = icmp slt i64 %225, -96, !dbg !15
  %530 = sub nsw i64 -97, %225, !dbg !15
  %531 = select i1 %529, i64 %530, i64 %528, !dbg !15
  %532 = sdiv i64 %531, 128, !dbg !15
  %533 = sext i1 %529 to i64, !dbg !15
  %534 = xor i64 %532, %533, !dbg !15
  %535 = shl nsw i64 %534, 7, !dbg !15
  %536 = sub i64 %35, %535, !dbg !15
  %537 = shl i64 %527, 19, !dbg !15
  %538 = and i64 %537, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %539 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %538, !dbg !15
  %540 = getelementptr [2 x i8], ptr addrspace(1) %539, i64 %80, !dbg !15
  %541 = getelementptr [2 x i8], ptr addrspace(1) %540, i64 %536, !dbg !15
  %542 = getelementptr i8, ptr addrspace(1) %541, i64 192, !dbg !15
  %543 = getelementptr [2 x i8], ptr addrspace(1) %542, i64 %86, !dbg !15
  %544 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !15
  %545 = zext i32 %544 to i64, !dbg !15
  %546 = inttoptr i64 %545 to ptr addrspace(3), !dbg !15
  %address.local305 = getelementptr [2 x i8], ptr addrspace(1) %543, i64 %address.local138, !dbg !15
  %547 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %548 = ptrtoint ptr addrspace(1) %address.local305 to i64, !dbg !15
  %549 = sub i64 %548, %547, !dbg !15
  %dma.buffer.offset892 = trunc i64 %549 to i32, !dbg !15
  %550 = trunc i64 %547 to i32, !dbg !15
  %551 = insertelement <4 x i32> poison, i32 %550, i64 0, !dbg !15
  %552 = lshr i64 %547, 32, !dbg !15
  %553 = trunc i64 %552 to i32, !dbg !15
  %554 = insertelement <4 x i32> %551, i32 %553, i64 1, !dbg !15
  %555 = insertelement <4 x i32> %554, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc893 = insertelement <4 x i32> %555, i32 131072, i64 3, !dbg !15
  %556 = ptrtoint ptr addrspace(3) %546 to i32, !dbg !15
  %dma.buffer.lds894 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %556), !dbg !15
  %557 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds894, i32 %dma.buffer.offset892, <4 x i32> %dma.buffer.rsrc893) #10, !dbg !15
  %558 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !15
  %559 = zext i32 %558 to i64, !dbg !15
  %560 = inttoptr i64 %559 to ptr addrspace(3), !dbg !15
  %address.local306 = getelementptr [2 x i8], ptr addrspace(1) %543, i64 %address.local150, !dbg !15
  %561 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %562 = ptrtoint ptr addrspace(1) %address.local306 to i64, !dbg !15
  %563 = sub i64 %562, %561, !dbg !15
  %dma.buffer.offset889 = trunc i64 %563 to i32, !dbg !15
  %564 = trunc i64 %561 to i32, !dbg !15
  %565 = insertelement <4 x i32> poison, i32 %564, i64 0, !dbg !15
  %566 = lshr i64 %561, 32, !dbg !15
  %567 = trunc i64 %566 to i32, !dbg !15
  %568 = insertelement <4 x i32> %565, i32 %567, i64 1, !dbg !15
  %569 = insertelement <4 x i32> %568, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc890 = insertelement <4 x i32> %569, i32 131072, i64 3, !dbg !15
  %570 = ptrtoint ptr addrspace(3) %560 to i32, !dbg !15
  %dma.buffer.lds891 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %570), !dbg !15
  %571 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds891, i32 %dma.buffer.offset889, <4 x i32> %dma.buffer.rsrc890) #10, !dbg !15
  %572 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %573 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %574 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %575 = fmul <4 x half> %572, splat (half 0xH3015), !dbg !12
  %576 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %577 = fmul <4 x half> %573, splat (half 0xH3015), !dbg !12
  %578 = fmul <4 x half> %574, splat (half 0xH3015), !dbg !12
  %579 = fmul <4 x half> %576, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %580 = add nsw i64 %80, %8, !dbg !17
  %581 = shl i64 %580, 19, !dbg !17
  %582 = and i64 %581, 16252928, !dbg !17
  %583 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %582, !dbg !17
  %584 = getelementptr [2 x i8], ptr addrspace(1) %583, i64 %86, !dbg !17
  %585 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !17
  %586 = zext i32 %585 to i64, !dbg !17
  %587 = inttoptr i64 %586 to ptr addrspace(3), !dbg !17
  %address.local307 = shl nuw nsw i64 %address.local, 5, !dbg !17
  %address.local308 = and i64 %address.local307, 8064, !dbg !17
  %address.local309 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local310 = and i64 %address.local309, 24, !dbg !17
  %address.local311 = or disjoint i64 %address.local308, %address.local310, !dbg !17
  %address.local312 = getelementptr [2 x i8], ptr addrspace(1) %584, i64 %address.local311, !dbg !17
  %588 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %589 = ptrtoint ptr addrspace(1) %address.local312 to i64, !dbg !17
  %590 = sub i64 %589, %588, !dbg !17
  %dma.buffer.offset886 = trunc i64 %590 to i32, !dbg !17
  %591 = trunc i64 %588 to i32, !dbg !17
  %592 = insertelement <4 x i32> poison, i32 %591, i64 0, !dbg !17
  %593 = lshr i64 %588, 32, !dbg !17
  %594 = trunc i64 %593 to i32, !dbg !17
  %595 = insertelement <4 x i32> %592, i32 %594, i64 1, !dbg !17
  %596 = insertelement <4 x i32> %595, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc887 = insertelement <4 x i32> %596, i32 131072, i64 3, !dbg !17
  %597 = ptrtoint ptr addrspace(3) %587 to i32, !dbg !17
  %dma.buffer.lds888 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %597), !dbg !17
  %598 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds888, i32 %dma.buffer.offset886, <4 x i32> %dma.buffer.rsrc887) #10, !dbg !17
  %599 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !17
  %600 = zext i32 %599 to i64, !dbg !17
  %601 = inttoptr i64 %600 to ptr addrspace(3), !dbg !17
  %address.local313 = shl nuw nsw i64 %address.local140, 5, !dbg !17
  %address.local314 = and i64 %address.local313, 16256, !dbg !17
  %address.local315 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local316 = and i64 %address.local315, 24, !dbg !17
  %address.local317 = or disjoint i64 %address.local314, %address.local316, !dbg !17
  %address.local318 = getelementptr [2 x i8], ptr addrspace(1) %584, i64 %address.local317, !dbg !17
  %602 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %603 = ptrtoint ptr addrspace(1) %address.local318 to i64, !dbg !17
  %604 = sub i64 %603, %602, !dbg !17
  %dma.buffer.offset883 = trunc i64 %604 to i32, !dbg !17
  %605 = trunc i64 %602 to i32, !dbg !17
  %606 = insertelement <4 x i32> poison, i32 %605, i64 0, !dbg !17
  %607 = lshr i64 %602, 32, !dbg !17
  %608 = trunc i64 %607 to i32, !dbg !17
  %609 = insertelement <4 x i32> %606, i32 %608, i64 1, !dbg !17
  %610 = insertelement <4 x i32> %609, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc884 = insertelement <4 x i32> %610, i32 131072, i64 3, !dbg !17
  %611 = ptrtoint ptr addrspace(3) %601 to i32, !dbg !17
  %dma.buffer.lds885 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %611), !dbg !17
  %612 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds885, i32 %dma.buffer.offset883, <4 x i32> %dma.buffer.rsrc884) #10, !dbg !17
  %613 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !16
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %575, <4 x half> %613, <4 x float> %433) #9, !dbg !16
  %615 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %616 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !16
  %617 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %615, <4 x float> %614) #9, !dbg !16
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %575, <4 x half> %616, <4 x float> %437) #9, !dbg !16
  %619 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !16
  %620 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !16
  %621 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %619, <4 x float> %618) #9, !dbg !16
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %575, <4 x half> %620, <4 x float> %441) #9, !dbg !16
  %623 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %624 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !16
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %623, <4 x float> %622) #9, !dbg !16
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %575, <4 x half> %624, <4 x float> %445) #9, !dbg !16
  %627 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !16
  %628 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !16
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %627, <4 x float> %626) #9, !dbg !16
  %630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %575, <4 x half> %628, <4 x float> %449) #9, !dbg !16
  %631 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !16
  %632 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !16
  %633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %631, <4 x float> %630) #9, !dbg !16
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %575, <4 x half> %632, <4 x float> %453) #9, !dbg !16
  %635 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !16
  %636 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !16
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %635, <4 x float> %634) #9, !dbg !16
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %575, <4 x half> %636, <4 x float> %457) #9, !dbg !16
  %639 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !16
  %640 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !16
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %639, <4 x float> %638) #9, !dbg !16
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %575, <4 x half> %640, <4 x float> %461) #9, !dbg !16
  %643 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !16
  %644 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !16
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %643, <4 x float> %642) #9, !dbg !16
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %644, <4 x float> %465) #9, !dbg !16
  %647 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !16
  %648 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !16
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %579, <4 x half> %647, <4 x float> %646) #9, !dbg !16
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %648, <4 x float> %469) #9, !dbg !16
  %651 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !16
  %652 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !16
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %579, <4 x half> %651, <4 x float> %650) #9, !dbg !16
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %652, <4 x float> %473) #9, !dbg !16
  %655 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !16
  %656 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !16
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %579, <4 x half> %655, <4 x float> %654) #9, !dbg !16
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %656, <4 x float> %477) #9, !dbg !16
  %659 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !16
  %660 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !16
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %579, <4 x half> %659, <4 x float> %658) #9, !dbg !16
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %660, <4 x float> %481) #9, !dbg !16
  %663 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !16
  %664 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !16
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %579, <4 x half> %663, <4 x float> %662) #9, !dbg !16
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %664, <4 x float> %485) #9, !dbg !16
  %667 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !16
  %668 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !16
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %579, <4 x half> %667, <4 x float> %666) #9, !dbg !16
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %668, <4 x float> %489) #9, !dbg !16
  %671 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !16
  %672 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !16
  %673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %579, <4 x half> %671, <4 x float> %670) #9, !dbg !16
  %674 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %672, <4 x float> %492) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %676 = or disjoint i64 %49, 127, !dbg !13
  %677 = icmp uge i64 %11, %676, !dbg !13
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %579, <4 x half> %674, <4 x float> %675) #9, !dbg !16
  %679 = and i1 %29, %677, !dbg !13
  br i1 %679, label %680, label %681, !dbg !13

680:                                              ; preds = %48
  br label %825, !dbg !13

681:                                              ; preds = %48
  %682 = add nsw i64 %11, %25, !dbg !13
  %683 = shl nuw nsw i64 %26, 4, !dbg !13
  %684 = add nsw i64 %682, %683, !dbg !13
  %685 = and i32 %24, 240, !dbg !13
  %index.compact = zext nneg i32 %685 to i64, !dbg !13
  %686 = sub nsw i64 %684, %index.compact, !dbg !13
  %687 = and i64 %25, 48, !dbg !13
  %688 = or disjoint i64 %49, %687, !dbg !13
  %689 = add nsw i64 %686, 1, !dbg !22
  %690 = lshr i64 %689, 32, !dbg !23
  %compare.high = trunc i64 %690 to i32, !dbg !23
  %691 = lshr i64 %688, 32, !dbg !23
  %compare.high612 = trunc i64 %691 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high612, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high612, !dbg !23
  %compare.low.base = trunc i64 %689 to i32, !dbg !23
  %compare.low.base613 = trunc i64 %688 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base613, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %692 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low614 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result615 = select i1 %compare.high.equal, i1 %compare.low614, i1 %compare.high.order, !dbg !23
  %693 = select i1 %compare.result615, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base616 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local617 = or i32 %compare.local.base616, 2, !dbg !23
  %compare.low618 = icmp ugt i32 %compare.low.base, %compare.local617, !dbg !23
  %compare.result619 = select i1 %compare.high.equal, i1 %compare.low618, i1 %compare.high.order, !dbg !23
  %694 = select i1 %compare.result619, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base620 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local621 = or i32 %compare.local.base620, 3, !dbg !23
  %compare.low622 = icmp ugt i32 %compare.low.base, %compare.local621, !dbg !23
  %compare.result623 = select i1 %compare.high.equal, i1 %compare.low622, i1 %compare.high.order, !dbg !23
  %695 = select i1 %compare.result623, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base624 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local625 = or i32 %compare.local.base624, 4, !dbg !23
  %compare.low626 = icmp ugt i32 %compare.low.base, %compare.local625, !dbg !23
  %compare.result627 = select i1 %compare.high.equal, i1 %compare.low626, i1 %compare.high.order, !dbg !23
  %696 = select i1 %compare.result627, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base628 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local629 = or i32 %compare.local.base628, 5, !dbg !23
  %compare.low630 = icmp ugt i32 %compare.low.base, %compare.local629, !dbg !23
  %compare.result631 = select i1 %compare.high.equal, i1 %compare.low630, i1 %compare.high.order, !dbg !23
  %697 = select i1 %compare.result631, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base632 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local633 = or i32 %compare.local.base632, 6, !dbg !23
  %compare.low634 = icmp ugt i32 %compare.low.base, %compare.local633, !dbg !23
  %compare.result635 = select i1 %compare.high.equal, i1 %compare.low634, i1 %compare.high.order, !dbg !23
  %698 = select i1 %compare.result635, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base636 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local637 = or i32 %compare.local.base636, 7, !dbg !23
  %compare.low638 = icmp ugt i32 %compare.low.base, %compare.local637, !dbg !23
  %compare.result639 = select i1 %compare.high.equal, i1 %compare.low638, i1 %compare.high.order, !dbg !23
  %699 = select i1 %compare.result639, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base640 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local641 = or i32 %compare.local.base640, 8, !dbg !23
  %compare.low642 = icmp ugt i32 %compare.low.base, %compare.local641, !dbg !23
  %compare.result643 = select i1 %compare.high.equal, i1 %compare.low642, i1 %compare.high.order, !dbg !23
  %700 = select i1 %compare.result643, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base644 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local645 = or i32 %compare.local.base644, 9, !dbg !23
  %compare.low646 = icmp ugt i32 %compare.low.base, %compare.local645, !dbg !23
  %compare.result647 = select i1 %compare.high.equal, i1 %compare.low646, i1 %compare.high.order, !dbg !23
  %701 = select i1 %compare.result647, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base648 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local649 = or i32 %compare.local.base648, 10, !dbg !23
  %compare.low650 = icmp ugt i32 %compare.low.base, %compare.local649, !dbg !23
  %compare.result651 = select i1 %compare.high.equal, i1 %compare.low650, i1 %compare.high.order, !dbg !23
  %702 = select i1 %compare.result651, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base652 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local653 = or i32 %compare.local.base652, 11, !dbg !23
  %compare.low654 = icmp ugt i32 %compare.low.base, %compare.local653, !dbg !23
  %compare.result655 = select i1 %compare.high.equal, i1 %compare.low654, i1 %compare.high.order, !dbg !23
  %703 = select i1 %compare.result655, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base656 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local657 = or i32 %compare.local.base656, 12, !dbg !23
  %compare.low658 = icmp ugt i32 %compare.low.base, %compare.local657, !dbg !23
  %compare.result659 = select i1 %compare.high.equal, i1 %compare.low658, i1 %compare.high.order, !dbg !23
  %704 = select i1 %compare.result659, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base660 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local661 = or i32 %compare.local.base660, 13, !dbg !23
  %compare.low662 = icmp ugt i32 %compare.low.base, %compare.local661, !dbg !23
  %compare.result663 = select i1 %compare.high.equal, i1 %compare.low662, i1 %compare.high.order, !dbg !23
  %705 = select i1 %compare.result663, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base664 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local665 = or i32 %compare.local.base664, 14, !dbg !23
  %compare.low666 = icmp ugt i32 %compare.low.base, %compare.local665, !dbg !23
  %compare.result667 = select i1 %compare.high.equal, i1 %compare.low666, i1 %compare.high.order, !dbg !23
  %706 = select i1 %compare.result667, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base668 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local669 = or i32 %compare.local.base668, 15, !dbg !23
  %compare.low670 = icmp ugt i32 %compare.low.base, %compare.local669, !dbg !23
  %compare.result671 = select i1 %compare.high.equal, i1 %compare.low670, i1 %compare.high.order, !dbg !23
  %707 = select i1 %compare.result671, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base672 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local673 = or i32 %compare.local.base672, 64, !dbg !23
  %compare.low674 = icmp ugt i32 %compare.low.base, %compare.local673, !dbg !23
  %compare.result675 = select i1 %compare.high.equal, i1 %compare.low674, i1 %compare.high.order, !dbg !23
  %708 = select i1 %compare.result675, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base676 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local677 = or i32 %compare.local.base676, 65, !dbg !23
  %compare.low678 = icmp ugt i32 %compare.low.base, %compare.local677, !dbg !23
  %compare.result679 = select i1 %compare.high.equal, i1 %compare.low678, i1 %compare.high.order, !dbg !23
  %709 = select i1 %compare.result679, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base680 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local681 = or i32 %compare.local.base680, 66, !dbg !23
  %compare.low682 = icmp ugt i32 %compare.low.base, %compare.local681, !dbg !23
  %compare.result683 = select i1 %compare.high.equal, i1 %compare.low682, i1 %compare.high.order, !dbg !23
  %710 = select i1 %compare.result683, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base684 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local685 = or i32 %compare.local.base684, 67, !dbg !23
  %compare.low686 = icmp ugt i32 %compare.low.base, %compare.local685, !dbg !23
  %compare.result687 = select i1 %compare.high.equal, i1 %compare.low686, i1 %compare.high.order, !dbg !23
  %711 = select i1 %compare.result687, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base688 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local689 = or i32 %compare.local.base688, 68, !dbg !23
  %compare.low690 = icmp ugt i32 %compare.low.base, %compare.local689, !dbg !23
  %compare.result691 = select i1 %compare.high.equal, i1 %compare.low690, i1 %compare.high.order, !dbg !23
  %712 = select i1 %compare.result691, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base692 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local693 = or i32 %compare.local.base692, 69, !dbg !23
  %compare.low694 = icmp ugt i32 %compare.low.base, %compare.local693, !dbg !23
  %compare.result695 = select i1 %compare.high.equal, i1 %compare.low694, i1 %compare.high.order, !dbg !23
  %713 = select i1 %compare.result695, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base696 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local697 = or i32 %compare.local.base696, 70, !dbg !23
  %compare.low698 = icmp ugt i32 %compare.low.base, %compare.local697, !dbg !23
  %compare.result699 = select i1 %compare.high.equal, i1 %compare.low698, i1 %compare.high.order, !dbg !23
  %714 = select i1 %compare.result699, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base700 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local701 = or i32 %compare.local.base700, 71, !dbg !23
  %compare.low702 = icmp ugt i32 %compare.low.base, %compare.local701, !dbg !23
  %compare.result703 = select i1 %compare.high.equal, i1 %compare.low702, i1 %compare.high.order, !dbg !23
  %715 = select i1 %compare.result703, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base704 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local705 = or i32 %compare.local.base704, 72, !dbg !23
  %compare.low706 = icmp ugt i32 %compare.low.base, %compare.local705, !dbg !23
  %compare.result707 = select i1 %compare.high.equal, i1 %compare.low706, i1 %compare.high.order, !dbg !23
  %716 = select i1 %compare.result707, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base708 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local709 = or i32 %compare.local.base708, 73, !dbg !23
  %compare.low710 = icmp ugt i32 %compare.low.base, %compare.local709, !dbg !23
  %compare.result711 = select i1 %compare.high.equal, i1 %compare.low710, i1 %compare.high.order, !dbg !23
  %717 = select i1 %compare.result711, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base712 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local713 = or i32 %compare.local.base712, 74, !dbg !23
  %compare.low714 = icmp ugt i32 %compare.low.base, %compare.local713, !dbg !23
  %compare.result715 = select i1 %compare.high.equal, i1 %compare.low714, i1 %compare.high.order, !dbg !23
  %718 = select i1 %compare.result715, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base716 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local717 = or i32 %compare.local.base716, 75, !dbg !23
  %compare.low718 = icmp ugt i32 %compare.low.base, %compare.local717, !dbg !23
  %compare.result719 = select i1 %compare.high.equal, i1 %compare.low718, i1 %compare.high.order, !dbg !23
  %719 = select i1 %compare.result719, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base720 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local721 = or i32 %compare.local.base720, 76, !dbg !23
  %compare.low722 = icmp ugt i32 %compare.low.base, %compare.local721, !dbg !23
  %compare.result723 = select i1 %compare.high.equal, i1 %compare.low722, i1 %compare.high.order, !dbg !23
  %720 = select i1 %compare.result723, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base724 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local725 = or i32 %compare.local.base724, 77, !dbg !23
  %compare.low726 = icmp ugt i32 %compare.low.base, %compare.local725, !dbg !23
  %compare.result727 = select i1 %compare.high.equal, i1 %compare.low726, i1 %compare.high.order, !dbg !23
  %721 = select i1 %compare.result727, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base728 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local729 = or i32 %compare.local.base728, 78, !dbg !23
  %compare.low730 = icmp ugt i32 %compare.low.base, %compare.local729, !dbg !23
  %compare.result731 = select i1 %compare.high.equal, i1 %compare.low730, i1 %compare.high.order, !dbg !23
  %722 = select i1 %compare.result731, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base732 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 31) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local733 = or i32 %compare.local.base732, 79, !dbg !23
  %compare.low734 = icmp ugt i32 %compare.low.base, %compare.local733, !dbg !23
  %compare.result735 = select i1 %compare.high.equal, i1 %compare.low734, i1 %compare.high.order, !dbg !23
  %723 = select i1 %compare.result735, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %724 = add nsw i64 %686, 65, !dbg !22
  %725 = lshr i64 %724, 32, !dbg !23
  %compare.high736 = trunc i64 %725 to i32, !dbg !23
  %726 = lshr i64 %688, 32, !dbg !23
  %compare.high737 = trunc i64 %726 to i32, !dbg !23
  %compare.high.equal738 = icmp eq i32 %compare.high736, %compare.high737, !dbg !23
  %compare.high.order739 = icmp ugt i32 %compare.high736, %compare.high737, !dbg !23
  %compare.low.base740 = trunc i64 %724 to i32, !dbg !23
  %compare.low.base741 = trunc i64 %688 to i32, !dbg !23
  %compare.low742 = icmp ugt i32 %compare.low.base740, %compare.low.base741, !dbg !23
  %compare.result743 = select i1 %compare.high.equal738, i1 %compare.low742, i1 %compare.high.order739, !dbg !23
  %727 = select i1 %compare.result743, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base744 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 32) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local745 = or i32 %compare.local.base744, 1, !dbg !23
  %compare.low746 = icmp ugt i32 %compare.low.base740, %compare.local745, !dbg !23
  %compare.result747 = select i1 %compare.high.equal738, i1 %compare.low746, i1 %compare.high.order739, !dbg !23
  %728 = select i1 %compare.result747, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base748 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 33) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local749 = or i32 %compare.local.base748, 2, !dbg !23
  %compare.low750 = icmp ugt i32 %compare.low.base740, %compare.local749, !dbg !23
  %compare.result751 = select i1 %compare.high.equal738, i1 %compare.low750, i1 %compare.high.order739, !dbg !23
  %729 = select i1 %compare.result751, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base752 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 34) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local753 = or i32 %compare.local.base752, 3, !dbg !23
  %compare.low754 = icmp ugt i32 %compare.low.base740, %compare.local753, !dbg !23
  %compare.result755 = select i1 %compare.high.equal738, i1 %compare.low754, i1 %compare.high.order739, !dbg !23
  %730 = select i1 %compare.result755, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base756 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 35) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local757 = or i32 %compare.local.base756, 4, !dbg !23
  %compare.low758 = icmp ugt i32 %compare.low.base740, %compare.local757, !dbg !23
  %compare.result759 = select i1 %compare.high.equal738, i1 %compare.low758, i1 %compare.high.order739, !dbg !23
  %731 = select i1 %compare.result759, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base760 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 36) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local761 = or i32 %compare.local.base760, 5, !dbg !23
  %compare.low762 = icmp ugt i32 %compare.low.base740, %compare.local761, !dbg !23
  %compare.result763 = select i1 %compare.high.equal738, i1 %compare.low762, i1 %compare.high.order739, !dbg !23
  %732 = select i1 %compare.result763, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base764 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 37) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local765 = or i32 %compare.local.base764, 6, !dbg !23
  %compare.low766 = icmp ugt i32 %compare.low.base740, %compare.local765, !dbg !23
  %compare.result767 = select i1 %compare.high.equal738, i1 %compare.low766, i1 %compare.high.order739, !dbg !23
  %733 = select i1 %compare.result767, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base768 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 38) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local769 = or i32 %compare.local.base768, 7, !dbg !23
  %compare.low770 = icmp ugt i32 %compare.low.base740, %compare.local769, !dbg !23
  %compare.result771 = select i1 %compare.high.equal738, i1 %compare.low770, i1 %compare.high.order739, !dbg !23
  %734 = select i1 %compare.result771, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base772 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 39) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local773 = or i32 %compare.local.base772, 8, !dbg !23
  %compare.low774 = icmp ugt i32 %compare.low.base740, %compare.local773, !dbg !23
  %compare.result775 = select i1 %compare.high.equal738, i1 %compare.low774, i1 %compare.high.order739, !dbg !23
  %735 = select i1 %compare.result775, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base776 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 40) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local777 = or i32 %compare.local.base776, 9, !dbg !23
  %compare.low778 = icmp ugt i32 %compare.low.base740, %compare.local777, !dbg !23
  %compare.result779 = select i1 %compare.high.equal738, i1 %compare.low778, i1 %compare.high.order739, !dbg !23
  %736 = select i1 %compare.result779, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base780 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 41) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local781 = or i32 %compare.local.base780, 10, !dbg !23
  %compare.low782 = icmp ugt i32 %compare.low.base740, %compare.local781, !dbg !23
  %compare.result783 = select i1 %compare.high.equal738, i1 %compare.low782, i1 %compare.high.order739, !dbg !23
  %737 = select i1 %compare.result783, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base784 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 42) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local785 = or i32 %compare.local.base784, 11, !dbg !23
  %compare.low786 = icmp ugt i32 %compare.low.base740, %compare.local785, !dbg !23
  %compare.result787 = select i1 %compare.high.equal738, i1 %compare.low786, i1 %compare.high.order739, !dbg !23
  %738 = select i1 %compare.result787, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base788 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 43) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local789 = or i32 %compare.local.base788, 12, !dbg !23
  %compare.low790 = icmp ugt i32 %compare.low.base740, %compare.local789, !dbg !23
  %compare.result791 = select i1 %compare.high.equal738, i1 %compare.low790, i1 %compare.high.order739, !dbg !23
  %739 = select i1 %compare.result791, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base792 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 44) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local793 = or i32 %compare.local.base792, 13, !dbg !23
  %compare.low794 = icmp ugt i32 %compare.low.base740, %compare.local793, !dbg !23
  %compare.result795 = select i1 %compare.high.equal738, i1 %compare.low794, i1 %compare.high.order739, !dbg !23
  %740 = select i1 %compare.result795, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base796 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 45) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local797 = or i32 %compare.local.base796, 14, !dbg !23
  %compare.low798 = icmp ugt i32 %compare.low.base740, %compare.local797, !dbg !23
  %compare.result799 = select i1 %compare.high.equal738, i1 %compare.low798, i1 %compare.high.order739, !dbg !23
  %741 = select i1 %compare.result799, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base800 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 46) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local801 = or i32 %compare.local.base800, 15, !dbg !23
  %compare.low802 = icmp ugt i32 %compare.low.base740, %compare.local801, !dbg !23
  %compare.result803 = select i1 %compare.high.equal738, i1 %compare.low802, i1 %compare.high.order739, !dbg !23
  %742 = select i1 %compare.result803, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base804 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 47) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local805 = or i32 %compare.local.base804, 64, !dbg !23
  %compare.low806 = icmp ugt i32 %compare.low.base740, %compare.local805, !dbg !23
  %compare.result807 = select i1 %compare.high.equal738, i1 %compare.low806, i1 %compare.high.order739, !dbg !23
  %743 = select i1 %compare.result807, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base808 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 48) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local809 = or i32 %compare.local.base808, 65, !dbg !23
  %compare.low810 = icmp ugt i32 %compare.low.base740, %compare.local809, !dbg !23
  %compare.result811 = select i1 %compare.high.equal738, i1 %compare.low810, i1 %compare.high.order739, !dbg !23
  %744 = select i1 %compare.result811, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base812 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 49) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local813 = or i32 %compare.local.base812, 66, !dbg !23
  %compare.low814 = icmp ugt i32 %compare.low.base740, %compare.local813, !dbg !23
  %compare.result815 = select i1 %compare.high.equal738, i1 %compare.low814, i1 %compare.high.order739, !dbg !23
  %745 = select i1 %compare.result815, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base816 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 50) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local817 = or i32 %compare.local.base816, 67, !dbg !23
  %compare.low818 = icmp ugt i32 %compare.low.base740, %compare.local817, !dbg !23
  %compare.result819 = select i1 %compare.high.equal738, i1 %compare.low818, i1 %compare.high.order739, !dbg !23
  %746 = select i1 %compare.result819, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base820 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 51) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local821 = or i32 %compare.local.base820, 68, !dbg !23
  %compare.low822 = icmp ugt i32 %compare.low.base740, %compare.local821, !dbg !23
  %compare.result823 = select i1 %compare.high.equal738, i1 %compare.low822, i1 %compare.high.order739, !dbg !23
  %747 = select i1 %compare.result823, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base824 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 52) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local825 = or i32 %compare.local.base824, 69, !dbg !23
  %compare.low826 = icmp ugt i32 %compare.low.base740, %compare.local825, !dbg !23
  %compare.result827 = select i1 %compare.high.equal738, i1 %compare.low826, i1 %compare.high.order739, !dbg !23
  %748 = select i1 %compare.result827, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base828 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 53) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local829 = or i32 %compare.local.base828, 70, !dbg !23
  %compare.low830 = icmp ugt i32 %compare.low.base740, %compare.local829, !dbg !23
  %compare.result831 = select i1 %compare.high.equal738, i1 %compare.low830, i1 %compare.high.order739, !dbg !23
  %749 = select i1 %compare.result831, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base832 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 54) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local833 = or i32 %compare.local.base832, 71, !dbg !23
  %compare.low834 = icmp ugt i32 %compare.low.base740, %compare.local833, !dbg !23
  %compare.result835 = select i1 %compare.high.equal738, i1 %compare.low834, i1 %compare.high.order739, !dbg !23
  %750 = select i1 %compare.result835, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base836 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 55) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local837 = or i32 %compare.local.base836, 72, !dbg !23
  %compare.low838 = icmp ugt i32 %compare.low.base740, %compare.local837, !dbg !23
  %compare.result839 = select i1 %compare.high.equal738, i1 %compare.low838, i1 %compare.high.order739, !dbg !23
  %751 = select i1 %compare.result839, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base840 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 56) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local841 = or i32 %compare.local.base840, 73, !dbg !23
  %compare.low842 = icmp ugt i32 %compare.low.base740, %compare.local841, !dbg !23
  %compare.result843 = select i1 %compare.high.equal738, i1 %compare.low842, i1 %compare.high.order739, !dbg !23
  %752 = select i1 %compare.result843, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base844 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 57) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local845 = or i32 %compare.local.base844, 74, !dbg !23
  %compare.low846 = icmp ugt i32 %compare.low.base740, %compare.local845, !dbg !23
  %compare.result847 = select i1 %compare.high.equal738, i1 %compare.low846, i1 %compare.high.order739, !dbg !23
  %753 = select i1 %compare.result847, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base848 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 58) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local849 = or i32 %compare.local.base848, 75, !dbg !23
  %compare.low850 = icmp ugt i32 %compare.low.base740, %compare.local849, !dbg !23
  %compare.result851 = select i1 %compare.high.equal738, i1 %compare.low850, i1 %compare.high.order739, !dbg !23
  %754 = select i1 %compare.result851, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base852 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 59) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local853 = or i32 %compare.local.base852, 76, !dbg !23
  %compare.low854 = icmp ugt i32 %compare.low.base740, %compare.local853, !dbg !23
  %compare.result855 = select i1 %compare.high.equal738, i1 %compare.low854, i1 %compare.high.order739, !dbg !23
  %755 = select i1 %compare.result855, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base856 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 60) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local857 = or i32 %compare.local.base856, 77, !dbg !23
  %compare.low858 = icmp ugt i32 %compare.low.base740, %compare.local857, !dbg !23
  %compare.result859 = select i1 %compare.high.equal738, i1 %compare.low858, i1 %compare.high.order739, !dbg !23
  %756 = select i1 %compare.result859, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base860 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 61) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local861 = or i32 %compare.local.base860, 78, !dbg !23
  %compare.low862 = icmp ugt i32 %compare.low.base740, %compare.local861, !dbg !23
  %compare.result863 = select i1 %compare.high.equal738, i1 %compare.low862, i1 %compare.high.order739, !dbg !23
  %757 = select i1 %compare.result863, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base864 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 62) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local865 = or i32 %compare.local.base864, 79, !dbg !23
  %compare.low866 = icmp ugt i32 %compare.low.base740, %compare.local865, !dbg !23
  %compare.result867 = select i1 %compare.high.equal738, i1 %compare.low866, i1 %compare.high.order739, !dbg !23
  %758 = select i1 %compare.result867, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %759 = insertelement <32 x float> poison, float %692, i64 0, !dbg !13
  %760 = insertelement <32 x float> %759, float %693, i64 1, !dbg !13
  %761 = insertelement <32 x float> %760, float %694, i64 2, !dbg !13
  %762 = insertelement <32 x float> %761, float %695, i64 3, !dbg !13
  %763 = insertelement <32 x float> %762, float %696, i64 4, !dbg !13
  %764 = insertelement <32 x float> %763, float %697, i64 5, !dbg !13
  %765 = insertelement <32 x float> %764, float %698, i64 6, !dbg !13
  %766 = insertelement <32 x float> %765, float %699, i64 7, !dbg !13
  %767 = insertelement <32 x float> %766, float %700, i64 8, !dbg !13
  %768 = insertelement <32 x float> %767, float %701, i64 9, !dbg !13
  %769 = insertelement <32 x float> %768, float %702, i64 10, !dbg !13
  %770 = insertelement <32 x float> %769, float %703, i64 11, !dbg !13
  %771 = insertelement <32 x float> %770, float %704, i64 12, !dbg !13
  %772 = insertelement <32 x float> %771, float %705, i64 13, !dbg !13
  %773 = insertelement <32 x float> %772, float %706, i64 14, !dbg !13
  %774 = insertelement <32 x float> %773, float %707, i64 15, !dbg !13
  %775 = insertelement <32 x float> %774, float %708, i64 16, !dbg !13
  %776 = insertelement <32 x float> %775, float %709, i64 17, !dbg !13
  %777 = insertelement <32 x float> %776, float %710, i64 18, !dbg !13
  %778 = insertelement <32 x float> %777, float %711, i64 19, !dbg !13
  %779 = insertelement <32 x float> %778, float %712, i64 20, !dbg !13
  %780 = insertelement <32 x float> %779, float %713, i64 21, !dbg !13
  %781 = insertelement <32 x float> %780, float %714, i64 22, !dbg !13
  %782 = insertelement <32 x float> %781, float %715, i64 23, !dbg !13
  %783 = insertelement <32 x float> %782, float %716, i64 24, !dbg !13
  %784 = insertelement <32 x float> %783, float %717, i64 25, !dbg !13
  %785 = insertelement <32 x float> %784, float %718, i64 26, !dbg !13
  %786 = insertelement <32 x float> %785, float %719, i64 27, !dbg !13
  %787 = insertelement <32 x float> %786, float %720, i64 28, !dbg !13
  %788 = insertelement <32 x float> %787, float %721, i64 29, !dbg !13
  %789 = insertelement <32 x float> %788, float %722, i64 30, !dbg !13
  %790 = insertelement <32 x float> %789, float %723, i64 31, !dbg !13
  %791 = insertvalue [2 x <32 x float>] poison, <32 x float> %790, 0, !dbg !13
  %792 = insertelement <32 x float> poison, float %727, i64 0, !dbg !13
  %793 = insertelement <32 x float> %792, float %728, i64 1, !dbg !13
  %794 = insertelement <32 x float> %793, float %729, i64 2, !dbg !13
  %795 = insertelement <32 x float> %794, float %730, i64 3, !dbg !13
  %796 = insertelement <32 x float> %795, float %731, i64 4, !dbg !13
  %797 = insertelement <32 x float> %796, float %732, i64 5, !dbg !13
  %798 = insertelement <32 x float> %797, float %733, i64 6, !dbg !13
  %799 = insertelement <32 x float> %798, float %734, i64 7, !dbg !13
  %800 = insertelement <32 x float> %799, float %735, i64 8, !dbg !13
  %801 = insertelement <32 x float> %800, float %736, i64 9, !dbg !13
  %802 = insertelement <32 x float> %801, float %737, i64 10, !dbg !13
  %803 = insertelement <32 x float> %802, float %738, i64 11, !dbg !13
  %804 = insertelement <32 x float> %803, float %739, i64 12, !dbg !13
  %805 = insertelement <32 x float> %804, float %740, i64 13, !dbg !13
  %806 = insertelement <32 x float> %805, float %741, i64 14, !dbg !13
  %807 = insertelement <32 x float> %806, float %742, i64 15, !dbg !13
  %808 = insertelement <32 x float> %807, float %743, i64 16, !dbg !13
  %809 = insertelement <32 x float> %808, float %744, i64 17, !dbg !13
  %810 = insertelement <32 x float> %809, float %745, i64 18, !dbg !13
  %811 = insertelement <32 x float> %810, float %746, i64 19, !dbg !13
  %812 = insertelement <32 x float> %811, float %747, i64 20, !dbg !13
  %813 = insertelement <32 x float> %812, float %748, i64 21, !dbg !13
  %814 = insertelement <32 x float> %813, float %749, i64 22, !dbg !13
  %815 = insertelement <32 x float> %814, float %750, i64 23, !dbg !13
  %816 = insertelement <32 x float> %815, float %751, i64 24, !dbg !13
  %817 = insertelement <32 x float> %816, float %752, i64 25, !dbg !13
  %818 = insertelement <32 x float> %817, float %753, i64 26, !dbg !13
  %819 = insertelement <32 x float> %818, float %754, i64 27, !dbg !13
  %820 = insertelement <32 x float> %819, float %755, i64 28, !dbg !13
  %821 = insertelement <32 x float> %820, float %756, i64 29, !dbg !13
  %822 = insertelement <32 x float> %821, float %757, i64 30, !dbg !13
  %823 = insertelement <32 x float> %822, float %758, i64 31, !dbg !13
  %824 = insertvalue [2 x <32 x float>] %791, <32 x float> %823, 1, !dbg !13
  br label %825, !dbg !13

825:                                              ; preds = %680, %681
  %826 = phi [2 x <32 x float>] [ %824, %681 ], [ zeroinitializer, %680 ], !dbg !13
  br label %827, !dbg !13

827:                                              ; preds = %825
  %828 = extractvalue [2 x <32 x float>] %826, 0, !dbg !25
  %829 = shufflevector <32 x float> %828, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %830 = fadd <4 x float> %617, %829, !dbg !25
  %831 = extractelement <4 x float> %830, i64 0, !dbg !25
  %832 = extractelement <4 x float> %830, i64 1, !dbg !25
  %833 = extractelement <4 x float> %830, i64 2, !dbg !25
  %834 = extractelement <4 x float> %830, i64 3, !dbg !25
  %835 = shufflevector <32 x float> %828, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %836 = fadd <4 x float> %621, %835, !dbg !25
  %837 = extractelement <4 x float> %836, i64 0, !dbg !25
  %838 = extractelement <4 x float> %836, i64 1, !dbg !25
  %839 = extractelement <4 x float> %836, i64 2, !dbg !25
  %840 = extractelement <4 x float> %836, i64 3, !dbg !25
  %841 = shufflevector <32 x float> %828, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %842 = fadd <4 x float> %625, %841, !dbg !25
  %843 = extractelement <4 x float> %842, i64 0, !dbg !25
  %844 = extractelement <4 x float> %842, i64 1, !dbg !25
  %845 = extractelement <4 x float> %842, i64 2, !dbg !25
  %846 = extractelement <4 x float> %842, i64 3, !dbg !25
  %847 = shufflevector <32 x float> %828, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %848 = fadd <4 x float> %629, %847, !dbg !25
  %849 = extractelement <4 x float> %848, i64 0, !dbg !25
  %850 = extractelement <4 x float> %848, i64 1, !dbg !25
  %851 = extractelement <4 x float> %848, i64 2, !dbg !25
  %852 = extractelement <4 x float> %848, i64 3, !dbg !25
  %853 = shufflevector <32 x float> %828, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %854 = fadd <4 x float> %633, %853, !dbg !25
  %855 = extractelement <4 x float> %854, i64 0, !dbg !25
  %856 = extractelement <4 x float> %854, i64 1, !dbg !25
  %857 = extractelement <4 x float> %854, i64 2, !dbg !25
  %858 = extractelement <4 x float> %854, i64 3, !dbg !25
  %859 = shufflevector <32 x float> %828, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %860 = fadd <4 x float> %637, %859, !dbg !25
  %861 = extractelement <4 x float> %860, i64 0, !dbg !25
  %862 = extractelement <4 x float> %860, i64 1, !dbg !25
  %863 = extractelement <4 x float> %860, i64 2, !dbg !25
  %864 = extractelement <4 x float> %860, i64 3, !dbg !25
  %865 = shufflevector <32 x float> %828, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %866 = fadd <4 x float> %641, %865, !dbg !25
  %867 = extractelement <4 x float> %866, i64 0, !dbg !25
  %868 = extractelement <4 x float> %866, i64 1, !dbg !25
  %869 = extractelement <4 x float> %866, i64 2, !dbg !25
  %870 = extractelement <4 x float> %866, i64 3, !dbg !25
  %871 = shufflevector <32 x float> %828, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %872 = fadd <4 x float> %645, %871, !dbg !25
  %873 = extractelement <4 x float> %872, i64 0, !dbg !25
  %874 = extractelement <4 x float> %872, i64 1, !dbg !25
  %875 = extractelement <4 x float> %872, i64 2, !dbg !25
  %876 = extractelement <4 x float> %872, i64 3, !dbg !25
  %877 = extractvalue [2 x <32 x float>] %826, 1, !dbg !25
  %878 = shufflevector <32 x float> %877, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %879 = fadd <4 x float> %649, %878, !dbg !25
  %880 = extractelement <4 x float> %879, i64 0, !dbg !25
  %881 = extractelement <4 x float> %879, i64 1, !dbg !25
  %882 = extractelement <4 x float> %879, i64 2, !dbg !25
  %883 = extractelement <4 x float> %879, i64 3, !dbg !25
  %884 = shufflevector <32 x float> %877, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %885 = fadd <4 x float> %653, %884, !dbg !25
  %886 = extractelement <4 x float> %885, i64 0, !dbg !25
  %887 = extractelement <4 x float> %885, i64 1, !dbg !25
  %888 = extractelement <4 x float> %885, i64 2, !dbg !25
  %889 = extractelement <4 x float> %885, i64 3, !dbg !25
  %890 = shufflevector <32 x float> %877, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %891 = fadd <4 x float> %657, %890, !dbg !25
  %892 = extractelement <4 x float> %891, i64 0, !dbg !25
  %893 = extractelement <4 x float> %891, i64 1, !dbg !25
  %894 = extractelement <4 x float> %891, i64 2, !dbg !25
  %895 = extractelement <4 x float> %891, i64 3, !dbg !25
  %896 = shufflevector <32 x float> %877, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %897 = fadd <4 x float> %661, %896, !dbg !25
  %898 = extractelement <4 x float> %897, i64 0, !dbg !25
  %899 = extractelement <4 x float> %897, i64 1, !dbg !25
  %900 = extractelement <4 x float> %897, i64 2, !dbg !25
  %901 = extractelement <4 x float> %897, i64 3, !dbg !25
  %902 = shufflevector <32 x float> %877, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %903 = fadd <4 x float> %665, %902, !dbg !25
  %904 = extractelement <4 x float> %903, i64 0, !dbg !25
  %905 = extractelement <4 x float> %903, i64 1, !dbg !25
  %906 = extractelement <4 x float> %903, i64 2, !dbg !25
  %907 = extractelement <4 x float> %903, i64 3, !dbg !25
  %908 = shufflevector <32 x float> %877, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %909 = fadd <4 x float> %669, %908, !dbg !25
  %910 = extractelement <4 x float> %909, i64 0, !dbg !25
  %911 = extractelement <4 x float> %909, i64 1, !dbg !25
  %912 = extractelement <4 x float> %909, i64 2, !dbg !25
  %913 = extractelement <4 x float> %909, i64 3, !dbg !25
  %914 = shufflevector <32 x float> %877, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %915 = fadd <4 x float> %673, %914, !dbg !25
  %916 = extractelement <4 x float> %915, i64 0, !dbg !25
  %917 = extractelement <4 x float> %915, i64 1, !dbg !25
  %918 = extractelement <4 x float> %915, i64 2, !dbg !25
  %919 = extractelement <4 x float> %915, i64 3, !dbg !25
  %920 = shufflevector <32 x float> %877, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %921 = fadd <4 x float> %678, %920, !dbg !25
  %922 = extractelement <4 x float> %921, i64 0, !dbg !25
  %923 = extractelement <4 x float> %921, i64 1, !dbg !25
  %924 = extractelement <4 x float> %921, i64 2, !dbg !25
  %925 = extractelement <4 x float> %921, i64 3, !dbg !25
  %926 = call float @llvm.maxnum.f32(float %831, float 0xFFF0000000000000), !dbg !26
  %927 = call float @llvm.maxnum.f32(float %926, float %832), !dbg !26
  %928 = call float @llvm.maxnum.f32(float %927, float %833), !dbg !26
  %929 = call float @llvm.maxnum.f32(float %928, float %834), !dbg !26
  %930 = call float @llvm.maxnum.f32(float %929, float %837), !dbg !26
  %931 = call float @llvm.maxnum.f32(float %930, float %838), !dbg !26
  %932 = call float @llvm.maxnum.f32(float %931, float %839), !dbg !26
  %933 = call float @llvm.maxnum.f32(float %932, float %840), !dbg !26
  %934 = call float @llvm.maxnum.f32(float %933, float %843), !dbg !26
  %935 = call float @llvm.maxnum.f32(float %934, float %844), !dbg !26
  %936 = call float @llvm.maxnum.f32(float %935, float %845), !dbg !26
  %937 = call float @llvm.maxnum.f32(float %936, float %846), !dbg !26
  %938 = call float @llvm.maxnum.f32(float %937, float %849), !dbg !26
  %939 = call float @llvm.maxnum.f32(float %938, float %850), !dbg !26
  %940 = call float @llvm.maxnum.f32(float %939, float %851), !dbg !26
  %941 = call float @llvm.maxnum.f32(float %940, float %852), !dbg !26
  %942 = call float @llvm.maxnum.f32(float %941, float %855), !dbg !26
  %943 = call float @llvm.maxnum.f32(float %942, float %856), !dbg !26
  %944 = call float @llvm.maxnum.f32(float %943, float %857), !dbg !26
  %945 = call float @llvm.maxnum.f32(float %944, float %858), !dbg !26
  %946 = call float @llvm.maxnum.f32(float %945, float %861), !dbg !26
  %947 = call float @llvm.maxnum.f32(float %946, float %862), !dbg !26
  %948 = call float @llvm.maxnum.f32(float %947, float %863), !dbg !26
  %949 = call float @llvm.maxnum.f32(float %948, float %864), !dbg !26
  %950 = call float @llvm.maxnum.f32(float %949, float %867), !dbg !26
  %951 = call float @llvm.maxnum.f32(float %950, float %868), !dbg !26
  %952 = call float @llvm.maxnum.f32(float %951, float %869), !dbg !26
  %953 = call float @llvm.maxnum.f32(float %952, float %870), !dbg !26
  %954 = call float @llvm.maxnum.f32(float %953, float %873), !dbg !26
  %955 = call float @llvm.maxnum.f32(float %954, float %874), !dbg !26
  %956 = call float @llvm.maxnum.f32(float %955, float %875), !dbg !26
  %957 = call float @llvm.maxnum.f32(float %956, float %876), !dbg !26
  %958 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %959 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %958), !dbg !26
  %960 = add nuw nsw i32 %959, 64, !dbg !26
  %961 = xor i32 %959, 16, !dbg !26
  %962 = and i32 %960, 192, !dbg !26
  %963 = icmp samesign ult i32 %961, %962, !dbg !26
  %964 = select i1 %963, i32 %961, i32 %959, !dbg !26
  %965 = shl nuw nsw i32 %964, 2, !dbg !26
  %966 = bitcast float %957 to i32, !dbg !26
  %967 = call i32 @llvm.amdgcn.ds.bpermute(i32 %965, i32 %966), !dbg !26
  %968 = bitcast i32 %967 to float, !dbg !26
  %969 = call float @llvm.maxnum.f32(float %957, float %968), !dbg !26
  %970 = xor i32 %959, 32, !dbg !26
  %971 = icmp samesign ult i32 %970, %962, !dbg !26
  %972 = select i1 %971, i32 %970, i32 %959, !dbg !26
  %973 = shl nuw nsw i32 %972, 2, !dbg !26
  %974 = bitcast float %969 to i32, !dbg !26
  %975 = call i32 @llvm.amdgcn.ds.bpermute(i32 %973, i32 %974), !dbg !26
  %976 = bitcast i32 %975 to float, !dbg !26
  %977 = call float @llvm.maxnum.f32(float %969, float %976), !dbg !26
  %978 = extractvalue [2 x <1 x float>] %44, 0, !dbg !26
  %979 = insertelement <1 x float> poison, float %977, i64 0, !dbg !26
  %980 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %979, <1 x float> %978), !dbg !26
  %981 = call float @llvm.maxnum.f32(float %880, float 0xFFF0000000000000), !dbg !26
  %982 = call float @llvm.maxnum.f32(float %981, float %881), !dbg !26
  %983 = call float @llvm.maxnum.f32(float %982, float %882), !dbg !26
  %984 = call float @llvm.maxnum.f32(float %983, float %883), !dbg !26
  %985 = call float @llvm.maxnum.f32(float %984, float %886), !dbg !26
  %986 = call float @llvm.maxnum.f32(float %985, float %887), !dbg !26
  %987 = call float @llvm.maxnum.f32(float %986, float %888), !dbg !26
  %988 = call float @llvm.maxnum.f32(float %987, float %889), !dbg !26
  %989 = call float @llvm.maxnum.f32(float %988, float %892), !dbg !26
  %990 = call float @llvm.maxnum.f32(float %989, float %893), !dbg !26
  %991 = call float @llvm.maxnum.f32(float %990, float %894), !dbg !26
  %992 = call float @llvm.maxnum.f32(float %991, float %895), !dbg !26
  %993 = call float @llvm.maxnum.f32(float %992, float %898), !dbg !26
  %994 = call float @llvm.maxnum.f32(float %993, float %899), !dbg !26
  %995 = call float @llvm.maxnum.f32(float %994, float %900), !dbg !26
  %996 = call float @llvm.maxnum.f32(float %995, float %901), !dbg !26
  %997 = call float @llvm.maxnum.f32(float %996, float %904), !dbg !26
  %998 = call float @llvm.maxnum.f32(float %997, float %905), !dbg !26
  %999 = call float @llvm.maxnum.f32(float %998, float %906), !dbg !26
  %1000 = call float @llvm.maxnum.f32(float %999, float %907), !dbg !26
  %1001 = call float @llvm.maxnum.f32(float %1000, float %910), !dbg !26
  %1002 = call float @llvm.maxnum.f32(float %1001, float %911), !dbg !26
  %1003 = call float @llvm.maxnum.f32(float %1002, float %912), !dbg !26
  %1004 = call float @llvm.maxnum.f32(float %1003, float %913), !dbg !26
  %1005 = call float @llvm.maxnum.f32(float %1004, float %916), !dbg !26
  %1006 = call float @llvm.maxnum.f32(float %1005, float %917), !dbg !26
  %1007 = call float @llvm.maxnum.f32(float %1006, float %918), !dbg !26
  %1008 = call float @llvm.maxnum.f32(float %1007, float %919), !dbg !26
  %1009 = call float @llvm.maxnum.f32(float %1008, float %922), !dbg !26
  %1010 = call float @llvm.maxnum.f32(float %1009, float %923), !dbg !26
  %1011 = call float @llvm.maxnum.f32(float %1010, float %924), !dbg !26
  %1012 = call float @llvm.maxnum.f32(float %1011, float %925), !dbg !26
  %1013 = bitcast float %1012 to i32, !dbg !26
  %1014 = call i32 @llvm.amdgcn.ds.bpermute(i32 %965, i32 %1013), !dbg !26
  %1015 = bitcast i32 %1014 to float, !dbg !26
  %1016 = call float @llvm.maxnum.f32(float %1012, float %1015), !dbg !26
  %1017 = bitcast float %1016 to i32, !dbg !26
  %1018 = call i32 @llvm.amdgcn.ds.bpermute(i32 %973, i32 %1017), !dbg !26
  %1019 = bitcast i32 %1018 to float, !dbg !26
  %1020 = call float @llvm.maxnum.f32(float %1016, float %1019), !dbg !26
  %1021 = extractvalue [2 x <1 x float>] %44, 1, !dbg !26
  %1022 = insertelement <1 x float> poison, float %1020, i64 0, !dbg !26
  %1023 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1022, <1 x float> %1021), !dbg !26
  %1024 = shufflevector <1 x float> %980, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %1025 = fsub <4 x float> %830, %1024, !dbg !27
  %1026 = fsub <4 x float> %836, %1024, !dbg !27
  %1027 = fsub <4 x float> %842, %1024, !dbg !27
  %1028 = fsub <4 x float> %848, %1024, !dbg !27
  %1029 = fsub <4 x float> %854, %1024, !dbg !27
  %1030 = fsub <4 x float> %860, %1024, !dbg !27
  %1031 = fsub <4 x float> %866, %1024, !dbg !27
  %1032 = fsub <4 x float> %872, %1024, !dbg !27
  %1033 = shufflevector <1 x float> %1023, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %1034 = fsub <4 x float> %879, %1033, !dbg !27
  %1035 = fsub <4 x float> %885, %1033, !dbg !27
  %1036 = fsub <4 x float> %891, %1033, !dbg !27
  %1037 = fsub <4 x float> %897, %1033, !dbg !27
  %1038 = fsub <4 x float> %903, %1033, !dbg !27
  %1039 = fsub <4 x float> %909, %1033, !dbg !27
  %1040 = fsub <4 x float> %915, %1033, !dbg !27
  %1041 = fsub <4 x float> %921, %1033, !dbg !27
  %1042 = fsub <1 x float> %978, %980, !dbg !28
  %1043 = extractelement <1 x float> %1042, i64 0, !dbg !29
  %1044 = call float @llvm.amdgcn.exp2.f32(float %1043), !dbg !29
  %1045 = insertelement <4 x float> poison, float %1044, i64 0, !dbg !34
  %1046 = shufflevector <4 x float> %1045, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1047 = extractvalue [2 x <1 x float>] %43, 0, !dbg !30
  %1048 = extractelement <1 x float> %1047, i64 0, !dbg !30
  %1049 = fmul float %1048, %1044, !dbg !30
  %1050 = fsub <1 x float> %1021, %1023, !dbg !28
  %1051 = extractelement <1 x float> %1050, i64 0, !dbg !29
  %1052 = call float @llvm.amdgcn.exp2.f32(float %1051), !dbg !29
  %1053 = insertelement <4 x float> poison, float %1052, i64 0, !dbg !34
  %1054 = shufflevector <4 x float> %1053, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1055 = extractvalue [2 x <1 x float>] %43, 1, !dbg !30
  %1056 = extractelement <1 x float> %1055, i64 0, !dbg !30
  %1057 = fmul float %1056, %1052, !dbg !30
  %1058 = extractelement <4 x float> %1025, i64 0, !dbg !31
  %1059 = call float @llvm.amdgcn.exp2.f32(float %1058), !dbg !31
  %1060 = insertelement <4 x float> poison, float %1059, i64 0, !dbg !33
  %1061 = extractelement <4 x float> %1025, i64 1, !dbg !31
  %1062 = call float @llvm.amdgcn.exp2.f32(float %1061), !dbg !31
  %1063 = insertelement <4 x float> %1060, float %1062, i64 1, !dbg !33
  %1064 = fadd float %1059, %1062, !dbg !32
  %1065 = extractelement <4 x float> %1025, i64 2, !dbg !31
  %1066 = call float @llvm.amdgcn.exp2.f32(float %1065), !dbg !31
  %1067 = insertelement <4 x float> %1063, float %1066, i64 2, !dbg !33
  %1068 = fadd float %1064, %1066, !dbg !32
  %1069 = extractelement <4 x float> %1025, i64 3, !dbg !31
  %1070 = call float @llvm.amdgcn.exp2.f32(float %1069), !dbg !31
  %1071 = insertelement <4 x float> %1067, float %1070, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1071) #9, !dbg !33, !frisk.register_lifetimes !1
  %1072 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1072) #9, !dbg !32, !frisk.register_lifetimes !1
  %1073 = fadd float %1068, %1070, !dbg !32
  %1074 = extractelement <4 x float> %1026, i64 0, !dbg !31
  %1075 = call float @llvm.amdgcn.exp2.f32(float %1074), !dbg !31
  %1076 = insertelement <4 x float> poison, float %1075, i64 0, !dbg !33
  %1077 = fadd float %1073, %1075, !dbg !32
  %1078 = extractelement <4 x float> %1026, i64 1, !dbg !31
  %1079 = call float @llvm.amdgcn.exp2.f32(float %1078), !dbg !31
  %1080 = insertelement <4 x float> %1076, float %1079, i64 1, !dbg !33
  %1081 = fadd float %1077, %1079, !dbg !32
  %1082 = extractelement <4 x float> %1026, i64 2, !dbg !31
  %1083 = call float @llvm.amdgcn.exp2.f32(float %1082), !dbg !31
  %1084 = insertelement <4 x float> %1080, float %1083, i64 2, !dbg !33
  %1085 = fadd float %1081, %1083, !dbg !32
  %1086 = extractelement <4 x float> %1026, i64 3, !dbg !31
  %1087 = call float @llvm.amdgcn.exp2.f32(float %1086), !dbg !31
  %1088 = insertelement <4 x float> %1084, float %1087, i64 3, !dbg !33
  %math.result99 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1088) #9, !dbg !33, !frisk.register_lifetimes !1
  %1089 = fptrunc <4 x float> %math.result99 to <4 x half>, !dbg !33
  %math.packed100 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1089) #9, !dbg !32, !frisk.register_lifetimes !1
  %1090 = fadd float %1085, %1087, !dbg !32
  %1091 = extractelement <4 x float> %1027, i64 0, !dbg !31
  %1092 = call float @llvm.amdgcn.exp2.f32(float %1091), !dbg !31
  %1093 = insertelement <4 x float> poison, float %1092, i64 0, !dbg !33
  %1094 = fadd float %1090, %1092, !dbg !32
  %1095 = extractelement <4 x float> %1027, i64 1, !dbg !31
  %1096 = call float @llvm.amdgcn.exp2.f32(float %1095), !dbg !31
  %1097 = insertelement <4 x float> %1093, float %1096, i64 1, !dbg !33
  %1098 = fadd float %1094, %1096, !dbg !32
  %1099 = extractelement <4 x float> %1027, i64 2, !dbg !31
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !31
  %1101 = insertelement <4 x float> %1097, float %1100, i64 2, !dbg !33
  %1102 = fadd float %1098, %1100, !dbg !32
  %1103 = extractelement <4 x float> %1027, i64 3, !dbg !31
  %1104 = call float @llvm.amdgcn.exp2.f32(float %1103), !dbg !31
  %1105 = insertelement <4 x float> %1101, float %1104, i64 3, !dbg !33
  %math.result101 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1105) #9, !dbg !33, !frisk.register_lifetimes !1
  %1106 = fptrunc <4 x float> %math.result101 to <4 x half>, !dbg !33
  %math.packed102 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1106) #9, !dbg !32, !frisk.register_lifetimes !1
  %1107 = fadd float %1102, %1104, !dbg !32
  %1108 = extractelement <4 x float> %1028, i64 0, !dbg !31
  %1109 = call float @llvm.amdgcn.exp2.f32(float %1108), !dbg !31
  %1110 = insertelement <4 x float> poison, float %1109, i64 0, !dbg !33
  %1111 = fadd float %1107, %1109, !dbg !32
  %1112 = extractelement <4 x float> %1028, i64 1, !dbg !31
  %1113 = call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !31
  %1114 = insertelement <4 x float> %1110, float %1113, i64 1, !dbg !33
  %1115 = fadd float %1111, %1113, !dbg !32
  %1116 = extractelement <4 x float> %1028, i64 2, !dbg !31
  %1117 = call float @llvm.amdgcn.exp2.f32(float %1116), !dbg !31
  %1118 = insertelement <4 x float> %1114, float %1117, i64 2, !dbg !33
  %1119 = fadd float %1115, %1117, !dbg !32
  %1120 = extractelement <4 x float> %1028, i64 3, !dbg !31
  %1121 = call float @llvm.amdgcn.exp2.f32(float %1120), !dbg !31
  %1122 = insertelement <4 x float> %1118, float %1121, i64 3, !dbg !33
  %math.result103 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1122) #9, !dbg !33, !frisk.register_lifetimes !1
  %1123 = fptrunc <4 x float> %math.result103 to <4 x half>, !dbg !33
  %math.packed104 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1123) #9, !dbg !32, !frisk.register_lifetimes !1
  %1124 = fadd float %1119, %1121, !dbg !32
  %1125 = extractelement <4 x float> %1029, i64 0, !dbg !31
  %1126 = call float @llvm.amdgcn.exp2.f32(float %1125), !dbg !31
  %1127 = insertelement <4 x float> poison, float %1126, i64 0, !dbg !33
  %1128 = fadd float %1124, %1126, !dbg !32
  %1129 = extractelement <4 x float> %1029, i64 1, !dbg !31
  %1130 = call float @llvm.amdgcn.exp2.f32(float %1129), !dbg !31
  %1131 = insertelement <4 x float> %1127, float %1130, i64 1, !dbg !33
  %1132 = fadd float %1128, %1130, !dbg !32
  %1133 = extractelement <4 x float> %1029, i64 2, !dbg !31
  %1134 = call float @llvm.amdgcn.exp2.f32(float %1133), !dbg !31
  %1135 = insertelement <4 x float> %1131, float %1134, i64 2, !dbg !33
  %1136 = fadd float %1132, %1134, !dbg !32
  %1137 = extractelement <4 x float> %1029, i64 3, !dbg !31
  %1138 = call float @llvm.amdgcn.exp2.f32(float %1137), !dbg !31
  %1139 = insertelement <4 x float> %1135, float %1138, i64 3, !dbg !33
  %math.result105 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1139) #9, !dbg !33, !frisk.register_lifetimes !1
  %1140 = fptrunc <4 x float> %math.result105 to <4 x half>, !dbg !33
  %math.packed106 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1140) #9, !dbg !32, !frisk.register_lifetimes !1
  %1141 = fadd float %1136, %1138, !dbg !32
  %1142 = extractelement <4 x float> %1030, i64 0, !dbg !31
  %1143 = call float @llvm.amdgcn.exp2.f32(float %1142), !dbg !31
  %1144 = insertelement <4 x float> poison, float %1143, i64 0, !dbg !33
  %1145 = fadd float %1141, %1143, !dbg !32
  %1146 = extractelement <4 x float> %1030, i64 1, !dbg !31
  %1147 = call float @llvm.amdgcn.exp2.f32(float %1146), !dbg !31
  %1148 = insertelement <4 x float> %1144, float %1147, i64 1, !dbg !33
  %1149 = fadd float %1145, %1147, !dbg !32
  %1150 = extractelement <4 x float> %1030, i64 2, !dbg !31
  %1151 = call float @llvm.amdgcn.exp2.f32(float %1150), !dbg !31
  %1152 = insertelement <4 x float> %1148, float %1151, i64 2, !dbg !33
  %1153 = fadd float %1149, %1151, !dbg !32
  %1154 = extractelement <4 x float> %1030, i64 3, !dbg !31
  %1155 = call float @llvm.amdgcn.exp2.f32(float %1154), !dbg !31
  %1156 = insertelement <4 x float> %1152, float %1155, i64 3, !dbg !33
  %math.result107 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1156) #9, !dbg !33, !frisk.register_lifetimes !1
  %1157 = fptrunc <4 x float> %math.result107 to <4 x half>, !dbg !33
  %math.packed108 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1157) #9, !dbg !32, !frisk.register_lifetimes !1
  %1158 = fadd float %1153, %1155, !dbg !32
  %1159 = extractelement <4 x float> %1031, i64 0, !dbg !31
  %1160 = call float @llvm.amdgcn.exp2.f32(float %1159), !dbg !31
  %1161 = insertelement <4 x float> poison, float %1160, i64 0, !dbg !33
  %1162 = fadd float %1158, %1160, !dbg !32
  %1163 = extractelement <4 x float> %1031, i64 1, !dbg !31
  %1164 = call float @llvm.amdgcn.exp2.f32(float %1163), !dbg !31
  %1165 = insertelement <4 x float> %1161, float %1164, i64 1, !dbg !33
  %1166 = fadd float %1162, %1164, !dbg !32
  %1167 = extractelement <4 x float> %1031, i64 2, !dbg !31
  %1168 = call float @llvm.amdgcn.exp2.f32(float %1167), !dbg !31
  %1169 = insertelement <4 x float> %1165, float %1168, i64 2, !dbg !33
  %1170 = fadd float %1166, %1168, !dbg !32
  %1171 = extractelement <4 x float> %1031, i64 3, !dbg !31
  %1172 = call float @llvm.amdgcn.exp2.f32(float %1171), !dbg !31
  %1173 = insertelement <4 x float> %1169, float %1172, i64 3, !dbg !33
  %math.result109 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1173) #9, !dbg !33, !frisk.register_lifetimes !1
  %1174 = fptrunc <4 x float> %math.result109 to <4 x half>, !dbg !33
  %math.packed110 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1174) #9, !dbg !32, !frisk.register_lifetimes !1
  %1175 = fadd float %1170, %1172, !dbg !32
  %1176 = extractelement <4 x float> %1032, i64 0, !dbg !31
  %1177 = call float @llvm.amdgcn.exp2.f32(float %1176), !dbg !31
  %1178 = insertelement <4 x float> poison, float %1177, i64 0, !dbg !33
  %1179 = fadd float %1175, %1177, !dbg !32
  %1180 = extractelement <4 x float> %1032, i64 1, !dbg !31
  %1181 = call float @llvm.amdgcn.exp2.f32(float %1180), !dbg !31
  %1182 = insertelement <4 x float> %1178, float %1181, i64 1, !dbg !33
  %1183 = fadd float %1179, %1181, !dbg !32
  %1184 = extractelement <4 x float> %1032, i64 2, !dbg !31
  %1185 = call float @llvm.amdgcn.exp2.f32(float %1184), !dbg !31
  %1186 = insertelement <4 x float> %1182, float %1185, i64 2, !dbg !33
  %1187 = fadd float %1183, %1185, !dbg !32
  %1188 = extractelement <4 x float> %1032, i64 3, !dbg !31
  %1189 = call float @llvm.amdgcn.exp2.f32(float %1188), !dbg !31
  %1190 = insertelement <4 x float> %1186, float %1189, i64 3, !dbg !33
  %math.result111 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1190) #9, !dbg !33, !frisk.register_lifetimes !1
  %1191 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !33
  %math.packed112 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1191) #9, !dbg !32, !frisk.register_lifetimes !1
  %1192 = fadd float %1187, %1189, !dbg !32
  %1193 = extractelement <4 x float> %1034, i64 0, !dbg !31
  %1194 = call float @llvm.amdgcn.exp2.f32(float %1193), !dbg !31
  %1195 = insertelement <4 x float> poison, float %1194, i64 0, !dbg !33
  %1196 = extractelement <4 x float> %1034, i64 1, !dbg !31
  %1197 = call float @llvm.amdgcn.exp2.f32(float %1196), !dbg !31
  %1198 = insertelement <4 x float> %1195, float %1197, i64 1, !dbg !33
  %1199 = fadd float %1194, %1197, !dbg !32
  %1200 = extractelement <4 x float> %1034, i64 2, !dbg !31
  %1201 = call float @llvm.amdgcn.exp2.f32(float %1200), !dbg !31
  %1202 = insertelement <4 x float> %1198, float %1201, i64 2, !dbg !33
  %1203 = fadd float %1199, %1201, !dbg !32
  %1204 = extractelement <4 x float> %1034, i64 3, !dbg !31
  %1205 = call float @llvm.amdgcn.exp2.f32(float %1204), !dbg !31
  %1206 = insertelement <4 x float> %1202, float %1205, i64 3, !dbg !33
  %math.result113 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1206) #9, !dbg !33, !frisk.register_lifetimes !1
  %1207 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !33
  %math.packed114 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1207) #9, !dbg !32, !frisk.register_lifetimes !1
  %1208 = fadd float %1203, %1205, !dbg !32
  %1209 = extractelement <4 x float> %1035, i64 0, !dbg !31
  %1210 = call float @llvm.amdgcn.exp2.f32(float %1209), !dbg !31
  %1211 = insertelement <4 x float> poison, float %1210, i64 0, !dbg !33
  %1212 = fadd float %1208, %1210, !dbg !32
  %1213 = extractelement <4 x float> %1035, i64 1, !dbg !31
  %1214 = call float @llvm.amdgcn.exp2.f32(float %1213), !dbg !31
  %1215 = insertelement <4 x float> %1211, float %1214, i64 1, !dbg !33
  %1216 = fadd float %1212, %1214, !dbg !32
  %1217 = extractelement <4 x float> %1035, i64 2, !dbg !31
  %1218 = call float @llvm.amdgcn.exp2.f32(float %1217), !dbg !31
  %1219 = insertelement <4 x float> %1215, float %1218, i64 2, !dbg !33
  %1220 = fadd float %1216, %1218, !dbg !32
  %1221 = extractelement <4 x float> %1035, i64 3, !dbg !31
  %1222 = call float @llvm.amdgcn.exp2.f32(float %1221), !dbg !31
  %1223 = insertelement <4 x float> %1219, float %1222, i64 3, !dbg !33
  %math.result115 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1223) #9, !dbg !33, !frisk.register_lifetimes !1
  %1224 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !33
  %math.packed116 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1224) #9, !dbg !32, !frisk.register_lifetimes !1
  %1225 = fadd float %1220, %1222, !dbg !32
  %1226 = extractelement <4 x float> %1036, i64 0, !dbg !31
  %1227 = call float @llvm.amdgcn.exp2.f32(float %1226), !dbg !31
  %1228 = insertelement <4 x float> poison, float %1227, i64 0, !dbg !33
  %1229 = fadd float %1225, %1227, !dbg !32
  %1230 = extractelement <4 x float> %1036, i64 1, !dbg !31
  %1231 = call float @llvm.amdgcn.exp2.f32(float %1230), !dbg !31
  %1232 = insertelement <4 x float> %1228, float %1231, i64 1, !dbg !33
  %1233 = fadd float %1229, %1231, !dbg !32
  %1234 = extractelement <4 x float> %1036, i64 2, !dbg !31
  %1235 = call float @llvm.amdgcn.exp2.f32(float %1234), !dbg !31
  %1236 = insertelement <4 x float> %1232, float %1235, i64 2, !dbg !33
  %1237 = fadd float %1233, %1235, !dbg !32
  %1238 = extractelement <4 x float> %1036, i64 3, !dbg !31
  %1239 = call float @llvm.amdgcn.exp2.f32(float %1238), !dbg !31
  %1240 = insertelement <4 x float> %1236, float %1239, i64 3, !dbg !33
  %math.result117 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1240) #9, !dbg !33, !frisk.register_lifetimes !1
  %1241 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !33
  %math.packed118 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1241) #9, !dbg !32, !frisk.register_lifetimes !1
  %1242 = fadd float %1237, %1239, !dbg !32
  %1243 = extractelement <4 x float> %1037, i64 0, !dbg !31
  %1244 = call float @llvm.amdgcn.exp2.f32(float %1243), !dbg !31
  %1245 = insertelement <4 x float> poison, float %1244, i64 0, !dbg !33
  %1246 = fadd float %1242, %1244, !dbg !32
  %1247 = extractelement <4 x float> %1037, i64 1, !dbg !31
  %1248 = call float @llvm.amdgcn.exp2.f32(float %1247), !dbg !31
  %1249 = insertelement <4 x float> %1245, float %1248, i64 1, !dbg !33
  %1250 = fadd float %1246, %1248, !dbg !32
  %1251 = extractelement <4 x float> %1037, i64 2, !dbg !31
  %1252 = call float @llvm.amdgcn.exp2.f32(float %1251), !dbg !31
  %1253 = insertelement <4 x float> %1249, float %1252, i64 2, !dbg !33
  %1254 = fadd float %1250, %1252, !dbg !32
  %1255 = extractelement <4 x float> %1037, i64 3, !dbg !31
  %1256 = call float @llvm.amdgcn.exp2.f32(float %1255), !dbg !31
  %1257 = insertelement <4 x float> %1253, float %1256, i64 3, !dbg !33
  %math.result119 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1257) #9, !dbg !33, !frisk.register_lifetimes !1
  %1258 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !33
  %math.packed120 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1258) #9, !dbg !32, !frisk.register_lifetimes !1
  %1259 = fadd float %1254, %1256, !dbg !32
  %1260 = extractelement <4 x float> %1038, i64 0, !dbg !31
  %1261 = call float @llvm.amdgcn.exp2.f32(float %1260), !dbg !31
  %1262 = insertelement <4 x float> poison, float %1261, i64 0, !dbg !33
  %1263 = fadd float %1259, %1261, !dbg !32
  %1264 = extractelement <4 x float> %1038, i64 1, !dbg !31
  %1265 = call float @llvm.amdgcn.exp2.f32(float %1264), !dbg !31
  %1266 = insertelement <4 x float> %1262, float %1265, i64 1, !dbg !33
  %1267 = fadd float %1263, %1265, !dbg !32
  %1268 = extractelement <4 x float> %1038, i64 2, !dbg !31
  %1269 = call float @llvm.amdgcn.exp2.f32(float %1268), !dbg !31
  %1270 = insertelement <4 x float> %1266, float %1269, i64 2, !dbg !33
  %1271 = fadd float %1267, %1269, !dbg !32
  %1272 = extractelement <4 x float> %1038, i64 3, !dbg !31
  %1273 = call float @llvm.amdgcn.exp2.f32(float %1272), !dbg !31
  %1274 = insertelement <4 x float> %1270, float %1273, i64 3, !dbg !33
  %math.result121 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1274) #9, !dbg !33, !frisk.register_lifetimes !1
  %1275 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !33
  %math.packed122 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1275) #9, !dbg !32, !frisk.register_lifetimes !1
  %1276 = fadd float %1271, %1273, !dbg !32
  %1277 = extractelement <4 x float> %1039, i64 0, !dbg !31
  %1278 = call float @llvm.amdgcn.exp2.f32(float %1277), !dbg !31
  %1279 = insertelement <4 x float> poison, float %1278, i64 0, !dbg !33
  %1280 = fadd float %1276, %1278, !dbg !32
  %1281 = extractelement <4 x float> %1039, i64 1, !dbg !31
  %1282 = call float @llvm.amdgcn.exp2.f32(float %1281), !dbg !31
  %1283 = insertelement <4 x float> %1279, float %1282, i64 1, !dbg !33
  %1284 = fadd float %1280, %1282, !dbg !32
  %1285 = extractelement <4 x float> %1039, i64 2, !dbg !31
  %1286 = call float @llvm.amdgcn.exp2.f32(float %1285), !dbg !31
  %1287 = insertelement <4 x float> %1283, float %1286, i64 2, !dbg !33
  %1288 = fadd float %1284, %1286, !dbg !32
  %1289 = extractelement <4 x float> %1039, i64 3, !dbg !31
  %1290 = call float @llvm.amdgcn.exp2.f32(float %1289), !dbg !31
  %1291 = insertelement <4 x float> %1287, float %1290, i64 3, !dbg !33
  %math.result123 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1291) #9, !dbg !33, !frisk.register_lifetimes !1
  %1292 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !33
  %math.packed124 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1292) #9, !dbg !32, !frisk.register_lifetimes !1
  %1293 = fadd float %1288, %1290, !dbg !32
  %1294 = extractelement <4 x float> %1040, i64 0, !dbg !31
  %1295 = call float @llvm.amdgcn.exp2.f32(float %1294), !dbg !31
  %1296 = insertelement <4 x float> poison, float %1295, i64 0, !dbg !33
  %1297 = fadd float %1293, %1295, !dbg !32
  %1298 = extractelement <4 x float> %1040, i64 1, !dbg !31
  %1299 = call float @llvm.amdgcn.exp2.f32(float %1298), !dbg !31
  %1300 = insertelement <4 x float> %1296, float %1299, i64 1, !dbg !33
  %1301 = fadd float %1297, %1299, !dbg !32
  %1302 = extractelement <4 x float> %1040, i64 2, !dbg !31
  %1303 = call float @llvm.amdgcn.exp2.f32(float %1302), !dbg !31
  %1304 = insertelement <4 x float> %1300, float %1303, i64 2, !dbg !33
  %1305 = fadd float %1301, %1303, !dbg !32
  %1306 = extractelement <4 x float> %1040, i64 3, !dbg !31
  %1307 = call float @llvm.amdgcn.exp2.f32(float %1306), !dbg !31
  %1308 = insertelement <4 x float> %1304, float %1307, i64 3, !dbg !33
  %math.result125 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1308) #9, !dbg !33, !frisk.register_lifetimes !1
  %1309 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !33
  %math.packed126 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1309) #9, !dbg !32, !frisk.register_lifetimes !1
  %1310 = fadd float %1305, %1307, !dbg !32
  %1311 = extractelement <4 x float> %1041, i64 0, !dbg !31
  %1312 = call float @llvm.amdgcn.exp2.f32(float %1311), !dbg !31
  %1313 = insertelement <4 x float> poison, float %1312, i64 0, !dbg !33
  %1314 = fadd float %1310, %1312, !dbg !32
  %1315 = extractelement <4 x float> %1041, i64 1, !dbg !31
  %1316 = call float @llvm.amdgcn.exp2.f32(float %1315), !dbg !31
  %1317 = insertelement <4 x float> %1313, float %1316, i64 1, !dbg !33
  %1318 = fadd float %1314, %1316, !dbg !32
  %1319 = extractelement <4 x float> %1041, i64 2, !dbg !31
  %1320 = call float @llvm.amdgcn.exp2.f32(float %1319), !dbg !31
  %1321 = insertelement <4 x float> %1317, float %1320, i64 2, !dbg !33
  %1322 = fadd float %1318, %1320, !dbg !32
  %1323 = extractelement <4 x float> %1041, i64 3, !dbg !31
  %1324 = call float @llvm.amdgcn.exp2.f32(float %1323), !dbg !31
  %1325 = insertelement <4 x float> %1321, float %1324, i64 3, !dbg !33
  %math.result127 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1325) #9, !dbg !33, !frisk.register_lifetimes !1
  %1326 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !33
  %math.packed128 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1326) #9, !dbg !32, !frisk.register_lifetimes !1
  %1327 = fadd float %1322, %1324, !dbg !32
  %1328 = bitcast float %1192 to i32, !dbg !32
  %1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %965, i32 %1328), !dbg !32
  %1330 = bitcast i32 %1329 to float, !dbg !32
  %1331 = fadd float %1192, %1330, !dbg !32
  %1332 = bitcast float %1331 to i32, !dbg !32
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %973, i32 %1332), !dbg !32
  %1334 = bitcast i32 %1333 to float, !dbg !32
  %1335 = fadd float %1331, %1334, !dbg !32
  %1336 = fadd float %1335, %1049, !dbg !32
  %1337 = insertelement <1 x float> poison, float %1336, i64 0, !dbg !7
  %1338 = insertvalue [2 x <1 x float>] poison, <1 x float> %1337, 0, !dbg !7
  %1339 = bitcast float %1327 to i32, !dbg !32
  %1340 = call i32 @llvm.amdgcn.ds.bpermute(i32 %965, i32 %1339), !dbg !32
  %1341 = bitcast i32 %1340 to float, !dbg !32
  %1342 = fadd float %1327, %1341, !dbg !32
  %1343 = bitcast float %1342 to i32, !dbg !32
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %973, i32 %1343), !dbg !32
  %1345 = bitcast i32 %1344 to float, !dbg !32
  %1346 = fadd float %1342, %1345, !dbg !32
  %1347 = fadd float %1346, %1057, !dbg !32
  %1348 = insertelement <1 x float> poison, float %1347, i64 0, !dbg !7
  %1349 = insertvalue [2 x <1 x float>] %1338, <1 x float> %1348, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread319 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local320 = shl nuw nsw i32 %address.thread319, 3, !dbg !14
  %address.local321 = and i32 %address.local320, 120, !dbg !14
  %address.local322 = shl nuw nsw i32 %address.thread319, 5, !dbg !14
  %address.local323 = and i32 %address.local322, 1536, !dbg !14
  %address.local324 = or disjoint i32 %address.local321, %address.local323, !dbg !14
  %address.local325 = zext nneg i32 %address.local324 to i64, !dbg !14
  %address.local326 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local325, !dbg !14
  %1350 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %1351 = bitcast <4 x i32> %1350 to <8 x half>, !dbg !14
  %address.local327 = zext nneg i32 %address.thread319 to i64, !dbg !12
  %address.local328 = and i64 %address.local327, 60, !dbg !14
  %address.local329 = add nuw nsw i64 %address.local328, %address.local327, !dbg !14
  %address.local330 = shl nuw nsw i64 %address.local329, 3, !dbg !14
  %address.local331 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local332 = and i64 %address.local327, 63, !dbg !12
  %address.local333 = lshr i64 %address.local332, 4, !dbg !14
  %address.local334 = mul nuw nsw i64 %address.local333, 384, !dbg !14
  %address.local335 = getelementptr [2 x i8], ptr addrspace(3) %address.local331, i64 %address.local334, !dbg !14
  %address.local336 = lshr i64 %address.local327, 2, !dbg !16
  %address.local337 = mul nsw i64 %address.local336, -32, !dbg !14
  %address.local338 = getelementptr [2 x i8], ptr addrspace(3) %address.local335, i64 %address.local337, !dbg !14
  %address.local339 = getelementptr i8, ptr addrspace(3) %address.local338, i64 256, !dbg !14
  %1352 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %1353 = bitcast <4 x i32> %1352 to <8 x half>, !dbg !14
  %address.local340 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) %address.local340, i64 %address.local334, !dbg !14
  %address.local342 = getelementptr [2 x i8], ptr addrspace(3) %address.local341, i64 %address.local337, !dbg !14
  %address.local343 = getelementptr i8, ptr addrspace(3) %address.local342, i64 512, !dbg !14
  %1354 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %1355 = bitcast <4 x i32> %1354 to <8 x half>, !dbg !14
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) %address.local344, i64 %address.local334, !dbg !14
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) %address.local345, i64 %address.local337, !dbg !14
  %address.local347 = getelementptr i8, ptr addrspace(3) %address.local346, i64 768, !dbg !14
  %1356 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %1357 = bitcast <4 x i32> %1356 to <8 x half>, !dbg !14
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local325, !dbg !14
  %address.local349 = getelementptr i8, ptr addrspace(3) %address.local348, i64 4096, !dbg !14
  %1358 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %1359 = bitcast <4 x i32> %1358 to <8 x half>, !dbg !14
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) %address.local350, i64 %address.local334, !dbg !14
  %address.local352 = getelementptr [2 x i8], ptr addrspace(3) %address.local351, i64 %address.local337, !dbg !14
  %address.local353 = getelementptr i8, ptr addrspace(3) %address.local352, i64 4352, !dbg !14
  %1360 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %1361 = bitcast <4 x i32> %1360 to <8 x half>, !dbg !14
  %address.local354 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local355 = getelementptr [2 x i8], ptr addrspace(3) %address.local354, i64 %address.local334, !dbg !14
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) %address.local355, i64 %address.local337, !dbg !14
  %address.local357 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4608, !dbg !14
  %1362 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %1363 = bitcast <4 x i32> %1362 to <8 x half>, !dbg !14
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) %address.local358, i64 %address.local334, !dbg !14
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) %address.local359, i64 %address.local337, !dbg !14
  %address.local361 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4864, !dbg !14
  %1364 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1365 = bitcast <4 x i32> %1364 to <8 x half>, !dbg !14
  %1366 = add nsw i64 %80, %8, !dbg !17
  %1367 = shl i64 %1366, 19, !dbg !17
  %1368 = and i64 %1367, 16252928, !dbg !17
  %1369 = add i64 %1368, %86, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1370 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1369, !dbg !17
  %1371 = getelementptr i8, ptr addrspace(1) %1370, i64 64, !dbg !17
  %address.local362 = shl nuw nsw i64 %address.local327, 3, !dbg !11
  %address.local363 = and i64 %address.local362, 1536, !dbg !11
  %address.local364 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local363, !dbg !11
  %address.local365 = ptrtoint ptr addrspace(3) %address.local364 to i64, !dbg !11
  %address.local366 = trunc i64 %address.local365 to i32, !dbg !11
  %1372 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local366), !dbg !17
  %1373 = zext i32 %1372 to i64, !dbg !17
  %1374 = inttoptr i64 %1373 to ptr addrspace(3), !dbg !17
  %address.local367 = shl nuw nsw i64 %address.local327, 5, !dbg !17
  %address.local368 = and i64 %address.local367, 8064, !dbg !17
  %address.local369 = shl nuw nsw i64 %address.local327, 3, !dbg !17
  %address.local370 = and i64 %address.local369, 24, !dbg !17
  %address.local371 = or disjoint i64 %address.local368, %address.local370, !dbg !17
  %address.local372 = getelementptr [2 x i8], ptr addrspace(1) %1371, i64 %address.local371, !dbg !17
  %1375 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1376 = ptrtoint ptr addrspace(1) %address.local372 to i64, !dbg !17
  %1377 = sub i64 %1376, %1375, !dbg !17
  %dma.buffer.offset880 = trunc i64 %1377 to i32, !dbg !17
  %1378 = trunc i64 %1375 to i32, !dbg !17
  %1379 = insertelement <4 x i32> poison, i32 %1378, i64 0, !dbg !17
  %1380 = lshr i64 %1375, 32, !dbg !17
  %1381 = trunc i64 %1380 to i32, !dbg !17
  %1382 = insertelement <4 x i32> %1379, i32 %1381, i64 1, !dbg !17
  %1383 = insertelement <4 x i32> %1382, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc881 = insertelement <4 x i32> %1383, i32 131072, i64 3, !dbg !17
  %1384 = ptrtoint ptr addrspace(3) %1374 to i32, !dbg !17
  %dma.buffer.lds882 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1384), !dbg !17
  %1385 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds882, i32 %dma.buffer.offset880, <4 x i32> %dma.buffer.rsrc881) #10, !dbg !17
  %address.local373 = or disjoint i64 %address.local327, 256, !dbg !11
  %address.local374 = shl nuw nsw i64 %address.local373, 3, !dbg !11
  %address.local375 = and i64 %address.local374, 3584, !dbg !11
  %address.local376 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local375, !dbg !11
  %address.local377 = ptrtoint ptr addrspace(3) %address.local376 to i64, !dbg !11
  %address.local378 = trunc i64 %address.local377 to i32, !dbg !11
  %1386 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local378), !dbg !17
  %1387 = zext i32 %1386 to i64, !dbg !17
  %1388 = inttoptr i64 %1387 to ptr addrspace(3), !dbg !17
  %address.local379 = shl nuw nsw i64 %address.local373, 5, !dbg !17
  %address.local380 = and i64 %address.local379, 16256, !dbg !17
  %address.local381 = shl nuw nsw i64 %address.local327, 3, !dbg !17
  %address.local382 = and i64 %address.local381, 24, !dbg !17
  %address.local383 = or disjoint i64 %address.local380, %address.local382, !dbg !17
  %address.local384 = getelementptr [2 x i8], ptr addrspace(1) %1371, i64 %address.local383, !dbg !17
  %1389 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1390 = ptrtoint ptr addrspace(1) %address.local384 to i64, !dbg !17
  %1391 = sub i64 %1390, %1389, !dbg !17
  %dma.buffer.offset877 = trunc i64 %1391 to i32, !dbg !17
  %1392 = trunc i64 %1389 to i32, !dbg !17
  %1393 = insertelement <4 x i32> poison, i32 %1392, i64 0, !dbg !17
  %1394 = lshr i64 %1389, 32, !dbg !17
  %1395 = trunc i64 %1394 to i32, !dbg !17
  %1396 = insertelement <4 x i32> %1393, i32 %1395, i64 1, !dbg !17
  %1397 = insertelement <4 x i32> %1396, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc878 = insertelement <4 x i32> %1397, i32 131072, i64 3, !dbg !17
  %1398 = ptrtoint ptr addrspace(3) %1388 to i32, !dbg !17
  %dma.buffer.lds879 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1398), !dbg !17
  %1399 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds879, i32 %dma.buffer.offset877, <4 x i32> %dma.buffer.rsrc878) #10, !dbg !17
  %1400 = shufflevector <8 x half> %1351, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1401 = extractvalue [2 x <8 x float>] %42, 0, !dbg !34
  %1402 = shufflevector <8 x float> %1401, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1403 = fmul <4 x float> %1402, %1046, !dbg !34
  %1404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1400, <4 x float> %1403) #9, !dbg !14
  %1405 = shufflevector <8 x half> %1353, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1405, <4 x float> %1404) #9, !dbg !14
  %1407 = shufflevector <8 x half> %1355, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1407, <4 x float> %1406) #9, !dbg !14
  %1409 = shufflevector <8 x half> %1357, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1409, <4 x float> %1408) #9, !dbg !14
  %1411 = shufflevector <8 x half> %1359, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1411, <4 x float> %1410) #9, !dbg !14
  %1413 = shufflevector <8 x half> %1361, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1413, <4 x float> %1412) #9, !dbg !14
  %1415 = shufflevector <8 x half> %1363, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1415, <4 x float> %1414) #9, !dbg !14
  %1417 = shufflevector <8 x half> %1365, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1417, <4 x float> %1416) #9, !dbg !14
  %1419 = shufflevector <8 x half> %1351, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1420 = shufflevector <8 x float> %1401, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1421 = fmul <4 x float> %1420, %1046, !dbg !34
  %1422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1419, <4 x float> %1421) #9, !dbg !14
  %1423 = shufflevector <8 x half> %1353, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1423, <4 x float> %1422) #9, !dbg !14
  %1425 = shufflevector <8 x half> %1355, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1425, <4 x float> %1424) #9, !dbg !14
  %1427 = shufflevector <8 x half> %1357, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1427, <4 x float> %1426) #9, !dbg !14
  %1429 = shufflevector <8 x half> %1359, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1429, <4 x float> %1428) #9, !dbg !14
  %1431 = shufflevector <8 x half> %1361, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1431, <4 x float> %1430) #9, !dbg !14
  %1433 = shufflevector <8 x half> %1363, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1433, <4 x float> %1432) #9, !dbg !14
  %1435 = shufflevector <8 x half> %1365, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1435, <4 x float> %1434) #9, !dbg !14
  %1437 = extractvalue [2 x <8 x float>] %42, 1, !dbg !34
  %1438 = shufflevector <8 x float> %1437, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1439 = fmul <4 x float> %1438, %1054, !dbg !34
  %1440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1400, <4 x float> %1439) #9, !dbg !14
  %1441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1405, <4 x float> %1440) #9, !dbg !14
  %1442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1407, <4 x float> %1441) #9, !dbg !14
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1409, <4 x float> %1442) #9, !dbg !14
  %1444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1411, <4 x float> %1443) #9, !dbg !14
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1413, <4 x float> %1444) #9, !dbg !14
  %1446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1415, <4 x float> %1445) #9, !dbg !14
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1417, <4 x float> %1446) #9, !dbg !14
  %1448 = shufflevector <8 x float> %1437, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1449 = fmul <4 x float> %1448, %1054, !dbg !34
  %1450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1419, <4 x float> %1449) #9, !dbg !14
  %1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1423, <4 x float> %1450) #9, !dbg !14
  %1452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1425, <4 x float> %1451) #9, !dbg !14
  %1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1427, <4 x float> %1452) #9, !dbg !14
  %1454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1429, <4 x float> %1453) #9, !dbg !14
  %1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1431, <4 x float> %1454) #9, !dbg !14
  %1456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1433, <4 x float> %1455) #9, !dbg !14
  %1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1435, <4 x float> %1456) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1458 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %1459 = bitcast <4 x i32> %1458 to <8 x half>, !dbg !14
  %1460 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %1461 = bitcast <4 x i32> %1460 to <8 x half>, !dbg !14
  %1462 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %1463 = bitcast <4 x i32> %1462 to <8 x half>, !dbg !14
  %1464 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %1465 = bitcast <4 x i32> %1464 to <8 x half>, !dbg !14
  %1466 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %1467 = bitcast <4 x i32> %1466 to <8 x half>, !dbg !14
  %1468 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %1469 = bitcast <4 x i32> %1468 to <8 x half>, !dbg !14
  %1470 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %1471 = bitcast <4 x i32> %1470 to <8 x half>, !dbg !14
  %1472 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1473 = bitcast <4 x i32> %1472 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1474 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1369, !dbg !17
  %1475 = getelementptr i8, ptr addrspace(1) %1474, i64 128, !dbg !17
  %1476 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local366), !dbg !17
  %1477 = zext i32 %1476 to i64, !dbg !17
  %1478 = inttoptr i64 %1477 to ptr addrspace(3), !dbg !17
  %address.local385 = getelementptr [2 x i8], ptr addrspace(1) %1475, i64 %address.local371, !dbg !17
  %1479 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1480 = ptrtoint ptr addrspace(1) %address.local385 to i64, !dbg !17
  %1481 = sub i64 %1480, %1479, !dbg !17
  %dma.buffer.offset874 = trunc i64 %1481 to i32, !dbg !17
  %1482 = trunc i64 %1479 to i32, !dbg !17
  %1483 = insertelement <4 x i32> poison, i32 %1482, i64 0, !dbg !17
  %1484 = lshr i64 %1479, 32, !dbg !17
  %1485 = trunc i64 %1484 to i32, !dbg !17
  %1486 = insertelement <4 x i32> %1483, i32 %1485, i64 1, !dbg !17
  %1487 = insertelement <4 x i32> %1486, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc875 = insertelement <4 x i32> %1487, i32 131072, i64 3, !dbg !17
  %1488 = ptrtoint ptr addrspace(3) %1478 to i32, !dbg !17
  %dma.buffer.lds876 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1488), !dbg !17
  %1489 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds876, i32 %dma.buffer.offset874, <4 x i32> %dma.buffer.rsrc875) #10, !dbg !17
  %1490 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local378), !dbg !17
  %1491 = zext i32 %1490 to i64, !dbg !17
  %1492 = inttoptr i64 %1491 to ptr addrspace(3), !dbg !17
  %address.local386 = getelementptr [2 x i8], ptr addrspace(1) %1475, i64 %address.local383, !dbg !17
  %1493 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1494 = ptrtoint ptr addrspace(1) %address.local386 to i64, !dbg !17
  %1495 = sub i64 %1494, %1493, !dbg !17
  %dma.buffer.offset871 = trunc i64 %1495 to i32, !dbg !17
  %1496 = trunc i64 %1493 to i32, !dbg !17
  %1497 = insertelement <4 x i32> poison, i32 %1496, i64 0, !dbg !17
  %1498 = lshr i64 %1493, 32, !dbg !17
  %1499 = trunc i64 %1498 to i32, !dbg !17
  %1500 = insertelement <4 x i32> %1497, i32 %1499, i64 1, !dbg !17
  %1501 = insertelement <4 x i32> %1500, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc872 = insertelement <4 x i32> %1501, i32 131072, i64 3, !dbg !17
  %1502 = ptrtoint ptr addrspace(3) %1492 to i32, !dbg !17
  %dma.buffer.lds873 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1502), !dbg !17
  %1503 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds873, i32 %dma.buffer.offset871, <4 x i32> %dma.buffer.rsrc872) #10, !dbg !17
  %1504 = shufflevector <8 x half> %1459, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1505 = extractvalue [2 x <8 x float>] %45, 0, !dbg !34
  %1506 = shufflevector <8 x float> %1505, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1507 = fmul <4 x float> %1506, %1046, !dbg !34
  %1508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1504, <4 x float> %1507) #9, !dbg !14
  %1509 = shufflevector <8 x half> %1461, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1509, <4 x float> %1508) #9, !dbg !14
  %1511 = shufflevector <8 x half> %1463, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1511, <4 x float> %1510) #9, !dbg !14
  %1513 = shufflevector <8 x half> %1465, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1513, <4 x float> %1512) #9, !dbg !14
  %1515 = shufflevector <8 x half> %1467, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1515, <4 x float> %1514) #9, !dbg !14
  %1517 = shufflevector <8 x half> %1469, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1517, <4 x float> %1516) #9, !dbg !14
  %1519 = shufflevector <8 x half> %1471, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1519, <4 x float> %1518) #9, !dbg !14
  %1521 = shufflevector <8 x half> %1473, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1521, <4 x float> %1520) #9, !dbg !14
  %1523 = shufflevector <8 x half> %1459, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1524 = shufflevector <8 x float> %1505, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1525 = fmul <4 x float> %1524, %1046, !dbg !34
  %1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1523, <4 x float> %1525) #9, !dbg !14
  %1527 = shufflevector <8 x half> %1461, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1527, <4 x float> %1526) #9, !dbg !14
  %1529 = shufflevector <8 x half> %1463, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1529, <4 x float> %1528) #9, !dbg !14
  %1531 = shufflevector <8 x half> %1465, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1531, <4 x float> %1530) #9, !dbg !14
  %1533 = shufflevector <8 x half> %1467, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1533, <4 x float> %1532) #9, !dbg !14
  %1535 = shufflevector <8 x half> %1469, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1535, <4 x float> %1534) #9, !dbg !14
  %1537 = shufflevector <8 x half> %1471, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1537, <4 x float> %1536) #9, !dbg !14
  %1539 = shufflevector <8 x half> %1473, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1539, <4 x float> %1538) #9, !dbg !14
  %1541 = extractvalue [2 x <8 x float>] %45, 1, !dbg !34
  %1542 = shufflevector <8 x float> %1541, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1543 = fmul <4 x float> %1542, %1054, !dbg !34
  %1544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1504, <4 x float> %1543) #9, !dbg !14
  %1545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1509, <4 x float> %1544) #9, !dbg !14
  %1546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1511, <4 x float> %1545) #9, !dbg !14
  %1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1513, <4 x float> %1546) #9, !dbg !14
  %1548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1515, <4 x float> %1547) #9, !dbg !14
  %1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1517, <4 x float> %1548) #9, !dbg !14
  %1550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1519, <4 x float> %1549) #9, !dbg !14
  %1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1521, <4 x float> %1550) #9, !dbg !14
  %1552 = shufflevector <8 x float> %1541, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1553 = fmul <4 x float> %1552, %1054, !dbg !34
  %1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1523, <4 x float> %1553) #9, !dbg !14
  %1555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1527, <4 x float> %1554) #9, !dbg !14
  %1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1529, <4 x float> %1555) #9, !dbg !14
  %1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1531, <4 x float> %1556) #9, !dbg !14
  %1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1533, <4 x float> %1557) #9, !dbg !14
  %1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1535, <4 x float> %1558) #9, !dbg !14
  %1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1537, <4 x float> %1559) #9, !dbg !14
  %1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1539, <4 x float> %1560) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1562 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %1563 = bitcast <4 x i32> %1562 to <8 x half>, !dbg !14
  %1564 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %1565 = bitcast <4 x i32> %1564 to <8 x half>, !dbg !14
  %1566 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %1567 = bitcast <4 x i32> %1566 to <8 x half>, !dbg !14
  %1568 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %1569 = bitcast <4 x i32> %1568 to <8 x half>, !dbg !14
  %1570 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %1571 = bitcast <4 x i32> %1570 to <8 x half>, !dbg !14
  %1572 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %1573 = bitcast <4 x i32> %1572 to <8 x half>, !dbg !14
  %1574 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %1575 = bitcast <4 x i32> %1574 to <8 x half>, !dbg !14
  %1576 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1577 = bitcast <4 x i32> %1576 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1578 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1369, !dbg !17
  %1579 = getelementptr i8, ptr addrspace(1) %1578, i64 192, !dbg !17
  %1580 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local366), !dbg !17
  %1581 = zext i32 %1580 to i64, !dbg !17
  %1582 = inttoptr i64 %1581 to ptr addrspace(3), !dbg !17
  %address.local387 = getelementptr [2 x i8], ptr addrspace(1) %1579, i64 %address.local371, !dbg !17
  %1583 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1584 = ptrtoint ptr addrspace(1) %address.local387 to i64, !dbg !17
  %1585 = sub i64 %1584, %1583, !dbg !17
  %dma.buffer.offset868 = trunc i64 %1585 to i32, !dbg !17
  %1586 = trunc i64 %1583 to i32, !dbg !17
  %1587 = insertelement <4 x i32> poison, i32 %1586, i64 0, !dbg !17
  %1588 = lshr i64 %1583, 32, !dbg !17
  %1589 = trunc i64 %1588 to i32, !dbg !17
  %1590 = insertelement <4 x i32> %1587, i32 %1589, i64 1, !dbg !17
  %1591 = insertelement <4 x i32> %1590, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc869 = insertelement <4 x i32> %1591, i32 131072, i64 3, !dbg !17
  %1592 = ptrtoint ptr addrspace(3) %1582 to i32, !dbg !17
  %dma.buffer.lds870 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1592), !dbg !17
  %1593 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds870, i32 %dma.buffer.offset868, <4 x i32> %dma.buffer.rsrc869) #10, !dbg !17
  %1594 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local378), !dbg !17
  %1595 = zext i32 %1594 to i64, !dbg !17
  %1596 = inttoptr i64 %1595 to ptr addrspace(3), !dbg !17
  %address.local388 = getelementptr [2 x i8], ptr addrspace(1) %1579, i64 %address.local383, !dbg !17
  %1597 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1598 = ptrtoint ptr addrspace(1) %address.local388 to i64, !dbg !17
  %1599 = sub i64 %1598, %1597, !dbg !17
  %dma.buffer.offset = trunc i64 %1599 to i32, !dbg !17
  %1600 = trunc i64 %1597 to i32, !dbg !17
  %1601 = insertelement <4 x i32> poison, i32 %1600, i64 0, !dbg !17
  %1602 = lshr i64 %1597, 32, !dbg !17
  %1603 = trunc i64 %1602 to i32, !dbg !17
  %1604 = insertelement <4 x i32> %1601, i32 %1603, i64 1, !dbg !17
  %1605 = insertelement <4 x i32> %1604, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %1605, i32 131072, i64 3, !dbg !17
  %1606 = ptrtoint ptr addrspace(3) %1596 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1606), !dbg !17
  %1607 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %1608 = shufflevector <8 x half> %1563, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1609 = extractvalue [2 x <8 x float>] %46, 0, !dbg !34
  %1610 = shufflevector <8 x float> %1609, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1611 = fmul <4 x float> %1610, %1046, !dbg !34
  %1612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1608, <4 x float> %1611) #9, !dbg !14
  %1613 = shufflevector <8 x half> %1565, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1613, <4 x float> %1612) #9, !dbg !14
  %1615 = shufflevector <8 x half> %1567, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1615, <4 x float> %1614) #9, !dbg !14
  %1617 = shufflevector <8 x half> %1569, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1617, <4 x float> %1616) #9, !dbg !14
  %1619 = shufflevector <8 x half> %1571, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1619, <4 x float> %1618) #9, !dbg !14
  %1621 = shufflevector <8 x half> %1573, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1621, <4 x float> %1620) #9, !dbg !14
  %1623 = shufflevector <8 x half> %1575, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1623, <4 x float> %1622) #9, !dbg !14
  %1625 = shufflevector <8 x half> %1577, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1625, <4 x float> %1624) #9, !dbg !14
  %1627 = shufflevector <8 x half> %1563, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1628 = shufflevector <8 x float> %1609, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1629 = fmul <4 x float> %1628, %1046, !dbg !34
  %1630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1627, <4 x float> %1629) #9, !dbg !14
  %1631 = shufflevector <8 x half> %1565, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1631, <4 x float> %1630) #9, !dbg !14
  %1633 = shufflevector <8 x half> %1567, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1633, <4 x float> %1632) #9, !dbg !14
  %1635 = shufflevector <8 x half> %1569, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1635, <4 x float> %1634) #9, !dbg !14
  %1637 = shufflevector <8 x half> %1571, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1637, <4 x float> %1636) #9, !dbg !14
  %1639 = shufflevector <8 x half> %1573, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1639, <4 x float> %1638) #9, !dbg !14
  %1641 = shufflevector <8 x half> %1575, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1641, <4 x float> %1640) #9, !dbg !14
  %1643 = shufflevector <8 x half> %1577, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1643, <4 x float> %1642) #9, !dbg !14
  %1645 = extractvalue [2 x <8 x float>] %46, 1, !dbg !34
  %1646 = shufflevector <8 x float> %1645, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1647 = fmul <4 x float> %1646, %1054, !dbg !34
  %1648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1608, <4 x float> %1647) #9, !dbg !14
  %1649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1613, <4 x float> %1648) #9, !dbg !14
  %1650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1615, <4 x float> %1649) #9, !dbg !14
  %1651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1617, <4 x float> %1650) #9, !dbg !14
  %1652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1619, <4 x float> %1651) #9, !dbg !14
  %1653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1621, <4 x float> %1652) #9, !dbg !14
  %1654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1623, <4 x float> %1653) #9, !dbg !14
  %1655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1625, <4 x float> %1654) #9, !dbg !14
  %1656 = shufflevector <8 x float> %1645, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1657 = fmul <4 x float> %1656, %1054, !dbg !34
  %1658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1627, <4 x float> %1657) #9, !dbg !14
  %1659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1631, <4 x float> %1658) #9, !dbg !14
  %1660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1633, <4 x float> %1659) #9, !dbg !14
  %1661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1635, <4 x float> %1660) #9, !dbg !14
  %1662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1637, <4 x float> %1661) #9, !dbg !14
  %1663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1639, <4 x float> %1662) #9, !dbg !14
  %1664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1641, <4 x float> %1663) #9, !dbg !14
  %1665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1643, <4 x float> %1664) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1666 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %1667 = bitcast <4 x i32> %1666 to <8 x half>, !dbg !14
  %1668 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %1669 = bitcast <4 x i32> %1668 to <8 x half>, !dbg !14
  %1670 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %1671 = bitcast <4 x i32> %1670 to <8 x half>, !dbg !14
  %1672 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %1673 = bitcast <4 x i32> %1672 to <8 x half>, !dbg !14
  %1674 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %1675 = bitcast <4 x i32> %1674 to <8 x half>, !dbg !14
  %1676 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %1677 = bitcast <4 x i32> %1676 to <8 x half>, !dbg !14
  %1678 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %1679 = bitcast <4 x i32> %1678 to <8 x half>, !dbg !14
  %1680 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1681 = bitcast <4 x i32> %1680 to <8 x half>, !dbg !14
  %1682 = shufflevector <8 x half> %1667, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1683 = extractvalue [2 x <8 x float>] %47, 0, !dbg !34
  %1684 = shufflevector <8 x float> %1683, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1685 = fmul <4 x float> %1684, %1046, !dbg !34
  %1686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1682, <4 x float> %1685) #9, !dbg !14
  %1687 = shufflevector <8 x half> %1669, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1687, <4 x float> %1686) #9, !dbg !14
  %1689 = shufflevector <8 x half> %1671, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1689, <4 x float> %1688) #9, !dbg !14
  %1691 = shufflevector <8 x half> %1673, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1691, <4 x float> %1690) #9, !dbg !14
  %1693 = shufflevector <8 x half> %1675, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1693, <4 x float> %1692) #9, !dbg !14
  %1695 = shufflevector <8 x half> %1677, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1695, <4 x float> %1694) #9, !dbg !14
  %1697 = shufflevector <8 x half> %1679, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1697, <4 x float> %1696) #9, !dbg !14
  %1699 = shufflevector <8 x half> %1681, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1699, <4 x float> %1698) #9, !dbg !14
  %1701 = shufflevector <8 x half> %1667, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1702 = shufflevector <8 x float> %1683, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1703 = fmul <4 x float> %1702, %1046, !dbg !34
  %1704 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1701, <4 x float> %1703) #9, !dbg !14
  %1705 = shufflevector <8 x half> %1669, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1706 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1705, <4 x float> %1704) #9, !dbg !14
  %1707 = shufflevector <8 x half> %1671, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1708 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1707, <4 x float> %1706) #9, !dbg !14
  %1709 = shufflevector <8 x half> %1673, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1710 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1709, <4 x float> %1708) #9, !dbg !14
  %1711 = shufflevector <8 x half> %1675, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1712 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1711, <4 x float> %1710) #9, !dbg !14
  %1713 = shufflevector <8 x half> %1677, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1714 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1713, <4 x float> %1712) #9, !dbg !14
  %1715 = shufflevector <8 x half> %1679, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1715, <4 x float> %1714) #9, !dbg !14
  %1717 = shufflevector <8 x half> %1681, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1717, <4 x float> %1716) #9, !dbg !14
  %1719 = extractvalue [2 x <8 x float>] %47, 1, !dbg !34
  %1720 = shufflevector <8 x float> %1719, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1721 = fmul <4 x float> %1720, %1054, !dbg !34
  %1722 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1682, <4 x float> %1721) #9, !dbg !14
  %1723 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1687, <4 x float> %1722) #9, !dbg !14
  %1724 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1689, <4 x float> %1723) #9, !dbg !14
  %1725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1691, <4 x float> %1724) #9, !dbg !14
  %1726 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1693, <4 x float> %1725) #9, !dbg !14
  %1727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1695, <4 x float> %1726) #9, !dbg !14
  %1728 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1697, <4 x float> %1727) #9, !dbg !14
  %1729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1699, <4 x float> %1728) #9, !dbg !14
  %1730 = shufflevector <8 x float> %1719, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1731 = fmul <4 x float> %1730, %1054, !dbg !34
  %1732 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1701, <4 x float> %1731) #9, !dbg !14
  %1733 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1705, <4 x float> %1732) #9, !dbg !14
  %1734 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1707, <4 x float> %1733) #9, !dbg !14
  %1735 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1709, <4 x float> %1734) #9, !dbg !14
  %1736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1711, <4 x float> %1735) #9, !dbg !14
  %1737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1713, <4 x float> %1736) #9, !dbg !14
  %1738 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1715, <4 x float> %1737) #9, !dbg !14
  %1739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1717, <4 x float> %1738) #9, !dbg !14
  %1740 = shufflevector <4 x float> %1418, <4 x float> %1436, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1741 = insertvalue [2 x <8 x float>] poison, <8 x float> %1740, 0, !dbg !7
  %1742 = shufflevector <4 x float> %1447, <4 x float> %1457, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1743 = insertvalue [2 x <8 x float>] %1741, <8 x float> %1742, 1, !dbg !7
  %1744 = insertvalue [2 x <1 x float>] poison, <1 x float> %980, 0, !dbg !7
  %1745 = insertvalue [2 x <1 x float>] %1744, <1 x float> %1023, 1, !dbg !7
  %1746 = shufflevector <4 x float> %1522, <4 x float> %1540, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1747 = insertvalue [2 x <8 x float>] poison, <8 x float> %1746, 0, !dbg !7
  %1748 = shufflevector <4 x float> %1551, <4 x float> %1561, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1749 = insertvalue [2 x <8 x float>] %1747, <8 x float> %1748, 1, !dbg !7
  %1750 = shufflevector <4 x float> %1626, <4 x float> %1644, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1751 = insertvalue [2 x <8 x float>] poison, <8 x float> %1750, 0, !dbg !7
  %1752 = shufflevector <4 x float> %1655, <4 x float> %1665, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1753 = insertvalue [2 x <8 x float>] %1751, <8 x float> %1752, 1, !dbg !7
  %1754 = shufflevector <4 x float> %1700, <4 x float> %1718, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1755 = insertvalue [2 x <8 x float>] poison, <8 x float> %1754, 0, !dbg !7
  %1756 = shufflevector <4 x float> %1729, <4 x float> %1739, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1757 = insertvalue [2 x <8 x float>] %1755, <8 x float> %1756, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1758 = add nuw nsw i64 %41, 1, !dbg !7
  br label %40, !dbg !7

1759:                                             ; preds = %40
  %1760 = and i32 %24, 16, !dbg !7
  %.not2 = icmp eq i32 %1760, 0, !dbg !7
  %1761 = and i32 %24, 32, !dbg !7
  %.not3 = icmp eq i32 %1761, 0, !dbg !7
  %1762 = extractvalue [2 x <8 x float>] %47, 0, !dbg !7
  %1763 = extractelement <8 x float> %1762, i64 0, !dbg !7
  %1764 = extractelement <8 x float> %1762, i64 1, !dbg !7
  %1765 = extractelement <8 x float> %1762, i64 2, !dbg !7
  %1766 = extractelement <8 x float> %1762, i64 3, !dbg !7
  %1767 = select i1 %.not2, float %1764, float %1763, !dbg !7
  %1768 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1769 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1768), !dbg !7
  %1770 = add nuw nsw i32 %1769, 64, !dbg !7
  %1771 = xor i32 %1769, 16, !dbg !7
  %1772 = and i32 %1770, 192, !dbg !7
  %1773 = icmp samesign ult i32 %1771, %1772, !dbg !7
  %1774 = select i1 %1773, i32 %1771, i32 %1769, !dbg !7
  %1775 = shl nuw nsw i32 %1774, 2, !dbg !7
  %1776 = bitcast float %1767 to i32, !dbg !7
  %1777 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1776), !dbg !7
  %1778 = bitcast i32 %1777 to float, !dbg !7
  %1779 = select i1 %.not2, float %1763, float %1778, !dbg !7
  %1780 = select i1 %.not2, float %1778, float %1764, !dbg !7
  %1781 = select i1 %.not2, float %1766, float %1765, !dbg !7
  %1782 = bitcast float %1781 to i32, !dbg !7
  %1783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1782), !dbg !7
  %1784 = bitcast i32 %1783 to float, !dbg !7
  %1785 = select i1 %.not2, float %1765, float %1784, !dbg !7
  %1786 = select i1 %.not2, float %1784, float %1766, !dbg !7
  %1787 = select i1 %.not3, float %1785, float %1779, !dbg !7
  %1788 = xor i32 %1769, 32, !dbg !7
  %1789 = icmp samesign ult i32 %1788, %1772, !dbg !7
  %1790 = select i1 %1789, i32 %1788, i32 %1769, !dbg !7
  %1791 = shl nuw nsw i32 %1790, 2, !dbg !7
  %1792 = bitcast float %1787 to i32, !dbg !7
  %1793 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1792), !dbg !7
  %1794 = bitcast i32 %1793 to float, !dbg !7
  %1795 = select i1 %.not3, float %1779, float %1794, !dbg !7
  %1796 = select i1 %.not3, float %1794, float %1785, !dbg !7
  %1797 = select i1 %.not3, float %1786, float %1780, !dbg !7
  %1798 = bitcast float %1797 to i32, !dbg !7
  %1799 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1798), !dbg !7
  %1800 = bitcast i32 %1799 to float, !dbg !7
  %1801 = select i1 %.not3, float %1780, float %1800, !dbg !7
  %1802 = select i1 %.not3, float %1800, float %1786, !dbg !7
  %1803 = extractvalue [2 x <8 x float>] %46, 0, !dbg !7
  %1804 = extractelement <8 x float> %1803, i64 0, !dbg !7
  %1805 = extractelement <8 x float> %1803, i64 1, !dbg !7
  %1806 = extractelement <8 x float> %1803, i64 2, !dbg !7
  %1807 = extractelement <8 x float> %1803, i64 3, !dbg !7
  %1808 = select i1 %.not2, float %1805, float %1804, !dbg !7
  %1809 = bitcast float %1808 to i32, !dbg !7
  %1810 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1809), !dbg !7
  %1811 = bitcast i32 %1810 to float, !dbg !7
  %1812 = select i1 %.not2, float %1804, float %1811, !dbg !7
  %1813 = select i1 %.not2, float %1811, float %1805, !dbg !7
  %1814 = select i1 %.not2, float %1807, float %1806, !dbg !7
  %1815 = bitcast float %1814 to i32, !dbg !7
  %1816 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1815), !dbg !7
  %1817 = bitcast i32 %1816 to float, !dbg !7
  %1818 = select i1 %.not2, float %1806, float %1817, !dbg !7
  %1819 = select i1 %.not2, float %1817, float %1807, !dbg !7
  %1820 = select i1 %.not3, float %1818, float %1812, !dbg !7
  %1821 = bitcast float %1820 to i32, !dbg !7
  %1822 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1821), !dbg !7
  %1823 = bitcast i32 %1822 to float, !dbg !7
  %1824 = select i1 %.not3, float %1812, float %1823, !dbg !7
  %1825 = select i1 %.not3, float %1823, float %1818, !dbg !7
  %1826 = select i1 %.not3, float %1819, float %1813, !dbg !7
  %1827 = bitcast float %1826 to i32, !dbg !7
  %1828 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1827), !dbg !7
  %1829 = bitcast i32 %1828 to float, !dbg !7
  %1830 = select i1 %.not3, float %1813, float %1829, !dbg !7
  %1831 = select i1 %.not3, float %1829, float %1819, !dbg !7
  %1832 = extractvalue [2 x <8 x float>] %45, 0, !dbg !7
  %1833 = extractelement <8 x float> %1832, i64 0, !dbg !7
  %1834 = extractelement <8 x float> %1832, i64 1, !dbg !7
  %1835 = extractelement <8 x float> %1832, i64 2, !dbg !7
  %1836 = extractelement <8 x float> %1832, i64 3, !dbg !7
  %1837 = select i1 %.not2, float %1834, float %1833, !dbg !7
  %1838 = bitcast float %1837 to i32, !dbg !7
  %1839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1838), !dbg !7
  %1840 = bitcast i32 %1839 to float, !dbg !7
  %1841 = select i1 %.not2, float %1833, float %1840, !dbg !7
  %1842 = select i1 %.not2, float %1840, float %1834, !dbg !7
  %1843 = select i1 %.not2, float %1836, float %1835, !dbg !7
  %1844 = bitcast float %1843 to i32, !dbg !7
  %1845 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1844), !dbg !7
  %1846 = bitcast i32 %1845 to float, !dbg !7
  %1847 = select i1 %.not2, float %1835, float %1846, !dbg !7
  %1848 = select i1 %.not2, float %1846, float %1836, !dbg !7
  %1849 = select i1 %.not3, float %1847, float %1841, !dbg !7
  %1850 = bitcast float %1849 to i32, !dbg !7
  %1851 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1850), !dbg !7
  %1852 = bitcast i32 %1851 to float, !dbg !7
  %1853 = select i1 %.not3, float %1841, float %1852, !dbg !7
  %1854 = select i1 %.not3, float %1852, float %1847, !dbg !7
  %1855 = select i1 %.not3, float %1848, float %1842, !dbg !7
  %1856 = bitcast float %1855 to i32, !dbg !7
  %1857 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1856), !dbg !7
  %1858 = bitcast i32 %1857 to float, !dbg !7
  %1859 = select i1 %.not3, float %1842, float %1858, !dbg !7
  %1860 = select i1 %.not3, float %1858, float %1848, !dbg !7
  %1861 = extractvalue [2 x <8 x float>] %42, 0, !dbg !7
  %1862 = extractelement <8 x float> %1861, i64 0, !dbg !7
  %1863 = extractelement <8 x float> %1861, i64 1, !dbg !7
  %1864 = extractelement <8 x float> %1861, i64 2, !dbg !7
  %1865 = extractelement <8 x float> %1861, i64 3, !dbg !7
  %1866 = select i1 %.not2, float %1863, float %1862, !dbg !7
  %1867 = bitcast float %1866 to i32, !dbg !7
  %1868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1867), !dbg !7
  %1869 = bitcast i32 %1868 to float, !dbg !7
  %1870 = select i1 %.not2, float %1862, float %1869, !dbg !7
  %1871 = select i1 %.not2, float %1869, float %1863, !dbg !7
  %1872 = select i1 %.not2, float %1865, float %1864, !dbg !7
  %1873 = bitcast float %1872 to i32, !dbg !7
  %1874 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1873), !dbg !7
  %1875 = bitcast i32 %1874 to float, !dbg !7
  %1876 = select i1 %.not2, float %1864, float %1875, !dbg !7
  %1877 = select i1 %.not2, float %1875, float %1865, !dbg !7
  %1878 = select i1 %.not3, float %1876, float %1870, !dbg !7
  %1879 = bitcast float %1878 to i32, !dbg !7
  %1880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1879), !dbg !7
  %1881 = select i1 %.not3, float %1877, float %1871, !dbg !7
  %1882 = bitcast float %1881 to i32, !dbg !7
  %1883 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1882), !dbg !7
  %1884 = extractvalue [2 x <1 x float>] %43, 0, !dbg !18
  %1885 = shufflevector <1 x float> %1884, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1886 = extractelement <8 x float> %1762, i64 4, !dbg !7
  %1887 = extractelement <8 x float> %1762, i64 5, !dbg !7
  %1888 = extractelement <8 x float> %1762, i64 6, !dbg !7
  %1889 = extractelement <8 x float> %1762, i64 7, !dbg !7
  %1890 = select i1 %.not2, float %1887, float %1886, !dbg !7
  %1891 = bitcast float %1890 to i32, !dbg !7
  %1892 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1891), !dbg !7
  %1893 = bitcast i32 %1892 to float, !dbg !7
  %1894 = select i1 %.not2, float %1886, float %1893, !dbg !7
  %1895 = select i1 %.not2, float %1893, float %1887, !dbg !7
  %1896 = select i1 %.not2, float %1889, float %1888, !dbg !7
  %1897 = bitcast float %1896 to i32, !dbg !7
  %1898 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1897), !dbg !7
  %1899 = bitcast i32 %1898 to float, !dbg !7
  %1900 = select i1 %.not2, float %1888, float %1899, !dbg !7
  %1901 = select i1 %.not2, float %1899, float %1889, !dbg !7
  %1902 = select i1 %.not3, float %1900, float %1894, !dbg !7
  %1903 = bitcast float %1902 to i32, !dbg !7
  %1904 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1903), !dbg !7
  %1905 = bitcast i32 %1904 to float, !dbg !7
  %1906 = select i1 %.not3, float %1894, float %1905, !dbg !7
  %1907 = select i1 %.not3, float %1905, float %1900, !dbg !7
  %1908 = select i1 %.not3, float %1901, float %1895, !dbg !7
  %1909 = bitcast float %1908 to i32, !dbg !7
  %1910 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1909), !dbg !7
  %1911 = bitcast i32 %1910 to float, !dbg !7
  %1912 = select i1 %.not3, float %1895, float %1911, !dbg !7
  %1913 = select i1 %.not3, float %1911, float %1901, !dbg !7
  %1914 = extractelement <8 x float> %1803, i64 4, !dbg !7
  %1915 = extractelement <8 x float> %1803, i64 5, !dbg !7
  %1916 = extractelement <8 x float> %1803, i64 6, !dbg !7
  %1917 = extractelement <8 x float> %1803, i64 7, !dbg !7
  %1918 = select i1 %.not2, float %1915, float %1914, !dbg !7
  %1919 = bitcast float %1918 to i32, !dbg !7
  %1920 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1919), !dbg !7
  %1921 = bitcast i32 %1920 to float, !dbg !7
  %1922 = select i1 %.not2, float %1914, float %1921, !dbg !7
  %1923 = select i1 %.not2, float %1921, float %1915, !dbg !7
  %1924 = select i1 %.not2, float %1917, float %1916, !dbg !7
  %1925 = bitcast float %1924 to i32, !dbg !7
  %1926 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1925), !dbg !7
  %1927 = bitcast i32 %1926 to float, !dbg !7
  %1928 = select i1 %.not2, float %1916, float %1927, !dbg !7
  %1929 = select i1 %.not2, float %1927, float %1917, !dbg !7
  %1930 = select i1 %.not3, float %1928, float %1922, !dbg !7
  %1931 = bitcast float %1930 to i32, !dbg !7
  %1932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1931), !dbg !7
  %1933 = bitcast i32 %1932 to float, !dbg !7
  %1934 = select i1 %.not3, float %1922, float %1933, !dbg !7
  %1935 = select i1 %.not3, float %1933, float %1928, !dbg !7
  %1936 = select i1 %.not3, float %1929, float %1923, !dbg !7
  %1937 = bitcast float %1936 to i32, !dbg !7
  %1938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1937), !dbg !7
  %1939 = bitcast i32 %1938 to float, !dbg !7
  %1940 = select i1 %.not3, float %1923, float %1939, !dbg !7
  %1941 = select i1 %.not3, float %1939, float %1929, !dbg !7
  %1942 = extractelement <8 x float> %1832, i64 4, !dbg !7
  %1943 = extractelement <8 x float> %1832, i64 5, !dbg !7
  %1944 = extractelement <8 x float> %1832, i64 6, !dbg !7
  %1945 = extractelement <8 x float> %1832, i64 7, !dbg !7
  %1946 = select i1 %.not2, float %1943, float %1942, !dbg !7
  %1947 = bitcast float %1946 to i32, !dbg !7
  %1948 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1947), !dbg !7
  %1949 = bitcast i32 %1948 to float, !dbg !7
  %1950 = select i1 %.not2, float %1942, float %1949, !dbg !7
  %1951 = select i1 %.not2, float %1949, float %1943, !dbg !7
  %1952 = select i1 %.not2, float %1945, float %1944, !dbg !7
  %1953 = bitcast float %1952 to i32, !dbg !7
  %1954 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1953), !dbg !7
  %1955 = bitcast i32 %1954 to float, !dbg !7
  %1956 = select i1 %.not2, float %1944, float %1955, !dbg !7
  %1957 = select i1 %.not2, float %1955, float %1945, !dbg !7
  %1958 = select i1 %.not3, float %1956, float %1950, !dbg !7
  %1959 = bitcast float %1958 to i32, !dbg !7
  %1960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1959), !dbg !7
  %1961 = bitcast i32 %1960 to float, !dbg !7
  %1962 = select i1 %.not3, float %1950, float %1961, !dbg !7
  %1963 = select i1 %.not3, float %1961, float %1956, !dbg !7
  %1964 = select i1 %.not3, float %1957, float %1951, !dbg !7
  %1965 = bitcast float %1964 to i32, !dbg !7
  %1966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1965), !dbg !7
  %1967 = bitcast i32 %1966 to float, !dbg !7
  %1968 = select i1 %.not3, float %1951, float %1967, !dbg !7
  %1969 = select i1 %.not3, float %1967, float %1957, !dbg !7
  %1970 = extractelement <8 x float> %1861, i64 4, !dbg !7
  %1971 = extractelement <8 x float> %1861, i64 5, !dbg !7
  %1972 = extractelement <8 x float> %1861, i64 6, !dbg !7
  %1973 = extractelement <8 x float> %1861, i64 7, !dbg !7
  %1974 = select i1 %.not2, float %1971, float %1970, !dbg !7
  %1975 = bitcast float %1974 to i32, !dbg !7
  %1976 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1975), !dbg !7
  %1977 = bitcast i32 %1976 to float, !dbg !7
  %1978 = select i1 %.not2, float %1970, float %1977, !dbg !7
  %1979 = select i1 %.not2, float %1977, float %1971, !dbg !7
  %1980 = select i1 %.not2, float %1973, float %1972, !dbg !7
  %1981 = bitcast float %1980 to i32, !dbg !7
  %1982 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1981), !dbg !7
  %1983 = bitcast i32 %1982 to float, !dbg !7
  %1984 = select i1 %.not2, float %1972, float %1983, !dbg !7
  %1985 = select i1 %.not2, float %1983, float %1973, !dbg !7
  %1986 = select i1 %.not3, float %1984, float %1978, !dbg !7
  %1987 = bitcast float %1986 to i32, !dbg !7
  %1988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1987), !dbg !7
  %1989 = select i1 %.not3, float %1985, float %1979, !dbg !7
  %1990 = bitcast float %1989 to i32, !dbg !7
  %1991 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %1990), !dbg !7
  %1992 = extractvalue [2 x <8 x float>] %47, 1, !dbg !7
  %1993 = extractelement <8 x float> %1992, i64 0, !dbg !7
  %1994 = extractelement <8 x float> %1992, i64 1, !dbg !7
  %1995 = extractelement <8 x float> %1992, i64 2, !dbg !7
  %1996 = extractelement <8 x float> %1992, i64 3, !dbg !7
  %1997 = select i1 %.not2, float %1994, float %1993, !dbg !7
  %1998 = bitcast float %1997 to i32, !dbg !7
  %1999 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1998), !dbg !7
  %2000 = bitcast i32 %1999 to float, !dbg !7
  %2001 = select i1 %.not2, float %1993, float %2000, !dbg !7
  %2002 = select i1 %.not2, float %2000, float %1994, !dbg !7
  %2003 = select i1 %.not2, float %1996, float %1995, !dbg !7
  %2004 = bitcast float %2003 to i32, !dbg !7
  %2005 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2004), !dbg !7
  %2006 = bitcast i32 %2005 to float, !dbg !7
  %2007 = select i1 %.not2, float %1995, float %2006, !dbg !7
  %2008 = select i1 %.not2, float %2006, float %1996, !dbg !7
  %2009 = select i1 %.not3, float %2007, float %2001, !dbg !7
  %2010 = bitcast float %2009 to i32, !dbg !7
  %2011 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2010), !dbg !7
  %2012 = bitcast i32 %2011 to float, !dbg !7
  %2013 = select i1 %.not3, float %2001, float %2012, !dbg !7
  %2014 = select i1 %.not3, float %2012, float %2007, !dbg !7
  %2015 = select i1 %.not3, float %2008, float %2002, !dbg !7
  %2016 = bitcast float %2015 to i32, !dbg !7
  %2017 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2016), !dbg !7
  %2018 = bitcast i32 %2017 to float, !dbg !7
  %2019 = select i1 %.not3, float %2002, float %2018, !dbg !7
  %2020 = select i1 %.not3, float %2018, float %2008, !dbg !7
  %2021 = extractvalue [2 x <8 x float>] %46, 1, !dbg !7
  %2022 = extractelement <8 x float> %2021, i64 0, !dbg !7
  %2023 = extractelement <8 x float> %2021, i64 1, !dbg !7
  %2024 = extractelement <8 x float> %2021, i64 2, !dbg !7
  %2025 = extractelement <8 x float> %2021, i64 3, !dbg !7
  %2026 = select i1 %.not2, float %2023, float %2022, !dbg !7
  %2027 = bitcast float %2026 to i32, !dbg !7
  %2028 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2027), !dbg !7
  %2029 = bitcast i32 %2028 to float, !dbg !7
  %2030 = select i1 %.not2, float %2022, float %2029, !dbg !7
  %2031 = select i1 %.not2, float %2029, float %2023, !dbg !7
  %2032 = select i1 %.not2, float %2025, float %2024, !dbg !7
  %2033 = bitcast float %2032 to i32, !dbg !7
  %2034 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2033), !dbg !7
  %2035 = bitcast i32 %2034 to float, !dbg !7
  %2036 = select i1 %.not2, float %2024, float %2035, !dbg !7
  %2037 = select i1 %.not2, float %2035, float %2025, !dbg !7
  %2038 = select i1 %.not3, float %2036, float %2030, !dbg !7
  %2039 = bitcast float %2038 to i32, !dbg !7
  %2040 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2039), !dbg !7
  %2041 = bitcast i32 %2040 to float, !dbg !7
  %2042 = select i1 %.not3, float %2030, float %2041, !dbg !7
  %2043 = select i1 %.not3, float %2041, float %2036, !dbg !7
  %2044 = select i1 %.not3, float %2037, float %2031, !dbg !7
  %2045 = bitcast float %2044 to i32, !dbg !7
  %2046 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2045), !dbg !7
  %2047 = bitcast i32 %2046 to float, !dbg !7
  %2048 = select i1 %.not3, float %2031, float %2047, !dbg !7
  %2049 = select i1 %.not3, float %2047, float %2037, !dbg !7
  %2050 = extractvalue [2 x <8 x float>] %45, 1, !dbg !7
  %2051 = extractelement <8 x float> %2050, i64 0, !dbg !7
  %2052 = extractelement <8 x float> %2050, i64 1, !dbg !7
  %2053 = extractelement <8 x float> %2050, i64 2, !dbg !7
  %2054 = extractelement <8 x float> %2050, i64 3, !dbg !7
  %2055 = select i1 %.not2, float %2052, float %2051, !dbg !7
  %2056 = bitcast float %2055 to i32, !dbg !7
  %2057 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2056), !dbg !7
  %2058 = bitcast i32 %2057 to float, !dbg !7
  %2059 = select i1 %.not2, float %2051, float %2058, !dbg !7
  %2060 = select i1 %.not2, float %2058, float %2052, !dbg !7
  %2061 = select i1 %.not2, float %2054, float %2053, !dbg !7
  %2062 = bitcast float %2061 to i32, !dbg !7
  %2063 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2062), !dbg !7
  %2064 = bitcast i32 %2063 to float, !dbg !7
  %2065 = select i1 %.not2, float %2053, float %2064, !dbg !7
  %2066 = select i1 %.not2, float %2064, float %2054, !dbg !7
  %2067 = select i1 %.not3, float %2065, float %2059, !dbg !7
  %2068 = bitcast float %2067 to i32, !dbg !7
  %2069 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2068), !dbg !7
  %2070 = bitcast i32 %2069 to float, !dbg !7
  %2071 = select i1 %.not3, float %2059, float %2070, !dbg !7
  %2072 = select i1 %.not3, float %2070, float %2065, !dbg !7
  %2073 = select i1 %.not3, float %2066, float %2060, !dbg !7
  %2074 = bitcast float %2073 to i32, !dbg !7
  %2075 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2074), !dbg !7
  %2076 = bitcast i32 %2075 to float, !dbg !7
  %2077 = select i1 %.not3, float %2060, float %2076, !dbg !7
  %2078 = select i1 %.not3, float %2076, float %2066, !dbg !7
  %2079 = extractvalue [2 x <8 x float>] %42, 1, !dbg !7
  %2080 = extractelement <8 x float> %2079, i64 0, !dbg !7
  %2081 = extractelement <8 x float> %2079, i64 1, !dbg !7
  %2082 = extractelement <8 x float> %2079, i64 2, !dbg !7
  %2083 = extractelement <8 x float> %2079, i64 3, !dbg !7
  %2084 = select i1 %.not2, float %2081, float %2080, !dbg !7
  %2085 = bitcast float %2084 to i32, !dbg !7
  %2086 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2085), !dbg !7
  %2087 = bitcast i32 %2086 to float, !dbg !7
  %2088 = select i1 %.not2, float %2080, float %2087, !dbg !7
  %2089 = select i1 %.not2, float %2087, float %2081, !dbg !7
  %2090 = select i1 %.not2, float %2083, float %2082, !dbg !7
  %2091 = bitcast float %2090 to i32, !dbg !7
  %2092 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2091), !dbg !7
  %2093 = bitcast i32 %2092 to float, !dbg !7
  %2094 = select i1 %.not2, float %2082, float %2093, !dbg !7
  %2095 = select i1 %.not2, float %2093, float %2083, !dbg !7
  %2096 = select i1 %.not3, float %2094, float %2088, !dbg !7
  %2097 = bitcast float %2096 to i32, !dbg !7
  %2098 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2097), !dbg !7
  %2099 = select i1 %.not3, float %2095, float %2089, !dbg !7
  %2100 = bitcast float %2099 to i32, !dbg !7
  %2101 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2100), !dbg !7
  %2102 = extractvalue [2 x <1 x float>] %43, 1, !dbg !18
  %2103 = shufflevector <1 x float> %2102, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %2104 = extractelement <8 x float> %1992, i64 4, !dbg !7
  %2105 = extractelement <8 x float> %1992, i64 5, !dbg !7
  %2106 = extractelement <8 x float> %1992, i64 6, !dbg !7
  %2107 = extractelement <8 x float> %1992, i64 7, !dbg !7
  %2108 = select i1 %.not2, float %2105, float %2104, !dbg !7
  %2109 = bitcast float %2108 to i32, !dbg !7
  %2110 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2109), !dbg !7
  %2111 = bitcast i32 %2110 to float, !dbg !7
  %2112 = select i1 %.not2, float %2104, float %2111, !dbg !7
  %2113 = select i1 %.not2, float %2111, float %2105, !dbg !7
  %2114 = select i1 %.not2, float %2107, float %2106, !dbg !7
  %2115 = bitcast float %2114 to i32, !dbg !7
  %2116 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2115), !dbg !7
  %2117 = bitcast i32 %2116 to float, !dbg !7
  %2118 = select i1 %.not2, float %2106, float %2117, !dbg !7
  %2119 = select i1 %.not2, float %2117, float %2107, !dbg !7
  %2120 = select i1 %.not3, float %2118, float %2112, !dbg !7
  %2121 = bitcast float %2120 to i32, !dbg !7
  %2122 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2121), !dbg !7
  %2123 = bitcast i32 %2122 to float, !dbg !7
  %2124 = select i1 %.not3, float %2112, float %2123, !dbg !7
  %2125 = select i1 %.not3, float %2123, float %2118, !dbg !7
  %2126 = select i1 %.not3, float %2119, float %2113, !dbg !7
  %2127 = bitcast float %2126 to i32, !dbg !7
  %2128 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2127), !dbg !7
  %2129 = bitcast i32 %2128 to float, !dbg !7
  %2130 = select i1 %.not3, float %2113, float %2129, !dbg !7
  %2131 = select i1 %.not3, float %2129, float %2119, !dbg !7
  %2132 = extractelement <8 x float> %2021, i64 4, !dbg !7
  %2133 = extractelement <8 x float> %2021, i64 5, !dbg !7
  %2134 = extractelement <8 x float> %2021, i64 6, !dbg !7
  %2135 = extractelement <8 x float> %2021, i64 7, !dbg !7
  %2136 = select i1 %.not2, float %2133, float %2132, !dbg !7
  %2137 = bitcast float %2136 to i32, !dbg !7
  %2138 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2137), !dbg !7
  %2139 = bitcast i32 %2138 to float, !dbg !7
  %2140 = select i1 %.not2, float %2132, float %2139, !dbg !7
  %2141 = select i1 %.not2, float %2139, float %2133, !dbg !7
  %2142 = select i1 %.not2, float %2135, float %2134, !dbg !7
  %2143 = bitcast float %2142 to i32, !dbg !7
  %2144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2143), !dbg !7
  %2145 = bitcast i32 %2144 to float, !dbg !7
  %2146 = select i1 %.not2, float %2134, float %2145, !dbg !7
  %2147 = select i1 %.not2, float %2145, float %2135, !dbg !7
  %2148 = select i1 %.not3, float %2146, float %2140, !dbg !7
  %2149 = bitcast float %2148 to i32, !dbg !7
  %2150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2149), !dbg !7
  %2151 = bitcast i32 %2150 to float, !dbg !7
  %2152 = select i1 %.not3, float %2140, float %2151, !dbg !7
  %2153 = select i1 %.not3, float %2151, float %2146, !dbg !7
  %2154 = select i1 %.not3, float %2147, float %2141, !dbg !7
  %2155 = bitcast float %2154 to i32, !dbg !7
  %2156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2155), !dbg !7
  %2157 = bitcast i32 %2156 to float, !dbg !7
  %2158 = select i1 %.not3, float %2141, float %2157, !dbg !7
  %2159 = select i1 %.not3, float %2157, float %2147, !dbg !7
  %2160 = extractelement <8 x float> %2050, i64 4, !dbg !7
  %2161 = extractelement <8 x float> %2050, i64 5, !dbg !7
  %2162 = extractelement <8 x float> %2050, i64 6, !dbg !7
  %2163 = extractelement <8 x float> %2050, i64 7, !dbg !7
  %2164 = select i1 %.not2, float %2161, float %2160, !dbg !7
  %2165 = bitcast float %2164 to i32, !dbg !7
  %2166 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2165), !dbg !7
  %2167 = bitcast i32 %2166 to float, !dbg !7
  %2168 = select i1 %.not2, float %2160, float %2167, !dbg !7
  %2169 = select i1 %.not2, float %2167, float %2161, !dbg !7
  %2170 = select i1 %.not2, float %2163, float %2162, !dbg !7
  %2171 = bitcast float %2170 to i32, !dbg !7
  %2172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2171), !dbg !7
  %2173 = bitcast i32 %2172 to float, !dbg !7
  %2174 = select i1 %.not2, float %2162, float %2173, !dbg !7
  %2175 = select i1 %.not2, float %2173, float %2163, !dbg !7
  %2176 = select i1 %.not3, float %2174, float %2168, !dbg !7
  %2177 = bitcast float %2176 to i32, !dbg !7
  %2178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2177), !dbg !7
  %2179 = bitcast i32 %2178 to float, !dbg !7
  %2180 = select i1 %.not3, float %2168, float %2179, !dbg !7
  %2181 = select i1 %.not3, float %2179, float %2174, !dbg !7
  %2182 = select i1 %.not3, float %2175, float %2169, !dbg !7
  %2183 = bitcast float %2182 to i32, !dbg !7
  %2184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2183), !dbg !7
  %2185 = bitcast i32 %2184 to float, !dbg !7
  %2186 = select i1 %.not3, float %2169, float %2185, !dbg !7
  %2187 = select i1 %.not3, float %2185, float %2175, !dbg !7
  %2188 = extractelement <8 x float> %2079, i64 4, !dbg !7
  %2189 = extractelement <8 x float> %2079, i64 5, !dbg !7
  %2190 = extractelement <8 x float> %2079, i64 6, !dbg !7
  %2191 = extractelement <8 x float> %2079, i64 7, !dbg !7
  %2192 = select i1 %.not2, float %2189, float %2188, !dbg !7
  %2193 = bitcast float %2192 to i32, !dbg !7
  %2194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2193), !dbg !7
  %2195 = bitcast i32 %2194 to float, !dbg !7
  %2196 = select i1 %.not2, float %2188, float %2195, !dbg !7
  %2197 = select i1 %.not2, float %2195, float %2189, !dbg !7
  %2198 = select i1 %.not2, float %2191, float %2190, !dbg !7
  %2199 = bitcast float %2198 to i32, !dbg !7
  %2200 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2199), !dbg !7
  %2201 = bitcast i32 %2200 to float, !dbg !7
  %2202 = select i1 %.not2, float %2190, float %2201, !dbg !7
  %2203 = select i1 %.not2, float %2201, float %2191, !dbg !7
  %2204 = select i1 %.not3, float %2202, float %2196, !dbg !7
  %2205 = bitcast float %2204 to i32, !dbg !7
  %2206 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2205), !dbg !7
  %2207 = select i1 %.not3, float %2203, float %2197, !dbg !7
  %2208 = bitcast float %2207 to i32, !dbg !7
  %2209 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1791, i32 %2208), !dbg !7
  br i1 true, label %2210, label %2274, !dbg !20

2210:                                             ; preds = %1759
  %2211 = bitcast i32 %2206 to float, !dbg !7
  %2212 = select i1 %.not3, float %2196, float %2211, !dbg !7
  %2213 = insertelement <4 x float> poison, float %2212, i64 0, !dbg !18
  %2214 = bitcast i32 %2209 to float, !dbg !7
  %2215 = select i1 %.not3, float %2197, float %2214, !dbg !7
  %2216 = insertelement <4 x float> %2213, float %2215, i64 1, !dbg !18
  %2217 = select i1 %.not3, float %2211, float %2202, !dbg !7
  %2218 = insertelement <4 x float> %2216, float %2217, i64 2, !dbg !18
  %2219 = select i1 %.not3, float %2214, float %2203, !dbg !7
  %2220 = insertelement <4 x float> %2218, float %2219, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2103, !dbg !18
  %normalized = fmul <4 x float> %2220, %row.reciprocal, !dbg !18
  %2221 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %2222 = extractelement <4 x half> %2221, i64 3, !dbg !19
  %2223 = extractelement <4 x half> %2221, i64 2, !dbg !19
  %2224 = extractelement <4 x half> %2221, i64 1, !dbg !19
  %2225 = extractelement <4 x half> %2221, i64 0, !dbg !19
  %2226 = bitcast i32 %2098 to float, !dbg !7
  %2227 = select i1 %.not3, float %2088, float %2226, !dbg !7
  %2228 = insertelement <4 x float> poison, float %2227, i64 0, !dbg !18
  %2229 = bitcast i32 %2101 to float, !dbg !7
  %2230 = select i1 %.not3, float %2089, float %2229, !dbg !7
  %2231 = insertelement <4 x float> %2228, float %2230, i64 1, !dbg !18
  %2232 = select i1 %.not3, float %2226, float %2094, !dbg !7
  %2233 = insertelement <4 x float> %2231, float %2232, i64 2, !dbg !18
  %2234 = select i1 %.not3, float %2229, float %2095, !dbg !7
  %2235 = insertelement <4 x float> %2233, float %2234, i64 3, !dbg !18
  %normalized71 = fmul <4 x float> %2235, %row.reciprocal, !dbg !18
  %2236 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !19
  %2237 = extractelement <4 x half> %2236, i64 3, !dbg !19
  %2238 = extractelement <4 x half> %2236, i64 2, !dbg !19
  %2239 = extractelement <4 x half> %2236, i64 1, !dbg !19
  %2240 = extractelement <4 x half> %2236, i64 0, !dbg !19
  %2241 = bitcast i32 %1988 to float, !dbg !7
  %2242 = select i1 %.not3, float %1978, float %2241, !dbg !7
  %2243 = insertelement <4 x float> poison, float %2242, i64 0, !dbg !18
  %2244 = bitcast i32 %1991 to float, !dbg !7
  %2245 = select i1 %.not3, float %1979, float %2244, !dbg !7
  %2246 = insertelement <4 x float> %2243, float %2245, i64 1, !dbg !18
  %2247 = select i1 %.not3, float %2241, float %1984, !dbg !7
  %2248 = insertelement <4 x float> %2246, float %2247, i64 2, !dbg !18
  %2249 = select i1 %.not3, float %2244, float %1985, !dbg !7
  %2250 = insertelement <4 x float> %2248, float %2249, i64 3, !dbg !18
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1885, !dbg !18
  %normalized73 = fmul <4 x float> %2250, %row.reciprocal72, !dbg !18
  %2251 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !19
  %2252 = extractelement <4 x half> %2251, i64 3, !dbg !19
  %2253 = extractelement <4 x half> %2251, i64 2, !dbg !19
  %2254 = extractelement <4 x half> %2251, i64 1, !dbg !19
  %2255 = extractelement <4 x half> %2251, i64 0, !dbg !19
  %2256 = bitcast i32 %1880 to float, !dbg !7
  %2257 = select i1 %.not3, float %1870, float %2256, !dbg !7
  %2258 = insertelement <4 x float> poison, float %2257, i64 0, !dbg !18
  %2259 = bitcast i32 %1883 to float, !dbg !7
  %2260 = select i1 %.not3, float %1871, float %2259, !dbg !7
  %2261 = insertelement <4 x float> %2258, float %2260, i64 1, !dbg !18
  %2262 = select i1 %.not3, float %2256, float %1876, !dbg !7
  %2263 = insertelement <4 x float> %2261, float %2262, i64 2, !dbg !18
  %2264 = select i1 %.not3, float %2259, float %1877, !dbg !7
  %2265 = insertelement <4 x float> %2263, float %2264, i64 3, !dbg !18
  %normalized74 = fmul <4 x float> %2265, %row.reciprocal72, !dbg !18
  %2266 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !19
  %2267 = extractelement <4 x half> %2266, i64 3, !dbg !19
  %2268 = extractelement <4 x half> %2266, i64 2, !dbg !19
  %2269 = extractelement <4 x half> %2266, i64 1, !dbg !19
  %2270 = extractelement <4 x half> %2266, i64 0, !dbg !19
  %2271 = add nsw i64 %8, %15, !dbg !20
  %2272 = shl nsw i64 %2271, 19, !dbg !20
  %2273 = and i64 %2272, 16252928, !dbg !20
  %address.thread389 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local390 = zext nneg i32 %address.thread389 to i64, !dbg !12
  %address.local391 = add nsw i64 %18, %address.local390, !dbg !20
  %address.local392 = lshr i64 %address.local390, 6, !dbg !12
  %address.local393 = shl nuw nsw i64 %address.local392, 4, !dbg !20
  %address.local394 = add nsw i64 %address.local391, %address.local393, !dbg !20
  %address.local395 = and i64 %address.local390, 240, !dbg !20
  %address.local396 = sub nsw i64 %address.local394, %address.local395, !dbg !20
  %address.local397 = add nsw i64 %address.local396, %19, !dbg !20
  %address.local398 = shl nuw nsw i64 %address.local397, 7, !dbg !20
  %address.local399 = add nuw nsw i64 %2273, %address.local398, !dbg !20
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !20
  %address.local401 = lshr i64 %address.local390, 2, !dbg !12
  %address.local402 = and i64 %address.local401, 12, !dbg !12
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local402, !dbg !20
  store half %2270, ptr addrspace(1) %address.local403, align 2, !dbg !20
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !20
  %address.local405 = or disjoint i64 %address.local402, 1, !dbg !20
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local404, i64 %address.local405, !dbg !20
  store half %2269, ptr addrspace(1) %address.local406, align 2, !dbg !20
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !20
  %address.local408 = or disjoint i64 %address.local402, 2, !dbg !20
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local407, i64 %address.local408, !dbg !20
  store half %2268, ptr addrspace(1) %address.local409, align 2, !dbg !20
  %address.local410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !20
  %address.local411 = or disjoint i64 %address.local402, 3, !dbg !20
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local410, i64 %address.local411, !dbg !20
  store half %2267, ptr addrspace(1) %address.local412, align 2, !dbg !20
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !20
  %address.local414 = or disjoint i64 %address.local402, 16, !dbg !20
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local413, i64 %address.local414, !dbg !20
  store half %2255, ptr addrspace(1) %address.local415, align 2, !dbg !20
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !20
  %address.local417 = or disjoint i64 %address.local402, 17, !dbg !20
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local416, i64 %address.local417, !dbg !20
  store half %2254, ptr addrspace(1) %address.local418, align 2, !dbg !20
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !20
  %address.local420 = or disjoint i64 %address.local402, 18, !dbg !20
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local419, i64 %address.local420, !dbg !20
  store half %2253, ptr addrspace(1) %address.local421, align 2, !dbg !20
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !20
  %address.local423 = or disjoint i64 %address.local402, 19, !dbg !20
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local422, i64 %address.local423, !dbg !20
  store half %2252, ptr addrspace(1) %address.local424, align 2, !dbg !20
  %address.local425 = shl nsw i64 %address.local397, 7, !dbg !20
  %address.local426 = add i64 %address.local425, 8192, !dbg !20
  %address.local427 = add nuw nsw i64 %2273, %address.local426, !dbg !20
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !20
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local402, !dbg !20
  store half %2240, ptr addrspace(1) %address.local429, align 2, !dbg !20
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !20
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local430, i64 %address.local405, !dbg !20
  store half %2239, ptr addrspace(1) %address.local431, align 2, !dbg !20
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !20
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local432, i64 %address.local408, !dbg !20
  store half %2238, ptr addrspace(1) %address.local433, align 2, !dbg !20
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !20
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local434, i64 %address.local411, !dbg !20
  store half %2237, ptr addrspace(1) %address.local435, align 2, !dbg !20
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !20
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local436, i64 %address.local414, !dbg !20
  store half %2225, ptr addrspace(1) %address.local437, align 2, !dbg !20
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !20
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local438, i64 %address.local417, !dbg !20
  store half %2224, ptr addrspace(1) %address.local439, align 2, !dbg !20
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !20
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local440, i64 %address.local420, !dbg !20
  store half %2223, ptr addrspace(1) %address.local441, align 2, !dbg !20
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !20
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local442, i64 %address.local423, !dbg !20
  store half %2222, ptr addrspace(1) %address.local443, align 2, !dbg !20
  br label %2274, !dbg !20

2274:                                             ; preds = %2210, %1759
  br i1 true, label %2275, label %2313, !dbg !20

2275:                                             ; preds = %2274
  %2276 = insertelement <4 x float> poison, float %2180, i64 0, !dbg !18
  %2277 = insertelement <4 x float> %2276, float %2186, i64 1, !dbg !18
  %2278 = insertelement <4 x float> %2277, float %2181, i64 2, !dbg !18
  %2279 = insertelement <4 x float> %2278, float %2187, i64 3, !dbg !18
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2103, !dbg !18
  %normalized76 = fmul <4 x float> %2279, %row.reciprocal75, !dbg !18
  %2280 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !19
  %2281 = extractelement <4 x half> %2280, i64 3, !dbg !19
  %2282 = extractelement <4 x half> %2280, i64 2, !dbg !19
  %2283 = extractelement <4 x half> %2280, i64 1, !dbg !19
  %2284 = extractelement <4 x half> %2280, i64 0, !dbg !19
  %2285 = insertelement <4 x float> poison, float %2071, i64 0, !dbg !18
  %2286 = insertelement <4 x float> %2285, float %2077, i64 1, !dbg !18
  %2287 = insertelement <4 x float> %2286, float %2072, i64 2, !dbg !18
  %2288 = insertelement <4 x float> %2287, float %2078, i64 3, !dbg !18
  %normalized77 = fmul <4 x float> %2288, %row.reciprocal75, !dbg !18
  %2289 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !19
  %2290 = extractelement <4 x half> %2289, i64 3, !dbg !19
  %2291 = extractelement <4 x half> %2289, i64 2, !dbg !19
  %2292 = extractelement <4 x half> %2289, i64 1, !dbg !19
  %2293 = extractelement <4 x half> %2289, i64 0, !dbg !19
  %2294 = insertelement <4 x float> poison, float %1962, i64 0, !dbg !18
  %2295 = insertelement <4 x float> %2294, float %1968, i64 1, !dbg !18
  %2296 = insertelement <4 x float> %2295, float %1963, i64 2, !dbg !18
  %2297 = insertelement <4 x float> %2296, float %1969, i64 3, !dbg !18
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1885, !dbg !18
  %normalized79 = fmul <4 x float> %2297, %row.reciprocal78, !dbg !18
  %2298 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !19
  %2299 = extractelement <4 x half> %2298, i64 3, !dbg !19
  %2300 = extractelement <4 x half> %2298, i64 2, !dbg !19
  %2301 = extractelement <4 x half> %2298, i64 1, !dbg !19
  %2302 = extractelement <4 x half> %2298, i64 0, !dbg !19
  %2303 = insertelement <4 x float> poison, float %1853, i64 0, !dbg !18
  %2304 = insertelement <4 x float> %2303, float %1859, i64 1, !dbg !18
  %2305 = insertelement <4 x float> %2304, float %1854, i64 2, !dbg !18
  %2306 = insertelement <4 x float> %2305, float %1860, i64 3, !dbg !18
  %normalized80 = fmul <4 x float> %2306, %row.reciprocal78, !dbg !18
  %2307 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !19
  %2308 = extractelement <4 x half> %2307, i64 3, !dbg !19
  %2309 = extractelement <4 x half> %2307, i64 2, !dbg !19
  %2310 = extractelement <4 x half> %2307, i64 1, !dbg !19
  %2311 = extractelement <4 x half> %2307, i64 0, !dbg !19
  %2312 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread444 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local445 = zext nneg i32 %address.thread444 to i64, !dbg !12
  %address.local446 = add nsw i64 %18, %address.local445, !dbg !20
  %address.local447 = lshr i64 %address.local445, 6, !dbg !12
  %address.local448 = shl nuw nsw i64 %address.local447, 4, !dbg !20
  %address.local449 = add nsw i64 %address.local446, %address.local448, !dbg !20
  %address.local450 = and i64 %address.local445, 240, !dbg !20
  %address.local451 = sub nsw i64 %address.local449, %address.local450, !dbg !20
  %address.local452 = add nsw i64 %address.local451, %19, !dbg !20
  %address.local453 = shl nuw nsw i64 %address.local452, 7, !dbg !20
  %address.local454 = add nuw nsw i64 %2312, %address.local453, !dbg !20
  %address.local455 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local456 = lshr i64 %address.local445, 2, !dbg !12
  %address.local457 = and i64 %address.local456, 12, !dbg !12
  %address.local458 = or disjoint i64 %address.local457, 32, !dbg !20
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local455, i64 %address.local458, !dbg !20
  store half %2311, ptr addrspace(1) %address.local459, align 2, !dbg !20
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local461 = or disjoint i64 %address.local457, 33, !dbg !20
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local461, !dbg !20
  store half %2310, ptr addrspace(1) %address.local462, align 2, !dbg !20
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local464 = or disjoint i64 %address.local457, 34, !dbg !20
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local463, i64 %address.local464, !dbg !20
  store half %2309, ptr addrspace(1) %address.local465, align 2, !dbg !20
  %address.local466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local467 = or disjoint i64 %address.local457, 35, !dbg !20
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local466, i64 %address.local467, !dbg !20
  store half %2308, ptr addrspace(1) %address.local468, align 2, !dbg !20
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local470 = or disjoint i64 %address.local457, 48, !dbg !20
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local469, i64 %address.local470, !dbg !20
  store half %2302, ptr addrspace(1) %address.local471, align 2, !dbg !20
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local473 = or disjoint i64 %address.local457, 49, !dbg !20
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local472, i64 %address.local473, !dbg !20
  store half %2301, ptr addrspace(1) %address.local474, align 2, !dbg !20
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local476 = or disjoint i64 %address.local457, 50, !dbg !20
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local475, i64 %address.local476, !dbg !20
  store half %2300, ptr addrspace(1) %address.local477, align 2, !dbg !20
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local479 = or i64 %address.local456, 51, !dbg !20
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local478, i64 %address.local479, !dbg !20
  store half %2299, ptr addrspace(1) %address.local480, align 2, !dbg !20
  %address.local481 = shl nsw i64 %address.local452, 7, !dbg !20
  %address.local482 = add i64 %address.local481, 8192, !dbg !20
  %address.local483 = add nuw nsw i64 %2312, %address.local482, !dbg !20
  %address.local484 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !20
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local484, i64 %address.local458, !dbg !20
  store half %2293, ptr addrspace(1) %address.local485, align 2, !dbg !20
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !20
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local486, i64 %address.local461, !dbg !20
  store half %2292, ptr addrspace(1) %address.local487, align 2, !dbg !20
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !20
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local488, i64 %address.local464, !dbg !20
  store half %2291, ptr addrspace(1) %address.local489, align 2, !dbg !20
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !20
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local490, i64 %address.local467, !dbg !20
  store half %2290, ptr addrspace(1) %address.local491, align 2, !dbg !20
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !20
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local492, i64 %address.local470, !dbg !20
  store half %2284, ptr addrspace(1) %address.local493, align 2, !dbg !20
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !20
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local494, i64 %address.local473, !dbg !20
  store half %2283, ptr addrspace(1) %address.local495, align 2, !dbg !20
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !20
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local496, i64 %address.local476, !dbg !20
  store half %2282, ptr addrspace(1) %address.local497, align 2, !dbg !20
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !20
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local498, i64 %address.local479, !dbg !20
  store half %2281, ptr addrspace(1) %address.local499, align 2, !dbg !20
  br label %2313, !dbg !20

2313:                                             ; preds = %2275, %2274
  br i1 true, label %2314, label %2352, !dbg !20

2314:                                             ; preds = %2313
  %2315 = insertelement <4 x float> poison, float %2152, i64 0, !dbg !18
  %2316 = insertelement <4 x float> %2315, float %2158, i64 1, !dbg !18
  %2317 = insertelement <4 x float> %2316, float %2153, i64 2, !dbg !18
  %2318 = insertelement <4 x float> %2317, float %2159, i64 3, !dbg !18
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2103, !dbg !18
  %normalized82 = fmul <4 x float> %2318, %row.reciprocal81, !dbg !18
  %2319 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !19
  %2320 = extractelement <4 x half> %2319, i64 3, !dbg !19
  %2321 = extractelement <4 x half> %2319, i64 2, !dbg !19
  %2322 = extractelement <4 x half> %2319, i64 1, !dbg !19
  %2323 = extractelement <4 x half> %2319, i64 0, !dbg !19
  %2324 = insertelement <4 x float> poison, float %2042, i64 0, !dbg !18
  %2325 = insertelement <4 x float> %2324, float %2048, i64 1, !dbg !18
  %2326 = insertelement <4 x float> %2325, float %2043, i64 2, !dbg !18
  %2327 = insertelement <4 x float> %2326, float %2049, i64 3, !dbg !18
  %normalized83 = fmul <4 x float> %2327, %row.reciprocal81, !dbg !18
  %2328 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !19
  %2329 = extractelement <4 x half> %2328, i64 3, !dbg !19
  %2330 = extractelement <4 x half> %2328, i64 2, !dbg !19
  %2331 = extractelement <4 x half> %2328, i64 1, !dbg !19
  %2332 = extractelement <4 x half> %2328, i64 0, !dbg !19
  %2333 = insertelement <4 x float> poison, float %1934, i64 0, !dbg !18
  %2334 = insertelement <4 x float> %2333, float %1940, i64 1, !dbg !18
  %2335 = insertelement <4 x float> %2334, float %1935, i64 2, !dbg !18
  %2336 = insertelement <4 x float> %2335, float %1941, i64 3, !dbg !18
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1885, !dbg !18
  %normalized85 = fmul <4 x float> %2336, %row.reciprocal84, !dbg !18
  %2337 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !19
  %2338 = extractelement <4 x half> %2337, i64 3, !dbg !19
  %2339 = extractelement <4 x half> %2337, i64 2, !dbg !19
  %2340 = extractelement <4 x half> %2337, i64 1, !dbg !19
  %2341 = extractelement <4 x half> %2337, i64 0, !dbg !19
  %2342 = insertelement <4 x float> poison, float %1824, i64 0, !dbg !18
  %2343 = insertelement <4 x float> %2342, float %1830, i64 1, !dbg !18
  %2344 = insertelement <4 x float> %2343, float %1825, i64 2, !dbg !18
  %2345 = insertelement <4 x float> %2344, float %1831, i64 3, !dbg !18
  %normalized86 = fmul <4 x float> %2345, %row.reciprocal84, !dbg !18
  %2346 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !19
  %2347 = extractelement <4 x half> %2346, i64 3, !dbg !19
  %2348 = extractelement <4 x half> %2346, i64 2, !dbg !19
  %2349 = extractelement <4 x half> %2346, i64 1, !dbg !19
  %2350 = extractelement <4 x half> %2346, i64 0, !dbg !19
  %2351 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread500 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local501 = zext nneg i32 %address.thread500 to i64, !dbg !12
  %address.local502 = add nsw i64 %18, %address.local501, !dbg !20
  %address.local503 = lshr i64 %address.local501, 6, !dbg !12
  %address.local504 = shl nuw nsw i64 %address.local503, 4, !dbg !20
  %address.local505 = add nsw i64 %address.local502, %address.local504, !dbg !20
  %address.local506 = and i64 %address.local501, 240, !dbg !20
  %address.local507 = sub nsw i64 %address.local505, %address.local506, !dbg !20
  %address.local508 = add nsw i64 %address.local507, %19, !dbg !20
  %address.local509 = shl nuw nsw i64 %address.local508, 7, !dbg !20
  %address.local510 = add nuw nsw i64 %2351, %address.local509, !dbg !20
  %address.local511 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !20
  %address.local512 = lshr i64 %address.local501, 2, !dbg !12
  %address.local513 = and i64 %address.local512, 12, !dbg !12
  %address.local514 = or disjoint i64 %address.local513, 64, !dbg !20
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local511, i64 %address.local514, !dbg !20
  store half %2350, ptr addrspace(1) %address.local515, align 2, !dbg !20
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !20
  %address.local517 = or disjoint i64 %address.local513, 65, !dbg !20
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local517, !dbg !20
  store half %2349, ptr addrspace(1) %address.local518, align 2, !dbg !20
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !20
  %address.local520 = or disjoint i64 %address.local513, 66, !dbg !20
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local519, i64 %address.local520, !dbg !20
  store half %2348, ptr addrspace(1) %address.local521, align 2, !dbg !20
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !20
  %address.local523 = or disjoint i64 %address.local513, 67, !dbg !20
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local522, i64 %address.local523, !dbg !20
  store half %2347, ptr addrspace(1) %address.local524, align 2, !dbg !20
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !20
  %address.local526 = or disjoint i64 %address.local513, 80, !dbg !20
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local525, i64 %address.local526, !dbg !20
  store half %2341, ptr addrspace(1) %address.local527, align 2, !dbg !20
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !20
  %address.local529 = or disjoint i64 %address.local513, 81, !dbg !20
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local528, i64 %address.local529, !dbg !20
  store half %2340, ptr addrspace(1) %address.local530, align 2, !dbg !20
  %address.local531 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !20
  %address.local532 = or disjoint i64 %address.local513, 82, !dbg !20
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local531, i64 %address.local532, !dbg !20
  store half %2339, ptr addrspace(1) %address.local533, align 2, !dbg !20
  %address.local534 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !20
  %address.local535 = or disjoint i64 %address.local513, 83, !dbg !20
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local534, i64 %address.local535, !dbg !20
  store half %2338, ptr addrspace(1) %address.local536, align 2, !dbg !20
  %address.local537 = shl nsw i64 %address.local508, 7, !dbg !20
  %address.local538 = add i64 %address.local537, 8192, !dbg !20
  %address.local539 = add nuw nsw i64 %2351, %address.local538, !dbg !20
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local540, i64 %address.local514, !dbg !20
  store half %2332, ptr addrspace(1) %address.local541, align 2, !dbg !20
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local542, i64 %address.local517, !dbg !20
  store half %2331, ptr addrspace(1) %address.local543, align 2, !dbg !20
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local544, i64 %address.local520, !dbg !20
  store half %2330, ptr addrspace(1) %address.local545, align 2, !dbg !20
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local546, i64 %address.local523, !dbg !20
  store half %2329, ptr addrspace(1) %address.local547, align 2, !dbg !20
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local548, i64 %address.local526, !dbg !20
  store half %2323, ptr addrspace(1) %address.local549, align 2, !dbg !20
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local529, !dbg !20
  store half %2322, ptr addrspace(1) %address.local551, align 2, !dbg !20
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local532, !dbg !20
  store half %2321, ptr addrspace(1) %address.local553, align 2, !dbg !20
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local554, i64 %address.local535, !dbg !20
  store half %2320, ptr addrspace(1) %address.local555, align 2, !dbg !20
  br label %2352, !dbg !20

2352:                                             ; preds = %2314, %2313
  br i1 true, label %2353, label %2391, !dbg !20

2353:                                             ; preds = %2352
  %2354 = insertelement <4 x float> poison, float %2124, i64 0, !dbg !18
  %2355 = insertelement <4 x float> %2354, float %2130, i64 1, !dbg !18
  %2356 = insertelement <4 x float> %2355, float %2125, i64 2, !dbg !18
  %2357 = insertelement <4 x float> %2356, float %2131, i64 3, !dbg !18
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2103, !dbg !18
  %normalized88 = fmul <4 x float> %2357, %row.reciprocal87, !dbg !18
  %2358 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !19
  %2359 = extractelement <4 x half> %2358, i64 3, !dbg !19
  %2360 = extractelement <4 x half> %2358, i64 2, !dbg !19
  %2361 = extractelement <4 x half> %2358, i64 1, !dbg !19
  %2362 = extractelement <4 x half> %2358, i64 0, !dbg !19
  %2363 = insertelement <4 x float> poison, float %2013, i64 0, !dbg !18
  %2364 = insertelement <4 x float> %2363, float %2019, i64 1, !dbg !18
  %2365 = insertelement <4 x float> %2364, float %2014, i64 2, !dbg !18
  %2366 = insertelement <4 x float> %2365, float %2020, i64 3, !dbg !18
  %normalized89 = fmul <4 x float> %2366, %row.reciprocal87, !dbg !18
  %2367 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !19
  %2368 = extractelement <4 x half> %2367, i64 3, !dbg !19
  %2369 = extractelement <4 x half> %2367, i64 2, !dbg !19
  %2370 = extractelement <4 x half> %2367, i64 1, !dbg !19
  %2371 = extractelement <4 x half> %2367, i64 0, !dbg !19
  %2372 = insertelement <4 x float> poison, float %1906, i64 0, !dbg !18
  %2373 = insertelement <4 x float> %2372, float %1912, i64 1, !dbg !18
  %2374 = insertelement <4 x float> %2373, float %1907, i64 2, !dbg !18
  %2375 = insertelement <4 x float> %2374, float %1913, i64 3, !dbg !18
  %row.reciprocal90 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1885, !dbg !18
  %normalized91 = fmul <4 x float> %2375, %row.reciprocal90, !dbg !18
  %2376 = fptrunc <4 x float> %normalized91 to <4 x half>, !dbg !19
  %2377 = extractelement <4 x half> %2376, i64 3, !dbg !19
  %2378 = extractelement <4 x half> %2376, i64 2, !dbg !19
  %2379 = extractelement <4 x half> %2376, i64 1, !dbg !19
  %2380 = extractelement <4 x half> %2376, i64 0, !dbg !19
  %2381 = insertelement <4 x float> poison, float %1795, i64 0, !dbg !18
  %2382 = insertelement <4 x float> %2381, float %1801, i64 1, !dbg !18
  %2383 = insertelement <4 x float> %2382, float %1796, i64 2, !dbg !18
  %2384 = insertelement <4 x float> %2383, float %1802, i64 3, !dbg !18
  %normalized92 = fmul <4 x float> %2384, %row.reciprocal90, !dbg !18
  %2385 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !19
  %2386 = extractelement <4 x half> %2385, i64 3, !dbg !19
  %2387 = extractelement <4 x half> %2385, i64 2, !dbg !19
  %2388 = extractelement <4 x half> %2385, i64 1, !dbg !19
  %2389 = extractelement <4 x half> %2385, i64 0, !dbg !19
  %2390 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread556 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local557 = zext nneg i32 %address.thread556 to i64, !dbg !12
  %address.local558 = add nsw i64 %18, %address.local557, !dbg !20
  %address.local559 = lshr i64 %address.local557, 6, !dbg !12
  %address.local560 = shl nuw nsw i64 %address.local559, 4, !dbg !20
  %address.local561 = add nsw i64 %address.local558, %address.local560, !dbg !20
  %address.local562 = and i64 %address.local557, 240, !dbg !20
  %address.local563 = sub nsw i64 %address.local561, %address.local562, !dbg !20
  %address.local564 = add nsw i64 %address.local563, %19, !dbg !20
  %address.local565 = shl nuw nsw i64 %address.local564, 7, !dbg !20
  %address.local566 = add nuw nsw i64 %2390, %address.local565, !dbg !20
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !20
  %address.local568 = lshr i64 %address.local557, 2, !dbg !12
  %address.local569 = and i64 %address.local568, 12, !dbg !12
  %address.local570 = or disjoint i64 %address.local569, 96, !dbg !20
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local570, !dbg !20
  store half %2389, ptr addrspace(1) %address.local571, align 2, !dbg !20
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !20
  %address.local573 = or disjoint i64 %address.local569, 97, !dbg !20
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local573, !dbg !20
  store half %2388, ptr addrspace(1) %address.local574, align 2, !dbg !20
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !20
  %address.local576 = or disjoint i64 %address.local569, 98, !dbg !20
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local576, !dbg !20
  store half %2387, ptr addrspace(1) %address.local577, align 2, !dbg !20
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !20
  %address.local579 = or disjoint i64 %address.local569, 99, !dbg !20
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local578, i64 %address.local579, !dbg !20
  store half %2386, ptr addrspace(1) %address.local580, align 2, !dbg !20
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !20
  %address.local582 = or disjoint i64 %address.local569, 112, !dbg !20
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local582, !dbg !20
  store half %2380, ptr addrspace(1) %address.local583, align 2, !dbg !20
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !20
  %address.local585 = or disjoint i64 %address.local569, 113, !dbg !20
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local584, i64 %address.local585, !dbg !20
  store half %2379, ptr addrspace(1) %address.local586, align 2, !dbg !20
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !20
  %address.local588 = or disjoint i64 %address.local569, 114, !dbg !20
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local588, !dbg !20
  store half %2378, ptr addrspace(1) %address.local589, align 2, !dbg !20
  %address.local590 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !20
  %address.local591 = or i64 %address.local568, 115, !dbg !20
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local590, i64 %address.local591, !dbg !20
  store half %2377, ptr addrspace(1) %address.local592, align 2, !dbg !20
  %address.local593 = shl nsw i64 %address.local564, 7, !dbg !20
  %address.local594 = add i64 %address.local593, 8192, !dbg !20
  %address.local595 = add nuw nsw i64 %2390, %address.local594, !dbg !20
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local596, i64 %address.local570, !dbg !20
  store half %2371, ptr addrspace(1) %address.local597, align 2, !dbg !20
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local598, i64 %address.local573, !dbg !20
  store half %2370, ptr addrspace(1) %address.local599, align 2, !dbg !20
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local576, !dbg !20
  store half %2369, ptr addrspace(1) %address.local601, align 2, !dbg !20
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local602, i64 %address.local579, !dbg !20
  store half %2368, ptr addrspace(1) %address.local603, align 2, !dbg !20
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local604, i64 %address.local582, !dbg !20
  store half %2362, ptr addrspace(1) %address.local605, align 2, !dbg !20
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local606, i64 %address.local585, !dbg !20
  store half %2361, ptr addrspace(1) %address.local607, align 2, !dbg !20
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local588, !dbg !20
  store half %2360, ptr addrspace(1) %address.local609, align 2, !dbg !20
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local610, i64 %address.local591, !dbg !20
  store half %2359, ptr addrspace(1) %address.local611, align 2, !dbg !20
  br label %2391, !dbg !20

2391:                                             ; preds = %2353, %2352
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/baseline")
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
